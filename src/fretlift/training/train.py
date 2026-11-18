"""``fretlift-train``: train the transcription model.

Leave-one-player-out on GuitarSet (train on players 00-03, validate on 04,
test on 05), clean audio only::

    fretlift-train --manifest data/guitarset/manifest.jsonl \\
        --val-groups guitarset-player-04 --test-groups guitarset-player-05 --out runs/gs-clean

The same split with amp/distortion augmentation, also validating on distorted audio::

    fretlift-train --manifest data/guitarset/manifest.jsonl \\
        --val-groups guitarset-player-04 --test-groups guitarset-player-05 \\
        --augment full --val-distort high_gain --out runs/gs-aug

To train on more data starting from an earlier model (its run directory is
left alone), add ``--init runs/gs-aug/best.pt`` and a new ``--out``.

Each run directory gets ``best.pt`` (best validation score), ``last.pt``
(for --resume), ``history.csv``, ``thresholds.json`` (tuned on validation)
and ``test_report.json`` (the held-out test set, evaluated once at the end).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..transcription import model_path
from .audio import AudioCache
from .augment import AMP_PRESETS, AugmentConfig, Augmenter
from .config import DecodeConfig, ModelConfig
from .data import ClipDataset
from .evaluate import DEFAULT_CACHE, EvalAudio, evaluate_items, format_summary, summarize
from .inference import ModelPredictor, pick_device
from .manifest import assign_splits, filter_tags, read_manifests
from .model import TabCRNN, compute_loss, count_parameters
from .sweep import format_sweep, sweep_thresholds


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fretlift-train", description="Train the transcription model.")
    d = p.add_argument_group("data")
    d.add_argument("--manifest", action="append", required=True, help="manifest .jsonl (repeatable)")
    d.add_argument("--val-groups", default="", help="comma-separated groups for validation")
    d.add_argument("--test-groups", default="", help="comma-separated groups for the final test")
    d.add_argument("--tags", default="", help="only use items with any of these tags")
    d.add_argument("--exclude-tags", default="")
    d.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    d.add_argument("--clip-seconds", type=float, default=5.0)
    d.add_argument("--workers", type=int, default=None, help="data loader processes (default: min(4, CPUs))")
    a = p.add_argument_group("augmentation")
    a.add_argument("--augment", choices=["none", "full"], default="none")
    a.add_argument("--ir-dir", help="folder of cabinet impulse responses (WAV)")
    a.add_argument("--p-amp", type=float, default=0.7, help="share of clips sent through an amp")
    a.add_argument("--val-distort", choices=["none", *AMP_PRESETS], default="none",
                   help="also validate on distorted audio, and pick the best checkpoint on the average")
    o = p.add_argument_group("optimization")
    o.add_argument("--epochs", type=int, default=30)
    o.add_argument("--steps-per-epoch", type=int, default=500)
    o.add_argument("--batch-size", type=int, default=16)
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--weight-decay", type=float, default=1e-4)
    o.add_argument("--warmup-steps", type=int, default=300)
    o.add_argument("--onset-pos-weight", type=float, default=4.0)
    o.add_argument("--string-weight", type=float, default=0.5)
    o.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    m = p.add_argument_group("model")
    m.add_argument("--channels", type=int, default=32)
    m.add_argument("--hidden", type=int, default=256)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--resume", help="continue from a last.pt")
    p.add_argument("--init", help="start from this checkpoint's weights (and model size), with a fresh schedule")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-val-items", type=int, help="validate on a subset (faster epochs)")
    p.add_argument("--no-final-test", action="store_true")
    return p


def _validate(model, items, eval_audio, device, distorts, decode):
    model.eval()
    predictor = ModelPredictor(model, eval_audio, device=device, cache=False)
    scores = {}
    for dist in distorts:
        s = summarize(evaluate_items(items, predictor, decode=decode, distort=dist))["all"]
        key = "clean" if dist == "none" else dist
        scores[f"{key}_note_f1"] = s["note_f1"]
        scores[f"{key}_tab_f1"] = s["tab_f1"]
    note = [v for k, v in scores.items() if k.endswith("note_f1") and v is not None]
    scores["select"] = float(np.mean(note)) if note else 0.0
    return scores


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    items = filter_tags(read_manifests(args.manifest), _csv(args.tags), _csv(args.exclude_tags))
    splits = assign_splits(items, _csv(args.val_groups), _csv(args.test_groups))
    train_items, val_items, test_items = splits["train"], splits["val"], splits["test"]
    if args.max_val_items:
        val_items = val_items[: args.max_val_items]
    print(f"device={device}  train={len(train_items)}  val={len(val_items)}  test={len(test_items)} items")
    if not train_items or not val_items:
        print("error: need training and validation items; check --val-groups/--test-groups", file=sys.stderr)
        return 2

    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    model_cfg = ModelConfig(channels=args.channels, hidden=args.hidden)
    if init:
        model_cfg = ModelConfig.from_dict(init["model_config"])
    cache =AudioCache(Path(args.cache_dir) / "audio", model_cfg.sr)
    augmenter = None
    if args.augment == "full":
        augmenter = Augmenter(AugmentConfig(ir_dir=args.ir_dir, p_amp=args.p_amp), model_cfg.sr)
    print("Preparing training audio")
    train_ds = ClipDataset(
        train_items, cache, model_cfg, clip_seconds=args.clip_seconds,
        length=args.steps_per_epoch * args.batch_size, augmenter=augmenter, seed=args.seed,
    )
    workers = args.workers if args.workers is not None else min(4, os.cpu_count() or 1)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=workers, drop_last=True,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    eval_audio = EvalAudio(args.cache_dir, model_cfg.sr, ir_dir=args.ir_dir)
    distorts = ["none"] + ([args.val_distort] if args.val_distort != "none" else [])

    model = TabCRNN(model_cfg).to(device)
    if init:
        model.load_state_dict(init["model"])  # --resume, if also given, overrides this below
        print(f"starting from the weights in {args.init}")
    print(f"model: {count_parameters(model) / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch

    def lr_at(step):  # linear warmup, then cosine decay to 5%
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best = 1, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"resumed from epoch {ck['epoch']}")

    def save(path, epoch, scores, extra=None):
        torch.save({
            "model": model.state_dict(), "model_config": model_cfg.to_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "epoch": epoch, "scores": scores, "best": best,
            "args": vars(args), **(extra or {}),
        }, path)

    hist_path = out / "history.csv"
    hist_fields = ["epoch", "time_s", "lr", "loss", "frame", "onset", "string", "select", *(
        f"{('clean' if d == 'none' else d)}_{m}" for d in distorts for m in ("note_f1", "tab_f1"))]
    if not (args.resume and hist_path.exists()):
        with open(hist_path, "w", newline="") as f:
            csv.DictWriter(f, hist_fields).writeheader()

    def autocast():
        return torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()

    decode = DecodeConfig()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums: dict[str, float] = {}
        for i, batch in enumerate(loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast():
                out_ = model(batch["audio"])
            loss, parts = compute_loss(out_, batch, args.onset_pos_weight, args.string_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            if i % 50 == 0:
                print(f"  epoch {epoch} step {i}/{args.steps_per_epoch} loss {sums['total'] / i:.4f}", flush=True)
        n = max(1, args.steps_per_epoch)
        scores = _validate(model, val_items, eval_audio, device, distorts, decode)
        row = {"epoch": epoch, "time_s": round(time.time() - t0, 1), "lr": sched.get_last_lr()[0],
               "loss": sums.get("total", 0) / n, "frame": sums.get("frame", 0) / n, "onset": sums.get("onset", 0) / n,
               "string": sums.get("string", 0) / n, **scores}
        with open(hist_path, "a", newline="") as f:
            csv.DictWriter(f, hist_fields, extrasaction="ignore").writerow(row)
        shown = "  ".join(f"{k}={v:.3f}" for k, v in scores.items() if v is not None)
        improved = scores["select"] > best
        if improved:
            best = scores["select"]
        mark = "  * best" if improved else ""
        print(f"epoch {epoch}: loss {row['loss']:.4f}  {shown}  ({row['time_s']:.0f}s){mark}", flush=True)
        save(out / "last.pt", epoch, scores)
        if improved:
            save(out / "best.pt", epoch, scores)

    # Tune thresholds on validation (activations are computed once, then re-decoded).
    print("Tuning decoding thresholds on the validation set")
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    predictor = ModelPredictor(model, eval_audio, device=device)
    thresholds = {}
    for dist in distorts:
        result = sweep_thresholds(val_items, predictor, distort=dist, progress=False)
        thresholds[dist] = result
        print(f"[val, distort={dist}]\n{format_sweep(result)}\n")
    ck["thresholds"] = {"all": thresholds["none"]["all"], "by_distort": {d: r["all"] for d, r in thresholds.items()},
                        "per_tag": thresholds["none"]["per_tag"]}
    torch.save(ck, out / "best.pt")
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=1))

    if test_items and not args.no_final_test:
        t = ck["thresholds"]["all"]
        decode = DecodeConfig(onset_threshold=t["onset_threshold"], frame_threshold=t["frame_threshold"])
        report = {"decode": asdict(decode)}
        for dist in distorts:
            summary = summarize(evaluate_items(test_items, predictor, decode=decode, distort=dist))
            report[dist] = summary
            title = f"TEST | distort={dist} | onset={decode.onset_threshold} frame={decode.frame_threshold}"
            print(format_summary(summary, title))
        (out / "test_report.json").write_text(json.dumps(report, indent=1))
    print(f"done: {out / 'best.pt'}")
    print(f"To make tabs with it, copy it to {model_path()} (or point FRETLIFT_MODEL at it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
