"""Evaluate a transcriber (Basic Pitch or a trained checkpoint) on a manifest.

    fretlift-eval --manifest data/guitarset/manifest.jsonl --test-groups guitarset-player-05 \\
        --split test --basic-pitch
    fretlift-eval --manifest data/guitarset/manifest.jsonl --test-groups guitarset-player-05 \\
        --split test --model runs/fold05/best.pt --distort high_gain

Prints mean per-track scores for every tag (acoustic, electric, separated, ...)
so you can see where a model is weak, and writes per-track results as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..metrics import Score, note_score, string_accuracy, tab_score
from ..tuning import PRESETS
from ..types import Note
from .audio import AudioCache, write_wav
from .augment import AMP_PRESETS, apply_preset, load_irs
from .config import SR, DecodeConfig
from .labels import load_labels
from .manifest import Item, assign_splits, filter_tags, read_manifests

DEFAULT_CACHE = Path(".cache/fretlift")


class Predictor(Protocol):
    name: str

    def predict(
        self, item: Item, distort: str | None, decode: DecodeConfig, tuning: Sequence[int], strings: bool = True
    ) -> list[Note]: ...


# ------------------------------------------------------------- shared audio


def _seed(item: Item) -> int:
    return int(hashlib.md5(item.id.encode()).hexdigest()[:8], 16)


class EvalAudio:
    """Clean or deterministically distorted evaluation audio, as arrays or files."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, sr: int = SR, ir_dir: str | None = None):
        self.cache = AudioCache(Path(cache_dir) / "audio", sr)
        self.dir = Path(cache_dir) / "distorted"
        self.sr = sr
        self.irs = load_irs(ir_dir, sr)

    def array(self, item: Item, distort: str | None) -> np.ndarray:
        x = np.asarray(self.cache.load(item.audio), np.float32)
        return apply_preset(x, self.sr, distort, _seed(item), self.irs) if distort not in (None, "none") else x

    def path(self, item: Item, distort: str | None) -> Path:
        if distort in (None, "none"):
            return item.audio
        key = hashlib.md5(f"{item.audio}:{distort}:{len(self.irs)}".encode()).hexdigest()
        p = self.dir / f"{key}.wav"
        if not p.exists():
            write_wav(p, self.array(item, distort), self.sr)
        return p


def playable_range(tuning: Sequence[int], max_fret: int = 24) -> tuple[int, int]:
    return min(tuning), max(tuning) + max_fret


# ------------------------------------------------------------------ scoring


@dataclass
class TrackResult:
    id: str
    tags: tuple[str, ...]
    note: Score
    note_offset: Score
    tab: Score | None
    string_correct: int
    string_total: int


def evaluate_items(
    items: Sequence[Item],
    predictor: Predictor,
    *,
    decode: DecodeConfig | None = None,
    distort: str | None = None,
    strings: bool = True,
    max_fret: int = 22,
    progress: bool = False,
) -> list[TrackResult]:
    decode = decode or DecodeConfig()
    results = []
    for k, item in enumerate(items, start=1):
        ref, label_tuning = load_labels(item.labels)
        tuning = item.tuning or label_tuning or PRESETS["standard"]
        lo, hi = playable_range(tuning, max_fret)
        est = [n for n in predictor.predict(item, distort, decode, tuning, strings=strings) if lo <= n.pitch <= hi]
        correct, total = string_accuracy(ref, est) if strings else (0, 0)
        results.append(
            TrackResult(
                id=item.id,
                tags=item.tags,
                note=note_score(ref, est),
                note_offset=note_score(ref, est, offset_ratio=0.2),
                tab=tab_score(ref, est) if strings else None,
                string_correct=correct,
                string_total=total,
            )
        )
        if progress and (k % 10 == 0 or k == len(items)):
            print(f"  {k}/{len(items)} tracks", file=sys.stderr, flush=True)
    return results


def summarize(results: Sequence[TrackResult]) -> dict[str, dict]:
    """Mean per-track scores for all tracks and for each tag."""
    groups: dict[str, list[TrackResult]] = defaultdict(list)
    for r in results:
        groups["all"].append(r)
        for t in r.tags:
            groups[t].append(r)

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    out = {}
    for tag, rs in sorted(groups.items(), key=lambda kv: (kv[0] != "all", kv[0])):
        tabs = [r.tab.f1 for r in rs if r.tab is not None]
        sc, st = sum(r.string_correct for r in rs), sum(r.string_total for r in rs)
        out[tag] = {
            "tracks": len(rs),
            "note_p": mean([r.note.precision for r in rs]),
            "note_r": mean([r.note.recall for r in rs]),
            "note_f1": mean([r.note.f1 for r in rs]),
            "note_offset_f1": mean([r.note_offset.f1 for r in rs]),
            "tab_f1": mean(tabs),
            "string_acc": sc / st if st else None,
        }
    return out


COLUMNS = [
    ("tracks", "tracks", "{:d}"),
    ("note P", "note_p", "{:.3f}"),
    ("note R", "note_r", "{:.3f}"),
    ("note F1", "note_f1", "{:.3f}"),
    ("+offset F1", "note_offset_f1", "{:.3f}"),
    ("tab F1", "tab_f1", "{:.3f}"),
    ("string acc", "string_acc", "{:.3f}"),
]


def format_summary(summary: dict[str, dict], title: str = "") -> str:
    header = ["tag"] + [h for h, _, _ in COLUMNS]
    rows = [[tag] + ["-" if s[k] is None else f.format(s[k]) for _, k, f in COLUMNS] for tag, s in summary.items()]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    lines = [title] if title else []
    for r in [header, *rows]:
        lines.append("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))))
    return "\n".join(lines)


# ---------------------------------------------------------------------- CLI


def add_selection_args(p: argparse.ArgumentParser, default_split: str = "test") -> None:
    p.add_argument("--manifest", action="append", required=True, help="manifest .jsonl (repeatable)")
    p.add_argument("--split", default=default_split, choices=["train", "val", "test", "all"])
    p.add_argument("--val-groups", default="", help="comma-separated groups for validation")
    p.add_argument("--test-groups", default="", help="comma-separated groups for testing")
    p.add_argument("--tags", default="", help="only items with any of these tags (comma-separated)")
    p.add_argument("--exclude-tags", default="", help="drop items with any of these tags")
    p.add_argument("--limit", type=int, help="evaluate at most this many items (quick checks)")


def select_items(args) -> list[Item]:
    def csv(s):
        return [x for x in s.split(",") if x]

    items = filter_tags(read_manifests(args.manifest), csv(args.tags), csv(args.exclude_tags))
    if args.split != "all":
        items = assign_splits(items, csv(args.val_groups), csv(args.test_groups))[args.split]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No items selected (split={args.split}). Check --split, --*-groups and --tags.")
    return items


def add_predictor_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="trained checkpoint (.pt)")
    g.add_argument("--basic-pitch", action="store_true", help="evaluate the Basic Pitch baseline")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--distort", default="none", choices=["none", *AMP_PRESETS], help="fixed amp preset applied to the test audio"
    )
    p.add_argument("--ir-dir", help="cabinet impulse responses for --distort")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))


def make_predictor(args):
    eval_audio = EvalAudio(args.cache_dir, ir_dir=getattr(args, "ir_dir", None))
    if args.basic_pitch:
        from .baseline import BasicPitchPredictor

        return BasicPitchPredictor(eval_audio, Path(args.cache_dir) / "basic_pitch"), None
    from .inference import ModelPredictor, load_model, pick_device

    model, ckpt = load_model(args.model, pick_device(args.device))
    return ModelPredictor(model, eval_audio, device=pick_device(args.device)), ckpt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fretlift-eval", description=__doc__.splitlines()[0])
    add_selection_args(p)
    add_predictor_args(p)
    p.add_argument("--onset-threshold", type=float, help="default: the checkpoint's tuned value, else 0.5")
    p.add_argument("--frame-threshold", type=float, help="default: the checkpoint's tuned value, else 0.3")
    p.add_argument("--no-strings", action="store_true", help="skip string assignment (faster; no tab F1)")
    p.add_argument("--out", help="write per-track results and the summary as JSON")
    args = p.parse_args(argv)

    items = select_items(args)
    predictor, ckpt = make_predictor(args)
    decode = DecodeConfig()
    tuned = (ckpt or {}).get("thresholds", {}).get("all")
    if tuned:
        decode.onset_threshold, decode.frame_threshold = tuned["onset_threshold"], tuned["frame_threshold"]
    if args.onset_threshold is not None:
        decode.onset_threshold = args.onset_threshold
    if args.frame_threshold is not None:
        decode.frame_threshold = args.frame_threshold

    print(f"Evaluating {predictor.name} on {len(items)} items (distort={args.distort}, "
          f"onset={decode.onset_threshold}, frame={decode.frame_threshold})", file=sys.stderr)
    results = evaluate_items(
        items, predictor, decode=decode, distort=args.distort, strings=not args.no_strings, progress=True
    )
    summary = summarize(results)
    print(format_summary(summary, f"{predictor.name} | split={args.split} | distort={args.distort}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "predictor": predictor.name, "distort": args.distort, "decode": asdict(decode),
            "summary": summary, "tracks": [asdict(r) for r in results],
        }, indent=1))
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
