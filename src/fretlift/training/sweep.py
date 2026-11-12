"""Find the best onset/frame thresholds, per tag and overall, on validation data.

    fretlift-sweep --manifest data/guitarset/manifest.jsonl --manifest data/mine/manifest.jsonl \\
        --split val --val-groups guitarset-player-04 --basic-pitch --out thresholds.json

The output answers "is one setting good enough everywhere?": for every tag it
shows the best thresholds, the F1 they reach, and how much F1 you give up by
using the single best overall setting instead.

Tune on validation data, never on the test set, or the test score stops
meaning anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

from .config import DecodeConfig
from .evaluate import add_predictor_args, add_selection_args, evaluate_items, make_predictor, select_items
from .manifest import Item

DEFAULT_ONSETS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_FRAMES = (0.1, 0.2, 0.3, 0.4, 0.5)


def sweep_thresholds(
    items: Sequence[Item],
    predictor,
    *,
    distort: str | None = None,
    onsets: Sequence[float] = DEFAULT_ONSETS,
    frames: Sequence[float] = DEFAULT_FRAMES,
    base: DecodeConfig | None = None,
    progress: bool = True,
) -> dict:
    """Grid search on note F1. Returns {"all": best, "per_tag": {...}, "grid": [...]}."""
    base = base or DecodeConfig()
    grid = []
    combos = list(product(onsets, frames))
    for k, (on, fr) in enumerate(combos, start=1):
        decode = replace(base, onset_threshold=on, frame_threshold=fr)
        results = evaluate_items(items, predictor, decode=decode, distort=distort, strings=False)
        by_tag: dict[str, list[float]] = defaultdict(list)
        for r in results:
            by_tag["all"].append(r.note.f1)
            for t in r.tags:
                by_tag[t].append(r.note.f1)
        grid.append({"onset_threshold": on, "frame_threshold": fr, "f1": {t: float(np.mean(v)) for t, v in by_tag.items()}})
        if progress:
            f1 = grid[-1]["f1"]["all"]
            print(f"  [{k}/{len(combos)}] onset={on:.2f} frame={fr:.2f}  F1={f1:.3f}", file=sys.stderr, flush=True)

    def best_for(tag):
        g = max(grid, key=lambda g: g["f1"].get(tag, -1))
        return {"onset_threshold": g["onset_threshold"], "frame_threshold": g["frame_threshold"], "f1": g["f1"][tag]}

    overall = best_for("all")
    overall_cell = next(
        g
        for g in grid
        if (g["onset_threshold"], g["frame_threshold"]) == (overall["onset_threshold"], overall["frame_threshold"])
    )
    tags = sorted(t for t in grid[0]["f1"] if t != "all")
    per_tag = {}
    for t in tags:
        b = best_for(t)
        b["f1_with_overall_setting"] = overall_cell["f1"][t]
        b["f1_lost_with_overall_setting"] = b["f1"] - overall_cell["f1"][t]
        per_tag[t] = b
    return {"all": overall, "per_tag": per_tag, "grid": grid}


def format_sweep(result: dict) -> str:
    a = result["all"]
    lines = [
        f"Best single setting: --onset-threshold {a['onset_threshold']} --frame-threshold {a['frame_threshold']}"
        f"  (F1 {a['f1']:.3f})",
        "",
        f"{'tag':<20} {'best onset':>10} {'best frame':>10} {'best F1':>8} {'F1 @ single':>11} {'lost':>6}",
    ]
    for t, b in result["per_tag"].items():
        lines.append(
            f"{t:<20} {b['onset_threshold']:>10.2f} {b['frame_threshold']:>10.2f} {b['f1']:>8.3f} "
            f"{b['f1_with_overall_setting']:>11.3f} {b['f1_lost_with_overall_setting']:>6.3f}"
        )
    worst = max(result["per_tag"].items(), key=lambda kv: kv[1]["f1_lost_with_overall_setting"], default=None)
    if worst and worst[1]["f1_lost_with_overall_setting"] > 0.02:
        lines += ["", f"'{worst[0]}' loses {worst[1]['f1_lost_with_overall_setting']:.3f} F1 with the single setting: "
                  "worth a separate setting for that kind of recording."]
    elif worst:
        lines += ["", "No tag loses more than 0.02 F1 with the single setting, so one setting is fine."]
    return "\n".join(lines)


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fretlift-sweep", description=__doc__.splitlines()[0])
    add_selection_args(p, default_split="val")
    add_predictor_args(p)
    p.add_argument("--onsets", type=_floats, default=list(DEFAULT_ONSETS), help="comma-separated onset thresholds")
    p.add_argument("--frames", type=_floats, default=list(DEFAULT_FRAMES), help="comma-separated frame thresholds")
    p.add_argument("--out", help="write the full grid and best settings as JSON")
    args = p.parse_args(argv)
    if args.split == "test":
        print("warning: tuning on the test split makes its scores meaningless; use --split val", file=sys.stderr)

    items = select_items(args)
    predictor, _ = make_predictor(args)
    n_settings = len(args.onsets) * len(args.frames)
    print(f"Sweeping {n_settings} settings over {len(items)} items with {predictor.name}", file=sys.stderr)
    result = sweep_thresholds(items, predictor, distort=args.distort, onsets=args.onsets, frames=args.frames)
    print(format_sweep(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
