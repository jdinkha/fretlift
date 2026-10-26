"""Score transcribed notes against an answer key with note-level F1.

Uses mir_eval's standard transcription metric: a note counts as correct if its
pitch is within 50 cents and its onset within 50 ms of a reference note
(offsets ignored). This is the metric most transcription papers report, so the
numbers are comparable when you start training your own model.

Usage:
  fretlift samples/ode_to_joy.wav --no-separate --save-notes -o out
  python scripts/evaluate.py samples/ode_to_joy.notes.json out/ode_to_joy.notes.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    notes = json.load(open(path))["notes"]
    intervals = np.array([[n["start"], n["end"]] for n in notes], dtype=float).reshape(-1, 2)
    hz = np.array([440.0 * 2 ** ((n["pitch"] - 69) / 12) for n in notes], dtype=float)
    return intervals, hz


def main() -> None:
    ap = argparse.ArgumentParser(description="Note-level precision/recall/F1")
    ap.add_argument("reference", help="answer key JSON (from make_sample.py)")
    ap.add_argument("estimate", help="notes JSON written by `fretlift --save-notes`")
    ap.add_argument("--onset-tolerance", type=float, default=0.05)
    args = ap.parse_args()

    try:
        import mir_eval
    except ImportError:
        raise SystemExit("Install the eval extra: pip install -e .[eval]") from None

    ref_i, ref_p = load(args.reference)
    est_i, est_p = load(args.estimate)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=args.onset_tolerance, offset_ratio=None
    )
    print(f"reference notes: {len(ref_p)}  estimated notes: {len(est_p)}")
    print(f"precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")


if __name__ == "__main__":
    main()
