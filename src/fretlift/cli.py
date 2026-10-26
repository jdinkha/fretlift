"""Command-line interface: ``fretlift song.wav -o out/``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .export.musicxml import SUPPORTED_SUBDIVISIONS
from .tuning import PRESETS, parse_tuning


def _time_signature(s: str) -> tuple[int, int]:
    try:
        beats, beat_type = (int(x) for x in s.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("use the form 4/4") from None
    if beat_type not in (2, 4, 8) or beats < 1:
        raise argparse.ArgumentTypeError("beat type must be 2, 4 or 8")
    return beats, beat_type


def _tuning(s: str) -> tuple[int, ...]:
    try:
        return parse_tuning(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fretlift",
        description="Generate a guitar tab from a local audio file. "
        "Use it on recordings you have the rights to transcribe; see the README.",
    )
    p.add_argument("audio", help="path to a local audio file (wav, mp3, flac, ...)")
    p.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    p.add_argument("--no-separate", action="store_true", help="skip Demucs; use for solo guitar recordings")
    p.add_argument("--device", help="torch device for Demucs, e.g. cuda or cpu")
    p.add_argument("--shifts", type=int, default=1, help="Demucs passes to average: cleaner stem, N times slower")
    p.add_argument(
        "--channel",
        choices=("mix", "left", "right", "both"),
        default="mix",
        help="stereo channel to tab (default: mix); metal often pans two guitars hard left and right, "
        "and 'both' writes a tab for each side",
    )
    p.add_argument(
        "--remove-center",
        action="store_true",
        help="with --channel: strip what's identical in both channels, usually bass bleed",
    )
    p.add_argument(
        "--tuning",
        type=_tuning,
        default=PRESETS["standard"],
        help=f"preset ({', '.join(PRESETS)}) or notes low->high, e.g. D2,A2,D3,G3,B3,E4",
    )
    p.add_argument("--capo", type=int, default=0, help="capo fret (frets in the tab are relative to it)")
    p.add_argument("--tempo", type=float, help="override detected tempo (BPM)")
    p.add_argument("--time-signature", type=_time_signature, default=(4, 4), help="default 4/4")
    p.add_argument(
        "--subdivision",
        type=int,
        choices=SUPPORTED_SUBDIVISIONS,
        default=4,
        help="grid ticks per beat: 2=8ths, 4=16ths (default), 8=32nds",
    )
    p.add_argument("--onset-threshold", type=float, default=0.5, help="Basic Pitch onset threshold (0-1)")
    p.add_argument("--frame-threshold", type=float, default=0.3, help="Basic Pitch frame threshold (0-1)")
    p.add_argument("--min-note-ms", type=float, default=58.0, help="drop notes shorter than this")
    p.add_argument("--format", default="musicxml,ascii", help="comma-separated: musicxml, ascii")
    p.add_argument("--title", help="title written into the tab")
    p.add_argument("--save-notes", action="store_true", help="also write the raw notes as JSON")
    p.add_argument(
        "--keep-stem", action="store_true", help="also keep the separated guitar WAV (and one per side with --channel)"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())
    unknown = set(formats) - {"musicxml", "ascii"}
    if unknown:
        print(f"Unknown format(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    if args.remove_center and args.channel == "mix":
        print("--remove-center needs --channel left, right or both", file=sys.stderr)
        return 2

    from .pipeline import PipelineConfig, run

    cfg = PipelineConfig(
        separate=not args.no_separate,
        device=args.device,
        tuning=args.tuning,
        capo=args.capo,
        tempo=args.tempo,
        subdivision=args.subdivision,
        time_signature=args.time_signature,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        min_note_ms=args.min_note_ms,
        formats=formats,
        title=args.title,
        save_notes=args.save_notes,
        keep_stem=args.keep_stem,
        shifts=args.shifts,
        channel=args.channel,
        remove_center=args.remove_center,
    )
    try:
        outputs = run(args.audio, args.out, cfg)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for kind, path in outputs.items():
        print(f"{kind:14s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
