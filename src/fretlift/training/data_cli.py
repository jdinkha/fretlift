"""``fretlift-data``: build manifests, inspect them, preview augmentation."""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict

import numpy as np

from ..tuning import parse_tuning
from .augment import AMP_PRESETS


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def cmd_guitarset(args):
    from .datasets import download_guitarset, prepare_guitarset

    if args.download:
        download_guitarset(args.root, include_pickup=not args.no_pickup)
    prepare_guitarset(args.root, include_pickup=not args.no_pickup)


def cmd_guitar_techs(args):
    from .datasets import download_guitar_techs, prepare_guitar_techs

    if args.download:
        download_guitar_techs(args.root)
    prepare_guitar_techs(args.root)


def cmd_egdb(args):
    from .datasets import download_egdb, prepare_egdb

    if args.download:
        download_egdb(args.root)
    prepare_egdb(args.root)


def cmd_folder(args):
    from .datasets import prepare_folder

    prepare_folder(
        args.audio, args.labels, args.out, args.name, tags=_csv(args.tags), split=args.split,
        tuning=parse_tuning(args.tuning) if args.tuning else None, midi_strings=args.midi_strings,
    )


def cmd_mix_separate(args):
    from .manifest import assign_splits, filter_tags, read_manifests
    from .separate_mix import mix_and_separate

    items = filter_tags(read_manifests(args.manifest), _csv(args.tags))
    if args.split != "all":
        items = assign_splits(items, _csv(args.val_groups), _csv(args.test_groups))[args.split]
    if args.max_items:
        items = items[: args.max_items]
    print(f"Mixing and separating {len(items)} items")
    out = mix_and_separate(items, args.accompaniment, args.out, snr_db=(args.snr_min, args.snr_max), seed=args.seed,
                           device=args.device)
    print(f"wrote {out}")


def cmd_stats(args):
    from .labels import load_labels
    from .manifest import assign_splits, read_manifests

    items = read_manifests(args.manifest)
    splits = assign_splits(items, _csv(args.val_groups), _csv(args.test_groups))
    print(f"{len(items)} items, {len({i.group for i in items})} groups")
    for split, lst in splits.items():
        tags = Counter(t for it in lst for t in it.tags)
        notes, with_strings = 0, 0
        for it in lst:
            ns, _ = load_labels(it.labels)
            notes += len(ns)
            with_strings += sum(n.string is not None for n in ns)
        tag_str = ", ".join(f"{t}:{c}" for t, c in tags.most_common(8))
        print(f"  {split:5s} {len(lst):5d} items  {notes:7d} notes ({with_strings} with strings)  {tag_str}")
    by_group = defaultdict(set)
    for split, lst in splits.items():
        for it in lst:
            by_group[split].add(it.group)
    print("  groups: " + "  ".join(f"{s}={len(g)}" for s, g in by_group.items()))


def cmd_preview(args):
    from pathlib import Path

    from .audio import load_audio, write_wav
    from .augment import AugmentConfig, Augmenter, apply_preset, load_irs
    from .config import SR

    x = load_audio(args.audio, SR)[: int(args.seconds * SR)]
    out = Path(args.out)
    if args.preset:
        irs = load_irs(args.ir_dir, SR)
        for k in range(args.n):
            p = write_wav(out / f"{args.preset}_{k}.wav", apply_preset(x, SR, args.preset, args.seed + k, irs), SR)
            print(p)
        return
    aug = Augmenter(AugmentConfig(ir_dir=args.ir_dir), SR)
    rng = np.random.default_rng(args.seed)
    write_wav(out / "original.wav", x, SR)
    for k in range(args.n):
        y, _, desc = aug(x, [], rng)
        print(write_wav(out / f"aug_{k:02d}_{desc}.wav", y, SR))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="fretlift-data", description="Prepare datasets for training and evaluation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("guitarset", help="download/convert GuitarSet (acoustic, per-string labels)")
    s.add_argument("--root", required=True)
    s.add_argument("--download", action="store_true", help="download from Zenodo first (~1.5 GB)")
    s.add_argument("--no-pickup", action="store_true", help="skip the pickup-mix audio version")
    s.set_defaults(func=cmd_guitarset)

    s = sub.add_parser("guitar-techs", help="download/convert Guitar-TECHS (electric, DI + mic'd amp, per-string labels)")
    s.add_argument("--root", required=True)
    s.add_argument("--download", action="store_true", help="download from Zenodo first (~4.1 GB)")
    s.set_defaults(func=cmd_guitar_techs)

    s = sub.add_parser("egdb", help="download/convert EGDB (electric, DI + five amp renders, per-string labels)")
    s.add_argument("--root", required=True)
    s.add_argument("--download", action="store_true", help="download from Google Drive first")
    s.set_defaults(func=cmd_egdb)

    s = sub.add_parser("folder", help="your own recordings: audio files + matching label files")
    s.add_argument("--audio", required=True, help="folder of audio files")
    s.add_argument("--labels", required=True, help="folder of .json/.mid/.jams labels with the same names")
    s.add_argument("--out", required=True)
    s.add_argument("--name", required=True, help="dataset name, used in ids and as a tag")
    s.add_argument("--tags", default="", help="extra tags, e.g. electric,distorted")
    s.add_argument("--split", choices=["train", "val", "test"], help="put everything in one split (default: 80/10/10)")
    s.add_argument("--tuning", help="tuning preset or notes, if not standard")
    s.add_argument("--midi-strings", default="auto", choices=["auto", "none"])
    s.set_defaults(func=cmd_folder)

    s = sub.add_parser(
        "mix-separate", help="mix guitar with accompaniment and run Demucs (separation-artifact training data)"
    )
    s.add_argument("--manifest", action="append", required=True)
    s.add_argument("--accompaniment", required=True, help="folder of accompaniment audio you're allowed to use")
    s.add_argument("--out", required=True)
    s.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    s.add_argument("--val-groups", default="")
    s.add_argument("--test-groups", default="")
    s.add_argument("--tags", default="")
    s.add_argument("--snr-min", type=float, default=-6.0, help="guitar-to-accompaniment ratio range, dB")
    s.add_argument("--snr-max", type=float, default=6.0)
    s.add_argument("--max-items", type=int)
    s.add_argument("--device", help="cuda or cpu for Demucs")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_mix_separate)

    s = sub.add_parser("stats", help="summarize manifests and splits")
    s.add_argument("--manifest", action="append", required=True)
    s.add_argument("--val-groups", default="")
    s.add_argument("--test-groups", default="")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("preview-augment", help="render augmented versions of a file to listen to")
    s.add_argument("audio")
    s.add_argument("--out", default="previews")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--seconds", type=float, default=15.0)
    s.add_argument("--preset", choices=list(AMP_PRESETS), help="only this amp preset (as used by --distort)")
    s.add_argument("--ir-dir")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_preview)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
