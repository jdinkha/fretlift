"""Convert datasets into manifests + canonical label files.

* GuitarSet (acoustic, per-string labels): downloadable from Zenodo here.
* EGDB (electric, DI + amp renders): download it yourself; the converter
  discovers the layout instead of assuming one.
* Any folder of your own recordings with matching label files.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

from ..tuning import PRESETS
from .audio import AUDIO_SUFFIXES
from .labels import LABEL_SUFFIXES, load_labels, parse_jams, save_labels
from .manifest import Item, hash_split, write_manifest

log = logging.getLogger(__name__)

STANDARD = PRESETS["standard"]

# ---------------------------------------------------------------- GuitarSet

GUITARSET_URL = "https://zenodo.org/record/3371780/files/{name}?download=1"
# md5 checksums as published by mirdata's GuitarSet loader
GUITARSET_FILES = {
    "annotation.zip": "b39b78e63d3446f2e54ddb7a54df9b10",
    "audio_mono-mic.zip": "275966d6610ac34999b58426beb119c3",
    "audio_mono-pickup_mix.zip": "aecce79f425a44e2055e46f680e10f6a",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, next_report = 0, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                print(f"  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
                next_report += 0.1
    tmp.replace(dest)


def download_guitarset(root: str | Path, include_pickup: bool = True) -> None:
    """Download and unpack GuitarSet (about 1.5 GB with both audio versions)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, checksum in GUITARSET_FILES.items():
        if name == "audio_mono-pickup_mix.zip" and not include_pickup:
            continue
        target_dir = root / name.removesuffix(".zip")
        if target_dir.exists() and any(target_dir.rglob("*")):
            print(f"{target_dir.name}: already unpacked")
            continue
        archive = root / name
        if not archive.exists():
            print(f"Downloading {name}")
            _download(GUITARSET_URL.format(name=name), archive)
        if _md5(archive) != checksum:
            raise RuntimeError(f"{archive} failed its checksum; delete it and download again")
        print(f"Unpacking {name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(target_dir)
        archive.unlink()


def prepare_guitarset(root: str | Path, include_pickup: bool = True) -> Path:
    """Write ``<root>/manifest.jsonl``. Groups are player ids "00".."05"."""
    root = Path(root)
    jams_files = sorted(root.rglob("*.jams"))
    if not jams_files:
        raise FileNotFoundError(f"No .jams files under {root}. Run with --download, or unpack annotation.zip there.")
    wavs = {p.name: p for p in root.rglob("*.wav")}
    items, missing = [], 0
    for jams in jams_files:
        tid = jams.stem  # e.g. 00_BN1-129-Eb_comp
        player = tid.split("_")[0]
        mode = "solo" if tid.endswith("_solo") else "comp"
        labels = save_labels(root / "labels" / f"{tid}.json", parse_jams(jams), tuning=STANDARD, source="guitarset")
        versions = [("mic", f"{tid}_mic.wav")] + ([("pickup", f"{tid}_mix.wav")] if include_pickup else [])
        for kind, fname in versions:
            if fname not in wavs:
                missing += 1
                continue
            items.append(
                Item(
                    id=f"guitarset/{tid}/{kind}",
                    audio=wavs[fname],
                    labels=labels,
                    group=f"guitarset-player-{player}",
                    tags=("guitarset", "acoustic", kind, mode),
                    tuning=STANDARD,
                )
            )
    if not items:
        raise FileNotFoundError(f"Found annotations but no matching audio under {root}")
    if missing:
        log.warning("%d GuitarSet audio files were missing and skipped", missing)
    out = write_manifest(root / "manifest.jsonl", items)
    print(f"{len(items)} items from {len(jams_files)} excerpts -> {out}")
    return out


# --------------------------------------------------------------------- EGDB


def prepare_egdb(root: str | Path, midi_strings: str = "auto") -> Path:
    """Write ``<root>/manifest.jsonl`` for EGDB-style data.

    Expects folders of audio named ``audio_<tone>`` (``audio_DI`` for the
    clean direct input, the others for amp renders) and label files (MIDI or
    JAMS) somewhere under ``root`` whose names match the audio files.
    Every render of one take shares a group, so the take is never split across
    train and test.
    """
    root = Path(root)
    label_files = {p.stem: p for p in sorted(root.rglob("*")) if p.suffix.lower() in LABEL_SUFFIXES and p.is_file()}
    label_files = {k: v for k, v in label_files.items() if "fretlift" not in v.parts and v.name != "manifest.jsonl"}
    audio_dirs = [
        d
        for d in sorted(root.rglob("audio_*"))
        if d.is_dir() and "label" not in d.name.lower() and any(p.suffix.lower() in AUDIO_SUFFIXES for p in d.iterdir())
    ]
    if not audio_dirs or not label_files:
        found = sorted({p.parent.name for p in root.rglob("*") if p.is_file()})[:20]
        raise FileNotFoundError(
            f"Couldn't find EGDB-style data under {root}: expected audio_<tone>/ folders and MIDI/JAMS labels. "
            f"Folders with files found: {found}. If the layout differs, use `fretlift-data folder` per tone instead."
        )

    out_labels = root / "fretlift_labels"
    converted: dict[str, Path] = {}
    with_strings = 0
    items, unmatched = [], 0
    for d in audio_dirs:
        tone = d.name[len("audio_") :]
        kind = "di" if tone.lower() == "di" else "amp"
        for audio in sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES):
            src = label_files.get(audio.stem)
            if src is None:
                unmatched += 1
                continue
            if audio.stem not in converted:
                notes, _ = load_labels(src, midi_strings=midi_strings)
                with_strings += bool(notes) and all(n.string is not None for n in notes)
                converted[audio.stem] = save_labels(out_labels / f"{audio.stem}.json", notes, tuning=STANDARD, source="egdb")
            items.append(
                Item(
                    id=f"egdb/{audio.stem}/{tone}",
                    audio=audio,
                    labels=converted[audio.stem],
                    group=f"egdb-{audio.stem}",
                    tags=("egdb", "electric", kind, f"tone-{tone.lower()}"),
                    split=hash_split(f"egdb:{audio.stem}"),
                    tuning=STANDARD,
                )
            )
    if not items:
        raise FileNotFoundError("Found EGDB audio and labels, but no file names matched between them")
    if unmatched:
        log.warning("%d audio files had no matching label file and were skipped", unmatched)
    print(f"{len(items)} items, {len(converted)} takes, {len(audio_dirs)} tones; string labels for {with_strings} takes")
    return write_manifest(root / "manifest.jsonl", items)


# ------------------------------------------------------------------- folder


def _find_label(stem: str, labels_dir: Path) -> Path | None:
    for name in (stem, f"{stem}.labels", f"{stem}.notes"):
        for suffix in LABEL_SUFFIXES:
            p = labels_dir / f"{name}{suffix}"
            if p.exists():
                return p
    return None


def prepare_folder(
    audio_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    name: str,
    tags: Sequence[str] = (),
    split: str | None = None,
    tuning: Sequence[int] | None = None,
    midi_strings: str = "auto",
) -> Path:
    """Your own recordings: ``song.wav`` + ``song.json`` / ``song.mid`` / ``song.jams``.

    Each recording is its own group. With ``split=None`` recordings are spread
    80/10/10 over train/val/test by name; pass ``split="test"`` for a pure
    evaluation set (the usual choice for a hand-labelled set of hard examples).
    """
    audio_dir, labels_dir, out_dir = Path(audio_dir), Path(labels_dir), Path(out_dir)
    tuning = tuple(tuning) if tuning else STANDARD
    items, unmatched = [], []
    for audio in sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES):
        src = _find_label(audio.stem, labels_dir)
        if src is None:
            unmatched.append(audio.name)
            continue
        notes, label_tuning = load_labels(src, midi_strings=midi_strings)
        labels = save_labels(out_dir / "labels" / f"{audio.stem}.json", notes, tuning=label_tuning or tuning, source=name)
        items.append(
            Item(
                id=f"{name}/{audio.stem}",
                audio=audio,
                labels=labels,
                group=f"{name}-{audio.stem}",
                tags=(name, *tags),
                split=split or hash_split(f"{name}:{audio.stem}"),
                tuning=label_tuning or tuning,
            )
        )
    if unmatched:
        log.warning("No label file for: %s", ", ".join(unmatched[:10]))
    if not items:
        raise FileNotFoundError(f"No audio in {audio_dir} had a matching label file in {labels_dir}")
    out = write_manifest(out_dir / "manifest.jsonl", items)
    print(f"{len(items)} items -> {out}")
    return out

