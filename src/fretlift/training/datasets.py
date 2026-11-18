"""Convert datasets into manifests + canonical label files.

* GuitarSet (acoustic, per-string labels): downloadable from Zenodo here.
* Guitar-TECHS (electric, DI + mic'd amp, per-string labels): downloadable
  from Zenodo here.
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
from dataclasses import replace
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


def _fetch_zip(url: str, archive: Path, checksum: str, dest: Path, keep=lambda member: True) -> None:
    """Download ``archive`` (unless it's already there), check it, unpack the ``keep`` members, delete it."""
    if not archive.exists():
        print(f"Downloading {archive.name}")
        _download(url, archive)
    if _md5(archive) != checksum:
        raise RuntimeError(f"{archive} failed its checksum; delete it and download again")
    print(f"Unpacking {archive.name}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest, [m for m in z.namelist() if keep(m)])
    archive.unlink()


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
        _fetch_zip(GUITARSET_URL.format(name=name), root / name, checksum, target_dir)


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


# ------------------------------------------------------------- Guitar-TECHS

GUITAR_TECHS_URL = "https://zenodo.org/api/records/14963133/files/{name}/content"
GUITAR_TECHS_FILES = {  # md5 checksums from the Zenodo record
    "P1_chords.zip": "be9ef8bbdceb1912d565254e607a6d94",
    "P1_scales.zip": "9c0b98e8fb42a522df727ea8bf545e4f",
    "P1_singlenotes.zip": "ca0c4674dde3805574685a313f7c39eb",
    "P1_techniques.zip": "18634a41a6db5a8de10d07eb3122a872",
    "P2_chords.zip": "eb6f74dd19162237189281688ad7ad2e",
    "P2_scales.zip": "96664853872f51e5f8aa4447313b7cf5",
    "P2_singlenotes.zip": "40fbf03d8b04bb2cf42df20f36dc2254",
    "P2_techniques.zip": "f4189251ce50be25f06a173b2c2bba00",
    "P3_music.zip": "071ba80aecf00f4a31fbd167b3f22198",
}

# How late the MIDI pickup's notes are against the audio, in ms, per player and string (high e first).
# Measured by cross-correlating note starts with the DI audio's onset strength, pooled over each player's
# takes, minus the same measurement on GuitarSet (whose labels are aligned). Low strings lag most: pitch
# tracking needs a few cycles of the waveform. Left uncorrected, the model would learn late onsets.
GUITAR_TECHS_LATE_MS = {
    "P1": (18.9, 11.6, 14.5, 16.0, 21.8, 27.6),
    "P2": (8.7, 5.8, 4.4, 4.4, 11.6, 18.9),
    "P3": (8.7, 7.3, 5.8, 8.7, 7.3, 23.2),
}


def _guitar_techs_member(name: str) -> bool:
    # The labels and the two audio versions; skips the room-mic MP3s under video/ and macOS clutter.
    return not name.startswith("__MACOSX") and not name.endswith(".DS_Store") and ("/midi/" in name or "/audio/" in name)


def download_guitar_techs(root: str | Path) -> None:
    """Download and unpack Guitar-TECHS (4.1 GB of zips, about the same unpacked)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, checksum in GUITAR_TECHS_FILES.items():
        part = root / name.removesuffix(".zip")
        if part.exists() and any(part.rglob("*.mid")):
            print(f"{part.name}: already unpacked")
            continue
        _fetch_zip(GUITAR_TECHS_URL.format(name=name), root / name, checksum, root, keep=_guitar_techs_member)


def prepare_guitar_techs(root: str | Path) -> Path:
    """Write ``<root>/manifest.jsonl``: a DI item and a mic'd-amp item per take.

    Layout: ``P1_chords/midi/midi_<take>.mid`` with
    ``P1_chords/audio/{directinput,micamp}/<kind>_<take>.wav``. Groups are the
    players (``guitar-techs-P1`` .. ``-P3``), so a player can be held out with
    --val-groups/--test-groups; otherwise everything trains. Player 3 recorded
    the only real pieces of music, which makes them the natural validation set.
    """
    root = Path(root)
    midis = sorted(p for p in root.rglob("midi_*.mid") if "midi" in p.parent.parts)
    if not midis:
        raise FileNotFoundError(f"No Guitar-TECHS MIDI files under {root}. Run with --download.")
    items, missing, dropped, total = [], 0, 0, 0
    for midi in midis:
        midi_dir = next(p for p in midi.parents if p.name == "midi")
        part, sub = midi_dir.parent, midi.parent.relative_to(midi_dir)
        player, _, category = part.name.partition("_")
        take = midi.stem.removeprefix("midi_")
        notes, _ = load_labels(midi)
        # The MIDI pickup occasionally reports a fret the 22-fret guitars don't have; those notes didn't happen.
        # STANDARD runs low to high, so STANDARD[-s] is string s's open pitch.
        kept = [n for n in notes if n.string is None or 0 <= n.pitch - STANDARD[-n.string] <= 22]
        late = GUITAR_TECHS_LATE_MS.get(player)
        if late:
            kept = [n if n.string is None else replace(n, start=max(0.0, n.start - late[n.string - 1] / 1000),
                                                        end=n.end - late[n.string - 1] / 1000) for n in kept]
        dropped, total = dropped + len(notes) - len(kept), total + len(notes)
        key = "_".join((part.name, *sub.parts, take))
        labels = save_labels(root / "labels" / f"{key}.json", kept, tuning=STANDARD, source="guitar-techs")
        for kind, tag in (("directinput", "di"), ("micamp", "amp")):
            audio = part / "audio" / kind / sub / f"{kind}_{take}.wav"
            if not audio.exists():
                missing += 1
                continue
            items.append(
                Item(
                    id=f"guitar-techs/{key}/{tag}",
                    audio=audio,
                    labels=labels,
                    group=f"guitar-techs-{player}",
                    tags=("guitar-techs", "electric", tag, category),
                    tuning=STANDARD,
                )
            )
    if not items:
        raise FileNotFoundError(f"Found Guitar-TECHS MIDI under {root}, but none of the matching audio")
    if missing:
        log.warning("%d Guitar-TECHS audio files were missing and skipped", missing)
    out = write_manifest(root / "manifest.jsonl", items)
    print(f"{len(items)} items from {len(midis)} takes -> {out} (dropped {dropped} of {total} notes with impossible frets)")
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

