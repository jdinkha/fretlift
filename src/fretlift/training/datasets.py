"""Convert datasets into manifests + canonical label files.

* GuitarSet (acoustic, per-string labels): downloadable from Zenodo here.
* Guitar-TECHS (electric, DI + mic'd amp, per-string labels): downloadable
  from Zenodo here.
* EGDB (electric, DI + five amp renders, per-string labels): downloadable
  from Google Drive here.
* Any folder of your own recordings with matching label files.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from ..tuning import PRESETS
from ..types import Note
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


def _fetch_zips(jobs: list[tuple]) -> None:
    """Run ``_fetch_zip`` jobs all at once: Zenodo caps each connection's speed, not the total."""
    if not jobs:
        return
    with ThreadPoolExecutor(len(jobs)) as pool:
        for future in [pool.submit(_fetch_zip, *job) for job in jobs]:
            future.result()  # re-raises a failed download


def download_guitarset(root: str | Path, include_pickup: bool = True) -> None:
    """Download and unpack GuitarSet (about 1.5 GB with both audio versions)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, checksum in GUITARSET_FILES.items():
        if name == "audio_mono-pickup_mix.zip" and not include_pickup:
            continue
        target_dir = root / name.removesuffix(".zip")
        if target_dir.exists() and any(target_dir.rglob("*")):
            print(f"{target_dir.name}: already unpacked")
            continue
        jobs.append((GUITARSET_URL.format(name=name), root / name, checksum, target_dir))
    _fetch_zips(jobs)


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
    jobs = []
    for name, checksum in GUITAR_TECHS_FILES.items():
        part = root / name.removesuffix(".zip")
        if part.exists() and any(part.rglob("*.mid")):
            print(f"{part.name}: already unpacked")
            continue
        jobs.append((GUITAR_TECHS_URL.format(name=name), root / name, checksum, root, _guitar_techs_member))
    _fetch_zips(jobs)


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


EGDB_FOLDER = "https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_"
EGDB_TONES = ("DI", "Ftwin", "JCjazz", "Marshall", "Mesa", "Plexi")  # the DI and five amp renders of it
# The labels run early: pooled over all clips, the audio's onsets come 16 ms (DI) to 20 ms (renders) after
# them, less the ~3 ms the same measurement gives on GuitarSet. Per-clip estimates didn't agree between
# tones, so it's one constant.
EGDB_EARLY_S = 0.013


def download_egdb(root: str | Path) -> None:
    """Download EGDB from its Google Drive folder: 240 clips' MIDI, DI and amp renders.

    Files already there are skipped, so an interrupted download can just be run again.
    """
    import gdown  # only EGDB needs it: it's the way to read a public Drive folder

    root = Path(root)
    keep = {"audio_label", *(f"audio_{t}" for t in EGDB_TONES)}  # skips RealData/, 13 unlabelled-string clips
    files = gdown.download_folder(EGDB_FOLDER, output=str(root), skip_download=True, quiet=True, use_cookies=False)
    todo = [f for f in files if f.path.split("/")[0] in keep and not Path(f.local_path).exists()]
    print(f"Downloading {len(todo)} EGDB files")

    def get(f):
        Path(f.local_path).parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=f.id, output=f"{f.local_path}.part", quiet=True, use_cookies=False)
        Path(f"{f.local_path}.part").replace(f.local_path)  # only whole files count as already there

    try:
        with ThreadPoolExecutor(8) as pool:  # Drive, like Zenodo, is faster over several connections
            list(pool.map(get, todo))  # re-raises a failed download
    except gdown.exceptions.FileURLRetrievalError:
        left = sum(not Path(f.local_path).exists() for f in todo)
        raise RuntimeError(
            f"Google Drive stopped serving EGDB to this address ({left} files still to get; it limits bulk "
            "downloads for a while). The files so far are kept: run the same command again later for the rest."
        ) from None


def _egdb_notes(path: Path) -> list[Note]:
    """EGDB puts each string on its own MIDI channel: channel 0 is the high E, channel 5 the low E.

    (Track names and order vary between files, so the generic MIDI reader can't be used.)
    """
    import mido

    t, sounding, notes = 0.0, {}, []
    for msg in mido.MidiFile(path):  # all tracks merged, times in seconds
        t += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        key = (msg.channel, msg.note)
        if key in sounding:  # a note-off, or the same note struck again
            start, velocity = sounding.pop(key)
            notes.append(Note(start, t, msg.note, velocity / 127, msg.channel + 1 if msg.channel < 6 else None))
        if msg.type == "note_on" and msg.velocity:
            sounding[key] = (t, msg.velocity)
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def prepare_egdb(root: str | Path) -> Path:
    """Write ``<root>/manifest.jsonl``: an item per clip and tone.

    Layout: ``audio_label/<n>.midi`` with ``audio_<tone>/<n>.wav`` (or .flac). Each clip is
    its own group (``egdb-<n>``), shared by its tones, and everything trains.
    """
    root = Path(root)
    midis = sorted((root / "audio_label").glob("*.mid*"))
    if not midis:
        raise FileNotFoundError(f"No EGDB labels in {root / 'audio_label'}. Run with --download.")
    items, dropped, total = [], 0, 0
    for midi in midis:
        notes = _egdb_notes(midi)
        kept = [replace(n, start=n.start + EGDB_EARLY_S, end=n.end + EGDB_EARLY_S) for n in notes
                if n.string is not None and 0 <= n.pitch - STANDARD[-n.string] <= 24]
        dropped, total = dropped + len(notes) - len(kept), total + len(notes)
        labels = save_labels(root / "labels" / f"{midi.stem}.json", kept, tuning=STANDARD, source="egdb")
        for tone in EGDB_TONES:
            # .wav as downloaded, or .flac in a smaller mono copy (training mixes to mono anyway)
            audio = next((p for p in (root / f"audio_{tone}").glob(f"{midi.stem}.*") if p.suffix in AUDIO_SUFFIXES), None)
            if audio:
                kind = "di" if tone == "DI" else "amp"
                items.append(Item(id=f"egdb/{midi.stem}/{tone}", audio=audio, labels=labels, group=f"egdb-{midi.stem}",
                                  tags=("egdb", "electric", kind, f"tone-{tone.lower()}"), tuning=STANDARD))
    if not items:
        raise FileNotFoundError(f"Found EGDB labels under {root}, but none of the matching audio")
    out = write_manifest(root / "manifest.jsonl", items)
    print(f"{len(items)} items from {len(midis)} clips -> {out} (dropped {dropped} of {total} unplayable notes)")
    return out


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

