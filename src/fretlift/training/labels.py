"""Ground-truth labels: one canonical format, converters into it, and training targets.

Canonical label file (JSON)::

    {"version": 1,
     "tuning": [40, 45, 50, 55, 59, 64],          # optional, low -> high
     "source": "guitarset",                        # optional, free text
     "notes": [{"start": 0.51, "end": 0.98, "pitch": 64, "string": 1}, ...]}

``string`` uses the tab convention (1 = highest) and may be null when the
source only has pitches (most MIDI files). The string head of the model simply
isn't trained on those notes.

The ``*.notes.json`` files written by ``fretlift --save-notes`` load too.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..types import Note
from .config import IGNORE, N_STRINGS, PITCH_MIN

LABEL_SUFFIXES = (".json", ".jams", ".mid", ".midi")


# ----------------------------------------------------------------------------- io


def save_labels(path: str | Path, notes: Iterable[Note], tuning: Sequence[int] | None = None, source: str | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {"version": 1}
    if tuning is not None:
        doc["tuning"] = list(tuning)
    if source:
        doc["source"] = source
    doc["notes"] = [
        {"start": round(n.start, 5), "end": round(n.end, 5), "pitch": int(n.pitch), "string": n.string}
        for n in sorted(notes, key=lambda n: (n.start, n.pitch))
    ]
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return path


def load_labels(path: str | Path, midi_strings: str = "auto") -> tuple[list[Note], tuple[int, ...] | None]:
    """Load notes (and the tuning, if recorded) from any supported label file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jams":
        return parse_jams(path), None
    if suffix in (".mid", ".midi"):
        return parse_midi(path, strings=midi_strings), None
    if suffix == ".json":
        doc = json.loads(path.read_text())
        notes = [
            Note(start=float(n["start"]), end=float(n["end"]), pitch=int(round(n["pitch"])), string=n.get("string"))
            for n in doc.get("notes", [])
        ]
        tuning = tuple(doc["tuning"]) if doc.get("tuning") else None
        return notes, tuning
    raise ValueError(f"Unsupported label file: {path} (expected one of {LABEL_SUFFIXES})")


def _observations(data) -> Iterable[tuple[float, float, float]]:
    """JAMS stores observations either as a list of dicts or as parallel arrays."""
    if isinstance(data, dict):
        yield from zip(data["time"], data["duration"], data["value"])
    else:
        for obs in data:
            yield obs["time"], obs["duration"], obs["value"]


def parse_jams(path_or_doc, n_strings: int = N_STRINGS) -> list[Note]:
    """Read per-string notes from a GuitarSet-style JAMS file.

    GuitarSet has one ``note_midi`` annotation per string, with
    ``annotation_metadata.data_source`` = "0" for the low E string up to "5"
    for the high e string. Pitches are fractional (from pitch tracking) and
    get rounded.
    """
    doc = path_or_doc if isinstance(path_or_doc, dict) else json.loads(Path(path_or_doc).read_text())
    anns = [a for a in doc.get("annotations", []) if a.get("namespace") == "note_midi"]
    notes = []
    for order, ann in enumerate(anns):
        source = (ann.get("annotation_metadata") or {}).get("data_source")
        try:
            idx = int(source)
        except (TypeError, ValueError):
            idx = order
        string = n_strings - idx  # data_source 0 (low E) -> string 6
        for t, d, v in _observations(ann.get("data", [])):
            if v is None:
                continue
            notes.append(Note(start=float(t), end=float(t) + float(d), pitch=int(round(float(v))), string=string))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


STRING_NAMES = {"e": 1, "B": 2, "G": 3, "D": 4, "A": 5, "E": 6}  # case matters: "e" is high, "E" is low


def parse_midi(path: str | Path, strings: str = "auto") -> list[Note]:
    """Read notes from a MIDI file.

    ``strings="auto"``: one track or channel per string, as hexaphonic MIDI
    pickups and some tab programs export, gives string labels. Tracks named
    after their strings ("e", "B", "G", "D", "A", "E", as in Guitar-TECHS) are
    used by name, even when some strings never play. Otherwise a file with
    exactly six non-drum parts has them ranked by average pitch, so the order
    of the tracks doesn't matter. Anything else, or ``strings="none"``, gives
    notes without string labels.
    """
    try:
        import pretty_midi
    except ImportError as e:
        raise RuntimeError('Reading MIDI labels needs pretty_midi: pip install -e ".[train]"') from e

    pm = pretty_midi.PrettyMIDI(str(path))
    parts = [inst for inst in pm.instruments if inst.notes and not inst.is_drum]
    names = [inst.name.strip() for inst in parts]
    # A name can repeat: a stray note on another channel within the same string's track.
    by_name = strings == "auto" and bool(parts) and set(names) <= STRING_NAMES.keys()
    per_string = by_name or (strings == "auto" and len(parts) == N_STRINGS)
    if by_name:
        string_of = {id(inst): STRING_NAMES[name] for inst, name in zip(parts, names)}
    elif per_string:
        ranked = sorted(parts, key=lambda inst: -np.mean([n.pitch for n in inst.notes]))
        string_of = {id(inst): s for s, inst in enumerate(ranked, start=1)}
    notes = [
        Note(
            start=float(n.start),
            end=float(n.end),
            pitch=int(n.pitch),
            velocity=n.velocity / 127,
            string=string_of[id(inst)] if per_string else None,
        )
        for inst in parts
        for n in inst.notes
    ]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


# ------------------------------------------------------------------- transforms


def window_notes(notes: Sequence[Note], t0: float, t1: float) -> list[Note]:
    """Notes overlapping [t0, t1), shifted so t0 becomes 0 (starts may be negative)."""
    return [replace(n, start=n.start - t0, end=n.end - t0) for n in notes if n.end > t0 and n.start < t1]


def shift_pitch(notes: Sequence[Note], semitones: int) -> list[Note]:
    """Labels for pitch-shifted audio. Strings are unchanged.

    Shifting a recording down two semitones sounds like the same performance on
    a guitar tuned down a whole step: same strings, same frets, lower pitches.
    That's why the model predicts strings separately from pitches, so this kind
    of augmentation (and down-tuned metal) works without knowing the tuning.
    """
    return [replace(n, pitch=n.pitch + semitones) for n in notes]


def make_targets(
    notes: Sequence[Note],
    n_frames: int,
    fps: float,
    pitch_min: int = PITCH_MIN,
    n_pitches: int = 56,
    onset_frames: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame-level training targets.

    Returns ``frame`` [T, P] (note sounding), ``onset`` [T, P] (note starts in
    this frame or the next ``onset_frames - 1``), and ``string`` [T, P] int64:
    the string index (0 = highest string) where a labelled note sounds, IGNORE
    elsewhere.
    """
    frame = np.zeros((n_frames, n_pitches), np.float32)
    onset = np.zeros((n_frames, n_pitches), np.float32)
    string = np.full((n_frames, n_pitches), IGNORE, np.int64)
    for n in notes:
        p = n.pitch - pitch_min
        if not 0 <= p < n_pitches:
            continue
        on = int(round(n.start * fps))
        off = max(on + 1, int(round(n.end * fps)))
        a, b = max(on, 0), min(off, n_frames)
        if a >= b:
            continue
        frame[a:b, p] = 1.0
        if on >= 0:
            onset[on : min(on + onset_frames, n_frames), p] = 1.0
        if n.string is not None:
            string[a:b, p] = n.string - 1
    return frame, onset, string
