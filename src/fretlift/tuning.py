"""Tuning presets and note-name helpers."""

from __future__ import annotations

import re

PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Open-string pitches as MIDI numbers, ordered lowest string -> highest string.
PRESETS: dict[str, tuple[int, ...]] = {
    "standard": (40, 45, 50, 55, 59, 64),  # E2 A2 D3 G3 B3 E4
    "drop-d": (38, 45, 50, 55, 59, 64),
    "half-step-down": (39, 44, 49, 54, 58, 63),
    "full-step-down": (38, 43, 48, 53, 57, 62),
    "open-g": (38, 43, 50, 55, 59, 62),
    "open-d": (38, 45, 50, 54, 57, 62),
    "dadgad": (38, 45, 50, 55, 57, 62),
}

# Standard and drop tunings for every lowest-string note from G1 up to E2, under
# both sharp and flat names (c#-standard == db-standard).
_LOWEST = {
    "g": 31, "g#": 32, "ab": 32, "a": 33, "a#": 34, "bb": 34, "b": 35,
    "c": 36, "c#": 37, "db": 37, "d": 38, "d#": 39, "eb": 39, "e": 40,
}  # fmt: skip
# Standard: fourths, except a major third between the G and B strings (E standard: E A D G B E).
PRESETS |= {f"{n}-standard": tuple(low + i for i in (0, 5, 10, 15, 19, 24)) for n, low in _LOWEST.items()}
# Drop: the lowest string a whole step below the standard tuning of the rest (drop D: D A D G B E).
PRESETS |= {f"drop-{n}": tuple(low + i for i in (0, 7, 12, 17, 21, 26)) for n, low in _LOWEST.items()}

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")


def note_to_midi(name: str) -> int:
    """Convert a scientific pitch name such as ``E2`` or ``F#3`` to MIDI."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"Not a note name: {name!r} (expected something like E2 or F#3)")
    letter, accidental, octave = m.groups()
    pc = PITCH_CLASSES[letter.upper()] + {"#": 1, "b": -1, "": 0}[accidental]
    return 12 * (int(octave) + 1) + pc


def midi_to_name(pitch: int) -> str:
    return f"{SHARP_NAMES[pitch % 12]}{pitch // 12 - 1}"


def parse_tuning(spec: str) -> tuple[int, ...]:
    """Parse a preset name or a comma-separated list of notes, low to high.

    >>> parse_tuning("standard")
    (40, 45, 50, 55, 59, 64)
    >>> parse_tuning("b-standard")
    (35, 40, 45, 50, 54, 59)
    >>> parse_tuning("drop-c#")
    (37, 44, 49, 54, 58, 63)
    >>> parse_tuning("D2,A2,D3,G3,B3,E4")
    (38, 45, 50, 55, 59, 64)
    """
    key = spec.strip().lower()
    if key in PRESETS:
        return PRESETS[key]
    notes = tuple(note_to_midi(n) for n in spec.split(","))
    if len(notes) < 4:
        raise ValueError("A tuning needs at least 4 strings")
    if list(notes) != sorted(notes):
        raise ValueError("List the tuning from the lowest string to the highest")
    return notes
