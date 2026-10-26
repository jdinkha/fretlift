"""Small data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Note:
    """A transcribed note in absolute time."""

    start: float  # seconds
    end: float  # seconds
    pitch: int  # MIDI note number
    velocity: float = 1.0


@dataclass(frozen=True)
class TabNote:
    """A pitch placed on the fretboard.

    ``string`` follows the MusicXML / tab convention: 1 is the highest-pitched
    string (high e in standard tuning), 6 is the lowest.
    ``fret`` is relative to the capo (0 = open string or capo).
    """

    pitch: int
    string: int
    fret: int


@dataclass
class Event:
    """A group of notes that start at the same grid position.

    ``tick`` and ``duration`` are in grid units (``subdivision`` ticks per beat).
    """

    tick: int
    duration: int
    pitches: list[int]
    tab: list[TabNote] = field(default_factory=list)


@dataclass
class BeatGrid:
    """Beat positions detected in the recording."""

    tempo: float  # BPM
    beats: list[float]  # seconds
