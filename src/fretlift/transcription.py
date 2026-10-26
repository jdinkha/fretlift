"""Audio -> notes with Spotify's Basic Pitch."""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from pathlib import Path

from .types import Note

log = logging.getLogger(__name__)


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def drop_ghost_harmonics(notes: list[Note], ratio: float = 0.6, onset_tol: float = 0.05) -> list[Note]:
    """Drop quiet notes that are really overtones of a louder note starting at the same time.

    Basic Pitch often reports the 2nd-4th harmonics (an octave, an octave and a fifth,
    two octaves up) as extra, quieter notes. Real octaves are played at a similar level,
    so they survive. ``notes`` must be sorted by start time.
    """
    starts = [n.start for n in notes]
    kept = []
    for n in notes:
        lo, hi = bisect_left(starts, n.start - onset_tol), bisect_right(starts, n.start + onset_tol)
        if not any(n.pitch - m.pitch in (12, 19, 24) and n.velocity < ratio * m.velocity for m in notes[lo:hi]):
            kept.append(n)
    if len(kept) < len(notes):
        log.info("Dropped %d ghost harmonics", len(notes) - len(kept))
    return kept


def transcribe(
    audio_path: Path,
    *,
    min_pitch: int = 38,
    max_pitch: int = 88,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_ms: float = 58.0,
) -> list[Note]:
    """Return the notes Basic Pitch hears, limited to the instrument's range."""
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
    except ImportError as e:
        if e.name != "basic_pitch":
            raise  # installed but broken: show the real error
        raise RuntimeError("Basic Pitch isn't installed. Run `pip install -e .[ml]`.") from e

    log.info("Transcribing notes with Basic Pitch")
    _, _, note_events = predict(
        str(audio_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_ms,
        # A little margin so slightly sharp/flat notes at the edges survive.
        minimum_frequency=midi_to_hz(min_pitch - 0.5),
        maximum_frequency=midi_to_hz(max_pitch + 0.5),
        melodia_trick=True,
    )
    notes = [
        Note(start=float(s), end=float(e), pitch=int(p), velocity=float(a))
        for s, e, p, a, *_ in note_events
        if min_pitch <= int(p) <= max_pitch
    ]
    notes.sort(key=lambda n: (n.start, n.pitch))
    notes = drop_ghost_harmonics(notes)
    log.info("Found %d notes", len(notes))
    return notes
