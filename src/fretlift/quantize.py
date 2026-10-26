"""Snap transcribed notes onto a rhythmic grid built from detected beats."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import BeatGrid, Event, Note


def _extend_beats(beats: list[float], tempo: float, t_min: float, t_max: float) -> np.ndarray:
    """Return beat times covering [t_min, t_max], extrapolating at both ends."""
    b = np.asarray(sorted(beats), dtype=float)
    if len(b) >= 2:
        period = float(np.median(np.diff(b)))
    else:
        period = 60.0 / tempo if tempo > 0 else 0.5
        b = np.array([b[0] if len(b) else 0.0])

    before = []
    t = b[0] - period
    while t > t_min - period:
        before.append(t)
        t -= period
    after = []
    t = b[-1] + period
    while t < t_max + 2 * period:
        after.append(t)
        t += period
    return np.concatenate([np.array(before[::-1]), b, np.array(after)])


def quantize(notes: list[Note], grid: BeatGrid, subdivision: int = 4) -> list[Event]:
    """Group notes into events on a grid of ``subdivision`` ticks per beat.

    Tick 0 is the first beat at or before the first note, so measure 1 starts
    there. Each event's duration runs until the next event (a single voice of
    notes and chords, which is how most tabs are written), shortened if every
    note in it ends earlier, leaving a rest.
    """
    if not notes:
        return []
    if subdivision < 1:
        raise ValueError("subdivision must be >= 1")

    t_min = min(n.start for n in notes)
    t_max = max(n.end for n in notes)
    beats = _extend_beats(grid.beats, grid.tempo, t_min, t_max)

    def to_beats(t: float) -> float:
        i = int(np.clip(np.searchsorted(beats, t, side="right") - 1, 0, len(beats) - 2))
        frac = (t - beats[i]) / (beats[i + 1] - beats[i])
        return i + frac

    first_beat = int(np.floor(to_beats(t_min) + 1e-9))
    origin = first_beat * subdivision

    onsets: dict[int, set[int]] = defaultdict(set)
    ends: dict[int, int] = {}
    for n in notes:
        on = round(to_beats(n.start) * subdivision) - origin
        off = max(on + 1, round(to_beats(n.end) * subdivision) - origin)
        onsets[on].add(n.pitch)
        ends[on] = max(ends.get(on, on + 1), off)

    ticks = sorted(onsets)
    events: list[Event] = []
    for k, tick in enumerate(ticks):
        gap = ticks[k + 1] - tick if k + 1 < len(ticks) else ends[tick] - tick
        duration = max(1, min(gap, ends[tick] - tick))
        events.append(Event(tick=tick, duration=duration, pitches=sorted(onsets[tick])))
    return events
