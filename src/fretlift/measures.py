"""Lay events out into measures, filling gaps with rests and splitting at barlines."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Event


@dataclass
class Segment:
    tick: int  # absolute tick
    duration: int
    event: Event | None  # None means rest
    tie_start: bool = False  # continues into the next segment
    tie_stop: bool = False  # continued from the previous segment


def ticks_per_measure(time_signature: tuple[int, int], subdivision: int) -> int:
    beats, beat_type = time_signature
    total = beats * subdivision * 4 / beat_type
    if total != int(total):
        raise ValueError(f"{beats}/{beat_type} doesn't fit a grid of {subdivision} ticks per beat")
    return int(total)


def layout(events: list[Event], time_signature: tuple[int, int], subdivision: int) -> list[list[Segment]]:
    """Return a list of measures, each a list of segments that fill it exactly."""
    tpm = ticks_per_measure(time_signature, subdivision)
    flat: list[Segment] = []
    cursor = 0
    for e in sorted(events, key=lambda e: e.tick):
        if e.tick < cursor:  # overlapping input; shouldn't happen after quantize
            continue
        if e.tick > cursor:
            flat.append(Segment(cursor, e.tick - cursor, None))
        flat.append(Segment(e.tick, e.duration, e))
        cursor = e.tick + e.duration
    end = max(tpm, -(-cursor // tpm) * tpm)  # round up to a full measure
    if cursor < end:
        flat.append(Segment(cursor, end - cursor, None))

    measures: list[list[Segment]] = [[] for _ in range(end // tpm)]
    for seg in flat:
        tick, remaining, first = seg.tick, seg.duration, True
        while remaining > 0:
            m = tick // tpm
            room = (m + 1) * tpm - tick
            take = min(room, remaining)
            measures[m].append(
                Segment(
                    tick,
                    take,
                    seg.event,
                    tie_start=seg.event is not None and take < remaining,
                    tie_stop=seg.event is not None and not first,
                )
            )
            tick += take
            remaining -= take
            first = False
    return measures
