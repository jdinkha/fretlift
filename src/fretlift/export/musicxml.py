"""Write a MusicXML file with a 6-line TAB staff.

MusicXML opens in MuseScore, Guitar Pro, TuxGuitar and alphaTab, which makes
it a good interchange format for a first version.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..measures import Segment, layout
from ..tuning import SHARP_NAMES
from ..types import Event

# (length in quarter notes, MusicXML type, dots), longest first
_NOTE_VALUES = [
    (4.0, "whole", 0),
    (3.0, "half", 1),
    (2.0, "half", 0),
    (1.5, "quarter", 1),
    (1.0, "quarter", 0),
    (0.75, "eighth", 1),
    (0.5, "eighth", 0),
    (0.375, "16th", 1),
    (0.25, "16th", 0),
    (0.125, "32nd", 0),
]

SUPPORTED_SUBDIVISIONS = (1, 2, 4, 8)


def split_duration(ticks: int, subdivision: int) -> list[tuple[int, str, int]]:
    """Break a duration into notatable values: [(ticks, type, dots), ...]."""
    values = [
        (int(q * subdivision), name, dots)
        for q, name, dots in _NOTE_VALUES
        if (q * subdivision).is_integer() and q * subdivision >= 1
    ]
    out = []
    remaining = ticks
    while remaining > 0:
        for t, name, dots in values:
            if t <= remaining:
                out.append((t, name, dots))
                remaining -= t
                break
        else:  # pragma: no cover - 1 tick is always representable for supported grids
            raise ValueError(f"Can't notate {remaining} ticks at subdivision {subdivision}")
    return out


def _sub(parent: ET.Element, tag: str, text: str | int | None = None, **attrib: str) -> ET.Element:
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = str(text)
    return el


def _pitch(parent: ET.Element, midi: int) -> None:
    name = SHARP_NAMES[midi % 12]
    p = _sub(parent, "pitch")
    _sub(p, "step", name[0])
    if len(name) > 1:
        _sub(p, "alter", 1)
    _sub(p, "octave", midi // 12 - 1)


def _write_segment(measure: ET.Element, seg: Segment, subdivision: int) -> None:
    pieces = split_duration(seg.duration, subdivision)
    for k, (ticks, ntype, dots) in enumerate(pieces):
        tie_stop = seg.tie_stop or k > 0
        tie_start = seg.tie_start or k < len(pieces) - 1
        notes = seg.event.tab if seg.event is not None else [None]
        for i, tab in enumerate(notes):
            n = _sub(measure, "note")
            if i > 0:
                _sub(n, "chord")
            if tab is None:
                _sub(n, "rest")
            else:
                _pitch(n, tab.pitch)
            _sub(n, "duration", ticks)
            if tab is not None:
                if tie_stop:
                    _sub(n, "tie", type="stop")
                if tie_start:
                    _sub(n, "tie", type="start")
            _sub(n, "voice", 1)
            _sub(n, "type", ntype)
            for _ in range(dots):
                _sub(n, "dot")
            if tab is not None:
                notations = _sub(n, "notations")
                if tie_stop:
                    _sub(notations, "tied", type="stop")
                if tie_start:
                    _sub(notations, "tied", type="start")
                technical = _sub(notations, "technical")
                _sub(technical, "string", tab.string)
                _sub(technical, "fret", tab.fret)


def to_musicxml(
    events: list[Event],
    *,
    tuning: tuple[int, ...],
    capo: int = 0,
    tempo: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
    subdivision: int = 4,
    title: str = "Untitled",
) -> ET.ElementTree:
    if subdivision not in SUPPORTED_SUBDIVISIONS:
        raise ValueError(f"subdivision must be one of {SUPPORTED_SUBDIVISIONS}")

    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", title)
    ident = _sub(root, "identification")
    enc = _sub(ident, "encoding")
    _sub(enc, "software", "fretlift")
    part_list = _sub(root, "part-list")
    sp = _sub(part_list, "score-part", id="P1")
    _sub(sp, "part-name", "Guitar")
    part = _sub(root, "part", id="P1")

    for number, segments in enumerate(layout(events, time_signature, subdivision), start=1):
        measure = _sub(part, "measure", number=str(number))
        if number == 1:
            attrs = _sub(measure, "attributes")
            _sub(attrs, "divisions", subdivision)
            key = _sub(attrs, "key")
            _sub(key, "fifths", 0)
            time = _sub(attrs, "time")
            _sub(time, "beats", time_signature[0])
            _sub(time, "beat-type", time_signature[1])
            clef = _sub(attrs, "clef")
            _sub(clef, "sign", "TAB")
            _sub(clef, "line", 5)
            details = _sub(attrs, "staff-details")
            _sub(details, "staff-lines", len(tuning))
            for line, midi in enumerate(tuning, start=1):  # line 1 = lowest string
                st = _sub(details, "staff-tuning", line=str(line))
                name = SHARP_NAMES[midi % 12]
                _sub(st, "tuning-step", name[0])
                if len(name) > 1:
                    _sub(st, "tuning-alter", 1)
                _sub(st, "tuning-octave", midi // 12 - 1)
            if capo:
                _sub(details, "capo", capo)

            direction = _sub(measure, "direction", placement="above")
            dtype = _sub(direction, "direction-type")
            metronome = _sub(dtype, "metronome")
            _sub(metronome, "beat-unit", "quarter")
            _sub(metronome, "per-minute", round(tempo))
            _sub(direction, "sound", tempo=str(round(tempo, 2)))

        for seg in segments:
            _write_segment(measure, seg, subdivision)

    ET.indent(root)
    return ET.ElementTree(root)


def write_musicxml(path: str | Path, events: list[Event], **kwargs) -> Path:
    path = Path(path)
    tree = to_musicxml(events, **kwargs)
    with open(path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        )
        tree.write(f, encoding="utf-8", xml_declaration=False)
    return path
