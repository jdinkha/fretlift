"""Everything after the ML models, on the answer key of the bundled sample.

This checks quantize -> fret assignment -> export on perfect notes, so any error
in a real run can be pinned on transcription or beat tracking.
"""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

from fretlift.export import to_ascii, to_musicxml
from fretlift.fretting import assign_frets
from fretlift.quantize import quantize
from fretlift.tuning import PRESETS
from fretlift.types import BeatGrid, Note

ROOT = Path(__file__).resolve().parents[1]


def _load_sample_module():
    spec = importlib.util.spec_from_file_location("make_sample", ROOT / "scripts" / "make_sample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_answer_key_round_trips_to_a_tab():
    sample = _load_sample_module()
    bpm, lead_in = 100.0, 0.5
    # Use the answer key without rendering audio (fast).
    t, notes = lead_in, []
    for pitch, beats in sample.MELODY:
        notes.append(Note(t, t + beats * 60 / bpm * 0.95, pitch))
        t += beats * 60 / bpm
    grid = BeatGrid(bpm, [lead_in + i * 60 / bpm for i in range(40)])

    events = assign_frets(quantize(notes, grid, 4))
    assert [e.pitches[0] for e in events] == [p for p, _ in sample.MELODY]
    assert len(events) == len(sample.MELODY)
    # 32 beats of 4/4 -> 8 measures.
    root = to_musicxml(events, tuning=PRESETS["standard"], tempo=bpm).getroot()
    assert len(root.findall(".//measure")) == 8
    # The dotted-quarter + eighth rhythm survives quantization.
    types = [(n.find("type").text, n.find("dot") is not None) for n in root.findall(".//note")]
    assert ("quarter", True) in types and ("eighth", False) in types
    ET.fromstring(ET.tostring(root))
    assert "|" in to_ascii(events, tuning=PRESETS["standard"])
