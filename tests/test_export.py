import xml.etree.ElementTree as ET

from fretlift.export import to_ascii, to_musicxml, write_musicxml
from fretlift.export.musicxml import split_duration
from fretlift.fretting import assign_frets
from fretlift.measures import layout
from fretlift.tuning import PRESETS
from fretlift.types import Event

STD = PRESETS["standard"]


def sample_events():
    return assign_frets(
        [
            Event(0, 4, [40, 47, 52, 56, 59, 64]),  # E major, quarter
            Event(4, 2, [64]),
            Event(6, 2, [67]),
            Event(12, 8, [60]),  # crosses the barline -> tied
        ]
    )


def test_split_duration():
    assert split_duration(4, 4) == [(4, "quarter", 0)]
    assert split_duration(6, 4) == [(6, "quarter", 1)]
    assert split_duration(5, 4) == [(4, "quarter", 0), (1, "16th", 0)]
    assert split_duration(16, 4) == [(16, "whole", 0)]


def test_layout_fills_measures_and_ties_across_barlines():
    measures = layout(sample_events(), (4, 4), 4)
    assert len(measures) == 2
    for m in measures:
        assert sum(s.duration for s in m) == 16
    last_in_first = measures[0][-1]
    first_in_second = measures[1][0]
    assert last_in_first.tie_start and first_in_second.tie_stop


def test_musicxml_has_tab_staff_and_positions():
    root = to_musicxml(sample_events(), tuning=STD, tempo=96).getroot()
    assert root.find(".//clef/sign").text == "TAB"
    assert root.find(".//staff-details/staff-lines").text == "6"
    first = root.find(".//note")
    assert first.find(".//technical/string").text == "6"
    assert first.find(".//technical/fret").text == "0"
    chords = root.findall(".//note/chord")
    assert len(chords) == 5  # six-note chord = 1 note + 5 <chord/> notes
    assert root.find(".//sound").get("tempo") == "96"
    # Every measure adds up to a full 4/4 bar (chord notes don't add time).
    for measure in root.findall(".//measure"):
        total = sum(int(n.find("duration").text) for n in measure.findall("note") if n.find("chord") is None)
        assert total == 16


def test_write_musicxml_is_parseable(tmp_path):
    path = write_musicxml(tmp_path / "x.musicxml", sample_events(), tuning=STD)
    assert ET.parse(path).getroot().tag == "score-partwise"


def test_ascii_tab():
    text = to_ascii(sample_events(), tuning=STD, title="Demo")
    lines = text.splitlines()
    assert lines[0] == "Demo"
    tab_lines = [ln for ln in lines if "|" in ln]
    assert tab_lines[0].startswith("e|0")  # high e, open, first beat
    assert tab_lines[5].startswith("E|0")  # low E, open
