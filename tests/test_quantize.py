from fretlift.quantize import quantize
from fretlift.types import BeatGrid, Note


def grid(bpm=120.0, n=32, start=0.0):
    period = 60.0 / bpm
    return BeatGrid(tempo=bpm, beats=[start + i * period for i in range(n)])


def test_quarter_notes_land_on_beats():
    notes = [Note(i * 0.5 + 0.01, i * 0.5 + 0.45, 60 + i) for i in range(4)]
    events = quantize(notes, grid(), subdivision=4)
    assert [e.tick for e in events] == [0, 4, 8, 12]
    assert [e.duration for e in events] == [4, 4, 4, 4]


def test_simultaneous_notes_become_a_chord():
    notes = [Note(0.0, 1.0, 40), Note(0.012, 1.0, 47), Note(0.005, 1.0, 52)]
    (e,) = quantize(notes, grid())
    assert e.pitches == [40, 47, 52]


def test_short_note_leaves_rest():
    notes = [Note(0.0, 0.12, 60), Note(1.0, 1.4, 62)]  # 16th note, then a gap
    events = quantize(notes, grid())
    assert events[0].duration == 1
    assert events[1].tick == 8


def test_notes_before_first_detected_beat():
    # The beat tracker missed the pickup; the grid is extrapolated backwards.
    notes = [Note(0.0, 0.4, 60), Note(0.5, 0.9, 62)]
    events = quantize(notes, grid(start=0.5))
    assert [e.tick for e in events] == [0, 4]


def test_empty():
    assert quantize([], grid()) == []
