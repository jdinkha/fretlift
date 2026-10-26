import pytest

from fretlift.fretting import FretConfig, assign_frets, candidates
from fretlift.tuning import PRESETS
from fretlift.types import Event


def _frets_by_string(event):
    return {t.string: t.fret for t in event.tab}


def test_low_e_is_open_sixth_string():
    (e,) = assign_frets([Event(0, 4, [40])])
    assert _frets_by_string(e) == {6: 0}


def test_open_e_major_chord():
    (e,) = assign_frets([Event(0, 4, [40, 47, 52, 56, 59, 64])])
    assert _frets_by_string(e) == {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0}


def test_each_note_gets_its_own_string():
    for c in candidates([45, 50, 55, 59], FretConfig()):
        strings = [t.string for t in c.notes]
        assert len(strings) == len(set(strings))


def test_out_of_range_pitches_are_dropped():
    (e,) = assign_frets([Event(0, 4, [20, 40])])  # MIDI 20 is below any string
    assert [t.pitch for t in e.tab] == [40]
    assert assign_frets([Event(0, 4, [20])]) == []


def test_frets_go_up_to_24_and_no_higher():
    (e,) = assign_frets([Event(0, 4, [88])])  # E6 = 24th fret of the high e string
    assert _frets_by_string(e) == {1: 24}
    assert assign_frets([Event(0, 4, [89])]) == []
    with pytest.raises(ValueError):
        FretConfig(max_fret=25)


def test_capo_makes_frets_relative():
    (e,) = assign_frets([Event(0, 4, [42])], FretConfig(capo=2))  # F#2 = open low E with capo 2
    assert _frets_by_string(e) == {6: 0}


def test_drop_d_low_note():
    (e,) = assign_frets([Event(0, 4, [38])], FretConfig(tuning=PRESETS["drop-d"]))
    assert _frets_by_string(e) == {6: 0}


def test_melody_stays_in_one_position():
    # A one-octave C-major scale from C4 fits in a single four-fret box (5th
    # position) without the hand creeping up the neck note by note.
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    events = [Event(i, 1, [p]) for i, p in enumerate(scale)]
    frets = [e.tab[0].fret for e in assign_frets(events)]
    fretted = [f for f in frets if f > 0]
    assert max(fretted) - min(fretted) <= 3, frets


def test_high_melody_moves_up_the_neck():
    # Above fret 12 of the high e string there's only one option per note.
    events = [Event(i, 1, [p]) for i, p in enumerate([76, 77, 79])]
    assert [e.tab[0].fret for e in assign_frets(events)] == [12, 13, 15]
