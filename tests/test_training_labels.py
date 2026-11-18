import json

import numpy as np
import pytest

from fretlift.training.config import FPS, IGNORE, PITCH_MIN
from fretlift.training.labels import (
    load_labels,
    make_targets,
    parse_jams,
    parse_midi,
    save_labels,
    shift_pitch,
    window_notes,
)
from fretlift.types import Note


def jams_doc(per_string, dense=False):
    """GuitarSet-like JAMS: one note_midi annotation per string, data_source 0 = low E."""
    anns = [{"namespace": "beat_position", "annotation_metadata": {}, "data": []}]
    for idx, notes in enumerate(per_string):
        if dense:
            data = {"time": [t for t, _, _ in notes], "duration": [d for _, d, _ in notes],
                    "value": [v for _, _, v in notes], "confidence": [None] * len(notes)}
        else:
            data = [{"time": t, "duration": d, "value": v, "confidence": None} for t, d, v in notes]
        anns.append({"namespace": "note_midi", "annotation_metadata": {"data_source": str(idx)}, "data": data})
    return {"annotations": anns}


def test_parse_jams_strings_and_rounding():
    per_string = [[(0.5, 0.4, 40.2)], [], [], [], [], [(1.0, 0.5, 63.8)]]
    for dense in (False, True):
        notes = parse_jams(jams_doc(per_string, dense=dense))
        assert [(n.pitch, n.string) for n in notes] == [(40, 6), (64, 1)]
        assert notes[0].end == 0.9


def test_label_round_trip_and_phase1_notes(tmp_path):
    notes = [Note(0.1, 0.5, 64, string=1), Note(0.2, 0.6, 40)]
    p = save_labels(tmp_path / "x.json", notes, tuning=(40, 45, 50, 55, 59, 64), source="t")
    loaded, tuning = load_labels(p)
    assert [(n.pitch, n.string) for n in loaded] == [(64, 1), (40, None)]
    assert tuning == (40, 45, 50, 55, 59, 64)
    # `fretlift --save-notes` output loads too
    saved = {"tempo": 100, "notes": [{"start": 0, "end": 1, "pitch": 60, "velocity": 0.5}]}
    (tmp_path / "s.notes.json").write_text(json.dumps(saved))
    loaded, _ = load_labels(tmp_path / "s.notes.json")
    assert loaded[0].pitch == 60 and loaded[0].string is None


def test_make_targets():
    notes = [Note(0.1, 0.3, 64, string=1), Note(-0.2, 0.2, 40, string=6), Note(0.0, 0.1, 20)]
    frame, onset, string = make_targets(notes, 50, FPS, PITCH_MIN, 56)
    p64, p40 = 64 - PITCH_MIN, 40 - PITCH_MIN
    on = round(0.1 * FPS)
    assert frame[on, p64] == 1 and frame[on - 1, p64] == 0
    assert onset[on, p64] == 1 and onset[on + 1, p64] == 1 and onset[on + 2, p64] == 0
    assert string[on, p64] == 0  # string 1 -> class 0
    # started before the clip: sounding from frame 0, but no onset
    assert frame[0, p40] == 1 and onset[:, p40].sum() == 0 and string[0, p40] == 5
    assert frame.sum(axis=0)[np.arange(56) != p64].sum() == frame[:, p40].sum()  # out-of-range pitch 20 ignored
    assert (string[frame == 0] == IGNORE).all()


def test_window_and_shift():
    notes = [Note(0.5, 1.5, 60, string=2), Note(3.0, 3.5, 62)]
    w = window_notes(notes, 1.0, 2.0)
    assert len(w) == 1 and w[0].start == -0.5 and w[0].end == 0.5
    s = shift_pitch(notes, -2)
    assert [n.pitch for n in s] == [58, 60] and s[0].string == 2


def write_string_midi(path, notes_by_name):
    """A MIDI file with one track per string, named the way Guitar-TECHS names them."""
    pretty_midi = pytest.importorskip("pretty_midi")
    pm = pretty_midi.PrettyMIDI()
    for name, pitches in notes_by_name.items():
        inst = pretty_midi.Instrument(0, name=name)
        inst.notes = [pretty_midi.Note(100, p, 0.1 * k, 0.1 * k + 0.5) for k, p in enumerate(pitches)]
        pm.instruments.append(inst)
    pm.write(str(path))
    return path


def test_parse_midi_strings_by_track_name(tmp_path):
    # Only three strings play, and the B string plays higher than the G string, so
    # neither the six-part rule nor ranking by pitch would label these right.
    midi = write_string_midi(tmp_path / "a.mid", {"B": [67], "G": [60], "E": [40]})
    assert {(n.pitch, n.string) for n in parse_midi(midi)} == {(67, 2), (60, 3), (40, 6)}
    assert all(n.string is None for n in parse_midi(midi, strings="none"))
