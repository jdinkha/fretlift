import numpy as np

from fretlift.training.config import DecodeConfig
from fretlift.training.decode import decode_notes
from fretlift.tuning import PRESETS

FPS = 100.0  # round numbers for the test
STD = PRESETS["standard"]


def blank(T=200, P=56):
    return np.zeros((T, P), np.float32), np.zeros((T, P), np.float32)


def test_onset_then_sustain():
    frame, onset = blank()
    onset[10, 7] = 0.9
    frame[10:40, 7] = 0.8
    (n,) = decode_notes(frame, onset, fps=FPS, pitch_min=33)
    assert n.pitch == 40 and n.start == 0.10 and n.end == 0.40


def test_repeated_notes_split_on_new_onset():
    frame, onset = blank()
    frame[10:80, 7] = 0.9  # string rings throughout
    onset[10, 7] = onset[45, 7] = 0.9
    notes = decode_notes(frame, onset, fps=FPS, pitch_min=33)
    assert [(n.start, n.end) for n in notes] == [(0.10, 0.45), (0.45, 0.80)]


def test_thresholds_and_min_length():
    frame, onset = blank()
    onset[10, 7] = 0.4  # below default onset threshold
    frame[10:40, 7] = 0.9
    onset[100, 9] = 0.9
    frame[100:102, 9] = 0.9  # too short
    assert decode_notes(frame, onset, fps=FPS, pitch_min=33) == []
    low = DecodeConfig(onset_threshold=0.3)
    assert len(decode_notes(frame, onset, fps=FPS, pitch_min=33, cfg=low)) == 1


def test_plateau_and_close_peaks_give_one_note():
    frame, onset = blank()
    onset[10:12, 7] = 0.8  # plateau
    onset[13, 7] = 0.85  # second peak 3 frames later counts, 1-2 frames later doesn't
    onset[11, 7] = 0.8
    frame[10:60, 7] = 0.9
    notes = decode_notes(frame, onset, fps=FPS, pitch_min=33, cfg=DecodeConfig(min_gap_frames=4))
    assert len(notes) == 1 and notes[0].start == 0.10


def test_string_assignment_respects_tuning_and_chords():
    frame, onset = blank()
    S = np.full((200, 56, 6), 1 / 6, np.float32)
    # A chord: E2 (40) and B2 (47). The model thinks both are most likely on string 6.
    for p in (40, 47):
        onset[10, p - 33] = 0.9
        frame[10:50, p - 33] = 0.9
    S[:, 40 - 33] = [0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
    S[:, 47 - 33] = [0.0, 0.0, 0.0, 0.0, 0.3, 0.7]
    notes = decode_notes(frame, onset, fps=FPS, pitch_min=33, string_prob=S, tuning=STD)
    by_pitch = {n.pitch: n.string for n in notes}
    # E2 can only be string 6 in standard tuning, so B2 has to move to string 5.
    assert by_pitch == {40: 6, 47: 5}


def test_unplayable_pitch_gets_no_string():
    frame, onset = blank()
    onset[10, 0] = 0.9  # A1 is below standard tuning's low E
    frame[10:50, 0] = 0.9
    S = np.full((200, 56, 6), 1 / 6, np.float32)
    (n,) = decode_notes(frame, onset, fps=FPS, pitch_min=33, string_prob=S, tuning=STD)
    assert n.pitch == 33 and n.string is None
    (n,) = decode_notes(frame, onset, fps=FPS, pitch_min=33, string_prob=S, tuning=(33, 38, 43, 48, 52, 57))
    assert n.string == 6
