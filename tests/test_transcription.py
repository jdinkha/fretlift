from fretlift.transcription import drop_ghost_harmonics
from fretlift.types import Note


def test_quiet_octave_ghost_dropped_but_played_octave_kept():
    notes = [
        Note(0.50, 0.9, 64, 0.78),
        Note(0.50, 0.9, 76, 0.36),  # ghost: octave up, half as loud
        Note(1.00, 1.4, 52, 0.80),
        Note(1.01, 1.4, 64, 0.75),  # real octave: about as loud
    ]
    assert [n.pitch for n in drop_ghost_harmonics(notes)] == [64, 52, 64]
