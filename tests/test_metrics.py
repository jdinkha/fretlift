from fretlift.metrics import match_notes, note_score, string_accuracy, tab_score
from fretlift.types import Note


def N(start, pitch, string=None, dur=0.5):
    return Note(start, start + dur, pitch, string=string)


def test_perfect_and_empty():
    ref = [N(0, 60), N(1, 62)]
    assert note_score(ref, ref).f1 == 1.0
    assert note_score([], []).f1 == 1.0
    s = note_score(ref, [])
    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)


def test_onset_tolerance_is_inclusive_at_50ms():
    assert note_score([N(1.0, 60)], [N(1.05, 60)]).f1 == 1.0
    assert note_score([N(1.0, 60)], [N(1.051, 60)]).f1 == 0.0


def test_wrong_pitch_doesnt_match():
    assert note_score([N(0, 60)], [N(0, 61)]).f1 == 0.0


def test_each_note_matches_once():
    s = note_score([N(0, 60)], [N(0, 60), N(0.01, 60)])
    assert (s.n_match, s.precision, s.recall) == (1, 0.5, 1.0)


def test_matching_is_maximum_not_greedy():
    # Greedy (first ref takes nearest est) would match ref0-est1 and strand ref1.
    ref = [N(0.00, 60), N(0.06, 60)]
    est = [N(0.03, 60), N(0.01, 60)]
    assert len(match_notes(ref, est)) == 2


def test_long_chain_of_repeated_notes():
    # Dense repeated notes: augmenting paths get long; must not recurse/blow up.
    ref = [N(i * 0.04, 60, dur=0.03) for i in range(3000)]
    est = [N(i * 0.04 + 0.02, 60, dur=0.03) for i in range(3000)]
    assert note_score(ref, est).f1 == 1.0


def test_offset_criterion():
    ref = [Note(0, 1.0, 60)]
    assert note_score(ref, [Note(0, 1.15, 60)], offset_ratio=0.2).f1 == 1.0  # within 20% of 1 s
    assert note_score(ref, [Note(0, 1.5, 60)], offset_ratio=0.2).f1 == 0.0
    assert note_score(ref, [Note(0, 1.5, 60)]).f1 == 1.0  # offsets ignored by default


def test_tab_score_needs_right_string():
    ref = [N(0, 64, string=1), N(1, 64, string=2)]
    est = [N(0, 64, string=1), N(1, 64, string=1)]
    s = tab_score(ref, est)
    assert (s.n_match, s.f1) == (1, 0.5)
    assert note_score(ref, est).f1 == 1.0
    assert string_accuracy(ref, est) == (1, 2)


def test_tab_score_undefined_without_string_labels():
    assert tab_score([N(0, 60)], [N(0, 60, string=2)]) is None
