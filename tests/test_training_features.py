import numpy as np
import pytest

from fretlift.training.features import cqt_kernel_groups, cqt_reference, harmonic_shifts, midi_to_hz

SR, HOP = 22050, 256


@pytest.mark.parametrize("midi", [40, 52, 64, 76, 88])
def test_cqt_peaks_at_the_right_semitone(midi):
    t = np.arange(SR // 2) / SR
    C = cqt_reference(np.sin(2 * np.pi * midi_to_hz(midi) * t), SR, HOP, midi_to_hz(33), 168, 24)
    assert C.shape == (168, (len(t) - 1) // HOP + 1)
    assert C[:, C.shape[1] // 2].argmax() == 2 * (midi - 33)


def test_cqt_kernel_groups():
    groups = cqt_kernel_groups(SR, midi_to_hz(33), 168, 24)
    assert [g.shape[0] for g in groups] == [24] * 7
    assert all(g.shape[1] % 2 == 1 for g in groups)
    assert groups[0].shape[1] > groups[-1].shape[1] * 32  # low octaves need longer windows
    with pytest.raises(ValueError):
        cqt_kernel_groups(SR, midi_to_hz(33), 24 * 9, 24)  # above Nyquist


def test_harmonic_shifts():
    assert harmonic_shifts((0.5, 1, 2, 3, 4), 24) == [-24, 0, 24, 38, 48]
