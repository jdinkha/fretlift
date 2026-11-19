import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from fretlift.training.augment import AMP_PRESETS, AugmentConfig, Augmenter, amp_chain, apply_preset  # noqa: E402
from fretlift.training.features import midi_to_hz  # noqa: E402
from fretlift.types import Note  # noqa: E402

SR = 22050


def guitar_like(seconds=2.0, midi=40):
    t = np.arange(int(seconds * SR)) / SR
    f = midi_to_hz(midi)
    x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6))
    return (0.2 * x * np.exp(-2 * (t % 1.0))).astype(np.float32)


@pytest.mark.parametrize("preset", AMP_PRESETS)
def test_amp_chain_keeps_length_and_level(preset):
    x = guitar_like()
    y = amp_chain(x.astype(float), SR, preset, np.random.default_rng(0))
    assert y.shape == x.shape and np.isfinite(y).all()
    assert np.sqrt(np.mean(y**2)) == pytest.approx(np.sqrt(np.mean(x.astype(float) ** 2)), rel=1e-3)


def test_high_gain_adds_harmonics():
    t = np.arange(SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 110 * t)
    y = amp_chain(x, SR, "high_gain", np.random.default_rng(0))

    def harmonic_energy_ratio(s):
        spec = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), 1 / SR)
        fund = spec[np.abs(freqs - 110) < 3].sum()
        upper = sum(spec[np.abs(freqs - 110 * k) < 3].sum() for k in range(2, 10))
        return upper / fund

    assert harmonic_energy_ratio(y) > 10 * harmonic_energy_ratio(x) + 0.5


def test_silence_stays_silent():
    y = amp_chain(np.zeros(SR), SR, "high_gain", np.random.default_rng(0))
    assert np.all(y == 0)


def test_augmenter_is_seeded_and_shifts_labels():
    pytest.importorskip("librosa")
    cfg = AugmentConfig(p_pitch_shift=1.0, pitch_shift_range=(-2, -2), p_amp=1.0)
    aug = Augmenter(cfg, SR)
    x = guitar_like()
    notes = [Note(0.0, 1.0, 40, string=6)]
    y1, n1, d1 = aug(x, notes, np.random.default_rng(1))
    y2, n2, d2 = aug(x, notes, np.random.default_rng(1))
    assert np.array_equal(y1, y2) and d1 == d2
    assert n1[0].pitch == 38 and n1[0].string == 6  # same string, guitar "tuned down"
    assert y1.shape == x.shape and y1.dtype == np.float32


def test_augmenter_without_pitch_shift_keeps_labels():
    aug = Augmenter(AugmentConfig(p_pitch_shift=0.0), SR)
    notes = [Note(0.0, 1.0, 40, string=6)]
    for seed in range(5):
        y, n, _ = aug(guitar_like(), notes, np.random.default_rng(seed))
        assert n == notes and np.abs(y).max() <= 1.0


def test_augmenter_skips_the_amp_for_audio_that_had_one():
    aug = Augmenter(AugmentConfig(p_amp=1.0), SR)
    for seed in range(5):
        _, _, desc = aug(guitar_like(), [], np.random.default_rng(seed), amp=False)
        assert not set(desc.split("+")) & (set(AMP_PRESETS) - {"none"})


def test_apply_preset_is_deterministic():
    x = guitar_like()
    assert np.array_equal(apply_preset(x, SR, "crunch", 3), apply_preset(x, SR, "crunch", 3))
    assert not np.array_equal(apply_preset(x, SR, "crunch", 3), apply_preset(x, SR, "crunch", 4))
    assert np.array_equal(apply_preset(x, SR, "none", 3), x)
