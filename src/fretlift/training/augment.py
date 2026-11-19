"""Audio augmentation: turn clean, labelled guitar into many other tones.

Labelled distorted guitar is scarce. Clean labelled guitar isn't, and running
it through an amp chain doesn't move a single note, so the labels stay valid.
The chain here is a simple amp model built from standard DSP blocks:

    pre-EQ -> waveshaping distortion (1 or 2 stages, oversampled) -> tone stack
           -> cabinet (a real impulse response if you have some, else a filter model)
           -> room reverb -> noise -> level

plus pitch shifting (labels shifted to match, strings kept), which imitates
down-tuned guitars.

It's written with NumPy/SciPy so it runs anywhere and is easy to read and test.
For more realistic tones, drop cabinet impulse responses (WAV files; many free
ones exist, check their licenses) into a folder and pass ``ir_dir``. Rendering
through real amp captures (e.g. Neural Amp Modeler) is a natural upgrade.

Listen to what it does before training on it::

    fretlift-data preview-augment some_clean_guitar.wav --out previews/
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..types import Note
from .labels import shift_pitch

log = logging.getLogger(__name__)

AMP_PRESETS = ("clean_amp", "crunch", "high_gain")


@dataclass
class AugmentConfig:
    p_amp: float = 0.7  # chance a clip goes through an amp at all
    preset_weights: dict[str, float] = field(default_factory=lambda: {"clean_amp": 0.2, "crunch": 0.35, "high_gain": 0.45})
    p_pitch_shift: float = 0.3
    pitch_shift_range: tuple[int, int] = (-4, 2)  # semitones; mostly down, like down-tuned guitars
    p_eq: float = 0.5
    p_reverb: float = 0.3
    p_noise: float = 0.5
    noise_snr_db: tuple[float, float] = (20.0, 50.0)
    gain_db: tuple[float, float] = (-18.0, 3.0)
    ir_dir: str | None = None
    p_ir: float = 0.8  # chance to use a real cab IR when ir_dir has some


# ------------------------------------------------------------------- filters


def _biquad(kind: str, f0: float, sr: int, gain_db: float = 0.0, q: float = 0.707) -> tuple[np.ndarray, np.ndarray]:
    """RBJ audio-EQ-cookbook biquads: 'peak', 'lowshelf', 'highshelf'."""
    f0 = min(f0, 0.45 * sr)
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind in ("lowshelf", "highshelf"):
        sq = 2 * np.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
            a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
            a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    else:
        raise ValueError(kind)
    b, a = np.asarray(b), np.asarray(a)
    return b / a[0], a / a[0]


def eq(x: np.ndarray, sr: int, kind: str, f0: float, gain_db: float = 0.0, q: float = 0.707) -> np.ndarray:
    from scipy.signal import lfilter

    b, a = _biquad(kind, f0, sr, gain_db, q)
    return lfilter(b, a, x)


def butter(x: np.ndarray, sr: int, kind: str, f0: float, order: int = 2) -> np.ndarray:
    import scipy.signal as ss

    return ss.sosfilt(ss.butter(order, min(f0, 0.45 * sr), btype=kind, fs=sr, output="sos"), x)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


# ----------------------------------------------------------------- amp model


def waveshape(x: np.ndarray, drive_db: float, bias: float = 0.0) -> np.ndarray:
    """tanh soft clipping at 2x oversampling (keeps aliasing down). ``bias`` adds even harmonics."""
    from scipy.signal import resample_poly

    g = 10 ** (drive_db / 20)
    up = resample_poly(x, 2, 1)
    y = np.tanh(g * up + bias) - np.tanh(bias)
    return resample_poly(y, 1, 2)[: len(x)]


def synthetic_cab(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A rough 4x12-style cabinet: rolled-off lows and highs, a presence bump."""
    y = butter(x, sr, "highpass", rng.uniform(60, 110))
    y = butter(y, sr, "lowpass", rng.uniform(3800, 6500), order=4)
    return eq(y, sr, "peak", rng.uniform(1500, 3200), rng.uniform(-3, 4), q=1.0)


def amp_chain(
    x: np.ndarray, sr: int, preset: str, rng: np.random.Generator, irs: Sequence[np.ndarray] = (), p_ir: float = 0.8
):
    """Run ``x`` through a randomized amp of the given character."""
    from scipy.signal import fftconvolve

    level = _rms(x)
    if level < 1e-6:
        return x
    y = x / level * 0.1  # drive amounts below assume this input level (RMS 0.1, peaks ~0.3)
    if preset == "clean_amp":
        y = waveshape(y, rng.uniform(0, 10), bias=rng.uniform(0, 0.2))  # barely breaking up
        y = eq(y, sr, "lowshelf", 250, rng.uniform(-2, 4))
        y = eq(y, sr, "highshelf", 3000, rng.uniform(-3, 3))
    elif preset == "crunch":
        y = butter(y, sr, "highpass", rng.uniform(50, 120))
        y = waveshape(y, rng.uniform(20, 30), bias=rng.uniform(0, 0.3))
        y = eq(y, sr, "peak", rng.uniform(500, 900), rng.uniform(-3, 4), q=0.8)
    elif preset == "high_gain":
        y = butter(y, sr, "highpass", rng.uniform(80, 160))  # "tight" low end before the gain
        y = eq(y, sr, "peak", rng.uniform(650, 1000), rng.uniform(4, 10), q=rng.uniform(0.6, 1.2))  # overdrive-style boost
        y = waveshape(y, rng.uniform(32, 42), bias=rng.uniform(0, 0.3))
        y = butter(y, sr, "highpass", 80)
        y = waveshape(y, rng.uniform(12, 22), bias=rng.uniform(0, 0.2))
        y = eq(y, sr, "peak", rng.uniform(350, 700), rng.uniform(-9, -3), q=0.7)  # scooped mids
        y = eq(y, sr, "peak", rng.uniform(2000, 4000), rng.uniform(0, 5), q=1.0)  # presence
    else:
        raise ValueError(f"Unknown preset {preset!r}; choose from {AMP_PRESETS}")

    if irs and rng.random() < p_ir:
        ir = irs[int(rng.integers(len(irs)))]
        y = fftconvolve(y, ir)[: len(y)]
    else:
        y = synthetic_cab(y, sr, rng)
    return y * (level / _rms(y))


def room_reverb(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Exponentially decaying noise as a room impulse response."""
    from scipy.signal import fftconvolve

    rt60 = rng.uniform(0.2, 1.2)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir = butter(ir, sr, "lowpass", rng.uniform(3000, 8000))
    ir /= np.sqrt(np.sum(ir**2)) + 1e-12
    tail = fftconvolve(x, ir)[: len(x)]
    wet = rng.uniform(0.08, 0.3)
    return (1 - wet) * x + wet * tail * (_rms(x) / _rms(tail))


def add_noise(x: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Hiss and hum, ``snr_db`` below the signal."""
    noise = butter(rng.standard_normal(len(x)), sr, "lowpass", rng.uniform(2000, 9000))
    if rng.random() < 0.3:  # mains hum
        hz = 50 if rng.random() < 0.5 else 60
        t = np.arange(len(x)) / sr
        noise = noise + 0.5 * sum(np.sin(2 * np.pi * hz * k * t) / k for k in (1, 2, 3))
    return x + noise * (_rms(x) / _rms(noise)) * 10 ** (-snr_db / 20)


def pitch_shift(x: np.ndarray, sr: int, semitones: int) -> np.ndarray | None:
    try:
        import librosa
    except ImportError:
        return None
    return librosa.effects.pitch_shift(x.astype(np.float32), sr=sr, n_steps=semitones).astype(np.float64)


def load_irs(ir_dir: str | Path | None, sr: int, max_seconds: float = 0.5) -> list[np.ndarray]:
    if not ir_dir:
        return []
    from .audio import AUDIO_SUFFIXES, load_audio

    irs = []
    for p in sorted(Path(ir_dir).iterdir()):
        if p.suffix.lower() in AUDIO_SUFFIXES:
            ir = load_audio(p, sr)[: int(max_seconds * sr)].astype(np.float64)
            if np.any(ir):
                irs.append(ir / (np.sqrt(np.sum(ir**2)) + 1e-12))
    log.info("Loaded %d cabinet impulse responses from %s", len(irs), ir_dir)
    return irs


# ----------------------------------------------------------------- augmenter


class Augmenter:
    """Random augmentation for training clips. Returns (audio, notes, description)."""

    def __init__(self, cfg: AugmentConfig | None = None, sr: int = 22050):
        self.cfg = cfg or AugmentConfig()
        self.sr = sr
        self.irs = load_irs(self.cfg.ir_dir, sr)
        self._warned = False

    def __call__(self, audio: np.ndarray, notes: list[Note], rng: np.random.Generator, amp: bool = True):
        """``amp=False`` skips the fake amp, for audio that went through a real one."""
        c, sr = self.cfg, self.sr
        x = np.asarray(audio, np.float64)
        steps: list[str] = []

        if rng.random() < c.p_pitch_shift:
            lo, hi = c.pitch_shift_range
            k = int(rng.choice([s for s in range(lo, hi + 1) if s != 0]))
            y = pitch_shift(x, sr, k)
            if y is None:
                if not self._warned:
                    log.warning("librosa isn't installed, so pitch-shift augmentation is off")
                    self._warned = True
            else:
                x = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
                notes = shift_pitch(notes, k)
                steps.append(f"pitch{k:+d}")
        if rng.random() < c.p_eq:
            x = eq(x, sr, "lowshelf", 200, rng.uniform(-5, 5))
            x = eq(x, sr, "highshelf", 3000, rng.uniform(-6, 4))
            steps.append("eq")
        if amp and rng.random() < c.p_amp:
            names = list(c.preset_weights)
            w = np.array([c.preset_weights[n] for n in names], float)
            preset = names[int(rng.choice(len(names), p=w / w.sum()))]
            x = amp_chain(x, sr, preset, rng, self.irs, c.p_ir)
            steps.append(preset)
        if rng.random() < c.p_reverb:
            x = room_reverb(x, sr, rng)
            steps.append("reverb")
        if rng.random() < c.p_noise:
            snr = rng.uniform(*c.noise_snr_db)
            x = add_noise(x, sr, snr, rng)
            steps.append(f"noise{snr:.0f}dB")
        g = rng.uniform(*c.gain_db)
        x = x * 10 ** (g / 20)
        peak = np.max(np.abs(x)) if len(x) else 0.0
        if peak > 1.0:
            x = x / peak
        return x.astype(np.float32), notes, "+".join(steps) or "none"


def apply_preset(audio: np.ndarray, sr: int, preset: str, seed: int, irs: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Deterministic distortion for building fixed evaluation sets (labels unchanged)."""
    if preset in (None, "", "none"):
        return np.asarray(audio, np.float32)
    rng = np.random.default_rng(seed)
    x = amp_chain(np.asarray(audio, np.float64), sr, preset, rng, irs, p_ir=1.0 if irs else 0.0)
    x = add_noise(x, sr, 45.0, rng)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    return (x / peak * 0.9 if peak > 0 else x).astype(np.float32)
