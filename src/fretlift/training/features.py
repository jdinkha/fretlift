"""Constant-Q transform kernels and harmonic stacking (NumPy).

The CQT has bins spaced evenly in pitch (here 2 per semitone), which suits
music far better than a plain spectrogram: a semitone is the same number of
bins in every octave, so a convolution that learns "what a note looks like"
works at any pitch.

The kernels are built here in NumPy and applied as a strided 1-D convolution
in PyTorch (``model.CQT``), so the transform runs on the GPU inside the model
and training and inference always use exactly the same features.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def cqt_frequencies(fmin: float, n_bins: int, bins_per_octave: int) -> np.ndarray:
    return fmin * 2 ** (np.arange(n_bins) / bins_per_octave)


def cqt_kernel_groups(
    sr: int, fmin: float, n_bins: int, bins_per_octave: int, filter_scale: float = 1.0
) -> list[np.ndarray]:
    """Complex CQT kernels, one array [bins, K] per octave.

    Low notes need long windows to tell semitones apart, high notes don't, so
    each octave gets its own kernel length instead of padding everything to
    the longest one (about 4x less compute).
    """
    if fmin * 2 ** ((n_bins - 1) / bins_per_octave) >= sr / 2:
        raise ValueError("The highest CQT bin is above the Nyquist frequency")
    q = filter_scale / (2 ** (1 / bins_per_octave) - 1)
    freqs = cqt_frequencies(fmin, n_bins, bins_per_octave)
    lengths = [int(np.ceil(q * sr / f)) | 1 for f in freqs]  # odd, so each kernel has a centre sample
    groups = []
    for start in range(0, n_bins, bins_per_octave):
        idx = range(start, min(start + bins_per_octave, n_bins))
        K = max(lengths[i] for i in idx)
        kern = np.zeros((len(idx), K), np.complex64)
        for row, i in enumerate(idx):
            n = lengths[i]
            t = np.arange(n) - (n - 1) / 2
            win = np.hanning(n)
            off = K // 2 - (n - 1) // 2
            kern[row, off : off + n] = win * np.exp(2j * np.pi * freqs[i] * t / sr) / win.sum()
        groups.append(kern)
    return groups


def cqt_reference(x: np.ndarray, sr: int, hop: int, fmin: float, n_bins: int, bins_per_octave: int) -> np.ndarray:
    """Slow NumPy CQT magnitude [bins, frames], matching ``model.CQT``. For tests."""
    rows = []
    n_frames = (len(x) - 1) // hop + 1
    for kern in cqt_kernel_groups(sr, fmin, n_bins, bins_per_octave):
        K = kern.shape[1]
        xp = np.pad(x, (K // 2, K // 2))
        frames = np.stack([xp[t * hop : t * hop + K] for t in range(n_frames)], axis=1)
        rows.append(np.abs(kern @ frames))
    return np.concatenate(rows, axis=0)


def harmonic_shifts(harmonics: Sequence[float], bins_per_octave: int) -> list[int]:
    """CQT bin offset of each harmonic relative to the fundamental.

    Shifting the CQT by these amounts and stacking the results as channels
    puts the 2nd, 3rd, ... harmonics of a note directly "above" its
    fundamental, so a small convolution can see the whole harmonic series at
    once. That matters a lot for distorted guitar, where the harmonics are
    louder than the fundamental.
    """
    return [int(round(bins_per_octave * np.log2(h))) for h in harmonics]
