"""Isolate the guitar from a full mix with Demucs."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

GUITAR_MODELS = ("htdemucs_6s",)  # the Demucs models that have a guitar stem


def separate_guitar(
    audio_path: Path, work_dir: Path, model: str = "htdemucs_6s", device: str | None = None, shifts: int = 1
) -> Path:
    """Run Demucs and return the path of the separated guitar stem."""
    if model not in GUITAR_MODELS:
        raise ValueError(f"{model!r} has no guitar stem; use one of {GUITAR_MODELS}")
    try:
        import demucs.separate
    except ImportError as e:
        if e.name != "demucs":
            raise  # installed but broken: show the real error
        raise RuntimeError(
            "Demucs isn't installed. Run `pip install -e .[ml]`, or pass --no-separate "
            "if your file is already a solo guitar recording."
        ) from e

    args = ["-n", model, "--two-stems", "guitar", "--shifts", str(shifts), "-o", str(work_dir), str(audio_path)]
    if device:
        args += ["-d", device]
    log.info("Separating guitar with %s (this is the slow step on CPU)", model)
    demucs.separate.main(args)

    stem = work_dir / model / audio_path.stem / "guitar.wav"
    if not stem.exists():
        raise RuntimeError(f"Demucs finished but {stem} wasn't created")
    return stem


def pick_channel(audio_path: Path, work_dir: Path, channel: str = "mix", remove_center: bool = False) -> Path:
    """Return ``audio_path`` as-is ("mix"), or one stereo channel written to its own WAV.

    Metal often has two rhythm guitars panned hard left and right. Summed to mono
    they smear each other's notes, so tabbing each side on its own helps.
    ``remove_center`` also strips what both sides share, such as bass bleed.
    """
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        if channel != "mix":
            log.warning("%s isn't stereo, so --channel %s is ignored", audio_path.name, channel)
        return audio_path

    left, right = audio.T
    similarity = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
    log.info("Left/right similarity %.2f (1 = identical, 0 = unrelated)", similarity)
    if channel == "mix":
        # ponytail: one number for the whole song and a fixed threshold; per-section analysis if that misleads
        if similarity < 0.5:
            log.warning(
                "The left and right channels differ a lot, so there are probably two guitars panned apart. "
                "Try --channel both to tab each one."
            )
        return audio_path

    side = left if channel == "left" else right
    if remove_center:
        # Centred parts (usually bass, kick, vocal bleed) are identical and in phase on both sides;
        # hard-panned guitars aren't. Turn each time-frequency bin down by how centred it is.
        import librosa

        n_fft = 4096  # ponytail: fixed window, ~10 Hz bins at 44.1 kHz; tune if low notes still bleed
        L, R = librosa.stft(left, n_fft=n_fft), librosa.stft(right, n_fft=n_fft)
        centred = 2 * np.real(L * np.conj(R)) / (np.abs(L) ** 2 + np.abs(R) ** 2 + 1e-9)
        keep = 1 - np.clip(centred, 0, 1)
        side = librosa.istft((L if channel == "left" else R) * keep, n_fft=n_fft, length=len(side))

    out = work_dir / f"{audio_path.stem}.{channel}.wav"
    sf.write(str(out), side, sr)
    return out
