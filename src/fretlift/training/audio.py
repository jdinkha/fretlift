"""Audio loading, resampling and an on-disk cache."""

from __future__ import annotations

import hashlib
import os
import wave
from math import gcd
from pathlib import Path

import numpy as np

from .config import SR

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a")


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _read_wav_stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        raise RuntimeError(f"{path}: {8 * width}-bit WAV needs soundfile installed")
    return x.reshape(-1, ch).mean(axis=1), sr


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Mono float32 audio at ``sr``."""
    path = Path(path)
    try:
        import soundfile as sf

        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = data.mean(axis=1)
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(f'Reading {path.suffix} files needs soundfile: pip install -e ".[train]"') from None
        x, file_sr = _read_wav_stdlib(path)
    except Exception:
        # Formats libsndfile can't read (e.g. m4a): fall back to librosa/audioread.
        import librosa

        x, _ = librosa.load(str(path), sr=sr, mono=True)
        return x.astype(np.float32)
    return _resample(np.ascontiguousarray(x, np.float32), int(file_sr), sr)


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    """16-bit PCM WAV, using only the standard library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(audio, np.float64), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


class AudioCache:
    """Decoded, resampled audio stored as .npy files and memory-mapped.

    Decoding and resampling every clip on every step would dominate training
    time; this does it once per file. Delete the folder to rebuild it.
    """

    def __init__(self, root: str | Path, sr: int = SR):
        self.root = Path(root)
        self.sr = sr
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, audio_path: str | Path) -> Path:
        p = Path(audio_path).resolve()
        st = p.stat()
        key = hashlib.md5(f"{p}:{st.st_mtime_ns}:{st.st_size}:{self.sr}".encode()).hexdigest()
        return self.root / f"{key}.npy"

    def ensure(self, audio_path: str | Path) -> Path:
        cached = self.path_for(audio_path)
        if not cached.exists():
            tmp = cached.with_name(f"{cached.stem}.{os.getpid()}.tmp.npy")
            np.save(tmp, load_audio(audio_path, self.sr))
            os.replace(tmp, cached)
        return cached

    def load(self, audio_path: str | Path, mmap: bool = True) -> np.ndarray:
        return np.load(self.ensure(audio_path), mmap_mode="r" if mmap else None)
