"""Shared constants and configs for audio, labels and the model.

Anything that has to agree between the dataset, the model and decoding (sample
rate, hop size, pitch range) lives here, and is saved in every checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SR = 22050  # Hz; plenty for guitar, and halves the compute vs 44.1 kHz
HOP = 256  # samples per frame -> ~11.6 ms frames
FPS = SR / HOP  # ~86.1 frames per second
PITCH_MIN = 33  # A1: covers 7-strings and drop tunings well below standard E2 (40)
PITCH_MAX = 88  # E6: 24th fret of the high e string
N_STRINGS = 6
IGNORE = -100  # target value for "no string label here" (PyTorch's default ignore_index)


def n_frames(n_samples: int, hop: int = HOP) -> int:
    """Frames produced for ``n_samples`` of audio; frame t is centred on sample t * hop."""
    return (n_samples - 1) // hop + 1 if n_samples > 0 else 0


@dataclass
class ModelConfig:
    sr: int = SR
    hop: int = HOP
    pitch_min: int = PITCH_MIN
    pitch_max: int = PITCH_MAX
    n_strings: int = N_STRINGS
    # CQT front end: 2 bins per semitone, starting at pitch_min
    bins_per_octave: int = 24
    n_octaves: int = 7
    filter_scale: float = 1.0
    # Harmonic stacking (as in Basic Pitch): align harmonics with the fundamental
    harmonics: tuple[float, ...] = (0.5, 1, 2, 3, 4, 5, 6, 7)
    channels: int = 32
    hidden: int = 256
    dropout: float = 0.2

    @property
    def n_pitches(self) -> int:
        return self.pitch_max - self.pitch_min + 1

    @property
    def fps(self) -> float:
        return self.sr / self.hop

    def to_dict(self) -> dict:
        d = asdict(self)
        d["harmonics"] = list(self.harmonics)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        d = dict(d)
        if "harmonics" in d:
            d["harmonics"] = tuple(d["harmonics"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DecodeConfig:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_frames: int = 5  # ~58 ms, same as the Phase 1 CLI default
    min_gap_frames: int = 3  # two onsets of the same pitch closer than ~35 ms are one note
    chord_frames: int = 3  # onsets this close are one chord when assigning strings
