"""The transcription model: audio in, notes and strings out.

    waveform ─► CQT (2 bins/semitone) ─► harmonic stack ─► conv stack ─► BiGRU ─┬► frame   [T, P]
                                                                                ├► onset   [T, P]
                                                                                └► string  [T, P, 6]

* **frame:** is pitch p sounding at frame t?
* **onset:** does a note of pitch p start at frame t? Separate onsets are what
  split repeated notes and soft re-plucks, the main failure on gentle acoustic.
* **string:** if pitch p is sounding, which string is it on?

Predicting the string separately from the pitch (rather than one fret class
per string, as in TabCNN) keeps the model independent of tuning: a drop-C
metal riff and the same riff in standard tuning only differ in pitch, and
pitch-shift augmentation keeps every label valid. The tuning is only needed
at decode time, to turn (pitch, string) into a fret.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn

from .config import IGNORE, ModelConfig
from .features import cqt_kernel_groups, harmonic_shifts, midi_to_hz


class CQT(nn.Module):
    """Constant-Q magnitude as strided 1-D convolutions, one per octave."""

    def __init__(self, sr: int, hop: int, fmin: float, n_bins: int, bins_per_octave: int, filter_scale: float):
        super().__init__()
        self.hop = hop
        self.n_groups = 0
        for i, kern in enumerate(cqt_kernel_groups(sr, fmin, n_bins, bins_per_octave, filter_scale)):
            w = torch.cat([torch.from_numpy(kern.real.copy()), torch.from_numpy(kern.imag.copy())], 0)[:, None, :]
            # Rebuilt from the config on load, so not stored in checkpoints.
            self.register_buffer(f"kernel{i}", w.float(), persistent=False)
            self.n_groups += 1

    def forward(self, audio: torch.Tensor) -> torch.Tensor:  # [B, N] -> [B, F, T]
        x = audio[:, None, :]
        out = []
        for i in range(self.n_groups):
            w = getattr(self, f"kernel{i}")
            y = F.conv1d(x, w, stride=self.hop, padding=w.shape[-1] // 2)
            nb = w.shape[0] // 2
            out.append(torch.sqrt(y[:, :nb] ** 2 + y[:, nb:] ** 2 + 1e-12))
        return torch.cat(out, dim=1)


def log_normalize(mag: torch.Tensor, floor_db: float = -80.0, ref_floor_db: float = -50.0) -> torch.Tensor:
    """dB relative to the loudest bin in the clip, mapped to [0, 1].

    Makes the model indifferent to recording level. The reference never drops
    below ``ref_floor_db``, so near-silent clips stay near zero instead of
    having their noise floor amplified to full scale.
    """
    db = 20 * torch.log10(mag.clamp_min(1e-7))
    ref = db.amax(dim=(1, 2), keepdim=True).clamp_min(ref_floor_db)
    return ((db - ref).clamp_min(floor_db) - floor_db) / -floor_db


class HarmonicStack(nn.Module):
    """[B, F, T] -> [B, H, F_out, T]: channel h is the CQT shifted so harmonic h lines up with bin f."""

    def __init__(self, shifts: list[int], n_bins_in: int, n_bins_out: int):
        super().__init__()
        self.shifts = shifts
        self.n_out = n_bins_out
        self.lo = max(0, -min(shifts))
        self.hi = max(0, max(shifts) + n_bins_out - n_bins_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, (0, 0, self.lo, self.hi))
        return torch.stack([xp[:, self.lo + s : self.lo + s + self.n_out] for s in self.shifts], dim=1)


def _block(c_in: int, c_out: int, kernel, stride=1, padding=0, dilation=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class TabCRNN(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        if cfg.bins_per_octave % 12:
            raise ValueError("bins_per_octave must be a multiple of 12")
        bps = cfg.bins_per_octave // 12
        P, c = cfg.n_pitches, cfg.channels
        n_bins = cfg.bins_per_octave * cfg.n_octaves
        self.cqt = CQT(cfg.sr, cfg.hop, midi_to_hz(cfg.pitch_min), n_bins, cfg.bins_per_octave, cfg.filter_scale)
        self.stack = HarmonicStack(harmonic_shifts(cfg.harmonics, cfg.bins_per_octave), n_bins, P * bps)
        self.conv = nn.Sequential(
            _block(len(cfg.harmonics), c, (5, 5), padding=(2, 2)),
            _block(c, c, (3, 3), padding=(1, 1)),
            _block(c, c, (3, 3), padding=(1, 2), dilation=(1, 2)),  # wider in time
            _block(c, c, (bps, 1), stride=(bps, 1)),  # quarter-tone bins -> one row per semitone
            _block(c, c, (3, 3), padding=(1, 1)),
        )
        self.proj = nn.Sequential(nn.Linear(c * P, cfg.hidden), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout))
        self.rnn = nn.GRU(cfg.hidden, cfg.hidden // 2, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.frame_head = nn.Linear(cfg.hidden, P)
        self.onset_head = nn.Linear(cfg.hidden, P)
        self.string_head = nn.Linear(cfg.hidden, P * cfg.n_strings)

    def spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        # Long CQT sums lose precision in fp16, so keep this part in fp32.
        ctx = torch.autocast(device_type="cuda", enabled=False) if audio.is_cuda else contextlib.nullcontext()
        with ctx:
            return log_normalize(self.cqt(audio.float()))

    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stack(self.spectrogram(audio))  # [B, H, P*bps, T]
        x = self.conv(x)  # [B, C, P, T]
        B, C, P, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * P)
        x, _ = self.rnn(self.proj(x))
        x = self.drop(x)
        return {
            "frame": self.frame_head(x),
            "onset": self.onset_head(x),
            "string": self.string_head(x).view(B, T, P, self.cfg.n_strings),
        }


def compute_loss(
    out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], onset_pos_weight: float = 4.0, string_weight: float = 0.5
) -> tuple[torch.Tensor, dict[str, float]]:
    frame_t, onset_t, string_t = batch["frame"], batch["onset"], batch["string"]
    frame_l = F.binary_cross_entropy_with_logits(out["frame"].float(), frame_t)
    pos_weight = torch.tensor(onset_pos_weight, device=onset_t.device)
    onset_l = F.binary_cross_entropy_with_logits(out["onset"].float(), onset_t, pos_weight=pos_weight)
    mask = string_t != IGNORE
    if mask.any():
        string_l = F.cross_entropy(out["string"].float()[mask], string_t[mask])
    else:  # no string labels in this batch (e.g. pitch-only MIDI data)
        string_l = out["string"].sum() * 0.0
    total = frame_l + onset_l + string_weight * string_l
    return total, {"frame": frame_l.item(), "onset": onset_l.item(), "string": string_l.item(), "total": total.item()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
