"""Run a trained checkpoint on full-length audio."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ..types import Note
from .config import DecodeConfig, ModelConfig, n_frames
from .decode import decode_notes
from .model import TabCRNN


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[TabCRNN, dict]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model = TabCRNN(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_activations(
    model: TabCRNN,
    audio: np.ndarray,
    *,
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 2.0,
    batch_size: int = 4,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray]:
    """Probabilities for the whole recording: frame/onset [T, P], string [T, P, S].

    Long audio is processed in overlapping chunks; only the middle of each
    chunk is kept, so the recurrent layer always has context on both sides.
    """
    cfg = model.cfg
    device = device or next(model.parameters()).device
    hop, sr = cfg.hop, cfg.sr
    total = n_frames(len(audio), hop)
    if total == 0:
        P, S = cfg.n_pitches, cfg.n_strings
        return {
            "frame": np.zeros((0, P), np.float32),
            "onset": np.zeros((0, P), np.float32),
            "string": np.zeros((0, P, S), np.float32),
        }
    ov_f = max(1, int(round(overlap_seconds * sr / hop)))
    step_f = max(1, int(round(chunk_seconds * sr / hop)) - 2 * ov_f)
    ov, step = ov_f * hop, step_f * hop
    n_chunks = -(-total // step_f)
    padded = np.zeros(ov + n_chunks * step + ov + hop, np.float32)
    padded[ov : ov + len(audio)] = audio
    seg_len = step + 2 * ov

    outs: dict[str, list[np.ndarray]] = {"frame": [], "onset": [], "string": []}
    model.eval()
    for b in range(0, n_chunks, batch_size):
        idx = range(b, min(b + batch_size, n_chunks))
        x = torch.from_numpy(np.stack([padded[k * step : k * step + seg_len] for k in idx])).to(device)
        out = model(x)
        frame = torch.sigmoid(out["frame"])[:, ov_f : ov_f + step_f]
        onset = torch.sigmoid(out["onset"])[:, ov_f : ov_f + step_f]
        string = torch.softmax(out["string"].float(), dim=-1)[:, ov_f : ov_f + step_f]
        for name, t in (("frame", frame), ("onset", onset), ("string", string)):
            outs[name].extend(t.float().cpu().numpy())
    return {k: np.concatenate(v, axis=0)[:total] for k, v in outs.items()}


def transcribe_audio(
    model: TabCRNN,
    audio: np.ndarray,
    *,
    decode: DecodeConfig | None = None,
    tuning: Sequence[int] | None = None,
    capo: int = 0,
    max_fret: int = 22,
    device=None,
) -> list[Note]:
    act = predict_activations(model, audio, device=device)
    cfg = model.cfg
    return decode_notes(
        act["frame"], act["onset"], cfg=decode, fps=cfg.fps, pitch_min=cfg.pitch_min,
        string_prob=act["string"] if tuning is not None else None, tuning=tuning, capo=capo, max_fret=max_fret,
    )


class ModelPredictor:
    """Evaluation adapter. Caches activations, so threshold sweeps only re-decode."""

    name = "model"

    def __init__(self, model: TabCRNN, eval_audio, device=None, cache: bool = True):
        self.model = model
        self.audio = eval_audio
        self.device = device or next(model.parameters()).device
        self.cache = {} if cache else None

    def activations(self, item, distort):
        key = (item.id, distort)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        act = predict_activations(self.model, self.audio.array(item, distort), device=self.device)
        if self.cache is not None:
            self.cache[key] = act
        return act

    def predict(self, item, distort, decode: DecodeConfig, tuning: Sequence[int], strings: bool = True) -> list[Note]:
        act = self.activations(item, distort)
        cfg = self.model.cfg
        use_strings = strings and len(tuning) == cfg.n_strings
        return decode_notes(
            act["frame"], act["onset"], cfg=decode, fps=cfg.fps, pitch_min=cfg.pitch_min,
            string_prob=act["string"] if use_strings else None, tuning=tuning if use_strings else None,
        )
