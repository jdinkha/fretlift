"""PyTorch dataset of random, optionally augmented, labelled clips."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .audio import AudioCache
from .augment import Augmenter
from .config import ModelConfig, n_frames
from .labels import load_labels, make_targets, window_notes
from .manifest import Item


class ClipDataset(Dataset):
    """Each ``__getitem__`` returns a random clip from a random track.

    Tracks are picked in proportion to their length, so long recordings aren't
    under-sampled. "Epoch" just means ``length`` clips.

    Audio is memory-mapped from the cache and only opened inside each worker,
    so starting workers never copies the whole dataset.
    """

    def __init__(
        self,
        items: Sequence[Item],
        cache: AudioCache,
        model_cfg: ModelConfig,
        *,
        clip_seconds: float = 5.0,
        length: int = 8000,
        augmenter: Augmenter | None = None,
        seed: int = 0,
    ):
        if not items:
            raise ValueError("No training items")
        self.cfg = model_cfg
        self.clip = int(round(clip_seconds * model_cfg.sr))
        self.length = length
        self.augmenter = augmenter
        self.seed = seed
        self.paths, self.notes, durations = [], [], []
        self.amped = [any(t == "amp" for t in it.tags) for it in items]  # recorded through an amp already
        for k, it in enumerate(items, start=1):
            p = cache.ensure(it.audio)
            self.paths.append(str(p))
            durations.append(np.load(p, mmap_mode="r").shape[0])
            self.notes.append(load_labels(it.labels)[0])
            if k % 50 == 0 or k == len(items):
                print(f"  cached audio {k}/{len(items)}", flush=True)
        w = np.asarray(durations, float)
        self.weights = w / w.sum()
        self._pid = None

    def __len__(self) -> int:
        return self.length

    def _worker_state(self):
        if self._pid != os.getpid():  # first call in this process (a fresh worker)
            info = get_worker_info()
            self._rng = np.random.default_rng(info.seed % 2**32 if info else self.seed)
            self._arrays = {}
            self._pid = os.getpid()
        return self._rng

    def _array(self, k: int) -> np.ndarray:
        if k not in self._arrays:
            self._arrays[k] = np.load(self.paths[k], mmap_mode="r")
        return self._arrays[k]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = self._worker_state()
        k = int(rng.choice(len(self.paths), p=self.weights))
        arr = self._array(k)
        start = int(rng.integers(0, max(1, len(arr) - self.clip + 1)))
        audio = np.zeros(self.clip, np.float32)
        seg = np.asarray(arr[start : start + self.clip], np.float32)
        audio[: len(seg)] = seg
        sr = self.cfg.sr
        notes = window_notes(self.notes[k], start / sr, (start + self.clip) / sr)
        if self.augmenter is not None:
            audio, notes, _ = self.augmenter(audio, notes, rng, amp=not self.amped[k])
        frame, onset, string = make_targets(
            notes, n_frames(self.clip, self.cfg.hop), self.cfg.fps, self.cfg.pitch_min, self.cfg.n_pitches
        )
        return {
            "audio": torch.from_numpy(np.ascontiguousarray(audio, np.float32)),
            "frame": torch.from_numpy(frame),
            "onset": torch.from_numpy(onset),
            "string": torch.from_numpy(string),
        }
