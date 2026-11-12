"""Turn the model's frame-level probabilities into notes (and strings).

NumPy only, so threshold sweeps can re-decode cached model outputs quickly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import Note
from .config import PITCH_MIN, DecodeConfig


def _note_spans(frame_prob: np.ndarray, onset_prob: np.ndarray, cfg: DecodeConfig) -> list[tuple[int, int, int]]:
    """(start frame, end frame, pitch index) for every detected note.

    A note starts at a local peak of the onset probability above
    ``onset_threshold`` and lasts while the frame probability stays above
    ``frame_threshold``, or until the next onset of the same pitch (so repeated
    notes stay separate even when the string keeps ringing).
    """
    T, P = frame_prob.shape
    spans = []
    for p in range(P):
        o, f = onset_prob[:, p], frame_prob[:, p]
        above = o >= cfg.onset_threshold
        if not above.any():
            continue
        prev = np.concatenate(([-np.inf], o[:-1]))
        nxt = np.concatenate((o[1:], [-np.inf]))
        peaks = np.flatnonzero(above & (o > prev) & (o >= nxt))
        kept: list[int] = []
        for t in peaks:
            if not kept or t - kept[-1] >= cfg.min_gap_frames:
                kept.append(int(t))
        for k, t0 in enumerate(kept):
            limit = kept[k + 1] if k + 1 < len(kept) else T
            below = f[t0 + 1 : limit] < cfg.frame_threshold
            t1 = t0 + 1 + (int(np.argmax(below)) if below.any() else len(below))
            if t1 - t0 >= cfg.min_frames:
                spans.append((t0, t1, p))
    spans.sort()
    return spans


def _valid_strings(pitch: int, tuning: Sequence[int], capo: int, max_fret: int) -> list[int]:
    n = len(tuning)
    out = []
    for s in range(1, n + 1):  # string 1 = highest = tuning[-1]
        fret = pitch - (tuning[n - s] + capo)
        if 0 <= fret <= max_fret - capo:
            out.append(s)
    return out


def _assign_chord(options: list[list[tuple[int, float]]]) -> list[int | None]:
    """Best one-note-per-string assignment for a chord (exhaustive; chords are small)."""
    best: tuple[float, list[int | None]] = (-np.inf, [None] * len(options))

    def search(k: int, used: set[int], chosen: list[int | None], score: float, placed: int):
        nonlocal best
        if k == len(options):
            # Prefer placing more notes, then the higher total log-probability.
            total = placed * 1e6 + score
            if total > best[0]:
                best = (total, list(chosen))
            return
        for s, logp in options[k]:
            if s not in used:
                used.add(s)
                chosen.append(s)
                search(k + 1, used, chosen, score + logp, placed + 1)
                chosen.pop()
                used.remove(s)
        chosen.append(None)
        search(k + 1, used, chosen, score, placed)
        chosen.pop()

    if len(options) > 8:  # pathological; fall back to independent argmax
        return [max(o, key=lambda x: x[1])[0] if o else None for o in options]
    search(0, set(), [], 0.0, 0)
    return best[1]


def decode_notes(
    frame_prob: np.ndarray,
    onset_prob: np.ndarray,
    *,
    cfg: DecodeConfig | None = None,
    fps: float,
    pitch_min: int = PITCH_MIN,
    string_prob: np.ndarray | None = None,
    tuning: Sequence[int] | None = None,
    capo: int = 0,
    max_fret: int = 24,
) -> list[Note]:
    """Probabilities [T, P] (and optionally string probabilities [T, P, S]) -> notes.

    With ``string_prob`` and ``tuning``, every note also gets a string: the one
    the model finds most likely among the strings that can actually play that
    pitch in that tuning, with at most one note per string in a chord.
    """
    cfg = cfg or DecodeConfig()
    spans = _note_spans(frame_prob, onset_prob, cfg)
    strings: list[int | None] = [None] * len(spans)

    if string_prob is not None and tuning is not None:
        if string_prob.shape[-1] != len(tuning):
            raise ValueError(f"Model predicts {string_prob.shape[-1]} strings but the tuning has {len(tuning)}")
        options = []
        for t0, t1, p in spans:
            post = string_prob[t0:t1, p].mean(axis=0)  # string index 0 = string 1
            opts = [(s, float(np.log(post[s - 1] + 1e-9))) for s in _valid_strings(p + pitch_min, tuning, capo, max_fret)]
            options.append(opts)
        i = 0
        while i < len(spans):  # chords: onsets within chord_frames of the first note
            j = i + 1
            while j < len(spans) and spans[j][0] - spans[i][0] <= cfg.chord_frames:
                j += 1
            strings[i:j] = _assign_chord(options[i:j])
            i = j

    return [
        Note(start=t0 / fps, end=t1 / fps, pitch=p + pitch_min, string=s) for (t0, t1, p), s in zip(spans, strings)
    ]
