"""Audio -> notes.

The transcriber is the model you train yourself (docs/TRAINING.md), which also
hears which string each note is on. Until a checkpoint is installed at
``model_path()``, Basic Pitch stands in; it's deprecated and will be removed.
"""

from __future__ import annotations

import logging
import os
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from pathlib import Path

from .fretting import MAX_FRET
from .types import Note

log = logging.getLogger(__name__)


def model_path() -> Path:
    """Where the trained checkpoint is looked for: ``$FRETLIFT_MODEL``, else the per-user data folder."""
    return Path(os.environ.get("FRETLIFT_MODEL") or Path.home() / ".local/share/fretlift/model.pt")


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def drop_ghost_harmonics(notes: list[Note], ratio: float = 0.6, onset_tol: float = 0.05) -> list[Note]:
    """Drop quiet notes that are really overtones of a louder note starting at the same time.

    Basic Pitch often reports the 2nd-4th harmonics (an octave, an octave and a fifth,
    two octaves up) as extra, quieter notes. Real octaves are played at a similar level,
    so they survive. ``notes`` must be sorted by start time.
    """
    starts = [n.start for n in notes]
    kept = []
    for n in notes:
        lo, hi = bisect_left(starts, n.start - onset_tol), bisect_right(starts, n.start + onset_tol)
        if not any(n.pitch - m.pitch in (12, 19, 24) and n.velocity < ratio * m.velocity for m in notes[lo:hi]):
            kept.append(n)
    if len(kept) < len(notes):
        log.info("Dropped %d ghost harmonics", len(notes) - len(kept))
    return kept


def _transcribe_model(
    checkpoint: Path,
    audio_path: Path,
    onset_threshold: float | None,
    frame_threshold: float | None,
    min_note_ms: float,
    tuning: Sequence[int] | None,
    capo: int,
    max_fret: int,
    device: str | None,
) -> list[Note]:
    try:
        from .training.audio import load_audio
        from .training.config import DecodeConfig
        from .training.inference import load_model, pick_device, transcribe_audio
    except ImportError as e:
        if e.name != "torch":
            raise  # installed but broken: show the real error
        raise RuntimeError("The transcription model needs PyTorch. Run `pip install -e .[ml]`.") from e

    log.info("Transcribing notes with the trained model (%s)", checkpoint)
    model, ckpt = load_model(checkpoint, pick_device(device or "auto"))
    tuned = (ckpt.get("thresholds") or {}).get("all") or {}  # tuned on validation data at the end of training
    decode = DecodeConfig(min_frames=max(1, round(min_note_ms / 1000 * model.cfg.fps)))
    for name, given in (("onset_threshold", onset_threshold), ("frame_threshold", frame_threshold)):
        value = given if given is not None else tuned.get(name)
        if value is not None:
            setattr(decode, name, value)
    if tuning is not None and len(tuning) != model.cfg.n_strings:
        log.warning("The model hears %d strings but the tuning has %d, so strings come from the fingering search",
                    model.cfg.n_strings, len(tuning))
        tuning = None
    audio = load_audio(audio_path, model.cfg.sr)
    return transcribe_audio(model, audio, decode=decode, tuning=tuning, capo=capo, max_fret=max_fret)


def _transcribe_basic_pitch(
    audio_path: Path, min_pitch: int, max_pitch: int, onset_threshold: float, frame_threshold: float, min_note_ms: float
) -> list[Note]:
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
    except ImportError as e:
        if e.name != "basic_pitch":
            raise  # installed but broken: show the real error
        raise RuntimeError("Basic Pitch isn't installed. Run `pip install -e .[ml]`.") from e

    _, _, note_events = predict(
        str(audio_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_ms,
        # A little margin so slightly sharp/flat notes at the edges survive.
        minimum_frequency=midi_to_hz(min_pitch - 0.5),
        maximum_frequency=midi_to_hz(max_pitch + 0.5),
        melodia_trick=True,
    )
    notes = [Note(start=float(s), end=float(e), pitch=int(p), velocity=float(a)) for s, e, p, a, *_ in note_events]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return drop_ghost_harmonics(notes)


def transcribe(
    audio_path: Path,
    *,
    min_pitch: int = 38,
    max_pitch: int = 88,
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
    min_note_ms: float = 58.0,
    tuning: Sequence[int] | None = None,
    capo: int = 0,
    max_fret: int = MAX_FRET,
    device: str | None = None,
) -> list[Note]:
    """Return the notes in the recording, limited to the instrument's range.

    With a trained checkpoint installed and a ``tuning``, every note also has
    the string the model heard it on. Thresholds left as ``None`` use the
    values tuned at the end of training.
    """
    checkpoint = model_path()
    if checkpoint.exists():
        notes = _transcribe_model(
            checkpoint, audio_path, onset_threshold, frame_threshold, min_note_ms, tuning, capo, max_fret, device
        )
    else:
        log.warning(
            "No trained model at %s, so falling back to Basic Pitch, which is deprecated and will be removed. "
            "Train a model (docs/TRAINING.md) and copy its best.pt there.",
            checkpoint,
        )
        notes = _transcribe_basic_pitch(
            audio_path,
            min_pitch,
            max_pitch,
            0.5 if onset_threshold is None else onset_threshold,
            0.3 if frame_threshold is None else frame_threshold,
            min_note_ms,
        )
    notes = [n for n in notes if min_pitch <= n.pitch <= max_pitch]
    log.info("Found %d notes", len(notes))
    return notes
