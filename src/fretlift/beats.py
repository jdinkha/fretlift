"""Tempo and beat tracking with librosa."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .types import BeatGrid

log = logging.getLogger(__name__)


def track_beats(audio_path: Path, tempo: float | None = None) -> BeatGrid:
    """Detect beats. Pass ``tempo`` (BPM) to override tempo estimation.

    Run this on the full mix rather than the guitar stem when you can: drums
    make beats much easier to find.
    """
    try:
        import librosa
    except ImportError as e:
        if e.name != "librosa":
            raise  # installed but broken: show the real error
        raise RuntimeError("librosa isn't installed. Run `pip install -e .[ml]`.") from e

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    kwargs = {"bpm": tempo} if tempo else {}
    est_tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", **kwargs)
    est = float(np.atleast_1d(est_tempo)[0])
    beats = [float(b) for b in beats]

    if len(beats) >= 2:
        est = 60.0 / float(np.median(np.diff(beats)))
    if tempo:
        est = tempo
    if not beats:
        log.warning("No beats found; assuming a steady %.0f BPM from the start", est or 120)
        est = est or 120.0
        beats = [0.0]
    log.info("Tempo ~%.1f BPM, %d beats", est, len(beats))
    return BeatGrid(tempo=est, beats=beats)
