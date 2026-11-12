"""Basic Pitch as a baseline, with cached model outputs for cheap threshold sweeps.

Running the network is the slow part; turning its output into notes at a given
pair of thresholds is fast. So the raw output is cached once per file, and
notes are decoded from the cache using Basic Pitch's own note-creation code.
If that internal function isn't available in the installed version, it falls
back to re-running ``predict`` for every setting (correct, just slower).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..fretting import FretConfig, assign_strings
from ..transcription import midi_to_hz
from ..types import Note
from .config import HOP, SR, DecodeConfig
from .evaluate import EvalAudio, playable_range
from .manifest import Item

log = logging.getLogger(__name__)


class BasicPitchPredictor:
    name = "basic-pitch"

    def __init__(self, eval_audio: EvalAudio, cache_dir: str | Path):
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: F401
        except ImportError as e:
            raise SystemExit('Basic Pitch isn\'t installed: pip install -e ".[ml]"') from e
        self.audio = eval_audio
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fast: bool | None = None  # decided on first use

    # Basic Pitch analyses audio at 22050 Hz with a 256-sample hop, like this project.
    @staticmethod
    def _min_note_ms(decode: DecodeConfig) -> float:
        return decode.min_frames * HOP / SR * 1000

    def _predict(self, path: Path, decode: DecodeConfig, lo: int, hi: int):
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict

        return predict(
            str(path),
            ICASSP_2022_MODEL_PATH,
            onset_threshold=decode.onset_threshold,
            frame_threshold=decode.frame_threshold,
            minimum_note_length=self._min_note_ms(decode),
            minimum_frequency=midi_to_hz(lo - 0.5),
            maximum_frequency=midi_to_hz(hi + 0.5),
            melodia_trick=True,
        )

    def _cached_output(self, path: Path, decode: DecodeConfig, lo: int, hi: int) -> dict[str, np.ndarray]:
        st = path.stat()
        key = hashlib.md5(f"{path.resolve()}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()
        f = self.cache_dir / f"{key}.npz"
        if f.exists():
            with np.load(f) as z:
                return {k: z[k] for k in z.files}
        output, _, _ = self._predict(path, decode, lo, hi)
        np.savez_compressed(f, **{k: np.asarray(v) for k, v in output.items()})
        return output

    def _fast_events(self, output, decode: DecodeConfig, lo: int, hi: int):
        from basic_pitch import note_creation

        res = note_creation.model_output_to_notes(
            output,
            onset_thresh=decode.onset_threshold,
            frame_thresh=decode.frame_threshold,
            min_note_len=int(round(self._min_note_ms(decode) / 1000 * SR / HOP)),
            min_freq=midi_to_hz(lo - 0.5),
            max_freq=midi_to_hz(hi + 0.5),
            melodia_trick=True,
        )
        events = res[1] if isinstance(res, tuple) else res
        if events and not (isinstance(events[0], tuple | list) and len(events[0]) >= 3):
            raise TypeError("unexpected note event format")
        return events

    def _events(self, path: Path, decode: DecodeConfig, lo: int, hi: int):
        if self._fast is not False:
            output = self._cached_output(path, decode, lo, hi)
            try:
                fast = self._fast_events(output, decode, lo, hi)
            except (ImportError, AttributeError, TypeError, KeyError) as e:
                log.warning("Fast Basic Pitch decoding unavailable (%s); re-running the model per setting", e)
                self._fast = False
            else:
                if self._fast is None:  # check once that the fast path agrees with predict()
                    _, _, slow = self._predict(path, decode, lo, hi)
                    self._fast = abs(len(slow) - len(fast)) <= max(2, 0.02 * len(slow))
                    if not self._fast:
                        log.warning(
                            "Fast decoding disagreed with predict() (%d vs %d notes); using predict()", len(fast), len(slow)
                        )
                        return slow
                return fast
        _, _, events = self._predict(path, decode, lo, hi)
        return events

    def predict(
        self, item: Item, distort: str | None, decode: DecodeConfig, tuning: Sequence[int], strings: bool = True
    ) -> list[Note]:
        lo, hi = playable_range(tuning)
        events = self._events(self.audio.path(item, distort), decode, lo, hi)
        notes = [Note(start=float(e[0]), end=float(e[1]), pitch=int(e[2])) for e in events if lo <= int(e[2]) <= hi]
        notes.sort(key=lambda n: (n.start, n.pitch))
        # Basic Pitch only gives pitches, so strings come from the Phase 1 fingering search.
        return assign_strings(notes, FretConfig(tuning=tuple(tuning))) if strings else notes
