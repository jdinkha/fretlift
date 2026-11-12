"""End-to-end: audio file -> tab files."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .export import write_ascii, write_musicxml
from .fretting import MAX_FRET, FretConfig, assign_frets
from .quantize import quantize
from .tuning import PRESETS

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    separate: bool = True
    demucs_model: str = "htdemucs_6s"
    device: str | None = None
    tuning: tuple[int, ...] = PRESETS["standard"]
    capo: int = 0
    max_fret: int = MAX_FRET
    tempo: float | None = None
    subdivision: int = 4
    time_signature: tuple[int, int] = (4, 4)
    onset_threshold: float | None = None  # None = the trained model's tuned value
    frame_threshold: float | None = None
    min_note_ms: float = 58.0
    formats: tuple[str, ...] = ("musicxml", "ascii")
    title: str | None = None
    save_notes: bool = False
    keep_stem: bool = False
    shifts: int = 1
    channel: str = "mix"  # or "left" / "right" / "both"
    remove_center: bool = False


def run(audio_path: str | Path, out_dir: str | Path, cfg: PipelineConfig | None = None) -> dict[str, Path]:
    # Imported here so the core (quantize/fretting/export) works without ML deps.
    from .beats import track_beats
    from .separation import pick_channel, separate_guitar
    from .transcription import transcribe

    cfg = cfg or PipelineConfig()
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    title = cfg.title or audio_path.stem
    channels = ("left", "right") if cfg.channel == "both" else (cfg.channel,)
    outputs: dict[str, Path] = {}

    with tempfile.TemporaryDirectory(prefix="fretlift-") as tmp:
        guitar = audio_path
        if cfg.separate:
            guitar = separate_guitar(audio_path, Path(tmp), cfg.demucs_model, cfg.device, cfg.shifts)
            if cfg.keep_stem:
                outputs["stem"] = Path(shutil.copy(guitar, out_dir / f"{audio_path.stem}.guitar.wav"))

        lowest = min(cfg.tuning) + cfg.capo
        highest = max(cfg.tuning) + cfg.max_fret
        notes_by_channel = {}
        for ch in channels:
            track = pick_channel(guitar, Path(tmp), ch, cfg.remove_center)
            if cfg.separate and cfg.keep_stem and ch != "mix":
                outputs[f"stem.{ch}"] = Path(shutil.copy(track, out_dir / f"{audio_path.stem}.guitar.{ch}.wav"))
            notes_by_channel[ch] = transcribe(
                track,
                min_pitch=lowest,
                max_pitch=highest,
                onset_threshold=cfg.onset_threshold,
                frame_threshold=cfg.frame_threshold,
                min_note_ms=cfg.min_note_ms,
                tuning=cfg.tuning,
                capo=cfg.capo,
                max_fret=cfg.max_fret,
                device=cfg.device,
            )
        grid = track_beats(audio_path, tempo=cfg.tempo)  # full mix: drums help

    fret_cfg = FretConfig(tuning=cfg.tuning, capo=cfg.capo, max_fret=cfg.max_fret)
    for ch, notes in notes_by_channel.items():
        suffix = "" if ch == "mix" else f".{ch}"
        name = audio_path.stem + suffix

        if cfg.save_notes:
            p = out_dir / f"{name}.notes.json"
            data = {"tempo": grid.tempo, "beats": grid.beats, "notes": [asdict(n) for n in notes]}
            p.write_text(json.dumps(data, indent=1))
            outputs["notes" + suffix] = p

        events = quantize(notes, grid, cfg.subdivision)
        events = assign_frets(events, fret_cfg)
        if not events:
            log.warning("No playable notes were found, so the %s tab will be empty", name)

        common = dict(
            tuning=cfg.tuning,
            capo=cfg.capo,
            tempo=grid.tempo,
            time_signature=cfg.time_signature,
            subdivision=cfg.subdivision,
            title=title if ch == "mix" else f"{title} ({ch})",
        )
        if "musicxml" in cfg.formats:
            outputs["musicxml" + suffix] = write_musicxml(out_dir / f"{name}.musicxml", events, **common)
        if "ascii" in cfg.formats:
            outputs["ascii" + suffix] = write_ascii(out_dir / f"{name}.tab.txt", events, **common)
    return outputs
