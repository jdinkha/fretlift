"""Make training audio that looks like what the real pipeline sees.

At inference time the transcriber hears Demucs's guitar stem, not a clean
recording, and separation leaves artifacts (bleed from other instruments,
smeared attacks, "underwater" highs). This mixes labelled guitar with
accompaniment, runs Demucs on the mix, and adds the separated guitar as new
items with the original labels, so the model learns to cope with those
artifacts.

Accompaniment must be audio you're allowed to use: your own recordings, or
openly licensed stems such as Slakh2100's non-guitar tracks (check the
license). Don't use separated commercial songs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audio import AUDIO_SUFFIXES, load_audio, write_wav
from .manifest import Item, write_manifest

log = logging.getLogger(__name__)

DEMUCS_SR = 44100


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def make_mixes(
    items: Sequence[Item],
    accompaniment_dir: str | Path,
    out_dir: str | Path,
    snr_db: tuple[float, float] = (-6.0, 6.0),
    seed: int = 0,
) -> list[tuple[Item, Path]]:
    """Guitar + random accompaniment at a random guitar-to-accompaniment ratio."""
    rng = np.random.default_rng(seed)
    acc_files = sorted(p for p in Path(accompaniment_dir).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not acc_files:
        raise FileNotFoundError(f"No accompaniment audio in {accompaniment_dir}")
    mix_dir = Path(out_dir) / "mixes"
    out = []
    for k, item in enumerate(items, start=1):
        g = load_audio(item.audio, DEMUCS_SR).astype(np.float64)
        a = load_audio(acc_files[int(rng.integers(len(acc_files)))], DEMUCS_SR).astype(np.float64)
        if len(a) < len(g):
            a = np.tile(a, int(np.ceil(len(g) / max(len(a), 1))))
        start = int(rng.integers(0, len(a) - len(g) + 1))
        a = a[start : start + len(g)]
        snr = rng.uniform(*snr_db)
        a *= _rms(g) / _rms(a) * 10 ** (-snr / 20)
        mix = g + a
        mix *= 0.95 / max(np.max(np.abs(mix)), 1e-9)
        path = write_wav(mix_dir / f"{_safe(item.id)}.wav", mix, DEMUCS_SR)
        out.append((item, path))
        if k % 25 == 0 or k == len(items):
            print(f"  mixed {k}/{len(items)}", flush=True)
    return out


def separate_batch(
    paths: Sequence[Path], out_dir: Path, model: str = "htdemucs_6s", device: str | None = None, batch: int = 32
):
    try:
        import demucs.separate
    except ImportError as e:
        raise SystemExit('Demucs isn\'t installed: pip install -e ".[ml]"') from e
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i : i + batch]]
        args = ["-n", model, "--two-stems", "guitar", "-o", str(out_dir)] + (["-d", device] if device else []) + chunk
        print(f"  separating {i + 1}-{i + len(chunk)} of {len(paths)}", flush=True)
        demucs.separate.main(args)


def mix_and_separate(
    items: Sequence[Item],
    accompaniment_dir: str | Path,
    out_dir: str | Path,
    *,
    snr_db: tuple[float, float] = (-6.0, 6.0),
    seed: int = 0,
    model: str = "htdemucs_6s",
    device: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    mixes = make_mixes(items, accompaniment_dir, out_dir, snr_db, seed)
    sep_dir = out_dir / "separated"
    todo = [p for _, p in mixes if not (sep_dir / model / p.stem / "guitar.wav").exists()]
    if todo:
        separate_batch(todo, sep_dir, model, device)
    new_items = []
    for item, mix in mixes:
        stem = sep_dir / model / mix.stem / "guitar.wav"
        if not stem.exists():
            log.warning("No separated stem for %s", item.id)
            continue
        new_items.append(replace(item, id=f"{item.id}#separated", audio=stem, tags=(*item.tags, "separated")))
    return write_manifest(out_dir / "manifest.jsonl", new_items)
