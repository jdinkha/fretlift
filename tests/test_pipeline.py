"""Pipeline wiring, with the ML stages replaced by fakes (no models needed)."""

import json
from unittest import mock

import pytest

from fretlift.pipeline import PipelineConfig, run
from fretlift.separation import pick_channel
from fretlift.types import BeatGrid, Note


def test_pipeline_wiring(tmp_path):
    audio = tmp_path / "take.wav"
    audio.write_bytes(b"not really audio")
    notes = [Note(0.5 + i * 0.5, 0.9 + i * 0.5, p) for i, p in enumerate([40, 45, 50, 55])]
    grid = BeatGrid(120.0, [0.5 + i * 0.5 for i in range(16)])

    with (
        mock.patch("fretlift.transcription.transcribe", return_value=notes) as transcribe,
        mock.patch("fretlift.beats.track_beats", return_value=grid) as beats,
        mock.patch("fretlift.separation.separate_guitar") as separate,
        mock.patch("fretlift.separation.pick_channel", side_effect=lambda path, *_: path),
    ):
        out = run(audio, tmp_path / "out", PipelineConfig(separate=False, save_notes=True))

    separate.assert_not_called()
    assert transcribe.call_args.kwargs["min_pitch"] == 40  # low E in standard tuning
    beats.assert_called_once()
    assert set(out) == {"musicxml", "ascii", "notes"}
    assert "E|0" in out["ascii"].read_text()
    assert len(json.loads(out["notes"].read_text())["notes"]) == 4


def test_both_channels_separate_once(tmp_path):
    audio = tmp_path / "take.wav"
    audio.write_bytes(b"not really audio")

    with (
        mock.patch("fretlift.transcription.transcribe", return_value=[Note(0.5, 0.9, 40)]),
        mock.patch("fretlift.beats.track_beats", return_value=BeatGrid(120.0, [0.5, 1.0, 1.5, 2.0])),
        mock.patch("fretlift.separation.separate_guitar", return_value=audio) as separate,
        mock.patch("fretlift.separation.pick_channel", side_effect=lambda path, *_: path) as pick,
    ):
        out = run(audio, tmp_path / "out", PipelineConfig(channel="both", formats=("ascii",), keep_stem=True))

    separate.assert_called_once()
    assert [c.args[2] for c in pick.call_args_list] == ["left", "right"]
    assert sorted(p.name for p in out.values()) == [
        "take.guitar.left.wav",
        "take.guitar.right.wav",
        "take.guitar.wav",
        "take.left.tab.txt",
        "take.right.tab.txt",
    ]


def test_pick_channel_splits_two_panned_guitars(tmp_path, caplog):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    t = np.arange(8000) / 8000
    left, right = 0.5 * np.sin(2 * np.pi * 110 * t), 0.5 * np.sin(2 * np.pi * 173 * t)
    path = tmp_path / "two.wav"
    sf.write(str(path), np.stack([left, right], axis=1), 8000)

    assert pick_channel(path, tmp_path) == path
    assert "two guitars" in caplog.text
    np.testing.assert_allclose(sf.read(str(pick_channel(path, tmp_path, "right")))[0], right, atol=1e-3)


def test_remove_center_strips_bass_shared_by_both_sides(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    sr = 8000
    t = np.arange(sr) / sr
    bass = 0.3 * np.sin(2 * np.pi * 98 * t)  # centred: identical on both sides
    left, right = 0.3 * np.sin(2 * np.pi * 330 * t), 0.3 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "mix.wav"
    sf.write(str(path), np.stack([left + bass, right + bass], axis=1), sr)

    out, _ = sf.read(str(pick_channel(path, tmp_path, "left", remove_center=True)))
    spectrum = np.abs(np.fft.rfft(out))  # 1 Hz bins
    assert spectrum[98] < 0.1 * spectrum[330]  # bass gone, guitar kept
