"""Model, dataset and training-loop tests. Skipped unless PyTorch is installed (CI's train job)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from fretlift.training.audio import AudioCache, write_wav  # noqa: E402
from fretlift.training.config import ModelConfig, n_frames  # noqa: E402
from fretlift.training.data import ClipDataset  # noqa: E402
from fretlift.training.features import cqt_reference, midi_to_hz  # noqa: E402
from fretlift.training.inference import load_model, predict_activations  # noqa: E402
from fretlift.training.labels import save_labels  # noqa: E402
from fretlift.training.manifest import Item, write_manifest  # noqa: E402
from fretlift.training.model import CQT, TabCRNN, compute_loss  # noqa: E402
from fretlift.types import Note  # noqa: E402

TINY = ModelConfig(channels=8, hidden=32)
SR = TINY.sr


def test_cqt_module_matches_numpy_reference():
    t = np.arange(SR // 2) / SR
    x = (0.3 * np.sin(2 * np.pi * midi_to_hz(52) * t)).astype(np.float32)
    cqt = CQT(SR, TINY.hop, midi_to_hz(33), 168, 24, 1.0)
    got = cqt(torch.from_numpy(x)[None])[0].numpy()
    ref = cqt_reference(x.astype(np.float64), SR, TINY.hop, midi_to_hz(33), 168, 24)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-3)


def test_forward_shapes_match_targets():
    model = TabCRNN(TINY).eval()
    for n in (SR, SR * 2 + 123):
        out = model(torch.zeros(2, n))
        T = n_frames(n, TINY.hop)
        assert out["frame"].shape == (2, T, TINY.n_pitches)
        assert out["onset"].shape == (2, T, TINY.n_pitches)
        assert out["string"].shape == (2, T, TINY.n_pitches, 6)


def test_loss_with_and_without_string_labels():
    model = TabCRNN(TINY)
    out = model(torch.randn(2, SR) * 0.1)
    T, P = out["frame"].shape[1:]
    batch = {"frame": torch.zeros(2, T, P), "onset": torch.zeros(2, T, P), "string": torch.full((2, T, P), -100)}
    loss, parts = compute_loss(out, batch)
    assert torch.isfinite(loss) and parts["string"] == 0.0
    loss.backward()  # no labels must still give a valid graph
    batch["string"][0, :10, 5] = 3
    loss, parts = compute_loss(model(torch.randn(2, SR) * 0.1), batch)
    assert torch.isfinite(loss) and parts["string"] > 0


def test_chunked_inference_covers_every_frame():
    model = TabCRNN(TINY).eval()
    audio = np.random.default_rng(0).standard_normal(int(SR * 7.3)).astype(np.float32) * 0.05
    act = predict_activations(model, audio, chunk_seconds=3.0, overlap_seconds=0.5)
    T = n_frames(len(audio), TINY.hop)
    assert act["frame"].shape == (T, TINY.n_pitches)
    assert act["string"].shape == (T, TINY.n_pitches, 6)
    assert np.allclose(act["string"].sum(-1), 1, atol=1e-4)
    # With enough overlap, chunking changes little compared to one pass.
    whole = torch.sigmoid(model(torch.from_numpy(audio)[None])["frame"])[0].detach().numpy()
    assert np.abs(whole - act["frame"]).mean() < 0.05


def synth_dataset(root, n_tracks=4, seconds=3.0):
    """Pure tones on known strings, as a stand-in dataset."""
    rng = np.random.default_rng(0)
    items = []
    for k in range(n_tracks):
        t = np.arange(int(seconds * SR)) / SR
        audio = np.zeros_like(t)
        notes = []
        for j in range(6):
            pitch, string = [(40, 6), (45, 5), (50, 4), (55, 3), (59, 2), (64, 1)][int(rng.integers(6))]
            s, e = j * 0.5, j * 0.5 + 0.4
            m = (t >= s) & (t < e)
            audio[m] += 0.3 * np.sin(2 * np.pi * midi_to_hz(pitch) * t[m]) * np.exp(-3 * (t[m] - s))
            notes.append(Note(s, e, pitch, string=string))
        wav = write_wav(root / f"t{k}.wav", audio, SR)
        lab = save_labels(root / f"t{k}.json", notes)
        items.append(Item(f"synth/{k}", wav, lab, group=f"g{k}", tags=("synth",)))
    return items


def test_dataset_items(tmp_path):
    items = synth_dataset(tmp_path)
    ds = ClipDataset(items, AudioCache(tmp_path / "cache", SR), TINY, clip_seconds=1.0, length=5)
    b = ds[0]
    T = n_frames(SR, TINY.hop)
    assert b["audio"].shape == (SR,) and b["frame"].shape == (T, TINY.n_pitches)
    assert b["string"].dtype == torch.int64


def test_tiny_model_overfits(tmp_path):
    items = synth_dataset(tmp_path, n_tracks=1)
    ds = ClipDataset(items, AudioCache(tmp_path / "cache", SR), TINY, clip_seconds=3.0, length=4)
    batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in ("audio", "frame", "onset", "string")}
    torch.manual_seed(0)
    model = TabCRNN(TINY)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        loss, _ = compute_loss(model(batch["audio"]), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_train_cli_end_to_end(tmp_path):
    from fretlift.training.train import main

    items = synth_dataset(tmp_path / "data", n_tracks=4)
    for it, split in zip(items, ["train", "train", "val", "test"]):
        object.__setattr__(it, "split", split)
    manifest = write_manifest(tmp_path / "data" / "manifest.jsonl", items)
    run = tmp_path / "run"
    rc = main([
        "--manifest", str(manifest), "--out", str(run), "--epochs", "1", "--steps-per-epoch", "3",
        "--batch-size", "2", "--clip-seconds", "1", "--workers", "0", "--device", "cpu",
        "--channels", "8", "--hidden", "32", "--cache-dir", str(tmp_path / "cache"),
        "--augment", "full", "--val-distort", "crunch",
    ])
    assert rc == 0
    for name in ("best.pt", "last.pt", "history.csv", "thresholds.json", "test_report.json"):
        assert (run / name).exists(), name
    model, ckpt = load_model(run / "best.pt")
    assert "thresholds" in ckpt and model.cfg.channels == 8
    report = json.loads((run / "test_report.json").read_text())
    assert "none" in report and "crunch" in report

    # resume continues from the saved epoch
    rc = main([
        "--manifest", str(manifest), "--out", str(run), "--epochs", "2", "--steps-per-epoch", "2",
        "--batch-size", "2", "--clip-seconds", "1", "--workers", "0", "--device", "cpu",
        "--channels", "8", "--hidden", "32", "--cache-dir", str(tmp_path / "cache"),
        "--resume", str(run / "last.pt"), "--no-final-test",
    ])
    assert rc == 0
    assert len((run / "history.csv").read_text().strip().splitlines()) == 3  # header + 2 epochs

    # --init starts a new run from those weights; the model size comes from the checkpoint
    rc = main([
        "--manifest", str(manifest), "--out", str(tmp_path / "run2"), "--epochs", "1", "--steps-per-epoch", "1",
        "--batch-size", "2", "--clip-seconds", "1", "--workers", "0", "--device", "cpu",
        "--cache-dir", str(tmp_path / "cache"), "--init", str(run / "best.pt"), "--no-final-test",
    ])
    assert rc == 0
    model, ckpt = load_model(tmp_path / "run2" / "best.pt")
    assert model.cfg.channels == 8 and ckpt["epoch"] == 1


def test_eval_cli_on_checkpoint(tmp_path):
    from fretlift.training.evaluate import main as eval_main
    from fretlift.training.train import main as train_main

    items = synth_dataset(tmp_path / "data", n_tracks=3)
    for it, split in zip(items, ["train", "val", "test"]):
        object.__setattr__(it, "split", split)
    manifest = write_manifest(tmp_path / "data" / "manifest.jsonl", items)
    common = ["--cache-dir", str(tmp_path / "cache"), "--device", "cpu"]
    assert train_main(["--manifest", str(manifest), "--out", str(tmp_path / "run"), "--epochs", "1",
                       "--steps-per-epoch", "2", "--batch-size", "2", "--clip-seconds", "1", "--workers", "0",
                       "--channels", "8", "--hidden", "32", "--no-final-test", *common]) == 0
    out = tmp_path / "report.json"
    assert eval_main(["--manifest", str(manifest), "--split", "test", "--model", str(tmp_path / "run" / "best.pt"),
                      "--distort", "high_gain", "--out", str(out), *common]) == 0
    assert json.loads(out.read_text())["summary"]["all"]["tracks"] == 1


def test_pipeline_transcribes_with_the_installed_model(tmp_path, monkeypatch):
    from fretlift.transcription import transcribe
    from fretlift.tuning import PRESETS

    torch.manual_seed(0)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": TabCRNN(TINY).state_dict(), "model_config": TINY.to_dict()}, ckpt)
    monkeypatch.setenv("FRETLIFT_MODEL", str(ckpt))
    wav = write_wav(tmp_path / "a.wav", 0.3 * np.sin(2 * np.pi * 220 * np.arange(2 * SR) / SR), SR)
    # An untrained model's output is noise, so thresholds this low always give notes.
    notes = transcribe(wav, min_pitch=40, max_pitch=88, onset_threshold=0.01, frame_threshold=0.01,
                       tuning=PRESETS["standard"], device="cpu")
    assert notes and all(40 <= n.pitch <= 88 for n in notes)
    assert any(n.string for n in notes)
