import json
import zipfile

import numpy as np
import pytest

from fretlift.training.audio import AudioCache, load_audio, write_wav
from fretlift.training.datasets import _fetch_zips, _md5, prepare_folder, prepare_guitar_techs, prepare_guitarset
from fretlift.training.labels import load_labels, save_labels
from fretlift.training.manifest import Item, assign_splits, filter_tags, hash_split, read_manifest
from fretlift.types import Note

from .test_training_labels import jams_doc, write_string_midi


def tone(seconds=1.0, sr=44100, hz=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return 0.3 * np.sin(2 * np.pi * hz * t)


def fake_guitarset(root, players=("00", "01")):
    (root / "annotation").mkdir(parents=True)
    (root / "audio_mono-mic").mkdir()
    (root / "audio_mono-pickup_mix").mkdir()
    for p in players:
        for mode in ("comp", "solo"):
            tid = f"{p}_BN1-129-Eb_{mode}"
            doc = jams_doc([[(0.1, 0.3, 40.0)], [], [], [], [], [(0.5, 0.2, 64.1)]])
            (root / "annotation" / f"{tid}.jams").write_text(json.dumps(doc))
            write_wav(root / "audio_mono-mic" / f"{tid}_mic.wav", tone(), 44100)
            write_wav(root / "audio_mono-pickup_mix" / f"{tid}_mix.wav", tone(), 44100)


def test_prepare_guitarset(tmp_path):
    fake_guitarset(tmp_path)
    items = read_manifest(prepare_guitarset(tmp_path))
    assert len(items) == 8  # 2 players x 2 modes x (mic + pickup)
    assert {it.group for it in items} == {"guitarset-player-00", "guitarset-player-01"}
    mic = [it for it in items if "mic" in it.tags]
    assert len(mic) == 4 and all("acoustic" in it.tags for it in mic)
    notes, tuning = load_labels(mic[0].labels)
    assert [(n.pitch, n.string) for n in notes] == [(40, 6), (64, 1)]
    assert tuning == (40, 45, 50, 55, 59, 64)
    splits = assign_splits(items, val_groups=["guitarset-player-01"])
    assert len(splits["train"]) == 4 and len(splits["val"]) == 4 and not splits["test"]


def test_prepare_guitar_techs(tmp_path):
    part = tmp_path / "P3_music"
    (part / "midi").mkdir(parents=True)
    # 90 on the high e would be fret 26, which a 22-fret guitar can't play.
    write_string_midi(part / "midi" / "midi_01.mid", {"e": [90, 64], "E": [40]})
    for kind in ("directinput", "micamp"):
        write_wav(part / "audio" / kind / f"{kind}_01.wav", tone(), 48000)
    items = read_manifest(prepare_guitar_techs(tmp_path))
    assert {it.id for it in items} == {"guitar-techs/P3_music_01/di", "guitar-techs/P3_music_01/amp"}
    assert {it.group for it in items} == {"guitar-techs-P3"}
    assert all({"electric", "music"} <= set(it.tags) for it in items)
    notes, _ = load_labels(items[0].labels)
    assert sorted((n.pitch, n.string) for n in notes) == [(40, 6), (64, 1)]
    high_e = next(n for n in notes if n.string == 1)
    assert high_e.start == pytest.approx(0.1 - 0.0087)  # moved earlier by the pickup's measured delay


def test_fetch_zips_downloads_in_parallel_and_unpacks(tmp_path):
    jobs = []
    for name in ("a", "b"):
        src = tmp_path / "remote" / f"{name}.zip"
        src.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(src, "w") as z:
            z.writestr(f"{name}/midi/x.mid", "data")
            z.writestr("__MACOSX/junk", "")
        jobs.append((src.as_uri(), tmp_path / "data" / src.name, _md5(src), tmp_path / "data",
                     lambda m: not m.startswith("__MACOSX")))
    (tmp_path / "data").mkdir()
    _fetch_zips(jobs)
    assert (tmp_path / "data" / "a" / "midi" / "x.mid").exists() and (tmp_path / "data" / "b" / "midi" / "x.mid").exists()
    assert not (tmp_path / "data" / "__MACOSX").exists() and not list((tmp_path / "data").glob("*.zip"))
    bad = [(jobs[0][0], tmp_path / "data" / "c.zip", "0" * 32, tmp_path / "data")]
    with pytest.raises(RuntimeError, match="checksum"):
        _fetch_zips(bad)


def test_prepare_folder_eval_set(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "labels").mkdir()
    write_wav(tmp_path / "audio" / "riff1.wav", tone(), 44100)
    write_wav(tmp_path / "audio" / "nolabel.wav", tone(), 44100)
    save_labels(tmp_path / "labels" / "riff1.json", [Note(0, 0.5, 40, string=6)])
    items = read_manifest(prepare_folder(tmp_path / "audio", tmp_path / "labels", tmp_path / "out", "mine",
                                         tags=["electric", "distorted"], split="test", tuning=(38, 45, 50, 55, 59, 64)))
    assert len(items) == 1
    it = items[0]
    assert it.split == "test" and set(it.tags) == {"mine", "electric", "distorted"}
    assert it.tuning == (38, 45, 50, 55, 59, 64)
    assert it.audio.exists() and it.labels.exists()


def test_manifest_relative_paths_survive_moving(tmp_path):
    fake_guitarset(tmp_path / "gs", players=("00",))
    prepare_guitarset(tmp_path / "gs")
    moved = tmp_path / "moved"
    (tmp_path / "gs").rename(moved)
    items = read_manifest(moved / "manifest.jsonl")
    assert all(it.audio.exists() and it.labels.exists() for it in items)


def test_split_rules():
    a = Item("a", None, None, group="g1")
    b = Item("b", None, None, group="g2", split="test")
    c = Item("c", None, None, group="g1", split="test")
    with pytest.raises(ValueError, match="more than one split"):
        assign_splits([a, c])
    out = assign_splits([a, b])
    assert out["train"] == [a] and out["test"] == [b]
    assert hash_split("x") == hash_split("x")
    counts = {s: 0 for s in ("train", "val", "test")}
    for i in range(2000):
        counts[hash_split(str(i))] += 1
    assert 0.07 < counts["test"] / 2000 < 0.13


def test_filter_tags():
    a = Item("a", None, None, "g", tags=("acoustic", "mic"))
    b = Item("b", None, None, "g", tags=("electric",))
    assert filter_tags([a, b], include=["electric"]) == [b]
    assert filter_tags([a, b], exclude=["mic"]) == [b]
    assert filter_tags([a, b]) == [a, b]


def test_audio_resample_and_cache(tmp_path):
    pytest.importorskip("scipy")
    p = write_wav(tmp_path / "t.wav", tone(1.0, 44100, 440.0), 44100)
    x = load_audio(p, 22050)
    assert abs(len(x) - 22050) <= 1
    spectrum = np.abs(np.fft.rfft(x))
    assert abs(np.argmax(spectrum) * 22050 / len(x) - 440) < 2
    cache = AudioCache(tmp_path / "cache", 22050)
    a = cache.load(p)
    assert a.shape == x.shape and np.allclose(a, x, atol=1e-4)
    assert cache.ensure(p) == cache.ensure(p)
