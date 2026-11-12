"""Evaluation and threshold sweeps, with a fake predictor (no model needed)."""

from dataclasses import replace

import pytest

from fretlift.training.config import DecodeConfig
from fretlift.training.evaluate import evaluate_items, format_summary, summarize
from fretlift.training.labels import load_labels, save_labels
from fretlift.training.manifest import Item
from fretlift.training.sweep import format_sweep, sweep_thresholds
from fretlift.types import Note


class FakePredictor:
    """Pretends quiet recordings need a low onset threshold and loud ones a high one."""

    name = "fake"

    def predict(self, item, distort, decode: DecodeConfig, tuning, strings=True):
        ref, _ = load_labels(item.labels)
        best = 0.3 if "quiet" in item.tags else 0.6
        miss = abs(decode.onset_threshold - best) > 0.05
        notes = ref[: len(ref) // 2] if miss else list(ref)
        if strings and "wrong-strings" in item.tags:
            notes = [replace(n, string=(n.string % 6) + 1) for n in notes]
        return notes


@pytest.fixture
def items(tmp_path):
    out = []
    for name, tags in [("a", ("quiet",)), ("b", ("quiet", "wrong-strings")), ("c", ("loud",))]:
        notes = [Note(i * 0.5, i * 0.5 + 0.4, 52 + i % 5, string=3) for i in range(10)]
        lab = save_labels(tmp_path / f"{name}.json", notes)
        out.append(Item(name, tmp_path / f"{name}.wav", lab, group=name, tags=tags))
    return out


def test_evaluate_and_summarize(items):
    results = evaluate_items(items, FakePredictor(), decode=DecodeConfig(onset_threshold=0.3))
    s = summarize(results)
    assert s["quiet"]["note_f1"] == 1.0
    assert s["quiet"]["tab_f1"] == pytest.approx(0.5)  # b has right notes, wrong strings
    assert s["loud"]["note_f1"] < 1.0
    assert s["all"]["tracks"] == 3
    table = format_summary(s, "title")
    assert "quiet" in table and "tab F1" in table


def test_sweep_finds_per_tag_settings(items):
    result = sweep_thresholds(items, FakePredictor(), onsets=(0.3, 0.6), frames=(0.3,), progress=False)
    assert result["per_tag"]["quiet"]["onset_threshold"] == 0.3
    assert result["per_tag"]["loud"]["onset_threshold"] == 0.6
    assert result["all"]["onset_threshold"] == 0.3  # 2 of 3 tracks are quiet
    assert result["per_tag"]["loud"]["f1_lost_with_overall_setting"] > 0.2
    assert "worth a separate setting" in format_sweep(result)


def test_out_of_range_predictions_are_dropped(items, tmp_path):
    class Low(FakePredictor):
        def predict(self, *a, **k):
            return super().predict(*a, **k) + [Note(0, 1, 20)]

    (r,) = evaluate_items(items[:1], Low(), decode=DecodeConfig(onset_threshold=0.3))
    assert r.note.precision == 1.0
