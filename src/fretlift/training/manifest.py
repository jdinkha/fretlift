"""Dataset manifests: one JSON line per recording.

Every dataset (GuitarSet, EGDB, your own recordings, separated mixes) is
converted to the same shape, so training and evaluation never need to know
where data came from::

    {"id": "guitarset/00_BN1-129-Eb_comp/mic", "audio": "audio_mono-mic/....wav",
     "labels": "labels/00_BN1-129-Eb_comp.json", "group": "00",
     "tags": ["guitarset", "acoustic", "mic", "comp"], "split": null, "tuning": [40, ...]}

``group`` keeps related recordings on the same side of a split: all six
GuitarSet excerpts by one player, or every amp render of one EGDB take.
Splitting by group is what stops a model being tested on audio it has
effectively already heard.

Paths are stored relative to the manifest file.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Item:
    id: str
    audio: Path
    labels: Path
    group: str
    tags: tuple[str, ...] = ()
    split: str | None = None  # fixed split; None = decide from group at training time
    tuning: tuple[int, ...] | None = None


def read_manifest(path: str | Path) -> list[Item]:
    path = Path(path)
    base = path.parent
    items = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: {e}") from None
        items.append(
            Item(
                id=d["id"],
                audio=(base / d["audio"]).resolve(),
                labels=(base / d["labels"]).resolve(),
                group=str(d.get("group", d["id"])),
                tags=tuple(d.get("tags", ())),
                split=d.get("split"),
                tuning=tuple(d["tuning"]) if d.get("tuning") else None,
            )
        )
    return items


def write_manifest(path: str | Path, items: Iterable[Item]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent.resolve()

    def rel(p: Path) -> str:
        return Path(os.path.relpath(Path(p).resolve(), base)).as_posix()

    lines = []
    for it in items:
        d = {"id": it.id, "audio": rel(it.audio), "labels": rel(it.labels), "group": it.group, "tags": list(it.tags)}
        if it.split:
            d["split"] = it.split
        if it.tuning:
            d["tuning"] = list(it.tuning)
        lines.append(json.dumps(d))
    path.write_text("\n".join(lines) + "\n")
    return path


def read_manifests(paths: Sequence[str | Path]) -> list[Item]:
    items: list[Item] = []
    for p in paths:
        items.extend(read_manifest(p))
    ids = [it.id for it in items]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(f"Duplicate item ids across manifests, e.g. {dupes}")
    return items


def hash_split(key: str, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    """Deterministic split from a string, so it never changes between runs."""
    x = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if x < test_frac:
        return "test"
    if x < test_frac + val_frac:
        return "val"
    return "train"


def assign_splits(
    items: Sequence[Item], val_groups: Iterable[str] = (), test_groups: Iterable[str] = ()
) -> dict[str, list[Item]]:
    """Items with a fixed ``split`` keep it; others go by group (train by default)."""
    val_groups, test_groups = set(val_groups), set(test_groups)
    if val_groups & test_groups:
        raise ValueError(f"Groups can't be in both val and test: {sorted(val_groups & test_groups)}")
    out: dict[str, list[Item]] = {s: [] for s in SPLITS}
    for it in items:
        if it.split:
            if it.split not in out:
                raise ValueError(f"{it.id}: unknown split {it.split!r}")
            out[it.split].append(it)
        elif it.group in test_groups:
            out["test"].append(it)
        elif it.group in val_groups:
            out["val"].append(it)
        else:
            out["train"].append(it)
    # Guard against leakage through mislabelled fixed splits.
    seen: dict[str, set[str]] = defaultdict(set)
    for split, lst in out.items():
        for it in lst:
            seen[it.group].add(split)
    leaks = [g for g, s in seen.items() if len(s) > 1]
    if leaks:
        raise ValueError(f"Groups appear in more than one split: {sorted(leaks)[:5]}")
    return out


def filter_tags(items: Iterable[Item], include: Iterable[str] = (), exclude: Iterable[str] = ()) -> list[Item]:
    """Keep items with any of ``include`` (all if empty) and none of ``exclude``."""
    include, exclude = set(include), set(exclude)
    return [it for it in items if (not include or include & set(it.tags)) and not exclude & set(it.tags)]
