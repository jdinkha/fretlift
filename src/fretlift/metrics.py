"""Note-level transcription metrics.

These follow the definitions in mir_eval's ``transcription`` module (the
standard in transcription papers), so the numbers are comparable:

* A reference and an estimated note **match** if they have the same pitch and
  their onsets are within ``onset_tolerance`` (50 ms by default). With
  ``offset_ratio`` set, offsets must also be within
  ``max(offset_ratio * reference duration, offset_min_tolerance)``.
* Matching is a maximum bipartite matching, so each note is used at most once.
* **Tab F1** additionally requires the same string, which (given the pitch
  and tuning) means the same fret. That's the metric that says whether the tab
  is right, not just the notes.

Pure Python, so the core package stays free of heavy dependencies.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .types import Note

_EPS = 1e-6  # float slack, so a note exactly 50 ms off still matches


@dataclass(frozen=True)
class Score:
    precision: float
    recall: float
    f1: float
    n_ref: int
    n_est: int
    n_match: int

    @classmethod
    def from_counts(cls, n_match: int, n_ref: int, n_est: int) -> Score:
        if n_ref == 0 and n_est == 0:
            return cls(1.0, 1.0, 1.0, 0, 0, 0)
        p = n_match / n_est if n_est else 0.0
        r = n_match / n_ref if n_ref else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return cls(p, r, f, n_ref, n_est, n_match)


def _max_matching(adj: list[list[int]], n_right: int) -> list[tuple[int, int]]:
    """Maximum bipartite matching (Kuhn's algorithm, iterative DFS)."""
    match_l = [-1] * len(adj)
    match_r = [-1] * n_right
    for u in range(len(adj)):
        if not adj[u]:
            continue
        parent: dict[int, int] = {}
        visited: set[int] = set()
        stack = [(u, iter(adj[u]))]
        found = -1
        while stack and found < 0:
            x, it = stack[-1]
            for v in it:
                if v in visited:
                    continue
                visited.add(v)
                parent[v] = x
                if match_r[v] < 0:
                    found = v
                else:
                    stack.append((match_r[v], iter(adj[match_r[v]])))
                break
            else:
                stack.pop()
        v = found
        while v >= 0:  # flip the augmenting path
            x = parent[v]
            nxt = match_l[x]
            match_l[x], match_r[v] = v, x
            v = -1 if x == u else nxt
    return [(i, j) for i, j in enumerate(match_l) if j >= 0]


def match_notes(
    ref: Sequence[Note],
    est: Sequence[Note],
    *,
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = None,
    offset_min_tolerance: float = 0.05,
    match_strings: bool = False,
) -> list[tuple[int, int]]:
    """Return matched (ref index, est index) pairs."""

    def key(n: Note):
        return (round(n.pitch), n.string) if match_strings else round(n.pitch)

    by_key: dict[object, list[tuple[float, int]]] = defaultdict(list)
    for j, n in enumerate(est):
        if match_strings and n.string is None:
            continue
        by_key[key(n)].append((n.start, j))
    starts: dict[object, list[float]] = {}
    for k, lst in by_key.items():
        lst.sort()
        starts[k] = [s for s, _ in lst]

    adj: list[list[int]] = []
    for r in ref:
        cands: list[int] = []
        k = key(r)
        if k in by_key and not (match_strings and r.string is None):
            lo = bisect_left(starts[k], r.start - onset_tolerance - _EPS)
            hi = bisect_right(starts[k], r.start + onset_tolerance + _EPS)
            for _, j in by_key[k][lo:hi]:
                if offset_ratio is not None:
                    tol = max(offset_ratio * (r.end - r.start), offset_min_tolerance)
                    if abs(est[j].end - r.end) > tol + _EPS:
                        continue
                cands.append(j)
        adj.append(cands)
    return _max_matching(adj, len(est))


def note_score(ref: Sequence[Note], est: Sequence[Note], **kwargs) -> Score:
    """Onset + pitch F1 (pass ``offset_ratio=0.2`` to also require offsets)."""
    return Score.from_counts(len(match_notes(ref, est, **kwargs)), len(ref), len(est))


def tab_score(ref: Sequence[Note], est: Sequence[Note], *, onset_tolerance: float = 0.05) -> Score | None:
    """Onset + pitch + string F1. ``None`` if the reference has no string labels."""
    if not ref or any(n.string is None for n in ref):
        return None
    pairs = match_notes(ref, est, onset_tolerance=onset_tolerance, match_strings=True)
    return Score.from_counts(len(pairs), len(ref), len(est))


def string_accuracy(ref: Sequence[Note], est: Sequence[Note], *, onset_tolerance: float = 0.05) -> tuple[int, int]:
    """Among notes found correctly (pitch + onset), how many are on the right string.

    Returns ``(correct, total)``; total counts pairs where both notes have a string.
    This separates "heard the note" from "put it in the right place".
    """
    pairs = match_notes(ref, est, onset_tolerance=onset_tolerance)
    both = [(i, j) for i, j in pairs if ref[i].string is not None and est[j].string is not None]
    return sum(ref[i].string == est[j].string for i, j in both), len(both)
