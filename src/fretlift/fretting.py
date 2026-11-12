"""Choose a string and fret for every note.

The same pitch can usually be played in two to four places on the neck, so a
transcription is not a tab until each note has a position. This module scores
candidate fingerings with simple playability costs and picks the cheapest path
through the whole piece with dynamic programming (Viterbi).

The search state is (fingering, hand position), where the hand position is the
fret under the index finger and the hand covers ``max_span`` frets from there.
Tracking the hand explicitly stops the classic failure of note-by-note greedy
methods, where the hand "creeps" up the neck one small step at a time.

* static cost (per chord or note): wide stretches and high positions cost more
* shift cost (between neighbours): changing hand position costs a flat amount
  plus a little per fret travelled; open strings can be played from anywhere

It's a heuristic baseline. Phase 3 of the roadmap replaces or reranks these
costs with a model trained on real tabs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from .tuning import PRESETS
from .types import Event, Note, TabNote

log = logging.getLogger(__name__)

MAX_FRET = 24  # frets 0 (open) to 24: 25 positions per string


@dataclass
class FretConfig:
    tuning: tuple[int, ...] = PRESETS["standard"]  # low -> high
    capo: int = 0
    max_fret: int = MAX_FRET  # highest fret on the neck (absolute, not capo-relative)
    max_span: int = 4  # frets the fretting hand covers comfortably
    w_span: float = 0.3
    w_overstretch: float = 5.0
    w_height: float = 0.05
    w_shift: float = 1.0  # flat cost of moving the hand to a new position
    w_slide: float = 0.15  # extra cost per fret moved
    # ponytail: untuned guess (3 = a lone note isn't worth a hand shift there and back, two notes are).
    # Tune it against real tabs once a trained model exists.
    w_string: float = 3.0  # cost per note placed on a different string than the model heard
    beam: int = 16  # candidate fingerings kept per event

    def __post_init__(self):
        if not 0 <= self.max_fret <= MAX_FRET:
            raise ValueError(f"max_fret must be between 0 and {MAX_FRET}, got {self.max_fret}")


@dataclass(frozen=True)
class _Fingering:
    notes: tuple[TabNote, ...]
    cost: float
    lo: int | None  # lowest fretted fret, None if all open
    hi: int | None


def _string_number(index_low_to_high: int, n_strings: int) -> int:
    return n_strings - index_low_to_high


def _options(pitch: int, cfg: FretConfig) -> list[tuple[int, int]]:
    """(string index low->high, fret) pairs that produce ``pitch``."""
    top = cfg.max_fret - cfg.capo
    out = []
    for i, open_pitch in enumerate(cfg.tuning):
        fret = pitch - (open_pitch + cfg.capo)
        if 0 <= fret <= top:
            out.append((i, fret))
    return out


def _static_cost(frets: list[int], cfg: FretConfig) -> tuple[float, int | None, int | None]:
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0.0, None, None
    lo, hi = min(fretted), max(fretted)
    span = hi - lo
    cost = (
        cfg.w_span * span
        + cfg.w_overstretch * max(0, span - (cfg.max_span - 1))
        + cfg.w_height * (sum(fretted) / len(fretted))
    )
    return cost, lo, hi


def candidates(pitches: list[int], cfg: FretConfig, prefer: dict[int, int] | None = None) -> list[_Fingering]:
    """All ways to play ``pitches`` at once, one note per string, cheapest first.

    ``prefer`` maps pitches to the string the transcription model heard them
    on; other strings stay possible but cost ``cfg.w_string`` each.
    """
    prefer = prefer or {}
    n = len(cfg.tuning)
    playable = sorted({p for p in pitches if _options(p, cfg)})
    dropped = set(pitches) - set(playable)
    if dropped:
        log.debug("Dropping out-of-range pitches %s", sorted(dropped))
    if len(playable) > n:
        # More notes than strings: keep the lowest and the highest ones.
        keep = n // 2
        playable = playable[:keep] + playable[-(n - keep):]
    if not playable:
        return []

    opts = [_options(p, cfg) for p in playable]
    results: list[_Fingering] = []

    def search(k: int, used: set[int], chosen: list[tuple[int, int]]) -> None:
        if k == len(playable):
            frets = [f for _, f in chosen]
            cost, lo, hi = _static_cost(frets, cfg)
            notes = tuple(
                TabNote(pitch=p, string=_string_number(s, n), fret=f)
                for p, (s, f) in zip(playable, chosen)
            )
            cost += cfg.w_string * sum(prefer.get(t.pitch, t.string) != t.string for t in notes)
            results.append(_Fingering(notes, cost, lo, hi))
            return
        for s, f in opts[k]:
            if s not in used:
                used.add(s)
                chosen.append((s, f))
                search(k + 1, used, chosen)
                chosen.pop()
                used.remove(s)

    search(0, set(), [])
    if not results and len(playable) > 1:
        # No way to give every note its own string; drop the middle voice and retry.
        mid = playable[len(playable) // 2]
        return candidates([p for p in playable if p != mid], cfg, prefer)
    results.sort(key=lambda c: c.cost)
    return results[: cfg.beam]


def _shift(p_from: int, p_to: int, cfg: FretConfig) -> float:
    return 0.0 if p_from == p_to else cfg.w_shift + cfg.w_slide * abs(p_to - p_from)


def _hand_positions(c: _Fingering, cfg: FretConfig, n_positions: int) -> list[int]:
    """Index-finger frets from which every fretted note in ``c`` is reachable."""
    if c.lo is None:  # open strings only: playable from any position
        return list(range(1, n_positions + 1))
    first = max(1, c.hi - cfg.max_span + 1)
    if first > c.lo:  # an overstretch (already penalised in the static cost)
        return [min(c.lo, n_positions)]
    return list(range(first, min(c.lo, n_positions) + 1))


def assign_frets(events: list[Event], cfg: FretConfig | None = None) -> list[Event]:
    """Fill in ``event.tab`` for every event. Unplayable events are removed."""
    cfg = cfg or FretConfig()
    cands = [candidates(e.pitches, cfg, e.strings) for e in events]
    kept = [(e, c) for e, c in zip(events, cands) if c]
    if len(kept) < len(events):
        log.info("Skipped %d events with no playable notes", len(events) - len(kept))
    if not kept:
        return []

    n_positions = max(1, cfg.max_fret - cfg.capo - cfg.max_span + 2)
    positions = range(1, n_positions + 1)

    # states[i] = list of (candidate index, hand position); score/back are parallel lists.
    states: list[list[tuple[int, int]]] = []
    scores: list[list[float]] = []
    backs: list[list[int]] = []
    for i, (_, cand) in enumerate(kept):
        st = [(j, p) for j, c in enumerate(cand) for p in _hand_positions(c, cfg, n_positions)]
        if i == 0:
            sc = [cand[j].cost for j, _ in st]
            bk = [-1] * len(st)
        else:
            # Cheapest way to end the previous event at each hand position...
            best_at: dict[int, tuple[float, int]] = {}
            for k, ((_, p), s) in enumerate(zip(states[-1], scores[-1])):
                if p not in best_at or s < best_at[p][0]:
                    best_at[p] = (s, k)
            # ...then the cheapest way to arrive at each new position.
            reach: dict[int, tuple[float, int]] = {}
            for p in positions:
                reach[p] = min(
                    ((s + _shift(q, p, cfg), k) for q, (s, k) in best_at.items()),
                    key=lambda t: t[0],
                )
            sc = [reach[p][0] + cand[j].cost for j, p in st]
            bk = [reach[p][1] for _, p in st]
        states.append(st)
        scores.append(sc)
        backs.append(bk)

    k = min(range(len(scores[-1])), key=scores[-1].__getitem__)
    path = [0] * len(kept)
    for i in range(len(kept) - 1, -1, -1):
        path[i] = states[i][k][0]
        k = backs[i][k]

    out = []
    for (event, cand), j in zip(kept, path):
        chosen = cand[j]
        # Lowest string first, the usual order for chord notes in MusicXML.
        tab = sorted(chosen.notes, key=lambda t: -t.string)
        out.append(Event(tick=event.tick, duration=event.duration, pitches=[t.pitch for t in tab], tab=tab))
    return out


def assign_strings(notes: list[Note], cfg: FretConfig | None = None, chord_tol: float = 0.035) -> list[Note]:
    """Give every note a string with the fingering search, for transcribers that only hear pitches.

    Notes starting within ``chord_tol`` seconds of each other are fingered as
    one chord. Notes the search can't place keep ``string=None``.
    """
    groups: list[list[Note]] = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if groups and n.start - groups[-1][0].start <= chord_tol:
            groups[-1].append(n)
        else:
            groups.append([n])
    events = [Event(tick=i, duration=1, pitches=sorted({n.pitch for n in g})) for i, g in enumerate(groups)]
    placed = {e.tick: {t.pitch: t.string for t in e.tab} for e in assign_frets(events, cfg)}
    return [replace(n, string=placed.get(i, {}).get(n.pitch)) for i, g in enumerate(groups) for n in g]
