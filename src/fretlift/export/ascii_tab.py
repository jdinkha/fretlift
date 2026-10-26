"""Plain-text tab, handy for the terminal and for quick diffs in tests."""

from __future__ import annotations

from pathlib import Path

from ..measures import layout
from ..tuning import SHARP_NAMES
from ..types import Event


def to_ascii(
    events: list[Event],
    *,
    tuning: tuple[int, ...],
    capo: int = 0,
    tempo: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
    subdivision: int = 4,
    title: str = "Untitled",
    measures_per_line: int = 4,
) -> str:
    n = len(tuning)
    # Row 0 is string 1 (highest), printed at the top like a normal tab.
    labels = [SHARP_NAMES[tuning[n - 1 - r] % 12] for r in range(n)]
    labels = [lab.lower() if r == 0 and lab == "E" else lab for r, lab in enumerate(labels)]
    width = max(len(lab) for lab in labels)

    rendered: list[list[str]] = []
    for segments in layout(events, time_signature, subdivision):
        rows = [""] * n
        for seg in segments:
            for k in range(seg.duration):
                cells = ["-"] * n
                if seg.event is not None and k == 0 and not seg.tie_stop:
                    for tab in seg.event.tab:
                        cells[tab.string - 1] = str(tab.fret)
                cw = max(2, max(len(c) for c in cells) + 1)
                for r in range(n):
                    rows[r] += cells[r].ljust(cw, "-")
        rendered.append(rows)

    header = [f"{title}", f"Tempo: {round(tempo)} BPM   Time: {time_signature[0]}/{time_signature[1]}"]
    tuning_names = " ".join(SHARP_NAMES[p % 12] for p in tuning)
    header.append(f"Tuning: {tuning_names}" + (f"   Capo: {capo}" if capo else ""))
    lines = header + [""]
    for start in range(0, len(rendered), measures_per_line):
        chunk = rendered[start : start + measures_per_line]
        for r in range(n):
            lines.append(labels[r].ljust(width) + "|" + "|".join(m[r] for m in chunk) + "|")
        lines.append("")
    return "\n".join(lines)


def write_ascii(path: str | Path, events: list[Event], **kwargs) -> Path:
    path = Path(path)
    path.write_text(to_ascii(events, **kwargs), encoding="utf-8")
    return path
