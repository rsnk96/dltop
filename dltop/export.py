"""Serialize a stats snapshot to Markdown / HTML / TSV / metadata text.

Pure functions over plain dataclasses so they are trivially unit-testable and
usable outside the TUI (e.g. a future --oneshot CLI mode).
"""

from __future__ import annotations

import html
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dltop.metrics import WindowStats

_HEADERS = ("Metric", "Source", "Now", "Mean", "Median", "Stddev", "Unit")


@dataclass(frozen=True)
class StatRow:
    """One Table-tab row: a metric with its windowed statistics."""

    metric: str
    source: str
    unit: str
    stats: WindowStats


@dataclass(frozen=True)
class GpuMeta:
    """Hardware identity of one GPU for the metadata export."""

    index: int
    name: str
    vram_gib: float


@dataclass(frozen=True)
class CaptureMeta:
    """Everything needed to make a pasted stats table self-describing."""

    captured_at: str
    version: str
    window_s: float
    n_samples: int
    interval_s: float
    hostname: str
    os_desc: str
    cpu_desc: str
    ram_gib: float
    gpus: list[GpuMeta]
    driver: str
    cuda: str
    prom_endpoints: list[str]


def _num(v: float) -> str:
    """Format a stat value: one decimal, no trailing unit."""
    return f"{v:.1f}"


def _cells(row: StatRow) -> list[str]:
    s = row.stats
    return [row.metric, row.source, _num(s.now), _num(s.mean), _num(s.median), _num(s.stddev), row.unit]


def _window_note(window_s: float) -> str:
    return f"window: last {window_s:.0f} s"


def to_markdown(rows: Sequence[StatRow], window_s: float) -> str:
    """Render rows as a GitHub-flavoured Markdown table with a window note."""
    out = [
        "| " + " | ".join(_HEADERS) + " |",
        "| " + " | ".join("---" for _ in _HEADERS) + " |",
    ]
    out.extend("| " + " | ".join(_cells(r)) + " |" for r in rows)
    out.append("")
    out.append(f"_Mean/Median/Stddev {_window_note(window_s)}_")
    return "\n".join(out)


def to_html(rows: Sequence[StatRow], window_s: float) -> str:
    """Render rows as a minimal HTML table (paste target: docs, wikis, email)."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in _HEADERS)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in _cells(r)) + "</tr>" for r in rows)
    caption = html.escape(f"Mean/Median/Stddev {_window_note(window_s)}")
    return f"<table><caption>{caption}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def to_tsv(rows: Sequence[StatRow], window_s: float) -> str:
    """Render rows tab-separated (paste target: Excel / Google Sheets)."""
    out = ["\t".join(_HEADERS)]
    out.extend("\t".join(_cells(r)) for r in rows)
    out.append("")
    out.append(f"Mean/Median/Stddev {_window_note(window_s)}")
    return "\n".join(out)


def to_metadata_markdown(meta: CaptureMeta) -> str:
    """Render the capture context block described in the design (§3)."""
    lines = [
        "## dltop capture metadata",
        f"- captured:   {meta.captured_at} · dltop {meta.version}",
        f"- window:     last {meta.window_s:.0f} s · {meta.n_samples} samples @ {meta.interval_s:g} s interval",
        f"- host:       {meta.hostname} · {meta.os_desc}",
        f"- cpu:        {meta.cpu_desc}",
        f"- ram:        {meta.ram_gib:.1f} GiB",
    ]
    lines.extend(
        f"- gpu {g.index}:      {g.name} · {g.vram_gib:.0f} GiB · driver {meta.driver} · CUDA {meta.cuda}"
        for g in meta.gpus
    )
    if meta.prom_endpoints:
        lines.append("- prometheus: " + ", ".join(meta.prom_endpoints))
    return "\n".join(lines)


def host_cpu_desc() -> str:
    """Best-effort human CPU description, e.g. ``AMD Ryzen 9 7940HS``."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"
