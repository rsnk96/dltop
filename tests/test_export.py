"""Pure-function tests for the four exporters (GPU-less)."""

from __future__ import annotations

from dltop.export import CaptureMeta, GpuMeta, StatRow, to_html, to_markdown, to_metadata_markdown, to_tsv
from dltop.metrics import WindowStats

ROWS = [
    StatRow("CPU", "host", "%", WindowStats(now=38.2, mean=41.0, median=40.1, stddev=3.2, n_samples=120)),
    StatRow("GPU SM", "GPU 0", "%", WindowStats(now=82.4, mean=76.9, median=78.2, stddev=6.1, n_samples=120)),
    StatRow("num_requests_running", ":8000", "", WindowStats(now=3.0, mean=2.6, median=3.0, stddev=0.9, n_samples=55)),
]


def test_markdown_shape() -> None:
    md = to_markdown(ROWS, window_s=60.0)
    lines = md.strip().splitlines()
    assert lines[0].startswith("| Metric | Source | Now | Mean | Median | Stddev | Unit |")
    assert lines[1].startswith("| ---")
    assert "| CPU | host | 38.2 | 41.0 | 40.1 | 3.2 | % |" in md
    assert "window: last 60 s" in md


def test_tsv_is_tab_separated_with_header() -> None:
    tsv = to_tsv(ROWS, window_s=60.0)
    lines = tsv.strip().splitlines()
    assert lines[0].split("\t") == ["Metric", "Source", "Now", "Mean", "Median", "Stddev", "Unit"]
    assert lines[1].split("\t")[0:3] == ["CPU", "host", "38.2"]


def test_html_is_a_table_with_escaping() -> None:
    rows = [StatRow("a<b", "host", "", WindowStats(1.0, 1.0, 1.0, 0.0, 1))]
    html = to_html(rows, window_s=60.0)
    assert html.startswith("<table>")
    assert "a&lt;b" in html
    assert "</table>" in html


def test_metadata_markdown_contains_context() -> None:
    meta = CaptureMeta(
        captured_at="2026-07-07 12:02:10",
        version="0.2.0",
        window_s=60.0,
        n_samples=120,
        interval_s=0.5,
        hostname="gpu-box-03",
        os_desc="Ubuntu 24.04 (6.17.0-35-generic)",
        cpu_desc="Ryzen 5975WX · 32C/64T",
        ram_gib=503.5,
        gpus=[GpuMeta(0, "RTX 4090", 24.0)],
        driver="560.35.03",
        cuda="12.6",
        prom_endpoints=[":8000 (vllm, 28 metrics)"],
    )
    md = to_metadata_markdown(meta)
    for needle in (
        "dltop capture metadata",
        "last 60 s",
        "120 samples",
        "0.5 s interval",
        "gpu-box-03",
        "Ryzen",
        "RTX 4090",
        "560.35.03",
        "CUDA 12.6",
        ":8000",
    ):
        assert needle in md
