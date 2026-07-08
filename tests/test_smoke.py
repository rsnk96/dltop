"""Smoke tests for dltop.

Runnable on a GPU-less CI host: we never reach `init_nvml()` because `--help`
causes argparse to exit(0) before anything else runs. Tests that need real GPU
state belong in a separate integration suite, not here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_module_imports() -> None:
    """Importing the module must not touch NVML or spawn subprocesses."""
    import dltop

    assert hasattr(dltop, "main")
    assert callable(dltop.main)


def test_cli_help_via_entrypoint() -> None:
    """The installed `dltop` console script responds to --help cleanly."""
    result = subprocess.run(
        ["dltop", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--interval" in result.stdout
    assert "--no-dcgm" in result.stdout


def test_cli_help_via_module() -> None:
    """`python -m dltop --help` also works (invariant for editable installs)."""
    result = subprocess.run(
        [sys.executable, "-m", "dltop", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--interval" in result.stdout


def test_readme_exists_and_is_non_trivial() -> None:
    """Guards pyproject.toml's `readme = "README.md"` claim against file loss."""
    readme = Path(__file__).resolve().parent.parent / "README.md"
    assert readme.is_file(), f"missing {readme}"
    assert len(readme.read_text()) > 200


def test_public_constants_are_plausible() -> None:
    """Sanity-check DCGM field constants -- catches accidental list/dict drift."""
    from dltop.sources.dcgm import DCGM_FIELD_NAMES, DCGM_FIELD_ORDER

    assert len(DCGM_FIELD_ORDER) == len(DCGM_FIELD_NAMES)
    assert set(DCGM_FIELD_ORDER) == set(DCGM_FIELD_NAMES.keys())
    # Names referenced elsewhere in the rendering code -- keep them stable.
    expected_names = {"sm_active", "tensor_active", "fp32_active", "fp16_active", "fp64_active"}
    assert set(DCGM_FIELD_NAMES.values()) == expected_names


def test_all_tab_combines_every_series_with_four_active() -> None:
    """The default 'All' tab overlays every renderable series but enables only four.

    WHY: 'All' is the first/default screen. If a future edit broke the concatenation
    of the three tabs (dropping or duplicating a series) or changed the active set,
    the user's very first view would silently lose a metric or open cluttered with
    all 18 lines on. This pins the contract: every series present exactly once, and
    only CPU / RAM / GPU SM / GPU VRAM visible by default.
    """
    from dltop.app import DltopApp
    from dltop.models import COMPUTE_SERIES_DCGM, MEMORY_SERIES, SYSTEM_SERIES

    app = DltopApp(interval=0.5, no_dcgm=True)
    app._compute_series = COMPUTE_SERIES_DCGM  # set by _prepare_sources() at compose time
    series = app._all_series()
    names = [name for name, _, _, _ in series]

    expected = {name for name, _, _, _ in (*COMPUTE_SERIES_DCGM, *MEMORY_SERIES, *SYSTEM_SERIES)}
    assert set(names) == expected, "All tab must contain every renderable series"
    assert len(names) == len(set(names)), "no series may appear twice"
    on_by_default = {name for name, _, _, default in series if default}
    assert on_by_default == {"cpu", "ram", "sm", "vram"}


def test_clamp_pct_handles_nan_and_range() -> None:
    """Regression guard for the NaN check -- PLR0124 rewrite used math.isnan."""
    import math

    from dltop.models import _clamp_pct

    assert _clamp_pct(float("nan")) == 0.0
    assert _clamp_pct(-5.0) == 0.0
    assert _clamp_pct(150.0) == 100.0
    assert _clamp_pct(42.0) == 42.0
    assert not math.isnan(_clamp_pct(float("nan")))


def test_timeseries_plot_renders_before_first_push() -> None:
    """Textual can paint TimeSeriesPlot during initial layout, before the store has any data.

    WHY: the widget is composed inside a TabbedContent and gets a paint pass while
    its backing store is still empty. If rasterising empty series raised (the old
    plotext empty-state IndexError did, on short canvases), the whole TUI dies on
    the first frame. This guards the empty-data path: it must yield a valid canvas
    and a renderable, never an exception.
    """
    from rich.text import Text

    from dltop.metrics import MetricStore
    from dltop.models import COMPUTE_SERIES_DCGM
    from dltop.widgets.plot import TimeSeriesPlot

    plot = TimeSeriesPlot(MetricStore(), COMPUTE_SERIES_DCGM, "test", "compute-plot")
    # Pure rasterisation must not raise with empty ring buffers...
    glyphs, owners = plot._rasterize(plot._visible_series(), 40, 40)
    assert isinstance(glyphs, dict)
    assert isinstance(owners, dict)
    # ...and render() must return a Rich renderable rather than blow up.
    assert isinstance(plot.render(), Text)


def test_render_never_crashes_the_tui() -> None:
    """``TimeSeriesPlot.render`` must swallow any draw error and return a placeholder.

    WHY: render runs every ~0.5 s for the life of the process. A single malformed
    frame (a transient size of 0, a bad sample, a library edge case) must never
    propagate out of render() — that would tear down the entire monitor. The
    placeholder keeps the chart slot alive so the next frame can recover.
    """
    from rich.text import Text

    from dltop.metrics import MetricStore
    from dltop.models import COMPUTE_SERIES_DCGM
    from dltop.widgets.plot import TimeSeriesPlot

    plot = TimeSeriesPlot(MetricStore(), COMPUTE_SERIES_DCGM, "boom", "compute-plot")

    def explode() -> object:
        msg = "simulated draw failure"
        raise RuntimeError(msg)

    plot._build_chart = explode  # type: ignore[method-assign]
    result = plot.render()
    assert isinstance(result, Text)
    assert "boom" in str(result)


def test_overlapping_series_are_never_hidden() -> None:
    """Two series at the same value must BOTH survive on the shared line canvas.

    WHY: a terminal cell holds one colour, so naive last-writer-wins overlaying
    makes one of two coincident lines vanish — the classic failure where four idle
    engines all sit at 0% but only one colour shows. The renderer instead records
    every series passing through a cell in ``owners`` and interleaves their colours
    by column, so none is hidden while each keeps its true value. If a future edit
    drops a coincident series from ``owners`` (e.g. overwrites instead of appends),
    a user could no longer tell "NVDEC idle" from "all engines but one idle".
    """
    import time

    from dltop.metrics import MetricStore
    from dltop.models import COMPUTE_SERIES_NVML
    from dltop.widgets.plot import TimeSeriesPlot

    store = MetricStore()
    plot = TimeSeriesPlot(store, COMPUTE_SERIES_NVML, "test", "compute-plot")
    names = [s[0] for s in COMPUTE_SERIES_NVML]
    now = time.time()
    # First two series share the exact same value; the rest sit at 0%.
    values = {names[0]: 50.0, names[1]: 50.0, names[2]: 0.0, names[3]: 0.0}
    for name, val in values.items():
        store.record(name, val, ts=now)
        store.record(name, val, ts=now + 1)

    vis = plot._visible_series()
    idx = {name: i for i, (name, _, _) in enumerate(vis)}
    _glyphs, owners = plot._rasterize(vis, 60, 40)

    # Both equal-valued series must appear somewhere on the canvas (neither dropped).
    drawn = set().union(*owners.values()) if owners else set()
    assert idx[names[0]] in drawn, "first coincident series was hidden"
    assert idx[names[1]] in drawn, "second coincident series was hidden"
    # And at least one shared cell must list BOTH, so colour interleaving can show them.
    assert any(
        idx[names[0]] in holders and idx[names[1]] in holders for holders in owners.values()
    ), "coincident series never share a cell -> interleaving cannot reveal both"
