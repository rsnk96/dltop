"""Layout tests: domain-grouped charts and GPU show/hide toggles (demo mode)."""

from __future__ import annotations

from dltop.app import DltopApp
from dltop.models import HOST_SERIES, per_gpu
from dltop.widgets.plot import TimeSeriesPlot


def test_per_gpu_suffixes_names_only() -> None:
    defs = per_gpu([("sm", 15, "SM", True)], 1)
    assert defs == [("sm@1", 15, "SM", True)]


def test_host_series_cover_requested_metrics() -> None:
    names = [n for n, _, _, _ in HOST_SERIES["all"]]
    assert names == ["cpu", "ram", "disk_r", "disk_w", "net_rx", "net_tx"]


async def test_every_tab_stacks_host_then_per_gpu_charts() -> None:
    app = DltopApp(interval=0.1, no_dcgm=True, demo_gpus=2, no_discover=True)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        for tab in ("all", "compute", "memory", "system"):
            assert app.query_one(f"#{tab}-host-plot", TimeSeriesPlot)
            assert app.query_one(f"#{tab}-gpu0-plot", TimeSeriesPlot)
            assert app.query_one(f"#{tab}-gpu1-plot", TimeSeriesPlot)


async def test_gpu_toggle_hides_that_gpus_charts_everywhere() -> None:
    app = DltopApp(interval=0.1, no_dcgm=True, demo_gpus=2, no_discover=True)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        checkbox = app.query_one("#gpu-cb-1")
        checkbox.value = False
        await pilot.pause()
        blocks = list(app.query(".gpu-chart-1"))
        assert blocks
        assert all(not w.display for w in blocks)
        blocks0 = list(app.query(".gpu-chart-0"))
        assert blocks0
        assert all(w.display for w in blocks0)


async def test_single_gpu_shows_no_gpu_toggle_row() -> None:
    app = DltopApp(interval=0.1, no_dcgm=True, demo_gpus=1, no_discover=True)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        assert not app.query("GpuToggles")
