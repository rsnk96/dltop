"""Headless rendering tests for TimeSeriesPlot backed by MetricStore (GPU-less)."""

from __future__ import annotations

from textual.app import App, ComposeResult

from dltop.metrics import MetricStore
from dltop.widgets.plot import TimeSeriesPlot

SERIES = [("cpu", 208, "CPU", True), ("ram", 46, "RAM", False)]


class PlotApp(App):
    """Minimal Textual app hosting one ``TimeSeriesPlot`` bound to a given store."""

    def __init__(self, store: MetricStore) -> None:
        """Store the ``MetricStore`` the composed plot will read from."""
        super().__init__()
        self._store = store

    def compose(self) -> ComposeResult:
        """Yield a single plot bound to ``self._store``."""
        yield TimeSeriesPlot(self._store, SERIES, "test chart", plot_id="p")


async def test_plot_renders_store_data() -> None:
    store = MetricStore()
    for i in range(50):
        store.record("cpu", 50.0 + (i % 5), ts=1000.0 + i)
    app = PlotApp(store)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        plot = app.query_one("#p", TimeSeriesPlot)
        rendered = plot._build_chart().plain  # rich Text of the full chart
        assert "test chart" in rendered
        assert any(g in rendered for g in "─╭╮╰╯│")  # line strokes drawn from store data
        plot.set_visible("cpu", visible=False)
        await pilot.pause()


async def test_plot_empty_store_renders_without_crash() -> None:
    app = PlotApp(MetricStore())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        plot = app.query_one("#p", TimeSeriesPlot)
        assert "test chart" in plot._build_chart().plain
