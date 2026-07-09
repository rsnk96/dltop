"""Multi-series box-drawing time-series chart with colour interleaving."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from rich.text import Text
from textual.widget import Widget

if TYPE_CHECKING:
    from rich.console import RenderableType

    from dltop.metrics import MetricStore
    from dltop.models import SeriesDef

AXIS_STYLE = "color(244)"  # dim grey for axis ticks and labels
START_MARKER_GLYPH = "┊"  # dotted vertical rule marking when dltop started logging
START_MARKER_STYLE = "dim color(244)"


class TimeSeriesPlot(Widget):
    """Multi-series overlaid time-series chart with interleaved colours.

    Each series' history lives in a shared ``MetricStore`` ring buffer of
    ``(wall_timestamp, value)`` points, written by the app's sampling loop.
    Every paint rasterises all visible series as box-drawing strokes
    (``─ ╭ ╮ ╰ ╯ │``, the crisp continuous look of nvtop/asciichart) onto one
    shared character canvas at their *true* vertical positions.

    A terminal cell can hold only one foreground colour, painted by whichever
    series draws into it last — so plain overlaying hides every line but the
    last whenever two share a cell (e.g. several idle engines all at 0%). We
    instead **interleave**: every series passing through a cell is recorded, the
    line shape stays continuous, but the cell's colour rotates between those
    series by column index. No line is ever fully hidden, and — unlike a
    vertical offset — every line keeps its real value, so the y-axis still
    means what it says.
    """

    DEFAULT_CSS = """
    TimeSeriesPlot {
        height: 1fr;
        min-height: 14;
    }
    """

    def __init__(
        self,
        store: MetricStore,
        series: list[SeriesDef],
        chart_title: str,
        plot_id: str,
        *,
        scale_to_peak: bool = False,
    ) -> None:
        """Build a chart over ``store`` for ``series`` (name, ansi256_idx, label, default_visible)."""
        super().__init__(id=plot_id)
        self._store = store
        self._series_defs = series
        self._visible: dict[str, bool] = {name: default for name, _, _, default in series}
        self._chart_title = chart_title
        self._y_max = 100.0
        self._scale_to_peak = scale_to_peak

    def set_visible(self, name: str, *, visible: bool) -> None:
        """Toggle a series on/off and force an immediate redraw."""
        if name in self._visible:
            self._visible[name] = visible
            self.refresh()

    def set_y_max(self, y_max: float) -> None:
        """Override the vertical upper bound (used when series are MB/s, not %)."""
        self._y_max = max(1.0, y_max)

    def replot(self) -> None:
        """Public redraw entry point; call after any batch of ``store.record*`` calls."""
        self.refresh()

    # -- rendering -------------------------------------------------------------

    def render(self) -> RenderableType:
        """Render the chart, never letting a draw glitch kill the whole monitor."""
        try:
            return self._build_chart()
        except Exception:  # noqa: BLE001 - a chart must never crash the TUI
            return Text(f"[{self._chart_title}: rendering…]", style="dim")

    def _visible_series(self) -> list[tuple[str, int, str]]:
        """Return ``(name, colour, label)`` for every currently-visible series."""
        return [(n, c, lbl) for (n, c, lbl, _) in self._series_defs if self._visible.get(n)]

    def _visible_slice(self, name: str, width: int) -> tuple[list[tuple[float, float]], int]:
        """Return the last ``width`` samples for ``name`` plus their left column offset.

        One buffer slot per display column, no resampling -- ``offset`` is how many
        columns on the left are still blank (0 once ``name`` has ``width`` samples).
        """
        buf = self._store.tail(name, width)
        return buf, width - len(buf)

    def _marker_column(self, plot_w: int) -> int | None:
        """Column of the "logging started" marker, or ``None`` once the chart is full."""
        if not self._series_defs:
            return None
        _, offset = self._visible_slice(self._series_defs[0][0], plot_w)
        return offset if offset > 0 else None

    @staticmethod
    def _mark(
        glyphs: dict[tuple[int, int], str],
        owners: dict[tuple[int, int], list[int]],
        cell: tuple[int, int],
        glyph: str,
        sidx: int,
    ) -> None:
        """Place ``glyph`` in ``cell`` and record ``sidx`` as passing through it.

        A corner/vertical wins over a plain "─" so a line crossing a flat run
        stays visually connected; the series index is always appended (even when
        the glyph isn't overwritten) so colour interleaving still sees every owner.
        """
        existing = glyphs.get(cell)
        if existing is None or existing == "─":
            glyphs[cell] = glyph
        holders = owners.setdefault(cell, [])
        if sidx not in holders:
            holders.append(sidx)

    def _mark_edge(
        self,
        glyphs: dict[tuple[int, int], str],
        owners: dict[tuple[int, int], list[int]],
        cell: tuple[int, int],
        prev: int | None,
        sidx: int,
    ) -> None:
        """Draw one column of the box-drawing line edge: "─", "│", corners "╭╮╰╯"."""
        cx, cy = cell
        if prev is None or prev == cy:
            self._mark(glyphs, owners, (cx, cy), "─", sidx)
        elif cy < prev:  # line rising: corner up on the left, down-turn at the top
            self._mark(glyphs, owners, (cx, prev), "╯", sidx)
            self._mark(glyphs, owners, (cx, cy), "╭", sidx)
            for yy in range(cy + 1, prev):
                self._mark(glyphs, owners, (cx, yy), "│", sidx)
        else:  # line falling
            self._mark(glyphs, owners, (cx, prev), "╮", sidx)
            self._mark(glyphs, owners, (cx, cy), "╰", sidx)
            for yy in range(prev + 1, cy):
                self._mark(glyphs, owners, (cx, yy), "│", sidx)

    def _rasterize(
        self,
        vis: list[tuple[str, int, str]],
        plot_w: int,
        plot_h: int,
    ) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], list[int]]]:
        """Plot every visible series as box-drawing strokes (the nvtop/asciichart look).

        One glyph per character cell: "─" for a flat step, "│" for a vertical run,
        and rounded corners "╭ ╮ ╰ ╯" where the line turns. Returns ``(glyphs,
        owners)`` keyed by cell: ``glyphs`` is the line shape and ``owners`` is the
        ordered list of series indices passing through each cell, which drives the
        colour interleaving in :meth:`_build_chart`.
        """
        glyphs: dict[tuple[int, int], str] = {}
        owners: dict[tuple[int, int], list[int]] = {}
        rows = max(1, plot_h - 1)
        for sidx, (name, _, _) in enumerate(vis):
            buf, offset = self._visible_slice(name, plot_w)
            y_max = self._y_max
            if self._scale_to_peak:
                peak = max((v for _, v in buf if math.isfinite(v)), default=0.0)
                y_max = peak if peak > 0 else 1.0
            prev: int | None = None
            for cx in range(plot_w):
                if cx < offset:  # not enough history yet to reach this column -- leave blank
                    prev = None
                    continue
                _, raw = buf[cx - offset]
                value = raw if math.isfinite(raw) else 0.0
                frac = min(1.0, max(0.0, value / y_max))
                cy = round((1.0 - frac) * rows)
                self._mark_edge(glyphs, owners, (cx, cy), prev, sidx)
                prev = cy
        return glyphs, owners

    def _build_chart(self) -> RenderableType:
        width = self.size.width or 80
        height = self.size.height or 14
        title = Text(self._chart_title, style="bold", justify="center")
        if width < 12 or height < 5:  # too small to draw a chart
            return title
        gutter = 6  # width of the "100%┤ " y-axis label column (values are percentages)
        plot_w = max(8, width - gutter)
        plot_h = max(3, height - 2)  # leave a row for the title and the time axis
        vis = self._visible_series()
        glyphs, owners = self._rasterize(vis, plot_w, plot_h)
        colours = [c for _, c, _ in vis]
        marker_cx = self._marker_column(plot_w)

        # In-chart legend in the top-left corner: one "● label" per visible series,
        # in the series colour, overlaying the plot (as plotext's legend did). Keep
        # the chart at least 16 columns wide; if labels are too long for that, drop
        # the legend rather than crowd out the data.
        labels = [lbl for _, _, lbl in vis]
        legend_w = 3 + max((len(lbl) for lbl in labels), default=0)  # " ● " + label
        n_legend = min(len(vis), plot_h) if vis and plot_w - legend_w >= 16 else 0

        lines: list[Text] = [title]
        for cy in range(plot_h):
            line = Text()
            if cy % 2 == 0 and plot_h > 1:
                # Percent unit lives on each tick label, e.g. "100%│", so the axis is
                # self-describing. Use the same "│" as the unlabelled rows (not a "┤"
                # tick) so the axis is one straight vertical line — a "┤" glyph renders
                # its stroke slightly right of "│", making the axis bump into the chart.
                if self._scale_to_peak:
                    line.append(f"{100 * (1 - cy / (plot_h - 1)):3.0f}ᵖ│ ", style=AXIS_STYLE)
                else:
                    line.append(f"{self._y_max * (1 - cy / (plot_h - 1)):3.0f}%│ ", style=AXIS_STYLE)
            else:
                line.append("    │ ", style=AXIS_STYLE)
            start_cx = 0
            if cy < n_legend:
                _, colour, label = vis[cy]
                line.append(" ")  # small inset from the axis
                line.append(f"● {label}", style=f"color({colour})")
                line.append(" " * (legend_w - 3 - len(label)))  # pad to align the column
                start_cx = legend_w
            for cx in range(start_cx, plot_w):
                glyph = glyphs.get((cx, cy))
                if not glyph:
                    if cx == marker_cx:
                        line.append(START_MARKER_GLYPH, style=START_MARKER_STYLE)
                    else:
                        line.append(" ")
                    continue
                holders = owners[(cx, cy)]
                winner = holders[cx % len(holders)]  # rotate colour by column -> interleave
                line.append(glyph, style=f"color({colours[winner]})")
            lines.append(line)
        lines.append(self._time_axis(width, gutter, plot_w))
        return Text("\n").join(lines)

    def _time_axis(self, width: int, gutter: int, plot_w: int) -> Text:
        """Bottom axis: oldest *visible* sample time on the left, newest on the right.

        The ring buffer holds far more history than a chart displays (see HISTORY_LEN),
        so this must read the same on-screen slice `_rasterize` draws, not the whole
        buffer -- otherwise the left label stays pinned to app start long after the
        buffer has scrolled well past what's actually on screen.
        """
        row = [" "] * width
        if self._series_defs:
            buf, _ = self._visible_slice(self._series_defs[0][0], plot_w)
            if buf:
                lo = time.strftime("%H:%M:%S", time.localtime(buf[0][0]))
                hi = time.strftime("%H:%M:%S", time.localtime(buf[-1][0]))
                for i, ch in enumerate(lo):
                    if gutter + i < width:
                        row[gutter + i] = ch
                for i, ch in enumerate(hi):
                    if 0 <= width - len(hi) + i < width:
                        row[width - len(hi) + i] = ch
        return Text("".join(row), style=AXIS_STYLE)
