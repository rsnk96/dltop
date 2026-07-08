"""Table tab: instantaneous + windowed stats for every series, with copy-out."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Static

from dltop.export import to_html, to_markdown, to_metadata_markdown, to_tsv

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.app import ComposeResult

    from dltop.export import CaptureMeta, StatRow

# Fixed column widths so the merged "window" band above the stat columns aligns.
_W_METRIC, _W_SOURCE, _W_STAT = 26, 8, 9
_STAT_COLS = ("Now", "Mean", "Median", "Stddev")


class StatsTable(Vertical):
    """Buttons + window band + DataTable; contents re-pulled via ``refresh_stats``."""

    DEFAULT_CSS = """
    StatsTable { height: 1fr; }
    StatsTable #copy-row { height: 3; padding: 0 1; }
    StatsTable Button { margin: 0 2 0 0; min-width: 16; }
    StatsTable #stats-window-band { height: 1; color: $warning; }
    StatsTable DataTable { height: 1fr; }
    """

    def __init__(
        self,
        window_s: float,
        rows_provider: Callable[[], list[StatRow]],
        meta_provider: Callable[[], CaptureMeta],
    ) -> None:
        """Build the tab; providers are called on every refresh/copy."""
        super().__init__()
        self._window_s = window_s
        self._rows_provider = rows_provider
        self._meta_provider = meta_provider

    def compose(self) -> ComposeResult:  # noqa: D102
        with Horizontal(id="copy-row"):
            yield Button("Copy as Markdown", id="copy-md")
            yield Button("Copy as Web table", id="copy-html")
            yield Button("Copy as Excel (TSV)", id="copy-tsv")
            yield Button("Copy metadata", id="copy-meta")
        yield Static(self._band_text(), id="stats-window-band")
        yield DataTable(id="stats-table", zebra_stripes=True)

    def _band_text(self) -> str:
        """Return a merged-cell band aligned over the four stat columns."""
        # Each DataTable column renders one cell-padding space either side, so a
        # width-w column occupies w+2 cells. The Metric and Source columns thus
        # end at (w+2) each, and the Now column starts exactly there.
        offset = (_W_METRIC + 2) + (_W_SOURCE + 2)
        span = 4 * (_W_STAT + 2) - 2
        label = f" stats over last {self._window_s:.0f} s (--window) "
        return " " * offset + f"┌{label:─^{span}}┐"

    def on_mount(self) -> None:  # noqa: D102
        table = self.query_one("#stats-table", DataTable)
        table.cursor_type = "row"
        table.add_column("Metric", width=_W_METRIC)
        table.add_column("Source", width=_W_SOURCE)
        for col in _STAT_COLS:
            table.add_column(col, width=_W_STAT)
        table.add_column("Unit", width=6)

    def refresh_stats(self) -> None:
        """Repaint the table from the providers (called by the app tick)."""
        table = self.query_one("#stats-table", DataTable)
        table.clear()
        for row in self._rows_provider():
            s = row.stats
            table.add_row(
                row.metric,
                row.source,
                f"{s.now:.1f}",
                f"{s.mean:.1f}",
                f"{s.median:.1f}",
                f"{s.stddev:.1f}",
                row.unit,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Copy the current snapshot in the pressed button's format."""
        rows = self._rows_provider()
        match event.button.id:
            case "copy-md":
                text, what = to_markdown(rows, self._window_s), "Markdown"
            case "copy-html":
                text, what = to_html(rows, self._window_s), "web table (HTML)"
            case "copy-tsv":
                text, what = to_tsv(rows, self._window_s), "Excel table (TSV)"
            case "copy-meta":
                text, what = to_metadata_markdown(self._meta_provider()), "capture metadata"
            case _:
                return
        self.app.copy_to_clipboard(text)
        self.notify(f"Copied {what} ✓", timeout=3)
