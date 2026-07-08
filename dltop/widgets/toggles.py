"""Checkbox rows that toggle chart series on and off."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox

from dltop.models import SeriesDef, _swatch_hex
from dltop.widgets.plot import TimeSeriesPlot

if TYPE_CHECKING:
    from textual.app import ComposeResult


class SeriesToggles(Vertical):
    """Wrapping rows of checkboxes that toggle individual series on/off.

    Checkbox is used (not Switch) because the ``[x] VRAM %`` idiom reads as
    "show this series" with zero ambiguity, whereas a Switch can look like two
    unexplained squares next to each label.

    Series are chunked into rows of at most ``_PER_ROW`` so a long list (the
    "All" tab has 18) wraps onto several lines instead of overflowing off the
    right edge where the trailing toggles become unreachable.
    """

    _PER_ROW = 6

    DEFAULT_CSS = """
    SeriesToggles {
        height: auto;
        padding: 0 1;
        background: $surface;
    }
    SeriesToggles Horizontal {
        height: auto;
    }
    SeriesToggles Checkbox {
        margin: 0 2 0 0;
        width: auto;
        height: 1;
        border: none;
        background: transparent;
    }
    """

    def __init__(self, plot_id: str, series: list[SeriesDef]) -> None:
        """Create one checkbox per series, routed to the plot with id ``plot_id``."""
        super().__init__()
        self._plot_id = plot_id
        self._series = series

    def compose(self) -> ComposeResult:  # noqa: D102
        for start in range(0, len(self._series), self._PER_ROW):
            with Horizontal():
                for name, color_idx, label, default in self._series[start : start + self._PER_ROW]:
                    yield Checkbox(
                        f"[{_swatch_hex(color_idx)}]●[/] {label}",
                        value=default,
                        id=f"{self._plot_id}-cb-{name}",
                    )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Forward the flip to the sibling plot. id format: '<plot_id>-cb-<name>'."""
        cb_id = event.checkbox.id or ""
        prefix = f"{self._plot_id}-cb-"
        if not cb_id.startswith(prefix):
            return
        name = cb_id[len(prefix) :]
        plot = self.app.query_one(f"#{self._plot_id}", TimeSeriesPlot)
        plot.set_visible(name, visible=event.value)
