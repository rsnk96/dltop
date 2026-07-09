"""Checkbox rows that toggle chart series on and off."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox, Static

from dltop.models import GpuState, SeriesDef, _swatch_hex
from dltop.widgets.plot import TimeSeriesPlot

if TYPE_CHECKING:
    from textual.app import ComposeResult


def _id_safe(name: str) -> str:
    """Encode a series name for use inside a Textual widget id.

    Textual ids may only contain letters, numbers, underscores and hyphens, but
    series names carry two illegal characters: per-GPU names look like ``sm@1``
    (``@``) and Prometheus names look like ``prom:9199:queue_depth`` (``:``).
    Series names never contain a hyphen (base names use ``_``, Prometheus metric
    names are ``[a-zA-Z0-9_:]``), so the hyphenated tokens below can only come
    from encoding and are unambiguous to reverse with :func:`_from_id_safe`.
    """
    return name.replace("@", "-at-").replace(":", "-colon-")


def _from_id_safe(safe: str) -> str:
    """Invert :func:`_id_safe`."""
    return safe.replace("-colon-", ":").replace("-at-", "@")


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
    /* On focus Textual adds `border: tall $border`, but these checkboxes are
       only 1 row tall, so that border renders as a solid blue block that hides
       the whole label. Drop the border and the block-cursor label paint; use a
       plain underline as the focus cue instead. !important beats the focus and
       component-class defaults. */
    SeriesToggles Checkbox:focus {
        border: none !important;
        background-tint: 0% !important;
    }
    SeriesToggles Checkbox:focus > .toggle--label {
        color: $text !important;
        background: transparent !important;
        text-style: underline !important;
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
                        id=f"{self._plot_id}-cb-{_id_safe(name)}",
                    )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Forward the flip to the sibling plot. id format: '<plot_id>-cb-<id-safe-name>'."""
        cb_id = event.checkbox.id or ""
        prefix = f"{self._plot_id}-cb-"
        if not cb_id.startswith(prefix):
            return
        name = _from_id_safe(cb_id[len(prefix) :])
        plot = self.app.query_one(f"#{self._plot_id}", TimeSeriesPlot)
        plot.set_visible(name, visible=event.value)


class GpuToggles(Horizontal):
    """One checkbox per GPU; unchecking hides that GPU's charts on every tab."""

    DEFAULT_CSS = """
    GpuToggles { height: auto; padding: 0 1; background: $surface; }
    GpuToggles Checkbox { margin: 0 2 0 0; width: auto; height: 1; border: none; background: transparent; }
    GpuToggles Checkbox:focus { border: none !important; background-tint: 0% !important; }
    GpuToggles Checkbox:focus > .toggle--label {
        color: $text !important;
        background: transparent !important;
        text-style: underline !important;
    }
    GpuToggles Static { width: auto; padding: 0 1 0 0; color: $accent; text-style: bold; }
    """

    def __init__(self, gpus: list[GpuState]) -> None:
        """Build the row for ``gpus`` (only composed when there is more than one)."""
        super().__init__()
        self._gpus = gpus

    def compose(self) -> ComposeResult:  # noqa: D102
        yield Static("GPUs:")
        for g in self._gpus:
            yield Checkbox(f"GPU {g.index} · {g.name}", value=True, id=f"gpu-cb-{g.index}")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Show/hide every chart block belonging to the toggled GPU."""
        cb_id = event.checkbox.id or ""
        if not cb_id.startswith("gpu-cb-"):
            return
        idx = cb_id.removeprefix("gpu-cb-")
        for block in self.app.query(f".gpu-chart-{idx}"):
            block.display = event.value
