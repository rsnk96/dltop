"""Series-toggle id encoding: names with ``@`` / ``:`` must be Textual-id-safe."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Checkbox

from dltop.widgets.toggles import SeriesToggles, _from_id_safe, _id_safe


def test_id_safe_roundtrips_at_and_colon() -> None:
    for name in ("cpu", "sm@0", "tensor@11", "prom:9199:demo_queue_depth", "prom:5000:node:cpu:rate@1"):
        encoded = _id_safe(name)
        assert ":" not in encoded
        assert "@" not in encoded
        assert _from_id_safe(encoded) == name


class _TogglesApp(App):
    def __init__(self, series: list[tuple[str, int, str, bool]]) -> None:
        super().__init__()
        self._series = series

    def compose(self) -> ComposeResult:
        yield SeriesToggles("all-prom-plot", self._series)


async def test_prometheus_named_toggles_mount_without_bad_identifier() -> None:
    """Prometheus series (colon-laden names) must compose as valid Textual ids."""
    series = [
        ("prom:9199:demo_queue_depth", 51, "◍ demo_queue_depth :9199", False),
        ("prom:9199:demo_frames_total", 46, "◍ demo_frames_total :9199", False),
    ]
    app = _TogglesApp(series)
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        # Every checkbox mounted (no BadIdentifier raised) and its id decodes back.
        boxes = list(app.query(Checkbox))
        assert len(boxes) == len(series)
        decoded = {_from_id_safe((cb.id or "").removeprefix("all-prom-plot-cb-")) for cb in boxes}
        assert decoded == {name for name, _, _, _ in series}
