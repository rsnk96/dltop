"""Table tab: stats rows render and all four copy buttons hit the clipboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dltop.app import DltopApp
from dltop.widgets.stats_table import StatsTable

if TYPE_CHECKING:
    import pytest


async def _demo_app_with_data() -> DltopApp:
    return DltopApp(interval=0.05, no_dcgm=True, demo_gpus=2, window_s=60.0, no_discover=True)


async def test_table_tab_lists_host_and_per_gpu_rows() -> None:
    app = await _demo_app_with_data()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        for _ in range(3):
            await pilot.pause(0.1)
        app.query_one("#tabs").active = "tab-table"
        await pilot.pause()
        table = app.query_one("#stats-table")
        sources = {table.get_row_at(i)[1] for i in range(table.row_count)}
        assert {"host", "GPU 0", "GPU 1"} <= set(map(str, sources))
        band = app.query_one(StatsTable)._band_text()
        assert "60 s" in band


async def test_window_band_aligns_with_stat_columns() -> None:
    """The band's left corner must sit exactly where the 'Now' column starts."""
    app = await _demo_app_with_data()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        app.query_one("#tabs").active = "tab-table"
        await pilot.pause()
        table = app.query_one("#stats-table")
        # Compute where the first stat column ("Now") actually begins, summing the
        # rendered width (content + cell padding) of the columns before it.
        now_offset = 0
        for column in table.columns.values():
            if str(column.label) == "Now":
                break
            now_offset += column.get_render_width(table)
        band = app.query_one(StatsTable)._band_text()
        assert band.index("┌") == now_offset


async def test_copy_buttons_export_each_format(monkeypatch: pytest.MonkeyPatch) -> None:
    app = await _demo_app_with_data()
    copied: list[str] = []
    # Patch the clipboard seam the widget imports, so the test captures the text
    # without actually shelling out to a system clipboard tool.
    monkeypatch.setattr(
        "dltop.widgets.stats_table.copy_to_clipboard",
        lambda _app, text: copied.append(text),
    )
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        for _ in range(3):
            await pilot.pause(0.1)
        app.query_one("#tabs").active = "tab-table"
        await pilot.pause()
        for btn in ("#copy-md", "#copy-html", "#copy-tsv", "#copy-meta"):
            await pilot.click(btn)
        assert len(copied) == 4
        md, html_out, tsv, meta = copied
        assert md.startswith("| Metric |")
        assert html_out.startswith("<table>")
        assert "\t" in tsv.splitlines()[0]
        assert meta.startswith("## dltop capture metadata")
        assert "Demo GPU 0" in meta
