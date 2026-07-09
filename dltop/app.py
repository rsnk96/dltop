"""Main Textual app: info cards + tabbed multi-series charts + process table."""

from __future__ import annotations

import contextlib
import math
import platform
import socket
import sys
import time
from typing import TYPE_CHECKING

from loguru import logger
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane

from dltop._version import __version__
from dltop.export import CaptureMeta, GpuMeta, StatRow, host_cpu_desc
from dltop.metrics import MetricStore
from dltop.models import (
    _PALETTE,
    GPU_SERIES_DCGM,
    GPU_SERIES_NVML,
    HOST_SERIES,
    SERIES_LABELS,
    SERIES_UNITS,
    SystemState,
    _fmt_bandwidth,
    _pct_color,
    per_gpu,
    split_series_name,
)
from dltop.sources.dcgm import DcgmProbe
from dltop.sources.demo import DemoSource
from dltop.sources.nvml import _collect_processes, init_nvml, pynvml, sample_nvml
from dltop.sources.prometheus import PromEndpoint, PromScraper, discover
from dltop.sources.system import init_system, psutil, sample_system
from dltop.widgets.cards import InfoCard
from dltop.widgets.plot import TimeSeriesPlot
from dltop.widgets.scroll import HintedScroll
from dltop.widgets.stats_table import StatsTable
from dltop.widgets.toggles import GpuToggles, SeriesToggles

if TYPE_CHECKING:
    from psutil import Process as PsutilProcess
    from textual.app import ComposeResult

    from dltop.models import GpuState, SeriesDef

TRANSIENT_NOTES_SECONDS = 8.0


class DltopApp(App):
    """Main Textual app: info cards + tabbed multi-series charts + process table."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #cards-row {
        height: auto;
        padding: 0 1;
    }
    #status-banner {
        height: auto;
        padding: 0 1;
        color: $warning;
    }
    #status-banner.hidden {
        display: none;
    }
    #pause-banner {
        height: 1;
        padding: 0 1;
        background: $warning;
        color: $text;
        text-style: bold;
        text-align: center;
    }
    #pause-banner.hidden {
        display: none;
    }
    TabbedContent {
        height: 1fr;
        min-height: 18;
    }
    .tab-scroll {
        /* Subtle, thin scrollbar -- the partial next chart is the main
           "there's more below" hint; the bar just confirms it. */
        scrollbar-size-vertical: 1;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-color: $panel-lighten-2;
        scrollbar-color-hover: $accent;
        scrollbar-color-active: $accent;
    }
    .chart-block {
        height: auto;
    }
    .chart-block TimeSeriesPlot {
        /* Kept short on purpose: the next chart peeks in below the fold so
           it's obvious the tab scrolls. */
        height: 11;
        min-height: 8;
    }
    .procs {
        height: auto;
        border-top: solid $accent;
        margin-top: 1;
    }
    .scroll-tail {
        height: 3;
    }
    .paused-label {
        color: $warning;
        text-style: bold;
    }
    """

    BINDINGS = [  # noqa: RUF012
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause"),
        Binding("space", "toggle_pause", "Pause", show=False),
    ]

    def __init__(
        self,
        *,
        interval: float,
        no_dcgm: bool,
        demo_gpus: int | None = None,
        window_s: float = 60.0,
        no_discover: bool = False,
    ) -> None:
        """Configure sampling ``interval`` (seconds) and whether to skip DCGM.

        ``demo_gpus`` selects the synthetic :class:`DemoSource` in place of real NVML/DCGM
        hardware (``None`` disables demo mode). ``window_s`` is the stats window for the
        Table tab. ``no_discover`` disables Prometheus ``/metrics`` auto-discovery.
        """
        super().__init__()
        self.interval = interval
        self.no_dcgm = no_dcgm
        self.demo = DemoSource(demo_gpus) if demo_gpus else None
        self.window_s = window_s
        self.no_discover = no_discover
        self._t0 = time.monotonic()
        self.store = MetricStore()
        self.gpus: list[GpuState] = []
        self.sys_state: SystemState = SystemState()
        self.dcgm: DcgmProbe | None = None
        self.have_profiling = False
        self.dcgm_note = ""
        self.prom_endpoints: list[PromEndpoint] = []
        self.prom: PromScraper | None = None
        self._prom_note = ""
        self.paused = False
        self._nv_driver = ""
        self._cuda_ver = ""
        self._proc_cache: dict[int, PsutilProcess] = {}

    # -- bring-up --------------------------------------------------------------------

    def _prepare_sources(self) -> None:
        if self.demo is not None:
            self.gpus = self.demo.init_gpus()
            self.sys_state = SystemState()
            self.have_profiling = True  # demo fabricates the full DCGM series
            self._nv_driver, self._cuda_ver = "demo", "demo"
        else:
            self.gpus = init_nvml()
            if not self.gpus:
                sys.exit("No NVIDIA GPUs detected by NVML")
            self.sys_state = init_system()

            with contextlib.suppress(pynvml.NVMLError):
                raw = pynvml.nvmlSystemGetDriverVersion()
                self._nv_driver = raw.decode() if isinstance(raw, bytes) else raw
            with contextlib.suppress(pynvml.NVMLError):
                v = pynvml.nvmlSystemGetCudaDriverVersion()
                self._cuda_ver = f"{v // 1000}.{(v % 1000) // 10}"

            if self.no_dcgm:
                self.dcgm_note = "DCGM disabled by --no-dcgm. Showing NVML overall SM% (lumped CUDA+Tensor+RT)."
            else:
                ok_cli, err_cli = DcgmProbe.cli_available()
                if not ok_cli:
                    self.dcgm_note = f"DCGM unavailable: {err_cli}"
                else:
                    ok_prof, err_prof = DcgmProbe.profiling_supported()
                    if ok_prof:
                        self.dcgm = DcgmProbe(len(self.gpus), interval_ms=int(self.interval * 1000))
                        self.dcgm.start()
                        self.have_profiling = True
                    else:
                        self.dcgm_note = f"DCGM present but profiling fields not available on this GPU. {err_prof}"

        if not self.no_discover:
            self.prom_endpoints = discover()
            if self.prom_endpoints:
                self.prom = PromScraper(self.prom_endpoints, self.store, interval_s=max(self.interval, 1.0))
                self.prom.start()
                found = ", ".join(f":{ep.port} ({ep.name}, {len(ep.metrics)} metrics)" for ep in self.prom_endpoints)
                self._prom_note = f"Prometheus: scraping {found}"
            else:
                self._prom_note = ""

    def _gpu_series_table(self) -> dict[str, list[SeriesDef]]:
        """Return the per-GPU series table: DCGM's richer split, or NVML's lumped SM%."""
        return GPU_SERIES_DCGM if self.have_profiling else GPU_SERIES_NVML

    def _chart_block(self, plot_id: str, title: str, series: list[SeriesDef], *, classes: str = "") -> Vertical:
        """Wrap one chart + its series toggles in a fixed-height ``Vertical`` block."""
        return Vertical(
            TimeSeriesPlot(self.store, series, title, plot_id=plot_id),
            SeriesToggles(plot_id, series),
            classes=f"chart-block {classes}".strip(),
        )

    def _compose_tab(self, tab: str) -> ComposeResult:
        """Yield the host chart followed by one chart per GPU, all for domain ``tab``."""
        host_title = "Host — CPU · RAM · Disk · Network" if tab == "all" else f"Host — {tab}"
        yield self._chart_block(f"{tab}-host-plot", host_title, HOST_SERIES[tab])
        for g in self.gpus:
            yield self._chart_block(
                f"{tab}-gpu{g.index}-plot",
                f"GPU {g.index} · {g.name}",
                per_gpu(self._gpu_series_table()[tab], g.index),
                classes=f"gpu-chart-{g.index}",
            )
        if tab == "all" and self.prom_endpoints:
            series = self._prom_series_defs()
            plot = TimeSeriesPlot(
                self.store,
                series,
                "Prometheus — % of peak ◍",
                plot_id="all-prom-plot",
                scale_to_peak=True,
            )
            yield Vertical(plot, SeriesToggles("all-prom-plot", series), classes="chart-block")
        # Processes ride at the very bottom of the same scroll, so the whole tab
        # is one continuous top-to-bottom read rather than two separate panes.
        yield DataTable(zebra_stripes=True, classes="procs")
        # A little dead space after the table so scrolling to the end visibly
        # bottoms out -- it's obvious there's nothing more below.
        yield Static("", classes="scroll-tail")

    def _prom_series_defs(self) -> list[SeriesDef]:
        """One series per discovered metric, sorted by name and on by default."""
        palette = list(_PALETTE)
        pairs = sorted(
            ((ep.port, metric) for ep in self.prom_endpoints for metric in ep.metrics),
            key=lambda pm: (pm[1], pm[0]),
        )
        defs: list[SeriesDef] = []
        for n, (port, metric) in enumerate(pairs):
            label = f"◍ {metric[:28]} :{port}"
            defs.append((f"prom:{port}:{metric}", palette[n % len(palette)], label, True))
        return defs

    # -- layout ----------------------------------------------------------------------

    def compose(self) -> ComposeResult:  # noqa: D102
        self._prepare_sources()
        yield Header(show_clock=True, name=self._header_name())

        yield Static("", id="status-banner", classes="hidden")
        yield Static("⏸  PAUSED — press p to resume", id="pause-banner", classes="hidden")

        with Horizontal(id="cards-row"):
            for g in self.gpus:
                yield InfoCard(f" GPU {g.index} · {g.name} ", card_id=f"gpu-card-{g.index}")
                yield InfoCard(" Memory ", card_id=f"mem-card-{g.index}")
                yield InfoCard(" Power / Thermal ", card_id=f"power-card-{g.index}")
                yield InfoCard(" PCIe ", card_id=f"pcie-card-{g.index}")
            yield InfoCard(" System ", card_id="sys-card")

        if len(self.gpus) > 1:
            yield GpuToggles(self.gpus)
        with TabbedContent(id="tabs"):
            for tab, title in (("all", "All"), ("compute", "Compute"), ("memory", "Memory"), ("system", "System")):
                with TabPane(title, id=f"tab-{tab}"):
                    yield HintedScroll(*self._compose_tab(tab))
            with TabPane("Table", id="tab-table"):
                yield StatsTable(self.window_s, self._stat_rows, self._capture_meta)

        yield Footer()

    def _header_name(self) -> str:
        mode = "DEMO" if self.demo is not None else ("DCGM" if self.have_profiling else "NVML")
        return f"dltop · driver {self._nv_driver or '?'} · CUDA {self._cuda_ver or '?'} · mode: {mode}"

    # -- on_mount: prime things that need widgets to exist ----------------------------

    def on_mount(self) -> None:  # noqa: D102
        self.title = "dltop"
        self.sub_title = self._header_name()

        for procs in self.query(".procs").results(DataTable):
            procs.add_columns("GPU", "PID", "USER", "TYPE", "GPU-MEM", "%CPU", "HOST-MEM", "COMMAND")
            procs.cursor_type = "row"

        banner_text = "\n".join(note for note in (self.dcgm_note, self._prom_note) if note)
        if banner_text:
            banner = self.query_one("#status-banner", Static)
            banner.update(banner_text)
            banner.remove_class("hidden")
            self.set_timer(TRANSIENT_NOTES_SECONDS, lambda: banner.add_class("hidden"))

        self.set_interval(self.interval, self._tick)
        # One immediate sample so graphs aren't empty on first paint.  Deferred via
        # call_after_refresh (not a direct call) because widgets composed inside
        # TabbedContent/Horizontal may not be mounted yet when on_mount fires on newer
        # Textual; a synchronous _tick() then races query_one() -> NoMatches and crashes.
        self.call_after_refresh(self._tick)

    # -- sampling & redraw -----------------------------------------------------------

    def _tick(self) -> None:
        if self.paused:
            return
        if self.demo is not None:
            self.demo.sample(self.gpus, self.sys_state, time.monotonic() - self._t0)
        else:
            for g in self.gpus:
                sample_nvml(g)
                if self.dcgm is not None:
                    g.dcgm = self.dcgm.snapshot(g.index)
            sample_system(self.sys_state)
        self._push_series()
        # A `set_interval` timer callback can still be in flight when the app starts
        # tearing down (e.g. a test's `run_test()` context exiting) -- widgets may
        # already be unmounted by the time this runs. That's not a real bug, so
        # swallow it the same way TimeSeriesPlot.render() never lets a draw glitch
        # kill the whole monitor.
        try:
            self._refresh_cards()
            self._refresh_procs()
            self.query_one(StatsTable).refresh_stats()
        except NoMatches:
            # Benign at teardown (widgets already gone); during normal operation
            # it would signal a real id/query mismatch, so leave a debug trace
            # rather than freezing the panels completely silently.
            logger.debug("_tick refresh skipped: queried a widget that is not mounted")

    def _push_series(self) -> None:
        """Record one sample per series: plain names for host, ``name@{i}`` per GPU.

        Values are raw (unclamped) -- the plot clamps its own y-axis visually, but
        the Table tab (Task 6) needs the true numbers, e.g. actual MB/s not a
        percent-axis-friendly cap.
        """
        samples: dict[str, float] = {
            "cpu": self.sys_state.cpu_pct,
            "ram": self.sys_state.ram_pct,
            "disk_r": self.sys_state.disk_read_mbs,
            "disk_w": self.sys_state.disk_write_mbs,
            "net_rx": self.sys_state.net_rx_mbs,
            "net_tx": self.sys_state.net_tx_mbs,
        }
        for g in self.gpus:
            i = g.index
            samples[f"nvenc@{i}"] = g.nvenc_pct
            samples[f"nvdec@{i}"] = g.nvdec_pct
            samples[f"vram@{i}"] = g.mem_pct
            samples[f"mbw@{i}"] = g.mbw_pct
            samples[f"power@{i}"] = (g.power_w / g.power_limit_w * 100.0) if g.power_limit_w else 0.0
            samples[f"pcie_tx@{i}"] = (g.pcie_tx_mbs / g.pcie_max_mbs * 100.0) if g.pcie_max_mbs else 0.0
            samples[f"pcie_rx@{i}"] = (g.pcie_rx_mbs / g.pcie_max_mbs * 100.0) if g.pcie_max_mbs else 0.0
            if self.have_profiling:
                samples[f"sm@{i}"] = g.dcgm.get("sm_active", 0.0)
                samples[f"tensor@{i}"] = g.dcgm.get("tensor_active", 0.0)
                samples[f"fp32@{i}"] = g.dcgm.get("fp32_active", 0.0)
                samples[f"fp16@{i}"] = g.dcgm.get("fp16_active", 0.0)
                samples[f"fp64@{i}"] = g.dcgm.get("fp64_active", 0.0)
            else:
                samples[f"sm@{i}"] = g.sm_overall_pct
        self.store.record_many(samples, ts=time.time())
        for plot in self.query(TimeSeriesPlot):
            plot.replot()

    def _refresh_cards(self) -> None:
        for g in self.gpus:
            self.query_one(f"#gpu-card-{g.index}", InfoCard).update_rows(
                [
                    ("Temp", f"[{_pct_color((g.temp_c - 30) * 2)}]{g.temp_c:.0f}°C[/]"),
                    ("Fan", "—" if math.isnan(g.fan_pct) else f"{g.fan_pct:.0f}%"),
                    ("SM clk", f"{g.sm_clock_mhz:.0f} MHz"),
                    ("Mem clk", f"{g.mem_clock_mhz:.0f} MHz"),
                ],
            )
            self.query_one(f"#mem-card-{g.index}", InfoCard).update_rows(
                [
                    (
                        "VRAM",
                        f"[{_pct_color(g.mem_pct)}]{g.mem_used_mb / 1024:.2f}"
                        f" / {g.mem_total_mb / 1024:.2f} GiB[/] ({g.mem_pct:.0f}%)",
                    ),
                    ("Bandwidth", f"[{_pct_color(g.mbw_pct)}]{g.mbw_pct:.0f}%[/] util"),
                ],
            )
            power_pct = (g.power_w / g.power_limit_w * 100.0) if g.power_limit_w else 0.0
            self.query_one(f"#power-card-{g.index}", InfoCard).update_rows(
                [
                    (
                        "Power",
                        f"[{_pct_color(power_pct)}]{g.power_w:.0f}"
                        f" / {g.power_limit_w:.0f} W[/] ({power_pct:.0f}%)",
                    ),
                    ("NVENC", f"[yellow]{g.nvenc_pct:.0f}%[/]"),
                    ("NVDEC", f"[yellow]{g.nvdec_pct:.0f}%[/]"),
                ],
            )
            link_tag = f"Gen{g.pcie_curr_gen}x{g.pcie_curr_width}" if g.pcie_curr_gen else "?"
            tx_pct = (g.pcie_tx_mbs / g.pcie_max_mbs * 100.0) if g.pcie_max_mbs else 0.0
            rx_pct = (g.pcie_rx_mbs / g.pcie_max_mbs * 100.0) if g.pcie_max_mbs else 0.0
            self.query_one(f"#pcie-card-{g.index}", InfoCard).update_rows(
                [
                    ("Link", f"[bright_white]{link_tag}[/]"),
                    ("TX", f"[{_pct_color(tx_pct)}]{_fmt_bandwidth(g.pcie_tx_mbs)}[/] ({tx_pct:.1f}%)"),
                    ("RX", f"[{_pct_color(rx_pct)}]{_fmt_bandwidth(g.pcie_rx_mbs)}[/] ({rx_pct:.1f}%)"),
                ],
            )

        sys_rows: list[tuple[str, str]] = []
        if psutil is None:
            sys_rows.append(("psutil", "not installed"))
        else:
            sys_rows.extend(
                [
                    (
                        "CPU",
                        f"[{_pct_color(self.sys_state.cpu_pct)}]{self.sys_state.cpu_pct:.0f}%[/]"
                        f" ({self.sys_state.cpu_count_physical}C/{self.sys_state.cpu_count_logical}T)",
                    ),
                    (
                        "Load",
                        f"{self.sys_state.load_1:.2f}  {self.sys_state.load_5:.2f}  {self.sys_state.load_15:.2f}",
                    ),
                    (
                        "RAM",
                        f"[{_pct_color(self.sys_state.ram_pct)}]{self.sys_state.ram_used_gib:.1f}"
                        f" / {self.sys_state.ram_total_gib:.1f} GiB[/]",
                    ),
                    (
                        "IO",
                        f"d {self.sys_state.disk_read_mbs:.1f}/{self.sys_state.disk_write_mbs:.1f}"
                        f"  n {self.sys_state.net_rx_mbs:.1f}/{self.sys_state.net_tx_mbs:.1f} MB/s",
                    ),
                ],
            )
        self.query_one("#sys-card", InfoCard).update_rows(sys_rows)

    def _refresh_procs(self) -> None:
        # Enumerate once; the process table is composed once per chart tab so it
        # sits at the bottom of each tab's scroll -- repaint every copy.
        procs = self.demo.processes() if self.demo is not None else _collect_processes(self.gpus, self._proc_cache)
        if procs:
            rows = [
                (
                    str(p["gpu"]),
                    str(p["pid"]),
                    p["user"],
                    p["type"],
                    f"{p['mem_mb']:.0f} MiB",
                    "—" if math.isnan(p["cpu_pct"]) else f"{p['cpu_pct']:.0f}",
                    "—" if math.isnan(p["rss_mb"]) else f"{p['rss_mb']:.0f}M",
                    p["cmd"][:120],
                )
                for p in procs
            ]
        else:
            rows = [("—", "—", "—", "—", "—", "—", "—", "(no GPU processes)")]
        for table in self.query(".procs").results(DataTable):
            table.clear()
            for row in rows:
                table.add_row(*row)

    # -- Table tab (Task 6) -----------------------------------------------------------

    def _stat_rows(self) -> list[StatRow]:
        """Every recorded series as a StatRow: host first, then GPUs, then Prometheus."""
        rows: list[StatRow] = []
        for name in sorted(self.store.names(), key=self._row_sort_key):
            stats = self.store.stats(name, self.window_s)
            if stats is None:
                continue
            if name.startswith("prom:"):
                _, port, metric = name.split(":", 2)
                rows.append(StatRow(metric, f":{port}", "", stats))
                continue
            base, gpu_idx = split_series_name(name)
            label = SERIES_LABELS.get(base, base)
            source = "host" if gpu_idx is None else f"GPU {gpu_idx}"
            rows.append(StatRow(label, source, SERIES_UNITS.get(base, ""), stats))
        return rows

    @staticmethod
    def _row_sort_key(name: str) -> tuple[int, str]:
        if name.startswith("prom:"):
            return (2, name)
        return (1, name) if "@" in name else (0, name)

    def _capture_meta(self) -> CaptureMeta:
        """Snapshot of measurement context for the Copy-metadata button."""
        n = self.store.stats("cpu", self.window_s)
        return CaptureMeta(
            captured_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            version=__version__,
            window_s=self.window_s,
            n_samples=n.n_samples if n else 0,
            interval_s=self.interval,
            hostname=socket.gethostname(),
            os_desc=f"{platform.system()} {platform.release()}",
            cpu_desc=(f"{host_cpu_desc()} · {self.sys_state.cpu_count_physical}C/{self.sys_state.cpu_count_logical}T"),
            ram_gib=self.sys_state.ram_total_gib,
            gpus=[GpuMeta(g.index, g.name, g.mem_total_mb / 1024.0) for g in self.gpus],
            driver=self._nv_driver or "?",
            cuda=self._cuda_ver or "?",
            prom_endpoints=[f":{ep.port} ({ep.name}, {len(ep.metrics)} metrics)" for ep in self.prom_endpoints],
        )

    # -- bindings --------------------------------------------------------------------

    def action_toggle_pause(self) -> None:
        """Freeze sampling; readouts and graphs hold their last values."""
        self.paused = not self.paused
        banner = self.query_one("#pause-banner", Static)
        if self.paused:
            banner.remove_class("hidden")
            self.sub_title = self._header_name() + "   ⏸ PAUSED"
        else:
            banner.add_class("hidden")
            self.sub_title = self._header_name()

    # -- shutdown --------------------------------------------------------------------

    def on_unmount(self) -> None:  # noqa: D102
        if self.dcgm is not None:
            self.dcgm.stop()
        if self.prom is not None:
            self.prom.stop()
        if self.demo is None:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()
