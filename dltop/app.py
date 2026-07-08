"""Main Textual app: info cards + tabbed multi-series charts + process table."""

from __future__ import annotations

import contextlib
import math
import sys
import time
from typing import TYPE_CHECKING

from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane

from dltop.metrics import MetricStore
from dltop.models import (
    ALL_ACTIVE_BY_DEFAULT,
    ALL_RECOLOUR,
    COMPUTE_SERIES_DCGM,
    COMPUTE_SERIES_NVML,
    MEMORY_SERIES,
    SYSTEM_SERIES,
    SeriesDef,
    SystemState,
    _fmt_bandwidth,
    _pct_color,
)
from dltop.sources.dcgm import DcgmProbe
from dltop.sources.demo import DemoSource
from dltop.sources.nvml import _collect_processes, init_nvml, pynvml, sample_nvml
from dltop.sources.system import init_system, psutil, sample_system
from dltop.widgets.cards import InfoCard
from dltop.widgets.plot import TimeSeriesPlot
from dltop.widgets.toggles import SeriesToggles

if TYPE_CHECKING:
    from psutil import Process as PsutilProcess
    from textual.app import ComposeResult

    from dltop.models import GpuState

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
    #procs {
        height: auto;
        max-height: 12;
        border-top: solid $accent;
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
        self.paused = False
        self._compute_series: list[SeriesDef] = []
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
            self._compute_series = COMPUTE_SERIES_DCGM
            return
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
        self._compute_series = COMPUTE_SERIES_DCGM if self.have_profiling else COMPUTE_SERIES_NVML

    def _all_series(self) -> list[SeriesDef]:
        """Every renderable series on one chart, only the four headline metrics on.

        Concatenates the Compute (DCGM or NVML), Memory and System series. Names are
        unique across tabs, so the combined chart's ring buffers never collide.
        """
        combined: list[SeriesDef] = []
        seen: set[str] = set()
        for name, colour, label, _ in (*self._compute_series, *MEMORY_SERIES, *SYSTEM_SERIES):
            if name in seen:
                continue
            seen.add(name)
            combined.append((name, ALL_RECOLOUR.get(name, colour), label, name in ALL_ACTIVE_BY_DEFAULT))
        return combined

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

        all_series = self._all_series()
        with TabbedContent(id="tabs"):
            with TabPane("All", id="tab-all"), Vertical():
                yield TimeSeriesPlot(self.store, all_series, "All metrics", plot_id="all-plot")
                yield SeriesToggles("all-plot", all_series)
            with TabPane("Compute", id="tab-compute"), Vertical():
                yield TimeSeriesPlot(
                    self.store,
                    self._compute_series,
                    "Compute: CPU, GPU SM, encode/decode",
                    plot_id="compute-plot",
                )
                yield SeriesToggles("compute-plot", self._compute_series)
            with TabPane("Memory", id="tab-memory"), Vertical():
                yield TimeSeriesPlot(
                    self.store, MEMORY_SERIES, "Memory: RAM, GPU VRAM, VRAM bandwidth", plot_id="memory-plot"
                )
                yield SeriesToggles("memory-plot", MEMORY_SERIES)
            with TabPane("System", id="tab-system"), Vertical():
                yield TimeSeriesPlot(
                    self.store, SYSTEM_SERIES, "System: PCIe, GPU power, disk, network", plot_id="system-plot"
                )
                yield SeriesToggles("system-plot", SYSTEM_SERIES)

        yield VerticalScroll(DataTable(id="procs", zebra_stripes=True))

        yield Footer()

    def _header_name(self) -> str:
        mode = "DEMO" if self.demo is not None else ("DCGM" if self.have_profiling else "NVML")
        return f"dltop · driver {self._nv_driver or '?'} · CUDA {self._cuda_ver or '?'} · mode: {mode}"

    # -- on_mount: prime things that need widgets to exist ----------------------------

    def on_mount(self) -> None:  # noqa: D102
        self.title = "dltop"
        self.sub_title = self._header_name()

        procs = self.query_one("#procs", DataTable)
        procs.add_columns("GPU", "PID", "USER", "TYPE", "GPU-MEM", "%CPU", "HOST-MEM", "COMMAND")
        procs.cursor_type = "row"

        if self.dcgm_note:
            banner = self.query_one("#status-banner", Static)
            banner.update(self.dcgm_note)
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
        self._refresh_cards()
        self._refresh_procs()

    def _push_series(self) -> None:
        now = time.time()
        # Across multiple GPUs we aggregate by max so a hot GPU isn't smoothed away.

        # -- Compute: host CPU + GPU compute engines + media encode/decode --------------
        compute: dict[str, float] = {
            "cpu": self.sys_state.cpu_pct,
            "nvenc": max((g.nvenc_pct for g in self.gpus), default=0.0),
            "nvdec": max((g.nvdec_pct for g in self.gpus), default=0.0),
        }
        if self.have_profiling:
            compute["sm"] = max((g.dcgm.get("sm_active", 0.0) for g in self.gpus), default=0.0)
            compute["tensor"] = max((g.dcgm.get("tensor_active", 0.0) for g in self.gpus), default=0.0)
            compute["fp32"] = max((g.dcgm.get("fp32_active", 0.0) for g in self.gpus), default=0.0)
            compute["fp16"] = max((g.dcgm.get("fp16_active", 0.0) for g in self.gpus), default=0.0)
            compute["fp64"] = max((g.dcgm.get("fp64_active", 0.0) for g in self.gpus), default=0.0)
        else:
            compute["sm"] = max((g.sm_overall_pct for g in self.gpus), default=0.0)

        # -- Memory: host RAM + GPU VRAM + GPU VRAM bandwidth ---------------------------
        memory = {
            "ram": self.sys_state.ram_pct,
            "vram": max((g.mem_pct for g in self.gpus), default=0.0),
            "mbw": max((g.mbw_pct for g in self.gpus), default=0.0),
        }

        # -- System: PCIe + GPU power + disk + network ----------------------------------
        agg_tx = sum(g.pcie_tx_mbs for g in self.gpus)
        agg_rx = sum(g.pcie_rx_mbs for g in self.gpus)
        max_peak = max((g.pcie_max_mbs for g in self.gpus), default=0.0)
        pcie_tx_pct = (agg_tx / max_peak * 100.0) if max_peak else 0.0
        pcie_rx_pct = (agg_rx / max_peak * 100.0) if max_peak else 0.0
        system = {
            "pcie_tx": pcie_tx_pct,
            "pcie_rx": pcie_rx_pct,
            "power": max(
                ((g.power_w / g.power_limit_w * 100.0) if g.power_limit_w else 0.0 for g in self.gpus),
                default=0.0,
            ),
            # Disk/Net are MB/s so clamp to a generous cap for the shared 0-100 axis.
            "disk_r": min(100.0, self.sys_state.disk_read_mbs),
            "disk_w": min(100.0, self.sys_state.disk_write_mbs),
            "net_rx": min(100.0, self.sys_state.net_rx_mbs),
            "net_tx": min(100.0, self.sys_state.net_tx_mbs),
        }

        self.store.record_many({**compute, **memory, **system}, ts=now)
        # The "All" tab reads the same names from the same store, so it needs no
        # separate feed -- every plot just re-reads whatever names it cares about.
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
        procs = self.demo.processes() if self.demo is not None else _collect_processes(self.gpus, self._proc_cache)
        table = self.query_one("#procs", DataTable)
        table.clear()
        if not procs:
            table.add_row("—", "—", "—", "—", "—", "—", "—", "(no GPU processes)")
            return
        for p in procs:
            table.add_row(
                str(p["gpu"]),
                str(p["pid"]),
                p["user"],
                p["type"],
                f"{p['mem_mb']:.0f} MiB",
                "—" if math.isnan(p["cpu_pct"]) else f"{p['cpu_pct']:.0f}",
                "—" if math.isnan(p["rss_mb"]) else f"{p['rss_mb']:.0f}M",
                p["cmd"][:120],
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
        if self.demo is None:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()
