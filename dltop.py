"""dltop -- a top/htop-style GPU monitor tailored for CV and AI workloads.

Splits live GPU utilization into two categories:

    * Compute (CV/AI): SM active, Tensor pipe, FP32/FP16/FP64 pipes -- via DCGM profiling
    * Media engines: NVENC, NVDEC -- via NVML

The UI is built on Textual with box-drawing line charts (the nvtop/asciichart
look), so many metrics share one chart instead of stacking into tall bars.
Overlapping series are colour-interleaved per column so no line is ever hidden
behind another while every line keeps its true value (see ``TimeSeriesPlot``).

When DCGM profiling metrics are unavailable (e.g. consumer GeForce cards where
NVIDIA gates profiling fields, or DCGM not installed), we transparently fall
back to NVML's single overall SM% and show a status banner telling the user how
to enable the full split on a data-center GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Checkbox, DataTable, Footer, Header, Static, TabbedContent, TabPane

if TYPE_CHECKING:

    from rich.console import RenderableType
    from textual.app import ComposeResult

try:
    import pynvml
except ImportError:
    sys.exit("nvidia-ml-py is required. Install with: pip install nvidia-ml-py")

try:
    import psutil
except ImportError:
    psutil = None  # process table degrades gracefully without it


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

HISTORY_LEN = 240  # ~2 min of samples at 0.5 s cadence
TRANSIENT_NOTES_SECONDS = 8.0  # how long startup banners (DCGM status) stay visible

AXIS_STYLE = "color(244)"  # dim grey for axis ticks and labels

# PCIe theoretical peak bandwidth per lane per direction (MB/s). Gen3+ uses 128b/130b
# encoding (hence the non-round numbers); see PCI-SIG spec.
PCIE_GEN_MBS_PER_LANE = {
    1: 250.0,
    2: 500.0,
    3: 984.6,
    4: 1969.2,
    5: 3938.4,
    6: 7876.8,
}

# DCGM profiling field IDs (see dcgm_fields.h). All return a fraction 0..1.
# Ordered deliberately; the `dcgmi dmon` output columns match this order.
DCGM_FIELD_ORDER: list[int] = [1002, 1004, 1007, 1008, 1006]
DCGM_FIELD_NAMES: dict[int, str] = {
    1002: "sm_active",
    1004: "tensor_active",
    1007: "fp32_active",
    1008: "fp16_active",
    1006: "fp64_active",
}

DCGM_INSTALL_HINT = (
    "Install NVIDIA Data Center GPU Manager and enable the service to get the\n"
    "Tensor vs FP32/FP16/FP64 split (works on Tesla/Quadro/A/H/L-series only):\n"
    "  Ubuntu:  sudo apt install datacenter-gpu-manager\n"
    "           sudo systemctl --now enable nvidia-dcgm\n"
    "  Docs:    https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html\n"
    "Note: NVIDIA gates profiling fields on GeForce cards; no software toggle unlocks them."
)

# Each series is rendered through two paths that have to agree visually:
#  - line strokes in the chart (Rich `color(<idx>)` styles)
#  - a `●` swatch in the Textual legend (Rich markup)
# Terminals render truecolor only if `COLORTERM=truecolor`; under tmux/screen
# (`tmux-256color`) hex is silently downsampled to 256-color and several of our
# distinct hexes collapse to the same ANSI index. To stay distinguishable on
# every terminal, we anchor each color to an explicit ANSI-256 index for the
# chart and pair it with the matching exact hex for the legend.
_PALETTE: dict[int, str] = {
    15: "#FFFFFF",  # white
    21: "#0000FF",  # pure blue
    39: "#00AFFF",  # deep sky blue
    46: "#00FF00",  # bright green
    51: "#00FFFF",  # cyan
    93: "#8700FF",  # bright purple
    196: "#FF0000",  # pure red
    201: "#FF00FF",  # magenta / hot pink
    208: "#FF8700",  # orange
    220: "#FFD700",  # gold
    226: "#FFFF00",  # yellow
}


def _swatch_hex(idx: int) -> str:
    """Return the `#RRGGBB` for a palette index, for use in Textual markup."""
    return _PALETTE[idx]


# Each entry is (name, ansi256_index, label, default_visible). Indices come from
# `_PALETTE` above and are picked for max in-tab separation on a 256-color terminal.
#
# Three tabs, grouped by what an operator reasons about together:
#   * Compute — host + GPU compute engines + media encode/decode
#   * Memory  — host RAM, GPU VRAM, GPU VRAM bandwidth
#   * System  — PCIe, GPU power, disk, network
SeriesDef = tuple[str, int, str, bool]

# Compute tab. The DCGM variant adds the Tensor/FP32/FP16/FP64 split that only
# data-center GPUs expose; the NVML variant lumps it all into a single SM%.
COMPUTE_SERIES_DCGM: list[SeriesDef] = [
    ("cpu", 208, "CPU", True),  # orange
    ("sm", 15, "GPU SM", True),  # white  — overall
    ("tensor", 201, "Tensor", True),  # magenta / hot pink
    ("fp32", 39, "FP32", True),  # deep sky blue
    ("fp16", 46, "FP16", True),  # green
    ("fp64", 226, "FP64", True),  # yellow
    ("nvenc", 220, "NVENC", True),  # gold
    ("nvdec", 93, "NVDEC", True),  # purple
]
COMPUTE_SERIES_NVML: list[SeriesDef] = [
    ("cpu", 208, "CPU", True),  # orange
    ("sm", 15, "GPU SM", True),  # white
    ("nvenc", 220, "NVENC", True),  # gold
    ("nvdec", 93, "NVDEC", True),  # purple
]
MEMORY_SERIES: list[SeriesDef] = [
    ("ram", 208, "RAM", True),  # orange
    ("vram", 51, "GPU VRAM", True),  # cyan
    ("mbw", 46, "GPU VRAM BW", True),  # green
]
SYSTEM_SERIES: list[SeriesDef] = [
    ("pcie_tx", 51, "PCIe ↑", True),  # cyan
    ("pcie_rx", 15, "PCIe ↓", True),  # white
    ("power", 196, "GPU Power %", True),  # red
    ("disk_r", 46, "Disk Read (MB/s)", True),  # green
    ("disk_w", 226, "Disk Write (MB/s)", True),  # yellow
    ("net_rx", 201, "Net RX (MB/s)", True),  # magenta
    ("net_tx", 208, "Net TX (MB/s)", True),  # orange
]

# The "All" tab overlays every renderable series on one chart but starts with only
# the four headline metrics enabled — the at-a-glance "is anything busy?" view.
ALL_ACTIVE_BY_DEFAULT = ("cpu", "ram", "sm", "vram")
# RAM is orange in the Memory tab (same as CPU); recolour it on the combined chart
# so the two default-on host metrics stay distinct from each other.
ALL_RECOLOUR = {"ram": 46}  # green


# --------------------------------------------------------------------------------------
# Data model (framework-agnostic)
# --------------------------------------------------------------------------------------


@dataclass
class GpuState:
    """Snapshot + short history for a single GPU."""

    index: int
    name: str
    mem_total_mb: float = 0.0
    mem_used_mb: float = 0.0
    power_w: float = 0.0
    power_limit_w: float = 0.0
    temp_c: float = 0.0
    fan_pct: float = float("nan")
    sm_clock_mhz: float = 0.0
    mem_clock_mhz: float = 0.0
    sm_clock_max_mhz: float = 0.0
    mem_clock_max_mhz: float = 0.0
    mbw_pct: float = 0.0
    nvenc_pct: float = 0.0
    nvdec_pct: float = 0.0
    sm_overall_pct: float = 0.0
    pcie_tx_mbs: float = 0.0
    pcie_rx_mbs: float = 0.0
    pcie_max_mbs: float = 0.0
    pcie_curr_gen: int = 0
    pcie_curr_width: int = 0
    mem_pct: float = 0.0
    dcgm: dict[str, float] = field(default_factory=dict)


@dataclass
class SystemState:
    """System-wide CPU / RAM / disk / network -- context for the GPU workload."""

    cpu_pct: float = 0.0
    cpu_per_core: list[float] = field(default_factory=list)
    cpu_count_physical: int = 0
    cpu_count_logical: int = 0
    load_1: float = 0.0
    load_5: float = 0.0
    load_15: float = 0.0

    ram_used_gib: float = 0.0
    ram_total_gib: float = 0.0
    ram_pct: float = 0.0
    swap_used_gib: float = 0.0
    swap_total_gib: float = 0.0
    swap_pct: float = 0.0

    disk_read_mbs: float = 0.0
    disk_write_mbs: float = 0.0
    net_rx_mbs: float = 0.0
    net_tx_mbs: float = 0.0

    prev_disk: tuple[int, int] | None = None
    prev_net: tuple[int, int] | None = None
    prev_ts: float = 0.0


# --------------------------------------------------------------------------------------
# NVML sampling
# --------------------------------------------------------------------------------------


def init_nvml() -> list[GpuState]:
    """Initialize NVML and return one GpuState per visible GPU."""
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    gpus: list[GpuState] = []
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        state = GpuState(index=i, name=name, mem_total_mb=mem.total / 1e6)
        with contextlib.suppress(pynvml.NVMLError):
            gen = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
            width = int(pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
            state.pcie_max_mbs = PCIE_GEN_MBS_PER_LANE.get(gen, 0.0) * width
        gpus.append(state)
    return gpus


def sample_nvml(gpu: GpuState) -> None:
    """Populate NVML-derived fields on `gpu` in place."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu.mem_used_mb = mem.used / 1e6
    gpu.mem_pct = (gpu.mem_used_mb / gpu.mem_total_mb * 100.0) if gpu.mem_total_mb else 0.0

    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu.sm_overall_pct = float(util.gpu)
    gpu.mbw_pct = float(util.memory)

    with contextlib.suppress(pynvml.NVMLError):
        enc, _ = pynvml.nvmlDeviceGetEncoderUtilization(handle)
        gpu.nvenc_pct = float(enc)
    with contextlib.suppress(pynvml.NVMLError):
        dec, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
        gpu.nvdec_pct = float(dec)
    with contextlib.suppress(pynvml.NVMLError):
        gpu.temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    with contextlib.suppress(pynvml.NVMLError):
        gpu.power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        gpu.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    with contextlib.suppress(pynvml.NVMLError):
        gpu.fan_pct = float(pynvml.nvmlDeviceGetFanSpeed(handle))
    with contextlib.suppress(pynvml.NVMLError):
        gpu.sm_clock_mhz = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    with contextlib.suppress(pynvml.NVMLError):
        gpu.mem_clock_mhz = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    if gpu.sm_clock_max_mhz == 0.0:
        with contextlib.suppress(pynvml.NVMLError):
            gpu.sm_clock_max_mhz = float(pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM))
    if gpu.mem_clock_max_mhz == 0.0:
        with contextlib.suppress(pynvml.NVMLError):
            gpu.mem_clock_max_mhz = float(pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    with contextlib.suppress(pynvml.NVMLError):
        gpu.pcie_tx_mbs = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
        gpu.pcie_rx_mbs = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
    with contextlib.suppress(pynvml.NVMLError):
        gpu.pcie_curr_gen = int(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
        gpu.pcie_curr_width = int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))


# --------------------------------------------------------------------------------------
# System sampling
# --------------------------------------------------------------------------------------


def init_system() -> SystemState:
    """Initialize a SystemState and prime psutil's internal counters."""
    state = SystemState()
    if psutil is None:
        return state
    state.cpu_count_physical = psutil.cpu_count(logical=False) or 0
    state.cpu_count_logical = psutil.cpu_count(logical=True) or 0
    psutil.cpu_percent(interval=None, percpu=False)
    psutil.cpu_percent(interval=None, percpu=True)
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()
    state.prev_disk = (disk.read_bytes, disk.write_bytes) if disk else None
    state.prev_net = (net.bytes_recv, net.bytes_sent) if net else None
    state.prev_ts = time.monotonic()
    return state


def sample_system(state: SystemState) -> None:
    """Refresh `state` in place with the latest CPU / RAM / disk / net snapshot."""
    if psutil is None:
        return
    now = time.monotonic()
    dt = max(1e-3, now - state.prev_ts)

    state.cpu_pct = psutil.cpu_percent(interval=None, percpu=False)
    state.cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

    with contextlib.suppress(OSError):
        state.load_1, state.load_5, state.load_15 = psutil.getloadavg()

    vm = psutil.virtual_memory()
    state.ram_used_gib = (vm.total - vm.available) / (1024**3)
    state.ram_total_gib = vm.total / (1024**3)
    state.ram_pct = vm.percent

    sw = psutil.swap_memory()
    state.swap_used_gib = sw.used / (1024**3)
    state.swap_total_gib = sw.total / (1024**3)
    state.swap_pct = sw.percent

    disk = psutil.disk_io_counters()
    if disk and state.prev_disk is not None:
        state.disk_read_mbs = max(0.0, (disk.read_bytes - state.prev_disk[0]) / dt / 1e6)
        state.disk_write_mbs = max(0.0, (disk.write_bytes - state.prev_disk[1]) / dt / 1e6)
    if disk:
        state.prev_disk = (disk.read_bytes, disk.write_bytes)

    net = psutil.net_io_counters()
    if net and state.prev_net is not None:
        state.net_rx_mbs = max(0.0, (net.bytes_recv - state.prev_net[0]) / dt / 1e6)
        state.net_tx_mbs = max(0.0, (net.bytes_sent - state.prev_net[1]) / dt / 1e6)
    if net:
        state.prev_net = (net.bytes_recv, net.bytes_sent)

    state.prev_ts = now


# --------------------------------------------------------------------------------------
# DCGM probe (via `dcgmi dmon` subprocess)
# --------------------------------------------------------------------------------------


class DcgmProbe:
    """Streams DCGM profiling metrics by parsing `dcgmi dmon` stdout."""

    def __init__(self, gpu_count: int, interval_ms: int = 500) -> None:
        """Create a streamer for ``gpu_count`` GPUs sampled every ``interval_ms`` ms."""
        self.gpu_count = gpu_count
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen[str] | None = None
        self.latest: dict[int, dict[str, float]] = {i: {} for i in range(gpu_count)}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def cli_available() -> tuple[bool, str]:
        """Cheap precheck: dcgmi binary present and hostengine reachable."""
        if shutil.which("dcgmi") is None:
            return False, "dcgmi binary not found on PATH"
        try:
            r = subprocess.run(
                ["dcgmi", "discovery", "-l"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"dcgmi call failed: {exc}"
        if r.returncode != 0:
            msg = r.stderr.strip() or r.stdout.strip()
            return False, f"dcgmi exited {r.returncode}: {msg}"
        return True, ""

    @staticmethod
    def profiling_supported() -> tuple[bool, str]:
        """Try a brief watch on the profiling fields we actually want.

        Returns (True, "") if the profiling module loads and streams at least
        one data row, else (False, <reason>). Catches NVIDIA's GeForce restriction
        (DCGM error -33, "module not currently loaded").
        """
        fields = ",".join(str(f) for f in DCGM_FIELD_ORDER)
        try:
            r = subprocess.run(  # noqa: S603
                ["dcgmi", "dmon", "-e", fields, "-d", "500", "-c", "1"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"dcgmi dmon timed out or failed: {exc}"
        combined = r.stdout + r.stderr
        if "Error" in combined or "-33" in combined:
            first_err = next((ln.strip() for ln in combined.splitlines() if "Error" in ln), combined.strip())
            return False, first_err
        if not any(line.lstrip().startswith("GPU ") for line in r.stdout.splitlines()):
            return False, "dcgmi dmon returned no GPU data rows"
        return True, ""

    def start(self) -> None:
        """Spawn the `dcgmi dmon` subprocess and the background reader thread."""
        fields = ",".join(str(f) for f in DCGM_FIELD_ORDER)
        cmd = ["dcgmi", "dmon", "-e", fields, "-d", str(self.interval_ms)]
        logger.info("Starting DCGM stream: {}", " ".join(cmd))
        self.proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line or not line.startswith("GPU "):
                continue
            parts = line.split()
            if len(parts) < 2 + len(DCGM_FIELD_ORDER):
                continue
            try:
                gpu_idx = int(parts[1])
            except ValueError:
                continue
            snap: dict[str, float] = {}
            for fid, token in zip(DCGM_FIELD_ORDER, parts[2 : 2 + len(DCGM_FIELD_ORDER)], strict=True):
                try:
                    snap[DCGM_FIELD_NAMES[fid]] = float(token) * 100.0
                except ValueError:
                    snap[DCGM_FIELD_NAMES[fid]] = float("nan")
            with self._lock:
                self.latest[gpu_idx] = snap

    def snapshot(self, gpu_idx: int) -> dict[str, float]:
        """Return the most recent profiling snapshot for `gpu_idx`, or empty dict."""
        with self._lock:
            return dict(self.latest.get(gpu_idx, {}))

    def stop(self) -> None:
        """Signal the reader thread to exit and terminate the `dcgmi` subprocess."""
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=2)


# --------------------------------------------------------------------------------------
# Process enumeration
# --------------------------------------------------------------------------------------


def _collect_processes(gpus: list[GpuState]) -> list[dict]:
    """Enumerate GPU processes via NVML. Annotate with psutil when available."""
    out: list[dict] = []
    for g in gpus:
        handle = pynvml.nvmlDeviceGetHandleByIndex(g.index)
        seen: dict[int, dict] = {}
        for attr, ptype in [
            ("nvmlDeviceGetComputeRunningProcesses_v3", "C"),
            ("nvmlDeviceGetGraphicsRunningProcesses_v3", "G"),
        ]:
            try:
                fn = getattr(pynvml, attr)
            except AttributeError:
                fn = getattr(pynvml, attr.replace("_v3", ""), None)
            if fn is None:
                continue
            with contextlib.suppress(pynvml.NVMLError):
                for p in fn(handle):
                    row = seen.setdefault(p.pid, {"type": ptype, "mem": 0})
                    row["type"] = "C+G" if row["type"] != ptype else row["type"]
                    row["mem"] = max(row.get("mem", 0), p.usedGpuMemory or 0)
        for pid, info in seen.items():
            entry: dict = {
                "gpu": g.index,
                "pid": pid,
                "type": info["type"],
                "mem_mb": info["mem"] / 1e6,
                "user": "?",
                "cmd": "?",
                "cpu_pct": float("nan"),
                "rss_mb": float("nan"),
            }
            if psutil is not None:
                with contextlib.suppress(psutil.Error):
                    proc = psutil.Process(pid)
                    entry["user"] = proc.username()
                    cmdline = proc.cmdline()
                    entry["cmd"] = " ".join(cmdline) if cmdline else proc.name()
                    entry["cpu_pct"] = proc.cpu_percent(interval=None)
                    entry["rss_mb"] = proc.memory_info().rss / 1e6
            out.append(entry)
    out.sort(key=lambda r: (-r["mem_mb"], r["gpu"], r["pid"]))
    return out


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _clamp_pct(pct: float) -> float:
    """Clamp a percentage to [0, 100], treating NaN as 0."""
    if math.isnan(pct):
        return 0.0
    return max(0.0, min(100.0, pct))


def _fmt_bandwidth(mbs: float) -> str:
    """Format a bandwidth in MB/s, auto-promoting to GB/s for >= 1 GB/s."""
    if mbs < 1024:
        return f"{mbs:.1f} MB/s"
    return f"{mbs / 1024:.2f} GB/s"


def _pct_color(pct: float) -> str:
    """nvitop-style threshold colour for inline Rich markup."""
    p = _clamp_pct(pct)
    if p < 30:
        return "green"
    if p < 70:
        return "yellow"
    return "red"


# --------------------------------------------------------------------------------------
# Textual widgets
# --------------------------------------------------------------------------------------


class InfoCard(Static):
    """Bordered key/value card.

    Content is a Rich-markup string; rows are updated atomically via
    :meth:`update_rows` so paint flicker is avoided.
    """

    DEFAULT_CSS = """
    InfoCard {
        border: round $accent;
        padding: 0 1;
        height: auto;
        width: 1fr;
        margin: 0 1 0 0;
        content-align: left top;
    }
    """

    def __init__(self, title: str, card_id: str) -> None:
        """Build a card with the given border ``title`` and widget ``card_id``."""
        super().__init__(id=card_id)
        self.border_title = title

    def update_rows(self, rows: list[tuple[str, str]]) -> None:
        """Repaint the card body with (label, value) pairs."""
        max_k = max((len(k) for k, _ in rows), default=0)
        body = "\n".join(f"[bold cyan]{k:<{max_k}}[/]  {v}" for k, v in rows)
        self.update(body)


class TimeSeriesPlot(Widget):
    """Multi-series overlaid time-series chart with interleaved colours.

    Each series keeps its own ring buffer of ``(wall_timestamp, value)`` points.
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
        series: list[SeriesDef],
        chart_title: str,
        plot_id: str,
    ) -> None:
        """Build a chart for ``series`` (name, ansi256_idx, label, default_visible) tuples."""
        super().__init__(id=plot_id)
        self._series_defs = series
        self._data: dict[str, deque[tuple[float, float]]] = {
            name: deque(maxlen=HISTORY_LEN) for name, _, _, _ in series
        }
        self._visible: dict[str, bool] = {name: default for name, _, _, default in series}
        self._chart_title = chart_title
        self._y_max = 100.0

    def push(self, samples: dict[str, float], *, ts: float | None = None) -> None:
        """Append the latest sample for any series named in ``samples``."""
        when = ts if ts is not None else time.time()
        for name, value in samples.items():
            if name in self._data:
                self._data[name].append((when, value))

    def set_visible(self, name: str, *, visible: bool) -> None:
        """Toggle a series on/off and force an immediate redraw."""
        if name in self._visible:
            self._visible[name] = visible
            self.refresh()

    def set_y_max(self, y_max: float) -> None:
        """Override the vertical upper bound (used when series are MB/s, not %)."""
        self._y_max = max(1.0, y_max)

    def replot(self) -> None:
        """Public redraw entry point; call after any batch of ``push`` calls."""
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

    @staticmethod
    def _resample(values: list[float], width: int) -> list[float]:
        """Stretch/compress ``values`` to exactly ``width`` points by linear interpolation.

        All series are pushed in lockstep (same count, same timestamps), so a
        uniform stretch keeps them time-aligned with one another on the canvas.
        """
        n = len(values)
        if n == 0:
            return [0.0] * width
        if n == 1 or width == 1:
            return [values[-1]] * width
        out: list[float] = []
        for i in range(width):
            pos = i * (n - 1) / (width - 1)
            lo = int(pos)
            hi = min(n - 1, lo + 1)
            frac = pos - lo
            out.append(values[lo] * (1.0 - frac) + values[hi] * frac)
        return out

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
            ys = self._resample([v for _, v in self._data[name]], plot_w)
            prev: int | None = None
            for cx in range(plot_w):
                value = ys[cx] if math.isfinite(ys[cx]) else 0.0
                frac = min(1.0, max(0.0, value / self._y_max))
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
                    line.append(" ")
                    continue
                holders = owners[(cx, cy)]
                winner = holders[cx % len(holders)]  # rotate colour by column -> interleave
                line.append(glyph, style=f"color({colours[winner]})")
            lines.append(line)
        lines.append(self._time_axis(width, gutter))
        return Text("\n").join(lines)

    def _time_axis(self, width: int, gutter: int) -> Text:
        """Bottom axis: oldest sample time on the left, newest on the right."""
        first: float | None = None
        last: float | None = None
        for dq in self._data.values():
            if dq:
                first = dq[0][0] if first is None else min(first, dq[0][0])
                last = dq[-1][0] if last is None else max(last, dq[-1][0])
        row = [" "] * width
        if first is not None and last is not None:
            lo = time.strftime("%H:%M:%S", time.localtime(first))
            hi = time.strftime("%H:%M:%S", time.localtime(last))
            for i, ch in enumerate(lo):
                if gutter + i < width:
                    row[gutter + i] = ch
            for i, ch in enumerate(hi):
                if 0 <= width - len(hi) + i < width:
                    row[width - len(hi) + i] = ch
        return Text("".join(row), style=AXIS_STYLE)


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


# --------------------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------------------


class CvtopApp(App):
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

    def __init__(self, *, interval: float, no_dcgm: bool) -> None:
        """Configure sampling ``interval`` (seconds) and whether to skip DCGM."""
        super().__init__()
        self.interval = interval
        self.no_dcgm = no_dcgm
        self.gpus: list[GpuState] = []
        self.sys_state: SystemState = SystemState()
        self.dcgm: DcgmProbe | None = None
        self.have_profiling = False
        self.dcgm_note = ""
        self.paused = False
        self._start_ts = time.monotonic()
        self._compute_series: list[SeriesDef] = []
        self._nv_driver = ""
        self._cuda_ver = ""

    # -- bring-up --------------------------------------------------------------------

    def _prepare_sources(self) -> None:
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
                yield TimeSeriesPlot(all_series, "All metrics", plot_id="all-plot")
                yield SeriesToggles("all-plot", all_series)
            with TabPane("Compute", id="tab-compute"), Vertical():
                yield TimeSeriesPlot(
                    self._compute_series, "Compute: CPU, GPU SM, encode/decode", plot_id="compute-plot"
                )
                yield SeriesToggles("compute-plot", self._compute_series)
            with TabPane("Memory", id="tab-memory"), Vertical():
                yield TimeSeriesPlot(MEMORY_SERIES, "Memory: RAM, GPU VRAM, VRAM bandwidth", plot_id="memory-plot")
                yield SeriesToggles("memory-plot", MEMORY_SERIES)
            with TabPane("System", id="tab-system"), Vertical():
                yield TimeSeriesPlot(SYSTEM_SERIES, "System: PCIe, GPU power, disk, network", plot_id="system-plot")
                yield SeriesToggles("system-plot", SYSTEM_SERIES)

        yield VerticalScroll(DataTable(id="procs", zebra_stripes=True))

        yield Footer()

    def _header_name(self) -> str:
        mode = "DCGM" if self.have_profiling else "NVML"
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
        all_plot = self.query_one("#all-plot", TimeSeriesPlot)
        compute_plot = self.query_one("#compute-plot", TimeSeriesPlot)
        memory_plot = self.query_one("#memory-plot", TimeSeriesPlot)
        system_plot = self.query_one("#system-plot", TimeSeriesPlot)

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

        compute_plot.push(compute, ts=now)
        memory_plot.push(memory, ts=now)
        system_plot.push(system, ts=now)
        # The "All" tab overlays everything; push() only updates the names it knows.
        for samples in (compute, memory, system):
            all_plot.push(samples, ts=now)

        for plot in (all_plot, compute_plot, memory_plot, system_plot):
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
        procs = _collect_processes(self.gpus)
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
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--interval", type=float, default=0.5, help="sampling interval, seconds")
    ap.add_argument("--no-dcgm", action="store_true", help="skip DCGM probe entirely")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    app = CvtopApp(interval=args.interval, no_dcgm=args.no_dcgm)
    app.run()


if __name__ == "__main__":
    main()
