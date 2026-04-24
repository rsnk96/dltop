"""DL-oriented GPU monitor TUI.

Splits live GPU utilization into two categories:

    * Compute (DL): SM active, Tensor pipe, FP32/FP16/FP64 pipes -- via DCGM profiling
    * Media engines: NVENC, NVDEC -- via NVML

When DCGM profiling metrics are unavailable (e.g. consumer GeForce cards where
NVIDIA gates profiling fields, or DCGM not installed), we transparently fall
back to NVML's single overall SM% and show a footer note telling the user how
to enable the full split on a data-center GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass, field

from loguru import logger
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

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

SPARK_CHARS = " ⡀⣀⣄⣤⣦⣶⣷⣿"  # Braille dot progression (fill from bottom); less harsh than solid blocks
FULL_CHAR = "⣿"  # used as the full-cell fill in multi-row graphs
HISTORY_LEN = 240  # enough to fill a 200+ col timeseries row at 0.5s cadence
TRANSIENT_NOTES_SECONDS = 3.0  # how long startup-only banners (DCGM status, hint lines) stay visible

# Vertical rows per graph. Taller = more resolution, but eats vertical space.
# Using 4 rows gives 32 sub-pixel levels per column (4 * 8 sub-blocks per char).
GRAPH_HEIGHT_TALL = 5  # single "primary" compute metric (GeForce fallback)
GRAPH_HEIGHT_MEDIUM = 3  # one-of-many DCGM pipe metrics and system CPU/RAM
GRAPH_HEIGHT_SHORT = 2  # secondary media metrics (NVENC/NVDEC)

# Compute section colors (per field, consistent across bar/sparkline/table)
COLOR_SM = "bright_cyan"
COLOR_TENSOR = "bright_magenta"
COLOR_FP32 = "bright_blue"
COLOR_FP16 = "blue"
COLOR_FP64 = "blue"
COLOR_NVENC = "bright_yellow"
COLOR_NVDEC = "yellow"
COLOR_MBW = "bright_green"
COLOR_FREQ = "bright_white"
COLOR_POWER = "bright_red"
COLOR_EMPTY = "grey35"

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


# --------------------------------------------------------------------------------------
# GPU state
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
    fan_pct: float = float("nan")  # NaN if no fan sensor (common on laptops)
    sm_clock_mhz: float = 0.0
    mem_clock_mhz: float = 0.0
    sm_clock_max_mhz: float = 0.0
    mem_clock_max_mhz: float = 0.0
    mbw_pct: float = 0.0  # NVML util.memory -- % of time memory controller was busy
    nvenc_pct: float = 0.0
    nvdec_pct: float = 0.0
    sm_overall_pct: float = 0.0  # NVML util.gpu -- combined CUDA+Tensor+RT
    pcie_tx_mbs: float = 0.0  # PCIe transmit MB/s (GPU -> host)
    pcie_rx_mbs: float = 0.0  # PCIe receive MB/s (host -> GPU)
    dcgm: dict[str, float] = field(default_factory=dict)

    sm_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    tensor_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    fp32_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    fp16_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    fp64_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    nvenc_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    nvdec_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    mbw_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    freq_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))  # SM clock % of max
    power_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))  # power % of limit


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

    cpu_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    ram_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    disk_r_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    net_rx_hist: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    # Previous counters + timestamp for rate calculation
    _prev_disk: tuple[int, int] | None = None
    _prev_net: tuple[int, int] | None = None
    _prev_ts: float = 0.0


# --------------------------------------------------------------------------------------
# UI state (view toggle + pause)
# --------------------------------------------------------------------------------------


VIEW_MODES = ("all", "gpu", "system")


@dataclass
class UIState:
    """Interactive UI flags mutated by the keyboard thread.

    Writes happen from the keyboard thread; reads happen from the render loop.
    A lock is not strictly required because CPython's GIL makes bool/str
    assignments atomic, and the loop re-reads each frame.
    """

    view_mode: str = "all"
    paused: bool = False
    stop: bool = False


def _start_keyboard_thread(ui: UIState) -> threading.Thread | None:
    """Spawn a daemon thread that reads single keypresses from stdin.

    Keys:
      q / Q       -> request shutdown
      space / p   -> toggle pause (freezes sampling; graphs stop advancing)
      v / Tab     -> cycle view: all -> gpu -> system -> all
      g           -> jump to GPU-only view
      s           -> jump to system-only view
      a           -> jump to combined (all) view

    If stdin isn't a TTY (e.g. piped), no thread is started; the UI still runs
    but is non-interactive.
    """
    if not sys.stdin.isatty():
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def _restore() -> None:
        with contextlib.suppress(termios.error):
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _loop() -> None:
        try:
            tty.setcbreak(fd)
            while not ui.stop:
                # 0.2 s poll so Ctrl-C / stop flag is noticed quickly
                if not select.select([sys.stdin], [], [], 0.2)[0]:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                if ch in ("q", "Q"):
                    ui.stop = True
                elif ch in (" ", "p", "P"):
                    ui.paused = not ui.paused
                elif ch in ("v", "V", "\t"):
                    i = VIEW_MODES.index(ui.view_mode)
                    ui.view_mode = VIEW_MODES[(i + 1) % len(VIEW_MODES)]
                elif ch in ("g", "G"):
                    ui.view_mode = "gpu"
                elif ch in ("s", "S"):
                    ui.view_mode = "system"
                elif ch in ("a", "A"):
                    ui.view_mode = "all"
        finally:
            _restore()

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


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
        gpus.append(GpuState(index=i, name=name, mem_total_mb=mem.total / 1e6))
    return gpus


def sample_nvml(gpu: GpuState) -> None:
    """Populate NVML-derived fields on `gpu` in place."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu.mem_used_mb = mem.used / 1e6

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
    # PCIe throughput (KB/s over ~20ms window) -> MB/s
    with contextlib.suppress(pynvml.NVMLError):
        gpu.pcie_tx_mbs = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
        gpu.pcie_rx_mbs = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0


# --------------------------------------------------------------------------------------
# System (CPU / RAM / disk / net) sampling
# --------------------------------------------------------------------------------------


def init_system() -> SystemState:
    """Initialize a SystemState and prime psutil's internal counters."""
    state = SystemState()
    if psutil is None:
        return state
    state.cpu_count_physical = psutil.cpu_count(logical=False) or 0
    state.cpu_count_logical = psutil.cpu_count(logical=True) or 0
    # Prime cpu_percent so first real sample isn't 0 or a massive spike
    psutil.cpu_percent(interval=None, percpu=False)
    psutil.cpu_percent(interval=None, percpu=True)
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()
    state._prev_disk = (disk.read_bytes, disk.write_bytes) if disk else None
    state._prev_net = (net.bytes_recv, net.bytes_sent) if net else None
    state._prev_ts = time.monotonic()
    return state


def sample_system(state: SystemState) -> None:
    """Refresh `state` in place with the latest CPU / RAM / disk / net snapshot."""
    if psutil is None:
        return
    now = time.monotonic()
    dt = max(1e-3, now - state._prev_ts)

    state.cpu_pct = psutil.cpu_percent(interval=None, percpu=False)
    state.cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

    with contextlib.suppress(OSError):
        load = psutil.getloadavg()
        state.load_1, state.load_5, state.load_15 = load

    vm = psutil.virtual_memory()
    state.ram_used_gib = (vm.total - vm.available) / (1024**3)
    state.ram_total_gib = vm.total / (1024**3)
    state.ram_pct = vm.percent

    sw = psutil.swap_memory()
    state.swap_used_gib = sw.used / (1024**3)
    state.swap_total_gib = sw.total / (1024**3)
    state.swap_pct = sw.percent

    disk = psutil.disk_io_counters()
    if disk and state._prev_disk is not None:
        dr = (disk.read_bytes - state._prev_disk[0]) / dt / 1e6
        dw = (disk.write_bytes - state._prev_disk[1]) / dt / 1e6
        state.disk_read_mbs = max(0.0, dr)
        state.disk_write_mbs = max(0.0, dw)
    if disk:
        state._prev_disk = (disk.read_bytes, disk.write_bytes)

    net = psutil.net_io_counters()
    if net and state._prev_net is not None:
        rx = (net.bytes_recv - state._prev_net[0]) / dt / 1e6
        tx = (net.bytes_sent - state._prev_net[1]) / dt / 1e6
        state.net_rx_mbs = max(0.0, rx)
        state.net_tx_mbs = max(0.0, tx)
    if net:
        state._prev_net = (net.bytes_recv, net.bytes_sent)

    state._prev_ts = now
    state.cpu_hist.append(state.cpu_pct)
    state.ram_hist.append(state.ram_pct)
    state.disk_r_hist.append(state.disk_read_mbs)
    state.net_rx_hist.append(state.net_rx_mbs)


# --------------------------------------------------------------------------------------
# DCGM probe (via `dcgmi dmon` subprocess)
# --------------------------------------------------------------------------------------


class DcgmProbe:
    """Streams DCGM profiling metrics by parsing `dcgmi dmon` stdout.

    Subprocess is used instead of pydcgm because DCGM's Python bindings live in
    install-dependent paths (snap vs apt vs CUDA repo). The CLI is the one stable
    entry point across installs.
    """

    def __init__(self, gpu_count: int, interval_ms: int = 500) -> None:
        self.gpu_count = gpu_count
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen[str] | None = None
        self.latest: dict[int, dict[str, float]] = {i: {} for i in range(gpu_count)}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._startup_error: str = ""

    @staticmethod
    def cli_available() -> tuple[bool, str]:
        """Cheap precheck: dcgmi binary present and hostengine reachable."""
        if shutil.which("dcgmi") is None:
            return False, "dcgmi binary not found on PATH"
        try:
            r = subprocess.run(
                ["dcgmi", "discovery", "-l"],
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
        one data row, else (False, <reason>). This catches NVIDIA's GeForce
        restriction (DCGM error -33, "module not currently loaded").
        """
        fields = ",".join(str(f) for f in DCGM_FIELD_ORDER)
        try:
            r = subprocess.run(
                ["dcgmi", "dmon", "-e", fields, "-d", "500", "-c", "1"],
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
        fields = ",".join(str(f) for f in DCGM_FIELD_ORDER)
        cmd = ["dcgmi", "dmon", "-e", fields, "-d", str(self.interval_ms)]
        logger.info("Starting DCGM stream: {}", " ".join(cmd))
        self.proc = subprocess.Popen(  # noqa: S603 -- cmd is a fixed argv list, no shell
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
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
        with self._lock:
            return dict(self.latest.get(gpu_idx, {}))

    def stop(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=2)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

# Room reserved around a bar within the inner panel area:
# 2 chars leading indent + <label_width> + 2 chars gap + <bar> + 2 chars gap + 7 chars "100.0%"
LABEL_WIDTH = 13
PCT_WIDTH = 7
BAR_FIXED_OVERHEAD = 2 + LABEL_WIDTH + 2 + 2 + PCT_WIDTH
# Panel borders + internal padding eat ~4 cols on each side.
PANEL_CHROME = 8


def _clamp_pct(pct: float) -> float:
    if pct != pct:  # NaN
        return 0.0
    return max(0.0, min(100.0, pct))


def _bar_color(pct: float) -> str:
    """Green / yellow / red based on utilization, nvitop-style."""
    p = _clamp_pct(pct)
    if p < 30:
        return "green"
    if p < 70:
        return "yellow"
    return "red"


def _ts_block(
    label: str,
    history: deque[float],
    current: float,
    spark_width: int,
    height: int,
    *,
    fill_color: str | None = None,
    readout_override: str | None = None,
) -> Group:
    """Build a multi-row filled-area timeseries chart.

    Renders an nvitop-style stacked block chart with `height` rows. Each column
    is one historical sample, filled from the bottom up. With 8 sub-pixel levels
    per block character, the effective vertical resolution is `height * 8`.

    - If `fill_color` is given (pipe-identity metrics like Tensor, NVENC), every
      filled cell uses that colour so the pipe is identifiable at a glance.
    - If `fill_color` is None (threshold metrics like SM, CPU, RAM), each column
      is coloured by its own value (green/yellow/red) — temporal heatmap showing
      when the metric crossed thresholds.

    Returns a Group of `height` Text lines: the top row shows the label and
    current-value readout, remaining rows are graph-only (label column padded).
    """
    curr = _clamp_pct(current)
    # Shrink spark_width if the readout is longer than the default 7-char budget
    # (e.g. "2505 MHz", "80/115 W"), so the row doesn't wrap on narrow terminals.
    readout_len = (2 + len(readout_override)) if readout_override else PCT_WIDTH
    extra = max(0, readout_len - PCT_WIDTH)
    spark_width = max(10, spark_width - extra)
    data = list(history)[-spark_width:] if history else []
    pad = spark_width - len(data)

    # Per-column (eighths_filled, style) — eighths_filled in [0, height*8]
    columns: list[tuple[int, str]] = []
    for _ in range(pad):
        columns.append((0, COLOR_EMPTY))
    for v in data:
        vv = _clamp_pct(v)
        eighths = int(round(vv / 100.0 * height * 8))
        eighths = max(0, min(height * 8, eighths))
        cell_style = fill_color or _bar_color(vv)
        columns.append((eighths, cell_style))

    readout_style = fill_color or _bar_color(curr)
    readout = f"  {readout_override}" if readout_override else f"  {curr:5.1f}%"
    label_prefix = f"  {label:<{LABEL_WIDTH}}  "
    blank_prefix = "  " + " " * LABEL_WIDTH + "  "

    lines: list[Text] = []
    for row_idx in range(height):
        # Row 0 = top of chart, row (height-1) = bottom.
        row_from_bottom = height - 1 - row_idx
        line = Text()
        if row_idx == 0:
            line.append(label_prefix, style="bold white")
        else:
            line.append(blank_prefix)
        # The "no data" baseline cells: draw a thin flat line at the bottom row only
        for col_idx, (eighths, style) in enumerate(columns):
            if col_idx < pad:
                # Leading pad region: render a dim baseline "▁" on the bottom row
                if row_from_bottom == 0:
                    line.append("▁", style=COLOR_EMPTY)
                else:
                    line.append(" ")
                continue
            full = eighths // 8
            partial = eighths % 8
            if row_from_bottom < full:
                line.append(FULL_CHAR, style=style)
            elif row_from_bottom == full and partial > 0:
                line.append(SPARK_CHARS[partial], style=style)
            else:
                line.append(" ")
        if row_idx == 0:
            line.append(readout, style=readout_style)
        else:
            line.append(" " * len(readout))
        lines.append(line)
    return Group(*lines)


def _kv_line(parts: list[tuple[str, str, str]], width: int) -> Text:
    """Render a "k v │ k v │ ..." line, dropping trailing segments that don't fit.

    Each tuple is (label, value, value_style). Label is styled cyan, separator grey.
    Segments are appended greedily; any that would overflow `width` are skipped so
    the line never wraps on narrow terminals.
    """
    line = Text("  ")
    sep = "  │  "
    for i, (label, value, vstyle) in enumerate(parts):
        seg = Text()
        if i > 0:
            seg.append(sep, style="grey35")
        seg.append(label, style="cyan")
        seg.append(value, style=vstyle)
        if line.cell_len + seg.cell_len > width - 2:
            break
        line.append_text(seg)
    return line


def _quick_status(gpu: GpuState, width: int) -> Text:
    """nvitop-style one-line summary: key|value pairs separated by │."""
    fan = "—" if gpu.fan_pct != gpu.fan_pct else f"{gpu.fan_pct:.0f}%"
    mem_used = gpu.mem_used_mb / 1024.0
    mem_total = gpu.mem_total_mb / 1024.0
    mem_pct = (gpu.mem_used_mb / gpu.mem_total_mb * 100.0) if gpu.mem_total_mb else 0.0
    pwr_pct = (gpu.power_w / gpu.power_limit_w * 100.0) if gpu.power_limit_w else 0.0

    parts: list[tuple[str, str, str]] = [
        ("Temp ", f"{gpu.temp_c:.0f}°C", _bar_color((gpu.temp_c - 30) * 2)),
        ("Fan ", fan, "bold white"),
        ("Pwr ", f"{gpu.power_w:.0f}/{gpu.power_limit_w:.0f}W", _bar_color(pwr_pct)),
        ("Mem ", f"{mem_used:.2f}/{mem_total:.2f}GiB ({mem_pct:.0f}%)", _bar_color(mem_pct)),
        ("PCIe↑ ", f"{gpu.pcie_tx_mbs:.0f}MB/s", "bold white"),
        ("PCIe↓ ", f"{gpu.pcie_rx_mbs:.0f}MB/s", "bold white"),
        ("SM ", f"{gpu.sm_clock_mhz:.0f}MHz", "white"),
        ("MClk ", f"{gpu.mem_clock_mhz:.0f}MHz", "white"),
    ]
    return _kv_line(parts, width)


def _core_heatmap(per_core: list[float]) -> Text:
    """One-block-per-core utilization heatmap, e.g. ██▆▆▄▄▁▁ color-coded."""
    out = Text()
    for v in per_core:
        vv = _clamp_pct(v)
        bins = len(SPARK_CHARS) - 1
        ch = SPARK_CHARS[int(round(vv / 100.0 * bins))]
        out.append(ch, style=_bar_color(vv))
    return out


def render_system_panel(sys_state: SystemState, gpus: list[GpuState], *, width: int) -> Panel:
    """System-wide panel: CPU, RAM, disk, net -- context for visual analytics pipelines."""
    if psutil is None:
        body = Text("psutil not installed -- pip install psutil for system metrics", style="yellow")
        return Panel(body, title=" System ", title_align="left", box=box.HEAVY, border_style="bright_blue")

    inner_width = max(30, width - PANEL_CHROME)
    spark_width = max(10, inner_width - BAR_FIXED_OVERHEAD)

    body: list = []

    # CPU timeseries (threshold-coloured: green/yellow/red per cell)
    body.append(_ts_block("CPU", sys_state.cpu_hist, sys_state.cpu_pct, spark_width, GRAPH_HEIGHT_MEDIUM))

    if sys_state.cpu_per_core:
        heatmap_label = f"  {'Cores':<{LABEL_WIDTH}}  "
        heatmap = Text(heatmap_label, style="bold white")
        heatmap.append_text(_core_heatmap(sys_state.cpu_per_core))
        heatmap.append(
            f"   {sys_state.cpu_count_physical}P / {sys_state.cpu_count_logical}L   "
            f"load {sys_state.load_1:.2f} {sys_state.load_5:.2f} {sys_state.load_15:.2f}",
            style="grey50",
        )
        body.append(heatmap)

    # RAM timeseries (threshold-coloured)
    body.append(_ts_block("RAM", sys_state.ram_hist, sys_state.ram_pct, spark_width, GRAPH_HEIGHT_MEDIUM))
    ram_info = Text()
    ram_info.append(f"  {'':<{LABEL_WIDTH}}  ", style="bold white")
    ram_info.append(
        f"{sys_state.ram_used_gib:.1f} / {sys_state.ram_total_gib:.1f} GiB",
        style="bright_white",
    )
    if sys_state.swap_total_gib > 0:
        swap_style = _bar_color(sys_state.swap_pct)
        ram_info.append(
            f"   swap {sys_state.swap_used_gib:.2f} / {sys_state.swap_total_gib:.1f} GiB ({sys_state.swap_pct:.0f}%)",
            style=swap_style,
        )
    body.append(ram_info)
    body.append(Text(""))

    # I/O line -- disk + net + aggregate PCIe (dynamic, drops segments on narrow screens)
    pcie_tx_total = sum(g.pcie_tx_mbs for g in gpus)
    pcie_rx_total = sum(g.pcie_rx_mbs for g in gpus)
    io_parts: list[tuple[str, str, str]] = [
        ("Disk R ", f"{sys_state.disk_read_mbs:6.1f} MB/s", "bright_cyan"),
        ("Disk W ", f"{sys_state.disk_write_mbs:6.1f} MB/s", "cyan"),
        ("Net RX ", f"{sys_state.net_rx_mbs:6.1f} MB/s", "bright_green"),
        ("Net TX ", f"{sys_state.net_tx_mbs:6.1f} MB/s", "green"),
        ("GPU PCIe↑ ", f"{pcie_tx_total:6.1f} MB/s", "bright_yellow"),
        ("GPU PCIe↓ ", f"{pcie_rx_total:6.1f} MB/s", "yellow"),
    ]
    body.append(_kv_line(io_parts, inner_width))

    title = Text(" System ", style="bold white on blue")
    return Panel(
        Group(*body),
        title=title,
        title_align="left",
        box=box.HEAVY,
        border_style="bright_blue",
        padding=(0, 2),
    )


def render_gpu_panel(gpu: GpuState, *, have_profiling: bool, width: int, show_transient: bool) -> Panel:
    """Render one GPU as a single nvitop-style panel, all metrics as timeseries."""
    inner_width = max(30, width - PANEL_CHROME)
    spark_width = max(10, inner_width - BAR_FIXED_OVERHEAD)

    body: list = []
    body.append(_quick_status(gpu, inner_width))
    body.append(Text(""))

    # Compute (DL) section — every row is a tall filled-area timeseries
    body.append(Rule(Text(" Compute (DL) ", style="bold bright_cyan"), style="bright_cyan", align="left"))
    if have_profiling and gpu.dcgm:
        h = GRAPH_HEIGHT_MEDIUM
        body.append(_ts_block("SM Active", gpu.sm_hist, gpu.dcgm.get("sm_active", 0.0), spark_width, h))
        body.append(
            _ts_block(
                "Tensor", gpu.tensor_hist, gpu.dcgm.get("tensor_active", 0.0), spark_width, h, fill_color=COLOR_TENSOR
            )
        )
        body.append(
            _ts_block(
                "FP32 Pipe", gpu.fp32_hist, gpu.dcgm.get("fp32_active", 0.0), spark_width, h, fill_color=COLOR_FP32
            )
        )
        body.append(
            _ts_block(
                "FP16 Pipe", gpu.fp16_hist, gpu.dcgm.get("fp16_active", 0.0), spark_width, h, fill_color=COLOR_FP16
            )
        )
        body.append(
            _ts_block(
                "FP64 Pipe", gpu.fp64_hist, gpu.dcgm.get("fp64_active", 0.0), spark_width, h, fill_color=COLOR_FP64
            )
        )
    else:
        body.append(_ts_block("GPU (SM)", gpu.sm_hist, gpu.sm_overall_pct, spark_width, GRAPH_HEIGHT_TALL))
        if show_transient:
            body.append(Text("  (DCGM profiling unavailable — bar lumps CUDA + Tensor + RT)", style="dim italic"))

    # Memory bandwidth — time % memory controller is busy (nvitop's "MEM %")
    body.append(_ts_block("MBW", gpu.mbw_hist, gpu.mbw_pct, spark_width, GRAPH_HEIGHT_MEDIUM, fill_color=COLOR_MBW))

    body.append(Text(""))

    # Frequency & Power section — normalized to max / limit so both fit 0-100
    body.append(Rule(Text(" Frequency & Power ", style="bold bright_red"), style="bright_red", align="left"))
    freq_pct = (gpu.sm_clock_mhz / gpu.sm_clock_max_mhz * 100.0) if gpu.sm_clock_max_mhz else 0.0
    power_pct = (gpu.power_w / gpu.power_limit_w * 100.0) if gpu.power_limit_w else 0.0
    freq_readout = f"{gpu.sm_clock_mhz:.0f}/{gpu.sm_clock_max_mhz:.0f} MHz"
    power_readout = f"{gpu.power_w:.0f}/{gpu.power_limit_w:.0f} W"
    body.append(
        _ts_block(
            "SM Clock",
            gpu.freq_hist,
            freq_pct,
            spark_width,
            GRAPH_HEIGHT_SHORT,
            fill_color=COLOR_FREQ,
            readout_override=freq_readout,
        )
    )
    body.append(
        _ts_block(
            "Power",
            gpu.power_hist,
            power_pct,
            spark_width,
            GRAPH_HEIGHT_SHORT,
            fill_color=COLOR_POWER,
            readout_override=power_readout,
        )
    )

    body.append(Text(""))

    # Media section
    body.append(Rule(Text(" Media Engines ", style="bold bright_yellow"), style="bright_yellow", align="left"))
    body.append(
        _ts_block("NVENC", gpu.nvenc_hist, gpu.nvenc_pct, spark_width, GRAPH_HEIGHT_SHORT, fill_color=COLOR_NVENC)
    )
    body.append(
        _ts_block("NVDEC", gpu.nvdec_hist, gpu.nvdec_pct, spark_width, GRAPH_HEIGHT_SHORT, fill_color=COLOR_NVDEC)
    )

    title = Text()
    title.append(f" GPU {gpu.index} ", style="bold white on blue")
    title.append(f" {gpu.name} ", style="bold bright_cyan")
    return Panel(
        Group(*body),
        title=title,
        title_align="left",
        box=box.HEAVY,
        border_style="bright_cyan",
        padding=(0, 2),
    )


# --------------------------------------------------------------------------------------
# Process table
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


def render_processes_panel(procs: list[dict], width: int) -> Panel:
    """nvitop-style processes panel.

    Fixed-width columns use `no_wrap=True` so Rich does not collapse them when
    the COMMAND column gets greedy; only COMMAND is allowed to truncate.
    """
    table = Table(
        box=None,
        expand=True,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("GPU", justify="right", width=3, no_wrap=True)
    table.add_column("PID", justify="right", width=7, no_wrap=True)
    table.add_column("USER", width=12, no_wrap=True, overflow="ellipsis")
    table.add_column("TYPE", justify="center", width=4, no_wrap=True)
    table.add_column("GPU-MEM", justify="right", width=11, no_wrap=True)
    table.add_column("%CPU", justify="right", width=5, no_wrap=True)
    table.add_column("HOST-MEM", justify="right", width=9, no_wrap=True)
    table.add_column("COMMAND", ratio=1, no_wrap=True, overflow="ellipsis")

    if not procs:
        table.add_row("—", "—", "—", "—", "—", "—", "—", Text("(no GPU processes)", style="dim italic"))
    else:
        for p in procs:
            mem_color = _bar_color(p["mem_mb"] / 1024.0 * 10)  # cheap heuristic
            table.add_row(
                Text(str(p["gpu"]), style="bright_cyan"),
                str(p["pid"]),
                p["user"],
                Text(p["type"], style="bold bright_magenta" if "C" in p["type"] else "bright_yellow"),
                Text(f"{p['mem_mb']:>6.0f} MiB", style=mem_color),
                "—" if p["cpu_pct"] != p["cpu_pct"] else f"{p['cpu_pct']:.0f}",
                "—" if p["rss_mb"] != p["rss_mb"] else f"{p['rss_mb']:.0f}M",
                p["cmd"],
            )

    title = Text(" Processes ", style="bold white on blue")
    return Panel(table, title=title, title_align="left", box=box.HEAVY, border_style="bright_blue", padding=(0, 1))


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--interval", type=float, default=0.5, help="sampling interval, seconds")
    ap.add_argument("--no-dcgm", action="store_true", help="skip DCGM probe entirely")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    gpus = init_nvml()
    if not gpus:
        sys.exit("No NVIDIA GPUs detected by NVML")
    sys_state = init_system()

    dcgm: DcgmProbe | None = None
    dcgm_note = ""
    have_profiling = False

    if args.no_dcgm:
        dcgm_note = "DCGM disabled by --no-dcgm. Showing NVML overall SM% (lumped CUDA+Tensor+RT)."
    else:
        ok_cli, err_cli = DcgmProbe.cli_available()
        if not ok_cli:
            dcgm_note = f"DCGM unavailable: {err_cli}\n\n{DCGM_INSTALL_HINT}"
        else:
            ok_prof, err_prof = DcgmProbe.profiling_supported()
            if ok_prof:
                dcgm = DcgmProbe(len(gpus), interval_ms=int(args.interval * 1000))
                dcgm.start()
                have_profiling = True
            else:
                dcgm_note = (
                    "DCGM present but profiling fields not available on this GPU.\n"
                    f"  {err_prof}\n\n{DCGM_INSTALL_HINT}"
                )

    console = Console()
    ui = UIState()

    def _sigint(*_: object) -> None:
        ui.stop = True

    signal.signal(signal.SIGINT, _sigint)
    _start_keyboard_thread(ui)

    driver = ""
    with contextlib.suppress(pynvml.NVMLError):
        raw = pynvml.nvmlSystemGetDriverVersion()
        driver = raw.decode() if isinstance(raw, bytes) else raw
    cuda_ver = ""
    with contextlib.suppress(pynvml.NVMLError):
        v = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_ver = f"{v // 1000}.{(v % 1000) // 10}"

    def _title_bar(width: int) -> Text:
        mode = "DCGM profiling" if have_profiling else "NVML only"
        mode_style = "bold bright_green" if have_profiling else "bold bright_yellow"
        left = Text()
        left.append(" DL-GPUMON ", style="bold white on blue")
        left.append(f"  Driver {driver or '?'}   CUDA {cuda_ver or '?'}   ", style="cyan")
        left.append("mode: ", style="white")
        left.append(mode, style=mode_style)
        left.append(f"   view: {ui.view_mode}", style="bold bright_cyan")
        if ui.paused:
            left.append("   ⏸ PAUSED", style="bold black on bright_yellow")
        right = Text("  [a]ll [g]pu [s]ys  [space]pause  [q]uit ", style="dim")
        pad = max(1, width - left.cell_len - right.cell_len)
        bar = Text()
        bar.append_text(left)
        bar.append(" " * pad)
        bar.append_text(right)
        return bar

    start_ts = time.monotonic()

    try:
        with Live(console=console, refresh_per_second=max(2, int(1 / args.interval)), screen=True) as live:
            while not ui.stop:
                # When paused, we freeze sampling entirely — current readouts and graph
                # stay at their pre-pause values. Keyboard thread still runs.
                if not ui.paused:
                    for g in gpus:
                        sample_nvml(g)
                        if dcgm is not None:
                            g.dcgm = dcgm.snapshot(g.index)

                        if have_profiling and g.dcgm:
                            g.sm_hist.append(g.dcgm.get("sm_active", 0.0))
                            g.tensor_hist.append(g.dcgm.get("tensor_active", 0.0))
                            g.fp32_hist.append(g.dcgm.get("fp32_active", 0.0))
                            g.fp16_hist.append(g.dcgm.get("fp16_active", 0.0))
                            g.fp64_hist.append(g.dcgm.get("fp64_active", 0.0))
                        else:
                            g.sm_hist.append(g.sm_overall_pct)
                        g.nvenc_hist.append(g.nvenc_pct)
                        g.nvdec_hist.append(g.nvdec_pct)
                        g.mbw_hist.append(g.mbw_pct)
                        freq_pct = (g.sm_clock_mhz / g.sm_clock_max_mhz * 100.0) if g.sm_clock_max_mhz else 0.0
                        power_pct = (g.power_w / g.power_limit_w * 100.0) if g.power_limit_w else 0.0
                        g.freq_hist.append(freq_pct)
                        g.power_hist.append(power_pct)

                    sample_system(sys_state)

                width = console.size.width
                elapsed = time.monotonic() - start_ts
                show_transient = elapsed < TRANSIENT_NOTES_SECONDS
                procs = _collect_processes(gpus)

                renderables: list = [_title_bar(width)]
                if ui.view_mode in ("all", "system"):
                    renderables.append(render_system_panel(sys_state, gpus, width=width))
                if ui.view_mode in ("all", "gpu"):
                    renderables.extend(
                        render_gpu_panel(g, have_profiling=have_profiling, width=width, show_transient=show_transient)
                        for g in gpus
                    )
                renderables.append(render_processes_panel(procs, width=width))
                if dcgm_note and show_transient:
                    remaining = TRANSIENT_NOTES_SECONDS - elapsed
                    title = Text()
                    title.append(" DCGM status ", style="bold black on yellow")
                    title.append(f"  (dismissing in {remaining:0.0f}s) ", style="dim yellow")
                    renderables.append(
                        Panel(
                            Text(dcgm_note, style="yellow"),
                            title=title,
                            title_align="left",
                            box=box.HEAVY,
                            border_style="yellow",
                            padding=(0, 1),
                        ),
                    )
                live.update(Group(*renderables))
                time.sleep(args.interval)
    finally:
        if dcgm is not None:
            dcgm.stop()
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
