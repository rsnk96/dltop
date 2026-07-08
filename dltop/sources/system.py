"""Host CPU / RAM / disk / network sampling via psutil (gracefully optional)."""

from __future__ import annotations

import contextlib
import time

from dltop.models import SystemState

try:
    import psutil
except ImportError:
    psutil = None  # process table degrades gracefully without it


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
