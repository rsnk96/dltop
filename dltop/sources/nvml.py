"""NVML-backed GPU sampling and process enumeration."""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING

from dltop.models import PCIE_GEN_MBS_PER_LANE, GpuState
from dltop.sources.system import psutil

if TYPE_CHECKING:
    from psutil import Process as PsutilProcess

try:
    import pynvml
except ImportError:
    sys.exit("nvidia-ml-py is required. Install with: pip install nvidia-ml-py")


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
# Process enumeration
# --------------------------------------------------------------------------------------


def _collect_processes(gpus: list[GpuState], proc_cache: dict[int, PsutilProcess]) -> list[dict]:
    """Enumerate GPU processes via NVML. Annotate with psutil when available.

    ``proc_cache`` must be the same dict across calls: ``psutil.Process.cpu_percent()``
    reports a meaningful value only when called at least twice on the *same* Process
    instance (it diffs against its own last call), so a fresh ``Process(pid)`` every
    tick would always report 0.0. Stale pids are pruned once they drop out of view.
    """
    out: list[dict] = []
    live_pids: set[int] = set()
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
            live_pids.add(pid)
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
                    proc = proc_cache.get(pid)
                    if proc is None:
                        proc = psutil.Process(pid)
                        proc_cache[pid] = proc
                    entry["user"] = proc.username()
                    cmdline = proc.cmdline()
                    entry["cmd"] = " ".join(cmdline) if cmdline else proc.name()
                    entry["cpu_pct"] = proc.cpu_percent(interval=None)
                    entry["rss_mb"] = proc.memory_info().rss / 1e6
            out.append(entry)
    for pid in proc_cache.keys() - live_pids:
        del proc_cache[pid]
    out.sort(key=lambda r: (-r["mem_mb"], r["gpu"], r["pid"]))
    return out
