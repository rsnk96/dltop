"""Data model, palette and series definitions shared by sources and widgets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

HISTORY_LEN = 600  # ring-buffer capacity per series; one slot per display column, no resampling

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
# Design rule (non-negotiable): host metrics (CPU/RAM/Disk/Net) are NEVER mixed
# into a GPU's chart. Each chart shows exactly one domain: host, or a single GPU.
# Within each domain, four tabs group what an operator reasons about together:
#   * Compute — compute engines + media encode/decode (GPU) / CPU (host)
#   * Memory  — RAM (host) / VRAM + VRAM bandwidth (GPU)
#   * System  — disk + network (host) / PCIe + power (GPU)
SeriesDef = tuple[str, int, str, bool]

_HOST_ALL: list[SeriesDef] = [
    ("cpu", 208, "CPU", True),  # orange
    ("ram", 46, "RAM", True),  # green
    ("disk_r", 226, "Disk R (MB/s)", True),  # yellow
    ("disk_w", 220, "Disk W (MB/s)", True),  # gold
    ("net_rx", 201, "Net RX (MB/s)", True),  # magenta
    ("net_tx", 93, "Net TX (MB/s)", True),  # purple
]
# Compute tab. The DCGM variant adds the Tensor/FP32/FP16/FP64 split that only
# data-center GPUs expose; the NVML variant lumps it all into a single SM%.
_GPU_COMPUTE_DCGM: list[SeriesDef] = [
    ("sm", 15, "GPU SM", True),  # white  — overall
    ("tensor", 201, "Tensor", True),  # magenta / hot pink
    ("fp32", 39, "FP32", True),  # deep sky blue
    ("fp16", 46, "FP16", True),  # green
    ("fp64", 226, "FP64", True),  # yellow
    ("nvenc", 220, "NVENC", True),  # gold
    ("nvdec", 93, "NVDEC", True),  # purple
]
_GPU_COMPUTE_NVML: list[SeriesDef] = [
    ("sm", 15, "GPU SM", True),  # white
    ("nvenc", 220, "NVENC", True),  # gold
    ("nvdec", 93, "NVDEC", True),  # purple
]
_GPU_MEMORY: list[SeriesDef] = [
    ("vram", 51, "VRAM", True),  # cyan
    ("mbw", 46, "VRAM BW", True),  # green
]
_GPU_SYSTEM: list[SeriesDef] = [
    ("pcie_tx", 51, "PCIe ↑", True),  # cyan
    ("pcie_rx", 15, "PCIe ↓", True),  # white
    ("power", 196, "Power %", True),  # red
]

HOST_SERIES: dict[str, list[SeriesDef]] = {
    "all": _HOST_ALL,
    "compute": [("cpu", 208, "CPU", True)],
    "memory": [("ram", 208, "RAM", True)],
    "system": _HOST_ALL[2:],  # disk + network
}
GPU_SERIES_DCGM: dict[str, list[SeriesDef]] = {
    "all": [*_GPU_COMPUTE_DCGM, *_GPU_MEMORY, *_GPU_SYSTEM],
    "compute": _GPU_COMPUTE_DCGM,
    "memory": _GPU_MEMORY,
    "system": _GPU_SYSTEM,
}
GPU_SERIES_NVML: dict[str, list[SeriesDef]] = {
    "all": [*_GPU_COMPUTE_NVML, *_GPU_MEMORY, *_GPU_SYSTEM],
    "compute": _GPU_COMPUTE_NVML,
    "memory": _GPU_MEMORY,
    "system": _GPU_SYSTEM,
}

# Table-tab metadata (Task 6): units and labels by base series name.
SERIES_UNITS: dict[str, str] = {
    "cpu": "%",
    "ram": "%",
    "disk_r": "MB/s",
    "disk_w": "MB/s",
    "net_rx": "MB/s",
    "net_tx": "MB/s",
    "sm": "%",
    "tensor": "%",
    "fp32": "%",
    "fp16": "%",
    "fp64": "%",
    "nvenc": "%",
    "nvdec": "%",
    "vram": "%",
    "mbw": "%",
    "pcie_tx": "%",
    "pcie_rx": "%",
    "power": "%",
}
SERIES_LABELS: dict[str, str] = {
    name: label for defs in (_HOST_ALL, _GPU_COMPUTE_DCGM, _GPU_MEMORY, _GPU_SYSTEM) for name, _, label, _ in defs
}


def per_gpu(defs: list[SeriesDef], gpu_index: int) -> list[SeriesDef]:
    """Suffix every series name with ``@{gpu_index}`` for per-GPU identity."""
    return [(f"{name}@{gpu_index}", colour, label, default) for name, colour, label, default in defs]


def split_series_name(name: str) -> tuple[str, int | None]:
    """Split ``sm@1`` into ``("sm", 1)``; plain host names return ``(name, None)``."""
    base, sep, idx = name.partition("@")
    return (base, int(idx)) if sep else (base, None)


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
