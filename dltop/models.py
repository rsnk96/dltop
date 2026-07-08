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
