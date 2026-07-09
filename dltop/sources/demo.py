"""Synthetic N-GPU telemetry: powers --demo, screenshots, and GPU-less tests.

The waveform sketches a mixed-precision training loop: GPU compute pulses,
host CPU feeds it between pulses, memory and power track the compute. Each
GPU gets a phase offset so multi-GPU charts visibly differ.
"""

from __future__ import annotations

import math

from dltop.models import GpuState, SystemState

_PERIOD_S = 9.0


def _jitter(t: float) -> float:
    return 1.6 * math.sin(t * 7.3) + 0.9 * math.sin(t * 17.1 + 1.0)


def _clip(v: float) -> float:
    return max(0.0, min(100.0, v))


class DemoSource:
    """Fabricates GpuState/SystemState samples for ``n_gpus`` imaginary GPUs."""

    def __init__(self, n_gpus: int = 2) -> None:
        """Create a demo source for ``n_gpus`` synthetic devices."""
        self.n_gpus = max(1, n_gpus)

    def init_gpus(self) -> list[GpuState]:
        """Return one plausibly-specced synthetic GPU per index."""
        gpus = []
        for i in range(self.n_gpus):
            g = GpuState(index=i, name=f"Demo GPU {i} (RTX 4090)", mem_total_mb=24_000.0)
            g.power_limit_w = 450.0
            g.pcie_max_mbs = 63_015.0
            g.pcie_curr_gen, g.pcie_curr_width = 5, 16
            g.sm_clock_max_mhz, g.mem_clock_max_mhz = 2520.0, 10_501.0
            gpus.append(g)
        return gpus

    def sample(self, gpus: list[GpuState], sys_state: SystemState, t: float) -> None:
        """Fill ``gpus`` and ``sys_state`` with the demo waveform at time ``t`` seconds."""
        for g in gpus:
            phase = (t + g.index * 2.7) * (2 * math.pi / _PERIOD_S)
            sm = _clip(78 + 14 * math.sin(phase) + _jitter(t + g.index))
            g.sm_overall_pct = sm
            g.dcgm = {
                "sm_active": sm,
                "tensor_active": _clip(0.86 * sm + _jitter(t * 0.7 + g.index)),
                "fp32_active": _clip(24 + 7 * math.sin(phase * 1.3)),
                "fp16_active": _clip(0.80 * sm + _jitter(t * 1.1 + g.index)),
                "fp64_active": 0.0,
            }
            g.nvenc_pct = _clip(12 + 10 * math.sin(phase * 0.5 + g.index))
            g.nvdec_pct = _clip(18 + 12 * math.sin(phase * 0.6 + 1.0 + g.index))
            g.mem_pct = _clip(72 + 6 * math.sin(t / 15.0 + g.index))
            g.mem_used_mb = g.mem_total_mb * g.mem_pct / 100.0
            g.mbw_pct = _clip(0.72 * sm + _jitter(t + g.index))
            g.power_w = g.power_limit_w * _clip(74 + 12 * math.sin(phase)) / 100.0
            g.temp_c = 55 + 12 * math.sin(phase) / 2
            g.fan_pct = _clip(45 + 20 * math.sin(phase))
            g.sm_clock_mhz, g.mem_clock_mhz = 1980.0, 10_251.0
            g.pcie_tx_mbs = _clip(20 + 10 * math.sin(phase)) / 100.0 * g.pcie_max_mbs * 0.3
            g.pcie_rx_mbs = _clip(34 + 14 * math.sin(phase + 0.5)) / 100.0 * g.pcie_max_mbs * 0.3
        sys_state.cpu_pct = _clip(38 + 13 * math.sin(t * 2 * math.pi / _PERIOD_S + math.pi) + _jitter(t * 1.3))
        sys_state.cpu_count_physical, sys_state.cpu_count_logical = 32, 64
        sys_state.load_1, sys_state.load_5, sys_state.load_15 = 12.4, 10.1, 8.7
        sys_state.ram_total_gib = 251.6
        sys_state.ram_pct = _clip(41 + 5 * math.sin(t / 13.0))
        sys_state.ram_used_gib = sys_state.ram_total_gib * sys_state.ram_pct / 100.0
        sys_state.disk_read_mbs = _clip(46 + 22 * math.sin(t / 6.0))
        sys_state.disk_write_mbs = _clip(14 + 7 * math.sin(t / 4.0))
        sys_state.net_rx_mbs = _clip(56 + 20 * math.sin(t / 7.0 + 1.0))
        sys_state.net_tx_mbs = _clip(20 + 9 * math.sin(t / 5.0))

    def processes(self) -> list[dict]:
        """Synthetic rows matching the schema of ``_collect_processes``."""
        return [
            {
                "gpu": i,
                "pid": 41000 + i,
                "type": "C",
                "mem_mb": 18_400.0,
                "user": "mluser",
                "cmd": f"python train.py --config configs/run{i}.toml",
                "cpu_pct": 210.0,
                "rss_mb": 9_800.0,
            }
            for i in range(self.n_gpus)
        ]
