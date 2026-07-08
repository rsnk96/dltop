#!/usr/bin/env python3
"""Capture marketing screenshots (PNG) of every dltop tab.

Runs dltop headless (Textual's ``run_test`` pilot), drives it with a gentle
*cyclical* CPU + GPU demo workload so the graphs show lively waves rather than
flat idle lines, lets it accumulate data for a warm-up window so the chart is
well populated, exports one SVG per tab, then rasterises each to a PNG (the
format the README embeds, since GitHub renders PNGs reliably).

Why synthetic telemetry rather than real load: dltop is typically run on a busy
shared GPU, and spinning a real CUDA job to "make the graphs move" would steal
compute/VRAM from whatever is actually training. Synthetic samples give
reproducible, good-looking images without touching the real device. The process
table, however, is read live (so the shots are authentic) and is sanitised:
any ``molmo`` command has the value after ``--dataset`` stripped, because the
dataset path can be sensitive.

Usage::

    python scripts/capture_screenshots.py                  # -> assets/screenshots/*.png
    python scripts/capture_screenshots.py --warmup 25 --width 168 --height 46

Requires the ``screenshots`` extra (``pip install -e ".[screenshots]"``) for
``cairosvg``, which rasterises Textual's SVG export into PNG.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

# the dltop package lives at the repo root, one level up from this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dltop.app as app_module
import dltop.sources.nvml as nvml_source
from dltop.app import DltopApp
from dltop.widgets.plot import TimeSeriesPlot

if TYPE_CHECKING:
    from dltop.models import GpuState

TABS = ("tab-all", "tab-compute", "tab-memory", "tab-system")


# A "mixed-precision training step" cycle: GPU compute pulses, host CPU feeds it
# between pulses, memory/power track the compute. Period ~9 s so a 20 s warm-up
# shows roughly two clean cycles. Values are percentages; _jitter adds a little
# deterministic high-frequency wobble so the lines read as live, not hand-drawn.
def _jitter(t: float) -> float:
    return 1.6 * math.sin(t * 7.3) + 0.9 * math.sin(t * 17.1 + 1.0)


def _demo_sample(t: float) -> dict[str, float]:
    phase = t * (2 * math.pi / 9.0)
    sm = 78 + 14 * math.sin(phase) + _jitter(t)
    return {
        "cpu": 38 + 13 * math.sin(phase + math.pi) + _jitter(t * 1.3),
        "sm": sm,
        "tensor": 0.86 * sm + _jitter(t * 0.7),
        "fp32": 24 + 7 * math.sin(phase * 1.3),
        "fp16": 0.80 * sm + _jitter(t * 1.1),
        "fp64": 0.0,
        "nvenc": 0.0,
        "nvdec": 0.0,
        "ram": 41 + 5 * math.sin(t / 13.0),
        "vram": 80 + 6 * math.sin(t / 15.0) + 0.4 * _jitter(t),
        "mbw": 0.72 * sm + _jitter(t),
        "pcie_tx": 20 + 10 * math.sin(phase),
        "pcie_rx": 34 + 14 * math.sin(phase + 0.5),
        "power": 74 + 12 * math.sin(phase),
        "disk_r": 46 + 22 * math.sin(t / 6.0),
        "disk_w": 14 + 7 * math.sin(t / 4.0),
        "net_rx": 56 + 20 * math.sin(t / 7.0 + 1.0),
        "net_tx": 20 + 9 * math.sin(t / 5.0),
    }


def _stage_cards(app: DltopApp) -> None:
    """Pin the info-cards to a believable training snapshot, consistent with the graphs."""
    if app.gpus:
        g = app.gpus[0]
        g.mem_used_mb, g.mem_pct = 68000, 80
        g.power_w, g.power_limit_w = 582, 700
        g.temp_c, g.sm_clock_mhz, g.mem_clock_mhz = 70, 1980, 2619
        g.nvenc_pct = g.nvdec_pct = 0
        g.mbw_pct, g.sm_overall_pct = 70, 80
        g.pcie_curr_gen, g.pcie_curr_width, g.pcie_max_mbs = 5, 16, 63015
        g.pcie_tx_mbs, g.pcie_rx_mbs = 1180, 2360
    s = app.sys_state
    s.cpu_pct, s.cpu_count_physical, s.cpu_count_logical = 38, 64, 128
    s.load_1, s.load_5, s.load_15 = 12.4, 10.1, 8.7
    s.ram_used_gib, s.ram_total_gib, s.ram_pct = 181.0, 503.5, 36
    s.disk_read_mbs, s.disk_write_mbs, s.net_rx_mbs, s.net_tx_mbs = 58.0, 14.0, 72.0, 22.0


def _install_process_sanitiser() -> None:
    """Strip the value after ``--dataset`` from any ``molmo`` process command.

    The dataset path can be sensitive, so it must never appear in a published
    screenshot. Wraps dltop's process collector so the scrub also reaches the
    rendered table cells, not just the SVG text.
    """
    original = nvml_source._collect_processes
    pattern = re.compile(r"(--dataset)(=|\s+)\S+")

    def collect(gpus: list[GpuState], proc_cache):  # noqa: ANN001, ANN202
        procs = original(gpus, proc_cache)
        for proc in procs:
            if "molmo" in proc.get("cmd", ""):
                proc["cmd"] = pattern.sub(r"\1", proc["cmd"])
        return procs

    nvml_source._collect_processes = collect
    app_module._collect_processes = collect


async def _capture(warmup: float, size: tuple[int, int], interval: float) -> dict[str, str]:
    """Drive the app for ``warmup`` seconds of demo data, return ``{tab_name: svg}``."""
    plot_ids = ("all-plot", "compute-plot", "memory-plot", "system-plot")
    app = DltopApp(interval=interval, no_dcgm=False)
    svgs: dict[str, str] = {}
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app.paused = True  # stop real sampling; we drive synthetic demo telemetry
        _stage_cards(app)
        app._refresh_cards()
        app._refresh_procs()  # live (sanitised) process table

        # Warm-up: feed cyclical samples in real time so the graph fills with
        # ~`warmup` seconds of history before we record.
        start = time.monotonic()
        plots = [app.query_one(f"#{pid}", TimeSeriesPlot) for pid in plot_ids]
        while time.monotonic() - start < warmup:
            sample = _demo_sample(time.monotonic() - start)
            app.store.record_many(sample)
            for plot in plots:
                plot.replot()
            await asyncio.sleep(interval)
        print(f"warmed up for {warmup:.0f}s ({len(app.store.tail('cpu', 10_000))} samples)")

        for tab in TABS:
            app.query_one("#tabs").active = tab
            await pilot.pause()
            for plot in plots:
                plot.replot()
            await pilot.pause()
            svgs[tab.split("-", 1)[1]] = app.export_screenshot()
    return svgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("assets/screenshots"), help="where to write the SVGs")
    parser.add_argument("--warmup", type=float, default=20.0, help="seconds of data to accumulate before capture")
    parser.add_argument("--width", type=int, default=168, help="terminal columns")
    parser.add_argument("--height", type=int, default=46, help="terminal rows")
    parser.add_argument("--interval", type=float, default=0.4, help="seconds between demo samples")
    args = parser.parse_args()

    import cairosvg  # local import: only the screenshots extra needs it

    _install_process_sanitiser()
    svgs = asyncio.run(_capture(args.warmup, (args.width, args.height), args.interval))
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, svg in svgs.items():
        path = args.outdir / f"{name}.png"
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(path))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
