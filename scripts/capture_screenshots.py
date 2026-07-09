#!/usr/bin/env python3
"""Capture marketing screenshots (PNG) of every dltop tab.

Runs dltop headless (Textual's ``run_test`` pilot) in **demo mode**
(``--demo``'s ``DemoSource``), so the screenshots need no NVIDIA hardware and
never touch a real GPU or process table -- the demo GPUs and their synthetic
telemetry are entirely fabricated, so there is nothing sensitive to sanitise.

The app's own ``set_interval`` tick drives sampling during the warm-up window;
this script just waits for it to accumulate enough history, then switches
through each tab, exporting one SVG per tab and rasterising each to a PNG
(the format the README embeds, since GitHub renders PNGs reliably).

Usage::

    python scripts/capture_screenshots.py                  # -> assets/screenshots/*.png
    python scripts/capture_screenshots.py --warmup 25 --width 168 --height 46

Requires the ``screenshots`` extra (``pip install -e ".[screenshots]"``) for
``cairosvg``, which rasterises Textual's SVG export into PNG.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# the dltop package lives at the repo root, one level up from this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dltop.app import DltopApp

TABS = ("tab-all", "tab-compute", "tab-memory", "tab-system", "tab-table")


async def _capture(warmup: float, size: tuple[int, int], interval: float) -> dict[str, str]:
    """Run dltop in demo mode for ``warmup`` seconds, return ``{tab_name: svg}``."""
    app = DltopApp(interval=interval, no_dcgm=True, demo_gpus=2, no_discover=True)
    svgs: dict[str, str] = {}
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        start = time.monotonic()
        while time.monotonic() - start < warmup:  # noqa: ASYNC110 - polling wall-clock warm-up, not an event
            await asyncio.sleep(interval)
        # Drop focus so no widget shows its focus ring/underline in the still.
        app.set_focus(None)
        for tab in TABS:
            app.query_one("#tabs").active = tab
            await pilot.pause()
            svgs[tab.split("-", 1)[1]] = app.export_screenshot()
    return svgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("assets/screenshots"), help="where to write the PNGs")
    parser.add_argument("--warmup", type=float, default=20.0, help="seconds of data to accumulate before capture")
    parser.add_argument("--width", type=int, default=168, help="terminal columns")
    parser.add_argument("--height", type=int, default=46, help="terminal rows")
    parser.add_argument("--interval", type=float, default=0.4, help="seconds between demo samples")
    args = parser.parse_args()

    import cairosvg  # local import: only the screenshots extra needs it

    svgs = asyncio.run(_capture(args.warmup, (args.width, args.height), args.interval))
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, svg in svgs.items():
        path = args.outdir / f"{name}.png"
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(path))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
