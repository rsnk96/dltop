"""CLI entry point."""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from dltop._version import __version__
from dltop.app import DltopApp


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="dltop -- a top/htop-style GPU monitor for CV and AI workloads.")
    ap.add_argument("-i", "--interval", type=float, default=0.5, help="sampling interval, seconds")
    ap.add_argument("--no-dcgm", action="store_true", help="skip DCGM probe entirely")
    ap.add_argument("--window", type=float, default=60.0, help="stats window for the Table tab, seconds")
    ap.add_argument("--no-discover", action="store_true", help="disable Prometheus /metrics auto-discovery")
    ap.add_argument(
        "--demo",
        type=int,
        nargs="?",
        const=2,
        default=None,
        metavar="N",
        help="synthetic N-GPU telemetry (default 2); no NVIDIA hardware needed",
    )
    ap.add_argument("--version", action="version", version=f"dltop {__version__}")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    app = DltopApp(
        interval=args.interval,
        no_dcgm=args.no_dcgm,
        demo_gpus=args.demo,
        window_s=args.window,
        no_discover=args.no_discover,
    )
    app.run()
