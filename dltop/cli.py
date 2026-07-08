"""CLI entry point."""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from dltop.app import DltopApp


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="dltop -- a top/htop-style GPU monitor for CV and AI workloads.")
    ap.add_argument("-i", "--interval", type=float, default=0.5, help="sampling interval, seconds")
    ap.add_argument("--no-dcgm", action="store_true", help="skip DCGM probe entirely")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    app = DltopApp(interval=args.interval, no_dcgm=args.no_dcgm)
    app.run()
