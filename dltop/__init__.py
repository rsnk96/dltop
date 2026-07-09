"""dltop -- a top/htop-style GPU monitor tailored for CV and AI workloads.

Splits live GPU utilization into two categories:

    * Compute (CV/AI): SM active, Tensor pipe, FP32/FP16/FP64 pipes -- via DCGM profiling
    * Media engines: NVENC, NVDEC -- via NVML

The UI is built on Textual with box-drawing line charts (the nvtop/asciichart
look), so many metrics share one chart instead of stacking into tall bars.
Overlapping series are colour-interleaved per column so no line is ever hidden
behind another while every line keeps its true value (see ``TimeSeriesPlot``).

When DCGM profiling metrics are unavailable (e.g. consumer GeForce cards where
NVIDIA gates profiling fields, or DCGM not installed), we transparently fall
back to NVML's single overall SM% and show a status banner telling the user how
to enable the full split on a data-center GPU.
"""

from dltop._version import __version__
from dltop.cli import main

__all__ = ["__version__", "main"]
