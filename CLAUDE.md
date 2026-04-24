
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`dl-gpumon` is a TUI that splits live NVIDIA GPU utilization into DL compute (CUDA/Tensor/FP*) versus media engines (NVENC/NVDEC). Target: public PyPI package — preserve backwards-compatible CLI flags and public behaviour across edits.

## Layout

Single-module project: all code lives in `dl_gpumon.py` at the repo root. This is intentional — do not split into a package (`dl_gpumon/__init__.py`, submodules, etc.) unless explicitly asked.

Entry point: `dl-gpumon` → `dl_gpumon:main` (defined in `pyproject.toml`).

## Architecture

Two-source sampling model, pick one at runtime:

- **DCGM path** — `dcgmi dmon` subprocess streams per-GPU FP16/FP32/FP64/Tensor/SM-Active fields. Requires the `nvidia-dcgm` systemd service and is data-center-GPU only.
- **NVML path** — `pynvml` fallback. Used when DCGM is unavailable (consumer GeForce) or when invoked with `--no-dcgm`. In this mode, SM% is lumped (can't separate Tensor/FP lanes).

Both paths feed the same `Rich`-rendered TUI. When adding metrics, add them to both paths or clearly document the DCGM-only gap in the UI.

## Dependencies

- `nvidia-ml-py` and `rich` are hard requirements.
- `psutil` is **gracefully optional** — the system-stats panel degrades to a "psutil not installed" message rather than raising. Preserve this pattern for any future optional deps (import inside a try/except at module load, set a sentinel).

## Commands

- Install (editable): `pip install -e .`
- Run: `dl-gpumon` (flags: `-i/--interval <sec>`, `--no-dcgm`)
- Lint: `ruff check .`
- Format: `black .` and `ruff check --fix .`

There is **no pytest suite and no CI**. Do not run `pytest` — it will do nothing useful. After non-trivial edits, smoke-test by running `dl-gpumon --help` and (if a GPU is present) `dl-gpumon -i 1` for a few seconds and confirming no tracebacks.

## Ruff config

`select = ["ALL"]` with these ignores (in `pyproject.toml`): `D203`, `D213`, `COM812`, `PLR2004`. If a new rule fires noisily across the file for a good reason, discuss before adding to the ignore list — the bar is "conflicts with formatter" or "genuinely not applicable", not "annoying right now".
