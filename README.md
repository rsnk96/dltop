# dltop

A `htop`/`nvitop`-style GPU monitor, tailored for computer-vision and deep-learning workloads.

Where `nvitop` gives you one uniform compute-utilization bar, `dltop` splits live GPU utilization into the two things an AI/CV engineer actually cares about on the same GPU:

- **Compute utilization** — CUDA Streaming Multiprocessor / Tensor / FP32 / FP16 / FP64 activity (via DCGM)
- **Media engines** — NVENC (encode) and NVDEC (decode) throughput (via NVML)

This helps identify the choke point for different scales of load on a streaming-analytics pipeline. Many metrics share **one interleaved chart** instead of stacking into tall bars: overlapping series alternate colour cell-by-cell, so no line is ever hidden behind another while each keeps its true value.

When DCGM is unavailable (consumer GeForce cards, or DCGM not installed), `dltop` falls back to NVML's single lumped SM% (the aggregate compute-utilization proxy shown by `nvidia-smi`) and prints a footer telling you how to enable the full split on a data-center GPU.

### Multi-GPU

Every chart is grouped **one domain per GPU** — never mixed onto a shared axis. On a box with more than one GPU, `dltop` adds a host chart plus one chart per GPU per tab (so a 2-GPU box shows 3 stacked charts on the Compute tab: Host, GPU 0, GPU 1), an info card row per GPU, and a toggle bar to show/hide individual GPUs across every chart at once.

## Screenshots

Captured in `--demo` mode (synthetic 2-GPU telemetry — see [Demo mode](#demo-mode-no-gpu-needed)). Five tabs, all on one shared time-series axis.

### All
The default overview — every metric on one axis, with **CPU, RAM, GPU SM and GPU VRAM** on by default. Glance once: is anything busy?

<img src="https://github.com/rsnk96/dltop/releases/download/v0.2.1/all.png" alt="dltop — All tab" width="100%">

### Compute
Host CPU plus the GPU compute engines. On data-center GPUs the DCGM split breaks SM into **Tensor, FP32, FP16 and FP64** — see exactly which math your model is doing.

<img src="https://github.com/rsnk96/dltop/releases/download/v0.2.1/compute.png" alt="dltop — Compute tab" width="100%">

### Memory
Host **RAM**, GPU **VRAM** and **VRAM bandwidth** — watch memory fill as a model loads and catch bandwidth saturation.

<img src="https://github.com/rsnk96/dltop/releases/download/v0.2.1/memory.png" alt="dltop — Memory tab" width="100%">

### System
**PCIe** in/out, GPU **power** draw, **disk** and **network** — the plumbing around the GPU.

<img src="https://github.com/rsnk96/dltop/releases/download/v0.2.1/system.png" alt="dltop — System tab" width="100%">

### Table
Every series — host, per-GPU, and any auto-discovered Prometheus metric — as a row of rolling-window stats (**Now, Mean, Median, Stddev**) over the last `--window` seconds. Four buttons copy the current snapshot straight to the clipboard: **Markdown**, **web table (HTML)**, **Excel (TSV)**, or a **capture-metadata** block (timestamp, dltop version, hostname, OS, CPU, RAM, GPU(s), driver/CUDA versions, and any Prometheus endpoints) — handy for pasting into a bug report or a training-run log.

<img src="https://github.com/rsnk96/dltop/releases/download/v0.2.1/table.png" alt="dltop — Table tab" width="100%">

> The screenshots above are hosted as assets on the [latest release](https://github.com/rsnk96/dltop/releases/latest) rather than committed to the repo, so a clone stays small and the images render identically on GitHub and PyPI. Regenerate them anytime with `pip install -e ".[screenshots]"` then `python scripts/capture_screenshots.py` (writes to `assets/screenshots/`, which is git-ignored), and re-upload with `gh release upload <tag> assets/screenshots/*.png`.

## Installation

### From PyPI (recommended)

```bash
pipx install dltop
```

[`pipx`](https://pipx.pypa.io) installs `dltop` into its own isolated environment while still putting the `dltop` command on your `PATH` — the right way to install a standalone CLI tool. A plain `pip install` works too:

```bash
pip install dltop
```

### For development

```bash
git clone https://github.com/rsnk96/dltop && cd dltop
pip install -e ".[test]"
```

Python 3.11+ required. The tool is Linux-only (needs NVML + optionally `dcgmi`).

## Usage

Once installed, `dltop` is callable from any directory — the same way `htop` is:

```bash
dltop                   # 0.5s refresh, DCGM if available
dltop -i 1              # 1s refresh
dltop --no-dcgm         # NVML-only (skip DCGM even if installed)
dltop --window 120      # stats window for the Table tab, seconds (default 60)
dltop --no-discover     # disable Prometheus /metrics auto-discovery
dltop --demo            # synthetic 2-GPU telemetry, no NVIDIA hardware needed
dltop --demo 4          # synthetic N-GPU telemetry
dltop --version         # print the installed version and exit
dltop --help
```

### Key bindings (in the TUI)

| Key | Action |
|---|---|
| `q`, `Q` | Quit |
| `space`, `p` | Pause / resume sampling |

Switch tabs (**All · Compute · Memory · System · Table**) by clicking them, or with `←` / `→` when the tab bar is focused. Each chart has per-series checkboxes underneath to toggle individual lines on or off.

### Demo mode (no GPU needed)

`--demo [N]` swaps NVML/DCGM for a synthetic `N`-GPU source (default 2) so you can try, develop, or screenshot `dltop` on a laptop with no NVIDIA hardware at all — it's also what `scripts/capture_screenshots.py` uses to generate the images above.

### Prometheus auto-discovery

On startup (unless `--no-discover` is passed), `dltop` enumerates locally listening TCP ports (via `psutil`, falling back to `ss -lnt`), probes each with a short-timeout `GET /metrics`, and treats any response that looks like Prometheus text-exposition format as a hit. Up to 30 gauge/counter metrics per discovered endpoint are added as extra, default-off series on the **All** tab, scraped on a background thread every `max(--interval, 1)` seconds. Histograms and summaries are skipped (their component series aren't meaningful as single lines). Disable the whole probe with `--no-discover` if you'd rather not have `dltop` touch other local sockets.

## Related projects

`dltop` owes a lot to the terminal monitors that came before it:

- [htop](https://htop.dev) — the classic interactive process and CPU monitor.
- [btop](https://github.com/aristocratos/btop) — a polished all-round system monitor (CPU, memory, disk, network).
- [nvtop](https://github.com/Syllo/nvtop) — an htop-style monitor for AMD / NVIDIA / Intel GPUs.
- [nvitop](https://github.com/XuehaiPan/nvitop) — an interactive NVIDIA GPU process viewer.

We use and admire all of these. None of them, individually or together, met the needs we had internally: a single view that splits GPU compute into the lanes a CV/AI engineer reasons about — SM, Tensor, FP16/FP32/FP64 — **and** the media engines (NVENC/NVDEC), overlaid against host CPU/RAM/IO on one time-series, so you can see where a streaming-analytics pipeline actually bottlenecks. So we built `dltop`.

## Enabling the full compute split (DCGM)

On a data-center GPU (A100, H100, L4, T4, etc.) you can install DCGM to unlock the per-lane breakdown:

```bash
# Ubuntu / Debian
sudo apt install datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm
```

Consumer GeForce cards (RTX 4090, 3090, etc.) have these profiling fields gated by NVIDIA at the driver level — DCGM will install but return no profiling data. `dltop` detects this at startup and falls back silently.

## Runtime requirements

- NVIDIA driver + CUDA runtime (any version supported by `nvidia-ml-py`)
- DCGM service (optional, for the full compute-lane split)
- A terminal (TTY) if you want the keyboard shortcuts — piping to a non-TTY still renders frames but disables keystroke handling

## License

MIT
