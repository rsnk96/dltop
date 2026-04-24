# cvtop

A `top`/`htop`-style GPU monitor, tailored for computer-vision and deep-learning workloads.

Where `nvitop` gives you one uniform "SM%" bar, `cvtop` splits live GPU utilization into the two things a CV/AI engineer actually cares about on the same silicon:

- **Compute lanes** — CUDA/Tensor/FP32/FP16/FP64 activity, via NVIDIA DCGM profiling fields
- **Media engines** — NVENC (encode) and NVDEC (decode) throughput, via NVML

When DCGM is unavailable (consumer GeForce cards, or DCGM not installed), `cvtop` falls back to NVML's single lumped SM% and prints a footer telling you how to enable the full split on a data-center GPU.

## Installation

### From GitHub (current)

```bash
pip install git+https://github.com/<your-user>/cvtop
```

### From PyPI

Not yet published.

### For development

```bash
git clone https://github.com/<your-user>/cvtop && cd cvtop
pip install -e ".[test]"
```

Python 3.11+ required. The tool is Linux-only (needs NVML + optionally `dcgmi`).

## Usage

Once installed, `cvtop` is callable from any directory — the same way `htop` is:

```bash
cvtop                   # 0.5s refresh, DCGM if available
cvtop -i 1              # 1s refresh
cvtop --no-dcgm         # NVML-only (skip DCGM even if installed)
cvtop --help
```

### Key bindings (in the TUI)

| Key | Action |
|---|---|
| `q`, `Q` | Quit |
| `space`, `p` | Pause / unpause sampling |
| `v`, `Tab` | Cycle view: all → gpu-only → system-only |
| `g` / `s` / `a` | Jump directly to GPU / System / All view |

## Enabling the full compute split (DCGM)

On a data-center GPU (A100, H100, L4, T4, etc.) you can install DCGM to unlock per-lane breakdown:

```bash
# Ubuntu / Debian
sudo apt install datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm
```

Consumer GeForce cards (RTX 4090, 3090, etc.) have these profiling fields gated by NVIDIA at the driver level — DCGM will install but return no profiling data. `cvtop` detects this at startup and falls back silently.

## Runtime requirements

- NVIDIA driver + CUDA runtime (any version supported by `nvidia-ml-py`)
- DCGM service (optional, for full compute-lane split)
- A terminal (TTY) if you want the keyboard shortcuts — piping to a non-TTY still renders frames but disables keystroke handling

## License

MIT
