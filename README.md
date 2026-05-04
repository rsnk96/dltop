# dltop

A `htop`/`nvitop`-style GPU monitor, tailored for computer-vision and deep-learning workloads.

Where `nvitop` gives you one uniform compute utilization bar, `dltop` splits live GPU utilization into the two things a AI/CV engineer actually cares about on the same GPU:

- **Compute utilization** — CUDA Streaming Multiplexer/Tensor/FP32/FP16/FP64 activity
- **Media engines** — NVENC (encode) and NVDEC (decode) throughput, via NVML

This helps identify what the choke point might be for different scales of loads on a streaming analytics pipeline. These are some sample images of the different views -
* The Compute Tab: <img width="1902" height="1079" alt="image" src="https://github.com/user-attachments/assets/959367ce-24f7-4fc6-a434-153d62e682ad" />
* The Media and Power Tab: <img width="1902" height="1080" alt="image" src="https://github.com/user-attachments/assets/c74bfb36-9206-4f88-928b-4d20e027976c" />
* The System Tab: <img width="1902" height="1079" alt="image" src="https://github.com/user-attachments/assets/bb32edb6-a5e1-4f36-9a44-467601cd86d0" />



When DCGM is unavailable (consumer GeForce cards, or DCGM not installed), `dltop` falls back to NVML's single lumped SM% (which is an aggregate compute utilization proxy, and the default number shown in nvidia-smi volatile memory utilization) and prints a footer telling you how to enable the full split on a data-center GPU.

## Installation

### From GitHub (current)

```bash
pip install git+https://github.com/<your-user>/dltop
```

### For development

```bash
git clone https://github.com/<your-user>/dltop && cd dltop
pip install -e ".[test]"
```

Python 3.11+ required. The tool is Linux-only (needs NVML + optionally `dcgmi`).

## Usage

Once installed, `dltop` is callable from any directory — the same way `htop` is:

```bash
dltop                   # 0.5s refresh, DCGM if available
dltop -i 1              # 1s refresh
dltop --no-dcgm         # NVML-only (skip DCGM even if installed)
dltop --help
```

### Key bindings (in the TUI)

| Key | Action |
|---|---|
| `q`, `Q` | Quit |
| `space`, `p` | Pause / unpause sampling |
| `v`, `Tab` | Cycle view: all → gpu-only → system-only |
| `g` / `s` / `a` | Jump directly to GPU / System / All view |
| left arrow / right arrow | Rotate between the different views |

## Enabling the full compute split (DCGM)

On a data-center GPU (A100, H100, L4, T4, etc.) you can install DCGM to unlock per-lane breakdown:

```bash
# Ubuntu / Debian
sudo apt install datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm
```

Consumer GeForce cards (RTX 4090, 3090, etc.) have these profiling fields gated by NVIDIA at the driver level — DCGM will install but return no profiling data. `dltop` detects this at startup and falls back silently.

## Runtime requirements

- NVIDIA driver + CUDA runtime (any version supported by `nvidia-ml-py`)
- DCGM service (optional, for full compute-lane split)
- A terminal (TTY) if you want the keyboard shortcuts — piping to a non-TTY still renders frames but disables keystroke handling

## License

MIT
