"""DCGM profiling metrics streamed by parsing `dcgmi dmon` stdout."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading

from loguru import logger

# DCGM profiling field IDs (see dcgm_fields.h). All return a fraction 0..1.
# Ordered deliberately; the `dcgmi dmon` output columns match this order.
DCGM_FIELD_ORDER: list[int] = [1002, 1004, 1007, 1008, 1006]
DCGM_FIELD_NAMES: dict[int, str] = {
    1002: "sm_active",
    1004: "tensor_active",
    1007: "fp32_active",
    1008: "fp16_active",
    1006: "fp64_active",
}

DCGM_INSTALL_HINT = (
    "Install NVIDIA Data Center GPU Manager and enable the service to get the\n"
    "Tensor vs FP32/FP16/FP64 split (works on Tesla/Quadro/A/H/L-series only):\n"
    "  Ubuntu:  sudo apt install datacenter-gpu-manager\n"
    "           sudo systemctl --now enable nvidia-dcgm\n"
    "  Docs:    https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html\n"
    "Note: NVIDIA gates profiling fields on GeForce cards; no software toggle unlocks them."
)


# --------------------------------------------------------------------------------------
# DCGM probe (via `dcgmi dmon` subprocess)
# --------------------------------------------------------------------------------------


class DcgmProbe:
    """Streams DCGM profiling metrics by parsing `dcgmi dmon` stdout."""

    def __init__(self, gpu_count: int, interval_ms: int = 500) -> None:
        """Create a streamer for ``gpu_count`` GPUs sampled every ``interval_ms`` ms."""
        self.gpu_count = gpu_count
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen[str] | None = None
        self.latest: dict[int, dict[str, float]] = {i: {} for i in range(gpu_count)}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def cli_available() -> tuple[bool, str]:
        """Cheap precheck: dcgmi binary present and hostengine reachable."""
        if shutil.which("dcgmi") is None:
            return False, "dcgmi binary not found on PATH"
        try:
            r = subprocess.run(
                ["dcgmi", "discovery", "-l"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"dcgmi call failed: {exc}"
        if r.returncode != 0:
            msg = r.stderr.strip() or r.stdout.strip()
            return False, f"dcgmi exited {r.returncode}: {msg}"
        return True, ""

    @staticmethod
    def profiling_supported() -> tuple[bool, str]:
        """Try a brief watch on the profiling fields we actually want.

        Returns (True, "") if the profiling module loads and streams at least
        one data row, else (False, <reason>). Catches NVIDIA's GeForce restriction
        (DCGM error -33, "module not currently loaded").
        """
        fields = ",".join(str(f) for f in DCGM_FIELD_ORDER)
        try:
            r = subprocess.run(  # noqa: S603
                ["dcgmi", "dmon", "-e", fields, "-d", "500", "-c", "1"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"dcgmi dmon timed out or failed: {exc}"
        combined = r.stdout + r.stderr
        if "Error" in combined or "-33" in combined:
            first_err = next((ln.strip() for ln in combined.splitlines() if "Error" in ln), combined.strip())
            return False, first_err
        if not any(line.lstrip().startswith("GPU ") for line in r.stdout.splitlines()):
            return False, "dcgmi dmon returned no GPU data rows"
        return True, ""

    def start(self) -> None:
        """Spawn the `dcgmi dmon` subprocess and the background reader thread."""
        fields = ",".join(str(f) for f in DCGM_FIELD_ORDER)
        cmd = ["dcgmi", "dmon", "-e", fields, "-d", str(self.interval_ms)]
        logger.info("Starting DCGM stream: {}", " ".join(cmd))
        self.proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line or not line.startswith("GPU "):
                continue
            parts = line.split()
            if len(parts) < 2 + len(DCGM_FIELD_ORDER):
                continue
            try:
                gpu_idx = int(parts[1])
            except ValueError:
                continue
            snap: dict[str, float] = {}
            for fid, token in zip(DCGM_FIELD_ORDER, parts[2 : 2 + len(DCGM_FIELD_ORDER)], strict=True):
                try:
                    snap[DCGM_FIELD_NAMES[fid]] = float(token) * 100.0
                except ValueError:
                    snap[DCGM_FIELD_NAMES[fid]] = float("nan")
            with self._lock:
                self.latest[gpu_idx] = snap

    def snapshot(self, gpu_idx: int) -> dict[str, float]:
        """Return the most recent profiling snapshot for `gpu_idx`, or empty dict."""
        with self._lock:
            return dict(self.latest.get(gpu_idx, {}))

    def stop(self) -> None:
        """Signal the reader thread to exit and terminate the `dcgmi` subprocess."""
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=2)
