"""Smoke tests for cvtop.

Runnable on a GPU-less CI host: we never reach `init_nvml()` because `--help`
causes argparse to exit(0) before anything else runs. Tests that need real GPU
state belong in a separate integration suite, not here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_module_imports() -> None:
    """Importing the module must not touch NVML or spawn subprocesses."""
    import cvtop

    assert hasattr(cvtop, "main")
    assert callable(cvtop.main)


def test_cli_help_via_entrypoint() -> None:
    """The installed `cvtop` console script responds to --help cleanly."""
    result = subprocess.run(
        ["cvtop", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--interval" in result.stdout
    assert "--no-dcgm" in result.stdout


def test_cli_help_via_module() -> None:
    """`python -m cvtop --help` also works (invariant for editable installs)."""
    result = subprocess.run(
        [sys.executable, "-m", "cvtop", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--interval" in result.stdout


def test_readme_exists_and_is_non_trivial() -> None:
    """Guards pyproject.toml's `readme = "README.md"` claim against file loss."""
    readme = Path(__file__).resolve().parent.parent / "README.md"
    assert readme.is_file(), f"missing {readme}"
    assert len(readme.read_text()) > 200


def test_public_constants_are_plausible() -> None:
    """Sanity-check DCGM field constants -- catches accidental list/dict drift."""
    import cvtop

    assert len(cvtop.DCGM_FIELD_ORDER) == len(cvtop.DCGM_FIELD_NAMES)
    assert set(cvtop.DCGM_FIELD_ORDER) == set(cvtop.DCGM_FIELD_NAMES.keys())
    # Names referenced elsewhere in the rendering code -- keep them stable.
    expected_names = {"sm_active", "tensor_active", "fp32_active", "fp16_active", "fp64_active"}
    assert set(cvtop.DCGM_FIELD_NAMES.values()) == expected_names


def test_clamp_pct_handles_nan_and_range() -> None:
    """Regression guard for the NaN check -- PLR0124 rewrite used math.isnan."""
    import math

    import cvtop

    assert cvtop._clamp_pct(float("nan")) == 0.0
    assert cvtop._clamp_pct(-5.0) == 0.0
    assert cvtop._clamp_pct(150.0) == 100.0
    assert cvtop._clamp_pct(42.0) == 42.0
    assert not math.isnan(cvtop._clamp_pct(float("nan")))
