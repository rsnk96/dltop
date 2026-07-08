"""Demo-source tests: the app must run fully without NVML hardware."""

from __future__ import annotations

from dltop.app import DltopApp
from dltop.models import SystemState
from dltop.sources.demo import DemoSource


def test_demo_source_shapes() -> None:
    demo = DemoSource(n_gpus=3)
    gpus = demo.init_gpus()
    assert [g.index for g in gpus] == [0, 1, 2]
    sys_state = SystemState()
    demo.sample(gpus, sys_state, t=5.0)
    for g in gpus:
        assert 0.0 <= g.sm_overall_pct <= 100.0
        assert 0.0 <= g.dcgm["tensor_active"] <= 100.0
        assert g.mem_total_mb > 0
    assert 0.0 <= sys_state.cpu_pct <= 100.0
    assert demo.processes(), "demo must show a believable process table"


def test_gpu_phases_differ() -> None:
    demo = DemoSource(n_gpus=2)
    gpus = demo.init_gpus()
    demo.sample(gpus, SystemState(), t=3.0)
    assert gpus[0].sm_overall_pct != gpus[1].sm_overall_pct


async def test_app_runs_headless_in_demo_mode() -> None:
    app = DltopApp(interval=0.1, no_dcgm=True, demo_gpus=2, no_discover=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert len(app.gpus) == 2
        assert app.have_profiling  # demo exposes the full DCGM-style series
