"""Unit tests for MetricStore and window statistics (GPU-less)."""

from __future__ import annotations

import pytest

from dltop.metrics import MetricStore, WindowStats


def test_record_and_tail_order() -> None:
    store = MetricStore(maxlen=4)
    for i in range(6):
        store.record("cpu", float(i), ts=float(i))
    assert store.tail("cpu", 10) == [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]  # ring buffer trims
    assert store.tail("cpu", 2) == [(4.0, 4.0), (5.0, 5.0)]
    assert store.tail("unknown", 5) == []  # unknown series is empty, not an error


def test_latest_and_names() -> None:
    store = MetricStore()
    assert store.latest("cpu") is None
    store.record_many({"cpu": 10.0, "ram": 20.0}, ts=1.0)
    assert store.latest("cpu") == 10.0
    assert sorted(store.names()) == ["cpu", "ram"]


def test_window_values_respects_window_and_skips_non_finite() -> None:
    store = MetricStore()
    store.record("x", 1.0, ts=0.0)
    store.record("x", float("nan"), ts=50.0)
    store.record("x", 3.0, ts=60.0)
    store.record("x", 5.0, ts=100.0)
    assert store.window_values("x", window_s=60.0, now=100.0) == [3.0, 5.0]  # ts >= 40, NaN dropped


def test_stats_math() -> None:
    store = MetricStore()
    for ts, v in [(1.0, 2.0), (2.0, 4.0), (3.0, 4.0), (4.0, 4.0), (5.0, 5.0), (6.0, 5.0), (7.0, 7.0), (8.0, 9.0)]:
        store.record("x", v, ts=ts)
    s = store.stats("x", window_s=100.0, now=8.0)
    assert s == WindowStats(now=9.0, mean=5.0, median=4.5, stddev=pytest.approx(2.138, abs=1e-3), n_samples=8)


def test_stats_single_sample_has_zero_stddev() -> None:
    store = MetricStore()
    store.record("x", 42.0, ts=1.0)
    s = store.stats("x", window_s=60.0, now=1.0)
    assert s == WindowStats(now=42.0, mean=42.0, median=42.0, stddev=0.0, n_samples=1)


def test_stats_empty_window_is_none() -> None:
    store = MetricStore()
    assert store.stats("x", window_s=60.0) is None
    store.record("x", float("inf"), ts=1.0)
    assert store.stats("x", window_s=60.0, now=1.0) is None
