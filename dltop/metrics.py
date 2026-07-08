"""Central metric history: named ring buffers plus rolling-window statistics."""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dltop.models import HISTORY_LEN

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class WindowStats:
    """Instantaneous value plus rolling-window aggregates for one series."""

    now: float
    mean: float
    median: float
    stddev: float
    n_samples: int


class MetricStore:
    """Named ring buffers of ``(timestamp, value)`` samples.

    Sources write via :meth:`record` / :meth:`record_many`; charts and the
    stats table read via :meth:`tail`, :meth:`latest` and :meth:`stats`.
    Unknown series read as empty rather than raising, so read-side widgets
    never need to know which sources are active.
    """

    def __init__(self, maxlen: int = HISTORY_LEN) -> None:
        """Create a store whose per-series ring buffers hold ``maxlen`` samples."""
        self._maxlen = maxlen
        self._data: dict[str, deque[tuple[float, float]]] = {}

    def record(self, name: str, value: float, ts: float | None = None) -> None:
        """Append one sample; the series is created on first use."""
        when = time.time() if ts is None else ts
        buf = self._data.get(name)
        if buf is None:
            buf = self._data[name] = deque(maxlen=self._maxlen)
        buf.append((when, value))

    def record_many(self, samples: Mapping[str, float], ts: float | None = None) -> None:
        """Append one sample per entry, all sharing a single timestamp."""
        when = time.time() if ts is None else ts
        for name, value in samples.items():
            self.record(name, value, ts=when)

    def names(self) -> list[str]:
        """Return every series name that has received at least one sample."""
        return list(self._data)

    def latest(self, name: str) -> float | None:
        """Return the most recent value for ``name``, or None if never recorded."""
        buf = self._data.get(name)
        return buf[-1][1] if buf else None

    def tail(self, name: str, n: int) -> list[tuple[float, float]]:
        """Return the last ``n`` samples (oldest first); empty for unknown series."""
        buf = self._data.get(name)
        if not buf or n <= 0:
            return []
        return list(buf)[-n:]

    def window_values(self, name: str, window_s: float, now: float | None = None) -> list[float]:
        """Return finite values sampled within the trailing ``window_s`` seconds."""
        buf = self._data.get(name)
        if not buf:
            return []
        cutoff = (time.time() if now is None else now) - window_s
        return [v for ts, v in buf if ts >= cutoff and math.isfinite(v)]

    def stats(self, name: str, window_s: float, now: float | None = None) -> WindowStats | None:
        """Compute Now/mean/median/stddev over the window; None if it is empty."""
        values = self.window_values(name, window_s, now=now)
        if not values:
            return None
        return WindowStats(
            now=values[-1],
            mean=statistics.fmean(values),
            median=statistics.median(values),
            stddev=statistics.stdev(values) if len(values) >= 2 else 0.0,
            n_samples=len(values),
        )
