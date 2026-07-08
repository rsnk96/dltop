"""Discover and scrape Prometheus /metrics endpoints on localhost.

Stdlib only: enumerating listeners uses psutil when available (``ss -lnt``
fallback), probing/scraping uses urllib with short timeouts, and the
text-exposition parser handles exactly what dltop renders — gauges and
counters. Histograms and summaries are skipped: their component series are
meaningless as single lines on a chart.
"""

from __future__ import annotations

import http.client
import math
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from dltop.sources.system import psutil

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dltop.metrics import MetricStore

MAX_METRICS_PER_ENDPOINT = 30
MAX_RESPONSE_BYTES = 1_000_000
_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


@dataclass
class PromEndpoint:
    """One discovered /metrics endpoint and its (stable) chosen metric names."""

    port: int
    name: str
    metrics: list[str] = field(default_factory=list)


def _parse_type_line(line: str, types: dict[str, str]) -> None:
    """Record ``# TYPE <name> <type>`` into ``types`` (silently skipped if malformed)."""
    parts = line.split()
    if len(parts) >= 4:
        types[parts[2]] = parts[3]


def _accumulate_sample(line: str, sums: dict[str, float], order: list[str]) -> None:
    """Parse one exposition sample line and add its value into the running sum."""
    m = _SAMPLE_RE.match(line)
    if m is None:
        return
    name, _, value_token = m.groups()
    try:
        value = float(value_token)
    except ValueError:
        return
    if not math.isfinite(value):
        return
    if name not in sums:
        order.append(name)
    sums[name] = sums.get(name, 0.0) + value


def parse_exposition(text: str) -> dict[str, tuple[str, float]]:
    """Parse Prometheus text exposition into ``{name: (type, summed_value)}``."""
    types: dict[str, str] = {}
    sums: dict[str, float] = {}
    order: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("# TYPE "):
            _parse_type_line(line, types)
        elif line and not line.startswith("#"):
            _accumulate_sample(line, sums, order)
    skip_types = {"histogram", "summary"}
    histo_parents = {n for n, t in types.items() if t in skip_types}
    return {
        name: (types.get(name, "gauge"), sums[name])
        for name in order
        if name.removesuffix("_bucket").removesuffix("_sum").removesuffix("_count") not in histo_parents
    }


def listening_ports() -> set[int]:
    """Enumerate locally listening TCP ports (psutil, else ``ss -lnt``)."""
    if psutil is not None:
        try:
            return {c.laddr.port for c in psutil.net_connections(kind="tcp") if c.status == "LISTEN" and c.laddr}
        except psutil.Error as exc:  # pragma: no cover - permission-dependent
            logger.debug("psutil.net_connections failed ({}), falling back to ss", exc)
    try:
        r = subprocess.run(["ss", "-lnt"], capture_output=True, text=True, timeout=3, check=False)  # noqa: S607
    except (OSError, subprocess.TimeoutExpired):
        return set()
    ports: set[int] = set()
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4 and ":" in parts[3]:
            with_port = parts[3].rsplit(":", 1)[1]
            if with_port.isdigit():
                ports.add(int(with_port))
    return ports


def _probe(port: int, timeout: float) -> str | None:
    """Fetch /metrics on ``port``; return the body only if it looks like exposition."""
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - localhost only
            body = resp.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException):
        # A non-HTTP service on this port makes urllib raise http.client
        # exceptions (e.g. BadStatusLine) that are NOT OSError subclasses; a
        # probe of a random listener must never crash discovery.
        return None
    if "# TYPE" in body or _SAMPLE_RE.match(body.lstrip().splitlines()[0] if body.strip() else ""):
        return body
    return None


def _port_owner(port: int) -> str:
    """Best-effort process name listening on ``port`` (else ``port<N>``)."""
    if psutil is not None:
        try:
            for c in psutil.net_connections(kind="tcp"):
                if c.status == "LISTEN" and c.laddr and c.laddr.port == port and c.pid:
                    return psutil.Process(c.pid).name()
        except psutil.Error:  # pragma: no cover - permission-dependent
            pass
    return f"port{port}"


def discover(ports: Iterable[int] | None = None, timeout: float = 0.3) -> list[PromEndpoint]:
    """Probe candidate ports in parallel; return endpoints exposing Prometheus metrics."""
    candidates = sorted(set(ports) if ports is not None else listening_ports())
    if not candidates:
        return []
    endpoints: list[PromEndpoint] = []
    with ThreadPoolExecutor(max_workers=min(32, len(candidates))) as pool:
        for port, body in zip(candidates, pool.map(lambda p: _probe(p, timeout), candidates), strict=True):
            if body is None:
                continue
            parsed = parse_exposition(body)
            chosen = [n for n, (t, _) in parsed.items() if t in ("gauge", "counter")][:MAX_METRICS_PER_ENDPOINT]
            if chosen:
                endpoints.append(PromEndpoint(port=port, name=_port_owner(port), metrics=chosen))
    logger.info("Prometheus discovery: {} endpoint(s) found", len(endpoints))
    return endpoints


class PromScraper:
    """Background thread recording gauge values and counter rates into the store."""

    def __init__(self, endpoints: list[PromEndpoint], store: MetricStore, interval_s: float) -> None:
        """Scrape ``endpoints`` every ``interval_s`` seconds into ``store``."""
        self.endpoints = endpoints
        self.store = store
        self.interval_s = max(1.0, interval_s)
        self._prev: dict[str, tuple[float, float]] = {}  # series -> (ts, raw value)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def scrape_once(self, now: float) -> None:
        """Fetch every endpoint once and record samples stamped ``now``."""
        for ep in self.endpoints:
            body = _probe(ep.port, timeout=2.0)
            if body is None:
                continue
            parsed = parse_exposition(body)
            for metric in ep.metrics:
                if metric not in parsed:
                    continue
                mtype, value = parsed[metric]
                series = f"prom:{ep.port}:{metric}"
                if mtype == "counter":
                    prev = self._prev.get(series)
                    self._prev[series] = (now, value)
                    if prev is None or now <= prev[0] or value < prev[1]:  # first point or reset
                        continue
                    self.store.record(series, (value - prev[1]) / (now - prev[0]), ts=now)
                else:
                    self.store.record(series, value, ts=now)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.scrape_once(time.time())

    def start(self) -> None:
        """Start the background scrape thread."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the scrape thread to exit."""
        self._stop.set()
