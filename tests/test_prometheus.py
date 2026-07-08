"""Prometheus discovery/parsing/scraping against a local stdlib HTTP server."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from dltop.metrics import MetricStore
from dltop.sources.prometheus import (
    MAX_METRICS_PER_ENDPOINT,
    PromScraper,
    discover,
    parse_exposition,
)

EXPO = """\
# HELP vllm_num_requests_running Number of requests currently running.
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running{model="llama"} 3
# TYPE vllm_request_success_total counter
vllm_request_success_total{code="200"} 100
vllm_request_success_total{code="500"} 20
# TYPE vllm_e2e_latency_seconds histogram
vllm_e2e_latency_seconds_bucket{le="0.5"} 10
vllm_e2e_latency_seconds_sum 12.5
vllm_e2e_latency_seconds_count 40
untyped_metric 7.5
bad_value_metric NaN
"""


class _Handler(BaseHTTPRequestHandler):
    body = EXPO

    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_error(404)
            return
        payload = type(self).body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:  # silence test output
        del args


@pytest.fixture
def metrics_server() -> Iterator[int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


def test_parse_exposition_sums_labels_and_skips_histograms() -> None:
    parsed = parse_exposition(EXPO)
    assert parsed["vllm_num_requests_running"] == ("gauge", 3.0)
    assert parsed["vllm_request_success_total"] == ("counter", 120.0)
    assert parsed["untyped_metric"] == ("gauge", 7.5)
    assert "vllm_e2e_latency_seconds_bucket" not in parsed
    assert "vllm_e2e_latency_seconds_sum" not in parsed
    assert "bad_value_metric" not in parsed


def test_discover_finds_only_real_exposition_endpoints(metrics_server: int) -> None:
    found = discover(ports=[metrics_server, 1], timeout=0.3)
    assert [ep.port for ep in found] == [metrics_server]
    assert "vllm_num_requests_running" in found[0].metrics
    assert len(found[0].metrics) <= MAX_METRICS_PER_ENDPOINT


def test_scraper_records_gauges_and_counter_rates(metrics_server: int) -> None:
    store = MetricStore()
    ep_list = discover(ports=[metrics_server], timeout=0.3)
    scraper = PromScraper(ep_list, store, interval_s=1.0)
    port = metrics_server
    scraper.scrape_once(now=100.0)
    assert store.latest(f"prom:{port}:vllm_num_requests_running") == 3.0
    assert store.latest(f"prom:{port}:vllm_request_success_total") is None  # rate needs 2 points
    _Handler.body = EXPO.replace('code="200"} 100', 'code="200"} 130')
    scraper.scrape_once(now=110.0)
    assert store.latest(f"prom:{port}:vllm_request_success_total") == pytest.approx(3.0)  # +30 over 10 s
    _Handler.body = EXPO  # counter reset: 150 -> 120 must be skipped, not negative
    scraper.scrape_once(now=120.0)
    assert store.latest(f"prom:{port}:vllm_request_success_total") == pytest.approx(3.0)
