"""Tests for the webhook HTTP server."""

import json
import socket
import time
import urllib.request
from typing import Any, Dict, List, Tuple
from unittest import mock

import pytest

from shared.plugins.webhook.config import RouteConfig, WebhookConfig
from shared.plugins.webhook.http_server import WebhookHTTPServer


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class TestWebhookHTTPServer:
    """Integration tests for WebhookHTTPServer."""

    def _make_server(self, routes=None, secret=None):
        """Create a server with a free port and a callback that records events."""
        port = _find_free_port()
        if routes is None:
            # Localhost integration tests of the HTTP server itself (not auth):
            # opt the route into unsigned acceptance so the fail-closed gate
            # doesn't 401 the request under test.
            routes = {"generic": RouteConfig(path="/webhook",
                                             allow_unauthenticated=True)}

        config = WebhookConfig(
            port=port,
            host="127.0.0.1",
            secret=secret,
            routes=routes,
        )

        received: List[Tuple[str, str, Dict, Any]] = []

        def on_webhook(route_name, event_type, headers, payload):
            received.append((route_name, event_type, headers, payload))

        server = WebhookHTTPServer(config, on_webhook)
        return server, port, received

    def _post(self, port, path, payload, headers=None):
        """Send a POST request to the test server."""
        url = f"http://127.0.0.1:{port}{path}"
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def _get(self, port, path):
        """Send a GET request to the test server."""
        url = f"http://127.0.0.1:{port}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_start_and_stop(self):
        server, port, _ = self._make_server()
        server.start()
        assert server.is_running
        server.stop()
        assert not server.is_running

    def test_post_valid_webhook(self):
        server, port, received = self._make_server()
        server.start()
        try:
            status, body = self._post(port, "/webhook", {"action": "test"})
            assert status == 200
            assert body["status"] == "accepted"
            assert len(received) == 1
            assert received[0][0] == "generic"  # route_name
            assert received[0][3] == {"action": "test"}  # payload
        finally:
            server.stop()

    def test_post_unmatched_path(self):
        server, port, received = self._make_server()
        server.start()
        try:
            status, body = self._post(port, "/unknown", {})
            assert status == 404
            assert len(received) == 0
        finally:
            server.stop()

    def test_get_rejected(self):
        server, port, _ = self._make_server()
        server.start()
        try:
            status, body = self._get(port, "/webhook")
            assert status == 405
        finally:
            server.stop()

    def test_multiple_routes(self):
        routes = {
            "github": RouteConfig(path="/webhook/github", allow_unauthenticated=True),
            "slack": RouteConfig(path="/webhook/slack", allow_unauthenticated=True),
        }
        server, port, received = self._make_server(routes=routes)
        server.start()
        try:
            self._post(port, "/webhook/github", {"ref": "main"})
            self._post(port, "/webhook/slack", {"text": "hello"})
            assert len(received) == 2
            assert received[0][0] == "github"
            assert received[1][0] == "slack"
        finally:
            server.stop()

    def test_event_type_from_header(self):
        routes = {
            "github": RouteConfig(
                path="/webhook/github",
                event_type_header="X-GitHub-Event",
                allow_unauthenticated=True,
            ),
        }
        server, port, received = self._make_server(routes=routes)
        server.start()
        try:
            self._post(
                port, "/webhook/github", {"action": "opened"},
                headers={"X-GitHub-Event": "pull_request"},
            )
            assert received[0][1] == "pull_request"  # event_type
        finally:
            server.stop()

    def test_body_too_large(self):
        routes = {"generic": RouteConfig(path="/webhook")}
        port = _find_free_port()
        config = WebhookConfig(
            port=port,
            host="127.0.0.1",
            routes=routes,
            max_body_size=10,  # Very small limit
        )
        received = []
        server = WebhookHTTPServer(config, lambda *a: received.append(a))
        server.start()
        try:
            status, body = self._post(port, "/webhook", {"large": "x" * 100})
            assert status == 413
            assert len(received) == 0
        finally:
            server.stop()

    def test_get_stats(self):
        server, port, _ = self._make_server()
        server.start()
        try:
            self._post(port, "/webhook", {"a": 1})
            self._post(port, "/webhook", {"b": 2})
            stats = server.get_stats()
            assert stats["listening"] is True
            assert stats["total_events_received"] == 2
            assert stats["routes"][0]["events_received"] == 2
        finally:
            server.stop()

    def test_stats_when_not_running(self):
        server, port, _ = self._make_server()
        stats = server.get_stats()
        assert stats["listening"] is False
        assert stats["total_events_received"] == 0

    def test_stats_include_tls_field(self):
        server, port, _ = self._make_server()
        stats = server.get_stats()
        assert stats["tls_enabled"] is False


class TestIPAllowlist:
    """Tests for IP allowlist filtering."""

    def test_check_ip_allowed_empty_list(self):
        """Empty allowlist means all IPs are allowed."""
        config = WebhookConfig(port=0, allowed_ips=[])
        server = WebhookHTTPServer(config, lambda *a: None)
        assert server.check_ip_allowed("1.2.3.4") is True

    def test_check_ip_allowed_single_ip(self):
        config = WebhookConfig(port=0, allowed_ips=["127.0.0.1"])
        server = WebhookHTTPServer(config, lambda *a: None)
        assert server.check_ip_allowed("127.0.0.1") is True
        assert server.check_ip_allowed("10.0.0.1") is False

    def test_check_ip_allowed_cidr(self):
        config = WebhookConfig(port=0, allowed_ips=["192.168.1.0/24"])
        server = WebhookHTTPServer(config, lambda *a: None)
        assert server.check_ip_allowed("192.168.1.50") is True
        assert server.check_ip_allowed("192.168.2.50") is False

    def test_check_ip_allowed_ipv6(self):
        config = WebhookConfig(port=0, allowed_ips=["::1"])
        server = WebhookHTTPServer(config, lambda *a: None)
        assert server.check_ip_allowed("::1") is True
        assert server.check_ip_allowed("::2") is False

    def test_check_ip_allowed_ipv4_mapped_ipv6(self):
        """::ffff:127.0.0.1 should match 127.0.0.1 in allowlist."""
        config = WebhookConfig(port=0, allowed_ips=["127.0.0.1"])
        server = WebhookHTTPServer(config, lambda *a: None)
        assert server.check_ip_allowed("::ffff:127.0.0.1") is True

    def test_blocked_ip_returns_403(self):
        """Integration: blocked IP gets 403 from actual HTTP server."""
        port = _find_free_port()
        routes = {"generic": RouteConfig(path="/webhook")}
        # Allowlist only 10.x — localhost (127.0.0.1) will be blocked
        config = WebhookConfig(
            port=port, host="127.0.0.1", routes=routes,
            allowed_ips=["10.0.0.0/8"],
        )
        received = []
        server = WebhookHTTPServer(config, lambda *a: received.append(a))
        server.start()
        try:
            url = f"http://127.0.0.1:{port}/webhook"
            body = json.dumps({"test": True}).encode()
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                assert False, "Expected HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 403
            assert len(received) == 0
            assert server.requests_blocked_ip >= 1
        finally:
            server.stop()


class TestRateLimiting:
    """Tests for per-IP rate limiting."""

    def test_no_rate_limit(self):
        """rate_limit_per_second=0 means unlimited."""
        config = WebhookConfig(port=0, rate_limit_per_second=0)
        server = WebhookHTTPServer(config, lambda *a: None)
        for _ in range(100):
            assert server.check_rate_limit("1.2.3.4") is True

    def test_rate_limit_blocks_excess(self):
        """Burst beyond the limit should be blocked."""
        config = WebhookConfig(port=0, rate_limit_per_second=2)
        server = WebhookHTTPServer(config, lambda *a: None)
        # First 2 should pass (bucket starts full)
        assert server.check_rate_limit("1.2.3.4") is True
        assert server.check_rate_limit("1.2.3.4") is True
        # Third should be blocked (bucket empty)
        assert server.check_rate_limit("1.2.3.4") is False

    def test_rate_limit_refills(self):
        """Tokens refill over time."""
        import time
        config = WebhookConfig(port=0, rate_limit_per_second=10)
        server = WebhookHTTPServer(config, lambda *a: None)
        # Exhaust all tokens
        for _ in range(10):
            server.check_rate_limit("1.2.3.4")
        assert server.check_rate_limit("1.2.3.4") is False
        # Wait for refill
        time.sleep(0.15)
        assert server.check_rate_limit("1.2.3.4") is True

    def test_rate_limit_per_ip(self):
        """Each IP gets its own bucket."""
        config = WebhookConfig(port=0, rate_limit_per_second=1)
        server = WebhookHTTPServer(config, lambda *a: None)
        assert server.check_rate_limit("1.1.1.1") is True
        assert server.check_rate_limit("1.1.1.1") is False  # exhausted
        assert server.check_rate_limit("2.2.2.2") is True   # separate bucket

    def test_rate_limit_integration_429(self):
        """Integration: rate-limited requests get 429 from actual server."""
        port = _find_free_port()
        routes = {"generic": RouteConfig(path="/webhook", allow_unauthenticated=True)}
        config = WebhookConfig(
            port=port, host="127.0.0.1", routes=routes,
            rate_limit_per_second=1,
        )
        received = []
        server = WebhookHTTPServer(config, lambda *a: received.append(a))
        server.start()
        try:
            url = f"http://127.0.0.1:{port}/webhook"
            body = json.dumps({"test": True}).encode()

            def _do_post():
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    resp = urllib.request.urlopen(req, timeout=5)
                    return resp.status
                except urllib.error.HTTPError as e:
                    return e.code

            # First request should succeed
            assert _do_post() == 200
            # Rapid second should be rate-limited
            assert _do_post() == 429
            assert server.requests_blocked_rate >= 1
        finally:
            server.stop()
