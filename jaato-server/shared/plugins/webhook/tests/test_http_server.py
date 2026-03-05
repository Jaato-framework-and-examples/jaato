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
            routes = {"generic": RouteConfig(path="/webhook")}

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
            "github": RouteConfig(path="/webhook/github"),
            "slack": RouteConfig(path="/webhook/slack"),
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
