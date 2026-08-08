"""EgressProxyManager per-session lifecycle (§5.11c)."""

import socket

from server.egress_proxy.config import AllowlistConfig
from server.egress_proxy.manager import EgressProxyManager


def _allow():
    return AllowlistConfig(allowed_hosts=["api.anthropic.com"], allowed_ports=[443])


def test_start_returns_loopback_url_and_is_reachable():
    mgr = EgressProxyManager()
    try:
        url = mgr.start_proxy_for_session("s1", _allow())
        assert url.startswith("http://127.0.0.1:")
        port = int(url.rsplit(":", 1)[1])
        # the port is actually listening
        socket.create_connection(("127.0.0.1", port), 2).close()
    finally:
        mgr.shutdown()


def test_start_is_idempotent_per_session():
    mgr = EgressProxyManager()
    try:
        url1 = mgr.start_proxy_for_session("s1", _allow())
        url2 = mgr.start_proxy_for_session("s1", _allow())
        assert url1 == url2  # same proxy reused
    finally:
        mgr.shutdown()


def test_distinct_sessions_get_distinct_ports():
    mgr = EgressProxyManager()
    try:
        u1 = mgr.start_proxy_for_session("s1", _allow())
        u2 = mgr.start_proxy_for_session("s2", _allow())
        assert u1 != u2
    finally:
        mgr.shutdown()


def test_stop_closes_the_port():
    mgr = EgressProxyManager()
    url = mgr.start_proxy_for_session("s1", _allow())
    port = int(url.rsplit(":", 1)[1])
    mgr.stop_proxy_for_session("s1")
    assert mgr.get_proxy_url("s1") is None
    # connecting to the stopped port should now fail
    try:
        socket.create_connection(("127.0.0.1", port), 1).close()
        raised = False
    except OSError:
        raised = True
    assert raised


def test_stop_unknown_session_is_noop():
    mgr = EgressProxyManager()
    mgr.stop_proxy_for_session("nope")  # must not raise


def test_telemetry_aggregates():
    mgr = EgressProxyManager()
    try:
        mgr.start_proxy_for_session("s1", _allow())
        mgr.start_proxy_for_session("s2", _allow())
        tel = mgr.get_telemetry()
        assert tel["active_proxies"] == 2
        assert set(tel["totals"]) == {
            "connections_total", "allowed_total", "denied_total", "malformed_total"}
        assert set(tel["per_session"]) == {"s1", "s2"}
    finally:
        mgr.shutdown()
