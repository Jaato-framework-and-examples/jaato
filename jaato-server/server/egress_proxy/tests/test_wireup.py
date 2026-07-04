"""Wire-up glue: opt-in env injection + teardown (§5.11e)."""

import socket

from server.egress_proxy import wireup


def _reset():
    # Ensure a clean singleton per test.
    wireup.shutdown_all()
    wireup._manager = None


def test_no_allowlist_returns_empty_env_and_no_proxy():
    _reset()
    env, errs = wireup.egress_env_for_session("s1", None)
    assert env == {} and errs == []
    # nothing started
    assert wireup._manager is None or wireup.get_manager().get_proxy_url("s1") is None


def test_valid_allowlist_starts_proxy_and_sets_env():
    _reset()
    try:
        env, errs = wireup.egress_env_for_session(
            "s1", {"allowed_hosts": ["api.anthropic.com"], "allowed_ports": [443]})
        assert errs == []
        assert env["HTTPS_PROXY"].startswith("http://127.0.0.1:")
        assert env["HTTPS_PROXY"] == env["https_proxy"] == env["HTTP_PROXY"]
        assert "127.0.0.1" in env["NO_PROXY"]
        # proxy is actually listening
        port = int(env["HTTPS_PROXY"].rsplit(":", 1)[1])
        socket.create_connection(("127.0.0.1", port), 2).close()
    finally:
        _reset()


def test_invalid_allowlist_returns_errors_and_no_proxy():
    _reset()
    env, errs = wireup.egress_env_for_session("s1", {"allowed_hosts": []})
    assert env == {}
    assert errs  # non-empty
    assert wireup.get_manager().get_proxy_url("s1") is None
    _reset()


def test_teardown_stops_proxy():
    _reset()
    env, _ = wireup.egress_env_for_session(
        "s1", {"allowed_hosts": ["x.com"]})
    port = int(env["HTTPS_PROXY"].rsplit(":", 1)[1])
    wireup.egress_teardown("s1")
    assert wireup.get_manager().get_proxy_url("s1") is None
    try:
        socket.create_connection(("127.0.0.1", port), 1).close()
        raised = False
    except OSError:
        raised = True
    assert raised
    _reset()


def test_teardown_before_any_start_is_noop():
    _reset()
    wireup.egress_teardown("never")  # must not raise
