"""Tests for the client-side doctor — focus on the daemon-env report.

The redaction is security-relevant: the doctor reads the daemon's full
environ and must NEVER print a secret-valued var, only that it is set.  The
"relevant" set is INJECTED (the framework's introspected read-set) so the test
is pure-jaato_sdk — it does not require jaato-server on the path.
"""

from jaato_sdk.doctor import check_daemon_env, DaemonInfo


def _info(env):
    return DaemonInfo(socket_path="/tmp/x", socket_exists=True,
                      listening=True, env=env)


# A stand-in for the framework's introspected {name: tier} read-set.
_KNOWN = {
    "JAATO_GC_THRESHOLD": "daemon",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "daemon",   # not JAATO_ — still reported
    "JAATO_NEBIUS_API_KEY": "daemon",          # secret → redacted
    "JAATO_LSP_TIMEOUT": "runner",             # runner-tier — tier-labeled
}


def test_daemon_env_reports_framework_readset_across_namespaces():
    detail = check_daemon_env(_info({
        "HOME": "/h", "PATH": "/p", "FOO_BAR": "x",     # not in read-set → excluded
        "JAATO_GC_THRESHOLD": "80.0",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317",
        "JAATO_LSP_TIMEOUT": "30",
    }), _KNOWN)[0].detail
    assert "JAATO_GC_THRESHOLD=80.0 [daemon]" in detail
    assert "OTEL_EXPORTER_OTLP_ENDPOINT=http://otel:4317 [daemon]" in detail  # non-JAATO included
    assert "JAATO_LSP_TIMEOUT=30 [runner]" in detail                          # tier label
    assert "HOME=" not in detail and "FOO_BAR=" not in detail                 # excluded


def test_daemon_env_redacts_secret_valued_vars():
    detail = check_daemon_env(
        _info({"JAATO_NEBIUS_API_KEY": "sk-secret-value"}), _KNOWN)[0].detail
    assert "sk-secret-value" not in detail        # never printed
    assert "<set, redacted>" in detail


def test_daemon_env_warns_without_introspector():
    c = check_daemon_env(_info({"JAATO_GC_THRESHOLD": "80.0"}), None)[0]
    assert c.status == "WARN"


def test_daemon_env_unavailable_when_not_listening():
    info = DaemonInfo(socket_path="/tmp/x", socket_exists=False,
                      listening=False, env=None)
    assert check_daemon_env(info, _KNOWN)[0].status == "WARN"


def test_daemon_env_reports_defaults_when_none_set():
    c = check_daemon_env(_info({"HOME": "/h"}), _KNOWN)[0]
    assert c.status == "PASS" and "defaults" in c.detail


# ── WebSocket transport preflight (check_websocket) ──────────────────────
#
# WS clients use the TS SDK; the doctor preflights the DAEMON side they hit:
# port listening, bearer-token file present + 0600, auth mode.

import socket as _socket

from jaato_sdk.doctor import check_websocket, PASS, FAIL, WARN


def _ws_info(pid=None):
    return DaemonInfo(socket_path="/tmp/x", socket_exists=False,
                      listening=False, pid=pid)


def test_websocket_bad_address():
    checks = check_websocket("not-a-port", _ws_info())
    assert checks[0].name == "websocket" and checks[0].status == FAIL


def test_websocket_port_listening_and_token_ok(tmp_path):
    srv = _socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    tok = tmp_path / "ws.token"
    tok.write_text("secret")
    tok.chmod(0o600)
    try:
        checks = check_websocket(f"127.0.0.1:{port}", _ws_info(),
                                 ws_token_file=str(tok))
    finally:
        srv.close()
    by = {c.name: c for c in checks}
    assert by["ws-port"].status == PASS
    assert by["ws-token"].status == PASS
    assert "Bearer" in by["ws-token"].detail


def test_websocket_port_down_fails():
    checks = check_websocket("127.0.0.1:1", _ws_info())   # nothing on :1
    by = {c.name: c for c in checks}
    assert by["ws-port"].status == FAIL


def test_websocket_loose_token_mode_warns(tmp_path):
    srv = _socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    tok = tmp_path / "ws.token"
    tok.write_text("secret")
    tok.chmod(0o644)   # too permissive
    try:
        checks = check_websocket(f"127.0.0.1:{port}", _ws_info(),
                                 ws_token_file=str(tok))
    finally:
        srv.close()
    by = {c.name: c for c in checks}
    assert by["ws-token"].status == WARN and "0600" in by["ws-token"].detail
