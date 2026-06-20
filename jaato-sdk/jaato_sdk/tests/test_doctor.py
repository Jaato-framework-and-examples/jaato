"""Tests for the client-side doctor — focus on the daemon-env report.

The redaction is security-relevant: the doctor reads the daemon's full
environ and must NEVER print a secret-valued var, only that it is set.
"""

from jaato_sdk.doctor import check_daemon_env, DaemonInfo


def _info(env):
    return DaemonInfo(socket_path="/tmp/x", socket_exists=True,
                      listening=True, env=env)


def test_daemon_env_scopes_to_jaato_and_redacts_secrets():
    detail = check_daemon_env(_info({
        "HOME": "/h", "PATH": "/p",                       # non-JAATO → excluded
        "JAATO_PROVIDER": "nebius",
        "JAATO_GC_THRESHOLD": "80.0",
        "JAATO_NEBIUS_API_KEY": "sk-secret-value",        # secret → redacted
        "JAATO_WS_TOKEN": "tok-value",                    # secret → redacted
    }))[0].detail
    assert "JAATO_PROVIDER=nebius" in detail
    assert "JAATO_GC_THRESHOLD=80.0" in detail
    # secrets must never appear verbatim
    assert "sk-secret-value" not in detail and "tok-value" not in detail
    assert detail.count("<set, redacted>") == 2
    # non-JAATO vars are not reported
    assert "HOME=" not in detail and "PATH=" not in detail


def test_daemon_env_unavailable_when_not_listening():
    info = DaemonInfo(socket_path="/tmp/x", socket_exists=False,
                      listening=False, env=None)
    c = check_daemon_env(info)[0]
    assert c.status == "WARN"


def test_daemon_env_reports_defaults_when_no_jaato_vars():
    c = check_daemon_env(_info({"HOME": "/h"}))[0]
    assert c.status == "PASS" and "defaults" in c.detail
