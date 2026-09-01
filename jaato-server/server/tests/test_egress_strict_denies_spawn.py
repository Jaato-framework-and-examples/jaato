"""The spawn path must not swallow a *required* egress-enforcement failure (#696).

``_apply_egress_env`` is deliberately fail-safe: any wire-up error degrades to
"session runs without egress restriction" rather than breaking spawn.  The one
exception is the strict posture — when the operator set
``JAATO_EGRESS_NFT_ENFORCE=strict`` they asked for a kernel-enforced gate, so a
session that would otherwise start with no gate at all must be denied instead.
"""

import pytest

from server import runner_spawn
from server.egress_proxy.errors import EgressEnforcementError


def _apply(monkeypatch, raiser):
    """Run ``_apply_egress_env`` with the wire-up stubbed to ``raiser``."""
    import server.egress_proxy.wireup as wireup
    monkeypatch.setattr(wireup, "egress_env_for_session", raiser)
    env = {"PATH": "/usr/bin"}
    runner_spawn._apply_egress_env(
        "s1", {"egress_allowlist": {"allowed_hosts": ["x.com"]}}, env)
    return env


def test_strict_enforcement_failure_propagates(monkeypatch):
    def _raise(session_id, allowlist):
        raise EgressEnforcementError("gate unavailable")

    with pytest.raises(EgressEnforcementError):
        _apply(monkeypatch, _raise)


def test_other_wireup_errors_still_degrade(monkeypatch):
    def _raise(session_id, allowlist):
        raise RuntimeError("proxy would not bind")

    env = _apply(monkeypatch, _raise)          # must not raise
    assert env == {"PATH": "/usr/bin"}         # and leaves the env untouched


def test_success_merges_proxy_env(monkeypatch):
    def _ok(session_id, allowlist):
        return {"HTTPS_PROXY": "http://127.0.0.1:9"}, []

    env = _apply(monkeypatch, _ok)
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:9"


def test_no_allowlist_leaves_env_untouched(monkeypatch):
    import server.egress_proxy.wireup as wireup
    calls = []

    def _spy(session_id, allowlist):
        calls.append(allowlist)
        return {}, []

    monkeypatch.setattr(wireup, "egress_env_for_session", _spy)
    env = {"PATH": "/usr/bin"}
    runner_spawn._apply_egress_env("s1", {}, env)
    assert env == {"PATH": "/usr/bin"}
    assert calls == [None]
