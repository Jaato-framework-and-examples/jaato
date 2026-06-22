"""Reliability Phase-2 interactive escalation gate: a reactor-escalated tool on
an interactive client is re-confirmed (prompted) even when auto-approved;
strict no-op otherwise (no session / not escalated / headless)."""
from types import SimpleNamespace
from shared.plugins.permission.plugin import PermissionPlugin
from shared.session_context import _current_session
from jaato_sdk.events import ClientType


def _fake_session(escalated, client_type):
    state = {} if escalated is None else {"reliability:escalated_tools": escalated}
    return SimpleNamespace(
        get_session_state=lambda k, d=None: state.get(k, d),
        _presentation_context=SimpleNamespace(client_type=client_type))


def _force(tool, escalated, client_type):
    token = _current_session.set(_fake_session(escalated, client_type))
    try:
        return PermissionPlugin()._reliability_force_prompt(tool)
    finally:
        _current_session.reset(token)


def test_escalated_interactive_forces_prompt():
    for ct in (ClientType.TERMINAL, ClientType.WEB, ClientType.CHAT):
        assert _force("svc", ["svc"], ct) is True, ct


def test_headless_never_forces():
    assert _force("svc", ["svc"], ClientType.API) is False


def test_not_escalated_does_not_force():
    assert _force("svc", [], ClientType.TERMINAL) is False
    assert _force("svc", ["other"], ClientType.TERMINAL) is False
    assert _force("svc", None, ClientType.TERMINAL) is False


def test_no_session_context_does_not_force():
    # No session set in this context → get_current_session raises → False.
    assert PermissionPlugin()._reliability_force_prompt("svc") is False


def test_escalation_bypasses_allow_all(monkeypatch):
    p = PermissionPlugin()
    p._allow_all = True
    p._policy = None  # falls through to 'not_initialized' after the gates
    monkeypatch.setattr(p, "_reliability_force_prompt", lambda t: True)
    _, meta = p.check_permission("svc", {})
    assert meta["method"] != "allow_all"           # allow_all bypassed by escalation
    # control: without escalation, allow_all still applies
    monkeypatch.setattr(p, "_reliability_force_prompt", lambda t: False)
    allowed, meta2 = p.check_permission("svc", {})
    assert allowed is True and meta2["method"] == "allow_all"
