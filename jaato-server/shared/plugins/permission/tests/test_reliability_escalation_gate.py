"""Reliability Phase-2 escalation enforcement: a reactor-escalated tool is
re-confirmed ("ask") on interactive clients and denied ("deny", T1) on headless
clients; strict no-op (None) otherwise (no session / not escalated / unknown)."""
from types import SimpleNamespace
from shared.plugins.permission.plugin import PermissionPlugin
from shared.session_context import _current_session
from jaato_sdk.events import ClientType


def _fake_session(escalated, client_type, approved=None):
    state = {}
    if escalated is not None:
        state["reliability:escalated_tools"] = escalated
    if approved is not None:
        state["reliability:approved_tools"] = approved
    return SimpleNamespace(
        get_session_state=lambda k, d=None: state.get(k, d),
        _presentation_context=SimpleNamespace(client_type=client_type))


def _action(tool, escalated, client_type, approved=None):
    token = _current_session.set(_fake_session(escalated, client_type, approved))
    try:
        return PermissionPlugin()._reliability_escalation_action(tool)
    finally:
        _current_session.reset(token)


def test_escalated_interactive_asks():
    for ct in (ClientType.TERMINAL, ClientType.WEB, ClientType.CHAT):
        assert _action("svc", ["svc"], ct) == "ask", ct


def test_escalated_headless_denies():            # T1
    assert _action("svc", ["svc"], ClientType.API) == "deny"


def test_approved_overrides_escalated():         # T3 resume primitive
    # An escalated tool that is ALSO approved → no enforcement (allowed),
    # overriding both the headless deny and the interactive ask.
    assert _action("svc", ["svc"], ClientType.API, approved=["svc"]) is None
    assert _action("svc", ["svc"], ClientType.TERMINAL, approved=["svc"]) is None
    # control: approval names a DIFFERENT tool → enforcement still applies.
    assert _action("svc", ["svc"], ClientType.API, approved=["other"]) == "deny"


def test_not_escalated_no_action():
    assert _action("svc", [], ClientType.TERMINAL) is None
    assert _action("svc", ["other"], ClientType.TERMINAL) is None
    assert _action("svc", None, ClientType.API) is None


def test_unknown_presentation_no_action():
    # Escalated but no resolvable client_type → no enforcement (safe default).
    assert _action("svc", ["svc"], None) is None


def test_no_session_context_no_action():
    # No session set in this context → get_current_session raises → None.
    assert PermissionPlugin()._reliability_escalation_action("svc") is None


def test_ask_bypasses_allow_all(monkeypatch):
    p = PermissionPlugin()
    p._allow_all = True
    p._policy = None  # falls through to 'not_initialized' after the gates
    monkeypatch.setattr(p, "_reliability_escalation_action", lambda t: "ask")
    _, meta = p.check_permission("svc", {})
    assert meta["method"] != "allow_all"           # allow_all bypassed by escalation
    monkeypatch.setattr(p, "_reliability_escalation_action", lambda t: None)
    allowed, meta2 = p.check_permission("svc", {})
    assert allowed is True and meta2["method"] == "allow_all"   # control


def test_deny_blocks_headless_escalated_tool(monkeypatch):   # T1 enforcement
    p = PermissionPlugin()
    p._allow_all = True       # even under allow_all, the deny wins
    p._policy = None
    monkeypatch.setattr(p, "_reliability_escalation_action", lambda t: "deny")
    allowed, meta = p.check_permission("svc", {})
    assert allowed is False
    assert meta["method"] == "reliability_escalation_denied"
