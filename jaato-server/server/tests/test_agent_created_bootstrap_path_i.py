"""Regression-pin tests for Path I — bootstrap-scope AgentCreatedEvent emit.

**Root cause** (cycle-11 verdict):

Path F installed the ``_AgentUIHooksNotificationShim`` inside
``_handle_session_send_message`` (per-RPC-request scope).  Events
whose callback fires DURING send_message are captured; events
that fire OUTSIDE send_message (bootstrap, post-completion) are
silently dropped — runner-side ``_ui_hooks`` is still None at
those times.

``on_agent_created`` fires at session bootstrap — BEFORE any
send_message arrives.  Pre-§7c the equivalent in-process call
emitted ``AgentCreatedEvent`` via the daemon-side ``_ui_hooks``;
post-§7c the runner-side ``_ui_hooks`` is None, so the event
silently drops.  Without ``AgentCreatedEvent`` the TUI has no
agent registry and every subsequent agent-keyed event
(``ToolCallStartEvent``, ``PermissionRequestedEvent``) refs an
unknown agent → discarded.

**Path I fix** (Option A — direct daemon-side emit):

I.1 ``_create_main_agent`` (core.py:2301) emits
    ``AgentCreatedEvent`` after constructing the local
    ``AgentState``.  Daemon owns session-lifecycle, so
    synthesizing the event daemon-side is architecturally
    natural.

I.2 Dead-letter comment "AgentCreatedEvent is NOT emitted here -
    it's handled by ServerAgentHooks.on_agent_created() when
    set_ui_hooks() is called" replaced — that hook-driven path
    has been dead since §7c.

I.3 ``AgentState`` constructed with full field set (profile_name
    + parent_agent_id) matching the hook's payload.

Tests pin:
- _create_main_agent emits AgentCreatedEvent
- Event payload matches the hook's payload shape (field-by-field)
- Pre-registered agent path (hook ran first) still skips emit
- AgentState carries profile_name + parent_agent_id
- AST: _create_main_agent body contains AgentCreatedEvent
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import AgentCreatedEvent


# ----------------------------------------------------------------------
# I.1 — _create_main_agent emits AgentCreatedEvent
# ----------------------------------------------------------------------


def _make_server(profile=None, agent_id="main", display_name=None,
                 session_id="20260603_120000"):
    """Construct a minimal JaatoServer with __new__ to avoid __init__'s
    cost.  Sets only the fields ``_create_main_agent`` reads.

    ``session_id`` (server 0.6.175+): the daemon session_id ``JaatoServer``
    binds at __init__ (line 315 ``self._session_id = session_id``).
    ``_create_main_agent`` now reads ``self._session_id`` and stamps
    it onto the emitted AgentCreatedEvent so cascade observers can
    correlate the event to a daemon session_id without maintaining
    their own ``agent_id → session_id`` map.
    """
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events: List[Any] = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = agent_id
    srv._main_agent_display_name = display_name
    srv._profile = profile
    srv._agents = {}
    srv._selected_agent_id = None
    srv._session_id = session_id
    return srv


def test_create_main_agent_emits_agent_created_event() -> None:
    """Pin: bootstrap-scope ``AgentCreatedEvent`` emits.  Without
    this the TUI has no agent registry and every subsequent
    agent-keyed event drops."""
    srv = _make_server(display_name="Worker")
    srv._create_main_agent()

    created_events = [
        e for e in srv._emitted_events
        if isinstance(e, AgentCreatedEvent)
    ]
    assert len(created_events) == 1, (
        f"Path I regression: expected exactly 1 AgentCreatedEvent, "
        f"got {len(created_events)}.  Without it the TUI has no "
        f"agent registry and all subsequent agent-keyed events "
        f"(ToolCallStartEvent, PermissionRequestedEvent) silently "
        f"drop."
    )


def test_emitted_event_payload_matches_hook_shape() -> None:
    """Pin: the bootstrap-scope emit produces a payload field-by-
    field equivalent to what ``ServerAgentHooks.on_agent_created``
    would have emitted pre-§7c."""
    profile = MagicMock()
    profile.name = "researcher"
    srv = _make_server(
        profile=profile,
        agent_id="custom-agent",
        display_name="Custom Display",
    )
    srv._create_main_agent()

    created_events = [
        e for e in srv._emitted_events
        if isinstance(e, AgentCreatedEvent)
    ]
    assert len(created_events) == 1
    evt = created_events[0]
    assert evt.agent_id == "custom-agent"
    assert evt.agent_name == "Custom Display"
    assert evt.agent_type == "main"
    assert evt.profile_name == "researcher"
    assert evt.parent_agent_id is None
    assert isinstance(evt.created_at, str)  # ISO timestamp


def test_pre_registered_agent_skips_emit() -> None:
    """Pin: defensive — if some other code path pre-registered the
    agent in ``server._agents`` (test fixtures, ephemeral re-init),
    ``_create_main_agent`` early-returns without re-emitting."""
    from server.core import AgentState

    srv = _make_server()
    # Pre-register.
    srv._agents["main"] = AgentState(
        agent_id="main", name="x", agent_type="main",
    )
    srv._create_main_agent()

    created_events = [
        e for e in srv._emitted_events
        if isinstance(e, AgentCreatedEvent)
    ]
    assert created_events == [], (
        "Path I regression: pre-registered agent path is supposed "
        "to skip emit (avoid duplicate events).  Re-emitting would "
        "confuse the TUI's agent-registry state machine."
    )


def test_agent_state_carries_profile_name_and_parent() -> None:
    """Pin: ``AgentState`` constructed with full field set so
    downstream consumers (toolbar, subagent dispatcher) see the
    correct metadata."""
    profile = MagicMock()
    profile.name = "researcher"
    srv = _make_server(profile=profile)
    srv._create_main_agent()

    state = srv._agents["main"]
    assert state.profile_name == "researcher"
    assert state.parent_agent_id is None
    assert state.agent_type == "main"


def test_no_profile_yields_none_profile_name() -> None:
    """Pin: when no profile, ``profile_name`` is None on both
    the AgentState and the emitted event."""
    srv = _make_server(profile=None)
    srv._create_main_agent()

    state = srv._agents["main"]
    assert state.profile_name is None

    created_events = [
        e for e in srv._emitted_events
        if isinstance(e, AgentCreatedEvent)
    ]
    assert created_events[0].profile_name is None


def test_default_display_name_falls_back_to_profile_name() -> None:
    """Pin: when ``_main_agent_display_name`` is None, fallback to
    ``profile.name`` if a profile exists, else ``"Main Agent"``."""
    profile = MagicMock()
    profile.name = "researcher"

    srv = _make_server(profile=profile, display_name=None)
    srv._create_main_agent()
    assert srv._agents["main"].name == "researcher"

    srv2 = _make_server(profile=None, display_name=None)
    srv2._create_main_agent()
    assert srv2._agents["main"].name == "Main Agent"


def test_explicit_display_name_wins_over_profile_name() -> None:
    """Pin: when ``_main_agent_display_name`` is set, it wins
    over profile.name."""
    profile = MagicMock()
    profile.name = "researcher"
    srv = _make_server(profile=profile, display_name="Override")
    srv._create_main_agent()
    assert srv._agents["main"].name == "Override"


# ----------------------------------------------------------------------
# AST pin
# ----------------------------------------------------------------------


def test_create_main_agent_source_contains_emit() -> None:
    """Pin via AST: ``_create_main_agent`` body contains a call
    to ``self.emit(AgentCreatedEvent(...))``.  Catches accidental
    revert to the pre-Path-I "hook will emit it" comment."""
    from server.core import JaatoServer

    src = inspect.getsource(JaatoServer._create_main_agent)
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    emit_with_agent_created = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Look for self.emit(...) call.
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "emit"
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and len(node.args) >= 1
            ):
                arg = node.args[0]
                # Inner call should construct AgentCreatedEvent.
                if (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Name)
                    and arg.func.id == "AgentCreatedEvent"
                ):
                    emit_with_agent_created = True
                    break
    assert emit_with_agent_created, (
        "Path I regression: _create_main_agent body lacks a "
        "self.emit(AgentCreatedEvent(...)) call.  Without it the "
        "TUI has no agent registry at bootstrap and Layer 9 is "
        "back."
    )


# ----------------------------------------------------------------------
# Server 0.6.175 / SDK 0.14.5 — session_id field on AgentCreatedEvent
# ----------------------------------------------------------------------


def test_main_agent_event_carries_session_id() -> None:
    """Pin: ``_create_main_agent`` stamps ``session_id`` onto the
    emitted AgentCreatedEvent from ``self._session_id``.  Cascade
    observers (e.g. cascade_develop.py walker) need this for per-
    stage session_id correlation without maintaining their own
    ``agent_id → session_id`` map.
    """
    srv = _make_server(session_id="20260603_120000")
    srv._create_main_agent()

    created_events = [
        e for e in srv._emitted_events
        if isinstance(e, AgentCreatedEvent)
    ]
    assert len(created_events) == 1
    assert created_events[0].session_id == "20260603_120000", (
        "session_id must be stamped on the bootstrap-scope "
        "AgentCreatedEvent for cascade observer correlation"
    )


def test_main_agent_event_session_id_empty_when_unset() -> None:
    """Pin: when ``self._session_id`` is None (defensive — shouldn't
    happen in production but the path is reachable in tests), the
    emit site falls back to empty string rather than passing None
    through (the SDK schema declares ``session_id: str = ""``,
    pydantic would coerce None to error)."""
    srv = _make_server(session_id=None)
    srv._create_main_agent()

    created_events = [
        e for e in srv._emitted_events
        if isinstance(e, AgentCreatedEvent)
    ]
    assert len(created_events) == 1
    assert created_events[0].session_id == ""


def test_sdk_schema_has_session_id_field() -> None:
    """Pin the schema contract: AgentCreatedEvent's pydantic model
    has a ``session_id`` field with str default.  Catches accidental
    schema regression that would silently drop the field via the
    base Event's ``extra='ignore'`` config."""
    fields = AgentCreatedEvent.model_fields
    assert "session_id" in fields, (
        "SDK schema regression: AgentCreatedEvent.session_id "
        "missing.  Cascade observers (cascade_develop.py walker) "
        "rely on this field — without it ``getattr(evt, 'session_id', "
        "'')`` silently returns empty and per-stage correlation "
        "breaks."
    )
    # Default must be empty string (not None) — older constructors
    # that omit the kwarg still produce a valid event.
    assert fields["session_id"].default == ""


def test_sdk_constructor_accepts_session_id_kwarg() -> None:
    """Pin: AgentCreatedEvent constructor accepts the session_id
    kwarg and stores it.  Forwards-compat guarantee for callers
    that explicitly pass it."""
    evt = AgentCreatedEvent(
        agent_id="main",
        agent_name="Worker",
        agent_type="main",
        session_id="20260603_120000",
    )
    assert evt.session_id == "20260603_120000"
    assert evt.agent_id == "main"


def test_sdk_constructor_omitted_session_id_defaults_to_empty() -> None:
    """Pin: callers that DON'T pass session_id (older code paths,
    pre-0.6.175 callers) still construct a valid event with
    session_id == ''."""
    evt = AgentCreatedEvent(
        agent_id="main",
        agent_name="Worker",
        agent_type="main",
    )
    assert evt.session_id == ""
