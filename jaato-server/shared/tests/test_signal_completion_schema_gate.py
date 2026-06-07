"""Tests for the completion-schema gate on signal_completion (2026-06-07+).

Pre-fix ``signal_completion`` was exposed regardless of whether the
profile declared a ``completion_payload_schema``.  Profiles without
a schema fell through to the legacy ``{summary: string}`` shape
with no validation — a "done" toggle with no rigorous payload
contract.  The vLLM smoke 2026-06-07 surfaced the awkwardness:
the persona referenced ``signal_completion`` but the profile had
no schema, so the tool appeared on the wire under the legacy
shape, and the smoke entered a degenerate loop.

The fix: declare a schema → get signal_completion; don't declare
one → the tool is hidden and the session terminates naturally
when the model emits text without further tool calls.

Applies uniformly to root AND subagent sessions.  The pre-existing
interactive-root filter (hide for ``client_type ∈ {TERMINAL, WEB,
CHAT}``) is a SECOND, independent gate — both fire OR-wise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_lifecycle(
    payload_schema=None,
    parent_session=None,
    presentation_context=None,
):
    """Build a ``LifecycleTools`` exposing just enough surface for
    ``_should_hide_signal_completion`` + ``get_tool_schemas`` to
    run without the rest of session.configure()."""
    from shared.lifecycle_tools import LifecycleTools

    session = SimpleNamespace(
        _parent_session=parent_session,
        _presentation_context=presentation_context,
    )
    tools = LifecycleTools.__new__(LifecycleTools)
    tools._session = session
    tools._payload_schema = payload_schema
    return tools


def _api_presentation():
    """ClientType.API presentation_context — headless / cascade."""
    from jaato_sdk.events import ClientType
    return SimpleNamespace(client_type=ClientType.API)


def _terminal_presentation():
    from jaato_sdk.events import ClientType
    return SimpleNamespace(client_type=ClientType.TERMINAL)


# ============================================================
# Gate 1 (new 2026-06-07): no schema → hide
# ============================================================


class TestSchemaGate:
    """Pin: no ``completion_payload_schema`` → ``signal_completion``
    is hidden, even in cases where the pre-existing interactive-root
    filter wouldn't have hidden it.
    """

    def test_no_schema_no_parent_api_client_hidden(self):
        """Root + API client + no schema → hidden.  This is the
        vLLM-smoke case that triggered the change."""
        tools = _make_lifecycle(
            payload_schema=None,
            parent_session=None,
            presentation_context=_api_presentation(),
        )
        assert tools._should_hide_signal_completion() is True

    def test_no_schema_no_parent_no_presentation_hidden(self):
        """Root + no presentation_context + no schema → hidden."""
        tools = _make_lifecycle(
            payload_schema=None,
            parent_session=None,
            presentation_context=None,
        )
        assert tools._should_hide_signal_completion() is True

    def test_no_schema_subagent_hidden(self):
        """Subagent + no schema → hidden.  The pre-existing
        precedent "subagents always see signal_completion" applied
        only to the interactive-root filter (gate 2); the new
        schema gate (gate 1) is unconditional — subagents that need
        signal_completion must declare a schema too."""
        parent = SimpleNamespace()
        tools = _make_lifecycle(
            payload_schema=None,
            parent_session=parent,
            presentation_context=None,
        )
        assert tools._should_hide_signal_completion() is True


# ============================================================
# Gate 2 (pre-existing): interactive-root filter
# ============================================================


class TestInteractiveRootFilter:
    """Pin: when a schema IS declared, the interactive-root filter
    behaves as it did pre-fix.  Subagents always see the tool;
    interactive roots don't; headless API roots do."""

    SCHEMA = {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "string"}},
    }

    def test_schema_subagent_exposed(self):
        """Subagent + schema → exposed (regardless of
        presentation_context)."""
        parent = SimpleNamespace()
        tools = _make_lifecycle(
            payload_schema=self.SCHEMA,
            parent_session=parent,
            presentation_context=_terminal_presentation(),
        )
        assert tools._should_hide_signal_completion() is False

    def test_schema_api_root_exposed(self):
        """Headless API root + schema → exposed.  Cascade
        entry-points, one-shot orchestrators."""
        tools = _make_lifecycle(
            payload_schema=self.SCHEMA,
            parent_session=None,
            presentation_context=_api_presentation(),
        )
        assert tools._should_hide_signal_completion() is False

    def test_schema_terminal_root_hidden(self):
        """Interactive TERMINAL root + schema → still hidden.
        Pre-existing rule: interactive sessions don't terminate via
        signal_completion."""
        tools = _make_lifecycle(
            payload_schema=self.SCHEMA,
            parent_session=None,
            presentation_context=_terminal_presentation(),
        )
        assert tools._should_hide_signal_completion() is True

    def test_schema_no_presentation_exposed(self):
        """Schema + no presentation_context → exposed (defensive
        default — unknown client_type is treated as cascade)."""
        tools = _make_lifecycle(
            payload_schema=self.SCHEMA,
            parent_session=None,
            presentation_context=None,
        )
        assert tools._should_hide_signal_completion() is False


# ============================================================
# get_tool_schemas integration
# ============================================================


class TestGetToolSchemasReflectGate:
    """Pin: ``get_tool_schemas`` respects the gate.  When hidden,
    the returned schema list does NOT contain ``signal_completion``."""

    def test_no_schema_returns_no_signal_completion(self):
        tools = _make_lifecycle(
            payload_schema=None,
            parent_session=None,
            presentation_context=_api_presentation(),
        )
        names = [s.name for s in tools.get_tool_schemas()]
        assert "signal_completion" not in names

    def test_schema_api_root_returns_signal_completion(self):
        tools = _make_lifecycle(
            payload_schema={
                "type": "object",
                "required": ["status"],
                "properties": {"status": {"type": "string"}},
            },
            parent_session=None,
            presentation_context=_api_presentation(),
        )
        names = [s.name for s in tools.get_tool_schemas()]
        assert "signal_completion" in names
