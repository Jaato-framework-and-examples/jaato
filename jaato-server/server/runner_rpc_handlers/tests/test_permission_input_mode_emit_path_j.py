"""Regression-pin tests for Path J — PermissionInputModeEvent companion emit.

**Root cause** (cycle-12 worker-corrected diagnosis):

The TUI listens for ``PermissionInputModeEvent``, NOT
``PermissionRequestedEvent``.  Grep:

  rich_client.py:    1 ref to PermissionInputModeEvent, 0 to Requested
  headless_mode.py:  1 ref to PermissionInputModeEvent, 0 to Requested

Pre-§7c daemon flow:
  Runner ASK → daemon permission_plugin.callback → emits BOTH
  PermissionRequestedEvent + PermissionInputModeEvent.

Post-§7c RPC flow:
  Runner ASK → PromptOperatorHandler.handle() → emitted ONLY
  PermissionRequestedEvent.  The companion control event
  (PermissionInputModeEvent) — which the TUI listens for — was
  never emitted, so the TUI's input mode never switched.

Same architectural pattern as Path I (cycle 11): a daemon-side
emit path that pre-§7c was driven in-process by a daemon plugin
became dead post-§7c.

**Path J fix** (Option A — mirror Path I):

PromptOperatorHandler.handle() emits BOTH events.  Order matters:
PermissionRequestedEvent first (content), then
PermissionInputModeEvent (control signal that switches TUI
input mode).

Tests pin:
- handle() emits BOTH event types per ASK (not one)
- Order: Requested first, InputMode second (TUI render then
  switch-mode)
- Both events carry the SAME request_id for TUI correlation
- Both carry agent_id, tool_name from payload
- response_options propagate to InputMode event
- tool_args propagate to InputMode event (when present)
- call_id is None (Path J.A backlog: PromptPayload doesn't carry it)
- editable_metadata is None (Path J.B backlog: requires schema
  lookup)
- AST: handle source references both event class names
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from typing import Any, List

import pytest

from jaato_sdk.events import (
    PermissionInputModeEvent,
    PermissionRequestedEvent,
)
from server.runner_rpc_handlers.prompt_operator import PromptOperatorHandler
from shared.plugins.permission.types import PromptPayload


# ----------------------------------------------------------------------
# J — handle() emits BOTH event types per ASK
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_emits_both_events_per_ask() -> None:
    """Pin: each ASK emits BOTH PermissionRequestedEvent AND
    PermissionInputModeEvent.  Pre-Path-J only the former was
    emitted; the TUI listens for the latter, so input mode never
    switched."""
    emitted: List[Any] = []
    handler = PromptOperatorHandler(emitted.append)

    payload = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
        tool_args={"command": "ls"},
        response_options=[{"key": "y", "label": "yes", "action": "ALLOW"}],
    )

    async def _resolve() -> None:
        await asyncio.sleep(0.02)
        handler.resolve_response("r-1", "y")

    asyncio.create_task(_resolve())
    await handler.handle(payload.to_dict())

    assert len(emitted) == 2, (
        f"Path J regression: expected 2 events per ASK; got "
        f"{len(emitted)}.  Missing PermissionInputModeEvent → TUI "
        f"never switches input mode → user's 'y' keypress is "
        f"queued as chat instead of being routed to the ASK."
    )
    req_events = [e for e in emitted if isinstance(e, PermissionRequestedEvent)]
    mode_events = [e for e in emitted if isinstance(e, PermissionInputModeEvent)]
    assert len(req_events) == 1
    assert len(mode_events) == 1


@pytest.mark.asyncio
async def test_event_order_requested_before_input_mode() -> None:
    """Pin: PermissionRequestedEvent emits BEFORE
    PermissionInputModeEvent.  Order matters — TUI renders the
    prompt content first, then switches input mode."""
    emitted: List[Any] = []
    handler = PromptOperatorHandler(emitted.append)

    payload = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
    )

    async def _resolve() -> None:
        await asyncio.sleep(0.02)
        handler.resolve_response("r-1", "y")

    asyncio.create_task(_resolve())
    await handler.handle(payload.to_dict())

    assert isinstance(emitted[0], PermissionRequestedEvent), (
        "Path J regression: first emit must be "
        "PermissionRequestedEvent (content event).  Reversed order "
        "would make TUI switch input mode before rendering the "
        "prompt — user sees Answer> with no prompt to answer."
    )
    assert isinstance(emitted[1], PermissionInputModeEvent)


@pytest.mark.asyncio
async def test_both_events_share_request_id() -> None:
    """Pin: both events carry the SAME request_id so the TUI can
    correlate the input-mode signal to the prompt content."""
    emitted: List[Any] = []
    handler = PromptOperatorHandler(emitted.append)

    payload = PromptPayload(
        request_id="unique-req-42",
        session_id="s-1", tool_name="cli",
    )

    async def _resolve() -> None:
        await asyncio.sleep(0.02)
        handler.resolve_response("unique-req-42", "y")

    asyncio.create_task(_resolve())
    await handler.handle(payload.to_dict())

    assert emitted[0].request_id == "unique-req-42"
    assert emitted[1].request_id == "unique-req-42"


@pytest.mark.asyncio
async def test_input_mode_event_carries_payload_fields() -> None:
    """Pin: PermissionInputModeEvent fields are populated from the
    PromptPayload."""
    emitted: List[Any] = []
    handler = PromptOperatorHandler(emitted.append)

    payload = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli_based_tool",
        agent_id="main",
        tool_args={"command": "echo hello"},
        response_options=[
            {"key": "y", "label": "yes", "action": "ALLOW"},
            {"key": "n", "label": "no", "action": "DENY"},
        ],
    )

    async def _resolve() -> None:
        await asyncio.sleep(0.02)
        handler.resolve_response("r-1", "y")

    asyncio.create_task(_resolve())
    await handler.handle(payload.to_dict())

    mode_evt = emitted[1]
    assert mode_evt.agent_id == "main"
    assert mode_evt.tool_name == "cli_based_tool"
    assert len(mode_evt.response_options) == 2
    assert mode_evt.tool_args == {"command": "echo hello"}


@pytest.mark.asyncio
async def test_input_mode_call_id_propagates_from_payload() -> None:
    """Pin: call_id propagates from PromptPayload through to the
    emitted PermissionInputModeEvent.

    Phase 4 §4.1 (J.A) closure.  Pre-§4.1 this pin asserted
    ``call_id is None`` because PromptPayload didn't carry the
    field.  Post-§4.1 the runner-side permission plugin populates
    ``context['call_id']``, the runner-RPC channel reads it into
    ``PromptPayload.call_id``, and PromptOperatorHandler threads
    it through to ``PermissionInputModeEvent.call_id`` for TUI
    per-tool-block correlation.
    """
    emitted: List[Any] = []
    handler = PromptOperatorHandler(emitted.append)

    payload = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
        call_id="tool-call-abc-123",
    )

    async def _resolve() -> None:
        await asyncio.sleep(0.02)
        handler.resolve_response("r-1", "y")

    asyncio.create_task(_resolve())
    await handler.handle(payload.to_dict())

    assert emitted[1].call_id == "tool-call-abc-123", (
        "Phase 4 §4.1 regression: call_id didn't propagate from "
        "PromptPayload to PermissionInputModeEvent.  TUI cannot "
        "correlate the permission prompt to its per-tool-block "
        "widget when parallel tools are in flight."
    )


@pytest.mark.asyncio
async def test_input_mode_call_id_defaults_to_none_when_omitted() -> None:
    """Pin: when the runner doesn't populate call_id (legacy/
    forward-compat scenarios), PermissionInputModeEvent.call_id
    is None — preserves the Path J.A backward-compat contract."""
    emitted: List[Any] = []
    handler = PromptOperatorHandler(emitted.append)

    payload = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
    )

    async def _resolve() -> None:
        await asyncio.sleep(0.02)
        handler.resolve_response("r-1", "y")

    asyncio.create_task(_resolve())
    await handler.handle(payload.to_dict())

    assert emitted[1].call_id is None


@pytest.mark.asyncio
async def test_input_mode_editable_metadata_is_none_path_j_b() -> None:
    """Pin: editable_metadata is None — Path J.B backlog
    (requires permission_plugin schema lookup; follow-up if
    editing flow surfaces the need)."""
    emitted: List[Any] = []
    handler = PromptOperatorHandler(emitted.append)

    payload = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
    )

    async def _resolve() -> None:
        await asyncio.sleep(0.02)
        handler.resolve_response("r-1", "y")

    asyncio.create_task(_resolve())
    await handler.handle(payload.to_dict())

    assert emitted[1].editable_metadata is None


@pytest.mark.asyncio
async def test_emit_failure_on_second_event_cleans_pending() -> None:
    """Pin: defensive — if the second emit (InputModeEvent) raises,
    the pending future is cleaned up (no leak)."""
    call_count = [0]

    def _flaky_emit(event: Any) -> None:
        call_count[0] += 1
        if isinstance(event, PermissionInputModeEvent):
            raise RuntimeError("simulated transport failure")

    handler = PromptOperatorHandler(_flaky_emit)
    payload = PromptPayload(
        request_id="r-flaky", session_id="s-1", tool_name="cli",
    )

    with pytest.raises(RuntimeError, match="simulated transport"):
        await handler.handle(payload.to_dict())

    # Pending dict must be clean.
    assert "r-flaky" not in handler._pending, (
        "Path J regression: handler leaked pending future when "
        "the second event emit raised.  The try/except cleanup "
        "must cover BOTH emit calls."
    )


# ----------------------------------------------------------------------
# AST pin
# ----------------------------------------------------------------------


def test_handle_source_references_both_event_classes() -> None:
    """Pin via AST: ``handle`` body constructs both
    ``PermissionRequestedEvent`` AND ``PermissionInputModeEvent``."""
    src = inspect.getsource(PromptOperatorHandler.handle)
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    found_requested = False
    found_input_mode = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "PermissionRequestedEvent":
                found_requested = True
            elif node.func.id == "PermissionInputModeEvent":
                found_input_mode = True

    assert found_requested, (
        "Path J regression: handle() lost PermissionRequestedEvent "
        "construction."
    )
    assert found_input_mode, (
        "Path J regression: handle() lacks PermissionInputModeEvent "
        "construction.  TUI's permission mode won't engage; cycle-10 "
        "rendering gap returns."
    )
