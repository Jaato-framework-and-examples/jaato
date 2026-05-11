"""Regression-pin tests for §7c Step 7.3 — ``respond_to_permission``
dual-path routing.

Per the §7c Step 7 disposition audit (commit 285449d0), pre-§7.3
``respond_to_permission`` pushed responses to
``_channel_input_queue`` — orphaned for runner-fired ASKs (the
runner-side permission plugin's ASKs go through the
``client.prompt_operator`` RPC, not through any queue the daemon
reads).

Step 7.3 reroutes ``respond_to_permission`` to try TWO paths:

1. **Runner-fired ASK (post-seat-flip)**: call
   ``PromptOperatorHandler.resolve_response(request_id, response,
   edited_arguments=...)`` — returns True if a future was pending.
2. **Daemon-fired ASK (legacy fallback)**: check
   ``_pending_permission_request_id`` and push to
   ``_channel_input_queue``.

Tests pin:

- Runner-fired ASKs (handler has pending future) resolve via Path 1
- Daemon-fired ASKs (handler returns False; legacy
  ``_pending_permission_request_id`` matches) resolve via Path 2
- Unknown requests (neither path resolves) emit an
  ``ErrorEvent``
- ``edited_arguments`` propagate through Path 1 (handler call)
- ``edited_arguments`` propagate through Path 2 (legacy
  ``_pending_edited_arguments`` stash)
- E2E ASK round-trip: runner sends payload → handler emits
  ``PermissionRequestedEvent`` → daemon calls
  ``respond_to_permission`` → runner's outgoing_call awaiter
  resolves with the response
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from jaato_sdk.events import ErrorEvent, PermissionRequestedEvent
from server.core import JaatoServer
from server.runner_rpc_handlers.prompt_operator import (
    PromptOperatorHandler,
)
from server.runner_rpc_server import RunnerRPCServer
from shared.plugins.permission.types import PromptPayload, PromptResponse


class _FakeRPC:
    def __init__(self) -> None:
        self.rpc_server = RunnerRPCServer()


def _make_server(*, with_handler: bool = True) -> JaatoServer:
    """Minimal JaatoServer for Step 7.3 routing tests."""
    srv = JaatoServer.__new__(JaatoServer)
    srv._on_event = lambda e: None
    srv._emitted: List[Any] = []
    srv.emit = lambda e: srv._emitted.append(e)  # type: ignore[method-assign]
    srv.registry = None
    srv.permission_plugin = None
    srv._runner_rpc = None
    srv._spawned_runner = None
    srv._prompt_operator_handler = None

    import queue
    srv._channel_input_queue = queue.Queue()
    srv._pending_permission_request_id: Optional[str] = None
    srv._pending_edited_arguments: Optional[Dict[str, Any]] = None

    if with_handler:
        srv._prompt_operator_handler = PromptOperatorHandler(
            emit_event=srv.emit,
        )
    return srv


def _make_payload(request_id: str = "req-1") -> PromptPayload:
    return PromptPayload(
        request_id=request_id,
        session_id="sess-1",
        agent_id="main",
        tool_name="write_file",
        tool_args={"path": "/tmp/x"},
        response_options=[{"key": "y", "label": "yes"}],
    )


# ----------------------------------------------------------------------
# Path 1 — runner-fired ASK resolves through handler
# ----------------------------------------------------------------------


def test_path_1_runner_fired_ask_resolves_through_handler() -> None:
    """When the handler has a pending future for ``request_id``,
    ``respond_to_permission`` resolves it via Path 1 — no
    queue push."""
    srv = _make_server()
    handler = srv._prompt_operator_handler

    async def _drive() -> Any:
        # Spawn the handler's handle() to register a pending future.
        ask_task = asyncio.create_task(
            handler.handle(_make_payload("req-7-3-1").to_dict()),
        )
        await asyncio.sleep(0)  # yield so the future is registered
        # Resolve via respond_to_permission (Path 1).
        srv.respond_to_permission("req-7-3-1", "y")
        result = await asyncio.wait_for(ask_task, timeout=1.0)
        return result

    result = asyncio.run(_drive())
    assert result["response"] == "y"
    # Path 1 short-circuits before reaching the queue.
    assert srv._channel_input_queue.empty()


def test_path_1_propagates_edited_arguments() -> None:
    srv = _make_server()
    handler = srv._prompt_operator_handler

    async def _drive() -> Any:
        ask_task = asyncio.create_task(
            handler.handle(_make_payload("req-7-3-2").to_dict()),
        )
        await asyncio.sleep(0)
        srv.respond_to_permission(
            "req-7-3-2", "e",
            edited_arguments={"path": "/tmp/y"},
        )
        return await asyncio.wait_for(ask_task, timeout=1.0)

    result = asyncio.run(_drive())
    assert result["response"] == "e"
    assert result["edited_arguments"] == {"path": "/tmp/y"}


# ----------------------------------------------------------------------
# Path 2 — daemon-fired ASK falls through to legacy queue
# ----------------------------------------------------------------------


def test_path_2_daemon_fired_ask_falls_through_to_queue() -> None:
    """Handler has no pending future; legacy
    ``_pending_permission_request_id`` matches → push to queue."""
    srv = _make_server()
    srv._pending_permission_request_id = "req-7-3-3"

    srv.respond_to_permission("req-7-3-3", "n")

    # Queue got the response.
    assert srv._channel_input_queue.get_nowait() == "n"


def test_path_2_propagates_edited_arguments_via_legacy_stash() -> None:
    srv = _make_server()
    srv._pending_permission_request_id = "req-7-3-4"

    srv.respond_to_permission(
        "req-7-3-4", "e",
        edited_arguments={"path": "/tmp/z"},
    )

    assert srv._pending_edited_arguments == {"path": "/tmp/z"}
    assert srv._channel_input_queue.get_nowait() == "e"


def test_path_2_works_when_handler_is_none() -> None:
    """Sessions that fall back to in-process execution have
    ``_prompt_operator_handler = None``.  Path 2 must still work."""
    srv = _make_server(with_handler=False)
    srv._pending_permission_request_id = "req-7-3-5"

    srv.respond_to_permission("req-7-3-5", "y")

    assert srv._channel_input_queue.get_nowait() == "y"


# ----------------------------------------------------------------------
# Unknown request — neither path resolves
# ----------------------------------------------------------------------


def test_unknown_request_emits_error_event() -> None:
    """Neither handler future nor legacy pending id matches —
    emit ErrorEvent (preserves pre-§7.3 contract for stale
    responses)."""
    srv = _make_server()
    # No pending future, no _pending_permission_request_id match.

    srv.respond_to_permission("req-unknown", "y")

    # Queue must remain empty.
    assert srv._channel_input_queue.empty()
    # ErrorEvent emitted.
    assert len(srv._emitted) == 1
    assert isinstance(srv._emitted[0], ErrorEvent)
    assert "Unknown permission request" in srv._emitted[0].error


def test_unknown_request_when_no_handler_attached() -> None:
    """Same unknown-request behavior when no handler is attached
    (sessions without runner-RPC)."""
    srv = _make_server(with_handler=False)

    srv.respond_to_permission("req-unknown", "y")

    assert len(srv._emitted) == 1
    assert isinstance(srv._emitted[0], ErrorEvent)


# ----------------------------------------------------------------------
# Routing-priority pin: Path 1 wins over Path 2 when both could resolve
# ----------------------------------------------------------------------


def test_handler_path_wins_over_legacy_path_when_both_match() -> None:
    """When the handler has a pending future AND
    ``_pending_permission_request_id`` happens to match the same
    request_id (unusual but possible during the transitional
    window), Path 1 wins — runner-fired ASKs are the canonical
    post-seat-flip mode."""
    srv = _make_server()
    handler = srv._prompt_operator_handler
    srv._pending_permission_request_id = "req-collision"

    async def _drive() -> Any:
        ask_task = asyncio.create_task(
            handler.handle(_make_payload("req-collision").to_dict()),
        )
        await asyncio.sleep(0)
        srv.respond_to_permission("req-collision", "a")
        return await asyncio.wait_for(ask_task, timeout=1.0)

    result = asyncio.run(_drive())
    assert result["response"] == "a"
    # Path 1 won — queue stays empty.
    assert srv._channel_input_queue.empty()


# ----------------------------------------------------------------------
# E2E pin: full ASK round-trip via the handler
# ----------------------------------------------------------------------


def test_e2e_full_ask_roundtrip_through_handler_and_responder() -> None:
    """Pin the full Step 7 round-trip end-to-end:

    1. Runner-side payload arrives at the handler (simulated
       by calling ``handler.handle(payload)`` directly).
    2. Handler emits ``PermissionRequestedEvent`` on the
       daemon's event channel.
    3. Daemon-side ``respond_to_permission(request_id, "y")``
       routes through Path 1 → resolves the handler's future.
    4. The handler returns a ``PromptResponse`` dict that the
       runner's ``RunnerRPCClient.prompt_operator`` would
       deserialize back into a ``PromptResponse`` instance.

    This is the daemon-half of the round-trip.  Step 7.2's
    ``registry.runner_rpc_client`` attachment + the existing
    permission-plugin RunnerRPCChannel complete the runner-half.
    """
    srv = _make_server()
    handler = srv._prompt_operator_handler

    async def _run() -> Dict[str, Any]:
        # Step 1: runner-side payload.
        payload = _make_payload("req-e2e")

        # Step 2: handler.handle dispatched in a task; it'll emit
        # an event + register a future.
        ask_task = asyncio.create_task(
            handler.handle(payload.to_dict()),
        )

        # Yield + verify the event landed on the daemon's channel.
        await asyncio.sleep(0)
        # Path J (cycle 12): handler emits 2 events per ASK
        # (PermissionRequestedEvent + PermissionInputModeEvent).
        assert len(srv._emitted) == 2
        event = srv._emitted[0]
        assert isinstance(event, PermissionRequestedEvent)
        assert event.request_id == "req-e2e"
        assert event.agent_id == "main"
        assert event.tool_name == "write_file"

        # Step 3: daemon-side responder routes through Path 1.
        srv.respond_to_permission("req-e2e", "y")

        # Step 4: the handler's future resolves; result dict
        # comes back ready for ``PromptResponse.from_dict``.
        result = await asyncio.wait_for(ask_task, timeout=1.0)

        # Verify the round-trip preserves the request_id contract.
        assert result["request_id"] == "req-e2e"
        assert result["response"] == "y"

        # And the response deserializes back to PromptResponse —
        # the runner-side ``RunnerRPCClient.prompt_operator`` does
        # exactly this step.
        prompt_response = PromptResponse.from_dict(result)
        assert prompt_response.request_id == "req-e2e"
        assert prompt_response.response == "y"
        return result

    asyncio.run(_run())
