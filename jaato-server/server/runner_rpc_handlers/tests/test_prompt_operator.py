"""Tests for the daemon-side ``client.prompt_operator`` handler.

Phase 3 §3.2.1.

Three test surfaces:
1. Handler-only (PromptOperatorHandler in isolation, with a stub
   event-emit callable + manual resolve_response calls).
2. Types round-trip (PromptPayload ↔ dict, PromptResponse ↔ dict).
3. End-to-end through the bidirectional RPC channel (runner-side
   wrapper → daemon-side dispatcher → handler → resolve →
   response).
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List

import pytest

from jaato_sdk.events import PermissionRequestedEvent

from server.runner.envelope import RequestEnvelope, ResponseEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.rpc_client import RunnerRPCClient as RunnerSideClient, RunnerRPCError
from server.runner_rpc_client import RunnerRPCClient as DaemonRPCClient
from server.runner_rpc_handlers.prompt_operator import (
    PromptOperatorHandler,
    register as register_prompt_operator,
)
from server.runner_rpc_server import RunnerRPCServer

from shared.plugins.permission.types import PromptPayload, PromptResponse


# ----------------------------------------------------------------------
# Types round-trip
# ----------------------------------------------------------------------


def test_prompt_payload_round_trip() -> None:
    p = PromptPayload(
        request_id="req-1",
        session_id="sess-1",
        tool_name="cli_based_tool",
        tool_args={"command": "echo hi"},
        prompt_lines=["Run command:", "echo hi"],
        response_options=[
            {"key": "y", "label": "yes"},
            {"key": "n", "label": "no"},
        ],
        format_hint="default",
        warnings="dangerous: rm",
        warning_level="warning",
        agent_id="agent-A",
    )
    d = p.to_dict()
    back = PromptPayload.from_dict(d)
    assert back == p


def test_prompt_payload_minimal_round_trip() -> None:
    p = PromptPayload(
        request_id="r",
        session_id="s",
        tool_name="t",
    )
    back = PromptPayload.from_dict(p.to_dict())
    assert back == p


def test_prompt_response_round_trip() -> None:
    r = PromptResponse(
        request_id="r",
        response="y",
        edited_arguments={"command": "echo hello"},
        comment="ok",
    )
    back = PromptResponse.from_dict(r.to_dict())
    assert back == r


def test_prompt_response_minimal_round_trip() -> None:
    r = PromptResponse(request_id="r", response="n")
    back = PromptResponse.from_dict(r.to_dict())
    assert back == r


# ----------------------------------------------------------------------
# Handler in isolation
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_emits_event_and_awaits_response() -> None:
    """handle() emits a PermissionRequestedEvent + PermissionInputModeEvent
    (Path J cycle 12) and awaits resolve."""
    from jaato_sdk.events import PermissionInputModeEvent

    emitted: List[Any] = []

    def _emit(event: Any) -> None:
        emitted.append(event)

    handler = PromptOperatorHandler(_emit)
    payload = PromptPayload(
        request_id="r-1", session_id="s-1", tool_name="cli",
        tool_args={"command": "ls"},
    )

    # Dispatch the handle() call concurrently with a resolve.
    async def _resolve_after_a_bit() -> None:
        await asyncio.sleep(0.05)
        ok = handler.resolve_response("r-1", "y")
        assert ok is True

    asyncio.create_task(_resolve_after_a_bit())
    result = await handler.handle(payload.to_dict())

    # Path J: handler emits 2 events per ASK (was 1 pre-Path-J).
    assert len(emitted) == 2
    assert isinstance(emitted[0], PermissionRequestedEvent)
    assert isinstance(emitted[1], PermissionInputModeEvent)
    assert emitted[0].request_id == "r-1"
    assert emitted[0].tool_name == "cli"
    assert emitted[0].tool_args == {"command": "ls"}

    response = PromptResponse.from_dict(result)
    assert response.response == "y"
    assert response.request_id == "r-1"


@pytest.mark.asyncio
async def test_handler_resolve_with_edited_args() -> None:
    handler = PromptOperatorHandler(lambda e: None)

    async def _resolve() -> None:
        await asyncio.sleep(0.02)
        handler.resolve_response(
            "r-2", "e",
            edited_arguments={"command": "ls -la"},
            comment="added flags",
        )

    asyncio.create_task(_resolve())
    result = await handler.handle(
        PromptPayload(
            request_id="r-2", session_id="s", tool_name="cli", tool_args={},
        ).to_dict()
    )
    response = PromptResponse.from_dict(result)
    assert response.response == "e"
    assert response.edited_arguments == {"command": "ls -la"}
    assert response.comment == "added flags"


@pytest.mark.asyncio
async def test_handler_duplicate_request_id_rejects() -> None:
    handler = PromptOperatorHandler(lambda e: None)

    # Issue first prompt — never resolved; future stays pending.
    fut = asyncio.create_task(
        handler.handle(
            PromptPayload(
                request_id="dup", session_id="s", tool_name="t",
            ).to_dict()
        )
    )
    await asyncio.sleep(0.05)  # let the task register the request

    # Second prompt with same id raises.
    with pytest.raises(ValueError, match="duplicate request_id"):
        await handler.handle(
            PromptPayload(
                request_id="dup", session_id="s", tool_name="t",
            ).to_dict()
        )

    # Cleanup: cancel the still-pending first task.
    fut.cancel()
    try:
        await fut
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_handler_missing_request_id_rejects() -> None:
    handler = PromptOperatorHandler(lambda e: None)
    with pytest.raises(ValueError, match="request_id is required"):
        await handler.handle({"session_id": "s", "tool_name": "t"})


@pytest.mark.asyncio
async def test_handler_resolve_unknown_id_returns_false() -> None:
    handler = PromptOperatorHandler(lambda e: None)
    assert handler.resolve_response("never-was-pending", "y") is False


@pytest.mark.asyncio
async def test_handler_timeout_raises() -> None:
    """When prompt_timeout is set and no response arrives, the handler
    raises asyncio.TimeoutError (visible to the caller as an envelope
    error after dispatcher serialization)."""
    handler = PromptOperatorHandler(lambda e: None, prompt_timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await handler.handle(
            PromptPayload(
                request_id="t-1", session_id="s", tool_name="t",
            ).to_dict()
        )


@pytest.mark.asyncio
async def test_handler_shutdown_cancels_pending() -> None:
    handler = PromptOperatorHandler(lambda e: None)

    fut = asyncio.create_task(
        handler.handle(
            PromptPayload(
                request_id="sd-1", session_id="s", tool_name="t",
            ).to_dict()
        )
    )
    await asyncio.sleep(0.02)
    handler.shutdown()
    with pytest.raises(RuntimeError, match="shutting down"):
        await fut


@pytest.mark.asyncio
async def test_handler_emit_failure_cleans_pending_dict() -> None:
    """When emit_event raises, the handler clears the pending entry
    so subsequent calls don't see a duplicate-id error."""

    def _bad_emit(event: PermissionRequestedEvent) -> None:
        raise ConnectionError("client disconnected")

    handler = PromptOperatorHandler(_bad_emit)
    with pytest.raises(ConnectionError):
        await handler.handle(
            PromptPayload(
                request_id="e-1", session_id="s", tool_name="t",
            ).to_dict()
        )
    # Should be no pending entry left; re-using id works.
    assert "e-1" not in handler._pending


# ----------------------------------------------------------------------
# Concurrent prompts (same handler, different request_ids)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_prompts_resolve_independently() -> None:
    handler = PromptOperatorHandler(lambda e: None)

    async def _ask(req_id: str, response: str) -> PromptResponse:
        async def _resolve() -> None:
            # Stagger to ensure independence.
            await asyncio.sleep(0.05 if req_id == "a" else 0.02)
            handler.resolve_response(req_id, response)

        asyncio.create_task(_resolve())
        result = await handler.handle(
            PromptPayload(
                request_id=req_id, session_id="s", tool_name="t",
            ).to_dict()
        )
        return PromptResponse.from_dict(result)

    rs = await asyncio.gather(_ask("a", "y"), _ask("b", "n"))
    assert {r.request_id: r.response for r in rs} == {"a": "y", "b": "n"}


# ----------------------------------------------------------------------
# End-to-end through the bidirectional RPC channel
# ----------------------------------------------------------------------


class _RPCPair:
    def __init__(self, runner_rpc, runner_thread, daemon_client):
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair(rpc_server: RunnerRPCServer) -> _RPCPair:
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _no_executor(name, args):
        return False, {"error": "no executor"}

    runner_rpc = RunnerRPC(runner_sock, _no_executor)
    thread = threading.Thread(
        target=runner_rpc.serve, name="rpc-serve-test", daemon=True,
    )
    thread.start()

    daemon_client = DaemonRPCClient(
        daemon_sock, runner_pid=0, rpc_server=rpc_server,
    )
    await daemon_client.start()
    return _RPCPair(runner_rpc, thread, daemon_client)


async def _teardown(pair: _RPCPair) -> None:
    pair.runner_rpc.shutdown()
    pair.runner_thread.join(timeout=2)
    pair.daemon_client._closed = True
    if pair.daemon_client._writer is not None:
        try:
            pair.daemon_client._writer.close()
            await pair.daemon_client._writer.wait_closed()
        except Exception:
            pass
    if pair.daemon_client._read_task is not None:
        pair.daemon_client._read_task.cancel()
        try:
            await pair.daemon_client._read_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_e2e_prompt_operator_round_trip() -> None:
    """Runner-side wrapper → daemon-side dispatcher → handler →
    response → runner-side wrapper returns PromptResponse."""
    emitted: List[PermissionRequestedEvent] = []
    handler = PromptOperatorHandler(lambda e: emitted.append(e))

    rpc_server = RunnerRPCServer()
    register_prompt_operator(rpc_server, handler)

    pair = await _make_pair(rpc_server)
    try:
        # Resolve the prompt as soon as the event fires.
        async def _watch_and_resolve() -> None:
            for _ in range(50):  # poll up to 1s
                if emitted:
                    handler.resolve_response(emitted[-1].request_id, "y")
                    return
                await asyncio.sleep(0.02)

        watch_task = asyncio.create_task(_watch_and_resolve())

        # Run the synchronous outgoing call on a worker thread.
        runner_wrapper = RunnerSideClient(pair.runner_rpc)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: runner_wrapper.prompt_operator(
                PromptPayload(
                    request_id="e2e-1",
                    session_id="s",
                    tool_name="cli_based_tool",
                    tool_args={"command": "echo hi"},
                ),
                timeout=2,
            ),
        )

        await watch_task
        assert response.request_id == "e2e-1"
        assert response.response == "y"
        # Path J (cycle 12): handler emits 2 events per ASK
        # (PermissionRequestedEvent + PermissionInputModeEvent).
        assert len(emitted) == 2
        assert emitted[0].tool_name == "cli_based_tool"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_unknown_method_raises_runner_rpc_error() -> None:
    """The runner-side wrapper translates ``ok=False`` envelopes into
    :class:`RunnerRPCError`."""
    pair = await _make_pair(RunnerRPCServer())
    try:
        wrapper = RunnerSideClient(pair.runner_rpc)
        loop = asyncio.get_running_loop()
        with pytest.raises(RunnerRPCError, match="UnknownMethod"):
            await loop.run_in_executor(
                None,
                lambda: wrapper.prompt_operator(
                    PromptPayload(
                        request_id="e2e-x", session_id="s", tool_name="t",
                    ),
                    timeout=2,
                ),
            )
    finally:
        await _teardown(pair)
