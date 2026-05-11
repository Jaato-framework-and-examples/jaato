"""Regression-pin tests for §7c Step 7.1 — daemon-side
PromptOperatorHandler wiring.

Per the §7c Step 7 disposition audit (commit 285449d0), the
gap was: ``PromptOperatorHandler`` is defined but never
instantiated in production code.  Step 7.1 closes that gap by
constructing + registering the handler inside
:meth:`JaatoServer.set_runner_rpc`.

Pre-§7.1 the §7d disposition audit's finding #6 also flagged
"bidirectional dispatch needs plumbing" — Step 7.1's
investigation revealed this was incorrect: the daemon-side
``RunnerRPCClient`` already lazy-constructs a ``RunnerRPCServer``
in ``__init__`` (line 152) and the read loop already dispatches
``KIND_REQUEST`` frames at line 355.  The only missing piece
was the handler instantiation + registration.

Tests pin:

- ``JaatoServer.__init__`` declares the ``_prompt_operator_handler``
  field (defaults to None)
- ``set_runner_rpc(rpc_client, spawned)`` with non-None rpc_client
  constructs a handler bound to ``self.emit``
- ``set_runner_rpc`` registers the handler under
  ``"client.prompt_operator"`` on the rpc_client's rpc_server
- ``set_runner_rpc(None, None)`` resets the handler to None
- ``shutdown()`` calls the handler's ``shutdown()``
- ``shutdown()`` is forward-compat with tests that bypass
  ``__init__`` via ``JaatoServer.__new__`` (uses ``getattr``)
- The handler's ``emit_event`` callback emits via
  ``JaatoServer.emit`` (the daemon's event channel)
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional
from unittest.mock import MagicMock

from jaato_sdk.events import PermissionRequestedEvent
from server.core import JaatoServer
from server.runner_rpc_handlers.prompt_operator import (
    PromptOperatorHandler,
)
from server.runner_rpc_server import RunnerRPCServer
from shared.plugins.permission.types import PromptPayload


class _FakeRPC:
    """Stand-in for RunnerRPCClient.  Only the ``rpc_server`` accessor
    is exercised by ``set_runner_rpc``; we use a real
    ``RunnerRPCServer`` so the registration can be inspected."""

    def __init__(self) -> None:
        self.rpc_server = RunnerRPCServer()


def _make_server() -> JaatoServer:
    """Build a minimal JaatoServer for unit-testing 7.1 wiring."""
    srv = JaatoServer.__new__(JaatoServer)
    srv._on_event = lambda e: None
    srv._emitted: list = []
    srv.emit = lambda e: srv._emitted.append(e)  # type: ignore[method-assign]
    srv.registry = None
    srv.permission_plugin = None
    srv._runner_rpc = None
    srv._spawned_runner = None
    srv._prompt_operator_handler = None
    return srv


# ----------------------------------------------------------------------
# Field-declaration pin
# ----------------------------------------------------------------------


def test_init_declares_prompt_operator_handler_field() -> None:
    """Pin ``JaatoServer.__init__`` declares
    ``_prompt_operator_handler`` (defaults to None)."""
    import inspect
    src = inspect.getsource(JaatoServer.__init__)
    assert "_prompt_operator_handler" in src, (
        "§7c Step 7.1 regression: ``_prompt_operator_handler`` field "
        "declaration missing from JaatoServer.__init__."
    )


# ----------------------------------------------------------------------
# set_runner_rpc instantiation + registration pins
# ----------------------------------------------------------------------


def test_set_runner_rpc_instantiates_handler() -> None:
    """With a non-None rpc_client, ``set_runner_rpc`` constructs
    a ``PromptOperatorHandler`` and stores it on
    ``self._prompt_operator_handler``."""
    srv = _make_server()
    rpc = _FakeRPC()
    srv.set_runner_rpc(rpc, MagicMock())

    assert srv._prompt_operator_handler is not None
    assert isinstance(
        srv._prompt_operator_handler, PromptOperatorHandler,
    )


def test_set_runner_rpc_registers_handler_under_client_prompt_operator() -> None:
    """The handler's ``handle`` method is registered on the
    rpc_server's dispatch table under
    ``"client.prompt_operator"``."""
    srv = _make_server()
    rpc = _FakeRPC()
    srv.set_runner_rpc(rpc, MagicMock())

    assert rpc.rpc_server.has_handler("client.prompt_operator")


def test_set_runner_rpc_none_clears_handler() -> None:
    """When ``rpc_client=None`` (session falls back to in-process
    execution), the handler stays None."""
    srv = _make_server()
    srv.set_runner_rpc(None, None)
    assert srv._prompt_operator_handler is None


def test_set_runner_rpc_handler_bound_to_server_emit() -> None:
    """The handler's ``emit_event`` callback fires
    ``JaatoServer.emit`` so the permission-requested event
    flows to the daemon's event channel."""
    srv = _make_server()
    rpc = _FakeRPC()
    srv.set_runner_rpc(rpc, MagicMock())

    handler = srv._prompt_operator_handler
    assert handler is not None
    # ``_emit_event`` is the constructor-bound callable.
    event = PermissionRequestedEvent(
        agent_id="main",
        request_id="req-1",
        tool_name="x",
        tool_args={},
        response_options=[{"key": "y", "label": "yes"}],
    )
    handler._emit_event(event)
    assert len(srv._emitted) == 1
    assert srv._emitted[0] is event


# ----------------------------------------------------------------------
# shutdown teardown pins
# ----------------------------------------------------------------------


def test_shutdown_calls_handler_shutdown() -> None:
    """``JaatoServer.shutdown()`` cancels in-flight prompts via
    the handler's ``shutdown()``."""
    srv = _make_server()
    rpc = _FakeRPC()
    srv.set_runner_rpc(rpc, MagicMock())

    handler = srv._prompt_operator_handler
    # Force a graceful shutdown path by avoiding the rest of
    # shutdown's transport-close logic (which the unit test
    # fixture doesn't fully wire).  Stub the parts we don't
    # exercise; the handler-shutdown branch is what we pin.
    srv._runner_rpc = None
    srv._spawned_runner = None

    srv.shutdown()
    assert handler._closed is True
    assert srv._prompt_operator_handler is None


def test_shutdown_forward_compat_with_bypassed_init() -> None:
    """Tests that bypass ``__init__`` via ``JaatoServer.__new__``
    must still get a clean ``shutdown()`` — the handler-teardown
    block uses ``getattr`` so missing attribute is fine."""
    srv = JaatoServer.__new__(JaatoServer)
    srv.registry = None
    srv.permission_plugin = None
    srv._runner_rpc = None
    srv._spawned_runner = None
    # Deliberately do NOT set _prompt_operator_handler.
    srv.shutdown()  # Must not raise.


# ----------------------------------------------------------------------
# Integration pin: ASK round-trip routes through the handler
# ----------------------------------------------------------------------


def test_handler_routes_ask_through_emit_and_resolves() -> None:
    """End-to-end pin (single-process, async): a runner-fired
    ASK payload routes through the handler's ``handle()``,
    fires ``JaatoServer.emit(PermissionRequestedEvent)``, and
    the matching ``resolve_response(request_id, ...)`` returns
    the response."""
    srv = _make_server()
    rpc = _FakeRPC()
    srv.set_runner_rpc(rpc, MagicMock())
    handler = srv._prompt_operator_handler
    assert handler is not None

    # Simulated runner ASK payload.
    payload = PromptPayload(
        request_id="req-42",
        session_id="sess-1",
        agent_id="main",
        tool_name="write_file",
        tool_args={"path": "/tmp/x"},
        response_options=[
            {"key": "y", "label": "yes"},
            {"key": "n", "label": "no"},
        ],
    )

    async def _run() -> dict:
        # Schedule the handler's awaitable + the resolution.
        ask_task = asyncio.create_task(handler.handle(payload.to_dict()))
        # Yield once so the handler emits + registers its future.
        await asyncio.sleep(0)
        assert len(srv._emitted) == 1
        emitted_event = srv._emitted[0]
        assert isinstance(emitted_event, PermissionRequestedEvent)
        assert emitted_event.request_id == "req-42"
        # Resolve from the daemon-side response path.
        ok = handler.resolve_response("req-42", "y")
        assert ok is True
        result = await asyncio.wait_for(ask_task, timeout=1.0)
        return result

    result = asyncio.run(_run())
    assert result["response"] == "y"
    assert result["request_id"] == "req-42"


# ----------------------------------------------------------------------
# Wiring lineage: confirm the bidirectional dispatch IS pre-wired
# (correcting Step 7 audit finding #6)
# ----------------------------------------------------------------------


def test_runner_rpc_client_lazy_constructs_rpc_server() -> None:
    """Pin: ``RunnerRPCClient.__init__`` lazy-constructs a
    ``RunnerRPCServer`` when none is passed (correcting the
    Step 7 audit's finding #6 — the bidirectional dispatch
    infrastructure was already wired; only the handler
    instantiation was missing)."""
    import socket
    from server.runner_rpc_client import RunnerRPCClient

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Construct without an explicit rpc_server.  Use a fresh
        # event loop to avoid clobbering pytest's.
        loop = asyncio.new_event_loop()
        try:
            client = RunnerRPCClient(a, runner_pid=0, loop=loop)
            assert client.rpc_server is not None
            assert isinstance(client.rpc_server, RunnerRPCServer)
        finally:
            loop.close()
    finally:
        a.close()
        b.close()
