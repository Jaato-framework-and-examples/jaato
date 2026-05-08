"""Tests for the Phase 3 §3.2 bidirectional RPC surface.

Phase 2 only had daemon→runner direction (RunnerRPCClient sends
requests; RunnerRPC dispatches them).  Phase 3 task 3.2 adds the
inverse direction so runner-side plugins can call back into the
daemon for capabilities only the daemon owns
(``client.prompt_operator``, ``apparmor.add_reference_fragment``,
``telemetry.publish``).

These tests exercise the new surface end-to-end:
- Runner-side ``RunnerRPC.outgoing_call(method, args)`` →
  Daemon-side ``RunnerRPCClient`` read-loop catches the
  ``request`` frame → dispatches to ``RunnerRPCServer`` → writes
  back the ``response`` frame → runner's read-loop resolves the
  outgoing future.

The fixture sets up a real socketpair with both sides running
on a single asyncio event loop (the daemon side is async; the
runner side runs its serve loop in a worker thread).
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List

import pytest

from server.runner.envelope import RequestEnvelope, ResponseEnvelope
from server.runner.rpc import RunnerRPC
from server.runner_rpc_client import RunnerRPCClient
from server.runner_rpc_server import RunnerRPCServer


# ----------------------------------------------------------------------
# Fixture: real socketpair, runner serve loop in a thread, daemon
# RunnerRPCClient on the asyncio loop.
# ----------------------------------------------------------------------


class _Pair:
    """Holds both ends of the bidirectional channel + lifecycle hooks."""

    def __init__(
        self,
        runner_rpc: RunnerRPC,
        runner_thread: threading.Thread,
        daemon_client: RunnerRPCClient,
    ) -> None:
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair(rpc_server: RunnerRPCServer) -> _Pair:
    """Build a real bidirectional channel.

    The runner side runs its serve loop in a worker thread; the
    daemon side adopts its socket onto the asyncio loop.
    """
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    # Default executor — we don't exercise the daemon→runner path here
    # so the executor can be a no-op.
    def _default_executor(name: str, args: Dict[str, Any]):
        return False, {"error": "no executor for this test"}

    runner_rpc = RunnerRPC(runner_sock, _default_executor)
    thread = threading.Thread(
        target=runner_rpc.serve, name="runner-serve-test", daemon=True,
    )
    thread.start()

    daemon_client = RunnerRPCClient(
        daemon_sock,
        runner_pid=0,  # no real subprocess in this test
        rpc_server=rpc_server,
    )
    await daemon_client.start()
    return _Pair(runner_rpc, thread, daemon_client)


async def _teardown_pair(pair: _Pair) -> None:
    pair.runner_rpc.shutdown()
    pair.runner_thread.join(timeout=2)
    # The daemon-side close path expects a runner pid; pass None to
    # short-circuit the waitpid ladder for tests.
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


# ----------------------------------------------------------------------
# Happy-path round-trip
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outgoing_call_round_trip() -> None:
    """Runner calls into the daemon; daemon's handler returns; runner
    sees the result envelope."""
    server = RunnerRPCServer()

    async def _ping(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"echo": args.get("payload"), "from": "daemon"}

    server.register("ping", _ping)
    pair = await _make_pair(server)
    try:
        # Run the synchronous outgoing_call on a thread (mimics a
        # real plugin's worker-thread execution context).
        loop = asyncio.get_running_loop()
        env = await loop.run_in_executor(
            None,
            lambda: pair.runner_rpc.outgoing_call(
                "ping", {"payload": "hello"}, timeout=2,
            ),
        )
        assert env.ok is True
        assert env.result == {"echo": "hello", "from": "daemon"}
        assert env.error is None
    finally:
        await _teardown_pair(pair)


@pytest.mark.asyncio
async def test_outgoing_call_unknown_method_returns_typed_error() -> None:
    """Daemon-side ``UnknownMethod`` flows back as an envelope error,
    NOT as a Python exception."""
    server = RunnerRPCServer()
    pair = await _make_pair(server)
    try:
        loop = asyncio.get_running_loop()
        env = await loop.run_in_executor(
            None,
            lambda: pair.runner_rpc.outgoing_call(
                "nonexistent", {}, timeout=2,
            ),
        )
        assert env.ok is False
        assert env.error is not None
        assert env.error.type == "UnknownMethod"
        assert "nonexistent" in env.error.message
    finally:
        await _teardown_pair(pair)


@pytest.mark.asyncio
async def test_handler_exception_serialized_to_error_payload() -> None:
    """Handler raises → typed error with sanitized traceback."""
    server = RunnerRPCServer(workspace_root="/tmp/test-ws")

    async def _boom(args: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("kaboom")

    server.register("boom", _boom)
    pair = await _make_pair(server)
    try:
        loop = asyncio.get_running_loop()
        env = await loop.run_in_executor(
            None,
            lambda: pair.runner_rpc.outgoing_call("boom", {}, timeout=2),
        )
        assert env.ok is False
        assert env.error is not None
        assert env.error.type == "ValueError"
        assert "kaboom" in env.error.message
        assert env.error.traceback is not None
        assert "ValueError: kaboom" in env.error.traceback
    finally:
        await _teardown_pair(pair)


# ----------------------------------------------------------------------
# Concurrent outgoing calls
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_outgoing_calls_complete_independently() -> None:
    """Two outgoing calls in flight at once; responses arrive in
    arbitrary order; both resolve to the right future."""
    server = RunnerRPCServer()
    barrier = asyncio.Barrier(2) if hasattr(asyncio, "Barrier") else None

    async def _slow_ping(args: Dict[str, Any]) -> Dict[str, Any]:
        # Stagger so the two responses interleave on the wire.
        delay = float(args.get("delay", 0.0))
        await asyncio.sleep(delay)
        return {"id": args.get("id")}

    server.register("ping", _slow_ping)
    pair = await _make_pair(server)
    try:
        loop = asyncio.get_running_loop()
        # Issue two concurrent outgoing calls from worker threads.
        fut_a = loop.run_in_executor(
            None,
            lambda: pair.runner_rpc.outgoing_call(
                "ping", {"id": "a", "delay": 0.05}, timeout=2,
            ),
        )
        fut_b = loop.run_in_executor(
            None,
            lambda: pair.runner_rpc.outgoing_call(
                "ping", {"id": "b", "delay": 0.0}, timeout=2,
            ),
        )
        env_a, env_b = await asyncio.gather(fut_a, fut_b)
        assert env_a.ok and env_a.result == {"id": "a"}
        assert env_b.ok and env_b.result == {"id": "b"}
    finally:
        await _teardown_pair(pair)


# ----------------------------------------------------------------------
# Channel-closed semantics
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outgoing_call_after_shutdown_raises() -> None:
    server = RunnerRPCServer()
    pair = await _make_pair(server)
    try:
        pair.runner_rpc.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            pair.runner_rpc.outgoing_call("ping", {}, timeout=1)
    finally:
        # Already shut down on runner side; tear down daemon side only.
        pair.runner_thread.join(timeout=2)
        if pair.daemon_client._read_task is not None:
            pair.daemon_client._read_task.cancel()


# ----------------------------------------------------------------------
# RunnerRPCServer unit tests (no socket — direct dispatch)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rpc_server_dispatch_unknown_method() -> None:
    server = RunnerRPCServer()
    env = await server.dispatch(RequestEnvelope(id=1, method="x", args={}))
    assert env.id == 1
    assert env.ok is False
    assert env.error is not None
    assert env.error.type == "UnknownMethod"


@pytest.mark.asyncio
async def test_rpc_server_dispatch_handler_returns_dict() -> None:
    server = RunnerRPCServer()

    async def _handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"received": dict(args)}

    server.register("ping", _handler)
    env = await server.dispatch(
        RequestEnvelope(id=42, method="ping", args={"k": "v"}),
    )
    assert env.id == 42
    assert env.ok is True
    assert env.result == {"received": {"k": "v"}}


@pytest.mark.asyncio
async def test_rpc_server_register_overwrite_warns(caplog) -> None:
    server = RunnerRPCServer()

    async def _h1(args):
        return "first"

    async def _h2(args):
        return "second"

    server.register("ping", _h1)
    with caplog.at_level("WARNING"):
        server.register("ping", _h2)
    assert any("re-registering" in r.message for r in caplog.records)
    env = await server.dispatch(RequestEnvelope(id=1, method="ping", args={}))
    assert env.result == "second"


@pytest.mark.asyncio
async def test_rpc_server_handler_traceback_sanitized(tmp_path) -> None:
    """Workspace path in handler exception traceback gets redacted."""
    ws = str(tmp_path / "ws")
    server = RunnerRPCServer(workspace_root=ws)

    async def _explode(args):
        raise FileNotFoundError(f"could not open {ws}/secret.txt")

    server.register("explode", _explode)
    env = await server.dispatch(RequestEnvelope(id=1, method="explode", args={}))
    assert env.ok is False
    assert env.error is not None
    # Workspace path replaced in the message.
    assert "<WORKSPACE>/secret.txt" in env.error.message
    assert ws not in env.error.message
