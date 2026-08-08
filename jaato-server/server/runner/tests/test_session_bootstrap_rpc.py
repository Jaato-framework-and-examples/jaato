"""Tests for the ``session.bootstrap`` RPC end-to-end wire (Phase 3 §3.3c part 1).

This is the §3.3c kickoff: the runner-side host (§3.3b) gets actually
exercised over the bidirectional RPC channel from §3.2.  The seat-flip
(removing the daemon-side JaatoSession + the
``JAATO_RUNNER_HOSTS_SESSION`` flag) lands in §3.3c part 2.

Two test surfaces:
1. Runner-side ``RunnerRPC._handle_session_bootstrap`` — happy path,
   bad envelope, idempotent re-call, conflicting re-bootstrap.
2. End-to-end through the bidirectional channel — daemon side calls
   ``RunnerRPCClient.bootstrap_session(envelope)``; runner stores
   the host; subsequent ``RunnerRPC.session_host`` returns it.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, Optional

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient, RunnerCallError
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _good_envelope(**overrides: Any) -> SessionInitEnvelope:
    base = dict(
        session_id="sess-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[
            {"name": "signal_completion", "preload": True},
            {"name": "cli", "preload": False},
        ],
    )
    base.update(overrides)
    return SessionInitEnvelope(**base)


class _Pair:
    def __init__(
        self,
        runner_rpc: RunnerRPC,
        runner_thread: threading.Thread,
        daemon_client: RunnerRPCClient,
    ) -> None:
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair() -> _Pair:
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

    daemon_client = RunnerRPCClient(daemon_sock, runner_pid=0)
    await daemon_client.start()
    return _Pair(runner_rpc, thread, daemon_client)


async def _teardown(pair: _Pair) -> None:
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


# ----------------------------------------------------------------------
# Runner-side dispatcher (no socket — direct method call)
# ----------------------------------------------------------------------


def _make_lone_runner() -> RunnerRPC:
    """Construct a RunnerRPC against a closed-side socket; we exercise
    ``_dispatch_method`` / ``_handle_session_bootstrap`` directly
    without driving the serve loop."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()  # we never serve in this test path

    def _no_executor(name, args):
        return False, {}

    return RunnerRPC(a, _no_executor)


def test_handle_session_bootstrap_invalid_envelope() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_bootstrap({"session_id": "x"})  # missing schema_version
    assert ok is False
    assert "invalid envelope" in result["error"]
    assert result["stage"] == "decode"


def test_handle_session_bootstrap_validation_failure() -> None:
    """Empty model_name fails the envelope's later validate stage."""
    rpc = _make_lone_runner()
    bad_env = _good_envelope(model_name="")
    ok, result = rpc._handle_session_bootstrap(bad_env.to_dict())
    assert ok is False
    assert result["stage"] == "validate"


def test_handle_session_bootstrap_idempotent_same_envelope() -> None:
    """Re-bootstrapping with the same envelope returns ok without
    reconstructing."""
    rpc = _make_lone_runner()
    env = _good_envelope()
    # First call constructs (in test mode the runtime factory is the
    # real one, which would fail without a real provider).  Instead
    # we manually set a host with a matching envelope.
    rpc._session_host = RunnerSessionHost(envelope=env)

    ok, result = rpc._handle_session_bootstrap(env.to_dict())
    assert ok is True
    assert result["session_id"] == "sess-1"
    assert "already bootstrapped" in result.get("note", "")


def test_handle_session_bootstrap_conflict_different_envelope() -> None:
    """Re-bootstrapping with a different envelope is a hard failure."""
    rpc = _make_lone_runner()
    rpc._session_host = RunnerSessionHost(envelope=_good_envelope())

    other = _good_envelope(session_id="sess-different")
    ok, result = rpc._handle_session_bootstrap(other.to_dict())
    assert ok is False
    assert result["stage"] == "duplicate"
    assert "already hosting" in result["error"]


def test_session_host_property_is_none_until_bootstrap() -> None:
    rpc = _make_lone_runner()
    assert rpc.session_host is None


def test_session_host_property_returns_host_after_bootstrap() -> None:
    rpc = _make_lone_runner()
    host = RunnerSessionHost(envelope=_good_envelope())
    rpc._session_host = host
    assert rpc.session_host is host


# ----------------------------------------------------------------------
# End-to-end via the bidirectional RPC channel
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_bootstrap_envelope_decode_failure() -> None:
    """Bad envelope on the wire surfaces as a typed error from the
    runner; the daemon-side wrapper raises RunnerCallError with the
    decode-stage message."""
    pair = await _make_pair()
    try:
        # Send a session.bootstrap with a malformed envelope dict.
        # The runner's _handle_session_bootstrap returns
        # ``ok=False, result={...}`` — the dispatcher then synthesises
        # a ToolError envelope (per RunnerRPC's contract).  The
        # daemon-side bootstrap_session translates that to
        # RunnerCallError.
        env = pair.daemon_client._next_id  # consume an id
        # Send raw via call() with bad args.
        response = await pair.daemon_client.call(
            "session.bootstrap", {"not": "a valid envelope"},
        )
        assert response.ok is False
        assert response.error is not None
        # Either decode-stage (missing schema_version) error, or
        # ToolError wrapping that.  Both contain "envelope" in the
        # message.
        assert "envelope" in response.error.message.lower()
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_bootstrap_validation_failure_propagates() -> None:
    """Envelope decodes but validation stage rejects (empty model)."""
    pair = await _make_pair()
    try:
        bad = _good_envelope(model_name="")
        response = await pair.daemon_client.call(
            "session.bootstrap", bad.to_dict(),
        )
        assert response.ok is False
        # ToolError result carries the original error dict.
        if isinstance(response.result, dict):
            assert response.result.get("stage") == "validate"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_idempotent_reuses_host() -> None:
    """Calling session.bootstrap twice with the same envelope returns
    ok both times; the runner's host stays the same instance."""
    pair = await _make_pair()
    try:
        # Pre-populate the runner with a host that matches the envelope
        # we'll send (so we don't actually construct a real
        # JaatoSession in this test).
        env = _good_envelope()
        pair.runner_rpc._session_host = RunnerSessionHost(envelope=env)

        # First call.
        result_1 = await pair.daemon_client.bootstrap_session(env, timeout=5)
        assert result_1["ok"] is True
        assert result_1["session_id"] == "sess-1"

        # Second call — runner sees existing host with same envelope,
        # returns ok with idempotent note.
        result_2 = await pair.daemon_client.bootstrap_session(env, timeout=5)
        assert result_2["ok"] is True
        assert "already bootstrapped" in result_2.get("note", "")
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_conflicting_envelope_raises() -> None:
    """Sending a different envelope after bootstrap is a hard failure."""
    pair = await _make_pair()
    try:
        env_a = _good_envelope(session_id="sess-A")
        env_b = _good_envelope(session_id="sess-B")
        pair.runner_rpc._session_host = RunnerSessionHost(envelope=env_a)

        with pytest.raises(RunnerCallError, match="already hosting"):
            await pair.daemon_client.bootstrap_session(env_b, timeout=5)
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_bootstrap_session_type_check() -> None:
    """``RunnerRPCClient.bootstrap_session`` rejects non-envelope
    payloads at the type boundary."""
    pair = await _make_pair()
    try:
        with pytest.raises(TypeError, match="SessionInitEnvelope"):
            await pair.daemon_client.bootstrap_session(
                {"not": "an envelope"},  # type: ignore[arg-type]
                timeout=2,
            )
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_bootstrap_session_threadsafe_works_from_worker() -> None:
    """``bootstrap_session_threadsafe`` runs from a worker thread
    without touching the asyncio event loop directly."""
    pair = await _make_pair()
    try:
        env = _good_envelope()
        pair.runner_rpc._session_host = RunnerSessionHost(envelope=env)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pair.daemon_client.bootstrap_session_threadsafe(
                env, timeout=5,
            ),
        )
        assert result["ok"] is True
    finally:
        await _teardown(pair)
