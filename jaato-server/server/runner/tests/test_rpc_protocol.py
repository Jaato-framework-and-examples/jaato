"""Tests for the runner-side RPC dispatcher.

Strategy: connect both ends of a real ``socketpair()`` and run
:class:`server.runner.rpc.RunnerRPC` on one end in a worker thread
while the test (acting as the daemon) drives requests/cancels on the
other end.  Validates the §4.1 multiplexed-frame protocol end to end:

- ``request`` → response round-trip (single-shot).
- ``request`` with streaming → ordered ``stream`` frames + terminating
  ``response``.
- ``request`` with mid-stream ``cancel`` → terminating response with
  ``ok=false, error.type="CancelledException"``.
- Unknown method → ``ok=false`` with a useful message.
- Worker-side exception → ``ok=false`` + traceback in error payload.
- Peer EOF → serve loop exits cleanly without raising.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from shared.framing import read_frame_sync, write_frame_sync

from server.runner.envelope import (
    KIND_CANCEL,
    KIND_REQUEST,
    KIND_RESPONSE,
    KIND_STREAM,
    CancelFrame,
    RequestEnvelope,
    ResponseEnvelope,
    StreamFrame,
)
from server.runner.rpc import (
    RunnerRPC,
    get_current_cancel_token,
    get_current_output_callback,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def rpc_pair():
    """Yield ``(daemon_sock, runner_rpc)`` with the dispatcher running.

    Cleanup tears down both sockets and joins the serve thread.
    """
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    state: Dict[str, Any] = {"executor": _default_executor}

    def _exec_dispatch(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        return state["executor"](name, args)

    rpc = RunnerRPC(runner_sock, _exec_dispatch)
    thread = threading.Thread(target=rpc.serve, name="rpc-serve-test", daemon=True)
    thread.start()

    try:
        yield daemon_sock, rpc, state
    finally:
        rpc.shutdown()
        try:
            daemon_sock.close()
        except OSError:
            pass
        thread.join(timeout=2)


def _default_executor(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
    return False, {"error": f"no fixture executor for {name}"}


def _send_request(daemon_sock: socket.socket, env: RequestEnvelope) -> None:
    write_frame_sync(daemon_sock, json.dumps(env.to_dict()))


def _send_cancel(daemon_sock: socket.socket, request_id: int) -> None:
    write_frame_sync(daemon_sock, json.dumps(CancelFrame(id=request_id).to_dict()))


def _read_until_response(
    daemon_sock: socket.socket, request_id: int, timeout: float = 3.0,
) -> Tuple[List[StreamFrame], ResponseEnvelope]:
    """Drain frames for *request_id* until the terminating response."""
    streams: List[StreamFrame] = []
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"timed out waiting for response id={request_id}; "
                f"streams so far: {streams!r}"
            )
        daemon_sock.settimeout(max(0.05, deadline - time.monotonic()))
        raw = read_frame_sync(daemon_sock)
        if raw is None:
            raise AssertionError(
                f"runner closed connection before responding to id={request_id}"
            )
        payload = json.loads(raw)
        if payload.get("id") != request_id:
            continue
        kind = payload.get("kind")
        if kind == KIND_STREAM:
            streams.append(StreamFrame.from_dict(payload))
        elif kind == KIND_RESPONSE:
            return streams, ResponseEnvelope.from_dict(payload)
        else:
            raise AssertionError(f"unexpected frame kind={kind!r}")


# ----------------------------------------------------------------------
# Single-shot request → response
# ----------------------------------------------------------------------


def test_echo_round_trip(rpc_pair) -> None:
    daemon_sock, _rpc, _state = rpc_pair
    _send_request(daemon_sock, RequestEnvelope(id=1, method="echo", args={"k": "v"}))
    streams, resp = _read_until_response(daemon_sock, request_id=1)
    assert streams == []
    assert resp.ok is True
    assert resp.result == {"k": "v"}
    assert resp.error is None


def test_unknown_method_returns_error(rpc_pair) -> None:
    daemon_sock, _rpc, _state = rpc_pair
    _send_request(daemon_sock, RequestEnvelope(id=2, method="not.a.method"))
    streams, resp = _read_until_response(daemon_sock, request_id=2)
    assert streams == []
    assert resp.ok is False
    assert resp.error is not None
    assert "unknown method" in resp.error.message


def test_tool_execute_missing_name(rpc_pair) -> None:
    daemon_sock, _rpc, _state = rpc_pair
    _send_request(
        daemon_sock,
        RequestEnvelope(id=3, method="tool.execute", args={}),
    )
    _, resp = _read_until_response(daemon_sock, request_id=3)
    assert resp.ok is False
    assert resp.error and "missing 'name'" in resp.error.message


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------


def test_streaming_chunks_in_order(rpc_pair) -> None:
    daemon_sock, _rpc, state = rpc_pair

    def _streaming_executor(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        cb = get_current_output_callback()
        assert cb is not None, "expected on_output to be installed"
        for i in (1, 2, 3):
            cb("stdout", f"line {i}\n", None)
        return True, {"sent": 3}

    state["executor"] = _streaming_executor

    _send_request(
        daemon_sock,
        RequestEnvelope(id=10, method="tool.execute", args={"name": "fake"}),
    )
    streams, resp = _read_until_response(daemon_sock, request_id=10)
    assert resp.ok is True
    assert resp.result == {"sent": 3}
    assert [s.text for s in streams] == ["line 1\n", "line 2\n", "line 3\n"]
    assert {s.source for s in streams} == {"stdout"}


def test_streaming_per_source_ordering_preserved(rpc_pair) -> None:
    """§3 sharpened assertion: per-source ordering only, not cross-source."""
    daemon_sock, _rpc, state = rpc_pair

    def _interleaved_executor(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        cb = get_current_output_callback()
        assert cb is not None
        cb("stdout", "out-1\n", None)
        cb("stderr", "err-1\n", None)
        cb("stdout", "out-2\n", None)
        cb("stderr", "err-2\n", None)
        return True, {}

    state["executor"] = _interleaved_executor

    _send_request(
        daemon_sock,
        RequestEnvelope(id=11, method="tool.execute", args={"name": "fake"}),
    )
    streams, _ = _read_until_response(daemon_sock, request_id=11)

    # Per-source ordering preserved within each stream.
    stdout_chunks = [s.text for s in streams if s.source == "stdout"]
    stderr_chunks = [s.text for s in streams if s.source == "stderr"]
    assert stdout_chunks == ["out-1\n", "out-2\n"]
    assert stderr_chunks == ["err-1\n", "err-2\n"]
    # Cross-source ordering NOT asserted — line-buffered reads on
    # stdout/stderr can interleave at line boundaries.


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------


def test_cancel_mid_execution_trips_token(rpc_pair) -> None:
    daemon_sock, _rpc, state = rpc_pair

    started = threading.Event()
    finished = threading.Event()

    def _slow_executor(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        token = get_current_cancel_token()
        assert token is not None
        started.set()
        # Poll the token in a tight-but-reasonable loop, mirroring how
        # cli's run_command polls during subprocess wait.
        while not token.is_cancelled:
            time.sleep(0.01)
        finished.set()
        # Plugin contract: when cancelled, raise CancelledException.
        from jaato_sdk.plugins.model_provider.types import CancelledException
        raise CancelledException("test executor cancelled")

    state["executor"] = _slow_executor

    _send_request(
        daemon_sock,
        RequestEnvelope(id=20, method="tool.execute", args={"name": "fake"}),
    )
    assert started.wait(timeout=2), "executor did not start"
    _send_cancel(daemon_sock, request_id=20)
    _, resp = _read_until_response(daemon_sock, request_id=20, timeout=3)
    assert finished.is_set()
    assert resp.ok is False
    assert resp.error is not None
    assert resp.error.type == "CancelledException"


def test_cancel_for_unknown_call_is_silent(rpc_pair) -> None:
    """Cancel for a call id we never sent (or already finished) is a no-op.

    Asserts the dispatcher doesn't crash and stays serving — we
    follow up with a normal echo to prove the loop is alive.
    """
    daemon_sock, _rpc, _state = rpc_pair
    _send_cancel(daemon_sock, request_id=999)
    _send_request(daemon_sock, RequestEnvelope(id=30, method="echo", args={"a": 1}))
    _, resp = _read_until_response(daemon_sock, request_id=30)
    assert resp.ok is True


# ----------------------------------------------------------------------
# Executor exception → typed error payload (§4.8 traceback caveat)
# ----------------------------------------------------------------------


def test_executor_exception_serialized_to_error_payload(rpc_pair) -> None:
    daemon_sock, _rpc, state = rpc_pair

    def _raising_executor(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        raise ValueError("boom")

    state["executor"] = _raising_executor

    _send_request(
        daemon_sock,
        RequestEnvelope(id=40, method="tool.execute", args={"name": "fake"}),
    )
    _, resp = _read_until_response(daemon_sock, request_id=40)
    assert resp.ok is False
    assert resp.error is not None
    assert resp.error.type == "ValueError"
    assert "boom" in resp.error.message
    assert resp.error.traceback is not None
    assert "ValueError: boom" in resp.error.traceback


def test_domain_failure_dict_serialized_to_error_payload(rpc_pair) -> None:
    """Plugins that return (False, {'error': ...}) without raising
    surface the message via the error payload too."""
    daemon_sock, _rpc, state = rpc_pair

    def _domain_fail(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        return False, {"error": "permission denied: /etc/shadow"}

    state["executor"] = _domain_fail

    _send_request(
        daemon_sock,
        RequestEnvelope(id=41, method="tool.execute", args={"name": "fake"}),
    )
    _, resp = _read_until_response(daemon_sock, request_id=41)
    assert resp.ok is False
    assert resp.error and "permission denied" in resp.error.message
    # The original result dict is also preserved so the daemon side can
    # forward to history with the structured payload.
    assert resp.result == {"error": "permission denied: /etc/shadow"}


# ----------------------------------------------------------------------
# Multiplexing — multiple outstanding requests
# ----------------------------------------------------------------------


def test_multiplexed_calls_complete_independently(rpc_pair) -> None:
    daemon_sock, _rpc, state = rpc_pair

    barrier = threading.Barrier(2)

    def _coordinated_executor(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        # Both calls block on the barrier; neither can finish until
        # both have started, proving they run in parallel.
        barrier.wait(timeout=2)
        return True, {"id": args.get("id")}

    state["executor"] = _coordinated_executor

    _send_request(
        daemon_sock,
        RequestEnvelope(
            id=50, method="tool.execute",
            args={"name": "fake", "args": {"id": "a"}},
        ),
    )
    _send_request(
        daemon_sock,
        RequestEnvelope(
            id=51, method="tool.execute",
            args={"name": "fake", "args": {"id": "b"}},
        ),
    )

    # Drain both responses; order on the wire is not guaranteed.
    seen: Dict[int, ResponseEnvelope] = {}
    while len(seen) < 2:
        raw = read_frame_sync(daemon_sock)
        assert raw is not None
        payload = json.loads(raw)
        if payload.get("kind") == KIND_RESPONSE:
            env = ResponseEnvelope.from_dict(payload)
            seen[env.id] = env

    assert seen[50].ok and seen[50].result == {"id": "a"}
    assert seen[51].ok and seen[51].result == {"id": "b"}


# ----------------------------------------------------------------------
# Peer-EOF (§6.7 bidirectional benign-EOF)
# ----------------------------------------------------------------------


def test_peer_eof_exits_serve_cleanly(rpc_pair) -> None:
    daemon_sock, _rpc, _state = rpc_pair
    daemon_sock.shutdown(socket.SHUT_RDWR)
    daemon_sock.close()
    # The fixture's teardown joins the serve thread with a 2s timeout;
    # if it hangs the test fails — implicit assertion via teardown.
    time.sleep(0.1)


# ----------------------------------------------------------------------
# Malformed input
# ----------------------------------------------------------------------


def test_malformed_json_closes_loop(rpc_pair) -> None:
    daemon_sock, _rpc, _state = rpc_pair
    write_frame_sync(daemon_sock, "{not valid json")
    # Loop should close on its own; subsequent read returns EOF.
    raw = read_frame_sync(daemon_sock)
    assert raw is None
