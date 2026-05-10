"""Tests for the session.try_completion_nudge RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.4.3a, prerequisite
for the 6.6.4.3b seat-flip).

Atomic check-and-increment for the completion-nudge guard —
collapses 3 daemon-side reaches into the runner-side
JaatoSession (``_signal_completion_called`` read,
``_completion_nudges_fired`` read, increment) into one
round-trip.

Wraps the new public method ``JaatoSession.try_completion_nudge``
(missing-method gap closed alongside this commit).

Tests pin:

- happy paths: should-nudge True / False / budget-exhausted
- counter mutation: increments on True, no change on False
- arg validation: missing / non-int / bool ``max_nudges`` →
  stage="decode"
- no_host / no_session error paths
- session class lacking the public method → stage="missing_method"
- underlying method raises → stage="call"
- dispatch routing: ``session.try_completion_nudge`` reaches
  the handler
- e2e via daemon-side wrapper round-trip
- e2e via threadsafe wrapper from a background thread
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, Optional, Tuple

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from server.runner_rpc_client import RunnerRPCClient, RunnerCallError
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-nudge-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in mirroring JaatoSession's
    ``try_completion_nudge`` semantics (read+inc on
    ``_signal_completion_called`` / ``_completion_nudges_fired``)."""

    def __init__(
        self,
        *,
        signal_completion_called: bool = False,
        completion_nudges_fired: int = 0,
        raise_on_call: Optional[Exception] = None,
    ) -> None:
        self._signal_completion_called = signal_completion_called
        self._completion_nudges_fired = completion_nudges_fired
        self._raise = raise_on_call
        self.calls: list = []

    def try_completion_nudge(self, max_nudges: int) -> Tuple[bool, int]:
        self.calls.append(max_nudges)
        if self._raise is not None:
            raise self._raise
        if (
            not self._signal_completion_called
            and self._completion_nudges_fired < max_nudges
        ):
            self._completion_nudges_fired += 1
            return True, self._completion_nudges_fired
        return False, self._completion_nudges_fired


class _SessionWithoutMethod:
    """Stand-in for an old JaatoSession (rolling-upgrade)
    without the ``try_completion_nudge`` public method —
    handler must surface stage="missing_method"."""

    pass


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Direct handler tests
# ----------------------------------------------------------------------


def test_try_completion_nudge_should_nudge_true_increments() -> None:
    """Agent didn't call signal_completion AND budget remains —
    handler returns should_nudge=True and reports the
    post-increment count."""
    rpc = _make_lone_runner()
    session = _FakeSession(
        signal_completion_called=False,
        completion_nudges_fired=0,
    )
    _install(rpc, session)

    ok, result = rpc._handle_session_try_completion_nudge(
        {"max_nudges": 2},
    )
    assert ok is True
    assert result == {"should_nudge": True, "nudges_fired": 1}
    # Counter actually mutated.
    assert session._completion_nudges_fired == 1
    assert session.calls == [2]


def test_try_completion_nudge_signal_completion_called_returns_false() -> None:
    """Agent called signal_completion → no nudge needed; counter
    untouched."""
    rpc = _make_lone_runner()
    session = _FakeSession(
        signal_completion_called=True,
        completion_nudges_fired=0,
    )
    _install(rpc, session)

    ok, result = rpc._handle_session_try_completion_nudge(
        {"max_nudges": 2},
    )
    assert ok is True
    assert result == {"should_nudge": False, "nudges_fired": 0}
    assert session._completion_nudges_fired == 0


def test_try_completion_nudge_budget_exhausted_returns_false() -> None:
    """Counter at max → no nudge fires; counter untouched."""
    rpc = _make_lone_runner()
    session = _FakeSession(
        signal_completion_called=False,
        completion_nudges_fired=2,
    )
    _install(rpc, session)

    ok, result = rpc._handle_session_try_completion_nudge(
        {"max_nudges": 2},
    )
    assert ok is True
    assert result == {"should_nudge": False, "nudges_fired": 2}
    assert session._completion_nudges_fired == 2


def test_try_completion_nudge_zero_max_returns_false() -> None:
    """max_nudges=0 → no nudge regardless of state.  Defensive
    against caller misconfiguration."""
    rpc = _make_lone_runner()
    session = _FakeSession(
        signal_completion_called=False,
        completion_nudges_fired=0,
    )
    _install(rpc, session)

    ok, result = rpc._handle_session_try_completion_nudge(
        {"max_nudges": 0},
    )
    assert ok is True
    assert result == {"should_nudge": False, "nudges_fired": 0}


def test_try_completion_nudge_increments_through_budget() -> None:
    """Successive calls increment until budget exhausted."""
    rpc = _make_lone_runner()
    session = _FakeSession(
        signal_completion_called=False,
        completion_nudges_fired=0,
    )
    _install(rpc, session)

    # 1st: should fire, count → 1
    ok, r1 = rpc._handle_session_try_completion_nudge({"max_nudges": 2})
    assert ok and r1 == {"should_nudge": True, "nudges_fired": 1}
    # 2nd: should fire, count → 2
    ok, r2 = rpc._handle_session_try_completion_nudge({"max_nudges": 2})
    assert ok and r2 == {"should_nudge": True, "nudges_fired": 2}
    # 3rd: budget exhausted
    ok, r3 = rpc._handle_session_try_completion_nudge({"max_nudges": 2})
    assert ok and r3 == {"should_nudge": False, "nudges_fired": 2}


def test_try_completion_nudge_rejects_missing_max_nudges() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    ok, result = rpc._handle_session_try_completion_nudge({})
    assert ok is False
    assert result["stage"] == "decode"


def test_try_completion_nudge_rejects_non_int_max_nudges() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    for bad in ["2", 2.5, None, [2], {"n": 2}]:
        ok, result = rpc._handle_session_try_completion_nudge(
            {"max_nudges": bad},
        )
        assert ok is False, f"should reject max_nudges={bad!r}"
        assert result["stage"] == "decode"


def test_try_completion_nudge_rejects_bool_max_nudges() -> None:
    """bool is an int subclass in Python — explicitly reject so
    callers don't accidentally pass True/False thinking it'll
    coerce to 1/0."""
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession())

    for bad in [True, False]:
        ok, result = rpc._handle_session_try_completion_nudge(
            {"max_nudges": bad},
        )
        assert ok is False, f"should reject max_nudges={bad!r}"
        assert result["stage"] == "decode"


def test_try_completion_nudge_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_try_completion_nudge(
        {"max_nudges": 2},
    )
    assert ok is False
    assert result["stage"] == "no_host"


def test_try_completion_nudge_missing_method_returns_error() -> None:
    """Rolling-upgrade scenario: session class without the
    public method → stage="missing_method"."""
    rpc = _make_lone_runner()
    _install(rpc, _SessionWithoutMethod())

    ok, result = rpc._handle_session_try_completion_nudge(
        {"max_nudges": 2},
    )
    assert ok is False
    assert result["stage"] == "missing_method"


def test_try_completion_nudge_underlying_raise_returns_call_error() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(raise_on_call=RuntimeError("boom"))
    _install(rpc, session)

    ok, result = rpc._handle_session_try_completion_nudge(
        {"max_nudges": 2},
    )
    assert ok is False
    assert result["stage"] == "call"
    assert "boom" in result["error"]


def test_dispatch_routes_session_try_completion_nudge() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(
        signal_completion_called=False,
        completion_nudges_fired=0,
    )
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.try_completion_nudge",
        args={"max_nudges": 2},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"should_nudge": True, "nudges_fired": 1}


# ----------------------------------------------------------------------
# E2E: daemon-side wrapper round-trip
# ----------------------------------------------------------------------


class _Pair:
    def __init__(self, runner_rpc, runner_thread, daemon_client) -> None:
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
        target=runner_rpc.serve, name="rpc-nudge-test", daemon=True,
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


@pytest.mark.asyncio
async def test_e2e_try_completion_nudge_round_trip_true() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession(
            signal_completion_called=False,
            completion_nudges_fired=0,
        )
        _install(pair.runner_rpc, session)

        should_nudge, nudges_fired = (
            await pair.daemon_client.session_try_completion_nudge(2)
        )
        assert should_nudge is True
        assert nudges_fired == 1
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_try_completion_nudge_round_trip_false() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession(
            signal_completion_called=True,
            completion_nudges_fired=0,
        )
        _install(pair.runner_rpc, session)

        should_nudge, nudges_fired = (
            await pair.daemon_client.session_try_completion_nudge(2)
        )
        assert should_nudge is False
        assert nudges_fired == 0
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_try_completion_nudge_threadsafe_from_thread() -> None:
    """The threadsafe wrapper is the path the daemon's
    ``_start_model_thread`` will adopt in §7c step 6.6.4.3b —
    pin it works from a background thread."""
    pair = await _make_pair()
    try:
        session = _FakeSession(
            signal_completion_called=False,
            completion_nudges_fired=0,
        )
        _install(pair.runner_rpc, session)

        result_holder: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                result_holder["value"] = (
                    pair.daemon_client.session_try_completion_nudge_threadsafe(2)
                )
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        # Yield to the event loop so the threadsafe call's coro
        # actually runs.
        for _ in range(50):
            if "value" in result_holder or "error" in result_holder:
                break
            await asyncio.sleep(0.05)
        t.join(timeout=2)

        assert "error" not in result_holder, result_holder.get("error")
        assert result_holder["value"] == (True, 1)
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_try_completion_nudge_missing_method_raises() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _SessionWithoutMethod())
        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_try_completion_nudge(2)
        # Error surfaces the "lacks public" handler text — pin
        # the method name so the daemon-side caller can grep
        # the cause without parsing stage codes.
        assert "try_completion_nudge" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# Real JaatoSession: pin the public method exists and matches contract
# ----------------------------------------------------------------------


def test_jaato_session_has_public_method() -> None:
    """Pin the missing-method gap is closed — the runner-side
    handler's ``getattr(session, 'try_completion_nudge', None)``
    finds an actual method on real JaatoSession."""
    from shared.jaato_session import JaatoSession
    # Method exists on the class (not just instance) so the
    # missing-method check is structural, not stateful.
    assert callable(getattr(JaatoSession, "try_completion_nudge", None))
