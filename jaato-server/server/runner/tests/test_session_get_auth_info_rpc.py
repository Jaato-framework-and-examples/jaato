"""Tests for the session.get_auth_info RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.4.5c.1).

Reads the credential-source description string from the runner-
side session's provider.  Replaces 2 daemon-side reaches into
``self._jaato.auth_info`` (core.py:2073, 4481) — the property
reads ``_session._provider.get_auth_info()`` daemon-side, which
post-6.6.4.3b seat-flip is the wrong (dead) session.

Lands the missing public method
:meth:`JaatoSession.get_auth_info` alongside the RPC handler
(missing-method gap caught by the §7c step 6.6.4.5c.0 audit
at commit a88676ca).

Tests pin:

- happy paths: provider with get_auth_info → returns its string
- empty-string path: no provider attached → returns ""
- empty-string path: provider lacks get_auth_info → returns ""
- empty-string path: provider's get_auth_info returns None → ""
- empty-string path: provider's get_auth_info raises → ""
- arg validation: handler ignores args (no required args)
- no_host / no_session error paths
- session class lacking the public method → stage="missing_method"
- underlying method raises → stage="call"
- dispatch routing
- e2e round-trip via async + threadsafe wrappers
- structural pin: real JaatoSession exposes the public method
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


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-auth-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeProvider:
    def __init__(
        self,
        info: Optional[str] = None,
        raise_on_call: Optional[Exception] = None,
    ) -> None:
        self._info = info
        self._raise = raise_on_call

    def get_auth_info(self) -> Optional[str]:
        if self._raise is not None:
            raise self._raise
        return self._info


class _ProviderlessProvider:
    """Provider stand-in that doesn't implement get_auth_info."""

    pass


class _FakeSession:
    """Stand-in mirroring JaatoSession.get_auth_info semantics:
    reads ``self._provider.get_auth_info()`` if both exist."""

    def __init__(
        self,
        *,
        provider: Optional[Any] = None,
        raise_on_get: Optional[Exception] = None,
    ) -> None:
        self._provider = provider
        self._raise = raise_on_get
        self.calls = 0

    def get_auth_info(self) -> str:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        if self._provider and hasattr(self._provider, 'get_auth_info'):
            try:
                return str(self._provider.get_auth_info() or "")
            except Exception:  # noqa: BLE001
                return ""
        return ""


class _SessionWithoutMethod:
    """Stand-in for an old JaatoSession that lacks get_auth_info —
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


def test_get_auth_info_with_provider_returns_string() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(provider=_FakeProvider(info="API key from ~/.jaato"))
    _install(rpc, session)

    ok, result = rpc._handle_session_get_auth_info()
    assert ok is True
    assert result == {"auth_info": "API key from ~/.jaato"}
    assert session.calls == 1


def test_get_auth_info_no_provider_returns_empty_string() -> None:
    """Pre-configure() session has no provider — return ""."""
    rpc = _make_lone_runner()
    session = _FakeSession(provider=None)
    _install(rpc, session)

    ok, result = rpc._handle_session_get_auth_info()
    assert ok is True
    assert result == {"auth_info": ""}


def test_get_auth_info_provider_without_method_returns_empty() -> None:
    """Provider class doesn't implement get_auth_info — return ""."""
    rpc = _make_lone_runner()
    session = _FakeSession(provider=_ProviderlessProvider())
    _install(rpc, session)

    ok, result = rpc._handle_session_get_auth_info()
    assert ok is True
    assert result == {"auth_info": ""}


def test_get_auth_info_provider_returns_none_coerced_to_empty() -> None:
    """Provider's get_auth_info returns None — coerced to ""."""
    rpc = _make_lone_runner()
    session = _FakeSession(provider=_FakeProvider(info=None))
    _install(rpc, session)

    ok, result = rpc._handle_session_get_auth_info()
    assert ok is True
    assert result == {"auth_info": ""}


def test_get_auth_info_provider_raises_returns_empty() -> None:
    """Provider's get_auth_info raises — fake session swallows
    (matches JaatoSession.get_auth_info best-effort contract)."""
    rpc = _make_lone_runner()
    session = _FakeSession(
        provider=_FakeProvider(raise_on_call=RuntimeError("provider boom")),
    )
    _install(rpc, session)

    ok, result = rpc._handle_session_get_auth_info()
    assert ok is True
    assert result == {"auth_info": ""}


def test_get_auth_info_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_auth_info()
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_auth_info_missing_method_returns_error() -> None:
    """Rolling-upgrade scenario: session class without
    get_auth_info → stage="missing_method"."""
    rpc = _make_lone_runner()
    _install(rpc, _SessionWithoutMethod())

    ok, result = rpc._handle_session_get_auth_info()
    assert ok is False
    assert result["stage"] == "missing_method"


def test_get_auth_info_method_raises_returns_call_error() -> None:
    """When the session.get_auth_info method itself raises (not
    the wrapped provider call), surface stage="call"."""
    rpc = _make_lone_runner()
    session = _FakeSession(
        raise_on_get=RuntimeError("session boom"),
    )
    _install(rpc, session)

    ok, result = rpc._handle_session_get_auth_info()
    assert ok is False
    assert result["stage"] == "call"
    assert "session boom" in result["error"]


def test_dispatch_routes_session_get_auth_info() -> None:
    rpc = _make_lone_runner()
    session = _FakeSession(provider=_FakeProvider(info="OAuth"))
    _install(rpc, session)

    env = RequestEnvelope(
        id=1, method="session.get_auth_info", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result == {"auth_info": "OAuth"}


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
        target=runner_rpc.serve, name="rpc-auth-info-test", daemon=True,
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
async def test_e2e_get_auth_info_round_trip() -> None:
    pair = await _make_pair()
    try:
        session = _FakeSession(
            provider=_FakeProvider(info="PKCE OAuth"),
        )
        _install(pair.runner_rpc, session)

        info = await pair.daemon_client.session_get_auth_info()
        assert info == "PKCE OAuth"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_get_auth_info_threadsafe_from_thread() -> None:
    """The threadsafe wrapper is the path daemon callers use from
    worker threads."""
    pair = await _make_pair()
    try:
        session = _FakeSession(
            provider=_FakeProvider(info="API key from env"),
        )
        _install(pair.runner_rpc, session)

        result_holder: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                result_holder["value"] = (
                    pair.daemon_client.session_get_auth_info_threadsafe()
                )
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        for _ in range(50):
            if "value" in result_holder or "error" in result_holder:
                break
            await asyncio.sleep(0.05)
        t.join(timeout=2)

        assert "error" not in result_holder, result_holder.get("error")
        assert result_holder["value"] == "API key from env"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_get_auth_info_missing_method_raises() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _SessionWithoutMethod())
        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_get_auth_info()
        assert "get_auth_info" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# Real JaatoSession: pin the public method exists and matches contract
# ----------------------------------------------------------------------


def test_jaato_session_has_public_get_auth_info_method() -> None:
    """Pin the missing-method gap is closed — the runner-side
    handler's ``getattr(session, 'get_auth_info', None)`` finds
    an actual method on real JaatoSession."""
    from shared.jaato_session import JaatoSession
    assert callable(getattr(JaatoSession, "get_auth_info", None))


def test_jaato_session_get_auth_info_returns_empty_when_no_provider() -> None:
    """Pin the public method returns "" when no provider is
    attached (defensive pre-configure() contract)."""
    from shared.jaato_session import JaatoSession
    # Construct a bare instance via __new__ to avoid full init.
    s = JaatoSession.__new__(JaatoSession)
    s._provider = None
    assert s.get_auth_info() == ""
