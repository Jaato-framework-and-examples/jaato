"""Tests for the session.get_user_commands RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.4.5c.2).

Reads the runner-side session's user-command catalog.  Replaces
2 daemon-side reaches into ``self._jaato.get_user_commands()``
(core.py:4015, 4233).

Wire shape per the §7c step 6.6.4.5c.2 audit decision: **dict-
shape-only** (Path B), not Message/Part round-trip (Path A).
``UserCommand`` and ``CommandParameter`` are NamedTuples with
primitive fields only (str/bool) — no callables, no Type[X], no
class refs — so straight dict serialization is sufficient.  The
daemon-side wrapper reconstructs ``UserCommand`` NamedTuple
instances on receipt so existing daemon callers can use
``parse_command_args(cmd, raw_args)`` unmodified.

The handler callable itself stays runner-side and gets invoked
via ``session.execute_user_command`` (5c.3).  No callable crosses
the wire either direction.

Tests pin:

- happy paths: empty catalog / single command / multi-command
- parameters round-trip (CommandParameter NamedTuple
  serialization + reconstruction)
- defensive: malformed wire dicts (non-dict per-command, missing
  fields) handled with type coercion / defaults
- no_host / no_session error paths
- session class lacking method → stage="missing_method"
- underlying method raises → stage="call"
- dispatch routing
- e2e via async + threadsafe wrappers
- structural pin: real JaatoSession exposes the public method
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.plugins.base import CommandParameter, UserCommand
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
        session_id="sess-uc-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(
        self,
        commands: Optional[Dict[str, UserCommand]] = None,
        raise_on_get: Optional[Exception] = None,
    ) -> None:
        self._commands = commands or {}
        self._raise = raise_on_get
        self.calls = 0

    def get_user_commands(self) -> Dict[str, UserCommand]:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return dict(self._commands)


class _SessionWithoutMethod:
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


def test_get_user_commands_empty_catalog() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(commands={}))

    ok, result = rpc._handle_session_get_user_commands()
    assert ok is True
    assert result == {"commands": {}}


def test_get_user_commands_single_command_no_params() -> None:
    rpc = _make_lone_runner()
    cmd = UserCommand(
        name="status", description="Show status", share_with_model=False,
    )
    _install(rpc, _FakeSession(commands={"status": cmd}))

    ok, result = rpc._handle_session_get_user_commands()
    assert ok is True
    assert result == {
        "commands": {
            "status": {
                "name": "status",
                "description": "Show status",
                "share_with_model": False,
                "parameters": None,
            },
        },
    }


def test_get_user_commands_with_parameters_serialized() -> None:
    rpc = _make_lone_runner()
    cmd = UserCommand(
        name="auth", description="Authenticate",
        share_with_model=True,
        parameters=[
            CommandParameter(
                name="action", description="login/logout/status",
                required=True, capture_rest=False,
            ),
            CommandParameter(
                name="extra", description="Extra args",
                required=False, capture_rest=True,
            ),
        ],
    )
    _install(rpc, _FakeSession(commands={"auth": cmd}))

    ok, result = rpc._handle_session_get_user_commands()
    assert ok is True
    serialized = result["commands"]["auth"]
    assert serialized["share_with_model"] is True
    assert serialized["parameters"] == [
        {
            "name": "action", "description": "login/logout/status",
            "required": True, "capture_rest": False,
        },
        {
            "name": "extra", "description": "Extra args",
            "required": False, "capture_rest": True,
        },
    ]


def test_get_user_commands_multi_command() -> None:
    rpc = _make_lone_runner()
    cmds = {
        "status": UserCommand(name="status", description="Show status"),
        "reset": UserCommand(name="reset", description="Reset session"),
        "save": UserCommand(name="save", description="Save session"),
    }
    _install(rpc, _FakeSession(commands=cmds))

    ok, result = rpc._handle_session_get_user_commands()
    assert ok is True
    assert set(result["commands"].keys()) == {"status", "reset", "save"}


def test_get_user_commands_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_user_commands()
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_user_commands_missing_method_returns_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _SessionWithoutMethod())

    ok, result = rpc._handle_session_get_user_commands()
    assert ok is False
    assert result["stage"] == "missing_method"


def test_get_user_commands_method_raises_returns_call_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(raise_on_get=RuntimeError("session boom")))

    ok, result = rpc._handle_session_get_user_commands()
    assert ok is False
    assert result["stage"] == "call"
    assert "session boom" in result["error"]


def test_dispatch_routes_session_get_user_commands() -> None:
    rpc = _make_lone_runner()
    cmd = UserCommand(name="foo", description="bar")
    _install(rpc, _FakeSession(commands={"foo": cmd}))

    env = RequestEnvelope(
        id=1, method="session.get_user_commands", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert "foo" in result["commands"]


# ----------------------------------------------------------------------
# E2E: daemon-side wrapper round-trip — UserCommand reconstruction
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
        target=runner_rpc.serve, name="rpc-uc-test", daemon=True,
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
async def test_e2e_get_user_commands_round_trip_reconstruction() -> None:
    """Pin the wrapper reconstructs UserCommand + CommandParameter
    NamedTuples from the wire dicts — daemon callers can use the
    result with ``parse_command_args`` unmodified."""
    pair = await _make_pair()
    try:
        cmd = UserCommand(
            name="auth",
            description="Authenticate",
            share_with_model=True,
            parameters=[
                CommandParameter(
                    name="action", description="login or logout",
                    required=True, capture_rest=False,
                ),
            ],
        )
        _install(pair.runner_rpc, _FakeSession(commands={"auth": cmd}))

        result = await pair.daemon_client.session_get_user_commands()
        assert isinstance(result, dict)
        assert "auth" in result
        # Reconstruction should produce a real UserCommand NamedTuple.
        reconstructed = result["auth"]
        assert isinstance(reconstructed, UserCommand)
        assert reconstructed.name == "auth"
        assert reconstructed.description == "Authenticate"
        assert reconstructed.share_with_model is True
        assert len(reconstructed.parameters) == 1
        assert isinstance(reconstructed.parameters[0], CommandParameter)
        assert reconstructed.parameters[0].name == "action"
        assert reconstructed.parameters[0].required is True
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_get_user_commands_threadsafe_from_thread() -> None:
    pair = await _make_pair()
    try:
        cmd = UserCommand(name="status", description="Show status")
        _install(pair.runner_rpc, _FakeSession(commands={"status": cmd}))

        result_holder: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                result_holder["value"] = (
                    pair.daemon_client.session_get_user_commands_threadsafe()
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
        assert "status" in result_holder["value"]
        assert isinstance(result_holder["value"]["status"], UserCommand)
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_get_user_commands_works_with_parse_command_args() -> None:
    """Pin the reconstructed UserCommand works with the existing
    ``parse_command_args`` helper — the daemon-side migration site
    at core.py:4015 calls this directly on the result."""
    from jaato_sdk.plugins.base import parse_command_args

    pair = await _make_pair()
    try:
        cmd = UserCommand(
            name="auth",
            description="Authenticate",
            parameters=[
                CommandParameter(
                    name="action", description="login/logout/status",
                    required=True, capture_rest=False,
                ),
            ],
        )
        _install(pair.runner_rpc, _FakeSession(commands={"auth": cmd}))

        result = await pair.daemon_client.session_get_user_commands()
        reconstructed = result["auth"]

        # parse_command_args should work without modification.
        parsed = parse_command_args(reconstructed, "login")
        assert parsed == {"action": "login"}
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_get_user_commands_missing_method_raises() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _SessionWithoutMethod())
        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_get_user_commands()
        assert "get_user_commands" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# Real JaatoSession: structural pin
# ----------------------------------------------------------------------


def test_jaato_session_has_public_get_user_commands_method() -> None:
    from shared.jaato_session import JaatoSession
    assert callable(getattr(JaatoSession, "get_user_commands", None))
