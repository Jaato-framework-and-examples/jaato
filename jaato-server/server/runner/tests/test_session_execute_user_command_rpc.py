"""Tests for the session.execute_user_command RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.4.5c.3).

Invokes a user command on the runner-side session.  Replaces the
daemon-side reach into ``self._jaato.execute_user_command(name,
args)`` (core.py:4044).

Wire shape per the §7c step 6.6.4.5c.3 audit decision: **Path A**
(per-type reconstruction) bounded to 3 result shapes.  Pre-
implementation grep verified the daemon does structured access
on ``HelpLines.lines`` (line 4073) and dict keys for the "model"
command (line 4078) + IPC-return fallback (line 4099).  Path B
(stringify-pre-wire) was killed by that grep — it would have
broken the structured-access invariants.

Wire format::

    {"_kind": "HelpLines", "lines": [[text, style], ...]}
    {"_kind": "dict", "value": <json-dict>}
    {"_kind": "str", "value": <str>}

Plus a separate ``shared: bool`` field for the second half of the
``tuple[Any, bool]`` return.

The handler callable stays runner-side; daemon dispatches a
name-keyed invocation per the user's pre-greenlit "invoke-by-name"
pattern.  No callable crosses the wire either direction.

Tests pin (one regression-pin per return type as the audit
prescribed):

- happy paths: HelpLines / dict / str / None / int (other-coerced)
- shared=True / shared=False round-trip
- HelpLines: lines list with mixed text/style tuples reconstructs
  correctly (re-tupled on receipt — wire flattens to lists)
- dict: nested values preserved
- arg validation: missing/non-str name, non-dict args
- no_host / no_session / missing_method / call error paths
- dispatch routing
- e2e via async + threadsafe wrappers
- e2e: HelpLines reconstruction works with the daemon-side
  `isinstance(result, HelpLines)` check at core.py:4073
- e2e: dict reconstruction works with the `result.get("success")`
  pattern at core.py:4078
- structural pin: real JaatoSession exposes the public method
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, Optional, Tuple

import pytest

from jaato_sdk.plugins.base import HelpLines
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
        session_id="sess-exec-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    """Stand-in returning configurable result + shared flag."""

    def __init__(
        self,
        result: Any = None,
        shared: bool = False,
        raise_on_call: Optional[Exception] = None,
    ) -> None:
        self._result = result
        self._shared = shared
        self._raise = raise_on_call
        self.calls: list = []

    def execute_user_command(
        self, name: str, args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, bool]:
        self.calls.append((name, dict(args or {})))
        if self._raise is not None:
            raise self._raise
        return self._result, self._shared


class _SessionWithoutMethod:
    pass


def _install(rpc: RunnerRPC, session: Any) -> None:
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )


# ----------------------------------------------------------------------
# Direct handler tests — per-type result serialization
# ----------------------------------------------------------------------


def test_execute_user_command_helplines_result_tagged() -> None:
    """HelpLines result serialized with _kind="HelpLines" and
    list-of-lists for the lines payload."""
    rpc = _make_lone_runner()
    helplines = HelpLines(lines=[
        ("Title", "bold"),
        ("Description text", ""),
        ("Note", "dim"),
    ])
    _install(rpc, _FakeSession(result=helplines, shared=False))

    ok, result = rpc._handle_session_execute_user_command(
        {"name": "auth", "args": {"action": "help"}},
    )
    assert ok is True
    assert result["shared"] is False
    assert result["result"] == {
        "_kind": "HelpLines",
        "lines": [
            ["Title", "bold"],
            ["Description text", ""],
            ["Note", "dim"],
        ],
    }


def test_execute_user_command_dict_result_tagged() -> None:
    """dict result tagged with _kind="dict" — covers the "model"
    command's ``{"success": ..., "current_model": ...}`` shape."""
    rpc = _make_lone_runner()
    dict_result = {"success": True, "current_model": "claude-opus-4-7"}
    _install(rpc, _FakeSession(result=dict_result, shared=False))

    ok, result = rpc._handle_session_execute_user_command(
        {"name": "model", "args": {"action": "select"}},
    )
    assert ok is True
    assert result["result"] == {
        "_kind": "dict",
        "value": {"success": True, "current_model": "claude-opus-4-7"},
    }


def test_execute_user_command_str_result_tagged() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(result="Authentication successful", shared=True))

    ok, result = rpc._handle_session_execute_user_command(
        {"name": "auth", "args": {"action": "login"}},
    )
    assert ok is True
    assert result["shared"] is True
    assert result["result"] == {
        "_kind": "str", "value": "Authentication successful",
    }


def test_execute_user_command_none_result_coerced_to_empty_string() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(result=None, shared=False))

    ok, result = rpc._handle_session_execute_user_command(
        {"name": "noop", "args": {}},
    )
    assert ok is True
    assert result["result"] == {"_kind": "str", "value": ""}


def test_execute_user_command_int_result_stringified() -> None:
    """Non-dict / non-HelpLines / non-str value coerces via str()."""
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(result=42, shared=False))

    ok, result = rpc._handle_session_execute_user_command(
        {"name": "count", "args": {}},
    )
    assert ok is True
    assert result["result"] == {"_kind": "str", "value": "42"}


def test_execute_user_command_shared_flag_round_trip() -> None:
    """Both shared=True and shared=False propagate."""
    rpc = _make_lone_runner()

    for shared in (True, False):
        _install(rpc, _FakeSession(result="ok", shared=shared))
        ok, result = rpc._handle_session_execute_user_command(
            {"name": "x", "args": {}},
        )
        assert ok is True
        assert result["shared"] is shared


def test_execute_user_command_method_invoked_with_name_and_args() -> None:
    """Verify the runner invokes session.execute_user_command(name,
    args) with the exact values from the wire envelope."""
    rpc = _make_lone_runner()
    session = _FakeSession(result="ok")
    _install(rpc, session)

    rpc._handle_session_execute_user_command(
        {"name": "auth", "args": {"action": "login", "extra": "foo"}},
    )
    assert session.calls == [
        ("auth", {"action": "login", "extra": "foo"}),
    ]


# ----------------------------------------------------------------------
# Arg validation
# ----------------------------------------------------------------------


def test_execute_user_command_rejects_missing_name() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(result="ok"))

    ok, result = rpc._handle_session_execute_user_command({"args": {}})
    assert ok is False
    assert result["stage"] == "decode"


def test_execute_user_command_rejects_non_str_name() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(result="ok"))

    for bad in [42, None, ["list"], {"dict": True}, ""]:
        ok, result = rpc._handle_session_execute_user_command(
            {"name": bad, "args": {}},
        )
        assert ok is False, f"should reject name={bad!r}"
        assert result["stage"] == "decode"


def test_execute_user_command_rejects_non_dict_args() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(result="ok"))

    for bad in [["list"], "string", 42]:
        ok, result = rpc._handle_session_execute_user_command(
            {"name": "x", "args": bad},
        )
        assert ok is False, f"should reject args={bad!r}"
        assert result["stage"] == "decode"


def test_execute_user_command_omitted_args_treated_as_empty() -> None:
    """args is optional; missing/None coerces to {}."""
    rpc = _make_lone_runner()
    session = _FakeSession(result="ok")
    _install(rpc, session)

    ok, _ = rpc._handle_session_execute_user_command({"name": "x"})
    assert ok is True
    assert session.calls[-1] == ("x", {})


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_execute_user_command_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_execute_user_command(
        {"name": "x", "args": {}},
    )
    assert ok is False
    assert result["stage"] == "no_host"


def test_execute_user_command_missing_method_returns_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _SessionWithoutMethod())

    ok, result = rpc._handle_session_execute_user_command(
        {"name": "x", "args": {}},
    )
    assert ok is False
    assert result["stage"] == "missing_method"


def test_execute_user_command_method_raises_returns_call_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(raise_on_call=ValueError("bad command")))

    ok, result = rpc._handle_session_execute_user_command(
        {"name": "x", "args": {}},
    )
    assert ok is False
    assert result["stage"] == "call"
    assert "bad command" in result["error"]


def test_dispatch_routes_session_execute_user_command() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(result="ok", shared=True))

    env = RequestEnvelope(
        id=1, method="session.execute_user_command",
        args={"name": "x", "args": {}},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result["shared"] is True


# ----------------------------------------------------------------------
# E2E: daemon-side wrapper round-trip — per-type reconstruction
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
        target=runner_rpc.serve, name="rpc-exec-test", daemon=True,
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
async def test_e2e_helplines_reconstruction_works_with_isinstance() -> None:
    """Pin the wrapper reconstructs a real HelpLines instance — the
    daemon-side migration site at core.py:4073 does
    ``isinstance(result, HelpLines)`` to detect pager-display
    commands.  Reconstruction must produce a real HelpLines."""
    pair = await _make_pair()
    try:
        helplines = HelpLines(lines=[
            ("Auth Help", "bold"),
            ("", ""),
            ("Usage: /auth <action>", ""),
        ])
        _install(pair.runner_rpc, _FakeSession(result=helplines, shared=False))

        result, shared = await pair.daemon_client.session_execute_user_command(
            "auth", {"action": "help"},
        )
        # Reconstruction produces a real HelpLines NamedTuple.
        assert isinstance(result, HelpLines)
        assert shared is False
        # Lines re-tupled (wire flattens to lists; reconstruction
        # restores the List[tuple] contract).
        assert result.lines == [
            ("Auth Help", "bold"),
            ("", ""),
            ("Usage: /auth <action>", ""),
        ]
        # Each line entry is a tuple, not a list.
        for entry in result.lines:
            assert isinstance(entry, tuple)
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_dict_reconstruction_works_with_get_pattern() -> None:
    """Pin the wrapper preserves dict shape — the daemon-side
    migration site at core.py:4078 does
    ``isinstance(result, dict)`` + ``result.get("success")`` /
    ``result["current_model"]`` for the "model" command."""
    pair = await _make_pair()
    try:
        model_result = {
            "success": True,
            "current_model": "claude-opus-4-7",
            "previous_model": "claude-sonnet-4-6",
        }
        _install(pair.runner_rpc, _FakeSession(result=model_result, shared=False))

        result, shared = await pair.daemon_client.session_execute_user_command(
            "model", {"action": "select", "model": "claude-opus-4-7"},
        )
        # Reconstruction preserves dict shape.
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert result["current_model"] == "claude-opus-4-7"
        assert result["previous_model"] == "claude-sonnet-4-6"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_str_reconstruction_returns_string() -> None:
    pair = await _make_pair()
    try:
        _install(
            pair.runner_rpc, _FakeSession(result="Logged in", shared=True),
        )
        result, shared = await pair.daemon_client.session_execute_user_command(
            "auth", {"action": "login"},
        )
        assert result == "Logged in"
        assert shared is True
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_threadsafe_from_thread() -> None:
    """The threadsafe wrapper is the path daemon callers use from
    the IPC handler thread."""
    pair = await _make_pair()
    try:
        helplines = HelpLines(lines=[("Hello", "")])
        _install(pair.runner_rpc, _FakeSession(result=helplines, shared=False))

        result_holder: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                result_holder["value"] = (
                    pair.daemon_client
                    .session_execute_user_command_threadsafe(
                        "x", {"foo": "bar"},
                    )
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
        result, shared = result_holder["value"]
        assert isinstance(result, HelpLines)
        assert shared is False
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_missing_method_raises() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _SessionWithoutMethod())
        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_execute_user_command(
                "x", {},
            )
        assert "execute_user_command" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# Real JaatoSession: structural pin
# ----------------------------------------------------------------------


def test_jaato_session_has_public_execute_user_command_method() -> None:
    from shared.jaato_session import JaatoSession
    assert callable(getattr(JaatoSession, "execute_user_command", None))
