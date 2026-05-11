"""Tests for the session.get_model_completions RPC handler +
daemon-side wrapper (Phase 3 §7c step 6.6.4.5c.4).

Gets completion candidates for the "model" command's subcommand
arguments.  Replaces 2 daemon-side reaches into
``self._jaato.get_model_completions(args)``:

- ``core.py:4285`` — model-name list for the toolbar
- ``command_router.py:1149`` — model-subcommand expansion for the
  IPC completion catalog (model + subcommand-name pairs)

Wire shape per the §7c step 6.6.4.5c.4 audit decision: **dict-
shape-only** (Path A, mirror of 5c.2's UserCommand pattern).
``CommandCompletion`` is a NamedTuple with primitive fields only
(``value: str``, ``description: str``).

Mid-implementation audit catch: 5c.0 inventory only counted the
core.py site; cross-grep surfaced the command_router site that
reads BOTH ``.value`` AND ``.description``.  Without 5c.4
migration, command_router's model-subcommand expansion would
silently stop working post-5e field removal.

Tests pin (matching the 5c.2 cadence: ~12-15 tests):

- happy paths: empty args / ["select"] / multi-element args
- defensive: empty list returned by underlying method
- arg validation: non-list args
- omitted args coerced to []
- no_host / no_session / missing_method / call error paths
- dispatch routing
- e2e: CommandCompletion reconstruction preserves .value AND
  .description attr access (the command_router site reads both)
- e2e via async + threadsafe wrappers
- structural pin: real JaatoSession exposes the public method
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.plugins.base import CommandCompletion
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
        session_id="sess-mc-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakeSession:
    def __init__(
        self,
        completions: Optional[List[CommandCompletion]] = None,
        raise_on_call: Optional[Exception] = None,
    ) -> None:
        self._completions = completions or []
        self._raise = raise_on_call
        self.calls: List[List[str]] = []

    def get_model_completions(self, args: List[str]) -> List[CommandCompletion]:
        self.calls.append(list(args))
        if self._raise is not None:
            raise self._raise
        return list(self._completions)


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


def test_get_model_completions_empty_args_returns_subcommands() -> None:
    """args=[] is the "show subcommands" path
    (list/select/help)."""
    rpc = _make_lone_runner()
    subcommands = [
        CommandCompletion(value="list", description="Show models"),
        CommandCompletion(value="select", description="Switch model"),
        CommandCompletion(value="help", description="Show help"),
    ]
    _install(rpc, _FakeSession(completions=subcommands))

    ok, result = rpc._handle_session_get_model_completions({"args": []})
    assert ok is True
    assert result == {
        "completions": [
            {"value": "list", "description": "Show models"},
            {"value": "select", "description": "Switch model"},
            {"value": "help", "description": "Show help"},
        ],
    }


def test_get_model_completions_select_args_returns_model_names() -> None:
    rpc = _make_lone_runner()
    models = [
        CommandCompletion(value="claude-opus-4-7", description="Opus 4.7"),
        CommandCompletion(value="claude-sonnet-4-6", description="Sonnet 4.6"),
    ]
    _install(rpc, _FakeSession(completions=models))

    ok, result = rpc._handle_session_get_model_completions(
        {"args": ["select"]},
    )
    assert ok is True
    assert result["completions"] == [
        {"value": "claude-opus-4-7", "description": "Opus 4.7"},
        {"value": "claude-sonnet-4-6", "description": "Sonnet 4.6"},
    ]


def test_get_model_completions_empty_list_returned() -> None:
    """Underlying method returns [] (no matches)."""
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(completions=[]))

    ok, result = rpc._handle_session_get_model_completions({"args": []})
    assert ok is True
    assert result == {"completions": []}


def test_get_model_completions_method_invoked_with_args() -> None:
    """Verify the runner passes the args list through to
    session.get_model_completions verbatim."""
    rpc = _make_lone_runner()
    session = _FakeSession(completions=[])
    _install(rpc, session)

    rpc._handle_session_get_model_completions(
        {"args": ["select", "claude"]},
    )
    assert session.calls == [["select", "claude"]]


def test_get_model_completions_omitted_args_coerced_to_empty() -> None:
    """args is optional; missing/None coerces to []."""
    rpc = _make_lone_runner()
    session = _FakeSession(completions=[])
    _install(rpc, session)

    ok, _ = rpc._handle_session_get_model_completions({})
    assert ok is True
    assert session.calls == [[]]


def test_get_model_completions_args_coerced_to_str() -> None:
    """Non-str entries in args are coerced via str() — defensive
    against IPC-layer type drift."""
    rpc = _make_lone_runner()
    session = _FakeSession(completions=[])
    _install(rpc, session)

    rpc._handle_session_get_model_completions(
        {"args": ["select", 42]},
    )
    assert session.calls == [["select", "42"]]


def test_get_model_completions_rejects_non_list_args() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(completions=[]))

    for bad in [{"dict": True}, "string", 42]:
        ok, result = rpc._handle_session_get_model_completions(
            {"args": bad},
        )
        assert ok is False, f"should reject args={bad!r}"
        assert result["stage"] == "decode"


def test_get_model_completions_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_get_model_completions({"args": []})
    assert ok is False
    assert result["stage"] == "no_host"


def test_get_model_completions_missing_method_returns_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _SessionWithoutMethod())

    ok, result = rpc._handle_session_get_model_completions({"args": []})
    assert ok is False
    assert result["stage"] == "missing_method"


def test_get_model_completions_method_raises_returns_call_error() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(raise_on_call=RuntimeError("provider down")))

    ok, result = rpc._handle_session_get_model_completions({"args": []})
    assert ok is False
    assert result["stage"] == "call"
    assert "provider down" in result["error"]


def test_dispatch_routes_session_get_model_completions() -> None:
    rpc = _make_lone_runner()
    _install(rpc, _FakeSession(completions=[
        CommandCompletion(value="x", description="y"),
    ]))

    env = RequestEnvelope(
        id=1, method="session.get_model_completions",
        args={"args": []},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result["completions"] == [{"value": "x", "description": "y"}]


# ----------------------------------------------------------------------
# E2E: daemon-side wrapper round-trip — CommandCompletion reconstruction
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
        target=runner_rpc.serve, name="rpc-mc-test", daemon=True,
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
async def test_e2e_completion_reconstruction_preserves_both_fields() -> None:
    """Pin the wrapper reconstructs CommandCompletion NamedTuples
    that preserve both ``.value`` AND ``.description`` attr access —
    the command_router.py:1149 migration site reads both fields::

        commands.append({
            "name": f"model {sub.value}",
            "description": sub.description or "",
        })
    """
    pair = await _make_pair()
    try:
        completions = [
            CommandCompletion(value="list", description="Show models"),
            CommandCompletion(value="select", description="Switch model"),
        ]
        _install(pair.runner_rpc, _FakeSession(completions=completions))

        result = await pair.daemon_client.session_get_model_completions([])
        assert isinstance(result, list)
        assert len(result) == 2
        # Each entry is a real CommandCompletion NamedTuple.
        for entry in result:
            assert isinstance(entry, CommandCompletion)
        # Both fields preserved (mirrors command_router.py reads).
        assert result[0].value == "list"
        assert result[0].description == "Show models"
        assert result[1].value == "select"
        assert result[1].description == "Switch model"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_value_only_consumption_works() -> None:
    """Pin the wrapper supports ``.value``-only consumption — the
    core.py:4285 migration site reads only ``.value`` for the
    model-name list."""
    pair = await _make_pair()
    try:
        models = [
            CommandCompletion(value="opus", description="Powerful"),
            CommandCompletion(value="sonnet", description="Balanced"),
        ]
        _install(pair.runner_rpc, _FakeSession(completions=models))

        result = await pair.daemon_client.session_get_model_completions(
            ["select"],
        )
        names = [c.value for c in result]
        assert names == ["opus", "sonnet"]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_threadsafe_from_thread() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _FakeSession(completions=[
            CommandCompletion(value="x", description="y"),
        ]))

        result_holder: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                result_holder["value"] = (
                    pair.daemon_client
                    .session_get_model_completions_threadsafe(["select"])
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
        assert len(result_holder["value"]) == 1
        assert result_holder["value"][0].value == "x"
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_missing_method_raises() -> None:
    pair = await _make_pair()
    try:
        _install(pair.runner_rpc, _SessionWithoutMethod())
        with pytest.raises(RunnerCallError) as exc_info:
            await pair.daemon_client.session_get_model_completions([])
        assert "get_model_completions" in str(exc_info.value)
    finally:
        await _teardown(pair)


# ----------------------------------------------------------------------
# Real JaatoSession: structural pin
# ----------------------------------------------------------------------


def test_jaato_session_has_public_get_model_completions_method() -> None:
    from shared.jaato_session import JaatoSession
    assert callable(getattr(JaatoSession, "get_model_completions", None))
