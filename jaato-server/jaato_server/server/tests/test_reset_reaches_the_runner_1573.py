"""``reset`` clears the conversation it says it clears (#1573).

``reset`` ("Clear conversation history") was advertised by every client
and by the daemon's own help text, parsed into a server command, and sent
as ``CommandRequest("reset")`` -- and ``JaatoServer.execute_command``
looked it up among the runner's PLUGIN user commands, where nothing
registers it, so it answered ``Unknown command: reset``.  Meanwhile
``JaatoServer.clear_history``, which implements it, had no caller, and when
called it swallowed a runner failure and announced "History cleared"
anyway.

These tests drive the real dispatch -- ``SessionManager.handle_request``
with a ``CommandRequest`` -- into a real ``JaatoServer.execute_command``,
over a real :class:`RunnerRPCClient` / :class:`RunnerRPC` socketpair, to a
real :class:`JaatoSession` on the runner side.  A test that called
``clear_history()`` directly would have passed against the tree that had
the defect, whose whole problem was that nothing called it.
"""
from __future__ import annotations

import socket
import threading
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import (
    CommandRequest,
    ContextUpdatedEvent,
    InstructionBudgetEvent,
    SystemMessageEvent,
)
from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from jaato_server.server.core import AgentState, JaatoServer
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.runner.session import RunnerSessionHost
from jaato_server.server.runner_rpc_client import RunnerRPCClient
from jaato_server.server.session_manager import Session, SessionManager
from jaato_server.shared.instruction_budget import (
    InstructionBudget,
    InstructionSource,
)
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"

REVERSIONS = [
    Reversion(
        because="reset is not a built-in (the reported state)",
        target=_CORE,
        find="        builtin = self._BUILTIN_COMMANDS.get(command.lower())\n",
        replace="        builtin = None\n",
        test="test_reset_through_the_command_path_clears_the_runners_history",
    ),
    Reversion(
        because="the daemon resets under a running turn",
        target=_CORE,
        find=(
            "        if self._model_running:\n"
            "            return {\n"
            "                \"error\": (\n"
            "                    \"History not cleared: a turn"
        ),
        replace=(
            "        if False:\n"
            "            return {\n"
            "                \"error\": (\n"
            "                    \"History not cleared: a turn"
        ),
        test="test_a_running_turn_is_refused_and_nothing_changes",
    ),
    Reversion(
        because="the runner resets under a running turn",
        target=_RPC,
        find=(
            "        if bool(getattr(session, \"is_running\", False)):\n"
            "            return False, {\n"
            "                \"error\": (\n"
            "                    \"session.reset: a turn"
        ),
        replace=(
            "        if False:\n"
            "            return False, {\n"
            "                \"error\": (\n"
            "                    \"session.reset: a turn"
        ),
        test="test_the_runner_refuses_a_turn_the_daemon_has_not_seen",
    ),
    Reversion(
        because="a runner failure is reported as success",
        target=_CORE,
        find=(
            "            return {\"error\": f\"History not cleared: {exc}\", "
            "\"stage\": \"runner\"}\n"
        ),
        replace="            answer = None\n",
        test="test_a_failed_runner_reset_is_an_error_not_a_success",
    ),
    Reversion(
        because="the instruction budget keeps the old conversation",
        target=_RPC,
        find="            session._update_conversation_budget()\n",
        replace="            pass\n",
        test="test_the_budget_clients_are_sent_shows_the_emptied_window",
    ),
    Reversion(
        because="the session is not re-marked dirty after the command",
        target=_SM,
        find="            session.is_dirty = True\n            # Format result properly\n",
        replace="            pass\n            # Format result properly\n",
        test="test_a_save_during_the_reset_does_not_leave_it_unpersisted",
    ),
]


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------


class _Registry:
    def __init__(self) -> None:
        self.cleared = 0

    def broadcast_history_cleared(self) -> None:
        self.cleared += 1


def _real_session() -> JaatoSession:
    runtime = MagicMock()
    runtime.create_provider.return_value = MagicMock()
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.registry = None
    session = JaatoSession(runtime, "m")
    session.configure()
    # Attached after configure(): only the reset's broadcast is observed.
    runtime.registry = _Registry()
    session.reset_session([
        Message(role=Role.USER, parts=[Part(text="question " * 50)]),
        Message(role=Role.MODEL, parts=[Part(text="answer " * 50)]),
    ])
    budget = InstructionBudget(session_id="s1", agent_id="main")
    budget.set_entry(InstructionSource.CONVERSATION, 500)
    session._instruction_budget = budget
    return session


class _Runner:
    def __init__(self, workspace: Any) -> None:
        envelope = SessionInitEnvelope(
            session_id="s1", workspace_path=str(workspace),
            profile_name="p", provider_name="anthropic",
            model_name="m", plugins=[],
        )
        self.session = _real_session()
        daemon_sock, runner_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM)
        self.rpc = RunnerRPC(runner_sock, lambda n, a: (False, {}))
        self.rpc._session_host = RunnerSessionHost(
            envelope=envelope, runtime=None, session=self.session,
        )
        self.thread = threading.Thread(
            target=self.rpc.serve, name="rpc-1573", daemon=True)
        self.daemon_sock = daemon_sock
        self.client: Any = None

    async def start(self) -> None:
        self.thread.start()
        self.client = RunnerRPCClient(self.daemon_sock, runner_pid=0)
        await self.client.start()

    async def stop(self) -> None:
        self.rpc.shutdown()
        self.thread.join(timeout=2)
        if self.client is not None and not self.client._closed:
            self.client._closed = True
            if self.client._read_task is not None:
                self.client._read_task.cancel()


def _server(client: Any) -> JaatoServer:
    srv = JaatoServer.__new__(JaatoServer)
    main = AgentState(agent_id="main", name="main", agent_type="main")
    main.history = ["stale mirror"]
    for key, value in dict(
        registry=None, permission_plugin=None, _runner_rpc=client,
        _spawned_runner=None, _pool_manager_ref=None,
        _runner_ready=threading.Event(), _agents={"main": main},
        _session_id="s1", _main_agent_id="main", _model_running=False,
        _original_inputs=["hello"], _cached_context_limit=None,
        _session_env={}, _pricing_loaded=False, _workspace_path=None,
        _model_name="m", _model_provider="anthropic",
    ).items():
        setattr(srv, key, value)
    srv.emitted = []  # type: ignore[attr-defined]
    srv.emit = srv.emitted.append  # type: ignore[method-assign]
    return srv


class _Harness:
    def __init__(self, workspace: Any, server: JaatoServer) -> None:
        self.manager = SessionManager()
        self.to_client: List[Any] = []
        self.manager._emit_to_client = (  # type: ignore[method-assign]
            lambda cid, ev: self.to_client.append(ev))
        self.session = Session(
            session_id="s1", name="s1", server=server,
            created_at="2026-10-06T00:00:00",
            last_activity="2026-10-06T00:00:00",
            workspace_path=str(workspace),
        )
        self.manager._sessions["s1"] = self.session

    def send(self, command: str) -> None:
        self.manager.handle_request(
            "c1", "s1", CommandRequest(command=command, args=[]))

    def messages(self) -> List[SystemMessageEvent]:
        return [e for e in self.to_client if isinstance(e, SystemMessageEvent)]


@pytest.fixture
async def runner(tmp_path):
    r = _Runner(tmp_path)
    await r.start()
    try:
        yield r
    finally:
        await r.stop()


async def _drive(harness: _Harness, command: str) -> None:
    import asyncio
    # handle_request runs in the loop's executor on every daemon path;
    # the runner RPCs are *_threadsafe and must not run on the loop.
    await asyncio.get_running_loop().run_in_executor(
        None, harness.send, command)


def _history_len(runner: _Runner) -> int:
    return len(runner.session.get_history())


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


async def test_reset_through_the_command_path_clears_the_runners_history(
    runner, tmp_path,
) -> None:
    server = _server(runner.client)
    harness = _Harness(tmp_path, server)
    assert _history_len(runner) == 2
    await _drive(harness, "reset")
    assert _history_len(runner) == 0
    replies = harness.messages()
    assert [m.message for m in replies] == ["History cleared"]
    assert replies[0].style == "info"
    # Plugins heard it; the daemon-side mirror and save inputs went too.
    assert runner.session._runtime.registry.cleared == 1
    assert server._agents["main"].history == []
    assert server._original_inputs == []


async def test_a_running_turn_is_refused_and_nothing_changes(
    runner, tmp_path,
) -> None:
    server = _server(runner.client)
    server._model_running = True
    harness = _Harness(tmp_path, server)
    await _drive(harness, "reset")
    assert _history_len(runner) == 2
    [reply] = harness.messages()
    assert reply.style == "error"
    assert "a turn is running" in reply.message
    assert server._agents["main"].history == ["stale mirror"]


async def test_the_runner_refuses_a_turn_the_daemon_has_not_seen(
    runner, tmp_path,
) -> None:
    runner.session._is_running = True
    try:
        server = _server(runner.client)
        harness = _Harness(tmp_path, server)
        await _drive(harness, "reset")
    finally:
        runner.session._is_running = False
    assert _history_len(runner) == 2
    [reply] = harness.messages()
    assert reply.style == "error"
    assert "a turn is running" in reply.message


async def test_a_failed_runner_reset_is_an_error_not_a_success(
    runner, tmp_path, monkeypatch,
) -> None:
    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(runner.session, "reset_session", _boom)
    server = _server(runner.client)
    harness = _Harness(tmp_path, server)
    await _drive(harness, "reset")
    [reply] = harness.messages()
    assert reply.style == "error"
    assert "History not cleared" in reply.message
    assert "disk on fire" in reply.message
    assert server._agents["main"].history == ["stale mirror"]


async def test_the_budget_clients_are_sent_shows_the_emptied_window(
    runner, tmp_path,
) -> None:
    server = _server(runner.client)
    harness = _Harness(tmp_path, server)
    await _drive(harness, "reset")
    assert any(isinstance(e, ContextUpdatedEvent) for e in server.emitted)
    budgets = [e for e in server.emitted if isinstance(e, InstructionBudgetEvent)]
    assert budgets, "no InstructionBudgetEvent after reset"
    entries = budgets[-1].budget_snapshot["entries"]
    # The harness seeded 500 tokens of conversation; the reset must report 0.
    assert entries["conversation"]["total_tokens"] == 0, entries


async def test_a_save_during_the_reset_does_not_leave_it_unpersisted(
    runner, tmp_path, monkeypatch,
) -> None:
    server = _server(runner.client)
    harness = _Harness(tmp_path, server)
    real_reset = runner.session.reset_session

    def _reset_while_a_save_lands(*a: Any, **k: Any) -> None:
        # A save that read the history before this reset finishes now and
        # clears the dirty mark the request set before dispatch.
        harness.session.is_dirty = False
        real_reset(*a, **k)

    monkeypatch.setattr(runner.session, "reset_session", _reset_while_a_save_lands)
    await _drive(harness, "reset")
    assert _history_len(runner) == 0
    assert harness.session.is_dirty is True


def test_no_runner_is_refused_by_name() -> None:
    server = _server(None)
    result = server.execute_command("reset", [])
    assert result["stage"] == "no_runner"
    assert "History not cleared" in result["error"]
