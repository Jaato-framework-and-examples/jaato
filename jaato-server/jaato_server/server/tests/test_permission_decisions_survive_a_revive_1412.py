"""A runtime permission decision survives an unload, in the plugin that enforces it (#1412).

#706 made ``permissions allow|deny|default`` and ``always`` / ``never``
answers survive a detach: ``PermissionPlugin`` gained
``get_persistence_state()`` / ``restore_persistence_state()`` and the
session manager saved and restored them.  Through the DAEMON's registry.

On a runner-served session -- the default -- there are two
``PermissionPlugin`` objects and only the runner's decides anything; it is
the one every ``permissions`` command and every prompt answer mutates.  So:

* the save snapshotted the daemon's copy, which nothing had changed, got
  ``None``, and wrote nothing;
* the restore, when there was anything to restore, put it into the
  daemon's copy, and the runner's enforcer came back on the profile's
  ``ask``.

A reattach inside the #1106 unload grace kept the runner and lost nothing,
which is why it read as intermittent.  A real unload -- the grace expired,
a daemon restart, a ``session.wake`` from disk -- lost every decision, and
a lost ``never`` is worse than a lost grant.

These tests use a real :class:`RunnerRPCClient` over a socketpair to a
real :class:`RunnerRPC` on its own thread, with the enforcer built by the
bootstrap's own :func:`build_session_permission_plugin`.  Nothing fakes the
runner's plugin: the defect is about WHICH instance is used, and a stub of
it could not have the defect.
"""
from __future__ import annotations

import ast
import asyncio
import json
import pathlib
import socket
import threading
from typing import Any, Dict, List

import pytest

from jaato_sdk.events import PermissionStatusEvent
from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from jaato_server.server.core import JaatoServer
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.runner.session import (
    RunnerSessionHost,
    build_session_permission_plugin,
)
from jaato_server.server.runner_rpc_client import RunnerRPCClient
from jaato_server.server.session_manager import Session, SessionManager
from jaato_server.shared.plugins.permission.plugin import PermissionPlugin
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"


# ----------------------------------------------------------------------
# Harness: a real client, a real runner, the bootstrap's own enforcer
# ----------------------------------------------------------------------


class _Runtime:
    def __init__(self, plugin: Any) -> None:
        self.permission_plugin = plugin


class _RunnerSession:
    """The runner-side session: a history, the enforcer, the command path."""

    def __init__(self, plugin: Any) -> None:
        self._runtime = _Runtime(plugin)

    def get_history(self) -> List[Message]:
        return [Message(role=Role.USER, parts=[Part(text="hi")])]

    def get_history_raw(self) -> List[Message]:
        return self.get_history()

    def execute_user_command(self, name: str, args: Dict[str, Any]):
        assert name == "permissions"
        return self._runtime.permission_plugin.execute_permissions(args), False

    def close_session(self) -> None:
        pass


class _Runner:
    """One runner process: an RPC server on a thread, a fresh enforcer."""

    def __init__(self, workspace: pathlib.Path, session_id: str) -> None:
        envelope = SessionInitEnvelope(
            session_id=session_id, workspace_path=str(workspace),
            profile_name="p", provider_name="anthropic",
            model_name="m", plugins=[],
        )
        self.plugin = build_session_permission_plugin(envelope, str(workspace))
        daemon_sock, runner_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM)
        self.rpc = RunnerRPC(runner_sock, lambda n, a: (False, {}))
        self.rpc._session_host = RunnerSessionHost(
            envelope=envelope, runtime=None,
            session=_RunnerSession(self.plugin),
        )
        self.thread = threading.Thread(
            target=self.rpc.serve, name=f"rpc-1412-{session_id}", daemon=True)
        self.daemon_sock = daemon_sock
        self.client: Any = None

    def rebootstrap(self, session_id: str, workspace: pathlib.Path) -> None:
        """A pool slot handed to the NEXT session: same RPC, new host."""
        envelope = SessionInitEnvelope(
            session_id=session_id, workspace_path=str(workspace),
            profile_name="p", provider_name="anthropic",
            model_name="m", plugins=[],
        )
        self.plugin = build_session_permission_plugin(envelope, str(workspace))
        self.rpc._session_host = RunnerSessionHost(
            envelope=envelope, runtime=None,
            session=_RunnerSession(self.plugin),
        )

    async def start(self) -> None:
        self.thread.start()
        self.client = RunnerRPCClient(self.daemon_sock, runner_pid=0)
        await self.client.start()

    def server(self, session_id: str) -> JaatoServer:
        srv = JaatoServer.__new__(JaatoServer)
        for key, value in dict(
            registry=None, permission_plugin=None, _runner_rpc=self.client,
            _spawned_runner=None, _pool_manager_ref=None,
            _runner_ready=threading.Event(), _agents={},
            _session_id=session_id, _main_agent_id="main",
        ).items():
            setattr(srv, key, value)
        srv.emitted = []  # type: ignore[attr-defined]
        srv.emit = srv.emitted.append  # type: ignore[method-assign]
        return srv

    def command(self, *args: str) -> str:
        result, _ = self.client.session_execute_user_command_threadsafe(
            "permissions", {"args": list(args)})
        return str(result)

    async def stop(self) -> None:
        self.rpc.shutdown()
        self.thread.join(timeout=2)
        if self.client is not None and not self.client._closed:
            self.client._closed = True
            if self.client._read_task is not None:
                self.client._read_task.cancel()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _session(manager: SessionManager, server: Any, workspace) -> Session:
    session = Session(
        session_id="s1", name="s1", server=server,
        created_at="2026-09-29T00:00:00",
        last_activity="2026-09-29T00:00:00",
        workspace_path=str(workspace),
    )
    manager._sessions["s1"] = session
    return session


def _saved_plugin_states(manager: SessionManager, workspace) -> Dict[str, Any]:
    record = manager._session_storage_dir(str(workspace)) / "s1.json"
    return json.loads(record.read_text()).get("metadata", {}).get(
        "plugin_states", {})


def _decide(runner: _Runner) -> None:
    """The operator's decisions, through the real command path."""
    runner.command("default", "allow")
    runner.command("deny", "rm_everything")


def _status_defaults(server: Any) -> List[str]:
    return [e.effective_default for e in server.emitted
            if isinstance(e, PermissionStatusEvent)]


# ----------------------------------------------------------------------
# The reported failure: unload, revive onto a NEW runner
# ----------------------------------------------------------------------


async def test_the_save_asks_the_enforcer(workspace) -> None:
    """The record carries the decisions the RUNNER's plugin holds."""
    a = _Runner(workspace, "s1")
    await a.start()
    try:
        await asyncio.to_thread(_decide, a)
        manager = SessionManager()
        session = _session(manager, a.server("s1"), workspace)
        assert await asyncio.to_thread(manager._save_session, session)
        saved = _saved_plugin_states(manager, workspace).get("permission")
        assert saved is not None, (
            "the save wrote no permission snapshot: it read the daemon's "
            "copy, which no `permissions` command ever reaches")
        assert saved["session_default_policy"] == "allow"
        assert saved["session_blacklist"] == ["rm_everything"]
    finally:
        await a.stop()


async def test_decisions_reach_the_new_runners_enforcer(workspace) -> None:
    """Save on runner A, unload, revive on runner B: B enforces them."""
    a = _Runner(workspace, "s1")
    await a.start()
    manager = SessionManager()
    try:
        await asyncio.to_thread(_decide, a)
        session = _session(manager, a.server("s1"), workspace)
        assert await asyncio.to_thread(manager._save_session, session)
    finally:
        await a.stop()

    b = _Runner(workspace, "s1")
    await b.start()
    try:
        # The new runner starts on the profile, as a revive would.
        assert b.plugin.get_permission_status()["effective_default"] == "ask"
        server_b = b.server("s1")
        states = _saved_plugin_states(manager, workspace)
        known = await asyncio.to_thread(
            manager._restore_plugin_states, server_b, states)

        policy = b.plugin._policy
        assert policy.session_default_policy == "allow"
        assert "rm_everything" in policy.session_blacklist
        assert b.plugin.get_permission_status()["effective_default"] == "allow"
        # The status bar is told the restored default, as the enforcer holds it.
        assert _status_defaults(server_b) == ["allow"]
        # And the revived session knows it as its last known snapshot.
        assert known == states["permission"]
    finally:
        await b.stop()


async def test_a_failed_ask_keeps_the_last_snapshot(workspace) -> None:
    """A runner that does not answer must not erase a denial (#1355's rule)."""
    a = _Runner(workspace, "s1")
    await a.start()
    try:
        await asyncio.to_thread(_decide, a)
        manager = SessionManager()
        session = _session(manager, a.server("s1"), workspace)
        assert await asyncio.to_thread(manager._save_session, session)

        def refuse(*_a, **_k):
            raise RuntimeError("runner went quiet")

        a.client.session_get_permission_persistence_threadsafe = refuse
        assert await asyncio.to_thread(manager._save_session, session)
        saved = _saved_plugin_states(manager, workspace).get("permission")
        assert saved is not None and saved["session_blacklist"] == [
            "rm_everything"]
    finally:
        await a.stop()


async def test_an_answered_nothing_clears_the_snapshot(workspace) -> None:
    """`permissions clear` is a decision too: an answered None is written."""
    a = _Runner(workspace, "s1")
    await a.start()
    try:
        await asyncio.to_thread(_decide, a)
        manager = SessionManager()
        session = _session(manager, a.server("s1"), workspace)
        assert await asyncio.to_thread(manager._save_session, session)
        await asyncio.to_thread(a.command, "clear")
        assert await asyncio.to_thread(manager._save_session, session)
        assert "permission" not in _saved_plugin_states(manager, workspace)
    finally:
        await a.stop()


async def test_a_pool_slot_does_not_carry_decisions_to_the_next_session(
        workspace, tmp_path) -> None:
    """Same runner, next session: its enforcer is new and holds nothing."""
    a = _Runner(workspace, "s1")
    await a.start()
    try:
        await asyncio.to_thread(_decide, a)
        first = a.plugin
        other_ws = tmp_path / "ws2"
        other_ws.mkdir()
        a.rebootstrap("s2", other_ws)
        assert a.plugin is not first
        state = await asyncio.to_thread(
            a.client.session_get_permission_persistence_threadsafe)
        assert state is None
        assert a.plugin.get_permission_status()["effective_default"] == "ask"
    finally:
        await a.stop()


def test_the_bootstrap_builds_a_new_enforcer_each_time(tmp_path) -> None:
    """The freshness the no-leak property rests on."""
    env = SessionInitEnvelope(
        session_id="x", workspace_path=str(tmp_path), profile_name="p",
        provider_name="anthropic", model_name="m", plugins=[],
    )
    one = build_session_permission_plugin(env, str(tmp_path))
    two = build_session_permission_plugin(env, str(tmp_path))
    assert isinstance(one, PermissionPlugin) and one is not two


def test_load_restores_through_the_helper_and_keeps_its_answer() -> None:
    """``_load_session_impl`` routes plugin states through the helper that
    knows about the runner, and hands its answer to the revived Session.
    Read from the source: driving a whole revive would spawn a runner."""
    tree = ast.parse(pathlib.Path(_SM).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_load_session_impl")
    text = ast.unparse(fn)
    assert "self._restore_plugin_states(" in text
    assert "permission_state=restored_permission_state" in text


# ----------------------------------------------------------------------
# Control: with no runner the daemon's plugin IS the enforcer
# ----------------------------------------------------------------------


class _Registry:
    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin

    def list_exposed(self) -> List[str]:
        return ["permission"]

    def get_plugin(self, name: str) -> Any:
        return self._plugin if name == "permission" else None


def _daemon_local_server(plugin: Any) -> Any:
    srv = JaatoServer.__new__(JaatoServer)
    srv.registry = _Registry(plugin)
    srv.permission_plugin = plugin
    srv._runner_rpc = None
    return srv


def test_daemon_local_saves_and_restores_its_own_plugin(tmp_path) -> None:
    env = SessionInitEnvelope(
        session_id="x", workspace_path=str(tmp_path), profile_name="p",
        provider_name="anthropic", model_name="m", plugins=[],
    )
    plugin = build_session_permission_plugin(env, str(tmp_path))
    plugin.execute_permissions({"args": ["deny", "rm_everything"]})
    manager = SessionManager()
    session = Session(session_id="s", name="s",
                      server=_daemon_local_server(plugin),
                      created_at="2026-09-29T00:00:00")
    states = manager._collect_plugin_states(session)
    assert states["permission"]["session_blacklist"] == ["rm_everything"]

    fresh = build_session_permission_plugin(env, str(tmp_path))
    manager._restore_plugin_states(_daemon_local_server(fresh), states)
    assert "rm_everything" in fresh._policy.session_blacklist


# ----------------------------------------------------------------------
# The runner-side verbs
# ----------------------------------------------------------------------


def test_restore_refuses_a_state_that_is_not_a_dict() -> None:
    rpc = RunnerRPC.__new__(RunnerRPC)
    ok, payload = rpc._handle_session_restore_permission_persistence(
        {"state": None})
    assert not ok and payload["stage"] == "args"


REVERSIONS = [
    Reversion(
        target=_SM,
        find="""        if runner_served:
            pstate = self._runner_permission_state(session)
            if pstate:
                plugin_states["permission"] = pstate
        return plugin_states""",
        replace="""        return plugin_states""",
        test="test_the_save_asks_the_enforcer",
        because=(
            "the save reading only the daemon's copy, which no `permissions` "
            "command reaches, so a runner-served session saves nothing"
        ),
    ),
    Reversion(
        target=_SM,
        find="""            if runner_served and plugin_name == "permission":
                self._restore_runner_permission_state(server, plugin_state)
                continue""",
        replace="",
        test="test_decisions_reach_the_new_runners_enforcer",
        because=(
            "the restore putting the decisions into the daemon's copy, so "
            "the revived runner's enforcer comes back on the profile's ask"
        ),
    ),
    Reversion(
        target=_SM,
        find="""            return session.permission_state
        session.permission_state = state
        return state""",
        replace="""            return None
        session.permission_state = state
        return state""",
        test="test_a_failed_ask_keeps_the_last_snapshot",
        because=(
            "a runner that did not answer being saved as 'nothing decided', "
            "which erases a denial from the record"
        ),
    ),
    Reversion(
        target=_SM,
        find="""        emit_status = getattr(server, "emit_permission_status", None)
        if callable(emit_status):
            emit_status()""",
        replace="",
        test="test_decisions_reach_the_new_runners_enforcer",
        because=(
            "the status bar never being told the restored default, so it "
            "shows the profile's ask while the enforcer allows"
        ),
    ),
    Reversion(
        target=_RPC,
        find="""        if not isinstance(state, dict):
            return False, {
                "error": (
                    "session.restore_permission_persistence: 'state' must be "
                    f"a dict, got {type(state).__name__}"
                ),
                "stage": "args",
            }""",
        replace="",
        test="test_restore_refuses_a_state_that_is_not_a_dict",
        because="a malformed snapshot reaching the enforcer's restore",
    ),
]
