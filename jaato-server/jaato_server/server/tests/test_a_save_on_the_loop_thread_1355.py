"""A session save or shutdown driven from the daemon loop reaches the runner (#1355).

Every ``*_threadsafe`` wrapper on :class:`RunnerRPCClient` schedules a
coroutine onto the daemon loop and BLOCKS until it finishes.  Called from
that loop's own thread it waits for work only that thread could do, so the
client refuses it (``_guard_not_on_loop``, #631).  The daemon's exit path
called ``SessionManager.shutdown()`` directly inside its ``async def
start``, so on every restart:

* the save's ``session_get_history_threadsafe`` was refused, the save
  returned ``False`` and the session's state was lost;
* ``session.end`` / ``session.shutdown`` were refused, so pool slots fell
  through to a cold close;
* ``rpc.close()`` was waited on with a bare ``run_coroutine_threadsafe``
  that nothing guarded, stalling the loop the full 10 s per session.

The same class was reachable on a LIVE daemon: the IPC and WS
``CommandListRequest`` handlers, the WS workspace inspect/delete handlers
and the standalone WS handlers ran sync code that lists sessions and asks
every runner for its history and commands.

These tests use a real :class:`RunnerRPCClient` over a socketpair to a
real :class:`RunnerRPC` serving on its own thread, and drive the code from
inside a running loop, because the defect is a relationship between two
threads and a stub that answers synchronously cannot have it.
"""
from __future__ import annotations

import ast
import asyncio
import gc
import json
import pathlib
import socket
import threading
import time
import warnings
from typing import Any, List

import pytest

from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from jaato_server.server.core import JaatoServer
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.runner.session import RunnerSessionHost
from jaato_server.server.runner_rpc_client import RunnerRPCClient
from jaato_server.server.session_manager import Session, SessionManager
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"
_CORE = "jaato-server/jaato_server/server/core.py"
_MAIN = "jaato-server/jaato_server/server/__main__.py"
_IPC = "jaato-server/jaato_server/server/ipc.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find="""        await asyncio.to_thread(self.stop_lifetime_watchdog)
        await asyncio.to_thread(self.shutdown)""",
        replace="""        self.stop_lifetime_watchdog()
        self.shutdown()""",
        because="the daemon's exit path saves on the loop thread again, so "
                "the history fetch is refused and the session is not saved",
        test="test_daemon_shutdown_saves_a_dirty_session",
    ),
    Reversion(
        target=_SM,
        find="""        if getattr(server, "_runner_released", False):""",
        replace="""        if False:""",
        because="a save after the runner was released writes an empty "
                "history over the good record",
        test="test_a_save_after_the_runner_was_released_keeps_the_record",
    ),
    Reversion(
        target=_CORE,
        find="""            if _on_loop_thread(loop):""",
        replace="""            if False:""",
        because="a shutdown that still reaches the close on the loop "
                "thread stalls it for the full timeout",
        test="test_a_close_on_the_loop_thread_does_not_stall_it",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_rpc_client.py",
        find="""        self._guard_not_on_loop("_run_threadsafe", coro)""",
        replace="""        self._guard_not_on_loop("_run_threadsafe")""",
        because="a refused call leaks its coroutine as a 'never awaited' "
                "RuntimeWarning beside the real error",
        test="test_a_refused_call_leaks_no_coroutine",
    ),
    Reversion(
        target=_IPC,
        find="""                commands = await asyncio.to_thread(
                    self._on_command_list_request)""",
        replace="""                commands = self._on_command_list_request()""",
        because="the IPC command-list handler asks every runner for its "
                "history on the loop thread again",
        test="test_no_async_def_calls_a_runner_blocking_entry_point",
    ),
]


# ----------------------------------------------------------------------
# Harness: a real client talking to a real runner on its own thread
# ----------------------------------------------------------------------


class _RunnerSession:
    """The runner-side session: a history and a close hook, nothing else."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def get_history(self) -> List[Message]:
        return [Message(role=Role.USER, parts=[Part(text="keep me")])]

    def get_history_raw(self) -> List[Message]:
        return self.get_history()

    def close_session(self) -> None:
        self.closed.set()


class _Harness:
    def __init__(self, tmp_path: pathlib.Path) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir()
        self.runner_session = _RunnerSession()
        daemon_sock, runner_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM)
        self.runner = RunnerRPC(runner_sock, lambda n, a: (False, {}))
        self.runner._session_host = RunnerSessionHost(
            envelope=SessionInitEnvelope(
                session_id="s1", workspace_path=str(self.workspace),
                profile_name="p", provider_name="anthropic",
                model_name="m", plugins=[],
            ),
            runtime=None,
            session=self.runner_session,
        )
        self.thread = threading.Thread(
            target=self.runner.serve, name="rpc-1355", daemon=True)
        self.daemon_sock = daemon_sock
        self.client: Any = None
        self.server: Any = None
        self.manager: Any = None
        self.session: Any = None

    async def start(self) -> None:
        self.thread.start()
        self.client = RunnerRPCClient(self.daemon_sock, runner_pid=0)
        await self.client.start()
        srv = JaatoServer.__new__(JaatoServer)
        for key, value in dict(
            registry=None, permission_plugin=None, _runner_rpc=self.client,
            _spawned_runner=None, _pool_manager_ref=None,
            _runner_ready=threading.Event(), _agents={}, _session_id="s1",
            _main_agent_id="main",
        ).items():
            setattr(srv, key, value)
        self.server = srv
        self.manager = SessionManager()
        self.session = Session(
            session_id="s1", name="s1", server=srv,
            created_at="2026-09-28T00:00:00",
            last_activity="2026-09-28T00:00:00",
            workspace_path=str(self.workspace),
        )
        self.manager._sessions["s1"] = self.session

    def record(self) -> pathlib.Path:
        return self.manager._session_storage_dir(str(self.workspace)) / "s1.json"

    def saved_history(self) -> list:
        return json.loads(self.record().read_text()).get("history", [])

    async def stop(self) -> None:
        self.runner.shutdown()
        self.thread.join(timeout=2)
        if self.client is not None and not self.client._closed:
            self.client._closed = True
            if self.client._read_task is not None:
                self.client._read_task.cancel()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return _Harness(tmp_path)


# ----------------------------------------------------------------------
# The reported failure, and the entry point that fixes it
# ----------------------------------------------------------------------


async def test_daemon_shutdown_saves_a_dirty_session(harness) -> None:
    """The daemon's exit path, driven from its loop, writes the session."""
    await harness.start()
    try:
        await asyncio.wait_for(harness.manager.shutdown_from_loop(), 30)
        assert harness.record().exists(), "the session was not saved"
        assert len(harness.saved_history()) == 1
        # And session.shutdown reached the runner, rather than being
        # refused and falling through to the transport close.
        assert harness.runner_session.closed.is_set()
    finally:
        await harness.stop()


async def test_the_daemon_exit_path_uses_the_off_loop_shutdown() -> None:
    """``JaatoDaemon.start`` awaits ``shutdown_from_loop``; it does not call
    ``shutdown()`` itself.  Read from the source, since driving the whole
    daemon here would test everything but this line."""
    src = pathlib.Path(__file__).resolve().parents[1] / "__main__.py"
    tree = ast.parse(src.read_text())
    start = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "start")
    text = ast.unparse(start)
    assert "await self._session_manager.shutdown_from_loop()" in text
    assert "self._session_manager.shutdown()" not in text


async def test_a_save_after_the_runner_was_released_keeps_the_record(
        harness) -> None:
    """An async save racing an unload finds the runner gone.  It must not
    write the empty history it would fetch over the good record."""
    await harness.start()
    try:
        assert await asyncio.to_thread(
            harness.manager._save_session, harness.session)
        await asyncio.to_thread(harness.server.shutdown)
        again = await asyncio.to_thread(
            harness.manager._save_session, harness.session)
        assert again is False
        assert len(harness.saved_history()) == 1
    finally:
        await harness.stop()


async def test_a_server_that_never_had_a_runner_still_saves(
        harness) -> None:
    """The control: ``_runner_rpc is None`` without a release is a server
    with no runner, and it still saves (with an empty history)."""
    await harness.start()
    try:
        harness.server._runner_rpc = None
        ok = await asyncio.to_thread(
            harness.manager._save_session, harness.session)
        assert ok is True
        assert harness.saved_history() == []
    finally:
        harness.server._runner_rpc = harness.client
        await harness.stop()


# ----------------------------------------------------------------------
# Backstops: a caller that still reaches these on the loop
# ----------------------------------------------------------------------


async def test_a_close_on_the_loop_thread_does_not_stall_it(harness) -> None:
    """``JaatoServer.shutdown`` called on the loop used to block 10 s on
    ``rpc.close()``.  Now the close is scheduled and it returns."""
    await harness.start()
    try:
        started = time.monotonic()
        harness.server.shutdown()
        assert time.monotonic() - started < 3.0
    finally:
        await harness.stop()


async def test_a_refused_call_leaks_no_coroutine(harness) -> None:
    """The guard still refuses, and closes the coroutine it was handed."""
    await harness.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError, match="event-loop thread"):
                harness.client.session_get_history_threadsafe()
            gc.collect()
        leaked = [w for w in caught if "never awaited" in str(w.message)]
        assert leaked == []
    finally:
        await harness.stop()


# ----------------------------------------------------------------------
# AST guard: no async def calls a runner-blocking sync entry point
# ----------------------------------------------------------------------

#: Sync methods that reach a ``*_threadsafe`` runner RPC and were called
#: from an ``async def`` before #1355.  Each is checked below to really
#: reach one, so the list cannot go stale silently.
_BLOCKING_ENTRY_POINTS = {
    "get_command_list", "list_sessions", "_sessions_loaded_in",
    "_session_rows", "execute_command", "history_page", "get_history",
    "get_available_commands", "_on_command_list_request", "save_session",
    "save_all", "stop_session",
}
#: Names too common to judge alone: flagged only on these receivers.
_RECEIVER_SCOPED = {
    "shutdown": ("_session_manager", "_jaato_server"),
    "stop": ("_jaato_server",),
}

_SERVER_DIR = pathlib.Path(__file__).resolve().parents[1]


def _blocking_calls_in_async_defs() -> List[str]:
    found = []
    for path in sorted(_SERVER_DIR.rglob("*.py")):
        if "tests" in path.relative_to(_SERVER_DIR).parts:
            continue
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            awaited = {
                id(n.value) for n in ast.walk(fn)
                if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
            }
            stack = list(fn.body)
            while stack:
                node = stack.pop()
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.Lambda, ast.ClassDef)):
                    continue       # runs elsewhere, not in this coroutine
                stack.extend(ast.iter_child_nodes(node))
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and id(node) not in awaited):
                    continue
                attr = node.func.attr
                receiver = ast.unparse(node.func.value)
                scoped = _RECEIVER_SCOPED.get(attr)
                if (attr.endswith("_threadsafe")
                        and attr != "run_coroutine_threadsafe"
                        or attr in _BLOCKING_ENTRY_POINTS
                        or scoped and receiver.endswith(scoped)):
                    found.append(
                        f"{path.name}:{node.lineno} {fn.name}: "
                        f"{receiver}.{attr}()")
    return found


def test_no_async_def_calls_a_runner_blocking_entry_point() -> None:
    """Such a call runs on the loop thread; hand it to
    ``asyncio.to_thread`` and await it instead."""
    assert _blocking_calls_in_async_defs() == []


def test_every_listed_entry_point_really_reaches_the_runner() -> None:
    """The list above is a claim about the call graph; check it, so a name
    that stops reaching a wrapper is removed rather than trusted."""
    files = ["core.py", "session_manager.py", "command_router.py",
             "websocket.py", "ipc.py"]
    bodies = {}
    for name in files:
        tree = ast.parse((_SERVER_DIR / name).read_text())
        for cls in ast.walk(tree):
            if isinstance(cls, ast.ClassDef):
                for fn in cls.body:
                    if isinstance(fn, ast.FunctionDef):
                        bodies.setdefault(fn.name, []).append(fn)

    def called(fn):
        return {n.func.attr for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)}

    def wrappers(fn):
        # ``rpc.x_threadsafe(...)`` and ``getattr(rpc, "x_threadsafe")``.
        names = called(fn) | {
            n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        return {a for a in names if a.endswith("_threadsafe")
                and a != "run_coroutine_threadsafe"}

    reaches = {name for name, fns in bodies.items()
               if any(wrappers(f) for f in fns)}
    grew = True
    while grew:
        grew = False
        for name, fns in bodies.items():
            if name not in reaches and any(called(f) & reaches for f in fns):
                reaches.add(name)
                grew = True
    listed = (_BLOCKING_ENTRY_POINTS | set(_RECEIVER_SCOPED)) - {
        "_on_command_list_request",   # a callback slot, bound to get_command_list
    }
    assert listed - reaches == set()
