"""Tests for ``server.runner_spawn.spawn_session_runner`` (Phase 3
§3.12).

The helper extracts the runner-spawn primitive that both the IPC
apparmor pre-init hook and the WS-server apparmor pre-init hook
share.  These tests pin its contract using stub spawner +
RunnerRPCClient + JaatoServer fixtures so the test exercises the
helper's wiring (lifecycle ordering, kwargs, log path computation)
rather than fork/exec'ing a real subprocess.

Coverage:

- Successful spawn: ``RunnerSpawner.spawn`` is called with the
  envelope's profile/session/workspace; ``RunnerRPCClient`` is
  constructed against the parent socket and started; the rpc
  handle is attached onto the JaatoServer via
  ``set_runner_rpc(rpc, spawned)``.
- ``daemon_loop=None`` raises ``RuntimeError`` immediately (no
  spawn attempt).
- log_path is derived from ``<workspace>/.jaato/logs/runner-<sid>.log``.
- log_path is None for empty workspace_path (defensive).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


class _FakeSpawned:
    """Stand-in for ``server.runner_spawner.SpawnedRunner``."""

    def __init__(self, pid: int = 12345) -> None:
        self.pid = pid
        self.parent_socket = object()  # opaque handle


class _FakeSpawner:
    """Stand-in for ``server.runner_spawner.RunnerSpawner``."""

    instances: List["_FakeSpawner"] = []

    def __init__(self) -> None:
        self.spawn_calls: List[dict] = []
        _FakeSpawner.instances.append(self)

    def spawn(self, **kwargs: Any) -> _FakeSpawned:
        self.spawn_calls.append(dict(kwargs))
        return _FakeSpawned()


class _FakeRunnerRPCClient:
    """Stand-in for ``server.runner_rpc_client.RunnerRPCClient``."""

    instances: List["_FakeRunnerRPCClient"] = []

    def __init__(
        self,
        parent_socket: Any,
        runner_pid: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.parent_socket = parent_socket
        self.runner_pid = runner_pid
        self.loop = loop
        self.started = False
        _FakeRunnerRPCClient.instances.append(self)

    async def start(self) -> None:
        self.started = True


class _FakeJaatoServer:
    """Stand-in for ``server.core.JaatoServer``."""

    def __init__(self) -> None:
        self.runner_rpc: Optional[_FakeRunnerRPCClient] = None
        self.spawned: Optional[_FakeSpawned] = None

    def set_runner_rpc(self, rpc: Any, spawned: Any) -> None:
        self.runner_rpc = rpc
        self.spawned = spawned


@pytest.fixture
def daemon_loop():
    """An asyncio loop running on a worker thread so
    run_coroutine_threadsafe has somewhere to dispatch."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    ready.wait()
    yield loop

    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)
    loop.close()


@pytest.fixture(autouse=True)
def _patch_spawner_and_client():
    """Replace RunnerSpawner + RunnerRPCClient with stubs."""
    _FakeSpawner.instances.clear()
    _FakeRunnerRPCClient.instances.clear()
    with patch(
        "server.runner_spawner.RunnerSpawner", _FakeSpawner,
    ), patch(
        "server.runner_rpc_client.RunnerRPCClient", _FakeRunnerRPCClient,
    ):
        yield


# ----------------------------------------------------------------------
# Successful spawn
# ----------------------------------------------------------------------


def test_spawn_calls_spawner_with_session_fields(
    daemon_loop, tmp_path,
) -> None:
    from server.runner_spawn import spawn_session_runner

    server = _FakeJaatoServer()
    workspace = str(tmp_path)

    spawn_session_runner(
        server=server,
        session_id="sess-1",
        workspace_path=workspace,
        profile_name="jaato-ws-sess-1",
        daemon_loop=daemon_loop,
    )

    assert len(_FakeSpawner.instances) == 1
    spawn_call = _FakeSpawner.instances[0].spawn_calls[0]
    assert spawn_call["session_id"] == "sess-1"
    assert spawn_call["workspace_path"] == workspace
    assert spawn_call["profile_name"] == "jaato-ws-sess-1"


def test_spawn_attaches_rpc_to_server(daemon_loop, tmp_path) -> None:
    from server.runner_spawn import spawn_session_runner

    server = _FakeJaatoServer()
    spawn_session_runner(
        server=server,
        session_id="sess-2",
        workspace_path=str(tmp_path),
        profile_name="jaato-ws-sess-2",
        daemon_loop=daemon_loop,
    )

    assert server.runner_rpc is not None
    assert isinstance(server.runner_rpc, _FakeRunnerRPCClient)
    assert server.spawned is not None
    # The RPC client was started (via run_coroutine_threadsafe).
    assert server.runner_rpc.started is True


def test_spawn_log_path_uses_workspace_dot_jaato_logs(
    daemon_loop, tmp_path,
) -> None:
    from server.runner_spawn import spawn_session_runner

    server = _FakeJaatoServer()
    workspace = str(tmp_path)
    spawn_session_runner(
        server=server,
        session_id="sess-log",
        workspace_path=workspace,
        profile_name="p",
        daemon_loop=daemon_loop,
    )

    spawn_call = _FakeSpawner.instances[0].spawn_calls[0]
    assert spawn_call["log_path"] == f"{workspace}/.jaato/logs/runner-sess-log.log"


def test_spawn_log_path_none_for_empty_workspace(
    daemon_loop,
) -> None:
    from server.runner_spawn import spawn_session_runner

    server = _FakeJaatoServer()
    spawn_session_runner(
        server=server,
        session_id="sess-no-ws",
        workspace_path="",
        profile_name="p",
        daemon_loop=daemon_loop,
    )

    spawn_call = _FakeSpawner.instances[0].spawn_calls[0]
    assert spawn_call["log_path"] is None


# ----------------------------------------------------------------------
# daemon_loop=None → fail fast
# ----------------------------------------------------------------------


def test_spawn_without_daemon_loop_raises_runtime_error() -> None:
    from server.runner_spawn import spawn_session_runner

    server = _FakeJaatoServer()
    with pytest.raises(RuntimeError, match="daemon loop unavailable"):
        spawn_session_runner(
            server=server,
            session_id="sess-no-loop",
            workspace_path="/tmp/ws",
            profile_name="p",
            daemon_loop=None,  # type: ignore[arg-type]
        )

    # No spawn attempt was made — failed before reaching the spawner.
    assert _FakeSpawner.instances == []
    assert server.runner_rpc is None


# ----------------------------------------------------------------------
# WS hook signature: 4-arg form
# ----------------------------------------------------------------------


def test_ws_apparmor_pre_init_hook_accepts_client_id_kwarg() -> None:
    """Phase 3 §3.12: the WS apparmor pre-init hook signature is the
    4-arg form ``(server, session_id, workspace_path, client_id)``.
    Inspect the function via the SessionManager pre-initialize-hook
    dispatcher to confirm modern-path engagement.

    The introspection logic in
    ``SessionManager._run_pre_initialize_hooks`` counts positional
    parameters (POSITIONAL_ONLY +
    POSITIONAL_OR_KEYWORD); 4 → new-style, 3 → legacy.  After §3.12
    the WS hook should report >= 4.
    """
    import inspect

    # The hook is a closure inside JaatoWSServer.set_command_router.
    # Build a minimal WS server + router stub to register the hook,
    # then introspect the registered callable.
    from server.websocket import JaatoWSServer

    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._apparmor = None
    ws._cgroups = None
    ws._workspace_root = "/tmp/ws"
    ws._event_loop = None  # captured later in real start()

    class _FakeSM:
        def __init__(self):
            self.pre_init_hooks: List[Any] = []
            self.session_hooks: List[Any] = []

        def add_pre_initialize_hook(self, hook):
            self.pre_init_hooks.append(hook)

        def add_session_hook(self, hook):
            self.session_hooks.append(hook)

    class _FakeRouter:
        def __init__(self):
            self._session_manager = _FakeSM()

    router = _FakeRouter()
    JaatoWSServer.set_command_router(ws, router)

    assert router._session_manager.pre_init_hooks, (
        "WS server didn't register a pre-init hook"
    )
    hook = router._session_manager.pre_init_hooks[0]
    sig = inspect.signature(hook)
    n_positional = sum(
        1 for p in sig.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    assert n_positional >= 4, (
        f"WS apparmor pre-init hook is still 3-arg (n={n_positional}); "
        f"§3.12 migration to 4-arg form did not land"
    )
    # The 4th param must be named client_id for self-documentation.
    param_names = list(sig.parameters)
    assert "client_id" in param_names, (
        f"WS hook signature missing client_id param; got {param_names!r}"
    )
