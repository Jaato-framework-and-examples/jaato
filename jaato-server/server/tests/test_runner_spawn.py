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


# ----------------------------------------------------------------------
# Phase 5 §5.1b — mainline RuntimeLimits app-layer passthrough
# ----------------------------------------------------------------------


class _ProfileWithLimits:
    """Tiny stand-in for ``SubagentProfile`` exposing the field
    ``spawn_session_runner`` reads.  Avoids importing the full
    profile machinery into this lightweight test module."""

    def __init__(self, runtime_limits: Any) -> None:
        self.runtime_limits = runtime_limits


def test_app_layer_fields_forwarded_when_profile_sets_them(
    daemon_loop, tmp_path,
) -> None:
    """Phase 5 §5.1b: profile carrying ``runtime_limits`` with
    ``tool_timeout_seconds`` + ``max_output_bytes`` set must
    propagate those values into ``RunnerSpawner.spawn``'s
    matching kwargs, which the spawner forwards as
    ``JAATO_RUNNER_*`` env vars."""
    from server.runner_spawn import spawn_session_runner
    from shared.runtime_limits import RuntimeLimits

    server = _FakeJaatoServer()
    server._profile = _ProfileWithLimits(
        RuntimeLimits(
            tool_timeout_seconds=30.0,
            max_output_bytes=8192,
        ),
    )

    spawn_session_runner(
        server=server,
        session_id="sess-1",
        workspace_path=str(tmp_path),
        profile_name="jaato-ws-sess-1",
        daemon_loop=daemon_loop,
    )

    spawn_call = _FakeSpawner.instances[0].spawn_calls[0]
    assert spawn_call["max_output_chars"] == 8192
    assert spawn_call["tool_timeout_seconds"] == 30.0


def test_no_profile_passes_none_for_both_kwargs(
    daemon_loop, tmp_path,
) -> None:
    """Inline-spec / no-profile sessions have ``server._profile is
    None``.  Both app-layer kwargs forward as ``None``;
    ``RunnerSpawner.spawn`` then skips the env-var write."""
    from server.runner_spawn import spawn_session_runner

    server = _FakeJaatoServer()
    # _FakeJaatoServer doesn't set _profile in __init__; the
    # getattr-with-default path inside spawn_session_runner must
    # tolerate the missing attr.
    spawn_session_runner(
        server=server,
        session_id="sess-1",
        workspace_path=str(tmp_path),
        profile_name="jaato-ws-sess-1",
        daemon_loop=daemon_loop,
    )

    spawn_call = _FakeSpawner.instances[0].spawn_calls[0]
    assert spawn_call["max_output_chars"] is None
    assert spawn_call["tool_timeout_seconds"] is None


def test_profile_without_runtime_limits_passes_none(
    daemon_loop, tmp_path,
) -> None:
    """Profile with ``runtime_limits=None`` (the SubagentProfile
    default) → both kwargs forward as ``None``.  No mainline
    defaulting: §5.1b is wiring-only."""
    from server.runner_spawn import spawn_session_runner

    server = _FakeJaatoServer()
    server._profile = _ProfileWithLimits(runtime_limits=None)

    spawn_session_runner(
        server=server,
        session_id="sess-1",
        workspace_path=str(tmp_path),
        profile_name="jaato-ws-sess-1",
        daemon_loop=daemon_loop,
    )

    spawn_call = _FakeSpawner.instances[0].spawn_calls[0]
    assert spawn_call["max_output_chars"] is None
    assert spawn_call["tool_timeout_seconds"] is None


def test_kernel_fields_do_not_leak_into_app_layer_kwargs(
    daemon_loop, tmp_path,
) -> None:
    """Profile that sets only the kernel-enforced fields
    (memory/pids/cpu_weight) → both app-layer kwargs stay ``None``.
    Kernel fields wire through ``CgroupsManager.provision_cgroup``
    elsewhere; the spawn kwargs only carry app-layer values."""
    from server.runner_spawn import spawn_session_runner
    from shared.runtime_limits import RuntimeLimits

    server = _FakeJaatoServer()
    server._profile = _ProfileWithLimits(
        RuntimeLimits(memory_max_mb=4096, pids_max=64, cpu_weight=100),
    )

    spawn_session_runner(
        server=server,
        session_id="sess-1",
        workspace_path=str(tmp_path),
        profile_name="jaato-ws-sess-1",
        daemon_loop=daemon_loop,
    )

    spawn_call = _FakeSpawner.instances[0].spawn_calls[0]
    assert spawn_call["max_output_chars"] is None
    assert spawn_call["tool_timeout_seconds"] is None


def test_partial_app_layer_supplied_forwards_only_set_fields(
    daemon_loop, tmp_path,
) -> None:
    """Per-field independence: profile sets only
    ``tool_timeout_seconds`` → timeout kwarg is 30.0, output kwarg
    is ``None``.  Mirrors the no-defaulting decision (§3.2 of the
    §5.1b audit)."""
    from server.runner_spawn import spawn_session_runner
    from shared.runtime_limits import RuntimeLimits

    server = _FakeJaatoServer()
    server._profile = _ProfileWithLimits(
        RuntimeLimits(tool_timeout_seconds=30.0),
    )

    spawn_session_runner(
        server=server,
        session_id="sess-1",
        workspace_path=str(tmp_path),
        profile_name="jaato-ws-sess-1",
        daemon_loop=daemon_loop,
    )

    spawn_call = _FakeSpawner.instances[0].spawn_calls[0]
    assert spawn_call["tool_timeout_seconds"] == 30.0
    assert spawn_call["max_output_chars"] is None


# ----------------------------------------------------------------------
# Pool routing (pool PR 4)
# ----------------------------------------------------------------------


class _FakePoolManager:
    """Stand-in for :class:`server.runner_pool.PoolManager`.

    Controls the acquire_slot return value per-test so the routing
    branches are pinned without spawning real subprocesses.

    Phase 2 (server 0.6.144+): ``acquire_slot`` now accepts
    ``cascade_driver_id`` (recorded for assertion) and returns a
    :class:`PoolSlot` instead of a raw tuple.
    """

    def __init__(self, slot=None) -> None:
        self.slot = slot
        self.acquire_calls = 0
        self.last_cascade_id = None

    def acquire_slot(self, cascade_driver_id=None):
        self.acquire_calls += 1
        self.last_cascade_id = cascade_driver_id
        return self.slot


def _stub_socket() -> Any:
    """Lightweight stub for the slot's daemon-side socket."""
    return MagicMock(name="slot_daemon_socket")


def test_pool_routing_uses_slot_when_flag_enabled(
    daemon_loop, tmp_path, monkeypatch,
) -> None:
    """Pool slot is consumed when env flag is on + pool has a slot +
    session opted out of AppArmor."""
    from server.runner_spawn import spawn_session_runner

    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "true")
    server = _FakeJaatoServer()
    from server.runner_pool import PoolSlot
    pool = _FakePoolManager(slot=PoolSlot(pid=99999, sock=_stub_socket()))

    spawn_session_runner(
        server=server,
        session_id="sess-pool",
        workspace_path=str(tmp_path),
        profile_name="",
        daemon_loop=daemon_loop,
        disable_confine=True,
        pool_manager=pool,
    )

    # Pool was consulted.
    assert pool.acquire_calls == 1
    # Cold spawn was NOT invoked (slot served the session).
    assert _FakeSpawner.instances == []
    # The slot pid + socket are wired through the SpawnedRunner.
    assert server.spawned is not None
    assert server.spawned.pid == 99999


def test_pool_routing_falls_back_when_flag_explicitly_disabled(
    daemon_loop, tmp_path, monkeypatch,
) -> None:
    """Pool PR 5e: default is enabled.  Explicit ``false`` disables.

    Operators who set ``JAATO_RUNNER_POOL_ENABLED=false`` opt OUT of
    the pool — sessions cold-spawn.  Recognised falsy values:
    ``0`` / ``false`` / ``no`` / ``off`` (case-insensitive)."""
    from server.runner_spawn import spawn_session_runner

    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "false")
    server = _FakeJaatoServer()
    from server.runner_pool import PoolSlot
    pool = _FakePoolManager(slot=PoolSlot(pid=99999, sock=_stub_socket()))

    spawn_session_runner(
        server=server,
        session_id="sess-disabled",
        workspace_path=str(tmp_path),
        profile_name="",
        daemon_loop=daemon_loop,
        disable_confine=True,
        pool_manager=pool,
    )

    assert pool.acquire_calls == 0
    assert len(_FakeSpawner.instances) == 1  # cold-spawned


def test_pool_routing_enabled_by_default(
    daemon_loop, tmp_path, monkeypatch,
) -> None:
    """Pool PR 5e: when the env var is unset (deleted), the pool is
    enabled by default.  Pre-PR-5e this test verified the opposite —
    that empty meant disabled."""
    from server.runner_spawn import spawn_session_runner

    monkeypatch.delenv("JAATO_RUNNER_POOL_ENABLED", raising=False)
    server = _FakeJaatoServer()
    from server.runner_pool import PoolSlot
    pool = _FakePoolManager(slot=PoolSlot(pid=99999, sock=_stub_socket()))

    spawn_session_runner(
        server=server,
        session_id="sess-default",
        workspace_path=str(tmp_path),
        profile_name="",
        daemon_loop=daemon_loop,
        disable_confine=True,
        pool_manager=pool,
    )

    assert pool.acquire_calls == 1
    assert _FakeSpawner.instances == []  # pool-served, no cold-spawn


def test_pool_routing_falls_back_when_pool_empty(
    daemon_loop, tmp_path, monkeypatch,
) -> None:
    """Empty pool → cold-spawn."""
    from server.runner_spawn import spawn_session_runner

    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "1")
    server = _FakeJaatoServer()
    pool = _FakePoolManager(slot=None)  # empty

    spawn_session_runner(
        server=server,
        session_id="sess-empty",
        workspace_path=str(tmp_path),
        profile_name="",
        daemon_loop=daemon_loop,
        disable_confine=True,
        pool_manager=pool,
    )

    assert pool.acquire_calls == 1
    assert len(_FakeSpawner.instances) == 1  # cold-spawned


def test_pool_routing_consumes_slot_when_apparmor_opted_in(
    daemon_loop, tmp_path, monkeypatch,
) -> None:
    """PR 5a: AppArmor opt-in sessions are NOW pool-eligible.  The
    slot self-confines via envelope.profile_name in step 1c of
    bootstrap_session.  Pre-PR-5a these sessions fell through to
    cold-spawn (disable_confine gate).  Post-PR-5a they consume a
    pool slot and the slot carries the profile_name on its
    SpawnedRunner handle."""
    from server.runner_spawn import spawn_session_runner

    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "true")
    server = _FakeJaatoServer()
    from server.runner_pool import PoolSlot
    pool = _FakePoolManager(slot=PoolSlot(pid=99999, sock=_stub_socket()))

    spawn_session_runner(
        server=server,
        session_id="sess-apparmor",
        workspace_path=str(tmp_path),
        profile_name="jaato-ws-sess-apparmor",
        daemon_loop=daemon_loop,
        disable_confine=False,  # apparmor opted-in
        pool_manager=pool,
    )

    assert pool.acquire_calls == 1
    assert _FakeSpawner.instances == []  # NOT cold-spawned
    assert server.spawned is not None
    assert server.spawned.profile_name == "jaato-ws-sess-apparmor"


def test_pool_routing_skipped_when_cgroup_attach_set(
    daemon_loop, tmp_path, monkeypatch,
) -> None:
    """Session needs cgroup_attach → never consult the pool (slot
    is already in daemon cgroup; migration is PR 5)."""
    from server.runner_spawn import spawn_session_runner

    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "true")
    server = _FakeJaatoServer()
    from server.runner_pool import PoolSlot
    pool = _FakePoolManager(slot=PoolSlot(pid=99999, sock=_stub_socket()))

    spawn_session_runner(
        server=server,
        session_id="sess-cgroup",
        workspace_path=str(tmp_path),
        profile_name="",
        daemon_loop=daemon_loop,
        disable_confine=True,
        cgroup_attach=lambda: None,  # needs cgroup migration
        pool_manager=pool,
    )

    assert pool.acquire_calls == 0
    assert len(_FakeSpawner.instances) == 1


def test_pool_routing_no_pool_manager_falls_back(
    daemon_loop, tmp_path, monkeypatch,
) -> None:
    """``pool_manager=None`` is the default; routing is skipped."""
    from server.runner_spawn import spawn_session_runner

    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "true")
    server = _FakeJaatoServer()

    spawn_session_runner(
        server=server,
        session_id="sess-none",
        workspace_path=str(tmp_path),
        profile_name="",
        daemon_loop=daemon_loop,
        disable_confine=True,
        pool_manager=None,
    )

    assert len(_FakeSpawner.instances) == 1
