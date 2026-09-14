"""The envelope PRODUCER must populate ``runtime_limits`` — and the
CONSUMER must arm the plugin that enforces it (#735).

Sibling of ``test_envelope_carries_budget_control.py``, and the same
"field exists, nothing sets it" class — one rung worse.  For
``budget_control`` the field existed and nobody populated it.  For
``tool_timeout_seconds`` / ``max_output_bytes`` there was no field at
all: the caps travelled as process-startup env
(``JAATO_RUNNER_TOOL_TIMEOUT_SECONDS`` / ``JAATO_RUNNER_MAX_OUTPUT_CHARS``)
and landed on ``server/runner/tool_executor.ToolExecutor`` — the Phase-2
cli-only ``execute_fn`` that ``RunnerRPC`` bypasses whenever a session
host exists, i.e. on every path that dispatches ``session.bootstrap``,
i.e. all of them.

Measured on a real daemon at ``main`` @ ``2bd2456``: a profile declaring
``tool_timeout_seconds: 2`` ran a ``sleep 60`` for **60.02 s** on the
pool path AND on cold-spawn, and ``max_output_bytes: 1000`` truncated at
the compile-time 50 000.  The pre-existing guard
(``test_app_layer_fields_forwarded_when_profile_sets_them``) asserted the
env write on cold-spawn and was green throughout — which is why this
file asserts three DIFFERENT things:

1. the producer populates the field, on the path every session takes;
2. a POOL-SERVED session's bootstrap envelope carries it (the default
   path, and the one the issue was filed about);
3. ``configure()`` ARMS the cli plugin with it
   (``test_configure_arms_the_subprocess_plugins.py``).

(3) is the load-bearing one.  (1) and (2) would both pass on a tree
where the value rides the wire and the runner drops it on the floor,
which is exactly the shape the env mechanism had.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import patch

import pytest

from shared.runtime_limits import RuntimeLimits


# ----------------------------------------------------------------------
# Profile / server stand-ins (same shape as the budget_control sibling)
# ----------------------------------------------------------------------

def _profile(limits=None, **kw):
    base = dict(
        name="p", description="d", provider="openrouter", model="m",
        plugins=[], preloaded_plugins=set(), plugin_configs={},
        tool_scopes={}, model_tiers={}, gc=None, runtime_limits=limits,
        env={}, completion_payload_schema=None, spawn_payload_schema=None,
        completion_processors=[], budget_control=None, quirks={},
        apparmor=False, apparmor_fragments=None, max_turns=10,
        system_instructions=None, agent_params={},
        suppress_base_instructions=False, config_root=None, inherits=None,
        icon=None, description_for_model=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _server(profile):
    return SimpleNamespace(
        _profile=profile, config_root=None, _main_agent_id="main",
        _cascade_driver_id=None,
    )


def _caps():
    """Both subprocess caps, at values nothing else in the tree uses."""
    return RuntimeLimits(tool_timeout_seconds=120.0, max_output_bytes=8192)


# ----------------------------------------------------------------------
# 1. The producer
# ----------------------------------------------------------------------

def test_build_session_envelope_populates_runtime_limits():
    """THE regression: the producer, on the path every session takes.

    ``build_session_envelope`` runs for pool-served and cold-spawned
    sessions alike.  The pre-#735 lookup lived inside
    ``spawn_session_runner``'s ``if spawned is None:`` cold-spawn
    branch, which a pool slot never reaches.
    """
    from server.runner_spawn import build_session_envelope
    env = build_session_envelope(
        server=_server(_profile(_caps())), session_id="s1",
        workspace_path="/tmp/ws", profile_name="p",
    )
    assert env.runtime_limits is not None, (
        "envelope.runtime_limits is None — the profile declared caps but "
        "the producer dropped them; the session's tools run UNBOUNDED"
    )
    assert env.runtime_limits["tool_timeout_seconds"] == 120.0
    assert env.runtime_limits["max_output_bytes"] == 8192


def test_limitless_profile_leaves_runtime_limits_none():
    """Absence must stay absence.

    ``None`` is what lets ``_apply_runtime_limits`` tell "nobody
    declared limits" from "somebody declared these" — which matters
    because sessions on one runtime SHARE the plugin registry, so a
    limitless session must not be able to disarm a sibling's cap.
    """
    from server.runner_spawn import build_session_envelope
    env = build_session_envelope(
        server=_server(_profile(None)), session_id="s2",
        workspace_path="/tmp/ws", profile_name="p",
    )
    assert env.runtime_limits is None


def test_runtime_limits_survive_the_wire_from_the_producer():
    """Producer -> to_dict -> from_dict -> the runner's re-parse.

    The end-to-end wire claim: what the producer sets is what the
    runner can actually build a ``RuntimeLimits`` from.
    """
    from server.runner.session import _runtime_limits_from_envelope
    from server.runner_spawn import build_session_envelope
    from shared.session_envelope import SessionInitEnvelope
    env = build_session_envelope(
        server=_server(_profile(_caps())), session_id="s3",
        workspace_path="/tmp/ws", profile_name="p",
    )
    revived = SessionInitEnvelope.from_dict(env.to_dict())
    assert _runtime_limits_from_envelope(revived) == _caps()


def test_the_whole_block_rides_not_just_the_two_subprocess_caps():
    """The kernel trio is carried too.

    Not because the runner enforces it — the cgroup is written
    daemon-side before the runner is forked — but because
    ``_runtime_limits_to_dict`` is the profile's own serialiser and a
    partial copy would be a second, divergent definition of what the
    block is.
    """
    from server.runner_spawn import build_session_envelope
    env = build_session_envelope(
        server=_server(_profile(RuntimeLimits(
            memory_max_mb=512, pids_max=64, cpu_weight=200,
            tool_timeout_seconds=5.0, max_output_bytes=100,
            max_parallel_tools=2,
        ))),
        session_id="s4", workspace_path="/tmp/ws", profile_name="p",
    )
    assert env.runtime_limits["memory_max_mb"] == 512
    assert env.runtime_limits["pids_max"] == 64
    assert env.runtime_limits["max_parallel_tools"] == 2
    # v6's standalone field keeps its own value, for daemon/runner skew.
    assert env.max_parallel_tools == 2


def test_isolated_subrunner_carries_the_EFFECTIVE_block():
    """The third spawn path, and the one where the two blocks differ.

    An isolated subagent's limits go through ``apply_isolated_defaults``
    first, so ``profile.runtime_limits`` is not what the session runs
    under.  Arming it with the profile's own block would install a
    weaker cap than the caller provisioned.
    """
    from server.session_manager import _isolated_limits
    effective = RuntimeLimits(tool_timeout_seconds=30.0, pids_max=64)
    assert _isolated_limits(effective, _profile(_caps())) is effective


def test_isolated_subrunner_falls_back_to_the_profile_block():
    """A caller that resolved nothing still behaves as it did pre-#735."""
    from server.session_manager import _isolated_limits
    assert _isolated_limits(None, _profile(_caps())) == _caps()


def test_a_stand_in_profile_does_not_take_the_envelope_down():
    """``_runtime_limits_to_dict`` calls ``dataclasses.asdict``.

    Both producers therefore type-CHECK the declared attribute rather
    than duck-typing it: an object that is not the dataclass means
    "nobody declared limits", not an exception that aborts the whole
    envelope build over one field.
    """
    from unittest.mock import MagicMock

    from server.runner_spawn import _profile_runtime_limits
    from server.session_manager import _isolated_limits
    assert _profile_runtime_limits(MagicMock()) is None
    assert _isolated_limits(None, MagicMock()) is None
    assert _profile_runtime_limits(None) is None


def test_malformed_wire_block_degrades_instead_of_refusing_bootstrap():
    """A block this runner cannot parse means "nobody declared limits".

    The value was validated at profile-load time daemon-side, so
    anything unparseable here came from a NEWER daemon.  Refusing the
    bootstrap would turn forward-compat skew into a session that cannot
    start at all.
    """
    from server.runner.session import _runtime_limits_from_envelope
    env = SimpleNamespace(runtime_limits={"tool_timeout_seconds": -1})
    assert _runtime_limits_from_envelope(env) is None


# ----------------------------------------------------------------------
# 2. The pool-served guard — the one the issue asks for by name
# ----------------------------------------------------------------------

class _FakeSpawned:
    """Stand-in for ``server.runner_spawner.SpawnedRunner``."""

    def __init__(self, pid: int = 12345) -> None:
        self.pid = pid
        self.parent_socket = object()


class _FakeSpawner:
    """Records cold-spawns.

    Staying EMPTY is how a test proves the pool served the session,
    rather than merely asserting that some env var was written.
    """

    instances: List["_FakeSpawner"] = []

    def __init__(self) -> None:
        self.spawn_calls: List[dict] = []
        _FakeSpawner.instances.append(self)

    def spawn(self, **kwargs: Any) -> _FakeSpawned:
        self.spawn_calls.append(dict(kwargs))
        return _FakeSpawned()


class _RecordingRPCClient:
    """Stand-in for ``RunnerRPCClient`` that RECORDS the bootstrap envelope.

    ``test_runner_spawn._FakeRunnerRPCClient`` has no
    ``bootstrap_session_threadsafe`` at all, so the dispatch raises
    ``AttributeError`` into ``dispatch_bootstrap_envelope``'s
    ``except Exception`` and any envelope assertion built on it is
    unreachable.  Recording it is what makes the pool-served claim
    testable at all.
    """

    instances: List["_RecordingRPCClient"] = []

    def __init__(self, parent_socket, runner_pid, loop) -> None:
        self.parent_socket = parent_socket
        self.runner_pid = runner_pid
        self.loop = loop
        self.started = False
        self.bootstrapped: List[Any] = []
        _RecordingRPCClient.instances.append(self)

    async def start(self) -> None:
        self.started = True

    def bootstrap_session_threadsafe(self, envelope, timeout=30.0):
        self.bootstrapped.append(envelope)
        return {"ok": True}


class _FakeSlot:
    """Stand-in for a pre-warm pool slot (``PoolSlot``)."""

    def __init__(self, pid: int = 99999) -> None:
        self.pid = pid
        self.sock = object()
        self.cascade_id = None
        self.rpc = None


class _FakePoolManager:
    """Always hands out a warm slot, so the session is pool-served."""

    def __init__(self) -> None:
        self.slot = _FakeSlot()
        self.acquire_calls = 0

    def acquire_slot(self, **_kw) -> _FakeSlot:
        self.acquire_calls += 1
        return self.slot


class _CapturingServer:
    """Minimal ``JaatoServer`` stand-in for the two spawn helpers."""

    def __init__(self, profile) -> None:
        self._profile = profile
        self.config_root = None
        self._main_agent_id = "main"
        self._cascade_driver_id = None
        self.runner_rpc = None
        self.spawned = None
        self.runner_ready = False

    def set_runner_rpc(self, rpc, spawned) -> None:
        self.runner_rpc = rpc
        self.spawned = spawned

    def mark_runner_ready(self) -> None:
        """``dispatch_bootstrap_envelope`` calls this in its ``finally``."""
        self.runner_ready = True


@pytest.fixture
def daemon_loop():
    """An asyncio loop on a worker thread, for run_coroutine_threadsafe."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)
    loop.close()


@pytest.fixture(autouse=True)
def _stub_spawn_machinery():
    """Replace RunnerSpawner + RunnerRPCClient with the recorders above."""
    _FakeSpawner.instances.clear()
    _RecordingRPCClient.instances.clear()
    with patch("server.runner_spawner.RunnerSpawner", _FakeSpawner), \
            patch("server.runner_rpc_client.RunnerRPCClient",
                  _RecordingRPCClient):
        yield


def _spawn_and_bootstrap(server, daemon_loop, tmp_path, *, pool):
    """Run both halves of a session start and return the envelope sent.

    ``spawn_session_runner`` chooses the path; ``dispatch_bootstrap_envelope``
    is what actually hands the runner its ``SessionInitEnvelope``.  Asserting
    on the latter is what makes these claims about the SESSION rather than
    about spawn kwargs.
    """
    from server.runner_spawn import (dispatch_bootstrap_envelope,
                                     spawn_session_runner)
    spawn_session_runner(
        server=server, session_id="sess-pool",
        workspace_path=str(tmp_path), profile_name="",
        daemon_loop=daemon_loop, disable_confine=True,
        pool_manager=pool,
    )
    dispatch_bootstrap_envelope(
        server=server, session_id="sess-pool",
        workspace_path=str(tmp_path), profile_name="",
    )
    assert server.runner_rpc.bootstrapped, (
        "no session.bootstrap envelope was recorded — the dispatch was "
        "swallowed, so nothing below asserts anything"
    )
    return server.runner_rpc.bootstrapped[-1]


def test_pool_served_session_gets_the_cap(daemon_loop, tmp_path,
                                          monkeypatch):
    """The guard #735 asks for by name: a POOL-SERVED session.

    ``_FakeSpawner.instances`` staying empty is the proof that a slot
    served it.  A pool slot is forked from the daemon's template before
    this session existed and can never receive a per-session env var, so
    the envelope is the only vehicle that can reach it.
    """
    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "true")
    server = _CapturingServer(_profile(_caps()))
    pool = _FakePoolManager()

    env = _spawn_and_bootstrap(server, daemon_loop, tmp_path, pool=pool)

    assert pool.acquire_calls == 1
    assert _FakeSpawner.instances == [], (
        "a runner was cold-spawned — this test no longer exercises the "
        "pool path it exists to guard"
    )
    assert env.runtime_limits is not None, (
        "the POOL-served session's bootstrap envelope carries no caps; "
        "this is issue #735 exactly"
    )
    assert env.runtime_limits["tool_timeout_seconds"] == 120.0
    assert env.runtime_limits["max_output_bytes"] == 8192


def test_cold_spawned_session_gets_the_same_cap(daemon_loop, tmp_path,
                                                monkeypatch):
    """The other half, on the same vehicle.

    The issue's table marked cold-spawn ✅ because it forwards the env
    pair.  It does — to an executor no bootstrapped session dispatches
    through.  Asserting the ENVELOPE here rather than the spawn kwargs
    is what makes this a claim about the session's caps instead of about
    a string in ``os.environ``.
    """
    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "false")
    server = _CapturingServer(_profile(_caps()))

    env = _spawn_and_bootstrap(server, daemon_loop, tmp_path, pool=None)

    assert len(_FakeSpawner.instances) == 1, "expected a cold spawn"
    assert env.runtime_limits["tool_timeout_seconds"] == 120.0
    assert env.runtime_limits["max_output_bytes"] == 8192


def test_both_paths_deliver_identical_caps(daemon_loop, tmp_path,
                                           monkeypatch):
    """Same profile, two spawn paths, one answer.

    The issue's complaint is not only that the cap was missing but that
    it was *conditionally* missing — "same profile, two different
    behaviours depending on pool occupancy, with nothing in the logs
    saying which one you got".
    """
    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "true")
    pooled = _spawn_and_bootstrap(
        _CapturingServer(_profile(_caps())), daemon_loop, tmp_path,
        pool=_FakePoolManager())
    monkeypatch.setenv("JAATO_RUNNER_POOL_ENABLED", "false")
    cold = _spawn_and_bootstrap(
        _CapturingServer(_profile(_caps())), daemon_loop, tmp_path,
        pool=None)
    assert pooled.runtime_limits == cold.runtime_limits
