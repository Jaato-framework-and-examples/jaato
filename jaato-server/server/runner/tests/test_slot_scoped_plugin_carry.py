"""Slot-scoped plugin carry-over across cascade session boundaries (#890).

The bug these pin: ``reset_for_next_session()`` preserved state on a plugin
instance that the next ``session.bootstrap`` discarded.  The pool slot was
reused, but every bootstrap built a fresh ``PluginRegistry`` and re-ran
``discover()`` — which calls ``create_plugin()`` — and nothing shut the
outgoing registry down.  So a plugin owning an OS process (``lsp`` owns a
language server) started a new one per cascade stage and left the old one
running with no owner: three live jdtls, ~2.3 GB, on a 5-subphase run.

The tests are written around a fake plugin that counts starts and stops the
way jdtls would, so "one server per cascade, not one per stage" is asserted
directly rather than inferred from which methods were called.
"""

from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.base import TRAIT_SLOT_SCOPED
from server.runner import slot_plugins
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from shared.plugins.registry import PluginRegistry
from shared.session_envelope import SessionInitEnvelope


@pytest.fixture(autouse=True)
def _clean_slot_store():
    """Each test starts and ends with an empty process-level store."""
    slot_plugins.release_all(reason="test setup")
    yield
    slot_plugins.release_all(reason="test teardown")


# ======================================================================
# Fakes
# ======================================================================


class _ServerOwningPlugin:
    """Stands in for ``lsp``: ``initialize()`` starts an expensive server.

    ``servers_started`` is the number this instance ever launched;
    ``servers_running`` is how many are still alive.  Together they express
    both halves of #890 — the cold-start tax (starts) and the leak
    (running at the end of the cascade).
    """

    plugin_traits = frozenset({TRAIT_SLOT_SCOPED})

    def __init__(self, name: str = "lsp") -> None:
        self.name = name
        self._initialized = False
        self.servers_started = 0
        self.servers_running = 0
        self.session_ids: List[Optional[str]] = []
        self.resets = 0
        self.config: Dict[str, Any] = {}

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self._initialized:
            return  # the guard that makes carry-over free
        self.config = dict(config or {})
        self.session_ids.append(self.config.get("session_id"))
        self._initialized = True
        self.servers_started += 1
        self.servers_running += 1

    def shutdown(self) -> None:
        self._initialized = False
        self.servers_running = 0

    def reset_for_next_session(self) -> None:
        self.resets += 1

    def set_session_id(self, session_id: Optional[str]) -> None:
        self.session_ids.append(session_id)

    # Minimal ToolPlugin surface the registry touches.
    def get_tool_schemas(self):
        return []

    def get_executors(self):
        return {}


class _PerSessionPlugin:
    """A plugin that does NOT declare the trait — must not be carried."""

    def __init__(self, name: str = "references") -> None:
        self.name = name
        self._initialized = False
        self.shutdowns = 0

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self.shutdowns += 1
        self._initialized = False

    def reset_for_next_session(self) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def get_executors(self):
        return {}


def _envelope(
    session_id: str,
    *,
    cascade: Optional[str] = "cascade-880f27a9",
    workspace: Optional[str] = "/tmp/ws",
) -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id=session_id,
        workspace_path=workspace,
        profile_name="codegen",
        provider_name="openrouter",
        model_name="openrouter_haiku",
        plugins=[],
        cascade_driver_id=cascade,
    )


def _registry_with(plugins: Dict[str, Any], configs: Dict[str, Any]) -> PluginRegistry:
    """A real registry holding pre-built plugins, exposed as bootstrap would."""
    registry = PluginRegistry()
    for name, plugin in plugins.items():
        registry.adopt_plugin(name, plugin)
        registry._adopted.discard(name)  # these were "discovered", not carried
    for name in plugins:
        registry.expose_tool(name, configs.get(name))
    return registry


# ======================================================================
# The reported symptom
# ======================================================================


def test_one_language_server_across_three_cascade_stages() -> None:
    """The issue's own measurement, inverted into an assertion.

    Three sequential sessions on one slot must start ONE server and leave
    none running afterwards — not three started and three alive.
    """
    lsp = _ServerOwningPlugin()
    configs = {"lsp": {"languageServers": {"java": {}}}}

    for i, sid in enumerate(("20260908_171006", "20260908_171228", "20260908_171429")):
        env = _envelope(sid)
        registry = PluginRegistry()
        adopted = slot_plugins.adopt_into(registry, env, configs)
        if not adopted:
            registry.adopt_plugin("lsp", lsp)
            registry._adopted.discard("lsp")
        registry.set_session_id(sid)
        registry.expose_tool("lsp", configs["lsp"])

        assert lsp.servers_started == 1, (
            f"stage {i}: a second server was started — the carried instance "
            f"was not adopted"
        )
        slot_plugins.park_from(registry, env)

    assert lsp.servers_running == 1, "the warm server should still be parked"
    slot_plugins.release_all(reason="cascade complete")
    assert lsp.servers_running == 0, "slot teardown must reap the server"


def test_dropped_registry_no_longer_leaks_a_running_server() -> None:
    """A plugin that is NOT carried is shut down, not abandoned.

    Step 4 of the issue's causal chain: nothing called ``shutdown()`` on the
    outgoing registry, so a per-session plugin's resources outlived it.
    """
    per_session = _PerSessionPlugin()
    registry = _registry_with({"references": per_session}, {})

    slot_plugins.park_from(registry, _envelope("s1"))

    assert per_session.shutdowns == 1
    assert slot_plugins.carried_names() == []


# ======================================================================
# When carry-over must NOT happen
# ======================================================================


def test_different_workspace_does_not_inherit_a_warm_server() -> None:
    """A slot recycled onto another tree gets a fresh plugin."""
    lsp = _ServerOwningPlugin()
    registry = _registry_with({"lsp": lsp}, {})
    slot_plugins.park_from(registry, _envelope("s1", workspace="/tmp/ws-a"))
    assert slot_plugins.carried_names() == ["lsp"]

    arriving = PluginRegistry()
    adopted = slot_plugins.adopt_into(
        arriving, _envelope("s2", workspace="/tmp/ws-b"), {},
    )

    assert adopted == []
    assert lsp.servers_running == 0, (
        "the unclaimable instance must be shut down, not left running"
    )
    assert slot_plugins.carried_names() == []


def test_different_cascade_does_not_inherit_a_warm_server() -> None:
    lsp = _ServerOwningPlugin()
    registry = _registry_with({"lsp": lsp}, {})
    slot_plugins.park_from(registry, _envelope("s1", cascade="cascade-aaa"))

    arriving = PluginRegistry()
    adopted = slot_plugins.adopt_into(
        arriving, _envelope("s2", cascade="cascade-bbb"), {},
    )

    assert adopted == []
    assert lsp.servers_running == 0


def test_standalone_session_parks_nothing() -> None:
    """No cascade means no next stage to keep the state for."""
    lsp = _ServerOwningPlugin()
    registry = _registry_with({"lsp": lsp}, {})

    parked, errors = slot_plugins.park_from(registry, _envelope("s1", cascade=None))

    assert parked == []
    assert errors == []
    assert lsp.servers_running == 0


def test_changed_plugin_config_forces_a_fresh_instance() -> None:
    """A profile that redeclares its servers must not reuse the old ones."""
    lsp = _ServerOwningPlugin()
    registry = _registry_with({"lsp": lsp}, {"lsp": {"languageServers": {"java": {}}}})
    slot_plugins.park_from(registry, _envelope("s1"))

    arriving = PluginRegistry()
    adopted = slot_plugins.adopt_into(
        arriving, _envelope("s2"), {"lsp": {"languageServers": {"python": {}}}},
    )

    assert adopted == []
    assert lsp.servers_running == 0


def test_session_identity_keys_do_not_defeat_the_fingerprint() -> None:
    """session_id / agent_name differ every session by construction.

    Counting them would make every comparison a miss and the carry-over
    dead code, which is the bug wearing a different hat.
    """
    lsp = _ServerOwningPlugin()
    registry = _registry_with(
        {"lsp": lsp},
        {"lsp": {"languageServers": {"java": {}}, "session_id": "s1",
                 "agent_name": "codegen-1.2", "workspace_path": "/tmp/ws"}},
    )
    slot_plugins.park_from(registry, _envelope("s1"))

    arriving = PluginRegistry()
    adopted = slot_plugins.adopt_into(
        arriving, _envelope("s2"),
        {"lsp": {"languageServers": {"java": {}}, "session_id": "s2",
                 "agent_name": "codegen-2.1", "workspace_path": "/tmp/ws"}},
    )

    assert adopted == ["lsp"]
    assert lsp.servers_running == 1


# ======================================================================
# Adoption mechanics on the registry
# ======================================================================


def test_adopted_plugin_survives_discovery() -> None:
    """Discovery must not construct a rival to a carried instance.

    Both discovery paths skip a name that is already registered; adoption
    relies on that, so it has to run BEFORE ``discover()``.
    """
    registry = PluginRegistry()
    carried = _ServerOwningPlugin(name="todo")
    registry.adopt_plugin("todo", carried)

    registry.discover(tier_filter="runner")

    assert registry.get_plugin("todo") is carried
    assert registry.is_adopted("todo")


def test_adopting_does_not_re_initialize() -> None:
    """``expose_all`` still runs; the plugin's own guard is what saves the
    cold start."""
    lsp = _ServerOwningPlugin()
    lsp.initialize({"session_id": "s1"})
    assert lsp.servers_started == 1

    registry = PluginRegistry()
    registry.adopt_plugin("lsp", lsp)
    registry.set_session_id("s2")
    registry.expose_tool("lsp", {"session_id": "s2"})

    assert lsp.servers_started == 1
    assert lsp.servers_running == 1


def test_set_session_id_is_broadcast_to_plugins_that_track_it() -> None:
    """A carried plugin's initialize() early-returns, so nothing else
    refreshes the session identity it logs under."""
    lsp = _ServerOwningPlugin()
    registry = PluginRegistry()
    registry.adopt_plugin("lsp", lsp)

    registry.set_session_id("20260908_171228")

    assert lsp.session_ids[-1] == "20260908_171228"


def test_shutdown_all_skips_what_the_caller_is_carrying() -> None:
    lsp = _ServerOwningPlugin()
    other = _PerSessionPlugin()
    registry = _registry_with({"lsp": lsp, "references": other}, {})

    errors = registry.shutdown_all(skip={"lsp"})

    assert errors == []
    assert lsp.servers_running == 1
    assert other.shutdowns == 1


def test_shutdown_all_reports_but_does_not_re_raise() -> None:
    """A failed teardown must not wedge the slot."""

    class _Exploding(_PerSessionPlugin):
        def shutdown(self) -> None:
            raise RuntimeError("boom")

    registry = _registry_with({"references": _Exploding()}, {})
    assert registry.shutdown_all() == ["references"]


# ======================================================================
# session.end wiring
# ======================================================================


class _FakeRegistry:
    def __init__(self, plugins: Dict[str, Any]) -> None:
        self._plugins = plugins
        self._configs: Dict[str, Any] = {}
        self._enrichment_only: set = set()
        self.skipped: Optional[set] = None

    def list_available(self) -> List[str]:
        return list(self._plugins)

    def get_plugin(self, name: str):
        return self._plugins.get(name)

    def get_plugin_source(self, name: str):
        return None

    def shutdown_all(self, skip=None) -> List[str]:
        self.skipped = set(skip or ())
        for name, plugin in self._plugins.items():
            if name not in self.skipped:
                plugin.shutdown()
        return []


def _rpc_with(plugins: Dict[str, Any], envelope: SessionInitEnvelope) -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()
    rpc = RunnerRPC(a, lambda name, args: (False, {"error": "no executor"}))
    runtime = MagicMock()
    runtime.registry = _FakeRegistry(plugins)
    session = MagicMock()
    session._runtime = runtime
    rpc._session_host = RunnerSessionHost(
        envelope=envelope, runtime=runtime, session=session,
    )
    return rpc


def test_session_end_parks_slot_scoped_and_shuts_the_rest_down() -> None:
    lsp = _ServerOwningPlugin()
    lsp.initialize({})
    other = _PerSessionPlugin()
    rpc = _rpc_with({"lsp": lsp, "references": other}, _envelope("s1"))

    ok, result = rpc._handle_session_end()

    assert ok is True
    assert result["errors"] == []
    assert result["plugins_carried"] == ["lsp"]
    assert lsp.resets == 1, "the reset contract still runs on the carried instance"
    assert lsp.servers_running == 1
    assert other.shutdowns == 1


def test_session_end_with_reset_errors_parks_nothing() -> None:
    """The daemon does not pool a slot whose reset raised, so parking for a
    next session that will never arrive would only defer the teardown."""

    class _BadReset(_ServerOwningPlugin):
        def reset_for_next_session(self) -> None:
            raise RuntimeError("boom")

    lsp = _BadReset()
    lsp.initialize({})
    rpc = _rpc_with({"lsp": lsp}, _envelope("s1"))

    ok, result = rpc._handle_session_end()

    assert ok is True
    assert result["errors"]
    assert result["plugins_carried"] == []
    assert lsp.servers_running == 0
    assert slot_plugins.carried_names() == []


def test_session_shutdown_releases_everything() -> None:
    """The cold path: the slot is not going back to the pool."""
    parked = _ServerOwningPlugin()
    parked.initialize({})
    live = _ServerOwningPlugin(name="todo")
    live.initialize({})

    registry = _registry_with({"lsp": parked}, {})
    slot_plugins.park_from(registry, _envelope("s1"))
    assert slot_plugins.carried_names() == ["lsp"]

    rpc = _rpc_with({"todo": live}, _envelope("s2"))
    ok, _result = rpc._handle_session_shutdown()

    assert ok is True
    assert parked.servers_running == 0, "parked instance must be reaped"
    assert live.servers_running == 0, "the current registry must be reaped too"
    assert slot_plugins.carried_names() == []
