"""One runner bootstrap initialises each configured plugin once (#1564).

A librarian session's runner log showed its configured plugins initialised
twice, ~0.3 s apart, and ``references`` loading its embedding model twice.
The second pass was ``JaatoSession._apply_plugin_configs`` (#950) handing
the profile's ``plugin_configs`` back to ``PluginRegistry.expose_tool``,
after ``bootstrap_session`` had already initialised every plugin through
``expose_all`` with the same blocks.

``expose_tool`` decided whether to rebuild by comparing the RAW block
against the stored config, and the stored one never matched: it had been
augmented with ``workspace_path`` / ``config_root`` / ``session_id`` /
``agent_name`` and, on the runner, merged over runner defaults such as
``references``' ``channel_type: queue``.  So every configured plugin was
shut down and re-initialised, and the second life ran WITHOUT those runner
defaults.

These tests drive the real runner configuration step
(``_configure_runtime_plugins``) and a real ``JaatoSession.configure`` on
the registry it built, with counting stand-ins for the plugins (so no
embedding model is loaded), and pin:

* the bootstrap block, applied again by the session: one ``initialize``;
* the runner default survives that second application;
* a subagent whose block genuinely differs still re-initialises (#950);
* a subagent whose block only adds its own name relabels in place (#951);
* ``permission`` is stashed for the scoped policy, never re-initialised
  through the registry (#957).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from jaato_server.server.runner import session as runner_session
from jaato_server.server.runner.session import _configure_runtime_plugins
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.registry import PluginRegistry
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_REGISTRY = "jaato-server/jaato_server/shared/plugins/registry.py"

REVERSIONS = [
    Reversion(
        target=_REGISTRY,
        find=(
            "        elif self._config_already_in_force(name, config):\n"
            "            reason = \"config already in force\"\n"
        ),
        replace="",
        test="test_the_bootstrap_block_applied_again_does_not_reinitialise",
        because="a raw block never equals the augmented stored config, so "
                "every configured plugin is initialised twice per session",
    ),
    Reversion(
        target=_REGISTRY,
        find="        effective = self._augment_plugin_config(config) or {}\n",
        replace="        effective = dict(config)\n",
        test="test_a_subagent_with_a_different_workspace_view_reinitialises",
        because="comparing un-augmented keys ignores the framework values the "
                "plugin would be re-initialised with",
    ),
    Reversion(
        target=_REGISTRY,
        find="            if key not in _IDENTITY_ONLY_CONFIG_KEYS\n        )\n",
        replace="        )\n",
        test="test_a_subagent_naming_itself_relabels_without_reinitialising",
        because="a child's own agent name, beside an unchanged block, "
                "rebuilds the parent's live plugin",
    ),
]


class _CountingPlugin:
    """A plugin that counts its lifecycle and records its configs."""

    PARALLEL_INIT = False

    def __init__(self, name: str) -> None:
        self._name = name
        self.configs: List[Optional[Dict[str, Any]]] = []
        self.shutdowns = 0
        self.agent_names: List[Any] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def initialize_calls(self) -> int:
        return len(self.configs)

    @property
    def config(self) -> Dict[str, Any]:
        return self.configs[-1] or {}

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.configs.append(dict(config) if config else None)

    def shutdown(self) -> None:
        self.shutdowns += 1

    def set_agent_name(self, name: Any) -> None:
        self.agent_names.append(name)

    def get_tool_schemas(self):
        return []

    def get_executors(self):
        return {}

    def get_user_commands(self):
        return []

    def get_auto_approved_tools(self):
        return []

    def get_system_instructions(self):
        return None


_PLUGIN_NAMES = ("references", "file_edit", "permission")


class _StubRuntime:
    """The runtime surface ``_configure_runtime_plugins`` touches."""

    def __init__(self) -> None:
        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Any = None

    def configure_plugins(self, registry, permission_plugin=None,
                          ledger=None, reliability_plugin=None) -> None:
        self.registry = registry
        self.permission_plugin = permission_plugin


PROFILE_BLOCKS: Dict[str, Dict[str, Any]] = {
    "references": {"lookup_strategy": "hybrid"},
    "file_edit": {"max_edit_span_chars": 400},
    "permission": {"policy": {"defaultPolicy": "deny"}},
}


@pytest.fixture
def bootstrapped(monkeypatch, tmp_path):
    """The registry a real runner bootstrap leaves behind, with counting
    stand-ins registered in place of the discovered plugins."""

    def _discover(self, *args, **kwargs):
        for name in _PLUGIN_NAMES:
            self._plugins.setdefault(name, _CountingPlugin(name))
        return list(_PLUGIN_NAMES)

    monkeypatch.setattr(PluginRegistry, "discover", _discover)
    # The runner seeds the PROCESS's workspace context (os.environ and a
    # ContextVar) with no reset, because a runner serves one session.  In
    # the test process that would leak into every later test, and it is
    # not what is under test here.
    monkeypatch.setattr(runner_session, "_set_runner_workspace_context",
                        lambda *_args, **_kwargs: None)
    envelope = SessionInitEnvelope(
        session_id="sess-1564",
        workspace_path=str(tmp_path),
        config_root=str(tmp_path / ".jaato"),
        profile_name="librarian",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[{"name": n} for n in _PLUGIN_NAMES],
        plugin_configs={k: dict(v) for k, v in PROFILE_BLOCKS.items()},
        agent_id="main",
    )
    runtime = _StubRuntime()
    _configure_runtime_plugins(runtime, envelope)
    assert runtime.registry is not None
    return runtime.registry


def _session(registry: PluginRegistry, **context: Any) -> JaatoSession:
    runtime = MagicMock()
    runtime.registry = registry
    session = JaatoSession(runtime, "claude-sonnet-4-6")
    if context:
        session.set_agent_context(**context)
    return session


def _configure(session: JaatoSession, blocks: Dict[str, Dict[str, Any]]):
    session.configure(
        skip_provider=True,
        plugins=list(_PLUGIN_NAMES),
        plugin_configs={k: dict(v) for k, v in blocks.items()},
    )


def _plugin(registry: PluginRegistry, name: str) -> _CountingPlugin:
    return registry._plugins[name]


def test_the_bootstrap_block_applied_again_does_not_reinitialise(bootstrapped):
    references = _plugin(bootstrapped, "references")
    assert references.initialize_calls == 1, "bootstrap initialises once"

    _configure(_session(bootstrapped), PROFILE_BLOCKS)

    assert references.initialize_calls == 1, (
        "the session re-initialised a plugin with the block it already runs "
        "under; references would load its embedding model twice"
    )
    assert references.shutdowns == 0
    assert _plugin(bootstrapped, "file_edit").initialize_calls == 1


def test_the_runner_default_survives_the_session_configure(bootstrapped):
    _configure(_session(bootstrapped), PROFILE_BLOCKS)

    references = _plugin(bootstrapped, "references")
    assert references.config.get("channel_type") == "queue"
    assert references.config.get("lookup_strategy") == "hybrid"


def test_a_subagent_with_a_genuinely_different_block_reinitialises(
        bootstrapped):
    """#950 unchanged: a child's own block reaches the shared plugin."""
    child = _session(bootstrapped, agent_type="subagent",
                     agent_name="researcher")
    _configure(child, {"references": {"lookup_strategy": "tags_only"}})

    references = _plugin(bootstrapped, "references")
    assert references.initialize_calls == 2
    assert references.shutdowns == 1
    assert references.config["lookup_strategy"] == "tags_only"


def test_a_subagent_with_a_different_workspace_view_reinitialises(
        bootstrapped, tmp_path):
    """A framework value the plugin would be re-initialised with is part of
    the comparison, so a changed one still rebuilds."""
    bootstrapped.set_workspace_path(str(tmp_path / "elsewhere"))
    _configure(_session(bootstrapped), {"file_edit": {"max_edit_span_chars": 400}})

    file_edit = _plugin(bootstrapped, "file_edit")
    assert file_edit.initialize_calls == 2
    assert file_edit.config["workspace_path"] == str(tmp_path / "elsewhere")


def test_a_subagent_naming_itself_relabels_without_reinitialising(
        bootstrapped):
    """#951 unchanged: the child's name beside an unchanged block is a
    relabel, not a rebuild."""
    child = _session(bootstrapped, agent_type="subagent",
                     agent_name="researcher")
    _configure(child, {"file_edit": {"max_edit_span_chars": 400}})

    file_edit = _plugin(bootstrapped, "file_edit")
    assert file_edit.initialize_calls == 1
    assert file_edit.shutdowns == 0
    assert file_edit.agent_names == ["researcher"]


def test_permission_is_stashed_not_reinitialised(bootstrapped):
    """#957 unchanged: the block goes to the scoped-policy route."""
    session = _session(bootstrapped)
    _configure(session, PROFILE_BLOCKS)

    assert _plugin(bootstrapped, "permission").initialize_calls == 1
    assert session._permission_config == PROFILE_BLOCKS["permission"]
