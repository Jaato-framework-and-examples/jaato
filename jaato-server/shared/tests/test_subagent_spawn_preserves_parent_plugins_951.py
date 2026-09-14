"""A subagent spawn must not rebuild the parent's live plugins (#951).

Subagents share the parent's ``PluginRegistry``, so they share each
plugin INSTANCE.  Both in-process spawn paths in
``shared/plugins/subagent/plugin.py`` stamp ``agent_name`` into the
config of every plugin the child profile *lists* — including the ones it
declares no configuration for — and ``JaatoSession._apply_plugin_configs``
hands each of those to ``registry.expose_tool``.  That stamp is
load-bearing: on a registry whose ``expose_all`` was filtered to the ROOT
profile's plugins (``requested_plugins=``), it is the only thing that
exposes a plugin the child lists and the parent never used.

What it must NOT do is reconfigure the parent.  ``expose_tool`` answered
any changed config with ``shutdown()`` + ``initialize()``, so a dict
holding nothing but the child's name REPLACED the parent's whole
configuration on a shared, live object, mid-turn.  Measured against a
real registry before the fix:

* ``file_edit`` lost ``max_edit_span_chars`` and had
  ``allow_full_replace`` turned back on — a whole-file-rewrite path the
  profile author deliberately closed, reopened by a spawn, and left open
  for the rest of the parent's life;
* every live ``interactive_shell`` PTY the parent owned was killed, its
  next ``shell_input`` answered ``No session with id 'session_0'``.

The fix is ``PluginRegistry._config_requires_reinit``: a config whose
only key is an identity key relabels the plugin in place instead of
rebuilding it.  These tests pin all four halves — the parent survives,
the child's plugin is still exposed, a REAL config block still
re-initializes (#950 unchanged), and the ``subagent`` plugin's self-spawn
guard still tracks the most recent spawner.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

from shared.plugins.registry import PluginRegistry


class _StubPlugin:
    """Minimal plugin that counts its own initialize/shutdown cycles."""

    PARALLEL_INIT = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.initialize_calls: List[Optional[Dict[str, Any]]] = []
        self.shutdown_calls = 0
        self.setting: Optional[str] = None
        self.live_state = "parent-owned"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.initialize_calls.append(config)
        self.setting = (config or {}).get("setting")
        self._initialized = True

    def get_tool_schemas(self):
        return []

    def get_executors(self):
        return {}

    def get_user_commands(self):
        return []

    def get_auto_approved_tools(self):
        return []

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.live_state = None
        self._initialized = False


def _registry_with(names: List[str]) -> PluginRegistry:
    reg = PluginRegistry()
    for name in names:
        reg._plugins[name] = _StubPlugin(name)
    return reg


# The config a spawn hands to ``expose_tool`` for a plugin the child
# profile lists but does not configure.
def _spawn_stamp(agent: str) -> Dict[str, Any]:
    return {"agent_name": agent}


class TestParentPluginsSurviveASpawn:
    """The identity stamp must not shut anything down."""

    def test_identity_only_config_does_not_rebuild_an_exposed_plugin(self):
        reg = _registry_with(["alpha"])
        reg.expose_tool("alpha", {"agent_name": "escriba", "setting": "parent"})
        alpha = reg._plugins["alpha"]

        assert reg.expose_tool("alpha", _spawn_stamp("documentalista")) is True

        assert alpha.shutdown_calls == 0, (
            "a subagent spawn tore down the parent's shared plugin instance"
        )
        assert len(alpha.initialize_calls) == 1
        assert alpha.live_state == "parent-owned"

    def test_the_parents_configuration_is_not_replaced_by_the_stamp(self):
        reg = _registry_with(["alpha"])
        reg.expose_tool("alpha", {"agent_name": "escriba", "setting": "parent"})
        alpha = reg._plugins["alpha"]

        reg.expose_tool("alpha", _spawn_stamp("documentalista"))

        assert alpha.setting == "parent", (
            "the child's name-only config replaced the parent's settings"
        )
        assert reg._configs["alpha"].get("setting") == "parent"

    def test_file_edit_keeps_the_constraints_its_profile_set(self, tmp_path):
        """The measured case, on the real plugin.

        ``allow_full_replace: False`` is a profile author closing the
        whole-file-rewrite path for a weak model; ``max_edit_span_chars``
        caps a single targeted edit.  Both used to be reset to their
        permissive defaults by any spawn.
        """
        os.makedirs(tmp_path / ".jaato", exist_ok=True)
        reg = PluginRegistry()
        reg.discover()
        reg.set_workspace_path(str(tmp_path))
        reg.set_config_root(str(tmp_path / ".jaato"))
        reg.set_agent_name("escriba")
        reg.expose_tool("file_edit", {
            "agent_name": "escriba",
            "max_edit_span_chars": 400,
            "allow_full_replace": False,
        })
        file_edit = reg.get_plugin("file_edit")
        assert file_edit._max_edit_span_chars == 400
        assert file_edit._allow_full_replace is False

        reg.expose_tool("file_edit", _spawn_stamp("documentalista"))

        assert file_edit._max_edit_span_chars == 400, (
            "a spawn removed the parent's per-edit cap"
        )
        assert file_edit._allow_full_replace is False, (
            "a spawn reopened the whole-file-rewrite path the profile closed"
        )

    def test_a_live_interactive_shell_session_survives_a_spawn(self, tmp_path):
        """The parent's PTYs belong to the parent's turn.

        ``InteractiveShellPlugin.shutdown()`` closes every session it
        holds and clears the dict, so the rebuild killed the parent's
        ``psql`` / ``ssh`` / debugger mid-conversation and the model was
        told the session id did not exist.  A stand-in session object is
        registered rather than a real PTY: what is under test is the
        plugin's own teardown running or not, and a live ``pexpect``
        child makes the assertion depend on process scheduling.
        """
        pytest.importorskip("pexpect")

        class _StandInSession:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        reg = PluginRegistry()
        reg.discover()
        reg.set_workspace_path(str(tmp_path))
        reg.expose_tool("interactive_shell", {"agent_name": "escriba"})
        shell = reg.get_plugin("interactive_shell")
        parent_pty = _StandInSession()
        shell._sessions["session_0"] = parent_pty

        reg.expose_tool("interactive_shell", _spawn_stamp("documentalista"))

        assert shell._sessions.get("session_0") is parent_pty, (
            "a subagent spawn dropped the parent's live PTY session"
        )
        assert not parent_pty.closed, (
            "a subagent spawn closed the parent's live PTY session"
        )


class TestWhatMustKeepWorking:
    """The stamp's real job, and #950's contract, are both untouched."""

    def test_a_listed_but_unconfigured_plugin_is_still_exposed(self):
        """The reason the stamp exists.

        ``expose_all(requested_plugins=...)`` initializes only the root
        profile's plugins, so a child that lists ``file_edit`` against a
        parent that never used it reaches ``expose_tool`` with nothing
        but its own name — and must still be exposed.
        """
        reg = _registry_with(["alpha"])
        assert not reg.is_exposed("alpha")

        assert reg.expose_tool("alpha", _spawn_stamp("documentalista")) is True

        assert reg.is_exposed("alpha")
        assert len(reg._plugins["alpha"].initialize_calls) == 1

    def test_a_real_config_block_still_reinitializes(self):
        """#950 unchanged: a profile's ``plugin_configs`` still land."""
        reg = _registry_with(["alpha"])
        reg.expose_tool("alpha", {"agent_name": "escriba", "setting": "parent"})
        alpha = reg._plugins["alpha"]

        reg.expose_tool("alpha", {"agent_name": "child", "setting": "child"})

        assert alpha.shutdown_calls == 1
        assert alpha.setting == "child"

    def test_an_unchanged_config_is_still_a_no_op(self):
        reg = _registry_with(["alpha"])
        reg.expose_tool("alpha", {"agent_name": "escriba", "setting": "parent"})
        alpha = reg._plugins["alpha"]
        stored = dict(reg._configs["alpha"])

        reg.expose_tool("alpha", stored)

        assert alpha.shutdown_calls == 0
        assert len(alpha.initialize_calls) == 1


class TestTheSelfSpawnGuardStillTracksTheSpawner:
    """``agent_name`` is a trace label for every plugin but one.

    On ``subagent`` it is also ``_self_profile_name``, which
    ``spawn_subagent`` compares against the profile it is asked to spawn
    in order to refuse a self-spawn loop.  Relabelling in place has to
    keep that value moving exactly as the rebuild did.
    """

    def test_relabel_reaches_the_subagent_plugins_self_spawn_guard(self):
        reg = PluginRegistry()
        reg.discover()
        reg.expose_tool("subagent", {"agent_name": "escriba"})
        subagent = reg.get_plugin("subagent")
        assert subagent._self_profile_name == "escriba"

        reg.expose_tool("subagent", _spawn_stamp("documentalista"))

        assert subagent._self_profile_name == "documentalista", (
            "the self-spawn guard stopped tracking the most recent spawner"
        )

    def test_relabelling_does_not_reset_the_subagent_registry(self):
        """``SubagentPlugin.shutdown()`` clears ``_owner_counters`` and
        drops ``_parent_session``; a spawn must not do that to the
        parent's own subagent bookkeeping."""
        reg = PluginRegistry()
        reg.discover()
        reg.expose_tool("subagent", {"agent_name": "escriba"})
        subagent = reg.get_plugin("subagent")
        sentinel = object()
        subagent._parent_session = sentinel
        subagent._owner_counters["owner-1"] = 3

        reg.expose_tool("subagent", _spawn_stamp("documentalista"))

        assert subagent._parent_session is sentinel
        assert subagent._owner_counters.get("owner-1") == 3
