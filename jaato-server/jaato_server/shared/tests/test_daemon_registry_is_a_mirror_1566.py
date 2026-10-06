"""The daemon's registry of a runner-served session is a resource-less mirror (#1566).

On the default path a session's tools run in its runner, against the
runner's own registry.  The daemon built a second registry for every
``session.new`` and initialized every plugin in it, so each session paid
discovery twice and started its per-session process resources twice: an
MCP thread that connected every ``.mcp.json`` server, an LSP thread with
its language servers, the interactive-shell reaper and, for a profile
listing ``references`` with a compatible indexed bundle, an embedding
model in the DAEMON process, which never runs reference matching.

The daemon still reads those instances (the AppArmor rule walk, command
completions, tool status, auth and session-independent commands,
``daemon_callable`` bodies, the subagent / todo / permission lookups and
the plugin-status replay), so the registry keeps discovering, initializing
and exposing every plugin.  What changed:

* ``JaatoServer._mark_registry_runner_hosted`` marks the registry when the
  session has a runner, before ``expose_all``;
* the registry stamps ``runner_hosted_session: True`` into each plugin's
  config (never settable from a profile);
* ``mcp`` / ``lsp`` start no thread, ``interactive_shell`` no reaper, and
  ``references`` never loads its embedding model, eagerly or on demand;
* the entry-point scan, the largest single cost of discovery, is memoised
  per process and invalidated by any ``sys.path`` directory's mtime.

No real model, MCP server or language server is used.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from jaato_server.shared.plugins import registry as registry_mod
from jaato_server.shared.plugins.interactive_shell.plugin import InteractiveShellPlugin
from jaato_server.shared.plugins.lsp.plugin import LSPToolPlugin
from jaato_server.shared.plugins.mcp.plugin import MCPToolPlugin
from jaato_server.shared.plugins.references import plugin as references_mod
from jaato_server.shared.plugins.references.embedding_types import EmbeddingResult
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.plugins.registry import (
    RUNNER_HOSTED_SESSION_KEY,
    PluginRegistry,
    is_runner_hosted_mirror,
    scan_entry_points,
)
from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"
_REGISTRY = "jaato-server/jaato_server/shared/plugins/registry.py"
_MCP = "jaato-server/jaato_server/shared/plugins/mcp/plugin.py"
_LSP = "jaato-server/jaato_server/shared/plugins/lsp/plugin.py"
_SHELL = "jaato-server/jaato_server/shared/plugins/interactive_shell/plugin.py"
_REFS = "jaato-server/jaato_server/shared/plugins/references/plugin.py"

MODEL = "all-MiniLM-L6-v2"

REVERSIONS = [
    Reversion(
        target=_CORE,
        find="                        self._mark_registry_runner_hosted()\n",
        replace="",
        test="TestTheDaemonMarksItsRegistry::test_initialize_marks_before_expose_all",
        because="the daemon would initialize every plugin of a runner-served "
                "session as if it ran the tools itself",
    ),
    Reversion(
        target=_CORE,
        find="            setter(self._runner_rpc is not None)\n",
        replace="            setter(False)\n",
        test="TestTheDaemonMarksItsRegistry::test_a_runner_marks_the_registry",
        because="the mark would not follow whether a runner serves the session",
    ),
    Reversion(
        target=_REGISTRY,
        find="            augmented[RUNNER_HOSTED_SESSION_KEY] = True\n",
        replace="            pass\n",
        test="TestTheStamp::test_a_marked_registry_stamps_every_plugin",
        because="no plugin would be told it is the daemon's mirror",
    ),
    Reversion(
        target=_REGISTRY,
        find="        augmented.pop(RUNNER_HOSTED_SESSION_KEY, None)\n",
        replace="",
        test="TestTheStamp::test_a_profile_cannot_supply_the_stamp",
        because="a profile could strip the runner's own copy of its resources",
    ),
    Reversion(
        target=_MCP,
        find="        if is_runner_hosted_mirror(config):\n",
        replace="        if False:\n",
        test="TestPluginsHonourTheStamp::test_mcp_starts_no_thread",
        because="every session would connect its MCP servers twice",
    ),
    Reversion(
        target=_LSP,
        find="        if is_runner_hosted_mirror(config):\n",
        replace="        if False:\n",
        test="TestPluginsHonourTheStamp::test_lsp_starts_no_thread",
        because="every session would start its language servers twice",
    ),
    Reversion(
        target=_SHELL,
        find="        if not is_runner_hosted_mirror(config):\n",
        replace="        if True:\n",
        test="TestPluginsHonourTheStamp::test_interactive_shell_starts_no_reaper",
        because="the daemon would keep a reaper thread per session for shells it never spawns",
    ),
    Reversion(
        target=_REFS,
        find=("        if self._runner_hosted_mirror:\n"
              "            verdict.reason"),
        replace=("        if False:\n"
                 "            verdict.reason"),
        test="TestTheDaemonNeverLoadsAnEmbeddingModel::test_bootstrap_does_not_load",
        because="a profile listing references with a compatible index would "
                "load the model in the daemon (#1566)",
    ),
    Reversion(
        target=_REFS,
        find=("        if self._runner_hosted_mirror:\n"
              "            # The daemon's copy"),
        replace=("        if False:\n"
                 "            # The daemon's copy"),
        test="TestTheDaemonNeverLoadsAnEmbeddingModel::test_no_caller_loads_it_later",
        because="a later semantic call would load the model in the daemon",
    ),
    Reversion(
        target=_REGISTRY,
        find="    if cached is not None:\n        return cached\n",
        replace="",
        test="TestTheEntryPointScanIsMemoised::test_a_second_scan_reuses_the_first",
        because="every registry would rescan every distribution's metadata",
    ),
]


# --------------------------------------------------------------- the daemon


def _bare_server(runner_rpc: Any):
    from jaato_server.server.core import JaatoServer
    server = JaatoServer.__new__(JaatoServer)
    server.registry = PluginRegistry(model_name="x")
    server._runner_rpc = runner_rpc
    return server


class TestTheDaemonMarksItsRegistry:
    def test_a_runner_marks_the_registry(self):
        server = _bare_server(runner_rpc=object())
        server._mark_registry_runner_hosted()
        assert server.registry._runner_hosted is True

    def test_no_runner_leaves_it_unmarked(self):
        server = _bare_server(runner_rpc=None)
        server._mark_registry_runner_hosted()
        assert server.registry._runner_hosted is False

    def test_initialize_marks_before_expose_all(self):
        """``_run_load_plugins`` marks the registry, then exposes it.

        Asserted on the call site: ``initialize()`` needs a provider and a
        runner, and what the defect would be is a missing call.
        """
        source = (Path(__file__).resolve().parents[2] / "server" / "core.py").read_text()
        tree = ast.parse(source)
        load = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "_run_load_plugins")
        calls = [(c.lineno, c.func.attr) for c in ast.walk(load)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
        mark = [ln for ln, name in calls if name == "_mark_registry_runner_hosted"]
        expose = [ln for ln, name in calls if name == "expose_all"]
        assert mark and expose and min(mark) < min(expose)


# ---------------------------------------------------------------- the stamp


class TestTheStamp:
    def test_a_marked_registry_stamps_every_plugin(self):
        reg = PluginRegistry(model_name="x")
        reg.set_runner_hosted(True)
        cfg = reg._augment_plugin_config({"a": 1}, "mcp")
        assert is_runner_hosted_mirror(cfg)
        assert is_runner_hosted_mirror(reg._augment_plugin_config(None, "lsp"))

    def test_an_unmarked_registry_stamps_nothing(self):
        reg = PluginRegistry(model_name="x")
        reg.set_workspace_path("/tmp/ws")
        cfg = reg._augment_plugin_config({"a": 1}, "mcp")
        assert RUNNER_HOSTED_SESSION_KEY not in cfg
        assert reg._augment_plugin_config({"a": 1}, "mcp") is not None

    def test_a_profile_cannot_supply_the_stamp(self):
        reg = PluginRegistry(model_name="x")
        reg.set_workspace_path("/tmp/ws")
        cfg = reg._augment_plugin_config({RUNNER_HOSTED_SESSION_KEY: True}, "mcp")
        assert not is_runner_hosted_mirror(cfg)

    def test_only_the_literal_true_counts(self):
        assert not is_runner_hosted_mirror({RUNNER_HOSTED_SESSION_KEY: "yes"})
        assert not is_runner_hosted_mirror(None)
        assert is_runner_hosted_mirror({RUNNER_HOSTED_SESSION_KEY: True})


# ------------------------------------------------------------ the plugins


def _mirror(ws: Path) -> Dict[str, Any]:
    return {"workspace_path": str(ws), "session_id": "s1",
            RUNNER_HOSTED_SESSION_KEY: True}


class TestPluginsHonourTheStamp:
    """Each case has a control: the same call without the stamp starts it."""

    @pytest.mark.parametrize("cls", [MCPToolPlugin, LSPToolPlugin])
    def test_control_without_the_stamp_starts_the_thread(self, cls, tmp_path, monkeypatch):
        started: List[bool] = []
        monkeypatch.setattr(cls, "_ensure_thread", lambda self: started.append(True))
        plugin = cls()
        plugin.initialize({"workspace_path": str(tmp_path)})
        assert started == [True]

    def test_mcp_starts_no_thread(self, tmp_path, monkeypatch):
        started: List[bool] = []
        monkeypatch.setattr(MCPToolPlugin, "_ensure_thread", lambda self: started.append(True))
        plugin = MCPToolPlugin()
        plugin.initialize(_mirror(tmp_path))
        assert started == [] and plugin._initialized
        assert plugin.get_command_completions("mcp", [])  # completions still answer

    def test_lsp_starts_no_thread(self, tmp_path, monkeypatch):
        started: List[bool] = []
        monkeypatch.setattr(LSPToolPlugin, "_ensure_thread", lambda self: started.append(True))
        plugin = LSPToolPlugin()
        plugin.initialize(_mirror(tmp_path))
        assert started == [] and plugin._initialized

    def test_interactive_shell_starts_no_reaper(self, tmp_path, monkeypatch):
        started: List[bool] = []
        monkeypatch.setattr(InteractiveShellPlugin, "_start_reaper",
                            lambda self: started.append(True))
        InteractiveShellPlugin().initialize({"workspace_root": str(tmp_path)})
        assert started == [True]  # control
        started.clear()
        InteractiveShellPlugin().initialize(
            {"workspace_root": str(tmp_path), RUNNER_HOSTED_SESSION_KEY: True})
        assert started == []

    def test_a_marked_registry_starts_no_thread_and_exposes_the_same_plugins(self, tmp_path):
        """End to end: the real discovery, both ways, in one empty workspace."""
        (tmp_path / ".jaato").mkdir()

        def build(marked: bool):
            before = {t.ident for t in threading.enumerate()}
            reg = PluginRegistry(model_name="x")
            reg.discover()
            reg.set_workspace_path(str(tmp_path))
            reg.set_config_root(str(tmp_path / ".jaato"))
            reg.set_session_id("s1")
            reg.set_runner_hosted(marked)
            reg.expose_all({})
            new = [t.name for t in threading.enumerate() if t.ident not in before]
            exposed = sorted(reg.list_exposed())
            reg.unexpose_all()
            return exposed, new

        exposed_mirror, threads_mirror = build(True)
        exposed_full, _threads_full = build(False)
        assert exposed_mirror == exposed_full
        assert threads_mirror == []


# --------------------------------------------------------------- references


class _FakeProvider:
    def __init__(self) -> None:
        self.model_name = MODEL
        self.dimensions = 3
        self.available = False
        self.loads = 0

    def load_model(self) -> bool:
        self.loads += 1
        self.available = True
        return True

    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(embedding=[0.1, 0.2, 0.3], model=MODEL, dimensions=3)

    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def factory(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    state: Dict[str, Any] = {"built": []}

    def discover(config):
        provider = _FakeProvider()
        state["built"].append(provider)
        return provider, None

    monkeypatch.setattr(references_mod, "discover_embedding_subsystem", discover)
    return state


def _indexed_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"
    refs.mkdir(parents=True)
    (refs / "bundle.json").write_text("{}")
    (refs / "embedding_config.json").write_text(json.dumps({
        "embedding_model": MODEL, "embedding_dimensions": 3,
        "embedding_sidecar": "refs.npy", "rows": [], "reconcile_mode": "off",
    }))
    return ws


def _references(ws: Path, **extra: Any) -> ReferencesPlugin:
    plugin = ReferencesPlugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "hybrid",
                       "embedding_model": MODEL,
                       "session_enables_plugin": True, **extra})
    return plugin


class TestTheDaemonNeverLoadsAnEmbeddingModel:
    def test_control_the_runners_copy_loads_it(self, tmp_path, factory):
        _references(_indexed_workspace(tmp_path))
        assert sum(p.loads for p in factory["built"]) == 1

    def test_bootstrap_does_not_load(self, tmp_path, factory):
        plugin = _references(_indexed_workspace(tmp_path),
                             **{RUNNER_HOSTED_SESSION_KEY: True})
        assert factory["built"] == []
        assert plugin._embedding_provider is None

    def test_no_caller_loads_it_later(self, tmp_path, factory):
        plugin = _references(_indexed_workspace(tmp_path),
                             **{RUNNER_HOSTED_SESSION_KEY: True})
        assert plugin._ensure_embedding_provider() is None
        plugin._reattach_matchers()
        assert sum(p.loads for p in factory["built"]) == 0


# ------------------------------------------------------- the entry-point scan


class TestTheEntryPointScanIsMemoised:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch, tmp_path):
        registry_mod._ENTRY_POINT_SCAN_CACHE.clear()
        self.calls: List[str] = []
        self.answer = [SimpleNamespace(name="a")]

        def fake(group=None):
            self.calls.append(group)
            return list(self.answer)

        monkeypatch.setattr(registry_mod.importlib.metadata, "entry_points", fake)
        self.dir = tmp_path / "site"
        self.dir.mkdir()
        monkeypatch.setattr(sys, "path", [str(self.dir)])
        yield
        registry_mod._ENTRY_POINT_SCAN_CACHE.clear()

    def test_a_second_scan_reuses_the_first(self):
        first = scan_entry_points("jaato.plugins")
        second = scan_entry_points("jaato.plugins")
        assert first == second and self.calls == ["jaato.plugins"]

    def test_an_install_on_sys_path_is_seen(self):
        scan_entry_points("jaato.plugins")
        self.answer = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
        (self.dir / "new_dist-1.0.dist-info").mkdir()
        st = os.stat(self.dir)
        os.utime(self.dir, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        assert [ep.name for ep in scan_entry_points("jaato.plugins")] == ["a", "b"]
        assert len(self.calls) == 2

    def test_groups_are_kept_apart(self):
        scan_entry_points("jaato.plugins")
        scan_entry_points("jaato.enrichment_plugins")
        assert self.calls == ["jaato.plugins", "jaato.enrichment_plugins"]
