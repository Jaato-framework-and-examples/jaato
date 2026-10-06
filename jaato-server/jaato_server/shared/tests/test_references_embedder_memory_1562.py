"""The references plugin loads, and keeps, an embedding model only for a session that can use it (#1562, #1563, #1565).

#1482 deferred the model when no bundle had a vector index. Three ways it
was still loaded, or kept, for nothing:

1. **#1562, an index built with another model.** Bootstrap was eager for
   any indexed bundle, then ``_attach_matcher`` skipped every bundle for
   the model mismatch: ~680 MB and ~11 s spent attaching nothing. Eager
   now needs an indexed bundle whose ``embedding_model`` equals the model
   the provider would load (the configured ``embedding_model``, else the
   provider's own default, read from a provider that is constructed and
   NOT loaded). ``_init_bundle_matchers`` does not load the model to
   attach nothing either.
2. **#1563, a profile that does not enable the plugin.** The runner
   initializes every discovered plugin. It now records the profile's
   plugin list on the registry, which stamps ``session_enables_plugin``
   into each plugin's config; ``False`` defers the embedder.
3. **#1565, a pool slot.** ``reset_for_next_session`` was a no-op, so the
   provider (torch plus weights) stayed on the slot. It is released now,
   and a later semantic call reloads it.

No real model is used: ``discover_embedding_subsystem`` is replaced with a
factory that counts constructions and loads.
"""

from __future__ import annotations

import ast
import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from jaato_server.server.runner.session import _record_session_plugins
from jaato_server.shared.plugins.references import plugin as plugin_mod
from jaato_server.shared.plugins.references.embedding_types import EmbeddingResult
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.plugins.registry import (
    SESSION_ENABLES_PLUGIN_KEY,
    PluginRegistry,
)
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_REGISTRY = "jaato-server/jaato_server/shared/plugins/registry.py"

DEFAULT_MODEL = "all-MiniLM-L6-v2"
OTHER_MODEL = "BAAI/bge-small-en-v1.5"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="        verdict.eager = verdict.compatible > 0\n",
        replace="        verdict.eager = True\n",
        because="an index built with another model would load the provider's model for nothing (#1562)",
        test="TestModelCompatibility::test_configured_model_mismatch_defers",
    ),
    Reversion(
        target=_PLUGIN,
        find="                and not self._compatible_indexed_bundle_count(provider_model)):\n",
        replace="                and False):\n",
        because="attaching matchers would load the model only to skip every bundle (#1562)",
        test="TestModelCompatibility::test_default_model_mismatch_constructs_but_never_loads",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if self._session_enables_plugin is False:\n",
        replace="        if False:\n",
        because="a session that cannot reach the plugin would still load the model (#1563)",
        test="TestSessionEnables::test_not_enabled_defers",
    ),
    Reversion(
        target=_REGISTRY,
        find="        if name is not None and self._session_plugins is not None:\n",
        replace="        if False:\n",
        because="the plugin would never be told the session does not enable it (#1563)",
        test="TestSessionEnables::test_registry_stamps_whether_the_profile_lists_the_plugin",
    ),
    Reversion(
        target=_PLUGIN,
        find='        self._release_embedder("reset_for_next_session")\n',
        replace="        pass\n",
        because="the model would stay resident on a pool slot into the next session (#1565)",
        test="TestReset::test_reset_releases_the_provider",
    ),
    Reversion(
        target=_PLUGIN,
        find="        _ask_provider_to_release(provider)\n",
        replace="",
        because="a provider offering an explicit unload would never be asked for it (#1565)",
        test="TestReset::test_reset_asks_the_provider_to_unload",
    ),
]


class FakeProvider:
    """Counts loads; ``model_name`` is the provider's default model."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.dimensions = 3
        self.available = False
        self.loads = 0
        self.unloaded = 0

    def load_model(self) -> bool:
        self.loads += 1
        self.available = True
        return True

    def unload_model(self) -> None:
        self.unloaded += 1
        self.available = False

    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(embedding=[0.1, 0.2, 0.3], model=self.model_name, dimensions=3)

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def factory(monkeypatch, tmp_path):
    """Counting discovery; holds providers WEAKLY so release is observable."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    for var in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME"):
        monkeypatch.delenv(var, raising=False)
    state: Dict[str, Any] = {"built": [], "loads": [], "unloads": []}

    def discover(config):
        provider = FakeProvider()
        state["built"].append(weakref.ref(provider))
        state["last"] = provider  # strong ref the test drops when it wants
        return provider, None

    monkeypatch.setattr(plugin_mod, "discover_embedding_subsystem", discover)
    return state


def _workspace(tmp_path: Path, model: str) -> Path:
    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"
    refs.mkdir(parents=True)
    (refs / "bundle.json").write_text("{}")
    (refs / "embedding_config.json").write_text(json.dumps({
        "embedding_model": model, "embedding_dimensions": 3,
        "embedding_sidecar": "refs.npy", "rows": [], "reconcile_mode": "off",
    }))
    return ws


def _plugin(ws: Path, **extra: Any) -> ReferencesPlugin:
    plugin = ReferencesPlugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "hybrid", **extra})
    return plugin


def _loads(state: Dict[str, Any]) -> int:
    return sum(p.loads for p in (r() for r in state["built"]) if p is not None)


class TestModelCompatibility:
    def test_configured_model_mismatch_defers(self, tmp_path, factory, caplog):
        with caplog.at_level("INFO"):
            plugin = _plugin(_workspace(tmp_path, OTHER_MODEL), embedding_model=DEFAULT_MODEL)
        assert plugin._indexed_bundle_count() == 1
        assert factory["built"] == []
        assert plugin._embedding_provider is None
        line = next(r.getMessage() for r in caplog.records if "embedding provider" in r.getMessage())
        assert "deferred" in line and "indexed_bundles=1/1 compatible=0" in line

    def test_default_model_mismatch_constructs_but_never_loads(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, OTHER_MODEL))
        assert len(factory["built"]) == 1  # constructed to learn its default
        assert _loads(factory) == 0
        assert all(b.matcher is None for b in plugin._bundles)

    def test_matching_model_is_eager(self, tmp_path, factory, caplog):
        with caplog.at_level("INFO"):
            plugin = _plugin(_workspace(tmp_path, DEFAULT_MODEL))
        assert _loads(factory) == 1
        assert plugin._embedding_provider is not None
        assert any("compatible=1" in r.getMessage() for r in caplog.records)

    def test_deferred_provider_still_loads_for_a_later_caller(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, OTHER_MODEL))
        assert plugin.embed_texts(["x"])["ok"]
        assert _loads(factory) == 1


class TestSessionEnables:
    def test_not_enabled_defers(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, DEFAULT_MODEL),
                         **{SESSION_ENABLES_PLUGIN_KEY: False})
        assert factory["built"] == []
        assert plugin._embedding_provider is None
        # Deferral, not refusal: a caller that needs vectors still gets them.
        assert plugin.embed_texts(["x"])["ok"]
        assert _loads(factory) == 1

    def test_enabled_or_unknown_is_eager(self, tmp_path, factory):
        _plugin(_workspace(tmp_path, DEFAULT_MODEL), **{SESSION_ENABLES_PLUGIN_KEY: True})
        assert _loads(factory) == 1

    def test_registry_stamps_whether_the_profile_lists_the_plugin(self):
        registry = PluginRegistry()
        assert registry._augment_plugin_config({"a": 1}, "references") == {"a": 1}
        registry.set_session_plugins(["cli"])
        assert registry._augment_plugin_config(None, "references")[SESSION_ENABLES_PLUGIN_KEY] is False
        assert registry._augment_plugin_config({}, "cli")[SESSION_ENABLES_PLUGIN_KEY] is True
        # An operator value wins, as for every framework key.
        assert registry._augment_plugin_config(
            {SESSION_ENABLES_PLUGIN_KEY: True}, "references")[SESSION_ENABLES_PLUGIN_KEY] is True

    def test_runner_records_the_profiles_plugins(self):
        registry = PluginRegistry()
        _record_session_plugins(registry, SimpleNamespace(plugins=[{"name": "cli"}, {"name": "todo"}]))
        assert registry._session_plugins == frozenset({"cli", "todo"})
        unknown = PluginRegistry()
        _record_session_plugins(unknown, SimpleNamespace(plugins=None))
        assert unknown._session_plugins is None

    def test_runner_records_before_initializing_plugins(self):
        path = Path(__file__).resolve().parents[2] / "server" / "runner" / "session.py"
        tree = ast.parse(path.read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_configure_runtime_plugins")
        lines = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                lines.setdefault(name, node.lineno)
        assert "_record_session_plugins" in lines
        assert lines["_record_session_plugins"] < lines["expose_all"]


class TestReset:
    def test_reset_releases_the_provider(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, DEFAULT_MODEL))
        ref = factory["built"][0]
        del factory["last"]
        assert plugin._embedding_provider is not None
        plugin.reset_for_next_session()
        assert plugin._embedding_provider is None
        assert all(b.matcher is None for b in plugin._bundles)
        gc.collect()
        assert ref() is None, "something still holds the embedding provider"
        # A later semantic call reloads it.
        assert plugin.embed_texts(["x"])["ok"]
        assert len(factory["built"]) == 2 and _loads(factory) == 1

    def test_reset_asks_the_provider_to_unload(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, DEFAULT_MODEL))
        provider = factory["last"]
        plugin.reset_for_next_session()
        assert provider.unloaded == 1

    def test_shutdown_releases_and_forgets_the_config(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, DEFAULT_MODEL))
        provider = factory["last"]
        plugin.shutdown()
        assert plugin._embedding_provider is None and provider.unloaded == 1
        assert plugin._cached_init_config is None
