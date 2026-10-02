"""A session loads the embedding model only when it can use it (#1482).

Since #1478 a driver creates UNINDEXED bundles, and ``initialize()`` loaded
the embedding provider whenever any bundle existed. So every later session
in such a workspace paid the model load (~13 s of imports plus HEAD
requests to the hub) at bootstrap, with no index to match against and even
under ``lookup_strategy: tags_only``. Bootstraps crossed their deadline.

Properties, each a way it could go wrong:

1. **No load at bootstrap under ``tags_only``**, whatever the bundles hold.
2. **No load at bootstrap with no index anywhere**, under ``hybrid``.
3. **An indexed bundle under ``hybrid`` still loads at bootstrap**
   (unchanged).
4. **A deferred provider loads exactly once**, on the first caller, even
   when several callers arrive at once.
5. **A cached model is loaded offline first**: ``local_files_only=True``
   and the hub's offline switches, restored afterwards; online only when
   that fails.
6. **A failed load is not retried on every call.**
7. **An index that appears mid-session (#1145) loads the provider then.**

No real model is used: ``discover_embedding_subsystem`` is replaced with a
factory that counts constructions.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from jaato_server.shared.plugins.references import plugin as plugin_mod
from jaato_server.shared.plugins.references.embedding_types import EmbeddingResult
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_LOAD = "jaato-server/jaato_server/shared/plugins/references/embedding_load.py"

MODEL = "all-MiniLM-L6-v2"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find='        return (self._lookup_strategy in ("hybrid", "semantic_only")\n'
             "                and self._indexed_bundle_count() > 0)\n",
        replace="        return self._indexed_bundle_count() > 0\n",
        because="a tags_only session would load a model it never queries",
        test="TestBootstrap::test_tags_only_never_loads",
    ),
    Reversion(
        target=_PLUGIN,
        find='        return (self._lookup_strategy in ("hybrid", "semantic_only")\n'
             "                and self._indexed_bundle_count() > 0)\n",
        replace='        return (self._lookup_strategy in ("hybrid", "semantic_only")\n'
                "                and bool(self._bundles))\n",
        because="any unindexed bundle would load the model at bootstrap (#1482)",
        test="TestBootstrap::test_hybrid_without_an_index_defers",
    ),
    Reversion(
        target=_PLUGIN,
        find="        with self._embedding_lock:\n",
        replace="        if True:\n",
        because="concurrent first callers would each load their own copy of the model",
        test="TestDeferredLoad::test_concurrent_first_callers_load_once",
    ),
    Reversion(
        target=_LOAD,
        find="    if model_cached_locally(name):\n",
        replace="    if False:\n",
        because="a cached model would still be loaded online, with HEAD requests",
        test="TestOfflineFirst::test_a_cached_model_loads_offline_first",
    ),
    Reversion(
        target=_PLUGIN,
        find="                self._embedding_load_failed = True\n",
        replace="",
        because="a model that cannot load would be retried (and warned about) every call",
        test="TestDeferredLoad::test_a_failed_load_is_not_retried",
    ),
]


class FakeProvider:
    """Counts loads; records how each load was asked for."""

    def __init__(self, ok: bool = True, offline_ok: bool = True) -> None:
        self.model_name = MODEL
        self.dimensions = 3
        self.available = False
        self.loads: List[Dict[str, Any]] = []
        self._ok = ok
        self._offline_ok = offline_ok

    def load_model(self, local_files_only: bool = False) -> bool:
        self.loads.append({"local_files_only": local_files_only,
                           "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")})
        time.sleep(0.02)
        if local_files_only and not self._offline_ok:
            return False
        self.available = self._ok
        return self._ok

    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(embedding=[0.1, 0.2, 0.3], model=MODEL, dimensions=3)

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def factory(monkeypatch, tmp_path):
    """Replace discovery with a counting factory; isolate the hub cache."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    for var in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME", "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    state: Dict[str, Any] = {"built": [], "kwargs": {}}

    def discover(config):
        time.sleep(0.02)  # widen the race a missing lock would lose
        provider = FakeProvider(**state["kwargs"])
        state["built"].append(provider)
        return provider, None

    monkeypatch.setattr(plugin_mod, "discover_embedding_subsystem", discover)
    return state


def _workspace(tmp_path: Path, *, indexed: bool) -> Path:
    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"
    refs.mkdir(parents=True)
    (refs / "bundle.json").write_text("{}")
    if indexed:
        _add_index(refs)
    return ws


def _add_index(refs: Path) -> None:
    (refs / "embedding_config.json").write_text(json.dumps({
        "embedding_model": MODEL, "embedding_dimensions": 3,
        "embedding_sidecar": "refs.npy", "rows": [], "reconcile_mode": "off",
    }))


def _plugin(ws: Path, strategy: str) -> ReferencesPlugin:
    plugin = ReferencesPlugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": strategy})
    return plugin


def _cache_model(tmp_path: Path) -> None:
    snap = (tmp_path / "hub-cache" / f"models--sentence-transformers--{MODEL}"
            / "snapshots" / "abc123")
    snap.mkdir(parents=True)


class TestBootstrap:
    def test_tags_only_never_loads(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, indexed=True), "tags_only")
        assert plugin._bundles
        assert factory["built"] == []
        assert plugin._embedding_provider is None

    def test_hybrid_without_an_index_defers(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, indexed=False), "hybrid")
        assert plugin._bundles and plugin._indexed_bundle_count() == 0
        assert factory["built"] == []

    def test_hybrid_with_an_index_loads_at_bootstrap(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, indexed=True), "hybrid")
        assert len(factory["built"]) == 1
        assert plugin._embedding_provider is factory["built"][0]
        assert len(factory["built"][0].loads) == 1


class TestDeferredLoad:
    def test_first_caller_loads_once(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, indexed=False), "hybrid")
        first = plugin.embed_texts(["a"])
        second = plugin._execute_compute_embedding({"input": "b"})
        probe = plugin.embed_texts([])
        assert first["ok"] and probe["ok"] and "embedding" in second
        assert len(factory["built"]) == 1
        assert len(factory["built"][0].loads) == 1

    def test_concurrent_first_callers_load_once(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, indexed=False), "tags_only")
        barrier = threading.Barrier(8)
        results: List[Any] = []

        def call():
            barrier.wait()
            results.append(plugin.embed_texts(["x"]))

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r["ok"] for r in results)
        assert len(factory["built"]) == 1
        assert len(factory["built"][0].loads) == 1

    def test_a_failed_load_is_not_retried(self, tmp_path, factory, caplog):
        factory["kwargs"] = {"ok": False}
        plugin = _plugin(_workspace(tmp_path, indexed=False), "hybrid")
        with caplog.at_level("WARNING"):
            for _ in range(3):
                assert plugin.embed_texts(["x"])["category"] == "no_provider"
        assert sum(len(p.loads) for p in factory["built"]) == 1
        assert sum("failed to load" in r.getMessage() for r in caplog.records) == 1

    def test_an_index_that_appears_mid_session_loads_then(self, tmp_path, factory):
        ws = _workspace(tmp_path, indexed=False)
        plugin = _plugin(ws, "hybrid")
        assert factory["built"] == []
        _add_index(ws / ".jaato" / "references")
        plugin._discover_and_load_bundles()
        plugin._reattach_matchers()
        assert len(factory["built"]) == 1
        assert plugin._embedding_provider is not None


class TestOfflineFirst:
    def test_a_cached_model_loads_offline_first(self, tmp_path, factory):
        _cache_model(tmp_path)
        plugin = _plugin(_workspace(tmp_path, indexed=False), "hybrid")
        assert plugin.embed_texts(["x"])["ok"]
        loads = factory["built"][0].loads
        assert loads == [{"local_files_only": True, "HF_HUB_OFFLINE": "1"}]
        assert "HF_HUB_OFFLINE" not in os.environ

    def test_offline_failure_falls_back_online(self, tmp_path, factory):
        _cache_model(tmp_path)
        factory["kwargs"] = {"offline_ok": False}
        plugin = _plugin(_workspace(tmp_path, indexed=False), "hybrid")
        assert plugin.embed_texts(["x"])["ok"]
        loads = factory["built"][0].loads
        assert [l["local_files_only"] for l in loads] == [True, False]
        assert loads[1]["HF_HUB_OFFLINE"] is None

    def test_an_uncached_model_loads_online(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, indexed=False), "hybrid")
        assert plugin.embed_texts(["x"])["ok"]
        assert factory["built"][0].loads == [
            {"local_files_only": False, "HF_HUB_OFFLINE": None}]
