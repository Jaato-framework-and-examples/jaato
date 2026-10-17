"""The references catalog: one snapshot per process, never edited, loaded on use.

docs/design/references-storage-scaling.md, phase 1. The plugin used to keep
its catalog as a list it loaded and edited itself, so every holder in one
process (the daemon's mirror registry builds one per session) loaded and
held its own copy: about 25 MB per session at 10,000 references, even for a
session whose profile does not list ``references``. Pinned here:

1. Holders of the same catalog in one process share one snapshot object
   (``store.store_for``).
2. A snapshot is never edited: a session that needs a different value (a
   path resolved for it, an inline narrowing) works on a copy.
3. The store holds its snapshot weakly: when the last holder lets go, the
   catalog is freed.
4. A session that does not enable the plugin does not load the catalog at
   ``initialize()``; its first read does (a deferred load is never a
   refused one).
5. A write made through one holder reaches another on its next read.
6. A by-id lookup is a dict lookup, not a scan of the catalog.
7. Configuring a runtime does not load a deferred catalog: the runtime
   used to build every exposed plugin's system instructions at
   ``configure_plugins`` and cache a value only a registry-less runtime
   could read, so the deferral never held.
8. A session whose profile does not list ``references`` neither counts its
   instructions in the budget nor loads its catalog for them: the budget
   reads the plugin set the prompt reads.
9. A change on disk is applied by deriving the next generation from the
   current one: only the files that moved are parsed again, and the result
   equals a full load of the same disk.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import pytest

from jaato_server.shared.plugins.bundle_common.bundle import write_bundle_manifest
from jaato_server.shared.plugins.references import store as store_mod
from jaato_server.shared.plugins.references.plugin import create_plugin
from jaato_server.shared.plugins.registry import SESSION_ENABLES_PLUGIN_KEY
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_STORE = "jaato-server/jaato_server/shared/plugins/references/store.py"
_RUNTIME = "jaato-server/jaato_server/shared/jaato_runtime.py"

REVERSIONS = [
    Reversion(
        target=_STORE,
        find="        store = _STORES.get(key)\n",
        replace="        store = None\n",
        because=(
            "every holder would get a store of its own and load its own "
            "catalog, which is the per-session copy the store exists to remove"
        ),
        test="TestOneSnapshotPerProcess::test_two_sessions_share_one_snapshot",
    ),
    Reversion(
        target=_PLUGIN,
        find="        fresh = copy.copy(source)\n        fresh.resolved_path = resolved\n",
        replace="        fresh = source\n        fresh.resolved_path = resolved\n",
        because=(
            "resolving a path for one session would write it into the "
            "shared snapshot, so another session reading the same catalog "
            "sees a value it never resolved"
        ),
        test="TestASnapshotIsNeverEdited::test_a_session_resolves_on_a_copy",
    ),
    Reversion(
        target=_STORE,
        find="        self._current = weakref.ref(snap)\n",
        replace="        self._current = lambda: snap\n",
        because=(
            "the store would keep every catalog it loaded alive after its "
            "last session let go"
        ),
        test="TestTheLastHolderFreesIt::test_the_snapshot_goes_with_its_holders",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if defer_catalog:\n",
        replace="        if False:\n",
        because=(
            "a session that never uses references would load the whole "
            "catalog at bootstrap, on the daemon's mirror and in the runner"
        ),
        test="TestADeferredCatalog::test_initialize_loads_nothing",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if self._catalog_loaded or self._deferred_config is None:\n            return\n        with self._catalog_lock:",
        replace="        return\n        with self._catalog_lock:",
        because=(
            "a deferred catalog would never load, so a session that does "
            "need it (a typed command, a subagent enabling the plugin) "
            "would find no references at all"
        ),
        test="TestADeferredCatalog::test_the_first_read_loads_it",
    ),
    Reversion(
        target=_PLUGIN,
        find="            snapshot = self._store.current()\n            if snapshot is self._snapshot:\n",
        replace="            snapshot = self._snapshot\n            if snapshot is self._snapshot:\n",
        because=(
            "a session would keep the snapshot it started with, so a "
            "reference another session wrote never reaches it"
        ),
        test="TestAWriteReachesTheOtherHolder::test_the_other_session_sees_it",
    ),
    Reversion(
        target=_RUNTIME,
        find="        # Get auto-approved tools from plugins\n",
        replace=(
            "        self._registry.get_system_instructions()\n"
            "        # Get auto-approved tools from plugins\n"
        ),
        because=(
            "configuring the runtime would ask every exposed plugin for its "
            "instructions, and the references plugin builds them from its "
            "catalog, so a deferred catalog loads at every session start"
        ),
        test="TestConfiguringARuntime::test_it_loads_no_deferred_catalog",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/instruction_budget_builder.py",
        find="        for plugin_name in instruction_plugin_names(runtime.registry, plugin_names):\n",
        replace="        for plugin_name in runtime.registry._exposed:\n",
        because=(
            "the budget would count, and so load, the instructions of every "
            "exposed plugin, including ones the session's prompt never carries"
        ),
        test="TestTheBudgetCountsWhatIsSent::test_an_unlisted_plugin_is_not_counted",
    ),
    Reversion(
        target=_STORE,
        find="        units = _rederive_units(previous, moved)\n",
        replace="        units = None\n",
        because=(
            "every change would re-read the whole catalog, 23 s at 100k "
            "references where deriving costs 0.4 s"
        ),
        test="TestTheNextGenerationIsDerived::test_only_the_moved_file_is_parsed",
    ),
    Reversion(
        target=_STORE,
        find="            resolved = resolve_for_context(source, resolve_root)\n",
        replace="            resolved = None\n",
        because=(
            "a root reference parsed again would keep the path relative to "
            "the reference file, where a full load resolves it"
        ),
        test="TestTheNextGenerationIsDerived::test_it_equals_a_full_load[edit-root-local]",
    ),
]


@pytest.fixture(autouse=True)
def _private_home(tmp_path, monkeypatch):
    """Keep the user tier (``~/.jaato/references``) out of every case."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("REFERENCES_CONFIG_PATH", raising=False)


def _write_ref(ws: Path, ref_id: str, **extra: Any) -> None:
    refs = ws / ".jaato" / "references"
    refs.mkdir(parents=True, exist_ok=True)
    data = {"id": ref_id, "name": ref_id, "description": f"about {ref_id}",
            "type": "inline", "mode": "selectable", "tags": ["ops"],
            "content": f"# {ref_id}", **extra}
    (refs / f"{ref_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _age(ws: Path) -> None:
    """Move every mtime out of the unsettled window, as an hour-old tree."""
    past = 1_000_000_000
    for dirpath, dirnames, filenames in os.walk(ws):
        for name in filenames + dirnames:
            os.utime(os.path.join(dirpath, name), (past, past))
        os.utime(dirpath, (past, past))


def _session(ws: Path, **config: Any):
    plugin = create_plugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "tags_only", **config})
    plugin.set_workspace_path(str(ws))
    return plugin


def _catalog(n: int, tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    for i in range(n):
        _write_ref(ws, f"ref-{i}")
    _age(ws)
    return ws


class TestOneSnapshotPerProcess:
    def test_two_sessions_share_one_snapshot(self, tmp_path):
        ws = _catalog(3, tmp_path)
        first, second = _session(ws), _session(ws)
        assert first._snapshot is second._snapshot
        assert first._source("ref-1") is second._source("ref-1")

    def test_another_workspace_gets_its_own(self, tmp_path):
        ws = _catalog(2, tmp_path)
        other = tmp_path / "other"
        _write_ref(other, "elsewhere")
        assert _session(ws)._snapshot is not _session(other)._snapshot


class TestASnapshotIsNeverEdited:
    def test_a_session_resolves_on_a_copy(self, tmp_path):
        # A sub-bundle reference keeps the relative path the load gave it;
        # resolving it for a session makes it absolute.
        ws = tmp_path / "ws"
        doc = ws / "docs" / "guide.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("# guide", encoding="utf-8")
        team = ws / ".jaato" / "references" / "team"
        team.mkdir(parents=True)
        write_bundle_manifest(team, name="team")
        (team / "guide.json").write_text(json.dumps({
            "id": "guide", "name": "guide", "description": "d", "type": "local",
            "mode": "selectable", "path": str(doc)}), encoding="utf-8")
        _age(ws)
        plugin = _session(ws)
        shared = plugin._snapshot.get("guide")
        before = shared.resolved_path
        assert before and not Path(before).is_absolute()

        mine = plugin._resolved_for_session(shared)

        assert mine is not shared
        assert mine.resolved_path == str(doc.resolve())
        assert shared.resolved_path == before
        assert plugin._snapshot.get("guide").resolved_path == before


class TestTheLastHolderFreesIt:
    def test_the_snapshot_goes_with_its_holders(self, tmp_path):
        ws = _catalog(2, tmp_path)
        plugin = _session(ws)
        store = plugin._store
        plugin.shutdown()
        del plugin
        gc.collect()
        assert store._current is not None and store._current() is None


class TestADeferredCatalog:
    def test_initialize_loads_nothing(self, tmp_path, monkeypatch):
        ws = _catalog(3, tmp_path)
        loads = []
        real = store_mod.ReferenceStore._load
        monkeypatch.setattr(store_mod.ReferenceStore, "_load",
                            lambda self, prev: loads.append(1) or real(self, prev))
        plugin = _session(ws, **{SESSION_ENABLES_PLUGIN_KEY: False})
        assert loads == []
        assert plugin._catalog_loaded is False

    def test_the_first_read_loads_it(self, tmp_path):
        ws = _catalog(3, tmp_path)
        plugin = _session(ws, **{SESSION_ENABLES_PLUGIN_KEY: False})
        result = plugin.get_executors()["listReferences"]({})
        assert {s["id"] for s in result["sources"]} == {"ref-0", "ref-1", "ref-2"}
        assert plugin._catalog_loaded is True

    def test_an_enabled_session_loads_at_initialize(self, tmp_path):
        ws = _catalog(2, tmp_path)
        plugin = _session(ws, **{SESSION_ENABLES_PLUGIN_KEY: True})
        assert plugin._catalog_loaded is True


class TestAWriteReachesTheOtherHolder:
    def test_the_other_session_sees_it(self, tmp_path):
        ws = _catalog(2, tmp_path)
        reader, writer = _session(ws), _session(ws)
        assert reader._snapshot is writer._snapshot
        _write_ref(ws, "new-one")
        # Out of the 2 s window in which a fresh stamp is not trusted
        # (catalog_watch.settle), so one load answers both holders.
        _age(ws)
        writer._reload_from_disk()

        listed = reader.get_executors()["listReferences"]({})
        assert "new-one" in {s["id"] for s in listed["sources"]}
        assert reader._snapshot is writer._snapshot


class TestThePublicListIsAList:
    def test_get_sources_returns_a_list_the_caller_owns(self, tmp_path):
        ws = _catalog(2, tmp_path)
        plugin = _session(ws)
        listed = plugin.get_sources()
        assert isinstance(listed, list)
        listed.clear()
        assert {s.id for s in plugin.get_sources()} == {"ref-0", "ref-1"}


class TestALookupIsADictLookup:
    def test_get_does_not_scan(self, tmp_path):
        ws = _catalog(3, tmp_path)
        plugin = _session(ws)

        class NoIteration(tuple):
            def __iter__(self):
                raise AssertionError("a by-id lookup iterated the catalog")

        plugin._snapshot.sources = NoIteration(plugin._snapshot.sources)
        assert plugin._source("ref-2").id == "ref-2"


class TestConfiguringARuntime:
    def test_it_loads_no_deferred_catalog(self, tmp_path):
        from jaato_server.shared.jaato_runtime import JaatoRuntime
        from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
        from jaato_server.shared.plugins.registry import PluginRegistry

        ws = _catalog(3, tmp_path)
        registry = PluginRegistry()
        registry.set_workspace_path(str(ws))
        registry.set_session_plugins(["cli"])
        plugin = ReferencesPlugin()
        registry.register_plugin(plugin, expose=True, config={
            "workspace_path": str(ws), "lookup_strategy": "tags_only"})
        assert plugin._catalog_loaded is False

        JaatoRuntime(provider_name="echo", workspace_path=ws).configure_plugins(registry)

        assert plugin._catalog_loaded is False


class TestTheBudgetCountsWhatIsSent:
    def test_an_unlisted_plugin_is_not_counted(self, tmp_path):
        from jaato_server.shared.jaato_runtime import JaatoRuntime
        from jaato_server.shared.jaato_session import JaatoSession
        from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
        from jaato_server.shared.plugins.registry import PluginRegistry

        ws = _catalog(3, tmp_path)
        registry = PluginRegistry()
        registry.set_workspace_path(str(ws))
        registry.set_session_plugins(["cli"])
        plugin = ReferencesPlugin()
        registry.register_plugin(plugin, expose=True, config={
            "workspace_path": str(ws), "lookup_strategy": "tags_only"})
        runtime = JaatoRuntime(provider_name="echo", workspace_path=ws)
        runtime.configure_plugins(registry)

        session = JaatoSession(runtime, "test-model", agent_id="main")
        session.configure(skip_provider=True, plugins=["cli"])
        counted = {r.child_key for r in session._collect_instruction_texts(None)}

        assert "references" not in counted
        assert plugin._catalog_loaded is False


def _derive_fixture(tmp_path: Path) -> Path:
    """A root directory and one sub-bundle, aged out of the unsettled window."""
    ws = tmp_path / "ws"
    for i in range(4):
        _write_ref(ws, f"root-{i}")
    doc = ws / "docs" / "guide.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("# guide", encoding="utf-8")
    _write_ref(ws, "root-local", type="local", path="../../docs/guide.md")
    team = ws / ".jaato" / "references" / "team"
    team.mkdir(parents=True)
    write_bundle_manifest(team, name="team")
    for i in range(3):
        (team / f"team-{i}.json").write_text(json.dumps({
            "id": f"team-{i}", "name": f"team-{i}", "description": "d",
            "type": "inline", "mode": "selectable", "content": "x"}), encoding="utf-8")
    _age(ws)
    return ws


def _edit(path: Path, **changes: Any) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def _change(ws: Path, case: str) -> None:
    refs = ws / ".jaato" / "references"
    team = refs / "team"
    if case == "edit-root":
        _edit(refs / "root-1.json", description="changed")
    elif case == "edit-root-local":
        _edit(refs / "root-local.json", description="changed")
    elif case == "edit-bundle":
        _edit(team / "team-1.json", tags=["new"])
    elif case == "add-bundle":
        (team / "team-9.json").write_text(json.dumps({
            "id": "team-9", "name": "n", "description": "d", "type": "inline",
            "mode": "selectable", "content": "x"}), encoding="utf-8")
    elif case == "add-root-duplicate":
        _write_ref(ws, "team-0")
    elif case == "delete-root":
        (refs / "root-2.json").unlink()
    elif case == "rename-id":
        _edit(team / "team-2.json", id="team-renamed")
    elif case == "broken-json":
        (refs / "root-3.json").write_text("{", encoding="utf-8")
    elif case == "new-sub-bundle":
        other = refs / "other"
        other.mkdir()
        write_bundle_manifest(other, name="other")
        (other / "o.json").write_text(json.dumps({
            "id": "o", "name": "o", "description": "d", "type": "inline",
            "mode": "selectable", "content": "x"}), encoding="utf-8")


def _shape(snap) -> Any:
    return ([(s.bundle_name, s.to_dict(), s.resolved_path) for s in snap.sources],
            [s.id for s in snap.config.sources],
            sorted(snap.watch))


class TestTheNextGenerationIsDerived:
    CASES = ["edit-root", "edit-root-local", "edit-bundle", "add-bundle",
             "add-root-duplicate", "delete-root", "rename-id", "broken-json",
             "new-sub-bundle"]

    @pytest.mark.parametrize("case", CASES)
    def test_it_equals_a_full_load(self, tmp_path, case):
        ws = _derive_fixture(tmp_path)
        plugin = _session(ws)
        before = plugin._snapshot
        _change(ws, case)

        derived = plugin._store.current()
        full = store_mod.ReferenceStore(plugin._store.key)._load(None)

        assert derived is not before
        assert _shape(derived) == _shape(full)

    def test_only_the_moved_file_is_parsed(self, tmp_path, monkeypatch):
        ws = _derive_fixture(tmp_path)
        plugin = _session(ws)
        before = plugin._snapshot
        parsed = []
        real = store_mod.parse_reference_file
        monkeypatch.setattr(store_mod, "parse_reference_file",
                            lambda path, root: parsed.append(path.name) or real(path, root))
        loads = []
        real_load = store_mod.ReferenceStore._load
        monkeypatch.setattr(store_mod.ReferenceStore, "_load",
                            lambda self, prev: loads.append(1) or real_load(self, prev))
        _change(ws, "edit-bundle")

        derived = plugin._store.current()

        assert parsed == ["team-1.json"] and loads == []
        unchanged = before.get("team-0")
        assert derived.get("team-0") is unchanged
        assert derived.get("team-1").tags == ["new"]
