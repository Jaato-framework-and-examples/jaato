"""Smoke test for :class:`ReferencesEntryHandler` against a real plugin.

These tests prove the protocol implementation works end-to-end against
a live :class:`ReferencesPlugin`. The bundle CRUD code itself doesn't
yet dispatch through the registry (Phase 8) — but the handler is
exercised here so any wiring breakage is caught immediately, before
Phase 8 starts depending on it.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from shared.plugins.bundle_common.bundle import (
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
)
from shared.plugins.bundle_common.handler import (
    BundleEntry,
    BundleEntryHandler,
    BundleEntryRegistry,
    registry as global_registry,
)
from shared.plugins.references.entry_handler import ReferencesEntryHandler
from shared.plugins.references.plugin import ReferencesPlugin


def _write_ref(directory: Path, sid: str):
    directory.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "id": sid,
        "name": sid.replace("-", " ").title(),
        "description": f"Desc for {sid}",
        "type": "local",
        "path": f"./dummy/{sid}.md",
        "mode": "selectable",
    }
    (directory / f"{sid}.json").write_text(json.dumps(payload, indent=2))


def _write_manifest(directory: Path, *, rows, model="mock-model", dim=4):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": model,
        "embedding_dimensions": dim,
        "embedding_sidecar": "references.embeddings.npy",
        "rows": list(rows),
    }))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Workspace with a free reference and one sub-bundle holding one ref."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"
    refs.mkdir(parents=True)
    _write_ref(refs, "free-ref")

    teammate = refs / "teammate"
    _write_ref(teammate, "team-doc")
    _write_manifest(teammate, rows=["team-doc"])
    return ws


@pytest.fixture
def plugin(workspace):
    """An initialized references plugin against ``workspace``."""
    p = ReferencesPlugin()
    p.set_workspace_path(str(workspace))
    p.initialize({"lookup_strategy": "tags_only"})
    yield p
    p.shutdown()


class TestHandlerSatisfiesProtocol:
    def test_isinstance_check(self, plugin):
        handler = ReferencesEntryHandler(plugin)
        assert isinstance(handler, BundleEntryHandler)

    def test_kind_is_references(self, plugin):
        assert ReferencesEntryHandler(plugin).kind == "references"

    def test_domain_subpath_is_dot_jaato_references(self, plugin):
        assert ReferencesEntryHandler(plugin).domain_subpath == Path(".jaato/references")


class TestHandlerEnumeration:
    def test_list_entries_covers_free_and_bundled(self, plugin):
        handler = ReferencesEntryHandler(plugin)

        ids = {e.id for e in handler.list_entries()}

        assert {"free-ref", "team-doc"} <= ids

    def test_list_entries_records_bundle_membership(self, plugin):
        handler = ReferencesEntryHandler(plugin)

        by_id = {e.id: e for e in handler.list_entries()}

        assert by_id["free-ref"].bundle_name == ""
        assert by_id["team-doc"].bundle_name == "teammate"

    def test_list_entries_records_tier(self, plugin):
        handler = ReferencesEntryHandler(plugin)

        for entry in handler.list_entries():
            assert entry.bundle_tier == BUNDLE_TIER_WORKSPACE

    def test_list_entries_skips_orphaned_sources(self, plugin, workspace):
        """If a source is in the catalog but its file vanished from
        disk, list_entries omits it (the bundle subsystem only
        operates on entries it can act on)."""
        # Manually rip the file out without notifying the catalog.
        (workspace / ".jaato" / "references" / "free-ref.json").unlink()

        handler = ReferencesEntryHandler(plugin)
        ids = {e.id for e in handler.list_entries()}

        assert "free-ref" not in ids

    def test_find_entry_returns_match(self, plugin):
        handler = ReferencesEntryHandler(plugin)
        found = handler.find_entry("team-doc")
        assert found is not None
        assert found.id == "team-doc"
        assert found.bundle_name == "teammate"

    def test_find_entry_returns_none_for_unknown(self, plugin):
        handler = ReferencesEntryHandler(plugin)
        assert handler.find_entry("ghost") is None


class TestHandlerRegistrationLifecycle:
    def test_initialize_registers_handler_in_global_registry(self, plugin):
        handler = global_registry.get("references")
        assert isinstance(handler, ReferencesEntryHandler)

    def test_shutdown_unregisters_handler(self, workspace):
        p = ReferencesPlugin()
        p.set_workspace_path(str(workspace))
        p.initialize({"lookup_strategy": "tags_only"})
        assert global_registry.get("references") is not None

        p.shutdown()

        assert global_registry.get("references") is None

    def test_reinitialize_replaces_handler(self, workspace):
        p1 = ReferencesPlugin()
        p1.set_workspace_path(str(workspace))
        p1.initialize({"lookup_strategy": "tags_only"})
        first = global_registry.get("references")

        p2 = ReferencesPlugin()
        p2.set_workspace_path(str(workspace))
        p2.initialize({"lookup_strategy": "tags_only"})
        second = global_registry.get("references")

        assert first is not second
        assert second._plugin is p2  # pyright: ignore[reportAttributeAccessIssue]

        # Cleanup so subsequent tests don't see leftover state.
        p1.shutdown()
        p2.shutdown()


class TestRegistryAggregation:
    """The references handler plugs into the cross-domain registry surface."""

    def test_list_all_entries_includes_references(self, plugin):
        # Use an isolated registry so we don't fight global state.
        local = BundleEntryRegistry()
        local.register(ReferencesEntryHandler(plugin))

        ids = {e.id for e in local.list_all_entries()}

        assert {"free-ref", "team-doc"} <= ids

    def test_find_entry_by_kind(self, plugin):
        local = BundleEntryRegistry()
        local.register(ReferencesEntryHandler(plugin))

        hit = local.find_entry("references", "team-doc")

        assert hit is not None
        assert hit.kind == "references"
        assert hit.id == "team-doc"
