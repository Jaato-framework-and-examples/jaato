"""Smoke test for :class:`ProfilesEntryHandler`.

Validates the bundle subsystem can dispatch to a second domain
(profiles) alongside references. The handler exposes profile JSONs
as free entries — see the module docstring on
``shared.plugins.subagent.entry_handler`` for the deliberate
not-yet-implemented surface and the planned follow-up shape.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from shared.plugins.bundle_common.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
)
from shared.plugins.bundle_common.handler import (
    BundleEntry,
    BundleEntryHandler,
    BundleEntryRegistry,
)
from shared.plugins.subagent.entry_handler import ProfilesEntryHandler


class _FakeSubagentPlugin:
    """Minimal stand-in for SubagentPlugin — only the bits the handler reads."""

    def __init__(self, workspace_path: str) -> None:
        self._workspace_path = workspace_path

    def reload_profiles(self) -> None:
        # The real plugin re-walks profile dirs and rebuilds its
        # in-memory cache; the fake doesn't keep state, so this is a
        # no-op. The handler's reload_catalog should still call us.
        self.reloaded = True


def _write_profile(directory: Path, name: str, *, ext: str = ".json") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}{ext}"
    payload: Dict[str, Any] = {
        "name": name,
        "model": "mock-model",
        "system_instructions": f"You are {name}.",
    }
    if ext == ".json":
        path.write_text(json.dumps(payload, indent=2))
    else:
        path.write_text(f"name: {name}\nmodel: mock-model\n")
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Workspace with one workspace-tier profile and one user-tier profile."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws_profiles = ws / ".jaato" / "profiles"
    user_profiles = tmp_path / "home" / ".jaato" / "profiles"
    _write_profile(ws_profiles, "reviewer")
    _write_profile(user_profiles, "personal")
    return ws


@pytest.fixture
def handler(workspace):
    plugin = _FakeSubagentPlugin(str(workspace))
    return ProfilesEntryHandler(plugin)


class TestIdentity:
    def test_kind(self, handler):
        assert handler.kind == "profiles"

    def test_domain_subpath(self, handler):
        assert handler.domain_subpath == Path(".jaato/profiles")

    def test_isinstance_of_protocol(self, handler):
        assert isinstance(handler, BundleEntryHandler)


class TestEnumeration:
    def test_list_entries_covers_both_tiers(self, handler):
        ids = {e.id for e in handler.list_entries()}
        assert {"reviewer", "personal"} <= ids

    def test_list_entries_records_tier(self, handler):
        by_id = {e.id: e for e in handler.list_entries()}
        assert by_id["reviewer"].bundle_tier == BUNDLE_TIER_WORKSPACE
        assert by_id["personal"].bundle_tier == BUNDLE_TIER_USER

    def test_workspace_shadows_user(self, workspace, tmp_path, monkeypatch):
        # Add a user-tier profile with the same id as the workspace one.
        user_profiles = tmp_path / "home" / ".jaato" / "profiles"
        _write_profile(user_profiles, "reviewer")
        plugin = _FakeSubagentPlugin(str(workspace))
        h = ProfilesEntryHandler(plugin)

        # Only one 'reviewer' entry — workspace wins.
        reviewers = [e for e in h.list_entries() if e.id == "reviewer"]
        assert len(reviewers) == 1
        assert reviewers[0].bundle_tier == BUNDLE_TIER_WORKSPACE

    def test_yaml_profiles_are_discovered(self, workspace, handler):
        ws_profiles = workspace / ".jaato" / "profiles"
        _write_profile(ws_profiles, "yaml-one", ext=".yaml")

        ids = {e.id for e in handler.list_entries()}
        assert "yaml-one" in ids

    def test_find_entry_returns_match(self, handler):
        hit = handler.find_entry("reviewer")
        assert hit is not None
        assert hit.id == "reviewer"
        assert hit.kind == "profiles"

    def test_find_entry_unknown_returns_none(self, handler):
        assert handler.find_entry("ghost") is None

    def test_list_bundles_is_empty(self, handler):
        # Profile bundles aren't supported on this build — see docstring.
        assert handler.list_bundles() == []


class TestMutations:
    def test_delete_entry_removes_file(self, workspace, handler):
        entry = handler.find_entry("reviewer")
        assert entry is not None
        handler.delete_entry(entry)
        assert not entry.file_path.exists()
        assert handler.find_entry("reviewer") is None

    def test_move_entry_to_bundle_not_implemented(self, handler):
        entry = handler.find_entry("reviewer")
        with pytest.raises(NotImplementedError, match="bundle add"):
            handler.move_entry_to_bundle(entry, None)

    def test_move_entry_to_free_not_implemented(self, handler):
        entry = handler.find_entry("reviewer")
        with pytest.raises(NotImplementedError, match="bundle eject"):
            handler.move_entry_to_free(entry, BUNDLE_TIER_WORKSPACE)

    def test_create_empty_bundle_not_implemented(self, handler):
        with pytest.raises(NotImplementedError, match="bundle create"):
            handler.create_empty_bundle("foo", BUNDLE_TIER_WORKSPACE)

    def test_delete_bundle_not_implemented(self, handler):
        # Pass a fake Bundle since list_bundles always returns empty.
        from shared.plugins.bundle_common.bundle import Bundle
        bundle = Bundle(
            name="foo",
            directory=Path("/tmp/foo"),
            embedding_model="x",
            embedding_dimensions=1,
            embedding_sidecar="x.npy",
        )
        with pytest.raises(NotImplementedError, match="bundle delete"):
            handler.delete_bundle(bundle)


class TestReloadCatalog:
    def test_reload_calls_plugin_hook(self, workspace):
        plugin = _FakeSubagentPlugin(str(workspace))
        h = ProfilesEntryHandler(plugin)

        h.reload_catalog()

        assert getattr(plugin, "reloaded", False) is True


class TestRegistryAggregation:
    """The profiles handler plugs into the cross-domain registry surface."""

    def test_list_all_entries_includes_profiles(self, handler):
        local = BundleEntryRegistry()
        local.register(handler)
        ids = {e.id for e in local.list_all_entries()}
        assert {"reviewer", "personal"} <= ids

    def test_find_entry_by_kind(self, handler):
        local = BundleEntryRegistry()
        local.register(handler)
        hit = local.find_entry("profiles", "reviewer")
        assert hit is not None
        assert hit.kind == "profiles"


class TestCrossKindIntegrationWithBundlePlugin:
    """The bundle plugin's list/add/eject/remove dispatchers correctly
    surface (or politely refuse) profile-kind operations alongside
    references-kind ones."""

    def test_bundle_list_shows_profiles_section(self, handler):
        from shared.plugins.bundle.plugin import BundlePlugin

        local = BundleEntryRegistry()
        local.register(handler)
        plugin = BundlePlugin(registry=local)

        result = plugin._execute_bundle_cmd({"subcommand": "list", "target": ""})

        text = "\n".join(line for line, _ in result.lines)
        # No profile bundles, so the section header is suppressed (only
        # kinds with bundles surface). list_bundles returns [], so the
        # "no bundles loaded" placeholder appears.
        assert "no bundles" in text.lower()

    def test_bundle_add_profile_errors_with_friendly_message(self, handler):
        from shared.plugins.bundle.plugin import BundlePlugin

        local = BundleEntryRegistry()
        local.register(handler)
        plugin = BundlePlugin(registry=local)

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "profiles:reviewer --to teammate",
        })

        # The bundle plugin tries to find the target bundle in
        # handler.list_bundles() (empty), so the user gets the
        # "Unknown profiles bundle" error before NotImplementedError
        # ever fires. Either error message is acceptable.
        assert "error" in result

    def test_bundle_remove_profile_works(self, handler, workspace):
        from shared.plugins.bundle.plugin import BundlePlugin

        local = BundleEntryRegistry()
        local.register(handler)
        plugin = BundlePlugin(registry=local)

        result = plugin._execute_bundle_cmd({
            "subcommand": "remove",
            "target": "profiles:reviewer",
        })

        assert result["status"] == "ok"
        # File is gone.
        assert not (workspace / ".jaato" / "profiles" / "reviewer.json").exists()
