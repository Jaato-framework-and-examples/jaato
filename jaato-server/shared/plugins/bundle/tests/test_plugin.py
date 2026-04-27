"""Tests for the top-level :class:`BundlePlugin`.

Covers cross-kind dispatch via the :class:`BundleEntryRegistry`. The
plugin is instantiated with an isolated registry so we can wire up a
stub handler without touching the global one.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shared.plugins.bundle.plugin import BundlePlugin
from shared.plugins.bundle_common.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
    Bundle,
)
from shared.plugins.bundle_common.handler import (
    BundleEntry,
    BundleEntryHandler,
    BundleEntryRegistry,
)


class _FakeHandler(BundleEntryHandler):
    """Minimal stub handler that records dispatched calls.

    Owns one bundle and one entry by default; tests mutate ``self``
    directly to set up specific scenarios.
    """

    def __init__(
        self,
        kind: str = "stub",
        bundles: List[Bundle] = None,
        entries: List[BundleEntry] = None,
    ) -> None:
        self._kind = kind
        self._bundles = list(bundles) if bundles else []
        self._entries = list(entries) if entries else []
        self.move_to_bundle_calls: List[Dict[str, Any]] = []
        self.move_to_free_calls: List[Dict[str, Any]] = []
        self.delete_calls: List[str] = []
        self.reload_calls = 0
        self.reconcile_calls: List[str] = []

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def domain_subpath(self) -> Path:
        return Path(f".jaato/{self._kind}")

    def list_entries(self) -> List[BundleEntry]:
        return list(self._entries)

    def list_bundles(self) -> List[Bundle]:
        return list(self._bundles)

    def find_entry(self, entry_id):
        return next((e for e in self._entries if e.id == entry_id), None)

    def move_entry_to_bundle(self, entry, target_bundle):
        self.move_to_bundle_calls.append({
            "entry": entry.id,
            "target": target_bundle.qualified_ref,
        })
        # Update the in-memory state to mimic a real move.
        new_entry = BundleEntry(
            id=entry.id,
            kind=entry.kind,
            file_path=target_bundle.directory / entry.file_path.name,
            bundle_name=target_bundle.name,
            bundle_tier=target_bundle.tier,
        )
        self._entries = [new_entry if e.id == entry.id else e for e in self._entries]
        return new_entry.file_path

    def move_entry_to_free(self, entry, target_tier):
        self.move_to_free_calls.append({
            "entry": entry.id,
            "tier": target_tier,
        })
        new_entry = BundleEntry(
            id=entry.id,
            kind=entry.kind,
            file_path=Path(f"/tmp/{self._kind}/{entry.file_path.name}"),
            bundle_name="",
            bundle_tier=target_tier,
        )
        self._entries = [new_entry if e.id == entry.id else e for e in self._entries]
        return new_entry.file_path

    def delete_entry(self, entry):
        self.delete_calls.append(entry.id)
        self._entries = [e for e in self._entries if e.id != entry.id]

    def reload_catalog(self):
        self.reload_calls += 1

    def reconcile_bundle(self, bundle):
        self.reconcile_calls.append(bundle.qualified_ref)
        return None


def _bundle(name: str, tier: str = BUNDLE_TIER_WORKSPACE) -> Bundle:
    return Bundle(
        name=name,
        directory=Path(f"/tmp/{tier}/{name or 'root'}"),
        embedding_model="m",
        embedding_dimensions=4,
        embedding_sidecar="x.npy",
        tier=tier,
    )


def _entry(eid: str, kind: str, *, bundle_name: str = "", tier: str = BUNDLE_TIER_WORKSPACE) -> BundleEntry:
    return BundleEntry(
        id=eid,
        kind=kind,
        file_path=Path(f"/tmp/{kind}/{eid}.json"),
        bundle_name=bundle_name,
        bundle_tier=tier,
    )


@pytest.fixture
def plugin_with_stub():
    """A BundlePlugin wired to a fresh registry containing one stub handler."""
    registry = BundleEntryRegistry()
    handler = _FakeHandler(
        kind="references",
        bundles=[
            _bundle("teammate"),
            _bundle("personal", BUNDLE_TIER_USER),
        ],
        entries=[
            _entry("free-ref", "references"),
            _entry("team-doc", "references", bundle_name="teammate"),
        ],
    )
    registry.register(handler)
    plugin = BundlePlugin(registry=registry)
    return plugin, handler


class TestPluginShape:
    def test_name_is_bundle(self, plugin_with_stub):
        plugin, _ = plugin_with_stub
        assert plugin.name == "bundle"

    def test_no_model_tools(self, plugin_with_stub):
        plugin, _ = plugin_with_stub
        assert plugin.get_tool_schemas() == []

    def test_command_is_not_shared_with_model(self, plugin_with_stub):
        plugin, _ = plugin_with_stub
        cmds = plugin.get_user_commands()
        assert len(cmds) == 1
        assert cmds[0].name == "bundle"
        assert cmds[0].share_with_model is False


class TestList:
    def test_list_groups_by_kind(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({"subcommand": "list", "target": ""})

        text = "\n".join(line for line, _ in result.lines)
        assert "references:" in text
        assert "teammate" in text
        assert "personal" in text

    def test_list_with_no_handlers(self):
        plugin = BundlePlugin(registry=BundleEntryRegistry())

        result = plugin._execute_bundle_cmd({"subcommand": "list", "target": ""})

        text = "\n".join(line for line, _ in result.lines)
        assert "no domain handlers" in text.lower()

    def test_default_subcommand_is_list(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({})

        text = "\n".join(line for line, _ in result.lines)
        assert "BUNDLES" in text


class TestAdd:
    def test_add_dispatches_to_handler(self, plugin_with_stub):
        plugin, handler = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "references:free-ref --to teammate",
        })

        assert result["status"] == "ok"
        assert handler.move_to_bundle_calls == [
            {"entry": "free-ref", "target": "workspace:teammate"},
        ]
        assert handler.reload_calls == 1
        # Source was free, only target reconciles.
        assert handler.reconcile_calls == ["workspace:teammate"]

    def test_add_unknown_kind(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "agents:foo --to teammate",
        })

        assert "error" in result
        assert "unknown kind" in result["error"]

    def test_add_unknown_entry(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "references:ghost --to teammate",
        })

        assert "error" in result
        assert "Unknown references entry" in result["error"]

    def test_add_unknown_target_bundle(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "references:free-ref --to ghost-bundle",
        })

        assert "error" in result
        assert "Unknown references bundle" in result["error"]

    def test_add_kind_qualifier_required(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "free-ref --to teammate",
        })

        assert "error" in result
        assert "kind prefix" in result["error"]

    def test_add_already_in_target(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "references:team-doc --to teammate",
        })

        assert "error" in result
        assert "already in" in result["error"]

    def test_add_reconciles_both_source_and_target(self, plugin_with_stub):
        plugin, handler = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "references:team-doc --to user:personal",
        })

        assert result["status"] == "ok"
        # Both sides reconcile (source was 'teammate', target is 'user:personal').
        assert "workspace:teammate" in handler.reconcile_calls
        assert "user:personal" in handler.reconcile_calls


class TestEject:
    def test_eject_dispatches_to_handler(self, plugin_with_stub):
        plugin, handler = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "eject",
            "target": "references:team-doc",
        })

        assert result["status"] == "ok"
        assert handler.move_to_free_calls == [
            {"entry": "team-doc", "tier": BUNDLE_TIER_WORKSPACE},
        ]
        # Source bundle reconciles; no target.
        assert handler.reconcile_calls == ["workspace:teammate"]

    def test_eject_already_free(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "eject",
            "target": "references:free-ref",
        })

        assert "error" in result
        assert "already free" in result["error"]

    def test_eject_takes_one_token(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "eject",
            "target": "references:team-doc extra",
        })

        assert "error" in result
        assert "Usage" in result["error"]


class TestRemove:
    def test_remove_dispatches_to_handler(self, plugin_with_stub):
        plugin, handler = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "remove",
            "target": "references:team-doc",
        })

        assert result["status"] == "ok"
        assert handler.delete_calls == ["team-doc"]
        # Source bundle reconciles after deletion.
        assert handler.reconcile_calls == ["workspace:teammate"]

    def test_remove_free_entry(self, plugin_with_stub):
        plugin, handler = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "remove",
            "target": "references:free-ref",
        })

        assert result["status"] == "ok"
        assert handler.delete_calls == ["free-ref"]
        # No source bundle for a free entry.
        assert handler.reconcile_calls == []


class TestHelp:
    def test_help_lists_all_subcommands(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({"subcommand": "help", "target": ""})

        text = "\n".join(line for line, _ in result.lines)
        for verb in ("list", "add", "eject", "remove", "help"):
            assert verb in text


class TestUnknownSubcommand:
    def test_unknown_subcommand_errors_with_hint(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        result = plugin._execute_bundle_cmd({
            "subcommand": "nonsense", "target": "",
        })

        assert "error" in result
        assert "list" in result["error"]


class TestCompletions:
    def test_top_level_completions(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        comps = plugin.get_command_completions("bundle", [""])

        values = {c.value for c in comps}
        assert {"list", "add", "eject", "remove", "help"} <= values

    def test_add_completions_offer_kind_prefixed_entries(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        comps = plugin.get_command_completions("bundle", ["add", ""])

        values = {c.value for c in comps}
        assert "references:free-ref" in values
        assert "references:team-doc" in values

    def test_eject_completions_offer_kind_prefixed_entries(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        comps = plugin.get_command_completions("bundle", ["eject", ""])

        values = {c.value for c in comps}
        assert "references:team-doc" in values

    def test_add_to_completions_offer_bundle_names(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        comps = plugin.get_command_completions(
            "bundle", ["add", "references:free-ref", "--to", ""],
        )

        values = {c.value for c in comps}
        assert "teammate" in values
        assert "personal" in values
        # Scope-qualified forms also surfaced.
        assert "workspace:teammate" in values
        assert "user:personal" in values

    def test_other_command_returns_empty(self, plugin_with_stub):
        plugin, _ = plugin_with_stub
        assert plugin.get_command_completions("references", [""]) == []
