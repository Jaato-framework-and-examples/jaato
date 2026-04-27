"""Tests for the top-level :class:`BundlePlugin`.

Covers cross-kind dispatch via the :class:`BundleEntryRegistry`. The
plugin is instantiated with an isolated registry so we can wire up a
stub handler without touching the global one.

The pack/unpack tests use a real :class:`ReferencesEntryHandler` over
a temp workspace because the v2 archive format requires concrete
manifests, sidecars, and an embedding-config validator path that the
stub handler doesn't simulate. See ``test_plugin.py`` in
``bundle_common/tests`` for stub-only pack/unpack coverage.
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
from shared.plugins.references.entry_handler import ReferencesEntryHandler
from shared.plugins.references.plugin import ReferencesPlugin


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

    def test_pack_completes_with_bundle_names(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        comps = plugin.get_command_completions("bundle", ["pack", ""])

        values = {c.value for c in comps}
        assert "teammate" in values
        assert "personal" in values

    def test_pack_scope_value_completions(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        comps = plugin.get_command_completions(
            "bundle", ["pack", "teammate", "--scope", ""],
        )

        values = {c.value for c in comps}
        assert {"workspace", "user"} <= values

    def test_unpack_flag_completions(self, plugin_with_stub):
        plugin, _ = plugin_with_stub

        comps = plugin.get_command_completions(
            "bundle", ["unpack", "archive.tar.gz", "--"],
        )

        values = {c.value for c in comps}
        assert {"--scope", "--into", "--overwrite", "--merge", "--no-reconcile"} <= values


# ===========================================================================
# Pack / unpack — exercised against a real ReferencesEntryHandler so the
# v2 archive format is validated end-to-end (the stub handler doesn't
# simulate manifests/sidecars deeply enough for the packer's plumbing).
# ===========================================================================


def _write_ref_json(directory: Path, sid: str) -> None:
    """Drop a LOCAL ref JSON whose payload is a relative file in the bundle dir."""
    directory.mkdir(parents=True, exist_ok=True)
    payload_dir = directory / "payloads"
    payload_dir.mkdir(exist_ok=True)
    payload_file = payload_dir / f"{sid}.md"
    payload_file.write_text(f"# {sid}\nbody")
    (directory / f"{sid}.json").write_text(json.dumps({
        "id": sid,
        "name": sid.replace("-", " ").title(),
        "description": f"Desc for {sid}",
        "type": "local",
        "path": str(payload_file.resolve()),
        "mode": "selectable",
    }))


def _write_manifest(directory: Path, *, rows, model="mock-model", dim=4) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / EMBEDDING_CONFIG_FILENAME).write_text(json.dumps({
        "embedding_model": model,
        "embedding_dimensions": dim,
        "embedding_sidecar": "vectors.npy",
        "rows": list(rows),
    }))
    # Empty sidecar stand-in — pack copies bytes, doesn't parse.
    (directory / "vectors.npy").write_bytes(b"\x93NUMPY-FAKE")


@pytest.fixture
def workspace_with_ref(tmp_path, monkeypatch):
    """Workspace with one references sub-bundle ('teammate') holding one ref."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"
    teammate = refs / "teammate"
    _write_ref_json(teammate, "team-doc")
    _write_manifest(teammate, rows=[])  # rows empty — reconcile would populate
    return ws


@pytest.fixture
def plugin_with_real_refs(workspace_with_ref):
    """Bundle plugin wired to a real ReferencesPlugin in an isolated registry."""
    registry = BundleEntryRegistry()
    refs_plugin = ReferencesPlugin()
    refs_plugin.set_workspace_path(str(workspace_with_ref))
    refs_plugin.initialize({"lookup_strategy": "tags_only"})

    # initialize() registers the references handler with the *global*
    # registry; replicate the same in our isolated one so the bundle
    # plugin sees it. Unregister from global so we don't leak.
    from shared.plugins.bundle_common.handler import (
        registry as global_registry,
    )
    global_registry.unregister("references")
    handler = ReferencesEntryHandler(refs_plugin)
    registry.register(handler)

    bundle_plugin = BundlePlugin(registry=registry)
    bundle_plugin.set_workspace_path(str(workspace_with_ref))

    yield bundle_plugin, refs_plugin, workspace_with_ref

    refs_plugin.shutdown()


class TestPack:
    def test_pack_writes_archive_at_default_path(self, plugin_with_real_refs):
        plugin, _refs, ws = plugin_with_real_refs

        result = plugin._execute_bundle_cmd({
            "subcommand": "pack",
            "target": "teammate",
        })

        assert result["status"] == "ok"
        # Default archive lives under the workspace root.
        archive = ws / "teammate-workspace.tar.gz"
        assert archive.is_file()
        assert result["archive_path"] == str(archive.resolve())

    def test_pack_with_explicit_to(self, plugin_with_real_refs, tmp_path):
        plugin, _refs, _ws = plugin_with_real_refs
        out = tmp_path / "out.tar.gz"

        result = plugin._execute_bundle_cmd({
            "subcommand": "pack",
            "target": f"teammate --to {out}",
        })

        assert result["status"] == "ok"
        assert out.is_file()

    def test_pack_unknown_bundle_errors(self, plugin_with_real_refs):
        plugin, _refs, _ws = plugin_with_real_refs

        result = plugin._execute_bundle_cmd({
            "subcommand": "pack",
            "target": "ghost",
        })

        assert "error" in result
        assert "no bundle" in result["error"].lower()

    def test_pack_records_per_kind_breakdown(self, plugin_with_real_refs):
        plugin, _refs, _ws = plugin_with_real_refs

        result = plugin._execute_bundle_cmd({
            "subcommand": "pack",
            "target": "teammate",
        })

        kinds = result["kinds"]
        assert len(kinds) == 1
        assert kinds[0]["kind"] == "references"
        assert kinds[0]["entry_count"] == 1


class TestUnpack:
    def test_pack_then_unpack_roundtrip(self, plugin_with_real_refs, tmp_path):
        plugin, refs_plugin, ws = plugin_with_real_refs

        # Pack into a temp archive…
        archive = tmp_path / "out.tar.gz"
        pack_result = plugin._execute_bundle_cmd({
            "subcommand": "pack",
            "target": f"teammate --to {archive}",
        })
        assert pack_result["status"] == "ok"

        # …and unpack into a fresh recipient workspace.
        recipient = tmp_path / "recipient"
        recipient.mkdir()
        recipient_refs = recipient / ".jaato" / "references"
        recipient_refs.mkdir(parents=True)

        # Re-target the plugin at the recipient workspace.
        new_registry = BundleEntryRegistry()
        new_refs = ReferencesPlugin()
        new_refs.set_workspace_path(str(recipient))
        new_refs.initialize({"lookup_strategy": "tags_only"})
        from shared.plugins.bundle_common.handler import (
            registry as global_registry,
        )
        global_registry.unregister("references")
        new_registry.register(ReferencesEntryHandler(new_refs))
        new_plugin = BundlePlugin(registry=new_registry)
        new_plugin.set_workspace_path(str(recipient))

        unpack_result = new_plugin._execute_bundle_cmd({
            "subcommand": "unpack",
            "target": f"{archive}",
        })
        try:
            assert unpack_result["status"] == "ok"
            installed = recipient_refs / "teammate"
            assert (installed / EMBEDDING_CONFIG_FILENAME).is_file()
            assert (installed / "team-doc.json").is_file()
            assert unpack_result["target"] == "workspace:teammate"
        finally:
            new_refs.shutdown()

    def test_unpack_missing_archive_errors(self, plugin_with_real_refs, tmp_path):
        plugin, _refs, _ws = plugin_with_real_refs

        result = plugin._execute_bundle_cmd({
            "subcommand": "unpack",
            "target": str(tmp_path / "nope.tar.gz"),
        })

        assert "error" in result
        # FileNotFoundError surfaces verbatim from bundle_common.
        assert "not found" in result["error"].lower() or "no such" in result["error"].lower()

    def test_unpack_collision_without_flag(self, plugin_with_real_refs, tmp_path):
        plugin, _refs, _ws = plugin_with_real_refs
        archive = tmp_path / "out.tar.gz"
        plugin._execute_bundle_cmd({
            "subcommand": "pack",
            "target": f"teammate --to {archive}",
        })

        # First unpack fine (target_name defaults to "teammate" — but
        # the source workspace already has a teammate bundle).
        result = plugin._execute_bundle_cmd({
            "subcommand": "unpack",
            "target": f"{archive}",
        })

        assert "error" in result
        # "already" appears in the bundle_common error message
        # ("target X already contains a bundle").
        assert "already" in result["error"].lower()

    def test_unpack_no_reconcile_flag(self, plugin_with_real_refs, tmp_path):
        plugin, _refs, _ws = plugin_with_real_refs
        archive = tmp_path / "out.tar.gz"
        plugin._execute_bundle_cmd({
            "subcommand": "pack",
            "target": f"teammate --to {archive}",
        })

        recipient = tmp_path / "recipient"
        recipient_refs = recipient / ".jaato" / "references"
        recipient_refs.mkdir(parents=True)
        new_refs = ReferencesPlugin()
        new_refs.set_workspace_path(str(recipient))
        new_refs.initialize({"lookup_strategy": "tags_only"})
        from shared.plugins.bundle_common.handler import (
            registry as global_registry,
        )
        global_registry.unregister("references")
        new_registry = BundleEntryRegistry()
        new_registry.register(ReferencesEntryHandler(new_refs))
        new_plugin = BundlePlugin(registry=new_registry)
        new_plugin.set_workspace_path(str(recipient))

        try:
            result = new_plugin._execute_bundle_cmd({
                "subcommand": "unpack",
                "target": f"{archive} --no-reconcile",
            })

            assert result["status"] == "ok"
            assert result["reconciled"] == []
        finally:
            new_refs.shutdown()
