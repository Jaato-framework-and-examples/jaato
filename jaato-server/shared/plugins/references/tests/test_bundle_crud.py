"""Tests for the new ``bundle`` user command surface (CRUD).

These exercise the membership operations:
    bundle create / delete / add / eject / remove

and confirm that the ``_post_membership_change`` reconciler re-runs after
each disk-mutating operation. The reconcile / merge / pack / unpack
subcommands continue to be covered by ``test_bundle_wiring.py`` (and the
pack/unpack roundtrip suite).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from shared.plugins.references.bundle import (
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    EMBEDDING_CONFIG_FILENAME,
    ROOT_BUNDLE_NAME,
)
from shared.plugins.references.embedding_types import EmbeddingResult
from shared.plugins.references.models import ReferenceSource
from shared.plugins.references.plugin import ReferencesPlugin


def _write_ref(directory: Path, sid: str, *, mode="selectable"):
    directory.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "id": sid,
        "name": sid.replace("-", " ").title(),
        "description": f"Desc for {sid}",
        "type": "local",
        "path": f"./dummy/{sid}.md",
        "mode": mode,
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


class _FakeProvider:
    """Minimal embedding provider for tests that need ``bundle create``."""

    def __init__(self, *, model="mock-model", dim=4):
        self.model_name = model
        self.dimensions = dim
        self.available = True

    def load_model(self):
        self.available = True
        return True

    def embed_text(self, text):
        return EmbeddingResult(embedding=[0.0] * self.dimensions, model=self.model_name, dimensions=self.dimensions)

    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]

    def embed_text_as_array(self, text):
        return [0.0] * self.dimensions


def _make_plugin(workspace: Path, *, with_provider=True) -> ReferencesPlugin:
    """Initialize a ReferencesPlugin against ``workspace`` for CRUD tests.

    ``with_provider=True`` injects a fake embedding provider so commands
    that need one (``create``, the post-op reconcile) can run without
    the entry-point discovery dance.
    """
    plugin = ReferencesPlugin()
    plugin.set_workspace_path(str(workspace))
    plugin.initialize({"lookup_strategy": "tags_only"})
    if with_provider:
        plugin._embedding_provider = _FakeProvider()
    return plugin


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A workspace with one sub-bundle and one free reference.

    Layout::

        ws/.jaato/references/
            free-ref.json        (free, no enclosing bundle)
            teammate/
                embedding_config.json
                team-doc.json

    ``HOME`` is redirected at ``tmp_path / "home"`` so user-tier ops
    don't touch the real ``~/.jaato/references``.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"
    refs.mkdir(parents=True)

    # Free reference: lives in .jaato/references/ but no manifest there.
    _write_ref(refs, "free-ref")

    # Sub-bundle with one reference.
    teammate = refs / "teammate"
    _write_ref(teammate, "team-doc")
    _write_manifest(teammate, rows=["team-doc"])

    return ws


class TestBundleCreate:
    def test_create_workspace_bundle(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "create",
            "target": "newbundle",
        })

        assert result["status"] == "ok"
        bundle_dir = workspace / ".jaato" / "references" / "newbundle"
        assert (bundle_dir / EMBEDDING_CONFIG_FILENAME).is_file()
        manifest = json.loads((bundle_dir / EMBEDDING_CONFIG_FILENAME).read_text())
        assert manifest["embedding_model"] == "mock-model"
        assert manifest["embedding_dimensions"] == 4
        assert manifest["rows"] == []
        assert any(b.name == "newbundle" for b in plugin._bundles)

    def test_create_user_bundle(self, workspace, tmp_path):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "create",
            "target": "personal --scope user",
        })

        assert result["status"] == "ok"
        user_root = tmp_path / "home" / ".jaato" / "references"
        assert (user_root / "personal" / EMBEDDING_CONFIG_FILENAME).is_file()
        new_bundle = next(
            (b for b in plugin._bundles if b.name == "personal"), None,
        )
        assert new_bundle is not None
        assert new_bundle.tier == BUNDLE_TIER_USER

    def test_create_existing_bundle_errors(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "create",
            "target": "teammate",  # already exists in workspace
        })

        assert "error" in result
        assert "already exists" in result["error"]

    def test_create_without_provider_errors(self, workspace):
        plugin = _make_plugin(workspace, with_provider=False)
        plugin._embedding_provider = None

        result = plugin._execute_bundle_cmd({
            "subcommand": "create",
            "target": "newone",
        })

        assert "error" in result
        assert "embedding provider" in result["error"]


class TestBundleDelete:
    def test_delete_empty_bundle(self, workspace):
        plugin = _make_plugin(workspace)
        # Create a fresh empty bundle so we can delete it.
        plugin._execute_bundle_cmd({
            "subcommand": "create", "target": "tmpbundle",
        })

        result = plugin._execute_bundle_cmd({
            "subcommand": "delete", "target": "tmpbundle",
        })

        assert result["status"] == "ok"
        assert not (workspace / ".jaato" / "references" / "tmpbundle").exists()

    def test_delete_non_empty_without_force_errors(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "delete", "target": "teammate",
        })

        assert "error" in result
        assert "not empty" in result["error"]
        # Still on disk.
        assert (workspace / ".jaato" / "references" / "teammate").exists()

    def test_delete_non_empty_with_force(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "delete", "target": "teammate --force",
        })

        assert result["status"] == "ok"
        assert not (workspace / ".jaato" / "references" / "teammate").exists()
        assert not any(b.name == "teammate" for b in plugin._bundles)
        assert all(s.id != "team-doc" for s in plugin._sources)


class TestBundleAdd:
    def test_add_free_reference_into_bundle(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "free-ref --to teammate",
        })

        assert result["status"] == "ok"
        # JSON physically moved.
        teammate_dir = workspace / ".jaato" / "references" / "teammate"
        refs_dir = workspace / ".jaato" / "references"
        assert (teammate_dir / "free-ref.json").is_file()
        assert not (refs_dir / "free-ref.json").exists()
        # In-memory bundle membership reflects the move.
        moved = next((s for s in plugin._sources if s.id == "free-ref"), None)
        assert moved is not None
        assert moved.bundle_name == "teammate"

    def test_add_unknown_ref_id_errors(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "ghost-id --to teammate",
        })

        assert "error" in result
        assert "Unknown reference id" in result["error"]

    def test_add_unknown_target_errors(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "free-ref --to ghost-bundle",
        })

        assert "error" in result
        assert "Unknown target bundle" in result["error"]

    def test_add_already_in_target_errors(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "add",
            "target": "team-doc --to teammate",
        })

        assert "error" in result
        assert "already in" in result["error"]


class TestBundleEject:
    def test_eject_from_subbundle_to_tier_root(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "eject", "target": "team-doc",
        })

        assert result["status"] == "ok"
        teammate_dir = workspace / ".jaato" / "references" / "teammate"
        refs_dir = workspace / ".jaato" / "references"
        # JSON moved out of the bundle, up to the tier root.
        assert (refs_dir / "team-doc.json").is_file()
        assert not (teammate_dir / "team-doc.json").exists()
        # In-memory bundle_name cleared.
        ejected = next((s for s in plugin._sources if s.id == "team-doc"), None)
        assert ejected is not None
        assert ejected.bundle_name == ""

    def test_eject_free_reference_errors(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "eject", "target": "free-ref",
        })

        assert "error" in result
        assert "already free" in result["error"]


class TestBundleRemove:
    def test_remove_free_reference_deletes_file(self, workspace):
        plugin = _make_plugin(workspace)
        refs_dir = workspace / ".jaato" / "references"
        assert (refs_dir / "free-ref.json").is_file()

        result = plugin._execute_bundle_cmd({
            "subcommand": "remove", "target": "free-ref",
        })

        assert result["status"] == "ok"
        assert not (refs_dir / "free-ref.json").exists()
        assert all(s.id != "free-ref" for s in plugin._sources)

    def test_remove_bundled_reference_deletes_file(self, workspace):
        """Removing a bundled reference deletes its JSON; the reconcile
        list is empty under tags_only (sidecar is moot for tag matching)
        but the file is gone and the catalog drops the source."""
        plugin = _make_plugin(workspace)
        teammate_dir = workspace / ".jaato" / "references" / "teammate"
        assert (teammate_dir / "team-doc.json").is_file()

        result = plugin._execute_bundle_cmd({
            "subcommand": "remove", "target": "team-doc",
        })

        assert result["status"] == "ok"
        assert not (teammate_dir / "team-doc.json").exists()
        assert all(s.id != "team-doc" for s in plugin._sources)

    def test_remove_unknown_id_errors(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "remove", "target": "ghost-id",
        })

        assert "error" in result
        assert "Unknown reference id" in result["error"]


class TestBundleHelp:
    def test_bundle_help_lists_all_subcommands(self, workspace):
        plugin = _make_plugin(workspace)

        result = plugin._execute_bundle_cmd({
            "subcommand": "help", "target": "",
        })

        text = "\n".join(line for line, _tag in result.lines)
        for verb in (
            "list", "create", "delete", "add", "eject", "remove",
            "reconcile", "merge", "pack", "unpack",
        ):
            assert verb in text, f"'{verb}' missing from bundle help"


class TestBundleCompletions:
    def test_top_level_completions_lists_all_verbs(self, workspace):
        plugin = _make_plugin(workspace)

        comps = plugin.get_command_completions("bundle", [""])

        values = {c.value for c in comps}
        assert {"list", "create", "delete", "add", "eject", "remove"} <= values
        assert {"reconcile", "merge", "pack", "unpack", "help"} <= values

    def test_add_completes_with_ref_ids(self, workspace):
        plugin = _make_plugin(workspace)

        comps = plugin.get_command_completions("bundle", ["add", ""])

        values = {c.value for c in comps}
        assert "free-ref" in values
        assert "team-doc" in values

    def test_create_scope_value_completions(self, workspace):
        plugin = _make_plugin(workspace)

        comps = plugin.get_command_completions(
            "bundle", ["create", "newbundle", "--scope", ""],
        )

        values = {c.value for c in comps}
        assert {"workspace", "user"} <= values

    def test_references_completions_no_longer_offer_bundle_verbs(self, workspace):
        plugin = _make_plugin(workspace)

        comps = plugin.get_command_completions("references", [""])

        values = {c.value for c in comps}
        # Reference-level ops are still offered.
        assert {"list", "select", "unselect", "reload"} <= values
        # Bundle-level ops have moved out.
        assert "bundles" not in values
        assert "reconcile" not in values
        assert "merge" not in values
        assert "pack" not in values
        assert "unpack" not in values
