"""Tests for the 'references reload' subcommand.

Tests the ability to reload the reference catalog from disk, preserving
selections for sources that still exist and dropping stale ones.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from ..plugin import ReferencesPlugin, create_plugin
from ..models import ReferenceSource, SourceType, InjectionMode


def _make_ref_file(directory: Path, ref_id: str, name: str, **kwargs) -> Path:
    """Create a reference JSON file in the given directory."""
    data = {
        "id": ref_id,
        "name": name,
        "description": kwargs.get("description", f"Description for {ref_id}"),
        "type": kwargs.get("type", "inline"),
        "mode": kwargs.get("mode", "selectable"),
        "tags": kwargs.get("tags", []),
    }
    if data["type"] == "inline":
        data["content"] = kwargs.get("content", f"Content for {ref_id}")
    elif data["type"] == "local":
        data["path"] = kwargs.get("path", f"./{ref_id}.md")

    file_path = directory / f"{ref_id}.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


class TestReloadBasic:
    """Basic reload behaviour tests."""

    def test_reload_returns_reloaded_status(self, tmp_path):
        """Reload returns status='reloaded' and the source count."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        _make_ref_file(refs_dir, "ref-a", "Ref A")

        plugin = create_plugin()
        plugin.initialize({"workspace_path": str(tmp_path)})

        result = plugin._cmd_references_reload()

        assert result["status"] == "reloaded"
        assert result["total_sources"] == 1

    def test_reload_picks_up_new_source(self, tmp_path):
        """A reference file added after initialize is found on reload."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        _make_ref_file(refs_dir, "ref-a", "Ref A")

        plugin = create_plugin()
        plugin.initialize({"workspace_path": str(tmp_path)})
        assert len(plugin._sources) == 1

        # Add a second reference file
        _make_ref_file(refs_dir, "ref-b", "Ref B")

        result = plugin._cmd_references_reload()

        assert result["total_sources"] == 2
        assert "ref-b" in result.get("added", [])
        ids = {s.id for s in plugin._sources}
        assert "ref-a" in ids
        assert "ref-b" in ids

    def test_reload_detects_removed_source(self, tmp_path):
        """A reference file deleted from disk is reported as removed."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        _make_ref_file(refs_dir, "ref-a", "Ref A")
        path_b = _make_ref_file(refs_dir, "ref-b", "Ref B")

        plugin = create_plugin()
        plugin.initialize({"workspace_path": str(tmp_path)})
        assert len(plugin._sources) == 2

        # Remove ref-b
        path_b.unlink()

        result = plugin._cmd_references_reload()

        assert result["total_sources"] == 1
        assert "ref-b" in result.get("removed", [])

    def test_reload_without_workspace_returns_error(self):
        """Reload errors when no workspace path is available."""
        plugin = create_plugin()
        # Don't set workspace_path or project_root
        plugin._workspace_path = None
        plugin._project_root = None

        result = plugin._cmd_references_reload()

        assert "error" in result


class TestReloadPreservesSelections:
    """Tests that reload preserves selected state correctly."""

    def test_surviving_selection_preserved(self, tmp_path):
        """A selected source that still exists stays selected after reload."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        _make_ref_file(refs_dir, "ref-a", "Ref A")
        _make_ref_file(refs_dir, "ref-b", "Ref B")

        plugin = create_plugin()
        plugin.initialize({"workspace_path": str(tmp_path)})
        plugin._cmd_references_select("ref-a")

        result = plugin._cmd_references_reload()

        assert "ref-a" in plugin._selected_source_ids
        assert "dropped_selected" not in result

    def test_stale_selection_dropped(self, tmp_path):
        """A selected source that no longer exists on disk is dropped."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        _make_ref_file(refs_dir, "ref-a", "Ref A")
        path_b = _make_ref_file(refs_dir, "ref-b", "Ref B")

        plugin = create_plugin()
        plugin.initialize({"workspace_path": str(tmp_path)})
        plugin._cmd_references_select("ref-b")
        assert "ref-b" in plugin._selected_source_ids

        # Remove ref-b from disk
        path_b.unlink()

        result = plugin._cmd_references_reload()

        assert "ref-b" not in plugin._selected_source_ids
        assert "ref-b" in result.get("dropped_selected", [])

    def test_selection_order_preserved(self, tmp_path):
        """Selection order from previous state is maintained."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        _make_ref_file(refs_dir, "ref-a", "Ref A")
        _make_ref_file(refs_dir, "ref-b", "Ref B")
        _make_ref_file(refs_dir, "ref-c", "Ref C")

        plugin = create_plugin()
        plugin.initialize({"workspace_path": str(tmp_path)})
        plugin._cmd_references_select("ref-c")
        plugin._cmd_references_select("ref-a")

        plugin._cmd_references_reload()

        # Order should match the original selection order
        assert plugin._selected_source_ids[:2] == ["ref-c", "ref-a"]


class TestReloadClearsAuthorizations:
    """Tests that reload properly manages sandbox authorizations."""

    def test_reload_clears_and_reauthorizes(self, tmp_path):
        """Reload calls clear_authorized_paths, then re-authorizes surviving selections."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        _make_ref_file(refs_dir, "ref-a", "Ref A")

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.get_plugin.return_value = None
        plugin.set_plugin_registry(mock_registry)

        plugin.initialize({"workspace_path": str(tmp_path)})
        plugin._cmd_references_select("ref-a")

        plugin._cmd_references_reload()

        mock_registry.clear_authorized_paths.assert_called_with("references")


class TestReloadCommandIntegration:
    """Tests for the reload subcommand via the command dispatch."""

    def test_dispatch_routes_to_reload(self, tmp_path):
        """'references reload' dispatches to _cmd_references_reload."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)
        _make_ref_file(refs_dir, "ref-a", "Ref A")

        plugin = create_plugin()
        plugin.initialize({"workspace_path": str(tmp_path)})

        result = plugin._execute_references_cmd({"subcommand": "reload"})

        assert result["status"] == "reloaded"

    def test_reload_in_completions(self):
        """'reload' appears in command completions."""
        plugin = create_plugin()
        completions = plugin.get_command_completions("references", [])

        values = [c.value for c in completions]
        assert "reload" in values

    def test_reload_completion_with_partial(self):
        """Partial 'rel' completes to 'reload'."""
        plugin = create_plugin()
        completions = plugin.get_command_completions("references", ["rel"])

        values = [c.value for c in completions]
        assert "reload" in values

    def test_help_mentions_reload(self):
        """Help text includes the reload subcommand."""
        plugin = create_plugin()
        help_result = plugin._cmd_references_help()

        texts = [line[0] for line in help_result.lines]
        assert any("reload" in t for t in texts)


class TestReloadTransitive:
    """Tests that transitive resolution is re-applied after reload."""

    def test_transitive_reapplied_after_reload(self, tmp_path):
        """Transitive references are re-resolved when selections survive."""
        refs_dir = tmp_path / ".jaato" / "references"
        refs_dir.mkdir(parents=True)

        # ref-a mentions ref-b in its content
        _make_ref_file(refs_dir, "ref-a", "Ref A",
                       content="See ref-b for details.")
        _make_ref_file(refs_dir, "ref-b", "Ref B",
                       content="Detailed info here.")

        plugin = create_plugin()
        plugin.initialize({
            "workspace_path": str(tmp_path),
            "transitive_injection": True,
        })

        # Select ref-a — should transitively pull in ref-b
        plugin._cmd_references_select("ref-a")
        assert "ref-b" in plugin._selected_source_ids

        # Reload — transitive should be re-applied
        plugin._cmd_references_reload()

        assert "ref-a" in plugin._selected_source_ids
        assert "ref-b" in plugin._selected_source_ids
