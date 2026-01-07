"""Tests for the artifact tracker plugin."""

import json
import os
import pytest
import tempfile
from unittest.mock import patch

from ..plugin import (
    ArtifactTrackerPlugin,
    _normalize_path,
    _detect_workspace_root,
    create_plugin,
)


class TestNormalizePath:
    """Tests for path normalization."""

    def test_normalize_relative_path(self):
        """Test that relative paths are converted to absolute."""
        result = _normalize_path("doc.md")
        assert os.path.isabs(result)
        assert result.endswith("doc.md")

    def test_normalize_dot_prefix(self):
        """Test that ./ prefix is handled."""
        result = _normalize_path("./doc.md")
        assert os.path.isabs(result)
        assert result.endswith("doc.md")

    def test_normalize_double_dot(self):
        """Test that .. references are collapsed."""
        result = _normalize_path("./foo/../doc.md")
        assert os.path.isabs(result)
        assert ".." not in result
        assert result.endswith("doc.md")

    def test_normalize_empty_string(self):
        """Test that empty string returns empty string."""
        result = _normalize_path("")
        assert result == ""


class TestDetectWorkspaceRoot:
    """Tests for workspace root detection."""

    def test_detect_jaato_workspace_root(self, monkeypatch):
        """Test detection from JAATO_WORKSPACE_ROOT env var."""
        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", "/test/workspace")
        monkeypatch.delenv("workspaceRoot", raising=False)
        result = _detect_workspace_root()
        # The path should be resolved/normalized
        assert result is not None
        assert "workspace" in result.lower()

    def test_detect_workspace_root_env(self, monkeypatch):
        """Test detection from workspaceRoot env var."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.setenv("workspaceRoot", "/test/project")
        result = _detect_workspace_root()
        assert result is not None
        assert "project" in result.lower()

    def test_detect_priority_jaato_first(self, monkeypatch):
        """Test that JAATO_WORKSPACE_ROOT takes priority."""
        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", "/priority/workspace")
        monkeypatch.setenv("workspaceRoot", "/fallback/workspace")
        result = _detect_workspace_root()
        assert "priority" in result.lower()

    def test_detect_none_when_unset(self, monkeypatch):
        """Test that None is returned when no env vars are set."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)
        result = _detect_workspace_root()
        assert result is None


class TestArtifactTrackerPluginDisplayPaths:
    """Tests for relative path display in artifact tracker plugin."""

    @pytest.fixture
    def plugin_with_workspace(self, monkeypatch, tmp_path):
        """Create a plugin with a workspace root configured."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "auto_load": False,
        })
        return plugin, workspace

    @pytest.fixture
    def plugin_no_workspace(self, monkeypatch):
        """Create a plugin without workspace root."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = create_plugin()
        plugin.initialize({"auto_load": False})
        return plugin

    def test_to_display_path_within_workspace(self, plugin_with_workspace):
        """Test that paths within workspace are converted to relative."""
        plugin, workspace = plugin_with_workspace

        abs_path = os.path.join(workspace, "src", "main.py")
        result = plugin._to_display_path(abs_path)

        assert result == os.path.join("src", "main.py")
        assert not os.path.isabs(result)

    def test_to_display_path_workspace_root(self, plugin_with_workspace):
        """Test that workspace root itself returns '.'."""
        plugin, workspace = plugin_with_workspace

        result = plugin._to_display_path(workspace)
        assert result == "."

    def test_to_display_path_outside_workspace(self, plugin_with_workspace, tmp_path):
        """Test that paths outside workspace remain absolute."""
        plugin, workspace = plugin_with_workspace

        outside_path = str(tmp_path / "other" / "file.py")
        result = plugin._to_display_path(outside_path)

        # Should return the original path (not converted)
        assert result == outside_path

    def test_to_display_path_no_workspace(self, plugin_no_workspace):
        """Test that paths are unchanged when no workspace is set."""
        plugin = plugin_no_workspace

        abs_path = "/some/absolute/path/file.py"
        result = plugin._to_display_path(abs_path)

        assert result == abs_path

    def test_to_display_paths_list(self, plugin_with_workspace):
        """Test converting a list of paths to display paths."""
        plugin, workspace = plugin_with_workspace

        paths = [
            os.path.join(workspace, "src", "a.py"),
            os.path.join(workspace, "tests", "test_a.py"),
        ]
        result = plugin._to_display_paths(paths)

        assert result == [
            os.path.join("src", "a.py"),
            os.path.join("tests", "test_a.py"),
        ]


class TestTrackArtifactDisplayPaths:
    """Tests for trackArtifact tool returning display paths."""

    @pytest.fixture
    def plugin_with_workspace(self, monkeypatch, tmp_path):
        """Create a plugin with a workspace root configured."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "auto_load": False,
        })
        return plugin, workspace

    def test_track_artifact_returns_relative_path(self, plugin_with_workspace):
        """Test that trackArtifact returns relative paths."""
        plugin, workspace = plugin_with_workspace

        abs_path = os.path.join(workspace, "docs", "README.md")
        result = plugin._execute_track_artifact({
            "path": abs_path,
            "artifact_type": "document",
            "description": "Main readme",
        })

        assert result["success"] is True
        assert result["path"] == os.path.join("docs", "README.md")
        assert "docs" in result["message"]

    def test_track_artifact_related_to_relative(self, plugin_with_workspace):
        """Test that related_to paths are also converted to relative."""
        plugin, workspace = plugin_with_workspace

        doc_path = os.path.join(workspace, "docs", "api.md")
        source_path = os.path.join(workspace, "src", "api.py")

        result = plugin._execute_track_artifact({
            "path": doc_path,
            "artifact_type": "document",
            "description": "API documentation",
            "related_to": [source_path],
        })

        assert result["success"] is True
        assert result["related_to"] == [os.path.join("src", "api.py")]


class TestListArtifactsDisplayPaths:
    """Tests for listArtifacts tool returning display paths."""

    @pytest.fixture
    def plugin_with_artifacts(self, monkeypatch, tmp_path):
        """Create a plugin with some tracked artifacts."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "auto_load": False,
        })

        # Track a few artifacts
        plugin._execute_track_artifact({
            "path": os.path.join(workspace, "docs", "README.md"),
            "artifact_type": "document",
            "description": "Main readme",
            "related_to": [os.path.join(workspace, "src", "main.py")],
        })

        return plugin, workspace

    def test_list_artifacts_returns_relative_paths(self, plugin_with_artifacts):
        """Test that listArtifacts returns relative paths."""
        plugin, workspace = plugin_with_artifacts

        result = plugin._execute_list_artifacts({})

        assert result["total"] == 1
        artifact = result["artifacts"][0]
        assert artifact["path"] == os.path.join("docs", "README.md")
        assert artifact["related_to"] == [os.path.join("src", "main.py")]


class TestCheckRelatedDisplayPaths:
    """Tests for checkRelated tool returning display paths."""

    @pytest.fixture
    def plugin_with_relations(self, monkeypatch, tmp_path):
        """Create a plugin with related artifacts."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "auto_load": False,
        })

        source_path = os.path.join(workspace, "src", "api.py")

        # Track doc that depends on api.py
        plugin._execute_track_artifact({
            "path": os.path.join(workspace, "docs", "api.md"),
            "artifact_type": "document",
            "description": "API documentation",
            "related_to": [source_path],
        })

        return plugin, workspace

    def test_check_related_returns_relative_paths(self, plugin_with_relations):
        """Test that checkRelated returns relative paths."""
        plugin, workspace = plugin_with_relations

        source_path = os.path.join(workspace, "src", "api.py")
        result = plugin._execute_check_related({"path": source_path})

        assert result["path"] == os.path.join("src", "api.py")
        assert result["related_count"] == 1
        assert result["related"][0]["path"] == os.path.join("docs", "api.md")

    def test_check_related_no_matches_relative_path(self, plugin_with_relations):
        """Test that checkRelated shows relative path when no matches."""
        plugin, workspace = plugin_with_relations

        unrelated_path = os.path.join(workspace, "src", "other.py")
        result = plugin._execute_check_related({"path": unrelated_path})

        assert result["path"] == os.path.join("src", "other.py")
        assert result["related_count"] == 0


class TestNotifyChangeDisplayPaths:
    """Tests for notifyChange tool returning display paths."""

    @pytest.fixture
    def plugin_with_relations(self, monkeypatch, tmp_path):
        """Create a plugin with related artifacts."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "auto_load": False,
        })

        source_path = os.path.join(workspace, "src", "api.py")

        # Track doc that depends on api.py
        plugin._execute_track_artifact({
            "path": os.path.join(workspace, "docs", "api.md"),
            "artifact_type": "document",
            "description": "API documentation",
            "related_to": [source_path],
        })

        return plugin, workspace

    def test_notify_change_returns_relative_paths(self, plugin_with_relations):
        """Test that notifyChange returns relative paths."""
        plugin, workspace = plugin_with_relations

        source_path = os.path.join(workspace, "src", "api.py")
        result = plugin._execute_notify_change({
            "path": source_path,
            "reason": "Added new endpoint",
        })

        assert result["success"] is True
        assert result["changed_path"] == os.path.join("src", "api.py")
        assert result["flagged_count"] == 1
        assert result["flagged_artifacts"][0]["path"] == os.path.join("docs", "api.md")


class TestAcknowledgeReviewDisplayPaths:
    """Tests for acknowledgeReview tool returning display paths."""

    @pytest.fixture
    def plugin_with_flagged(self, monkeypatch, tmp_path):
        """Create a plugin with a flagged artifact."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "auto_load": False,
        })

        # Track and flag an artifact
        plugin._execute_track_artifact({
            "path": os.path.join(workspace, "docs", "api.md"),
            "artifact_type": "document",
            "description": "API documentation",
        })
        plugin._execute_flag_for_review({
            "path": os.path.join(workspace, "docs", "api.md"),
            "reason": "Source changed",
        })

        return plugin, workspace

    def test_acknowledge_review_returns_relative_path(self, plugin_with_flagged):
        """Test that acknowledgeReview returns relative paths."""
        plugin, workspace = plugin_with_flagged

        doc_path = os.path.join(workspace, "docs", "api.md")
        result = plugin._execute_acknowledge_review({
            "path": doc_path,
            "notes": "Reviewed, no changes needed",
        })

        assert result["success"] is True
        assert result["path"] == os.path.join("docs", "api.md")

    def test_acknowledge_review_not_found_relative(self, plugin_with_flagged):
        """Test that not found error shows relative path."""
        plugin, workspace = plugin_with_flagged

        nonexistent = os.path.join(workspace, "docs", "nonexistent.md")
        result = plugin._execute_acknowledge_review({"path": nonexistent})

        assert "error" in result
        assert os.path.join("docs", "nonexistent.md") in result["error"]


class TestRemoveArtifactDisplayPaths:
    """Tests for removeArtifact tool returning display paths."""

    @pytest.fixture
    def plugin_with_artifact(self, monkeypatch, tmp_path):
        """Create a plugin with a tracked artifact."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "auto_load": False,
        })

        plugin._execute_track_artifact({
            "path": os.path.join(workspace, "docs", "api.md"),
            "artifact_type": "document",
            "description": "API documentation",
        })

        return plugin, workspace

    def test_remove_artifact_returns_relative_path(self, plugin_with_artifact):
        """Test that removeArtifact returns relative paths."""
        plugin, workspace = plugin_with_artifact

        doc_path = os.path.join(workspace, "docs", "api.md")
        result = plugin._execute_remove_artifact({"path": doc_path})

        assert result["success"] is True
        assert result["path"] == os.path.join("docs", "api.md")

    def test_remove_artifact_not_found_relative(self, plugin_with_artifact):
        """Test that not found error shows relative path."""
        plugin, workspace = plugin_with_artifact

        nonexistent = os.path.join(workspace, "docs", "nonexistent.md")
        result = plugin._execute_remove_artifact({"path": nonexistent})

        assert "error" in result
        assert os.path.join("docs", "nonexistent.md") in result["error"]


class TestSystemInstructionsDisplayPaths:
    """Tests for get_system_instructions returning display paths."""

    @pytest.fixture
    def plugin_with_review_needed(self, monkeypatch, tmp_path):
        """Create a plugin with artifacts needing review."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "auto_load": False,
        })

        # Track and flag an artifact
        plugin._execute_track_artifact({
            "path": os.path.join(workspace, "docs", "api.md"),
            "artifact_type": "document",
            "description": "API documentation",
        })
        plugin._execute_flag_for_review({
            "path": os.path.join(workspace, "docs", "api.md"),
            "reason": "Source changed",
        })

        return plugin, workspace

    def test_system_instructions_uses_relative_paths(self, plugin_with_review_needed):
        """Test that system instructions show relative paths."""
        plugin, workspace = plugin_with_review_needed

        instructions = plugin.get_system_instructions()

        # Should contain relative path, not absolute
        assert os.path.join("docs", "api.md") in instructions
        assert workspace not in instructions
