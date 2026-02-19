"""Tests for the artifact tracker plugin."""

import json
import os
import pytest
import tempfile
from unittest.mock import patch

from ..plugin import (
    ArtifactTrackerPlugin,
    _normalize_path_standalone,
    _detect_workspace_root,
    create_plugin,
)


class TestNormalizePathStandalone:
    """Tests for standalone path normalization (uses CWD)."""

    def test_normalize_relative_path(self):
        """Test that relative paths are converted to absolute (against CWD)."""
        result = _normalize_path_standalone("doc.md")
        assert os.path.isabs(result)
        assert result.endswith("doc.md")

    def test_normalize_dot_prefix(self):
        """Test that ./ prefix is handled."""
        result = _normalize_path_standalone("./doc.md")
        assert os.path.isabs(result)
        assert result.endswith("doc.md")

    def test_normalize_double_dot(self):
        """Test that .. references are collapsed."""
        result = _normalize_path_standalone("./foo/../doc.md")
        assert os.path.isabs(result)
        assert ".." not in result
        assert result.endswith("doc.md")

    def test_normalize_empty_string(self):
        """Test that empty string returns empty string."""
        result = _normalize_path_standalone("")
        assert result == ""


class TestNormalizePathWithWorkspace:
    """Tests for instance method path normalization (uses workspace_root)."""

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

    def test_relative_path_resolved_against_workspace(self, plugin_with_workspace):
        """Test that relative paths are resolved against workspace_root, not CWD."""
        plugin, workspace = plugin_with_workspace

        # Relative path should be joined with workspace_root
        result = plugin._normalize_path("customer-domain-api/file.java")

        expected = os.path.normpath(os.path.join(workspace, "customer-domain-api/file.java"))
        assert result == expected

    def test_relative_path_with_dot_prefix(self, plugin_with_workspace):
        """Test that ./prefix relative paths are resolved against workspace_root."""
        plugin, workspace = plugin_with_workspace

        result = plugin._normalize_path("./src/main.py")

        expected = os.path.normpath(os.path.join(workspace, "src/main.py"))
        assert result == expected

    def test_relative_path_with_double_dot(self, plugin_with_workspace):
        """Test that ../ in paths are collapsed correctly with workspace_root."""
        plugin, workspace = plugin_with_workspace

        result = plugin._normalize_path("foo/../bar/file.py")

        expected = os.path.normpath(os.path.join(workspace, "bar/file.py"))
        assert result == expected

    def test_absolute_path_unchanged(self, plugin_with_workspace):
        """Test that absolute paths are just normalized, not joined with workspace."""
        plugin, workspace = plugin_with_workspace

        abs_path = "/some/absolute/path/file.py"
        result = plugin._normalize_path(abs_path)

        # Absolute path should just be normalized, not joined with workspace
        assert result == os.path.normpath(abs_path)

    def test_empty_string_unchanged(self, plugin_with_workspace):
        """Test that empty string returns empty string."""
        plugin, workspace = plugin_with_workspace

        result = plugin._normalize_path("")
        assert result == ""

    def test_relative_path_without_workspace_uses_cwd(self, plugin_no_workspace):
        """Test that relative paths fall back to CWD when no workspace_root is set."""
        plugin = plugin_no_workspace

        result = plugin._normalize_path("doc.md")

        # Should fall back to os.path.abspath behavior (against CWD)
        expected = os.path.abspath("doc.md")
        assert result == expected

    def test_subagent_workspace_different_from_cwd(self, monkeypatch, tmp_path):
        """Test the specific bug scenario: subagent workspace differs from process CWD.

        This is the bug we're fixing: when a subagent has a different workspace
        than the main process CWD, relative paths should resolve against the
        subagent's workspace, not the process CWD.
        """
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        # Simulate: CWD is /home/user/jaato
        # Subagent workspace is .../tests_enablement_2.0/test_2/
        cwd = os.getcwd()
        subagent_workspace = str(tmp_path / "tests_enablement_2.0" / "test_2")
        os.makedirs(subagent_workspace, exist_ok=True)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": subagent_workspace,
            "auto_load": False,
        })

        # Relative path: customer-domain-api/file.java
        # BUG: would become {CWD}/customer-domain-api/file.java (wrong)
        # FIX: should become {subagent_workspace}/customer-domain-api/file.java (correct)
        result = plugin._normalize_path("customer-domain-api/file.java")

        # Should NOT be relative to CWD
        wrong_result = os.path.normpath(os.path.join(cwd, "customer-domain-api/file.java"))
        assert result != wrong_result, "Bug: path resolved against CWD instead of workspace_root"

        # Should be relative to subagent_workspace
        expected = os.path.normpath(os.path.join(subagent_workspace, "customer-domain-api/file.java"))
        assert result == expected


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


class TestInitializationClearsStaleState:
    """Tests that initialization removes stale .artifact_tracker.json from previous sessions."""

    def test_stale_state_file_removed_on_init(self, monkeypatch, tmp_path):
        """Initializing a new session removes any existing state file."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        storage_path = str(tmp_path / ".jaato" / ".artifact_tracker.json")
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Simulate leftover state from a previous session
        stale_data = {
            "artifacts": {
                "stale-id": {
                    "artifact_id": "stale-id",
                    "path": "/old/file.md",
                    "artifact_type": "document",
                    "description": "Leftover from last session",
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-01T00:00:00",
                    "review_status": "current",
                    "tags": [],
                    "related_to": [],
                }
            }
        }
        with open(storage_path, "w") as f:
            json.dump(stale_data, f)

        # Default init (auto_load=False by default) should remove the file
        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "storage_path": storage_path,
        })

        # File should be removed
        assert not os.path.exists(storage_path)
        # Registry should be empty
        result = plugin._execute_list_artifacts({})
        assert result["total"] == 0

    def test_auto_load_true_preserves_state(self, monkeypatch, tmp_path):
        """When auto_load=True, existing state is loaded instead of cleared."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        storage_path = str(tmp_path / ".jaato" / ".artifact_tracker.json")
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Write state that should be loaded
        state_data = {
            "artifacts": {
                "existing-id": {
                    "artifact_id": "existing-id",
                    "path": os.path.join(workspace, "docs", "readme.md"),
                    "artifact_type": "document",
                    "description": "Preserved artifact",
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-01T00:00:00",
                    "review_status": "current",
                    "tags": [],
                    "related_to": [],
                }
            }
        }
        with open(storage_path, "w") as f:
            json.dump(state_data, f)

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "storage_path": storage_path,
            "auto_load": True,
        })

        # State should be loaded
        result = plugin._execute_list_artifacts({})
        assert result["total"] == 1

    def test_init_no_file_does_not_error(self, monkeypatch, tmp_path):
        """Initialization succeeds when there is no existing state file."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        storage_path = str(tmp_path / ".jaato" / ".artifact_tracker.json")
        # Deliberately do NOT create the file

        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "storage_path": storage_path,
        })

        # Should start with an empty registry
        result = plugin._execute_list_artifacts({})
        assert result["total"] == 0

    def test_reinit_clears_previous_session_artifacts(self, monkeypatch, tmp_path):
        """Re-initializing the plugin clears artifacts tracked in the prior session."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        workspace = str(tmp_path / "workspace")
        os.makedirs(workspace, exist_ok=True)

        storage_path = str(tmp_path / ".jaato" / ".artifact_tracker.json")

        # First session: track an artifact
        plugin = create_plugin()
        plugin.initialize({
            "workspace_root": workspace,
            "storage_path": storage_path,
        })
        plugin._execute_track_artifact({
            "path": os.path.join(workspace, "docs", "api.md"),
            "artifact_type": "document",
            "description": "API docs",
        })
        result = plugin._execute_list_artifacts({})
        assert result["total"] == 1

        # Simulate session end
        plugin.shutdown()

        # Second session: re-initialize (simulates new session on same workspace)
        plugin2 = create_plugin()
        plugin2.initialize({
            "workspace_root": workspace,
            "storage_path": storage_path,
        })

        # Should start clean — no artifacts from previous session
        result2 = plugin2._execute_list_artifacts({})
        assert result2["total"] == 0
