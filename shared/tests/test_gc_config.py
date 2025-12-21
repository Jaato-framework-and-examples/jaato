"""Tests for GC configuration loading and GCProfileConfig."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from shared.plugins.gc import load_gc_from_file, GCConfig
from shared.plugins.subagent.config import GCProfileConfig, SubagentProfile, discover_profiles


class TestGCProfileConfig:
    """Tests for GCProfileConfig dataclass."""

    def test_default_values(self):
        """Test GCProfileConfig has correct defaults."""
        config = GCProfileConfig()

        assert config.type == "truncate"
        assert config.threshold_percent == 80.0
        assert config.preserve_recent_turns == 5
        assert config.notify_on_gc is True
        assert config.summarize_middle_turns is None
        assert config.max_turns is None
        assert config.plugin_config == {}

    def test_custom_values(self):
        """Test GCProfileConfig with custom values."""
        config = GCProfileConfig(
            type="hybrid",
            threshold_percent=75.0,
            preserve_recent_turns=10,
            notify_on_gc=False,
            summarize_middle_turns=15,
            max_turns=100,
            plugin_config={"key": "value"}
        )

        assert config.type == "hybrid"
        assert config.threshold_percent == 75.0
        assert config.preserve_recent_turns == 10
        assert config.notify_on_gc is False
        assert config.summarize_middle_turns == 15
        assert config.max_turns == 100
        assert config.plugin_config == {"key": "value"}

    def test_from_dict_with_all_fields(self):
        """Test GCProfileConfig.from_dict with all fields."""
        data = {
            "type": "summarize",
            "threshold_percent": 70.0,
            "preserve_recent_turns": 3,
            "notify_on_gc": False,
            "summarize_middle_turns": 20,
            "max_turns": 50,
            "plugin_config": {"option": True}
        }

        config = GCProfileConfig.from_dict(data)

        assert config.type == "summarize"
        assert config.threshold_percent == 70.0
        assert config.preserve_recent_turns == 3
        assert config.notify_on_gc is False
        assert config.summarize_middle_turns == 20
        assert config.max_turns == 50
        assert config.plugin_config == {"option": True}

    def test_from_dict_with_minimal_fields(self):
        """Test GCProfileConfig.from_dict with minimal fields."""
        data = {"type": "hybrid"}

        config = GCProfileConfig.from_dict(data)

        assert config.type == "hybrid"
        assert config.threshold_percent == 80.0  # default
        assert config.preserve_recent_turns == 5  # default
        assert config.notify_on_gc is True  # default

    def test_from_dict_empty(self):
        """Test GCProfileConfig.from_dict with empty dict uses defaults."""
        config = GCProfileConfig.from_dict({})

        assert config.type == "truncate"
        assert config.threshold_percent == 80.0


class TestSubagentProfileWithGC:
    """Tests for SubagentProfile with gc field."""

    def test_profile_without_gc(self):
        """Test SubagentProfile without gc field."""
        profile = SubagentProfile(
            name="test",
            description="Test profile",
            plugins=["cli"]
        )

        assert profile.gc is None

    def test_profile_with_gc(self):
        """Test SubagentProfile with gc field."""
        gc_config = GCProfileConfig(type="hybrid", threshold_percent=75.0)
        profile = SubagentProfile(
            name="test",
            description="Test profile",
            plugins=["cli"],
            gc=gc_config
        )

        assert profile.gc is not None
        assert profile.gc.type == "hybrid"
        assert profile.gc.threshold_percent == 75.0


class TestDiscoverProfilesWithGC:
    """Tests for discover_profiles with gc field."""

    def test_discover_profile_with_gc_field(self):
        """Test that discover_profiles parses gc field correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_data = {
                "name": "gc-test-agent",
                "description": "Agent with GC config",
                "plugins": ["cli"],
                "gc": {
                    "type": "hybrid",
                    "threshold_percent": 85.0,
                    "preserve_recent_turns": 8,
                    "summarize_middle_turns": 12
                }
            }

            profile_path = Path(temp_dir) / "gc-test-agent.json"
            with open(profile_path, "w") as f:
                json.dump(profile_data, f)

            profiles = discover_profiles(temp_dir)

            assert "gc-test-agent" in profiles
            profile = profiles["gc-test-agent"]
            assert profile.gc is not None
            assert profile.gc.type == "hybrid"
            assert profile.gc.threshold_percent == 85.0
            assert profile.gc.preserve_recent_turns == 8
            assert profile.gc.summarize_middle_turns == 12

    def test_discover_profile_without_gc_field(self):
        """Test that discover_profiles works without gc field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_data = {
                "name": "no-gc-agent",
                "description": "Agent without GC",
                "plugins": ["cli"]
            }

            profile_path = Path(temp_dir) / "no-gc-agent.json"
            with open(profile_path, "w") as f:
                json.dump(profile_data, f)

            profiles = discover_profiles(temp_dir)

            assert "no-gc-agent" in profiles
            profile = profiles["no-gc-agent"]
            assert profile.gc is None


class TestLoadGCFromFile:
    """Tests for load_gc_from_file function."""

    def test_load_valid_gc_json(self):
        """Test loading valid gc.json file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gc_data = {
                "type": "hybrid",
                "threshold_percent": 75.0,
                "preserve_recent_turns": 10,
                "notify_on_gc": True,
                "summarize_middle_turns": 15
            }

            gc_path = Path(temp_dir) / "gc.json"
            with open(gc_path, "w") as f:
                json.dump(gc_data, f)

            # Mock the plugin loading
            with patch('shared.plugins.gc.load_gc_plugin') as mock_load:
                mock_plugin = MagicMock()
                mock_plugin.name = "gc_hybrid"
                mock_load.return_value = mock_plugin

                result = load_gc_from_file(str(gc_path))

                assert result is not None
                gc_plugin, gc_config = result
                assert gc_plugin.name == "gc_hybrid"
                assert gc_config.threshold_percent == 75.0
                assert gc_config.preserve_recent_turns == 10

                # Verify load_gc_plugin was called with correct args
                mock_load.assert_called_once()
                call_args = mock_load.call_args
                assert call_args[0][0] == "gc_hybrid"
                assert call_args[0][1]["preserve_recent_turns"] == 10
                assert call_args[0][1]["notify_on_gc"] is True
                assert call_args[0][1]["summarize_middle_turns"] == 15

    def test_load_gc_json_missing_file(self):
        """Test load_gc_from_file returns None for missing file."""
        result = load_gc_from_file("/nonexistent/path/gc.json")
        assert result is None

    def test_load_gc_json_invalid_json(self):
        """Test load_gc_from_file returns None for invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gc_path = Path(temp_dir) / "gc.json"
            with open(gc_path, "w") as f:
                f.write("{ invalid json }")

            result = load_gc_from_file(str(gc_path))
            assert result is None

    def test_load_gc_json_plugin_not_found(self):
        """Test load_gc_from_file returns None if plugin not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gc_data = {"type": "nonexistent_gc"}

            gc_path = Path(temp_dir) / "gc.json"
            with open(gc_path, "w") as f:
                json.dump(gc_data, f)

            with patch('shared.plugins.gc.load_gc_plugin') as mock_load:
                mock_load.side_effect = ValueError("Plugin not found")

                result = load_gc_from_file(str(gc_path))
                assert result is None

    def test_load_gc_json_type_prefix_handling(self):
        """Test that type names are normalized (truncate -> gc_truncate)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gc_data = {"type": "truncate"}  # Without gc_ prefix

            gc_path = Path(temp_dir) / "gc.json"
            with open(gc_path, "w") as f:
                json.dump(gc_data, f)

            with patch('shared.plugins.gc.load_gc_plugin') as mock_load:
                mock_plugin = MagicMock()
                mock_load.return_value = mock_plugin

                load_gc_from_file(str(gc_path))

                # Should be called with gc_truncate
                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == "gc_truncate"

    def test_load_gc_json_already_prefixed(self):
        """Test that already prefixed type names work."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gc_data = {"type": "gc_summarize"}  # Already has gc_ prefix

            gc_path = Path(temp_dir) / "gc.json"
            with open(gc_path, "w") as f:
                json.dump(gc_data, f)

            with patch('shared.plugins.gc.load_gc_plugin') as mock_load:
                mock_plugin = MagicMock()
                mock_load.return_value = mock_plugin

                load_gc_from_file(str(gc_path))

                # Should keep gc_summarize as-is
                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == "gc_summarize"

    def test_load_gc_json_default_path(self):
        """Test default path is .jaato/gc.json."""
        # Save current directory
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)

                # Create .jaato/gc.json
                jaato_dir = Path(temp_dir) / ".jaato"
                jaato_dir.mkdir()
                gc_data = {"type": "truncate"}
                gc_path = jaato_dir / "gc.json"
                with open(gc_path, "w") as f:
                    json.dump(gc_data, f)

                with patch('shared.plugins.gc.load_gc_plugin') as mock_load:
                    mock_plugin = MagicMock()
                    mock_load.return_value = mock_plugin

                    # Call without path - should use default
                    result = load_gc_from_file()

                    assert result is not None
            finally:
                os.chdir(original_cwd)
