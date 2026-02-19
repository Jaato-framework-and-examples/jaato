"""Tests for subagent profile auto-discovery."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from ..config import SubagentConfig, SubagentProfile, discover_profiles
from ..plugin import SubagentPlugin


class TestDiscoverProfiles:
    """Tests for the discover_profiles function."""

    def test_discover_json_profiles(self):
        """Test discovering profiles from JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test profile JSON file
            profile_data = {
                "name": "test_agent",
                "description": "A test agent",
                "plugins": ["cli", "todo"],
                "max_turns": 5,
            }
            profile_path = Path(tmpdir) / "test_agent.json"
            profile_path.write_text(json.dumps(profile_data))

            # Discover profiles
            profiles = discover_profiles(tmpdir)

            assert "test_agent" in profiles
            profile = profiles["test_agent"]
            assert profile.name == "test_agent"
            assert profile.description == "A test agent"
            assert profile.plugins == ["cli", "todo"]
            assert profile.max_turns == 5

    def test_discover_profile_name_from_filename(self):
        """Test that profile name defaults to filename if not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a profile without explicit name
            profile_data = {
                "description": "Agent from filename",
                "plugins": ["mcp"],
            }
            profile_path = Path(tmpdir) / "my_custom_agent.json"
            profile_path.write_text(json.dumps(profile_data))

            profiles = discover_profiles(tmpdir)

            assert "my_custom_agent" in profiles
            assert profiles["my_custom_agent"].description == "Agent from filename"

    def test_discover_multiple_profiles(self):
        """Test discovering multiple profiles from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple profile files
            profiles_data = [
                {"name": "agent1", "description": "First agent", "plugins": ["cli"]},
                {"name": "agent2", "description": "Second agent", "plugins": ["mcp"]},
                {"name": "agent3", "description": "Third agent", "plugins": ["todo"]},
            ]
            for data in profiles_data:
                path = Path(tmpdir) / f"{data['name']}.json"
                path.write_text(json.dumps(data))

            profiles = discover_profiles(tmpdir)

            assert len(profiles) == 3
            assert "agent1" in profiles
            assert "agent2" in profiles
            assert "agent3" in profiles

    def test_discover_nonexistent_directory(self):
        """Test that non-existent directory returns empty dict."""
        profiles = discover_profiles("/nonexistent/path/to/profiles")
        assert profiles == {}

    def test_discover_empty_directory(self):
        """Test that empty directory returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles = discover_profiles(tmpdir)
            assert profiles == {}

    def test_discover_skips_invalid_json(self):
        """Test that invalid JSON files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid profile
            valid_path = Path(tmpdir) / "valid.json"
            valid_path.write_text(json.dumps({
                "name": "valid",
                "description": "Valid profile"
            }))

            # Create invalid JSON file
            invalid_path = Path(tmpdir) / "invalid.json"
            invalid_path.write_text("{ not valid json }")

            profiles = discover_profiles(tmpdir)

            assert len(profiles) == 1
            assert "valid" in profiles

    def test_discover_skips_non_dict_json(self):
        """Test that JSON files not containing dicts are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON with array instead of dict
            array_path = Path(tmpdir) / "array.json"
            array_path.write_text(json.dumps(["item1", "item2"]))

            # Create valid profile
            valid_path = Path(tmpdir) / "valid.json"
            valid_path.write_text(json.dumps({
                "name": "valid",
                "description": "Valid profile"
            }))

            profiles = discover_profiles(tmpdir)

            assert len(profiles) == 1
            assert "valid" in profiles

    def test_discover_skips_non_profile_files(self):
        """Test that non-JSON/YAML files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create various non-profile files
            (Path(tmpdir) / "readme.txt").write_text("Some readme")
            (Path(tmpdir) / "script.py").write_text("print('hello')")
            (Path(tmpdir) / ".gitignore").write_text("*.pyc")

            # Create valid profile
            valid_path = Path(tmpdir) / "valid.json"
            valid_path.write_text(json.dumps({
                "name": "valid",
                "description": "Valid profile"
            }))

            profiles = discover_profiles(tmpdir)

            assert len(profiles) == 1
            assert "valid" in profiles

    def test_discover_relative_path(self):
        """Test discovering profiles with relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profiles subdir
            profiles_dir = Path(tmpdir) / ".jaato" / "profiles"
            profiles_dir.mkdir(parents=True)

            profile_path = profiles_dir / "test.json"
            profile_path.write_text(json.dumps({
                "name": "test",
                "description": "Test profile"
            }))

            # Use relative path with base_path
            profiles = discover_profiles(".jaato/profiles", base_path=tmpdir)

            assert "test" in profiles

    def test_discover_all_profile_fields(self):
        """Test that all profile fields are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_data = {
                "name": "full_agent",
                "description": "Agent with all fields",
                "plugins": ["cli", "mcp", "todo"],
                "plugin_configs": {"cli": {"timeout": 30}},
                "system_instructions": "You are a helpful assistant.",
                "model": "gemini-2.5-pro",
                "max_turns": 20,
                "auto_approved": True,
                "icon": ["[*]", "| |", "---"],
                "icon_name": "custom_icon",
            }
            profile_path = Path(tmpdir) / "full_agent.json"
            profile_path.write_text(json.dumps(profile_data))

            profiles = discover_profiles(tmpdir)

            profile = profiles["full_agent"]
            assert profile.name == "full_agent"
            assert profile.description == "Agent with all fields"
            assert profile.plugins == ["cli", "mcp", "todo"]
            assert profile.plugin_configs == {"cli": {"timeout": 30}}
            assert profile.system_instructions == "You are a helpful assistant."
            assert profile.model == "gemini-2.5-pro"
            assert profile.max_turns == 20
            assert profile.auto_approved is True
            assert profile.icon == ["[*]", "| |", "---"]
            assert profile.icon_name == "custom_icon"


class TestDiscoverYamlProfiles:
    """Tests for discovering YAML profiles (requires PyYAML)."""

    @pytest.fixture
    def yaml_available(self):
        """Check if PyYAML is available."""
        try:
            import yaml
            return True
        except ImportError:
            return False

    def test_discover_yaml_profiles(self, yaml_available):
        """Test discovering profiles from YAML files."""
        if not yaml_available:
            pytest.skip("PyYAML not installed")

        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_data = {
                "name": "yaml_agent",
                "description": "A YAML-defined agent",
                "plugins": ["cli"],
            }
            profile_path = Path(tmpdir) / "yaml_agent.yaml"
            profile_path.write_text(yaml.dump(profile_data))

            profiles = discover_profiles(tmpdir)

            assert "yaml_agent" in profiles
            assert profiles["yaml_agent"].description == "A YAML-defined agent"

    def test_discover_yml_extension(self, yaml_available):
        """Test discovering profiles with .yml extension."""
        if not yaml_available:
            pytest.skip("PyYAML not installed")

        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_data = {
                "name": "yml_agent",
                "description": "A YML-defined agent",
                "plugins": ["mcp"],
            }
            profile_path = Path(tmpdir) / "yml_agent.yml"
            profile_path.write_text(yaml.dump(profile_data))

            profiles = discover_profiles(tmpdir)

            assert "yml_agent" in profiles


class TestSubagentConfigAutoDiscover:
    """Tests for SubagentConfig auto-discover settings."""

    def test_config_defaults(self):
        """Test that auto_discover_profiles defaults to True."""
        config = SubagentConfig(project="test", location="us-central1")
        assert config.auto_discover_profiles is True
        assert config.profiles_dir == ".jaato/profiles"

    def test_config_from_dict_with_auto_discover(self):
        """Test from_dict parses auto_discover settings."""
        data = {
            "project": "test",
            "location": "us-central1",
            "auto_discover_profiles": False,
            "profiles_dir": "custom/profiles/path",
        }
        config = SubagentConfig.from_dict(data)

        assert config.auto_discover_profiles is False
        assert config.profiles_dir == "custom/profiles/path"

    def test_config_from_dict_defaults(self):
        """Test from_dict uses defaults for missing auto_discover settings."""
        data = {
            "project": "test",
            "location": "us-central1",
        }
        config = SubagentConfig.from_dict(data)

        assert config.auto_discover_profiles is True
        assert config.profiles_dir == ".jaato/profiles"


class TestPluginAutoDiscovery:
    """Tests for plugin initialization with auto-discovery."""

    def test_plugin_discovers_profiles_on_init(self):
        """Test that plugin discovers profiles during initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profiles directory
            profiles_dir = Path(tmpdir) / ".jaato" / "profiles"
            profiles_dir.mkdir(parents=True)

            # Create a test profile
            profile_path = profiles_dir / "discovered_agent.json"
            profile_path.write_text(json.dumps({
                "name": "discovered_agent",
                "description": "Auto-discovered agent",
                "plugins": ["cli"],
            }))

            # Change to temp directory so relative path works
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                plugin = SubagentPlugin()
                plugin.initialize({
                    "auto_discover_profiles": True,
                    "profiles_dir": ".jaato/profiles",
                })

                assert "discovered_agent" in plugin._config.profiles
                assert plugin._config.profiles["discovered_agent"].description == "Auto-discovered agent"
            finally:
                os.chdir(original_cwd)

    def test_plugin_skips_discovery_when_disabled(self):
        """Test that plugin skips discovery when auto_discover_profiles is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profiles directory with a profile
            profiles_dir = Path(tmpdir) / ".jaato" / "profiles"
            profiles_dir.mkdir(parents=True)

            profile_path = profiles_dir / "should_not_discover.json"
            profile_path.write_text(json.dumps({
                "name": "should_not_discover",
                "description": "Should not be discovered",
            }))

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                plugin = SubagentPlugin()
                plugin.initialize({
                    "auto_discover_profiles": False,
                })

                assert "should_not_discover" not in plugin._config.profiles
            finally:
                os.chdir(original_cwd)

    def test_explicit_profiles_take_precedence(self):
        """Test that explicit profiles take precedence over discovered ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profiles directory
            profiles_dir = Path(tmpdir) / ".jaato" / "profiles"
            profiles_dir.mkdir(parents=True)

            # Create a discovered profile with same name as explicit one
            profile_path = profiles_dir / "my_agent.json"
            profile_path.write_text(json.dumps({
                "name": "my_agent",
                "description": "Discovered version",
                "plugins": ["mcp"],
            }))

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                plugin = SubagentPlugin()
                plugin.initialize({
                    "auto_discover_profiles": True,
                    "profiles_dir": ".jaato/profiles",
                    "profiles": {
                        "my_agent": {
                            "description": "Explicit version",
                            "plugins": ["cli"],
                        }
                    }
                })

                # Explicit profile should take precedence
                assert plugin._config.profiles["my_agent"].description == "Explicit version"
                assert plugin._config.profiles["my_agent"].plugins == ["cli"]
            finally:
                os.chdir(original_cwd)

    def test_merge_explicit_and_discovered_profiles(self):
        """Test that explicit and discovered profiles are merged correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profiles directory
            profiles_dir = Path(tmpdir) / ".jaato" / "profiles"
            profiles_dir.mkdir(parents=True)

            # Create discovered profile
            profile_path = profiles_dir / "discovered.json"
            profile_path.write_text(json.dumps({
                "name": "discovered",
                "description": "A discovered agent",
            }))

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                plugin = SubagentPlugin()
                plugin.initialize({
                    "auto_discover_profiles": True,
                    "profiles_dir": ".jaato/profiles",
                    "profiles": {
                        "explicit": {
                            "description": "An explicit agent",
                        }
                    }
                })

                # Both profiles should exist
                assert "explicit" in plugin._config.profiles
                assert "discovered" in plugin._config.profiles
            finally:
                os.chdir(original_cwd)
