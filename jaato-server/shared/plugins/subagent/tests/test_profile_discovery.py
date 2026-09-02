"""Tests for subagent profile auto-discovery."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from ..config import SubagentConfig, SubagentProfile, ProfileDiscoveryResult, discover_profiles
from ..plugin import SubagentPlugin


@pytest.fixture(autouse=True)
def _isolate_discovery_tiers(monkeypatch, tmp_path):
    """Confine discover_profiles to the WORKSPACE tier for these tests.

    ``discover_profiles`` deliberately merges three sources: the workspace
    dir, ``~/.jaato/profiles/``, and profiles registered by jaato-premium via
    the ``jaato.premium`` -> ``profiles`` entry point.  Tests that create one
    profile in a tmpdir and assert on the total were therefore measuring the
    developer's machine: on this checkout the user tier contributed 6 and an
    installed jaato-premium contributed 15, so a test expecting 1 saw 21.  On
    a clean CI box (empty HOME, no premium) the same tests pass -- which is
    exactly why this went unnoticed, on top of CI never running this path.

    Point HOME at an empty tmp dir and stub the premium tier out, so these
    tests measure the workspace tier they actually populate.  Tests that WANT
    cross-tier behaviour should exercise it explicitly rather than inherit it
    from whoever runs them.
    """
    empty_home = tmp_path / "home"
    (empty_home / ".jaato" / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty_home))
    monkeypatch.setattr(
        "shared.plugins.subagent.config._discover_premium_profiles",
        lambda: {},
    )


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
            result = discover_profiles(tmpdir)

            assert "test_agent" in result.profiles
            profile = result.profiles["test_agent"]
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

            result = discover_profiles(tmpdir)

            assert "my_custom_agent" in result.profiles
            assert result.profiles["my_custom_agent"].description == "Agent from filename"

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

            result = discover_profiles(tmpdir)

            assert len(result.profiles) == 3
            assert "agent1" in result.profiles
            assert "agent2" in result.profiles
            assert "agent3" in result.profiles

    def test_discover_runtime_limits_field(self):
        """runtime_limits is parsed into a RuntimeLimits dataclass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_data = {
                "name": "limited",
                "description": "Limited",
                "plugins": ["cli"],
                "runtime_limits": {
                    "memory_max_mb": 1024,
                    "pids_max": 256,
                    "tool_timeout_seconds": 60,
                },
            }
            profile_path = Path(tmpdir) / "limited.json"
            profile_path.write_text(json.dumps(profile_data))

            result = discover_profiles(tmpdir)
            assert "limited" in result.profiles
            limits = result.profiles["limited"].runtime_limits
            assert limits is not None
            assert limits.memory_max_mb == 1024
            assert limits.pids_max == 256
            assert limits.tool_timeout_seconds == 60
            assert limits.cpu_weight is None  # not set

    def test_discover_invalid_runtime_limits_reports_error(self):
        """Invalid runtime_limits surface as a parse error, not a crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_data = {
                "name": "bad",
                "description": "Bad limits",
                "runtime_limits": {"cpu_weight": 99999},  # out of range
            }
            (Path(tmpdir) / "bad.json").write_text(json.dumps(profile_data))

            result = discover_profiles(tmpdir)
            assert "bad" not in result.profiles
            assert "bad" in result.errors
            assert "runtime_limits" in result.errors["bad"]

    def test_discover_nonexistent_directory(self):
        """Test that non-existent directory returns empty dict."""
        result = discover_profiles("/nonexistent/path/to/profiles")
        assert result.profiles == {}
        assert result.errors == {}

    def test_discover_empty_directory(self):
        """Test that empty directory returns empty result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_profiles(tmpdir)
            assert result.profiles == {}
            assert result.errors == {}

    def test_discover_skips_invalid_json(self):
        """Test that invalid JSON files are reported as errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid profile
            valid_path = Path(tmpdir) / "valid.json"
            valid_path.write_text(json.dumps({
                "name": "valid",
                "description": "Valid profile",
                # `plugins` is required now; [] is the documented
                # minimal framework set (permission/reliability/lifecycle).
                "plugins": []
            }))

            # Create invalid JSON file
            invalid_path = Path(tmpdir) / "invalid.json"
            invalid_path.write_text("{ not valid json }")

            result = discover_profiles(tmpdir)

            assert len(result.profiles) == 1
            assert "valid" in result.profiles
            assert "invalid" in result.errors
            assert "Invalid JSON" in result.errors["invalid"]

    def test_discover_skips_non_dict_json(self):
        """Test that JSON files not containing dicts are reported as errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON with array instead of dict
            array_path = Path(tmpdir) / "array.json"
            array_path.write_text(json.dumps(["item1", "item2"]))

            # Create valid profile
            valid_path = Path(tmpdir) / "valid.json"
            valid_path.write_text(json.dumps({
                "name": "valid",
                "description": "Valid profile",
                # `plugins` is required now; [] is the documented
                # minimal framework set (permission/reliability/lifecycle).
                "plugins": []
            }))

            result = discover_profiles(tmpdir)

            assert len(result.profiles) == 1
            assert "valid" in result.profiles
            assert "array" in result.errors
            assert "JSON object" in result.errors["array"]

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
                "description": "Valid profile",
                # `plugins` is required now; [] is the documented
                # minimal framework set (permission/reliability/lifecycle).
                "plugins": []
            }))

            result = discover_profiles(tmpdir)

            assert len(result.profiles) == 1
            assert "valid" in result.profiles

    def test_discover_relative_path(self):
        """Test discovering profiles with relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profiles subdir
            profiles_dir = Path(tmpdir) / ".jaato" / "profiles"
            profiles_dir.mkdir(parents=True)

            profile_path = profiles_dir / "test.json"
            profile_path.write_text(json.dumps({
                "name": "test",
                "description": "Test profile",
                "plugins": [],
            }))

            # Use relative path with base_path
            result = discover_profiles(".jaato/profiles", base_path=tmpdir)

            assert "test" in result.profiles

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
            }
            profile_path = Path(tmpdir) / "full_agent.json"
            profile_path.write_text(json.dumps(profile_data))

            result = discover_profiles(tmpdir)

            profile = result.profiles["full_agent"]
            assert profile.name == "full_agent"
            assert profile.description == "Agent with all fields"
            assert profile.plugins == ["cli", "mcp", "todo"]
            assert profile.plugin_configs == {"cli": {"timeout": 30}}
            assert profile.system_instructions == "You are a helpful assistant."
            assert profile.model == "gemini-2.5-pro"
            assert profile.max_turns == 20
            # icon / icon_name dropped: removed from the profile schema in
            # 635b00ec, so "all fields" no longer includes them.


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

            result = discover_profiles(tmpdir)

            assert "yaml_agent" in result.profiles
            assert result.profiles["yaml_agent"].description == "A YAML-defined agent"

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

            result = discover_profiles(tmpdir)
            assert "yml_agent" in result.profiles

    def test_apparmor_fragments_equivalent_in_json_and_yaml(self, yaml_available):
        """Piece 1 (2026-05-14): a profile authored with
        ``apparmor_fragments: [host_validator]`` in JSON and the
        equivalent YAML must produce identical ``SubagentProfile``
        instances.  Enforces
        [[feedback-profile-json-yaml-sync]] — JSON ≡ YAML for
        every field including any new ones."""
        if not yaml_available:
            pytest.skip("PyYAML not installed")

        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            json_data = {
                "name": "via_json",
                "description": "Host validator stage (via JSON)",
                "plugins": ["cli"],
                "apparmor": True,
                "apparmor_fragments": ["host_validator", "kb-enablement-2"],
            }
            (Path(tmpdir) / "via_json.json").write_text(json.dumps(json_data))
            yaml_data = dict(json_data)
            yaml_data["name"] = "via_yaml"
            yaml_data["description"] = "Host validator stage (via YAML)"
            (Path(tmpdir) / "via_yaml.yaml").write_text(yaml.dump(yaml_data))

            result = discover_profiles(tmpdir)

            assert "via_json" in result.profiles
            assert "via_yaml" in result.profiles
            j = result.profiles["via_json"]
            y = result.profiles["via_yaml"]
            assert j.apparmor is True and y.apparmor is True
            assert j.apparmor_fragments == ["host_validator", "kb-enablement-2"]
            assert y.apparmor_fragments == ["host_validator", "kb-enablement-2"]

    def test_apparmor_fragments_empty_list_preserved_in_both(self, yaml_available):
        """Explicit ``apparmor_fragments: []`` survives both
        deserialisers without getting silently normalised to None.
        Distinct shapes are load-bearing."""
        if not yaml_available:
            pytest.skip("PyYAML not installed")

        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            json_data = {
                "name": "locked_down_json",
                "description": "Locked-down cascade stage (via JSON)",
                "plugins": [],
                "apparmor": True,
                "apparmor_fragments": [],
            }
            (Path(tmpdir) / "locked_down_json.json").write_text(json.dumps(json_data))
            yaml_data = dict(json_data)
            yaml_data["name"] = "locked_down_yaml"
            yaml_data["description"] = "Locked-down cascade stage (via YAML)"
            (Path(tmpdir) / "locked_down_yaml.yaml").write_text(yaml.dump(yaml_data))

            result = discover_profiles(tmpdir)

            assert result.profiles["locked_down_json"].apparmor_fragments == []
            assert result.profiles["locked_down_yaml"].apparmor_fragments == []
            # Sanity: neither is None.
            assert result.profiles["locked_down_json"].apparmor_fragments is not None
            assert result.profiles["locked_down_yaml"].apparmor_fragments is not None


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
                "plugins": [],
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


class TestScanProfilesDirLogProvenance:
    """Regression pin (2026-05-15): the per-pass summary log emitted
    by ``_scan_profiles_dir`` must list only profile names ACTUALLY
    registered in THAT pass — not the cumulative ``profiles`` dict
    across all preceding passes.

    The bug: before this fix the line read

        Discovered 24 profile(s) from /home/X/.jaato/profiles:
          codegen, host_validator, _base_kb_pipeline, ...

    where ``codegen`` etc. came from a prior workspace-tier pass.
    Operators reading the log assumed those names lived under
    ``~/.jaato/profiles/`` when they didn't.  The count (24) was
    per-pass; the name list was cumulative; the line was internally
    inconsistent and externally misleading.
    """

    def test_summary_log_lists_only_this_pass_names(self, caplog):
        """Pass A registers two names; pass B registers two
        different names.  Pass B's log line must NOT contain
        pass A's names."""
        from ..config import _scan_profiles_dir

        with tempfile.TemporaryDirectory() as d_a, \
             tempfile.TemporaryDirectory() as d_b:
            (Path(d_a) / "alpha.json").write_text(
                json.dumps({"name": "alpha", "description": "from A", "plugins": []})
            )
            (Path(d_a) / "beta.json").write_text(
                json.dumps({"name": "beta", "description": "from A", "plugins": []})
            )
            (Path(d_b) / "gamma.json").write_text(
                json.dumps({"name": "gamma", "description": "from B", "plugins": []})
            )
            (Path(d_b) / "delta.json").write_text(
                json.dumps({"name": "delta", "description": "from B", "plugins": []})
            )

            profiles = {}
            errors = {}
            with caplog.at_level(
                "INFO", logger="shared.plugins.subagent.config",
            ):
                _scan_profiles_dir(Path(d_a), profiles, errors)
                _scan_profiles_dir(Path(d_b), profiles, errors)

            info_lines = [
                r.message for r in caplog.records
                if r.levelname == "INFO" and "Discovered" in r.message
            ]
            assert len(info_lines) == 2, (
                f"Expected one summary log per pass; got: {info_lines}"
            )
            a_line, b_line = info_lines
            # Pass A reports its 2 names; doesn't see B's yet.
            assert "alpha" in a_line and "beta" in a_line
            assert "gamma" not in a_line and "delta" not in a_line
            # Pass B reports only ITS 2 names, not A's.
            assert "gamma" in b_line and "delta" in b_line, (
                f"Pass B should list gamma/delta: {b_line}"
            )
            assert "alpha" not in b_line and "beta" not in b_line, (
                "Pass B's log line includes pass A's names — "
                "regression: provenance mismatch.  Line was: "
                + b_line
            )

    def test_summary_count_matches_listed_names(self, caplog):
        """Count and name-list must agree.  Pre-fix the count was
        per-pass (correct) but the names were cumulative (wrong);
        the line was internally inconsistent: ``N profile(s):`` plus
        a list of length ≠ N.  Pin: count equals comma-separated
        list length, for every pass."""
        from ..config import _scan_profiles_dir
        import re

        with tempfile.TemporaryDirectory() as d_a, \
             tempfile.TemporaryDirectory() as d_b:
            for i in range(3):
                (Path(d_a) / f"a_{i}.json").write_text(
                    json.dumps({"name": f"a_{i}", "description": ""})
                )
            for i in range(2):
                (Path(d_b) / f"b_{i}.json").write_text(
                    json.dumps({"name": f"b_{i}", "description": ""})
                )

            profiles = {}
            errors = {}
            with caplog.at_level(
                "INFO", logger="shared.plugins.subagent.config",
            ):
                _scan_profiles_dir(Path(d_a), profiles, errors)
                _scan_profiles_dir(Path(d_b), profiles, errors)

            for record in caplog.records:
                if record.levelname != "INFO" or "Discovered" not in record.message:
                    continue
                m = re.match(
                    r"Discovered (\d+) profile\(s\) from .*?: (.+)",
                    record.message,
                )
                assert m, f"unexpected log shape: {record.message}"
                count = int(m.group(1))
                names = [n.strip() for n in m.group(2).split(",")]
                assert count == len(names), (
                    f"count/name-list mismatch: {count} vs "
                    f"{len(names)} names in {record.message!r}"
                )


class TestSuppressInheritedProcessorsEndToEnd:
    """#791 measured through ``discover_profiles``, the surface a profile
    author actually writes against — YAML in, resolved profile out."""

    BASE = """
name: base
description: Base worker
plugins: []
completion_processors:
  - script: scripts/processors/accept.py
    name: acceptance
  - script: scripts/processors/audit.py
max_turns: 4
env:
  STAGE: worker
"""

    def _write(self, tmpdir, child_yaml):
        (Path(tmpdir) / "base.yaml").write_text(self.BASE)
        (Path(tmpdir) / "child.yaml").write_text(child_yaml)
        return discover_profiles(tmpdir)

    def test_child_declines_one_processor_and_keeps_the_rest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._write(tmpdir, """
name: child
description: Interrogate
plugins: []
inherits: [base]
suppress_inherited_processors:
  - acceptance
""")
            assert not result.errors
            child = result.profiles["child"]
            assert [p.script for p in child.completion_processors] == [
                "scripts/processors/audit.py"]
            # ...and every ceiling the base declared survives
            assert child.max_turns == 4
            assert child.env == {"STAGE": "worker"}

    def test_empty_completion_processors_still_inherits(self):
        """The behaviour #791 measured and the docs never stated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._write(tmpdir, """
name: child
description: Child
plugins: []
inherits: [base]
completion_processors: []
""")
            assert not result.errors
            assert len(result.profiles["child"].completion_processors) == 2

    def test_stale_suppression_fails_the_profile_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._write(tmpdir, """
name: child
description: Child
plugins: []
inherits: [base]
suppress_inherited_processors:
  - renamed_away
""")
            assert "child" not in result.profiles
            assert "renamed_away" in result.errors["child"]
