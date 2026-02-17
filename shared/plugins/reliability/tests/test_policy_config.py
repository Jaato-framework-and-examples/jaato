"""Tests for file-based policy configuration loading."""

import json
import pytest
from pathlib import Path

from ..policy_config import (
    generate_default_config_safe,
    get_default_policy_config_path,
    load_policy_config,
    resolve_policy_config_path,
)
from ..types import (
    NudgeType,
    PatternDetectionConfig,
    PatternSeverity,
    PrerequisitePolicy,
)
from ..plugin import ReliabilityPlugin


class TestResolveConfigPath:
    """Tests for config file resolution."""

    def test_no_file_returns_none(self, tmp_path):
        """Returns None when no config file exists at either level."""
        assert resolve_policy_config_path(str(tmp_path)) is None

    def test_workspace_file_found(self, tmp_path):
        """Finds config file at workspace level."""
        config_dir = tmp_path / ".jaato"
        config_dir.mkdir()
        config_file = config_dir / "reliability-policies.json"
        config_file.write_text("{}")

        result = resolve_policy_config_path(str(tmp_path))
        assert result == config_file

    def test_user_file_found(self, tmp_path, monkeypatch):
        """Finds config file at user level when workspace has none."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        user_dir = fake_home / ".jaato"
        user_dir.mkdir()
        user_file = user_dir / "reliability-policies.json"
        user_file.write_text("{}")

        # No workspace path
        result = resolve_policy_config_path(None)
        assert result == user_file

    def test_workspace_takes_precedence(self, tmp_path, monkeypatch):
        """Workspace config takes precedence over user config."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        # Create both files
        ws_dir = tmp_path / "workspace" / ".jaato"
        ws_dir.mkdir(parents=True)
        ws_file = ws_dir / "reliability-policies.json"
        ws_file.write_text('{"from": "workspace"}')

        user_dir = fake_home / ".jaato"
        user_dir.mkdir()
        user_file = user_dir / "reliability-policies.json"
        user_file.write_text('{"from": "user"}')

        result = resolve_policy_config_path(str(tmp_path / "workspace"))
        assert result == ws_file


class TestGetDefaultConfigPath:
    """Tests for default config path generation."""

    def test_with_workspace(self, tmp_path):
        """Returns workspace-level path when workspace is set."""
        result = get_default_policy_config_path(str(tmp_path))
        assert result == tmp_path / ".jaato" / "reliability-policies.json"

    def test_without_workspace(self, monkeypatch, tmp_path):
        """Returns user-level path when no workspace is set."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        result = get_default_policy_config_path(None)
        assert result == tmp_path / ".jaato" / "reliability-policies.json"


class TestLoadPatternDetection:
    """Tests for pattern_detection section parsing."""

    def test_empty_file(self, tmp_path):
        """Empty config returns no overrides."""
        config_file = tmp_path / ".jaato" / "reliability-policies.json"
        config_file.parent.mkdir()
        config_file.write_text("{}")

        config, policies, warnings = load_policy_config(config_path=config_file)
        assert config is None
        assert policies == []
        assert warnings == []

    def test_repetitive_call_threshold(self, tmp_path):
        """Parses repetitive_call_threshold correctly."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "pattern_detection": {
                "repetitive_call_threshold": 5,
            }
        }))

        config, _, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert config is not None
        assert config.repetitive_call_threshold == 5

    def test_error_retry_threshold(self, tmp_path):
        """Parses error_retry_threshold correctly."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "pattern_detection": {
                "error_retry_threshold": 2,
            }
        }))

        config, _, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert config is not None
        assert config.error_retry_threshold == 2

    def test_all_pattern_detection_fields(self, tmp_path):
        """Parses all supported pattern_detection fields."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "pattern_detection": {
                "repetitive_call_threshold": 4,
                "error_retry_threshold": 2,
                "introspection_loop_threshold": 3,
                "max_reads_before_action": 8,
                "max_turn_duration_seconds": 60.0,
                "introspection_tool_names": ["list_tools"],
                "read_only_tools": ["readFile", "glob"],
                "action_tools": ["writeFile"],
                "announce_phrases": ["let me", "starting"],
            }
        }))

        config, _, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert config is not None
        assert config.repetitive_call_threshold == 4
        assert config.error_retry_threshold == 2
        assert config.introspection_loop_threshold == 3
        assert config.max_reads_before_action == 8
        assert config.max_turn_duration_seconds == 60.0
        assert config.introspection_tool_names == {"list_tools"}
        assert config.read_only_tools == {"readFile", "glob"}
        assert config.action_tools == {"writeFile"}
        assert config.announce_phrases == ["let me", "starting"]

    def test_invalid_threshold_warns(self, tmp_path):
        """Invalid threshold values produce warnings."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "pattern_detection": {
                "repetitive_call_threshold": -1,
            }
        }))

        config, _, warnings = load_policy_config(config_path=config_file)
        assert len(warnings) == 1
        assert "repetitive_call_threshold" in warnings[0]
        # Invalid field means no config override produced
        assert config is None

    def test_non_object_pattern_detection_warns(self, tmp_path):
        """Non-object pattern_detection produces a warning."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "pattern_detection": "invalid"
        }))

        config, _, warnings = load_policy_config(config_path=config_file)
        assert len(warnings) == 1
        assert "'pattern_detection' must be an object" in warnings[0]
        assert config is None


class TestLoadPrerequisitePolicies:
    """Tests for prerequisite_policies section parsing."""

    def test_single_policy(self, tmp_path):
        """Parses a single valid prerequisite policy."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "plan_before_update",
                    "prerequisite_tool": "createPlan",
                    "gated_tools": ["updateStep"],
                    "lookback_turns": 5,
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert len(policies) == 1

        policy = policies[0]
        assert policy.policy_id == "plan_before_update"
        assert policy.prerequisite_tool == "createPlan"
        assert policy.gated_tools == {"updateStep"}
        assert policy.lookback_turns == 5
        assert policy.owner_plugin == "policy_config"

    def test_multiple_gated_tools(self, tmp_path):
        """Parses multiple gated tools."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "read_before_write",
                    "prerequisite_tool": "readFile",
                    "gated_tools": ["writeFile", "updateFile", "replaceFile"],
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert len(policies) == 1
        assert policies[0].gated_tools == {"writeFile", "updateFile", "replaceFile"}

    def test_nudge_templates(self, tmp_path):
        """Parses nudge_templates correctly."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "test_policy",
                    "prerequisite_tool": "checkFirst",
                    "gated_tools": ["doAction"],
                    "nudge_templates": {
                        "minor": ["gentle", "Please call checkFirst first."],
                        "moderate": ["direct", "You MUST call checkFirst (violation #{count})."],
                        "severe": ["interrupt", "BLOCKED: doAction requires checkFirst."],
                    },
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert len(policies) == 1

        nudges = policies[0].nudge_templates
        assert PatternSeverity.MINOR in nudges
        assert nudges[PatternSeverity.MINOR] == (
            NudgeType.GENTLE_REMINDER,
            "Please call checkFirst first.",
        )
        assert PatternSeverity.MODERATE in nudges
        assert nudges[PatternSeverity.MODERATE] == (
            NudgeType.DIRECT_INSTRUCTION,
            "You MUST call checkFirst (violation #{count}).",
        )
        assert PatternSeverity.SEVERE in nudges
        assert nudges[PatternSeverity.SEVERE] == (
            NudgeType.INTERRUPT,
            "BLOCKED: doAction requires checkFirst.",
        )

    def test_custom_expected_action_template(self, tmp_path):
        """Parses custom expected_action_template."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "custom_template",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                    "expected_action_template": "Run {prerequisite_tool} before {tool_name} please",
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert policies[0].expected_action_template == "Run {prerequisite_tool} before {tool_name} please"

    def test_missing_required_field_skips(self, tmp_path):
        """Missing required fields skip the policy with a warning."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "incomplete",
                    # missing prerequisite_tool and gated_tools
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert len(policies) == 0
        assert len(warnings) == 1
        assert "'prerequisite_tool'" in warnings[0]

    def test_empty_gated_tools_skips(self, tmp_path):
        """Empty gated_tools list skips the policy."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "empty_gates",
                    "prerequisite_tool": "prep",
                    "gated_tools": [],
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert len(policies) == 0
        assert len(warnings) == 1

    def test_invalid_nudge_type_warns(self, tmp_path):
        """Invalid nudge type produces a warning but keeps the policy."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "bad_nudge",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                    "nudge_templates": {
                        "minor": ["nonexistent_type", "message"],
                    },
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert len(policies) == 1
        assert len(warnings) == 1
        assert "unknown nudge type" in warnings[0]
        # The policy was created, but the nudge template was skipped
        assert PatternSeverity.MINOR not in policies[0].nudge_templates

    def test_multiple_policies(self, tmp_path):
        """Parses multiple policies from the array."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "policy_a",
                    "prerequisite_tool": "toolA",
                    "gated_tools": ["gatedA"],
                },
                {
                    "policy_id": "policy_b",
                    "prerequisite_tool": "toolB",
                    "gated_tools": ["gatedB1", "gatedB2"],
                },
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert len(policies) == 2
        assert policies[0].policy_id == "policy_a"
        assert policies[1].policy_id == "policy_b"

    def test_non_array_warns(self, tmp_path):
        """Non-array prerequisite_policies produces a warning."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": "not_an_array"
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert len(policies) == 0
        assert len(warnings) == 1
        assert "must be an array" in warnings[0]

    def test_severity_thresholds_parsed(self, tmp_path):
        """Parses severity_thresholds into PatternSeverity -> int mapping."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "strict_policy",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                    "severity_thresholds": {
                        "minor": 0,
                        "moderate": 0,
                        "severe": 0,
                    },
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        assert len(policies) == 1
        thresholds = policies[0].severity_thresholds
        assert thresholds[PatternSeverity.MINOR] == 0
        assert thresholds[PatternSeverity.MODERATE] == 0
        assert thresholds[PatternSeverity.SEVERE] == 0

    def test_severity_thresholds_custom_escalation(self, tmp_path):
        """Custom thresholds allowing more violations before escalation."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "lenient_policy",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                    "severity_thresholds": {
                        "minor": 0,
                        "moderate": 3,
                        "severe": 6,
                    },
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        thresholds = policies[0].severity_thresholds
        assert thresholds[PatternSeverity.MINOR] == 0
        assert thresholds[PatternSeverity.MODERATE] == 3
        assert thresholds[PatternSeverity.SEVERE] == 6

    def test_severity_thresholds_partial(self, tmp_path):
        """Only some severities specified — others omitted."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "partial",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                    "severity_thresholds": {
                        "severe": 1,
                    },
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert warnings == []
        thresholds = policies[0].severity_thresholds
        assert PatternSeverity.SEVERE in thresholds
        assert PatternSeverity.MINOR not in thresholds

    def test_severity_thresholds_invalid_value_warns(self, tmp_path):
        """Non-integer threshold value produces a warning."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "bad_threshold",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                    "severity_thresholds": {
                        "minor": "not_a_number",
                    },
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert len(policies) == 1
        assert len(warnings) == 1
        assert "non-negative integer" in warnings[0]
        # The invalid entry was skipped
        assert PatternSeverity.MINOR not in policies[0].severity_thresholds

    def test_severity_thresholds_negative_warns(self, tmp_path):
        """Negative threshold value produces a warning."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "neg_threshold",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                    "severity_thresholds": {
                        "minor": -1,
                    },
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert len(policies) == 1
        assert len(warnings) == 1
        assert "non-negative integer" in warnings[0]

    def test_severity_thresholds_unknown_severity_warns(self, tmp_path):
        """Unknown severity key produces a warning."""
        config_file = tmp_path / "policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "unknown_sev",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                    "severity_thresholds": {
                        "critical": 0,
                    },
                }
            ]
        }))

        _, policies, warnings = load_policy_config(config_path=config_file)
        assert len(policies) == 1
        assert len(warnings) == 1
        assert "unknown severity" in warnings[0]


class TestLoadErrors:
    """Tests for error conditions."""

    def test_invalid_json(self, tmp_path):
        """Invalid JSON produces a warning."""
        config_file = tmp_path / "policies.json"
        config_file.write_text("not json {{{")

        config, policies, warnings = load_policy_config(config_path=config_file)
        assert config is None
        assert policies == []
        assert len(warnings) == 1
        assert "Invalid JSON" in warnings[0]

    def test_non_object_root(self, tmp_path):
        """Non-object root element produces a warning."""
        config_file = tmp_path / "policies.json"
        config_file.write_text('"just a string"')

        config, policies, warnings = load_policy_config(config_path=config_file)
        assert config is None
        assert policies == []
        assert len(warnings) == 1
        assert "root element must be a JSON object" in warnings[0]

    def test_missing_file_returns_error(self, tmp_path):
        """Explicitly passing a non-existent path returns a read error."""
        config_file = tmp_path / "nonexistent.json"

        config, policies, warnings = load_policy_config(config_path=config_file)
        assert config is None
        assert policies == []
        assert len(warnings) == 1
        assert "Cannot read" in warnings[0]


class TestGenerateDefaultConfig:
    """Tests for default config generation."""

    def test_generates_valid_json(self):
        """Generated config is valid JSON."""
        content = generate_default_config_safe()
        data = json.loads(content)
        assert "pattern_detection" in data
        assert "prerequisite_policies" in data

    def test_contains_example_policy(self):
        """Generated config contains the example plan_before_update policy."""
        content = generate_default_config_safe()
        data = json.loads(content)
        policies = data["prerequisite_policies"]
        assert len(policies) == 1
        assert policies[0]["policy_id"] == "example_plan_before_update"
        assert policies[0]["prerequisite_tool"] == "createPlan"
        assert "updateStep" in policies[0]["gated_tools"]

    def test_roundtrip_parse(self):
        """Generated default config parses without warnings."""
        import tempfile
        content = generate_default_config_safe()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(content)
            f.flush()
            config_path = Path(f.name)

        try:
            config, policies, warnings = load_policy_config(config_path=config_path)
            assert warnings == []
            assert config is not None
            assert len(policies) == 1
        finally:
            config_path.unlink()


class TestPluginPolicyIntegration:
    """Tests for plugin-level policy loading and reload."""

    def _make_config_file(self, tmp_path, data):
        """Helper to create a config file."""
        config_dir = tmp_path / ".jaato"
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "reliability-policies.json"
        config_file.write_text(json.dumps(data))
        return config_file

    def test_load_on_set_workspace_path(self, tmp_path):
        """Policies are loaded when workspace path is set."""
        self._make_config_file(tmp_path, {
            "prerequisite_policies": [
                {
                    "policy_id": "auto_loaded",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                }
            ]
        })

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)
        plugin.set_workspace_path(str(tmp_path))

        assert len(plugin._file_policies) == 1
        assert plugin._file_policies[0].policy_id == "auto_loaded"

    def test_reload_replaces_old_policies(self, tmp_path):
        """Reloading replaces previously loaded file policies."""
        config_file = self._make_config_file(tmp_path, {
            "prerequisite_policies": [
                {
                    "policy_id": "original",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["action"],
                }
            ]
        })

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)
        plugin.set_workspace_path(str(tmp_path))
        assert len(plugin._file_policies) == 1
        assert plugin._file_policies[0].policy_id == "original"

        # Update the file
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "replaced",
                    "prerequisite_tool": "newPrep",
                    "gated_tools": ["newAction"],
                }
            ]
        }))

        count, warnings = plugin.load_file_policies()
        assert count == 1
        assert warnings == []
        assert len(plugin._file_policies) == 1
        assert plugin._file_policies[0].policy_id == "replaced"

        # Old policy should be removed from detector
        detector = plugin.get_pattern_detector()
        assert "original" not in detector._prerequisite_policies
        assert "replaced" in detector._prerequisite_policies

    def test_reload_preserves_programmatic_policies(self, tmp_path):
        """Reloading file policies does not remove programmatic policies."""
        self._make_config_file(tmp_path, {
            "prerequisite_policies": [
                {
                    "policy_id": "from_file",
                    "prerequisite_tool": "filePrep",
                    "gated_tools": ["fileAction"],
                }
            ]
        })

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)

        # Register a programmatic policy
        prog_policy = PrerequisitePolicy(
            policy_id="programmatic",
            prerequisite_tool="progPrep",
            gated_tools={"progAction"},
        )
        plugin.register_prerequisite_policy(prog_policy)

        plugin.set_workspace_path(str(tmp_path))

        # Both should be present
        detector = plugin.get_pattern_detector()
        assert "programmatic" in detector._prerequisite_policies
        assert "from_file" in detector._prerequisite_policies

        # Reload — programmatic should survive
        plugin.load_file_policies()
        assert "programmatic" in detector._prerequisite_policies
        assert "from_file" in detector._prerequisite_policies

    def test_pattern_config_override_applied(self, tmp_path):
        """Pattern detection config from file is applied."""
        self._make_config_file(tmp_path, {
            "pattern_detection": {
                "repetitive_call_threshold": 7,
                "max_reads_before_action": 10,
            }
        })

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)
        plugin.set_workspace_path(str(tmp_path))

        detector = plugin.get_pattern_detector()
        assert detector._config.repetitive_call_threshold == 7
        assert detector._config.max_reads_before_action == 10

    def test_unregister_cleans_gated_tool_index(self, tmp_path):
        """Unregistering file policies cleans the gated tool reverse index."""
        self._make_config_file(tmp_path, {
            "prerequisite_policies": [
                {
                    "policy_id": "to_remove",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["gatedTool"],
                }
            ]
        })

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)
        plugin.set_workspace_path(str(tmp_path))

        detector = plugin.get_pattern_detector()
        assert "gatedTool" in detector._gated_tool_to_policies
        assert len(detector._gated_tool_to_policies["gatedTool"]) == 1

        # Remove file policies
        plugin._unregister_file_policies()
        # The list should be empty (key may still exist but list is empty)
        assert len(detector._gated_tool_to_policies.get("gatedTool", [])) == 0


class TestPoliciesCommand:
    """Tests for the 'reliability policies' command handler."""

    def _make_config_file(self, tmp_path, data):
        """Helper to create a config file."""
        config_dir = tmp_path / ".jaato"
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "reliability-policies.json"
        config_file.write_text(json.dumps(data))
        return config_file

    def test_policies_status_no_file(self, tmp_path):
        """Status shows 'not found' when no config file exists."""
        plugin = ReliabilityPlugin()
        plugin._workspace_path = str(tmp_path)
        result = plugin.handle_command("policies", "status")
        assert "(not found)" in result
        assert "Loaded policies: 0" in result

    def test_policies_status_with_policies(self, tmp_path):
        """Status shows loaded policies."""
        self._make_config_file(tmp_path, {
            "prerequisite_policies": [
                {
                    "policy_id": "plan_check",
                    "prerequisite_tool": "createPlan",
                    "gated_tools": ["updateStep"],
                    "lookback_turns": 5,
                }
            ]
        })

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)
        plugin.set_workspace_path(str(tmp_path))

        result = plugin.handle_command("policies", "status")
        assert "Loaded policies: 1" in result
        assert "plan_check" in result
        assert "createPlan" in result
        assert "updateStep" in result

    def test_policies_reload(self, tmp_path):
        """Reload command reloads policies from config file."""
        config_file = self._make_config_file(tmp_path, {
            "prerequisite_policies": [
                {
                    "policy_id": "v1",
                    "prerequisite_tool": "prep",
                    "gated_tools": ["act"],
                }
            ]
        })

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)
        plugin.set_workspace_path(str(tmp_path))
        assert plugin._file_policies[0].policy_id == "v1"

        # Update file
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "v2",
                    "prerequisite_tool": "newPrep",
                    "gated_tools": ["newAct"],
                }
            ]
        }))

        result = plugin.handle_command("policies", "reload")
        assert "Reloaded from" in result
        assert "Prerequisite policies loaded: 1" in result
        assert plugin._file_policies[0].policy_id == "v2"

    def test_policies_reload_no_file(self, tmp_path):
        """Reload command handles missing config file."""
        plugin = ReliabilityPlugin()
        plugin._workspace_path = str(tmp_path)

        result = plugin.handle_command("policies", "reload")
        assert "No reliability-policies.json found" in result

    def test_policies_path_with_file(self, tmp_path):
        """Path command shows active config file."""
        self._make_config_file(tmp_path, {})

        plugin = ReliabilityPlugin()
        plugin._workspace_path = str(tmp_path)

        result = plugin.handle_command("policies", "path")
        assert "Active config file:" in result

    def test_policies_path_no_file(self, tmp_path):
        """Path command shows expected locations when no file exists."""
        plugin = ReliabilityPlugin()
        plugin._workspace_path = str(tmp_path)

        result = plugin.handle_command("policies", "path")
        assert "No config file found" in result
        assert "reliability policies edit" in result

    def test_policies_unknown_subcommand(self, tmp_path):
        """Unknown subcommand shows usage."""
        plugin = ReliabilityPlugin()
        plugin._workspace_path = str(tmp_path)

        result = plugin.handle_command("policies", "bogus")
        assert "Unknown policies subcommand" in result
        assert "status|reload|edit|path" in result


class TestPoliciesCompletion:
    """Tests for command completion of the policies subcommand."""

    def test_level1_includes_policies(self):
        """Top-level completions include 'policies'."""
        plugin = ReliabilityPlugin()
        completions = plugin.get_command_completions(["reliability"])
        values = [c.value for c in completions]
        assert "policies" in values

    def test_level2_policies_subcommands(self):
        """Second-level completions for 'policies' include all subcommands."""
        plugin = ReliabilityPlugin()
        completions = plugin.get_command_completions(["reliability", "policies"])
        values = [c.value for c in completions]
        assert "status" in values
        assert "reload" in values
        assert "edit" in values
        assert "path" in values


class TestGetSeverity:
    """Tests for PrerequisitePolicy.get_severity() method."""

    def test_default_thresholds(self):
        """With empty severity_thresholds, uses built-in defaults."""
        policy = PrerequisitePolicy(
            policy_id="test",
            prerequisite_tool="prep",
            gated_tools={"action"},
        )
        assert policy.get_severity(0) == PatternSeverity.MINOR
        assert policy.get_severity(1) == PatternSeverity.MODERATE
        assert policy.get_severity(2) == PatternSeverity.SEVERE
        assert policy.get_severity(10) == PatternSeverity.SEVERE

    def test_immediate_severe(self):
        """All thresholds at 0 means SEVERE from the first violation."""
        policy = PrerequisitePolicy(
            policy_id="strict",
            prerequisite_tool="prep",
            gated_tools={"action"},
            severity_thresholds={
                PatternSeverity.MINOR: 0,
                PatternSeverity.MODERATE: 0,
                PatternSeverity.SEVERE: 0,
            },
        )
        assert policy.get_severity(0) == PatternSeverity.SEVERE
        assert policy.get_severity(5) == PatternSeverity.SEVERE

    def test_lenient_thresholds(self):
        """High thresholds allow many violations before escalation."""
        policy = PrerequisitePolicy(
            policy_id="lenient",
            prerequisite_tool="prep",
            gated_tools={"action"},
            severity_thresholds={
                PatternSeverity.MINOR: 0,
                PatternSeverity.MODERATE: 3,
                PatternSeverity.SEVERE: 6,
            },
        )
        assert policy.get_severity(0) == PatternSeverity.MINOR
        assert policy.get_severity(2) == PatternSeverity.MINOR
        assert policy.get_severity(3) == PatternSeverity.MODERATE
        assert policy.get_severity(5) == PatternSeverity.MODERATE
        assert policy.get_severity(6) == PatternSeverity.SEVERE
        assert policy.get_severity(100) == PatternSeverity.SEVERE

    def test_partial_thresholds_only_severe(self):
        """Only SEVERE specified — falls back to MINOR for lower counts."""
        policy = PrerequisitePolicy(
            policy_id="partial",
            prerequisite_tool="prep",
            gated_tools={"action"},
            severity_thresholds={
                PatternSeverity.SEVERE: 1,
            },
        )
        # 0 violations: SEVERE threshold (1) not met, MODERATE not defined,
        # MINOR not defined → falls through to default MINOR
        assert policy.get_severity(0) == PatternSeverity.MINOR
        assert policy.get_severity(1) == PatternSeverity.SEVERE
        assert policy.get_severity(5) == PatternSeverity.SEVERE


class TestSeverityThresholdsDetectorIntegration:
    """Tests that per-policy severity thresholds flow through to the detector."""

    def test_custom_thresholds_affect_detection(self, tmp_path):
        """Detector uses policy's severity_thresholds, not hardcoded values."""
        # Create a config with immediate-severe policy
        config_dir = tmp_path / ".jaato"
        config_dir.mkdir()
        config_file = config_dir / "reliability-policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "immediate_block",
                    "prerequisite_tool": "createPlan",
                    "gated_tools": ["updateStep"],
                    "severity_thresholds": {
                        "minor": 0,
                        "moderate": 0,
                        "severe": 0,
                    },
                }
            ]
        }))

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)

        detected = []
        plugin.set_pattern_hook(lambda p: detected.append(p))
        plugin.set_workspace_path(str(tmp_path))

        # First call to updateStep without createPlan
        plugin.on_turn_start(1)
        plugin.on_tool_called("updateStep", {})

        assert len(detected) == 1
        # Because all thresholds are 0, the very first violation is SEVERE
        assert detected[0].severity == PatternSeverity.SEVERE
        assert detected[0].policy_id == "immediate_block"

    def test_lenient_thresholds_delay_escalation(self, tmp_path):
        """Lenient thresholds keep severity low for early violations."""
        config_dir = tmp_path / ".jaato"
        config_dir.mkdir()
        config_file = config_dir / "reliability-policies.json"
        config_file.write_text(json.dumps({
            "prerequisite_policies": [
                {
                    "policy_id": "lenient",
                    "prerequisite_tool": "createPlan",
                    "gated_tools": ["updateStep"],
                    "severity_thresholds": {
                        "minor": 0,
                        "moderate": 5,
                        "severe": 10,
                    },
                }
            ]
        }))

        plugin = ReliabilityPlugin()
        plugin.enable_pattern_detection(True)

        detected = []
        plugin.set_pattern_hook(lambda p: detected.append(p))
        plugin.set_workspace_path(str(tmp_path))

        # Simulate multiple violations across turns
        for turn in range(1, 5):
            plugin.on_turn_start(turn)
            plugin.on_tool_called("updateStep", {})

        # All 4 violations should be MINOR (threshold for moderate is 5)
        assert len(detected) == 4
        for p in detected:
            assert p.severity == PatternSeverity.MINOR
