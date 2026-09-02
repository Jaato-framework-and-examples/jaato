"""Tests for validate_profile() in subagent config."""

import pytest

from shared.plugins.subagent.config import validate_profile


class TestValidateProfile:
    """Tests for the subagent profile validator."""

    def test_valid_minimal_profile(self):
        data = {"name": "test-agent", "description": "A test agent"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True
        assert errors == []
        assert warnings == []

    def test_valid_full_profile(self):
        data = {
            "name": "skill-code-001",
            "description": "[ADD Flow] Add circuit breaker",
            "plugins": ["cli", "references", "template"],
            "plugin_configs": {
                "references": {"preselected": ["eri-001"]},
                "lsp": {"config_path": "/workspace/.lsp.json"},
            },
            "system_instructions": "Read the knowledge base first.",
            "model": "gemini-2.5-flash",
            "provider": "google_genai",
            "max_turns": 15,
            "icon": ["╔══╗", "║CB║", "╚══╝"],
            "icon_name": "circuit_breaker",
            "gc": {
                "type": "hybrid",
                "threshold_percent": 80.0,
                "target_percent": 60.0,
                "pressure_percent": 90.0,
                "preserve_recent_turns": 5,
                "notify_on_gc": True,
                "summarize_middle_turns": 10,
            },
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True
        assert errors == []

    def test_missing_name(self):
        data = {"description": "No name"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'name' is required" in e for e in errors)

    def test_missing_description(self):
        data = {"name": "no-desc"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'description' is required" in e for e in errors)

    def test_plugins_not_list(self):
        data = {"name": "test", "description": "test", "plugins": "cli"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'plugins' must be an array" in e for e in errors)

    def test_plugins_not_strings(self):
        data = {"name": "test", "description": "test", "plugins": ["cli", 42]}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'plugins' must contain only strings" in e for e in errors)

    def test_plugin_configs_not_dict(self):
        data = {"name": "test", "description": "test", "plugin_configs": "bad"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'plugin_configs' must be an object" in e for e in errors)

    def test_plugin_configs_value_not_dict(self):
        data = {"name": "test", "description": "test", "plugin_configs": {"cli": "bad"}}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("plugin_configs['cli'] must be an object" in e for e in errors)

    def test_max_turns_not_int(self):
        data = {"name": "test", "description": "test", "max_turns": "ten"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'max_turns' must be an integer" in e for e in errors)

    def test_max_turns_zero(self):
        data = {"name": "test", "description": "test", "max_turns": 0}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'max_turns' must be a positive integer" in e for e in errors)

    def test_max_turns_negative(self):
        data = {"name": "test", "description": "test", "max_turns": -1}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'max_turns' must be a positive integer" in e for e in errors)

    def test_max_turns_bool_rejected(self):
        """bool is a subclass of int in Python, but should be rejected."""
        data = {"name": "test", "description": "test", "max_turns": True}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'max_turns' must be an integer" in e for e in errors)

    def test_model_not_string(self):
        data = {"name": "test", "description": "test", "model": 123}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'model' must be a string" in e for e in errors)

    def test_provider_not_string(self):
        data = {"name": "test", "description": "test", "provider": 123}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'provider' must be a string" in e for e in errors)

    # test_icon_not_list / test_icon_wrong_length / test_icon_not_strings /
    # test_icon_name_not_string were removed with the fields they validated:
    # commit 635b00ec "Remove icon and icon_name from profiles".  Kept as a
    # note so the gap in coverage reads as deliberate rather than lost.

    def test_gc_not_dict(self):
        data = {"name": "test", "description": "test", "gc": "truncate"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'gc' must be an object" in e for e in errors)

    def test_gc_invalid_type(self):
        data = {"name": "test", "description": "test", "gc": {"type": "magic"}}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("gc.type 'magic' is invalid" in e for e in errors)

    def test_gc_valid_types(self):
        for gc_type in ("truncate", "summarize", "hybrid", "budget"):
            data = {"name": "test", "description": "test", "gc": {"type": gc_type}}
            is_valid, errors, warnings = validate_profile(data)
            assert is_valid is True, f"gc.type '{gc_type}' should be valid"

    def test_gc_threshold_out_of_range(self):
        data = {
            "name": "test",
            "description": "test",
            "gc": {"threshold_percent": 150},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("gc.threshold_percent must be between 0 and 100" in e for e in errors)

    def test_gc_threshold_negative(self):
        data = {
            "name": "test",
            "description": "test",
            "gc": {"threshold_percent": -5},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("gc.threshold_percent must be between 0 and 100" in e for e in errors)

    def test_gc_threshold_not_number(self):
        data = {
            "name": "test",
            "description": "test",
            "gc": {"threshold_percent": "high"},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("gc.threshold_percent must be a number" in e for e in errors)

    def test_gc_preserve_recent_turns_negative(self):
        data = {
            "name": "test",
            "description": "test",
            "gc": {"preserve_recent_turns": -1},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("gc.preserve_recent_turns must be non-negative" in e for e in errors)

    def test_gc_max_turns_zero(self):
        data = {
            "name": "test",
            "description": "test",
            "gc": {"max_turns": 0},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("gc.max_turns must be a positive integer" in e for e in errors)

    def test_not_a_dict(self):
        is_valid, errors, warnings = validate_profile("not a dict")
        assert is_valid is False
        assert any("JSON object" in e for e in errors)

    def test_env_valid_dict(self):
        """Valid env dict is accepted."""
        data = {
            "name": "test",
            "description": "test",
            "env": {"DB_HOST": "localhost", "DB_PORT": "5432"},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True
        assert errors == []

    def test_env_null_accepted(self):
        """Null env is accepted."""
        data = {"name": "test", "description": "test", "env": None}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True

    def test_env_not_dict(self):
        """Non-dict env is rejected."""
        data = {"name": "test", "description": "test", "env": "DB_HOST=localhost"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'env' must be an object" in e for e in errors)

    def test_env_non_string_value(self):
        """Non-string env values are rejected."""
        data = {"name": "test", "description": "test", "env": {"PORT": 5432}}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("env['PORT'] must be a string" in e for e in errors)

    def test_env_with_variable_refs(self):
        """Env values with ${VAR} references are valid strings."""
        data = {
            "name": "test",
            "description": "test",
            "env": {"API_URL": "${STAGING_HOST}/api"},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True

    def test_env_with_secret_uris(self):
        """Env values with vault:// URIs are valid strings."""
        data = {
            "name": "test",
            "description": "test",
            "env": {"DB_PASSWORD": "vault://secret/myapp#db_password"},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True

    def test_null_optional_fields(self):
        """Null values for optional fields should be accepted."""
        data = {
            "name": "test",
            "description": "test",
            "model": None,
            "provider": None,
            "icon": None,
            "icon_name": None,
            "gc": None,
            "runtime_limits": None,
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True
        assert errors == []

    def test_runtime_limits_not_dict(self):
        data = {"name": "test", "description": "test", "runtime_limits": "512mb"}
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("'runtime_limits' must be an object" in e for e in errors)

    def test_runtime_limits_valid(self):
        data = {
            "name": "test", "description": "test",
            "runtime_limits": {
                "memory_max_mb": 512,
                "pids_max": 256,
                "cpu_weight": 200,
                "tool_timeout_seconds": 60,
                "max_output_bytes": 1048576,
            },
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True, f"unexpected errors: {errors}"

    def test_runtime_limits_invalid_memory(self):
        data = {
            "name": "test", "description": "test",
            "runtime_limits": {"memory_max_mb": -1},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("memory_max_mb" in e for e in errors)

    def test_runtime_limits_invalid_cpu_weight(self):
        # cpu_weight bounds are [1, 10000]
        data = {
            "name": "test", "description": "test",
            "runtime_limits": {"cpu_weight": 99999},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is False
        assert any("cpu_weight" in e for e in errors)

    def test_runtime_limits_app_layer_only_is_valid(self):
        # tool_timeout_seconds and max_output_bytes alone (no kernel
        # limits) is a legitimate config — the cgroup is skipped, but
        # the limits still apply at the app layer.
        data = {
            "name": "test", "description": "test",
            "runtime_limits": {"tool_timeout_seconds": 30},
        }
        is_valid, errors, warnings = validate_profile(data)
        assert is_valid is True
