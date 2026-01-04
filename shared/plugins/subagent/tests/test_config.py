"""Tests for subagent config module - variable expansion."""

import os
import pytest
from unittest.mock import patch

from ..config import (
    expand_variables,
    _expand_string,
    _find_workspace_root,
    expand_plugin_configs,
)


class TestExpandString:
    """Tests for _expand_string helper."""

    def test_no_variables(self):
        """Strings without ${} pass through unchanged."""
        assert _expand_string("hello world", {}) == "hello world"

    def test_simple_variable(self):
        """Simple ${VAR} expansion works."""
        assert _expand_string("${name}", {"name": "Alice"}) == "Alice"

    def test_variable_in_path(self):
        """Variables expand within paths."""
        result = _expand_string("${home}/projects", {"home": "/home/user"})
        assert result == "/home/user/projects"

    def test_multiple_variables(self):
        """Multiple variables in one string."""
        result = _expand_string(
            "${prefix}/${name}/${suffix}",
            {"prefix": "a", "name": "b", "suffix": "c"}
        )
        assert result == "a/b/c"

    def test_env_var_fallback(self):
        """Falls back to environment variables."""
        with patch.dict(os.environ, {"TEST_VAR": "from_env"}):
            result = _expand_string("${TEST_VAR}", {})
            assert result == "from_env"

    def test_context_takes_precedence(self):
        """Context values take precedence over env vars."""
        with patch.dict(os.environ, {"HOME": "/original/home"}):
            result = _expand_string("${HOME}", {"HOME": "/custom/home"})
            assert result == "/custom/home"

    def test_undefined_variable_kept(self):
        """Undefined variables are kept as-is."""
        result = _expand_string("${UNDEFINED_VAR}", {})
        assert result == "${UNDEFINED_VAR}"


class TestExpandVariables:
    """Tests for expand_variables function."""

    def test_string_expansion(self):
        """Strings are expanded."""
        result = expand_variables("${cwd}/file.txt")
        assert "${cwd}" not in result
        assert "file.txt" in result

    def test_dict_expansion(self):
        """Dict values are recursively expanded."""
        input_dict = {
            "path": "${cwd}/config",
            "nested": {
                "file": "${HOME}/data"
            }
        }
        result = expand_variables(input_dict)
        assert "${cwd}" not in result["path"]
        assert "${HOME}" not in result["nested"]["file"]

    def test_list_expansion(self):
        """List items are expanded."""
        input_list = ["${cwd}/a", "${cwd}/b"]
        result = expand_variables(input_list)
        for item in result:
            assert "${cwd}" not in item

    def test_non_string_passthrough(self):
        """Non-string types pass through unchanged."""
        assert expand_variables(42) == 42
        assert expand_variables(3.14) == 3.14
        assert expand_variables(True) is True
        assert expand_variables(None) is None

    def test_default_context_includes_cwd(self):
        """Default context includes current working directory."""
        result = expand_variables("${cwd}")
        assert result == os.getcwd()

    def test_custom_context(self):
        """Custom context variables are used."""
        result = expand_variables(
            "${projectPath}/.config",
            {"projectPath": "/my/project"}
        )
        assert result == "/my/project/.config"

    def test_workspace_root_expansion(self):
        """${workspaceRoot} expands to detected workspace."""
        result = expand_variables("${workspaceRoot}")
        # Should return some path (either detected .git root or cwd)
        assert result and os.path.isabs(result)


class TestExpandPluginConfigs:
    """Tests for expand_plugin_configs function."""

    def test_empty_configs(self):
        """Empty config dict returns empty."""
        assert expand_plugin_configs({}) == {}

    def test_plugin_path_expansion(self):
        """Plugin config paths are expanded."""
        configs = {
            "lsp": {"config_path": "${cwd}/.lsp.json"},
            "mcp": {"config_path": "${cwd}/.mcp.json"}
        }
        result = expand_plugin_configs(configs)

        cwd = os.getcwd()
        assert result["lsp"]["config_path"] == f"{cwd}/.lsp.json"
        assert result["mcp"]["config_path"] == f"{cwd}/.mcp.json"

    def test_with_custom_context(self):
        """Custom context passed to plugin configs."""
        configs = {
            "references": {"base_path": "${projectPath}/docs"}
        }
        result = expand_plugin_configs(
            configs,
            {"projectPath": "/app"}
        )
        assert result["references"]["base_path"] == "/app/docs"

    def test_nested_config_values(self):
        """Nested config values are expanded."""
        configs = {
            "myPlugin": {
                "paths": {
                    "input": "${HOME}/input",
                    "output": "${HOME}/output"
                }
            }
        }
        result = expand_plugin_configs(configs)
        home = os.environ.get("HOME", "")
        assert result["myPlugin"]["paths"]["input"] == f"{home}/input"
        assert result["myPlugin"]["paths"]["output"] == f"{home}/output"


class TestFindWorkspaceRoot:
    """Tests for _find_workspace_root helper."""

    def test_returns_path(self):
        """Returns a valid path."""
        result = _find_workspace_root()
        assert result
        assert os.path.isabs(result)

    def test_returns_string(self):
        """Returns string type."""
        result = _find_workspace_root()
        assert isinstance(result, str)

    def test_override_takes_precedence(self):
        """Explicit override parameter takes precedence over everything."""
        result = _find_workspace_root(override="/custom/override")
        assert result == "/custom/override"

    def test_env_var_takes_precedence_over_detection(self):
        """JAATO_WORKSPACE_ROOT env var takes precedence over auto-detection."""
        with patch.dict(os.environ, {"JAATO_WORKSPACE_ROOT": "/env/workspace"}):
            result = _find_workspace_root()
            assert result == "/env/workspace"

    def test_override_takes_precedence_over_env_var(self):
        """Override parameter takes precedence over env var."""
        with patch.dict(os.environ, {"JAATO_WORKSPACE_ROOT": "/env/workspace"}):
            result = _find_workspace_root(override="/explicit/override")
            assert result == "/explicit/override"

    def test_dotenv_workspaceRoot_is_used(self):
        """workspaceRoot env var (from .env) is used when JAATO_WORKSPACE_ROOT not set."""
        # Clear JAATO_WORKSPACE_ROOT, set workspaceRoot (simulating .env load)
        with patch.dict(os.environ, {"workspaceRoot": "/from/dotenv"}, clear=False):
            # Make sure JAATO_WORKSPACE_ROOT is not set
            os.environ.pop("JAATO_WORKSPACE_ROOT", None)
            result = _find_workspace_root()
            assert result == "/from/dotenv"

    def test_jaato_env_takes_precedence_over_dotenv(self):
        """JAATO_WORKSPACE_ROOT takes precedence over workspaceRoot."""
        with patch.dict(os.environ, {
            "JAATO_WORKSPACE_ROOT": "/jaato/root",
            "workspaceRoot": "/dotenv/root"
        }):
            result = _find_workspace_root()
            assert result == "/jaato/root"

    def test_relative_path_dot_is_resolved(self):
        """Relative path '.' is resolved to absolute cwd."""
        result = _find_workspace_root(override=".")
        assert result == os.getcwd()
        assert os.path.isabs(result)

    def test_relative_path_subdir_is_resolved(self):
        """Relative path like './subdir' is resolved."""
        result = _find_workspace_root(override="./subdir")
        expected = os.path.join(os.getcwd(), "subdir")
        assert result == expected
        assert os.path.isabs(result)

    def test_dotenv_relative_path_is_resolved(self):
        """workspaceRoot=. from .env is resolved to cwd."""
        with patch.dict(os.environ, {"workspaceRoot": "."}, clear=False):
            os.environ.pop("JAATO_WORKSPACE_ROOT", None)
            result = _find_workspace_root()
            assert result == os.getcwd()


class TestWorkspaceRootOverride:
    """Tests for workspace_root_override in expand functions."""

    def test_expand_variables_with_override(self):
        """expand_variables uses workspace_root_override for ${workspaceRoot}."""
        result = expand_variables(
            "${workspaceRoot}/config.json",
            workspace_root_override="/my/workspace"
        )
        assert result == "/my/workspace/config.json"

    def test_expand_variables_nested_with_override(self):
        """Nested dicts use workspace_root_override."""
        input_data = {
            "path": "${workspaceRoot}/templates",
            "nested": {
                "file": "${workspaceRoot}/.jaato/config.json"
            }
        }
        result = expand_variables(input_data, workspace_root_override="/project")
        assert result["path"] == "/project/templates"
        assert result["nested"]["file"] == "/project/.jaato/config.json"

    def test_expand_plugin_configs_with_override(self):
        """expand_plugin_configs uses workspace_root_override."""
        configs = {
            "template": {"base_path": "${workspaceRoot}"},
            "lsp": {"config_path": "${workspaceRoot}/.lsp.json"}
        }
        result = expand_plugin_configs(
            configs,
            workspace_root_override="/custom/workspace"
        )
        assert result["template"]["base_path"] == "/custom/workspace"
        assert result["lsp"]["config_path"] == "/custom/workspace/.lsp.json"

    def test_context_overrides_workspace_root(self):
        """Context workspaceRoot takes precedence over auto-detection."""
        # If context explicitly provides workspaceRoot, it should be used
        result = expand_variables(
            "${workspaceRoot}/file",
            context={"workspaceRoot": "/from/context"}
        )
        assert result == "/from/context/file"
