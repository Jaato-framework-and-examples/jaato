"""Tests for JaatoRuntime - shared environment for the jaato framework."""

import pytest
from unittest.mock import MagicMock, patch

from ..jaato_runtime import JaatoRuntime


class TestJaatoRuntimeInitialization:
    """Tests for JaatoRuntime initialization."""

    def test_init_default_provider(self):
        """Test default provider name."""
        runtime = JaatoRuntime()
        assert runtime.provider_name == "google_genai"

    def test_init_custom_provider(self):
        """Test custom provider name."""
        runtime = JaatoRuntime(provider_name="anthropic")
        assert runtime.provider_name == "anthropic"

    def test_not_connected_initially(self):
        """Test that runtime is not connected initially."""
        runtime = JaatoRuntime()
        assert not runtime.is_connected

    def test_properties_none_initially(self):
        """Test that properties are None initially."""
        runtime = JaatoRuntime()
        assert runtime.project is None
        assert runtime.location is None
        assert runtime.registry is None
        assert runtime.permission_plugin is None
        assert runtime.ledger is None


class TestJaatoRuntimeConnect:
    """Tests for JaatoRuntime.connect()."""

    def test_connect_sets_project_and_location(self):
        """Test that connect sets project and location."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        assert runtime.project == "my-project"
        assert runtime.location == "us-central1"
        assert runtime.is_connected

    def test_connect_multiple_times(self):
        """Test that connect can be called multiple times."""
        runtime = JaatoRuntime()
        runtime.connect("project-1", "us-central1")
        runtime.connect("project-2", "eu-west1")

        assert runtime.project == "project-2"
        assert runtime.location == "eu-west1"


class TestJaatoRuntimeConfigurePlugins:
    """Tests for JaatoRuntime.configure_plugins()."""

    def test_configure_plugins_stores_registry(self):
        """Test that configure_plugins stores the registry."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = []
        mock_registry.get_exposed_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        assert runtime.registry == mock_registry

    def test_configure_plugins_stores_permission_plugin(self):
        """Test that configure_plugins stores the permission plugin."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = []
        mock_registry.get_exposed_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        mock_permission = MagicMock()
        mock_permission.get_tool_schemas.return_value = []
        mock_permission.get_executors.return_value = {}
        mock_permission.get_system_instructions.return_value = None

        runtime.configure_plugins(mock_registry, permission_plugin=mock_permission)

        assert runtime.permission_plugin == mock_permission

    def test_configure_plugins_caches_tool_schemas(self):
        """Test that configure_plugins caches tool schemas."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_schema = MagicMock()
        mock_schema.name = "test_tool"

        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = [mock_schema]
        mock_registry.get_exposed_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        schemas = runtime.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "test_tool"


class TestJaatoRuntimeCreateSession:
    """Tests for JaatoRuntime.create_session()."""

    def test_create_session_requires_connection(self):
        """Test that create_session requires runtime to be connected."""
        runtime = JaatoRuntime()

        with pytest.raises(RuntimeError, match="not connected"):
            runtime.create_session("gemini-2.5-flash")

    def test_create_session_requires_plugins(self):
        """Test that create_session requires plugins to be configured."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        with pytest.raises(RuntimeError, match="not configured"):
            runtime.create_session("gemini-2.5-flash")

    @patch('shared.jaato_runtime.load_provider')
    def test_create_session_returns_session(self, mock_load_provider):
        """Test that create_session returns a JaatoSession."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        # Setup mock registry
        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = []
        mock_registry.get_exposed_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        # Setup mock provider
        mock_provider = MagicMock()
        mock_load_provider.return_value = mock_provider

        runtime.configure_plugins(mock_registry)
        session = runtime.create_session("gemini-2.5-flash")

        assert session is not None
        assert session.model_name == "gemini-2.5-flash"
        assert session.runtime == runtime


class TestJaatoRuntimeCreateProvider:
    """Tests for JaatoRuntime.create_provider()."""

    def test_create_provider_requires_connection(self):
        """Test that create_provider requires runtime to be connected."""
        runtime = JaatoRuntime()

        with pytest.raises(RuntimeError, match="not connected"):
            runtime.create_provider("gemini-2.5-flash")

    @patch('shared.jaato_runtime.load_provider')
    def test_create_provider_returns_provider(self, mock_load_provider):
        """Test that create_provider returns a provider instance."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_provider = MagicMock()
        mock_load_provider.return_value = mock_provider

        provider = runtime.create_provider("gemini-2.5-flash")

        assert provider == mock_provider
        mock_provider.connect.assert_called_once_with("gemini-2.5-flash")


class TestJaatoRuntimeGetToolSchemas:
    """Tests for JaatoRuntime.get_tool_schemas()."""

    def test_get_tool_schemas_empty_without_registry(self):
        """Test that get_tool_schemas returns empty list without registry."""
        runtime = JaatoRuntime()
        assert runtime.get_tool_schemas() == []

    def test_get_tool_schemas_returns_cached(self):
        """Test that get_tool_schemas returns cached schemas."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_schema = MagicMock()
        mock_schema.name = "cached_tool"

        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = [mock_schema]
        mock_registry.get_exposed_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        schemas = runtime.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "cached_tool"

    def test_get_tool_schemas_filtered_by_plugin_names(self):
        """Test that get_tool_schemas can filter by plugin names."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_schema_cli = MagicMock()
        mock_schema_cli.name = "cli_tool"
        mock_schema_mcp = MagicMock()
        mock_schema_mcp.name = "mcp_tool"

        mock_cli_plugin = MagicMock()
        mock_cli_plugin.get_tool_schemas.return_value = [mock_schema_cli]
        mock_mcp_plugin = MagicMock()
        mock_mcp_plugin.get_tool_schemas.return_value = [mock_schema_mcp]

        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = [mock_schema_cli, mock_schema_mcp]
        mock_registry.get_exposed_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.side_effect = lambda name: {
            'cli': mock_cli_plugin,
            'mcp': mock_mcp_plugin,
            'subagent': None,
            'background': None
        }.get(name)

        runtime.configure_plugins(mock_registry)

        # Filter by plugin names
        schemas = runtime.get_tool_schemas(plugin_names=['cli'])
        assert len(schemas) == 1
        assert schemas[0].name == "cli_tool"


class TestJaatoRuntimeGetExecutors:
    """Tests for JaatoRuntime.get_executors()."""

    def test_get_executors_empty_without_registry(self):
        """Test that get_executors returns empty dict without registry."""
        runtime = JaatoRuntime()
        assert runtime.get_executors() == {}

    def test_get_executors_returns_cached(self):
        """Test that get_executors returns cached executors."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        def mock_executor(args):
            return "result"

        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = []
        mock_registry.get_exposed_executors.return_value = {"test_tool": mock_executor}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        executors = runtime.get_executors()
        assert "test_tool" in executors
        assert executors["test_tool"] == mock_executor


class TestJaatoRuntimeGetSystemInstructions:
    """Tests for JaatoRuntime.get_system_instructions()."""

    def test_get_system_instructions_none_without_registry(self):
        """Test that get_system_instructions returns None without registry."""
        runtime = JaatoRuntime()
        assert runtime.get_system_instructions() is None

    def test_get_system_instructions_returns_cached(self):
        """Test that get_system_instructions returns cached instructions."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = []
        mock_registry.get_exposed_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = "Be helpful."
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        instructions = runtime.get_system_instructions()
        assert instructions == "Be helpful."

    def test_get_system_instructions_with_additional(self):
        """Test that get_system_instructions can add additional instructions."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_exposed_tool_schemas.return_value = []
        mock_registry.get_exposed_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = "Be helpful."
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        instructions = runtime.get_system_instructions(additional="Be concise.")
        assert "Be concise." in instructions
        assert "Be helpful." in instructions
