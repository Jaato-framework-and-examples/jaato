"""Tests for JaatoSession - per-agent conversation session."""

import pytest
from unittest.mock import MagicMock, patch

from ..jaato_session import JaatoSession


class TestJaatoSessionInitialization:
    """Tests for JaatoSession initialization."""

    def test_init_stores_runtime_and_model(self):
        """Test that __init__ stores runtime and model."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        assert session.runtime == mock_runtime
        assert session.model_name == "gemini-2.5-flash"

    def test_not_configured_initially(self):
        """Test that session is not configured initially."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        assert not session.is_configured

    def test_default_agent_context(self):
        """Test default agent context is main."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        assert session._agent_type == "main"
        assert session._agent_name is None


class TestJaatoSessionSetAgentContext:
    """Tests for JaatoSession.set_agent_context()."""

    def test_set_agent_context_updates_type(self):
        """Test that set_agent_context updates agent type."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        session.set_agent_context(agent_type="subagent", agent_name="researcher")

        assert session._agent_type == "subagent"
        assert session._agent_name == "researcher"


class TestJaatoSessionConfigure:
    """Tests for JaatoSession.configure()."""

    def test_configure_creates_provider(self):
        """Test that configure creates a provider."""
        mock_runtime = MagicMock()
        mock_provider = MagicMock()
        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure()

        mock_runtime.create_provider.assert_called_once_with("gemini-2.5-flash")
        assert session.is_configured

    def test_configure_with_tools_subset(self):
        """Test that configure can use a tool subset."""
        mock_runtime = MagicMock()
        mock_provider = MagicMock()
        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        mock_schema = MagicMock()
        mock_schema.name = "cli_tool"
        mock_runtime.get_tool_schemas.return_value = [mock_schema]

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure(tools=["cli"])

        mock_runtime.get_tool_schemas.assert_called_with(["cli"])

    def test_configure_with_system_instructions(self):
        """Test that configure can add system instructions."""
        mock_runtime = MagicMock()
        mock_provider = MagicMock()
        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = "Combined instructions"
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure(system_instructions="Be a researcher.")

        mock_runtime.get_system_instructions.assert_called_with(
            plugin_names=None,
            additional="Be a researcher."
        )


class TestJaatoSessionSendMessage:
    """Tests for JaatoSession.send_message()."""

    def test_send_message_requires_configuration(self):
        """Test that send_message requires session to be configured."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        with pytest.raises(RuntimeError, match="not configured"):
            session.send_message("Hello")

    def test_send_message_returns_response(self):
        """Test that send_message returns response text."""
        mock_runtime = MagicMock()
        mock_provider = MagicMock()

        # Setup provider response
        mock_response = MagicMock()
        mock_response.text = "Hello back!"
        mock_response.function_calls = []
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_provider.send_message.return_value = mock_response

        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = MagicMock()
        mock_runtime.registry.enrich_prompt.return_value = MagicMock(prompt="Hello")
        mock_runtime.permission_plugin = None
        mock_runtime.ledger = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure()

        response = session.send_message("Hello")

        assert response == "Hello back!"


class TestJaatoSessionGetHistory:
    """Tests for JaatoSession.get_history()."""

    def test_get_history_empty_without_provider(self):
        """Test that get_history returns empty list without provider."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        assert session.get_history() == []

    def test_get_history_delegates_to_provider(self):
        """Test that get_history delegates to provider."""
        mock_runtime = MagicMock()
        mock_provider = MagicMock()
        mock_provider.get_history.return_value = ["msg1", "msg2"]

        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure()

        history = session.get_history()
        assert history == ["msg1", "msg2"]


class TestJaatoSessionGetTurnAccounting:
    """Tests for JaatoSession.get_turn_accounting()."""

    def test_get_turn_accounting_empty_initially(self):
        """Test that turn accounting is empty initially."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        assert session.get_turn_accounting() == []


class TestJaatoSessionGetContextUsage:
    """Tests for JaatoSession.get_context_usage()."""

    def test_get_context_usage_returns_dict(self):
        """Test that get_context_usage returns a dict."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        usage = session.get_context_usage()

        assert isinstance(usage, dict)
        assert "model" in usage
        assert "context_limit" in usage
        assert "total_tokens" in usage


class TestJaatoSessionResetSession:
    """Tests for JaatoSession.reset_session()."""

    def test_reset_session_clears_turn_accounting(self):
        """Test that reset_session clears turn accounting."""
        mock_runtime = MagicMock()
        mock_provider = MagicMock()
        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure()

        # Add some turn accounting
        session._turn_accounting = [{"tokens": 100}]

        session.reset_session()

        assert session._turn_accounting == []


class TestJaatoSessionGCPlugin:
    """Tests for JaatoSession GC plugin integration."""

    def test_set_gc_plugin_stores_plugin(self):
        """Test that set_gc_plugin stores the plugin."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        mock_gc = MagicMock()
        mock_config = MagicMock()

        session.set_gc_plugin(mock_gc, mock_config)

        assert session._gc_plugin == mock_gc
        assert session._gc_config == mock_config

    def test_remove_gc_plugin_clears_plugin(self):
        """Test that remove_gc_plugin clears the plugin."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        mock_gc = MagicMock()
        session.set_gc_plugin(mock_gc)

        session.remove_gc_plugin()

        assert session._gc_plugin is None
        mock_gc.shutdown.assert_called_once()

    def test_manual_gc_requires_plugin(self):
        """Test that manual_gc requires a GC plugin."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        with pytest.raises(RuntimeError, match="No GC plugin"):
            session.manual_gc()


class TestJaatoSessionPluginIntegration:
    """Tests for JaatoSession session plugin integration."""

    def test_set_session_plugin_stores_plugin(self):
        """Test that set_session_plugin stores the plugin."""
        mock_runtime = MagicMock()
        mock_provider = MagicMock()
        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure()

        mock_session_plugin = MagicMock()
        mock_session_plugin.get_user_commands.return_value = []
        mock_session_plugin.get_executors.return_value = {}
        mock_session_plugin.get_tool_schemas.return_value = []

        mock_config = MagicMock()
        mock_config.auto_resume_last = False

        session.set_session_plugin(mock_session_plugin, mock_config)

        assert session._session_plugin == mock_session_plugin
        assert session._session_config == mock_config


class TestJaatoSessionGenerate:
    """Tests for JaatoSession.generate()."""

    def test_generate_requires_configuration(self):
        """Test that generate requires session to be configured."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        with pytest.raises(RuntimeError, match="not configured"):
            session.generate("Hello")

    def test_generate_returns_text(self):
        """Test that generate returns response text."""
        mock_runtime = MagicMock()
        mock_provider = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_provider.generate.return_value = mock_response

        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure()

        result = session.generate("Hello")

        assert result == "Generated text"
