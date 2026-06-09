"""Tests for JaatoSession - per-agent conversation session."""

import pytest
from unittest.mock import MagicMock, patch

from ..jaato_session import JaatoSession
from jaato_sdk.plugins.model_provider.types import Part, FunctionCall, FinishReason, Role


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

    def test_configure_defers_provider_creation(self):
        """Pin: configure() does NOT eagerly create the provider.

        2026-05-13 deferred-provider-INIT change.  Pre-change,
        ``configure()`` called ``runtime.create_provider()`` which
        triggered ``provider.initialize()`` — 9s of network handshake
        on the bootstrap critical path.  Post-change, configure stashes
        the creation args and ``_ensure_provider()`` constructs the
        provider on first model use.

        Two pins:
          1. configure() does NOT call create_provider
          2. is_configured returns True anyway (decoupled from
             ``_provider is not None``)
        """
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

        # Pre-change: create_provider called once during configure.
        # Post-change: NOT called during configure (deferred to _ensure_provider).
        mock_runtime.create_provider.assert_not_called()
        # is_configured is True even though _provider is None.
        assert session.is_configured
        assert session._provider is None
        # Args were stashed for the lazy path.
        assert session._provider_lazy_pending is not None
        assert session._provider_lazy_pending['model_name'] == "gemini-2.5-flash"

    def test_ensure_provider_creates_provider_lazily(self):
        """Pin: ``_ensure_provider()`` creates the provider using the
        args stashed by configure().  Idempotent — second call is
        a no-op that returns the cached provider."""
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
        assert session._provider is None  # not yet

        # First _ensure_provider call — provider gets created.
        result = session._ensure_provider()
        assert result is mock_provider
        assert session._provider is mock_provider
        mock_runtime.create_provider.assert_called_once_with(
            "gemini-2.5-flash",
            provider_name=None,
            skip_model_test=False,
            plugin_configs=None,
        )

        # Second call — idempotent, no additional create_provider call.
        result2 = session._ensure_provider()
        assert result2 is mock_provider
        mock_runtime.create_provider.assert_called_once()  # still just once

    def test_ensure_provider_returns_none_in_skip_provider_mode(self):
        """Pin: skip_provider (auth-pending) mode means the lazy path
        ALSO doesn't create a provider — ``_ensure_provider()`` returns
        None.  The post-auth handler is responsible for stashing the
        pending args and triggering a fresh _ensure_provider call."""
        mock_runtime = MagicMock()
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure(skip_provider=True)
        assert session._provider_lazy_pending is None

        result = session._ensure_provider()
        assert result is None
        mock_runtime.create_provider.assert_not_called()

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

        mock_runtime.get_tool_schemas.assert_called_with(["cli"], preloaded_plugins=set())

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
            additional="Be a researcher.",
            presentation_context=None,
            include_base=True,
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
        """Test that send_message returns response text via provider.complete()."""
        from jaato_sdk.plugins.model_provider.types import TokenUsage, TurnResult

        mock_runtime = MagicMock()
        mock_provider = MagicMock()

        # Setup provider response with parts
        mock_response = MagicMock()
        mock_response.parts = [Part.from_text("Hello back!")]
        mock_response.finish_reason = FinishReason.STOP
        mock_response.usage = TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15)
        mock_response.get_text.return_value = "Hello back!"

        # Mock streaming support (enabled by default)
        mock_provider.supports_streaming.return_value = True
        mock_provider.complete.return_value = TurnResult.from_provider_response(mock_response)

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

    def test_get_history_returns_session_owned_history(self):
        """Test that get_history returns from session-owned history."""
        from jaato_sdk.plugins.model_provider.types import Message, Role

        mock_runtime = MagicMock()
        mock_provider = MagicMock()
        msgs = [
            Message.from_text(Role.USER, "msg1"),
            Message(role=Role.MODEL, parts=[Part.from_text("msg2")]),
        ]
        mock_provider.get_history.return_value = msgs

        mock_runtime.create_provider.return_value = mock_provider
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.registry = None
        mock_runtime.permission_plugin = None

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session.configure()

        # In stateless mode, history is session-owned and starts empty
        # (provider sync is skipped). Populate it directly.
        session._history.replace(list(msgs))

        history = session.get_history()
        assert len(history) == 2
        assert history[0].parts[0].text == "msg1"
        assert history[1].parts[0].text == "msg2"


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
        """Test that generate returns response text via complete()."""
        from jaato_sdk.plugins.model_provider.types import TurnResult

        mock_runtime = MagicMock()
        mock_provider = MagicMock()

        mock_response = MagicMock()
        mock_response.parts = [Part.from_text("Generated text")]
        mock_response.get_text.return_value = "Generated text"
        mock_response.finish_reason = FinishReason.STOP
        mock_provider.complete.return_value = TurnResult.from_provider_response(mock_response)

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
        # Verify complete() was called with a single user message
        mock_provider.complete.assert_called_once()
        call_args = mock_provider.complete.call_args
        messages = call_args[0][0]  # First positional arg
        assert len(messages) == 1
        assert messages[0].role == Role.USER


class TestJaatoSessionTurnProgress:
    """Tests for JaatoSession._emit_turn_progress()."""

    def test_emit_turn_progress_calls_ui_hooks(self):
        """Test that _emit_turn_progress calls ui_hooks.on_turn_progress."""
        mock_runtime = MagicMock()
        mock_ui_hooks = MagicMock()

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session._ui_hooks = mock_ui_hooks
        session._agent_id = "main"

        # Mock get_context_usage to return a percent_used value
        session.get_context_usage = MagicMock(return_value={
            'percent_used': 25.5,
            'total_tokens': 1000,
        })

        # Mock _update_conversation_budget to avoid side effects
        # (it updates conversation tokens and emits instruction budget)
        session._update_conversation_budget = MagicMock()

        turn_data = {'prompt': 800, 'output': 200, 'total': 1000}
        session._emit_turn_progress(turn_data, pending_tool_calls=3)

        mock_ui_hooks.on_turn_progress.assert_called_once_with(
            agent_id="main",
            total_tokens=1000,
            prompt_tokens=800,
            output_tokens=200,
            percent_used=25.5,
            pending_tool_calls=3,
            cache_read_tokens=None,
            cache_creation_tokens=None,
        )

        # Verify conversation budget is updated (which also emits instruction budget)
        session._update_conversation_budget.assert_called_once()

    def test_emit_turn_progress_no_hooks_no_error(self):
        """Test that _emit_turn_progress does nothing when no ui_hooks set."""
        mock_runtime = MagicMock()

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session._ui_hooks = None

        turn_data = {'prompt': 100, 'output': 50, 'total': 150}
        # Should not raise any error
        session._emit_turn_progress(turn_data, pending_tool_calls=0)

    def test_emit_turn_progress_handles_missing_turn_data(self):
        """Test that _emit_turn_progress handles missing keys in turn_data."""
        mock_runtime = MagicMock()
        mock_ui_hooks = MagicMock()

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session._ui_hooks = mock_ui_hooks
        session._agent_id = "test"

        session.get_context_usage = MagicMock(return_value={
            'percent_used': 10.0,
        })

        # Mock _update_conversation_budget to avoid side effects
        session._update_conversation_budget = MagicMock()

        # Empty turn_data - should use defaults of 0
        turn_data = {}
        session._emit_turn_progress(turn_data, pending_tool_calls=1)

        mock_ui_hooks.on_turn_progress.assert_called_once_with(
            agent_id="test",
            total_tokens=0,
            prompt_tokens=0,
            output_tokens=0,
            percent_used=10.0,
            pending_tool_calls=1,
            cache_read_tokens=None,
            cache_creation_tokens=None,
        )


class TestJaatoSessionFrameworkEnrichment:
    """Tests for JaatoSession._get_framework_enrichments()."""

    def test_detects_system_reminder_tag(self):
        """Test that system reminder tags are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "Some text <system-reminder>Remember this</system-reminder> more text"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["system-reminder"]

    def test_detects_system_notice_gc(self):
        """Test that [System: ...] GC notices are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "[System: Context reduced by 50%] Continuing conversation..."
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["gc"]

    def test_detects_system_notice_cancellation(self):
        """Test that [System: ...] cancellation notices are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "[System: Your previous response was cancelled by the user]"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["cancellation"]

    def test_detects_system_notice_multimodal(self):
        """Test that [System: ...] multimodal notices are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "[System: The following image files are referenced: photo.jpg]"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["multimodal"]

    def test_detects_system_notice_session(self):
        """Test that [System: ...] session notices are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "[System: This conversation has been ongoing for a while...]"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["session"]

    def test_detects_memory_injection(self):
        """Test that memory injection marker is detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "💡 **Available Memories**\n- Memory 1\n- Memory 2"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["memory"]

    def test_detects_hidden_waypoint(self):
        """Test that hidden waypoint tags are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "<hidden><waypoint-restore>Restored to checkpoint</waypoint-restore></hidden>"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["waypoint"]

    def test_detects_hidden_streaming(self):
        """Test that hidden streaming tags are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "<hidden><streaming_updates>New data available</streaming_updates></hidden>"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["streaming"]

        # Also test streaming with tool prefix format
        text2 = "<hidden>[tool_name] chunk content</hidden>"
        enrichments2 = session._get_framework_enrichments(text2)
        assert enrichments2 == ["streaming"]

    def test_detects_hidden_nudge(self):
        """Test that hidden nudge tags are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "<hidden>Your response indicated TOOL_USE but contained no function call.</hidden>"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == ["nudge"]

    def test_detects_multiple_enrichments(self):
        """Test that multiple enrichment types are detected."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = (
            "💡 **Available Memories**\n- Memory 1\n"
            "<system-reminder>Remember this</system-reminder>\n"
            "[System: GC completed]"
        )
        enrichments = session._get_framework_enrichments(text)
        assert "system-reminder" in enrichments
        assert "memory" in enrichments
        assert "gc" in enrichments

    def test_no_enrichment_in_plain_text(self):
        """Test that plain user text is not flagged as enrichment."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        text = "Please help me fix this bug in my Python code"
        enrichments = session._get_framework_enrichments(text)
        assert enrichments == []

    def test_empty_text_not_enrichment(self):
        """Test that empty text is not flagged as enrichment."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        assert session._get_framework_enrichments("") == []
        assert session._get_framework_enrichments(None) == []


class TestContextLimitRecovery:
    """Tests for context limit error recovery and truncation."""

    def test_truncate_preserves_first_lines(self):
        """Test that truncation keeps the first N lines of large results."""
        from jaato_sdk.plugins.model_provider.types import ToolResult

        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # Create a result with many lines (100 lines, each ~40 chars = ~1000 tokens)
        large_content = "\n".join([f"Line {i}: Some content here" for i in range(100)])
        tool_results = [
            ToolResult(call_id="1", name="read_file", result=large_content, is_error=False)
        ]

        # Request truncation: current=128500, limit=128000
        # Target is 80% of limit = 102400, so we need to remove 26100 tokens
        truncated = session._truncate_results_to_fit(
            tool_results, current_tokens=128500, limit_tokens=128000
        )

        # Should have truncated
        assert truncated[0].result != large_content
        # Should preserve first 20 lines (the default)
        for i in range(20):
            assert f"Line {i}:" in truncated[0].result
        # Should have truncation notice
        assert "[NOTICE:" in truncated[0].result
        assert "automatically truncated" in truncated[0].result

    def test_truncate_skips_small_results(self):
        """Test that small results are not truncated."""
        from jaato_sdk.plugins.model_provider.types import ToolResult

        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # Small result (< 200 estimated tokens)
        small_content = "Short result"
        tool_results = [
            ToolResult(call_id="1", name="echo", result=small_content, is_error=False)
        ]

        # Even with context exceeded, small results should not be truncated
        truncated = session._truncate_results_to_fit(
            tool_results, current_tokens=128100, limit_tokens=128000
        )

        # Should NOT be truncated (too small to be worth it)
        assert truncated[0].result == small_content

    def test_truncate_targets_largest_first(self):
        """Test that truncation targets the largest results first."""
        from jaato_sdk.plugins.model_provider.types import ToolResult

        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        small_content = "Small result"
        # Make content large enough (~1000 tokens = ~4000 chars)
        large_content = "\n".join([f"Line {i}: {'x' * 30}" for i in range(100)])

        tool_results = [
            ToolResult(call_id="1", name="small_tool", result=small_content, is_error=False),
            ToolResult(call_id="2", name="large_tool", result=large_content, is_error=False),
        ]

        # Request truncation with context exceeded
        truncated = session._truncate_results_to_fit(
            tool_results, current_tokens=128500, limit_tokens=128000
        )

        # Small should be unchanged (too small to truncate)
        assert truncated[0].result == small_content
        # Large should be truncated
        assert truncated[1].result != large_content
        assert "[NOTICE:" in truncated[1].result

    def test_truncate_with_unparseable_tokens_uses_aggressive_default(self):
        """Test that unparseable token counts trigger aggressive truncation."""
        from jaato_sdk.plugins.model_provider.types import ToolResult

        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        large_content = "\n".join([f"Line {i}: content" for i in range(100)])
        tool_results = [
            ToolResult(call_id="1", name="read_file", result=large_content, is_error=False)
        ]

        # When token counts can't be parsed (0, 0), use aggressive default (50% of results)
        truncated = session._truncate_results_to_fit(
            tool_results, current_tokens=0, limit_tokens=0
        )

        assert truncated[0].result != large_content
        assert "[NOTICE:" in truncated[0].result

    def test_truncate_uses_char_based_for_few_lines(self):
        """Test that char-based truncation is used when content has few lines."""
        from jaato_sdk.plugins.model_provider.types import ToolResult

        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # Large content with only 3 lines (simulates JSON or base64)
        large_single_line = "x" * 100000  # ~25000 tokens in ~3 lines
        large_content = f"line1\n{large_single_line}\nline3"
        tool_results = [
            ToolResult(call_id="1", name="read_file", result=large_content, is_error=False)
        ]

        # Request truncation with significant overflow
        truncated = session._truncate_results_to_fit(
            tool_results, current_tokens=148000, limit_tokens=128000
        )

        # Should have truncated using char-based method
        assert truncated[0].result != large_content
        assert "[NOTICE:" in truncated[0].result
        assert "characters" in truncated[0].result  # Should mention characters, not lines
        # Should be much shorter than original
        assert len(truncated[0].result) < len(large_content) / 2

    def test_sync_budget_after_truncation(self):
        """Test that budget is adjusted after truncation."""
        from jaato_sdk.plugins.model_provider.types import ToolResult
        from ..instruction_budget import InstructionBudget, InstructionSource

        mock_runtime = MagicMock()
        mock_runtime.ledger = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # Create an instruction budget
        session._instruction_budget = InstructionBudget.create_default(
            session_id="test",
            agent_id="main",
            agent_type="main",
            context_limit=100000,
        )
        # Set some conversation tokens
        session._instruction_budget.update_tokens(InstructionSource.CONVERSATION, 5000)

        # Create original and truncated results
        large_content = "x" * 4000  # ~1000 tokens
        small_content = "x" * 400   # ~100 tokens

        original = [ToolResult(call_id="1", name="tool", result=large_content, is_error=False)]
        truncated = [ToolResult(call_id="1", name="tool", result=small_content, is_error=False)]

        # Sync budget
        session._sync_budget_after_truncation(original, truncated)

        # Budget tokens are NOT adjusted inline (Bug C fix) — the budget
        # rebuilds from actual history at turn-end via _update_conversation_budget().
        # The method only records the event in the ledger.

        # Ledger should record the event
        mock_runtime.ledger._record.assert_called_once()
        call_args = mock_runtime.ledger._record.call_args
        assert call_args[0][0] == 'context-limit-truncation'

    def test_try_gc_for_context_recovery_with_gc_plugin(self):
        """Test that GC is attempted during context limit recovery when plugin is available."""
        from ..plugins.gc import GCConfig, GCResult, GCTriggerReason

        mock_runtime = MagicMock()
        mock_runtime.ledger = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # Set up a mock GC plugin
        mock_gc_plugin = MagicMock()
        mock_gc_result = GCResult(
            success=True,
            items_collected=2,
            tokens_before=10000,
            tokens_after=5000,
            plugin_name="gc_budget",
            trigger_reason=GCTriggerReason.CONTEXT_LIMIT,
        )
        mock_gc_plugin.collect.return_value = ([], mock_gc_result)

        session._gc_plugin = mock_gc_plugin
        session._gc_config = GCConfig()

        # Attempt GC recovery
        result = session._try_gc_for_context_recovery(on_output=None)

        # Should have called the GC plugin
        assert mock_gc_plugin.collect.called
        # Should return True (GC helped)
        assert result is True
        # Should have been called with CONTEXT_LIMIT reason
        call_args = mock_gc_plugin.collect.call_args
        assert call_args[0][3] == GCTriggerReason.CONTEXT_LIMIT

    def test_try_gc_for_context_recovery_without_gc_plugin(self):
        """Test that GC recovery gracefully handles missing GC plugin."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # No GC plugin configured
        session._gc_plugin = None
        session._gc_config = None

        # Should return False without error
        result = session._try_gc_for_context_recovery(on_output=None)
        assert result is False

    def test_try_gc_for_context_recovery_gc_frees_nothing(self):
        """Test that GC recovery returns False when GC frees nothing."""
        from ..plugins.gc import GCConfig, GCResult, GCTriggerReason

        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # Set up a mock GC plugin that frees nothing
        mock_gc_plugin = MagicMock()
        mock_gc_result = GCResult(
            success=True,
            items_collected=0,
            tokens_before=5000,
            tokens_after=5000,
            plugin_name="gc_budget",
            trigger_reason=GCTriggerReason.CONTEXT_LIMIT,
        )
        mock_gc_plugin.collect.return_value = ([], mock_gc_result)

        session._gc_plugin = mock_gc_plugin
        session._gc_config = GCConfig()

        # Attempt GC recovery
        result = session._try_gc_for_context_recovery(on_output=None)

        # Should return False (GC didn't help)
        assert result is False


class TestWaypointWiring:
    """Tests for the JaatoSession → WaypointPlugin set_session_callbacks
    wiring.  Without this, every waypoint is saved with
    history_snapshot=None — and downstream consumers (waypoint_info
    metadata, premium handoff fork_from_waypoint) get nothing to read."""

    def _build_session_with_waypoint_plugin(self):
        mock_runtime = MagicMock()
        mock_runtime.create_provider.return_value = MagicMock()
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.permission_plugin = None

        # Registry returns a waypoint plugin (and nothing else for other
        # lookups so wire-loop iterates cleanly).
        mock_waypoint_plugin = MagicMock()
        mock_runtime.registry = MagicMock()
        mock_runtime.registry._exposed = []
        mock_runtime.registry.get_plugin.side_effect = lambda name: (
            mock_waypoint_plugin if name == "waypoint" else None
        )
        mock_runtime.registry.collect_prerequisite_policies.return_value = []

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        return session, mock_waypoint_plugin

    def test_set_session_callbacks_invoked_on_configure(self):
        """The wiring fires during configure() so waypoints created
        afterwards capture history."""
        session, waypoint_plugin = self._build_session_with_waypoint_plugin()

        session.configure()

        waypoint_plugin.set_session_callbacks.assert_called_once()
        kwargs = waypoint_plugin.set_session_callbacks.call_args.kwargs
        assert "get_history" in kwargs
        assert "serialize_history" in kwargs
        assert "get_turn_index" in kwargs

    def test_get_history_callback_returns_session_history(self):
        """The wired get_history callable must return the actual session's
        live history list (not a stale snapshot)."""
        session, waypoint_plugin = self._build_session_with_waypoint_plugin()
        session.configure()

        # Append a message after wiring — the callback should see it.
        from jaato_sdk.plugins.model_provider.types import Message
        session._history.append(Message.from_text(Role.USER, "hi"))

        get_history = waypoint_plugin.set_session_callbacks.call_args.kwargs[
            "get_history"
        ]
        history = get_history()
        assert len(history) == 1
        assert history[0].parts[0].text == "hi"

    def test_serialize_history_produces_json_string_round_trippable(self):
        """The json.dumps wrapper must produce a string that
        deserialize_history can round-trip — that's the contract premium's
        future fork_from_waypoint will rely on."""
        import json
        from jaato_sdk.plugins.model_provider.types import Message
        from ..plugins.session.serializer import deserialize_history

        session, waypoint_plugin = self._build_session_with_waypoint_plugin()
        session.configure()

        serialize_history = waypoint_plugin.set_session_callbacks.call_args.kwargs[
            "serialize_history"
        ]
        msgs = [
            Message.from_text(Role.USER, "first"),
            Message.from_text(Role.MODEL, "second"),
        ]
        snapshot = serialize_history(msgs)

        assert isinstance(snapshot, str)
        round_tripped = deserialize_history(json.loads(snapshot))
        assert len(round_tripped) == 2
        assert round_tripped[0].parts[0].text == "first"
        assert round_tripped[1].parts[0].text == "second"
        assert round_tripped[0].role == Role.USER
        assert round_tripped[1].role == Role.MODEL

    def test_get_turn_index_callback_tracks_session_turn(self):
        """Turn index callback reads the live session counter, so a
        waypoint created mid-session captures the right turn position."""
        session, waypoint_plugin = self._build_session_with_waypoint_plugin()
        session.configure()

        get_turn_index = waypoint_plugin.set_session_callbacks.call_args.kwargs[
            "get_turn_index"
        ]
        assert get_turn_index() == 0  # fresh session

        session._turn_index = 7
        assert get_turn_index() == 7  # tracks live state

    def test_no_wiring_when_waypoint_plugin_absent(self):
        """When waypoint isn't in the profile's plugin set, registry
        returns None and the wiring is silently skipped — no crash."""
        mock_runtime = MagicMock()
        mock_runtime.create_provider.return_value = MagicMock()
        mock_runtime.get_tool_schemas.return_value = []
        mock_runtime.get_executors.return_value = {}
        mock_runtime.get_system_instructions.return_value = None
        mock_runtime.permission_plugin = None
        mock_runtime.registry = MagicMock()
        mock_runtime.registry._exposed = []
        mock_runtime.registry.get_plugin.return_value = None
        mock_runtime.registry.collect_prerequisite_policies.return_value = []

        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # Should not raise.
        session.configure()
        assert session.is_configured


class TestSetInitialHistory:
    """set_initial_history is the spawn-from-snapshot primitive consumed by
    create_headless_session(initial_history=...) and (downstream) premium's
    fork_session_from_history reactor action."""

    def _fresh_session(self):
        mock_runtime = MagicMock()
        return JaatoSession(mock_runtime, "gemini-2.5-flash")

    def _make_user_message(self, text: str):
        from jaato_sdk.plugins.model_provider.types import Message
        return Message.from_text(Role.USER, text)

    def test_seeds_empty_history(self):
        session = self._fresh_session()
        msgs = [
            self._make_user_message("hello"),
            self._make_user_message("world"),
        ]

        session.set_initial_history(msgs)

        assert len(session._history.messages_ref) == 2
        assert session._history.messages_ref[0].parts[0].text == "hello"
        assert session._history.messages_ref[1].parts[0].text == "world"

    def test_takes_a_copy_not_aliased(self):
        """Caller mutations to the source list must not bleed into the
        seeded history (delegated to SessionHistory.replace)."""
        session = self._fresh_session()
        msgs = [self._make_user_message("first")]

        session.set_initial_history(msgs)
        msgs.append(self._make_user_message("second"))

        assert len(session._history.messages_ref) == 1

    def test_rejects_non_empty_history(self):
        """Defensive guard: refuse to overwrite an existing conversation."""
        session = self._fresh_session()
        session._history.append(self._make_user_message("preexisting"))

        with pytest.raises(RuntimeError, match="empty history"):
            session.set_initial_history(
                [self._make_user_message("attempted overwrite")]
            )

    def test_rejects_mid_turn_session(self):
        """Defensive guard: refuse if session is in the middle of a turn."""
        session = self._fresh_session()
        session._is_running = True

        with pytest.raises(RuntimeError, match="idle session"):
            session.set_initial_history([self._make_user_message("nope")])

    def test_does_not_touch_system_instruction(self):
        """The new session's system instruction is independent of the
        replayed user/assistant turns — confirm we don't accidentally
        write to it."""
        session = self._fresh_session()
        session._system_instruction = "you are agent X"

        session.set_initial_history([self._make_user_message("turn 1")])

        assert session._system_instruction == "you are agent X"


class TestGetToolSchemas:
    """Tests for ``JaatoSession.get_tool_schemas`` (Phase 3 §7c step 3b).

    Public read accessor replacing daemon-side reads of the
    private ``self._tools`` attribute.
    """

    def test_returns_empty_list_before_configure(self):
        """Pre-configure: ``self._tools`` is None.  The accessor
        returns ``[]`` so callers can iterate unconditionally."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        assert session._tools is None
        assert session.get_tool_schemas() == []

    def test_returns_copy_of_tools_after_configure(self):
        """Post-configure: returns a list of the resolved schemas.
        The returned list is a copy — callers can't mutate the
        session's internal state by appending to the result."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        schema_a = MagicMock(name="schema_a")
        schema_b = MagicMock(name="schema_b")
        session._tools = [schema_a, schema_b]

        result = session.get_tool_schemas()

        assert result == [schema_a, schema_b]
        # Mutating the result must not touch session state.
        result.append(MagicMock(name="injected"))
        assert len(session._tools) == 2

    def test_returns_empty_list_when_tools_empty(self):
        """``self._tools = []`` (post-configure with no tools) is
        a legitimate state — the accessor returns ``[]`` (not
        ``None``) so callers iterate cleanly."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")
        session._tools = []
        assert session.get_tool_schemas() == []


class TestRestoreTurnAccounting:
    """Tests for ``JaatoSession.restore_turn_accounting`` (Phase 3
    §7c step 6.6.1.0).

    Public surface replacing the daemon's private-attribute
    write at ``server/session_manager.py:2558-2559``.
    Prerequisite for the upcoming ``session.restore_turn_accounting``
    runner-RPC handler (§7c step 6.6.1.2).
    """

    def test_replaces_existing_turn_accounting(self):
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "claude-sonnet-4-6")
        # Seed with prior turns.
        session._turn_accounting = [{"prompt_tokens": 100}]
        new_turns = [{"prompt_tokens": 200}, {"prompt_tokens": 300}]

        session.restore_turn_accounting(new_turns)

        assert session._turn_accounting == new_turns

    def test_takes_a_copy_not_aliased(self):
        """Caller-mutation must not propagate into session state."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "claude-sonnet-4-6")
        turns = [{"prompt_tokens": 100}]

        session.restore_turn_accounting(turns)
        turns.append({"prompt_tokens": 999})  # mutate caller's list

        # Session state unchanged.
        assert session._turn_accounting == [{"prompt_tokens": 100}]

    def test_empty_list_clears(self):
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "claude-sonnet-4-6")
        session._turn_accounting = [{"prompt_tokens": 100}]

        session.restore_turn_accounting([])

        assert session._turn_accounting == []


class TestRestoreConversationBudget:
    """Tests for ``JaatoSession.restore_conversation_budget``
    (Phase 3 §7c step 6.6.1.0).

    Public surface replacing the daemon's reach through
    ``session.instruction_budget.restore_conversation_from_snapshot``
    at ``server/session_manager.py:2592-2593``.  Prerequisite
    for the upcoming ``session.restore_conversation_budget``
    runner-RPC handler (§7c step 6.6.1.3).
    """

    def test_forwards_to_instruction_budget(self):
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "claude-sonnet-4-6")
        mock_budget = MagicMock()
        session._instruction_budget = mock_budget
        snapshot = {"tokens": 500, "items": []}

        session.restore_conversation_budget(snapshot)

        mock_budget.restore_conversation_from_snapshot.assert_called_once_with(
            snapshot,
        )

    def test_noops_when_no_budget(self):
        """Pre-configure: ``self._instruction_budget`` is None.
        Method is a clean no-op (does NOT raise) — matches the
        daemon caller's existing ``if jaato_session.instruction_budget:``
        guard semantics."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "claude-sonnet-4-6")
        assert session._instruction_budget is None

        # Should not raise.
        session.restore_conversation_budget({"tokens": 500})


class TestJaatoSessionReplayMessagesLazyProvider:
    """Tests for replay_messages's lazy-provider integration (PR-216).

    Background: the 2026-05-13 lazy-provider-INIT refactor moved
    provider construction out of ``configure()`` into a lazy
    ``_ensure_provider()`` call triggered on first model use.
    ``send_message`` was updated to call ``_ensure_provider()``
    before checking ``self._provider``.  ``replay_messages`` missed
    that refactor and kept a bare ``if not self._provider`` check —
    which surfaces as ``"Session not configured — cannot replay"``
    on a fully-configured session whose provider hasn't been
    materialised yet.

    Canonical caller hitting the bug: forensic-fork sessions created
    via ``SessionManager.create_headless_session`` then invoked from
    ``session_ops.interrogate_session.replay_messages`` — the fork
    is configured, but no ``send_message`` ever fired on it so the
    provider was never lazy-created.

    Fix: call ``_ensure_provider()`` before the check, mirror
    ``send_message:3560``'s pattern.
    """

    def test_ensure_provider_called_before_check(self):
        """The fix: ``_ensure_provider`` MUST be called before the
        ``if not self._provider`` check.  Pre-PR-216 it wasn't, so
        fully-configured sessions with deferred-INIT providers
        raised on every replay_messages call.

        We assert on the call-ordering contract (the actual defect)
        rather than driving the full ``provider.complete`` integration
        — that's covered at the runner-RPC + provider-plugin layer
        and would require deep mock-chain plumbing here for marginal
        signal.
        """
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "glm-5-turbo")
        session._configured = True

        # Patch _ensure_provider so we can assert it was called BEFORE
        # the ``if not self._provider`` check.  Leave self._provider
        # = None after the call so we short-circuit at the no-provider
        # raise (the test pins the call-ordering contract; integration
        # is covered elsewhere).
        with patch.object(
            session, "_ensure_provider", return_value=None,
        ) as mock_ensure:
            session._provider_lazy_pending = None  # skip_provider mode
            with pytest.raises(RuntimeError, match="no provider"):
                session.replay_messages([], timeout=1.0)

        mock_ensure.assert_called_once()

    def test_skip_provider_mode_still_raises_with_clearer_error(self):
        """In skip_provider (auth-pending) mode,
        ``_provider_lazy_pending`` is None → ``_ensure_provider``
        returns None → replay_messages raises.  The error message
        now points at the actual condition (no provider materialisable)
        rather than the misleading 'Session not configured' from
        pre-PR-216."""
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "glm-5-turbo")
        session._configured = True  # session IS configured
        session._provider_lazy_pending = None  # but no provider can be made

        with pytest.raises(RuntimeError, match="no provider"):
            session.replay_messages([], timeout=1.0)


class TestForceNarrationBetweenTools:
    """Probe B (2026-06-09): ``JAATO_FORCE_NARRATION_BETWEEN_TOOLS=true``
    env var gates a synthetic-user-prompt injection after every tool
    result append.  Closes the small-model narration-skipping failure
    class (qwen3-14b @ temp=0 in tool-mode skips narration regardless
    of persona prose AND in-context examples — see
    ``feedback_small_model_narration_skipping_is_structural``).
    """

    def test_default_is_false_no_injection(self, monkeypatch):
        """Without the env var, ``_force_narration_between_tools`` is
        False and no synthetic prompt is injected."""
        monkeypatch.delenv("JAATO_FORCE_NARRATION_BETWEEN_TOOLS", raising=False)
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "Qwen/Qwen3-14B")
        assert session._force_narration_between_tools is False

    def test_env_var_true_enables(self, monkeypatch):
        monkeypatch.setenv("JAATO_FORCE_NARRATION_BETWEEN_TOOLS", "true")
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "Qwen/Qwen3-14B")
        assert session._force_narration_between_tools is True

    def test_env_var_one_enables(self, monkeypatch):
        monkeypatch.setenv("JAATO_FORCE_NARRATION_BETWEEN_TOOLS", "1")
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "Qwen/Qwen3-14B")
        assert session._force_narration_between_tools is True

    def test_env_var_false_disabled(self, monkeypatch):
        monkeypatch.setenv("JAATO_FORCE_NARRATION_BETWEEN_TOOLS", "false")
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "Qwen/Qwen3-14B")
        assert session._force_narration_between_tools is False

    def test_env_var_garbage_disabled(self, monkeypatch):
        """Anything not in {true, 1, yes, on} disables — including
        accidental empty / whitespace / typo values."""
        monkeypatch.setenv("JAATO_FORCE_NARRATION_BETWEEN_TOOLS", "ye")
        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "Qwen/Qwen3-14B")
        assert session._force_narration_between_tools is False
