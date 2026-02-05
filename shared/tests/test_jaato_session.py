"""Tests for JaatoSession - per-agent conversation session."""

import pytest
from unittest.mock import MagicMock, patch

from ..jaato_session import JaatoSession
from ..plugins.model_provider.types import Part, FunctionCall, FinishReason


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
        from ..plugins.model_provider.types import TokenUsage

        mock_runtime = MagicMock()
        mock_provider = MagicMock()

        # Setup provider response with parts
        mock_response = MagicMock()
        mock_response.parts = [Part.from_text("Hello back!")]
        mock_response.finish_reason = FinishReason.STOP
        mock_response.usage = TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15)

        # Mock streaming support (enabled by default)
        mock_provider.supports_streaming.return_value = True
        mock_provider.send_message_streaming.return_value = mock_response

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
        mock_response.parts = [Part.from_text("Generated text")]
        mock_response.get_text = lambda: "Generated text"
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
        from ..plugins.model_provider.types import ToolResult

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
        from ..plugins.model_provider.types import ToolResult

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
        from ..plugins.model_provider.types import ToolResult

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
        from ..plugins.model_provider.types import ToolResult

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
        from ..plugins.model_provider.types import ToolResult

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
        from ..plugins.model_provider.types import ToolResult
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

        # Budget should be reduced
        conv_entry = session._instruction_budget.get_entry(InstructionSource.CONVERSATION)
        assert conv_entry.tokens < 5000

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
            tokens_freed=5000,
            details="Freed 2 old turns"
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
        from ..plugins.gc import GCConfig, GCResult

        mock_runtime = MagicMock()
        session = JaatoSession(mock_runtime, "gemini-2.5-flash")

        # Set up a mock GC plugin that frees nothing
        mock_gc_plugin = MagicMock()
        mock_gc_result = GCResult(
            success=True,
            items_collected=0,
            tokens_freed=0,
            details="Nothing to collect"
        )
        mock_gc_plugin.collect.return_value = ([], mock_gc_result)

        session._gc_plugin = mock_gc_plugin
        session._gc_config = GCConfig()

        # Attempt GC recovery
        result = session._try_gc_for_context_recovery(on_output=None)

        # Should return False (GC didn't help)
        assert result is False
