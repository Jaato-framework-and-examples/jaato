"""Tests for OutputCallback functionality in JaatoClient."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Tuple

from shared.plugins.base import OutputCallback
from shared.jaato_client import JaatoClient


class TestOutputCallbackType:
    """Tests for OutputCallback type definition."""

    def test_callback_signature(self):
        """OutputCallback accepts (source, text, mode) parameters."""
        calls: List[Tuple[str, str, str]] = []

        def callback(source: str, text: str, mode: str) -> None:
            calls.append((source, text, mode))

        # Verify it matches OutputCallback signature
        cb: OutputCallback = callback
        cb("model", "Hello", "write")
        cb("cli", "output", "append")

        assert calls == [
            ("model", "Hello", "write"),
            ("cli", "output", "append"),
        ]


class TestRunChatLoopCallback:
    """Tests for _run_chat_loop callback invocation."""

    @pytest.fixture
    def mock_chat(self):
        """Create a mock chat object."""
        return MagicMock()

    @pytest.fixture
    def client_with_mock_chat(self, mock_chat):
        """Create a JaatoClient with mocked internals."""
        client = JaatoClient()
        client._client = MagicMock()
        client._model_name = "test-model"
        client._chat = mock_chat
        client._executor = None
        client._turn_accounting = []
        return client

    def _make_function_call(self, name: str, args: dict = None):
        """Create a mock function call with proper string name."""
        fc = MagicMock()
        fc.name = name  # Must be a string, not a MagicMock
        fc.args = args or {}
        return fc

    def test_intermediate_response_triggers_callback(self, client_with_mock_chat, mock_chat):
        """Callback is invoked with intermediate model text during function calling loop."""
        calls: List[Tuple[str, str, str]] = []

        def on_output(source: str, text: str, mode: str) -> None:
            calls.append((source, text, mode))

        # First response: text + function call
        first_response = MagicMock()
        first_response.text = "I'll help you with that."
        first_response.function_calls = [self._make_function_call("test_tool")]

        # Second response: just text (no more function calls)
        second_response = MagicMock()
        second_response.text = "Done!"
        second_response.function_calls = []

        mock_chat.send_message.side_effect = [first_response, second_response]

        # Mock token recording
        client_with_mock_chat._record_token_usage = Mock()
        client_with_mock_chat._accumulate_turn_tokens = Mock()

        result = client_with_mock_chat._run_chat_loop("test message", on_output)

        # Intermediate response should trigger callback
        assert ("model", "I'll help you with that.", "write") in calls
        # Final response should be returned, not sent to callback
        assert result == "Done!"
        # Only intermediate responses go to callback
        assert len(calls) == 1

    def test_no_intermediate_text_no_callback(self, client_with_mock_chat, mock_chat):
        """Callback is not invoked when model produces no intermediate text."""
        calls: List[Tuple[str, str, str]] = []

        def on_output(source: str, text: str, mode: str) -> None:
            calls.append((source, text, mode))

        # First response: function call but no text
        first_response = MagicMock()
        first_response.text = ""  # No text
        first_response.function_calls = [self._make_function_call("test_tool")]

        # Second response: just text
        second_response = MagicMock()
        second_response.text = "Result"
        second_response.function_calls = []

        mock_chat.send_message.side_effect = [first_response, second_response]

        client_with_mock_chat._record_token_usage = Mock()
        client_with_mock_chat._accumulate_turn_tokens = Mock()

        result = client_with_mock_chat._run_chat_loop("test", on_output)

        # No intermediate text = no callback
        assert len(calls) == 0
        assert result == "Result"

    def test_multiple_intermediate_responses(self, client_with_mock_chat, mock_chat):
        """Multiple intermediate responses each trigger the callback."""
        calls: List[Tuple[str, str, str]] = []

        def on_output(source: str, text: str, mode: str) -> None:
            calls.append((source, text, mode))

        # Three responses in the loop
        resp1 = MagicMock()
        resp1.text = "Step 1..."
        resp1.function_calls = [self._make_function_call("tool1")]

        resp2 = MagicMock()
        resp2.text = "Step 2..."
        resp2.function_calls = [self._make_function_call("tool2")]

        resp3 = MagicMock()
        resp3.text = "Final result"
        resp3.function_calls = []

        mock_chat.send_message.side_effect = [resp1, resp2, resp3]

        client_with_mock_chat._record_token_usage = Mock()
        client_with_mock_chat._accumulate_turn_tokens = Mock()

        result = client_with_mock_chat._run_chat_loop("test", on_output)

        # Two intermediate responses
        assert len(calls) == 2
        assert calls[0] == ("model", "Step 1...", "write")
        assert calls[1] == ("model", "Step 2...", "write")
        assert result == "Final result"

    def test_callback_source_is_model(self, client_with_mock_chat, mock_chat):
        """Callback source parameter is always 'model' for model responses."""
        calls: List[Tuple[str, str, str]] = []

        def on_output(source: str, text: str, mode: str) -> None:
            calls.append((source, text, mode))

        first_response = MagicMock()
        first_response.text = "Intermediate"
        first_response.function_calls = [self._make_function_call("tool")]

        second_response = MagicMock()
        second_response.text = "Final"
        second_response.function_calls = []

        mock_chat.send_message.side_effect = [first_response, second_response]

        client_with_mock_chat._record_token_usage = Mock()
        client_with_mock_chat._accumulate_turn_tokens = Mock()

        client_with_mock_chat._run_chat_loop("test", on_output)

        # All calls should have source="model"
        for source, text, mode in calls:
            assert source == "model"

    def test_callback_mode_is_write(self, client_with_mock_chat, mock_chat):
        """Callback mode parameter is always 'write' for new responses."""
        calls: List[Tuple[str, str, str]] = []

        def on_output(source: str, text: str, mode: str) -> None:
            calls.append((source, text, mode))

        first_response = MagicMock()
        first_response.text = "Response 1"
        first_response.function_calls = [self._make_function_call("tool")]

        second_response = MagicMock()
        second_response.text = "Final"
        second_response.function_calls = []

        mock_chat.send_message.side_effect = [first_response, second_response]

        client_with_mock_chat._record_token_usage = Mock()
        client_with_mock_chat._accumulate_turn_tokens = Mock()

        client_with_mock_chat._run_chat_loop("test", on_output)

        # All calls should have mode="write"
        for source, text, mode in calls:
            assert mode == "write"


class TestSendMessageCallback:
    """Tests for send_message callback integration."""

    def test_send_message_passes_callback_to_loop(self):
        """send_message passes on_output callback to _run_chat_loop."""
        client = JaatoClient()
        client._client = MagicMock()
        client._model_name = "test-model"
        client._chat = MagicMock()
        client._gc_plugin = None
        client._registry = None
        client._session_plugin = None

        # Mock _run_chat_loop to capture the callback
        captured_callback = None

        def mock_run_chat_loop(message, on_output):
            nonlocal captured_callback
            captured_callback = on_output
            return "result"

        client._run_chat_loop = mock_run_chat_loop

        def my_callback(source: str, text: str, mode: str) -> None:
            pass

        client.send_message("test", on_output=my_callback)

        assert captured_callback is my_callback


class TestToolExecutorCallback:
    """Tests for ToolExecutor output callback support."""

    def test_executor_stores_callback(self):
        """ToolExecutor stores output callback via set_output_callback."""
        from shared.ai_tool_runner import ToolExecutor

        executor = ToolExecutor()
        calls: List[Tuple[str, str, str]] = []

        def callback(source: str, text: str, mode: str) -> None:
            calls.append((source, text, mode))

        executor.set_output_callback(callback)
        assert executor.get_output_callback() is callback

    def test_executor_clears_callback_with_none(self):
        """ToolExecutor clears callback when set to None."""
        from shared.ai_tool_runner import ToolExecutor

        executor = ToolExecutor()

        def callback(source: str, text: str, mode: str) -> None:
            pass

        executor.set_output_callback(callback)
        assert executor.get_output_callback() is callback

        executor.set_output_callback(None)
        assert executor.get_output_callback() is None


class TestPermissionPluginCallback:
    """Tests for PermissionPlugin output callback support."""

    def test_permission_plugin_forwards_callback_to_actor(self):
        """PermissionPlugin forwards callback to its actor."""
        from shared.plugins.permission import PermissionPlugin
        from shared.plugins.permission.actors import ConsoleActor

        plugin = PermissionPlugin()
        mock_actor = MagicMock(spec=ConsoleActor)
        mock_actor.set_output_callback = MagicMock()

        plugin._actor = mock_actor

        def callback(source: str, text: str, mode: str) -> None:
            pass

        plugin.set_output_callback(callback)
        mock_actor.set_output_callback.assert_called_once_with(callback)


class TestConsoleActorCallback:
    """Tests for ConsoleActor output callback support."""

    def test_console_actor_uses_callback_for_output(self):
        """ConsoleActor uses callback when set."""
        from shared.plugins.permission.actors import ConsoleActor

        actor = ConsoleActor()
        calls: List[Tuple[str, str, str]] = []

        def callback(source: str, text: str, mode: str) -> None:
            calls.append((source, text, mode))

        actor.set_output_callback(callback)

        # Use the output func to emit a message
        actor._output_func("Test message")

        assert len(calls) == 1
        assert calls[0][0] == "permission"  # source
        assert calls[0][1] == "Test message"  # text
        assert calls[0][2] == "append"  # mode

    def test_console_actor_restores_default_on_none(self):
        """ConsoleActor restores default output when callback is None."""
        from shared.plugins.permission.actors import ConsoleActor

        actor = ConsoleActor()
        original_output = actor._output_func

        def callback(source: str, text: str, mode: str) -> None:
            pass

        actor.set_output_callback(callback)
        # Output func should now be the wrapper
        assert actor._output_func is not original_output

        actor.set_output_callback(None)
        # Should restore to default
        assert actor._output_func is actor._default_output_func
