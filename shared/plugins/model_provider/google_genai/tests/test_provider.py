"""Tests for GoogleGenAIProvider structured output support."""

import json
import pytest
from unittest.mock import MagicMock, patch

from ..provider import GoogleGenAIProvider
from ...types import ProviderResponse, TokenUsage, FinishReason


class TestStructuredOutput:
    """Tests for structured output (response_schema) functionality."""

    def test_supports_structured_output_returns_true(self):
        """Gemini provider should report structured output support."""
        provider = GoogleGenAIProvider()
        assert provider.supports_structured_output() is True

    def test_provider_response_has_structured_output_field(self):
        """ProviderResponse should have structured_output field."""
        response = ProviderResponse(text='{"key": "value"}')
        assert response.structured_output is None  # Not set by default

        response.structured_output = {"key": "value"}
        assert response.structured_output == {"key": "value"}
        assert response.has_structured_output is True

    def test_provider_response_has_structured_output_false_when_none(self):
        """has_structured_output should be False when not set."""
        response = ProviderResponse(text="plain text")
        assert response.has_structured_output is False

    @patch('google.genai.Client')
    def test_send_message_with_response_schema(self, mock_client_class):
        """send_message should pass response_schema to SDK config."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_chat = MagicMock()
        mock_client.chats.create.return_value = mock_chat

        # Mock response
        mock_response = MagicMock()
        mock_response.text = '{"name": "Alice", "age": 30}'
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text='{"name": "Alice", "age": 30}')]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30
        )
        mock_chat.send_message.return_value = mock_response

        # Create and configure provider
        provider = GoogleGenAIProvider()
        provider.initialize()
        provider.connect('gemini-2.5-flash')
        provider.create_session()

        # Define response schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }

        # Send message with schema
        response = provider.send_message("Tell me about Alice", response_schema=schema)

        # Verify config was passed
        call_args = mock_chat.send_message.call_args
        config = call_args.kwargs.get('config')
        assert config is not None
        assert config.response_mime_type == "application/json"
        assert config.response_schema == schema

        # Verify structured output was parsed
        assert response.structured_output == {"name": "Alice", "age": 30}
        assert response.has_structured_output is True

    @patch('google.genai.Client')
    def test_send_message_without_response_schema(self, mock_client_class):
        """send_message without schema should not set structured_output."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_chat = MagicMock()
        mock_client.chats.create.return_value = mock_chat

        # Mock response
        mock_response = MagicMock()
        mock_response.text = "Hello! How can I help?"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Hello! How can I help?")]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=5,
            candidates_token_count=10,
            total_token_count=15
        )
        mock_chat.send_message.return_value = mock_response

        # Create and configure provider
        provider = GoogleGenAIProvider()
        provider.initialize()
        provider.connect('gemini-2.5-flash')
        provider.create_session()

        # Send message without schema
        response = provider.send_message("Hello")

        # Verify no config override
        call_args = mock_chat.send_message.call_args
        config = call_args.kwargs.get('config')
        assert config is None

        # Verify no structured output
        assert response.structured_output is None
        assert response.has_structured_output is False

    @patch('google.genai.Client')
    def test_send_message_handles_invalid_json(self, mock_client_class):
        """send_message should handle invalid JSON gracefully."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_chat = MagicMock()
        mock_client.chats.create.return_value = mock_chat

        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.text = "not valid json {"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="not valid json {")]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=5,
            candidates_token_count=5,
            total_token_count=10
        )
        mock_chat.send_message.return_value = mock_response

        # Create and configure provider
        provider = GoogleGenAIProvider()
        provider.initialize()
        provider.connect('gemini-2.5-flash')
        provider.create_session()

        # Send message with schema (model returns invalid JSON)
        schema = {"type": "object"}
        response = provider.send_message("Test", response_schema=schema)

        # Should not raise, structured_output should be None
        assert response.structured_output is None
        assert response.text == "not valid json {"

    @patch('google.genai.Client')
    def test_send_tool_results_with_response_schema(self, mock_client_class):
        """send_tool_results should support response_schema."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_chat = MagicMock()
        mock_client.chats.create.return_value = mock_chat

        # Mock response
        mock_response = MagicMock()
        mock_response.text = '{"status": "success", "count": 42}'
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [
            MagicMock(text='{"status": "success", "count": 42}')
        ]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=20,
            candidates_token_count=10,
            total_token_count=30
        )
        mock_chat.send_message.return_value = mock_response

        # Create and configure provider
        provider = GoogleGenAIProvider()
        provider.initialize()
        provider.connect('gemini-2.5-flash')
        provider.create_session()

        # Import ToolResult
        from ...types import ToolResult

        # Send tool results with schema
        results = [ToolResult(call_id="123", name="test_tool", result={"data": "value"})]
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"}
            }
        }
        response = provider.send_tool_results(results, response_schema=schema)

        # Verify config was passed
        call_args = mock_chat.send_message.call_args
        config = call_args.kwargs.get('config')
        assert config is not None
        assert config.response_mime_type == "application/json"

        # Verify structured output
        assert response.structured_output == {"status": "success", "count": 42}


class TestProviderResponseProperties:
    """Tests for ProviderResponse dataclass."""

    def test_has_function_calls_true(self):
        """has_function_calls should be True when function_calls exist."""
        from ...types import FunctionCall
        response = ProviderResponse(
            function_calls=[FunctionCall(id="1", name="test", args={})]
        )
        assert response.has_function_calls is True

    def test_has_function_calls_false(self):
        """has_function_calls should be False when empty."""
        response = ProviderResponse(text="Hello")
        assert response.has_function_calls is False

    def test_has_structured_output_true(self):
        """has_structured_output should be True when set."""
        response = ProviderResponse(
            text='{"key": "value"}',
            structured_output={"key": "value"}
        )
        assert response.has_structured_output is True

    def test_has_structured_output_false(self):
        """has_structured_output should be False when None."""
        response = ProviderResponse(text="plain text")
        assert response.has_structured_output is False
