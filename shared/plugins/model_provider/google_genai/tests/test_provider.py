"""Tests for GoogleGenAIProvider."""

import json
import pytest
from unittest.mock import MagicMock, patch

from ..provider import GoogleGenAIProvider
from ...base import ProviderConfig
from ...types import ProviderResponse, TokenUsage, FinishReason
from ..errors import (
    CredentialsNotFoundError,
    CredentialsInvalidError,
    CredentialsPermissionError,
    ProjectConfigurationError,
)


def create_mock_client():
    """Create a mock genai.Client with list_models support."""
    mock_client = MagicMock()
    # Mock models.list() for connectivity verification
    mock_client.models.list.return_value = [MagicMock(name="gemini-2.5-flash")]
    return mock_client


class TestAuthentication:
    """Tests for authentication and initialization."""

    @patch('google.genai.Client')
    def test_initialize_with_api_key(self, mock_client_class):
        """Should use AI Studio endpoint with API key."""
        mock_client_class.return_value = create_mock_client()

        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-api-key",
            use_vertex_ai=False,
            auth_method="api_key"
        ))

        # Should create client with api_key, not vertexai
        mock_client_class.assert_called_once_with(api_key="test-api-key")
        assert provider._use_vertex_ai is False
        assert provider._auth_method == "api_key"

    @patch('google.genai.Client')
    def test_initialize_with_vertex_ai(self, mock_client_class):
        """Should use Vertex AI endpoint with project/location."""
        mock_client_class.return_value = create_mock_client()

        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(
            project="test-project",
            location="us-central1",
            use_vertex_ai=True,
            auth_method="adc"
        ))

        # Should create client with vertexai=True
        mock_client_class.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1"
        )
        assert provider._use_vertex_ai is True
        assert provider._project == "test-project"
        assert provider._location == "us-central1"

    def test_initialize_vertex_ai_missing_project_raises(self):
        """Should raise ProjectConfigurationError if project missing."""
        provider = GoogleGenAIProvider()

        with pytest.raises(ProjectConfigurationError) as exc_info:
            provider.initialize(ProviderConfig(
                location="us-central1",
                use_vertex_ai=True,
                auth_method="adc"
            ))

        assert "Project ID is required" in str(exc_info.value)

    def test_initialize_vertex_ai_missing_location_raises(self):
        """Should raise ProjectConfigurationError if location missing."""
        provider = GoogleGenAIProvider()

        with pytest.raises(ProjectConfigurationError) as exc_info:
            provider.initialize(ProviderConfig(
                project="test-project",
                use_vertex_ai=True,
                auth_method="adc"
            ))

        assert "Location is required" in str(exc_info.value)

    def test_initialize_api_key_missing_raises(self):
        """Should raise CredentialsNotFoundError if API key missing."""
        provider = GoogleGenAIProvider()

        with pytest.raises(CredentialsNotFoundError) as exc_info:
            provider.initialize(ProviderConfig(
                use_vertex_ai=False,
                auth_method="api_key"
            ))

        assert "api_key" in str(exc_info.value)

    @patch('google.genai.Client')
    def test_verify_connectivity_permission_error(self, mock_client_class):
        """Should wrap permission errors with actionable message."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("403 Permission denied")
        mock_client_class.return_value = mock_client

        provider = GoogleGenAIProvider()

        with pytest.raises(CredentialsPermissionError):
            provider.initialize(ProviderConfig(
                project="test-project",
                location="us-central1",
                use_vertex_ai=True,
                auth_method="adc"
            ))

    @patch.dict('os.environ', {
        'GOOGLE_GENAI_API_KEY': 'env-api-key'
    }, clear=True)
    @patch('google.genai.Client')
    def test_initialize_auto_detects_api_key_from_env(self, mock_client_class):
        """Should auto-detect API key from environment."""
        mock_client_class.return_value = create_mock_client()

        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(auth_method="auto"))

        # Should use AI Studio with env API key
        mock_client_class.assert_called_once_with(api_key="env-api-key")
        assert provider._use_vertex_ai is False

    @patch.dict('os.environ', {
        'JAATO_GOOGLE_PROJECT': 'env-project',
        'JAATO_GOOGLE_LOCATION': 'europe-west1'
    }, clear=True)
    @patch('google.genai.Client')
    def test_initialize_auto_detects_vertex_from_env(self, mock_client_class):
        """Should auto-detect Vertex AI config from environment."""
        mock_client_class.return_value = create_mock_client()

        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(auth_method="auto"))

        # Should use Vertex AI with env config
        mock_client_class.assert_called_once_with(
            vertexai=True,
            project="env-project",
            location="europe-west1"
        )


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
        mock_client = create_mock_client()
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

        # Create and configure provider with explicit config
        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(
            api_key="test-key",
            use_vertex_ai=False,
            auth_method="api_key"
        ))
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
        mock_client = create_mock_client()
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
        provider.initialize(ProviderConfig(
            api_key="test-key",
            use_vertex_ai=False,
            auth_method="api_key"
        ))
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
        mock_client = create_mock_client()
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
        provider.initialize(ProviderConfig(
            api_key="test-key",
            use_vertex_ai=False,
            auth_method="api_key"
        ))
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
        mock_client = create_mock_client()
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
        provider.initialize(ProviderConfig(
            api_key="test-key",
            use_vertex_ai=False,
            auth_method="api_key"
        ))
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
