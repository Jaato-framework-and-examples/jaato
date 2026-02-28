"""Tests for NIM provider."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from ..provider import NIMProvider, create_provider
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolResult,
    ToolSchema,
)
from ..errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ContextLimitError,
    InfrastructureError,
)
from ..converters import (
    sanitize_tool_name,
    get_original_tool_name,
    clear_tool_name_mapping,
    register_tool_name_mapping,
    tool_schema_to_openai,
    message_to_openai,
    message_from_openai,
    response_from_openai,
    map_finish_reason,
    serialize_history,
    deserialize_history,
)
from ..env import (
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    is_self_hosted,
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_LENGTH,
)


# ==================== Helpers ====================

def create_mock_response(
    text="Hello!",
    tool_calls=None,
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=20,
    reasoning_content=None,
):
    """Create a mock OpenAI ChatCompletion response."""
    mock_response = MagicMock()

    mock_choice = MagicMock()
    mock_choice.finish_reason = finish_reason
    mock_choice.message = MagicMock()
    mock_choice.message.content = text
    mock_choice.message.tool_calls = tool_calls or []
    mock_choice.message.reasoning_content = reasoning_content

    mock_response.choices = [mock_choice]

    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = prompt_tokens + completion_tokens

    return mock_response


def create_mock_tool_call(name="test_tool", args='{"key": "value"}', call_id="call_123"):
    """Create a mock tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return tc


# ==================== Environment Tests ====================

class TestEnvironment:
    """Tests for environment variable resolution."""

    def test_resolve_api_key_from_env(self):
        with patch.dict("os.environ", {"JAATO_NIM_API_KEY": "nvapi-test123"}):
            assert resolve_api_key() == "nvapi-test123"

    def test_resolve_api_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_api_key() is None

    def test_resolve_base_url_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL

    def test_resolve_base_url_from_env(self):
        with patch.dict("os.environ", {"JAATO_NIM_BASE_URL": "http://localhost:8000/v1"}):
            assert resolve_base_url() == "http://localhost:8000/v1"

    def test_resolve_context_length_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_context_length() == DEFAULT_CONTEXT_LENGTH

    def test_resolve_context_length_from_env(self):
        with patch.dict("os.environ", {"JAATO_NIM_CONTEXT_LENGTH": "131072"}):
            assert resolve_context_length() == 131072

    def test_resolve_context_length_invalid(self):
        with patch.dict("os.environ", {"JAATO_NIM_CONTEXT_LENGTH": "not-a-number"}):
            assert resolve_context_length() == DEFAULT_CONTEXT_LENGTH

    def test_is_self_hosted_localhost(self):
        assert is_self_hosted("http://localhost:8000/v1") is True
        assert is_self_hosted("http://127.0.0.1:8000/v1") is True

    def test_is_self_hosted_private_network(self):
        assert is_self_hosted("http://192.168.1.100:8000/v1") is True
        assert is_self_hosted("http://10.0.0.5:8000/v1") is True

    def test_is_self_hosted_public(self):
        assert is_self_hosted("https://integrate.api.nvidia.com/v1") is False
        assert is_self_hosted("https://custom-nim.example.com/v1") is False


# ==================== Converter Tests ====================

class TestToolNameSanitization:
    """Tests for tool name sanitization and reverse mapping."""

    def setup_method(self):
        clear_tool_name_mapping()

    def test_valid_name_unchanged(self):
        assert sanitize_tool_name("my_tool") == "my_tool"
        assert sanitize_tool_name("tool-name") == "tool-name"

    def test_dots_replaced(self):
        assert sanitize_tool_name("mcp.server.tool") == "mcp_server_tool"

    def test_colons_replaced(self):
        assert sanitize_tool_name("ns:tool") == "ns_tool"

    def test_truncation(self):
        long_name = "a" * 100
        assert len(sanitize_tool_name(long_name)) == 64

    def test_reverse_mapping(self):
        register_tool_name_mapping("mcp_server_tool", "mcp.server.tool")
        assert get_original_tool_name("mcp_server_tool") == "mcp.server.tool"

    def test_reverse_mapping_unknown(self):
        assert get_original_tool_name("unknown_tool") == "unknown_tool"

    def test_clear_mapping(self):
        register_tool_name_mapping("sanitized", "original")
        clear_tool_name_mapping()
        assert get_original_tool_name("sanitized") == "sanitized"


class TestToolSchemaConversion:
    """Tests for ToolSchema to OpenAI format conversion."""

    def setup_method(self):
        clear_tool_name_mapping()

    def test_basic_schema(self):
        schema = ToolSchema(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        result = tool_schema_to_openai(schema)

        assert result["type"] == "function"
        assert result["function"]["name"] == "read_file"
        assert result["function"]["description"] == "Read a file"
        assert result["function"]["parameters"]["type"] == "object"

    def test_schema_with_sanitization(self):
        schema = ToolSchema(
            name="mcp.server.tool",
            description="A tool",
            parameters={},
        )
        result = tool_schema_to_openai(schema)

        assert result["function"]["name"] == "mcp_server_tool"
        # Reverse mapping should work
        assert get_original_tool_name("mcp_server_tool") == "mcp.server.tool"


class TestMessageConversion:
    """Tests for Message <-> OpenAI format conversion."""

    def setup_method(self):
        clear_tool_name_mapping()

    def test_user_message(self):
        msg = Message.from_text(Role.USER, "Hello")
        result = message_to_openai(msg)

        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_assistant_message_text(self):
        msg = Message(role=Role.MODEL, parts=[Part(text="Hi there")])
        result = message_to_openai(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Hi there"

    def test_assistant_message_with_tool_calls(self):
        fc = FunctionCall(id="call_1", name="read_file", args={"path": "/tmp"})
        msg = Message(role=Role.MODEL, parts=[Part(function_call=fc)])
        result = message_to_openai(msg)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_1"
        assert result["tool_calls"][0]["function"]["name"] == "read_file"

    def test_tool_result_message(self):
        tr = ToolResult(call_id="call_1", name="read_file", result={"content": "file data"})
        msg = Message(role=Role.TOOL, parts=[Part(function_response=tr)])
        result = message_to_openai(msg)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert json.loads(result["content"]) == {"content": "file data"}

    def test_roundtrip_user_message(self):
        original = Message.from_text(Role.USER, "Hello world")
        openai_msg = message_to_openai(original)
        restored = message_from_openai(openai_msg)

        assert restored.role == Role.USER
        assert restored.parts[0].text == "Hello world"

    def test_roundtrip_assistant_message(self):
        original = Message(role=Role.MODEL, parts=[Part(text="Response")])
        openai_msg = message_to_openai(original)
        restored = message_from_openai(openai_msg)

        assert restored.role == Role.MODEL
        assert restored.parts[0].text == "Response"


class TestResponseConversion:
    """Tests for OpenAI response to ProviderResponse conversion."""

    def test_text_response(self):
        mock = create_mock_response(text="Hello!")
        result = response_from_openai(mock)

        assert result.get_text() == "Hello!"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.prompt_tokens == 10
        assert result.usage.output_tokens == 20

    def test_tool_call_response(self):
        tc = create_mock_tool_call(name="read_file", args='{"path": "/tmp"}')
        mock = create_mock_response(text=None, tool_calls=[tc], finish_reason="tool_calls")

        # Ensure message.content returns None for tool-only responses
        mock.choices[0].message.content = None

        result = response_from_openai(mock)

        assert result.finish_reason == FinishReason.TOOL_USE
        fc_parts = [p for p in result.parts if p.function_call]
        assert len(fc_parts) == 1
        assert fc_parts[0].function_call.name == "read_file"

    def test_empty_response(self):
        mock = MagicMock()
        mock.choices = []
        mock.usage = None

        result = response_from_openai(mock)
        assert result.parts == []
        assert result.finish_reason == FinishReason.UNKNOWN

    def test_reasoning_extraction(self):
        mock = create_mock_response(text="Answer", reasoning_content="Let me think...")
        result = response_from_openai(mock)

        assert result.thinking == "Let me think..."


class TestFinishReasonMapping:
    """Tests for finish reason string mapping."""

    def test_stop(self):
        assert map_finish_reason("stop") == FinishReason.STOP

    def test_length(self):
        assert map_finish_reason("length") == FinishReason.MAX_TOKENS

    def test_tool_calls(self):
        assert map_finish_reason("tool_calls") == FinishReason.TOOL_USE

    def test_content_filter(self):
        assert map_finish_reason("content_filter") == FinishReason.SAFETY

    def test_none(self):
        assert map_finish_reason(None) == FinishReason.UNKNOWN


class TestSerialization:
    """Tests for history serialization/deserialization."""

    def test_roundtrip_text_messages(self):
        history = [
            Message.from_text(Role.USER, "Hello"),
            Message(role=Role.MODEL, parts=[Part(text="Hi")]),
        ]
        data = serialize_history(history)
        restored = deserialize_history(data)

        assert len(restored) == 2
        assert restored[0].role == Role.USER
        assert restored[0].parts[0].text == "Hello"
        assert restored[1].role == Role.MODEL
        assert restored[1].parts[0].text == "Hi"

    def test_roundtrip_function_call(self):
        fc = FunctionCall(id="call_1", name="test", args={"key": "value"})
        history = [
            Message(role=Role.MODEL, parts=[Part(function_call=fc)]),
        ]
        data = serialize_history(history)
        restored = deserialize_history(data)

        assert restored[0].parts[0].function_call.name == "test"
        assert restored[0].parts[0].function_call.args == {"key": "value"}

    def test_roundtrip_tool_result(self):
        tr = ToolResult(call_id="call_1", name="test", result="output", is_error=False)
        history = [
            Message(role=Role.TOOL, parts=[Part(function_response=tr)]),
        ]
        data = serialize_history(history)
        restored = deserialize_history(data)

        assert restored[0].parts[0].function_response.call_id == "call_1"
        assert restored[0].parts[0].function_response.result == "output"


# ==================== Provider Tests ====================

class TestAuthentication:
    """Tests for authentication and initialization."""

    def test_initialize_without_key_raises(self):
        """Should raise APIKeyNotFoundError if no key and not self-hosted."""
        provider = NIMProvider()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIKeyNotFoundError) as exc_info:
                provider.initialize(ProviderConfig())

        assert "JAATO_NIM_API_KEY" in str(exc_info.value)

    @patch("shared.plugins.model_provider.nim.provider.get_openai_client_class")
    def test_initialize_with_api_key(self, mock_client_class):
        """Should initialize with key from config.api_key."""
        mock_client_class.return_value = MagicMock()

        provider = NIMProvider()
        provider.initialize(ProviderConfig(api_key="nvapi-test"))

        assert provider._api_key == "nvapi-test"
        assert provider._client is not None

    @patch("shared.plugins.model_provider.nim.provider.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_NIM_API_KEY": "nvapi-env"}, clear=True)
    def test_initialize_from_env(self, mock_client_class):
        """Should auto-detect key from JAATO_NIM_API_KEY env var."""
        mock_client_class.return_value = MagicMock()

        provider = NIMProvider()
        provider.initialize(ProviderConfig())

        assert provider._api_key == "nvapi-env"

    @patch("shared.plugins.model_provider.nim.provider.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_NIM_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_initialize_self_hosted_no_key(self, mock_client_class):
        """Should initialize without key for self-hosted endpoints."""
        mock_client_class.return_value = MagicMock()

        provider = NIMProvider()
        provider.initialize(ProviderConfig())

        assert provider._api_key is None
        assert provider._client is not None

    @patch("shared.plugins.model_provider.nim.provider.get_openai_client_class")
    def test_initialize_custom_base_url(self, mock_client_class):
        """Should use custom base_url from config.extra."""
        mock_client_class.return_value = MagicMock()

        provider = NIMProvider()
        provider.initialize(ProviderConfig(
            api_key="nvapi-test",
            extra={"base_url": "http://nim.internal:8080/v1"},
        ))

        assert provider._base_url == "http://nim.internal:8080/v1"

    @patch("shared.plugins.model_provider.nim.provider.get_openai_client_class")
    def test_initialize_custom_context_length(self, mock_client_class):
        """Should use custom context_length from config.extra."""
        mock_client_class.return_value = MagicMock()

        provider = NIMProvider()
        provider.initialize(ProviderConfig(
            api_key="nvapi-test",
            extra={"context_length": 131072},
        ))

        assert provider._context_length == 131072


class TestVerifyAuth:
    """Tests for verify_auth (must work before initialize)."""

    def test_verify_auth_with_key(self):
        provider = NIMProvider()
        with patch.dict("os.environ", {"JAATO_NIM_API_KEY": "nvapi-test"}):
            assert provider.verify_auth() is True

    def test_verify_auth_self_hosted(self):
        provider = NIMProvider()
        with patch.dict("os.environ", {"JAATO_NIM_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            assert provider.verify_auth() is True

    def test_verify_auth_no_key_raises(self):
        provider = NIMProvider()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIKeyNotFoundError):
                provider.verify_auth(allow_interactive=False)

    def test_verify_auth_no_key_returns_false(self):
        provider = NIMProvider()
        with patch.dict("os.environ", {}, clear=True):
            assert provider.verify_auth(allow_interactive=True) is False


class TestConnection:
    """Tests for connect and session management."""

    def test_connect_sets_model(self):
        provider = NIMProvider()
        provider._client = MagicMock()
        provider.connect("meta/llama-3.1-70b-instruct")
        assert provider.model_name == "meta/llama-3.1-70b-instruct"

    def test_is_connected(self):
        provider = NIMProvider()
        assert provider.is_connected is False

        provider._client = MagicMock()
        assert provider.is_connected is False

        provider._model_name = "test-model"
        assert provider.is_connected is True

    def test_stateless_no_session_state(self):
        """Provider should not hold any conversation state.

        ``create_session`` / ``get_history`` have been removed as part of
        the migration to the stateless ``complete()`` API.
        """
        provider = NIMProvider()
        assert not hasattr(provider, '_system_instruction')
        assert not hasattr(provider, '_tools')
        assert not hasattr(provider, '_history')


class TestCapabilities:
    """Tests for capability queries."""

    def test_supports_streaming(self):
        assert NIMProvider().supports_streaming() is True

    def test_supports_structured_output(self):
        assert NIMProvider().supports_structured_output() is True

    def test_supports_stop(self):
        assert NIMProvider().supports_stop() is True

    def test_supports_thinking_default(self):
        assert NIMProvider().supports_thinking() is False

    def test_supports_thinking_deepseek(self):
        provider = NIMProvider()
        provider._model_name = "deepseek/deepseek-r1"
        assert provider.supports_thinking() is True

    def test_name(self):
        assert NIMProvider().name == "nim"


class TestTokenManagement:
    """Tests for token counting and context limits."""

    def test_count_tokens(self):
        provider = NIMProvider()
        assert provider.count_tokens("abcd") == 1
        assert provider.count_tokens("a" * 100) == 25

    def test_get_context_limit(self):
        provider = NIMProvider()
        provider._context_length = 131072
        assert provider.get_context_limit() == 131072

    def test_get_token_usage(self):
        provider = NIMProvider()
        assert provider.get_token_usage().total_tokens == 0


class TestErrorClassification:
    """Tests for error classification and retry logic."""

    def test_classify_rate_limit(self):
        provider = NIMProvider()
        exc = RateLimitError(original_error="429")
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": True, "infra": False}

    def test_classify_infrastructure(self):
        provider = NIMProvider()
        exc = InfrastructureError(status_code=500)
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": False, "infra": True}

    def test_classify_unknown(self):
        provider = NIMProvider()
        result = provider.classify_error(ValueError("unknown"))
        assert result is None

    def test_retry_after_rate_limit(self):
        provider = NIMProvider()
        exc = RateLimitError(retry_after=30.0)
        assert provider.get_retry_after(exc) == 30.0

    def test_retry_after_other(self):
        provider = NIMProvider()
        assert provider.get_retry_after(ValueError("x")) is None


class TestCreateProvider:
    """Tests for factory function."""

    def test_create_provider(self):
        provider = create_provider()
        assert isinstance(provider, NIMProvider)
        assert provider.name == "nim"


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_clears_state(self):
        provider = NIMProvider()
        provider._client = MagicMock()
        provider._model_name = "test"

        provider.shutdown()

        assert provider._client is None
        assert provider._model_name is None
