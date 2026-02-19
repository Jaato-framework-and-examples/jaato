"""Tests for AnthropicProvider."""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from ..provider import AnthropicProvider, MODEL_CONTEXT_LIMITS
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import (
    ProviderResponse,
    TokenUsage,
    FinishReason,
    FunctionCall,
    ToolResult,
    ToolSchema,
    Message,
    Part,
    Role,
)
from ..errors import (
    APIKeyNotFoundError,
    APIKeyInvalidError,
    RateLimitError,
    ContextLimitError,
    ModelNotFoundError,
    OverloadedError,
)


def create_mock_response(
    text: str = "Hello!",
    tool_use: list = None,
    thinking: str = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    """Create a mock Anthropic response."""
    mock_response = MagicMock()
    mock_response.stop_reason = stop_reason

    # Build content blocks
    content = []
    if thinking:
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = thinking
        content.append(thinking_block)

    if text:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        content.append(text_block)

    if tool_use:
        for tu in tool_use:
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = tu.get("id", "toolu_123")
            tool_block.name = tu.get("name", "test_tool")
            tool_block.input = tu.get("input", {})
            content.append(tool_block)

    mock_response.content = content

    # Create usage
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens

    return mock_response


class TestAuthentication:
    """Tests for authentication and initialization."""

    def test_initialize_without_api_key_raises(self):
        """Should raise APIKeyNotFoundError if no API key provided."""
        provider = AnthropicProvider()

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(APIKeyNotFoundError) as exc_info:
                provider.initialize(ProviderConfig())

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch('anthropic.Anthropic')
    def test_initialize_with_api_key(self, mock_client_class):
        """Should initialize with API key from config."""
        mock_client_class.return_value = MagicMock()

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test123"))

        assert provider._api_key == "sk-ant-test123"
        mock_client_class.assert_called_once()

    @patch('anthropic.Anthropic')
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'sk-ant-env-key'}, clear=True)
    def test_initialize_from_env(self, mock_client_class):
        """Should auto-detect API key from ANTHROPIC_API_KEY env var."""
        mock_client_class.return_value = MagicMock()

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig())

        assert provider._api_key == "sk-ant-env-key"

    @patch('anthropic.Anthropic')
    def test_initialize_with_caching_enabled(self, mock_client_class):
        """Should enable caching when configured."""
        mock_client_class.return_value = MagicMock()

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={'enable_caching': True}
        ))

        assert provider._enable_caching is True

    @patch('anthropic.Anthropic')
    def test_initialize_with_thinking_enabled(self, mock_client_class):
        """Should enable extended thinking when configured."""
        mock_client_class.return_value = MagicMock()

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={
                'enable_thinking': True,
                'thinking_budget': 15000
            }
        ))

        assert provider._enable_thinking is True
        assert provider._thinking_budget == 15000


class TestConnection:
    """Tests for model connection."""

    @patch('anthropic.Anthropic')
    def test_connect_sets_model(self, mock_client_class):
        """connect() should set the model name."""
        mock_client_class.return_value = MagicMock()

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        assert provider.model_name == 'claude-sonnet-4-20250514'
        assert provider.is_connected is True

    @patch('anthropic.Anthropic')
    def test_is_connected_false_without_model(self, mock_client_class):
        """is_connected should be False without model."""
        mock_client_class.return_value = MagicMock()

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))

        assert provider.is_connected is False

    def test_list_models_returns_known_models(self):
        """list_models() should return known Claude model IDs."""
        provider = AnthropicProvider()
        models = provider.list_models()

        assert 'claude-sonnet-4-20250514' in models
        assert 'claude-3-5-sonnet-20241022' in models
        assert 'claude-opus-4-5-20251101' in models

    def test_list_models_with_prefix_filter(self):
        """list_models() should filter by prefix."""
        provider = AnthropicProvider()

        claude_4_models = provider.list_models(prefix='claude-sonnet-4')
        assert all(m.startswith('claude-sonnet-4') for m in claude_4_models)

        claude_3_5_models = provider.list_models(prefix='claude-3-5')
        assert all(m.startswith('claude-3-5') for m in claude_3_5_models)


class TestMessaging:
    """Tests for sending messages."""

    @patch('anthropic.Anthropic')
    def test_send_message_returns_response(self, mock_client_class):
        """send_message() should return ProviderResponse."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(text="Hello!")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        response = provider.send_message("Hi")

        assert response.get_text() == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.output_tokens == 20

    @patch('anthropic.Anthropic')
    def test_send_message_adds_to_history(self, mock_client_class):
        """send_message() should add messages to history."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(text="Response")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        provider.send_message("Question")
        history = provider.get_history()

        assert len(history) == 2  # User message + assistant response
        assert history[0].role == Role.USER
        assert history[1].role == Role.MODEL

    @patch('anthropic.Anthropic')
    def test_send_message_with_system_instruction(self, mock_client_class):
        """send_message() should include system instruction."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session(system_instruction="You are helpful.")

        provider.send_message("Hi")

        # Verify system was included in call
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert 'system' in call_kwargs
        assert call_kwargs['system'] == "You are helpful."

    @patch('anthropic.Anthropic')
    def test_generate_one_shot(self, mock_client_class):
        """generate() should work without session."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(text="Generated")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        response = provider.generate("One-shot prompt")

        assert response.get_text() == "Generated"


class TestExtendedThinking:
    """Tests for extended thinking feature."""

    @patch('anthropic.Anthropic')
    def test_response_includes_thinking(self, mock_client_class):
        """Response should include thinking when present."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(
            text="Final answer",
            thinking="Let me think about this step by step..."
        )
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={'enable_thinking': True}
        ))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        response = provider.send_message("Complex question")

        assert response.get_text() == "Final answer"
        assert response.thinking == "Let me think about this step by step..."
        assert response.has_thinking is True

    @patch('anthropic.Anthropic')
    def test_thinking_config_passed_to_api(self, mock_client_class):
        """Thinking config should be passed to API when enabled."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={
                'enable_thinking': True,
                'thinking_budget': 8000
            }
        ))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        provider.send_message("Think about this")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert 'thinking' in call_kwargs
        assert call_kwargs['thinking']['type'] == 'enabled'
        assert call_kwargs['thinking']['budget_tokens'] == 8000


class TestFunctionCalling:
    """Tests for function calling support."""

    @patch('anthropic.Anthropic')
    def test_send_message_with_tools(self, mock_client_class):
        """send_message() should pass tools to API."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        tools = [ToolSchema(
            name='get_weather',
            description='Get weather for a location',
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        )]
        provider.create_session(tools=tools)

        provider.send_message("What's the weather?")

        # Verify tools were passed with input_schema
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert 'tools' in call_kwargs
        assert call_kwargs['tools'][0]['name'] == 'get_weather'
        assert 'input_schema' in call_kwargs['tools'][0]

    @patch('anthropic.Anthropic')
    def test_extracts_function_calls_from_response(self, mock_client_class):
        """Should extract function calls from response."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(
            text=None,
            tool_use=[{
                "id": "toolu_abc123",
                "name": "get_weather",
                "input": {"location": "NYC"}
            }],
            stop_reason="tool_use"
        )
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        response = provider.send_message("Get weather")

        function_calls = response.get_function_calls()
        assert len(function_calls) == 1
        assert function_calls[0].name == "get_weather"
        assert function_calls[0].args == {"location": "NYC"}
        assert function_calls[0].id == "toolu_abc123"
        assert response.finish_reason == FinishReason.TOOL_USE

    @patch('anthropic.Anthropic')
    def test_send_tool_results(self, mock_client_class):
        """send_tool_results() should send results back to model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(
            text="The weather is sunny."
        )
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        results = [ToolResult(
            call_id="toolu_abc123",
            name="get_weather",
            result={"temp": 72, "condition": "sunny"}
        )]

        response = provider.send_tool_results(results)

        assert response.get_text() == "The weather is sunny."


class TestErrorHandling:
    """Tests for error handling."""

    @patch('anthropic.Anthropic')
    def test_handles_401_unauthorized(self, mock_client_class):
        """Should raise APIKeyInvalidError on 401."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("401 Unauthorized")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-bad"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        with pytest.raises(APIKeyInvalidError):
            provider.send_message("Test")

    @patch('anthropic.Anthropic')
    def test_handles_429_rate_limit(self, mock_client_class):
        """Should raise RateLimitError on 429."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("429 Rate limit exceeded")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        with pytest.raises(RateLimitError):
            provider.send_message("Test")

    @patch('anthropic.Anthropic')
    def test_handles_529_overloaded(self, mock_client_class):
        """Should raise OverloadedError on 529."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("529 Overloaded")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        with pytest.raises(OverloadedError):
            provider.send_message("Test")

    @patch('anthropic.Anthropic')
    def test_handles_context_length_error(self, mock_client_class):
        """Should raise ContextLimitError on context overflow."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("context_length_exceeded")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        with pytest.raises(ContextLimitError):
            provider.send_message("Test")


class TestTokenManagement:
    """Tests for token counting and context limits."""

    def test_count_tokens_estimate(self):
        """count_tokens() should return estimate when API unavailable."""
        provider = AnthropicProvider()

        # ~4 chars per token
        count = provider.count_tokens("Hello world!")  # 12 chars
        assert count == 3  # 12 // 4

    def test_get_context_limit_claude_4(self):
        """get_context_limit() should return correct limits for Claude 4."""
        provider = AnthropicProvider()
        provider._model_name = 'claude-sonnet-4-20250514'

        assert provider.get_context_limit() == 200_000

    def test_get_context_limit_claude_opus(self):
        """get_context_limit() should return correct limits for Claude Opus."""
        provider = AnthropicProvider()
        provider._model_name = 'claude-opus-4-5-20251101'

        assert provider.get_context_limit() == 200_000

    def test_get_context_limit_unknown_model(self):
        """get_context_limit() should return default for unknown models."""
        provider = AnthropicProvider()
        provider._model_name = 'unknown-model'

        assert provider.get_context_limit() == 200_000  # Default


class TestCapabilities:
    """Tests for capability checking."""

    def test_supports_structured_output_false(self):
        """Anthropic should not report native structured output support."""
        provider = AnthropicProvider()
        provider._model_name = 'claude-sonnet-4-20250514'

        assert provider.supports_structured_output() is False

    def test_supports_thinking_claude_4(self):
        """Claude 4 models should support thinking."""
        provider = AnthropicProvider()
        provider._model_name = 'claude-sonnet-4-20250514'

        assert provider.supports_thinking() is True

    def test_supports_thinking_claude_opus(self):
        """Claude Opus 4.5 should support thinking."""
        provider = AnthropicProvider()
        provider._model_name = 'claude-opus-4-5-20251101'

        assert provider.supports_thinking() is True


class TestSerialization:
    """Tests for history serialization."""

    def test_serialize_deserialize_history(self):
        """Should round-trip history through serialization."""
        provider = AnthropicProvider()

        history = [
            Message(role=Role.USER, parts=[Part(text="Hello")]),
            Message(role=Role.MODEL, parts=[Part(text="Hi there!")]),
        ]

        serialized = provider.serialize_history(history)
        deserialized = provider.deserialize_history(serialized)

        assert len(deserialized) == 2
        assert deserialized[0].role == Role.USER
        assert deserialized[0].parts[0].text == "Hello"
        assert deserialized[1].role == Role.MODEL
        assert deserialized[1].parts[0].text == "Hi there!"


class TestProviderProperties:
    """Tests for provider properties."""

    def test_name_property(self):
        """name property should return 'anthropic'."""
        provider = AnthropicProvider()
        assert provider.name == "anthropic"

    @patch('anthropic.Anthropic')
    def test_shutdown_cleans_up(self, mock_client_class):
        """shutdown() should clean up resources."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session()

        provider.shutdown()

        assert provider._client is None
        assert provider._model_name is None


class TestPromptCaching:
    """Tests for prompt caching feature."""

    @patch('anthropic.Anthropic')
    def test_caching_adds_cache_control_to_system(self, mock_client_class):
        """When caching enabled, system instruction should have cache_control."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={'enable_caching': True}
        ))
        provider.connect('claude-sonnet-4-20250514')
        provider.create_session(system_instruction="You are helpful.")

        provider.send_message("Hi")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system = call_kwargs['system']

        # Should be a list with cache_control
        assert isinstance(system, list)
        assert system[0]['cache_control'] == {"type": "ephemeral"}

    @patch('anthropic.Anthropic')
    def test_caching_adds_cache_control_to_tools(self, mock_client_class):
        """When caching enabled, last tool should have cache_control."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={'enable_caching': True}
        ))
        provider.connect('claude-sonnet-4-20250514')

        tools = [
            ToolSchema(name='tool1', description='First tool', parameters={}),
            ToolSchema(name='tool2', description='Second tool', parameters={}),
        ]
        provider.create_session(tools=tools)

        provider.send_message("Hi")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        tools_sent = call_kwargs['tools']

        # Last tool should have cache_control
        assert 'cache_control' in tools_sent[-1]
        assert tools_sent[-1]['cache_control'] == {"type": "ephemeral"}


class TestStreamingCapabilities:
    """Tests for streaming support."""

    def test_supports_streaming_true(self):
        """Anthropic should support streaming."""
        provider = AnthropicProvider()
        assert provider.supports_streaming() is True

    def test_supports_stop_true(self):
        """Anthropic should support stop (cancellation)."""
        provider = AnthropicProvider()
        assert provider.supports_stop() is True


class TestStreamingConverters:
    """Tests for streaming helper functions in converters."""

    def test_extract_text_from_stream_event(self):
        """Should extract text from text_delta event."""
        from ..converters import extract_text_from_stream_event

        # Create mock event
        event = MagicMock()
        event.type = "content_block_delta"
        event.delta = MagicMock()
        event.delta.type = "text_delta"
        event.delta.text = "Hello"

        result = extract_text_from_stream_event(event)
        assert result == "Hello"

    def test_extract_text_from_non_text_event(self):
        """Should return None for non-text events."""
        from ..converters import extract_text_from_stream_event

        event = MagicMock()
        event.type = "message_start"

        result = extract_text_from_stream_event(event)
        assert result is None

    def test_extract_content_block_start_tool_use(self):
        """Should extract tool_use block info."""
        from ..converters import extract_content_block_start

        event = MagicMock()
        event.type = "content_block_start"
        event.index = 0
        event.content_block = MagicMock()
        event.content_block.type = "tool_use"
        event.content_block.id = "toolu_123"
        event.content_block.name = "get_weather"

        result = extract_content_block_start(event)
        assert result["type"] == "tool_use"
        assert result["id"] == "toolu_123"
        assert result["name"] == "get_weather"

    def test_extract_message_start_usage(self):
        """Should extract usage from message_start event."""
        from ..converters import extract_message_start

        event = MagicMock()
        event.type = "message_start"
        event.message = MagicMock()
        event.message.usage = MagicMock()
        event.message.usage.input_tokens = 100
        event.message.usage.output_tokens = 10

        result = extract_message_start(event)
        assert result.prompt_tokens == 100
        assert result.output_tokens == 10
        assert result.total_tokens == 110
