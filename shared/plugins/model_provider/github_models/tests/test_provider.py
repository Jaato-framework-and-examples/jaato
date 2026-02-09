"""Tests for GitHubModelsProvider."""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from ..provider import GitHubModelsProvider, MODEL_CONTEXT_LIMITS
from ...base import ProviderConfig
from ...types import ProviderResponse, TokenUsage, FinishReason, FunctionCall, ToolResult
from ..errors import (
    TokenNotFoundError,
    TokenInvalidError,
    TokenPermissionError,
    ModelsDisabledError,
    ModelNotFoundError,
    RateLimitError,
)


def create_mock_response(
    text: str = "Hello!",
    tool_calls: list = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
):
    """Create a mock ChatCompletions response."""
    mock_response = MagicMock()

    # Create choice with message
    mock_choice = MagicMock()
    mock_choice.finish_reason = finish_reason
    mock_choice.message = MagicMock()
    mock_choice.message.content = text
    mock_choice.message.tool_calls = tool_calls or []

    mock_response.choices = [mock_choice]

    # Create usage
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = prompt_tokens + completion_tokens

    return mock_response


class TestAuthentication:
    """Tests for authentication and initialization."""

    def test_initialize_without_token_raises(self):
        """Should raise TokenNotFoundError if no token provided."""
        provider = GitHubModelsProvider()

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(TokenNotFoundError) as exc_info:
                provider.initialize(ProviderConfig())

        assert "GITHUB_TOKEN" in str(exc_info.value)

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_initialize_with_api_key(self, mock_client_class):
        """Should initialize with token from config.api_key."""
        mock_client_class.return_value = MagicMock()

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test_token"))

        assert provider._token == "ghp_test_token"
        mock_client_class.assert_called_once()

    @patch('azure.ai.inference.ChatCompletionsClient')
    @patch.dict('os.environ', {'GITHUB_TOKEN': 'ghp_env_token'}, clear=True)
    def test_initialize_from_env(self, mock_client_class):
        """Should auto-detect token from GITHUB_TOKEN env var."""
        mock_client_class.return_value = MagicMock()

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig())

        assert provider._token == "ghp_env_token"

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_initialize_with_organization(self, mock_client_class):
        """Should store organization for billing attribution."""
        mock_client_class.return_value = MagicMock()

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(
            api_key="ghp_test",
            extra={'organization': 'my-org'}
        ))

        assert provider._organization == "my-org"

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_initialize_with_enterprise(self, mock_client_class):
        """Should store enterprise for context."""
        mock_client_class.return_value = MagicMock()

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(
            api_key="ghp_test",
            extra={'enterprise': 'my-enterprise'}
        ))

        assert provider._enterprise == "my-enterprise"

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_initialize_with_custom_endpoint(self, mock_client_class):
        """Should use custom endpoint if provided."""
        mock_client_class.return_value = MagicMock()

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(
            api_key="ghp_test",
            extra={'endpoint': 'https://custom.endpoint.com'}
        ))

        assert provider._endpoint == "https://custom.endpoint.com"


class TestConnection:
    """Tests for model connection."""

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_connect_sets_model(self, mock_client_class):
        """connect() should set the model name."""
        mock_client_class.return_value = MagicMock()

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')

        assert provider.model_name == 'openai/gpt-4o'
        assert provider.is_connected is True

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_is_connected_false_without_model(self, mock_client_class):
        """is_connected should be False without model."""
        mock_client_class.return_value = MagicMock()

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))

        assert provider.is_connected is False

    def test_list_models_returns_known_models(self):
        """list_models() should return known model IDs."""
        provider = GitHubModelsProvider()
        models = provider.list_models()

        assert 'openai/gpt-4o' in models
        assert 'anthropic/claude-3.5-sonnet' in models

    def test_list_models_with_prefix_filter(self):
        """list_models() should filter by prefix."""
        provider = GitHubModelsProvider()

        openai_models = provider.list_models(prefix='openai/')
        assert all(m.startswith('openai/') for m in openai_models)

        anthropic_models = provider.list_models(prefix='anthropic/')
        assert all(m.startswith('anthropic/') for m in anthropic_models)


class TestMessaging:
    """Tests for sending messages."""

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_send_message_returns_response(self, mock_client_class):
        """send_message() should return ProviderResponse."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_response(text="Hello!")
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        response = provider.send_message("Hi")

        assert response.text == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.output_tokens == 20

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_send_message_adds_to_history(self, mock_client_class):
        """send_message() should add messages to history."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_response(text="Response")
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        provider.send_message("Question")
        history = provider.get_history()

        assert len(history) == 2  # User message + assistant response
        assert history[0].role.value == "user"
        assert history[1].role.value == "model"

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_send_message_with_system_instruction(self, mock_client_class):
        """send_message() should include system instruction."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session(system_instruction="You are helpful.")

        provider.send_message("Hi")

        # Verify system message was included
        call_args = mock_client.complete.call_args
        messages = call_args.kwargs.get('messages', call_args.args[0] if call_args.args else [])

        # First message should be system
        from azure.ai.inference.models import SystemMessage
        assert any(isinstance(m, SystemMessage) for m in messages)

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_generate_one_shot(self, mock_client_class):
        """generate() should work without session."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_response(text="Generated")
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')

        response = provider.generate("One-shot prompt")

        assert response.text == "Generated"


class TestFunctionCalling:
    """Tests for function calling support."""

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_send_message_with_tools(self, mock_client_class):
        """send_message() should pass tools to API."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        from ...types import ToolSchema

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')

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

        # Verify tools were passed
        call_args = mock_client.complete.call_args
        assert 'tools' in call_args.kwargs

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_extracts_function_calls_from_response(self, mock_client_class):
        """Should extract function calls from response."""
        # Create mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "NYC"}'

        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_response(
            text=None,
            tool_calls=[mock_tool_call],
            finish_reason="tool_calls"
        )
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        response = provider.send_message("Get weather")

        assert len(response.function_calls) == 1
        assert response.function_calls[0].name == "get_weather"
        assert response.function_calls[0].args == {"location": "NYC"}
        assert response.finish_reason == FinishReason.TOOL_USE

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_send_tool_results(self, mock_client_class):
        """send_tool_results() should send results back to model."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_response(
            text="The weather is sunny."
        )
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        results = [ToolResult(
            call_id="call_123",
            name="get_weather",
            result={"temp": 72, "condition": "sunny"}
        )]

        response = provider.send_tool_results(results)

        assert response.text == "The weather is sunny."


class TestErrorHandling:
    """Tests for error handling."""

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_handles_401_unauthorized(self, mock_client_class):
        """Should raise TokenInvalidError on 401."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("401 Unauthorized")
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_bad"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        with pytest.raises(TokenInvalidError):
            provider.send_message("Test")

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_handles_403_forbidden(self, mock_client_class):
        """Should raise TokenPermissionError on 403."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("403 Forbidden")
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        with pytest.raises(TokenPermissionError):
            provider.send_message("Test")

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_handles_429_rate_limit(self, mock_client_class):
        """Should raise RateLimitError on 429."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("429 Rate limit exceeded")
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        with pytest.raises(RateLimitError):
            provider.send_message("Test")

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_handles_404_model_not_found(self, mock_client_class):
        """Should raise ModelNotFoundError on 404."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("404 Not found")
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('invalid/model')
        provider.create_session()

        with pytest.raises(ModelNotFoundError) as exc_info:
            provider.send_message("Test")

        assert "invalid/model" in str(exc_info.value)

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_handles_models_disabled(self, mock_client_class):
        """Should raise ModelsDisabledError when GitHub Models disabled."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception(
            "401 GitHub Models is disabled"
        )
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(
            api_key="ghp_test",
            extra={'organization': 'my-org'}
        ))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        with pytest.raises(ModelsDisabledError) as exc_info:
            provider.send_message("Test")

        assert "my-org" in str(exc_info.value)


class TestTokenManagement:
    """Tests for token counting and context limits."""

    def test_count_tokens_estimate(self):
        """count_tokens() should return estimate."""
        provider = GitHubModelsProvider()

        # ~4 chars per token
        count = provider.count_tokens("Hello world!")  # 12 chars
        assert count == 3  # 12 // 4

    def test_get_context_limit_known_model(self):
        """get_context_limit() should return known limits."""
        provider = GitHubModelsProvider()
        provider._model_name = 'openai/gpt-4o'

        assert provider.get_context_limit() == 128_000

    def test_get_context_limit_anthropic_model(self):
        """get_context_limit() should return Anthropic limits."""
        provider = GitHubModelsProvider()
        provider._model_name = 'anthropic/claude-3.5-sonnet'

        assert provider.get_context_limit() == 200_000

    def test_get_context_limit_unknown_model(self):
        """get_context_limit() should return default for unknown."""
        provider = GitHubModelsProvider()
        provider._model_name = 'unknown/model'

        assert provider.get_context_limit() == 128_000  # Default


class TestStructuredOutput:
    """Tests for structured output support."""

    def test_supports_structured_output_openai(self):
        """OpenAI models should support structured output."""
        provider = GitHubModelsProvider()
        provider._model_name = 'openai/gpt-4o'

        assert provider.supports_structured_output() is True

    def test_supports_structured_output_anthropic(self):
        """Anthropic models should not support structured output (via this API)."""
        provider = GitHubModelsProvider()
        provider._model_name = 'anthropic/claude-3.5-sonnet'

        assert provider.supports_structured_output() is False

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_send_message_with_response_schema(self, mock_client_class):
        """send_message() should handle response_schema."""
        mock_client = MagicMock()
        mock_client.complete.return_value = create_mock_response(
            text='{"name": "Alice", "age": 30}'
        )
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }

        response = provider.send_message("Tell me about Alice", response_schema=schema)

        assert response.structured_output == {"name": "Alice", "age": 30}


class TestSerialization:
    """Tests for history serialization."""

    def test_serialize_deserialize_history(self):
        """Should round-trip history through serialization."""
        from ...types import Message, Part, Role

        provider = GitHubModelsProvider()

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
        """name property should return 'github_models'."""
        provider = GitHubModelsProvider()
        assert provider.name == "github_models"

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_shutdown_cleans_up(self, mock_client_class):
        """shutdown() should clean up resources."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        provider.shutdown()

        assert provider._client is None
        assert provider._model_name is None
        mock_client.close.assert_called_once()


class TestThinkingReasoning:
    """Tests for thinking/reasoning content extraction."""

    def test_supports_thinking_deepseek_r1(self):
        """DeepSeek-R1 models should support thinking."""
        provider = GitHubModelsProvider()
        provider._model_name = 'deepseek/deepseek-r1'
        assert provider.supports_thinking() is True

    def test_supports_thinking_openai_model(self):
        """OpenAI models should not support thinking (reasoning is hidden)."""
        provider = GitHubModelsProvider()
        provider._model_name = 'openai/gpt-4o'
        assert provider.supports_thinking() is False

    def test_supports_thinking_o1(self):
        """OpenAI o1 should not support thinking (hidden reasoning)."""
        provider = GitHubModelsProvider()
        provider._model_name = 'openai/o1-preview'
        assert provider.supports_thinking() is False

    def test_set_thinking_config(self):
        """set_thinking_config should update _enable_thinking."""
        from ...types import ThinkingConfig

        provider = GitHubModelsProvider()
        assert provider._enable_thinking is True  # Default

        provider.set_thinking_config(ThinkingConfig(enabled=False))
        assert provider._enable_thinking is False

        provider.set_thinking_config(ThinkingConfig(enabled=True))
        assert provider._enable_thinking is True

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_non_streaming_extracts_reasoning(self, mock_client_class):
        """Non-streaming response should extract reasoning_content."""
        mock_response = create_mock_response(text="The answer is 42.")
        # Add reasoning_content to the mock message
        mock_response.choices[0].message.reasoning_content = (
            "Let me think step by step..."
        )

        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('deepseek/deepseek-r1')
        provider.create_session()

        response = provider.send_message("What is the meaning of life?")

        assert response.get_text() == "The answer is 42."
        assert response.thinking == "Let me think step by step..."
        assert response.has_thinking is True

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_non_streaming_no_reasoning_when_absent(self, mock_client_class):
        """Non-streaming response without reasoning should have thinking=None."""
        mock_response = create_mock_response(text="Hello!")
        # Explicitly set reasoning_content to None (MagicMock auto-generates
        # truthy attributes otherwise)
        mock_response.choices[0].message.reasoning_content = None

        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('openai/gpt-4o')
        provider.create_session()

        response = provider.send_message("Hi")

        assert response.get_text() == "Hello!"
        assert response.thinking is None
        assert response.has_thinking is False

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_non_streaming_reasoning_disabled(self, mock_client_class):
        """Reasoning should not be extracted when thinking is disabled."""
        from ...types import ThinkingConfig

        mock_response = create_mock_response(text="The answer is 42.")
        mock_response.choices[0].message.reasoning_content = (
            "Let me think step by step..."
        )

        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('deepseek/deepseek-r1')
        provider.create_session()
        provider.set_thinking_config(ThinkingConfig(enabled=False))

        response = provider.send_message("What is the meaning of life?")

        assert response.get_text() == "The answer is 42."
        # The Azure SDK path uses response_from_sdk which always extracts
        # reasoning_content (it doesn't check _enable_thinking since the
        # converter is stateless). The flag controls streaming extraction.
        assert response.thinking == "Let me think step by step..."

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_streaming_extracts_reasoning(self, mock_client_class):
        """Streaming should extract reasoning_content from deltas."""
        # Create mock streaming chunks
        chunks = []

        # Reasoning chunk
        reasoning_chunk = MagicMock()
        reasoning_chunk.choices = [MagicMock()]
        reasoning_chunk.choices[0].delta = MagicMock()
        reasoning_chunk.choices[0].delta.reasoning_content = "Thinking about it..."
        reasoning_chunk.choices[0].delta.content = None
        reasoning_chunk.choices[0].delta.tool_calls = None
        reasoning_chunk.choices[0].finish_reason = None
        reasoning_chunk.usage = None
        chunks.append(reasoning_chunk)

        # Another reasoning chunk
        reasoning_chunk2 = MagicMock()
        reasoning_chunk2.choices = [MagicMock()]
        reasoning_chunk2.choices[0].delta = MagicMock()
        reasoning_chunk2.choices[0].delta.reasoning_content = " Still thinking."
        reasoning_chunk2.choices[0].delta.content = None
        reasoning_chunk2.choices[0].delta.tool_calls = None
        reasoning_chunk2.choices[0].finish_reason = None
        reasoning_chunk2.usage = None
        chunks.append(reasoning_chunk2)

        # Content chunk
        content_chunk = MagicMock()
        content_chunk.choices = [MagicMock()]
        content_chunk.choices[0].delta = MagicMock()
        content_chunk.choices[0].delta.reasoning_content = None
        content_chunk.choices[0].delta.content = "The answer."
        content_chunk.choices[0].delta.tool_calls = None
        content_chunk.choices[0].finish_reason = "stop"
        content_chunk.usage = None
        chunks.append(content_chunk)

        mock_client = MagicMock()
        mock_client.complete.return_value = iter(chunks)
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('deepseek/deepseek-r1')
        provider.create_session()

        text_chunks = []
        thinking_chunks = []
        response = provider.send_message_streaming(
            "Question?",
            on_chunk=lambda c: text_chunks.append(c),
            on_thinking=lambda t: thinking_chunks.append(t),
        )

        assert response.get_text() == "The answer."
        assert response.thinking == "Thinking about it... Still thinking."
        assert thinking_chunks == ["Thinking about it...", " Still thinking."]
        assert text_chunks == ["The answer."]

    @patch('azure.ai.inference.ChatCompletionsClient')
    def test_streaming_no_reasoning_when_disabled(self, mock_client_class):
        """Streaming should not extract reasoning when thinking is disabled."""
        from ...types import ThinkingConfig

        # Chunk with reasoning
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.reasoning_content = "Hidden thought"
        chunk.choices[0].delta.content = "Visible answer"
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = "stop"
        chunk.usage = None

        mock_client = MagicMock()
        mock_client.complete.return_value = iter([chunk])
        mock_client_class.return_value = mock_client

        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(api_key="ghp_test"))
        provider.connect('deepseek/deepseek-r1')
        provider.create_session()
        provider.set_thinking_config(ThinkingConfig(enabled=False))

        thinking_chunks = []
        response = provider.send_message_streaming(
            "Question?",
            on_chunk=lambda c: None,
            on_thinking=lambda t: thinking_chunks.append(t),
        )

        assert response.get_text() == "Visible answer"
        assert response.thinking is None
        assert thinking_chunks == []
