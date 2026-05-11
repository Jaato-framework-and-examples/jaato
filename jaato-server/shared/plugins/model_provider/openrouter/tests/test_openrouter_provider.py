"""Tests for the OpenRouter provider."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ..converters import (
    deserialize_history,
    get_original_tool_name,
    map_finish_reason,
    message_from_openai,
    message_to_openai,
    response_from_openai,
    sanitize_tool_name,
    serialize_history,
    tool_schema_to_openai,
)
from ..env import (
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_LENGTH,
    HEADER_APP_TITLE,
    HEADER_HTTP_REFERER,
    resolve_api_key,
    resolve_app_title,
    resolve_base_url,
    resolve_context_length,
    resolve_http_referer,
)
from ..errors import (
    APIKeyNotFoundError,
    InfrastructureError,
    RateLimitError,
)
from ..provider import OpenRouterProvider, create_provider
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
    ToolSchema,
)


# ==================== Helpers ====================


def create_mock_response(
    text="Hello!",
    tool_calls=None,
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=20,
    reasoning=None,
    reasoning_content=None,
):
    """Create a mock OpenAI ChatCompletion response."""
    mock_response = MagicMock()

    mock_choice = MagicMock()
    mock_choice.finish_reason = finish_reason
    mock_choice.message = MagicMock()
    mock_choice.message.content = text
    mock_choice.message.tool_calls = tool_calls or []
    mock_choice.message.reasoning = reasoning
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
        with patch.dict("os.environ", {"JAATO_OPENROUTER_API_KEY": "sk-or-test123"}):
            assert resolve_api_key() == "sk-or-test123"

    def test_resolve_api_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.openrouter.auth.get_stored_api_key",
                return_value=None,
            ):
                assert resolve_api_key() is None

    def test_resolve_base_url_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL

    def test_resolve_base_url_from_env(self):
        with patch.dict(
            "os.environ", {"JAATO_OPENROUTER_BASE_URL": "https://custom/api/v1"}
        ):
            assert resolve_base_url() == "https://custom/api/v1"

    def test_resolve_context_length_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_context_length() == DEFAULT_CONTEXT_LENGTH

    def test_resolve_context_length_from_env(self):
        with patch.dict("os.environ", {"JAATO_OPENROUTER_CONTEXT_LENGTH": "131072"}):
            assert resolve_context_length() == 131072

    def test_resolve_context_length_invalid(self):
        with patch.dict(
            "os.environ", {"JAATO_OPENROUTER_CONTEXT_LENGTH": "not-a-number"}
        ):
            assert resolve_context_length() == DEFAULT_CONTEXT_LENGTH

    def test_resolve_http_referer_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert "jaato" in resolve_http_referer().lower()

    def test_resolve_http_referer_from_env(self):
        with patch.dict(
            "os.environ", {"JAATO_OPENROUTER_HTTP_REFERER": "https://example.com"}
        ):
            assert resolve_http_referer() == "https://example.com"

    def test_resolve_app_title_from_env(self):
        with patch.dict("os.environ", {"JAATO_OPENROUTER_APP_TITLE": "MyApp"}):
            assert resolve_app_title() == "MyApp"

    def test_attribution_header_names(self):
        # Verify we use the OpenRouter-canonical header names.
        assert HEADER_HTTP_REFERER == "HTTP-Referer"
        assert HEADER_APP_TITLE == "X-OpenRouter-Title"


# ==================== Converter Tests ====================


class TestToolNameMapping:
    """Tests for hash-derived tool name IDs and reverse mapping."""

    def test_returns_hash_id(self):
        result = sanitize_tool_name("my_tool")
        assert result.startswith("t_")
        assert len(result) == 10

    def test_deterministic(self):
        assert sanitize_tool_name("my_tool") == sanitize_tool_name("my_tool")

    def test_different_names_produce_different_ids(self):
        assert sanitize_tool_name("read_file") != sanitize_tool_name("write_file")

    def test_reverse_mapping(self):
        tool_id = sanitize_tool_name("mcp.server.tool")
        assert get_original_tool_name(tool_id) == "mcp.server.tool"


class TestToolSchemaConversion:
    """Tests for ToolSchema to OpenAI format conversion."""

    def test_basic_schema(self):
        schema = ToolSchema(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        result = tool_schema_to_openai(schema)

        assert result["type"] == "function"
        assert result["function"]["name"] == sanitize_tool_name("read_file")
        assert result["function"]["description"] == "Read a file"
        assert result["function"]["parameters"]["type"] == "object"


class TestMessageConversion:
    """Tests for Message <-> OpenAI format conversion."""

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
        assert (
            result["tool_calls"][0]["function"]["name"]
            == sanitize_tool_name("read_file")
        )

    def test_tool_result_message(self):
        tr = ToolResult(call_id="call_1", name="read_file", result={"x": 1})
        msg = Message(role=Role.TOOL, parts=[Part(function_response=tr)])
        result = message_to_openai(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert json.loads(result["content"]) == {"x": 1}


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
        mock.choices[0].message.content = None

        result = response_from_openai(mock)
        assert result.finish_reason == FinishReason.TOOL_USE
        fc_parts = [p for p in result.parts if p.function_call]
        assert len(fc_parts) == 1
        assert fc_parts[0].function_call.name == "read_file"

    def test_reasoning_extraction_via_reasoning_field(self):
        mock = create_mock_response(text="Answer", reasoning="Let me think...")
        result = response_from_openai(mock)
        assert result.thinking == "Let me think..."

    def test_reasoning_extraction_via_reasoning_content_field(self):
        # Some upstreams (passed through by OpenRouter) use the legacy spelling.
        mock = create_mock_response(text="Answer", reasoning_content="Thinking...")
        # Drop the new-style field so the fallback triggers.
        mock.choices[0].message.reasoning = None
        result = response_from_openai(mock)
        assert result.thinking == "Thinking..."


class TestFinishReasonMapping:
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
    def test_roundtrip_text_messages(self):
        history = [
            Message.from_text(Role.USER, "Hello"),
            Message(role=Role.MODEL, parts=[Part(text="Hi")]),
        ]
        data = serialize_history(history)
        restored = deserialize_history(data)
        assert len(restored) == 2
        assert restored[0].role == Role.USER
        assert restored[1].role == Role.MODEL
        assert restored[1].parts[0].text == "Hi"

    def test_roundtrip_function_call(self):
        fc = FunctionCall(id="call_1", name="test", args={"k": "v"})
        history = [Message(role=Role.MODEL, parts=[Part(function_call=fc)])]
        data = serialize_history(history)
        restored = deserialize_history(data)
        assert restored[0].parts[0].function_call.name == "test"

    def test_roundtrip_tool_result(self):
        tr = ToolResult(call_id="c1", name="t", result="out", is_error=False)
        history = [Message(role=Role.TOOL, parts=[Part(function_response=tr)])]
        data = serialize_history(history)
        restored = deserialize_history(data)
        assert restored[0].parts[0].function_response.result == "out"


# ==================== Provider Tests ====================


class TestAuthentication:
    """Tests for authentication and initialization."""

    def test_initialize_without_key_raises(self):
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.openrouter.auth.get_stored_api_key",
                return_value=None,
            ):
                with pytest.raises(APIKeyNotFoundError) as exc_info:
                    provider.initialize(ProviderConfig())
        assert "JAATO_OPENROUTER_API_KEY" in str(exc_info.value)

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_initialize_with_api_key(self, mock_client_class):
        mock_client_class.return_value = MagicMock()

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))

        assert provider._api_key == "sk-or-test"
        assert provider._client is not None

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_OPENROUTER_API_KEY": "sk-or-env"}, clear=True)
    def test_initialize_from_env(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig())
        assert provider._api_key == "sk-or-env"

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_initialize_custom_base_url(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"base_url": "https://custom/api/v1"},
        ))
        assert provider._base_url == "https://custom/api/v1"

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_initialize_custom_context_length_marks_override(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"context_length": 200000},
        ))
        assert provider._context_length == 200000
        assert provider._context_length_override is True

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_initialize_passes_attribution_headers(self, mock_client_class):
        captured = {}

        def fake_client_class(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        mock_client_class.return_value = fake_client_class

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={
                "http_referer": "https://example.com",
                "app_title": "Example App",
            },
        ))

        headers = captured.get("default_headers") or {}
        assert headers.get(HEADER_HTTP_REFERER) == "https://example.com"
        assert headers.get(HEADER_APP_TITLE) == "Example App"


class TestProviderRouting:
    """Tests for the ``provider`` request-routing dict.

    OpenRouter's killer feature: pin / blacklist / sort upstream
    providers, require non-training upstreams, etc.  The dict is read
    from ``ProviderConfig.extra['provider']`` (which the runtime sources
    from ``plugin_configs.openrouter.provider``) and forwarded to every
    request via the OpenAI SDK's ``extra_body`` parameter.
    """

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_initialize_stores_provider_routing(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        routing = {
            "sort": "price",
            "data_collection": "deny",
            "ignore": ["Groq"],
            "order": ["Fireworks", "DeepInfra"],
            "allow_fallbacks": True,
        }
        provider.initialize(ProviderConfig(api_key="sk-or-test", extra={"provider": routing}))
        assert provider._provider_routing == routing
        # Defensive copy — mutating the source dict mustn't affect us.
        routing["ignore"].append("Together")
        assert provider._provider_routing["ignore"] == ["Groq"]

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_initialize_no_provider_routing_means_none(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        assert provider._provider_routing is None
        # Detailed-usage opt-in is unconditional so the response carries
        # cached_tokens / cost regardless of which model is selected.
        assert provider._build_extra_body() == {"usage": {"include": True}}

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_initialize_rejects_non_dict_provider(self, mock_client_class):
        # Legacy flat ``provider:`` key still accepted (with a deprecation
        # warning) for one release; type validation message references
        # the new namespacing key (``routing``).
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(TypeError, match="routing.*must be a dict"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"provider": ["Anthropic", "OpenAI"]},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_initialize_rejects_non_dict_routing(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(TypeError, match="routing.*must be a dict"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"routing": ["Anthropic", "OpenAI"]},
            ))

    def test_build_extra_body_includes_provider(self):
        provider = OpenRouterProvider()
        provider._provider_routing = {"sort": "throughput"}
        assert provider._build_extra_body() == {
            "provider": {"sort": "throughput"},
            "usage": {"include": True},
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_complete_forwards_provider_routing_via_extra_body(self, mock_client_class):
        # Capture the kwargs passed to chat.completions.create.
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = create_mock_response(
            text="ok", finish_reason="stop"
        )
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"provider": {"sort": "price", "data_collection": "deny"}},
        ))
        provider.connect("anthropic/claude-3.5-sonnet", skip_model_test=True)
        provider.complete([Message.from_text(Role.USER, "hi")])

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "provider": {"sort": "price", "data_collection": "deny"},
            "usage": {"include": True},
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_complete_always_sends_usage_opt_in(self, mock_client_class):
        # Detailed-usage opt-in is unconditional — without it OpenRouter
        # omits ``cost`` and ``cache_creation_input_tokens`` from the
        # response, so the daemon's per-turn ledger would lose savings.
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = create_mock_response(
            text="ok", finish_reason="stop"
        )
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        provider.connect("openai/gpt-4o", skip_model_test=True)
        provider.complete([Message.from_text(Role.USER, "hi")])

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"usage": {"include": True}}

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_streaming_path_also_forwards_provider_routing(self, mock_client_class):
        fake_client = MagicMock()
        # Streaming returns an iterable of chunks; an empty list is fine
        # — we only care about what was passed in.
        fake_client.chat.completions.create.return_value = iter([])
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"provider": {"order": ["Fireworks"]}},
        ))
        provider.connect("meta-llama/llama-3.3-70b-instruct", skip_model_test=True)

        chunks = []
        provider.complete(
            [Message.from_text(Role.USER, "hi")],
            on_chunk=chunks.append,
        )

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "provider": {"order": ["Fireworks"]},
            "usage": {"include": True},
        }
        # And the streaming flag must still be set on the same call.
        assert call_kwargs.get("stream") is True


class TestThinkingKnobs:
    """Tests for the flat-key thinking convention (matches Anthropic / Antigravity).

    OpenRouter accepts ``enable_thinking`` (bool), ``thinking_budget`` (int)
    and ``thinking_level`` (low/medium/high) under
    ``plugin_configs.openrouter`` — the same key names other thinking-capable
    providers already accept — and translates them into OpenRouter's
    ``reasoning`` request body shape (``effort`` / ``max_tokens``).
    """

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_disabled_by_default(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        assert provider._enable_thinking is False
        # ``usage.include`` is always on; no ``reasoning`` block when disabled.
        assert provider._build_extra_body() == {"usage": {"include": True}}

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_thinking_budget_emits_max_tokens(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"enable_thinking": True, "thinking_budget": 8192},
        ))
        assert provider._build_extra_body() == {
            "reasoning": {"max_tokens": 8192},
            "usage": {"include": True},
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_thinking_level_emits_effort(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"enable_thinking": True, "thinking_level": "high"},
        ))
        assert provider._build_extra_body() == {
            "reasoning": {"effort": "high"},
            "usage": {"include": True},
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_level_takes_precedence_over_budget(self, mock_client_class):
        # ``effort`` is more portable across upstreams (OpenAI o-series doesn't
        # accept arbitrary token caps), so when both are set we prefer it.
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={
                "enable_thinking": True,
                "thinking_budget": 8192,
                "thinking_level": "medium",
            },
        ))
        assert provider._build_extra_body() == {
            "reasoning": {"effort": "medium"},
            "usage": {"include": True},
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_disabled_suppresses_reasoning_even_with_budget(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"enable_thinking": False, "thinking_budget": 8192},
        ))
        assert "reasoning" not in provider._build_extra_body()

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_invalid_level_raises(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="thinking_level.*low.*medium.*high"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"enable_thinking": True, "thinking_level": "extreme"},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_complete_forwards_reasoning_via_extra_body(self, mock_client_class):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = create_mock_response(
            text="ok", finish_reason="stop"
        )
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"enable_thinking": True, "thinking_level": "low"},
        ))
        provider.connect("openai/o1-preview", skip_model_test=True)
        provider.complete([Message.from_text(Role.USER, "hi")])

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "reasoning": {"effort": "low"},
            "usage": {"include": True},
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_provider_routing_and_reasoning_compose_in_extra_body(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={
                "provider": {"sort": "price"},
                "enable_thinking": True,
                "thinking_budget": 4096,
            },
        ))
        body = provider._build_extra_body()
        assert body == {
            "provider": {"sort": "price"},
            "reasoning": {"max_tokens": 4096},
            "usage": {"include": True},
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_set_thinking_config_runtime_update(self, mock_client_class):
        # Framework-level ThinkingConfig updates should also flow.
        from jaato_sdk.plugins.model_provider.types import ThinkingConfig as TC
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        assert provider._build_extra_body() == {"usage": {"include": True}}

        provider.set_thinking_config(TC(enabled=True, budget=2048))
        assert provider._build_extra_body() == {
            "reasoning": {"max_tokens": 2048},
            "usage": {"include": True},
        }

        provider.set_thinking_config(TC(enabled=False, budget=2048))
        assert "reasoning" not in provider._build_extra_body()

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_set_thinking_config_keeps_profile_level(self, mock_client_class):
        # thinking_level has no equivalent in ThinkingConfig, so a runtime
        # update without budget mustn't wipe the profile-set level.
        from jaato_sdk.plugins.model_provider.types import ThinkingConfig as TC
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"enable_thinking": True, "thinking_level": "high"},
        ))
        provider.set_thinking_config(TC(enabled=True, budget=0))
        # effort still wins because thinking_level is still set.
        assert provider._build_extra_body() == {
            "reasoning": {"effort": "high"},
            "usage": {"include": True},
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_disabled_drops_thinking_from_batch_response(self, mock_client_class):
        # DeepSeek-R1 always reasons regardless of request — when the user
        # disables thinking we still drop it on the way out so they don't
        # see chain-of-thought they didn't ask for.
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = create_mock_response(
            text="answer", reasoning="chain of thought", finish_reason="stop"
        )
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))  # disabled
        provider.connect("deepseek/deepseek-r1", skip_model_test=True)
        result = provider.complete([Message.from_text(Role.USER, "hi")])
        assert result.response.thinking is None


class TestConfigNamespacing:
    """Tests for the four-layer config namespacing introduced in 0.6.23.

    Layers (under ``plugin_configs.openrouter``):
      - Top-level: api_key / http_referer / app_title
      - api_params: temperature / top_p / top_k / enable_thinking / ...
      - routing: OpenRouter ``provider`` extension dict
      - framework_overrides: context_length / base_url

    Backward compatibility: the same keys are also read from the legacy
    flat position with a deprecation warning.
    """

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_new_shape_no_deprecation_warning(self, mock_client_class, caplog):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with caplog.at_level("WARNING"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={
                    "api_params": {"temperature": 0.55, "top_p": 1.0},
                    "routing": {"sort": "throughput", "ignore": ["AtlasCloud"]},
                    "framework_overrides": {"context_length": 32768},
                },
            ))
        assert provider._temperature == 0.55
        assert provider._top_p == 1.0
        assert provider._provider_routing == {"sort": "throughput", "ignore": ["AtlasCloud"]}
        assert provider._context_length == 32768
        assert provider._context_length_override is True
        legacy_warnings = [r for r in caplog.records if "legacy" in r.getMessage().lower()]
        assert legacy_warnings == [], (
            f"new shape should not emit deprecation warnings, got: "
            f"{[r.getMessage() for r in legacy_warnings]}"
        )

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_legacy_flat_shape_still_works_with_warnings(
        self, mock_client_class, caplog,
    ):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with caplog.at_level("WARNING"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={
                    "temperature": 0.55,
                    "top_p": 1.0,
                    "provider": {"sort": "throughput"},
                    "context_length": 32768,
                },
            ))
        assert provider._temperature == 0.55
        assert provider._top_p == 1.0
        assert provider._provider_routing == {"sort": "throughput"}
        assert provider._context_length == 32768
        legacy_warnings = [r for r in caplog.records if "legacy" in r.getMessage().lower()]
        assert len(legacy_warnings) >= 4, (
            f"expected ≥4 deprecation warnings (temperature/top_p/provider/context_length), "
            f"got {len(legacy_warnings)}"
        )

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_new_layer_wins_over_legacy_flat_key(self, mock_client_class):
        # When both the nested layer key and the legacy flat key are
        # present, the nested form wins (legacy is the deprecation
        # fallback only).
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={
                "api_params": {"temperature": 0.7},
                "temperature": 0.3,
                "routing": {"sort": "price"},
                "provider": {"sort": "throughput"},
            },
        ))
        assert provider._temperature == 0.7
        assert provider._provider_routing == {"sort": "price"}

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_thinking_knobs_in_api_params(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={
                "api_params": {
                    "enable_thinking": True,
                    "thinking_budget": 16384,
                    "thinking_level": "high",
                },
            },
        ))
        assert provider._enable_thinking is True
        assert provider._thinking_budget == 16384
        assert provider._thinking_level == "high"


class TestVerifyAuth:
    """Tests for verify_auth (must work before initialize)."""

    def test_verify_auth_with_env_key(self):
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {"JAATO_OPENROUTER_API_KEY": "sk-or-test"}):
            assert provider.verify_auth() is True

    def test_verify_auth_with_profile_key(self):
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.openrouter.auth.try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                cfg = ProviderConfig(extra={"api_key": "sk-or-profile"})
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_no_key_raises(self):
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.openrouter.auth.try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                with pytest.raises(APIKeyNotFoundError):
                    provider.verify_auth(allow_interactive=False)

    def test_verify_auth_no_key_returns_false(self):
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.openrouter.auth.try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(allow_interactive=True) is False

    def test_verify_auth_surfaces_broken_credentials(self):
        provider = OpenRouterProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.openrouter.auth.try_load_credentials_with_reason",
                return_value=(None, "invalid JSON at /tmp/openrouter_auth.json"),
            ):
                messages = []
                with pytest.raises(APIKeyNotFoundError):
                    provider.verify_auth(on_message=messages.append)
                joined = "\n".join(messages)
                assert "could not be loaded" in joined
                assert "invalid JSON" in joined


class TestConnection:
    """Tests for connect and catalog lookup."""

    def test_connect_sets_model(self):
        provider = OpenRouterProvider()
        provider._client = MagicMock()
        # Skip the catalog network call.
        provider.connect("anthropic/claude-3.5-sonnet", skip_model_test=True)
        assert provider.model_name == "anthropic/claude-3.5-sonnet"

    def test_is_connected(self):
        provider = OpenRouterProvider()
        assert provider.is_connected is False

        provider._client = MagicMock()
        assert provider.is_connected is False

        provider._model_name = "openai/gpt-4o"
        assert provider.is_connected is True

    def test_connect_uses_catalog_context_length(self):
        provider = OpenRouterProvider()
        provider._client = MagicMock()
        # Pretend the catalog reports 200k for this model.
        provider._catalog_cache = [
            {"id": "anthropic/claude-3.5-sonnet", "context_length": 200000},
        ]
        provider.connect("anthropic/claude-3.5-sonnet")
        assert provider.get_context_limit() == 200000

    def test_connect_keeps_explicit_override(self):
        provider = OpenRouterProvider()
        provider._client = MagicMock()
        provider._context_length = 50000
        provider._context_length_override = True
        provider._catalog_cache = [
            {"id": "x/y", "context_length": 200000},
        ]
        provider.connect("x/y")
        assert provider.get_context_limit() == 50000


class TestListModels:
    """Tests for list_models against the OpenRouter catalog."""

    def test_list_models_returns_catalog_ids_sorted(self):
        provider = OpenRouterProvider()
        provider._catalog_cache = [
            {"id": "openai/gpt-4o", "context_length": 128000},
            {"id": "anthropic/claude-3.5-sonnet", "context_length": 200000},
            {"id": "meta-llama/llama-3.3-70b-instruct", "context_length": 131072},
        ]
        models = provider.list_models()
        assert models == sorted(models)
        assert "openai/gpt-4o" in models

    def test_list_models_prefix_filter(self):
        provider = OpenRouterProvider()
        provider._catalog_cache = [
            {"id": "openai/gpt-4o"},
            {"id": "anthropic/claude-3.5-sonnet"},
            {"id": "openai/o1-preview"},
        ]
        models = provider.list_models(prefix="openai/")
        assert models == ["openai/gpt-4o", "openai/o1-preview"]

    def test_list_models_empty_on_network_failure(self):
        provider = OpenRouterProvider()
        with patch("httpx.get", side_effect=Exception("boom")):
            assert provider.list_models() == []


class TestCapabilities:
    """Tests for capability queries."""

    def test_supports_streaming(self):
        assert OpenRouterProvider().supports_streaming() is True

    def test_supports_structured_output(self):
        assert OpenRouterProvider().supports_structured_output() is True

    def test_supports_stop(self):
        assert OpenRouterProvider().supports_stop() is True

    def test_supports_thinking_default(self):
        assert OpenRouterProvider().supports_thinking() is False

    def test_supports_thinking_for_known_reasoner(self):
        provider = OpenRouterProvider()
        provider._model_name = "deepseek/deepseek-r1"
        assert provider.supports_thinking() is True

        provider._model_name = "openai/o1-preview"
        assert provider.supports_thinking() is True

    def test_name(self):
        assert OpenRouterProvider().name == "openrouter"


class TestTokenManagement:
    """Tests for token counting and context limits."""

    def test_count_tokens(self):
        provider = OpenRouterProvider()
        assert provider.count_tokens("abcd") == 1
        assert provider.count_tokens("a" * 100) == 25

    def test_get_context_limit(self):
        provider = OpenRouterProvider()
        provider._context_length = 131072
        assert provider.get_context_limit() == 131072


class TestErrorClassification:
    """Tests for error classification and retry logic."""

    def test_classify_rate_limit(self):
        provider = OpenRouterProvider()
        exc = RateLimitError(original_error="429")
        assert provider.classify_error(exc) == {
            "transient": True,
            "rate_limit": True,
            "infra": False,
        }

    def test_classify_infrastructure(self):
        provider = OpenRouterProvider()
        exc = InfrastructureError(status_code=500)
        assert provider.classify_error(exc) == {
            "transient": True,
            "rate_limit": False,
            "infra": True,
        }

    def test_classify_unknown(self):
        provider = OpenRouterProvider()
        assert provider.classify_error(ValueError("unknown")) is None

    def test_retry_after_rate_limit(self):
        provider = OpenRouterProvider()
        assert provider.get_retry_after(RateLimitError(retry_after=30.0)) == 30.0


class TestCreateProvider:
    """Tests for the factory function."""

    def test_create_provider(self):
        provider = create_provider()
        assert isinstance(provider, OpenRouterProvider)
        assert provider.name == "openrouter"


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_clears_state(self):
        provider = OpenRouterProvider()
        provider._client = MagicMock()
        provider._model_name = "openai/gpt-4o"

        provider.shutdown()
        assert provider._client is None
        assert provider._model_name is None
