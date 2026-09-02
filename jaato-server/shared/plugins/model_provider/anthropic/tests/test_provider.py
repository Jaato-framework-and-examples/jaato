"""Tests for AnthropicProvider."""

import pytest
from unittest.mock import MagicMock, patch

from ..provider import AnthropicProvider
from shared.plugins.model_provider.base import ProviderConfig
from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    ToolSchema,
    TurnResult,
    Message,
    Part,
    Role,
)
from ..errors import (
    APIKeyNotFoundError,
    APIKeyInvalidError,
    RateLimitError,
    ContextLimitError,
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
    """Create a mock Anthropic response.

    Uses ``spec=[]`` for the usage object to prevent MagicMock from
    auto-creating attributes like ``cache_creation_input_tokens`` which
    would be MagicMock instances instead of None (breaking ``> 0`` checks
    in ``extract_usage_from_response``).
    """
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

    # Create usage with explicit attributes (spec=[] prevents auto-creation
    # of cache/thinking attributes that would be MagicMock instead of None)
    mock_response.usage = MagicMock(spec=[])
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

    def test_verify_auth_reads_config_param_before_initialize(self):
        """verify_auth must read the ``config`` PARAMETER, not
        ``self._config``, so it works BEFORE initialize().

        Regression for the in-process runtime path
        (``JaatoRuntime.verify_auth``) which creates a FRESH provider with
        ``config=None`` and does NOT call ``initialize()`` — so
        ``self._config`` is unset.  The provider previously read
        ``getattr(self._config, 'extra', ...)`` at verify time and raised
        ``AttributeError: 'AnthropicProvider' object has no attribute
        '_config'``.  After the fix it reaches the normal credential check
        and raises ``APIKeyNotFoundError`` (no creds), NOT AttributeError.
        """
        provider = AnthropicProvider()  # fresh — no initialize()
        cfg = ProviderConfig(extra={"workspace_path": "/ws", "config_root": "/cr"})
        with patch.dict('os.environ', {}, clear=True), \
                patch('shared.plugins.model_provider.anthropic.provider.'
                      'try_load_tokens_with_reason', return_value=(None, None)), \
                patch('shared.plugins.model_provider.anthropic.provider.'
                      'get_valid_access_token', return_value=None):
            with pytest.raises(APIKeyNotFoundError):
                provider.verify_auth(allow_interactive=False, config=cfg)

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


class TestConfigNamespacing:
    """Tests for the four-layer config namespacing introduced in 0.6.24.

    Layers (under ``plugin_configs.anthropic``):
      - Top-level: api_key / oauth_token (auth identity)
      - api_params: temperature / top_p / top_k / max_tokens /
                    enable_thinking / thinking_budget
      - framework_overrides: rare escape hatches (none defined today)

    Backward compatibility: same keys are also read from the legacy
    flat position with a deprecation warning per key.
    """

    @patch('anthropic.Anthropic')
    def test_new_shape_no_deprecation_warning(self, mock_client_class, caplog):
        mock_client_class.return_value = MagicMock()
        provider = AnthropicProvider()
        with caplog.at_level("WARNING"):
            provider.initialize(ProviderConfig(
                api_key="sk-ant-test",
                extra={
                    "api_params": {
                        "temperature": 0.0,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_tokens": 4096,
                        "enable_thinking": False,
                        "thinking_budget": 5000,
                    },
                },
            ))
        assert provider._temperature == 0.0
        assert provider._top_p == 0.95
        assert provider._top_k == 40
        assert provider._max_tokens_override == 4096
        assert provider._enable_thinking is False
        assert provider._thinking_budget == 5000
        legacy_warnings = [r for r in caplog.records if "legacy" in r.getMessage().lower()]
        assert legacy_warnings == [], (
            f"new shape should not emit deprecation warnings, got: "
            f"{[r.getMessage() for r in legacy_warnings]}"
        )

    @patch('anthropic.Anthropic')
    def test_legacy_flat_shape_still_works_with_warnings(
        self, mock_client_class, caplog,
    ):
        mock_client_class.return_value = MagicMock()
        provider = AnthropicProvider()
        with caplog.at_level("WARNING"):
            provider.initialize(ProviderConfig(
                api_key="sk-ant-test",
                extra={
                    "temperature": 0.5,
                    "top_p": 1.0,
                    "enable_thinking": True,
                    "thinking_budget": 8000,
                },
            ))
        assert provider._temperature == 0.5
        assert provider._top_p == 1.0
        assert provider._enable_thinking is True
        assert provider._thinking_budget == 8000
        legacy_warnings = [r for r in caplog.records if "legacy" in r.getMessage().lower()]
        assert len(legacy_warnings) >= 4, (
            f"expected ≥4 deprecation warnings (temperature/top_p/enable_thinking/"
            f"thinking_budget), got {len(legacy_warnings)}"
        )

    @patch('anthropic.Anthropic')
    def test_new_layer_wins_over_legacy_flat_key(self, mock_client_class):
        # When both the nested layer key and the legacy flat key are
        # present, the nested form wins (legacy is the deprecation
        # fallback only).
        mock_client_class.return_value = MagicMock()
        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={
                "api_params": {"temperature": 0.7},
                "temperature": 0.3,
                "enable_thinking": False,        # legacy
            },
        ))
        assert provider._temperature == 0.7
        # enable_thinking still picked up via legacy fallback when
        # not in api_params at all
        assert provider._enable_thinking is False

    @patch('anthropic.Anthropic')
    def test_no_sampling_config_means_none(self, mock_client_class):
        # When the profile sets nothing, instance vars stay at their
        # __init__ defaults (None) so complete() omits them from
        # messages.create() and the API server-side default applies.
        mock_client_class.return_value = MagicMock()
        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        assert provider._temperature is None
        assert provider._top_p is None
        assert provider._top_k is None
        assert provider._max_tokens_override is None


class TestComplete_SamplingParams:
    """Verify sampling params reach messages.create() when set."""

    @patch('anthropic.Anthropic')
    def test_temperature_passed_to_messages_create(self, mock_client_class):
        from jaato_sdk.plugins.model_provider.types import Message, Role
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # mock the response to look like an Anthropic Message
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5,
                                          cache_creation_input_tokens=0,
                                          cache_read_input_tokens=0)
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={"api_params": {"temperature": 0.0, "top_p": 0.95, "top_k": 40,
                                    "max_tokens": 2048}},
        ))
        provider.connect("claude-sonnet-4-20250514")
        provider.complete(
            messages=[Message.from_text(Role.USER, "hi")],
            system_instruction="be brief",
        )
        # Verify the sampling params reached the SDK
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["max_tokens"] == 2048

    @patch('anthropic.Anthropic')
    def test_omit_sampling_when_unset(self, mock_client_class):
        from jaato_sdk.plugins.model_provider.types import Message, Role
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5,
                                          cache_creation_input_tokens=0,
                                          cache_read_input_tokens=0)
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect("claude-sonnet-4-20250514")
        provider.complete(
            messages=[Message.from_text(Role.USER, "hi")],
            system_instruction="be brief",
        )
        call_kwargs = mock_client.messages.create.call_args.kwargs
        # Sampling params absent → API server-side default applies
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs
        assert "top_k" not in call_kwargs
        # max_tokens still set (framework default)
        assert "max_tokens" in call_kwargs


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


class TestErrorHandling:
    """Tests for error handling via complete().

    connect() calls _verify_model_responds() which hits the API, so the mock
    must return a valid response for connect() before being switched to an
    error for the complete() call.
    """

    @patch('anthropic.Anthropic')
    def test_handles_401_unauthorized(self, mock_client_class):
        """Should raise APIKeyInvalidError on 401."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-bad"))
        provider.connect('claude-sonnet-4-20250514')

        mock_client.messages.create.side_effect = Exception("401 Unauthorized")
        messages = [Message.from_text(Role.USER, "Test")]
        with pytest.raises(APIKeyInvalidError):
            provider.complete(messages)

    @patch('anthropic.Anthropic')
    def test_handles_429_rate_limit(self, mock_client_class):
        """Should raise RateLimitError on 429."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        mock_client.messages.create.side_effect = Exception("429 Rate limit exceeded")
        messages = [Message.from_text(Role.USER, "Test")]
        with pytest.raises(RateLimitError):
            provider.complete(messages)

    @patch('anthropic.Anthropic')
    def test_handles_529_overloaded(self, mock_client_class):
        """Should raise OverloadedError on 529."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        mock_client.messages.create.side_effect = Exception("529 Overloaded")
        messages = [Message.from_text(Role.USER, "Test")]
        with pytest.raises(OverloadedError):
            provider.complete(messages)

    @patch('anthropic.Anthropic')
    def test_handles_context_length_error(self, mock_client_class):
        """Should raise ContextLimitError on context overflow."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        mock_client.messages.create.side_effect = Exception("context_length_exceeded")
        messages = [Message.from_text(Role.USER, "Test")]
        with pytest.raises(ContextLimitError):
            provider.complete(messages)

    @patch('anthropic.Anthropic')
    def test_api_connection_error_propagates_to_caller(self, mock_client_class):
        """PR #177 (2026-05-21): anthropic SDK network-layer errors
        (APIConnectionError, APITimeoutError) must escape
        ``provider.complete()`` so ``with_retry`` sees them and
        exponential-backoffs.  Pre-fix, the SDK error was swallowed
        into ``TurnResult.from_exception()``; ``with_retry`` saw
        fn() return normally and never retried.  Surfaced by peer's
        kb-orchestrator v152-retry-11 (Finding C)."""
        import anthropic
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        # Real anthropic.APIConnectionError (the actual SDK class, not
        # a jaato-typed wrapper).  Constructed with a mock request per
        # the SDK's expected signature; fallback to bare-string ctor
        # if SDK shape varies.
        try:
            request = MagicMock()
            conn_err = anthropic.APIConnectionError(request=request)
        except TypeError:
            conn_err = anthropic.APIConnectionError("Connection error")
        mock_client.messages.create.side_effect = conn_err
        messages = [Message.from_text(Role.USER, "Test")]

        # MUST escape — NOT swallowed into TurnResult.
        with pytest.raises(anthropic.APIConnectionError):
            provider.complete(messages)

    @patch('anthropic.Anthropic')
    def test_api_timeout_error_propagates_to_caller(self, mock_client_class):
        """PR #177: anthropic.APITimeoutError follows the same path
        as APIConnectionError — both must escape provider.complete
        so with_retry can classify them transient."""
        import anthropic
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        try:
            request = MagicMock()
            timeout_err = anthropic.APITimeoutError(request=request)
        except TypeError:
            timeout_err = anthropic.APITimeoutError("Request timeout")
        mock_client.messages.create.side_effect = timeout_err
        messages = [Message.from_text(Role.USER, "Test")]

        with pytest.raises(anthropic.APITimeoutError):
            provider.complete(messages)


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

    def test_get_context_limit_unknown_model_raises(self):
        """No hardcoded fallback: an unknown model with no override raises
        rather than returning a guessed default."""
        provider = AnthropicProvider()
        provider._model_name = 'unknown-model'

        with pytest.raises(ValueError, match="no known context window"):
            provider.get_context_limit()

    def test_get_context_limit_override_knob_wins(self):
        """framework_overrides.context_length overrides the per-model table —
        the escape hatch for unlisted models / budget-down."""
        provider = AnthropicProvider()
        provider._model_name = 'claude-sonnet-4-20250514'  # would be 200_000
        provider._context_length_knob = 100_000

        assert provider.get_context_limit() == 100_000

    def test_get_context_limit_override_knob_covers_unknown_model(self):
        provider = AnthropicProvider()
        provider._model_name = 'some-future-model'
        provider._context_length_knob = 500_000

        assert provider.get_context_limit() == 500_000


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

        provider.shutdown()

        assert provider._client is None
        assert provider._model_name is None


class TestPromptCaching:
    """Tests for cache plugin delegation via complete().

    Cache annotation logic now lives in the ``cache_anthropic`` plugin.
    These tests verify the provider's delegation behavior -- the plugin's
    own test suite (``cache_anthropic/tests/test_plugin.py``) covers the
    annotation logic in detail.
    """

    @patch('anthropic.Anthropic')
    def test_no_cache_control_without_plugin(self, mock_client_class):
        """Without a cache plugin, no cache_control annotations should appear."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        messages = [Message.from_text(Role.USER, "Hi")]
        provider.complete(messages, system_instruction="You are helpful.")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system = call_kwargs['system']

        # Without plugin, no cache_control should be added
        assert isinstance(system, list)
        assert 'cache_control' not in system[0]

    @patch('anthropic.Anthropic')
    def test_cache_plugin_delegation(self, mock_client_class):
        """With a cache plugin attached, prepare_request should be called."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        tools = [
            ToolSchema(name='tool1', description='First tool', parameters={}),
            ToolSchema(name='tool2', description='Second tool', parameters={}),
        ]

        # Attach a mock cache plugin
        mock_plugin = MagicMock()
        mock_plugin.prepare_request.return_value = {
            "system": [{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "tool1"}, {"name": "tool2", "cache_control": {"type": "ephemeral"}}],
            "messages": [],
            "cache_breakpoint_index": -1,
        }
        provider.set_cache_plugin(mock_plugin)

        messages = [Message.from_text(Role.USER, "Hi")]
        provider.complete(
            messages,
            system_instruction="You are helpful.",
            tools=tools,
        )

        # Plugin's prepare_request should have been called
        mock_plugin.prepare_request.assert_called_once()

        # The annotated system/tools from the plugin should be in the API call
        call_kwargs = mock_client.messages.create.call_args.kwargs
        system = call_kwargs['system']
        assert system[0]['cache_control'] == {"type": "ephemeral"}


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
        event.message.usage = MagicMock(spec=[
            "input_tokens", "output_tokens",
        ])
        event.message.usage.input_tokens = 100
        event.message.usage.output_tokens = 10

        result = extract_message_start(event)
        assert result.prompt_tokens == 100
        assert result.output_tokens == 10
        assert result.total_tokens == 110


class TestComplete:
    """Tests for the stateless complete() method.

    complete() is the provider's sole API method. The caller passes the full
    message list (and optional system instruction, tools, cancel token) and
    receives a TurnResult. The provider holds no conversation history.
    """

    @patch('anthropic.Anthropic')
    def test_complete_returns_response(self, mock_client_class):
        """complete() should return a TurnResult with text and usage."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(text="Batch response")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        messages = [Message.from_text(Role.USER, "Hello")]

        result = provider.complete(messages)

        assert result.text == "Batch response"
        assert result.response.usage.prompt_tokens == 10
        assert result.response.usage.output_tokens == 20

    @patch('anthropic.Anthropic')
    def test_complete_with_system_instruction(self, mock_client_class):
        """complete() should pass system instruction to API."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        messages = [Message.from_text(Role.USER, "Hi")]
        provider.complete(
            messages,
            system_instruction="You are a helpful assistant.",
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert 'system' in call_kwargs
        # System blocks are wrapped in list format
        assert isinstance(call_kwargs['system'], list)
        assert "helpful assistant" in call_kwargs['system'][0]['text']

    @patch('anthropic.Anthropic')
    def test_complete_with_tools(self, mock_client_class):
        """complete() should pass tools to API."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        tools = [ToolSchema(
            name='get_weather',
            description='Get weather',
            parameters={"type": "object", "properties": {}, "required": []}
        )]
        messages = [Message.from_text(Role.USER, "Weather?")]

        provider.complete(messages, tools=tools)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert 'tools' in call_kwargs
        from shared.tool_id_map import name_to_id
        assert call_kwargs['tools'][0]['name'] == name_to_id('get_weather')

    @patch('anthropic.Anthropic')
    def test_complete_extracts_function_calls(self, mock_client_class):
        """complete() should extract function calls from response."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(
            text=None,
            tool_use=[{
                "id": "toolu_xyz",
                "name": "search",
                "input": {"query": "hello"}
            }],
            stop_reason="tool_use",
        )
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        messages = [Message.from_text(Role.USER, "Search for hello")]
        result = provider.complete(messages)

        fcs = result.response.get_function_calls()
        assert len(fcs) == 1
        assert fcs[0].name == "search"
        assert fcs[0].args == {"query": "hello"}
        assert result.finish_reason == FinishReason.TOOL_USE

    @patch('anthropic.Anthropic')
    def test_complete_multi_turn_conversation(self, mock_client_class):
        """complete() should handle multi-turn conversations."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(
            text="I remember your name"
        )
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        # Build a multi-turn conversation
        messages = [
            Message.from_text(Role.USER, "My name is Alice"),
            Message(role=Role.MODEL, parts=[Part(text="Nice to meet you, Alice!")]),
            Message.from_text(Role.USER, "What's my name?"),
        ]

        result = provider.complete(messages)
        assert result.text == "I remember your name"

        # Verify all messages were sent to the API
        call_args = mock_client.messages.create.call_args
        api_messages = call_args.kwargs['messages']
        assert len(api_messages) == 3  # user, assistant, user

    @patch('anthropic.Anthropic')
    def test_complete_updates_last_usage(self, mock_client_class):
        """complete() should update get_token_usage() for accounting."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(
            input_tokens=50, output_tokens=30,
        )
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        messages = [Message.from_text(Role.USER, "Hi")]
        provider.complete(messages)

        usage = provider.get_token_usage()
        assert usage.prompt_tokens == 50
        assert usage.output_tokens == 30

    @patch('anthropic.Anthropic')
    def test_complete_works_after_initialize_and_connect(self, mock_client_class):
        """complete() should work with just initialize() and connect()."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(text="OK")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        messages = [Message.from_text(Role.USER, "Hi")]
        result = provider.complete(messages)
        assert result.text == "OK"

    @patch('anthropic.Anthropic')
    def test_complete_validates_tool_use_pairing(self, mock_client_class):
        """complete() should validate/repair history with unpaired tool_use."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(text="Fixed")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        # Create history with an unpaired tool_use followed by a user message
        # validate_tool_use_pairing should strip the unpaired tool_use
        messages = [
            Message.from_text(Role.USER, "Do something"),
            Message(role=Role.MODEL, parts=[
                Part.from_function_call(FunctionCall(id="t1", name="tool", args={}))
            ]),
            # Missing TOOL message with tool_result — cancelled before results sent
            Message.from_text(Role.USER, "Actually, stop"),
        ]

        # Should not raise
        result = provider.complete(messages)
        assert result.text == "Fixed"

    @patch('anthropic.Anthropic')
    def test_complete_error_handling(self, mock_client_class):
        """complete() should route API errors through _handle_api_error."""
        mock_client = MagicMock()
        # Return valid response for connect()'s verify call
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        # Now set the error for the actual complete() call
        mock_client.messages.create.side_effect = Exception("429 Rate limit exceeded")

        messages = [Message.from_text(Role.USER, "Hi")]

        with pytest.raises(RateLimitError):
            provider.complete(messages)

    @patch('anthropic.Anthropic')
    def test_complete_with_thinking(self, mock_client_class):
        """complete() should include thinking in response."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(
            text="Answer", thinking="Let me think...",
        )
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-ant-test",
            extra={'enable_thinking': True},
        ))
        provider.connect('claude-sonnet-4-20250514')

        messages = [Message.from_text(Role.USER, "Complex question")]
        result = provider.complete(messages)

        assert result.text == "Answer"
        assert result.response.thinking == "Let me think..."

    @patch('anthropic.Anthropic')
    def test_complete_with_cache_plugin(self, mock_client_class):
        """complete() should delegate to cache plugin when attached."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response()
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        # Attach cache plugin
        mock_plugin = MagicMock()
        mock_plugin.prepare_request.return_value = {
            "system": [{"type": "text", "text": "Cached system"}],
            "tools": [],
            "messages": [],
        }
        mock_plugin._budget_bp3_message_id = None
        provider.set_cache_plugin(mock_plugin)

        messages = [Message.from_text(Role.USER, "Hi")]
        provider.complete(messages, system_instruction="System")

        # Plugin's prepare_request should have been called
        mock_plugin.prepare_request.assert_called_once()

    @patch('anthropic.Anthropic')
    def test_complete_does_not_mutate_input_messages(self, mock_client_class):
        """complete() must not mutate the caller's message list."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = create_mock_response(text="OK")
        mock_client_class.return_value = mock_client

        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(api_key="sk-ant-test"))
        provider.connect('claude-sonnet-4-20250514')

        messages = [Message.from_text(Role.USER, "Hello")]
        original_len = len(messages)

        provider.complete(messages)

        # The caller's list must not have been modified
        assert len(messages) == original_len
