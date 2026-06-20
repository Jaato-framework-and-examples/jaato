"""Tests for Nebius provider."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from ..provider import NebiusProvider, create_provider
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
        with patch.dict("os.environ", {"JAATO_NEBIUS_API_KEY": "nbk-test-test123"}):
            assert resolve_api_key() == "nbk-test-test123"

    def test_resolve_api_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_api_key() is None

    def test_resolve_base_url_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_base_url() == DEFAULT_BASE_URL

    def test_resolve_base_url_from_env(self):
        with patch.dict("os.environ", {"JAATO_NEBIUS_BASE_URL": "http://localhost:8000/v1"}):
            assert resolve_base_url() == "http://localhost:8000/v1"

    def test_resolve_context_length_unset_is_none(self):
        # No hardcoded fallback (no-fallback rule): unset env → None, so the
        # provider raises a "not configured" error rather than guessing a default.
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_context_length() is None

    def test_resolve_context_length_from_env(self):
        with patch.dict("os.environ", {"JAATO_NEBIUS_CONTEXT_LENGTH": "131072"}):
            assert resolve_context_length() == 131072

    def test_resolve_context_length_invalid_is_none(self):
        with patch.dict("os.environ", {"JAATO_NEBIUS_CONTEXT_LENGTH": "not-a-number"}):
            assert resolve_context_length() is None

    def test_is_self_hosted_localhost(self):
        assert is_self_hosted("http://localhost:8000/v1") is True
        assert is_self_hosted("http://127.0.0.1:8000/v1") is True

    def test_is_self_hosted_private_network(self):
        assert is_self_hosted("http://192.168.1.100:8000/v1") is True
        assert is_self_hosted("http://10.0.0.5:8000/v1") is True

    def test_is_self_hosted_public(self):
        assert is_self_hosted("https://api.tokenfactory.nebius.com/v1") is False
        assert is_self_hosted("https://custom-nim.example.com/v1") is False


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

    def test_reverse_mapping_unknown(self):
        assert get_original_tool_name("t_nonexist") == "t_nonexist"


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

    def test_schema_roundtrip(self):
        schema = ToolSchema(
            name="mcp.server.tool",
            description="A tool",
            parameters={},
        )
        result = tool_schema_to_openai(schema)

        tool_id = result["function"]["name"]
        assert tool_id.startswith("t_")
        assert get_original_tool_name(tool_id) == "mcp.server.tool"


class TestMessageConversion:
    """Tests for Message <-> OpenAI format conversion."""

    def test_user_message(self):
        msg = Message.from_text(Role.USER, "Hello")
        result, = message_to_openai(msg)   # single-element list

        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_assistant_message_text(self):
        msg = Message(role=Role.MODEL, parts=[Part(text="Hi there")])
        result, = message_to_openai(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Hi there"

    def test_user_message_with_image_marshals_image_url_block(self):
        # A vision-declared OpenAI-compat model must actually RECEIVE the image:
        # an inline_data part becomes an OpenAI image_url content block (base64
        # data URL).  Without this the image was silently dropped and only the
        # text reached the wire ("no image attached").
        import base64
        png = b"\x89PNG\r\n\x1a\nFAKEPNGBYTES"
        msg = Message(role=Role.USER, parts=[
            Part(text="What is in this image?"),
            Part(inline_data={"mime_type": "image/png", "data": png}),
        ])
        result, = message_to_openai(msg)

        assert result["role"] == "user"
        # content is now a LIST of blocks (text + image), not a bare string
        assert isinstance(result["content"], list)
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        img_blocks = [b for b in result["content"] if b["type"] == "image_url"]
        assert text_blocks[0]["text"] == "What is in this image?"
        assert len(img_blocks) == 1
        expected = "data:image/png;base64," + base64.b64encode(png).decode("utf-8")
        assert img_blocks[0]["image_url"]["url"] == expected

    def test_user_message_image_only_no_text(self):
        png = b"rawbytes"
        msg = Message(role=Role.USER, parts=[
            Part(inline_data={"mime_type": "image/jpeg", "data": png}),
        ])
        result, = message_to_openai(msg)
        assert isinstance(result["content"], list)
        assert all(b["type"] == "image_url" for b in result["content"])
        assert result["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_tool_result_image_routed_to_followup_user_message(self):
        # OpenAI-compat tool messages can't carry images, so a tool result with
        # an image attachment (readFile on a PNG) surfaces the image as a
        # follow-up user message; the tool message keeps the text result.
        from jaato_sdk.plugins.model_provider.types import Attachment
        tr = ToolResult(
            call_id="c1", name="readFile",
            result={"path": "x.png", "type": "image"},
            attachments=[Attachment(mime_type="image/png", data=b"PNGBYTES",
                                    display_name="x.png")],
        )
        msg = Message(role=Role.TOOL, parts=[Part(function_response=tr)])
        out = message_to_openai(msg)
        assert out[0]["role"] == "tool"
        assert "x.png" in out[0]["content"]      # text result, no raw bytes
        assert out[-1]["role"] == "user"
        imgs = [b for b in out[-1]["content"] if b["type"] == "image_url"]
        assert len(imgs) == 1
        assert imgs[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_tool_result_without_image_emits_no_followup(self):
        tr = ToolResult(call_id="c1", name="grep", result={"matches": 3})
        msg = Message(role=Role.TOOL, parts=[Part(function_response=tr)])
        out = message_to_openai(msg)
        assert len(out) == 1 and out[0]["role"] == "tool"

    def test_text_only_user_message_stays_a_string(self):
        # Regression: no images -> plain-string content, unchanged wire shape.
        msg = Message(role=Role.USER, parts=[Part(text="just text")])
        result, = message_to_openai(msg)
        assert result["content"] == "just text"
        assert isinstance(result["content"], str)

    def test_assistant_message_with_tool_calls(self):
        fc = FunctionCall(id="call_1", name="read_file", args={"path": "/tmp"})
        msg = Message(role=Role.MODEL, parts=[Part(function_call=fc)])
        result, = message_to_openai(msg)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_1"
        assert result["tool_calls"][0]["function"]["name"] == sanitize_tool_name("read_file")

    def test_tool_result_message(self):
        tr = ToolResult(call_id="call_1", name="read_file", result={"content": "file data"})
        msg = Message(role=Role.TOOL, parts=[Part(function_response=tr)])
        result, = message_to_openai(msg)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert json.loads(result["content"]) == {"content": "file data"}

    def test_parallel_tool_results_all_reach_wire(self):
        """N parallel function_response parts → N wire tool messages (not
        just the first).  Shared converter used by nim/vllm/lmstudio/
        tensorrt_llm — the parallel-tool-result truncation bug, 2026-06-12."""
        trs = [
            ToolResult(call_id=f"call_{i}", name="call_service",
                       result={"docs": [{"a": f"artifact-{i}"}]})
            for i in range(5)
        ]
        msg = Message(role=Role.TOOL,
                      parts=[Part(function_response=tr) for tr in trs])
        result = message_to_openai(msg)
        assert len(result) == 5
        assert [m["tool_call_id"] for m in result] == [f"call_{i}" for i in range(5)]
        assert [json.loads(m["content"])["docs"][0]["a"] for m in result] == \
            [f"artifact-{i}" for i in range(5)]

    def test_roundtrip_user_message(self):
        original = Message.from_text(Role.USER, "Hello world")
        openai_msg, = message_to_openai(original)
        restored = message_from_openai(openai_msg)

        assert restored.role == Role.USER
        assert restored.parts[0].text == "Hello world"

    def test_roundtrip_assistant_message(self):
        original = Message(role=Role.MODEL, parts=[Part(text="Response")])
        openai_msg, = message_to_openai(original)
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
        provider = NebiusProvider()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIKeyNotFoundError) as exc_info:
                provider.initialize(ProviderConfig())

        assert "JAATO_NEBIUS_API_KEY" in str(exc_info.value)

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    def test_initialize_with_api_key(self, mock_client_class):
        """Should initialize with key from config.api_key."""
        mock_client_class.return_value = MagicMock()

        provider = NebiusProvider()
        # context_length is now required (no hardcoded default); set it so this
        # auth-focused test reaches client creation.
        provider.initialize(ProviderConfig(
            api_key="nbk-test-test", extra={"context_length": 131072},
        ))

        assert provider._api_key == "nbk-test-test"
        assert provider._client is not None

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_NEBIUS_API_KEY": "nbk-test-env"}, clear=True)
    def test_initialize_from_env(self, mock_client_class):
        """Should auto-detect key from JAATO_NEBIUS_API_KEY env var."""
        mock_client_class.return_value = MagicMock()

        provider = NebiusProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 131072}))

        assert provider._api_key == "nbk-test-env"

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_NEBIUS_BASE_URL": "http://localhost:8000/v1"}, clear=True)
    def test_initialize_self_hosted_no_key(self, mock_client_class):
        """Should initialize without key for self-hosted endpoints."""
        mock_client_class.return_value = MagicMock()

        provider = NebiusProvider()
        provider.initialize(ProviderConfig(extra={"context_length": 131072}))

        assert provider._api_key is None
        assert provider._client is not None

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_connect_raises_when_context_unresolved(self, mock_client_class):
        """No hardcoded fallback: with the catalog empty (no per-model entry)
        AND no manual override, connect() raises rather than guessing a
        default context window.  init() stays cheap and does NOT raise (the
        window is bootstrapped at connect from the catalog)."""
        mock_client_class.return_value = MagicMock()
        provider = NebiusProvider()
        provider.initialize(ProviderConfig(api_key="nbk-test-test"))  # no raise
        provider._catalog_cache = []  # empty catalog -> detect returns None
        with pytest.raises(ValueError, match="context_length could not be resolved"):
            provider.connect("some/model")

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    @patch.dict("os.environ", {"JAATO_NEBIUS_CONTEXT_LENGTH": "200000"}, clear=True)
    def test_env_knob_stashed_at_init_resolves_at_connect(self, mock_client_class):
        """The env override is stashed at init and used at connect when the
        catalog has no entry for the model."""
        mock_client_class.return_value = MagicMock()
        provider = NebiusProvider()
        provider.initialize(ProviderConfig(api_key="nbk-test-test"))
        assert provider._context_length_knob == 200000
        assert provider._context_length == 0  # not resolved until connect
        provider._catalog_cache = []
        provider.connect("some/model")
        assert provider._context_length == 200000

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    def test_catalog_context_beats_knob_at_connect(self, mock_client_class):
        """Catalog auto-detect is PRIMARY: the per-model context_length from
        GET /v1/models wins over the manual knob."""
        mock_client_class.return_value = MagicMock()
        with patch.dict("os.environ", {"JAATO_NEBIUS_CONTEXT_LENGTH": "200000"}):
            provider = NebiusProvider()
            provider.initialize(ProviderConfig(api_key="nbk-test-test"))
            provider._catalog_cache = [
                {"id": "meta-llama/Llama-3.3-70B-Instruct",
                 "context_length": 131072,
                 "architecture": {"modality": "text->text"}},
            ]
            provider.connect("meta-llama/Llama-3.3-70B-Instruct")
            assert provider._context_length == 131072  # catalog wins over 200000

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    def test_initialize_stashes_profile_knob_over_env(self, mock_client_class):
        """Profile knob (config.extra.context_length) wins over the env tier
        when stashed at init."""
        mock_client_class.return_value = MagicMock()
        with patch.dict("os.environ", {"JAATO_NEBIUS_CONTEXT_LENGTH": "200000"}):
            provider = NebiusProvider()
            provider.initialize(ProviderConfig(
                api_key="nbk-test-test", extra={"context_length": 131072},
            ))
            assert provider._context_length_knob == 131072

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    def test_initialize_custom_base_url(self, mock_client_class):
        """Should use custom base_url from config.extra."""
        mock_client_class.return_value = MagicMock()

        provider = NebiusProvider()
        provider.initialize(ProviderConfig(
            api_key="nbk-test-test",
            extra={"base_url": "http://nim.internal:8080/v1", "context_length": 131072},
        ))

        assert provider._base_url == "http://nim.internal:8080/v1"

    @patch("shared.plugins.model_provider.nebius.provider.get_openai_client_class")
    def test_initialize_stashes_custom_context_length_knob(self, mock_client_class):
        """A custom context_length from config.extra is stashed as the knob
        (used at connect when the catalog lacks the model)."""
        mock_client_class.return_value = MagicMock()

        provider = NebiusProvider()
        provider.initialize(ProviderConfig(
            api_key="nbk-test-test",
            extra={"context_length": 131072},
        ))

        assert provider._context_length_knob == 131072


class TestVerifyAuth:
    """Tests for verify_auth (must work before initialize)."""

    def test_verify_auth_with_key(self):
        provider = NebiusProvider()
        with patch.dict("os.environ", {"JAATO_NEBIUS_API_KEY": "nbk-test-test"}):
            assert provider.verify_auth() is True

    def test_verify_auth_honors_profile_extra_api_key(self):
        """The pre-init gate must accept a profile-supplied key in
        config.extra['api_key'] — the shape the runtime builds for
        verify_auth (ProviderConfig(extra=plugin_configs[nebius])) when a
        profile sets plugin_configs.nebius.api_key: pass://... .  With no env
        var and no stored creds, this used to wrongly fail (parity bug vs
        openrouter/zhipuai)."""
        provider = NebiusProvider()
        cfg = ProviderConfig(extra={"api_key": "nbk-test-from-profile"})
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.nebius.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),  # no stored creds
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_honors_profile_top_level_api_key(self):
        """config.api_key (top-level) is also honored."""
        provider = NebiusProvider()
        cfg = ProviderConfig(api_key="nbk-test-top-level")
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.nebius.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                assert provider.verify_auth(config=cfg) is True

    def test_verify_auth_profile_key_beats_missing_env_and_creds(self):
        """End-to-end of the kb scenario: no env, no stored creds, key only in
        the profile config -> verify_auth returns True (does NOT raise)."""
        provider = NebiusProvider()
        cfg = ProviderConfig(extra={"api_key": "nbk-test-x", "workspace_path": "/tmp"})
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.nebius.auth"
                ".try_load_credentials_with_reason",
                return_value=(None, None),
            ):
                # allow_interactive=False is the bootstrap path; must not raise.
                assert provider.verify_auth(allow_interactive=False, config=cfg) is True

    def test_verify_auth_self_hosted(self):
        provider = NebiusProvider()
        with patch.dict("os.environ", {"JAATO_NEBIUS_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            assert provider.verify_auth() is True

    def test_verify_auth_no_key_raises(self):
        provider = NebiusProvider()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIKeyNotFoundError):
                provider.verify_auth(allow_interactive=False)

    def test_verify_auth_no_key_returns_false(self):
        provider = NebiusProvider()
        with patch.dict("os.environ", {}, clear=True):
            assert provider.verify_auth(allow_interactive=True) is False

    def test_verify_auth_surfaces_broken_credentials(self):
        """Broken credential file must surface the reason, not be
        swallowed into a generic "no key" error.

        Mirrors the Zhipu AI fix: a corrupt / unreadable credential
        file used to be indistinguishable from a missing one, which
        produced misleading "No API key found" messages while the real
        problem (corrupt JSON, missing api_key, permission error) stayed
        hidden.  The provider now emits the actual load reason.
        """
        provider = NebiusProvider()
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "shared.plugins.model_provider.nebius.auth.try_load_credentials_with_reason",
                return_value=(None, "invalid JSON at /tmp/nebius_auth.json: Expecting value"),
            ):
                messages = []
                # Not self-hosted, no key -> should raise, but first emit the reason
                with pytest.raises(APIKeyNotFoundError):
                    provider.verify_auth(on_message=messages.append)
                joined = "\n".join(messages)
                assert "could not be loaded" in joined
                assert "invalid JSON" in joined


class TestConnection:
    """Tests for connect and session management."""

    def test_connect_sets_model_and_bootstraps_context(self):
        provider = NebiusProvider()
        provider._client = MagicMock()
        provider._catalog_cache = [
            {"id": "meta-llama/Llama-3.1-70B-Instruct", "context_length": 131072,
             "architecture": {"modality": "text->text"}},
        ]
        provider.connect("meta-llama/Llama-3.1-70B-Instruct")
        assert provider.model_name == "meta-llama/Llama-3.1-70B-Instruct"
        assert provider.get_context_limit() == 131072

    def test_is_connected(self):
        provider = NebiusProvider()
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
        provider = NebiusProvider()
        assert not hasattr(provider, '_system_instruction')
        assert not hasattr(provider, '_tools')
        assert not hasattr(provider, '_history')


class TestCapabilities:
    """Tests for capability queries."""

    def test_supports_streaming(self):
        assert NebiusProvider().supports_streaming() is True

    def test_supports_structured_output(self):
        assert NebiusProvider().supports_structured_output() is True

    def test_supports_stop(self):
        assert NebiusProvider().supports_stop() is True

    def test_supports_thinking_default(self):
        assert NebiusProvider().supports_thinking() is False

    def test_supports_thinking_deepseek(self):
        provider = NebiusProvider()
        provider._model_name = "deepseek/deepseek-r1"
        assert provider.supports_thinking() is True

    def test_name(self):
        assert NebiusProvider().name == "nebius"


class TestTokenManagement:
    """Tests for token counting and context limits."""

    def test_count_tokens(self):
        provider = NebiusProvider()
        assert provider.count_tokens("abcd") == 1
        assert provider.count_tokens("a" * 100) == 25

    def test_get_context_limit(self):
        provider = NebiusProvider()
        provider._context_length = 131072
        assert provider.get_context_limit() == 131072

    def test_get_token_usage(self):
        provider = NebiusProvider()
        assert provider.get_token_usage().total_tokens == 0


class TestErrorClassification:
    """Tests for error classification and retry logic."""

    def test_classify_rate_limit(self):
        provider = NebiusProvider()
        exc = RateLimitError(original_error="429")
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": True, "infra": False}

    def test_classify_infrastructure(self):
        provider = NebiusProvider()
        exc = InfrastructureError(status_code=500)
        result = provider.classify_error(exc)
        assert result == {"transient": True, "rate_limit": False, "infra": True}

    def test_classify_unknown(self):
        provider = NebiusProvider()
        result = provider.classify_error(ValueError("unknown"))
        assert result is None

    def test_retry_after_rate_limit(self):
        provider = NebiusProvider()
        exc = RateLimitError(retry_after=30.0)
        assert provider.get_retry_after(exc) == 30.0

    def test_retry_after_other(self):
        provider = NebiusProvider()
        assert provider.get_retry_after(ValueError("x")) is None


class TestCreateProvider:
    """Tests for factory function."""

    def test_create_provider(self):
        provider = create_provider()
        assert isinstance(provider, NebiusProvider)
        assert provider.name == "nebius"


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_clears_state(self):
        provider = NebiusProvider()
        provider._client = MagicMock()
        provider._model_name = "test"

        provider.shutdown()

        assert provider._client is None
        assert provider._model_name is None


# ==================== Catalog + Modality Detection ====================

class TestCatalogAndModalities:
    """Nebius bootstraps context + INPUT modalities from the
    ``GET /v1/models`` catalog (RichModel.context_length /
    architecture.modality, an OpenRouter-style ``input->output`` string).
    """

    def _provider_with_catalog(self, catalog):
        provider = NebiusProvider()
        provider._catalog_cache = catalog  # seed cache; detect wins
        return provider

    def test_lookup_context_length(self):
        p = self._provider_with_catalog([
            {"id": "m1", "context_length": 131072},
        ])
        assert p._lookup_context_length("m1") == 131072
        assert p._lookup_context_length("absent") is None

    def test_modality_text_only(self):
        p = self._provider_with_catalog([
            {"id": "llama", "context_length": 131072,
             "architecture": {"modality": "text->text"}},
        ])
        p._model_name = "llama"
        assert p.modalities() == {"text"}
        assert p.supports_modality("image") is False

    def test_modality_vision_input(self):
        # OpenRouter-style multimodal input on the left of the arrow.
        p = self._provider_with_catalog([
            {"id": "qwen-vl", "context_length": 32768,
             "architecture": {"modality": "text+image->text"}},
        ])
        assert p.modalities(model="qwen-vl") == {"text", "image"}

    def test_modality_knob_when_model_absent(self):
        p = self._provider_with_catalog([])  # empty catalog -> detect None
        p._model_name = "uncatalogued"
        p._modalities_knob = ["text", "image"]
        assert p.modalities() == {"text", "image"}

    def test_modality_text_floor_when_unknown(self):
        p = self._provider_with_catalog([])
        p._model_name = "uncatalogued"
        assert p.modalities() == {"text"}  # never a false image claim

    def test_modality_missing_architecture_falls_through(self):
        p = self._provider_with_catalog([{"id": "m", "context_length": 8192}])
        p._model_name = "m"
        p._modalities_knob = ["text", "image"]
        assert p.modalities() == {"text", "image"}  # knob, since no arch field

    def test_list_models_from_catalog(self):
        p = self._provider_with_catalog([
            {"id": "b/model"}, {"id": "a/model"}, {"id": "a/other"},
        ])
        assert p.list_models() == ["a/model", "a/other", "b/model"]
        assert p.list_models(prefix="a/") == ["a/model", "a/other"]

    def test_fetch_catalog_parses_and_caches(self):
        p = NebiusProvider()
        p._base_url = "https://api.tokenfactory.nebius.com/v1"
        p._api_key = "nbk-test-key"
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"data": [{"id": "x", "context_length": 1}]}
        fake_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            first = p._fetch_catalog()
            second = p._fetch_catalog()  # cached: no second network call
        assert first == [{"id": "x", "context_length": 1}]
        assert second is first
        assert mock_get.call_count == 1
        # Bearer auth header carried on the catalog GET.
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer nbk-test-key"

    def test_fetch_catalog_network_failure_returns_empty_not_cached(self):
        p = NebiusProvider()
        p._base_url = "https://api.tokenfactory.nebius.com/v1"
        with patch("httpx.get", side_effect=Exception("boom")):
            assert p._fetch_catalog() == []
        assert p._catalog_cache is None  # not cached -> next call retries


# ==================== api_params Passthrough ====================

class TestApiParams:
    """plugin_configs.nebius.api_params forwarded onto chat.completions.create
    (the determinism + tool_choice knob; parity with openrouter).  Closes the
    third nebius parity gap after verify_auth (#302/#303).
    """

    def _provider(self, api_params):
        with patch(
            "shared.plugins.model_provider.nebius.provider.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = NebiusProvider()
            p.initialize(ProviderConfig(
                api_key="nbk-test", extra={"api_params": api_params},
            ))
        return p

    def test_parsed_and_filtered_at_init(self):
        p = self._provider({
            "temperature": 0.0, "tool_choice": "required", "max_tokens": 4096,
            "top_k": 40, "bogus": 1,  # unsupported -> dropped (warning)
        })
        assert p._api_params == {
            "temperature": 0.0, "tool_choice": "required", "max_tokens": 4096,
        }

    def test_temperature_zero_survives(self):
        # The determinism case: 0.0 must NOT be falsy-dropped.
        p = self._provider({"temperature": 0.0})
        kwargs = {}
        p._apply_api_params(kwargs, tool_choice=None)
        assert kwargs["temperature"] == 0.0

    def test_tool_choice_forwarded_with_tools(self):
        p = self._provider({"tool_choice": "required"})
        kwargs = {"tools": [{"type": "function"}]}
        p._apply_api_params(kwargs, tool_choice=None)
        assert kwargs["tool_choice"] == "required"

    def test_tool_choice_dropped_without_tools(self):
        # OpenAI rejects tool_choice without tools; profile-wide "required"
        # must not break a tool-less call (e.g. GC summarization).
        p = self._provider({"tool_choice": "required", "temperature": 0.0})
        kwargs = {}
        p._apply_api_params(kwargs, tool_choice=None)
        assert "tool_choice" not in kwargs
        assert kwargs["temperature"] == 0.0

    def test_per_call_tool_choice_overrides_profile(self):
        from shared.tool_id_map import name_to_id
        p = self._provider({"tool_choice": "required"})
        kwargs = {"tools": [1]}
        named = {"type": "function", "function": {"name": "x"}}
        p._apply_api_params(kwargs, tool_choice=named)
        # Per-call wins over the profile's "required"; the function name is
        # mapped to its wire id (name_to_id) like the tools array (#332).
        assert kwargs["tool_choice"] == {
            "type": "function", "function": {"name": name_to_id("x")}}

    def test_non_dict_api_params_raises(self):
        with patch(
            "shared.plugins.model_provider.nebius.provider.get_openai_client_class"
        ) as mc:
            mc.return_value = MagicMock()
            p = NebiusProvider()
            with pytest.raises(TypeError):
                p.initialize(ProviderConfig(
                    api_key="nbk-test", extra={"api_params": "required"},
                ))

    def test_complete_batch_forwards_api_params_to_create(self):
        # End-to-end: complete() must hand api_params to chat.completions.create.
        p = self._provider({"temperature": 0.0, "tool_choice": "required",
                            "max_tokens": 256})
        p._model_name = "deepseek-ai/DeepSeek-V3"
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return create_mock_response(text=None,
                                        tool_calls=[create_mock_tool_call()],
                                        finish_reason="tool_calls")

        p._client.chat.completions.create = fake_create
        schema = ToolSchema(name="discovery_result", description="d", parameters={
            "type": "object", "properties": {}})
        p.complete([Message.from_text(Role.USER, "go")], tools=[schema])
        assert captured["temperature"] == 0.0
        assert captured["tool_choice"] == "required"
        assert captured["max_tokens"] == 256
        assert "tools" in captured
