"""Tests for the OpenRouter provider."""

import json
from unittest.mock import MagicMock, patch

import pytest

from types import SimpleNamespace

from ..converters import (
    deserialize_history,
    get_original_tool_name,
    map_finish_reason,
    message_from_openai,
    message_to_openai,
    read_chunk_error,
    response_from_openai,
    sanitize_tool_name,
    serialize_history,
    tool_schema_to_openai,
)
from ..env import (
    DEFAULT_APP_CATEGORIES,
    DEFAULT_BASE_URL,
    HEADER_APP_CATEGORIES,
    HEADER_APP_TITLE,
    HEADER_HTTP_REFERER,
    resolve_api_key,
    resolve_app_categories,
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
from ..provider import _extract_generation_id
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


@pytest.fixture(autouse=True)
def _default_context_knob(monkeypatch):
    """Provide a manual ``context_length`` fallback so ``connect()``'s
    tier-1 auto-detect resolves (via the knob) in tests that don't seed a
    catalog — connect now fails fast without a resolvable window.  Tests
    asserting catalog/detect behavior seed ``_catalog_cache`` (which wins,
    detect-primary); the env-resolver tests override this via
    ``patch.dict``.
    """
    monkeypatch.setenv("JAATO_OPENROUTER_CONTEXT_LENGTH", "200000")


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

    def test_resolve_context_length_unset_is_none(self):
        # No hardcoded default — unset env resolves to None; the provider
        # auto-detects the per-model window from the catalog instead.
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_context_length() is None

    def test_resolve_context_length_from_env(self):
        with patch.dict("os.environ", {"JAATO_OPENROUTER_CONTEXT_LENGTH": "131072"}):
            assert resolve_context_length() == 131072

    def test_resolve_context_length_invalid(self):
        with patch.dict(
            "os.environ", {"JAATO_OPENROUTER_CONTEXT_LENGTH": "not-a-number"}
        ):
            assert resolve_context_length() is None

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

    def test_resolve_app_categories_default(self):
        with patch.dict("os.environ", {}, clear=True):
            # Default puts jaato into the cli-agent category — the
            # natural fit per https://openrouter.ai/docs/app-attribution.
            assert resolve_app_categories() == list(DEFAULT_APP_CATEGORIES)
            assert "cli-agent" in resolve_app_categories()

    def test_resolve_app_categories_from_env_single(self):
        with patch.dict(
            "os.environ", {"JAATO_OPENROUTER_APP_CATEGORIES": "writing-assistant"}
        ):
            assert resolve_app_categories() == ["writing-assistant"]

    def test_resolve_app_categories_from_env_csv(self):
        with patch.dict(
            "os.environ",
            {"JAATO_OPENROUTER_APP_CATEGORIES": "cli-agent, programming-app"},
        ):
            # Whitespace around entries is stripped.
            assert resolve_app_categories() == ["cli-agent", "programming-app"]

    def test_resolve_app_categories_env_empty_string_means_no_categories(self):
        with patch.dict(
            "os.environ", {"JAATO_OPENROUTER_APP_CATEGORIES": ""}
        ):
            # An empty env var is an explicit "no categories" opt-out,
            # distinct from "unset → defaults" semantics.
            assert resolve_app_categories() == []

    def test_attribution_header_names(self):
        # Verify we use the OpenRouter-canonical header names.
        assert HEADER_HTTP_REFERER == "HTTP-Referer"
        assert HEADER_APP_TITLE == "X-OpenRouter-Title"
        assert HEADER_APP_CATEGORIES == "X-OpenRouter-Categories"


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


class TestStrictToolSchemaConversion:
    """Strict tool-use mode knob (server 0.6.118+).

    See ``provider.py`` docstring on ``self._strict_tools`` for the
    rule context.  Setting ``api_params.strict_tools: true`` in the
    profile threads ``strict=True`` into the converter, which emits
    ``"strict": True`` as a sibling of ``parameters`` inside the
    function definition.  OpenRouter forwards this to supported
    upstreams for grammar-constrained tool-arg sampling.
    """

    def _basic_schema(self):
        from jaato_sdk.plugins.model_provider.types import ToolSchema
        return ToolSchema(
            name="emit_status",
            description="Emit a status report.",
            parameters={
                "type": "object",
                "properties": {"version": {"type": "string", "const": "1.0"}},
                "required": ["version"],
                "additionalProperties": False,
            },
        )

    def test_strict_false_default_omits_flag(self):
        """Default (strict=False) preserves the legacy advisory shape —
        no ``"strict"`` field in the function dict."""
        result = tool_schema_to_openai(self._basic_schema())
        assert "strict" not in result["function"]

    def test_strict_true_emits_flag(self):
        """``strict=True`` emits ``"strict": True`` as a sibling of
        ``parameters`` inside the function dict."""
        result = tool_schema_to_openai(self._basic_schema(), strict=True)
        assert result["function"]["strict"] is True
        # The flag is a sibling of parameters, not inside it.
        assert "strict" not in result["function"]["parameters"]

    def test_strict_does_not_disable_parameter_sanitization(self):
        """The existing ``const`` → ``enum`` rewrite still happens even
        when strict is on — strict-mode upstreams accept both, but the
        sanitization is benign and removing it would be a separate
        concern."""
        result = tool_schema_to_openai(self._basic_schema(), strict=True)
        version_param = result["function"]["parameters"]["properties"]["version"]
        # const got rewritten to enum (existing sanitizer behaviour preserved).
        assert version_param.get("enum") == ["1.0"]
        assert "const" not in version_param
        # Type is preserved.
        assert version_param["type"] == "string"

    def test_strict_list_form_threads_flag_to_every_tool(self):
        """``tool_schemas_to_openai(..., strict=True)`` propagates the
        flag to every converted function in the output list."""
        from shared.plugins.model_provider.openrouter.converters import (
            tool_schemas_to_openai,
        )
        a = self._basic_schema()
        from jaato_sdk.plugins.model_provider.types import ToolSchema
        b = ToolSchema(
            name="another_tool", description="...", parameters={"type": "object"},
        )
        result = tool_schemas_to_openai([a, b], strict=True)
        assert result is not None
        assert all(t["function"]["strict"] is True for t in result)

    def test_strict_list_form_default_omits_flag(self):
        """When ``strict`` is not passed (or False), no function carries
        the flag — preserves backward-compatible default."""
        from shared.plugins.model_provider.openrouter.converters import (
            tool_schemas_to_openai,
        )
        a = self._basic_schema()
        result = tool_schemas_to_openai([a])
        assert result is not None
        assert all("strict" not in t["function"] for t in result)


class TestMessageConversion:
    """Tests for Message <-> OpenAI format conversion."""

    def test_user_message(self):
        msg = Message.from_text(Role.USER, "Hello")
        result, = message_to_openai(msg)   # single-element list
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_user_message_with_image_marshals_image_url_block(self):
        # OpenRouter declares vision via the catalog, but the wire converter
        # only emitted text — the image was silently dropped and the model
        # confabulated.  An inline_data part must become an OpenAI image_url
        # content block (base64 data URL).
        import base64
        png = b"\x89PNG\r\n\x1a\nFAKEPNG"
        msg = Message(role=Role.USER, parts=[
            Part(text="What is in this image?"),
            Part(inline_data={"mime_type": "image/png", "data": png}),
        ])
        result, = message_to_openai(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        img = [b for b in result["content"] if b["type"] == "image_url"]
        txt = [b for b in result["content"] if b["type"] == "text"]
        assert txt[0]["text"] == "What is in this image?"
        assert len(img) == 1
        assert img[0]["image_url"]["url"] == (
            "data:image/png;base64," + base64.b64encode(png).decode("utf-8")
        )

    def test_text_only_user_message_stays_a_string(self):
        # Regression: no images -> plain-string content, unchanged wire shape.
        msg = Message(role=Role.USER, parts=[Part(text="just text")])
        result, = message_to_openai(msg)
        assert result["content"] == "just text"
        assert isinstance(result["content"], str)

    def test_tool_result_image_routed_to_followup_user_message(self):
        # OpenAI/OpenRouter tool messages can't carry images, so a tool result
        # with an image attachment (readFile on a PNG) surfaces the image as a
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
        assert "x.png" in out[0]["content"]
        assert out[-1]["role"] == "user"
        imgs = [b for b in out[-1]["content"] if b["type"] == "image_url"]
        assert len(imgs) == 1
        assert imgs[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_tool_result_without_image_emits_no_followup(self):
        tr = ToolResult(call_id="c1", name="grep", result={"matches": 3})
        msg = Message(role=Role.TOOL, parts=[Part(function_response=tr)])
        out = message_to_openai(msg)
        assert len(out) == 1 and out[0]["role"] == "tool"

    def test_assistant_message_text(self):
        msg = Message(role=Role.MODEL, parts=[Part(text="Hi there")])
        result, = message_to_openai(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Hi there"

    def test_assistant_message_with_tool_calls(self):
        fc = FunctionCall(id="call_1", name="read_file", args={"path": "/tmp"})
        msg = Message(role=Role.MODEL, parts=[Part(function_call=fc)])
        result, = message_to_openai(msg)

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
        result, = message_to_openai(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert json.loads(result["content"]) == {"x": 1}

    def test_parallel_tool_results_all_reach_wire(self):
        """A TOOL message with N parallel function_response parts must
        produce N wire ``role:"tool"`` messages — one per call_id — NOT just
        the first (the parallel-tool-result truncation bug, 2026-06-12
        build_descriptor: 7 parallel call_service results, only #1 reached
        the model)."""
        trs = [
            ToolResult(call_id=f"call_{i}", name="call_service",
                       result={"docs": [{"a": f"artifact-{i}"}]})
            for i in range(7)
        ]
        msg = Message(role=Role.TOOL,
                      parts=[Part(function_response=tr) for tr in trs])
        result = message_to_openai(msg)
        assert len(result) == 7, "all 7 parallel results must reach the wire"
        assert [m["tool_call_id"] for m in result] == [f"call_{i}" for i in range(7)]
        assert all(m["role"] == "tool" for m in result)
        assert [json.loads(m["content"])["docs"][0]["a"] for m in result] == \
            [f"artifact-{i}" for i in range(7)]


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

    def test_error(self):
        # OpenRouter mid-stream error per
        # https://openrouter.ai/docs/api/reference/streaming
        # ("Errors After Tokens Have Been Sent") — finish_reason="error"
        # accompanies the unified error chunk and must map to the
        # framework's ERROR outcome.
        assert map_finish_reason("error") == FinishReason.ERROR


class TestReadChunkError:
    """Tests for the OpenRouter mid-stream error extractor.

    The streaming spec puts the error payload at the top level of the
    chunk alongside ``choices``.  The OpenAI SDK doesn't model the
    field, so it lands in ``model_extra`` on real responses; tests
    typically attach it as a plain attribute or dict on a namespace.
    """

    def test_none_chunk(self):
        assert read_chunk_error(None) is None

    def test_chunk_without_error(self):
        chunk = SimpleNamespace(choices=[])
        assert read_chunk_error(chunk) is None

    def test_chunk_error_as_dict(self):
        chunk = SimpleNamespace(
            error={"code": "server_error", "message": "Provider disconnected"},
            choices=[],
        )
        result = read_chunk_error(chunk)
        assert result == {"code": "server_error", "message": "Provider disconnected"}

    def test_chunk_error_as_object(self):
        err_obj = SimpleNamespace(code="rate_limit", message="too many")
        chunk = SimpleNamespace(error=err_obj, choices=[])
        result = read_chunk_error(chunk)
        assert result == {"code": "rate_limit", "message": "too many"}

    def test_chunk_error_in_model_extra(self):
        chunk = SimpleNamespace(
            model_extra={"error": {"code": "bad_gateway", "message": "502"}},
            choices=[],
        )
        assert read_chunk_error(chunk) == {"code": "bad_gateway", "message": "502"}

    def test_magicmock_chunk_without_explicit_error(self):
        # MagicMock auto-generates child mocks for every attribute access,
        # so a naive ``getattr(chunk, "error", None)`` would always look
        # truthy.  read_chunk_error must NOT be fooled into reporting an
        # error when the test didn't explicitly set one.
        chunk = MagicMock()
        # Don't set ``error`` — MagicMock will auto-vivify it.
        assert read_chunk_error(chunk) is None


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
    def test_initialize_custom_context_length_stashed_as_knob(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"context_length": 200000},
        ))
        # The manual value is stashed as the fallback knob; the actual
        # window is resolved at connect() (catalog auto-detect PRIMARY).
        assert provider._context_length_knob == 200000

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


class TestAppAttributionIdentity:
    """Which APPLICATION the attribution headers name.

    Before ``shared/app_identity.py`` every product built on the SDK sent
    ``X-OpenRouter-Title: jaato``, so an integrator could not see their own
    app on the OpenRouter dashboard.  These pin the four-tier precedence:
    profile knob > ``JAATO_OPENROUTER_*`` env > the framework-stamped
    application identity > jaato itself.
    """

    def _headers(self, extra=None, env=None):
        """Initialize a provider and return the headers it would send."""
        captured = {}

        def fake_client_class(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        base_env = {"JAATO_OPENROUTER_API_KEY": "sk-or-test"}
        base_env.update(env or {})
        with patch.dict("os.environ", base_env, clear=True), patch(
            "shared.plugins.model_provider.openrouter.provider."
            "get_openai_client_class",
            return_value=fake_client_class,
        ):
            OpenRouterProvider().initialize(
                ProviderConfig(extra=dict(extra or {})),
            )
        return captured.get("default_headers") or {}

    def test_unconfigured_still_reports_as_the_framework(self):
        headers = self._headers()
        assert headers.get(HEADER_APP_TITLE) == "jaato"
        assert "jaato" in headers.get(HEADER_HTTP_REFERER, "").lower()

    def test_stamped_identity_names_the_application(self):
        headers = self._headers(
            extra={"app_identity": {
                "name": "Acme Copilot",
                "url": "https://acme.example",
                "powered_by": True,
            }},
        )
        assert headers.get(HEADER_APP_TITLE) == "Acme Copilot (powered by jaato)"
        assert headers.get(HEADER_HTTP_REFERER) == "https://acme.example"

    def test_app_can_opt_out_of_the_powered_by_suffix(self):
        headers = self._headers(
            extra={"app_identity": {"name": "Acme", "powered_by": False}},
        )
        assert headers.get(HEADER_APP_TITLE) == "Acme"

    def test_app_env_names_the_application(self):
        headers = self._headers(env={"JAATO_APP_NAME": "EnvApp"})
        assert headers.get(HEADER_APP_TITLE) == "EnvApp (powered by jaato)"

    def test_openrouter_env_outranks_the_identity(self):
        headers = self._headers(
            extra={"app_identity": {"name": "Acme"}},
            env={"JAATO_OPENROUTER_APP_TITLE": "Narrower"},
        )
        assert headers.get(HEADER_APP_TITLE) == "Narrower"

    def test_profile_knob_outranks_everything(self):
        headers = self._headers(
            extra={
                "app_identity": {"name": "Acme"},
                "app_title": "FromProfile",
                "http_referer": "https://profile.example",
            },
            env={"JAATO_OPENROUTER_APP_TITLE": "FromEnv"},
        )
        assert headers.get(HEADER_APP_TITLE) == "FromProfile"
        assert headers.get(HEADER_HTTP_REFERER) == "https://profile.example"

    def test_empty_openrouter_env_still_suppresses_the_header(self):
        # Long-standing opt-out: an explicitly empty env var means "send no
        # header", and must not fall through to the identity tier.
        headers = self._headers(
            extra={"app_identity": {"name": "Acme"}},
            env={"JAATO_OPENROUTER_APP_TITLE": ""},
        )
        assert HEADER_APP_TITLE not in headers

    def test_header_injection_via_the_app_name_is_neutralised(self):
        headers = self._headers(
            extra={"app_identity": {"name": "Acme\r\nX-Evil: 1"}},
        )
        title = headers.get(HEADER_APP_TITLE, "")
        assert "\r" not in title and "\n" not in title

    # -- categories: the third attribution value ------------------------

    def test_unconfigured_still_claims_the_framework_category(self):
        assert self._headers().get(HEADER_APP_CATEGORIES) == "cli-agent"

    def test_a_named_app_does_not_inherit_the_frameworks_category(self):
        # Filing a Slack bot under "cli-agent" mis-files it; no header is
        # the honest answer until the app declares its own.
        headers = self._headers(extra={"app_identity": {"name": "Acme Bot"}})
        assert HEADER_APP_CATEGORIES not in headers

    def test_a_named_app_sends_the_categories_it_declared(self):
        headers = self._headers(
            extra={"app_identity": {
                "name": "Acme Bot", "categories": ["chat-bot", "productivity"],
            }},
        )
        assert headers.get(HEADER_APP_CATEGORIES) == "chat-bot,productivity"

    def test_app_categories_env_names_the_categories(self):
        headers = self._headers(
            env={"JAATO_APP_NAME": "Acme Bot",
                 "JAATO_APP_CATEGORIES": "chat-bot"},
        )
        assert headers.get(HEADER_APP_CATEGORIES) == "chat-bot"

    def test_openrouter_categories_env_outranks_the_identity(self):
        headers = self._headers(
            extra={"app_identity": {"name": "Acme", "categories": ["chat-bot"]}},
            env={"JAATO_OPENROUTER_APP_CATEGORIES": "writing-assistant"},
        )
        assert headers.get(HEADER_APP_CATEGORIES) == "writing-assistant"

    def test_profile_knob_outranks_the_identity_categories(self):
        headers = self._headers(
            extra={
                "app_identity": {"name": "Acme", "categories": ["chat-bot"]},
                "app_categories": ["productivity"],
            },
        )
        assert headers.get(HEADER_APP_CATEGORIES) == "productivity"

    def test_a_category_outside_openrouters_taxonomy_is_dropped_not_fatal(self):
        # JAATO_APP_CATEGORIES is provider-agnostic, so it may legitimately
        # carry a slug some other directory uses.  Losing the listing is the
        # right cost; killing every session in the deployment is not.
        headers = self._headers(
            extra={"app_identity": {
                "name": "Acme", "categories": ["Customer Support", "chat-bot"],
            }},
        )
        assert headers.get(HEADER_APP_CATEGORIES) == "chat-bot"

    def test_the_profile_knob_still_fails_loud_on_a_bad_slug(self):
        # Authored, reviewed, OpenRouter-specific config: a typo there is
        # worth an exception.
        with pytest.raises(ValueError, match="lowercase"):
            self._headers(extra={"app_categories": ["Customer Support"]})


class TestExtraHeaders:
    """Tests for the ``extra_headers`` profile knob.

    Primary use case is OpenRouter's provider-specific beta header
    passthrough (e.g. ``x-anthropic-beta: interleaved-thinking-2025-05-14``).
    See https://openrouter.ai/docs/features/provider-routing
    "Provider-Specific Headers".
    """

    def _capture_client_kwargs(self):
        captured = {}

        def fake_client_class(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        return captured, fake_client_class

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_extra_headers_merged_into_default_headers(self, mock_client_class):
        captured, fake = self._capture_client_kwargs()
        mock_client_class.return_value = fake

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={
                "extra_headers": {
                    "x-anthropic-beta": (
                        "fine-grained-tool-streaming-2025-05-14,"
                        "interleaved-thinking-2025-05-14"
                    ),
                },
            },
        ))

        headers = captured.get("default_headers") or {}
        assert headers.get("x-anthropic-beta") == (
            "fine-grained-tool-streaming-2025-05-14,"
            "interleaved-thinking-2025-05-14"
        )
        assert provider._extra_headers == {
            "x-anthropic-beta": (
                "fine-grained-tool-streaming-2025-05-14,"
                "interleaved-thinking-2025-05-14"
            ),
        }

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_extra_headers_can_override_attribution(self, mock_client_class):
        # Profile values must win on key collisions so a profile can
        # set a different app title without touching framework defaults.
        captured, fake = self._capture_client_kwargs()
        mock_client_class.return_value = fake

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={
                "http_referer": "https://framework.example",
                "extra_headers": {HEADER_HTTP_REFERER: "https://profile.example"},
            },
        ))
        headers = captured.get("default_headers") or {}
        assert headers.get(HEADER_HTTP_REFERER) == "https://profile.example"

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_extra_headers_absent_means_no_extras(self, mock_client_class):
        captured, fake = self._capture_client_kwargs()
        mock_client_class.return_value = fake
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        # Default headers should only contain framework attribution (or
        # nothing at all if env vars aren't set).
        headers = captured.get("default_headers") or {}
        assert "x-anthropic-beta" not in headers
        assert provider._extra_headers == {}

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_extra_headers_rejects_non_dict(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(TypeError, match="extra_headers"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"extra_headers": "x-anthropic-beta: foo"},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_extra_headers_rejects_non_string_values(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(TypeError, match="str→str"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"extra_headers": {"x-foo": 123}},
            ))


class TestAppCategories:
    """Tests for the ``X-OpenRouter-Categories`` attribution header.

    See https://openrouter.ai/docs/app-attribution.  Jaato defaults to
    the ``cli-agent`` category; a profile can override or opt out.
    """

    def _capture_client_kwargs(self):
        captured = {}

        def fake_client_class(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        return captured, fake_client_class

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_default_sends_cli_agent_category(self, mock_client_class):
        captured, fake = self._capture_client_kwargs()
        mock_client_class.return_value = fake

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))

        headers = captured.get("default_headers") or {}
        assert headers.get(HEADER_APP_CATEGORIES) == "cli-agent"
        assert provider._app_categories == ["cli-agent"]

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    @patch.dict(
        "os.environ",
        {"JAATO_OPENROUTER_APP_CATEGORIES": "cli-agent,programming-app"},
        clear=True,
    )
    def test_env_var_overrides_default(self, mock_client_class):
        captured, fake = self._capture_client_kwargs()
        mock_client_class.return_value = fake

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        headers = captured.get("default_headers") or {}
        assert headers.get(HEADER_APP_CATEGORIES) == "cli-agent,programming-app"

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_profile_overrides_default(self, mock_client_class):
        captured, fake = self._capture_client_kwargs()
        mock_client_class.return_value = fake

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"app_categories": ["personal-agent", "writing-assistant"]},
        ))
        headers = captured.get("default_headers") or {}
        assert (
            headers.get(HEADER_APP_CATEGORIES)
            == "personal-agent,writing-assistant"
        )

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    @patch.dict("os.environ", {}, clear=True)
    def test_empty_list_opts_out_of_categories(self, mock_client_class):
        # An explicit empty list opts out of the header entirely —
        # distinct from "no profile setting → defaults".
        captured, fake = self._capture_client_kwargs()
        mock_client_class.return_value = fake

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"app_categories": []},
        ))
        headers = captured.get("default_headers") or {}
        assert HEADER_APP_CATEGORIES not in headers
        assert provider._app_categories == []

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_rejects_non_list(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(TypeError, match="app_categories"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"app_categories": "cli-agent"},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_rejects_uppercase_entry(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="lowercase"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"app_categories": ["CLI-AGENT"]},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_rejects_too_long_entry(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="30 characters"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"app_categories": ["a" * 31]},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_rejects_too_many_entries(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="at most 5"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"app_categories": ["a", "b", "c", "d", "e", "f"]},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_rejects_leading_hyphen(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="lowercase"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"app_categories": ["-bad"]},
            ))


class TestModelsFallback:
    """Tests for the ``api_params.models`` cross-model fallback list.

    OpenRouter walks each candidate in order when the primary ``model``
    fails (outage / context-limit / safety).  Required to take
    advantage of ``routing.sort = {by: ..., partition: "none"}``.
    """

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_models_stored_and_forwarded_via_extra_body(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={
                "api_params": {
                    "models": [
                        "anthropic/claude-sonnet-4.5",
                        "openai/gpt-5-mini",
                        "google/gemini-3-flash-preview",
                    ],
                },
            },
        ))
        assert provider._models_fallback == [
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-5-mini",
            "google/gemini-3-flash-preview",
        ]
        body = provider._build_extra_body()
        assert body.get("models") == [
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-5-mini",
            "google/gemini-3-flash-preview",
        ]

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_models_default_empty_omits_field(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        assert provider._models_fallback == []
        body = provider._build_extra_body()
        # The field must NOT be present when unset — sending an empty
        # list would tell OpenRouter "no fallbacks allowed" rather than
        # "use defaults".
        assert "models" not in body

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_models_extra_body_uses_fresh_list_per_call(self, mock_client_class):
        # Guard against accidental in-flight mutation: each call to
        # _build_extra_body() must return a list that's independent
        # from the provider's stored fallback list.
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"models": ["a", "b"]}},
        ))
        body = provider._build_extra_body()
        body["models"].append("c")
        assert provider._models_fallback == ["a", "b"]
        body2 = provider._build_extra_body()
        assert body2["models"] == ["a", "b"]

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_models_rejects_non_list(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(TypeError, match="models"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"api_params": {"models": "openai/gpt-5-mini"}},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_models_rejects_non_string_entries(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(TypeError, match="non-empty"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"api_params": {"models": ["openai/gpt-5-mini", ""]}},
            ))


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
        # Streaming returns an iterable of chunks.  One terminal chunk is
        # the minimum a real stream sends: a stream that ends without a
        # finish reason is now an error in its own right (#687), and this
        # test is about what was passed IN.
        terminal = MagicMock()
        terminal.choices = [MagicMock()]
        terminal.choices[0].delta.content = None
        terminal.choices[0].delta.tool_calls = None
        terminal.choices[0].delta.reasoning = None
        terminal.choices[0].finish_reason = "stop"
        terminal.usage = None
        terminal.error = None
        terminal.model_extra = {}
        fake_client.chat.completions.create.return_value = iter([terminal])
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


class TestServiceTierKnob:
    """Tests for ``api_params.service_tier`` (OpenAI-style processing tier).

    ``"flex"`` buys discounted best-effort processing, ``"priority"``
    low-latency processing; OpenRouter forwards the field to
    tier-supporting upstreams.  Default is unset — the field must not
    appear on the wire unless the profile opts in.
    """

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_unset_by_default_and_absent_from_wire(self, mock_client_class):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = create_mock_response(
            text="ok", finish_reason="stop"
        )
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        assert provider._service_tier is None

        provider.connect("openai/gpt-4o", skip_model_test=True)
        provider.complete([Message.from_text(Role.USER, "hi")])
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert "service_tier" not in call_kwargs

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_flex_forwarded_on_the_wire(self, mock_client_class):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = create_mock_response(
            text="ok", finish_reason="stop"
        )
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"service_tier": "flex"}},
        ))
        assert provider._service_tier == "flex"

        provider.connect("openai/gpt-4o", skip_model_test=True)
        provider.complete([Message.from_text(Role.USER, "hi")])
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["service_tier"] == "flex"

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_value_is_normalised_to_lowercase(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"service_tier": "Priority"}},
        ))
        assert provider._service_tier == "priority"

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_invalid_tier_raises(self, mock_client_class):
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="service_tier.*flex.*priority"):
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"api_params": {"service_tier": "turbo"}},
            ))

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_legacy_flat_key_still_read(self, mock_client_class):
        # The _knob fallback accepts the legacy flat position with a
        # deprecation warning, same as the other api_params knobs.
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"service_tier": "flex"},
        ))
        assert provider._service_tier == "flex"


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
        assert provider._context_length_knob == 32768
        legacy_warnings = [r for r in caplog.records if "legacy" in r.getMessage().lower()]
        assert legacy_warnings == [], (
            f"new shape should not emit deprecation warnings, got: "
            f"{[r.getMessage() for r in legacy_warnings]}"
        )

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_api_params_max_tokens_propagates_to_chat_completions(
        self, mock_client_class,
    ):
        """``api_params.max_tokens`` must reach the wire as a top-level
        ``max_tokens`` field on ``chat.completions.create``.

        OpenRouter does a pre-flight credit-vs-max-tokens check before
        running the request and rejects with 402 when the requested
        ``max_tokens`` exceeds what the account's balance can afford.
        Without this knob wired, low-balance accounts hit the 402 even
        on smoke-test workloads that would only emit a few dozen tokens
        of actual output.  See the smoke harness PR thread (2026-06-06).
        """
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = create_mock_response(
            text="ok", finish_reason="stop"
        )
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"max_tokens": 256}},
        ))
        provider.connect("anthropic/claude-3.5-sonnet", skip_model_test=True)
        assert provider._max_tokens == 256
        provider.complete([Message.from_text(Role.USER, "hi")])
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 256

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_api_params_max_tokens_omitted_when_unset(self, mock_client_class):
        """When ``api_params.max_tokens`` is not set, the request body must
        not carry a ``max_tokens`` field — letting OpenRouter forward the
        upstream's own default (e.g. the model's catalog max-output).

        Matches the temperature / top_p contract: omitted when unset so
        the upstream picks its own default rather than the framework
        smuggling a value in.
        """
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = create_mock_response(
            text="ok", finish_reason="stop"
        )
        mock_client_class.return_value = lambda **kw: fake_client

        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        provider.connect("openai/gpt-4o", skip_model_test=True)
        assert provider._max_tokens is None
        provider.complete([Message.from_text(Role.USER, "hi")])
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs

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
        assert provider._context_length_knob == 32768
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
    def test_strict_tools_knob_default_false(self, mock_client_class):
        """Without ``api_params.strict_tools``, the provider stays in
        advisory mode (legacy default, no behavior change)."""
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"temperature": 0.5}},
        ))
        assert provider._strict_tools is False

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_strict_tools_knob_true_propagates(self, mock_client_class):
        """``api_params.strict_tools: true`` flips the provider into
        strict mode."""
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"strict_tools": True}},
        ))
        assert provider._strict_tools is True

    @patch("shared.plugins.model_provider.openrouter.provider.get_openai_client_class")
    def test_strict_tools_knob_false_explicit(self, mock_client_class):
        """Setting ``strict_tools: false`` explicitly leaves the
        provider in advisory mode (no surprise from explicit declaration)."""
        mock_client_class.return_value = MagicMock()
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"strict_tools": False}},
        ))
        assert provider._strict_tools is False

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
        # Seed the catalog so tier-1 auto-detect resolves the window
        # (connect now fails fast without a resolvable context).
        provider._catalog_cache = [
            {"id": "anthropic/claude-3.5-sonnet", "context_length": 200000},
        ]
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

    def test_catalog_detect_wins_over_manual_knob(self):
        # Auto-detect PRIMARY (2026-06-10): the catalog-reported window
        # wins over an explicit context_length knob.
        provider = OpenRouterProvider()
        provider._client = MagicMock()
        provider._context_length_knob = 50000   # explicit fallback
        provider._catalog_cache = [
            {"id": "x/y", "context_length": 200000},  # server truth
        ]
        provider.connect("x/y")
        assert provider.get_context_limit() == 200000

    def test_connect_falls_back_to_knob_when_model_absent_from_catalog(self):
        # Model not in the catalog → no detect → manual knob fallback.
        provider = OpenRouterProvider()
        provider._client = MagicMock()
        provider._context_length_knob = 50000
        provider._catalog_cache = [{"id": "other/model", "context_length": 200000}]
        provider.connect("x/y", skip_model_test=True)
        assert provider.get_context_limit() == 50000

    def test_connect_raises_when_nothing_resolves(self):
        # No catalog entry, no knob → fail-fast (no hardcoded default).
        provider = OpenRouterProvider()
        provider._client = MagicMock()
        provider._catalog_cache = [{"id": "other/model", "context_length": 200000}]
        with pytest.raises(ValueError, match="context_length could not be resolved"):
            provider.connect("x/y", skip_model_test=True)


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
        provider._last_generation_id = "gen-abc123"

        provider.shutdown()
        assert provider._client is None
        assert provider._model_name is None
        assert provider._last_generation_id is None


# ==================== Streaming Spec Compliance ====================


def _make_chunk(
    *,
    content=None,
    finish_reason=None,
    tool_calls=None,
    reasoning=None,
    error=None,
    usage=None,
):
    """Build a streaming chunk that looks like the OpenAI SDK's
    ``ChatCompletionChunk`` for the bits ``_stream_response`` reads.

    Errors and choice deltas are kept faithful to OpenRouter's
    streaming spec:
    https://openrouter.ai/docs/api/reference/streaming
    """
    chunk = MagicMock()
    if error is not None:
        chunk.error = error
    else:
        # Don't let MagicMock auto-vivify ``error`` into a child mock —
        # read_chunk_error's MagicMock-resistance test depends on the
        # absence being detectable, but production code path tolerates
        # either shape.  Setting to None is unambiguous.
        chunk.error = None
    chunk.model_extra = None

    if content is None and finish_reason is None and tool_calls is None and reasoning is None:
        chunk.choices = []
    else:
        choice = MagicMock()
        choice.finish_reason = finish_reason
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = tool_calls
        delta.reasoning = reasoning
        delta.reasoning_content = None
        choice.delta = delta
        chunk.choices = [choice]

    if usage is not None:
        chunk.usage = usage
    else:
        chunk.usage = None
    return chunk


def _make_stream(chunks, *, headers=None):
    """Build a fake OpenAI ``Stream`` that yields ``chunks`` and tracks
    ``close()`` calls.  Optional ``headers`` populate ``.response.headers``
    so ``_extract_generation_id`` has something to read.
    """
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    if headers is not None:
        stream.response = SimpleNamespace(headers=headers)
    else:
        stream.response = None
    return stream


def _build_provider_for_streaming():
    """Construct an initialized provider with a fake OpenAI client."""
    provider = OpenRouterProvider()
    provider._client = MagicMock()
    provider._model_name = "openai/gpt-4o"
    provider._enable_thinking = False
    return provider


class TestStreamCancellationClosesConnection:
    """When the cancel token fires mid-stream, the SDK ``Stream`` must
    be ``.close()``d so the underlying HTTP connection is aborted.

    Per https://openrouter.ai/docs/api/reference/streaming
    ("Stream Cancellation"): aborting the connection is what stops
    upstream model processing and billing on supported providers.
    """

    def test_cancel_closes_stream(self):
        from jaato_sdk.plugins.model_provider.types import CancelToken

        provider = _build_provider_for_streaming()

        # Stream of three text chunks; cancel fires before chunk 2.
        cancel = CancelToken()
        chunks = [
            _make_chunk(content="hello "),
            _make_chunk(content="world"),
            _make_chunk(content="!"),
        ]

        def fake_create(**kwargs):
            return _make_stream(chunks)

        provider._client.chat.completions.create = fake_create
        stream_ref = {}

        # Wrap fake_create to capture the stream so we can assert close().
        def capture_create(**kwargs):
            s = _make_stream(chunks)
            stream_ref["stream"] = s
            return s

        provider._client.chat.completions.create = capture_create

        collected = []

        def on_chunk(text: str) -> None:
            collected.append(text)
            # Trip cancellation after the first chunk.
            if len(collected) == 1:
                cancel.cancel()

        result = provider._stream_response(
            messages=[],
            kwargs={},
            on_chunk=on_chunk,
            cancel_token=cancel,
        )

        assert result.finish_reason == FinishReason.CANCELLED
        # Either we broke after the first chunk or the second
        # arrived too — at least one chunk must have been delivered.
        assert collected == ["hello "] or collected == ["hello ", "world"]
        stream_ref["stream"].close.assert_called_once()

    def test_close_called_on_normal_completion(self):
        provider = _build_provider_for_streaming()
        chunks = [
            _make_chunk(content="hi"),
            _make_chunk(finish_reason="stop"),
        ]
        captured = {}

        def capture_create(**kwargs):
            s = _make_stream(chunks)
            captured["stream"] = s
            return s

        provider._client.chat.completions.create = capture_create

        provider._stream_response(
            messages=[],
            kwargs={},
            on_chunk=lambda _t: None,
        )

        captured["stream"].close.assert_called_once()


class TestStreamMidStreamError:
    """OpenRouter's mid-stream error spec: a chunk arrives with a
    top-level ``error`` field after some content has been streamed.

    The provider must raise an exception so retry / error-handling
    layers see the failure instead of treating it as a clean truncation.
    """

    def test_mid_stream_error_chunk_raises_infrastructure_error(self):
        provider = _build_provider_for_streaming()
        chunks = [
            _make_chunk(content="partial "),
            _make_chunk(
                error={"code": "server_error", "message": "Provider disconnected"},
                content="",
                finish_reason="error",
            ),
        ]
        captured = {}

        def capture_create(**kwargs):
            s = _make_stream(chunks)
            captured["stream"] = s
            return s

        provider._client.chat.completions.create = capture_create

        with pytest.raises(InfrastructureError) as exc_info:
            provider._stream_response(
                messages=[],
                kwargs={},
                on_chunk=lambda _t: None,
            )

        # The error message and code should both survive into the
        # raised exception so operators can see what OpenRouter said.
        assert "Provider disconnected" in str(exc_info.value)
        assert "server_error" in str(exc_info.value)
        # And the connection still gets closed.
        captured["stream"].close.assert_called_once()


class TestExtractGenerationId:
    """OpenRouter returns ``X-Generation-Id`` on every chat / completions
    response (header is set on both batch and streaming wrappers).
    """

    def test_extracts_from_headers_get(self):
        target = SimpleNamespace(
            headers=SimpleNamespace(get=lambda k: "gen-xyz" if k.lower() == "x-generation-id" else None)
        )
        # When the SDK wraps with a ``.response`` attribute (streams):
        stream = SimpleNamespace(response=target)
        assert _extract_generation_id(stream) == "gen-xyz"

    def test_extracts_from_dict_headers(self):
        batch = SimpleNamespace(headers={"x-generation-id": "gen-batch-1"})
        assert _extract_generation_id(batch) == "gen-batch-1"

    def test_returns_none_when_absent(self):
        target = SimpleNamespace(headers=SimpleNamespace(get=lambda k: None))
        stream = SimpleNamespace(response=target)
        assert _extract_generation_id(stream) is None

    def test_never_raises(self):
        # Defensive: even on a wildly wrong shape, never raise.
        assert _extract_generation_id(None) is None
        assert _extract_generation_id(SimpleNamespace()) is None
        assert _extract_generation_id(SimpleNamespace(response=SimpleNamespace())) is None

    def test_stream_path_records_generation_id_on_provider(self):
        provider = _build_provider_for_streaming()
        chunks = [_make_chunk(content="hi"), _make_chunk(finish_reason="stop")]
        headers = SimpleNamespace(get=lambda k: "gen-stream-42" if k.lower() == "x-generation-id" else None)

        def capture_create(**kwargs):
            return _make_stream(chunks, headers=headers)

        provider._client.chat.completions.create = capture_create

        provider._stream_response(
            messages=[],
            kwargs={},
            on_chunk=lambda _t: None,
        )

        assert provider.get_last_generation_id() == "gen-stream-42"


class TestParallelToolCallsKnob:
    """``api_params.parallel_tool_calls`` profile knob — when set,
    propagates as a kwarg to ``chat.completions.create``.  OpenRouter
    forwards verbatim to upstream OpenAI / Anthropic / vLLM hosts.
    """

    def _make_provider_with_api_params(self, api_params: dict):
        """Construct an initialized provider with the given api_params
        block under config.extra."""
        from shared.plugins.model_provider.base import ProviderConfig
        provider = OpenRouterProvider()
        provider._verify_connectivity = lambda *a, **k: None  # type: ignore[assignment]
        provider._trace = lambda _msg: None  # type: ignore[assignment]
        provider._list_models_for_catalog = lambda: []  # type: ignore[assignment]
        cfg = ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": api_params, "context_length": 32768},
        )
        provider.initialize(cfg)
        return provider

    def test_default_is_none(self):
        provider = self._make_provider_with_api_params({})
        assert provider._parallel_tool_calls is None

    def test_knob_false_stored(self):
        provider = self._make_provider_with_api_params(
            {"parallel_tool_calls": False}
        )
        assert provider._parallel_tool_calls is False

    def test_knob_true_stored(self):
        provider = self._make_provider_with_api_params(
            {"parallel_tool_calls": True}
        )
        assert provider._parallel_tool_calls is True


# ==================== Modality Detection Tests ====================


class TestModalityDetection:
    """OpenRouter resolves a model's INPUT modalities from the catalog's
    ``architecture.input_modalities`` (detect-PRIMARY, self-updating),
    falling back to the manual ``modalities`` knob, then the text-only
    floor.  Foundation for vision via the model-tier roles
    (``docs/design/multimodal-model-support.md``).
    """

    def _provider_with_catalog(self, catalog):
        provider = OpenRouterProvider()
        provider._catalog_cache = catalog  # seed cache; detect wins
        return provider

    def test_text_floor_when_not_connected(self):
        provider = OpenRouterProvider()
        provider._model_name = None
        assert provider.modalities() == {"text"}

    def test_detect_vision_model_from_catalog(self):
        provider = self._provider_with_catalog([
            {"id": "google/gemini-3-pro",
             "architecture": {"input_modalities": ["text", "image"]}},
        ])
        provider._model_name = "google/gemini-3-pro"
        assert provider.modalities() == {"text", "image"}
        assert provider.supports_modality("image") is True

    def test_detect_text_only_model_from_catalog(self):
        provider = self._provider_with_catalog([
            {"id": "openai/gpt-5-mini",
             "architecture": {"input_modalities": ["text"]}},
        ])
        provider._model_name = "openai/gpt-5-mini"
        assert provider.modalities() == {"text"}
        assert provider.supports_modality("image") is False

    def test_knob_used_when_model_absent_from_catalog(self):
        # Self-hosted gateway / catalog gap: detect returns None, the
        # manual assertion takes over.
        provider = self._provider_with_catalog([])  # empty catalog
        provider._model_name = "some/uncatalogued-vision-model"
        provider._modalities_knob = ["text", "image"]
        assert provider.modalities() == {"text", "image"}

    def test_text_floor_when_model_absent_and_no_knob(self):
        provider = self._provider_with_catalog([])
        provider._model_name = "some/uncatalogued-model"
        provider._modalities_knob = None
        # Unknown image-support degrades to the safe text floor (never a
        # false image claim).
        assert provider.modalities() == {"text"}

    def test_detect_wins_over_knob(self):
        provider = self._provider_with_catalog([
            {"id": "openai/gpt-5-mini",
             "architecture": {"input_modalities": ["text"]}},
        ])
        provider._model_name = "openai/gpt-5-mini"
        provider._modalities_knob = ["text", "image"]  # stale assertion
        # Live catalog beats the manual knob.
        assert provider.modalities() == {"text"}

    def test_missing_architecture_field_falls_through(self):
        provider = self._provider_with_catalog([
            {"id": "weird/model"},  # no architecture key
        ])
        provider._model_name = "weird/model"
        provider._modalities_knob = ["text", "image"]
        assert provider.modalities() == {"text", "image"}

    def test_modalities_knob_parsed_from_config(self):
        with patch(
            "shared.plugins.model_provider.openrouter.provider."
            "get_openai_client_class"
        ) as mock_client_class:
            mock_client_class.return_value = MagicMock()
            provider = OpenRouterProvider()
            provider.initialize(ProviderConfig(
                api_key="sk-or-test",
                extra={"framework_overrides": {"modalities": ["text", "image"]}},
            ))
        assert provider._modalities_knob == ["text", "image"]

    def test_modalities_knob_rejects_non_list(self):
        with patch(
            "shared.plugins.model_provider.openrouter.provider."
            "get_openai_client_class"
        ) as mock_client_class:
            mock_client_class.return_value = MagicMock()
            provider = OpenRouterProvider()
            with pytest.raises(TypeError):
                provider.initialize(ProviderConfig(
                    api_key="sk-or-test",
                    extra={"framework_overrides": {"modalities": "image"}},
                ))
