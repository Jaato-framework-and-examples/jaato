"""Tests for the OpenRouter prompt-caching wire format.

Covers three layers:

  1. Model detection / config resolution (``cache.py``).
  2. Request annotation — ``cache_control`` on system block and last
     tool dict (``converters.py``).
  3. Response parsing — ``prompt_tokens_details.cached_tokens``,
     ``cache_creation_input_tokens``, ``cost`` mapped onto
     :class:`TokenUsage`.
  4. End-to-end: a stubbed ``initialize() + connect() + complete()``
     flow against a mock OpenAI client verifying that the body the
     OpenAI SDK would PUT matches the OpenRouter spec.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from jaato_sdk.plugins.model_provider.types import (
    Message,
    Part,
    Role,
    ToolSchema,
)

from ..cache import (
    EXPLICIT_CACHE_MODEL_HINTS,
    make_cache_control,
    model_supports_explicit_cache,
    resolve_cache_active,
)
from ..converters import (
    apply_cache_usage,
    extract_usage,
    system_message_with_cache,
    tool_schemas_to_openai,
)
from shared.tool_id_map import name_to_id
from ..provider import OpenRouterProvider
from shared.plugins.model_provider.base import ProviderConfig


@pytest.fixture(autouse=True)
def _default_context_knob(monkeypatch):
    """Manual context_length fallback so connect()'s tier-1 auto-detect
    resolves (no catalog seeded in these cache tests)."""
    monkeypatch.setenv("JAATO_OPENROUTER_CONTEXT_LENGTH", "200000")


# ==================== cache.py ====================


class TestModelDetection:
    """Resolution of ``cache_prompt`` against model id."""

    @pytest.mark.parametrize("model", [
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-opus-4-20250514",
        "google/gemini-1.5-pro-latest",
        "google/gemini-2.5-flash",
        "google/gemini-3-pro-preview",
    ])
    def test_explicit_cache_models(self, model):
        assert model_supports_explicit_cache(model) is True

    @pytest.mark.parametrize("model", [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "deepseek/deepseek-chat",
        "moonshotai/kimi-k2",
        "x-ai/grok-2",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-pro",       # bare gemini-pro is the legacy 1.0 model
        None,
        "",
    ])
    def test_automatic_or_no_cache_models(self, model):
        assert model_supports_explicit_cache(model) is False

    def test_case_insensitive(self):
        assert model_supports_explicit_cache("Anthropic/Claude-3.5-Sonnet") is True

    def test_hints_constant_documents_match_set(self):
        assert "anthropic/" in EXPLICIT_CACHE_MODEL_HINTS

    @pytest.mark.parametrize("setting,model,expected", [
        ("auto", "anthropic/claude-3.5-sonnet", True),
        ("auto", "openai/gpt-4o", False),
        (None, "anthropic/claude-3.5-sonnet", True),
        (None, "openai/gpt-4o", False),
        (True, "openai/gpt-4o", True),
        ("on", "openai/gpt-4o", True),
        ("true", "openai/gpt-4o", True),
        ("1", "openai/gpt-4o", True),
        (False, "anthropic/claude-3.5-sonnet", False),
        ("off", "anthropic/claude-3.5-sonnet", False),
        ("false", "anthropic/claude-3.5-sonnet", False),
    ])
    def test_resolve_cache_active(self, setting, model, expected):
        assert resolve_cache_active(setting, model) is expected


class TestCacheControl:
    def test_default_ttl(self):
        assert make_cache_control() == {"type": "ephemeral"}

    def test_explicit_5m_ttl(self):
        assert make_cache_control("5m") == {"type": "ephemeral"}

    def test_extended_1h_ttl(self):
        assert make_cache_control("1h") == {"type": "ephemeral", "ttl": "1h"}


# ==================== Request annotation ====================


class TestSystemMessageWithCache:
    def test_no_cache_returns_flat_string_content(self):
        msg = system_message_with_cache("you are helpful", None)
        assert msg == {"role": "system", "content": "you are helpful"}

    def test_with_cache_returns_content_part_list(self):
        cc = {"type": "ephemeral"}
        msg = system_message_with_cache("long stable prefix", cc)
        assert msg["role"] == "system"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "long stable prefix"
        assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_is_copied_not_shared(self):
        cc = {"type": "ephemeral"}
        msg = system_message_with_cache("hi", cc)
        msg["content"][0]["cache_control"]["mutated"] = True
        assert "mutated" not in cc, "cache_control input must not be aliased"


class TestToolSchemaCaching:
    @pytest.fixture
    def tools(self):
        return [
            ToolSchema(name="zebra", description="z", parameters={"type": "object"}),
            ToolSchema(name="alpha", description="a", parameters={"type": "object"}),
            ToolSchema(name="middle", description="m", parameters={"type": "object"}),
        ]

    def test_no_cache_preserves_order(self, tools):
        out = tool_schemas_to_openai(tools)
        # Tool names round-trip through the deterministic hash, so the
        # assertion uses ``name_to_id`` rather than the human-readable name.
        names = [t["function"]["name"] for t in out]
        assert names == [name_to_id("zebra"), name_to_id("alpha"), name_to_id("middle")]
        assert all("cache_control" not in t for t in out)

    def test_cache_control_sorts_and_stamps_last(self, tools):
        out = tool_schemas_to_openai(tools, cache_control={"type": "ephemeral"})
        names = [t["function"]["name"] for t in out]
        # Order must be deterministic across turns so the cache prefix
        # doesn't invalidate.  Sort is by hashed name (the wire form) —
        # still stable, just not human-alphabetic.
        assert names == sorted(names), "tools must be sorted for cache stability"
        assert "cache_control" not in out[0]
        assert "cache_control" not in out[1]
        assert out[-1]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_empty_schemas_list(self):
        assert tool_schemas_to_openai([], cache_control={"type": "ephemeral"}) is None
        assert tool_schemas_to_openai(None, cache_control={"type": "ephemeral"}) is None


# ==================== Response parsing ====================


def _usage_obj(**fields):
    """Build a usage-like object that mimics the OpenAI SDK's Pydantic model
    just enough for the cache-extraction helpers — getattr returns the
    value when present and ``None`` otherwise (the SDK shape).
    """
    return SimpleNamespace(**fields)


class TestApplyCacheUsage:
    def test_cached_tokens_from_prompt_tokens_details(self):
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            prompt_tokens_details=_usage_obj(cached_tokens=900),
        )
        apply_cache_usage(raw, usage)
        assert usage.cache_read_tokens == 900

    def test_cache_creation_input_tokens(self):
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            cache_creation_input_tokens=500,
        )
        apply_cache_usage(raw, usage)
        assert usage.cache_creation_tokens == 500

    def test_cache_write_tokens_from_prompt_tokens_details(self):
        """The shape OpenRouter actually sends (issue #699).

        Captured from the wire on 2026-08-29, anthropic/claude-sonnet-4.6
        with a cache_control breakpoint, two identical calls 4s apart:

            cold: {"cached_tokens": 0,    "cache_write_tokens": 4403}
            warm: {"cached_tokens": 4403, "cache_write_tokens": 0}

        The write count is NESTED beside the read count and spelled
        differently from Anthropic's native field.  Reading only the
        top-level name left `cache_creation_tokens` permanently None
        while the write was billed.
        """
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(
            prompt_tokens=4412,
            completion_tokens=4,
            total_tokens=4416,
            prompt_tokens_details=_usage_obj(
                cached_tokens=0, cache_write_tokens=4403),
        )
        apply_cache_usage(raw, usage)
        assert usage.cache_creation_tokens == 4403
        assert usage.cache_read_tokens is None   # zero is not a hit

    def test_the_warm_half_of_the_same_exchange(self):
        """The second call: the read lands, the write goes quiet."""
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(
            prompt_tokens=4412,
            prompt_tokens_details=_usage_obj(
                cached_tokens=4403, cache_write_tokens=0),
        )
        apply_cache_usage(raw, usage)
        assert usage.cache_read_tokens == 4403
        assert usage.cache_creation_tokens is None

    def test_cache_write_tokens_as_a_plain_dict(self):
        """``prompt_tokens_details`` arrives as a dict on some paths."""
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(
            prompt_tokens=4412,
            prompt_tokens_details={"cached_tokens": 900,
                                   "cache_write_tokens": 4403},
        )
        apply_cache_usage(raw, usage)
        assert usage.cache_read_tokens == 900
        assert usage.cache_creation_tokens == 4403

    def test_the_anthropic_native_name_still_works_as_a_fallback(self):
        """Kept deliberately: this helper serves several upstream shapes,
        and an upstream that passes Anthropic's field through must not
        regress because OpenRouter's spelling was added."""
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(
            prompt_tokens=1000,
            cache_creation_input_tokens=500,
            prompt_tokens_details=_usage_obj(cached_tokens=0),
        )
        apply_cache_usage(raw, usage)
        assert usage.cache_creation_tokens == 500

    def test_the_nested_name_wins_over_the_top_level_one(self):
        """If both arrive, the one OpenRouter actually sends is the
        authority — the fallback exists for shapes that lack it."""
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(
            prompt_tokens=1000,
            cache_creation_input_tokens=111,
            prompt_tokens_details=_usage_obj(cache_write_tokens=4403),
        )
        apply_cache_usage(raw, usage)
        assert usage.cache_creation_tokens == 4403

    def test_cost_field_populates_cost_usd(self):
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(prompt_tokens=10, cost=0.0042)
        apply_cache_usage(raw, usage)
        assert usage.cost_usd == pytest.approx(0.0042)

    def test_extras_via_model_extra(self):
        """Real OpenAI SDK Pydantic objects route unknown fields through
        ``model_extra``; the helper must read both paths."""
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()

        class FakeUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150
            prompt_tokens_details = None
            model_extra = {
                "cache_creation_input_tokens": 250,
                "cost": 0.01,
            }
        apply_cache_usage(FakeUsage(), usage)
        assert usage.cache_creation_tokens == 250
        assert usage.cost_usd == pytest.approx(0.01)

    def test_no_cache_fields_leaves_usage_untouched(self):
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        apply_cache_usage(_usage_obj(prompt_tokens=10), usage)
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None
        assert usage.cost_usd is None

    def test_zero_or_negative_cached_tokens_ignored(self):
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        usage = TokenUsage()
        raw = _usage_obj(
            prompt_tokens_details=_usage_obj(cached_tokens=0),
            cache_creation_input_tokens=0,
        )
        apply_cache_usage(raw, usage)
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None


class TestExtractUsageEnd2End:
    def test_full_usage_with_cache_fields(self):
        response = MagicMock()
        response.usage = _usage_obj(
            prompt_tokens=2000,
            completion_tokens=300,
            total_tokens=2300,
            prompt_tokens_details=_usage_obj(cached_tokens=1800),
            cache_creation_input_tokens=200,
            cost=0.0125,
        )
        usage = extract_usage(response)
        # 2000 on the wire, of which 1800 were read from cache and 200
        # written to it — OpenRouter counts both INSIDE ``prompt_tokens``
        # and ``TokenUsage`` counts them beside it, so nothing is left as
        # new input here.  See issue #758.
        assert usage.prompt_tokens == 0
        assert usage.output_tokens == 300
        assert usage.total_tokens == 2300
        assert usage.cache_read_tokens == 1800
        assert usage.cache_creation_tokens == 200
        assert usage.cost_usd == pytest.approx(0.0125)


# ==================== End-to-end ====================


@pytest.fixture
def configured_provider():
    provider = OpenRouterProvider()
    config = ProviderConfig(
        api_key="sk-or-test",
        extra={"api_params": {"cache_prompt": "auto"}},
    )
    with patch.dict(
        "os.environ",
        {"JAATO_OPENROUTER_API_KEY": "sk-or-test"},
        clear=False,
    ):
        provider.initialize(config)
    return provider


class TestEndToEndCacheWiring:
    def _stub_completion(self, provider, capture):
        """Replace ``provider._client.chat.completions.create`` with a stub
        that records its kwargs and returns a minimal valid response."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "ok"
        response.choices[0].message.tool_calls = []
        response.choices[0].message.reasoning = None
        response.choices[0].message.reasoning_content = None
        response.choices[0].finish_reason = "stop"
        response.usage = _usage_obj(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=_usage_obj(cached_tokens=80),
            cache_creation_input_tokens=20,
            cost=0.001,
        )

        def stub(**kwargs):
            capture["model"] = kwargs.get("model")
            capture["messages"] = kwargs.get("messages")
            capture["tools"] = kwargs.get("tools")
            capture["extra_body"] = kwargs.get("extra_body")
            return response

        provider._client = MagicMock()
        provider._client.chat.completions.create = stub

    def test_anthropic_model_stamps_breakpoints(self, configured_provider):
        provider = configured_provider
        provider.connect("anthropic/claude-3.5-sonnet", skip_model_test=True)
        captured: dict = {}
        self._stub_completion(provider, captured)

        provider.complete(
            messages=[Message(role=Role.USER, parts=[Part(text="hi")])],
            system_instruction="long system prompt",
            tools=[
                ToolSchema(name="zebra", description="z", parameters={"type": "object"}),
                ToolSchema(name="alpha", description="a", parameters={"type": "object"}),
            ],
        )

        system_msg = captured["messages"][0]
        assert system_msg["role"] == "system"
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

        tools = captured["tools"]
        wire_names = [t["function"]["name"] for t in tools]
        assert wire_names == sorted(wire_names)
        assert "cache_control" not in tools[0]
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}

        assert captured["extra_body"]["usage"] == {"include": True}

    def test_openai_model_omits_breakpoints(self, configured_provider):
        provider = configured_provider
        provider.connect("openai/gpt-4o", skip_model_test=True)
        captured: dict = {}
        self._stub_completion(provider, captured)

        provider.complete(
            messages=[Message(role=Role.USER, parts=[Part(text="hi")])],
            system_instruction="long system prompt",
            tools=[
                ToolSchema(name="zebra", description="z", parameters={"type": "object"}),
            ],
        )

        system_msg = captured["messages"][0]
        assert system_msg == {"role": "system", "content": "long system prompt"}
        assert "cache_control" not in captured["tools"][0]
        # Detailed-usage opt-in still goes on the wire for automatic-cache
        # upstreams so we report cached_tokens / cost from the response.
        assert captured["extra_body"]["usage"] == {"include": True}

    def test_force_off_skips_breakpoints_for_anthropic(self):
        provider = OpenRouterProvider()
        config = ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"cache_prompt": False}},
        )
        provider.initialize(config)
        provider.connect("anthropic/claude-3.5-sonnet", skip_model_test=True)
        captured: dict = {}
        self._stub_completion(provider, captured)

        provider.complete(
            messages=[Message(role=Role.USER, parts=[Part(text="hi")])],
            system_instruction="long system prompt",
            tools=[
                ToolSchema(name="alpha", description="a", parameters={"type": "object"}),
            ],
        )
        system_msg = captured["messages"][0]
        assert system_msg == {"role": "system", "content": "long system prompt"}
        assert "cache_control" not in captured["tools"][0]

    def test_force_on_stamps_breakpoints_for_openai(self):
        provider = OpenRouterProvider()
        config = ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"cache_prompt": True}},
        )
        provider.initialize(config)
        provider.connect("openai/gpt-4o", skip_model_test=True)
        captured: dict = {}
        self._stub_completion(provider, captured)

        provider.complete(
            messages=[Message(role=Role.USER, parts=[Part(text="hi")])],
            system_instruction="long system prompt",
            tools=[
                ToolSchema(name="alpha", description="a", parameters={"type": "object"}),
            ],
        )
        system_msg = captured["messages"][0]
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert captured["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_invalid_cache_ttl_raises(self):
        provider = OpenRouterProvider()
        config = ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"cache_ttl": "10m"}},
        )
        with pytest.raises(ValueError, match="cache_ttl"):
            provider.initialize(config)

    def test_extended_ttl_emits_1h_breakpoint(self):
        provider = OpenRouterProvider()
        config = ProviderConfig(
            api_key="sk-or-test",
            extra={"api_params": {"cache_prompt": True, "cache_ttl": "1h"}},
        )
        provider.initialize(config)
        provider.connect("anthropic/claude-3.5-sonnet", skip_model_test=True)
        captured: dict = {}
        self._stub_completion(provider, captured)

        provider.complete(
            messages=[Message(role=Role.USER, parts=[Part(text="hi")])],
            system_instruction="long system prompt",
            tools=[
                ToolSchema(name="alpha", description="a", parameters={"type": "object"}),
            ],
        )
        assert (
            captured["messages"][0]["content"][0]["cache_control"]
            == {"type": "ephemeral", "ttl": "1h"}
        )
        assert (
            captured["tools"][-1]["cache_control"]
            == {"type": "ephemeral", "ttl": "1h"}
        )

    def test_response_usage_populates_cache_tokens(self, configured_provider):
        provider = configured_provider
        provider.connect("anthropic/claude-3.5-sonnet", skip_model_test=True)
        captured: dict = {}
        self._stub_completion(provider, captured)

        provider.complete(
            messages=[Message(role=Role.USER, parts=[Part(text="hi")])],
            system_instruction="sys",
            tools=None,
        )
        usage = provider.get_token_usage()
        assert usage.cache_read_tokens == 80
        assert usage.cache_creation_tokens == 20
        assert usage.cost_usd == pytest.approx(0.001)
