"""Integration tests for the ``prose_tool_calls`` model quirk.

Exercises the quirk through ``OpenAICompatProvider.complete()`` on a
minimal stub subclass with a fake OpenAI client (the same mocking
boundary the per-provider suites use), asserting the wire shape end to
end: no ``tools`` array, hashed ids in the system prompt, prose-replayed
tool history, and fenced ``tool_call`` responses parsed back into
``FunctionCall`` parts.  Also covers the openrouter provider's quirk
knob (its ``complete()`` shares the same ``_prose_tools`` seams).
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
    ToolSchema,
    TurnOutcome,
)

from shared.tool_id_map import name_to_id
from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider._openai_compat.base import (
    OpenAICompatProvider,
)


WEATHER_TOOL = ToolSchema(
    name="get_weather",
    description="Look up the weather",
    parameters={"type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]},
)


class _FakeClient:
    """Records ``chat.completions.create`` kwargs; returns a scripted reply."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.reply_text = "ok"
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.reply_text, tool_calls=None,
                                  reasoning_content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                  total_tokens=15,
                                  prompt_tokens_details=None),
        )

    def close(self) -> None:
        pass


class _StubError(Exception):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(str(kwargs))


class _StubProvider(OpenAICompatProvider):
    """Minimal concrete subclass — fake client, fixed context window."""

    _ERR_AUTHENTICATION = _StubError
    _ERR_RATE_LIMIT = _StubError
    _ERR_MODEL_NOT_FOUND = _StubError
    _ERR_CONTEXT_LIMIT = _StubError
    _ERR_INFRASTRUCTURE = _StubError

    def __init__(self) -> None:
        super().__init__()
        self.fake_client = _FakeClient()

    @property
    def name(self) -> str:
        return "stub"

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        self._base_url = "http://stub"
        self._api_key = "stub"

    def _detect_context(self, config: ProviderConfig) -> Optional[int]:
        return 32768

    def _context_error_message(self) -> str:
        return "unreachable"

    def _create_client(self) -> Any:
        return self.fake_client

    def verify_auth(self, allow_interactive=False, on_message=None,
                    config=None) -> bool:
        return True

    def get_auth_info(self) -> str:
        return "none (stub)"


def _make_provider(quirk: bool = True) -> _StubProvider:
    provider = _StubProvider()
    extra = {"quirks": {"prose_tool_calls": True}} if quirk else {}
    provider.initialize(ProviderConfig(extra=extra))
    provider.connect("cheap-model")
    return provider


def _user(text: str) -> Message:
    return Message.from_text(Role.USER, text)


class TestRequestShape:
    def test_no_tools_array_schemas_in_system_prompt(self):
        provider = _make_provider()
        provider.complete([_user("hi")], system_instruction="Be terse.",
                          tools=[WEATHER_TOOL])
        request = provider.fake_client.requests[-1]
        assert "tools" not in request
        system = request["messages"][0]
        assert system["role"] == "system"
        assert "Be terse." in system["content"]
        assert name_to_id("get_weather") in system["content"]
        assert "get_weather" not in system["content"]

    def test_quirk_off_keeps_native_tools_array(self):
        provider = _make_provider(quirk=False)
        provider.complete([_user("hi")], tools=[WEATHER_TOOL])
        request = provider.fake_client.requests[-1]
        assert "tools" in request

    def test_quirk_without_tools_is_inert(self):
        provider = _make_provider()
        provider.complete([_user("hi")], system_instruction="Be terse.")
        request = provider.fake_client.requests[-1]
        assert request["messages"][0]["content"] == "Be terse."

    def test_tool_history_replayed_as_prose(self):
        provider = _make_provider()
        history = [
            _user("weather in Oslo?"),
            Message(role=Role.MODEL, parts=[Part(
                function_call=FunctionCall(id="c1", name="get_weather",
                                           args={"city": "Oslo"}))]),
            Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
                call_id="c1", name="get_weather", result={"temp_c": -3}))]),
        ]
        provider.complete(history, tools=[WEATHER_TOOL])
        wire_messages = provider.fake_client.requests[-1]["messages"]
        # No native tool_calls / role:"tool" structures anywhere...
        assert all("tool_calls" not in m for m in wire_messages)
        assert all(m["role"] != "tool" for m in wire_messages)
        # ...the model's call is a fence, the result a user message.
        assistant = [m for m in wire_messages if m["role"] == "assistant"]
        assert any("```tool_call" in m["content"] for m in assistant)
        assert any('"temp_c": -3' in m["content"]
                   for m in wire_messages if m["role"] == "user")

    def test_tool_choice_dropped_in_prose_mode(self):
        # OpenAI rejects tool_choice without tools; the existing guard in
        # _apply_api_params must also cover the prose path (tools absent).
        provider = _make_provider()
        provider.complete([_user("hi")], tools=[WEATHER_TOOL],
                          tool_choice={"type": "function",
                                       "function": {"name": "get_weather"}})
        assert "tool_choice" not in provider.fake_client.requests[-1]


class TestResponseParsing:
    def test_fenced_call_becomes_function_call(self):
        provider = _make_provider()
        wire_id = name_to_id("get_weather")
        provider.fake_client.reply_text = (
            'Checking.\n```tool_call\n'
            '{"name": "' + wire_id + '", "arguments": {"city": "Oslo"}}\n```')
        result = provider.complete([_user("weather?")], tools=[WEATHER_TOOL])
        assert result.outcome == TurnOutcome.TOOL_USE
        calls = result.response.get_function_calls()
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].args == {"city": "Oslo"}
        assert result.response.get_text().strip() == "Checking."

    def test_plain_answer_stays_response(self):
        provider = _make_provider()
        provider.fake_client.reply_text = "It is cold."
        result = provider.complete([_user("weather?")], tools=[WEATHER_TOOL])
        assert result.outcome == TurnOutcome.RESPONSE
        assert result.text == "It is cold."

    def test_quirk_off_leaves_fences_as_text(self):
        provider = _make_provider(quirk=False)
        provider.fake_client.reply_text = (
            '```tool_call\n{"name": "x", "arguments": {}}\n```')
        result = provider.complete([_user("hi")], tools=[WEATHER_TOOL])
        assert result.outcome == TurnOutcome.RESPONSE
        assert "tool_call" in result.text


class TestOpenRouterKnob:
    def test_initialize_reads_quirk(self):
        # openrouter is standalone (not an OpenAICompatProvider subclass);
        # its complete() shares the same _prose_tools seams, so the knob
        # read is the integration point worth pinning here.
        from shared.plugins.model_provider.openrouter.provider import (
            OpenRouterProvider,
        )
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(
            api_key="sk-or-test",
            extra={"quirks": {"prose_tool_calls": True}},
        ))
        assert provider._prose_tool_calls is True

    def test_default_off(self):
        from shared.plugins.model_provider.openrouter.provider import (
            OpenRouterProvider,
        )
        provider = OpenRouterProvider()
        provider.initialize(ProviderConfig(api_key="sk-or-test"))
        assert provider._prose_tool_calls is False
