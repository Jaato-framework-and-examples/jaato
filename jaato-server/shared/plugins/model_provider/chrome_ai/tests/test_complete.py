"""Turn-level tests for ChromeAIProvider.complete()."""

import pytest

from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
    ToolSchema,
    TurnOutcome,
)

from shared.tool_id_map import name_to_id

from shared.plugins.model_provider.chrome_ai.errors import (
    ChromeAIConnectionError,
)
from .conftest import user_message


WEATHER_TOOL = ToolSchema(
    name="get_weather",
    description="Look up the weather",
    parameters={"type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]},
)


class TestTextTurns:
    def test_simple_text(self, fake_bridge, connected_provider):
        fake_bridge.turn_scripts = [[
            {"type": "chunk", "text": "Hel"},
            {"type": "chunk", "text": "lo!"},
            {"type": "done", "text": "Hello!",
             "usage": {"inputUsage": 100, "inputQuota": 9216}},
        ]]
        chunks = []
        result = connected_provider.complete(
            [user_message("hi")], on_chunk=chunks.append)
        assert result.is_success
        assert result.text == "Hello!"
        assert chunks == ["Hel", "lo!"]

    def test_usage_reported(self, fake_bridge, connected_provider):
        fake_bridge.turn_scripts = [[
            {"type": "done", "text": "ok",
             "usage": {"inputUsage": 123, "inputQuota": 9216}},
        ]]
        usages = []
        connected_provider.complete([user_message("hi")],
                                    on_usage_update=usages.append)
        assert len(usages) == 1
        assert usages[0].total_tokens == 123  # session-reported usage wins
        assert usages[0].output_tokens >= 1
        assert connected_provider.get_token_usage() is usages[0]

    def test_system_and_history_in_initial_prompts(self, fake_bridge,
                                                   connected_provider):
        history = [
            user_message("first"),
            Message.from_text(Role.MODEL, "reply"),
            user_message("second"),
        ]
        connected_provider.complete(history, system_instruction="Be terse.")
        payload = fake_bridge.payloads[-1]
        roles = [m["role"] for m in payload["initialPrompts"]]
        assert roles == ["system", "user", "assistant"]
        assert payload["initialPrompts"][0]["content"] == "Be terse."
        assert payload["prompt"] == "second"

    def test_sampling_knobs_forwarded(self, fake_bridge, make_provider):
        provider = make_provider({
            "cdp_url": "http://localhost:9222",
            "api_params": {"temperature": 0.2, "top_k": 5},
        })
        provider.connect("gemini-nano")
        provider.complete([user_message("hi")])
        payload = fake_bridge.payloads[-1]
        assert payload["temperature"] == 0.2
        assert payload["topK"] == 5

    def test_sampling_omitted_when_unset(self, fake_bridge,
                                         connected_provider):
        connected_provider.complete([user_message("hi")])
        payload = fake_bridge.payloads[-1]
        assert "temperature" not in payload
        assert "topK" not in payload


class TestToolTurns:
    def test_tool_call_parsed(self, fake_bridge, connected_provider):
        # The model emits the HASHED wire id (that's all it ever sees).
        wire_id = name_to_id("get_weather")
        text = ('I will check.\n```tool_call\n'
                '{"name": "' + wire_id + '", "arguments": {"city": "Oslo"}}'
                '\n```')
        fake_bridge.turn_scripts = [[{"type": "done", "text": text,
                                      "usage": {}}]]
        seen = []
        result = connected_provider.complete(
            [user_message("weather in Oslo?")], tools=[WEATHER_TOOL],
            on_function_call=seen.append)
        assert result.outcome == TurnOutcome.TOOL_USE
        calls = result.response.get_function_calls()
        assert len(calls) == 1
        assert calls[0].name == "get_weather"  # mapped back to the human name
        assert calls[0].args == {"city": "Oslo"}
        assert calls[0].id  # correlation id present
        assert seen == calls
        assert result.response.get_text().strip() == "I will check."

    def test_tool_section_in_system_prompt(self, fake_bridge,
                                           connected_provider):
        connected_provider.complete([user_message("hi")],
                                    system_instruction="Be terse.",
                                    tools=[WEATHER_TOOL])
        system = fake_bridge.payloads[-1]["initialPrompts"][0]
        assert system["role"] == "system"
        assert "Be terse." in system["content"]
        # Hashed wire id on the wire; human name never leaks to the model.
        assert name_to_id("get_weather") in system["content"]
        assert "get_weather" not in system["content"]
        assert "tool_call" in system["content"]

    def test_no_tools_no_parsing(self, fake_bridge, connected_provider):
        text = '```tool_call\n{"name": "x", "arguments": {}}\n```'
        fake_bridge.turn_scripts = [[{"type": "done", "text": text,
                                      "usage": {}}]]
        result = connected_provider.complete([user_message("hi")])
        # Without tools the fence is plain text, not a call.
        assert result.outcome == TurnOutcome.RESPONSE
        assert "tool_call" in result.text

    def test_tool_result_history_becomes_prompt(self, fake_bridge,
                                                connected_provider):
        history = [
            user_message("weather in Oslo?"),
            Message(role=Role.MODEL, parts=[Part(
                function_call=FunctionCall(id="c1", name="get_weather",
                                           args={"city": "Oslo"}))]),
            Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
                call_id="c1", name="get_weather",
                result={"temp_c": -3}))]),
        ]
        connected_provider.complete(history, tools=[WEATHER_TOOL])
        payload = fake_bridge.payloads[-1]
        # The tool result is the continuation prompt, labeled by wire id...
        assert name_to_id("get_weather") in payload["prompt"]
        assert "get_weather" not in payload["prompt"]
        assert '"temp_c": -3' in payload["prompt"]
        # ...and the model's own call is replayed in fence syntax.
        assistant = [m for m in payload["initialPrompts"]
                     if m["role"] == "assistant"]
        assert any("```tool_call" in m["content"] for m in assistant)


class TestStructuredOutput:
    def test_schema_forwarded_and_parsed(self, fake_bridge,
                                         connected_provider):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        fake_bridge.turn_scripts = [[{"type": "done", "text": '{"ok": true}',
                                      "usage": {}}]]
        result = connected_provider.complete([user_message("hi")],
                                             response_schema=schema)
        assert fake_bridge.payloads[-1]["responseConstraint"] == schema
        assert result.response.structured_output == {"ok": True}

    def test_invalid_json_leaves_structured_none(self, fake_bridge,
                                                 connected_provider):
        fake_bridge.turn_scripts = [[{"type": "done", "text": "not json",
                                      "usage": {}}]]
        result = connected_provider.complete(
            [user_message("hi")], response_schema={"type": "object"})
        assert result.is_success
        assert result.response.structured_output is None


class TestCancellation:
    def test_pre_cancelled_token(self, connected_provider):
        token = CancelToken()
        token.cancel("user stop")
        result = connected_provider.complete([user_message("hi")],
                                             cancel_token=token)
        assert result.is_cancelled

    def test_cancelled_mid_turn(self, fake_bridge, connected_provider):
        fake_bridge.turn_scripts = [[
            {"type": "chunk", "text": "partial"},
            {"type": "cancelled"},
        ]]
        result = connected_provider.complete([user_message("hi")],
                                             cancel_token=CancelToken())
        assert result.is_cancelled
        assert result.response.finish_reason == FinishReason.CANCELLED
        assert result.response.get_text() == "partial"

    def test_cancel_token_wires_abort(self, fake_bridge, connected_provider):
        fake_bridge.turn_scripts = [[{"type": "cancelled"}]]
        token = CancelToken()
        connected_provider.complete([user_message("hi")], cancel_token=token)
        token.cancel("late")  # the registered callback fires bridge.abort
        assert fake_bridge.aborted


class TestErrors:
    def test_generation_error_is_error_result(self, fake_bridge,
                                              connected_provider):
        fake_bridge.turn_scripts = [[
            {"type": "error", "message": "input too large"},
        ]]
        result = connected_provider.complete([user_message("hi")])
        assert result.is_error
        assert "input too large" in (result.error_message or "")

    def test_turn_timeout_raises_transient(self, fake_bridge, make_provider):
        provider = make_provider({"cdp_url": "http://localhost:9222",
                                  "turn_timeout": 0.2})
        provider.connect("gemini-nano")
        fake_bridge.turn_scripts = [[]]  # silence
        with pytest.raises(ChromeAIConnectionError, match="no events"):
            provider.complete([user_message("hi")])
        assert fake_bridge.aborted  # the hung run was aborted

    def test_dead_browser_reconnects(self, fake_bridge, connected_provider):
        assert fake_bridge.connect_calls == 1
        fake_bridge.alive = False
        fake_bridge.turn_scripts = [[{"type": "done", "text": "back",
                                      "usage": {}}]]
        result = connected_provider.complete([user_message("hi")])
        assert result.is_success
        assert fake_bridge.connect_calls == 2
