"""Unit tests for the shared prose-emulated tool-calling machinery
(``model_provider/_prose_tools.py``).

The module is consumed by chrome_ai (always) and by the OpenAI-compat /
openrouter providers behind the ``prose_tool_calls`` quirk; these tests
cover it once at the source so every consumer inherits the guarantees
(hashed wire ids, parallel tool results, graceful parse degradation).
"""

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

from shared.tool_id_map import name_to_id
from shared.plugins.model_provider._prose_tools import (
    NON_TEXT_PLACEHOLDER,
    augment_system_with_tools,
    message_to_prose_chat,
    messages_to_prose_chat,
    parse_tool_calls,
    read_prose_tool_calls_quirk,
    rewrite_prose_tool_calls,
    serialize_tool_call,
    tool_schemas_to_prompt,
)


TOOLS = [
    ToolSchema(name="get_weather", description="Weather lookup",
               parameters={"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}),
    ToolSchema(name="run_query", description="SQL query",
               parameters={"type": "object",
                           "properties": {"sql": {"type": "string"}}}),
]


class TestToolPrompt:
    def test_empty_tools(self):
        assert tool_schemas_to_prompt([]) == ""

    def test_contains_hashed_ids_and_protocol(self):
        prompt = tool_schemas_to_prompt(TOOLS)
        assert name_to_id("get_weather") in prompt
        assert name_to_id("run_query") in prompt
        assert "Weather lookup" in prompt
        assert "```tool_call" in prompt
        assert '"city"' in prompt  # compact schema included
        assert "Never invent an id" in prompt
        # No concrete example id in the preamble: a tiny model copies it
        # verbatim and then reports "tool <example> not available".
        assert "t_1a2b3c4d" not in prompt

    def test_human_names_never_reach_the_model(self):
        # The framework-wide no-human-name-on-the-wire contract (see
        # shared/tool_id_map.py) applies to the prompt-injected section
        # exactly as it does to real tools arrays.
        prompt = tool_schemas_to_prompt(TOOLS)
        assert "get_weather" not in prompt
        assert "run_query" not in prompt

    def test_augment_combines_instruction_and_tools(self):
        combined = augment_system_with_tools("Be terse.", TOOLS)
        assert combined.startswith("Be terse.")
        assert name_to_id("get_weather") in combined

    def test_augment_without_tools(self):
        assert augment_system_with_tools("Be terse.", None) == "Be terse."
        assert augment_system_with_tools(None, None) is None


class TestMessageConversion:
    def test_user_text(self):
        out = message_to_prose_chat(Message.from_text(Role.USER, "hello"))
        assert out == [{"role": "user", "content": "hello"}]

    def test_model_maps_to_assistant(self):
        out = message_to_prose_chat(Message.from_text(Role.MODEL, "hi"))
        assert out == [{"role": "assistant", "content": "hi"}]

    def test_function_call_replays_as_hashed_fence(self):
        msg = Message(role=Role.MODEL, parts=[
            Part(text="Checking."),
            Part(function_call=FunctionCall(id="c1", name="lookup",
                                            args={"q": "x"})),
        ])
        (wire,) = message_to_prose_chat(msg)
        assert serialize_tool_call(name_to_id("lookup"),
                                   {"q": "x"}) in wire["content"]
        assert "lookup" not in wire["content"]

    def test_parallel_tool_results_emit_one_message_each(self):
        msg = Message(role=Role.TOOL, parts=[
            Part(function_response=ToolResult(call_id="c1", name="a",
                                              result={"n": 1})),
            Part(function_response=ToolResult(call_id="c2", name="b",
                                              result={"n": 2})),
        ])
        out = message_to_prose_chat(msg)
        assert len(out) == 2
        assert all(m["role"] == "user" for m in out)
        assert '"n": 1' in out[0]["content"] and "c1" in out[0]["content"]
        assert '"n": 2' in out[1]["content"] and "c2" in out[1]["content"]

    def test_thoughts_never_replayed(self):
        msg = Message(role=Role.MODEL, parts=[
            Part(thought="secret"), Part(text="visible"),
        ])
        (wire,) = message_to_prose_chat(msg)
        assert wire["content"] == "visible"

    def test_inline_data_becomes_placeholder(self):
        msg = Message(role=Role.USER, parts=[
            Part(inline_data={"mime_type": "image/png", "data": b"x"}),
        ])
        (wire,) = message_to_prose_chat(msg)
        assert NON_TEXT_PLACEHOLDER in wire["content"]

    def test_history_flattens(self):
        out = messages_to_prose_chat([
            Message.from_text(Role.USER, "a"),
            Message(role=Role.TOOL, parts=[
                Part(function_response=ToolResult(call_id="c", name="t",
                                                  result="ok")),
            ]),
        ])
        assert [m["role"] for m in out] == ["user", "user"]


class TestParse:
    def test_no_fences(self):
        clean, calls = parse_tool_calls("just a plain answer")
        assert clean == "just a plain answer"
        assert calls == []

    def test_single_call(self):
        text = ('Let me check.\n```tool_call\n'
                '{"name": "t_deadbeef", "arguments": {"city": "Oslo"}}\n```')
        clean, calls = parse_tool_calls(text)
        assert clean == "Let me check."
        assert calls == [("t_deadbeef", {"city": "Oslo"})]

    def test_multiple_fences(self):
        text = ('```tool_call\n{"name": "a", "arguments": {}}\n```\n'
                'and\n'
                '```tool_call\n{"name": "b", "arguments": {"x": 1}}\n```')
        clean, calls = parse_tool_calls(text)
        assert calls == [("a", {}), ("b", {"x": 1})]
        assert clean == "and"

    def test_array_body_yields_multiple_calls(self):
        text = ('```tool_call\n'
                '[{"name": "a", "arguments": {}},'
                ' {"name": "b", "arguments": {"x": 1}}]\n```')
        _, calls = parse_tool_calls(text)
        assert calls == [("a", {}), ("b", {"x": 1})]

    def test_args_alias_and_defaults(self):
        assert parse_tool_calls(
            '```tool_call\n{"name": "a", "args": {"k": "v"}}\n```'
        )[1] == [("a", {"k": "v"})]
        assert parse_tool_calls(
            '```tool_call\n{"name": "a"}\n```')[1] == [("a", {})]
        assert parse_tool_calls(
            '```tool_call\n{"name": "a", "arguments": "raw"}\n```'
        )[1] == [("a", {"value": "raw"})]

    def test_malformed_json_left_in_text(self):
        clean, calls = parse_tool_calls(
            'Before\n```tool_call\nnot json\n```\nAfter')
        assert calls == []
        assert "not json" in clean  # visible, not swallowed
        assert "Before" in clean and "After" in clean

    def test_missing_name_left_in_text(self):
        clean, calls = parse_tool_calls(
            '```tool_call\n{"arguments": {}}\n```')
        assert calls == []
        assert '"arguments"' in clean

    def test_multiline_json_body(self):
        text = ('```tool_call\n{\n  "name": "q",\n'
                '  "arguments": {"sql": "SELECT 1"}\n}\n```')
        assert parse_tool_calls(text)[1] == [("q", {"sql": "SELECT 1"})]


def _text_response(text, finish=FinishReason.STOP):
    return ProviderResponse(parts=[Part.from_text(text)],
                            usage=TokenUsage(total_tokens=5),
                            finish_reason=finish)


class TestRewrite:
    def test_no_calls_returns_same_object(self):
        response = _text_response("plain answer")
        assert rewrite_prose_tool_calls(
            response, call_id_prefix="p") is response

    def test_calls_become_function_call_parts(self):
        wire_id = name_to_id("get_weather")
        response = _text_response(
            'Checking.\n```tool_call\n'
            '{"name": "' + wire_id + '", "arguments": {"city": "Oslo"}}\n```')
        out = rewrite_prose_tool_calls(response, call_id_prefix="stub")
        assert out.finish_reason == FinishReason.TOOL_USE
        calls = out.get_function_calls()
        assert len(calls) == 1
        assert calls[0].name == "get_weather"  # mapped back to human name
        assert calls[0].args == {"city": "Oslo"}
        assert calls[0].id.startswith("stub-prose-")
        assert out.get_text().strip() == "Checking."
        assert out.usage.total_tokens == 5  # usage preserved

    def test_hallucinated_id_passes_through(self):
        # Unknown ids survive id_to_name unchanged so jaato's executor can
        # return a structured unknown-tool error the model recovers from.
        response = _text_response(
            '```tool_call\n{"name": "t_00000000", "arguments": {}}\n```')
        out = rewrite_prose_tool_calls(response, call_id_prefix="p")
        assert out.get_function_calls()[0].name == "t_00000000"

    def test_cancelled_response_untouched(self):
        # Tools parsed from a truncated, user-aborted stream must not run.
        response = _text_response(
            '```tool_call\n{"name": "t_deadbeef", "arguments": {}}\n```',
            finish=FinishReason.CANCELLED)
        out = rewrite_prose_tool_calls(response, call_id_prefix="p")
        assert out is response
        assert out.get_function_calls() == []

    def test_unique_call_ids(self):
        response = _text_response(
            '```tool_call\n[{"name": "a", "arguments": {}},'
            ' {"name": "b", "arguments": {}}]\n```')
        out = rewrite_prose_tool_calls(response, call_id_prefix="p")
        ids = [c.id for c in out.get_function_calls()]
        assert len(ids) == len(set(ids)) == 2


class TestQuirkReader:
    def test_default_off(self):
        assert read_prose_tool_calls_quirk({}) is False
        assert read_prose_tool_calls_quirk(None) is False

    def test_enabled(self):
        extra = {"quirks": {"prose_tool_calls": True}}
        assert read_prose_tool_calls_quirk(extra) is True

    def test_non_dict_quirks_ignored(self):
        assert read_prose_tool_calls_quirk(
            {"quirks": "yes"}, provider="stub") is False
