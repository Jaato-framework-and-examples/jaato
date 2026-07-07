"""Tests for the prompt-injected tool section and tool_call parsing."""

from jaato_sdk.plugins.model_provider.types import ToolSchema

from shared.tool_id_map import name_to_id
from shared.plugins.model_provider.chrome_ai.converters import (
    tool_schemas_to_prompt,
)
from shared.plugins.model_provider.chrome_ai.tooling import parse_tool_calls


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
        assert "Never invent ids" in prompt

    def test_human_names_never_reach_the_model(self):
        # The framework-wide no-human-name-on-the-wire contract (see
        # shared/tool_id_map.py) applies to the prompt-injected section
        # exactly as it does to real tools arrays.
        prompt = tool_schemas_to_prompt(TOOLS)
        assert "get_weather" not in prompt
        assert "run_query" not in prompt


class TestParse:
    def test_no_fences(self):
        clean, calls = parse_tool_calls("just a plain answer")
        assert clean == "just a plain answer"
        assert calls == []

    def test_single_call(self):
        text = ('Let me check.\n'
                '```tool_call\n'
                '{"name": "t_deadbeef", "arguments": {"city": "Oslo"}}\n'
                '```')
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
                ' {"name": "b", "arguments": {"x": 1}}]\n'
                '```')
        _, calls = parse_tool_calls(text)
        assert calls == [("a", {}), ("b", {"x": 1})]

    def test_args_alias_accepted(self):
        text = '```tool_call\n{"name": "a", "args": {"k": "v"}}\n```'
        _, calls = parse_tool_calls(text)
        assert calls == [("a", {"k": "v"})]

    def test_missing_arguments_defaults_empty(self):
        text = '```tool_call\n{"name": "a"}\n```'
        _, calls = parse_tool_calls(text)
        assert calls == [("a", {})]

    def test_non_dict_arguments_wrapped(self):
        text = '```tool_call\n{"name": "a", "arguments": "raw"}\n```'
        _, calls = parse_tool_calls(text)
        assert calls == [("a", {"value": "raw"})]

    def test_malformed_json_left_in_text(self):
        text = 'Before\n```tool_call\nnot json at all\n```\nAfter'
        clean, calls = parse_tool_calls(text)
        assert calls == []
        assert "not json at all" in clean  # visible, not swallowed
        assert "Before" in clean and "After" in clean

    def test_missing_name_left_in_text(self):
        text = '```tool_call\n{"arguments": {}}\n```'
        clean, calls = parse_tool_calls(text)
        assert calls == []
        assert '"arguments"' in clean

    def test_hallucinated_ids_still_emitted(self):
        # Unknown ids pass through: after id_to_name (identity for unknown
        # ids) jaato's executor returns a structured unknown-tool error
        # the model can recover from.
        text = '```tool_call\n{"name": "t_00000000", "arguments": {}}\n```'
        _, calls = parse_tool_calls(text)
        assert calls == [("t_00000000", {})]

    def test_multiline_json_body(self):
        text = ('```tool_call\n'
                '{\n  "name": "run_query",\n'
                '  "arguments": {"sql": "SELECT 1"}\n}\n'
                '```')
        _, calls = parse_tool_calls(text)
        assert calls == [("run_query", {"sql": "SELECT 1"})]
