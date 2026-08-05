"""Tests for jaato Message → Prompt API message conversion."""

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
)

from shared.tool_id_map import name_to_id
from shared.plugins.model_provider.chrome_ai.converters import (
    NON_TEXT_PLACEHOLDER,
    message_to_prompt_api,
    messages_to_prompt_api,
    serialize_tool_call,
)


def test_user_text():
    out = message_to_prompt_api(Message.from_text(Role.USER, "hello"))
    assert out == [{"role": "user", "content": "hello"}]


def test_model_maps_to_assistant():
    out = message_to_prompt_api(Message.from_text(Role.MODEL, "hi"))
    assert out == [{"role": "assistant", "content": "hi"}]


def test_model_function_call_round_trips_as_fence():
    msg = Message(role=Role.MODEL, parts=[
        Part(text="Checking."),
        Part(function_call=FunctionCall(id="c1", name="lookup",
                                        args={"q": "x"})),
    ])
    (wire,) = message_to_prompt_api(msg)
    assert wire["role"] == "assistant"
    assert "Checking." in wire["content"]
    # Replayed with the hashed wire id — the human name never reaches
    # the model, matching what it originally emitted.
    assert serialize_tool_call(name_to_id("lookup"), {"q": "x"}) in wire["content"]
    assert "lookup" not in wire["content"]


def test_parallel_tool_results_emit_one_message_each():
    # One TOOL message with N function_response parts must yield N wire
    # messages — dropping all but [0] is the classic silent-parallel bug.
    msg = Message(role=Role.TOOL, parts=[
        Part(function_response=ToolResult(call_id="c1", name="a",
                                          result={"n": 1})),
        Part(function_response=ToolResult(call_id="c2", name="b",
                                          result={"n": 2})),
    ])
    out = message_to_prompt_api(msg)
    assert len(out) == 2
    assert all(m["role"] == "user" for m in out)
    assert '"n": 1' in out[0]["content"] and "c1" in out[0]["content"]
    assert '"n": 2' in out[1]["content"] and "c2" in out[1]["content"]


def test_thoughts_are_never_replayed():
    msg = Message(role=Role.MODEL, parts=[
        Part(thought="secret reasoning"),
        Part(text="visible"),
    ])
    (wire,) = message_to_prompt_api(msg)
    assert "secret" not in wire["content"]
    assert wire["content"] == "visible"


def test_inline_data_becomes_placeholder():
    msg = Message(role=Role.USER, parts=[
        Part(text="look:"),
        Part(inline_data={"mime_type": "image/png", "data": b"\x89PNG"}),
    ])
    (wire,) = message_to_prompt_api(msg)
    assert NON_TEXT_PLACEHOLDER in wire["content"]


def test_empty_message_yields_nothing():
    assert message_to_prompt_api(Message(role=Role.MODEL, parts=[])) == []


def test_messages_to_prompt_api_flattens():
    out = messages_to_prompt_api([
        Message.from_text(Role.USER, "a"),
        Message(role=Role.TOOL, parts=[
            Part(function_response=ToolResult(call_id="c", name="t",
                                              result="ok")),
        ]),
    ])
    assert [m["role"] for m in out] == ["user", "user"]
