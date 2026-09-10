"""Tests for the Bedrock Converse wire conversion.

The Converse wire differs from every other wire in this tree in ways that are
each a 400 when got wrong rather than a degradation, so the cases here are
organised by rejection: a format outside a closed vocabulary, an empty
content array, a document name with a dot in it, a tool name that is not
``^[a-zA-Z0-9_-]{1,64}$``.
"""

import base64

import pytest

from jaato_sdk.plugins.model_provider.types import (
    Attachment,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
    ToolSchema,
)
from shared.tool_id_map import name_to_id

from ..converters import (
    deserialize_history,
    finish_reason_from_bedrock,
    message_to_bedrock,
    messages_to_bedrock,
    response_from_bedrock,
    serialize_history,
    system_to_bedrock,
    tool_choice_to_bedrock,
    tool_config_to_bedrock,
    tool_schemas_to_bedrock,
    usage_from_bedrock,
)

_PNG = b"\x89PNG\r\n\x1a\nfake-image"
_PDF = b"%PDF-1.4 fake %%EOF"
_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt fake"


# ---------------------------------------------------------------- media blocks

@pytest.mark.parametrize("mime,expected_key,expected_format", [
    ("image/png", "image", "png"),
    ("image/jpeg", "image", "jpeg"),
    ("image/jpg", "image", "jpeg"),      # the spelling browsers emit
    ("image/webp", "image", "webp"),
    ("application/pdf", "document", "pdf"),
    ("text/markdown", "document", "md"),
    ("audio/wav", "audio", "wav"),
    ("audio/mpeg", "audio", "mp3"),
    ("video/mp4", "video", "mp4"),
])
def test_each_mime_lands_in_its_own_block(mime, expected_key, expected_format):
    """A mime reaches the block Bedrock names for it, with Bedrock's format."""
    msg = Message(role=Role.USER, parts=[
        Part(inline_data={"mime_type": mime, "data": b"payload"})])
    block = message_to_bedrock(msg)["content"][0]
    assert expected_key in block
    assert block[expected_key]["format"] == expected_format
    assert block[expected_key]["source"]["bytes"] == b"payload"


@pytest.mark.parametrize("mime", [
    "image/svg+xml",     # Converse names four image formats; svg is not one
    "audio/amr",
    "application/zip",
    None,                # a part with no declared mime at all
])
def test_a_format_outside_the_vocabulary_is_withheld_not_relabelled(mime):
    """#829's invariant: an unnamed format is declined, never renamed.

    Relabelling is the outcome worse than either carrying the bytes or
    declining them, because the model is told it is looking at something it
    is not.
    """
    msg = Message(role=Role.USER, parts=[
        Part(inline_data={"mime_type": mime, "data": b"payload"})])
    assert message_to_bedrock(msg)["content"] == []


def test_raw_pcm_is_accepted_only_when_its_parameters_agree():
    """``pcm`` asserts s16le, so a contradicting parameter withholds."""
    agreeing = Message(role=Role.USER, parts=[Part(inline_data={
        "mime_type": "audio/pcm;rate=24000;channels=1", "data": b"raw"})])
    assert "audio" in message_to_bedrock(agreeing)["content"][0]

    contradicting = Message(role=Role.USER, parts=[Part(inline_data={
        "mime_type": "audio/pcm;bits=24", "data": b"raw"})])
    assert message_to_bedrock(contradicting)["content"] == []


def test_big_endian_pcm_is_refused():
    """RFC 2586 makes ``audio/L16`` big-endian; relabelled samples are noise."""
    msg = Message(role=Role.USER, parts=[
        Part(inline_data={"mime_type": "audio/L16;rate=16000", "data": b"raw"})])
    assert message_to_bedrock(msg)["content"] == []


def test_a_document_name_is_sanitised_for_bedrocks_charset():
    """Bedrock rejects a document name outside ``[alnum \\s-()[]]``.

    ``report.pdf`` — the obvious value to pass — is a 400, so the dots have
    to go before the request does.
    """
    msg = Message(role=Role.USER, parts=[Part(inline_data={
        "mime_type": "application/pdf", "data": _PDF,
        "display_name": "q3/report.final.pdf"})])
    name = message_to_bedrock(msg)["content"][0]["document"]["name"]
    assert name == "q3 report final pdf"


def test_a_document_with_no_name_still_gets_one():
    """The member is required, and an empty string is also a 400."""
    msg = Message(role=Role.USER, parts=[
        Part(inline_data={"mime_type": "application/pdf", "data": _PDF})])
    assert message_to_bedrock(msg)["content"][0]["document"]["name"]


# ---------------------------------------------------------------- tool results

def test_a_tool_result_carries_its_text_then_its_attachments():
    result = ToolResult(call_id="call-1", name="readFile",
                        result={"path": "x.png"},
                        attachments=[Attachment(mime_type="image/png",
                                                data=_PNG, display_name="x.png")])
    block = message_to_bedrock(
        Message(role=Role.TOOL, parts=[Part(function_response=result)]),
    )["content"][0]["toolResult"]
    assert block["toolUseId"] == "call-1"
    assert "text" in block["content"][0]
    assert block["content"][1]["image"]["source"]["bytes"] == _PNG
    assert "status" not in block           # success is the default


def test_a_failed_tool_result_says_so():
    result = ToolResult(call_id="c", name="t", result="boom", is_error=True)
    block = message_to_bedrock(
        Message(role=Role.TOOL, parts=[Part(function_response=result)]),
    )["content"][0]["toolResult"]
    assert block["status"] == "error"


def test_an_unsupported_attachment_becomes_a_note_not_a_silence():
    """A model handed a result whose picture vanished answers as though the
    tool returned nothing, so the withhold is stated."""
    result = ToolResult(call_id="c", name="t", result={"ok": True},
                        attachments=[Attachment(mime_type="application/zip",
                                                data=b"pk", display_name="a.zip")])
    block = message_to_bedrock(
        Message(role=Role.TOOL, parts=[Part(function_response=result)]),
    )["content"][0]["toolResult"]
    note = block["content"][-1]["text"]
    assert "withheld" in note.lower()
    assert "a.zip" in note
    assert b"pk" not in str(block).encode()


def test_a_tool_result_is_never_empty():
    """Converse rejects an empty ``toolResult.content`` array."""
    result = ToolResult(call_id="c", name="t", result="")
    block = message_to_bedrock(
        Message(role=Role.TOOL, parts=[Part(function_response=result)]),
    )["content"][0]["toolResult"]
    assert block["content"]


def test_untrusted_results_keep_their_boundary():
    """The untrusted wrapper is model-facing text, so it must survive the
    conversion that produces the model-facing text."""
    result = ToolResult(call_id="c", name="web_fetch", result="hello",
                        untrusted=True, untrusted_source="web_fetch")
    block = message_to_bedrock(
        Message(role=Role.TOOL, parts=[Part(function_response=result)]),
    )["content"][0]["toolResult"]
    assert "web_fetch" in block["content"][0]["text"]


# ---------------------------------------------------------------- message list

def test_consecutive_same_role_messages_are_merged():
    """Converse requires alternating roles, and a tool-result turn is several
    internal messages that all become ``user``."""
    msgs = [
        Message(role=Role.MODEL, parts=[Part.from_function_call(
            FunctionCall(id="c", name="t", args={}))]),
        Message(role=Role.USER, parts=[Part(text="a")]),
        Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
            call_id="c", name="t", result="r"))]),
        Message(role=Role.MODEL, parts=[Part(text="b")]),
    ]
    wire = messages_to_bedrock(msgs)
    assert [m["role"] for m in wire] == ["assistant", "user", "assistant"]
    # The USER text and the TOOL result merged into one `user` message.
    assert len(wire[1]["content"]) == 2


def test_a_message_that_converts_to_nothing_is_dropped():
    """An empty ``content`` array is a 400; the merge above makes dropping safe."""
    msgs = [
        Message(role=Role.USER, parts=[Part(text="a")]),
        Message(role=Role.MODEL, parts=[Part(thought="deliberately unsent")]),
        Message(role=Role.USER, parts=[Part(text="b")]),
    ]
    wire = messages_to_bedrock(msgs)
    assert [m["role"] for m in wire] == ["user"]
    assert [b["text"] for b in wire[0]["content"]] == ["a", "b"]


def test_an_empty_text_part_produces_no_block():
    """Converse rejects an empty ``text`` member."""
    msg = Message(role=Role.USER, parts=[Part(text="")])
    assert message_to_bedrock(msg)["content"] == []


def test_a_function_call_sends_its_arguments_as_an_object():
    """Unlike OpenAI's ``arguments``, Converse's ``input`` is a document."""
    msg = Message(role=Role.MODEL, parts=[Part.from_function_call(
        FunctionCall(id="tu-1", name="readFile", args={"path": "x"}))])
    use = message_to_bedrock(msg)["content"][0]["toolUse"]
    assert use["input"] == {"path": "x"}
    assert use["toolUseId"] == "tu-1"


# ---------------------------------------------------------------- tool config

def test_tool_names_are_hashed_on_both_surfaces():
    """The wire enforces ``^[a-zA-Z0-9_-]{1,64}$``, which an MCP name fails,
    and the model must not read behaviour off the string either."""
    schemas = [ToolSchema(name="mcp.server.readFile", description="d",
                          parameters={"type": "object", "properties": {}})]
    wired = tool_schemas_to_bedrock(schemas)
    assert wired[0]["toolSpec"]["name"] == name_to_id("mcp.server.readFile")

    choice = tool_choice_to_bedrock(
        {"type": "function", "function": {"name": "mcp.server.readFile"}})
    assert choice == {"tool": {"name": name_to_id("mcp.server.readFile")}}


@pytest.mark.parametrize("given,expected", [
    ("auto", {"auto": {}}),
    ("required", {"any": {}}),
    ("any", {"any": {}}),
    ("none", None),          # Converse says this by sending no toolConfig
    (None, None),
])
def test_tool_choice_strings_translate(given, expected):
    assert tool_choice_to_bedrock(given) == expected


def test_a_tool_choice_without_tools_is_not_emitted():
    """A ``toolChoice`` with no ``tools`` is a 400 on this wire."""
    assert tool_config_to_bedrock(None, "required") is None


def test_cache_points_close_the_system_array_and_the_tool_list():
    """Everything BEFORE a cachePoint is cacheable, so the marker goes last."""
    system = system_to_bedrock("be helpful", cache=True, cache_ttl="1h")
    assert system[-1] == {"cachePoint": {"type": "default", "ttl": "1h"}}

    config = tool_config_to_bedrock(
        [ToolSchema(name="t", description="d", parameters={})],
        cache=True, cache_ttl="1h")
    assert config["tools"][-1] == {"cachePoint": {"type": "default", "ttl": "1h"}}


def test_an_unrecognised_cache_ttl_is_dropped_not_forwarded():
    """Bedrock rejects a TTL outside its two-value enum."""
    system = system_to_bedrock("s", cache=True, cache_ttl="90m")
    assert system[-1] == {"cachePoint": {"type": "default"}}


def test_no_cache_points_unless_asked():
    assert system_to_bedrock("s") == [{"text": "s"}]


# ---------------------------------------------------------------- responses

def test_a_response_becomes_parts_reasoning_and_usage():
    response = {
        "output": {"message": {"role": "assistant", "content": [
            {"reasoningContent": {"reasoningText": {"text": "hmm",
                                                    "signature": "sig"}}},
            {"text": "here you go"},
            {"toolUse": {"toolUseId": "tu-1", "name": name_to_id("readFile"),
                         "input": {"path": "x"}}},
        ]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 3, "totalTokens": 13,
                  "cacheReadInputTokens": 100},
    }
    parsed = response_from_bedrock(response)
    assert parsed.get_text() == "here you go"
    assert parsed.thinking == "hmm"
    assert parsed.finish_reason == FinishReason.TOOL_USE
    calls = parsed.get_function_calls()
    assert calls[0].name == "readFile"      # the hash is resolved back
    assert parsed.usage.cache_read_tokens == 100


def test_usage_buckets_stay_disjoint():
    """TokenUsage documents prompt_tokens as EXCLUDING the cache buckets,
    and Bedrock reports them the same way."""
    usage = usage_from_bedrock({"inputTokens": 5, "outputTokens": 2,
                                "cacheReadInputTokens": 90,
                                "cacheWriteInputTokens": 7})
    assert (usage.prompt_tokens, usage.cache_read_tokens,
            usage.cache_creation_tokens) == (5, 90, 7)


@pytest.mark.parametrize("reason,expected", [
    ("end_turn", FinishReason.STOP),
    ("stop_sequence", FinishReason.STOP),
    ("tool_use", FinishReason.TOOL_USE),
    ("max_tokens", FinishReason.MAX_TOKENS),
    ("guardrail_intervened", FinishReason.SAFETY),
    ("content_filtered", FinishReason.SAFETY),
])
def test_stop_reasons_map(reason, expected):
    assert finish_reason_from_bedrock(reason) == expected


def test_an_unknown_stop_reason_is_unknown_not_stop():
    """Bedrock adds reasons as it adds models; reading a new one as a clean
    finish is how a refusal becomes an answer."""
    assert finish_reason_from_bedrock("some_future_reason") == FinishReason.UNKNOWN
    assert finish_reason_from_bedrock(None) == FinishReason.UNKNOWN


# ---------------------------------------------------------------- persistence

def test_history_round_trips():
    history = [
        Message(role=Role.USER, parts=[
            Part(text="look"),
            Part(inline_data={"mime_type": "image/png", "data": _PNG})]),
        Message(role=Role.MODEL, parts=[
            Part(thought="thinking"),
            Part.from_function_call(FunctionCall(id="tu", name="t",
                                                 args={"a": 1}))]),
        Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
            call_id="tu", name="t", result={"ok": True}))]),
    ]
    restored = deserialize_history(serialize_history(history))
    assert [m.role for m in restored] == [Role.USER, Role.MODEL, Role.TOOL]
    assert restored[0].parts[1].inline_data["data"] == _PNG
    assert restored[1].parts[0].thought == "thinking"
    assert restored[2].parts[0].function_response.result == {"ok": True}


def test_serialized_binary_is_base64_so_the_json_is_valid():
    history = [Message(role=Role.USER, parts=[
        Part(inline_data={"mime_type": "image/png", "data": _PNG})])]
    assert base64.b64encode(_PNG).decode() in serialize_history(history)


# ------------------------------------------------- unpaired tool calls (a 400)

def _call(msg_id):
    return Message(role=Role.MODEL, parts=[
        Part(text="working"),
        Part.from_function_call(FunctionCall(id=msg_id, name="t", args={}))])


def _answer(msg_id):
    return Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
        call_id=msg_id, name="t", result="ok"))])


def _tool_ids(wire):
    return [b["toolUse"]["toolUseId"] for m in wire for b in m["content"]
            if "toolUse" in b]


def _result_ids(wire):
    return [b["toolResult"]["toolUseId"] for m in wire for b in m["content"]
            if "toolResult" in b]


def test_a_paired_tool_call_survives():
    wire = messages_to_bedrock([
        Message.from_text(Role.USER, "go"), _call("tu-1"), _answer("tu-1")])
    assert _tool_ids(wire) == ["tu-1"]
    assert _result_ids(wire) == ["tu-1"]


def test_an_unanswered_tool_call_is_dropped():
    """Converse rejects a toolUse with no toolResult, and one orphan makes
    every LATER request in that session a 400 — a session that cannot be
    recovered by talking to it."""
    wire = messages_to_bedrock([
        Message.from_text(Role.USER, "go"), _call("tu-1"),
        Message.from_text(Role.USER, "never mind")])
    assert _tool_ids(wire) == []


def test_the_text_beside_a_dropped_call_survives():
    """Dropping the orphan must not discard what the model said."""
    wire = messages_to_bedrock([
        Message.from_text(Role.USER, "go"), _call("tu-1"),
        Message.from_text(Role.USER, "never mind")])
    assert any(b.get("text") == "working" for m in wire for b in m["content"])


def test_a_result_answering_nothing_is_dropped():
    """The other direction: history that kept the answer but lost the call."""
    wire = messages_to_bedrock([
        Message.from_text(Role.USER, "go"), _answer("tu-ghost")])
    assert _result_ids(wire) == []
