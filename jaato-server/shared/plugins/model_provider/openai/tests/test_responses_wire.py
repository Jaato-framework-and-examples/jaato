"""Tests for the Responses-API wire: converters, parsing, streaming.

The Responses API is a different shape from Chat Completions in every
place that matters (item list vs messages, flat tool definitions, typed
SSE events, differently-named usage fields), so it gets its own suite
rather than a couple of cases bolted onto the chat one.  The cases below
are organised by the claim they defend, not by the function they call.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from jaato_sdk.plugins.model_provider.types import (
    Attachment,
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    Role,
    StreamInterruptedError,
    ToolResult,
    ToolSchema,
)
from shared.tool_id_map import name_to_id

from ..provider import OpenAIProvider
from ..responses import _ResponsesAccumulator
from ..responses_converters import (
    finish_reason_from_response,
    history_to_responses,
    message_to_responses,
    parts_from_responses_output,
    thinking_from_responses_output,
    tool_schemas_to_responses,
    usage_from_responses,
)


# ==================== Fixtures / builders ====================

def _provider(api_mode="responses"):
    provider = OpenAIProvider()
    provider._client = MagicMock()
    provider._model_name = "gpt-5.1"
    provider._api_mode = api_mode
    provider._enable_thinking = True
    provider._trace = lambda _msg: None
    return provider


def _event(**fields):
    """A Responses stream event as a plain dict.

    Dicts rather than SDK models on purpose: ``_get`` must read both, and
    a suite built only on ``MagicMock`` would pass while attribute access
    silently invented every field it asked for.
    """
    return fields


def _stream(events):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(events)
    stream.close = MagicMock()
    return stream


def _completed(output=None, usage=None, status="completed", **extra):
    response = {"status": status, "output": output or [], "usage": usage}
    response.update(extra)
    return _event(type=f"response.{status}", response=response)


# ==================== Tool definitions ====================

class TestToolDefinitions:
    def test_the_shape_is_flat_not_nested(self):
        """Chat nests name/parameters under ``function``; Responses does not,
        and sending the chat shape here is a 400."""
        wire = tool_schemas_to_responses([ToolSchema(
            name="readFile", description="read a file",
            parameters={"type": "object", "properties": {}},
        )])
        assert wire[0]["type"] == "function"
        assert "function" not in wire[0]
        assert wire[0]["parameters"] == {"type": "object", "properties": {}}

    def test_the_human_name_never_reaches_the_wire(self):
        wire = tool_schemas_to_responses([ToolSchema(
            name="RESPONSES_LEAK_SENTINEL_toolName",
            description="d", parameters={},
        )])
        blob = json.dumps(wire)
        assert "RESPONSES_LEAK_SENTINEL_toolName" not in blob
        assert name_to_id("RESPONSES_LEAK_SENTINEL_toolName") in blob

    def test_no_tools_means_no_array(self):
        assert tool_schemas_to_responses([]) is None
        assert tool_schemas_to_responses(None) is None


# ==================== History → input items ====================

class TestHistoryConversion:
    def test_a_user_turn_becomes_input_text(self):
        items = message_to_responses(
            Message(role=Role.USER, parts=[Part(text="hello")]))
        assert items == [{
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }]

    def test_an_image_becomes_input_image(self):
        items = message_to_responses(Message(role=Role.USER, parts=[
            Part(text="what is this"),
            Part(inline_data={"mime_type": "image/png", "data": b"\x89PNG"}),
        ]))
        blocks = items[0]["content"]
        assert blocks[1]["type"] == "input_image"
        assert blocks[1]["image_url"].startswith("data:image/png;base64,")

    def test_a_pdf_becomes_input_file(self):
        items = message_to_responses(Message(role=Role.USER, parts=[
            Part(inline_data={"mime_type": "application/pdf",
                              "data": b"%PDF-1.4", "display_name": "d.pdf"}),
        ]))
        block = items[0]["content"][0]
        assert block["type"] == "input_file"
        assert block["filename"] == "d.pdf"

    def test_content_a_wire_cannot_carry_is_stated_not_dropped(self):
        """A model told "here is the file" and shown nothing confabulates."""
        items = message_to_responses(Message(role=Role.USER, parts=[
            Part(text="watch this"),
            Part(inline_data={"mime_type": "video/mp4", "data": b"\x00\x00"}),
        ]))
        texts = [b["text"] for b in items[0]["content"]
                 if b["type"] == "input_text"]
        assert any("Attachment withheld" in t for t in texts)

    def test_an_assistant_call_is_a_top_level_item_keyed_by_call_id(self):
        items = message_to_responses(Message(role=Role.MODEL, parts=[
            Part(text="calling"),
            Part(function_call=FunctionCall(
                id="call_9", name="readFile", args={"path": "x"})),
        ]))
        assert items[0]["role"] == "assistant"
        assert items[0]["content"][0]["type"] == "output_text"
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_9"
        assert items[1]["name"] == name_to_id("readFile")
        assert json.loads(items[1]["arguments"]) == {"path": "x"}

    def test_every_parallel_result_gets_its_own_output_item(self):
        """One internal TOOL message can carry N results; emitting only the
        first silently drops N-1 answers out of the conversation."""
        items = message_to_responses(Message(role=Role.TOOL, parts=[
            Part(function_response=ToolResult(
                call_id="c1", name="readFile", result={"ok": 1})),
            Part(function_response=ToolResult(
                call_id="c2", name="readFile", result={"ok": 2})),
        ]))
        assert [i["call_id"] for i in items] == ["c1", "c2"]
        assert all(i["type"] == "function_call_output" for i in items)

    def test_a_tool_results_image_follows_as_a_user_item(self):
        """``function_call_output.output`` is a string, so media cannot ride
        it; a follow-up user item is where a vision model can see it."""
        items = message_to_responses(Message(role=Role.TOOL, parts=[
            Part(function_response=ToolResult(
                call_id="c1", name="screenshot", result={"path": "s.png"},
                attachments=[Attachment(mime_type="image/png",
                                        data=b"\x89PNG", display_name="s.png")],
            )),
        ]))
        assert items[0]["type"] == "function_call_output"
        assert items[1]["role"] == "user"
        assert any(b["type"] == "input_image" for b in items[1]["content"])

    def test_history_flattens_in_order(self):
        items = history_to_responses([
            Message(role=Role.USER, parts=[Part(text="a")]),
            Message(role=Role.MODEL, parts=[Part(text="b")]),
        ])
        assert [i["role"] for i in items] == ["user", "assistant"]


# ==================== Response → internal ====================

class TestResponseParsing:
    def test_text_and_calls_come_back_in_wire_order(self):
        parts = parts_from_responses_output([
            {"type": "message", "content": [
                {"type": "output_text", "text": "thinking about it"}]},
            {"type": "function_call", "call_id": "c1",
             "name": name_to_id("readFile"), "arguments": '{"path": "x"}'},
        ])
        assert parts[0].text == "thinking about it"
        assert parts[1].function_call.name == "readFile"
        assert parts[1].function_call.id == "c1"

    def test_a_refusal_is_the_answer_not_an_empty_turn(self):
        parts = parts_from_responses_output([
            {"type": "message", "content": [
                {"type": "refusal", "refusal": "I can't help with that."}]},
        ])
        assert parts[0].text == "I can't help with that."

    def test_reasoning_items_do_not_become_parts(self):
        parts = parts_from_responses_output([
            {"type": "reasoning", "summary": [{"type": "summary_text",
                                               "text": "considering"}]},
        ])
        assert parts == []

    def test_reasoning_summaries_reach_the_thinking_channel(self):
        thinking = thinking_from_responses_output([
            {"type": "reasoning", "summary": [
                {"type": "summary_text", "text": "first "},
                {"type": "summary_text", "text": "second"}]},
        ])
        assert thinking == "first second"

    def test_unreadable_arguments_stay_unreadable(self):
        """The session refuses such a call and tells the model, rather than
        executing a zero-argument call it never made (#750)."""
        parts = parts_from_responses_output([
            {"type": "function_call", "call_id": "c1",
             "name": name_to_id("readFile"), "arguments": '{"path": '},
        ])
        assert parts[0].function_call.unreadable_args is not None

    def test_a_hallucinated_tool_id_is_recorded_as_itself(self):
        """Dressing an id the process never issued up as a resolved name
        hides the invention (#873)."""
        parts = parts_from_responses_output([
            {"type": "function_call", "call_id": "c1",
             "name": "t_deadbeef", "arguments": "{}"},
        ])
        assert parts[0].function_call.name == "t_deadbeef"


class TestUsage:
    def test_the_responses_field_names_are_read(self):
        usage = usage_from_responses(
            {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
        assert usage.prompt_tokens == 100
        assert usage.output_tokens == 20

    def test_cached_tokens_are_taken_out_of_the_prompt_total(self):
        """They are a SUBSET of ``input_tokens`` on this wire; left in, they
        are counted on both sides of every downstream sum (#758)."""
        usage = usage_from_responses({
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 80},
        })
        assert usage.cache_read_tokens == 80
        assert usage.prompt_tokens == 20

    def test_absent_usage_is_zero_not_a_crash(self):
        assert usage_from_responses(None).total_tokens == 0


class TestFinishReason:
    @pytest.mark.parametrize("response,expected", [
        ({"status": "completed"}, FinishReason.STOP),
        ({"status": "failed"}, FinishReason.ERROR),
        ({"status": "incomplete",
          "incomplete_details": {"reason": "max_output_tokens"}},
         FinishReason.MAX_TOKENS),
        ({"status": "incomplete",
          "incomplete_details": {"reason": "content_filter"}},
         FinishReason.SAFETY),
    ])
    def test_status_maps_to_the_internal_reason(self, response, expected):
        assert finish_reason_from_response(response) == expected

    def test_an_unmapped_status_is_unknown_not_incomplete(self):
        """``UNKNOWN`` means "ended, label unmapped" and stays a success;
        ``INCOMPLETE`` means "never ended" and is terminal (#687)."""
        assert finish_reason_from_response({"status": "queued"}) == \
            FinishReason.UNKNOWN

    def test_the_apis_incomplete_is_not_jaatos_INCOMPLETE(self):
        """The API's word for "ended early" must not collide with the
        framework's word for "never ended"."""
        reason = finish_reason_from_response(
            {"status": "incomplete",
             "incomplete_details": {"reason": "max_output_tokens"}})
        assert reason is not FinishReason.INCOMPLETE


# ==================== Streaming ====================

class TestStreaming:
    def test_text_deltas_reach_the_caller_as_they_arrive(self):
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _event(type="response.output_text.delta", output_index=0,
                   delta="Hel"),
            _event(type="response.output_text.delta", output_index=0,
                   delta="lo"),
            _completed(output=[{"type": "message", "content": [
                {"type": "output_text", "text": "Hello"}]}]),
        ])
        seen = []
        result = provider._complete_via_responses(
            [], None, None, on_chunk=seen.append)
        assert seen == ["Hel", "lo"]
        assert result.get_text() == "Hello"
        assert result.finish_reason == FinishReason.STOP

    def test_the_terminal_object_is_what_becomes_history(self):
        """Stream for the UX; parse the API's own account for the truth."""
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _event(type="response.output_text.delta", output_index=0,
                   delta="par"),
            _completed(output=[{"type": "message", "content": [
                {"type": "output_text", "text": "partial and then some"}]}]),
        ])
        result = provider._complete_via_responses([], None, None,
                                                  on_chunk=lambda _t: None)
        assert result.get_text() == "partial and then some"

    def test_a_tool_call_is_assembled_across_argument_deltas(self):
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _event(type="response.output_item.added", output_index=0,
                   item={"type": "function_call", "call_id": "c1",
                         "name": name_to_id("readFile"), "arguments": ""}),
            _event(type="response.function_call_arguments.delta",
                   output_index=0, delta='{"pa'),
            _event(type="response.function_call_arguments.delta",
                   output_index=0, delta='th": "x"}'),
            _completed(output=[{"type": "function_call", "call_id": "c1",
                                "name": name_to_id("readFile"),
                                "arguments": '{"path": "x"}'}]),
        ])
        result = provider._complete_via_responses([], None, None,
                                                  on_chunk=lambda _t: None)
        assert result.finish_reason == FinishReason.TOOL_USE
        call = result.get_function_calls()[0]
        assert call.name == "readFile"
        assert call.args == {"path": "x"}

    def test_reasoning_summaries_reach_on_thinking(self):
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _event(type="response.reasoning_summary_text.delta",
                   delta="weighing options"),
            _completed(),
        ])
        thoughts = []
        provider._complete_via_responses(
            [], None, None, on_chunk=lambda _t: None,
            on_thinking=thoughts.append)
        assert thoughts == ["weighing options"]

    def test_usage_is_reported_from_the_terminal_event(self):
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _completed(usage={"input_tokens": 7, "output_tokens": 3,
                              "total_tokens": 10}),
        ])
        updates = []
        result = provider._complete_via_responses(
            [], None, None, on_chunk=lambda _t: None,
            on_usage_update=updates.append)
        assert result.usage.total_tokens == 10
        assert updates and updates[0].total_tokens == 10

    def test_an_unknown_event_type_is_ignored_not_fatal(self):
        """The event vocabulary grows; raising on an addition would break a
        session for something that costs nothing to skip."""
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _event(type="response.some_future_thing.delta", delta="?"),
            _completed(output=[{"type": "message", "content": [
                {"type": "output_text", "text": "fine"}]}]),
        ])
        result = provider._complete_via_responses([], None, None,
                                                  on_chunk=lambda _t: None)
        assert result.get_text() == "fine"


class TestStreamTermination:
    def test_a_stream_that_stops_arriving_raises(self):
        """No terminal event and no cancellation is an interrupted turn, not
        a finished one (#687)."""
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _event(type="response.output_text.delta", output_index=0,
                   delta="half an ans"),
        ])
        with pytest.raises(StreamInterruptedError):
            provider._complete_via_responses([], None, None,
                                             on_chunk=lambda _t: None)

    def test_cancellation_is_not_an_interruption(self):
        provider = _provider()
        cancel = CancelToken()
        provider._client.responses.create = lambda **_kw: _stream([
            _event(type="response.output_text.delta", output_index=0,
                   delta="one"),
            _event(type="response.output_text.delta", output_index=0,
                   delta="two"),
        ])
        result = provider._complete_via_responses(
            [], None, None, cancel_token=cancel,
            on_chunk=lambda _t: cancel.cancel())
        assert result.finish_reason == FinishReason.CANCELLED

    def test_the_stream_is_closed_on_normal_completion(self):
        provider = _provider()
        captured = {}

        def create(**_kw):
            captured["stream"] = _stream([_completed()])
            return captured["stream"]

        provider._client.responses.create = create
        provider._complete_via_responses([], None, None,
                                         on_chunk=lambda _t: None)
        captured["stream"].close.assert_called_once()

    def test_cancelling_also_closes_the_client_pool(self):
        """``stream.close()`` alone does not send TCP-FIN, so the upstream
        keeps generating (and billing) the whole response."""
        provider = _provider()
        cancel = CancelToken()
        provider._client.responses.create = lambda **_kw: _stream([
            _event(type="response.output_text.delta", output_index=0,
                   delta="x"),
            _event(type="response.output_text.delta", output_index=0,
                   delta="y"),
        ])
        provider._complete_via_responses(
            [], None, None, cancel_token=cancel,
            on_chunk=lambda _t: cancel.cancel())
        provider._client.close.assert_called_once()


# ==================== Request assembly ====================

class TestRequestAssembly:
    def test_the_system_prompt_becomes_instructions(self):
        provider = _provider()
        captured = {}
        provider._client.responses.create = (
            lambda **kw: captured.update(kw) or _stream([_completed()]))
        provider._complete_via_responses(
            [], "You are helpful.", None, on_chunk=lambda _t: None)
        assert captured["instructions"] == "You are helpful."
        assert captured["input"] == []

    def test_store_is_false_so_the_history_stays_ours(self):
        provider = _provider()
        captured = {}
        provider._client.responses.create = (
            lambda **kw: captured.update(kw) or _stream([_completed()]))
        provider._complete_via_responses([], None, None,
                                         on_chunk=lambda _t: None)
        assert captured["store"] is False

    def test_a_profile_may_turn_server_side_storage_back_on(self):
        provider = _provider()
        provider._api_params = {"store": True}
        captured = {}
        provider._client.responses.create = (
            lambda **kw: captured.update(kw) or _stream([_completed()]))
        provider._complete_via_responses([], None, None,
                                         on_chunk=lambda _t: None)
        assert captured["store"] is True

    def test_tool_choice_is_dropped_when_the_turn_sends_no_tools(self):
        """The API rejects ``tool_choice`` without ``tools``."""
        provider = _provider()
        provider._api_params = {"tool_choice": "required"}
        captured = {}
        provider._client.responses.create = (
            lambda **kw: captured.update(kw) or _stream([_completed()]))
        provider._complete_via_responses([], None, None,
                                         on_chunk=lambda _t: None)
        assert "tool_choice" not in captured

    def test_a_named_tool_choice_is_mapped_to_the_wire_id(self):
        """Tool names are hashed, so forcing one by human name is rejected
        upstream as "not found in tools list"."""
        provider = _provider()
        captured = {}
        provider._client.responses.create = (
            lambda **kw: captured.update(kw) or _stream([_completed()]))
        provider._complete_via_responses(
            [], None,
            [ToolSchema(name="readFile", description="d", parameters={})],
            tool_choice={"type": "function", "name": "readFile"},
            on_chunk=lambda _t: None)
        assert captured["tool_choice"]["name"] == name_to_id("readFile")

    def test_a_response_schema_asks_for_json(self):
        provider = _provider()
        captured = {}
        provider._client.responses.create = (
            lambda **kw: captured.update(kw) or _stream([_completed()]))
        provider._complete_via_responses(
            [], None, None, response_schema={"type": "object"},
            on_chunk=lambda _t: None)
        assert captured["text"] == {"format": {"type": "json_object"}}


# ==================== Non-streaming ====================

class TestBatch:
    def test_a_returned_object_needs_no_termination_proof(self):
        provider = _provider()
        provider._client.responses.create = lambda **_kw: {
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "batched"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        result = provider._complete_via_responses([], None, None)
        assert result.get_text() == "batched"
        assert result.finish_reason == FinishReason.STOP


# ==================== complete() integration ====================

class TestCompleteDispatch:
    def test_the_responses_wire_is_used_when_selected(self):
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _completed(output=[{"type": "message", "content": [
                {"type": "output_text", "text": "hi"}]}]),
        ])
        result = provider.complete([], on_chunk=lambda _t: None)
        assert result.text == "hi"
        provider._client.chat.completions.create.assert_not_called()

    def test_structured_output_is_parsed_out_of_the_text(self):
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _completed(output=[{"type": "message", "content": [
                {"type": "output_text", "text": '{"answer": 42}'}]}]),
        ])
        result = provider.complete(
            [], response_schema={"type": "object"}, on_chunk=lambda _t: None)
        assert result.response.structured_output == {"answer": 42}

    def test_prose_where_json_was_asked_for_leaves_the_field_unset(self):
        """The text is more useful than an exception with it discarded."""
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _completed(output=[{"type": "message", "content": [
                {"type": "output_text", "text": "sorry, no"}]}]),
        ])
        result = provider.complete(
            [], response_schema={"type": "object"}, on_chunk=lambda _t: None)
        assert result.response.structured_output is None

    def test_usage_is_recorded_for_the_ledger(self):
        provider = _provider()
        provider._client.responses.create = lambda **_kw: _stream([
            _completed(usage={"input_tokens": 5, "output_tokens": 2,
                              "total_tokens": 7}),
        ])
        provider.complete([], on_chunk=lambda _t: None)
        assert provider.get_token_usage().total_tokens == 7

    def test_an_unconnected_provider_refuses(self):
        provider = _provider()
        provider._model_name = None
        with pytest.raises(RuntimeError, match="not connected"):
            provider.complete([])


# ==================== Accumulator ====================

class TestAccumulator:
    def test_slots_come_back_in_output_index_order(self):
        """Interleaving is the API's, not "text first then calls"."""
        acc = _ResponsesAccumulator()
        acc.open_item(1, {"type": "function_call", "call_id": "c1",
                          "name": name_to_id("readFile"), "arguments": "{}"})
        acc.add_text(0, "before the call")
        parts = acc.parts()
        assert parts[0].text == "before the call"
        assert parts[1].function_call is not None

    def test_the_done_event_replaces_the_assembled_arguments(self):
        """The API's own copy cannot have lost a delta."""
        acc = _ResponsesAccumulator()
        acc.open_item(0, {"type": "function_call", "call_id": "c1",
                          "name": name_to_id("readFile"), "arguments": ""})
        acc.add_arguments(0, '{"path": "trunc')
        acc.set_arguments(0, '{"path": "whole"}')
        assert acc.parts()[0].function_call.args == {"path": "whole"}

    def test_has_text_is_false_until_the_model_writes_some(self):
        acc = _ResponsesAccumulator()
        assert acc.has_text is False
        acc.add_text(0, "a")
        assert acc.has_text is True
