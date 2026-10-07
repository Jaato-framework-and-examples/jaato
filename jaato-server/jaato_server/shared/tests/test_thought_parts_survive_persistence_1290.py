"""A reasoning part survives persistence, and a revived session replays it (#1290).

``serialize_part`` recognised four part shapes -- text, function call,
function response, inline data -- and a reasoning part (``text=None``,
``thought='...'``) matched none of them.  It fell through to
``{'type': 'unknown', 'repr': repr(part)}``, and ``deserialize_part`` turned
that back into ``Part(text="[Unrecognized part: Part(...)]")``: an ordinary
TEXT part inside a MODEL message.  So on the providers that keep reasoning in
history on purpose (``replay_reasoning = True``: MiniMax, Kimi, MiMo), a
session revived from disk sent a Python repr of its own reasoning to the
model as assistant content, and sent no ``reasoning_content`` at all.

The same serializer is the runner-RPC history wire (``session.get_history``,
``agent_history_updated``, ``session.set_history``) and the subagent state
file, so the loss was not confined to a cold revive.

These tests pin:

* a thought-only part, and a message mixing thought, text and a function
  call, round-trip through the serializer and through JSON;
* a part carrying text AND reasoning keeps both;
* every ``Part`` field is one the serializer persists -- a field added later
  fails here rather than being dropped the way ``thought`` was;
* an ``unknown`` entry is DROPPED, never turned into text, and dropping never
  leaves an empty MODEL message in history;
* legacy records -- both the ``unknown``/repr shape and the text shape it
  became -- are recovered in MODEL messages only, by PARSING the repr, never
  evaluating it;
* end to end: a session state written to disk and read back drives the real
  MiMo provider into a request whose assistant message carries
  ``reasoning_content`` and contains no ``[Unrecognized part``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from unittest.mock import MagicMock, patch

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
    ToolSchema,
)
from jaato_server.shared.history_invariant import repair_history, validate_history
from jaato_server.shared.plugins.model_provider._openai_compat.converters import (
    history_to_openai,
)
from jaato_server.shared.plugins.session import serializer as ser
from jaato_server.shared.plugins.session.base import SessionState
from jaato_server.shared.tests.reversion import Reversion

_SERIALIZER = "jaato-server/jaato_server/shared/plugins/session/serializer.py"

REASONING = "The store_memory call failed because the tag list was empty."

#: The exact text the pre-#1290 fallback produced for a thought part, as
#: observed live on a revived MiniMax session.
LEGACY_REPR = (
    "Part(text=None, function_call=None, function_response=None, "
    "inline_data=None, thought=" + repr(REASONING) + ", "
    "executable_code=None, code_execution_result=None)"
)
LEGACY_TEXT = "[Unrecognized part: " + LEGACY_REPR + "]"


def _json_round_trip(obj):
    """What a record looks like after it has been written to disk."""
    return json.loads(json.dumps(obj))


def _tool_loop_history() -> list:
    """A tool-call loop mid-flight: reasoning + a call, then its result."""
    return [
        Message.from_text(Role.USER, "read x"),
        Message(role=Role.MODEL, parts=[
            Part(thought=REASONING),
            Part(text="Reading it now."),
            Part(function_call=FunctionCall(
                id="c1", name="readFile", args={"path": "x"})),
        ]),
        Message(role=Role.TOOL, parts=[Part(function_response=ToolResult(
            call_id="c1", name="readFile", result={"content": "hi"}))]),
    ]


def _state(history) -> SessionState:
    from datetime import datetime
    now = datetime(2026, 9, 24, 15, 0, 0)
    return SessionState(session_id="20260924_134528", history=history,
                        created_at=now, updated_at=now)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_a_thought_only_part_round_trips():
    record = _json_round_trip(ser.serialize_part(Part(thought=REASONING)))

    assert record == {"type": "thought", "thought": REASONING}
    back = ser.deserialize_part(record)
    assert back == Part(thought=REASONING)


def test_a_mixed_message_round_trips_losslessly():
    message = Message(role=Role.MODEL, parts=[
        Part(thought=REASONING),
        Part(text="Reading it now."),
        Part(function_call=FunctionCall(id="c1", name="readFile",
                                        args={"path": "x"})),
    ], model="MiniMax-M3", provider="minimax")

    back = ser.deserialize_message(
        _json_round_trip(ser.serialize_message(message)))

    assert back.role == Role.MODEL
    assert back.parts == message.parts
    assert (back.model, back.provider) == ("MiniMax-M3", "minimax")


def test_a_part_carrying_text_and_reasoning_keeps_both():
    part = Part(text="the answer", thought=REASONING)

    record = _json_round_trip(ser.serialize_part(part))

    assert record["type"] == "text"
    assert record["thought"] == REASONING
    assert ser.deserialize_part(record) == part


def test_code_execution_parts_round_trip_too():
    for part in (Part(executable_code="print(1)"),
                 Part(code_execution_result="1\n")):
        record = _json_round_trip(ser.serialize_part(part))
        assert record["type"] != "unknown"
        assert ser.deserialize_part(record) == part


def test_every_part_field_is_one_the_serializer_persists():
    """A field added to ``Part`` must be taught to the serializer.

    ``thought`` was a ``Part`` field for as long as it was being dropped;
    nothing compared the two sets.
    """
    fields = {f.name for f in dataclasses.fields(Part)}
    assert fields == set(ser.PERSISTED_PART_FIELDS)


# ---------------------------------------------------------------------------
# The unknown fallback
# ---------------------------------------------------------------------------


def test_an_unknown_part_is_dropped_not_turned_into_text(caplog):
    with caplog.at_level(logging.WARNING, logger=ser.__name__):
        back = ser.deserialize_part({"type": "unknown",
                                     "repr": "Part(something=1)"})

    assert back is None
    assert "dropping a part recorded as 'unknown'" in caplog.text
    # The content of the repr is not logged, only its size.
    assert "something" not in caplog.text


def test_an_unknown_part_in_a_user_message_is_dropped_not_rendered():
    record = {"role": "user", "parts": [
        {"type": "text", "text": "hello"},
        {"type": "unknown", "repr": LEGACY_REPR},
    ]}

    back = ser.deserialize_message(record)

    assert back.parts == [Part(text="hello")]


def test_serializing_an_empty_part_names_fields_and_writes_no_content(caplog):
    with caplog.at_level(logging.WARNING, logger=ser.__name__):
        record = ser.serialize_part(Part())

    assert record == {"type": "unknown", "fields": []}
    assert "recorded as 'unknown'" in caplog.text


def test_a_message_emptied_by_dropping_is_removed_from_history():
    records = [
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
        {"role": "model", "parts": [{"type": "unknown", "fields": ["x"]}]},
        {"role": "user", "parts": [{"type": "text", "text": "again"}]},
    ]

    history = ser.deserialize_history(records)

    assert [m.role for m in history] == [Role.USER, Role.USER]
    assert all(m.parts for m in history)


def test_a_restored_thought_only_turn_satisfies_the_history_invariant():
    """#674's invariant must not treat a thought part as empty content.

    A thought-only MODEL message is exactly what a restored reasoning turn
    can be, and ``_drop_empty_content`` removing it would strip the
    ``reasoning_content`` MiMo answers 400 without.
    """
    history = ser.deserialize_history(_json_round_trip([
        {"role": "user", "parts": [{"type": "text", "text": "go"}]},
        {"role": "model", "parts": [{"type": "thought", "thought": REASONING}]},
    ]))

    assert validate_history(history) == []
    assert repair_history(history) is history


# ---------------------------------------------------------------------------
# Legacy records
# ---------------------------------------------------------------------------


def test_a_legacy_text_record_is_restored_as_a_thought_part():
    record = {"role": "model", "parts": [
        {"type": "text", "text": LEGACY_TEXT},
        {"type": "text", "text": "Reading it now."},
    ]}

    back = ser.deserialize_message(record)

    assert back.parts == [Part(thought=REASONING), Part(text="Reading it now.")]


def test_a_legacy_unknown_record_is_restored_as_a_thought_part():
    record = {"role": "model", "parts": [{"type": "unknown", "repr": LEGACY_REPR}]}

    assert ser.deserialize_message(record).parts == [Part(thought=REASONING)]


def test_legacy_repair_handles_quotes_and_escapes_in_the_reasoning():
    tricky = "it's \"quoted\"\nand has a newline and a \\ backslash and ñ"
    expr = ("Part(text=None, function_call=None, function_response=None, "
            "inline_data=None, thought=" + repr(tricky) + ", "
            "executable_code=None, code_execution_result=None)")
    record = {"role": "model",
              "parts": [{"type": "text", "text": "[Unrecognized part: " + expr + "]"}]}

    assert ser.deserialize_message(record).parts == [Part(thought=tricky)]


def test_legacy_repair_is_confined_to_model_messages():
    record = {"role": "user", "parts": [{"type": "text", "text": LEGACY_TEXT}]}

    assert ser.deserialize_message(record).parts == [Part(text=LEGACY_TEXT)]


def test_text_that_only_quotes_the_legacy_shape_is_left_alone():
    quoted = "The log said " + LEGACY_TEXT + " which was odd."
    record = {"role": "model", "parts": [{"type": "text", "text": quoted}]}

    assert ser.deserialize_message(record).parts == [Part(text=quoted)]


def test_a_legacy_repr_is_parsed_never_evaluated(tmp_path, caplog):
    marker = tmp_path / "executed"
    hostile = (
        "[Unrecognized part: Part(text=None, thought=__import__('pathlib')"
        f".Path({str(marker)!r}).write_text('x'))]"
    )
    record = {"role": "model", "parts": [
        {"type": "text", "text": hostile},
        {"type": "text", "text": "kept"},
    ]}

    with caplog.at_level(logging.WARNING, logger=ser.__name__):
        back = ser.deserialize_message(record)

    assert not marker.exists()
    assert back.parts == [Part(text="kept")]
    assert "could not be parsed safely" in caplog.text


def test_legacy_repair_refuses_structured_fields_and_unknown_names():
    for expr in (
        "Part(text=None, function_call=FunctionCall(id='a', name='b', args={}),"
        " thought='x')",
        "Part(text='visible', thought='x')",
        "Part(thought='x', signature='y')",
        "Part('x')",
        "NotPart(thought='x')",
        "Part(thought='x', thought='y')",
    ):
        assert ser._part_from_repr(expr) is None, expr


# ---------------------------------------------------------------------------
# End to end: a revived session's request
# ---------------------------------------------------------------------------

_CLIENT_CLASS = ("jaato_server.shared.plugins.model_provider._openai_compat."
                 "base.get_openai_client_class")


def _mimo_provider():
    from jaato_server.shared.plugins.model_provider.base import ProviderConfig
    from jaato_server.shared.plugins.model_provider.mimo.provider import MiMoProvider

    with patch(_CLIENT_CLASS) as mc:
        mc.return_value = MagicMock()
        provider = MiMoProvider()
        provider.initialize(ProviderConfig(api_key="sk-test", extra={}))
    provider._catalog_cache = []
    provider.connect("mimo-v2.5-pro")
    return provider


def _mimo_request_for(history) -> dict:
    """Drive the real MiMo provider and return the request it built."""
    provider = _mimo_provider()
    sent: dict = {}
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = MagicMock(content="done", tool_calls=[],
                               reasoning_content=None)
    response.choices = [choice]
    response.usage = None

    def create(**kwargs):
        sent.update(kwargs)
        return response

    provider._client.chat.completions.create = create
    provider.complete(history, tools=[ToolSchema(
        name="readFile", description="d",
        parameters={"type": "object", "properties": {}})])
    return sent


def test_a_revived_session_replays_its_reasoning_as_reasoning_content():
    on_disk = _json_round_trip(ser.serialize_session_state(
        _state(_tool_loop_history())))
    revived = ser.deserialize_session_state(on_disk)

    sent = _mimo_request_for(revived.history)

    assistant = sent["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == REASONING
    assert "[Unrecognized part" not in json.dumps(sent)
    assert REASONING not in json.dumps(assistant.get("content"))


def test_a_legacy_record_revives_into_reasoning_content_too():
    record = _json_round_trip(ser.serialize_session_state(
        _state(_tool_loop_history())))
    # Rewrite the reasoning part the way a pre-#1290 save stored it.
    record["history"][1]["parts"][0] = {"type": "text", "text": LEGACY_TEXT}

    revived = ser.deserialize_session_state(record)
    sent = _mimo_request_for(revived.history)

    assert sent["messages"][1]["reasoning_content"] == REASONING
    assert "[Unrecognized part" not in json.dumps(sent)


def test_minimax_replay_fields_carry_the_revived_reasoning():
    from jaato_server.shared.plugins.model_provider.minimax.provider import (
        MiniMaxProvider,
    )

    revived = ser.deserialize_history(_json_round_trip(
        ser.serialize_history(_tool_loop_history())))
    wire = history_to_openai(
        revived, reasoning_fields=MiniMaxProvider()._reasoning_replay_fields)

    assistant = wire[1]
    assert assistant["reasoning_content"] == REASONING
    assert assistant["reasoning_details"][0]["text"] == REASONING
    assert "[Unrecognized part" not in json.dumps(wire)


# ---------------------------------------------------------------------------
# The client-facing history shape
# ---------------------------------------------------------------------------


def test_the_history_event_names_a_thought_part_instead_of_its_repr():
    from jaato_server.server.command_router import CommandRouter

    assert CommandRouter._serialize_part(Part(thought=REASONING)) == {
        "type": "thought", "thought": REASONING}
    assert CommandRouter._serialize_part(Part(text="t", thought=REASONING)) == {
        "type": "text", "text": "t", "thought": REASONING}


# ---------------------------------------------------------------------------
# Reversions
# ---------------------------------------------------------------------------

REVERSIONS = [
    Reversion(
        target=_SERIALIZER,
        find=(
            "    for name in _AUXILIARY_FIELDS:\n"
            "        value = getattr(part, name, None)\n"
            "        if value is not None:\n"
            "            return {'type': name, name: value}\n"
        ),
        replace="",
        test="test_a_thought_only_part_round_trips",
        because=(
            "a reasoning part with no primary serializer branch falls to "
            "'unknown' and never reaches a revived session"
        ),
    ),
    Reversion(
        target=_SERIALIZER,
        find=(
            "    for name in _AUXILIARY_FIELDS:\n"
            "        value = getattr(part, name, None)\n"
            "        if value is not None:\n"
            "            return {'type': name, name: value}\n"
        ),
        replace="",
        test="test_a_revived_session_replays_its_reasoning_as_reasoning_content",
        because=(
            "without the thought branch a revived MiMo tool loop carries no "
            "reasoning_content, which that vendor answers 400"
        ),
    ),
    Reversion(
        target=_SERIALIZER,
        find=(
            "is not replayed as text\",\n"
            "            described,\n"
            "        )\n"
            "        return None\n"
        ),
        replace=(
            "is not replayed as text\",\n"
            "            described,\n"
            "        )\n"
            "        return Part(text=f\"[Unrecognized part: "
            "{data.get('repr', '?')}]\")\n"
        ),
        test="test_an_unknown_part_is_dropped_not_turned_into_text",
        because=(
            "an unrestorable part turned back into assistant TEXT, replaying "
            "a Python repr to the model as something it said"
        ),
    ),
    Reversion(
        target=_SERIALIZER,
        find="            data[name] = value\n",
        replace="            pass\n",
        test="test_a_part_carrying_text_and_reasoning_keeps_both",
        because="a part carrying text and reasoning loses the reasoning",
    ),
    Reversion(
        target=_SERIALIZER,
        find="        if role == Role.MODEL and isinstance(raw, dict):\n",
        replace="        if False:\n",
        test="test_a_legacy_text_record_is_restored_as_a_thought_part",
        because=(
            "a record written before the fix keeps replaying the repr text "
            "forever, since every save writes it back"
        ),
    ),
]
