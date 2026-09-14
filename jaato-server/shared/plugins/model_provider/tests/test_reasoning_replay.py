"""The reasoning-replay seam (docs/design/minimax-kimi-mimo-providers.md §3).

Every thinking model in the MiniMax, Kimi and MiMo lineups asks the client
to send the assistant's ``reasoning_content`` back on the next request of a
tool-call loop — MiMo answers 400 without it.  Before the seam the session
dropped thought parts from history and the shared OpenAI converter ignored
them on replay, so no OpenAI-compatible provider could satisfy that rule.

Five touches, each pinned here:

1. the streaming loop and the batch path put the turn's reasoning in a
   leading ``Part.thought`` when the provider opts in;
2. the session keeps that part in history only for an opted-in provider;
3. ``message_to_openai`` replays it as ``reasoning_content`` (or whatever
   the provider's ``_reasoning_replay_fields`` says), with ``content: ""``
   next to ``tool_calls``;
4. GC sizes it;
5. persistence round-trips it.

And the negative space: with ``replay_reasoning`` off (every provider
before this seam) the wire and the history are byte-identical to before.
"""

from unittest.mock import MagicMock

from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
)
from shared.plugins.gc.utils import estimate_message_tokens
from shared.plugins.model_provider._openai_compat.base import OpenAICompatProvider
from shared.plugins.model_provider._openai_compat.converters import (
    deserialize_message,
    history_to_openai,
    message_to_openai,
    serialize_message,
)


# ------------------------------------------------------------------ fixtures

class _Errors:
    class Auth(Exception):
        pass


class _Provider(OpenAICompatProvider):
    """The smallest inheritor: name + error classes, nothing else."""

    _ERR_AUTHENTICATION = _Errors.Auth
    _ERR_RATE_LIMIT = _Errors.Auth
    _ERR_MODEL_NOT_FOUND = _Errors.Auth
    _ERR_CONTEXT_LIMIT = _Errors.Auth
    _ERR_INFRASTRUCTURE = _Errors.Auth

    @property
    def name(self) -> str:
        return "fake"


class _ReplayingProvider(_Provider):
    replay_reasoning = True


def _build(cls):
    p = cls()
    p._client = MagicMock()
    p._model_name = "m"
    p._trace = lambda _m: None
    return p


def _chunk(*, content=None, reasoning=None, tool_call=None, finish=None):
    chunk = MagicMock()
    chunk.usage = None
    choice = MagicMock()
    choice.finish_reason = finish
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning
    delta.audio = None
    if tool_call:
        tc = MagicMock()
        tc.index = 0
        tc.id = tool_call["id"]
        tc.function = MagicMock()
        tc.function.name = tool_call["name"]
        tc.function.arguments = tool_call["arguments"]
        delta.tool_calls = [tc]
    else:
        delta.tool_calls = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _stream(chunks):
    s = MagicMock()
    s.__iter__ = lambda self: iter(chunks)
    return s


def _reasoned_turn(thought="because", text=None, call=True):
    parts = [Part(thought=thought)]
    if text:
        parts.append(Part(text=text))
    if call:
        parts.append(Part(function_call=FunctionCall(
            id="c1", name="readFile", args={"path": "x"})))
    return Message(role=Role.MODEL, parts=parts)


# ------------------------------------------------------- 3. the converter

def test_thought_is_dropped_when_the_wire_does_not_replay():
    (wire,) = message_to_openai(_reasoned_turn())
    assert "reasoning_content" not in wire
    assert wire["content"] is None          # byte-identical to before the seam


def test_thought_is_replayed_as_reasoning_content():
    (wire,) = message_to_openai(
        _reasoned_turn(), reasoning_fields=lambda t: {"reasoning_content": t})
    assert wire["reasoning_content"] == "because"
    assert wire["tool_calls"][0]["id"] == "c1"


def test_a_replayed_tool_turn_sends_empty_content_not_null():
    """XiaomiMiMo/MiMo#44: ``content: null`` next to ``tool_calls`` was the
    failing payload; every vendor example sends the empty string."""
    (wire,) = message_to_openai(
        _reasoned_turn(), reasoning_fields=lambda t: {"reasoning_content": t})
    assert wire["content"] == ""


def test_text_next_to_thought_is_kept_as_content():
    (wire,) = message_to_openai(
        _reasoned_turn(text="done"),
        reasoning_fields=lambda t: {"reasoning_content": t})
    assert wire["content"] == "done"
    assert wire["reasoning_content"] == "because"


def test_a_vendor_shape_is_whatever_the_callable_returns():
    def minimax_shape(t):
        return {"reasoning_content": t,
                "reasoning_details": [{"type": "reasoning.text", "text": t}]}
    (wire,) = message_to_openai(_reasoned_turn(), reasoning_fields=minimax_shape)
    assert wire["reasoning_details"][0]["text"] == "because"


def test_thought_on_a_user_message_is_never_replayed():
    (wire,) = message_to_openai(
        Message(role=Role.USER, parts=[Part(thought="x"), Part(text="hi")]),
        reasoning_fields=lambda t: {"reasoning_content": t})
    assert "reasoning_content" not in wire


def test_history_forwards_the_replay_shape():
    wire = history_to_openai(
        [Message(role=Role.USER, parts=[Part(text="q")]), _reasoned_turn()],
        reasoning_fields=lambda t: {"reasoning_content": t})
    assert wire[1]["reasoning_content"] == "because"


# ---------------------------------------------- 1. the provider's parts

def test_streaming_puts_reasoning_in_a_leading_thought_part_when_opted_in():
    p = _build(_ReplayingProvider)
    p._client.chat.completions.create = lambda **kw: _stream([
        _chunk(reasoning="think "),
        _chunk(reasoning="hard"),
        _chunk(tool_call={"id": "c1", "name": "readFile",
                          "arguments": '{"path": "x"}'}),
        _chunk(finish="tool_calls"),
    ])
    r = p._stream_response(messages=[], kwargs={}, on_chunk=lambda _t: None)
    assert r.thinking == "think hard"                 # what the UI reads
    assert r.parts[0].thought == "think hard"         # what history keeps
    assert r.parts[1].function_call.id == "c1"
    assert r.finish_reason == FinishReason.TOOL_USE


def test_streaming_leaves_parts_alone_when_not_opted_in():
    p = _build(_Provider)
    p._client.chat.completions.create = lambda **kw: _stream([
        _chunk(reasoning="think"), _chunk(content="hi"), _chunk(finish="stop"),
    ])
    r = p._stream_response(messages=[], kwargs={}, on_chunk=lambda _t: None)
    assert r.thinking == "think"
    assert all(part.thought is None for part in r.parts)


def test_streaming_reads_reasoning_through_the_delta_hook():
    """A wire that streams reasoning under another field overrides one
    method; the loop never reads ``reasoning_content`` directly."""
    class _Other(_ReplayingProvider):
        def _reasoning_from_delta(self, delta):
            return getattr(delta, "vendor_reasoning", None)

    p = _build(_Other)
    chunk = _chunk(content="hi", finish="stop")
    chunk.choices[0].delta.vendor_reasoning = "via hook"
    chunk.choices[0].delta.reasoning_content = "ignored"
    p._client.chat.completions.create = lambda **kw: _stream([chunk])
    r = p._stream_response(messages=[], kwargs={}, on_chunk=lambda _t: None)
    assert r.thinking == "via hook"
    assert r.parts[0].thought == "via hook"


def test_batch_path_attaches_the_thought_part_and_replays_history():
    p = _build(_ReplayingProvider)
    sent = {}

    def create(**kw):
        sent.update(kw)
        resp = MagicMock()
        msg = MagicMock()
        msg.content = "answer"
        msg.tool_calls = None
        msg.reasoning_content = "batch thought"
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        resp.choices = [choice]
        resp.usage = None
        return resp

    p._client.chat.completions.create = create
    result = p.complete(
        [Message(role=Role.USER, parts=[Part(text="q")]), _reasoned_turn()])
    # history went out with the replay ...
    assert sent["messages"][1]["reasoning_content"] == "because"
    # ... and the new turn carries its own reasoning as a part.
    assert result.response.parts[0].thought == "batch thought"
    assert result.response.parts[1].text == "answer"


def test_batch_path_replays_nothing_when_not_opted_in():
    p = _build(_Provider)
    sent = {}

    def create(**kw):
        sent.update(kw)
        resp = MagicMock()
        msg = MagicMock()
        msg.content = "answer"
        msg.tool_calls = None
        msg.reasoning_content = None
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        resp.choices = [choice]
        resp.usage = None
        return resp

    p._client.chat.completions.create = create
    p.complete([_reasoned_turn()])
    assert "reasoning_content" not in sent["messages"][0]
    assert sent["messages"][0]["content"] is None


def test_default_replay_shape_is_reasoning_content():
    assert _Provider()._reasoning_replay_fields("t") == {"reasoning_content": "t"}


# -------------------------------------------------- 2. the session gate

def _session_with(provider):
    from shared.jaato_session import JaatoSession
    from shared.session_history import SessionHistory
    s = JaatoSession.__new__(JaatoSession)
    s._history = SessionHistory()
    s._provider = provider
    s._model_name = "m"
    return s


def test_session_keeps_the_thought_part_for_a_replaying_provider():
    s = _session_with(_ReplayingProvider())
    s._add_model_response_to_history(ProviderResponse(parts=[
        Part(thought="t"), Part(text="a")]))
    parts = s._history.messages[0].parts
    assert [p.thought for p in parts] == ["t", None]


def test_session_drops_the_thought_part_otherwise():
    s = _session_with(_Provider())
    s._add_model_response_to_history(ProviderResponse(parts=[
        Part(thought="t"), Part(text="a")]))
    parts = s._history.messages[0].parts
    assert [p.thought for p in parts] == [None]


def test_a_mock_provider_does_not_count_as_opted_in():
    """A test double answers every attribute with a truthy mock; the gate
    must read ``is True`` or every mocked session starts keeping thoughts."""
    s = _session_with(MagicMock())
    s._add_model_response_to_history(ProviderResponse(parts=[
        Part(thought="t"), Part(text="a")]))
    assert all(p.thought is None for p in s._history.messages[0].parts)


def test_a_thought_only_turn_is_not_an_empty_history_entry():
    """Thought alone is not content: the message is still appended (the
    replay needs it) but nothing else about the turn changes."""
    s = _session_with(_ReplayingProvider())
    s._add_model_response_to_history(ProviderResponse(parts=[Part(thought="t")]))
    assert len(s._history.messages) == 1


# ------------------------------------------------------- 4. GC sees it

def test_gc_counts_replayed_reasoning():
    bare = Message(role=Role.MODEL, parts=[Part(text="a" * 400)])
    reasoned = Message(role=Role.MODEL,
                       parts=[Part(thought="r" * 4000), Part(text="a" * 400)])
    assert estimate_message_tokens(reasoned) > estimate_message_tokens(bare) + 500


# --------------------------------------------- 5. persistence round-trip

def test_thought_survives_serialize_deserialize():
    back = deserialize_message(serialize_message(_reasoned_turn()))
    assert back.parts[0].thought == "because"
    assert back.parts[1].function_call.id == "c1"
