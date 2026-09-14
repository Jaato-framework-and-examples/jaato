"""Bytes on the daemon ↔ runner RPC wire (#920).

The reported failure: a voice session died mid-turn with

    RunnerRPCClient: peer sent oversized frame:
        Message too large: 16745978 bytes (cap 10485760)

for a **3.83 MB** attachment.  Two independent defects produced it, and
this module pins both plus the guard that keeps either from being fatal
again:

1. ``rpc._write`` encoded with ``json.dumps(payload, default=str)``, so
   ``bytes`` were serialised as their Python repr — 4.2x the payload,
   and a string the receiver could never decode back to bytes.  Fixed
   by :mod:`server.runner.json_codec` (base64 under a marker key).

2. the ``agent_history_updated`` notification handed raw ``Message``
   objects to that encoder, which stringified each whole message —
   audio bytes and all.  Fixed by serialising the snapshot with the
   canonical session serializer, the same shape ``session.get_history``
   already used.

3. an oversized frame is refused at the WRITE side now.  The reader
   consumes the length prefix but not the body, so it cannot skip such
   a frame and closes the channel — one bad frame took down every
   in-flight call and ended the session.

Lives in ``server/tests`` rather than beside the dispatcher in
``server/runner/tests`` because only the former is a CI suite leg
(``.github/workflows/ci-tests.yml``); a regression guard CI never runs
is a comment.
"""

from __future__ import annotations

import io
import json
import math
import socket
import struct
import threading
import wave
from typing import Any, Dict, List, Tuple

import pytest

from shared.framing import MAX_MESSAGE_SIZE, read_frame_sync

from server.core import _deserialize_wire_history
from server.runner.envelope import KIND_RESPONSE, ResponseEnvelope
from server.runner.json_codec import (
    BYTES_MARKER_KEY,
    dumps,
    frame_size,
    loads,
)
from server.runner.rpc import (
    RunnerRPC,
    _AgentUIHooksNotificationShim,
    _serialize_history_for_wire,
)

from jaato_sdk.plugins.model_provider.types import Message, Part, Role


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _wav(seconds: int = 2, rate: int = 16000) -> bytes:
    """A real WAV blob — the issue's repro, shortened by default.

    High-byte samples matter: those are exactly the bytes ``repr()``
    expands to ``\\xNN``, so a sine wave is a fair (slightly
    conservative) stand-in for recorded speech.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(6000 * math.sin(i / 25.0)))
            for i in range(rate * seconds)
        ))
    return buf.getvalue()


def _audio_message(payload: bytes) -> Message:
    return Message(
        role=Role.USER,
        parts=[Part(inline_data={"mime_type": "audio/wav", "data": payload})],
    )


class _FakeRPC:
    """Records ``emit_notification`` calls (mirrors the shim tests)."""

    _NOTIF_AGENT_HISTORY_UPDATED = "agent_history_updated"

    def __init__(self) -> None:
        self.emissions: List[Tuple[int, str, Dict[str, Any]]] = []

    def emit_notification(self, request_id, event_type, payload=None):
        self.emissions.append((request_id, event_type, dict(payload or {})))


# ----------------------------------------------------------------------
# 1. The codec
# ----------------------------------------------------------------------


class TestJSONCodec:
    def test_bytes_round_trip(self):
        """The whole point: what goes in as bytes comes out as bytes."""
        payload = {"data": bytes(range(256))}
        assert loads(dumps(payload)) == payload

    def test_bytes_survive_nesting(self):
        payload = {"a": [{"b": (b"\x00\xff",)}]}
        # tuples come back as lists; the bytes are what matter
        assert loads(dumps(payload))["a"][0]["b"] == [b"\x00\xff"]

    def test_bytearray_and_memoryview_normalise_to_bytes(self):
        assert loads(dumps({"d": bytearray(b"ab")}))["d"] == b"ab"
        assert loads(dumps({"d": memoryview(b"ab")}))["d"] == b"ab"

    def test_repr_encoding_does_not_round_trip(self):
        """Why ``default=str`` was wrong even below the frame cap.

        Small payloads passed the cap and delivered a Python repr
        *string* that nothing would decode — the size failure was the
        loud half of the bug, this is the quiet one.
        """
        legacy = json.loads(json.dumps({"d": b"\x00\xff"}, default=str))["d"]
        assert legacy == "b'\\x00\\xff'"
        assert loads(dumps({"d": b"\x00\xff"}))["d"] == b"\x00\xff"

    def test_base64_is_far_smaller_than_repr(self):
        raw = _wav(seconds=2)
        legacy = len(json.dumps({"d": raw}, default=str))
        fixed = len(dumps({"d": raw}))
        assert legacy / len(raw) > 4.0     # the reported 4.2x
        assert fixed / len(raw) < 1.4      # base64 + JSON quoting

    def test_non_bytes_objects_still_stringify(self):
        """``default=str`` stays the fallback for diagnostic values."""
        class Exotic:
            def __repr__(self) -> str:
                return "<exotic>"

        assert loads(dumps({"o": Exotic()}))["o"] == "<exotic>"

    def test_ordinary_dicts_pass_through_untouched(self):
        payload = {"kind": "response", "id": 7, "result": {"a": [1, 2]}}
        assert loads(dumps(payload)) == payload

    def test_marker_with_invalid_base64_is_left_alone(self):
        """A malformed marker must not take down the read loop."""
        raw = json.dumps({"d": {BYTES_MARKER_KEY: "not base64!!"}})
        assert loads(raw)["d"] == {BYTES_MARKER_KEY: "not base64!!"}

    def test_frame_size_is_the_wire_size(self):
        encoded = dumps({"t": "héllo", "d": b"\x00\xff"})
        assert frame_size(encoded) == len(encoded.encode("utf-8"))

    def test_decodes_frames_written_before_the_fix(self):
        """Rolling upgrade: a peer that predates the codec still parses."""
        assert loads(json.dumps({"kind": "response", "id": 1})) == {
            "kind": "response", "id": 1,
        }


# ----------------------------------------------------------------------
# 2. The history notification
# ----------------------------------------------------------------------


class TestHistorySerialization:
    def test_message_history_is_serialised_not_stringified(self):
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        shim.on_agent_history_updated(
            agent_id="main", history=[_audio_message(b"\x00\xff" * 8)],
        )
        _, event_type, payload = rpc.emissions[0]
        assert event_type == "agent_history_updated"
        part = payload["history"][0]["parts"][0]
        assert part["type"] == "inline_data"
        assert part["mime_type"] == "audio/wav"
        assert isinstance(part["data"], str)      # base64, not a repr

    def test_dict_history_still_passes_through(self):
        """In-process callers already hand dicts; they must not change."""
        rpc = _FakeRPC()
        shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
        history = [{"role": "user", "content": "hello"}]
        shim.on_agent_history_updated(agent_id="main", history=history)
        assert rpc.emissions[0][2]["history"] == history

    def test_one_bad_message_does_not_drop_the_history(self):
        class Exploding:
            @property
            def to_dict(self):
                raise RuntimeError("boom")

        out = _serialize_history_for_wire([Exploding(), {"role": "user"}])
        assert len(out) == 2
        assert out[0] == {"role": "system", "content": "<unserialisable>"}

    def test_the_reported_utterance_now_fits_in_a_frame(self):
        """The regression itself, at the reported duration.

        120 s of 16 kHz mono audio is 3.84 MB and is exactly what
        ``MAX_UTTERANCE_SECONDS`` produces — a legal press of the
        push-to-talk key.  As a repr it was a 16.13 MB frame the peer
        refused; serialised it is 5.12 MB, half the cap.
        """
        raw = _wav(seconds=120)
        history = [_audio_message(raw)]
        legacy = len(json.dumps(
            {"kind": "notification", "payload": {"history": history}},
            default=str,
        ))
        fixed = frame_size(dumps({
            "kind": "notification",
            "payload": {"history": _serialize_history_for_wire(history)},
        }))
        assert legacy > MAX_MESSAGE_SIZE
        assert fixed < MAX_MESSAGE_SIZE

    def test_daemon_side_restores_messages_with_their_bytes(self):
        """The other half: ``AgentState.history`` holds Messages again."""
        raw = _wav(seconds=1)
        wire = loads(dumps({
            "history": _serialize_history_for_wire([_audio_message(raw)]),
        }))
        restored = _deserialize_wire_history(wire["history"])
        assert isinstance(restored[0], Message)
        assert restored[0].parts[0].inline_data["data"] == raw

    def test_daemon_side_keeps_a_non_dict_payload_as_is(self):
        msg = _audio_message(b"\x00")
        assert _deserialize_wire_history([msg]) == [msg]
        assert _deserialize_wire_history(None) == []


# ----------------------------------------------------------------------
# 3. The oversize guard
# ----------------------------------------------------------------------


@pytest.fixture
def rpc_pair():
    """``(daemon_sock, rpc)`` with the dispatcher serving on the other end."""
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _exec(name: str, args: Dict[str, Any]):
        return False, {"error": f"no executor for {name}"}

    rpc = RunnerRPC(runner_sock, _exec)
    thread = threading.Thread(target=rpc.serve, name="rpc-oversize", daemon=True)
    thread.start()
    try:
        yield daemon_sock, rpc
    finally:
        rpc.shutdown()
        try:
            daemon_sock.close()
        except OSError:
            pass
        thread.join(timeout=2)


class TestOversizedFrameGuard:
    def test_oversized_frame_is_not_written(self, rpc_pair):
        _, rpc = rpc_pair
        assert rpc._write({"kind": "stream", "id": 1, "text": "x"}) is True
        huge = {"kind": "stream", "id": 1, "text": "x" * (MAX_MESSAGE_SIZE + 1)}
        assert rpc._write(huge) is False

    def test_oversized_response_becomes_a_typed_error(self, rpc_pair):
        """The call fails; the channel — and the session — survive.

        Pre-fix the frame went out, the daemon read a length prefix it
        could not follow, and closed the transport: every in-flight
        call died with "runner RPC closed before id=N responded".
        """
        daemon_sock, rpc = rpc_pair
        rpc._emit_response(
            request_id=42, ok=True,
            result={"blob": "x" * (MAX_MESSAGE_SIZE + 1)},
        )
        daemon_sock.settimeout(3.0)
        payload = loads(read_frame_sync(daemon_sock))
        assert payload["kind"] == KIND_RESPONSE
        env = ResponseEnvelope.from_dict(payload)
        assert env.id == 42
        assert env.ok is False
        assert env.error is not None
        assert env.error.type == "FrameTooLargeError"

    def test_outgoing_call_fails_fast_instead_of_hanging(self, rpc_pair):
        """A request that never left must not be waited on.

        ``outgoing_call`` blocks a runner worker thread on a future the
        daemon can only complete if it received the request.
        """
        _, rpc = rpc_pair
        with pytest.raises(RuntimeError, match="was not sent"):
            rpc.outgoing_call(
                "client.prompt_operator",
                {"blob": "x" * (MAX_MESSAGE_SIZE + 1)},
                timeout=2.0,
            )
        assert not rpc._outgoing_calls

    def test_bytes_reach_the_daemon_decoder_as_bytes(self, rpc_pair):
        """End to end over a real socket, runner → daemon."""
        daemon_sock, rpc = rpc_pair
        raw = _wav(seconds=1)
        assert rpc._write({"kind": "stream", "id": 9, "blob": raw}) is True
        daemon_sock.settimeout(3.0)
        assert loads(read_frame_sync(daemon_sock))["blob"] == raw


# ----------------------------------------------------------------------
# 4. The daemon → runner direction
# ----------------------------------------------------------------------


class TestDaemonSideWrites:
    """The other half of the channel.

    Daemon→runner writes used a bare ``json.dumps``, which raises
    ``TypeError`` on bytes rather than mangling them — a different
    failure from the runner's, and one the shared codec removes.  The
    oversize guard is symmetric for the same reason it exists runner-
    side: the runner cannot skip an oversized frame either, and its
    read loop closes the channel on one.
    """

    async def _client(self) -> Tuple[socket.socket, Any]:
        from server.runner_rpc_client import RunnerRPCClient
        daemon_sock, runner_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM,
        )
        client = RunnerRPCClient(daemon_sock, runner_pid=0)
        await client.start()
        return runner_sock, client

    async def _shutdown(self, client) -> None:
        client._closed = True
        if client._writer is not None:
            client._writer.close()
        if client._read_task is not None:
            client._read_task.cancel()

    @pytest.mark.asyncio
    async def test_bytes_cross_as_base64_not_a_type_error(self):
        runner_sock, client = await self._client()
        try:
            await client._write_frame_json({"kind": "request", "d": b"\x00\xff"})
            runner_sock.settimeout(3.0)
            assert loads(read_frame_sync(runner_sock))["d"] == b"\x00\xff"
        finally:
            await self._shutdown(client)
            runner_sock.close()

    @pytest.mark.asyncio
    async def test_oversized_frame_raises_instead_of_being_written(self):
        from shared.framing import FrameTooLargeError
        runner_sock, client = await self._client()
        try:
            with pytest.raises(FrameTooLargeError):
                await client._write_frame_json({
                    "kind": "request",
                    "d": "x" * (MAX_MESSAGE_SIZE + 1),
                })
        finally:
            await self._shutdown(client)
            runner_sock.close()

    @pytest.mark.asyncio
    async def test_a_refused_request_leaves_no_in_flight_entry(self):
        """The caller gets the error; nothing is left waiting on a reply."""
        from shared.framing import FrameTooLargeError
        runner_sock, client = await self._client()
        try:
            with pytest.raises(FrameTooLargeError):
                await client.call(
                    "session.send_message",
                    {"prompt": "x" * (MAX_MESSAGE_SIZE + 1)},
                )
            assert not client._in_flight
            assert not client._stream_cbs
            assert not client._notification_cbs
        finally:
            await self._shutdown(client)
            runner_sock.close()
