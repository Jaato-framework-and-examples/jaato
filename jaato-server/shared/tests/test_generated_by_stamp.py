"""Model-generated media carries a machine-readable provenance stamp (Art. 50(2)).

Regulation (EU) 2024/1689 Art. 50(2): the output of an AI system that
generates audio, image, video or text must be "marked in a machine-readable
format and detectable as artificially generated".  In force since 2 August
2026.  The framework's boundary is the event protocol, and the event
protocol is where a client learns what it is about to show a person -- so
the stamp lives on ``ToolOutputEvent.generated_by`` (protocol 1.14) and is
put there at the one place that knows both that the bytes are the model's
and which binding produced them: ``JaatoSession._deliver_model_media``.
``docs/design/eu-ai-act.md`` §4.3.

Pinned here:

A. every chunk of the model's own media is stamped ``{"kind": "ai", ...}``
   naming the active provider and model;
B. a chunk a tool merely relayed carries NOTHING -- a fetched image is not
   AI-generated because an agent fetched it -- while a tool-result
   attachment carries exactly what its producer claimed;
C. the stamp crosses the runner->daemon wire and lands on the event, and
   is absent from a text chunk's frame, which must stay byte-identical.
"""

from __future__ import annotations

from jaato_sdk.events import (
    GENERATED_BY_AI, MODEL_MEDIA_CALL_ID, PROTOCOL_VERSION, ToolOutputEvent,
    ai_generated_by,
)
from jaato_sdk.plugins.model_provider.types import Attachment, MediaDelta
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_SESSION = "jaato-server/shared/jaato_session.py"

REVERSIONS = [
    Reversion(
        target=_SESSION,
        find="                final=delta.final,\n                generated_by=self._model_provenance(),",
        replace="                final=delta.final,",
        because=(
            "the model's own media must carry the Art. 50(2) marking; an "
            "unstamped chunk is output nothing can detect as AI-generated"
        ),
        test="test_the_models_own_media_is_stamped_with_its_binding",
    ),
]


class _Hooks:
    def __init__(self):
        self.calls = []

    def on_tool_output(self, **kw):
        self.calls.append(kw)


def _session(hooks, provider="openrouter", model="openai/gpt-audio"):
    from shared.jaato_session import JaatoSession

    s = JaatoSession.__new__(JaatoSession)
    s._agent_id = "main"
    s._ui_hooks = hooks
    s._trace = lambda *a, **k: None
    s._active_provider_name = provider
    s._model_name = model
    s._daemon_session_id = "20260918_101500"
    return s


# ----------------------------------------------------------------- A. model

def test_the_models_own_media_is_stamped_with_its_binding():
    hooks = _Hooks()
    _session(hooks)._deliver_model_media(
        MediaDelta(mime_type="audio/pcm", data=b"\x01", sequence=0, final=True))
    [call] = hooks.calls
    assert call["call_id"] == MODEL_MEDIA_CALL_ID
    assert call["generated_by"] == {
        "kind": GENERATED_BY_AI, "provider": "openrouter",
        "model": "openai/gpt-audio", "session_id": "20260918_101500",
        "agent_id": "main",
    }


def test_the_stamp_omits_what_it_does_not_know_rather_than_sending_null():
    assert ai_generated_by(None, None) == {"kind": "ai"}
    hooks = _Hooks()
    s = _session(hooks, provider=None, model=None)
    del s._daemon_session_id
    s._deliver_model_media(MediaDelta(mime_type="audio/pcm", data=b"\x01"))
    assert hooks.calls[0]["generated_by"] == {"kind": "ai", "agent_id": "main"}


def test_a_bare_session_still_delivers():
    """A stamp reader that raises turns a delivered chunk into a traced
    failure -- the media must reach the client even when the session
    cannot say who produced it."""
    from shared.jaato_session import JaatoSession

    s = JaatoSession.__new__(JaatoSession)
    s._agent_id = "main"
    hooks = _Hooks()
    s._ui_hooks = hooks
    s._trace = lambda *a, **k: None
    s._deliver_model_media(MediaDelta(mime_type="audio/pcm", data=b"\x01"))
    assert len(hooks.calls) == 1


# ------------------------------------------------------------ B. attachment

class _Result:
    call_id = "call_7"
    name = "generate_image"


def test_a_relayed_attachment_claims_nothing_and_a_produced_one_its_producers_claim():
    hooks = _Hooks()
    s = _session(hooks)
    stamped = ai_generated_by("openrouter", "some/image-model")
    s._emit_withheld_attachments_to_clients(_Result(), [
        Attachment(mime_type="image/png", data=b"\x89PNG"),
        Attachment(mime_type="image/png", data=b"\x89PNG", generated_by=stamped),
    ])
    assert [c["generated_by"] for c in hooks.calls] == [None, stamped]


def test_attachment_default_is_no_claim():
    assert Attachment(mime_type="image/png", data=b"x").generated_by is None


# ------------------------------------------------------------------ C. wire

def test_the_runner_frame_carries_it_only_beside_bytes():
    """A text chunk's frame is the hottest notification on the runner
    boundary and must stay byte-identical; the stamp rides only with bytes."""
    from server.runner.rpc import _AgentUIHooksNotificationShim

    class _Rpc:
        _NOTIF_TOOL_OUTPUT = "tool_output"
        frames = []

        def emit_notification(self, request_id, event_type, payload):
            self.frames.append(payload)

    rpc = _Rpc()
    shim = _AgentUIHooksNotificationShim(rpc, request_id=1)
    stamp = ai_generated_by("p", "m")
    shim.on_tool_output(agent_id="main", call_id="c", chunk="hi", generated_by=stamp)
    shim.on_tool_output(agent_id="main", call_id=MODEL_MEDIA_CALL_ID, chunk="",
                        mime_type="audio/pcm", data_b64="AQ==", final=True,
                        generated_by=stamp)
    text, media = rpc.frames
    assert text == {"agent_id": "main", "call_id": "c", "chunk": "hi"}
    assert media["generated_by"] == stamp


def test_the_event_declares_the_field_and_the_version_says_so():
    ev = ToolOutputEvent(agent_id="main", call_id=MODEL_MEDIA_CALL_ID,
                         mime_type="audio/pcm", data_b64="AQ==",
                         generated_by=ai_generated_by("p", "m"))
    assert ev.generated_by == {"kind": "ai", "provider": "p", "model": "m"}
    assert ToolOutputEvent(agent_id="a", call_id="c", chunk="x").generated_by is None
    major, minor = PROTOCOL_VERSION.split(".")
    assert (int(major), int(minor)) >= (1, 14)


def test_the_daemon_dispatcher_forwards_it():
    from server.core import _dispatch_tool_output

    hooks = _Hooks()
    stamp = ai_generated_by("p", "m")
    _dispatch_tool_output(hooks, {
        "agent_id": "main", "call_id": MODEL_MEDIA_CALL_ID, "chunk": "",
        "mime_type": "audio/pcm", "data_b64": "AQ==", "final": True,
        "generated_by": stamp,
    }, "main")
    assert hooks.calls[0]["generated_by"] == stamp
    # A text frame takes the text path and carries no media keyword at all.
    _dispatch_tool_output(hooks, {"agent_id": "main", "call_id": "c", "chunk": "hi"}, "main")
    assert "generated_by" not in hooks.calls[1]
