"""The facade's media sink, and the guarantees it must not quietly drop.

Two holes these cover, both found by writing a real media client on top
of the convenience layer:

* the facade collected only ``AGENT_OUTPUT`` text, so a session that
  answers OUT LOUD had no route through it at all and the caller fell
  back to the low-level ``subscribe`` plumbing the module exists to
  own;
* ``open_session`` forwarded every connection knob EXCEPT
  ``min_protocol_version``, so adopting the sugar silently dropped the
  caller's compatibility guarantee -- and a daemon too old to send
  media fields looks exactly like a model that chose not to speak.
"""
import inspect

from jaato_sdk.client.convenience import Session, open_session
from jaato_sdk.events import MODEL_MEDIA_CALL_ID, ToolOutputEvent


class _FakeClient:
    """Records subscriptions and lets a test push events through them."""

    def __init__(self):
        self.handlers = {}

    def subscribe(self, event_type, handler):
        self.handlers.setdefault(event_type, []).append(handler)
        return lambda: self.handlers[event_type].remove(handler)

    def emit(self, event_type, ev):
        for h in list(self.handlers.get(event_type, [])):
            h(ev)


def _session():
    return Session(_FakeClient(), "sess-1")


def _speech(seq):
    return ToolOutputEvent(call_id=MODEL_MEDIA_CALL_ID, sequence=seq,
                           mime_type="audio/pcm", data_b64="AA==")


class TestMediaSink:
    def test_model_speech_reaches_the_sink(self):
        s = _session()
        got = []
        unsub = s._subscribe_media(got.append)
        from jaato_sdk.events import EventType
        s._client.emit(EventType.TOOL_OUTPUT, _speech(0))
        s._client.emit(EventType.TOOL_OUTPUT, _speech(1))
        unsub()
        assert [e.sequence for e in got] == [0, 1]

    def test_a_tools_media_does_not(self):
        """A tool's attachment belongs to that tool call, not to the answer."""
        s = _session()
        got = []
        s._subscribe_media(got.append)
        from jaato_sdk.events import EventType
        s._client.emit(EventType.TOOL_OUTPUT, ToolOutputEvent(
            call_id="call_7", mime_type="image/png", data_b64="AA=="))
        assert got == []

    def test_plain_text_output_does_not(self):
        s = _session()
        got = []
        s._subscribe_media(got.append)
        from jaato_sdk.events import EventType
        s._client.emit(EventType.TOOL_OUTPUT,
                       ToolOutputEvent(call_id="call_7", chunk="hello"))
        assert got == []

    def test_no_sink_costs_no_subscription(self):
        """Callers wanting no media must not pay for one.

        Asserted against TOOL_OUTPUT specifically, not against an empty
        handler map: ``Session.__init__`` always subscribes to
        ``PERMISSION_REQUESTED``, so "nothing is subscribed" was never
        the claim being made here.
        """
        from jaato_sdk.events import EventType
        s = _session()
        unsub = s._subscribe_media(None)
        assert EventType.TOOL_OUTPUT not in s._client.handlers
        unsub()  # the no-op must still be callable from the finally block

    def test_all_three_methods_offer_it(self):
        for name in ("ask", "complete", "stream"):
            params = inspect.signature(getattr(Session, name)).parameters
            assert "on_media" in params, f"{name} cannot deliver model speech"


class TestFacadeKeepsTheProtocolGuarantee:
    def test_open_session_accepts_min_protocol_version(self):
        assert "min_protocol_version" in inspect.signature(open_session).parameters

    def test_it_reaches_the_client_constructor(self):
        seen = {}

        class _Ctor:
            def __init__(self, *a, **kw):
                seen.update(kw)

        open_session(_Ctor, min_protocol_version="1.4")
        assert seen.get("min_protocol_version") == "1.4"

    def test_absent_by_default_so_the_ctor_keeps_its_own(self):
        seen = {}

        class _Ctor:
            def __init__(self, *a, **kw):
                seen.update(kw)

        open_session(_Ctor)
        assert "min_protocol_version" not in seen
