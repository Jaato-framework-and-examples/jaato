"""``session.message`` on the command router (protocol 1.23).

The client-tier form of the ``courier`` plugin's ``send_to_session``.
Pinned here:

- the SENDER is the caller's own session, resolved from ``client_id`` --
  never from the payload, so a client can only speak as the session it is
  attached to;
- the receipt travels as ONE typed ``SessionMessageResultEvent`` with the
  caller's ``request_id`` echoed, whatever the outcome, so a driver
  branches on ``status`` rather than parsing a ``SystemMessageEvent``;
- a caller with no session gets a ``SessionError`` (the reply that settles
  an ``ask()``, #1007) and the manager is never reached;
- a request with neither text nor attachments is refused by name before
  the manager is reached;
- ``attachments`` and ``request_id`` are payload-only, as on ``session.wake``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from jaato_sdk.events import ErrorEvent, SessionMessageResultEvent
from jaato_server.server.command_router import CommandRouter


def _router(session_id="s-a"):
    r = CommandRouter.__new__(CommandRouter)
    r._session_manager = MagicMock()
    session = MagicMock() if session_id else None
    if session is not None:
        session.session_id = session_id
    r._session_manager.get_client_session.return_value = session
    r._event_sink = MagicMock()
    return r


def _only_answer(router):
    sent = [c[0][1] for c in router._event_sink.send_event.call_args_list]
    assert len(sent) == 1, sent
    return sent[0]


def test_the_sender_is_the_callers_session_and_the_receipt_is_typed():
    r = _router("s-a")
    r._session_manager.deliver_group_message.return_value = {
        "status": "accepted", "message_id": "m1", "target_session_id": "s-b",
        "sibling_name": "beta", "group_key": "cid:c1", "woken": True,
        "headless": True, "bytes": 5, "attachments": 0,
    }
    r._handle_session_message(
        "c1", [], {"target": "beta", "text": "hello", "request_id": "req-9",
                   "event_id": "e1"})

    r._session_manager.deliver_group_message.assert_called_once_with(
        "s-a", "beta", "hello", attachments=[], file_refs=[],
        text_attachments=[], event_id="e1")
    evt = _only_answer(r)
    assert isinstance(evt, SessionMessageResultEvent)
    assert evt.request_id == "req-9"
    assert (evt.status, evt.ok, evt.target) == ("accepted", True, "beta")
    assert evt.message_id == "m1" and evt.target_session_id == "s-b"
    assert evt.group_key == "cid:c1" and evt.woken and evt.headless


def test_positional_args_join_the_message():
    r = _router("s-a")
    r._session_manager.deliver_group_message.return_value = {"status": "queued"}
    r._handle_session_message("c1", ["s-b", "the", "file", "is", "free"], None)
    r._session_manager.deliver_group_message.assert_called_once_with(
        "s-a", "s-b", "the file is free", attachments=[], file_refs=[],
        text_attachments=[], event_id=None)
    assert _only_answer(r).ok is True


def test_a_refusal_is_a_typed_result_with_ok_false():
    r = _router("s-a")
    r._session_manager.deliver_group_message.return_value = {
        "status": "ambiguous", "candidates": ["s-b", "s-c"],
        "error": "names 2 sessions"}
    r._handle_session_message("c1", [], {"target": "worker", "text": "hi"})
    evt = _only_answer(r)
    assert isinstance(evt, SessionMessageResultEvent)
    assert evt.ok is False and evt.status == "ambiguous"
    assert evt.candidates == ["s-b", "s-c"] and "2 sessions" in evt.error


def test_a_duplicate_is_a_benign_success():
    r = _router("s-a")
    r._session_manager.deliver_group_message.return_value = {
        "status": "duplicate", "detail": "already actioned"}
    r._handle_session_message("c1", [], {"target": "s-b", "text": "hi"})
    evt = _only_answer(r)
    assert evt.ok is True and evt.status == "duplicate"


def test_no_session_is_a_session_error_and_never_reaches_the_manager():
    r = _router(session_id=None)
    r._handle_session_message("c1", [], {"target": "s-b", "text": "hi",
                                         "request_id": "req-1"})
    evt = _only_answer(r)
    assert isinstance(evt, ErrorEvent)
    assert evt.error_type == "SessionError" and evt.request_id == "req-1"
    r._session_manager.deliver_group_message.assert_not_called()


def test_no_content_is_refused_by_name_before_the_manager():
    r = _router("s-a")
    r._handle_session_message("c1", [], {"target": "s-b", "text": ""})
    evt = _only_answer(r)
    assert evt.status == "refused" and evt.ok is False
    r._session_manager.deliver_group_message.assert_not_called()


def test_attachments_alone_are_content_and_travel_as_dicts_only():
    r = _router("s-a")
    r._session_manager.deliver_group_message.return_value = {"status": "accepted"}
    att = {"mime_type": "audio/wav", "data": "AAAA", "display_name": "n.wav"}
    r._handle_session_message(
        "c1", [], {"target": "s-b", "text": "", "attachments": [att, "junk"]})
    r._session_manager.deliver_group_message.assert_called_once_with(
        "s-a", "s-b", "", attachments=[att], file_refs=[],
        text_attachments=[], event_id=None)


def test_the_dispatcher_routes_the_verb():
    """The verb must be wired, not only implemented."""
    from jaato_sdk.events import CommandRequest
    r = _router("s-a")
    r._session_manager.deliver_group_message.return_value = {"status": "accepted"}
    r._session_manager.get_session.return_value = None
    called = {}
    r._handle_session_message = lambda cid, args, payload: called.update(
        cid=cid, args=args, payload=payload)
    r.handle_request("c1", "s-a", CommandRequest(
        command="session.message", args=["s-b", "hi"], payload=None))
    assert called == {"cid": "c1", "args": ["s-b", "hi"], "payload": None}
