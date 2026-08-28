"""A solely-``%name --help`` message must close its turn lifecycle.

Regression for the ``%<prompt> --help`` STALL: a Telegram/WS message whose
entire text is ``%travel-plan-request --help`` produced ZERO further events for
120s, tripping the client's stall detector and ending the session.

Root cause: ``handle_request`` intercepts the help ref (emits a HelpTextEvent),
strips the message to empty, and returns early WITHOUT dispatching a model turn
— and WITHOUT emitting any turn-completion event.  A client that waits for a
per-message completion signal then never sees the turn finish.

Fix: the help-only early-return now emits a synthetic ``TurnCompletedEvent`` to
the requesting client so the turn closes cleanly.  This test drives the real
``handle_request`` SendMessageRequest path with mocked session/server/plugin and
pins: (a) the help is delivered, (b) a TurnCompletedEvent follows, (c) NO model
turn is dispatched.
"""
import threading
from types import SimpleNamespace

from jaato_sdk.events import (
    SendMessageRequest,
    HelpTextEvent,
    TurnCompletedEvent,
)
from jaato_sdk.plugins.base import HelpLines
from server.session_manager import SessionManager


def _sm_with_help_prompt(emitted):
    """A bare SessionManager wired to reach the help-only branch.

    ``emitted`` collects (client_id, event) pairs from _emit_to_client.
    ``server.send_message`` records calls so the test can assert NO model
    turn ran.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._event_callback = lambda cid, ev: emitted.append((cid, ev))
    # Production _emit_to_client attributes the event to the client's
    # session, so a skeleton without this map is not a SessionManager
    # any caller would meet.  Supplied rather than defended against in
    # the method: a getattr fallback there would let a fixture drift
    # further from production and hide the next divergence too.
    sm._client_to_session = {}
    sm._save_session = lambda session: None

    # prompt_library stub: %foo --help resolves to HelpLines.
    prompt_plugin = SimpleNamespace(
        _execute_prompt_command=lambda args: HelpLines(
            lines=[("usage: %foo", ""), ("  --help", "")]
        )
    )
    registry = SimpleNamespace(get_plugin=lambda name: prompt_plugin)
    runtime = SimpleNamespace(registry=registry)
    send_calls = []
    sent = threading.Event()

    def _send_message(text, attachments=None):
        send_calls.append(text)
        sent.set()

    server = SimpleNamespace(
        _runtime=runtime,
        get_all_session_env=lambda: {},
        send_message=_send_message,
        _sent_event=sent,
    )
    session = SimpleNamespace(
        server=server,
        workspace_path="/ws",
        user_inputs=[],
        is_dirty=False,
        last_activity="",
    )
    sm.get_session = lambda sid: session
    return sm, send_calls, sent


def test_help_only_message_emits_turn_completed_and_no_model_turn():
    emitted = []
    sm, send_calls, _sent = _sm_with_help_prompt(emitted)

    sm.handle_request("client-1", "sess-1",
                      SendMessageRequest(text="%foo --help"))

    types = [type(ev).__name__ for _, ev in emitted]
    # (a) help delivered ...
    assert "HelpTextEvent" in types, types
    # (b) ... and the turn is closed so the client doesn't stall.
    assert "TurnCompletedEvent" in types, types
    # ... targeted at the requesting client.
    assert all(cid == "client-1" for cid, _ in emitted)
    # (c) NO model turn dispatched for a pure-help message.
    assert send_calls == []


def test_help_ref_mixed_with_text_still_runs_a_turn():
    """A message that is NOT solely help refs (help ref + real text) must
    still dispatch a model turn — the early-return/turn-complete short-circuit
    is only for the pure-help case."""
    emitted = []
    sm, send_calls, sent = _sm_with_help_prompt(emitted)

    sm.handle_request("client-1", "sess-1",
                      SendMessageRequest(text="%foo --help\nnow do the thing"))

    # The model turn runs in a daemon thread — wait for it deterministically.
    assert sent.wait(timeout=5), "expected a model turn for the residual text"
    # The help still delivered, and a model turn ran.
    assert any(isinstance(ev, HelpTextEvent) for _, ev in emitted)
    assert send_calls
    # No synthetic TurnCompletedEvent on this path (the real turn emits it).
    assert not any(isinstance(ev, TurnCompletedEvent) for _, ev in emitted)
