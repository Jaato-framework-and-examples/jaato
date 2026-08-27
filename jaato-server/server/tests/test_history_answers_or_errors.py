"""A history request is answered, or refused — never met with silence.

``_handle_history_request`` guarded on ``if session and session.server:`` and
the method ENDED there.  When the guard failed it emitted nothing at all, so
a client could not distinguish "this session has no history" from "you have
no session" and simply waited out its own timeout.  Absent and empty,
collapsed on the wire.

It is reachable in normal operation, not only on misuse.  The daemon's
cascade policy detaches a cid-stamped session's clients when it terminates,
to release its slot — so a cascade driver asking for the ledger of the arm
that just finished arrives after the detach and finds no session of its own.
That is how a consumer building an eval harness hit it: a pooled arm came
back with an empty ledger, which reads as "the agent made no tool calls"
rather than "nobody answered".

Collapsing those two fabricates verdicts about the model out of the driver's
own blind spot.
"""

from __future__ import annotations

from types import SimpleNamespace

from server.command_router import CommandRouter


class _Sink:
    def __init__(self):
        self.sent = []

    def send_event(self, client_id, event):
        self.sent.append((client_id, event))

    def get_client_user(self, client_id):
        return None


def _router(session):
    r = CommandRouter.__new__(CommandRouter)
    r._event_sink = _Sink()
    r._session_manager = SimpleNamespace(
        get_client_session=lambda cid: session)
    return r


def test_no_attached_session_is_reported_not_swallowed():
    r = _router(None)

    r._handle_history_request("client-1", SimpleNamespace(
        agent_id="main", request_id="req-1"))

    assert r._event_sink.sent, (
        "a history request with no attached session emitted NOTHING; the "
        "caller cannot tell that from a session with no history"
    )
    cid, ev = r._event_sink.sent[-1]
    assert cid == "client-1"
    assert type(ev).__name__ == "ErrorEvent"
    assert ev.error_type == "NoAttachedSession"


def test_the_refusal_correlates_with_the_request():
    """An answer nobody can match to their request is barely an answer.

    The SDK filters waits on ``request_id``; an unstamped event is filed as
    incidental and the caller keeps waiting — the same way a cascade refusal
    went unseen until it was stamped.
    """
    r = _router(None)

    r._handle_history_request("c", SimpleNamespace(
        agent_id="main", request_id="req-xyz"))

    _, ev = r._event_sink.sent[-1]
    assert ev.request_id == "req-xyz"


def test_a_session_without_a_server_is_also_reported():
    """Both halves of the guard, not just the first.

    ``session and session.server`` — a loaded session mid-teardown has the
    first and not the second, and it fell through the same hole.
    """
    r = _router(SimpleNamespace(server=None))

    r._handle_history_request("c", SimpleNamespace(
        agent_id="main", request_id=None))

    assert r._event_sink.sent, "a session without a server emitted nothing"
    assert type(r._event_sink.sent[-1][1]).__name__ == "ErrorEvent"


def test_the_error_is_recoverable():
    """The connection is fine; this is an answer about one request.

    Marking it unrecoverable would tell a client to reconnect over a
    question it can simply ask differently.
    """
    r = _router(None)
    r._handle_history_request("c", SimpleNamespace(
        agent_id="main", request_id=None))

    assert r._event_sink.sent[-1][1].recoverable is True


def test_the_handler_has_no_silent_exit():
    """Checked in the SOURCE: the shape that produced the bug is a guard
    with no ``else``, and it regresses by someone adding another one.
    """
    import ast
    import pathlib

    src = pathlib.Path(
        "jaato-server/server/command_router.py").read_text(encoding="utf-8")
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef)
        and n.name == "_handle_history_request"
    )
    tops = [n for n in fn.body if isinstance(n, ast.If)]
    assert tops, "the handler no longer guards; re-aim this test"
    assert all(n.orelse for n in tops), (
        "a top-level guard in _handle_history_request has no else branch — "
        "when it fails the handler returns without emitting, and the caller "
        "waits out its own timeout with no way to tell why"
    )
