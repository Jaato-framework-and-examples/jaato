"""``SessionTerminatedEvent`` must be the LAST event a session emits.

It was not.  Quiescence is detected INSIDE ``send_message``, and every turn
driver fires ``on_agent_turn_completed`` AFTER ``send_message`` returns -- so
the terminal notification preceded the final turn event of the very turn it
was reporting the end of.  Measured against a live daemon:

    before   SessionTerminatedEvent @6.92 , TurnCompletedEvent @6.92
    after    TurnCompletedEvent @6.57 , SessionTerminatedEvent @6.57

Out-of-order on its own terms, and a consumer ACTED on it.  The daemon's
cascade policy detaches a cid-stamped session's clients on
``SessionTerminatedEvent`` -- deliberately, to release a slot that once
stayed pinned for 6m43s -- so the ``TurnCompletedEvent`` that arrived
afterwards reached NOBODY:

    before   with-cid: [SessionTerminatedEvent]              <- turn lost
    after    with-cid: [TurnCompletedEvent, SessionTerminatedEvent]

A completion-gated cascade arm therefore came back ``turns=0, tokens=0`` with
its work done and its file on disk -- a silent zero that reads as "the model
did nothing".  Two symptoms, one cause: not a cascade bug, an ordering bug
that the cascade policy was the first to notice.

The split of responsibility is the fix: the SESSION knows whether quiescence
is due, only the DRIVER knows when the turn's own events are finished.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


class _Hooks:
    def __init__(self):
        self.calls = []

    def on_session_quiescent(self, agent_id, reason="natural"):
        self.calls.append(("quiescent", agent_id, reason))


def _session(hooks):
    from shared.jaato_session import JaatoSession

    s = JaatoSession.__new__(JaatoSession)
    s._agent_id = "main"
    s._ui_hooks = hooks
    s._quiescent_due_reason = None
    return s


def test_a_pending_quiescence_is_not_emitted_until_flushed():
    """The whole fix in one assertion: recording is not emitting."""
    hooks = _Hooks()
    s = _session(hooks)
    s._quiescent_due_reason = "natural"

    assert hooks.calls == [], "quiescence emitted before the driver flushed it"

    s.flush_session_quiescent()

    assert hooks.calls == [("quiescent", "main", "natural")]


def test_flushing_twice_emits_once():
    """Drivers may flush after every turn; only a due one fires.

    A terminal event delivered twice is worse than one delivered late -- a
    consumer that tears down on the first would act on a session it has
    already released.
    """
    hooks = _Hooks()
    s = _session(hooks)
    s._quiescent_due_reason = "natural"

    s.flush_session_quiescent()
    s.flush_session_quiescent()

    assert len(hooks.calls) == 1


def test_flushing_with_nothing_due_is_a_no_op():
    hooks = _Hooks()
    s = _session(hooks)

    s.flush_session_quiescent()

    assert hooks.calls == []


def test_a_raising_hook_does_not_break_wind_down():
    """A consumer's handler must not be able to strand a session."""
    class _Boom:
        def on_session_quiescent(self, agent_id, reason="natural"):
            raise RuntimeError("consumer blew up")

    s = _session(_Boom())
    s._quiescent_due_reason = "natural"

    s.flush_session_quiescent()          # must not raise
    assert s._quiescent_due_reason is None, (
        "a raising hook left the notification pending, so the next turn "
        "would re-emit a terminal event for a session already terminated"
    )


# ------------------------------------------------------- the source guard

def test_the_session_never_emits_quiescence_inline():
    """Read from the SOURCE, because re-adding the inline call is how this
    regresses -- and it would regress silently: every runtime test above
    still passes with an extra inline emission, since they drive the flush
    directly rather than a real turn.
    """
    src = pathlib.Path(
        "jaato-server/shared/jaato_session.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    flush = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "flush_session_quiescent"
    )
    flush_lines = set(range(flush.lineno, (flush.end_lineno or flush.lineno) + 1))

    calls = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "on_session_quiescent"
        and n.lineno not in flush_lines
    ]
    assert not calls, (
        f"on_session_quiescent is called outside flush_session_quiescent at "
        f"lines {calls}; an inline emission puts the terminal event back "
        "before the turn's own events"
    )


@pytest.mark.parametrize("path,label", [
    ("jaato-server/server/runner/rpc.py", "daemon path"),
    ("jaato-server/shared/jaato_client.py", "facade path"),
    ("jaato-server/shared/plugins/subagent/plugin.py", "in-process subagents"),
])
def test_every_turn_driver_flushes(path, label):
    """A driver that fires the turn hook must flush.

    Partial coverage is its own defect: the terminal event would arrive last
    on some paths and not others, and a consumer cannot tell which kind of
    session it is holding.  Three drivers, three flushes -- checked per file
    so a new driver added to one layer fails here rather than in production.
    """
    src = pathlib.Path(path).read_text(encoding="utf-8")

    assert "on_agent_turn_completed(" in src, (
        f"{label}: fixture assumption broken — this file no longer fires the "
        "turn hook, so the pairing it is meant to guard has moved"
    )
    assert "flush_session_quiescent()" in src, (
        f"{label}: fires on_agent_turn_completed but never flushes pending "
        "quiescence — SessionTerminatedEvent will precede the turn's own "
        "events on this path"
    )
