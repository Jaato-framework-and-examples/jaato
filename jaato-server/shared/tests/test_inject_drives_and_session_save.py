"""``inject_prompt_to_session`` must drive an idle target; ``session.save`` exists.

Two surfaces the perpetual-monologue cascade found missing:

- Its watchdog nudged a stalled session with ``inject_prompt(source_type=
  "user")`` after 180s of silence.  It fired twice and produced NOTHING.
  ``inject_prompt`` starts a turn only while ``_on_continuation_needed`` is
  installed — which is for the duration of a ``send_message`` RPC — so an
  idle, undriven session queues it and nothing drains it.  The documented
  cascade-watchdog pattern was a no-op.  ``send_to_sibling`` fixed its own
  copy in #612; this is the shared primitive that reactors, webhook handlers
  and watchdogs all reach through.

- Its evidence for the strongest claim stayed a model's paraphrase because
  the sending session's transcript was never re-saved, and no command exposed
  ``SessionManager.save_session``.  Forcing an unload by attaching away is a
  side effect standing in for an interface — and a silent no-op when the
  client is already attached elsewhere, which is what happened to them.
"""

import inspect
import threading

import pytest

from server.command_router import CommandRouter
from server.session_manager import SessionManager


def _sm(running):
    """A SessionManager whose target answers the OFFER, not a flag.

    ``running`` sets what the target session says when asked -- which is now
    the only thing that decides delivery.  ``_model_running`` is still set on
    the fake server so a test that regressed to reading the replica would be
    reading a value that DISAGREES with the session, and would fail loudly
    rather than accidentally agreeing.
    """
    sm = SessionManager.__new__(SessionManager)
    offered = []

    class _RPC:
        def session_offer_message_threadsafe(self, text, *, source_id=None,
                                             source_type=None,
                                             require_idle=False, timeout=None):
            offered.append((text, source_id, source_type))
            return "queued" if running else "needs_turn"

    s = type("S", (), {})()
    s.session_id = "s-1"
    # The replica is deliberately the OPPOSITE of the truth: any code that
    # regresses to reading it gets the wrong answer, every time, on purpose.
    s.server = type("V", (), {
        "_model_running": not running,
        "_runner_rpc": _RPC(),
        "_terminal_reason": None,
    })()
    sm._sessions = {"s-1": s}
    sm._lock = threading.RLock()
    sm.drove = []
    sm.offered = offered
    sm.send_message_to_session = lambda sid, text: sm.drove.append((sid, text)) or True
    return sm


def test_an_idle_target_is_driven_not_queued():
    """The watchdog's case: nothing was listening, so nothing happened."""
    sm = _sm(running=False)
    assert sm.inject_prompt_to_session("s-1", "are you stuck?") is True
    assert sm.drove == [("s-1", "are you stuck?")]


def test_a_busy_target_is_still_queued_not_preempted():
    """A running turn must not be preempted — that is the tier's whole point."""
    sm = _sm(running=True)
    assert sm.inject_prompt_to_session("s-1", "mid-turn steer") is True
    assert sm.drove == [], "a busy target was preempted"
    assert len(sm.offered) == 1, "the message must reach the session's queue"


def test_the_decision_ignores_the_daemon_side_replica():
    """The step-2 invariant, stated as a test.

    ``_sm`` sets ``_model_running`` to the OPPOSITE of what the session
    answers.  If any of this path regressed to reading the daemon's replica
    it would take the wrong branch here — which is precisely the live failure:
    a peer idle for ~30s still read as busy, so its message was queued behind
    a drain that had already run.
    """
    idle = _sm(running=False)          # replica says busy, session says idle
    assert idle.inject_prompt_to_session("s-1", "wake up") is True
    assert idle.drove == [("s-1", "wake up")], "followed the stale replica"

    busy = _sm(running=True)           # replica says idle, session says busy
    assert busy.inject_prompt_to_session("s-1", "steer") is True
    assert busy.drove == [], "followed the stale replica"


def test_an_unloaded_target_is_still_refused():
    sm = _sm(running=False)
    assert sm.inject_prompt_to_session("ghost", "hello") is False
    assert sm.drove == []


def test_the_shared_primitive_is_what_changed():
    """Fixed here, not at each caller.

    Reactors, webhook handlers and watchdogs all reach sessions through this
    one method — cloning the queue-or-drive decision per caller is what
    produced the original bug (see shared.message_delivery).
    """
    src = inspect.getsource(SessionManager.deliver_prompt_to_session)
    assert "send_message_to_session" in src
    assert "_model_running" in src

    # And the boolean form must stay a pure VIEW of that one decision.  If it
    # ever re-implements the busy check, the two forms can disagree about the
    # same delivery -- which is the cloning this whole module exists to stop.
    bool_src = inspect.getsource(SessionManager.inject_prompt_to_session)
    assert "deliver_prompt_to_session" in bool_src
    assert "_model_running" not in bool_src, (
        "inject_prompt_to_session must delegate, not re-decide"
    )


def test_the_sdk_inject_handler_goes_through_the_decision():
    """The fourth clone -- the one that made no decision at all.

    ``InjectPromptRequest`` used to call the runner's inject directly,
    bypassing the shared primitive, so an SDK/driver inject could not start a
    turn under any circumstances.  Into an idle session that queued into a
    queue with no drainer, permanently: the session became unreachable, and a
    watchdog's nudge landed in the same dead queue as the message it was sent
    to rescue.
    """
    src = inspect.getsource(SessionManager.handle_request)
    marker = "elif isinstance(event, InjectPromptRequest):"
    assert marker in src
    handler = src.split(marker, 1)[1].split("elif isinstance(", 1)[0]

    # CODE ONLY.  The handler's comment explains what it no longer does and
    # names the old call, so a raw substring check reads the explanation as
    # the behaviour -- an over-broad guard that fails on its own docs.
    code = "\n".join(
        line for line in handler.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert "deliver_prompt_to_session" in code
    assert "session_inject_prompt_threadsafe" not in code, (
        "the handler must not reach past the queue-or-drive decision"
    )


# ----------------------------------------------------------------------
# session.save
# ----------------------------------------------------------------------

def test_session_save_is_routed():
    assert '"session.save"' in inspect.getsource(CommandRouter._dispatch)
    assert hasattr(CommandRouter, "_handle_session_save")


def test_save_defaults_to_the_attached_session():
    src = inspect.getsource(CommandRouter._handle_session_save)
    assert "or session_id" in src, "must default to the caller's session"


def test_not_loaded_is_reported_as_a_distinct_fact():
    """``save_session`` returns False for NOT LOADED, not for a write failure.

    An unloaded session is already on disk, so telling the caller "save
    failed" would send them looking for a problem that does not exist.
    """
    src = inspect.getsource(CommandRouter._handle_session_save)
    assert "not loaded" in src.lower()
    assert "already persisted" in src.lower()


def test_save_session_still_reports_missing_sessions():
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    assert sm.save_session("ghost") is False
