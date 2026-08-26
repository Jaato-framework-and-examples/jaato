"""``unreachable`` was five outcomes wearing one word.

``deliver_prompt_to_session`` returned it from FIVE places -- no server
attached, no runner channel, a runner too old to expose the offer verb, the
offer raising, and the DRIVE failing after the target answered ``needs_turn``
-- and the single prose reason attached to the word described only the fourth:
*"on a timeout the message may still have been enqueued and only the
acknowledgement lost."*

For the other four **nothing was ever offered**, so that sentence was not
vague, it was FALSE.  It warned the sender about a duplicate that could not
exist, and a careful sender told its retry might duplicate declines to
re-send -- abandoning a message that definitely never arrived.  The word
meant to prevent a wrong action caused one.

The split is on the ONE axis a sender can act on: *was anything put in
flight?*  The four mechanisms underneath ``unreachable`` deliberately did not
become words -- a sender cannot act on "no runner channel" differently from
"no server attached" -- so they are distinguished in the LOG, where the
operator who can act on them reads.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import threading

import pytest

from server.session_manager import SessionManager, _delivery_failure_reason
from shared.message_delivery import (
    ACCEPTED, DELIVERED, NOT_CONFIRMED, UNREACHABLE,
)


# --------------------------------------------------------------------------
# fixtures: one per producer
# --------------------------------------------------------------------------

def _manager(server) -> SessionManager:
    sm = SessionManager.__new__(SessionManager)
    sess = type("S", (), {})()
    sess.session_id = "s-1"
    sess.server = server
    sm._sessions = {"s-1": sess}
    sm._lock = threading.RLock()
    return sm


def _server(rpc, terminal_reason=None):
    return type("V", (), {
        "_runner_rpc": rpc,
        "_terminal_reason": terminal_reason,
        "_model_running": True,
    })()


class _OfferRaises:
    """Producer 4 -- the offer WAS made and the answer was lost."""

    def session_offer_message_threadsafe(self, text, **kw):
        raise TimeoutError()          # str() == ""


class _NoOfferVerb:
    """Producer 3 -- a runner predating the atomic offer verb (#620)."""


class _NeedsTurn:
    """Producer 5's precondition -- the target is idle, so DRIVE it."""

    def session_offer_message_threadsafe(self, text, **kw):
        return "needs_turn"


@pytest.fixture(autouse=True)
def _capture(request):
    """Every test here reads the log, because that is where mechanism lives."""
    records: list[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("server.session_manager")
    handler = _Cap()
    logger.addHandler(handler)
    prev = logger.level
    logger.setLevel(logging.WARNING)
    request.node._records = records
    yield records
    logger.removeHandler(handler)
    logger.setLevel(prev)


def _messages(records) -> str:
    return "\n".join(r.getMessage() for r in records)


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------

def test_a_lost_answer_is_not_confirmed(_capture):
    """Producer 4 is the ONLY one where a retry can duplicate."""
    sm = _manager(_server(_OfferRaises()))

    assert sm.deliver_prompt_to_session("s-1", "hello") == NOT_CONFIRMED

    log = _messages(_capture)
    assert "DELIVERY_NOT_CONFIRMED" in log
    assert "cause=offer_failed" in log
    # str(TimeoutError()) is "", so the TYPE has to carry the meaning.
    assert "TimeoutError" in log


@pytest.mark.parametrize("label,server,cause", [
    ("no_server", None, "cause=no_server"),
    ("no_runner_channel", _server(None), "cause=no_runner_channel"),
    ("offer_verb_absent", _server(_NoOfferVerb()), "cause=offer_verb_absent"),
])
def test_every_structural_failure_is_unreachable_and_names_itself(
    label, server, cause, _capture,
):
    """One word to the sender; the mechanism to the operator.

    All four are equally retry-safe (nothing was enqueued) and equally futile
    until repaired, so they share a status.  They want completely different
    FIXES -- restart the runner, wait for spawn, attach a server -- so they do
    not share a log line.
    """
    sm = _manager(server)

    assert sm.deliver_prompt_to_session("s-1", "hello") == UNREACHABLE

    log = _messages(_capture)
    assert "DELIVERY_UNREACHABLE" in log
    assert cause in log, f"{label}: the log did not name the mechanism: {log!r}"


def test_a_failed_drive_is_unreachable_not_not_confirmed(_capture, monkeypatch):
    """Producer 5: the target said idle, and the drive still did not start.

    Nothing was enqueued on EITHER path -- the offer declined to queue (that
    is what ``needs_turn`` means) and the drive failed -- so this is
    retry-safe, and belongs with the structural four rather than with the
    lost-answer case.
    """
    sm = _manager(_server(_NeedsTurn()))
    monkeypatch.setattr(
        SessionManager, "send_message_to_session",
        lambda self, sid, text: False, raising=True,
    )

    assert sm.deliver_prompt_to_session("s-1", "hello") == UNREACHABLE

    log = _messages(_capture)
    assert "cause=drive_failed" in log


def test_a_successful_drive_is_still_accepted(_capture, monkeypatch):
    """The control: the split must not change the path that works."""
    sm = _manager(_server(_NeedsTurn()))
    monkeypatch.setattr(
        SessionManager, "send_message_to_session",
        lambda self, sid, text: True, raising=True,
    )

    assert sm.deliver_prompt_to_session("s-1", "hello") == ACCEPTED


# --------------------------------------------------------------------------
# what the sender is told
# --------------------------------------------------------------------------

def test_neither_transport_failure_reads_as_success():
    """The invariant the whole vocabulary exists for, extended by one word.

    ``not_confirmed`` is the one that could plausibly be argued into
    ``DELIVERED`` -- the message may genuinely be in the target's queue.  It
    is not, and must not be: a MAYBE rendered as a YES is the same silent
    stall as any other failure read as success.
    """
    assert UNREACHABLE not in DELIVERED
    assert NOT_CONFIRMED not in DELIVERED


def test_the_structural_reason_does_not_warn_about_duplicates():
    """The regression, stated as the sentence that must not come back.

    This is the whole defect in one assertion: the four structural failures
    used to carry "the message may still have been enqueued", which is only
    true of the fifth.
    """
    never = _delivery_failure_reason(UNREACHABLE)

    assert "NOTHING WAS SENT" in never
    assert "SAFE" in never
    assert "may still have been enqueued" not in never
    assert "DELIVER IT TWICE" not in never


def test_the_lost_answer_reason_does_warn_about_duplicates():
    maybe = _delivery_failure_reason(NOT_CONFIRMED)

    assert "MAY DELIVER IT TWICE" in maybe
    assert "may be in its queue" in maybe


# --------------------------------------------------------------------------
# the drive failure's own two causes (#626's defect, one layer down)
# --------------------------------------------------------------------------

def test_the_drive_failure_reason_is_not_discarded_at_debug(_capture):
    """``send_message_to_session`` returns False for TWO reasons.

    Its docstring named only one ("the target isn't loaded"), and the other
    -- the dispatch raising -- was logged at DEBUG and discarded.  That is
    verbatim the defect #626 fixed one layer up, in the same file: *"the
    caller is being told the delivery FAILED, so the reason has to be
    somewhere."*
    """
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()

    assert sm.send_message_to_session("s-missing", "hello") is False

    log = _messages(_capture)
    assert "DRIVE_FAILED" in log
    assert "cause=not_loaded" in log


def test_a_raising_dispatch_names_its_exception_type(_capture):
    """And it must name the TYPE, because str(TimeoutError()) is ``""``."""
    sm = SessionManager.__new__(SessionManager)
    sess = type("S", (), {})()
    sess.session_id = "s-1"
    sm._sessions = {"s-1": sess}
    sm._lock = threading.RLock()

    def _boom(*a, **kw):
        raise TimeoutError()

    sm.handle_request = _boom

    assert sm.send_message_to_session("s-1", "hello") is False

    log = _messages(_capture)
    assert "cause=dispatch_raised" in log
    assert "TimeoutError" in log, (
        f"an empty str(exc) left the line naming nothing: {log!r}"
    )


# --------------------------------------------------------------------------
# the structural guard: no producer may go back to sharing a word silently
# --------------------------------------------------------------------------

def test_every_unreachable_producer_logs_a_cause():
    """Checked in the SOURCE, because a sixth producer is the way this
    regresses.

    Adding ``return UNREACHABLE`` without a log line puts a mechanism back
    behind a word that cannot express it, and no runtime test covers a branch
    that does not exist yet.  Read as an AST rather than by grepping text so
    that a mention inside a comment or docstring cannot satisfy it.
    """
    src = pathlib.Path(
        "jaato-server/server/session_manager.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "deliver_prompt_to_session"
    )

    returns = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Return)
        and isinstance(n.value, ast.Name)
        and n.value.id in ("UNREACHABLE", "NOT_CONFIRMED")
    ]
    assert len(returns) == 5, (
        f"expected the five known producers, found {len(returns)}; a new one "
        "needs a log line naming its mechanism before this bound moves"
    )

    # Every one of them is preceded by a logger.warning in the same block.
    warns = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "warning"
    ]
    assert len(warns) >= len(returns), (
        f"{len(returns)} transport failures but only {len(warns)} warnings: "
        "a failure the sender cannot explain and the operator cannot see"
    )

    # And each warning names a cause= token, so the log is greppable by
    # mechanism rather than only by status.
    causes = {
        kw for w in warns
        for a in w.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str)
        for kw in ("no_server", "no_runner_channel", "offer_verb_absent",
                   "offer_failed", "drive_failed")
        if f"cause={kw}" in a.value
    }
    assert causes == {
        "no_server", "no_runner_channel", "offer_verb_absent",
        "offer_failed", "drive_failed",
    }, f"a producer stopped naming its mechanism: found {sorted(causes)}"
