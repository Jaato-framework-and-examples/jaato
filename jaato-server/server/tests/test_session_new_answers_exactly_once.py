"""Every ``session.new`` produces exactly one CORRELATED answer frame.

#882's option 3, scoped as #975's comment insisted — to every outcome, not
to the refusals:

    a single guard asserting "every `session.new` produces exactly one
    response frame" would close the class rather than the instance.

THE INVARIANT, stated precisely enough to test.  The client's create-wait
(``IPCClient._await_session_info``) accepts exactly two event shapes,
``SessionInfoEvent`` and ``ErrorEvent``, and discards either one whose
``request_id`` does not equal the id it sent (``_correlates``; the fallback
that accepts an unstamped event applies only to a daemon older than protocol
1.1).  So "produces a response frame" is not enough — an UNCORRELATED answer
is, from the caller's side, identical to no answer at all: it waits out its
full 60 s and raises ``SessionNotConfirmed``.  The invariant is therefore:

    every ``session.new`` emits exactly one ``SessionInfoEvent`` or
    ``ErrorEvent`` carrying the originating ``request_id``.

WHAT EACH ISSUE WAS.  Both were this invariant broken, from opposite ends:

============  ============================================================
#882          five of the seven refusals in ``_create_session_impl`` built
              an ``ErrorEvent`` inline and never set ``request_id`` —
              including profile-not-found and spawn-schema validation, the
              issue's two repros.  The answer WAS on the wire and the
              client dropped it.
#975          the SUCCESS path's fallback confirmation (the ``except``
              around ``_build_session_info_event``) did the same, so a
              session that was created, bootstrapped and serving RPCs
              answered its caller with a 60 s timeout.
============  ============================================================

An audit limited to refusals would have passed while #975 stayed broken,
which is why the behavioural table below drives successes too.

TWO HALVES, DELIBERATELY.

* **AST** — no answer-shaped event may be constructed inside
  ``_create_session_impl`` except as an argument to
  ``_answer_session_new``.  Structural, so a refusal path added later is
  stamped whether or not its author remembers the rule; per-branch review
  is exactly what failed five times here.
* **Behavioural** — a table of outcomes, each driven end-to-end against a
  fake, each asserting ONE correlated frame.  The AST half cannot see a
  path that answers nothing at all (an early ``return ""``, or an
  exception), and that is half of #882.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, Dict, List, Optional

import pytest

import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: the spawn-payload refusal (#882's primary repro)
#: goes around the answer funnel and emits an uncorrelated ErrorEvent again.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="""                self._answer_session_new(client_id, ErrorEvent(
                    error=err_msg,
                    error_type="SpawnPayloadValidationError",""",
        replace="""                self._emit_to_client(client_id, ErrorEvent(
                    error=err_msg,
                    error_type="SpawnPayloadValidationError",""",
        because=(
            "a session.new refusal that bypasses the answer funnel emits an "
            "ErrorEvent with no request_id, which the client's create-wait "
            "discards -- the caller then waits out its full timeout and is "
            "told the session may exist (#882)"
        ),
        test=(
            "jaato-server/server/tests/"
            "test_session_new_answers_exactly_once.py::"
            "test_create_path_answers_only_through_the_funnel"
        ),
    ),
]

SESSION_MANAGER = (
    pathlib.Path(__file__).resolve().parents[2] / "server" / "session_manager.py"
)

#: The event classes the client's create-wait will accept as an answer.
ANSWER_EVENTS = {"SessionInfoEvent", "ErrorEvent"}

#: The one sanctioned emitter.  Everything else in the create path that
#: constructs an ANSWER_EVENTS instance is a path that can go unstamped.
ANSWER_FUNNEL = "_answer_session_new"


# ---------------------------------------------------------------- AST half


def _create_impl_node() -> ast.FunctionDef:
    tree = ast.parse(SESSION_MANAGER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_create_session_impl":
            return node
    raise AssertionError(
        "SessionManager._create_session_impl not found -- if it was renamed, "
        "point this guard at the new name rather than deleting it"
    )


def _callee_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def test_create_path_answers_only_through_the_funnel() -> None:
    """No answer-shaped event is emitted from the create path unfunnelled.

    Walks every ``Call`` in ``_create_session_impl``.  A construction of
    ``SessionInfoEvent`` / ``ErrorEvent`` is allowed only when it reaches
    ``_answer_session_new`` — the call that stamps the correlation id and
    counts the frame — either as a direct argument, or as the value of a
    local the funnel is then handed.  The second shape is the success
    path's: the confirmation is built inside a ``try`` and the fallback
    inside its ``except``, and both land on one funnel call afterwards.
    """
    impl = _create_impl_node()

    # Names the funnel is handed, so `_info = SessionInfoEvent(...)` up the
    # function counts as funnelled.  Deliberately name-based rather than a
    # dataflow analysis: the point is to force the construction through ONE
    # emitter, and a local that is never passed to it still fails below.
    funnelled: set[int] = set()
    funnelled_names: set[str] = set()
    for node in ast.walk(impl):
        if isinstance(node, ast.Call) and _callee_name(node) == ANSWER_FUNNEL:
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Name):
                    funnelled_names.add(arg.id)
                for inner in ast.walk(arg):
                    funnelled.add(id(inner))

    for node in ast.walk(impl):
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if targets & funnelled_names:
            for inner in ast.walk(node.value):
                funnelled.add(id(inner))

    offenders: List[str] = []
    for node in ast.walk(impl):
        if not isinstance(node, ast.Call):
            continue
        if _callee_name(node) not in ANSWER_EVENTS:
            continue
        if id(node) in funnelled:
            continue
        offenders.append(f"{_callee_name(node)}() at line {node.lineno}")

    assert not offenders, (
        "session.new answer-shaped events built outside the "
        f"{ANSWER_FUNNEL}() funnel: {offenders}. Every refusal and every "
        "confirmation must go through it -- it is what stamps request_id, "
        "and an ErrorEvent without one is discarded by the client's "
        "create-wait, which then reports SessionNotConfirmed 60s later "
        "(#882, #975)."
    )


def test_the_funnel_stamps_the_correlation_id() -> None:
    """The funnel's whole job is the stamp; assert the source still does it."""
    src = SESSION_MANAGER.read_text(encoding="utf-8")
    start = src.index(f"    def {ANSWER_FUNNEL}(")
    body = src[start:start + 4000]
    assert "event.request_id = record.request_id" in body, (
        f"{ANSWER_FUNNEL} no longer stamps request_id onto the answer. "
        "Without the stamp every path this guard protects is uncorrelated "
        "again, and the AST check above would still pass."
    )


# -------------------------------------------------------- behavioural half


class _Recorder:
    """Collects what the daemon emitted to one client, as the wire would."""

    def __init__(self) -> None:
        self.events: List[Any] = []

    def __call__(self, client_id: str, event: Any) -> None:
        self.events.append(event)

    def answers(self, request_id: Optional[str]) -> List[Any]:
        """The frames the client's create-wait would ACCEPT for this call.

        Mirrors ``IPCClient._await_session_info`` + ``_correlates`` on a
        modern daemon: the shape must match AND the id must equal.
        """
        out = []
        for ev in self.events:
            if type(ev).__name__ not in ANSWER_EVENTS:
                continue
            if getattr(ev, "request_id", None) != request_id:
                continue
            out.append(ev)
        return out


def _manager(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder):
    """A SessionManager wired to ``recorder``, with no disk and no runner."""
    from server.session_manager import SessionManager

    mgr = SessionManager.__new__(SessionManager)
    import threading

    mgr._lock = threading.RLock()
    mgr._sessions = {}
    mgr._client_to_session = {}
    mgr._reserved_session_ids = set()
    mgr._session_new_answer = threading.local()
    mgr._event_callback = recorder
    return mgr


#: (name, what _create_session_impl is made to do, whether a session exists)
#:
#: Each row is one OUTCOME of session.new.  The third column is what the
#: answer must be able to say about ``may_exist``: a refusal before
#: registration is safe to retry, a failure after it is not (#975).
_OUTCOMES = [
    ("returns empty with no answer", lambda self, *a, **k: "", False),
    ("raises", lambda self, *a, **k: 1 / 0, False),
]


@pytest.mark.parametrize("label,impl,session_exists", _OUTCOMES)
def test_every_outcome_answers_exactly_once(
    monkeypatch: pytest.MonkeyPatch, label: str, impl, session_exists: bool,
) -> None:
    """An impl that answers nothing still produces one correlated frame.

    This is the half the AST scan cannot reach: a ``return ""`` down a path
    nobody stamped, and an exception escaping the impl, both emit no event
    at all — which is #882's symptom exactly. ``create_session``'s record
    closes over both.
    """
    from server.session_manager import SessionManager

    recorder = _Recorder()
    mgr = _manager(monkeypatch, recorder)
    monkeypatch.setattr(SessionManager, "_create_session_impl", impl, raising=True)

    try:
        mgr.create_session("client_1", "s", request_id="rq-42")
    except ZeroDivisionError:
        pass  # the impl's own failure is re-raised by design

    answers = recorder.answers("rq-42")
    assert len(answers) == 1, (
        f"outcome {label!r} produced {len(answers)} correlated answer frames, "
        f"expected exactly 1. Emitted: "
        f"{[type(e).__name__ for e in recorder.events]}"
    )
    assert type(answers[0]).__name__ == "ErrorEvent"
    assert answers[0].details is None or "created_session_id" not in (
        answers[0].details or {}
    ), "nothing was created, so the answer must not claim a session exists"


def test_an_answered_outcome_is_not_answered_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last-resort frame never fires on top of a path that answered.

    The funnel counts frames, so a well-behaved refusal stays a single
    answer. Without this, closing #882 would create the opposite defect —
    two frames, the second of which says nothing useful.
    """
    from server.session_manager import SessionManager
    from jaato_sdk.events import ErrorEvent

    recorder = _Recorder()
    mgr = _manager(monkeypatch, recorder)

    def _refuses(self, client_id, *a, **k):
        self._answer_session_new(client_id, ErrorEvent(
            error="no such profile", error_type="ProfileNotFoundError",
            recoverable=True,
        ))
        return ""

    monkeypatch.setattr(
        SessionManager, "_create_session_impl", _refuses, raising=True)
    mgr.create_session("client_1", "s", request_id="rq-7")

    answers = recorder.answers("rq-7")
    assert len(answers) == 1
    assert answers[0].error_type == "ProfileNotFoundError", (
        "the path's own reason must survive -- the last-resort frame is a "
        "backstop, never a replacement for a refusal that said why"
    )


def test_a_failure_after_registration_says_a_session_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#975's half of ``may_exist``: the answer names the live session.

    ``SessionRefused.may_exist`` is ``False`` because a refusal is normally
    raised before anything is allocated. A failure raised AFTER the session
    is registered inverts every clause of that, and the daemon is the only
    party that can tell the caller which it was.
    """
    from server.session_manager import SessionManager

    recorder = _Recorder()
    mgr = _manager(monkeypatch, recorder)

    def _fails_late(self, client_id, *a, **k):
        # The session became real (the impl latches its id on the record,
        # as the real one does when it registers into ``_sessions``), then
        # something after it blew up.
        self._session_new_answer.record.created_session_id = "20260911_110500"
        raise RuntimeError("state snapshot failed")

    monkeypatch.setattr(
        SessionManager, "_create_session_impl", _fails_late, raising=True)
    with pytest.raises(RuntimeError):
        mgr.create_session("client_1", "s", request_id="rq-9")

    answers = recorder.answers("rq-9")
    assert len(answers) == 1
    assert (answers[0].details or {}).get(
        "created_session_id") == "20260911_110500", (
        "a create that failed after registering the session must say so, or "
        "the SDK reports may_exist=False about a session that is running "
        "and answering RPCs (#975)"
    )


def test_sdk_refusal_reads_the_daemons_answer() -> None:
    """``SessionRefused.may_exist`` follows the daemon, not an assumption."""
    from jaato_sdk import SessionRefused

    plain = SessionRefused("refused", error_type="ProfileNotFoundError")
    assert plain.may_exist is False
    assert plain.session_id is None

    late = SessionRefused("refused", session_id="20260911_110500")
    assert late.may_exist is True, (
        "a refusal naming a created session must report may_exist=True -- "
        "retrying it really does make a second session (#975)"
    )
    assert late.session_id == "20260911_110500"


def test_a_headless_create_is_exempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """No client, no wire, no answer owed — and no crash trying to send one."""
    from server.session_manager import SessionManager

    recorder = _Recorder()
    mgr = _manager(monkeypatch, recorder)
    monkeypatch.setattr(
        SessionManager, "_create_session_impl",
        lambda self, *a, **k: "", raising=True)

    mgr.create_session(None, "s", request_id="rq-1")
    assert recorder.events == []
