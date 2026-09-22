"""A ``return`` in a ``finally`` discards the exception in flight (#1077).

``server/core.py`` carried FOUR of them, all inside one ~300-line ``finally``
in ``model_thread`` -- the closure ``_start_model_thread`` launches on the
daemon's model thread.  Python 3.14 makes the shape a ``SyntaxWarning``
(PEP 765), which is how it was found; nothing here is a diagnosis of a
reported incident.

WHAT WAS ACTUALLY EXPOSED, because it is narrower than the warning implies.
The handlers above that ``finally`` catch ``Exception`` and
``KeyboardInterrupt``, and a thread target's return value is read by nobody,
so the ordinary paths were already fine.  Two cases were not:

1. a ``BaseException`` that is neither of the two caught -- ``SystemExit``,
   ``GeneratorExit``, an injected ``asyncio.CancelledError``;
2. **an exception raised INSIDE either ``except`` handler.**  The
   ``except Exception`` arm runs ~100 lines of failure reporting (event
   emission, teardown, logging).  A failure while reporting a failure was
   discarded along with the provider error it was in the middle of
   reporting -- on the model thread, where nothing else would say so.

Both only bite when the ``finally`` REACHES one of its four exits: a
terminal error, a stashed continuation, a drained user send, or a completion
nudge.  A wind-down that simply falls off the end always propagated.  That
is why the two behavioural tests here stage a stashed continuation -- it is
the cheapest of the four to reach, and without it they would pass against
the defect.

The fix lifts the wind-down out of the ``finally`` into ``_finish_turn``
verbatim.  The four exits become ordinary returns from an ordinary function,
the ``finally`` is one call that falls off its end, and an in-flight
exception survives to ``threading.excepthook``.

WHAT EACH TEST WOULD MISS IF NEUTERED
-------------------------------------

* :func:`test_no_return_inside_a_finally_in_core` is the structural guard.
  Neuter it -- scope it to a line range, stop recursing, drop ``Return`` from
  the node set -- and a ``return`` put back inside any ``finally`` in this
  file goes unnoticed, which is the whole defect returning.  It is
  deliberately file-scoped rather than statement-scoped: the four sites were
  in one ``finally`` today, and #1077 is about the shape, not those lines.
* :func:`test_the_wind_down_still_has_its_early_exits` is what stops the
  guard above passing for the wrong reason.  A ``finally`` with no returns
  because the wind-down was DELETED satisfies the AST check perfectly.
* :func:`test_a_base_exception_survives_the_wind_down` and
  :func:`test_a_failure_while_reporting_a_failure_survives` are the two
  exposures, driven through the real ``_start_model_thread`` on a real
  thread.  Neuter either by dropping the stashed continuation and it passes
  against the defect, because the wind-down then never reaches an exit.
* :func:`test_the_wind_down_still_restarts_the_turn` pins the half that must
  NOT change: the continuation still starts a fresh model thread.  Without
  it, "no exception was swallowed" would be satisfiable by a wind-down that
  stopped doing its job.

VERIFIED NON-VACUOUS, twice, and the counts are part of the claim:

* against the real pre-fix ``core.py`` (the wind-down back inside the
  ``finally``, four ``return``s and all): **4 fail**.
* against the registered reversion below, which is the defect in its
  smallest form: **3 fail** -- the same three, minus the early-exit count,
  which that reversion does not disturb.

``test_the_wind_down_still_restarts_the_turn`` passes in both, and is
supposed to: it pins what #1077 must NOT change.  Both behavioural failures
read ``assert [] == [...]`` -- an empty ``threading.excepthook`` capture,
which is the swallow itself.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from typing import Any, List, Optional

from jaato_server.server.core import JaatoServer
from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"
_CORE_PY = Path(__file__).resolve().parents[1] / "core.py"

#: The reversion is the defect in its smallest form: one bare ``return`` at
#: the end of the ``finally``.  It compiles, and it is exactly what the four
#: original sites did on every path that reached them -- run the wind-down,
#: then discard whatever was in flight.
_FIXED = """            finally:
                # No ``return`` may live in here -- see ``_finish_turn``
                # above and #1077.  Anything in flight propagates.
                _finish_turn()
"""
_BROKEN = """            finally:
                # No ``return`` may live in here -- see ``_finish_turn``
                # above and #1077.  Anything in flight propagates.
                _finish_turn()
                return
"""

REVERSIONS = [
    Reversion(
        target=_CORE,
        find=_FIXED,
        replace=_BROKEN,
        test="test_no_return_inside_a_finally_in_core",
        because="a return is back inside a finally in core.py",
    ),
    Reversion(
        target=_CORE,
        find=_FIXED,
        replace=_BROKEN,
        test="test_a_failure_while_reporting_a_failure_survives",
        because=(
            "an exception raised inside the except handler is swallowed by "
            "the finally again"
        ),
    ),
]


# ---------------------------------------------------------------------------
# The structural guard
# ---------------------------------------------------------------------------


#: Nodes whose body owns its own ``return`` / ``break`` / ``continue``.  A
#: ``return`` inside one of these, declared inside a ``finally``, leaves the
#: FUNCTION, not the ``finally`` -- which is the shape of the #1077 fix
#: itself, so flagging it would make the guard un-satisfiable.
_OWN_SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

#: Loops declared inside a ``finally``: a ``break`` bound to one transfers
#: control WITHIN the ``finally`` and drops nothing.  PEP 765 does not warn
#: about it either.
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def _returns_in_finally(source: str) -> List[ast.AST]:
    """Every ``return`` / ``break`` / ``continue`` that LEAVES a ``finally``.

    ``break`` and ``continue`` are in the node set because PEP 765 warns about
    all three for the same reason -- a ``finally`` that transfers control out
    of itself drops the exception in flight -- and excluding them would leave
    the same defect reachable by another spelling.  Both exclusions above are
    about what the statement is BOUND to, not about where it is written.
    """
    found: List[ast.AST] = []

    def walk(node: ast.AST, in_loop: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _OWN_SCOPE):
                continue
            if isinstance(child, ast.Return):
                found.append(child)
            elif isinstance(child, (ast.Break, ast.Continue)) and not in_loop:
                found.append(child)
            walk(child, in_loop or isinstance(child, _LOOPS))

    # ``except*`` is a separate node type, not a ``Try`` subclass, and a
    # ``finally`` on one swallows exactly the same way.  Looked up rather than
    # named so the guard still imports on an interpreter without it.
    try_types = tuple(
        t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None
    )

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, try_types):
            continue
        for stmt in node.finalbody:
            if isinstance(stmt, _OWN_SCOPE):
                continue
            if isinstance(stmt, (ast.Return, ast.Break, ast.Continue)):
                found.append(stmt)
            walk(stmt, isinstance(stmt, _LOOPS))
    return found


def test_no_return_inside_a_finally_in_core() -> None:
    """No ``finally`` in ``server/core.py`` transfers control out of itself.

    Replaces ``python3.14 -W error::SyntaxWarning``, which cannot be run on
    every developer's interpreter.  Asserting the property from the AST is
    the same claim, available on 3.11.
    """
    source = _CORE_PY.read_text(encoding="utf-8")
    assert "def model_thread()" in source, (
        "core.py no longer defines model_thread, so this guard is aimed at "
        "nothing.  Re-aim it rather than deleting it -- an unrun guard must "
        "not read as a passing one."
    )
    offenders = _returns_in_finally(source)
    assert not offenders, (
        "return/break/continue inside a `finally` in "
        f"{_CORE} at line(s) "
        f"{', '.join(str(n.lineno) for n in offenders)}.\n\n"
        "A `finally` that transfers control out of itself DISCARDS whatever "
        "exception is in flight (PEP 765; a SyntaxWarning on 3.14).  In this "
        "file that is the daemon's model thread, where the discarded thing "
        "is typically the account of why a session just failed -- including "
        "a failure raised while REPORTING one.\n\n"
        "Fix: move the work into a helper the `finally` calls, so the exits "
        "are ordinary returns from an ordinary function.  See `_finish_turn` "
        "and #1077."
    )


def test_the_finally_scan_discriminates() -> None:
    """The scan above must see the defect and must not see its lookalike.

    Driven on synthetic source rather than on ``core.py``, because the file
    is (correctly) clean: a scanner that returned ``[]`` unconditionally
    passes :func:`test_no_return_inside_a_finally_in_core` forever, which is
    the way this guard would most plausibly stop discriminating.

    The lookalike is the fix itself in miniature -- a ``return`` inside a
    function DECLARED in the ``finally``.  PEP 765 does not warn about it and
    neither does this: the return belongs to that function.
    """
    defect = "def f():\n    try:\n        g()\n    finally:\n        return 1\n"
    lookalike = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    finally:\n"
        "        def inner():\n"
        "            return 1\n"
        "        inner()\n"
    )
    nested_loop = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    finally:\n"
        "        for x in y:\n"
        "            break\n"
    )

    assert len(_returns_in_finally(defect)) == 1, "the plain defect is missed"
    assert _returns_in_finally(lookalike) == [], (
        "a return inside a function declared in the `finally` was flagged.  "
        "That is the SHAPE OF THE FIX -- flagging it makes the guard "
        "un-satisfiable, which gets guards deleted."
    )
    assert _returns_in_finally(nested_loop) == [], (
        "a `break` bound to a loop INSIDE the `finally` was flagged; it "
        "transfers control within the finally, not out of it"
    )


def test_the_wind_down_still_has_its_early_exits() -> None:
    """The guard above must not be passing because the code went away.

    ``_finish_turn`` is the old ``finally`` body, and its four exits are the
    reason it exists as a function: each hands the rest of the turn to a
    freshly started model thread.  A ``finally`` with nothing in it satisfies
    the AST check and breaks the daemon.
    """
    tree = ast.parse(_CORE_PY.read_text(encoding="utf-8"))
    finishers = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_finish_turn"
    ]
    assert len(finishers) == 1, (
        f"expected exactly one `_finish_turn` in {_CORE}, found "
        f"{len(finishers)}"
    )
    returns = [n for n in ast.walk(finishers[0]) if isinstance(n, ast.Return)]
    assert len(returns) >= 4, (
        f"`_finish_turn` has {len(returns)} early exit(s); the wind-down it "
        "was lifted out of had four (terminal error, stashed continuation, "
        "drained user send, completion nudge).  Losing one means a turn that "
        "used to hand off to a fresh model thread now falls through into the "
        "status report below it."
    )


# ---------------------------------------------------------------------------
# The two exposures, driven end to end
# ---------------------------------------------------------------------------


class _InjectedCancel(BaseException):
    """Stands in for the third row of #1077's table.

    A ``BaseException`` that neither handler catches: ``SystemExit``,
    ``GeneratorExit``, an ``asyncio.CancelledError`` injected into the
    thread.  Named rather than reusing ``SystemExit`` because
    ``threading.excepthook`` silently ignores that one, which would make the
    capture below empty for a reason unrelated to the defect.
    """


class _ReadyStub:
    """``server._runner_ready``, with a way to raise from the first line.

    ``model_thread``'s ``try`` opens with ``_runner_ready.is_set()``, so
    raising there puts an exception in flight without needing a runner, a
    workspace or a session env.
    """

    def __init__(self, exc: Optional[BaseException] = None) -> None:
        self._exc = exc

    def is_set(self) -> bool:
        if self._exc is not None:
            raise self._exc
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:  # pragma: no cover
        return True


class _RecordingServer(JaatoServer):
    """Keeps a handle on the model thread the finally would null out.

    ``model_thread``'s wind-down sets ``_model_thread = None`` as its second
    statement, so reading the attribute after ``_start_model_thread`` returns
    is a race with the thread it just started.  Recording the assignment is
    not.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_model_thread" and value is not None:
            object.__setattr__(self, "_thread_seen", value)
        object.__setattr__(self, name, value)


def _server(
    ready_exc: Optional[BaseException] = None,
    emit_raises_on: Optional[str] = None,
) -> _RecordingServer:
    """A ``JaatoServer`` carrying exactly what this path touches.

    ``__new__`` rather than a constructed server: ``initialize()`` wants a
    provider, a workspace and a runner, and none of them is on the code path
    under test.  Every attribute set here is read by ``model_thread`` or its
    wind-down; a missing one fails loudly as an ``AttributeError`` on the
    thread rather than passing quietly.
    """
    srv = _RecordingServer.__new__(_RecordingServer)
    srv._thread_seen = None
    srv._model_running = False
    srv._model_thread = None
    srv._terminal_reason = None
    srv._runner_ready = _ReadyStub(ready_exc)
    # No runner: the wind-down's drain and nudge probes are both gated on
    # ``_runner_rpc is not None``, so they are skipped and the stashed
    # continuation below is the exit that gets taken.
    srv._runner_rpc = None
    srv._pending_continuation_lock = threading.Lock()
    srv._pending_continuations = [("continue", [])]
    srv._agents = {}
    srv._main_agent_id = "main"
    srv._waiting_for_channel_input = False
    srv._profile = None
    srv.emitted = []
    srv.restarted = []

    def emit(event: Any) -> None:
        srv.emitted.append(event)
        if emit_raises_on is not None and type(event).__name__ == emit_raises_on:
            raise RuntimeError("emit failed while reporting the turn's failure")

    srv.emit = emit
    srv._start_model_thread = (
        lambda prompt, attachments=None: srv.restarted.append(prompt)
    )
    srv._trace = lambda *a, **k: None
    return srv


def _run_turn(srv: _RecordingServer) -> List[Any]:
    """Run one real model thread, returning what escaped it.

    ``threading.excepthook`` is where an unhandled exception on a thread
    goes; capturing it is the only way to observe the difference between
    "propagated" and "swallowed", since the target's return value is read by
    nobody -- which is precisely why the defect was invisible.
    """
    caught: List[Any] = []
    previous = threading.excepthook
    threading.excepthook = caught.append
    try:
        JaatoServer._start_model_thread(srv, "hi")
        assert srv._thread_seen is not None, "no model thread was started"
        srv._thread_seen.join(timeout=30)
        assert not srv._thread_seen.is_alive(), "model thread did not finish"
    finally:
        threading.excepthook = previous
    return caught


def test_a_base_exception_survives_the_wind_down() -> None:
    """Row 3 of #1077: a ``BaseException`` neither handler catches.

    An injected cancellation is not an ``Exception``, so it reaches the
    ``finally`` still in flight.  With a stashed continuation waiting, the
    wind-down takes its second exit -- which used to be a ``return``, and
    used to be where the cancellation stopped existing.
    """
    srv = _server(ready_exc=_InjectedCancel("cancelled mid-turn"))

    caught = _run_turn(srv)

    assert [type(c.exc_value).__name__ for c in caught] == ["_InjectedCancel"], (
        "the injected BaseException did not escape the model thread.  An "
        "empty capture is the swallow itself: the wind-down's `return` "
        "discarded it and the thread exited reporting nothing."
    )


def test_a_failure_while_reporting_a_failure_survives() -> None:
    """Row 4 of #1077, the expensive one.

    The turn fails, ``except Exception`` starts reporting it, and the report
    itself raises.  ``terminal_error`` is assigned AFTER the emit, so it is
    still ``None`` when the wind-down runs -- the turn looks ordinary, takes
    the continuation exit, and the second failure used to vanish with the
    first one's account of itself.
    """
    srv = _server(emit_raises_on="ErrorEvent")

    caught = _run_turn(srv)

    assert [type(c.exc_value).__name__ for c in caught] == ["RuntimeError"], (
        "a failure raised inside the `except Exception` handler did not "
        "escape the model thread.  This is the case #1077 calls expensive: "
        "the session fails, the reporting of it fails, and neither reaches "
        "anybody."
    )
    assert "ErrorEvent" in [type(e).__name__ for e in srv.emitted], (
        "the handler never got as far as emitting, so the test is not "
        "exercising the failure-while-reporting path it claims to"
    )


def test_the_wind_down_still_restarts_the_turn() -> None:
    """The half that must NOT change: the exits still hand off.

    Both tests above would also pass against a wind-down that stopped
    draining continuations -- an exception propagates nicely from code that
    does nothing.  The stashed message must still become a fresh model
    thread, on the propagating path as on the ordinary one.
    """
    srv = _server(ready_exc=_InjectedCancel("cancelled mid-turn"))

    _run_turn(srv)

    assert srv.restarted == ["continue"], (
        "the stashed continuation was not replayed into a fresh model "
        f"thread (saw {srv.restarted!r}).  #1077 changes which exceptions "
        "survive the wind-down and nothing else."
    )
    assert srv._model_running is False, (
        "the wind-down did not clear `_model_running`, so a later send would "
        "be treated as arriving mid-turn"
    )
