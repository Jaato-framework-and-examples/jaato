"""Our plumbing failing must not be read as the agent failing.

The daemon's model thread ends on a bare ``except Exception`` and terminates
the session for anything it catches:

    except Exception as e:
        terminal_error = e
    ...
    _emit_error_termination_from_exc(terminal_error)
        -> SessionTerminatedEvent(reason="error")
        -> the cascade policy UNLOADS the session
        -> it goes COLD, and a cold sibling is not woken by a sibling message

So a daemon-side ``TimeoutError`` — our own event loop failing to schedule a
coroutine — was treated exactly like a provider rejecting us permanently.
Observed twice on two builds: a stalled loop killed a healthy cascade half
~2.5 minutes in, and its sibling spent the rest of the run sending into a
session that could no longer be woken.

The fix enumerates what must NOT terminate rather than what must.  That
direction is deliberate: an unlisted framework-internal type still dies, which
is the status quo, whereas listing what MUST terminate would let an unlisted
PROVIDER error survive and ``COMPLETION_NUDGE`` cycle on it — the bug the
terminal path exists to stop.
"""

from __future__ import annotations

import inspect

from server.runner_rpc_client import (
    DaemonLoopTimeout,
    RunnerAnswerTimeout,
    RunnerRPCTimeout,
)


def test_transport_timeouts_are_still_TimeoutError():
    """The subclassing is load-bearing, not cosmetic.

    ``ipc.py``, ``command_router.py``, ``session_manager.py`` and
    ``apparmor.py`` all catch ``TimeoutError``.  A fresh exception type would
    silently stop being caught by every one of them.
    """
    for cls in (RunnerRPCTimeout, DaemonLoopTimeout, RunnerAnswerTimeout):
        assert issubclass(cls, TimeoutError), (
            f"{cls.__name__} must remain catchable as TimeoutError"
        )

    caught = False
    try:
        raise DaemonLoopTimeout("loop stalled")
    except TimeoutError:
        caught = True
    assert caught, "an existing `except TimeoutError` would no longer fire"


def test_both_layers_are_distinguishable_by_type_now():
    """#625 made them distinguishable by MESSAGE; this makes them so by TYPE.

    A message is for a human reading a log.  The model thread needs to branch,
    and branching on message text is the thing this codebase keeps being
    bitten by.
    """
    assert issubclass(DaemonLoopTimeout, RunnerRPCTimeout)
    assert issubclass(RunnerAnswerTimeout, RunnerRPCTimeout)
    assert not issubclass(DaemonLoopTimeout, RunnerAnswerTimeout)


def test_the_transport_branch_EXITS_and_cannot_fall_through():
    """The branch must EXIT, checked by parsing it rather than reading near it.

    #628 shipped this branch with no ``return``.  The comment inside it said
    "``terminal_error`` stays None, so the finally skips the termination
    branch" -- and nothing implemented that.  ``terminal_error = e`` sat below,
    outside any ``else``, and ran unconditionally.  A cascade half still died
    3.5 minutes in, with the new WARNING and the old fatal INFO one
    millisecond apart on the same exception.

    The guard that shipped alongside it asserted ``terminal_error`` did not
    appear BETWEEN the branch and the terminal log line -- a window that could
    not contain the assignment, which sits after both.  It was vacuous in the
    only direction that mattered and passed on broken code.

    So this reads the AST: find the ``isinstance(e, RunnerRPCTimeout)`` guard
    and require its body to end in a statement that leaves the handler.
    """
    import ast
    import inspect
    import textwrap

    import server.core as core_mod

    tree = ast.parse(inspect.getsource(core_mod))

    branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "RunnerRPCTimeout" in ast.dump(node.test)
        and "isinstance" in ast.dump(node.test)
    ]
    assert len(branches) == 1, (
        f"expected exactly one transport-timeout guard, found {len(branches)}"
    )
    branch = branches[0]

    last = branch.body[-1]
    assert isinstance(last, (ast.Return, ast.Raise, ast.Continue)), (
        f"the transport branch ends in {type(last).__name__}, so control "
        f"FALLS THROUGH into the terminal path below it and the session dies "
        f"anyway.  It must end in a statement that leaves the handler."
    )
    assert not branch.orelse, (
        "an else here would mean the terminal path is reachable only via it; "
        "that is a different (also valid) shape -- update this guard "
        "deliberately rather than letting both drift"
    )


def test_the_terminal_assignment_is_unreachable_for_transport_timeouts():
    """Complement: the fatal assignment must sit AFTER the branch's exit.

    Stated as a property of the source rather than of a window, because the
    previous version of this test proved a window empty and proved nothing.
    """
    import ast
    import inspect

    import server.core as core_mod

    tree = ast.parse(inspect.getsource(core_mod))
    handlers = [
        h for h in ast.walk(tree)
        if isinstance(h, ast.ExceptHandler)
        and any(
            isinstance(n, ast.If) and "RunnerRPCTimeout" in ast.dump(n.test)
            for n in ast.walk(h)
        )
    ]
    assert len(handlers) == 1, "expected one handler carrying the guard"
    handler = handlers[0]

    guard_idx = next(
        i for i, stmt in enumerate(handler.body)
        if isinstance(stmt, ast.If) and "RunnerRPCTimeout" in ast.dump(stmt.test)
    )
    assigns_after = [
        t.id
        for stmt in handler.body[guard_idx + 1:]
        for n in ast.walk(stmt)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    assert "terminal_error" in assigns_after, (
        "terminal_error is no longer assigned after the guard -- either the "
        "terminal path moved (update this test) or it was removed, which "
        "would let a PROVIDER error survive and nudge-cycle"
    )


def test_provider_errors_still_terminate():
    """The half that must not regress.

    Terminating on a provider error is why this path exists: without it
    COMPLETION_NUDGE restarts the model thread into the same failure.  The fix
    narrows the catch-all; it must not empty it.
    """
    import server.core as core_mod

    src = inspect.getsource(core_mod)
    assert "_emit_error_termination_from_exc" in src
    assert "terminal_error = e" in src, (
        "nothing stamps terminal_error any more -- the terminal path is dead "
        "and a provider error would nudge-cycle"
    )
