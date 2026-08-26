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


def test_the_model_thread_spares_the_session_for_transport_timeouts():
    """Checked in source: the branch must exist and must not set terminal_error.

    A runtime test would need a live model thread, a stalled loop and a
    cascade policy; the property that matters is one branch, and its absence
    is what killed the sessions.
    """
    import server.core as core_mod

    src = inspect.getsource(core_mod)
    idx = src.index("MODEL_THREAD_TERMINAL_ERROR error_type=%s")
    # the guard must come BEFORE the terminal branch
    head = src[:idx]
    guard = head.rindex("isinstance(e, RunnerRPCTimeout)")
    assert guard > head.rindex("except Exception as e:"), (
        "the transport-timeout guard must sit inside the catch-all, before "
        "the terminal path"
    )

    # CODE ONLY.  The branch's comment explains what it deliberately does
    # NOT do and names terminal_error while doing so -- a raw substring check
    # reads the explanation as the behaviour.  Third time today; the rule is
    # that a guard must look at what runs, not at what is written near it.
    window = "\n".join(
        line for line in src[guard:idx].splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert "terminal_error" not in window, (
        "the transport branch must NOT stamp terminal_error -- that is what "
        "routes it to SessionTerminatedEvent(reason='error')"
    )
    assert "recoverable=True" in window, (
        "the emitted ErrorEvent must say the session survives"
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
