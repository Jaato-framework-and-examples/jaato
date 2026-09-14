"""A turn that ran must terminate the session's event stream, usage or not.

#881.  ``_turn_accounting`` is a USAGE ledger -- a turn lands in it only when
the provider reported tokens -- and the post-turn event fan-out gated on that
ledger growing.  So a provider that completed a turn and reported nothing
produced NEITHER ``TurnCompletedEvent`` NOR ``SessionTerminatedEvent``, and a
driver blocked on either (``ask()`` / ``stream()`` on the first,
``complete()`` on the second) waited out its own timeout while the daemon sat
content and logged nothing on either side.

Measured against a live daemon with the ``echo`` provider: dropping
``plugin_configs.echo.usage`` was the whole difference between a cascade that
returns and one that hangs.  Nothing about that is specific to ``echo`` -- a
stream that never delivers a usage frame, a gateway that strips the field, a
zero-cost cached turn all land in the same state.  ``echo`` is where it is
guaranteed, and it is the first provider every new harness reaches for.

WHAT THE OLD GATE ACTUALLY BOUGHT, since that is what decides the fix.  The
rpc docstring justified ``total > 0`` as refused-turn suppression, and a
refused turn is indeed suppressed -- but not by it: ``send_message`` returns
before the chat loop is entered, so no ``turn_data`` is ever built and the
append site is never reached.  The gate was therefore suppressing exactly one
thing nobody asked it to: a turn that ran and was not metered.

Option 2 of the issue (append every turn, drop the gate) is the one NOT taken,
and ``test_the_usage_ledger_still_only_records_metered_turns`` is what pins
that: ``len(_turn_accounting)`` is read as a metered-turn count by
``get_context_usage()['turns']``, by ``get_consumption``'s
unattributed-turn reconciliation and by the persisted ``turn_count``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from server.runner.rpc import RunnerRPC
from shared.jaato_session import JaatoSession
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_SESSION = "jaato-server/shared/jaato_session.py"
_RPC = "jaato-server/server/runner/rpc.py"

assert (Path(__file__).resolve().parents[1] / "jaato_session.py").is_file()

REVERSIONS = [
    Reversion(
        target=_SESSION,
        find="""        self._turns_ran += 1
        self._last_turn_ran = turn_data
        if turn_data['total'] > 0:
            self._turn_accounting.append(turn_data)""",
        replace="""        if turn_data['total'] > 0:
            self._turns_ran += 1
            self._last_turn_ran = turn_data
            self._turn_accounting.append(turn_data)""",
        test="test_a_turn_that_reported_no_usage_still_counts_as_having_run",
        because="the lifecycle counter is gated on usage again",
    ),
    Reversion(
        target=_RPC,
        find="""        last_turn = RunnerRPC._last_turn_ran(session, turn_accounting)
        if last_turn is None:
            return""",
        replace="""        last_turn = (turn_accounting or [None])[-1]
        if last_turn is None:
            return""",
        test="test_an_unmetered_turn_still_fires_both_halves_of_the_terminus",
        because=("the post-turn payload is sourced from the usage ledger "
                 "again, so an unmetered turn has nothing to report and "
                 "reports nothing"),
    ),
]


# --------------------------------------------------------------- the session


def _turn(total: int = 0, **extra) -> dict:
    """A turn-accounting dict shaped like the one the chat loop builds."""
    data = {
        'prompt': 0, 'output': 0, 'total': total,
        'spend_total': 0, 'spend_prompt': 0, 'spend_output': 0,
        'cost_usd': None, 'duration_seconds': 1.5,
        'finish_reason': 'stop', 'function_calls': [],
    }
    data.update(extra)
    return data


def _bare_session() -> SimpleNamespace:
    """The fields ``_record_turn_ran`` touches, with the real methods bound.

    Same shape as ``test_budget_runtime._session``: a double carrying exactly
    what the methods under test read, so the REAL implementation runs rather
    than a paraphrase of it.
    """
    s = SimpleNamespace(
        _turns_ran=0, _last_turn_ran=None, _turn_accounting=[],
        _unmetered_warning_emitted=False,
        _active_provider_name="echo", _model_name="echo",
        _trace=lambda *a, **k: None,
    )
    for name in ("_record_turn_ran", "get_turns_ran", "get_last_turn_ran",
                 "get_turn_accounting", "_warn_unmetered_turn_once"):
        setattr(s, name, getattr(JaatoSession, name).__get__(s))
    return s


def test_a_turn_that_reported_no_usage_still_counts_as_having_run():
    """The whole of #881 at its source."""
    s = _bare_session()
    unmetered = _turn(total=0)
    s._record_turn_ran(unmetered)

    assert s.get_turns_ran() == 1, (
        "a turn that ran was not counted as having run; the post-turn "
        "fan-out gates on this and would emit nothing"
    )
    assert s.get_last_turn_ran() is unmetered, (
        "no payload for the turn that ran -- the ledger is empty, so the "
        "event would have nothing to report"
    )


def test_the_usage_ledger_still_only_records_metered_turns():
    """Option 2 is the one NOT taken.

    ``len(_turn_accounting)`` is read as a METERED-turn count in three
    places; appending an all-zero turn would change what each of them means.
    """
    s = _bare_session()
    s._record_turn_ran(_turn(total=0))
    metered = _turn(total=1200)
    s._record_turn_ran(metered)

    assert s.get_turns_ran() == 2
    assert s.get_turn_accounting() == [metered], (
        "an unmetered turn entered the usage ledger; every consumer of "
        "len(turn_accounting) now counts a turn that spent nothing"
    )


def test_a_refused_turn_records_nothing_because_it_never_reaches_here():
    """The suppression the old gate was credited with, located properly.

    ``send_message`` returns at the budget gate before the chat loop, so
    ``_record_turn_ran`` is never called for a refusal.  Asserted as the
    absence of a call rather than as a branch inside it, because a branch
    here would be a second suppression path competing with the early return.
    """
    s = _bare_session()
    assert s.get_turns_ran() == 0
    assert s.get_last_turn_ran() is None


# ------------------------------------------------------------- the fan-out


class _Hooks:
    def __init__(self):
        self.turns = []
        self.contexts = []
        self.histories = []

    def on_agent_turn_completed(self, **kw):
        self.turns.append(kw)

    def on_agent_context_updated(self, **kw):
        self.contexts.append(kw)

    def on_agent_history_updated(self, **kw):
        self.histories.append(kw)


def _rpc_session(hooks, *, turns_ran, last_turn, ledger):
    flushed = []
    s = SimpleNamespace(
        _ui_hooks=hooks, _agent_id="main",
        get_turns_ran=lambda: turns_ran,
        get_last_turn_ran=lambda: last_turn,
        get_turn_accounting=lambda: list(ledger),
        get_context_usage=lambda: {},
        get_history=lambda: [],
        flush_session_quiescent=lambda: flushed.append(True),
    )
    s._flushed = flushed
    return s


def _fwd(session, turns_before):
    RunnerRPC._forward_post_turn_hooks(SimpleNamespace(), session, turns_before)


def test_an_unmetered_turn_still_fires_both_halves_of_the_terminus():
    """The symptom, at the site that produces it.

    BOTH halves matter and neither substitutes for the other:
    ``on_agent_turn_completed`` is what ``ask()`` / ``stream()`` settle on,
    ``flush_session_quiescent`` is what ``complete()`` settles on.  Option 3
    of the issue -- moving only the flush out of the gate -- is what this
    asserts against.
    """
    hooks = _Hooks()
    s = _rpc_session(hooks, turns_ran=1, last_turn=_turn(total=0), ledger=[])
    _fwd(s, turns_before=0)

    assert len(hooks.turns) == 1, (
        "no TurnCompletedEvent for a turn that ran; ask()/stream() block "
        f"until their own timeout (hooks saw {hooks.turns})"
    )
    assert hooks.turns[0]["total_tokens"] == 0
    assert hooks.turns[0]["finish_reason"] == "stop"
    assert hooks.turns[0]["duration_seconds"] == 1.5, (
        "the turn reported no TOKENS; it still ran for a real duration"
    )
    assert s._flushed == [True], (
        "no SessionTerminatedEvent flush for a turn that ran; complete() "
        "blocks until its own timeout"
    )


def test_a_turn_that_did_not_run_is_still_not_forwarded():
    """Refused / no-op paths stay suppressed, now on the lifecycle count.

    Firing here would re-emit the PREVIOUS turn's tokens and duration: a
    client counting turns over-counts, one summing tokens double-counts.
    """
    hooks = _Hooks()
    previous = _turn(total=2504, spend_total=4654)
    s = _rpc_session(hooks, turns_ran=1, last_turn=previous, ledger=[previous])
    _fwd(s, turns_before=1)            # the count did not move

    assert hooks.turns == []
    assert s._flushed == []


def test_turn_number_counts_turns_that_ran_not_ledger_entries():
    """The ordinal follows the lifecycle, so two unmetered turns differ.

    ``jaato-tui/agent_registry.py`` indexes a list by ``turn_number``;
    sourcing it from the ledger gave every unmetered turn the same ordinal,
    so the second overwrote the first.
    """
    hooks = _Hooks()
    s = _rpc_session(hooks, turns_ran=4, last_turn=_turn(total=0), ledger=[])
    _fwd(s, turns_before=3)

    assert hooks.turns[0]["turn_number"] == 3


def test_a_session_without_the_lifecycle_accessors_behaves_as_before():
    """The fallback is the PRE-#881 behaviour, not an invention.

    A duck-typed double or an out-of-tree session class has no
    ``get_turns_ran``; it must still get the ledger-length gate it had, not
    an exception and not an unguarded double emission.
    """
    hooks = _Hooks()
    old = SimpleNamespace(
        _ui_hooks=hooks, _agent_id="main",
        get_turn_accounting=lambda: [_turn(total=99)],
        get_context_usage=lambda: {},
        get_history=lambda: [],
    )
    _fwd(old, turns_before=0)
    assert len(hooks.turns) == 1
    assert hooks.turns[0]["total_tokens"] == 99

    hooks2 = _Hooks()
    old.get_turn_accounting = lambda: [_turn(total=99)]
    old._ui_hooks = hooks2
    _fwd(old, turns_before=1)          # ledger did not grow
    assert hooks2.turns == []


# ------------------------------------------- the signal that was not there


def test_an_unmetered_turn_says_so_once_naming_the_provider(caplog):
    """#688 item 3: there was no signal at all.

    Landing #881 alone would stop the hang and leave the operator with an
    empty consumption report, a ``tokens`` / ``usd`` ceiling being fed zero,
    and nothing anywhere saying why.  Once per session, because the condition
    belongs to the provider rather than to the turn.
    """
    import logging

    s = _bare_session()
    with caplog.at_level(logging.WARNING, logger="shared.jaato_session"):
        s._record_turn_ran(_turn(total=0))
        s._record_turn_ran(_turn(total=0))
        s._record_turn_ran(_turn(total=0))

    hits = [r for r in caplog.records if "no token usage" in r.getMessage()]
    assert len(hits) == 1, (
        f"expected exactly one warning across three unmetered turns, "
        f"got {len(hits)}"
    )
    msg = hits[0].getMessage()
    assert "echo" in msg, f"the warning does not name the provider: {msg}"
    assert "budget_control" in msg, (
        f"the warning does not say what silently stops working: {msg}"
    )


def test_a_metered_turn_says_nothing(caplog):
    """The warning must not fire for a session whose provider reports."""
    import logging

    s = _bare_session()
    with caplog.at_level(logging.WARNING, logger="shared.jaato_session"):
        s._record_turn_ran(_turn(total=1200))

    assert [r for r in caplog.records
            if "no token usage" in r.getMessage()] == []
