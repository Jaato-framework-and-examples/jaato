"""budget_control binds INSIDE a turn, not only at its end (#955).

A subagent made 196 tool calls under ``tool_calls: 100`` with an ``abort``
rung at 100% and nothing fired, degraded or logged.  The tracker was wired
and the profile reached the session (its ``gc`` block demonstrably applied);
what was missing was the moment of observation.  ``_budget_observe_turn``
was the only writer of ``tool_calls`` / ``seconds`` / ``turns`` and it runs
in the turn's ``finally`` -- so a loop inside ONE ``send_message`` could not
cross any of them however long it ran, and a runaway loop is precisely the
turn that never ends.

The fix observes ``tool_calls`` and ``seconds`` as calls complete
(``_budget_observe_tool_calls``, from every path that records a call), keeps
the turn-end observation as the closing entry that settles the remainder
without double-counting, and traces every ceiling crossing and every fired
rung so a ladder that does not fire is distinguishable from one that is not
wired (the issue's question 3).

Session-level tests reuse the ``test_budget_runtime`` double: a
SimpleNamespace carrying exactly the attributes the methods under test
touch, with the real helpers bound.
"""

import ast
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from shared.jaato_session import (
    _BUDGET_SECONDS_OBSERVED,
    _BUDGET_TOOL_CALLS_OBSERVED,
    JaatoSession,
)
from shared.tests.test_budget_runtime import _cfg, _session
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_SESSION_PY = Path(__file__).resolve().parents[1] / "jaato_session.py"
_SESSION = "jaato-server/shared/jaato_session.py"

REVERSIONS = [
    Reversion(
        target=_SESSION,
        find="""        if self._budget_tracker is None or count <= 0:
            return
        try:
            turn_data[_BUDGET_TOOL_CALLS_OBSERVED] = (""",
        replace="""        if self._budget_tracker is None or count <= 0 or True:
            return
        try:
            turn_data[_BUDGET_TOOL_CALLS_OBSERVED] = (""",
        test="test_tool_calls_ceiling_fires_on_the_call_that_crosses_it",
        because="tool calls are once again counted only at turn end",
    ),
    Reversion(
        target=_SESSION,
        find="""            self._budget_observe_tool_calls(turn_data, 1)

        return tool_results
""",
        replace="""
        return tool_results
""",
        test="test_every_path_that_records_a_tool_call_observes_it",
        because="the sequential path records a call without observing it",
    ),
    Reversion(
        target=_SESSION,
        find="""            self._budget_note_ceilings()
            self._apply_budget_rungs(fired)
        except Exception as exc:  # noqa: BLE001
            logger.warning("budget: tool-call observation failed: %s", exc)""",
        replace="""            self._apply_budget_rungs(fired)
        except Exception as exc:  # noqa: BLE001
            logger.warning("budget: tool-call observation failed: %s", exc)""",
        test="test_ceiling_crossing_is_traced_once_even_without_an_abort_rung",
        because="a ceiling crossed by a ladder that does not abort is silent again",
    ),
    Reversion(
        target=_SESSION,
        find="""            and not getattr(self, "_budget_exhausted_reason", None)
""",
        replace="",
        test="test_no_completion_nudge_for_a_session_at_its_ceiling",
        because="a session at its ceiling is re-prompted into a refused turn",
    ),
]


def _sess(budget=None, trace=None):
    """The runtime double, plus the #955 helpers bound to it."""
    s = _session(budget=budget)
    for name in ("_budget_observe_tool_calls", "_budget_observe_turn",
                 "_budget_note_ceilings", "_budget_trace",
                 "_budget_trace_rung"):
        setattr(s, name, (lambda n: (lambda *a, **k:
                getattr(JaatoSession, n)(s, *a, **k)))(name))
    s._budget_unobserved_seconds = JaatoSession._budget_unobserved_seconds
    s._get_trace_prefix = lambda: "session:subagent:documentalista"
    if trace is not None:
        s._trace = trace.append
    return s


def _turn_data(started_seconds_ago=0.0, calls=0):
    start = datetime.now() - timedelta(seconds=started_seconds_ago)
    return {
        "start_time": start.isoformat(),
        "duration_seconds": None,
        "function_calls": [{"name": "selectReferences"} for _ in range(calls)],
    }


def _abort_at_100(**limits):
    return _cfg(limits=limits, degrade=[{"at": 100, "action": "abort"}])


# --------------------------------------------- the ceiling binds mid-turn

def test_tool_calls_ceiling_fires_on_the_call_that_crosses_it():
    """The issue's profile: tool_calls: 100, abort at 100%.  196 calls ran.
    The 100th call must stop the session, inside the turn."""
    s = _sess(budget=_abort_at_100(tool_calls=100))
    td = _turn_data()
    for _ in range(99):
        s._budget_observe_tool_calls(td, 1)
    assert not hasattr(s, "_stopped")
    assert JaatoSession._refuse_if_budget_exhausted(s) is None

    s._budget_observe_tool_calls(td, 1)

    assert "budget_exhausted" in getattr(s, "_stopped", "")
    assert s._budget_terminal_action == "abort"
    reason = JaatoSession._refuse_if_budget_exhausted(s)
    assert reason is not None and "tool_calls" in reason


def test_a_parallel_batch_is_counted_as_a_whole():
    """The parallel path observes once per batch; the count is the batch."""
    s = _sess(budget=_abort_at_100(tool_calls=100))
    td = _turn_data()
    s._budget_observe_tool_calls(td, 8)
    assert s._budget_tracker.usage.tool_calls == 8
    assert td[_BUDGET_TOOL_CALLS_OBSERVED] == 8


def test_seconds_ceiling_binds_mid_turn():
    """``seconds: 600`` in the issue was not reached, but it had the same
    defect: a stuck loop never reports its duration until it ends."""
    s = _sess(budget=_abort_at_100(seconds=5))
    td = _turn_data(started_seconds_ago=6)
    s._budget_observe_tool_calls(td, 1)
    assert "budget_exhausted" in getattr(s, "_stopped", "")
    assert s._budget_tracker.usage.seconds >= 5


def test_an_unbudgeted_session_is_untouched():
    s = _sess()
    td = _turn_data(calls=2)
    before = dict(td)
    s._budget_observe_tool_calls(td, 1)
    s._budget_observe_turn(td)
    assert td == before


# --------------------------------------- turn end settles, never doubles

def test_turn_end_does_not_recount_what_was_observed_mid_turn():
    s = _sess(budget=_abort_at_100(tool_calls=100))
    td = _turn_data(started_seconds_ago=1, calls=3)
    for _ in range(3):
        s._budget_observe_tool_calls(td, 1)
    td["duration_seconds"] = 1.5
    s._budget_observe_turn(td)
    usage = s._budget_tracker.usage
    assert usage.tool_calls == 3
    assert usage.turns == 1


def test_turn_end_still_counts_calls_no_path_observed():
    """A path that records a call without observing it (none today, see the
    AST guard) is still counted at turn end -- the closing entry settles
    whatever is left, so a budget is exact whichever loop produced it."""
    s = _sess(budget=_abort_at_100(tool_calls=100))
    td = _turn_data(calls=3)
    td["duration_seconds"] = 0.5
    s._budget_observe_turn(td)
    assert s._budget_tracker.usage.tool_calls == 3


def test_seconds_are_handed_out_in_disjoint_slices():
    s = _sess(budget=_abort_at_100(seconds=1000))
    td = _turn_data(started_seconds_ago=10)
    s._budget_observe_tool_calls(td, 1)
    mid = s._budget_tracker.usage.seconds
    assert 9.9 <= mid <= 11.0
    td["duration_seconds"] = 12.0
    s._budget_observe_turn(td)
    assert abs(s._budget_tracker.usage.seconds - 12.0) < 1e-6


def test_bookkeeping_keys_are_popped_before_turn_accounting_sees_them():
    """``turn_data`` is appended to ``_turn_accounting`` right after the
    turn-end observation, and that list is persisted and emitted."""
    s = _sess(budget=_abort_at_100(tool_calls=100))
    td = _turn_data(started_seconds_ago=1, calls=1)
    s._budget_observe_tool_calls(td, 1)
    assert _BUDGET_TOOL_CALLS_OBSERVED in td and _BUDGET_SECONDS_OBSERVED in td
    td["duration_seconds"] = 1.0
    s._budget_observe_turn(td)
    assert _BUDGET_TOOL_CALLS_OBSERVED not in td
    assert _BUDGET_SECONDS_OBSERVED not in td


def test_unreadable_start_time_is_no_news_not_a_guess():
    td = {"start_time": "not-a-timestamp", "function_calls": []}
    assert JaatoSession._budget_unobserved_seconds(td) is None
    assert JaatoSession._budget_unobserved_seconds({"function_calls": []}) is None


# ------------------------------------------------ the ladder is audible

def test_ceiling_crossing_is_traced_once_even_without_an_abort_rung():
    """Question 3 of the issue: a ladder that never logs is
    indistinguishable from a ladder that is not wired."""
    lines = []
    s = _sess(budget=_cfg(limits={"tool_calls": 2},
                          degrade=[{"at": 100, "action": "finalize"}]),
              trace=lines)
    td = _turn_data()
    for _ in range(4):
        s._budget_observe_tool_calls(td, 1)
    ceilings = [l for l in lines if l.startswith("BUDGET CEILING dim=tool_calls")]
    assert len(ceilings) == 1, lines
    assert "used=2 limit=2" in ceilings[0]
    rungs = [l for l in lines if l.startswith("BUDGET RUNG at=100%")]
    assert len(rungs) == 1 and "action=finalize" in rungs[0]
    assert not hasattr(s, "_stopped")          # finalize never stops


def test_abort_traces_the_exhaustion_it_latches():
    lines = []
    s = _sess(budget=_abort_at_100(tool_calls=1), trace=lines)
    s._budget_observe_tool_calls(_turn_data(), 1)
    assert any(l.startswith("BUDGET EXHAUSTED") for l in lines), lines
    assert any("action=abort" in l for l in lines if l.startswith("BUDGET RUNG"))


def test_a_rung_skipped_as_backwards_is_traced_too():
    lines = []
    s = _sess(budget=_cfg(limits={"tool_calls": 10},
                          degrade=[{"at": 50, "action": "finalize"},
                                   {"at": 100, "action": "finalize"}]),
              trace=lines)
    s._budget_applied_rung_pct = 100.0        # the pool already pushed 100%
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(tool_calls=5))
    assert any(l.startswith("BUDGET RUNG_SKIPPED at=50%") for l in lines), lines


def test_trace_failure_never_reaches_the_turn():
    def boom(_msg):
        raise RuntimeError("trace sink is broken")
    s = _sess(budget=_abort_at_100(tool_calls=1))
    s._trace = boom
    s._budget_observe_tool_calls(_turn_data(), 1)   # must not raise
    assert "budget_exhausted" in getattr(s, "_stopped", "")


# --------------------------------------- an exhausted session is not nudged

def test_no_completion_nudge_for_a_session_at_its_ceiling():
    s = SimpleNamespace(_signal_completion_called=False,
                        _completion_nudges_fired=0,
                        _budget_exhausted_reason="budget_exhausted (x)")
    assert JaatoSession.try_completion_nudge(s, 2) == (False, 0)
    s._budget_exhausted_reason = None
    assert JaatoSession.try_completion_nudge(s, 2) == (True, 1)


# ---------------------------------------------------------- the AST guard

def _functions_recording_and_observing():
    tree = ast.parse(_SESSION_PY.read_text())
    recording, observing = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            target = call.func.value
            if (call.func.attr == "append"
                    and isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "turn_data"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "function_calls"):
                recording.add(node.name)
            if call.func.attr == "_budget_observe_tool_calls":
                observing.add(node.name)
    return recording, observing


def test_every_path_that_records_a_tool_call_observes_it():
    """Turn end settles unobserved calls, so a forgotten site would still be
    COUNTED -- at the end of a turn that, for a runaway loop, never comes.
    That is the original defect; keep it from returning one path at a time."""
    recording, observing = _functions_recording_and_observing()
    assert recording, "no path records tool calls -- did turn_data change shape?"
    missing = recording - observing
    assert not missing, (
        f"{sorted(missing)} record tool calls in turn_data['function_calls'] "
        "without calling _budget_observe_tool_calls -- a tool_calls / seconds "
        "ceiling cannot bind mid-turn on that path (#955)"
    )
