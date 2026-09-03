"""Guard: the completion gate bounds itself, and only on wrong answers.

`completion_processors` is the framework's fix-until-it-passes loop, and
until issue #768 nothing bounded it.  The processor refuses, the agent
re-claims completion, the processor refuses again.  Observed 2026-09-01:
**seven refusals in 156 seconds**, some nine seconds apart, every one
reporting the same two errors, with no work in between.  The arm ended
BLOCKED having spent its whole budget on the loop, where the run before
it had reached a graded verdict.  Nothing upstream catches this —
``max_turns`` bounds the session rather than this gate, and
``MAX_COMPLETION_NUDGES`` bounds the opposite direction (an agent that
stops WITHOUT signalling).

Authors who hit it wrote their own counter in a module-level global,
which worked only because ``LifecycleTools`` happens to load processors
once per session — undocumented, so a change to per-call loading would
have left every such processor running with a ceiling that had silently
stopped existing (#765).

So this module asserts the four things that make the bound real, each
against the framework's own behaviour rather than against prose:

1. the load-once-per-session caching the counter rests on;
2. ``max_refusals`` blocks exactly that many times, then applies
   ``on_exhausted``;
3. a broken gate — a raise, a malformed return, a load failure — never
   consumes a refusal and is never waved through by exhaustion, because
   an error path returning the same value as success is the defect class
   that produced three separate failures in the session that motivated
   #768;
4. a ``faults[]`` entry (an environment fault) costs no refusal and
   blocks only the one round-trip the agent needs to record it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from shared.completion_processors import (
    LoadedProcessor,
    ProcessorInvocationResult,
    invoke_processors,
)
from shared.plugins.subagent.config import (
    CompletionProcessor,
    _parse_completion_processors,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


#: Put the defect back: drop the ceiling and let the gate refuse forever.
#: That unbounded refusal IS #768 — the seven-refusals-in-156-seconds loop
#: — so neutering the budget must turn this guard red.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/completion_processors.py",
        find="    if lp.refusals >= ceiling:",
        replace="    if False:",
        test="test_max_refusals_blocks_exactly_that_many_times",
        because="the refusal ceiling that terminates the fix-until-it-passes "
                "loop; without it the processor refuses and the agent "
                "re-claims completion until the session's budget is gone",
    ),
]


class _Ctx:
    """The bits of ``RenderContext`` a validate-only processor reads."""

    workspace_path = "/nonexistent"
    config_root = None
    agent_params: Dict[str, Any] = {}
    tool_calls: List[Dict[str, Any]] = []


def _loaded(validate, **entry) -> LoadedProcessor:
    """One loaded processor wrapping *validate*, configured by *entry*."""
    return LoadedProcessor(
        processor=CompletionProcessor(script="p.py", **entry),
        validate_fn=validate,
    )


def _run(lp: LoadedProcessor) -> ProcessorInvocationResult:
    return invoke_processors([lp], payload={}, context=_Ctx())


# ------------------------------------------- 1. the caching the bound rests on

def test_processors_are_loaded_once_per_session():
    """The counter's home.

    ``LifecycleTools`` must resolve the profile's processors once and
    reuse the ``LoadedProcessor`` objects, because that is where
    ``refusals`` accumulates.  Asserted by counting ``load_processors``
    calls across three ``signal_completion``-shaped invocations rather
    than by reading the code, so a refactor to per-call loading fails
    here instead of silently zeroing every ceiling.
    """
    from shared import lifecycle_tools

    calls = {"n": 0}
    real = lifecycle_tools.LifecycleTools.__init__

    class _Session:
        workspace_path = "/nonexistent"
        runtime = None
        _agent_params: Dict[str, Any] = {}
        _completion_processors = [CompletionProcessor(script="p.py")]

        def get_history(self):
            return []

    import shared.completion_processors as cp

    real_load = cp.load_processors

    def counting(*a, **kw):
        calls["n"] += 1
        return real_load(*a, **kw)

    cp.load_processors = counting
    try:
        tools = lifecycle_tools.LifecycleTools.__new__(
            lifecycle_tools.LifecycleTools)
        tools._session = _Session()
        tools._processors_loaded = None
        for _ in range(3):
            tools._run_completeness_gate({})
    finally:
        cp.load_processors = real_load
        assert real is lifecycle_tools.LifecycleTools.__init__

    assert calls["n"] == 1, (
        f"load_processors ran {calls['n']} times across three calls in one "
        f"session; the refusal counter lives on the LoadedProcessor and "
        f"reloading resets it, so the ceiling stops existing (#765, #768)")


# --------------------------------------------------- 2. the ceiling itself

def test_unbounded_is_still_the_default():
    """No ``max_refusals`` = the pre-#768 behaviour, blocking forever.

    The bound is opt-in: a profile that says nothing must not silently
    acquire a ceiling that lets an unfinished completion through.
    """
    lp = _loaded(lambda p, c: ["still wrong"])
    for _ in range(10):
        assert _run(lp).has_fatal
    assert lp.processor.max_refusals is None


def test_max_refusals_blocks_exactly_that_many_times():
    """Three refusals, then the fourth call is allowed through."""
    lp = _loaded(lambda p, c: ["still wrong"], max_refusals=3)
    for attempt in range(3):
        assert _run(lp).has_fatal, f"attempt {attempt} should have blocked"
    fourth = _run(lp)
    assert not fourth.has_fatal, (
        "the fourth call is past the ceiling and on_exhausted defaults to "
        "'allow' — an unfinished completion that gets graded carries "
        "information, a BLOCKED arm carries none")
    assert any("still wrong" in m for _p, m in fourth.warned), (
        "the errors must survive as warnings; waving them through without "
        "an audit trail is not what 'allow' means")
    assert lp.refusals == 3


def test_on_exhausted_fail_keeps_blocking():
    """The other real choice: never accept an unfinished completion."""
    lp = _loaded(lambda p, c: ["still wrong"], max_refusals=2,
                 on_exhausted="fail")
    for _ in range(5):
        assert _run(lp).has_fatal
    assert lp.refusals == 2, "the ceiling still stops counting past itself"


def test_one_refusal_per_invocation_not_per_message():
    """A processor reporting five failures has still only refused once."""
    lp = _loaded(lambda p, c: [f"e{i}" for i in range(5)], max_refusals=2)
    _run(lp)
    assert lp.refusals == 1


def test_the_refusal_names_the_remaining_attempts():
    """#768 rule 7: the return is read by a model about to try again.

    So it must say how many attempts are left and that re-sending an
    unchanged claim spends one — a report phrased for a human log leaves
    the model with no way to know the loop is bounded at all.
    """
    lp = _loaded(lambda p, c: ["still wrong"], max_refusals=3)
    messages = " ".join(m for _p, m in _run(lp).failed)
    assert "2 further attempt" in messages
    assert "spends one" in messages
    last = None
    for _ in range(2):
        last = " ".join(m for _p, m in _run(lp).failed)
    assert "last attempt" in last


def test_a_passing_processor_never_consumes_the_budget():
    lp = _loaded(lambda p, c: [], max_refusals=1)
    for _ in range(5):
        assert not _run(lp).has_fatal
    assert lp.refusals == 0


def test_warnings_and_incomplete_never_consume_the_budget():
    lp = _loaded(
        lambda p, c: {"warnings": ["w"], "incomplete": ["i"]}, max_refusals=1)
    for _ in range(3):
        out = _run(lp)
        assert not out.has_fatal
    assert lp.refusals == 0


# ------------------------------------------------- 3. a broken gate is not a pass

@pytest.mark.parametrize("validate,because", [
    (lambda p, c: (_ for _ in ()).throw(RuntimeError("boom")),
     "a validate that raised did not run its checks"),
    (lambda p, c: "not a list",
     "a validate returning the wrong shape did not report a verdict"),
    (lambda p, c: None,
     "a validate returning nothing reported nothing"),
])
def test_a_broken_gate_blocks_and_costs_no_refusal(validate, because):
    """The most repeated defect class in this codebase: an error path
    returning the same value as success.

    A gate that did not RUN must never read as a gate that PASSED, and
    the agent must not pay retries for a fault its fix cannot address.
    ``None`` is the exception that proves the rule — it is the shape a
    Python function returns when it falls off the end, which is why it
    is asserted here rather than assumed.
    """
    lp = _loaded(validate, max_refusals=1)
    for _ in range(4):
        assert _run(lp).has_fatal, because
    assert lp.refusals == 0, because


def test_a_raise_is_never_waved_through_by_exhaustion():
    """Even with the budget spent, a gate that cannot run keeps blocking."""
    def boom(payload, context):
        raise RuntimeError("the checking script itself broke")

    real = _loaded(lambda p, c: ["wrong"], max_refusals=1)
    _run(real)
    _run(real)  # budget spent; a wrong answer would now be allowed

    broken = _loaded(boom, max_refusals=1)
    for _ in range(4):
        assert _run(broken).has_fatal, (
            "exhaustion accepts an unfinished ANSWER; it must never accept a "
            "gate that is not running")
    assert broken.refusals == 0


def test_a_load_error_costs_no_refusal():
    lp = LoadedProcessor(
        processor=CompletionProcessor(script="missing.py", max_refusals=1),
        load_error="completion_processor 'missing.py' could not be located",
    )
    for _ in range(3):
        assert _run(lp).has_fatal
    assert lp.refusals == 0


# ------------------------------------------------------ 4. environment faults

def test_a_fault_blocks_once_and_costs_no_refusal():
    """#768 rule 6: an unfixable fault must not burn the retry budget.

    It still blocks once — the agent needs one round-trip to record the
    fault in its payload — and is advisory afterwards, because blocking
    repeatedly on something no retry can clear is the non-terminating
    loop this whole mechanism exists to prevent.
    """
    lp = _loaded(
        lambda p, c: {"faults": ["acceptance.sh is missing — retrying will "
                                 "not fix this; record it in errors[]"]},
        max_refusals=3,
    )
    first = _run(lp)
    assert first.has_fatal
    assert any("acceptance.sh" in m for _p, m in first.failed)

    for _ in range(4):
        later = _run(lp)
        assert not later.has_fatal, (
            "a fault blocks exactly one round-trip; blocking again is the "
            "loop that does not terminate")
        assert any("acceptance.sh" in m for _p, m in later.warned)

    assert lp.refusals == 0, (
        "the whole point of the fault channel is that the refusal budget "
        "stays available for genuine check failures")


def test_faults_and_errors_compose():
    """A processor may report both in one return; only errors are budgeted."""
    lp = _loaded(
        lambda p, c: {"errors": ["wrong answer"], "faults": ["env broke"]},
        max_refusals=2,
    )
    out = _run(lp)
    joined = " ".join(m for _p, m in out.failed)
    assert "wrong answer" in joined and "env broke" in joined
    assert lp.refusals == 1
    assert lp.fault_blocks_used == 1


# ---------------------------------------------------------- 5. the profile keys

def test_the_keys_parse_off_a_profile():
    procs = _parse_completion_processors([
        {"script": "a.py", "max_refusals": 3, "on_exhausted": "fail"},
    ])
    assert procs[0].max_refusals == 3
    assert procs[0].on_exhausted == "fail"


@pytest.mark.parametrize("bad", [-1, "3", 3.0, True, None])
def test_an_invalid_ceiling_degrades_to_unbounded_not_to_a_guess(bad):
    """A typo must not invent a ceiling the author did not write — the
    conservative direction for this key is the pre-#768 behaviour."""
    procs = _parse_completion_processors([
        {"script": "a.py", "max_refusals": bad},
    ])
    assert procs[0].max_refusals is None


def test_an_invalid_exhaustion_policy_falls_back_to_the_default():
    procs = _parse_completion_processors([
        {"script": "a.py", "max_refusals": 1, "on_exhausted": "explode"},
    ])
    assert procs[0].on_exhausted == "allow"
