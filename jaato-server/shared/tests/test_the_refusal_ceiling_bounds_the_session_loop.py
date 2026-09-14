"""Guard: the refusal ceiling bounds THE LOOP, not just the invocation.

The defect in jaato #768/#770 is a loop: the processor refuses, the agent
re-claims completion, the processor refuses again. Seven refusals in 156
seconds, all reporting the same two errors, no work in between; the run ended
with its whole budget spent where the run before it had reached a graded
verdict.

`test_completion_processor_refusal_budget.py` proves the ceiling at the
INVOCATION boundary — it hands `invoke_processors` a `LoadedProcessor` it
built itself and counts. That is necessary and it is not sufficient, because
the loop has a second half the unit never touches: the counter lives on the
`LoadedProcessor`, and it only accumulates because `LifecycleTools` loads
processors ONCE per session and hands the same instances to every
`signal_completion` call. A guard that constructs the `LoadedProcessor` by
hand supplies that persistence itself, so it cannot notice the framework
ceasing to supply it — and the failure mode when that goes is silent: every
processor keeps working, and every ceiling quietly stops existing.

So this module drives the loop the agent is actually in. One
`LifecycleTools`, built the way a session builds it, called repeatedly the way
a retrying agent calls it, with the processor resolved off disk through the
real `script_loader` tier. Nothing here constructs a `LoadedProcessor`.

#770 asks for it sabotage-first: "a processor that always refuses must
terminate at the declared ceiling. Revert the enforcement and watch the
session loop instead — the failing case is the one that does not stop on its
own, so bound the test by turn count." Hence two shapes, deliberately paired:

* the BOUNDED gate must stop refusing inside a fixed number of calls;
* the UNBOUNDED one must still be refusing when that same number is up.

The second is what makes the first mean something. A test that only asserted
"it stopped" would also pass against a gate that never blocked at all, and a
gate that never blocks is the other way to lose the measurement.

Bounded by CALL COUNT rather than by wall clock for the reason #767's
conformance invariant is: the failing case does not stop on its own, so a test
that waited for it to stop would hang instead of failing, and a hang in CI
reads as infrastructure rather than as the defect it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from shared.lifecycle_tools import LifecycleTools
from shared.plugins.subagent.config import _parse_completion_processors
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


#: Put the defect back: reload the processors on every call.
#:
#: This is the reversion the INVOCATION-level guard cannot make, and the whole
#: reason this module exists. Loading per call is a plausible refactor — it
#: looks like a cache nobody needs — and it silently zeroes every ceiling in
#: the tree, because a fresh `LoadedProcessor` starts at `refusals = 0`. The
#: processors all keep working. Only the bound disappears, and it disappears
#: without an error, which is the failure mode #765 flagged and #768 acted on.
#:
#: The anchor is the `signal_completion` site specifically (the
#: `prepare_completion` path below it has the same two lines), pinned by the
#: import block above it.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/lifecycle_tools.py",
        find="            from .dynamic_instructions import build_render_context\n"
             "            if self._processors_loaded is None:",
        replace="            from .dynamic_instructions import build_render_context\n"
                "            if True:",
        test="test_an_always_refusing_gate_stops_refusing_at_the_ceiling",
        because="the load-once-per-session caching the refusal counter lives "
                "on; reloading per call resets it to zero every time, so the "
                "ceiling silently stops existing while every processor goes "
                "on working",
    ),
]


#: How many times a retrying agent is simulated before the test gives up.
#:
#: Generous against every ceiling used here (3 at most) and finite by
#: construction: the failing case is the one that never stops, so the bound is
#: what turns a hang into an assertion.
CALL_BUDGET = 12

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

PAYLOAD = {"summary": "done"}

#: A gate that can never be satisfied — the shape the loop forms around.
ALWAYS_REFUSES = (
    "def validate(payload, context):\n"
    "    return ['the checks still fail; fix them and signal again']\n"
)


class _Hooks:
    """Captures `on_agent_completed`, so "did it actually finish" is a fact."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def on_agent_completed(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _Runtime:
    def __init__(self, config_root: Optional[str] = None) -> None:
        self._config_root = config_root


class _Session:
    """The session surface `LifecycleTools` reads, and nothing more.

    Deliberately a double rather than a real `JaatoSession`: what is under
    test is the mediation between a retrying agent and the processor list, and
    a real session would drag a provider, a runtime and a plugin registry into
    a test about a counter. Everything the counter depends on — the profile's
    parsed processors, the workspace the script resolves against, and the
    single `LifecycleTools` that holds the loaded list — is real.
    """

    def __init__(self, processors, workspace: Path) -> None:
        self._completion_payload_schema = SCHEMA
        self._completion_processors = processors
        self._agent_id = "main"
        self.workspace_path = str(workspace)
        self._ui_hooks = _Hooks()
        self._agent_params: Dict[str, Any] = {}
        self._signal_completion_called = False
        self.runtime = _Runtime()
        self._runtime = self.runtime

    def get_history(self) -> List[Any]:
        return []

    def get_context_usage(self) -> Dict[str, Any]:
        return {}


def _gate(tmp_path: Path, body: str = ALWAYS_REFUSES, *, name: str = "gate",
          **entry) -> LifecycleTools:
    """A `LifecycleTools` over one on-disk processor, wired as a session wires it.

    The entry goes through `_parse_completion_processors` — the profile
    parser — rather than being constructed as a dataclass, so a `max_refusals`
    the parser stops understanding fails here instead of being quietly dropped
    on the way to a `CompletionProcessor` this test built itself.
    """
    script = tmp_path / ".jaato" / "scripts" / "processors" / f"{name}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")
    spec = {"script": f"scripts/processors/{name}.py", "name": name}
    spec.update(entry)
    processors = _parse_completion_processors([spec])
    assert processors, "the parser rejected the test's own processor entry"
    return LifecycleTools(_Session(processors, tmp_path))


def _drive(lt: LifecycleTools, budget: int = CALL_BUDGET) -> List[str]:
    """Re-claim completion until it is accepted, or until *budget* is spent.

    Returns one outcome per call — `"validation_failed"` for a refusal,
    `"completed"` for an acceptance — so an assertion can talk about the SHAPE
    of the loop rather than only its length.
    """
    outcomes: List[str] = []
    for _ in range(budget):
        result = lt._execute_signal_completion(dict(PAYLOAD))
        outcome = result.get("error") or result.get("status") or "?"
        outcomes.append(outcome)
        if outcome == "completed":
            break
    return outcomes


# --------------------------------------------------------------------------
# The pair that makes the claim: bounded stops, unbounded does not.
# --------------------------------------------------------------------------

def test_an_always_refusing_gate_stops_refusing_at_the_ceiling(tmp_path):
    """`max_refusals: 3` means three refusals, then the loop ends.

    The acceptance is the assertion, not the refusals: a gate that blocks
    forever is the incident, and `on_exhausted: allow` is the framework
    choosing a FAIL verdict (which carries information) over a BLOCKED one
    (which carries none).
    """
    lt = _gate(tmp_path, max_refusals=3, on_exhausted="allow")
    outcomes = _drive(lt)

    assert outcomes[-1] == "completed", (
        f"an always-refusing gate was still refusing after {len(outcomes)} "
        f"attempts ({outcomes}) — the ceiling does not bound the loop, which "
        f"is jaato #768's seven-refusals-in-156-seconds incident"
    )
    assert outcomes == ["validation_failed"] * 3 + ["completed"], (
        f"expected exactly 3 refusals then an acceptance; got {outcomes}"
    )


def test_without_a_ceiling_the_loop_does_not_stop_on_its_own(tmp_path):
    """The counterpart, and the reason the test above means anything.

    With no `max_refusals` the behaviour is the pre-#768 one — unbounded, on
    purpose, because a profile that says nothing must not silently acquire a
    ceiling that lets an unfinished completion through. What that costs is
    exactly this: the loop runs until something outside it stops it.
    """
    lt = _gate(tmp_path)
    outcomes = _drive(lt)

    assert outcomes == ["validation_failed"] * CALL_BUDGET, (
        f"an UNBOUNDED gate stopped refusing by itself ({outcomes}) — either "
        f"the default acquired a ceiling it should not have, or this test is "
        f"no longer measuring the loop and the test above proves nothing"
    )
    assert not lt._session._ui_hooks.calls, (
        "the completion event fired for a gate that never accepted"
    )


# --------------------------------------------------------------------------
# The exhaustion policy is a declared choice, at loop level too.
# --------------------------------------------------------------------------

def test_on_exhausted_fail_keeps_the_loop_blocked(tmp_path):
    """`fail` is the other real answer, for callers where an unfinished
    completion is worse than none — a run that writes to a shared store, say.

    It does NOT terminate the loop, and that is the point of it being a
    choice: the ceiling stops the gate lying about why it is blocking, not
    the blocking itself.
    """
    lt = _gate(tmp_path, max_refusals=2, on_exhausted="fail")
    outcomes = _drive(lt)

    assert outcomes == ["validation_failed"] * CALL_BUDGET, (
        f"on_exhausted='fail' let the completion through ({outcomes}); the "
        f"caller that chooses it is the one for whom an unfinished completion "
        f"is the worse outcome"
    )


def test_the_completion_really_lands_once_the_ceiling_is_spent(tmp_path):
    """Exhaustion under `allow` completes the session, it does not merely
    stop saying no.

    Read off `on_agent_completed`, because that is what a driver waiting on
    `complete()` is waiting for. A gate that returned `status: completed`
    without firing it would hang every caller of the SDK facade.
    """
    lt = _gate(tmp_path, max_refusals=1, on_exhausted="allow")
    outcomes = _drive(lt)

    assert outcomes == ["validation_failed", "completed"]
    assert len(lt._session._ui_hooks.calls) == 1, (
        f"the completion was accepted but AgentCompletedEvent did not fire "
        f"({lt._session._ui_hooks.calls!r}) — a driver would wait forever"
    )


# --------------------------------------------------------------------------
# #770's open question, settled: whose budget is it?
# --------------------------------------------------------------------------

def test_each_processor_carries_its_own_ceiling(tmp_path):
    """Per processor per session — not one budget shared across the profile.

    #770 left this open ("Is the counter per session, or per processor per
    session? Multiple processors on one profile currently share nothing").
    The answer is per processor, and it follows from where the counter lives:
    on each `LoadedProcessor`. It is also the only answer that composes —
    processors are merged from a profile's inheritance chain (#791), so a
    shared budget would mean a base profile's gate spending the ceiling of a
    child's, and adding an unrelated gate would tighten every existing one.

    Pinned because the alternative is a plausible refactor with no visible
    symptom: a shared counter still bounds the loop, just sooner and for the
    wrong reason.
    """
    strict = tmp_path / ".jaato" / "scripts" / "processors"
    strict.mkdir(parents=True, exist_ok=True)
    (strict / "first.py").write_text(ALWAYS_REFUSES, encoding="utf-8")
    (strict / "second.py").write_text(ALWAYS_REFUSES, encoding="utf-8")

    processors = _parse_completion_processors([
        {"script": "scripts/processors/first.py", "name": "first",
         "max_refusals": 1, "on_exhausted": "allow"},
        {"script": "scripts/processors/second.py", "name": "second",
         "max_refusals": 3, "on_exhausted": "allow"},
    ])
    assert len(processors) == 2
    lt = LifecycleTools(_Session(processors, tmp_path))
    outcomes = _drive(lt)

    # The longer ceiling governs: the completion is refused while EITHER gate
    # is still objecting, so the loop ends when the slower one is spent.
    assert outcomes == ["validation_failed"] * 3 + ["completed"], (
        f"expected the 3-refusal gate to govern while the 1-refusal gate went "
        f"advisory; got {outcomes}. A shared budget would have ended it after "
        f"one or two."
    )
    spent = [lp.refusals for lp in lt._processors_loaded]
    assert spent == [1, 3], (
        f"each processor should have spent its OWN ceiling and no more; got "
        f"{spent}"
    )


# --------------------------------------------------------------------------
# What the retrying agent is actually told.
# --------------------------------------------------------------------------

def test_the_agent_is_told_how_many_attempts_remain(tmp_path):
    """The count reaches the model, in the result it reads before retrying.

    #770 asked whether it belonged there rather than in every author's
    hand-written string. It does: the framework owns the count, so an author
    who wrote the sentence themselves would be quoting a number they cannot
    see, and one who omitted it leaves the model re-sending an unchanged claim
    with no way to know it is nearly out of attempts.
    """
    lt = _gate(tmp_path, max_refusals=3, on_exhausted="allow")
    first = lt._execute_signal_completion(dict(PAYLOAD))

    assert first["error"] == "validation_failed"
    blob = " ".join(first.get("processor_errors") or [])
    assert "2" in blob, (
        f"the first of three refusals did not tell the agent that two "
        f"attempts remain: {blob!r}"
    )
    assert "the checks still fail" in blob, (
        f"the processor's own message did not survive alongside the "
        f"framework's budget note: {blob!r}"
    )


def test_a_broken_gate_is_never_waved_through_by_the_ceiling(tmp_path):
    """Exhaustion accepts an unfinished ANSWER, never a gate that is not running.

    Drives the loop past the ceiling against a processor that raises. If
    exhaustion applied here, a session would complete on a gate that never
    graded anything — the error-path-returns-success class this whole area
    is prone to, arriving by the one door built to let completions through.
    """
    lt = _gate(
        tmp_path,
        "def validate(payload, context):\n"
        "    raise RuntimeError('the checker is broken')\n",
        max_refusals=1, on_exhausted="allow",
    )
    outcomes = _drive(lt)

    assert outcomes == ["validation_failed"] * CALL_BUDGET, (
        f"a raising processor was eventually accepted ({outcomes}); the "
        f"refusal ceiling must not apply to a gate that is not running"
    )
    assert all(lp.refusals == 0 for lp in lt._processors_loaded), (
        "a broken gate spent the agent's refusals; no retry it makes can fix "
        "a processor that raises"
    )


def test_the_processors_are_loaded_once_across_the_whole_loop(tmp_path):
    """The persistence the counter rests on, asserted where it is supplied.

    The invocation-level guard checks `load_processors` is called once by
    counting calls; this checks the consequence at the loop's own boundary —
    the SAME objects carry the count from one `signal_completion` to the next.
    Identity rather than call count, so a future cache that reloaded but
    restored the counter would still pass, and one that quietly handed back
    fresh objects would not.
    """
    lt = _gate(tmp_path, max_refusals=3, on_exhausted="allow")
    lt._execute_signal_completion(dict(PAYLOAD))
    first = lt._processors_loaded
    assert first is not None and first[0].refusals == 1

    lt._execute_signal_completion(dict(PAYLOAD))
    assert lt._processors_loaded is first, (
        "LifecycleTools handed out a different processor list on the second "
        "call; the refusal counter lives on these objects, so reloading them "
        "resets every ceiling to zero without any error being raised"
    )
    assert first[0].refusals == 2
