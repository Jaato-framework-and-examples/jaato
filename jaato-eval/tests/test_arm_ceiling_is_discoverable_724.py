"""The ceiling that cut the arm names itself, and the knob that raises it.

THE DEFECT (#724).  A task declaring ``budget.seconds: 1800`` had its arm
cut at 900s and was told only:

    arm exceeded the harness ceiling of 900s and was cut short

The number, and nothing else.  Three wall-clock concepts exist and only
two were discoverable from a manifest:

  1. ``budget.seconds``                  -- the POOL, in the manifest
  2. ``budget_control.limits.seconds``   -- one session, in the profile
  3. the per-arm ceiling (900s default)  -- the harness, settable ONLY via
     ``--arm-timeout``, and named in no manifest documentation at all

So the author's reasonable conclusion -- *my declared 1800 is being
ignored* -- was wrong in a way nothing available to them could correct.

WHAT WAS **NOT** DONE, DELIBERATELY.  The issue originally proposed
feeding ``budget.seconds`` into the arm timeout; its author then withdrew
that in a comment, because the split is by design -- an arm whose ceiling
moved with what earlier arms spent would not be a reproducible
measurement, and a pool's ``seconds`` is reconciled when a session ENDS,
so it cannot bound a session that never ends.  Implementing the original
proposal would have been a regression.  These tests therefore assert the
gates stay SEPARATE, and that the separation is stated rather than left
to be inferred.
"""

import pathlib

from jaato_eval.manifest import BudgetSpec
from jaato_eval.runner import (
    ARM_TIMEOUT_FLAG,
    DEFAULT_ARM_TIMEOUT_SECONDS,
    arm_ceiling_advice,
    describe_arm_timeout,
    effective_arm_timeout,
)

#: NO ``REVERSIONS`` LIST HERE, DELIBERATELY.
#:
#: ``test_every_guard_detects_its_own_reversion`` discovers guard modules
#: under ``_PACKAGES = ("jaato-server/shared/tests",
#: "jaato-server/server/tests")`` only.  This module is in
#: ``jaato-eval/tests``, so a ``REVERSIONS`` list declared here would be
#: read by nothing while looking exactly like a covered one — which is
#: the decorative-guard failure that suite exists to prevent, wearing its
#: own convention as a disguise.
#:
#: Both assertions were instead verified by hand, each reverted alone:
#:
#:   * the BLOCKED message put back to ``"arm exceeded the harness
#:     ceiling of {limit:.0f}s"`` (no knob, no disclaimer)
#:     -> ``test_the_blocked_message_names_the_knob`` FAILS, and
#:     ``test_runner_integration`` fails with it;
#:   * ``arm_ceiling_advice``'s skip predicate forced to ``if True:``
#:     (never warn) -> ``test_a_pool_allowance_over_the_arm_ceiling_warns``
#:     and ``test_each_offending_task_is_named`` FAIL.
#:
#: Extending the meta-guard's reach to this package is worth doing and is
#: not this change: it means teaching its module-name resolution a third
#: package, which is shared machinery every other guard depends on.


class _Task:
    """Minimal stand-in for a discovered manifest.

    ``arm_ceiling_advice`` reads two attributes and is typed loosely to
    avoid a manifest import cycle, so the test exercises it the way the
    CLI does rather than constructing a full TaskManifest.
    """

    def __init__(self, task_id: str, seconds=None):
        self.task_id = task_id
        limits = {"usd": 1.0}
        if seconds is not None:
            limits["seconds"] = seconds
        self.budget = BudgetSpec(limits=limits)


class TestTheEffectiveCeiling:
    def test_unset_resolves_to_the_harness_default(self):
        assert effective_arm_timeout(None) == DEFAULT_ARM_TIMEOUT_SECONDS

    def test_an_explicit_value_wins(self):
        assert effective_arm_timeout(1800) == 1800.0

    def test_zero_disables_and_is_not_read_as_unset(self):
        """``0`` means no ceiling; resolving it to the default would cut
        an arm the operator deliberately unbounded."""
        assert effective_arm_timeout(0) == 0.0

    def test_the_source_is_told_apart(self):
        """Calling an operator's own 600 the 'harness default' would be a
        fresh false statement inside the message written to stop one."""
        assert "default" in describe_arm_timeout(None)
        assert ARM_TIMEOUT_FLAG in describe_arm_timeout(None)
        assert "default" not in describe_arm_timeout(600)
        assert ARM_TIMEOUT_FLAG in describe_arm_timeout(600)


class TestTheMessageNamesTheKnob:
    def test_the_blocked_message_names_the_knob(self):
        """The BLOCKED text must carry the flag and disclaim the pool gate.

        Read from the source rather than by provoking a real timeout: the
        arm would have to actually hang, and a test that sleeps past a
        ceiling is the flakiest possible way to assert a string.
        """
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "jaato_eval" / "runner.py").read_text(encoding="utf-8")
        marker = "arm exceeded the per-arm ceiling"
        assert marker in src, (
            "the timeout message no longer calls this the per-arm ceiling; "
            "#724 is about an author who could not tell it apart from the "
            "task pool's budget.seconds"
        )
        start = src.index(marker)
        window = src[start:start + 700]
        assert "describe_arm_timeout" in window, (
            f"the BLOCKED message does not say where its ceiling came from. "
            f"Naming {DEFAULT_ARM_TIMEOUT_SECONDS:.0f}s without naming "
            f"{ARM_TIMEOUT_FLAG} leaves the author exactly where #724 found "
            f"them."
        )
        assert "budget.seconds" in window, (
            "the BLOCKED message does not distinguish itself from the task "
            "pool's budget.seconds — the confusion the issue reported"
        )


class TestTheSilentDowngradeIsAnnounced:
    def test_a_pool_allowance_over_the_arm_ceiling_warns(self):
        """#724's exact configuration: 1800 declared, 900 enforced."""
        lines = arm_ceiling_advice([_Task("t", seconds=1800)], None)
        assert len(lines) == 1, lines
        assert "1800" in lines[0] and "900" in lines[0]
        assert ARM_TIMEOUT_FLAG in lines[0], (
            "the warning does not name the flag that would raise the ceiling"
        )

    def test_an_allowance_under_the_ceiling_is_silent(self):
        """The example task declares 600 under a 900 ceiling. Warning there
        would train operators to ignore the warning."""
        assert arm_ceiling_advice([_Task("t", seconds=600)], None) == []

    def test_equal_is_not_a_downgrade(self):
        assert arm_ceiling_advice([_Task("t", seconds=900)], None) == []

    def test_a_raised_ceiling_clears_the_warning(self):
        """Acting on the advice must actually silence it."""
        assert arm_ceiling_advice([_Task("t", seconds=1800)], 1800) == []

    def test_a_disabled_ceiling_warns_about_nothing(self):
        """With ``--arm-timeout 0`` no arm is cut, so nothing is downgraded."""
        assert arm_ceiling_advice([_Task("t", seconds=99999)], 0) == []

    def test_a_task_declaring_no_seconds_is_silent(self):
        assert arm_ceiling_advice([_Task("t")], None) == []

    def test_each_offending_task_is_named(self):
        lines = arm_ceiling_advice(
            [_Task("slow", seconds=1800), _Task("fine", seconds=60),
             _Task("slower", seconds=3600)], None)
        assert len(lines) == 2
        assert any(line.startswith("slow:") for line in lines)
        assert any(line.startswith("slower:") for line in lines)


class TestTheGatesStaySeparate:
    def test_budget_seconds_does_not_become_the_arm_ceiling(self):
        """The withdrawn proposal, asserted as NOT implemented.

        If a later change wires the manifest into the timeout, this fails
        — which is the point: the issue's author established that doing so
        breaks arm reproducibility.
        """
        task = _Task("t", seconds=1800)
        assert task.budget.limits["seconds"] == 1800
        assert effective_arm_timeout(None) == DEFAULT_ARM_TIMEOUT_SECONDS, (
            "the arm ceiling now depends on something other than "
            "--arm-timeout; budget.seconds is the POOL's clock and must not "
            "move a per-arm ceiling (#724 comment)"
        )

    def test_all_three_gates_are_documented_where_an_author_reads(self):
        """BudgetSpec named two of three; the third is the one that cut."""
        doc = BudgetSpec.__doc__ or ""
        assert "budget_control" in doc
        assert ARM_TIMEOUT_FLAG in doc, (
            "BudgetSpec's docstring does not mention the per-arm ceiling's "
            "flag. It is the gate an author meets and the only one they "
            "could not discover from the manifest (#724)."
        )
        assert "900" in doc, (
            "BudgetSpec's docstring does not state the default arm ceiling"
        )
