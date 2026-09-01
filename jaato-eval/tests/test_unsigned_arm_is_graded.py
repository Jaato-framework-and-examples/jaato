"""An agent that never signed off still produced evidence — jaato #773.

THE OBSERVABLE
==============

Before this, every error terminal out of ``Session.complete()`` was caught
by ``run_arm``'s blanket handler and recorded BLOCKED.  For one of them
that is the wrong state.  When the framework's completion-nudge budget
runs out (``NudgeExhausted``) the agent has run, worked, and committed;
the only thing missing is its ``signal_completion`` call.  The arm was
reported as *nothing to grade* with a complete tree sitting on disk.

Two shapes, and the issue rests on the weaker one:

* a tree that **passes** every grader — the success is lost outright and
  the model is reported as unmeasured on a task it solved.  Reconstructed
  from sweep run 19 by re-running that arm's graders against the tree it
  left, not observed as a BLOCKED-hiding-a-PASS row.
* a tree that **fails** — this one WAS observed live, on the first sweep
  after #767 merged.  It is reported as unmeasured too, and since
  ``report.py`` keeps blocked arms out of the pass-rate denominator by
  design, an arm that genuinely failed silently *raises* the model's
  score.  A measurement bias, not just a missing row.

Both are tested here because the fix is the same and the second is the
claim that does not depend on a reconstruction.

WHAT THESE TESTS HOLD
=====================

Sabotage-first, as the issue asks: revert
:func:`jaato_eval.sign_off.is_unsigned_terminal`'s use in ``_run_session``
and the first two tests come back BLOCKED with a graded-looking tree in
the workspace, which is the observable the issue is named after.

They also pin the two edges that make the rule safe rather than merely
permissive: every OTHER error terminal still blocks (a daemon that died
mid-turn leaves a tree nobody can vouch for), and the graders that read
the sign-off still block on the arm that is missing it.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jaato_eval.arm import ArmResult, ArmSpec
from jaato_eval.graders.base import GraderContext
from jaato_eval.graders.judge import JudgeGrader
from jaato_eval.manifest import GraderSpec
from jaato_eval.sign_off import is_unsigned_terminal
from jaato_eval.verdict import BLOCKED, FAIL, PASS, Verdict

from .test_runner_integration import RunnerHarness

#: What the daemon says when the nudge ceiling is reached, quoted from a
#: live sweep.  Carried through the fakes as prose only: nothing here
#: DECIDES on it, because the engine's rule reads the terminal's TYPE —
#: a test that asserted on this wording would pass an implementation that
#: pattern-matched error text, which is the implementation not to have.
NUDGE_SUMMARY = ("Agent loop exhausted 2 completion nudges without "
                 "calling signal_completion")

#: A task whose only grader reads the completion payload.  Kept beside the
#: workspace-reading one so the per-grader split can be tested at all: the
#: point of the split is that these two adapters diverge on the same arm.
PAYLOAD_TASK = """
id: t/payload
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: write answer.txt containing READY
harness:
  profile: worker
graders:
  - kind: processor
    script: accepts.py
"""

PROCESSOR = """
def validate(payload, context):
    return [] if payload.get("done") else ["not done"]
"""


class UnsignedArmIsGraded(RunnerHarness):
    """The workspace-reading half: a verdict, not a BLOCKED row."""

    def test_nudge_exhausted_arm_with_a_passing_tree_reports_PASS(self):
        """The run-19 reconstruction the issue was filed on.

        Every commit landed, the tree passes the task's grader, and the
        third nudge produced nothing.  Recording that as BLOCKED reports a
        solved task as unmeasured.
        """
        result = self._run({
            "writes": "READY\n",
            "agent_error": "NudgeExhausted",
            "termination_detail": NUDGE_SUMMARY,
        })
        self.assertEqual(result.state, PASS)
        self.assertIsNone(result.blocked_reason)

    def test_nudge_exhausted_arm_with_a_failing_tree_reports_FAIL(self):
        """The live case (issue comment): the weaker claim, and the one
        actually observed.

        BLOCKED here would lose the difference between "the model tried
        and got it wrong" and "we learned nothing" — and because blocked
        arms leave the pass-rate denominator, the failure would improve
        the model's score.
        """
        result = self._run({
            "writes": "not ready\n",
            "agent_error": "NudgeExhausted",
            "termination_detail": NUDGE_SUMMARY,
        })
        self.assertEqual(result.state, FAIL)
        self.assertIsNone(result.blocked_reason)

    def test_the_missing_sign_off_is_recorded_on_the_result(self):
        """A verdict is not the whole story: the arm ended badly too.

        ``error`` set with ``blocked_reason`` unset is precisely the
        record this change adds — the agent produced evidence AND failed
        to declare itself done — and it is what a reader of the JSONL has
        to distinguish this arm from a clean one.
        """
        result = self._run({
            "writes": "READY\n",
            "agent_error": "NudgeExhausted",
            "termination_detail": NUDGE_SUMMARY,
        })
        self.assertIsNotNone(result.error)
        self.assertIn("NudgeExhausted", result.error)
        self.assertIn(NUDGE_SUMMARY, result.error)
        self.assertIn("error", result.to_dict())
        self.assertEqual(result.to_dict()["error"], result.error)

    def test_the_grader_says_it_graded_an_unsigned_arm(self):
        """The verdict carries the caveat even when it is a PASS.

        The exit code means what it always means, but an arm that never
        declared itself done behaved differently from its siblings, and a
        reader comparing them should not have to go to another file for
        that.
        """
        result = self._run({
            "writes": "READY\n",
            "agent_error": "NudgeExhausted",
            "termination_detail": NUDGE_SUMMARY,
        })
        evidence = " ".join(
            line for v in result.verdicts for line in v.evidence)
        self.assertIn("never called signal_completion", evidence)

    def test_every_other_error_terminal_still_blocks(self):
        """The rule is a named exemption, not a general amnesty.

        A daemon that died mid-turn leaves a tree nobody can vouch for,
        and grading it would report on the interruption.  Typed terminals
        that are not the unsigned one keep the conservative reading.
        """
        result = self._run({
            "writes": "READY\n",
            "agent_error": "RunnerCallError",
            "termination_detail": "runner died mid-turn",
        })
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("RunnerCallError", result.blocked_reason)
        self.assertIn("runner died mid-turn", result.blocked_reason)

    def test_an_untyped_session_failure_still_blocks(self):
        """An untyped failure is not guessed at.

        ``_describe_session_failure`` already refuses to invent a type for
        a failure the daemon did not name; the exemption follows the same
        rule, because a failure treated as a known one is how a broken
        daemon starts producing verdicts.
        """
        result = self._run({"raise": "daemon unreachable"})
        self.assertEqual(result.state, BLOCKED)


class UnsignedArmBlocksItsPayloadGraders(RunnerHarness):
    """The payload-reading half: what the missing sign-off does invalidate."""

    task_yaml = PAYLOAD_TASK

    def setUp(self):
        super().setUp()
        (self.root / "cfg" / "accepts.py").write_text(PROCESSOR)

    def test_processor_blocks_and_names_the_sign_off_not_the_schema(self):
        """There is no post-hoc substitute for the payload.

        The generic "declares no completion_payload_schema, or the agent
        never completed" wording sends its reader to check a schema that
        is fine — the same misdirection the judge adapter already records
        against its own guess.  When the engine knows the cause it says it.
        """
        result = self._run({
            "writes": "READY\n",
            "agent_error": "NudgeExhausted",
            "termination_detail": NUDGE_SUMMARY,
        })
        self.assertEqual(result.state, BLOCKED)
        reason = " ".join(v.blocked_reason for v in result.verdicts)
        self.assertIn("never called signal_completion", reason)
        self.assertIn("NudgeExhausted", reason)
        self.assertIn("not implicated", reason)
        # And the arm itself is still not the thing that was blocked: the
        # grader was.  ``blocked_reason`` unset is what says the workspace
        # existed and was offered to every grader.
        self.assertIsNone(result.blocked_reason)


class UnsignedArmBlocksItsJudge(unittest.TestCase):
    """The judge is a payload reader, and blocks — but says why correctly.

    Constructed directly rather than through an arm: the guard fires
    before the judge opens its session, so no SDK stub is needed, and the
    fact that none is needed is itself the assertion — a judge that
    reached the daemon on an unsigned arm would spend a model call to
    score a workspace listing against a claim that does not exist.
    """

    def _context(self, tmp):
        return GraderContext(
            workspace_path=Path(tmp), config_root=Path(tmp),
            payload=None,
            termination_reason="error",
            termination_error_type="NudgeExhausted",
            termination_detail=NUDGE_SUMMARY,
        )

    def test_judge_blocks_naming_the_sign_off_not_a_truncated_run(self):
        spec = GraderSpec(kind="judge", config={"profile": "rubric"})
        with tempfile.TemporaryDirectory() as tmp:
            verdict = JudgeGrader(spec).grade(self._context(tmp))
        self.assertEqual(verdict.state, BLOCKED)
        self.assertIn("never called signal_completion", verdict.blocked_reason)
        self.assertIn("NudgeExhausted", verdict.blocked_reason)
        # The wording the arm is NOT: a tree the agent worked to a stop of
        # its own is not an interrupted one, and telling an operator it was
        # sends them looking for a failure that did not happen.
        self.assertNotIn("truncated", verdict.blocked_reason)


def _spec() -> ArmSpec:
    """An ArmSpec whose only used field is the task id (for ``arm_id``)."""
    return ArmSpec(task=SimpleNamespace(task_id="t/echo"),
                   profile_set="cheap", repeat=0)


class UnsignedArmIsVisibleWhileTheSweepRuns(unittest.TestCase):
    """The live progress line says so too.

    A ✓ on an arm whose agent never declared itself done is correct and
    incomplete: the operator watching the sweep would read it as a clean
    pass, and only find out by opening the results file afterwards.
    """

    def test_progress_line_marks_a_graded_but_unsigned_arm(self):
        from jaato_eval.cli import _progress

        result = ArmResult(spec=_spec())
        result.verdicts = [Verdict(grader_id="script:compiles",
                                   claim="it compiles", state=PASS)]
        result.error = f"NudgeExhausted: {NUDGE_SUMMARY}"
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            _progress(result)
        line = captured.getvalue()
        self.assertIn("✓", line)
        self.assertIn("without a sign-off", line)
        self.assertIn("NudgeExhausted", line)

    def test_a_clean_arm_gets_no_marker(self):
        from jaato_eval.cli import _progress

        result = ArmResult(spec=_spec())
        result.verdicts = [Verdict(grader_id="script:compiles",
                                   claim="it compiles", state=PASS)]
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            _progress(result)
        self.assertNotIn("sign-off", captured.getvalue())


class SignOffRule(unittest.TestCase):
    """The rule itself, in one place because both callers share it."""

    def test_only_the_named_terminal_is_exempt(self):
        self.assertTrue(is_unsigned_terminal("NudgeExhausted"))
        for other in ("RunnerCallError", "APIError", "CascadeExhaustedError",
                      "", None, "nudgeexhausted"):
            with self.subTest(error_type=other):
                self.assertFalse(is_unsigned_terminal(other))


if __name__ == "__main__":
    unittest.main()
