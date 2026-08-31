"""The two entry points a CI job actually branches on.

``run`` and ``report`` both return the verdict exit codes, and a job that
treats 2 as success is the vacuous pass this engine exists to refuse.
These pin the codes and the two flags that had no coverage: ``--resume``
(threaded to the sweep) and the ``report`` subcommand (never tested at
all — it is the half a CI job uses to re-pivot a results file without
re-spending on the arms).
"""
import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from jaato_eval.cli import build_parser, cmd_report, cmd_run, main
from jaato_eval.results import ResultStore


#: The smallest manifest ``discover_tasks`` accepts; the sweep itself is
#: stubbed out, so nothing here is ever executed.
_TASK = """
id: t/echo
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: say READY
harness:
  profile: worker
graders:
  - kind: script
    run: "true"
"""


def _record(arm_id, state, task="t/a", profile_set="cheap"):
    return {"arm_id": arm_id, "task_id": task, "profile_set": profile_set,
            "repeat": 0, "state": state, "blocked_reason": "" if state != "BLOCKED"
            else "nothing was exercised", "verdicts": [], "usage": {},
            "duration_seconds": 1.0, "turns": 1, "finish_reason": "stop",
            "payload_hash": None, "error": None}


class ReportSubcommandCase(unittest.TestCase):
    """``jaato_eval report <file>`` — pivot without re-running anything."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "results.jsonl"
        self.addCleanup(self.tmp.cleanup)

    def _write(self, *states):
        with self.path.open("w") as fh:
            for i, s in enumerate(states):
                fh.write(json.dumps(_record(f"t/a@cheap#{i}", s)) + "\n")

    def _report(self):
        args = build_parser().parse_args(["report", str(self.path)])
        return cmd_report(args)

    def test_all_pass_exits_zero(self):
        self._write("PASS", "PASS")
        self.assertEqual(self._report(), 0)

    def test_a_fail_exits_one(self):
        self._write("PASS", "FAIL")
        self.assertEqual(self._report(), 1)

    def test_a_blocked_exits_two(self):
        self._write("PASS", "BLOCKED")
        self.assertEqual(self._report(), 2)

    def test_fail_outranks_blocked(self):
        """A real defect must not be masked by an unexercised arm."""
        self._write("FAIL", "BLOCKED")
        self.assertEqual(self._report(), 1)

    def test_no_records_exits_two_not_zero(self):
        """Nothing ran is not success.  A job treating 2 as green is the
        vacuous pass; reporting 0 here would hand it one."""
        self._write()
        self.assertEqual(self._report(), 2)

    def test_a_missing_file_exits_two(self):
        self.assertEqual(self._report(), 2)

    def test_a_truncated_trailing_line_does_not_lose_the_file(self):
        """A sweep killed mid-write leaves a partial line; the records
        before it must still pivot."""
        self._write("PASS", "FAIL")
        with self.path.open("a") as fh:
            fh.write('{"arm_id": "t/a@cheap#2", "sta')
        self.assertEqual(self._report(), 1)


class ResumeCase(unittest.TestCase):
    """``--resume`` skips arms already recorded, so a killed sweep does
    not re-spend on completed ones."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "results.jsonl"
        self.addCleanup(self.tmp.cleanup)

    def test_completed_ids_are_what_resume_skips(self):
        store = ResultStore(self.path)
        with self.path.open("w") as fh:
            fh.write(json.dumps(_record("t/a@cheap#0", "PASS")) + "\n")
            fh.write(json.dumps(_record("t/a@cheap#1", "FAIL")) + "\n")
        self.assertEqual(store.completed_arm_ids(),
                         {"t/a@cheap#0", "t/a@cheap#1"})

    def test_a_blocked_arm_counts_as_completed(self):
        """Deliberate: BLOCKED is a recorded outcome.  Re-running it on
        resume would spend again on an arm whose result is already known,
        and the operator can delete the line to force a retry."""
        store = ResultStore(self.path)
        with self.path.open("w") as fh:
            fh.write(json.dumps(_record("t/a@cheap#0", "BLOCKED")) + "\n")
        self.assertIn("t/a@cheap#0", store.completed_arm_ids())

    def test_the_flag_reaches_the_parser(self):
        args = build_parser().parse_args(["run", "tasks", "--resume"])
        self.assertTrue(args.resume)
        self.assertFalse(build_parser().parse_args(["run", "tasks"]).resume)

    def test_arm_timeout_reaches_the_parser(self):
        args = build_parser().parse_args(["run", "tasks", "--arm-timeout", "12"])
        self.assertEqual(args.arm_timeout, 12.0)
        self.assertIsNone(build_parser().parse_args(["run", "tasks"]).arm_timeout)


class WorkspaceRootIsAbsoluteCase(unittest.TestCase):
    """``--workspaces`` is resolved HERE, before it reaches the daemon.

    A workspace path is sent across a socket to a daemon with its own cwd
    and a lifetime longer than any sweep.  Left relative, the harness
    materialised each arm's fixture in one directory while the daemon ran
    the agent in another whenever the two had been started from different
    places — the agent got its worktree without its fixture, the grader
    got the fixture without a repository, and neither side raised
    anything (issue #742).

    The assertion is on ABSOLUTENESS, not on the resolved value matching
    some expected directory: a check of the latter kind passes whenever
    the two processes happen to share a cwd, which is exactly how this
    went unnoticed for several green runs.
    """

    def _captured_workspace_root(self, argv):
        seen = {}

        async def fake_sweep(arms, **kwargs):
            seen["workspace_root"] = kwargs["workspace_root"]
            return []

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "t"
            (task_dir / "fixture").mkdir(parents=True)
            (task_dir / "cfg").mkdir()
            (task_dir / "task.yaml").write_text(_TASK)
            args = build_parser().parse_args(
                ["run", str(task_dir), "--out", str(Path(tmp) / "r.jsonl")]
                + argv)
            with patch("jaato_eval.cli.run_sweep", fake_sweep):
                cmd_run(args)
        return seen["workspace_root"]

    def test_the_default_relative_workspaces_dir_is_resolved(self):
        root = self._captured_workspace_root([])
        self.assertTrue(root.is_absolute(), f"{root} is relative")
        self.assertEqual(root, Path(".jaato-eval-workspaces").resolve())

    def test_an_explicitly_relative_workspaces_dir_is_resolved(self):
        root = self._captured_workspace_root(["--workspaces", "scratch/ws"])
        self.assertTrue(root.is_absolute(), f"{root} is relative")

    def test_an_absolute_workspaces_dir_is_passed_through(self):
        root = self._captured_workspace_root(["--workspaces", "/abs/ws"])
        self.assertEqual(root, Path("/abs/ws"))


class MainDispatchCase(unittest.TestCase):

    def test_report_is_reachable_through_main(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "r.jsonl"
        p.write_text(json.dumps(_record("t/a@cheap#0", "FAIL")) + "\n")
        self.assertEqual(main(["report", str(p)]), 1)


if __name__ == "__main__":
    unittest.main()
