"""One table, two consumers — the contract a grader is handed (jaato #1127).

THE DEFECT.  ``JAATO_EVAL_PYTHON`` names the interpreter jaato-eval runs
under, and therefore the one that HAS ``jaato_sdk``.  The driver-arm
documentation tells an author to use it::

    harness:
      run: '"$JAATO_EVAL_PYTHON" -m ta_cascade.analyze'
    graders:
      - kind: script
        run: '"$JAATO_EVAL_PYTHON" -m ta_cascade.score'

Only the first line worked.  ``driver_environment`` added the interpreter
and the script grader built its own environment out of ``os.environ`` plus
the params, so the scorer expanded ``""``, the shell reported exit 127,
and the adapter — correctly, for what it could see — read that as a
missing toolchain and BLOCKED.  Measured on the first real ``kind:
driver`` consumer: every arm BLOCKED while every arm's driver exited 0
with its workspace on disk.

There was no portable way around it in a manifest: ``python`` is the
``PATH`` bet #1112 removed for the driver, and an absolute path is a
per-host constant in a committed file.

THE FIX is the argument :mod:`jaato_eval.params` already made for the
task's inputs, applied to the rest of the table: ONE builder
(:func:`jaato_eval.contract.arm_environment`), called by the driver and by
the script grader, so the two cannot disagree about a name or a value.
The grader receives the whole table rather than the interpreter alone —
the workspace and config root are what it grades against, the socket is
the daemon the ARM ran on (``GraderContext.socket_path`` exists for that
reason), and the cascade id is how it finds the arm's session records.

WHAT IS PINNED, each against the way it could go wrong:

* the interpreter reaches a script grader, and is the same string the
  driver was given — a table built twice is the defect, so the two are
  compared rather than each checked against ``sys.executable``;
* absence is absence: a variable the arm has no value for is UNSET, not
  empty, which is what makes ``set -u`` and ``[ -n "$V" ]`` work;
* a session arm gets the table too — the export was never driver-only;
* the params keep behaving exactly as they did, collision included,
  because this change must not be visible to a task that used only them;
* the version did not bump: reaching a second consumer is additive.
"""
from __future__ import annotations

import json
import sys
import textwrap
import unittest
from pathlib import Path

from jaato_eval.contract import CONTRACT_VERSION, arm_environment
from jaato_eval.driver import driver_environment
from jaato_eval.graders.script import ScriptGrader
from jaato_eval.manifest import GraderSpec, load_manifest
from jaato_eval.verdict import BLOCKED, PASS

from tests.test_driver_arm import DRIVER, DriverHarness, _events
from tests.test_graders import _context

#: Every ``JAATO_EVAL*`` variable the process was handed, as JSON on
#: stdout.  Written by a grader and by a driver alike, so one test can put
#: the two tables side by side.
DUMP = textwrap.dedent("""\
    import json, os, sys
    json.dump({k: v for k, v in os.environ.items()
               if k.startswith("JAATO_EVAL")}, sys.stdout)
""")


# ---------------------------------------------------------------------------
# The adapter, directly: what one grader sees for one context
# ---------------------------------------------------------------------------

class GraderEnvironmentCase(unittest.TestCase):
    """What a ``script`` grader's command is handed."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        (self.ws / "dump.py").write_text(DUMP)

    def env(self, **ctx):
        """The contract as the grader's own command reports it."""
        spec = GraderSpec(kind="script",
                          config={"run": f'"{sys.executable}" dump.py > env.json'})
        verdict = ScriptGrader(spec).grade(_context(self.ws, **ctx))
        self.assertEqual(verdict.state, PASS,
                         verdict.blocked_reason or verdict.detail)
        return json.loads((self.ws / "env.json").read_text())

    def test_the_interpreter_reaches_the_grader(self):
        """The reported symptom: ``"$JAATO_EVAL_PYTHON" -m pkg.score`` ran
        ``""`` and the shell answered 127, which the adapter reads as a
        missing toolchain."""
        self.assertEqual(self.env()["JAATO_EVAL_PYTHON"], sys.executable)

    def test_a_grader_can_invoke_the_interpreter_by_name(self):
        """Not just present — usable, through the shell, as documented."""
        spec = GraderSpec(
            kind="script",
            config={"run": '"$JAATO_EVAL_PYTHON" -c "import sys; sys.exit(0)"'})
        verdict = ScriptGrader(spec).grade(_context(self.ws))
        self.assertEqual(verdict.state, PASS, verdict.blocked_reason)

    def test_an_empty_interpreter_variable_is_the_127_this_closes(self):
        """The control.  Without the export the expansion is empty and the
        shell cannot run it — so the test above is measuring the fix and
        not the shell being forgiving."""
        spec = GraderSpec(kind="script", config={"run": '"$NOT_SET_ANYWHERE" -c pass'})
        verdict = ScriptGrader(spec).grade(_context(self.ws))
        self.assertEqual(verdict.state, BLOCKED)
        self.assertIn("not found", verdict.blocked_reason)

    def test_the_workspace_and_config_root_reach_the_grader(self):
        cfg = self.ws / "cfg"
        cfg.mkdir()
        env = self.env(config_root=cfg)
        self.assertEqual(env["JAATO_EVAL_WORKSPACE"], str(self.ws))
        self.assertEqual(env["JAATO_EVAL_CONFIG_ROOT"], str(cfg))

    def test_the_socket_reaches_the_grader(self):
        """A grader that opens its own session must reach the daemon the
        ARM ran on — the reason ``GraderContext.socket_path`` exists."""
        self.assertEqual(self.env(socket_path="/tmp/j.sock")["JAATO_EVAL_SOCKET"],
                         "/tmp/j.sock")

    def test_the_cascade_id_reaches_the_grader(self):
        self.assertEqual(self.env(cascade_id="cid-pool")["JAATO_EVAL_CASCADE_ID"],
                         "cid-pool")

    def test_a_value_the_arm_does_not_have_is_unset_not_empty(self):
        """``set -u`` and ``[ -n "$V" ]`` are working guards only if the
        variable is genuinely absent; an empty one reads as a value."""
        env = self.env()
        self.assertNotIn("JAATO_EVAL_SOCKET", env)
        self.assertNotIn("JAATO_EVAL_CASCADE_ID", env)

    def test_the_run_marker_and_the_params_are_unchanged(self):
        """A task using only the #762 export must not notice this at all."""
        env = self.env(agent_params={"issue_id": "716"})
        self.assertEqual(env["JAATO_EVAL"], "1")
        self.assertEqual(env["JAATO_EVAL_PARAM_ISSUE_ID"], "716")
        self.assertEqual(json.loads(env["JAATO_EVAL_PARAMS"]), {"issue_id": "716"})

    def test_a_param_collision_is_still_blocked_not_arbitrated(self):
        """The one failure the table has, reported the grader's way rather
        than raised as the driver's ``ValueError``."""
        spec = GraderSpec(kind="script", config={"run": "true"})
        verdict = ScriptGrader(spec).grade(
            _context(self.ws, agent_params={"issue-id": "715", "issue_id": "716"}))
        self.assertEqual(verdict.state, BLOCKED)
        self.assertIn("rename", verdict.blocked_reason)

    def test_the_contract_version_is_announced_to_the_grader_too(self):
        self.assertEqual(self.env()["JAATO_EVAL_CONTRACT"], CONTRACT_VERSION)


# ---------------------------------------------------------------------------
# The table, as a table: the two consumers cannot disagree
# ---------------------------------------------------------------------------

class OneBuilderCase(unittest.TestCase):
    """The driver's table and the grader's are one object, not two."""

    ARGS = dict(workspace=Path("/w"), config_root=Path("/c"),
                params={"TICKER": "NVDA"}, cascade_id="cid-1",
                socket_path="/tmp/j.sock")

    def test_the_driver_environment_is_the_arm_environment(self):
        """Compared whole rather than key by key: a row added to one side
        and forgotten on the other is exactly the defect."""
        table, collision = arm_environment(**self.ARGS)
        self.assertIsNone(collision)
        self.assertEqual(driver_environment(**self.ARGS), table)

    def test_a_collision_is_one_rule_reported_two_ways(self):
        args = dict(self.ARGS, params={"issue-id": 1, "issue_id": 2})
        table, collision = arm_environment(**args)
        self.assertEqual(table, {})
        self.assertIn("rename", collision)
        with self.assertRaises(ValueError) as caught:
            driver_environment(**args)
        self.assertIn(collision, str(caught.exception))

    def test_reaching_a_second_consumer_did_not_bump_the_version(self):
        """Additive, like a new variable: a driver that has never heard of
        the grader's copy behaves exactly as it did, so refusing on the
        version would refuse a table it does understand."""
        self.assertEqual(CONTRACT_VERSION, "1")


# ---------------------------------------------------------------------------
# End to end: a real arm, a real grader subprocess
# ---------------------------------------------------------------------------

#: A driver task whose GRADER dumps its environment, so the assertion is
#: on what the engine actually handed a grader rather than on a context
#: the test built.
DRIVER_TASK = """
id: t/driver
environment:
  fixture: fixture
  config_root: cfg
input:
  params: {{TICKER: NVDA}}
harness:
  kind: driver
  run: "{python} driver.py ok sid-a"
graders:
  - kind: script
    run: "{python} dump.py > grader_env.json"
"""

#: The session-arm counterpart.  The export was never driver-only, and a
#: session arm's grader has the same reason to name the interpreter.
SESSION_TASK = """
id: t/echo
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: write answer.txt containing READY
  agent_params: {{issue_id: "716"}}
harness:
  profile: worker
graders:
  - kind: script
    run: "{python} dump.py > grader_env.json"
"""


class DriverArmEndToEndCase(DriverHarness):
    """The arm the issue was reported from, driven through the engine."""

    def _task(self, template):
        (self.root / "fixture" / "driver.py").write_text(DRIVER)
        (self.root / "fixture" / "dump.py").write_text(DUMP)
        (self.root / "task.yaml").write_text(
            template.format(python=json.dumps(sys.executable)[1:-1]))
        self.task = load_manifest(self.root / "task.yaml")

    def _tables(self, **kw):
        """``(what the driver saw, what its grader saw)`` for one arm."""
        self._task(DRIVER_TASK)
        result = self._run({"observer_events": _events("sid-a")},
                           keep_workspace=True, **kw)
        self.assertEqual(result.state, PASS,
                         result.blocked_reason or result.verdicts)
        ws = self._workspace(result)
        return (json.loads((ws / "env.json").read_text()),
                json.loads((ws / "grader_env.json").read_text()))

    def test_the_grader_and_the_driver_read_one_table(self):
        """The whole claim, on a real arm: same interpreter, same socket,
        same cid, same params.  Asserted as an equality between the two
        processes' environments rather than against ``sys.executable``,
        because what failed was the two sides disagreeing."""
        driver_env, grader_env = self._tables(
            socket_path="/tmp/j.sock", cascade_driver_id="cid-pool")
        self.assertEqual(grader_env, driver_env)
        self.assertEqual(grader_env["JAATO_EVAL_PYTHON"], sys.executable)
        self.assertEqual(grader_env["JAATO_EVAL_SOCKET"], "/tmp/j.sock")
        self.assertEqual(grader_env["JAATO_EVAL_CASCADE_ID"], "cid-pool")
        self.assertEqual(grader_env["JAATO_EVAL_PARAM_TICKER"], "NVDA")

    def test_an_unpooled_driver_arm_hands_its_grader_the_minted_cid(self):
        """The cid is the arm's, not the pool's, when there is no pool —
        and the grader gets the one the driver's sessions were stamped
        with, so it can find their records."""
        driver_env, grader_env = self._tables()
        cid = driver_env["JAATO_EVAL_CASCADE_ID"]
        self.assertTrue(cid.startswith("jaato-eval-"))
        self.assertEqual(grader_env["JAATO_EVAL_CASCADE_ID"], cid)

    def test_a_session_arm_grader_gets_the_table_too(self):
        self._task(SESSION_TASK)
        result = self._run({"writes": "READY\n", "payload": {"done": True}},
                           keep_workspace=True, socket_path="/tmp/j.sock")
        self.assertEqual(result.state, PASS,
                         result.blocked_reason or result.verdicts)
        ws = self._workspace(result)
        env = json.loads((ws / "grader_env.json").read_text())
        self.assertEqual(env["JAATO_EVAL_PYTHON"], sys.executable)
        self.assertEqual(env["JAATO_EVAL_WORKSPACE"], str(ws))
        self.assertEqual(env["JAATO_EVAL_SOCKET"], "/tmp/j.sock")
        self.assertEqual(env["JAATO_EVAL_PARAM_ISSUE_ID"], "716")
        # No pool was declared, so the arm ran un-cid'd and the variable
        # is absent rather than empty.
        self.assertNotIn("JAATO_EVAL_CASCADE_ID", env)


if __name__ == "__main__":
    unittest.main()
