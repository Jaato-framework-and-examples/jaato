"""An arm that is a DRIVER, not a session — jaato #1110.

``harness.kind: driver`` runs a process the engine hands a contract to, and
the process opens as many sessions as it likes.  These drive that arm end
to end through the same stub SDK the session arm uses (the observer's
registration is one RPC on the same fake client) and a REAL subprocess —
the driver is a small script the test writes into the fixture, so the
contract is read by an actual child process and the ceiling kills an
actual process group.

What the tests pin, each against the way it could go wrong:

* the contract reaches the driver as environment, under the names the
  module docstring tables;
* the exit-code vocabulary sorts into the engine's two buckets — and the
  non-zero case is an UNSIGNED arm (graded per grader), not BLOCKED;
* an ending the driver did NOT choose stays out of that vocabulary: a
  shell that could not run the command, and a signal nobody here sent,
  are BLOCKED rather than FAILed through a grader;
* sessions are attributed to the arm by the record in ITS workspace, so a
  concurrent sibling's sessions under a shared pool cid are not counted;
* the observer is registered BEFORE the process starts, with event
  CLASSES;
* an arm killed at the ceiling stops the sessions its driver left behind.
"""
from __future__ import annotations

import json
import sys
import textwrap
import unittest
from pathlib import Path

from jaato_eval.driver import (CONTRACT_VERSION, EX_TEMPFAIL, DriverOutcome,
                               arm_cascade_id, attributed_sessions,
                               driver_environment, record_binding)
from jaato_eval.manifest import load_manifest
from jaato_eval.sign_off import (DRIVER_STOPPED_SHORT, describe_unsigned,
                                 is_unsigned_terminal)
from jaato_eval.verdict import BLOCKED, FAIL, PASS

from tests.test_runner_integration import (RunnerHarness, _TerminatedEvent,
                                           _TurnEvent)

#: The driver.  It writes the contract it received (so a test can read it
#: back), persists a session record per id it was told — standing in for
#: the daemon, which writes ``<workspace>/.jaato/sessions/<sid>.json`` at
#: creation — and then ends the way its mode says.
DRIVER = textwrap.dedent("""\
    import json, os, sys, time, pathlib
    mode = sys.argv[1]
    sids = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
    ws = pathlib.Path(os.environ["JAATO_EVAL_WORKSPACE"])
    (ws / "env.json").write_text(json.dumps(
        {k: v for k, v in os.environ.items() if k.startswith("JAATO_EVAL")}))
    (ws / "cwd.txt").write_text(os.getcwd())
    sessions = ws / ".jaato" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    for sid in sids:
        (sessions / f"{sid}.json").write_text(json.dumps({
            "session_id": sid,
            "profile_snapshot": {"model": "openai/gpt-5-mini",
                                 "provider": "openrouter"},
            # Small on purpose: the max-merge against the observer's
            # sums must be decided by the EVENTS in the tests that sum
            # them, and by this record only where a test says so.
            "budget_usage": {"usd": 0.001, "tokens": 30, "turns": 1},
        }))
    if mode == "hang":
        time.sleep(30)
    if mode in ("ok", "short"):
        (ws / "answer.txt").write_text("READY\\n")
    if mode == "wrong":
        (ws / "answer.txt").write_text("nope\\n")
    if mode == "short":
        print("stage 3 died: budget", file=sys.stderr)
        sys.exit(3)
    if mode == "tempfail":
        print("daemon unreachable", file=sys.stderr)
        sys.exit(75)
    sys.exit(0)
""")

TASK = """
id: t/driver
environment:
  fixture: fixture
  config_root: cfg
input:
  params: {{TICKER: NVDA, TRADE_DATE: "2026-08-28"}}
harness:
  kind: driver
  run: "{python} driver.py {mode} {sids}"
graders:
  - kind: script
    run: "grep -qx READY answer.txt"
{extra_graders}
"""


#: The same task with an arbitrary ``run`` — for the endings that happen
#: BEFORE any driver does: a command the shell cannot run at all, and a
#: process a signal ends.  ``run`` is substituted as a JSON string, which
#: is a valid YAML double-quoted scalar whatever the command contains.
RAW_TASK = """
id: t/driver
environment:
  fixture: fixture
  config_root: cfg
input:
  params: {{WORD: READY}}
harness:
  kind: driver
  run: {run}
graders:
  - kind: script
    run: "grep -qx READY answer.txt"
"""


#: A payload-reading grader, indented as a second item of ``graders:``.
PROCESSOR_GRADER = "  - kind: processor\n    script: check.py\n"


class _Created:
    """Mirrors ``AgentCreatedEvent``'s consumed surface: the session id."""

    def __init__(self, session_id):
        self.session_id = session_id


def _events(session_id, cost=0.01, terminal="natural"):
    """A session's life as the observer sees it: created, one turn, ended."""
    created = _Created(session_id)
    turn = _TurnEvent(cost=cost)
    turn.session_id = session_id
    ended = _TerminatedEvent(reason=terminal)
    ended.session_id = session_id
    return [("AGENT_CREATED", created), ("TURN_COMPLETED", turn),
            ("SESSION_TERMINATED", ended)]


class DriverHarness(RunnerHarness):
    """The stub rig, with a driver task instead of a session one."""

    def _driver_task(self, mode, sids="sid-a", extra_graders=""):
        (self.root / "fixture" / "driver.py").write_text(DRIVER)
        (self.root / "task.yaml").write_text(TASK.format(
            python=sys.executable, mode=mode, sids=sids,
            extra_graders=extra_graders))
        self.task = load_manifest(self.root / "task.yaml")

    def _raw_driver_task(self, run):
        """A driver task whose ``run`` is whatever the test needs."""
        (self.root / "fixture" / "driver.py").write_text(DRIVER)
        (self.root / "task.yaml").write_text(RAW_TASK.format(run=json.dumps(run)))
        self.task = load_manifest(self.root / "task.yaml")

    def _workspace(self, result):
        return (self.root / "ws"
                / result.spec.arm_id.replace("/", "_").replace("#", "_"))


class ExitCodeCase(DriverHarness):
    """The vocabulary: 0 graded, 75 BLOCKED, anything else unsigned."""

    def test_exit_zero_is_graded_and_passes(self):
        self._driver_task("ok")
        result = self._run({"observer_events": _events("sid-a")})
        self.assertEqual(result.state, PASS)
        self.assertIsNone(result.error)
        self.assertIsNone(result.blocked_reason)

    def test_exit_zero_with_a_wrong_tree_fails(self):
        """The driver ran to its end; its graders decide, and may FAIL."""
        self._driver_task("wrong")
        result = self._run({"observer_events": _events("sid-a")})
        self.assertEqual(result.state, FAIL)

    def test_tempfail_is_blocked_and_names_the_driver_s_reason(self):
        self._driver_task("tempfail")
        result = self._run({"observer_events": _events("sid-a")})
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("EX_TEMPFAIL", result.blocked_reason)
        self.assertIn("daemon unreachable", result.blocked_reason)
        self.assertEqual(result.verdicts, [])

    def test_a_non_zero_exit_is_an_unsigned_arm_not_a_blocked_one(self):
        """The tree is real; the sign-off is what is missing.

        Recording it BLOCKED would repeat jaato #773 for drivers: a
        failing tree leaves the pass-rate denominator and improves the
        score.  The script grader runs; the record carries ``error`` with
        ``blocked_reason`` unset.
        """
        self._driver_task("short")
        result = self._run({"observer_events": _events("sid-a")})
        self.assertEqual(result.state, PASS)
        self.assertIsNone(result.blocked_reason)
        self.assertIn(DRIVER_STOPPED_SHORT, result.error)
        self.assertIn("exited 3", result.error)
        self.assertIn("stage 3 died", result.error)
        ctx = self.behaviour["graded_context"]
        self.assertEqual(ctx.termination_reason, "exit 3")
        self.assertTrue(ctx.missing_sign_off)
        evidence = " ".join(result.verdicts[0].evidence)
        self.assertIn("driver stopped short", evidence)

    def test_a_payload_reading_grader_blocks_naming_the_driver(self):
        """Not the schema, not the agent: the driver."""
        (self.root / "cfg" / "check.py").write_text(
            "def validate(payload, context):\n    return []\n")
        self._driver_task("short", extra_graders=PROCESSOR_GRADER)
        result = self._run({"observer_events": _events("sid-a")})
        processor = result.verdicts[1]
        self.assertEqual(processor.state, BLOCKED)
        self.assertIn("driver stopped short", processor.blocked_reason)
        self.assertNotIn("signal_completion", processor.blocked_reason)

    def test_a_gradeable_driver_arm_still_has_no_payload(self):
        """Exit 0 says the tree is gradeable, not that a payload exists;
        a processor grader on such an arm blocks rather than validating
        ``None``."""
        (self.root / "cfg" / "check.py").write_text(
            "def validate(payload, context):\n    return []\n")
        self._driver_task("ok", extra_graders=PROCESSOR_GRADER)
        result = self._run({"observer_events": _events("sid-a")})
        self.assertEqual(result.verdicts[0].state, PASS)
        self.assertEqual(result.verdicts[1].state, BLOCKED)
        self.assertIsNone(self.behaviour["graded_context"].payload)


class NotTheDriversChoiceCase(DriverHarness):
    """Endings the exit-code vocabulary must not read as the driver's.

    Each of these would otherwise be an UNSIGNED arm: the script grader
    runs against a tree nothing produced, FAILs, and an environment fault
    is counted against the pass rate — the case BLOCKED exists for, and
    the one the ``script`` grader has always read 127 as.
    """

    def test_a_command_the_shell_cannot_find_is_blocked_not_graded(self):
        """Exit 127, measured: `run: python driver.py` where `python` is
        not on the engine's PATH, no driver executed."""
        self._raw_driver_task("jaato-eval-no-such-interpreter driver.py")
        result = self._run({"observer_events": []})
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("exit 127", result.blocked_reason)
        self.assertIn("not found", result.blocked_reason)
        # The remedy, where the reader is: the contract carries one.
        self.assertIn("JAATO_EVAL_PYTHON", result.blocked_reason)
        # No grader ran, so nothing FAILed an arm that never happened.
        self.assertEqual(result.verdicts, [])
        self.assertIsNone(result.error)

    def test_a_command_that_is_not_executable_is_blocked(self):
        """Exit 126 — found, and the shell could still not run it."""
        script = self.root / "fixture" / "not-executable.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o644)
        self._raw_driver_task("./not-executable.sh")
        result = self._run({"observer_events": []})
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("exit 126", result.blocked_reason)
        self.assertIn("not executable", result.blocked_reason)
        self.assertEqual(result.verdicts, [])

    def test_a_signal_nobody_here_sent_is_blocked_naming_it(self):
        """A negative return code — an OOM kill's shape.

        ``exec`` so the shell is REPLACED by the process that dies,
        whatever shell this host has: the signalled process is then the
        one the engine waited on, which is what makes the return code
        negative rather than the shell's own 128+N.
        """
        self._raw_driver_task(
            f"exec {sys.executable} -c "
            f"'import os, signal; os.kill(os.getpid(), signal.SIGKILL)'")
        result = self._run({"observer_events": []})
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("SIGKILL", result.blocked_reason)
        self.assertIn("the engine did not send it", result.blocked_reason)
        self.assertEqual(result.verdicts, [])

    def test_the_classification_is_a_table(self):
        """The rule itself, including the two codes it must NOT claim."""
        def fault(code):
            return DriverOutcome(exit_code=code, timed_out=False,
                                 stderr_tail=[], stdout_tail=[],
                                 duration_seconds=0.0).fault

        self.assertIsNone(fault(0))
        self.assertIsNone(fault(3), "a code the driver chose is its verdict")
        # 128+N is a code a driver may legally choose (`sys.exit(137)`),
        # so it stays the driver's; only the unambiguous negative form of
        # the same fact is read as a signal.
        self.assertIsNone(fault(137))
        self.assertIn("EX_TEMPFAIL", fault(EX_TEMPFAIL))
        self.assertIn("exit 127", fault(127))
        self.assertIn("exit 126", fault(126))
        self.assertIn("SIGKILL", fault(-9))
        # A ceiling kill has no exit code at all, so nothing to classify:
        # the engine's own kill never reaches the vocabulary.
        self.assertIsNone(fault(None))

    def test_a_signal_this_platform_cannot_name_is_still_reported(self):
        """The number is the fact; the name is the help."""
        outcome = DriverOutcome(exit_code=-999, timed_out=False,
                                stderr_tail=[], stdout_tail=[],
                                duration_seconds=0.0)
        self.assertIn("signal 999", outcome.fault)


class ContractCase(DriverHarness):
    """What the driver process was handed."""

    def test_the_contract_reaches_the_driver_as_environment(self):
        self._driver_task("ok")
        result = self._run({"observer_events": _events("sid-a")},
                           keep_workspace=True, socket_path="/tmp/j.sock",
                           cascade_driver_id="cid-pool")
        ws = self._workspace(result)
        env = json.loads((ws / "env.json").read_text())
        self.assertEqual(env["JAATO_EVAL"], "1")
        self.assertEqual(env["JAATO_EVAL_CONTRACT"], CONTRACT_VERSION)
        self.assertEqual(env["JAATO_EVAL_WORKSPACE"], str(ws))
        self.assertTrue(env["JAATO_EVAL_CONFIG_ROOT"].endswith("cfg"))
        self.assertEqual(env["JAATO_EVAL_SOCKET"], "/tmp/j.sock")
        self.assertEqual(env["JAATO_EVAL_CASCADE_ID"], "cid-pool")
        self.assertEqual(env["JAATO_EVAL_PARAM_TICKER"], "NVDA")
        self.assertEqual(env["JAATO_EVAL_PARAM_TRADE_DATE"], "2026-08-28")
        self.assertEqual(json.loads(env["JAATO_EVAL_PARAMS"]),
                         {"TICKER": "NVDA", "TRADE_DATE": "2026-08-28"})
        # The workspace is also the working directory, and carries the
        # .env the engine writes for the sweep's model axis.
        self.assertEqual((ws / "cwd.txt").read_text(), str(ws))
        self.assertIn("JAATO_PROFILE_SET=cheap", (ws / ".env").read_text())

    def test_the_contract_names_the_interpreter_that_has_the_sdk(self):
        """``run`` inherits the engine's PATH and nothing else, so a task
        written as ``python driver.py`` is a bet on that host's PATH — one
        this probe lost, with exit 127 and no driver executed.  The
        interpreter jaato-eval runs under is the one that HAS jaato_sdk,
        and a driver is an SDK client."""
        self._driver_task("ok")
        result = self._run({"observer_events": _events("sid-a")},
                           keep_workspace=True)
        env = json.loads((self._workspace(result) / "env.json").read_text())
        self.assertEqual(env["JAATO_EVAL_PYTHON"], sys.executable)

    def test_the_new_variable_did_not_bump_the_contract_version(self):
        """Additive: a driver that has never heard of it behaves as it
        did, so refusing the table on the version would refuse a table it
        does understand."""
        self.assertEqual(CONTRACT_VERSION, "1")

    def test_no_socket_means_no_variable(self):
        """Absent, not empty: the driver falls back to the SDK default
        exactly as the engine's own clients do."""
        self._driver_task("ok")
        result = self._run({"observer_events": _events("sid-a")},
                           keep_workspace=True)
        env = json.loads((self._workspace(result) / "env.json").read_text())
        self.assertNotIn("JAATO_EVAL_SOCKET", env)

    def test_params_reach_the_graders_as_agent_params(self):
        """The scorer reads the same two variables the driver was given."""
        self._driver_task("ok")
        self._run({"observer_events": _events("sid-a")})
        ctx = self.behaviour["graded_context"]
        self.assertEqual(ctx.agent_params,
                         {"TICKER": "NVDA", "TRADE_DATE": "2026-08-28"})

    def test_a_pooled_task_hands_the_pool_cid_to_the_driver(self):
        self.assertEqual(arm_cascade_id("t@s#0", "cid-pool"), "cid-pool")

    def test_an_unpooled_arm_still_gets_a_cid(self):
        """The observer needs one; a session arm could run un-cid'd, a
        driver arm cannot."""
        a = arm_cascade_id("t/x@cheap#0", None)
        b = arm_cascade_id("t/x@cheap#0", None)
        self.assertTrue(a.startswith("jaato-eval-t-x-cheap-0-"))
        self.assertNotEqual(a, b)

    def test_driver_environment_refuses_a_param_collision(self):
        with self.assertRaises(ValueError):
            driver_environment(workspace=Path("/w"), config_root=Path("/c"),
                               cascade_id="cid", socket_path=None,
                               params={"issue-id": 1, "issue_id": 2})


class ObserverCase(DriverHarness):
    """Registered before the process, keyed per session, attributed by
    the workspace."""

    def test_registered_before_the_driver_starts_with_event_classes(self):
        self._driver_task("ok")
        self._run({"observer_events": _events("sid-a")}, cascade_driver_id="cid-1")
        [registration] = self.behaviour["observers"]
        self.assertEqual(registration["cid"], "cid-1")
        self.assertEqual(registration["role"], "observer")
        self.assertEqual(set(registration["event_types"]),
                         {"AgentCreatedEvent", "TurnCompletedEvent",
                          "SessionTerminatedEvent", "ErrorEvent"})
        # No session was created on the engine's own connection: the
        # driver opened them, in another process.
        self.assertNotIn("seen_kwargs", self.behaviour)

    def test_usage_is_summed_over_the_arm_s_sessions(self):
        self._driver_task("ok", sids="sid-a,sid-b")
        events = _events("sid-a", cost=0.01) + _events("sid-b", cost=0.02)
        result = self._run({"observer_events": events})
        self.assertEqual(result.session_ids, ["sid-a", "sid-b"])
        self.assertEqual(result.session_id, "sid-a")
        self.assertEqual(result.turns, 2)
        self.assertAlmostEqual(result.usage["cost_usd"], 0.03)
        self.assertEqual(result.usage["spend_total_tokens"], 240)

    def test_a_sibling_arm_s_session_under_the_shared_cid_is_not_counted(self):
        """Two arms of one pooled task share the cid, so the observer sees
        both.  The record in THIS workspace is what says whose."""
        self._driver_task("ok", sids="sid-a")
        events = _events("sid-a", cost=0.01) + _events("sid-other", cost=5.0)
        result = self._run({"observer_events": events})
        self.assertEqual(result.session_ids, ["sid-a"])
        self.assertAlmostEqual(result.usage["cost_usd"], 0.01)
        self.assertEqual(result.turns, 1)

    def test_a_session_the_observer_missed_is_still_the_arm_s(self):
        """Opened before the registration was applied: on disk, not on the
        wire.  Its tracker is the floor."""
        self._driver_task("ok", sids="sid-a,sid-late")
        result = self._run({"observer_events": _events("sid-a", cost=0.01)})
        self.assertEqual(result.session_ids, ["sid-a", "sid-late"])
        # The observer saw one turn; the two records persist one each.
        # The merge never reports less than either source, so the late
        # session's tracker is what makes it two.
        self.assertEqual(result.turns, 2)

    def test_model_and_provider_come_from_the_first_session_s_record(self):
        """SessionInfoEvent is answered to the creating client, never
        routed to an observer — so the record is the source."""
        self._driver_task("ok")
        result = self._run({"observer_events": _events("sid-a")})
        self.assertEqual(result.model, "openai/gpt-5-mini")
        self.assertEqual(result.provider, "openrouter")

    def test_no_sessions_at_all_is_a_measured_nothing(self):
        self._driver_task("ok", sids="")
        result = self._run({"observer_events": []})
        self.assertEqual(result.session_ids, [])
        self.assertIsNone(result.session_id)
        self.assertIsNone(result.model)
        self.assertEqual(result.state, PASS)

    def test_an_abnormal_session_terminal_reaches_the_detail(self):
        self._driver_task("short")
        events = _events("sid-a", terminal="budget_exhausted")
        self._run({"observer_events": events})
        ctx = self.behaviour["graded_context"]
        self.assertIn("sid-a: budget_exhausted", ctx.termination_detail)

    def test_an_observer_that_cannot_connect_blocks_the_arm(self):
        """Nothing was exercised, and the driver was never started."""
        self._driver_task("ok")
        result = self._run({"raise_on_connect": True})
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("could not start", result.blocked_reason)


class CeilingCase(DriverHarness):
    """The per-arm ceiling kills the process group and stops what it left."""

    def test_a_driver_that_never_finishes_is_blocked_and_its_sessions_stopped(self):
        self._driver_task("hang", sids="sid-a")
        created = _Created("sid-a")
        result = self._run({"observer_events": [("AGENT_CREATED", created)]},
                           arm_timeout_seconds=0.5)
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("per-arm ceiling", result.blocked_reason)
        self.assertIn("--arm-timeout", result.blocked_reason)
        self.assertEqual(result.verdicts, [])
        # The driver had opened sid-a and it had no terminal: stopped.
        self.assertEqual(self.behaviour["stopped"], ["sid-a"])
        # And the record is still complete: which session, and the spend
        # its persisted tracker shows, since no turn event ever arrived.
        self.assertEqual(result.session_ids, ["sid-a"])
        self.assertAlmostEqual(result.usage["cost_usd"], 0.001)

    def test_a_session_that_already_ended_is_not_stopped_again(self):
        self._driver_task("hang", sids="sid-a")
        result = self._run({"observer_events": _events("sid-a")},
                           arm_timeout_seconds=0.5)
        self.assertEqual(result.state, BLOCKED)
        self.assertNotIn("stopped", self.behaviour)

    def test_zero_disables_the_ceiling(self):
        self._driver_task("ok")
        result = self._run({"observer_events": _events("sid-a")},
                           arm_timeout_seconds=0)
        self.assertEqual(result.state, PASS)


class WorkspaceCase(DriverHarness):
    def test_workspace_discarded_by_default(self):
        self._driver_task("ok")
        result = self._run({"observer_events": _events("sid-a")})
        self.assertFalse(self._workspace(result).exists())

    def test_budget_ceiling_is_unknown_not_resolved_from_a_profile(self):
        """A driver arm names no profile; a ceiling read off one of its
        stages would be presented as the arm's."""
        self._driver_task("ok")
        result = self._run({"observer_events": _events("sid-a")})
        self.assertIsNone(result.budget_ceiling)


class SignOffRuleCase(unittest.TestCase):
    """The one rule, widened by exactly one member."""

    def test_driver_stopped_short_is_an_unsigned_terminal(self):
        self.assertTrue(is_unsigned_terminal(DRIVER_STOPPED_SHORT))

    def test_the_description_names_the_driver_not_the_agent(self):
        text = describe_unsigned(DRIVER_STOPPED_SHORT)
        self.assertIn("driver", text)
        self.assertNotIn("signal_completion", text)
        self.assertIn("signal_completion", describe_unsigned("NudgeExhausted"))


class AttributionRuleCase(unittest.TestCase):
    """The pure half of attribution, without a process."""

    class _Obs:
        def __init__(self, order):
            self.order = list(order)
            self.sessions = {sid: object() for sid in order}

    def test_records_decide_and_keep_creation_order(self):
        obs = self._Obs(["b", "a", "x"])
        self.assertEqual(attributed_sessions(obs, {"a": {}, "b": {}, "c": {}}),
                         ["b", "a", "c"])

    def test_no_records_takes_everything_seen(self):
        obs = self._Obs(["b", "a"])
        self.assertEqual(attributed_sessions(obs, {}), ["b", "a"])

    def test_record_binding_reads_the_snapshot_and_stays_unknown_otherwise(self):
        self.assertEqual(record_binding({"profile_snapshot": {"model": "m", "provider": "p"}}),
                         ("m", "p"))
        self.assertEqual(record_binding({"model_name": "legacy", "profile_snapshot": {"provider": "p"}}),
                         ("legacy", "p"))
        self.assertEqual(record_binding({}), (None, None))


if __name__ == "__main__":
    unittest.main()
