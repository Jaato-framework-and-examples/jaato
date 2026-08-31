"""Grader adapters sort harness faults from agent faults."""
import tempfile
import unittest
from pathlib import Path

from jaato_eval.graders import REGISTRY, GraderContext
from jaato_eval.graders.processor import ProcessorGrader, _reads_ledger
from jaato_eval.graders.script import ScriptGrader
from jaato_eval.ledger import LedgerResult
from jaato_eval.manifest import GRADER_KINDS, GraderSpec
from jaato_eval.verdict import BLOCKED, FAIL, PASS


def _context(tmp, **kw):
    kw.setdefault("workspace_path", tmp)
    kw.setdefault("config_root", tmp)
    return GraderContext(**kw)


class RegistryCase(unittest.TestCase):
    def test_registry_matches_manifest_kinds(self):
        """The two copies of the kind list must agree — executed, so the
        copy can fail."""
        self.assertEqual(set(REGISTRY), set(GRADER_KINDS))


class ScriptGraderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def grade(self, config, **ctx):
        return ScriptGrader(GraderSpec(kind="script", config=config)).grade(
            _context(self.ws, **ctx))

    def test_exit_zero_passes(self):
        self.assertEqual(self.grade({"run": "true"}).state, PASS)

    def test_nonzero_exit_fails(self):
        """A command that ran and rejected the work is a real failure."""
        self.assertEqual(self.grade({"run": "false"}).state, FAIL)

    def test_missing_command_is_blocked_not_failed(self):
        """A toolchain absent from the runner says nothing about the agent."""
        v = self.grade({"run": "definitely-not-a-real-binary-xyz"})
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("not found", v.blocked_reason)

    def test_timeout_is_blocked(self):
        v = self.grade({"run": "sleep 5", "timeout_seconds": 0.2})
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("exceeded", v.blocked_reason)

    def test_truncated_arm_is_blocked(self):
        """A workspace produced by a cut-short run must not be graded."""
        v = self.grade({"run": "true"}, finish_reason="max_tokens")
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("max_tokens", v.blocked_reason)

    def test_runs_in_the_workspace(self):
        (self.ws / "marker").write_text("x")
        self.assertEqual(self.grade({"run": "test -f marker"}).state, PASS)

    def test_evidence_captured_on_failure(self):
        v = self.grade({"run": "echo boom >&2; false"})
        self.assertEqual(v.state, FAIL)
        self.assertTrue(any("boom" in line for line in v.evidence))


class ScriptGraderParamsCase(unittest.TestCase):
    """The task's own inputs reach the shell (jaato #762).

    A script grader cannot read ``GraderContext``, so an input-dependent
    check used to have no choice but to hardcode the input — after which
    re-pointing the task at a different input graded every arm against
    the old one's criteria, silently.  These exercise the export that
    lets the grader follow the input instead.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def grade(self, run, params, **ctx):
        spec = GraderSpec(kind="script", config={"run": run})
        return ScriptGrader(spec).grade(
            _context(self.ws, agent_params=params, **ctx))

    def emit(self, run, params):
        """Run ``run``, returning what it wrote to stdout."""
        v = self.grade(f"{run} > out.txt", params)
        self.assertEqual(v.state, PASS, v.blocked_reason or v.detail)
        return (self.ws / "out.txt").read_text().strip()

    def test_scalar_param_is_visible(self):
        self.assertEqual(
            self.emit('printf %s "$JAATO_EVAL_PARAM_ISSUE_ID"',
                      {"issue_id": "716"}),
            "716")

    def test_the_grader_follows_the_input(self):
        """The defect itself: the same manifest, a changed input, and a
        verdict that moves with it rather than staying behind."""
        run = '[ "$JAATO_EVAL_PARAM_ISSUE_ID" = "716" ]'
        self.assertEqual(self.grade(run, {"issue_id": "716"}).state, PASS)
        self.assertEqual(self.grade(run, {"issue_id": "715"}).state, FAIL)

    def test_key_is_upper_cased_and_shell_safe(self):
        """``issue-id`` has no environment name of its own; it gets one."""
        self.assertEqual(
            self.emit('printf %s "$JAATO_EVAL_PARAM_ISSUE_ID"',
                      {"issue-id": "716"}),
            "716")

    def test_non_scalar_params_are_json(self):
        """A dict has no obvious flat representation, so it gets the one
        a grader can parse back."""
        self.assertEqual(
            self.emit('printf %s "$JAATO_EVAL_PARAM_SPEC"',
                      {"spec": {"b": 2, "a": [1, "x"]}}),
            '{"a": [1, "x"], "b": 2}')

    def test_bool_is_the_json_spelling_not_pythons(self):
        """The author wrote ``true`` in YAML; ``True`` would be a value
        their shell comparison never matches."""
        self.assertEqual(
            self.emit('printf %s "$JAATO_EVAL_PARAM_STRICT"',
                      {"strict": True}),
            "true")

    def test_none_is_null_not_empty(self):
        """An explicit null must stay distinguishable from an absent key —
        empty string is what both would otherwise look like."""
        self.assertEqual(
            self.emit('printf %s "$JAATO_EVAL_PARAM_TARGET"',
                      {"target": None}),
            "null")

    def test_whole_mapping_is_exported_as_json(self):
        """The only way a shell can tell 'no such parameter' from 'the
        parameter is empty'."""
        self.assertEqual(
            self.emit('printf %s "$JAATO_EVAL_PARAMS"',
                      {"issue_id": "716", "repo": "org/thing"}),
            '{"issue_id": "716", "repo": "org/thing"}')

    def test_absent_param_leaves_the_variable_unset(self):
        """Not exported-and-empty: ``set -u`` is then a working guard for
        a grader that wants absence to be loud."""
        v = self.grade('[ -z "${JAATO_EVAL_PARAM_NOPE+set}" ]',
                       {"issue_id": "716"})
        self.assertEqual(v.state, PASS)

    def test_set_u_makes_a_missing_param_fail_loudly(self):
        """The documented convention, executed: a grader that opts in
        does not pass vacuously on an input the task never declared."""
        v = self.grade('set -u; printf %s "$JAATO_EVAL_PARAM_NOPE"', {})
        self.assertEqual(v.state, FAIL)

    def test_colliding_keys_are_blocked_not_arbitrated(self):
        """Two keys wanting one variable is a task defect.  Picking a
        winner would grade against an input the arm may not have run
        with — exactly the disagreement this export removes."""
        v = self.grade("true", {"issue-id": "715", "issue_id": "716"})
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("JAATO_EVAL_PARAM_ISSUE_ID", v.blocked_reason)
        self.assertIn("rename", v.blocked_reason)

    def test_params_do_not_displace_the_run_marker(self):
        v = self.grade('[ "$JAATO_EVAL" = "1" ]', {"issue_id": "716"})
        self.assertEqual(v.state, PASS)

    def test_unserializable_value_does_not_escape_the_grader(self):
        """A value JSON has no encoding for still reaches the shell as
        JSON — a quoted string — rather than raising into the sweep
        driver, which would take down an arm's other graders too."""
        self.assertEqual(
            self.emit('printf %s "$JAATO_EVAL_PARAM_WHERE"',
                      {"where": Path("/tmp/x")}),
            '"/tmp/x"')


PROC_PAYLOAD_ONLY = '''
def validate(payload, context):
    return [] if payload.get("ok") else ["not ok"]
'''

PROC_READS_LEDGER = '''
def validate(payload, context):
    return ["no calls"] if not context.tool_calls else []
'''

PROC_MENTIONS_LEDGER_IN_DOCSTRING = '''
"""This processor documents context.tool_calls but never reads it."""
def validate(payload, context):
    return []
'''


class ProcessorGraderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, source):
        p = self.root / name
        p.write_text(source)
        return p

    def grade(self, name, source, **ctx):
        self.write(name, source)
        spec = GraderSpec(kind="processor", config={"script": name})
        return ProcessorGrader(spec).grade(_context(self.root, **ctx))

    def test_payload_only_processor_passes(self):
        v = self.grade("p.py", PROC_PAYLOAD_ONLY, payload={"ok": True})
        self.assertEqual(v.state, PASS)

    def test_payload_only_processor_fails_on_errors(self):
        v = self.grade("p.py", PROC_PAYLOAD_ONLY, payload={"ok": False})
        self.assertEqual(v.state, FAIL)
        self.assertIn("not ok", v.evidence[0])

    def test_ledger_reader_blocked_when_ledger_unfaithful(self):
        """Grading on a best-effort pairing could credit a retry's success
        to the call that failed, so it must not be graded at all."""
        v = self.grade("p.py", PROC_READS_LEDGER, payload={"ok": True},
                       ledger=LedgerResult(entries=[], faithful=False,
                                           reason="call_id absent on the wire"))
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("call_id absent", v.blocked_reason)

    def test_ledger_reader_runs_when_ledger_faithful(self):
        v = self.grade("p.py", PROC_READS_LEDGER, payload={"ok": True},
                       ledger=LedgerResult(entries=[{"name": "x"}], faithful=True))
        self.assertEqual(v.state, PASS)

    def test_docstring_mention_does_not_trip_the_gate(self):
        """Substring matching would gate every processor, since they all
        document the contract. The gate matches attribute access."""
        self.assertFalse(_reads_ledger(PROC_MENTIONS_LEDGER_IN_DOCSTRING))
        v = self.grade("p.py", PROC_MENTIONS_LEDGER_IN_DOCSTRING, payload={},
                       ledger=LedgerResult(faithful=False, reason="r"))
        self.assertEqual(v.state, PASS)

    def test_unparseable_processor_gates_conservatively(self):
        self.assertTrue(_reads_ledger("def validate(:"))

    def test_missing_payload_is_blocked(self):
        v = self.grade("p.py", PROC_PAYLOAD_ONLY, payload=None)
        self.assertEqual(v.state, BLOCKED)

    def test_missing_processor_file_is_blocked(self):
        spec = GraderSpec(kind="processor", config={"script": "absent.py"})
        v = ProcessorGrader(spec).grade(_context(self.root, payload={}))
        self.assertEqual(v.state, BLOCKED)

    def test_raising_processor_is_blocked_not_failed(self):
        v = self.grade("p.py", "def validate(p, c):\n    raise RuntimeError('x')\n",
                       payload={})
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("RuntimeError", v.blocked_reason)


if __name__ == "__main__":
    unittest.main()
