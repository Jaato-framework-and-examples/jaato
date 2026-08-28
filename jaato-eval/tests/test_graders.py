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
