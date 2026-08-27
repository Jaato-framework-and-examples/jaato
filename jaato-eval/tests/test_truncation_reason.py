"""``tool_use`` means two opposite things; these pin which is which.

Found by the first live run against a real daemon: a schema-driven arm
that wrote its file, signalled completion and produced a valid payload
was reported BLOCKED, because the terminal turn's finish_reason was
``"tool_use"`` — the same value a genuinely truncated arm carries.
"""
import unittest
from pathlib import Path

from jaato_eval.graders.base import GraderContext
from jaato_eval.graders.script import ScriptGrader
from jaato_eval.manifest import GraderSpec


def _ctx(**kw):
    return GraderContext(workspace_path=Path("."), config_root=Path("."), **kw)


class TruncationReasonCase(unittest.TestCase):

    def test_completion_payload_settles_it_despite_tool_use(self):
        """The regression itself: signal_completion ends the session, so a
        complete schema-driven arm's LAST turn is always 'tool_use'."""
        ctx = _ctx(finish_reason="tool_use", payload={"file_written": "a.txt"})
        self.assertIsNone(ctx.truncation_reason)

    def test_tool_use_without_a_payload_is_truncation(self):
        """Same value, opposite meaning: stopped to run tools, then nothing."""
        ctx = _ctx(finish_reason="tool_use", payload=None)
        reason = ctx.truncation_reason
        self.assertIsNotNone(reason)
        self.assertIn("tool_use", reason)

    def test_stop_without_a_payload_is_complete(self):
        """A prose profile declares no payload; its terminus is 'stop'."""
        self.assertIsNone(_ctx(finish_reason="stop", payload=None).truncation_reason)

    def test_max_tokens_is_truncation_and_names_itself(self):
        reason = _ctx(finish_reason="max_tokens", payload=None).truncation_reason
        self.assertIn("max_tokens", reason)

    def test_max_tokens_after_a_payload_is_not_truncation(self):
        """A payload is the declared terminus; nothing can follow it."""
        self.assertIsNone(
            _ctx(finish_reason="max_tokens", payload={"x": 1}).truncation_reason)


class ScriptGraderHonoursItCase(unittest.TestCase):
    """The grader must consume the predicate, not re-derive it.

    Without this the fix lives in a property nothing calls — the shape of
    the original defect, which was the rule copied into two graders.
    """

    def setUp(self):
        self.spec = GraderSpec(kind="script", config={"run": "true"})

    def test_schema_arm_is_graded_not_blocked(self):
        ctx = _ctx(finish_reason="tool_use", payload={"ok": True})
        verdict = ScriptGrader(self.spec).grade(ctx)
        self.assertEqual(verdict.state, "PASS", verdict.blocked_reason)

    def test_truncated_arm_is_blocked_with_a_reason(self):
        ctx = _ctx(finish_reason="max_tokens", payload=None)
        verdict = ScriptGrader(self.spec).grade(ctx)
        self.assertEqual(verdict.state, "BLOCKED")
        self.assertIn("max_tokens", verdict.blocked_reason)


if __name__ == "__main__":
    unittest.main()
