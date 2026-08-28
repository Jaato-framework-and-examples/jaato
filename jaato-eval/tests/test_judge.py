"""The judge grader — the third grader kind, and the only one that spends.

Every other adapter reads what the arm left behind; this one opens a
SECOND session per arm and asks it to score the work.  That makes its
guards the load-bearing part: an unguarded judge spends a model call to
score a run that was truncated, or already known bad, or that produced
nothing to score.

The SDK surface it touches is two calls — ``IPCClient.session(**kwargs)``
and ``session.complete(prompt)`` — so it gets a focused stub here rather
than widening the runner's.
"""
import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path

from jaato_eval.graders.base import GraderContext
from jaato_eval.graders.judge import JudgeGrader, _render_prompt
from jaato_eval.manifest import GraderSpec
from jaato_eval.verdict import BLOCKED, FAIL, PASS

_STUBBED = ("jaato_sdk", "jaato_sdk.client", "jaato_sdk.client.ipc")


def _install(payload=None, raises=None, seen=None):
    """Minimal jaato_sdk exposing only what the judge calls."""
    class _Session:
        def __init__(self, client):
            self.client = client

        async def complete(self, prompt):
            if seen is not None:
                seen["prompt"] = prompt
            if raises is not None:
                raise raises
            return payload

    class _Ctx:
        def __init__(self, kwargs):
            if seen is not None:
                seen["kwargs"] = kwargs

        async def __aenter__(self):
            return _Session(self)

        async def __aexit__(self, *exc):
            return False

    class _IPCClient:
        @staticmethod
        def session(**kwargs):
            return _Ctx(kwargs)

    sdk = types.ModuleType("jaato_sdk")
    client = types.ModuleType("jaato_sdk.client")
    ipc = types.ModuleType("jaato_sdk.client.ipc")
    ipc.IPCClient = _IPCClient
    for name, mod in (("jaato_sdk", sdk), ("jaato_sdk.client", client),
                      ("jaato_sdk.client.ipc", ipc)):
        sys.modules[name] = mod


class JudgeCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "answer.txt").write_text("READY\n")
        self._displaced = {n: sys.modules[n] for n in _STUBBED if n in sys.modules}
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._uninstall)

    def _uninstall(self):
        for n in _STUBBED:
            sys.modules.pop(n, None)
        sys.modules.update(self._displaced)

    def _ctx(self, **kw):
        kw.setdefault("payload", {"done": True})
        return GraderContext(workspace_path=self.ws, config_root=self.ws, **kw)

    def _grade(self, config, ctx=None, **stub):
        _install(**stub)
        return JudgeGrader(GraderSpec(kind="judge", config=config)).grade(ctx or self._ctx())

    # ---- the guards that exist to avoid spending -------------------------

    def test_no_profile_is_blocked(self):
        v = self._grade({})
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("profile", v.blocked_reason)

    def test_truncated_arm_is_not_judged(self):
        """Scoring a truncated run scores the interruption, not the work."""
        v = self._grade({"profile": "rubric"},
                        self._ctx(payload=None, finish_reason="max_tokens"))
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("max_tokens", v.blocked_reason)

    def test_unmet_gate_skips_the_judge(self):
        v = self._grade({"profile": "rubric", "gate_on": ["script:x"]},
                        self._ctx(prior_verdicts={"script:x": FAIL}))
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("script:x=FAIL", v.blocked_reason)

    def test_a_gate_that_never_ran_counts_as_unmet(self):
        """Otherwise a manifest ordering mistake silently disables the gate."""
        v = self._grade({"profile": "rubric", "gate_on": "script:missing"},
                        self._ctx(prior_verdicts={}))
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("not run", v.blocked_reason)

    def test_met_gate_lets_the_judge_run(self):
        v = self._grade({"profile": "rubric", "gate_on": ["script:x"]},
                        self._ctx(prior_verdicts={"script:x": PASS}),
                        payload={"score": 0.9})
        self.assertEqual(v.state, PASS)

    # ---- what comes back -------------------------------------------------

    def test_score_at_or_above_threshold_passes(self):
        """At the threshold, not merely above it — >= is the documented rule."""
        v = self._grade({"profile": "rubric", "threshold": 0.7}, payload={"score": 0.7})
        self.assertEqual(v.state, PASS)
        self.assertIn("0.700", v.detail)

    def test_score_below_threshold_fails(self):
        v = self._grade({"profile": "rubric", "threshold": 0.7}, payload={"score": 0.69})
        self.assertEqual(v.state, FAIL)

    def test_a_judge_that_returned_nothing_is_blocked_not_failed(self):
        """No payload means the judge did not judge — that is not a low score."""
        v = self._grade({"profile": "rubric"}, payload=None)
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("completion_payload_schema", v.blocked_reason)

    def test_missing_score_field_is_blocked_and_names_the_keys(self):
        v = self._grade({"profile": "rubric"}, payload={"verdict": "good"})
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("verdict", v.blocked_reason)

    def test_non_numeric_score_is_blocked_not_failed(self):
        v = self._grade({"profile": "rubric"}, payload={"score": "excellent"})
        self.assertEqual(v.state, BLOCKED)

    def test_custom_score_field(self):
        v = self._grade({"profile": "rubric", "score_field": "rating", "threshold": 3},
                        payload={"rating": 4})
        self.assertEqual(v.state, PASS)

    def test_session_failure_is_blocked_not_failed(self):
        """A judge that errored graded nothing; that says nothing about the arm."""
        v = self._grade({"profile": "rubric"}, raises=RuntimeError("daemon gone"))
        self.assertEqual(v.state, BLOCKED)
        self.assertIn("daemon gone", v.blocked_reason)

    def test_reasoning_is_attached_as_evidence(self):
        v = self._grade({"profile": "rubric"},
                        payload={"score": 0.9, "reasoning": "wrote the file"})
        self.assertTrue(any("wrote the file" in e for e in v.evidence))

    # ---- what the judge is told ------------------------------------------

    def test_the_judge_receives_the_payload_and_the_listing(self):
        seen = {}
        self._grade({"profile": "rubric"}, self._ctx(payload={"file_written": "answer.txt"}),
                    payload={"score": 1.0}, seen=seen)
        self.assertIn("answer.txt", seen["prompt"])
        self.assertIn("file_written", seen["prompt"])

    def test_the_judge_session_gets_workspace_and_config_root(self):
        seen = {}
        self._grade({"profile": "rubric"}, payload={"score": 1.0}, seen=seen)
        self.assertEqual(seen["kwargs"]["profile"], "rubric")
        self.assertEqual(seen["kwargs"]["workspace_path"], str(self.ws))
        self.assertEqual(seen["kwargs"]["config_root"], str(self.ws))


if __name__ == "__main__":
    unittest.main()


class JudgeUsesTheRunsDaemonCase(unittest.TestCase):
    """The judge must score on the daemon the ARM ran on.

    Found by exercising the judge for the first time: it read socket_path
    from its manifest config only, so a sweep pointed at a non-default
    daemon would have had its arms on one and its judges on another —
    silently, since the client default resolves to a real socket.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self._displaced = {n: sys.modules[n] for n in _STUBBED if n in sys.modules}
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._uninstall)

    def _uninstall(self):
        for n in _STUBBED:
            sys.modules.pop(n, None)
        sys.modules.update(self._displaced)

    def _run(self, config, **ctx_kw):
        seen = {}
        _install(payload={"score": 1.0}, seen=seen)
        ctx = GraderContext(workspace_path=self.ws, config_root=self.ws,
                            payload={"ok": True}, **ctx_kw)
        JudgeGrader(GraderSpec(kind="judge", config=config)).grade(ctx)
        return seen["kwargs"]

    def test_the_runs_socket_reaches_the_judge(self):
        k = self._run({"profile": "rubric"}, socket_path="/tmp/run.sock")
        self.assertEqual(k["socket_path"], "/tmp/run.sock")

    def test_the_runs_socket_wins_over_a_manifest_override(self):
        k = self._run({"profile": "rubric", "socket_path": "/tmp/manifest.sock"},
                      socket_path="/tmp/run.sock")
        self.assertEqual(k["socket_path"], "/tmp/run.sock")

    def test_manifest_override_still_works_with_no_run_socket(self):
        k = self._run({"profile": "rubric", "socket_path": "/tmp/manifest.sock"})
        self.assertEqual(k["socket_path"], "/tmp/manifest.sock")

    def test_no_socket_anywhere_leaves_the_client_default(self):
        k = self._run({"profile": "rubric"})
        self.assertNotIn("socket_path", k)
