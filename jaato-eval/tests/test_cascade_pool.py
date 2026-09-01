"""The per-task cascade pool — the second budget gate.

jaato has two independent budget gates and an eval sweep wants both:
the per-arm ceiling a task declares in its profile's ``budget_control``
(enforced by the daemon, no engine code), and this pool, shared by a
task's arms so that ``repeats: N`` cannot run away.

These pin the engine's half: one pool per budgeted task, none for a task
that declared none, and the cid reaching the arms that should draw on it.

Two tests pinning a cascade-observer registration were REMOVED once
jaato #643 landed.  They asserted the engine called ``cascade_register``,
which was a workaround for a cid'd session receiving no TURN_COMPLETED at
its signal_completion terminus — an event-ordering bug, since fixed for
every consumer.  Retested on 9a4bf437: a pooled arm reports turns=1 with
no registration.  A test asserting an absent call pins an implementation
detail rather than a behaviour, so the reasoning lives in
``runner._ArmSession`` instead.
"""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jaato_eval.manifest import load_manifest
from jaato_eval.sweep import build_matrix, run_sweep
from jaato_eval.results import ResultStore

from tests.test_runner_integration import _install_stub_sdk, _STUBBED

BUDGETED = """
id: t/budgeted
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: write answer.txt containing READY
harness:
  profile: worker
budget:
  tokens: 12000
  usd: 0.5
  degrade:
    - at: 50%
      model_tiers:
        planner: {model: cheap/model, provider: openrouter}
    - at: 100%
      action: abort
graders:
  - kind: script
    run: "grep -qx READY answer.txt"
repeats: 2
"""

UNBUDGETED = BUDGETED.split("budget:")[0].replace("t/budgeted", "t/plain") + """graders:
  - kind: script
    run: "grep -qx READY answer.txt"
"""


class CascadePoolCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._displaced = {n: sys.modules[n] for n in _STUBBED if n in sys.modules}
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._uninstall)

    def _uninstall(self):
        for name in _STUBBED:
            sys.modules.pop(name, None)
        sys.modules.update(self._displaced)

    def _task(self, name, text):
        d = self.root / name
        (d / "fixture").mkdir(parents=True)
        (d / "cfg").mkdir()
        (d / "task.yaml").write_text(text)
        return load_manifest(d / "task.yaml")

    def _sweep(self, tasks, behaviour, **kw):
        _install_stub_sdk(behaviour)
        arms = build_matrix(tasks, [])
        store = ResultStore(self.root / "out.jsonl")
        return asyncio.run(run_sweep(
            arms, store=store, workspace_root=self.root / "ws",
            concurrency=1, **kw))

    def test_one_pool_per_budgeted_task_with_its_ladder(self):
        b = {"writes": "READY\n"}
        self._sweep([self._task("a", BUDGETED)], b)
        pools = b.get("pools", [])
        self.assertEqual(len(pools), 1, "one task, one pool")
        self.assertEqual(pools[0]["limits"], {"tokens": 12000.0, "usd": 0.5})
        self.assertEqual(len(pools[0]["degrade"]), 2, "the rung ladder travels")
        self.assertIn("t-budgeted", pools[0]["cid"], "cid names its task")

    def test_both_repeats_draw_on_the_same_pool(self):
        """A pool the repeats did not share would not bound the task."""
        b = {"writes": "READY\n"}
        results = self._sweep([self._task("a", BUDGETED)], b)
        self.assertEqual(len(results), 2)
        self.assertEqual(b["seen_kwargs"]["cascade_driver_id"],
                         b["pools"][0]["cid"])

    def test_every_subscription_precedes_create_session(self):
        """The refusal watch must be installed before create_session.

        A pool with no headroom announces the refusal WHILE create_session
        is in flight, so a handler installed afterwards never sees it and
        the arm waits out a thirty-second runner-readiness timeout instead,
        reporting a generic failure that names nothing.

        The other three are held here for consistency, not necessity:
        moving them after create was tried against a live daemon and
        changed nothing.  (An earlier version of this docstring blamed the
        ordering for a pooled arm's turns=0 — that was wrong; the observer
        registration is what fixes it, and the two were tested apart.)
        """
        b = {"writes": "READY\n"}
        self._sweep([self._task("a", BUDGETED)], b)
        self.assertGreaterEqual(
            b["subscribed_before_create"],
            {"TURN_COMPLETED", "SESSION_TERMINATED", "HISTORY", "ERROR"},
            "a subscription installed after create_session receives nothing")

    def test_each_arm_records_the_headroom_it_arrived_with(self):
        """jaato #777.  Three arms sharing one $6.00 pool spent
        $3.81 + $0.17 + $2.03 and the last was killed mid-work with
        ``budget_exhausted`` — which reads as a model failure until the row
        can show it arrived at a pool already 63% gone.  Read PER ARM,
        immediately before it starts: one reading taken up front would put
        the same number on every row."""
        b = {"writes": "READY\n"}
        results = self._sweep([self._task("a", BUDGETED)], b)
        self.assertEqual(len(b["budget_gets"]), len(results),
                         "one reading per arm, not one per sweep")
        for result in results:
            self.assertEqual(result.pool_on_arrival["usage_fraction"], 0.635)
            self.assertEqual(result.pool_on_arrival["cascade_driver_id"],
                             b["pools"][0]["cid"])

    def test_a_pool_that_never_answers_leaves_the_reading_absent(self):
        """Reporting must never fail an arm, and an unread pool must never
        render as an empty one."""
        b = {"writes": "READY\n", "pool_reply": None}
        # The production wait is deliberately generous; a suite that paid
        # it per arm would spend most of its runtime asleep.
        with patch("jaato_eval.pool.SNAPSHOT_TIMEOUT_SECONDS", 0.05):
            results = self._sweep([self._task("a", BUDGETED)], b)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsNone(result.pool_on_arrival)

    def test_an_unpooled_task_asks_for_no_reading(self):
        b = {"writes": "READY\n"}
        results = self._sweep([self._task("p", UNBUDGETED)], b)
        self.assertEqual(b.get("budget_gets", []), [])
        self.assertIsNone(results[0].pool_on_arrival)

    def test_a_task_with_no_budget_gets_no_pool_and_no_owner(self):
        """Opening an owner connection anyway taxes every sweep."""
        b = {"writes": "READY\n"}
        self._sweep([self._task("p", UNBUDGETED)], b)
        self.assertEqual(b.get("pools", []), [])
        self.assertEqual([c for c in b["clients"] if c.is_owner], [])
        self.assertNotIn("cascade_driver_id", b["seen_kwargs"])

    def test_the_owner_is_closed_when_the_sweep_ends(self):
        """A pool belongs to the connection that declared it.

        Leaking the owner leaks the pool; closing it early would take the
        pool out from under arms still running.
        """
        b = {"writes": "READY\n"}
        self._sweep([self._task("a", BUDGETED)], b)
        owners = [c for c in b["clients"] if c.is_owner]
        self.assertEqual(len(owners), 1, "exactly one owner declares the pools")
        self.assertTrue(owners[0].disconnected)


if __name__ == "__main__":
    unittest.main()


class ResumeSkipsRecordedArmsCase(CascadePoolCase):
    """End-to-end: --resume must not re-spend on an arm already recorded.

    The pieces were covered (completed_arm_ids, the parser flag); the
    WIRING was not, and a resume that silently re-ran everything would
    look identical to one that worked — the sweep completes either way.
    The witness is how many sessions the stub was asked to open.
    """

    def test_a_recorded_arm_is_not_rerun(self):
        task = self._task("a", BUDGETED)          # repeats: 2 -> two arms
        b = {"writes": "READY\n"}
        first = self._sweep([task], b)
        self.assertEqual(len(first), 2)
        opened_first = len(b["clients"])

        b2 = {"writes": "READY\n"}
        again = self._sweep([task], b2, resume=True)
        self.assertEqual(again, [], "both arms were already recorded")
        self.assertEqual([c for c in b2.get("clients", []) if not c.is_owner], [],
                         "resume opened a session for an arm it should have skipped")
        self.assertGreater(opened_first, 0)

    def test_without_resume_the_same_arms_run_again(self):
        """The contrast that makes the test above mean something."""
        task = self._task("a", BUDGETED)
        b = {"writes": "READY\n"}
        self._sweep([task], b)
        b2 = {"writes": "READY\n"}
        again = self._sweep([task], b2)
        self.assertEqual(len(again), 2)
