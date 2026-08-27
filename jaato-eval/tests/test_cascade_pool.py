"""The per-task cascade pool — the second budget gate.

jaato has two independent budget gates and an eval sweep wants both:
the per-arm ceiling a task declares in its profile's ``budget_control``
(enforced by the daemon, no engine code), and this pool, shared by a
task's arms so that ``repeats: N`` cannot run away.

These pin the engine's half: one pool per budgeted task, none for a task
that declared none, and the cid reaching the arms that should draw on it.
"""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

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

    def _sweep(self, tasks, behaviour):
        _install_stub_sdk(behaviour)
        arms = build_matrix(tasks, [])
        store = ResultStore(self.root / "out.jsonl")
        return asyncio.run(run_sweep(
            arms, store=store, workspace_root=self.root / "ws", concurrency=1))

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

    def test_pooled_arms_register_as_observers_before_create(self):
        """Without this an arm sees no turns and no history at all.

        Measured live: a cid'd arm whose client did not register received
        only AgentStatusChangedEvent and AgentOutputEvent, and came back
        turns=0, tokens=0 with an empty ledger while its file was written.
        Observer, never owner — a cid admits one owner and N arms share one.
        """
        b = {"writes": "READY\n"}
        self._sweep([self._task("a", BUDGETED)], b)
        obs = b.get("observers", [])
        self.assertTrue(obs, "a pooled arm must observe its cid")
        for o in obs:
            self.assertEqual(o["role"], "observer")
            self.assertTrue(o["before_create"])
            self.assertIn("TurnCompletedEvent", o["event_types"])
            self.assertIn("HistoryEvent", o["event_types"])

    def test_an_unpooled_arm_registers_no_observer(self):
        b = {"writes": "READY\n"}
        self._sweep([self._task("p", UNBUDGETED)], b)
        self.assertEqual(b.get("observers", []), [])

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
