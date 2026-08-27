"""End-to-end arm execution against a stubbed SDK.

The daemon is not available in unit-test environments, but the runner's
contract with the SDK is small and stable: open a session, subscribe to
``TURN_COMPLETED`` and ``SESSION_TERMINATED``, call ``complete()``, call
``request_history()``, read ``HISTORY``.  Stubbing exactly that surface exercises the whole arm —
fixture materialisation, usage accumulation, ledger reconstruction,
grader dispatch, verdict roll-up — without a live model.

If the SDK's shape changes underneath these stubs, the real runner
breaks and these tests keep passing; that is the known limit of a stub.
What they do establish is that the runner's own logic is correct given
that contract, which is the part this package owns.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path

from jaato_eval.arm import ArmSpec
from jaato_eval.manifest import load_manifest
from jaato_eval.verdict import BLOCKED, FAIL, PASS

TASK = """
id: t/echo
environment:
  fixture: fixture
  config_root: cfg
input:
  prompt: write answer.txt containing READY
harness:
  profile: worker
graders:
  - kind: script
    run: "grep -qx READY answer.txt"
"""


class _Usage:
    def __init__(self, cost=0.01):
        self.prompt_tokens = 100
        self.output_tokens = 20
        self.spend_total_tokens = 120
        self.cost_usd = cost


class _TurnEvent:
    def __init__(self, finish_reason="stop", cost=0.01):
        self.finish_reason = finish_reason
        self.usage = _Usage(cost)


class _HistoryEvent:
    def __init__(self, history):
        self.history = history


class _TerminatedEvent:
    """Mirrors ``SessionTerminatedEvent``'s consumed surface.

    Every real session emits one.  ``reason="natural"`` is the ordinary
    wind-down; ``budget_exhausted`` / ``error`` are the two that name a
    stop the turn stream cannot report.
    """

    def __init__(self, reason="natural", details="", error_summary=None):
        self.reason = reason
        self.details = details
        self.error_summary = error_summary


class _FakeClient:
    def __init__(self, workspace, behaviour):
        self.workspace = Path(workspace)
        self.behaviour = behaviour
        self._handlers = {}

    def subscribe(self, event_type, handler):
        self._handlers.setdefault(event_type, []).append(handler)
        return lambda: None

    subscribe_once = subscribe

    def _emit(self, event_type, event):
        for h in self._handlers.get(event_type, []):
            h(event)

    async def request_history(self, agent_id="main"):
        self._emit("HISTORY", _HistoryEvent(self.behaviour.get("history", [])))


class _FakeSession:
    def __init__(self, client):
        self.client = client

    async def complete(self, prompt):
        b = self.client.behaviour
        if b.get("raise"):
            raise RuntimeError(b["raise"])
        if b.get("writes") is not None:
            (self.client.workspace / "answer.txt").write_text(b["writes"])
        self.client._emit("TURN_COMPLETED",
                          _TurnEvent(finish_reason=b.get("finish_reason", "stop")))
        # A real session always winds down with one of these; omitting it
        # would let the engine's SESSION_TERMINATED handling go untested
        # while every stubbed test still passed.
        self.client._emit("SESSION_TERMINATED", _TerminatedEvent(
            reason=b.get("termination_reason", "natural"),
            details=b.get("termination_detail", "")))
        return b.get("payload")


#: Module names the stub occupies.  Tracked so teardown restores exactly
#: what was displaced instead of wiping the namespace — another test may
#: have loaded the real completion_processors from the checkout.
_STUBBED = ("jaato_sdk", "jaato_sdk.client", "jaato_sdk.client.ipc",
            "jaato_sdk.events")


def _install_stub_sdk(behaviour):
    """Put a minimal jaato_sdk into sys.modules for the duration of a test."""
    class _Ctx:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            client = _FakeClient(self.kwargs["workspace_path"], behaviour)
            behaviour["seen_kwargs"] = self.kwargs
            return _FakeSession(client)

        async def __aexit__(self, *exc):
            return False

    class _IPCClient:
        @staticmethod
        def session(**kwargs):
            return _Ctx(kwargs)

    sdk = types.ModuleType("jaato_sdk")
    client_mod = types.ModuleType("jaato_sdk.client")
    ipc_mod = types.ModuleType("jaato_sdk.client.ipc")
    ipc_mod.IPCClient = _IPCClient
    events_mod = types.ModuleType("jaato_sdk.events")
    events_mod.EventType = types.SimpleNamespace(
        TURN_COMPLETED="TURN_COMPLETED", HISTORY="HISTORY",
        SESSION_TERMINATED="SESSION_TERMINATED")
    for name, mod in (("jaato_sdk", sdk), ("jaato_sdk.client", client_mod),
                      ("jaato_sdk.client.ipc", ipc_mod),
                      ("jaato_sdk.events", events_mod)):
        sys.modules[name] = mod


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "fixture").mkdir()
        (self.root / "cfg").mkdir()
        (self.root / "task.yaml").write_text(TASK)
        self.task = load_manifest(self.root / "task.yaml")
        self._displaced = {name: sys.modules[name] for name in _STUBBED
                           if name in sys.modules}
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._uninstall)

    def _uninstall(self):
        """Restore sys.modules to exactly what it was before the stub.

        Deleting every ``jaato_sdk*`` entry would also evict the real
        ``completion_processors`` that ``tests/_real_sdk`` loads from the
        checkout, making an unrelated test class pass or fail depending on
        collection order.
        """
        for name in _STUBBED:
            if name in sys.modules:
                del sys.modules[name]
        for name, module in self._displaced.items():
            sys.modules[name] = module

    def _run(self, behaviour, **kw):
        from jaato_eval.runner import run_arm
        _install_stub_sdk(behaviour)
        spec = ArmSpec(task=self.task, profile_set="cheap", repeat=0)
        return asyncio.run(run_arm(spec, workspace_root=self.root / "ws", **kw))

    def test_agent_does_the_work_arm_passes(self):
        result = self._run({"writes": "READY\n", "payload": {"done": True}})
        self.assertEqual(result.state, PASS)
        self.assertEqual(result.turns, 1)
        self.assertAlmostEqual(result.usage["cost_usd"], 0.01)
        self.assertEqual(result.usage["spend_total_tokens"], 120)
        self.assertIsNotNone(result.payload_hash)

    def test_agent_does_it_wrong_arm_fails(self):
        result = self._run({"writes": "not ready\n", "payload": {"done": True}})
        self.assertEqual(result.state, FAIL)

    def test_session_error_is_blocked_not_failed(self):
        """A daemon that refused the session says nothing about the model."""
        result = self._run({"raise": "daemon unreachable"})
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("daemon unreachable", result.blocked_reason)

    def test_truncated_turn_blocks_the_grader(self):
        result = self._run({"writes": "READY\n", "finish_reason": "max_tokens"})
        self.assertEqual(result.state, BLOCKED)
        self.assertEqual(result.finish_reason, "max_tokens")

    def test_budget_ceiling_blocks_the_arm_and_names_itself(self):
        """The ceiling stop must survive all the way to the verdict.

        The whole arm looks successful from the turn stream: the file is
        written and finish_reason is 'stop'.  Only SESSION_TERMINATED
        knows the session then refused further turns, so this is the test
        that fails if the engine ever stops subscribing to it.
        """
        result = self._run({
            "writes": "READY\n",
            "termination_reason": "budget_exhausted",
            "termination_detail": "self-enforced: tokens 1314%",
        })
        self.assertEqual(result.state, BLOCKED)
        reason = " ".join(v.blocked_reason for v in result.verdicts)
        self.assertIn("budget ceiling", reason)
        self.assertIn("1314%", reason)

    def test_ordinary_windown_does_not_block(self):
        """reason='natural' is every healthy session; it must stay silent."""
        result = self._run({"writes": "READY\n", "termination_reason": "natural"})
        self.assertEqual(result.state, PASS)

    def test_profile_set_reaches_the_env_file(self):
        """The sweep's model axis travels via .env in the workspace."""
        result = self._run({"writes": "READY\n"}, keep_workspace=True)
        ws = self.root / "ws" / result.spec.arm_id.replace("/", "_").replace("#", "_")
        self.assertIn("JAATO_PROFILE_SET=cheap", (ws / ".env").read_text())

    def test_config_root_is_sent_separately_from_workspace(self):
        """The agent must not be able to edit the config that governs it."""
        behaviour = {"writes": "READY\n"}
        self._run(behaviour)
        kwargs = behaviour["seen_kwargs"]
        self.assertNotEqual(kwargs["workspace_path"], kwargs["config_root"])
        self.assertTrue(kwargs["config_root"].endswith("cfg"))

    def test_workspace_discarded_by_default(self):
        result = self._run({"writes": "READY\n"})
        ws = self.root / "ws" / result.spec.arm_id.replace("/", "_").replace("#", "_")
        self.assertFalse(ws.exists())

    def test_repeats_produce_identical_hash_for_identical_payload(self):
        a = self._run({"writes": "READY\n", "payload": {"x": 1}})
        b = self._run({"writes": "READY\n", "payload": {"x": 1}})
        self.assertEqual(a.payload_hash, b.payload_hash)


if __name__ == "__main__":
    unittest.main()
