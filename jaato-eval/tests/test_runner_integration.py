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
import json
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
    def __init__(self, finish_reason="stop", cost=0.01, completion_gap=None):
        self.finish_reason = finish_reason
        self.usage = _Usage(cost)
        # jaato #654.  Rides EXACTLY ONE event and is read-and-cleared, so
        # the stub mirrors that: only the turn that carries it has it.
        self.completion_gap = completion_gap


class _HistoryEvent:
    def __init__(self, history):
        self.history = history


class _SessionInfoEvent:
    """Mirrors ``SessionInfoEvent``'s consumed surface.

    The daemon emits one WHILE ``create_session`` is still in flight and a
    second once the provider is ready, so the stub emits it from inside
    create: a stub that emitted it afterwards would let a handler
    subscribed too late still pass, which is the exact ordering bug the
    real subscription order exists to avoid.
    """

    def __init__(self, model_name="", model_provider=""):
        self.model_name = model_name
        self.model_provider = model_provider


class _SystemMessageEvent:
    """Mirrors ``SystemMessageEvent`` — how ``cascade.budget.get`` answers.

    The pool reading is not a return value: the daemon replies on the
    event stream with JSON in ``message``, which is why the engine has to
    latch it by shape.
    """

    def __init__(self, message=""):
        self.message = message


class _TerminatedEvent:
    """Mirrors ``SessionTerminatedEvent``'s consumed surface.

    Every real session emits one.  ``reason="natural"`` is the ordinary
    wind-down; ``budget_exhausted`` / ``error`` are the two that name a
    stop the turn stream cannot report.
    """

    def __init__(self, reason="natural", details="", error_summary=None,
                 error_type=None):
        self.reason = reason
        self.details = details
        self.error_summary = error_summary
        # The terminal's TYPE.  ``reason="error"`` alone cannot tell a
        # daemon that died mid-turn from an agent that finished and never
        # signed off, and the engine sorts those into opposite states.
        self.error_type = error_type


class _ErrorEvent:
    """Mirrors ``ErrorEvent``'s consumed surface.

    A cascade pool refusing a spawn arrives this way and ONLY this way:
    ``create_session`` still hands back a session id for the refused
    session, so a stub that only made create raise would test a path the
    daemon does not take.
    """

    def __init__(self, error_type, error=""):
        self.error_type = error_type
        self.error = error


class _AgentError(Exception):
    """Mirrors ``convenience.AgentError``'s consumed surface.

    ``Session.complete`` raises this on an error terminal, AFTER the
    ``SESSION_TERMINATED`` handlers have run — the ordering matters, since
    the engine reads the terminal's type off either source.
    """

    def __init__(self, error_type, error_summary):
        self.error_type = error_type
        self.error_summary = error_summary
        super().__init__(f"{error_type}: {error_summary}")


class _FakeClient:
    """Stands in for ``IPCClient`` in both roles the engine uses.

    Arm side: connect -> subscribe -> create_session -> Session.  Owner
    side: connect -> cascade_budget_set.  One class, as in the real SDK.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.behaviour = _BEHAVIOUR[0]
        self.workspace = Path(kwargs["workspace_path"]) if kwargs.get(
            "workspace_path") else None
        self._handlers = {}
        self.is_owner = False
        self.created = False
        self.disconnected = False
        self.behaviour.setdefault("clients", []).append(self)

    async def connect(self, timeout=None):
        self.behaviour.setdefault("connected", []).append(timeout)
        return True

    async def disconnect(self):
        self.disconnected = True

    async def cascade_register(self, cid, role="observer", event_types=None):
        """Arms must observe their cid or the daemon sends them nothing.

        A cid'd session's events fan out to the cid's registered
        cascade-clients rather than to the connection that created it.
        Recorded so a test can hold that this happens, and BEFORE create.
        """
        self.behaviour.setdefault("observers", []).append(
            {"cid": cid, "role": role,
             "event_types": [getattr(e, "__name__", e) for e in (event_types or [])],
             # Per-CLIENT, not per-behaviour: the behaviour dict is shared
             # across a sweep's arms, so a global flag would read False for
             # every arm after the first and quietly stop testing anything.
             "before_create": not self.created})

    async def cascade_budget_set(self, cid, limits, degrade=None):
        # Marks this client as the pool owner, so a test can assert the
        # OWNER was closed rather than merely that something was.
        self.is_owner = True
        self.behaviour.setdefault("pools", []).append(
            {"cid": cid, "limits": limits, "degrade": degrade})

    async def cascade_budget_get(self, cid):
        """Answer on the EVENT STREAM, as the daemon does.

        ``pool_reply`` lets a test shape the answer; absent, the reply is a
        plausible half-consumed pool, so every sweep test exercises the
        snapshot path rather than only the "no reply" fallback.
        """
        self.behaviour.setdefault("budget_gets", []).append(cid)
        reply = self.behaviour.get("pool_reply", {
            "declared": True, "limits": {"usd": 6.0},
            "remaining": {"usd": 2.19}, "usage_fraction": 0.635,
            "pressure": "63% of usd",
        })
        if reply is None:
            return
        self._emit("SYSTEM_MESSAGE", _SystemMessageEvent(
            message=json.dumps({"cascade_driver_id": cid, **reply})))

    async def create_session(self, **create_kwargs):
        self.behaviour["seen_kwargs"] = {**self.kwargs, **create_kwargs}
        # Snapshot WHICH events were already subscribed.  With a
        # cascade_driver_id the daemon delivers nothing to a client that
        # subscribes after this point, so the order is load-bearing and a
        # set captured here is the only way a stub can hold it.
        self.behaviour["subscribed_before_create"] = set(self._handlers)
        self.created = True
        # The binding, announced the way the daemon announces it: from
        # inside create, before the sid is returned.
        self._emit("SESSION_INFO", _SessionInfoEvent(
            model_name=self.behaviour.get("model_name", "openai/gpt-5-mini"),
            model_provider=self.behaviour.get("model_provider", "openrouter")))
        refuse = self.behaviour.get("refuse_spawn")
        if refuse:
            # Dispatched to handlers BEFORE the sid is returned — the
            # ordering the SDK pins, and the reason a handler subscribed
            # before create can latch the refusal at all.
            self._emit("ERROR", _ErrorEvent(
                refuse, "cascade has no headroom left on tokens"))
        if self.behaviour.get("raise"):
            raise RuntimeError(self.behaviour["raise"])
        return "sid-1"

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
    """Stands in for ``convenience.Session`` — owns ``complete()``."""

    def __init__(self, client, session_id=None, on_permission=None):
        self.client = client
        self.session_id = session_id

    async def complete(self, prompt):
        b = self.client.behaviour
        if b.get("hang"):
            await asyncio.sleep(b["hang"])
        if b.get("writes") is not None:
            (self.client.workspace / "answer.txt").write_text(b["writes"])
        if b.get("session_log") is not None:
            # Where the daemon actually puts it: JAATO_SESSION_LOG_DIR
            # defaults to `.jaato/logs`, resolved against the WORKSPACE, and
            # the handler names the file after the session and the client.
            logs = self.client.workspace / ".jaato" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"session_{self.session_id}_client_c1.log").write_text(
                b["session_log"])
        for i in range(int(b.get("turns", 1))):
            self.client._emit("TURN_COMPLETED", _TurnEvent(
                finish_reason=b.get("finish_reason", "stop"),
                # carried by ONE turn only, and deliberately not the last,
                # so a driver sampling the final event would miss it
                completion_gap=(b.get("completion_gap") if i == 0 else None)))
        # A real session always winds down with one of these; omitting it
        # would let the engine's SESSION_TERMINATED handling go untested
        # while every stubbed test still passed.
        error_type = b.get("agent_error")
        self.client._emit("SESSION_TERMINATED", _TerminatedEvent(
            reason=("error" if error_type
                    else b.get("termination_reason", "natural")),
            details=b.get("termination_detail", ""),
            error_type=error_type))
        # An error terminal reaches the caller as an exception, not as a
        # return: the terminal event fires first, then ``complete()``
        # raises.  A stub that only emitted the event would let the engine
        # walk on into grading and test a path no real session takes.
        if error_type:
            raise _AgentError(error_type, b.get("termination_detail", ""))
        return b.get("payload")


#: The behaviour dict the fakes read.  A one-slot box rather than a
#: constructor argument, because the engine now builds its own clients and
#: the stub cannot hand them anything.
_BEHAVIOUR = [{}]


#: Module names the stub occupies.  Tracked so teardown restores exactly
#: what was displaced instead of wiping the namespace — another test may
#: have loaded the real completion_processors from the checkout.
_STUBBED = ("jaato_sdk", "jaato_sdk.client", "jaato_sdk.client.ipc",
            "jaato_sdk.client.convenience", "jaato_sdk.events")


def _install_stub_sdk(behaviour):
    """Put a minimal jaato_sdk into sys.modules for the duration of a test."""
    _BEHAVIOUR[0] = behaviour

    sdk = types.ModuleType("jaato_sdk")
    client_mod = types.ModuleType("jaato_sdk.client")
    ipc_mod = types.ModuleType("jaato_sdk.client.ipc")
    ipc_mod.IPCClient = _FakeClient
    conv_mod = types.ModuleType("jaato_sdk.client.convenience")
    conv_mod.Session = _FakeSession
    conv_mod.AgentError = _AgentError
    events_mod = types.ModuleType("jaato_sdk.events")
    events_mod.EventType = types.SimpleNamespace(
        TURN_COMPLETED="TURN_COMPLETED", HISTORY="HISTORY",
        SESSION_TERMINATED="SESSION_TERMINATED", ERROR="ERROR",
        AGENT_ERROR="AGENT_ERROR",
        # The two the per-arm report needs: the binding (jaato #777's join
        # key, model and provider) and the pool reading.
        SESSION_INFO="SESSION_INFO", SYSTEM_MESSAGE="SYSTEM_MESSAGE")
    events_mod.ClientType = types.SimpleNamespace(API="API")
    # The event CLASSES the engine names when registering as a cascade
    # observer.  cascade_register filters on type-name, so the stub only
    # needs objects whose __name__ matches.
    for cls_name in ("TurnCompletedEvent", "HistoryEvent",
                     "SessionTerminatedEvent", "ErrorEvent"):
        setattr(events_mod, cls_name, type(cls_name, (), {}))
    for name, mod in (("jaato_sdk", sdk), ("jaato_sdk.client", client_mod),
                      ("jaato_sdk.client.ipc", ipc_mod),
                      ("jaato_sdk.client.convenience", conv_mod),
                      ("jaato_sdk.events", events_mod)):
        sys.modules[name] = mod


class RunnerHarness(unittest.TestCase):
    """The stub rig, with no assertions of its own.

    Split from :class:`RunnerCase` so a sibling suite can drive an arm
    through the same fakes without also inheriting — and re-running, under
    its own manifest — every test declared here.  Carries no ``test_``
    methods deliberately: it is a fixture, and anything asserted on it
    would run twice for each subclass.
    """

    #: The manifest each arm runs.  Overridden by a subclass that needs a
    #: different grader set (``setUp`` reads it after writing the file).
    task_yaml = TASK

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "fixture").mkdir()
        (self.root / "cfg").mkdir()
        (self.root / "task.yaml").write_text(self.task_yaml)
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
        self.behaviour = behaviour
        _install_stub_sdk(behaviour)
        import jaato_eval.runner as _r
        _orig = _r._CONTEXT_SPY
        _r._CONTEXT_SPY = lambda c: behaviour.__setitem__("graded_context", c)
        self.addCleanup(lambda: setattr(_r, "_CONTEXT_SPY", _orig))
        spec = ArmSpec(task=self.task, profile_set="cheap", repeat=0)
        return asyncio.run(run_arm(spec, workspace_root=self.root / "ws", **kw))

class RunnerCase(RunnerHarness):
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

    def test_pool_refusal_names_the_pool_not_a_daemon_fault(self):
        """An exhausted pool and a broken daemon are opposite calls to action.

        Both arrive as an exception out of session creation; only
        ``error_type`` separates "the ceiling I declared did its job" from
        "go look at the daemon".
        """
        result = self._run({"refuse_spawn": "CascadeExhaustedError"})
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("cascade budget pool is exhausted", result.blocked_reason)

    def test_refusal_of_an_unstated_type_is_not_given_one(self):
        """A refusal the daemon did not type must not be guessed at."""
        result = self._run({"raise": "daemon unreachable"})
        self.assertEqual(result.state, BLOCKED)
        self.assertNotIn("cascade", result.blocked_reason.lower())

    def test_cascade_id_reaches_the_session(self):
        result = self._run({"writes": "READY\n"}, cascade_driver_id="cid-42")
        self.assertEqual(result.state, PASS)
        self.assertEqual(self.behaviour["seen_kwargs"]["cascade_driver_id"],
                         "cid-42")

    def test_no_cascade_id_means_no_kwarg(self):
        """An un-pooled arm must not send an empty cid the daemon would read."""
        self._run({"writes": "READY\n"})
        self.assertNotIn("cascade_driver_id", self.behaviour["seen_kwargs"])

    def test_an_arm_that_never_finishes_is_blocked_not_hung(self):
        """A benchmark must bound its own arms.

        A task pool's `seconds` cannot: it is reconciled when a session
        ENDS, so a session that never ends never consumes it and the pool
        cannot abort it.  Measured twice against a live daemon — a slow
        model kept turning past sixteen minutes and each sweep died on the
        operator's own `timeout`, losing the report and one arm with it.
        """
        result = self._run({"hang": 30}, arm_timeout_seconds=0.2)
        self.assertEqual(result.state, BLOCKED)
        self.assertIn("harness ceiling", result.blocked_reason)

    def test_a_timed_out_arm_is_blocked_never_failed(self):
        """It was cut short; that says nothing about the configuration."""
        result = self._run({"hang": 30, "writes": "not ready\n"},
                           arm_timeout_seconds=0.2)
        self.assertEqual(result.state, BLOCKED)
        self.assertEqual(result.verdicts, [])

    def test_zero_disables_the_ceiling(self):
        result = self._run({"writes": "READY\n"}, arm_timeout_seconds=0)
        self.assertEqual(result.state, PASS)

    def test_completion_gap_is_latched_not_sampled_from_the_last_turn(self):
        """It rides EXACTLY ONE event and is read-and-cleared.

        The stub puts it on the FIRST of three turns, so a driver that
        reads the field off the final TurnCompletedEvent sees None and
        reports the arm as an unexplained empty payload — which is the
        state jaato #654 exists to end.
        """
        result = self._run({"writes": "READY\n", "turns": 3,
                            "completion_gap": "not_signalled_after_nudges"})
        self.assertEqual(result.turns, 3)
        ctx = self.behaviour["graded_context"]
        self.assertEqual(ctx.completion_gap, "not_signalled_after_nudges")

    def test_no_gap_on_a_healthy_arm(self):
        """A legitimately multi-turn session must never set it."""
        self._run({"writes": "READY\n", "turns": 3})
        self.assertIsNone(self.behaviour["graded_context"].completion_gap)

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
