"""A plan the model created reaches the client on a runner-served session.

Reported from the web client: ``createPlan`` completed (the tool row
showed a tick) and the Plan panel kept saying *No plan yet*.  The TUI's
Ctrl+P panel is fed by the same events and was equally empty.

``todo`` is runner-tier, so on the default path the plugin reports into
the RUNNER's instance, whose reporter was the bootstrap's
``MemoryReporter`` -- stored, read by nobody.  The daemon's
``_setup_plan_hooks`` armed a ``LivePlanReporter`` on the DAEMON's
instance, which no runner-served session calls.  The same shape as the
description-callback gap (``test_description_callback_bridge_phase_4_4``),
closed the same way: the reporter is installed per turn on the runner and
its callbacks emit ``plan_*`` notification frames; the daemon demuxer
turns them into the plan events through the builders the in-process path
uses, so the two processes cannot disagree about a step's shape.
"""
from __future__ import annotations

from typing import Any, Dict, List

from jaato_sdk.events import (
    AgentOutputEvent,
    PlanClearedEvent,
    PlanStepUpdatedEvent,
    PlanUpdatedEvent,
)
from server.runner.rpc import RunnerRPC


# ------------------------------------------------------------ fakes

class _FakeTodo:
    def __init__(self) -> None:
        self._reporter = "memory-reporter"


class _FakeSubagent:
    def __init__(self) -> None:
        self._plan_reporter = None

    def set_plan_reporter(self, reporter) -> None:
        self._plan_reporter = reporter


def _session(todo=None, subagent=None):
    plugins = {"todo": todo, "subagent": subagent}

    class _Registry:
        def get_plugin(self, name):
            return plugins.get(name)

    class _Runtime:
        registry = _Registry()

    class _Session:
        _runtime = _Runtime()

    return _Session()


def _rpc(captured: List[Dict[str, Any]]) -> RunnerRPC:
    rpc = RunnerRPC.__new__(RunnerRPC)
    rpc.emit_notification = lambda **kw: captured.append(kw)
    return rpc


def _server():
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted_events = []
    srv.emit = lambda e: srv._emitted_events.append(e)
    srv._main_agent_id = "main"
    srv._agents = {}
    srv._cached_context_limit = 0
    srv._traces = []
    srv._trace = lambda m: srv._traces.append(m)
    return srv


# ------------------------------------------------------------ runner side

def test_install_swaps_the_todo_reporter_and_hands_it_to_the_subagent_plugin():
    todo, sub = _FakeTodo(), _FakeSubagent()
    rpc = _rpc([])
    originals = rpc._install_session_notification_callbacks(_session(todo, sub), request_id=1)
    assert todo._reporter != "memory-reporter"
    assert todo._reporter.name == "live_panel"
    assert sub._plan_reporter is todo._reporter
    assert originals["todo_reporter"] == "memory-reporter"
    assert originals["subagent_plan_reporter"] is None


def test_restore_puts_the_original_reporters_back():
    todo, sub = _FakeTodo(), _FakeSubagent()
    rpc = _rpc([])
    sess = _session(todo, sub)
    originals = rpc._install_session_notification_callbacks(sess, request_id=1)
    rpc._restore_session_notification_callbacks(sess, originals)
    assert todo._reporter == "memory-reporter"
    assert sub._plan_reporter is None


def test_a_session_without_a_todo_plugin_installs_nothing():
    rpc = _rpc([])
    originals = rpc._install_session_notification_callbacks(_session(None, None), request_id=1)
    assert "todo_reporter" not in originals
    assert "subagent_plan_reporter" not in originals


def _plan(**over):
    from jaato_sdk.plugins.todo.models import TodoPlan, TodoStep

    plan = TodoPlan(plan_id="p1", created_at="2026-09-16T00:00:00+00:00", title="Prueba",
                    steps=[TodoStep(step_id="s1", sequence=1, description="Crear el proyecto")])
    for k, v in over.items():
        setattr(plan, k, v)
    return plan


def test_the_reporter_emits_plan_frames_with_the_reporters_own_dicts():
    captured: List[Dict[str, Any]] = []
    todo = _FakeTodo()
    rpc = _rpc(captured)
    rpc._install_session_notification_callbacks(_session(todo), request_id=7)
    plan = _plan()
    todo._reporter.report_plan_created(plan, agent_id=None)
    kinds = [c["event_type"] for c in captured]
    assert kinds == ["plan_updated", "plan_output"]
    assert all(c["request_id"] == 7 for c in captured)
    upd = captured[0]["payload"]
    assert upd["agent_name"] is None
    assert upd["plan"]["title"] == "Prueba"
    assert upd["plan"]["steps"][0]["description"] == "Crear el proyecto"
    assert captured[1]["payload"] == {"source": "plan", "text": "Plan created: Prueba",
                                      "mode": "write", "agent_name": None}

    captured.clear()
    step = plan.steps[0]
    todo._reporter.report_step_update(plan, step, agent_id="worker")
    assert captured[0]["event_type"] == "plan_step_updated"
    assert captured[0]["payload"]["agent_name"] == "worker"
    assert captured[0]["payload"]["step"]["content"] == "Crear el proyecto"


def test_a_failing_emit_does_not_take_the_report_down():
    todo = _FakeTodo()
    rpc = RunnerRPC.__new__(RunnerRPC)

    def _boom(**kw):
        raise RuntimeError("channel closed")

    rpc.emit_notification = _boom
    rpc._install_session_notification_callbacks(_session(todo), request_id=1)
    todo._reporter.report_plan_created(_plan(), agent_id=None)   # must not raise


# ------------------------------------------------------------ daemon side

def test_demuxer_turns_plan_updated_into_the_event_the_in_process_path_emits():
    srv = _server()
    handler = srv._build_send_message_notification_handler()
    handler("plan_updated", {"plan": {
        "title": "Prueba",
        "steps": [{"step_id": "s1", "sequence": 1, "description": "Crear el proyecto",
                   "status": "pending", "result": None, "error": None}],
    }, "agent_name": None})
    evs = [e for e in srv._emitted_events if isinstance(e, PlanUpdatedEvent)]
    assert len(evs) == 1
    assert evs[0].agent_id == "main"
    assert evs[0].plan_name == "Prueba"
    assert evs[0].steps[0]["content"] == "Crear el proyecto"
    assert evs[0].steps[0]["step_id"] == "s1"
    # The snapshot carries the step's sequence too, so a client orders the
    # full plan the way it orders the deltas.
    assert evs[0].steps[0]["sequence"] == 1
    # Byte-for-byte the in-process builder's answer.
    direct = srv._plan_updated_event({"title": "Prueba", "steps": [{
        "step_id": "s1", "sequence": 1, "description": "Crear el proyecto",
        "status": "pending", "result": None, "error": None}]}, None)
    assert direct.steps == evs[0].steps and direct.plan_name == evs[0].plan_name


def test_demuxer_routes_the_other_three_plan_frames():
    srv = _server()
    handler = srv._build_send_message_notification_handler()
    handler("plan_step_updated", {"step": {"step_id": "s1", "sequence": 1, "content": "x", "status": "completed", "result": "ok"}, "agent_name": None})
    handler("plan_cleared", {"agent_name": None})
    handler("plan_output", {"source": "plan", "text": "Plan created: x", "mode": "write"})
    kinds = [type(e).__name__ for e in srv._emitted_events]
    assert kinds == ["PlanStepUpdatedEvent", "PlanClearedEvent", "AgentOutputEvent"]
    step = srv._emitted_events[0]
    assert isinstance(step, PlanStepUpdatedEvent) and step.status == "completed" and step.result == "ok"
    assert isinstance(srv._emitted_events[1], PlanClearedEvent)
    out = srv._emitted_events[2]
    assert isinstance(out, AgentOutputEvent) and out.source == "plan" and out.text == "Plan created: x"


def test_a_subagents_profile_name_is_mapped_to_its_agent_id():
    srv = _server()

    class _Agent:
        profile_name = "worker"

    srv._agents = {"sub-1": _Agent()}
    handler = srv._build_send_message_notification_handler()
    handler("plan_updated", {"plan": {"title": "t", "steps": []}, "agent_name": "worker"})
    assert srv._emitted_events[0].agent_id == "sub-1"


def test_a_malformed_frame_emits_a_default_event_not_a_crash():
    srv = _server()
    handler = srv._build_send_message_notification_handler()
    handler("plan_updated", {})
    ev = srv._emitted_events[0]
    assert isinstance(ev, PlanUpdatedEvent) and ev.plan_name == "Plan" and ev.steps == []


# --- the fourth frame carries the agent too ---------------------------------
#
# Three of the four plan frames carried ``agent_name``; ``plan_output`` did
# not, and ``_plan_output_event`` hardcoded ``_main_agent_id``.  So a
# SUBAGENT's plan PANEL was attributed correctly while its "Plan created:"
# / "[2] FAILED:" lines landed in the MAIN agent's transcript -- the two
# halves of one report disagreeing about whose work it was.  Only newly
# visible because this bridge is what makes the frames reach a client.


def test_a_subagents_plan_output_line_is_attributed_to_the_subagent():
    """The defect: this line used to land in the main agent's transcript."""
    srv = _server()

    class _Agent:
        profile_name = "worker"

    srv._agents = {"sub-1": _Agent()}
    handler = srv._build_send_message_notification_handler()
    handler("plan_output", {"source": "plan", "text": "Plan created: x",
                            "mode": "write", "agent_name": "worker"})
    out = srv._emitted_events[0]
    assert isinstance(out, AgentOutputEvent)
    assert out.agent_id == "sub-1"


def test_plan_output_without_an_agent_still_falls_back_to_main():
    """A reporter that names no agent keeps the old behaviour."""
    srv = _server()
    handler = srv._build_send_message_notification_handler()
    handler("plan_output", {"source": "plan", "text": "x", "mode": "write"})
    assert srv._emitted_events[0].agent_id == srv._main_agent_id


def test_every_plan_frame_the_reporter_emits_carries_an_agent():
    """The guard: all FOUR frames, which is what core.py's builder table claims.

    Asserted on the reporter's own emissions rather than on hand-built
    payloads, so a callback that drops the id on its way to the frame
    fails here even if the demuxer would have resolved it.
    """
    from jaato_sdk.plugins.todo.models import StepStatus

    captured: List[Dict[str, Any]] = []
    todo = _FakeTodo()
    rpc = _rpc(captured)
    rpc._install_session_notification_callbacks(_session(todo), request_id=7)
    plan = _plan()

    todo._reporter.report_plan_created(plan, agent_id="worker")
    plan.steps[0].status = StepStatus.FAILED
    plan.steps[0].error = "boom"
    todo._reporter.report_step_update(plan, plan.steps[0], agent_id="worker")

    kinds = [c["event_type"] for c in captured]
    assert "plan_output" in kinds, f"no plan_output among {kinds}"
    for c in captured:
        assert "agent_name" in c["payload"], (
            f"{c['event_type']} dropped the agent; all four plan frames "
            f"carry it, which is what core.py's builder table asserts"
        )
        assert c["payload"]["agent_name"] == "worker", c["event_type"]
