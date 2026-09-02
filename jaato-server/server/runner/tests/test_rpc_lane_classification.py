"""Every RPC method must be classified into a lane, deliberately.

The runner used to serve everything except ``session.bootstrap`` from ONE
8-worker pool.  ``session.send_message`` holds a worker for an entire turn and
``tool.execute`` is called with ``timeout=None``, while ``session.offer_message``
is a lock, a bool read and a list append -- so a control-plane RPC's latency
was bounded by the slowest work in the pool, and a delivery could be reported
``unreachable`` on a 2s timeout because an unrelated tool was still running.

The split fixes that.  This file exists so the split cannot rot: the
classification is a CRITERION ("does this handler run model or user code?"),
and a criterion only helps if someone applies it.  The test below makes
"nobody applied it" a build failure instead of a silent latency cliff -- the
enforcement sitting at zero distance from the thing it protects, rather than
in a convention a future author has to remember.
"""

from __future__ import annotations

import re
from pathlib import Path

from server.runner.rpc import MAIN_THREAD_METHODS, WORK_LANE_METHODS

_RPC_SOURCE = Path(__file__).resolve().parents[1] / "rpc.py"


def _served_methods() -> set[str]:
    """Every method string the dispatcher branches on."""
    text = _RPC_SOURCE.read_text(encoding="utf-8")
    return set(re.findall(r'env\.method == "([a-z_.]+)"', text))


def test_the_dispatcher_serves_something():
    """Guard the guard: a regex that matches nothing would pass everything."""
    served = _served_methods()
    assert len(served) > 20, (
        f"only {len(served)} methods found -- the extraction regex has "
        f"probably drifted from the dispatch style, which would make every "
        f"assertion below vacuous"
    )
    assert "session.send_message" in served
    assert "tool.execute" in served


def test_every_rpc_method_has_a_lane():
    """No method may reach a lane by accident.

    A new verb that runs model or user code and is NOT added to
    ``WORK_LANE_METHODS`` would fall through to the control lane and block it
    for every session on that runner -- which is the failure this split
    exists to prevent, reintroduced by omission.  So classification is
    mandatory rather than defaulted.
    """
    served = _served_methods()
    classified = WORK_LANE_METHODS | MAIN_THREAD_METHODS | _CONTROL_PLANE
    unclassified = sorted(served - classified)
    assert not unclassified, (
        "these RPC methods are in no lane:\n  "
        + "\n  ".join(unclassified)
        + "\n\nAsk: does the handler RUN MODEL OR USER CODE (calls the "
          "provider, replays the model loop, invokes a tool or a user "
          "command)?  If yes add it to WORK_LANE_METHODS in "
          "server/runner/rpc.py.  If no -- it reads or sets session state -- "
          "add it to _CONTROL_PLANE in this file."
    )


def test_the_lanes_do_not_overlap():
    assert not (WORK_LANE_METHODS & MAIN_THREAD_METHODS)
    assert not (WORK_LANE_METHODS & _CONTROL_PLANE)
    assert not (MAIN_THREAD_METHODS & _CONTROL_PLANE)


def test_no_lane_names_a_method_the_dispatcher_does_not_serve():
    """A stale entry is a lie about coverage."""
    served = _served_methods()
    for name, lane in (
        ("WORK_LANE_METHODS", WORK_LANE_METHODS),
        ("MAIN_THREAD_METHODS", MAIN_THREAD_METHODS),
        ("_CONTROL_PLANE", _CONTROL_PLANE),
    ):
        stale = sorted(lane - served)
        assert not stale, f"{name} names unserved method(s): {stale}"


#: The control plane, enumerated HERE rather than in ``rpc.py`` on purpose:
#: production needs only "is it work?", and duplicating the complement into
#: the module would create a second list to keep in sync.  The test owns it,
#: and ``test_no_lane_names_a_method_the_dispatcher_does_not_serve`` keeps it
#: honest in both directions.
#:
#: Every entry below reads or sets session state.  None runs model or user
#: code.  ``session.try_completion_nudge`` looks like it might -- it does not;
#: it is an atomic check-and-increment of a counter.
_CONTROL_PLANE = frozenset({
    "session.append_history_message",
    "session.apply_budget_degrade",
    "session.end",
    "session.get_all_session_state",
    "session.get_auth_info",
    "session.get_budget_exhausted",
    "session.get_budget_usage",
    "session.get_context_limit",
    "session.get_context_usage",
    "session.get_history",
    "session.get_model_completions",
    "session.get_rendered_system_instruction",
    "session.get_session_state",
    "session.get_tool_schemas",
    "session.get_turn_accounting",
    "session.get_user_commands",
    "session.health_check",
    "session.inject_prompt",
    "session.is_running",
    "session.offer_message",
    "session.register_client_tools",
    "session.request_stop",
    "session.reset",
    "session.resolve_fork_point",
    "session.restore_budget_usage",
    "session.restore_conversation_budget",
    "session.restore_turn_accounting",
    "session.set_initial_history",
    "session.set_parallel_tools_override",
    "session.set_presentation_context",
    "session.set_reference_authorizer",
    "session.set_session_state",
    "session.set_streaming_enabled",
    "session.set_terminal_width",
    "session.shutdown",
    "session.snapshot_conversation_budget",
    "session.snapshot_instruction_budget",
    "session.try_completion_nudge",
    "session.try_drain_pending_user",
    "subagent.forward_event",
})


def test_a_control_rpc_is_served_while_the_work_lane_is_full():
    """The behaviour the split buys, end to end over a real socketpair.

    Saturate the WORK lane with tool calls that never return, then issue a
    control-plane RPC.  Before the split it queued behind them and the caller
    timed out; now it is answered while they are still blocked.
    """
    import json
    import socket
    import threading

    from server.runner.envelope import RequestEnvelope
    from server.runner.rpc import RunnerRPC
    from shared.framing import read_frame_sync, write_frame_sync

    WORK_WORKERS = 2
    release = threading.Event()
    started = threading.Semaphore(0)

    def _blocking_tool(name, args, **kw):
        started.release()
        release.wait(timeout=10.0)      # occupies a work thread
        return True, {"ok": True}   # (ok, payload) -- the executor contract

    daemon_sock, runner_sock = socket.socketpair()
    rpc = RunnerRPC(
        runner_sock, _blocking_tool,
        max_workers=WORK_WORKERS, control_workers=2,
    )
    serve_thread = threading.Thread(target=rpc.serve, daemon=True)
    serve_thread.start()
    try:
        for i in range(WORK_WORKERS):
            write_frame_sync(daemon_sock, json.dumps(RequestEnvelope(
                id=i + 1, method="tool.execute",
                args={"name": "slow", "args": {}},
            ).to_dict()))
        for _ in range(WORK_WORKERS):
            assert started.acquire(timeout=5.0), "work lane never filled"

        # Work lane is now fully occupied and will stay so until `release`.
        write_frame_sync(daemon_sock, json.dumps(RequestEnvelope(
            id=99, method="session.health_check", args={},
        ).to_dict()))

        daemon_sock.settimeout(5.0)
        seen = []
        for _ in range(5):
            raw = read_frame_sync(daemon_sock)
            if raw is None:
                break
            frame = json.loads(raw)
            seen.append(frame.get("id"))
            if frame.get("id") == 99:
                break

        assert 99 in seen, (
            "the control RPC was not answered while the work lane was full "
            f"-- responses seen: {seen}.  That is the single-pool behaviour "
            f"this split removes."
        )
    finally:
        release.set()
        rpc.shutdown()
        daemon_sock.close()
        serve_thread.join(timeout=5.0)
