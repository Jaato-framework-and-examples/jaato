"""#988 — a cancel frame that arrives before the worker starts.

``test_cancel_token_trips_runner_cancel`` (``server/test_runner_rpc_client.py``)
asserts the same guarantee end to end through a real runner subprocess, and it
is a **timing bet**: it trips a token 0.2 s in and hopes the worker thread won
a race that a loaded host loses.  Measured on a 4-core machine carrying 16 busy
loops, 12 of 12 standalone runs lost the cancel and reported ``ok=True`` after
running the command to completion; idle, 10 of 10 passed.  That is how the
defect wore "flaky test" for a day.

**This module removes the bet.**  The loaded case is reproduced by
*construction* rather than by wall clock: the dispatcher is built with a
single work-lane worker, that worker is occupied by a call which blocks until
the test releases it, and the cancelled call is therefore PROVABLY still
queued — its ``_handle_request`` has not executed a line.  There is no sleep
anywhere and no timing assumption; the only waits are on
:class:`threading.Event` objects tied to observable state, and on frames the
runner writes.

The frame ordering is asserted through the wire rather than by polling: the
serve loop decodes frames one at a time on one thread, so a response to a
**later** request is proof that the earlier cancel frame has already been
handled.  The probe request is a control-lane method, which has its own pool
and so is not blocked behind the occupied work-lane worker.

What fails here without the fix (registration inside ``_handle_request``):

* ``cancel_stats()['unknown'] == 1`` instead of ``tripped == 1``
* the queued call runs to completion and answers ``ok=True``
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest

from shared.framing import read_frame_sync, write_frame_sync

from server.runner.envelope import (
    KIND_RESPONSE,
    KIND_STREAM,
    CancelFrame,
    RequestEnvelope,
    ResponseEnvelope,
)
from server.runner.rpc import RunnerRPC, get_current_cancel_token


# A generous ceiling on "the other thread is not going to do this at all".
# It bounds a hang; no assertion depends on anything happening *within* it.
JOIN_TIMEOUT = 10.0


class _GatedExecutor:
    """Work-lane executor whose calls block until the test releases them.

    Holding the single work-lane worker is what reproduces #988's loaded
    case deterministically: while ``hold`` is unset the pool has no free
    worker, so a second request is queued and its ``_handle_request``
    provably has not run.

    Attributes:
        started: set the first time a call enters the executor.  The test
            waits on it to know the worker is occupied.
        hold: cleared at construction; every call blocks on it.  The test
            sets it to let the queued call proceed.
        cancelled_ids: request ids (in arrival order) whose cancel token
            was already tripped when the executor looked.  This is the
            executor's own view of the guarantee — the token has to be
            visible through the thread-local, not merely tripped in the
            dispatcher's table.
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.hold = threading.Event()
        self.cancelled_ids: List[str] = []
        self._lock = threading.Lock()

    def __call__(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        tag = str(args.get("tag") or name)
        self.started.set()
        self.hold.wait(JOIN_TIMEOUT)
        token = get_current_cancel_token()
        if token is not None and token.is_cancelled:
            with self._lock:
                self.cancelled_ids.append(tag)
            return False, {"error": "cancelled", "cancelled": True}
        return True, {"tag": tag}


@pytest.fixture
def single_worker_rpc():
    """``(daemon_sock, rpc, executor)`` with a ONE-worker dispatcher serving.

    One work-lane worker is the whole point: it makes "the worker has not
    registered the call yet" a structural fact rather than a race the test
    hopes to hit.
    """
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )
    executor = _GatedExecutor()
    rpc = RunnerRPC(runner_sock, executor, max_workers=1, control_workers=2)
    thread = threading.Thread(
        target=rpc.serve, name="rpc-serve-988", daemon=True,
    )
    thread.start()
    try:
        yield daemon_sock, rpc, executor
    finally:
        executor.hold.set()
        rpc.shutdown()
        try:
            daemon_sock.close()
        except OSError:
            pass
        thread.join(timeout=JOIN_TIMEOUT)


def _send(sock: socket.socket, payload: Dict[str, Any]) -> None:
    write_frame_sync(sock, json.dumps(payload))


def _request(request_id: int, tag: str) -> Dict[str, Any]:
    return RequestEnvelope(
        id=request_id,
        method="tool.execute",
        args={"name": "gated", "args": {"tag": tag}},
    ).to_dict()


def _read_response(
    sock: socket.socket, request_id: int,
) -> ResponseEnvelope:
    """Read frames until the terminating response for *request_id*.

    Stream frames and responses for other ids are skipped.  A closed
    channel is an assertion failure rather than a silent ``None``.
    """
    sock.settimeout(JOIN_TIMEOUT)
    while True:
        raw = read_frame_sync(sock)
        if raw is None:
            raise AssertionError(
                f"runner closed the channel before answering id={request_id}"
            )
        payload = json.loads(raw)
        if payload.get("kind") == KIND_STREAM:
            continue
        if payload.get("kind") != KIND_RESPONSE:
            continue
        if payload.get("id") != request_id:
            continue
        return ResponseEnvelope.from_dict(payload)


def _barrier_through_reader(
    daemon_sock: socket.socket, rpc: RunnerRPC, request_id: int,
) -> None:
    """Block until the serve loop has decoded every frame sent so far.

    Sends a control-lane request and waits for its response.  The serve
    loop reads frames sequentially on one thread, so its answer to THIS
    request proves it has already handled everything written before it —
    the cancel frame included.  Control lane, because the work lane's only
    worker is deliberately occupied.
    """
    assert "unknown.control.probe" not in _work_lane_names(), (
        "the barrier probe must be a CONTROL-lane method, or it queues "
        "behind the occupied work-lane worker and never answers"
    )
    _send(daemon_sock, RequestEnvelope(
        id=request_id, method="unknown.control.probe", args={},
    ).to_dict())
    env = _read_response(daemon_sock, request_id)
    assert env.ok is False      # unknown method — we only want the ordering


def _work_lane_names() -> frozenset:
    from server.runner.rpc import WORK_LANE_METHODS
    return WORK_LANE_METHODS


# ----------------------------------------------------------------------
# The regression
# ----------------------------------------------------------------------


def test_cancel_for_a_queued_call_is_honoured(single_worker_rpc) -> None:
    """A cancel that lands while the call is still QUEUED trips it.

    This is #988 reproduced without a clock: the work lane's single
    worker is held by call 1, so call 2's worker body cannot have run
    when call 2's cancel frame arrives.  Registering the call on the
    reader thread is what makes the cancel land somewhere.
    """
    daemon_sock, rpc, executor = single_worker_rpc

    # 1. Occupy the only work-lane worker.
    _send(daemon_sock, _request(1, "holder"))
    assert executor.started.wait(JOIN_TIMEOUT), "executor never started"

    # 2. Queue a second call.  It CANNOT be running: the one worker is
    #    inside executor.hold.wait().
    _send(daemon_sock, _request(2, "queued"))

    # 3. Cancel the queued call, then prove the reader consumed it.
    _send(daemon_sock, CancelFrame(id=2).to_dict())
    _barrier_through_reader(daemon_sock, rpc, request_id=99)

    # 4. The cancel matched an in-flight id -- it was not dropped as
    #    "unknown".  Pre-fix this read unknown=1, tripped=0.
    stats = rpc.cancel_stats()
    assert stats["received"] == 1
    assert stats["tripped"] == 1, (
        f"cancel for a queued call was not honoured: {stats}"
    )
    assert stats["unknown"] == 0
    assert stats["late"] == 0

    # 5. Release the worker.  The queued call now runs and must observe a
    #    token that is ALREADY tripped, through the thread-local the
    #    dispatcher publishes to it.
    executor.hold.set()
    env2 = _read_response(daemon_sock, 2)
    assert env2.ok is False
    assert "queued" in executor.cancelled_ids


def test_cancel_for_an_unknown_id_is_counted_and_not_silent(
    single_worker_rpc, caplog,
) -> None:
    """A cancel naming an id the runner never saw is loud (#988).

    The two miss branches are different events and must not read alike:
    this one means the daemon believes a call is in flight that is not,
    and used to be a DEBUG line indistinguishable from "no cancel was
    requested".
    """
    daemon_sock, rpc, _executor = single_worker_rpc

    with caplog.at_level("WARNING", logger="server.runner.rpc"):
        _send(daemon_sock, CancelFrame(id=4242).to_dict())
        _barrier_through_reader(daemon_sock, rpc, request_id=98)

    stats = rpc.cancel_stats()
    assert stats["received"] == 1
    assert stats["unknown"] == 1
    assert stats["tripped"] == 0
    warnings = [
        rec.getMessage() for rec in caplog.records
        if rec.levelname == "WARNING"
    ]
    assert any("4242" in msg for msg in warnings), (
        f"no WARNING naming the id: {warnings}"
    )


def test_cancel_after_completion_is_benign_and_counted_late(
    single_worker_rpc,
) -> None:
    """A cancel racing a completion is routine — DEBUG, counted ``late``.

    The distinction matters because promoting this one to WARNING would
    make the signal useless: any token tripped near the end of a turn
    produces it.
    """
    daemon_sock, rpc, executor = single_worker_rpc

    executor.hold.set()             # calls complete immediately
    _send(daemon_sock, _request(1, "done"))
    env = _read_response(daemon_sock, 1)
    assert env.ok is True

    _send(daemon_sock, CancelFrame(id=1).to_dict())
    _barrier_through_reader(daemon_sock, rpc, request_id=97)

    stats = rpc.cancel_stats()
    assert stats["received"] == 1
    assert stats["late"] == 1
    assert stats["unknown"] == 0
    assert stats["tripped"] == 0


# ----------------------------------------------------------------------
# The session-hosted path shares this machinery (#988 "not established")
# ----------------------------------------------------------------------


def test_registration_is_on_the_reader_thread_for_every_method() -> None:
    """``serve`` registers the call before dispatching, on EVERY branch.

    #988 exercised only the Phase-2 cli-only path and could not say
    whether the session-hosted one was affected.  It is the same
    machinery: ``_active_calls`` is written by ``_register_call`` from
    ``serve``, and all three dispatch branches — the synchronous
    ``session.bootstrap`` one, the work lane (which carries
    ``session.send_message``) and the control lane — receive the token
    it returned.  An AST guard rather than prose, so a fourth branch
    added later cannot quietly reinstate the worker-side registration.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(RunnerRPC.serve))
    tree = ast.parse(src)

    register_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_register_call"
    ]
    assert len(register_calls) == 1, (
        "serve must register each call exactly once, on the reader thread"
    )

    dispatches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute)
             and node.func.attr == "_handle_request")
            or (isinstance(node.func, ast.Attribute)
                and node.func.attr == "submit")
        )
    ]
    assert dispatches, "serve dispatches nothing — test is looking at the wrong code"
    for node in dispatches:
        names = {
            a.id for a in node.args if isinstance(a, ast.Name)
        }
        assert "call_token" in names, (
            "every dispatch branch must hand the worker the token "
            "_register_call published, or a cancel reaches a different "
            "object than the tool polls (#988)"
        )


def test_work_lane_carries_the_session_hosted_verbs() -> None:
    """The verbs a real deployment cancels are on the fixed path.

    ``tool.execute`` is what #988 measured; ``session.send_message`` is
    what ``client.stop()`` cancels in a session-hosted runner, which is
    the deployment shape.  Both go through the work-lane branch asserted
    above, so the fix covers both.
    """
    names = _work_lane_names()
    assert "tool.execute" in names
    assert "session.send_message" in names
