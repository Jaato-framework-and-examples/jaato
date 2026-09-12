"""A dispatched runner RPC that the runner does not have must fail (#856).

THE FAILURE.  A session's second turn was registered in the daemon's
``_in_flight`` and never happened.  Sixty-eight minutes later the client
was still blocked in ``session.ask()`` and ``py-spy`` showed every thread
in all three processes idle: no lock held, no read outstanding, the pipe
healthy, the runner alive with an empty work queue.  The daemon's whole
record of the event was four ``[RPC_DIAG] daemon _in_flight SET id=30..33``
lines, which say what the daemon INTENDED and nothing about what happened
to it.

Not #851.  There a runner DIES, the daemon sees a closed pipe, and every
in-flight future fails with ``RunnerCallError``.  Here the transport is
healthy and the request evaporates, so the caller waits forever.  One of
those two reaches a person as an error; the other is indistinguishable
from the agent thinking.

WHAT IS PINNED HERE.

* ``call`` records the DISPATCH (``_dispatched``), not only the
  registration — the gap the incident fell into had no observable form.
* A call that goes silent is reconciled against the runner over the
  CONTROL lane, so the answer arrives while the work lane is busy.
* A runner that says "never heard of it" fails that call with
  :class:`RunnerDispatchLost`, naming the id and the session.
* :class:`RunnerDispatchLost` is a ``RunnerRPCTimeout`` and therefore
  NOT a ``RunnerCallError``: the turn dies, the session lives, and a
  client can tell this apart from #851 by ``error_type`` alone.
* A runner that says "yes, that id is mine" buys the call another full
  window — the watchdog must never become a wall-clock cap on a turn.

WHY THIS IS NOT A TIMING TEST.  The drop is INJECTED, not raced: the
fake runner below decodes the request and deliberately never answers it,
which is a property of the harness rather than of the clock.  The
deadline is set to 50 ms and every assertion is made by awaiting the
watchdog task or the call itself under a generous outer bound, so a
loaded CPU makes these tests slower and cannot make them fail.  Nothing
here asserts an ordering between two timers.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional

import pytest

from shared.framing import read_frame_sync, write_frame_sync

from server.runner.json_codec import dumps, loads
from server.runner_rpc_client import (
    DEFAULT_DISPATCH_ACK_TIMEOUT,
    RunnerCallError,
    RunnerDispatchLost,
    RunnerRPCClient,
    RunnerRPCTimeout,
    _resolve_dispatch_ack_timeout,
)


#: Short enough that the suite does not wait on it, long enough that the
#: fake runner thread is comfortably scheduled first.  No assertion
#: depends on its value — see the module docstring.
ACK = 0.05

#: Outer bound on any awaited call.  1000x ``ACK``: it exists so a
#: genuine hang fails the suite instead of wedging it, never to decide a
#: race.
OUTER = 10.0


class FakeRunner:
    """The runner end of the socketpair, driven from a reader thread.

    Answers ``session.health_check`` from :attr:`active` / :attr:`known`
    — the reconciliation payload the real
    :meth:`server.runner.rpc.RunnerRPC._handle_session_health_check`
    builds from ``_active_calls`` and its seen-id window — and treats
    every OTHER method according to :attr:`drop`:

    * ``drop=True``  — decode the request, record its id, answer
      nothing.  This IS the injected lost dispatch.
    * ``drop=False`` — answer immediately with an ok response.

    Lifecycle: construct, :meth:`start` the thread, :meth:`stop` it in a
    ``finally``.  The thread owns the runner socket for its whole life;
    the test touches only the recorded lists, which are appended under
    the GIL and read after a synchronising await.
    """

    def __init__(self, sock: socket.socket, *, drop: bool = True) -> None:
        self.sock = sock
        self.drop = drop
        #: Request ids this runner decoded but deliberately did not answer.
        self.dropped: List[int] = []
        #: Ids the runner reports as RUNNING right now.
        self.active: List[int] = []
        #: Ids the runner remembers having been asked to run.
        self.known: List[int] = []
        #: Set once a health-check probe has been answered, so a test can
        #: assert the reconciliation actually happened rather than infer it.
        self.probed = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- internals ----------------------------------------------------

    def _serve(self) -> None:
        while not self._stop:
            try:
                raw = read_frame_sync(self.sock)
            except (OSError, ValueError):
                return
            if raw is None:
                return
            try:
                payload = loads(raw)
            except Exception:  # noqa: BLE001
                return
            if payload.get("kind") != "request":
                continue
            self._on_request(payload)

    def _on_request(self, payload: Dict[str, Any]) -> None:
        rid = int(payload["id"])
        if payload.get("method") == "session.health_check":
            self._respond(rid, {
                "has_host": True,
                "ready": True,
                "session_id": "sess-856",
                "tool_count": 3,
                "active_call_ids": sorted(self.active),
                "known_request_ids": sorted(self.known),
                # Deliberately ABOVE every id under test, exactly as the
                # real runner reports it: the probe is itself a request
                # and is registered before its own handler runs.  A
                # reconciler comparing against this would call every id
                # "received", which is the one answer that makes the
                # whole mechanism useless.
                "highest_request_id": rid,
            })
            self.probed.set()
            return
        if self.drop:
            self.dropped.append(rid)
            return
        self.active.append(rid)
        self.known.append(rid)
        self._respond(rid, {"response": "answered"})

    def _respond(self, rid: int, result: Dict[str, Any]) -> None:
        write_frame_sync(self.sock, dumps({
            "kind": "response", "id": rid, "ok": True,
            "result": result, "error": None,
        }))


async def _client(
    *, drop: bool = True, ack: Optional[float] = ACK,
) -> "tuple[FakeRunner, RunnerRPCClient]":
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )
    runner = FakeRunner(runner_sock, drop=drop)
    runner.start()
    client = RunnerRPCClient(
        daemon_sock, runner_pid=0, dispatch_ack_timeout=ack,
    )
    await client.start()
    client._session_id = "sess-856"
    return runner, client


async def _shutdown(client: RunnerRPCClient, runner: FakeRunner) -> None:
    client._closed = True
    if client._writer is not None:
        client._writer.close()
    if client._read_task is not None:
        client._read_task.cancel()
    runner.stop()


# ----------------------------------------------------------------------
# The acceptance criteria, one test each
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_dropped_dispatch_fails_the_call_instead_of_hanging():
    """Acceptance 1 + 3: the drop is injected; the caller gets an error.

    Before #856 this coroutine never returned.  ``OUTER`` is the
    difference between a failing test and a wedged suite.
    """
    runner, client = await _client(drop=True)
    try:
        with pytest.raises(RunnerDispatchLost) as caught:
            await asyncio.wait_for(
                client.call("session.send_message", {"prompt": "turn two"}),
                timeout=OUTER,
            )
    finally:
        await _shutdown(client, runner)

    assert runner.dropped, "the harness did not actually drop a request"
    assert runner.probed.is_set(), (
        "the call failed without ever asking the runner what it had — "
        "that is a wall-clock cap, not a reconciliation"
    )
    message = str(caught.value)
    assert f"id={runner.dropped[0]}" in message
    assert "sess-856" in message
    assert "session.send_message" in message


@pytest.mark.asyncio
async def test_the_error_names_the_id_the_session_and_the_verdict():
    """Acceptance 1: "with the id and session named"."""
    runner, client = await _client(drop=True)
    try:
        with pytest.raises(RunnerDispatchLost) as caught:
            await asyncio.wait_for(
                client.call("tool.execute", {"name": "cli_based_tool"}),
                timeout=OUTER,
            )
    finally:
        await _shutdown(client, runner)

    message = str(caught.value)
    assert "never_received" in message, message
    assert "runner_pid=0" in message, message
    # The counter is the assertable form of the WARNING line.
    assert client.dispatch_lost_count() == 1


@pytest.mark.asyncio
async def test_a_lost_dispatch_is_not_a_dead_runner():
    """Acceptance 2: distinguishable from #851 in the client-visible error.

    #851's shape is ``RunnerCallError`` raised from the read loop's
    teardown; a lost dispatch must not be mistakable for it, in either
    direction.  The type relationship is the whole of the distinction a
    client sees, because ``ErrorEvent.error_type`` is
    ``type(exc).__name__``.
    """
    assert issubclass(RunnerDispatchLost, RunnerRPCTimeout)
    assert not issubclass(RunnerDispatchLost, RunnerCallError)
    assert not issubclass(RunnerCallError, RunnerDispatchLost)
    # And the consequence that matters: core.py's model thread spares a
    # RunnerRPCTimeout and terminates the session for everything else.
    assert isinstance(RunnerDispatchLost("x"), RunnerRPCTimeout)
    assert not isinstance(RunnerCallError("x"), RunnerRPCTimeout)


@pytest.mark.asyncio
async def test_a_runner_that_owns_the_id_keeps_its_call_alive():
    """The watchdog is a reconciliation, NOT a wall-clock cap on a turn.

    A model turn legitimately runs for minutes with nothing on the wire.
    Here the runner answers the probe with the id in ``active_call_ids``,
    which must buy another full window — repeatedly.  Failing this test
    would mean shipping a fix that kills healthy long turns, which is
    worse than the hang it replaces.
    """
    runner, client = await _client(drop=True)
    try:
        task = asyncio.ensure_future(
            client.call("session.send_message", {"prompt": "a long turn"})
        )
        # Let the request reach the runner, then declare it running.
        for _ in range(200):
            if runner.dropped:
                break
            await asyncio.sleep(0.005)
        assert runner.dropped, "the request never reached the fake runner"
        runner.active.append(runner.dropped[0])
        runner.known.append(runner.dropped[0])

        # Several windows must pass with the call still alive.
        await asyncio.sleep(ACK * 6)
        assert not task.done(), (
            "a call the runner claims to be running was failed anyway"
        )
        assert runner.probed.is_set(), "no reconciliation happened at all"
        assert client.dispatch_lost_count() == 0
        task.cancel()
    finally:
        await _shutdown(client, runner)


@pytest.mark.asyncio
async def test_a_finished_id_reports_a_lost_response_not_a_lost_dispatch():
    """The third verdict: the runner RAN it and the response vanished.

    Same outcome for the caller — the call fails — and a different
    account in the log, which is the only place the difference can be
    acted on.
    """
    runner, client = await _client(drop=True)
    try:
        task = asyncio.ensure_future(
            client.call("session.send_message", {"prompt": "hi"})
        )
        for _ in range(200):
            if runner.dropped:
                break
            await asyncio.sleep(0.005)
        assert runner.dropped
        # Known, but not active: the runner ran it to completion.
        runner.known.append(runner.dropped[0])
        with pytest.raises(RunnerDispatchLost) as caught:
            await asyncio.wait_for(task, timeout=OUTER)
    finally:
        await _shutdown(client, runner)
    assert "RESPONSE frame is what was lost" in str(caught.value)


@pytest.mark.asyncio
async def test_a_normal_call_is_untouched_and_leaves_no_watchdog():
    """The common path pays nothing and leaves nothing running."""
    runner, client = await _client(drop=False)
    try:
        response = await asyncio.wait_for(
            client.call("session.send_message", {"prompt": "hi"}),
            timeout=OUTER,
        )
        assert response.ok
        assert client.dispatch_lost_count() == 0
        assert client._dispatch_watchdogs == {}
        assert client._dispatched == {}
        assert client._frame_seen == {}
    finally:
        await _shutdown(client, runner)


@pytest.mark.asyncio
async def test_the_dispatch_is_recorded_between_write_and_response():
    """Ask 3: the DISPATCH is observable, not only the registration.

    ``_in_flight SET id=N`` says the daemon meant to send.  ``_dispatched``
    says the frame reached the wire.  An ``_in_flight`` entry with no
    ``_dispatched`` entry is exactly the state the incident's four log
    lines could not distinguish from a healthy in-flight turn.
    """
    runner, client = await _client(drop=True)
    try:
        task = asyncio.ensure_future(
            client.call("session.send_message", {"prompt": "hi"})
        )
        for _ in range(200):
            if runner.dropped:
                break
            await asyncio.sleep(0.005)
        rid = runner.dropped[0]
        assert client._in_flight.get(rid) is not None
        assert client._dispatched.get(rid) == "session.send_message"
        task.cancel()
    finally:
        await _shutdown(client, runner)


@pytest.mark.asyncio
async def test_the_probe_itself_is_never_watched():
    """No recursion: the reconciler must not reconcile itself.

    ``session.health_check`` IS the probe.  Arming a watchdog on it
    would have a silent probe start a second probe, unbounded.
    """
    runner, client = await _client(drop=True)
    try:
        task = asyncio.ensure_future(client.session_health_check(timeout=1.0))
        await asyncio.sleep(ACK * 3)
        assert client._dispatch_watchdogs == {}, (
            "the reconciliation probe armed a watchdog on itself"
        )
        task.cancel()
    finally:
        await _shutdown(client, runner)


@pytest.mark.asyncio
async def test_the_watchdog_can_be_switched_off():
    """``dispatch_ack_timeout=None`` restores the pre-#856 behaviour.

    Kept reachable because an operator who would rather hang than risk
    a false positive must have a way to say so — and because the
    difference between "on" and "off" is what a bisect needs.
    """
    runner, client = await _client(drop=True, ack=None)
    try:
        task = asyncio.ensure_future(
            client.call("session.send_message", {"prompt": "hi"})
        )
        await asyncio.sleep(ACK * 6)
        assert not task.done()
        assert client._dispatch_watchdogs == {}
        assert not runner.probed.is_set()
        task.cancel()
    finally:
        await _shutdown(client, runner)


# ----------------------------------------------------------------------
# The knob
# ----------------------------------------------------------------------


def test_zero_disables_and_nonsense_does_not(monkeypatch):
    """``0`` disables, following the provider-deadline convention.

    A negative or unparseable value falls back to the DEFAULT rather
    than disabling: "unbounded" is the bug this exists to fix, so it
    must not be reachable by typo.
    """
    monkeypatch.delenv("JAATO_RUNNER_ACK_TIMEOUT", raising=False)
    assert _resolve_dispatch_ack_timeout() == DEFAULT_DISPATCH_ACK_TIMEOUT

    monkeypatch.setenv("JAATO_RUNNER_ACK_TIMEOUT", "0")
    assert _resolve_dispatch_ack_timeout() is None

    monkeypatch.setenv("JAATO_RUNNER_ACK_TIMEOUT", "45.5")
    assert _resolve_dispatch_ack_timeout() == 45.5

    for bad in ("-1", "banana", ""):
        monkeypatch.setenv("JAATO_RUNNER_ACK_TIMEOUT", bad)
        assert _resolve_dispatch_ack_timeout() == DEFAULT_DISPATCH_ACK_TIMEOUT


def test_the_default_window_clears_the_one_unanswerable_call():
    """The default must not race the call during which nobody can answer.

    A long call is NOT a problem for this watchdog: the probe finds the
    id in ``active_call_ids`` and buys another window, which is why
    ``session.replay_messages`` can carry a 180s cap without interacting
    with a 120s window at all.

    ``session.bootstrap`` is the exception, and the only one.  It runs
    on the runner's MAIN thread — synchronously, so ``aa_change_profile``
    confines the thread that later spawns the workers — which means the
    reader thread is inside it and NEITHER lane answers.  A probe during
    bootstrap would time out and read as a lost dispatch.

    Two independent things stop that, and this pins the cheaper one: the
    window sits clear above bootstrap's own timeout, so bootstrap's
    deadline always fires first.  ``_bootstrap_in_flight`` is the other,
    and the test below covers it.
    """
    import inspect

    import server.runner_rpc_client as mod
    from server.runner.rpc import MAIN_THREAD_METHODS

    assert MAIN_THREAD_METHODS == {"session.bootstrap"}, (
        "a second reader-thread-blocking method appeared; give it the "
        "same two protections bootstrap has"
    )
    default = inspect.signature(
        mod.RunnerRPCClient.bootstrap_session
    ).parameters["timeout"].default
    assert isinstance(default, (int, float)) and default > 0

    assert DEFAULT_DISPATCH_ACK_TIMEOUT > float(default) * 2, (
        f"the watchdog window ({DEFAULT_DISPATCH_ACK_TIMEOUT}s) does not "
        f"clear session.bootstrap's own deadline ({default}s), during "
        f"which the runner cannot answer a probe at all"
    )


@pytest.mark.asyncio
async def test_a_bootstrap_in_flight_suppresses_the_probe():
    """The other protection: do not interrogate a runner that cannot answer.

    While ``session.bootstrap`` is outstanding the runner's reader thread
    is inside it, so a probe would go unanswered and the verdict would be
    ``unreachable`` — a lost-dispatch report for a session that is merely
    starting up.  The watchdog waits instead; bootstrap carries its own
    deadline.
    """
    runner, client = await _client(drop=True)
    try:
        boot = asyncio.ensure_future(
            client.call("session.bootstrap", {"session_id": "sess-856"})
        )
        other = asyncio.ensure_future(
            client.call("session.send_message", {"prompt": "hi"})
        )
        await asyncio.sleep(ACK * 6)
        assert not runner.probed.is_set(), (
            "the watchdog interrogated a runner whose reader thread is "
            "blocked in bootstrap"
        )
        assert not boot.done() and not other.done()
        assert client.dispatch_lost_count() == 0
        boot.cancel()
        other.cancel()
    finally:
        await _shutdown(client, runner)


# ----------------------------------------------------------------------
# The runner's half of the reconciliation
# ----------------------------------------------------------------------


def test_the_runner_reports_what_it_actually_received():
    """Ask 2, runner side: ``health_check`` carries the transport view.

    And the window is a WINDOW: the probe's own registration must not
    be able to answer for an id that never arrived, which is what a
    ``highest_request_id`` comparison would do.
    """
    from server.runner.rpc import SEEN_REQUEST_ID_MEMORY, RunnerRPC

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        rpc = RunnerRPC(a, lambda name, args: (True, {}))
        rpc._register_call(7)
        rpc._register_call(8)
        ok, status = rpc._handle_session_health_check()
        assert ok
        assert status["active_call_ids"] == [7, 8]
        assert status["known_request_ids"] == [7, 8]
        assert status["highest_request_id"] == 8
        assert status["has_host"] is False   # reported without a session

        # A completed call leaves the window but not the memory.
        with rpc._active_lock:
            rpc._active_calls.pop(7)
        _, status = rpc._handle_session_health_check()
        assert status["active_call_ids"] == [8]
        assert 7 in status["known_request_ids"]

        assert SEEN_REQUEST_ID_MEMORY >= 64, (
            "the window has to outlive the ack deadline by a wide margin"
        )
    finally:
        a.close()
        b.close()
