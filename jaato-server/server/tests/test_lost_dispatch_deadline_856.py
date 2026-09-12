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
  :class:`RunnerDispatchNotReceived`, naming the id and the session.
* **The window's FULLNESS, not its size, is what makes an answer safe.**
  It is a ``deque(maxlen=...)``, so it has evicted something only when
  full; until then "not in the window" proves "never registered".  A
  FULL window whose floor sits above the id is answered
  ``indeterminate``, never ``never_received`` — that single
  misclassification is the one that says "safe to retry" about work
  that already ran.
* :class:`RunnerDispatchLost` is a ``RunnerRPCTimeout`` and therefore
  NOT a ``RunnerCallError``: the turn dies, the session lives, and a
  client can tell this apart from #851 by ``error_type`` alone.
* A runner that says "yes, that id is mine" buys the call another full
  window — the watchdog must never become a wall-clock cap on a turn.
* **"did not happen" and "happened, result lost" are different types**,
  and the difference reaches a client as structured evidence rather
  than as prose.

WHY ``recoverable`` IS NOT THAT FIELD.  In this tree
``ErrorEvent.recoverable`` means *this session can continue*: every
``recoverable=False`` site in ``server/core.py`` is a configuration or
provider-connect failure that ends initialisation, and the SDK's
``_await_session_info`` documents that it deliberately ignores the flag
for terminal-ness ("any ``ErrorEvent`` terminates the wait, regardless
of its ``recoverable`` flag").  A lost dispatch is precisely the case
where the session DOES continue — that is what sparing a
``RunnerRPCTimeout`` in the model thread buys — so flipping the flag to
encode retry-safety would assert something false about session
viability to every consumer that reads it the documented way.

``ErrorEvent.details`` is the field whose own comment says "this is what
a driver branches on", with ``error`` left as the human sentence.  That
is where ``verdict`` and ``may_have_run`` go, in the same shape
``SessionRefused.may_exist`` takes.

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
    RunnerDispatchNotReceived,
    RunnerDispatchUnknown,
    RunnerRPCClient,
    RunnerRPCTimeout,
    RunnerResultLost,
    classify_reconciliation,
    _resolve_dispatch_ack_timeout,
    _VERDICT_EXCEPTIONS,
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
        #: Ids the runner remembers having been asked to run — its
        #: bounded seen-id window.
        self.known: List[int] = []
        #: The window's ``maxlen``, reported to the daemon.  A window
        #: that is NOT full has evicted nothing, which is what lets the
        #: daemon say "never received" rather than "cannot tell".
        self.window_capacity: int = 256
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
            # Faithful to ``RunnerRPC``: ``_register_call`` runs on the
            # READER thread, before the handler, so the probe is in both
            # the active set and the seen-id window of the very answer it
            # produces.  Getting this wrong is not cosmetic — the
            # classifier reads an empty window beside a nonzero
            # high-water mark as "this runner's answer cannot be
            # interpreted", which is exactly what an unfaithful fake
            # looks like.
            self.known.append(rid)
            self._respond(rid, {
                "has_host": True,
                "ready": True,
                "session_id": "sess-856",
                "tool_count": 3,
                "active_call_ids": sorted(set(self.active) | {rid}),
                "known_request_ids": sorted(set(self.known)),
                "seen_window_capacity": self.window_capacity,
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
            # The injected loss: the dispatcher never gets it, so it
            # enters NEITHER the active set nor the window.
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
async def test_a_finished_id_is_a_different_type_that_is_unsafe_to_retry():
    """The verdict a caller must not treat like the others.

    The runner RAN the call: history advanced, tools executed, side
    effects are in the world, and only the daemon's view of the result
    was lost.  Re-sending the turn re-executes it.  A single
    ``RunnerDispatchLost`` made that indistinguishable from the case
    where nothing happened, so the distinction is a TYPE — the rule
    ``SessionNotConfirmed`` already states for ``create_session``.
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
        with pytest.raises(RunnerResultLost) as caught:
            await asyncio.wait_for(task, timeout=OUTER)
        exc = caught.value
    finally:
        await _shutdown(client, runner)

    assert isinstance(exc, RunnerDispatchLost), "the base must still catch it"
    assert exc.verdict == "finished"
    assert exc.may_have_run is True
    assert "side effects have already happened" in str(exc)

    details = exc.as_details()
    assert details["may_have_run"] is True
    assert details["verdict"] == "finished"
    assert details["method"] == "session.send_message"
    assert details["session_id"] == "sess-856"

    assert client.dispatch_lost_count() == 1
    assert client.dispatch_lost_may_have_run_count() == 1


@pytest.mark.asyncio
async def test_a_never_received_id_is_the_one_type_safe_to_retry():
    """Its counterpart, and the only verdict allowed to say so."""
    runner, client = await _client(drop=True)
    try:
        with pytest.raises(RunnerDispatchNotReceived) as caught:
            await asyncio.wait_for(
                client.call("session.send_message", {"prompt": "turn two"}),
                timeout=OUTER,
            )
        exc = caught.value
    finally:
        await _shutdown(client, runner)

    assert exc.verdict == "never_received"
    assert exc.may_have_run is False
    assert exc.as_details()["may_have_run"] is False
    # Counted, but NOT as a may-have-run: the two numbers are what an
    # operator triages on.
    assert client.dispatch_lost_count() == 1
    assert client.dispatch_lost_may_have_run_count() == 0


def test_only_one_verdict_may_claim_the_work_did_not_happen():
    """The fail-safe invariant, asserted over the whole table.

    A verdict added later inherits ``may_have_run = True`` from the base
    unless its author deliberately proves otherwise, so this fails if
    anything new claims retry-safety without being the one verdict that
    can establish it.
    """
    from server.runner_rpc_client import _VERDICT_EXCEPTIONS

    safe = {
        verdict for verdict, cls in _VERDICT_EXCEPTIONS.items()
        if cls.may_have_run is False
    }
    assert safe == {"never_received"}, (
        f"these verdicts claim the work did not happen: {sorted(safe)}.  "
        f"Only 'never_received' can establish that; every other verdict "
        f"is a form of 'we could not find out', and a wrong 'False' costs "
        f"a caller a duplicated turn"
    )
    assert RunnerDispatchLost.may_have_run is True, (
        "the BASE must default to the safe reading, so a subclass that "
        "forgets to decide does not inherit 'safe to retry'"
    )


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


# ----------------------------------------------------------------------
# The boundary: what a CLIENT can act on
# ----------------------------------------------------------------------


def test_the_retry_safety_answer_survives_the_event_boundary():
    """A client must separate the two without parsing a message string.

    ``_transport_error_details`` is what ``core.py``'s model thread puts
    on ``ErrorEvent.details`` — the field whose own comment says "this
    is what a driver branches on", with ``error`` left as the human
    sentence.  Computing ``may_have_run`` and then dropping it at the
    boundary would leave the distinction visible only in prose.
    """
    from server.core import _transport_error_details

    ran = RunnerResultLost(
        "x", request_id=7, method="session.send_message", session_id="s1",
    )
    not_run = RunnerDispatchNotReceived("x", request_id=8, method="tool.execute")

    assert _transport_error_details(ran) == {
        "verdict": "finished",
        "may_have_run": True,
        "request_id": 7,
        "method": "session.send_message",
        "session_id": "s1",
    }
    assert _transport_error_details(not_run)["may_have_run"] is False
    assert _transport_error_details(not_run)["session_id"] is None

    # A transport failure with no structured evidence leaves the key
    # ABSENT rather than present-and-empty: absent says "this failure
    # has none", an empty dict says "it has some and it is blank".
    assert _transport_error_details(RunnerCallError("dead runner")) is None
    assert _transport_error_details(RunnerRPCTimeout("slow")) is None


def test_recoverable_is_not_the_retry_safety_field_and_must_not_become_one():
    """``recoverable`` means "this SESSION can continue", not "retry me".

    Every ``recoverable=False`` site in ``server/core.py`` is a
    configuration or provider-connect failure that ends initialisation,
    and the SDK's ``_await_session_info`` documents that it ignores the
    flag for terminal-ness.  A lost dispatch is exactly the case where
    the session DOES continue — that is what sparing a
    ``RunnerRPCTimeout`` in the model thread buys — so encoding
    retry-safety there would assert something false about session
    viability to every consumer reading it the documented way.

    Pinned as an AST assertion over the emit site rather than as prose,
    because the tempting edit is a one-word one.
    """
    import ast
    from pathlib import Path

    source = Path(
        __file__
    ).resolve().parents[1].joinpath("core.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    emits = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "ErrorEvent"
        and any(
            kw.arg == "details"
            and "_transport_error_details" in ast.dump(kw.value)
            for kw in node.keywords
        )
    ]
    assert len(emits) == 1, (
        "expected exactly one transport-error ErrorEvent carrying details"
    )
    recoverable = [
        kw.value for kw in emits[0].keywords if kw.arg == "recoverable"
    ]
    assert recoverable and recoverable[0].value is True, (
        "the transport-error ErrorEvent must keep recoverable=True: the "
        "session survives a lost dispatch, and the retry-safety answer "
        "belongs in details.may_have_run"
    )


# ----------------------------------------------------------------------
# The classifier, which is pure — so the dangerous direction is testable
# exactly, with no socket, no thread and no clock.
# ----------------------------------------------------------------------


def _status(known, *, active=(), capacity=256, highest=None):
    """One ``session.health_check`` answer, shaped like the runner's."""
    return {
        "active_call_ids": list(active),
        "known_request_ids": list(known),
        "seen_window_capacity": capacity,
        "highest_request_id": highest if highest is not None else (
            max([*known, *active], default=0)
        ),
    }


def test_a_partly_filled_window_proves_the_call_never_arrived():
    """The ordinary lost dispatch, and the ONLY safe-to-retry answer.

    The window is a ``deque(maxlen=...)``: until it is full it has
    evicted nothing, so an id missing from it was genuinely never
    registered.  ``min(known)`` is then just the oldest id the runner
    happens to have seen — NOT a floor — and reading it as one would
    report ``indeterminate`` here, because a lost id is by construction
    older than the probe that asks about it.  That would be fail-safe
    and useless: the safe verdict would never be reachable at all.
    """
    # id 1 was lost; the probe is id 2 and is the only id the runner has.
    verdict, detail = classify_reconciliation(1, _status([2], active=[2]))
    assert verdict == "never_received", detail
    assert "evicted nothing" in detail


def test_a_full_window_that_rolled_past_the_id_must_not_claim_it_never_ran():
    """The composed defect the window size alone does not bound.

    Once a caller acts differently on the verdicts, an id evicted from
    the window must NOT read as ``never_received`` — that is the single
    misclassification that says "safe to retry" about work that already
    ran.  Full window, floor above the id: the answer is silence, not a
    denial.
    """
    full = list(range(1000, 1000 + 256))          # exactly at capacity
    verdict, detail = classify_reconciliation(42, _status(full, capacity=256))
    assert verdict == "indeterminate", detail
    assert _VERDICT_EXCEPTIONS[verdict].may_have_run is True

    # The same window, one entry short of full: nothing was evicted, so
    # the same id IS conclusively absent.  One entry is the whole
    # difference between "may have run" and "safe to retry", which is
    # why fullness and not size is the bound.
    verdict, _ = classify_reconciliation(42, _status(full[1:], capacity=256))
    assert verdict == "never_received"


def test_an_id_above_a_full_windows_floor_is_still_conclusive():
    """Eviction only silences ids BELOW the floor."""
    full = list(range(100, 100 + 256))
    # 200 is inside the range the window covers and simply absent from it.
    verdict, detail = classify_reconciliation(
        200, _status([i for i in full if i != 200] + [999], capacity=256),
    )
    assert verdict == "never_received", detail


def test_a_runner_that_reports_no_capacity_is_read_conservatively():
    """An answer we cannot size is never read as safe-to-retry.

    The capacity is reported by the runner rather than assumed by the
    daemon.  If it is missing, a non-empty window is treated as full —
    which can only move an answer from ``never_received`` towards
    ``indeterminate``, never the other way.
    """
    status = _status([2], active=[2])
    status.pop("seen_window_capacity")
    verdict, _ = classify_reconciliation(1, status)
    assert verdict == "indeterminate"


def test_active_and_finished_are_decided_before_any_window_reasoning():
    """The two verdicts the window cannot affect."""
    assert classify_reconciliation(
        7, _status([7], active=[7]))[0] == "running"
    verdict, detail = classify_reconciliation(7, _status([7]))
    assert verdict == "finished"
    assert "side effects have already happened" in detail


def test_a_runner_that_remembers_nothing_while_having_run_things():
    """Uninterpretable, and therefore not safe.

    Unreachable with a real ``deque``, and it is precisely the shape an
    unfaithful test double takes — which is how this was caught.
    """
    verdict, _ = classify_reconciliation(1, _status([], highest=9))
    assert verdict == "indeterminate"
    # A runner that has registered NOTHING is a different claim, and a
    # conclusive one.
    verdict, _ = classify_reconciliation(1, _status([], highest=0))
    assert verdict == "never_received"


def test_a_runner_can_answer_a_call_with_nothing_and_stay_healthy():
    """One demonstrated mechanism for the observed state (#856 triage).

    #920 made the runner refuse to WRITE an oversized frame rather than
    write one the peer cannot skip, and turned that drop into a small
    typed error for the call it belonged to.  That substitution is
    guarded by ``if not ok or self._closed: return`` — so it happens for
    a SUCCESS response and not for an ERROR one, on the reasoning that
    "an error frame that did not fit will not fit a second time".  The
    substitute, though, is small by construction; what did not fit is
    the original, whose ``result`` dict can carry megabytes of tool
    output beside the traceback.

    The consequence is exactly the state the issue describes: the call
    is answered with NOTHING, the channel stays OPEN, the runner returns
    to idle with an empty active set, and the daemon waits forever.
    Distinct from #851 in the one way that matters — there is no EOF, so
    nothing fails the in-flight future.

    This is NOT a claim that it is what happened in the reported
    incident: that would need the runner's own log, which the report
    does not carry.  It is a demonstration that the class of failure is
    reachable, and it is the case the ``finished`` verdict answers — the
    id has left ``active_call_ids`` and is still in
    ``known_request_ids``, so the daemon reports a lost RESPONSE rather
    than a lost dispatch.
    """
    from shared.framing import MAX_MESSAGE_SIZE
    from server.runner.envelope import ErrorPayload
    from server.runner.rpc import RunnerRPC

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        rpc = RunnerRPC(a, lambda name, args: (True, {}))
        oversized = "x" * (MAX_MESSAGE_SIZE + 1024)
        b.settimeout(0.3)

        # ok=True: #920's substitution fires, so the caller IS answered.
        rpc._emit_response(
            request_id=41, ok=True, result={"stdout": oversized},
        )
        assert read_frame_sync(b) is not None
        assert rpc._closed is False

        # ok=False: nothing is written at all.
        rpc._emit_response(
            request_id=42, ok=False, result={"stdout": oversized},
            error=ErrorPayload(type="ToolError", message=oversized),
        )
        with pytest.raises(socket.timeout):
            read_frame_sync(b)
        assert rpc._closed is False, (
            "the channel stayed open, which is what makes this NOT #851: "
            "no EOF reaches the daemon, so no in-flight future fails"
        )
    finally:
        a.close()
        b.close()


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
