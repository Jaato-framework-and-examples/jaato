"""#1033 — a bootstrap that failed, discovered at an unrelated verb.

THE REPORT.  ``session.new`` began failing intermittently on a live daemon
with::

    RunnerCallError: session.get_context_usage failed:
    ToolError: session not bootstrapped on this runner

``session.get_context_usage`` is a read-only toolbar RPC.  It is not what
failed; it is what happened to ask first.

THE CHAIN, top to bottom.  ``dispatch_bootstrap_envelope`` catches every
bootstrap failure and does not propagate it — deliberately: it emits
``SessionTerminatedEvent`` so cascade observers see the death, and it calls
``mark_runner_ready()`` so a warm pool slot cannot strand a client-tool push
on a readiness timeout.  Both of those are right and both still happen.  What
did not happen is anyone being TOLD.  So ``_bootstrap_session_common`` carried
straight on to ``server.initialize()``, which asks the runner for its context
usage with no guard (``core.py``'s "Emit initial context update so toolbar
shows correct usage at startup") — the immediate neighbour of the one call in
that function that IS guarded, whose comment names this very contingency:
"the runner-side handler may return ``stage="no_session"`` if
``session.bootstrap`` RPC hasn't completed".  The runner then answered
``no_host``, correctly, and the wrapper raised out of session creation.

WHAT WAS *NOT* WRONG, and is deliberately left alone:

* ``RunnerRPC._require_ready_session``.  ``no_host`` is the honest answer to
  "what is this session's context usage" when the runner hosts no session.
  Making it wait, retry or answer optimistically would turn a visible refusal
  into a session reporting usage nobody measured.
* the runner's lane split.  The issue's hypothesis was a control-lane call
  served by a retired worker mid-bootstrap.  That is not reachable on this
  path: ``session.bootstrap`` runs SYNCHRONOUSLY on the reader thread, so no
  frame can even be DECODED while it runs, and the daemon's own ordering is
  strictly sequential — spawn, dispatch bootstrap, then initialize.  The
  ``no_host`` answer is not a race; it is a bootstrap that failed.

THE FIX.  The outcome is recorded where it is known
(``dispatch_bootstrap_envelope`` → ``JaatoServer.note_runner_bootstrap_outcome``)
and consulted where session creation is decided
(``SessionManager._initialize_or_refuse``).  A runner that hosts no session is
asked no ``session.*`` question at all, and the refusal names the bootstrap
failure — an AppArmor confinement mismatch, a provider connect failure, a 30 s
bootstrap timeout — instead of a toolbar read.

WHAT THIS MODULE CANNOT DO.  The daemon-side crash itself needs a live
provider to reach (``initialize()`` returns False on "No model bound" long
before the usage read), so the tests below pin the ORDERING INVARIANT that
makes the crash unreachable, plus — over a real socket, against the real
dispatcher — the downstream half of the chain: that a refused bootstrap is
exactly what makes every later session verb answer with the reported string.
"""

from __future__ import annotations

import ast
import inspect
import json
import socket
import textwrap
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from jaato_server.shared.framing import read_frame_sync, write_frame_sync

from jaato_server.server.runner.envelope import (
    KIND_RESPONSE,
    KIND_STREAM,
    RequestEnvelope,
    ResponseEnvelope,
)
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.session_manager import SessionManager

try:  # pragma: no cover - import shape differs per invocation
    from jaato_server.shared.tests.reversion import (
        Reversion,
    )
except Exception:  # pragma: no cover
    Reversion = None  # type: ignore[assignment]


_SESSION_MANAGER = "jaato-server/jaato_server/server/session_manager.py"
_RUNNER_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"


REVERSIONS = [] if Reversion is None else [
    Reversion(
        target=_SESSION_MANAGER,
        find="        if not initialize_or_refuse(server, envelope.session_id):",
        replace="        if not server.initialize():",
        test="test_the_creation_funnel_settles_the_bootstrap_before_initialize",
        because=(
            "session creation stops consulting the bootstrap outcome and "
            "walks back into initialize()'s unguarded context-usage RPC"
        ),
    ),
    Reversion(
        target=_SESSION_MANAGER,
        find="""    bootstrap_error = getattr(server, "runner_bootstrap_error", None)
    if not bootstrap_error:
        return server.initialize()""",
        replace="""    bootstrap_error = None
    if not bootstrap_error:
        return server.initialize()""",
        test="test_a_hostless_runner_is_never_asked_for_its_context_usage",
        because=(
            "the refusal stops reading the recorded outcome, so a failed "
            "bootstrap initializes as though it had succeeded"
        ),
    ),
    Reversion(
        target=_RUNNER_SPAWN,
        find="""        _note_bootstrap_outcome(
            server, f"{type(exc).__name__}: {exc_message(exc)}",
        )""",
        replace="""        pass""",
        test="test_a_failed_bootstrap_dispatch_is_recorded_on_the_server",
        because=(
            "the one place that knows the bootstrap failed stops recording "
            "it, so every consumer sees 'nothing known to be wrong'"
        ),
    ),
]


# A generous ceiling on "the other thread is not going to do this at all".
# Nothing below asserts that anything happens WITHIN it; it bounds a hang.
JOIN_TIMEOUT = 10.0


# ----------------------------------------------------------------------
# Doubles
# ----------------------------------------------------------------------


class _RecordingServer:
    """The half of ``JaatoServer`` this seam touches.

    Deliberately NOT a real ``JaatoServer``: the property under test is
    that ``initialize`` is not REACHED, and a double that raises when it
    is called states that far more sharply than a real one that would
    return False for its own unrelated reasons.
    """

    def __init__(self, bootstrap_error: Optional[str] = None) -> None:
        self.runner_bootstrap_error = bootstrap_error
        self.initialize_calls = 0
        self.events: List[Any] = []
        self.initialize_result = True

    def initialize(self) -> bool:
        self.initialize_calls += 1
        return self.initialize_result

    def emit(self, event: Any) -> None:
        self.events.append(event)


class _FakeBootstrapRPC:
    """``RunnerRPCClient`` stand-in for ``dispatch_bootstrap_envelope``."""

    def __init__(self, exc: Optional[BaseException] = None) -> None:
        self._exc = exc
        self.calls = 0

    def bootstrap_session_threadsafe(
        self, envelope: Any, timeout: float = 30.0,
    ) -> Dict[str, Any]:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return {"ok": True, "ready": True, "session_id": "s"}


class _DispatchServer:
    """Minimal server surface ``dispatch_bootstrap_envelope`` drives.

    Carries the REAL ``note_runner_bootstrap_outcome`` /
    ``runner_bootstrap_error`` pair by delegating to a real
    ``JaatoServer`` would be heavier than this seam needs; the recorder
    contract is one setter and one reader, and both are asserted here
    against ``JaatoServer``'s own implementation in
    ``test_the_recorder_contract_is_the_one_jaato_server_implements``.
    """

    def __init__(self) -> None:
        self.runner_rpc: Any = None
        self._recorded: Optional[str] = None
        self.ready_marks = 0
        self.events: List[Any] = []
        self.terminations: List[BaseException] = []

    # -- the recorder ---------------------------------------------------
    def note_runner_bootstrap_outcome(self, error: Optional[str]) -> None:
        self._recorded = error or None

    @property
    def runner_bootstrap_error(self) -> Optional[str]:
        return self._recorded

    # -- everything the dispatch also touches ---------------------------
    def mark_runner_ready(self) -> None:
        self.ready_marks += 1

    def emit(self, event: Any) -> None:
        self.events.append(event)

    def _emit_tool_id_registry_from_schemas(self) -> None:
        pass

    def _emit_error_termination_from_exc(
        self, exc: BaseException, **kw: Any,
    ) -> None:
        self.terminations.append(exc)


class _NullExecutor:
    """Phase-2 ``execute_fn`` stand-in — never reached by these tests."""

    def __call__(self, name: str, args: Dict[str, Any]):
        return False, {"error": f"no such tool: {name}"}


# ----------------------------------------------------------------------
# Wire helpers (same technique as test_cancel_before_worker_registers_988)
# ----------------------------------------------------------------------


def _send(sock: socket.socket, payload: Dict[str, Any]) -> None:
    write_frame_sync(sock, json.dumps(payload))


def _read_response(
    sock: socket.socket, request_id: int,
) -> ResponseEnvelope:
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


@pytest.fixture
def serving_rpc():
    """``(daemon_sock, rpc)`` with the REAL dispatcher serving a socketpair."""
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )
    rpc = RunnerRPC(
        runner_sock, _NullExecutor(), max_workers=1, control_workers=2,
    )
    thread = threading.Thread(
        target=rpc.serve, name="rpc-serve-1033", daemon=True,
    )
    thread.start()
    try:
        yield daemon_sock, rpc
    finally:
        rpc.shutdown()
        try:
            daemon_sock.close()
        except OSError:
            pass
        thread.join(timeout=JOIN_TIMEOUT)


def _envelope_args(session_id: str = "sess-1033") -> Dict[str, Any]:
    from jaato_server.server.runner.envelope import SessionInitEnvelope
    return SessionInitEnvelope(
        session_id=session_id,
        workspace_path="/tmp/ws-1033",
        profile_name="",
        provider_name="mock",
        model_name="mock-model",
    ).to_dict()


# ----------------------------------------------------------------------
# The downstream half of the chain, over the real wire
# ----------------------------------------------------------------------


def test_a_refused_bootstrap_makes_every_session_verb_answer_no_host(
    serving_rpc, monkeypatch,
) -> None:
    """The reported string is what a FAILED bootstrap looks like later.

    Drives the real ``RunnerRPC.serve`` loop over a real socket.  The
    bootstrap is made to fail the way #1026's per-thread verification
    fails one — ``BootstrapError("confine", ...)`` — and the very next
    frame asks the question the live daemon asked.

    This test is green before the fix as well as after: the runner's
    behaviour here is CORRECT and is not what #1033 changes.  It is here
    because it is the evidence for the diagnosis — it shows that the
    string in the report identifies the victim and not the cause, and it
    fails the day someone "fixes" ``_require_ready_session`` by making it
    answer optimistically.
    """
    from jaato_server.server.runner import session as runner_session

    def _boom(envelope, recycle_pools=None):  # noqa: ANN001
        raise runner_session.BootstrapError(
            "confine",
            "threads outside profile jaato-ws-sess-1033: tid=42 "
            "label='jaato-ws-previous (enforce)'",
        )

    monkeypatch.setattr(runner_session, "bootstrap_session", _boom)

    daemon_sock, rpc = serving_rpc
    _send(daemon_sock, RequestEnvelope(
        id=1, method="session.bootstrap", args=_envelope_args(),
    ).to_dict())
    boot = _read_response(daemon_sock, 1)

    assert boot.ok is False
    assert boot.result["stage"] == "confine", (
        "the bootstrap must report the stage that failed; it is the only "
        "place the real cause is named"
    )

    _send(daemon_sock, RequestEnvelope(
        id=2, method="session.get_context_usage", args={},
    ).to_dict())
    usage = _read_response(daemon_sock, 2)

    assert usage.ok is False
    assert usage.result["stage"] == "no_host"
    assert usage.result["error"] == "session not bootstrapped on this runner"
    assert "confine" not in json.dumps(usage.result), (
        "the later verb cannot name the bootstrap failure — which is "
        "precisely why the daemon must not ask it"
    )


def test_a_session_verb_after_a_GOOD_bootstrap_is_not_refused(
    serving_rpc, monkeypatch,
) -> None:
    """Control: the ``no_host`` answer tracks the bootstrap, nothing else.

    Without this, the test above proves only that some error occurred.
    """
    from jaato_server.server.runner import session as runner_session

    host = SimpleNamespace(
        envelope=None,
        is_ready=True,
        session_id="sess-1033",
        session=SimpleNamespace(
            get_context_usage=lambda: {"total_tokens": 7},
        ),
        runtime=None,
    )
    monkeypatch.setattr(
        runner_session, "bootstrap_session",
        lambda envelope, recycle_pools=None: host,
    )

    daemon_sock, rpc = serving_rpc
    _send(daemon_sock, RequestEnvelope(
        id=1, method="session.bootstrap", args=_envelope_args(),
    ).to_dict())
    assert _read_response(daemon_sock, 1).ok is True

    _send(daemon_sock, RequestEnvelope(
        id=2, method="session.get_context_usage", args={},
    ).to_dict())
    usage = _read_response(daemon_sock, 2)
    assert usage.ok is True
    assert usage.result["usage"]["total_tokens"] == 7


# ----------------------------------------------------------------------
# The recorder: dispatch_bootstrap_envelope tells someone
# ----------------------------------------------------------------------


def test_a_failed_bootstrap_dispatch_is_recorded_on_the_server() -> None:
    """A bootstrap RPC that raised leaves the reason on the server.

    Before #1033 the dispatch logged a WARNING, emitted a terminal event,
    marked the runner ready and returned — and the outcome reached no
    caller, which is why session creation proceeded into a runner that
    hosts no session.
    """
    from jaato_server.server.runner_spawn import dispatch_bootstrap_envelope

    server = _DispatchServer()
    server.runner_rpc = _FakeBootstrapRPC(
        exc=RuntimeError("session.bootstrap: AppArmor confinement mismatch"),
    )

    dispatch_bootstrap_envelope(
        server=server,
        session_id="sess-1033",
        workspace_path="/tmp/ws-1033",
        profile_name="jaato-ws-sess-1033",
    )

    assert server.runner_bootstrap_error, (
        "a failed session.bootstrap must be recorded on the server — "
        "without it the failure reaches no caller and session creation "
        "discovers it at whichever session.* verb asks first (#1033)"
    )
    assert "AppArmor confinement mismatch" in server.runner_bootstrap_error
    assert "RuntimeError" in server.runner_bootstrap_error, (
        "the exception TYPE must survive: an arg-less exception (a bare "
        "TimeoutError) stringifies to '' and would record nothing"
    )
    # The pre-existing side effects are contractual — keep them.
    assert server.ready_marks == 1
    assert server.terminations, (
        "the SessionTerminatedEvent path must still fire — recording the "
        "outcome is an ADDITION to the dispatch's contract, not a "
        "replacement for the visibility it already provided"
    )


def test_an_argless_bootstrap_failure_still_records_something() -> None:
    """``TimeoutError()`` is truthy and stringifies to ``""``.

    The 30 s bootstrap deadline is one of the real ways this fails, and a
    recorder that stored ``str(exc)`` would store the empty string — which
    reads downstream as "nothing known to be wrong", i.e. the defect.
    """
    from jaato_server.server.runner_spawn import dispatch_bootstrap_envelope

    server = _DispatchServer()
    server.runner_rpc = _FakeBootstrapRPC(exc=TimeoutError())

    dispatch_bootstrap_envelope(
        server=server, session_id="s", workspace_path=None, profile_name="",
    )

    assert server.runner_bootstrap_error
    assert "TimeoutError" in server.runner_bootstrap_error


def test_a_successful_bootstrap_records_no_failure() -> None:
    """Success clears the field rather than merely not setting it."""
    from jaato_server.server.runner_spawn import dispatch_bootstrap_envelope

    server = _DispatchServer()
    server.note_runner_bootstrap_outcome("a previous failure")
    server.runner_rpc = _FakeBootstrapRPC()

    dispatch_bootstrap_envelope(
        server=server, session_id="s", workspace_path=None, profile_name="",
    )

    assert server.runner_bootstrap_error is None


def test_a_spawn_that_never_populated_the_rpc_is_recorded_too() -> None:
    """The dispatch's other failure path — no ``runner_rpc`` at all.

    It returns early, so it is the path most likely to be missed; and it
    means exactly the same thing downstream (this runner hosts no
    session).
    """
    from jaato_server.server.runner_spawn import dispatch_bootstrap_envelope

    server = _DispatchServer()          # runner_rpc stays None
    dispatch_bootstrap_envelope(
        server=server, session_id="s", workspace_path=None, profile_name="",
    )
    assert server.runner_bootstrap_error


def test_the_recorder_contract_is_the_one_jaato_server_implements() -> None:
    """The doubles above are not inventing a surface.

    ``JaatoServer`` must carry the setter the dispatch calls and the
    reader the refusal consults, and success must clear a prior failure.
    """
    from jaato_server.server.core import JaatoServer

    srv = JaatoServer(workspace_path=None, session_id="contract-1033")
    assert srv.runner_bootstrap_error is None
    srv.note_runner_bootstrap_outcome("boom")
    assert srv.runner_bootstrap_error == "boom"
    srv.note_runner_bootstrap_outcome(None)
    assert srv.runner_bootstrap_error is None


# ----------------------------------------------------------------------
# The refusal: a hostless runner is asked nothing
# ----------------------------------------------------------------------


def _refuse(server: Any, session_id: str = "sess-1033") -> bool:
    """Call the real seam.

    Imported inside the call rather than at module scope on purpose: a
    missing seam must fail the tests that assert it, not the collection
    of the whole module — the two wire tests above are green before the
    fix and after it, and an ImportError would hide both.
    """
    from jaato_server.server.session_manager import initialize_or_refuse
    return initialize_or_refuse(server, session_id)


def test_a_hostless_runner_is_never_asked_for_its_context_usage() -> None:
    """THE REGRESSION.  A failed bootstrap refuses before ``initialize()``.

    ``initialize()`` is where the unguarded ``session_get_context_usage``
    lives, so "not reached" is the whole property: there is no ordering
    inside that function that could make the RPC safe, because the RPC is
    correct and the runner is dead.
    """
    server = _RecordingServer(bootstrap_error="ConfinementError: tid=42")

    assert _refuse(server) is False
    assert server.initialize_calls == 0, (
        "initialize() asks the runner for its context usage with no guard; "
        "reaching it with a hostless runner is #1033"
    )

    errors = [e for e in server.events
              if getattr(e, "error_type", None) == "RunnerBootstrapFailed"]
    assert len(errors) == 1, (
        "the refusal must name what actually failed — the whole cost of "
        "the defect was a client told 'session.get_context_usage failed'"
    )
    assert "ConfinementError: tid=42" in errors[0].error
    assert errors[0].recoverable is False


def test_a_healthy_bootstrap_initializes_exactly_as_before() -> None:
    """No bootstrap failure recorded → the seam is a pass-through.

    Covers the embedded and standalone-WS servers too: they dispatch no
    bootstrap at all, record nothing, and must be unaffected.
    """
    ok = _RecordingServer(bootstrap_error=None)
    assert _refuse(ok) is True
    assert ok.initialize_calls == 1
    assert ok.events == []

    failing = _RecordingServer(bootstrap_error=None)
    failing.initialize_result = False
    assert _refuse(failing) is False
    assert failing.initialize_calls == 1, (
        "initialize()'s own verdict must still be its own — the seam adds "
        "a precondition, it does not replace the decision"
    )


def test_an_empty_bootstrap_error_is_not_a_failure() -> None:
    """``""`` means "nothing recorded", not "failed with no message".

    The recorder normalises falsy to ``None``; this pins the reader's half
    so a future recorder that stores ``str(exc)`` for an arg-less
    exception cannot silently refuse every session on the host.
    """
    server = _RecordingServer(bootstrap_error="")
    assert _refuse(server) is True
    assert server.initialize_calls == 1


# ----------------------------------------------------------------------
# Structural: the funnel cannot walk around the precondition
# ----------------------------------------------------------------------


def _calls_in(func, attr: str) -> List[ast.Call]:
    """Every ``<expr>.<attr>(...)`` call in *func*."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


def _name_calls_in(func, name: str) -> List[ast.Call]:
    """Every bare ``name(...)`` call in *func*."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
    ]


def test_the_creation_funnel_settles_the_bootstrap_before_initialize() -> None:
    """``_construct_and_initialize_server`` reaches ``initialize`` only via the seam.

    A behavioural test cannot see a second call site added later, and this
    funnel is shared by session CREATE and by disk-RESTORE — both of which
    run against a runner whose bootstrap may have failed.
    """
    func = SessionManager._construct_and_initialize_server
    direct = _calls_in(func, "initialize")
    seam = _name_calls_in(func, "initialize_or_refuse")

    assert len(seam) == 1, (
        "the one session-creation funnel must consult the bootstrap "
        "outcome before initializing (#1033)"
    )
    assert direct == [], (
        "``server.initialize()`` is called directly in the creation "
        "funnel, so a runner whose session.bootstrap failed is initialized "
        "anyway and the failure resurfaces at session.get_context_usage "
        "(#1033)"
    )


def test_the_runner_predicate_is_left_alone() -> None:
    """``_require_ready_session`` still refuses, with the same words.

    The fix must not have been taken on the runner side.  A predicate that
    waits, retries or answers optimistically converts a visible refusal
    into a session reporting usage nobody measured.
    """
    from jaato_server.server.runner.rpc import RunnerRPC as _RPC

    src = textwrap.dedent(inspect.getsource(_RPC._require_ready_session))
    assert "session not bootstrapped on this runner" in src
    assert "no_host" in src
    for banned in ("sleep", "wait(", "retry", "while "):
        assert banned not in src, (
            f"``_require_ready_session`` gained {banned!r} — no_host is the "
            f"honest answer to the question as asked; #1033 is that the "
            f"question reaches a hostless runner at all"
        )
