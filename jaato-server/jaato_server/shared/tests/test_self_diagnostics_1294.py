"""Self-diagnostics (#1294): a live confinement re-probe of the CALLER'S OWN
session, kept apart from what the daemon already cached about it.

Three properties make this honest, mirroring the family this repeats
(#951's decision observability, #1283's memory rail, #1014's mode
predicates), and each is guarded by a reversion below:

**The request cannot name another session.**  ``DiagnosticsRequest`` adds
no session-naming field of its own -- it inherits ``Event.session_id``
like every event does, but that field is stamped by the router on
OUTGOING events and read by nothing on the way in.  The daemon always
answers for whichever session the connection is ATTACHED to, resolved
before this module is reached; a bogus value a client puts in the
request's inherited field is simply never consulted, which is the
behavioural guarantee this file asserts directly.

**Only the owner may look.**  ``may_view_diagnostics`` matches #1283's
``may_curate`` shape exactly, per #1294's own suggested default: the
workspace owner may see these facts, anyone may on an unowned workspace,
and an identity-less connection on an owned workspace may not.

**Cached and live are never merged into one verdict.**  A cached
``sandbox_mode`` reading "confined" is precisely what #1253 was filed
about -- so the answer keeps the daemon's own record (``sandbox_mode``,
``confinement_id``, ``runner_identity``) and the runner's live re-probe
(``probe``) as two fields a client can compare, never letting one quietly
become the other.  And a probe that could not run reports that honestly
(``ok=False``, ``enforced=False``) -- never a guessed verdict either way.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.events import DiagnosticsRequest, DiagnosticsResultEvent
from jaato_server.server.diagnostics_verbs import (
    answer_diagnostics_request,
    may_view_diagnostics,
)
from jaato_server.server.runner.bootstrap import probe_confinement_now
from jaato_server.server.runner.rpc import NAMED_METHOD_HANDLERS, RunnerRPC
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.tests.reversion import Reversion

_VERBS = "jaato-server/jaato_server/server/diagnostics_verbs.py"
_BOOTSTRAP = "jaato-server/jaato_server/server/runner/bootstrap.py"


# --------------------------------------------------------------- the wire shape


def test_diagnostics_request_declares_no_session_naming_field_of_its_own():
    """``DiagnosticsRequest`` adds nothing beyond ``request_id`` to the
    base ``Event`` shape.

    Every ``Event`` carries an inherited ``session_id`` -- stamped by the
    router on OUTGOING events (``SessionManager._emit_to_session``) so a
    cascade observer can attribute activity events, and read by nothing
    on the incoming side.  So the request has the field on the wire (a
    client could set it, meaninglessly) and no verb here declares its
    OWN one -- the actual boundary is behavioural, asserted below:
    whatever a caller puts in ``event.session_id`` is never read.
    """
    own_fields = set(DiagnosticsRequest.model_fields) - set(
        DiagnosticsRequest.__bases__[0].model_fields)
    assert own_fields == {"request_id"}


# ------------------------------------------------------- probe_confinement_now


def _fake_task_dir(tmp_path: Path, labels: Dict[int, Optional[str]]) -> str:
    """Build ``<dir>/<tid>/attr/current`` for each ``{tid: label}`` -- the
    same fabricated procfs shape #1023's own guard uses."""
    root = tmp_path / "task"
    for tid, label in labels.items():
        attr = root / str(tid) / "attr"
        attr.mkdir(parents=True, exist_ok=True)
        if label is not None:
            (attr / "current").write_text(label)
    return str(root)


def test_a_probe_that_cannot_read_proc_says_so_not_a_guess(tmp_path):
    """No ``attr/current`` to read at all: ``ok=False``, and NEITHER
    ``enforced`` nor ``confined`` is guessed ``True`` -- absence of
    evidence stays absence of evidence."""
    missing = str(tmp_path / "does-not-exist" / "attr" / "current")
    result = probe_confinement_now("jaato-ws-sess1", proc_attr_path=missing)
    assert result["ok"] is False
    assert result["enforced"] is False
    assert result["confined"] is False
    assert result["scan"] is None
    assert result["error"]


def test_a_live_uniformly_confined_process_reports_enforced(tmp_path):
    """The positive case: the probe genuinely can tell "confined, and the
    kernel is enforcing it" when that is what every thread reports."""
    attr = tmp_path / "attr_current"
    attr.write_text("jaato-ws-sess1 (enforce)\n")
    task_dir = _fake_task_dir(tmp_path, {
        201: "jaato-ws-sess1 (enforce)\n",
        202: "jaato-ws-sess1 (enforce)\n",
    })
    result = probe_confinement_now(
        "jaato-ws-sess1", proc_attr_path=str(attr), task_dir=task_dir)
    assert result["ok"] is True
    assert result["enforced"] is True
    assert result["confined"] is True
    assert result["current_mode"] == "enforce"
    assert result["scan"]["uniform"] is True
    assert result["scan"]["divergent"] == 0


def test_a_live_probe_names_its_divergent_threads(tmp_path):
    """The exact shape #1023 measured live: one thread outside the
    session's own profile, named by tid and label."""
    attr = tmp_path / "attr_current"
    attr.write_text("jaato-ws-sess1 (enforce)\n")
    task_dir = _fake_task_dir(tmp_path, {
        101: "unconfined",
        102: "jaato-ws-sess1 (enforce)\n",
    })
    result = probe_confinement_now(
        "jaato-ws-sess1", proc_attr_path=str(attr), task_dir=task_dir)
    assert result["ok"] is True
    assert result["scan"]["uniform"] is False
    assert result["scan"]["divergent"] == 1
    tids = [t["tid"] for t in result["scan"]["divergent_threads"]]
    assert 101 in tids


REVERSIONS_BOOTSTRAP = [
    Reversion(
        target=_BOOTSTRAP,
        find="""    except OSError as exc:
        return {
            "ok": False,
            "error": f"could not read {proc_attr_path}: {type(exc).__name__}: {exc}",
            "expected_profile": expected_profile,
            "current_profile": "",
            "current_mode": None,
            "enforced": False,
            "confined": False,
            "scan": None,
        }""",
        replace="""    except OSError as exc:
        return {
            "ok": False,
            "error": f"could not read {proc_attr_path}: {type(exc).__name__}: {exc}",
            "expected_profile": expected_profile,
            "current_profile": "",
            "current_mode": None,
            "enforced": True,
            "confined": False,
            "scan": None,
        }""",
        test="test_a_probe_that_cannot_read_proc_says_so_not_a_guess",
        because=(
            "a probe that could not read /proc reported enforced=True -- "
            "a guessed confinement verdict where absence of evidence is "
            "the only honest answer (#1014's posture, applied here)"
        ),
    ),
]


# ----------------------------------------------------------- the owner gate


@pytest.mark.parametrize("owner,user,allowed", [
    (None, None, True),               # unowned: anyone who can see it
    (None, "app:bob", True),
    ("app:alice", "app:alice", True),
    ("app:alice", "app:bob", False),
    ("app:alice", None, False),       # identity-less on an owned workspace
])
def test_only_the_owner_may_view_diagnostics(owner, user, allowed):
    assert may_view_diagnostics(owner, user) is allowed


class _Identity:
    def __init__(self, apparmor_profile: str = "", pid: int = 4242):
        self.apparmor_profile = apparmor_profile
        self._pid = pid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runner_pid": self._pid,
            "pool_served": True,
            "pool_slot_pid": self._pid,
            "cascade_driver_id": None,
            "apparmor_profile": self.apparmor_profile,
            "stale": False,
        }


class _Session:
    """A stand-in for the daemon's own ``Session`` record."""

    def __init__(
        self, *, sandbox_mode: Optional[str] = None,
        apparmor_profile: str = "", workspace_path: str = "/ws/a",
    ):
        self.sandbox_mode = sandbox_mode
        self.runner_identity = _Identity(apparmor_profile)
        self.workspace_path = workspace_path


class _ProbeServer:
    """A stand-in for ``JaatoServer.diagnostics_probe``."""

    def __init__(self, answer: Dict[str, Any]):
        self.answer = answer
        self.calls = 0

    def diagnostics_probe(self, *, timeout: float = 5.0) -> Dict[str, Any]:
        self.calls += 1
        return self.answer


def _runner_answered_probe(**probe_overrides: Any) -> Dict[str, Any]:
    probe = {
        "ok": True, "error": "", "expected_profile": "jaato-ws-a",
        "current_profile": "jaato-ws-a", "current_mode": "enforce",
        "enforced": True, "confined": True,
        "scan": {"scanned": 2, "matched": 2, "divergent": 0,
                  "unreadable": 0, "gone": 0, "uniform": True,
                  "route": "task_dir", "divergent_threads": [],
                  "unreadable_threads": []},
    }
    probe.update(probe_overrides)
    return {
        "probe": probe,
        "notebook_boundary_kind": None,
        "consumption": {"turns": 3},
        "protocol_version": "1.25",
    }


def test_a_refused_caller_never_reaches_the_probe():
    server = _ProbeServer(_runner_answered_probe())
    session = _Session(sandbox_mode="apparmor", apparmor_profile="jaato-ws-a")
    answer = answer_diagnostics_request(
        server, DiagnosticsRequest(request_id="r1"),
        session_id="s1", user_id="app:bob", owner="app:alice",
        session=session,
    )
    assert server.calls == 0, "a non-owner's request reached the probe"
    assert (answer.ok, answer.category) == (False, "not_owner")
    assert answer.request_id == "r1"
    assert answer.probe is None


def test_an_owner_gets_the_live_probe():
    server = _ProbeServer(_runner_answered_probe())
    session = _Session(sandbox_mode="apparmor", apparmor_profile="jaato-ws-a")
    answer = answer_diagnostics_request(
        server, DiagnosticsRequest(request_id="r2"),
        session_id="s1", user_id="app:alice", owner="app:alice",
        session=session,
    )
    assert server.calls == 1
    assert answer.ok is True
    assert answer.probe is not None and answer.probe["enforced"] is True
    assert answer.consumption == {"turns": 3}


def test_no_session_is_answered_under_the_callers_request_id():
    answer = answer_diagnostics_request(
        None, DiagnosticsRequest(request_id="r3"),
        session_id="", user_id=None, owner=None,
    )
    assert (answer.ok, answer.category, answer.request_id) == (
        False, "no_session", "r3")


REVERSIONS_GATE = [
    Reversion(
        target=_VERBS,
        find="    return owner is None or (user_id is not None and owner == user_id)",
        replace="    return True",
        test="test_only_the_owner_may_view_diagnostics",
        because="anyone who can see a workspace could read its confinement facts",
    ),
    Reversion(
        target=_VERBS,
        find="    elif not allowed:",
        replace="    elif False:",
        test="test_a_refused_caller_never_reaches_the_probe",
        because="the owner gate computed and never applied",
    ),
]


# ----------------------------------------- cached vs live, never conflated


def test_the_cached_sandbox_mode_is_never_overwritten_by_the_live_probe():
    """The #1253 shape, reproduced deliberately: a session's RECORD claims
    ``apparmor`` (enforced) while the live probe, this instant, finds the
    kernel applying only ``complain`` mode -- #1014's own "no boundary"
    finding, live.  The two must stay two facts: a client that read the
    LIVE value where the CACHED one belongs would be told a lie in the
    field named for the record."""
    server = _ProbeServer(_runner_answered_probe(
        enforced=False, current_mode="complain", current_profile="jaato-ws-a",
    ))
    session = _Session(sandbox_mode="apparmor", apparmor_profile="jaato-ws-a")
    answer = answer_diagnostics_request(
        server, DiagnosticsRequest(request_id="r4"),
        session_id="s1", user_id="app:alice", owner="app:alice",
        session=session,
    )
    # The cached claim is reported exactly as the record holds it --
    assert answer.sandbox_mode == "apparmor"
    assert answer.confinement_id == "jaato-ws-a"
    # -- and the live probe, which disagrees with it, is reported too,
    # never folded into (or replacing) the cached field.
    assert answer.probe["enforced"] is False
    assert answer.probe["current_mode"] == "complain"


def test_no_runner_still_answers_the_cached_facts():
    """A session with no runner (embedded, standalone WS) has nothing to
    probe, but the daemon's own cached facts are still a real answer --
    ``ok`` stays True, only ``probe`` is None, and ``category`` says why."""
    server = _ProbeServer({
        "ok": False, "category": "no_runner",
        "error": "this session has no runner subprocess to probe",
    })
    session = _Session(sandbox_mode=None, apparmor_profile="")
    answer = answer_diagnostics_request(
        server, DiagnosticsRequest(request_id="r5"),
        session_id="s1", user_id="app:alice", owner="app:alice",
        session=session,
    )
    assert answer.ok is True
    assert answer.probe is None
    assert answer.category == "no_runner"
    assert answer.sandbox_mode is None
    assert answer.runner_identity is not None  # still reported, even if empty


REVERSIONS_CONFLATION = [
    Reversion(
        target=_VERBS,
        find="""            sandbox_mode=sandbox_mode,
            consumption=probe_answer.get("consumption"),""",
        replace="""            sandbox_mode=(probe_answer.get("probe") or {}).get("current_mode") or sandbox_mode,
            consumption=probe_answer.get("consumption"),""",
        test="test_the_cached_sandbox_mode_is_never_overwritten_by_the_live_probe",
        because=(
            "the daemon's cached sandbox_mode silently replaced by the "
            "live probe's current_mode -- the exact conflation #1253 was "
            "filed about, reintroduced by the feature meant to prevent it"
        ),
    ),
]


# ------------------------------------------------------------- the audit line


def test_every_outcome_is_traced(caplog):
    caplog.set_level(logging.INFO, logger="jaato_server.server.diagnostics_verbs")

    server = _ProbeServer(_runner_answered_probe())
    session = _Session(sandbox_mode="apparmor", apparmor_profile="jaato-ws-a")

    answer_diagnostics_request(
        server, DiagnosticsRequest(request_id="r6"),
        session_id="s1", user_id="app:bob", owner="app:alice",
        session=session,
    )  # refused
    answer_diagnostics_request(
        server, DiagnosticsRequest(request_id="r7"),
        session_id="s1", user_id="app:alice", owner="app:alice",
        session=session,
    )  # allowed

    lines = [r.message for r in caplog.records if "[DIAGNOSTICS]" in r.message]
    assert len(lines) == 2, "both a refusal and an allowed probe must be traced"
    assert any("allowed=False" in ln for ln in lines)
    assert any("allowed=True" in ln for ln in lines)


REVERSIONS_AUDIT = [
    Reversion(
        target=_VERBS,
        find="""    _trace_probe(user_id, session_id, allowed, result)
    return result""",
        replace="    return result",
        test="test_every_outcome_is_traced",
        because=(
            "a live re-probe of a security boundary happened with no "
            "audit line at all -- #1294 asks that its own use never be "
            "silent, allowed or refused alike"
        ),
    ),
]


# ------------------------------------------------------- the daemon's one arm


class _Resolver:
    def __init__(self, owner):
        self.owner = owner

    def owner_for(self, path):
        return self.owner


def _manager(session, owner) -> "tuple[SessionManager, List[Any]]":
    sm = SessionManager.__new__(SessionManager)
    sent: List[Any] = []
    sm.get_session = lambda sid: session if sid == "s1" else None  # type: ignore[method-assign]
    sm._emit_to_client = lambda cid, ev: sent.append((cid, ev))  # type: ignore[method-assign]
    sm._app_secret_resolver = _Resolver(owner)
    return sm, sent


def test_handle_request_refuses_a_non_owner():
    server = _ProbeServer(_runner_answered_probe())
    session = _Session(sandbox_mode="apparmor", apparmor_profile="jaato-ws-a")
    session.server = server
    sm, sent = _manager(session, owner="app:alice")
    sm.handle_request("c1", "s1", DiagnosticsRequest(request_id="r8"),
                       user_id="app:bob")
    assert server.calls == 0
    [(cid, event)] = sent
    assert cid == "c1" and isinstance(event, DiagnosticsResultEvent)
    assert (event.ok, event.category) == (False, "not_owner")


def test_handle_request_lets_the_owner_through():
    server = _ProbeServer(_runner_answered_probe())
    session = _Session(sandbox_mode="apparmor", apparmor_profile="jaato-ws-a")
    session.server = server
    sm, sent = _manager(session, owner="app:alice")
    sm.handle_request("c1", "s1", DiagnosticsRequest(request_id="r9"),
                       user_id="app:alice")
    assert server.calls == 1
    [(_cid, event)] = sent
    assert isinstance(event, DiagnosticsResultEvent) and event.ok is True


def test_handle_request_always_uses_its_own_session_id_not_the_events():
    """``Event`` inherits a ``session_id`` field a client CAN set on the
    wire -- so the guarantee is behavioural: even when a request carries
    a bogus one naming a session this connection was never attached to,
    the router's OWN ``session_id`` argument (the connection's actual
    attached session, resolved before this module is reached) is what
    gets served, never anything read off the request body."""
    server = _ProbeServer(_runner_answered_probe())
    honest_session = _Session(sandbox_mode="apparmor", apparmor_profile="jaato-ws-a")
    honest_session.server = server
    sm, sent = _manager(honest_session, owner="app:alice")
    event = DiagnosticsRequest(request_id="r10", session_id="someone-elses-session")
    sm.handle_request("c1", "s1", event, user_id="app:alice")
    assert server.calls == 1, "the router's own session_id must be the one served"
    [(_cid, answered)] = sent
    assert answered.ok is True


# ------------------------------------------------------------- the runner side


class _Registry:
    def get_plugin(self, name: str) -> Any:
        return None


class _Runtime:
    def __init__(self, registry: Any):
        self.registry = registry


class _FakeSession:
    def __init__(self):
        self._runtime = _Runtime(_Registry())

    def get_consumption(self, detail: str) -> Dict[str, Any]:
        return {"turns": 1, "detail": detail}


class _Envelope:
    def __init__(self, profile_name: str):
        self.profile_name = profile_name


class _Host:
    def __init__(self, profile_name: str = "jaato-ws-sess1"):
        self.envelope = _Envelope(profile_name)
        self.session = _FakeSession()


def test_the_runner_handler_serves_its_own_session():
    rpc = RunnerRPC.__new__(RunnerRPC)
    rpc._session_host = _Host()
    rpc._session_lock = threading.Lock()
    ok, answer = rpc._handle_session_diagnostics({})
    assert ok is True
    assert answer["probe"]["expected_profile"] == "jaato-ws-sess1"
    assert answer["consumption"] == {"turns": 1, "detail": "summary"}
    assert answer["protocol_version"]


def test_the_runner_handler_refuses_with_no_host():
    rpc = RunnerRPC.__new__(RunnerRPC)
    rpc._session_host = None
    rpc._session_lock = threading.Lock()
    ok, answer = rpc._handle_session_diagnostics({})
    assert ok is False
    assert answer["stage"] == "no_host"


def test_session_diagnostics_is_a_named_runner_method():
    assert NAMED_METHOD_HANDLERS.get("session.diagnostics") == \
        "_handle_session_diagnostics"


REVERSIONS = (
    REVERSIONS_BOOTSTRAP + REVERSIONS_GATE + REVERSIONS_CONFLATION
    + REVERSIONS_AUDIT
)
