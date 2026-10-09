"""A confined session runs only on its runner's own word that it is confined (#1253).

#1100 and #1253 measured, on a live daemon, a WS session whose first runner
(a pre-warm pool slot) spawned ~0.4-3 s BEFORE the session's AppArmor
profile was provisioned.  The slot bootstrapped with an empty
``profile_name``, skipped self-confinement, and served the session
unconfined for 27 minutes, while the post-init hook provisioned the profile
afterwards and recorded ``sandbox_mode: apparmor``.  The paths have since
been ordered one by one (provision, then spawn; refuse a provisioning or
spawn failure).  What was still missing is the END-TO-END check: nothing
asked the runner what it wears, so any path that skipped the hook's spawn,
raised inside it, or spawned elsewhere still reached ``initialize()``.

The invariant pinned here, through the REAL WS pre-init hook, the REAL
hook runner (which swallows a hook's exception), the REAL
``dispatch_bootstrap_envelope`` and the REAL ``initialize_or_refuse``:

1. provisioning happens before the spawn, and the spawn carries the profile;
2. a confinement-required session initializes only when its runner reports
   (in its ``session.bootstrap`` answer) the label it was provisioned;
3. a runner reporting any other label -- ``unconfined`` above all -- is
   refused, ``RunnerBootstrapFailed``;
4. a hook that returns before spawning, or raises after deciding, leaves a
   refused session, never an unconfined one;
5. the record's ``sandbox_mode`` is the mode the runner reported.

**No kernel here.**  ``AppArmorManager`` and the runner's answer are
fabricated, as in every confinement guard in this tree (#1023 / #1033 /
#1100).  What this proves is that the framework refuses to run a
confinement-wanting session without positive evidence; that the kernel
then enforces is verified only on an enforcing host
(``tests/integration/test_phase2_multitenant_apparmor.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"
_WS = "jaato-server/jaato_server/server/websocket.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_EVIDENCE = "jaato-server/jaato_server/server/confinement_evidence.py"

REVERSIONS = [
    # The door: ``runner_bootstrap_error`` stops consulting the evidence,
    # so a runner that wears nothing is initialized.
    Reversion(
        target=_CORE,
        find="        return self._runner_bootstrap_error or self.confinement_shortfall",
        replace="        return self._runner_bootstrap_error  # #1253 reversion",
        test="test_a_runner_reporting_unconfined_is_refused",
        because=(
            "a confinement-required session whose runner reports "
            "'unconfined' is initialized again, served with no boundary"
        ),
    ),
    # The comparison: any reported label is taken as the boundary.
    Reversion(
        target=_EVIDENCE,
        find="    if not _wears(backend, label, raw):\n        return (",
        replace="    if False:  # #1253 reversion\n        return (",
        test="test_a_runner_reporting_unconfined_is_refused",
        because="the runner's reported label is no longer compared with the boundary",
    ),
    # The decision is recorded before the hook can exit: moved back below
    # the daemon-loop check, the no-loop exit leaves an unconfined session.
    Reversion(
        target=_WS,
        find="            confinement_required = _ws_confinement_required(ws_server, server)\n",
        replace="            confinement_required = _ws_confinement_available(ws_server)\n",
        test="test_a_hook_that_never_spawns_leaves_a_refused_session",
        because=(
            "a WS hook that returns before spawning (no daemon loop) again "
            "leaves a session that initializes with no runner and no boundary"
        ),
    ),
    # The runner stops reporting what it wears.
    Reversion(
        target=_RPC,
        find=(
            "            # it provisioned (``server.confinement_evidence``).\n"
            '            "confinement": {"label": _own_label()},'
        ),
        replace=(
            "            # it provisioned (``server.confinement_evidence``).\n"
            '            "confinement": None,  # #1253 reversion'
        ),
        test="test_the_runner_reports_the_label_it_wears",
        because="the runner's bootstrap answer no longer says what it wears",
    ),
    # The record is written from the runner's report, not the render.
    Reversion(
        target=_EVIDENCE,
        find="    return reported if isinstance(reported, str) and reported else derived",
        replace="    return derived  # #1253 reversion",
        test="test_the_record_says_the_mode_the_runner_reported",
        because="the session record again states the provisioned mode, not the worn one",
    ),
]


# ---------------------------------------------------------------- fakes


class _FakeAppArmor:
    """The ``AppArmorManager`` surface the WS hook reaches, recording order."""

    def __init__(self, events: List[str]) -> None:
        self._events = events
        self._ids: Dict[str, str] = {}

    def is_available(self) -> bool:
        return True

    def confinement_id_for_boundary(self, workspace_path: str, **kwargs: Any) -> str:
        return "ws-1253-0123456789ab"

    def provision_profile(self, session_id: str, workspace_path: str, **kwargs: Any) -> bool:
        self._events.append("provision")
        self._ids[session_id] = kwargs.get("confinement_id") or session_id
        return True

    def get_profile_name(self, session_id: str) -> str:
        return f"jaato-ws-{self._ids.get(session_id, session_id)}"


class _FakeAdapter:
    def __init__(self, loop: Any) -> None:
        self._event_loop = loop


class _FakeRPC:
    """A runner answering ``session.bootstrap`` with a given worn label."""

    def __init__(self, events: List[str], label: Optional[str]) -> None:
        self._events = events
        self._label = label
        self.envelopes: List[Any] = []

    def bootstrap_session_threadsafe(self, envelope: Any, timeout: float = 0) -> Dict[str, Any]:
        self._events.append("bootstrap")
        self.envelopes.append(envelope)
        return {"ok": True, "confinement": {"label": self._label}}


class _Envelope:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _ws_server(root: str, events: List[str], loop: Any = "<loop>") -> Any:
    from jaato_server.server.websocket import JaatoWSServer
    from jaato_server.server.ws_tickets import AppCredentialStore
    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._clients = {}
    ws._app_managers = {}
    ws._app_provisioners = {}
    ws._app_credentials = AppCredentialStore({})
    ws._apparmor = _FakeAppArmor(events)
    ws._cgroups = None
    ws._workspace_root = root
    ws._event_sink_adapter = _FakeAdapter(loop)
    return ws


class _Router:
    def __init__(self, sm: Any) -> None:
        self._session_manager = sm


def _session_manager(ws: Any) -> Any:
    """A real SessionManager with the WS hooks registered on it."""
    from jaato_server.server.session_manager import SessionManager
    from jaato_server.server.websocket import JaatoWSServer
    sm = SessionManager.__new__(SessionManager)
    sm._pre_initialize_hooks = []
    sm._session_hooks = []
    sm.add_pre_initialize_hook = sm._pre_initialize_hooks.append  # type: ignore[assignment]
    sm.add_session_hook = sm._session_hooks.append  # type: ignore[assignment]
    JaatoWSServer.set_command_router(ws, _Router(sm))
    return sm


def _server(tmp_path: Any, sid: str) -> Any:
    """A real JaatoServer, never initialized for real: ``initialize`` records."""
    from jaato_server.server.core import JaatoServer
    server = JaatoServer(workspace_path=str(tmp_path), session_id=sid)
    server.initialized = []  # type: ignore[attr-defined]
    server.initialize = lambda: server.initialized.append(True) or True  # type: ignore[method-assign]
    server.emitted = []  # type: ignore[attr-defined]
    server.emit = server.emitted.append  # type: ignore[method-assign]
    return server


def _drive(tmp_path: Any, *, label: Optional[str], loop: Any = "<loop>",
           plugin_rules_raise: bool = False) -> Dict[str, Any]:
    """One WS session creation, from the pre-init hooks to ``initialize_or_refuse``."""
    from jaato_server.server import runner_spawn
    from jaato_server.server.session_manager import initialize_or_refuse

    root = tmp_path / "ws_root"
    workspace = root / "ws"
    workspace.mkdir(parents=True)
    events: List[str] = []
    ws = _ws_server(str(root), events, loop)
    sm = _session_manager(ws)
    sid = "20261009_120000"
    server = _server(workspace, sid)
    rpc = _FakeRPC(events, label)
    spawns: List[Dict[str, Any]] = []

    def fake_spawn(**kwargs: Any) -> None:
        events.append("spawn")
        spawns.append(kwargs)
        kwargs["server"]._runner_rpc = rpc  # what set_runner_rpc stores

    def fake_envelope(**kwargs: Any) -> Any:
        return _Envelope(**kwargs)

    def plugin_rules(**kwargs: Any) -> Any:
        if plugin_rules_raise:
            raise RuntimeError("plugin rules exploded")
        return None

    with patch.object(runner_spawn, "spawn_session_runner", fake_spawn), \
         patch.object(runner_spawn, "build_session_envelope", fake_envelope), \
         patch("jaato_server.server.apparmor.resolve_plugin_apparmor_rules", plugin_rules), \
         patch.object(runner_spawn, "_emit_bootstrap_terminated", lambda **k: None), \
         patch.object(type(server), "_emit_tool_id_registry_from_schemas", lambda self: None):
        sm._run_pre_initialize_hooks(server, sid, str(workspace), None)
        ok = initialize_or_refuse(server, sid)
    return {"ok": ok, "events": events, "spawns": spawns, "server": server,
            "rpc": rpc, "ws": ws}


def _profile(result: Dict[str, Any]) -> str:
    return result["spawns"][0]["profile_name"]


# ---------------------------------------------------------------- tests


def test_provisioning_happens_before_the_spawn(tmp_path: Any) -> None:
    result = _drive(tmp_path, label="jaato-ws-ws-1253-0123456789ab (enforce)")
    assert result["events"] == ["provision", "spawn", "bootstrap"]
    assert _profile(result) == "jaato-ws-ws-1253-0123456789ab"
    assert result["spawns"][0]["disable_confine"] is False
    assert result["rpc"].envelopes[0].confinement_required is True


def test_a_runner_wearing_its_profile_is_initialized(tmp_path: Any) -> None:
    result = _drive(tmp_path, label="jaato-ws-ws-1253-0123456789ab (enforce)")
    assert result["ok"] is True
    assert result["server"].initialized == [True]
    assert result["server"].runner_bootstrap_error is None


def test_a_runner_reporting_unconfined_is_refused(tmp_path: Any) -> None:
    """The live #1253 state: a slot that wears nothing.  Refused by name."""
    result = _drive(tmp_path, label="unconfined")
    assert result["ok"] is False
    assert result["server"].initialized == []
    errors = [e for e in result["server"].emitted
              if getattr(e, "error_type", "") == "RunnerBootstrapFailed"]
    assert errors and "unconfined" in errors[0].error
    assert "jaato-ws-ws-1253-0123456789ab" in errors[0].error


def test_a_runner_wearing_another_profile_is_refused(tmp_path: Any) -> None:
    result = _drive(tmp_path, label="jaato-ws-someone-else (enforce)")
    assert result["ok"] is False
    assert result["server"].initialized == []


def test_a_runner_that_reports_nothing_is_refused(tmp_path: Any) -> None:
    result = _drive(tmp_path, label=None)
    assert result["ok"] is False
    assert result["server"].initialized == []


def test_a_hook_that_never_spawns_leaves_a_refused_session(tmp_path: Any) -> None:
    """No daemon loop: the hook returns before provisioning or spawning.

    Before #1253's fix this was the unconfined path -- no runner, in-process
    tools, and the post-init hook then recorded ``apparmor``.
    """
    result = _drive(tmp_path, label="unconfined", loop=None)
    assert result["spawns"] == []
    assert result["ok"] is False
    assert result["server"].initialized == []


def test_a_hook_that_raises_after_deciding_leaves_a_refused_session(tmp_path: Any) -> None:
    """The hook runner swallows a hook's exception; the session must not run."""
    result = _drive(tmp_path, label="unconfined", plugin_rules_raise=True)
    assert result["spawns"] == []
    assert result["ok"] is False
    assert result["server"].initialized == []


def test_the_record_says_the_mode_the_runner_reported(tmp_path: Any) -> None:
    from jaato_server.server.confinement_evidence import recorded_sandbox_mode
    result = _drive(tmp_path, label="jaato-ws-ws-1253-0123456789ab (complain)")
    assert result["ok"] is True
    # The render said enforce; the kernel said complain.  The record follows
    # the kernel (#1014).
    assert recorded_sandbox_mode(result["server"], "apparmor") == "apparmor-complain"


def test_an_unconfined_session_needs_no_evidence(tmp_path: Any) -> None:
    """No AppArmor wanted: nothing changes (no report, initialize runs)."""
    from jaato_server.server.core import JaatoServer
    from jaato_server.server.session_manager import initialize_or_refuse
    server = JaatoServer(workspace_path=str(tmp_path), session_id="s")
    server.initialize = lambda: True  # type: ignore[method-assign]
    assert server.runner_bootstrap_error is None
    assert initialize_or_refuse(server, "s") is True


def test_the_runner_reports_the_label_it_wears() -> None:
    """``session.bootstrap``'s answer carries the runner's own attr/current."""
    import threading

    from jaato_server.server.runner import envelope as envelope_mod
    from jaato_server.server.runner import rpc as rpc_mod
    from jaato_server.server.runner import session as session_mod

    class _Host:
        is_ready = True
        session_id = "s"
        runtime = None

    runner = rpc_mod.RunnerRPC.__new__(rpc_mod.RunnerRPC)
    runner._session_lock = threading.Lock()
    runner._session_host = None
    with patch.object(envelope_mod.SessionInitEnvelope, "from_dict",
                      classmethod(lambda cls, args: object())), \
         patch.object(session_mod, "bootstrap_session", lambda env, **k: _Host()), \
         patch("jaato_server.shared.lsm_label.read_own_context",
               lambda *a, **k: "jaato-ws-x (enforce)"):
        ok, answer = runner._handle_session_bootstrap({})
    assert ok is True
    assert answer["confinement"] == {"label": "jaato-ws-x (enforce)"}
