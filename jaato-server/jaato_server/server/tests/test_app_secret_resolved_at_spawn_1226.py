"""``app://`` secret references resolved at spawn by the owning application (#1226).

The keystone of the per-user-GitHub epic: a workspace ``.env`` carries a
REFERENCE (``GH_TOKEN=app://github``), and the daemon resolves it at every
session spawn by asking the application that OWNS the workspace, over the
#1074 bind channel.  These guards drive the real resolution path
(``JaatoServer._resolve_session_env`` + ``AppSecretResolver`` + a recording
transport) and the real WS-side transport/revocation/expiry machinery, and
assert the §6.1 rules the design makes load-bearing:

* an OWNED workspace receives the resolved value, asking only the application
  the owner is qualified under;
* an unowned workspace, an unreachable/refusing application, and a daemon with
  no resolver all DROP the reference — never forward the literal — with the
  strict ``?required`` form refusing the bootstrap instead;
* nothing resolved is persisted (the ``.env`` and the profile ``env:`` keep
  ``app://<name>``);
* an expiring value is due for a pre-expiry reload;
* revocation reloads only the calling application's own sessions.
"""

import os
import tempfile

import pytest

from jaato_server.server.core import JaatoServer, AppSecretResolutionError
from jaato_server.server.app_secret import AppSecretResolver, AppSecretAnswer
from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"
_WS = "jaato-server/jaato_server/server/websocket.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _server(tmp_path, env_lines, owner, transport, session_id="s1"):
    """A ``JaatoServer`` whose env file holds ``env_lines`` and whose app://
    resolver reports ``owner`` for the workspace and answers via ``transport``.
    """
    ws = str(tmp_path)
    envf = os.path.join(ws, ".env")
    with open(envf, "w") as fh:
        fh.write(env_lines)
    resolver = AppSecretResolver(owner_of=lambda p: owner, transport=transport)
    srv = JaatoServer(env_file=envf, provider=None, workspace_path=ws,
                      session_id=session_id)
    srv.set_app_secret_resolver(resolver)
    return srv, envf


def _recording_transport(answer):
    calls = []

    def transport(app_id, user, workspace, name, timeout):
        calls.append({"app_id": app_id, "user": user,
                      "workspace": workspace, "name": name})
        return answer

    transport.calls = calls
    return transport


# --------------------------------------------------------------------------
# resolution at spawn
# --------------------------------------------------------------------------

def test_owned_workspace_receives_resolved_value(tmp_path):
    transport = _recording_transport(
        AppSecretAnswer(status="ok", value="tok-123")
    )
    srv, _ = _server(tmp_path, "GH_TOKEN=app://github\nPLAIN=x\n",
                     "acme:alice", transport)
    srv._resolve_session_env()
    assert srv._session_env["GH_TOKEN"] == "tok-123"
    # unrelated variables are untouched
    assert srv._session_env["PLAIN"] == "x"


def test_asks_only_the_application_the_owner_is_qualified_under(tmp_path):
    transport = _recording_transport(AppSecretAnswer(status="ok", value="t"))
    srv, _ = _server(tmp_path, "GH_TOKEN=app://github\n", "acme:alice",
                     transport)
    srv._resolve_session_env()
    assert transport.calls == [{
        "app_id": "acme", "user": "alice",
        "workspace": str(tmp_path), "name": "github",
    }]


def test_an_unowned_workspace_resolves_nothing(tmp_path):
    transport = _recording_transport(AppSecretAnswer(status="ok", value="t"))
    srv, _ = _server(tmp_path, "GH_TOKEN=app://github\n", None, transport)
    srv._resolve_session_env()
    assert "GH_TOKEN" not in srv._session_env
    assert transport.calls == []  # the application was never asked


def test_unreachable_application_drops_the_variable(tmp_path, caplog):
    transport = _recording_transport(
        AppSecretAnswer(status="unreachable", detail="not connected")
    )
    srv, _ = _server(tmp_path, "GH_TOKEN=app://github\n", "acme:alice",
                     transport)
    with caplog.at_level("WARNING"):
        srv._resolve_session_env()
    # dropped, and NEVER left as the literal reference
    assert "GH_TOKEN" not in srv._session_env
    assert any("app://" in r.getMessage() and "dropped" in r.getMessage()
               for r in caplog.records)


def test_no_resolver_drops_the_reference(tmp_path):
    ws = str(tmp_path)
    envf = os.path.join(ws, ".env")
    with open(envf, "w") as fh:
        fh.write("GH_TOKEN=app://github\n")
    srv = JaatoServer(env_file=envf, provider=None, workspace_path=ws,
                      session_id="s")
    # no resolver injected (IPC / embedded posture)
    srv._resolve_session_env()
    assert "GH_TOKEN" not in srv._session_env


def test_required_form_refuses_the_bootstrap(tmp_path):
    transport = _recording_transport(
        AppSecretAnswer(status="denied", detail="no binding")
    )
    srv, _ = _server(tmp_path, "GH_TOKEN=app://github?required\n",
                     "acme:alice", transport)
    with pytest.raises(AppSecretResolutionError):
        srv._resolve_session_env()


def test_required_form_with_no_resolver_refuses_the_bootstrap(tmp_path):
    ws = str(tmp_path)
    envf = os.path.join(ws, ".env")
    with open(envf, "w") as fh:
        fh.write("GH_TOKEN=app://github?required\n")
    srv = JaatoServer(env_file=envf, provider=None, workspace_path=ws,
                      session_id="s")
    with pytest.raises(AppSecretResolutionError):
        srv._resolve_session_env()


def test_the_resolved_value_is_not_persisted(tmp_path):
    transport = _recording_transport(AppSecretAnswer(status="ok", value="tok"))
    srv, envf = _server(tmp_path, "GH_TOKEN=app://github\n", "acme:alice",
                        transport)
    srv._resolve_session_env()
    assert srv._session_env["GH_TOKEN"] == "tok"
    # the .env on disk keeps the REFERENCE, not the secret
    on_disk = open(envf).read()
    assert "app://github" in on_disk
    assert "tok" not in on_disk


def test_expiry_is_recorded_for_the_reload(tmp_path):
    transport = _recording_transport(
        AppSecretAnswer(status="ok", value="tok",
                        expires_at="2099-01-01T00:00:00+00:00")
    )
    srv, _ = _server(tmp_path, "GH_TOKEN=app://github\n", "acme:alice",
                     transport)
    srv._resolve_session_env()
    assert srv._app_secret_expiries == {"GH_TOKEN": "2099-01-01T00:00:00+00:00"}


# --------------------------------------------------------------------------
# the WS-side transport, revocation and correlation
# --------------------------------------------------------------------------

def _ws():
    from jaato_server.server.websocket import JaatoWSServer
    return JaatoWSServer(host="localhost", port=0)


def _fake_conn(app_id, kind):
    from jaato_server.server.websocket import ClientConnection
    return ClientConnection(
        websocket=object(), client_id=f"c-{app_id}", connected_at="now",
        subscriptions=set(), app_id=app_id, auth_kind=kind,
    )


def test_transport_asks_no_other_application():
    from jaato_server.server.websocket import AUTH_KIND_APP
    ws = _ws()
    ws._clients["c-acme"] = _fake_conn("acme", AUTH_KIND_APP)
    # An application that is not connected has no bind channel to ask.
    assert ws._app_connection_for("acme") == "c-acme"
    assert ws._app_connection_for("other") is None
    answer = ws._resolve_app_secret_over_bind_channel(
        "other", "bob", "/ws", "github", 0.1,
    )
    assert answer.status == "unreachable"
    assert not answer.ok


def test_a_result_resolves_its_pending_request():
    import concurrent.futures
    from jaato_sdk.events import SecretResolveResultEvent
    ws = _ws()
    fut: concurrent.futures.Future = concurrent.futures.Future()
    ws._pending_secret_resolves["r1"] = fut
    ws._deliver_secret_result(
        SecretResolveResultEvent(request_id="r1", status="ok", value="tok")
    )
    assert fut.done()
    assert fut.result().value == "tok"
    # request_id gone; a late/duplicate answer is a no-op
    assert "r1" not in ws._pending_secret_resolves
    ws._deliver_secret_result(
        SecretResolveResultEvent(request_id="r1", status="ok", value="late")
    )


def test_answer_translation_downgrades_ok_without_value():
    from jaato_server.server.websocket import _answer_from_result
    from jaato_sdk.events import SecretResolveResultEvent
    ok = _answer_from_result(
        SecretResolveResultEvent(request_id="r", status="ok", value="v",
                                 expires_at="2099-01-01T00:00:00+00:00")
    )
    assert ok.ok and ok.value == "v" and ok.expires_at.startswith("2099")
    empty = _answer_from_result(
        SecretResolveResultEvent(request_id="r", status="ok", value=None)
    )
    assert empty.status == "error" and not empty.ok
    denied = _answer_from_result(
        SecretResolveResultEvent(request_id="r", status="denied", detail="no")
    )
    assert denied.status == "denied" and denied.detail == "no"


def test_reload_is_scoped_to_the_calling_application():
    from jaato_sdk.events import SecretReloadRequest

    class _SM:
        def __init__(self):
            self.reloaded_for = []

        def reload_owner_sessions(self, qualified):
            self.reloaded_for.append(qualified)
            return 2

    class _Router:
        def __init__(self, sm):
            self._session_manager = sm

    sm = _SM()
    ws = _ws()
    ws._command_router = _Router(sm)
    result = ws._reload_owner_sessions(
        "acme", SecretReloadRequest(request_id="r", user="alice")
    )
    assert result.status == "ok" and result.reloaded == 2
    # qualified with the CALLING application's id, never the request's claim
    assert sm.reloaded_for == ["acme:alice"]


def test_reload_from_non_app_connection_is_denied():
    from jaato_sdk.events import SecretReloadRequest, serialize_event
    ws = _ws()
    msg = serialize_event(SecretReloadRequest(request_id="r", user="alice"))
    denied = ws._secret_reload_denied(msg)
    assert denied.status == "denied" and denied.request_id == "r"


# --------------------------------------------------------------------------
# SessionManager: expiry sweep + owner-scoped revocation
# --------------------------------------------------------------------------

class _FakeServer:
    def __init__(self, expiries):
        self._app_secret_expiries = dict(expiries)


class _FakeSession:
    def __init__(self, created_by=None, workspace_path=None, expiries=None):
        self.created_by = created_by
        self.workspace_path = workspace_path
        self.server = _FakeServer(expiries or {})


def _sm():
    from jaato_server.server.session_manager import SessionManager
    return SessionManager()


def test_a_near_expiry_session_is_due_for_reload():
    from jaato_server.server.session_manager import SessionManager
    # margin default 300s; a token expiring at t+100 is due at t; one at
    # t+10000 is not.
    now = 1_000_000.0
    from datetime import datetime, timezone
    soon = datetime.fromtimestamp(now + 100, timezone.utc).isoformat()
    far = datetime.fromtimestamp(now + 10_000, timezone.utc).isoformat()
    due = _FakeSession(expiries={"GH_TOKEN": soon})
    not_due = _FakeSession(expiries={"GH_TOKEN": far})
    none = _FakeSession(expiries={})
    assert SessionManager._session_app_secret_due(due, now, 300.0) is True
    assert SessionManager._session_app_secret_due(not_due, now, 300.0) is False
    assert SessionManager._session_app_secret_due(none, now, 300.0) is False


def test_expiry_sweep_reloads_only_due_sessions(monkeypatch):
    from datetime import datetime, timezone
    now = 2_000_000.0
    soon = datetime.fromtimestamp(now + 50, timezone.utc).isoformat()
    far = datetime.fromtimestamp(now + 99_999, timezone.utc).isoformat()
    sm = _sm()
    sm._sessions = {
        "due": _FakeSession(expiries={"GH_TOKEN": soon}),
        "later": _FakeSession(expiries={"GH_TOKEN": far}),
        "plain": _FakeSession(expiries={}),
    }
    reloaded = []
    monkeypatch.setattr(
        sm, "reload_session_env",
        lambda sid: reloaded.append(sid) or {"found": True, "was_processing": False},
    )
    sm._sweep_app_secret_expiry(now=now)
    assert reloaded == ["due"]


def test_reload_owner_sessions_scopes_by_owner(monkeypatch):
    sm = _sm()
    # resolver so the workspace-owner half of the rule is live
    sm._app_secret_resolver = AppSecretResolver(
        owner_of=lambda p: "acme:alice" if p == "/ws-alice" else "acme:bob",
        transport=lambda *a: AppSecretAnswer(status="ok", value="x"),
    )
    sm._sessions = {
        "created": _FakeSession(created_by="acme:alice", workspace_path="/other"),
        "owns_ws": _FakeSession(created_by=None, workspace_path="/ws-alice"),
        "bobs": _FakeSession(created_by="acme:bob", workspace_path="/ws-bob"),
    }
    reloaded = []
    monkeypatch.setattr(
        sm, "reload_session_env",
        lambda sid: reloaded.append(sid) or {"found": True},
    )
    count = sm.reload_owner_sessions("acme:alice")
    assert set(reloaded) == {"created", "owns_ws"}
    assert count == 2


# --------------------------------------------------------------------------
# the resolver is installed BEFORE the env is resolved, on every spawn path
# --------------------------------------------------------------------------

def test_construct_installs_resolver_before_resolving_env():
    """``_construct_and_initialize_server`` — the ONE seam every spawn path
    (WS create, cascade stage, wake, revive) shares — sets the app:// resolver
    on the server BEFORE it resolves the env, or the very first resolution
    would see no resolver and drop every reference.
    """
    import ast
    import inspect
    from jaato_server.server.session_manager import SessionManager

    src = inspect.getsource(SessionManager._construct_and_initialize_server)
    tree = ast.parse(_dedent(src))
    set_line = _resolve_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "set_app_secret_resolver":
                set_line = node.lineno
            elif node.func.attr == "_resolve_session_env":
                _resolve_line = node.lineno
    assert set_line is not None, "set_app_secret_resolver is never called"
    assert _resolve_line is not None, "_resolve_session_env is never called"
    assert set_line < _resolve_line, (
        "the resolver must be installed before the env is resolved"
    )


def _dedent(src):
    import textwrap
    return textwrap.dedent(src)


REVERSIONS = [
    Reversion(
        target=_CORE,
        find="        self._session_env.pop(key, None)\n        if required:",
        replace="        if required:",
        because=(
            "an unresolved app:// reference must be DROPPED from the env, "
            "never left as the literal — a literal GH_TOKEN=app://github in a "
            "subprocess 401s confusingly (§6.1)"
        ),
        test="test_unreachable_application_drops_the_variable",
    ),
    Reversion(
        target=_CORE,
        find="        if required:\n            raise AppSecretResolutionError(",
        replace="        if False:\n            raise AppSecretResolutionError(",
        because=(
            "the strict app://name?required form must turn a resolution "
            "failure into a bootstrap refusal, not a silent drop (§6.1)"
        ),
        test="test_required_form_refuses_the_bootstrap",
    ),
    Reversion(
        target=_WS,
        find="            if conn.auth_kind == AUTH_KIND_APP and conn.app_id == app_id:",
        replace="            if conn.auth_kind == AUTH_KIND_APP:",
        because=(
            "the daemon must ask ONLY the application the owner is qualified "
            "under; matching any app connection would let app-b be asked "
            "about app-a's user (§6.1)"
        ),
        test="test_transport_asks_no_other_application",
    ),
    Reversion(
        target=_SM,
        find="            if stamp >= dt.timestamp() - margin:",
        replace="            if False:",
        because=(
            "a session whose resolved app:// value is within the refresh "
            "margin of expiry must be due for a pre-expiry reload (§6.3)"
        ),
        test="test_a_near_expiry_session_is_due_for_reload",
    ),
    Reversion(
        target=_WS,
        find='        qualified = f"{app_id}:{event.user}"',
        replace='        qualified = event.user',
        because=(
            "secret.reload must be scoped to the CALLING application by "
            "qualifying the user with the connection's app_id, so one "
            "application cannot reload another's user (§6.4)"
        ),
        test="test_reload_is_scoped_to_the_calling_application",
    ),
]
