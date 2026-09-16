"""Attribution survives the joint between the transport and the record.

Two transports now authenticate their clients — IPC from ``SO_PEERCRED``
and WS from a bound ticket (#1074) — and each has a suite proving its own
``get_client_user`` returns the right string:
``test_ipc_peer_entitlement.py`` and ``test_identity_at_connect_1074.py``.
A third family (``test_session_envelope.py``, the session serializer's
tests) proves ``created_by`` ROUND-TRIPS once something has set it.

Neither end tested the JOINT.  The chain is one spine with two heads —

    sink.get_client_user(client_id)          <- per transport
        -> CommandRouter                     <- transport-agnostic
        -> Session.created_by / user_id
        -> the session record, the ledger, the DECISION line, user.id

— and ``CommandRouter`` is the single consumer that asks.  So a correct
producer and a correct record could sit either side of a consumer that
never carried the value, and every existing test would still pass.  It
did: ``session.default`` reached ``create_session`` with no ``created_by``
at all, while its two siblings twelve lines away in the same file both
read the sink.  ``_create_session_impl`` defaults the parameter to
``None``, so a session opened through ``IPCClient.get_default_session()``
was anonymous in all four artefacts on BOTH transports.

WHY THE SINKS ARE REAL.  A fake sink returning ``"ana"`` would assert
that ``CommandRouter`` passes along whatever it is handed, which is a
statement about ``CommandRouter`` and not about attribution.  These build
a real ``JaatoIPCServer`` holding a fabricated peer and a real
``JaatoWSServer`` holding a ticket-resolved connection, so the string
under assertion is the one each transport actually derives.

WHAT IS FAKE, and why that is the right half to fake: ``SessionManager``,
which records the kwargs it is called with.  The record/ledger/trace
half is covered by the round-trip suites above; what was missing is
whether the value ever ARRIVES, and that is a call-argument question.

Adding a transport is one row in :data:`_TRANSPORTS`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from jaato_sdk.events import CommandRequest, PermissionResponseRequest
from server.command_router import CommandRouter
from server.websocket import HAS_WEBSOCKETS
from shared.peer_identity import PeerCredentials
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_ROUTER = "jaato-server/server/command_router.py"
_MANAGER = "jaato-server/server/session_manager.py"

#: The two halves of the joint fail differently, so each is reverted on
#: its own.  An earlier draft aimed both at the same parametrized case and
#: the meta-guard rejected it: the router half was replaced with text that
#: left the original keyword in place (a duplicate-argument SyntaxError,
#: which is not the defect), and the manager half named a test whose
#: ``SessionManager`` is a double -- so sabotaging the real method changed
#: nothing it could see.  A reversion has to break the thing the named
#: test actually executes.
REVERSIONS = [
    Reversion(
        target=_ROUTER,
        find="""        default_session_id = self._session_manager.get_or_create_default(
            client_id, workspace_path=workspace_path,
            # The transport's authenticated user, read HERE for the same
            # reason ``session.new`` and the post-auth create read it here:
            # the sink is the only thing that knows, and the event body
            # must never be able to claim it (#859).
            created_by=self._event_sink.get_client_user(client_id),
        )""",
        replace="""        default_session_id = self._session_manager.get_or_create_default(
            client_id, workspace_path=workspace_path
        )""",
        test="test_the_default_session_is_attributed[ipc-ana]",
        because=(
            "session.default stops asking the sink, so a session opened "
            "through IPCClient.get_default_session() is anonymous in the "
            "record, the ledger and the DECISION line on every transport"
        ),
    ),
    Reversion(
        target=_MANAGER,
        find="""        return self.create_session(
            client_id, workspace_path=workspace_path,
            created_by=created_by,
        )""",
        replace="""        return self.create_session(
            client_id, workspace_path=workspace_path,
        )""",
        test="test_the_manager_forwards_the_identity_it_was_given",
        because=(
            "the manager accepts the identity the router read and drops it "
            "on the floor -- the silent-parameter-default shape, one layer "
            "below the one the router reversion covers"
        ),
    ),
]


# --------------------------------------------------------------- the sinks

def _ipc_sink() -> Tuple[Any, str, str]:
    """A real ``JaatoIPCServer`` holding one connection with a peer.

    No socket is bound: everything under test reads ``self._clients``,
    and binding one would make this a test about the filesystem.  Mirrors
    ``test_ipc_peer_entitlement._ipc_server_with_client``.
    """
    from server.ipc import IPCClientConnection, JaatoIPCServer

    server = JaatoIPCServer(socket_path="/tmp/does-not-need-to-exist.sock")
    server._clients["ipc_1"] = IPCClientConnection(
        reader=None, writer=None, client_id="ipc_1",
        session_id=None, connected_at="now",
        peer=PeerCredentials(uid=1000, gid=1000, username="ana"),
    )
    return server, "ipc_1", "ana"


def _ws_sink() -> Tuple[Any, str, str]:
    """A real ``JaatoWSServer`` behind its ``EventSink`` adapter.

    The ticket is minted through the real registry and redeemed through
    the real ``_resolve_connection_auth``, so ``acme:alice`` is the
    qualified identity the daemon derives rather than one written here.
    """
    from unittest.mock import MagicMock

    from server.websocket import (
        AUTH_KIND_TICKET, ClientConnection, JaatoWSServer,
        WSEventSinkAdapter,
    )
    from server.ws_tickets import AppCredentialStore

    app_token = "app-credential-0123456789abcdef"
    srv = JaatoWSServer(
        host="127.0.0.1", port=0,
        required_token="shared-bearer-token",
        app_credentials=AppCredentialStore({"acme": app_token}),
    )
    ticket, _expires = srv._ticket_registry.bind(app_id="acme", user="alice")

    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {ticket}"}
    request.path = "/"
    websocket = MagicMock()
    websocket.request = request

    auth = srv._resolve_connection_auth(websocket)
    assert auth is not None and auth.kind == AUTH_KIND_TICKET, auth
    srv._clients["ws_1"] = ClientConnection(
        websocket=websocket, client_id="ws_1",
        connected_at="2026-01-01T00:00:00+00:00", subscriptions=set(),
        user_id=auth.user, app_id=auth.app_id, auth_kind=auth.kind,
    )
    return WSEventSinkAdapter(srv), "ws_1", "acme:alice"


#: ``(id, factory)``.  The id carries the expected identity so a failing
#: row names the transport AND what it should have produced.
_TRANSPORTS = [
    pytest.param(_ipc_sink, id="ipc-ana"),
    pytest.param(
        _ws_sink, id="ws-acme:alice",
        marks=pytest.mark.skipif(
            not HAS_WEBSOCKETS, reason="websockets package not installed",
        ),
    ),
]


# ------------------------------------------------------- the recorded half

class _RecordingManager:
    """A ``SessionManager`` that answers, and remembers how it was asked.

    Only the surface ``CommandRouter`` reaches on these three paths; an
    attribute it wants and does not find is a real signal that the router
    grew a dependency this test is no longer exercising.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def _record(self, name: str, kwargs: Dict[str, Any]) -> None:
        self.calls.append((name, kwargs))

    def kwargs_of(self, name: str) -> Dict[str, Any]:
        matching = [kw for n, kw in self.calls if n == name]
        assert matching, (
            f"{name} was never called -- the router took a different path, "
            f"so this case asserts nothing. Calls: "
            f"{[n for n, _ in self.calls]}"
        )
        return matching[-1]

    # -- the surface ------------------------------------------------------
    def get_client_session(self, client_id: str) -> None:
        return None

    def create_session(self, *args: Any, **kwargs: Any) -> str:
        self._record("create_session", kwargs)
        return "sess-1"

    def get_or_create_default(self, *args: Any, **kwargs: Any) -> str:
        self._record("get_or_create_default", kwargs)
        return "sess-1"

    def handle_request(self, *args: Any, **kwargs: Any) -> None:
        self._record("handle_request", kwargs)


def _router(factory) -> Tuple[CommandRouter, _RecordingManager, str, str]:
    sink, client_id, expected = factory()
    manager = _RecordingManager()
    return (
        CommandRouter(manager, sink, daemon_plugins={}),
        manager, client_id, expected,
    )


# ------------------------------------------------- 1. the three consumers

@pytest.mark.parametrize("factory", _TRANSPORTS)
def test_a_new_session_is_attributed(factory) -> None:
    """``session.new`` — the path that already read the sink."""
    router, manager, client_id, expected = _router(factory)
    router.handle_request(
        client_id, "", CommandRequest(command="session.new", args=[]),
    )
    assert manager.kwargs_of("create_session")["created_by"] == expected


@pytest.mark.parametrize("factory", _TRANSPORTS)
def test_the_default_session_is_attributed(factory) -> None:
    """``session.default`` — the path that did not.

    ``IPCClient.get_default_session()`` is its only caller and it is a
    public SDK method, so this was reachable by every client of both
    transports rather than being an internal corner.
    """
    router, manager, client_id, expected = _router(factory)
    router.handle_request(
        client_id, "", CommandRequest(command="session.default", args=[]),
    )
    assert manager.kwargs_of("get_or_create_default")["created_by"] == expected


@pytest.mark.parametrize("factory", _TRANSPORTS)
def test_a_permission_answer_is_attributed(factory) -> None:
    """The other consumer: who ANSWERED, not who created (#859).

    Reaches ``PermissionResolvedEvent.user_id`` and the ledger's
    ``permission-check`` row, so it is a second spine off the same head
    and would not be covered by the two creation cases.
    """
    router, manager, client_id, expected = _router(factory)
    router.handle_request(
        client_id, "sess-1",
        PermissionResponseRequest(request_id="req-1", response="y"),
    )
    assert manager.kwargs_of("handle_request")["user_id"] == expected


# ------------------------------------------------------- 2. the controls

@pytest.mark.parametrize("factory", _TRANSPORTS)
def test_the_identity_is_the_transports_own(factory) -> None:
    """Non-vacuity: the sink really is deriving it.

    Every assertion above compares against a constant, and a constant
    matches an accident as readily as a mechanism.  This one states that
    the value came from the transport under test -- so a refactor that
    hardcoded ``"ana"`` somewhere in the router fails here even though the
    three cases above would still pass.
    """
    sink, client_id, expected = factory()
    assert sink.get_client_user(client_id) == expected
    assert sink.get_client_user("nobody-connected-under-this-id") is None


@pytest.mark.parametrize("factory", _TRANSPORTS)
def test_an_unauthenticated_client_is_attributed_to_nobody(factory) -> None:
    """The other direction, and the one that must not become an empty string.

    ``created_by=""`` is falsy, so every ``if user_id and ...`` ownership
    guard short-circuits exactly as it does for an unauthenticated client
    -- the fail-open ``test_identity_at_connect_1074`` refuses at the WS
    door.  Here it is the consumer's half: an unknown client must produce
    ``None``, which #859's writers OMIT rather than record.
    """
    router, manager, _client_id, _expected = _router(factory)
    router.handle_request(
        "a-client-this-transport-never-accepted", "",
        CommandRequest(command="session.default", args=[]),
    )
    assert manager.kwargs_of("get_or_create_default")["created_by"] is None


# --------------------------------------- 3. the other half of the joint

class _FakeManagerSelf:
    """Just enough ``SessionManager`` for ``get_or_create_default``.

    The method reaches five attributes and nothing else on the create
    branch.  Binding the REAL method onto this rather than driving a
    constructed ``SessionManager`` follows
    ``test_relative_paths_do_not_cross_the_daemon_boundary``: what is
    under test is one method's forwarding, and standing up a manager
    would make the test about session bootstrap instead.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.RLock()
        self._sessions: Dict[str, Any] = {}
        self.created_with: Dict[str, Any] = {}

    def _get_persisted_sessions(self, workspace_path=None):
        return []

    def _workspaces_match(self, a, b) -> bool:      # pragma: no cover
        return False

    def create_session(self, *args: Any, **kwargs: Any) -> str:
        self.created_with = kwargs
        return "sess-1"


def _get_or_create_default(manager: _FakeManagerSelf, **kwargs: Any) -> str:
    from server.session_manager import SessionManager

    bound = SessionManager.get_or_create_default.__get__(
        manager, SessionManager)
    return bound("client-1", **kwargs)


def test_the_manager_forwards_the_identity_it_was_given() -> None:
    """The router reads it; the manager has to carry it.

    Covered separately because the cases above fake ``SessionManager``
    entirely -- so they prove the value LEAVES the router and say nothing
    about whether it arrives at ``create_session``.  Both halves were
    broken before this change, and either alone is enough to lose the
    attribution.
    """
    manager = _FakeManagerSelf()
    _get_or_create_default(manager, workspace_path=None, created_by="ana")
    assert manager.created_with["created_by"] == "ana"


def test_attaching_does_not_restamp_someone_elses_session() -> None:
    """The asymmetry: ``created_by`` is who CREATED it.

    ``get_or_create_default`` attaches when a session for the workspace
    already exists, and that branch must not carry the identity through --
    re-stamping a session with whoever attached next would replace a true
    fact with a plausible one.  Asserted by the create branch never being
    reached.
    """
    manager = _FakeManagerSelf()
    manager._workspaces_match = lambda a, b: True       # type: ignore
    existing = type("S", (), {
        "session_id": "sess-existing",
        "workspace_path": "/w",
        "attached_clients": set(),
        "server": type("Srv", (), {
            "emit_current_state": lambda self, *a, **k: None,
        })(),
    })()
    manager._sessions = {"sess-existing": existing}
    manager._client_to_session: Dict[str, str] = {}
    manager._emit_to_client = lambda *a, **k: None       # type: ignore
    manager._build_session_info_event = lambda s: None   # type: ignore

    result = _get_or_create_default(
        manager, workspace_path="/w", created_by="someone-else")

    assert result == "sess-existing"
    assert manager.created_with == {}, (
        "the attach branch reached create_session, so this test is not "
        "asserting what it claims"
    )
