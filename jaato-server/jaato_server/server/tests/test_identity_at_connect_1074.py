"""Identity is established at CONNECT, and an app credential binds only (#1074).

A WS client used to be attributed to a user by a **message** — the
``auth.token`` frame a daemon extension validates against one
hard-configured OIDC realm.  Two consequences, and the second is the one
this module pins:

1. one daemon serves one realm, so a second application with its own
   userbase cannot share it; and
2. **identity is opt-in**, so declining to present one is the permissive
   path.  A client that completes the bearer handshake and never sends
   ``auth.token`` is attributed to nobody, and every ownership guard
   written ``if user_id and ...`` short-circuits for exactly that client.

The route added here mints a short-lived, single-use **ticket** bound to a
user the application has already authenticated in its own realm, and the
daemon resolves it during the Upgrade — so the identity exists before the
first frame and cannot be declined by omission.

What this module pins is the part a future change could silently undo:

A. **the ticket is spent at connect.**  A single-use ticket that is peeked
   at rather than consumed opens any number of connections, which is the
   difference between a connect credential and a bearer token;
B. **an application credential binds and nothing else.**  It authorises
   minting identities; letting it also drive a session would make it a
   super-user of every workspace on the daemon, attributed to no person;
C. **one application cannot reach another's users.**  ``app_id`` is taken
   from the credential rather than the request, is refused if it could make
   the qualified identity ambiguous, and scopes both revoke routes;
D. **a user the daemon would read as unauthenticated is refused at the
   door.**  ``created_by=""`` is falsy, so an empty ``user`` reproduces the
   fail-open this whole mechanism exists to close, arriving through the
   front door;
E. **the credentials file is refused when other principals can read it**,
   the rule ``--ws-token-file`` already applies; and
F. **with no application credentials configured, nothing changes.**  The
   connection-auth path is the single shared-digest comparison it has
   always been, and both verbs answer ``denied``.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import (
    PROTOCOL_VERSION,
    EventType,
    TicketBindRequest,
    TicketBindResultEvent,
    TicketRevokeRequest,
    TicketRevokeResultEvent,
    deserialize_event,
    serialize_event,
)
from jaato_server.server.websocket import (
    AUTH_KIND_APP,
    AUTH_KIND_OPEN,
    AUTH_KIND_SHARED,
    AUTH_KIND_TICKET,
    HAS_WEBSOCKETS,
    JaatoWSServer,
)
from jaato_server.server.ws_tickets import (
    AppCredentialStore,
    AppCredentialsError,
    BoundIdentity,
    MAX_TICKET_TTL_SECONDS,
    TicketCapacityError,
    TicketRegistry,
    credential_digest,
    load_app_credentials,
)
from jaato_server.shared.tests.reversion import Reversion

_WS = "jaato-server/jaato_server/server/websocket.py"
_TICKETS = "jaato-server/jaato_server/server/ws_tickets.py"

REVERSIONS = [
    Reversion(
        target=_WS,
        find="""        identity = self._ticket_registry.resolve_digest(digest, consume=consume)""",
        replace="""        identity = self._ticket_registry.resolve_digest(digest, consume=False)""",
        test="test_a_single_use_ticket_opens_exactly_one_connection",
        because=(
            "a captured ticket opens any number of connections, so it is a "
            "bearer token with a shorter life rather than a connect credential"
        ),
    ),
    Reversion(
        target=_WS,
        find="""        if self._client_auth_kind(client_id) == AUTH_KIND_APP:""",
        replace="""        if False and self._client_auth_kind(client_id) == AUTH_KIND_APP:""",
        test="test_an_app_credential_connection_cannot_drive_a_session",
        because=(
            "the long-lived credential that mints identities can also open "
            "sessions on every workspace, attributed to no person"
        ),
    ),
    Reversion(
        target=_TICKETS,
        find="""            if app_id is not None and entry.identity.app_id != app_id:""",
        replace="""            if False and entry.identity.app_id != app_id:""",
        test="test_one_application_cannot_revoke_anothers_ticket",
        because=(
            "one application revokes another's outstanding tickets, and the "
            "verb becomes an existence oracle across the app boundary"
        ),
    ),
    Reversion(
        target=_TICKETS,
        find="""                if entry.identity.app_id == app_id
                and entry.identity.user == user""",
        replace="""                if entry.identity.user == user""",
        test="test_one_application_cannot_log_out_anothers_user",
        because=(
            "an application logs out every other application's identically "
            "named user"
        ),
    ),
    Reversion(
        target=_WS,
        find="""        if client.auth_kind == AUTH_KIND_TICKET:""",
        replace="""        if False and client.auth_kind == AUTH_KIND_TICKET:""",
        test="test_a_connect_established_identity_is_not_overwritten_by_a_message",
        because=(
            "a later message re-attributes a connection whose identity was "
            "established at connect, so assertion-over-a-message wins after "
            "all -- as a replacement rather than as an omission"
        ),
    ),
    Reversion(
        target=_TICKETS,
        find="""_APP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")""",
        replace="""_APP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")""",
        test="test_an_app_id_that_would_make_the_qualified_identity_ambiguous_is_refused",
        because=(
            "two different (app, user) pairs qualify to one string, so the "
            "ownership guards compare equal across the boundary"
        ),
    ),
    Reversion(
        target=_TICKETS,
        find="""    if not isinstance(user, str) or not user:
        raise ValueError("user must be a non-empty string")""",
        replace="""    if not isinstance(user, str):
        raise ValueError("user must be a string")""",
        test="test_an_empty_user_is_refused_at_the_door",
        because=(
            "an empty created_by is falsy, so every `if user_id and ...` "
            "ownership guard short-circuits exactly as it does for an "
            "unauthenticated client -- the fail-open, through the front door"
        ),
    ),
    Reversion(
        target=_TICKETS,
        find="""    if sys.platform != "win32" and mode & (stat.S_IRWXG | stat.S_IRWXO):""",
        replace="""    if False and mode & (stat.S_IRWXG | stat.S_IRWXO):""",
        test="test_a_group_readable_credentials_file_is_refused",
        because=(
            "a credential for every identity an application can assert is "
            "read from a file anything on the host can read"
        ),
    ),
]


pytestmark = pytest.mark.skipif(
    not HAS_WEBSOCKETS, reason="websockets package not installed"
)


# --------------------------------------------------------------- fixtures

_APP_TOKEN = "app-credential-0123456789abcdef"
_OTHER_TOKEN = "other-credential-0123456789abcd"
_SHARED_TOKEN = "shared-bearer-token"


def _fake_ws(credential: Optional[str] = None) -> MagicMock:
    """The minimum surface the connect-time auth resolver reads.

    Mirrors ``server/test_websocket_auth.py::_fake_ws``: the real
    ``websockets.ServerConnection`` exposes ``request.headers`` and
    ``request.path``, and mocking it avoids binding a socket.
    """
    request = MagicMock()
    request.headers = (
        {"Authorization": f"Bearer {credential}"} if credential else {}
    )
    request.path = "/"
    ws = MagicMock()
    ws.request = request
    return ws


def _server(
    *,
    token: Optional[str] = _SHARED_TOKEN,
    apps: Optional[dict] = None,
) -> JaatoWSServer:
    """A server instance that never binds a port.

    ``apps`` defaults to the two-application store the cross-application
    cases need; pass ``{}`` for the unconfigured (pre-#1074) posture.
    """
    if apps is None:
        apps = {"acme": _APP_TOKEN, "other": _OTHER_TOKEN}
    return JaatoWSServer(
        host="127.0.0.1",
        port=0,
        required_token=token,
        app_credentials=AppCredentialStore(apps),
    )


def _attach(srv: JaatoWSServer, client_id: str, auth) -> None:
    """Register an already-authenticated client without a real socket.

    ``_handle_client`` is a coroutine that owns a live connection; every
    dispatch-level assertion here only needs the ``ClientConnection`` it
    would have built, stamped with the same verdict.
    """
    from jaato_server.server.websocket import ClientConnection

    srv._clients[client_id] = ClientConnection(
        websocket=MagicMock(),
        client_id=client_id,
        connected_at="2026-01-01T00:00:00+00:00",
        subscriptions=set(),
        user_id=auth.user,
        app_id=auth.app_id,
        auth_kind=auth.kind,
    )


def _sent(srv: JaatoWSServer, client_id: str):
    """Every event ``_send_to_client`` delivered to ``client_id``, decoded."""
    ws = srv._clients[client_id].websocket
    return [deserialize_event(call.args[0]) for call in ws.send.call_args_list]


async def _bind(srv: JaatoWSServer, client_id: str, **kwargs):
    """Drive one ``ticket.bind`` through the real dispatch path."""
    request = TicketBindRequest(**kwargs)
    await srv._dispatch_client_message(client_id, serialize_event(request))
    return _sent(srv, client_id)[-1]


# ------------------------------------------------------- A. spent at connect

@pytest.mark.asyncio
async def test_a_single_use_ticket_opens_exactly_one_connection():
    """The second presentation of a spent ticket is refused.

    This is the property that separates a *connect* credential from a
    bearer token: the ticket buys one attributed connection and is gone.
    The connection it opened is unaffected — identity was copied onto the
    ``ClientConnection`` at accept time and is never re-derived.
    """
    srv = _server()
    _attach(srv, "app", _auth_app(srv, "acme"))
    result = await _bind(srv, "app", request_id="r1", user="alice")
    assert result.status == "bound"

    first = srv._resolve_connection_auth(_fake_ws(result.ticket))
    assert first is not None
    assert first.kind == AUTH_KIND_TICKET
    assert first.user == "acme:alice"

    second = srv._resolve_connection_auth(_fake_ws(result.ticket))
    assert second is None, "a single-use ticket was accepted twice"


@pytest.mark.asyncio
async def test_the_predicate_does_not_spend_the_ticket_it_reports_on():
    """``_check_ws_token`` peeks; only the accept path consumes.

    A bool-returning helper that silently spent a credential would be a
    trap for every caller after the first.
    """
    srv = _server()
    _attach(srv, "app", _auth_app(srv, "acme"))
    result = await _bind(srv, "app", request_id="r1", user="alice")

    ws = _fake_ws(result.ticket)
    assert srv._check_ws_token(ws) is True
    assert srv._check_ws_token(ws) is True
    assert srv._resolve_connection_auth(ws) is not None


@pytest.mark.asyncio
async def test_a_non_single_use_ticket_opens_several():
    """``single_use=False`` is the weaker posture, and it is opt-in."""
    srv = _server()
    _attach(srv, "app", _auth_app(srv, "acme"))
    result = await _bind(
        srv, "app", request_id="r1", user="alice", single_use=False
    )
    for _ in range(3):
        auth = srv._resolve_connection_auth(_fake_ws(result.ticket))
        assert auth is not None and auth.user == "acme:alice"


# ------------------------------------------------------------ B. bind-only

def _auth_app(srv: JaatoWSServer, app_id: str):
    """Resolve the connection auth an app credential would be accepted with."""
    token = _APP_TOKEN if app_id == "acme" else _OTHER_TOKEN
    auth = srv._resolve_connection_auth(_fake_ws(token))
    assert auth is not None and auth.kind == AUTH_KIND_APP
    assert auth.app_id == app_id
    return auth


@pytest.mark.asyncio
async def test_an_app_credential_connection_cannot_drive_a_session():
    """Every non-ticket verb from an app connection is refused.

    ``send_message`` stands in for the whole surface: the gate is on the
    connection kind rather than on a list of verbs, so a verb added later
    is covered whether or not its author remembers this rule.
    """
    from jaato_sdk.events import SendMessageRequest

    srv = _server()
    _attach(srv, "app", _auth_app(srv, "acme"))
    await srv._dispatch_client_message(
        "app", serialize_event(SendMessageRequest(text="run something"))
    )
    answers = _sent(srv, "app")
    assert len(answers) == 1
    assert answers[0].type == EventType.ERROR
    assert "application-credential connection" in answers[0].error


@pytest.mark.asyncio
async def test_a_ticket_connection_may_drive_a_session_and_not_bind():
    """The privilege is not transitive.

    Holding a ticket is being a user; it must not let that user mint more
    identities.  So the ticket connection reaches the ordinary dispatcher
    (the error below is the session-level one, not the bind-only refusal)
    and its ``ticket.bind`` is denied.
    """
    srv = _server()
    _attach(srv, "app", _auth_app(srv, "acme"))
    minted = await _bind(srv, "app", request_id="r1", user="alice")
    user_auth = srv._resolve_connection_auth(_fake_ws(minted.ticket))
    _attach(srv, "user", user_auth)

    denied = await _bind(srv, "user", request_id="r2", user="mallory")
    assert denied.status == "denied"
    assert denied.ticket == ""

    from jaato_sdk.events import StopRequest

    await srv._dispatch_client_message(
        "user", serialize_event(StopRequest())
    )
    last = _sent(srv, "user")[-1]
    assert "application-credential connection" not in getattr(last, "error", "")


@pytest.mark.asyncio
async def test_a_connect_established_identity_is_not_overwritten_by_a_message():
    """``set_client_user`` does not re-attribute a ticket connection.

    The two identity routes are alternatives, not layers.  Letting the
    message win would reintroduce assertion-over-a-message as a
    *replacement* — the same shape the ticket route removes, arriving from
    the other side.  Every other connection kind is unchanged, which the
    second half of this test is the control for.
    """
    srv = _server()
    _attach(srv, "app", _auth_app(srv, "acme"))
    minted = await _bind(srv, "app", request_id="r1", user="alice")
    _attach(srv, "user", srv._resolve_connection_auth(_fake_ws(minted.ticket)))

    srv.set_client_user("user", "somebody-else")
    assert srv.get_client_user("user") == "acme:alice"

    # Control: the premium SSO route on a shared-token connection is
    # unchanged, so the refusal above is about the ticket, not about the
    # hook having stopped working.
    _attach(srv, "shared", srv._resolve_connection_auth(_fake_ws(_SHARED_TOKEN)))
    srv.set_client_user("shared", "sso-user")
    assert srv.get_client_user("shared") == "sso-user"


@pytest.mark.asyncio
async def test_a_shared_token_connection_may_not_bind():
    """The daemon-wide token says "may drive this daemon", not who.

    It cannot mint identities, because it names no application to qualify
    them with.
    """
    srv = _server()
    _attach(srv, "shared", srv._resolve_connection_auth(_fake_ws(_SHARED_TOKEN)))
    result = await _bind(srv, "shared", request_id="r1", user="alice")
    assert result.status == "denied"
    assert result.request_id == "r1", "a refusal must stay correlatable"


# ------------------------------------------------ C. one application's users

def test_the_daemon_qualifies_the_identity_itself():
    """Two applications each holding an ``alice`` do not compare equal."""
    assert BoundIdentity("acme", "alice").qualified == "acme:alice"
    assert (
        BoundIdentity("acme", "alice").qualified
        != BoundIdentity("other", "alice").qualified
    )


def test_bind_takes_app_id_from_the_credential_not_the_request():
    """``TicketBindRequest`` has no ``app_id`` field at all.

    A field would be a place for a caller to assert which application it
    is, which is exactly what the credential already established.
    """
    assert "app_id" not in TicketBindRequest.model_fields


@pytest.mark.asyncio
async def test_one_application_cannot_revoke_anothers_ticket():
    """A ticket bound by another application answers ``not_found``.

    The same answer an unknown ticket gives, so the verb is not an
    existence oracle across the application boundary.
    """
    srv = _server()
    _attach(srv, "acme", _auth_app(srv, "acme"))
    _attach(srv, "other", _auth_app(srv, "other"))
    minted = await _bind(srv, "acme", request_id="r1", user="alice")

    await srv._dispatch_client_message(
        "other",
        serialize_event(
            TicketRevokeRequest(request_id="r2", ticket=minted.ticket)
        ),
    )
    answer = _sent(srv, "other")[-1]
    assert answer.status == "not_found"
    assert answer.revoked == 0

    # And the ticket still works for the application that minted it.
    assert srv._resolve_connection_auth(_fake_ws(minted.ticket)) is not None


@pytest.mark.asyncio
async def test_one_application_cannot_log_out_anothers_user():
    """``revoke_user`` is scoped too — logout is per application."""
    srv = _server()
    _attach(srv, "acme", _auth_app(srv, "acme"))
    _attach(srv, "other", _auth_app(srv, "other"))
    minted = await _bind(srv, "acme", request_id="r1", user="alice")

    await srv._dispatch_client_message(
        "other",
        serialize_event(TicketRevokeRequest(request_id="r2", user="alice")),
    )
    assert _sent(srv, "other")[-1].revoked == 0
    assert srv._resolve_connection_auth(_fake_ws(minted.ticket)) is not None


@pytest.mark.asyncio
async def test_an_application_revokes_its_own():
    """The control run: without it, every case above could pass vacuously."""
    srv = _server()
    _attach(srv, "acme", _auth_app(srv, "acme"))
    minted = await _bind(srv, "acme", request_id="r1", user="alice")

    await srv._dispatch_client_message(
        "acme",
        serialize_event(
            TicketRevokeRequest(request_id="r2", ticket=minted.ticket)
        ),
    )
    answer = _sent(srv, "acme")[-1]
    assert answer.status == "revoked"
    assert answer.revoked == 1
    assert srv._resolve_connection_auth(_fake_ws(minted.ticket)) is None


@pytest.mark.asyncio
async def test_revoke_requires_exactly_one_selector():
    """Both, or neither, has two readings — and guessing revokes the wrong
    thing."""
    srv = _server()
    _attach(srv, "acme", _auth_app(srv, "acme"))
    for kwargs in ({}, {"ticket": "t", "user": "alice"}):
        await srv._dispatch_client_message(
            "acme",
            serialize_event(TicketRevokeRequest(request_id="r", **kwargs)),
        )
        assert _sent(srv, "acme")[-1].status == "invalid"


def test_an_app_id_that_would_make_the_qualified_identity_ambiguous_is_refused(
    tmp_path,
):
    """``:`` in an app id makes two (app, user) pairs qualify to one string.

    ``a:b`` + ``c`` and ``a`` + ``b:c`` both render ``a:b:c``, which is the
    value the ownership guards compare.
    """
    path = _creds_file(tmp_path, {"acme:prod": _APP_TOKEN})
    with pytest.raises(AppCredentialsError) as exc:
        load_app_credentials(path)
    assert "app id" in str(exc.value)


def test_two_applications_sharing_one_credential_is_refused():
    """Otherwise the resolved ``app_id`` depends on iteration order."""
    with pytest.raises(ValueError):
        AppCredentialStore({"acme": _APP_TOKEN, "other": _APP_TOKEN})


# ----------------------------------------------------- D. an unusable user

def test_an_empty_user_is_refused_at_the_door():
    """``created_by=""`` is falsy and reads as unauthenticated everywhere."""
    registry = TicketRegistry()
    with pytest.raises(ValueError):
        registry.bind(app_id="acme", user="")


@pytest.mark.parametrize("user", ["ali\nce", "ali\x00ce", "x" * 300])
def test_a_user_the_daemon_would_have_to_log_or_store_badly_is_refused(user):
    """A newline forges a log line; the value is also persisted."""
    registry = TicketRegistry()
    with pytest.raises(ValueError):
        registry.bind(app_id="acme", user=user)


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, -1, MAX_TICKET_TTL_SECONDS + 1])
async def test_a_ttl_outside_the_range_is_refused_not_clamped(ttl):
    """Silently issuing something other than what was asked for is how an
    integrator comes to believe a ticket lasts a day."""
    srv = _server()
    _attach(srv, "acme", _auth_app(srv, "acme"))
    result = await _bind(
        srv, "acme", request_id="r1", user="alice", ttl_seconds=ttl
    )
    assert result.status == "invalid"
    assert result.ticket == ""


def test_an_expired_ticket_resolves_to_nothing():
    """Enforcement is on a monotonic clock, so a test states its instant."""
    now = [1000.0]
    registry = TicketRegistry(clock=lambda: now[0])
    ticket, _ = registry.bind(app_id="acme", user="alice", ttl_seconds=60)
    now[0] = 1059.0
    assert registry.resolve(ticket, consume=False) is not None
    now[0] = 1060.0
    assert registry.resolve(ticket) is None


def test_the_registry_refuses_rather_than_evicting_at_its_ceiling():
    """Evicting would turn one misbehaving application into failed logins
    for another."""
    registry = TicketRegistry(max_outstanding=2)
    first, _ = registry.bind(app_id="acme", user="alice")
    registry.bind(app_id="acme", user="bob")
    with pytest.raises(TicketCapacityError):
        registry.bind(app_id="acme", user="carol")
    assert registry.resolve(first, consume=False) is not None


def test_no_plaintext_credential_is_retained():
    """Both stores are keyed by digest; ``bind`` hands the plaintext out and
    keeps none of it."""
    registry = TicketRegistry()
    ticket, _ = registry.bind(app_id="acme", user="alice")
    assert ticket not in repr(registry)
    assert all(
        ticket not in repr(v) for v in vars(registry).values()
    )
    store = AppCredentialStore({"acme": _APP_TOKEN})
    assert _APP_TOKEN not in repr(store)
    assert store.lookup(credential_digest(_APP_TOKEN)) == "acme"


# ------------------------------------------------------- E. the file's mode

def _creds_file(tmp_path: Path, mapping: dict, mode: int = 0o600) -> Path:
    path = tmp_path / "ws-apps.json"
    path.write_text(json.dumps(mapping))
    os.chmod(path, mode)
    return path


def test_a_well_formed_credentials_file_loads(tmp_path):
    """The control run for every refusal below."""
    store = load_app_credentials(_creds_file(tmp_path, {"acme": _APP_TOKEN}))
    assert store.app_ids() == ("acme",)
    assert store.lookup(credential_digest(_APP_TOKEN)) == "acme"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_a_group_readable_credentials_file_is_refused(tmp_path):
    """The rule ``--ws-token-file`` already applies, for a credential that
    stands for every identity its application can assert."""
    path = _creds_file(tmp_path, {"acme": _APP_TOKEN}, mode=0o640)
    assert Path(path).stat().st_mode & stat.S_IRWXG
    with pytest.raises(AppCredentialsError) as exc:
        load_app_credentials(path)
    assert "0600" in str(exc.value)


def test_an_empty_credentials_file_is_refused(tmp_path):
    """A configured flag that authorises nobody is a mistake, not a
    posture."""
    with pytest.raises(AppCredentialsError):
        load_app_credentials(_creds_file(tmp_path, {}))


def test_a_short_credential_is_refused(tmp_path):
    """It is long-lived and sits in a file — the one credential here a
    guessing attacker has time to work on."""
    with pytest.raises(AppCredentialsError):
        load_app_credentials(_creds_file(tmp_path, {"acme": "short"}))


def test_a_malformed_credentials_file_is_refused(tmp_path):
    path = tmp_path / "ws-apps.json"
    path.write_text("not json at all")
    os.chmod(path, 0o600)
    with pytest.raises(AppCredentialsError):
        load_app_credentials(path)


# --------------------------------------------- F. the unconfigured posture

def test_with_no_app_credentials_auth_is_the_shared_digest_alone():
    """The hard requirement: opt-in, so no existing deployment changes."""
    srv = _server(apps={})
    assert srv._resolve_connection_auth(_fake_ws(_SHARED_TOKEN)).kind == (
        AUTH_KIND_SHARED
    )
    assert srv._resolve_connection_auth(_fake_ws("wrong")) is None
    assert srv._resolve_connection_auth(_fake_ws(None)) is None
    assert srv._resolve_connection_auth(_fake_ws(_APP_TOKEN)) is None


def test_with_neither_token_nor_app_credentials_every_connection_is_accepted():
    """``--ws-unsafe-no-auth``, unchanged."""
    srv = _server(token=None, apps={})
    for credential in (None, "anything"):
        auth = srv._resolve_connection_auth(_fake_ws(credential))
        assert auth is not None and auth.kind == AUTH_KIND_OPEN
        assert auth.user is None and auth.app_id is None


def test_app_credentials_alone_turn_auth_on():
    """Configuring applications is the fail-CLOSED direction: it must not
    leave the daemon accepting unauthenticated connections."""
    srv = _server(token=None)
    assert srv._resolve_connection_auth(_fake_ws(None)) is None
    assert srv._resolve_connection_auth(_fake_ws(_APP_TOKEN)).kind == (
        AUTH_KIND_APP
    )


@pytest.mark.asyncio
async def test_the_bind_verb_is_denied_where_the_feature_is_off():
    """And the refusal says the feature is off, which is a different next
    step from "this connection is the wrong kind"."""
    srv = _server(apps={})
    _attach(srv, "shared", srv._resolve_connection_auth(_fake_ws(_SHARED_TOKEN)))
    result = await _bind(srv, "shared", request_id="r1", user="alice")
    assert result.status == "denied"
    assert "--ws-app-credentials" in (result.detail or "")


# ------------------------------------------------------------- the protocol

def test_the_protocol_minor_is_bumped_for_the_new_verbs():
    """Four new events, in both directions — a MINOR under the compat rule."""
    major, minor = PROTOCOL_VERSION.split(".")[:2]
    assert major == "1"
    assert int(minor) >= 10


@pytest.mark.parametrize(
    "cls",
    [
        TicketBindRequest,
        TicketBindResultEvent,
        TicketRevokeRequest,
        TicketRevokeResultEvent,
    ],
)
def test_each_ticket_event_round_trips(cls):
    """Registered in ``_EVENT_CLASSES``, which is also what the TypeScript
    codegen walks — a class missing from it is invisible to the TS SDK."""
    event = cls()
    assert isinstance(deserialize_event(serialize_event(event)), cls)
