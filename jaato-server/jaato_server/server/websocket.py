"""WebSocket Server for Jaato.

This module provides a WebSocket server that wraps JaatoServer,
enabling real-time bidirectional communication with multiple clients.

Usage:
    from jaato_server.server.websocket import JaatoWSServer

    server = JaatoWSServer(host="localhost", port=8080)
    await server.start()  # Blocks until shutdown
"""

import asyncio
import concurrent.futures
import errno
import hashlib
import hmac
import json
import logging
import os
import re
import ssl
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlsplit
import threading

try:
    import websockets
    from websockets import ServerConnection
    from websockets.exceptions import ConnectionClosed
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    ServerConnection = Any

from jaato_server.shared.apparmor_label import SANDBOX_MODE_SOFT, sandbox_mode_for_profile
from .core import JaatoServer
from .ws_tickets import (
    AppCredentialStore,
    BoundIdentity,
    TicketCapacityError,
    TicketRegistry,
    credential_digest,
)
from .workspace_provisioner import WorkspaceProvisioner, ProvisionedWorkspace
from .apparmor import AppArmorManager
from .cgroups import CgroupsManager
from .session_logging import set_logging_context, clear_logging_context
from .transfer_limits import STAGE_PER_FILE_LIMIT, STAGE_TOTAL_LIMIT
from jaato_sdk.events import (
    Event,
    EventType,
    PROTOCOL_VERSION,
    ConnectedEvent,
    ErrorEvent,
    SystemMessageEvent,
    serialize_event,
    deserialize_event,
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    ClarificationBatchResponseEvent,
    ReferenceSelectionResponseRequest,
    StopRequest,
    CommandRequest,
    # Workspace management events
    WorkspaceListRequest,
    WorkspaceListEvent,
    WorkspaceCreateRequest,
    WorkspaceCreatedEvent,
    WorkspaceSelectRequest,
    WorkspaceDeleteRequest,
    WorkspaceDeletedEvent,
    ConfigStatusEvent,
    ConfigUpdateRequest,
    ConfigUpdatedEvent,
    # Identity at connect — the ticket bind channel (#1074)
    TicketBindRequest,
    TicketBindResultEvent,
    TicketRevokeRequest,
    TicketRevokeResultEvent,
    # app:// secret resolution (#1226) — the bind channel's daemon->app verb
    # and its application->daemon revocation counterpart.
    SecretResolveRequest,
    SecretResolveResultEvent,
    SecretReloadRequest,
    SecretReloadResultEvent,
)
from .app_secret import AppSecretAnswer, AppSecretResolver
from .workspace_manager import WorkspaceManager
from .event_sink import EventSink


logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    """Return True when env var ``name`` is set to a truthy value.

    Matches the framework's standard truthy spelling (``1``/``true``/``yes``,
    case-insensitive). Absent or any other value reads as False.
    """
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _record_bootstrap_refusal(server: Any, reason: str) -> None:
    """#1253: record a bootstrap-outcome refusal on *server*.

    The WS pre-init hook fails a confinement-required session CLOSED by
    returning without spawning; ``session_manager.initialize_or_refuse``
    reads ``server.runner_bootstrap_error`` afterwards to refuse the session
    by name (``RunnerBootstrapFailed``) rather than letting it come up and
    run model-driven work with no kernel boundary.  Best-effort — a server
    object that predates ``note_runner_bootstrap_outcome`` (a test double)
    simply records nothing, which reads downstream as "nothing known to be
    wrong", the same answer a session with no runner gives.
    """
    note = getattr(server, "note_runner_bootstrap_outcome", None)
    if callable(note):
        note(reason)


def _report_confined_spawn_failure(
    server: Any,
    session_id: str,
    exc: BaseException,
    confinement_required: bool,
) -> None:
    """#1253: handle a WS runner-spawn failure per the confinement invariant.

    A confinement-required session's spawn failure is REFUSED — an in-process
    fallback would run the model's tools in the daemon process with no
    AppArmor boundary, the same silent bypass the provisioning-failure path
    refuses.  A genuinely-unconfined session keeps the in-process fallback it
    always had.  Extracted from ``_apparmor_pre_init_hook`` so the hook does
    not grow past its complexity baseline (the #812 / #1167 / #1179 move).
    """
    if confinement_required:
        logger.warning(
            "AppArmor pre-init: runner spawn failed for session %s "
            "(%s: %s) but confinement is required — REFUSING the session "
            "rather than falling back to unconfined in-process tool "
            "execution (#1253)",
            session_id, type(exc).__name__, exc, exc_info=True,
        )
        _record_bootstrap_refusal(
            server,
            "AppArmor confinement required but runner spawn failed "
            f"({type(exc).__name__}: {exc}); refusing to fall back to "
            "unconfined in-process execution (#1253)",
        )
        return
    logger.warning(
        "AppArmor pre-init: runner spawn failed for session %s "
        "(%s: %s) — falling back to in-process tool execution; "
        "post-init hook will downgrade to soft mode",
        session_id, type(exc).__name__, exc, exc_info=True,
    )


def _hand_managed_root_to(session_manager: Any, workspace_root: Optional[str]) -> None:
    """#1280: tell *session_manager* which workspaces this WS server manages.

    The WS pre-init hook threads ``managed_workspace_root`` into its own
    spawn, but a premium WUI ``session.new`` is spawned by
    ``SessionManager._spawn_session_runner_unconditional``, which had no way
    to learn the root: the daemon wires ``set_apparmor_dependencies`` before
    it constructs this server.  Called from :meth:`JaatoWSServer.set_command_router`,
    the seam that registers the pre-init hook, so both spawn paths see the
    same root.

    A session manager without ``set_managed_workspace_root`` (a test double,
    an out-of-tree manager) is left alone.  Extracted so
    ``set_command_router`` gains no branch.
    """
    setter = getattr(session_manager, "set_managed_workspace_root", None)
    if callable(setter):
        setter(workspace_root)


# Default path for servers.json (contains TLS config)
_SERVERS_JSON = Path.home() / ".jaato" / "servers.json"


# ── File-staging caps (defaults; both apply to every staging call) ──
# Per-file: protects against a single huge upload tying up the daemon.
# Total:    protects against many smaller uploads totalling something huge.
# Both are advisory defaults — when we add per-deployment config, these
# become the fallback when the operator hasn't set values themselves.
# ONE definition, in ``transfer_limits``: the cross-workspace copy a group
# message makes into its target's inbox is bounded by the same two numbers
# (design §4.5), so they must not be able to drift apart.
DEFAULT_STAGE_PER_FILE_LIMIT = STAGE_PER_FILE_LIMIT
DEFAULT_STAGE_TOTAL_LIMIT    = STAGE_TOTAL_LIMIT

#: The largest single WebSocket message the daemon accepts, in bytes.
#:
#: ``websockets`` refuses any message over its ``max_size`` by closing the
#: connection with code 1009, and its default is 1 MiB.  The daemon never
#: set one, so every staged file over 1 MiB (one BINARY frame per file)
#: closed the connection mid-upload while the staging cap above allows
#: 10 MB: the client reconnected, no ``StageFilesEvent`` ever arrived, and
#: the upload read as a two-minute hang.
#:
#: The default covers the two largest payloads the protocol carries in
#: one message: a staged file at the per-file cap (a raw binary frame),
#: and the same file sent inline as base64 (``staged_files`` on
#: ``session.new``, ``attachments`` on ``send_message``), which is 4/3 of
#: it -- plus :data:`WS_MESSAGE_ENVELOPE_HEADROOM` for the JSON around it.
#: ``--ws-max-message-size`` overrides it; the effective value is
#: advertised to clients in ``ConnectedEvent.server_info`` as
#: ``max_message_size`` so a client can refuse a file BEFORE sending it.
WS_MESSAGE_ENVELOPE_HEADROOM = 1024 * 1024
DEFAULT_WS_MAX_MESSAGE_SIZE = 16 * 1024 * 1024

#: The value ``websockets`` applies when none is passed, and so the limit
#: every daemon before ``max_message_size`` was advertised enforced.  A
#: client that finds no ``max_message_size`` in ``server_info`` is talking
#: to such a daemon and should assume this.  Also the floor the
#: ``--ws-max-message-size`` flag accepts: going below it would refuse
#: messages every existing client already sends.
LEGACY_WS_MAX_MESSAGE_SIZE = 1024 * 1024

#: WebSocket close codes that are an ordinary end of a connection: a
#: normal close (1000), a peer going away (1001, a browser tab closing),
#: no status (1005) and an abnormal drop with no close frame (1006, a
#: network blip or a killed process).  Anything else -- 1009 "message too
#: big" above all -- means one side REFUSED something, and is logged at
#: WARNING rather than silently discarded.
_ORDINARY_CLOSE_CODES = frozenset({1000, 1001, 1005, 1006})


def describe_connection_close(exc: BaseException) -> "tuple[Optional[int], str]":
    """Return ``(code, text)`` for a ``ConnectionClosed``.

    ``code`` is the close code this server SENT when it initiated the
    close (the case that matters: a 1009 is the daemon refusing a
    message), else the one it received, else ``None`` when neither side
    sent a close frame.  ``text`` is a one-line rendering for a log line.
    Tolerates objects without ``sent``/``rcvd`` so a future
    ``websockets`` that reshapes the exception degrades to "unknown"
    rather than raising inside the connection handler's ``except``.
    """
    sent = getattr(exc, "sent", None)
    rcvd = getattr(exc, "rcvd", None)
    frame = sent if sent is not None else rcvd
    if frame is None:
        return None, "no close frame (connection dropped)"
    code = getattr(frame, "code", None)
    reason = getattr(frame, "reason", "") or ""
    side = "sent" if frame is sent else "received"
    return code, f"{side} close {code}" + (f" ({reason})" if reason else "")


def parse_ws_max_message_size(value: str) -> int:
    """Parse ``--ws-max-message-size``: bytes, or a ``K``/``M``/``G`` suffix.

    Suffixes are binary (``16M`` is 16 MiB).  Raises ``ValueError`` naming
    the problem for a malformed value or one below
    :data:`LEGACY_WS_MAX_MESSAGE_SIZE`, so the daemon refuses to start
    rather than silently applying a limit nobody asked for.
    """
    text = (value or "").strip()
    m = re.fullmatch(r"(?i)(\d+)\s*([kmg]?)(?:i?b)?", text)
    if not m:
        raise ValueError(
            f"--ws-max-message-size {value!r}: expected a byte count, "
            f"optionally with a K/M/G suffix (e.g. 16M)"
        )
    scale = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[m.group(2).lower()]
    size = int(m.group(1)) * scale
    if size < LEGACY_WS_MAX_MESSAGE_SIZE:
        raise ValueError(
            f"--ws-max-message-size {value!r} is {size} bytes, below the "
            f"{LEGACY_WS_MAX_MESSAGE_SIZE}-byte floor every client already "
            f"relies on"
        )
    return size


def _safe_staged_filename(name: str) -> Optional[Path]:
    """Validate a staged-file name and return the workspace-relative ``Path``.

    Returns ``None`` when the name is unsafe (absolute, contains ``..``
    components, or empty).  The caller logs a warning and skips the
    entry — never raises.
    """
    if not name:
        return None
    clean = Path(name)
    if clean.is_absolute() or ".." in clean.parts:
        return None
    return clean


def _write_staged_payload(ws_path: Path, name: Path, payload: bytes) -> None:
    """Atomically write ``payload`` into ``ws_path / name``.

    Caller is responsible for path-safety validation (use
    :func:`_safe_staged_filename`) and size enforcement.  Creates parent
    directories as needed.
    """
    dest = ws_path / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)


def _materialize_staged_files(
    ws_path: Path,
    staged_files: list,
) -> int:
    """Write inline base64 ``staged_files`` entries into a workspace.

    Legacy entry point used by the premium ``session.new`` envelope
    (``[{"name": <relpath>, "data": <base64>}, ...]``) and by the
    HTTP ``/api/task/artifacts`` upload route.  Kept for back-compat
    with those clients; new code should prefer the binary-framed
    :class:`~jaato_sdk.events.StageFilesRequest` flow handled by
    :meth:`JaatoWSServer._handle_stage_files_request`.

    Unsafe entries (absolute paths, ``..`` components, invalid base64,
    non-dict entries, missing ``name``) are logged and skipped; the
    caller never sees an exception.

    Args:
        ws_path: Root of the workspace.  All writes are rooted here;
            the protections ensure they stay here.
        staged_files: Iterable of ``{"name": str, "data": base64_str}``
            dicts.  Non-dict entries are skipped.

    Returns:
        The number of files successfully written.
    """
    import base64

    written = 0
    for entry in staged_files or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        data = entry.get("data", "")
        clean = _safe_staged_filename(name)
        if clean is None:
            logger.warning("Rejecting staged_file with unsafe path: %s", name)
            continue
        try:
            # ``validate=True`` — reject non-alphabet characters rather than
            # silently ignoring them (default).  Without this, malformed
            # input still produces "valid" output bytes, which is a
            # security smell even if the name is workspace-bounded.
            payload = base64.b64decode(data, validate=True)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Rejecting staged_file %s: invalid base64 (%s)", name, exc,
            )
            continue
        _write_staged_payload(ws_path, clean, payload)
        written += 1
    return written


def load_tls_context(servers_json: Optional[Path] = None) -> Optional[ssl.SSLContext]:
    """Build an ``ssl.SSLContext`` from the ``tls`` section in servers.json.

    Args:
        servers_json: Path to servers.json. Defaults to ``~/.jaato/servers.json``.

    Returns:
        An ``ssl.SSLContext`` configured for TLS server mode, or ``None``
        if the file is missing, has no ``tls`` section, or cert files don't exist.
    """
    path = servers_json or _SERVERS_JSON
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    tls = data.get("tls")
    if not tls:
        return None

    cert = Path(tls.get("cert", "")).expanduser()
    key = Path(tls.get("key", "")).expanduser()
    ca_cert = Path(tls.get("ca_cert", "")).expanduser()

    if not cert.exists() or not key.exists():
        logger.warning(
            "TLS cert (%s) or key (%s) not found — falling back to plain ws://",
            cert, key,
        )
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert), str(key))
    if ca_cert.exists():
        ctx.load_verify_locations(str(ca_cert))
    # Do not require client certificates — browser dashboard uses SSO,
    # not mTLS.  Peer gossip connections handle mTLS separately.
    ctx.verify_mode = ssl.CERT_NONE

    logger.info("TLS context loaded: cert=%s, ca=%s", cert, ca_cert)
    return ctx


class WSEventSinkAdapter:
    """Adapts ``JaatoWSServer`` to the ``EventSink`` protocol.

    ``CommandRouter`` and ``SessionManager`` call ``send_event()`` from
    synchronous model/session threads.  This adapter bridges that into
    the async WebSocket world by scheduling coroutines on the WS
    server's event loop via ``asyncio.run_coroutine_threadsafe()``.

    ``client_id`` values that don't belong to any connected WebSocket
    client are silently ignored, which is the contract of ``EventSink``
    and allows ``CompositeEventSink`` to fan-out safely.

    Per-client session/workspace state is tracked locally so the
    ``CommandRouter`` can query workspace paths for WS clients.
    """

    def __init__(self, ws_server: "JaatoWSServer") -> None:
        self._ws = ws_server
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        # Per-client tracking (mirrors IPC server's client fields)
        self._client_sessions: Dict[str, str] = {}        # client_id -> session_id
        self._client_workspaces: Dict[str, Optional[str]] = {}  # client_id -> workspace

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the event loop for thread-safe scheduling.

        Must be called from the async context (e.g., inside ``start()``).
        """
        self._event_loop = loop

    def send_event(self, client_id: str, event) -> None:
        """Send an event to a WebSocket client (thread-safe).

        Silently ignores unknown ``client_id`` values.
        """
        if client_id not in self._ws._clients:
            return
        if not self._event_loop:
            return

        async def _send():
            await self._ws._send_to_client(client_id, event)

        asyncio.run_coroutine_threadsafe(_send(), self._event_loop)

    def broadcast_event(self, event) -> None:
        """Send an event to every connected WebSocket client (thread-safe).

        Used for daemon-wide events (HandoffGate transitions) that
        don't belong to a session.  Snapshots the client id list before
        scheduling the per-client sends so concurrent connect/disconnect
        doesn't race with the iteration.
        """
        if not self._event_loop:
            return
        # Snapshot the keys before scheduling — _ws._clients can mutate
        # while the coroutine is queued.
        client_ids = list(self._ws._clients.keys())
        if not client_ids:
            return

        async def _send_all():
            for cid in client_ids:
                if cid in self._ws._clients:
                    await self._ws._send_to_client(cid, event)

        asyncio.run_coroutine_threadsafe(_send_all(), self._event_loop)

    def set_client_session(self, client_id: str, session_id: str) -> None:
        """Associate a client with a session."""
        self._client_sessions[client_id] = session_id

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        """The workspace path for a client.

        What ``workspace.select`` bridged into this adapter, else the
        selection the WS server's :class:`WorkspaceManager` holds for the
        same client.  Two stores carried one fact -- the bridge wrote the
        first at select time and nothing else did -- so a client whose
        selection reached the manager by any other route (a session created
        for it, a selection restored on reconnect) declared no workspace
        here, and ``workspace.ignore`` refused a caller whose header named
        one.  Reading the manager makes the selection the source of truth
        and the local map a cache of it.
        """
        declared = self._client_workspaces.get(client_id)
        if declared:
            return declared
        manager = getattr(self._ws, "_workspace_manager", None)
        if manager is None:
            return None
        try:
            selected = manager.get_selected_workspace(client_id=client_id)
        except Exception:  # noqa: BLE001 -- a lookup must never fail a verb
            return None
        return selected.path if selected and selected.path else None

    def set_client_workspace(self, client_id: str, workspace_path: str) -> None:
        """Associate a workspace path with a client."""
        self._client_workspaces[client_id] = workspace_path

    def clear_client_workspace(self, client_id: str) -> None:
        """Forget a client's workspace path (its workspace was deleted)."""
        self._client_workspaces.pop(client_id, None)

    def visible_workspace_paths(self, client_id: str) -> Optional[List[str]]:
        """The workspace paths this client's user may see (protocol 1.13).

        ``None`` -- no scoping -- when the server runs no workspace manager
        or the connection carries no identity; otherwise the paths of
        ``WorkspaceManager.list_workspaces(for_user=...)``, the same rule the
        workspace verbs apply, so ``session.list`` cannot show a session in a
        workspace ``workspace.select`` would refuse.
        """
        return self._ws.visible_workspace_paths(client_id)

    def get_client_user(self, client_id: str) -> Optional[str]:
        """Get the authenticated user for a WS client."""
        return self._ws.get_client_user(client_id)

    def set_client_user(self, client_id: str, user_id: str) -> None:
        """Associate a user identity with a WS client."""
        self._ws.set_client_user(client_id, user_id)

    def get_client_peer(self, client_id: str) -> None:
        """No peer credential on a WebSocket — always ``None``.

        A WS client may be on another machine, so there is no local OS
        account for the kernel to vouch for.  Identity on this transport is
        the bearer token, a bound user ticket (#1074), or whatever an auth
        middleware attaches through :meth:`set_client_user` -- none of which
        is a peer CREDENTIAL.  A ticket says which person an application
        vouched for; it does not name an OS account, and every session runs
        as the daemon's uid either way.  So the client-path entitlement
        guards stay inert here and this transport's own auth is what
        applies.
        """
        return None

    def remove_client(self, client_id: str) -> None:
        """Clean up tracking state when a client disconnects."""
        self._client_sessions.pop(client_id, None)
        self._client_workspaces.pop(client_id, None)


def _get_server_version() -> str:
    """Read the jaato-server package version from installed metadata."""
    from importlib.metadata import version as pkg_version
    return pkg_version("jaato-server")


def _peek_message_type(message: str) -> str:
    """Read the ``type`` field of a raw client frame without committing to it.

    Cheaper than a decision it cannot make: :meth:`~JaatoWSServer._dispatch_client_message`
    has to know whether a frame is a ``ticket.*`` verb before it knows
    whether the connection may send anything else, and the typed
    deserialisation that follows belongs to whichever branch wins.

    Returns ``""`` for anything unparseable, so the frame falls through to
    the ordinary dispatcher and produces the ``Invalid JSON`` error it always
    did — the peek never becomes a second place that reports malformed input.
    """
    try:
        raw = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    value = raw.get("type", "")
    return value if isinstance(value, str) else ""


def _peek_request_id(message: str) -> str:
    """Read ``request_id`` off a frame that may not survive typed parsing.

    A refusal must still be correlatable: one bind channel serves many
    concurrent logins, so a ``denied`` result with no ``request_id`` cannot
    be attributed to the login it refused. Returns ``""`` when the frame does
    not carry one, which is the same thing a caller that sent none gets.
    """
    try:
        raw = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    value = raw.get("request_id", "")
    return value if isinstance(value, str) else ""


def _answer_from_result(result: "SecretResolveResultEvent") -> AppSecretAnswer:
    """Translate an application's ``secret.resolve.result`` into an :class:`AppSecretAnswer`.

    An ``ok`` status with no ``value`` is downgraded to ``error`` — the
    application claimed success and delivered nothing, which must not read as a
    resolved empty secret (dropping is the honest outcome).
    """
    if result.status == "ok":
        if result.value is None:
            return AppSecretAnswer(
                status="error", detail="application answered ok with no value",
            )
        return AppSecretAnswer(
            status="ok", value=result.value, expires_at=result.expires_at,
        )
    return AppSecretAnswer(status=result.status or "error", detail=result.detail)


#: Connection kinds a WS client can be accepted as.  ``AUTH_KIND_APP`` is the
#: only one that may call the ``ticket.*`` verbs, and the only one that may
#: NOT drive a session — see :meth:`JaatoWSServer._dispatch_client_message`.
AUTH_KIND_OPEN = "open"      #: no auth configured (``--ws-unsafe-no-auth``)
AUTH_KIND_SHARED = "shared"  #: the daemon-wide bearer token
AUTH_KIND_APP = "app"        #: an application credential — bind-only
AUTH_KIND_TICKET = "ticket"  #: a per-user ticket — carries an identity

#: The verbs an app-credential connection may send.  Everything else is
#: refused, which is what makes "bind-only" a property of the transport
#: rather than an instruction in documentation.
TICKET_VERBS = frozenset({
    EventType.TICKET_BIND_REQUEST.value,
    EventType.TICKET_REVOKE_REQUEST.value,
})

#: The bind-channel verbs #1226 adds, both from an app-credential connection:
#: ``secret.resolve.result`` (the app answering the daemon's ``secret.resolve``)
#: and ``secret.reload`` (the app asking the daemon to re-resolve an owner's
#: sessions).  Routed like ``TICKET_VERBS`` — refused for any connection that
#: is not an app credential — so the "bind-only" boundary covers them too.
SECRET_BIND_VERBS = frozenset({
    EventType.SECRET_RESOLVE_RESULT.value,
    EventType.SECRET_RELOAD_REQUEST.value,
})


@dataclass(frozen=True)
class ConnectionAuth:
    """How one accepted connection authenticated, and as whom.

    Produced by :meth:`JaatoWSServer._resolve_connection_auth` during the
    Upgrade, **before** any frame is read, and copied onto the
    :class:`ClientConnection`.  It is decided once: nothing a client sends
    later can change its kind or its identity, which is the whole point of
    #1074 — identity established at connect cannot be declined by omission.

    Attributes:
        kind: One of the ``AUTH_KIND_*`` constants above.
        app_id: The application, for ``app`` and ``ticket`` connections;
            ``None`` otherwise.
        user: The QUALIFIED identity (``"<app_id>:<user>"``) for a ``ticket``
            connection; ``None`` otherwise, including for an ``app``
            connection — an app credential names an application, never a
            person, so attributing a session to it would invent a user.
    """

    kind: str
    app_id: Optional[str] = None
    user: Optional[str] = None


@dataclass
class ClientConnection:
    """Represents a connected client.

    ``user_id`` / ``app_id`` / ``auth_kind`` are the identity the connection
    was ACCEPTED with.  Two ways they get populated, and the difference is
    #1074's subject:

    * At connect, from :class:`ConnectionAuth` — a user ticket resolves to a
      qualified identity before the first frame, so a client cannot decline
      to present one.
    * After connect, by :meth:`JaatoWSServer.set_client_user` — the
      jaato-premium SSO route, where the client sends a JWT as a message and
      an extension validates it.  Unchanged, and still opt-in by nature: a
      client on that route that never sends the message stays unattributed.
    """
    websocket: ServerConnection
    client_id: str
    connected_at: str
    subscriptions: Set[str]  # Event types to receive (empty = all)
    user_id: Optional[str] = None  # Authenticated user identity (set by auth middleware)
    app_id: Optional[str] = None   # Binding application (#1074), when known
    auth_kind: str = AUTH_KIND_OPEN  # One of the AUTH_KIND_* constants


class JaatoWSServer:
    """WebSocket server wrapping JaatoServer.

    Handles:
    - Multiple client connections
    - Event broadcasting
    - Request routing
    - Connection lifecycle

    Example:
        server = JaatoWSServer(host="localhost", port=8080)

        # Option 1: Run standalone
        asyncio.run(server.start())

        # Option 2: Start in background
        await server.start_background()
        # ... do other things ...
        await server.stop()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        workspace_root: Optional[str] = None,
        apparmor: Optional[bool] = None,
        cgroups: Optional[bool] = None,
        cgroups_root: str = "/sys/fs/cgroup/jaato",
        default_template: str = "default",
        workspace_max_age: int = 86400,
        ssl_context: Optional[ssl.SSLContext] = None,
        required_token: Optional[str] = None,
        app_credentials: Optional[AppCredentialStore] = None,
        max_message_size: Optional[int] = None,
    ):
        """Initialize the WebSocket server.

        Args:
            host: Host to bind to.
            port: Port to bind to.
            workspace_root: Root directory for workspaces. Remote clients select
                from subdirectories; each workspace has its own .env file that
                determines the provider.
            apparmor: Enable AppArmor confinement for provisioned workspaces.
                ``None`` (default) auto-detects: confine when available, else
                log a WARNING and degrade to directory sandboxing.  ``True``
                *requires* AppArmor — :meth:`start` raises rather than start
                unconfined.  ``False`` disables confinement.  Setting
                ``JAATO_REQUIRE_APPARMOR=1`` promotes ``None`` to ``True``;
                combining it with ``apparmor=False`` is a contradiction that
                :meth:`start` rejects.
            cgroups: Enable per-session cgroup v2 runtime limits.
                ``None`` (default) auto-detects availability.  ``True`` requires
                cgroup v2 with the configured root delegated.  ``False`` disables
                kernel-enforced limits — application-layer limits
                (``tool_timeout_seconds``, ``max_output_bytes``) still apply.
                Orthogonal to ``apparmor``: AppArmor controls *what's reachable*,
                cgroups controls *how much can be consumed*.
            cgroups_root: Parent cgroup v2 directory.  Must already exist with
                ``memory``, ``pids``, and ``cpu`` listed in
                ``cgroup.subtree_control`` (operator setup; see
                :mod:`server.cgroups`).
            default_template: Name of the default workspace template to copy
                when auto-provisioning (default: ``"default"``).
            workspace_max_age: Maximum age in seconds for provisioned workspaces
                before the reaper removes them (default: 86400 = 24h).
            ssl_context: Optional ``ssl.SSLContext`` for TLS. When provided the
                server listens on ``wss://`` instead of ``ws://``. Use
                :func:`load_tls_context` to build one from ``servers.json``.
            required_token: Bearer token clients must present in the WebSocket
                Upgrade request to be accepted. ``None`` disables auth (open
                accept). Clients present the token via either an
                ``Authorization: Bearer <token>`` header (Python/curl/proxies)
                or a ``?token=<token>`` query parameter (browsers, which
                cannot set custom headers on ``new WebSocket(...)``). The
                stored value is the SHA-256 digest only — the plaintext is
                discarded after construction. Tokens are compared with
                :func:`hmac.compare_digest`.
            app_credentials: Optional :class:`~server.ws_tickets.AppCredentialStore`
                of long-lived **application** credentials (#1074). Each
                authorises one connection to call ``ticket.bind`` /
                ``ticket.revoke`` and nothing else — it cannot open a
                session. ``None`` or an empty store is the default and the
                pre-#1074 posture exactly: no connection can be an
                app-credential connection, no ticket can exist, and
                :meth:`_resolve_connection_auth` collapses to the single
                shared-digest comparison it has always been.
            max_message_size: The largest single WebSocket message accepted,
                in bytes (``None`` = :data:`DEFAULT_WS_MAX_MESSAGE_SIZE`).
                A larger message closes the connection with 1009, which is
                a ``websockets`` rule, not a jaato one; the value is
                advertised in ``ConnectedEvent.server_info`` so clients
                check against it before sending.
        """
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        self.host = host
        self.port = port
        self._ssl_context = ssl_context
        self._max_message_size: int = max_message_size or DEFAULT_WS_MAX_MESSAGE_SIZE
        if self._max_message_size < LEGACY_WS_MAX_MESSAGE_SIZE:
            raise ValueError(
                f"max_message_size {self._max_message_size} is below the "
                f"{LEGACY_WS_MAX_MESSAGE_SIZE}-byte floor"
            )
        self._workspace_root = workspace_root
        # Stored as digest only — plaintext token never lives on the
        # instance after construction. ``None`` means auth is disabled
        # and every connection is accepted (legacy behaviour).
        self._expected_token_digest: Optional[bytes] = (
            hashlib.sha256(required_token.encode("utf-8")).digest()
            if required_token
            else None
        )

        # Application credentials and the tickets they mint (#1074).
        # The store holds digests only; the registry holds digests and
        # never a plaintext ticket. Both are empty on a daemon that
        # configured no app credentials, which is what makes the feature
        # opt-in: `_app_credentials` falsy means no connection can ever
        # be an app-credential connection, so no ticket can ever be
        # bound, so the ticket tier of the auth resolver is unreachable.
        self._app_credentials: AppCredentialStore = (
            app_credentials or AppCredentialStore({})
        )
        self._ticket_registry = TicketRegistry()
        if self._app_credentials:
            logger.info(
                "WS app credentials loaded for %d application(s): %s — "
                "these may call ticket.bind/ticket.revoke and may NOT open "
                "a session",
                len(self._app_credentials.app_ids()),
                ", ".join(self._app_credentials.app_ids()),
            )

        # #1226: outstanding ``secret.resolve`` requests, request_id -> Future.
        # The daemon sends a ``secret.resolve`` to an app-credential connection
        # (from an executor thread, via run_coroutine_threadsafe) and blocks on
        # the future; the ``secret.resolve.result`` frame the application sends
        # back resolves it.  A thread lock rather than the asyncio lock, because
        # the transport is entered from a worker thread and the result is
        # delivered on the event loop.
        self._pending_secret_resolves: Dict[
            str, "concurrent.futures.Future[SecretResolveResultEvent]"
        ] = {}
        self._pending_secret_lock = threading.Lock()
        # The in-tree app:// resolver, handed to every session this server's
        # SessionManager constructs.  Its two callables are bound methods, so
        # it is safe to build before the event loop or workspace manager exist.
        self._app_secret_resolver = AppSecretResolver(
            owner_of=self._owner_for_workspace_path,
            transport=self._resolve_app_secret_over_bind_channel,
        )

        # Connection interceptors registered by daemon extensions.
        # See ``set_connection_interceptor()`` for the protocol.
        self._interceptors: list = []

        # Server state
        self._server: Optional[Any] = None
        self._clients: Dict[str, ClientConnection] = {}
        self._client_counter = 0
        self._lock = asyncio.Lock()

        # Workspace manager (if workspace_root provided)
        self._workspace_manager: Optional[WorkspaceManager] = None

        # Workspace provisioner for auto-provisioning session workspaces
        self._provisioner: Optional[WorkspaceProvisioner] = None

        # AppArmor manager for per-session confinement
        self._apparmor: Optional[AppArmorManager] = None
        self._apparmor_mode = apparmor  # None=auto, True=required, False=disabled

        # Require-confinement opt-in: JAATO_REQUIRE_APPARMOR promotes the
        # auto (None) mode to required (True), so the startup gate fails
        # closed rather than silently degrading to directory sandboxing.
        # An explicit --no-apparmor (False) is a direct contradiction and
        # surfaces as a config error at start() rather than being
        # silently overridden.
        self._apparmor_required_env = _env_flag("JAATO_REQUIRE_APPARMOR")
        if self._apparmor_required_env and self._apparmor_mode is None:
            self._apparmor_mode = True

        # Cgroups manager for per-session runtime limits.  Sibling to
        # AppArmor — same lifecycle, same graceful degradation.
        self._cgroups: Optional["CgroupsManager"] = None  # noqa: F821 (lazy import)
        self._cgroups_mode = cgroups
        self._cgroups_root = cgroups_root

        self._default_template = default_template
        self._workspace_max_age = workspace_max_age

        # Per-client provisioned workspace tracking
        self._client_provisioned: Dict[str, ProvisionedWorkspace] = {}

        # Workspace ID → session manager session ID mapping for AppArmor
        # teardown.  Profiles are provisioned under the session manager's ID
        # but the workspace reaper only knows workspace IDs.
        self._workspace_to_session_id: Dict[str, str] = {}

        # Core server (runs in thread)
        self._jaato_server: Optional[JaatoServer] = None
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()

        # Daemon-mode command routing.
        # When running as part of JaatoDaemon, the command router handles
        # session/tool/auth commands.  When None, the WS server handles
        # commands directly via JaatoServer (standalone mode).
        self._command_router = None  # Set by set_command_router()
        self._event_sink_adapter: Optional[WSEventSinkAdapter] = None

        # Extension message handlers: {type_string: async callback(ws, message_dict)}
        # Registered by daemon extensions for custom WS message types.
        self._message_handlers: Dict[str, Any] = {}

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

    def set_command_router(self, router) -> None:
        """Set the daemon-mode command router.

        When set, incoming ``CommandRequest``, ``SendMessageRequest``, etc.
        are delegated to the ``CommandRouter`` instead of being handled
        directly by a per-WS ``JaatoServer``.

        Called by ``JaatoDaemon.start()`` after constructing the router.
        Registers a session hook to apply AppArmor confinement and set
        ``sandbox_mode`` on each newly created session, and hands this
        server's ``workspace_root`` to the router's session manager
        (``set_managed_workspace_root``, #1280) so the session manager's own
        runner-spawn path treats the same workspaces as daemon-managed as
        the pre-init hook does.

        Args:
            router: ``CommandRouter`` instance.
        """
        self._command_router = router

        # AppArmor session wiring runs in TWO phases (server 0.6.49+):
        #
        # 1. Pre-initialize hook (``_apparmor_pre_init_hook``):
        #    - Runs BEFORE ``server.initialize()`` so the profile is
        #      already loaded by the time ``configure()`` runs prefetch
        #      scripts.  Closes the gap where prefetch ran unconfined.
        #    - Does ONLY profile provisioning — needs ``workspace_path``
        #      (passed directly to the hook), nothing from the
        #      initialized server.
        #
        # 2. Post-initialize hook (``_apparmor_session_hook``):
        #    - Runs AFTER ``server.initialize()`` succeeds.
        #    - Applies confinement to the executor (depends on
        #      ``server._jaato`` being constructed), wires reference
        #      authorizer, sets cgroup runtime limits.
        #    - Skips profile provisioning since the pre-init hook
        #      already loaded it (idempotent ``apparmor_parser -r``
        #      would also work but the skip avoids the redundant
        #      subprocess call).
        #
        # Note: self._apparmor is initialized lazily in start(), which
        # runs after set_command_router(), so we check it at hook
        # execution time rather than registration time.
        ws_server = self
        sm = router._session_manager
        # #1280: the session manager's own spawn path (the one a WUI
        # ``session.new`` takes) needs this root as much as the hook below.
        _hand_managed_root_to(sm, getattr(self, "_workspace_root", None))

        def _apparmor_pre_init_hook(
            server: JaatoServer,
            session_id: str,
            workspace_path: Optional[str],
            client_id: Optional[str] = None,
        ) -> None:
            """Spawn the per-session runner unconditionally for
            WS-provisioned sessions; layer apparmor confinement atop
            iff the kernel module is available.

            Phase 3 §7a (WS counterpart to the IPC always-spawn
            refactor in ``SessionManager._provision_ipc_apparmor_and_spawn_runner``).
            Pre-§7a this skipped both apparmor AND spawn when
            apparmor was unavailable; post-§7a the spawn is
            unconditional so the runner-RPC dispatch surface is
            always available for the seat-flip's ``self._jaato.X``
            migrations.

            Lifecycle:
            1. Skip when no workspace_path (no cwd target for the
               runner).
            2. Skip when workspace is NOT under the WS server's
               workspace_root (IPC / user-CWD session — IPC hook
               handles its own).
            3. Resolve the daemon loop; skip if unavailable
               (start() should always have captured it; defensive
               only).
            4. **Apparmor (opt-in via host availability)**: if the
               WS server has an :class:`AppArmorManager` and it's
               available, provision the per-session profile.  On
               provisioning failure, log a warning and continue
               unconfined.
            5. **Spawn (unconditional)**: spawn the runner with the
               provisioned profile (or empty + ``disable_confine=True``
               if apparmor is unavailable / failed).  Spawn failure
               logs a warning and the session falls back to
               in-process tool execution.

            ``client_id`` is unused by the WS gate (which keys off
            workspace-under-WS-root rather than per-client opt-in)
            but accepting it keeps the dispatcher's modern path
            engaged without falling through to the legacy compat
            branch in ``_run_pre_initialize_hooks``.

            Server 0.6.50+: ALSO stashes the confine-context factory
            on the server so ``configure()``'s dynamic-instructions
            expansion wraps prefetch scripts in
            ``apparmor_confine(profile)``.  The factory is propagated
            onto the runtime once ``initialize()`` constructs it
            (see ``core.py`` step 2).
            """
            if not workspace_path:
                # #1296: this hook fires for EVERY session bootstrap on
                # this daemon, not only WS-provisioned ones (it is
                # registered on the shared, transport-agnostic
                # ``_pre_initialize_hooks`` list), so a session created
                # with no workspace_path at all — an IPC/embedded/API
                # session that never selected one — reaches here
                # routinely.  DEBUG, not WARNING: making this branch loud
                # would reproduce exactly the log-spam this issue exists
                # to avoid, one level up from branch (c) below.
                logger.debug(
                    "AppArmor pre-init: no workspace_path for session %s "
                    "— nothing to confine, skipping",
                    session_id,
                )
                return

            # Gate: only WS-provisioned sessions (workspace under
            # WS server's root) take this path.  Non-WS sessions
            # (IPC / user-CWD) are handled by the IPC hook.
            try:
                ws_workspace_root = os.path.realpath(ws_server._workspace_root)
                sess_workspace = os.path.realpath(workspace_path)
            except OSError as exc:
                # #1296: unlike the two branches around it, this one is
                # NOT the routine "this session isn't mine" exit — it
                # only fires once a real workspace_path was handed in and
                # the kernel refused to resolve it (or the WS server's
                # own configured root cannot be resolved), which is
                # anomalous rather than a normal session shape.  WARNING,
                # naming both paths so a broken symlink or a vanished
                # root is diagnosable from the log alone.
                logger.warning(
                    "AppArmor pre-init: realpath failed for session %s "
                    "(workspace_path=%r, ws_workspace_root=%r): %s: %s "
                    "— skipping AppArmor confinement for this session",
                    session_id, workspace_path, ws_server._workspace_root,
                    type(exc).__name__, exc,
                )
                return
            if not (
                sess_workspace == ws_workspace_root
                or sess_workspace.startswith(ws_workspace_root + os.sep)
            ):
                # #1296: the ROUTINE exit — every non-WS session
                # (IPC / user-CWD) takes this branch, since the hook is
                # registered for every session bootstrap regardless of
                # transport.  DEBUG only, so it stays greppable on
                # demand without adding one line per ordinary session
                # creation at the daemon's default log level.
                logger.debug(
                    "AppArmor pre-init: session %s workspace %s is not "
                    "under the WS server's workspace_root %s — IPC or "
                    "user-CWD session, not WS-provisioned",
                    session_id, sess_workspace, ws_workspace_root,
                )
                return  # IPC or user-CWD session — not WS-provisioned

            daemon_loop = ws_server._event_loop
            if daemon_loop is None:
                logger.warning(
                    "AppArmor pre-init: ws_server has no daemon loop "
                    "captured — runner not spawned for session %s",
                    session_id,
                )
                return

            # ----- Apparmor (opt-in via host availability) -----
            apparmor = ws_server._apparmor
            profile_name = ""  # empty = unconfined (disable_confine=True)
            # #1253: a WS-provisioned session is CONFIGURED for confinement
            # whenever the daemon holds an available AppArmorManager — there
            # is no per-session WS opt-out; every such session gets a profile.
            # That fact is SEPARATE from whether ``profile_name`` ends up
            # populated: if provisioning fails, profile_name stays empty but
            # confinement was still required, and spawning a runner then would
            # serve model-driven work with NO kernel boundary while the
            # session record still reports ``sandbox_mode: apparmor`` — a
            # silent bypass (#1100 deferred this; the live evidence is #1253).
            # ``confinement_required`` is the invariant's discriminator: it
            # rides the envelope to the runner (defence in depth), and below
            # it turns a provisioning/spawn failure into a REFUSED session
            # rather than an unconfined one.
            confinement_required = (
                apparmor is not None and apparmor.is_available()
            )
            if confinement_required:
                # Phase 0/1 (template v20+, 2026-05-16): resolve plugin-
                # contributed rules so the WS-spawned session profile
                # mirrors the IPC path's plugin-contribution flow.
                # Without this, references plugin's HF + torch grants
                # (Phase 1) would be present in IPC but missing in WS
                # sessions.
                from jaato_server.server.apparmor import resolve_plugin_apparmor_rules
                plugin_rules = resolve_plugin_apparmor_rules(
                    server=server,
                    profile=getattr(server, "_profile", None),
                    session_id=session_id,
                    workspace_path=workspace_path,
                    # #1293: was hardcoded ``None``.  ``server.config_root``
                    # is already set by ``_construct_and_initialize_server``
                    # (BEFORE pre-init hooks run) from the same
                    # ``envelope.config_root`` the IPC path
                    # (``_provision_ipc_apparmor_and_spawn_runner``) threads
                    # through — reading it here keeps this hook's plugin
                    # rule set (and thus the rendered profile) consistent
                    # with what the IPC path would have computed for the
                    # same session, instead of silently omitting any
                    # config_root-relative grant a plugin contributes.
                    config_root=getattr(server, "config_root", None),
                    managed_workspace_root=ws_workspace_root,
                )
                # #1033: the profile is named after the BOUNDARY, so a
                # pre-warm slot that already wears it needs no
                # ``aa_change_profile`` — which it could not perform for
                # the threads it already has (#1023).
                if apparmor.provision_profile(
                    session_id, workspace_path,
                    plugin_rules=plugin_rules,
                    confinement_id=apparmor.confinement_id_for_boundary(
                        workspace_path, plugin_rules=plugin_rules),
                ):
                    profile_name = apparmor.get_profile_name(session_id)
                else:
                    # #1253 FAIL CLOSED: confinement was required and the
                    # profile did NOT provision.  Refuse the session rather
                    # than spawn a runner that would serve work unconfined.
                    # ``_record_bootstrap_refusal`` is what
                    # ``session_manager.initialize_or_refuse`` reads after the
                    # pre-init hooks run, so recording the failure here (and
                    # returning without spawning) turns the whole session into
                    # a named ``RunnerBootstrapFailed`` refusal.  This is the
                    # ordering-note posture #1014 established for a weakened
                    # boundary: never silent, never a downgrade to unconfined.
                    logger.warning(
                        "AppArmor pre-init: provision_profile failed for "
                        "session %s but confinement is required (an AppArmor "
                        "profile could not be loaded) — REFUSING the session "
                        "rather than spawning an unconfined runner (#1253)",
                        session_id,
                    )
                    _record_bootstrap_refusal(
                        server,
                        "AppArmor confinement required but profile "
                        "provisioning failed; refusing to spawn an "
                        "unconfined runner (#1253)",
                    )
                    return

            # ----- Cgroups (opt-in via host availability) -----
            # Phase 3 §7d: provision the per-session cgroup BEFORE
            # spawn so the forked runner can be migrated into it
            # pre-exec.  Child processes (cli, interactive_shell
            # PTY children) inherit by default per cgroup-v2 kernel
            # contract.  Pre-§7d the cgroup was provisioned in the
            # post-init session_hook and the daemon-side
            # set_runtime_limits wired plugin-level attach_cb
            # preexec_fn's; post-§7d the runner subprocess is
            # already in the cgroup, so plugin-level preexec_fn's
            # are no-ops via inheritance.
            cgroups = ws_server._cgroups
            cgroup_attach: Optional[Callable[[], None]] = None
            cgroup_profile = getattr(server, "_profile", None)
            cgroup_limits = (
                getattr(cgroup_profile, "runtime_limits", None)
                if cgroup_profile else None
            )
            if (
                cgroup_limits is not None
                and cgroups is not None
                and cgroups.is_available()
            ):
                try:
                    if cgroups.provision_cgroup(session_id, cgroup_limits):
                        ws_workspace_id = os.path.basename(sess_workspace)
                        ws_server._workspace_to_session_id[ws_workspace_id] = session_id
                        if cgroup_limits.has_kernel_limits():
                            logger.info(
                                "Cgroup runtime limits applied to session "
                                "%s pre-spawn (memory=%s pids=%s "
                                "cpu_weight=%s)",
                                session_id, cgroup_limits.memory_max_mb,
                                cgroup_limits.pids_max,
                                cgroup_limits.cpu_weight,
                            )
                    cgroup_attach = cgroups.make_attach_callback(session_id)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "Cgroup pre-spawn provisioning failed for "
                        "session %s (%s: %s) — runner will spawn in the "
                        "daemon's cgroup",
                        session_id, type(exc).__name__, exc,
                    )

            # Phase 3 §7a: spawn the runner regardless of apparmor
            # outcome.  The dispatch surface is always available;
            # confinement is layered atop iff apparmor succeeded.
            # Phase 3 §7d: ``cgroup_attach`` (when non-None) is
            # invoked between fork() and exec() to migrate the
            # runner into the per-session cgroup.
            try:
                from jaato_server.server.runner_spawn import (
                    spawn_session_runner,
                    dispatch_bootstrap_envelope,
                )
                spawn_session_runner(
                    server=server,
                    session_id=session_id,
                    workspace_path=workspace_path,
                    profile_name=profile_name,
                    daemon_loop=daemon_loop,
                    disable_confine=(profile_name == ""),
                    cgroup_attach=cgroup_attach,
                    pool_manager=getattr(ws_server, "_pool_manager_ref", None),
                    # Phase 2 cascade-sharing (server 0.6.144+):
                    # cascade_driver_id stashed on the server during
                    # _construct_and_initialize_server (see
                    # session_manager.py — same pattern as _agent_params).
                    cascade_driver_id=getattr(server, "_cascade_driver_id", None),
                    # #1225: this hook only runs for WS-provisioned sessions
                    # under the server's workspace_root (gated at the top), so
                    # passing that root turns the workspace HOME default on
                    # for exactly the sessions the daemon manages.
                    managed_workspace_root=ws_workspace_root,
                )
            except Exception as exc:  # noqa: BLE001 — spawn boundary
                # #1253: a confinement-required session's spawn failure is
                # REFUSED (an in-process fallback would run the model's tools
                # unconfined in the daemon process); a genuinely-unconfined
                # session keeps its in-process fallback.  The decision lives
                # in ``_report_confined_spawn_failure`` so the hook stays at
                # its complexity baseline (the #812 / #1167 / #1179 move).
                _report_confined_spawn_failure(
                    server, session_id, exc, confinement_required)
                return

            # Phase 3 §7c step 2: dispatch the session.bootstrap RPC
            # so the runner-side JaatoSession host is populated.
            # Pre-§7c-step-2 the WS path skipped this dispatch
            # entirely (only IPC sent the envelope), leaving every
            # WS session with a NULL runner-side host.  Failure is
            # logged but does not propagate — the daemon-side
            # JaatoSession is still authoritative during the §7c
            # rollout window.
            dispatch_bootstrap_envelope(
                server=server,
                session_id=session_id,
                workspace_path=workspace_path,
                profile_name=profile_name,
                # #1253: carry the invariant to the runner.  On this core
                # path provisioning already succeeded (a failure returned
                # above), so profile_name is populated and the runner-side
                # gate is a no-op; the flag is the defence-in-depth backstop
                # for any spawn path — a pre-warm slot, or a peer daemon —
                # that reaches the runner with confinement required and no
                # profile.  ``_maybe_self_confine`` then RAISES rather than
                # running the session unconfined.
                confinement_required=confinement_required,
                # #1225: fold the workspace-HOME default into the envelope's
                # cli / interactive_shell / notebook configs for this managed
                # workspace.
                managed_workspace_root=ws_workspace_root,
            )

        def _apparmor_session_hook(server: JaatoServer, session_id: str) -> None:
            sess = sm.get_session(session_id)
            if not sess or not sess.workspace_path:
                return

            apparmor = ws_server._apparmor
            cgroups = ws_server._cgroups

            # The "WS-provisioned" gate that AppArmor uses also applies
            # to cgroups: kernel-enforced runtime limits target untrusted
            # WS clients, not local IPC sessions.  We compute it once
            # against whichever manager exists.  If neither is configured,
            # nothing to do.
            anchor = apparmor if apparmor else cgroups
            if anchor is None:
                return

            # WS-provisioned sessions live under the manager's
            # workspace_root.  Reuse the AppArmor manager's path even when
            # AppArmor is disabled — both managers' workspace root is the
            # same WS server workspace_root, so this gate is correct for
            # cgroups too.
            try:
                ws_workspace_root = os.path.realpath(ws_server._workspace_root)
                sess_workspace = os.path.realpath(sess.workspace_path)
            except OSError:
                return
            if not (
                sess_workspace == ws_workspace_root
                or sess_workspace.startswith(ws_workspace_root + os.sep)
            ):
                logger.debug(
                    "Sandbox/limits skipped for non-provisioned session %s "
                    "(workspace %s not under %s — IPC or user-CWD session)",
                    session_id, sess_workspace, ws_workspace_root,
                )
                return

            # ---------- Cgroups: per-session runtime limits ----------
            # Provisioned independently of AppArmor — a session may have
            # limits without sandboxing (or vice versa).  The limits
            # come from the SubagentProfile attached to the JaatoServer.
            #
            # Whether or not the kernel layer is available, we still
            # hand the app-layer ``RuntimeLimits`` to the executor so
            # that ``tool_timeout_seconds`` and ``max_output_bytes`` are
            # enforced at the Python layer.  The attach callback is a
            # no-op when the cgroup wasn't created (or doesn't exist),
            # so passing it through unconditionally is safe.
            profile = getattr(server, "_profile", None)
            limits = getattr(profile, "runtime_limits", None) if profile else None
            if limits is not None:
                attach_cb = None
                event_reader = None
                if cgroups and cgroups.is_available():
                    if cgroups.provision_cgroup(session_id, limits):
                        ws_workspace_id = os.path.basename(sess.workspace_path)
                        ws_server._workspace_to_session_id[ws_workspace_id] = session_id
                        if limits.has_kernel_limits():
                            logger.info(
                                "Cgroup runtime limits applied to session %s "
                                "(memory=%s pids=%s cpu_weight=%s)",
                                session_id, limits.memory_max_mb,
                                limits.pids_max, limits.cpu_weight,
                            )
                    attach_cb = cgroups.make_attach_callback(session_id)
                    event_reader = cgroups.make_event_reader(session_id)
                # Hand attach_cb + limits + event_reader to the executor.
                # attach_cb and event_reader are no-ops when cgroups are
                # unavailable, so the call is unconditional once limits
                # exist on the profile.  event_reader feeds OTel
                # telemetry (jaato.cgroup.oom_kill_delta etc.) via the
                # executor's per-tool-call snapshot/diff in execute().
                server.set_runtime_limits(attach_cb, limits, event_reader)

            # ---------- AppArmor: per-session sandboxing ----------
            # AppArmor confinement is intended for WS-provisioned sessions
            # only — multi-tenant deployments where untrusted or semi-
            # trusted clients attach over WebSocket and need kernel-
            # enforced filesystem isolation (see ``docs/apparmor-setup.md``).
            #
            # IPC sessions run on behalf of the local user, whose shell
            # already has full filesystem access.  Confining them does not
            # add security (the user can bypass by invoking python directly)
            # but DOES create real correctness problems:
            #
            # - ``~/.jaato/*_auth.json`` and any path outside the session's
            #   workspace become unreadable — breaks verify_auth, reference
            #   selection, external ``sandbox add`` paths, etc.
            # - Thread-pool workers leak profile state across sessions when
            #   the restore-to-unconfined transition fails (it's gated on a
            #   file-write rule that the profile doesn't grant).
            if not apparmor or not apparmor.is_available():
                if apparmor is not None:
                    sess.sandbox_mode = SANDBOX_MODE_SOFT
                return

            # Profile provisioning happens in the pre-initialize hook
            # (server 0.6.49+) so prefetch runs confined.  Re-run here
            # for resilience: if the pre-init hook failed (e.g.
            # AppArmor unavailable at that point), this catches it
            # and downgrades cleanly to soft mode.  ``apparmor_parser -r``
            # is idempotent so the redundant call is safe when both
            # phases succeed.  Plugin-rules resolution mirrors the
            # pre-init hook (template v20+).
            from jaato_server.server.apparmor import resolve_plugin_apparmor_rules
            plugin_rules = resolve_plugin_apparmor_rules(
                server=server,
                profile=getattr(server, "_profile", None),
                session_id=session_id,
                workspace_path=sess.workspace_path,
                # #1293: mirror the pre-init hook — read the server's own
                # resolved config_root rather than hardcoding ``None``.
                config_root=getattr(server, "config_root", None),
                managed_workspace_root=ws_workspace_root,
            )
            if not apparmor.provision_profile(
                session_id, sess.workspace_path,
                plugin_rules=plugin_rules,
                confinement_id=apparmor.confinement_id_for_boundary(
                    sess.workspace_path, plugin_rules=plugin_rules),
            ):
                # #1253: reaching here means confinement was REQUIRED for this
                # WS-provisioned session — the host has an available
                # AppArmorManager (the ``is_available`` gate above) and the
                # workspace is under the WS root — yet the profile did not
                # load.  The pre-init hook (#1260) fails this closed BEFORE
                # spawn on the core path; this resilience re-run runs AFTER the
                # runner already spawned, so it cannot un-spawn — but a silent
                # downgrade to ``soft`` is exactly the invisible boundary loss
                # #1253 is about.  Announce it at WARNING (the #1014 posture:
                # never a silent downgrade of a boundary) and record the
                # TRUTHFUL ``soft`` mode, so the record does not claim a
                # boundary the kernel is not enforcing.
                logger.warning(
                    "AppArmor confinement required for session %s (WS-"
                    "provisioned, host supports it) but profile provisioning "
                    "failed in the post-init hook — the runner is NOT kernel-"
                    "confined; recording sandbox_mode=soft rather than "
                    "silently claiming enforcement (#1253/#1014)",
                    session_id,
                )
                sess.sandbox_mode = SANDBOX_MODE_SOFT
                return

            # Phase 2 (confined runner): kernel-level profile is loaded
            # but the daemon does NOT confine its own threads.  Tool
            # execution runs in the per-session runner subprocess
            # (see server/runner/), which self-confines via
            # aa_change_profile against this profile.  The
            # ``tool_hat`` sub-profile transition stays in the runner-
            # side ToolExecutor (Phase 3).
            #
            # Hand over the per-session reference authorizer so the
            # references plugin can mutate the kernel profile when
            # selectReferences grants new readonly paths.  This is a
            # daemon-tier RPC primitive (the runner consumes it via
            # apparmor.add_fragment in Phase 3); the daemon-side
            # state-keeping is unchanged in Phase 2.
            authorizer = ws_server.get_reference_authorizer(session_id)
            if authorizer is not None:
                server.set_reference_authorizer(authorizer)
            # #1014: record the MODE, not merely that a profile loaded.
            # Under ``JAATO_APPARMOR_COMPLAIN`` the kernel logs denials and
            # allows them, and a record asserting ``"apparmor"`` about that
            # session is a durable false claim of enforcement.  Same
            # vocabulary as the IPC path
            # (``SessionManager._provision_apparmor_for_session``).
            sess.sandbox_mode = sandbox_mode_for_profile(
                complain=apparmor.profile_is_complain_mode(session_id),
            )
            # Record mapping so the workspace reaper can teardown
            # the profile by workspace ID.  (Uses the module-level ``os``
            # imported at the top — a local ``import os`` here would make
            # ``os`` function-local throughout ``_apparmor_session_hook``
            # and raise ``UnboundLocalError`` at the earlier
            # ``os.path.realpath`` call.)
            workspace_id = os.path.basename(sess.workspace_path)
            ws_server._workspace_to_session_id[workspace_id] = session_id
            logger.info(
                "AppArmor profile provisioned for session %s (workspace %s); "
                "runner will self-confine on spawn",
                session_id, workspace_id,
            )

        # Register both hooks.  The pre-init hook runs before
        # ``server.initialize()`` so prefetch scripts (and any other
        # configure-time work) execute with the AppArmor profile already
        # loaded.  The post-init hook runs after ``initialize()`` to
        # wire confinement onto the now-constructed executor.
        sm.add_pre_initialize_hook(_apparmor_pre_init_hook)
        sm.add_session_hook(_apparmor_session_hook)

    def set_client_user(self, client_id: str, user_id: str) -> None:
        """Associate an authenticated user identity with a WS client.

        Called by auth middleware (e.g., Keycloak SSO in jaato-premium)
        after the client's identity is verified.  The identity is stored
        on the ``ClientConnection`` and propagated to sessions the client
        creates (``Session.created_by``) and to extension message handler
        callbacks.

        **A connect-established identity is not overwritten.** When the
        connection presented a user ticket (#1074), its identity was resolved
        from a credential the application minted, before the first frame, and
        a later message asserting a different one is refused with a WARNING.
        The two routes are alternatives rather than layers: allowing the
        message to win would reintroduce, as a *replacement*, precisely the
        assertion-over-a-message shape the ticket route exists to remove.
        Every other connection kind behaves exactly as before.

        Args:
            client_id: The WS client ID.
            user_id: Authenticated user identifier (username, email, or sub claim).
        """
        client = self._clients.get(client_id)
        if not client:
            return
        if client.auth_kind == AUTH_KIND_TICKET:
            if user_id != client.user_id:
                logger.warning(
                    "Refusing to re-attribute client %s: its identity was "
                    "established at connect from a bound ticket (%s); a "
                    "message asserting %r does not override it",
                    client_id, client.user_id, user_id,
                )
            return
        client.user_id = user_id

    def get_client_user(self, client_id: str) -> Optional[str]:
        """Get the authenticated user identity for a WS client.

        Returns:
            The user ID string, or ``None`` if not authenticated.
        """
        client = self._clients.get(client_id)
        return client.user_id if client else None

    def visible_workspace_paths(self, client_id: str) -> Optional[List[str]]:
        """See ``WSEventSinkAdapter.visible_workspace_paths``."""
        if not self._workspace_manager:
            return None
        user = self.get_client_user(client_id)
        if user is None:
            return None
        return [ws.path for ws in self._workspace_manager.list_workspaces(for_user=user)
                if ws.path]

    def _sessions_loaded_in(self, workspace_path: str) -> List[str]:
        """Ids of the LOADED sessions running under *workspace_path*."""
        if not self._command_router:
            return []
        sm = self._command_router._session_manager
        root = os.path.normpath(workspace_path)
        found: List[str] = []
        for info in sm.list_sessions():
            if not info.is_loaded or not info.workspace_path:
                continue
            wp = os.path.normpath(info.workspace_path)
            if wp == root or wp.startswith(root + os.sep):
                found.append(info.session_id)
        return found

    def register_message_handler(
        self,
        message_type: str,
        handler: Any,
    ) -> None:
        """Register an async handler for a custom WS message type.

        Daemon extensions call this (via ``_ExtensionContext.ws_server``)
        to handle custom message types on client WS connections — the same
        connections used for built-in session/command messages.

        The handler is called when a client sends a JSON message whose
        ``type`` field matches *message_type* and no built-in handler
        recognises it.  Built-in types cannot be overridden.

        Args:
            message_type: The ``type`` field value to match
                (e.g., ``"reconnect.snapshot"``).
            handler: Async callback with signature
                ``async def handler(ws, message, user, client_id) -> None``.
                *ws* is the raw ``websockets.ServerConnection``;
                *message* is the parsed JSON dict;
                *user* is the authenticated user ID (``None`` if unauthenticated);
                *client_id* is the server-assigned client identifier.

        Example::

            async def handle_snapshot(ws, message, user, client_id):
                snapshot = build_snapshot(message["session_id"], user)
                await ws.send(json.dumps({"type": "reconnect.snapshot", **snapshot}))

            ws_server.register_message_handler("reconnect.snapshot", handle_snapshot)
        """
        if message_type in self._message_handlers:
            logger.warning(
                "Overwriting existing handler for message type '%s'",
                message_type,
            )
        self._message_handlers[message_type] = handler
        logger.debug("Registered WS message handler: %s", message_type)

    def get_event_sink_adapter(self) -> WSEventSinkAdapter:
        """Return (create if needed) the ``WSEventSinkAdapter`` for this server.

        The adapter implements ``EventSink`` and is registered with the
        ``CompositeEventSink`` in ``JaatoDaemon.start()``.
        """
        if self._event_sink_adapter is None:
            self._event_sink_adapter = WSEventSinkAdapter(self)
        return self._event_sink_adapter

    async def start(self) -> None:
        """Start the server and block until shutdown.

        This method:
        1. Initializes WorkspaceManager (workspace_root required)
        2. Starts the WebSocket server
        3. Runs event broadcasting loop
        4. Blocks until stop() is called

        JaatoServer initialization is deferred until a workspace is selected
        and configured by the client.
        """
        # Initialize workspace manager if root is provided.
        # When running in daemon mode without workspace_root, the WS server
        # still accepts peer gossip connections and IPC-attached client events.
        if self._workspace_root:
            self._workspace_manager = WorkspaceManager(self._workspace_root)
            self._workspace_manager.discover_workspaces()
            logger.info(f"Workspace mode enabled, root: {self._workspace_root}")

            # Initialize workspace provisioner for auto-provisioning
            self._provisioner = WorkspaceProvisioner(
                self._workspace_root,
                default_template=self._default_template,
            )

            # Initialize AppArmor manager.  Pass the daemon's main
            # asyncio loop (we're inside ``async start()``) so that
            # AppArmor mutations triggered from confined worker
            # threads — e.g. selectReferences fragment writes — get
            # dispatched here for execution in the unconfined main
            # loop, instead of EACCESing on the file write.
            self._apparmor = AppArmorManager(
                workspace_root=self._workspace_root,
                loop=asyncio.get_running_loop(),
            )
            if self._apparmor_mode is False:
                if self._apparmor_required_env:
                    # Contradiction: env requires confinement, flag disables
                    # it. Refuse rather than guess which the operator meant.
                    raise RuntimeError(
                        "Contradictory AppArmor configuration: "
                        "JAATO_REQUIRE_APPARMOR=1 requires confinement but "
                        "--no-apparmor disables it. Resolve by unsetting one."
                    )
                logger.info("AppArmor confinement disabled by configuration")
                self._apparmor = None
            elif self._apparmor_mode is True and not self._apparmor.is_available():
                # Required-but-unavailable: fail closed. The operator
                # explicitly opted into confinement (--apparmor or
                # JAATO_REQUIRE_APPARMOR); starting unconfined would leave
                # only the bypassable directory-sandbox heuristic, which is
                # not what "required" means. Refuse to start.
                reason = self._apparmor.unavailable_reason or "reason unknown"
                raise RuntimeError(
                    "AppArmor confinement required but not available "
                    f"({reason}). Refusing to start unconfined — workspace "
                    "isolation would rely on directory sandboxing only. "
                    "Install/enable AppArmor (and the apparmor_parser "
                    "sudoers rule), or drop the requirement with "
                    "--no-apparmor / unset JAATO_REQUIRE_APPARMOR to accept "
                    "directory-sandbox-only isolation."
                )
            elif self._apparmor and self._apparmor.is_available():
                logger.info("AppArmor confinement enabled")

            # Initialize cgroups manager (orthogonal to AppArmor — runtime
            # limits, not sandboxing).  Same auto-detect / required /
            # disabled tristate as AppArmor.
            self._cgroups = CgroupsManager(root=self._cgroups_root)
            if self._cgroups_mode is False:
                logger.info("Cgroups runtime limits disabled by configuration")
                self._cgroups = None
            elif self._cgroups_mode is True and not self._cgroups.is_available():
                logger.warning(
                    "Cgroups runtime limits required but not available — "
                    "kernel-enforced caps will be skipped (app-layer caps still apply)"
                )
            elif self._cgroups and self._cgroups.is_available():
                logger.info("Cgroups runtime limits enabled (root=%s)",
                            self._cgroups_root)

            # Start workspace reaper
            def _on_workspace_reaped(workspace_id: str) -> None:
                # Look up the session manager's session ID from the
                # workspace ID.  Both AppArmor profiles and cgroups are
                # provisioned under the session manager's ID, not the
                # workspace UUID.
                # ``pop`` here would race with a still-active reverse
                # lookup if we extend teardown later, so we read first
                # and pop only at the end.
                session_id = self._workspace_to_session_id.get(
                    workspace_id, workspace_id
                )
                if self._apparmor and self._apparmor.is_available():
                    # #1033: a boundary-derived profile outlives its
                    # session — a pooled slot may still be idle inside
                    # it, waiting for the next session of its cascade —
                    # so ask the pool before unloading.  Without a pool
                    # this is exactly the call it always was.
                    _pool = getattr(self, "_pool_manager_ref", None)
                    _name = self._apparmor.get_profile_name(session_id)
                    if _pool is not None and _pool.profile_in_use(_name):
                        logger.info(
                            "Workspace reaper: leaving AppArmor profile %s "
                            "loaded — a pooled runner slot is still "
                            "confined to it", _name,
                        )
                    else:
                        self._apparmor.teardown_profile(session_id)
                if self._cgroups and self._cgroups.is_available():
                    self._cgroups.teardown_cgroup(session_id)
                self._workspace_to_session_id.pop(workspace_id, None)

            self._provisioner.start_reaper(
                interval_seconds=3600,
                max_age_seconds=self._workspace_max_age,
                on_teardown=_on_workspace_reaped,
            )

        # Bind event loop for the WSEventSinkAdapter (thread-safe scheduling)
        if self._event_sink_adapter:
            self._event_sink_adapter.bind_loop(asyncio.get_running_loop())

        # Start WebSocket server
        try:
            serve_kwargs: Dict[str, Any] = dict(
                ping_interval=30,
                ping_timeout=10,
                # Without it ``websockets`` applies 1 MiB, and a staged
                # file over that closed the connection mid-upload.
                max_size=self._max_message_size,
            )
            if self._ssl_context:
                serve_kwargs["ssl"] = self._ssl_context

            async with websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                **serve_kwargs,
            ) as server:
                self._server = server
                scheme = "wss" if self._ssl_context else "ws"
                logger.info(
                    f"WebSocket server listening on {scheme}://{self.host}:{self.port} "
                    f"(max message {self._max_message_size} bytes)"
                )

                # Run event broadcaster and wait for shutdown
                broadcast_task = asyncio.create_task(self._broadcast_loop())

                try:
                    await self._shutdown_event.wait()
                finally:
                    broadcast_task.cancel()
                    try:
                        await broadcast_task
                    except asyncio.CancelledError:
                        pass
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                raise OSError(
                    e.errno,
                    f"Cannot start WebSocket server: port {self.port} is already in use",
                ) from None
            raise

        # Cleanup
        if self._jaato_server:
            self._jaato_server.shutdown()

        logger.info("Server stopped")

    async def start_background(self) -> None:
        """Start the server in a background task.

        Returns immediately. Use stop() to shut down.
        """
        asyncio.create_task(self.start())
        # Give server time to start
        await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the server gracefully."""
        self._shutdown_event.set()

        # Stop workspace reaper
        if self._provisioner:
            self._provisioner.stop_reaper()

        # Close all client connections
        async with self._lock:
            for client in list(self._clients.values()):
                try:
                    await client.websocket.close(1001, "Server shutting down")
                except Exception:
                    pass
            self._clients.clear()

    def _on_server_event(self, event: Event) -> None:
        """Callback from JaatoServer - queue event for broadcasting."""
        # This is called from a different thread (model thread)
        # Use asyncio.run_coroutine_threadsafe to safely queue
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                self._event_queue.put(event),
                loop
            )
        except RuntimeError:
            # No event loop running yet - server not started
            pass

    async def _broadcast_loop(self) -> None:
        """Continuously broadcast events to all clients."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for event with timeout (to check shutdown)
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                # Broadcast to all clients
                await self._broadcast(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    async def _broadcast(self, event: Event) -> None:
        """Broadcast an event to all connected clients."""
        if not self._clients:
            return

        message = serialize_event(event)

        async with self._lock:
            disconnected = []

            for client_id, client in self._clients.items():
                try:
                    await client.websocket.send(message)
                except ConnectionClosed:
                    disconnected.append(client_id)
                except Exception as e:
                    logger.error(f"Send error to {client_id}: {e}")
                    disconnected.append(client_id)

            # Remove disconnected clients
            for client_id in disconnected:
                del self._clients[client_id]
                logger.info(f"Client disconnected: {client_id}")

    def set_connection_interceptor(
        self,
        check: Callable,
        handler: Callable,
    ) -> None:
        """Register an interceptor for incoming WebSocket connections.

        Interceptors are evaluated in registration order **before** normal
        client handling.  When ``check(websocket)`` returns ``True``, the
        connection is handed off to ``handler(websocket)`` and never enters
        the regular client flow.

        This is the primary mechanism for daemon extensions (e.g., gossip
        clustering) to route special connections to custom handlers.

        Args:
            check: A callable ``(websocket) -> bool`` that inspects the
                inbound connection (e.g., checking request headers) and
                returns ``True`` if this interceptor should handle it.
            handler: An async callable ``(websocket) -> None`` that takes
                over the connection when ``check`` returns ``True``.
                The handler is responsible for the full connection lifecycle.

        Example (from a daemon extension's ``start()`` method)::

            ws_server.set_connection_interceptor(
                check=lambda ws: (
                    ws.request
                    and ws.request.headers.get("X-My-Header") == "true"
                ),
                handler=self._handle_special_connection,
            )
        """
        self._interceptors.append((check, handler))

    def _extract_presented_token(self, websocket: ServerConnection) -> Optional[str]:
        """Pull a bearer token out of the Upgrade request.

        Two locations are accepted, in priority order:

        1. ``Authorization: Bearer <token>`` header — the conventional
           form. Used by Python clients, curl, and reverse proxies.
        2. ``?token=<token>`` query parameter on the request URI — used
           by browsers, since the standard ``WebSocket`` constructor does
           not let JavaScript set custom request headers.

        Returns the raw presented token (without prefix) or ``None`` if
        no candidate is found.
        """
        request = getattr(websocket, "request", None)
        if request is None:
            return None

        auth = ""
        try:
            auth = request.headers.get("Authorization", "") or ""
        except Exception:
            auth = ""
        if auth.startswith("Bearer "):
            token = auth[len("Bearer "):].strip()
            if token:
                return token

        path = getattr(request, "path", "") or ""
        if "?" in path:
            try:
                query = parse_qs(urlsplit(path).query)
            except Exception:
                query = {}
            values = query.get("token") or []
            if values and values[0]:
                return values[0]

        return None

    def _resolve_connection_auth(
        self,
        websocket: ServerConnection,
        *,
        consume: bool = True,
    ) -> Optional[ConnectionAuth]:
        """Decide whether to accept this Upgrade, and as whom (#1074).

        The one door. Three tiers are consulted in order, and the presented
        credential is hashed exactly once however many it falls through:

        1. **The shared bearer token** — one digest, daemon-wide, compared
           with :func:`hmac.compare_digest`. This tier is byte-identical to
           the pre-#1074 check and is consulted first, so a deployment that
           configures nothing else behaves exactly as it did.
        2. **An application credential** — a dict lookup keyed by the digest,
           returning the ``app_id``. Accepted as ``AUTH_KIND_APP``:
           bind-only, refused every other verb.
        3. **A user ticket** — the same lookup against the
           :class:`~server.ws_tickets.TicketRegistry`, returning a
           :class:`~server.ws_tickets.BoundIdentity`. Accepted as
           ``AUTH_KIND_TICKET`` and, crucially, **carrying an identity before
           the first frame** — which is what makes declining to present one
           unrepresentable rather than merely discouraged.

        A dict lookup is not constant-time, and that is a considered choice
        rather than an oversight: what it compares is a SHA-256 *digest*, so
        a timing signal about it is not a timing signal about the credential
        that produced it. The single expected value that IS compared directly
        stays on :func:`hmac.compare_digest`. See ``server/ws_tickets.py``.

        Auth is considered CONFIGURED when either a shared token or at least
        one app credential exists. With neither, every connection is accepted
        as ``AUTH_KIND_OPEN`` — the ``--ws-unsafe-no-auth`` posture, unchanged.
        Configuring app credentials alone therefore turns auth ON, which is
        the fail-closed direction.

        Args:
            websocket: The inbound connection, for its Upgrade request.
            consume: When ``False``, a single-use ticket is peeked at rather
                than spent. Only :meth:`_check_ws_token` passes ``False``; the
                accept path must consume, or one captured ticket would open
                any number of connections.

        Returns:
            The :class:`ConnectionAuth` to stamp on the connection, or
            ``None`` to reject it with WS code 1008.
        """
        auth_configured = (
            self._expected_token_digest is not None or bool(self._app_credentials)
        )
        if not auth_configured:
            return ConnectionAuth(kind=AUTH_KIND_OPEN)

        presented = self._extract_presented_token(websocket)
        if not presented:
            return None
        digest = credential_digest(presented)

        if self._expected_token_digest is not None and hmac.compare_digest(
            digest, self._expected_token_digest
        ):
            return ConnectionAuth(kind=AUTH_KIND_SHARED)

        app_id = self._app_credentials.lookup(digest)
        if app_id is not None:
            return ConnectionAuth(kind=AUTH_KIND_APP, app_id=app_id)

        identity = self._ticket_registry.resolve_digest(digest, consume=consume)
        if identity is not None:
            return ConnectionAuth(
                kind=AUTH_KIND_TICKET,
                app_id=identity.app_id,
                user=identity.qualified,
            )

        return None

    def _check_ws_token(self, websocket: ServerConnection) -> bool:
        """Whether this connection would be accepted — a NON-consuming probe.

        The boolean half of :meth:`_resolve_connection_auth`, kept because a
        bare "is this credential acceptable" predicate is a reasonable thing
        to ask and because its answer is what the WS auth contract has always
        been documented as. It passes ``consume=False``: a predicate that
        silently spent a single-use ticket would be a trap for every caller
        after the first.

        The accept path calls :meth:`_resolve_connection_auth` directly — it
        needs the identity, not a yes/no, and it must consume.
        """
        return self._resolve_connection_auth(websocket, consume=False) is not None

    async def _handle_client(self, websocket: ServerConnection) -> None:
        """Handle a single client connection.

        Before normal client handling, registered interceptors are checked.
        If any interceptor's ``check`` returns ``True``, the connection is
        handed off to that interceptor's ``handler`` and this method returns.

        After interceptors, connection auth is enforced (if configured) by
        :meth:`_resolve_connection_auth`, and its verdict is STAMPED on the
        :class:`ClientConnection` — so a ticket-authenticated client carries
        its identity from before its first frame (#1074). Failures get an
        immediate 1008 (Policy Violation) close so the client sees a clean
        rejection rather than the connection hanging.
        """
        # Check registered interceptors (e.g., peer gossip connections)
        for check, handler in self._interceptors:
            try:
                if check(websocket):
                    await handler(websocket)
                    return
            except Exception as exc:
                logger.error("Connection interceptor failed: %s", exc)
                return

        # Bearer-token auth gate. Runs after interceptors so that
        # extension-owned connection types (e.g., gossip peers) can
        # implement their own auth.
        auth = self._resolve_connection_auth(websocket)
        if auth is None:
            remote = getattr(websocket, "remote_address", None)
            logger.warning("Rejecting WS client from %s: bearer auth failed", remote)
            try:
                await websocket.close(code=1008, reason="auth failed")
            except Exception:
                pass
            return

        # Assign client ID
        async with self._lock:
            self._client_counter += 1
            client_id = f"client_{self._client_counter}"

            client = ClientConnection(
                websocket=websocket,
                client_id=client_id,
                connected_at=datetime.now(timezone.utc).isoformat(),
                subscriptions=set(),
                # Identity as authenticated at connect (#1074). For every
                # pre-#1074 posture these are the historical defaults:
                # user_id None, app_id None, kind "open"/"shared".
                user_id=auth.user,
                app_id=auth.app_id,
                auth_kind=auth.kind,
            )
            self._clients[client_id] = client

        logger.info(f"Client connected: {client_id} from {websocket.remote_address}")

        # Send connected event
        try:
            server_info = {
                "client_id": client_id,
                "workspace_mode": self._workspace_manager is not None,
                "server_version": _get_server_version(),
                # The size limits a client must respect, so it can refuse a
                # file before sending it instead of having the connection
                # closed under it.  Absent on older daemons, which enforce
                # LEGACY_WS_MAX_MESSAGE_SIZE whatever the staging caps say.
                **self.message_limits(),
            }

            if self._jaato_server:
                server_info["model_provider"] = self._jaato_server.model_provider
                server_info["model_name"] = self._jaato_server.model_name

            connected_event = ConnectedEvent(
                protocol_version=PROTOCOL_VERSION,
                server_info=server_info,
            )
            await websocket.send(serialize_event(connected_event))

            # Handle incoming messages
            async for message in websocket:
                await self._dispatch_client_message(client_id, message)

        except ConnectionClosed as exc:
            self._log_connection_close(client_id, exc)
        except Exception as e:
            import traceback as _tb
            logger.error(f"Client error {client_id}: {e}\n{''.join(_tb.format_exception(e))}")
        finally:
            # Remove client
            async with self._lock:
                if client_id in self._clients:
                    del self._clients[client_id]
            # Detach from session via the transport-agnostic helper
            # (prevents stale client_ids in attached_clients and the
            # downstream inotify leak through workspace_monitor).
            if self._command_router:
                self._command_router.handle_client_disconnect(client_id)
            # Clean up per-client state
            if self._workspace_manager:
                self._workspace_manager.remove_client(client_id)
            self._client_provisioned.pop(client_id, None)
            if self._event_sink_adapter:
                self._event_sink_adapter.remove_client(client_id)
            logger.info(f"Client disconnected: {client_id}")

    # =========================================================================
    # Identity at connect — the ticket bind channel (#1074)
    # =========================================================================

    def _log_connection_close(self, client_id: str, exc: BaseException) -> None:
        """Log how a client connection ended.

        A close is normally uneventful and stays at DEBUG.  A close this
        server was forced into -- 1009 "message too big" above all -- used
        to vanish in a bare ``except ConnectionClosed: pass``, leaving a
        client waiting for a reply to a message the daemon never accepted
        and no trace on this side, so anything outside
        :data:`_ORDINARY_CLOSE_CODES` is logged at WARNING.
        """
        code, text = describe_connection_close(exc)
        if code is None or code in _ORDINARY_CLOSE_CODES:
            logger.debug(f"Client {client_id} connection closed: {text}")
            return
        hint = (
            f" -- a message exceeded max_message_size "
            f"({self._max_message_size} bytes; --ws-max-message-size)"
            if code == 1009 else ""
        )
        logger.warning(f"Client {client_id} connection closed abnormally: {text}{hint}")

    async def _dispatch_client_message(self, client_id: str, message: str) -> None:
        """Route one client frame, honouring the app-credential boundary.

        Sits between the receive loop and :meth:`_handle_message` and answers
        two questions the ordinary dispatcher must not have to:

        1. is this a ``ticket.*`` verb, which belongs to the bind channel
           rather than to a session; and
        2. is this an APP-credential connection, which may send nothing else.

        Gate 2 is what makes "bind-only" a property of the transport. An app
        credential authorises minting identities; letting it also drive a
        session would make it a super-user of every workspace on the daemon,
        attributed to no person. Fail-closed, and central: one check here
        covers every verb, including ones added later.

        A connection that is not an app credential is unaffected — every
        frame reaches :meth:`_handle_message` exactly as before — except that
        a ``ticket.*`` verb from it is answered ``"denied"`` rather than
        falling through to "Unknown request type", so a misconfigured
        integration learns the rule instead of a generic refusal.
        """
        verb = _peek_message_type(message)
        if verb in TICKET_VERBS:
            await self._handle_ticket_message(client_id, verb, message)
            return
        if verb in SECRET_BIND_VERBS:
            await self._handle_secret_bind_message(client_id, verb, message)
            return
        if self._client_auth_kind(client_id) == AUTH_KIND_APP:
            await self._send_error(
                client_id,
                f"'{verb or 'this request'}' is not available on an "
                "application-credential connection: it may only call "
                "ticket.bind / ticket.revoke. Bind a user ticket and "
                "connect with that to open a session.",
            )
            return
        await self._handle_message(client_id, message)

    def _client_auth_kind(self, client_id: str) -> str:
        """The ``AUTH_KIND_*`` this client was accepted with.

        A client that has already disconnected reads as ``AUTH_KIND_OPEN``,
        which is inert: every branch that consults this asks whether the kind
        IS ``AUTH_KIND_APP``, and a vanished client has nothing to refuse.
        """
        client = self._clients.get(client_id)
        return client.auth_kind if client else AUTH_KIND_OPEN

    def get_client_app(self, client_id: str) -> Optional[str]:
        """The application a WS client authenticated as, or ``None``.

        Populated for both credential kinds #1074 introduces — an app
        credential names itself, and a user ticket names the application that
        bound it — and ``None`` for the shared token and for open accept.
        """
        client = self._clients.get(client_id)
        return client.app_id if client else None

    async def _handle_ticket_message(
        self, client_id: str, verb: str, message: str
    ) -> None:
        """Parse and dispatch one ``ticket.*`` frame.

        Refuses any connection that is not an app credential BEFORE parsing
        the body, so the refusal cannot depend on the shape of what was sent.
        The denial names the reason rather than saying only "denied": a
        deployment that has configured no app credentials at all is the
        common case, and "the feature is off here" is a different next step
        from "this connection is the wrong kind".
        """
        app_id = self.get_client_app(client_id)
        if self._client_auth_kind(client_id) != AUTH_KIND_APP or not app_id:
            await self._send_to_client(
                client_id, self._ticket_denied(verb, message)
            )
            return
        try:
            event = deserialize_event(message)
        except (json.JSONDecodeError, ValueError) as exc:
            await self._send_error(client_id, f"Invalid {verb} request: {exc}")
            return
        if isinstance(event, TicketBindRequest):
            await self._send_to_client(
                client_id, self._bind_ticket(app_id, event)
            )
        elif isinstance(event, TicketRevokeRequest):
            await self._send_to_client(
                client_id, self._revoke_ticket(app_id, event)
            )
        else:  # pragma: no cover - TICKET_VERBS and the pair are one set
            await self._send_error(client_id, f"Unhandled ticket verb: {verb}")

    def _ticket_denied(self, verb: str, message: str):
        """Build the ``denied`` result for a connection that may not bind."""
        request_id = _peek_request_id(message)
        detail = (
            "ticket.bind / ticket.revoke require a connection authenticated "
            "by an application credential"
        )
        if not self._app_credentials:
            detail += "; this daemon has none configured (--ws-app-credentials)"
        if verb == EventType.TICKET_REVOKE_REQUEST.value:
            return TicketRevokeResultEvent(
                request_id=request_id, status="denied", detail=detail
            )
        return TicketBindResultEvent(
            request_id=request_id, status="denied", detail=detail
        )

    def _bind_ticket(
        self, app_id: str, event: "TicketBindRequest"
    ) -> "TicketBindResultEvent":
        """Mint a ticket for ``event.user`` on behalf of ``app_id``.

        ``app_id`` is the caller's own authenticated application, taken from
        the connection rather than from the request — the request has no such
        field, precisely so that qualification cannot be forged or forgotten.

        Every failure mints nothing and is reported by a ``status`` a caller
        can branch on: ``invalid`` for a request the registry refused,
        ``capacity`` for a daemon already holding its ceiling of live
        tickets.
        """
        try:
            ticket, expires_at = self._ticket_registry.bind(
                app_id=app_id,
                user=event.user,
                ttl_seconds=event.ttl_seconds,
                single_use=event.single_use,
            )
        except ValueError as exc:
            return TicketBindResultEvent(
                request_id=event.request_id, status="invalid", detail=str(exc)
            )
        except TicketCapacityError as exc:
            logger.warning("ticket.bind refused for app %r: %s", app_id, exc)
            return TicketBindResultEvent(
                request_id=event.request_id, status="capacity", detail=str(exc)
            )
        identity = BoundIdentity(app_id=app_id, user=event.user)
        logger.info(
            "ticket.bind: app=%s user=%s ttl=%ss single_use=%s",
            app_id, event.user, event.ttl_seconds, event.single_use,
        )
        return TicketBindResultEvent(
            request_id=event.request_id,
            status="bound",
            ticket=ticket,
            qualified=identity.qualified,
            app_id=app_id,
            expires_at=expires_at,
        )

    def _revoke_ticket(
        self, app_id: str, event: "TicketRevokeRequest"
    ) -> "TicketRevokeResultEvent":
        """Revoke one ticket, or every outstanding ticket of one user.

        Scoped to ``app_id`` on both routes, so one application can neither
        revoke nor log out another's users. A ticket belonging to a different
        application answers ``not_found`` — the same answer an unknown ticket
        gives — so this verb reveals nothing about tickets the caller did not
        mint.

        Exactly one of ``ticket`` / ``user`` is required: a request naming
        both has two incompatible readings, and picking one silently is how
        the wrong thing gets revoked.
        """
        if bool(event.ticket) == bool(event.user):
            return TicketRevokeResultEvent(
                request_id=event.request_id,
                status="invalid",
                detail="supply exactly one of 'ticket' or 'user'",
            )
        if event.ticket:
            revoked = 1 if self._ticket_registry.revoke(
                event.ticket, app_id=app_id
            ) else 0
        else:
            revoked = self._ticket_registry.revoke_user(app_id, event.user)
        logger.info(
            "ticket.revoke: app=%s %s revoked=%d",
            app_id,
            f"user={event.user}" if event.user else "ticket=<redacted>",
            revoked,
        )
        return TicketRevokeResultEvent(
            request_id=event.request_id,
            status="revoked" if revoked else "not_found",
            revoked=revoked,
        )

    # =========================================================================
    # app:// secret resolution (#1226)
    # =========================================================================

    @property
    def app_secret_resolver(self) -> AppSecretResolver:
        """The in-tree ``app://`` resolver this server offers (#1226).

        Handed to the daemon's ``SessionManager`` by the wiring in
        ``server/__main__.py`` so every session it constructs resolves
        ``app://`` references against the application that owns the workspace,
        over this server's bind channel.
        """
        return self._app_secret_resolver

    def _owner_for_workspace_path(self, workspace_path: str) -> Optional[str]:
        """The qualified owner ``app:user`` of the workspace at ``workspace_path``.

        Reads ``WorkspaceManager``, so ``None`` on a server with no workspace
        manager (which cannot own workspaces anyway) and for a path that is not
        a known, owned workspace.
        """
        manager = self._workspace_manager
        if manager is None:
            return None
        return manager.owner_for_path(workspace_path)

    async def _handle_secret_bind_message(
        self, client_id: str, verb: str, message: str
    ) -> None:
        """Dispatch a ``secret.*`` bind-channel frame from an app credential.

        Both verbs require an app-credential connection, refused BEFORE parsing
        so the refusal cannot depend on the body — the same rule
        :meth:`_handle_ticket_message` applies.  ``secret.resolve.result`` is
        the application answering the daemon's ``secret.resolve`` and carries no
        reply; ``secret.reload`` is a request the daemon answers.
        """
        app_id = self.get_client_app(client_id)
        if self._client_auth_kind(client_id) != AUTH_KIND_APP or not app_id:
            # A result frame from a non-app connection is simply dropped (no
            # reply is expected for it); a reload request is denied by name.
            if verb == EventType.SECRET_RELOAD_REQUEST.value:
                await self._send_to_client(
                    client_id, self._secret_reload_denied(message)
                )
            else:
                logger.debug(
                    "dropping %s from non-app connection %s", verb, client_id,
                )
            return
        try:
            event = deserialize_event(message)
        except (json.JSONDecodeError, ValueError) as exc:
            await self._send_error(client_id, f"Invalid {verb} request: {exc}")
            return
        if isinstance(event, SecretResolveResultEvent):
            self._deliver_secret_result(event)
            return
        if isinstance(event, SecretReloadRequest):
            await self._send_to_client(
                client_id, self._reload_owner_sessions(app_id, event)
            )
            return
        await self._send_error(client_id, f"Unhandled secret verb: {verb}")

    def _secret_reload_denied(self, message: str) -> "SecretReloadResultEvent":
        """Build the ``denied`` answer for a ``secret.reload`` from a non-app connection."""
        detail = (
            "secret.reload requires a connection authenticated by an "
            "application credential"
        )
        if not self._app_credentials:
            detail += "; this daemon has none configured (--ws-app-credentials)"
        return SecretReloadResultEvent(
            request_id=_peek_request_id(message), status="denied", detail=detail
        )

    def _deliver_secret_result(self, event: "SecretResolveResultEvent") -> None:
        """Resolve the pending future for a ``secret.resolve.result`` frame.

        A result whose ``request_id`` matches no outstanding request (a late
        answer after the daemon's deadline, or a duplicate) is dropped: the
        caller has already given up and dropped the reference.
        """
        with self._pending_secret_lock:
            fut = self._pending_secret_resolves.pop(event.request_id, None)
        if fut is None or fut.done():
            return
        fut.set_result(event)

    def _reload_owner_sessions(
        self, app_id: str, event: "SecretReloadRequest"
    ) -> "SecretReloadResultEvent":
        """Re-resolve the env of the calling application's sessions for one user (§6.4).

        ``app_id`` is the connection's own authenticated application, so the
        owner is qualified here (``f"{app_id}:{user}"``) and the reload can only
        touch that application's users — one application can never reload
        another's ``alice``.
        """
        if not event.user:
            return SecretReloadResultEvent(
                request_id=event.request_id, status="denied",
                detail="secret.reload requires a non-empty 'user'",
            )
        qualified = f"{app_id}:{event.user}"
        reloaded = 0
        sm = self._command_router._session_manager if self._command_router else None
        if sm is not None and hasattr(sm, "reload_owner_sessions"):
            reloaded = sm.reload_owner_sessions(qualified)
        logger.info(
            "secret.reload: app=%s user=%s reloaded=%d",
            app_id, event.user, reloaded,
        )
        return SecretReloadResultEvent(
            request_id=event.request_id, status="ok", reloaded=reloaded,
        )

    def _resolve_app_secret_over_bind_channel(
        self, app_id: str, user: str, workspace: str, name: str, timeout: float,
    ) -> AppSecretAnswer:
        """Send ``secret.resolve`` to application ``app_id`` and await its answer.

        The :class:`AppSecretResolver`'s transport (#1226).  Called from an
        executor thread (``_resolve_session_env`` runs off the event loop), so
        it schedules the send on the loop and blocks on a future the
        ``secret.resolve.result`` frame resolves.  Never raises for an absent
        application, a closed loop or a timeout — each is an
        :class:`AppSecretAnswer` the caller drops on.
        """
        target = self._app_connection_for(app_id)
        if target is None:
            return AppSecretAnswer(
                status="unreachable",
                detail=f"application {app_id!r} has no bind connection",
            )
        loop = self._event_loop
        if loop is None:
            return AppSecretAnswer(
                status="unreachable", detail="WS event loop not running",
            )
        request_id = uuid.uuid4().hex
        fut: "concurrent.futures.Future[SecretResolveResultEvent]" = (
            concurrent.futures.Future()
        )
        with self._pending_secret_lock:
            self._pending_secret_resolves[request_id] = fut
        request = SecretResolveRequest(
            request_id=request_id, user=user, workspace=workspace, name=name,
        )
        try:
            asyncio.run_coroutine_threadsafe(
                self._send_to_client(target, request), loop,
            )
            result = fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            with self._pending_secret_lock:
                self._pending_secret_resolves.pop(request_id, None)
            return AppSecretAnswer(
                status="unreachable",
                detail=f"no secret.resolve.result within {timeout:g}s",
            )
        except Exception as exc:  # noqa: BLE001 -- any transport error is a drop
            with self._pending_secret_lock:
                self._pending_secret_resolves.pop(request_id, None)
            return AppSecretAnswer(status="unreachable", detail=str(exc))
        return _answer_from_result(result)

    def _app_connection_for(self, app_id: str) -> Optional[str]:
        """A client_id of a live app-credential connection for ``app_id``, or ``None``.

        Several connections may authenticate as one application (the BFF's bind
        channel plus its own reconnects); any one of them can answer, so the
        first match is returned.
        """
        for cid, conn in self._clients.items():
            if conn.auth_kind == AUTH_KIND_APP and conn.app_id == app_id:
                return cid
        return None

    async def _handle_message(self, client_id: str, message: str) -> None:
        """Handle an incoming message from a client.

        Reached through :meth:`_dispatch_client_message`, which has already
        refused the frame if it is a ``ticket.*`` verb or if the connection is
        an app credential (#1074).

        Args:
            client_id: The client's ID.
            message: The JSON message.
        """
        try:
            event = deserialize_event(message)
        except json.JSONDecodeError as e:
            await self._send_error(client_id, f"Invalid JSON: {e}")
            return
        except ValueError:
            # Unknown event type — check extension message handlers
            # before discarding.
            try:
                raw = json.loads(message)
            except json.JSONDecodeError:
                return
            msg_type = raw.get("type", "")
            handler = self._message_handlers.get(msg_type)
            if handler:
                async with self._lock:
                    client = self._clients.get(client_id)
                if client:
                    await handler(client.websocket, raw, client.user_id, client_id)
            else:
                await self._send_error(client_id, f"Unknown message type: {msg_type}")
            return

        # --- Workspace management (transport-level, all modes) ---
        # Workspace negotiation is a transport concern, not a command concern.
        # These events are handled by the WS server regardless of whether a
        # CommandRouter is present (daemon mode) or not (standalone mode).
        is_workspace_request = isinstance(event, (
            WorkspaceListRequest,
            WorkspaceCreateRequest,
            WorkspaceSelectRequest,
            WorkspaceDeleteRequest,
            ConfigUpdateRequest,
        ))
        if is_workspace_request:
            await self._handle_workspace_event(client_id, event)
            return

        # --- External events (from <jaato-task> web component) ---
        from jaato_sdk.events import ExternalEventRequest
        if isinstance(event, ExternalEventRequest):
            await self._handle_external_event(client_id, event)
            return

        # --- File staging into a workspace (multi-frame: TEXT + N binary) ---
        # The handler reads the raw binary payload frames inline before
        # the per-connection receive loop can dispatch the next message,
        # so frame ordering is preserved without coordination state.
        # A download (1.20) is the same pair of frames in the other direction.
        if await self._dispatch_workspace_file_transfer(client_id, event):
            return

        # --- Daemon-mode delegation ---
        # When running as part of JaatoDaemon, route session/command events
        # through the CommandRouter for unified dispatch across transports.
        if self._command_router:
            # Re-parse the raw JSON so side-channel fields (e.g. ``staged_files``
            # on session.new) are accessible to the daemon handler.  These
            # fields are NOT part of the typed ``CommandRequest`` dataclass —
            # they piggyback on the WS JSON envelope so the web component can
            # ship inline file payloads without a separate HTTP upload.
            # Re-parsing is cheap (``deserialize_event`` already succeeded, so
            # the JSON is valid).
            try:
                raw_data = json.loads(message)
            except json.JSONDecodeError:
                raw_data = None
            await self._handle_message_daemon(client_id, event, raw_data=raw_data)
            return

        # --- Standalone mode ---
        if not self._jaato_server:
            await self._send_error(client_id, "No workspace selected")
            return

        # Set logging context for session-specific log routing
        if self._jaato_server and self._workspace_manager:
            selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
            workspace_path = selected.path if selected else None
            session_env = self._jaato_server.get_all_session_env()
            # Use workspace name as session_id for WebSocket mode
            session_id = selected.name if selected else "websocket"
            set_logging_context(
                session_id=session_id,
                client_id=client_id,
                workspace_path=workspace_path,
                session_env=session_env,
            )

        # Route by event type
        if isinstance(event, SendMessageRequest):
            # Capture context for thread (ContextVars don't propagate to threads)
            if self._jaato_server and self._workspace_manager:
                selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
                ctx_workspace = selected.path if selected else None
                ctx_session_env = self._jaato_server.get_all_session_env()
                ctx_session_id = selected.name if selected else "websocket"
                ctx_client_id = client_id

                def run_with_context():
                    set_logging_context(
                        session_id=ctx_session_id,
                        client_id=ctx_client_id,
                        workspace_path=ctx_workspace,
                        session_env=ctx_session_env,
                    )
                    try:
                        self._jaato_server.send_message(
                            event.text,
                            event.attachments if event.attachments else None
                        )
                    finally:
                        clear_logging_context()

                await asyncio.get_event_loop().run_in_executor(None, run_with_context)
            else:
                # Fallback without context
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._jaato_server.send_message(
                        event.text,
                        event.attachments if event.attachments else None
                    )
                )

        elif isinstance(event, PermissionResponseRequest):
            self._jaato_server.respond_to_permission(
                event.request_id,
                event.response,
                edited_arguments=event.edited_arguments,
                # Attribute the decision to the authenticated WS user (#859).
                user_id=self.get_client_user(client_id),
            )

        elif isinstance(event, ClarificationResponseRequest):
            self._jaato_server.respond_to_clarification(
                event.request_id,
                event.response
            )

        elif isinstance(event, ClarificationBatchResponseEvent):
            self._jaato_server.respond_to_clarification_batch(
                event.request_id,
                event.answers,
                cancelled=event.cancelled,
                answer_attachments=event.answer_attachments,
            )

        elif isinstance(event, ReferenceSelectionResponseRequest):
            self._jaato_server.respond_to_reference_selection(
                event.request_id,
                event.response
            )

        elif isinstance(event, StopRequest):
            self._jaato_server.stop()

        elif isinstance(event, CommandRequest):
            result = self._jaato_server.execute_command(
                event.command,
                event.args
            )
            # HelpLines results are already emitted via HelpTextEvent, skip
            if not (isinstance(result, dict) and "_pager" in result):
                # Send result as system message
                await self._send_to_client(
                    client_id,
                    SystemMessageEvent(
                        message=json.dumps(result),
                        style="info",
                    )
                )

        else:
            await self._send_error(client_id, f"Unknown request type: {event.type}")

    async def _handle_external_event(self, client_id: str, event) -> None:
        """Handle an ``ExternalEventRequest`` from a WebSocket client.

        Publishes the external event on the session's ``EventBus`` so that
        agents subscribed via ``subscribeToEvents(event_types=['external_event'])``
        are woken via ``inject_prompt()``, and so that it sinks onward to the
        daemon-wide reactor bus.

        This method's job is the half only the transport can do: say WHICH
        session the caller is driving, and which ``JaatoServer`` owns it.  The
        translation into a bus event is
        :func:`server.external_event.publish_external_event`, shared with the
        IPC path (``SessionManager._handle_external_event_request``) since
        issue #1167 -- two copies of "what an external event looks like on the
        bus" is the shape this tree treats as a defect.

        Both failure answers reach the client as a WS error frame and neither
        raises: a session with no bus, or a client driving no session at all,
        must not cost the connection.

        Args:
            client_id: The WS client that sent the message.
            event: Deserialized ``ExternalEventRequest``.
        """
        from jaato_server.server.external_event import publish_external_event

        # Resolve the session attached to this client
        session_id = ""
        if self._event_sink_adapter:
            session_id = self._event_sink_adapter._client_sessions.get(client_id, "")

        if not session_id:
            await self._send_error(client_id, "No session attached — cannot deliver external event")
            return

        # Find the session's owning JaatoServer.
        # In daemon mode: SessionManager → Session → JaatoServer.
        # Phase 3 §7c step 6.6.3.6: the bus is read from the daemon-side
        # ``server._runtime`` (daemon-tier per §4.2; mirrors the migration in
        # core.py's event_bus property at §7c step 6.2), which is what
        # ``publish_external_event`` does -- so this only has to pick the
        # right server.
        owner = None
        if self._command_router:
            sm = self._command_router._session_manager
            session_obj = sm.get_session(session_id) if hasattr(sm, 'get_session') else None
            owner = session_obj.server if session_obj else None
        else:
            owner = self._jaato_server

        delivery = publish_external_event(
            owner,
            name=event.name,
            data=event.data,
            timestamp=event.timestamp,
            source="websocket",
        )
        if not delivery.ok:
            await self._send_error(client_id, delivery.error)
            return

        logger.debug(
            "External event '%s' published to session %s, notified %d subscriber(s)",
            event.name, session_id, delivery.notified,
        )

    async def _handle_workspace_event(self, client_id: str, event: Event) -> None:
        """Handle workspace management events (transport-level concern).

        Routes to the existing workspace handlers and, when a workspace is
        selected, bridges the resolved path to the ``WSEventSinkAdapter``
        so the ``CommandRouter`` can query it via ``get_client_workspace()``.
        """
        if isinstance(event, WorkspaceListRequest):
            await self._handle_workspace_list(client_id)
        elif isinstance(event, WorkspaceCreateRequest):
            await self._handle_workspace_create(client_id, event.name)
        elif isinstance(event, WorkspaceDeleteRequest):
            await self._handle_workspace_delete(client_id, event.name)
        elif isinstance(event, WorkspaceSelectRequest):
            await self._handle_workspace_select(client_id, event.name)
            # Bridge selected workspace path to the event sink adapter
            # so CommandRouter can resolve it via get_client_workspace()
            if self._event_sink_adapter and self._workspace_manager:
                selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
                if selected and selected.path:
                    self._event_sink_adapter.set_client_workspace(client_id, selected.path)
        elif isinstance(event, ConfigUpdateRequest):
            await self._handle_config_update(
                client_id, event.provider, event.model, event.api_key,
            )

    async def _handle_stage_files_request(self, client_id: str, event) -> None:
        """Handle ``StageFilesRequest`` — multi-frame staging into a workspace.

        Wire protocol (per-connection, frames preserve order):

        1. Caller has just sent this :class:`StageFilesRequest` as one
           TEXT frame (``deserialize_event`` already produced the object).
        2. ``len(event.files)`` raw BINARY frames follow, one per file,
           in the same order as ``event.files``.
        3. We respond with one TEXT frame carrying
           :class:`StageFilesEvent`.

        Frame ordering is guaranteed because the caller's
        ``async for message in websocket`` loop awaits this coroutine to
        completion before pulling the next message — so our explicit
        ``websocket.recv()`` calls below pick up exactly the binary
        payloads the client sent next.

        Validation runs *before* any binary frames are read so a
        rejection case (no workspace, total cap exceeded, bad name) is
        a single round-trip rather than a half-consumed stream.  When we
        do reject mid-stream (per-frame size mismatch) we still drain
        the remaining binary frames so the connection state stays in
        sync — otherwise the next text-frame would arrive while the
        server is still expecting binary.
        """
        from jaato_sdk.events import StageFilesEvent

        workspace_path = self._resolve_staging_workspace(client_id, event.workspace_id)
        if workspace_path is None:
            # No workspace context — refuse without reading any binary
            # frames.  Client must catch up by sending a workspace.select
            # or session.new before retrying.
            await self._send_to_client(client_id, StageFilesEvent(
                workspace_id=event.workspace_id,
                staged=[],
                failed=[{
                    "name": spec.name,
                    "category": "workspace_not_found",
                    "error": (
                        f"No workspace selected for client {client_id} "
                        f"(workspace_id={event.workspace_id!r})"
                    ),
                } for spec in event.files],
            ))
            # Still drain the binary frames the client is about to send;
            # otherwise the per-connection receive loop reads them as
            # the next ``message`` and dispatch breaks.
            await self._drain_binary_frames(client_id, len(event.files))
            return

        # Up-front size validation — both per-file and total.  Refusing
        # before reading binary frames keeps a misbehaving client from
        # tying up the daemon with a 500 MB upload that we'd reject
        # anyway.
        cap_per_file = DEFAULT_STAGE_PER_FILE_LIMIT
        cap_total = DEFAULT_STAGE_TOTAL_LIMIT
        declared_total = sum(max(spec.size, 0) for spec in event.files)
        if declared_total > cap_total:
            await self._send_to_client(client_id, StageFilesEvent(
                workspace_id=event.workspace_id,
                staged=[],
                failed=[{
                    "name": spec.name,
                    "category": "size_limit_total",
                    "error": (
                        f"declared total {declared_total} bytes exceeds "
                        f"cap {cap_total}"
                    ),
                } for spec in event.files],
            ))
            await self._drain_binary_frames(client_id, len(event.files))
            return

        async with self._lock:
            client = self._clients.get(client_id)
        if client is None:
            return  # Disconnected mid-flight; nothing to do.

        ws = client.websocket
        staged: list = []
        failed: list = []

        for spec in event.files:
            # Per-file declared-size and name validation BEFORE pulling
            # the binary.  When we skip pulling we still need to drain
            # one frame to keep the stream aligned.
            if spec.size > cap_per_file:
                failed.append({
                    "name": spec.name,
                    "category": "size_limit_per_file",
                    "error": (
                        f"declared size {spec.size} bytes exceeds "
                        f"per-file cap {cap_per_file}"
                    ),
                })
                await self._drain_binary_frames(client_id, 1)
                continue

            clean = _safe_staged_filename(spec.name)
            if clean is None:
                failed.append({
                    "name": spec.name,
                    "category": "unsafe_path",
                    "error": (
                        "name must be a non-empty workspace-relative "
                        "path with no '..' components"
                    ),
                })
                await self._drain_binary_frames(client_id, 1)
                continue

            try:
                payload = await ws.recv()
            except Exception as exc:
                # Connection died mid-stream; return whatever we have so
                # far.  Subsequent files can't be drained.
                failed.append({
                    "name": spec.name,
                    "category": "io_error",
                    "error": f"connection closed mid-staging: {exc}",
                })
                break

            if isinstance(payload, str):
                # Client sent a TEXT frame where we expected BINARY.
                # Treat as fatal — the protocol is violated and any
                # further drain attempts would be reading misordered
                # data.  Mark this file failed and stop reading.
                failed.append({
                    "name": spec.name,
                    "category": "size_mismatch",
                    "error": "expected BINARY frame, got TEXT (protocol violation)",
                })
                break

            if len(payload) != spec.size:
                failed.append({
                    "name": spec.name,
                    "category": "size_mismatch",
                    "error": (
                        f"declared {spec.size} bytes, frame carried "
                        f"{len(payload)} bytes"
                    ),
                })
                continue

            try:
                _write_staged_payload(Path(workspace_path), clean, payload)
            except OSError as exc:
                failed.append({
                    "name": spec.name,
                    "category": "io_error",
                    "error": str(exc),
                })
                continue

            staged.append(spec.name)

        await self._send_to_client(client_id, StageFilesEvent(
            workspace_id=event.workspace_id,
            staged=staged,
            failed=failed,
        ))
        logger.info(
            "Stage files for %s: workspace=%s staged=%d failed=%d",
            client_id, workspace_path, len(staged), len(failed),
        )

    def _resolve_staging_workspace(
        self, client_id: str, workspace_id: str,
    ) -> Optional[str]:
        """Resolve the absolute workspace path for a staging request.

        Empty ``workspace_id`` → the client's currently-attached
        workspace.  Non-empty → must match that workspace's basename;
        cross-client staging is intentionally not allowed without a
        per-user auth model (see project_backlog_ws_per_user_auth).

        Returns the absolute path, or ``None`` if the client has no
        workspace or the requested id doesn't match.

        **It asks the question the router already answers.**

        ``CommandRouter.resolve_caller_workspace`` is this daemon's one
        definition of *which workspace a verb acts in for this client* --
        the attached session's, else the transport's session, else what the
        client declared -- and its docstring gives the reason staging needs
        exactly that order: *a session's path outranks a declared one
        because the session's tree is what ``WorkspaceMonitor`` watches and
        what the panel shows.*

        Staging used to answer it a second way, and wrongly: it read
        ``_client_provisioned`` FIRST.  That map is stamped when a
        ``session.new`` arrives from a client with no workspace (the daemon
        auto-provisions one) and is cleared only on DISCONNECT -- never when
        the client later selects a workspace.  So on one connection:

        ===============================  ================  ==============
        order                            session runs in   file lands in
        ===============================  ================  ==============
        ``select, new``                  named             named
        ``new, select``                  provisioned       provisioned
        ``new, select, new``             **named**         **provisioned**
        ===============================  ================  ==============

        Only the third diverges, which is why it reads as intermittent, and
        it is an ordinary flow: create a session, end it (in workspace mode
        that returns to the workspace list), open a named workspace, create
        a session, attach a file.  Nothing reports it -- the write succeeds,
        so the daemon answers ``StageFilesEvent(staged=[name], failed=[])``,
        the client says the file is in the workspace, the model cannot find
        it, and the files panel never shows it.

        Reordering the two stores here would have fixed that row and broken
        the second: after ``new, select`` the SESSION is still in the
        provisioned workspace, and a file attached to it belongs there, not
        in whatever the client selected afterwards.  Measured both ways --
        which is why this delegates rather than growing a third ordering.

        The provisioned map stays as the LAST fallback, for a caller that
        reached :meth:`provision_workspace` directly: the one path that
        stamps it without telling the adapter or the router.
        """
        adapter = self._event_sink_adapter
        declared = adapter.get_client_workspace(client_id) if adapter else None
        if not declared and self._workspace_manager is not None:
            selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
            declared = selected.path if selected else None

        router = getattr(self, "_command_router", None)
        if router is not None and hasattr(router, "resolve_caller_workspace"):
            session_id = adapter._client_sessions.get(client_id) if adapter else None
            current_path, _sources = router.resolve_caller_workspace(
                client_id, declared, session_id,
            )
        else:
            current_path = declared

        if not current_path:
            provisioned = self._client_provisioned.get(client_id)
            current_path = provisioned.path if provisioned is not None else None

        if not current_path:
            return None

        if workspace_id and os.path.basename(current_path) != workspace_id:
            return None

        return current_path

    async def _dispatch_workspace_file_transfer(self, client_id: str, event) -> bool:
        """Route the two binary-framed file verbs; ``True`` when one handled it.

        Upload (``StageFilesRequest``) and download
        (``WorkspaceFileFetchRequest``) both move raw binary frames beside a
        TEXT event, so both must be handled HERE, on the receive loop, before
        anything else reads a frame -- and never reach the command router,
        which has no binary channel.
        """
        from jaato_sdk.events import StageFilesRequest, WorkspaceFileFetchRequest
        if isinstance(event, StageFilesRequest):
            await self._handle_stage_files_request(client_id, event)
            return True
        if isinstance(event, WorkspaceFileFetchRequest):
            await self._handle_file_fetch_request(client_id, event)
            return True
        return False

    async def _handle_file_fetch_request(self, client_id: str, event) -> None:
        """Handle ``WorkspaceFileFetchRequest`` -- download one workspace file.

        The reverse of :meth:`_handle_stage_files_request`.  The workspace is
        the one staging would write into (:meth:`_resolve_staging_workspace`,
        which delegates to the router's one definition of *which workspace a
        verb acts in for this client*), and whether the path may leave it is
        :func:`jaato_server.server.workspace_download.resolve_download`'s decision -- this
        method only moves bytes.

        On success, and unless ``metadata_only``, the header is followed by
        ONE binary frame; both are written by
        :meth:`_send_to_client_with_binary` under the send lock, so no other
        event can land between them and a client may take "the next binary
        frame" as this file.  The header's ``size`` is the length READ, not
        the earlier ``stat``, so the two cannot disagree.  The read runs off
        the event loop: a 50 MB file must not stall every other connection.
        """
        from jaato_sdk.events import WorkspaceFileContentEvent
        from .workspace_download import read_download, resolve_download

        def _answer(**fields: Any) -> WorkspaceFileContentEvent:
            return WorkspaceFileContentEvent(
                request_id=event.request_id,
                metadata_only=event.metadata_only,
                **fields,
            )

        workspace_path = self._resolve_staging_workspace(client_id, "")
        if workspace_path is None:
            await self._send_to_client(client_id, _answer(
                ok=False, path=event.path, category="workspace_not_found",
                error=f"No workspace selected for client {client_id}",
            ))
            return
        target = resolve_download(workspace_path, event.path)
        if not target.ok:
            logger.info(
                "file fetch refused for %s: path=%r category=%s",
                client_id, event.path, target.category,
            )
            await self._send_to_client(client_id, _answer(
                ok=False, path=event.path, category=target.category,
                error=target.error,
            ))
            return
        meta = dict(path=target.relpath, name=target.name, mime_type=target.mime_type)
        if event.metadata_only:
            await self._send_to_client(client_id, _answer(ok=True, size=target.size, **meta))
            return
        try:
            data = await asyncio.to_thread(read_download, target)
        except OSError as exc:
            await self._send_to_client(client_id, _answer(
                ok=False, category="io_error", error=str(exc), **meta,
            ))
            return
        await self._send_to_client_with_binary(
            client_id, _answer(ok=True, size=len(data), **meta), data,
        )

    async def _send_to_client_with_binary(
        self, client_id: str, event: Event, data: bytes,
    ) -> None:
        """Send ``event`` then ``data`` as one binary frame, back to back.

        Both writes happen under ``self._lock`` -- the lock every other send
        to a client takes (:meth:`_send_to_client`, :meth:`_broadcast`) -- so
        the binary frame is always the very next frame after its header.
        That adjacency is the whole protocol: the header carries no id the
        binary frame could be matched by.
        """
        async with self._lock:
            client = self._clients.get(client_id)
            if not client:
                return
            try:
                await client.websocket.send(serialize_event(event))
                await client.websocket.send(data)
            except Exception as e:
                logger.error(f"Send error to {client_id}: {e}")

    async def _drain_binary_frames(self, client_id: str, count: int) -> None:
        """Consume ``count`` binary frames without processing them.

        Used after a fatal validation failure so the per-connection
        receive loop doesn't dispatch the leftover binary payloads as
        if they were JSON messages.  Best-effort: a closed connection
        or a TEXT frame appearing where we expected BINARY both stop
        the drain loop.
        """
        if count <= 0:
            return
        async with self._lock:
            client = self._clients.get(client_id)
        if client is None:
            return
        ws = client.websocket
        for _ in range(count):
            try:
                frame = await ws.recv()
            except Exception:
                return
            if isinstance(frame, str):
                # Out of sync with the client; further drains would be
                # consuming the wrong frames.  Leave whatever's left for
                # the main loop to surface as "Unknown message type".
                logger.warning(
                    "stage_files: drain hit TEXT frame for %s; "
                    "stream alignment lost", client_id,
                )
                return

    async def _handle_message_daemon(
        self,
        client_id: str,
        event: Event,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle a message when running in daemon mode.

        Delegates all events to the ``CommandRouter`` via
        ``run_in_executor`` so the async event loop is not blocked.

        The router uses the ``EventSink`` to send responses back to this
        client (via ``WSEventSinkAdapter``).

        Exceptions from the router are caught and sent back to the client
        as ``ErrorEvent`` so that a single bad request does not tear down
        the entire WebSocket connection.

        Args:
            client_id: The client's ID.
            event: The deserialized event.
            raw_data: The raw parsed JSON envelope for the message, when
                available.  Used to access side-channel fields on
                ``session.new`` such as ``staged_files`` — inline file
                payloads that aren't part of the typed ``CommandRequest``
                dataclass but travel alongside it in the WS envelope.
        """
        from jaato_sdk.events import (
            ClientConfigRequest, CommandListRequest, CommandListEvent,
            ToolsRegisterClientRequest, ToolExecuteResultEvent,
        )

        # Handle CommandListRequest directly (same as IPC path)
        if isinstance(event, CommandListRequest):
            if self._command_router:
                commands = self._command_router.get_command_list()
                await self._send_to_client(
                    client_id, CommandListEvent(commands=commands),
                )
            return

        # Handle client-side tool registration
        if isinstance(event, ToolsRegisterClientRequest):
            # Buffer tools if no session yet (registered after session.new)
            if not self._event_sink_adapter or not self._event_sink_adapter._client_sessions.get(client_id):
                if not hasattr(self, '_pending_client_tools'):
                    self._pending_client_tools = {}
                self._pending_client_tools[client_id] = event.tools
                self._pending_client_categories = getattr(self, '_pending_client_categories', {})
                self._pending_client_categories[client_id] = event.categories
                # Buffer the SCHEMAS daemon-side too so the session.new flow seeds
                # JaatoServer.client_tool_schemas BEFORE the spawn reads it for
                # envelope.client_tools — the proxy-executor apply below (after
                # handle_request) races the spawn (PR #349 e2e bug).  Transport
                # keeps its own buffer for the post-session.new executor register.
                self._command_router._session_manager.buffer_client_tools(
                    client_id, event.tools)
                logger.info("Buffered %d client tools for %s (session pending)", len(event.tools), client_id)
            else:
                self._register_client_tools(client_id, event.tools, event.categories)
            return

        # Handle client-side tool execution result
        if isinstance(event, ToolExecuteResultEvent):
            self._handle_tool_execute_result(client_id, event)
            return

        # Resolve session_id from the adapter's tracking
        session_id = ""
        if self._event_sink_adapter:
            session_id = self._event_sink_adapter._client_sessions.get(client_id, "")

        # Auto-provision a workspace for WS clients that don't have one
        # when they request a new session.  This mirrors the standalone WS
        # flow but plugs into the daemon command-router path.
        from jaato_sdk.events import CommandRequest
        if (isinstance(event, CommandRequest)
                and event.command.lower() == "session.new"
                and self._provisioner
                and self._event_sink_adapter
                and not self._event_sink_adapter.get_client_workspace(client_id)):
            import uuid as _uuid
            provisioned = await self.provision_workspace(
                session_id=f"ws_{_uuid.uuid4().hex[:8]}",
                client_id=client_id,
            )
            if provisioned:
                # Materialise inline ``staged_files`` into the workspace.
                # The <jaato-task> web component ships files in the WS
                # envelope alongside session.new to avoid the HTTP
                # ``/api/task/artifacts`` roundtrip (which depends on the
                # dashboard's SSO cookie and breaks cross-origin embeds).
                # Both ``staged_files`` and the ``--artifacts`` HTTP-upload
                # path are supported; when both are present they apply in
                # order (staged_files first, artifacts overlays).
                staged_files = (
                    getattr(event, 'staged_files', None)
                    or (raw_data.get('staged_files') if raw_data else None)
                )
                if staged_files:
                    written = _materialize_staged_files(
                        Path(provisioned.path), staged_files,
                    )
                    logger.info(
                        "Staged %d file(s) inline into workspace %s",
                        written, provisioned.path,
                    )

                # Copy staged artifacts into the workspace if present.
                # The dashboard passes --artifacts <staging_id> in the
                # session.new args after uploading via /api/task/artifacts.
                args = event.args if hasattr(event, 'args') else []
                staging_id = None
                for i, arg in enumerate(args):
                    if arg == "--artifacts" and i + 1 < len(args):
                        staging_id = args[i + 1]
                        break
                if staging_id:
                    import shutil
                    import tempfile
                    from pathlib import Path as _Path
                    staging_dir = _Path(tempfile.gettempdir()) / f"jaato-artifacts-{staging_id}"
                    if staging_dir.is_dir():
                        ws_path = _Path(provisioned.path)
                        for item in staging_dir.iterdir():
                            dest = ws_path / item.name
                            if item.is_dir():
                                shutil.copytree(item, dest, dirs_exist_ok=True)
                            else:
                                shutil.copy2(item, dest)
                        shutil.rmtree(staging_dir, ignore_errors=True)
                        logger.info(
                            "Copied staged artifacts %s into workspace %s",
                            staging_id, provisioned.path,
                        )

                self._event_sink_adapter.set_client_workspace(
                    client_id, provisioned.path,
                )
                self._client_provisioned[client_id] = provisioned
                logger.info(
                    "Auto-provisioned workspace for WS client %s: %s",
                    client_id, provisioned.path,
                )

        try:
            # ClientConfigRequest must be processed synchronously (same as IPC)
            if isinstance(event, ClientConfigRequest):
                self._command_router.handle_request(client_id, session_id, event)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    self._command_router.handle_request,
                    client_id,
                    session_id,
                    event,
                )
            # After session.new or session.attach completes, register any
            # buffered client tools that were sent before the session existed.
            if (isinstance(event, CommandRequest)
                    and event.command.lower() in ("session.new", "session.attach")):
                if (hasattr(self, '_pending_client_tools')
                        and client_id in self._pending_client_tools):
                    pending = self._pending_client_tools.pop(client_id)
                    pending_cats = getattr(self, '_pending_client_categories', {}).pop(client_id, None)
                    # Wires tools + drives any deferred wake AFTER the runner
                    # push (see _register_client_tools._push).
                    self._register_client_tools(client_id, pending, pending_cats)
                else:
                    # No buffered host tools → drive any deferred wake for the
                    # now-attached session immediately.
                    _sid = (self._event_sink_adapter._client_sessions.get(client_id)
                            if self._event_sink_adapter else None)
                    if _sid:
                        try:
                            self._command_router._session_manager.drive_pending_wake(_sid)
                        except Exception:
                            logger.exception(
                                "deferred-wake drive on no-tools attach failed "
                                "for %s", _sid)

        except Exception as exc:
            logger.error(
                "Command routing failed for client %s: %s", client_id, exc,
                exc_info=True,
            )
            await self._send_error(client_id, f"Internal error: {exc}")

    async def _send_to_client(self, client_id: str, event: Event) -> None:
        """Send an event to a specific client."""
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                try:
                    await client.websocket.send(serialize_event(event))
                except Exception as e:
                    logger.error(f"Send error to {client_id}: {e}")

    async def _send_error(self, client_id: str, error: str) -> None:
        """Send an error event to a client."""
        await self._send_to_client(
            client_id,
            ErrorEvent(error=error, error_type="RequestError")
        )

    # =========================================================================
    # Client-side tool execution
    # =========================================================================

    def _register_client_tools(
        self,
        client_id: str,
        tools: list,
        categories: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register client-provided tools as proxies in the session's registry.

        Each tool becomes a real tool in the model's tool list. When the model
        calls it, the executor sends a ``tool.execute_request`` to the WS
        client and waits for ``tool.execute_result``.

        Args:
            client_id: The WebSocket client that owns the tools.
            tools: List of tool definition dicts.
            categories: Optional mapping of category name → description
                for categories introduced by these client tools.
        """
        if not self._event_sink_adapter:
            return
        session_id = self._event_sink_adapter._client_sessions.get(client_id, "")
        if not session_id or not self._command_router:
            return

        session = self._command_router._session_manager.get_session(session_id)
        if not session or not session.server or not session.server.registry:
            return

        import threading
        from jaato_sdk.plugins.model_provider.types import ToolSchema
        from jaato_sdk.events import ToolExecuteRequestEvent

        registry = session.server.registry

        for tool_def in tools:
            tool_name = tool_def.get('name', '')
            if not tool_name:
                continue

            description = tool_def.get('description', '')
            parameters = tool_def.get('parameters', {})
            category = tool_def.get('category', '')
            timeout = tool_def.get('timeout', 30000) / 1000.0  # ms -> seconds
            auto_approve = tool_def.get('auto_approve', True)

            # Create a waiting mechanism for this client's responses
            if not hasattr(self, '_client_tool_waiters'):
                self._client_tool_waiters = {}

            # Build the proxy executor.
            # The executor sends the request to whichever clients are
            # currently attached to the session (not a hardcoded client_id).
            # If no client is connected, it waits for one to appear —
            # same behavior as permission requests during disconnection.
            ws_server = self
            sid = session_id

            def make_executor(tname, tout):
                def executor(args):
                    import uuid
                    import time
                    call_id = str(uuid.uuid4())[:8]

                    # Create a threading Event to wait for the result
                    waiter = threading.Event()
                    result_holder = {'result': None, 'error': None}
                    ws_server._client_tool_waiters[call_id] = (waiter, result_holder)

                    request_event = ToolExecuteRequestEvent(
                        call_id=call_id,
                        agent_id='',
                        tool_name=tname,
                        tool_args=args,
                    )

                    def _send_to_session_clients():
                        """Send the request to all clients attached to the session."""
                        if not ws_server._event_sink_adapter or not ws_server._event_sink_adapter._event_loop:
                            return False
                        sent = False
                        for cid, csid in ws_server._event_sink_adapter._client_sessions.items():
                            if csid == sid:
                                import asyncio
                                asyncio.run_coroutine_threadsafe(
                                    ws_server._send_to_client(cid, request_event),
                                    ws_server._event_sink_adapter._event_loop,
                                )
                                sent = True
                        return sent

                    # Try to send; if no client connected, poll until one appears
                    deadline = time.time() + tout
                    sent = _send_to_session_clients()
                    while not sent and time.time() < deadline:
                        time.sleep(1.0)
                        sent = _send_to_session_clients()

                    if not sent:
                        return {'error': f'No client connected to receive tool call {tname}'}

                    # Wait for result with remaining timeout
                    remaining = max(0, deadline - time.time())
                    if waiter.wait(timeout=remaining):
                        if result_holder['error']:
                            return {'error': result_holder['error']}
                        # Success: the client's DECODED result verbatim (a
                        # native dict), symmetric with an in-process tool — NOT
                        # wrapped under a "result" envelope (which buried the
                        # real fields from the ledger / enrichment).
                        return result_holder['result']
                    else:
                        return {'error': f'Client tool {tname} timed out after {tout}s'}
                return executor

            executor = make_executor(tool_name, timeout)

            # Register as a tool in the session's registry
            schema = ToolSchema(
                name=tool_name,
                description=description + ' [client-provided]',
                parameters=parameters,
                category=category or None,
            )
            registry.register_core_tool(schema, executor, auto_approved=auto_approve)
            # Track the schema so the RUNNER-tier model receives it (the daemon
            # register_core_tool above only reaches the daemon registry; the
            # model runs in the runner subprocess).  spawn_session_runner seeds
            # envelope.client_tools from here for register-before-session.new.
            session.server.client_tool_schemas[tool_name] = {
                "name": tool_name,
                "description": description,
                "parameters": parameters,
                "category": category or "",
            }

            logger.info(
                "Registered client tool '%s' for client %s (timeout=%ss, auto_approve=%s)",
                tool_name, client_id, timeout, auto_approve,
            )

        # Register client-provided category descriptions so list_tools
        # shows them instead of empty strings.
        if categories and isinstance(categories, dict):
            for cat_name, cat_desc in categories.items():
                if cat_name and cat_desc:
                    registry.register_category(cat_name, cat_desc)

        # Refresh the runtime's tool schema list so the model sees new tools.
        # Phase 3 §7c step 6.6.4.5a: read ``session.server._runtime`` directly
        # instead of going through ``session.server._jaato.get_runtime()``.
        runtime = session.server._runtime if session.server else None
        if runtime and hasattr(runtime, '_all_tool_schemas'):
            existing = {s.name for s in runtime._all_tool_schemas}
            for name, schema in registry._core_tools.items():
                if name not in existing:
                    runtime._all_tool_schemas.append(schema)
            logger.info("Refreshed runtime tool list for client %s", client_id)

        # Glue the schemas to the RUNNER-tier model (where the model actually
        # runs — the daemon-runtime refresh above is daemon-side only).  For a
        # tool registered AFTER session.new, this RPC registers it on the LIVE
        # runner registry so the model's next get_tool_schemas surfaces it
        # without a session restart.  On RE-ATTACH this push is REQUIRED, not
        # cosmetic: the re-spawned runner does NOT already hold the client tools
        # (they don't survive in the restored envelope), so a dropped push leaves
        # the turn unservable — which is why the push thread below waits for
        # bootstrap-complete rather than firing best-effort into a not-ready slot.
        runner_rpc = getattr(session.server, "_runner_rpc", None)
        if runner_rpc is not None:
            # Off the daemon event loop (this runs in the async dispatch; the
            # threadsafe RPC's future.result() against the same loop would
            # deadlock).  Best-effort background thread; proxy already registered.
            import threading

            def _push(server=session.server, t=tools, cid=client_id,
                      sid=session.session_id):
                try:
                    # The rpc handle can be live BEFORE the runner finishes
                    # session.bootstrap — most acutely on a reused warm pool slot
                    # (handle reused on claim, bootstrap still running).  Pushing
                    # into that window hit a 15s TimeoutError and the runner never
                    # got the client tools -> unservable turn (the re-attach
                    # stall).  Gate on bootstrap-complete (mark_runner_ready),
                    # mirroring the send-path gate.
                    ready = getattr(server, "_runner_ready", None)
                    if ready is not None and not ready.wait(timeout=30.0):
                        logger.warning(
                            "mid-session client-tool push for %s: runner not "
                            "ready within 30s — skipping", cid)
                        return
                    # Re-read the current rpc after readiness (robust to a
                    # re-spawn during the wait).
                    rpc = getattr(server, "_runner_rpc", None)
                    if rpc is None:
                        return  # runner torn down during the wait
                    rpc.session_register_client_tools_threadsafe(t)
                    # Runner is ready and the client tools are now registered
                    # on it — re-emit the tool-id registry OFF the loop.  The
                    # runner RPC session_get_tool_schemas (in
                    # _build_tool_id_mappings) can only run off-loop, and it
                    # supplies the runner-tier names (prompt.* + these client
                    # tools) that the on-loop synchronous emit below skips to
                    # avoid the daemon-side prompt_library walk.  The WS event
                    # sink is thread-safe (run_coroutine_threadsafe).
                    server._emit_tool_id_registry_from_schemas()
                    # Deferred-turn (Option 2): the runner now has this client's
                    # host tools, so drive any wake deferred while the session was
                    # cold — the turn's schema will include these tools.  After
                    # the push (not at attach-end) closes the drive-races-tool-
                    # wiring gap.  Idempotent.
                    try:
                        self._command_router._session_manager.drive_pending_wake(sid)
                    except Exception:
                        logger.exception(
                            "deferred-wake drive after client-tool push failed "
                            "for %s", sid)
                except Exception:
                    logger.exception(
                        "mid-session client-tool runner push failed for %s", cid)

            threading.Thread(target=_push, daemon=True).start()

        # Emit updated tool ID registry so clients can resolve IDs for the
        # newly-registered client-provided tools.  This runs SYNCHRONOUSLY on
        # the event loop; the daemon walk in _build_tool_id_mappings ALWAYS
        # excludes runner-tier (prompt_library's ~15s filesystem walk blocked
        # the loop on a cold re-attach + self-blocked the register-RPC send
        # above).  Daemon-tier names map here; runner-tier names arrive from
        # the off-loop _push re-emit once the runner is ready.
        if session.server:
            session.server._emit_tool_id_registry_from_schemas()

    def _handle_tool_execute_result(self, client_id: str, event) -> None:
        """Route a tool execution result back to the waiting executor thread."""
        if not hasattr(self, '_client_tool_waiters'):
            return
        call_id = event.call_id
        entry = self._client_tool_waiters.pop(call_id, None)
        if not entry:
            logger.warning("No waiter for tool result call_id=%s", call_id)
            return
        waiter, result_holder = entry
        # Decode the JSON-encoded host-tool result to its native value
        # (symmetric with the SDK's encode) so it records like an in-process
        # tool result — see decode_client_tool_result.
        from jaato_server.server.client_tools import decode_client_tool_result
        result_holder['result'] = decode_client_tool_result(event.result)
        result_holder['error'] = event.error
        waiter.set()

    # =========================================================================
    # Workspace Management Handlers
    # =========================================================================

    async def _handle_workspace_list(self, client_id: str) -> None:
        """Handle workspace list request."""
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        # Scoped to the connection's authenticated user (#1074 tickets):
        # their own and the unowned workspaces.  No identity, no scoping.
        workspaces = self._workspace_manager.list_workspaces(
            for_user=self.get_client_user(client_id))
        await self._send_to_client(
            client_id,
            WorkspaceListEvent(
                root=str(self._workspace_manager.workspace_root),
                workspaces=[ws.to_dict() for ws in workspaces],
            )
        )

    async def _handle_workspace_create(self, client_id: str, name: str) -> None:
        """Handle workspace creation request."""
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        try:
            ws_info = self._workspace_manager.create_workspace(
                name, owner=self.get_client_user(client_id))
            # name/path are the event's declared identity fields; the dict is
            # the whole row.  Sending the dict ALONE reached clients as an
            # event with no name (the model dropped the undeclared key).
            await self._send_to_client(
                client_id,
                WorkspaceCreatedEvent(
                    name=ws_info.name,
                    path=ws_info.path,
                    workspace=ws_info.to_dict(),
                )
            )
        except ValueError as e:
            await self._send_error(client_id, str(e))

    async def _handle_workspace_delete(self, client_id: str, name: str) -> None:
        """Handle ``workspace.delete`` (protocol 1.13).

        Answers with one ``WorkspaceDeletedEvent`` whatever happened.  The
        manager refuses containment, ownership and other clients' selections;
        the loaded-session check needs the session manager, which lives
        behind the command router, so it is resolved here and handed in.
        """
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return
        try:
            path = self._workspace_manager.get_workspace_path(name)
            in_use = self._sessions_loaded_in(str(path)) if path else []
            self._workspace_manager.delete_workspace(
                name,
                user=self.get_client_user(client_id),
                in_use_by=in_use,
                client_id=client_id,
            )
        except ValueError as e:
            await self._send_to_client(
                client_id, WorkspaceDeletedEvent(name=name, ok=False, error=str(e)))
            return
        if self._event_sink_adapter and path is not None:
            if self._event_sink_adapter.get_client_workspace(client_id) == str(path):
                self._event_sink_adapter.clear_client_workspace(client_id)
        await self._send_to_client(client_id, WorkspaceDeletedEvent(name=name, ok=True))

    async def _handle_workspace_select(self, client_id: str, name: str) -> None:
        """Handle workspace selection request.

        This selects the workspace and returns its configuration status.
        Per-client workspace tracking is used so multiple clients can
        select different workspaces simultaneously.
        """
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        try:
            ws_info = self._workspace_manager.select_workspace(
                name, client_id=client_id, user=self.get_client_user(client_id))
            config_status = self._workspace_manager.get_config_status(name)

            # Send config status to client
            await self._send_to_client(
                client_id,
                ConfigStatusEvent(
                    workspace=name,
                    configured=ws_info.configured,
                    provider=ws_info.provider,
                    model=ws_info.model,
                    available_providers=config_status.get("available_providers", []),
                    missing_fields=config_status.get("missing_fields", []),
                )
            )

        except ValueError as e:
            await self._send_error(client_id, str(e))

    async def provision_workspace(
        self,
        client_id: str,
        session_id: str,
        template: Optional[str] = None,
    ) -> Optional[ProvisionedWorkspace]:
        """Auto-provision an isolated workspace for a remote session.

        Creates a new workspace directory, applies a template, and
        optionally sets up AppArmor confinement.

        Args:
            client_id: The client requesting the workspace.
            session_id: Session identifier (used as workspace directory name).
            template: Template name to apply (default: server's default_template).

        Returns:
            The provisioned workspace, or None on failure.
        """
        if not self._provisioner:
            logger.warning("Cannot provision workspace: no provisioner configured")
            return None

        try:
            workspace = self._provisioner.provision(
                session_id=session_id,
                client_id=client_id,
                template=template,
            )
        except ValueError as e:
            logger.error("Workspace provision failed: %s", e)
            return None

        # AppArmor profile is provisioned later in the session hook
        # (_apparmor_session_hook) using the session manager's session ID,
        # not the workspace UUID.  This ensures /tmp/jaato-{session_id}-**
        # matches what get_environment() returns to the agent.

        self._client_provisioned[client_id] = workspace
        return workspace

    def get_apparmor_confinement(
        self,
        session_id: str,
    ) -> Optional[Callable]:
        """Get the AppArmor BASE-profile confinement context for a session.

        Used for prefetch (configure-time dynamic-instructions
        expansion) and reactor dispatch — both need read access to
        user-authored config in ``.jaato/`` to load agent personas,
        validate completion payloads, etc.

        Tool execution uses :meth:`get_apparmor_tool_confinement`
        instead (server 0.6.55+) — that enters the sub-profile with
        added read-denies on user-authored config.

        Returns a zero-argument callable suitable for passing to
        ``JaatoServer.set_pre_init_confine_context()``, or ``None`` if
        AppArmor is not available.

        Args:
            session_id: Session identifier.

        Returns:
            Confinement context factory, or ``None``.
        """
        if not self._apparmor or not self._apparmor.is_available():
            return None
        from .apparmor import make_confine_context
        profile_name = self._apparmor.get_profile_name(session_id)
        return make_confine_context(profile_name)

    def get_apparmor_tool_confinement(
        self,
        session_id: str,
    ) -> Optional[Callable]:
        """Get the AppArmor tool-execution confinement context
        (server 0.6.55+, template v13+).

        Returns a zero-argument callable that enters the per-session
        ``tool_hat`` sub-profile.  Used by
        ``JaatoServer.set_apparmor_confinement()`` to wrap each tool
        call in ``ToolExecutor.execute`` so LLM-driven file reads
        can't see user-authored config (other agents' personas,
        profile JSON, schemas, scripts, instructions, reactors.json).

        Returns ``None`` if AppArmor is not available.

        Args:
            session_id: Session identifier.

        Returns:
            Tool-confinement context factory, or ``None``.
        """
        if not self._apparmor or not self._apparmor.is_available():
            return None
        from .apparmor import make_tool_confine_context
        profile_name = self._apparmor.get_profile_name(session_id)
        return make_tool_confine_context(profile_name)

    def get_reference_authorizer(self, session_id: str):
        """Get the AppArmor reference-fragment authorizer for a session.

        Returns a ``ReferenceAuthorizer`` suitable for passing to
        ``JaatoServer.set_reference_authorizer()`` so the references
        plugin can grant kernel-level readonly access to selected
        reference paths via per-session AppArmor fragments.

        Returns ``None`` when AppArmor is unavailable for this server;
        the references plugin then skips the fragment path entirely
        and relies on the in-process sandbox_manager allowlist alone.
        """
        if not self._apparmor or not self._apparmor.is_available():
            return None
        from .apparmor import ReferenceAuthorizer
        return ReferenceAuthorizer(self._apparmor, session_id)

    async def _handle_config_update(
        self,
        client_id: str,
        provider: str,
        model: Optional[str],
        api_key: Optional[str],
    ) -> None:
        """Handle workspace configuration update request.

        After successfully updating the workspace config, initializes a
        ``JaatoServer`` for the workspace so the client can start sending
        messages.  If auto-provisioning is active, the workspace is
        provisioned first and AppArmor confinement is applied.
        """
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
        if not selected:
            await self._send_error(client_id, "No workspace selected")
            return

        try:
            result = self._workspace_manager.update_config(
                provider=provider,
                model=model,
                api_key=api_key,
                name=selected.name,
            )

            await self._send_to_client(
                client_id,
                ConfigUpdatedEvent(
                    workspace=result["workspace"],
                    provider=result["provider"],
                    model=result["model"],
                    success=result["success"],
                )
            )

            # Initialize JaatoServer now that the workspace is configured
            if result["success"]:
                await self._initialize_server_for_workspace(client_id, selected)

        except ValueError as e:
            await self._send_error(client_id, str(e))

    async def _initialize_server_for_workspace(
        self,
        client_id: str,
        workspace_info: Any,
    ) -> None:
        """Initialize a JaatoServer for the selected workspace.

        Creates the server from the workspace's ``.env`` file, initializes
        it in a background thread, and optionally applies AppArmor
        confinement.

        This is called after ``_handle_config_update()`` succeeds, meaning
        the workspace has a valid provider configuration.

        Args:
            client_id: The requesting client.
            workspace_info: The ``WorkspaceInfo`` for the selected workspace.
        """
        env_file = self._workspace_manager.get_env_file(workspace_info.name)
        if not env_file or not env_file.exists():
            await self._send_error(client_id, "Workspace .env file not found")
            return

        # Auto-provision an isolated workspace directory if provisioner is
        # configured.  This creates a session-specific subdirectory under
        # {root}/sessions/ with template contents and AppArmor confinement.
        session_id = workspace_info.name  # Use workspace name as session ID
        provisioned_ws = None
        if self._provisioner:
            provisioned_ws = await self.provision_workspace(
                client_id=client_id,
                session_id=session_id,
            )
            if provisioned_ws:
                # Use the provisioned workspace's .env instead
                provisioned_env = Path(provisioned_ws.path) / ".env"
                if provisioned_env.exists():
                    env_file = provisioned_env

        # Server 0.6.71+: bootstrap (JaatoServer construction +
        # initialize) runs in a fresh ContextVar context so this
        # session is isolated from any values inherited from the
        # caller's asyncio task.  See
        # ``shared.session_context.run_in_fresh_session_context``.
        from jaato_server.shared.session_context import run_in_fresh_session_context

        ws_path = provisioned_ws.path if provisioned_ws else workspace_info.path

        def _bootstrap_and_initialize() -> "tuple[Optional[JaatoServer], bool]":
            srv = JaatoServer(
                env_file=str(env_file),
                on_event=self._on_server_event,
                workspace_path=ws_path,
            )
            return srv, srv.initialize()

        # The whole bootstrap runs in an executor thread (initialize
        # is blocking); the executor's worker uses our fresh-context
        # wrapper internally so ContextVar inheritance from the parent
        # async task can't leak.
        server, success = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_in_fresh_session_context(_bootstrap_and_initialize),
        )

        if not success:
            await self._send_error(client_id, "Failed to initialize server")
            return

        self._jaato_server = server

        # Phase 2 (confined runner): standalone WS-server mode no
        # longer installs daemon-thread confinement.  The kernel-level
        # profile is provisioned during session creation; the runner
        # subprocess self-confines on spawn.  See
        # docs/design/per_session_confined_runner.md §4.6.

        await self._send_to_client(
            client_id,
            SystemMessageEvent(
                message=f"Server initialized: {server.model_provider}/{server.model_name}",
                style="info",
            ),
        )

    # =========================================================================
    # Status Methods
    # =========================================================================

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server is not None and not self._shutdown_event.is_set()

    def message_limits(self) -> Dict[str, int]:
        """The size limits this server enforces, as advertised to clients.

        Rides ``ConnectedEvent.server_info``.  ``max_message_size`` is the
        largest single WebSocket message accepted; the two staging caps
        are the per-file and per-request limits ``StageFilesRequest`` is
        judged against.  ``stage_per_file_limit`` is the SMALLER of the
        staging cap and the message size, because a staged file travels
        as one binary message and a file larger than that cannot arrive
        however generous the staging cap is.
        """
        return {
            "max_message_size": self._max_message_size,
            "stage_per_file_limit": min(
                DEFAULT_STAGE_PER_FILE_LIMIT, self._max_message_size
            ),
            "stage_total_limit": DEFAULT_STAGE_TOTAL_LIMIT,
        }

    def get_server_info(self) -> Dict[str, Any]:
        """Get server status information."""
        info = {
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running,
            "client_count": self.client_count,
            "workspace_mode": self._workspace_manager is not None,
            "model_provider": self._jaato_server.model_provider if self._jaato_server else None,
            "model_name": self._jaato_server.model_name if self._jaato_server else None,
            "is_processing": self._jaato_server.is_processing if self._jaato_server else False,
        }

        if self._workspace_manager:
            selected = self._workspace_manager.get_selected_workspace()
            info["workspace_root"] = str(self._workspace_manager.workspace_root)
            info["selected_workspace"] = selected.name if selected else None

        if self._provisioner:
            info["provisioned_workspaces"] = len(self._provisioner.list_workspaces())
            info["available_templates"] = self._provisioner.list_templates()

        if self._apparmor:
            info["apparmor_available"] = self._apparmor.is_available()

        return info


# =============================================================================
# Standalone Entry Point
# =============================================================================

async def main():
    """Run the WebSocket server standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Jaato WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--workspace-root",
        metavar="PATH",
        required=True,
        help="Root directory for workspaces (remote clients select from subdirectories)",
    )
    parser.add_argument(
        "--apparmor",
        default=None,
        action="store_true",
        dest="apparmor",
        help="Require AppArmor confinement: the server refuses to start if "
             "confinement is unavailable, rather than silently degrading to "
             "directory-sandbox-only isolation (default: auto-detect, which "
             "warns and degrades). Equivalent to JAATO_REQUIRE_APPARMOR=1.",
    )
    parser.add_argument(
        "--no-apparmor",
        action="store_false",
        dest="apparmor",
        help="Disable AppArmor confinement",
    )
    parser.add_argument(
        "--cgroups",
        default=None,
        action="store_true",
        dest="cgroups",
        help="Enable per-session cgroup v2 runtime limits (default: auto-detect)",
    )
    parser.add_argument(
        "--no-cgroups",
        action="store_false",
        dest="cgroups",
        help="Disable cgroup v2 runtime limits (app-layer caps still apply)",
    )
    parser.add_argument(
        "--cgroups-root",
        default="/sys/fs/cgroup/jaato",
        help="Parent cgroup v2 directory with delegated controllers "
             "(default: /sys/fs/cgroup/jaato)",
    )
    parser.add_argument(
        "--workspace-template",
        default="default",
        help="Default template for auto-provisioned workspaces (default: 'default')",
    )
    parser.add_argument(
        "--workspace-max-age",
        type=int,
        default=86400,
        help="Max age in seconds for provisioned workspaces (default: 86400)",
    )
    parser.add_argument(
        "--ws-max-message-size",
        metavar="SIZE",
        default=None,
        help="Largest WebSocket message accepted, in bytes or with a K/M/G "
             f"suffix (default: {DEFAULT_WS_MAX_MESSAGE_SIZE // (1024 * 1024)}M). "
             "A staged file travels as one message, so this also bounds "
             "the largest file a client can attach.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    try:
        max_message_size = (
            parse_ws_max_message_size(args.ws_max_message_size)
            if args.ws_max_message_size else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # #1168.  This entry point spawns per-session runners under its own
    # uid exactly as ``python -m server`` does, and it is the one the two
    # deployment guides the warning cites actually print
    # (docs/apparmor-setup.md, docs/runtime-limits-setup.md), so the
    # posture has to be announced here too -- a warning that fires on one
    # of two documented daemons is a warning an operator learns to
    # disbelieve.  No ``--umask`` flag here deliberately: this server's
    # flag surface is the isolation posture, and ``JAATO_UMASK`` is the
    # host-scoped knob that works on every entry point without one.
    from jaato_server.server.process_posture import apply_process_posture
    apply_process_posture()

    server = JaatoWSServer(
        host=args.host,
        port=args.port,
        workspace_root=args.workspace_root,
        apparmor=args.apparmor,
        cgroups=args.cgroups,
        cgroups_root=args.cgroups_root,
        default_template=args.workspace_template,
        workspace_max_age=args.workspace_max_age,
        max_message_size=max_message_size,
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
