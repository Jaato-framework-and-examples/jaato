"""Transport-agnostic command dispatcher for the Jaato daemon.

Extracted from ``JaatoDaemon._handle_session_request_inner()`` so that
both IPC and WebSocket transports route through the same dispatch logic.

The router owns no transport state — it receives events and emits
responses through the ``EventSink`` protocol.
"""

import json
import logging
from dataclasses import dataclass
import os
import pathlib
import uuid
from typing import Any, Dict, List, Optional

from jaato_sdk.events import Event
from jaato_server.server.event_sink import EventSink, client_peer
from jaato_server.server.session_manager import SessionManager
from jaato_server.server.session_logging import set_logging_context, clear_logging_context
from jaato_server.shared.path_utils import describe_relative_path
from jaato_server.shared.session_id import is_safe_session_id
from jaato_server.shared.peer_identity import unreachable_client_paths

logger = logging.getLogger(__name__)


#: The verbs that reach ANOTHER session, each handled by a method taking
#: ``(client_id, args, payload)``.  A table rather than three ``elif``
#: branches so ``_dispatch`` -- frozen in the complexity baseline -- does not
#: grow by one decision per verb of this family.
_SESSION_MESSAGING_VERBS = {
    "session.send": "_handle_session_send",
    "session.wake": "_handle_session_wake",
    "session.message": "_handle_session_message",
}


def _mapping_attachments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The ``attachments`` a payload carries, mapping entries only.

    Shared by the ``session.wake`` and ``session.message`` decoders so the
    two verbs cannot disagree about what an attachment on the wire is.  A
    path, a number or any other non-mapping entry is dropped here rather
    than handed onward as something the multimodal path would have to
    re-check (#845): the daemon cannot read a client's files, which is why
    ``_normalize_attachments`` expands paths on the sending side.
    """
    raw = payload.get("attachments")
    return [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []


@dataclass
class _MessageRequest:
    """One decoded ``session.message``: the fields the two accepted shapes
    agree on, plus the payload-only ones."""
    target: str
    text: str
    event_id: Optional[str]
    request_id: Optional[str]
    attachments: List[dict]
    file_refs: List[Any]
    text_attachments: List[dict]

    @property
    def has_content(self) -> bool:
        return bool(self.text or self.attachments or self.file_refs
                    or self.text_attachments)


def _decode_message_request(
    args: List[Any], payload: Optional[dict],
) -> _MessageRequest:
    """Decode a ``session.message`` request from either accepted shape.

    The structured ``payload`` wins over positional ``args`` field by
    field; the trailing positional form joins the remainder so a typed
    message need not be quoted.  ``attachments``, ``file_refs``,
    ``text_attachments`` and ``request_id`` are payload-only -- bytes have
    no positional spelling (#845), and ``CommandRequest`` has no field for
    a correlation id, so the SDK puts it in the payload and it is echoed
    from there.  Only mapping attachments survive, as for ``session.wake``;
    ``file_refs`` and ``text_attachments`` are handed on as lists and
    judged row by row by the daemon method, which refuses a malformed row
    BY INDEX rather than dropping it (design §4.5).
    """
    p = payload or {}
    target = p.get("target") or (args[0] if len(args) > 0 else None)
    text = p.get("text") or (" ".join(args[1:]) if len(args) > 1 else "")
    request_id = p.get("request_id")
    refs = p.get("file_refs")
    texts = p.get("text_attachments")
    return _MessageRequest(
        target=str(target).strip() if target is not None else "",
        text=str(text or ""),
        event_id=p.get("event_id"),
        request_id=str(request_id) if request_id is not None else None,
        attachments=_mapping_attachments(p),
        file_refs=list(refs) if isinstance(refs, list) else [],
        text_attachments=list(texts) if isinstance(texts, list) else [],
    )


def _message_result_event(request_id: Optional[str], target: str,
                          receipt: Dict[str, Any]) -> Any:
    """Render a ``deliver_group_message`` receipt as the typed result event.

    ``ok`` is the DELIVERED set, ``spooled`` (the target's durable inbox
    holds it -- phase 2) and the benign ``duplicate`` (a redelivered
    ``event_id`` is an idempotent no-op, not a failed delivery -- the
    ``session.wake`` reading).
    """
    from jaato_sdk.events import SessionMessageResultEvent
    status = str(receipt.get("status") or "refused")
    return SessionMessageResultEvent(
        request_id=request_id,
        target=target,
        status=status,
        ok=status in ("accepted", "queued", "spooled", "duplicate"),
        spooled=bool(receipt.get("spooled")),
        message_id=receipt.get("message_id") or "",
        target_session_id=receipt.get("target_session_id") or "",
        sibling_name=receipt.get("sibling_name") or "",
        group_key=receipt.get("group_key") or "",
        woken=bool(receipt.get("woken")),
        headless=bool(receipt.get("headless")),
        candidates=list(receipt.get("candidates") or []),
        files=list(receipt.get("files") or []),
        error=receipt.get("error") or receipt.get("detail") or "",
    )


def _decode_wake_request(
    args: List[Any], payload: Optional[dict],
) -> "tuple[Optional[str], Optional[str], str, Optional[str], List[dict]]":
    """Decode a ``session.wake`` request from either accepted shape.

    Returns ``(session_id, text, source, event_id, attachments)``.  The
    structured ``payload`` wins over positional ``args`` field by field, as
    it always has.

    ``attachments`` (#845) is payload-only: bytes have no positional
    spelling, and a positional string would be a CLIENT-SIDE path the daemon
    cannot read (the whole reason ``_normalize_attachments`` expands paths on
    the sending side, especially cross-host WS).  Only mapping entries
    survive; anything else is dropped here rather than being handed onward
    as something the multimodal path would have to re-check.  Dropping is
    not silent in effect — a wake whose attachments were ALL unusable and
    carried no text fails the caller's usage check and is refused by name.
    """
    p = payload or {}
    session_id = p.get("session_id") or (args[0] if len(args) > 0 else None)
    text = p.get("text") or (args[1] if len(args) > 1 else None)
    source = p.get("source") or (args[2] if len(args) > 2 else "user")
    event_id = p.get("event_id") or (args[3] if len(args) > 3 else None)
    attachments = _mapping_attachments(p)
    return session_id, text, source, event_id, attachments



def _json_safe(data: Any) -> Any:
    """A rendering's structured half, reduced to what the wire can carry.

    ``explain`` renderers build dicts for a CLI that prints them with
    ``json.dumps(..., default=str)``, so a value the wire cannot encode — a
    ``Path``, an enum, a dataclass — is normal rather than exceptional here.
    Encoding with that same fallback and decoding back is what makes the
    daemon's ``--json`` output the CLI's own, instead of a frame the
    serialiser refuses and a caller left waiting for a reply that never
    comes.

    The SHAPE is preserved, never normalised: ``explain profile`` renders an
    array, and turning it into an object here would make the daemon's
    ``--json`` differ from the same command's local ``--json``.  Only a value
    that cannot be encoded at all degrades, to an empty object, because a
    frame that is never written leaves the caller waiting.
    """
    import json
    try:
        return json.loads(json.dumps(data, default=str))
    except Exception:                             # pragma: no cover - defensive
        return {}


def _daemon_version() -> str:
    """This daemon's ``jaato-server`` version, for a remote rendering's byline.

    Best-effort: an answer that could not name the install is still a useful
    answer, and a diagnostic verb must not fail on its own provenance line.
    """
    try:
        from importlib.metadata import version as pkg_version
        return pkg_version("jaato-server")
    except Exception:                             # pragma: no cover - defensive
        return ""


def _integration_refresh_fields(result: Any) -> Dict[str, Any]:
    """Read the ``--refresh`` contract's four fields off whatever it returned.

    ``scaffold.integration`` (1.21) surfaces the ``jaato-scaffold integration
    --refresh`` result the ``integrations`` module produces — deliberately
    NOT re-deriving the safe-state rule (#1261 owns it), only reading its
    documented ``--json`` shape: ``state_before`` / ``state_after`` /
    ``changed`` / ``skipped_reason``, plus the human ``lines``.

    Tolerant of a mapping (the ``--json`` dict) or an object carrying the
    same attributes, and of missing keys, so the handler shapes one event
    however the contract is spelled — a diagnostic verb must not fail on the
    provenance of its own result.
    """
    def field(key: str, default: Any) -> Any:
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)

    lines = field("lines", [])
    if not isinstance(lines, list):
        lines = []
    return {
        "state_before": str(field("state_before", "") or ""),
        "state_after": str(field("state_after", "") or ""),
        "changed": bool(field("changed", False)),
        "skipped_reason": str(field("skipped_reason", "") or ""),
        "text": "\n".join(str(line) for line in lines),
    }


class CommandRouter:
    """Transport-agnostic command dispatcher for the Jaato daemon.

    Handles command routing for session management (``session.new``,
    ``session.list``, etc.), tool management (``tools.list``, ``tools.enable``,
    etc.), daemon-level plugin commands (auth providers), and delegates
    session-scoped requests to ``SessionManager``.

    This class owns no transport state — it receives events and emits
    responses through the ``EventSink`` protocol.

    Lifecycle:
        1. Constructed by ``JaatoDaemon.start()`` with dependencies.
        2. ``handle_request()`` called by each transport's message handler.
        3. Events emitted via ``EventSink.send_event()``.
        4. Session-scoped events delegated to ``SessionManager.handle_request()``.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        event_sink: EventSink,
        daemon_plugins: dict,
    ):
        """Initialize the command router.

        Args:
            session_manager: The daemon's session manager instance.
            event_sink: Transport-agnostic event delivery sink (composite
                of IPC + WS sinks in daemon mode).
            daemon_plugins: Dict of name -> plugin instance for
                session-independent plugins (auth providers, etc.).
        """
        self._session_manager = session_manager
        self._event_sink = event_sink
        self._daemon_plugins = daemon_plugins

        # ``SessionInfoEvent.sessions`` is the same listing ``session.list``
        # renders, and it was built unscoped -- so a client the #1113
        # boundary shows 2 sessions was handed every session on the daemon
        # the moment it attached.  The manager holds an event callback
        # rather than an ``EventSink`` and cannot ask a transport about a
        # client, and a second copy of the rule living there is how the two
        # answers come to differ again.  This is the one place holding both
        # halves, so the router lends the manager its own method.
        #
        # Tolerated when absent, the shape ``client_peer`` and
        # ``client_visible_workspaces`` already have for a sink predating a
        # method: this constructor is handed arbitrary objects and must not
        # raise at one that is not a full ``SessionManager``.  Announced,
        # never silent -- without it the snapshot is unscoped, which is the
        # defect this exists to close.
        lend = getattr(session_manager, "set_visible_sessions_resolver", None)
        if lend is None:
            logger.warning(
                "session manager %s cannot scope its state snapshot: "
                "SessionInfoEvent.sessions will list every session on this "
                "daemon, not the ones each client may see",
                type(session_manager).__name__)
        else:
            lend(self._sessions_visible_to)

        # Pending workspace mismatch requests: client_id -> {request_id, session_id, ...}
        self._pending_workspace_mismatch: dict = {}

        # Pending post-auth setup requests: client_id -> {request_id, provider_name}
        self._pending_post_auth: dict = {}

    def handle_client_disconnect(self, client_id: str) -> None:
        """Notify the router that a transport client has disconnected.

        Detaches ``client_id`` from any session it was attached to.
        ``SessionManager.detach_client`` also calls
        ``_maybe_unload_session`` which releases per-session resources
        (workspace monitor inotify handle, etc.) once no other clients
        are attached.

        Called by transport servers (IPC, WS) from their disconnect
        handlers.  Per-transport cleanup (e.g. WS
        ``workspace_manager.remove_client`` /
        ``event_sink_adapter.remove_client``) stays in the transport
        server — only the session-detachment step is transport-agnostic.

        Phase 2 cascade-as-client (server 0.6.156+): also clean up
        any cascade-client registrations this client made.  Without
        this, IPC client crash → registrations leak until the GC
        sweep timeout (300s default).  Explicit cleanup ensures
        immediate resource release.
        """
        self._session_manager.detach_client(client_id)
        self._session_manager.unregister_all_cascade_clients_for_connection(
            client_id
        )

    def handle_request(
        self,
        client_id: str,
        session_id: str,
        event: Event,
    ) -> None:
        """Dispatch a client request to the appropriate handler.

        This is the main entry point, called from transport handlers
        (IPC ``on_session_request``, WS ``_handle_message``).

        Sets up per-session logging context, dispatches the event, and
        clears the logging context on exit.

        Args:
            client_id: The requesting client's ID.
            session_id: The client's current session ID (may be empty).
            event: The event to dispatch.
        """
        # Set logging context from existing session (if any) so all logger
        # calls in this thread are routed to per-session log files.
        existing_session = self._session_manager.get_client_session(client_id)
        workspace = self._event_sink.get_client_workspace(client_id)
        if existing_session and existing_session.server:
            set_logging_context(
                session_id=existing_session.session_id,
                client_id=client_id,
                workspace_path=existing_session.workspace_path or workspace,
                session_env=existing_session.server.get_all_session_env(),
            )
        elif session_id and workspace:
            # No loaded session yet but we have identifiers (e.g. attach path)
            set_logging_context(
                session_id=session_id,
                client_id=client_id,
                workspace_path=workspace,
            )

        try:
            self._dispatch(client_id, session_id, event)
        finally:
            clear_logging_context()

    def _handle_set_workspace(
        self,
        client_id: str,
        args: List[str],
    ) -> None:
        """Register the client's workspace, or refuse a relative one.

        ``set_workspace`` is the first command a client sends after
        connecting, and the workspace it registers is inherited by every
        session that client later creates.  A RELATIVE path is refused
        rather than absolutised: the daemon would resolve it against its
        OWN cwd, which is not the client's and which a daemon restart can
        change, so the session would run in a different directory from the
        one the client writes to and reads back — with no error on either
        side (issue #742).

        Args:
            client_id: The requesting client.
            args: The command's arguments; ``args[0]`` is the workspace.
        """
        workspace_path = args[0] if args else None
        if not workspace_path:
            return
        bad = describe_relative_path(
            "workspace", workspace_path,
            origin="the daemon boundary (set_workspace)",
        )
        if bad:
            logger.error("Client %s: set_workspace refused: %s",
                         client_id, bad)
            from jaato_sdk.events import ErrorEvent
            self._event_sink.send_event(client_id, ErrorEvent(
                error=f"set_workspace: {bad}",
                error_type="RelativePathAcrossBoundary",
                recoverable=True,
            ))
            return
        refusals = unreachable_client_paths(
            [("workspace", workspace_path)],
            # Through the shared tolerance, not a direct attribute read: a
            # sink predating ``get_client_peer`` must contribute "no peer"
            # rather than raise, exactly as it does inside the composite.
            client_peer(self._event_sink, client_id),
        )
        if refusals:
            error = (
                "set_workspace refused — the connecting account cannot "
                "reach that path:\n" + "\n".join(f"  - {m}" for m in refusals)
            )
            logger.error("Client %s: %s", client_id, error)
            from jaato_sdk.events import ErrorEvent
            self._event_sink.send_event(client_id, ErrorEvent(
                error=error,
                error_type="PeerPathNotReachable",
                recoverable=True,
            ))
            return

        self._event_sink.set_client_workspace(client_id, workspace_path)
        logger.debug(f"Client {client_id} workspace set to: {workspace_path}")

    def _dispatch(
        self,
        client_id: str,
        session_id: str,
        event: Event,
    ) -> None:
        """Inner dispatch with logging context already set."""
        # Handle tool disable request (direct registry call, no response events)
        from jaato_sdk.events import ToolDisableRequest
        if isinstance(event, ToolDisableRequest):
            session = self._session_manager.get_client_session(client_id)
            if session and session.server and session.server.registry:
                session.server.registry.disable_tool(event.tool_name)
            return

        # Handle session management commands
        from jaato_sdk.events import CommandRequest

        if isinstance(event, CommandRequest):
            cmd = event.command.lower()

            # Handle set_workspace command (sent by client on connect)
            if cmd == "set_workspace":
                self._handle_set_workspace(client_id, event.args)
                return

            # Get client's workspace path for session operations
            workspace_path = self._event_sink.get_client_workspace(client_id)

            if cmd == "session.new":
                self._handle_session_new(
                    client_id,
                    event.args,
                    workspace_path,
                    payload=event.payload,
                )
                return

            elif cmd == "session.attach":
                self._handle_session_attach(client_id, session_id, event.args, workspace_path)
                return

            elif cmd == "session.list":
                self._handle_session_list(client_id, session_id)
                return

            elif cmd == "session.profiles":
                from jaato_sdk.events import SessionProfilesEvent
                profiles, parse_errors = self._session_manager.list_profiles(
                    workspace_path=workspace_path,
                )
                self._event_sink.send_event(client_id, SessionProfilesEvent(
                    profiles=profiles,
                    parse_errors=parse_errors,
                ))
                return

            elif cmd == "session.default":
                self._handle_session_default(client_id, workspace_path)
                return

            elif cmd == "session.end":
                self._handle_session_end(client_id, session_id)
                return

            elif cmd == "session.delete":
                self._handle_session_delete(client_id, event.args)
                return

            elif cmd in ("session.orphans", "session.stop", "session.reload_env"):
                self._dispatch_orphan_command(
                    cmd, client_id, event.args, session_id=session_id)
                return

            elif cmd == "session.help":
                self._handle_session_help(client_id)
                return

            elif cmd == "session.snapshot_workspace":
                self._handle_snapshot_workspace(client_id, event.args, workspace_path)
                return

            elif cmd == "session.save":
                self._handle_session_save(client_id, session_id, event.args)
                return

            elif cmd in ("session.send", "session.wake", "session.message"):
                # One branch for the three verbs that reach ANOTHER session
                # (nudge a named stage, wake by id, message a group peer):
                # same shape, same (args, payload) contract.  The literals
                # stay here so a reader (and the session.send guard) finds
                # the verb where every other verb is.
                getattr(self, _SESSION_MESSAGING_VERBS[cmd])(
                    client_id, event.args, event.payload)
                return

            elif cmd == "session.bind_wake":
                self._handle_session_bind_wake(client_id, event.args, event.payload)
                return

            elif cmd == "session.unbind_wake":
                self._handle_session_unbind_wake(client_id, event.args, event.payload)
                return

            elif self._dispatch_prefixed_command(
                    cmd, client_id, event.args, event.payload, workspace_path,
                    session_id=session_id):
                return

            # Tools commands - handled per-session
            elif cmd.startswith("tools."):
                self._handle_tools_command(client_id, cmd, event.args)
                return

        # Handle WorkspaceMismatchResponseRequest
        from jaato_sdk.events import WorkspaceMismatchResponseRequest, WorkspaceMismatchResolvedEvent
        if isinstance(event, WorkspaceMismatchResponseRequest):
            self._handle_workspace_mismatch_response(client_id, event)
            return

        # Handle HistoryRequest
        from jaato_sdk.events import HistoryRequest, HistoryEvent
        if isinstance(event, HistoryRequest):
            self._handle_history_request(client_id, event)
            return

        # Handle daemon-level plugin commands (session-independent plugins).
        # These are always routed through the daemon path regardless of session
        # state, because they need daemon-level features (e.g., post-auth wizard).
        if isinstance(event, CommandRequest):
            plugin = self._find_daemon_plugin_for_command(event.command)
            if plugin:
                self._execute_daemon_command(client_id, plugin, event.command, event.args)
                return

        # Handle post-auth setup response
        from jaato_sdk.events import PostAuthSetupResponse
        if isinstance(event, PostAuthSetupResponse):
            self._handle_post_auth_response(client_id, event)
            return

        # Route to session.  The transport's authenticated user rides
        # along so a permission response can be attributed (#859).
        self._session_manager.handle_request(
            client_id, session_id, event,
            user_id=self._event_sink.get_client_user(client_id),
            # The kernel-vouched account on the far end of this connection.
            # Read here for the same reason ``user_id`` is: the transport
            # owns it and the event body must never be able to claim it.
            peer=self._event_sink.get_client_peer(client_id),
        )

    # ------------------------------------------------------------------
    # Session commands
    # ------------------------------------------------------------------

    def _dispatch_orphan_command(
        self, cmd: str, client_id: str, args: list,
        session_id: Optional[str] = None,
    ) -> None:
        """Route the session-administration verbs that share one branch.

        One branch in :meth:`_dispatch` for all of them, rather than one
        each: that method sits at its cyclomatic-complexity baseline and may
        not grow.  The two orphan verbs (#812) are one feature — you list
        orphans in order to stop one — and ``session.reload_env`` rides the
        same branch because it is the same shape: an operator verb about a
        loaded session's runtime state, answered with one confirmation line.

        Args:
            cmd: ``"session.orphans"``, ``"session.stop"`` or
                ``"session.reload_env"``.
            client_id: The requesting client.
            args: The command's argv tail.
            session_id: The caller's own session, which ``reload_env``
                targets when no id is given.
        """
        if cmd == "session.orphans":
            self._handle_session_orphans(client_id)
        elif cmd == "session.reload_env":
            self._handle_session_reload_env(client_id, session_id, args)
        else:
            self._handle_session_stop(client_id, args)

    def _dispatch_prefixed_command(
        self, cmd: str, client_id: str, args: list, payload: Any,
        workspace_path: Optional[str], session_id: Optional[str] = None,
    ) -> bool:
        """Route the ``cascade.*`` family and ``workspace.ignore`` from ONE branch.

        :meth:`_dispatch` is frozen at its complexity baseline, so the
        ``cascade.`` arm it already paid for is widened into a table rather
        than joined by a sibling: the two families share the property that
        the verb is daemon-level (no session round-trip) and answers on the
        caller's own channel.

        Returns:
            True when the command was handled; False when ``cmd`` belongs to
            neither family, so the caller keeps dispatching.
        """
        if cmd.startswith("cascade."):
            return self._dispatch_cascade_command(cmd, client_id, args, payload)
        if cmd == "workspace.ignore":
            self._handle_workspace_ignore(
                client_id, args, workspace_path, session_id=session_id)
            return True
        if cmd == "scaffold.explain":
            self._handle_scaffold_explain(
                client_id, args, workspace_path, session_id=session_id)
            return True
        if cmd == "scaffold.integration":
            self._handle_scaffold_integration(
                client_id, args, workspace_path, session_id=session_id)
            return True
        return False

    def resolve_caller_workspace(
        self, client_id: str, client_workspace: Optional[str],
        session_id: Optional[str] = None,
    ) -> "tuple[Optional[str], Dict[str, Optional[str]]]":
        """The workspace a daemon-level verb acts in for *client_id*.

        Three sources, first hit wins, all of them returned so a refusal
        can say which were empty rather than "no workspace":

        1. the session the manager has this client attached to;
        2. the session the TRANSPORT says the client is on (``session_id``
           from ``handle_request`` -- the WS adapter's own map, which can
           know the session across a reconnect the manager has not yet
           re-bound);
        3. the workspace the client declared (IPC ``set_workspace``, or the
           WS selection).

        A session's path outranks a declared one because the session's tree
        is what ``WorkspaceMonitor`` watches and what the panel shows.
        """
        session = self._session_manager.get_client_session(client_id)
        attached = getattr(session, "workspace_path", None) if session else None
        by_id = None
        if session_id and hasattr(self._session_manager, "get_session"):
            target = self._session_manager.get_session(session_id)
            by_id = getattr(target, "workspace_path", None) if target else None
        sources = {"attached_session": attached or None,
                   "transport_session": by_id or None,
                   "declared": client_workspace or None}
        return (attached or by_id or client_workspace or None), sources

    def _handle_workspace_ignore(
        self, client_id: str, args: list, client_workspace: Optional[str],
        session_id: Optional[str] = None,
    ) -> None:
        """Handle ``workspace.ignore <path>`` (protocol 1.12).

        Toggles one exact entry in the caller's workspace ``.gitignore`` —
        the TUI workspace panel's ``i`` key, served daemon-side so a remote
        client (the web coding UI) can do what the TUI does by writing the
        file itself.  The edit is
        :func:`jaato_sdk.gitignore_toggle.toggle_gitignore_pattern` on both
        routes, so one press means one thing whichever client made it.

        Which ``.gitignore``: :meth:`resolve_caller_workspace` -- the
        SESSION's workspace when the caller is attached to one (by the
        manager's binding, or by the session id the transport handed in),
        else the workspace the client declared or selected.
        The session's is the tree ``WorkspaceMonitor`` watches — and the
        monitor reloads its parser on this very write, so the pattern binds
        every later file event.  Entries the panel already shows are NOT
        pruned; the client's own hide is for that.

        The path is a PATTERN written into a file inside the workspace, never
        a path the daemon resolves, so #742's relative-path rule does not
        apply; what is refused instead is anything that is not a workspace
        entry (:func:`validate_ignore_pattern`).  Every outcome — including
        a refusal — answers with one ``WorkspaceIgnoreResultEvent``, because
        the caller is a panel that has to render *something* for the press.
        """
        import os

        from jaato_sdk.events import WorkspaceIgnoreResultEvent
        from jaato_sdk.gitignore_toggle import (
            toggle_gitignore_pattern, validate_ignore_pattern,
        )

        pattern = args[0] if args else ""

        def answer(**fields: Any) -> None:
            self._event_sink.send_event(
                client_id, WorkspaceIgnoreResultEvent(path=pattern, **fields))

        reason = validate_ignore_pattern(pattern)
        if reason:
            answer(ok=False, error=f"workspace.ignore: {reason}")
            return

        workspace, sources = self.resolve_caller_workspace(
            client_id, client_workspace, session_id)
        if not workspace:
            # Name what was looked at: "no workspace" alone sent a reader
            # who could see their workspace in the header to the wrong place.
            checked = ", ".join(
                f"{k}={'none' if v is None else repr(v)}"
                for k, v in sources.items())
            logger.warning("workspace.ignore: client=%s session=%s has no "
                           "resolvable workspace (%s)", client_id,
                           session_id or "-", checked)
            answer(ok=False, error=f"workspace.ignore: the caller has no "
                                   f"workspace ({checked})")
            return

        gitignore_path = os.path.join(workspace, ".gitignore")
        try:
            existing = ""
            if os.path.exists(gitignore_path):
                with open(gitignore_path, "r", encoding="utf-8") as fh:
                    existing = fh.read()
            new_content, ignored = toggle_gitignore_pattern(existing, pattern)
            with open(gitignore_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)
        except OSError as exc:
            logger.warning("workspace.ignore: client=%s could not write %s: %s",
                           client_id, gitignore_path, exc)
            answer(ok=False, error=f"workspace.ignore: could not write "
                                   f"{gitignore_path}: {exc}",
                   gitignore_path=gitignore_path)
            return

        logger.info("workspace.ignore: client=%s %s %r in %s", client_id,
                    "added" if ignored else "removed", pattern, gitignore_path)
        answer(ok=True, ignored=ignored, gitignore_path=gitignore_path)

    def _handle_scaffold_explain(
        self, client_id: str, args: list, client_workspace: Optional[str],
        session_id: Optional[str] = None,
    ) -> None:
        """Handle ``scaffold.explain [topic] [name]`` (protocol 1.18).

        ``jaato-scaffold explain`` introspects the framework installed in the
        CALLING process.  That is the whole answer while the CLI and the
        daemon share a virtualenv, and silently wrong the moment they do not
        — the normal shape of a deployed application: ``jaato-sdk`` in the
        application's own ``.venv``, driving a daemon owned by a different
        user.  There are then TWO installs, and a topic contributed through
        ``jaato.scaffold_topics`` by a package only the daemon has (today:
        jaato-premium's ``reactors``) exists in one of them.  The CLI's
        refusal reads as *no such topic exists*, which sends a reader looking
        for a feature they already have.  This verb is how the other process
        is asked.

        **One dispatch, not two.**  It calls
        :func:`shared.scaffold.__main__.render_topic` — the same function the
        CLI calls — so the daemon cannot grow a second opinion about what a
        topic answers, which is the failure this seam exists to remove rather
        than one to reproduce over a socket.

        The answer always carries ``topics``: the catalog THIS install can
        render.  A refusal that only says "unknown" is exactly as unhelpful
        remotely as it was locally, and the caller's own catalog is by
        construction the wrong one to list.

        Every outcome answers with one ``ScaffoldExplainEvent`` — the caller
        is blocked on a reply, so a refusal that emitted nothing would be
        indistinguishable from a daemon that does not serve the verb, which
        is the state the protocol floor exists to keep visible.
        """
        from jaato_sdk.events import ScaffoldExplainEvent

        topic = (args[0] if len(args) > 0 else "") or None
        name = (args[1] if len(args) > 1 else "") or None
        topics: list = []

        def answer(**fields: Any) -> None:
            # `topics` rides EVERY outcome, which is what makes the event's
            # own promise true: "which topics does this daemon have" is
            # precisely the question a failed lookup raises, and it is the
            # one question the caller's own catalog cannot answer.
            self._event_sink.send_event(
                client_id,
                ScaffoldExplainEvent(topic=topic or "",
                                     topics=topics,
                                     server_version=_daemon_version(),
                                     **fields))

        try:
            from jaato_server.shared.scaffold.__main__ import render_topic, scope_catalog
        except Exception as exc:
            # jaato-server is what serves this daemon, so the import failing
            # is a broken install rather than a missing optional extra — but
            # a diagnostic verb that raises is worse than one that says so.
            logger.warning("scaffold.explain: client=%s cannot load the "
                           "scaffold renderer: %s", client_id, exc)
            answer(ok=False,
                   error=f"scaffold.explain: this daemon cannot load its own "
                         f"scaffold renderer: {exc}")
            return

        try:
            topics = list(scope_catalog())
        except Exception:                         # pragma: no cover - defensive
            topics = []

        workspace, _sources = self.resolve_caller_workspace(
            client_id, client_workspace, session_id)

        try:
            ok, data, text, error = render_topic(topic, name, workspace or ".")
        except Exception as exc:                  # pragma: no cover - defensive
            logger.warning("scaffold.explain: client=%s topic=%r raised: %s",
                           client_id, topic, exc)
            answer(ok=False, error=f"scaffold.explain {topic!r}: {exc}")
            return

        logger.info("scaffold.explain: client=%s topic=%r ok=%s", client_id,
                    topic, ok)
        answer(ok=ok, text=text, data=_json_safe(data), error=error)

    def _handle_scaffold_integration(
        self, client_id: str, args: list, client_workspace: Optional[str],
        session_id: Optional[str] = None,
    ) -> None:
        """Handle ``scaffold.integration <name>`` (protocol 1.21).

        The sibling of :meth:`_handle_scaffold_explain`.  ``explain`` renders
        a topic from the daemon's install; this RUNS ``jaato-scaffold
        integration <name> --refresh`` into the caller's own workspace, on the
        daemon's install and host.  The point is the same "which install?"
        answer: the payload the integration writes (the ``jaato-sdk`` skill)
        carries a stamp naming the version of whichever ``jaato-server`` runs
        it, and the workspace directory is on that host — so the install that
        serves the session is the one that must write it.  An application that
        carried its own copy of the skill could drift from the framework;
        asking the daemon means it never can.

        **The ``--refresh`` rule is #1261's, not this handler's.**  It calls
        :func:`integrations.refresh` — the one place the safe-state decision
        (apply ``absent`` / ``stale`` / ``outdated``, leave ``edited`` /
        ``diverged`` / ``unstamped`` alone) lives — and reads its documented
        ``--json`` fields off the result rather than re-deriving them, so the
        daemon cannot grow a second opinion about when a local edit is
        overwritten.

        The workspace is the caller's own, from
        :meth:`resolve_caller_workspace` — the same entitlement path
        ``scaffold.explain`` and ``workspace.file.fetch`` use, so there is no
        directory parameter to check.  Every outcome answers with one
        :class:`ScaffoldIntegrationEvent`, because the caller is blocked on a
        reply and silence would be indistinguishable from a daemon that does
        not serve the verb — the state the protocol floor keeps visible.  A
        refresh the contract DECLINED to apply (an edited copy) is ``ok=True``
        with a ``skipped_reason``: leaving a local edit alone is correct, not
        a failure.
        """
        from jaato_sdk.events import ScaffoldIntegrationEvent

        name = (args[0] if args else "") or ""
        available: list = []

        def answer(**fields: Any) -> None:
            # `available` rides EVERY outcome so an unknown-name refusal is
            # actionable: the caller's own list of integrations is by
            # construction the wrong one, exactly as `scaffold.explain`'s
            # `topics` is.
            self._event_sink.send_event(
                client_id,
                ScaffoldIntegrationEvent(integration=name,
                                         available=available,
                                         server_version=_daemon_version(),
                                         **fields))

        try:
            from jaato_server.shared.scaffold import integrations
        except Exception as exc:                  # pragma: no cover - defensive
            logger.warning("scaffold.integration: client=%s cannot load the "
                           "scaffold integrations module: %s", client_id, exc)
            answer(ok=False,
                   error=f"scaffold.integration: this daemon cannot load its "
                         f"own scaffold code: {exc}")
            return

        try:
            available = list(integrations.available())
        except Exception:                         # pragma: no cover - defensive
            available = []

        if not name:
            answer(ok=False,
                   error=f"scaffold.integration: an integration name is "
                         f"required; this daemon ships: "
                         f"{', '.join(available) or '(none)'}")
            return

        # Validate the name against what this build ships BEFORE resolving a
        # workspace or calling refresh — an unknown integration is the
        # caller's mistake and wants a listing, not a stack trace from the
        # payload copier.
        if name not in available:
            answer(ok=False,
                   error=f"scaffold.integration: unknown integration "
                         f"{name!r}; this daemon ships: "
                         f"{', '.join(available) or '(none)'}")
            return

        workspace, sources = self.resolve_caller_workspace(
            client_id, client_workspace, session_id)
        if not workspace:
            checked = ", ".join(
                f"{k}={'none' if v is None else repr(v)}"
                for k, v in sources.items())
            logger.warning("scaffold.integration: client=%s session=%s has no "
                           "resolvable workspace (%s)", client_id,
                           session_id or "-", checked)
            answer(ok=False,
                   error=f"scaffold.integration: the caller has no workspace "
                         f"({checked})")
            return

        try:
            dest = integrations.target_dir(name, user=False, workspace=workspace)
            result = integrations.refresh(name, dest)
        except Exception as exc:
            logger.warning("scaffold.integration: client=%s name=%r raised: %s",
                           client_id, name, exc)
            answer(ok=False, error=f"scaffold.integration {name!r}: {exc}")
            return

        fields = _integration_refresh_fields(result)
        logger.info("scaffold.integration: client=%s name=%r %s->%s changed=%s",
                    client_id, name, fields["state_before"],
                    fields["state_after"], fields["changed"])
        answer(ok=True, target=str(dest), **fields)

    def _dispatch_cascade_command(
        self, cmd: str, client_id: str, args: list, payload: Any = None,
    ) -> bool:
        """Route a ``cascade.*`` command to its handler.

        Extracted from :meth:`_dispatch` as a table rather than left as six
        ``elif`` arms: that method is over the complexity ceiling and frozen
        at its recorded size, so a feature that needs a new branch has to
        pay for one somewhere.  This block was the cheapest to lift — six
        uniform arms differing only in the handler and whether it takes the
        payload.

        Behaviour is unchanged: each command reaches the same handler with
        the same arguments, and an unrecognised ``cascade.*`` string falls
        through to the caller's remaining dispatch exactly as before.

        Args:
            cmd: The full command string, e.g. ``"cascade.budget.set"``.
            client_id: The requesting client.
            args: The command's argv tail.
            payload: The request's structured payload, for the two commands
                that read one.

        Returns:
            True when the command was handled; False when ``cmd`` is not a
            known ``cascade.*`` verb, so the caller keeps dispatching.
        """
        if cmd == "cascade.register":
            self._handle_cascade_register(client_id, args)
        elif cmd == "cascade.unregister":
            self._handle_cascade_unregister(client_id, args)
        elif cmd == "cascade.budget.set":
            self._handle_cascade_budget_set(client_id, args, payload)
        elif cmd == "cascade.budget.get":
            self._handle_cascade_budget_get(client_id, args)
        elif cmd == "cascade.budget.clear":
            self._handle_cascade_budget_clear(client_id, args)
        elif cmd == "cascade.cancel":
            self._handle_cascade_cancel(client_id, args)
        else:
            return False
        return True

    def _handle_session_new(
        self,
        client_id: str,
        args: list,
        workspace_path: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle ``session.new`` command.

        Accepted flags (CLI argv path, used by the TUI):
            --profile <name>            Runtime config (model, plugins, GC, etc.)
            --sibling-name <slug>          Cascade-scoped address other sessions
                                        use to reach this one via
                                        send_to_sibling.  Shape
                                        ^[a-z0-9][a-z0-9_-]{0,31}$, unique
                                        within the cascade.
            --agent <name>              Agent whose rendered markdown becomes
                                        the session's system instructions
            --instructions <text|@path> FULL OVERRIDE — replace the assembled
                                        system instruction with the supplied
                                        text (or the contents of @path).
                                        Drops the agent's own prompt and
                                        plugin tool hints too.  Use when you
                                        need a specific, minimal system
                                        prompt and nothing else.
            --no-instructions           PARTIAL SUPPRESSION — drop only the
                                        BASE layer (``.jaato/instructions/*``
                                        + premium baseline).  Agent prompt,
                                        plugin instructions, and framework
                                        constants still reach the model.
                                        The usual choice for fitting a
                                        session into a small context window.
            key=value                   Agent parameters (substituted into the
                                        agent's ``{{param}}`` placeholders)

        Remaining bare arguments are treated as the session name.

        SDK-only path (not exposed in TUI argv):
            ``payload['spec']``  — Inline profile spec dict (model, provider,
                                  plugins, plugin_configs, system_instructions,
                                  gc, etc.).  Mutually exclusive with the
                                  ``--profile`` flag above.  Lets SDK clients
                                  create sessions with custom config without
                                  writing a profile JSON to disk.  Validation
                                  and parsing happen in
                                  ``SessionManager.create_session``.
        """
        # Read FIRST, not at the call site below: every refusal this parser
        # can emit is an answer to this ``session.new``, and the client's
        # create-wait discards an ErrorEvent that does not carry the
        # correlation id (#882).
        request_id = (payload or {}).get("request_id")
        name = None
        profile_name = None
        agent_name = None
        system_instruction_override: Optional[str] = None
        suppress_base_instructions: bool = False
        agent_params: Dict[str, str] = {}
        cascade_driver_id: Optional[str] = None
        sibling_name: Optional[str] = None
        args_iter = iter(args)
        for arg in args_iter:
            if arg == "--profile":
                profile_name = next(args_iter, None)
            elif arg == "--agent":
                agent_name = next(args_iter, None)
            elif arg == "--instructions":
                from jaato_sdk.events import ErrorEvent
                raw = next(args_iter, None)
                if raw is None:
                    self._event_sink.send_event(client_id, ErrorEvent(
                        error="--instructions requires a value (text or @filepath)",
                        error_type="UsageError",
                        recoverable=True,
                        request_id=request_id,
                    ))
                    return
                system_instruction_override = self._resolve_instructions_value(
                    raw, workspace_path, client_id, request_id=request_id,
                )
                if system_instruction_override is None:
                    return  # error already emitted
            elif arg == "--no-instructions":
                suppress_base_instructions = True
            elif arg == "--sibling-name":
                # Cascade-scoped ADDRESS for sibling messaging (design §4).
                # Validated server-side for shape and for uniqueness within
                # the cascade; a bad or taken name fails the create rather
                # than silently producing an unaddressable session.
                sibling_name = next(args_iter, None)
            elif arg == "--cascade-driver-id":
                # Phase 2 cascade-sharing (server 0.6.144+): opaque
                # tenant ID identifying the cascade this session
                # belongs to.  Subsequent sessions of the same cascade
                # can reuse this session's pool slot (warm plugin
                # state + warm LSP server connections) — see
                # docs/design/runner-cascade-sharing.md.
                cascade_driver_id = next(args_iter, None)
            elif "=" in arg:
                key, _, value = arg.partition("=")
                agent_params[key] = value
            elif name is None:
                name = arg

        # Inline profile spec — SDK-only escape hatch carried in
        # CommandRequest.payload (no argv equivalent).  Validation
        # (mutual exclusion with profile_name, required fields) lives
        # in SessionManager.create_session so both paths share it.
        inline_profile_data = (payload or {}).get("spec")

        created_by = self._event_sink.get_client_user(client_id)
        new_session_id = self._session_manager.create_session(
            client_id, name, workspace_path=workspace_path,
            profile_name=profile_name,
            agent_name=agent_name,
            agent_params=agent_params if agent_params else None,
            created_by=created_by,
            system_instruction_override=system_instruction_override,
            suppress_base_instructions=suppress_base_instructions,
            inline_profile_data=inline_profile_data,
            cascade_driver_id=cascade_driver_id,
            sibling_name=sibling_name,
            # Correlation id from the generic payload escape hatch.  Echoed on
            # whichever event answers this create, so the caller can tell its
            # own answer from a concurrent one.
            request_id=request_id,
        )
        if new_session_id:
            # Update logging context now that session_id is known.
            # create_session() loaded the .env, so fetch session_env.
            new_session = self._session_manager.get_client_session(client_id)
            session_env = (
                new_session.server.get_all_session_env()
                if new_session and new_session.server else {}
            )
            set_logging_context(
                session_id=new_session_id,
                client_id=client_id,
                workspace_path=workspace_path,
                session_env=session_env,
            )
            logger.info(f"Session {new_session_id} created and context set")
            self._event_sink.set_client_session(client_id, new_session_id)
        else:
            self._hint_available_auth_providers(client_id)

    def _handle_session_attach(
        self, client_id: str, session_id: str, args: list,
        workspace_path: Optional[str],
    ) -> None:
        """Handle ``session.attach`` command."""
        if not args:
            return

        target_session_id = args[0]
        # Client-supplied id — reject a traversal / injection id before it
        # reaches the persistence / cgroup / apparmor sinks (defense in depth
        # with the sink-side validation; this gives a clean early error).
        if not is_safe_session_id(target_session_id):
            from jaato_sdk.events import ErrorEvent
            self._event_sink.send_event(client_id, ErrorEvent(
                error="invalid session_id: must match [A-Za-z0-9._-] "
                      "(1-256 chars) with no '..'",
                error_type="UsageError",
                recoverable=True,
            ))
            return
        if self._refuse_foreign_session(client_id, target_session_id):
            return
        # Check for workspace mismatch
        mismatch = self._session_manager.check_workspace_mismatch(
            target_session_id, workspace_path
        )
        if mismatch:
            session_workspace, client_workspace = mismatch
            # Emit mismatch event and wait for user response
            request_id = str(uuid.uuid4())
            self._pending_workspace_mismatch[client_id] = {
                "request_id": request_id,
                "session_id": target_session_id,
                "session_workspace": session_workspace,
                "client_workspace": client_workspace,
            }
            from jaato_sdk.events import WorkspaceMismatchRequestedEvent
            self._event_sink.send_event(client_id, WorkspaceMismatchRequestedEvent(
                request_id=request_id,
                session_id=target_session_id,
                session_workspace=session_workspace,
                client_workspace=client_workspace,
                response_options=[
                    {"key": "s", "label": "switch", "action": "switch",
                     "description": f"Switch to session workspace: {session_workspace}"},
                    {"key": "c", "label": "cancel", "action": "cancel",
                     "description": "Stay in current session"},
                ],
                prompt_lines=[
                    f"Workspace mismatch detected:",
                    f"  Session workspace: {session_workspace}",
                    f"  Your workspace:    {client_workspace}",
                    f"",
                    f"Choose an option:",
                    f"  [s] Switch to session's workspace",
                    f"  [c] Cancel and stay in current session",
                ],
            ))
            return

        # No mismatch, proceed with attach
        # Set context before attach so initialization logs are routed
        set_logging_context(
            session_id=target_session_id,
            client_id=client_id,
            workspace_path=workspace_path,
        )
        if self._session_manager.attach_session(
            client_id, target_session_id, workspace_path=workspace_path
        ):
            # Update context with session_env now that server is loaded
            attached = self._session_manager.get_client_session(client_id)
            if attached and attached.server:
                set_logging_context(
                    session_env=attached.server.get_all_session_env(),
                )
            self._event_sink.set_client_session(client_id, target_session_id)

    def _handle_session_save(
        self, client_id: str, session_id: str, args: list,
    ) -> None:
        """Handle ``session.save`` — flush a LIVE session's state to disk.

        ``SessionManager.save_session`` has always existed; nothing exposed
        it.  So a driver that wanted a session's transcript on disk had to
        force an unload — attach elsewhere and let the orphan sweep save it —
        which is a side effect standing in for an interface, and one that
        silently does nothing when the client is already attached elsewhere.

        Reported by the perpetual-monologue cascade, whose evidence for its
        strongest claim stayed a model's paraphrase rather than an artifact
        because the sending session's transcript was never re-saved.

        Saves the CALLER's session by default; pass a session id to save
        another (a driver saving a stage it is not attached to).  Saving is
        idempotent and does not disturb a running turn -- it writes the state
        as it stands.
        """
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent
        target = (args[0] if args else None) or session_id
        if not target:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=("session.save: no session — attach first, or pass "
                       "a session id"),
                error_type="UsageError",
                recoverable=True,
            ))
            return
        if self._session_manager.save_session(target):
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message=f"session.save: {target} written to disk",
            ))
            return
        # False means NOT LOADED — a distinct fact from a write failure, and
        # the caller needs it: an unloaded session is already on disk.
        self._event_sink.send_event(client_id, ErrorEvent(
            error=(f"session.save: {target} is not loaded in memory "
                   f"(an unloaded session is already persisted)"),
            error_type="SessionSaveError",
            recoverable=True,
        ))

    def _handle_session_send(
        self, client_id: str, args: list, payload: Optional[dict],
    ) -> None:
        """Handle ``session.send`` — nudge a NAMED session in a cascade.

        Design §9, client tier: a human or script reaches a stage directly,
        without the model relaying and without knowing an opaque session id.

        Accepts a structured ``payload`` (SDK callers) —
        ``{cascade_driver_id, sibling_name, text}`` — or positional ``args``
        ``[cascade_driver_id, sibling_name, text...]``.  The trailing
        positional form joins the remainder so a typed message need not be
        quoted; a message is prose and the shell has already split it.

        Authentication is the transport's boundary (IPC socket-mode / WS
        bearer token), as for ``session.wake``: this handler runs only for
        callers already past that gate.

        Distinct from ``session.wake``, which REVIVES a cold session and
        drives a turn.  This reaches a LOADED session only, and reports
        ``sibling_cold`` rather than quietly resurrecting one — the two are
        different acts and conflating them would make the smaller one
        silently perform the larger.
        """
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent
        p = payload or {}
        cid = p.get("cascade_driver_id") or (args[0] if len(args) > 0 else None)
        name = p.get("sibling_name") or (args[1] if len(args) > 1 else None)
        text = p.get("text") or (" ".join(args[2:]) if len(args) > 2 else None)
        if not cid or not name or not text:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=("session.send requires <cascade_driver_id> "
                       "<sibling_name> <message>"),
                error_type="UsageError",
                recoverable=True,
            ))
            return

        receipt = self._session_manager.send_to_named_session(cid, name, text)
        # ``accepted`` (a turn was started) and ``queued`` (the target is
        # mid-turn) are both successful DELIVERIES; everything else is a
        # refusal with a reason.
        if receipt.get("status") not in ("accepted", "queued"):
            self._event_sink.send_event(client_id, ErrorEvent(
                error=receipt.get("error", "session.send refused"),
                error_type="SessionSendError",
                recoverable=True,
            ))
            return
        # A receipt, not a reply: this says the message was handed to the
        # session, never that it was read or acted on.
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=f"session.send: {receipt['status']} → {name!r}",
        ))

    def _handle_session_message(
        self, client_id: str, args: list, payload: Optional[dict],
    ) -> None:
        """Handle ``session.message`` — a message from the CALLER'S session to
        another session in a common group (protocol 1.23).

        The client-tier form of the ``courier`` plugin's ``send_to_session``:
        the same daemon method, ``SessionManager.deliver_group_message``, with
        the sender being the caller's own session — resolved from
        ``client_id``, never from the payload, so a client can only speak AS
        the session it is attached to.  A cold target is woken.

        Accepts a structured ``payload`` (SDK callers) —
        ``{target, text, attachments?, file_refs?, text_attachments?,
        event_id?, request_id?}`` — or positional ``args``
        ``[target, text...]``; see :func:`_decode_message_request`.

        Always answers with ONE :class:`SessionMessageResultEvent` carrying
        the receipt — a driver branches on ``status``, so the receipt travels
        as fields rather than as a ``SystemMessageEvent`` string — except for
        a caller with no session, which is a ``SessionError`` (the reply that
        settles a client's ``ask()`` rather than hanging it, #1007).
        """
        from jaato_sdk.events import ErrorEvent
        req = _decode_message_request(args, payload)

        session = self._session_manager.get_client_session(client_id)
        if session is None:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=("session.message: no active session — attach to (or "
                       "create) the session you want to speak as"),
                error_type="SessionError",
                recoverable=True,
                request_id=req.request_id,
            ))
            return
        if not req.target or not req.has_content:
            receipt = {"status": "refused",
                       "error": ("session.message requires <target> and "
                                 "<message> (or attachments, file_refs or "
                                 "text_attachments)")}
        else:
            receipt = self._session_manager.deliver_group_message(
                session.session_id, req.target, req.text,
                attachments=req.attachments, file_refs=req.file_refs,
                text_attachments=req.text_attachments, event_id=req.event_id,
            )
        self._event_sink.send_event(
            client_id, _message_result_event(req.request_id, req.target, receipt))

    def _handle_session_wake(
        self, client_id: str, args: list, payload: Optional[dict],
    ) -> None:
        """Handle ``session.wake`` — start a USER turn on a session, reviving it
        if cold, for the client-agnostic wake primitive.

        Accepts a structured ``payload`` (SDK callers) —
        ``{session_id, text, source?, event_id?, attachments?}`` — or
        positional ``args`` ``[session_id, text, source?, event_id?]``.
        Authentication is the
        transport's boundary (IPC socket-mode / WS bearer token / the HTTP
        shim's #498 fail-closed check); this handler runs only for callers
        already past that gate.  On refusal it emits an ``ErrorEvent`` with the
        reason; on success the woken turn's output flows to the session's
        attached clients (the caller need not be one).

        ``attachments`` (protocol 1.5+, payload form only — bytes have no
        positional spelling) carries binary content in the same canonical wire
        shape ``SendMessageRequest`` accepts, so a session whose input is
        audio can be resumed with an utterance rather than only described in
        text (#845).  AN ATTACHMENT IS CONTENT: a wake carrying attachments
        and no text is valid — for a spoken utterance the attachment IS the
        message (#838) — so the usage check requires text OR attachments,
        not text.
        """
        from jaato_sdk.events import ErrorEvent
        session_id, text, source, event_id, attachments = _decode_wake_request(
            args, payload)
        if not session_id or not (text or attachments):
            self._event_sink.send_event(client_id, ErrorEvent(
                error=("session.wake requires session_id and text "
                       "(or attachments)"),
                error_type="UsageError",
                recoverable=True,
            ))
            return
        outcome, detail = self._session_manager.wake_session(
            session_id, text or "", source=source, event_id=event_id,
            attachments=attachments,
        )
        # Only genuine failures surface as an error.  OK and DUPLICATE are both
        # successes — a redelivered event_id is an idempotent no-op, not a
        # failed delivery (an HTTP shim maps both to 2xx; erroring here would
        # make every at-least-once redelivery look failed + trigger retries).
        if not outcome.is_success:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=f"session.wake refused ({outcome.value}): {detail}",
                error_type="WakeError",
                recoverable=True,
            ))

    def _handle_session_bind_wake(
        self, client_id: str, args: list, payload: Optional[dict],
    ) -> None:
        """Handle ``session.bind_wake`` — declare a wake binding for the CALLER'S
        OWN session (the SESSION-owned half of the wake contract).

        The bound session is always the caller's current session (resolved from
        ``client_id``), so a caller can only bind ITSELF — hijack-proof by
        construction.  Accepts a structured ``payload``
        ``{wake_ref, trust_keys: [PEM...], ttl_seconds?}`` (the real path —
        PEM keys are multi-line) or positional ``args`` ``[wake_ref, key...]``.
        Always replies with a :class:`WakeBindResultEvent` carrying the
        ``BindOutcome`` (route on it), and on success the echoed ``wake_ref`` +
        binding ``expires_at``.
        """
        from jaato_sdk.events import WakeBindResultEvent
        p = payload or {}
        wake_ref = p.get("wake_ref") or (args[0] if len(args) > 0 else "")
        raw_keys = p.get("trust_keys")
        if raw_keys is None:
            raw_keys = list(args[1:]) if len(args) > 1 else []
        # Normalize trust_keys so a malformed payload never crashes the handler
        # or silently splits a key: a lone PEM string → one-element list; a
        # list → keep only str items; anything else → empty (the registry then
        # returns NO_KEYS / MALFORMED_KEY, a clean outcome).
        if isinstance(raw_keys, str):
            trust_keys = [raw_keys]
        elif isinstance(raw_keys, (list, tuple)):
            trust_keys = [k for k in raw_keys if isinstance(k, str)]
        else:
            trust_keys = []
        # Coerce ttl to int-or-None so a bad type (e.g. a JSON string) never
        # reaches the registry's arithmetic; on failure fall back to the default.
        raw_ttl = p.get("ttl_seconds")
        ttl_seconds: Optional[int] = None
        if raw_ttl is not None:
            try:
                ttl_seconds = int(raw_ttl)
            except (TypeError, ValueError):
                ttl_seconds = None

        session = self._session_manager.get_client_session(client_id)
        if session is None or not session.session_id:
            self._event_sink.send_event(client_id, WakeBindResultEvent(
                wake_ref=wake_ref or "", outcome="no_session",
                detail="caller has no active session to bind"))
            return

        outcome = self._session_manager.bind_wake(
            wake_ref, session.session_id, session.workspace_path,
            list(trust_keys), ttl_seconds,
            # Capture the caller session's cid so a deferred wake reaches its
            # cascade observers and the observer survives the session going cold.
            cascade_driver_id=getattr(session, "cascade_driver_id", None))
        expires_at = 0.0
        if outcome.is_ok:
            b = self._session_manager.resolve_wake_binding(wake_ref)
            if b is not None:
                expires_at = b.expires_at
        self._event_sink.send_event(client_id, WakeBindResultEvent(
            wake_ref=wake_ref or "", outcome=outcome.value,
            detail=f"bind_wake: {outcome.value}", expires_at=expires_at,
            # Surface the daemon's public wake endpoint so the caller can embed
            # it as the relay's routing marker (no bot-side URL config).
            endpoint=self._session_manager.wake_public_url))

    def _handle_session_unbind_wake(
        self, client_id: str, args: list, payload: Optional[dict],
    ) -> None:
        """Handle ``session.unbind_wake`` — remove the caller's own wake binding
        (owner-guarded).  ``{wake_ref}`` payload or ``[wake_ref]`` args."""
        from jaato_sdk.events import WakeBindResultEvent
        p = payload or {}
        wake_ref = p.get("wake_ref") or (args[0] if len(args) > 0 else "")
        session = self._session_manager.get_client_session(client_id)
        if session is None or not session.session_id:
            self._event_sink.send_event(client_id, WakeBindResultEvent(
                wake_ref=wake_ref or "", outcome="no_session",
                detail="caller has no active session"))
            return
        outcome = self._session_manager.unbind_wake(wake_ref, session.session_id)
        self._event_sink.send_event(client_id, WakeBindResultEvent(
            wake_ref=wake_ref or "", outcome=outcome.value,
            detail=f"unbind_wake: {outcome.value}"))

    def _sessions_visible_to(self, client_id: str) -> list:
        """The daemon's sessions, scoped to what this client's user may see.

        The boundary is the one the transport reports through
        ``visible_workspace_paths`` (protocol 1.13): ``None`` -- IPC, or a
        WS connection carrying no identity -- means the unscoped listing
        every client always got.  A list means a session is shown when it
        runs in one of those workspaces, or when this user created it
        (``created_by``, #859), and hidden otherwise -- another user's
        session in a shared workspace included.  ``session.list`` renders
        this set and ``session.attach`` admits only members of it, so the
        listing is never wider or narrower than what the verb accepts.
        """
        from jaato_server.server.event_sink import client_visible_workspaces

        sessions = self._session_manager.list_sessions()
        paths = client_visible_workspaces(self._event_sink, client_id)
        if paths is None:
            return sessions
        user = self._event_sink.get_client_user(client_id)
        if not isinstance(user, str):
            user = None
        roots = [os.path.normpath(p) for p in paths]

        def _inside(workspace_path: Optional[str]) -> bool:
            if not workspace_path:
                return False
            wp = os.path.normpath(workspace_path)
            return any(wp == r or wp.startswith(r + os.sep) for r in roots)

        return [s for s in sessions
                if (user is not None and s.created_by == user) or _inside(s.workspace_path)]

    def _refuse_foreign_session(
        self,
        client_id: str,
        target_session_id: str,
        verb: str = "session.attach",
    ) -> bool:
        """Refuse *verb* on a session outside the caller's boundary.

        Returns True (and has answered the client) when the command must
        not proceed.  Unscoped transports never refuse here.

        ``session.delete`` takes the same gate as ``session.attach``,
        which it did not before: #1113 wrote the boundary in terms of the
        two verbs that READ (``session.list`` renders the set,
        ``session.attach`` admits only members of it) and the verb that
        DESTROYS was simply not among them.  It mattered less while the
        snapshot leak handed every id to every client and cold deletes
        silently did nothing; with both fixed, an id is the only thing
        standing between one user and another user's session record, and
        an id here is a second-granularity timestamp.
        """
        from jaato_server.server.event_sink import client_visible_workspaces
        if client_visible_workspaces(self._event_sink, client_id) is None:
            return False
        if any(s.session_id == target_session_id
               for s in self._sessions_visible_to(client_id)):
            return False
        from jaato_sdk.events import ErrorEvent
        self._event_sink.send_event(client_id, ErrorEvent(
            error=f"{verb}: {target_session_id} is not one of your sessions",
            error_type="SessionError",
            recoverable=True,
        ))
        logger.info("%s: client=%s refused foreign session %s",
                    verb, client_id, target_session_id)
        return True

    def _handle_session_list(self, client_id: str, session_id: str) -> None:
        """Handle ``session.list`` command."""
        sessions = self._sessions_visible_to(client_id)
        from jaato_sdk.events import SessionListEvent

        # Get client's current session to mark it in the list
        current_session_id = session_id  # From the event

        # Send structured session data - client handles formatting
        session_data = [{
            "id": s.session_id,
            "name": s.name or "",
            "description": s.description or "",
            "model_provider": s.model_provider or "",
            "model_name": s.model_name or "",
            "is_loaded": s.is_loaded,
            "is_current": s.session_id == current_session_id,
            "client_count": s.client_count,
            "turn_count": s.turn_count,
            "workspace_path": s.workspace_path or "",
            # #812: which sessions nothing is consuming, and which process
            # each is using.  Additive keys on an already free-form dict, so
            # no protocol bump and an older client simply ignores them.
            "orphaned": s.orphaned,
            "runner": s.runner,
            # #1138: which session is WAITING ON YOU, and since when.  The
            # listing is the only channel that can answer it -- a
            # ``PermissionRequestedEvent`` reaches
            # ``session.attached_clients`` and ``_client_to_session`` is
            # 1:1, so a client working in session A never learns that B is
            # blocked.  Carried here and not only computed: a field
            # resolved, rendered and reaching no client is #1133.
            #
            # These two DO carry a protocol bump (1.17) where #812's pair
            # did not, and the difference is what a client does with them:
            # ``orphaned`` / ``runner`` are diagnostics a human reads,
            # while ``awaiting`` gates whether a client interrupts a
            # person.  A client that cannot tell "nothing is waiting" from
            # "this daemon never says" reports the first when the truth is
            # the second, and ``ConnectedEvent.protocol_version`` is the
            # only way to ask.
            "awaiting": s.awaiting,
            "awaiting_since": s.awaiting_since,
            # Session group messaging, phase 2: how many messages wait in
            # this session's durable inbox.  A diagnostic in #812's shape
            # (additive, no bump): a cold session with a pending message is
            # the one the watchdog is about to revive.
            "inbox_pending": s.inbox_pending,
        } for s in sessions]

        self._event_sink.send_event(client_id, SessionListEvent(sessions=session_data))

    def _handle_session_default(
        self, client_id: str, workspace_path: Optional[str],
    ) -> None:
        """Handle ``session.default`` command."""
        default_session_id = self._session_manager.get_or_create_default(
            client_id, workspace_path=workspace_path,
            # The transport's authenticated user, read HERE for the same
            # reason ``session.new`` and the post-auth create read it here:
            # the sink is the only thing that knows, and the event body
            # must never be able to claim it (#859).
            created_by=self._event_sink.get_client_user(client_id),
        )
        if default_session_id:
            # Update context now that session exists
            default_session = self._session_manager.get_client_session(client_id)
            if default_session and default_session.server:
                set_logging_context(
                    session_id=default_session_id,
                    client_id=client_id,
                    workspace_path=workspace_path,
                    session_env=default_session.server.get_all_session_env(),
                )
            self._event_sink.set_client_session(client_id, default_session_id)
        else:
            # Session creation failed (e.g., missing MODEL_NAME).
            self._hint_available_auth_providers(client_id)

    def _handle_cascade_budget_set(
        self, client_id: str, args: list, payload: Any = None,
    ) -> None:
        """Handle ``cascade.budget.set`` — declare a cascade's AGGREGATE cap.

        Wire: ``args = [cascade_driver_id]``, ``payload = {"limits": {...},
        "degrade": [...]}`` (the same ``budget_control`` shape a profile
        uses, so authors write one grammar).

        Declared on the cascade OWNER rather than a profile because a cap is
        a runtime aggregate over one live cid, not a property of a reusable
        template — see docs/design/budget-control-degradation.md §3.1.
        """
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent
        from jaato_server.shared.budget_control import BudgetControlConfig

        if not args:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=("cascade.budget.set requires args: [cascade_driver_id] "
                       "and a payload of {limits, degrade}"),
                error_type="UsageError", recoverable=True))
            return
        cid = args[0]
        try:
            config = BudgetControlConfig.from_dict(payload or {})
        except Exception as exc:  # noqa: BLE001 — author error, not a crash
            self._event_sink.send_event(client_id, ErrorEvent(
                error=f"cascade.budget.set: invalid budget: {exc}",
                error_type="UsageError", recoverable=True))
            return
        if config is None:
            self._event_sink.send_event(client_id, ErrorEvent(
                error="cascade.budget.set: payload declared no limits",
                error_type="UsageError", recoverable=True))
            return
        self._session_manager.set_cascade_budget(cid, config)
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=(f"cascade.budget.set: cid={cid} "
                     f"limits={dict(config.limits)}"),
            level="info"))

    def _handle_cascade_budget_get(self, client_id: str, args: list) -> None:
        """Handle ``cascade.budget.get`` — report a cascade's headroom.

        Replies with a ``SystemMessageEvent`` carrying JSON:
        ``{"cascade_driver_id", "limits", "remaining", "usage_fraction",
        "pressure"}``, or ``{"declared": false}`` when the cid is uncapped.

        This is the client-side witness for the pool depleting across
        stages — independent corroboration of the daemon's clamp decision
        rather than only the framework reporting what it decided.
        """
        import json
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent

        if not args:
            self._event_sink.send_event(client_id, ErrorEvent(
                error="cascade.budget.get requires args: [cascade_driver_id]",
                error_type="UsageError", recoverable=True))
            return
        cid = args[0]
        pool = self._session_manager.get_cascade_budget(cid)
        if pool is None:
            body = {"cascade_driver_id": cid, "declared": False}
        else:
            body = {
                "cascade_driver_id": cid,
                "declared": True,
                "limits": dict(pool.config.limits),
                "remaining": pool.remaining(),
                "usage_fraction": pool.usage_fraction(),
                "pressure": pool.describe_pressure(),
                # Scope, stated at the point of use.  "cascade budget"
                # reads as "the most this cascade can cost" and it is NOT
                # that: a child whose spawn declared its own budget_control
                # is outside this pot entirely.  Someone who knew the rule,
                # and had the warning in front of them, still summed
                # budgeted children into this figure and reported a
                # catastrophic ceiling failure that was entirely correct
                # behaviour.  A caveat in prose did not prevent it; a field
                # in the payload might.
                "covers": (
                    "sessions in this cascade that did NOT declare their own "
                    "budget_control; children with their own budget are "
                    "accounted separately and are not bounded by this pot"
                ),
            }
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=json.dumps(body), level="info"))

    def _handle_cascade_budget_clear(self, client_id: str, args: list) -> None:
        """Handle ``cascade.budget.clear`` — drop a cascade's pool."""
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent

        if not args:
            self._event_sink.send_event(client_id, ErrorEvent(
                error="cascade.budget.clear requires args: [cascade_driver_id]",
                error_type="UsageError", recoverable=True))
            return
        cid = args[0]
        self._session_manager.clear_cascade_budget(cid)
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=f"cascade.budget.clear: cid={cid}", level="info"))

    def _handle_cascade_register(self, client_id: str, args: list) -> None:
        """Handle ``cascade.register`` command (Phase 2 cascade-as-client).

        Wire format: ``args = [cascade_driver_id, role, *event_types]``
        where role is ``"owner"`` or ``"observer"`` and event_types is
        a list of event type-names (empty list = subscribe to all).

        Server-side: registers an in-process cascade-client via
        SessionManager whose callback routes matching events back to
        this connected client via the existing event_sink.  The
        registration entry uses a namespaced client_id
        ``_cascade:{cid}:{connection_client_id}`` so disconnect
        cleanup can match by suffix.

        Errors (bad args / duplicate registration / owner conflict)
        surface as ErrorEvent to the requesting client and the
        registration is dropped.
        """
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent

        if not args or len(args) < 2:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=(
                    "cascade.register requires args: "
                    "[cascade_driver_id, role, *event_types]"
                ),
                error_type="UsageError",
                recoverable=True,
            ))
            return

        cascade_driver_id = args[0]
        role = args[1]
        event_types = set(args[2:]) if len(args) > 2 else None

        # The cascade-client registration identifier — namespaced so
        # disconnect cleanup can match by suffix (one client may
        # register for multiple cids).
        cascade_client_id = f"_cascade:{cascade_driver_id}:{client_id}"

        # Callback closure routes events back to this connected
        # client via the existing event_sink.  Capture client_id by
        # value so multiple registrations don't shadow each other.
        connection_client_id = client_id

        def _cascade_event_callback(event):
            self._event_sink.send_event(connection_client_id, event)

        try:
            self._session_manager.register_in_process_client(
                client_id=cascade_client_id,
                callback=_cascade_event_callback,
                cascade_driver_id=cascade_driver_id,
                role=role,
                event_types=event_types,
                # server 0.6.178+: pass the raw connection id so the
                # routing-layer dedup at
                # ``_dispatch_to_cascade_clients_by_cid`` can skip
                # this entry when ``_route_bootstrap_event`` is
                # already delivering via the direct-IPC path to the
                # same connection.  Without this, the bootstrap-time
                # AgentCreatedEvent arrives twice on cascade_develop
                # walker's SDK queue (kb-side report 2026-06-03,
                # 0.6.177 falsification: PR-207 compared the wrong
                # identifier, ``cascade_client_id`` is the namespaced
                # registration id, NOT the raw connection id this
                # callback delivers to).
                delivery_target_id=connection_client_id,
            )
        except ValueError as exc:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=str(exc),
                error_type="CascadeRegistrationError",
                recoverable=True,
            ))
            return

        # Confirm registration so the SDK iterator can start yielding.
        # Uses SystemMessageEvent (existing typed event) to avoid a
        # new event class for Phase 2 — Phase 3 may upgrade to a
        # typed CascadeRegisteredEvent if downstream code wants it.
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=(
                f"cascade.register: registered cid={cascade_driver_id} "
                f"role={role} event_types="
                f"{sorted(event_types) if event_types else 'ALL'}"
            ),
            style="system",
        ))
        logger.info(
            "cascade.register: client=%s cid=%s role=%s event_types=%s",
            client_id, cascade_driver_id, role,
            sorted(event_types) if event_types else "ALL",
        )

    def _handle_cascade_unregister(self, client_id: str, args: list) -> None:
        """Handle ``cascade.unregister`` command (Phase 2).

        Wire format: ``args = [cascade_driver_id]``.

        Removes this client's registration for the given cid.
        Idempotent — silent no-op if already unregistered (e.g., the
        SDK iterator's auto-cleanup fired after disconnect cleanup
        already ran).
        """
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent

        if not args:
            self._event_sink.send_event(client_id, ErrorEvent(
                error="cascade.unregister requires args: [cascade_driver_id]",
                error_type="UsageError",
                recoverable=True,
            ))
            return

        cascade_driver_id = args[0]
        cascade_client_id = f"_cascade:{cascade_driver_id}:{client_id}"
        removed = self._session_manager.unregister_cascade_client(
            cascade_driver_id, cascade_client_id,
        )
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=(
                f"cascade.unregister: cid={cascade_driver_id} "
                f"{'removed' if removed else 'not-found (idempotent)'}"
            ),
            style="system",
        ))
        logger.info(
            "cascade.unregister: client=%s cid=%s removed=%s",
            client_id, cascade_driver_id, removed,
        )

    def _handle_cascade_cancel(self, client_id: str, args: list) -> None:
        """Handle ``cascade.cancel`` command.

        Wire format: ``args = [cascade_driver_id]``.

        Cancels every loaded session whose ``cascade_driver_id``
        matches.  Reactor extensions consult
        :meth:`SessionManager.is_cid_cancelled` before firing on
        ``AgentCompletedEvent`` so the cascade stops spawning new
        sessions.  Designed for kb-side ^C → IPC verb ergonomic
        (cascade_develop.py SIGINT handler).

        Idempotent — re-cancelling an already-cancelled cid returns
        zero counts but keeps the marker set so reactor suppression
        stays active.

        Args:
            client_id: The IPC/WS client that sent the command.
                Receives the SystemMessageEvent confirmation.
            args: Single-element list containing the cascade_driver_id.
        """
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent

        if not args:
            self._event_sink.send_event(client_id, ErrorEvent(
                error="cascade.cancel requires args: [cascade_driver_id]",
                error_type="UsageError",
                recoverable=True,
            ))
            return

        cascade_driver_id = args[0]
        result = self._session_manager.cancel_cascade(cascade_driver_id)

        # Confirmation message back to the caller — operator sees what
        # got reaped without grepping logs.
        if result["stopped_count"] == 0:
            msg = (
                f"cascade.cancel: cid={cascade_driver_id} "
                f"no loaded sessions matched (cid marked cancelled — "
                f"reactor suppression engaged)"
            )
        else:
            msg = (
                f"cascade.cancel: cid={cascade_driver_id} "
                f"cancelled {result['stopped_count']} session(s): "
                f"{result['cancelled_session_ids']}"
            )
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=msg,
            style="system",
        ))
        logger.info(
            "cascade.cancel: client=%s cid=%s stopped_count=%d",
            client_id, cascade_driver_id, result["stopped_count"],
        )

    def _handle_session_orphans(self, client_id: str) -> None:
        """Handle ``session.orphans`` — list sessions nothing is consuming (#812).

        Wire: no args.  Answers with a :class:`SessionListEvent` whose rows
        are :meth:`SessionManager.list_orphan_sessions` dicts — the same
        event type ``session.list`` uses, so a client that can render one can
        render the other, and the richer per-orphan keys (``orphaned_seconds``,
        the effective bounds, the unload grace and what is left of it (#1106),
        the ``runner`` identity) ride along.

        An orphan is a LOADED session with no attached client at all — not
        even the synthetic headless marker a woken or cascade-driven session
        carries.  See :meth:`SessionManager._is_orphaned`.

        Pairs with ``session.stop <id>``: the ``session_id`` of a row here is
        exactly what that verb takes.

        Args:
            client_id: The IPC/WS client that asked.  Receives the listing.
        """
        from jaato_sdk.events import SessionListEvent

        orphans = self._session_manager.list_orphan_sessions()
        self._event_sink.send_event(
            client_id, SessionListEvent(sessions=orphans))
        logger.info(
            "session.orphans: client=%s reported %d orphaned session(s): %s",
            client_id, len(orphans), [o["session_id"] for o in orphans],
        )

    def _handle_session_stop(self, client_id: str, args: list) -> None:
        """Handle ``session.stop <session_id>`` — stop ONE session by id (#812).

        Wire: ``args = [session_id]``.

        Distinct from ``session.end``, which stops the CALLER's own session,
        and from ``cascade.cancel``, which stops a whole cascade.  Neither
        could stop the one session an operator is looking at — which is what
        left "kill a circumstantially-identified runner on a shared daemon,
        or wait for the budget to burn" as the only two options in #812.

        Confirms back to the caller naming what happened, because "stopped a
        session that was mid-turn" and "stopped a session that was already
        idle" and "no such session is loaded" are three different answers and
        an operator acts differently on each.

        Args:
            client_id: The IPC/WS client that sent the command.
            args: Single-element list containing the session id to stop.
        """
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent

        if not args:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=("session.stop requires args: [session_id].  Use "
                       "session.orphans or session.list to find one."),
                error_type="UsageError",
                recoverable=True,
            ))
            return

        target = args[0]
        result = self._session_manager.stop_session(
            target, reason="operator_request")

        if not result["found"]:
            msg = (f"session.stop: {target} is not loaded — nothing to stop "
                   f"(a cold session consumes nothing)")
        elif result["was_processing"]:
            msg = (f"session.stop: {target} was mid-turn; cancellation "
                   f"requested (it stops at its next check point)")
        else:
            msg = f"session.stop: {target} was idle; terminated"
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=msg, style="system",
        ))
        logger.info(
            "session.stop: client=%s target=%s found=%s was_processing=%s",
            client_id, target, result["found"], result["was_processing"],
        )

    def _handle_session_reload_env(
        self, client_id: str, session_id: Optional[str], args: list,
    ) -> None:
        """Handle ``session.reload_env [session_id]``.

        Re-resolves the session's workspace ``.env`` (plus profile ``env:``
        and overrides) and has the runner re-apply it and rebuild the
        provider -- the way a credential stored with ``<provider>-auth key``
        or a ``.env`` line written after the runner booted reaches a session
        that is already open.  Defaults to the CALLER's session; an explicit
        id targets any loaded one, the way ``session.stop`` does.

        Confirms with one line naming the outcome, because the four answers
        call for different next moves: rebuilt (and on which credential
        source), busy mid-turn (retry when idle), not loaded, or a provider
        that would not rebuild on the new environment (the reason, verbatim).
        """
        from jaato_sdk.events import ErrorEvent, SystemMessageEvent

        target = args[0] if args else session_id
        if not target:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=("session.reload_env: no session -- attach to one or pass "
                       "its id as the first argument"),
                error_type="UsageError",
                recoverable=True,
            ))
            return

        outcome = self._session_manager.reload_session_env(target)
        if not outcome["found"]:
            msg = f"session.reload_env: {target} is not loaded -- nothing to reload"
        elif outcome["was_processing"]:
            msg = (f"session.reload_env: {target} is mid-turn; retry once it is "
                   f"idle (nothing was changed)")
        elif not outcome["ok"]:
            msg = (f"session.reload_env: {target}: environment re-resolved but "
                   f"the provider did not rebuild -- {outcome['error']}")
        else:
            result = outcome["result"] or {}
            source = result.get("auth_info") or "credential source not reported"
            msg = (f"session.reload_env: {target} reloaded {result.get('applied', 0)} "
                   f"env keys; provider {result.get('provider')} / "
                   f"{result.get('model')} rebuilt ({source})")
        style = "system" if outcome["ok"] else "warning"
        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=msg, style=style,
        ))
        logger.info(
            "session.reload_env: client=%s target=%s found=%s ok=%s was_processing=%s",
            client_id, target, outcome["found"], outcome["ok"],
            outcome["was_processing"],
        )

    def _maybe_reload_live_session_after_auth(
        self, client_id: str, plugin, args: list,
    ) -> None:
        """After ``<provider>-auth key|login`` succeeds, refresh the caller's live session.

        The credential the command just stored is on disk; the session this
        client is attached to resolved its credential at bootstrap and will
        not look again (see :meth:`_handle_session_reload_env`).  When that
        session runs the SAME provider the plugin authenticates, reload it
        now, so the very next turn uses the new credential instead of failing
        on the old one and sending the user to ``session.new``.

        Gated three ways so it never fires as a surprise: only for the
        actions that establish a credential (``login`` / ``key``, never
        ``status``); only when the client has a live session; only when that
        session's provider is the plugin's.  A session on another provider
        is left alone -- its credentials did not change.
        """
        action = str(args[0]).lower() if args else ""
        if action not in ("login", "key"):
            return
        session = self._session_manager.get_client_session(client_id)
        if session is None or session.server is None:
            return
        provider = getattr(plugin, "provider_name", None)
        active = getattr(session.server, "model_provider", None)
        if not provider or provider != active:
            return
        self._handle_session_reload_env(client_id, session.session_id, [])

    def _handle_session_end(self, client_id: str, session_id: str) -> None:
        """Handle ``session.end`` command.

        Cancellation-aware semantics (server 0.6.27+):

        - If the session is NOT currently processing (the agent already
          completed and the turn wrap-up has settled, OR the session is
          idle), this is a clean termination — no in-flight work to
          cancel, no spurious ``user_cancelled`` log marker.
        - If the session IS processing, stop() cancels via the cancel
          token (existing behavior).  This path is for explicit
          mid-turn cancellation.

        After either path, emits ``SessionTerminatedEvent`` (the new
        first-class event) AND the legacy
        ``SystemMessageEvent("[SESSION_TERMINATED]")`` for backward
        compatibility with clients that haven't migrated to the typed
        event yet.

        ``reason`` field distinguishes:
        - ``"client_request"`` — session was idle, end_session called
          cleanly.
        - ``"stopped"`` — session was processing, end_session cancelled.
        """
        session = self._session_manager.get_client_session(client_id)
        was_stopped = False
        agent_id = None
        if session and session.server:
            agent_id = getattr(session.server, "_main_agent_id", None) or "main"
            # stop() returns True only when it actually cancelled
            # in-flight work.  An idle session returns False — the
            # close is graceful and produces no user_cancelled marker.
            was_stopped = bool(session.server.stop())

        from jaato_sdk.events import SystemMessageEvent, SessionTerminatedEvent
        # Typed event — the canonical signal for clients.
        self._session_manager._emit_to_session(
            session_id,
            SessionTerminatedEvent(
                session_id=session_id,
                agent_id=agent_id,
                reason="stopped" if was_stopped else "client_request",
            ),
        )
        # Legacy string-based marker — kept for backward compatibility.
        # Clients reading the typed event can ignore this.  Will be
        # deprecated in a future release.
        self._session_manager._emit_to_session(
            session_id,
            SystemMessageEvent(message="[SESSION_TERMINATED]", style="system"),
        )

    def _handle_session_delete(self, client_id: str, args: list) -> None:
        """Handle ``session.delete`` command."""
        from jaato_sdk.events import SystemMessageEvent
        if not args:
            return

        session_id_to_delete = args[0]
        if self._refuse_foreign_session(
                client_id, session_id_to_delete, verb="session.delete"):
            return
        if self._session_manager.delete_session(session_id_to_delete):
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message=f"Session '{session_id_to_delete}' deleted.",
                style="info",
            ))
        else:
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message=f"Session '{session_id_to_delete}' not found.",
                style="warning",
            ))

    def _handle_session_help(self, client_id: str) -> None:
        """Handle ``session.help`` command."""
        from jaato_sdk.events import HelpTextEvent
        help_lines = [
            ("Session Command", "bold"),
            ("", ""),
            ("Manage multiple conversation sessions. Each session has its own", ""),
            ("conversation history, model state, and workspace.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    session [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list              List all available sessions", "dim"),
            ("                      Shows ID, description, model, and status", "dim"),
            ("", ""),
            ("    new [name]        Create a new session", "dim"),
            ("                      Optional name for easier identification", "dim"),
            ("", ""),
            ("    attach <id>       Attach to an existing session", "dim"),
            ("                      Loads session from disk if not in memory", "dim"),
            ("", ""),
            ("    delete <id>       Delete a session permanently", "dim"),
            ("                      Removes both memory and disk state", "dim"),
            ("", ""),
            ("    save [id]         Flush a live session's state to disk", "dim"),
            ("                      Defaults to the attached session.", "dim"),
            ("", ""),
            ("    orphans           List LOADED sessions with no client", "dim"),
            ("                      attached — nothing is consuming their", "dim"),
            ("                      results.  Shows how long each has been", "dim"),
            ("                      orphaned, its wall-clock bounds, and the", "dim"),
            ("                      runner pid executing it.", "dim"),
            ("", ""),
            ("    stop <id>         Stop ANY loaded session by id", "dim"),
            ("                      Cancels it mid-turn if it is running.", "dim"),
            ("                      Unlike 'end' (your own session) this", "dim"),
            ("                      takes an id, so an operator can stop a", "dim"),
            ("                      session whose client is gone.", "dim"),
            ("", ""),
            ("    send <cid> <name> <message>", "dim"),
            ("                      Nudge a NAMED session in a cascade directly.", "dim"),
            ("                      Reaches a LOADED session only — use wake to", "dim"),
            ("                      revive a resting one.", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    session list               List all sessions", "dim"),
            ("    session new                Create unnamed session", "dim"),
            ("    session new myproject      Create session named 'myproject'", "dim"),
            ("    session attach 20251207    Attach to session by ID", "dim"),
            ("    session delete 20251207    Delete session by ID", "dim"),
            ("    session orphans            List sessions nobody is watching", "dim"),
            ("    session stop 20251207      Stop that session, whoever made it", "dim"),
            ("", ""),
            ("SESSION STATES", "bold"),
            ("    Sessions can be in different states:", ""),
            ("    - Loaded: Currently in memory, ready for use", "dim"),
            ("    - On disk: Saved to disk, will be loaded on attach", "dim"),
            ("    - Processing: Currently running a model turn", "dim"),
            ("", ""),
            ("PERSISTENCE", "bold"),
            ("    Sessions are automatically saved to:", ""),
            ("        .jaato/sessions/<session_id>.json", "dim"),
            ("", ""),
            ("    Each session stores:", ""),
            ("    - Conversation history", "dim"),
            ("    - Model and provider settings", "dim"),
            ("    - Workspace path", "dim"),
            ("    - Session description (auto-generated)", "dim"),
            ("", ""),
            ("RELATED COMMANDS", "bold"),
            ("    save              Manually save current session", "dim"),
            ("    resume <id>       Resume a saved session (alias for attach)", "dim"),
            ("    reset             Clear current session history", "dim"),
        ]
        self._event_sink.send_event(client_id, HelpTextEvent(lines=help_lines))

    # ------------------------------------------------------------------
    # Workspace snapshot
    # ------------------------------------------------------------------

    def _handle_snapshot_workspace(
        self,
        client_id: str,
        args: list,
        requester_workspace: Optional[str],
    ) -> None:
        """Handle ``session.snapshot_workspace`` command.

        Creates a read-only copy of a target session's workspace inside
        the requesting session's workspace.  Runs outside any session's
        AppArmor confinement (daemon-level command), so it can read the
        target workspace even though the requester's confined tools
        cannot.

        Args:
            client_id: The requesting client.
            args: ``[target_session_id]`` — the session whose workspace
                to snapshot.  Destination defaults to
                ``<requester_workspace>/.jaato/replay/<uuid>/``.
            requester_workspace: The requesting client's workspace path
                (for computing the destination).
        """
        from jaato_sdk.events import SystemMessageEvent

        if not args:
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message="Usage: session.snapshot_workspace <target_session_id>",
                style="warning",
            ))
            return

        target_session_id = args[0]
        if not requester_workspace:
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message="Cannot snapshot: requester has no workspace.",
                style="error",
            ))
            return

        try:
            result = self._session_manager.snapshot_workspace(
                target_session_id, requester_workspace,
            )
        except (ValueError, TimeoutError, OSError) as exc:
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message=f"Snapshot failed: {exc}",
                style="error",
            ))
            return

        self._event_sink.send_event(client_id, SystemMessageEvent(
            message=json.dumps(result),
            style="info",
        ))

    # ------------------------------------------------------------------
    # Tools commands
    # ------------------------------------------------------------------

    def _handle_tools_command(self, client_id: str, cmd: str, args: list) -> None:
        """Handle ``tools.*`` commands."""
        session = self._session_manager.get_client_session(client_id)
        if not session or not session.server:
            from jaato_sdk.events import SystemMessageEvent
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message="No active session. Use 'session attach' first.",
                style="warning",
            ))
            return

        tools_subcmd = cmd.split(".", 1)[1] if "." in cmd else "list"
        from jaato_sdk.events import ToolStatusEvent

        if tools_subcmd == "list":
            tools = self._get_tool_status(session.server)
            self._event_sink.send_event(client_id, ToolStatusEvent(tools=tools))
        elif tools_subcmd == "enable" and args:
            result = self._tools_enable(session.server, args[0])
            tools = self._get_tool_status(session.server)
            self._event_sink.send_event(client_id, ToolStatusEvent(tools=tools, message=result))
        elif tools_subcmd == "disable" and args:
            result = self._tools_disable(session.server, args[0])
            tools = self._get_tool_status(session.server)
            self._event_sink.send_event(client_id, ToolStatusEvent(tools=tools, message=result))
        elif tools_subcmd == "help":
            from jaato_sdk.events import HelpTextEvent
            help_lines = [
                ("Tools Command", "bold"),
                ("", ""),
                ("Manage tools available to the model. Tools can be enabled or disabled", ""),
                ("to control what capabilities the model has access to.", ""),
                ("", ""),
                ("USAGE", "bold"),
                ("    tools [subcommand] [args]", ""),
                ("", ""),
                ("SUBCOMMANDS", "bold"),
                ("    list              List all tools with their enabled/disabled status", "dim"),
                ("                      (this is the default when no subcommand is given)", "dim"),
                ("", ""),
                ("    enable <name>     Enable a specific tool by name", "dim"),
                ("    enable all        Enable all tools at once", "dim"),
                ("", ""),
                ("    disable <name>    Disable a specific tool by name", "dim"),
                ("    disable all       Disable all tools at once", "dim"),
                ("", ""),
                ("    help              Show this help message", "dim"),
                ("", ""),
                ("EXAMPLES", "bold"),
                ("    tools                    Show all tools and their status", "dim"),
                ("    tools list               Same as above", "dim"),
                ("    tools enable Bash        Enable the Bash tool", "dim"),
                ("    tools disable web_search Disable web search", "dim"),
                ("    tools enable all         Enable all tools", "dim"),
                ("", ""),
                ("NOTES", "bold"),
                ("    - Tool names are case-sensitive", "dim"),
                ("    - Disabled tools will not be available for the model to use", "dim"),
                ("    - Use 'tools list' to see available tool names", "dim"),
            ]
            self._event_sink.send_event(client_id, HelpTextEvent(lines=help_lines))
        else:
            from jaato_sdk.events import SystemMessageEvent
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message="Usage: tools list | tools enable <name> | tools disable <name> | tools help",
                style="dim",
            ))

    # ------------------------------------------------------------------
    # Workspace mismatch
    # ------------------------------------------------------------------

    def _handle_workspace_mismatch_response(self, client_id: str, event) -> None:
        """Handle ``WorkspaceMismatchResponseRequest``."""
        from jaato_sdk.events import WorkspaceMismatchResolvedEvent, SystemMessageEvent

        pending = self._pending_workspace_mismatch.pop(client_id, None)
        if not pending or pending["request_id"] != event.request_id:
            logger.warning(f"No pending workspace mismatch request for client {client_id}")
            return

        response = event.response.lower()
        target_session_id = pending["session_id"]
        session_workspace = pending["session_workspace"]

        if response in ("s", "switch"):
            # User chose to switch to session's workspace
            if self._session_manager.attach_session(client_id, target_session_id):
                self._event_sink.set_client_session(client_id, target_session_id)
                self._event_sink.send_event(client_id, WorkspaceMismatchResolvedEvent(
                    request_id=event.request_id,
                    session_id=target_session_id,
                    action="switch",
                ))
                self._event_sink.send_event(client_id, SystemMessageEvent(
                    message=f"Attached to session. Working directory: {session_workspace}",
                    style="info",
                ))
        else:
            # Cancel or unknown response
            self._event_sink.send_event(client_id, WorkspaceMismatchResolvedEvent(
                request_id=event.request_id,
                session_id=target_session_id,
                action="cancel",
            ))
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message="Attach cancelled.",
                style="dim",
            ))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _handle_history_request(self, client_id: str, event) -> None:
        """Handle ``HistoryRequest``."""
        from jaato_sdk.events import HistoryEvent

        session = self._session_manager.get_client_session(client_id)
        if session and session.server:
            history = session.server.get_history(event.agent_id)
            turn_accounting = session.server.get_turn_accounting(event.agent_id)

            history_data = []
            for msg in history:
                msg_data = {
                    "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                    "parts": [self._serialize_part(p) for p in (msg.parts or [])],
                }
                if getattr(msg, 'model', None) is not None:
                    msg_data['model'] = msg.model
                if getattr(msg, 'provider', None) is not None:
                    msg_data['provider'] = msg.provider
                history_data.append(msg_data)

            self._event_sink.send_event(client_id, HistoryEvent(
                agent_id=event.agent_id,
                history=history_data,
                turn_accounting=turn_accounting or [],
            ))

    # ------------------------------------------------------------------
    # Daemon plugin commands
    # ------------------------------------------------------------------

        else:
            # ANSWER, OR SAY WHY.  This guard used to fall through to the
            # end of the method and emit NOTHING, so a client could not tell
            # "no history" from "not your session" and simply waited out its
            # own timeout.  Absent and empty, collapsed on the wire.
            #
            # It is reachable in normal operation, not just on misuse: the
            # cascade policy detaches a cid-stamped session's clients when it
            # terminates (to release its slot), so a driver asking for the
            # ledger of the arm that just finished arrives AFTER the
            # detach and finds no session of its own.
            #
            # ``recoverable=True`` because the connection is fine -- this is
            # an answer about one request, not a transport failure.
            from jaato_sdk.events import ErrorEvent
            self._event_sink.send_event(client_id, ErrorEvent(
                error=(
                    "history is unavailable: this connection has no session "
                    "attached. A cascade session is detached from its "
                    "creator when it terminates, so its history must be "
                    "fetched before termination or read from the persisted "
                    "session record."
                ),
                error_type="NoAttachedSession",
                recoverable=True,
                request_id=getattr(event, "request_id", None),
            ))

    def _find_daemon_plugin_for_command(self, command: str):
        """Find a daemon-level plugin that provides a user command.

        Args:
            command: The command name to find.

        Returns:
            The plugin instance or None.
        """
        for plugin in self._daemon_plugins.values():
            if hasattr(plugin, 'get_user_commands'):
                for cmd in plugin.get_user_commands():
                    if cmd.name == command:
                        return plugin
        return None

    def _execute_daemon_command(
        self,
        client_id: str,
        plugin,
        command: str,
        args: list,
    ) -> None:
        """Execute a user command on a daemon-level plugin (no session required).

        Sets up output callback to route plugin output to the client via events,
        parses arguments, and handles HelpLines results.

        Args:
            client_id: The requesting client.
            plugin: The daemon-level plugin instance.
            command: The command name.
            args: Raw argument list from the client.
        """
        from jaato_sdk.events import HelpTextEvent, SystemMessageEvent
        from jaato_sdk.plugins.base import parse_command_args, HelpLines

        # Inject the client's workspace path into the plugin so credential
        # storage functions resolve to the session workspace, not cwd.
        workspace = self._event_sink.get_client_workspace(client_id)
        if workspace and hasattr(plugin, '_workspace_path'):
            plugin._workspace_path = workspace

        # Buffer plugin._emit() output — daemon commands run outside any agent
        # context, so we accumulate output and send as a SystemMessageEvent.
        output_parts = []
        if hasattr(plugin, 'set_output_callback'):
            def output_callback(source: str, text: str, mode: str) -> None:
                output_parts.append(text)
            plugin.set_output_callback(output_callback)

        try:
            # Find the UserCommand definition for arg parsing
            cmd_def = None
            for cmd in plugin.get_user_commands():
                if cmd.name == command:
                    cmd_def = cmd
                    break

            parsed_args = parse_command_args(cmd_def, ' '.join(args)) if cmd_def else {}
            result = plugin.execute_user_command(command, parsed_args)

            # Send accumulated _emit() output as a single system message
            if output_parts:
                combined = "".join(output_parts).rstrip("\n")
                if combined:
                    self._event_sink.send_event(client_id, SystemMessageEvent(
                        message=combined,
                        style="info",
                    ))

            if isinstance(result, HelpLines):
                self._event_sink.send_event(client_id, HelpTextEvent(lines=result.lines))
            elif isinstance(result, str) and result:
                self._event_sink.send_event(client_id, SystemMessageEvent(
                    message=result,
                    style="info",
                ))

            # After auth command execution, check if credentials are now valid
            # and offer to set up a session with the provider.
            if hasattr(plugin, 'verify_credentials') and plugin.verify_credentials():
                # A live session on this provider resolved its credential at
                # bootstrap and would keep the stale one; refresh it first.
                self._maybe_reload_live_session_after_auth(client_id, plugin, args)
                self._offer_post_auth_setup(client_id, plugin)

        except Exception as e:
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message=f"Command error: {e}",
                style="error",
            ))

        finally:
            if hasattr(plugin, 'set_output_callback'):
                plugin.set_output_callback(None)

    def _resolve_instructions_value(
        self,
        raw: str,
        workspace_path: Optional[str],
        client_id: str,
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a ``--instructions`` value into the literal text the session sees.

        Two forms accepted:

        - **Literal text** — e.g. ``--instructions "You are terse."``.
          The value is returned verbatim.
        - **File reference** — e.g. ``--instructions @prompts/min.md``.
          A leading ``@`` is stripped and the rest is read from disk.
          Relative paths resolve against ``workspace_path``; absolute
          paths are honoured as-is.  ``~`` expands.

        Args:
            raw: The flag's value, literal text or ``@path``.
            workspace_path: Base for a relative ``@path``.
            client_id: Who to report a failure to.
            request_id: Correlation id of the ``session.new`` this flag
                belongs to.  STAMPED ON THE REFUSAL: the client's
                create-wait accepts only a correlated ``ErrorEvent``, so
                an unstamped one is discarded and the caller waits out its
                full timeout instead of learning the path was unreadable
                (#882).

        Returns the resolved text, or ``None`` if the file reference
        could not be read (a correlated ``ErrorEvent`` has been emitted to
        the client in that case).
        """
        from jaato_sdk.events import ErrorEvent

        if not raw.startswith("@"):
            return raw

        path_str = raw[1:]
        if not path_str:
            self._event_sink.send_event(client_id, ErrorEvent(
                error="--instructions @ requires a path after the @",
                error_type="UsageError",
                recoverable=True,
                request_id=request_id,
            ))
            return None

        path = pathlib.Path(path_str).expanduser()
        if not path.is_absolute() and workspace_path:
            path = pathlib.Path(workspace_path) / path
        try:
            return path.read_text(encoding="utf-8").rstrip("\n")
        except (OSError, UnicodeDecodeError) as exc:
            self._event_sink.send_event(client_id, ErrorEvent(
                error=f"--instructions @{path_str}: {exc}",
                error_type="UsageError",
                recoverable=True,
                request_id=request_id,
            ))
            return None

    def _hint_available_auth_providers(self, client_id: str) -> None:
        """Send a hint listing available auth providers after session creation fails.

        Iterates daemon-level plugins with the ``TRAIT_AUTH_PROVIDER`` trait and
        emits a message showing their login commands.
        """
        from jaato_sdk.plugins.base import TRAIT_AUTH_PROVIDER

        hints: list[str] = []
        for plugin in self._daemon_plugins.values():
            traits = getattr(plugin, 'plugin_traits', frozenset())
            if TRAIT_AUTH_PROVIDER not in traits:
                continue
            display_name = getattr(plugin, 'provider_display_name', plugin.name)
            commands = plugin.get_user_commands() if hasattr(plugin, 'get_user_commands') else []
            cmd_name = commands[0].name if commands else plugin.name
            hints.append(f"  {cmd_name} login  — {display_name}")

        if hints:
            from jaato_sdk.events import SystemMessageEvent
            msg = "Available providers:\n" + "\n".join(hints)
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message=msg,
                style="dim",
            ))

    def _offer_post_auth_setup(self, client_id: str, plugin) -> None:
        """Emit PostAuthSetupEvent to offer session creation after auth success."""
        from jaato_sdk.events import PostAuthSetupEvent

        provider_name = getattr(plugin, 'provider_name', '')
        if not provider_name:
            return

        # Check if client already has an active session
        has_active_session = False
        current_provider = ""
        current_model = ""
        session = self._session_manager.get_client_session(client_id)
        if session and session.server:
            has_active_session = True
            current_provider = getattr(session.server, '_provider_name', '') or ""
            current_model = getattr(session.server, '_model_name', '') or ""

        workspace_path = self._event_sink.get_client_workspace(client_id) or ""

        models = []
        if hasattr(plugin, 'get_default_models'):
            models = plugin.get_default_models()

        request_id = str(uuid.uuid4())
        self._pending_post_auth[client_id] = {
            "request_id": request_id,
            "provider_name": provider_name,
            "credential_env_vars": getattr(plugin, 'credential_env_vars', []),
        }

        self._event_sink.send_event(client_id, PostAuthSetupEvent(
            request_id=request_id,
            provider_name=provider_name,
            provider_display_name=getattr(plugin, 'provider_display_name', provider_name),
            available_models=models,
            has_active_session=has_active_session,
            current_provider=current_provider,
            current_model=current_model,
            workspace_path=workspace_path,
        ))

    def _handle_post_auth_response(self, client_id: str, event) -> None:
        """Handle PostAuthSetupResponse from client.

        Creates/reconfigures session and optionally writes .env file.
        """
        from jaato_sdk.events import SystemMessageEvent

        pending = self._pending_post_auth.pop(client_id, None)
        if not pending or pending["request_id"] != event.request_id:
            logger.warning(f"No pending post-auth request for client {client_id}")
            return

        if not event.connect:
            return

        provider_name = pending["provider_name"]
        model_name = event.model_name

        if not model_name:
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message="No model selected, skipping session setup.",
                style="dim",
            ))
            return

        # Strip provider prefix from model name if present (e.g., "zhipuai/glm-4.7" -> "glm-4.7")
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]

        workspace_path = self._event_sink.get_client_workspace(client_id)

        # Persist to .env if requested
        if event.persist_env and workspace_path:
            credential_env_vars = pending.get("credential_env_vars", [])
            self._persist_env(workspace_path, provider_name, model_name, credential_env_vars)
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message=f"Saved JAATO_PROVIDER={provider_name} and MODEL_NAME={model_name} to .env",
                style="info",
            ))

        # Create a new session with the authenticated provider.
        session_id = self._session_manager.create_session(
            client_id, None, workspace_path=workspace_path,
            env_overrides={
                "JAATO_PROVIDER": provider_name,
                "MODEL_NAME": model_name,
            },
            created_by=self._event_sink.get_client_user(client_id),
        )
        if session_id:
            set_logging_context(
                session_id=session_id,
                client_id=client_id,
                workspace_path=workspace_path,
            )
            self._event_sink.set_client_session(client_id, session_id)

            self._event_sink.send_event(client_id, SystemMessageEvent(
                message=f"Session created with {provider_name} / {model_name}",
                style="success",
            ))
        else:
            self._event_sink.send_event(client_id, SystemMessageEvent(
                message="Failed to create session.",
                style="error",
            ))

    # ------------------------------------------------------------------
    # Command list (for client autocomplete)
    # ------------------------------------------------------------------

    def get_command_list(self) -> list:
        """Get list of available commands for clients.

        Returns:
            List of {name, description} dicts.
        """
        commands = []

        # Static session management commands (handled by daemon)
        session_commands = [
            {"name": "session list", "description": "List all sessions"},
            {"name": "session new", "description": "Create a new session"},
            {"name": "session attach", "description": "Attach to an existing session"},
            {"name": "session wake", "description": "Wake a session by id (revive if cold) and start a turn"},
            {"name": "session bind_wake", "description": "Declare a wake binding (wake_ref + trust keys) for this session"},
            {"name": "session unbind_wake", "description": "Remove a wake binding for this session"},
            {"name": "session delete", "description": "Delete a session"},
            {"name": "session reload_env", "description": "Re-read this session's workspace .env and credentials and rebuild its provider"},
            {"name": "session help", "description": "Show detailed help for session command"},
        ]
        commands.extend(session_commands)

        # Static tools commands (handled by daemon)
        tools_commands = [
            {"name": "tools list", "description": "List all tools with status"},
            {"name": "tools enable", "description": "Enable a tool (or 'all')"},
            {"name": "tools disable", "description": "Disable a tool (or 'all')"},
            {"name": "tools help", "description": "Show detailed help for tools command"},
        ]
        commands.extend(tools_commands)

        # Session-independent plugin commands (auth plugins).
        for plugin in self._daemon_plugins.values():
            if hasattr(plugin, 'get_user_commands'):
                for cmd in plugin.get_user_commands():
                    if hasattr(plugin, 'get_command_completions'):
                        subcommands = plugin.get_command_completions(cmd.name, [])
                        if subcommands:
                            for sub in subcommands:
                                commands.append({
                                    "name": f"{cmd.name} {sub.value}",
                                    "description": sub.description or "",
                                })
                        else:
                            commands.append({
                                "name": cmd.name,
                                "description": cmd.description or "",
                            })
                    else:
                        commands.append({
                            "name": cmd.name,
                            "description": cmd.description or "",
                        })

        # Get commands from any active session
        if self._session_manager:
            sessions = self._session_manager.list_sessions()
            for session_info in sessions:
                if session_info.is_loaded:
                    session = self._session_manager.get_session(session_info.session_id)
                    if session and session.server:
                        # Get commands from server (with model subcommand expansion)
                        server_cmds = session.server.get_available_commands()
                        for name, description in server_cmds.items():
                            # Phase 3 §7c step 6.6.4.5c.4: route through
                            # runner-RPC.  Pivots the gate from ``_jaato``
                            # to ``_runner_rpc`` and reconstructs
                            # CommandCompletion NamedTuples daemon-side
                            # (wrapper preserves the ``.value`` /
                            # ``.description`` attr-access pattern).
                            if name == "model" and getattr(
                                session.server, '_runner_rpc', None,
                            ) is not None:
                                try:
                                    model_subs = (
                                        session.server._runner_rpc
                                        .session_get_model_completions_threadsafe([])
                                    )
                                except Exception:
                                    model_subs = []
                                if model_subs:
                                    for sub in model_subs:
                                        commands.append({
                                            "name": f"model {sub.value}",
                                            "description": sub.description or "",
                                        })
                                else:
                                    commands.append({"name": name, "description": description or ""})
                            else:
                                commands.append({
                                    "name": name,
                                    "description": description or "",
                                })

                        # Get commands from registry plugins
                        if session.server.registry:
                            for plugin_name in session.server.registry.list_exposed():
                                plugin = session.server.registry.get_plugin(plugin_name)
                                if plugin and hasattr(plugin, 'get_user_commands'):
                                    for cmd in plugin.get_user_commands():
                                        if hasattr(plugin, 'get_command_completions'):
                                            subcommands = plugin.get_command_completions(cmd.name, [])
                                            if subcommands:
                                                has_dynamic_completions = (
                                                    hasattr(plugin, 'get_memory_metadata')
                                                    or hasattr(plugin, 'get_service_metadata')
                                                )
                                                for sub in subcommands:
                                                    commands.append({
                                                        "name": f"{cmd.name} {sub.value}",
                                                        "description": sub.description or "",
                                                    })
                                                    if not has_dynamic_completions:
                                                        sub_completions = plugin.get_command_completions(
                                                            cmd.name, [sub.value, ""]
                                                        )
                                                        for sub2 in sub_completions:
                                                            commands.append({
                                                                "name": f"{cmd.name} {sub.value} {sub2.value}",
                                                                "description": sub2.description or "",
                                                            })
                                            else:
                                                commands.append({
                                                    "name": cmd.name,
                                                    "description": cmd.description or "",
                                                })
                                        else:
                                            commands.append({
                                                "name": cmd.name,
                                                "description": cmd.description or "",
                                            })

                        # Get commands from permission plugin
                        if session.server.permission_plugin:
                            perm = session.server.permission_plugin
                            if hasattr(perm, 'get_user_commands'):
                                for cmd in perm.get_user_commands():
                                    if hasattr(perm, 'get_command_completions'):
                                        subcommands = perm.get_command_completions(cmd.name, [])
                                        if subcommands:
                                            for sub in subcommands:
                                                commands.append({
                                                    "name": f"{cmd.name} {sub.value}",
                                                    "description": sub.description or "",
                                                })
                                                sub_completions = perm.get_command_completions(
                                                    cmd.name, [sub.value, ""]
                                                )
                                                for sub2 in sub_completions:
                                                    commands.append({
                                                        "name": f"{cmd.name} {sub.value} {sub2.value}",
                                                        "description": sub2.description or "",
                                                    })
                                        else:
                                            commands.append({
                                                "name": cmd.name,
                                                "description": cmd.description or "",
                                            })
                                    else:
                                        commands.append({
                                            "name": cmd.name,
                                            "description": cmd.description or "",
                                        })

                        # Got commands from one session, that's enough
                        break

        # Deduplicate by name
        seen = set()
        unique_commands = []
        for cmd in commands:
            if cmd["name"] not in seen:
                seen.add(cmd["name"])
                unique_commands.append(cmd)

        return unique_commands

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_status(server) -> list:
        """Get tool status as structured data.

        Args:
            server: JaatoServer instance.

        Returns:
            List of tool status dicts: {name, description, enabled, plugin}
        """
        tool_status = []

        if server.registry:
            tool_status.extend(server.registry.get_tool_status())

        if server.permission_plugin:
            for schema in server.permission_plugin.get_tool_schemas():
                tool_status.append({
                    'name': schema.name,
                    'description': schema.description,
                    'enabled': True,
                    'plugin': 'permission',
                })

        return tool_status

    @staticmethod
    def _tools_enable(server, tool_name: str) -> str:
        """Enable a tool.

        Args:
            server: JaatoServer instance.
            tool_name: Tool name or 'all'.

        Returns:
            Result message.
        """
        if not server.registry:
            return "No registry available."

        if tool_name.lower() == "all":
            count = 0
            for status in server.registry.get_tool_status():
                if not status.get('enabled', True):
                    server.registry.enable_tool(status['name'])
                    count += 1
            return f"Enabled {count} tools."

        if server.registry.enable_tool(tool_name):
            return f"Enabled tool: {tool_name}"
        return f"Tool not found or already enabled: {tool_name}"

    @staticmethod
    def _tools_disable(server, tool_name: str) -> str:
        """Disable a tool.

        Args:
            server: JaatoServer instance.
            tool_name: Tool name or 'all'.

        Returns:
            Result message.
        """
        if not server.registry:
            return "No registry available."

        if tool_name.lower() == "all":
            count = 0
            for status in server.registry.get_tool_status():
                if status.get('enabled', True):
                    server.registry.disable_tool(status['name'])
                    count += 1
            return f"Disabled {count} tools."

        if server.registry.disable_tool(tool_name):
            return f"Disabled tool: {tool_name}"
        return f"Tool not found or already disabled: {tool_name}"

    @staticmethod
    def _serialize_part(part) -> dict:
        """Serialize a message part to a dict.

        Args:
            part: Message Part object.

        Returns:
            Dict with part data.
        """
        if hasattr(part, 'text') and part.text is not None:
            return {"type": "text", "text": part.text}
        elif hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            return {
                "type": "function_call",
                "name": fc.name if hasattr(fc, 'name') else str(fc),
                "args": fc.args if hasattr(fc, 'args') else {},
                # THE IDENTIFIER IS ``fc.id`` HERE AND ``fr.call_id`` BELOW.
                #
                # Emitted under the response branch's key so the two Parts
                # can be paired on the wire, which is what
                # ``build_tool_call_ledger`` does in-process
                # (``getattr(fc, "id")`` against ``getattr(fr, "call_id")``).
                # Without it the call side carried NO identifier at all, so a
                # client reading ``request_history`` could not rebuild the
                # ledger that completion processors receive as
                # ``context.tool_calls``.
                #
                # Writing ``getattr(fc, "call_id", "")`` here -- mirroring the
                # branch below, which is the obvious thing to write -- would
                # emit the empty string FOREVER: ``FunctionCall`` has no such
                # field.  That is the same failure as the repr bug this
                # function already carries a comment about, in the same
                # function: a wrong attribute name and an absent one are
                # indistinguishable to ``getattr``, and the fallback produces
                # something that looks like a value.
                "call_id": getattr(fc, 'id', ''),
            }
        elif hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            # ``ToolResult`` has ``result``, never ``response`` — so the
            # old ``hasattr(fr, 'response')`` was ALWAYS False and every
            # tool response in request_history was sent as ``str(fr)``:
            # the dataclass REPR, not data.  A client could not read a
            # tool result structurally at all; it had to parse a Python
            # repr to recover ``is_error`` or the result dict, and a large
            # result was stringified whole into the history payload.
            #
            # The hasattr guard is what hid it: a wrong attribute name and
            # an absent one are indistinguishable to ``hasattr``, and the
            # fallback produced something that LOOKED like a value.
            return {
                "type": "function_response",
                "name": getattr(fr, 'name', ''),
                "call_id": getattr(fr, 'call_id', ''),
                "response": fr.result,
                "is_error": getattr(fr, 'is_error', False),
                # The untrusted-content boundary, readable WITHOUT parsing
                # a repr.  A client deciding how to display or re-feed a
                # tool result needs to know the text is attacker-authored.
                "untrusted": getattr(fr, 'untrusted', False),
                "untrusted_source": getattr(fr, 'untrusted_source', None),
            }
        else:
            return {"type": "unknown", "data": str(part)}

    @staticmethod
    def _persist_env(
        workspace_path: str,
        provider_name: str,
        model_name: str,
        credential_env_vars: Optional[List[str]] = None,
    ) -> None:
        """Write or update JAATO_PROVIDER and MODEL_NAME in workspace .env file.

        Only replaces active (uncommented) lines. Commented-out lines like
        ``#JAATO_PROVIDER=...`` are preserved untouched.

        When *credential_env_vars* is provided (from the auth plugin), any
        commented-out lines for those vars are annotated to indicate that
        credentials are stored securely in ``.jaato/`` and managed by the
        auth command.
        """
        env_path = os.path.join(workspace_path, '.env')
        lines = []
        seen_provider = False
        seen_model = False
        cred_vars = set(credential_env_vars or [])

        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('JAATO_PROVIDER='):
                        lines.append(f'JAATO_PROVIDER={provider_name}\n')
                        seen_provider = True
                    elif stripped.startswith('MODEL_NAME='):
                        lines.append(f'MODEL_NAME={model_name}\n')
                        seen_model = True
                    elif cred_vars and _is_commented_credential(stripped, cred_vars):
                        var_name = _extract_var_name(stripped)
                        lines.append(f'# {var_name}=<stored in .jaato/ — use {provider_name}-auth>\n')
                        cred_vars.discard(var_name)
                    else:
                        lines.append(line)

        if not seen_provider:
            lines.append(f'JAATO_PROVIDER={provider_name}\n')
        if not seen_model:
            lines.append(f'MODEL_NAME={model_name}\n')

        with open(env_path, 'w') as f:
            f.writelines(lines)


# Module-level helpers (moved from JaatoDaemon static methods)

def _is_commented_credential(stripped: str, cred_vars: set) -> bool:
    """Check if a stripped line is a commented-out credential env var."""
    if not stripped.startswith('#'):
        return False
    uncommented = stripped.lstrip('#').lstrip()
    return any(uncommented.startswith(f'{var}=') for var in cred_vars)


def _extract_var_name(stripped: str) -> str:
    """Extract the env var name from a commented line like '# ZHIPUAI_API_KEY=...'."""
    uncommented = stripped.lstrip('#').lstrip()
    return uncommented.split('=', 1)[0]
