"""Transport-agnostic command dispatcher for the Jaato daemon.

Extracted from ``JaatoDaemon._handle_session_request_inner()`` so that
both IPC and WebSocket transports route through the same dispatch logic.

The router owns no transport state — it receives events and emits
responses through the ``EventSink`` protocol.
"""

import json
import logging
import os
import pathlib
import uuid
from typing import Any, Dict, List, Optional

from jaato_sdk.events import Event
from server.event_sink import EventSink
from server.session_manager import SessionManager
from server.session_logging import set_logging_context, clear_logging_context
from shared.session_id import is_safe_session_id

logger = logging.getLogger(__name__)


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
                workspace_path = event.args[0] if event.args else None
                if workspace_path:
                    self._event_sink.set_client_workspace(client_id, workspace_path)
                    logger.debug(f"Client {client_id} workspace set to: {workspace_path}")
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

            elif cmd == "session.help":
                self._handle_session_help(client_id)
                return

            elif cmd == "session.snapshot_workspace":
                self._handle_snapshot_workspace(client_id, event.args, workspace_path)
                return

            elif cmd == "session.save":
                self._handle_session_save(client_id, session_id, event.args)
                return

            elif cmd == "session.send":
                self._handle_session_send(client_id, event.args, event.payload)
                return

            elif cmd == "session.wake":
                self._handle_session_wake(client_id, event.args, event.payload)
                return

            elif cmd == "session.bind_wake":
                self._handle_session_bind_wake(client_id, event.args, event.payload)
                return

            elif cmd == "session.unbind_wake":
                self._handle_session_unbind_wake(client_id, event.args, event.payload)
                return

            elif cmd == "cascade.register":
                self._handle_cascade_register(client_id, event.args)
                return

            elif cmd == "cascade.unregister":
                self._handle_cascade_unregister(client_id, event.args)
                return

            elif cmd == "cascade.budget.set":
                self._handle_cascade_budget_set(client_id, event.args, event.payload)
                return

            elif cmd == "cascade.budget.get":
                self._handle_cascade_budget_get(client_id, event.args)
                return

            elif cmd == "cascade.budget.clear":
                self._handle_cascade_budget_clear(client_id, event.args)
                return

            elif cmd == "cascade.cancel":
                self._handle_cascade_cancel(client_id, event.args)
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

        # Route to session
        self._session_manager.handle_request(client_id, session_id, event)

    # ------------------------------------------------------------------
    # Session commands
    # ------------------------------------------------------------------

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
                    ))
                    return
                system_instruction_override = self._resolve_instructions_value(
                    raw, workspace_path, client_id,
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
            request_id=(payload or {}).get("request_id"),
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

    def _handle_session_wake(
        self, client_id: str, args: list, payload: Optional[dict],
    ) -> None:
        """Handle ``session.wake`` — start a USER turn on a session, reviving it
        if cold, for the client-agnostic wake primitive.

        Accepts a structured ``payload`` (SDK callers) —
        ``{session_id, text, source?, event_id?}`` — or positional ``args``
        ``[session_id, text, source?, event_id?]``.  Authentication is the
        transport's boundary (IPC socket-mode / WS bearer token / the HTTP
        shim's #498 fail-closed check); this handler runs only for callers
        already past that gate.  On refusal it emits an ``ErrorEvent`` with the
        reason; on success the woken turn's output flows to the session's
        attached clients (the caller need not be one).
        """
        from jaato_sdk.events import ErrorEvent
        p = payload or {}
        session_id = p.get("session_id") or (args[0] if len(args) > 0 else None)
        text = p.get("text") or (args[1] if len(args) > 1 else None)
        source = p.get("source") or (args[2] if len(args) > 2 else "user")
        event_id = p.get("event_id") or (args[3] if len(args) > 3 else None)
        if not session_id or not text:
            self._event_sink.send_event(client_id, ErrorEvent(
                error="session.wake requires session_id and text",
                error_type="UsageError",
                recoverable=True,
            ))
            return
        outcome, detail = self._session_manager.wake_session(
            session_id, text, source=source, event_id=event_id,
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

    def _handle_session_list(self, client_id: str, session_id: str) -> None:
        """Handle ``session.list`` command."""
        sessions = self._session_manager.list_sessions()
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
        } for s in sessions]

        self._event_sink.send_event(client_id, SessionListEvent(sessions=session_data))

    def _handle_session_default(
        self, client_id: str, workspace_path: Optional[str],
    ) -> None:
        """Handle ``session.default`` command."""
        default_session_id = self._session_manager.get_or_create_default(
            client_id, workspace_path=workspace_path
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
        from shared.budget_control import BudgetControlConfig

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
    ) -> Optional[str]:
        """Resolve a ``--instructions`` value into the literal text the session sees.

        Two forms accepted:

        - **Literal text** — e.g. ``--instructions "You are terse."``.
          The value is returned verbatim.
        - **File reference** — e.g. ``--instructions @prompts/min.md``.
          A leading ``@`` is stripped and the rest is read from disk.
          Relative paths resolve against ``workspace_path``; absolute
          paths are honoured as-is.  ``~`` expands.

        Returns the resolved text, or ``None`` if the file reference
        could not be read (an ``ErrorEvent`` has been emitted to the
        client in that case).
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
