"""Transport-agnostic command dispatcher for the Jaato daemon.

Extracted from ``JaatoDaemon._handle_session_request_inner()`` so that
both IPC and WebSocket transports route through the same dispatch logic.

The router owns no transport state — it receives events and emits
responses through the ``EventSink`` protocol.
"""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from jaato_sdk.events import Event
from server.event_sink import EventSink
from server.session_manager import SessionManager
from server.session_logging import set_logging_context, clear_logging_context

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
                self._handle_session_new(client_id, event.args, workspace_path)
                return

            elif cmd == "session.attach":
                self._handle_session_attach(client_id, session_id, event.args, workspace_path)
                return

            elif cmd == "session.list":
                self._handle_session_list(client_id, session_id)
                return

            elif cmd == "session.profiles":
                from jaato_sdk.events import SessionProfilesEvent
                profiles = self._session_manager.list_profiles(
                    workspace_path=workspace_path,
                )
                self._event_sink.send_event(client_id, SessionProfilesEvent(profiles=profiles))
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
        self, client_id: str, args: list, workspace_path: Optional[str],
    ) -> None:
        """Handle ``session.new`` command."""
        # Parse args: positional name and --profile flag
        name = None
        profile_name = None
        args_iter = iter(args)
        for arg in args_iter:
            if arg == "--profile":
                profile_name = next(args_iter, None)
            elif name is None:
                name = arg

        new_session_id = self._session_manager.create_session(
            client_id, name, workspace_path=workspace_path,
            profile_name=profile_name,
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

    def _handle_session_end(self, client_id: str, session_id: str) -> None:
        """Handle ``session.end`` command."""
        session = self._session_manager.get_client_session(client_id)
        if session and session.server:
            session.server.stop()
        # Signal termination to all clients attached to this session
        from jaato_sdk.events import SystemMessageEvent
        self._session_manager._emit_to_session(
            session_id,
            SystemMessageEvent(message="[SESSION_TERMINATED]", style="system")
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
                            if name == "model" and hasattr(session.server, '_jaato'):
                                jaato = session.server._jaato
                                if jaato and hasattr(jaato, 'get_model_completions'):
                                    model_subs = jaato.get_model_completions([])
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
            }
        elif hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            return {
                "type": "function_response",
                "name": fr.name if hasattr(fr, 'name') else str(fr),
                "response": fr.response if hasattr(fr, 'response') else str(fr),
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
