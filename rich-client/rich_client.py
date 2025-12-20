#!/usr/bin/env python3
"""Rich TUI client with sticky plan display.

This client provides a terminal UI experience with:
- Sticky plan panel at the top showing current plan status
- Scrolling output panel below for model responses and tool output
- Full-screen alternate buffer for immersive experience

Requires an interactive TTY. For non-TTY environments, use simple-client.
"""

import os
import sys
import pathlib
import tempfile
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Add project root to path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add simple-client to path for reusable components
SIMPLE_CLIENT = ROOT / "simple-client"
if str(SIMPLE_CLIENT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_CLIENT))

from dotenv import load_dotenv

from shared import (
    JaatoClient,
    TokenLedger,
    PluginRegistry,
    PermissionPlugin,
    TodoPlugin,
    active_cert_bundle,
)
from shared.plugins.session import create_plugin as create_session_plugin, load_session_config
from shared.plugins.base import parse_command_args

# Reuse input handling from simple-client
from input_handler import InputHandler

# Rich TUI components
from pt_display import PTDisplay
from plan_reporter import create_live_reporter
from agent_registry import AgentRegistry


class RichClient:
    """Rich TUI client with sticky plan display.

    Uses PTDisplay (prompt_toolkit-based) to manage a full-screen layout with:
    - Sticky plan panel at top (hidden when no plan)
    - Scrolling output below
    - Integrated input prompt at bottom

    The plan panel updates in-place as plan steps progress,
    while model output scrolls naturally below.
    """

    def __init__(self, env_file: str = ".env", verbose: bool = True, provider: Optional[str] = None):
        self.verbose = verbose
        self.env_file = env_file
        self._provider = provider  # CLI override for provider
        self._jaato: Optional[JaatoClient] = None
        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Optional[PermissionPlugin] = None
        self.todo_plugin: Optional[TodoPlugin] = None
        self.ledger = TokenLedger()

        # Agent registry for tracking agents and their state
        self._agent_registry = AgentRegistry()

        # Rich TUI display (prompt_toolkit-based)
        self._display: Optional[PTDisplay] = None

        # Input handler (for file expansion, history, completions)
        self._input_handler = InputHandler()

        # Track original inputs for session export
        self._original_inputs: list[dict] = []

        # Flag to signal exit from input loop
        self._should_exit = False

        # Queue for permission/clarification input routing
        import queue
        self._channel_input_queue: queue.Queue[str] = queue.Queue()
        self._waiting_for_channel_input: bool = False
        # Pending response options from permission plugin (single source of truth)
        self._pending_response_options: Optional[list] = None

        # Background model thread tracking
        self._model_thread: Optional[threading.Thread] = None
        self._model_running: bool = False

        # Model info for status bar
        self._model_provider: str = ""
        self._model_name: str = ""

        # UI hooks reference for signaling agent status changes
        self._ui_hooks: Optional[Any] = None

    def log(self, msg: str) -> None:
        """Log message to output panel."""
        if self.verbose and self._display:
            self._display.add_system_message(msg, style="cyan")

    def _create_output_callback(self, stop_spinner_on_first: bool = False,
                                  suppress_sources: Optional[set] = None) -> Callable[[str, str, str], None]:
        """Create callback for real-time output to display.

        Args:
            stop_spinner_on_first: If True, stop the spinner on first output.
            suppress_sources: Set of source names to suppress (e.g., {"permission"})
        """
        first_output_received = [False]  # Use list for mutability in closure
        suppress = suppress_sources or set()

        def callback(source: str, text: str, mode: str) -> None:
            if self._display:
                # Skip suppressed sources (e.g., permission output shown in tool tree)
                if source in suppress:
                    return
                # Stop spinner on first output if requested
                if stop_spinner_on_first and not first_output_received[0]:
                    first_output_received[0] = True
                    self._display.stop_spinner()
                # Skip ALL sources when UI hooks are active - the hooks handle
                # routing all output (model, system, plugin) to the correct buffer
                # via on_agent_output. Without this, output gets duplicated.
                if self._agent_registry:
                    return
                self._display.append_output(source, text, mode)
        return callback

    def _try_execute_plugin_command(self, user_input: str) -> Optional[Any]:
        """Try to execute user input as a plugin-provided command."""
        if not self._jaato:
            return None

        user_commands = self._jaato.get_user_commands()
        if not user_commands:
            return None

        parts = user_input.strip().split(maxsplit=1)
        input_cmd = parts[0].lower() if parts else ""
        raw_args = parts[1] if len(parts) > 1 else ""

        command = None
        for cmd_name, cmd in user_commands.items():
            if input_cmd == cmd_name.lower():
                command = cmd
                break

        if not command:
            return None

        args = parse_command_args(command, raw_args)

        if command.name.lower() == "save":
            args["user_inputs"] = self._original_inputs.copy()

        try:
            result, shared = self._jaato.execute_user_command(command.name, args)
            self._display_command_result(command.name, result, shared)

            # Update status bar if model was changed
            if command.name.lower() == "model" and isinstance(result, dict):
                if result.get("success") and result.get("current_model"):
                    self._model_name = result["current_model"]
                    if self._display:
                        self._display.set_model_info(self._model_provider, self._model_name)

            if command.name.lower() == "resume" and isinstance(result, dict):
                user_inputs = result.get("user_inputs", [])
                if user_inputs:
                    self._restore_user_inputs(user_inputs)

            return result

        except Exception as e:
            if self._display:
                self._display.show_lines([(f"Error: {e}", "red")])
            return {"error": str(e)}

    def _display_command_result(
        self,
        command_name: str,
        result: Any,
        shared: bool
    ) -> None:
        """Display command result in output panel."""
        if not self._display:
            return

        # For plan command, the sticky panel handles display
        if command_name == "plan":
            return

        lines = [(f"[{command_name}]", "bold")]

        if isinstance(result, dict):
            for key, value in result.items():
                if not key.startswith('_'):
                    lines.append((f"  {key}: {value}", "dim"))
        else:
            # Handle multi-line strings by splitting on newlines
            result_str = str(result)
            if '\n' in result_str:
                for line in result_str.split('\n'):
                    lines.append((f"  {line}", "dim"))
            else:
                lines.append((f"  {result_str}", "dim"))

        if shared:
            lines.append(("  [Result shared with model]", "dim cyan"))

        self._display.show_lines(lines)

    def _restore_user_inputs(self, user_inputs: List[str]) -> None:
        """Restore user inputs to prompt history after session resume."""
        self._original_inputs = list(user_inputs)
        count = self._input_handler.restore_history(
            [entry["text"] if isinstance(entry, dict) else entry for entry in user_inputs]
        )
        if count:
            self.log(f"Restored {count} inputs to prompt history")

    def initialize(self) -> bool:
        """Initialize the client."""
        # Load environment from CWD or explicit --env-file path
        load_dotenv(self.env_file)

        # Check CA bundle
        active_bundle = active_cert_bundle(verbose=False)

        # Check required vars - MODEL_NAME always required
        model_name = os.environ.get("MODEL_NAME")
        if not model_name:
            print("Error: Missing required environment variable: MODEL_NAME")
            return False

        # Check auth method: API key (AI Studio) or Vertex AI
        api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("LOCATION")

        if not api_key and (not project_id or not location):
            print("Error: Set GOOGLE_GENAI_API_KEY for AI Studio, or PROJECT_ID and LOCATION for Vertex AI")
            return False

        # Initialize JaatoClient with optional provider override
        try:
            self._jaato = JaatoClient(provider_name=self._provider)
            if api_key:
                # AI Studio mode - just need model
                self._jaato.connect(model=model_name)
            else:
                # Vertex AI mode
                self._jaato.connect(project_id, location, model_name)
        except Exception as e:
            print(f"Error: Failed to connect: {e}")
            return False

        # Store model info for status bar (from jaato client)
        self._model_name = self._jaato.model_name or model_name
        self._model_provider = self._jaato.provider_name

        # Initialize plugin registry
        self.registry = PluginRegistry(model_name=model_name)
        self.registry.discover()

        # We'll configure the todo reporter after display is created
        # For now, use memory storage
        # Note: clarification and permission channels use "queue" type for TUI integration
        plugin_configs = {
            "todo": {
                "reporter_type": "console",  # Temporary, will be replaced
                "storage_type": "memory",
            },
            "references": {
                "channel_type": "queue",
                # Callbacks will be set after display is created
            },
            "clarification": {
                "channel_type": "queue",
                # Callbacks will be set after display is created
            },
        }
        self.registry.expose_all(plugin_configs)
        self.todo_plugin = self.registry.get_plugin("todo")

        # Initialize permission plugin with queue channel for TUI integration
        self.permission_plugin = PermissionPlugin()
        self.permission_plugin.initialize({
            "channel_type": "queue",
            "channel_config": {
                "use_colors": True,  # Enable ANSI colors for diff coloring
            },
            "policy": {
                "defaultPolicy": "ask",
                "whitelist": {"tools": [], "patterns": []},
                "blacklist": {"tools": [], "patterns": []},
            }
        })

        # Configure tools
        self._jaato.configure_tools(self.registry, self.permission_plugin, self.ledger)

        # Setup session plugin
        self._setup_session_plugin()

        # Register plugin commands for completion
        self._register_plugin_commands()

        return True

    def _setup_live_reporter(self) -> None:
        """Set up the live plan reporter after display is created."""
        if not self.todo_plugin or not self._display:
            return

        # Create live reporter with callbacks to display
        live_reporter = create_live_reporter(
            update_callback=self._display.update_plan,
            clear_callback=self._display.clear_plan,
            output_callback=self._create_output_callback(),
        )

        # Replace the todo plugin's reporter
        if hasattr(self.todo_plugin, '_reporter'):
            self.todo_plugin._reporter = live_reporter

    def _trace(self, msg: str) -> None:
        """Write trace message to file for debugging."""
        import datetime
        trace_path = os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        with open(trace_path, "a") as f:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] {msg}\n")
            f.flush()

    def _setup_queue_channels(self) -> None:
        """Set up queue-based channels for permission, clarification, and references.

        Queue channels display prompts in the output panel and receive user
        input via a shared queue. This avoids terminal mode switching issues
        that occur when the model runs in a background thread.
        """
        if not self._display:
            return

        def on_prompt_state_change(waiting: bool):
            """Called when channel starts/stops waiting for input."""
            self._waiting_for_channel_input = waiting
            self._trace(f"prompt_callback: waiting={waiting}")
            if self._display:
                # Pass the stored response options (from permission plugin)
                # This makes the permission plugin the single source of truth
                # for valid completion options
                response_options = self._pending_response_options if waiting else None
                self._display.set_waiting_for_channel_input(waiting, response_options)
                if waiting:
                    # Channel waiting for user input - stop spinner
                    self._display.stop_spinner()
                else:
                    # Channel finished - start spinner while model continues
                    self._display.start_spinner()

        # Create a cancel token provider that checks the session's current token
        def get_cancel_token():
            """Get the current cancel token from the session, if any."""
            if self._jaato:
                session = self._jaato.get_session()
                if session and hasattr(session, '_cancel_token'):
                    return session._cancel_token
            return None

        # Wrapper that acts like a CancelToken but checks the current session token
        class CancelTokenProxy:
            @property
            def is_cancelled(self):
                token = get_cancel_token()
                return token.is_cancelled if token else False

        cancel_token_proxy = CancelTokenProxy()

        # Set callbacks on clarification plugin channel
        if self.registry:
            clarification_plugin = self.registry.get_plugin("clarification")
            if clarification_plugin and hasattr(clarification_plugin, '_channel'):
                channel = clarification_plugin._channel
                if hasattr(channel, 'set_callbacks'):
                    channel.set_callbacks(
                        output_callback=self._create_output_callback(),
                        input_queue=self._channel_input_queue,
                        prompt_callback=on_prompt_state_change,
                        cancel_token=cancel_token_proxy,
                    )
                    self._trace("Clarification channel callbacks set (queue)")

            # Set callbacks on references plugin channel
            references_plugin = self.registry.get_plugin("references")
            if references_plugin and hasattr(references_plugin, '_channel'):
                channel = references_plugin._channel
                if hasattr(channel, 'set_callbacks'):
                    channel.set_callbacks(
                        output_callback=self._create_output_callback(),
                        input_queue=self._channel_input_queue,
                        prompt_callback=on_prompt_state_change,
                    )
                    self._trace("References channel callbacks set (queue)")

        # Set callbacks on permission plugin channel
        # Suppress "permission" source output since it's now shown in the tool tree
        if self.permission_plugin and hasattr(self.permission_plugin, '_channel'):
            channel = self.permission_plugin._channel
            self._trace(f"Permission channel type: {type(channel).__name__}")
            if channel and hasattr(channel, 'set_callbacks'):
                channel.set_callbacks(
                    output_callback=self._create_output_callback(suppress_sources={"permission"}),
                    input_queue=self._channel_input_queue,
                    prompt_callback=on_prompt_state_change,
                    cancel_token=cancel_token_proxy,
                )
                self._trace("Permission channel callbacks set (queue)")

    def _setup_retry_callback(self) -> None:
        """Set up retry callback to route rate limit messages to output panel.

        Routes retry notifications through the output callback system instead
        of printing directly to console, so they appear in the output panel.
        """
        if not self._jaato:
            return

        session = self._jaato.get_session()
        if not session:
            return

        # Create output callback for retry messages
        output_callback = self._create_output_callback()

        def on_retry(message: str, attempt: int, max_attempts: int, delay: float) -> None:
            """Route retry messages to output panel."""
            # Use source="retry" so UI can style appropriately
            output_callback("retry", message, "write")

        session.set_retry_callback(on_retry)
        self._trace("Retry callback configured for output panel")

    def _setup_agent_hooks(self) -> None:
        """Set up agent lifecycle hooks for UI integration."""
        if not self._jaato or not self._agent_registry:
            return

        # Import the protocol
        from shared.plugins.subagent.ui_hooks import AgentUIHooks

        # Create hooks implementation
        registry = self._agent_registry
        display = self._display

        class RichClientHooks:
            """UI hooks implementation for rich client."""

            def on_agent_created(self, agent_id, agent_name, agent_type, profile_name,
                               parent_agent_id, icon_lines, created_at):
                registry.create_agent(
                    agent_id=agent_id,
                    name=agent_name,
                    agent_type=agent_type,
                    profile_name=profile_name,
                    parent_agent_id=parent_agent_id,
                    icon_lines=icon_lines,
                    created_at=created_at
                )

            def on_agent_output(self, agent_id, source, text, mode):
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    # Stop spinner on first output from model
                    if source == "model" and buffer.spinner_active:
                        buffer.stop_spinner()
                    buffer.append(source, text, mode)
                    # Auto-scroll to bottom and refresh display
                    buffer.scroll_to_bottom()
                    if display:
                        display.refresh()

            def on_agent_status_changed(self, agent_id, status, error=None):
                registry.update_status(agent_id, status)
                # Start/stop spinner for this agent's buffer based on status
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    if status == "active":
                        buffer.start_spinner()
                        if display:
                            # Ensure the spinner animation timer is running
                            display.ensure_spinner_timer_running()
                            display.refresh()
                    elif status in ("done", "error"):
                        buffer.stop_spinner()
                        if display:
                            display.refresh()

            def on_agent_completed(self, agent_id, completed_at, success,
                                  token_usage=None, turns_used=None):
                registry.mark_completed(agent_id, completed_at)

            def on_agent_turn_completed(self, agent_id, turn_number, prompt_tokens,
                                       output_tokens, total_tokens, duration_seconds,
                                       function_calls):
                registry.update_turn_accounting(
                    agent_id, turn_number, prompt_tokens, output_tokens,
                    total_tokens, duration_seconds, function_calls
                )

            def on_agent_context_updated(self, agent_id, total_tokens, prompt_tokens,
                                        output_tokens, turns, percent_used):
                registry.update_context_usage(
                    agent_id, total_tokens, prompt_tokens,
                    output_tokens, turns, percent_used
                )

            def on_agent_history_updated(self, agent_id, history):
                registry.update_history(agent_id, history)

            def on_tool_call_start(self, agent_id, tool_name, tool_args, call_id=None):
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    buffer.add_active_tool(tool_name, tool_args, call_id=call_id)
                    if display:
                        display.refresh()

            def on_tool_call_end(self, agent_id, tool_name, success, duration_seconds,
                                  error_message=None, call_id=None):
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    buffer.mark_tool_completed(
                        tool_name, success, duration_seconds, error_message, call_id=call_id
                    )
                    if display:
                        display.refresh()

        hooks = RichClientHooks()

        # Store hooks reference for direct calls (e.g., when user sends input)
        self._ui_hooks = hooks

        # Register hooks with JaatoClient (main agent)
        self._jaato.set_ui_hooks(hooks)

        # Register hooks with SubagentPlugin if present
        if self.registry:
            subagent_plugin = self.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_ui_hooks'):
                subagent_plugin.set_ui_hooks(hooks)

    def _setup_permission_hooks(self) -> None:
        """Set up permission lifecycle hooks for UI integration.

        These hooks update the tool call tree to show permission status
        inline under each tool that requires permission.

        Also stores the response_options from the permission plugin so they
        can be passed to the input completer for context-aware autocompletion.
        """
        if not self.permission_plugin or not self._agent_registry:
            return

        registry = self._agent_registry
        display = self._display

        def on_permission_requested(tool_name: str, request_id: str, prompt_lines: list, response_options: list):
            """Called when permission prompt is shown.

            Args:
                tool_name: Name of the tool requesting permission.
                request_id: Unique identifier for this request.
                prompt_lines: Lines of text to display in the prompt.
                response_options: List of PermissionResponseOption objects
                    that define valid responses for autocompletion.
            """
            # Store the response options for the prompt_callback to use
            # This makes the permission plugin the single source of truth
            self._pending_response_options = response_options

            # Update the tool in the main agent's buffer
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_permission_pending(tool_name, prompt_lines)
                if display:
                    display.refresh()

        def on_permission_resolved(tool_name: str, request_id: str, granted: bool, method: str):
            """Called when permission is resolved."""
            # Clear pending options
            self._pending_response_options = None

            # Update the tool in the main agent's buffer
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_permission_resolved(tool_name, granted, method)
                if display:
                    display.refresh()

        self.permission_plugin.set_permission_hooks(
            on_requested=on_permission_requested,
            on_resolved=on_permission_resolved
        )

    def _setup_clarification_hooks(self) -> None:
        """Set up clarification lifecycle hooks for UI integration.

        These hooks update the tool call tree to show clarification status
        inline under the request_clarification tool, with questions shown
        one at a time.
        """
        if not self.registry or not self._agent_registry:
            return

        clarification_plugin = self.registry.get_plugin("clarification")
        if not clarification_plugin or not hasattr(clarification_plugin, 'set_clarification_hooks'):
            return

        registry = self._agent_registry
        display = self._display

        def on_clarification_requested(tool_name: str, prompt_lines: list):
            """Called when clarification session starts (context only)."""
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_clarification_pending(tool_name, prompt_lines)
                if display:
                    display.refresh()

        def on_clarification_resolved(tool_name: str):
            """Called when all clarification questions are answered."""
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_clarification_resolved(tool_name)
                if display:
                    display.refresh()

        def on_question_displayed(tool_name: str, question_index: int, total_questions: int, question_lines: list):
            """Called when each question is shown."""
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_clarification_question(tool_name, question_index, total_questions, question_lines)
                if display:
                    display.refresh()

        def on_question_answered(tool_name: str, question_index: int, answer_summary: str):
            """Called when user answers a question."""
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_question_answered(tool_name, question_index, answer_summary)
                if display:
                    display.refresh()

        clarification_plugin.set_clarification_hooks(
            on_requested=on_clarification_requested,
            on_resolved=on_clarification_resolved,
            on_question_displayed=on_question_displayed,
            on_question_answered=on_question_answered
        )

    def _setup_session_plugin(self) -> None:
        """Set up session persistence plugin."""
        if not self._jaato:
            return

        try:
            session_config = load_session_config()
            session_plugin = create_session_plugin()
            session_plugin.initialize({'storage_path': session_config.storage_path})
            self._jaato.set_session_plugin(session_plugin, session_config)

            if self.registry:
                self.registry.register_plugin(session_plugin, enrichment_only=True)

            if self.permission_plugin and hasattr(session_plugin, 'get_auto_approved_tools'):
                auto_approved = session_plugin.get_auto_approved_tools()
                if auto_approved:
                    self.permission_plugin.add_whitelist_tools(auto_approved)

        except Exception as e:
            pass  # Session plugin is optional

    def _get_plugin_commands_by_plugin(self) -> Dict[str, list]:
        """Collect plugin commands grouped by plugin name."""
        commands_by_plugin: Dict[str, list] = {}

        if self.registry:
            for plugin_name in self.registry.list_exposed():
                plugin = self.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'get_user_commands'):
                    commands = plugin.get_user_commands()
                    if commands:
                        commands_by_plugin[plugin_name] = commands

        if self.permission_plugin and hasattr(self.permission_plugin, 'get_user_commands'):
            commands = self.permission_plugin.get_user_commands()
            if commands:
                commands_by_plugin[self.permission_plugin.name] = commands

        if self._jaato:
            user_commands = self._jaato.get_user_commands()
            session_cmds = [cmd for name, cmd in user_commands.items()
                           if name in ('save', 'resume', 'sessions', 'delete-session', 'backtoturn')]
            if session_cmds:
                commands_by_plugin['session'] = session_cmds

        return commands_by_plugin

    def _register_plugin_commands(self) -> None:
        """Register plugin commands for autocompletion."""
        if not self._jaato:
            return

        user_commands = self._jaato.get_user_commands()
        if not user_commands:
            return

        completer_cmds = [(cmd.name, cmd.description) for cmd in user_commands.values()]
        self._input_handler.add_commands(completer_cmds)

        if hasattr(self._jaato, '_session_plugin') and self._jaato._session_plugin:
            session_plugin = self._jaato._session_plugin
            if hasattr(session_plugin, 'list_sessions'):
                self._input_handler.set_session_provider(session_plugin.list_sessions)

        # Set up plugin command argument completion
        self._setup_command_completion_provider()

    def _setup_command_completion_provider(self) -> None:
        """Set up the provider for plugin command argument completions."""
        if not self.registry:
            return

        command_to_plugin: dict = {}

        for plugin_name in self.registry.list_exposed():
            plugin = self.registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, 'get_command_completions'):
                if hasattr(plugin, 'get_user_commands'):
                    for cmd in plugin.get_user_commands():
                        command_to_plugin[cmd.name] = plugin

        if hasattr(self._jaato, '_session_plugin') and self._jaato._session_plugin:
            session_plugin = self._jaato._session_plugin
            if hasattr(session_plugin, 'get_command_completions'):
                if hasattr(session_plugin, 'get_user_commands'):
                    for cmd in session_plugin.get_user_commands():
                        command_to_plugin[cmd.name] = session_plugin

        if self.permission_plugin and hasattr(self.permission_plugin, 'get_command_completions'):
            if hasattr(self.permission_plugin, 'get_user_commands'):
                for cmd in self.permission_plugin.get_user_commands():
                    command_to_plugin[cmd.name] = self.permission_plugin

        # Built-in commands with special completion handling
        commands_with_completions = set(command_to_plugin.keys())
        commands_with_completions.add("model")  # Built-in model command
        commands_with_completions.add("tools enable")  # Tool management
        commands_with_completions.add("tools disable")  # Tool management

        def completion_provider(command: str, args: list) -> list:
            # Handle built-in model command
            if command == "model" and self._jaato:
                return self._jaato.get_model_completions(args)

            # Handle tools enable/disable completions
            if command == 'tools enable':
                # Show disabled tools (that can be enabled) + 'all'
                disabled = self.registry.list_disabled_tools() if self.registry else []
                completions = [('all', 'Enable all tools')]
                for tool in sorted(disabled):
                    completions.append((tool, 'disabled'))
                return completions
            elif command == 'tools disable':
                # Show enabled tools (that can be disabled) + 'all'
                if self.registry:
                    all_tools = self.registry.get_all_tool_names()
                    enabled = [t for t in all_tools if self.registry.is_tool_enabled(t)]
                else:
                    enabled = []
                completions = [('all', 'Disable all tools')]
                for tool in sorted(enabled):
                    completions.append((tool, 'enabled'))
                return completions

            # Handle plugin commands
            plugin = command_to_plugin.get(command)
            if plugin and hasattr(plugin, 'get_command_completions'):
                return plugin.get_command_completions(command, args)
            return []

        self._input_handler.set_command_completion_provider(
            completion_provider,
            commands_with_completions
        )

    def run_prompt(self, prompt: str) -> str:
        """Execute a prompt synchronously and return the response.

        This is used for single-prompt (non-interactive) mode only.
        For interactive mode, use _start_model_thread instead.
        """
        if not self._jaato:
            return "Error: Client not initialized"

        try:
            response = self._jaato.send_message(prompt, on_output=lambda s, t, m: print(f"[{s}] {t}"))
            return response if response else "(No response)"
        except Exception as e:
            return f"Error: {e}"

    def _start_model_thread(self, prompt: str) -> None:
        """Start the model call in a background thread.

        This allows the prompt_toolkit event loop to continue running,
        which is necessary for handling permission/clarification prompts.
        The model thread will update the display via callbacks.
        """
        if not self._jaato:
            if self._display:
                self._display.add_system_message("Error: Client not initialized", style="red")
            return

        if self._model_running:
            if self._display:
                self._display.add_system_message("Model is already running", style="yellow")
            return

        self._trace("_start_model_thread starting")

        # Spinner is already started by on_agent_status_changed hook
        # (called from _handle_input before this method)

        # Create callback that stops spinner on first output
        output_callback = self._create_output_callback(stop_spinner_on_first=True)

        # Initialize max_tokens_seen with CURRENT context usage (so we don't reset to 0)
        # This preserves the previous turn's context when a new turn starts
        current_usage = self._jaato.get_context_usage() if self._jaato else {}
        max_tokens_seen = {
            'prompt': current_usage.get('prompt_tokens', 0),
            'output': current_usage.get('output_tokens', 0),
            'total': current_usage.get('total_tokens', 0)
        }

        # Create usage update callback for real-time token accounting
        def usage_update_callback(usage) -> None:
            """Update status bar with real-time token usage during streaming."""
            # Skip zero values (initialization chunks, not real data)
            if usage.total_tokens == 0:
                return

            # Write to provider trace for debugging (same file as provider uses)
            import datetime
            trace_path = os.environ.get(
                "JAATO_PROVIDER_TRACE",
                os.path.join(tempfile.gettempdir(), "provider_trace.log")
            )
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] [rich_client_callback] received: prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}\n")
                    f.flush()
            except Exception as e:
                pass  # Ignore trace errors
            self._trace(f"[usage_callback] received: prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}")

            # Only update if we see HIGHER values (prevents backwards jumping)
            # The prompt_tokens is the most reliable indicator of context size
            # since it includes the full conversation history
            if usage.prompt_tokens >= max_tokens_seen['prompt']:
                max_tokens_seen['prompt'] = usage.prompt_tokens
                max_tokens_seen['output'] = max(max_tokens_seen['output'], usage.output_tokens)
                max_tokens_seen['total'] = max_tokens_seen['prompt'] + max_tokens_seen['output']
            else:
                # Skip update if prompt_tokens dropped (new API call with different context)
                self._trace(f"[usage_callback] skipping update: prompt {usage.prompt_tokens} < max {max_tokens_seen['prompt']}")
                return

            # Trace the condition check
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] [rich_client_callback] display={self._display is not None} jaato={self._jaato is not None}\n")
                    f.flush()
            except Exception:
                pass
            if self._display and self._jaato:
                # Get context limit for percentage calculation
                context_limit = self._jaato.get_context_limit()
                total_tokens = max_tokens_seen['total']
                percent_used = (total_tokens / context_limit * 100) if context_limit > 0 else 0
                tokens_remaining = max(0, context_limit - total_tokens)

                # Build usage dict for display update
                usage_dict = {
                    'total_tokens': total_tokens,
                    'prompt_tokens': max_tokens_seen['prompt'],
                    'output_tokens': max_tokens_seen['output'],
                    'context_limit': context_limit,
                    'percent_used': percent_used,
                    'tokens_remaining': tokens_remaining,
                }
                self._trace(f"[usage_callback] updating display: {percent_used:.1f}% used, {total_tokens} tokens")

                # Update agent registry if available (status bar reads from here)
                if self._agent_registry:
                    agent_id = self._agent_registry.get_selected_agent_id()
                    if agent_id:
                        self._agent_registry.update_context_usage(
                            agent_id=agent_id,
                            total_tokens=total_tokens,
                            prompt_tokens=max_tokens_seen['prompt'],
                            output_tokens=max_tokens_seen['output'],
                            turns=0,  # Don't know turn count during streaming
                            percent_used=percent_used
                        )

                # Also update display directly (fallback if no registry)
                self._display.update_context_usage(usage_dict)

                # Trace after update
                try:
                    with open(trace_path, "a") as f:
                        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        f.write(f"[{ts}] [rich_client_callback] update_context_usage called, percent={percent_used:.1f}%\n")
                        f.flush()
                except Exception:
                    pass

        def gc_threshold_callback(percent_used: float, threshold: float) -> None:
            """Handle GC threshold crossing notification."""
            self._trace(f"[gc_threshold] Context threshold crossed: {percent_used:.1f}% >= {threshold}%")
            # Show warning in output (optional - GC will happen automatically after turn)
            if self._display:
                self._display.add_system_message(
                    f"⚠ Context usage ({percent_used:.1f}%) exceeds threshold ({threshold}%). GC will run after this turn.",
                    style="yellow"
                )

        def model_thread():
            self._trace("[model_thread] started")
            self._model_running = True
            try:
                self._trace("[model_thread] calling send_message...")
                self._jaato.send_message(
                    prompt,
                    on_output=output_callback,
                    on_usage_update=usage_update_callback,
                    on_gc_threshold=gc_threshold_callback
                )
                self._trace(f"[model_thread] send_message returned")

                # Update context usage in status bar
                if self._display and self._jaato:
                    usage = self._jaato.get_context_usage()
                    self._display.update_context_usage(usage)

                # Add separator after model finishes
                # (response content is already shown via the callback)
                if self._display:
                    self._display.add_system_message("─" * 40, style="dim")
                    self._display.add_system_message("", style="dim")

            except KeyboardInterrupt:
                self._trace("[model_thread] KeyboardInterrupt")
                if self._display:
                    self._display.add_system_message("[Interrupted]", style="yellow")
            except Exception as e:
                self._trace(f"[model_thread] Exception: {e}")
                if self._display:
                    self._display.add_system_message(f"Error: {e}", style="red")
            finally:
                self._model_running = False
                self._model_thread = None
                # Ensure spinner is stopped (in case no output was received)
                if self._display:
                    self._display.stop_spinner()
                self._trace("[model_thread] finished")

        # Start model call in background thread
        self._model_thread = threading.Thread(target=model_thread, daemon=True)
        self._model_thread.start()
        self._trace("model thread started")

    def clear_history(self) -> None:
        """Clear conversation history."""
        if self._jaato:
            self._jaato.reset_session()
        self._original_inputs = []
        if self._display:
            self._display.clear_output()
            self._display.clear_plan()

    def _handle_input(self, user_input: str) -> None:
        """Handle user input from the prompt_toolkit input loop.

        Args:
            user_input: The text entered by the user.
        """
        # Check if pager is active - handle pager input first
        if self._display.pager_active:
            self._display.handle_pager_input(user_input)
            return

        # Route input to channel queue if waiting for permission/clarification
        if self._waiting_for_channel_input:
            # Check for 'v' to view full prompt in pager (only if truncated)
            if user_input.lower() == 'v':
                buffer = self._agent_registry.get_buffer("main")
                if buffer and buffer.has_truncated_pending_prompt():
                    prompt_data = buffer.get_pending_prompt_for_pager()
                    if prompt_data:
                        prompt_type, prompt_lines = prompt_data
                        # Show full prompt in pager (omit options line - it's in the original view)
                        title = "Permission Request" if prompt_type == "permission" else "Clarification Request"
                        lines = [(f"─── {title} ───", "bold cyan")]
                        for line in prompt_lines:
                            # Skip the options line (e.g. "[y]es [n]o [a]lways...")
                            if '[y]es' in line.lower() or '(type' in line.lower():
                                continue
                            # Color diff lines
                            if line.startswith('+') and not line.startswith('+++'):
                                lines.append((line, "green"))
                            elif line.startswith('-') and not line.startswith('---'):
                                lines.append((line, "red"))
                            elif line.startswith('@@'):
                                lines.append((line, "cyan"))
                            else:
                                lines.append((line, ""))
                        lines.append(("─" * 40, "dim"))
                        lines.append(("Press 'q' to close, Enter/Space for next page", "dim italic"))
                        self._display.show_lines(lines)
                        return
            # Don't echo answer - it's shown inline in the tool tree
            # Note: Input is proactively filtered in PTDisplay._on_input_changed()
            # so only valid permission responses can be typed
            self._channel_input_queue.put(user_input)
            self._trace(f"Input routed to channel queue: {user_input}")
            # Don't start spinner here - the channel may have more questions.
            # Spinner will be started when prompt_callback(False) indicates
            # the channel is done and model continues.
            return

        if not user_input:
            return

        if user_input.lower() in ('quit', 'exit', 'q'):
            self._display.add_system_message("Goodbye!", style="bold")
            self._display.stop()
            return

        if user_input.lower() == 'help':
            self._show_help()
            return

        if user_input.lower().startswith('tools'):
            self._handle_tools_command(user_input)
            return

        if user_input.lower() == 'plugins':
            self._show_plugins()
            return

        if user_input.lower() == 'reset':
            self.clear_history()
            self._display.show_lines([("[History cleared]", "yellow")])
            return

        if user_input.lower() == 'clear':
            self._display.clear_output()
            return

        if user_input.lower() == 'history':
            self._show_history()
            return

        if user_input.lower() == 'context':
            self._show_context()
            return

        if user_input.lower().startswith('export'):
            parts = user_input.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session_export.yaml"
            if filename.startswith('@'):
                filename = filename[1:]
            self._export_session(filename)
            return

        # Check for plugin commands
        plugin_result = self._try_execute_plugin_command(user_input)
        if plugin_result is not None:
            self._original_inputs.append({"text": user_input, "local": True})
            return

        # Track input
        self._original_inputs.append({"text": user_input, "local": False})

        # Show user input in output immediately
        self._display.append_output("user", user_input, "write")

        # Signal that main agent is now active (processing input)
        # This starts the spinner via the hooks
        if self._ui_hooks:
            self._ui_hooks.on_agent_status_changed(
                agent_id="main",
                status="active"
            )

        # Expand file references
        expanded_prompt = self._input_handler.expand_file_references(user_input)

        # Start model in background thread (non-blocking)
        # This allows the event loop to continue running for permission prompts
        self._start_model_thread(expanded_prompt)

    def run_interactive(self, initial_prompt: Optional[str] = None) -> None:
        """Run the interactive TUI loop.

        Args:
            initial_prompt: Optional prompt to run before entering interactive mode.
        """
        # Create the display with input handler and agent registry
        self._display = PTDisplay(
            input_handler=self._input_handler,
            agent_registry=self._agent_registry
        )

        # Set model info in status bar
        self._display.set_model_info(self._model_provider, self._model_name)

        # Set up stop callbacks for Ctrl-C handling
        self._display.set_stop_callbacks(
            stop_callback=lambda: self._jaato.stop() if self._jaato else False,
            is_running_callback=lambda: self._jaato.is_processing if self._jaato else False
        )

        # Set up the live reporter and queue channels
        self._setup_live_reporter()
        self._setup_queue_channels()

        # Set up retry callback to route rate limit messages to output panel
        self._setup_retry_callback()

        # Register UI hooks with jaato client and subagent plugin
        # This will create the main agent in the registry via set_ui_hooks()
        self._setup_agent_hooks()

        # Set up permission hooks for inline permission display in tool tree
        self._setup_permission_hooks()

        # Set up clarification hooks for inline clarification display in tool tree
        self._setup_clarification_hooks()

        # Load release name from file
        release_name = "Jaato Rich TUI Client"
        release_file = pathlib.Path(__file__).parent / "release_name.txt"
        if release_file.exists():
            release_name = release_file.read_text().strip()

        # Add welcome messages
        self._display.add_system_message(
            release_name,
            style="bold cyan"
        )
        if self._input_handler.has_completion:
            self._display.add_system_message(
                "Tab completion enabled. Use @file to reference files, /command for slash commands.",
                style="dim"
            )
        self._display.add_system_message(
            "Type 'help' for commands, 'quit' to exit, Esc+Esc to clear input",
            style="dim"
        )
        self._display.add_system_message("", style="dim")

        # Validate TTY and start display
        self._display.start()

        try:
            # Run the prompt_toolkit input loop
            # Initial prompt (if provided) is auto-submitted once event loop starts
            self._display.run_input_loop(self._handle_input, initial_prompt=initial_prompt)
        except (EOFError, KeyboardInterrupt):
            pass

        print("Goodbye!")

    def _get_all_tool_schemas(self) -> list:
        """Get all tool schemas from registry and plugins."""
        all_decls = []
        if self.registry:
            all_decls.extend(self.registry.get_exposed_tool_schemas())
        if self.permission_plugin:
            all_decls.extend(self.permission_plugin.get_tool_schemas())
        if self._jaato and hasattr(self._jaato, '_session_plugin') and self._jaato._session_plugin:
            if hasattr(self._jaato._session_plugin, 'get_tool_schemas'):
                all_decls.extend(self._jaato._session_plugin.get_tool_schemas())
        return all_decls

    def _handle_tools_command(self, user_input: str) -> None:
        """Handle the tools command with subcommands.

        Subcommands:
            tools / tools list  - List all tools with enabled/disabled status
            tools enable <name> - Enable a specific tool
            tools disable <name> - Disable a specific tool
            tools enable all    - Enable all tools
            tools disable all   - Disable all tools

        Args:
            user_input: The full user input string starting with 'tools'.
        """
        if not self._display:
            return

        parts = user_input.strip().split()
        subcommand = parts[1].lower() if len(parts) > 1 else "list"
        args = parts[2:] if len(parts) > 2 else []

        if subcommand == "list" or subcommand == "tools":
            self._show_tools_with_status()
            return

        if subcommand == "enable":
            if not args:
                self._display.show_lines([
                    ("[Error: Specify a tool name or 'all']", "red"),
                    ("  Usage: tools enable <tool_name>", "dim"),
                    ("         tools enable all", "dim"),
                ])
                return
            self._tools_enable(args[0])
            return

        if subcommand == "disable":
            if not args:
                self._display.show_lines([
                    ("[Error: Specify a tool name or 'all']", "red"),
                    ("  Usage: tools disable <tool_name>", "dim"),
                    ("         tools disable all", "dim"),
                ])
                return
            self._tools_disable(args[0])
            return

        # Unknown subcommand - show help
        self._display.show_lines([
            (f"[Unknown subcommand: {subcommand}]", "yellow"),
            ("  Available subcommands:", ""),
            ("    tools list         - List all tools with status", "dim"),
            ("    tools enable <n>   - Enable a tool (or 'all')", "dim"),
            ("    tools disable <n>  - Disable a tool (or 'all')", "dim"),
        ])

    def _get_tool_status(self) -> list:
        """Get status of all tools including enabled/disabled state."""
        status = []

        # Get registry tools with status
        if self.registry:
            status.extend(self.registry.get_tool_status())

        # Add permission plugin tools (always enabled)
        if self.permission_plugin:
            for schema in self.permission_plugin.get_tool_schemas():
                status.append({
                    'name': schema.name,
                    'description': schema.description,
                    'enabled': True,
                    'plugin': 'permission',
                })

        # Add session plugin tools (always enabled)
        if self._jaato and hasattr(self._jaato, '_session_plugin') and self._jaato._session_plugin:
            session_plugin = self._jaato._session_plugin
            if hasattr(session_plugin, 'get_tool_schemas'):
                for schema in session_plugin.get_tool_schemas():
                    status.append({
                        'name': schema.name,
                        'description': schema.description,
                        'enabled': True,
                        'plugin': 'session',
                    })

        return status

    def _show_tools_with_status(self) -> None:
        """Show tools with enabled/disabled status in output panel."""
        if not self._display:
            return

        tool_status = self._get_tool_status()

        if not tool_status:
            self._display.show_lines([("No tools available.", "yellow")])
            return

        # Group tools by plugin
        by_plugin: dict = {}
        for tool in tool_status:
            plugin = tool.get('plugin', 'unknown')
            if plugin not in by_plugin:
                by_plugin[plugin] = []
            by_plugin[plugin].append(tool)

        # Count enabled/disabled
        enabled_count = sum(1 for t in tool_status if t.get('enabled', True))
        disabled_count = len(tool_status) - enabled_count

        lines = [
            (f"Tools ({enabled_count} enabled, {disabled_count} disabled):", "bold"),
            ("  Use 'tools enable <name>' or 'tools disable <name>' to toggle", "dim"),
            ("", ""),
        ]

        for plugin_name in sorted(by_plugin.keys()):
            tools = by_plugin[plugin_name]
            lines.append((f"  [{plugin_name}]", "cyan"))

            for tool in sorted(tools, key=lambda t: t['name']):
                name = tool['name']
                desc = tool.get('description', '')
                enabled = tool.get('enabled', True)

                # Status indicator
                status = "✓" if enabled else "○"
                status_style = "green" if enabled else "red"

                # Truncate description if too long
                max_desc = 50
                if len(desc) > max_desc:
                    desc = desc[:max_desc - 3] + "..."

                lines.append((f"    {status} {name}: {desc}", status_style if not enabled else ""))

            lines.append(("", ""))

        lines.append(("  Legend: ✓ = enabled, ○ = disabled", "dim"))
        self._display.show_lines(lines)

    def _tools_enable(self, name: str) -> None:
        """Enable a tool or all tools.

        Args:
            name: Tool name or 'all'.
        """
        if not self._display:
            return

        if not self.registry:
            self._display.show_lines([("[Error: Registry not available]", "red")])
            return

        if name.lower() == "all":
            count = self.registry.enable_all_tools()
            self._refresh_session_tools()
            self._display.show_lines([(f"[Enabled all {count} tools]", "green")])
            return

        if self.registry.enable_tool(name):
            self._refresh_session_tools()
            self._display.show_lines([(f"[Enabled tool: {name}]", "green")])
        else:
            lines = [(f"[Error: Tool '{name}' not found]", "red")]
            available = self.registry.get_all_tool_names()
            if available:
                preview = ', '.join(sorted(available)[:10])
                lines.append((f"  Available tools: {preview}", "dim"))
                if len(available) > 10:
                    lines.append((f"  ... and {len(available) - 10} more", "dim"))
            self._display.show_lines(lines)

    def _tools_disable(self, name: str) -> None:
        """Disable a tool or all tools.

        Args:
            name: Tool name or 'all'.
        """
        if not self._display:
            return

        if not self.registry:
            self._display.show_lines([("[Error: Registry not available]", "red")])
            return

        if name.lower() == "all":
            count = self.registry.disable_all_tools()
            self._refresh_session_tools()
            self._display.show_lines([
                (f"[Disabled all {count} tools]", "yellow"),
                ("  Note: Permission and session tools remain available", "dim"),
            ])
            return

        if self.registry.disable_tool(name):
            self._refresh_session_tools()
            self._display.show_lines([(f"[Disabled tool: {name}]", "yellow")])
        else:
            lines = [(f"[Error: Tool '{name}' not found]", "red")]
            available = self.registry.get_all_tool_names()
            if available:
                preview = ', '.join(sorted(available)[:10])
                lines.append((f"  Available tools: {preview}", "dim"))
                if len(available) > 10:
                    lines.append((f"  ... and {len(available) - 10} more", "dim"))
            self._display.show_lines(lines)

    def _refresh_session_tools(self) -> None:
        """Refresh the session's tools after enabling/disabling."""
        if self._jaato and hasattr(self._jaato, '_session') and self._jaato._session:
            self._jaato._session.refresh_tools()
            self.log("[client] Session tools refreshed")

    def _show_plugins(self) -> None:
        """Show available plugins with status and descriptions."""
        if not self._display or not self.registry:
            return

        available = self.registry.list_available()
        exposed = set(self.registry.list_exposed())
        skipped = self.registry.list_skipped_plugins()

        lines = [("Available Plugins:", "bold")]
        lines.append(("", ""))

        for name in sorted(available):
            plugin = self.registry.get_plugin(name)
            if not plugin:
                continue

            # Determine status
            if name in exposed:
                status = "✓ enabled"
                status_style = "green"
                skip_reason = None
            elif name in skipped:
                status = "⊘ skipped"
                status_style = "yellow"
                skip_reason = skipped[name]  # List of required patterns
            else:
                status = "○ available"
                status_style = "dim"
                skip_reason = None

            # Get description from plugin
            description = self._get_plugin_description(plugin)

            # Format output
            lines.append((f"  {name}", "cyan"))
            lines.append((f"    Status: {status}", status_style))
            if skip_reason:
                patterns = ", ".join(skip_reason)
                lines.append((f"    Requires model: {patterns}", "yellow"))
            if description:
                # Wrap long descriptions
                if len(description) > 70:
                    description = description[:67] + "..."
                lines.append((f"    {description}", "dim"))
            lines.append(("", ""))

        # Add summary
        lines.append(("─" * 40, "dim"))
        lines.append((f"  Total: {len(available)} plugins ({len(exposed)} enabled, {len(skipped)} skipped)", "dim"))

        self._display.show_lines(lines)

    def _get_plugin_description(self, plugin) -> str:
        """Get a description for a plugin.

        Tries multiple sources:
        1. plugin.description property/attribute
        2. plugin.__doc__ (class docstring)
        3. First tool's description (if plugin has tools)
        4. Empty string as fallback
        """
        # Try description property
        if hasattr(plugin, 'description'):
            desc = plugin.description
            if callable(desc):
                desc = desc()
            if desc:
                return str(desc).strip().split('\n')[0]

        # Try class docstring
        if plugin.__doc__:
            # Take first non-empty line of docstring
            for line in plugin.__doc__.strip().split('\n'):
                line = line.strip()
                if line:
                    return line

        # Try first tool description
        if hasattr(plugin, 'get_tool_schemas'):
            try:
                schemas = plugin.get_tool_schemas()
                if schemas and schemas[0].description:
                    return f"Provides: {schemas[0].name}"
            except Exception:
                pass

        return ""

    def _show_history(self) -> None:
        """Show conversation history for SELECTED agent.

        Uses selected agent's history from the agent registry.
        """
        if not self._display:
            return

        # Get selected agent's history and accounting
        selected_agent = self._agent_registry.get_selected_agent()
        if not selected_agent:
            return

        history = selected_agent.history
        turn_accounting = selected_agent.turn_accounting

        # For main agent, also get turn boundaries
        turn_boundaries = []
        if selected_agent.agent_id == "main" and self._jaato:
            turn_boundaries = self._jaato.get_turn_boundaries()

        count = len(history)
        total_turns = len(turn_accounting) if turn_accounting else len(turn_boundaries)

        lines = [
            ("=" * 60, ""),
            (f"  Conversation History: {selected_agent.name}", "bold"),
            (f"  Agent: {selected_agent.agent_id} ({selected_agent.agent_type})", "dim"),
            (f"  Messages: {count}, Turns: {total_turns}", "dim"),
            ("  Tip: Use 'backtoturn <turn_id>' to revert to a specific turn (main agent only)", "dim"),
            ("=" * 60, ""),
        ]

        if count == 0:
            lines.append(("  (empty)", "dim"))
            lines.append(("", ""))
            self._display.show_lines(lines)
            return

        current_turn = 0
        turn_index = 0

        for i, content in enumerate(history):
            role = getattr(content, 'role', None) or 'unknown'
            parts = getattr(content, 'parts', None) or []

            is_user_text = (role == 'user' and parts and
                           hasattr(parts[0], 'text') and parts[0].text)

            # Print turn header if this starts a new turn
            if is_user_text:
                current_turn += 1
                lines.append(("", ""))
                lines.append(("─" * 60, ""))
                # Show timestamp in turn header if available
                turn_idx = current_turn - 1
                if turn_idx < len(turn_accounting) and 'start_time' in turn_accounting[turn_idx]:
                    start_time = turn_accounting[turn_idx]['start_time']
                    try:
                        dt = datetime.fromisoformat(start_time)
                        time_str = dt.strftime('%H:%M:%S')
                        lines.append((f"  ▶ TURN {current_turn}  [{time_str}]", "cyan"))
                    except (ValueError, TypeError):
                        lines.append((f"  ▶ TURN {current_turn}", "cyan"))
                else:
                    lines.append((f"  ▶ TURN {current_turn}", "cyan"))
                lines.append(("─" * 60, ""))

            role_label = "USER" if role == 'user' else "MODEL" if role == 'model' else role.upper()
            lines.append(("", ""))
            lines.append((f"  [{role_label}]", "bold"))

            if not parts:
                lines.append(("  (no content)", "dim"))
            else:
                for part in parts:
                    self._format_part(part, lines)

            # Show token accounting at end of turn
            is_last = (i == len(history) - 1)
            next_is_user_text = False
            if not is_last:
                next_content = history[i + 1]
                next_role = getattr(next_content, 'role', None) or 'unknown'
                next_parts = getattr(next_content, 'parts', None) or []
                next_is_user_text = (next_role == 'user' and next_parts and
                                    hasattr(next_parts[0], 'text') and next_parts[0].text)

            if (is_last or next_is_user_text) and turn_index < len(turn_accounting):
                turn = turn_accounting[turn_index]
                lines.append((f"  ─── tokens: {turn['prompt']} in / {turn['output']} out / {turn['total']} total", "dim"))
                if 'duration_seconds' in turn and turn['duration_seconds'] is not None:
                    duration = turn['duration_seconds']
                    lines.append((f"  ─── duration: {duration:.2f}s", "dim"))
                    func_calls = turn.get('function_calls', [])
                    if func_calls:
                        fc_total = sum(fc['duration_seconds'] for fc in func_calls)
                        model_time = duration - fc_total
                        lines.append((f"      model: {model_time:.2f}s, tools: {fc_total:.2f}s ({len(func_calls)} call(s))", "dim"))
                        for fc in func_calls:
                            lines.append((f"        - {fc['name']}: {fc['duration_seconds']:.2f}s", "dim"))
                turn_index += 1

        # Print totals
        if turn_accounting:
            total_prompt = sum(t['prompt'] for t in turn_accounting)
            total_output = sum(t['output'] for t in turn_accounting)
            total_all = sum(t['total'] for t in turn_accounting)
            total_duration = sum(t.get('duration_seconds', 0) or 0 for t in turn_accounting)
            total_fc_time = sum(
                sum(fc['duration_seconds'] for fc in t.get('function_calls', []))
                for t in turn_accounting
            )
            lines.append(("", ""))
            lines.append(("=" * 60, ""))
            lines.append((f"  Total: {total_prompt} in / {total_output} out / {total_all} total ({total_turns} turns)", "bold"))
            if total_duration > 0:
                total_model_time = total_duration - total_fc_time
                lines.append((f"  Time:  {total_duration:.2f}s total (model: {total_model_time:.2f}s, tools: {total_fc_time:.2f}s)", ""))
            lines.append(("=" * 60, ""))

        lines.append(("", ""))
        self._display.show_lines(lines)

    def _format_part(self, part: Any, lines: List[tuple]) -> None:
        """Format a single content part for history display.

        Args:
            part: A content part (text, function_call, or function_response).
            lines: List to append formatted lines to.
        """
        # Text content
        if hasattr(part, 'text') and part.text:
            text = part.text
            if len(text) > 500:
                text = text[:500] + f"... [{len(part.text)} chars total]"
            lines.append((f"  {text}", ""))

        # Function call
        elif hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            name = getattr(fc, 'name', 'unknown')
            args = getattr(fc, 'args', {})
            args_str = str(args)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            lines.append((f"  📤 CALL: {name}({args_str})", "yellow"))

        # Function response
        elif hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            name = getattr(fr, 'name', 'unknown')
            # ToolResult uses 'result' attribute, not 'response'
            response = getattr(fr, 'result', None) or getattr(fr, 'response', {})

            # Extract and display permission info first
            if isinstance(response, dict):
                perm = response.get('_permission')
                if perm:
                    decision = perm.get('decision', '?')
                    reason = perm.get('reason', '')
                    method = perm.get('method', '')
                    icon = '✓' if decision == 'allowed' else '✗'
                    style = "green" if decision == 'allowed' else "red"
                    lines.append((f"  {icon} Permission: {decision} via {method}", style))
                    if reason:
                        lines.append((f"    Reason: {reason}", "dim"))

            # Filter out _permission from display response
            if isinstance(response, dict):
                display_response = {k: v for k, v in response.items() if k != '_permission'}
            else:
                display_response = response

            resp_str = str(display_response)
            if len(resp_str) > 300:
                resp_str = resp_str[:300] + "..."
            lines.append((f"  📥 RESULT: {name} → {resp_str}", "green"))

        # Inline data (images, etc.)
        elif hasattr(part, 'inline_data') and part.inline_data:
            mime_type = part.inline_data.get('mime_type', 'unknown')
            data = part.inline_data.get('data')
            size = len(data) if data else 0
            lines.append((f"  📎 INLINE DATA: {mime_type} ({size} bytes)", "cyan"))

        else:
            # Unknown part type - show diagnostic info like simple client
            part_type = type(part).__name__
            # Show available attributes to help debugging
            attrs = [a for a in dir(part) if not a.startswith('_')]
            attr_preview = ', '.join(attrs[:5])
            if len(attrs) > 5:
                attr_preview += f", ... (+{len(attrs) - 5} more)"
            lines.append((f"  (unknown part: {part_type}, attrs: [{attr_preview}])", "yellow"))

    def _show_context(self) -> None:
        """Show context/token usage for SELECTED agent."""
        if not self._display:
            return

        # Get selected agent's context usage
        selected_agent = self._agent_registry.get_selected_agent()
        if not selected_agent:
            self._display.show_lines([("Context tracking not available", "yellow")])
            return

        usage = selected_agent.context_usage

        lines = [
            ("─" * 50, "dim"),
            (f"Context Usage: {selected_agent.name}", "bold"),
            (f"  Agent: {selected_agent.agent_id}", "dim"),
            (f"  Total tokens: {usage.get('total_tokens', 0)}", "dim"),
            (f"  Prompt tokens: {usage.get('prompt_tokens', 0)}", "dim"),
            (f"  Output tokens: {usage.get('output_tokens', 0)}", "dim"),
            (f"  Turns: {usage.get('turns', 0)}", "dim"),
            (f"  Percent used: {usage.get('percent_used', 0):.1f}%", "dim"),
            ("─" * 50, "dim"),
        ]

        self._display.show_lines(lines)

    def _export_session(self, filename: str) -> None:
        """Export session to YAML file."""
        if not self._display or not self._jaato:
            return

        try:
            from session_exporter import SessionExporter
            exporter = SessionExporter()
            history = self._jaato.get_history()
            result = exporter.export_to_yaml(history, self._original_inputs, filename)

            if result.get('success'):
                self._display.show_lines([
                    (f"Session exported to: {result['filename']}", "green")
                ])
            else:
                self._display.show_lines([
                    (f"Export failed: {result.get('error', 'Unknown error')}", "red")
                ])
        except ImportError:
            self._display.show_lines([
                ("Session exporter not available", "yellow")
            ])
        except Exception as e:
            self._display.show_lines([
                (f"Export error: {e}", "red")
            ])

    def _show_help(self) -> None:
        """Show help in output panel with pagination."""
        if not self._display:
            return

        help_lines = [
            ("Commands (auto-complete as you type):", "bold"),
            ("  help              - Show this help message", "dim"),
            ("  tools [subcmd]    - Manage tools available to the model", "dim"),
            ("                        tools list          - List all tools with status", "dim"),
            ("                        tools enable <n>    - Enable a tool (or 'all')", "dim"),
            ("                        tools disable <n>   - Disable a tool (or 'all')", "dim"),
            ("  plugins           - List available plugins with status", "dim"),
            ("  reset             - Clear conversation history", "dim"),
            ("  history           - Show full conversation history", "dim"),
            ("  context           - Show context window usage", "dim"),
            ("  export [file]     - Export session to YAML (default: session_export.yaml)", "dim"),
            ("  clear             - Clear output panel", "dim"),
            ("  quit              - Exit the client", "dim"),
            ("", "dim"),
        ]

        # Add plugin commands grouped by plugin
        commands_by_plugin = self._get_plugin_commands_by_plugin()
        if commands_by_plugin:
            help_lines.append(("Plugin-provided user commands:", "bold"))
            for plugin_name, commands in sorted(commands_by_plugin.items()):
                help_lines.append((f"  [{plugin_name}]", "cyan"))
                for cmd in commands:
                    padding = max(2, 14 - len(cmd.name))
                    shared_marker = " [shared with model]" if cmd.share_with_model else ""
                    help_lines.append((f"    {cmd.name}{' ' * padding}- {cmd.description}{shared_marker}", "dim"))
            help_lines.append(("", "dim"))

        help_lines.extend([
            ("When the model tries to use a tool, you'll see a permission prompt:", "bold"),
            ("  [y]es     - Allow this execution", "dim"),
            ("  [n]o      - Deny this execution", "dim"),
            ("  [a]lways  - Allow and remember for this session", "dim"),
            ("  [never]   - Deny and block for this session", "dim"),
            ("  [once]    - Allow just this once", "dim"),
            ("", "dim"),
            ("File references:", "bold"),
            ("  Use @path/to/file to include file contents in your prompt.", "dim"),
            ("  - @src/main.py      - Reference a file (contents included)", "dim"),
            ("  - @./config.json    - Reference with explicit relative path", "dim"),
            ("  - @~/documents/     - Reference with home directory", "dim"),
            ("  Completions appear automatically as you type after @.", "dim"),
            ("", "dim"),
            ("Slash commands:", "bold"),
            ("  Use /command_name [args...] to invoke slash commands from .jaato/commands/.", "dim"),
            ("  - Type / to see available commands with descriptions", "dim"),
            ("  - Pass arguments after the command name: /review file.py", "dim"),
            ("", "dim"),
            ("Multi-turn conversation:", "bold"),
            ("  The model remembers previous exchanges in this session.", "dim"),
            ("  Use 'reset' to start a fresh conversation.", "dim"),
            ("", "dim"),
            ("Keyboard shortcuts:", "bold"),
            ("  ↑/↓       - Navigate prompt history (or completion menu)", "dim"),
            ("  ←/→       - Move cursor within line", "dim"),
            ("  Ctrl+A/E  - Jump to start/end of line", "dim"),
            ("  TAB/Enter - Accept selected completion", "dim"),
            ("  Escape    - Dismiss completion menu", "dim"),
            ("  Esc+Esc   - Clear input", "dim"),
            ("  PgUp/PgDn - Scroll output up/down", "dim"),
            ("  Home/End  - Scroll to top/bottom of output", "dim"),
            ("", "dim"),
            ("Display:", "bold"),
            ("  The plan panel at top shows current plan status.", "dim"),
            ("  Model output scrolls in the panel below.", "dim"),
        ])

        # Show help with auto-pagination if needed
        self._display.show_lines(help_lines)

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.registry:
            self.registry.unexpose_all()
        if self.permission_plugin:
            self.permission_plugin.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rich TUI client with sticky plan display"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbose output"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Run a single prompt and exit (non-interactive mode)"
    )
    parser.add_argument(
        "--initial-prompt", "-i",
        type=str,
        help="Start with this prompt, then continue interactively"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Model provider to use (e.g., 'google_genai', 'github_models'). "
             "Overrides JAATO_PROVIDER env var."
    )
    args = parser.parse_args()

    # Check TTY before proceeding (except for single prompt mode)
    if not sys.stdout.isatty() and not args.prompt:
        sys.exit(
            "Error: rich-client requires an interactive terminal.\n"
            "Use simple-client for non-TTY environments."
        )

    client = RichClient(
        env_file=args.env_file,
        verbose=not args.quiet,
        provider=args.provider
    )

    if not client.initialize():
        sys.exit(1)

    try:
        if args.prompt:
            # Single prompt mode - run and exit (no TUI)
            response = client.run_prompt(args.prompt)
            print(response)
        elif args.initial_prompt:
            # Initial prompt mode - run prompt first, then continue interactively
            client.run_interactive(initial_prompt=args.initial_prompt)
        else:
            # Interactive mode
            client.run_interactive()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
