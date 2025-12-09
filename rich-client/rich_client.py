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
import threading
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


class RichClient:
    """Rich TUI client with sticky plan display.

    Uses PTDisplay (prompt_toolkit-based) to manage a full-screen layout with:
    - Sticky plan panel at top (hidden when no plan)
    - Scrolling output below
    - Integrated input prompt at bottom

    The plan panel updates in-place as plan steps progress,
    while model output scrolls naturally below.
    """

    def __init__(self, env_file: str = ".env", verbose: bool = True):
        self.verbose = verbose
        self.env_file = env_file
        self._jaato: Optional[JaatoClient] = None
        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Optional[PermissionPlugin] = None
        self.todo_plugin: Optional[TodoPlugin] = None
        self.ledger = TokenLedger()

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
        self._actor_input_queue: queue.Queue[str] = queue.Queue()
        self._waiting_for_actor_input: bool = False

        # Background model thread tracking
        self._model_thread: Optional[threading.Thread] = None
        self._model_running: bool = False

        # Model info for status bar
        self._model_provider: str = ""
        self._model_name: str = ""

    def log(self, msg: str) -> None:
        """Log message to output panel."""
        if self.verbose and self._display:
            self._display.add_system_message(msg, style="cyan")

    def _create_output_callback(self, stop_spinner_on_first: bool = False) -> Callable[[str, str, str], None]:
        """Create callback for real-time output to display.

        Args:
            stop_spinner_on_first: If True, stop the spinner on first output.
        """
        first_output_received = [False]  # Use list for mutability in closure

        def callback(source: str, text: str, mode: str) -> None:
            if self._display:
                # Stop spinner on first output if requested
                if stop_spinner_on_first and not first_output_received[0]:
                    first_output_received[0] = True
                    self._display.stop_spinner()
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
            lines.append((f"  {result}", "dim"))

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
        # Load environment
        env_path = ROOT / self.env_file
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv(self.env_file)

        # Check CA bundle
        active_bundle = active_cert_bundle(verbose=False)

        # Check required vars
        required_vars = ["PROJECT_ID", "LOCATION", "MODEL_NAME"]
        missing = [v for v in required_vars if not os.environ.get(v)]
        if missing:
            print(f"Error: Missing required environment variables: {', '.join(missing)}")
            return False

        project_id = os.environ["PROJECT_ID"]
        location = os.environ["LOCATION"]
        model_name = os.environ["MODEL_NAME"]

        # Initialize JaatoClient
        try:
            self._jaato = JaatoClient()
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
        # Note: clarification and permission actors use "queue" type for TUI integration
        plugin_configs = {
            "todo": {
                "reporter_type": "console",  # Temporary, will be replaced
                "storage_type": "memory",
            },
            "references": {
                "actor_type": "console",
            },
            "clarification": {
                "actor_type": "queue",
                # Callbacks will be set after display is created
            },
        }
        self.registry.expose_all(plugin_configs)
        self.todo_plugin = self.registry.get_plugin("todo")

        # Initialize permission plugin with queue actor for TUI integration
        self.permission_plugin = PermissionPlugin()
        self.permission_plugin.initialize({
            "actor_type": "queue",
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
        with open("/tmp/rich_client_trace.log", "a") as f:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] {msg}\n")
            f.flush()

    def _setup_queue_actors(self) -> None:
        """Set up queue-based actors for permission and clarification.

        Queue actors display prompts in the output panel and receive user
        input via a shared queue. This avoids terminal mode switching issues
        that occur when the model runs in a background thread.
        """
        if not self._display:
            return

        def on_prompt_state_change(waiting: bool):
            """Called when actor starts/stops waiting for input."""
            self._waiting_for_actor_input = waiting
            self._trace(f"prompt_callback: waiting={waiting}")
            if self._display:
                self._display.set_waiting_for_actor_input(waiting)

        # Set callbacks on clarification plugin actor
        if self.registry:
            clarification_plugin = self.registry.get_plugin("clarification")
            if clarification_plugin and hasattr(clarification_plugin, '_actor'):
                actor = clarification_plugin._actor
                if hasattr(actor, 'set_callbacks'):
                    actor.set_callbacks(
                        output_callback=self._create_output_callback(),
                        input_queue=self._actor_input_queue,
                        prompt_callback=on_prompt_state_change,
                    )
                    self._trace("Clarification actor callbacks set (queue)")

        # Set callbacks on permission plugin actor
        if self.permission_plugin and hasattr(self.permission_plugin, '_actor'):
            actor = self.permission_plugin._actor
            self._trace(f"Permission actor type: {type(actor).__name__}")
            if actor and hasattr(actor, 'set_callbacks'):
                actor.set_callbacks(
                    output_callback=self._create_output_callback(),
                    input_queue=self._actor_input_queue,
                    prompt_callback=on_prompt_state_change,
                )
                self._trace("Permission actor callbacks set (queue)")

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

        if not command_to_plugin:
            return

        def completion_provider(command: str, args: list) -> list:
            plugin = command_to_plugin.get(command)
            if plugin and hasattr(plugin, 'get_command_completions'):
                return plugin.get_command_completions(command, args)
            return []

        self._input_handler.set_command_completion_provider(
            completion_provider,
            set(command_to_plugin.keys())
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

        # Start spinner to show we're waiting for the model
        if self._display:
            self._display.start_spinner()

        # Create callback that stops spinner on first output
        output_callback = self._create_output_callback(stop_spinner_on_first=True)

        def model_thread():
            self._trace("[model_thread] started")
            self._model_running = True
            try:
                self._trace("[model_thread] calling send_message...")
                self._jaato.send_message(prompt, on_output=output_callback)
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

        # Route input to actor queue if waiting for permission/clarification
        if self._waiting_for_actor_input:
            # Show the answer in output panel
            if user_input:
                self._display.append_output("user", user_input, "write")
            self._actor_input_queue.put(user_input)
            self._trace(f"Input routed to actor queue: {user_input}")
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

        if user_input.lower() == 'tools':
            self._show_tools()
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
        # Create the display with input handler for completions
        self._display = PTDisplay(input_handler=self._input_handler)

        # Set model info in status bar
        self._display.set_model_info(self._model_provider, self._model_name)

        # Set up the live reporter and queue actors
        self._setup_live_reporter()
        self._setup_queue_actors()

        # Add welcome messages
        self._display.add_system_message(
            "Rich TUI Client - Sticky Plan Display",
            style="bold cyan"
        )
        if self._input_handler.has_completion:
            self._display.add_system_message(
                "Tab completion enabled. Use @file to reference files, /command for slash commands.",
                style="dim"
            )
        self._display.add_system_message(
            "Type 'help' for commands, 'quit' to exit",
            style="dim"
        )
        self._display.add_system_message("", style="dim")

        # Handle initial prompt if provided
        if initial_prompt:
            self._handle_input(initial_prompt)

        # Validate TTY
        self._display.start()

        try:
            # Run the prompt_toolkit input loop
            self._display.run_input_loop(self._handle_input)
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

    def _show_tools(self) -> None:
        """Show available tools in output panel."""
        if not self._display:
            return

        tools = self._get_all_tool_schemas()
        lines = [("Available Tools:", "bold")]
        for tool in tools:
            lines.append((f"  {tool.name}: {tool.description}", "dim"))

        self._display.show_lines(lines)

    def _show_history(self) -> None:
        """Show conversation history in output panel."""
        if not self._display or not self._jaato:
            return

        history = self._jaato.get_history()
        turn_accounting = self._jaato.get_turn_accounting()
        turn_boundaries = self._jaato.get_turn_boundaries()

        count = len(history)
        total_turns = len(turn_boundaries)

        lines = [
            ("─" * 50, "dim"),
            (f"Conversation History: {count} message(s), {total_turns} turn(s)", "bold"),
            ("Tip: Use 'backtoturn <turn_id>' to revert to a specific turn", "dim"),
        ]

        if count == 0:
            lines.append(("  (empty)", "dim"))
            self._display.show_lines(lines)
            return

        current_turn = 0
        turn_index = 0

        for i, content in enumerate(history):
            role = getattr(content, 'role', None) or 'unknown'
            parts = getattr(content, 'parts', None) or []

            is_user_text = (role == 'user' and parts and
                           hasattr(parts[0], 'text') and parts[0].text)

            if is_user_text:
                current_turn += 1
                lines.append((f"─ TURN {current_turn} ─", "cyan"))

            role_label = "USER" if role == 'user' else "MODEL" if role == 'model' else role.upper()

            for part in parts:
                if hasattr(part, 'text') and part.text:
                    text = part.text[:100] + "..." if len(part.text) > 100 else part.text
                    lines.append((f"  [{role_label}] {text}", "dim"))
                elif hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    lines.append((f"  [{role_label}] → {fc.name}(...)", "dim yellow"))
                elif hasattr(part, 'function_response') and part.function_response:
                    fr = part.function_response
                    lines.append((f"  [{role_label}] ← {fr.name} response", "dim green"))

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
                lines.append((f"    tokens: {turn['prompt']} in / {turn['output']} out", "dim"))
                turn_index += 1

        # Print totals
        if turn_accounting:
            total_prompt = sum(t['prompt'] for t in turn_accounting)
            total_output = sum(t['output'] for t in turn_accounting)
            lines.append(("─" * 50, "dim"))
            lines.append((
                f"Total: {total_prompt} prompt + {total_output} output = {total_prompt + total_output} tokens",
                "bold"
            ))

        self._display.show_lines(lines)

    def _show_context(self) -> None:
        """Show context/token usage in output panel."""
        if not self._display or not self._jaato:
            self._display.show_lines([("Context tracking not available", "yellow")])
            return

        usage = self._jaato.get_context_usage()

        lines = [
            ("─" * 50, "dim"),
            ("Context Usage:", "bold"),
            (f"  Total tokens: {usage['total_tokens']}", "dim"),
            (f"  Prompt tokens: {usage['prompt_tokens']}", "dim"),
            (f"  Output tokens: {usage['output_tokens']}", "dim"),
            (f"  Turns: {usage['turns']}", "dim"),
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
            ("  help          - Show this help message", "dim"),
            ("  tools         - List tools available to the model", "dim"),
            ("  reset         - Clear conversation history", "dim"),
            ("  history       - Show full conversation history", "dim"),
            ("  context       - Show context window usage", "dim"),
            ("  export [file] - Export session to YAML (default: session_export.yaml)", "dim"),
            ("  clear         - Clear output panel", "dim"),
            ("  quit          - Exit the client", "dim"),
            ("", "dim"),
        ]

        # Add plugin commands if available
        if self._jaato:
            user_commands = self._jaato.get_user_commands()
            if user_commands:
                help_lines.append(("Plugin Commands:", "bold"))
                for name, cmd in user_commands.items():
                    help_lines.append((f"  {name:12} - {cmd.description}", "dim"))
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
    args = parser.parse_args()

    # Check TTY before proceeding (except for single prompt mode)
    if not sys.stdout.isatty() and not args.prompt:
        sys.exit(
            "Error: rich-client requires an interactive terminal.\n"
            "Use simple-client for non-TTY environments."
        )

    client = RichClient(
        env_file=args.env_file,
        verbose=not args.quiet
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
