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
from live_display import LiveDisplay
from plan_reporter import create_live_reporter


class RichClient:
    """Rich TUI client with sticky plan display.

    Uses LiveDisplay to manage a split-screen layout with:
    - Sticky plan panel at top
    - Scrolling output below

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

        # Rich TUI display
        self._display: Optional[LiveDisplay] = None

        # Input handler (reused from simple-client)
        self._input_handler = InputHandler()

        # Track original inputs for session export
        self._original_inputs: list[dict] = []

    def log(self, msg: str) -> None:
        """Log message to output panel."""
        if self.verbose and self._display:
            self._display.add_system_message(msg, style="cyan")

    def _create_output_callback(self) -> Callable[[str, str, str], None]:
        """Create callback for real-time output to display."""
        def callback(source: str, text: str, mode: str) -> None:
            if self._display:
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
                self._display.add_system_message(f"Error: {e}", style="red")
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

        self._display.add_system_message(f"[{command_name}]", style="bold")

        if isinstance(result, dict):
            for key, value in result.items():
                if not key.startswith('_'):
                    self._display.add_system_message(f"  {key}: {value}", style="dim")
        else:
            self._display.add_system_message(f"  {result}", style="dim")

        if shared:
            self._display.add_system_message("  [Result shared with model]", style="dim cyan")

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

        # Initialize plugin registry
        self.registry = PluginRegistry(model_name=model_name)
        self.registry.discover()

        # We'll configure the todo reporter after display is created
        # For now, use memory storage
        plugin_configs = {
            "todo": {
                "reporter_type": "console",  # Temporary, will be replaced
                "storage_type": "memory",
            },
            "references": {
                "actor_type": "console",
            },
        }
        self.registry.expose_all(plugin_configs)
        self.todo_plugin = self.registry.get_plugin("todo")

        # Initialize permission plugin
        self.permission_plugin = PermissionPlugin()
        self.permission_plugin.initialize({
            "actor_type": "console",
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
        """Execute a prompt and return the response."""
        if not self._jaato:
            return "Error: Client not initialized"

        self.log("Sending prompt to model...")

        try:
            response = self._jaato.send_message(
                prompt,
                on_output=self._create_output_callback()
            )
            return response if response else '(No response)'

        except KeyboardInterrupt:
            return "[Interrupted]"
        except Exception as e:
            import traceback
            return f"Error: {e}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        if self._jaato:
            self._jaato.reset_session()
        self._original_inputs = []
        if self._display:
            self._display.clear_output()
            self._display.clear_plan()

    def run_interactive(self) -> None:
        """Run the interactive TUI loop."""
        # Create and start the live display
        self._display = LiveDisplay()

        # Build the prompt string with ANSI colors for InputHandler
        prompt_str = "\n\033[32mYou>\033[0m "

        with self._display:
            # Now set up the live reporter
            self._setup_live_reporter()

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

            while True:
                try:
                    # Get input with completion support (pauses live display temporarily)
                    user_input = self._display.get_input(
                        prompt_str,
                        input_handler=self._input_handler
                    )
                except (EOFError, KeyboardInterrupt):
                    self._display.add_system_message("Goodbye!", style="bold")
                    break

                if not user_input:
                    continue

                if user_input.lower() in ('quit', 'exit', 'q'):
                    self._display.add_system_message("Goodbye!", style="bold")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if user_input.lower() == 'tools':
                    self._show_tools()
                    continue

                if user_input.lower() == 'reset':
                    self.clear_history()
                    self._display.add_system_message(
                        "[History cleared]",
                        style="yellow"
                    )
                    continue

                if user_input.lower() == 'clear':
                    self._display.clear_output()
                    continue

                # Check for plugin commands
                plugin_result = self._try_execute_plugin_command(user_input)
                if plugin_result is not None:
                    self._original_inputs.append({"text": user_input, "local": True})
                    continue

                # Track input
                self._original_inputs.append({"text": user_input, "local": False})

                # Expand file references
                expanded_prompt = self._input_handler.expand_file_references(user_input)

                # Show user input in output
                self._display.append_output("user", user_input, "write")

                # Execute prompt
                response = self.run_prompt(expanded_prompt)

                # Response is already displayed via callback
                # Just add a separator
                self._display.add_system_message("─" * 40, style="dim")

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
        self._display.add_system_message("Available Tools:", style="bold")
        for tool in tools:
            self._display.add_system_message(f"  {tool.name}: {tool.description}", style="dim")

    def _show_help(self) -> None:
        """Show help in output panel."""
        if not self._display:
            return

        help_lines = [
            ("Commands:", "bold"),
            ("  help   - Show this help", "dim"),
            ("  tools  - List available tools", "dim"),
            ("  reset  - Clear conversation history", "dim"),
            ("  clear  - Clear output panel", "dim"),
            ("  quit   - Exit the client", "dim"),
            ("", "dim"),
            ("Display:", "bold"),
            ("  The plan panel at top shows current plan status.", "dim"),
            ("  Model output scrolls in the panel below.", "dim"),
            ("", "dim"),
        ]

        if self._input_handler.has_completion:
            help_lines.extend([
                ("Completion (auto-complete as you type):", "bold"),
                ("  Commands   - Type first letters to see matches", "dim"),
                ("  @file      - Tab to complete file paths", "dim"),
                ("  /command   - Slash commands from .jaato/commands/", "dim"),
                ("", "dim"),
                ("Keyboard:", "bold"),
                ("  ↑/↓        - Navigate history or completion menu", "dim"),
                ("  Tab/Enter  - Accept completion", "dim"),
                ("  Escape     - Dismiss completion menu", "dim"),
            ])

        # Add plugin commands if available
        if self._jaato:
            user_commands = self._jaato.get_user_commands()
            if user_commands:
                help_lines.append(("", "dim"))
                help_lines.append(("Plugin Commands:", "bold"))
                for name, cmd in user_commands.items():
                    help_lines.append((f"  {name:12} - {cmd.description}", "dim"))

        for line, style in help_lines:
            self._display.add_system_message(line, style=style)

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
        help="Path to .env file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbose output"
    )
    args = parser.parse_args()

    # Check TTY before proceeding
    if not sys.stdout.isatty():
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
        client.run_interactive()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
