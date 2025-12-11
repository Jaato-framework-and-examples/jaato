#!/usr/bin/env python3
"""Simple interactive console client demonstrating askPermission plugin behavior.

This client allows users to enter task descriptions, sends them to the model,
and shows interactive permission prompts in the terminal when tools are invoked.
Supports multi-turn conversation with history.
"""

import os
import sys
import pathlib
import readline  # Enables arrow key history navigation (fallback)
from typing import Optional, Dict, Any, List, Callable

# Add project root to path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

# Import local presentation and input modules
from terminal_ui import TerminalUI
from console_presenter import ConsolePresenter
from session_exporter import SessionExporter
from input_handler import InputHandler


class InteractiveClient:
    """Simple interactive console client with permission prompts and multi-turn history.

    Uses JaatoClient with SDK-managed conversation history.
    Delegates presentation to ConsolePresenter and input to InputHandler.
    """

    def __init__(self, env_file: str = ".env", verbose: bool = True):
        self.verbose = verbose
        self.env_file = env_file
        self._jaato: Optional[JaatoClient] = None
        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Optional[PermissionPlugin] = None
        self.todo_plugin: Optional[TodoPlugin] = None
        self.ledger = TokenLedger()

        # Initialize presentation and input components
        self._ui = TerminalUI()
        self._presenter = ConsolePresenter(self._ui)
        self._input_handler = InputHandler()
        self._exporter = SessionExporter()

        # Track original user inputs for export (before file reference expansion)
        # Each entry is {"text": str, "local": bool} where local=True for commands
        # that don't go to the model (plugin commands like "plan")
        self._original_inputs: list[dict] = []

    def log(self, msg: str) -> None:
        """Print message if verbose mode is enabled, with colorized [client] tag."""
        if self.verbose:
            if msg.startswith('[client]'):
                msg = self._ui.colorize('[client]', 'cyan') + msg[8:]
            print(msg)

    def _try_execute_plugin_command(self, user_input: str) -> Optional[Any]:
        """Try to execute user input as a plugin-provided command.

        Checks if the input matches a registered user command and executes it
        via JaatoClient. Uses the command's parameter schema for argument parsing.

        Args:
            user_input: The user's input string

        Returns:
            The command result if executed, or None if not a plugin command
        """
        if not self._jaato:
            return None

        # Get available user commands
        user_commands = self._jaato.get_user_commands()
        if not user_commands:
            return None

        # Parse input into command and arguments
        parts = user_input.strip().split(maxsplit=1)
        input_cmd = parts[0].lower() if parts else ""
        raw_args = parts[1] if len(parts) > 1 else ""

        # Check if input matches a command (case-insensitive)
        command = None
        for cmd_name, cmd in user_commands.items():
            if input_cmd == cmd_name.lower():
                command = cmd
                break

        if not command:
            return None

        # Parse arguments using the command's parameter schema
        args = parse_command_args(command, raw_args)

        # For save command, include user inputs for prompt history restoration
        if command.name.lower() == "save":
            args["user_inputs"] = self._original_inputs.copy()

        # Execute the command via JaatoClient
        try:
            result, shared = self._jaato.execute_user_command(command.name, args)

            # Display the result to the user
            self._presenter.display_command_result(command.name, result, shared)

            # For resume command, restore user inputs to prompt history
            if command.name.lower() == "resume" and isinstance(result, dict):
                user_inputs = result.get("user_inputs", [])
                if user_inputs:
                    self._restore_user_inputs(user_inputs)

            return result

        except Exception as e:
            print(f"\nError executing {command.name}: {e}")
            return {"error": str(e)}

    def _restore_user_inputs(self, user_inputs: List[str]) -> None:
        """Restore user inputs to prompt history after session resume.

        Args:
            user_inputs: List of user input strings from the resumed session.
        """
        self._original_inputs = list(user_inputs)
        count = self._input_handler.restore_history(
            [entry["text"] if isinstance(entry, dict) else entry for entry in user_inputs]
        )
        if count:
            self.log(f"[client] Restored {count} inputs to prompt history")

    def initialize(self) -> bool:
        """Initialize the client, loading config and connecting to Vertex AI."""
        # Load environment variables
        env_path = ROOT / self.env_file
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv(self.env_file)

        # Check for custom CA bundle
        active_bundle = active_cert_bundle(verbose=self.verbose)
        if active_bundle and self.verbose:
            self.log(f"[client] Using custom CA bundle: {active_bundle}")

        # Check required environment variables - MODEL_NAME always required
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

        # Initialize JaatoClient
        try:
            self._jaato = JaatoClient()
            if api_key:
                self.log(f"[client] Connecting to AI Studio")
                self._jaato.connect(model=model_name)
            else:
                self.log(f"[client] Connecting to Vertex AI (project={project_id}, location={location})")
                self._jaato.connect(project_id, location, model_name)
            self.log(f"[client] Using model: {model_name}")
            try:
                available_models = self._jaato.list_available_models()
                self.log(f"[client] Available models: {', '.join(available_models)}")
            except Exception as list_err:
                self.log(f"[client] Could not list available models: {list_err}")
        except Exception as e:
            print(f"Error: Failed to initialize Vertex AI client: {e}")
            return False

        # Initialize plugin registry with model name for requirements checking
        self.log("[client] Discovering plugins...")
        self.registry = PluginRegistry(model_name=model_name)
        discovered = self.registry.discover()
        self.log(f"[client] Found plugins: {discovered}")

        # Expose all discovered plugins with specific configs where needed
        plugin_configs = {
            "todo": {
                "reporter_type": "console",
                "storage_type": "memory",
            },
            "references": {
                "channel_type": "console",
            },
        }
        self.registry.expose_all(plugin_configs)
        self.todo_plugin = self.registry.get_plugin("todo")
        self.log(f"[client] Enabled plugins: {self.registry.list_exposed()}")

        # Initialize permission plugin with console channel
        self.log("[client] Initializing permission plugin with console channel...")
        self.permission_plugin = PermissionPlugin()
        self.permission_plugin.initialize({
            "channel_type": "console",
            "policy": {
                "defaultPolicy": "ask",
                "whitelist": {"tools": [], "patterns": []},
                "blacklist": {"tools": [], "patterns": []},
            }
        })
        self.log("[client] Permission plugin ready")

        # Configure tools on JaatoClient
        self._jaato.configure_tools(self.registry, self.permission_plugin, self.ledger)

        # Set up session persistence plugin
        self._setup_session_plugin()

        # Log available tools (including session plugin)
        all_decls = self.registry.get_exposed_tool_schemas()
        if self.permission_plugin:
            all_decls.extend(self.permission_plugin.get_tool_schemas())
        if self._jaato and hasattr(self._jaato, '_session_plugin') and self._jaato._session_plugin:
            if hasattr(self._jaato._session_plugin, 'get_tool_schemas'):
                all_decls.extend(self._jaato._session_plugin.get_tool_schemas())
        self.log(f"[client] Available tools: {[d.name for d in all_decls]}")

        # Register plugin-contributed tools as completable commands
        self._register_plugin_commands()

        return True

    def _setup_session_plugin(self) -> None:
        """Set up the session persistence plugin."""
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

            self.log(f"[client] Session plugin ready (storage: {session_config.storage_path})")

        except Exception as e:
            self.log(f"[client] Warning: Failed to initialize session plugin: {e}")

    def _close_session(self) -> None:
        """Close the current session, triggering auto-save if configured."""
        if self._jaato:
            try:
                self._jaato.close_session()
            except Exception as e:
                self.log(f"[client] Warning: Error closing session: {e}")

    def _register_plugin_commands(self) -> None:
        """Register plugin-contributed user commands for autocompletion."""
        if not self._jaato:
            return

        user_commands = self._jaato.get_user_commands()
        if not user_commands:
            return

        # Add to input handler for autocompletion
        completer_cmds = [(cmd.name, cmd.description) for cmd in user_commands.values()]
        self._input_handler.add_commands(completer_cmds)

        # Set up session ID completion if session plugin is available
        if hasattr(self._jaato, '_session_plugin') and self._jaato._session_plugin:
            session_plugin = self._jaato._session_plugin
            if hasattr(session_plugin, 'list_sessions'):
                self._input_handler.set_session_provider(session_plugin.list_sessions)

        # Set up plugin command argument completion
        self._setup_command_completion_provider()

        self.log(f"[client] Registered {len(user_commands)} plugin command(s) for completion")

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

        # Register permission plugin for command completions.
        # Unlike other plugins, the permission plugin is not exposed through the
        # registry because it acts as middleware (wrapping tool executors) rather
        # than providing tools directly. However, it still provides user commands
        # like "permissions allow/deny" that need completion support, so we
        # register it explicitly here.
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

    def run_prompt(
        self,
        prompt: str,
        on_output: Optional[Callable[[str, str, str], None]] = None
    ) -> str:
        """Execute a prompt and return the model's response.

        Args:
            prompt: The user's prompt text.
            on_output: Optional callback for real-time output.
        """
        if not self._jaato:
            return "Error: Client not initialized"

        self.log("\n[client] Sending prompt to model...")

        try:
            response = self._jaato.send_message(prompt, on_output)
            history_len = len(self._jaato.get_history())
            self.log(f"\n[client] Completed (history: {history_len} messages)")
            return response if response else '(No response text)'

        except KeyboardInterrupt:
            return "\n[Interrupted by user]"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during execution: {e}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        if self._jaato:
            self._jaato.reset_session()
        self._original_inputs = []
        self.log("[client] Conversation history cleared")

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

    def _export_session(self, filename: str) -> None:
        """Export current session to a YAML file for replay."""
        if not self._jaato:
            print("\n[No session to export - client not initialized]")
            return

        history = self._jaato.get_history()
        result = self._exporter.export_to_yaml(history, self._original_inputs, filename)

        if result['success']:
            print(f"\n[{result['message']}]")
            print(f"  Steps: {result['step_count']} interaction(s) + quit")
            print(f"  Replay with: python demo-scripts/run_demo.py {filename}")
        else:
            print(f"\n{result['error']}")

    def run_interactive(self, clear_history: bool = True, show_banner: bool = True) -> None:
        """Run the interactive prompt loop with multi-turn conversation.

        Args:
            clear_history: If True (default), clears conversation history at start.
            show_banner: If True (default), displays the welcome banner.
        """
        if show_banner:
            self._presenter.print_banner(has_completion=self._input_handler.has_completion)

        if clear_history:
            self.clear_history()

        prompt_str = self._presenter.format_prompt()

        while True:
            try:
                user_input = self._input_handler.get_input(prompt_str)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                self._close_session()
                break

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                self._close_session()
                break

            if user_input.lower() == 'help':
                self._presenter.print_help(self._get_plugin_commands_by_plugin())
                continue

            if user_input.lower() == 'tools':
                self._presenter.print_tools(self._get_all_tool_schemas())
                continue

            if user_input.lower() == 'reset':
                self.clear_history()
                print("[History cleared - starting fresh conversation]")
                continue

            if user_input.lower() == 'history':
                history = self._jaato.get_history() if self._jaato else []
                turn_accounting = self._jaato.get_turn_accounting() if self._jaato else []
                turn_boundaries = self._jaato.get_turn_boundaries() if self._jaato else []
                self._presenter.print_history(history, turn_accounting, turn_boundaries)
                continue

            if user_input.lower() == 'context':
                if self._jaato:
                    usage = self._jaato.get_context_usage()
                    self._presenter.print_context(usage)
                else:
                    print("\n[Context tracking not available - client not initialized]")
                continue

            if user_input.lower().startswith('export'):
                parts = user_input.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "session_export.yaml"
                if filename.startswith('@'):
                    filename = filename[1:]
                self._export_session(filename)
                continue

            # Check if input is a plugin-provided user command
            plugin_result = self._try_execute_plugin_command(user_input)
            if plugin_result is not None:
                self._original_inputs.append({"text": user_input, "local": True})
                continue

            # Track original input for export (before expansion)
            self._original_inputs.append({"text": user_input, "local": False})

            # Expand @file references to include file contents
            expanded_prompt = self._input_handler.expand_file_references(user_input)

            # Define callback for real-time output display
            def display_output(source: str, text: str, mode: str) -> None:
                if source == "model":
                    print(self._presenter.format_model_output(text))
                else:
                    print(text)

            # Execute the prompt with real-time output display
            response = self.run_prompt(expanded_prompt, on_output=display_output)

            # Display the final response (if any)
            if response and response != '(No response text)':
                print(self._presenter.format_model_output(response))

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.registry:
            self.registry.unexpose_all()
        if self.permission_plugin:
            self.permission_plugin.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive console client with permission prompts"
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

    client = InteractiveClient(
        env_file=args.env_file,
        verbose=not args.quiet
    )

    if not client.initialize():
        sys.exit(1)

    try:
        # Define callback for real-time output display (used by -p and -i modes)
        def display_output(source: str, text: str, mode: str) -> None:
            if source == "model":
                print(client._presenter.format_model_output(text))
            else:
                print(text)

        if args.prompt:
            # Single prompt mode - run and exit
            response = client.run_prompt(args.prompt, on_output=display_output)
            print(client._presenter.format_model_output(response))
        elif args.initial_prompt:
            # Initial prompt mode - show banner first, run prompt, then continue interactively
            client._presenter.print_banner(has_completion=client._input_handler.has_completion)
            readline.add_history(args.initial_prompt)
            response = client.run_prompt(args.initial_prompt, on_output=display_output)
            print(client._presenter.format_model_output(response))
            client.run_interactive(clear_history=False, show_banner=False)
        else:
            # Interactive mode
            client.run_interactive()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
