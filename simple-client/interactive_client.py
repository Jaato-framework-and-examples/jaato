#!/usr/bin/env python3
"""Simple interactive console client demonstrating askPermission plugin behavior.

This client allows users to enter task descriptions, sends them to the model,
and shows interactive permission prompts in the terminal when tools are invoked.
Supports multi-turn conversation with history.
"""

import os
import sys
import pathlib
import json
import readline  # Enables arrow key history navigation (fallback)
from typing import Optional, Dict, Any

# Try to import prompt_toolkit for enhanced completion
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.data_structures import Size
    import shutil
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    PromptSession = None
    ANSI = None

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

# Import file and command completion (local module)
try:
    from file_completer import CombinedCompleter, FileReferenceProcessor
    HAS_FILE_COMPLETER = True
except ImportError:
    HAS_FILE_COMPLETER = False
    CombinedCompleter = None
    FileReferenceProcessor = None


# ANSI color codes for terminal output (module-level for use in main())
ANSI_RESET = '\033[0m'
ANSI_BOLD = '\033[1m'


class InteractiveClient:
    """Simple interactive console client with permission prompts and multi-turn history.

    Uses JaatoClient with SDK-managed conversation history.
    """

    def __init__(self, env_file: str = ".env", verbose: bool = True):
        self.verbose = verbose
        self.env_file = env_file
        self._jaato: Optional[JaatoClient] = None
        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Optional[PermissionPlugin] = None
        self.todo_plugin: Optional[TodoPlugin] = None
        self.ledger = TokenLedger()

        # Initialize prompt_toolkit components if available
        self._pt_history = InMemoryHistory() if HAS_PROMPT_TOOLKIT else None
        self._completer = CombinedCompleter() if (HAS_PROMPT_TOOLKIT and HAS_FILE_COMPLETER) else None
        self._file_processor = FileReferenceProcessor() if HAS_FILE_COMPLETER else None

        # Prompt style for completion menu
        self._pt_style = Style.from_dict({
            'completion-menu.completion': 'bg:#333333 #ffffff',
            'completion-menu.completion.current': 'bg:#00aa00 #ffffff',
            'completion-menu.meta.completion': 'bg:#333333 #888888',
            'completion-menu.meta.completion.current': 'bg:#00aa00 #ffffff',
        }) if HAS_PROMPT_TOOLKIT else None

        # ANSI color codes for terminal output
        self._colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'green': '\033[32m',
            'cyan': '\033[36m',
        }

        # Track original user inputs for export (before file reference expansion)
        # Each entry is {"text": str, "local": bool} where local=True for commands
        # that don't go to the model (plugin commands like "plan")
        self._original_inputs: list[dict] = []

    def _c(self, text: str, color: str) -> str:
        """Apply ANSI color to text."""
        code = self._colors.get(color, '')
        return f"{code}{text}{self._colors['reset']}" if code else text

    def log(self, msg: str) -> None:
        """Print message if verbose mode is enabled, with colorized [client] tag."""
        if self.verbose:
            # Colorize [client] prefix
            if msg.startswith('[client]'):
                msg = self._c('[client]', 'cyan') + msg[8:]
            print(msg)

    def _get_user_input(self, prompt_str: str = None) -> str:
        """Get user input with command and file completion support.

        Uses prompt_toolkit if available for command and @file completion,
        falls back to standard input otherwise.

        Returns:
            User input string, or raises EOFError/KeyboardInterrupt
        """
        if prompt_str is None:
            prompt_str = f"\n{self._c('You>', 'green')} "

        if HAS_PROMPT_TOOLKIT and self._completer:
            # Use prompt_toolkit with ANSI-formatted prompt
            # Create output with enable_cpr=False to avoid CPR queries in PTY environments
            formatted_prompt = ANSI(prompt_str) if ANSI else prompt_str

            def get_size():
                cols, rows = shutil.get_terminal_size()
                return Size(rows=rows, columns=cols)

            output = Vt100_Output(sys.stdout, get_size=get_size, enable_cpr=False)
            session = PromptSession(
                completer=self._completer,
                history=self._pt_history,
                auto_suggest=AutoSuggestFromHistory(),
                style=self._pt_style,
                complete_while_typing=True,
                refresh_interval=0,
                output=output,
            )
            return session.prompt(formatted_prompt).strip()
        else:
            # Fallback to standard input
            return input(prompt_str).strip()

    def _expand_file_references(self, text: str) -> str:
        """Expand @file references to include file contents.

        If text contains @path/to/file references, reads those files
        and appends their contents to the prompt for model context.

        Returns:
            Original text with file contents appended, or original text if no references
        """
        if not self._file_processor:
            return text

        return self._file_processor.expand_references(text)

    def _try_execute_plugin_command(self, user_input: str) -> Optional[Any]:
        """Try to execute user input as a plugin-provided command.

        Checks if the input matches a registered user command and executes it
        via JaatoClient. If share_with_model is True, the result is automatically
        added to conversation history by JaatoClient.

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

        # Check if input matches a command (case-insensitive)
        input_lower = user_input.lower().strip()
        command_name = None
        for cmd_name in user_commands:
            if input_lower == cmd_name.lower():
                command_name = cmd_name
                break

        if not command_name:
            return None

        # Execute the command via JaatoClient
        try:
            result, shared = self._jaato.execute_user_command(command_name, {})

            # Display the result to the user
            self._display_command_result(command_name, result, shared)

            return result

        except Exception as e:
            print(f"\nError executing {command_name}: {e}")
            return {"error": str(e)}

    def _display_command_result(self, command_name: str, result: Any, shared: bool) -> None:
        """Display the result of a plugin command to the user.

        Args:
            command_name: Name of the executed command
            result: The command's return value
            shared: Whether the result was shared with the model
        """
        # Special formatting for plan command
        if command_name == "plan" and isinstance(result, dict):
            self._display_plan_result(result)
            if shared:
                print("  [Result shared with model]")
            return

        print(f"\n[{command_name}]")

        if isinstance(result, dict):
            # Pretty-print dict results
            for key, value in result.items():
                if key.startswith('_'):
                    continue  # Skip internal fields
                if isinstance(value, (list, dict)):
                    print(f"  {key}:")
                    if isinstance(value, list):
                        for item in value[:20]:  # Limit list display
                            if isinstance(item, dict):
                                # Format dict items compactly
                                item_str = ", ".join(f"{k}: {v}" for k, v in item.items())
                                print(f"    - {item_str}")
                            else:
                                print(f"    - {item}")
                        if len(value) > 20:
                            print(f"    ... and {len(value) - 20} more")
                    else:
                        print(f"    {json.dumps(value, indent=2)}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {result}")

        if shared:
            print("  [Result shared with model]")

    def _display_plan_result(self, result: Dict[str, Any]) -> None:
        """Display plan status in a user-friendly format.

        Args:
            result: The plan status dict from getPlanStatus
        """
        # Check for error
        if "error" in result:
            print(f"\n[plan] {result['error']}")
            return

        # Header with status
        status = result.get("status", "unknown")
        title = result.get("title", "Untitled Plan")

        # Status emoji and color hint
        status_display = {
            "pending": "📋 PENDING",
            "in_progress": "🔄 IN PROGRESS",
            "completed": "✅ COMPLETED",
            "failed": "❌ FAILED",
            "cancelled": "⚠️  CANCELLED",
        }.get(status, status.upper())

        print(f"\n{'=' * 60}")
        print(f"  {status_display}: {title}")
        print(f"{'=' * 60}")

        # Progress bar
        progress = result.get("progress", {})
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        in_prog = progress.get("in_progress", 0)
        pending = progress.get("pending", 0)
        percent = progress.get("percent", 0)

        if total > 0:
            bar_width = 40
            filled = int(bar_width * percent / 100)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"\n  Progress: [{bar}] {percent:.0f}%")
            print(f"  Steps: {completed} completed, {in_prog} in progress, {pending} pending, {failed} failed")

        # Summary if available
        summary = result.get("summary")
        if summary:
            print(f"\n  Summary: {summary}")

        # Steps list
        steps = result.get("steps", [])
        if steps:
            print(f"\n  Steps:")
            print(f"  {'-' * 56}")

            for step in sorted(steps, key=lambda s: s.get("sequence", 0)):
                seq = step.get("sequence", "?")
                desc = step.get("description", "")
                step_status = step.get("status", "pending")

                # Status indicator
                indicator = {
                    "pending": "○",
                    "in_progress": "◐",
                    "completed": "●",
                    "failed": "✗",
                    "skipped": "○",
                }.get(step_status, "?")

                # Truncate long descriptions
                max_desc_len = 50
                if len(desc) > max_desc_len:
                    desc = desc[:max_desc_len - 3] + "..."

                print(f"  {indicator} {seq}. {desc}")

                # Show result or error for completed/failed steps
                step_result = step.get("result")
                step_error = step.get("error")
                if step_result and step_status == "completed":
                    # Truncate long results
                    if len(step_result) > 60:
                        step_result = step_result[:57] + "..."
                    print(f"      └─ {step_result}")
                elif step_error and step_status == "failed":
                    if len(step_error) > 60:
                        step_error = step_error[:57] + "..."
                    print(f"      └─ Error: {step_error}")

        print(f"\n{'=' * 60}")

    def initialize(self) -> bool:
        """Initialize the client, loading config and connecting to Vertex AI."""
        # Load environment variables
        env_path = ROOT / self.env_file
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Try current directory
            load_dotenv(self.env_file)

        # Check for custom CA bundle
        active_bundle = active_cert_bundle(verbose=self.verbose)
        if active_bundle and self.verbose:
            self.log(f"[client] Using custom CA bundle: {active_bundle}")

        # Check required environment variables
        required_vars = ["PROJECT_ID", "LOCATION", "MODEL_NAME"]
        missing = [v for v in required_vars if not os.environ.get(v)]
        if missing:
            print(f"Error: Missing required environment variables: {', '.join(missing)}")
            print("Please set these in your .env file or environment.")
            return False

        project_id = os.environ["PROJECT_ID"]
        location = os.environ["LOCATION"]
        model_name = os.environ["MODEL_NAME"]

        # Initialize JaatoClient
        self.log(f"[client] Connecting to Vertex AI (project={project_id}, location={location})")
        try:
            self._jaato = JaatoClient()
            self._jaato.connect(project_id, location, model_name)
            self.log(f"[client] Using model: {model_name}")
            # List available models
            try:
                available_models = self._jaato.list_available_models()
                self.log(f"[client] Available models: {', '.join(available_models)}")
            except Exception as list_err:
                self.log(f"[client] Could not list available models: {list_err}")
        except Exception as e:
            print(f"Error: Failed to initialize Vertex AI client: {e}")
            return False

        # Initialize plugin registry
        self.log("[client] Discovering plugins...")
        self.registry = PluginRegistry()
        discovered = self.registry.discover()
        self.log(f"[client] Found plugins: {discovered}")

        # Expose all discovered plugins with specific configs where needed
        plugin_configs = {
            "todo": {
                "reporter_type": "console",
                "storage_type": "memory",
            },
            "references": {
                "actor_type": "console",
            },
        }
        self.registry.expose_all(plugin_configs)
        self.todo_plugin = self.registry.get_plugin("todo")
        self.log(f"[client] Enabled plugins: {self.registry.list_exposed()}")

        # Initialize permission plugin with console actor
        self.log("[client] Initializing permission plugin with console actor...")
        self.permission_plugin = PermissionPlugin()
        self.permission_plugin.initialize({
            "actor_type": "console",
            "policy": {
                "defaultPolicy": "ask",  # Ask for everything by default
                "whitelist": {"tools": [], "patterns": []},
                "blacklist": {"tools": [], "patterns": []},
            }
        })
        self.log("[client] Permission plugin ready")

        # Configure tools on JaatoClient
        self._jaato.configure_tools(self.registry, self.permission_plugin, self.ledger)

        # Log available tools
        all_decls = self.registry.get_exposed_declarations()
        if self.permission_plugin:
            all_decls.extend(self.permission_plugin.get_function_declarations())
        self.log(f"[client] Available tools: {[d.name for d in all_decls]}")

        # Register plugin-contributed tools as completable commands
        self._register_plugin_commands()

        return True

    def _register_plugin_commands(self) -> None:
        """Register plugin-contributed user commands for autocompletion.

        Collects user-facing commands from JaatoClient and adds them to the
        completer for autocomplete support.

        Note: Command execution and share_with_model handling is done by JaatoClient.
        """
        if not self._jaato or not self._completer:
            return

        # Get user commands from JaatoClient (which got them from registry)
        user_commands = self._jaato.get_user_commands()

        if not user_commands:
            return

        # Add to completer for autocompletion (need (name, description) tuples)
        completer_cmds = [(cmd.name, cmd.description) for cmd in user_commands.values()]
        self._completer.add_commands(completer_cmds)

        self.log(f"[client] Registered {len(user_commands)} plugin command(s) for completion")

    def run_prompt(self, prompt: str) -> str:
        """Execute a prompt and return the model's response.

        Tool calls will trigger interactive permission prompts.
        Uses SDK-managed conversation history for multi-turn context.
        """
        if not self._jaato:
            return "Error: Client not initialized"

        self.log(f"\n[client] Sending prompt to model...")

        try:
            response = self._jaato.send_message(prompt)

            history_len = len(self._jaato.get_history())
            self.log(f"\n[client] Completed (history: {history_len} messages)")

            return response if response else '(No response text)'

        except KeyboardInterrupt:
            return "\n[Interrupted by user]"
        except Exception as e:
            return f"Error during execution: {e}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        if self._jaato:
            self._jaato.reset_session()
        self._original_inputs = []
        self.log("[client] Conversation history cleared")

    def _print_banner(self) -> None:
        """Print the interactive client welcome banner."""
        print("\n" + "=" * 60)
        print("  Simple Interactive Client with Permission Prompts")
        print("=" * 60)
        print("\nEnter task descriptions for the model to execute.")
        print("Tool calls will prompt for your approval.")
        print("Use ↑/↓ arrows to navigate prompt history.")
        if HAS_PROMPT_TOOLKIT and self._completer:
            print("Commands auto-complete as you type (help, tools, reset, etc.).")
            print("Use @path/to/file to reference files (completions appear as you type).")
            print("Use /command to invoke slash commands (from .jaato/commands/).")
        print("Type 'quit' or 'exit' to stop, 'help' for guidance.\n")

    def run_interactive(self, clear_history: bool = True, show_banner: bool = True) -> None:
        """Run the interactive prompt loop with multi-turn conversation.

        Args:
            clear_history: If True (default), clears conversation history at start.
                          Set to False when continuing from an initial prompt.
            show_banner: If True (default), displays the welcome banner.
                        Set to False if banner was already shown.
        """
        if show_banner:
            self._print_banner()

        # Clear history at start of interactive session (unless continuing from initial prompt)
        if clear_history:
            self.clear_history()

        while True:
            try:
                user_input = self._get_user_input()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            if user_input.lower() == 'help':
                self._print_help()
                continue

            if user_input.lower() == 'tools':
                self._print_tools()
                continue

            if user_input.lower() == 'reset':
                self.clear_history()
                print("[History cleared - starting fresh conversation]")
                continue

            if user_input.lower() == 'history':
                self._print_history()
                continue

            if user_input.lower() == 'context':
                self._print_context()
                continue

            if user_input.lower().startswith('export'):
                # Parse optional filename: "export" or "export filename.yaml"
                parts = user_input.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "session_export.yaml"
                # Strip @ prefix if user used file completion
                if filename.startswith('@'):
                    filename = filename[1:]
                self._export_session(filename)
                continue

            # Check if input is a plugin-provided user command
            # (includes 'plan' from todo plugin, 'listReferences', 'selectReferences' from references plugin)
            plugin_result = self._try_execute_plugin_command(user_input)
            if plugin_result is not None:
                # Plugin command was executed, result already displayed
                # Track for export as a local command (doesn't go to model directly)
                self._original_inputs.append({"text": user_input, "local": True})
                continue

            # Track original input for export (before expansion)
            self._original_inputs.append({"text": user_input, "local": False})

            # Expand @file references to include file contents
            expanded_prompt = self._expand_file_references(user_input)

            # Execute the prompt
            response = self.run_prompt(expanded_prompt)
            print(f"\n{self._c('Model>', 'bold')} {response}")

    def _print_help(self) -> None:
        """Print help information."""
        print("""
Commands (auto-complete as you type):
  help          - Show this help message
  tools         - List tools available to the model
  reset         - Clear conversation history
  history       - Show full conversation history
  context       - Show context window usage
  export [file] - Export session to YAML for replay (default: session_export.yaml)
  quit          - Exit the client
  exit          - Exit the client""")

        # Dynamically list plugin-contributed commands
        self._print_plugin_commands()

        print("""
When the model tries to use a tool, you'll see a permission prompt:
  [y]es     - Allow this execution
  [n]o      - Deny this execution
  [a]lways  - Allow and remember for this session
  [never]   - Deny and block for this session
  [once]    - Allow just this once

File references:
  Use @path/to/file to include file contents in your prompt.
  - @src/main.py      - Reference a file (contents included)
  - @./config.json    - Reference with explicit relative path
  - @~/documents/     - Reference with home directory
  - @/absolute/path   - Absolute path reference
  Completions appear automatically as you type after @.
  Use ↑/↓ to navigate the dropdown, Enter or TAB to accept.

Slash commands:
  Use /command_name [args...] to invoke slash commands from .jaato/commands/.
  - Type / to see available commands with descriptions
  - Pass arguments after the command name: /review file.py
  - Command files use {{$1}}, {{$2}} for parameter substitution
  - Use {{$1:default}} for optional parameters with defaults
  Example: "/summarize" or "/review src/main.py"

Example prompts:
  - "List files in the current directory"
  - "Show me the git status"
  - "Review @src/utils.py for issues"
  - "Explain what @./README.md describes"
  - "/review src/main.py" - Invoke slash command with argument

Multi-turn conversation:
  The model remembers previous exchanges in this session.
  Use 'reset' to start a fresh conversation.

Keyboard shortcuts:
  ↑/↓       - Navigate prompt history (or completion menu)
  ←/→       - Move cursor within line
  Ctrl+A/E  - Jump to start/end of line
  TAB/Enter - Accept selected completion
  Escape    - Dismiss completion menu
""")

    def _print_tools(self) -> None:
        """Print available tools."""
        print("\nAvailable tools:")
        for decl in self.registry.get_exposed_declarations():
            print(f"  - {decl.name}: {decl.description}")
        print()

    def _print_plugin_commands(self) -> None:
        """Print plugin-contributed user commands grouped by plugin."""
        if not self.registry:
            return

        # Collect commands by plugin
        commands_by_plugin: Dict[str, list] = {}
        for plugin_name in self.registry.list_exposed():
            plugin = self.registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, 'get_user_commands'):
                commands = plugin.get_user_commands()
                if commands:
                    commands_by_plugin[plugin_name] = commands

        if not commands_by_plugin:
            return

        print("\nPlugin-provided user commands:")
        for plugin_name, commands in sorted(commands_by_plugin.items()):
            for cmd in commands:
                # Calculate padding for alignment
                padding = max(2, 18 - len(cmd.name))
                shared_marker = " [also available to the model as a tool]" if cmd.share_with_model else ""
                print(f"  {cmd.name}{' ' * padding}- {cmd.description} ({plugin_name}){shared_marker}")

    def _print_context(self) -> None:
        """Print context window usage statistics."""
        if not self._jaato:
            print("\n[Context tracking not available - client not initialized]")
            return

        usage = self._jaato.get_context_usage()

        print(f"\n{'=' * 50}")
        print(f"  Context Window Usage")
        print(f"{'=' * 50}")
        print(f"  Model: {usage['model']}")
        print(f"  Context limit: {usage['context_limit']:,} tokens")
        print()

        # Visual progress bar
        bar_width = 40
        filled = int(bar_width * usage['percent_used'] / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"  [{bar}] {usage['percent_used']:.2f}%")
        print()

        # Token breakdown
        print(f"  Tokens used:     {usage['total_tokens']:,}")
        print(f"    - Prompt:      {usage['prompt_tokens']:,}")
        print(f"    - Output:      {usage['output_tokens']:,}")
        print(f"  Tokens remaining: {usage['tokens_remaining']:,}")
        print(f"  Turns: {usage['turns']}")
        print(f"{'=' * 50}")
        print()

    def _print_history(self) -> None:
        """Print full conversation history with token accounting."""
        history = self._jaato.get_history() if self._jaato else []
        turn_accounting = self._jaato.get_turn_accounting() if self._jaato else []
        count = len(history)

        print(f"\n{'=' * 50}")
        print(f"  Conversation History: {count} message(s)")
        print(f"{'=' * 50}")

        if count == 0:
            print("  (empty)")
            print()
            return

        # Track which turn we're in (user text messages start a new turn)
        turn_index = 0

        for i, content in enumerate(history):
            role = getattr(content, 'role', None) or 'unknown'
            parts = getattr(content, 'parts', None) or []

            # Check if this is a new user turn (user message with text, not function response)
            is_user_text = (role == 'user' and parts and
                           hasattr(parts[0], 'text') and parts[0].text)

            print(f"\n[{i+1}] {role.upper()}")
            print("-" * 40)

            for part in parts:
                self._print_part(part)

            # Show token accounting at end of turn (after final model text response)
            # A turn ends when the next message is a user text message or at end of history
            is_last = (i == len(history) - 1)
            next_is_user_text = False
            if not is_last:
                next_content = history[i + 1]
                next_role = getattr(next_content, 'role', None) or 'unknown'
                next_parts = getattr(next_content, 'parts', None) or []
                next_is_user_text = (next_role == 'user' and next_parts and
                                    hasattr(next_parts[0], 'text') and next_parts[0].text)

            if (is_last or next_is_user_text) and turn_index < len(turn_accounting):
                tokens = turn_accounting[turn_index]
                print(f"  ─── tokens: {tokens['prompt']} in / {tokens['output']} out / {tokens['total']} total")
                turn_index += 1

        # Print totals
        if turn_accounting:
            total_prompt = sum(t['prompt'] for t in turn_accounting)
            total_output = sum(t['output'] for t in turn_accounting)
            total_all = sum(t['total'] for t in turn_accounting)
            print(f"\n{'=' * 50}")
            print(f"  Total: {total_prompt} in / {total_output} out / {total_all} total ({len(turn_accounting)} turns)")
            print(f"{'=' * 50}")

        print()

    def _print_part(self, part) -> None:
        """Print a single content part."""
        # Text content
        if hasattr(part, 'text') and part.text:
            text = part.text
            # Truncate very long text
            if len(text) > 500:
                text = text[:500] + f"... [{len(part.text)} chars total]"
            print(f"  {text}")

        # Function call
        elif hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            name = getattr(fc, 'name', 'unknown')
            args = getattr(fc, 'args', {})
            # Format args compactly
            args_str = str(args)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            print(f"  📤 CALL: {name}({args_str})")

        # Function response
        elif hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            name = getattr(fr, 'name', 'unknown')
            response = getattr(fr, 'response', {})

            # Extract and display permission info first (on separate line)
            if isinstance(response, dict):
                perm = response.get('_permission')
                if perm:
                    decision = perm.get('decision', '?')
                    reason = perm.get('reason', '')
                    method = perm.get('method', '')
                    icon = '✓' if decision == 'allowed' else '✗'
                    print(f"  {icon} Permission: {decision} via {method}")
                    if reason:
                        print(f"    Reason: {reason}")

            # Filter out _permission from display response
            if isinstance(response, dict):
                display_response = {k: v for k, v in response.items() if k != '_permission'}
            else:
                display_response = response

            # Format response compactly
            resp_str = str(display_response)
            if len(resp_str) > 300:
                resp_str = resp_str[:300] + "..."
            print(f"  📥 RESULT: {name} → {resp_str}")

        else:
            # Unknown part type
            print(f"  (unknown part: {type(part).__name__})")

    def _export_session(self, filename: str) -> None:
        """Export current session to a YAML file for replay.

        Generates a YAML file in the format expected by demo-scripts/run_demo.py,
        allowing the session to be replayed later. Uses original user inputs
        (before file reference expansion) for cleaner, replayable output.

        Args:
            filename: Path to the output YAML file.
        """
        try:
            import yaml
        except ImportError:
            print("\nError: PyYAML is required for export. Install with: pip install pyyaml")
            return

        if not self._jaato:
            print("\n[No session to export - client not initialized]")
            return

        if not self._original_inputs:
            print("\n[No conversation history to export]")
            return

        # Extract permission decisions from history, grouped by user turn
        history = self._jaato.get_history()
        turn_permissions: list[list[str]] = []
        current_permissions: list[str] = []
        in_user_turn = False

        for content in history:
            role = getattr(content, 'role', None) or 'unknown'
            parts = getattr(content, 'parts', None) or []

            if role == 'user':
                # Check if this is a user text message (starts new turn)
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text.strip()
                        if text.startswith('[User executed command:'):
                            continue
                        # Save previous turn's permissions and start new turn
                        if in_user_turn:
                            turn_permissions.append(current_permissions)
                        current_permissions = []
                        in_user_turn = True

            elif role == 'model':
                # Collect permission data from function responses
                for part in parts:
                    if hasattr(part, 'function_response') and part.function_response:
                        fr = part.function_response
                        response = getattr(fr, 'response', {})
                        if isinstance(response, dict):
                            perm = response.get('_permission')
                            if perm:
                                decision = perm.get('decision', '')
                                method = perm.get('method', '')
                                perm_value = self._map_permission_to_yaml(decision, method)
                                current_permissions.append(perm_value)

        # Don't forget the last turn's permissions
        if in_user_turn:
            turn_permissions.append(current_permissions)

        # Build steps from original inputs with matched permissions
        final_steps = []
        model_turn_index = 0  # Track index for model-bound prompts only
        for input_entry in self._original_inputs:
            user_input = input_entry["text"]
            is_local = input_entry["local"]

            if is_local:
                # Local commands (plugin commands like "plan") don't need permissions
                final_steps.append({
                    'type': user_input,
                    'local': True,
                })
            else:
                # Model-bound prompts may have permissions
                perms = turn_permissions[model_turn_index] if model_turn_index < len(turn_permissions) else []
                model_turn_index += 1

                # Determine permission value
                permission = 'y'  # Default
                if perms:
                    # Use the most permissive permission granted
                    # Priority: 'a' (always) > 'y' (yes) > 'n' (no)
                    if 'a' in perms:
                        permission = 'a'
                    elif 'y' in perms or 'once' in perms:
                        permission = 'y'
                    elif 'n' in perms:
                        permission = 'n'
                    elif 'never' in perms:
                        permission = 'never'

                final_steps.append({
                    'type': user_input,
                    'permission': permission,
                })

        # Add quit step
        final_steps.append({'type': 'quit', 'delay': 0.08})

        # Build the YAML document
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

        export_data = {
            'name': f'Session Export [{timestamp}]',
            'timeout': 120,
            'steps': final_steps,
        }

        # Write to file
        try:
            with open(filename, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"\n[Session exported to: {filename}]")
            print(f"  Steps: {len(final_steps) - 1} interaction(s) + quit")
            print(f"  Replay with: python demo-scripts/run_demo.py {filename}")
        except IOError as e:
            print(f"\nError writing file: {e}")

    def _map_permission_to_yaml(self, decision: str, method: str) -> str:
        """Map permission decision/method to YAML permission value.

        Args:
            decision: 'allowed' or 'denied'
            method: 'user', 'remembered', 'whitelist', etc.

        Returns:
            YAML permission value: 'y', 'n', 'a', 'never', 'once'
        """
        if decision == 'allowed':
            if method == 'remembered':
                return 'a'  # Was 'always' - permission remembered
            elif method == 'whitelist':
                return 'a'  # Auto-approved, use 'always' for replay
            else:
                return 'y'  # User approved this one
        else:  # denied
            if method == 'remembered':
                return 'never'  # Was 'never' - denial remembered
            else:
                return 'n'  # User denied this one

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.registry:
            self.registry.unexpose_all()
        if self.permission_plugin:
            self.permission_plugin.shutdown()
        # Note: todo_plugin is managed via registry.unexpose_all()


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
        if args.prompt:
            # Single prompt mode - run and exit
            response = client.run_prompt(args.prompt)
            print(f"\n{ANSI_BOLD}Model>{ANSI_RESET} {response}")
        elif args.initial_prompt:
            # Initial prompt mode - show banner first, run prompt, then continue interactively
            client._print_banner()
            readline.add_history(args.initial_prompt)  # Add to history for ↑ recall
            response = client.run_prompt(args.initial_prompt)
            print(f"\n{ANSI_BOLD}Model>{ANSI_RESET} {response}")
            client.run_interactive(clear_history=False, show_banner=False)
        else:
            # Interactive mode
            client.run_interactive()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
