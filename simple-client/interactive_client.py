#!/usr/bin/env python3
"""Simple interactive console client demonstrating askPermission plugin behavior.

This client allows users to enter task descriptions, sends them to the model,
and shows interactive permission prompts in the terminal when tools are invoked.
Supports multi-turn conversation with history.
"""

import os
import sys
import pathlib
import readline  # Enables arrow key history navigation
from typing import Optional

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

    def log(self, msg: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)

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
        except Exception as e:
            print(f"Error: Failed to initialize Vertex AI client: {e}")
            return False

        # Initialize plugin registry
        self.log("[client] Discovering plugins...")
        self.registry = PluginRegistry()
        discovered = self.registry.discover()
        self.log(f"[client] Found plugins: {discovered}")

        # Expose CLI plugin (MCP is optional based on .mcp.json)
        if "cli" in discovered:
            self.registry.expose_tool("cli")
            self.log("[client] CLI plugin enabled")

        if "mcp" in discovered:
            try:
                self.registry.expose_tool("mcp")
                self.log("[client] MCP plugin enabled")
            except Exception as e:
                self.log(f"[client] MCP plugin skipped: {e}")

        # Expose TODO plugin with console reporter for plan tracking
        if "todo" in discovered:
            self.registry.expose_tool("todo", {
                "reporter_type": "console",
                "storage_type": "memory",
            })
            self.todo_plugin = self.registry.get_plugin("todo")
            self.log("[client] TODO plugin enabled - plan tracking available")

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

        return True

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
        self.log("[client] Conversation history cleared")

    def run_interactive(self, clear_history: bool = True) -> None:
        """Run the interactive prompt loop with multi-turn conversation.

        Args:
            clear_history: If True (default), clears conversation history at start.
                          Set to False when continuing from an initial prompt.
        """
        print("\n" + "=" * 60)
        print("  Simple Interactive Client with Permission Prompts")
        print("=" * 60)
        print("\nEnter task descriptions for the model to execute.")
        print("Tool calls will prompt for your approval.")
        print("Use ↑/↓ arrows to navigate prompt history.")
        print("Type 'quit' or 'exit' to stop, 'help' for guidance.\n")

        # Clear history at start of interactive session (unless continuing from initial prompt)
        if clear_history:
            self.clear_history()

        while True:
            try:
                user_input = input("\nYou> ").strip()
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

            if user_input.lower() == 'plan':
                self._print_plan()
                continue

            # Execute the prompt
            response = self.run_prompt(user_input)
            print(f"\nModel> {response}")

    def _print_help(self) -> None:
        """Print help information."""
        print("""
Commands:
  help    - Show this help message
  tools   - List available tools
  reset   - Clear conversation history
  history - Show full conversation history
  plan    - Show current plan status
  quit    - Exit the client

When the model tries to use a tool, you'll see a permission prompt:
  [y]es     - Allow this execution
  [n]o      - Deny this execution
  [a]lways  - Allow and remember for this session
  [never]   - Deny and block for this session
  [once]    - Allow just this once

Example prompts:
  - "List files in the current directory"
  - "Show me the git status"
  - "What is the current date and time?"

Multi-turn conversation:
  The model remembers previous exchanges in this session.
  Use 'reset' to start a fresh conversation.

Keyboard shortcuts:
  ↑/↓       - Navigate through prompt history
  ←/→       - Move cursor within line
  Ctrl+A/E  - Jump to start/end of line
""")

    def _print_tools(self) -> None:
        """Print available tools."""
        print("\nAvailable tools:")
        for decl in self.registry.get_exposed_declarations():
            print(f"  - {decl.name}: {decl.description}")
        print()

    def _print_plan(self) -> None:
        """Print current plan status."""
        if not self.todo_plugin:
            print("\n[Plan tracking not available]")
            return

        plan = self.todo_plugin.get_current_plan()
        if not plan:
            print("\n[No active plan]")
            return

        progress = plan.get_progress()
        print(f"\n{'=' * 50}")
        print(f"  Plan: {plan.title}")
        print(f"  Status: {plan.status.value}")
        print(f"  Progress: {progress['completed']}/{progress['total']} ({progress['percent']:.0f}%)")
        print(f"{'=' * 50}")

        for step in sorted(plan.steps, key=lambda s: s.sequence):
            status_icons = {
                'pending': '○',
                'in_progress': '●',
                'completed': '✓',
                'failed': '✗',
                'skipped': '⊘',
            }
            icon = status_icons.get(step.status.value, '?')
            print(f"  {icon} {step.sequence}. {step.description} [{step.status.value}]")
            if step.result:
                print(f"      → {step.result}")
            if step.error:
                print(f"      ✗ {step.error}")

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
            role = getattr(content, 'role', 'unknown')
            parts = getattr(content, 'parts', [])

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
                next_role = getattr(next_content, 'role', 'unknown')
                next_parts = getattr(next_content, 'parts', [])
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
            print(f"\n{response}")
        elif args.initial_prompt:
            # Initial prompt mode - run prompt then continue interactively
            readline.add_history(args.initial_prompt)  # Add to history for ↑ recall
            response = client.run_prompt(args.initial_prompt)
            print(f"\nModel> {response}")
            client.run_interactive(clear_history=False)
        else:
            # Interactive mode
            client.run_interactive()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
