#!/usr/bin/env python3
"""Simple interactive console client demonstrating askPermission plugin behavior.

This client allows users to enter task descriptions, sends them to the model,
and shows interactive permission prompts in the terminal when tools are invoked.
"""

import os
import sys
import pathlib

# Add project root to path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Optional
from dotenv import load_dotenv

from shared import (
    genai,
    types,
    ToolExecutor,
    run_function_call_loop,
    TokenLedger,
    PluginRegistry,
    PermissionPlugin,
    active_cert_bundle,
)


class InteractiveClient:
    """Simple interactive console client with permission prompts."""

    def __init__(self, env_file: str = ".env", verbose: bool = True):
        self.verbose = verbose
        self.env_file = env_file
        self.client: Optional[genai.Client] = None
        self.model_name: Optional[str] = None
        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Optional[PermissionPlugin] = None
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
        self.model_name = os.environ["MODEL_NAME"]

        # Initialize Vertex AI client
        self.log(f"[client] Connecting to Vertex AI (project={project_id}, location={location})")
        try:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
            self.log(f"[client] Using model: {self.model_name}")
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

        # Initialize permission plugin with console actor
        self.log("[client] Initializing permission plugin with console actor...")
        self.permission_plugin = PermissionPlugin()
        self.permission_plugin.initialize({
            "actor_type": "console",
            "policy": {
                "default_policy": "ask",  # Ask for everything by default
                "whitelist": [],
                "blacklist": [],
            }
        })
        self.log("[client] Permission plugin ready - all tool calls will require approval")

        return True

    def run_prompt(self, prompt: str) -> str:
        """Execute a prompt and return the model's response.

        Tool calls will trigger interactive permission prompts.
        """
        if not self.client or not self.model_name:
            return "Error: Client not initialized"

        # Build executor with all exposed tools
        executor = ToolExecutor(ledger=self.ledger)

        # Register tool executors from plugins
        for name, fn in self.registry.get_exposed_executors().items():
            executor.register(name, fn)

        # Set permission plugin for enforcement
        executor.set_permission_plugin(self.permission_plugin)

        # Register askPermission tool for proactive checks
        for name, fn in self.permission_plugin.get_executors().items():
            executor.register(name, fn)

        # Build tool declarations
        all_decls = self.registry.get_exposed_declarations()
        all_decls.extend(self.permission_plugin.get_function_declarations())
        tool_decl = types.Tool(function_declarations=all_decls) if all_decls else None

        self.log(f"\n[client] Sending prompt to model...")
        self.log(f"[client] Available tools: {[d.name for d in all_decls]}")

        try:
            result = run_function_call_loop(
                self.client,
                self.model_name,
                [types.Part.from_text(text=prompt)],
                declared_tools=tool_decl,
                executor=executor,
                ledger=self.ledger,
                max_turns=20,
                trace=False
            )

            turns = result.get('turns', 1)
            self.log(f"\n[client] Completed in {turns} turn(s)")

            return result.get('text', '(No response text)')

        except KeyboardInterrupt:
            return "\n[Interrupted by user]"
        except Exception as e:
            return f"Error during execution: {e}"

    def run_interactive(self) -> None:
        """Run the interactive prompt loop."""
        print("\n" + "=" * 60)
        print("  Simple Interactive Client with Permission Prompts")
        print("=" * 60)
        print("\nEnter task descriptions for the model to execute.")
        print("Tool calls will prompt for your approval.")
        print("Type 'quit' or 'exit' to stop, 'help' for guidance.\n")

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

            # Execute the prompt
            response = self.run_prompt(user_input)
            print(f"\nModel> {response}")

    def _print_help(self) -> None:
        """Print help information."""
        print("""
Commands:
  help   - Show this help message
  tools  - List available tools
  quit   - Exit the client

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
""")

    def _print_tools(self) -> None:
        """Print available tools."""
        print("\nAvailable tools:")
        for decl in self.registry.get_exposed_declarations():
            print(f"  - {decl.name}: {decl.description}")
        print()

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
    args = parser.parse_args()

    client = InteractiveClient(
        env_file=args.env_file,
        verbose=not args.quiet
    )

    if not client.initialize():
        sys.exit(1)

    try:
        if args.prompt:
            # Single prompt mode
            response = client.run_prompt(args.prompt)
            print(f"\n{response}")
        else:
            # Interactive mode
            client.run_interactive()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
