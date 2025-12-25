"""Anthropic authentication plugin.

Provides user commands for OAuth authentication with Claude Pro/Max subscriptions.
This allows users to authenticate via browser and use their subscription instead
of API credits.

Commands:
    anthropic-auth login        - Open browser for OAuth authentication
    anthropic-auth code <code>  - Complete login with authorization code
    anthropic-auth logout       - Clear stored OAuth tokens
    anthropic-auth status       - Show current authentication status
"""

import threading
import webbrowser
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..base import (
    CommandCompletion,
    CommandParameter,
    ToolPlugin,
    UserCommand,
)
from ..model_provider.types import ToolSchema

# Type alias for output callback: (source, text, mode) -> None
OutputCallback = Callable[[str, str, str], None]


class AnthropicAuthPlugin:
    """Plugin for Anthropic OAuth authentication."""

    def __init__(self):
        """Initialize the plugin."""
        self._output_callback: Optional[OutputCallback] = None
        # Store pending auth state for two-step flow
        self._pending_code_verifier: Optional[str] = None
        self._pending_state: Optional[str] = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "anthropic_auth"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin."""
        pass

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set the output callback for real-time output during commands."""
        self._output_callback = callback

    def _emit(self, text: str, mode: str = "write") -> None:
        """Emit output through the callback if available."""
        if self._output_callback:
            self._output_callback("anthropic_auth", text, mode)

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return empty list - this plugin only provides user commands."""
        return []

    def get_executors(self) -> Dict[str, Any]:
        """Return executor for the user command."""
        return {
            "anthropic-auth": lambda args: self.execute_user_command("anthropic-auth", args),
        }

    def get_system_instructions(self) -> Optional[str]:
        """No system instructions for this plugin."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """User commands don't need permission approval."""
        return ["anthropic-auth"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for authentication."""
        return [
            UserCommand(
                name="anthropic-auth",
                description="Manage Anthropic OAuth authentication (login, logout, status)",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: login, logout, status, or code <auth_code>",
                        required=True,
                        capture_rest=True,  # Capture "code ABC123" as single string
                    ),
                ],
            ),
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Provide autocompletion for command arguments."""
        if command != "anthropic-auth":
            return []

        actions = [
            CommandCompletion("login", "Open browser to authenticate with Claude Pro/Max"),
            CommandCompletion("code", "Complete login with authorization code from browser"),
            CommandCompletion("logout", "Clear stored OAuth tokens"),
            CommandCompletion("status", "Show current authentication status"),
        ]

        if not args:
            return actions

        if len(args) == 1:
            partial = args[0].lower()
            return [a for a in actions if a.value.startswith(partial)]

        return []

    def execute_user_command(self, command: str, args: Dict[str, Any]) -> str:
        """Execute a user command."""
        if command != "anthropic-auth":
            return f"Unknown command: {command}"

        # Get raw action string (don't lowercase - auth codes are case-sensitive)
        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()

        # Handle "code <auth_code>" - extract code from raw string (preserve case)
        if action_lower.startswith("code "):
            auth_code = raw_action[5:].strip()
            return self._cmd_code(auth_code)
        elif action_lower == "code":
            self._emit("Usage: anthropic-auth code <authorization_code>\n")
            return ""
        elif action_lower == "login":
            return self._cmd_login()
        elif action_lower == "logout":
            return self._cmd_logout()
        elif action_lower == "status":
            return self._cmd_status()
        else:
            self._emit(
                f"Unknown action: '{raw_action}'\n\n"
                "Available actions:\n"
                "  login       - Open browser to authenticate with Claude Pro/Max\n"
                "  code <code> - Complete login with authorization code\n"
                "  logout      - Clear stored OAuth tokens\n"
                "  status      - Show current authentication status\n"
            )
            return ""

    def _cmd_login(self) -> str:
        """Handle the login command - step 1: open browser."""
        from ..model_provider.anthropic.oauth import build_auth_url

        # Build auth URL and store code_verifier and state for step 2
        auth_url, code_verifier, state = build_auth_url()
        self._pending_code_verifier = code_verifier
        self._pending_state = state

        # Emit to output panel
        self._emit("Opening browser for authentication...\n\n")
        self._emit(f"If browser doesn't open, visit:\n{auth_url}\n\n")
        self._emit("After authenticating, copy the authorization code and run:\n")
        self._emit("  anthropic-auth code <paste_code_here>\n")

        # Open browser in background thread (fire and forget)
        threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()

        return ""

    def _cmd_code(self, auth_code: str) -> str:
        """Handle the code command - step 2: exchange code for tokens."""
        from ..model_provider.anthropic.oauth import (
            exchange_code_for_tokens,
            save_tokens,
        )

        if not self._pending_code_verifier or not self._pending_state:
            self._emit(
                "✗ No pending login found.\n\n"
                "Please run 'anthropic-auth login' first to start the OAuth flow.\n"
            )
            return ""

        # Strip URL fragment if user copied the whole callback URL or code with fragment
        # The callback page shows: code#state, but we only need the code part
        if "#" in auth_code:
            auth_code = auth_code.split("#")[0]

        self._emit("Exchanging authorization code for tokens...\n")

        try:
            tokens = exchange_code_for_tokens(
                auth_code, self._pending_code_verifier, self._pending_state
            )
            save_tokens(tokens)

            # Clear pending state
            self._pending_code_verifier = None
            self._pending_state = None

            expires_at = datetime.fromtimestamp(tokens.expires_at)
            self._emit(
                "✓ Successfully authenticated with Claude Pro/Max subscription.\n\n"
                f"Access token expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Note: You may need to restart the session for the new tokens to take effect.\n"
            )
            return ""
        except Exception as e:
            self._emit(f"✗ Authentication failed: {e}\n")
            return ""

    def _cmd_logout(self) -> str:
        """Handle the logout command."""
        try:
            from ..model_provider.anthropic.oauth import clear_tokens, load_tokens

            tokens = load_tokens()
            if not tokens:
                self._emit("No OAuth tokens found. Already logged out.\n")
                return ""

            clear_tokens()
            self._emit(
                "✓ OAuth tokens cleared.\n\n"
                "You will need to use an API key (ANTHROPIC_API_KEY) or "
                "run 'anthropic-auth login' to re-authenticate.\n"
            )
            return ""
        except Exception as e:
            self._emit(f"✗ Failed to clear tokens: {e}\n")
            return ""

    def _cmd_status(self) -> str:
        """Handle the status command."""
        try:
            from ..model_provider.anthropic.oauth import load_tokens
            from ..model_provider.anthropic.env import resolve_api_key, resolve_oauth_token

            lines = ["Anthropic Authentication Status", "=" * 35, ""]

            # Check PKCE OAuth tokens
            tokens = load_tokens()
            if tokens:
                expires_at = datetime.fromtimestamp(tokens.expires_at)
                if tokens.is_expired:
                    lines.append("PKCE OAuth: ✗ Expired")
                    lines.append(f"  Expired at: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    lines.append("  Run 'anthropic-auth login' to re-authenticate")
                else:
                    lines.append("PKCE OAuth: ✓ Active")
                    lines.append(f"  Token expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    lines.append(f"  Access token: {tokens.access_token[:20]}...")
            else:
                lines.append("PKCE OAuth: Not configured")
                lines.append("  Run 'anthropic-auth login' to authenticate")

            lines.append("")

            # Check env var OAuth token
            oauth_token = resolve_oauth_token()
            if oauth_token:
                lines.append(f"Env OAuth Token: ✓ Set ({oauth_token[:15]}...)")
            else:
                lines.append("Env OAuth Token: Not set")
                lines.append("  Set ANTHROPIC_AUTH_TOKEN or CLAUDE_CODE_OAUTH_TOKEN")

            lines.append("")

            # Check API key
            api_key = resolve_api_key()
            if api_key:
                lines.append(f"API Key: ✓ Set ({api_key[:15]}...)")
            else:
                lines.append("API Key: Not set")
                lines.append("  Set ANTHROPIC_API_KEY for API credit usage")

            lines.append("")
            lines.append("Priority: PKCE OAuth > Env OAuth Token > API Key")

            self._emit("\n".join(lines) + "\n")
            return ""

        except Exception as e:
            self._emit(f"✗ Failed to check status: {e}\n")
            return ""


def create_plugin() -> AnthropicAuthPlugin:
    """Factory function for plugin discovery."""
    return AnthropicAuthPlugin()
