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
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from ..base import (
    CommandCompletion,
    CommandParameter,
    HelpLines,
    ToolPlugin,
    TRAIT_AUTH_PROVIDER,
    UserCommand,
)
from ..model_provider.types import ToolSchema

# Type alias for output callback: (source, text, mode) -> None
OutputCallback = Callable[[str, str, str], None]


class AnthropicAuthPlugin:
    """Plugin for Anthropic OAuth authentication.

    Declares the ``TRAIT_AUTH_PROVIDER`` trait so the server can
    auto-discover this plugin when the Anthropic provider needs credentials.
    The ``provider_name`` property identifies which provider this plugin
    authenticates for.
    """

    plugin_traits: FrozenSet[str] = frozenset({TRAIT_AUTH_PROVIDER})

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

    @property
    def provider_name(self) -> str:
        """Return the provider name this auth plugin serves."""
        return "anthropic"

    @property
    def provider_display_name(self) -> str:
        """Return human-readable provider name."""
        return "Anthropic (Claude)"

    @property
    def credential_env_vars(self) -> List[str]:
        """Return env var names used for credentials by this provider."""
        return ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]

    def get_default_models(self) -> List[Dict[str, str]]:
        """Return default models available for this provider."""
        return [
            {"name": "anthropic/claude-sonnet-4-5-20250929", "description": "Claude Sonnet 4.5 — balanced speed and capability"},
            {"name": "anthropic/claude-opus-4-6", "description": "Claude Opus 4.6 — most capable"},
            {"name": "anthropic/claude-haiku-4-5-20251001", "description": "Claude Haiku 4.5 — fastest"},
        ]

    def verify_credentials(self) -> bool:
        """Check if valid credentials exist after authentication."""
        try:
            from ..model_provider.anthropic.oauth import load_tokens
            from ..model_provider.anthropic.env import resolve_api_key, resolve_oauth_token
            tokens = load_tokens()
            if tokens and not tokens.is_expired:
                return True
            return bool(resolve_oauth_token() or resolve_api_key())
        except Exception:
            return False

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
            CommandCompletion("help", "Show detailed help for this command"),
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
        elif action_lower == "help":
            return self._cmd_help()
        else:
            self._emit(
                f"Unknown action: '{raw_action}'\n\n"
                "Available actions:\n"
                "  login       - Open browser to authenticate with Claude Pro/Max\n"
                "  code <code> - Complete login with authorization code\n"
                "  logout      - Clear stored OAuth tokens\n"
                "  status      - Show current authentication status\n"
                "  help        - Show detailed help\n"
            )
            return ""

    def _cmd_help(self) -> HelpLines:
        """Return detailed help text for pager display."""
        return HelpLines(lines=[
            ("Anthropic Auth Command", "bold"),
            ("", ""),
            ("Authenticate with Anthropic using OAuth PKCE flow. This enables access to", ""),
            ("Claude API using your Claude Pro or Max subscription without API credits.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    anthropic-auth <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    login             Open browser to authenticate with Anthropic", "dim"),
            ("                      Generates PKCE challenge and opens auth URL", "dim"),
            ("", ""),
            ("    code <auth_code>  Complete login with the authorization code", "dim"),
            ("                      Paste the code shown after browser authorization", "dim"),
            ("", ""),
            ("    logout            Clear stored OAuth tokens", "dim"),
            ("                      Removes credentials from keychain/keyring", "dim"),
            ("", ""),
            ("    status            Show current authentication status", "dim"),
            ("                      Displays token type and account info", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("AUTHENTICATION FLOW", "bold"),
            ("    1. Run 'anthropic-auth login'", ""),
            ("    2. Browser opens to Anthropic's auth page", ""),
            ("    3. Sign in with your Anthropic account", ""),
            ("    4. Copy the authorization code shown", ""),
            ("    5. Run 'anthropic-auth code <paste_code_here>'", ""),
            ("    6. Token is saved and ready to use", ""),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    anthropic-auth login              Start OAuth login", "dim"),
            ("    anthropic-auth code ABC123xyz     Complete login with code", "dim"),
            ("    anthropic-auth status             Check authentication status", "dim"),
            ("    anthropic-auth logout             Clear stored credentials", "dim"),
            ("", ""),
            ("TOKEN TYPES", "bold"),
            ("    OAuth Token (sk-ant-oat01-...)    From PKCE OAuth flow", "dim"),
            ("                                      Uses Pro/Max subscription", "dim"),
            ("    API Key (sk-ant-api03-...)        From console.anthropic.com", "dim"),
            ("                                      Uses API credits", "dim"),
            ("", ""),
            ("TOKEN STORAGE", "bold"),
            ("    Tokens are securely stored in your system keychain:", ""),
            ("    - macOS: Keychain Access", "dim"),
            ("    - Linux: Secret Service (GNOME Keyring, KWallet)", "dim"),
            ("    - Windows: Credential Manager", "dim"),
            ("", ""),
            ("ENVIRONMENT VARIABLES", "bold"),
            ("    ANTHROPIC_AUTH_TOKEN    OAuth token (from 'claude setup-token')", "dim"),
            ("    ANTHROPIC_API_KEY       API key (uses credits, not subscription)", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - OAuth tokens use your Pro/Max subscription (no API credits)", "dim"),
            ("    - Tokens auto-refresh when expired", "dim"),
            ("    - Alternative: Run 'claude setup-token' from Claude Code CLI", "dim"),
        ])

    def _cmd_login(self) -> str:
        """Handle the login command - step 1: open browser."""
        from ..model_provider.anthropic.oauth import build_auth_url, save_pending_auth

        # Build auth URL and store state for step 2 (both in-memory and file-based)
        auth_url, code_verifier, state = build_auth_url()
        self._pending_code_verifier = code_verifier
        self._pending_state = state
        save_pending_auth(code_verifier, state)  # Also save to file for cross-process access

        # Emit to output panel
        self._emit("Opening browser for authentication...\n\n")
        self._emit(f"If browser doesn't open, visit:\n{auth_url}\n\n")
        self._emit("After authenticating, copy the authorization code and run:\n  anthropic-auth code <paste_code_here>\n")

        # Open browser in background thread (fire and forget)
        threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()

        return ""

    def _cmd_code(self, auth_code: str) -> str:
        """Handle the code command - step 2: exchange code for tokens."""
        from ..model_provider.anthropic.oauth import (
            load_pending_auth,
            complete_interactive_login,
        )

        # Try in-memory state first, then fall back to file-based state
        code_verifier = self._pending_code_verifier
        state = self._pending_state

        if not code_verifier or not state:
            code_verifier, state = load_pending_auth()

        if not code_verifier or not state:
            self._emit(
                "✗ No pending login found.\n\n"
                "Please run 'anthropic-auth login' first to start the OAuth flow.\n"
            )
            return ""

        self._emit("Exchanging authorization code for tokens...\n")

        # Use shared function for token exchange
        def on_error(msg: str) -> None:
            self._emit(f"✗ {msg}\n")

        tokens = complete_interactive_login(auth_code, on_message=on_error)

        if tokens:
            # Clear in-memory state
            self._pending_code_verifier = None
            self._pending_state = None

            expires_at = datetime.fromtimestamp(tokens.expires_at)
            self._emit(
                "✓ Successfully authenticated with Claude Pro/Max subscription.\n\n"
                f"Access token expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Note: You may need to restart the session for the new tokens to take effect.\n"
            )
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
