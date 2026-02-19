"""Antigravity authentication plugin.

Provides user commands for Google OAuth authentication with Antigravity backend.
This allows users to authenticate via browser and access Gemini 3 and Claude models
through Google's Antigravity IDE backend.

Commands:
    antigravity-auth login        - Open browser for Google OAuth authentication
    antigravity-auth code <code>  - Complete login with authorization code
    antigravity-auth logout       - Clear stored OAuth tokens
    antigravity-auth status       - Show current authentication status
    antigravity-auth accounts     - List all authenticated accounts
"""

import threading
import webbrowser
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from jaato_sdk.plugins.base import (
    CommandCompletion,
    CommandParameter,
    HelpLines,
    ToolPlugin,
    TRAIT_AUTH_PROVIDER,
    UserCommand,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema

# Type alias for output callback: (source, text, mode) -> None
OutputCallback = Callable[[str, str, str], None]


class AntigravityAuthPlugin:
    """Plugin for Antigravity OAuth authentication.

    Declares the ``TRAIT_AUTH_PROVIDER`` trait so the server can
    auto-discover this plugin when the Antigravity provider needs credentials.
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
        return "antigravity_auth"

    @property
    def provider_name(self) -> str:
        """Return the provider name this auth plugin serves."""
        return "antigravity"

    @property
    def provider_display_name(self) -> str:
        """Return human-readable provider name."""
        return "Google Antigravity"

    @property
    def credential_env_vars(self) -> List[str]:
        """Return env var names used for credentials by this provider.

        Antigravity uses OAuth only — no env var for credentials.
        """
        return []

    def get_default_models(self) -> List[Dict[str, str]]:
        """Return default models available for this provider."""
        return [
            {"name": "antigravity-gemini-3-pro", "description": "Gemini 3 Pro via Antigravity"},
            {"name": "antigravity-gemini-3-flash", "description": "Gemini 3 Flash via Antigravity"},
            {"name": "antigravity-claude-sonnet-4-5", "description": "Claude Sonnet 4.5 via Antigravity"},
        ]

    def verify_credentials(self) -> bool:
        """Check if valid credentials exist after authentication."""
        try:
            from ..model_provider.antigravity.oauth import load_tokens
            tokens = load_tokens()
            return bool(tokens)
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
            self._output_callback("antigravity_auth", text, mode)

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return empty list - this plugin only provides user commands."""
        return []

    def get_executors(self) -> Dict[str, Any]:
        """Return executor for the user command."""
        return {
            "antigravity-auth": lambda args: self.execute_user_command("antigravity-auth", args),
        }

    def get_system_instructions(self) -> Optional[str]:
        """No system instructions for this plugin."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """User commands don't need permission approval."""
        return ["antigravity-auth"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for authentication."""
        return [
            UserCommand(
                name="antigravity-auth",
                description="Manage Antigravity OAuth authentication (login, logout, status, accounts)",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: login, logout, status, accounts, or code <auth_code>",
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
        if command != "antigravity-auth":
            return []

        actions = [
            CommandCompletion("login", "Open browser to authenticate with Google"),
            CommandCompletion("code", "Complete login with authorization code from browser"),
            CommandCompletion("logout", "Clear stored OAuth tokens for all accounts"),
            CommandCompletion("status", "Show current authentication status"),
            CommandCompletion("accounts", "List all authenticated Google accounts"),
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
        if command != "antigravity-auth":
            return f"Unknown command: {command}"

        # Get raw action string (don't lowercase - auth codes are case-sensitive)
        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()

        # Handle "code <auth_code>" - extract code from raw string (preserve case)
        if action_lower.startswith("code "):
            auth_code = raw_action[5:].strip()
            return self._cmd_code(auth_code)
        elif action_lower == "code":
            self._emit("Usage: antigravity-auth code <authorization_code>\n")
            return ""
        elif action_lower == "login":
            return self._cmd_login()
        elif action_lower == "logout":
            return self._cmd_logout()
        elif action_lower == "status":
            return self._cmd_status()
        elif action_lower == "accounts":
            return self._cmd_accounts()
        elif action_lower == "help":
            return self._cmd_help()
        else:
            self._emit(
                f"Unknown action: '{raw_action}'\n\n"
                "Available actions:\n"
                "  login       - Open browser to authenticate with Google\n"
                "  code <code> - Complete login with authorization code\n"
                "  logout      - Clear stored OAuth tokens\n"
                "  status      - Show current authentication status\n"
                "  accounts    - List all authenticated accounts\n"
                "  help        - Show detailed help\n"
            )
            return ""

    def _cmd_help(self) -> HelpLines:
        """Return detailed help text for pager display."""
        return HelpLines(lines=[
            ("Antigravity Auth Command", "bold"),
            ("", ""),
            ("Authenticate with Google for Antigravity backend access. This enables access", ""),
            ("to Gemini 3 and Claude models through Google's Antigravity IDE backend.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    antigravity-auth <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    login             Open browser to authenticate with Google", "dim"),
            ("                      Generates PKCE challenge and opens Google auth URL", "dim"),
            ("", ""),
            ("    code <auth_code>  Complete login with the authorization code", "dim"),
            ("                      Paste the code shown after browser authorization", "dim"),
            ("", ""),
            ("    logout            Clear stored OAuth tokens for all accounts", "dim"),
            ("                      Removes all credentials from keychain/keyring", "dim"),
            ("", ""),
            ("    status            Show current authentication status", "dim"),
            ("                      Displays active account and token info", "dim"),
            ("", ""),
            ("    accounts          List all authenticated Google accounts", "dim"),
            ("                      Shows which accounts have valid tokens", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("AUTHENTICATION FLOW", "bold"),
            ("    1. Run 'antigravity-auth login'", ""),
            ("    2. Browser opens to Google's auth page", ""),
            ("    3. Sign in with your Google account", ""),
            ("    4. Copy the authorization code shown", ""),
            ("    5. Run 'antigravity-auth code <paste_code_here>'", ""),
            ("    6. Token is saved and ready to use", ""),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    antigravity-auth login              Start OAuth login", "dim"),
            ("    antigravity-auth code ABC123xyz     Complete login with code", "dim"),
            ("    antigravity-auth status             Check authentication status", "dim"),
            ("    antigravity-auth accounts           List all authenticated accounts", "dim"),
            ("    antigravity-auth logout             Clear all credentials", "dim"),
            ("", ""),
            ("MULTI-ACCOUNT SUPPORT", "bold"),
            ("    Multiple Google accounts can be authenticated simultaneously.", ""),
            ("    The system auto-rotates between accounts to distribute quota.", "dim"),
            ("    Each login adds a new account without removing existing ones.", "dim"),
            ("", ""),
            ("AVAILABLE MODELS (Antigravity quota)", "bold"),
            ("    antigravity-gemini-3-pro            Gemini 3 Pro", "dim"),
            ("    antigravity-gemini-3-flash          Gemini 3 Flash", "dim"),
            ("    antigravity-claude-sonnet-4-5       Claude Sonnet 4.5", "dim"),
            ("    antigravity-claude-sonnet-4-5-thinking", "dim"),
            ("                                        Claude Sonnet 4.5 with thinking", "dim"),
            ("", ""),
            ("AVAILABLE MODELS (Gemini CLI quota)", "bold"),
            ("    gemini-2.5-flash                    Gemini 2.5 Flash", "dim"),
            ("    gemini-2.5-pro                      Gemini 2.5 Pro", "dim"),
            ("    gemini-3-flash-preview              Gemini 3 Flash Preview", "dim"),
            ("    gemini-3-pro-preview                Gemini 3 Pro Preview", "dim"),
            ("", ""),
            ("TOKEN STORAGE", "bold"),
            ("    Tokens are securely stored in your system keychain:", ""),
            ("    - macOS: Keychain Access", "dim"),
            ("    - Linux: Secret Service (GNOME Keyring, KWallet)", "dim"),
            ("    - Windows: Credential Manager", "dim"),
            ("", ""),
            ("ENVIRONMENT VARIABLES", "bold"),
            ("    JAATO_ANTIGRAVITY_QUOTA         'antigravity' or 'gemini-cli' (default: antigravity)", "dim"),
            ("    JAATO_ANTIGRAVITY_THINKING_LEVEL  Gemini 3 thinking: minimal/low/medium/high", "dim"),
            ("    JAATO_ANTIGRAVITY_THINKING_BUDGET Claude thinking budget (default: 8192)", "dim"),
            ("    JAATO_ANTIGRAVITY_AUTO_ROTATE   Enable multi-account rotation (default: true)", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - OAuth tokens auto-refresh when expired", "dim"),
            ("    - Auto-rotation distributes load across authenticated accounts", "dim"),
            ("    - Quota is separate between Antigravity and Gemini CLI backends", "dim"),
        ])

    def _cmd_login(self) -> str:
        """Handle the login command - step 1: open browser."""
        from ..model_provider.antigravity.oauth import build_auth_url, save_pending_auth

        # Build auth URL and store state for step 2 (both in-memory and file-based)
        auth_url, code_verifier, state = build_auth_url()
        self._pending_code_verifier = code_verifier
        self._pending_state = state
        save_pending_auth(code_verifier, state)  # Also save to file for cross-process access

        # Emit to output panel
        self._emit("Opening browser for Google authentication...\n\n")
        self._emit(f"If browser doesn't open, visit:\n{auth_url}\n\n")
        self._emit("After authenticating, copy the authorization code and run:\n  antigravity-auth code <paste_code_here>\n")

        # Open browser in background thread (fire and forget)
        threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()

        return ""

    def _cmd_code(self, auth_code: str) -> str:
        """Handle the code command - step 2: exchange code for tokens."""
        from ..model_provider.antigravity.oauth import (
            load_pending_auth,
            complete_interactive_login,
        )

        # Try in-memory state first, then fall back to file-based state
        code_verifier = self._pending_code_verifier
        state = self._pending_state

        if not code_verifier or not state:
            code_verifier, state = load_pending_auth()

        if not code_verifier:
            self._emit(
                "No pending login found.\n\n"
                "Please run 'antigravity-auth login' first to start the OAuth flow.\n"
            )
            return ""

        self._emit("Exchanging authorization code for tokens...\n")

        # Use shared function for token exchange
        def on_message(msg: str) -> None:
            self._emit(f"{msg}\n")

        account = complete_interactive_login(auth_code, on_message=on_message)

        if account:
            # Clear in-memory state
            self._pending_code_verifier = None
            self._pending_state = None

            expires_at = datetime.fromtimestamp(account.tokens.expires_at)
            self._emit(
                f"\nSuccessfully authenticated!\n\n"
                f"Email: {account.email}\n"
                f"Project ID: {account.project_id}\n"
                f"Token expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Note: You may need to restart the session for the new tokens to take effect.\n"
            )
        return ""

    def _cmd_logout(self) -> str:
        """Handle the logout command."""
        try:
            from ..model_provider.antigravity.oauth import clear_accounts, load_accounts

            manager = load_accounts()
            if not manager.accounts:
                self._emit("No accounts found. Already logged out.\n")
                return ""

            num_accounts = len(manager.accounts)
            clear_accounts()
            self._emit(
                f"Cleared {num_accounts} account(s).\n\n"
                "Run 'antigravity-auth login' to re-authenticate.\n"
            )
            return ""
        except Exception as e:
            self._emit(f"Failed to clear accounts: {e}\n")
            return ""

    def _cmd_status(self) -> str:
        """Handle the status command."""
        try:
            from ..model_provider.antigravity.oauth import load_accounts, get_valid_access_token
            from ..model_provider.antigravity.env import resolve_project_id, resolve_endpoint

            lines = ["Antigravity Authentication Status", "=" * 35, ""]

            # Check OAuth accounts
            manager = load_accounts()
            if manager.accounts:
                active = manager.get_active_account()
                total = len(manager.accounts)
                active_count = sum(1 for a in manager.accounts if a.is_active)

                lines.append(f"Accounts: {total} total, {active_count} active")
                lines.append("")

                if active:
                    expires_at = datetime.fromtimestamp(active.tokens.expires_at)
                    if active.tokens.is_expired:
                        lines.append(f"Active Account: {active.email}")
                        lines.append(f"  Status: Token expired")
                        lines.append(f"  Expired at: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        lines.append(f"Active Account: {active.email}")
                        lines.append(f"  Status: Active")
                        lines.append(f"  Project ID: {active.project_id}")
                        lines.append(f"  Token expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                lines.append("Accounts: None configured")
                lines.append("  Run 'antigravity-auth login' to authenticate")

            lines.append("")

            # Check environment overrides
            env_project = resolve_project_id()
            env_endpoint = resolve_endpoint()

            if env_project:
                lines.append(f"Env Project ID: {env_project}")
            if env_endpoint:
                lines.append(f"Env Endpoint: {env_endpoint}")

            if not env_project and not env_endpoint:
                lines.append("Env Overrides: None")

            self._emit("\n".join(lines) + "\n")
            return ""

        except Exception as e:
            self._emit(f"Failed to check status: {e}\n")
            return ""

    def _cmd_accounts(self) -> str:
        """Handle the accounts command - list all authenticated accounts."""
        try:
            from ..model_provider.antigravity.oauth import load_accounts

            manager = load_accounts()

            if not manager.accounts:
                self._emit("No accounts configured.\n\nRun 'antigravity-auth login' to add an account.\n")
                return ""

            lines = ["Authenticated Accounts", "=" * 35, ""]

            for i, account in enumerate(manager.accounts, 1):
                expires_at = datetime.fromtimestamp(account.tokens.expires_at)
                is_current = (i - 1) == manager.current_index

                status_parts = []
                if is_current:
                    status_parts.append("CURRENT")
                if not account.is_active:
                    status_parts.append("DISABLED")
                if account.tokens.is_expired:
                    status_parts.append("EXPIRED")
                elif account.is_rate_limited():
                    status_parts.append("RATE LIMITED")
                else:
                    status_parts.append("ACTIVE")

                status = ", ".join(status_parts)

                lines.append(f"{i}. {account.email}")
                lines.append(f"   Status: {status}")
                lines.append(f"   Project: {account.project_id}")
                lines.append(f"   Expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append("")

            lines.append("Tip: Run 'antigravity-auth login' to add another account for load balancing.")

            self._emit("\n".join(lines) + "\n")
            return ""

        except Exception as e:
            self._emit(f"Failed to list accounts: {e}\n")
            return ""


def create_plugin() -> AntigravityAuthPlugin:
    """Factory function for plugin discovery."""
    return AntigravityAuthPlugin()
