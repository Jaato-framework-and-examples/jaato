"""GitHub authentication plugin.

Provides user commands for OAuth authentication with GitHub using the device code flow.
This enables GitHub Copilot/Models access without manually creating a PAT.

Commands:
    github-auth login        - Start device code OAuth flow
    github-auth poll         - Poll for authorization (after entering code)
    github-auth logout       - Clear stored OAuth tokens
    github-auth status       - Show current authentication status
"""

import threading
import webbrowser
from typing import Any, Callable, Dict, List, Optional

from ..base import (
    CommandCompletion,
    CommandParameter,
    HelpLines,
    ToolPlugin,
    UserCommand,
)
from ..model_provider.types import ToolSchema

# Type alias for output callback: (source, text, mode) -> None
OutputCallback = Callable[[str, str, str], None]


class GitHubAuthPlugin:
    """Plugin for GitHub device code OAuth authentication."""

    def __init__(self):
        """Initialize the plugin."""
        self._output_callback: Optional[OutputCallback] = None
        # Store pending device code for polling
        self._pending_user_code: Optional[str] = None
        self._pending_verification_uri: Optional[str] = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "github_auth"

    @property
    def provider_name(self) -> str:
        """Return the provider name this auth plugin serves."""
        return "github_models"

    @property
    def provider_display_name(self) -> str:
        """Return human-readable provider name."""
        return "GitHub Models"

    def get_default_models(self) -> List[Dict[str, str]]:
        """Return default models available for this provider."""
        return [
            {"name": "github/gpt-4o", "description": "GPT-4o — fast and capable"},
            {"name": "github/gpt-4o-mini", "description": "GPT-4o mini — lightweight"},
            {"name": "github/o3-mini", "description": "O3 mini — reasoning model"},
        ]

    def verify_credentials(self) -> bool:
        """Check if valid credentials exist after authentication."""
        try:
            from ..model_provider.github_models.oauth import load_tokens
            from ..model_provider.github_models.env import resolve_token
            tokens = load_tokens()
            if tokens and tokens.access_token:
                return True
            return bool(resolve_token())
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
            self._output_callback("github_auth", text, mode)

    def _emit_progress(self, text: str) -> None:
        """Emit progress output that updates the same line."""
        if self._output_callback:
            self._output_callback("github_auth", text, "update")

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return empty list - this plugin only provides user commands."""
        return []

    def get_executors(self) -> Dict[str, Any]:
        """Return executor for the user command."""
        return {
            "github-auth": lambda args: self.execute_user_command("github-auth", args),
        }

    def get_system_instructions(self) -> Optional[str]:
        """No system instructions for this plugin."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """User commands don't need permission approval."""
        return ["github-auth"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for authentication."""
        return [
            UserCommand(
                name="github-auth",
                description="Manage GitHub OAuth authentication (login, logout, status)",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: login, poll, logout, or status",
                        required=True,
                        capture_rest=True,
                    ),
                ],
            ),
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Provide autocompletion for command arguments."""
        if command != "github-auth":
            return []

        actions = [
            CommandCompletion("login", "Start device code OAuth flow with GitHub"),
            CommandCompletion("poll", "Poll for authorization after entering code"),
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
        if command != "github-auth":
            return f"Unknown command: {command}"

        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()

        if action_lower == "login":
            return self._cmd_login()
        elif action_lower == "poll":
            return self._cmd_poll()
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
                "  login   - Start device code OAuth flow\n"
                "  poll    - Poll for authorization after entering code\n"
                "  logout  - Clear stored OAuth tokens\n"
                "  status  - Show current authentication status\n"
                "  help    - Show detailed help\n"
            )
            return ""

    def _cmd_help(self) -> HelpLines:
        """Return detailed help text for pager display."""
        return HelpLines(lines=[
            ("GitHub Auth Command", "bold"),
            ("", ""),
            ("Authenticate with GitHub using OAuth device code flow. This enables access", ""),
            ("to GitHub Models API without requiring a personal access token.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    github-auth <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    login             Start the device code OAuth flow", "dim"),
            ("                      Opens browser to github.com/login/device", "dim"),
            ("                      Polls automatically for authorization", "dim"),
            ("", ""),
            ("    poll              Manually poll for authorization", "dim"),
            ("                      Use if automatic polling was interrupted", "dim"),
            ("", ""),
            ("    logout            Clear stored OAuth tokens", "dim"),
            ("                      Removes credentials from keychain/keyring", "dim"),
            ("", ""),
            ("    status            Show current authentication status", "dim"),
            ("                      Displays token type and expiration", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("AUTHENTICATION FLOW", "bold"),
            ("    1. Run 'github-auth login'", ""),
            ("    2. Browser opens to github.com/login/device", ""),
            ("    3. Enter the displayed code (e.g., ABCD-1234)", ""),
            ("    4. Authorize the application", ""),
            ("    5. Token is automatically saved", ""),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    github-auth login         Start OAuth login", "dim"),
            ("    github-auth status        Check if authenticated", "dim"),
            ("    github-auth logout        Clear credentials", "dim"),
            ("", ""),
            ("TOKEN STORAGE", "bold"),
            ("    Tokens are securely stored in your system keychain:", ""),
            ("    - macOS: Keychain Access", "dim"),
            ("    - Linux: Secret Service (GNOME Keyring, KWallet)", "dim"),
            ("    - Windows: Credential Manager", "dim"),
            ("", ""),
            ("ALTERNATIVE: PERSONAL ACCESS TOKEN", "bold"),
            ("    If OAuth doesn't work, you can use a PAT:", ""),
            ("    1. Go to github.com/settings/tokens", "dim"),
            ("    2. Create token with 'models:read' scope", "dim"),
            ("    3. Set GITHUB_TOKEN environment variable", "dim"),
            ("", ""),
            ("ENVIRONMENT VARIABLES", "bold"),
            ("    GITHUB_TOKEN                  Personal access token (alternative to OAuth)", "dim"),
            ("    JAATO_GITHUB_ORGANIZATION     Organization for billing attribution", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - OAuth tokens expire and auto-refresh when needed", "dim"),
            ("    - Device code flow works without exposing secrets", "dim"),
            ("    - Uses GitHub Copilot's OAuth client ID", "dim"),
        ])

    def _cmd_login(self) -> str:
        """Handle the login command - start device code flow."""
        from ..model_provider.github_models.oauth import (
            start_device_flow,
            complete_device_flow,
            save_tokens,
        )

        try:
            # Start device code flow
            self._emit("Requesting device code from GitHub...\n\n")
            device_response = start_device_flow()

            # Store for potential manual polling
            self._pending_user_code = device_response.user_code
            self._pending_verification_uri = device_response.verification_uri

            # Display instructions
            self._emit("=" * 50 + "\n")
            self._emit(f"  Visit: {device_response.verification_uri}\n")
            self._emit(f"  Enter code: {device_response.user_code}\n")
            self._emit("=" * 50 + "\n\n")

            # Open browser in background
            threading.Thread(
                target=webbrowser.open,
                args=(device_response.verification_uri,),
                daemon=True
            ).start()

            self._emit("Browser opened. Waiting for authorization...\n")
            self._emit("(This may take a moment after you enter the code)\n\n")

            # Poll for token
            def on_message(msg: str) -> None:
                self._emit(f"{msg}\n")

            def on_progress(msg: str) -> None:
                self._emit_progress(msg)

            tokens = complete_device_flow(on_message=on_message, on_progress=on_progress)

            if tokens:
                # Mask the token for display
                masked = f"{tokens.access_token[:10]}...{tokens.access_token[-4:]}"
                self._emit(
                    "\n"
                    "=" * 50 + "\n"
                    "  Successfully authenticated with GitHub!\n"
                    "=" * 50 + "\n\n"
                    f"Access token: {masked}\n"
                    f"Scope: {tokens.scope or '(default)'}\n\n"
                    "Note: You may need to restart the session for the new token to take effect.\n"
                )
            else:
                self._emit(
                    "\n"
                    "Authentication failed or timed out.\n"
                    "Please try 'github-auth login' again.\n"
                )

            return ""

        except Exception as e:
            self._emit(f"Error: {e}\n")
            return ""

    def _cmd_poll(self) -> str:
        """Handle the poll command - manually poll for authorization."""
        from ..model_provider.github_models.oauth import complete_device_flow

        try:
            def on_message(msg: str) -> None:
                self._emit(f"{msg}\n")

            def on_progress(msg: str) -> None:
                self._emit_progress(msg)

            self._emit("Polling for authorization...\n")
            tokens = complete_device_flow(on_message=on_message, on_progress=on_progress)

            if tokens:
                masked = f"{tokens.access_token[:10]}...{tokens.access_token[-4:]}"
                self._emit(
                    "\n"
                    "Successfully authenticated with GitHub!\n\n"
                    f"Access token: {masked}\n"
                    f"Scope: {tokens.scope or '(default)'}\n\n"
                    "Note: You may need to restart the session for the new token to take effect.\n"
                )
            else:
                self._emit("No pending authorization or authentication failed.\n")

            return ""

        except Exception as e:
            self._emit(f"Error: {e}\n")
            return ""

    def _cmd_logout(self) -> str:
        """Handle the logout command."""
        try:
            from ..model_provider.github_models.oauth import clear_tokens, load_tokens

            tokens = load_tokens()
            if not tokens:
                self._emit("No OAuth tokens found. Already logged out.\n")
                return ""

            clear_tokens()
            self._emit(
                "OAuth tokens cleared.\n\n"
                "You will need to use GITHUB_TOKEN environment variable or "
                "run 'github-auth login' to re-authenticate.\n"
            )
            return ""
        except Exception as e:
            self._emit(f"Failed to clear tokens: {e}\n")
            return ""

    def _cmd_status(self) -> str:
        """Handle the status command."""
        try:
            from ..model_provider.github_models.oauth import (
                load_tokens,
                load_copilot_token,
            )
            from ..model_provider.github_models.env import resolve_token
            import time

            lines = ["GitHub Authentication Status", "=" * 35, ""]

            # Check stored OAuth tokens
            tokens = load_tokens()
            if tokens:
                masked = f"{tokens.access_token[:10]}...{tokens.access_token[-4:]}"
                lines.append("Device Code OAuth: Active")
                lines.append(f"  Access token: {masked}")
                lines.append(f"  Scope: {tokens.scope or '(default)'}")
            else:
                lines.append("Device Code OAuth: Not configured")
                lines.append("  Run 'github-auth login' to authenticate")

            lines.append("")

            # Check Copilot token
            copilot_token = load_copilot_token()
            if copilot_token:
                masked_copilot = f"{copilot_token.token[:10]}...{copilot_token.token[-4:]}"
                if copilot_token.is_expired():
                    lines.append("Copilot API Token: Expired (will refresh on next use)")
                else:
                    remaining = int(copilot_token.expires_at - time.time())
                    mins = remaining // 60
                    lines.append(f"Copilot API Token: Valid ({mins} min remaining)")
                lines.append(f"  Token: {masked_copilot}")
            else:
                if tokens:
                    lines.append("Copilot API Token: Not acquired yet")
                    lines.append("  Will be exchanged on first API call")
                else:
                    lines.append("Copilot API Token: N/A (no OAuth token)")

            lines.append("")

            # Check environment variable
            env_token = resolve_token()
            if env_token:
                masked_env = f"{env_token[:4]}...{env_token[-4:]}" if len(env_token) > 8 else "***"
                lines.append(f"GITHUB_TOKEN: Set ({masked_env})")
            else:
                lines.append("GITHUB_TOKEN: Not set")

            lines.append("")

            # Show priority
            if tokens:
                lines.append("Active auth: Device Code OAuth (Copilot API)")
            elif env_token:
                lines.append("Active auth: GITHUB_TOKEN environment variable")
            else:
                lines.append("Active auth: None - please authenticate")

            lines.append("")
            lines.append("Priority: Device Code OAuth > GITHUB_TOKEN env var")

            self._emit("\n".join(lines) + "\n")
            return ""

        except Exception as e:
            self._emit(f"Failed to check status: {e}\n")
            return ""


def create_plugin() -> GitHubAuthPlugin:
    """Factory function for plugin discovery."""
    return GitHubAuthPlugin()
