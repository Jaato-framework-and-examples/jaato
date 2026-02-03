"""Zhipu AI (Z.AI) authentication plugin.

Provides user commands for API key authentication with Z.AI GLM Coding Plan.
This allows users to securely store and manage their Z.AI API credentials.

Commands:
    zhipuai-auth login       - Interactively enter and validate API key
    zhipuai-auth logout      - Clear stored API credentials
    zhipuai-auth status      - Show current authentication status
"""

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


class ZhipuAIAuthPlugin:
    """Plugin for Zhipu AI API key authentication."""

    def __init__(self):
        """Initialize the plugin."""
        self._output_callback: Optional[OutputCallback] = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "zhipuai_auth"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin."""
        pass

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set the output callback for real-time output during commands."""
        self._output_callback = callback

    def _emit(self, text: str, mode: str = "write") -> None:
        """Emit output through the callback if available."""
        if self._output_callback:
            self._output_callback("zhipuai_auth", text, mode)

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return empty list - this plugin only provides user commands."""
        return []

    def get_executors(self) -> Dict[str, Any]:
        """Return executor for the user command."""
        return {
            "zhipuai-auth": lambda args: self.execute_user_command("zhipuai-auth", args),
        }

    def get_system_instructions(self) -> Optional[str]:
        """No system instructions for this plugin."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """User commands don't need permission approval."""
        return ["zhipuai-auth"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for authentication."""
        return [
            UserCommand(
                name="zhipuai-auth",
                description="Manage Zhipu AI (Z.AI) authentication (login, logout, status)",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: login, logout, status, or help",
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
        if command != "zhipuai-auth":
            return []

        actions = [
            CommandCompletion("login", "Enter and validate your Z.AI API key"),
            CommandCompletion("logout", "Clear stored API credentials"),
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
        if command != "zhipuai-auth":
            return f"Unknown command: {command}"

        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()

        if action_lower == "login":
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
                "  login   - Enter and validate your Z.AI API key\n"
                "  logout  - Clear stored API credentials\n"
                "  status  - Show current authentication status\n"
                "  help    - Show detailed help\n"
            )
            return ""

    def _cmd_help(self) -> str:
        """Return detailed help text."""
        help_text = """Zhipu AI (Z.AI) Auth Command

Manage authentication for Zhipu AI's GLM Coding Plan. This plugin securely
stores your API key for use with GLM models.

USAGE
    zhipuai-auth <action>

ACTIONS
    login             Interactively enter your Z.AI API key
                      The key will be validated before saving

    logout            Clear stored API credentials
                      Removes the stored API key

    status            Show current authentication status
                      Displays masked key and configuration

    help              Show this help message

GETTING AN API KEY
    1. International: Visit https://z.ai/model-api
    2. China: Visit https://open.bigmodel.cn/
    3. Sign up or log in to your account
    4. Navigate to the API keys section
    5. Generate a new API key
    6. Copy and use it with 'zhipuai-auth login'

EXAMPLES
    zhipuai-auth login              Enter your API key interactively
    zhipuai-auth status             Check authentication status
    zhipuai-auth logout             Clear stored credentials

TOKEN STORAGE
    Credentials are stored in:
    - Project: .jaato/zhipuai_auth.json (if .jaato/ exists)
    - User: ~/.jaato/zhipuai_auth.json (fallback)

    Files are created with restricted permissions (600 on Unix).

ENVIRONMENT VARIABLES
    ZHIPUAI_API_KEY       API key (takes precedence over stored key)
    ZHIPUAI_BASE_URL      Custom API endpoint (for enterprise users)

AVAILABLE MODELS
    glm-4.7               Latest model with native CoT reasoning (128K)
    glm-4.7-flash         Fast inference variant
    glm-4                 General purpose model
    glm-4v                Vision-enabled multimodal model
    glm-4-assistant       Optimized for agentic tasks

NOTES
    - API key is validated by making a test request before saving
    - Stored credentials are used automatically when connecting
    - Environment variable ZHIPUAI_API_KEY takes precedence"""
        self._emit(help_text)
        return ""

    def _cmd_login(self) -> str:
        """Handle the login command - prompt for API key."""
        from ..model_provider.zhipuai.auth import (
            login_with_key,
            get_stored_api_key,
            DEFAULT_ZHIPUAI_BASE_URL,
        )
        from ..model_provider.zhipuai.env import resolve_api_key

        # Check if already authenticated
        existing_key = resolve_api_key() or get_stored_api_key()
        if existing_key:
            masked = existing_key[:8] + "..." + existing_key[-4:] if len(existing_key) > 12 else "***"
            self._emit(
                f"Note: You already have a Z.AI API key configured ({masked}).\n"
                "Proceeding will replace it.\n\n"
            )

        self._emit("Zhipu AI (Z.AI) Authentication\n")
        self._emit("=" * 35 + "\n\n")
        self._emit("Get your API key from:\n")
        self._emit("  International: https://z.ai/model-api\n")
        self._emit("  China: https://open.bigmodel.cn/\n\n")

        # Since we can't do interactive input in plugin commands,
        # we need to provide instructions for the user
        self._emit(
            "To authenticate, set your API key using one of these methods:\n\n"
            "1. Environment variable (recommended for security):\n"
            "   export ZHIPUAI_API_KEY='your-api-key-here'\n\n"
            "2. Or use the provider config when starting jaato:\n"
            "   --model-provider zhipuai --api-key 'your-key'\n\n"
            "After setting the key, the provider will validate it on first use.\n"
        )

        return ""

    def _cmd_logout(self) -> str:
        """Handle the logout command."""
        try:
            from ..model_provider.zhipuai.auth import (
                clear_credentials,
                load_credentials,
            )

            creds = load_credentials()
            if not creds:
                self._emit("No stored credentials found. Already logged out.\n")
                return ""

            clear_credentials()
            self._emit(
                "Z.AI credentials cleared.\n\n"
                "You will need to set ZHIPUAI_API_KEY or run a new login "
                "to re-authenticate.\n"
            )
            return ""
        except Exception as e:
            self._emit(f"Failed to clear credentials: {e}\n")
            return ""

    def _cmd_status(self) -> str:
        """Handle the status command."""
        try:
            from ..model_provider.zhipuai.auth import (
                load_credentials,
                DEFAULT_ZHIPUAI_BASE_URL,
            )
            from ..model_provider.zhipuai.env import resolve_api_key, resolve_base_url

            lines = ["Zhipu AI (Z.AI) Authentication Status", "=" * 40, ""]

            # Check stored credentials
            creds = load_credentials()
            if creds:
                masked_key = (
                    creds.api_key[:8] + "..." + creds.api_key[-4:]
                    if len(creds.api_key) > 12
                    else "***"
                )
                saved_at = datetime.fromtimestamp(creds.created_at)
                lines.append(f"Stored Credentials: Active")
                lines.append(f"  API Key: {masked_key}")
                if creds.base_url:
                    lines.append(f"  Base URL: {creds.base_url}")
                lines.append(f"  Saved: {saved_at.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                lines.append("Stored Credentials: Not configured")

            lines.append("")

            # Check environment variable
            env_key = resolve_api_key()
            if env_key:
                masked = env_key[:8] + "..." + env_key[-4:] if len(env_key) > 12 else "***"
                lines.append(f"Environment API Key: Set ({masked})")
            else:
                lines.append("Environment API Key: Not set")
                lines.append("  Set ZHIPUAI_API_KEY environment variable")

            lines.append("")

            # Show effective configuration
            effective_key = env_key or (creds.api_key if creds else None)
            if effective_key:
                masked = effective_key[:8] + "..." + effective_key[-4:] if len(effective_key) > 12 else "***"
                lines.append(f"Effective API Key: {masked}")
                lines.append(f"  Source: {'Environment' if env_key else 'Stored credentials'}")
            else:
                lines.append("Effective API Key: None")
                lines.append("  Set ZHIPUAI_API_KEY or use 'zhipuai-auth login'")

            lines.append("")

            base_url = resolve_base_url()
            lines.append(f"Base URL: {base_url}")
            if base_url != DEFAULT_ZHIPUAI_BASE_URL:
                lines.append("  (custom endpoint)")

            lines.append("")
            lines.append("Priority: Environment > Stored Credentials")

            self._emit("\n".join(lines) + "\n")
            return ""

        except Exception as e:
            self._emit(f"Failed to check status: {e}\n")
            return ""


def create_plugin() -> ZhipuAIAuthPlugin:
    """Factory function for plugin discovery."""
    return ZhipuAIAuthPlugin()
