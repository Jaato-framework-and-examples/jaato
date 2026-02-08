"""Zhipu AI (Z.AI) authentication plugin.

Provides user commands for API key authentication with Z.AI GLM Coding Plan.
This allows users to securely store and manage their Z.AI API credentials.

Commands:
    zhipuai-auth login            - Show instructions for getting API key
    zhipuai-auth key <api_key>    - Validate and store API key
    zhipuai-auth logout           - Clear stored API credentials
    zhipuai-auth status           - Show current authentication status
"""

from datetime import datetime
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


class ZhipuAIAuthPlugin:
    """Plugin for Zhipu AI API key authentication."""

    def __init__(self):
        """Initialize the plugin."""
        self._output_callback: Optional[OutputCallback] = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "zhipuai_auth"

    @property
    def provider_name(self) -> str:
        """Return the provider name this auth plugin serves."""
        return "zhipuai"

    @property
    def provider_display_name(self) -> str:
        """Return human-readable provider name."""
        return "Zhipu AI (Z.AI)"

    def get_default_models(self) -> List[Dict[str, str]]:
        """Return default models available for this provider."""
        return [
            {"name": "zhipuai/glm-4.7", "description": "Latest model with native CoT reasoning (128K)"},
            {"name": "zhipuai/glm-4.7-flash", "description": "Fast inference variant"},
            {"name": "zhipuai/glm-4", "description": "General purpose model"},
        ]

    def verify_credentials(self) -> bool:
        """Check if valid credentials exist after authentication."""
        try:
            from ..model_provider.zhipuai.auth import get_stored_api_key
            from ..model_provider.zhipuai.env import resolve_api_key
            return bool(resolve_api_key() or get_stored_api_key())
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
                description="Manage Zhipu AI (Z.AI) authentication (login, key, logout, status)",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: login, key <api_key>, logout, status, or help",
                        required=True,
                        capture_rest=True,  # Capture "key ABC123" as single string
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
            CommandCompletion("login", "Show instructions for getting your Z.AI API key"),
            CommandCompletion("key", "Validate and store your API key: key <api_key>"),
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

        # Get raw action string (don't lowercase - API keys are case-sensitive)
        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()

        # Handle "key <api_key>" - extract key from raw string (preserve case)
        if action_lower.startswith("key "):
            api_key = raw_action[4:].strip()
            return self._cmd_key(api_key)
        elif action_lower == "key":
            self._emit("Usage: zhipuai-auth key <your_api_key>\n")
            self._emit("\nGet your API key from:\n")
            self._emit("  International: https://z.ai/model-api\n")
            self._emit("  China: https://open.bigmodel.cn/\n")
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
                "  login       - Show instructions for getting your API key\n"
                "  key <key>   - Validate and store your API key\n"
                "  logout      - Clear stored API credentials\n"
                "  status      - Show current authentication status\n"
                "  help        - Show detailed help\n"
            )
            return ""

    def _cmd_help(self) -> HelpLines:
        """Return detailed help text for pager display."""
        return HelpLines(lines=[
            ("Zhipu AI (Z.AI) Auth Command", "bold"),
            ("", ""),
            ("Manage authentication for Zhipu AI's GLM Coding Plan. This plugin securely", ""),
            ("stores your API key for use with GLM models.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    zhipuai-auth <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    login             Show instructions for getting your Z.AI API key", "dim"),
            ("", ""),
            ("    key <api_key>     Validate and store your API key", "dim"),
            ("                      The key is validated via test request before saving", "dim"),
            ("", ""),
            ("    logout            Clear stored API credentials", "dim"),
            ("                      Removes the stored API key", "dim"),
            ("", ""),
            ("    status            Show current authentication status", "dim"),
            ("                      Displays masked key and configuration", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("AUTHENTICATION FLOW", "bold"),
            ("    1. Run 'zhipuai-auth login' to see instructions", ""),
            ("    2. Visit https://z.ai/model-api or https://open.bigmodel.cn/", ""),
            ("    3. Sign in and generate an API key", ""),
            ("    4. Run 'zhipuai-auth key <paste_key_here>'", ""),
            ("    5. Key is validated and saved for future use", ""),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    zhipuai-auth login                      Show setup instructions", "dim"),
            ("    zhipuai-auth key abc123.xyz789          Store and validate API key", "dim"),
            ("    zhipuai-auth status                     Check authentication status", "dim"),
            ("    zhipuai-auth logout                     Clear stored credentials", "dim"),
            ("", ""),
            ("TOKEN STORAGE", "bold"),
            ("    Credentials are stored in:", ""),
            ("    - Project: .jaato/zhipuai_auth.json (if .jaato/ exists)", "dim"),
            ("    - User: ~/.jaato/zhipuai_auth.json (fallback)", "dim"),
            ("", ""),
            ("    Files are created with restricted permissions (600 on Unix).", "dim"),
            ("", ""),
            ("ENVIRONMENT VARIABLES", "bold"),
            ("    ZHIPUAI_API_KEY       API key (takes precedence over stored key)", "dim"),
            ("    ZHIPUAI_BASE_URL      Custom API endpoint (for enterprise users)", "dim"),
            ("", ""),
            ("AVAILABLE MODELS", "bold"),
            ("    glm-4.7               Latest model with native CoT reasoning (128K)", "dim"),
            ("    glm-4.7-flash         Fast inference variant", "dim"),
            ("    glm-4                 General purpose model", "dim"),
            ("    glm-4v                Vision-enabled multimodal model", "dim"),
            ("    glm-4-assistant       Optimized for agentic tasks", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - API key format is typically: {id}.{secret}", "dim"),
            ("    - API key is validated by making a test request before saving", "dim"),
            ("    - Stored credentials are used automatically when connecting", "dim"),
            ("    - Environment variable ZHIPUAI_API_KEY takes precedence", "dim"),
        ])

    def _cmd_login(self) -> str:
        """Handle the login command - show instructions."""
        from ..model_provider.zhipuai.auth import get_stored_api_key
        from ..model_provider.zhipuai.env import resolve_api_key

        # Check if already authenticated
        existing_key = resolve_api_key() or get_stored_api_key()
        if existing_key:
            masked = existing_key[:8] + "..." + existing_key[-4:] if len(existing_key) > 12 else "***"
            self._emit(
                f"Note: You already have a Z.AI API key configured ({masked}).\n"
                "Using 'zhipuai-auth key <new_key>' will replace it.\n\n"
            )

        self._emit("Zhipu AI (Z.AI) Authentication\n")
        self._emit("=" * 35 + "\n\n")
        self._emit("Step 1: Get your API key from:\n")
        self._emit("  International: https://z.ai/model-api\n")
        self._emit("  China: https://open.bigmodel.cn/\n\n")
        self._emit("Step 2: Copy your API key and run:\n")
        self._emit("  zhipuai-auth key <paste_your_key_here>\n\n")
        self._emit("The key will be validated and stored securely.\n")

        return ""

    def _cmd_key(self, api_key: str) -> str:
        """Handle the key command - validate and store API key."""
        from ..model_provider.zhipuai.auth import (
            login_with_key,
            validate_api_key,
            get_stored_api_key,
        )
        from ..model_provider.zhipuai.env import resolve_api_key

        if not api_key:
            self._emit("Error: No API key provided.\n")
            self._emit("Usage: zhipuai-auth key <your_api_key>\n")
            return ""

        # Check if this would replace an existing key
        existing_key = resolve_api_key() or get_stored_api_key()
        if existing_key and existing_key == api_key:
            self._emit("This API key is already configured.\n")
            return ""

        self._emit("Validating API key...\n")

        # Validate the key
        if validate_api_key(api_key):
            # Store it
            def on_message(msg: str) -> None:
                self._emit(f"{msg}\n")

            result = login_with_key(api_key, on_message=on_message)
            if result:
                self._emit("\n")
                self._emit("Successfully authenticated with Z.AI.\n")
                self._emit("Your API key has been stored securely.\n\n")
                self._emit("You can now use the zhipuai provider:\n")
                self._emit("  model zhipuai/glm-4.7\n")
            else:
                self._emit("\nFailed to store credentials.\n")
        else:
            self._emit("\nAPI key validation failed.\n\n")
            self._emit("Please check that:\n")
            self._emit("  - The key is correct and complete\n")
            self._emit("  - Your Z.AI account is active\n")
            self._emit("  - You have an active GLM Coding Plan subscription\n\n")
            self._emit("Get your key from:\n")
            self._emit("  https://z.ai/model-api (International)\n")
            self._emit("  https://open.bigmodel.cn/ (China)\n")

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
