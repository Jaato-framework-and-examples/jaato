"""NVIDIA NIM authentication plugin.

Provides user commands for API key authentication with NVIDIA NIM.
This allows users to securely store and manage their NIM API credentials
(nvapi-... keys from build.nvidia.com).

Commands:
    nim-auth login            - Show instructions for getting API key
    nim-auth key <api_key>    - Validate and store API key
    nim-auth logout           - Clear stored API credentials
    nim-auth status           - Show current authentication status
"""

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


class NIMAuthPlugin:
    """Plugin for NVIDIA NIM API key authentication.

    Declares the ``TRAIT_AUTH_PROVIDER`` trait so the server can
    auto-discover this plugin when the NIM provider needs credentials.
    The ``provider_name`` property identifies which provider this plugin
    authenticates for.
    """

    plugin_traits: FrozenSet[str] = frozenset({TRAIT_AUTH_PROVIDER})

    def __init__(self):
        """Initialize the plugin."""
        self._output_callback: Optional[OutputCallback] = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "nim_auth"

    @property
    def provider_name(self) -> str:
        """Return the provider name this auth plugin serves."""
        return "nim"

    @property
    def provider_display_name(self) -> str:
        """Return human-readable provider name."""
        return "NVIDIA NIM"

    @property
    def credential_env_vars(self) -> List[str]:
        """Return env var names used for credentials by this provider."""
        return ["JAATO_NIM_API_KEY"]

    def get_default_models(self) -> List[Dict[str, str]]:
        """Return default models available for this provider.

        These are popular models from the NIM catalog on build.nvidia.com.
        """
        return [
            {"name": "nim/meta/llama-3.3-70b-instruct", "description": "Llama 3.3 70B — strong general-purpose reasoning"},
            {"name": "nim/meta/llama-3.1-405b-instruct", "description": "Llama 3.1 405B — largest open model"},
            {"name": "nim/deepseek/deepseek-r1", "description": "DeepSeek-R1 — reasoning with chain-of-thought"},
            {"name": "nim/nvidia/llama-3.1-nemotron-70b-instruct", "description": "Nemotron 70B — NVIDIA-tuned Llama"},
        ]

    def verify_credentials(self) -> bool:
        """Check if valid credentials exist after authentication."""
        try:
            from ..model_provider.nim.env import resolve_api_key, is_self_hosted, resolve_base_url
            api_key = resolve_api_key()
            if api_key:
                return True
            return is_self_hosted(resolve_base_url())
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
            self._output_callback("nim_auth", text, mode)

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return empty list — this plugin only provides user commands."""
        return []

    def get_executors(self) -> Dict[str, Any]:
        """Return executor for the user command."""
        return {
            "nim-auth": lambda args: self.execute_user_command("nim-auth", args),
        }

    def get_system_instructions(self) -> Optional[str]:
        """No system instructions for this plugin."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """User commands don't need permission approval."""
        return ["nim-auth"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for authentication."""
        return [
            UserCommand(
                name="nim-auth",
                description="Manage NVIDIA NIM authentication (login, key, logout, status)",
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description="Action: login, key <api_key>, logout, status, or help",
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
        if command != "nim-auth":
            return []

        actions = [
            CommandCompletion("login", "Show instructions for getting your NIM API key"),
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
        if command != "nim-auth":
            return f"Unknown command: {command}"

        # Get raw action string (don't lowercase — API keys are case-sensitive)
        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()

        # Handle "key <api_key>" — extract key from raw string (preserve case)
        if action_lower.startswith("key "):
            api_key = raw_action[4:].strip()
            return self._cmd_key(api_key)
        elif action_lower == "key":
            self._emit("Usage: nim-auth key <your_api_key>\n")
            self._emit("\nGet your API key from:\n")
            self._emit("  https://build.nvidia.com/ → Settings → API Keys\n")
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
            ("NVIDIA NIM Auth Command", "bold"),
            ("", ""),
            ("Manage authentication for NVIDIA NIM (Inference Microservices). This plugin", ""),
            ("securely stores your API key for use with NIM-hosted models.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    nim-auth <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    login             Show instructions for getting your NIM API key", "dim"),
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
            ("    1. Run 'nim-auth login' to see instructions", ""),
            ("    2. Visit https://build.nvidia.com/", ""),
            ("    3. Sign in and generate an API key (nvapi-...)", ""),
            ("    4. Run 'nim-auth key <paste_key_here>'", ""),
            ("    5. Key is validated and saved for future use", ""),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    nim-auth login                          Show setup instructions", "dim"),
            ("    nim-auth key nvapi-abc123...            Store and validate API key", "dim"),
            ("    nim-auth status                         Check authentication status", "dim"),
            ("    nim-auth logout                         Clear stored credentials", "dim"),
            ("", ""),
            ("TOKEN STORAGE", "bold"),
            ("    Credentials are stored in:", ""),
            ("    - Project: .jaato/nim_auth.json (if .jaato/ exists)", "dim"),
            ("    - User: ~/.jaato/nim_auth.json (fallback)", "dim"),
            ("", ""),
            ("    Files are created with restricted permissions (600 on Unix).", "dim"),
            ("", ""),
            ("ENVIRONMENT VARIABLES", "bold"),
            ("    JAATO_NIM_API_KEY       API key (takes precedence over stored key)", "dim"),
            ("    JAATO_NIM_BASE_URL      Custom API endpoint (for self-hosted NIM)", "dim"),
            ("    JAATO_NIM_MODEL         Default model name", "dim"),
            ("    JAATO_NIM_CONTEXT_LENGTH  Override context window size", "dim"),
            ("", ""),
            ("SELF-HOSTED NIM", "bold"),
            ("    For self-hosted NIM containers, no API key is needed.", "dim"),
            ("    Set JAATO_NIM_BASE_URL to your container endpoint:", "dim"),
            ("      export JAATO_NIM_BASE_URL=http://localhost:8000/v1", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - API key format is: nvapi-...", "dim"),
            ("    - API key is validated by making a test request before saving", "dim"),
            ("    - Stored credentials are used automatically when connecting", "dim"),
            ("    - Environment variable JAATO_NIM_API_KEY takes precedence", "dim"),
        ])

    def _cmd_login(self) -> str:
        """Handle the login command — show instructions."""
        from ..model_provider.nim.env import resolve_api_key

        # Check if already authenticated
        existing_key = resolve_api_key()
        if existing_key:
            masked = existing_key[:8] + "..." + existing_key[-4:] if len(existing_key) > 12 else "***"
            self._emit(
                f"Note: You already have a NIM API key configured ({masked}).\n"
                "Using 'nim-auth key <new_key>' will replace it.\n\n"
            )

        self._emit("NVIDIA NIM Authentication\n")
        self._emit("=" * 30 + "\n\n")
        self._emit("Step 1: Get your API key from:\n")
        self._emit("  https://build.nvidia.com/\n")
        self._emit("  Sign in → Settings → API Keys → Generate\n\n")
        self._emit("Step 2: Copy your API key (nvapi-...) and run:\n")
        self._emit("  nim-auth key <paste_your_key_here>\n\n")
        self._emit("The key will be validated and stored securely.\n")

        return ""

    def _cmd_key(self, api_key: str) -> str:
        """Handle the key command — validate and store API key.

        Resolves the effective base URL from environment / stored credentials
        so the validation request hits the same endpoint the provider will use.
        """
        from ..model_provider.nim.auth import (
            login_with_key,
            validate_api_key,
        )
        from ..model_provider.nim.env import resolve_api_key, resolve_base_url

        if not api_key:
            self._emit("Error: No API key provided.\n")
            self._emit("Usage: nim-auth key <your_api_key>\n")
            return ""

        # Check if this would replace an existing key
        existing_key = resolve_api_key()
        if existing_key and existing_key == api_key:
            self._emit("This API key is already configured.\n")
            return ""

        self._emit("Validating API key...\n")

        # Use the same base URL the provider would use
        base_url = resolve_base_url()

        # Validate the key
        valid, detail = validate_api_key(api_key, base_url)
        if valid:
            def on_message(msg: str) -> None:
                self._emit(f"{msg}\n")

            result = login_with_key(api_key, base_url=base_url, on_message=on_message)
            if result:
                self._emit("\n")
                self._emit("Successfully authenticated with NVIDIA NIM.\n")
                self._emit("Your API key has been stored securely.\n\n")
                self._emit("You can now use the NIM provider:\n")
                self._emit("  model nim/meta/llama-3.3-70b-instruct\n")
            else:
                self._emit("\nFailed to store credentials.\n")
        else:
            if detail.startswith("network_error"):
                self._emit("\nCould not reach the NIM API.\n\n")
                self._emit(f"Detail: {detail}\n\n")
                self._emit("Please check that:\n")
                self._emit("  - You have internet connectivity\n")
                self._emit("  - integrate.api.nvidia.com is reachable from your network\n")
                self._emit("  - No firewall or proxy is blocking the request\n")
            else:
                self._emit("\nAPI key validation failed.\n\n")
                self._emit("Please check that:\n")
                self._emit("  - The key is correct and complete (starts with nvapi-)\n")
                self._emit("  - Your NVIDIA account is active\n")
                self._emit("  - You have NIM API access enabled\n\n")
                self._emit("Get your key from:\n")
                self._emit("  https://build.nvidia.com/ → Settings → API Keys\n")

        return ""

    def _cmd_logout(self) -> str:
        """Handle the logout command."""
        try:
            from ..model_provider.nim.auth import (
                clear_credentials,
                load_credentials,
            )

            creds = load_credentials()
            if not creds:
                self._emit("No stored credentials found. Already logged out.\n")
                return ""

            clear_credentials()
            self._emit(
                "NIM credentials cleared.\n\n"
                "You will need to set JAATO_NIM_API_KEY or run a new login "
                "to re-authenticate.\n"
            )
            return ""
        except Exception as e:
            self._emit(f"Failed to clear credentials: {e}\n")
            return ""

    def _cmd_status(self) -> str:
        """Handle the status command."""
        try:
            from ..model_provider.nim.auth import load_credentials
            from ..model_provider.nim.env import (
                DEFAULT_BASE_URL,
                ENV_NIM_API_KEY,
                resolve_api_key,
                resolve_base_url,
                is_self_hosted,
            )
            import os

            lines = ["NVIDIA NIM Authentication Status", "=" * 35, ""]

            # Check stored credentials
            creds = load_credentials()
            if creds:
                masked_key = (
                    creds.api_key[:8] + "..." + creds.api_key[-4:]
                    if len(creds.api_key) > 12
                    else "***"
                )
                saved_at = datetime.fromtimestamp(creds.created_at)
                lines.append("Stored Credentials: Active")
                lines.append(f"  API Key: {masked_key}")
                if creds.base_url:
                    lines.append(f"  Base URL: {creds.base_url}")
                lines.append(f"  Saved: {saved_at.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                lines.append("Stored Credentials: Not configured")

            lines.append("")

            # Check environment variable
            env_key = os.environ.get(ENV_NIM_API_KEY)
            if env_key:
                masked = env_key[:8] + "..." + env_key[-4:] if len(env_key) > 12 else "***"
                lines.append(f"Environment API Key: Set ({masked})")
            else:
                lines.append("Environment API Key: Not set")
                lines.append(f"  Set {ENV_NIM_API_KEY} environment variable")

            lines.append("")

            # Show effective configuration
            effective_key = resolve_api_key()
            if effective_key:
                masked = effective_key[:8] + "..." + effective_key[-4:] if len(effective_key) > 12 else "***"
                source = "Environment" if env_key else "Stored credentials"
                lines.append(f"Effective API Key: {masked}")
                lines.append(f"  Source: {source}")
            else:
                lines.append("Effective API Key: None")
                lines.append("  Set JAATO_NIM_API_KEY or use 'nim-auth login'")

            lines.append("")

            base_url = resolve_base_url()
            lines.append(f"Base URL: {base_url}")
            if base_url != DEFAULT_BASE_URL:
                lines.append("  (custom endpoint)")
            if is_self_hosted(base_url):
                lines.append("  (self-hosted — no API key required)")

            lines.append("")
            lines.append("Priority: Environment > Stored Credentials")

            self._emit("\n".join(lines) + "\n")
            return ""

        except Exception as e:
            self._emit(f"Failed to check status: {e}\n")
            return ""


def create_plugin() -> NIMAuthPlugin:
    """Factory function for plugin discovery."""
    return NIMAuthPlugin()
