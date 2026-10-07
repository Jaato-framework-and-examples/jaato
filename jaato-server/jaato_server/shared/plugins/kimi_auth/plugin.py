"""Moonshot AI Kimi authentication plugin.

Provides user commands for API key authentication with Moonshot AI Kimi — the
``nim_auth`` shape, pointed at the ``kimi`` provider's auth module.

Commands:
    kimi-auth login            - Show instructions for getting an API key
    kimi-auth key <api_key>    - Validate and store the API key
    kimi-auth logout           - Clear stored API credentials
    kimi-auth status           - Show current authentication status
"""

import os
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from jaato_sdk.plugins.base import (
    CommandCompletion,
    CommandParameter,
    HelpLines,
    TRAIT_AUTH_PROVIDER,
    UserCommand,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema

# Type alias for output callback: (source, text, mode) -> None
OutputCallback = Callable[[str, str, str], None]

COMMAND = "kimi-auth"


def _mask(key: str) -> str:
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "***"


class KimiAuthPlugin:
    """Plugin for Moonshot AI Kimi API key authentication.

    Declares the ``TRAIT_AUTH_PROVIDER`` trait so the server can
    auto-discover this plugin when the ``kimi`` provider needs
    credentials; ``provider_name`` identifies which provider.
    """

    plugin_traits: FrozenSet[str] = frozenset({TRAIT_AUTH_PROVIDER})

    def __init__(self):
        self._output_callback: Optional[OutputCallback] = None
        self._workspace_path: Optional[str] = None

    @property
    def name(self) -> str:
        return "kimi_auth"

    @property
    def provider_name(self) -> str:
        return "kimi"

    @property
    def provider_display_name(self) -> str:
        return "Moonshot AI Kimi"

    @property
    def credential_env_vars(self) -> List[str]:
        return ["JAATO_KIMI_API_KEY", "MOONSHOT_API_KEY"]

    def get_default_models(self) -> List[Dict[str, str]]:
        """Current models on the vendor's public API."""
        return [
            {"name": "kimi/kimi-k3", "description": "Kimi K3 — 1M context, always-on reasoning (reasoning_effort)"},
            {"name": "kimi/kimi-k2.7-code", "description": "Kimi K2.7 Code — 256K context, coding, always-on thinking"},
            {"name": "kimi/kimi-k2.6", "description": "Kimi K2.6 — 256K context, switchable thinking"},
        ]

    def verify_credentials(self) -> bool:
        """Check if valid credentials exist after authentication."""
        try:
            from ..model_provider.kimi.env import resolve_api_key, is_self_hosted, resolve_base_url
            if resolve_api_key():
                return True
            return is_self_hosted(resolve_base_url())
        except Exception:
            return False

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Store ``workspace_path`` so credential storage resolves project vs user tier."""
        self._workspace_path = (config or {}).get("workspace_path")

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        self._output_callback = callback

    def _emit(self, text: str, mode: str = "write") -> None:
        if self._output_callback:
            self._output_callback("kimi_auth", text, mode)

    def shutdown(self) -> None:
        pass

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset — no per-session state to clear."""
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        return []

    def get_executors(self) -> Dict[str, Any]:
        return {COMMAND: lambda args: self.execute_user_command(COMMAND, args)}

    def get_system_instructions(self) -> Optional[str]:
        return None

    def get_auto_approved_tools(self) -> List[str]:
        return [COMMAND]

    def get_user_commands(self) -> List[UserCommand]:
        return [
            UserCommand(
                name=COMMAND,
                description="Manage Moonshot AI Kimi authentication (login, key, logout, status)",
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

    def get_command_completions(self, command: str, args: List[str]) -> List[CommandCompletion]:
        if command != COMMAND:
            return []
        actions = [
            CommandCompletion("login", "Show instructions for getting your Kimi API key"),
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
        if command != COMMAND:
            return f"Unknown command: {command}"
        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()
        if action_lower.startswith("key "):
            return self._cmd_key(raw_action[4:].strip())
        if action_lower == "key":
            self._emit(f"Usage: {COMMAND} key <your_api_key>\n")
            self._emit("\nGet your API key from:\n  https://platform.kimi.ai/console/api-keys\n")
            return ""
        handlers = {
            "login": self._cmd_login,
            "logout": self._cmd_logout,
            "status": self._cmd_status,
            "help": self._cmd_help,
        }
        handler = handlers.get(action_lower)
        if handler:
            return handler()
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
        return HelpLines(lines=[
            ("Moonshot AI Kimi Auth Command", "bold"),
            ("", ""),
            ("Manage authentication for Moonshot AI Kimi.  Stores your API key securely for", ""),
            ("use with the 'kimi' provider.", ""),
            ("", ""),
            ("USAGE", "bold"),
            (f"    {COMMAND} <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    login             Show instructions for getting your API key", "dim"),
            ("    key <api_key>     Validate and store your API key", "dim"),
            ("    logout            Clear stored API credentials", "dim"),
            ("    status            Show current authentication status", "dim"),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("AUTHENTICATION FLOW", "bold"),
            (f"    1. Run '{COMMAND} login' to see instructions", ""),
            ("    2. Generate an API key at https://platform.kimi.ai/console/api-keys", ""),
            (f"    3. Run '{COMMAND} key <paste_key_here>'", ""),
            ("    4. The key is validated (a cheap authenticated GET) and saved", ""),
            ("", ""),
            ("TOKEN STORAGE", "bold"),
            ("    - Project: .jaato/kimi_auth.json (if .jaato/ exists)", "dim"),
            ("    - User: ~/.jaato/kimi_auth.json (fallback)", "dim"),
            ("    Files are created with restricted permissions (600 on Unix).", "dim"),
            ("", ""),
            ("ENVIRONMENT VARIABLES", "bold"),
            ("    JAATO_KIMI_API_KEY        API key (takes precedence over stored key)", "dim"),
            ("    MOONSHOT_API_KEY                the vendor's own variable, honoured next", "dim"),
            ("    JAATO_KIMI_BASE_URL       Custom API endpoint (region / plan hosts)", "dim"),
            ("    JAATO_KIMI_MODEL          Default model name", "dim"),
            ("    JAATO_KIMI_CONTEXT_LENGTH Override the context window", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - China platform: https://api.moonshot.cn/v1 (keys are per platform).", "dim"),
            ("    - Kimi Code plan keys use https://api.kimi.com/coding/v1 with the plan's own model ids (k3, kimi-for-coding) — set JAATO_KIMI_BASE_URL.", "dim"),
        ])

    def _cmd_login(self) -> str:
        from ..model_provider.kimi.env import resolve_api_key

        existing_key = resolve_api_key()
        if existing_key:
            self._emit(
                f"Note: You already have a Kimi API key configured ({_mask(existing_key)}).\n"
                f"Using '{COMMAND} key <new_key>' will replace it.\n\n"
            )
        self._emit("Moonshot AI Kimi Authentication\n")
        self._emit("=" * 30 + "\n\n")
        self._emit("Step 1: Get your API key from:\n")
        self._emit("  https://platform.kimi.ai/console/api-keys\n\n")
        self._emit("Step 2: Copy your API key (sk-...) and run:\n")
        self._emit(f"  {COMMAND} key <paste_your_key_here>\n\n")
        self._emit("The key will be validated and stored securely.\n")
        return ""

    def _cmd_key(self, api_key: str) -> str:
        """Validate and store an API key against the effective base URL."""
        from ..model_provider.kimi.auth import login_with_key, validate_api_key
        from ..model_provider.kimi.env import resolve_api_key, resolve_base_url

        if not api_key:
            self._emit(f"Error: No API key provided.\nUsage: {COMMAND} key <your_api_key>\n")
            return ""
        if resolve_api_key() == api_key:
            self._emit("This API key is already configured.\n")
            return ""
        self._emit("Validating API key...\n")
        base_url = resolve_base_url()
        valid, detail = validate_api_key(api_key, base_url)
        if valid:
            result = login_with_key(
                api_key, base_url=base_url,
                on_message=lambda msg: self._emit(f"{msg}\n"),
                workspace_path=self._workspace_path,
            )
            if result:
                self._emit("\nSuccessfully authenticated with Moonshot AI Kimi.\n")
                self._emit("Your API key has been stored securely.\n\n")
                self._emit("You can now use the kimi provider:\n")
                self._emit("  model kimi/kimi-k3\n")
            else:
                self._emit("\nFailed to store credentials.\n")
            return ""
        self._report_validation_failure(detail, base_url)
        return ""

    def _report_validation_failure(self, detail: str, base_url: str) -> None:
        code = detail.split(":", 1)[0]
        explanations = {
            "network_error": (
                f"Could not reach the Moonshot AI Kimi API at {base_url}.\n\n"
                "Please check connectivity, and that no firewall or proxy is\n"
                "blocking the request.\n"),
            "rate_limit": (
                "Moonshot AI Kimi rejected the validation request with a rate limit.\n"
                "The key was NOT saved.  Wait and try again.\n"),
            "payment_required": (
                "Moonshot AI Kimi reports the account balance is exhausted.  The key was\n"
                "NOT saved.  Top up at https://platform.kimi.ai/console/api-keys.\n"),
            "forbidden": (
                "Moonshot AI Kimi refused the key (HTTP 403).  The key was NOT saved.\n"
                "China platform: https://api.moonshot.cn/v1 (keys are per platform).\nKimi Code plan keys use https://api.kimi.com/coding/v1 with the plan's own model ids (k3, kimi-for-coding) — set JAATO_KIMI_BASE_URL.\n"),
            "server_error": (
                "Moonshot AI Kimi returned a server error while validating the key.\n"
                "The key was NOT saved.  Please retry in a few minutes.\n"),
            "http_error": (
                "Moonshot AI Kimi returned an unexpected response while validating the key.\n"
                "The key was NOT saved.\n"),
        }
        self._emit("\n" + explanations.get(code, (
            "API key validation failed.\n\n"
            "Please check that the key is correct and complete, that the\n"
            "account is active, and that the key matches the endpoint\n"
            f"({base_url}).  Keys are issued per platform (.ai vs .cn).\n"
            "Get your key from:\n  https://platform.kimi.ai/console/api-keys\n")))
        if detail:
            self._emit(f"\nDetail: {detail}\n")

    def _cmd_logout(self) -> str:
        try:
            from ..model_provider.kimi.auth import clear_credentials, load_credentials

            if not load_credentials(workspace_path=self._workspace_path):
                self._emit("No stored credentials found. Already logged out.\n")
                return ""
            clear_credentials(workspace_path=self._workspace_path)
            self._emit(
                "Moonshot AI Kimi credentials cleared.\n\n"
                "You will need to set JAATO_KIMI_API_KEY or run a new login "
                "to re-authenticate.\n"
            )
        except Exception as e:
            self._emit(f"Failed to clear credentials: {e}\n")
        return ""

    def _cmd_status(self) -> str:
        try:
            from ..model_provider.kimi.auth import load_credentials
            from ..model_provider.kimi.env import (
                DEFAULT_BASE_URL,
                ENV_KIMI_API_KEY,
                ENV_KIMI_VENDOR_API_KEY,
                resolve_api_key,
                resolve_base_url,
                is_self_hosted,
            )

            lines = ["Moonshot AI Kimi Authentication Status", "=" * 35, ""]
            creds = load_credentials(workspace_path=self._workspace_path)
            if creds:
                lines.append("Stored Credentials: Active")
                lines.append(f"  API Key: {_mask(creds.api_key)}")
                if creds.base_url:
                    lines.append(f"  Base URL: {creds.base_url}")
                saved_at = datetime.fromtimestamp(creds.created_at)
                lines.append(f"  Saved: {saved_at.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                lines.append("Stored Credentials: Not configured")
            lines.append("")

            env_source = None
            for var in (ENV_KIMI_API_KEY, ENV_KIMI_VENDOR_API_KEY):
                value = os.environ.get(var)
                if value:
                    lines.append(f"Environment API Key: Set ({_mask(value)}, {var})")
                    env_source = env_source or var
            if not env_source:
                lines.append("Environment API Key: Not set")
                lines.append(f"  Set {ENV_KIMI_API_KEY} (or {ENV_KIMI_VENDOR_API_KEY})")
            lines.append("")

            effective_key = resolve_api_key()
            if effective_key:
                lines.append(f"Effective API Key: {_mask(effective_key)}")
                lines.append(f"  Source: {env_source or 'Stored credentials'}")
            else:
                lines.append("Effective API Key: None")
                lines.append(f"  Set {ENV_KIMI_API_KEY} or use '{COMMAND} login'")
            lines.append("")

            base_url = resolve_base_url()
            lines.append(f"Base URL: {base_url}")
            if base_url != DEFAULT_BASE_URL:
                lines.append("  (custom endpoint)")
            if is_self_hosted(base_url):
                lines.append("  (self-hosted proxy — no API key required)")
            lines.append("")
            lines.append("Priority: Environment > Stored Credentials")
            self._emit("\n".join(lines) + "\n")
        except Exception as e:
            self._emit(f"Failed to check status: {e}\n")
        return ""


def create_plugin() -> KimiAuthPlugin:
    """Factory function for plugin discovery."""
    return KimiAuthPlugin()
