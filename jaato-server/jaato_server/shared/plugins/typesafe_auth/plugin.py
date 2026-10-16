"""TypeSafe authentication plugin: ``typesafe-auth``.

Stores a TypeSafe API key for the ``typesafe`` decision provider.  The
stored key is the LAST place the provider looks: the profile knob
``plugin_configs.typesafe.api_key`` comes first, then
``JAATO_TYPESAFE_API_KEY`` and ``TYPESAFE_API_KEY``.

Commands:
    typesafe-auth login            - Show how to get a key
    typesafe-auth key <api_key>    - Validate and store the key
    typesafe-auth logout           - Clear the stored key
    typesafe-auth status           - Show where the key would come from
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

OutputCallback = Callable[[str, str, str], None]

COMMAND = "typesafe-auth"
KEYS_URL = "https://console.typesafe.ai/keys"


def _mask(key: str) -> str:
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "***"


class TypeSafeAuthPlugin:
    """User commands for the ``typesafe`` provider's stored key.

    Declares ``TRAIT_AUTH_PROVIDER``; ``provider_name`` is ``typesafe``.
    """

    plugin_traits: FrozenSet[str] = frozenset({TRAIT_AUTH_PROVIDER})

    def __init__(self):
        self._output_callback: Optional[OutputCallback] = None
        self._workspace_path: Optional[str] = None

    @property
    def name(self) -> str:
        return "typesafe_auth"

    @property
    def provider_name(self) -> str:
        return "typesafe"

    @property
    def provider_display_name(self) -> str:
        return "TypeSafe"

    @property
    def credential_env_vars(self) -> List[str]:
        return ["JAATO_TYPESAFE_API_KEY", "TYPESAFE_API_KEY"]

    def get_default_models(self) -> List[Dict[str, str]]:
        """Jev ids TypeSafe's docs name.  Bound in decision_models:, not as model:."""
        return [
            {"name": "typesafe/jev-latest", "description": "Jev, the current release (decision model)"},
            {"name": "typesafe/jev-1.13.0", "description": "Jev 1.13 (decision model)"},
        ]

    def verify_credentials(self) -> bool:
        try:
            from ..model_provider.typesafe.env import resolve_api_key
            return bool(resolve_api_key())
        except Exception:
            return False

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Store ``workspace_path`` so storage resolves project vs user tier."""
        self._workspace_path = (config or {}).get("workspace_path")

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        self._output_callback = callback

    def _emit(self, text: str, mode: str = "write") -> None:
        if self._output_callback:
            self._output_callback("typesafe_auth", text, mode)

    def shutdown(self) -> None:
        pass

    def reset_for_next_session(self) -> None:
        """No per-session state."""

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
                description="Manage the TypeSafe API key (login, key, logout, status)",
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
            CommandCompletion("login", "Show how to get a TypeSafe API key"),
            CommandCompletion("key", "Validate and store your key: key <api_key>"),
            CommandCompletion("logout", "Clear the stored key"),
            CommandCompletion("status", "Show where the key would come from"),
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
        action = raw_action.lower()
        if action.startswith("key "):
            return self._cmd_key(raw_action[4:].strip())
        if action == "key":
            self._emit(f"Usage: {COMMAND} key <your_api_key>\n\nGet a key at:\n  {KEYS_URL}\n")
            return ""
        handler = {"login": self._cmd_login, "logout": self._cmd_logout,
                   "status": self._cmd_status, "help": self._cmd_help}.get(action)
        if handler:
            return handler()
        self._emit(
            f"Unknown action: '{raw_action}'\n\nAvailable actions: "
            "login, key <key>, logout, status, help\n"
        )
        return ""

    def _cmd_help(self) -> HelpLines:
        return HelpLines(lines=[
            ("TypeSafe Auth Command", "bold"),
            ("", ""),
            ("Stores a TypeSafe API key for the 'typesafe' decision provider.", ""),
            ("", ""),
            ("USAGE", "bold"),
            (f"    {COMMAND} <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    login             Show how to get a key", "dim"),
            ("    key <api_key>     Validate (one noul question) and store the key", "dim"),
            ("    logout            Clear the stored key", "dim"),
            ("    status            Show where the key would come from", "dim"),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("WHERE THE KEY COMES FROM (first wins)", "bold"),
            ("    1. plugin_configs.typesafe.api_key in the profile (pass:// ok)", "dim"),
            ("    2. JAATO_TYPESAFE_API_KEY", "dim"),
            ("    3. TYPESAFE_API_KEY (TypeSafe's own variable)", "dim"),
            ("    4. the key stored by this command", "dim"),
            ("", ""),
            ("STORAGE", "bold"),
            ("    - Project: .jaato/typesafe_auth.json (if .jaato/ exists)", "dim"),
            ("    - User: ~/.jaato/typesafe_auth.json (fallback)", "dim"),
            ("    Files are created with mode 600 on Unix.", "dim"),
        ])

    def _cmd_login(self) -> str:
        self._emit(
            "TypeSafe Authentication\n" + "=" * 23 + "\n\n"
            f"Step 1: Create a key at:\n  {KEYS_URL}\n\n"
            f"Step 2: Run:\n  {COMMAND} key <paste_your_key_here>\n\n"
            "The key is validated with one decision question and stored.\n"
            "A profile's plugin_configs.typesafe.api_key, or JAATO_TYPESAFE_API_KEY /\n"
            "TYPESAFE_API_KEY, still take precedence over the stored key.\n"
        )
        return ""

    def _cmd_key(self, api_key: str) -> str:
        from ..model_provider.typesafe.auth import login_with_key
        from ..model_provider.typesafe.env import resolve_base_url

        if not api_key:
            self._emit(f"Error: No API key provided.\nUsage: {COMMAND} key <your_api_key>\n")
            return ""
        self._emit("Validating API key...\n")
        base_url = resolve_base_url()
        messages: List[str] = []
        creds = login_with_key(api_key, base_url=base_url, on_message=messages.append,
                               workspace_path=self._workspace_path)
        if creds:
            self._emit("\n".join(messages) + "\n\nThe 'typesafe' provider can now answer "
                       "decisions.  Bind it in a profile:\n"
                       "  decision_models:\n    triage: {provider: typesafe, model: jev-latest}\n")
        else:
            self._emit("\n".join(messages) + "\nThe key was NOT saved.  Check it at "
                       f"{KEYS_URL} and that {base_url} is reachable.\n")
        return ""

    def _cmd_logout(self) -> str:
        from ..model_provider.typesafe.auth import clear_credentials, load_credentials

        if not load_credentials(workspace_path=self._workspace_path):
            self._emit("No stored key found.\n")
            return ""
        clear_credentials(workspace_path=self._workspace_path)
        self._emit("Stored TypeSafe key cleared.\n")
        return ""

    def _cmd_status(self) -> str:
        from ..model_provider.typesafe.auth import get_credential_file_path, load_credentials
        from ..model_provider.typesafe.env import (
            ENV_API_KEY, ENV_VENDOR_API_KEY, resolve_base_url,
        )

        lines = ["TypeSafe Authentication Status", "=" * 30, "",
                 "1. plugin_configs.typesafe.api_key: read from the session's profile "
                 "(not visible here)"]
        for i, var in ((2, ENV_API_KEY), (3, ENV_VENDOR_API_KEY)):
            value = os.environ.get(var)
            lines.append(f"{i}. {var}: " + (f"set ({_mask(value)})" if value else "not set"))
        creds = load_credentials(workspace_path=self._workspace_path)
        if creds:
            saved = datetime.fromtimestamp(creds.created_at).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"4. stored: {_mask(creds.api_key)} in "
                         f"{get_credential_file_path(workspace_path=self._workspace_path)} "
                         f"(saved {saved})")
        else:
            lines.append("4. stored: none")
        lines += ["", f"API root: {resolve_base_url()}"]
        self._emit("\n".join(lines) + "\n")
        return ""


def create_plugin() -> TypeSafeAuthPlugin:
    """Factory function for plugin discovery."""
    return TypeSafeAuthPlugin()
