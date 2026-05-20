"""OpenRouter authentication plugin.

Provides user commands for API key authentication with OpenRouter.
This allows users to securely store and manage their OpenRouter API
credentials (``sk-or-...`` keys from https://openrouter.ai/settings/keys).

Commands:
    openrouter-auth login            - Show instructions for getting your key
    openrouter-auth key <api_key>    - Validate and store API key
    openrouter-auth logout           - Clear stored API credentials
    openrouter-auth status           - Show current authentication status
    openrouter-auth help             - Show detailed help
"""

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


class OpenRouterAuthPlugin:
    """Plugin for OpenRouter API key authentication.

    Declares the ``TRAIT_AUTH_PROVIDER`` trait so the server can
    auto-discover this plugin when the OpenRouter provider needs
    credentials.  ``provider_name`` identifies which provider this
    plugin authenticates for.
    """

    plugin_traits: FrozenSet[str] = frozenset({TRAIT_AUTH_PROVIDER})

    def __init__(self) -> None:
        """Initialize the plugin."""
        self._output_callback: Optional[OutputCallback] = None
        self._workspace_path: Optional[str] = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "openrouter_auth"

    @property
    def provider_name(self) -> str:
        """Return the provider name this auth plugin serves."""
        return "openrouter"

    @property
    def provider_display_name(self) -> str:
        """Return human-readable provider name."""
        return "OpenRouter"

    @property
    def credential_env_vars(self) -> List[str]:
        """Return env var names used for credentials by this provider."""
        return ["JAATO_OPENROUTER_API_KEY"]

    def get_default_models(self) -> List[Dict[str, str]]:
        """Return a curated default-model list for OpenRouter.

        OpenRouter exposes hundreds of models — these are well-known
        defaults that span the major upstream providers and price
        tiers.  ``openrouter/auto`` lets OpenRouter pick the best
        model for each request.
        """
        return [
            {
                "name": "openrouter/openrouter/auto",
                "description": "Auto-router — OpenRouter picks the best model per request",
            },
            {
                "name": "openrouter/anthropic/claude-3.5-sonnet",
                "description": "Claude 3.5 Sonnet — strong general reasoning + tool use",
            },
            {
                "name": "openrouter/openai/gpt-4o",
                "description": "GPT-4o — multimodal flagship from OpenAI",
            },
            {
                "name": "openrouter/google/gemini-2.0-flash-exp:free",
                "description": "Gemini 2.0 Flash (experimental) — fast, free tier",
            },
            {
                "name": "openrouter/deepseek/deepseek-r1",
                "description": "DeepSeek-R1 — open reasoning model with chain-of-thought",
            },
            {
                "name": "openrouter/meta-llama/llama-3.3-70b-instruct",
                "description": "Llama 3.3 70B — open Meta flagship",
            },
        ]

    def verify_credentials(self) -> bool:
        """Check if valid credentials exist after authentication."""
        try:
            from ..model_provider.openrouter.env import resolve_api_key
            return resolve_api_key() is not None
        except Exception:
            return False

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin.

        Stores ``workspace_path`` from config so credential storage can
        resolve project-local vs user-global storage paths.
        """
        self._workspace_path = (config or {}).get("workspace_path")

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set the output callback for real-time output during commands."""
        self._output_callback = callback

    def _emit(self, text: str, mode: str = "write") -> None:
        """Emit output through the callback if available."""
        if self._output_callback:
            self._output_callback("openrouter_auth", text, mode)

    def shutdown(self) -> None:
        """Clean up resources."""
        pass

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset — NO-OP for this plugin.

        Phase 1 hotfix (server 0.6.148+): added to satisfy the
        ``ToolPlugin`` / ``EnrichmentPlugin`` protocol's runtime
        ``isinstance`` check.  Per Daniel's litmus test (see
        ``docs/design/runner-cascade-sharing.md`` §4.3), this
        plugin holds no per-session state that the next cascade
        session would benefit from having cleared.  Override in
        future PRs if the litmus test changes.
        """
        pass


    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return empty list — this plugin only provides user commands."""
        return []

    def get_executors(self) -> Dict[str, Any]:
        """Return executor for the user command."""
        return {
            "openrouter-auth": lambda args: self.execute_user_command(
                "openrouter-auth", args
            ),
        }

    def get_system_instructions(self) -> Optional[str]:
        """No system instructions for this plugin."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """User commands don't need permission approval."""
        return ["openrouter-auth"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for authentication."""
        return [
            UserCommand(
                name="openrouter-auth",
                description=(
                    "Manage OpenRouter authentication "
                    "(login, key, logout, status)"
                ),
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name="action",
                        description=(
                            "Action: login, key <api_key>, logout, status, or help"
                        ),
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
        if command != "openrouter-auth":
            return []

        actions = [
            CommandCompletion("login", "Show instructions for getting your API key"),
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
        if command != "openrouter-auth":
            return f"Unknown command: {command}"

        # Preserve case for the API key — only lowercase for action matching.
        raw_action = args.get("action", "").strip()
        action_lower = raw_action.lower()

        if action_lower.startswith("key "):
            api_key = raw_action[4:].strip()
            return self._cmd_key(api_key)
        elif action_lower == "key":
            self._emit("Usage: openrouter-auth key <your_api_key>\n")
            self._emit("\nGet your API key from:\n")
            self._emit("  https://openrouter.ai/settings/keys\n")
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
            ("OpenRouter Auth Command", "bold"),
            ("", ""),
            ("Manage authentication for OpenRouter (https://openrouter.ai),", ""),
            ("a unified gateway for hundreds of models from many vendors.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    openrouter-auth <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    login             Show instructions for getting your OpenRouter API key", "dim"),
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
            ("    1. Run 'openrouter-auth login' to see instructions", ""),
            ("    2. Visit https://openrouter.ai/settings/keys", ""),
            ("    3. Sign in and create an API key (sk-or-...)", ""),
            ("    4. Run 'openrouter-auth key <paste_key_here>'", ""),
            ("    5. Key is validated against GET /api/v1/key and saved", ""),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    openrouter-auth login                      Show setup instructions", "dim"),
            ("    openrouter-auth key sk-or-v1-abc...        Store and validate API key", "dim"),
            ("    openrouter-auth status                     Check authentication status", "dim"),
            ("    openrouter-auth logout                     Clear stored credentials", "dim"),
            ("", ""),
            ("TOKEN STORAGE", "bold"),
            ("    Credentials are stored in:", ""),
            ("    - Project: .jaato/openrouter_auth.json (if .jaato/ exists)", "dim"),
            ("    - User: ~/.jaato/openrouter_auth.json (fallback)", "dim"),
            ("", ""),
            ("    Files are created with restricted permissions (600 on Unix).", "dim"),
            ("", ""),
            ("ENVIRONMENT VARIABLES", "bold"),
            ("    JAATO_OPENROUTER_API_KEY        API key (takes precedence over stored key)", "dim"),
            ("    JAATO_OPENROUTER_BASE_URL       Custom API endpoint", "dim"),
            ("    JAATO_OPENROUTER_MODEL          Default model name", "dim"),
            ("    JAATO_OPENROUTER_CONTEXT_LENGTH Override context window size", "dim"),
            ("    JAATO_OPENROUTER_HTTP_REFERER   App-attribution: site URL", "dim"),
            ("    JAATO_OPENROUTER_APP_TITLE      App-attribution: site title", "dim"),
            ("", ""),
            ("MODELS", "bold"),
            ("    OpenRouter exposes hundreds of models behind one API key.", "dim"),
            ("    Use 'openrouter/auto' to let OpenRouter route to the best model,", "dim"),
            ("    or specify a model explicitly:", "dim"),
            ("      model openrouter/anthropic/claude-3.5-sonnet", "dim"),
            ("      model openrouter/openai/gpt-4o", "dim"),
            ("      model openrouter/deepseek/deepseek-r1", "dim"),
            ("    Browse the full catalog at https://openrouter.ai/models", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - API key format is: sk-or-...", "dim"),
            ("    - API key is validated by hitting GET /api/v1/key before saving", "dim"),
            ("    - Stored credentials are used automatically when connecting", "dim"),
            ("    - Environment variable JAATO_OPENROUTER_API_KEY takes precedence", "dim"),
        ])

    def _cmd_login(self) -> str:
        """Handle the login command — show instructions."""
        from ..model_provider.openrouter.env import resolve_api_key

        existing_key = resolve_api_key()
        if existing_key:
            masked = (
                existing_key[:8] + "..." + existing_key[-4:]
                if len(existing_key) > 12
                else "***"
            )
            self._emit(
                f"Note: You already have an OpenRouter API key configured ({masked}).\n"
                "Using 'openrouter-auth key <new_key>' will replace it.\n\n"
            )

        self._emit("OpenRouter Authentication\n")
        self._emit("=" * 30 + "\n\n")
        self._emit("Step 1: Get your API key from:\n")
        self._emit("  https://openrouter.ai/settings/keys\n")
        self._emit("  Sign in (Google/GitHub/email) → Keys → Create Key\n\n")
        self._emit("Step 2: Copy your API key (sk-or-...) and run:\n")
        self._emit("  openrouter-auth key <paste_your_key_here>\n\n")
        self._emit("The key will be validated and stored securely.\n")

        return ""

    def _cmd_key(self, api_key: str) -> str:
        """Handle the key command — validate and store API key."""
        from ..model_provider.openrouter.auth import (
            login_with_key,
            validate_api_key,
        )
        from ..model_provider.openrouter.env import (
            resolve_api_key,
            resolve_base_url,
        )

        if not api_key:
            self._emit("Error: No API key provided.\n")
            self._emit("Usage: openrouter-auth key <your_api_key>\n")
            return ""

        existing_key = resolve_api_key()
        if existing_key and existing_key == api_key:
            self._emit("This API key is already configured.\n")
            return ""

        self._emit("Validating API key...\n")

        base_url = resolve_base_url()
        valid, detail = validate_api_key(api_key, base_url)
        if valid:
            def on_message(msg: str) -> None:
                self._emit(f"{msg}\n")

            result = login_with_key(
                api_key,
                base_url=base_url,
                on_message=on_message,
                workspace_path=self._workspace_path,
            )
            if result:
                self._emit("\n")
                self._emit("Successfully authenticated with OpenRouter.\n")
                self._emit("Your API key has been stored securely.\n\n")
                self._emit("You can now use the OpenRouter provider, e.g.:\n")
                self._emit("  model openrouter/openrouter/auto\n")
                self._emit("  model openrouter/anthropic/claude-3.5-sonnet\n")
            else:
                self._emit("\nFailed to store credentials.\n")
        else:
            if detail.startswith("network_error"):
                self._emit("\nCould not reach OpenRouter to validate the key.\n\n")
                self._emit(f"Detail: {detail}\n\n")
                self._emit("Please check that:\n")
                self._emit("  - You have internet connectivity\n")
                self._emit("  - openrouter.ai is reachable from your network\n")
                self._emit("  - No firewall or proxy is blocking the request\n")
            elif detail.startswith("rate_limit"):
                self._emit("\nOpenRouter rejected the validation request with a rate limit.\n\n")
                self._emit(f"Detail: {detail}\n\n")
                self._emit(
                    "This usually means your account's quota is exhausted or too\n"
                    "many requests are hitting the API.  The key was NOT saved.\n"
                    "Top up credits at https://openrouter.ai/credits and retry.\n"
                )
            elif detail.startswith("payment_required"):
                self._emit("\nOpenRouter rejected the validation request: payment required.\n\n")
                self._emit(f"Detail: {detail}\n\n")
                self._emit(
                    "Your account requires credits or a paid plan.  The key was\n"
                    "NOT saved.  Add credits at https://openrouter.ai/credits.\n"
                )
            elif detail.startswith("server_error"):
                self._emit("\nOpenRouter returned a server error while validating the key.\n\n")
                self._emit(f"Detail: {detail}\n\n")
                self._emit(
                    "OpenRouter is likely temporarily unavailable.  The key was\n"
                    "NOT saved.  Please retry in a few minutes.\n"
                )
            elif detail.startswith("http_error"):
                self._emit("\nOpenRouter returned an unexpected response while validating the key.\n\n")
                self._emit(f"Detail: {detail}\n\n")
                self._emit(
                    "The key was NOT saved.  Please report this response to\n"
                    "support if it persists.\n"
                )
            else:
                self._emit("\nAPI key validation failed.\n\n")
                if detail:
                    self._emit(f"Detail: {detail}\n\n")
                self._emit("Please check that:\n")
                self._emit("  - The key is correct and complete (starts with sk-or-)\n")
                self._emit("  - Your OpenRouter account is active\n\n")
                self._emit("Get your key from:\n")
                self._emit("  https://openrouter.ai/settings/keys\n")

        return ""

    def _cmd_logout(self) -> str:
        """Handle the logout command."""
        try:
            from ..model_provider.openrouter.auth import (
                clear_credentials,
                load_credentials,
            )

            creds = load_credentials(workspace_path=self._workspace_path)
            if not creds:
                self._emit("No stored credentials found. Already logged out.\n")
                return ""

            clear_credentials(workspace_path=self._workspace_path)
            self._emit(
                "OpenRouter credentials cleared.\n\n"
                "You will need to set JAATO_OPENROUTER_API_KEY or run a new login "
                "to re-authenticate.\n"
            )
            return ""
        except Exception as e:
            self._emit(f"Failed to clear credentials: {e}\n")
            return ""

    def _cmd_status(self) -> str:
        """Handle the status command."""
        try:
            from ..model_provider.openrouter.auth import load_credentials
            from ..model_provider.openrouter.env import (
                DEFAULT_BASE_URL,
                ENV_OPENROUTER_API_KEY,
                resolve_api_key,
                resolve_app_title,
                resolve_base_url,
                resolve_http_referer,
            )
            import os

            lines = ["OpenRouter Authentication Status", "=" * 35, ""]

            creds = load_credentials(workspace_path=self._workspace_path)
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

            env_key = os.environ.get(ENV_OPENROUTER_API_KEY)
            if env_key:
                masked = (
                    env_key[:8] + "..." + env_key[-4:]
                    if len(env_key) > 12
                    else "***"
                )
                lines.append(f"Environment API Key: Set ({masked})")
            else:
                lines.append("Environment API Key: Not set")
                lines.append(f"  Set {ENV_OPENROUTER_API_KEY} environment variable")

            lines.append("")

            effective_key = resolve_api_key()
            if effective_key:
                masked = (
                    effective_key[:8] + "..." + effective_key[-4:]
                    if len(effective_key) > 12
                    else "***"
                )
                source = "Environment" if env_key else "Stored credentials"
                lines.append(f"Effective API Key: {masked}")
                lines.append(f"  Source: {source}")
            else:
                lines.append("Effective API Key: None")
                lines.append(
                    "  Set JAATO_OPENROUTER_API_KEY or use 'openrouter-auth login'"
                )

            lines.append("")

            base_url = resolve_base_url()
            lines.append(f"Base URL: {base_url}")
            if base_url != DEFAULT_BASE_URL:
                lines.append("  (custom endpoint)")

            lines.append("")
            lines.append(f"Attribution Referer: {resolve_http_referer()}")
            lines.append(f"Attribution Title:   {resolve_app_title()}")

            lines.append("")
            lines.append("Priority: Environment > Stored Credentials")

            self._emit("\n".join(lines) + "\n")
            return ""

        except Exception as e:
            self._emit(f"Failed to check status: {e}\n")
            return ""


def create_plugin() -> OpenRouterAuthPlugin:
    """Factory function for plugin discovery."""
    return OpenRouterAuthPlugin()
