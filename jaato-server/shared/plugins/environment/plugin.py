# shared/plugins/environment/plugin.py

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from jaato import ToolSchema
from jaato_sdk.plugins.model_provider.types import TRAIT_REPLAY_SAFE
from datetime import datetime, timezone
from shared.terminal_caps import detect as detect_terminal_caps
import json
import os
import platform
import shutil
import sys
import threading
import time

from shared.path_utils import is_msys2_environment, normalize_path, get_display_separator
from shared.plugins.runner_forwarding import RunnerForwardingMixin

if TYPE_CHECKING:
    from shared.jaato_session import JaatoSession

# Thread-local storage for session reference per agent context
# This prevents subagents from overwriting parent's session reference
_thread_local = threading.local()


class EnvironmentPlugin(RunnerForwardingMixin):
    """Plugin that provides environment awareness tools.

    Supports querying both external environment (OS, shell, architecture)
    and internal context (token usage, GC thresholds) when a session is set.
    """

    VALID_ASPECTS = ["os", "shell", "arch", "cwd", "terminal", "context", "session", "datetime", "network", "all"]

    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        return "environment"

    @property
    def _session(self) -> Optional['JaatoSession']:
        """Get the session for the current thread context.

        Uses thread-local storage so each agent (main or subagent) gets
        its own session reference, preventing subagents from overwriting
        the parent's session.
        """
        return getattr(_thread_local, 'session', None)

    def __init__(self):
        self._workspace_path: Optional[str] = None
        # Custom aspects registered by daemon extensions at runtime.
        # See ``register_aspect()`` for the protocol.
        self._custom_aspects: Dict[str, Any] = {}
        self._custom_aspect_descriptions: Dict[str, str] = {}

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called by registry with configuration."""
        pass

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        # Clear thread-local session for current thread
        if hasattr(_thread_local, 'session'):
            _thread_local.session = None

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


    def set_workspace_path(self, path: str) -> None:
        """Set the workspace root path for CWD reporting.

        Called by the PluginRegistry when broadcasting workspace path
        to all plugins. In daemon mode, os.getcwd() returns the server's
        directory — this method provides the client's actual workspace.

        Args:
            path: Absolute path to the workspace root directory.
        """
        self._workspace_path = path

    def set_session(self, session: 'JaatoSession') -> None:
        """Receive session reference for context usage queries.

        Called by JaatoSession when this plugin is registered.
        Stores in thread-local storage so each agent context gets its own session.

        Args:
            session: The JaatoSession instance.
        """
        _thread_local.session = session

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Declare the tools this plugin provides."""
        return [
            ToolSchema(
                name="get_environment",
                description=(
                    "Get information about the execution environment. "
                    "Use this to determine correct shell syntax, path formats, "
                    "available commands, and current token/context usage."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "aspect": {
                            "type": "string",
                            "enum": self.VALID_ASPECTS,
                            "description": self._build_aspect_description()
                        }
                    },
                    "required": []
                },
                category="system",
                discoverability="core",
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            )
        ]

    def get_executors(self) -> Dict[str, Any]:
        """Map tool names to executor functions.

        Phase 3 §3.4 wave 1: forwards via runner-RPC when a runner
        is attached; falls through to in-process otherwise.
        """
        return self.wrap_executors_for_runner_forwarding({
            "get_environment": self._get_environment,
        })

    def _get_environment(self, args: Dict[str, Any]) -> str:
        """
        Get environment information.

        Args:
            args: Dict with optional 'aspect' key.

        Returns:
            JSON string with environment details.
        """
        aspect = args.get("aspect", "all")

        if aspect not in self.VALID_ASPECTS:
            return json.dumps({
                "error": f"Invalid aspect '{aspect}'. Valid: {self.VALID_ASPECTS}"
            }, indent=2)

        result = {}

        if aspect in ("os", "all"):
            result["os"] = self._get_os_info()

        if aspect in ("shell", "all"):
            result["shell"] = self._get_shell_info()

        if aspect in ("arch", "all"):
            result["arch"] = self._get_arch_info()

        if aspect in ("cwd", "all"):
            result["cwd"] = normalize_path(self._workspace_path or os.getcwd())

        if aspect in ("terminal", "all"):
            result["terminal"] = self._get_terminal_info()

        if aspect in ("context", "all"):
            result["context"] = self._get_context_info()

        if aspect in ("session", "all"):
            result["session"] = self._get_session_info()

        if aspect in ("datetime", "all"):
            result["datetime"] = self._get_datetime_info()

        if aspect in ("network", "all"):
            result["network"] = self._get_network_info()

        # Custom aspects registered by extensions are excluded from "all"
        # — only returned when the model queries them directly.
        if aspect in self._custom_aspects:
            result[aspect] = self._custom_aspects[aspect]()

        # For single aspect (not "all"), flatten the response
        if aspect != "all" and len(result) == 1:
            result = result[aspect]

        return json.dumps(result, indent=2)

    def _get_os_info(self) -> Dict[str, str]:
        """Get operating system information."""
        system = platform.system()

        info = {
            "type": system.lower(),  # 'linux', 'darwin', 'windows'
            "name": system,
            "release": platform.release(),
        }

        # Add friendly name
        if system == "Darwin":
            info["friendly_name"] = "macOS"
        elif system == "Linux":
            info["friendly_name"] = "Linux"
        elif system == "Windows":
            info["friendly_name"] = "Windows"

        # Detect MSYS2/Git Bash environment on Windows
        if is_msys2_environment():
            msystem = os.environ.get('MSYSTEM', '')
            info["msys2"] = True
            info["msystem"] = msystem
            info["friendly_name"] = f"Windows (MSYS2/{msystem})"

        return info

    def _get_shell_info(self) -> Dict[str, Any]:
        """Get shell information.

        Distinguishes between:
        - default: The user's configured login shell ($SHELL on Unix, cmd on Windows)
        - current: The actual shell executing commands (detected via parent process)
        """
        system = platform.system()

        info: Dict[str, Any] = {
            "default": None,
            "current": None,
            "path_separator": os.pathsep,
            "dir_separator": get_display_separator(),
        }

        if system == "Windows":
            # Windows: check for MSYS2/Git Bash, PowerShell, or cmd
            info["default"] = "cmd"

            # Detect MSYS2/Git Bash first (takes priority on Windows)
            if is_msys2_environment():
                msystem = os.environ.get('MSYSTEM', 'MSYS')
                shell_env = os.environ.get('SHELL', '')
                shell_name = os.path.basename(shell_env) if shell_env else 'bash'
                info["default"] = shell_name
                info["current"] = shell_name
                info["msys2"] = True
                info["msystem"] = msystem
                # Under MSYS2, paths should use forward slashes
                info["dir_separator"] = "/"
                info["path_separator"] = ":"
            # Detect actual shell by checking PowerShell-specific env vars
            # PSModulePath is set by PowerShell but not cmd
            elif os.environ.get("PSModulePath"):
                # Distinguish PowerShell Core (pwsh) from Windows PowerShell
                ps_version = os.environ.get("PSVersionTable", "")
                if "Core" in ps_version or shutil.which("pwsh"):
                    # Check if running in pwsh specifically
                    # PSEdition env var or parent process would tell us
                    info["current"] = "pwsh"
                else:
                    info["current"] = "powershell"
            else:
                comspec = os.environ.get("ComSpec", "")
                if "powershell" in comspec.lower():
                    info["current"] = "powershell"
                elif "pwsh" in comspec.lower():
                    info["current"] = "pwsh"
                else:
                    info["current"] = "cmd"

            # Check if PowerShell is available
            info["powershell_available"] = shutil.which("powershell") is not None
            info["pwsh_available"] = shutil.which("pwsh") is not None

        else:
            # Unix-like: $SHELL is the login shell, not necessarily current
            shell_path = os.environ.get("SHELL", "/bin/sh")
            shell_name = os.path.basename(shell_path)
            info["default"] = shell_name
            info["path"] = shell_path

            # Try to detect the actual running shell via parent process
            current_shell = self._detect_current_shell_unix()
            info["current"] = current_shell if current_shell else shell_name

        return info

    def _detect_current_shell_unix(self) -> Optional[str]:
        """Detect the actual running shell on Unix-like systems.

        Checks parent process to determine what shell is actually executing,
        rather than relying on $SHELL which only indicates the login shell.

        Returns:
            Shell name (e.g., 'bash', 'zsh', 'fish') or None if detection fails.
        """
        import subprocess

        try:
            ppid = os.getppid()

            # Try /proc filesystem first (Linux)
            proc_comm = f"/proc/{ppid}/comm"
            if os.path.exists(proc_comm):
                with open(proc_comm, 'r') as f:
                    comm = f.read().strip()
                    # comm contains just the executable name
                    if comm in ('bash', 'zsh', 'fish', 'sh', 'dash', 'ksh', 'tcsh', 'csh'):
                        return comm
                    # Sometimes it's the full command, extract basename
                    basename = os.path.basename(comm)
                    if basename in ('bash', 'zsh', 'fish', 'sh', 'dash', 'ksh', 'tcsh', 'csh'):
                        return basename

            # Fallback: use ps command (works on macOS and other Unix)
            result = subprocess.run(
                ['ps', '-p', str(ppid), '-o', 'comm='],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                comm = result.stdout.strip()
                # ps may return full path or just name
                basename = os.path.basename(comm)
                # Handle common variations like '-bash' (login shell indicator)
                if basename.startswith('-'):
                    basename = basename[1:]
                if basename in ('bash', 'zsh', 'fish', 'sh', 'dash', 'ksh', 'tcsh', 'csh'):
                    return basename

        except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

        return None

    def _get_arch_info(self) -> Dict[str, str]:
        """Get architecture information."""
        machine = platform.machine()

        info = {
            "machine": machine,
            "processor": platform.processor() or machine,
        }

        # Add normalized architecture name
        machine_lower = machine.lower()
        if machine_lower in ("x86_64", "amd64"):
            info["normalized"] = "x86_64"
        elif machine_lower in ("arm64", "aarch64"):
            info["normalized"] = "arm64"
        elif machine_lower in ("i386", "i686", "x86"):
            info["normalized"] = "x86"
        else:
            info["normalized"] = machine_lower

        return info

    def _get_terminal_info(self) -> Dict[str, Any]:
        """Get terminal emulation and capability information.

        Delegates to shared.terminal_caps for detection logic, which
        caches results process-wide. This allows other plugins (e.g.,
        mermaid_formatter) to access the same data without triggering
        a model tool call.
        """
        return detect_terminal_caps()

    def _get_context_info(self) -> Dict[str, Any]:
        """Get context window usage and GC threshold information.

        Returns token usage, context limits, and garbage collection settings.
        Requires a session to be set via set_session().
        """
        if self._session is None:
            return {
                "error": "Session not available. Context info requires session injection.",
                "hint": "Use set_session_plugin() or ensure plugin is properly registered."
            }

        # Get context usage from session
        usage = self._session.get_context_usage()

        # Get GC config if available
        gc_config = getattr(self._session, '_gc_config', None)

        result = {
            # Token usage
            "model": usage.get("model", "unknown"),
            "context_limit": usage.get("context_limit", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "tokens_remaining": usage.get("tokens_remaining", 0),
            "percent_used": round(usage.get("percent_used", 0), 2),
            "turns": usage.get("turns", 0),
        }

        # Add GC thresholds if available
        if gc_config is not None:
            result["gc"] = {
                "threshold_percent": gc_config.threshold_percent,
                "auto_trigger": gc_config.auto_trigger,
                "preserve_recent_turns": gc_config.preserve_recent_turns,
            }
            if gc_config.max_turns is not None:
                result["gc"]["max_turns"] = gc_config.max_turns
        else:
            result["gc"] = None

        return result

    def _get_session_info(self) -> Dict[str, Any]:
        """Get session identifier and agent information.

        Returns session ID, agent type, and agent name.
        Requires a session to be set via set_session().
        """
        if self._session is None:
            return {
                "error": "Session not available. Session info requires session injection.",
                "hint": "Use set_session_plugin() or ensure plugin is properly registered."
            }

        result: Dict[str, Any] = {}

        # Get actual session ID from session plugin (e.g., "20251207_143022")
        session_plugin = getattr(self._session, '_session_plugin', None)
        if session_plugin and hasattr(session_plugin, 'get_current_session_id'):
            current_session_id = session_plugin.get_current_session_id()
            if current_session_id:
                result["session_id"] = current_session_id

        # Get agent identifier within the session (e.g., "main", "subagent_1")
        agent_id = getattr(self._session, '_agent_id', None)
        if agent_id:
            result["agent_id"] = agent_id

        # Get agent type (main, subagent, etc.)
        agent_type = getattr(self._session, '_agent_type', None)
        if agent_type:
            result["agent_type"] = agent_type

        # Get agent name if set
        agent_name = getattr(self._session, '_agent_name', None)
        if agent_name:
            result["agent_name"] = agent_name

        # Also expose via environment variable if set
        env_session_id = os.environ.get("JAATO_SESSION_ID")
        if env_session_id:
            result["env_session_id"] = env_session_id

        return result

    def _get_datetime_info(self) -> Dict[str, Any]:
        """Get current date, time, timezone, and UTC offset information.

        Returns:
            Dict containing local datetime, UTC datetime, timezone name, and UTC offset.
        """
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)

        # Get timezone name from time module (more reliable than datetime.tzname)
        # Use tm_zone if available (Unix), otherwise use tzname tuple (Windows)
        local_time = time.localtime()
        if hasattr(local_time, 'tm_zone'):
            tz_name = local_time.tm_zone
        else:
            # Fallback for platforms without tm_zone
            tz_name = time.tzname[local_time.tm_isdst] if local_time.tm_isdst >= 0 else time.tzname[0]

        # Calculate UTC offset
        # time.timezone is seconds west of UTC (negative for east)
        # time.daylight indicates if DST is defined, local_time.tm_isdst if DST is active
        if local_time.tm_isdst > 0 and time.daylight:
            utc_offset_seconds = -time.altzone
        else:
            utc_offset_seconds = -time.timezone

        # Format offset as ±HH:MM
        offset_hours, offset_remainder = divmod(abs(utc_offset_seconds), 3600)
        offset_minutes = offset_remainder // 60
        offset_sign = '+' if utc_offset_seconds >= 0 else '-'
        utc_offset_str = f"{offset_sign}{offset_hours:02d}:{offset_minutes:02d}"

        return {
            "local": now.strftime("%Y-%m-%d %H:%M:%S"),
            "utc": utc_now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": tz_name,
            "utc_offset": utc_offset_str,
            "iso_local": now.isoformat(),
            "iso_utc": utc_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def register_aspect(
        self,
        name: str,
        handler: Any,
        description: str = "",
    ) -> None:
        """Register a custom environment aspect provided by a daemon extension.

        Custom aspects extend the ``get_environment`` tool with additional
        data sources at runtime.  They are **excluded from the ``'all'``
        aspect** — the model must query them explicitly by name.

        This is the primary mechanism for daemon extensions to surface
        dynamic data (e.g., cluster topology, feature flags) through the
        environment tool without modifying the plugin's source code.

        The registered aspect automatically appears in:

        * ``VALID_ASPECTS`` — the validation list.
        * The ``aspect`` enum in the tool schema — so the model can
          discover and select it.
        * The tool schema description — using the ``description`` arg.

        Args:
            name: The aspect name (e.g., ``"jaato_agentic_servers"``).
                Must not collide with a built-in aspect.
            handler: A callable ``() -> dict`` that returns the aspect
                data.  Called each time the model queries this aspect.
            description: Human-readable description appended to the tool
                schema so the model knows what this aspect provides.

        Example (from a daemon extension's session hook)::

            env_plugin.register_aspect(
                "jaato_agentic_servers",
                lambda: self._get_cluster_topology(),
                description=(
                    "cluster topology: this server + peers with health, "
                    "load, and state (excluded from 'all'; query explicitly "
                    "when planning subagent delegation)"
                ),
            )
        """
        self._custom_aspects[name] = handler
        if description:
            self._custom_aspect_descriptions[name] = description
        if name not in self.VALID_ASPECTS:
            # Insert before 'all' so the enum ordering stays sensible.
            idx = self.VALID_ASPECTS.index("all")
            self.VALID_ASPECTS.insert(idx, name)

    def _build_aspect_description(self) -> str:
        """Build the ``aspect`` parameter description for the tool schema.

        Combines the built-in aspect descriptions with any custom aspects
        registered via ``register_aspect()``.  Custom aspects are appended
        dynamically, so the model always sees the current set.

        Returns:
            A single description string for the ``aspect`` enum.
        """
        parts = [
            "Which aspect of the environment to query. ",
            "'os' = operating system, ",
            "'shell' = shell type, ",
            "'arch' = CPU architecture, ",
            "'cwd' = current working directory, ",
            "'terminal' = terminal emulation, capabilities, and TTY detection, ",
            "'context' = token usage and GC thresholds, ",
            "'session' = current session identifier and agent info, ",
            "'datetime' = current date, time, timezone, and UTC offset, ",
            "'network' = proxy settings, proxy authentication, SSL/TLS config, and no-proxy rules, ",
        ]

        # Append custom aspect descriptions
        for name, desc in self._custom_aspect_descriptions.items():
            parts.append(f"'{name}' = {desc}, ")

        # Closing description for 'all'
        custom_names = list(self._custom_aspects.keys())
        if custom_names:
            exclusions = " or ".join(f"'{n}'" for n in custom_names)
            parts.append(f"'all' = everything except {exclusions} (default)")
        else:
            parts.append("'all' = all of the above (default)")

        return "".join(parts)

    def _get_network_info(self) -> Dict[str, Any]:
        """Get network connectivity configuration.

        Reports proxy settings, proxy authentication method, SSL/TLS
        verification config, and no-proxy rules by inspecting environment
        variables. Sensitive parts of proxy URLs (userinfo) are masked.

        Returns:
            Dict with 'proxy', 'proxy_auth', 'ssl', and 'no_proxy' sub-dicts.
        """
        result: Dict[str, Any] = {}

        # --- Proxy configuration ---
        proxy_info: Dict[str, Any] = {
            "http_proxy": self._mask_proxy_url(
                os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
            ),
            "https_proxy": self._mask_proxy_url(
                os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            ),
            "configured": False,
        }
        proxy_info["configured"] = bool(
            proxy_info["http_proxy"] or proxy_info["https_proxy"]
        )
        result["proxy"] = proxy_info

        # --- Proxy authentication ---
        auth_info: Dict[str, Any] = {
            "type": "none",
        }

        kerberos_enabled = os.environ.get("JAATO_KERBEROS_PROXY", "").lower() in (
            "true", "1", "yes",
        )
        if kerberos_enabled:
            auth_info["type"] = "kerberos"
            auth_info["kerberos_enabled"] = True
        elif proxy_info["configured"]:
            # Detect auth type from the raw proxy URL (before masking)
            raw_url = (
                os.environ.get("HTTPS_PROXY")
                or os.environ.get("https_proxy")
                or os.environ.get("HTTP_PROXY")
                or os.environ.get("http_proxy")
                or ""
            )
            if self._proxy_url_has_userinfo(raw_url):
                auth_info["type"] = "basic"

        result["proxy_auth"] = auth_info

        # --- SSL / TLS ---
        ssl_verify_raw = os.environ.get("JAATO_SSL_VERIFY")
        if ssl_verify_raw is not None:
            ssl_verify = ssl_verify_raw.lower() not in ("false", "0", "no")
        else:
            ssl_verify = True  # default

        ssl_info: Dict[str, Any] = {
            "verify": ssl_verify,
        }

        # Custom CA bundle paths recognised by common HTTP libraries
        for env_var in ("SSL_CERT_FILE", "SSL_CERT_DIR",
                        "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            val = os.environ.get(env_var)
            if val:
                ssl_info[env_var.lower()] = val

        result["ssl"] = ssl_info

        # --- No-proxy rules ---
        no_proxy_info: Dict[str, Any] = {}

        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        if no_proxy:
            no_proxy_info["no_proxy"] = no_proxy

        jaato_no_proxy = os.environ.get("JAATO_NO_PROXY")
        if jaato_no_proxy:
            no_proxy_info["jaato_no_proxy"] = jaato_no_proxy

        result["no_proxy"] = no_proxy_info if no_proxy_info else None

        return result

    @staticmethod
    def _mask_proxy_url(url: Optional[str]) -> Optional[str]:
        """Mask userinfo (credentials) in a proxy URL.

        Replaces ``user:password@`` with ``***:***@`` so that the proxy host
        and port remain visible without leaking secrets.

        Args:
            url: Raw proxy URL, or None.

        Returns:
            Masked URL string, or None if input was None/empty.
        """
        if not url:
            return None
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            if parsed.username or parsed.password:
                # Rebuild netloc with masked credentials
                masked_netloc = "***:***@"
                if parsed.hostname:
                    masked_netloc += parsed.hostname
                if parsed.port:
                    masked_netloc += f":{parsed.port}"
                return urlunparse(parsed._replace(netloc=masked_netloc))
        except Exception:
            pass
        return url

    @staticmethod
    def _proxy_url_has_userinfo(url: str) -> bool:
        """Check whether a proxy URL contains embedded credentials.

        Args:
            url: Raw proxy URL string.

        Returns:
            True if the URL contains a username (and optionally password).
        """
        if not url:
            return False
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return bool(parsed.username)
        except Exception:
            return False

    # ==================== Required Protocol Methods ====================

    def get_system_instructions(self) -> Optional[str]:
        """Instructions for the model about environment tools.

        Note: Sandbox awareness is handled by JaatoRuntime to ensure it's
        always included regardless of which plugins are exposed.
        """
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """Environment tools are safe, read-only operations."""
        return ["get_environment"]

    def get_user_commands(self) -> List:
        """No user commands provided."""
        return []


def create_plugin() -> EnvironmentPlugin:
    """Factory function to create the environment plugin instance."""
    return EnvironmentPlugin()
