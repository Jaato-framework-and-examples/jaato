# shared/plugins/environment/plugin.py

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from jaato import ToolSchema
import json
import os
import platform
import shutil

if TYPE_CHECKING:
    from shared.jaato_session import JaatoSession


class EnvironmentPlugin:
    """Plugin that provides environment awareness tools.

    Supports querying both external environment (OS, shell, architecture)
    and internal context (token usage, GC thresholds) when a session is set.
    """

    VALID_ASPECTS = ["os", "shell", "arch", "cwd", "terminal", "context", "all"]

    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        return "environment"

    def __init__(self):
        self._session: Optional['JaatoSession'] = None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called by registry with configuration."""
        pass

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        self._session = None

    def set_session(self, session: 'JaatoSession') -> None:
        """Receive session reference for context usage queries.

        Called by JaatoSession when this plugin is registered.

        Args:
            session: The JaatoSession instance.
        """
        self._session = session

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
                            "description": (
                                "Which aspect of the environment to query. "
                                "'os' = operating system, "
                                "'shell' = shell type, "
                                "'arch' = CPU architecture, "
                                "'cwd' = current working directory, "
                                "'terminal' = terminal emulation and capabilities, "
                                "'context' = token usage and GC thresholds, "
                                "'all' = everything (default)"
                            )
                        }
                    },
                    "required": []
                }
            )
        ]

    def get_executors(self) -> Dict[str, Any]:
        """Map tool names to executor functions."""
        return {
            "get_environment": self._get_environment
        }

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
            result["cwd"] = os.getcwd()

        if aspect in ("terminal", "all"):
            result["terminal"] = self._get_terminal_info()

        if aspect in ("context", "all"):
            result["context"] = self._get_context_info()

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

        return info

    def _get_shell_info(self) -> Dict[str, Any]:
        """Get shell information."""
        system = platform.system()

        info = {
            "default": None,
            "current": None,
            "path_separator": os.pathsep,
            "dir_separator": os.sep,
        }

        if system == "Windows":
            # Windows: check for PowerShell vs cmd
            info["default"] = "cmd"
            comspec = os.environ.get("ComSpec", "")
            if "powershell" in comspec.lower():
                info["current"] = "powershell"
            elif "pwsh" in comspec.lower():
                info["current"] = "pwsh"  # PowerShell Core
            else:
                info["current"] = "cmd"

            # Check if PowerShell is available
            info["powershell_available"] = shutil.which("powershell") is not None
            info["pwsh_available"] = shutil.which("pwsh") is not None

        else:
            # Unix-like: check SHELL env var
            shell_path = os.environ.get("SHELL", "/bin/sh")
            shell_name = os.path.basename(shell_path)
            info["default"] = shell_name
            info["current"] = shell_name
            info["path"] = shell_path

        return info

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
        """Get terminal emulation and capability information."""
        info: Dict[str, Any] = {
            "term": os.environ.get("TERM"),
            "term_program": os.environ.get("TERM_PROGRAM"),
            "colorterm": os.environ.get("COLORTERM"),
        }

        # Detect terminal multiplexers
        multiplexer = None
        if os.environ.get("TMUX"):
            multiplexer = "tmux"
        elif os.environ.get("STY"):
            multiplexer = "screen"
        elif "screen" in (info["term"] or ""):
            multiplexer = "screen"
        info["multiplexer"] = multiplexer

        # Detect color capability
        term = (info["term"] or "").lower()
        colorterm = (info["colorterm"] or "").lower()
        if colorterm in ("truecolor", "24bit") or "truecolor" in colorterm:
            info["color_depth"] = "24bit"
        elif "256color" in term or "256" in colorterm:
            info["color_depth"] = "256"
        elif term and term != "dumb":
            info["color_depth"] = "basic"
        else:
            info["color_depth"] = "none"

        # Detect if running in common terminal emulators
        term_program = info["term_program"] or ""
        if term_program:
            info["emulator"] = term_program
        elif "xterm" in term:
            info["emulator"] = "xterm-compatible"
        elif "linux" in term:
            info["emulator"] = "linux-console"
        else:
            info["emulator"] = None

        return info

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

    # ==================== Required Protocol Methods ====================

    def get_system_instructions(self) -> Optional[str]:
        """Instructions for the model about environment tools."""
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
