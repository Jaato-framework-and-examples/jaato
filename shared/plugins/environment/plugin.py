# shared/plugins/environment/plugin.py

from typing import Dict, List, Optional, Any
from jaato import ToolSchema
import json
import os
import platform
import shutil


class EnvironmentPlugin:
    """Plugin that provides environment awareness tools."""

    VALID_ASPECTS = ["os", "shell", "arch", "cwd", "all"]

    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        return "environment"

    def __init__(self):
        pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called by registry with configuration."""
        pass

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        pass

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Declare the tools this plugin provides."""
        return [
            ToolSchema(
                name="get_environment",
                description=(
                    "Get information about the local execution environment. "
                    "Use this to determine correct shell syntax, path formats, "
                    "and available commands for the current platform."
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
