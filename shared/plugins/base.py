"""Base protocol for tool plugins."""

from typing import Protocol, List, Dict, Any, Callable, Optional, runtime_checkable
from google.genai import types


@runtime_checkable
class ToolPlugin(Protocol):
    """Interface that all tool plugins must implement.

    Plugins provide tool executors and their FunctionDeclaration
    objects for the AI model to invoke.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        ...

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Return Vertex AI FunctionDeclaration objects for this plugin's tools."""
        ...

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return a mapping of tool names to their executor callables.

        Each executor should accept a dict of arguments and return a
        JSON-serializable result.
        """
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called once when the plugin is enabled.

        Args:
            config: Optional configuration dict for plugin-specific settings.
        """
        ...

    def shutdown(self) -> None:
        """Called when the plugin is disabled. Clean up resources here."""
        ...

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing this plugin's capabilities.

        These instructions are prepended to the user's prompt to help the model
        understand what tools are available and how to use them.

        Returns:
            A string with instructions, or None if no instructions are needed.
        """
        ...

    def get_auto_approved_tools(self) -> List[str]:
        """Return list of tool names that should be auto-approved without permission prompts.

        Tools returned here will be added to the permission whitelist automatically.
        Use this for tools that have no security implications (e.g., progress tracking).

        Returns:
            List of tool names, or empty list if all tools require permission.
        """
        ...
