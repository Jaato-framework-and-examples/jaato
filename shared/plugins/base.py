"""Base protocol for tool plugins."""

from typing import Protocol, List, Dict, Any, Callable, Optional, Tuple, runtime_checkable
from google.genai import types


# Type alias for user commands: (name, description)
UserCommand = Tuple[str, str]


@runtime_checkable
class ToolPlugin(Protocol):
    """Interface that all tool plugins must implement.

    Plugins provide two types of capabilities:
    1. Model tools: Functions the AI model can invoke via function calling
    2. User commands: Commands the user can invoke directly (without model mediation)

    Model tools are declared via get_function_declarations() and executed
    via get_executors(). User commands are declared via get_user_commands()
    and are typically handled by the interactive client.

    Note on "user": In this context, "user" refers to the entity directly
    interfacing with the client - this could be a human operator OR another
    AI agent in an agent-to-agent communication scenario. User commands are
    those that bypass the model's function calling and execute directly.
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

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands this plugin provides.

        User commands are different from model tools:
        - Model tools: Invoked by the AI via function calling (get_function_declarations)
        - User commands: Invoked directly by the user without model mediation

        The "user" here can be:
        - A human operator interacting with the client
        - Another AI agent in agent-to-agent communication scenarios

        Most plugins only provide model tools and should return an empty list here.
        Use this for plugins that also provide direct interaction commands.

        Returns:
            List of (command_name, description) tuples for autocompletion.
            Return empty list if no user-facing commands are provided.
        """
        ...
