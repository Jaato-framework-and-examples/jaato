"""Base protocol for tool plugins."""

from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Callable, Optional, NamedTuple, runtime_checkable
from google.genai import types


@dataclass
class PermissionDisplayInfo:
    """Display information for permission approval UI.

    Plugins can provide this to customize how their tools are displayed
    when requesting permission from the user/actor.

    Attributes:
        summary: Brief one-line description (e.g., "Update file: src/main.py")
        details: Full content to display (e.g., unified diff)
        format_hint: How to render details - "diff", "json", "text", "code"
        language: Programming language for syntax highlighting (when format_hint="code")
        truncated: Whether details were truncated due to size
        original_lines: Original line count before truncation (if truncated)
    """
    summary: str
    details: str
    format_hint: str = "text"
    language: Optional[str] = None
    truncated: bool = False
    original_lines: Optional[int] = None


class UserCommand(NamedTuple):
    """Declaration of a user-facing command.

    User commands can be invoked directly by the user (human or agent)
    without going through the model's function calling.

    Attributes:
        name: Command name for invocation and autocompletion.
        description: Brief description shown in autocompletion/help.
        share_with_model: If True, command output is added to conversation
            history so the model can see/use it. If False (default),
            output is only shown to the user.
    """
    name: str
    description: str
    share_with_model: bool = False


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

    # Optional method - not part of the required protocol, but recognized by
    # the permission system if implemented:
    #
    # def format_permission_request(
    #     self,
    #     tool_name: str,
    #     arguments: Dict[str, Any],
    #     actor_type: str
    # ) -> Optional[PermissionDisplayInfo]:
    #     """Format a permission request for display.
    #
    #     This optional method allows plugins to provide custom formatting for
    #     their tools when displayed in the permission approval UI. If not
    #     implemented or returns None, the default JSON display is used.
    #
    #     Args:
    #         tool_name: Name of the tool being executed
    #         arguments: Arguments passed to the tool
    #         actor_type: Type of actor requesting approval ("console", "webhook", "file")
    #
    #     Returns:
    #         PermissionDisplayInfo with formatted content, or None to use default.
    #     """
    #     ...
