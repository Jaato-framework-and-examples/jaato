"""Base protocol for tool plugins."""

import fnmatch
from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Callable, Optional, NamedTuple, runtime_checkable
from google.genai import types


# Output callback type for real-time output from model and plugins
#
# Parameters:
#   source: Origin of the output ("model", plugin name, "system", etc.)
#   text: The output text content
#   mode: How to handle the output:
#         - "write": Start a new output block
#         - "append": Add to the current block from the same source
#
# The frontend/client decides how to render (terminal, web UI, logging).
# Interleaving of outputs from different sources is a frontend concern.
OutputCallback = Callable[[str, str, str], None]


@dataclass
class PromptEnrichmentResult:
    """Result of prompt enrichment by a plugin.

    Plugins that subscribe to prompt enrichment can inspect and optionally
    modify user prompts before they are sent to the model.

    Attributes:
        prompt: The (possibly modified) prompt text.
        metadata: Optional metadata about the enrichment (e.g., detected references).
    """
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def model_matches_requirements(model_name: str, patterns: List[str]) -> bool:
    """Check if a model name matches any of the required patterns.

    Args:
        model_name: The model name to check (e.g., 'gemini-3-pro-preview').
        patterns: List of glob patterns (e.g., ['gemini-3-pro*', 'gemini-3.5-*']).

    Returns:
        True if model_name matches at least one pattern, False otherwise.
    """
    return any(fnmatch.fnmatch(model_name, pattern) for pattern in patterns)


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


class CommandCompletion(NamedTuple):
    """A completion option for command arguments.

    Used by plugins to provide autocompletion hints for their user commands.

    Attributes:
        value: The completion value to insert.
        description: Brief description shown in completion menu.
    """
    value: str
    description: str = ""


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
        """Return list of tool/command names that should be auto-approved without permission prompts.

        Tools returned here will be added to the permission whitelist automatically.
        Use this for:
        - Read-only tools with no security implications (e.g., progress tracking)
        - User commands that shouldn't trigger permission prompts (since they are
          invoked directly by the user, not by the model)

        IMPORTANT: User commands defined in get_user_commands() should typically
        be listed here. Since users invoke these commands directly (not the model),
        they shouldn't require permission approval. Forgetting to include user
        commands here will cause unexpected permission prompts.

        Returns:
            List of tool/command names, or empty list if all require permission.
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

    # ==================== Optional Protocol Extensions ====================
    #
    # The following methods are optional extensions to the base protocol.
    # Plugins can implement these for additional functionality.
    #
    # Model Requirements:
    #
    # def get_model_requirements(self) -> Optional[List[str]]:
    #     """Return glob patterns for models this plugin requires.
    #
    #     If the current model doesn't match any pattern, the plugin will
    #     not be loaded (graceful failure with warning).
    #
    #     Examples:
    #         ["gemini-3-pro*", "gemini-3.5-*"]  # Requires Gemini 3+
    #         ["gemini-2.5-*", "gemini-3-*"]     # Requires 2.5 or 3.x
    #         None                               # Works with any model (default)
    #
    #     Returns:
    #         List of glob patterns, or None if plugin works with any model.
    #     """
    #     ...
    #
    # Prompt Enrichment:
    #
    # def subscribes_to_prompt_enrichment(self) -> bool:
    #     """Return True if this plugin wants to enrich prompts before sending.
    #
    #     Plugins that subscribe will have their enrich_prompt() method called
    #     with the user's prompt before it is sent to the model. This allows
    #     plugins to:
    #     - Detect and process @references (e.g., @file.png, @url)
    #     - Add context or instructions based on prompt content
    #     - Track referenced resources for later tool calls
    #
    #     Returns:
    #         True to subscribe, False otherwise (default).
    #     """
    #     ...
    #
    # def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
    #     """Enrich a user prompt before sending to the model.
    #
    #     Called only if subscribes_to_prompt_enrichment() returns True.
    #     The plugin can inspect and modify the prompt, returning the
    #     (possibly modified) prompt along with metadata about what was found.
    #
    #     IMPORTANT: Plugins should NOT remove @references from the prompt.
    #     The framework handles @reference cleanup after all plugins have
    #     processed the prompt.
    #
    #     Args:
    #         prompt: The user's original prompt text.
    #
    #     Returns:
    #         PromptEnrichmentResult with the enriched prompt and metadata.
    #     """
    #     ...

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
    #
    # Command Completions:
    #
    # def get_command_completions(
    #     self,
    #     command: str,
    #     args: List[str]
    # ) -> List[CommandCompletion]:
    #     """Return completion options for a user command's arguments.
    #
    #     This optional method allows plugins to provide autocompletion for
    #     their user commands. The client calls this when the user is typing
    #     a command and requests completion (e.g., pressing Tab).
    #
    #     Args:
    #         command: The command name (e.g., "permissions")
    #         args: Arguments typed so far (may be empty or contain partial input)
    #               For "permissions default al", args would be ["default", "al"]
    #
    #     Returns:
    #         List of CommandCompletion options matching the current input.
    #         Return empty list if no completions available.
    #
    #     Example:
    #         # For "permissions " (no args yet)
    #         get_command_completions("permissions", [])
    #         -> [CommandCompletion("show", "Display policy"), ...]
    #
    #         # For "permissions default a"
    #         get_command_completions("permissions", ["default", "a"])
    #         -> [CommandCompletion("allow", "Auto-approve"), CommandCompletion("ask", "Prompt")]
    #     """
    #     ...
