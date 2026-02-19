"""Base protocol for tool plugins."""

import fnmatch
from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Callable, FrozenSet, Optional, NamedTuple, runtime_checkable

from .model_provider.types import ToolSchema


# ---------------------------------------------------------------------------
# Plugin-level traits
#
# Analogous to the tool-level ``TRAIT_*`` constants in
# ``model_provider.types``, but apply to **plugins** rather than individual
# tool schemas.  Plugins declare traits via a ``plugin_traits`` attribute
# (a ``FrozenSet[str]``) and consumers discover capabilities through
# trait membership.
# ---------------------------------------------------------------------------

TRAIT_AUTH_PROVIDER = "auth_provider"
"""Plugin-level trait identifying authentication plugins.

Plugins declaring this trait provide interactive authentication for a
model provider.  They MUST also expose a ``provider_name`` property
(``str``) returning the provider identifier they authenticate for
(e.g. ``"anthropic"``, ``"zhipuai"``, ``"github_models"``).

The server uses this trait to find all auth plugins, then matches
``provider_name`` to select the right one — no hardcoded mapping needed.

Usage::

    from shared.plugins.base import TRAIT_AUTH_PROVIDER

    class MyAuthPlugin:
        plugin_traits = frozenset({TRAIT_AUTH_PROVIDER})

        @property
        def provider_name(self) -> str:
            return "my_provider"
"""


# Output callback type for real-time output from model and plugins
#
# Parameters:
#   source: Origin of the output ("model", plugin name, "system", etc.)
#   text: The output text content
#   mode: How to handle the output:
#         - "write": Start a new output block
#         - "append": Add to the current block from the same source
#         - "flush": Process any pending/buffered output before continuing.
#                    Sent with source="system" and empty text before UI events
#                    like tool tree rendering. Clients with async/buffered output
#                    should ensure all pending text is displayed when receiving this.
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


@dataclass
class SystemInstructionEnrichmentResult:
    """Result of system instruction enrichment by a plugin.

    Plugins that subscribe to system instruction enrichment can inspect and
    optionally modify the combined system instructions before they are sent
    to the model. This is useful for extracting embedded content (like templates)
    that should be made available to tools.

    Attributes:
        instructions: The (possibly modified) system instructions text.
        metadata: Optional metadata about the enrichment.
    """
    instructions: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEnrichmentResult:
    """Result of tool result enrichment by a plugin.

    Plugins that subscribe to tool result enrichment can inspect and optionally
    modify tool execution results before they are sent back to the model. This
    is useful for extracting embedded content (like templates) from file reads
    or command outputs.

    Attributes:
        result: The (possibly modified) tool result text.
        metadata: Optional metadata about the enrichment.
    """
    result: str
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
    when requesting permission from the user/channel.

    Attributes:
        summary: Brief one-line description (e.g., "Update file: src/main.py")
        details: Full content to display (e.g., unified diff)
        format_hint: How to render details - "diff", "json", "text", "code"
        language: Programming language for syntax highlighting (when format_hint="code")
        truncated: Whether details were truncated due to size
        original_lines: Original line count before truncation (if truncated)
        warnings: Optional security/analysis warnings to display separately
        warning_level: Severity level for warnings ("info", "warning", "error")
        pre_validation_error: If set, the operation is known to fail.
            The permission system skips the prompt and returns this error
            directly to the model so it can retry.
    """
    summary: str
    details: str
    format_hint: str = "text"
    language: Optional[str] = None
    truncated: bool = False
    original_lines: Optional[int] = None
    warnings: Optional[str] = None
    warning_level: Optional[str] = None
    pre_validation_error: Optional[str] = None


@dataclass
class HelpLines:
    """Styled help text for display in the pager.

    Used by plugin commands to return help text with styling information.
    The server detects this type and emits HelpTextEvent for pager display.

    Attributes:
        lines: List of (text, style) tuples where style is one of:
               "bold", "dim", "" (normal), or semantic names.
    """
    lines: List[tuple]


class CommandCompletion(NamedTuple):
    """A completion option for command arguments.

    Used by plugins to provide autocompletion hints for their user commands.

    Attributes:
        value: The completion value to insert.
        description: Brief description shown in completion menu.
    """
    value: str
    description: str = ""


class CommandParameter(NamedTuple):
    """Definition of a command parameter for argument parsing.

    Used by plugins to declare the argument schema for their user commands,
    enabling generic argument parsing in the client.

    Attributes:
        name: Parameter name (used as key in parsed args dict).
        description: Brief description for help text.
        required: Whether the parameter is required (default: False).
        capture_rest: If True, this parameter captures all remaining args as a
            single string (useful for descriptions). Only valid for last param.
    """
    name: str
    description: str = ""
    required: bool = False
    capture_rest: bool = False


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
        parameters: Optional list of CommandParameter definitions for
            argument parsing. If provided, enables generic parsing.
    """
    name: str
    description: str
    share_with_model: bool = False
    parameters: Optional[List['CommandParameter']] = None


def parse_command_args(
    command: 'UserCommand',
    raw_args: str
) -> Dict[str, Any]:
    """Parse raw argument string into named arguments based on command schema.

    Uses the command's parameters definition to map positional arguments
    to named parameters.

    Args:
        command: The UserCommand with optional parameters definition.
        raw_args: Raw argument string from user input.

    Returns:
        Dictionary of named arguments. If command has no parameters defined,
        returns {"args": [list of split args]}.
    """
    raw_args = raw_args.strip()
    result: Dict[str, Any] = {}

    if not command.parameters:
        # No schema defined - return raw args as list
        return {"args": raw_args.split() if raw_args else []}

    if not raw_args:
        return result

    # Parse according to parameter definitions
    arg_parts = raw_args.split()
    arg_index = 0

    for param in command.parameters:
        if arg_index >= len(arg_parts):
            break

        if param.capture_rest:
            # Capture all remaining args as single string
            result[param.name] = ' '.join(arg_parts[arg_index:])
            break
        else:
            val = arg_parts[arg_index]
            # Only convert to int if purely numeric (no underscores)
            if val.isdigit():
                result[param.name] = int(val)
            else:
                result[param.name] = val
            arg_index += 1

    return result


@runtime_checkable
class ToolPlugin(Protocol):
    """Interface that all tool plugins must implement.

    Plugins provide two types of capabilities:
    1. Model tools: Functions the AI model can invoke via function calling
    2. User commands: Commands the user can invoke directly (without model mediation)

    Model tools are declared via get_tool_schemas() and executed via
    get_executors(). User commands are declared via get_user_commands()
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

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return provider-agnostic tool schemas for this plugin's tools.

        Each ToolSchema defines a tool that the model can invoke via
        function calling. The schema is converted to the appropriate
        provider-specific format by the model provider plugin.

        Returns:
            List of ToolSchema objects describing available tools.
        """
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
        - Model tools: Invoked by the AI via function calling (get_tool_schemas)
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
    #
    # System Instruction Enrichment:
    #
    # def subscribes_to_system_instruction_enrichment(self) -> bool:
    #     """Return True if this plugin wants to enrich system instructions.
    #
    #     Plugins that subscribe will have their enrich_system_instructions()
    #     method called with the combined system instructions from all plugins
    #     (including content from references plugin like MODULE.md). This allows
    #     plugins to:
    #     - Extract embedded templates for later use by tools
    #     - Add annotations or summaries based on instruction content
    #     - Process references or links found in system instructions
    #
    #     Returns:
    #         True to subscribe, False otherwise (default).
    #     """
    #     ...
    #
    # def get_system_instruction_enrichment_priority(self) -> int:
    #     """Return the priority for system instruction enrichment.
    #
    #     Lower values run first. Default is 50.
    #
    #     Returns:
    #         Integer priority (lower = earlier).
    #     """
    #     ...
    #
    # def enrich_system_instructions(
    #     self,
    #     instructions: str
    # ) -> SystemInstructionEnrichmentResult:
    #     """Enrich combined system instructions before sending to the model.
    #
    #     Called only if subscribes_to_system_instruction_enrichment() returns True.
    #     The plugin receives the combined system instructions from all plugins
    #     and can inspect/modify them.
    #
    #     Args:
    #         instructions: Combined system instructions text from all plugins.
    #
    #     Returns:
    #         SystemInstructionEnrichmentResult with enriched instructions.
    #     """
    #     ...
    #
    # Tool Result Enrichment:
    #
    # def subscribes_to_tool_result_enrichment(self) -> bool:
    #     """Return True if this plugin wants to enrich tool results.
    #
    #     Plugins that subscribe will have their enrich_tool_result() method
    #     called with each tool execution result before it is sent back to the
    #     model. This allows plugins to:
    #     - Extract embedded templates from file contents or command output
    #     - Add annotations based on result content
    #     - Process or transform tool outputs
    #
    #     Returns:
    #         True to subscribe, False otherwise (default).
    #     """
    #     ...
    #
    # def get_tool_result_enrichment_priority(self) -> int:
    #     """Return the priority for tool result enrichment.
    #
    #     Lower values run first. Default is 50.
    #
    #     Returns:
    #         Integer priority (lower = earlier).
    #     """
    #     ...
    #
    # def enrich_tool_result(
    #     self,
    #     tool_name: str,
    #     result: str
    # ) -> ToolResultEnrichmentResult:
    #     """Enrich a tool execution result before sending to the model.
    #
    #     Called only if subscribes_to_tool_result_enrichment() returns True.
    #     The plugin receives the tool name and its result string.
    #
    #     Args:
    #         tool_name: Name of the tool that produced the result.
    #         result: The tool's output as a string.
    #
    #     Returns:
    #         ToolResultEnrichmentResult with enriched result.
    #     """
    #     ...

    # Prerequisite Policies:
    #
    # def get_prerequisite_policies(self) -> List["PrerequisitePolicy"]:
    #     """Return prerequisite policies for the reliability plugin to enforce.
    #
    #     Prerequisite policies declare that certain tools (the "gated" tools)
    #     require another tool (the "prerequisite") to have been called recently
    #     before they can be used. The reliability plugin's PatternDetector
    #     generically enforces all registered policies.
    #
    #     This follows the same cross-plugin delegation pattern as
    #     evaluate_gc_policy() — the owning plugin declares what should happen,
    #     and the enforcement plugin (reliability) carries it out.
    #
    #     Returns:
    #         List of PrerequisitePolicy objects. Import the type from
    #         shared.plugins.reliability.types.
    #
    #     Example:
    #         from shared.plugins.reliability.types import (
    #             PrerequisitePolicy, PatternSeverity, NudgeType
    #         )
    #         return [PrerequisitePolicy(
    #             policy_id="template_check",
    #             prerequisite_tool="listAvailableTemplates",
    #             gated_tools={"writeNewFile", "updateFile"},
    #             nudge_templates={...},
    #         )]
    #     """
    #     ...

    # Optional method - not part of the required protocol, but recognized by
    # the permission system if implemented:
    #
    # def format_permission_request(
    #     self,
    #     tool_name: str,
    #     arguments: Dict[str, Any],
    #     channel_type: str
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
    #         channel_type: Type of channel requesting approval ("console", "webhook", "file")
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
    #
    # Interactive Channel Support:
    #
    # def supports_interactivity(self) -> bool:
    #     """Return True if this plugin has interactive features.
    #
    #     Interactive features include permission prompts, user questions,
    #     selection dialogs, progress reporting, or any other user interaction
    #     beyond tool execution.
    #
    #     Clients can use this to:
    #     - Warn users about plugins requiring interaction
    #     - Verify compatibility before loading
    #     - Choose appropriate initialization
    #
    #     Returns:
    #         True if plugin requires user interaction, False otherwise (default).
    #     """
    #     return False
    #
    # def get_supported_channels(self) -> List[str]:
    #     """Return list of channel types this plugin supports.
    #
    #     Channel types:
    #     - "console": Standard terminal input/output (stdin/stdout)
    #     - "queue": Callback-based I/O for TUI/rich clients
    #     - "webhook": HTTP-based remote interaction
    #     - "file": Filesystem-based communication
    #     - "auto": Automated responses (for testing/non-interactive mode)
    #
    #     Only relevant if supports_interactivity() returns True.
    #
    #     Returns:
    #         List of supported channel type strings. Empty list if not interactive.
    #     """
    #     return []
    #
    # def set_channel(
    #     self,
    #     channel_type: str,
    #     channel_config: Optional[Dict[str, Any]] = None
    # ) -> None:
    #     """Set the interaction channel for this plugin.
    #
    #     Called by the client to configure the plugin's interaction channel
    #     based on the client's capabilities (e.g., "queue" for TUI clients,
    #     "console" for terminal clients).
    #
    #     Args:
    #         channel_type: One of the types from get_supported_channels()
    #         channel_config: Optional channel-specific configuration
    #             For "queue" channels:
    #                 - output_callback: Callable[[str, str, str], None]
    #                   Called with (source, text, mode) for output
    #                 - input_queue: queue.Queue[str]
    #                   Queue to receive user input
    #                 - prompt_callback: Optional[Callable[[bool], None]]
    #                   Called with True when waiting for input, False when done
    #             For "console" channels:
    #                 - input_func: Callable for input (default: input)
    #                 - output_func: Callable for output (default: print)
    #             For "webhook" channels:
    #                 - endpoint: URL to send requests to
    #                 - auth_token: Bearer token for authorization
    #             For "file" channels:
    #                 - base_path: Directory for request/response files
    #
    #     Raises:
    #         ValueError: If channel_type is not supported
    #     """
    #     ...
    #
    # Auto-Wiring (Dependency Injection):
    #
    # The PluginRegistry automatically calls these methods on plugins that
    # implement them, providing dependency injection without explicit wiring
    # in client code.
    #
    # def set_plugin_registry(self, registry: 'PluginRegistry') -> None:
    #     """Receive the plugin registry for cross-plugin access.
    #
    #     Called by PluginRegistry.expose_tool() after initialization.
    #     Plugins that need to access other plugins (e.g., background plugin
    #     checking if a tool supports streaming) should implement this method.
    #
    #     Args:
    #         registry: The PluginRegistry instance managing all plugins.
    #
    #     Example:
    #         def set_plugin_registry(self, registry: 'PluginRegistry') -> None:
    #             self._registry = registry
    #
    #         def execute_tool(self, args):
    #             file_edit = self._registry.get_plugin("file_edit")
    #             if file_edit:
    #                 # Use file_edit plugin
    #     """
    #     ...
    #
    # def set_workspace_path(self, path: str) -> None:
    #     """Receive the workspace root path.
    #
    #     Called by PluginRegistry.set_workspace_path() when the workspace
    #     changes. Plugins that need workspace-relative operations (sandboxing,
    #     relative path resolution, etc.) should implement this method.
    #
    #     The PluginRegistry broadcasts to all exposed plugins implementing
    #     this method, so plugins don't need to register for updates.
    #
    #     Args:
    #         path: Absolute path to the workspace root directory.
    #
    #     Example:
    #         def set_workspace_path(self, path: str) -> None:
    #             self._workspace_path = path
    #             self._sandbox.set_allowed_root(path)
    #     """
    #     ...
    #
    # Session Persistence:
    #
    # Plugins that maintain state across turns (e.g., plans, task tracking)
    # can implement these methods to persist and restore their state when
    # sessions are saved/loaded. The session_manager calls these via hasattr()
    # checks, so they are optional.
    #
    # def get_persistence_state(self) -> Dict[str, Any]:
    #     """Return plugin state for session persistence.
    #
    #     Called by session_manager when saving a session. Return a dict
    #     containing all state that should survive session restart.
    #
    #     The returned dict must be JSON-serializable. Avoid including:
    #     - Callbacks or function references
    #     - File handles or connections
    #     - Thread-local data
    #
    #     Returns:
    #         Dict with JSON-serializable state.
    #
    #     Example:
    #         def get_persistence_state(self) -> Dict[str, Any]:
    #             return {
    #                 "agent_plan_ids": self._current_plan_ids,
    #                 "version": 1,  # For future migrations
    #             }
    #     """
    #     ...
    #
    # def restore_persistence_state(self, state: Dict[str, Any]) -> None:
    #     """Restore plugin state from session persistence.
    #
    #     Called by session_manager when loading a session. The state dict
    #     is exactly what was returned by get_persistence_state().
    #
    #     Plugins should:
    #     - Restore internal data structures
    #     - Re-register any dynamic hooks/callbacks that were lost
    #     - Handle version migrations if state format has changed
    #
    #     Args:
    #         state: State dict from get_persistence_state().
    #
    #     Example:
    #         def restore_persistence_state(self, state: Dict[str, Any]) -> None:
    #             self._current_plan_ids = state.get("agent_plan_ids", {})
    #             # Re-register callbacks that can't be serialized
    #             self._setup_hooks()
    #     """
    #     ...
