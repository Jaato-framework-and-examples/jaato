"""Client-side types inlined from shared.plugins.base.

These types are duplicated here so the TUI can be distributed independently
of jaato-server. The canonical definitions remain in shared/plugins/base.py
for the server; the TUI uses this local copy.

Types:
- HelpLines: Styled help text for pager display.
- CommandParameter: Definition of a command parameter for argument parsing.
- UserCommand: Declaration of a user-facing command.
- parse_command_args(): Generic argument parser for user commands.
- AgentUIHooks: Protocol for UI integration with agent lifecycle.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Protocol
from datetime import datetime


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


class CommandParameter:
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
    __slots__ = ('name', 'description', 'required', 'capture_rest')

    def __init__(
        self,
        name: str,
        description: str = "",
        required: bool = False,
        capture_rest: bool = False,
    ):
        self.name = name
        self.description = description
        self.required = required
        self.capture_rest = capture_rest


class UserCommand:
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
    __slots__ = ('name', 'description', 'share_with_model', 'parameters')

    def __init__(
        self,
        name: str,
        description: str,
        share_with_model: bool = False,
        parameters: Optional[List[CommandParameter]] = None,
    ):
        self.name = name
        self.description = description
        self.share_with_model = share_with_model
        self.parameters = parameters


def parse_command_args(
    command: UserCommand,
    raw_args: str,
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


class AgentUIHooks(Protocol):
    """Protocol for UI integration with agent lifecycle.

    These hooks allow the UI to track agent creation, execution, and completion
    for visualization purposes (e.g., agent panel in rich client).

    Hooks are called from both main agent and subagent execution paths.
    All hooks are optional - if not implemented, agents run normally without
    UI integration.

    Thread Safety:
        All hooks may be called from background threads (especially for subagents).
        Implementations must be thread-safe.
    """

    def on_agent_created(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        profile_name: Optional[str],
        parent_agent_id: Optional[str],
        icon_lines: Optional[List[str]],
        created_at: datetime,
    ) -> None: ...

    def on_agent_output(
        self,
        agent_id: str,
        source: str,
        text: str,
        mode: str,
    ) -> None: ...

    def on_agent_status_changed(
        self,
        agent_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None: ...

    def on_agent_completed(
        self,
        agent_id: str,
        completed_at: datetime,
        success: bool,
        token_usage: Optional[Dict[str, int]] = None,
        turns_used: Optional[int] = None,
    ) -> None: ...

    def on_agent_turn_completed(
        self,
        agent_id: str,
        turn_number: int,
        prompt_tokens: int,
        output_tokens: int,
        total_tokens: int,
        duration_seconds: float,
        function_calls: List[Dict[str, Any]],
    ) -> None: ...

    def on_agent_context_updated(
        self,
        agent_id: str,
        total_tokens: int,
        prompt_tokens: int,
        output_tokens: int,
        turns: int,
        percent_used: float,
    ) -> None: ...

    def on_turn_progress(
        self,
        agent_id: str,
        total_tokens: int,
        prompt_tokens: int,
        output_tokens: int,
        percent_used: float,
        pending_tool_calls: int,
    ) -> None: ...

    def on_agent_gc_config(
        self,
        agent_id: str,
        threshold: float,
        strategy: str,
        target_percent: Optional[float] = None,
        continuous_mode: bool = False,
    ) -> None: ...

    def on_agent_history_updated(
        self,
        agent_id: str,
        history: List[Any],
    ) -> None: ...

    def on_tool_call_start(
        self,
        agent_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> None: ...

    def on_tool_call_end(
        self,
        agent_id: str,
        tool_name: str,
        success: bool,
        duration_seconds: float,
        error_message: Optional[str] = None,
        call_id: Optional[str] = None,
        backgrounded: bool = False,
        continuation_id: Optional[str] = None,
        show_output: Optional[bool] = None,
        show_popup: Optional[bool] = None,
    ) -> None: ...

    def on_tool_output(
        self,
        agent_id: str,
        call_id: str,
        chunk: str,
    ) -> None: ...

    def on_agent_instruction_budget_updated(
        self,
        agent_id: str,
        budget_snapshot: Dict[str, Any],
    ) -> None: ...
