"""Event Protocol for Jaato Server.

This module defines all events for client-server communication.
Events are JSON-serializable dataclasses that flow over WebSocket.

Event Flow:
    Server -> Client: Status updates, output streaming, permission requests
    Client -> Server: Messages, permission responses, commands

Protocol Version: 1.0
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json


# =============================================================================
# Event Types
# =============================================================================

class EventType(str, Enum):
    """All event types in the protocol."""

    # Connection lifecycle
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

    # Agent lifecycle (Server -> Client)
    AGENT_CREATED = "agent.created"
    AGENT_OUTPUT = "agent.output"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_COMPLETED = "agent.completed"

    # Tool execution (Server -> Client)
    TOOL_CALL_START = "tool.call_start"
    TOOL_CALL_END = "tool.call_end"
    TOOL_OUTPUT = "tool.output"  # Live output chunk from running tool

    # Permission flow (Server <-> Client)
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_INPUT_MODE = "permission.input_mode"  # Signal client to enter permission input mode
    PERMISSION_RESOLVED = "permission.resolved"
    PERMISSION_RESPONSE = "permission.response"  # Client -> Server
    PERMISSION_STATUS = "permission.status"  # Server -> Client (status bar update)

    # Clarification flow (Server <-> Client)
    CLARIFICATION_REQUESTED = "clarification.requested"
    CLARIFICATION_INPUT_MODE = "clarification.input_mode"  # Signal client to enter clarification input mode
    CLARIFICATION_QUESTION = "clarification.question"
    CLARIFICATION_RESOLVED = "clarification.resolved"
    CLARIFICATION_RESPONSE = "clarification.response"  # Client -> Server

    # Reference selection flow (Server <-> Client)
    REFERENCE_SELECTION_REQUESTED = "reference_selection.requested"
    REFERENCE_SELECTION_RESOLVED = "reference_selection.resolved"
    REFERENCE_SELECTION_RESPONSE = "reference_selection.response"  # Client -> Server

    # Workspace mismatch flow (Server <-> Client)
    WORKSPACE_MISMATCH_REQUESTED = "workspace_mismatch.requested"
    WORKSPACE_MISMATCH_RESOLVED = "workspace_mismatch.resolved"
    WORKSPACE_MISMATCH_RESPONSE = "workspace_mismatch.response"  # Client -> Server

    # Plan updates (Server -> Client)
    PLAN_UPDATED = "plan.updated"
    PLAN_CLEARED = "plan.cleared"

    # Context/token updates (Server -> Client)
    CONTEXT_UPDATED = "context.updated"
    TURN_COMPLETED = "turn.completed"
    TURN_PROGRESS = "turn.progress"
    INSTRUCTION_BUDGET_UPDATED = "instruction_budget.updated"

    # Instruction budget (Client <-> Server)
    INSTRUCTION_BUDGET_REQUEST = "instruction_budget.request"  # Client -> Server

    # System messages (Server -> Client)
    SYSTEM_MESSAGE = "system.message"
    HELP_TEXT = "help.text"  # Detailed help output for commands
    ERROR = "error"
    INIT_PROGRESS = "init.progress"  # Initialization step progress
    RETRY = "retry"  # API retry with exponential backoff

    # Session management (Server -> Client)
    SESSION_LIST = "session.list"  # For user display (updates local cache too)
    SESSION_INFO = "session.info"  # Full state snapshot on connect/attach
    SESSION_DESCRIPTION_UPDATED = "session.description_updated"  # Description changed

    # Memory management (Server -> Client)
    MEMORY_LIST = "memory.list"  # Memory list for completion cache and pager display

    # Client requests (Client -> Server)
    SEND_MESSAGE = "message.send"
    STOP = "session.stop"
    COMMAND = "command.execute"
    COMMAND_LIST_REQUEST = "command.list_request"

    # Command list (Server -> Client)
    COMMAND_LIST = "command.list"

    # Tool status (Server -> Client)
    TOOL_STATUS = "tools.status"

    # Tool management (Client -> Server)
    TOOL_DISABLE_REQUEST = "tools.disable"

    # History (Client <-> Server)
    HISTORY_REQUEST = "history.request"
    HISTORY = "history"

    # Client configuration (Client -> Server)
    CLIENT_CONFIG = "client.config"

    # Mid-turn prompts (Server -> Client)
    MID_TURN_PROMPT_QUEUED = "mid_turn_prompt.queued"
    MID_TURN_PROMPT_INJECTED = "mid_turn_prompt.injected"
    MID_TURN_INTERRUPT = "mid_turn_prompt.interrupt"  # Streaming interrupted for user prompt

    # Session recovery (Server -> Client)
    INTERRUPTED_TURN_RECOVERED = "session.interrupted_turn_recovered"  # Turn recovered after reconnect

    # Workspace management (Client <-> Server)
    WORKSPACE_LIST_REQUEST = "workspace.list"  # Client -> Server
    WORKSPACE_LIST = "workspace.list_response"  # Server -> Client
    WORKSPACE_CREATE_REQUEST = "workspace.create"  # Client -> Server
    WORKSPACE_CREATED = "workspace.created"  # Server -> Client
    WORKSPACE_SELECT_REQUEST = "workspace.select"  # Client -> Server
    CONFIG_STATUS = "config.status"  # Server -> Client (response to workspace.select)
    CONFIG_UPDATE_REQUEST = "config.update"  # Client -> Server
    CONFIG_UPDATED = "config.updated"  # Server -> Client


# =============================================================================
# Base Event
# =============================================================================

@dataclass
class Event:
    """Base class for all events."""
    type: EventType
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert enum to string
        if isinstance(d.get('type'), EventType):
            d['type'] = d['type'].value
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


# =============================================================================
# Server -> Client Events
# =============================================================================

@dataclass
class ConnectedEvent(Event):
    """Sent when client connects successfully."""
    type: EventType = field(default=EventType.CONNECTED)
    protocol_version: str = "1.0"
    server_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCreatedEvent(Event):
    """Sent when a new agent (main or subagent) is created."""
    type: EventType = field(default=EventType.AGENT_CREATED)
    agent_id: str = ""
    agent_name: str = ""
    agent_type: str = ""  # "main" or "subagent"
    profile_name: Optional[str] = None
    parent_agent_id: Optional[str] = None
    icon_lines: Optional[List[str]] = None
    created_at: Optional[str] = None


@dataclass
class AgentOutputEvent(Event):
    """Streaming text output from an agent."""
    type: EventType = field(default=EventType.AGENT_OUTPUT)
    agent_id: str = ""
    source: str = ""  # "model", "tool", "system", plugin name
    text: str = ""
    mode: str = "write"  # "write" (new block) or "append" (continue)


@dataclass
class AgentStatusChangedEvent(Event):
    """Agent status change (active, idle, done, error)."""
    type: EventType = field(default=EventType.AGENT_STATUS_CHANGED)
    agent_id: str = ""
    status: str = ""  # "active", "idle" (waiting for input), "done", "error"
    error: Optional[str] = None


@dataclass
class AgentCompletedEvent(Event):
    """Agent has completed its task."""
    type: EventType = field(default=EventType.AGENT_COMPLETED)
    agent_id: str = ""
    completed_at: str = ""
    success: bool = True
    token_usage: Optional[Dict[str, int]] = None
    turns_used: Optional[int] = None


@dataclass
class ToolCallStartEvent(Event):
    """Tool execution has started."""
    type: EventType = field(default=EventType.TOOL_CALL_START)
    agent_id: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None


@dataclass
class ToolCallEndEvent(Event):
    """Tool execution has completed."""
    type: EventType = field(default=EventType.TOOL_CALL_END)
    agent_id: str = ""
    tool_name: str = ""
    call_id: Optional[str] = None
    success: bool = True
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    backgrounded: bool = False  # True when tool was auto-backgrounded (still producing output)


@dataclass
class ToolOutputEvent(Event):
    """Live output chunk from a running tool (tail -f style)."""
    type: EventType = field(default=EventType.TOOL_OUTPUT)
    agent_id: str = ""
    call_id: str = ""  # Required to correlate with specific tool call
    chunk: str = ""  # Output text chunk (may contain newlines)


@dataclass
class PermissionResponseOption:
    """A valid response option for permission prompts."""
    key: str  # Single char like "y", "n", "a"
    label: str  # Display label like "yes", "no", "always"
    action: str  # Action type: "allow", "deny", "whitelist", "blacklist"
    description: Optional[str] = None


@dataclass
class PermissionRequestedEvent(Event):
    """Permission is requested for a tool execution.

    Includes pre-formatted prompt lines (with diff for file edits) when available.
    """
    type: EventType = field(default=EventType.PERMISSION_REQUESTED)
    agent_id: str = ""  # Which agent is requesting permission
    request_id: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    response_options: List[Dict[str, str]] = field(default_factory=list)
    # ^ List of {key, label, action, description?}
    prompt_lines: Optional[List[str]] = None  # Pre-formatted prompt (with diff)
    format_hint: Optional[str] = None  # "diff" for colored diff display
    warnings: Optional[str] = None  # Security/analysis warnings to display separately
    warning_level: Optional[str] = None  # "info", "warning", "error"


@dataclass
class PermissionInputModeEvent(Event):
    """Signal client to enter permission input mode.

    Sent AFTER permission content has been emitted via AgentOutputEvent.
    This lightweight control event separates content delivery from input control.
    """
    type: EventType = field(default=EventType.PERMISSION_INPUT_MODE)
    agent_id: str = ""  # Which agent is requesting permission
    request_id: str = ""
    tool_name: str = ""
    call_id: Optional[str] = None  # Unique ID for matching tool call (parallel execution)
    response_options: List[Dict[str, str]] = field(default_factory=list)
    # ^ List of {key, label, action, description?}
    # Tool arguments for client-side editing (when edit option is available)
    tool_args: Optional[Dict[str, Any]] = None
    # Editable content metadata: {parameters: [...], format: "yaml"|"json"|"text"}
    editable_metadata: Optional[Dict[str, Any]] = None


@dataclass
class PermissionResolvedEvent(Event):
    """Permission has been resolved (granted or denied)."""
    type: EventType = field(default=EventType.PERMISSION_RESOLVED)
    agent_id: str = ""  # Which agent's permission was resolved
    request_id: str = ""
    tool_name: str = ""
    granted: bool = False
    method: str = ""  # "user", "whitelist", "blacklist", "default"


@dataclass
class PermissionStatusEvent(Event):
    """Permission status update for client toolbar display.

    Emitted after permission commands (default/suspend/resume) and
    permission resolutions that change the effective policy.
    """
    type: EventType = field(default=EventType.PERMISSION_STATUS)
    effective_default: str = "ask"  # "allow", "deny", or "ask"
    suspension_scope: Optional[str] = None  # "turn", "idle", "session", or None


@dataclass
class ClarificationRequestedEvent(Event):
    """Clarification session has started."""
    type: EventType = field(default=EventType.CLARIFICATION_REQUESTED)
    agent_id: str = ""  # Which agent is requesting clarification
    request_id: str = ""
    tool_name: str = ""
    context_lines: List[str] = field(default_factory=list)
    total_questions: int = 0


@dataclass
class ClarificationQuestionEvent(Event):
    """A single clarification question to answer."""
    type: EventType = field(default=EventType.CLARIFICATION_QUESTION)
    agent_id: str = ""  # Which agent is asking the question
    request_id: str = ""
    question_index: int = 0
    total_questions: int = 0
    question_type: str = ""  # "single_choice", "multiple_choice", "free_text"
    question_text: str = ""
    options: Optional[List[Dict[str, str]]] = None  # For choice questions


@dataclass
class ClarificationInputModeEvent(Event):
    """Signal client to enter clarification input mode.

    Sent AFTER clarification content has been emitted via AgentOutputEvent.
    This lightweight control event separates content delivery from input control.
    """
    type: EventType = field(default=EventType.CLARIFICATION_INPUT_MODE)
    agent_id: str = ""  # Which agent is requesting clarification
    request_id: str = ""
    tool_name: str = ""
    question_index: int = 0
    total_questions: int = 0


@dataclass
class ClarificationResolvedEvent(Event):
    """All clarification questions have been answered."""
    type: EventType = field(default=EventType.CLARIFICATION_RESOLVED)
    agent_id: str = ""  # Which agent's clarification was resolved
    request_id: str = ""
    tool_name: str = ""
    qa_pairs: List[List[str]] = field(default_factory=list)
    # ^ List of [question_text, answer_text] pairs for overview display


@dataclass
class ReferenceSelectionRequestedEvent(Event):
    """Reference selection has been requested.

    Sent when the model calls selectReferences and the user needs to choose
    which references to include.
    """
    type: EventType = field(default=EventType.REFERENCE_SELECTION_REQUESTED)
    agent_id: str = ""  # Which agent is requesting reference selection
    request_id: str = ""
    tool_name: str = ""
    prompt_lines: List[str] = field(default_factory=list)


@dataclass
class ReferenceSelectionResolvedEvent(Event):
    """Reference selection has been completed."""
    type: EventType = field(default=EventType.REFERENCE_SELECTION_RESOLVED)
    agent_id: str = ""  # Which agent's reference selection was resolved
    request_id: str = ""
    tool_name: str = ""
    selected_ids: List[str] = field(default_factory=list)


@dataclass
class WorkspaceMismatchResponseOption:
    """A valid response option for workspace mismatch prompts."""
    key: str  # Single char like "s", "n"
    label: str  # Display label like "switch", "new session"
    action: str  # Action type: "switch", "new_session", "cancel"
    description: Optional[str] = None


@dataclass
class WorkspaceMismatchRequestedEvent(Event):
    """Workspace mismatch detected when attaching to a session.

    Sent when a client tries to attach to a session that was created
    with a different workspace path. The client must choose to either
    switch to the session's workspace or create a new session.
    """
    type: EventType = field(default=EventType.WORKSPACE_MISMATCH_REQUESTED)
    request_id: str = ""
    session_id: str = ""
    session_workspace: str = ""  # The session's current workspace
    client_workspace: str = ""   # The client's workspace
    response_options: List[Dict[str, str]] = field(default_factory=list)
    # ^ List of {key, label, action, description?}
    prompt_lines: List[str] = field(default_factory=list)


@dataclass
class WorkspaceMismatchResolvedEvent(Event):
    """Workspace mismatch has been resolved."""
    type: EventType = field(default=EventType.WORKSPACE_MISMATCH_RESOLVED)
    request_id: str = ""
    session_id: str = ""
    action: str = ""  # "switch", "new_session", "cancel"
    new_session_id: Optional[str] = None  # Set if action is "new_session"


@dataclass
class PlanStepData:
    """A single step in a plan."""
    content: str
    status: str  # "pending", "in_progress", "completed"
    active_form: Optional[str] = None


@dataclass
class PlanUpdatedEvent(Event):
    """Plan has been created or updated."""
    type: EventType = field(default=EventType.PLAN_UPDATED)
    agent_id: str = ""
    plan_name: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    # ^ List of {content, status, active_form?, blocked_by?, depends_on?, received_outputs?}


@dataclass
class PlanClearedEvent(Event):
    """Plan has been cleared/completed."""
    type: EventType = field(default=EventType.PLAN_CLEARED)
    agent_id: str = ""


@dataclass
class ContextUpdatedEvent(Event):
    """Context window usage has changed."""
    type: EventType = field(default=EventType.CONTEXT_UPDATED)
    agent_id: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    context_limit: int = 0
    percent_used: float = 0.0
    tokens_remaining: int = 0
    turns: int = 0
    # GC configuration (included for status bar display)
    gc_threshold: Optional[float] = None  # GC trigger threshold percentage
    gc_strategy: Optional[str] = None  # GC strategy name (e.g., "truncate", "hybrid")
    gc_target_percent: Optional[float] = None  # Target usage after GC
    gc_continuous_mode: bool = False  # True if GC runs after every turn


@dataclass
class InstructionBudgetEvent(Event):
    """Instruction budget has been updated.

    Provides detailed breakdown of token usage by instruction source layer.
    Sent after session configuration and when budget changes significantly.

    The budget_snapshot contains:
    - session_id, agent_id, agent_type: Identity
    - context_limit, total_tokens, utilization_percent: Overall usage
    - gc_eligible_tokens, locked_tokens, preservable_tokens: GC info
    - entries: Per-source breakdown (system, session, plugin, enrichment, conversation)
    """
    type: EventType = field(default=EventType.INSTRUCTION_BUDGET_UPDATED)
    agent_id: str = ""
    budget_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnCompletedEvent(Event):
    """A conversation turn has completed."""
    type: EventType = field(default=EventType.TURN_COMPLETED)
    agent_id: str = ""
    turn_number: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Formatted output text (with syntax highlighting, validation, etc.)
    # Client can use this to replace raw streaming output with formatted version
    formatted_text: Optional[str] = None


@dataclass
class TurnProgressEvent(Event):
    """Incremental progress during turn execution.

    Emitted after each model response within a turn, enabling real-time
    token tracking before the turn completes.
    """
    type: EventType = field(default=EventType.TURN_PROGRESS)
    agent_id: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    context_limit: int = 0
    percent_used: float = 0.0
    tokens_remaining: int = 0
    pending_tool_calls: int = 0  # How many tool calls remain


@dataclass
class SystemMessageEvent(Event):
    """System message (info, warning, status)."""
    type: EventType = field(default=EventType.SYSTEM_MESSAGE)
    message: str = ""
    style: str = ""  # "info", "warning", "error", "success", "dim"


@dataclass
class HelpTextEvent(Event):
    """Detailed help text for commands.

    Sent in response to 'help' subcommands to display formatted help
    using the pager. Each line is a (text, style) tuple.
    """
    type: EventType = field(default=EventType.HELP_TEXT)
    lines: List[tuple] = field(default_factory=list)  # List of (text, style) tuples


@dataclass
class InitProgressEvent(Event):
    """Initialization progress update.

    Sent during session initialization to show progress on each step.
    Steps are shown in sequence with their status.
    """
    type: EventType = field(default=EventType.INIT_PROGRESS)
    step: str = ""  # Step name (e.g., "Loading plugins")
    status: str = "running"  # "running", "done", "error"
    message: str = ""  # Optional details (e.g., error message)
    step_number: int = 0  # Current step (1-based)
    total_steps: int = 0  # Total number of steps


@dataclass
class ErrorEvent(Event):
    """Error occurred."""
    type: EventType = field(default=EventType.ERROR)
    error: str = ""
    error_type: str = ""  # Exception class name
    recoverable: bool = True


@dataclass
class RetryEvent(Event):
    """API retry notification with exponential backoff.

    Sent when a transient error (rate limit, server error) is encountered
    and the system is retrying the request.
    """
    type: EventType = field(default=EventType.RETRY)
    message: str = ""  # Human-readable retry message
    attempt: int = 0  # Current attempt number (1-indexed)
    max_attempts: int = 0  # Maximum attempts configured
    delay: float = 0.0  # Delay in seconds before next attempt
    error_type: str = ""  # Type of error (rate_limit, transient)


@dataclass
class SessionListEvent(Event):
    """List of available sessions - for user display."""
    type: EventType = field(default=EventType.SESSION_LIST)
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    # ^ List of {id: str, name: str, created_at: str, last_active: str, ...}


@dataclass
class MemoryListEvent(Event):
    """List of available memories - for completion cache and pager display."""
    type: EventType = field(default=EventType.MEMORY_LIST)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    # ^ List of {id: str, description: str, tags: List[str]}


@dataclass
class SessionInfoEvent(Event):
    """Session state snapshot - sent on connect/attach with all data client needs.

    Includes current session info plus lists for completion/display:
    - sessions: All available sessions (for session commands)
    - tools: All available tools with enabled status (for tools commands)
    - models: Available model names (for model command)

    Client stores this locally and uses it for both completion and display.
    Server pushes updates when state changes.
    """
    type: EventType = field(default=EventType.SESSION_INFO)
    # Current session
    session_id: str = ""
    session_name: str = ""
    model_provider: str = ""
    model_name: str = ""
    # State snapshot for local use
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    # ^ [{id, name, model_provider, model_name, is_loaded, client_count, turn_count}, ...]
    tools: List[Dict[str, Any]] = field(default_factory=list)
    # ^ [{name, description, enabled, plugin}, ...]
    models: List[str] = field(default_factory=list)
    # ^ ["gemini-2.5-flash", "gemini-2.5-pro", ...]
    user_inputs: List[str] = field(default_factory=list)
    # ^ Command history for prompt restoration on reconnect
    memories: List[Dict[str, Any]] = field(default_factory=list)
    # ^ [{id, description, tags}, ...] for memory command completions


@dataclass
class SessionDescriptionUpdatedEvent(Event):
    """Session description was updated (by model calling session_describe)."""
    type: EventType = field(default=EventType.SESSION_DESCRIPTION_UPDATED)
    session_id: str = ""
    description: str = ""


# =============================================================================
# Workspace Management Events (Server -> Client)
# =============================================================================

@dataclass
class WorkspaceInfo:
    """Information about a single workspace."""
    name: str  # Relative path from workspace root (e.g., "project-a")
    configured: bool  # Has valid .env with provider
    provider: Optional[str] = None  # Provider if configured
    model: Optional[str] = None  # Model if configured
    last_accessed: Optional[str] = None  # ISO timestamp


@dataclass
class WorkspaceListEvent(Event):
    """Response to workspace.list - list of available workspaces."""
    type: EventType = field(default=EventType.WORKSPACE_LIST)
    root: str = ""  # Absolute path to workspace root
    workspaces: List[Dict[str, Any]] = field(default_factory=list)
    # ^ List of WorkspaceInfo as dicts


@dataclass
class WorkspaceCreatedEvent(Event):
    """Response to workspace.create - new workspace created."""
    type: EventType = field(default=EventType.WORKSPACE_CREATED)
    name: str = ""  # Relative path from workspace root
    path: str = ""  # Absolute path


@dataclass
class ConfigStatusEvent(Event):
    """Response to workspace.select - configuration status of selected workspace."""
    type: EventType = field(default=EventType.CONFIG_STATUS)
    workspace: str = ""  # Workspace name (relative path)
    configured: bool = False  # Has valid provider config
    provider: Optional[str] = None  # Current provider if set
    model: Optional[str] = None  # Current model if set
    available_providers: List[str] = field(default_factory=list)  # Providers that can be configured
    missing_fields: List[str] = field(default_factory=list)  # What's needed to complete config


@dataclass
class ConfigUpdatedEvent(Event):
    """Response to config.update - configuration was updated."""
    type: EventType = field(default=EventType.CONFIG_UPDATED)
    workspace: str = ""  # Workspace name
    provider: str = ""  # New provider
    model: Optional[str] = None  # New model if set
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Client -> Server Events (Requests)
# =============================================================================

@dataclass
class SendMessageRequest(Event):
    """Send a message to the model."""
    type: EventType = field(default=EventType.SEND_MESSAGE)
    text: str = ""
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    # ^ List of {type: "file", path: "..."} or {type: "image", data: "base64..."}


@dataclass
class PermissionResponseRequest(Event):
    """Respond to a permission request."""
    type: EventType = field(default=EventType.PERMISSION_RESPONSE)
    request_id: str = ""
    response: str = ""  # "y", "n", "a", "never", etc.
    # Edited tool arguments (set when response is "e" and client handled editing)
    edited_arguments: Optional[Dict[str, Any]] = None


@dataclass
class ClarificationResponseRequest(Event):
    """Respond to a clarification question."""
    type: EventType = field(default=EventType.CLARIFICATION_RESPONSE)
    request_id: str = ""
    question_index: int = 0
    response: str = ""  # User's answer


@dataclass
class ReferenceSelectionResponseRequest(Event):
    """Respond to a reference selection request."""
    type: EventType = field(default=EventType.REFERENCE_SELECTION_RESPONSE)
    request_id: str = ""
    response: str = ""  # User's selection (e.g., "1,3,4", "all", "none")


@dataclass
class WorkspaceMismatchResponseRequest(Event):
    """Respond to a workspace mismatch request."""
    type: EventType = field(default=EventType.WORKSPACE_MISMATCH_RESPONSE)
    request_id: str = ""
    response: str = ""  # "s" (switch), "n" (new session), "c" (cancel)


@dataclass
class StopRequest(Event):
    """Stop current operation (cancel generation)."""
    type: EventType = field(default=EventType.STOP)
    agent_id: Optional[str] = None  # None = all agents


@dataclass
class CommandRequest(Event):
    """Execute a command (like 'model', 'save', 'resume', etc.)."""
    type: EventType = field(default=EventType.COMMAND)
    command: str = ""
    args: List[str] = field(default_factory=list)


@dataclass
class GetInstructionBudgetRequest(Event):
    """Request current instruction budget for an agent.

    Server responds with InstructionBudgetEvent containing the budget snapshot.
    If agent_id is None or empty, returns budget for main agent.
    """
    type: EventType = field(default=EventType.INSTRUCTION_BUDGET_REQUEST)
    agent_id: Optional[str] = None  # None = main agent


@dataclass
class CommandListRequest(Event):
    """Request list of available commands from server."""
    type: EventType = field(default=EventType.COMMAND_LIST_REQUEST)


@dataclass
class CommandListEvent(Event):
    """List of available commands from server/plugins."""
    type: EventType = field(default=EventType.COMMAND_LIST)
    commands: List[Dict[str, str]] = field(default_factory=list)
    # ^ List of {name, description, ?subcommands}


@dataclass
class ToolStatusEvent(Event):
    """Tool status information for client display."""
    type: EventType = field(default=EventType.TOOL_STATUS)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    # ^ List of {name, description, enabled, plugin}
    message: str = ""  # Optional result message (for enable/disable operations)


@dataclass
class ToolDisableRequest(Event):
    """Client request to disable a tool.

    Directly calls registry.disable_tool() without generating response events.
    Used by headless mode to disable tools before starting event handling.
    """
    type: EventType = field(default=EventType.TOOL_DISABLE_REQUEST)
    tool_name: str = ""  # Tool to disable


@dataclass
class HistoryRequest(Event):
    """Client request for conversation history."""
    type: EventType = field(default=EventType.HISTORY_REQUEST)
    agent_id: str = "main"  # Which agent's history to get


@dataclass
class HistoryEvent(Event):
    """Conversation history from server."""
    type: EventType = field(default=EventType.HISTORY)
    agent_id: str = "main"
    history: List[Dict[str, Any]] = field(default_factory=list)
    # ^ List of serialized Message objects
    turn_accounting: List[Dict[str, int]] = field(default_factory=list)
    # ^ List of {prompt, output, total} per turn


# =============================================================================
# Workspace Management Requests (Client -> Server)
# =============================================================================

@dataclass
class WorkspaceListRequest(Event):
    """Client requests list of available workspaces."""
    type: EventType = field(default=EventType.WORKSPACE_LIST_REQUEST)


@dataclass
class WorkspaceCreateRequest(Event):
    """Client requests creation of a new workspace."""
    type: EventType = field(default=EventType.WORKSPACE_CREATE_REQUEST)
    name: str = ""  # Name for the new workspace (becomes subdirectory name)


@dataclass
class WorkspaceSelectRequest(Event):
    """Client selects a workspace to use for the session."""
    type: EventType = field(default=EventType.WORKSPACE_SELECT_REQUEST)
    name: str = ""  # Workspace name (relative path from root)


@dataclass
class ConfigUpdateRequest(Event):
    """Client updates workspace configuration (provider, model, API key)."""
    type: EventType = field(default=EventType.CONFIG_UPDATE_REQUEST)
    provider: str = ""  # Provider name (anthropic, google, github, etc.)
    model: Optional[str] = None  # Model name (optional, uses provider default)
    api_key: Optional[str] = None  # API key (optional, for non-OAuth providers)


@dataclass
class ClientConfigRequest(Event):
    """Client sends its configuration to the server.

    Sent after connection to apply client-specific settings like trace paths.
    The server applies these settings to the session/plugins.
    """
    type: EventType = field(default=EventType.CLIENT_CONFIG)
    # Environment overrides from client's .env
    trace_log_path: Optional[str] = None  # JAATO_TRACE_LOG
    provider_trace_log: Optional[str] = None  # PROVIDER_TRACE_LOG
    # Terminal width for formatting (enrichment notifications)
    terminal_width: Optional[int] = None
    # Client's working directory (for finding config files like .lsp.json)
    working_dir: Optional[str] = None
    # Path to client's .env file - server loads this for session creation
    # This provides all provider-related env vars (PROJECT_ID, JAATO_PROVIDER, etc.)
    env_file: Optional[str] = None


# =============================================================================
# Mid-Turn Prompt Events
# =============================================================================

@dataclass
class MidTurnPromptQueuedEvent(Event):
    """Sent when a user prompt is queued during model processing.

    Instead of returning an error when the user sends a message while the model
    is running, the message is queued and will be injected at the next natural
    pause point (between tool executions, after subagent completion, etc.).
    """
    type: EventType = field(default=EventType.MID_TURN_PROMPT_QUEUED)
    text: str = ""
    position_in_queue: int = 0  # 0-based position (usually 0, can be >0 if multiple queued)


@dataclass
class MidTurnPromptInjectedEvent(Event):
    """Sent when a queued prompt is injected into the conversation.

    This notifies the client that the queued prompt is now being processed
    by the model.
    """
    type: EventType = field(default=EventType.MID_TURN_PROMPT_INJECTED)
    text: str = ""


@dataclass
class MidTurnInterruptEvent(Event):
    """Sent when streaming is interrupted to process a mid-turn user prompt.

    This notifies the client that the model's current generation was interrupted
    because a user prompt arrived and needs to be processed immediately.
    The partial response is preserved and the user's prompt is being processed.
    """
    type: EventType = field(default=EventType.MID_TURN_INTERRUPT)
    partial_response_chars: int = 0  # How much of the response was generated before interrupt
    user_prompt_preview: str = ""  # First 100 chars of the user's prompt


@dataclass
class InterruptedTurnRecoveredEvent(Event):
    """Sent when the server recovers from an interrupted turn after reconnection.

    This event notifies the client that a turn was interrupted (e.g., by server
    restart) and has been recovered with synthetic error responses injected
    for any pending tool calls.
    """
    type: EventType = field(default=EventType.INTERRUPTED_TURN_RECOVERED)
    session_id: str = ""
    agent_id: str = ""
    recovered_calls: int = 0  # Number of tool calls that were recovered
    action_taken: str = ""  # What action was taken (e.g., "synthetic_error")


# =============================================================================
# Serialization Helpers
# =============================================================================

# Map of event type -> event class
_EVENT_CLASSES: Dict[str, type] = {
    EventType.CONNECTED.value: ConnectedEvent,
    EventType.AGENT_CREATED.value: AgentCreatedEvent,
    EventType.AGENT_OUTPUT.value: AgentOutputEvent,
    EventType.AGENT_STATUS_CHANGED.value: AgentStatusChangedEvent,
    EventType.AGENT_COMPLETED.value: AgentCompletedEvent,
    EventType.TOOL_CALL_START.value: ToolCallStartEvent,
    EventType.TOOL_CALL_END.value: ToolCallEndEvent,
    EventType.TOOL_OUTPUT.value: ToolOutputEvent,
    EventType.PERMISSION_REQUESTED.value: PermissionRequestedEvent,
    EventType.PERMISSION_INPUT_MODE.value: PermissionInputModeEvent,
    EventType.PERMISSION_RESOLVED.value: PermissionResolvedEvent,
    EventType.PERMISSION_STATUS.value: PermissionStatusEvent,
    EventType.CLARIFICATION_REQUESTED.value: ClarificationRequestedEvent,
    EventType.CLARIFICATION_INPUT_MODE.value: ClarificationInputModeEvent,
    EventType.CLARIFICATION_QUESTION.value: ClarificationQuestionEvent,
    EventType.CLARIFICATION_RESOLVED.value: ClarificationResolvedEvent,
    EventType.REFERENCE_SELECTION_REQUESTED.value: ReferenceSelectionRequestedEvent,
    EventType.REFERENCE_SELECTION_RESOLVED.value: ReferenceSelectionResolvedEvent,
    EventType.REFERENCE_SELECTION_RESPONSE.value: ReferenceSelectionResponseRequest,
    EventType.WORKSPACE_MISMATCH_REQUESTED.value: WorkspaceMismatchRequestedEvent,
    EventType.WORKSPACE_MISMATCH_RESOLVED.value: WorkspaceMismatchResolvedEvent,
    EventType.WORKSPACE_MISMATCH_RESPONSE.value: WorkspaceMismatchResponseRequest,
    EventType.PLAN_UPDATED.value: PlanUpdatedEvent,
    EventType.PLAN_CLEARED.value: PlanClearedEvent,
    EventType.CONTEXT_UPDATED.value: ContextUpdatedEvent,
    EventType.INSTRUCTION_BUDGET_UPDATED.value: InstructionBudgetEvent,
    EventType.TURN_COMPLETED.value: TurnCompletedEvent,
    EventType.TURN_PROGRESS.value: TurnProgressEvent,
    EventType.SYSTEM_MESSAGE.value: SystemMessageEvent,
    EventType.HELP_TEXT.value: HelpTextEvent,
    EventType.INIT_PROGRESS.value: InitProgressEvent,
    EventType.ERROR.value: ErrorEvent,
    EventType.RETRY.value: RetryEvent,
    EventType.SESSION_LIST.value: SessionListEvent,
    EventType.SESSION_INFO.value: SessionInfoEvent,
    EventType.MEMORY_LIST.value: MemoryListEvent,
    EventType.SESSION_DESCRIPTION_UPDATED.value: SessionDescriptionUpdatedEvent,
    EventType.SEND_MESSAGE.value: SendMessageRequest,
    EventType.PERMISSION_RESPONSE.value: PermissionResponseRequest,
    EventType.CLARIFICATION_RESPONSE.value: ClarificationResponseRequest,
    EventType.STOP.value: StopRequest,
    EventType.COMMAND.value: CommandRequest,
    EventType.INSTRUCTION_BUDGET_REQUEST.value: GetInstructionBudgetRequest,
    EventType.COMMAND_LIST_REQUEST.value: CommandListRequest,
    EventType.COMMAND_LIST.value: CommandListEvent,
    EventType.TOOL_STATUS.value: ToolStatusEvent,
    EventType.TOOL_DISABLE_REQUEST.value: ToolDisableRequest,
    EventType.HISTORY_REQUEST.value: HistoryRequest,
    EventType.HISTORY.value: HistoryEvent,
    EventType.CLIENT_CONFIG.value: ClientConfigRequest,
    EventType.MID_TURN_PROMPT_QUEUED.value: MidTurnPromptQueuedEvent,
    EventType.MID_TURN_PROMPT_INJECTED.value: MidTurnPromptInjectedEvent,
    EventType.MID_TURN_INTERRUPT.value: MidTurnInterruptEvent,
    EventType.INTERRUPTED_TURN_RECOVERED.value: InterruptedTurnRecoveredEvent,
    # Workspace management
    EventType.WORKSPACE_LIST_REQUEST.value: WorkspaceListRequest,
    EventType.WORKSPACE_LIST.value: WorkspaceListEvent,
    EventType.WORKSPACE_CREATE_REQUEST.value: WorkspaceCreateRequest,
    EventType.WORKSPACE_CREATED.value: WorkspaceCreatedEvent,
    EventType.WORKSPACE_SELECT_REQUEST.value: WorkspaceSelectRequest,
    EventType.CONFIG_STATUS.value: ConfigStatusEvent,
    EventType.CONFIG_UPDATE_REQUEST.value: ConfigUpdateRequest,
    EventType.CONFIG_UPDATED.value: ConfigUpdatedEvent,
}


def serialize_event(event: Event) -> str:
    """Serialize an event to JSON string."""
    return event.to_json()


def deserialize_event(json_str: str) -> Event:
    """Deserialize a JSON string to an event object.

    Args:
        json_str: JSON string representing an event.

    Returns:
        The deserialized event object.

    Raises:
        ValueError: If the event type is unknown.
        json.JSONDecodeError: If the JSON is invalid.
    """
    data = json.loads(json_str)
    event_type = data.get("type")

    if event_type not in _EVENT_CLASSES:
        raise ValueError(f"Unknown event type: {event_type}")

    event_class = _EVENT_CLASSES[event_type]

    # Convert type string back to enum
    data["type"] = EventType(event_type)

    # Remove unknown fields (forward compatibility)
    known_fields = {f.name for f in event_class.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in known_fields}

    return event_class(**filtered_data)


def create_event(event_type: EventType, **kwargs) -> Event:
    """Factory function to create an event by type.

    Args:
        event_type: The type of event to create.
        **kwargs: Event-specific fields.

    Returns:
        The created event object.
    """
    event_class = _EVENT_CLASSES.get(event_type.value)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class(**kwargs)
