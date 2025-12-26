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

    # Permission flow (Server <-> Client)
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_RESOLVED = "permission.resolved"
    PERMISSION_RESPONSE = "permission.response"  # Client -> Server

    # Clarification flow (Server <-> Client)
    CLARIFICATION_REQUESTED = "clarification.requested"
    CLARIFICATION_QUESTION = "clarification.question"
    CLARIFICATION_RESOLVED = "clarification.resolved"
    CLARIFICATION_RESPONSE = "clarification.response"  # Client -> Server

    # Plan updates (Server -> Client)
    PLAN_UPDATED = "plan.updated"
    PLAN_CLEARED = "plan.cleared"

    # Context/token updates (Server -> Client)
    CONTEXT_UPDATED = "context.updated"
    TURN_COMPLETED = "turn.completed"

    # System messages (Server -> Client)
    SYSTEM_MESSAGE = "system.message"
    ERROR = "error"

    # Session management (Server -> Client)
    SESSION_LIST = "session.list"
    SESSION_INFO = "session.info"

    # Client requests (Client -> Server)
    SEND_MESSAGE = "message.send"
    STOP = "session.stop"
    COMMAND = "command.execute"


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
    """Agent status change (active, done, error)."""
    type: EventType = field(default=EventType.AGENT_STATUS_CHANGED)
    agent_id: str = ""
    status: str = ""  # "active", "done", "error"
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


@dataclass
class PermissionResponseOption:
    """A valid response option for permission prompts."""
    key: str  # Single char like "y", "n", "a"
    label: str  # Display label like "yes", "no", "always"
    action: str  # Action type: "allow", "deny", "whitelist", "blacklist"
    description: Optional[str] = None


@dataclass
class PermissionRequestedEvent(Event):
    """Permission is requested for a tool execution."""
    type: EventType = field(default=EventType.PERMISSION_REQUESTED)
    request_id: str = ""
    tool_name: str = ""
    prompt_lines: List[str] = field(default_factory=list)
    response_options: List[Dict[str, str]] = field(default_factory=list)
    # ^ List of {key, label, action, description?}


@dataclass
class PermissionResolvedEvent(Event):
    """Permission has been resolved (granted or denied)."""
    type: EventType = field(default=EventType.PERMISSION_RESOLVED)
    request_id: str = ""
    tool_name: str = ""
    granted: bool = False
    method: str = ""  # "user", "whitelist", "blacklist", "default"


@dataclass
class ClarificationRequestedEvent(Event):
    """Clarification session has started."""
    type: EventType = field(default=EventType.CLARIFICATION_REQUESTED)
    request_id: str = ""
    tool_name: str = ""
    context_lines: List[str] = field(default_factory=list)
    total_questions: int = 0


@dataclass
class ClarificationQuestionEvent(Event):
    """A single clarification question to answer."""
    type: EventType = field(default=EventType.CLARIFICATION_QUESTION)
    request_id: str = ""
    question_index: int = 0
    total_questions: int = 0
    question_type: str = ""  # "single_choice", "multiple_choice", "free_text"
    question_text: str = ""
    options: Optional[List[Dict[str, str]]] = None  # For choice questions


@dataclass
class ClarificationResolvedEvent(Event):
    """All clarification questions have been answered."""
    type: EventType = field(default=EventType.CLARIFICATION_RESOLVED)
    request_id: str = ""
    tool_name: str = ""


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
    # ^ List of {content, status, active_form?}


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


@dataclass
class SystemMessageEvent(Event):
    """System message (info, warning, status)."""
    type: EventType = field(default=EventType.SYSTEM_MESSAGE)
    message: str = ""
    style: str = ""  # "info", "warning", "error", "success", "dim"


@dataclass
class ErrorEvent(Event):
    """Error occurred."""
    type: EventType = field(default=EventType.ERROR)
    error: str = ""
    error_type: str = ""  # Exception class name
    recoverable: bool = True


@dataclass
class SessionListEvent(Event):
    """List of available sessions."""
    type: EventType = field(default=EventType.SESSION_LIST)
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    # ^ List of {id: str, name: str, created_at: str, last_active: str, ...}


@dataclass
class SessionInfoEvent(Event):
    """Session information including model details."""
    type: EventType = field(default=EventType.SESSION_INFO)
    session_id: str = ""
    session_name: str = ""
    model_provider: str = ""
    model_name: str = ""


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


@dataclass
class ClarificationResponseRequest(Event):
    """Respond to a clarification question."""
    type: EventType = field(default=EventType.CLARIFICATION_RESPONSE)
    request_id: str = ""
    question_index: int = 0
    response: str = ""  # User's answer


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
    EventType.PERMISSION_REQUESTED.value: PermissionRequestedEvent,
    EventType.PERMISSION_RESOLVED.value: PermissionResolvedEvent,
    EventType.CLARIFICATION_REQUESTED.value: ClarificationRequestedEvent,
    EventType.CLARIFICATION_QUESTION.value: ClarificationQuestionEvent,
    EventType.CLARIFICATION_RESOLVED.value: ClarificationResolvedEvent,
    EventType.PLAN_UPDATED.value: PlanUpdatedEvent,
    EventType.PLAN_CLEARED.value: PlanClearedEvent,
    EventType.CONTEXT_UPDATED.value: ContextUpdatedEvent,
    EventType.TURN_COMPLETED.value: TurnCompletedEvent,
    EventType.SYSTEM_MESSAGE.value: SystemMessageEvent,
    EventType.ERROR.value: ErrorEvent,
    EventType.SESSION_LIST.value: SessionListEvent,
    EventType.SESSION_INFO.value: SessionInfoEvent,
    EventType.SEND_MESSAGE.value: SendMessageRequest,
    EventType.PERMISSION_RESPONSE.value: PermissionResponseRequest,
    EventType.CLARIFICATION_RESPONSE.value: ClarificationResponseRequest,
    EventType.STOP.value: StopRequest,
    EventType.COMMAND.value: CommandRequest,
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
