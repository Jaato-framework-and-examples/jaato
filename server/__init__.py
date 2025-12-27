"""Jaato Server - Multi-client AI assistant backend.

This package provides:
- JaatoServer: Core logic for AI interaction
- SessionManager: Multi-session support
- Event protocol: Typed events for client-server communication
- IPC server: Unix domain socket for local clients
- WebSocket server: Real-time communication for remote clients

Usage:
    # Start with IPC socket (local clients)
    python -m server --ipc-socket /tmp/jaato.sock

    # Start with WebSocket (remote clients)
    python -m server --web-socket :8080

    # Start as daemon (background)
    python -m server --ipc-socket /tmp/jaato.sock --daemon
"""

from .events import (
    # Base
    Event,
    EventType,
    # Server -> Client events
    AgentCreatedEvent,
    AgentOutputEvent,
    AgentStatusChangedEvent,
    AgentCompletedEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolOutputEvent,
    PermissionRequestedEvent,
    PermissionResolvedEvent,
    ClarificationRequestedEvent,
    ClarificationQuestionEvent,
    ClarificationResolvedEvent,
    PlanUpdatedEvent,
    PlanClearedEvent,
    ContextUpdatedEvent,
    TurnCompletedEvent,
    SystemMessageEvent,
    ErrorEvent,
    SessionListEvent,
    SessionInfoEvent,
    # Client -> Server events
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    StopRequest,
    CommandRequest,
    ClientConfigRequest,
    # Serialization
    serialize_event,
    deserialize_event,
)

from .core import JaatoServer
from .session_manager import SessionManager, RuntimeSessionInfo

__all__ = [
    # Core
    "JaatoServer",
    "SessionManager",
    "RuntimeSessionInfo",
    # Events
    "Event",
    "EventType",
    "AgentCreatedEvent",
    "AgentOutputEvent",
    "AgentStatusChangedEvent",
    "AgentCompletedEvent",
    "ToolCallStartEvent",
    "ToolCallEndEvent",
    "ToolOutputEvent",
    "PermissionRequestedEvent",
    "PermissionResolvedEvent",
    "ClarificationRequestedEvent",
    "ClarificationQuestionEvent",
    "ClarificationResolvedEvent",
    "PlanUpdatedEvent",
    "PlanClearedEvent",
    "ContextUpdatedEvent",
    "TurnCompletedEvent",
    "SystemMessageEvent",
    "ErrorEvent",
    "SessionListEvent",
    "SessionInfoEvent",
    "SendMessageRequest",
    "PermissionResponseRequest",
    "ClarificationResponseRequest",
    "StopRequest",
    "CommandRequest",
    "ClientConfigRequest",
    "serialize_event",
    "deserialize_event",
]
