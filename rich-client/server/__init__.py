"""Jaato Server - Headless backend for multi-client support.

This package provides:
- JaatoServer: Core logic extracted from RichClient
- Event protocol: Typed events for client-server communication
- WebSocket server: Real-time bidirectional communication

Usage:
    # Headless mode (server only)
    python rich_client.py --headless --port 8080

    # TUI + server mode
    python rich_client.py --expose-server :8080
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
    # Client -> Server events
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    StopRequest,
    CommandRequest,
    # Serialization
    serialize_event,
    deserialize_event,
)

from .core import JaatoServer

__all__ = [
    # Core
    "JaatoServer",
    # Events
    "Event",
    "EventType",
    "AgentCreatedEvent",
    "AgentOutputEvent",
    "AgentStatusChangedEvent",
    "AgentCompletedEvent",
    "ToolCallStartEvent",
    "ToolCallEndEvent",
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
    "SendMessageRequest",
    "PermissionResponseRequest",
    "ClarificationResponseRequest",
    "StopRequest",
    "CommandRequest",
    "serialize_event",
    "deserialize_event",
]
