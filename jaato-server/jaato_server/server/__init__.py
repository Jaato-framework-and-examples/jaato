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

# Lazy (#1549).  Importing any submodule (``runner_user``, ``pool_admin``,
# ``apparmor`` ...) runs this ``__init__`` first.  An eager
# ``from .core import JaatoServer`` here made reading one constant from
# ``jaato_server.server.*`` load the whole daemon: core, the session
# manager, permission, memory, todo, telemetry (~110 modules).  The
# re-exported names still resolve on first attribute access, so
# ``from jaato_server.server import JaatoServer`` keeps working.  Same shape
# as the subagent plugin package (#1267).
_LAZY_IMPORTS = {
    "JaatoServer": ".core",
    "SessionManager": ".session_manager",
    "RuntimeSessionInfo": ".session_manager",
    "Event": "jaato_sdk.events",
    "EventType": "jaato_sdk.events",
    "AgentCreatedEvent": "jaato_sdk.events",
    "AgentOutputEvent": "jaato_sdk.events",
    "AgentStatusChangedEvent": "jaato_sdk.events",
    "AgentCompletedEvent": "jaato_sdk.events",
    "ToolCallStartEvent": "jaato_sdk.events",
    "ToolCallEndEvent": "jaato_sdk.events",
    "ToolOutputEvent": "jaato_sdk.events",
    "PermissionRequestedEvent": "jaato_sdk.events",
    "PermissionResolvedEvent": "jaato_sdk.events",
    "ClarificationRequestedEvent": "jaato_sdk.events",
    "ClarificationQuestionEvent": "jaato_sdk.events",
    "ClarificationResolvedEvent": "jaato_sdk.events",
    "PlanUpdatedEvent": "jaato_sdk.events",
    "PlanClearedEvent": "jaato_sdk.events",
    "ContextUpdatedEvent": "jaato_sdk.events",
    "TurnCompletedEvent": "jaato_sdk.events",
    "SystemMessageEvent": "jaato_sdk.events",
    "ErrorEvent": "jaato_sdk.events",
    "RetryEvent": "jaato_sdk.events",
    "SessionListEvent": "jaato_sdk.events",
    "SessionInfoEvent": "jaato_sdk.events",
    "SendMessageRequest": "jaato_sdk.events",
    "PermissionResponseRequest": "jaato_sdk.events",
    "ClarificationResponseRequest": "jaato_sdk.events",
    "ClarificationBatchEvent": "jaato_sdk.events",
    "ClarificationBatchResponseEvent": "jaato_sdk.events",
    "StopRequest": "jaato_sdk.events",
    "CommandRequest": "jaato_sdk.events",
    "ClientConfigRequest": "jaato_sdk.events",
    "GetInstructionBudgetRequest": "jaato_sdk.events",
    "InstructionBudgetEvent": "jaato_sdk.events",
    "serialize_event": "jaato_sdk.events",
    "deserialize_event": "jaato_sdk.events",
}


def __getattr__(name):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


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
    "RetryEvent",
    "SessionListEvent",
    "SessionInfoEvent",
    "SendMessageRequest",
    "PermissionResponseRequest",
    "ClarificationResponseRequest",
    "ClarificationBatchEvent",
    "ClarificationBatchResponseEvent",
    "StopRequest",
    "CommandRequest",
    "ClientConfigRequest",
    "GetInstructionBudgetRequest",
    "InstructionBudgetEvent",
    "serialize_event",
    "deserialize_event",
]
