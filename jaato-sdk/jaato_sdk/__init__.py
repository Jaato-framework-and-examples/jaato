"""Jaato SDK - Protocol and client library for jaato server.

Usage:
    from jaato_sdk.client import IPCClient, IPCRecoveryClient
    from jaato_sdk.events import AgentOutputEvent, PermissionRequestedEvent
"""

from jaato_sdk.client import (
    IPCClient,
    IPCRecoveryClient,
    ConnectionState,
    RecoveryConfig,
)
from jaato_sdk.events import (
    Event,
    EventType,
    ClientType,
    PresentationContext,
    serialize_event,
    deserialize_event,
)
from jaato_sdk.trace import (
    trace,
    provider_trace,
    trace_write,
    resolve_trace_path,
)

__all__ = [
    # Client
    "IPCClient",
    "IPCRecoveryClient",
    "ConnectionState",
    "RecoveryConfig",
    # Events
    "Event",
    "EventType",
    "ClientType",
    "PresentationContext",
    "serialize_event",
    "deserialize_event",
    # Trace
    "trace",
    "provider_trace",
    "trace_write",
    "resolve_trace_path",
]
