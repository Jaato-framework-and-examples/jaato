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
    serialize_event,
    deserialize_event,
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
    "serialize_event",
    "deserialize_event",
]
