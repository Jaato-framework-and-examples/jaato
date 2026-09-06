"""Jaato SDK - Protocol and client library for jaato server.

Usage:
    from jaato_sdk.client import IPCClient, IPCRecoveryClient
    from jaato_sdk.events import AgentOutputEvent, PermissionRequestedEvent
"""

from jaato_sdk.client import (
    IPCClient,
    WSClient,
    WSRecoveryClient,
    IPCRecoveryClient,
    ConnectionState,
    RecoveryConfig,
)
from jaato_sdk.client.convenience import (
    Session,
    AgentError,
    PermissionUnhandled,
    ask,
)
# Exported from the package root ON PURPOSE.  The SDK's existing connection
# exceptions (``ReconnectingError``, ``ConnectionClosedError``) are NOT, so a
# consumer cannot catch them without reaching into ``jaato_sdk.client.recovery``
# -- and two out-of-tree consumers were found writing ``except ConnectionError``
# against the BUILTIN, which the SDK's same-named class does not subclass.  An
# exception nobody can conveniently import is one nobody catches.
from jaato_sdk.client.errors import (
    SessionCreateFailed,
    SessionNotConfirmed,
    SessionNotSent,
    SessionRefused,
)
from jaato_sdk.client.recovery import (
    ConnectionClosedError,
    ReconnectingError,
)
from jaato_sdk.events import (
    Event,
    EventType,
    ClientType,
    CommunicationStyle,
    MODEL_MEDIA_CALL_ID,
    PresentationContext,
    serialize_event,
    deserialize_event,
)
from jaato_sdk.constants import PRERENDERED_LINE_PREFIX
from jaato_sdk.helpers import (
    cache_hit_percent_from_counts,
    compute_cache_hit_percent,
    truncation_reason,
)
from jaato_sdk.templates import (
    HELPER_KEYWORDS,
    classify_template_evaluation_kind,
)
from jaato_sdk.completion_processors import ToolCallEntry
from jaato_sdk.cascade_authoring import ProcessorResult
from jaato_sdk.trace import (
    trace,
    provider_trace,
    trace_write,
    resolve_trace_path,
)

__all__ = [
    # Client
    "IPCClient",
    "WSClient",
    "WSRecoveryClient",
    "IPCRecoveryClient",
    "ConnectionState",
    "RecoveryConfig",
    # High-level convenience facade
    "Session",
    "AgentError",
    "PermissionUnhandled",
    "truncation_reason",
    "SessionCreateFailed",
    "SessionNotConfirmed",
    "SessionNotSent",
    "SessionRefused",
    "ConnectionClosedError",
    "ReconnectingError",
    "ask",
    # Events
    "Event",
    "EventType",
    "MODEL_MEDIA_CALL_ID",
    "ClientType",
    "CommunicationStyle",
    "PresentationContext",
    "serialize_event",
    "deserialize_event",
    # Constants
    "PRERENDERED_LINE_PREFIX",
    # Helpers
    "cache_hit_percent_from_counts",
    "compute_cache_hit_percent",
    # Template walker helpers (server 0.6.58+)
    "HELPER_KEYWORDS",
    "classify_template_evaluation_kind",
    # Completion-processor context shape (server 0.6.158+ / SDK 0.14.0+)
    "ToolCallEntry",
    # Completion-processor return contract (server 0.6.160+ / SDK 0.14.2+)
    "ProcessorResult",
    # Trace
    "trace",
    "provider_trace",
    "trace_write",
    "resolve_trace_path",
]
