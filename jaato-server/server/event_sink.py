"""Transport-agnostic event delivery protocol.

Defines the ``EventSink`` protocol that both IPC and WebSocket servers
implement, so that ``CommandRouter`` and ``SessionManager`` can emit events
without knowing which transport is in use.

``CompositeEventSink`` multiplexes delivery across multiple transports.
Each transport silently ignores ``client_id`` values it doesn't own, so
the composite can safely fan-out to all registered sinks.
"""

from typing import List, Optional, Protocol, runtime_checkable

from jaato_sdk.events import Event


@runtime_checkable
class EventSink(Protocol):
    """Transport-agnostic interface for routing events to clients.

    Both ``JaatoIPCServer`` and the WebSocket event sink adapter implement
    this protocol.  ``CommandRouter`` uses it to emit events without
    coupling to any specific transport.

    All methods must be **thread-safe** — they are called from model
    threads, session threads, and the daemon's request-handler threads.
    """

    def send_event(self, client_id: str, event: Event) -> None:
        """Send an event to a specific client.

        If ``client_id`` is not known to this transport, the call is
        silently ignored (another sink in the composite may own it).
        """
        ...

    def set_client_session(self, client_id: str, session_id: str) -> None:
        """Associate a client with a session (for broadcast targeting)."""
        ...

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        """Get the workspace path associated with a client."""
        ...

    def set_client_workspace(self, client_id: str, workspace_path: str) -> None:
        """Associate a workspace path with a client."""
        ...

    def get_client_user(self, client_id: str) -> Optional[str]:
        """Get the authenticated user identity for a client.

        Returns ``None`` if the client is not authenticated or if the
        transport does not support authentication (e.g., local IPC).
        """
        ...

    def set_client_user(self, client_id: str, user_id: str) -> None:
        """Associate an authenticated user identity with a client."""
        ...


class CompositeEventSink:
    """Multiplexes event delivery across multiple transport sinks.

    Each registered sink receives every call.  Unknown ``client_id``
    values are silently ignored by sinks that don't own them, so the
    composite can fan-out without checking ownership.

    Thread-safe by delegation — each underlying sink is individually
    thread-safe.
    """

    def __init__(self) -> None:
        self._sinks: List[EventSink] = []

    def add_sink(self, sink: EventSink) -> None:
        """Register a transport sink."""
        self._sinks.append(sink)

    def send_event(self, client_id: str, event: Event) -> None:
        """Fan-out to all registered sinks."""
        for sink in self._sinks:
            sink.send_event(client_id, event)

    def set_client_session(self, client_id: str, session_id: str) -> None:
        """Fan-out to all registered sinks."""
        for sink in self._sinks:
            sink.set_client_session(client_id, session_id)

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        """Return the first non-None workspace from any sink."""
        for sink in self._sinks:
            ws = sink.get_client_workspace(client_id)
            if ws is not None:
                return ws
        return None

    def set_client_workspace(self, client_id: str, workspace_path: str) -> None:
        """Fan-out to all registered sinks."""
        for sink in self._sinks:
            sink.set_client_workspace(client_id, workspace_path)

    def get_client_user(self, client_id: str) -> Optional[str]:
        """Return the first non-None user from any sink."""
        for sink in self._sinks:
            user = sink.get_client_user(client_id)
            if user is not None:
                return user
        return None

    def set_client_user(self, client_id: str, user_id: str) -> None:
        """Fan-out to all registered sinks."""
        for sink in self._sinks:
            sink.set_client_user(client_id, user_id)
