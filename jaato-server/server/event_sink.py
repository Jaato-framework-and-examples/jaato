"""Transport-agnostic event delivery protocol.

Defines the ``EventSink`` protocol that both IPC and WebSocket servers
implement, so that ``CommandRouter`` and ``SessionManager`` can emit events
without knowing which transport is in use.

``CompositeEventSink`` multiplexes delivery across multiple transports.
Each transport silently ignores ``client_id`` values it doesn't own, so
the composite can safely fan-out to all registered sinks.
"""

from typing import Any, TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from jaato_sdk.events import Event

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shared.peer_identity import PeerCredentials


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

    def broadcast_event(self, event: Event) -> None:
        """Send an event to **every** connected client on this transport.

        Used for daemon-wide events that don't belong to any specific
        session — currently the HandoffGate event family
        (``gate.announced`` / ``gate.released`` / ``gates.snapshot``)
        emitted by the jaato-premium reactor framework.

        Failures to deliver to individual clients are logged and
        swallowed; one disconnected client must never block delivery
        to the rest.

        Thread-safe.
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

    def visible_workspace_paths(self, client_id: str) -> Optional[List[str]]:
        """The workspace paths this client's authenticated user may see.

        ``None`` means "no scoping applies": the transport has no workspace
        manager (IPC), or the connection carries no identity.  A list --
        even an empty one -- is a boundary, and ``session.list`` /
        ``session.attach`` keep a session inside it only when it runs in one
        of these workspaces or was created by this user.
        """
        ...

    def get_client_peer(self, client_id: str) -> Optional["PeerCredentials"]:
        """The OS account that opened this client's connection.

        Distinct from :meth:`get_client_user`, which returns a string for
        ATTRIBUTION.  This returns the structured credential the
        client-path entitlement guards evaluate against — a uid and its
        groups, which is a question only a local socket can answer.

        ``None`` on every transport that cannot report one, which is all
        of them but IPC-on-Linux.  Callers must read it as "there is no
        peer to ask about" and fall back to whatever access control that
        transport does have (the WS bearer, the socket's file mode), never
        as a denial.
        """
        ...


def client_visible_workspaces(sink: Any, client_id: str) -> Optional[List[str]]:
    """``sink.visible_workspace_paths(client_id)``, tolerating a sink without it.

    Same shape as :func:`client_peer`: a transport predating the method (an
    out-of-tree sink) contributes "no scoping" rather than raising, because
    an unscoped listing is the answer every transport gave before.
    """
    fn = getattr(sink, "visible_workspace_paths", None)
    if fn is None:
        return None
    return fn(client_id)


def client_peer(sink: Any, client_id: str) -> Optional["PeerCredentials"]:
    """Ask a sink for a client's peer credential, tolerating one without it.

    ``get_client_peer`` post-dates the :class:`EventSink` protocol, so a
    sink written against the older shape — an out-of-tree transport, a
    test double — simply does not have the method.  An absent peer is a
    VALID answer on every transport but IPC-on-Linux, so its absence must
    read as "there is no peer to ask about" rather than raise.

    One definition because the tolerance is needed at two DIFFERENT
    boundaries: :class:`CompositeEventSink` tolerates its members, and a
    caller holding a sink directly (``CommandRouter``) tolerates its own.
    Deriving it twice is how the two come to disagree — which is exactly
    what happened: the composite tolerated absence and the router raised
    ``AttributeError`` on the same sink.
    """
    getter = getattr(sink, "get_client_peer", None)
    if getter is None:
        return None
    return getter(client_id)


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

    def broadcast_event(self, event: Event) -> None:
        """Fan-out a daemon-wide broadcast across all registered sinks.

        Each underlying sink iterates its own client registry and
        delivers the event; this composite just forwards the call.
        """
        for sink in self._sinks:
            sink.broadcast_event(event)

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

    def visible_workspace_paths(self, client_id: str) -> Optional[List[str]]:
        """Return the first sink's answer that scopes this client, else ``None``."""
        for sink in self._sinks:
            paths = client_visible_workspaces(sink, client_id)
            if paths is not None:
                return paths
        return None

    def get_client_peer(self, client_id: str) -> Optional["PeerCredentials"]:
        """Return the first non-None peer credential from any sink.

        Same shape as :meth:`get_client_user`: unknown ``client_id`` values
        are ignored by sinks that do not own them, so the first sink that
        recognises the id supplies the answer.  A sink predating this
        method (an out-of-tree transport) contributes ``None`` rather than
        raising, because an absent peer is a valid answer here and must not
        become an error on a transport that simply has none.
        """
        for sink in self._sinks:
            peer = client_peer(sink, client_id)
            if peer is not None:
                return peer
        return None
