"""Peer-to-peer gossip registry for multi-server Jaato deployments.

Phase 1 implements:
- Outbound WebSocket connections to peer servers
- Periodic heartbeat broadcast with health metrics
- Inbound heartbeat processing and peer liveness tracking
- Peer state machine: HEALTHY / DEGRADED / UNREACHABLE

If no ``servers.json`` is configured, none of this code is activated.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from jaato_sdk.events import (
    EventType,
    PeerHeartbeatEvent,
    serialize_event,
    deserialize_event,
)

if TYPE_CHECKING:
    from server.health import ServerHealthCollector

try:
    import websockets
    from websockets import ClientConnection
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    ClientConnection = Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class PeerState(str, Enum):
    """Liveness state of a peer server."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"


@dataclass
class GossipConfig:
    """Tuning knobs for the gossip protocol.

    Loaded from the ``gossip`` section of ``servers.json``.
    """

    heartbeat_interval_seconds: float = 5.0
    degraded_after_missed: int = 3
    unreachable_after_missed: int = 5


@dataclass
class ServerConfig:
    """Static configuration for a single peer from ``servers.json``."""

    name: str
    transport: str  # "ws" or "ipc" (ipc skipped for Phase 1)
    address: str
    tags: List[str] = field(default_factory=list)


@dataclass
class PeerEntry:
    """Runtime state for a single known peer.

    Holds the latest heartbeat, connection state, and the outbound WebSocket.
    """

    config: ServerConfig
    state: PeerState = PeerState.UNREACHABLE
    last_heartbeat: Optional[PeerHeartbeatEvent] = None
    last_heartbeat_at: Optional[float] = None  # time.monotonic()
    missed_count: int = 0
    _outbound_ws: Optional[Any] = field(default=None, repr=False)
    _connection_task: Optional[asyncio.Task] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# PeerRegistry
# ---------------------------------------------------------------------------

class PeerRegistry:
    """Manages gossip connections and peer liveness tracking.

    Lifecycle:
        1. Constructed with parsed ``servers.json`` data.
        2. ``start(health_collector)`` spawns async tasks for outbound
           connections and the heartbeat loop.
        3. Inbound peer WebSocket connections are handed off by the
           ``JaatoWSServer`` via ``handle_peer_connection()``.
        4. ``shutdown()`` cancels all tasks and closes connections.

    Thread safety: all mutable state is accessed only from the asyncio
    event loop — no locks required.
    """

    def __init__(
        self,
        server_id: str,
        server_name: str,
        gossip_config: GossipConfig,
        peer_configs: List[ServerConfig],
    ) -> None:
        self._server_id = server_id
        self._server_name = server_name
        self._gossip_config = gossip_config

        # Build peer table keyed by name
        self._peers: Dict[str, PeerEntry] = {
            cfg.name: PeerEntry(config=cfg)
            for cfg in peer_configs
        }

        self._health_collector: Optional["ServerHealthCollector"] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, health_collector: "ServerHealthCollector") -> None:
        """Spawn outbound connection loops and the heartbeat broadcaster."""
        self._health_collector = health_collector
        self._running = True

        # Outbound connections (one task per WS peer)
        for name, entry in self._peers.items():
            if entry.config.transport != "ws":
                logger.debug(
                    "Skipping non-WS peer %s (transport=%s)",
                    name, entry.config.transport,
                )
                continue
            entry._connection_task = asyncio.create_task(
                self._outbound_connection_loop(entry),
                name=f"peer-connect-{name}",
            )

        # Heartbeat broadcaster
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="peer-heartbeat",
        )

        logger.info(
            "PeerRegistry started: %d peer(s) configured", len(self._peers)
        )

    async def shutdown(self) -> None:
        """Cancel all tasks and close outbound WebSockets."""
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cancel connection tasks and close sockets
        for entry in self._peers.values():
            if entry._connection_task and not entry._connection_task.done():
                entry._connection_task.cancel()
                try:
                    await entry._connection_task
                except asyncio.CancelledError:
                    pass
            if entry._outbound_ws:
                try:
                    await entry._outbound_ws.close()
                except Exception:
                    pass
                entry._outbound_ws = None

        logger.info("PeerRegistry shut down")

    # ------------------------------------------------------------------
    # Inbound peer connections (called by JaatoWSServer)
    # ------------------------------------------------------------------

    async def handle_peer_connection(self, websocket: Any) -> None:
        """Handle an inbound WebSocket connection from a peer server.

        Reads heartbeat messages in a loop until the connection closes.
        The peer identifies itself via the ``server_name`` field inside each
        heartbeat (not via headers — headers are only used for the initial
        peer detection handshake in the WS server).

        Args:
            websocket: The inbound ``ServerConnection``.
        """
        peer_name = websocket.request.headers.get("X-Jaato-Peer-Name", "unknown")
        logger.info("Inbound peer connection from %s", peer_name)

        try:
            async for message in websocket:
                try:
                    event = deserialize_event(message)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "Invalid message from peer %s, ignoring", peer_name,
                    )
                    continue

                if isinstance(event, PeerHeartbeatEvent):
                    self._on_heartbeat_received(event.server_name, event)
                else:
                    logger.debug(
                        "Ignoring non-heartbeat event from peer %s: %s",
                        peer_name, event.type,
                    )
        except Exception as exc:
            logger.info("Peer %s disconnected: %s", peer_name, exc)

    # ------------------------------------------------------------------
    # Outbound connections
    # ------------------------------------------------------------------

    async def _outbound_connection_loop(self, entry: PeerEntry) -> None:
        """Maintain a persistent outbound WebSocket to a peer.

        Reconnects with exponential backoff (1 s → 60 s cap) on failure.
        """
        backoff = 1.0
        max_backoff = 60.0

        while self._running:
            try:
                extra_headers = {
                    "X-Jaato-Peer": "true",
                    "X-Jaato-Peer-Name": self._server_name,
                }
                async with websockets.connect(
                    entry.config.address,
                    additional_headers=extra_headers,
                ) as ws:
                    entry._outbound_ws = ws
                    backoff = 1.0  # reset on successful connect
                    logger.info(
                        "Connected to peer %s at %s",
                        entry.config.name, entry.config.address,
                    )

                    # Read inbound heartbeats from this peer
                    async for message in ws:
                        try:
                            event = deserialize_event(message)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if isinstance(event, PeerHeartbeatEvent):
                            self._on_heartbeat_received(
                                event.server_name, event,
                            )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                entry._outbound_ws = None
                logger.debug(
                    "Connection to peer %s failed: %s (retry in %.0fs)",
                    entry.config.name, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Heartbeat loop
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodically broadcast heartbeats and check peer liveness."""
        interval = self._gossip_config.heartbeat_interval_seconds

        while self._running:
            try:
                await asyncio.sleep(interval)

                # Collect health and build heartbeat
                snapshot = self._health_collector.collect()

                from server.websocket import _get_server_version
                heartbeat = PeerHeartbeatEvent(
                    server_id=self._health_collector.server_id,
                    server_name=self._health_collector.server_name,
                    server_version=_get_server_version(),
                    active_sessions=snapshot.active_sessions,
                    active_agents=snapshot.active_agents,
                    available_providers=snapshot.available_providers,
                    available_models=snapshot.available_models,
                    tags=snapshot.tags,
                    cpu_percent=snapshot.cpu_percent,
                    memory_percent=snapshot.memory_percent,
                    uptime_seconds=snapshot.uptime_seconds,
                )
                message = serialize_event(heartbeat)

                # Send to all connected peers
                for entry in self._peers.values():
                    if entry._outbound_ws:
                        try:
                            await entry._outbound_ws.send(message)
                        except Exception:
                            # Connection dead — will be cleaned up by
                            # _outbound_connection_loop on next reconnect.
                            entry._outbound_ws = None

                # Check liveness
                self._check_peer_liveness()

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in heartbeat loop")

    # ------------------------------------------------------------------
    # Heartbeat processing
    # ------------------------------------------------------------------

    def _on_heartbeat_received(
        self, peer_name: str, heartbeat: PeerHeartbeatEvent,
    ) -> None:
        """Process an inbound heartbeat from a peer.

        Transitions the peer to HEALTHY and resets the missed counter.
        """
        entry = self._peers.get(peer_name)
        if entry is None:
            logger.debug("Heartbeat from unknown peer %s, ignoring", peer_name)
            return

        old_state = entry.state
        entry.state = PeerState.HEALTHY
        entry.missed_count = 0
        entry.last_heartbeat = heartbeat
        entry.last_heartbeat_at = time.monotonic()

        if old_state != PeerState.HEALTHY:
            logger.info(
                "Peer %s transitioned %s -> HEALTHY", peer_name, old_state.value,
            )

    def _check_peer_liveness(self) -> None:
        """Increment missed counts and transition peer states."""
        degraded_threshold = self._gossip_config.degraded_after_missed
        unreachable_threshold = self._gossip_config.unreachable_after_missed

        for name, entry in self._peers.items():
            if entry.state == PeerState.HEALTHY:
                entry.missed_count += 1
                if entry.missed_count >= unreachable_threshold:
                    entry.state = PeerState.UNREACHABLE
                    logger.warning("Peer %s is now UNREACHABLE", name)
                elif entry.missed_count >= degraded_threshold:
                    entry.state = PeerState.DEGRADED
                    logger.warning("Peer %s is now DEGRADED", name)

            elif entry.state == PeerState.DEGRADED:
                entry.missed_count += 1
                if entry.missed_count >= unreachable_threshold:
                    entry.state = PeerState.UNREACHABLE
                    logger.warning("Peer %s is now UNREACHABLE", name)

            # UNREACHABLE peers just keep incrementing (no further transition)
            elif entry.state == PeerState.UNREACHABLE:
                entry.missed_count += 1

    # ------------------------------------------------------------------
    # Query interface (for Phase 2 environment aspect)
    # ------------------------------------------------------------------

    def get_peer_snapshots(self) -> List[PeerEntry]:
        """Return a shallow copy of all peer entries.

        The returned list is safe to iterate without affecting internal state.
        Callers should not mutate the PeerEntry objects.
        """
        return list(self._peers.values())
