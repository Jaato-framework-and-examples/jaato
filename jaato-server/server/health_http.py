"""Minimal HTTP health endpoint for gossip observability.

Exposes ``GET /health`` as a raw asyncio HTTP server (zero new dependencies).
Returns peer state as JSON for Docker healthchecks and E2E test assertions.

Lifecycle:
    1. Constructed with references to ``PeerRegistry`` and ``ServerHealthCollector``.
    2. ``start(host, port)`` binds a TCP server.
    3. ``stop()`` closes the server.

Only ``GET /health`` is handled. All other paths return 404.
"""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from server.health import ServerHealthCollector
    from server.peers import PeerRegistry

logger = logging.getLogger(__name__)


class HealthHTTPServer:
    """Raw asyncio HTTP server serving ``GET /health``.

    Uses ``asyncio.start_server`` with manual HTTP/1.1 response formatting.
    No external dependencies.

    Args:
        peer_registry: The gossip peer registry (may be ``None`` if gossip
            is not configured — the endpoint returns 503 in that case).
        health_collector: The health metrics collector.
    """

    def __init__(
        self,
        peer_registry: Optional["PeerRegistry"],
        health_collector: Optional["ServerHealthCollector"],
    ) -> None:
        self._peer_registry = peer_registry
        self._health_collector = health_collector
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self, host: str = "0.0.0.0", port: int = 9090) -> None:
        """Bind and start listening.

        Args:
            host: Bind address.
            port: TCP port.
        """
        self._server = await asyncio.start_server(
            self._handle_connection, host, port,
        )
        logger.info("Health HTTP server listening on %s:%d", host, port)

    async def stop(self) -> None:
        """Close the TCP server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Health HTTP server stopped")

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection.

        Reads the request line, dispatches to ``_handle_health`` or returns
        404, then closes the connection.
        """
        try:
            # Read request line (e.g. "GET /health HTTP/1.1\r\n")
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=5.0,
            )
            if not request_line:
                return

            parts = request_line.decode("utf-8", errors="replace").strip().split()
            if len(parts) < 2:
                self._send_response(writer, 400, {"error": "bad request"})
                return

            method, path = parts[0], parts[1]

            # Drain remaining headers (we don't need them)
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in (b"\r\n", b"\n", b""):
                    break

            if method == "GET" and path == "/health":
                self._handle_health(writer)
            else:
                self._send_response(writer, 404, {"error": "not found"})

        except asyncio.TimeoutError:
            pass
        except Exception:
            logger.debug("Health HTTP handler error", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # /health handler
    # ------------------------------------------------------------------

    def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """Build and send the ``/health`` JSON response.

        Returns 503 if no peer registry is configured, 200 otherwise.
        """
        if not self._peer_registry or not self._health_collector:
            self._send_response(writer, 503, {"error": "no peer registry"})
            return

        snapshot = self._health_collector.collect()
        peers = []
        for entry in self._peer_registry.get_peer_snapshots():
            peer_info = {
                "name": entry.config.name,
                "state": entry.state.value,
                "missed_count": entry.missed_count,
                "last_heartbeat_at": entry.last_heartbeat_at,
            }
            # Include metrics from last heartbeat if available
            if entry.last_heartbeat:
                peer_info["cpu_percent"] = entry.last_heartbeat.cpu_percent
                peer_info["memory_percent"] = entry.last_heartbeat.memory_percent
                peer_info["uptime_seconds"] = entry.last_heartbeat.uptime_seconds
            peers.append(peer_info)

        body = {
            "server_id": self._health_collector.server_id,
            "server_name": self._health_collector.server_name,
            "uptime_seconds": snapshot.uptime_seconds,
            "peers": peers,
        }

        self._send_response(writer, 200, body)

    # ------------------------------------------------------------------
    # HTTP response helper
    # ------------------------------------------------------------------

    @staticmethod
    def _send_response(
        writer: asyncio.StreamWriter,
        status_code: int,
        body: dict,
    ) -> None:
        """Write a minimal HTTP/1.1 JSON response."""
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 503: "Service Unavailable"}.get(
            status_code, "Unknown",
        )
        payload = json.dumps(body)
        response = (
            f"HTTP/1.1 {status_code} {reason}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{payload}"
        )
        writer.write(response.encode("utf-8"))
