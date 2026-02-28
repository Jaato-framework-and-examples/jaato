"""Tests for server.health_http — HealthHTTPServer."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from server.health_http import HealthHTTPServer
from server.health import ServerHealthSnapshot
from server.peers import PeerEntry, PeerState, ServerConfig

from jaato_sdk.events import PeerHeartbeatEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_peer_entry(
    name: str = "server-b",
    state: PeerState = PeerState.HEALTHY,
    missed_count: int = 0,
    last_heartbeat_at: float = 1234567890.0,
    cpu_percent: float = 15.0,
    memory_percent: float = 40.0,
) -> PeerEntry:
    """Create a PeerEntry with a populated last_heartbeat."""
    config = ServerConfig(name=name, transport="ws", address=f"ws://{name}:8080")
    heartbeat = PeerHeartbeatEvent(
        server_name=name,
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        uptime_seconds=120.0,
    )
    return PeerEntry(
        config=config,
        state=state,
        last_heartbeat=heartbeat,
        last_heartbeat_at=last_heartbeat_at,
        missed_count=missed_count,
    )


def _mock_registry(peers: list) -> MagicMock:
    registry = MagicMock()
    registry.get_peer_snapshots.return_value = peers
    return registry


def _mock_collector(
    server_id: str = "test-id",
    server_name: str = "server-a",
    uptime: float = 60.0,
) -> MagicMock:
    collector = MagicMock()
    collector.server_id = server_id
    collector.server_name = server_name
    collector.collect.return_value = ServerHealthSnapshot(
        cpu_percent=10.0,
        memory_percent=50.0,
        active_sessions=1,
        active_agents=2,
        uptime_seconds=uptime,
        available_providers=["google_genai"],
        available_models=["gemini-2.5-flash"],
        tags=["e2e"],
    )
    return collector


async def _fetch(port: int, path: str = "/health") -> tuple:
    """Make a raw HTTP GET and return (status_code, body_dict)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    request = f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n"
    writer.write(request.encode())
    await writer.drain()

    # Read response
    data = await asyncio.wait_for(reader.read(65536), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    response = data.decode("utf-8")
    # Split headers and body
    header_end = response.index("\r\n\r\n")
    status_line = response[:response.index("\r\n")]
    status_code = int(status_line.split()[1])
    body = json.loads(response[header_end + 4:])
    return status_code, body


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthReturns200WithPeers:
    """GET /health returns 200 with peer data when registry is configured."""

    @pytest.mark.asyncio
    async def test_health_returns_200_with_peers(self):
        peer = _make_peer_entry()
        registry = _mock_registry([peer])
        collector = _mock_collector()

        server = HealthHTTPServer(peer_registry=registry, health_collector=collector)
        await server.start("127.0.0.1", 0)

        # Get the actual bound port
        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _fetch(port)

            assert status == 200
            assert body["server_id"] == "test-id"
            assert body["server_name"] == "server-a"
            assert body["uptime_seconds"] == 60.0
            assert len(body["peers"]) == 1

            peer_data = body["peers"][0]
            assert peer_data["name"] == "server-b"
            assert peer_data["state"] == "healthy"
            assert peer_data["missed_count"] == 0
            assert peer_data["cpu_percent"] == 15.0
            assert peer_data["memory_percent"] == 40.0
        finally:
            await server.stop()


class TestHealthReturns503WithoutRegistry:
    """GET /health returns 503 when no peer registry is configured."""

    @pytest.mark.asyncio
    async def test_health_returns_503_without_registry(self):
        server = HealthHTTPServer(peer_registry=None, health_collector=None)
        await server.start("127.0.0.1", 0)

        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _fetch(port)

            assert status == 503
            assert body["error"] == "no peer registry"
        finally:
            await server.stop()


class TestHealthReturns404ForOtherPaths:
    """GET /foo returns 404."""

    @pytest.mark.asyncio
    async def test_health_returns_404_for_other_paths(self):
        registry = _mock_registry([])
        collector = _mock_collector()

        server = HealthHTTPServer(peer_registry=registry, health_collector=collector)
        await server.start("127.0.0.1", 0)

        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _fetch(port, "/foo")

            assert status == 404
            assert body["error"] == "not found"
        finally:
            await server.stop()


class TestHealthPeerWithoutHeartbeat:
    """Peer entry without a last_heartbeat omits metric fields."""

    @pytest.mark.asyncio
    async def test_peer_without_heartbeat(self):
        config = ServerConfig(name="server-c", transport="ws", address="ws://server-c:8080")
        peer = PeerEntry(config=config, state=PeerState.UNREACHABLE)
        registry = _mock_registry([peer])
        collector = _mock_collector()

        server = HealthHTTPServer(peer_registry=registry, health_collector=collector)
        await server.start("127.0.0.1", 0)

        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _fetch(port)

            assert status == 200
            peer_data = body["peers"][0]
            assert peer_data["name"] == "server-c"
            assert peer_data["state"] == "unreachable"
            # No heartbeat metrics
            assert "cpu_percent" not in peer_data
        finally:
            await server.stop()
