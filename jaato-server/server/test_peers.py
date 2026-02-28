"""Tests for server.peers — PeerRegistry state machine and config parsing."""

import time

import pytest

from server.peers import (
    GossipConfig,
    PeerEntry,
    PeerRegistry,
    PeerState,
    ServerConfig,
)
from jaato_sdk.events import PeerHeartbeatEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_heartbeat(server_name: str = "peer-1", **overrides) -> PeerHeartbeatEvent:
    """Create a minimal PeerHeartbeatEvent for testing."""
    defaults = dict(
        server_id="test-id",
        server_name=server_name,
        server_version="0.0.1",
        active_sessions=1,
        active_agents=2,
        available_providers=["google_genai"],
        available_models=["gemini-2.5-flash"],
        tags=["gpu"],
        cpu_percent=25.0,
        memory_percent=40.0,
        uptime_seconds=120.0,
    )
    defaults.update(overrides)
    return PeerHeartbeatEvent(**defaults)


def _make_registry(
    gossip_config: GossipConfig | None = None,
    peer_names: list[str] | None = None,
) -> PeerRegistry:
    """Create a PeerRegistry with simple defaults (no async tasks)."""
    if gossip_config is None:
        gossip_config = GossipConfig()
    if peer_names is None:
        peer_names = ["peer-1"]
    peer_configs = [
        ServerConfig(name=n, transport="ws", address=f"ws://host-{n}:8080")
        for n in peer_names
    ]
    return PeerRegistry(
        server_id="self-id",
        server_name="self",
        gossip_config=gossip_config,
        peer_configs=peer_configs,
    )


# ---------------------------------------------------------------------------
# Tests — state transitions
# ---------------------------------------------------------------------------

class TestHeartbeatReceived:
    """Receiving a heartbeat transitions a peer to HEALTHY."""

    def test_unreachable_to_healthy(self):
        registry = _make_registry()
        peer = registry._peers["peer-1"]
        assert peer.state == PeerState.UNREACHABLE  # initial state

        registry._on_heartbeat_received("peer-1", _make_heartbeat())

        assert peer.state == PeerState.HEALTHY
        assert peer.missed_count == 0
        assert peer.last_heartbeat is not None

    def test_degraded_to_healthy(self):
        registry = _make_registry()
        peer = registry._peers["peer-1"]
        peer.state = PeerState.DEGRADED
        peer.missed_count = 3

        registry._on_heartbeat_received("peer-1", _make_heartbeat())

        assert peer.state == PeerState.HEALTHY
        assert peer.missed_count == 0

    def test_healthy_stays_healthy(self):
        registry = _make_registry()
        peer = registry._peers["peer-1"]
        peer.state = PeerState.HEALTHY

        registry._on_heartbeat_received("peer-1", _make_heartbeat())

        assert peer.state == PeerState.HEALTHY

    def test_unknown_peer_ignored(self):
        registry = _make_registry()
        # Should not raise
        registry._on_heartbeat_received("unknown-peer", _make_heartbeat("unknown-peer"))
        assert "unknown-peer" not in registry._peers


class TestMissedHeartbeats:
    """Missing heartbeats transition HEALTHY → DEGRADED → UNREACHABLE."""

    def test_healthy_to_degraded(self):
        cfg = GossipConfig(degraded_after_missed=3, unreachable_after_missed=5)
        registry = _make_registry(gossip_config=cfg)
        peer = registry._peers["peer-1"]
        peer.state = PeerState.HEALTHY
        peer.missed_count = 0

        # Simulate 3 missed intervals
        for _ in range(3):
            registry._check_peer_liveness()

        assert peer.state == PeerState.DEGRADED

    def test_healthy_to_unreachable(self):
        cfg = GossipConfig(degraded_after_missed=3, unreachable_after_missed=5)
        registry = _make_registry(gossip_config=cfg)
        peer = registry._peers["peer-1"]
        peer.state = PeerState.HEALTHY
        peer.missed_count = 0

        # Simulate 5 missed intervals
        for _ in range(5):
            registry._check_peer_liveness()

        assert peer.state == PeerState.UNREACHABLE

    def test_degraded_to_unreachable(self):
        cfg = GossipConfig(degraded_after_missed=3, unreachable_after_missed=5)
        registry = _make_registry(gossip_config=cfg)
        peer = registry._peers["peer-1"]
        peer.state = PeerState.DEGRADED
        peer.missed_count = 3

        # Two more missed intervals (total 5)
        registry._check_peer_liveness()
        registry._check_peer_liveness()

        assert peer.state == PeerState.UNREACHABLE

    def test_unreachable_recovers_to_healthy(self):
        """UNREACHABLE + heartbeat → HEALTHY (direct, no intermediate)."""
        registry = _make_registry()
        peer = registry._peers["peer-1"]
        peer.state = PeerState.UNREACHABLE
        peer.missed_count = 10

        registry._on_heartbeat_received("peer-1", _make_heartbeat())

        assert peer.state == PeerState.HEALTHY
        assert peer.missed_count == 0


class TestPeerSnapshots:
    """get_peer_snapshots returns a copy of peer state."""

    def test_returns_list(self):
        registry = _make_registry(peer_names=["a", "b"])
        snapshots = registry.get_peer_snapshots()
        assert len(snapshots) == 2

    def test_returns_copy(self):
        registry = _make_registry()
        snapshots = registry.get_peer_snapshots()
        snapshots.clear()  # mutating returned list
        assert len(registry._peers) == 1  # internal state unaffected


class TestServersJsonParsing:
    """GossipConfig and ServerConfig from sample JSON."""

    def test_gossip_config_defaults(self):
        cfg = GossipConfig()
        assert cfg.heartbeat_interval_seconds == 5.0
        assert cfg.degraded_after_missed == 3
        assert cfg.unreachable_after_missed == 5

    def test_gossip_config_custom(self):
        cfg = GossipConfig(
            heartbeat_interval_seconds=10,
            degraded_after_missed=2,
            unreachable_after_missed=4,
        )
        assert cfg.heartbeat_interval_seconds == 10
        assert cfg.degraded_after_missed == 2
        assert cfg.unreachable_after_missed == 4

    def test_server_config(self):
        cfg = ServerConfig(
            name="gpu-box",
            transport="ws",
            address="ws://gpu-box.local:8080",
            tags=["gpu", "ollama"],
        )
        assert cfg.name == "gpu-box"
        assert cfg.transport == "ws"
        assert cfg.address == "ws://gpu-box.local:8080"
        assert cfg.tags == ["gpu", "ollama"]

    def test_server_config_default_tags(self):
        cfg = ServerConfig(name="x", transport="ws", address="ws://x:1")
        assert cfg.tags == []
