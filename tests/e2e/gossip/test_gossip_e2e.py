#!/usr/bin/env python3
"""E2E test: verify two jaato-server instances discover each other via gossip.

Uses only stdlib (urllib, json, time, sys). Exit code 0 = pass, 1 = fail.
Designed to run inside a Docker compose test-runner container.
"""

import json
import sys
import time
import urllib.request

SERVER_A_HEALTH = "http://server-a:9090/health"
SERVER_B_HEALTH = "http://server-b:9090/health"

POLL_INTERVAL = 1.0  # seconds between polls
TIMEOUT = 30.0       # max seconds to wait for each phase


def fetch_health(url: str) -> dict | None:
    """GET a health endpoint, return parsed JSON or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                return json.loads(resp.read())
    except Exception:
        return None


def wait_for_healthy(url: str, label: str) -> dict:
    """Poll until the health endpoint returns 200."""
    print(f"[{label}] Waiting for health endpoint at {url} ...")
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        data = fetch_health(url)
        if data is not None:
            print(f"[{label}] Health endpoint responding (uptime={data.get('uptime_seconds', '?'):.1f}s)")
            return data
        time.sleep(POLL_INTERVAL)
    print(f"[{label}] TIMEOUT: health endpoint did not respond within {TIMEOUT}s")
    sys.exit(1)


def wait_for_peer_healthy(url: str, label: str, peer_name: str) -> dict:
    """Poll until the server reports a specific peer as 'healthy'."""
    print(f"[{label}] Waiting for peer '{peer_name}' to become healthy ...")
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        data = fetch_health(url)
        if data:
            for peer in data.get("peers", []):
                if peer["name"] == peer_name and peer["state"] == "healthy":
                    print(f"[{label}] Peer '{peer_name}' is HEALTHY (missed_count={peer['missed_count']})")
                    return data
        time.sleep(POLL_INTERVAL)
    # Final attempt for error reporting
    data = fetch_health(url)
    peers_str = json.dumps(data.get("peers", []), indent=2) if data else "N/A"
    print(f"[{label}] TIMEOUT: peer '{peer_name}' not healthy within {TIMEOUT}s")
    print(f"[{label}] Current peers: {peers_str}")
    sys.exit(1)


def assert_peer_fields(peer: dict, label: str) -> None:
    """Verify peer data has expected fields."""
    required = ["name", "state", "missed_count", "last_heartbeat_at"]
    for field in required:
        assert field in peer, f"[{label}] Missing field '{field}' in peer: {peer}"
    # Heartbeat metrics should be populated
    assert "cpu_percent" in peer, f"[{label}] Missing 'cpu_percent' in peer: {peer}"
    assert "memory_percent" in peer, f"[{label}] Missing 'memory_percent' in peer: {peer}"
    print(f"[{label}] Peer '{peer['name']}' fields validated: "
          f"cpu={peer['cpu_percent']:.1f}%, mem={peer['memory_percent']:.1f}%")


def main():
    print("=" * 60)
    print("Gossip E2E Test")
    print("=" * 60)

    # Phase 1: Wait for both health endpoints to respond
    print("\n--- Phase 1: Health endpoints ---")
    wait_for_healthy(SERVER_A_HEALTH, "server-a")
    wait_for_healthy(SERVER_B_HEALTH, "server-b")

    # Phase 2: Wait for mutual peer discovery
    print("\n--- Phase 2: Peer discovery ---")
    data_a = wait_for_peer_healthy(SERVER_A_HEALTH, "server-a", "server-b")
    data_b = wait_for_peer_healthy(SERVER_B_HEALTH, "server-b", "server-a")

    # Phase 3: Validate peer data
    print("\n--- Phase 3: Validation ---")

    # server-a should see server-b as healthy
    peers_a = {p["name"]: p for p in data_a["peers"]}
    assert "server-b" in peers_a, f"server-a does not list server-b as peer: {list(peers_a.keys())}"
    assert_peer_fields(peers_a["server-b"], "server-a")

    # server-b should see server-a as healthy
    peers_b = {p["name"]: p for p in data_b["peers"]}
    assert "server-a" in peers_b, f"server-b does not list server-a as peer: {list(peers_b.keys())}"
    assert_peer_fields(peers_b["server-a"], "server-b")

    # Verify server identity fields
    assert data_a["server_name"] == "server-a", f"Unexpected server_name: {data_a['server_name']}"
    assert data_b["server_name"] == "server-b", f"Unexpected server_name: {data_b['server_name']}"
    assert data_a["server_id"], "server-a has empty server_id"
    assert data_b["server_id"], "server-b has empty server_id"
    assert data_a["server_id"] != data_b["server_id"], "Both servers have the same server_id"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
