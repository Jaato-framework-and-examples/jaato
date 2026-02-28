"""Tests for server.health — ServerHealthCollector."""

import time
from unittest.mock import MagicMock, patch

import pytest

from server.health import ServerHealthCollector, ServerHealthSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session_manager(loaded_count: int = 2, client_count: int = 3):
    """Return a mock SessionManager with ``list_sessions()``."""
    sessions = []
    for i in range(loaded_count):
        s = MagicMock()
        s.is_loaded = True
        s.client_count = client_count
        sessions.append(s)
    # Add an unloaded session
    unloaded = MagicMock()
    unloaded.is_loaded = False
    unloaded.client_count = 0
    sessions.append(unloaded)

    mgr = MagicMock()
    mgr.list_sessions.return_value = sessions
    return mgr


def _make_collector(**overrides):
    defaults = dict(
        session_manager=_mock_session_manager(),
        server_id="test-server-id",
        server_name="test-server",
        tags=["local"],
        start_time=time.monotonic() - 60,  # started 60s ago
        available_providers=["google_genai"],
        available_models=["gemini-2.5-flash"],
    )
    defaults.update(overrides)
    return ServerHealthCollector(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollect:
    """ServerHealthCollector.collect() returns a valid snapshot."""

    @patch("server.health.psutil")
    def test_collect_returns_valid_snapshot(self, mock_psutil):
        mock_psutil.cpu_percent.return_value = 15.5
        mock_psutil.virtual_memory.return_value = MagicMock(percent=42.3)

        collector = _make_collector()
        snap = collector.collect()

        assert isinstance(snap, ServerHealthSnapshot)
        assert snap.cpu_percent == 15.5
        assert snap.memory_percent == 42.3
        assert snap.active_sessions == 2  # 2 loaded sessions
        assert snap.active_agents == 6  # 2 loaded * 3 clients each
        assert snap.uptime_seconds >= 59.0  # started ~60s ago
        assert snap.available_providers == ["google_genai"]
        assert snap.available_models == ["gemini-2.5-flash"]
        assert snap.tags == ["local"]

    @patch("server.health.psutil")
    def test_uptime_increases(self, mock_psutil):
        mock_psutil.cpu_percent.return_value = 0.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=0.0)

        start = time.monotonic()
        collector = _make_collector(start_time=start)

        snap1 = collector.collect()
        time.sleep(0.05)
        snap2 = collector.collect()

        assert snap2.uptime_seconds > snap1.uptime_seconds
