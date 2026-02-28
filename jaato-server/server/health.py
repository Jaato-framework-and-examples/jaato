"""Server health collection for gossip heartbeats.

Provides a periodic health snapshot (CPU, memory, session/agent counts) that
the PeerRegistry includes in outbound heartbeat messages.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import psutil

if TYPE_CHECKING:
    from server.session_manager import SessionManager


@dataclass
class ServerHealthSnapshot:
    """Point-in-time health metrics for this server."""

    cpu_percent: float
    memory_percent: float
    active_sessions: int
    active_agents: int
    uptime_seconds: float
    available_providers: List[str]
    available_models: List[str]
    tags: List[str]


class ServerHealthCollector:
    """Collects health metrics for inclusion in gossip heartbeats.

    Constructed once at daemon startup. The ``collect()`` method is called
    every heartbeat interval (~5 s) by the PeerRegistry.

    Args:
        session_manager: SessionManager instance for session/agent counts.
        server_id: Stable UUID for this server.
        server_name: Human-readable name from servers.json.
        tags: Static tags from servers.json config.
        start_time: ``time.monotonic()`` captured at daemon startup.
        available_providers: Provider names with valid credentials.
        available_models: Models this server can serve.
    """

    def __init__(
        self,
        session_manager: "SessionManager",
        server_id: str,
        server_name: str,
        tags: List[str],
        start_time: float,
        available_providers: List[str],
        available_models: List[str],
    ) -> None:
        self._session_manager = session_manager
        self.server_id = server_id
        self.server_name = server_name
        self._tags = list(tags)
        self._start_time = start_time
        self._available_providers = list(available_providers)
        self._available_models = list(available_models)

    def collect(self) -> ServerHealthSnapshot:
        """Collect a health snapshot.

        Uses ``psutil.cpu_percent(interval=None)`` which returns the cached
        value from the last call — appropriate for the 5 s heartbeat cadence.
        """
        sessions = self._session_manager.list_sessions()
        active_sessions = sum(1 for s in sessions if s.is_loaded)
        active_agents = sum(s.client_count for s in sessions if s.is_loaded)

        return ServerHealthSnapshot(
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=psutil.virtual_memory().percent,
            active_sessions=active_sessions,
            active_agents=active_agents,
            uptime_seconds=time.monotonic() - self._start_time,
            available_providers=list(self._available_providers),
            available_models=list(self._available_models),
            tags=list(self._tags),
        )
