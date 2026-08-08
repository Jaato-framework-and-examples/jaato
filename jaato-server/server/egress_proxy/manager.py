"""Per-session egress proxy lifecycle (Phase 5 §5.11c).

The daemon owns one :class:`EgressProxyManager`.  When a session opts into an
egress allowlist, the manager spawns a :class:`ConnectAllowlistProxy` bound to
a free loopback port and hands back the ``http://127.0.0.1:NNNN`` URL that the
runner's ``HTTPS_PROXY`` should point at.  On session teardown the proxy is
stopped.  All operations are idempotent and thread-safe.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

from .config import AllowlistConfig
from .proxy import ConnectAllowlistProxy

logger = logging.getLogger(__name__)


class EgressProxyManager:
    """Maps ``session_id`` → running proxy; spawns/tears down per session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proxies: Dict[str, ConnectAllowlistProxy] = {}

    def start_proxy_for_session(
        self, session_id: str, allowlist: AllowlistConfig,
    ) -> str:
        """Spawn (or reuse) a proxy for ``session_id``; return its URL.

        Idempotent: if a proxy is already running for the session it is
        returned unchanged (the allowlist is fixed at first start — a policy
        change requires a teardown + restart).
        """
        with self._lock:
            existing = self._proxies.get(session_id)
            if existing is not None and existing.port is not None:
                return existing.proxy_url()
            proxy = ConnectAllowlistProxy(allowlist)
            proxy.start()
            self._proxies[session_id] = proxy
            logger.info("egress proxy started for session %s at %s",
                        session_id, proxy.proxy_url())
            return proxy.proxy_url()

    def stop_proxy_for_session(self, session_id: str) -> None:
        """Stop and forget the session's proxy.  Idempotent."""
        with self._lock:
            proxy = self._proxies.pop(session_id, None)
        if proxy is not None:
            proxy.stop()
            logger.info("egress proxy stopped for session %s", session_id)

    def get_proxy_url(self, session_id: str) -> Optional[str]:
        with self._lock:
            proxy = self._proxies.get(session_id)
        if proxy is None or proxy.port is None:
            return None
        return proxy.proxy_url()

    def get_telemetry(self) -> Dict[str, object]:
        """Aggregate per-session counters (for the daemon telemetry surface)."""
        with self._lock:
            proxies = dict(self._proxies)
        per_session = {sid: p.stats.snapshot() for sid, p in proxies.items()}
        totals = {"connections_total": 0, "allowed_total": 0,
                  "denied_total": 0, "malformed_total": 0}
        for snap in per_session.values():
            for k in totals:
                totals[k] += snap[k]
        return {
            "active_proxies": len(proxies),
            "totals": totals,
            "per_session": per_session,
        }

    def shutdown(self) -> None:
        """Stop every proxy (daemon shutdown)."""
        with self._lock:
            proxies = list(self._proxies.values())
            self._proxies.clear()
        for p in proxies:
            p.stop()
