"""Daemon wire-up glue for the egress proxy (Phase 5 §5.11e).

The daemon is a single process, so a process-wide :class:`EgressProxyManager`
singleton (keyed by ``session_id``) is the least-invasive integration: the
session-spawn path calls :func:`egress_env_for_session` to (opt-in) start a
proxy and get the ``HTTPS_PROXY`` env to hand the runner, and the
session-teardown path calls :func:`egress_teardown`.  No manager ownership has
to be threaded through ``SessionManager`` / ``JaatoServer``.

Everything here is **opt-in and fail-safe**: with no ``egress_allowlist``
configured, :func:`egress_env_for_session` returns an empty env and starts no
proxy, so the session-spawn path is byte-for-byte unchanged.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from .config import AllowlistConfig, validate_allowlist
from .manager import EgressProxyManager

logger = logging.getLogger(__name__)

_manager: Optional[EgressProxyManager] = None
_manager_lock = threading.Lock()

# Loopback must bypass the proxy (the runner reaches the proxy itself, and any
# daemon-local service, directly).
_NO_PROXY = "localhost,127.0.0.1,::1"


def get_manager() -> EgressProxyManager:
    """Return the process-wide egress proxy manager (lazily created)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = EgressProxyManager()
        return _manager


def egress_env_for_session(
    session_id: str, allowlist_data: Any,
) -> Tuple[Dict[str, str], List[str]]:
    """Start a per-session egress proxy iff ``allowlist_data`` is valid.

    Returns ``(env, errors)``:
    - ``allowlist_data`` falsy → ``({}, [])`` and no proxy (the common,
      unconfigured case — caller leaves the session env unchanged).
    - invalid allowlist → ``({}, [error, ...])`` and no proxy (the caller
      should surface the errors; the session runs without egress restriction
      rather than failing to start).
    - valid → proxy started, ``env`` carries HTTPS_PROXY/HTTP_PROXY/NO_PROXY
      (upper + lower case) to merge into the runner's ``session_env``.
    """
    if not allowlist_data:
        return {}, []
    errors = validate_allowlist(allowlist_data)
    if errors:
        logger.warning(
            "egress_allowlist invalid for session %s (running without egress "
            "restriction): %s", session_id, "; ".join(errors))
        return {}, errors
    cfg = AllowlistConfig.from_dict(allowlist_data)
    url = get_manager().start_proxy_for_session(session_id, cfg)
    env = {
        "HTTPS_PROXY": url, "https_proxy": url,
        "HTTP_PROXY": url, "http_proxy": url,
        "NO_PROXY": _NO_PROXY, "no_proxy": _NO_PROXY,
    }
    logger.info("egress: session %s pinned to %s (allowlist: %s)",
                session_id, url, cfg.allowed_hosts)
    return env, []


def egress_teardown(session_id: str) -> None:
    """Stop the session's egress proxy if one was started.  Idempotent."""
    if _manager is None:
        return
    _manager.stop_proxy_for_session(session_id)


def shutdown_all() -> None:
    """Stop every egress proxy (daemon shutdown)."""
    if _manager is None:
        return
    _manager.shutdown()
