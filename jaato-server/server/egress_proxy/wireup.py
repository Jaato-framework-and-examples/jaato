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
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .config import AllowlistConfig, validate_allowlist
from .manager import EgressProxyManager
from .nft import EgressNftManager

logger = logging.getLogger(__name__)

_manager: Optional[EgressProxyManager] = None
_nft_manager: Optional[EgressNftManager] = None
_manager_lock = threading.Lock()

# Loopback must bypass the proxy (the runner reaches the proxy itself, and any
# daemon-local service, directly).
_NO_PROXY = "localhost,127.0.0.1,::1"


def _nft_enforce_enabled() -> bool:
    """Hard cgroup-nft egress enforcement is opt-in (needs `nft` in the daemon's
    sudo NOPASSWD scope + a per-session cgroup).  Off by default → proxy-only
    ("cooperative") confinement, unchanged behavior."""
    return os.environ.get("JAATO_EGRESS_NFT_ENFORCE", "").strip().lower() in (  # env: opt-in hard cgroup-nft egress enforcement (§5.11d-v2); needs a per-session cgroup + nft
        "1", "true", "yes", "on")


def _session_cgroup_path(session_id: str) -> str:
    root = os.environ.get("JAATO_CGROUPS_ROOT", "/sys/fs/cgroup/jaato")  # env: parent cgroup-v2 dir for per-session cgroups (must exist + be delegated)
    return os.path.join(root, f"jaato-ws-{session_id}")


def get_nft_manager() -> EgressNftManager:
    global _nft_manager
    with _manager_lock:
        if _nft_manager is None:
            _nft_manager = EgressNftManager()
        return _nft_manager


def _maybe_enforce_nft(session_id: str, proxy_url: str) -> None:
    """Best-effort hard enforcement: gate the session's cgroup to the proxy at
    netfilter.  Opt-in; guarded so it can never break a session — a missing
    cgroup, missing `nft`, or an nft error degrades to proxy-only + a log."""
    if not _nft_enforce_enabled():
        return
    if not EgressNftManager.nft_available():
        logger.warning("egress: JAATO_EGRESS_NFT_ENFORCE set but `nft` not "
                       "found — session %s runs proxy-only", session_id)
        return
    try:
        port = urlparse(proxy_url).port
        if not port:
            return
        get_nft_manager().install(session_id, _session_cgroup_path(session_id), int(port))
    except Exception:  # pragma: no cover - defensive: never block spawn
        logger.warning("egress nft enforcement failed for session %s "
                       "(proxy-only)", session_id, exc_info=True)


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
    _maybe_enforce_nft(session_id, url)
    env = {
        "HTTPS_PROXY": url, "https_proxy": url,
        "HTTP_PROXY": url, "http_proxy": url,
        "NO_PROXY": _NO_PROXY, "no_proxy": _NO_PROXY,
    }
    logger.info("egress: session %s pinned to %s (allowlist: %s)",
                session_id, url, cfg.allowed_hosts)
    return env, []


def egress_teardown(session_id: str) -> None:
    """Stop the session's egress proxy + remove its nft table.  Idempotent."""
    if _nft_manager is not None:
        try:
            _nft_manager.remove(session_id)
        except Exception:  # pragma: no cover - defensive
            logger.warning("egress nft teardown failed for %s", session_id,
                           exc_info=True)
    if _manager is not None:
        _manager.stop_proxy_for_session(session_id)


def shutdown_all() -> None:
    """Stop every egress proxy + remove all nft tables (daemon shutdown)."""
    if _nft_manager is not None:
        _nft_manager.shutdown()
    if _manager is not None:
        _manager.shutdown()
