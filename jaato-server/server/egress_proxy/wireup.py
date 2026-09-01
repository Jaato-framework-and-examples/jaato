"""Daemon wire-up glue for the egress proxy (Phase 5 §5.11e).

The daemon is a single process, so a process-wide :class:`EgressProxyManager`
singleton (keyed by ``session_id``) is the least-invasive integration: the
session-spawn path calls :func:`egress_env_for_session` to (opt-in) start a
proxy and get the ``HTTPS_PROXY`` env to hand the runner, and the
session-teardown path calls :func:`egress_teardown`.  No manager ownership has
to be threaded through ``SessionManager`` / ``JaatoServer``.

Everything here is **opt-in**: with no ``egress_allowlist`` configured,
:func:`egress_env_for_session` returns an empty env and starts no proxy, so the
session-spawn path is byte-for-byte unchanged.  It is also fail-*safe* by
default — a kernel-level gate that cannot be installed degrades to proxy-only
confinement.  Deployments that need fail-*closed* set
``JAATO_EGRESS_NFT_ENFORCE=strict``, which turns that degradation into an
:class:`~server.egress_proxy.errors.EgressEnforcementError` and denies the
session instead.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .config import AllowlistConfig, validate_allowlist
from .errors import EgressEnforcementError
from .manager import EgressProxyManager
from .nft import EgressNftManager

logger = logging.getLogger(__name__)

_manager: Optional[EgressProxyManager] = None
_nft_manager: Optional[EgressNftManager] = None
_manager_lock = threading.Lock()

# Loopback must bypass the proxy: the runner reaches the proxy itself directly,
# and a CONNECT to a loopback address would be meaningless.  This only tells
# well-behaved clients not to *use* the proxy for loopback — it grants nothing.
# Reachability is decided by the nft gate, which accepts the proxy's own
# address:port and rejects the rest of loopback, IPv6 included (#696).
_NO_PROXY = "localhost,127.0.0.1,::1"


# ``JAATO_EGRESS_NFT_ENFORCE`` postures.  ``off`` is the default: proxy-only
# ("cooperative") confinement.  ``best_effort`` installs the cgroup-nft gate
# but degrades to proxy-only on any failure.  ``strict`` denies instead —
# a session that asked for the gate and did not get one does not start.
_ENFORCE_BEST_EFFORT = frozenset({"1", "true", "yes", "on"})
_ENFORCE_STRICT = frozenset({"strict", "required", "fail-closed", "fail_closed"})


def _nft_enforce_mode() -> str:
    """Return the configured enforcement posture: off / best_effort / strict.

    Hard cgroup-nft egress enforcement is opt-in (it needs `nft` in the
    daemon's sudo NOPASSWD scope + a per-session cgroup), so an unset or
    unrecognized value means ``off`` and leaves behavior unchanged.
    """
    raw = os.environ.get("JAATO_EGRESS_NFT_ENFORCE", "").strip().lower()  # env: opt-in hard cgroup-nft egress enforcement (§5.11d-v2); needs a per-session cgroup + nft.  "1"/"true"/"yes"/"on" = best-effort, "strict" = deny the session when the gate cannot be installed
    if raw in _ENFORCE_STRICT:
        return "strict"
    if raw in _ENFORCE_BEST_EFFORT:
        return "best_effort"
    return "off"


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
    """Gate the session's cgroup to the proxy at netfilter.  Opt-in.

    Under the default ``best_effort`` posture this is guarded so it can never
    break a session — a missing cgroup, missing `nft`, or an nft error
    degrades to proxy-only + a log.  Under ``strict`` the same conditions
    raise :class:`EgressEnforcementError` so the session is denied rather than
    started with the gate silently absent.
    """
    mode = _nft_enforce_mode()
    if mode == "off":
        return
    reason = None
    if not EgressNftManager.nft_available():
        reason = "`nft` not found"
    else:
        try:
            parsed = urlparse(proxy_url)
            port, host = parsed.port, parsed.hostname
            if not port or not host:
                reason = f"unusable proxy url {proxy_url!r}"
            elif not get_nft_manager().install(
                    session_id, _session_cgroup_path(session_id), int(port),
                    proxy_host=host):
                reason = "nft ruleset could not be installed"
        except Exception as exc:  # defensive: never block spawn (best-effort)
            reason = f"{type(exc).__name__}: {exc}"
            logger.debug("egress nft enforcement raised for session %s",
                         session_id, exc_info=True)
    if reason is None:
        return
    if mode == "strict":
        raise EgressEnforcementError(
            f"egress: hard enforcement required (JAATO_EGRESS_NFT_ENFORCE="
            f"strict) but unavailable for session {session_id}: {reason}")
    logger.warning("egress: hard enforcement unavailable for session %s (%s) "
                   "— running proxy-only", session_id, reason)


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

    Raises :class:`EgressEnforcementError` only under
    ``JAATO_EGRESS_NFT_ENFORCE=strict``, when the kernel-level gate was
    required but could not be installed; the proxy is torn down first, and the
    caller is expected to fail the session rather than start it unconfined.
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
    try:
        _maybe_enforce_nft(session_id, url)
    except EgressEnforcementError:
        # Strict posture: deny.  Drop the proxy we just started so the failed
        # spawn leaves no listener behind, then let the caller fail the session.
        egress_teardown(session_id)
        raise
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
