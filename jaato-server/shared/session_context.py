"""Thread-safe session context for plugin access.

Provides ``ContextVar``-based mechanisms for plugins to access the
current session and its environment without storing state on ``self``.
The session wiring code in ``JaatoSession`` sets the context var
before tool execution; plugins read it via ``get_current_session()``.

This eliminates an entire class of bugs where a plugin's
``set_session()`` stores the session on the shared plugin instance,
causing cross-session or cross-subagent data leakage.

Session environment
~~~~~~~~~~~~~~~~~~~

``get_session_env(key)`` reads environment variables from the
session-scoped ``ContextVar`` first, then falls back to ``os.environ``.
This avoids the race condition where concurrent sessions clobber each
other's values in the global ``os.environ`` dict.

``JaatoServer._with_session_env()`` sets the contextvar alongside
``os.environ`` (the latter is still needed for third-party code that
reads ``os.environ`` directly).  Because Python 3.12+
``ThreadPoolExecutor`` copies context to worker threads, parallel tool
execution sees the correct session env automatically.

Usage in plugins::

    from shared.session_context import get_current_session, get_session_env

    class MyPlugin:
        def _do_something(self):
            session = get_current_session()
            runtime = session._runtime
            ...

        def _read_token(self):
            token = get_session_env("GITHUB_TOKEN")
            ...

Usage in session wiring (already handled by JaatoSession)::

    from shared.session_context import set_current_session

    set_current_session(self)  # before tool execution
"""

import contextvars
import os
from contextvars import ContextVar, Token
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .jaato_session import JaatoSession

_current_session: ContextVar['JaatoSession'] = ContextVar('current_session')
_session_env: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    'session_env', default=None,
)
# Per-task workspace identity (server 0.6.68+).  Replaces the previous
# ``_in_workspace()`` os.environ mutation pattern, which clobbered values
# across concurrent overlapping sessions.  ContextVar is asyncio-task /
# thread-local — every concurrent session sees its own value
# regardless of timing of context-manager enter/exit.
_workspace_root: ContextVar[Optional[str]] = ContextVar(
    'workspace_root', default=None,
)
_config_root: ContextVar[Optional[str]] = ContextVar(
    'config_root', default=None,
)


def set_current_session(session: 'JaatoSession') -> None:
    """Set the current session for this thread/context.

    Called by JaatoSession during configure() and before tool execution
    in worker threads. Plugins should never call this.
    """
    _current_session.set(session)


def get_current_session() -> 'JaatoSession':
    """Get the current session for this thread/context.

    Raises:
        LookupError: If no session has been set in this context.
    """
    return _current_session.get()


# ── Session-scoped environment ──────────────────────────────────────────

def set_session_env(env: Dict[str, str]) -> None:
    """Set the session-scoped environment dict for this context.

    Called by ``JaatoServer._with_session_env()`` on entry.  Plugins
    should never call this directly.
    """
    _session_env.set(env)


def clear_session_env() -> None:
    """Clear the session-scoped environment for this context.

    Called by ``JaatoServer._with_session_env()`` on exit.
    """
    _session_env.set(None)


def get_session_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable, preferring the session-scoped value.

    Lookup order:

    1. Session-scoped env (``ContextVar``, set by ``_with_session_env()``).
    2. ``os.environ`` (global process environment).

    This avoids the race where concurrent sessions clobber each other's
    values in the global ``os.environ`` dict.
    """
    env = _session_env.get()
    if env is not None and key in env:
        return env[key]
    return os.environ.get(key, default)


# ── Per-task workspace identity (server 0.6.68+) ────────────────────────
#
# Use these helpers (NOT ``os.environ.get('JAATO_WORKSPACE_ROOT')``) for
# any session-time read of the active workspace / config-root.  The
# helpers fall back to ``os.environ`` so callers that run outside a
# session context (daemon startup, tests, CLI) still work.
#
# Pre-0.6.68 the active workspace was carried in ``os.environ`` —
# process-global state — and concurrent overlapping sessions clobbered
# each other.  Now ``JaatoServer._in_workspace()`` sets a per-task
# ``ContextVar`` instead, race-free across asyncio tasks and threads.


def set_workspace_root(value: Optional[str]) -> Token:
    """Set the per-task workspace root.  Returns a token for ``reset_workspace_root``."""
    return _workspace_root.set(value)


def reset_workspace_root(token: Token) -> None:
    """Reset the per-task workspace root to its previous value."""
    _workspace_root.reset(token)


def get_workspace_root(default: Optional[str] = None) -> Optional[str]:
    """Read the active workspace root.

    Lookup order:

    1. Per-task ``ContextVar`` (set by ``JaatoServer._in_workspace()``).
       Race-free across concurrent sessions.
    2. ``os.environ['JAATO_WORKSPACE_ROOT']`` for daemon-wide /
       startup-time / pre-session-context callers.
    3. ``default``.
    """
    value = _workspace_root.get()
    if value is not None:
        return value
    return os.environ.get('JAATO_WORKSPACE_ROOT', default)  # env: internal — set by the framework per session: workspace root; read via session_context helpers


def set_config_root(value: Optional[str]) -> Token:
    """Set the per-task config root.  Returns a token for ``reset_config_root``."""
    return _config_root.set(value)


def reset_config_root(token: Token) -> None:
    """Reset the per-task config root to its previous value."""
    _config_root.reset(token)


def get_config_root(default: Optional[str] = None) -> Optional[str]:
    """Read the active config root.

    Lookup order:

    1. Per-task ``ContextVar`` (set by ``JaatoServer._in_workspace()``).
       Race-free across concurrent sessions.
    2. ``os.environ['JAATO_CONFIG_ROOT']`` for daemon-wide /
       startup-time / pre-session-context callers.
    3. ``default``.
    """
    value = _config_root.get()
    if value is not None:
        return value
    return os.environ.get('JAATO_CONFIG_ROOT', default)  # env: internal — set by the framework per session: config root; read via session_context helpers


# ── Session-bootstrap isolation (server 0.6.71+) ────────────────────────
#
# The 0.6.68 fix established per-task ContextVar storage for
# ``_workspace_root`` / ``_config_root`` / ``_session_env`` /
# ``_current_session`` — race-free across CONCURRENT overlapping
# sessions.  But ContextVars are INHERITED by spawned tasks: a new
# asyncio task / sync function called from within an existing task
# starts with the parent's ContextVar values.  On a long-lived daemon
# that previously ran (e.g.) cascade work for workspace A, the parent
# task's ContextVars carry A's values.  A subsequent session-bootstrap
# for workspace B then starts with ``_workspace_root=A`` — leaking A's
# profile path into B's plugin initialisation.
#
# The fix: wrap every session-bootstrap entry point in
# ``contextvars.Context().run(...)`` — an empty Context where every
# ContextVar starts at its declared default.  Inside the fresh context,
# the session's own ``_in_workspace()`` sets the correct values cleanly,
# and they cannot bleed back to the parent task because ``Context.run()``
# snapshots-and-discards the inner mutations.  Each session-bootstrap
# starts from a known-clean state regardless of the daemon's task
# history.
#
# Bootstrap state that must SURVIVE the isolation lives on instance
# attributes (e.g. ``SessionManager._sessions``) — not in ContextVars
# — so the isolation is transparent to the rest of the daemon.

T = TypeVar("T")


_ISOLATED_ENV_VARS: Tuple[str, ...] = (
    "JAATO_WORKSPACE_ROOT",
    "JAATO_CONFIG_ROOT",
)


def run_in_fresh_session_context(
    fn: Callable[..., T], *args: Any, **kwargs: Any,
) -> T:
    """Run ``fn(*args, **kwargs)`` in a fresh ``contextvars.Context``
    AND with the workspace/config-root ``os.environ`` keys temporarily
    cleared.

    Used by every session-bootstrap entry point (``create_session``,
    ``_load_session``, ``run_ephemeral_session``, the standalone WS
    server's bootstrap) so each session starts from a known-clean
    state — see this module's "Session-bootstrap isolation" block
    above for the full rationale.

    **Why os.environ also gets cleared (server 0.6.72+).**

    ``contextvars.Context.run()`` isolates ContextVars but NOT
    ``os.environ`` (process-global).  ``get_workspace_root()`` /
    ``get_config_root()`` fall through to ``os.environ`` when the
    ContextVar is unset.  Inside a fresh Context, the ContextVar IS
    unset (default), so the fallback hits ``os.environ`` — which on a
    long-lived daemon may carry residue from a prior session's
    ``_in_workspace()`` mutation that didn't fully reset (race-y under
    concurrent tasks; see 0.6.68's documentation of the original
    cross-client leak).

    Empirically: v50 cascade for workspace A ran cleanly post-0.6.71,
    but a subsequent harness session-create for workspace B saw
    ``os.environ['JAATO_WORKSPACE_ROOT']=A`` leaked from cascade
    work — fresh ContextVar fell through to that leaked env var,
    profile discovery scanned A's ``.jaato/profiles/`` instead of
    B's, bootstrap stalled.  Restarting the daemon cleared it.

    Now we proactively isolate both layers: fresh Context (for
    jaato-side ContextVar readers) AND ``os.environ`` save-and-clear
    (for jaato-side fallback readers + third-party code that reads
    ``os.environ`` directly).  The save/restore is balanced via a
    try/finally so concurrent calls don't corrupt each other's
    ``os.environ`` state — at most one bootstrap runs at a time per
    OS thread, so there's no race within the helper itself.

    Notes:
        ``Context.run()`` is synchronous.  All current callers are
        sync; if an async session-bootstrap entry point is added
        later, wrap each call site in
        ``loop.run_in_executor(None, lambda: run_in_fresh_session_context(...))``.

        The ``os.environ`` clear is scoped to the workspace/config-root
        keys (``_ISOLATED_ENV_VARS``).  Other env vars (e.g. provider
        API keys) stay untouched — they're set at daemon startup or
        by per-session ``.env`` loading inside the bootstrap, both of
        which compose correctly with the isolation.
    """
    saved: Dict[str, Optional[str]] = {
        key: os.environ.pop(key, None) for key in _ISOLATED_ENV_VARS
    }
    try:
        return contextvars.Context().run(fn, *args, **kwargs)
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
