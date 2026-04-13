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

import os
from contextvars import ContextVar
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .jaato_session import JaatoSession

_current_session: ContextVar['JaatoSession'] = ContextVar('current_session')
_session_env: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    'session_env', default=None,
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
