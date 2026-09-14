"""Session-scoped environment reads for plugins.

An out-of-tree plugin — a distribution declaring
``[project.entry-points."jaato.plugins"]`` — builds against ``jaato_sdk``
and never needs to import the server package.  Reading a **credential**
was the one hole in that surface: the session-scoped read lived in
``shared.session_context`` (jaato-server), so a third-party connector
either took a hard dependency on the server or wrote
``os.environ.get(...)`` and shipped a cross-tenant bug (issue #918).

**Do not read ``os.environ`` directly for a per-session value.**  This is
not a tidiness preference.  ``JaatoServer._with_session_env()`` overlays
each session's ``env:`` map onto the daemon's process environment for the
duration of a turn, so on a daemon serving two sessions concurrently a
plain ``os.environ`` read can return **another session's token** —
non-deterministically, with no error and no log line.  For an
OAuth-bearing connector that is the difference between two tenants'
credentials staying separate and not.

Use :func:`get_session_env` instead::

    from jaato_sdk.session_env import get_session_env

    class GraphPlugin:
        def _token(self) -> str | None:
            return get_session_env("GRAPH_CLIENT_SECRET")

It reads the session-scoped :class:`~contextvars.ContextVar` first and
falls back to ``os.environ``, so the same call is correct inside the
daemon and outside it (tests, a CLI, a bare script) with no soft-import
dance.

**One ContextVar, one implementation.**  ``shared.session_context``
imports these names rather than defining its own, so the var the daemon
SETS is the var a plugin READS.  A second copy in the SDK would read
empty, fall through to ``os.environ``, and reintroduce the bug in a form
that looks fixed — which is why the server-side module re-exports instead
of mirroring.

Deliberately **not** exported here: ``get_current_session()``.  It hands
back a ``JaatoSession``, which is server-side by nature, and a plugin
reaching into ``session._runtime`` is not something to make easier from
out of tree.
"""

import os
from contextvars import ContextVar
from typing import Dict, Optional

__all__ = [
    "get_session_env",
    "set_session_env",
    "clear_session_env",
]


#: The session-scoped environment overlay.  Set by
#: ``JaatoServer._with_session_env()`` on the daemon side; ``None`` means
#: "no session context" (a CLI, a test, daemon startup), in which case
#: :func:`get_session_env` is a plain ``os.environ`` read.
#:
#: Module-private by convention: the ONE object identity that makes this
#: work is the reason ``shared.session_context`` imports this module
#: rather than declaring a var of its own.
_session_env: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "session_env", default=None,
)


def set_session_env(env: Dict[str, str]) -> None:
    """Set the session-scoped environment dict for this context.

    Called by ``JaatoServer._with_session_env()`` on entry.  **Plugins
    should never call this** — a plugin that sets the overlay is
    rewriting the environment of whatever session happens to be running
    in this context.
    """
    _session_env.set(env)


def clear_session_env() -> None:
    """Clear the session-scoped environment for this context.

    Called by ``JaatoServer._with_session_env()`` on exit.  Plugins
    should never call this.
    """
    _session_env.set(None)


def get_session_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable, preferring the session-scoped value.

    Lookup order:

    1. Session-scoped env (the ``ContextVar`` set by
       ``JaatoServer._with_session_env()``).
    2. ``os.environ`` (global process environment).
    3. *default*.

    **Prefer this over ``os.environ.get`` in any plugin.**  The daemon
    overlays each session's ``env:`` map onto the process environment for
    the duration of a turn, so concurrent sessions clobber each other's
    values there; the ContextVar is per-task/per-thread and does not.
    Reading the global dict for a credential can hand you another
    session's token, silently.  Python 3.12+ ``ThreadPoolExecutor``
    copies the context into worker threads, so parallel tool execution
    sees the right session env automatically.

    Args:
        key: Environment variable name.
        default: Returned when *key* is in neither layer.

    Returns:
        The session-scoped value, else the process value, else *default*.
    """
    env = _session_env.get()
    if env is not None and key in env:
        return env[key]
    return os.environ.get(key, default)
