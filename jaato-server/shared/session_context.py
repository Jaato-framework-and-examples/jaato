"""Thread-safe session context for plugin access.

Provides a ``ContextVar``-based mechanism for plugins to access the
current session without storing it on ``self``.  The session wiring
code in ``JaatoSession`` sets the context var before tool execution;
plugins read it via ``get_current_session()``.

This eliminates an entire class of bugs where a plugin's
``set_session()`` stores the session on the shared plugin instance,
causing cross-session or cross-subagent data leakage.

Usage in plugins::

    from shared.session_context import get_current_session

    class MyPlugin:
        def _do_something(self):
            session = get_current_session()
            runtime = session._runtime
            ...

Usage in session wiring (already handled by JaatoSession)::

    from shared.session_context import set_current_session

    set_current_session(self)  # before tool execution
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .jaato_session import JaatoSession

_current_session: ContextVar['JaatoSession'] = ContextVar('current_session')


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
