"""Session persistence collaborator extracted from ``JaatoSession``.

``SessionPersistence`` owns everything to do with saving/restoring a
conversation through a ``SessionPlugin``: the plugin + config references,
the plugin-wiring done at attach time, the save/resume/list/delete
operations, and the ``SessionState`` (de)serialization.

It holds a back-reference to its owning :class:`JaatoSession` and reaches
through it for the live conversation data it needs to snapshot (history,
turn accounting, attached session-state) and to apply on restore
(``reset_session``, ``set_session_state``). This bidirectional coupling
is intentional — persistence is a tightly-bound aspect of the session —
but isolating it here keeps the (de)serialization logic out of the
9k-line session class and independently testable.

``JaatoSession`` exposes ``_session_plugin`` / ``_session_config`` as
read-only properties delegating to this collaborator, so existing readers
(e.g. ``jaato_client``, ``refresh_tools``) are unaffected.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from .plugins.session import (
    SessionConfig,
    SessionInfo,
    SessionPlugin,
    SessionState,
)

if TYPE_CHECKING:
    from .jaato_session import JaatoSession


class SessionPersistence:
    """Manages a session's ``SessionPlugin`` and its save/restore flow.

    Attributes:
        plugin: The attached ``SessionPlugin`` (persistence backend), or
            ``None`` when no plugin is configured.
        config: The ``SessionConfig`` paired with ``plugin``, or ``None``.
    """

    def __init__(self, session: "JaatoSession") -> None:
        self._session = session
        self.plugin: Optional[SessionPlugin] = None
        self.config: Optional[SessionConfig] = None

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------
    def set_plugin(
        self,
        plugin: SessionPlugin,
        config: Optional[SessionConfig] = None,
    ) -> None:
        """Attach a session plugin and wire its commands/executors/tools.

        Registers any user commands, executors, and tool schemas the
        plugin exposes onto the owning session, then auto-resumes the last
        session when ``config.auto_resume_last`` is set.
        """
        session = self._session
        self.plugin = plugin
        self.config = config or SessionConfig()

        if hasattr(plugin, 'set_session'):
            plugin.set_session(session)

        if hasattr(plugin, 'get_user_commands'):
            for cmd in plugin.get_user_commands():
                session._user_commands[cmd.name] = cmd

        if hasattr(plugin, 'get_executors') and session._executor:
            for name, fn in plugin.get_executors().items():
                session._executor.register(name, fn)

        if hasattr(plugin, 'get_tool_schemas'):
            session_schemas = plugin.get_tool_schemas()
            if session_schemas:
                current_tools = list(session._tools) if session._tools else []
                current_tools.extend(session_schemas)
                session._tools = current_tools

        if self.config.auto_resume_last:
            state = self.plugin.on_session_start(self.config)
            if state:
                self.restore_state(state)

    def remove_plugin(self) -> None:
        """Detach and shut down the current session plugin."""
        if self.plugin:
            self.plugin.shutdown()
        self.plugin = None
        self.config = None

    # ------------------------------------------------------------------
    # Save / resume / list / delete
    # ------------------------------------------------------------------
    def save(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None,
    ) -> str:
        """Save the current session, returning the saved session id."""
        if not self.plugin:
            raise RuntimeError("No session plugin configured.")

        state = self.build_state(session_id, user_inputs)
        self.plugin.save(state)

        if hasattr(self.plugin, 'set_current_session_id'):
            self.plugin.set_current_session_id(state.session_id)

        return state.session_id

    def resume(self, session_id: str) -> SessionState:
        """Resume a previously saved session."""
        if not self.plugin:
            raise RuntimeError("No session plugin configured.")

        state = self.plugin.load(session_id)
        self.restore_state(state)
        return state

    def list(self) -> List[SessionInfo]:
        """List all available sessions."""
        if not self.plugin:
            raise RuntimeError("No session plugin configured.")
        return self.plugin.list_sessions()

    def delete(self, session_id: str) -> bool:
        """Delete a saved session."""
        if not self.plugin:
            raise RuntimeError("No session plugin configured.")
        return self.plugin.delete(session_id)

    # ------------------------------------------------------------------
    # State (de)serialization
    # ------------------------------------------------------------------
    def build_state(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None,
    ) -> SessionState:
        """Build a ``SessionState`` snapshot from the owning session."""
        session = self._session
        if not session_id:
            if (self.plugin and
                    hasattr(self.plugin, 'get_current_session_id')):
                session_id = self.plugin.get_current_session_id()
            if not session_id:
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        now = datetime.now()
        turn_accounting = session.get_turn_accounting()

        description = None
        if self.plugin and hasattr(self.plugin, '_session_description'):
            description = self.plugin._session_description

        # Snapshot session-attached state at save time.  Invokes every
        # registered provider so the snapshot reflects live values
        # (extension-owned incrementally-mutated structures stay
        # correct without push-on-every-mutation).  Empty dict
        # collapses to None so old persistence files round-trip
        # unchanged when nothing has been attached.
        attached_state = session.get_all_session_state()
        # ``profile_name`` not threaded through this construction path
        # — this branch fires on the runner-side standalone-client
        # save flow where the profile binding lives on the daemon-side
        # JaatoServer, not the runner-side JaatoSession.  SessionManager's
        # save path (which has access to ``server._profile.name``)
        # populates it correctly.  This path stays None; backward-
        # compat path for non-SessionManager-managed sessions.
        return SessionState(
            session_id=session_id,
            history=session.get_history(),
            created_at=now,
            updated_at=now,
            turn_count=len(turn_accounting),
            turn_accounting=turn_accounting,
            user_inputs=user_inputs or [],
            description=description,
            session_state=attached_state if attached_state else None,
        )

    def restore_state(self, state: SessionState) -> None:
        """Restore the owning session from a ``SessionState``."""
        session = self._session
        session.reset_session(state.history)
        session._turn_accounting = list(state.turn_accounting)
        # Re-attach session-state values from the persisted snapshot.
        # Routes through set_session_state so the JSON-serialisability
        # check fires (defensive: persisted state should already be
        # serialisable, but a corrupted file shouldn't silently
        # inject a non-JSON value into the live container).
        # Consumer hooks fire via the normal session-creation path
        # (when the persisted session is loaded by SessionManager) and
        # can re-register providers / instantiate runtime structures
        # from the restored values.
        if state.session_state:
            for key, value in state.session_state.items():
                session.set_session_state(key, value)

    # ------------------------------------------------------------------
    # Lifecycle notifications
    # ------------------------------------------------------------------
    def notify_turn_complete(self) -> None:
        """Notify the session plugin that a turn completed."""
        if not self.plugin or not self.config:
            return

        state = self.build_state()

        if hasattr(self.plugin, 'increment_turn_count'):
            self.plugin.increment_turn_count()

        self.plugin.on_turn_complete(state, self.config)

    def close(self) -> None:
        """Notify the session plugin that the session is ending."""
        if self.plugin and self.config:
            state = self.build_state()
            self.plugin.on_session_end(state, self.config)
