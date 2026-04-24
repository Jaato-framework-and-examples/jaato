"""Tests for SessionManager routing primitives consumed by daemon
extensions (reactor rules, webhook handlers).

These primitives let an extension address a loaded session by ID rather
than just acting on the session whose event triggered the extension.
The premium reactor framework's cross-session ``inject_prompt`` action
sits directly on top of ``inject_prompt_to_session``.
"""

from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from .session_manager import Session, SessionManager


class _FakeJaatoSession:
    """Captures inject_prompt calls so tests can assert on them."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Optional[str], Optional[Any]]] = []

    def inject_prompt(
        self,
        text: str,
        source_id: Optional[str] = None,
        source_type: Optional[Any] = None,
    ) -> None:
        self.calls.append((text, source_id, source_type))


class _FakeJaatoServer:
    """Just enough of JaatoServer's surface for the routing test.

    ``raise_on_get_session=True`` simulates the "session record exists
    but underlying JaatoSession not yet initialised" case (the real
    JaatoServer.get_session raises RuntimeError in that state).
    """

    def __init__(self, raise_on_get_session: bool = False) -> None:
        self.session = _FakeJaatoSession()
        self._raise = raise_on_get_session

    def get_session(self) -> _FakeJaatoSession:
        if self._raise:
            raise RuntimeError("No active session")
        return self.session


def _make_session(session_id: str, server: _FakeJaatoServer) -> Session:
    return Session(
        session_id=session_id,
        name=session_id,
        server=server,  # type: ignore[arg-type]
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_manager_with_session(
    session_id: str = "sess_1",
    raise_on_get_session: bool = False,
) -> Tuple[SessionManager, _FakeJaatoServer]:
    manager = SessionManager()
    server = _FakeJaatoServer(raise_on_get_session=raise_on_get_session)
    session = _make_session(session_id, server)
    # Bypass the full create flow — we just want a session record present
    # so the routing helper has something to look up.
    manager._sessions[session_id] = session
    return manager, server


class TestInjectPromptToSession:
    """Cross-session prompt routing: deliver to a session by ID."""

    def test_returns_true_and_forwards_when_session_loaded(self):
        manager, server = _make_manager_with_session("sess_1")

        ok = manager.inject_prompt_to_session(
            "sess_1",
            "hello from reactor",
            source_id="reactor",
        )

        assert ok is True
        assert server.session.calls == [
            ("hello from reactor", "reactor", None)
        ]

    def test_returns_false_when_session_not_loaded(self):
        manager = SessionManager()

        ok = manager.inject_prompt_to_session(
            "sess_missing",
            "no one is listening",
        )

        assert ok is False

    def test_returns_false_when_jaato_session_not_initialised(self):
        """Session record exists but the underlying JaatoSession isn't
        ready yet — routing must fail cleanly rather than crash."""
        manager, server = _make_manager_with_session(
            "sess_pending", raise_on_get_session=True
        )

        ok = manager.inject_prompt_to_session("sess_pending", "too early")

        assert ok is False
        assert server.session.calls == []  # never reached

    def test_passes_source_metadata_through(self):
        """source_id and source_type must reach inject_prompt verbatim
        so priority-based queueing works for cross-session injects too."""
        from shared.message_queue import SourceType

        manager, server = _make_manager_with_session("sess_2")

        ok = manager.inject_prompt_to_session(
            "sess_2",
            "system event",
            source_id="webhook:github",
            source_type=SourceType.EVENT,
        )

        assert ok is True
        assert server.session.calls == [
            ("system event", "webhook:github", SourceType.EVENT)
        ]

    def test_default_source_metadata_omitted(self):
        """When the caller doesn't supply source_id / source_type, the
        helper passes None and inject_prompt's own defaults take over."""
        manager, server = _make_manager_with_session("sess_3")

        manager.inject_prompt_to_session("sess_3", "plain")

        assert server.session.calls == [("plain", None, None)]
