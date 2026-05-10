"""Tests for SessionManager routing primitives consumed by daemon
extensions (reactor rules, webhook handlers).

These primitives let an extension address a loaded session by ID rather
than just acting on the session whose event triggered the extension.
The premium reactor framework's cross-session ``inject_prompt`` action
sits directly on top of ``inject_prompt_to_session``.

Phase 3 §7c step 6.6.3.6: ``inject_prompt_to_session`` was migrated to
forward via the runner-RPC ``session.inject_prompt`` instead of
reaching into the daemon-side session.  Tests updated accordingly.
"""

from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from .session_manager import Session, SessionManager


class _FakeRunnerRPC:
    """Captures session_inject_prompt_threadsafe calls.

    Phase 3 §7c step 6.6.3.6: the daemon's
    ``inject_prompt_to_session`` now forwards via the runner-side
    RPC.  This fake captures the wrapper invocation directly; the
    runner-side handler's wire shape is exercised in
    ``test_session_inject_prompt_rpc.py``.
    """

    def __init__(self, raise_on_call: bool = False) -> None:
        # Each tuple: (text, source_id, source_type-string)
        self.calls: List[Tuple[str, Optional[str], Optional[str]]] = []
        self._raise = raise_on_call

    def session_inject_prompt_threadsafe(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if self._raise:
            raise RuntimeError("RPC unavailable")
        self.calls.append((text, source_id, source_type))


class _FakeJaatoServer:
    """Just enough of JaatoServer's surface for the routing test.

    ``raise_on_get_session=True`` simulates the "session record exists
    but underlying RPC not yet initialised" case — i.e.
    ``self._runner_rpc is None`` post-§7c-step-6.6.3.6 (the
    real JaatoServer's runner_rpc is None until spawn completes).
    """

    def __init__(self, raise_on_get_session: bool = False) -> None:
        if raise_on_get_session:
            self._runner_rpc = None
        else:
            self._runner_rpc = _FakeRunnerRPC()

    @property
    def session(self):
        """Test-side accessor for the runner-RPC's captured calls.

        Returns an object exposing ``calls`` for assertion shape
        compatibility with the pre-§7c-step-6.6.3.6 test fakes.
        """
        return self._runner_rpc if self._runner_rpc is not None else _FakeRunnerRPC()


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
            # source_type=None is preserved across the wire (the
            # runner-side handler applies its own
            # SourceType.USER default downstream).
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
        so priority-based queueing works for cross-session injects too.

        Phase 3 §7c step 6.6.3.6: source_type crosses the runner-RPC
        wire as its lowercase string value (the SourceType enum
        is reconstructed runner-side per the §7c step 6.1 (3/3)
        ``session.inject_prompt`` handler at commit 14e57709)."""
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
            ("system event", "webhook:github", "event")
        ]

    def test_default_source_metadata_omitted(self):
        """When the caller doesn't supply source_id / source_type, the
        helper passes None and inject_prompt's own defaults take over."""
        manager, server = _make_manager_with_session("sess_3")

        manager.inject_prompt_to_session("sess_3", "plain")

        assert server.session.calls == [("plain", None, None)]
