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

import pytest

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

    def __init__(self, raise_on_call: bool = False, is_running=None) -> None:
        # Each tuple: (text, source_id, source_type-string)
        self.calls: List[Tuple[str, Optional[str], Optional[str]]] = []
        self._raise = raise_on_call
        # Lazy so a test can flip the server's flag after construction.
        self._is_running = is_running or (lambda: True)

    def session_offer_message_threadsafe(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        require_idle: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        """Stand-in for the runner's ATOMIC queue-or-report.

        The real one decides under ``_delivery_lock`` against the session's
        own ``_is_running``; the daemon no longer decides at all.  Recording
        the call in ``calls`` only on the queued branch mirrors that: a
        ``needs_turn`` answer enqueues NOTHING, so there is nothing to
        record and the daemon must drive instead.

        THE SIGNATURE IS PART OF THE FIXTURE.  ``require_idle`` shipped on the
        real method and not on this one, so every call landed in the caller's
        ``except`` branch: five tests named for the queue/drive decision were
        in fact exercising the delivery-failed path, and passed nothing they
        claimed to.  ``test_the_fake_matches_the_real_signature`` below is the
        guard -- a fixture that drifts from its subject tests the drift.
        """
        if self._raise:
            raise RuntimeError("RPC unavailable")
        if not self._is_running():
            return "needs_turn"
        self.calls.append((text, source_id, source_type))
        return "queued"

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

    def __init__(
        self,
        raise_on_get_session: bool = False,
        model_running: bool = True,
        terminal_reason: Optional[str] = None,
    ) -> None:
        if raise_on_get_session:
            self._runner_rpc = None
        else:
            self._runner_rpc = _FakeRunnerRPC(
                is_running=lambda: self._model_running,
            )
        # Defaults to BUSY so the forwarding tests below exercise the
        # forward path.  An IDLE target is DRIVEN, not injected -- injecting
        # into one queues into a queue with no drainer, which is the bug
        # ``deliver_prompt_to_session`` exists to prevent.
        self._model_running = model_running
        self._terminal_reason = terminal_reason

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
    model_running: bool = True,
    terminal_reason: Optional[str] = None,
) -> Tuple[SessionManager, _FakeJaatoServer]:
    manager = SessionManager()
    server = _FakeJaatoServer(
        raise_on_get_session=raise_on_get_session,
        model_running=model_running,
        terminal_reason=terminal_reason,
    )
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


class TestDeliverPromptToSession:
    """The STATUS-returning form: what the caller is TOLD, not just whether.

    The boolean form cannot distinguish "queued into a live turn" from "the
    target is dead", so a driver that got ``False`` -- or worse, the silent
    ``ok: True`` the SDK used to discard -- had no way to tell a busy peer
    from a gone one.  A caller that cannot tell those apart cannot recover
    from the second.
    """

    def test_idle_target_is_driven_not_injected(self):
        """The black-hole case: injecting into an idle session queues into a
        queue nothing drains, so an idle target must have a turn STARTED."""
        from shared.message_delivery import ACCEPTED

        manager, server = _make_manager_with_session(
            "sess_idle", model_running=False,
        )
        driven: List[Tuple[str, str]] = []
        manager.send_message_to_session = (  # type: ignore[method-assign]
            lambda sid, text: (driven.append((sid, text)), True)[1]
        )

        status = manager.deliver_prompt_to_session("sess_idle", "wake up")

        assert status == ACCEPTED
        assert driven == [("sess_idle", "wake up")]
        # And crucially NOT forwarded as an inject, which would have been
        # accepted into a queue with no drainer.
        assert server.session.calls == []

    def test_busy_target_is_queued(self):
        """A mid-turn target keeps the queue path: its running turn drains."""
        from shared.message_delivery import QUEUED

        manager, server = _make_manager_with_session(
            "sess_busy", model_running=True,
        )

        status = manager.deliver_prompt_to_session("sess_busy", "later")

        assert status == QUEUED
        assert server.session.calls == [("later", None, None)]

    def test_terminated_target_is_reported_not_delivered(self):
        """A dead target is reported from its OWN terminal stamp.

        Never inferred from silence: a slow target and a dead one produce
        identical nothing, so a caller that infers cannot be wrong and know
        it.  This is the state that had no spelling at all before.
        """
        from shared.message_delivery import DELIVERED, TERMINATED

        manager, server = _make_manager_with_session(
            "sess_dead", model_running=False, terminal_reason="error",
        )
        manager.send_message_to_session = (  # type: ignore[method-assign]
            lambda sid, text: pytest.fail(
                "a terminated session must not be driven"
            )
        )

        status = manager.deliver_prompt_to_session("sess_dead", "anyone there")

        assert status == TERMINATED
        assert status not in DELIVERED
        assert server.session.calls == []

    def test_missing_session_is_no_session_not_terminated(self):
        """"Gone" and "dead but present" are different situations.

        Collapsing them is what makes an absence claim unfalsifiable.
        """
        from shared.message_delivery import NO_SESSION, TERMINATED

        manager = SessionManager()

        status = manager.deliver_prompt_to_session("sess_missing", "hello")

        assert status == NO_SESSION
        assert status != TERMINATED

    def test_no_runner_channel_is_unreachable_not_refused(self):
        """A transport fault is not a decision by the target."""
        from shared.message_delivery import UNREACHABLE

        manager, _server = _make_manager_with_session(
            "sess_pending", raise_on_get_session=True,
        )

        status = manager.deliver_prompt_to_session("sess_pending", "too early")

        assert status == UNREACHABLE

    def test_only_accepted_and_queued_count_as_delivered(self):
        """The invariant the whole vocabulary exists for.

        "It will be consumed" and "it went nowhere" must not both render as
        success -- a caller that assumes delivery and is wrong gets a silent
        stall it cannot attribute, which is the expensive direction.
        """
        from shared.message_delivery import (
            ACCEPTED, DELIVERED, NO_SESSION, QUEUED, TERMINATED, UNREACHABLE,
        )

        assert DELIVERED == {ACCEPTED, QUEUED}
        for failure in (TERMINATED, NO_SESSION, UNREACHABLE):
            assert failure not in DELIVERED

    def test_bool_adapter_agrees_with_the_status(self):
        """``inject_prompt_to_session`` must stay a pure view of the status,
        so the two forms can never disagree about the same delivery."""
        from shared.message_delivery import DELIVERED

        for label, kwargs in (
            ("busy", {"model_running": True}),
            ("idle", {"model_running": False}),
            ("dead", {"model_running": False, "terminal_reason": "error"}),
            ("no-rpc", {"raise_on_get_session": True}),
        ):
            manager, _ = _make_manager_with_session(f"sess_{label}", **kwargs)
            manager.send_message_to_session = (  # type: ignore[method-assign]
                lambda sid, text: True
            )
            status = manager.deliver_prompt_to_session(f"sess_{label}", "x")
            ok = manager.inject_prompt_to_session(f"sess_{label}", "x")
            assert ok is (status in DELIVERED), (
                f"{label}: bool={ok} disagrees with status={status}"
            )


def test_the_fake_matches_the_real_signature():
    """A fake whose signature has drifted silently tests the ``except`` branch.

    ``require_idle`` was added to the real ``session_offer_message_threadsafe``
    and not to ``_FakeRunnerRPC``, so every call raised ``TypeError`` and
    ``deliver_prompt_to_session`` reported a delivery failure.  Five tests in
    this file named for the queue-or-drive decision never reached it.  They
    did not go green-and-wrong -- they went red, in a file CI does not run, so
    nothing said so.

    Compared as a SET of accepted keywords rather than by exact signature: the
    fake may narrow defaults or annotations freely, but it must not reject a
    keyword the caller actually passes.
    """
    import inspect

    from .runner_rpc_client import RunnerRPCClient

    real = set(inspect.signature(
        RunnerRPCClient.session_offer_message_threadsafe).parameters)
    fake = set(inspect.signature(
        _FakeRunnerRPC.session_offer_message_threadsafe).parameters)

    missing = real - fake
    assert not missing, (
        f"the fake rejects {sorted(missing)}, which the daemon passes; every "
        "call would land in the caller's except branch and the tests above "
        "would exercise delivery-failure instead of what they are named for"
    )
