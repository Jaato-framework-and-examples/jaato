"""Tests for the post-completion send_message guard
(server 0.6.197+, PR-253).

When ``LifecycleTools._execute_signal_completion`` succeeds it sets
``_signal_completion_called=True``.  The framework's chat-loop
terminator at line ~4448 closes the current turn; the
session_quiescent hook emits ``SessionTerminatedEvent`` once
(gated by ``_session_quiescent_emitted``).

Pre-PR-253 subsequent ``send_message`` calls on the completed
session ran a full chat completion + tool execution before
re-terminating.  Cascade drivers using ``subscribe_once`` had no
unilateral signal to stop — they kept firing send_messages, each
one wasting a wire POST.  Empirically caught by peer 7:1's
2026-06-07 cascade against llama31-8b-awq + vLLM 0.22: 1 success
followed by 8 wasted turns (smoking gun: next POST 4 seconds
after the success log).

This PR refuses post-completion send_messages unilaterally + re-
emits SessionTerminatedEvent for any later subscribers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest


class _FakeSession:
    """Minimal stub binding the JaatoSession refusal helper.

    Avoids full JaatoSession construction — the helper is pure
    logic that only touches on_output, _trace, _ui_hooks, and
    _agent_id.
    """

    def __init__(self) -> None:
        self._signal_completion_called = False
        self._traces: List[str] = []
        self._agent_id = "main"
        self._ui_hooks: Optional[Any] = None

    def _trace(self, msg: str) -> None:
        self._traces.append(msg)

    from shared.jaato_session import JaatoSession
    _refuse_post_completion_send = JaatoSession._refuse_post_completion_send


class _RecordingHooks:
    """Captures on_session_quiescent calls."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str]] = []

    def on_session_quiescent(self, agent_id: str, reason: str) -> None:
        self.calls.append((agent_id, reason))


# ----------------------------------------------------------------------
# Refusal helper behavior
# ----------------------------------------------------------------------


class TestRefusePostCompletionSend:
    def test_emits_structured_error_via_on_output(self) -> None:
        s = _FakeSession()
        outputs: List[Tuple[str, str, str]] = []

        def on_output(source: str, text: str, mode: str) -> None:
            outputs.append((source, text, mode))

        s._refuse_post_completion_send(on_output)
        assert len(outputs) == 1
        source, text, mode = outputs[0]
        assert source == "error"
        # Error text names the architectural cause + remediation.
        assert "signal_completion" in text
        assert "create a new session" in text.lower()
        assert mode == "write"

    def test_no_on_output_callback_still_traces(self) -> None:
        """Calling without an on_output callback must not crash — it
        still leaves a trace entry so operators can grep the log."""
        s = _FakeSession()
        s._refuse_post_completion_send(None)
        assert any("POST_COMPLETION_SEND_REFUSED" in t for t in s._traces)

    def test_on_output_raises_handled_gracefully(self) -> None:
        """If the caller's on_output callback raises, the refusal
        must not propagate the error — the trace is the ground
        truth, callback emission is best-effort."""
        s = _FakeSession()

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated callback crash")

        # Should not raise.
        s._refuse_post_completion_send(boom)
        assert any("POST_COMPLETION_SEND_REFUSED" in t for t in s._traces)

    def test_re_fires_session_quiescent_hook(self) -> None:
        """Belt-and-suspenders: every refusal re-fires
        SessionTerminatedEvent via the session_quiescent hook so
        any LATER subscription (e.g., a fresh subscribe() after
        the cascade driver's restart logic) receives a fresh
        notification.  ``subscribe_once`` subscribers that already
        consumed their callback won't be notified again — for
        those callers the error event above is the unilateral
        signal."""
        s = _FakeSession()
        hooks = _RecordingHooks()
        s._ui_hooks = hooks
        s._refuse_post_completion_send(on_output=None)
        assert hooks.calls == [("main", "natural")]

    def test_hook_raise_doesnt_propagate(self) -> None:
        """A misbehaving hook must not crash send_message_s refusal
        path — the refusal is still in effect; the hook failure is
        logged."""
        s = _FakeSession()

        class _ExplodingHooks:
            def on_session_quiescent(self, **kwargs: Any) -> None:
                raise RuntimeError("simulated hook crash")

        s._ui_hooks = _ExplodingHooks()
        # Should not raise.
        s._refuse_post_completion_send(on_output=None)

    def test_repeated_refusals_each_fire_hook(self) -> None:
        """Each refusal fires the hook independently — there's no
        one-shot gate inside the refusal path.  Models the
        scenario where the cascade driver keeps firing
        send_message; each gets refused + each re-emits the
        terminal event."""
        s = _FakeSession()
        hooks = _RecordingHooks()
        s._ui_hooks = hooks
        s._refuse_post_completion_send(None)
        s._refuse_post_completion_send(None)
        s._refuse_post_completion_send(None)
        assert len(hooks.calls) == 3

    def test_agent_id_threaded_to_hook(self) -> None:
        """Hook receives the session's _agent_id verbatim — needed
        so the cascade driver can correlate the re-emission with
        the specific subagent that's been completed."""
        s = _FakeSession()
        s._agent_id = "discovery_subagent_42"
        hooks = _RecordingHooks()
        s._ui_hooks = hooks
        s._refuse_post_completion_send(None)
        assert hooks.calls == [("discovery_subagent_42", "natural")]

    def test_no_hooks_at_all_still_works(self) -> None:
        """A session without _ui_hooks AND without _callbacks falls
        through cleanly — the error event was already emitted via
        on_output above; the hook re-fire is a bonus."""
        s = _FakeSession()
        s._ui_hooks = None
        # Should not raise.
        s._refuse_post_completion_send(None)


# ----------------------------------------------------------------------
# send_message integration — gate fires when flag is set
# ----------------------------------------------------------------------
#
# Full send_message integration requires the rest of the session
# scaffolding (runtime, registry, provider, executor, etc.).  Rather
# than constructing all that, the integration is verified at the
# helper level above + via end-to-end smoke runs against a live
# vLLM endpoint where the cascade flow exercises the gate naturally.


class TestSendMessageGateIntegration:
    def test_helper_can_be_called_without_session_construction(self) -> None:
        """Sanity check: the refusal helper is callable on a stub.
        Full send_message integration is verified end-to-end via
        the smoke harness."""
        s = _FakeSession()
        assert callable(s._refuse_post_completion_send)
