"""Spike test: _run_ephemeral_session_impl routes through create_headless_session.

The §7c seat-flip broke the ephemeral remote-spawn path (JaatoServer.send_message
lost its on_output kwarg -> TypeError) AND left it running unconfined
(client_id=None -> no runner/AppArmor).  CI stayed green only because the
pre-existing tests never executed the impl body — test_ephemeral_migration.py
mocked _FakeServer.send_message, premium test_remote_spawn.py mocked the whole
SessionManager.  This test closes that gap: it DRIVES the real
``_run_ephemeral_session_impl`` body — in-process-client registration, the
{model}-only source filter, the per-emit forwarder (streaming preserved off the
daemon lock), block-on-terminal, exception-on-failure, and the on_started STOP
hook — by stubbing create_headless_session to asynchronously drive the captured
in-process callback with synthetic EventBus events.
"""

from __future__ import annotations

import threading
import time

import pytest
from unittest.mock import MagicMock

from server.session_manager import SessionManager
from jaato_sdk.events import AgentOutputEvent, SessionTerminatedEvent


def _out(source: str, text: str, mode: str = "write") -> AgentOutputEvent:
    return AgentOutputEvent(agent_id="a", source=source, text=text, mode=mode)


def _terminated(reason: str, error_type=None, error_summary=None):
    return SessionTerminatedEvent(
        session_id="sess-123", agent_id="a", reason=reason,
        error_type=error_type, error_summary=error_summary,
    )


def _fake_self(emit_events, session_id="sess-123"):
    """Fake `self` whose create_headless_session asynchronously drives the
    captured in-process callback with `emit_events`, then returns session_id."""
    fake = MagicMock()
    cap: dict = {}

    def _register(client_id, callback, cascade_driver_id, role="observer", **kw):
        cap["cb"] = callback
    fake.register_in_process_client.side_effect = _register

    def _create_headless(**kwargs):
        cap["create_kwargs"] = kwargs
        if not session_id:
            return ""
        cb = cap["cb"]

        def _emit():
            time.sleep(0.02)  # ensure the body is blocked on done.wait()
            for ev in emit_events:
                cb(ev)
        threading.Thread(target=_emit, daemon=True).start()
        return session_id
    fake.create_headless_session.side_effect = _create_headless
    return fake, cap


def _run(fake, on_output=None, on_started=None, profile='{"model": "x", "provider": "p"}'):
    return SessionManager._run_ephemeral_session_impl(
        fake,
        profile,            # profile_json
        "",                 # inline_config_json
        "do it",            # prompt
        "worker",           # agent_name
        on_output,          # on_output
        None,               # workspace_path
        on_started,         # on_started
    )


class TestEphemeralHeadlessShim:
    def test_routes_through_headless_filters_source_streams_per_emit(self):
        chunks = []
        fake, cap = _fake_self([
            _out("model", "Hello "),
            _out("user", "do it"),          # prompt echo -> DROPPED
            _out("thinking", "hmm..."),     # CoT -> DROPPED (default)
            _out("tool", "{...}"),          # tool io -> DROPPED
            _out("model", "world", mode="append"),
            _terminated("natural"),
        ])
        result = _run(fake, on_output=lambda s, t, m: chunks.append((s, t)))

        # Only model output relayed, in order, per-emit (streaming preserved).
        assert chunks == [("model", "Hello "), ("model", "world")]
        assert result == "Hello world"
        # Routed through the CONFINED headless path with the inline profile +
        # a per-spawn cascade_driver_id — not an in-daemon JaatoServer.
        ck = cap["create_kwargs"]
        assert ck["inline_profile_data"]["model"] == "x"
        assert ck["cascade_driver_id"].startswith("ephemeral-")
        assert ck["initial_prompt"] == "do it"
        # Cleanup: client unregistered + session deleted.
        fake.unregister_cascade_client.assert_called_once()
        fake.delete_session.assert_called_once_with("sess-123")

    def test_error_terminal_raises_with_cause(self):
        fake, _ = _fake_self([
            _out("model", "partial"),
            _terminated("error", error_type="ProviderError", error_summary="boom"),
        ])
        with pytest.raises(RuntimeError) as exc:
            _run(fake)
        assert "ProviderError" in str(exc.value) and "boom" in str(exc.value)
        fake.delete_session.assert_called_once()  # cleanup still runs on failure

    def test_cancelled_terminal_raises(self):
        # stopped/cascade_cancelled are non-clean -> exception (success=False).
        fake, _ = _fake_self([_terminated("stopped")])
        with pytest.raises(RuntimeError, match="reason=stopped"):
            _run(fake)

    def test_on_started_receives_real_session_id_before_block(self):
        got = {}
        fake, _ = _fake_self([_out("model", "x"), _terminated("natural")])
        _run(fake, on_started=lambda sid: got.setdefault("sid", sid))
        assert got["sid"] == "sess-123"

    def test_boot_failure_returns_hint_no_delete(self):
        fake, _ = _fake_self([], session_id="")  # create_headless returns ""
        result = _run(fake)
        assert "initialization failed" in result.lower()
        fake.delete_session.assert_not_called()  # no session to clean

    def test_profile_inline_merge_reaches_headless(self):
        # Preserves the old migration test's merge coverage against the new
        # path: profile_data is applied first via setdefault (profile wins on
        # collision), inline_config fills the gaps; the merged dict is what
        # reaches create_headless_session as inline_profile_data.
        fake, cap = _fake_self([_out("model", "x"), _terminated("natural")])
        SessionManager._run_ephemeral_session_impl(
            fake,
            '{"model": "profile-model", "provider": "p"}',   # profile_json
            '{"model": "inline-model", "max_turns": 5}',     # inline_config_json
            "do it", "worker", None, None, None,
        )
        merged = cap["create_kwargs"]["inline_profile_data"]
        assert merged["model"] == "profile-model"  # profile wins collision
        assert merged["provider"] == "p"           # profile-only key
        assert merged["max_turns"] == 5            # inline fills the gap
