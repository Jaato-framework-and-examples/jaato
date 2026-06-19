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
from jaato_sdk.events import AgentOutputEvent, SessionTerminatedEvent, TurnCompletedEvent


def _out(source: str, text: str, mode: str = "write") -> AgentOutputEvent:
    return AgentOutputEvent(agent_id="a", source=source, text=text, mode=mode)


def _turn_completed() -> TurnCompletedEvent:
    return TurnCompletedEvent(agent_id="a")


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

    def test_natural_completion_via_turn_completed_no_session_terminated(self):
        # The production happy path: a single-shot ephemeral emits its output
        # then turn.completed, and the headless session goes IDLE — it does
        # NOT emit SessionTerminatedEvent on natural success.  Pre-fix the run
        # blocked on done.wait() until the 1800s timeout (gossip co-validation
        # 2026-06-19: origin got the full output stream but never
        # PeerAgentCompleted, because execute_spawn sends completion only after
        # run_ephemeral_session RETURNS).  The old tests only ever injected a
        # mocked SessionTerminatedEvent, so CI never exercised this path.
        chunks = []
        fake, cap = _fake_self([
            _out("model", "Hello "),
            _out("model", "world", mode="append"),
            _turn_completed(),          # natural terminal — NO SessionTerminatedEvent
        ])
        result = _run(fake, on_output=lambda s, t, m: chunks.append((s, t)))
        assert chunks == [("model", "Hello "), ("model", "world")]
        assert result == "Hello world"          # returns, does not hang
        fake.delete_session.assert_called_once_with("sess-123")  # clean teardown

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

    # ---- workspace handling (co-validation regression: no workspace ->
    #      no runner -> NoneType.session_send_message_threadsafe) ----

    def test_workspace_none_provisions_scratch(self):
        """An inline spawn with no workspace gets a scratch workspace, so the
        runner spawns (without a cwd the runner-spawn gate is skipped,
        ``server._runner_rpc`` stays None, and the model turn crashes)."""
        import os
        import shutil
        import tempfile
        fake, cap = _fake_self([_terminated("natural")])
        _run(fake)  # workspace_path=None (see _run's 7th positional arg)
        ws = cap["create_kwargs"]["workspace_path"]
        assert ws, "no workspace -> create_headless_session must get a scratch dir"
        assert ws.startswith(tempfile.gettempdir())
        assert os.path.basename(ws).startswith("jaato-ephemeral-")
        assert os.path.isdir(ws)
        # config_root is derived from the (scratch) workspace so the runner's
        # core plugins (file_edit's backup manager) resolve their backup base.
        # The auto "<workspace>/.jaato" fallback was removed (PR-147), so an
        # unset config_root makes file_edit fail to expose with a loud (but
        # non-fatal) RuntimeError traceback on every ephemeral spawn.
        assert cap["create_kwargs"]["config_root"] == os.path.join(ws, ".jaato")
        shutil.rmtree(ws, ignore_errors=True)  # don't leak the scratch dir

    def test_workspace_provided_passes_through(self):
        """A supplied workspace is forwarded verbatim — no scratch override."""
        fake, cap = _fake_self([_terminated("natural")])
        SessionManager._run_ephemeral_session_impl(
            fake, '{"model": "x", "provider": "p"}', "", "do it", "worker",
            None, "/real/workspace", None,
        )
        assert cap["create_kwargs"]["workspace_path"] == "/real/workspace"
        # config_root mirrors normal-session semantics: "<workspace>/.jaato".
        import os
        assert cap["create_kwargs"]["config_root"] == os.path.join(
            "/real/workspace", ".jaato"
        )

    def test_inline_spawn_decouples_agent_name_from_resolution(self):
        """For an inline spawn the display label rides ``session_name`` and
        ``agent_name`` is None — so ``_create_session_impl`` skips
        ``_resolve_agent`` (a display label is not a ``.jaato/agents`` persona;
        resolving it returned an empty session_id in co-validation)."""
        fake, cap = _fake_self([_terminated("natural")])
        _run(fake)  # agent_name="worker" + inline profile (merged truthy)
        ck = cap["create_kwargs"]
        assert ck["agent_name"] is None
        assert ck["session_name"] == "worker"
