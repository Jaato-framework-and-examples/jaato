"""Pin tests for bootstrap-time SessionTerminatedEvent emission
(server 0.6.169+).

Closes the bootstrap-time visibility gap documented in memory
``project_backlog_bootstrap_time_visibility_gap`` (filed 2026-05-27,
shipped 2026-05-31).  Before this fix, bootstrap-time failures
(SecretResolutionError, apparmor compose failure, runner RPC
timeout, plugin discovery errors, etc.) logged a WARNING but never
emitted ``SessionTerminatedEvent`` — cascade observers + reactor
rules matching ``session.terminated where reason='error'`` never
saw them, so cascades dead-ended silently at the IPC timeout (3min
typical) instead of producing the actionable error_type +
error_summary.

Empirical motivation (peer 7:1, 2026-05-31): gpg-agent passphrase
expiry → SecretResolutionError on first cascade.py launch → 3min
silent hang at the IPC layer before the operator saw any signal.
Defensive kb-side pre-flight `pass show` probe was rolled back per
Daniel's call: the framework should surface any bootstrap failure
class, not just one.

Pairs with PR-189 (SessionTerminated → EventBus bridge) +
PR-194 (centralized bootstrap-event router) + PR-186
(error_summary/error_type fields).  Together they form the full
visibility-on-stoppage chain: emit-side wiring + bus bridge +
matcher hoist + GC sweep guards + bootstrap-time emit.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestBootstrapFailureEmitsTerminated:
    """Validate that both bootstrap-failure paths in
    ``dispatch_bootstrap_envelope`` emit
    ``SessionTerminatedEvent(reason="error")`` with the underlying
    Exception's class name + message in ``error_type`` / ``error_summary``.
    """

    def _make_server(self):
        """Fake JaatoServer with the surface
        ``dispatch_bootstrap_envelope`` reads: ``runner_rpc``,
        ``emit``, ``_main_agent_id``.  Captures emitted events on
        a list so tests can assert against them."""
        server = MagicMock()
        server._main_agent_id = "build_judge"
        emitted: list = []
        server.emit = lambda ev: emitted.append(ev)
        return server, emitted

    def test_emit_on_runner_rpc_none_early_return(self, capsys):
        """Path A: ``server.runner_rpc`` is None (spawn helper
        raised before populating rpc).  Pre-0.6.169 this path
        logged DEBUG + returned silently.  Now emits
        SessionTerminatedEvent so cascade observers see it."""
        from jaato_sdk.events import SessionTerminatedEvent
        from server.runner_spawn import dispatch_bootstrap_envelope

        server, emitted = self._make_server()
        server.runner_rpc = None

        dispatch_bootstrap_envelope(
            server=server,
            session_id="20260531_222243",
            workspace_path="/tmp/x",
            profile_name="codegen",
        )

        terminated = [
            e for e in emitted
            if isinstance(e, SessionTerminatedEvent)
        ]
        assert len(terminated) == 1, (
            f"Expected exactly one SessionTerminatedEvent on "
            f"runner_rpc=None path; got {len(terminated)}"
        )
        ev = terminated[0]
        assert ev.session_id == "20260531_222243"
        assert ev.agent_id == "build_judge"
        assert ev.reason == "error"
        # Inner exception details captured.
        assert ev.error_type == "RuntimeError"
        assert "spawn_session_runner" in (ev.error_summary or "")

    def test_emit_on_bootstrap_rpc_exception(self):
        """Path B: bootstrap_session_threadsafe raises.  Covers
        SecretResolutionError, apparmor compose failure, runner
        RPC timeout, plugin discovery errors — any exception that
        escapes the bootstrap RPC.  Empirical case (peer 7:1,
        2026-05-31): gpg-agent passphrase expiry."""
        from jaato_sdk.events import SessionTerminatedEvent
        from server.runner_spawn import dispatch_bootstrap_envelope

        server, emitted = self._make_server()
        # Simulate SecretResolutionError on bootstrap RPC.
        class SecretResolutionError(RuntimeError):
            pass
        secret_exc = SecretResolutionError(
            "Failed to resolve secret 'pass://jaato/zhipuai/api-key': "
            "pass show failed: gpg: descifrado fallido"
        )
        server.runner_rpc = MagicMock()
        server.runner_rpc.bootstrap_session_threadsafe = MagicMock(
            side_effect=secret_exc,
        )

        dispatch_bootstrap_envelope(
            server=server,
            session_id="20260531_222243",
            workspace_path="/tmp/x",
            profile_name="codegen",
        )

        terminated = [
            e for e in emitted
            if isinstance(e, SessionTerminatedEvent)
        ]
        assert len(terminated) == 1
        ev = terminated[0]
        assert ev.session_id == "20260531_222243"
        assert ev.reason == "error"
        assert ev.error_type == "SecretResolutionError"
        assert "pass://jaato/zhipuai/api-key" in (ev.error_summary or "")
        assert "gpg" in (ev.error_summary or "")

    def test_emit_failure_does_not_mask_bootstrap_failure(self, caplog):
        """Defensive: if server.emit itself raises (transport
        error, no subscribers, etc.), the helper logs the
        emit-failure but does NOT raise — caller's bootstrap-
        failure handling continues unchanged.  Keeps the
        visibility-on-stoppage path from masking the underlying
        bootstrap failure."""
        from server.runner_spawn import _emit_bootstrap_terminated

        server = MagicMock()
        server._main_agent_id = "main"
        server.emit = MagicMock(side_effect=RuntimeError("emit boom"))

        # Should not raise.
        _emit_bootstrap_terminated(
            server=server,
            session_id="20260531_222243",
            exc=ValueError("original bootstrap error"),
        )
        # Both the emit-failure and root-cause survive in the log
        # trail.  We check for the helper's own diagnostic line.
        assert any(
            "_emit_bootstrap_terminated" in rec.message
            for rec in caplog.records
        ), (
            "Helper must log a diagnostic line when emit fails so "
            "the visibility-path failure is itself observable."
        )

    def test_emit_uses_main_fallback_when_agent_id_absent(self):
        """Defensive: ``server._main_agent_id`` may not be set
        when bootstrap fails early.  ``agent_id`` falls back to
        ``"main"`` so the field is always populated."""
        from jaato_sdk.events import SessionTerminatedEvent
        from server.runner_spawn import dispatch_bootstrap_envelope

        server = MagicMock()
        # Explicitly remove _main_agent_id to simulate early-fail.
        del server._main_agent_id
        server.runner_rpc = None
        emitted: list = []
        server.emit = lambda ev: emitted.append(ev)

        dispatch_bootstrap_envelope(
            server=server,
            session_id="sid",
            workspace_path=None,
            profile_name="any",
        )

        terminated = [
            e for e in emitted
            if isinstance(e, SessionTerminatedEvent)
        ]
        assert len(terminated) == 1
        assert terminated[0].agent_id == "main"
