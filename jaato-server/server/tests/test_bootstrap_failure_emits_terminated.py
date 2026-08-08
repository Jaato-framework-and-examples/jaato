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
        """Fake JaatoServer.  Bootstrap failure now ROUTES THROUGH the single
        error-termination chokepoint ``_emit_error_termination_from_exc`` (which
        emits AgentErrorEvent then SessionTerminatedEvent(reason=error) +
        stamps _terminal_reason), so tests assert on the delegation rather than
        the raw emit.  Field-correctness is covered by the chokepoint's own
        test."""
        server = MagicMock()
        server._main_agent_id = "build_judge"
        return server

    def test_emit_on_runner_rpc_none_early_return(self, capsys):
        """Path A: ``server.runner_rpc`` is None (spawn helper raised before
        populating rpc) → delegates the inner RuntimeError to the chokepoint
        with the session_id + agent_id."""
        from server.runner_spawn import dispatch_bootstrap_envelope

        server = self._make_server()
        server.runner_rpc = None

        dispatch_bootstrap_envelope(
            server=server,
            session_id="20260531_222243",
            workspace_path="/tmp/x",
            profile_name="codegen",
        )

        server._emit_error_termination_from_exc.assert_called_once()
        call = server._emit_error_termination_from_exc.call_args
        exc = call.args[0]
        assert isinstance(exc, RuntimeError)
        assert "spawn_session_runner" in str(exc)
        assert call.kwargs["session_id"] == "20260531_222243"
        assert call.kwargs["agent_id"] == "build_judge"

    def test_emit_on_bootstrap_rpc_exception(self):
        """Path B: bootstrap_session_threadsafe raises (SecretResolutionError,
        apparmor compose failure, RPC timeout, …) → the exact exception is
        delegated to the chokepoint.  Empirical case (peer 7:1, 2026-05-31):
        gpg-agent passphrase expiry."""
        from server.runner_spawn import dispatch_bootstrap_envelope

        server = self._make_server()
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

        server._emit_error_termination_from_exc.assert_called_once()
        passed_exc = server._emit_error_termination_from_exc.call_args.args[0]
        assert passed_exc is secret_exc
        assert "pass://jaato/zhipuai/api-key" in str(passed_exc)

    def test_emit_failure_does_not_mask_bootstrap_failure(self, caplog):
        """Defensive: if the chokepoint itself raises (transport error, no
        subscribers, etc.), the helper logs but does NOT raise — the caller's
        bootstrap-failure handling continues, the underlying failure isn't
        masked."""
        from server.runner_spawn import _emit_bootstrap_terminated

        server = MagicMock()
        server._main_agent_id = "main"
        server._emit_error_termination_from_exc = MagicMock(
            side_effect=RuntimeError("emit boom"),
        )

        # Should not raise.
        _emit_bootstrap_terminated(
            server=server,
            session_id="20260531_222243",
            exc=ValueError("original bootstrap error"),
        )
        assert any(
            "_emit_bootstrap_terminated" in rec.message
            for rec in caplog.records
        ), (
            "Helper must log a diagnostic line when the chokepoint fails so "
            "the visibility-path failure is itself observable."
        )

    def test_emit_uses_main_fallback_when_agent_id_absent(self):
        """Defensive: ``server._main_agent_id`` may not be set when bootstrap
        fails early.  ``agent_id`` falls back to ``"main"`` in the chokepoint
        delegation so the field is always populated."""
        from server.runner_spawn import dispatch_bootstrap_envelope

        server = MagicMock()
        del server._main_agent_id  # simulate early-fail
        server.runner_rpc = None

        dispatch_bootstrap_envelope(
            server=server,
            session_id="sid",
            workspace_path=None,
            profile_name="any",
        )

        server._emit_error_termination_from_exc.assert_called_once()
        assert server._emit_error_termination_from_exc.call_args.kwargs["agent_id"] == "main"
