"""Tests for pool PR 5a — per-slot AppArmor self-confine.

``bootstrap_session`` step 1c calls ``confine_to_profile`` when
``envelope.profile_name`` is set AND the runner is not already
confined.  This restores per-session AppArmor confinement parity
between cold-spawned and pool-served runners.

Two surfaces:

1. Pool slot path: starts unconfined, transitions to per-session
   profile via ``confine_to_profile`` in step 1c.
2. Cold-spawn path: already confined by ``__main__.py`` step 2,
   step 1c detects via ``/proc/self/attr/current`` and skips.

Tests stub the AppArmor bootstrap module so the suite runs on hosts
without an apparmor policy (CI, dev workstations, containers).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from server.runner.bootstrap import ThreadProfileScan
from server.runner.session import (
    BootstrapError,
    _maybe_self_confine,
    bootstrap_session,
)
from shared.session_envelope import SessionInitEnvelope

from .conftest import StubSession


def _envelope(profile_name: str = "") -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-5a",
        workspace_path="/tmp/ws",
        profile_name=profile_name,
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
    )


class _StubRuntime:
    def __init__(self) -> None:
        self._registry = object()
        self.is_connected = False

    def connect(self, project: str, location: str) -> None:
        self.is_connected = True

    def create_session(self, **kwargs: Any) -> Any:
        class _Executor:
            def set_apparmor_child_transition_callback(self, cb):
                # PR 102: //child install matrix expects this method on
                # the executor when envelope.profile_name is set (case
                # 3).  No-op here — this test file mocks
                # ``confine_to_profile`` directly to exercise
                # ``_maybe_self_confine`` orchestration; the //child
                # install step runs but the install itself doesn't
                # touch real kernel state.
                pass

        # Subclasses the shared conftest stub: bootstrap_session stamps
        # the session (set_daemon_session_id / set_client_user_id)
        # before the self-confine step under test (#736).
        class _S(StubSession):
            _session_env: Dict[str, str] = {}
            _executor = _Executor()

        return _S()


# ----------------------------------------------------------------------
# No-op cases — should NOT call confine_to_profile
# ----------------------------------------------------------------------


def test_maybe_self_confine_empty_profile_is_noop() -> None:
    """envelope.profile_name empty → operator opted out of
    confinement.  step 1c is a no-op (runner stays unconfined)."""
    with patch("server.runner.bootstrap.confine_to_profile") as confine_mock:
        _maybe_self_confine(_envelope(profile_name=""))
    confine_mock.assert_not_called()


def test_maybe_self_confine_already_confined_skips() -> None:
    """Cold-spawn path: ``__main__.py`` already confined.  step 1c
    detects via /proc/self/attr/current and skips the redundant
    transition (which would also fail — per-session profiles omit
    ``change_profile -> self``)."""
    with patch(
        "server.runner.bootstrap.read_current_profile",
        return_value="jaato-ws-sess-5a (enforce)",
    ), patch(
        # #1023: step 1c now walks every thread's own attr/current after
        # the process is confined.  This test SIMULATES confinement, so
        # the real walk would read the host's actual /proc — where no
        # thread is in ``jaato-ws-sess-5a`` — and correctly refuse.  Stub
        # it for the same reason the two calls above are stubbed.
        "server.runner.bootstrap.verify_thread_confinement",
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
    ) as confine_mock:
        _maybe_self_confine(_envelope(profile_name="jaato-ws-sess-5a"))
    confine_mock.assert_not_called()


def test_maybe_self_confine_already_confined_no_mode_suffix_skips() -> None:
    """``read_current_profile`` may return the profile name WITHOUT
    the enforcement-mode suffix on some kernels.  Match exact-equal
    too (not just prefix)."""
    with patch(
        "server.runner.bootstrap.read_current_profile",
        return_value="jaato-ws-sess-5a",
    ), patch(
        # See the note in the preceding test (#1023).
        "server.runner.bootstrap.verify_thread_confinement",
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
    ) as confine_mock:
        _maybe_self_confine(_envelope(profile_name="jaato-ws-sess-5a"))
    confine_mock.assert_not_called()


# ----------------------------------------------------------------------
# Active transition — pool slot path
# ----------------------------------------------------------------------


def test_maybe_self_confine_pool_slot_transitions() -> None:
    """Pool slot starts unconfined → step 1c calls
    ``confine_to_profile`` with envelope.profile_name."""
    with patch(
        "server.runner.bootstrap.read_current_profile",
        return_value="unconfined",
    ), patch(
        # See the note in ``..._already_confined_skips`` (#1023).
        "server.runner.bootstrap.verify_thread_confinement",
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
    ) as confine_mock:
        _maybe_self_confine(_envelope(profile_name="jaato-ws-pool-5a"))
    confine_mock.assert_called_once_with("jaato-ws-pool-5a")


# ----------------------------------------------------------------------
# Failure modes — must raise BootstrapError("confine", ...)
# ----------------------------------------------------------------------


def test_maybe_self_confine_kernel_refuses_raises_bootstrap_error() -> None:
    """``aa_change_profile`` failure surfaces as
    ``BootstrapError(stage='confine', ...)`` so daemon-side bootstrap
    dispatch logs the failure."""
    with patch(
        "server.runner.bootstrap.read_current_profile",
        return_value="unconfined",
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
        side_effect=RuntimeError(
            "aa_change_profile('jaato-ws-x') failed: errno=13 (Permission denied)"
        ),
    ):
        with pytest.raises(BootstrapError) as excinfo:
            _maybe_self_confine(_envelope(profile_name="jaato-ws-x"))
        assert excinfo.value.stage == "confine"
        assert "jaato-ws-x" in excinfo.value.message


def test_maybe_self_confine_mismatch_raises_bootstrap_error() -> None:
    """``ConfinementMismatchError`` from ``confine_to_profile``
    surfaces as ``BootstrapError(stage='confine', ...)``."""
    from server.runner.bootstrap import ConfinementMismatchError

    with patch(
        "server.runner.bootstrap.read_current_profile",
        return_value="unconfined",
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
        side_effect=ConfinementMismatchError(
            expected="jaato-ws-x",
            actual="unconfined",
        ),
    ):
        with pytest.raises(BootstrapError) as excinfo:
            _maybe_self_confine(_envelope(profile_name="jaato-ws-x"))
        assert excinfo.value.stage == "confine"
        # Helpful diagnostic mentions the pool slot context.
        assert "change_profile" in excinfo.value.message


# ----------------------------------------------------------------------
# End-to-end: bootstrap_session integrates step 1c
# ----------------------------------------------------------------------


def test_bootstrap_session_runs_self_confine_for_pool_slot() -> None:
    """End-to-end: bootstrap_session with envelope.profile_name set +
    runner unconfined → confine_to_profile is called BEFORE runtime
    construction.

    Extended for #1023: the worker-pool recycle and the per-thread
    verification both sit between the transition and runtime
    construction.  Recycling must precede verification (it removes the
    population the framework CAN remove, so what verification then finds
    is a thread neither lane reached), and both must precede any plugin
    or runtime code — that code is what the confinement exists to bound.
    """
    runtime = _StubRuntime()
    call_order: List[str] = []

    def _record_confine(profile: str) -> None:
        call_order.append(f"confine:{profile}")

    def _record_recycle(reason: str) -> Dict[str, Any]:
        call_order.append("recycle")
        return {}

    def _record_verify(profile: str, **kwargs: Any) -> Any:
        call_order.append("verify")
        # A real scan object, not a bare sentinel: the caller reads the
        # result to decide what to log, and a stub that cannot be read
        # would pass this test while hiding a broken caller.
        return ThreadProfileScan(
            expected=profile, matched=(1,), divergent=(), unreadable=(),
            gone=(), route="task_dir",
        )

    def _record_runtime_factory(envelope: SessionInitEnvelope) -> _StubRuntime:
        call_order.append("runtime_construct")
        return runtime

    with patch(
        "server.runner.bootstrap.read_current_profile",
        return_value="unconfined",
    ), patch(
        "server.runner.bootstrap.verify_thread_confinement",
        side_effect=_record_verify,
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
        side_effect=_record_confine,
    ):
        bootstrap_session(
            _envelope(profile_name="jaato-ws-e2e"),
            runtime_factory=_record_runtime_factory,
            recycle_pools=_record_recycle,
        )

    assert call_order == [
        "confine:jaato-ws-e2e", "recycle", "verify", "runtime_construct",
    ], (
        f"Self-confine, worker-pool recycle and per-thread verification "
        f"must all happen BEFORE runtime construction, in that order.  "
        f"Observed order: {call_order}"
    )


def test_bootstrap_session_propagates_confine_error_as_bootstrap_error() -> None:
    """A confinement failure in step 1c stops bootstrap before
    runtime construction — runtime_factory is never called."""
    factory_calls = []

    def _factory(envelope: SessionInitEnvelope) -> _StubRuntime:
        factory_calls.append(envelope)
        return _StubRuntime()

    with patch(
        "server.runner.bootstrap.read_current_profile",
        return_value="unconfined",
    ), patch(
        "server.runner.bootstrap.confine_to_profile",
        side_effect=RuntimeError("aa_change_profile failed: errno=13"),
    ):
        with pytest.raises(BootstrapError) as excinfo:
            bootstrap_session(
                _envelope(profile_name="jaato-ws-fail"),
                runtime_factory=_factory,
            )
        assert excinfo.value.stage == "confine"
    assert factory_calls == [], (
        "runtime_factory should not have been called when step 1c failed"
    )
