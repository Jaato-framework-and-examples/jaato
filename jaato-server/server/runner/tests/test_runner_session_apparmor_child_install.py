"""Tests for the Phase 5 §5.10c AppArmor //child transition install
in ``bootstrap_session`` (per peer review of e805e4d0).

Pins the audible-failure contract for the install step:

- ``envelope.profile_name`` empty  → install skipped silently (operator
  opted out of confinement via ``JAATO_RUNNER_DISABLE_CONFINE=1`` OR
  profile-less unconfined session).
- ``envelope.profile_name`` set    → install MUST succeed or
  bootstrap MUST raise ``BootstrapError("configure", ...)``.  A
  silent install failure would leave the session running with the
  escape vector open — exactly the gap §5.10 closes.

**Source-of-truth migration (PR 102, 2026-05-13).**  Pre-PR-102 this
matrix read ``os.environ.get("JAATO_RUNNER_PROFILE")``, which
``RunnerSpawner._build_env`` set on cold-spawn but the
template-fork-no-exec path never set on pool slots — so pool-served
sessions confined via PR 5a's ``_maybe_self_confine`` took the
"unconfined" case 1 branch here and silently skipped //child install,
re-opening the apparmor.py:413-449 escape vector.  Tests now drive
``envelope.profile_name`` directly (the canonical source); the env
var no longer affects this code path.

This mirrors the Phase 4 §4.3 PR #57 silent-isolation-downgrade
fix: when an operator opts into kernel-level confinement,
missing/broken wiring is a bootstrap-time error, not a runtime
degrade.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from server.runner.session import (
    BootstrapError,
    bootstrap_session,
)
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------


class _StubRuntimeWithSession:
    """Stub runtime + session pair used by the bootstrap tests.

    The runtime is pre-configured (``_registry`` truthy) so the Path D
    plugin-discovery step short-circuits — we're only interested in
    the §5.10c install step at the tail of ``bootstrap_session``.

    The created session exposes an ``_executor`` attribute whose
    shape varies between tests:

    - ``with_setter=True``  → executor has
      ``set_apparmor_child_transition_callback``; install succeeds.
    - ``with_setter=False`` → executor lacks the method; install
      must raise ``BootstrapError("configure", ...)``.
    - ``raises=<exc>``      → setter raises the supplied exception;
      install must raise ``BootstrapError`` wrapping it.
    """

    def __init__(
        self,
        *,
        with_setter: bool = True,
        raises: Any = None,
    ) -> None:
        self.is_connected = True
        self._registry = object()  # short-circuit Path D
        self._with_setter = with_setter
        self._raises = raises
        self.install_calls: list = []

    def connect(self, project: str, location: str) -> None:
        pass

    def configure_plugins(self, *args: Any, **kwargs: Any) -> None:
        pass

    def create_session(self, **kwargs: Any) -> Any:
        stub = self

        class _StubExecutor:
            def __init__(inner) -> None:
                if stub._with_setter:
                    # Bind set_apparmor_child_transition_callback so
                    # bootstrap can find + invoke it.
                    def _setter(cb):
                        if stub._raises is not None:
                            raise stub._raises
                        stub.install_calls.append(cb)
                    inner.set_apparmor_child_transition_callback = _setter

        class _StubSession:
            def __init__(inner) -> None:
                inner._executor = _StubExecutor()

        return _StubSession()


def _envelope(profile_name: str = "jaato-ws-sess-5-10c") -> SessionInitEnvelope:
    """Build an envelope with a configurable AppArmor profile name."""
    return SessionInitEnvelope(
        session_id="sess-5-10c",
        workspace_path="/tmp/ws",
        profile_name=profile_name,
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
    )


# ----------------------------------------------------------------------
# Audible-failure pins
# ----------------------------------------------------------------------


def test_bootstrap_raises_when_apparmor_install_fails_with_profile_set() -> None:
    """Pin: ``envelope.profile_name`` set + executor lacks the setter →
    bootstrap raises ``BootstrapError("configure", ...)``.

    The operator opted into kernel confinement via the envelope; a
    silent fallback to today's pre-§5.10c behavior would leave the
    escape vector open.  Matches the §4.3 PR #57 audible-failure
    pattern."""
    stub = _StubRuntimeWithSession(with_setter=False)

    with pytest.raises(BootstrapError) as exc_info:
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: stub,
        )

    assert exc_info.value.stage == "configure"
    msg = exc_info.value.message
    assert "AppArmor //child transition" in msg
    assert "envelope.profile_name" in msg
    assert "JAATO_RUNNER_DISABLE_CONFINE" in msg, (
        "error message must point operators at the documented escape "
        "hatch (JAATO_RUNNER_DISABLE_CONFINE=1) so they can opt out "
        "explicitly rather than silently degrading"
    )


def test_bootstrap_raises_when_setter_itself_crashes() -> None:
    """Pin: ``envelope.profile_name`` set + setter raises → bootstrap
    raises ``BootstrapError("configure", ...)`` wrapping the
    underlying exception.

    Distinguishes the "setter missing" case (hasattr fails) from the
    "setter present but broken" case.  Both are audible-failure
    territory under the §5.10c contract."""
    stub = _StubRuntimeWithSession(
        with_setter=True,
        raises=RuntimeError("install boom"),
    )

    with pytest.raises(BootstrapError) as exc_info:
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: stub,
        )

    assert exc_info.value.stage == "configure"
    assert "install crashed" in exc_info.value.message
    assert "install boom" in exc_info.value.message


def test_bootstrap_silent_when_envelope_profile_empty() -> None:
    """Pin: ``envelope.profile_name`` empty/None → install skipped,
    session constructs normally even when the executor lacks the
    setter.

    This is the unconfined-session path: profile-less startup or
    explicit JAATO_RUNNER_DISABLE_CONFINE=1.  The §5.10c contract
    doesn't apply when the runner itself isn't confined."""
    stub = _StubRuntimeWithSession(with_setter=False)

    host = bootstrap_session(
        _envelope(profile_name=""),
        runtime_factory=lambda env: stub,
    )

    assert host is not None
    assert stub.install_calls == []


def test_bootstrap_logs_info_on_successful_install(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin: happy path — ``envelope.profile_name`` set + setter present
    → install runs, INFO log records the profile name.

    Operators reviewing runner-side logs see one ``installed AppArmor
    //child transition callback`` line per session, which becomes the
    evidence trail that the escape-vector closure is wired."""
    import logging
    stub = _StubRuntimeWithSession(with_setter=True)

    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: stub,
        )

    # Setter was called once (with a callable — the transition cb).
    assert len(stub.install_calls) == 1
    assert callable(stub.install_calls[0])

    # INFO log records the install with the profile name embedded.
    install_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.INFO
        and "installed AppArmor" in r.message
        and "//child" in r.message
    ]
    assert install_msgs, (
        f"missing INFO log; got {[r.message for r in caplog.records]!r}"
    )
    assert any(
        "jaato-ws-sess-5-10c" in m for m in install_msgs
    ), "profile name must appear in the install log"


# ----------------------------------------------------------------------
# Phase 5 §5.10e — sub-runner skip
# ----------------------------------------------------------------------


def test_install_skipped_when_envelope_profile_is_subprofile(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Phase 5 §5.10e: ``envelope.profile_name`` contains the Audit 6
    sub-profile separator ``//`` → install is skipped silently
    (INFO log).  v15 design intent (PR #67) already drops the
    escape primitive from the sub-profile, so subprocesses inherit
    the no-escape posture by construction — no //child transition
    needed (and not installable: sub-profile lacks writable
    attr/current per the same v15 work).

    Setter MUST NOT be called.  Bootstrap MUST succeed.  INFO log
    MUST explain the skip with the profile name + audit reference."""
    import logging
    stub = _StubRuntimeWithSession(with_setter=True)

    with caplog.at_level(logging.INFO):
        host = bootstrap_session(
            _envelope(profile_name="jaato-ws-parent//subagent"),
            runtime_factory=lambda env: stub,
        )

    assert host is not None
    assert stub.install_calls == [], (
        "sub-runner skip path called the setter — §5.10e says it "
        "must not, because the sub-profile lacks writable "
        "attr/current and the transition would EACCES at preexec_fn"
    )
    skip_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.INFO
        and "sub-profile" in r.message
        and "skipping" in r.message
    ]
    assert skip_msgs, (
        f"missing INFO log explaining the sub-runner skip; "
        f"got {[r.message for r in caplog.records]!r}"
    )
    assert any(
        "jaato-ws-parent//subagent" in m for m in skip_msgs
    ), "sub-profile name must appear in the skip log"


def test_install_skipped_for_subprofile_even_if_setter_missing() -> None:
    """Pin: the sub-runner skip path bypasses the audible-failure
    contract.  Without this test, a refactor that moves the
    has_setter check ABOVE the sub-profile branch would re-impose
    the audible-failure on sub-runners — breaking isolated subagents
    on hosts that legitimately lack the setter (none today, but
    defensive).

    Setter missing + sub-profile name → no BootstrapError, no setter
    call.  The §5.10c audible-failure contract applies ONLY to main
    runners (case 3 in the three-case matrix)."""
    stub = _StubRuntimeWithSession(with_setter=False)

    host = bootstrap_session(
        _envelope(profile_name="jaato-ws-parent//subagent"),
        runtime_factory=lambda env: stub,
    )

    assert host is not None
    assert stub.install_calls == []


def test_main_runner_install_path_unchanged_post_5_10e(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin: main runners (profile name lacks ``//``) still run the
    §5.10c install path — happy + audible-failure both preserved."""
    import logging
    stub = _StubRuntimeWithSession(with_setter=True)

    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: stub,
        )

    assert len(stub.install_calls) == 1


def test_three_case_matrix_distinguished_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin: the three log lines for the three matrix cases are
    distinguishable.  Operators triaging an apparmor incident need
    to see WHY a session ended up unconfined or sub-profile-skipped
    — silent same-shape logs would hide bugs."""
    import logging

    # Case 1: empty profile.
    stub1 = _StubRuntimeWithSession(with_setter=True)
    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(profile_name=""),
            runtime_factory=lambda env: stub1,
        )
    case1_msgs = [
        r.message for r in caplog.records
        if "unconfined" in r.message and "skipping" in r.message
    ]
    assert case1_msgs, "case 1 (empty) should log unconfined-skip"
    caplog.clear()

    # Case 2: sub-profile.
    stub2 = _StubRuntimeWithSession(with_setter=True)
    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(profile_name="jaato-ws-x//child-y"),
            runtime_factory=lambda env: stub2,
        )
    case2_msgs = [
        r.message for r in caplog.records
        if "sub-profile" in r.message and "skipping" in r.message
    ]
    assert case2_msgs, "case 2 (sub-profile) should log sub-profile-skip"
    caplog.clear()

    # Case 3: main runner.
    stub3 = _StubRuntimeWithSession(with_setter=True)
    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: stub3,
        )
    case3_msgs = [
        r.message for r in caplog.records
        if "installed AppArmor" in r.message
    ]
    assert case3_msgs, "case 3 (main runner) should log install success"


# ----------------------------------------------------------------------
# PR 102 regression: pool slot has empty JAATO_RUNNER_PROFILE env var
# but envelope.profile_name carries the per-session profile.  Install
# MUST run for the pool-slot case — pre-PR-102 it was incorrectly
# skipped.
# ----------------------------------------------------------------------


def test_pool_slot_install_runs_when_env_empty_but_envelope_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin for the pool-slot AppArmor escape (PR 102).

    Pool slots inherit the daemon's os.environ via template-fork — the
    daemon never sets ``JAATO_RUNNER_PROFILE``, so the slot's env var
    is empty.  The runner is nevertheless confined to
    ``envelope.profile_name`` via PR 5a's ``_maybe_self_confine``.
    Pre-PR-102 this matrix read ``os.environ`` and took case 1
    (unconfined-skip), silently leaving the //child callback
    uninstalled → cli/interactive_shell subprocesses inherited the
    runner's base profile with escape primitives retained.

    Post-PR-102 the matrix reads ``envelope.profile_name`` so the
    pool-slot case correctly falls into case 3 (main-runner install).

    Test setup:
      - Delete JAATO_RUNNER_PROFILE env var (mimics pool slot env).
      - Envelope carries profile_name="jaato-ws-pool-victim".
      - Stub executor has the setter.

    Pre-PR-102 assertion: ``install_calls == []`` (skipped — bug).
    Post-PR-102 assertion: ``install_calls`` has one callable (fixed).
    """
    monkeypatch.delenv("JAATO_RUNNER_PROFILE", raising=False)
    stub = _StubRuntimeWithSession(with_setter=True)

    bootstrap_session(
        _envelope(profile_name="jaato-ws-pool-victim"),
        runtime_factory=lambda env: stub,
    )

    assert len(stub.install_calls) == 1, (
        f"Pool-slot //child install regression: expected exactly one "
        f"setter call (case 3 in matrix), got {len(stub.install_calls)}.  "
        f"Pre-PR-102 the matrix read os.environ.JAATO_RUNNER_PROFILE "
        f"which is empty for pool slots → case 1 (silently skipped) → "
        f"apparmor.py:413-449 escape vector reopened.  "
        f"Post-PR-102 the matrix reads envelope.profile_name which "
        f"carries the per-session profile."
    )


def test_pool_slot_with_apparmor_off_still_skips_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pool slot with operator opt-out (profile_name empty in envelope)
    → install skipped, no BootstrapError.  Pin that PR 102's fix
    doesn't accidentally make unconfined pool sessions fail loudly."""
    monkeypatch.delenv("JAATO_RUNNER_PROFILE", raising=False)
    stub = _StubRuntimeWithSession(with_setter=False)

    host = bootstrap_session(
        _envelope(profile_name=""),
        runtime_factory=lambda env: stub,
    )

    assert host is not None
    assert stub.install_calls == []
