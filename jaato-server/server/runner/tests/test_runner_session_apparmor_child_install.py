"""Tests for the Phase 5 §5.10c AppArmor //child transition install
in ``bootstrap_session`` (per peer review of e805e4d0).

Pins the audible-failure contract for the install step:

- ``JAATO_RUNNER_PROFILE`` empty  → install skipped silently (operator
  opted out of confinement via ``JAATO_RUNNER_DISABLE_CONFINE=1``).
- ``JAATO_RUNNER_PROFILE`` set    → install MUST succeed or
  bootstrap MUST raise ``BootstrapError("configure", ...)``.  A
  silent install failure would leave the session running with the
  escape vector open — exactly the gap §5.10 closes.

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
                    # ``hasattr`` returns True; track installs.
                    def _setter(cb: Any) -> None:
                        if stub._raises is not None:
                            raise stub._raises
                        stub.install_calls.append(cb)

                    inner.set_apparmor_child_transition_callback = _setter

        class _StubSession:
            def __init__(inner) -> None:
                inner._executor = _StubExecutor()

        return _StubSession()


def _envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-5-10c",
        workspace_path="/tmp/ws",
        profile_name="jaato-ws-sess-5-10c",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
    )


# ----------------------------------------------------------------------
# Audible-failure pins
# ----------------------------------------------------------------------


def test_bootstrap_raises_when_apparmor_install_fails_with_profile_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin: ``JAATO_RUNNER_PROFILE`` set + executor lacks the setter →
    bootstrap raises ``BootstrapError("configure", ...)``.

    The operator opted into kernel confinement via the env var; a
    silent fallback to today's pre-§5.10c behavior would leave the
    escape vector open.  Matches the §4.3 PR #57 audible-failure
    pattern."""
    monkeypatch.setenv("JAATO_RUNNER_PROFILE", "jaato-ws-sess-5-10c")
    stub = _StubRuntimeWithSession(with_setter=False)

    with pytest.raises(BootstrapError) as exc_info:
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: stub,
        )

    assert exc_info.value.stage == "configure"
    # Error message must explain WHY this is audible (operator opted in).
    msg = exc_info.value.message
    assert "AppArmor //child transition" in msg
    assert "JAATO_RUNNER_PROFILE" in msg
    assert "JAATO_RUNNER_DISABLE_CONFINE" in msg, (
        "error message must point operators at the documented escape "
        "hatch (JAATO_RUNNER_DISABLE_CONFINE=1) so they can opt out "
        "explicitly rather than silently degrading"
    )


def test_bootstrap_raises_when_setter_itself_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin: ``JAATO_RUNNER_PROFILE`` set + setter raises → bootstrap
    raises ``BootstrapError("configure", ...)`` wrapping the
    underlying exception.

    Distinguishes the "setter missing" case (hasattr fails) from the
    "setter present but broken" case.  Both are audible-failure
    territory under the §5.10c contract."""
    monkeypatch.setenv("JAATO_RUNNER_PROFILE", "jaato-ws-sess-5-10c")
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


def test_bootstrap_silent_when_runner_profile_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin: ``JAATO_RUNNER_PROFILE`` empty → install skipped, session
    constructs normally even when the executor lacks the setter.

    This is the JAATO_RUNNER_DISABLE_CONFINE=1 escape hatch: the
    operator explicitly opted OUT of kernel confinement, so the
    §5.10c contract doesn't apply.  Pin protects the escape hatch
    against accidental tightening that would refuse to start
    sessions in dev / test environments."""
    monkeypatch.delenv("JAATO_RUNNER_PROFILE", raising=False)
    stub = _StubRuntimeWithSession(with_setter=False)

    # No exception — install is silently skipped.
    host = bootstrap_session(
        _envelope(),
        runtime_factory=lambda env: stub,
    )

    assert host is not None
    # No setter calls happened (setter wasn't even there).
    assert stub.install_calls == []


def test_bootstrap_logs_info_on_successful_install(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin: happy path — JAATO_RUNNER_PROFILE set + setter present →
    install runs, INFO log records the profile name.

    Operators reviewing runner-side logs see one ``installed AppArmor
    //child transition callback`` line per session, which becomes the
    evidence trail that the escape-vector closure is wired."""
    import logging
    monkeypatch.setenv("JAATO_RUNNER_PROFILE", "jaato-ws-sess-5-10c")
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
        f"missing INFO log for the successful install; "
        f"got {[r.message for r in caplog.records]!r}"
    )
    assert any(
        "jaato-ws-sess-5-10c" in m for m in install_msgs
    ), "profile name must appear in the install log"


# ----------------------------------------------------------------------
# Phase 5 §5.10e — sub-runner skip
# ----------------------------------------------------------------------


def test_install_skipped_when_runner_profile_is_subprofile(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Phase 5 §5.10e: JAATO_RUNNER_PROFILE contains the Audit 6
    sub-profile separator ``//`` → install is skipped silently
    (INFO log).  v15 design intent (PR #67) already drops the
    escape primitive from the sub-profile, so subprocesses inherit
    the no-escape posture by construction — no //child transition
    needed (and not installable: sub-profile lacks writable
    attr/current per the same v15 work).

    Setter MUST NOT be called.  Bootstrap MUST succeed.  INFO log
    MUST explain the skip with the profile name + audit reference."""
    import logging
    monkeypatch.setenv(
        "JAATO_RUNNER_PROFILE",
        "jaato-ws-parent//subagent",
    )
    # Setter present but should never be called.
    stub = _StubRuntimeWithSession(with_setter=True)

    with caplog.at_level(logging.INFO):
        host = bootstrap_session(
            _envelope(),
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


def test_install_skipped_for_subprofile_even_if_setter_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin: the sub-runner skip path bypasses the audible-failure
    contract.  Without this test, a refactor that moves the
    has_setter check ABOVE the sub-profile branch would re-impose
    the audible-failure on sub-runners — breaking isolated subagents
    on hosts that legitimately lack the setter (none today, but
    defensive).

    Setter missing + sub-profile name → no BootstrapError, no setter
    call.  The §5.10c audible-failure contract applies ONLY to main
    runners (case 3 in the three-case matrix)."""
    monkeypatch.setenv(
        "JAATO_RUNNER_PROFILE",
        "jaato-ws-parent//subagent",
    )
    stub = _StubRuntimeWithSession(with_setter=False)

    # No exception — install path is bypassed entirely.
    host = bootstrap_session(
        _envelope(),
        runtime_factory=lambda env: stub,
    )

    assert host is not None
    assert stub.install_calls == []


def test_main_runner_install_path_unchanged_post_5_10e(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin: main runners (profile name lacks ``//``) still run the
    §5.10c install path — happy + audible-failure both preserved.

    This test pins the case-3 matrix branch.  Catches a regression
    where the §5.10e three-case refactor would have accidentally
    skipped install for main runners too."""
    import logging
    monkeypatch.setenv("JAATO_RUNNER_PROFILE", "jaato-ws-main_session")
    stub = _StubRuntimeWithSession(with_setter=True)

    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: stub,
        )

    assert len(stub.install_calls) == 1, (
        "main-runner install must still run after the §5.10e refactor"
    )
    install_msgs = [
        r.message for r in caplog.records
        if "installed AppArmor" in r.message
    ]
    assert install_msgs, (
        "main-runner install must still emit the INFO log"
    )


def test_three_case_matrix_distinguished_in_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Pin: each of the three cases (empty / sub-profile / main)
    emits a DISTINCT INFO log.  Operators inspecting runner logs
    can determine which branch fired without guessing — important
    for incident debugging where the install path is the suspect."""
    import logging

    # Case 1: empty profile.
    monkeypatch.delenv("JAATO_RUNNER_PROFILE", raising=False)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: _StubRuntimeWithSession(
                with_setter=False,
            ),
        )
    case1_msgs = {r.message for r in caplog.records}
    assert any("empty" in m and "unconfined" in m for m in case1_msgs)

    # Case 2: sub-profile.
    monkeypatch.setenv(
        "JAATO_RUNNER_PROFILE", "jaato-ws-parent//subagent",
    )
    caplog.clear()
    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: _StubRuntimeWithSession(
                with_setter=True,
            ),
        )
    case2_msgs = {r.message for r in caplog.records}
    assert any(
        "sub-profile" in m and "skipping" in m for m in case2_msgs
    )

    # Case 3: main runner.
    monkeypatch.setenv("JAATO_RUNNER_PROFILE", "jaato-ws-main")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: _StubRuntimeWithSession(
                with_setter=True,
            ),
        )
    case3_msgs = {r.message for r in caplog.records}
    assert any("installed AppArmor" in m for m in case3_msgs)


def test_sub_profile_template_drops_escape_rules(workspace_root_tmp=None):
    """Smoke test: pin that ``_render_sub_profile``'s body does NOT
    contain uncommented ``change_profile -> unconfined`` or writable
    ``/proc/*/attr/current`` rules.

    Catches accidental re-introduction by future template edits.
    The §5.10e skip-install correctness depends on the sub-profile
    keeping its v15 "no escape primitive" posture — a future PR
    that adds ``w,`` back to the sub-profile would silently
    re-open the gap §4.3.4 + v15 closed.  This test fires before
    that happens."""
    import re
    from server.apparmor import AppArmorManager

    # Use a minimal manager — _render_sub_profile only needs the
    # workspace_root + venv_path attrs the constructor sets.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        ws = Path(tmp) / "workspaces"
        ws.mkdir()
        (ws / "sessions").mkdir()
        mgr = AppArmorManager(
            workspace_root=str(ws),
            venv_path="/usr/local/venv",
            profile_dir=str(Path(tmp) / "profiles"),
        )
        body = mgr._render_sub_profile(
            parent_session_id="parent_x",
            subagent_id="agent_1",
            workspace_path="/workspace",
        )

    # Strip comments so DROP-block documentation doesn't fool the
    # assertion (same approach as the §5.10a //child template test).
    rule_lines = [
        line.strip() for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    # Must NOT carry change_profile -> unconfined (escape primitive).
    for rule in rule_lines:
        assert not re.match(
            r"change_profile\s+->\s+unconfined,$", rule,
        ), (
            f"sub-profile must NOT authorize "
            f"change_profile -> unconfined; got rule: {rule!r}.  "
            f"v15 design intent (PR #67) drops this; §5.10e "
            f"skip-install depends on it staying dropped."
        )

    # Must NOT carry writable /proc/.../attr/current (file-write
    # half of the escape primitive).  Owner-qualified READ is OK
    # (v15 added it for confine_to_profile.read_current_profile).
    for rule in rule_lines:
        # Match `(owner )?/proc/.../attr/current <perm>,` where
        # <perm> contains `w`.
        m = re.match(
            r"(owner\s+)?/proc/[\w*/]+/attr/current\s+([rwklm]+)\s*,$",
            rule,
        )
        if m:
            perms = m.group(2)
            assert "w" not in perms, (
                f"sub-profile must NOT grant write to "
                f"/proc/.../attr/current; got rule: {rule!r}.  "
                f"v15 design intent drops this; §5.10e "
                f"skip-install depends on it staying dropped."
            )
