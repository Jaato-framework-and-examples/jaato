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
