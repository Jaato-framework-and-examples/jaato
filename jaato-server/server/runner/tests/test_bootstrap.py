"""Tests for ``server.runner.bootstrap`` — runner self-confinement.

Bootstrap is OS-level (writes to /proc, talks to libapparmor) so the
tests use a fake libapparmor (just a Python object exposing an
``aa_change_profile`` callable) and a tmp-file standing in for
``/proc/self/attr/current``.  This keeps the unit tests host-portable
— a real apparmor confinement happens in the §2.6 integration test.
"""

from __future__ import annotations

import ctypes
import errno
from pathlib import Path
from typing import List

import pytest

from server.runner.bootstrap import (
    ConfinementMismatchError,
    _load_libapparmor,
    confine_to_profile,
    read_current_profile,
)


# ----------------------------------------------------------------------
# Fake libapparmor — just enough surface for confine_to_profile.
# ----------------------------------------------------------------------


class _FakeLib:
    """Minimal stand-in for the ctypes-loaded libapparmor."""

    def __init__(
        self,
        rc: int = 0,
        errno_to_set: int = 0,
        record_call_in: List[bytes] = None,
    ) -> None:
        self._rc = rc
        self._errno_to_set = errno_to_set
        self._records = record_call_in if record_call_in is not None else []

        class _AaChangeProfile:
            argtypes = None
            restype = None

            def __init__(_self) -> None:
                pass

            def __call__(_self, profile: bytes) -> int:
                self._records.append(profile)
                if self._errno_to_set:
                    ctypes.set_errno(self._errno_to_set)
                return self._rc

        self.aa_change_profile = _AaChangeProfile()


# ----------------------------------------------------------------------
# read_current_profile
# ----------------------------------------------------------------------


def test_read_current_profile_strips_trailing_newline(tmp_path: Path) -> None:
    p = tmp_path / "current"
    p.write_text("jaato-ws-test (enforce)\n")
    assert read_current_profile(str(p)) == "jaato-ws-test (enforce)"


def test_read_current_profile_unconfined(tmp_path: Path) -> None:
    p = tmp_path / "current"
    p.write_text("unconfined\n")
    assert read_current_profile(str(p)) == "unconfined"


# ----------------------------------------------------------------------
# confine_to_profile — happy path
# ----------------------------------------------------------------------


def test_confine_happy_path(tmp_path: Path) -> None:
    """aa_change_profile returns 0 + /proc agrees → no raise."""
    proc = tmp_path / "current"
    proc.write_text("jaato-ws-test (enforce)\n")
    records: List[bytes] = []
    fake = _FakeLib(rc=0, record_call_in=records)

    confine_to_profile(
        "jaato-ws-test",
        libapparmor=fake,  # type: ignore[arg-type]
        proc_attr_path=str(proc),
    )

    assert records == [b"jaato-ws-test"]


def test_confine_accepts_bare_profile_name_without_enforce_suffix(
    tmp_path: Path,
) -> None:
    """Some kernels report just ``<name>`` without ``(enforce)``."""
    proc = tmp_path / "current"
    proc.write_text("jaato-ws-test\n")
    fake = _FakeLib(rc=0)
    confine_to_profile(
        "jaato-ws-test",
        libapparmor=fake,  # type: ignore[arg-type]
        proc_attr_path=str(proc),
    )


# ----------------------------------------------------------------------
# confine_to_profile — failures
# ----------------------------------------------------------------------


def test_confine_rejects_empty_profile_name() -> None:
    with pytest.raises(RuntimeError, match="profile_name is empty"):
        confine_to_profile("", libapparmor=_FakeLib())  # type: ignore[arg-type]


def test_confine_aa_change_profile_failure_raises_with_errno(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "current"
    proc.write_text("unconfined\n")
    fake = _FakeLib(rc=-1, errno_to_set=errno.EACCES)

    with pytest.raises(RuntimeError) as excinfo:
        confine_to_profile(
            "jaato-ws-test",
            libapparmor=fake,  # type: ignore[arg-type]
            proc_attr_path=str(proc),
        )
    msg = str(excinfo.value)
    assert "aa_change_profile" in msg
    assert "errno=" in msg
    # The kernel-refused-transition hint should be present.
    assert "kernel refused" in msg


def test_confine_silent_failure_raises_mismatch(tmp_path: Path) -> None:
    """aa_change_profile returns 0 but kernel didn't actually transition.

    This is the §6.5 silent-failure scenario — caught by the
    /proc/self/attr/current cross-check, not by the libapparmor return
    value.
    """
    proc = tmp_path / "current"
    proc.write_text("unconfined\n")  # kernel says still unconfined
    fake = _FakeLib(rc=0)             # but the call "succeeded"

    with pytest.raises(ConfinementMismatchError) as excinfo:
        confine_to_profile(
            "jaato-ws-test",
            libapparmor=fake,  # type: ignore[arg-type]
            proc_attr_path=str(proc),
        )
    assert excinfo.value.expected == "jaato-ws-test"
    assert excinfo.value.actual == "unconfined"


def test_confine_wrong_profile_raises_mismatch(tmp_path: Path) -> None:
    """Profile transition lands in a DIFFERENT profile than requested."""
    proc = tmp_path / "current"
    proc.write_text("jaato-ws-someone-else (enforce)\n")
    fake = _FakeLib(rc=0)
    with pytest.raises(ConfinementMismatchError):
        confine_to_profile(
            "jaato-ws-test",
            libapparmor=fake,  # type: ignore[arg-type]
            proc_attr_path=str(proc),
        )


# ----------------------------------------------------------------------
# _load_libapparmor — only the negative case is portable.
# ----------------------------------------------------------------------


def test_load_libapparmor_missing_raises_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither the requested SONAME nor the find_library fallback
    resolves, surface a 'not installed' error — not a generic OSError —
    so the daemon's error message can distinguish 'apparmor missing'
    from 'kernel refused transition'.
    """
    import ctypes.util as _util
    monkeypatch.setattr(_util, "find_library", lambda name: None)

    with pytest.raises(RuntimeError, match="libapparmor not installed"):
        _load_libapparmor("libapparmor-this-does-not-exist.so.999")
