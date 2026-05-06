"""Tests for the inotify pressure soft-warn (server 0.6.54+).

Pin the contract that:
- Below the 80% threshold, ``check_inotify_pressure`` returns None.
- At/above the threshold, returns a WARNING message naming used/limit.
- On non-Linux (or unreadable procfs), returns None gracefully.
- Counter logic only counts inotify symlinks, not arbitrary FDs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from server.health_check import (
    _count_inotify_instances,
    _read_inotify_limit,
    check_inotify_pressure,
)


def test_returns_none_when_limit_unreadable(monkeypatch):
    """Non-Linux / procfs missing → None (silent no-op)."""
    monkeypatch.setattr(
        "server.health_check._read_inotify_limit", lambda: None,
    )
    assert check_inotify_pressure() is None


def test_returns_none_below_threshold(monkeypatch):
    """Usage at 50% of limit (well under 80%) → None."""
    monkeypatch.setattr(
        "server.health_check._read_inotify_limit", lambda: 100,
    )
    monkeypatch.setattr(
        "server.health_check._count_inotify_instances", lambda: 50,
    )
    assert check_inotify_pressure() is None


def test_returns_none_at_exactly_80_pct(monkeypatch):
    """At exactly 80% (boundary), still under the strict-greater-than
    check — no warning yet.  The threshold semantics are "WARN when
    we cross 80%", not "WARN at 80%".
    """
    monkeypatch.setattr(
        "server.health_check._read_inotify_limit", lambda: 100,
    )
    monkeypatch.setattr(
        "server.health_check._count_inotify_instances", lambda: 80,
    )
    assert check_inotify_pressure() is None


def test_returns_warning_above_threshold(monkeypatch):
    """Usage at 85% → warning string with current/limit counts."""
    monkeypatch.setattr(
        "server.health_check._read_inotify_limit", lambda: 100,
    )
    monkeypatch.setattr(
        "server.health_check._count_inotify_instances", lambda: 85,
    )
    msg = check_inotify_pressure()
    assert msg is not None
    assert "85/100" in msg
    assert "85%" in msg
    # Must mention the actionable knob
    assert "max_user_instances" in msg


def test_returns_warning_near_full(monkeypatch):
    """Usage at 99% → warning, percentage rounded correctly."""
    monkeypatch.setattr(
        "server.health_check._read_inotify_limit", lambda: 100,
    )
    monkeypatch.setattr(
        "server.health_check._count_inotify_instances", lambda: 99,
    )
    msg = check_inotify_pressure()
    assert msg is not None
    assert "99/100" in msg


def test_zero_or_negative_limit_treated_as_disabled(monkeypatch):
    """Limit reads of 0 or -1 (kernel pseudo-file edge cases) → None.

    Avoids divide-by-zero and zero-threshold spurious warnings.
    """
    for bad_limit in (0, -1):
        monkeypatch.setattr(
            "server.health_check._read_inotify_limit", lambda: bad_limit,
        )
        monkeypatch.setattr(
            "server.health_check._count_inotify_instances", lambda: 50,
        )
        assert check_inotify_pressure() is None, (
            f"limit={bad_limit} should disable the check"
        )


def test_count_inotify_handles_proc_unavailable(monkeypatch):
    """``os.listdir('/proc/self/fd')`` raises on non-Linux → 0 (graceful)."""
    def _raise(*args, **kw):
        raise OSError("not Linux")

    monkeypatch.setattr("os.listdir", _raise)
    assert _count_inotify_instances() == 0


def test_count_inotify_handles_unreadable_fd(monkeypatch, tmp_path):
    """An FD whose readlink raises (EACCES, gone) → skipped, not aborted."""
    fake_proc_fd = tmp_path / "proc_self_fd"
    fake_proc_fd.mkdir()
    # Two fake FDs: one inotify symlink, one that "fails" on readlink
    (fake_proc_fd / "1").symlink_to("/dev/null")
    (fake_proc_fd / "2").symlink_to("anon_inode:inotify")

    def fake_listdir(path):
        if path == "/proc/self/fd":
            return ["1", "2"]
        return []

    real_readlink = __import__("os").readlink

    def fake_readlink(path):
        if path == "/proc/self/fd/1":
            raise OSError("simulated failure")
        if path == "/proc/self/fd/2":
            return "anon_inode:inotify"
        return real_readlink(path)

    monkeypatch.setattr("os.listdir", fake_listdir)
    monkeypatch.setattr("os.readlink", fake_readlink)
    assert _count_inotify_instances() == 1
