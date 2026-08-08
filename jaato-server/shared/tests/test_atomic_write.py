"""Tests for ``shared.atomic_write`` (Phase 3 §3.14).

The atomic-write helper is the canonical implementation of the
SIGTERM-mid-write durability contract: any persistent state file
written through it survives a process crash at any point during the
write — leaves the target either pre-write or post-write, never
partial / corrupted.

These tests pin the contract:

- Happy-path round-trip (text + JSON).
- Parent directory auto-creation.
- ``SIGTERM-mid-write`` simulation: an exception raised between the
  temp-file write and the os.replace leaves the target unchanged
  AND removes the orphan temp file.
- ``SIGTERM-mid-fsync`` simulation: an exception during the temp-
  file write itself leaves the target unchanged AND removes the
  orphan temp file.
- ``SIGTERM-mid-replace`` simulation: an exception during os.replace
  leaves the target unchanged AND removes the orphan temp file.
- KeyboardInterrupt / SystemExit also clean up the orphan
  (BaseException catch — not just Exception).
- The atomic write does not leak temp files into the parent
  directory after success.
- Concurrent atomic writes to the same path leave a consistent
  final state (one of the two writes wins, no half-state).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from shared.atomic_write import atomic_write_json, atomic_write_text


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_atomic_write_text_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    atomic_write_text(target, "hello\nworld\n")
    assert target.read_text() == "hello\nworld\n"


def test_atomic_write_json_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    payload = {"key": "value", "list": [1, 2, 3], "nested": {"a": True}}
    atomic_write_json(target, payload)
    assert json.loads(target.read_text()) == payload


def test_atomic_write_json_preserves_unicode(tmp_path: Path) -> None:
    """ensure_ascii=False by default → unicode characters survive."""
    target = tmp_path / "unicode.json"
    payload = {"emoji": "★ unicode test ★"}
    atomic_write_json(target, payload)
    raw = target.read_text(encoding="utf-8")
    assert "★" in raw  # not escaped to ★


def test_atomic_write_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeply" / "data.json"
    assert not target.parent.exists()
    atomic_write_json(target, {"created": True})
    assert target.exists()
    assert json.loads(target.read_text()) == {"created": True}


def test_atomic_write_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    atomic_write_json(target, {"version": 1})
    atomic_write_json(target, {"version": 2})
    assert json.loads(target.read_text()) == {"version": 2}


# ----------------------------------------------------------------------
# Crash mid-write — SIGTERM durability contract
# ----------------------------------------------------------------------


def test_crash_mid_replace_leaves_target_intact_and_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate SIGTERM-mid-replace by patching ``os.replace`` to
    raise.  The target must still hold its pre-write content (or not
    exist if there was no pre-write content), and no orphan temp
    file must be left in the directory."""
    target = tmp_path / "state.json"
    atomic_write_json(target, {"version": 1})  # pre-write

    # Patch os.replace as imported by the atomic_write module to raise.
    import shared.atomic_write as aw

    def _bad_replace(src: str, dst: str) -> None:
        raise OSError("simulated SIGTERM during rename")

    monkeypatch.setattr(aw.os, "replace", _bad_replace)

    with pytest.raises(OSError, match="simulated SIGTERM"):
        atomic_write_json(target, {"version": 2})

    # Target unchanged.
    assert json.loads(target.read_text()) == {"version": 1}
    # No orphan temp files in the directory.
    leftovers = [
        p for p in tmp_path.iterdir()
        if p.name != "state.json"
    ]
    assert leftovers == [], (
        f"orphan temp files leaked into the persistent directory: "
        f"{leftovers!r}"
    )


def test_crash_during_write_leaves_target_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a crash mid-write (json.dumps inside atomic_write_json
    raises).  No temp file leaks; target unchanged."""
    target = tmp_path / "state.json"
    atomic_write_json(target, {"version": 1})

    import shared.atomic_write as aw

    def _bad_dumps(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("simulated crash during serialization")

    monkeypatch.setattr(aw.json, "dumps", _bad_dumps)

    with pytest.raises(RuntimeError, match="simulated crash"):
        atomic_write_json(target, {"version": 2})

    # Target unchanged.
    assert json.loads(target.read_text()) == {"version": 1}
    leftovers = [
        p for p in tmp_path.iterdir()
        if p.name != "state.json"
    ]
    assert leftovers == [], f"temp files leaked: {leftovers!r}"


def test_keyboard_interrupt_during_write_cleans_up_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cleanup path uses ``except BaseException`` so KeyboardInterrupt
    (Ctrl-C) and SystemExit (sys.exit) also clean up the temp file.
    Without this, a Ctrl-C mid-write leaves a turd in the persistent
    directory."""
    target = tmp_path / "state.json"
    import shared.atomic_write as aw

    def _bad_replace(src: str, dst: str) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(aw.os, "replace", _bad_replace)

    with pytest.raises(KeyboardInterrupt):
        atomic_write_json(target, {"version": 1})

    assert not target.exists()
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [], f"temp files leaked: {leftovers!r}"


def test_system_exit_during_write_cleans_up_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.json"
    import shared.atomic_write as aw

    def _bad_replace(src: str, dst: str) -> None:
        raise SystemExit(1)

    monkeypatch.setattr(aw.os, "replace", _bad_replace)

    with pytest.raises(SystemExit):
        atomic_write_json(target, {"version": 1})

    assert not target.exists()
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [], f"temp files leaked: {leftovers!r}"


# ----------------------------------------------------------------------
# Cleanliness — no temp leaks on success
# ----------------------------------------------------------------------


def test_successful_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"version": 1})
    contents = sorted(p.name for p in tmp_path.iterdir())
    assert contents == ["state.json"]


def test_repeated_writes_leave_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    for i in range(50):
        atomic_write_json(target, {"version": i})
    contents = sorted(p.name for p in tmp_path.iterdir())
    assert contents == ["state.json"]
    assert json.loads(target.read_text()) == {"version": 49}


# ----------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------


def test_concurrent_writes_yield_consistent_final_state(
    tmp_path: Path,
) -> None:
    """Two concurrent atomic writes to the same path: one wins, the
    other is overwritten.  The final file is always one of the two
    payloads — never half-state."""
    target = tmp_path / "race.json"

    payloads = [{"writer": i, "data": list(range(100))} for i in range(20)]
    threads = []
    for p in payloads:
        t = threading.Thread(
            target=lambda payload=p: atomic_write_json(target, payload),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    # Final file must parse as one of the payloads.
    final = json.loads(target.read_text())
    assert final in payloads, (
        "concurrent writes left an unexpected payload — race may have "
        "produced a half-state file"
    )
    # No orphan temp files.
    contents = sorted(p.name for p in tmp_path.iterdir())
    assert contents == ["race.json"]


# ----------------------------------------------------------------------
# fsync flag
# ----------------------------------------------------------------------


def test_fsync_disabled_still_writes_correctly(tmp_path: Path) -> None:
    """fsync=False is for high-frequency writes where post-crash
    data loss is acceptable.  Atomicity (no half-state) is
    preserved either way; only durability differs."""
    target = tmp_path / "data.txt"
    atomic_write_text(target, "no-fsync content", fsync=False)
    assert target.read_text() == "no-fsync content"
