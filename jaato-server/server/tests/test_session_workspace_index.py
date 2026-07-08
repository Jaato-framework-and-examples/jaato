"""SessionWorkspaceIndex — the daemon-owned session_id → workspace map used to
revive a cold session by id alone for the wake primitive.

Pins: record/resolve, unknown → None, persistence round-trip across instances,
cross-workspace id collision → fail-loud (ambiguous → None, never wrong), and
corrupt-file tolerance.
"""

import json

import pytest

from server.session_workspace_index import SessionWorkspaceIndex


def _idx(tmp_path):
    return SessionWorkspaceIndex(path=tmp_path / "index.json")


def test_record_and_resolve(tmp_path):
    idx = _idx(tmp_path)
    idx.record("20260704_120000", "/ws/a")
    assert idx.resolve("20260704_120000") == "/ws/a"


def test_unknown_id_resolves_none(tmp_path):
    assert _idx(tmp_path).resolve("nope") is None


def test_re_record_same_workspace_is_not_ambiguous(tmp_path):
    idx = _idx(tmp_path)
    idx.record("s1", "/ws/a")
    idx.record("s1", "/ws/a")  # same session re-saved — normal
    assert idx.resolve("s1") == "/ws/a"


def test_cross_workspace_collision_is_ambiguous_and_refused(tmp_path):
    idx = _idx(tmp_path)
    idx.record("20260704_120000", "/ws/a")
    idx.record("20260704_120000", "/ws/b")  # same second, different workspace
    # Fail-loud: refuse rather than pick wrong.
    assert idx.resolve("20260704_120000") is None


def test_persistence_round_trip(tmp_path):
    idx = _idx(tmp_path)
    idx.record("s1", "/ws/a")
    idx.record("dup", "/ws/a")
    idx.record("dup", "/ws/b")  # make it ambiguous
    # A fresh instance reading the same file sees both the map and ambiguity.
    idx2 = SessionWorkspaceIndex(path=tmp_path / "index.json")
    assert idx2.resolve("s1") == "/ws/a"
    assert idx2.resolve("dup") is None  # ambiguity persisted


def test_empty_inputs_ignored(tmp_path):
    idx = _idx(tmp_path)
    idx.record("", "/ws/a")
    idx.record("s1", "")
    assert idx.resolve("") is None
    assert idx.resolve("s1") is None


def test_workspaces_returns_distinct_paths(tmp_path):
    # Used by list_sessions to surface cold provisioned-workspace sessions.
    idx = _idx(tmp_path)
    assert idx.workspaces() == set()
    idx.record("s1", "/ws/a")
    idx.record("s2", "/ws/a")   # same workspace, two sessions → one path
    idx.record("s3", "/ws/b")
    assert idx.workspaces() == {"/ws/a", "/ws/b"}
    idx.forget("s3")
    assert idx.workspaces() == {"/ws/a"}


def test_corrupt_file_starts_empty_not_crash(tmp_path):
    p = tmp_path / "index.json"
    p.write_text("{ this is not json", encoding="utf-8")
    idx = SessionWorkspaceIndex(path=p)  # must not raise
    assert idx.resolve("anything") is None
    # And it can still record afterwards (rebuilds).
    idx.record("s1", "/ws/a")
    assert idx.resolve("s1") == "/ws/a"


def test_record_unchanged_skips_disk_write(tmp_path):
    # record() runs on every _save_session; an unchanged mapping must not
    # rewrite the file (avoid pointless I/O on the hot save path).
    idx = SessionWorkspaceIndex(path=tmp_path / "index.json")
    idx.record("s1", "/ws/a")
    writes = {"n": 0}
    orig = idx._save_locked
    def _counting():
        writes["n"] += 1
        return orig()
    idx._save_locked = _counting
    idx.record("s1", "/ws/a")   # identical mapping — no write
    assert writes["n"] == 0
    idx.record("s1", "/ws/b")   # changed (→ ambiguous) — writes once
    assert writes["n"] == 1


def test_atomic_write_produces_valid_json(tmp_path):
    p = tmp_path / "index.json"
    idx = SessionWorkspaceIndex(path=p)
    idx.record("s1", "/ws/a")
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["map"]["s1"] == "/ws/a"
    assert on_disk["ambiguous"] == []


def test_forget_drops_mapping(tmp_path):
    ix = _idx(tmp_path)
    ix.record("s1", "/ws/a")
    ix.forget("s1")
    assert ix.resolve("s1") is None
    ix.forget("s1")                         # idempotent — no error on unknown id


def test_forget_clears_ambiguous_mark(tmp_path):
    ix = _idx(tmp_path)
    ix.record("dup", "/ws/a")
    ix.record("dup", "/ws/b")               # → ambiguous
    assert ix.resolve("dup") is None        # refused while ambiguous
    ix.forget("dup")
    # ambiguity cleared: the id is fully forgotten and re-recordable clean
    ix.record("dup", "/ws/c")
    assert ix.resolve("dup") == "/ws/c"
