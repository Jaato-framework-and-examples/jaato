"""The index's ``membership`` section (session group messaging).

A COLD session is resolved across workspaces off this row -- "which
sessions does user X own" is a question a per-workspace listing cannot
answer -- so the row is written beside the workspace mapping and must
survive a daemon restart, be dropped with the session, and never break an
older index file.
"""
from jaato_server.server.session_workspace_index import SessionWorkspaceIndex


def _row(**kw):
    base = dict(created_by=None, cascade_driver_id=None, sibling_name=None)
    base.update(kw)
    return base


def test_membership_round_trips_through_disk(tmp_path):
    path = tmp_path / "idx.json"
    idx = SessionWorkspaceIndex(path)
    idx.record("s1", "/ws/a")
    idx.record_membership("s1", created_by="app:alice", cascade_driver_id="c1",
                          sibling_name="alpha")
    again = SessionWorkspaceIndex(path)
    assert again.membership("s1") == _row(created_by="app:alice",
                                          cascade_driver_id="c1",
                                          sibling_name="alpha")
    assert again.members() == {"s1": again.membership("s1")}


def test_an_all_none_row_is_recorded_not_skipped(tmp_path):
    """"No owner, no cascade" is an answer; an ABSENT row is not."""
    idx = SessionWorkspaceIndex(tmp_path / "idx.json")
    idx.record_membership("s1", created_by=None, cascade_driver_id=None,
                          sibling_name=None)
    assert idx.membership("s1") == _row()


def test_empty_strings_are_absent_facts(tmp_path):
    idx = SessionWorkspaceIndex(tmp_path / "idx.json")
    idx.record_membership("s1", created_by="", cascade_driver_id="",
                          sibling_name="")
    assert idx.membership("s1") == _row()


def test_forget_drops_the_row(tmp_path):
    path = tmp_path / "idx.json"
    idx = SessionWorkspaceIndex(path)
    idx.record_membership("s1", created_by="app:alice", cascade_driver_id=None,
                          sibling_name=None)
    idx.forget("s1")
    assert idx.membership("s1") is None
    assert SessionWorkspaceIndex(path).members() == {}


def test_an_unchanged_row_does_not_rewrite_the_file(tmp_path):
    path = tmp_path / "idx.json"
    idx = SessionWorkspaceIndex(path)
    idx.record_membership("s1", created_by="app:alice", cascade_driver_id=None,
                          sibling_name=None)
    before = path.stat().st_mtime_ns
    path.write_bytes(path.read_bytes())          # touch: a rewrite would differ
    stamp = path.stat().st_mtime_ns
    idx.record_membership("s1", created_by="app:alice", cascade_driver_id=None,
                          sibling_name=None)
    assert path.stat().st_mtime_ns == stamp and stamp >= before


def test_a_file_without_the_section_loads_with_no_memberships(tmp_path):
    path = tmp_path / "idx.json"
    path.write_text('{"map": {"s1": "/ws/a"}, "ambiguous": [], "identity": {}}')
    idx = SessionWorkspaceIndex(path)
    assert idx.resolve("s1") == "/ws/a"
    assert idx.members() == {} and idx.membership("s1") is None


def test_membership_is_independent_of_ambiguity(tmp_path):
    """Ambiguity is about the WORKSPACE; the group facts still answer."""
    idx = SessionWorkspaceIndex(tmp_path / "idx.json")
    idx.record("s1", "/ws/a")
    idx.record("s1", "/ws/b")
    idx.record_membership("s1", created_by="app:alice", cascade_driver_id=None,
                          sibling_name=None)
    assert idx.resolve("s1") is None
    assert idx.membership("s1")["created_by"] == "app:alice"
