"""A memoised session header must not outlive the bytes it was read from (#1137).

``FileSessionPlugin.list_sessions`` parses every record WHOLE to keep the
nine-scalar header ``deserialize_session_info`` returns, and discards the
transcript -- which is the rest of the file.  So a listing costs O(transcript),
and #1137 measured 1.43 s for 50 sessions of 900 turns.  A listing is asked
for far more often than a record changes (the web picker calls ``session.list``
on open and ``sessionListSilent`` refreshes it in the background, and
``SessionManager.list_sessions`` does that per known workspace), so the parse
is now memoised against a ``stat`` of the file it came from.

**A cache is a correctness claim, and this file is about the claim rather than
the speed.**  The whole value of the memo is that it answers without looking at
the record; the whole risk is that the record has changed and it answers anyway.
The listing is what a person picks a session from, so a permanently stale row
is not a cosmetic fault -- it is a session showing someone else's description,
or a turn count that stopped moving, with nothing anywhere saying so.

THE CONTRACT, in four parts:

1.  **A hit is indistinguishable from a parse.**  Every field of
    :class:`SessionInfo`, not a subset somebody chose -- the set is read from
    the dataclass, so a field added later is covered without anyone
    remembering to add it here.  ``cascade_driver_id`` and ``sibling_name``
    are the two that matter most: :class:`SessionInfo` documents them as what
    keeps a COLD session's address visible, so a listing is the one place
    they are ever read.

2.  **A record that changed is re-read.**  The stamp is
    ``(mtime_ns, size, inode)``, and that is still not sufficient: a
    filesystem with coarse ``st_mtime`` granularity can report the same mtime
    for a write that lands *after* the read that filled the memo.  So an entry
    read within a second of its record's own mtime is not trusted.  Without
    that rule a same-tick rewrite of equal size is served from the memo
    forever, because a session that is never saved again never produces a new
    stamp to invalidate it.

3.  **The memo is bounded by what is on disk.**  A deleted session stops being
    remembered on the first listing after its deletion.  Unbounded would make
    a long-lived daemon accumulate an entry per session that ever existed.

4.  **A failed parse is remembered as a failure, not as an absence.**  A
    corrupt record was re-read and re-warned about on every poll.  "Not
    cached" and "cached, and the answer is that it cannot be used" are
    different facts, which is why the lookup has a MISS sentinel rather than
    returning ``None`` for both.

WHAT THIS DOES NOT TEST, deliberately: how fast anything is.  A timing
assertion here would be flaky on a shared runner and would not be the
property worth pinning -- the measurements live in the PR.  What IS pinned is
that the memo is being *used*, because every "it is not stale" test below
passes trivially against a memo that never hits, and a test that cannot fail
proves nothing.
"""

import dataclasses
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from shared.plugins.session import listing_cache
from shared.plugins.session.base import SessionInfo, SessionState
from shared.plugins.session.file_session import FileSessionPlugin
from shared.plugins.session.serializer import serialize_session_state
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


_SRC = "jaato-server/shared/plugins/session/listing_cache.py"


REVERSIONS = [
    Reversion(
        target=_SRC,
        find="""        if entry.observed_ns - entry.stamp.mtime_ns <= self.RACY_WINDOW_NS:""",
        replace="""        if False:""",
        test="test_a_same_tick_rewrite_is_not_served_from_the_memo",
        because="a record rewritten within its filesystem's timestamp "
                "granularity being served from the memo for the life of the "
                "daemon, because a session that is never saved again never "
                "produces a new stamp to invalidate the entry",
    ),
    Reversion(
        target=_SRC,
        find="""        return cls(st.st_mtime_ns, st.st_size, st.st_ino)""",
        replace="""        return cls(st.st_mtime_ns, 0, st.st_ino)""",
        test="test_a_rewrite_that_preserves_mtime_is_re_read",
        because="a record replaced by a copy that carried its timestamps "
                "over -- a backup restore, `cp -p`, an archive extraction -- "
                "being served from the memo, because mtime alone cannot see "
                "a write that kept it",
    ),
    Reversion(
        target=_SRC,
        find="""        return cls(st.st_mtime_ns, st.st_size, st.st_ino)""",
        replace="""        return cls(st.st_mtime_ns, st.st_size, 0)""",
        test="test_a_rewrite_onto_a_new_inode_is_re_read",
        because="a record replaced WHOLE -- which is what `save` does, "
                "writing a sibling and renaming it into position -- being "
                "served from the memo whenever the replacement happened to "
                "match on timestamp and size",
    ),
    Reversion(
        target=_SRC,
        find="""            for name in [n for n in entries if n not in keep]:
                del entries[name]""",
        replace="""            pass""",
        test="test_the_memo_is_bounded_by_what_is_on_disk",
        because="a daemon accumulating a memo entry for every session that "
                "ever existed in every workspace it has listed",
    ),
    Reversion(
        target=_SRC,
        find="""        if entry.value is None:
            return None
        return dataclasses.replace(entry.value)""",
        replace="""        if entry.value is None:
            return None
        return entry.value""",
        test="test_a_hit_hands_back_a_copy",
        because="a caller that was served FROM the memo holding the memo's "
                "own object, so its edit becomes the next listing's answer",
    ),
    Reversion(
        target=_SRC,
        find="""            None if value is None else dataclasses.replace(value),""",
        replace="""            value,""",
        test="test_the_memo_does_not_keep_the_parse_it_handed_out",
        because="the memo keeping the very object the PARSE handed its "
                "caller, so an edit to a freshly-listed session becomes the "
                "next listing's answer -- the copy on the way out cannot "
                "see this one, because this reference never went through it",
    ),
]

# A note for whoever reads the two copy reversions above and reaches for
# "one check, one door" (#688): these are two DOORS, not two checks on one.
# A ``SessionInfo`` escapes the memo by two paths -- the parse hands one to
# its caller on the way IN, and a hit hands one out -- and neither copy can
# see the other's reference.  The meta-guard is what establishes that:
# with either copy alone, the test naming the other FAILS.


# --------------------------------------------------------------------------
# Fixtures: real records, written by the real serializer, read by the real
# plugin.  Nothing here fakes the parse -- the memo's whole job is to be
# indistinguishable from it.
# --------------------------------------------------------------------------

def _state(session_id: str, **over) -> SessionState:
    fields = dict(
        session_id=session_id,
        history=[],
        created_at=datetime(2026, 9, 1, 12, 0, 0),
        updated_at=datetime(2026, 9, 1, 12, 30, 0),
        description=f"{session_id} description",
        turn_count=7,
        profile_name="researcher",
        workspace_path="/workspace/project",
        cascade_driver_id="cascade-1",
        sibling_name="stage-a",
    )
    fields.update(over)
    return SessionState(**fields)


def _write(directory: Path, state: SessionState, age_s: float = 3600.0) -> Path:
    """Write a record and backdate it, so it is memoisable by default.

    ``age_s`` is how long ago the record was written.  The default puts it
    well clear of the racy window, which is where a record a picker lists
    actually sits; a test about the window passes ``age_s=0``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{state.session_id}.json"
    path.write_text(json.dumps(serialize_session_state(state)), encoding="utf-8")
    if age_s:
        when = time.time() - age_s
        os.utime(path, (when, when))
    return path


@pytest.fixture
def store(tmp_path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
def plugin() -> FileSessionPlugin:
    return FileSessionPlugin()


def _only(plugin: FileSessionPlugin, store: Path) -> SessionInfo:
    infos = plugin.list_sessions(storage_dir=store)
    assert len(infos) == 1, infos
    return infos[0]


# --------------------------------------------------------------------------
# 0.  The memo is actually used.
#
# Every staleness test below would pass against a memo that never hits, so
# this is the control that makes the rest of the file mean something.
# --------------------------------------------------------------------------

def test_the_memo_is_used_at_all(plugin, store):
    _write(store, _state("20260901_000001"))
    plugin.list_sessions(storage_dir=store)
    assert plugin._listing_cache.stats() == {str(store): 1}

    # The sharper control is that the SECOND listing does not touch the file.
    # Replace the record's bytes with something unparseable while holding its
    # stamp fixed -- same size, same inode, same mtime -- and a memo that is
    # being used still answers, where a re-read would skip the file as corrupt.
    path = store / "20260901_000001.json"
    st = path.stat()
    junk = "{ not json"
    path.write_text(junk + " " * (st.st_size - len(junk)), encoding="utf-8")
    os.utime(path, ns=(st.st_mtime_ns, st.st_mtime_ns))

    after = path.stat()
    assert after.st_size == st.st_size, "fixture failed to hold size equal"
    assert after.st_ino == st.st_ino, "fixture failed to hold the inode"
    assert after.st_mtime_ns == st.st_mtime_ns, "fixture failed to pin mtime"

    assert _only(plugin, store).description == "20260901_000001 description"


# --------------------------------------------------------------------------
# 1.  A hit is indistinguishable from a parse.
# --------------------------------------------------------------------------

def test_a_hit_carries_every_field_a_parse_carries(plugin, store):
    """Field-for-field, over the dataclass's own field list.

    Read from :func:`dataclasses.fields` rather than spelled out, so a field
    added to :class:`SessionInfo` later is covered here without anyone
    remembering this file exists.  ``cascade_driver_id`` / ``sibling_name``
    are the live concern: a listing is the only place a COLD session's
    address is read.
    """
    _write(store, _state("20260901_000001"))

    fresh = _only(plugin, store)              # parse
    assert plugin._listing_cache.stats()      # ... and it was memoised
    hit = _only(plugin, store)                # memo

    names = [f.name for f in dataclasses.fields(SessionInfo)]
    assert names, "SessionInfo has no fields -- this test would assert nothing"
    for name in names:
        assert getattr(hit, name) == getattr(fresh, name), name


def test_the_memo_does_not_keep_the_parse_it_handed_out(plugin, store):
    """The way IN.

    The listing that FILLS the memo also returns that parse to its caller.
    Storing that instance would leave the memo holding an object someone
    else already has a reference to.  The copy on the way out cannot help
    here: this reference never passed through it.
    """
    _write(store, _state("20260901_000001"))

    parsed = _only(plugin, store)                 # the parse, memoised
    parsed.description = "mutated by the first caller"

    assert _only(plugin, store).description == "20260901_000001 description"


def test_a_hit_hands_back_a_copy(plugin, store):
    """The way OUT.

    A caller served FROM the memo must not be holding the memo's object
    either.  Distinct from the test above: this reference is handed out by
    the lookup, so the copy on the way in has already happened and cannot
    protect against it.
    """
    _write(store, _state("20260901_000001"))

    plugin.list_sessions(storage_dir=store)       # fill the memo
    hit = _only(plugin, store)                    # served from it
    hit.description = "mutated by a later caller"

    assert _only(plugin, store).description == "20260901_000001 description"


def test_listing_order_and_membership_are_unchanged(plugin, store):
    """The memo changes where a header comes from, not what a listing is."""
    _write(store, _state("20260901_000001",
                         updated_at=datetime(2026, 9, 1, 10, 0, 0)))
    _write(store, _state("20260901_000002",
                         updated_at=datetime(2026, 9, 1, 12, 0, 0)))
    _write(store, _state("20260901_000003",
                         updated_at=datetime(2026, 9, 1, 11, 0, 0)))

    expected = ["20260901_000002", "20260901_000003", "20260901_000001"]
    assert [i.session_id for i in plugin.list_sessions(storage_dir=store)] == expected
    assert [i.session_id for i in plugin.list_sessions(storage_dir=store)] == expected


def test_two_workspaces_do_not_answer_for_each_other(plugin, tmp_path):
    """One plugin instance serves every workspace the daemon knows."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, _state("20260901_000001", description="in a"))
    _write(b, _state("20260901_000001", description="in b"))

    assert _only(plugin, a).description == "in a"
    assert _only(plugin, b).description == "in b"
    # ... and again, now that both are memoised.
    assert _only(plugin, a).description == "in a"
    assert _only(plugin, b).description == "in b"


# --------------------------------------------------------------------------
# 2.  A record that changed is re-read.
# --------------------------------------------------------------------------

def test_a_rewritten_record_is_re_read(plugin, store):
    """The ordinary case: a session takes a turn and is saved.

    Caught by the mtime alone, which is why the two tests below exist: they
    exercise the stamp's other two components, each on the case where mtime
    cannot see the change.
    """
    _write(store, _state("20260901_000001"))
    assert _only(plugin, store).turn_count == 7

    _write(store, _state("20260901_000001", turn_count=8,
                         description="renamed after a turn"))
    again = _only(plugin, store)
    assert again.turn_count == 8
    assert again.description == "renamed after a turn"


def _rewrite_holding_mtime(path: Path, state: SessionState) -> os.stat_result:
    """Replace a record's bytes in place and put its mtime back.

    What a backup restore, ``cp -p`` or an archive extraction does: the
    content is new and the timestamp is the old one, so mtime cannot see it.
    """
    before = path.stat()
    path.write_text(json.dumps(serialize_session_state(state)), encoding="utf-8")
    os.utime(path, ns=(before.st_mtime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_mtime_ns == before.st_mtime_ns, "fixture failed to pin mtime"
    return before


def test_a_rewrite_that_preserves_mtime_is_re_read(plugin, store):
    """Only ``size`` can see this one.

    The record is settled (an hour old), so the racy-window rule trusts the
    entry, and the mtime is put back, so mtime matches.  The rewrite changes
    the description's LENGTH, so the file's size is the only component of
    the stamp that moved.
    """
    path = _write(store, _state("20260901_000001", description="short"))
    assert _only(plugin, store).description == "short"

    before = _rewrite_holding_mtime(
        path, _state("20260901_000001", description="a considerably longer "
                                                    "description than before"))
    assert path.stat().st_size != before.st_size, "fixture failed to move size"
    assert path.stat().st_ino == before.st_ino, "fixture unexpectedly moved inode"

    assert _only(plugin, store).description == ("a considerably longer "
                                                "description than before")


def test_a_rewrite_onto_a_new_inode_is_re_read(plugin, store):
    """Only ``inode`` can see this one.

    Modelled on what :meth:`FileSessionPlugin.save` does -- write a sibling,
    ``os.replace`` it into position -- with the timestamp and the size held
    equal so that the other two components of the stamp match.  On this tree
    every save takes that path, so the inode is what makes the stamp notice
    a save whatever the filesystem does with timestamps.
    """
    path = _write(store, _state("20260901_000001", description="A" * 30))
    assert _only(plugin, store).description == "A" * 30

    before = path.stat()
    replacement = path.with_suffix(".json.tmp")
    replacement.write_text(
        json.dumps(serialize_session_state(
            _state("20260901_000001", description="B" * 30))),
        encoding="utf-8")
    os.replace(replacement, path)
    os.utime(path, ns=(before.st_mtime_ns, before.st_mtime_ns))

    after = path.stat()
    assert after.st_size == before.st_size, "fixture failed to hold size equal"
    assert after.st_mtime_ns == before.st_mtime_ns, "fixture failed to pin mtime"
    if after.st_ino == before.st_ino:
        pytest.skip("filesystem reused the inode; this test cannot isolate it")

    assert _only(plugin, store).description == "B" * 30


def test_a_same_tick_rewrite_is_not_served_from_the_memo(plugin, store):
    """The stamp cannot see this rewrite; the racy-window rule can.

    The record is written, listed, then rewritten IN PLACE to an identical
    size with its mtime forced back to the same value -- so
    ``(mtime_ns, size, inode)`` is byte-identical across a change of content.
    That is what a coarse-granularity filesystem does to two writes in the
    same second, and it is the case where a stamp alone serves a header the
    record no longer has, permanently, for a session that is never saved
    again.
    """
    # ``age_s=0``: written now, so the entry lands inside the racy window.
    path = _write(store, _state("20260901_000001", description="A" * 40),
                  age_s=0)
    pinned = path.stat()

    assert _only(plugin, store).description == "A" * 40

    rewritten = _state("20260901_000001", description="B" * 40)
    path.write_text(json.dumps(serialize_session_state(rewritten)),
                    encoding="utf-8")
    os.utime(path, ns=(pinned.st_mtime_ns, pinned.st_mtime_ns))

    after = path.stat()
    assert after.st_size == pinned.st_size, "fixture failed to hold size equal"
    assert after.st_ino == pinned.st_ino, "fixture failed to hold the inode"
    assert after.st_mtime_ns == pinned.st_mtime_ns, "fixture failed to pin mtime"

    assert _only(plugin, store).description == "B" * 40


def test_a_settled_record_is_memoised_although_a_fresh_one_is_not(plugin, store):
    """The window is a window, not a switch that turned the memo off.

    Paired with the test above deliberately: that one shows a fresh record is
    re-read, and this one shows the memo still works for the records a picker
    is actually listing, so the fix is not "never cache anything".
    """
    _write(store, _state("20260901_000001"), age_s=0)     # fresh
    _write(store, _state("20260901_000002"), age_s=3600)  # settled

    plugin.list_sessions(storage_dir=store)

    fresh_st = (store / "20260901_000001.json").stat()
    settled_st = (store / "20260901_000002.json").stat()
    cache = plugin._listing_cache
    d = str(store)
    assert cache.lookup(d, "20260901_000001.json", fresh_st) is listing_cache.MISS
    assert cache.lookup(d, "20260901_000002.json", settled_st) is not listing_cache.MISS


def test_a_deleted_record_leaves_the_listing(plugin, store):
    _write(store, _state("20260901_000001"))
    _write(store, _state("20260901_000002"))
    assert len(plugin.list_sessions(storage_dir=store)) == 2

    (store / "20260901_000001.json").unlink()
    assert [i.session_id for i in plugin.list_sessions(storage_dir=store)] == [
        "20260901_000002"]


def test_a_record_that_vanishes_mid_listing_is_skipped(plugin, store, monkeypatch):
    """A concurrent ``delete`` must not take the listing down.

    ``glob`` then ``stat`` is two syscalls with a gap, and ``session.delete``
    runs on another thread.
    """
    _write(store, _state("20260901_000001"))
    _write(store, _state("20260901_000002"))

    real_stat = Path.stat

    def vanishing(self, *a, **kw):
        if self.name == "20260901_000001.json":
            raise FileNotFoundError(self)
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", vanishing)
    assert [i.session_id for i in plugin.list_sessions(storage_dir=store)] == [
        "20260901_000002"]


# --------------------------------------------------------------------------
# 3.  The memo is bounded by what is on disk.
# --------------------------------------------------------------------------

def test_the_memo_is_bounded_by_what_is_on_disk(plugin, store):
    for n in range(5):
        _write(store, _state(f"2026090{n}_00000{n}"))
    plugin.list_sessions(storage_dir=store)
    assert plugin._listing_cache.stats()[str(store)] == 5

    for n in range(3):
        (store / f"2026090{n}_00000{n}.json").unlink()
    plugin.list_sessions(storage_dir=store)
    assert plugin._listing_cache.stats()[str(store)] == 2


def test_pruning_one_directory_leaves_the_others_alone(plugin, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, _state("20260901_000001"))
    _write(b, _state("20260901_000002"))
    plugin.list_sessions(storage_dir=a)
    plugin.list_sessions(storage_dir=b)

    (a / "20260901_000001.json").unlink()
    plugin.list_sessions(storage_dir=a)

    assert plugin._listing_cache.stats() == {str(b): 1}


def test_shutdown_forgets_everything(plugin, store):
    _write(store, _state("20260901_000001"))
    plugin.list_sessions(storage_dir=store)
    assert plugin._listing_cache.stats()

    plugin.shutdown()
    assert plugin._listing_cache.stats() == {}


def test_the_memo_survives_the_cascade_stage_boundary(plugin, store):
    """``reset_for_next_session`` clears per-SESSION state; this is not that.

    The memo describes the records on disk, and the next stage of a cascade
    lists the same directory -- so clearing it would make every stage
    boundary pay a cold listing for headers that had not moved.
    """
    _write(store, _state("20260901_000001"))
    plugin.list_sessions(storage_dir=store)

    plugin.reset_for_next_session()
    assert plugin._listing_cache.stats() == {str(store): 1}


# --------------------------------------------------------------------------
# 4.  A failed parse is remembered as a failure, not as an absence.
# --------------------------------------------------------------------------

def test_a_corrupt_record_is_skipped_and_remembered(plugin, store, capsys):
    _write(store, _state("20260901_000001"))
    store.mkdir(parents=True, exist_ok=True)
    bad = store / "20260901_000002.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    when = time.time() - 3600
    os.utime(bad, (when, when))

    assert [i.session_id for i in plugin.list_sessions(storage_dir=store)] == [
        "20260901_000001"]
    first = capsys.readouterr().out
    assert "corrupted session file" in first

    # Second listing: still skipped, and no longer re-read, so the warning
    # is not repeated once per poll for the life of the daemon.
    assert [i.session_id for i in plugin.list_sessions(storage_dir=store)] == [
        "20260901_000001"]
    assert "corrupted session file" not in capsys.readouterr().out

    st = bad.stat()
    assert plugin._listing_cache.lookup(str(store), bad.name, st) is None


def test_a_corrupt_record_that_is_repaired_comes_back(plugin, store):
    """A remembered failure is about the BYTES, not about the file name."""
    store.mkdir(parents=True, exist_ok=True)
    bad = store / "20260901_000001.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    assert plugin.list_sessions(storage_dir=store) == []

    _write(store, _state("20260901_000001"))
    assert _only(plugin, store).session_id == "20260901_000001"


def test_a_record_missing_a_required_key_is_treated_as_corrupt(plugin, store):
    """``deserialize_session_info`` raises ``KeyError`` without ``session_id``."""
    store.mkdir(parents=True, exist_ok=True)
    (store / "20260901_000001.json").write_text(
        json.dumps({"description": "no id here"}), encoding="utf-8")
    assert plugin.list_sessions(storage_dir=store) == []


# --------------------------------------------------------------------------
# The cache object itself, at the seams the plugin does not exercise.
# --------------------------------------------------------------------------

def test_miss_is_not_none(store):
    """"Nothing is known" and "known to be unusable" must stay distinguishable."""
    assert listing_cache.MISS is not None
    cache = listing_cache.SessionListingCache()
    _write(store, _state("20260901_000001"))
    st = (store / "20260901_000001.json").stat()
    assert cache.lookup(str(store), "20260901_000001.json", st) is listing_cache.MISS

    cache.store(str(store), "20260901_000001.json", st,
                listing_cache.now_ns(), None)
    assert cache.lookup(str(store), "20260901_000001.json", st) is None


def test_retain_on_an_unknown_directory_is_a_no_op():
    cache = listing_cache.SessionListingCache()
    cache.retain("/nowhere", [])
    assert cache.stats() == {}
