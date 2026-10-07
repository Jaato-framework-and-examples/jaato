"""``server.session_inbox`` -- the durable per-session inbox (phase 2).

The module is stdlib-only and pure disk: these cases pin the on-disk
contract the session manager builds on, so a change to the layout or the
envelope is caught here rather than three layers up.
"""

import base64
import json

from jaato_server.server import session_inbox as inbox
from jaato_server.server.session_inbox import InboxEntry


def _entry(sid="s-1", mid="m-1", *, created_at=1.0, expires_at=9e12, **kw):
    return InboxEntry(message_id=mid, session_id=sid, kind="peer",
                      source="peer:s-a", source_id="s-a", text="hi",
                      created_at=created_at, expires_at=expires_at, **kw)


def test_the_inbox_sits_beside_the_record_and_is_invisible_to_its_glob(tmp_path):
    """The record listing globs ``*.json`` at the top of the storage dir;
    the inbox is a DIRECTORY there, so it never reads as a session."""
    inbox.spool(tmp_path, _entry())
    assert (tmp_path / "s-1.inbox" / "m-1.json").is_file()
    assert [p.name for p in tmp_path.glob("*.json")] == []


def test_bytes_are_spooled_as_files_and_the_envelope_carries_a_manifest(tmp_path):
    att = [{"mime_type": "audio/wav", "data": base64.b64encode(b"\x00\xff").decode(),
            "display_name": "../q.wav", "attachment_id": "sha256:1"}]
    entry = inbox.spool(tmp_path, _entry(), att)
    [row] = entry.attachments
    assert row["mime_type"] == "audio/wav" and row["attachment_id"] == "sha256:1"
    assert row["file"] == "m-1/00-q.wav", "the display name is a basename, never a path"
    assert (tmp_path / "s-1.inbox" / row["file"]).read_bytes() == b"\x00\xff"
    envelope = json.loads((tmp_path / "s-1.inbox" / "m-1.json").read_text())
    assert "data" not in json.dumps(envelope), "the envelope never carries the bytes"
    assert envelope["version"] == 1


def test_load_attachments_reinflates_the_wire_shape(tmp_path):
    att = [{"mime_type": "image/png", "data": base64.b64encode(b"png").decode(),
            "display_name": "a.png", "attachment_id": "sha256:2"}]
    entry = inbox.spool(tmp_path, _entry(), att)
    assert inbox.load_attachments(tmp_path, entry) == att


def test_a_vanished_file_is_skipped_not_fatal(tmp_path):
    att = [{"mime_type": "image/png", "data": "cG5n", "display_name": "a.png"}]
    entry = inbox.spool(tmp_path, _entry(), att)
    (tmp_path / "s-1.inbox" / entry.attachments[0]["file"]).unlink()
    assert inbox.load_attachments(tmp_path, entry) == []


def test_pending_is_oldest_first_and_count_needs_no_parse(tmp_path):
    inbox.spool(tmp_path, _entry(mid="b", created_at=2.0))
    inbox.spool(tmp_path, _entry(mid="a", created_at=1.0))
    inbox.spool(tmp_path, _entry(mid="c", created_at=2.0))
    assert [e.message_id for e in inbox.pending(tmp_path, "s-1")] == ["a", "b", "c"]
    assert inbox.count_pending(tmp_path, "s-1") == 3
    assert inbox.count_pending(tmp_path, "nobody") == 0


def test_an_unreadable_envelope_is_skipped_and_left_in_place(tmp_path):
    inbox.spool(tmp_path, _entry(mid="good"))
    (tmp_path / "s-1.inbox" / "bad.json").write_text("{not json")
    assert [e.message_id for e in inbox.pending(tmp_path, "s-1")] == ["good"]
    assert (tmp_path / "s-1.inbox" / "bad.json").exists(), "never deleted"


def test_remove_takes_the_bytes_and_the_directory_when_empty(tmp_path):
    att = [{"mime_type": "image/png", "data": "cG5n", "display_name": "a.png"}]
    e1 = inbox.spool(tmp_path, _entry(mid="1"), att)
    e2 = inbox.spool(tmp_path, _entry(mid="2"))
    inbox.remove(tmp_path, e1)
    assert not (tmp_path / "s-1.inbox" / "1").exists()
    assert (tmp_path / "s-1.inbox").is_dir()
    inbox.remove(tmp_path, e2)
    assert not (tmp_path / "s-1.inbox").exists()
    inbox.remove(tmp_path, e2)  # idempotent


def test_remove_all_and_find_event_id(tmp_path):
    inbox.spool(tmp_path, _entry(mid="1", event_id="e-1"))
    inbox.spool(tmp_path, _entry(mid="2"))
    assert inbox.find_event_id(tmp_path, "s-1", "e-1").message_id == "1"
    assert inbox.find_event_id(tmp_path, "s-1", "e-9") is None
    assert inbox.find_event_id(tmp_path, "s-1", None) is None
    inbox.remove_all(tmp_path, "s-1")
    assert inbox.pending(tmp_path, "s-1") == []


def test_update_rewrites_in_place_and_round_trips_every_field(tmp_path):
    entry = inbox.spool(tmp_path, _entry(defer_until_client=True, wake_ref="pr#1",
                                         cascade_driver_id="cid", event_id="e"))
    entry.attempts, entry.last_error, entry.runner_queued = 2, "busy", True
    inbox.update(tmp_path, entry)
    [back] = inbox.pending(tmp_path, "s-1")
    assert back == entry


def test_expired_is_judged_against_a_given_instant():
    assert _entry(expires_at=10.0).expired(now=10.0) is True
    assert _entry(expires_at=10.0).expired(now=9.9) is False


def test_no_half_written_envelope_survives(tmp_path):
    """Every write is a temp file plus ``os.replace``: after a spool the
    directory holds the envelope and nothing else."""
    inbox.spool(tmp_path, _entry())
    assert sorted(p.name for p in (tmp_path / "s-1.inbox").iterdir()) == ["m-1.json"]


def test_the_module_is_stdlib_only():
    """It is read from the daemon's listing path and must import nothing
    that path cannot -- the ``session_groups`` rule."""
    import ast
    import pathlib
    import sys
    source = pathlib.Path(inbox.__file__).read_text()
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add((node.module or "").split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    assert names <= stdlib, names - stdlib
