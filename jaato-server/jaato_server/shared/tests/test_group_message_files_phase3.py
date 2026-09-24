"""Files on a group message -- session group messaging, phase 3 (design §4.5).

Phase 1 carried text and inline bytes; phase 2 made "it will be
processed" a guarantee.  Neither let a peer hand another session a FILE
except by pasting it into the 8 KiB body, and a path in the sender's
workspace is not a usable reference for a runner confined to another one.
Phase 3 adds ``file_refs`` and ``text_attachments`` to the message, and
the rule for each is pinned here with a reversion where the guard would
otherwise be decorative:

- a file reference is a CLAIM the daemon verifies: inside the sender's
  own workspace (judged on the resolved path, so a symlink cannot carry a
  file out), a regular file that exists, and not a credential file;
- a target in the SAME workspace is handed the relative path plus the
  digest and size the daemon measured, and nothing is copied;
- a target in ANOTHER workspace gets a COPY under its own inbox, bounded
  by the staging caps, with the copy's digest re-verified;
- ONE reference that does not pass refuses the whole message, and takes
  back whatever was already copied -- a peer never acts on a manifest with
  a file silently missing from it;
- a text attachment is inlined under a fence the peer's text cannot close,
  up to 32 KiB; beyond that it becomes a file;
- a spooled message keeps its manifest and the drive renders it; the
  delivered files OUTLIVE the envelope, because the target reads them on
  later turns too;
- a message that was copied for and then not delivered takes its copies
  back, and the receipt says ``discarded`` rather than ``copied``.

Lives in ``shared/tests`` so the reversion meta-suite walks it.
"""

import hashlib
import os
import pathlib

import pytest

from jaato_sdk.plugins.model_provider.types import tool_result_is_error
from jaato_server.server import session_inbox
from jaato_server.server.command_router import (
    _decode_message_request, _message_result_event,
)
from jaato_server.shared.plugins.courier.plugin import CourierPlugin
from jaato_server.shared.tool_result_builder import split_executor_result
from jaato_server.shared.tests.reversion import Reversion
from .test_durable_inbox_phase2 import ATT, _inbox, _send, _session, _sm, _ws

_SM = "jaato-server/jaato_server/server/session_manager.py"
_SI = "jaato-server/jaato_server/server/session_inbox.py"


def _write(ws, rel, data=b"# q3\nrevenue up\n"):
    p = pathlib.Path(ws) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _pair(tmp_path, *, same_ws):
    """Sender ``s-a`` and idle target ``s-b``, in one workspace or two."""
    ws_a = _ws(tmp_path, "A")
    ws_b = ws_a if same_ws else _ws(tmp_path, "B")
    sm = _sm(tmp_path, _session("s-a", ws=ws_a), _session("s-b", ws=ws_b))
    return sm, ws_a, ws_b


def _files_dir(sm, ws, sid, message_id):
    return session_inbox.files_dir(sm._session_storage_dir(ws), sid, message_id)


# ------------------------------------------------------------ references

def test_a_same_workspace_reference_is_delivered_in_place_with_digest_and_size(tmp_path):
    sm, ws, _ = _pair(tmp_path, same_ws=True)
    data = b"# q3\nrevenue up\n"
    _write(ws, "reports/q3.md", data)
    r = _send(sm, file_refs=["reports/q3.md"])
    assert r["status"] == "accepted", r
    [row] = r["files"]
    assert row["disposition"] == "referenced" and row["path"] == "reports/q3.md"
    assert row["sha256"] == hashlib.sha256(data).hexdigest() and row["size"] == len(data)
    assert row["mime_type"] == "text/markdown"
    text = sm.delivered[-1][1]
    assert "reports/q3.md" in text and row["sha256"] in text and "in your workspace" in text
    assert "read them with your own tools" in text
    assert not (pathlib.Path(ws) / ".jaato" / "sessions").exists(), "nothing copied"


def test_a_cross_workspace_reference_is_copied_into_the_targets_inbox(tmp_path):
    sm, ws_a, ws_b = _pair(tmp_path, same_ws=False)
    data = os.urandom(3000)
    _write(ws_a, "out/report.pdf", data)
    r = _send(sm, file_refs=[{"path": "out/report.pdf", "workspace": ws_a}])
    assert r["status"] == "accepted", r
    [row] = r["files"]
    assert row["disposition"] == "copied"
    assert row["path"] == f".jaato/sessions/s-b.inbox/files/{r['message_id']}/00-report.pdf"
    copy = pathlib.Path(ws_b) / row["path"]
    assert copy.read_bytes() == data
    assert row["sha256"] == hashlib.sha256(data).hexdigest()
    text = sm.delivered[-1][1]
    assert row["path"] in text and "copied here from the sender's workspace" in text
    assert not (pathlib.Path(ws_a) / "out" / ".jaato").exists()


def test_a_reference_outside_the_senders_workspace_refuses_the_whole_message(tmp_path):
    sm, ws_a, _ = _pair(tmp_path, same_ws=False)
    secret = tmp_path / "secret.txt"
    secret.write_text("hunter2")
    for ref in ("../secret.txt", str(secret),
                {"path": "secret.txt", "workspace": str(tmp_path)}):
        r = _send(sm, text="see", file_refs=[ref])
        assert r["status"] == "refused", (ref, r)
        assert r["files"][0]["reason"] == "outside_sender_workspace"
        assert "outside your workspace" in r["error"]
    # A link planted INSIDE the workspace is judged by what it points at.
    os.symlink(secret, pathlib.Path(ws_a) / "link.txt")
    r = _send(sm, text="see", file_refs=["link.txt"])
    assert r["status"] == "refused" and r["files"][0]["reason"] == "outside_sender_workspace"
    assert sm.delivered == [], "a refused message reaches nobody"


def test_a_missing_file_a_directory_and_a_credential_file_are_refused_by_name(tmp_path):
    sm, ws_a, _ = _pair(tmp_path, same_ws=False)
    (pathlib.Path(ws_a) / "dir").mkdir()
    _write(ws_a, ".env", b"KEY=1")
    _write(ws_a, ".jaato/openai_auth.json", b"{}")
    _write(ws_a, ".env.example", b"KEY=")
    for ref, reason in (("nope.md", "not_found"), ("dir", "not_a_file"),
                        (".env", "credential"), (".jaato/openai_auth.json", "credential")):
        r = _send(sm, text="see", file_refs=[ref])
        assert r["status"] == "refused" and r["files"][0]["reason"] == reason, (ref, r)
        assert r["files"][0]["name"] == os.path.basename(ref)
    assert _send(sm, text="see", file_refs=[".env.example"])["status"] == "accepted"


def test_a_malformed_row_is_refused_by_index_and_a_file_only_message_is_content(tmp_path):
    sm, ws_a, _ = _pair(tmp_path, same_ws=True)
    _write(ws_a, "a.md")
    assert _send(sm, text="see", file_refs=[42])["error"].startswith(
        "send_to_session: file_refs[0] must be")
    assert _send(sm, text="see", text_attachments=["x"])["error"].startswith(
        "send_to_session: text_attachments[0] must be")
    # A reference is content: no text needed (#838's rule, one kind over).
    assert _send(sm, text="", file_refs=["a.md"])["status"] == "accepted"
    assert _send(sm, text="")["status"] == "refused"


def test_referencing_needs_the_senders_workspace(tmp_path):
    sm, _, _ = _pair(tmp_path, same_ws=False)
    sm._sessions["s-a"].workspace_path = None
    r = _send(sm, text="see", file_refs=["a.md"])
    assert r["status"] == "refused" and r["files"][0]["reason"] == "sender_has_no_workspace"


# ------------------------------------------------------------ all or nothing

def test_one_refused_reference_takes_back_what_was_already_copied(tmp_path):
    sm, ws_a, ws_b = _pair(tmp_path, same_ws=False)
    _write(ws_a, "good.md")
    r = _send(sm, text="see", file_refs=["good.md", "missing.md"])
    assert r["status"] == "refused", r
    assert [f["disposition"] for f in r["files"]] == ["discarded", "refused"]
    assert sm.delivered == []
    inbox = pathlib.Path(ws_b) / ".jaato" / "sessions" / "s-b.inbox"
    assert not (inbox / "files").exists(), "the copy was taken back"


def test_a_file_over_the_per_file_cap_is_refused_and_named(tmp_path, monkeypatch):
    sm, ws_a, ws_b = _pair(tmp_path, same_ws=False)
    monkeypatch.setattr(session_inbox, "FILE_COPY_PER_FILE_LIMIT", 10)
    _write(ws_a, "big.bin", b"x" * 11)
    r = _send(sm, text="see", file_refs=["big.bin"])
    assert r["status"] == "refused" and r["files"][0]["reason"] == "file_too_large"
    assert "11 bytes, over the 10-byte" in r["error"]
    # Same file, same workspace: nothing is copied, so no cap applies.
    same, ws, _ = _pair(tmp_path, same_ws=True)
    _write(ws, "big.bin", b"x" * 11)
    assert _send(same, text="see", file_refs=["big.bin"])["status"] == "accepted"


def test_the_per_message_total_bounds_the_copies(tmp_path, monkeypatch):
    sm, ws_a, _ = _pair(tmp_path, same_ws=False)
    monkeypatch.setattr(session_inbox, "FILE_COPY_TOTAL_LIMIT", 15)
    _write(ws_a, "one.bin", b"x" * 10)
    _write(ws_a, "two.bin", b"y" * 10)
    r = _send(sm, text="see", file_refs=["one.bin", "two.bin"])
    assert r["status"] == "refused"
    assert [f["disposition"] for f in r["files"]] == ["discarded", "refused"]
    assert r["files"][1]["reason"] == "message_files_too_large"


def test_a_copy_whose_digest_does_not_match_is_refused(tmp_path, monkeypatch):
    sm, ws_a, _ = _pair(tmp_path, same_ws=False)
    _write(ws_a, "a.md")
    real = session_inbox.store_file

    def lying(*a, **kw):
        out = real(*a, **kw)
        return {**out, "sha256": "0" * 64}
    monkeypatch.setattr(session_inbox, "store_file", lying)
    r = _send(sm, text="see", file_refs=["a.md"])
    assert r["status"] == "refused" and r["files"][0]["reason"] == "copy_mismatch"


# ------------------------------------------------------------ text attachments

def test_a_text_attachment_is_inlined_under_a_fence_the_text_cannot_close(tmp_path):
    sm, _, _ = _pair(tmp_path, same_ws=True)
    patch = "--- a\n+++ b\n```\nclosed?\n```\n"
    r = _send(sm, text="", text_attachments=[
        {"name": "fix.patch", "text": patch, "mime_type": "text/x-diff"}])
    assert r["status"] == "accepted", r
    assert r["files"] == [{"name": "fix.patch", "disposition": "inlined",
                           "size": len(patch.encode())}]
    text = sm.delivered[-1][1]
    assert f"--- fix.patch (text/x-diff, {len(patch.encode())} bytes) ---" in text
    assert "````\n" + patch + "\n````" in text, "a fence longer than the text's own"
    assert "text attachment(s) delivered with this message" in text


def test_a_text_attachment_over_the_inline_cap_becomes_a_file(tmp_path, monkeypatch):
    sm, ws_a, ws_b = _pair(tmp_path, same_ws=False)
    monkeypatch.setattr(session_inbox, "TEXT_ATTACHMENT_INLINE_CAP", 8)
    r = _send(sm, text="see", text_attachments=[
        {"name": "short", "text": "12345678"}, {"name": "long.txt", "text": "x" * 9}])
    assert r["status"] == "accepted", r
    assert r["files"][0]["disposition"] == "inlined"
    long = r["files"][1]
    assert long["disposition"] == "copied" and long["reason"] == "over_inline_cap"
    assert (pathlib.Path(ws_b) / long["path"]).read_text() == "x" * 9
    text = sm.delivered[-1][1]
    assert "--- short (text/plain, 8 bytes) ---" in text and long["path"] in text


# ------------------------------------------------------------ spool and lifetime

def test_a_spooled_message_keeps_its_manifest_and_the_files_outlive_the_envelope(tmp_path):
    ws_a, ws_b = _ws(tmp_path, "A"), _ws(tmp_path, "B")
    sm = _sm(tmp_path, _session("s-a", ws=ws_a), _session("s-b", ws=ws_b, running=True))
    _write(ws_a, "r.md", b"report")
    r = _send(sm, attachments=ATT, file_refs=["r.md"],
              text_attachments=[{"name": "n.txt", "text": "note"}])
    assert r["status"] == "spooled", r
    [entry] = _inbox(sm, ws_b, "s-b")
    assert entry.files[0]["path"] == r["files"][0]["path"]
    assert entry.text_attachments == [{"display_name": "n.txt", "text": "note",
                                       "mime_type": "text/plain"}]
    copy = pathlib.Path(ws_b) / r["files"][0]["path"]
    assert copy.read_bytes() == b"report"

    sm._sessions["s-b"].server._model_running = False
    assert sm.drain_session_inbox("s-b", trigger="turn_end") is True
    text = sm.delivered[-1][1]
    assert r["files"][0]["path"] in text and "--- n.txt (text/plain, 4 bytes) ---" in text
    assert _inbox(sm, ws_b, "s-b") == [], "the envelope is gone"
    assert copy.read_bytes() == b"report", "the delivered file is not"


def test_an_undelivered_message_takes_its_copies_back(tmp_path):
    ws_a, ws_b = _ws(tmp_path, "A"), _ws(tmp_path, "B")
    sm = _sm(tmp_path, _session("s-a", ws=ws_a), _session("s-b", ws=ws_b, running=True))
    _write(ws_a, "r.md")
    sm._group_pending["s-b"] = 20            # at the cap: backpressure
    r = _send(sm, text="see", file_refs=["r.md"], pending_cap=20)
    assert r["status"] == "refused", r
    assert r["files"] == [dict(r["files"][0], disposition="discarded")]
    assert not (pathlib.Path(ws_b) / ".jaato" / "sessions" / "s-b.inbox" / "files").exists()


# ------------------------------------------------------------ surfaces

def test_the_router_decodes_the_two_keys_and_the_result_event_carries_files():
    req = _decode_message_request([], {"target": "s-b", "text": "",
                                       "file_refs": ["a.md", {"path": "b"}],
                                       "text_attachments": [{"name": "n", "text": "t"}]})
    assert req.file_refs == ["a.md", {"path": "b"}] and req.has_content
    assert req.text_attachments == [{"name": "n", "text": "t"}]
    assert _decode_message_request([], {"target": "s-b", "file_refs": "a.md"}).file_refs == []
    ev = _message_result_event("r1", "s-b", {
        "status": "refused", "files": [{"name": "a", "disposition": "refused",
                                        "reason": "not_found"}], "error": "no"})
    assert ev.files[0]["reason"] == "not_found" and ev.ok is False


def test_the_plugin_forwards_the_keys_and_refuses_a_non_list():
    plugin = CourierPlugin()
    plugin.set_plugin_registry(type("R", (), {"session_id": "s-a"})())
    calls = []

    class _Mgr:
        def deliver_group_message(self, sid, target, text, **kw):
            calls.append(kw)
            return {"status": "accepted", "files": kw["file_refs"]}
    plugin.set_session_manager(_Mgr())
    body = plugin._execute_send_to_session(
        {"target": "s-b", "message": "", "file_refs": ["a.md"]})
    assert calls[0]["file_refs"] == ["a.md"] and calls[0]["text_attachments"] == []
    ok, payload = split_executor_result(body)
    assert ok and not tool_result_is_error(payload)
    ok, payload = split_executor_result(plugin._execute_send_to_session(
        {"target": "s-b", "message": "hi", "file_refs": "a.md"}))
    assert ok is False and "must be arrays" in payload["error"]


def test_the_inbox_store_writes_atomically_and_names_files_by_index(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    row = session_inbox.store_file(tmp_path, "s-1", "m-1", 3, "../x/report.md", source=src)
    assert row == {"file": "s-1.inbox/files/m-1/03-report.md",
                   "sha256": hashlib.sha256(b"abc").hexdigest(), "size": 3}
    row2 = session_inbox.store_file(tmp_path, "s-1", "m-1", 4, "n.txt", data=b"hi")
    assert (tmp_path / row2["file"]).read_bytes() == b"hi"
    assert sorted(p.name for p in (tmp_path / "s-1.inbox" / "files" / "m-1").iterdir()) == [
        "03-report.md", "04-n.txt"], "no temp file survives"
    assert session_inbox.count_pending(tmp_path, "s-1") == 0, "files are not envelopes"
    session_inbox.remove_files(tmp_path, "s-1", "m-1")
    assert not (tmp_path / "s-1.inbox" / "files").exists()
    with pytest.raises(OSError):
        session_inbox.store_file(tmp_path, "s-1", "m-2", 0, "x", source=tmp_path / "nope")
    assert not (tmp_path / "s-1.inbox" / "files" / "m-2").exists() or not any(
        (tmp_path / "s-1.inbox" / "files" / "m-2").iterdir())


# ------------------------------------------------------------- reversions

REVERSIONS = [
    Reversion(
        target=_SM,
        find='        if not inside:\n            return None, "", "outside_sender_workspace"\n',
        replace='        if not inside and False:\n            return None, "", "outside_sender_workspace"\n',
        because="a path outside the sender's workspace is referenced or copied: any "
                "file the daemon can read is one message away from another workspace",
        test="test_a_reference_outside_the_senders_workspace_refuses_the_whole_message",
    ),
    Reversion(
        target=_SM,
        find='        if is_credential_path(rel):\n            return None, rel, "credential"\n',
        replace='        if is_credential_path(rel) and False:\n            return None, rel, "credential"\n',
        because="a workspace .env is copied into another workspace by messaging a session there",
        test="test_a_missing_file_a_directory_and_a_credential_file_are_refused_by_name",
    ),
    Reversion(
        target=_SM,
        find='        if size > session_inbox.FILE_COPY_PER_FILE_LIMIT:\n            return None, "file_too_large"\n',
        replace='        if size > session_inbox.FILE_COPY_PER_FILE_LIMIT and False:\n            return None, "file_too_large"\n',
        because="the per-file copy cap is not applied; a message copies more than a stage may",
        test="test_a_file_over_the_per_file_cap_is_refused_and_named",
    ),
    Reversion(
        target=_SM,
        find="        if payload.refusal is not None:\n            self._release_event_id(event_id)\n            return payload.refusal\n",
        replace="        if payload.refusal is not None and False:\n            self._release_event_id(event_id)\n            return payload.refusal\n",
        because="a message with a refused reference is delivered anyway, with that file "
                "silently missing from the manifest the peer acts on",
        test="test_one_refused_reference_takes_back_what_was_already_copied",
    ),
    Reversion(
        target=_SM,
        find='        if stored["sha256"] != digest:\n            return _refuse_file(payload, index, name, "copy_mismatch")\n',
        replace='        if stored["sha256"] != digest and False:\n            return _refuse_file(payload, index, name, "copy_mismatch")\n',
        because="a copy whose bytes differ from the source is delivered under the source's digest",
        test="test_a_copy_whose_digest_does_not_match_is_refused",
    ),
    Reversion(
        target=_SM,
        find="            self._release_event_id(event_id)\n            self._discard_group_files(member, message_id, payload)\n",
        replace="            self._release_event_id(event_id)\n",
        because="a message that was not delivered leaves its copies in the target's inbox, "
                "and the receipt says copied",
        test="test_an_undelivered_message_takes_its_copies_back",
    ),
    Reversion(
        target=_SI,
        find="    shutil.rmtree(root / entry.message_id, ignore_errors=True)\n    try:\n        if root.is_dir() and not any(root.iterdir()):\n",
        replace="    shutil.rmtree(root / entry.message_id, ignore_errors=True)\n    shutil.rmtree(root / FILES_DIRNAME / entry.message_id, ignore_errors=True)\n    try:\n        if root.is_dir() and not any(root.iterdir()):\n",
        because="the drive that delivers a spooled message deletes the files it just told the "
                "target to read",
        test="test_a_spooled_message_keeps_its_manifest_and_the_files_outlive_the_envelope",
    ),
]
