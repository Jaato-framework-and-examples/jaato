"""Staging never writes through a symlink planted in the workspace (#1386).

A workspace is model-writable, so the agent can plant a link on a file
(``foo -> /etc/...``) or on a parent directory (``sub -> /``).  Staging used
``mkdir(parents=True)`` then ``write_bytes``, both of which follow links, so
the next staging naming that path wrote the client's bytes wherever the link
pointed: on a root daemon, anywhere on the host.

Every staging route now writes through
:func:`jaato_server.server.contained_write.write_contained`: each parent
component is checked inside the resolved workspace before it is created or
entered, a destination that is a link is refused (``unsafe_path``), and the
write is a temp file plus ``os.replace``.  The Phase 3 group-message copies
into a session's inbox create their directories the same way.

Routes driven here: ``StageFilesRequest`` (``_handle_stage_files_request``),
the inline ``staged_files`` / ``/api/task/artifacts`` helper
(``_materialize_staged_files``), and ``session_inbox.store_file`` /
``spool``.
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import StageFilesRequest, StagedFileSpec, deserialize_event
from jaato_server.server import session_inbox
from jaato_server.server.websocket import _materialize_staged_files
from jaato_server.shared.tests.reversion import Reversion

_CW = "jaato-server/jaato_server/server/contained_write.py"
_INBOX = "jaato-server/jaato_server/server/session_inbox.py"

REVERSIONS = [
    Reversion(
        target=_CW,
        find="            if not under(real, root):\n",
        replace="            if False:\n",
        test="test_a_linked_parent_directory_does_not_carry_the_write_outside",
        because="a parent directory linked out of the workspace is descended into and written",
    ),
    Reversion(
        target=_CW,
        find="    if os.path.islink(dest):\n        raise PathLeavesRoot(",
        replace="    if False:\n        raise PathLeavesRoot(",
        test="test_a_linked_file_is_refused_and_reported",
        because="a destination that is a symlink is not refused and reported as unsafe_path",
    ),
    Reversion(
        target=_INBOX,
        find="        contained_dir(os.path.realpath(storage_dir), rel, create=True)\n",
        replace="        os.makedirs(os.path.join(storage_dir, rel), exist_ok=True)\n",
        test="test_a_linked_inbox_does_not_carry_a_copied_file_outside",
        because="a group-message file copy follows a link planted on the session's inbox",
    ),
]


@pytest.fixture
def layout(tmp_path):
    """A workspace, and a directory outside it the links point into."""
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    return ws, outside


@pytest.fixture
def ws_server():
    from jaato_server.server.websocket import JaatoWSServer
    from jaato_server.server.ws_tickets import AppCredentialStore

    server = JaatoWSServer.__new__(JaatoWSServer)
    # No per-application workspace roots configured.
    server._clients = {}
    server._app_managers = {}
    server._app_provisioners = {}
    server._app_credentials = AppCredentialStore({})
    server._clients = {}
    server._lock = asyncio.Lock()
    server._client_provisioned = {}
    server._workspace_manager = None
    server._event_sink_adapter = None
    return server


def _stage(server, ws: Path, files: List[tuple]) -> Any:
    """Run one ``StageFilesRequest`` for ``[(name, bytes), ...]``; return the event."""
    frames = iter([data for _, data in files])
    sent: list = []
    fake_ws = MagicMock()

    async def _send(msg):
        sent.append(msg)

    async def _recv():
        return next(frames)

    fake_ws.send = _send
    fake_ws.recv = _recv
    client = MagicMock()
    client.websocket = fake_ws
    server._clients["c1"] = client
    server._client_provisioned["c1"] = MagicMock(path=str(ws))
    req = StageFilesRequest(
        workspace_id=ws.name,
        files=[StagedFileSpec(name=n, size=len(d)) for n, d in files],
    )
    asyncio.run(server._handle_stage_files_request("c1", req))
    return deserialize_event(sent[-1])


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_a_linked_file_is_refused_and_reported(ws_server, layout):
    ws, outside = layout
    victim = outside / "victim"
    victim.write_bytes(b"original")
    (ws / "foo").symlink_to(victim)

    ev = _stage(ws_server, ws, [("foo", b"pwned")])

    assert victim.read_bytes() == b"original"
    assert ev.staged == []
    assert ev.failed[0]["category"] == "unsafe_path"
    assert "symlink leaving the workspace" in ev.failed[0]["error"]
    assert (ws / "foo").is_symlink(), "the refusal must not touch the link either"


def test_a_linked_parent_directory_does_not_carry_the_write_outside(ws_server, layout):
    ws, outside = layout
    (ws / "sub").symlink_to(outside, target_is_directory=True)

    ev = _stage(ws_server, ws, [("sub/new.txt", b"pwned"), ("sub/deep/er.txt", b"pwned")])

    assert list(outside.iterdir()) == []
    assert ev.staged == []
    assert [f["category"] for f in ev.failed] == ["unsafe_path", "unsafe_path"]


def test_a_dangling_link_is_refused_the_same_way(ws_server, layout):
    """Containment is judged before existence: no oracle for the host."""
    ws, outside = layout
    (ws / "gone").symlink_to(outside / "does-not-exist", target_is_directory=True)
    (ws / "there").symlink_to(outside, target_is_directory=True)

    ev = _stage(ws_server, ws, [("gone/x", b"a"), ("there/x", b"a")])

    assert list(outside.iterdir()) == []
    first, second = ev.failed
    assert first["category"] == second["category"] == "unsafe_path"
    assert first["error"].replace("gone", "?") == second["error"].replace("there", "?")


def test_the_inline_route_skips_a_linked_entry_and_never_raises(layout):
    ws, outside = layout
    victim = outside / "victim"
    victim.write_bytes(b"original")
    (ws / "foo").symlink_to(victim)
    (ws / "sub").symlink_to(outside, target_is_directory=True)

    written = _materialize_staged_files(ws, [
        {"name": "foo", "data": _b64(b"pwned")},
        {"name": "sub/new.txt", "data": _b64(b"pwned")},
        {"name": "ok/nested.txt", "data": _b64(b"fine")},
    ])

    assert written == 1
    assert victim.read_bytes() == b"original"
    assert sorted(p.name for p in outside.iterdir()) == ["victim"]
    assert (ws / "ok" / "nested.txt").read_bytes() == b"fine"


def test_an_ordinary_nested_path_still_stages(ws_server, layout):
    """Control: the containment walk does not refuse legitimate work."""
    ws, _ = layout
    (ws / "a").mkdir()
    (ws / "inside").symlink_to(ws / "a", target_is_directory=True)

    ev = _stage(ws_server, ws, [
        ("a/b/c.txt", b"one"),
        ("fresh/dir/d.txt", b"two"),
        ("inside/e.txt", b"three"),  # a link that stays inside is fine
        ("a/b/c.txt", b"again"),     # overwriting a regular file
    ])

    assert ev.failed == []
    assert (ws / "a" / "b" / "c.txt").read_bytes() == b"again"
    assert (ws / "fresh" / "dir" / "d.txt").read_bytes() == b"two"
    assert (ws / "a" / "e.txt").read_bytes() == b"three"


def test_a_linked_inbox_does_not_carry_a_copied_file_outside(layout):
    ws, outside = layout
    storage = ws / ".jaato" / "sessions"
    storage.mkdir(parents=True)
    (storage / "s1.inbox").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        session_inbox.store_file(storage, "s1", "m1", 0, "report.md", data=b"pwned")

    assert list(outside.iterdir()) == []


def test_an_inbox_copy_still_lands_normally(layout):
    ws, _ = layout
    storage = ws / ".jaato" / "sessions"

    stored = session_inbox.store_file(storage, "s1", "m1", 0, "report.md", data=b"ok")

    assert (storage / stored["file"]).read_bytes() == b"ok"
    assert os.path.realpath(storage / stored["file"]).startswith(os.path.realpath(ws))
