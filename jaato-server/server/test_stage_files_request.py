"""Tests for the binary-framed ``StageFilesRequest`` WS protocol.

Multi-frame protocol: client sends one TEXT ``StageFilesRequest`` then
``len(files)`` raw BINARY frames, one per declared file, in order.
Server validates each frame's length against the declared ``size``,
materialises into the target workspace, and replies with one
``StageFilesEvent``.

The handler is security-sensitive on three axes — path safety, size
caps, and frame-stream alignment after a rejection.  These tests pin
each rejection path and the alignment behaviour, with the websocket
mocked so we don't need a live connection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from jaato_sdk.events import (
    StageFilesEvent,
    StageFilesRequest,
    StagedFileSpec,
)


# Importing JaatoWSServer pulls in heavy server deps; do it lazily inside
# fixtures so a missing optional module (e.g. an unrelated provider) doesn't
# fail collection of these focused tests.
@pytest.fixture
def ws_server(tmp_path):
    from server.websocket import JaatoWSServer

    server = JaatoWSServer.__new__(JaatoWSServer)
    # Minimal state the handler actually touches.
    server._clients = {}
    server._lock = asyncio.Lock()
    server._client_provisioned = {}
    server._workspace_manager = None
    server._event_sink_adapter = None
    return server


def _attach_client(server, client_id: str, workspace_path: Path, frames: List[Any]):
    """Wire a fake WS client whose ``recv()`` returns ``frames`` in order.

    Each frame may be ``bytes`` (binary), ``str`` (text), or an
    ``Exception`` instance which will be raised on that recv.
    """
    fake_ws = MagicMock()
    sent = []

    async def _send(msg):
        sent.append(msg)

    fake_ws.send = _send

    iterator = iter(frames)

    async def _recv():
        try:
            f = next(iterator)
        except StopIteration:
            raise RuntimeError("recv called with no frames left")
        if isinstance(f, BaseException):
            raise f
        return f

    fake_ws.recv = _recv

    client = MagicMock()
    client.websocket = fake_ws
    server._clients[client_id] = client

    # Provisioned workspace is what _resolve_staging_workspace finds first.
    provisioned = MagicMock()
    provisioned.path = str(workspace_path)
    server._client_provisioned[client_id] = provisioned

    return fake_ws, sent


def _last_response_event(sent_messages):
    """Decode the last serialised event the server sent."""
    from jaato_sdk.events import deserialize_event
    assert sent_messages, "server sent no response"
    return deserialize_event(sent_messages[-1])


# ----------------------------------------------------------------------
# Happy paths
# ----------------------------------------------------------------------

class TestHappyPath:

    def test_single_file_is_written(self, ws_server, tmp_path):
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        payload = b"hello, world"
        fake_ws, sent = _attach_client(
            ws_server, "c1", ws_path, frames=[payload],
        )

        req = StageFilesRequest(
            workspace_id=ws_path.name,
            files=[StagedFileSpec(name="hello.txt", size=len(payload))],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        assert (ws_path / "hello.txt").read_bytes() == payload
        ev = _last_response_event(sent)
        assert isinstance(ev, StageFilesEvent)
        assert ev.staged == ["hello.txt"]
        assert ev.failed == []

    def test_multiple_files_in_order(self, ws_server, tmp_path):
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        a, b, c = b"AAA", b"BBBB", b"CC"
        fake_ws, sent = _attach_client(
            ws_server, "c1", ws_path, frames=[a, b, c],
        )

        req = StageFilesRequest(
            workspace_id=ws_path.name,
            files=[
                StagedFileSpec(name="a.bin", size=3),
                StagedFileSpec(name="sub/b.bin", size=4),
                StagedFileSpec(name="c.bin", size=2),
            ],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        assert (ws_path / "a.bin").read_bytes() == a
        assert (ws_path / "sub" / "b.bin").read_bytes() == b
        assert (ws_path / "c.bin").read_bytes() == c
        ev = _last_response_event(sent)
        assert ev.staged == ["a.bin", "sub/b.bin", "c.bin"]
        assert ev.failed == []

    def test_empty_workspace_id_uses_current(self, ws_server, tmp_path):
        """Empty workspace_id resolves to the client's currently-attached
        workspace — the common case for telegram-style auto-provisioning."""
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        payload = b"x"
        _, sent = _attach_client(ws_server, "c1", ws_path, frames=[payload])

        req = StageFilesRequest(
            workspace_id="",  # explicit "use my current workspace"
            files=[StagedFileSpec(name="x.bin", size=1)],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        assert (ws_path / "x.bin").read_bytes() == payload
        ev = _last_response_event(sent)
        assert ev.staged == ["x.bin"]


# ----------------------------------------------------------------------
# Workspace resolution
# ----------------------------------------------------------------------

class TestWorkspaceResolution:

    def test_no_workspace_attached_returns_failure(self, ws_server, tmp_path):
        # No _attach_client → no provisioned workspace.  Must still drain
        # the binary frames so the next connection message stays aligned.
        fake_ws = MagicMock()
        sent = []

        async def _send(msg): sent.append(msg)
        fake_ws.send = _send

        drain_frames = [b"x" * 5]

        recv_calls = iter(drain_frames)

        async def _recv(): return next(recv_calls)
        fake_ws.recv = _recv

        client = MagicMock()
        client.websocket = fake_ws
        ws_server._clients["c1"] = client

        req = StageFilesRequest(
            workspace_id="ws_abc",
            files=[StagedFileSpec(name="a.bin", size=5)],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        ev = _last_response_event(sent)
        assert ev.staged == []
        assert len(ev.failed) == 1
        assert ev.failed[0]["category"] == "workspace_not_found"

    def test_wrong_workspace_id_returns_failure(self, ws_server, tmp_path):
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        # Single drain frame even though we're rejecting up-front.
        _, sent = _attach_client(ws_server, "c1", ws_path, frames=[b"x"])

        req = StageFilesRequest(
            workspace_id="some_other_workspace",  # mismatch
            files=[StagedFileSpec(name="a.bin", size=1)],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        ev = _last_response_event(sent)
        assert ev.staged == []
        assert ev.failed[0]["category"] == "workspace_not_found"
        # File must NOT have been written.
        assert not (ws_path / "a.bin").exists()


# ----------------------------------------------------------------------
# Size validation
# ----------------------------------------------------------------------

class TestSizeValidation:

    def test_size_mismatch_per_file_marks_failed(self, ws_server, tmp_path):
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        # Declared size 100 but frame is only 5 bytes.
        _, sent = _attach_client(ws_server, "c1", ws_path, frames=[b"short"])

        req = StageFilesRequest(
            workspace_id=ws_path.name,
            files=[StagedFileSpec(name="a.bin", size=100)],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        ev = _last_response_event(sent)
        assert ev.staged == []
        assert ev.failed[0]["category"] == "size_mismatch"
        assert "100" in ev.failed[0]["error"]
        assert not (ws_path / "a.bin").exists()

    def test_per_file_cap_rejects_without_reading_payload(
        self, ws_server, tmp_path,
    ):
        from server.websocket import DEFAULT_STAGE_PER_FILE_LIMIT
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        # Frame supplied solely so the drain pass picks it up; the
        # handler should reject before recv is called for this file.
        _, sent = _attach_client(ws_server, "c1", ws_path, frames=[b"x"])

        too_big = DEFAULT_STAGE_PER_FILE_LIMIT + 1
        req = StageFilesRequest(
            workspace_id=ws_path.name,
            files=[StagedFileSpec(name="a.bin", size=too_big)],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        ev = _last_response_event(sent)
        assert ev.failed[0]["category"] == "size_limit_per_file"
        assert not (ws_path / "a.bin").exists()

    def test_total_cap_rejects_without_reading_any(
        self, ws_server, tmp_path,
    ):
        from server.websocket import DEFAULT_STAGE_PER_FILE_LIMIT, DEFAULT_STAGE_TOTAL_LIMIT
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        # Two frames the server should drain without writing.
        _, sent = _attach_client(ws_server, "c1", ws_path, frames=[b"x", b"y"])

        # Two files each at the per-file cap → over the total cap.
        per_file = DEFAULT_STAGE_PER_FILE_LIMIT
        n = (DEFAULT_STAGE_TOTAL_LIMIT // per_file) + 1
        files = [StagedFileSpec(name=f"f{i}.bin", size=per_file) for i in range(n)]
        # Adjust frames count so drain works.
        ws_server._clients["c1"].websocket.recv = _make_recv_iter([b"x"] * n)

        req = StageFilesRequest(workspace_id=ws_path.name, files=files)
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        ev = _last_response_event(sent)
        assert ev.staged == []
        # Every file flagged with the total-cap category.
        assert all(f["category"] == "size_limit_total" for f in ev.failed)


# ----------------------------------------------------------------------
# Path safety
# ----------------------------------------------------------------------

class TestPathSafety:

    @pytest.mark.parametrize("bad_name", [
        "/etc/passwd",          # absolute
        "../escape.txt",        # parent traversal
        "sub/../../escape.txt", # nested traversal
        "",                     # empty
    ])
    def test_unsafe_paths_rejected(self, ws_server, tmp_path, bad_name):
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        _, sent = _attach_client(ws_server, "c1", ws_path, frames=[b"x"])

        req = StageFilesRequest(
            workspace_id=ws_path.name,
            files=[StagedFileSpec(name=bad_name, size=1)],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        ev = _last_response_event(sent)
        assert ev.failed[0]["category"] == "unsafe_path"
        # The frame was drained, so no file ended up under ws_path.
        assert list(ws_path.iterdir()) == []


# ----------------------------------------------------------------------
# Stream-alignment behaviour after a rejection
# ----------------------------------------------------------------------

class TestStreamAlignment:

    def test_text_frame_where_binary_expected_aborts_remainder(
        self, ws_server, tmp_path,
    ):
        """A protocol violation (TEXT instead of BINARY) is fatal; we
        cannot trust subsequent frames are correctly positioned."""
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        # First frame is TEXT (wrong), second would be a valid binary
        # but we should never read it.
        _, sent = _attach_client(
            ws_server, "c1", ws_path, frames=["{not binary}", b"valid"],
        )

        req = StageFilesRequest(
            workspace_id=ws_path.name,
            files=[
                StagedFileSpec(name="a.bin", size=12),
                StagedFileSpec(name="b.bin", size=5),
            ],
        )
        asyncio.run(ws_server._handle_stage_files_request("c1", req))

        ev = _last_response_event(sent)
        assert ev.staged == []
        assert ev.failed[0]["category"] == "size_mismatch"
        assert "TEXT" in ev.failed[0]["error"]
        # Second file not in the failed list — we aborted before
        # touching it.
        assert len(ev.failed) == 1


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_recv_iter(frames):
    """Build an awaitable recv() that returns frames in order."""
    iterator = iter(frames)

    async def _recv():
        try:
            return next(iterator)
        except StopIteration:
            raise RuntimeError("recv called with no frames left")
    return _recv
