"""A staged file over 1 MiB closed the connection and the client waited 120 s.

Reported from the web client with a 1.4 MB PDF: the chip's progress bar
slid for two minutes and then read *"stageFiles: no workspace.files.staged
response after 120000 ms"*.  Nothing on the daemon side said why.

``websockets`` refuses any message over its ``max_size`` by closing the
connection with code 1009, and its default is 1 MiB.  The daemon never
passed one, while the staging cap it enforces is 10 MB per file -- and a
staged file travels as ONE binary message.  So every file between 1 MiB
and 10 MB closed the socket mid-upload; the connection handler swallowed
the ``ConnectionClosed`` with a bare ``pass``; the client reconnected as a
new client that had never sent the request; and the SDK waited out its
deadline for an answer nobody would send.

Three changes, each pinned here against a real socket:

* ``test_a_file_over_one_mebibyte_is_accepted`` -- the server passes an
  explicit ``max_size`` (16 MiB by default, covering the per-file cap even
  when a client sends it base64-inline).  Fails with 1009 on the default.
* ``test_the_handshake_advertises_the_limits`` -- ``server_info`` carries
  ``max_message_size`` and the staging caps, so a client refuses a file
  before sending it.  A daemon that advertises nothing is an older one
  enforcing 1 MiB, which is what the SDK assumes in that case.
* ``test_a_1009_close_is_logged_at_warning`` -- the close the daemon was
  forced into is no longer silent.

What the daemon CANNOT do is refuse such a file politely.  ``websockets``
reads incoming frames on its own, so an over-size frame closes the
connection the moment it arrives -- before any handler could answer the
request it belongs to (an up-front refusal was tried and lost that race
every time).  So the answer to "this file is too big" has to come from
the client, from the advertised limits; and the SDK now rejects a pending
``stageFiles`` the instant its connection closes instead of waiting out
the deadline.

The server is started with ``start()`` on an ephemeral port rather than
through ``websockets.serve`` directly, because the defect was in what
``start()`` passes to ``serve``: a test that built its own server would
choose its own ``max_size`` and pass on the broken tree.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pytest

from jaato_server.server.websocket import (
    DEFAULT_STAGE_PER_FILE_LIMIT,
    DEFAULT_STAGE_TOTAL_LIMIT,
    DEFAULT_WS_MAX_MESSAGE_SIZE,
    HAS_WEBSOCKETS,
    LEGACY_WS_MAX_MESSAGE_SIZE,
    WS_MESSAGE_ENVELOPE_HEADROOM,
    JaatoWSServer,
    describe_connection_close,
    parse_ws_max_message_size,
)
from jaato_server.shared.tests.reversion import Reversion

pytestmark = pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")

_WS = "jaato-server/jaato_server/server/websocket.py"

REVERSIONS = [
    Reversion(
        target=_WS,
        find="""                max_size=self._max_message_size,
            )""",
        replace="""            )""",
        test="test_a_file_over_one_mebibyte_is_accepted",
        because="without an explicit max_size websockets applies 1 MiB and closes the connection on any staged file above it, while the staging cap allows 10 MB",
    ),
    Reversion(
        target=_WS,
        find="""                **self.message_limits(),
            }""",
        replace="""            }""",
        test="test_the_handshake_advertises_the_limits",
        because="a client that is not told the daemon's limits cannot refuse a file before sending it, and an over-size message is not refused, it closes the connection",
    ),
    Reversion(
        target=_WS,
        find="""        logger.warning(f"Client {client_id} connection closed abnormally: {text}{hint}")""",
        replace="""        logger.debug(f"Client {client_id} connection closed abnormally: {text}{hint}")""",
        test="test_a_1009_close_is_logged_at_warning",
        because="a close the daemon was forced into must not vanish from its log",
    ),
]

# The reported file: 1.4 MB, over 1 MiB, far under the 10 MB staging cap.
_PDF_SIZE = 1_400_000


async def _started(server: JaatoWSServer) -> Tuple[asyncio.Task, int]:
    """Run ``server.start()`` and return (task, bound port)."""
    task = asyncio.create_task(server.start())
    for _ in range(200):
        if getattr(server, "_server", None) is not None:
            break
        await asyncio.sleep(0.01)
    assert server._server is not None, "server did not bind"
    port = server._server.sockets[0].getsockname()[1]
    return task, port


async def _stop(server: JaatoWSServer, task: asyncio.Task) -> None:
    await server.stop()
    try:
        await asyncio.wait_for(task, timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()


async def _stage(
    port: int, size: int, *, workspace: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[int]]:
    """Connect, stage one file of ``size`` bytes, return what came back.

    Returns ``(connected_event, staged_event_or_None, close_code_or_None)``.
    """
    import websockets
    from jaato_sdk.events import StageFilesRequest, StagedFileSpec, serialize_event

    async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
        connected = json.loads(await ws.recv())
        await ws.send(serialize_event(StageFilesRequest(
            workspace_id="",
            files=[StagedFileSpec(name="report.pdf", size=size)],
        )))
        await ws.send(b"\x07" * size)
        staged: Optional[Dict[str, Any]] = None
        close_code: Optional[int] = None
        try:
            while True:
                frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if frame.get("type") == "workspace.files.staged":
                    staged = frame
                    # The connection must still be usable after the drain.
                    await ws.ping()
                    break
        except websockets.ConnectionClosed as exc:
            close_code, _ = describe_connection_close(exc)
        if staged is not None and close_code is None:
            try:
                await asyncio.wait_for(ws.ping(), timeout=2)
            except websockets.ConnectionClosed as exc:
                close_code, _ = describe_connection_close(exc)
        return connected, staged, close_code


def _server(tmp_path, max_message_size: Optional[int] = None) -> JaatoWSServer:
    server = JaatoWSServer(
        host="127.0.0.1", port=0, required_token=None,
        max_message_size=max_message_size,
    )
    target = str(tmp_path)
    # Give the connection a workspace to stage into, so the answer is about
    # the file rather than about workspace selection.
    server._resolve_staging_workspace = lambda client_id, workspace_id: target  # type: ignore[assignment]
    return server


def test_the_default_covers_the_per_file_cap_sent_inline_as_base64():
    """The default is sized from the cap, not picked: a file at the cap sent
    base64-inline (``staged_files`` / ``attachments``) plus its JSON must fit."""
    base64_len = 4 * -(-DEFAULT_STAGE_PER_FILE_LIMIT // 3)
    assert base64_len + WS_MESSAGE_ENVELOPE_HEADROOM <= DEFAULT_WS_MAX_MESSAGE_SIZE


def test_a_file_over_one_mebibyte_is_accepted(tmp_path):
    async def run():
        server = _server(tmp_path)
        task, port = await _started(server)
        try:
            return await _stage(port, _PDF_SIZE, workspace=True)
        finally:
            await _stop(server, task)

    _connected, staged, close_code = asyncio.run(run())
    assert close_code is None, f"connection closed with {close_code}"
    assert staged is not None
    assert staged["staged"] == ["report.pdf"], staged
    assert (tmp_path / "report.pdf").stat().st_size == _PDF_SIZE


def test_the_handshake_advertises_the_limits(tmp_path):
    async def run():
        server = _server(tmp_path, max_message_size=2 * 1024 * 1024)
        task, port = await _started(server)
        try:
            import websockets
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                return json.loads(await ws.recv())
        finally:
            await _stop(server, task)

    info = asyncio.run(run())["server_info"]
    assert info["max_message_size"] == 2 * 1024 * 1024
    # A file travels as one message, so the per-file cap never exceeds it.
    assert info["stage_per_file_limit"] == 2 * 1024 * 1024
    assert info["stage_total_limit"] == DEFAULT_STAGE_TOTAL_LIMIT


def test_a_1009_close_is_logged_at_warning(tmp_path, caplog):
    async def run():
        server = _server(tmp_path, max_message_size=LEGACY_WS_MAX_MESSAGE_SIZE)
        task, port = await _started(server)
        try:
            import websockets
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                await ws.recv()
                await ws.send(b"x" * (LEGACY_WS_MAX_MESSAGE_SIZE + 1))
                try:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                except websockets.ConnectionClosed:
                    pass
            # Let the server's handler reach its except clause.
            for _ in range(100):
                if not server._clients:
                    break
                await asyncio.sleep(0.01)
        finally:
            await _stop(server, task)

    with caplog.at_level(logging.DEBUG, logger="jaato_server.server.websocket"):
        asyncio.run(run())
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "1009" in r.getMessage()]
    assert warnings, [r.getMessage() for r in caplog.records]
    assert "max_message_size" in warnings[0].getMessage()


def test_an_ordinary_close_stays_quiet(tmp_path, caplog):
    """Control: a client that closes normally must not produce the warning,
    or the warning is noise an operator learns to ignore."""
    async def run():
        server = _server(tmp_path)
        task, port = await _started(server)
        try:
            import websockets
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.recv()
            for _ in range(100):
                if not server._clients:
                    break
                await asyncio.sleep(0.01)
        finally:
            await _stop(server, task)

    with caplog.at_level(logging.DEBUG, logger="jaato_server.server.websocket"):
        asyncio.run(run())
    assert not [r for r in caplog.records if "closed abnormally" in r.getMessage()]


@pytest.mark.parametrize("raw, expected", [
    ("16M", 16 * 1024 * 1024),
    ("1048576", 1024 * 1024),
    ("2g", 2 * 1024 ** 3),
    ("32MiB", 32 * 1024 * 1024),
])
def test_the_flag_parses_sizes(raw, expected):
    assert parse_ws_max_message_size(raw) == expected


@pytest.mark.parametrize("raw", ["", "lots", "512K", "-5M", "1.5M"])
def test_the_flag_refuses_nonsense_and_values_below_the_floor(raw):
    with pytest.raises(ValueError):
        parse_ws_max_message_size(raw)
