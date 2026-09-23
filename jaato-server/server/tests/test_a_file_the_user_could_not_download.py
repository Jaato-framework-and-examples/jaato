"""A remote client could put a file INTO a workspace and take none OUT.

``StageFilesRequest`` uploads; there was no reverse, so an asset the agent
produced in a server-provisioned workspace was reachable only by somebody
with a shell on the host.  ``workspace.file.fetch`` (protocol 1.20) is the
download: one TEXT ``WorkspaceFileContentEvent`` header and, on success,
ONE raw binary frame of exactly ``size`` bytes.

Two halves are pinned here, and each could fail silently:

* **what may leave** -- :func:`server.workspace_download.resolve_download`.
  A download link is something the MODEL can offer (``offer_download``), so
  the rules must hold whatever it was asked: nothing outside the workspace
  (symlinks followed first), and never the files that hold a provider key.
  Every refusal is paired with the same call one step apart, because a
  refusal that would have happened anyway proves nothing.
* **the adjacency the protocol rests on** -- the header carries no id the
  binary frame could be matched by, so the client takes "the next binary
  frame" as the file.  The two sends must happen under the one send lock,
  back to back.  The handler test asserts the frame order a client sees.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import (
    WorkspaceFileContentEvent,
    WorkspaceFileFetchRequest,
    deserialize_event,
)
from server.workspace_download import (
    is_credential_path,
    resolve_download,
)
from shared.tests.reversion import Reversion

_DL = "jaato-server/server/workspace_download.py"
_WS = "jaato-server/server/websocket.py"

REVERSIONS = [
    Reversion(
        target=_DL,
        find="    if target == root or root not in target.parents:",
        replace="    if False:",
        test="test_a_path_that_climbs_out_is_refused",
        because="a download names a path, and one that climbs out of the workspace would hand any file the daemon user can read to whoever holds the connection",
    ),
    Reversion(
        target=_DL,
        find="    target = candidate.resolve(strict=False)",
        replace="    target = Path(os.path.normpath(candidate))",
        test="test_a_symlink_is_judged_by_where_it_points",
        because="the agent's own file tools can plant a link inside the workspace; judging the link rather than its target lets it carry a file from outside",
    ),
    Reversion(
        target=_DL,
        find="    if is_credential_path(relpath):",
        replace="    if False:",
        test="test_the_workspace_env_never_leaves",
        because="the workspace .env holds the provider key config.update wrote, and offer_download lets the MODEL propose a link, so the key must be unreachable by name",
    ),
    Reversion(
        target=_WS,
        find="                await client.websocket.send(serialize_event(event))\n                await client.websocket.send(data)",
        replace="                await client.websocket.send(data)\n                await client.websocket.send(serialize_event(event))",
        test="test_the_bytes_follow_their_header",
        because="the header carries no id the binary frame could be matched by, so a client takes the next binary frame as the file; sent in the other order it would read bytes as JSON",
    ),
]


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "out").mkdir(parents=True)
    (root / "out" / "report.pdf").write_bytes(b"%PDF-1.7 fake")
    (root / ".jaato").mkdir()
    return root


# ---------------------------------------------------------------- the rules

def test_a_workspace_file_resolves_with_its_metadata(ws: Path) -> None:
    t = resolve_download(str(ws), "out/report.pdf")
    assert t.ok, t
    assert (t.relpath, t.name, t.size, t.mime_type) == (
        "out/report.pdf", "report.pdf", 13, "application/pdf",
    )


def test_an_absolute_path_inside_the_workspace_is_accepted(ws: Path) -> None:
    t = resolve_download(str(ws), str(ws / "out" / "report.pdf"))
    assert t.ok and t.relpath == "out/report.pdf"


def test_a_path_that_climbs_out_is_refused(ws: Path, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("x")  # EXISTS: refusal, not not_found
    assert resolve_download(str(ws), "../secret.txt").category == "unsafe_path"
    assert resolve_download(str(ws), str(tmp_path / "secret.txt")).category == "unsafe_path"
    # control: the same shape one step inside is fine
    assert resolve_download(str(ws), "out/../out/report.pdf").ok


def test_a_symlink_is_judged_by_where_it_points(ws: Path, tmp_path: Path) -> None:
    (tmp_path / "outside.txt").write_text("not yours")
    os.symlink(tmp_path / "outside.txt", ws / "out" / "link.txt")
    assert resolve_download(str(ws), "out/link.txt").category == "unsafe_path"
    # control: a link that stays inside the workspace is followed
    os.symlink(ws / "out" / "report.pdf", ws / "out" / "inner.pdf")
    t = resolve_download(str(ws), "out/inner.pdf")
    assert t.ok and t.relpath == "out/report.pdf"


def test_the_workspace_env_never_leaves(ws: Path) -> None:
    (ws / ".env").write_text("JAATO_PROVIDER=x\n")
    (ws / ".env.example").write_text("JAATO_PROVIDER=\n")
    (ws / ".jaato" / "nim_auth.json").write_text("{}")
    assert resolve_download(str(ws), ".env").category == "credential"
    assert resolve_download(str(ws), ".jaato/nim_auth.json").category == "credential"
    # controls: documentation of the variables is not a secret, and a
    # file merely NAMED *_auth.json outside .jaato is the user's own data
    assert resolve_download(str(ws), ".env.example").ok
    (ws / "out" / "x_auth.json").write_text("{}")
    assert resolve_download(str(ws), "out/x_auth.json").ok


def test_credential_names_are_matched_exactly() -> None:
    assert is_credential_path(".env")
    assert is_credential_path("sub/.env")
    assert is_credential_path(".jaato/anthropic_auth.json")
    assert not is_credential_path("envelope.env.md")
    assert not is_credential_path("jaato/nim_auth.json")


def test_what_is_not_a_regular_file_says_so(ws: Path) -> None:
    assert resolve_download(str(ws), "out").category == "not_a_file"
    assert resolve_download(str(ws), "out/missing.pdf").category == "not_found"
    assert resolve_download(str(ws), "").category == "unsafe_path"
    assert resolve_download(str(ws), ".").category == "unsafe_path"


def test_a_file_over_the_cap_is_refused(ws: Path) -> None:
    assert resolve_download(str(ws), "out/report.pdf", max_bytes=5).category == "too_large"
    assert resolve_download(str(ws), "out/report.pdf", max_bytes=13).ok


# ----------------------------------------------------------- the transport

@pytest.fixture
def server(ws: Path):
    from server.websocket import JaatoWSServer

    srv = JaatoWSServer.__new__(JaatoWSServer)
    srv._clients = {}
    srv._lock = asyncio.Lock()
    srv._client_provisioned = {}
    srv._workspace_manager = None
    srv._event_sink_adapter = None
    provisioned = MagicMock()
    provisioned.path = str(ws)
    srv._client_provisioned["c1"] = provisioned
    return srv


def _connect(srv) -> List[Any]:
    sent: List[Any] = []

    async def _send(msg: Any) -> None:
        sent.append(msg)

    client = MagicMock()
    client.websocket.send = _send
    srv._clients["c1"] = client
    return sent


def _fetch(srv, path: str, metadata_only: bool = False) -> None:
    asyncio.run(srv._handle_file_fetch_request(
        "c1", WorkspaceFileFetchRequest(request_id="r1", path=path, metadata_only=metadata_only),
    ))


def test_the_bytes_follow_their_header(server) -> None:
    sent = _connect(server)
    _fetch(server, "out/report.pdf")
    assert len(sent) == 2
    header = deserialize_event(sent[0])
    assert isinstance(header, WorkspaceFileContentEvent)
    assert header.ok and header.request_id == "r1" and header.size == 13
    assert sent[1] == b"%PDF-1.7 fake"


def test_metadata_only_sends_no_bytes(server) -> None:
    sent = _connect(server)
    _fetch(server, "out/report.pdf", metadata_only=True)
    assert len(sent) == 1
    header = deserialize_event(sent[0])
    assert header.ok and header.metadata_only and header.size == 13


def test_a_refusal_sends_a_header_and_no_bytes(server) -> None:
    sent = _connect(server)
    _fetch(server, "../../etc/passwd")
    assert len(sent) == 1
    header = deserialize_event(sent[0])
    assert not header.ok and header.category == "unsafe_path"
    assert header.request_id == "r1"


def test_a_client_in_no_workspace_is_told_so(server) -> None:
    server._client_provisioned.clear()
    sent = _connect(server)
    _fetch(server, "out/report.pdf")
    header = deserialize_event(sent[0])
    assert not header.ok and header.category == "workspace_not_found"
