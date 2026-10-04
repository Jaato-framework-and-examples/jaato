"""Any workspace file can be found by name, hidden or not (protocol 1.32).

The Files panel lists only what changed, so a file nobody touched, one the
user hid, or one git ignores could not be reached from the web client.
``workspace.files.search`` walks the whole workspace.  These tests pin what
it returns and what it must not:

- dotfiles and gitignored files are found; only ``.git`` internals are not;
- a directory symlink is not followed, so a link cannot widen the search
  beyond the workspace;
- a walk that stops at its bound says so (``truncated``), because "no
  match" from a partial search is not "no such file";
- credential files are listed but marked, so a client offers no download;
- the WS dispatch routes the request, and a client in no workspace is told.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import (
    WorkspaceFilesSearchRequest,
    WorkspaceFilesSearchResultEvent,
    deserialize_event,
)
from jaato_server.server import workspace_file_search as wfs
from jaato_server.server.workspace_file_search import search_workspace
from jaato_server.shared.tests.reversion import Reversion

_FS = "jaato-server/jaato_server/server/workspace_file_search.py"
_WS = "jaato-server/jaato_server/server/websocket.py"

REVERSIONS = [
    Reversion(
        target=_FS,
        find="                if entry.is_dir(follow_symlinks=False):",
        replace="                if entry.is_dir():",
        test="test_a_directory_symlink_is_not_followed",
        because="the agent can plant a directory link in the workspace; following it would list files from the rest of the host",
    ),
    Reversion(
        target=_FS,
        find="                    if entry.name not in SKIPPED_DIRS:",
        replace="                    if True:",
        test="test_git_internals_are_not_searched",
        because="the .git object store is thousands of entries nobody locates by name, and they crowd the real matches out of the capped answer",
    ),
    Reversion(
        target=_FS,
        find="                result.truncated = True",
        replace="                pass",
        test="test_a_walk_that_stops_early_says_so",
        because="a search that stopped at its bound and reports nothing reads as 'no such file'",
    ),
    Reversion(
        target=_WS,
        find="        if isinstance(event, WorkspaceFilesSearchRequest):",
        replace="        if False:",
        test="test_the_ws_dispatch_routes_a_search",
        because="without the route the verb is never answered and the client waits out its timeout",
    ),
]


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "report_writer.py").write_text("x")
    (root / "docs").mkdir()
    (root / "docs" / "report.md").write_text("hello")
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / "node_modules" / "lib" / "report.js").write_text("js")
    (root / ".hidden_report").write_text("dot")
    (root / ".env").write_text("KEY=secret")
    (root / ".git" / "objects").mkdir(parents=True)
    (root / ".git" / "objects" / "report_blob").write_text("git")
    (root / ".gitignore").write_text("node_modules/\n")
    return root


def _paths(result) -> List[str]:
    return [m["path"] for m in result.matches]


def test_hidden_and_ignored_files_are_found(ws: Path) -> None:
    found = _paths(search_workspace(str(ws), "report"))
    assert "docs/report.md" in found
    assert ".hidden_report" in found
    assert "node_modules/lib/report.js" in found
    assert "src/pkg/report_writer.py" in found


def test_git_internals_are_not_searched(ws: Path) -> None:
    assert not any(p.startswith(".git/") for p in _paths(search_workspace(str(ws), "report")))


def test_every_term_must_appear_and_case_is_ignored(ws: Path) -> None:
    assert _paths(search_workspace(str(ws), "SRC Writer")) == ["src/pkg/report_writer.py"]
    assert _paths(search_workspace(str(ws), "")) == []


def test_a_name_match_ranks_above_a_directory_match(ws: Path) -> None:
    (ws / "report_dir").mkdir()
    (ws / "report_dir" / "notes.txt").write_text("n")
    found = _paths(search_workspace(str(ws), "report"))
    assert found.index("docs/report.md") < found.index("report_dir/notes.txt")


def test_a_credential_file_is_listed_and_marked(ws: Path) -> None:
    [env] = search_workspace(str(ws), ".env").matches
    assert env == {"path": ".env", "size": 10, "credential": True}


def test_a_directory_symlink_is_not_followed(ws: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "report_secret.txt").write_text("s")
    os.symlink(outside, ws / "link")
    assert not any(p.startswith("link/") for p in _paths(search_workspace(str(ws), "report")))


def test_the_answer_is_capped_and_counts_the_rest(ws: Path) -> None:
    result = search_workspace(str(ws), "report", max_results=2)
    assert len(result.matches) == 2
    assert result.total == 4
    assert not result.truncated


def test_a_walk_that_stops_early_says_so(ws: Path, monkeypatch) -> None:
    monkeypatch.setattr(wfs, "MAX_SCANNED_ENTRIES", 3)
    assert search_workspace(str(ws), "report").truncated


# ----------------------------------------------------------- the transport

@pytest.fixture
def server(ws: Path):
    from jaato_server.server.websocket import JaatoWSServer
    from jaato_server.server.ws_tickets import AppCredentialStore

    srv = JaatoWSServer.__new__(JaatoWSServer)
    # No per-application workspace roots configured.
    srv._clients = {}
    srv._app_managers = {}
    srv._app_provisioners = {}
    srv._app_credentials = AppCredentialStore({})
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


def test_the_ws_dispatch_routes_a_search(server) -> None:
    sent = _connect(server)
    handled = asyncio.run(server._dispatch_workspace_file_transfer(
        "c1", WorkspaceFilesSearchRequest(request_id="s1", query="report.md"),
    ))
    assert handled
    [answer] = [deserialize_event(m) for m in sent]
    assert isinstance(answer, WorkspaceFilesSearchResultEvent)
    assert answer.ok and answer.request_id == "s1" and answer.query == "report.md"
    assert [m["path"] for m in answer.matches] == ["docs/report.md"]
    assert answer.total == 1


def test_a_client_in_no_workspace_is_told_so(server) -> None:
    server._client_provisioned.clear()
    sent = _connect(server)
    asyncio.run(server._handle_file_search_request(
        "c1", WorkspaceFilesSearchRequest(request_id="s2", query="report"),
    ))
    answer = deserialize_event(sent[0])
    assert not answer.ok and answer.category == "workspace_not_found" and answer.request_id == "s2"
