"""``WorkspaceInfo.sources`` -- the git checkouts a workspace holds (1.27).

Derived from ``.git/HEAD`` and ``.git/config`` read as files; no git process
runs.  Credentials in a remote URL must never survive into a source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_server.server.workspace_manager import WorkspaceManager
from jaato_server.server.workspace_sources import (
    parse_remote,
    read_head_branch,
    workspace_sources,
)


def _fake_checkout(path: Path, head: str = "ref: refs/heads/main",
                   origin: str | None = None) -> Path:
    git = path / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(head + "\n")
    config = "[core]\n\tbare = false\n"
    if origin is not None:
        config += f'[remote "upstream"]\n\turl = https://example.com/x/y.git\n'
        config += f'[remote "origin"]\n\turl = {origin}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
    (git / "config").write_text(config)
    return path


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/octo/repo.git", ("github", "octo/repo")),
    ("https://github.com/octo/repo", ("github", "octo/repo")),
    ("https://x-access-token:ghp_SECRET@github.com/octo/repo.git", ("github", "octo/repo")),
    ("https://user:pw@gitlab.com/grp/sub/proj.git", ("gitlab", "grp/sub/proj")),
    ("git@github.com:octo/repo.git", ("github", "octo/repo")),
    ("ssh://git@github.com/octo/repo.git", ("github", "octo/repo")),
    ("ssh://git@git.corp.example:2222/team/app.git", ("git.corp.example", "team/app")),
    ("/srv/git/local.git", ("", "")),
    ("", ("", "")),
])
def test_parse_remote(url, expected):
    assert parse_remote(url) == expected


def test_a_credential_in_the_remote_never_survives(tmp_path):
    _fake_checkout(tmp_path / "ws" / "app",
                   origin="https://x-access-token:ghp_SECRET@github.com/octo/app.git")
    sources = workspace_sources(tmp_path / "ws")
    assert "ghp_SECRET" not in json.dumps(sources)
    assert sources == [{"forge": "github", "repo": "octo/app",
                        "branch": "main", "path": "app"}]


def test_root_and_children_skipping_hidden_and_framework_dirs(tmp_path):
    ws = tmp_path / "ws"
    _fake_checkout(ws, origin="git@github.com:me/root.git")
    _fake_checkout(ws / "b-lib", head="ref: refs/heads/feature/x",
                   origin="https://gitlab.com/g/lib.git")
    _fake_checkout(ws / "a-nogit-origin")          # no origin
    _fake_checkout(ws / ".hidden", origin="git@github.com:h/h.git")
    _fake_checkout(ws / ".jaato", origin="git@github.com:j/j.git")
    (ws / "plain").mkdir()
    assert workspace_sources(ws) == [
        {"forge": "github", "repo": "me/root", "branch": "main", "path": "."},
        {"forge": "", "repo": "", "branch": "main", "path": "a-nogit-origin"},
        {"forge": "gitlab", "repo": "g/lib", "branch": "feature/x", "path": "b-lib"},
    ]


def test_detached_head_reports_the_short_sha(tmp_path):
    repo = _fake_checkout(tmp_path / "r", head="0123456789abcdef0123456789abcdef01234567")
    assert read_head_branch(repo / ".git") == "0123456"


def test_a_gitdir_file_is_followed(tmp_path):
    real = _fake_checkout(tmp_path / "real", origin="git@github.com:o/wt.git")
    wt = tmp_path / "ws" / "wt"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {real / '.git'}\n")
    assert workspace_sources(tmp_path / "ws") == [
        {"forge": "github", "repo": "o/wt", "branch": "main", "path": "wt"}]


def test_real_git_checkout(tmp_path):
    import subprocess
    repo = tmp_path / "ws" / "real"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "trunk", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin",
                    "https://github.com/octo/real.git"], check=True)
    assert workspace_sources(tmp_path / "ws") == [
        {"forge": "github", "repo": "octo/real", "branch": "trunk", "path": "real"}]


def test_the_listing_carries_sources_and_the_registry_does_not(tmp_path):
    root = tmp_path / "root"
    _fake_checkout(root / "proj" / "app", origin="git@github.com:o/app.git")
    (root / "proj" / ".jaato").mkdir()
    registry = tmp_path / "registry.json"
    manager = WorkspaceManager(str(root), registry_path=registry)
    (ws,) = manager.list_workspaces()
    assert ws.to_dict()["sources"] == [
        {"forge": "github", "repo": "o/app", "branch": "main", "path": "app"}]
    rows = json.loads(registry.read_text())["workspaces"]
    assert "sources" not in rows[0]
