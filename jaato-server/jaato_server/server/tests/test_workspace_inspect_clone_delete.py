"""``workspace.inspect``, ``workspace.clone`` and ``delete --stop_sessions`` (1.26).

The WS handlers are driven on a ``JaatoWSServer.__new__`` carrying a real
``WorkspaceManager`` and a fake session manager, so the refusals are the
manager's own.  Clones run against a LOCAL bare repository through the
injectable URL builder -- no test touches the network.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from jaato_sdk.events import (
    WorkspaceCloneRequest,
    WorkspaceDeleteRequest,
    WorkspaceInspectRequest,
)
from jaato_server.server import workspace_clone
from jaato_server.server.websocket import JaatoWSServer
from jaato_server.server.workspace_clone import (
    clone_repos,
    git_auth_env,
    resolve_workspace_token,
    scrub,
    spec_error,
)
from jaato_server.server.workspace_inspect import (
    repo_status,
    session_counts,
    tree_size,
)
from jaato_server.server.workspace_manager import WorkspaceManager


def _git(*args: str, cwd: Path | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def bare_repo(tmp_path: Path) -> Path:
    """A bare repo with branches ``main`` and ``dev``, one commit each."""
    src = tmp_path / "src"
    src.mkdir()
    _git("init", "-q", "-b", "main", cwd=src)
    (src / "README").write_text("hello\n")
    _git("add", ".", cwd=src)
    _git("commit", "-q", "-m", "init", cwd=src)
    _git("checkout", "-q", "-b", "dev", cwd=src)
    (src / "DEV").write_text("dev\n")
    _git("add", ".", cwd=src)
    _git("commit", "-q", "-m", "dev", cwd=src)
    bare = tmp_path / "remote" / "octo" / "repo.git"
    bare.parent.mkdir(parents=True)
    _git("clone", "-q", "--bare", str(src), str(bare))
    return bare


def _url_for(tmp_path: Path):
    return lambda repo: str(tmp_path / "remote" / f"{repo}.git")


# ------------------------------------------------------------------- clone

def _collect():
    events: List[Dict[str, Any]] = []

    async def emit(fields):
        events.append(dict(fields))

    return events, emit


def test_clone_emits_queued_then_progress_to_done(tmp_path, bare_repo):
    ws = tmp_path / "ws"
    ws.mkdir()
    events, emit = _collect()
    asyncio.run(clone_repos(ws, [{"repo": "octo/repo", "branch": "dev",
                                  "forge": "github"}], None, emit,
                            url_for=_url_for(tmp_path)))
    states = [e["state"] for e in events]
    assert states[0] == "queued"
    assert states[-1] == "done"
    assert "checkout" in states
    assert events[-1]["done"] == events[-1]["total"] == 1
    assert (ws / "repo" / "DEV").read_text() == "dev\n"
    assert _git("-C", str(ws / "repo"), "rev-parse", "--abbrev-ref", "HEAD").strip() == "dev"


def test_every_repo_is_queued_before_any_is_cloned(tmp_path, bare_repo):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "repo").mkdir()          # the first target already exists
    events, emit = _collect()
    asyncio.run(clone_repos(ws, [
        {"repo": "octo/repo", "branch": "main"},
        {"repo": "../evil", "branch": "main"},
        {"repo": "octo/other", "branch": "main", "forge": "gitlab"},
    ], None, emit, url_for=_url_for(tmp_path)))
    assert [e["state"] for e in events[:3]] == ["queued"] * 3
    finals = [e for e in events[3:] if e["state"] in ("done", "failed")]
    assert [e["state"] for e in finals] == ["failed"] * 3
    assert "already exists" in finals[0]["error"]
    assert "invalid repo" in finals[1]["error"]
    assert "forge not supported" in finals[2]["error"]
    assert [e["done"] for e in finals] == [1, 2, 3]
    assert list((ws / "repo").iterdir()) == []   # the existing dir untouched


def test_a_failed_clone_removes_its_partial_target(tmp_path, bare_repo):
    ws = tmp_path / "ws"
    ws.mkdir()
    events, emit = _collect()
    asyncio.run(clone_repos(ws, [{"repo": "octo/missing", "branch": "main"}],
                            None, emit, url_for=_url_for(tmp_path)))
    assert events[-1]["state"] == "failed"
    assert events[-1]["error"]
    assert not (ws / "missing").exists()


def test_a_bad_branch_fails_and_leaves_nothing(tmp_path, bare_repo):
    ws = tmp_path / "ws"
    ws.mkdir()
    events, emit = _collect()
    asyncio.run(clone_repos(ws, [{"repo": "octo/repo", "branch": "nope"}],
                            None, emit, url_for=_url_for(tmp_path)))
    assert events[-1]["state"] == "failed"
    assert not (ws / "repo").exists()


@pytest.mark.parametrize("repo,branch,forge,ok", [
    ("octo/repo", "main", "github", True),
    ("octo/repo", "", "github", True),
    ("octo/..", "main", "github", False),
    ("octo/.jaato", "main", "github", False),
    ("octo", "main", "github", False),
    ("octo/repo", "-x", "github", False),
    ("octo/repo", "a..b", "github", False),
    ("octo/repo", "has space", "github", False),
    ("octo/repo", "main", "bitbucket", False),
])
def test_spec_validation(repo, branch, forge, ok):
    assert (spec_error(repo, branch, forge) == "") is ok


def test_the_token_rides_the_environment_never_argv(tmp_path, bare_repo, monkeypatch):
    seen: Dict[str, Any] = {}
    real = asyncio.create_subprocess_exec

    async def spy(*argv, env=None, **kw):
        seen["argv"], seen["env"] = list(argv), env
        return await real(*argv, env=env, **kw)

    monkeypatch.setattr(workspace_clone.asyncio, "create_subprocess_exec", spy)
    ws = tmp_path / "ws"
    ws.mkdir()
    events, emit = _collect()
    asyncio.run(clone_repos(ws, [{"repo": "octo/repo", "branch": "main"}],
                            "ghp_TOPSECRET", emit, url_for=_url_for(tmp_path)))
    assert events[-1]["state"] == "done"
    assert not any("ghp_TOPSECRET" in a for a in seen["argv"])
    env = seen["env"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_KEY_0"] == "credential.helper"
    assert env["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_1"].startswith("AUTHORIZATION: basic ")
    assert "ghp_TOPSECRET" not in (ws / "repo" / ".git" / "config").read_text()


def test_anonymous_env_still_resets_the_credential_helper():
    env = git_auth_env(None)
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "credential.helper"
    assert "GIT_CONFIG_KEY_1" not in env


def test_scrub_masks_raw_and_basic_forms():
    import base64
    basic = base64.b64encode(b"x-access-token:tok123").decode()
    assert scrub(f"a tok123 b {basic}", "tok123") == "a *** b ***"
    assert scrub("nothing", None) == "nothing"


def test_token_is_read_from_the_workspace_env(tmp_path):
    (tmp_path / ".env").write_text("GH_TOKEN=ghp_literal\n")
    assert resolve_workspace_token(tmp_path) == "ghp_literal"
    assert resolve_workspace_token(tmp_path / "nope") is None


def test_an_app_reference_is_resolved_for_the_workspace_owner(tmp_path):
    from jaato_server.server.app_secret import AppSecretAnswer
    (tmp_path / ".env").write_text("GH_TOKEN=app://github\n")
    asked = []

    class _Resolver:
        def owner_for(self, path):
            return "acme:alice"

        def resolve_reference(self, ref, context):
            asked.append((ref.name, context.workspace_owner, context.workspace_path))
            return AppSecretAnswer(status="ok", value="ghu_minted")

    assert resolve_workspace_token(tmp_path, _Resolver()) == "ghu_minted"
    assert asked == [("github", "acme:alice", str(tmp_path))]
    # No resolver wired: the reference is never forwarded as a token.
    assert resolve_workspace_token(tmp_path, None) is None


# ----------------------------------------------------------------- inspect

def test_tree_size_sums_files_and_gives_up_past_the_bound(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 10)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b").write_bytes(b"y" * 5)
    os.symlink(tmp_path / "a", tmp_path / "link")
    assert tree_size(tmp_path) == 15
    assert tree_size(tmp_path, max_entries=2) is None


def test_session_counts_by_state(tmp_path):
    ws = str(tmp_path)
    rows = [
        SimpleNamespace(workspace_path=ws, is_loaded=False, awaiting=None),
        SimpleNamespace(workspace_path=ws, is_loaded=True, awaiting=None),
        SimpleNamespace(workspace_path=ws, is_loaded=True, awaiting="permission"),
        SimpleNamespace(workspace_path=str(tmp_path / "other"), is_loaded=True, awaiting=None),
        SimpleNamespace(workspace_path=None, is_loaded=True, awaiting=None),
    ]
    assert session_counts(rows, ws) == {"total": 3, "waiting": 1, "awake": 1, "sleeping": 1}


def test_repo_status_counts_uncommitted_and_unpushed(tmp_path, bare_repo):
    clone = tmp_path / "c"
    _git("clone", "-q", "-b", "main", str(bare_repo), str(clone))
    assert repo_status(clone) == {"uncommitted": 0, "unpushed": 0, "error": ""}
    (clone / "new").write_text("n\n")
    (clone / "README").write_text("changed\n")
    assert repo_status(clone)["uncommitted"] == 2
    _git("add", ".", cwd=clone)
    _git("commit", "-q", "-m", "local", cwd=clone)
    assert repo_status(clone) == {"uncommitted": 0, "unpushed": 1, "error": ""}
    _git("checkout", "-q", "-b", "no-upstream", cwd=clone)
    assert repo_status(clone) == {"uncommitted": 0, "unpushed": None, "error": ""}


def test_repo_status_reports_a_git_failure(tmp_path):
    status = repo_status(tmp_path / "not-a-repo")
    assert status["uncommitted"] is None and status["error"]


# ------------------------------------------------------- the WS handlers

class _Sessions:
    """The two ``SessionManager`` methods the handlers reach."""

    def __init__(self, rows):
        self.rows = rows
        self.deleted: List[str] = []

    def list_sessions(self):
        return [r for r in self.rows if r.session_id not in self.deleted]

    def delete_session(self, session_id):
        self.deleted.append(session_id)
        return True


def _server(root: Path, rows=(), user=None):
    manager = WorkspaceManager(str(root), registry_path=root.parent / "reg.json")
    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._workspace_manager = manager
    ws._event_sink_adapter = None
    ws._app_secret_resolver = None
    sessions = _Sessions(list(rows))
    ws._command_router = SimpleNamespace(_session_manager=sessions)
    sent: List[Any] = []

    async def send(client_id, event):
        sent.append(event)

    async def send_error(client_id, message):
        sent.append(("error", message))

    ws._send_to_client = send
    ws._send_error = send_error
    ws.get_client_user = lambda client_id: user
    return ws, manager, sessions, sent


async def _drive(ws, event):
    await ws._handle_workspace_event("c1", event)
    tasks = list(getattr(ws, "_background_tasks", ()))
    if tasks:
        await asyncio.gather(*tasks)


def _root_with(tmp_path: Path, owner: str | None = None) -> Path:
    root = tmp_path / "root"
    (root / "proj" / ".jaato").mkdir(parents=True)
    (root / "proj" / "notes.txt").write_text("12345")
    if owner:
        m = WorkspaceManager(str(root), registry_path=root.parent / "reg.json")
        m.list_workspaces()
        m._workspaces["proj"].owner = owner
        m._save_registry()
    return root


def test_inspect_answers_with_counts_repos_and_size(tmp_path, bare_repo):
    root = _root_with(tmp_path)
    _git("clone", "-q", "-b", "main", str(bare_repo), str(root / "proj" / "repo"))
    rows = [SimpleNamespace(session_id="s1", workspace_path=str(root / "proj"),
                            is_loaded=False, awaiting=None)]
    ws, _, _, sent = _server(root, rows)
    asyncio.run(_drive(ws, WorkspaceInspectRequest(name="proj", request_id="r1")))
    (ev,) = sent
    assert ev.type == "workspace.inspected" and ev.ok and ev.request_id == "r1"
    assert ev.sessions == {"total": 1, "waiting": 0, "awake": 0, "sleeping": 1}
    (repo,) = ev.repos
    assert repo["path"] == "repo" and repo["uncommitted"] == 0 and repo["unpushed"] == 0
    assert ev.size_bytes and ev.size_bytes >= 5
    assert ev.path == str((root / "proj").resolve())


@pytest.mark.parametrize("name", ["../etc", "", "missing"])
def test_inspect_refuses_what_select_refuses(tmp_path, name):
    ws, _, _, sent = _server(_root_with(tmp_path))
    asyncio.run(_drive(ws, WorkspaceInspectRequest(name=name, request_id="r")))
    (ev,) = sent
    assert ev.ok is False and ev.error and ev.request_id == "r"


def test_inspect_and_clone_refuse_another_users_workspace(tmp_path):
    root = _root_with(tmp_path, owner="app:bob")
    ws, _, _, sent = _server(root, user="app:alice")
    asyncio.run(_drive(ws, WorkspaceInspectRequest(name="proj", request_id="r")))
    asyncio.run(_drive(ws, WorkspaceCloneRequest(
        name="proj", request_id="c", repos=[{"repo": "o/r", "branch": "main"}])))
    inspect_ev, clone_ev = sent
    assert inspect_ev.ok is False and "another user" in inspect_ev.error
    assert clone_ev.state == "failed" and clone_ev.repo == "" and clone_ev.request_id == "c"


def test_clone_handler_streams_into_the_workspace(tmp_path, bare_repo, monkeypatch):
    monkeypatch.setattr(workspace_clone, "github_url", _url_for(tmp_path))
    monkeypatch.setattr(workspace_clone.clone_repos, "__defaults__",
                        (_url_for(tmp_path),))
    root = _root_with(tmp_path)
    ws, _, _, sent = _server(root)
    asyncio.run(_drive(ws, WorkspaceCloneRequest(
        name="proj", request_id="c1", repos=[{"repo": "octo/repo", "branch": "main"}])))
    assert sent[0].state == "queued" and sent[-1].state == "done"
    assert all(e.type == "workspace.clone_progress" and e.request_id == "c1"
               and e.name == "proj" for e in sent)
    assert (root / "proj" / "repo" / "README").is_file()


def _loaded(session_id, path):
    return SimpleNamespace(session_id=session_id, workspace_path=str(path),
                           is_loaded=True, awaiting=None)


def test_delete_refuses_loaded_sessions_by_default(tmp_path):
    root = _root_with(tmp_path)
    ws, _, sessions, sent = _server(root, [_loaded("s1", root / "proj")])
    asyncio.run(_drive(ws, WorkspaceDeleteRequest(name="proj")))
    (ev,) = sent
    assert ev.ok is False and "loaded session" in ev.error
    assert sessions.deleted == [] and (root / "proj").is_dir()


def test_delete_with_stop_sessions_deletes_them_then_the_workspace(tmp_path):
    root = _root_with(tmp_path)
    ws, _, sessions, sent = _server(root, [_loaded("s1", root / "proj"),
                                           _loaded("s2", tmp_path / "elsewhere")])
    asyncio.run(_drive(ws, WorkspaceDeleteRequest(name="proj", stop_sessions=True)))
    (ev,) = sent
    assert ev.ok is True
    assert sessions.deleted == ["s1"]
    assert not (root / "proj").exists()


def test_stop_sessions_never_stops_a_session_when_the_delete_is_refused(tmp_path):
    root = _root_with(tmp_path, owner="app:bob")
    ws, _, sessions, sent = _server(root, [_loaded("s1", root / "proj")], user="app:alice")
    asyncio.run(_drive(ws, WorkspaceDeleteRequest(name="proj", stop_sessions=True)))
    (ev,) = sent
    assert ev.ok is False and "another user" in ev.error
    assert sessions.deleted == []

    root2 = tmp_path / "second"
    (root2 / "proj" / ".jaato").mkdir(parents=True)
    ws2, manager2, sessions2, sent2 = _server(root2, [_loaded("s1", root2 / "proj")])
    manager2.select_workspace("proj", client_id="other-client")
    asyncio.run(_drive(ws2, WorkspaceDeleteRequest(name="proj", stop_sessions=True)))
    (ev2,) = sent2
    assert ev2.ok is False and "other client" in ev2.error
    assert sessions2.deleted == []
