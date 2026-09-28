"""An application that cannot reach a workspace has the daemon write its reference.

Reported from a deployment whose BFF runs as an ordinary account beside a root
daemon: GitHub was connected and bound to ten workspaces, and no session had
``GH_TOKEN``.  The daemon resolves ``app://`` only for references it finds in
the workspace ``.env``, and the BFF, which writes that line, could not reach
``/root/.jaato/workspaces``, so it recorded each binding and skipped the file.
``workspace.app_write`` (1.30) has the daemon, which owns the workspaces, do
the write on the application's behalf, under rules the daemon enforces.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from jaato_sdk.events import (
    EventType,
    WorkspaceAppWriteRequest,
    deserialize_event,
)
from jaato_server.server import workspace_app_write as waw
from jaato_server.server.websocket import SECRET_BIND_VERBS, JaatoWSServer
from jaato_server.shared.tests.reversion import Reversion

_MOD = "jaato-server/jaato_server/server/workspace_app_write.py"
_WS = "jaato-server/jaato_server/server/websocket.py"

REVERSIONS = [
    Reversion(
        target=_MOD,
        find="    if value is not None and parse_app_secret_reference(value) is None:\n",
        replace="    if False:\n",
        test="test_a_literal_value_is_refused_and_nothing_is_written",
        because="the verb writes a secret to disk",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/contained_write.py",
        find="            if not under(real, root):\n",
        replace="            if False:\n",
        test="test_a_planted_symlink_cannot_carry_the_write_outside",
        because="a root daemon writes wherever a model-planted link points",
    ),
    Reversion(
        target=_MOD,
        find="            if parse_app_secret_reference(current) is None:\n                actions[name] = \"kept-literal\"\n",
        replace="            if False:\n                actions[name] = \"kept-literal\"\n",
        test="test_a_removal_leaves_a_literal_token_alone",
        because="unbinding deletes a token the user typed",
    ),
    Reversion(
        target=_WS,
        find="        if self._owner_for_workspace_path(event.workspace) != owner:\n",
        replace="        if False:\n",
        test="test_only_the_owner_may_write",
        because="one application writes into another's workspace",
    ),
    Reversion(
        target=_WS,
        find="    EventType.WORKSPACE_APP_WRITE_REQUEST.value,\n})",
        replace="})",
        test="test_the_verb_rides_the_bind_channel",
        because="the verb falls to the app-connection refusal and never answers",
    ),
]


def _server(owner: str | None):
    return SimpleNamespace(_owner_for_workspace_path=lambda path: owner)


def _write(ws, owner, **kw):
    event = WorkspaceAppWriteRequest(request_id="r1", user="alice", workspace=str(ws), **kw)
    return JaatoWSServer._apply_workspace_app_write(_server(owner), "webcoder", event)


def test_the_reference_line_is_written_beside_what_is_there(tmp_path):
    (tmp_path / ".env").write_text("JAATO_MINIMAX_API_KEY=k\n")
    ans = _write(tmp_path, "webcoder:alice", env={"GH_TOKEN": "app://github"})
    assert ans.status == "ok" and ans.env == {"GH_TOKEN": "written"}
    assert (tmp_path / ".env").read_text() == "JAATO_MINIMAX_API_KEY=k\nGH_TOKEN=app://github\n"
    again = _write(tmp_path, "webcoder:alice", env={"GH_TOKEN": "app://github"})
    assert again.env == {"GH_TOKEN": "unchanged"}


def test_a_literal_value_is_refused_and_nothing_is_written(tmp_path):
    ans = _write(tmp_path, "webcoder:alice",
                 env={"GH_TOKEN": "ghu_secret"}, files=[{"path": ".home/.gitconfig", "content": "x"}])
    assert ans.status == "denied"
    assert not (tmp_path / ".env").exists() and not (tmp_path / ".home").exists()


def test_a_removal_leaves_a_literal_token_alone(tmp_path):
    (tmp_path / ".env").write_text("GH_TOKEN=ghp_typed_by_the_user\n")
    ans = _write(tmp_path, "webcoder:alice", env={"GH_TOKEN": None})
    assert ans.env == {"GH_TOKEN": "kept-literal"}
    assert "ghp_typed_by_the_user" in (tmp_path / ".env").read_text()
    (tmp_path / ".env").write_text("GH_TOKEN=app://github\n")
    assert _write(tmp_path, "webcoder:alice", env={"GH_TOKEN": None}).env == {"GH_TOKEN": "removed"}
    assert (tmp_path / ".env").read_text() == ""


def test_only_the_owner_may_write(tmp_path):
    for owner in (None, "webcoder:bob", "otherapp:alice"):
        ans = _write(tmp_path, owner, env={"GH_TOKEN": "app://github"})
        assert ans.status == "not_found"
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("path", [
    ".env", "../x", "/etc/passwd", ".home/.bashrc", ".jaato/instructions/../../x.md",
    ".jaato/instructions/sub/x.md", ".jaato/profiles/x.yaml", ".jaato/instructions/x.txt",
])
def test_only_allow_listed_paths(tmp_path, path):
    ans = _write(tmp_path, "webcoder:alice", files=[{"path": path, "content": "x"}])
    assert ans.status == "denied"


def test_a_planted_symlink_cannot_carry_the_write_outside(tmp_path):
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir(); outside.mkdir()
    os.symlink(outside, ws / ".home")
    ans = _write(ws, "webcoder:alice", files=[{"path": ".home/.gitconfig", "content": "[user]\n"}])
    assert ans.files[0]["action"] == "error"
    assert not (outside / ".gitconfig").exists()


def test_a_managed_file_respects_a_copy_the_user_made_their_own(tmp_path):
    marker = "<!-- jaato-managed: github-guidance v1 — delete this line to keep your own edits -->\n"
    path = ".jaato/instructions/40-github.md"
    f = {"path": path, "content": marker + "rules\n", "managed_by": "github-guidance"}
    assert _write(tmp_path, "webcoder:alice", files=[f]).files[0]["action"] == "written"
    assert _write(tmp_path, "webcoder:alice", files=[f]).files[0]["action"] == "unchanged"
    (tmp_path / path).write_text("my own rules\n")
    assert _write(tmp_path, "webcoder:alice", files=[f]).files[0]["action"] == "skipped-user-file"
    gone = {"path": path, "content": None, "managed_by": "github-guidance"}
    assert _write(tmp_path, "webcoder:alice", files=[gone]).files[0]["action"] == "skipped-user-file"
    (tmp_path / path).write_text(marker + "rules\n")
    assert _write(tmp_path, "webcoder:alice", files=[gone]).files[0]["action"] == "removed"


def test_the_gitconfig_is_private(tmp_path):
    _write(tmp_path, "webcoder:alice", files=[{"path": ".home/.gitconfig", "content": "[user]\n"}])
    assert (tmp_path / ".home" / ".gitconfig").stat().st_mode & 0o777 == 0o600


def test_the_verb_rides_the_bind_channel():
    assert EventType.WORKSPACE_APP_WRITE_REQUEST.value in SECRET_BIND_VERBS
    ev = deserialize_event('{"type": "workspace.app_write", "request_id": "x", "user": "a",'
                           ' "workspace": "/w", "env": {"GH_TOKEN": "app://github"}}')
    assert isinstance(ev, WorkspaceAppWriteRequest) and ev.env == {"GH_TOKEN": "app://github"}


def test_the_text_transform_matches_the_bffs():
    assert waw.upsert_env_line("", "K", "v") == "K=v\n"
    assert waw.upsert_env_line("A=1\nexport K=old\n", "K", "v") == "A=1\nK=v\n"
    assert waw.upsert_env_line("K=v\n", "K", None) == ""
