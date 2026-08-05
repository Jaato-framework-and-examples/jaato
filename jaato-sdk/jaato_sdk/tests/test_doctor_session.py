"""``jaato-doctor --session <id>`` — the runtime session diagnostic.

Reads a session's logs under ``<workspace>/.jaato/logs/`` and reports whether
the runner-tier path plugins resolved the workspace (PASS=<ws>) or got
``workspace=none`` (FAIL) — the one-command form of the manual log-archaeology
the #344 runner-workspace regression required.  Synthetic logs keep it
deterministic (no live daemon / session timing).
"""

import os

from jaato_sdk.doctor import check_session, _plugin_workspaces, PASS, FAIL, WARN


def _mk(ws, sid, workspace_value, runner=True):
    logs = ws / ".jaato" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (ws / ".jaato" / "sessions").mkdir(parents=True, exist_ok=True)
    name = f"runner-{sid}.log" if runner else f"session_{sid}_client_ipc_1.log"
    (logs / name).write_text(
        "2026-06-21 [INFO] bootstrap ...\n"
        "FilesystemQueryPlugin initialized (max_results=500, timeout=30s, "
        f"workspace={workspace_value}, allow_tmp=True)\n")
    return logs


def _by(checks):
    return {c.name: c for c in checks}


def test_resolved_workspace_passes(tmp_path):
    _mk(tmp_path, "20260621_000001", str(tmp_path))
    (tmp_path / ".jaato" / "sessions" / "20260621_000001.json").write_text("{}")
    by = _by(check_session("20260621_000001", str(tmp_path)))
    assert by["runner-workspace"].status == PASS
    assert str(tmp_path) in by["runner-workspace"].detail
    assert by["session-record"].status == PASS


def test_workspace_none_fails_with_fix_pointer(tmp_path):
    _mk(tmp_path, "s1", "none")
    by = _by(check_session("s1", str(tmp_path)))
    assert by["runner-workspace"].status == FAIL
    assert "workspace=none" in by["runner-workspace"].detail
    assert "working_dir" in by["runner-workspace"].detail   # names the fix


def test_no_plugin_init_warns(tmp_path):
    logs = tmp_path / ".jaato" / "logs"
    logs.mkdir(parents=True)
    (logs / "session_s2_client_ipc_1.log").write_text("no plugin lines here\n")
    by = _by(check_session("s2", str(tmp_path)))
    assert by["runner-workspace"].status == WARN


def test_latest_resolves_newest_session(tmp_path):
    _mk(tmp_path, "aaa", "/old/ws")
    _mk(tmp_path, "bbb", str(tmp_path))
    logs = tmp_path / ".jaato" / "logs"
    os.utime(logs / "runner-aaa.log", (1000, 1000))   # explicit: bbb newer
    os.utime(logs / "runner-bbb.log", (2000, 2000))
    by = _by(check_session("latest", str(tmp_path)))
    assert by["session"].detail.endswith("bbb")
    assert by["runner-workspace"].status == PASS


def test_latest_no_sessions_fails(tmp_path):
    (tmp_path / ".jaato" / "logs").mkdir(parents=True)
    by = _by(check_session("latest", str(tmp_path)))
    assert by["session"].status == FAIL


def test_plugin_workspaces_prefers_nonnone(tmp_path):
    p = tmp_path / "r.log"
    p.write_text("FilesystemQueryPlugin initialized (workspace=none, allow_tmp=True)\n"
                 "FilesystemQueryPlugin initialized (workspace=/the/ws, allow_tmp=True)\n")
    vals = _plugin_workspaces(p)
    assert vals == ["none", "/the/ws"]
