"""The trace and the ledger a runner appends to are the workspace owner's (#1674).

``.jaato/logs/session_trace.jsonl`` and ``.jaato/logs/ledger.jsonl`` are
appended to by the daemon (its own trace lines, the Art. 50(1)
announcement row) AND by the session's runner.  Under a runner-uid policy
that drops (#1168), a root daemon that created either file first created it
``root:root 0644``, and the runner -- the workspace owner -- could never
append again: ``ledger: cannot append to ... ([Errno 13] Permission
denied)``, the runner's ledger rows lost.  #1528 covered the session record
and the client logs, not these two writers, which live in the SDK and in
``shared`` and cannot ask the daemon's policy.

Now :mod:`jaato_sdk.owned_append` is the one append both go through, and
the daemon installs :func:`server.runner_user.workspace_append_owner` as its
owner resolver when its policy drops.  A file the daemon creates is the
owner's from its first byte; one an earlier daemon left root-owned is
handed back on the daemon's next append, and at spawn
(:func:`server.runner_user.hand_over_audit_files`), since the daemon may
never append to a session's ledger again.

Ownership is simulated as in the #1528 guard: ``workspace_file_owner``
names a foreign uid and ``os.fchown`` / ``os.lchown`` are recorded per
inode.  Files the test creates are owned by its own uid, which is
``os.geteuid()``, so they read as "left by the daemon".
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

import pytest

from jaato_sdk import owned_append
from jaato_sdk.trace import trace_write
from jaato_server.server import runner_user
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.token_accounting import TokenLedger

REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/trace.py",
        find="        with open_append(resolved) as f:\n",
        replace='        with open(resolved, "a") as f:\n',
        test="test_a_trace_the_daemon_creates_is_the_owners",
        because="the trace opened with a plain append, created as the daemon",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/token_accounting.py",
        find='        with open_append(path, encoding="utf-8") as f:\n',
        replace='        with open(path, "a", encoding="utf-8") as f:\n',
        test="test_a_ledger_an_earlier_daemon_left_is_repaired_on_append",
        because="the ledger appended as the daemon, root-owned forever, the "
                "runner's rows refused with EACCES",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/process_posture.py",
        find="    runner_user.install_append_owner_resolver(policy)\n",
        replace="",
        test="test_the_daemon_installs_the_resolver_when_its_policy_drops",
        because="the resolver never installed, so every append is the daemon's",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_spawn.py",
        find="        hand_over_audit_files(\n",
        replace="        (lambda *a: None)(\n",
        test="test_the_spawn_hands_back_files_the_daemon_left",
        because="a ledger the daemon will not append to again this session "
                "left root-owned when the runner starts",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/owned_append.py",
        find="    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:\n",
        replace="    if not stat.S_ISREG(st.st_mode):\n",
        test="test_a_planted_hard_link_is_refused",
        because="a hard link planted in the workspace to a root-owned file "
                "elsewhere handed to the workspace owner",
    ),
]

OWNER: Tuple[int, int] = (65534, 65534)


@pytest.fixture
def owned_ws(tmp_path, monkeypatch) -> Tuple[str, Callable[[str], int]]:
    """``(workspace, uid_of)`` under the ``workspace-owner`` policy."""
    ws = tmp_path / "ws"
    (ws / ".jaato" / "logs").mkdir(parents=True)
    ledger: Dict[int, int] = {}

    def fake_fchown(fd, uid, gid):
        ledger[os.fstat(fd).st_ino] = uid

    def fake_lchown(path, uid, gid):
        ledger[os.lstat(path).st_ino] = uid

    monkeypatch.setattr(os, "fchown", fake_fchown)
    monkeypatch.setattr(os, "lchown", fake_lchown)
    monkeypatch.setattr(
        runner_user, "workspace_file_owner",
        lambda w: OWNER if w and os.path.realpath(w) == os.path.realpath(ws) else None)
    monkeypatch.setattr(runner_user, "_policy", runner_user.POLICY_WORKSPACE_OWNER)
    monkeypatch.setenv("JAATO_WORKSPACE_ROOT", str(ws))
    runner_user.install_append_owner_resolver(runner_user.POLICY_WORKSPACE_OWNER)
    yield str(ws), lambda p: ledger.get(os.lstat(p).st_ino, os.lstat(p).st_uid)
    owned_append.set_owner_resolver(None)


def test_a_trace_the_daemon_creates_is_the_owners(owned_ws):
    ws, uid_of = owned_ws
    trace_write("T", "hello", ".jaato/logs/session_trace.jsonl")
    path = os.path.join(ws, ".jaato", "logs", "session_trace.jsonl")
    assert open(path).read().endswith("hello\n")
    assert uid_of(path) == OWNER[0]


def test_a_ledger_an_earlier_daemon_left_is_repaired_on_append(owned_ws):
    ws, uid_of = owned_ws
    path = os.path.join(ws, ".jaato", "logs", "ledger.jsonl")
    with open(path, "w") as f:
        f.write('{"stage": "old"}\n')
    assert uid_of(path) == os.geteuid()
    led = TokenLedger(path=path)
    led._record("announcement", {"x": 1})
    led._append_to_disk()
    assert not led._append_failed
    assert uid_of(path) == OWNER[0]
    assert len(open(path).read().splitlines()) == 2


def test_a_chained_ledgers_lock_is_the_owners(owned_ws):
    ws, uid_of = owned_ws
    path = os.path.join(ws, ".jaato", "logs", "ledger.jsonl")
    led = TokenLedger(path=path, integrity="sha256-chain")
    led._record("announcement", {"x": 1})
    led._append_to_disk()
    assert uid_of(path) == OWNER[0]
    assert uid_of(path + ".lock") == OWNER[0]


def test_the_spawn_hands_back_files_the_daemon_left(owned_ws, monkeypatch):
    ws, uid_of = owned_ws
    logs = os.path.join(ws, ".jaato", "logs")
    for name in ("ledger.jsonl", "ledger.jsonl.lock", "session_trace.jsonl"):
        open(os.path.join(logs, name), "w").close()
    from jaato_server.server import runner_spawn

    class _Server:
        _session_env = {
            "LEDGER_PATH": ".jaato/logs/ledger.jsonl",
            "JAATO_TRACE_LOG": os.path.join(logs, "session_trace.jsonl"),
        }

    user = runner_user.user_for_uid(OWNER[0], OWNER[1], "test")
    monkeypatch.setattr(runner_spawn, "resolve_runner_user", lambda **k: user)
    monkeypatch.setattr(runner_spawn, "prepare_runner_owned_paths", lambda *a: None)
    runner_spawn._prepare_runner_user(
        _Server(), peer=None, session_id="s1", workspace_path=ws,
        profile_name=None, log_path=None, workspace_home=None)
    for name in ("ledger.jsonl", "ledger.jsonl.lock", "session_trace.jsonl"):
        assert uid_of(os.path.join(logs, name)) == OWNER[0], name


def test_the_daemon_installs_the_resolver_when_its_policy_drops(monkeypatch):
    from jaato_server.server import process_posture
    monkeypatch.setattr(process_posture, "announce_root_daemon_once", lambda p: None)
    monkeypatch.setattr(runner_user, "announce_if_unprivileged", lambda p: False)
    monkeypatch.setattr(runner_user, "_policy", runner_user.POLICY_DAEMON)
    try:
        process_posture.apply_process_posture(runner_uid_policy="workspace-owner")
        assert owned_append._resolver is runner_user.workspace_append_owner
        process_posture.apply_process_posture(runner_uid_policy="daemon")
        assert owned_append._resolver is None
    finally:
        owned_append.set_owner_resolver(None)


def test_the_daemon_policy_appends_as_before(owned_ws, monkeypatch):
    ws, uid_of = owned_ws
    monkeypatch.setattr(runner_user, "_policy", runner_user.POLICY_DAEMON)
    path = os.path.join(ws, ".jaato", "logs", "ledger.jsonl")
    led = TokenLedger(path=path)
    led._record("response", {})
    led._append_to_disk()
    assert uid_of(path) == os.geteuid()


def test_a_path_outside_the_workspace_is_never_handed(owned_ws, tmp_path):
    ws, uid_of = owned_ws
    outside = tmp_path / "shared" / "trace.log"
    trace_write("T", "x", str(outside))
    assert uid_of(str(outside)) == os.geteuid()


def test_a_planted_symlink_is_refused(owned_ws, tmp_path):
    ws, uid_of = owned_ws
    target = tmp_path / "elsewhere.txt"
    target.write_text("keep\n")
    path = os.path.join(ws, ".jaato", "logs", "ledger.jsonl")
    os.symlink(target, path)
    with pytest.raises(OSError):
        owned_append.open_append(path)
    assert target.read_text() == "keep\n"
    assert uid_of(str(target)) == os.geteuid()


def test_a_planted_hard_link_is_refused(owned_ws, tmp_path):
    ws, uid_of = owned_ws
    target = tmp_path / "elsewhere.txt"
    target.write_text("keep\n")
    path = os.path.join(ws, ".jaato", "logs", "ledger.jsonl")
    os.link(target, path)
    with pytest.raises(PermissionError):
        owned_append.open_append(path)
    assert not owned_append.hand_over_existing(path)
    assert uid_of(str(target)) == os.geteuid()
    assert target.read_text() == "keep\n"
