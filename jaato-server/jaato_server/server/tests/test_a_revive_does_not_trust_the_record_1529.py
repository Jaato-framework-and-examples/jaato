"""A revive does not trust a session record it cannot authenticate (#1529).

A session record lives in ``<ws>/.jaato/sessions/<id>.json``, which the
session's own code can rewrite: the workspace-wide ``rwkl`` grant covers it
in ``//child`` (``python -c``, a script, a notebook cell), and under a
dropping ``--runner-uid-policy`` it is the runner account's own file. A
revive (``session.wake``, a reattach after unload, a daemon restart)
rebuilt the session from it, taking from it the fields that decide the
revived boundary:

* ``sandbox_mode`` -- ``null`` disarmed the IPC kernel boundary;
* ``profile_snapshot`` -- plugins, plugin configs (the permission policy),
  ``runtime_limits`` (``seccomp: off``), AppArmor fragments,
  ``scrub_secret_env`` and ``env``, whose ``pass://`` URIs the daemon
  resolved and shipped to the runner;
* ``metadata.plugin_states.permission`` -- saved ``always`` rules;
* ``workspace_path``, ``config_root``, ``created_by``.

Two layers now:

1. **The daemon seals what it writes** (``server.record_seal``, HMAC under
   a key in its own ``~/.jaato``), and a record that does not verify is
   narrowed (``server.record_distrust``): the profile comes from disk, the
   permission rules keep only denials, the workspace is where the record
   was read from, membership comes from the daemon's index, and kernel
   confinement is ARMED rather than read back.
2. **``//child`` cannot write ``.jaato/sessions/``** (template v46), nor
   the isolated sub-runner a top-level record.

These tests write records to disk and drive the real
``SessionManager._load_session_impl`` with the real ``FileSessionPlugin``.
Only the server construction is stubbed, at the seam that receives the
envelope.  No enforcing kernel was involved.
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from jaato_server.server import record_seal
from jaato_server.server.apparmor import AppArmorManager
from jaato_server.server.record_distrust import narrow_permission_state
from jaato_server.server.session_manager import SessionManager
from jaato_server.server.session_workspace_index import SessionWorkspaceIndex
from jaato_server.shared.confinement_grants import (
    ConfinementGrants,
    profile_body_rules,
)
from jaato_server.shared.plugins.session.base import SessionState
from jaato_server.shared.plugins.subagent.config import profile_to_snapshot
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"
_SEAL = "jaato-server/jaato_server/server/record_seal.py"
_DISTRUST = "jaato-server/jaato_server/server/record_distrust.py"
_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find='        if getattr(state, "record_verified", None) is True:\n'
             '            return RecordTrust(trusted=True)\n',
        replace='        if True:\n'
                '            return RecordTrust(trusted=True)\n',
        because="every record is trusted again, so an edited snapshot "
                "decides the revived boundary",
        test="TestAnUnsealedRecord::test_the_profile_comes_from_disk",
    ),
    Reversion(
        target=_SEAL,
        find="    return hmac.compare_digest(mac, _mac(record, key))\n",
        replace="    return True\n",
        because="any record carrying a seal-shaped key verifies, so an edit "
                "after the daemon sealed it is trusted",
        test="TestASealedRecord::test_an_edit_after_sealing_is_not_trusted",
    ),
    Reversion(
        target=_SM,
        find="    if not trusted:\n        return True\n"
             "    return True if sandbox_mode_is_kernel",
        replace="    if False:\n        return True\n"
                "    return True if sandbox_mode_is_kernel",
        because="an edited sandbox_mode: null disarms kernel confinement on "
                "the revive",
        test="TestAnUnsealedRecord::test_confinement_is_armed",
    ),
    Reversion(
        target=_DISTRUST,
        find="    narrowed = narrow_permission_state(states.get(\"permission\"))\n",
        replace="    narrowed = states.get(\"permission\")\n",
        because="an always rule from an edited record reaches the enforcer",
        test="TestAnUnsealedRecord::test_the_enforcer_gets_only_denials",
    ),
    Reversion(
        target=_DISTRUST,
        find="    trusted = loaded_from or indexed\n",
        replace="    trusted = getattr(state, \"workspace_path\", None) or "
                "loaded_from or indexed\n",
        because="a record names another directory as its workspace and the "
                "runner is spawned (and confined) there",
        test="TestAnUnsealedRecord::test_the_workspace_is_where_it_was_read",
    ),
    Reversion(
        target=_APPARMOR,
        find='    audit deny "{workspace_path}/.jaato/sessions/" wlk,\n'
             '    audit deny "{workspace_path}/.jaato/sessions/**" wlk,\n',
        replace="",
        because="a subprocess the model drives can rewrite its own session "
                "record",
        test="TestTheKernel::test_child_cannot_write_a_record",
    ),
]

SID = "20261005_120000"

_PROFILE = """\
name: worker
description: a narrow worker
model: echo-model
provider: echo
plugins: [file_edit]
runtime_limits:
  seccomp: default
  pids_max: 64
plugin_configs:
  permission:
    policy:
      defaultPolicy: ask
env:
  SAFE_VAR: plain
"""


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with one narrow profile, and a private daemon HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    record_seal.reset_cache()
    root = tmp_path / "ws"
    (root / ".jaato" / "profiles").mkdir(parents=True)
    (root / ".jaato" / "profiles" / "worker.yaml").write_text(_PROFILE)
    yield root
    record_seal.reset_cache()


def _manager(tmp_path) -> SessionManager:
    sm = SessionManager()
    sm._session_workspace_index = SessionWorkspaceIndex(
        path=tmp_path / "index.json")
    return sm


def _widened_snapshot(sm: SessionManager, ws: pathlib.Path) -> Dict[str, Any]:
    profile, err = sm._resolve_profile(
        "worker", workspace_path=str(ws), config_root=None, env_file=None)
    assert profile is not None, err
    snap = profile_to_snapshot(profile)
    snap["plugins"] = ["file_edit", "cli", "notebook"]
    snap["runtime_limits"]["seccomp"] = "off"
    snap["runtime_limits"]["pids_max"] = None
    snap["plugin_configs"]["permission"] = {"policy": {"defaultPolicy": "allow"}}
    snap["env"]["STOLEN"] = "pass://victim/token"
    snap["scrub_secret_env"] = "none"
    return snap


def _widened_state(sm: SessionManager, ws: pathlib.Path) -> SessionState:
    now = datetime.now()
    return SessionState(
        session_id=SID,
        history=[],
        created_at=now,
        updated_at=now,
        profile_name="worker",
        profile_snapshot=_widened_snapshot(sm, ws),
        workspace_path="/etc",
        config_root=str(ws / "planted-config"),
        sandbox_mode=None,
        created_by="someone-else:victim",
        metadata={"plugin_states": {"permission": {
            "session_whitelist": ["*"],
            "session_blacklist": ["dangerous_tool"],
            "session_default_policy": "allow",
        }}},
    )


def _write(sm: SessionManager, ws: pathlib.Path, state: SessionState,
           seal: bool) -> pathlib.Path:
    storage = sm._session_storage_dir(str(ws))
    sm._write_record(state, storage, None, seal=seal)
    return storage / f"{SID}.json"


def _revive(sm: SessionManager, ws: pathlib.Path,
            server: Any = None) -> Tuple[Any, List[Any]]:
    """Drive ``_load_session_impl``; return ``(session, envelopes)``.

    ``server=None`` stops the revive at construction (the envelope is all
    that is wanted); otherwise construction returns *server* and the
    revive runs to the end.
    """
    envelopes: List[Any] = []

    def _construct(_self, envelope):
        envelopes.append(envelope)
        if server is None:
            return None, None
        return server, None

    with patch.object(SessionManager, "_construct_and_initialize_server",
                      _construct), \
            patch.object(SessionManager, "_start_workspace_monitor",
                         lambda *a, **k: None):
        session = sm._load_session_impl(SID, None, str(ws))
    return session, envelopes


def _runner_server() -> MagicMock:
    server = MagicMock()
    server.registry = None
    server._runtime = None
    return server


def _restored_permission(server: MagicMock) -> Optional[Dict[str, Any]]:
    restorer = server._runner_rpc.session_restore_permission_persistence_threadsafe
    if not restorer.call_args_list:
        return None
    return restorer.call_args_list[-1].args[0]


# ----------------------------------------------------------------------
# Layer 1: an unsealed record is narrowed
# ----------------------------------------------------------------------


class TestAnUnsealedRecord:
    def test_the_profile_comes_from_disk(self, ws, tmp_path):
        sm = _manager(tmp_path)
        _write(sm, ws, _widened_state(sm, ws), seal=False)
        _, envelopes = _revive(sm, ws)
        profile = envelopes[0].profile
        assert profile.plugins == ["file_edit"]
        assert profile.runtime_limits.seccomp == "default"
        assert profile.runtime_limits.pids_max == 64
        assert profile.plugin_configs["permission"]["policy"]["defaultPolicy"] == "ask"
        assert "STOLEN" not in (profile.env or {})
        assert profile.scrub_secret_env in (None, "default")

    def test_confinement_is_armed(self, ws, tmp_path):
        sm = _manager(tmp_path)
        _write(sm, ws, _widened_state(sm, ws), seal=False)
        _, envelopes = _revive(sm, ws)
        assert envelopes[0].apparmor is True
        assert envelopes[0].sandbox_mode is None

    def test_the_workspace_is_where_it_was_read(self, ws, tmp_path):
        sm = _manager(tmp_path)
        _write(sm, ws, _widened_state(sm, ws), seal=False)
        _, envelopes = _revive(sm, ws)
        assert envelopes[0].workspace_path == str(ws)
        # The planted config root is not used either.
        assert envelopes[0].config_root == str(ws / ".jaato")

    def test_membership_comes_from_the_daemons_index(self, ws, tmp_path):
        sm = _manager(tmp_path)
        sm._session_workspace_index.record_membership(
            SID, created_by="app:owner", cascade_driver_id=None,
            sibling_name=None)
        _write(sm, ws, _widened_state(sm, ws), seal=False)
        _, envelopes = _revive(sm, ws)
        assert envelopes[0].created_by == "app:owner"

    def test_the_enforcer_gets_only_denials(self, ws, tmp_path):
        sm = _manager(tmp_path)
        _write(sm, ws, _widened_state(sm, ws), seal=False)
        server = _runner_server()
        session, _ = _revive(sm, ws, server)
        assert session is not None
        assert _restored_permission(server) == {
            "session_blacklist": ["dangerous_tool"]}

    def test_its_saves_stay_unsealed_until_the_boundary_is_redecided(
            self, ws, tmp_path):
        """A clientless revive provisions nothing; sealing it would launder
        the edited ``sandbox_mode`` into a value the next revive trusts."""
        sm = _manager(tmp_path)
        _write(sm, ws, _widened_state(sm, ws), seal=False)
        session, _ = _revive(sm, ws, _runner_server())
        assert session.record_trusted is False

    def test_an_inline_profile_is_refused(self, ws, tmp_path):
        sm = _manager(tmp_path)
        state = _widened_state(sm, ws)
        state.profile_spec = {"model": "m", "provider": "echo"}
        _write(sm, ws, state, seal=False)
        session, envelopes = _revive(sm, ws)
        assert session is None and envelopes == []

    def test_a_profile_that_no_longer_resolves_is_refused(self, ws, tmp_path):
        sm = _manager(tmp_path)
        _write(sm, ws, _widened_state(sm, ws), seal=False)
        (ws / ".jaato" / "profiles" / "worker.yaml").unlink()
        session, envelopes = _revive(sm, ws, _runner_server())
        assert session is None
        assert envelopes == []


# ----------------------------------------------------------------------
# A sealed record is the daemon's own
# ----------------------------------------------------------------------


class TestASealedRecord:
    def test_a_sealed_record_is_restored_as_written(self, ws, tmp_path):
        """#787 holds for a record the daemon wrote: the frozen recipe."""
        sm = _manager(tmp_path)
        _write(sm, ws, _widened_state(sm, ws), seal=True)
        _, envelopes = _revive(sm, ws)
        assert envelopes[0].profile.runtime_limits.seccomp == "off"
        # A sealed ``null`` means the daemon decided "unconfined": no
        # override, so the fresh opt-in decides.
        assert envelopes[0].apparmor is None

    def test_an_edit_after_sealing_is_not_trusted(self, ws, tmp_path):
        sm = _manager(tmp_path)
        path = _write(sm, ws, _widened_state(sm, ws), seal=True)
        data = json.loads(path.read_text())
        data["sandbox_mode"] = None
        data["profile_snapshot"]["runtime_limits"]["seccomp"] = "off"
        data["profile_snapshot"]["runtime_limits"]["pids_max"] = 1
        path.write_text(json.dumps(data))
        _, envelopes = _revive(sm, ws)
        assert envelopes[0].profile.runtime_limits.pids_max == 64
        assert envelopes[0].apparmor is True

    def test_the_key_is_private_to_the_daemon(self, ws):
        record_seal.load_key()
        key = record_seal.default_key_path()
        assert key.is_file()
        assert (key.stat().st_mode & 0o777) == 0o600


def test_narrowing_keeps_only_denials():
    assert narrow_permission_state({
        "session_whitelist": ["a"], "session_blacklist": ["b", "b"],
        "session_default_policy": "allow"}) == {"session_blacklist": ["b"]}
    assert narrow_permission_state({"session_default_policy": "deny"}) == {
        "session_default_policy": "deny"}
    assert narrow_permission_state({"session_whitelist": ["a"]}) is None
    assert narrow_permission_state("junk") is None


# ----------------------------------------------------------------------
# Layer 2: the kernel
# ----------------------------------------------------------------------

WS = "/workspace"
RECORD = f"{WS}/.jaato/sessions/{SID}.json"


def _rendered(tmp_path) -> Tuple[str, str]:
    (tmp_path / "w" / "sessions").mkdir(parents=True)
    (tmp_path / "p").mkdir()
    manager = AppArmorManager(workspace_root=str(tmp_path / "w"),
                              venv_path="/usr/local/venv",
                              profile_dir=str(tmp_path / "p"))
    sub = manager._render_sub_profile(
        parent_session_id="parent-A", subagent_id="agent-B",
        workspace_path=WS)
    return manager._render_profile("s1", WS), sub


def _grants(rules) -> ConfinementGrants:
    return ConfinementGrants(profile_name="x", exec_scope=None, rules=list(rules))


def _base_rules(profile: str):
    head = profile.split("  profile tool_hat", 1)[0]
    return [line.strip() for line in head.splitlines()
            if line.strip().startswith(('"/', "/", "audit deny", "owner", "deny"))]


def _flat_rules(body: str):
    return [line.strip() for line in body.splitlines()
            if line.strip().startswith(('"/', "/", "audit deny", "owner", "deny"))]


class TestTheKernel:
    def test_child_cannot_write_a_record(self, tmp_path):
        profile, _ = _rendered(tmp_path)
        child = _grants(profile_body_rules(profile, "child"))
        assert child.verdict(RECORD, "w") is False
        assert child.verdict(f"{WS}/.jaato/sessions/{SID}/subagents/a.json",
                             "w") is False
        assert child.verdict(f"{WS}/.jaato/sessions/", "w") is False

    def test_child_still_writes_the_workspace(self, tmp_path):
        profile, _ = _rendered(tmp_path)
        child = _grants(profile_body_rules(profile, "child"))
        assert child.verdict(f"{WS}/src/a.py", "w") is True

    def test_the_runner_still_writes_session_state(self, tmp_path):
        profile, _ = _rendered(tmp_path)
        base = _grants(_base_rules(profile))
        assert base.verdict(f"{WS}/.jaato/sessions/{SID}/plans/p.yaml",
                            "w") is True

    def test_the_isolated_sub_runner_cannot_write_a_record(self, tmp_path):
        _, sub = _rendered(tmp_path)
        assert _grants(_flat_rules(sub)).verdict(RECORD, "w") is False
