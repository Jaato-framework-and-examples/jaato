"""A qualified profile binds that file, at creation and on every revive (#1588).

A session created with ``profile="openrouter/telegram_chat"`` -- a set
leaf at ``profiles/openrouter/telegram_chat.yaml`` -- was bound, and
snapshotted for every later revive, to a DIFFERENT file that declares the
same ``name:``: a flat ``profiles/telegram_chat.yaml``.

Creation resolved the qualified path correctly.  What the record kept was
``profile_name = "telegram_chat"``, the resolved profile's ``name:``, and
every revive that re-resolves the profile from disk resolved that bare
name, which the flat file wins whenever ``JAATO_PROFILE_SET`` is not the
set.  The reported record was written by a daemon predating #1529's seal
and revived by one with it, so the revive treated it as unsealed, dropped
its snapshot, re-resolved ``telegram_chat`` -- and the next save
snapshotted the flat profile (the field is write-once, and #1529 had just
emptied it), so every later revive restored the wrong profile.

Now:

* the record carries ``profile_ref`` (record 2.12): the profile as the
  session requested it, and a revive resolves it before the bare name;
* a qualified ``<set>/<name>`` binds only a file read from that set's
  directory, so a set file that fails to load is a refusal, not a silent
  substitute from ``profiles/``;
* two files declaring one ``name:`` in one directory are announced at
  discovery, and ``jaato-scaffold validate`` reports both shapes.

These tests write profiles to disk and drive the real
``SessionManager._create_session_impl`` / ``_save_session`` /
``_load_session_impl`` with the real ``FileSessionPlugin``; only server
construction is stubbed, at the seam that receives the envelope.
"""

from __future__ import annotations

import logging
import pathlib
from datetime import datetime
from types import SimpleNamespace
from typing import Any, List, Tuple
from unittest.mock import patch

import pytest

from jaato_server.server import record_seal
from jaato_server.server.session_manager import Session, SessionManager
from jaato_server.server.session_workspace_index import SessionWorkspaceIndex
from jaato_server.shared.plugins.session.base import SessionState
from jaato_server.shared.session_context import (
    reset_config_root,
    reset_workspace_root,
    set_config_root,
    set_workspace_root,
)
from jaato_server.shared.plugins.subagent import config as subagent_config
from jaato_server.shared.plugins.subagent.config import (
    discover_profiles,
    profile_to_snapshot,
)
from jaato_server.shared.scaffold.validate import validate_workspace
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"
_SER = "jaato-server/jaato_server/shared/plugins/session/serializer.py"
_CFG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"
_VAL = "jaato-server/jaato_server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find="            profile_ref=profile_ref,\n",
        replace="",
        because="the created session records no ref, so a revive resolves "
                "the bare name and binds the flat file",
        test="TestCreate::test_the_created_session_remembers_the_ref",
    ),
    Reversion(
        target=_SM,
        find="                    profile_ref=session.profile_ref,  # 2.12+ (#1588)\n",
        replace="",
        because="the save drops the ref, so the record names only the "
                "bare name again",
        test="TestSave::test_the_record_carries_the_ref_and_the_leaf",
    ),
    Reversion(
        target=_SM,
        find='        requested = getattr(state, "profile_ref", None) or state.profile_name\n',
        replace="        requested = state.profile_name\n",
        because="a revive re-resolves the bare name, which the flat file wins",
        test="TestRevive::test_a_sealed_record_without_a_snapshot",
    ),
    Reversion(
        target=_SER,
        find="        profile_ref=data.get('profile_ref'),  # None on pre-2.12 records\n",
        replace="",
        because="the ref is written and never read back, so an unsealed "
                "revive (#1529) binds the flat file -- the reported case",
        test="TestRevive::test_an_unsealed_record_binds_the_leaf",
    ),
    Reversion(
        target=_SM,
        find="    in_set = (\n        force_profile_set is None\n",
        replace="    in_set = True or (\n        force_profile_set is None\n",
        because="a set file that does not load is silently replaced by the "
                "flat file of the same name",
        test="TestResolve::test_a_set_file_that_does_not_load_is_refused",
    ),
    Reversion(
        target=_VAL,
        find="    _check_profile_name_collisions(config_root, out)\n",
        replace="",
        because="validate says nothing about two files declaring one name",
        test="TestValidate::test_a_set_file_and_a_flat_file",
    ),
    Reversion(
        target=_CFG,
        find="    if winner and Path(winner).parent == file_path.parent and key not in _WARNED_DUPLICATES:\n",
        replace="    if False:\n",
        because="two files in one directory declaring one name are resolved "
                "by listing order, silently",
        test="TestDiscovery::test_a_same_directory_duplicate_is_announced",
    ),
]

SID = "20261005_160350"

_BASE = """\
name: _base_tc
description: shared base
model: echo-model
provider: echo
plugins: [cli]
"""
_LEAF = """\
name: tc
inherits: [_base_tc]
description: LEAF voice-first
plugins: [cli, todo]
"""
_FLAT = """\
name: tc
description: FLAT old profile
model: echo-model
provider: echo
plugins: [file_edit]
"""


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with the reported layout, and a private daemon HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # Discovery falls back to these when no config root is passed; a test
    # earlier in the run may have left either set.
    for var in ("JAATO_PROFILE_SET", "JAATO_CONFIG_ROOT", "JAATO_WORKSPACE_ROOT"):
        monkeypatch.delenv(var, raising=False)
    config_token = set_config_root(None)
    workspace_token = set_workspace_root(None)
    record_seal.reset_cache()
    root = tmp_path / "ws"
    profiles = root / ".jaato" / "profiles"
    (profiles / "openrouter").mkdir(parents=True)
    (profiles / "_base_tc.yaml").write_text(_BASE)
    (profiles / "openrouter" / "tc.yaml").write_text(_LEAF)
    (profiles / "tc.yaml").write_text(_FLAT)
    yield root
    record_seal.reset_cache()
    reset_config_root(config_token)
    reset_workspace_root(workspace_token)


def _manager(tmp_path) -> SessionManager:
    sm = SessionManager()
    sm._session_workspace_index = SessionWorkspaceIndex(
        path=tmp_path / "index.json")
    return sm


def _capturing_construct(envelopes: List[Any]):
    def _construct(_self, envelope):
        envelopes.append(envelope)
        return None, None
    return _construct


def _revive(sm: SessionManager, ws: pathlib.Path) -> Any:
    envelopes: List[Any] = []
    with patch.object(SessionManager, "_construct_and_initialize_server",
                      _capturing_construct(envelopes)), \
            patch.object(SessionManager, "_start_workspace_monitor",
                         lambda *a, **k: None):
        sm._load_session_impl(SID, None, str(ws))
    assert envelopes, "the revive never reached server construction"
    return envelopes[0].profile


def _record(sm: SessionManager, ws: pathlib.Path, *, seal: bool,
            ref: Any = "openrouter/tc", snapshot: Any = None) -> None:
    now = datetime.now()
    state = SessionState(
        session_id=SID, history=[], created_at=now, updated_at=now,
        profile_name="tc", profile_ref=ref, profile_snapshot=snapshot,
        workspace_path=str(ws),
    )
    sm._write_record(state, sm._session_storage_dir(str(ws)), None, seal=seal)


# ----------------------------------------------------------------------


class TestResolve:
    def test_the_qualified_request_binds_the_leaf(self, ws, tmp_path):
        profile, err = _manager(tmp_path)._resolve_profile(
            "openrouter/tc", str(ws))
        assert err is None
        assert profile.description == "LEAF voice-first"

    def test_a_bare_name_binds_the_flat_file_with_no_set(self, ws, tmp_path):
        # The control: nothing selects the set, so the flat file is right.
        profile, _ = _manager(tmp_path)._resolve_profile("tc", str(ws))
        assert profile.description == "FLAT old profile"

    def test_a_selected_set_still_wins_a_bare_name(self, ws, tmp_path):
        # The profile-set design is unchanged.
        (ws / ".env").write_text("JAATO_PROFILE_SET=openrouter\n")
        profile, _ = _manager(tmp_path)._resolve_profile(
            "tc", str(ws), env_file=str(ws / ".env"))
        assert profile.description == "LEAF voice-first"

    def test_a_set_file_that_does_not_load_is_refused(self, ws, tmp_path):
        leaf = ws / ".jaato" / "profiles" / "openrouter" / "tc.yaml"
        leaf.write_text("name: tc\ndescription: no plugins key\n")
        profile, err = _manager(tmp_path)._resolve_profile(
            "openrouter/tc", str(ws))
        assert profile is None
        assert "openrouter" in err

    def test_a_dot_dot_set_is_not_scanned(self, ws, tmp_path):
        profile, err = _manager(tmp_path)._resolve_profile("../tc", str(ws))
        assert profile is None
        assert "not a profile set" in err


class TestCreate:
    def _create(self, sm, ws) -> Any:
        envelopes: List[Any] = []
        with patch.object(SessionManager, "_construct_and_initialize_server",
                          _capturing_construct(envelopes)):
            sm._create_session_impl(
                "client_1", workspace_path=str(ws),
                profile_name="openrouter/tc")
        assert envelopes, "creation never reached server construction"
        return envelopes[0]

    def test_the_created_session_binds_the_leaf(self, ws, tmp_path):
        envelope = self._create(_manager(tmp_path), ws)
        assert envelope.profile.description == "LEAF voice-first"

    def test_the_created_session_remembers_the_ref(self, ws, tmp_path):
        envelope = self._create(_manager(tmp_path), ws)
        assert envelope.profile_ref == "openrouter/tc"


class TestSave:
    def test_the_record_carries_the_ref_and_the_leaf(self, ws, tmp_path):
        sm = _manager(tmp_path)
        leaf, _ = sm._resolve_profile("openrouter/tc", str(ws))
        server = SimpleNamespace(
            _profile=leaf, _runner_rpc=None, _runner_released=False,
            registry=None, main_agent_id="main", _agents={},
            _main_agent_display_name=None, _agent_params=None,
            _effective_budget_control=None, seccomp_posture=None,
        )
        session = Session(
            session_id=SID, name="s", server=server,
            created_at=datetime.now().isoformat(), workspace_path=str(ws),
            profile_ref="openrouter/tc",
        )
        assert sm._save_session(session)
        state = sm._read_record(SID, sm._session_storage_dir(str(ws)))
        assert state.record_verified is True
        assert state.profile_ref == "openrouter/tc"
        assert state.profile_snapshot["description"] == "LEAF voice-first"


class TestRevive:
    def test_a_sealed_record_without_a_snapshot(self, ws, tmp_path):
        sm = _manager(tmp_path)
        _record(sm, ws, seal=True)
        assert _revive(sm, ws).description == "LEAF voice-first"

    def test_an_unsealed_record_binds_the_leaf(self, ws, tmp_path):
        # The reported case: a record #1529 cannot vouch for has its
        # snapshot dropped and its profile re-derived from disk.
        sm = _manager(tmp_path)
        leaf, _ = sm._resolve_profile("openrouter/tc", str(ws))
        _record(sm, ws, seal=False, snapshot=profile_to_snapshot(leaf))
        assert _revive(sm, ws).description == "LEAF voice-first"

    def test_a_sealed_snapshot_is_restored(self, ws, tmp_path):
        sm = _manager(tmp_path)
        leaf, _ = sm._resolve_profile("openrouter/tc", str(ws))
        _record(sm, ws, seal=True, snapshot=profile_to_snapshot(leaf))
        assert _revive(sm, ws).description == "LEAF voice-first"

    def test_a_pre_2_12_record_still_revives_by_name(self, ws, tmp_path):
        # Not migrated: a record with no ref re-resolves its bare name, as
        # it always did.  Stated, not fixed.
        sm = _manager(tmp_path)
        _record(sm, ws, seal=True, ref=None)
        assert _revive(sm, ws).description == "FLAT old profile"


class TestDiscovery:
    def test_a_same_directory_duplicate_is_announced(self, ws, caplog):
        subagent_config._WARNED_DUPLICATES.clear()
        profiles = ws / ".jaato" / "profiles"
        (profiles / "zz_tc_copy.yaml").write_text(_FLAT)
        with caplog.at_level(logging.WARNING):
            result = discover_profiles(".jaato/profiles", base_path=str(ws),
                                       session_env={})
        assert result.sources["tc"].endswith("tc.yaml")
        assert any("zz_tc_copy.yaml" in r.getMessage() for r in caplog.records)

    def test_a_set_override_is_recorded_not_warned(self, ws, caplog):
        subagent_config._WARNED_DUPLICATES.clear()
        with caplog.at_level(logging.WARNING):
            result = discover_profiles(
                ".jaato/profiles", base_path=str(ws),
                force_profile_set="openrouter", session_env={})
        assert result.sources["tc"].endswith("openrouter/tc.yaml")
        assert result.collisions["tc"][0].endswith("profiles/tc.yaml")
        assert not [r for r in caplog.records
                    if "declare name" in r.getMessage()]


class TestValidate:
    def _codes(self, ws) -> List[Tuple[str, str]]:
        return [(d.severity, d.code) for d in validate_workspace(str(ws))
                if d.code.startswith("profile_name_")]

    def test_a_set_file_and_a_flat_file(self, ws):
        assert ("warn", "profile_name_collision") in self._codes(ws)

    def test_two_sets_defining_one_agent_are_not_reported(self, ws):
        (ws / ".jaato" / "profiles" / "tc.yaml").unlink()
        other = ws / ".jaato" / "profiles" / "dumb"
        other.mkdir()
        (other / "tc.yaml").write_text(_LEAF)
        assert self._codes(ws) == []

    def test_two_files_in_one_directory_are_an_error(self, ws):
        (ws / ".jaato" / "profiles" / "zz_tc_copy.yaml").write_text(_FLAT)
        assert ("error", "profile_name_duplicate") in self._codes(ws)
