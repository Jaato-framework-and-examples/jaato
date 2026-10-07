"""An isolated sub-runner is handed its config; it never probes for it.

The SELinux phase 3 kernel run (a1d59347) got the sub-runner past the
spawn and then lost it in bootstrap:

* ``JaatoRuntime._load_base_system_instructions`` asked
  ``<ws>/.jaato/instructions`` ``is_dir()``.  The isolated boundary denies
  that directory down to ``getattr``, and ``Path.is_dir()`` re-raises
  ``EACCES``, so the bootstrap crashed.
* ``load_gc_from_file`` asked ``~/.jaato/gc.json`` ``exists()`` for the
  same reason, so every isolated session that got that far ran with no GC.
* the isolated envelope built ``gc`` from ``gc_obj.config``, an attribute
  ``GCProfileConfig`` does not have (#1133 on this path), so a profile's
  ``gc:`` block lost every number but ``type``.

The daemon, which can read all of it, now resolves the config tiers for
such a session and says so on the envelope
(``config_resolved_by_daemon``); the runner then reads neither tier.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.plugins.subagent.config import (
    GCProfileConfig,
    build_inline_profile,
)
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_RUNTIME = "jaato-server/jaato_server/shared/jaato_runtime.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"
_RUNNER = "jaato-server/jaato_server/server/runner/session.py"

REVERSIONS = [
    Reversion(
        target=_RUNTIME,
        find="        if not self._read_config_tiers:\n",
        replace="        if False:\n",
        test="test_a_runtime_told_not_to_read_the_tiers_never_probes_them",
        because="the isolated sub-runner's bootstrap would stat the denied "
                "instructions directory and crash",
    ),
    Reversion(
        target=_RUNNER,
        find="        read_config_tiers=not envelope.config_resolved_by_daemon,\n",
        replace="",
        test="test_the_runner_builds_its_runtime_from_the_envelope",
        because="the envelope's statement would never reach the runtime",
    ),
    Reversion(
        target=_RUNNER,
        find="        if not gc_result and envelope.config_resolved_by_daemon:\n",
        replace="        if False:\n",
        test="test_the_runner_never_reads_gc_json_for_an_isolated_session",
        because="the sub-runner would probe the denied gc.json and run "
                "without GC",
    ),
    Reversion(
        target=_SM,
        find="        return gc_obj.to_dict(), None\n",
        replace='        return {"type": gc_obj.type}, None\n',
        test="test_the_isolated_envelope_carries_the_whole_gc_block",
        because="an isolated subagent's gc: block would lose every number "
                "but its type (#1133 on this path)",
    ),
    Reversion(
        target=_SM,
        find="            config_resolved_by_daemon=True,\n",
        replace="",
        test="test_the_isolated_envelope_says_the_daemon_resolved_the_tiers",
        because="the sub-runner would read the tiers its boundary denies",
    ),
]


# ---------------------------------------------------------------- runtime


def _workspace_with_instructions(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / ".jaato" / "instructions").mkdir(parents=True)
    (ws / ".jaato" / "instructions" / "00-base.md").write_text("WORKSPACE TIER")
    return ws


def _runtime(ws: Path, read_config_tiers: bool):
    from jaato_server.shared.jaato_runtime import JaatoRuntime
    return JaatoRuntime(provider_name="echo", workspace_path=ws,
                        read_config_tiers=read_config_tiers)


def test_a_runtime_told_not_to_read_the_tiers_never_probes_them(tmp_path, monkeypatch):
    """The kernel's refusal, reproduced: probing the denied dir raises."""
    ws = _workspace_with_instructions(tmp_path)
    denied = str(ws / ".jaato" / "instructions")
    real = Path.is_dir

    def is_dir(self):
        if str(self) == denied:
            raise PermissionError(13, "Permission denied", denied)
        return real(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    base = _runtime(ws, read_config_tiers=False).get_base_system_instructions()
    assert "WORKSPACE TIER" not in (base or "")


def test_a_runtime_that_reads_the_tiers_still_reads_them(tmp_path):
    """The control: every other session keeps its workspace tier."""
    ws = _workspace_with_instructions(tmp_path)
    base = _runtime(ws, read_config_tiers=True).get_base_system_instructions()
    assert "WORKSPACE TIER" in base


def test_the_runner_builds_its_runtime_from_the_envelope():
    from jaato_server.server.runner.session import _default_runtime_factory

    envelope = SessionInitEnvelope(
        session_id="s1", workspace_path="/w", profile_name="",
        model_name="echo-1", provider_name="echo", plugins=[],
        config_resolved_by_daemon=True)
    assert _default_runtime_factory(envelope)._read_config_tiers is False


# ---------------------------------------------------------------- GC


class _FakeSession:
    def __init__(self):
        self._gc_plugin = None
        self._gc_config = None

    def set_gc_plugin(self, plugin, config=None):
        self._gc_plugin = plugin
        self._gc_config = config


def _envelope(**over):
    fields = dict(session_id="s1", workspace_path="/w", profile_name="",
                  model_name="echo-1", provider_name="echo", plugins=[],
                  config_resolved_by_daemon=True)
    fields.update(over)
    return SessionInitEnvelope(**fields)


def test_the_runner_never_reads_gc_json_for_an_isolated_session(monkeypatch):
    from jaato_server.server.runner import session as runner_session
    from jaato_server.shared.plugins import gc as gc_mod

    reads = []
    monkeypatch.setattr(gc_mod, "load_gc_from_file",
                        lambda **kw: reads.append(kw))
    sess = _FakeSession()
    runner_session._install_gc(sess, _envelope(gc_file=None))
    assert reads == []
    assert sess._gc_plugin is None


def test_the_runner_installs_the_gc_json_the_daemon_read():
    from jaato_server.server.runner.session import _install_gc

    sess = _FakeSession()
    _install_gc(sess, _envelope(gc_file={"type": "truncate",
                                         "threshold_percent": 55.0}))
    assert sess._gc_plugin is not None
    assert sess._gc_config.threshold_percent == 55.0


def test_the_envelope_round_trips_the_new_fields():
    env = _envelope(gc_file={"type": "budget"})
    back = SessionInitEnvelope.from_dict(env.to_dict())
    assert back.config_resolved_by_daemon is True
    assert back.gc_file == {"type": "budget"}
    assert SessionInitEnvelope.from_dict(
        _envelope(config_resolved_by_daemon=False).to_dict()
    ).config_resolved_by_daemon is False


# ---------------------------------------------------------------- daemon


def _payload(**over):
    payload = {"name": "researcher", "description": "d", "model": "m",
               "provider": "anthropic", "plugins": ["cli"], "plugin_configs": {}}
    payload.update(over)
    return payload


def _isolated_envelope(profile, workspace: str):
    return SessionManager._build_isolated_envelope(
        MagicMock(), profile=profile, isolated_session_id="iso-1",
        workspace_path=workspace, sub_apparmor_profile="", agent_params=None)


def test_the_isolated_envelope_carries_the_whole_gc_block(tmp_path):
    profile = build_inline_profile(_payload(), name="researcher", description="d")
    profile.gc = GCProfileConfig(type="budget", threshold_percent=55.0)
    env = _isolated_envelope(profile, str(tmp_path))
    assert env.gc["type"] == "budget" and env.gc["threshold_percent"] == 55.0
    assert env.gc_file is None


def test_the_isolated_envelope_ships_the_workspace_gc_json(tmp_path):
    (tmp_path / ".jaato").mkdir()
    (tmp_path / ".jaato" / "gc.json").write_text(json.dumps({"type": "budget"}))
    profile = build_inline_profile(_payload(), name="researcher", description="d")
    env = _isolated_envelope(profile, str(tmp_path))
    assert env.gc is None
    assert env.gc_file == {"type": "budget"}


def test_the_isolated_envelope_says_the_daemon_resolved_the_tiers(tmp_path):
    profile = build_inline_profile(_payload(), name="researcher", description="d")
    assert _isolated_envelope(profile, str(tmp_path)).config_resolved_by_daemon is True
