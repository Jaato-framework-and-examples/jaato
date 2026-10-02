"""An isolated sub-runner runs as its parent's runner user (#1168).

The SELinux phase 3 kernel run (573f1470), ``--runner-uid-policy
workspace-owner``: the parent runner dropped to the workspace owner, and
the isolated sub-runner it spawned ran as ROOT. Its spawn line had no
``runs_as``, its log no ``running as``, and its write into the owner's
workspace was refused for want of ``dac_override``, which the isolated
domains rightly withhold. The isolated spawn path never applied the uid
policy at all.

It inherits the parent's resolved user rather than re-resolving the
policy: under ``peer`` a sub-runner has no IPC peer to resolve from.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_server.server.confinement.base import ConfinementHandle
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.plugins.subagent.config import build_inline_profile
from jaato_server.shared.privilege_drop import RunnerUser
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find="        return stashed_runner_user(server) if server is not None else None\n",
        replace="        return None\n",
        test="test_the_parent_user_is_the_parents_stashed_user",
        because="the sub-runner would run as root under a parent dropped "
                "to the workspace owner",
    ),
    Reversion(
        target=_SM,
        find="                runner_user=self._parent_runner_user(parent_session_id),\n",
        replace="",
        test="test_the_spawn_path_passes_the_parent_user",
        because="the parent's user would never reach the sub-runner's spawn",
    ),
    Reversion(
        target=_SM,
        find="            confinement=confinement,\n            runner_user=runner_user,\n        )\n\n        rpc",
        replace="            confinement=confinement,\n        )\n\n        rpc",
        test="test_the_sub_runner_is_spawned_as_that_user_after_the_hand_over",
        because="the spawner would not drop the sub-runner's privileges",
    ),
    Reversion(
        target=_SM,
        find="            user_tier_files=user_tier_snapshot(runner_user),\n",
        replace="            user_tier_files=user_tier_snapshot(None),\n",
        test="test_the_envelope_carries_the_user_and_its_user_tier",
        because="a dropped sub-runner would be handed the daemon's user tier",
    ),
]

_USER = RunnerUser(uid=1000, gid=1000, groups=(1000,), username="apanoia",
                   home="/home/apanoia", source="workspace-owner")


def _manager(backend=None):
    import threading
    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._isolated_sub_runners = {}
    sm._sessions = {}
    sm._selinux_backend = backend
    sm._apparmor_manager = MagicMock()
    return sm


def _with_parent(sm, user):
    sm._sessions["sess-A"] = SimpleNamespace(server=SimpleNamespace(_runner_user=user))
    return sm


def _handle():
    label = "u:r:jaato_isolated_t:s0:c1,c2"
    return ConfinementHandle(backend="selinux", label=label,
                             confinement_id="cid", child_label=label)


def test_the_parent_user_is_the_parents_stashed_user():
    sm = _with_parent(_manager(), _USER)
    assert sm._parent_runner_user("sess-A") is _USER
    assert _manager()._parent_runner_user("sess-A") is None


def _payload():
    return {"name": "researcher", "description": "d", "model": "m",
            "provider": "anthropic", "plugins": ["cli"], "plugin_configs": {}}


def test_the_spawn_path_passes_the_parent_user():
    backend = MagicMock()
    backend.provision_isolated.return_value = _handle()
    sm = _with_parent(_manager(backend), _USER)
    seen = {}

    def spawn(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the spawn arguments")

    sm._do_spawn_isolated_runner = spawn
    sm._spawn_isolated_runner(
        parent_session_id="sess-A", subagent_id="agent-1",
        profile_payload=_payload(), task="t", workspace_path="/work")
    assert seen["runner_user"] is _USER


def test_the_sub_runner_is_spawned_as_that_user_after_the_hand_over(monkeypatch):
    from jaato_server.server import runner_spawner, runner_user

    order = []
    monkeypatch.setattr(runner_user, "prepare_runner_owned_paths",
                        lambda user, dirs, files: order.append(("hand_over", user, list(files))))

    def spawn(self, **kwargs):
        order.append(("spawn", kwargs.get("runner_user")))
        raise RuntimeError("stop after the spawn")

    monkeypatch.setattr(runner_spawner.RunnerSpawner, "spawn", spawn)
    sm = _manager(MagicMock())
    sm._daemon_loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="stop after the spawn"):
            sm._do_spawn_isolated_runner(
                parent_session_id="sess-A", subagent_id="a1",
                isolated_session_id="sess-A__sub_a1", workspace_path="/work",
                sub_apparmor_profile="", cgroup_path="", profile=MagicMock(),
                confinement=_handle(), effective_runtime_limits=MagicMock(),
                agent_params=None, runner_user=_USER)
    finally:
        sm._daemon_loop.close()
    log = "/work/.jaato/logs/runner-sess-A__sub_a1.log"
    assert order[0][0] == "hand_over" and order[0][1] is _USER and log in order[0][2]
    assert order[1] == ("spawn", _USER)


def test_the_envelope_carries_the_user_and_its_user_tier(monkeypatch, tmp_path):
    from jaato_server.server import runner_spawn

    asked = []
    monkeypatch.setattr(runner_spawn, "user_tier_snapshot",
                        lambda user: asked.append(user) or {})
    profile = build_inline_profile(_payload(), name="researcher", description="d")
    env = SessionManager._build_isolated_envelope(
        MagicMock(), profile=profile, isolated_session_id="iso-1",
        workspace_path=str(tmp_path), sub_apparmor_profile="",
        agent_params=None, runner_user=_USER)
    assert env.runner_user == _USER.to_dict()
    assert asked == [_USER]
