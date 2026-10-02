"""Phase 3: an isolated sub-runner under SELinux (selinux-backend.md §5.3).

The AppArmor isolated sub-runner works in its PARENT's workspace and is
isolated by what its flat sub-profile denies: reads on the persona/config
entries and ``prompts/``, every exec, the user tier.  Under SELinux the
same boundary is a domain (``jaato_isolated_t``, or ``jaato_isolated_ro_t``
for the read-only tightening) at the parent's level, with the
read-denied entries labelled ``jaato_agent_config_t`` / ``jaato_prompts_t``
so the domain can be refused them.  The subpath tightening has no SELinux
form and is refused by name.

What runs where it is checked: the policy rules in the ``selinux-policy``
CI job (``jaato-server/selinux/tests``), the domain on a kernel by the
phase 3 handoff.  Here: provisioning, the daemon's spawn path, the
envelope, and the agreement between the two LSMs' read-deny lists.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jaato_server.server.confinement import Boundary, selinux_labels
from jaato_server.server.confinement.base import ConfinementHandle
from jaato_server.server.session_manager import SessionManager, SubRunnerHandle
from jaato_server.server.tests.test_selinux_provision_2b import (
    _Kernel,
    _backend,
    _level,
    _type,
    _workspace,
)
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"
_SEL = "jaato-server/jaato_server/server/confinement/selinux.py"
_LABELS = "jaato-server/jaato_server/server/confinement/selinux_labels.py"
_SUBAGENT = "jaato-server/jaato_server/shared/plugins/subagent/plugin.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find='        if tightenings.get("isolated_workspace_subpath"):\n',
        replace='        if False:\n',
        test="test_a_subpath_tightening_is_refused",
        because="an isolated sub-runner asked to stay in a subpath would "
                "run over the whole workspace",
    ),
    Reversion(
        target=_SEL,
        find="        domain = ISOLATED_RO_DOMAIN if read_only else ISOLATED_DOMAIN\n",
        replace="        domain = ISOLATED_DOMAIN\n",
        test="test_read_only_picks_the_read_only_domain",
        because="isolated_read_only_workspace would run read-write",
    ),
    Reversion(
        target=_SEL,
        find="        return self._provision(session_id, boundary, domain, domain)\n",
        replace="        return self._provision(session_id, boundary, domain, CHILD_DOMAIN)\n",
        test="test_the_isolated_domain_is_flat",
        because="a sub-runner's subprocesses would leave its domain for "
                "jaato_child_t, which reads the persona config",
    ),
    Reversion(
        target=_SM,
        find="    if confinement is not None:\n        return selinux_descriptor(confinement)\n",
        replace="",
        test="test_the_envelope_names_the_isolated_domain",
        because="the sub-runner would be told no boundary and refuse, or "
                "run unconfirmed",
    ),
    Reversion(
        target=_SM,
        find="        if not sub_profile_name:\n            return\n",
        replace="",
        test="test_a_selinux_rollback_does_not_touch_apparmor",
        because="an SELinux spawn failure would try to unload an AppArmor "
                "sub-profile nobody loaded",
    ),
    Reversion(
        target=_LABELS,
        find="            if parts[1] == PROMPTS_DIR:\n                return PROMPTS_TYPE\n",
        replace="",
        test="test_prompts_get_their_own_type",
        because="an isolated sub-runner could read the prompt library, "
                "which AppArmor denies it",
    ),
    Reversion(
        target=_SUBAGENT,
        find='getattr(self._parent_session, "_daemon_session_id", None) or ""',
        replace='getattr(self._parent_session, "_session_id", None) or ""',
        test="test_an_isolated_spawn_names_its_parent_session",
        because="every isolated spawn would send an empty parent id and be "
                "refused by the daemon before any boundary is provisioned",
    ),
    Reversion(
        target=_SEL,
        find="            selinux_labels.RUNNER_LOG_TYPE, ctx.level))\n",
        replace="            selinux_labels.WORKSPACE_TYPE, ctx.level))\n",
        test="test_the_sub_runner_log_is_labelled_for_append",
        because="a read-only sub-runner could not append to its own log and "
                "would run with none (phase 3 kernel run)",
    ),
    Reversion(
        target=_SM,
        find="                self._selinux_backend.prepare_runner_log(confinement, log_path)\n",
        replace="                pass\n",
        test="test_the_daemon_labels_the_log_before_the_spawn",
        because="the log would keep the workspace type the read-only domain "
                "may not write",
    ),
]

_OWN_ROLE = "unconfined_u:unconfined_r"


@pytest.fixture(autouse=True)
def _session_tmpdirs_under_tmp_path(tmp_path, monkeypatch):
    from jaato_server.server.confinement import selinux

    monkeypatch.setattr(
        selinux, "session_tmpdir",
        lambda sid, cid=None: str(tmp_path / "systmp" / f"jaato-{cid}" / sid))


# ---------------------------------------------------------------- backend


def test_the_isolated_domain_runs_at_its_parents_level(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    backend = _backend(kernel, tmp_path)
    parent = backend.provision("s1", Boundary(workspace_path=str(ws)))
    child = backend.provision_isolated(
        "s1__sub_a", Boundary(workspace_path=str(ws)), read_only=False)
    assert _type(child.label) == "jaato_isolated_t"
    assert _level(child.label) == _level(parent.label)
    assert child.label.startswith(_OWN_ROLE)
    assert child.confinement_id != parent.confinement_id


def test_the_isolated_domain_is_flat(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    handle = _backend(kernel, tmp_path).provision_isolated(
        "s1__sub_a", Boundary(workspace_path=str(ws)), read_only=False)
    assert handle.child_label == handle.label


def test_read_only_picks_the_read_only_domain(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    handle = _backend(kernel, tmp_path).provision_isolated(
        "s1__sub_a", Boundary(workspace_path=str(ws)), read_only=True)
    assert _type(handle.label) == "jaato_isolated_ro_t"
    assert handle.grants["domain"] == "jaato_isolated_ro_t"


def test_prompts_get_their_own_type(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    (ws / ".jaato" / "prompts").mkdir()
    (ws / ".jaato" / "prompts" / "p.md").write_text("x")
    _backend(kernel, tmp_path).provision("s1", Boundary(workspace_path=str(ws)))
    assert _type(kernel.labels[str(ws / ".jaato" / "prompts" / "p.md")]) == "jaato_prompts_t"


def _apparmor_read_denied():
    """The ``.jaato`` entries the AppArmor isolated sub-profile read-denies."""
    from jaato_server.server.apparmor import AppArmorManager

    mgr = AppArmorManager.__new__(AppArmorManager)
    mgr._premium_root = None
    mgr._venv_path = "/venv"
    mgr._source_root = "/src"
    body = mgr._render_sub_profile("p", "a", "/ws")
    return {m.group(1) for m in re.finditer(
        r'audit deny "/ws/\.jaato/([^/"*]+)[^"]*"\s+r,', body)}


def test_the_read_denies_agree_with_the_apparmor_isolated_profile():
    """One list per LSM; this keeps them the same list."""
    denied = _apparmor_read_denied()
    assert denied, "the AppArmor body's read-denies were not found"
    assert denied == set(selinux_labels.ISOLATED_UNREADABLE) | {selinux_labels.PROMPTS_DIR}


def test_every_unreadable_entry_is_authored():
    names = {name for name, _ in selinux_labels.authored_entries()}
    assert set(selinux_labels.ISOLATED_UNREADABLE) <= names


# ---------------------------------------------------------------- daemon


def _payload():
    return {"name": "researcher", "description": "d", "model": "m",
            "provider": "anthropic", "plugins": ["cli"], "plugin_configs": {}}


def _manager(backend):
    sm = SessionManager.__new__(SessionManager)
    import threading
    sm._lock = threading.RLock()
    sm._isolated_sub_runners = {}
    sm._sessions = {}
    sm._selinux_backend = backend
    sm._apparmor_manager = MagicMock()
    return sm


def _handle(label="u:r:jaato_isolated_t:s0:c1,c2"):
    return ConfinementHandle(backend="selinux", label=label,
                             confinement_id="cid", child_label=label)


def test_a_subpath_tightening_is_refused():
    backend = MagicMock()
    sm = _manager(backend)
    result = sm._spawn_isolated_runner(
        parent_session_id="sess-A", subagent_id="agent-1",
        profile_payload=_payload(), task="t", workspace_path="/work",
        sub_profile_tightenings={"isolated_workspace_subpath": "docs"})
    assert result["ok"] is False and result["stage"] == "sub_profile"
    assert "isolated_workspace_subpath" in result["error"]
    backend.provision_isolated.assert_not_called()


def test_the_daemon_provisions_the_isolated_domain_and_never_apparmor():
    backend = MagicMock()
    backend.provision_isolated.return_value = _handle()
    sm = _manager(backend)
    seen = {}

    def spawn(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the spawn arguments")

    sm._do_spawn_isolated_runner = spawn
    sm._spawn_isolated_runner(
        parent_session_id="sess-A", subagent_id="agent-1",
        profile_payload=_payload(), task="t", workspace_path="/work",
        sub_profile_tightenings={"isolated_read_only_workspace": True})
    args, kwargs = backend.provision_isolated.call_args
    assert args[0] == "sess-A__sub_agent-1"
    assert kwargs["read_only"] is True
    assert seen["confinement"] is backend.provision_isolated.return_value
    assert seen["sub_apparmor_profile"] == ""
    sm._apparmor_manager.provision_sub_profile.assert_not_called()


def test_a_selinux_rollback_does_not_touch_apparmor():
    sm = _manager(MagicMock())
    sm._rollback_isolated_resources(
        parent_session_id="sess-A", subagent_id="agent-1",
        isolated_session_id="sess-A__sub_agent-1", cgroup_path="",
        sub_profile_name="")
    sm._apparmor_manager.teardown_sub_profile.assert_not_called()


def test_an_apparmor_rollback_still_unloads_its_sub_profile():
    sm = _manager(None)
    sm._rollback_isolated_resources(
        parent_session_id="sess-A", subagent_id="agent-1",
        isolated_session_id="sess-A__sub_agent-1", cgroup_path="",
        sub_profile_name="jaato-ws-sess-A//agent-1")
    sm._apparmor_manager.teardown_sub_profile.assert_called_once()


def test_the_envelope_names_the_isolated_domain():
    from jaato_server.shared.plugins.subagent.config import build_inline_profile

    profile = build_inline_profile(_payload(), name="researcher", description="d")
    handle = _handle()
    envelope = SessionManager._build_isolated_envelope(
        MagicMock(), profile=profile, isolated_session_id="iso-1",
        workspace_path="/tmp/ws", sub_apparmor_profile="",
        agent_params=None, confinement=handle)
    assert envelope.profile_name == ""
    assert envelope.confinement == {
        "backend": "selinux", "label": handle.label,
        "child_label": handle.label, "confinement_id": "cid"}


def test_the_runner_accepts_the_isolated_descriptor():
    from jaato_server.server.runner import lsm_confine

    handle = _handle()
    envelope = SimpleNamespace(profile_name="", confinement={
        "backend": "selinux", "label": handle.label,
        "child_label": handle.label, "confinement_id": "cid"})
    resolved = lsm_confine.resolve(envelope)
    assert resolved.label == resolved.child_label == handle.label


# ---------------------------------------------------------------- runner


def test_an_isolated_spawn_names_its_parent_session():
    """The runner names the parent by the id ``JaatoSession`` actually holds.

    The daemon refuses an empty ``parent_session_id``.  The plugin read
    ``_session_id`` / ``session_id``, which no ``JaatoSession`` has, so
    every isolated spawn on the runner path was refused, on any LSM.
    Found by driving ``live_session.py --isolated`` on an unconfined
    daemon.
    """
    from jaato_server.shared.jaato_session import JaatoSession
    from jaato_server.shared.plugins.subagent.config import SubagentProfile
    from jaato_server.shared.plugins.subagent.plugin import SubagentPlugin

    parent = JaatoSession.__new__(JaatoSession)
    parent.set_daemon_session_id("sess-A")
    rpc = MagicMock()
    rpc.spawn_isolated_runner.return_value = {"ok": True}
    plugin = SubagentPlugin()
    plugin._runtime = SimpleNamespace(
        registry=SimpleNamespace(runner_rpc_client=rpc))
    plugin._parent_session = parent
    plugin._dispatch_isolated_spawn(
        agent_id="a1", task="t", workspace_path="/w", agent_params=None,
        display_name="a1",
        profile=SubagentProfile(name="p", description="d", model="m",
                                provider="anthropic", plugins=["file_edit"]))
    assert rpc.spawn_isolated_runner.call_args.kwargs["parent_session_id"] == "sess-A"


# ---------------------------------------------------------------- the log


def test_the_sub_runner_log_is_labelled_for_append(tmp_path):
    kernel = _Kernel()
    ws = _workspace(tmp_path)
    backend = _backend(kernel, tmp_path)
    handle = backend.provision_isolated(
        "s1__sub_a", Boundary(workspace_path=str(ws)), read_only=True)
    log = ws / ".jaato" / "logs" / "runner-s1__sub_a.log"
    backend.prepare_runner_log(handle, str(log))
    assert log.exists() and (log.stat().st_mode & 0o777) == 0o600
    assert _type(kernel.labels[str(log)]) == "jaato_runner_log_t"
    assert _level(kernel.labels[str(log)]) == _level(handle.label)


def test_the_daemon_labels_the_log_before_the_spawn(monkeypatch):
    import asyncio
    from jaato_server.server import runner_spawner

    order = []
    backend = MagicMock()
    backend.prepare_runner_log.side_effect = lambda h, p: order.append(("label", p))

    def spawn(self, **kwargs):
        order.append(("spawn", kwargs["log_path"]))
        raise RuntimeError("stop after the spawn")

    monkeypatch.setattr(runner_spawner.RunnerSpawner, "spawn", spawn)
    sm = _manager(backend)
    sm._daemon_loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="stop after the spawn"):
            sm._do_spawn_isolated_runner(
                parent_session_id="sess-A", subagent_id="a1",
                isolated_session_id="sess-A__sub_a1", workspace_path="/work",
                sub_apparmor_profile="", cgroup_path="", profile=MagicMock(),
                confinement=_handle(), effective_runtime_limits=MagicMock(),
                agent_params=None)
    finally:
        sm._daemon_loop.close()
    log = "/work/.jaato/logs/runner-sess-A__sub_a1.log"
    assert order == [("label", log), ("spawn", log)]
