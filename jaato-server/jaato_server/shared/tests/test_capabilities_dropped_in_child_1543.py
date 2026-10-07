"""Model-driven subprocesses run with no capabilities (#1543).

A confined payload kept the daemon's full capability bounding set, so under
a root daemon a ``cli`` command started every ``exec`` with every capability
permitted and effective, and only the LSM stood between it and their use.
The runner now composes a capability drop into the ``//child`` preexec:
bounding set before the LSM transition (it needs ``CAP_SETPCAP``, which only
the runner's profile grants), the process sets and NO_NEW_PRIVS after it.

Most tests drive the drop through a recording fake libc, so they say the same
thing as root and as an ordinary user.  The kernel tests run a real ``cli``
subprocess: as root they assert every set is empty; as an ordinary user the
bounding set cannot be reduced, and they assert the ``partial`` posture and
empty process sets instead.  No AppArmor or SELinux kernel is needed: the LSM
transition is a no-op stand-in, which is the part the drop does not depend on.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from jaato_server.shared import capability_drop as cd
from jaato_server.shared.tests.reversion import Reversion

_CD = "jaato-server/jaato_server/shared/capability_drop.py"
_SESSION = "jaato-server/jaato_server/server/runner/session.py"
_CFG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"
_AA = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_CD,
        find="        drop.before_transition()\n        lsm_transition()\n",
        replace="        lsm_transition()\n",
        test="test_the_bounding_set_is_dropped_before_the_transition",
        because="the bounding set is never reduced, so a root payload keeps "
                "every capability across exec",
    ),
    Reversion(
        target=_CD,
        find="        self._libc.capset(ctypes.byref(self._hdr), self._data)\n",
        replace="",
        test="test_the_process_sets_are_cleared_after_the_transition",
        because="permitted / effective survive into the payload when the "
                "bounding drop was refused",
    ),
    Reversion(
        target=_CD,
        find="        self._libc.capset(ctypes.byref(self._hdr), self._data)\n"
             "        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)\n",
        replace="        self._libc.capset(ctypes.byref(self._hdr), self._data)\n",
        test="test_the_process_sets_are_cleared_after_the_transition",
        because="under seccomp: off nothing sets NO_NEW_PRIVS, and a root "
                "exec recomputes permitted as the whole bounding set",
    ),
    Reversion(
        target=_SESSION,
        find="    lsm_cb = capability_drop.compose_child_preexec(lsm_cb, caps)\n",
        replace="",
        test="test_the_runner_composes_the_drop_into_the_child_preexec",
        because="the plan is made and never reaches a subprocess",
    ),
    Reversion(
        target=_CFG,
        find='    if "none" in values:\n        return "none"\n',
        replace="",
        test="test_inheritance_is_most_restrictive_wins",
        because="a child list re-keeps capabilities a parent dropped with "
                "'none'",
    ),
    Reversion(
        target=_AA,
        find="  # //child, and PR_CAPBSET_DROP needs CAP_SETPCAP.\n"
             "  capability setpcap,\n",
        replace="  # //child, and PR_CAPBSET_DROP needs CAP_SETPCAP.\n",
        test="test_only_the_runner_bodies_grant_setpcap",
        because="the bounding drop is refused by the base profile on every "
                "AppArmor host",
    ),
]


# ---------------------------------------------------------------- fakes


class _FakeFn:
    def __init__(self, log: List[Any], name: str) -> None:
        self.log, self.name, self.restype = log, name, None

    def __call__(self, *args: Any) -> int:
        self.log.append((self.name, args[:2] if self.name == "prctl" else ()))
        return 0


class _FakeLibc:
    def __init__(self, log: List[Any]) -> None:
        self.prctl = _FakeFn(log, "prctl")
        self.capset = _FakeFn(log, "capset")


def _fake_drop(log: List[Any]) -> cd.CapabilityDrop:
    real = cd.prepare(())
    return cd.CapabilityDrop(real.keep_mask, (3, 21), _FakeLibc(log),
                             real._hdr, real._data)


def _composed(log: List[Any]):
    plan = cd.CapabilityPlan(cd.POSTURE_DROPPED, drop=_fake_drop(log))
    return cd.compose_child_preexec(lambda: log.append(("lsm", ())), plan)


# ----------------------------------------------------------- the order


def test_the_bounding_set_is_dropped_before_the_transition():
    log: List[Any] = []
    _composed(log)()
    lsm = log.index(("lsm", ()))
    drops = [i for i, e in enumerate(log)
             if e == ("prctl", (cd.PR_CAPBSET_DROP, 3))]
    assert drops and drops[0] < lsm, log


def test_the_process_sets_are_cleared_after_the_transition():
    log: List[Any] = []
    _composed(log)()
    lsm = log.index(("lsm", ()))
    after = [name for name, _ in log[lsm + 1:]]
    assert after.count("capset") == 1, log
    assert ("prctl", (cd.PR_SET_NO_NEW_PRIVS, 1)) in log[lsm + 1:], log
    assert ("prctl", (cd.PR_CAP_AMBIENT, cd.PR_CAP_AMBIENT_CLEAR_ALL)) in log[lsm + 1:]


def test_the_capset_data_keeps_nothing_by_default():
    drop = cd.prepare(())
    for word in drop._data:
        assert (word.effective, word.permitted, word.inheritable) == (0, 0, 0)
    assert drop.keep_mask == 0
    assert set(drop.drop) == set(range(cd._last_cap() + 1))


def test_a_kept_capability_is_left_in_every_set_it_can_be():
    drop = cd.prepare(["CAP_NET_BIND_SERVICE"])
    bit = 1 << cd.CAPABILITIES["net_bind_service"]
    assert drop.keep_mask == bit
    assert cd.CAPABILITIES["net_bind_service"] not in drop.drop
    assert drop._data[0].inheritable == 0


# --------------------------------------------------------------- postures


@pytest.mark.parametrize("probed, posture, installed", [
    ((True, True), cd.POSTURE_DROPPED, True),
    ((False, True), cd.POSTURE_PARTIAL, True),
    ((False, False), cd.POSTURE_ABSENT, False),
])
def test_the_posture_follows_the_probe(probed, posture, installed):
    plan = cd.plan_for_session(None, boundary_active=True,
                               prober=lambda _d: probed)
    assert plan.posture == posture
    assert (plan.drop is not None) == installed
    assert cd.current_posture()["posture"] == posture
    if posture != cd.POSTURE_DROPPED:
        assert cd.current_posture()["reason"]


def test_inherit_and_unconfined_install_nothing(caplog):
    lsm = lambda: None  # noqa: E731
    plan = cd.plan_for_session("inherit", boundary_active=True,
                               prober=lambda _d: (True, True))
    assert plan.posture == cd.POSTURE_INHERIT
    assert cd.compose_child_preexec(lsm, plan) is lsm
    assert "inherit" in caplog.text
    plan = cd.plan_for_session(None, boundary_active=False)
    assert plan.posture == cd.POSTURE_UNCONFINED
    assert cd.compose_child_preexec(lsm, plan) is lsm


def test_an_unknown_name_is_not_kept_and_is_said():
    plan = cd.plan_for_session(["net_raw", "flying"], boundary_active=True,
                               prober=lambda _d: (True, True))
    assert plan.kept == ("net_raw",)
    assert plan.as_dict()["ignored"] == ["flying"]


# ------------------------------------------------------------- the profile


def test_runtime_limits_accepts_the_three_forms():
    from jaato_server.shared.runtime_limits import RuntimeLimits
    assert RuntimeLimits.from_dict({"capabilities": "none"}).capabilities == "none"
    assert RuntimeLimits.from_dict({"capabilities": "inherit"}).capabilities == "inherit"
    assert RuntimeLimits.from_dict(
        {"capabilities": ["net_raw"]}).capabilities == ("net_raw",)
    with pytest.raises(ValueError):
        RuntimeLimits.from_dict({"capabilities": "all"})
    with pytest.raises(ValueError):
        RuntimeLimits.from_dict({"capabilities": [""]})


def test_the_scalar_vocabularies_agree():
    from jaato_server.shared.runtime_limits import CAPABILITY_MODES
    assert tuple(CAPABILITY_MODES) == tuple(cd.MODES)


def test_inheritance_is_most_restrictive_wins():
    from jaato_server.shared.plugins.subagent.config import _merged_capabilities
    from jaato_server.shared.runtime_limits import RuntimeLimits as RL
    assert _merged_capabilities([RL(capabilities="none"),
                                 RL(capabilities=["net_raw"])]) == "none"
    assert _merged_capabilities([RL(capabilities="inherit"),
                                 RL(capabilities=["net_raw"])]) == ("net_raw",)
    assert _merged_capabilities([RL(capabilities=["net_raw", "kill"]),
                                 RL(capabilities=["kill"])]) == ("kill",)
    assert _merged_capabilities([RL(capabilities="inherit")]) == "inherit"
    assert _merged_capabilities([RL()]) is None


def test_validate_reports_unknown_names_and_inherit():
    from jaato_server.shared.runtime_limits import RuntimeLimits
    from jaato_server.shared.scaffold.validate import _check_capabilities
    found: List[Any] = []

    def add(sev, code, *_a, **_k):
        found.append((sev, code))
    _check_capabilities(SimpleNamespace(runtime_limits=RuntimeLimits(
        capabilities=["sys_flying"])), add)
    _check_capabilities(SimpleNamespace(runtime_limits=RuntimeLimits(
        capabilities="inherit")), add)
    assert ("error", "capability_unknown") in found
    assert ("warn", "capabilities_inherited") in found


# --------------------------------------------------------------- the runner


def test_the_runner_composes_the_drop_into_the_child_preexec(monkeypatch):
    from jaato_server.server.runner import lsm_confine
    from jaato_server.server.runner import session as runner_session

    log: List[Any] = []
    monkeypatch.setattr(lsm_confine, "child_transition_callback",
                        lambda *a, **k: (lambda: log.append(("lsm", ()))))
    monkeypatch.setattr(
        cd, "plan_for_session",
        lambda *a, **k: cd.CapabilityPlan(cd.POSTURE_DROPPED,
                                          drop=_fake_drop(log)))
    runner_session._CHILD_PREEXEC_CACHE.clear()
    envelope = SimpleNamespace(runtime_limits=None, seccomp_program=None)
    confinement = SimpleNamespace(backend="apparmor", label="jaato-ws-x",
                                  child_label="")
    try:
        runner_session._child_preexec(envelope, confinement)()
    finally:
        runner_session._CHILD_PREEXEC_CACHE.clear()
    names = [name for name, _ in log]
    assert "capset" in names and "lsm" in names, log
    assert names.index("prctl") < names.index("lsm") < names.index("capset")


def test_only_the_runner_bodies_grant_setpcap(tmp_path):
    from jaato_server.server.apparmor import AppArmorManager
    workspace = tmp_path / "ws"
    workspace.mkdir()
    text = AppArmorManager(workspace_root=str(tmp_path))._render_profile(
        "sid", str(workspace))
    hat = text.index("profile tool_hat {")
    child = text.index("profile child {")

    def grants(body: str) -> bool:
        return any(" ".join(line.split()) == "capability setpcap,"
                   for line in body.splitlines())
    assert grants(text[:hat]), "base cannot drop a child's bounding set"
    assert grants(text[hat:child]), "tool_hat cannot drop a child's bounding set"
    assert not grants(text[child:]), "//child must hold no capability"
    assert AppArmorManager._TEMPLATE_VERSION >= 47


# ----------------------------------------------------------------- kernel


def _status(text: str) -> Dict[str, int]:
    out = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.startswith("Cap"):
            out[key] = int(value.strip(), 16)
        elif key == "NoNewPrivs":
            out[key] = int(value.strip())
    return out


def test_a_cli_subprocess_runs_with_no_capabilities(monkeypatch, tmp_path: Path):
    from jaato_server.server.runner import lsm_confine
    from jaato_server.server.runner import session as runner_session
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin

    monkeypatch.setattr(lsm_confine, "child_transition_callback",
                        lambda *a, **k: (lambda: None))
    runner_session._CHILD_PREEXEC_CACHE.clear()
    envelope = SimpleNamespace(runtime_limits=None, seccomp_program=None)
    confinement = SimpleNamespace(backend="apparmor", label="jaato-ws-x",
                                  child_label="")
    plugin = CLIToolPlugin()
    plugin.initialize({"workspace_root": str(tmp_path)})
    try:
        plugin.set_apparmor_child_transition_callback(
            runner_session._child_preexec(envelope, confinement))
        (tmp_path / "status.py").write_text(
            "print(open('/proc/self/status').read())\n")
        result = plugin._execute({"command": "python3 status.py"})
    finally:
        plugin.shutdown()
        runner_session._CHILD_PREEXEC_CACHE.clear()
    assert result.get("returncode") == 0, result
    status = _status(result["stdout"])
    for key in ("CapInh", "CapPrm", "CapEff", "CapAmb"):
        assert status[key] == 0, (key, status)
    assert status["NoNewPrivs"] == 1
    if os.geteuid() == 0:
        assert status["CapBnd"] == 0, status
        assert cd.current_posture()["posture"] == cd.POSTURE_DROPPED
    else:
        # Not root: no CAP_SETPCAP to reduce the bounding set with.
        assert cd.current_posture()["posture"] == cd.POSTURE_PARTIAL
