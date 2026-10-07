"""The runner's half of SELinux confinement (selinux-backend.md §7; 2b).

No kernel.  ``/proc/self/attr/*`` and the thread task dir are fabricated,
so what is pinned is what the runner READS and WRITES:

* an envelope naming an SELinux boundary resolves to both domains, and is
  refused when it also names an AppArmor profile or lacks a domain;
* step 1c CONFIRMS the runner already wears its domain (entered by the
  exec the daemon set up), and refuses one that does not;
* a subprocess's exec context is the CHILD domain, written unbuffered;
* a thread wearing any other context is divergence (#1023), with no
  AppArmor sub-profile leniency.
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from jaato_server.server.runner import bootstrap, lsm_confine
from jaato_server.server.runner.session import BootstrapError
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_LSM = "jaato-server/jaato_server/server/runner/lsm_confine.py"

REVERSIONS = [
    Reversion(
        target=_LSM,
        find="        if actual != label:\n",
        replace="        if False:\n",
        test="test_a_runner_outside_its_domain_is_refused",
        because="a pool slot or a runner started without the exec "
                "transition would serve the session unconfined while the "
                "record said selinux",
    ),
    Reversion(
        target=_LSM,
        find="        return _selinux_exec_transition(child_label)\n",
        replace="        return _selinux_exec_transition(label)\n",
        test="test_subprocesses_exec_into_the_child_domain",
        because="subprocesses would run in jaato_runner_t, which may set "
                "exec contexts: the escape //child exists to close",
    ),
    Reversion(
        target=_LSM,
        find="    if profile:\n        raise _refuse(\n"
             "            f\"envelope names an SELinux boundary and the AppArmor profile \"\n",
        replace="    if False:\n        raise _refuse(\n"
                "            f\"envelope names an SELinux boundary and the AppArmor profile \"\n",
        test="test_an_envelope_naming_both_lsms_is_refused",
        because="the runner would pick one of two contradictory boundaries",
    ),
    Reversion(
        target=_LSM,
        find="    return label.replace(\"\\x00\", \"\").strip() == expected\n",
        replace="    return True\n",
        test="test_a_thread_in_another_context_is_divergence",
        because="a thread outside the domain would pass the #1023 check",
    ),
]

_RUNNER = "unconfined_u:unconfined_r:jaato_runner_t:s0:c1,c2"
_CHILD = "unconfined_u:unconfined_r:jaato_child_t:s0:c1,c2"
_DESCRIPTOR: Dict[str, Any] = {"backend": "selinux", "label": _RUNNER, "child_label": _CHILD}


def _envelope(profile: str = "", confinement: Any = _DESCRIPTOR) -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="s1", workspace_path="/tmp/ws", profile_name=profile,
        provider_name="echo", model_name="m", confinement=confinement)


def test_an_selinux_envelope_resolves_to_both_domains() -> None:
    conf = lsm_confine.resolve(_envelope())
    assert (conf.backend, conf.label, conf.child_label) == ("selinux", _RUNNER, _CHILD)


def test_an_envelope_naming_both_lsms_is_refused() -> None:
    with pytest.raises(BootstrapError, match="one LSM"):
        lsm_confine.resolve(_envelope(profile="jaato-ws-x"))


def test_an_envelope_without_a_child_domain_is_refused() -> None:
    with pytest.raises(BootstrapError, match="child_label"):
        lsm_confine.resolve(_envelope(confinement={"backend": "selinux", "label": _RUNNER}))


def test_a_runner_in_its_domain_is_confirmed() -> None:
    with patch.object(lsm_confine, "_own_selinux_label", return_value=_RUNNER):
        lsm_confine.self_confine("selinux", _RUNNER)


def test_a_runner_outside_its_domain_is_refused() -> None:
    with patch.object(lsm_confine, "_own_selinux_label",
                      return_value="unconfined_u:unconfined_r:unconfined_t:s0"):
        with pytest.raises(BootstrapError, match="exec transition"):
            lsm_confine.self_confine("selinux", _RUNNER)


def test_subprocesses_exec_into_the_child_domain(tmp_path: Path) -> None:
    attr = tmp_path / "exec"
    attr.write_text("")
    with patch.object(lsm_confine, "_ATTR_EXEC", str(attr)):
        lsm_confine.child_transition_callback("selinux", _RUNNER, _CHILD)()
    assert attr.read_text() == _CHILD


def _task_dir(tmp_path: Path, labels: Dict[int, str]) -> str:
    for tid, label in labels.items():
        d = tmp_path / str(tid) / "attr"
        d.mkdir(parents=True)
        (d / "current").write_text(label + "\0")
    return str(tmp_path)


def test_every_thread_in_the_domain_verifies(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, {11: _RUNNER, 12: _RUNNER})
    scan = bootstrap.verify_thread_confinement(
        _RUNNER, task_dir=task_dir, grace_seconds=0,
        matcher=lsm_confine._same_selinux_context)
    assert not scan.divergent and len(scan.matched) == 2


def test_a_thread_in_another_context_is_divergence(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, {11: _RUNNER, 12: "unconfined_u:unconfined_r:unconfined_t:s0"})
    with pytest.raises(bootstrap.ThreadConfinementDivergence):
        bootstrap.verify_thread_confinement(
            _RUNNER, task_dir=task_dir, grace_seconds=0,
            matcher=lsm_confine._same_selinux_context)


def test_the_dispatcher_verifies_selinux_with_exact_equality() -> None:
    with patch.object(bootstrap, "verify_thread_confinement") as verify:
        lsm_confine.verify_threads("selinux", _RUNNER)
    assert verify.call_args.kwargs["matcher"] is lsm_confine._same_selinux_context
