"""A notebook kernel in an enforcing SELinux ``jaato_child_t`` gets the kernel tier (#1519).

``kernel_sandbox`` knew two tiers, AppArmor and the audit hook, so on an
SELinux host the kernel always got the audit tier.  That tier refuses
``ctypes.dlopen(None)``, so ``import ctypes`` (and numpy, pandas, scipy, ...)
failed in a cell although the kernel ran in a real kernel boundary.

The SELinux tier needs positive evidence on three counts (#1014):

- the kernel's OWN context is ``jaato_child_t`` (the domain a cell cannot
  leave, #1323's rule), and the one the runner exec'd it into;
- the host is not read as permissive by the kernel (``/sys/fs/selinux/enforce``,
  when the domain may read it);
- the DAEMON attested the host enforcing and neither domain permissive.  A
  confined task cannot ask that itself (no ``security_t`` read, no
  ``compute_av``), so the attestation travels daemon -> envelope -> runner ->
  kernel argv, never the environment.

And one property that must not regress: the kernel imports the label reader
before installing its audit hook, so that reader must not import ``ctypes``,
or ``ctypes.pythonapi`` would be in ``sys.modules`` on the audit tier (#1011).

No SELinux kernel is involved: contexts, the host switch and the active LSM
are fabricated.  What is checked is that the framework decides the tier from
them correctly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from jaato_server.shared import lsm_label
from jaato_server.shared.plugins.notebook import kernel_sandbox
from jaato_server.shared.tests.reversion import Reversion

_NB = "jaato-server/jaato_server/shared/plugins/notebook"
_LSM = "jaato-server/jaato_server/shared/lsm_label.py"
_CONF = "jaato-server/jaato_server/server/confinement"

CHILD = "system_u:system_r:jaato_child_t:s0:c1,c2"
RUNNER = "system_u:system_r:jaato_runner_t:s0:c1,c2"

REVERSIONS = [
    Reversion(
        target=f"{_NB}/kernel_sandbox.py",
        find="        return BOUNDARY_SELINUX, f\"SELinux-enforced domain {context}\"\n",
        replace="        pass\n",
        test="test_an_enforcing_child_kernel_gets_the_selinux_tier_and_ctypes",
        because="a kernel in jaato_child_t falls to the audit tier and cannot import numpy",
    ),
    Reversion(
        target=_LSM,
        find="SELINUX_CELL_BOUNDARY_DOMAINS = frozenset({SELINUX_CHILD_DOMAIN})\n",
        replace=("SELINUX_CELL_BOUNDARY_DOMAINS = frozenset("
                 "{SELINUX_CHILD_DOMAIN, \"jaato_runner_t\"})\n"),
        test="test_a_kernel_in_the_runner_domain_uses_the_audit_hook",
        because="a kernel the setexeccon step missed would claim a kernel boundary",
    ),
    Reversion(
        target=_LSM,
        find="    if (host_enforcing or selinux_host_enforcing)() is False:\n        return None\n",
        replace="",
        test="test_a_permissive_host_uses_the_audit_hook",
        because="a permissive host would be claimed as a kernel boundary",
    ),
    Reversion(
        target=_LSM,
        find="    return cleaned if enforcing_attested else None\n",
        replace="    return cleaned\n",
        test="test_without_the_daemons_attestation_the_kernel_uses_the_audit_hook",
        because="a permissive domain (unknowable from inside it) would count as enforcing",
    ),
    Reversion(
        target=_LSM,
        find="    if detected != BACKEND_SELINUX and not expected_label:\n        return None\n",
        replace="    if detected != BACKEND_SELINUX:\n        return None\n",
        test="test_an_unreadable_lsm_list_still_gets_the_selinux_tier",
        because="inside jaato_child_t securityfs and selinuxfs are security_t and "
                "unreadable, so the backend reads 'none' on every SELinux host and "
                "every kernel fell to the audit tier",
    ),
    Reversion(
        target=_LSM,
        find="    if detected == BACKEND_APPARMOR:\n        return None\n",
        replace="",
        test="test_off_selinux_the_context_is_not_consulted",
        because="on an AppArmor host a label shaped like a context would be read as one",
    ),
    Reversion(
        target=_LSM,
        find="import os\nimport threading\n",
        replace="import ctypes\nimport os\nimport threading\n",
        test="test_the_kernel_imports_no_ctypes_before_its_hook",
        because="ctypes.pythonapi would be loaded before the audit hook, an escape (#1011)",
    ),
    Reversion(
        target=f"{_CONF}/base.py",
        find="        \"enforcing\": bool(handle.enforcing_attested),\n",
        replace="",
        test="test_the_descriptor_carries_the_daemons_attestation",
        because="no runner would ever learn the boundary is enforced",
    ),
    Reversion(
        target=f"{_CONF}/selinux.py",
        find="        return self._domain_permissive(child_label) is False\n",
        replace="        return True\n",
        test="test_a_permissive_child_domain_is_not_attested",
        because="semanage permissive -a jaato_child_t would still be attested enforcing",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner/session.py",
        find="        enforcing_attested=bool(getattr(confinement, \"enforcing_attested\", False)),\n",
        replace="        enforcing_attested=False,\n",
        test="test_the_runner_records_the_boundary_it_confirmed",
        because="the notebook backend would never pass the attestation to a kernel",
    ),
    Reversion(
        target=f"{_NB}/backends/subprocess_kernel.py",
        find="        containment_args += self._selinux_kernel_args()\n",
        replace="",
        test="test_the_kernel_is_given_the_child_label_in_argv",
        because="the kernel never learns which context it was exec'd into",
    ),
    Reversion(
        target=f"{_NB}/kernel_main.py",
        find="        selinux_enforcing_attested=bool(args.selinux_enforcing_attested),\n",
        replace="",
        test="test_kernel_main_passes_the_selinux_args_to_containment",
        because="the attestation is parsed and dropped",
    ),
    Reversion(
        target=f"{_NB}/kernel_sandbox.py",
        find="    BOUNDARY_SELINUX: (\n        \"Cells are bounded by an enforcing SELinux domain",
        replace="    \"_unused\": (\n        \"Cells are bounded by an enforcing SELinux domain",
        test="test_the_notice_names_selinux",
        because="the model is told nothing about the tier it is on",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/environment/runtime.py",
        find="        report.setdefault(\"notebook\", {}).update(boundary)\n",
        replace="        pass\n",
        test="test_the_runtime_aspect_names_the_notebook_tier",
        because="get_environment(aspect='runtime') would not say which tier cells run on",
    ),
]


# ---- the kernel's decision, in a fresh interpreter --------------------------

_SCRIPT = textwrap.dedent("""
    import json, sys
    from jaato_server.shared import lsm_label
    from jaato_server.shared.plugins.notebook import kernel_sandbox
    preloaded = "ctypes" in sys.modules
    ctx, host, attested, expected, backend = json.loads(sys.argv[2])
    lsm_label.active_lsm_backend = lambda **_k: backend
    lsm_label.read_own_context = lambda *_a, **_k: ctx
    lsm_label.selinux_host_enforcing = lambda *_a, **_k: host
    kind, _ = kernel_sandbox.establish_containment(
        sys.argv[1], selinux_child_label=expected,
        selinux_enforcing_attested=attested)
    try:
        import ctypes  # what numpy does first
        ctypes_ok = True
    except Exception:
        ctypes_ok = False
    try:
        import numpy  # noqa: F401
        numpy_ok = True
    except ImportError as exc:
        numpy_ok = None if type(exc) is ModuleNotFoundError else False
    except Exception:
        numpy_ok = False
    print(json.dumps({"kind": kind, "ctypes": ctypes_ok, "numpy": numpy_ok,
                      "preloaded": preloaded}))
""")


def _kernel(tmp_path, ctx, *, host=True, attested=True, expected=CHILD,
            backend="selinux"):
    """Run the containment decision as the kernel would, then a cell's import."""
    res = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path),
         json.dumps([ctx, host, attested, expected, backend])],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_an_enforcing_child_kernel_gets_the_selinux_tier_and_ctypes(tmp_path):
    out = _kernel(tmp_path, CHILD)
    assert out["kind"] == kernel_sandbox.BOUNDARY_SELINUX
    assert out["ctypes"], "import ctypes refused although the kernel is in jaato_child_t"
    # numpy, where installed: the import the issue reported refused.
    assert out["numpy"] in (True, None)


def test_an_unreadable_host_switch_leaves_it_to_the_attestation(tmp_path):
    # jaato_child_t may not read /sys/fs/selinux/enforce on the shipped policy.
    out = _kernel(tmp_path, CHILD, host=None)
    assert out["kind"] == kernel_sandbox.BOUNDARY_SELINUX


def test_a_kernel_in_the_runner_domain_uses_the_audit_hook(tmp_path):
    out = _kernel(tmp_path, RUNNER, expected=None)
    assert out["kind"] == kernel_sandbox.BOUNDARY_AUDIT
    assert not out["ctypes"]
    assert out["numpy"] in (False, None)


def test_a_permissive_host_uses_the_audit_hook(tmp_path):
    out = _kernel(tmp_path, CHILD, host=False)
    assert out["kind"] == kernel_sandbox.BOUNDARY_AUDIT
    assert not out["ctypes"]


def test_without_the_daemons_attestation_the_kernel_uses_the_audit_hook(tmp_path):
    # A permissive DOMAIN cannot be seen from inside it; no attestation, no tier.
    out = _kernel(tmp_path, CHILD, attested=False)
    assert out["kind"] == kernel_sandbox.BOUNDARY_AUDIT
    assert not out["ctypes"]


@pytest.mark.parametrize("ctx", [None, "", "garbage"])
def test_an_unreadable_context_uses_the_audit_hook(tmp_path, ctx):
    out = _kernel(tmp_path, ctx)
    assert out["kind"] == kernel_sandbox.BOUNDARY_AUDIT


def test_a_context_other_than_the_one_exec_d_into_uses_the_audit_hook(tmp_path):
    other = "system_u:system_r:jaato_child_t:s0:c7,c8"
    out = _kernel(tmp_path, other)
    assert out["kind"] == kernel_sandbox.BOUNDARY_AUDIT


def test_off_selinux_the_context_is_not_consulted(tmp_path):
    out = _kernel(tmp_path, CHILD, backend="apparmor")
    assert out["kind"] == kernel_sandbox.BOUNDARY_AUDIT


def test_an_unreadable_lsm_list_still_gets_the_selinux_tier(tmp_path):
    # Inside jaato_child_t, /sys/kernel/security/lsm and /sys/fs/selinux are
    # security_t: active_lsm_backend() reads "none" on every SELinux host.
    # The spawner's label matching the kernel's own context is the evidence.
    out = _kernel(tmp_path, CHILD, backend="none", host=None)
    assert out["kind"] == kernel_sandbox.BOUNDARY_SELINUX
    assert out["ctypes"]


def test_an_unreadable_lsm_list_without_a_named_label_uses_the_audit_hook(tmp_path):
    out = _kernel(tmp_path, CHILD, backend="none", expected=None)
    assert out["kind"] == kernel_sandbox.BOUNDARY_AUDIT


def test_detection_with_securityfs_and_selinuxfs_unreadable(tmp_path, monkeypatch):
    # The real detection, its three sources all unreadable, as in the domain.
    missing = dict(lsm_list_path=str(tmp_path / "no-lsm"),
                   apparmor_path=str(tmp_path / "no-apparmor"),
                   selinux_path=str(tmp_path / "no-selinuxfs"))
    real = lsm_label.active_lsm_backend
    assert real(**missing) == lsm_label.BACKEND_NONE
    monkeypatch.setattr(lsm_label, "active_lsm_backend",
                        lambda **_k: real(**missing))
    got = lsm_label.selinux_cell_boundary(
        raw=CHILD, expected_label=CHILD, enforcing_attested=True,
        host_enforcing=lambda: lsm_label.selinux_host_enforcing(
            str(tmp_path / "no-selinuxfs" / "enforce")))
    assert got == CHILD


def test_the_kernel_imports_no_ctypes_before_its_hook(tmp_path):
    # Whatever the tier, the decision itself must not load ctypes.
    for ctx in (CHILD, RUNNER):
        assert _kernel(tmp_path, ctx)["preloaded"] is False


# ---- AppArmor unchanged -----------------------------------------------------

def test_apparmor_child_still_wins_first(monkeypatch, tmp_path):
    installed = []
    monkeypatch.setattr(kernel_sandbox, "apparmor_enforced_profile",
                        lambda: "jaato-ws-abc//child")
    monkeypatch.setattr(kernel_sandbox, "install", installed.append)

    def _not_asked(**_k):
        raise AssertionError("the SELinux tier was consulted under AppArmor")

    monkeypatch.setattr(lsm_label, "selinux_cell_boundary", _not_asked)
    kind, _ = kernel_sandbox.establish_containment(str(tmp_path))
    assert kind == kernel_sandbox.BOUNDARY_APPARMOR
    assert not installed


def test_apparmor_base_profile_still_uses_the_audit_hook(monkeypatch, tmp_path):
    installed = []
    monkeypatch.setattr(kernel_sandbox, "apparmor_enforced_profile",
                        lambda: "jaato-ws-abc")
    monkeypatch.setattr(kernel_sandbox, "install", installed.append)
    monkeypatch.setattr(lsm_label, "active_lsm_backend", lambda **_k: "apparmor")
    kind, _ = kernel_sandbox.establish_containment(str(tmp_path))
    assert kind == kernel_sandbox.BOUNDARY_AUDIT
    assert installed


# ---- daemon -> envelope -> runner -> kernel ---------------------------------

def _handle(attested):
    from jaato_server.server.confinement.base import ConfinementHandle
    return ConfinementHandle(
        backend="selinux", label=RUNNER, confinement_id="id", child_label=CHILD,
        enforcing_attested=attested)


def test_the_descriptor_carries_the_daemons_attestation():
    from jaato_server.server.confinement.base import selinux_descriptor
    from jaato_server.server.runner.lsm_confine import _resolve_selinux
    for attested in (True, False):
        descriptor = selinux_descriptor(_handle(attested))
        assert descriptor["enforcing"] is attested
        assert _resolve_selinux(RUNNER, "", descriptor).enforcing_attested is attested
    # An older daemon's descriptor (no key) is not an attestation.
    old = {"backend": "selinux", "label": RUNNER, "child_label": CHILD}
    assert _resolve_selinux(RUNNER, "", old).enforcing_attested is False


def _backend(host, perm):
    from jaato_server.server.confinement.selinux import SELinuxBackend
    be = SELinuxBackend.__new__(SELinuxBackend)
    be._host_enforcing = lambda: host
    be._domain_permissive = lambda ctx: perm.get(ctx.split(":")[2])
    return be


def test_a_permissive_child_domain_is_not_attested():
    attest = lambda be, runner_perm: be._attest_enforcing(RUNNER, CHILD, runner_perm)  # noqa: E731
    assert attest(_backend(True, {"jaato_child_t": False}), False) is True
    assert attest(_backend(True, {"jaato_child_t": True}), False) is False
    assert attest(_backend(True, {"jaato_child_t": None}), False) is False
    assert attest(_backend(True, {"jaato_child_t": False}), None) is False
    assert attest(_backend(False, {"jaato_child_t": False}), False) is False
    assert attest(_backend(None, {"jaato_child_t": False}), False) is False


def test_the_runner_records_the_boundary_it_confirmed(monkeypatch):
    from jaato_server.server.runner import lsm_confine, session
    from jaato_server.server.runner.lsm_confine import RunnerConfinement
    monkeypatch.setattr(lsm_confine, "self_confine", lambda *a, **k: None)
    monkeypatch.setattr(session, "_retire_and_verify_threads", lambda *a, **k: None)
    try:
        session._confirm_selinux_domain(RunnerConfinement(
            backend="selinux", label=RUNNER, child_label=CHILD,
            enforcing_attested=True), None)
        recorded = lsm_label.selinux_session_boundary()
        assert recorded is not None
        assert recorded.child_label == CHILD
        assert recorded.enforcing_attested is True
    finally:
        lsm_label.set_selinux_session_boundary(None)


@pytest.fixture
def recorded_boundary():
    lsm_label.set_selinux_session_boundary(lsm_label.SELinuxSessionBoundary(
        label=RUNNER, child_label=CHILD, enforcing_attested=True))
    yield
    lsm_label.set_selinux_session_boundary(None)


def _subprocess_backend(tmp_path):
    from jaato_server.shared.plugins.notebook.backends.subprocess_kernel import (
        SubprocessKernelBackend,
    )
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    return be


def test_the_backend_expects_the_selinux_tier_only_with_the_transition(
        recorded_boundary, tmp_path):
    be = _subprocess_backend(tmp_path)
    assert be.boundary_kind() == kernel_sandbox.BOUNDARY_AUDIT
    be.set_apparmor_child_transition(lambda: None)
    assert be.boundary_kind() == kernel_sandbox.BOUNDARY_SELINUX
    allowed, text = be.execution_boundary()
    assert allowed and "SELinux" in text and "jaato_child_t" in text


def test_an_unattested_boundary_is_not_expected(tmp_path):
    lsm_label.set_selinux_session_boundary(lsm_label.SELinuxSessionBoundary(
        label=RUNNER, child_label=CHILD, enforcing_attested=False))
    try:
        be = _subprocess_backend(tmp_path)
        be.set_apparmor_child_transition(lambda: None)
        assert be.boundary_kind() == kernel_sandbox.BOUNDARY_AUDIT
        assert be._selinux_kernel_args() == []
    finally:
        lsm_label.set_selinux_session_boundary(None)


def test_the_kernel_is_given_the_child_label_in_argv(
        recorded_boundary, monkeypatch, tmp_path):
    from jaato_server.shared.plugins.notebook.backends import subprocess_kernel
    seen = []

    def _capture(argv, close_on_failure=(), **_kw):
        seen.append(list(argv))
        for fd in close_on_failure:
            try:
                os.close(fd)
            except OSError:
                pass
        raise OSError("captured")

    monkeypatch.setattr(subprocess_kernel, "_popen_or_close", _capture)
    be = _subprocess_backend(tmp_path)
    be.set_apparmor_child_transition(lambda: None)
    with pytest.raises(Exception):
        be.create_notebook("t")
    assert seen, "no kernel spawn attempted"
    argv = seen[0]
    i = argv.index("--selinux-child-label")
    assert argv[i + 1] == CHILD
    assert "--selinux-enforcing-attested" in argv


def test_kernel_main_passes_the_selinux_args_to_containment(monkeypatch, tmp_path):
    from jaato_server.shared.plugins.notebook import kernel_main
    calls = []

    def _establish(root, **kw):
        calls.append(kw)
        return kernel_sandbox.BOUNDARY_SELINUX, "test"

    monkeypatch.setattr(kernel_sandbox, "establish_containment", _establish)
    monkeypatch.chdir(tmp_path)
    r2k_r, r2k_w = os.pipe()
    k2r_r, k2r_w = os.pipe()
    os.close(r2k_w)  # EOF: the kernel loop ends after READY
    try:
        kernel_main.main([
            "--workspace-root", str(tmp_path),
            "--read-fd", str(r2k_r), "--write-fd", str(k2r_w),
            "--selinux-child-label", CHILD, "--selinux-enforcing-attested",
        ])
    finally:
        os.close(k2r_r)
    assert calls[0]["selinux_child_label"] == CHILD
    assert calls[0]["selinux_enforcing_attested"] is True


# ---- what the model is told -------------------------------------------------

def test_the_notice_names_selinux():
    lines = kernel_sandbox.boundary_notice(kernel_sandbox.BOUNDARY_SELINUX)
    text = " ".join(lines)
    assert "SELinux" in text and "jaato_child_t" in text
    assert "import ctypes" in text and "numpy" in text


def test_the_runtime_aspect_names_the_notebook_tier():
    from types import SimpleNamespace
    from jaato_server.shared.plugins.environment import runtime as runtime_mod
    from jaato_server.shared.apparmor_label import parse_label

    backend = SimpleNamespace(boundary_kind=lambda: kernel_sandbox.BOUNDARY_SELINUX)
    notebook = SimpleNamespace(_backends={"subprocess": backend},
                               _active_backend_name="subprocess")
    registry = SimpleNamespace(
        get_plugin=lambda name: notebook if name == "notebook" else None)
    report = runtime_mod.runtime_report(
        registry, None, None,
        read_label=lambda: parse_label("unconfined"), grants=lambda: None)
    block = report["notebook"]
    assert block["boundary"] == kernel_sandbox.BOUNDARY_SELINUX
    assert any("SELinux" in line for line in block["boundary_notice"])
    assert runtime_mod.runtime_summary(report)["notebook_boundary"] == "selinux"
