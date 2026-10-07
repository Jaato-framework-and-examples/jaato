"""The notebook kernel runs in ``//child``, a profile a cell cannot leave (#1323).

A confined session's BASE profile (``jaato-ws-<id>``) keeps
``change_profile -> unconfined`` and write access to
``/proc/self/attr/current``, because the framework restores its own threads
with them.  ``cli`` and ``interactive_shell`` move every child process into
the ``//child`` sub-profile, which drops those rules, through a
``preexec_fn`` callback the ``ToolExecutor`` forwards to them.  The notebook
plugin never took that callback, so its kernel inherited the base profile,
and on the AppArmor tier the kernel installs no hook of its own: a cell could
write ``changeprofile unconfined`` and leave confinement.  The in-process
``local`` backend was worse still: it ran cells IN the runner whenever the
runner wore an enforced profile.

Pinned here:

- the notebook plugin takes the transition callback and hands it to the
  subprocess backend;
- the kernel is spawned through it, and a failed transition starts no kernel;
- a kernel that finds itself in a profile it could leave (base, ``tool_hat``)
  does not count AppArmor as its boundary and installs the audit hook;
- that hook refuses the write that would unconfine the kernel;
- the ``local`` backend no longer runs cells because AppArmor is enforced;
- ``//child`` may read its own label (template v37), and the notebook plugin
  grants exec on the kernel's interpreter.

No kernel is involved: this container has no enforcing AppArmor.  What is
checked is that the framework asks for the transition and the grants, and
behaves correctly when the label says where it is.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest

from jaato_server.shared.plugins.notebook import kernel_sandbox
from jaato_server.shared.plugins.notebook.backends.local import (
    INPROCESS_OPT_IN_ENV,
    LocalJupyterBackend,
)
from jaato_server.shared.plugins.notebook.backends.subprocess_kernel import (
    SubprocessKernelBackend,
)
from jaato_server.shared.plugins.notebook.types import ExecutionStatus, OutputType
from jaato_server.shared.tests.reversion import Reversion

_NB = "jaato-server/jaato_server/shared/plugins/notebook"
_AA = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=f"{_NB}/plugin.py",
        find="            sub.set_apparmor_child_transition(callback)\n",
        replace="            pass\n",
        test="test_the_plugin_hands_the_transition_to_the_subprocess_backend",
        because="the kernel never learns the //child transition cli gets",
    ),
    Reversion(
        target=f"{_NB}/backends/subprocess_kernel.py",
        find="            preexec_fn=self._kernel_preexec(),\n",
        replace="            preexec_fn=_set_pdeathsig,\n",
        test="test_the_kernel_is_spawned_through_the_transition",
        because="the kernel is spawned without entering //child",
    ),
    Reversion(
        target=f"{_NB}/kernel_sandbox.py",
        find="    profile = cell_boundary_profile()\n    if profile:\n",
        replace="    profile = apparmor_enforced_profile()\n    if profile:\n",
        test="test_a_kernel_in_a_profile_it_can_leave_uses_the_audit_hook",
        because="a kernel left in the base profile trusts a boundary a cell can remove",
    ),
    Reversion(
        target=f"{_NB}/backends/local.py",
        find=(
            "        if self._allow_inprocess_opt_in:\n"
            "            if not self._opt_in_warning_logged:\n"
        ),
        replace=(
            "        if _apparmor_enforced_profile():\n"
            "            return True, \"AppArmor-enforced confinement\"\n"
            "        if self._allow_inprocess_opt_in:\n"
            "            if not self._opt_in_warning_logged:\n"
        ),
        test="test_the_local_backend_does_not_run_cells_because_apparmor_is_enforced",
        because="in-process cells run in the base profile, which they can leave",
    ),
    Reversion(
        target=_AA,
        find="    owner /proc/*/attr/current       r,\n",
        replace="",
        test="test_child_may_read_its_own_label_and_not_write_it",
        because="a kernel in //child cannot read its label and drops to the audit tier",
    ),
    Reversion(
        target=f"{_NB}/plugin.py",
        find="        ) + kernel_interpreter_apparmor_rules()\n",
        replace="        )\n",
        test="test_the_notebook_plugin_grants_exec_on_the_kernel_interpreter",
        because="a fragment-scoped //child cannot exec the kernel's interpreter",
    ),
]


def _text(result):
    return "".join(o.content for o in result.outputs
                   if o.output_type in (OutputType.STDOUT, OutputType.RESULT))


# ---- the callback reaches the kernel ----------------------------------------

def test_the_plugin_hands_the_transition_to_the_subprocess_backend(tmp_path):
    from jaato_server.shared.plugins.notebook.plugin import create_plugin
    plugin = create_plugin()
    plugin.initialize({"workspace_root": str(tmp_path)})

    def transition():
        pass

    plugin.set_apparmor_child_transition_callback(transition)
    sub = plugin._backends["subprocess"]
    assert sub._apparmor_child_transition is transition
    # A re-initialize rebuilds the backends; the transition must survive it.
    plugin.initialize({"workspace_root": str(tmp_path)})
    assert plugin._backends["subprocess"]._apparmor_child_transition is transition


def test_the_tool_executor_forwards_the_transition_to_notebook(tmp_path):
    # The runner installs the callback through ToolExecutor, which forwards
    # it to every exposed plugin that implements the setter.
    from jaato_server.shared.ai_tool_runner import ToolExecutor
    from jaato_server.shared.plugins.notebook.plugin import create_plugin
    plugin = create_plugin()
    plugin.initialize({"workspace_root": str(tmp_path)})

    class _Registry:
        def list_exposed(self):
            return ["notebook"]

        def get_plugin(self, name):
            return plugin if name == "notebook" else None

    executor = ToolExecutor()
    executor._registry = _Registry()

    def transition():
        pass

    executor.set_apparmor_child_transition_callback(transition)
    assert plugin._backends["subprocess"]._apparmor_child_transition is transition


def test_the_kernel_is_spawned_through_the_transition(tmp_path):
    # The callback runs in the forked child, so it leaves evidence on disk.
    marker = tmp_path / "entered-child"

    def transition():
        with open(marker, "w") as f:
            f.write(str(os.getpid()))

    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    be.set_apparmor_child_transition(transition)
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, "import os\nprint(os.getpid())")
        assert r.status == ExecutionStatus.COMPLETED, r.error_message
        assert marker.exists(), "the kernel was not spawned through the transition"
        assert marker.read_text() == _text(r).strip()
    finally:
        be.shutdown()


def test_a_failed_transition_starts_no_kernel(tmp_path):
    # Fail closed, as cli does: a kernel that could not enter //child would
    # sit in the base profile, which a cell can leave.
    def transition():
        raise PermissionError("changeprofile denied")

    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    be.set_apparmor_child_transition(transition)
    fds_before = set(os.listdir("/proc/self/fd"))
    try:
        with pytest.raises(subprocess.SubprocessError):
            be.create_notebook("t")
        assert not be._kernels
        leaked = set(os.listdir("/proc/self/fd")) - fds_before
        assert not leaked, f"a failed spawn leaked fds {sorted(leaked)}"
    finally:
        be.shutdown()


# ---- the kernel's own tier decision -----------------------------------------

@pytest.mark.parametrize("label, leaves", [
    ("jaato-ws-abc", True),
    ("jaato-ws-abc//tool_hat", True),
    ("jaato-ws-abc//child", False),
    ("jaato-ws-abc__sub_2", False),
    ("some-other-profile", False),
    (None, False),
])
def test_which_profiles_a_cell_can_leave(label, leaves):
    assert kernel_sandbox.profile_can_leave_itself(label) is leaves


def _establish(monkeypatch, tmp_path, label):
    installed = []
    monkeypatch.setattr(kernel_sandbox, "apparmor_enforced_profile", lambda: label)
    monkeypatch.setattr(kernel_sandbox, "install", installed.append)
    kind, _ = kernel_sandbox.establish_containment(str(tmp_path))
    return kind, installed


def test_a_kernel_in_a_profile_it_can_leave_uses_the_audit_hook(monkeypatch, tmp_path):
    kind, installed = _establish(monkeypatch, tmp_path, "jaato-ws-abc")
    assert kind == kernel_sandbox.BOUNDARY_AUDIT
    assert installed, "no audit hook installed for a kernel in the base profile"


def test_a_kernel_in_child_counts_apparmor(monkeypatch, tmp_path):
    kind, installed = _establish(monkeypatch, tmp_path, "jaato-ws-abc//child")
    assert kind == kernel_sandbox.BOUNDARY_APPARMOR
    assert not installed


def test_the_audit_hook_refuses_the_write_that_unconfines(tmp_path):
    # No AppArmor here, so a real kernel runs on the audit tier: the tier a
    # kernel left in the base profile now falls back to.
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id,
                       "open('/proc/self/attr/current', 'w').write("
                       "'changeprofile unconfined')")
        assert r.status == ExecutionStatus.FAILED
        assert "containment" in (r.error_message or "")
    finally:
        be.shutdown()


def test_the_backend_reports_the_tier_the_kernel_will_have(monkeypatch, tmp_path):
    from jaato_server.shared.plugins.notebook.backends import subprocess_kernel
    monkeypatch.setattr(subprocess_kernel, "apparmor_enforced_profile",
                        lambda: "jaato-ws-abc")
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    assert be.boundary_kind() == kernel_sandbox.BOUNDARY_AUDIT
    be.set_apparmor_child_transition(lambda: None)
    assert be.boundary_kind() == kernel_sandbox.BOUNDARY_APPARMOR
    assert "//child" in be.execution_boundary()[1]


# ---- the in-process backend --------------------------------------------------

def test_the_local_backend_does_not_run_cells_because_apparmor_is_enforced(monkeypatch):
    monkeypatch.delenv(INPROCESS_OPT_IN_ENV, raising=False)
    monkeypatch.setattr(
        "jaato_server.shared.plugins.notebook.backends.local._apparmor_enforced_profile",
        lambda: "jaato-ws-abc")
    be = LocalJupyterBackend()
    be.initialize()
    nb = be.create_notebook("t").notebook_id
    r = be.execute(nb, "1 + 1")
    assert r.status == ExecutionStatus.FAILED
    assert r.error_name == "InProcessExecutionRefused"
    assert be.boundary_kind() == kernel_sandbox.BOUNDARY_NONE


# ---- the rendered profile ----------------------------------------------------

@pytest.fixture
def manager(tmp_path):
    from jaato_server.server.apparmor import AppArmorManager
    return AppArmorManager(
        workspace_root=str(tmp_path / "workspaces"),
        venv_path="/usr/local/venv",
        profile_dir=str(tmp_path / "profiles"),
    )


def _child(profile: str) -> str:
    i = profile.find("profile child {")
    assert i > 0, "no //child sub-profile in the rendered profile"
    return profile[i:]


def test_child_may_read_its_own_label_and_not_write_it(manager, tmp_path):
    for fragments in (None, []):
        child = _child(manager._render_profile(
            "s1", str(tmp_path / "workspaces" / "ws"),
            requested_fragments=fragments))
        assert re.search(r"^\s*owner /proc/\*/attr/current\s+r,", child, re.M)
        assert not re.search(r"^\s*[^#\n]*attr/current\s+[a-z]*w", child, re.M), (
            "//child must not be able to write attr/current")
        assert not re.search(r"^\s*change_profile\b", child, re.M)


def test_the_notebook_plugin_grants_exec_on_the_kernel_interpreter():
    from jaato_server.shared.plugins.notebook.plugin import create_plugin
    rules = create_plugin().get_apparmor_rules(
        workspace_path="/ws", session_id="s", config_root=None, plugin_config={})
    assert f'"{os.path.realpath(sys.executable)}" ix,' in rules


def test_the_rendered_profile_with_the_notebook_grant_compiles(manager, tmp_path):
    parser = shutil.which("apparmor_parser")
    if not parser:
        pytest.skip("apparmor_parser not installed")
    from jaato_server.shared.plugins.notebook.plugin import kernel_interpreter_apparmor_rules
    for fragments in (None, []):
        prof = tmp_path / f"candidate-{fragments is None}.aa"
        prof.write_text(manager._render_profile(
            "s1", str(tmp_path / "workspaces" / "ws"),
            requested_fragments=fragments,
            plugin_rules=kernel_interpreter_apparmor_rules()))
        res = subprocess.run([parser, "-Q", "-K", str(prof)],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
