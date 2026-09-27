"""The notebook's prompt notice names the tier its kernel will get (#1357).

On a confined runner the notebook's system-prompt notice (#1012) described
the AUDIT tier ("`import ctypes` is REFUSED"), and the kernel then started
in ``//child`` and reported the APPARMOR tier.  The runner rendered the
prompt while building the session, and installed the ``//child`` transition
only afterwards, so the backend's ``boundary_kind()`` answered without it.
The runner now hands the transition to the registry's plugins before the
session is built.

CI has no AppArmor kernel; the runner's own label is stubbed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from jaato_server.server.runner import session as runner_session
from jaato_server.shared.plugins.notebook import kernel_sandbox
from jaato_server.shared.plugins.notebook.backends import subprocess_kernel
from jaato_server.shared.tests.reversion import Reversion

_RS = "jaato-server/jaato_server/server/runner/session.py"

REVERSIONS = [
    Reversion(
        target=_RS,
        find="    child_cb = _prearm_child_callback(envelope, runtime)\n",
        replace="    child_cb = None\n",
        test="test_bootstrap_arms_the_plugins_before_building_the_session",
        because="the prompt is rendered before the notebook knows about //child",
    ),
    Reversion(
        target=_RS,
        find="                plugin.set_apparmor_child_transition_callback(child_cb)\n        return child_cb\n",
        replace="                pass\n        return child_cb\n",
        test="test_the_pre_arm_reaches_the_notebook_and_its_notice",
        because="the pre-arm builds the callback and hands it to no plugin",
    ),
]


class _Registry:
    def __init__(self, plugins):
        self._plugins = plugins

    def list_exposed(self):
        return list(self._plugins)

    def get_plugin(self, name):
        return self._plugins.get(name)


def _notebook(tmp_path):
    from jaato_server.shared.plugins.notebook.plugin import create_plugin
    plugin = create_plugin()
    plugin.initialize({"workspace_root": str(tmp_path)})
    return plugin


def _confined_runner(monkeypatch):
    # The runner wears its base profile, which can leave itself (#1323).
    monkeypatch.setattr(subprocess_kernel, "apparmor_enforced_profile",
                        lambda: "jaato-ws-abc")


def test_the_pre_arm_reaches_the_notebook_and_its_notice(monkeypatch, tmp_path):
    _confined_runner(monkeypatch)
    plugin = _notebook(tmp_path)
    audit = kernel_sandbox.boundary_notice(kernel_sandbox.BOUNDARY_AUDIT)[0]
    assert audit in plugin._boundary_instruction_block()

    runtime = SimpleNamespace(_registry=_Registry({"notebook": plugin}))
    envelope = SimpleNamespace(profile_name="jaato-ws-abc")
    cb = runner_session._prearm_child_callback(envelope, runtime)

    assert cb is not None
    assert plugin._backends["subprocess"]._apparmor_child_transition is cb
    block = plugin._boundary_instruction_block()
    assert kernel_sandbox.boundary_notice(kernel_sandbox.BOUNDARY_APPARMOR)[0] in block
    assert audit not in block


def test_no_pre_arm_for_an_unconfined_or_sub_runner(tmp_path):
    plugin = _notebook(tmp_path)
    runtime = SimpleNamespace(_registry=_Registry({"notebook": plugin}))
    for name in ("", "jaato-ws-abc//subagent"):
        assert runner_session._prearm_child_callback(
            SimpleNamespace(profile_name=name), runtime) is None
    assert plugin._backends["subprocess"]._apparmor_child_transition is None


def test_bootstrap_arms_the_plugins_before_building_the_session():
    tree = ast.parse(Path(runner_session.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "bootstrap_session")
    lines = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            lines.setdefault(node.func.id, node.lineno)
    assert "_prearm_child_callback" in lines, "the pre-arm is not called"
    assert lines["_prearm_child_callback"] < lines["_build_session"]


def test_step_four_installs_the_same_callable(tmp_path):
    installed = []
    executor = SimpleNamespace(
        set_apparmor_child_transition_callback=installed.append)
    session = SimpleNamespace(_executor=executor)

    def cb():
        pass

    runner_session._maybe_install_child_callback(
        SimpleNamespace(profile_name="jaato-ws-abc"), session, cb)
    assert installed == [cb]
