"""A notebook discloses that in-process cells import the daemon's jaato (#1413).

A session developing jaato inside jaato ran ``pip install -e ./jaato-server``
against its checkout, got rc=0, and its in-cell tests went on exercising the
DAEMON's code: the kernel is launched with the daemon's jaato import dirs
(``_kernel_argv``, kept by #1322), so ``import jaato_server`` and
``pytest.main`` in a cell never see the checkout.  That stays as it is; what
changes is that the model is told, on every surface it already reads, and
only when the workspace holds jaato's own source.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from jaato_server.shared import jaato_self_shadowing as shadow_mod
from jaato_server.shared.apparmor_label import parse_label
from jaato_server.shared.plugins.environment import runtime as runtime_mod
from jaato_server.shared.plugins.notebook import kernel_sandbox
from jaato_server.shared.tests.reversion import Reversion

_HELPER = "jaato-server/jaato_server/shared/jaato_self_shadowing.py"
_NOTEBOOK = "jaato-server/jaato_server/shared/plugins/notebook/plugin.py"
_RUNTIME = "jaato-server/jaato_server/shared/plugins/environment/runtime.py"

_MARK = "import the daemon's jaato"

REVERSIONS = [
    Reversion(
        target=_NOTEBOOK,
        find="            lines = tuple(lines) + (shadow,)\n",
        replace="            lines = tuple(lines)\n",
        test="test_the_standing_notice_discloses_it",
        because="the system-prompt boundary notice would not mention the shadowing",
    ),
    Reversion(
        target=_NOTEBOOK,
        find="            notes.append(shadow)\n",
        replace="            pass\n",
        test="test_the_first_result_of_a_kernel_discloses_it",
        because="the per-kernel announcement would not mention the shadowing",
    ),
    Reversion(
        target=_RUNTIME,
        find='        report["notebook"] = shadow\n',
        replace="        pass\n",
        test="test_the_runtime_aspect_discloses_it",
        because="get_environment(aspect='runtime') would not report it",
    ),
    Reversion(
        target=_HELPER,
        find="                    continue  # the daemon runs this very checkout\n",
        replace="                    pass\n",
        test="test_a_checkout_the_daemon_runs_from_is_not_reported",
        because="a daemon running from the workspace itself would be reported as shadowed",
    ),
]


def _jaato_checkout(root):
    pkg = root / "jaato-server" / "jaato_server"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    return pkg


def _notebook(workspace):
    from jaato_server.shared.plugins.notebook.plugin import create_plugin
    plugin = create_plugin()
    plugin.initialize({"workspace_root": str(workspace)})
    return plugin


def _runtime(workspace):
    return runtime_mod.runtime_report(
        None, None, str(workspace),
        read_label=lambda: parse_label("unconfined"), grants=lambda: None)


# -- the helper --------------------------------------------------------------


def test_a_checkout_at_the_root_or_in_a_clone_is_reported(tmp_path):
    assert shadow_mod.workspace_jaato_shadowing(str(tmp_path)) is None
    pkg = _jaato_checkout(tmp_path / "jaato")
    report = shadow_mod.workspace_jaato_shadowing(str(tmp_path))
    assert report["workspace"] == {"jaato_server": [str(pkg.resolve())]}
    assert "!python -m pytest" in report["note"]


def test_an_editable_install_in_the_workspace_venv_is_reported(tmp_path):
    src = tmp_path / "elsewhere"
    src.mkdir()
    dist = (tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
            / "jaato_sdk-0.1.dist-info")
    dist.mkdir(parents=True)
    (dist / "direct_url.json").write_text(json.dumps(
        {"url": f"file://{src}", "dir_info": {"editable": True}}))
    report = shadow_mod.workspace_jaato_shadowing(str(tmp_path))
    assert report["workspace"] == {"jaato_sdk": [str(src.resolve())]}


def test_a_checkout_the_daemon_runs_from_is_not_reported(tmp_path):
    pkg = _jaato_checkout(tmp_path)
    assert shadow_mod.workspace_jaato_shadowing(
        str(tmp_path), daemon_dirs={"jaato_server": str(pkg)}) is None


# -- the surfaces ------------------------------------------------------------


def test_the_standing_notice_discloses_it(tmp_path):
    assert _MARK not in _notebook(tmp_path)._boundary_instruction_block()
    _jaato_checkout(tmp_path)
    block = _notebook(tmp_path)._boundary_instruction_block()
    assert block and _MARK in block


def test_the_first_result_of_a_kernel_discloses_it(tmp_path):
    first = SimpleNamespace(boundary_kind=kernel_sandbox.BOUNDARY_AUDIT)
    notes = _notebook(tmp_path)._boundary_announcement(first)[
        "execution_boundary"]["notes"]
    assert not any(_MARK in n for n in notes)
    _jaato_checkout(tmp_path)
    notes = _notebook(tmp_path)._boundary_announcement(first)[
        "execution_boundary"]["notes"]
    assert any(_MARK in n for n in notes)


def test_the_runtime_aspect_discloses_it(tmp_path):
    assert "notebook" not in _runtime(tmp_path)
    assert "notebook" not in runtime_mod.runtime_summary(_runtime(tmp_path))
    _jaato_checkout(tmp_path)
    report = _runtime(tmp_path)
    block = report["notebook"]
    assert block["imports_daemon_jaato"] is True
    assert "jaato_server" in block["workspace_paths"]
    assert _MARK in runtime_mod.runtime_summary(report)["notebook"]
