"""The notebook kernel imports jaato's dependencies from the runner, not the OS (#1457).

A tool-venv made from a runner VENV is created ``--system-site-packages``, so
its ``sys.path`` carries the OS interpreter's site entries after its own
site-packages: on Debian/Ubuntu ``/usr/lib/python3/dist-packages`` (and the
user site), holding OS copies of jaato's dependencies.  The kernel's ``-c``
bootstrap used to APPEND the runner's import dirs, so those copies won, and an
old ``typing_extensions`` with no ``Sentinel`` killed the kernel before its
first cell.  CI's interpreter has no such ``dist-packages``, so the #1322
tests could not see it.

These tests build the shadowing on any host: a directory that reaches the
venv's ``sys.path`` after its own site-packages (a ``.pth``, the way a system
or user site entry arrives) holding a stub ``pydantic`` with nothing in it.
Pinned:

- the kernel starts past the stub and runs a cell;
- a package the model installed into the tool-venv still wins over the
  runner's copy inside a cell;
- with no workspace HOME, the daemon account's user site is not on the
  kernel's path;
- with a workspace HOME, that HOME's user site is (it is the workspace's own).
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

from jaato_server.shared.plugins.workspace_venv import (
    ensure_workspace_venv,
    venv_site_packages,
)
from jaato_server.shared.tests.reversion import Reversion

_KERNEL = "jaato-server/jaato_server/shared/plugins/notebook/backends/subprocess_kernel.py"

REVERSIONS = [
    Reversion(
        target=_KERNEL,
        find="        + f\"sys.path[i:i] = {list(import_dirs)!r}; \"\n",
        replace="        + f\"sys.path.extend({list(import_dirs)!r}); \"\n",
        test="test_the_kernel_starts_past_a_shadowing_site_entry",
        because="appended runner dirs lose to the OS copies of jaato's dependencies",
    ),
    Reversion(
        target=_KERNEL,
        find="        + f\"sys.path[i:i] = {list(import_dirs)!r}; \"\n",
        replace="        + f\"sys.path[0:0] = {list(import_dirs)!r}; \"\n",
        test="test_a_package_installed_into_the_venv_still_wins",
        because="prepended runner dirs shadow what the model installed into the venv",
    ),
    Reversion(
        target=_KERNEL,
        find="    flags = [] if user_site_is_workspaces else [\"-s\"]\n",
        replace="    flags = []\n",
        test="test_the_daemon_accounts_user_site_is_not_the_kernels",
        because="the kernel reads the daemon account's ~/.local packages",
    ),
    Reversion(
        target=_KERNEL,
        find="                           user_site_is_workspaces=bool(home_path)),\n",
        replace="                           user_site_is_workspaces=False),\n",
        test="test_a_workspace_home_user_site_stays_visible",
        because="the workspace's own user site disappears from the kernel",
    ),
]

_VENV_REL = os.path.join(".jaato", "tool-venv")


def _stub_pydantic(where):
    """A ``pydantic`` that imports and has nothing in it, as an old OS copy might."""
    os.makedirs(os.path.join(where, "pydantic"), exist_ok=True)
    with open(os.path.join(where, "pydantic", "__init__.py"), "w") as f:
        f.write("STUB = True\n")


def _user_site(home):
    v = sys.version_info
    return os.path.join(home, ".local", "lib", f"python{v.major}.{v.minor}",
                        "site-packages")


def _backend(workspace, **extra):
    from jaato_server.shared.plugins.notebook.backends.subprocess_kernel import (
        SubprocessKernelBackend,
    )
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(workspace),
                   "workspace_venv": _VENV_REL, **extra})
    return be


def _run_cell(be, code):
    from jaato_server.shared.plugins.notebook.types import (
        ExecutionStatus,
        OutputType,
    )
    nb = be.create_notebook("t")
    r = be.execute(nb.notebook_id, textwrap.dedent(code))
    assert r.status == ExecutionStatus.COMPLETED, r.error_message
    return "".join(o.content for o in r.outputs
                   if o.output_type in (OutputType.STDOUT, OutputType.RESULT))


@pytest.fixture
def system_site_venv(monkeypatch):
    """The venv a runner VENV makes: ``--system-site-packages``, user site on.

    Forced on a base-interpreter runner too (CI), so the user-site cases are
    exercised wherever the suite runs.
    """
    from jaato_server.shared.plugins import workspace_venv as wv
    monkeypatch.setattr(wv, "_system_site_packages_are_the_runners",
                        lambda base_python=None: False)
    monkeypatch.delenv("PYTHONUSERBASE", raising=False)
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)


def test_the_kernel_starts_past_a_shadowing_site_entry(tmp_path):
    venv = str(tmp_path / _VENV_REL)
    ensure_workspace_venv(venv)
    shadow = tmp_path / "os-site"
    _stub_pydantic(str(shadow))
    with open(os.path.join(venv_site_packages(venv), "zz_os_site.pth"), "w") as f:
        f.write(f"{shadow}\n")
    be = _backend(tmp_path)
    try:
        out = _run_cell(be, """
            import pydantic
            print(getattr(pydantic, "STUB", False))
        """)
    finally:
        be.shutdown()
    assert out.strip() == "False", "the kernel imported the shadowing pydantic"


def test_a_package_installed_into_the_venv_still_wins(tmp_path):
    venv = str(tmp_path / _VENV_REL)
    ensure_workspace_venv(venv)
    # python-dotenv is in the runner (a jaato-sdk dependency); the kernel
    # does not import it to start, so the venv's copy is what a cell gets.
    pkg = os.path.join(venv_site_packages(venv), "dotenv")
    os.makedirs(pkg)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("WHERE = 'venv'\n")
    be = _backend(tmp_path)
    try:
        out = _run_cell(be, """
            import dotenv
            print(getattr(dotenv, "WHERE", "runner"))
        """)
    finally:
        be.shutdown()
    assert out.strip() == "venv"


def test_the_daemon_accounts_user_site_is_not_the_kernels(
        tmp_path, monkeypatch, system_site_venv):
    home = tmp_path / "daemon-home"
    user_site = _user_site(str(home))
    _stub_pydantic(user_site)
    monkeypatch.setenv("HOME", str(home))
    ws = tmp_path / "ws"
    ws.mkdir()
    be = _backend(ws)
    try:
        out = _run_cell(be, f"""
            import sys
            print({user_site!r} in sys.path)
        """)
    finally:
        be.shutdown()
    assert out.strip() == "False", "the kernel reads the daemon account's user site"


def test_a_workspace_home_user_site_stays_visible(
        tmp_path, monkeypatch, system_site_venv):
    monkeypatch.setenv("HOME", str(tmp_path / "daemon-home"))
    ws = tmp_path / "ws"
    home = ws / ".home"
    probe_dir = _user_site(str(home))
    os.makedirs(probe_dir)
    with open(os.path.join(probe_dir, "jaato_probe_1457.py"), "w") as f:
        f.write("OK = True\n")
    be = _backend(ws, workspace_home=".home")
    try:
        out = _run_cell(be, """
            import jaato_probe_1457
            print(jaato_probe_1457.OK)
        """)
    finally:
        be.shutdown()
    assert out.strip() == "True"
