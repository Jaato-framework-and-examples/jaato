"""A tool-venv interpreter imports what the workspace installed, not the daemon's jaato (#1322).

The workspace tool-venv (#1274) used to carry a ``.pth`` listing the runner's
whole site-packages plus jaato's source roots.  It was there for the notebook
kernel, which is jaato's own module, but every interpreter started from the
venv got it, including the ones ``cli`` and ``interactive_shell`` start for
the model.  A session developing jaato then imported the daemon's installed
``jaato_server`` instead of the checkout it was editing:

- a plain ``python -m pytest`` in the checkout tested the daemon's copy;
- ``pip install -e <checkout>`` lost too, because an editable install's
  finder is consulted only after the normal path search, and the bridged
  directory is on that path;
- pip saw the daemon's packages as already installed.

Now the ``.pth`` bridges the runner's ``pip`` and nothing else (the venv is
created without pip), and the kernel gets jaato's import dirs in its own
launch argv.  Pinned here:

- a tool-venv interpreter, started with the environment ``cli`` gives it,
  cannot import ``jaato_server`` or the runner's third-party packages;
- an editable install into the tool-venv resolves to the checkout;
- ``pip`` works there and does not report the daemon's packages;
- pip installs into the venv, offline, from a wheel;
- the notebook kernel still starts under the venv, and a process a cell
  starts does not inherit jaato's import dirs;
- a runner installed into a BASE interpreter (a container's system Python,
  a CI image) does not hand its install over through
  ``--system-site-packages`` either: the flag is not passed, and an existing
  venv has it switched off.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
import zipfile

import pytest

from jaato_server.shared.plugins.workspace_venv import (
    apply_venv_to_env,
    ensure_workspace_venv,
    runner_site_dirs,
    venv_python,
    venv_site_packages,
)
from jaato_server.shared.tests.reversion import Reversion

_VENV = "jaato-server/jaato_server/shared/plugins/workspace_venv.py"
_KERNEL = "jaato-server/jaato_server/shared/plugins/notebook/backends/subprocess_kernel.py"

REVERSIONS = [
    Reversion(
        target=_VENV,
        find='        system_site = [] if leaks_runner else ["--system-site-packages"]\n',
        replace='        system_site = ["--system-site-packages"]\n',
        test="test_a_base_interpreter_runner_creates_no_system_site_venv",
        because="a runner in a base interpreter hands its install over via the system site",
    ),
    Reversion(
        target=_VENV,
        find="    elif leaks_runner:\n        _exclude_system_site_packages(venv_path)\n",
        replace="",
        test="test_an_existing_venv_stops_seeing_a_base_runners_packages",
        because="a venv created before #1322 keeps seeing the runner's install",
    ),
    Reversion(
        target=_VENV,
        find="            lines.append(bridge_dir)\n",
        replace="            lines.extend([bridge_dir, *kernel_import_dirs()])\n",
        test="test_an_editable_install_resolves_to_the_checkout",
        because="the bridged runner site-packages shadows the workspace's editable install",
    ),
    Reversion(
        target=_VENV,
        find="            lines.append(bridge_dir)\n",
        replace="            lines.extend([bridge_dir, *kernel_import_dirs()])\n",
        test="test_pip_does_not_report_the_daemons_packages",
        because="pip in the tool-venv sees the daemon's packages as installed",
    ),
    Reversion(
        target=_KERNEL,
        find="            import_dirs = kernel_import_dirs()\n",
        replace="            import_dirs = []\n",
        test="test_the_notebook_kernel_still_runs_under_the_venv",
        because="a venv kernel with no import dirs cannot import its own module",
    ),
    Reversion(
        target=_KERNEL,
        find="            import_dirs = kernel_import_dirs()\n",
        replace=(
            "            import_dirs = []\n"
            "            kernel_env[\"PYTHONPATH\"] = os.pathsep.join(\n"
            "                [kernel_env.get(\"PYTHONPATH\", \"\"), *kernel_import_dirs()])\n"
        ),
        test="test_a_process_a_cell_starts_does_not_inherit_the_kernel_bridge",
        because="dirs on the kernel's PYTHONPATH reach every process a cell starts",
    ),
]


@pytest.fixture(scope="module")
def venv(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("ws") / "tool-venv")
    ensure_workspace_venv(path)
    return path


def _run(venv_path, *argv, cwd):
    """Run a command the way ``cli`` runs one: with the venv activated."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    apply_venv_to_env(env, venv_path)
    return subprocess.run(list(argv), capture_output=True, text=True,
                          cwd=cwd, env=env)


def test_a_tool_venv_interpreter_cannot_import_the_daemons_jaato(venv, tmp_path):
    r = _run(venv, venv_python(venv), "-c", "import jaato_server", cwd=tmp_path)
    assert r.returncode != 0, (
        "the tool-venv imported jaato_server with no install of its own:\n"
        + r.stdout)
    assert "No module named 'jaato_server'" in r.stderr


def test_the_runners_third_party_packages_are_not_bridged(venv, tmp_path):
    # pydantic is in the runner's site-packages; only pip may come across.
    r = _run(venv, venv_python(venv), "-c",
             "import sys; print('\\n'.join(sys.path))", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    on_path = set(r.stdout.split())
    assert not on_path & set(runner_site_dirs())


def test_an_editable_install_resolves_to_the_checkout(venv, tmp_path):
    """The finder shape setuptools' editable install writes (#1322's case 2).

    A meta-path finder appended to ``sys.meta_path`` is consulted AFTER the
    normal path search, so any directory on ``sys.path`` holding a
    ``jaato_server`` wins over it.
    """
    checkout = tmp_path / "checkout"
    (checkout / "jaato_server").mkdir(parents=True)
    (checkout / "jaato_server" / "__init__.py").write_text("WHERE = 'checkout'\n")
    site = venv_site_packages(venv)
    finder = os.path.join(site, "__editable___fake_jaato_finder.py")
    pth = os.path.join(site, "__editable__.fake_jaato.pth")
    with open(finder, "w") as f:
        f.write(textwrap.dedent(f"""
            import importlib.machinery, sys
            class _Finder:
                @classmethod
                def find_spec(cls, name, path=None, target=None):
                    if name != "jaato_server":
                        return None
                    return importlib.machinery.PathFinder.find_spec(
                        name, [{str(checkout)!r}])
            def install():
                sys.meta_path.append(_Finder)
        """))
    with open(pth, "w") as f:
        f.write("import __editable___fake_jaato_finder; "
                "__editable___fake_jaato_finder.install()\n")
    try:
        r = _run(venv, venv_python(venv), "-c",
                 "import jaato_server; print(jaato_server.__file__)",
                 cwd=tmp_path)
    finally:
        os.unlink(pth)
        os.unlink(finder)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().startswith(str(checkout)), (
        "an editable install into the tool-venv lost to another copy:\n"
        + r.stdout)


def test_pip_works_in_the_tool_venv(venv, tmp_path):
    r = _run(venv, os.path.join(venv, "bin", "pip"), "--version", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("pip ")


def test_pip_does_not_report_the_daemons_packages(venv, tmp_path):
    r = _run(venv, os.path.join(venv, "bin", "pip"), "show", "jaato-server",
             cwd=tmp_path)
    assert r.returncode != 0, (
        "pip in the tool-venv reports the daemon's jaato-server as installed:\n"
        + r.stdout)


def _wheel(tmp_path):
    name = "jaato_probe_1322"
    path = tmp_path / f"{name}-0.1-py3-none-any.whl"
    info = f"{name}-0.1.dist-info"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}/__init__.py", "OK = True\n")
        z.writestr(f"{info}/METADATA",
                   f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1\n")
        z.writestr(f"{info}/WHEEL",
                   "Wheel-Version: 1.0\nGenerator: test\n"
                   "Root-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(f"{info}/RECORD",
                   f"{name}/__init__.py,,\n{info}/METADATA,,\n"
                   f"{info}/WHEEL,,\n{info}/RECORD,,\n")
    return name, str(path)


def test_pip_installs_into_the_tool_venv(venv, tmp_path):
    name, wheel = _wheel(tmp_path)
    r = _run(venv, os.path.join(venv, "bin", "pip"), "install", "--no-index",
             "--no-deps", "--disable-pip-version-check", wheel, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    r = _run(venv, venv_python(venv), "-c",
             f"import {name}; print({name}.__file__)", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().startswith(venv_site_packages(venv))


def _kernel_backend(workspace):
    from jaato_server.shared.plugins.notebook.backends.subprocess_kernel import (
        SubprocessKernelBackend,
    )
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(workspace),
                   "workspace_venv": ".jaato/tool-venv"})
    return be


def _text(result):
    from jaato_server.shared.plugins.notebook.types import OutputType
    return "".join(o.content for o in result.outputs
                   if o.output_type in (OutputType.STDOUT, OutputType.RESULT))


def test_the_notebook_kernel_still_runs_under_the_venv(tmp_path):
    from jaato_server.shared.plugins.notebook.types import ExecutionStatus
    be = _kernel_backend(tmp_path)
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, "import sys\nprint(sys.prefix)")
        assert r.status == ExecutionStatus.COMPLETED, r.error_message
        assert _text(r).strip() == os.path.realpath(
            str(tmp_path / ".jaato" / "tool-venv"))
    finally:
        be.shutdown()


def test_a_process_a_cell_starts_does_not_inherit_the_kernel_bridge(tmp_path):
    from jaato_server.shared.plugins.notebook.types import ExecutionStatus
    be = _kernel_backend(tmp_path)
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, textwrap.dedent("""
            import subprocess, sys
            r = subprocess.run([sys.executable, "-c", "import jaato_server"],
                               capture_output=True)
            print("child-import-rc", r.returncode)
        """))
        assert r.status == ExecutionStatus.COMPLETED, r.error_message
        out = _text(r)
        assert "child-import-rc" in out, out
        assert "child-import-rc 0" not in out, (
            "a process started from a cell imported jaato_server through the "
            "kernel's bridge")
    finally:
        be.shutdown()


# ---- a runner installed into a base interpreter --------------------------------

def _created_with(monkeypatch, tmp_path, leaks_runner):
    from jaato_server.shared.plugins import workspace_venv as wv
    monkeypatch.setattr(wv, "_system_site_packages_are_the_runners",
                        lambda base_python=None: leaks_runner)
    argv = []
    real_run = subprocess.run

    def run(cmd, *a, **k):
        if "-m" in cmd and "venv" in cmd:
            argv.extend(cmd)
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(wv.subprocess, "run", run)
    venv = str(tmp_path / "tool-venv")
    ensure_workspace_venv(venv)
    return argv, venv


def test_a_base_interpreter_runner_creates_no_system_site_venv(monkeypatch, tmp_path):
    argv, venv = _created_with(monkeypatch, tmp_path, leaks_runner=True)
    assert argv and "--system-site-packages" not in argv
    cfg = open(os.path.join(venv, "pyvenv.cfg")).read()
    assert "include-system-site-packages = false" in cfg


def test_a_venv_runner_keeps_the_system_site(monkeypatch, tmp_path):
    # A runner venv's system site is the OS interpreter's, not the runner's.
    argv, _ = _created_with(monkeypatch, tmp_path, leaks_runner=False)
    assert "--system-site-packages" in argv


def _cfg_with(venv, value):
    os.makedirs(os.path.join(venv, "lib", "python3.99", "site-packages"))
    with open(os.path.join(venv, "pyvenv.cfg"), "w") as f:
        f.write(f"home = /x\ninclude-system-site-packages = {value}\nversion = 3.99\n")


def test_an_existing_venv_stops_seeing_a_base_runners_packages(monkeypatch, tmp_path):
    from jaato_server.shared.plugins import workspace_venv as wv
    monkeypatch.setattr(wv, "_system_site_packages_are_the_runners",
                        lambda base_python=None: True)
    venv = str(tmp_path / "old-venv")
    _cfg_with(venv, "true")
    ensure_workspace_venv(venv)
    cfg = open(os.path.join(venv, "pyvenv.cfg")).read()
    assert "include-system-site-packages = false" in cfg
    assert "home = /x" in cfg and "version = 3.99" in cfg


def test_an_existing_venv_is_never_switched_on(monkeypatch, tmp_path):
    from jaato_server.shared.plugins import workspace_venv as wv
    monkeypatch.setattr(wv, "_system_site_packages_are_the_runners",
                        lambda base_python=None: False)
    venv = str(tmp_path / "old-venv")
    _cfg_with(venv, "false")
    ensure_workspace_venv(venv)
    assert "include-system-site-packages = false" in open(
        os.path.join(venv, "pyvenv.cfg")).read()
