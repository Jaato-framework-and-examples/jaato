"""The doctor names client/daemon checkout skew, and grades it correctly.

WHAT WENT WRONG (#823).  A client and the daemon it talks to can resolve
``jaato_sdk`` from DIFFERENT checkouts — the daemon launched with a
``PYTHONPATH`` pointing at a feature branch, the client launched without one
and inheriting the venv's editable install.  Every surface reports healthy.
Then the daemon sends ``ToolOutputEvent.mime_type``, the client's event class
has nowhere to put it, pydantic drops it on ingest, and an event handler dies
with ``AttributeError`` several frames from anything the reader wrote.

WHY THESE ASSERT VERDICTS AND NOT OUTPUT.  "the doctor prints a line" is
decorative: it passes for a check that prints the same line whatever it found.
The whole value of #823's suggestion is the SEVERITY SPLIT — WARN when only
the version differs (a legitimate rolling upgrade), FAIL when the two resolved
paths are different working trees, which is never intentional in a dev loop —
so each case here builds a filesystem that makes exactly one of those true and
asserts the status.

The discriminator is measured, not declared: a working tree has the
distribution's ``pyproject.toml`` beside the package directory, an unpacked
install in ``site-packages`` does not.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys
from pathlib import Path

import pytest

from jaato_sdk import doctor as D
from jaato_server.shared.tests.reversion import Reversion

_DOCTOR_SRC = pathlib.Path(D.__file__)

REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/doctor.py",
        find="""    if _is_working_tree(client) or _is_working_tree(daemon):
        return Check(label, FAIL,""",
        replace="""    if False:
        return Check(label, FAIL,""",
        test="test_two_different_working_trees_FAIL",
        because="two working trees reported as a mere rolling upgrade",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/doctor.py",
        find="""    if os.path.realpath(client) == os.path.realpath(daemon):
        return Check(label, PASS,""",
        replace="""    if True:
        return Check(label, PASS,""",
        test="test_two_different_installed_copies_WARN",
        because="every comparison answering PASS, which is the pre-#823 silence",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/doctor.py",
        find="""    checks += check_home_match(info)
    checks += check_checkout_skew(info)""",
        replace="""    checks += check_home_match(info)""",
        test="test_the_preflight_run_actually_includes_the_skew_check",
        because="the check existing and never being run",
    ),
]


# --------------------------------------------------------------------------
# Building the two shapes on disk
# --------------------------------------------------------------------------

def _make_checkout(root, dist: str, pkg: str, version: str):
    """A source CHECKOUT: ``<root>/<dist>/pyproject.toml`` + ``<dist>/<pkg>/``."""
    tree = root / dist
    (tree / pkg).mkdir(parents=True)
    (tree / pkg / "__init__.py").write_text("", encoding="utf-8")
    (tree / "pyproject.toml").write_text(
        f'[project]\nname = "{dist}"\nversion = "{version}"\n', encoding="utf-8")
    return tree


def _make_install(root, dist: str, pkg: str, version: str):
    """An unpacked INSTALL: ``<root>/site-packages/{<pkg>, <dist>-<v>.dist-info}``."""
    site = root / "site-packages"
    (site / pkg).mkdir(parents=True)
    (site / pkg / "__init__.py").write_text("", encoding="utf-8")
    info = site / f"{dist}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n\nbody\n",
        encoding="utf-8")
    (info / "top_level.txt").write_text(f"{pkg}\n", encoding="utf-8")
    return site


def _daemon(env):
    """A listening daemon whose environ is ``env`` and whose exe is ours."""
    return D.DaemonInfo(socket_path="/tmp/x.sock", socket_exists=True,
                        listening=True, pid=1234, pid_alive=True,
                        env=env, executable=sys.executable)


@pytest.fixture
def one_package(monkeypatch):
    """Narrow the scan to one package so a case builds one tree, not two."""
    monkeypatch.setattr(D, "_SKEW_PACKAGES", ("jaato_sdk",))


def _statuses(checks):
    return {c.name: c.status for c in checks}


# --------------------------------------------------------------------------
# The three cases the issue names
# --------------------------------------------------------------------------

def test_same_tree_PASSes(tmp_path, monkeypatch, one_package):
    """Both sides resolve one directory — the state a dev loop should be in."""
    tree = _make_checkout(tmp_path, "jaato-sdk", "jaato_sdk", "0.19.0")
    monkeypatch.setattr(D, "_client_package_dir", lambda pkg: tree / pkg)
    checks = D.check_checkout_skew(_daemon({"PYTHONPATH": str(tree)}))
    assert _statuses(checks) == {"jaato_sdk checkout": D.PASS}, \
        [(c.name, c.status, c.detail) for c in checks]
    assert "0.19.0" in checks[0].detail, "the resolved version belongs in the line"


def test_two_different_working_trees_FAIL(tmp_path, monkeypatch, one_package):
    """#823's own reproduction: an editable install vs another branch.

    FAIL, not WARN — this is never intentional, and the cost of missing it is
    a silent field drop that reads as "media delivery is broken".
    """
    mine = _make_checkout(tmp_path / "a", "jaato-sdk", "jaato_sdk", "0.19.0")
    theirs = _make_checkout(tmp_path / "b", "jaato-sdk", "jaato_sdk", "0.19.0")
    monkeypatch.setattr(D, "_client_package_dir", lambda pkg: mine / pkg)
    checks = D.check_checkout_skew(_daemon({"PYTHONPATH": str(theirs)}))
    assert _statuses(checks) == {"jaato_sdk checkout": D.FAIL}, \
        [(c.name, c.status, c.detail) for c in checks]
    assert str(mine) in checks[0].detail and str(theirs) in checks[0].detail, \
        "a reader must be able to see WHICH two trees, not just that there are two"


def test_two_different_installed_copies_WARN(tmp_path, monkeypatch, one_package):
    """A version-only difference: two installs, no checkout — a rolling upgrade.

    WARN rather than FAIL because it is a legitimate operational state; the
    detail still names both versions, because it is also how an incompatible
    upgrade looks.
    """
    mine = _make_install(tmp_path / "a", "jaato-sdk", "jaato_sdk", "0.19.0")
    theirs = _make_install(tmp_path / "b", "jaato-sdk", "jaato_sdk", "0.18.0")
    monkeypatch.setattr(D, "_client_package_dir", lambda pkg: mine / pkg)
    checks = D.check_checkout_skew(_daemon({"PYTHONPATH": str(theirs)}))
    assert _statuses(checks) == {"jaato_sdk checkout": D.WARN}, \
        [(c.name, c.status, c.detail) for c in checks]
    assert "0.19.0" in checks[0].detail and "0.18.0" in checks[0].detail, \
        f"both versions belong in the line. Got: {checks[0].detail}"


def test_a_working_tree_against_an_install_is_still_a_FAIL(tmp_path, monkeypatch,
                                                           one_package):
    """One side a checkout is enough: they are not the same code."""
    mine = _make_checkout(tmp_path / "a", "jaato-sdk", "jaato_sdk", "0.19.0")
    theirs = _make_install(tmp_path / "b", "jaato-sdk", "jaato_sdk", "0.19.0")
    monkeypatch.setattr(D, "_client_package_dir", lambda pkg: mine / pkg)
    checks = D.check_checkout_skew(_daemon({"PYTHONPATH": str(theirs)}))
    assert _statuses(checks) == {"jaato_sdk checkout": D.FAIL}


# --------------------------------------------------------------------------
# The cases where the honest answer is "I cannot tell"
# --------------------------------------------------------------------------

def test_no_daemon_pythonpath_compares_against_the_installed_package(
        tmp_path, monkeypatch, one_package):
    """The issue's own rule: absent PYTHONPATH means the installed package."""
    site = _make_install(tmp_path, "jaato-sdk", "jaato_sdk", "0.19.0")
    monkeypatch.setattr(D, "_client_package_dir", lambda pkg: site / pkg)
    monkeypatch.setattr(D, "_installed_package_dir", lambda pkg: site / pkg)
    checks = D.check_checkout_skew(_daemon({}))
    assert _statuses(checks) == {"jaato_sdk checkout": D.PASS}
    assert "installed package" in checks[0].detail or str(site) in checks[0].detail


def test_neither_side_shadowing_is_the_straight_comparison(monkeypatch,
                                                           one_package):
    """When the CALLER is not shadowing either, its resolution IS the answer.

    Deriving the installed path a second way here could disagree with the
    client's own ``find_spec`` result and manufacture a skew where there is
    none — so that resolution is not attempted, and the source line says why.
    """
    monkeypatch.setattr(D, "_my_pythonpath", list)
    monkeypatch.setattr(D, "_installed_package_dir",
                        lambda pkg: pytest.fail("must not re-resolve"))
    path, source = D._daemon_package_dir("jaato_sdk", [], Path("/venv/jaato_sdk"))
    assert path == Path("/venv/jaato_sdk")
    assert "same one you resolve" in source


def test_a_pep660_editable_install_is_resolved_through_metadata(tmp_path,
                                                                monkeypatch):
    """The dominant dev-loop shape, and the one ``sys.path`` cannot answer.

    A PEP 660 editable install registers a META-PATH finder, not a path entry,
    so walking ``sys.path`` finds nothing for it — which would have made the
    check say "cannot tell" in precisely the environment #823 describes.
    ``direct_url.json`` records the working tree it points at.
    """
    import importlib.metadata as md

    tree = _make_checkout(tmp_path, "jaato-sdk", "jaato_sdk", "0.19.0")
    payload = ('{"url": "file://' + str(tree) +
               '", "dir_info": {"editable": true}}')

    class _Dist:
        def read_text(self, name):
            return payload if name == "direct_url.json" else None

    monkeypatch.setattr(md, "packages_distributions",
                        lambda: {"jaato_sdk": ["jaato-sdk"]})
    monkeypatch.setattr(md, "distribution", lambda name: _Dist())
    assert D._editable_source_dir("jaato_sdk") == tree / "jaato_sdk"


def test_a_non_editable_distribution_yields_no_source_dir(monkeypatch):
    """Only an EDITABLE install names a working tree; a wheel names none."""
    import importlib.metadata as md

    class _Dist:
        def read_text(self, name):
            return '{"url": "https://pypi/x.whl", "archive_info": {}}'

    monkeypatch.setattr(md, "packages_distributions",
                        lambda: {"jaato_sdk": ["jaato-sdk"]})
    monkeypatch.setattr(md, "distribution", lambda name: _Dist())
    assert D._editable_source_dir("jaato_sdk") is None


def test_unreadable_metadata_is_none_not_an_exception(monkeypatch):
    """This runs inside a diagnostic; metadata that explodes must not."""
    import importlib.metadata as md

    def boom():
        raise RuntimeError("no metadata here")

    monkeypatch.setattr(md, "packages_distributions", boom)
    assert D._editable_source_dir("jaato_sdk") is None


def test_a_daemon_on_another_interpreter_says_it_cannot_tell(
        tmp_path, monkeypatch, one_package):
    """The false PASS this must never produce.

    With no daemon PYTHONPATH, "the installed package" means installed for the
    DAEMON's interpreter.  Resolving the caller's own site-packages instead
    would compare a path against itself and answer PASS about two environments
    that were never compared.
    """
    site = _make_install(tmp_path, "jaato-sdk", "jaato_sdk", "0.19.0")
    monkeypatch.setattr(D, "_client_package_dir", lambda pkg: site / pkg)
    monkeypatch.setattr(D, "_installed_package_dir", lambda pkg: site / pkg)
    info = _daemon({})
    info.executable = str(tmp_path / "other-venv" / "bin" / "python")
    checks = D.check_checkout_skew(info)
    assert _statuses(checks) == {"jaato_sdk checkout": D.WARN}, \
        [(c.name, c.status, c.detail) for c in checks]
    assert "cannot resolve" in checks[0].detail


def test_no_daemon_is_no_finding():
    """Nothing observed is not a finding — the check contributes no rows."""
    down = D.DaemonInfo(socket_path="/tmp/x.sock", socket_exists=False,
                        listening=False)
    assert D.check_checkout_skew(down) == []


def test_an_unreadable_environ_warns_rather_than_guessing():
    """A host without /proc must say so, not silently skip or assert."""
    opaque = D.DaemonInfo(socket_path="/tmp/x.sock", socket_exists=True,
                          listening=True, pid=1, pid_alive=True, env=None)
    checks = D.check_checkout_skew(opaque)
    assert [c.status for c in checks] == [D.WARN]


def test_an_unresolvable_package_warns_and_does_not_raise(monkeypatch,
                                                          one_package):
    """A diagnostic that raises is worse than a vague one."""
    monkeypatch.setattr(D, "_client_package_dir", lambda pkg: None)
    monkeypatch.setattr(D, "_installed_package_dir", lambda pkg: None)
    checks = D.check_checkout_skew(_daemon({}))
    assert [c.status for c in checks] == [D.WARN]


def test_every_probe_survives_a_hostile_filesystem(tmp_path):
    """The helpers run inside an error path; none of them may raise."""
    missing = tmp_path / "nope" / "jaato_sdk"
    assert _version(missing) == ""
    assert D._is_working_tree(missing) is False
    assert D._resolve_package_on(["", "\0bad", str(tmp_path)], "jaato_sdk") is None
    assert D._proc_exe(-1) is None


def _version(pkgdir):
    return D._version_at(pkgdir)


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def test_the_preflight_run_actually_includes_the_skew_check():
    """A check nothing calls is a check nobody gets.

    An AST assertion over ``run_checks`` rather than a drive of it: calling
    ``run_checks`` runs every OTHER check too, and this claim is about one
    call site, not about the health of an unrelated probe on the machine
    running the suite.  The claim is exact — ``check_checkout_skew`` is
    invoked from the function the CLI's preflight path calls.
    """
    tree = ast.parse(_DOCTOR_SRC.read_text(encoding="utf-8"))
    run_checks = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "run_checks"),
                      None)
    assert run_checks is not None, "run_checks has been renamed; this guard is blind"
    called = {n.func.id for n in ast.walk(run_checks)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "check_checkout_skew" in called, (
        "run_checks does not call check_checkout_skew, so the checkout-skew "
        f"verdict never reaches anyone. It calls: {sorted(called)}"
    )


def test_a_FAIL_makes_the_doctor_exit_nonzero(capsys):
    """The severity split only means anything if FAIL is a gate."""
    rc = D._print([D.Check("jaato_sdk checkout", D.FAIL, "different trees")])
    assert rc == 1
    rc_warn = D._print([D.Check("jaato_sdk checkout", D.WARN, "rolling upgrade")])
    assert rc_warn == 0


def test_the_daemons_interpreter_is_probed_at_observation_time(monkeypatch):
    """``probe_daemon`` fills ``executable``; the check reads, never re-probes."""
    monkeypatch.setattr(D, "_socket_listening", lambda p: True)
    monkeypatch.setattr(D, "_read_pid", lambda p: os.getpid())
    monkeypatch.setattr(D, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(D, "_proc_environ", lambda pid: {"HOME": "/h"})
    info = D.probe_daemon("/tmp/x.sock", "/tmp/x.pid")
    assert info.executable, "the interpreter must be observed beside the environ"
