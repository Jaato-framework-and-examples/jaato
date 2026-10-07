"""The conformance daemon must import the tree the tests came from (#1050).

``ConformanceDaemon`` spawns ``python -m server`` as a SEPARATE PROCESS, so it
does not inherit pytest's rootdir ``sys.path`` insertion.  With a bare
``os.environ`` it resolved ``server`` / ``shared`` / ``jaato_sdk`` through the
editable install -- one fixed checkout, whatever tree pytest is running from.

Run the suite in a ``git worktree`` and the two halves of one pytest
invocation execute two different trees.  That was found as a FAILURE, which
was luck: the shared checkout happened to lack a fix the branch had.  Reverse
it and the suite goes green about code it never exercised, with nothing in the
output naming which tree was loaded.

These tests are deliberately NOT marked ``conformance``.  They are about the
fixture, not about a daemon, and a guard that only runs under the opt-in
marker would not run in the leg where this regression would reappear.

NO ``REVERSIONS`` BLOCK, because the meta-guard cannot reach one here:
``_guard_modules`` walks ``shared/tests`` and ``server/tests`` only, so a
list declared in this package would be silently undiscovered -- decorative in
exactly the way that suite exists to prevent.  Importing it would also be the
first ``shared`` import in the SDK, which the package does not do.  Both
reversions were therefore checked by hand and are recorded here so the next
reader can repeat them:

* remove ``env=env`` from the ``Popen`` call ->
  ``test_the_spawn_is_given_an_env_naming_this_tree`` fails (verified);
* make ``tree_roots`` take the server tier from resolution instead of the
  anchor's checkout -> ``test_the_server_tier_comes_from_the_anchors_own
  _checkout`` fails (verified against the first draft, which did exactly
  that).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from jaato_sdk.conformance import daemon as dmod
from jaato_sdk.conformance.daemon import (
    ConformanceDaemon,
    DaemonTreeMismatch,
    TREE_MODULES,
    _package_root,
    _resolve,
    daemon_env,
    expected_origins,
    tree_pythonpath,
    tree_roots,
)

_REAL_FIND_SPEC = dmod.importlib.util.find_spec


def _pin_sdk_to(origin: Path) -> None:
    """Make ``_resolve("jaato_sdk")`` answer *origin*, leaving the rest real.

    A fake checkout on disk is the only way to exercise the layout branch
    without a second clone, and pinning ONLY the anchor keeps the fallback
    path answering honestly.
    """
    class _Spec:
        def __init__(self, origin):
            self.origin = str(origin)

    def _pinned(name, *args, **kwargs):
        if name == "jaato_sdk":
            return _Spec(origin)
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    dmod.importlib.util.find_spec = _pinned


def _unpin_sdk() -> None:
    dmod.importlib.util.find_spec = _REAL_FIND_SPEC


# --------------------------------------------------------------- path shapes

def test_a_package_contributes_its_parent_directory():
    """``<root>/pkg/__init__.py`` is importable from ``<root>``, not ``<root>/pkg``."""
    assert _package_root("/a/b/pkg/__init__.py") == str(Path("/a/b"))


def test_a_single_module_contributes_its_own_directory():
    """A non-package module has no grandparent to climb to."""
    assert _package_root("/a/b/lonely.py") == str(Path("/a/b"))


def test_the_anchor_is_the_checkout_jaato_sdk_came_from():
    """``jaato_sdk`` identifies the tree the test code came from, by construction."""
    roots = tree_roots()
    sdk_origin = _resolve("jaato_sdk")
    assert sdk_origin is not None
    assert roots["sdk"] == _package_root(sdk_origin)


def test_the_server_tier_comes_from_the_anchors_own_checkout(tmp_path):
    """THE CORRECTION.  Resolution alone reproduces the split it must repair.

    Measured in the configuration this was found in: pytest running the SDK
    leg from a worktree inserts ``<worktree>/jaato-sdk`` on ``sys.path`` and
    NOT ``<worktree>/jaato-server``, so ``jaato_sdk`` resolves to the
    worktree while ``server`` and ``shared`` still resolve through the
    editable install.  Taking the server tier from wherever it happens to
    resolve would carry that split into the daemon.
    """
    checkout = tmp_path / "a-worktree"
    for pkg in ("jaato-sdk/jaato_sdk", "jaato-server/jaato_server",
                "jaato-server/jaato_server/server",
                "jaato-server/jaato_server/shared"):
        (checkout / pkg).mkdir(parents=True)
        (checkout / pkg / "__init__.py").write_text("")

    fake_sdk = checkout / "jaato-sdk" / "jaato_sdk" / "__init__.py"
    _pin_sdk_to(fake_sdk)
    try:
        roots = tree_roots()
    finally:
        _unpin_sdk()

    assert roots["sdk"] == str(checkout / "jaato-sdk")
    assert roots["server"] == str(checkout / "jaato-server"), (
        "the server tier must come from the anchor's checkout, not from "
        "wherever `server` happens to resolve"
    )


def test_without_a_colocated_checkout_it_falls_back_to_resolution(tmp_path):
    """An installed tree has only one of everything, so resolution is right there."""
    lonely = tmp_path / "site-packages" / "jaato_sdk"
    lonely.mkdir(parents=True)
    (lonely / "__init__.py").write_text("")

    _pin_sdk_to(lonely / "__init__.py")
    try:
        roots = tree_roots()
    finally:
        _unpin_sdk()

    assert roots["sdk"] == str(tmp_path / "site-packages")
    # No <tmp>/jaato-server exists, so `server` falls back to this process's
    # own resolution rather than being dropped.
    assert roots.get("server") == _package_root(_resolve("jaato_server"))


def test_a_module_that_resolves_to_nothing_yields_none(monkeypatch):
    """Absence of evidence is not divergence.

    ``_resolve`` feeds a comparison that must only ever judge names BOTH
    sides answered for; a fabricated path would make an unusual layout look
    like a mismatch.
    """
    monkeypatch.setattr(dmod.importlib.util, "find_spec", lambda name: None)
    assert _resolve("jaato_sdk") is None


def test_find_spec_raising_does_not_break_collection(monkeypatch):
    """``find_spec`` imports parent packages, so it can raise anything."""
    def _boom(name):
        raise RuntimeError("a plugin exploded on import")
    monkeypatch.setattr(dmod.importlib.util, "find_spec", _boom)
    assert _resolve("shared") is None


def test_every_expected_origin_sits_under_a_root_on_the_path():
    """The child is judged against the tree chosen for it, so the two must agree."""
    path = tree_pythonpath()
    for name, origin in expected_origins().items():
        assert any(origin.startswith(root + os.sep) for root in path), name


# ----------------------------------------------------------------- the env

def test_the_env_carries_this_tree_on_pythonpath():
    env = daemon_env()
    parts = env["PYTHONPATH"].split(os.pathsep)
    for root in tree_pythonpath():
        assert root in parts


def test_an_operator_pythonpath_is_kept_but_ranks_below_ours():
    """Dropping it would trade this bug for the opposite one; so would losing."""
    env = daemon_env({"PYTHONPATH": "/operator/choice"})
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert "/operator/choice" in parts
    assert parts.index("/operator/choice") > parts.index(tree_pythonpath()[0])


def test_the_rest_of_the_environment_survives():
    env = daemon_env({"PYTHONPATH": "/x", "JAATO_SOMETHING": "keep me"})
    assert env["JAATO_SOMETHING"] == "keep me"


# ------------------------------------------------- the regression pin itself

class _FakeProc:
    returncode = None

    def poll(self):
        return None


def _no_daemon(monkeypatch, tmp_path, preflight):
    """Drive ``start()`` to the spawn without starting anything."""
    seen = {}

    def _fake_popen(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(dmod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(dmod, "_can_connect", lambda path: True)
    monkeypatch.setattr(ConformanceDaemon, "_preflight",
                        lambda self, env: preflight)
    monkeypatch.delenv("JAATO_CONFORMANCE_SOCKET", raising=False)
    return seen


def test_the_spawn_is_given_an_env_naming_this_tree(monkeypatch, tmp_path):
    """THE PIN.  On the pre-#1050 tree ``Popen`` got no ``env`` at all."""
    seen = _no_daemon(monkeypatch, tmp_path, preflight=expected_origins())
    ConformanceDaemon(tmp_path).start()

    env = seen["kwargs"].get("env")
    assert env is not None, "Popen was given no env — the child inherits os.environ"
    parts = env["PYTHONPATH"].split(os.pathsep)
    for root in tree_pythonpath():
        assert root in parts


def test_the_preflight_runs_in_the_daemons_own_cwd(monkeypatch, tmp_path):
    """``-c`` and ``-m`` both put the cwd on ``sys.path`` ahead of PYTHONPATH.

    Asking the question from anywhere else answers about a different search
    path than the daemon will have.
    """
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(dmod.subprocess, "run", _fake_run)
    d = ConformanceDaemon(tmp_path)
    d._out_handle = None
    assert d._preflight({"PYTHONPATH": "/x"}) is None
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"] == {"PYTHONPATH": "/x"}


# ------------------------------------------------------------- the refusal

def test_divergence_refuses_before_the_daemon_is_started(monkeypatch, tmp_path):
    mine = expected_origins()
    name = sorted(mine)[0]
    theirs = dict(mine)
    theirs[name] = "/somewhere/else/entirely/__init__.py"

    seen = _no_daemon(monkeypatch, tmp_path, preflight=theirs)
    with pytest.raises(DaemonTreeMismatch) as excinfo:
        ConformanceDaemon(tmp_path).start()

    assert "cmd" not in seen, "the daemon was started despite the mismatch"
    message = str(excinfo.value)
    assert name in message
    assert "/somewhere/else/entirely" in message
    assert mine[name] in message, "the message must name BOTH sides"


def test_an_unreadable_preflight_does_not_refuse(monkeypatch, tmp_path):
    """Only positive evidence counts — jaato #1023's rule, same reasoning.

    A resolution read successfully that names another file proves the daemon
    would run other code.  A preflight that could not run proves nothing, and
    failing closed on it would take down every suite on a machine whose only
    fault is being unusual.
    """
    seen = _no_daemon(monkeypatch, tmp_path, preflight=None)
    ConformanceDaemon(tmp_path).start()
    assert seen["kwargs"].get("env") is not None


def test_a_module_only_one_side_resolves_is_not_divergence(monkeypatch, tmp_path):
    seen = _no_daemon(monkeypatch, tmp_path,
                      preflight={name: None for name in TREE_MODULES})
    ConformanceDaemon(tmp_path).start()
    assert "cmd" in seen


def test_external_mode_is_exempt(monkeypatch, tmp_path):
    """The operator supplied the daemon and owns what is running in it."""
    called = []
    monkeypatch.setattr(ConformanceDaemon, "_verify_child_tree",
                        lambda self, env: called.append(env))
    monkeypatch.setattr(dmod, "_can_connect", lambda path: True)
    monkeypatch.setenv("JAATO_CONFORMANCE_SOCKET", "/tmp/operators.sock")

    ConformanceDaemon(tmp_path).start()
    assert called == []
