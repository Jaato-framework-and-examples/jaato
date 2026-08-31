"""Every guard must fail when the thing it guards is put back.

WHY THIS EXISTS.  A guard that cannot notice its own reversion is not
evidence, and nothing in this repository checked that until now.  The
discipline was manual: break the code, watch the guard go red, restore.
Done once, by whoever wrote it, and then never again -- so a guard that
stopped discriminating went on reporting green, and the only way anyone
found out was by accident.

That is not hypothetical.  Three guards passed under sabotage this month
before the sabotage was run by hand, and three separate CI defects shipped
in one day -- a ratchet that skipped silently when it could not find the
tree, one that measured build artifacts so it was green only on CI, and
102 baselines that had drifted for months in a directory no workflow ran.
Every one of them reported success while exercising nothing.

THE MODEL IS BORROWED from the pattern corpus's ``certify/``, which had it
first: a claim carries the one-line change that SHOULD break it, and a
selftest layer applies that change and asserts the claim notices.  What was
a habit becomes an artifact that runs on every commit.

THREE-VALUED, AND THE THIRD VALUE IS THE POINT::

    PASS     the reversion was applied and the guard FAILED.  Working.
    FAIL     the reversion was applied and the guard PASSED anyway.
             The guard is decorative.
    BLOCKED  the reversion could not be applied -- its anchor is no longer
             in the source.  NOT a pass: nothing was exercised, and the
             guard's status is unknown.

``BLOCKED`` is the state this repository keeps shipping by accident.  A
suite that renders it as green is the failure it exists to prevent.

DECLARING A REVERSION.  A guard module exposes::

    REVERSIONS = [
        Reversion(
            target="jaato-server/server/session_manager.py",
            find="...exact source the fix introduced...",
            replace="...what it looked like before...",
            because="what the guard should notice",
        ),
    ]

``find`` must be the FIXED text and ``replace`` the BROKEN text: this puts
the code back the way it was, which is what "reversion" means and what the
guard was written against.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[3]

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"


@dataclass(frozen=True)
class Reversion:
    """One way to put a guarded defect back."""
    target: str      #: repo-relative file to edit
    find: str        #: the FIXED text (must be present, exactly once)
    replace: str     #: the BROKEN text it becomes
    because: str     #: what the guard is supposed to notice
    test: str        #: the ONE test that must fail. See below.
    #
    # NAMING THE TEST IS NOT PEDANTRY.  The first version of this asserted
    # only that the MODULE failed, and a module is many tests: neutering
    # one assertion was masked by a sibling in the same file, and the
    # meta-guard reported the decorative guard as working.  That is the
    # over-broad match -- the anchor matched something, just not the thing
    # under test.  Caught only by sabotaging this suite itself.


#: Guard modules live in two packages; a reversion names the module, not
#: the path, so the suite resolves it rather than each declaration
#: repeating a directory that could drift.
_PACKAGES = ("jaato-server/shared/tests", "jaato-server/server/tests")


def _module_path(module_name: str) -> str:
    for pkg in _PACKAGES:
        cand = ROOT / pkg / f"{module_name}.py"
        if cand.is_file():
            return f"{pkg}/{module_name}.py"
    raise AssertionError(
        f"cannot locate {module_name}.py under {_PACKAGES}. A guard that "
        f"cannot be found cannot be run, and an unrun guard must not read "
        f"as a working one."
    )


def _guard_modules() -> List[Tuple[str, object]]:
    """Every test module in this package that declares REVERSIONS."""
    found = []
    dirs = [str(ROOT / p) for p in _PACKAGES]
    for mod in pkgutil.iter_modules(dirs):
        if not mod.name.startswith("test_"):
            continue
        if mod.name == Path(__file__).stem:
            continue
        try:
            pkg = ("shared.tests" if (ROOT / _PACKAGES[0] /
                                      f"{mod.name}.py").is_file()
                   else "server.tests")
            m = importlib.import_module(f"{pkg}.{mod.name}")
        except Exception:
            # A module that will not import is a problem for its OWN test
            # run, which will say so far more usefully than a name here.
            continue
        if getattr(m, "REVERSIONS", None):
            found.append((mod.name, m))
    return sorted(found)


def _clear_pycache(root: Path) -> None:
    """Bytecode caching invalidates on (size, mtime), not content.

    A reversion that preserves file size -- a reorder, a same-length
    rename -- restored within the same second reuses the SABOTAGED
    bytecode while the source on disk reads correct.  That cost a whole
    round of hand-sabotage once; it must not cost this suite anything.
    """
    for p in root.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)


def _apply(rev: Reversion) -> Tuple[str, str]:
    """Apply *rev*; return (state, detail). Caller must restore."""
    path = ROOT / rev.target
    if not path.is_file():
        return BLOCKED, f"{rev.target} does not exist"
    src = path.read_text(encoding="utf-8")
    n = src.count(rev.find)
    if n != 1:
        return BLOCKED, (
            f"the reversion's anchor appears {n} times in {rev.target} "
            f"(need exactly 1). The source moved and this reversion is "
            f"stale -- it is NOT known whether the guard still works."
        )
    path.write_text(src.replace(rev.find, rev.replace, 1), encoding="utf-8")
    return PASS, ""


def _run_guard(module_name: str, test_name: str) -> int:
    """Run ONE test of a guard module in a SUBPROCESS; return its exit code.

    A subprocess because this process has already imported the module
    under test and the code it inspects; re-running in-process would
    read stale imports and report on the pre-reversion tree.
    """
    _clear_pycache(ROOT / "jaato-server")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "jaato-server"), str(ROOT / "jaato-sdk"),
         env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x",
         f"{_module_path(module_name)}::{test_name}"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    return proc.returncode


_CASES = [
    (name, mod, rev)
    for name, mod in _guard_modules()
    for rev in mod.REVERSIONS
]


def test_at_least_one_guard_declares_a_reversion():
    """Anchor.  With no cases, every assertion below is vacuous.

    The failure this catches is the suite discovering nothing and passing
    -- which is the exact shape of the defects it was written for.
    """
    assert _CASES, (
        "no guard module declares REVERSIONS, so this suite exercises "
        "nothing and reports success. Either the declarations were removed "
        "or discovery is broken; both are failures, not empty states."
    )


@pytest.mark.parametrize(
    "module_name,rev",
    [(n, r) for n, _m, r in _CASES],
    ids=[f"{n}::{r.test}" for n, _m, r in _CASES],
)
def test_the_guard_fails_when_its_defect_is_put_back(module_name, rev):
    path = ROOT / rev.target
    # BYTES, not text: several files in this tree are CRLF, and a
    # read_text/write_text round trip silently rewrites every line
    # ending in them -- leaving the whole file "modified" in a working
    # tree the suite is supposed to leave exactly as it found it.
    original = path.read_bytes() if path.is_file() else None
    state, detail = _apply(rev)
    if state == BLOCKED:
        pytest.fail(
            f"BLOCKED (this is NOT a pass): {detail}\n\n"
            f"guard: {module_name}\nshould notice: {rev.because}"
        )
    try:
        code = _run_guard(module_name, rev.test)
    finally:
        if original is not None:
            path.write_bytes(original)
        _clear_pycache(ROOT / "jaato-server")

    assert code != 0, (
        f"{module_name}::{rev.test} PASSED with its defect put back.\n\n"
        f"  reverted : {rev.target}\n"
        f"  should notice: {rev.because}\n\n"
        f"The guard is decorative: it reports success whether or not the "
        f"thing it guards is true."
    )
