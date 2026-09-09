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
import importlib.util
import os
import pkgutil
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


def _invalidate_bytecode(path: Path) -> None:
    """Drop the cached bytecode for the ONE file a case rewrites.

    Bytecode caching invalidates on (size, mtime), not content.  A
    reversion that preserves file size -- a reorder, a same-length
    rename -- written and restored within the same second reuses the
    SABOTAGED bytecode while the source on disk reads correct.  That
    cost a whole round of hand-sabotage once; it must not cost this
    suite anything.  So every write of a target is followed by this
    call: the sabotage in :func:`_apply`, and the restore in the case's
    ``finally``.

    Only the file a case WRITES can be stale, so only its cache entry
    is dropped.  Until #913 this cleared every ``__pycache__`` under
    ``jaato-server/`` instead, which made all 76 subprocesses recompile
    the tree from source for a guarantee about one file: measured 16.8s
    per case against 8.8s with the cache warm, so roughly ten minutes
    of recompilation.  That is most of a leg with a 20-minute CI budget
    which was measured at 18m36s on main -- this suite grew until it
    nearly stopped running at all, which is its own failure mode in a
    different costume.

    :func:`importlib.util.cache_from_source` rather than a glob: it
    answers with the exact path THIS interpreter would read, honouring
    ``sys.pycache_prefix`` and the version tag, and the subprocess is
    the same interpreter with the same environment.  A target that is
    not Python -- a workflow, a doc -- has no bytecode and is a no-op.
    """
    if path.suffix != ".py":
        return
    try:
        cached = Path(importlib.util.cache_from_source(str(path)))
    except (NotImplementedError, ValueError):  # pragma: no cover
        return
    cached.unlink(missing_ok=True)


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
    _invalidate_bytecode(path)
    return PASS, ""


def _run_guard(module_name: str, test_name: str) -> int:
    """Run ONE test of a guard module in a SUBPROCESS; return its exit code.

    A subprocess because this process has already imported the module
    under test and the code it inspects; re-running in-process would
    read stale imports and report on the pre-reversion tree.

    The sabotaged file's bytecode was already dropped by :func:`_apply`,
    so nothing is invalidated here -- every OTHER module's cache is
    still valid and the subprocess starts warm.
    """
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


# ============================================================
# The invalidation this suite's own correctness rests on
# ============================================================


def _stale_cache_probe(tmp_path, invalidate) -> str:
    """Compile a module, rewrite it behind the cache key, re-import it.

    Returns what the fresh interpreter SAW.  A ``.pyc`` records the
    source's size and mtime-in-whole-seconds, so the rewrite keeps the
    size and the mtime is restored to the exact value the ``.pyc``
    recorded.  The cache key therefore still matches and a reader that
    trusts it answers with the FIRST version -- which is precisely the
    sabotage-reads-as-restored failure the invalidation exists to
    prevent.

    The mtime is pinned rather than relying on both writes landing in
    one clock second: that races the second boundary, and a control
    that passes only most of the time is not a control.
    """
    import py_compile
    mod = tmp_path / "sabotage_probe.py"
    mod.write_text("VALUE = 'aaa'\n", encoding="utf-8")
    py_compile.compile(str(mod), doraise=True)      # populate the cache
    recorded = mod.stat()

    mod.write_text("VALUE = 'bbb'\n", encoding="utf-8")   # same size...
    os.utime(mod, (recorded.st_atime, recorded.st_mtime))  # ...same mtime
    invalidate(mod)

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sabotage_probe; print(sabotage_probe.VALUE)"],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    return proc.stdout.strip()


def test_the_targeted_invalidation_actually_invalidates(tmp_path):
    """The whole suite is worthless if a rewritten file reads stale.

    Narrowing the wipe from the whole tree to one file (#913) is only
    safe if the one file is genuinely dropped, so this exercises the
    real mechanism -- compile, rewrite the source behind an unchanged
    cache key, import in a fresh interpreter -- rather than asserting
    that a path was unlinked.
    """
    assert _stale_cache_probe(tmp_path, _invalidate_bytecode) == "bbb", (
        "a rewritten source file was re-imported as its PREVIOUS "
        "contents: _invalidate_bytecode did not drop the cached "
        "bytecode, so a size-preserving reversion would be tested "
        "against the wrong source"
    )


def test_the_probe_would_notice_a_do_nothing_invalidation(tmp_path):
    """The control.  Without it the test above passes on any platform
    that never caches, and would go on passing if the invalidation were
    deleted entirely."""
    stale = _stale_cache_probe(tmp_path, lambda _p: None)
    assert stale == "aaa", (
        "the probe cannot distinguish an invalidated file from a stale "
        "one on this platform, so the test above proves nothing"
    )


def test_a_non_python_target_is_a_no_op(tmp_path):
    """Reversions target workflows and docs too; they have no bytecode."""
    doc = tmp_path / "notes.md"
    doc.write_text("hello\n", encoding="utf-8")
    _invalidate_bytecode(doc)          # must not raise
    assert doc.read_text(encoding="utf-8") == "hello\n"


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
            _invalidate_bytecode(path)

    assert code != 0, (
        f"{module_name}::{rev.test} PASSED with its defect put back.\n\n"
        f"  reverted : {rev.target}\n"
        f"  should notice: {rev.because}\n\n"
        f"The guard is decorative: it reports success whether or not the "
        f"thing it guards is true."
    )
