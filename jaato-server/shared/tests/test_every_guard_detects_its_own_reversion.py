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

WHERE THE SABOTAGE HAPPENS -- AND WHY NOT HERE (#995)
=====================================================

Every case rewrites a source file.  Until #995 it rewrote **the file in
the working tree**, and put it back in a per-case ``finally``.  A
``finally`` is an exception handler, not a crash-safety mechanism: it
covers a pass and it covers a raise, and it covers none of SIGKILL, a CI
job timeout, a container restart, or an operator's interrupt landing
between the write and the restore.  In all of those the sabotage is
simply left on disk, indistinguishable from deliberate work in progress.

That was measured, not feared.  Seven of nine agent runs in one session
left a dirty tree, and on one occasion a ``git add -A`` swept up the
suite's live sabotage of ``openrouter/auth.py`` and committed the
deletion of ``__repr__ = secret_safe_repr("api_key")`` -- #721's
protection against an API key reaching a log through a default ``repr``.
It was caught by diffing failing test IDs against a baseline, which is
not a mechanism anyone should have to rely on.

So the working tree is no longer written **at all**.  One disposable
copy of the checkout is made per session (:func:`worktree_sandbox`), and
every sabotage, every guard subprocess and every restore happens inside
it.  The difference between the two designs is the difference between a
failure that is recoverable and one that is unreachable: there is no
interrupt, signal or timeout that can leave the developer's tree
modified, because no code path here ever opens a file under
:data:`ROOT` for writing.

Three things hold that property up rather than asserting it:

* :func:`_sandbox_path` is the only way a case obtains a path to write,
  and it REFUSES a path that resolves outside the sandbox or inside
  ``ROOT`` -- symlinks included, since it resolves before it compares.
* Each case records the real file's bytes before it starts and asserts
  they are unchanged afterwards (:func:`_assert_working_tree_untouched`).
  A future change that reintroduces an in-place write fails loudly on
  the case that made it, rather than silently on somebody's next commit.
* :func:`test_a_sabotage_never_reaches_the_working_tree` states the
  property directly, on a real reversion.

WHAT THE SANDBOX DOES NOT WEAKEN.  A guard subprocess reads the copy --
the copy is what ``PYTHONPATH`` and ``cwd`` point at -- so if the
isolation were broken and it read the real tree instead, it would be
reading UNSABOTAGED source, the guard would pass, and the case would
fail as "decorative".  A green run of this suite is therefore itself
evidence that the sandbox is being read: there is no configuration of
this mechanism in which a broken copy reports success.

COST.  One copy of ~2.4k files, plus a ``compileall`` pass so the
subprocesses start with warm bytecode -- a couple of seconds once per
session, against ~5s per case.  Measured on this tree a guard case runs
marginally FASTER in the sandbox than in the checkout.

NO LONGER TRUE, and deliberately recorded: the folklore that this suite
must not run under ``pytest -n`` or alongside anything else, and that
``git status`` is untrustworthy near it.  Both were consequences of the
in-place design.  Each xdist worker builds its own sandbox (session
scope is per worker), which costs disk rather than correctness.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

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

#: Import roots the guard subprocess needs, in the order it should try
#: them.  Prepended to any inherited ``PYTHONPATH`` so the sandbox's copy
#: of a package is the one that resolves.  ``jaato-tui`` is here because
#: the checkout's editable installs cover it too; a guard importing it
#: must get the sandbox's copy like everything else.
_IMPORT_ROOTS = ("jaato-server", "jaato-sdk", "jaato-tui")

#: Names never copied into the sandbox.  VCS metadata, virtual
#: environments, caches and agent scratch -- none of which any guard
#: reads, and which together dwarf the source: measured on one working
#: checkout at 189 MB of tracked files against 8.3 GB under ``.claude``
#: alone, where sibling agents keep their worktrees.
#:
#: ``__pycache__`` is excluded on purpose rather than for size: a copied
#: ``.pyc`` records the ORIGINAL file's path in ``co_filename``, so
#: ``inspect.getsource`` on a module loaded from one would read the real
#: checkout -- a pointer back into the tree this sandbox exists to leave
#: alone.  The copy is compiled fresh instead (:func:`_precompile`),
#: which costs about a second and leaves no such pointer.
#:
#: An over-eager exclusion cannot pass silently: a target that is not in
#: the copy is named by
#: :func:`test_every_reversion_target_exists_in_the_sandbox`, and every
#: case using it reports ``BLOCKED``, which fails.
_NOT_COPIED = (
    ".git", ".venv", "venv", ".claude", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".hypothesis", ".nox",
    "node_modules", "htmlcov", ".coverage",
)


def _tail(text: str, limit: int = 1200) -> str:
    """The last of *text*, for a BLOCKED message.

    Bounded because a failing guard's output can be a whole traceback
    and the useful part -- pytest's verdict line -- is at the end.
    """
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


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


#: Modules discovery could not import THAT DECLARE REVERSIONS, as
#: ``(name, reason)``.  Populated by :func:`_guard_modules`, asserted
#: empty below.
_IMPORT_FAILURES: List[Tuple[str, str]] = []

#: A module-level ``REVERSIONS`` binding, read from SOURCE.  Text, not
#: an import: the question is only asked about a module that already
#: failed to import, so importing it to find out is not available.
_DECLARES_REVERSIONS = re.compile(r"^REVERSIONS\s*[:=]", re.MULTILINE)


def _source_declares_reversions(finder, name: str) -> bool:
    """Does *name*'s source bind ``REVERSIONS`` at module level?

    Best effort: a source that cannot be read answers ``True``, because
    "I could not tell" must not quietly become "it does not matter".
    """
    for package in _PACKAGES:
        candidate = ROOT / package / f"{name}.py"
        if candidate.is_file():
            try:
                return bool(_DECLARES_REVERSIONS.search(
                    candidate.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                return True
    return True


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
        except Exception as exc:
            # A module that will not import is a problem for its OWN test
            # run, which will say so far more usefully than a name here --
            # UNLESS it declares REVERSIONS, in which case every one of
            # them is now silently unexercised while this suite still
            # reports success.  That distinction is why the source is
            # read rather than the failure simply recorded: these two
            # packages hold 384 test modules and 82 declare reversions,
            # so flagging every import failure would fail this suite for
            # modules that contribute nothing to it (#1065).
            if _source_declares_reversions(mod.module_finder, mod.name):
                _IMPORT_FAILURES.append(
                    (mod.name, f"{type(exc).__name__}: {exc}"))
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


# ============================================================
# The sandbox: a disposable copy of the checkout (#995)
# ============================================================


def _copy_worktree(dest: Path) -> None:
    """Copy the checkout into *dest*, minus the directories above.

    ``copy2`` preserves mtimes, which is not required for correctness --
    :func:`_precompile` records whatever stats the copy ends up with --
    but keeps the sandbox a faithful mirror, which matters for any guard
    that reads a timestamp.
    """
    shutil.copytree(
        ROOT, dest, symlinks=True, copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns(*_NOT_COPIED),
    )


def _precompile(sandbox: Path) -> None:
    """Warm the sandbox's bytecode so each case's subprocess starts hot.

    Best effort: a module that will not compile is a problem for its own
    test run, and an unwarmed sandbox is slow rather than wrong.  The
    return code is deliberately ignored for that reason.
    """
    roots = [str(sandbox / name) for name in _IMPORT_ROOTS
             if (sandbox / name).is_dir()]
    if not roots:
        return
    subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "-j0", *roots],
        capture_output=True, text=True, timeout=600,
    )


@pytest.fixture(scope="session")
def worktree_sandbox(tmp_path_factory) -> Iterator[Path]:
    """One disposable copy of the checkout, shared by every case.

    Built lazily, so the probe tests below cost nothing, and shared,
    because the copy is the expensive part and each case restores what
    it touched.  A case that dies mid-sabotage poisons only this copy,
    and only for a run that is already over.

    A failure to build is raised, never skipped: a case that cannot be
    set up has exercised nothing, which is the ``BLOCKED`` state this
    suite exists to stop reporting as green.

    Removed at session end so only one copy of the checkout is on disk
    at a time; a removal that fails is ignored, because a leftover
    directory under the temp root is litter and the thing this fixture
    exists to prevent is a modified *checkout*.
    """
    dest = tmp_path_factory.mktemp("guard_sandbox") / "checkout"
    assert not dest.resolve().is_relative_to(ROOT.resolve()), (
        f"the sandbox {dest} is inside the working tree {ROOT}; a "
        f"sabotage written there is a sabotage of the checkout"
    )
    _copy_worktree(dest)
    _precompile(dest)
    try:
        yield dest
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def _sandbox_path(sandbox: Path, target: str) -> Path:
    """Resolve a reversion's target INSIDE the sandbox, or refuse.

    The single gate on every write this module performs.  It resolves
    before it compares, so a symlink in the copy that points back into
    the checkout is refused rather than followed, and it names ``ROOT``
    explicitly rather than trusting that "inside the sandbox" implies
    "outside the tree" -- two conditions, because a future caller could
    hand it a sandbox built in the wrong place.
    """
    path = sandbox / target
    resolved = path.resolve()
    if not resolved.is_relative_to(sandbox.resolve()):
        raise AssertionError(
            f"reversion target {target!r} resolves to {resolved}, outside "
            f"the sandbox {sandbox}. Refusing to write it."
        )
    if resolved.is_relative_to(ROOT.resolve()):
        raise AssertionError(
            f"reversion target {target!r} resolves into the working tree "
            f"({resolved}). This suite never writes to the checkout; see "
            f"the module docstring (#995)."
        )
    return path


def _assert_working_tree_untouched(
        real: Path, before: Optional[bytes], rev: Reversion) -> None:
    """The tripwire: the checkout's copy of the target did not change.

    Cheap (one small file per case) and it converts any future in-place
    write from a silent dirty tree into a named failure on the case that
    performed it.

    A changed file has two possible authors -- this suite, or somebody
    editing the checkout while it runs -- so the message names which,
    by checking whether what is on disk is exactly this reversion's
    output.  Either is worth stopping for: the second means the guard
    just ran against source that is no longer the source under review.
    """
    after = real.read_bytes() if real.is_file() else None
    if after == before:
        return
    sabotaged = (before is not None and after is not None
                 and after == before.replace(rev.find.encode(),
                                             rev.replace.encode(), 1))
    culprit = (
        "THIS SUITE wrote its sabotage into the checkout -- the #995 "
        "defect. Restore the file from git and fix the write."
        if sabotaged else
        "it was not written by this suite (the bytes are not this "
        "reversion's output), so something else edited the checkout "
        "mid-run and this guard reported on stale source."
    )
    pytest.fail(f"{real} CHANGED while this case ran: {culprit}")


def _apply(rev: Reversion, sandbox: Path) -> Tuple[str, str]:
    """Apply *rev* inside *sandbox*; return (state, detail).

    The caller restores.  Nothing under :data:`ROOT` is opened for
    writing here or anywhere else in this module.
    """
    path = _sandbox_path(sandbox, rev.target)
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


#: pytest's own exit codes, and what each one is evidence OF.  Only
#: TESTS_FAILED is evidence that the guard noticed its defect: the run
#: reached the test body and the test rejected the sabotaged source.
#:
#: The others are why this exists (#1065).  ``assert code != 0`` read
#: every one of them as detection, so a ``test`` naming a nodeid pytest
#: could not resolve -- exit 4, no test body run at all -- certified the
#: guard forever.  23 in-tree reversions were in that state when this
#: was written, and none of them produced any signal.
_OK, _TESTS_FAILED, _INTERRUPTED = 0, 1, 2
_INTERNAL_ERROR, _USAGE_ERROR, _NO_TESTS = 3, 4, 5

_EXIT_MEANING = {
    _OK: "the test PASSED with the defect present",
    _TESTS_FAILED: "the test failed",
    _INTERRUPTED: "the run was interrupted",
    _INTERNAL_ERROR: "pytest hit an internal error",
    _USAGE_ERROR: "pytest could not be invoked as asked -- usually a "
                  "`test` naming a nodeid that does not resolve",
    _NO_TESTS: "the nodeid matched no test",
}


class _GuardRun(NamedTuple):
    """What one guard subprocess did.

    ``output`` is kept because #1065's whole difficulty was that it was
    not: the five outcomes are indistinguishable after the fact from an
    exit code alone, and pytest's own complaint names the bad nodeid.
    """

    code: int
    output: str


def _run_guard(module_name: str, test_name: str, sandbox: Path) -> _GuardRun:
    """Run ONE test of a guard module in a SUBPROCESS; report code + output.

    A subprocess because this process has already imported the module
    under test and the code it inspects; re-running in-process would
    read stale imports and report on the pre-reversion tree.

    It runs against the SANDBOX -- ``cwd`` and the leading ``PYTHONPATH``
    entries both point there.  The checkout's editable installs append
    their finders to ``sys.meta_path``, which sits after the ordinary
    path finder, so the sandbox's copy of a package is what resolves.
    Any inherited ``PYTHONPATH`` is kept but demoted behind ours: a
    developer's entry must not silently re-point an import at the
    unsabotaged tree.

    The sabotaged file's bytecode was already dropped by :func:`_apply`,
    so nothing is invalidated here -- every OTHER module's cache is
    still valid and the subprocess starts warm.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(sandbox / name) for name in _IMPORT_ROOTS]
        + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x",
         f"{_module_path(module_name)}::{test_name}"],
        cwd=str(sandbox), env=env, capture_output=True, text=True,
        timeout=600,
    )
    return _GuardRun(proc.returncode,
                     _tail(proc.stdout) or _tail(proc.stderr))


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


# ============================================================
# The property that makes an interrupted run harmless (#995)
# ============================================================


def test_a_target_escaping_the_sandbox_is_refused(tmp_path):
    """Traversal and symlink, the two ways a write leaves the copy."""
    sandbox = tmp_path / "checkout"
    (sandbox / "jaato-server").mkdir(parents=True)

    with pytest.raises(AssertionError, match="outside the sandbox"):
        _sandbox_path(sandbox, "../escaped.py")

    link = sandbox / "jaato-server" / "linked.py"
    link.symlink_to(ROOT / "jaato-server" / "conftest.py")
    with pytest.raises(AssertionError, match="outside the sandbox"):
        _sandbox_path(sandbox, "jaato-server/linked.py")


def test_a_sandbox_built_inside_the_checkout_is_refused():
    """The second condition, which containment alone would not catch.

    A sandbox placed under :data:`ROOT` satisfies "inside the sandbox"
    for every target it holds, so a single check would happily write
    into the working tree.  The fixture refuses to build one there; this
    is the same refusal one layer down, where the write happens.
    """
    with pytest.raises(AssertionError, match="working tree"):
        _sandbox_path(ROOT / "jaato-server", "shared/tests/anything.py")


def test_every_reversion_target_exists_in_the_sandbox(worktree_sandbox):
    """The copy is complete for the cases that will run against it.

    A missing target surfaces as ``BLOCKED``, which already fails -- but
    it blames the reversion for being stale rather than the sandbox for
    being short, so state it here instead.

    Byte-equality against the checkout is deliberately NOT asserted: the
    sandbox is a snapshot taken once per session, and a developer or a
    concurrent agent editing the tree afterwards makes it legitimately
    differ.  The snapshot is the source under review for this run.
    """
    assert not worktree_sandbox.resolve().is_relative_to(ROOT.resolve())
    missing = sorted({rev.target for _n, _m, rev in _CASES
                      if not (worktree_sandbox / rev.target).is_file()})
    assert not missing, (
        f"the sandbox copy is missing reversion targets: {missing}\n"
        f"Either _NOT_COPIED excludes something a guard reads, or the "
        f"copy failed part way. Every case naming one of these would "
        f"report BLOCKED and blame the wrong thing."
    )


def test_a_sabotage_never_reaches_the_working_tree(worktree_sandbox):
    """The #995 property, stated on a real reversion.

    Applies one, checks the copy changed and the checkout did not, and
    restores.  Should this ever fail, the failure is the whole issue:
    an interrupted run would leave that edit in somebody's tree.
    """
    _name, _mod, rev = _CASES[0]
    real = ROOT / rev.target
    before = real.read_bytes()
    copied = _sandbox_path(worktree_sandbox, rev.target)
    original = copied.read_bytes()
    try:
        state, detail = _apply(rev, worktree_sandbox)
        assert state != BLOCKED, detail
        assert copied.read_bytes() != original, "the sandbox was not written"
        assert real.read_bytes() == before, (
            f"applying a reversion modified {real} in the WORKING TREE. "
            f"An interrupted run would leave that edit on disk (#995)."
        )
    finally:
        copied.write_bytes(original)
        _invalidate_bytecode(copied)


def test_a_nodeid_that_resolves_to_nothing_is_not_detection(worktree_sandbox):
    """#1065, driven through the real run path rather than asserted.

    Runs a nodeid that exists nowhere and checks what comes back.  The
    old rule was ``assert code != 0``, and this exit satisfies it -- so
    every assertion here is one the previous version passed.

    The point is the pair: the exit is non-zero AND it is not
    ``TESTS_FAILED``.  Reading only the first is what certified 23
    reversions that ran no test at all.
    """
    name, _mod, _rev = _CASES[0]
    run = _run_guard(name, "test_a_name_no_module_in_this_tree_defines",
                     worktree_sandbox)

    assert run.code != _OK, (
        "a nodeid matching nothing should not report success; if it does, "
        "this probe cannot distinguish the two failures it exists for"
    )
    assert run.code != _TESTS_FAILED, (
        f"pytest exited {run.code} for a nodeid that resolves to nothing, "
        f"but TESTS_FAILED means a test body ran and rejected the source. "
        f"If these ever coincide the verdict below cannot tell "
        f"'the guard noticed' from 'there was no guard'."
    )
    assert run.code in (_USAGE_ERROR, _NO_TESTS), (
        f"expected USAGE_ERROR or NO_TESTS for an unresolvable nodeid, "
        f"got {run.code}. The classification would then send this to the "
        f"wrong branch:\n{run.output}"
    )
    assert run.output, (
        "pytest's own complaint was discarded, so a BLOCKED message "
        "cannot name the bad nodeid -- which is what made this class "
        "invisible to its authors"
    )


def test_the_resolution_rule_matches_pytests_own_selection():
    """``_resolves`` accepts what ``pytest path::name`` accepts.

    Too strict and every parametrised or class-nested reversion reports
    as broken; too loose and the check passes the typos it exists to
    catch.  Both directions are asserted, because the first draft of
    this rule was too strict in exactly the way the last case covers.
    """
    collected = {
        "test_plain",
        "test_parametrised[alpha]",
        "test_parametrised[beta]",
        "TestClass::test_nested",
    }
    assert _resolves("test_plain", collected)
    assert _resolves("test_parametrised", collected), (
        "a bare name must select all of its parametrisations, as pytest does"
    )
    assert _resolves("test_parametrised[beta]", collected)
    assert _resolves("TestClass::test_nested", collected)
    assert _resolves("TestClass", collected), (
        "a class name must select its tests, as pytest does"
    )

    # The control.  Without these the rule could return True always.
    assert not _resolves("test_nested", collected), (
        "a class-nested test named WITHOUT its class resolves to nothing "
        "-- 22 of the 23 reversions #1065 found were this exact shape, so "
        "a rule that accepts it catches none of them"
    )
    assert not _resolves("test_absent", collected)
    assert not _resolves("test_pla", collected), (
        "a bare prefix is not a selector; accepting one would make the "
        "boundary check meaningless"
    )


def test_discovery_imported_every_module_it_walked():
    """A module discovery could not import contributes no cases, silently.

    ``_guard_modules`` swallows the ImportError and moves on, which is
    right for the module's own test run and wrong here: its REVERSIONS
    vanish from this suite while the suite still reports success -- the
    exact "green while exercising nothing" shape it exists to stop.

    Measured while writing this: an ad-hoc probe with one wrong
    ``PYTHONPATH`` entry discovered 227 cases instead of 235 and said
    nothing, and the eight it dropped were never audited.
    """
    assert not _IMPORT_FAILURES, (
        "discovery could not import these test modules, so any REVERSIONS "
        "they declare are silently unexercised:\n"
        + "\n".join(f"  {n}: {why}" for n, why in _IMPORT_FAILURES)
    )


def _collect_nodeids(modules: List[str], sandbox: Path) -> Dict[str, Set[str]]:
    """Every nodeid pytest can see in *modules*, keyed by BASENAME.

    Basename, not path: pytest reports collected nodeids relative to its
    own rootdir (``jaato-server``), while this suite spells module paths
    relative to the repo root.  Keying on the spelling this module uses
    makes every lookup miss and every reversion look broken -- which is
    what a first draft of this check did, reporting all 227 as
    unresolvable.  Basenames are unique across the guard corpus.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(sandbox / name) for name in _IMPORT_ROOTS]
        + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", *modules],
        cwd=str(sandbox), env=env, capture_output=True, text=True,
        timeout=600,
    )
    out: Dict[str, Set[str]] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" not in line or line.startswith("<"):
            continue
        path, _, nodeid = line.partition("::")
        out.setdefault(path.rsplit("/", 1)[-1], set()).add(nodeid)
    return out


def _resolves(candidate: str, collected: Set[str]) -> bool:
    """Does *candidate* select anything, by pytest's own rules?

    Exact, or a prefix at a ``[`` (one parametrisation of many) or a
    ``::`` (a class, selecting its tests) boundary -- which is what
    ``pytest path::name`` accepts.
    """
    return any(c == candidate
               or c.startswith(candidate + "[")
               or c.startswith(candidate + "::")
               for c in collected)


def test_every_reversion_names_a_test_that_exists(worktree_sandbox):
    """Each ``rev.test`` resolves BEFORE anything is sabotaged.

    One collection pass over the whole corpus, and the cheap half of
    #1065: a nodeid that resolves to nothing makes pytest exit 4, which
    the old run-side read as "the guard detected its reversion". The
    verdict below now refuses that exit -- but it refuses it once per
    case, mid-run, after a sabotage. This says it up front, for every
    case at once, at the point an author can act on it.

    It found 23 in-tree reversions certifying nothing when it was added:
    22 naming a class-nested test without its class, and one carrying the
    whole repo-relative path in ``test`` -- the doubled nodeid the issue
    was filed about.
    """
    modules = sorted({_module_path(name) for name, _m, _r in _CASES})
    collected = _collect_nodeids(modules, worktree_sandbox)
    assert collected, (
        "collection produced no nodeids at all, so this check would "
        "pass every reversion vacuously"
    )

    broken = []
    for name, _mod, rev in _CASES:
        base = _module_path(name).rsplit("/", 1)[-1]
        if base not in collected:
            broken.append(f"  {name} :: {rev.test!r}  (module collected nothing)")
        elif not _resolves(rev.test, collected[base]):
            broken.append(f"  {name} :: {rev.test!r}")

    assert not broken, (
        f"{len(broken)} reversion(s) name a `test` that resolves to no "
        f"test. pytest exits USAGE_ERROR on each, no test body runs, and "
        f"the guard is certified by nothing:\n" + "\n".join(broken)
        + "\n\n`test` is the nodeid WITHIN the module -- 'test_x', or "
        "'TestClass::test_x' for a class-nested test. Not a path, and "
        "not a bare name when the test lives in a class."
    )


@pytest.mark.parametrize(
    "module_name,rev",
    [(n, r) for n, _m, r in _CASES],
    ids=[f"{n}::{r.test}" for n, _m, r in _CASES],
)
def test_the_guard_fails_when_its_defect_is_put_back(
        module_name, rev, worktree_sandbox):
    real = ROOT / rev.target
    real_before = real.read_bytes() if real.is_file() else None
    # BYTES, not text: several files in this tree are CRLF, and a
    # read_text/write_text round trip silently rewrites every line
    # ending in them -- leaving the whole file "modified" in a sandbox
    # the next case expects to find exactly as this one did.
    path = _sandbox_path(worktree_sandbox, rev.target)
    original = path.read_bytes() if path.is_file() else None
    state, detail = _apply(rev, worktree_sandbox)
    if state == BLOCKED:
        pytest.fail(
            f"BLOCKED (this is NOT a pass): {detail}\n\n"
            f"guard: {module_name}\nshould notice: {rev.because}"
        )
    try:
        run = _run_guard(module_name, rev.test, worktree_sandbox)
    finally:
        if original is not None:
            path.write_bytes(original)
            _invalidate_bytecode(path)

    _assert_working_tree_untouched(real, real_before, rev)

    if run.code == _TESTS_FAILED:
        return                       # the guard noticed.  The only pass.

    if run.code == _OK:
        pytest.fail(
            f"{module_name}::{rev.test} PASSED with its defect put back.\n\n"
            f"  reverted : {rev.target}\n"
            f"  should notice: {rev.because}\n\n"
            f"The guard is decorative: it reports success whether or not "
            f"the thing it guards is true."
        )

    # Anything else means no test body ran, so nothing was certified --
    # the same state a failed sabotage produces, and reported the same
    # way rather than as a pass (#1065).
    pytest.fail(
        f"BLOCKED (this is NOT a pass): pytest exited "
        f"{run.code} -- {_EXIT_MEANING.get(run.code, 'unrecognised exit')}."
        f"\n\nNo test body ran, so it is NOT known whether "
        f"{module_name}::{rev.test} detects its reversion.\n"
        f"  guard: {module_name}\n"
        f"  test : {rev.test!r}   <- must be the nodeid WITHIN the module "
        f"(a class-nested test needs its class, e.g. 'TestX::test_y')\n"
        f"  should notice: {rev.because}\n\n"
        f"pytest said:\n{run.output}"
    )
