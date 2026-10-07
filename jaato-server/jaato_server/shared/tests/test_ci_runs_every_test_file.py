"""Every test file on disk must be named by a commit-triggered workflow.

WHY THIS EXISTS.  The recurring failure this guard closes is not any one
missing path; it is that **nothing asserted the mapping between test
files on disk and the paths CI hands to pytest**.  Measured on the tree
at ``f8d7a4c``: 699 test files existed, 581 were run by a workflow that
fires on a commit, and 118 were run by no such workflow -- eight
directories, ~1560 tests, of which 31 were already red.  Two of those
directories were reachable only through ``workflow_dispatch`` publish
workflows: manual, so in practice never (#736).

The repo had been bitten by this shape five separate times before anyone
wrote it down, and the CI file says so in its own comments:
``server/tests`` and ``shared/plugins`` rotted for months;
``jaato-sdk/tests`` was believed to be the last gap and was not;
``jaato-sdk/jaato_sdk/tests`` was a second directory "one path segment
away" carrying 102 drifted baselines; #716 added a fourth
(``shared/scaffold/tests``).  Every fix so far was "someone noticed".

The denominator moves on its own, which is the argument for a check
rather than another audit: 699 files when #736 was filed, 809 two days
before it was fixed, 843 on the commit that fixed it.  A gap that widens
with every directory added cannot be closed by counting it once.

WHAT THIS ASSERTS, EXACTLY -- and a green run here must not be read as
more than this sentence says:

    Every test file is **named by** a step of a workflow that fires on a
    commit.

It does NOT assert that the file's tests **execute**.  A collection
error can take a whole file's tests with it -- ``test_runtime_limits_e2e
.py``'s unguarded cgroup probe did exactly that, host-state dependently,
and CI's ``--continue-on-collection-errors`` is what contains it.

It does NOT assert that a step's filters **select** the file.  A path
reached through a marker-filtered invocation -- the SDK leg runs
``pytest -q -m conformance jaato-sdk/jaato_sdk/conformance/`` -- is
counted as covering that whole directory, because the alternative is
evaluating pytest's selection language, which is a reimplementation of
pytest.  So a file added to ``conformance/`` WITHOUT
``pytestmark = pytest.mark.conformance`` is named by a commit-triggered
step, deselected by its filter, and reads as covered here while never
running.  (Those tests are deselected by default via pyproject
``addopts``, which is what makes the marker load-bearing rather than
decorative.)  Same class as the skip below: named is not selected, and
selected is not run.

It does NOT assert that the tests **ran rather than SKIPPED**, and this
one is not hypothetical -- it bit the very PR that added this guard.
``jaato-server/tests/`` was measured green-and-wire-ready on two
separate hosts before being wired here.  Both lacked AppArmor, where
``test_phase2_multitenant_apparmor.py`` skips at module level; CI HAS
AppArmor, so the first commit-triggered run to reach that body found it
had rotted through -- it drove a synchronous ``IPCClient`` API (
``connect(workspace=..., apparmor=...)``, ``session_new``,
``wait_for_idle``, ``get_history``) that no longer exists anywhere in
the SDK.  A skip is indistinguishable from a pass in a summary line,
and "green" measured in the wrong environment is not green.  Coverage
is not execution, and execution on YOUR host is not execution on CI's.

It does NOT assert that the result **means** anything independent of
what else ran in that process.  211 tests in this tree were measured as
scope-dependent: the failure sets of a directory-scoped run and a
whole-tree run were **disjoint**, every test failing in one passing in
the other, all of them green in isolation.

Those are five different properties.  Only this one is cheap to check,
and it is still the property whose absence produced eight directories
and five rediscoveries.

HOW IT WORKS.  Walk the tree for ``test_*.py`` / ``*_test.py``; parse
the pytest invocations out of ``.github/workflows/*.y*ml`` (the parser is
``scripts/ci_workflows.py``, shared with ``scripts/check.py``); a file is
covered when some commit-triggered pytest INVOCATION names it, or names
a directory above it, and that same invocation does not ``--ignore`` it.
Existential: one leg running a file is coverage, however many other legs
exclude it.

RATCHET, NOT THRESHOLD.  ``UNCOVERED`` is an allowlist of deliberate
exclusions with a reason beside each, following
``test_cyclomatic_complexity_audit.py`` (a frozen BASELINE, "not a
threshold, a ratchet") and ``test_session_env_audit.py`` (a frozen
allowlist with a companion staleness test).  It may only SHRINK: an
entry that becomes covered, or that no longer matches anything, fails --
so wiring a directory in forces its removal, and the list cannot decay
into decoration.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS = ROOT / ".github" / "workflows"

#: Directories that never hold shipped test files.
_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    "build", "dist", ".tox", ".mypy_cache", ".ruff_cache", ".eggs",
})

# ---------------------------------------------------------------------------
# UNCOVERED -- test files/directories deliberately run by NO
# commit-triggered workflow.  An entry may be a file or a directory
# prefix; a directory means "new files here are knowingly uncovered",
# which is a claim that needs its reason beside it.
#
# Per-file entries for a whole directory were deliberately rejected: a
# list of 63 runner tests would churn on every commit that adds one, and
# would teach people to append to the allowlist reflexively.  A directory
# entry states the DECISION once.
#
# This list may only SHRINK.  Adding to it is allowed and must be
# intentional; test_allowlist_has_no_stale_entries fails once an entry
# becomes covered or stops matching anything, so wiring a directory in
# forces its removal in the same commit.
# ---------------------------------------------------------------------------
UNCOVERED: Dict[str, str] = {
    "out-of-tree-plugins/":
        "Not shipped and not installed by any job -- it exists to prove "
        "the entry-point plugin surface works from outside the tree. Same "
        "exclusion test_cyclomatic_complexity_audit.py already documents.",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_test_files() -> Set[str]:
    """Every ``test_*.py`` / ``*_test.py`` in the repo, build dirs aside.

    Returns repo-relative POSIX paths, which is the one spelling every
    other function here compares in.
    """
    found: Set[str] = set()
    for path in ROOT.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        name = path.name
        if name.startswith("test_") or name.endswith("_test.py"):
            found.add(path.relative_to(ROOT).as_posix())
    return found


# ---------------------------------------------------------------------------
# Workflow parsing -- shared with scripts/check.py (#1415)
#
# The parser lives in ``scripts/ci_workflows.py`` so the local check and
# this guard read ONE answer to "what does CI run".  It is loaded by PATH,
# relative to this file, so that inside the reversion meta-guard's
# sandbox the sandbox's copy is the one read.
# ---------------------------------------------------------------------------

def _load_script(name: str):
    """Import ``scripts/<name>.py`` from the tree this file lives in."""
    mod_name = f"_jaato_scripts_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_cw = _load_script("ci_workflows")
_triggers = _cw.triggers
_is_commit_triggered = _cw.is_commit_triggered
_Invocation = _cw.Invocation
_covered_invocations = _cw.covered_invocations


def _all_covered_roots(invocations: List[_Invocation]) -> Set[str]:
    """Every path any invocation names, ignores aside.

    For the self-probe that asks whether the scan resolved anything at
    all -- a question about the PARSER, not about coverage.
    """
    return set().union(*(inv.covered for inv in invocations)) \
        if invocations else set()


def _under(path: str, root: str) -> bool:
    """Is ``path`` the file ``root``, or a file beneath directory ``root``?"""
    return path == root or (root.endswith("/") and path.startswith(root))


def _is_covered(path: str, invocations: List[_Invocation]) -> bool:
    """Does SOME invocation run ``path`` without ignoring it?

    Existential, not universal: one leg naming the file is coverage,
    however many other legs exclude it.
    """
    return any(
        any(_under(path, r) for r in inv.covered)
        and not any(_under(path, r) for r in inv.ignored)
        for inv in invocations
    )


def _allowlisted(path: str) -> bool:
    return any(
        path == entry or (entry.endswith("/") and path.startswith(entry))
        or fnmatch.fnmatch(path, entry)
        for entry in UNCOVERED
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_the_scan_sees_the_workflow_it_is_supposed_to_see() -> None:
    """Self-check: a parser that extracts nothing is SILENTLY GREEN.

    This is the one test here I would not skip.  Every other failure
    mode of this guard is loud; a broken parser makes every file read as
    covered and the guard passes forever -- and a guard that cannot fail
    is worse than no guard, because it is read as an assurance.

    Three specific ways the parser has already been observed to break,
    each asserted below: the ``on: -> True`` YAML 1.1 boolean, the
    matrix ``run:`` strings that are not reachable from step ``run:``
    keys, and a path spelling that resolves to nothing on disk.
    """
    ci = WORKFLOWS / "ci-tests.yml"
    assert ci.is_file(), "ci-tests.yml is the workflow this repo gates on"
    doc = yaml.safe_load(ci.read_text())

    assert _triggers(doc) is not None, (
        "the `on:` key read as absent -- PyYAML parses the bare key `on` "
        "as the BOOLEAN True (YAML 1.1), so `doc['on']` misses it and "
        "every workflow reads as untriggered"
    )
    assert _is_commit_triggered(doc), (
        "ci-tests.yml fires on pull_request and on push to main; a parser "
        "that says otherwise collects no steps at all"
    )

    invocations = _covered_invocations()
    covered = _all_covered_roots(invocations)
    assert covered, (
        "no pytest path was resolved out of any workflow. Every test file "
        "would then read as uncovered -- or, with one comparison flipped, "
        "as covered -- and this guard would assert nothing"
    )
    assert len(covered) >= 5, (
        f"only {len(covered)} pytest path(s) resolved; ci-tests.yml names "
        "far more than that across its matrix legs, so the scan is "
        "probably reading step `run:` keys and missing the matrix strings "
        "that reach the step as ${{ matrix.leg.run }}"
    )

    known = "jaato-server/jaato_server/shared/tests/test_ci_runs_every_test_file.py"
    assert Path(ROOT / known).is_file(), "this file moved; update the probe"
    assert _is_covered(known, invocations), (
        f"{known} is named by the contract-guards job and must read as "
        "covered; if it does not, path spellings are not comparable and "
        "every verdict below is noise"
    )


def test_no_uncovered_test_files_outside_the_allowlist() -> None:
    """A test file no commit-triggered step names fails HERE.

    On the commit that adds it -- which is the whole point.  A new test
    DIRECTORY is the case that actually recurs: eight of them
    accumulated before anyone went looking, each invisible because the
    covered directories had almost the same names.
    """
    files = _discover_test_files()
    assert files, "no test files discovered at all; the walk is broken"

    invocations = _covered_invocations()
    uncovered = sorted(
        f for f in files
        if not _is_covered(f, invocations) and not _allowlisted(f)
    )
    if uncovered:
        by_dir: Dict[str, int] = {}
        for f in uncovered:
            by_dir[str(Path(f).parent) + "/"] = by_dir.get(
                str(Path(f).parent) + "/", 0) + 1
        summary = "\n".join(
            f"  {n:4d}  {d}" for d, n in sorted(
                by_dir.items(), key=lambda kv: -kv[1])
        )
        pytest.fail(
            f"{len(uncovered)} test file(s) in "
            f"{len(by_dir)} director(ies) are run by NO commit-triggered "
            f"workflow, so nothing they assert can block a merge:\n"
            f"{summary}\n\n"
            "Either add the path to a step in .github/workflows/"
            "ci-tests.yml, or -- if the exclusion is deliberate -- add an "
            "entry to UNCOVERED in this file WITH THE REASON. Run this "
            "module as a script to print the current set."
        )


def test_allowlist_has_no_stale_entries() -> None:
    """An UNCOVERED entry that is now covered, or matches nothing, fails.

    This is the half that makes the list a ratchet rather than
    decoration: wiring a directory into CI forces deleting its entry in
    the same commit, so the allowlist can only shrink.  It is the same
    staleness discipline ``test_session_env_audit.py`` applies to its
    own allowlist and ``test_cyclomatic_complexity_audit.py`` to its
    BASELINE.
    """
    files = _discover_test_files()
    invocations = _covered_invocations()

    stale: List[str] = []
    for entry in sorted(UNCOVERED):
        matched = [
            f for f in files
            if f == entry
            or (entry.endswith("/") and f.startswith(entry))
            or fnmatch.fnmatch(f, entry)
        ]
        if not matched:
            stale.append(
                f"  {entry!r}: matches no test file on disk -- the files "
                f"were moved or deleted; drop the entry"
            )
            continue
        still_uncovered = [
            f for f in matched if not _is_covered(f, invocations)
        ]
        if not still_uncovered:
            stale.append(
                f"  {entry!r}: every file it covers is now named by a "
                f"commit-triggered workflow; delete the entry"
            )
    assert not stale, (
        "UNCOVERED has stale entries. It may only shrink -- an entry that "
        "outlives its reason turns the allowlist into decoration:\n"
        + "\n".join(stale)
    )


def test_an_ignore_binds_to_the_invocation_that_carries_it() -> None:
    """One leg's ``--ignore`` must not cancel another leg's coverage.

    The scan used to union ``covered`` and ``ignored`` across every
    invocation in every workflow and let the union of ignores win.  With
    a single set of each, "some leg runs this file" and "some leg
    excludes this file" cannot both be represented, and the second
    silently beat the first -- so a file named explicitly by one job read
    as UNCOVERED the moment any other job ignored it.

    That is failure in the expensive direction: the guard exists to catch
    a test file going unrun, and the union rule instead failed a
    repository whose CI had just stopped running one file TWICE.  It was
    reachable the first time anyone wrote an ``--ignore`` for a file
    named elsewhere, which is #1080's fix to the duplicated meta-guard.

    Asserted twice over: on synthetic invocations, which pin the rule
    whatever the workflows say, and on the real workflows for the file
    that made it reachable.
    """
    only = "jaato-server/jaato_server/shared/tests/test_only_here.py"
    runs_it = _Invocation(covered={only}, ignored=set())
    excludes_it = _Invocation(
        covered={"jaato-server/jaato_server/shared/tests/"}, ignored={only},
    )

    assert _is_covered(only, [runs_it, excludes_it]), (
        "a file one invocation names and another ignores must read as "
        "covered; it runs, in the first one"
    )
    assert _is_covered(only, [excludes_it, runs_it]), (
        "the verdict must not depend on the order invocations were "
        "parsed in -- workflow files are read in sorted name order"
    )
    assert not _is_covered(only, [excludes_it]), (
        "with nothing left running the file, it is uncovered -- "
        "otherwise the rule accepts every file and asserts nothing"
    )

    # And live, on the file #1080 moved: the `reversion-guard` job names
    # it, the `shared/tests` leg ignores it.  If this ever reads False
    # the meta-guard runs nowhere and the de-duplication went too far.
    meta = ("jaato-server/jaato_server/shared/tests/"
            "test_every_guard_detects_its_own_reversion.py")
    invocations = _covered_invocations()
    assert Path(ROOT / meta).is_file(), "the meta-guard moved; update this"
    assert any(any(_under(meta, r) for r in inv.ignored)
               for inv in invocations), (
        f"{meta} is no longer ignored anywhere, so this probe has stopped "
        "exercising the per-invocation rule -- point it at whatever file "
        "carries an --ignore now, or drop it"
    )
    assert _is_covered(meta, invocations), (
        f"{meta} reads as uncovered. Some job must still RUN it: it is "
        "the suite that proves every other guard detects its own "
        "reversion, and nothing else would notice it going unrun"
    )


# ---------------------------------------------------------------------------
# Reversions -- the meta-suite
# (test_every_guard_detects_its_own_reversion) discovers this list by
# name and asserts each one makes the NAMED test fail.  A guard that
# cannot notice its own reversion is not evidence.
# ---------------------------------------------------------------------------
from jaato_server.shared.tests.reversion import (  # noqa: E402
    Reversion,
)

REVERSIONS = [
    Reversion(
        target=".github/workflows/ci-tests.yml",
        find="                jaato-server/jaato_server/server/runner/tests/ || rc=1\n",
        replace="",
        because=(
            "un-wiring a test tree from the only commit-triggered "
            "workflow must be caught on the commit that does it -- 63 "
            "runner test files went unrun for months exactly this way"
        ),
        test="test_no_uncovered_test_files_outside_the_allowlist",
    ),
    Reversion(
        target="scripts/ci_workflows.py",
        find='    if "on" in doc:\n        return doc["on"]\n    return doc.get(True)',
        replace='    return doc.get("on")',
        because=(
            "PyYAML reads the bare key `on:` as the boolean True, so "
            "dropping the fallback makes every workflow read as "
            "untriggered -- the parser then resolves no paths and the "
            "self-check must catch it, because the coverage test itself "
            "would go green-and-vacuous"
        ),
        test="test_the_scan_sees_the_workflow_it_is_supposed_to_see",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/tests/test_ci_runs_every_test_file.py",
        find="""however many other legs exclude it.
    \"\"\"
    return any(
        any(_under(path, r) for r in inv.covered)
        and not any(_under(path, r) for r in inv.ignored)
        for inv in invocations
    )""",
        replace="""however many other legs exclude it.
    \"\"\"
    if any(any(_under(path, r) for r in inv.ignored)
           for inv in invocations):
        return False
    return any(any(_under(path, r) for r in inv.covered)
               for inv in invocations)""",
        because=(
            "the global-union rule this replaced: one leg's --ignore "
            "cancelling every other leg's coverage, so a file named "
            "explicitly by contract-guards read as unrun the moment the "
            "shared/tests leg stopped duplicating it"
        ),
        test="test_an_ignore_binds_to_the_invocation_that_carries_it",
    ),
]


if __name__ == "__main__":                       # pragma: no cover
    # Print the current uncovered set, grouped, so a deliberate
    # re-freeze of UNCOVERED is one command -- the same affordance
    # test_cyclomatic_complexity_audit.py offers for its BASELINE.
    _files = _discover_test_files()
    _inv = _covered_invocations()
    _unc = sorted(f for f in _files if not _is_covered(f, _inv))
    print(f"{len(_files)} test files, "
          f"{len(_files) - len(_unc)} covered, {len(_unc)} UNCOVERED")
    _by: Dict[str, int] = {}
    for _f in _unc:
        _d = str(Path(_f).parent) + "/"
        _by[_d] = _by.get(_d, 0) + 1
    for _d, _n in sorted(_by.items(), key=lambda kv: -kv[1]):
        print(f"  {_n:4d}  {_d}")
