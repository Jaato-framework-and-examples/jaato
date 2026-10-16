"""The Python range a wheel declares must be exactly the range CI tests.

WHY THIS EXISTS.  Every workflow in this repo pins a single interpreter,
no matrix, while the four published distributions declared
``requires-python = ">=3.10"``.  So pip installed them on 3.10, 3.11,
3.13 and 3.14, and **nothing in CI had ever run a test on any of them**
(#1076).  The metadata was a promise the build did not check, and the
failure mode it produces is the expensive one: a defect whose only
trigger is a version CI never runs is invisible until a user reports it.

#1076 offered two honest resolutions -- test the declared range, or
declare the tested range.  This tree took the second, in two steps.
First the FLOOR was raised to the version CI installed (``>=3.12``),
leaving the ceiling open: ``>=3.12`` still promised 3.13 and 3.14, and
CI ran neither.  Then CI moved to the interpreter production runs
(3.14) and the range was CLOSED on both sides: ``>=3.14,<3.15`` -- the
declared range is the one interpreter that is tested, nothing more.
This guard is what stops the declaration and CI drifting apart again,
in either direction and at either end:

* someone lowers the floor, or drops/raises the ceiling, and CI keeps
  testing only 3.14 -- the original defect, at either end;
* someone pins a workflow to an interpreter OUTSIDE the declared range
  -- less obvious and worse, because then CI's own ``pip install -e`` is
  the thing pip refuses, so the break is total rather than latent.

WHAT THIS ASSERTS, EXACTLY.  Six properties, and a green run here must
not be read as more than these sentences say:

1. Every published distribution declares the SAME floor.  They are
   installed into one interpreter, so a disagreement is not a range --
   it is the highest of them, silently.
2. That floor is among the interpreter versions a commit-triggered
   workflow installs for a job that runs ``pytest``.  "Commit-triggered"
   excludes the ``workflow_dispatch``-only publish workflows: a version
   only a manual run touches is, in practice, never run (the finding
   #736 records).  "Runs pytest" excludes jobs that only build or
   deploy.
3. Every published distribution declares an upper bound, and they all
   declare the SAME one.  An open ceiling is a promise about every
   future interpreter, which no CI can keep.
4. Every minor version inside each distribution's declared range is
   installed by a commit-triggered job that runs pytest.  With one
   pinned interpreter this means floor == the CI version and ceiling ==
   the next minor; a matrix would let the range widen by exactly the
   versions it adds.
5. Every interpreter version ANY workflow installs lies inside the
   declared range -- including the publish workflows, which build the
   wheels.
6. Each distribution's ``Programming Language :: Python :: X.Y``
   classifiers are exactly the versions in its declared range.  A
   classifier is not machine-enforced by pip, so it is the one place a
   stale claim survives a correct ``requires-python``.

WHAT THIS DOES NOT ASSERT.  That the version CI installs is the version
the tests RAN under -- a job that installs 3.14 and then shells out to a
system ``python3`` would pass here.  Nothing in this tree does that for
a test of the packages (the ``selinux-policy`` job runs Fedora's own
``python3`` on purpose: it tests a policy module, installs no
distribution, and pins no version through ``setup-python``, so it is
invisible to this scan by construction).  The check is a
declaration-versus-declaration comparison, which is what makes it a few
milliseconds of YAML and TOML rather than a build.

SCOPE.  ``out-of-tree-plugins/moon-phase`` is deliberately excluded.  It
is an example of what a third party's package looks like, it is never
published from here, and its range is its author's to choose -- which is
the entire point of the directory.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]

#: The distributions this repository publishes.  Each is installed into
#: the same interpreter as the others, which is why they must agree.
PUBLISHED_DISTS = (
    "jaato-sdk",
    "jaato-server",
    "jaato-tui",
    "jaato-eval",
)

Version = Tuple[int, int]

#: A declared range: the inclusive floor and the EXCLUSIVE ceiling, or
#: ``None`` for a ceiling that was not declared.
Range = Tuple[Version, Optional[Version]]

_FLOOR_RE = re.compile(r"^>=\s*(\d+)\.(\d+)$")
_CEILING_RE = re.compile(r"^<\s*(\d+)\.(\d+)$")
_CLASSIFIER_RE = re.compile(
    r"^Programming Language :: Python :: (\d+)\.(\d+)$"
)
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")


def _shape_failure(requires_python: str, where: str) -> None:
    pytest.fail(
        f"{where}: requires-python = {requires_python!r} is not a bare "
        f">=X.Y floor with at most one <X.Z ceiling, which is the only "
        f"shape this guard parses. Widening the policy means widening "
        f"_parse_range too -- see this module's docstring."
    )


def _parse_range(requires_python: str, where: str) -> Range:
    """Return the ``(floor, ceiling)`` of a ``requires-python`` value.

    Deliberately narrow.  The specifier grammar allows far more than this
    repo uses (``!=``, ``~=``, ``<=``, several of each), and a parser that
    guessed at those would be asserting something it had not read.  A
    value outside the one shape in use -- exactly one ``>=X.Y`` and at
    most one ``<X.Z``, in either order -- fails by NAME, so whoever
    widens the policy also widens the check.  The ceiling is returned as
    ``None`` when absent, rather than rejected here, so the test that
    REQUIRES one can say so in its own words.
    """
    floor: Optional[Version] = None
    ceiling: Optional[Version] = None
    for clause in requires_python.split(","):
        clause = clause.strip()
        floor_match = _FLOOR_RE.match(clause)
        ceiling_match = _CEILING_RE.match(clause)
        if floor_match is not None and floor is None:
            floor = int(floor_match.group(1)), int(floor_match.group(2))
        elif ceiling_match is not None and ceiling is None:
            ceiling = int(ceiling_match.group(1)), int(ceiling_match.group(2))
        else:
            _shape_failure(requires_python, where)
    if floor is None:
        _shape_failure(requires_python, where)
    return floor, ceiling


def _parse_floor(requires_python: str, where: str) -> Version:
    """The inclusive floor of a ``requires-python`` value."""
    return _parse_range(requires_python, where)[0]


def _versions_in(declared: Range, where: str) -> List[Version]:
    """Every minor version inside a CLOSED declared range.

    Only a range within one major version is enumerable this way; a
    ceiling in another major (``<4``) names an unbounded set of minors,
    which no CI pins, so it is refused by name rather than guessed at.
    """
    floor, ceiling = declared
    assert ceiling is not None, f"{where}: no ceiling to enumerate up to"
    if ceiling[0] != floor[0] or ceiling <= floor:
        pytest.fail(
            f"{where}: the declared range >={_fmt(floor)},<{_fmt(ceiling)} "
            f"is not a non-empty range within one major version, so its "
            f"minor versions cannot be enumerated against CI's pins."
        )
    return [(floor[0], minor) for minor in range(floor[1], ceiling[1])]


def _pyproject(dist: str) -> dict:
    path = ROOT / dist / "pyproject.toml"
    assert path.exists(), f"missing {path}"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _declared_ranges() -> Dict[str, Range]:
    ranges: Dict[str, Range] = {}
    for dist in PUBLISHED_DISTS:
        project = _pyproject(dist).get("project", {})
        requires = project.get("requires-python")
        assert requires, f"{dist}/pyproject.toml declares no requires-python"
        ranges[dist] = _parse_range(requires, f"{dist}/pyproject.toml")
    return ranges


def _declared_floors() -> Dict[str, Version]:
    return {dist: rng[0] for dist, rng in _declared_ranges().items()}


def _workflow_files() -> List[Path]:
    workflows = ROOT / ".github" / "workflows"
    files = sorted(
        [*workflows.glob("*.yml"), *workflows.glob("*.yaml")]
    )
    assert files, f"no workflows found under {workflows}"
    return files


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _triggers(doc: dict) -> dict:
    """YAML reads the bare key ``on:`` as the boolean ``True``."""
    raw = doc.get(True, doc.get("on"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return {raw: None}
    if isinstance(raw, list):
        return {key: None for key in raw}
    return {}


def _is_commit_triggered(doc: dict) -> bool:
    return bool({"push", "pull_request"} & set(_triggers(doc)))


def _matrix_versions(job: dict) -> List[str]:
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    values = matrix.get("python-version")
    if values is None:
        return []
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(values)]


def _job_python_versions(job: dict) -> List[Version]:
    """Interpreter versions a job installs, resolving a matrix reference."""
    literal: List[str] = []
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        with_block = step.get("with") or {}
        if "python-version" not in with_block:
            continue
        value = str(with_block["python-version"]).strip()
        if "${{" in value:
            # A matrix reference. Resolve it from the job's own matrix
            # rather than skipping: a matrix is precisely the shape that
            # would let the declared range widen past one interpreter, and
            # a guard that ignored it would not notice the day it lands.
            literal.extend(_matrix_versions(job))
        else:
            literal.append(value)

    versions: List[Version] = []
    for value in literal:
        match = _VERSION_RE.match(value)
        if match is None:
            # A floating spec ("3.x", "pypy-3.10") pins no version this
            # check can compare. Skipping is the safe direction: it can
            # only make the guard miss a version, never invent one.
            continue
        versions.append((int(match.group(1)), int(match.group(2))))
    return versions


def _job_runs_pytest(job: dict) -> bool:
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if "pytest" in str(step.get("run") or ""):
            return True
    return False


def _tested_versions() -> Set[Version]:
    """Versions installed by a commit-triggered job that runs pytest."""
    tested: Set[Version] = set()
    for path in _workflow_files():
        doc = _load(path)
        if not _is_commit_triggered(doc):
            continue
        for job in (doc.get("jobs") or {}).values():
            if not isinstance(job, dict) or not _job_runs_pytest(job):
                continue
            tested.update(_job_python_versions(job))
    return tested


def _all_pinned_versions() -> Dict[Version, List[str]]:
    """Every version any workflow installs, mapped to where it is pinned."""
    pinned: Dict[Version, List[str]] = {}
    for path in _workflow_files():
        doc = _load(path)
        for name, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for version in _job_python_versions(job):
                pinned.setdefault(version, []).append(f"{path.name}:{name}")
    return pinned


def _fmt(version: Version) -> str:
    return f"{version[0]}.{version[1]}"


def test_published_distributions_declare_one_floor():
    """They install into one interpreter, so a disagreement is not a range."""
    floors = _declared_floors()
    distinct = set(floors.values())
    assert len(distinct) == 1, (
        "the published distributions disagree about their Python floor: "
        + ", ".join(
            f"{dist} {_fmt(floor)}" for dist, floor in sorted(floors.items())
        )
        + ". They are installed together, so the effective floor is the "
        "highest of them and the others' metadata is a claim nothing "
        "honours."
    )


def test_the_declared_floor_is_a_version_ci_runs_tests_on():
    """#1076: the floor was ``>=3.10`` and no CI job had ever run it."""
    # max(), not "the" floor: when the four disagree the effective floor
    # is the highest of them (which is what the test above reports), and
    # reading an arbitrary element of a set would make this test's verdict
    # depend on set ordering.
    floor = max(_declared_floors().values())
    tested = _tested_versions()
    assert tested, (
        "no commit-triggered workflow job both installs a Python version "
        "and runs pytest -- this guard cannot be satisfied, and neither "
        "can any claim about what is tested."
    )
    assert floor in tested, (
        f"requires-python declares a floor of {_fmt(floor)}, and no "
        f"commit-triggered job that runs pytest installs it. CI tests "
        f"{sorted(_fmt(v) for v in tested)}. Either lower the CI pin to "
        f"the floor (or add it to a matrix), or raise the floor to what "
        f"is actually tested -- #1076 is the ticket for having done "
        f"neither."
    )


def test_published_distributions_declare_one_ceiling():
    """An open ceiling promises every future interpreter; no CI keeps that."""
    ceilings = {dist: rng[1] for dist, rng in _declared_ranges().items()}
    open_ended = sorted(dist for dist, ceiling in ceilings.items()
                        if ceiling is None)
    assert not open_ended, (
        f"{', '.join(open_ended)} declare(s) requires-python with no upper "
        f"bound, so pip installs them on every interpreter newer than the "
        f"floor -- versions no CI job has ever run. Close the range at the "
        f"next minor after the newest version CI tests (#1076)."
    )
    distinct = set(ceilings.values())
    assert len(distinct) == 1, (
        "the published distributions disagree about their Python ceiling: "
        + ", ".join(
            f"{dist} <{_fmt(ceiling)}"
            for dist, ceiling in sorted(ceilings.items())
        )
        + ". They are installed together, so the effective ceiling is the "
        "lowest of them and the others' metadata is a claim nothing "
        "honours."
    )


@pytest.mark.parametrize("dist", PUBLISHED_DISTS)
def test_every_version_in_the_declared_range_is_tested(dist: str):
    """#1076: the declared range was wider than the one interpreter tested."""
    declared = _declared_ranges()[dist]
    where = f"{dist}/pyproject.toml"
    if declared[1] is None:
        pytest.fail(
            f"{where} declares no ceiling, so its range has untested "
            f"versions without end -- see "
            f"test_published_distributions_declare_one_ceiling."
        )
    tested = _tested_versions()
    untested = [v for v in _versions_in(declared, where) if v not in tested]
    assert not untested, (
        f"{where} declares requires-python "
        f">={_fmt(declared[0])},<{_fmt(declared[1])}, and no "
        f"commit-triggered job that runs pytest installs "
        f"{', '.join(_fmt(v) for v in untested)}. CI tests "
        f"{sorted(_fmt(v) for v in tested)}. Either narrow the declared "
        f"range to what is tested or add the version to a CI test job."
    )


def test_every_ci_pin_satisfies_the_declared_floor():
    """The reverse drift: a pin outside the declared range breaks the install.

    Named for the half it first guarded; it checks both ends.  A pin
    below the floor or at/above the ceiling is one on which pip refuses
    this repo's own editable installs.
    """
    floor = max(_declared_floors().values())
    ceilings = [rng[1] for rng in _declared_ranges().values()
                if rng[1] is not None]
    ceiling = min(ceilings) if ceilings else None
    offenders = {
        version: where
        for version, where in _all_pinned_versions().items()
        if version < floor or (ceiling is not None and version >= ceiling)
    }
    bound = f">={_fmt(floor)}" + (f",<{_fmt(ceiling)}" if ceiling else "")
    assert not offenders, (
        f"requires-python declares {bound}, and these workflow jobs "
        f"install an interpreter outside it, on which pip will refuse "
        f"this repo's own editable installs: "
        + "; ".join(
            f"{_fmt(version)} ({', '.join(where)})"
            for version, where in sorted(offenders.items())
        )
    )


@pytest.mark.parametrize("dist", PUBLISHED_DISTS)
def test_classifiers_do_not_outlive_the_floor(dist: str):
    """pip ignores classifiers, so a stale claim survives a correct range.

    Named for the half it first guarded; it checks the classifiers name
    exactly the versions inside the declared range -- none below the
    floor, none at or above the ceiling, and each version within it.
    """
    project = _pyproject(dist).get("project", {})
    where = f"{dist}/pyproject.toml"
    floor, ceiling = _parse_range(project.get("requires-python", ""), where)
    claimed: Set[Version] = set()
    for classifier in project.get("classifiers", []):
        match = _CLASSIFIER_RE.match(str(classifier).strip())
        if match is not None:
            claimed.add((int(match.group(1)), int(match.group(2))))

    stale = sorted(
        version for version in claimed
        if version < floor or (ceiling is not None and version >= ceiling)
    )
    assert not stale, (
        f"{where} declares requires-python >={_fmt(floor)}"
        + (f",<{_fmt(ceiling)}" if ceiling else "")
        + " and still advertises "
        + ", ".join(_fmt(version) for version in stale)
        + " in its classifiers. pip does not read classifiers, so this is "
        "the copy of the claim that outlives the fix -- PyPI shows it on "
        "the project page."
    )
    expected = (
        _versions_in((floor, ceiling), where) if ceiling else [floor]
    )
    missing = [version for version in expected if version not in claimed]
    assert not missing, (
        f"{where} declares a range containing "
        f"{', '.join(_fmt(v) for v in missing)} and carries no matching "
        f"'Programming Language :: Python :: X.Y' classifier."
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
        target="jaato-server/pyproject.toml",
        find='requires-python = ">=3.14,<3.15"',
        replace='requires-python = ">=3.10,<3.15"',
        because=(
            "one distribution quietly claiming a lower floor than its "
            "siblings is the #1076 shape in miniature -- pip honours the "
            "highest of the four, so the lower claim is metadata nothing "
            "implements and no install fails to reveal it"
        ),
        test="test_published_distributions_declare_one_floor",
    ),
    Reversion(
        target=".github/workflows/codegen-ts-events.yml",
        find='python-version: "3.14"',
        replace='python-version: "3.11"',
        because=(
            "a workflow installing an interpreter below the declared "
            "floor is the reverse drift, and it is the worse one: pip "
            "refuses this repo's own editable installs there, so the "
            "job breaks outright rather than latently"
        ),
        test="test_every_ci_pin_satisfies_the_declared_floor",
    ),
    Reversion(
        target="jaato-tui/pyproject.toml",
        find='    "Programming Language :: Python :: 3.14",\n',
        replace="",
        because=(
            "pip does not read classifiers, so they are where a stale "
            "version claim survives a corrected requires-python -- and "
            "the project page on PyPI is what a reader sees first"
        ),
        test="test_classifiers_do_not_outlive_the_floor",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/tests/test_declared_python_floor_is_tested.py",
        # The trailing newline is what makes this anchor unique: the same
        # text appears in the function above, followed by a REAL newline,
        # while here it is followed by the two characters backslash-n.
        find='    raw = doc.get(True, doc.get("on"))\n',
        replace='    raw = doc.get("on")\n',
        because=(
            "PyYAML reads the bare key `on:` as the boolean True, so "
            "dropping the fallback makes every workflow read as "
            "untriggered -- the scan then finds no tested versions at "
            "all and this guard would go green-and-vacuous, which is the "
            "one failure it must not have"
        ),
        test="test_the_declared_floor_is_a_version_ci_runs_tests_on",
    ),
    Reversion(
        target="jaato-sdk/pyproject.toml",
        find='requires-python = ">=3.14,<3.15"',
        replace='requires-python = ">=3.14"',
        because=(
            "an open ceiling is the half of #1076 the floor change left "
            "behind: `>=3.14` promises 3.15, 3.16 and every interpreter "
            "after them, and no CI job has run any of them"
        ),
        test="test_published_distributions_declare_one_ceiling",
    ),
    Reversion(
        target="jaato-eval/pyproject.toml",
        find='requires-python = ">=3.14,<3.15"',
        replace='requires-python = ">=3.14,<3.16"',
        because=(
            "a ceiling one minor too high declares 3.15 supported while "
            "every CI job installs 3.14 -- a bound that exists but does "
            "not match what is tested is the same promise nobody checks"
        ),
        test="test_every_version_in_the_declared_range_is_tested",
    ),
    Reversion(
        target=".github/workflows/ci-tests.yml",
        # The timeout line is what makes this anchor unique: three jobs in
        # this file pin the same interpreter, and only contract-guards has
        # a 30-minute cap directly above its checkout.
        find=(
            "    timeout-minutes: 30\n"
            "    steps:\n"
            "      - uses: actions/checkout@v5\n"
            "      - uses: actions/setup-python@v6\n"
            "        with:\n"
            '          python-version: "3.14"\n'
        ),
        replace=(
            "    timeout-minutes: 30\n"
            "    steps:\n"
            "      - uses: actions/checkout@v5\n"
            "      - uses: actions/setup-python@v6\n"
            "        with:\n"
            '          python-version: "3.12"\n'
        ),
        because=(
            "one test job left on the old interpreter still leaves 3.14 "
            "tested by the others, so only the pin-within-range check can "
            "see it -- and pip refuses that job's own editable installs"
        ),
        test="test_every_ci_pin_satisfies_the_declared_floor",
    ),
    Reversion(
        target=".github/workflows/deploy-pages.yml",
        find='python-version: "3.14"',
        replace='python-version: "3.15"',
        because=(
            "a workflow pinned at or above the declared ceiling is the "
            "reverse drift at the top end: pip refuses to install the "
            "packages there, which only a ceiling-aware check notices"
        ),
        test="test_every_ci_pin_satisfies_the_declared_floor",
    ),
    Reversion(
        target="jaato-sdk/pyproject.toml",
        find='    "Programming Language :: Python :: 3.14",\n',
        replace=(
            '    "Programming Language :: Python :: 3.14",\n'
            '    "Programming Language :: Python :: 3.15",\n'
        ),
        because=(
            "a classifier above the ceiling advertises an untested "
            "interpreter on the PyPI page while requires-python refuses "
            "it -- the stale claim at the top end of the range"
        ),
        test="test_classifiers_do_not_outlive_the_floor",
    ),
]
