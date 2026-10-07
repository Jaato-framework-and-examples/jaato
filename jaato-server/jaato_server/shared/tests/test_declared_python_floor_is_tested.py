"""The Python floor a wheel declares must be one CI actually runs tests on.

WHY THIS EXISTS.  Every workflow in this repo pins a single interpreter
-- ``python-version: "3.12"``, no matrix -- while the four published
distributions declared ``requires-python = ">=3.10"``.  So pip installed
them on 3.10, 3.11, 3.13 and 3.14, and **nothing in CI had ever run a
test on any of them** (#1076).  The metadata was a promise the build did
not check, and the failure mode it produces is the expensive one: a
defect whose only trigger is a version CI never runs is invisible until
a user reports it.  The local dev venv was 3.11, so it did not reproduce
in development either.

#1076 offered two honest resolutions -- test the declared range, or
declare the tested range.  This tree took the second: the floor is now
``>=3.12``, the version CI installs.  This guard is what stops the two
drifting apart again, in either direction:

* someone lowers ``requires-python`` (or adds a distribution declaring a
  lower one) and CI keeps testing only 3.12 -- the original defect;
* someone raises the floor above what CI installs -- less obvious and
  worse, because then CI's own ``pip install -e`` is the thing pip
  refuses, so the break is total rather than latent.

WHAT THIS ASSERTS, EXACTLY.  Four properties, and a green run here must
not be read as more than these four sentences say:

1. Every published distribution declares the SAME floor.  They are
   installed into one interpreter, so a disagreement is not a range --
   it is the highest of them, silently.
2. That floor is among the interpreter versions a commit-triggered
   workflow installs for a job that runs ``pytest``.  "Commit-triggered"
   excludes the ``workflow_dispatch``-only publish workflows: a version
   only a manual run touches is, in practice, never run (the finding
   #736 records).  "Runs pytest" excludes jobs that only build or
   deploy.
3. Every interpreter version ANY workflow installs satisfies the
   declared floor -- including the publish workflows, which build the
   wheels.
4. No distribution carries a ``Programming Language :: Python :: X.Y``
   classifier below its own floor, and each carries the floor's own
   version.  A classifier is not machine-enforced by pip, so it is the
   one place the stale claim survives a correct ``requires-python``.

WHAT THIS DOES NOT ASSERT -- and the omission is the unclosed half of
#1076, not an oversight.  ``>=3.12`` has **no upper bound**, so it still
promises 3.13 and 3.14, and CI runs neither.  A guard cannot close that
without either a test matrix or an upper bound, and asserting an upper
bound this project has not decided on would be inventing policy from a
test file.  So: the floor is tested, the ceiling is still a promise
nobody checks.  Read ``requires-python`` accordingly.

It also does not assert that the version CI installs is the version the
tests RAN under -- a job that installs 3.12 and then shells out to a
system ``python3`` would pass here.  Nothing in this tree does that; the
check is a declaration-versus-declaration comparison, which is what
makes it a few milliseconds of YAML and TOML rather than a build.

SCOPE.  ``out-of-tree-plugins/moon-phase`` is deliberately excluded.  It
is an example of what a third party's package looks like, it is never
published from here, and its floor is its author's to choose -- which is
the entire point of the directory.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

_FLOOR_RE = re.compile(r"^>=\s*(\d+)\.(\d+)$")
_CLASSIFIER_RE = re.compile(
    r"^Programming Language :: Python :: (\d+)\.(\d+)$"
)
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")


def _parse_floor(requires_python: str, where: str) -> Version:
    """Return the ``(major, minor)`` floor of a ``requires-python`` value.

    Deliberately narrow.  The specifier grammar allows far more than this
    repo uses (``!=``, ``~=``, multiple clauses), and a parser that
    guessed at those would be asserting something it had not read.  A
    value outside the one shape in use fails by NAME, so whoever widens
    the policy also widens the check.
    """
    match = _FLOOR_RE.match(requires_python.strip())
    if match is None:
        pytest.fail(
            f"{where}: requires-python = {requires_python!r} is not a bare "
            f">=X.Y floor, which is the only shape this guard parses. "
            f"Widening the policy means widening _parse_floor too -- see "
            f"this module's docstring."
        )
    return int(match.group(1)), int(match.group(2))


def _pyproject(dist: str) -> dict:
    path = ROOT / dist / "pyproject.toml"
    assert path.exists(), f"missing {path}"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _declared_floors() -> Dict[str, Version]:
    floors: Dict[str, Version] = {}
    for dist in PUBLISHED_DISTS:
        project = _pyproject(dist).get("project", {})
        requires = project.get("requires-python")
        assert requires, f"{dist}/pyproject.toml declares no requires-python"
        floors[dist] = _parse_floor(requires, f"{dist}/pyproject.toml")
    return floors


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
            # rather than skipping: a matrixed job is precisely the shape
            # that would close the unchecked-ceiling half of #1076, and a
            # guard that ignored it would not notice the day it lands.
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


def test_every_ci_pin_satisfies_the_declared_floor():
    """The reverse drift: a floor above what CI installs breaks the install."""
    floor = max(_declared_floors().values())
    offenders = {
        version: where
        for version, where in _all_pinned_versions().items()
        if version < floor
    }
    assert not offenders, (
        f"requires-python declares a floor of {_fmt(floor)}, and these "
        f"workflow jobs install an older interpreter, on which pip will "
        f"refuse this repo's own editable installs: "
        + "; ".join(
            f"{_fmt(version)} ({', '.join(where)})"
            for version, where in sorted(offenders.items())
        )
    )


@pytest.mark.parametrize("dist", PUBLISHED_DISTS)
def test_classifiers_do_not_outlive_the_floor(dist: str):
    """pip ignores classifiers, so a stale claim survives a correct floor."""
    project = _pyproject(dist).get("project", {})
    floor = _parse_floor(
        project.get("requires-python", ""), f"{dist}/pyproject.toml"
    )
    claimed: Set[Version] = set()
    for classifier in project.get("classifiers", []):
        match = _CLASSIFIER_RE.match(str(classifier).strip())
        if match is not None:
            claimed.add((int(match.group(1)), int(match.group(2))))

    stale = sorted(version for version in claimed if version < floor)
    assert not stale, (
        f"{dist}/pyproject.toml declares requires-python >={_fmt(floor)} "
        f"and still advertises "
        + ", ".join(_fmt(version) for version in stale)
        + " in its classifiers. pip does not read classifiers, so this is "
        "the copy of the claim that outlives the fix -- PyPI shows it on "
        "the project page."
    )
    assert floor in claimed, (
        f"{dist}/pyproject.toml declares requires-python >={_fmt(floor)} "
        f"and carries no 'Programming Language :: Python :: {_fmt(floor)}' "
        f"classifier."
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
        find='requires-python = ">=3.12"',
        replace='requires-python = ">=3.10"',
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
        find='python-version: "3.12"',
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
        find='    "Programming Language :: Python :: 3.12",\n',
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
]
