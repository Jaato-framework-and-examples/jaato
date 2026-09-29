"""What CI runs, read from ``.github/workflows/`` and nowhere else.

ONE PARSER, TWO READERS.  ``test_ci_runs_every_test_file.py`` asks this
module which paths the commit-triggered workflows hand to pytest, to prove
every test file is covered.  ``scripts/check.py`` asks it which command
lines to run locally, to give a contributor (or an agent) one command that
runs what CI runs (#1415).  Both read the same functions, so the question
"what does CI run?" has one answer.  A second parser, or a hand-kept list
of commands in ``check.py``, is how the local check would drift from the
workflows, which is the failure #1415 was filed about.

Dependencies: the standard library and PyYAML.  PyYAML is a core
dependency of ``jaato-server``, so any venv the repo's tests run in has
it; it is needed because a workflow file is YAML and the ``on:`` key
alone is enough to make a hand-rolled reader wrong (see :func:`triggers`).

WHAT A LEG IS.  One job, or one expansion of a job's ``strategy.matrix``,
named the way GitHub shows it in the checks list: the job's ``name:`` with
``${{ matrix.* }}`` substituted, or the job id when there is no name.
:func:`legs` returns each leg with its steps' ``run:`` text after the same
substitution, and marks a leg as a *pytest leg* when at least one of its
steps holds a pytest invocation.  Only the pytest legs are something
``check.py`` runs; the Node jobs are reported by name.
"""

from __future__ import annotations

import glob as _glob
import itertools
import re
import shlex
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

#: Flags that consume the NEXT token, so that token is a value and not a
#: path.  ``-m`` is the interesting one -- see :func:`pytest_invocations`.
VALUE_FLAGS = frozenset({
    "-m", "-k", "-p", "-n", "-o", "-c", "-r", "--deselect", "--rootdir",
    "--override-ini", "--maxfail", "--tb", "--junitxml", "--cov",
})


# ---------------------------------------------------------------------------
# Workflow-level
# ---------------------------------------------------------------------------

def triggers(doc: dict) -> object:
    """The workflow's ``on:`` value.

    GOTCHA, and it is the one that makes a reader silently vacuous:
    PyYAML parses the bare key ``on:`` as the BOOLEAN ``True`` (YAML 1.1
    treats ``on``/``off``/``yes``/``no`` as booleans).  Read only
    ``doc["on"]`` and every workflow in this repo looks untriggered, no
    step is ever collected, every file reads as uncovered -- or, with the
    comparison the other way round, every file reads as covered and the
    guard passes forever.
    """
    if "on" in doc:
        return doc["on"]
    return doc.get(True)


def is_commit_triggered(doc: dict) -> bool:
    """True iff this workflow fires on a PR or on a push to a BRANCH.

    ``workflow_dispatch`` is manual, which #736 correctly calls
    "indistinguishable from never" -- two directories were reachable only
    that way and had never gated anything.  A push filtered to ``tags:``
    alone is likewise not commit-triggered for branch work: it fires
    after the merge it should have been able to block.
    """
    on = triggers(doc)
    if on is None:
        return False
    if isinstance(on, str):
        return on == "pull_request" or on == "push"
    if isinstance(on, list):
        return "pull_request" in on or "push" in on
    if not isinstance(on, dict):
        return False
    if "pull_request" in on or "pull_request_target" in on:
        return True
    if "push" not in on:
        return False
    push = on["push"]
    if not isinstance(push, dict):
        return True          # bare `push:` -- every branch
    # `tags:` only, with no branch filter, fires on a release rather than
    # on the commit that needs gating.
    return "branches" in push or "branches-ignore" in push or not push


def commit_triggered_workflows() -> List[tuple]:
    """``(path, parsed_doc)`` for every commit-triggered workflow, sorted."""
    out = []
    for wf in sorted(WORKFLOWS.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(wf.read_text())
        except yaml.YAMLError as exc:      # pragma: no cover
            raise AssertionError(f"{wf.name} is not parseable YAML: {exc}")
        if isinstance(doc, dict) and is_commit_triggered(doc):
            out.append((wf, doc))
    return out


def scalars(node: object) -> List[str]:
    """Every string scalar anywhere in a parsed YAML tree.

    GOTCHA: the suite paths live in ``strategy.matrix.leg[].run`` and
    reach the step as ``${{ matrix.leg.run }}``.  A scan that read only
    step ``run:`` keys would find that literal expression and conclude
    the entire ``suite`` job runs no tests at all.  Recursing over every
    scalar models neither the matrix nor the expression language, and
    needs to model neither -- which is why the coverage guard uses this
    and :func:`legs` (which DOES substitute) is checked against it.
    """
    out: List[str] = []
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            out.extend(scalars(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(scalars(v))
    return out


# ---------------------------------------------------------------------------
# pytest command lines
# ---------------------------------------------------------------------------

def resolve(token: str) -> List[str]:
    """Repo-relative paths a pytest argument names, or ``[]``.

    Existence on disk is the filter.  It keeps ``rc=1``, ``exit``, ``||``
    and every other shell token out without a keyword blacklist that
    would need extending for the next idiom.  A ``::nodeid`` suffix is
    stripped, and a glob (``server/test_*.py``) is expanded -- a glob is
    a legitimate and self-maintaining way to name a set of files.
    """
    token = token.split("::", 1)[0]
    if not token or token.startswith("-"):
        return []
    matches = _glob.glob(str(ROOT / token), recursive=True)
    out: List[str] = []
    for m in matches:
        p = Path(m)
        try:
            rel = p.relative_to(ROOT).as_posix()
        except ValueError:       # pragma: no cover - absolute path outside
            continue
        out.append(rel + "/" if p.is_dir() else rel)
    return out


class Invocation(NamedTuple):
    """One ``pytest`` command line: what it runs, and what it excludes.

    The pair is kept TOGETHER because ``--ignore`` binds to the
    invocation that carries it and to nothing else.  Unioning the
    ignores across invocations makes one leg's exclusion silently cancel
    another leg's coverage (#1080).
    """

    covered: Set[str]
    ignored: Set[str]


def pytest_invocations(command: str) -> List[Invocation]:
    """One :class:`Invocation` per pytest command line in a shell snippet.

    Per LINE rather than per step, because a step legitimately holds
    several invocations -- the `server (daemon tier)` leg runs one pytest
    per tree -- and an ``--ignore`` on the third must not reach the first.
    """
    out: List[Invocation] = []
    for line in command.replace("\\\n", " ").splitlines():
        covered: Set[str] = set()
        ignored: Set[str] = set()
        if "pytest" not in line:
            continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        if "pytest" not in tokens:
            continue
        # GOTCHA: `-m` is both `-m conformance` (a marker, whose value must
        # be skipped) and `python -m pytest` (the invocation itself).
        # Anchoring on the pytest token and reading only what FOLLOWS it
        # settles both.
        rest = tokens[tokens.index("pytest") + 1:]
        pending_ignore = False
        skip_next = False
        for tok in rest:
            if pending_ignore:
                pending_ignore = False
                ignored.update(resolve(tok))
                continue
            if skip_next:
                skip_next = False
                continue
            if tok.startswith(("--ignore=", "--ignore-glob=")):
                ignored.update(resolve(tok.split("=", 1)[1]))
                continue
            if tok in ("--ignore", "--ignore-glob"):
                pending_ignore = True
                continue
            if tok in VALUE_FLAGS:
                skip_next = True
                continue
            if tok.startswith("-"):
                continue
            covered.update(resolve(tok))
        if covered or ignored:
            out.append(Invocation(covered, ignored))
    return out


def covered_invocations() -> List[Invocation]:
    """Every pytest invocation across every commit-triggered workflow.

    Read from every scalar of each document (:func:`scalars`), with no
    model of jobs or matrices.  That independence is what lets the
    ``check.py`` guard compare :func:`legs` against it.
    """
    out: List[Invocation] = []
    for _wf, doc in commit_triggered_workflows():
        for scalar in scalars(doc):
            out.extend(pytest_invocations(scalar))
    return out


# ---------------------------------------------------------------------------
# Jobs and legs, as GitHub names them
# ---------------------------------------------------------------------------

_MATRIX_EXPR = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_.-]+)\s*\}\}")


def _lookup(values: Dict[str, object], dotted: str) -> Optional[object]:
    node: object = values
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def substitute_matrix(text: str, values: Dict[str, object]) -> str:
    """Replace ``${{ matrix.a.b }}`` with its value; leave other ``${{ }}``.

    Only the matrix context is modelled because it is the only one that
    changes WHAT runs.  An expression naming a matrix key that does not
    exist is left as written, so it stays visible rather than becoming
    an empty command.
    """
    def repl(m: "re.Match[str]") -> str:
        val = _lookup(values, m.group(1))
        return m.group(0) if val is None or isinstance(val, (dict, list)) \
            else str(val)
    return _MATRIX_EXPR.sub(repl, text)


def _matrix_combinations(job: dict) -> List[Dict[str, object]]:
    """Every value assignment of ``strategy.matrix``; ``[{}]`` for none.

    The cartesian product of the list-valued keys.  ``include`` and
    ``exclude`` change the product in ways this does not model, so a
    matrix using them is REFUSED rather than approximated: a local check
    that silently runs a different set of legs than CI is the defect.
    """
    matrix = (job.get("strategy") or {}).get("matrix")
    if not matrix:
        return [{}]
    if not isinstance(matrix, dict):
        raise ValueError(f"unsupported matrix shape: {matrix!r}")
    if "include" in matrix or "exclude" in matrix:
        raise ValueError(
            "matrix include/exclude is not modelled; extend "
            "scripts/ci_workflows._matrix_combinations before using it")
    keys = [k for k in matrix]
    lists = []
    for k in keys:
        v = matrix[k]
        if not isinstance(v, list):
            raise ValueError(f"matrix.{k} is not a list: {v!r}")
        lists.append(v)
    return [dict(zip(keys, combo)) for combo in itertools.product(*lists)]


class Step(NamedTuple):
    """One ``run:`` step of a leg, after matrix substitution."""

    name: str
    run: str
    working_directory: Optional[str]


class Leg(NamedTuple):
    """One check as GitHub shows it: a job, or one matrix expansion of it.

    ``name`` is the check's display name (``contract-guards``,
    ``suite (shared/tests)``).  ``test_steps`` are the steps holding a
    pytest invocation -- the part of the leg ``check.py`` runs; the
    install steps before them are the environment, which a local check
    uses as it finds it.
    """

    workflow: str
    job_id: str
    name: str
    steps: List[Step]

    @property
    def test_steps(self) -> List[Step]:
        return [s for s in self.steps if pytest_invocations(s.run)]

    @property
    def is_pytest_leg(self) -> bool:
        return bool(self.test_steps)

    def invocations(self) -> List[Invocation]:
        out: List[Invocation] = []
        for s in self.test_steps:
            out.extend(pytest_invocations(s.run))
        return out


def legs() -> List[Leg]:
    """Every leg of every commit-triggered workflow, in file and job order."""
    out: List[Leg] = []
    for wf, doc in commit_triggered_workflows():
        wf_name = str(doc.get("name") or wf.stem)
        for job_id, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for values in _matrix_combinations(job):
                raw_name = str(job.get("name") or job_id)
                name = substitute_matrix(raw_name, values)
                steps: List[Step] = []
                for i, step in enumerate(job.get("steps") or []):
                    if not isinstance(step, dict) or "run" not in step:
                        continue
                    steps.append(Step(
                        name=substitute_matrix(
                            str(step.get("name") or f"step {i + 1}"), values),
                        run=substitute_matrix(str(step["run"]), values),
                        working_directory=step.get("working-directory"),
                    ))
                out.append(Leg(wf_name, str(job_id), name, steps))
    return out
