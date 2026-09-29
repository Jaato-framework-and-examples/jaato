"""``scripts/check.py --all`` runs exactly the pytest invocations CI runs (#1415).

WHY THIS EXISTS.  ``check.py`` is the one local command meant to answer
"did I run what CI will run?".  A session that had run 2,048 tests and
reported green pushed past the complexity ratchet, because the ratchet
lives in the ``contract-guards`` job and nobody had told it to run that.
A local check is only worth running if it cannot quietly run something
else, so this asserts the two readings of ``.github/workflows/`` agree:

* :func:`ci_workflows.covered_invocations` -- every pytest command line in
  every commit-triggered workflow, found by scanning every YAML scalar
  with no model of jobs or matrices.  It is what
  ``test_ci_runs_every_test_file.py`` proves covers every test file.
* the commands ``check.plan("all")`` would execute, which go through the
  job/matrix model (:func:`ci_workflows.legs`) and substitution of
  ``${{ matrix.* }}``, and are parsed back into invocations here.

The comparison is of MULTISETS of ``(covered, ignored)`` pairs: a leg
dropped, a matrix expansion that substitutes nothing, an ``--ignore``
moved to the wrong invocation, or a leg run twice all make them differ.

What it does NOT assert: that the local ENVIRONMENT matches CI's (CI
installs ``[all]`` extras and has AppArmor), nor that the default mode's
diff-based selection is complete.  ``--all`` is the claim; this is its
guard.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _load_script(name: str):
    """Import ``scripts/<name>.py`` from the tree this file lives in.

    By path, so inside the reversion meta-guard's sandbox the sandbox's
    copy is the one read.  ``check.py`` imports ``ci_workflows`` by name,
    so ``scripts/`` goes on ``sys.path`` first -- and any cached copy
    from another tree is dropped.
    """
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    for cached in ("ci_workflows",):
        mod = sys.modules.get(cached)
        if mod is not None and Path(mod.__file__).parent != ROOT / "scripts":
            del sys.modules[cached]
    spec = importlib.util.spec_from_file_location(
        f"_jaato_check_guard_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _key(inv) -> tuple:
    return (tuple(sorted(inv.covered)), tuple(sorted(inv.ignored)))


def _planned_invocations(check, cw):
    out = []
    for planned in check.plan("all"):
        for command in planned.commands:
            out.extend(cw.pytest_invocations(command.run))
    return out


def test_check_all_runs_the_invocations_the_workflows_declare():
    check = _load_script("check")
    cw = check.cw

    declared = Counter(_key(i) for i in cw.covered_invocations())
    planned = Counter(_key(i) for i in _planned_invocations(check, cw))

    assert declared, (
        "the workflow scan found no pytest invocation at all, so this "
        "comparison would pass vacuously -- see "
        "test_ci_runs_every_test_file's self-check")
    missing = declared - planned
    extra = planned - declared
    assert not missing and not extra, (
        "scripts/check.py --all does not run what the workflows run.\n"
        f"  in the workflows, not run by check.py: "
        f"{sorted(missing.elements())}\n"
        f"  run by check.py, not in the workflows: "
        f"{sorted(extra.elements())}\n"
        "check.py must take its commands from scripts/ci_workflows.py "
        "(jobs, matrix expansion, run: text), never from a list of its own")


def test_check_names_legs_as_the_checks_list_does():
    """A failure must map one-to-one onto a check a reviewer sees.

    GitHub names a matrix leg ``<job name>`` with ``${{ matrix.* }}``
    substituted; a literal ``${{`` in a leg name means the substitution
    did not happen and the name printed on failure would match nothing.
    """
    check = _load_script("check")
    names = [p.name for p in check.plan("all")]
    assert check.ALWAYS_JOB in names
    assert check.REVERSION_JOB in names
    assert any(n.startswith("suite (") for n in names), names
    assert not [n for n in names if "${{" in n], names
    assert len(names) == len(set(names)), f"duplicate leg names: {names}"


def test_default_mode_always_runs_contract_guards_and_scopes_the_rest():
    check = _load_script("check")
    none = [p.name for p in check.plan("default", [])]
    assert none == [check.ALWAYS_JOB], (
        f"with no change, only {check.ALWAYS_JOB} should run; got {none}")

    touched = [p.name for p in check.plan(
        "default", ["jaato-server/jaato_server/server/core.py"])]
    assert check.ALWAYS_JOB in touched
    assert any("daemon tier" in n for n in touched), (
        "a change to server/core.py must select the leg whose tests "
        f"cover server/; got {touched}")
    assert check.REVERSION_JOB not in touched, (
        "the whole meta-guard (~20 min) is --all only; the default mode "
        "scopes it via --reversions")


# ---------------------------------------------------------------------------
# Reversions -- see test_every_guard_detects_its_own_reversion.
# ---------------------------------------------------------------------------
from jaato_server.shared.tests.reversion import Reversion  # noqa: E402

REVERSIONS = [
    Reversion(
        target="scripts/check.py",
        find="""    if mode == "all":
        return [Planned(leg.name, leg.workflow, _commands(leg), "--all")
                for leg in legs]""",
        replace="""    if mode == "all":
        return [Planned(leg.name, leg.workflow, _commands(leg), "--all")
                for leg in legs if leg.job_id != REVERSION_JOB]""",
        because=(
            "--all silently leaving a job out is the drift #1415 is about: "
            "the local check would report green on a leg CI still runs"
        ),
        test="test_check_all_runs_the_invocations_the_workflows_declare",
    ),
    Reversion(
        target="scripts/ci_workflows.py",
        find="    return _MATRIX_EXPR.sub(repl, text)",
        replace="    return text",
        because=(
            "without matrix substitution every suite leg's step is the "
            "literal ${{ matrix.leg.run }}, so --all would run none of "
            "the suite and name its legs with a raw expression"
        ),
        test="test_check_all_runs_the_invocations_the_workflows_declare",
    ),
]
