#!/usr/bin/env python3
"""Run locally what CI runs, before pushing (#1415).

    python scripts/check.py                # contract-guards + the suite legs the diff touches
    python scripts/check.py --reversions   # ... + the reversion meta-guard, scoped to changed guards
    python scripts/check.py --all          # every pytest leg, as CI runs it
    python scripts/check.py --list         # print the plan and the pre-flight, run nothing

Run it with the interpreter of the venv you develop in (``.venv/bin/python
scripts/check.py``): the leg commands call ``pytest`` / ``pip``, and the
directory of ``sys.executable`` is put first on ``PATH`` so they resolve
to that venv.  Nothing is installed by this script except what a leg's
own command installs (the SDK leg's ``pip install ./jaato-web-coder-server/plugin``).

WHY IT EXISTS.  A session ran the whole ``server/tests`` suite, reported
green, and pushed past the complexity ratchet, which lives in the
required ``contract-guards`` job it never ran.  The list of "which checks
to run" lived only in ``.github/workflows/``, so every session had to be
told it file by file, and that list drifted.

THE COMMANDS ARE READ FROM THE WORKFLOWS.  Every command this script runs
is the ``run:`` text of a workflow step, with ``${{ matrix.* }}``
substituted, obtained from ``scripts/ci_workflows.py`` -- the same parser
``test_ci_runs_every_test_file.py`` uses to prove every test file is run
by CI.  ``test_check_runs_what_ci_runs.py`` asserts the invocations
``--all`` would run equal the ones that parser finds, so the two cannot
disagree.  The only names written here are the two job ids with a role
(``contract-guards`` always runs; ``reversion-guard`` is scoped), and the
script refuses to run if either disappears from the workflows.

WHAT THE DEFAULT MODE SELECTS, AND WHAT IT DOES NOT PROMISE.  A suite leg
is selected when a changed file lies under one of its SCOPES: each
directory the leg hands pytest, widened from ``<pkg>/tests/`` to
``<pkg>/`` so a source change selects the leg that tests it (not widened
to a distribution root that holds another leg's scope, e.g. ``jaato-server/``).
That is a heuristic.  ``--all`` is the authority, and CI is the authority
over ``--all``: the environment differs (CI installs ``[all]`` extras,
has AppArmor, is Linux 3.12).

EXIT STATUS.  0 when every selected leg passed; otherwise the exit status
of the first failing leg, after printing its name exactly as the checks
list shows it.  2 for a usage or environment error.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_workflows as cw  # noqa: E402

ROOT = cw.ROOT

#: Job id that always runs in full: fast (~30-40s) and required.
ALWAYS_JOB = "contract-guards"
#: Job id holding the reversion meta-guard: scoped by ``--reversions``,
#: whole in ``--all``, absent from the default mode (~20 minutes).
REVERSION_JOB = "reversion-guard"

DEFAULT_BASE = "origin/main"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class Command(NamedTuple):
    """One shell command this script runs: a workflow step's ``run:``."""

    step: str
    cwd: str          # repo-relative; "" is the repo root
    run: str


class Planned(NamedTuple):
    """A leg to run, named as the CI checks list names it."""

    name: str
    workflow: str
    commands: List[Command]
    why: str


def _commands(leg: cw.Leg) -> List[Command]:
    return [Command(s.name, s.working_directory or "", s.run)
            for s in leg.test_steps]


def _pytest_legs() -> List[cw.Leg]:
    return [leg for leg in cw.legs() if leg.is_pytest_leg]


def _require_roles(legs: Sequence[cw.Leg]) -> None:
    ids = {leg.job_id for leg in legs}
    missing = [j for j in (ALWAYS_JOB, REVERSION_JOB) if j not in ids]
    if missing:
        raise SystemExit(
            f"check.py: job(s) {missing} are not in any commit-triggered "
            f"workflow any more. Update ALWAYS_JOB / REVERSION_JOB in "
            f"scripts/check.py to the job that now plays that role.")


def leg_scopes(legs: Sequence[cw.Leg]) -> dict:
    """Leg name -> repo-relative path prefixes whose change selects it.

    A file the leg names is its own scope.  A directory ``X/tests/`` is
    widened to ``X/`` so a change to the code it tests selects it -- unless
    ``X/`` is a distribution root (has a ``pyproject.toml``) holding some
    OTHER leg's scope, where widening would make every change to that
    distribution select this leg (``jaato-server/tests/`` would claim all
    of ``jaato-server/``).
    """
    raw: dict = {}
    for leg in legs:
        roots: Set[str] = set()
        for inv in leg.invocations():
            roots |= inv.covered
        raw[leg.name] = roots

    def widened(root: str) -> Optional[str]:
        if root.endswith("/tests/"):
            return root[: -len("tests/")]
        return None

    out: dict = {}
    for leg in legs:
        scopes: Set[str] = set()
        for root in raw[leg.name]:
            wide = widened(root)
            if wide is None:
                scopes.add(root)
                continue
            if (ROOT / wide / "pyproject.toml").is_file() and any(
                    other != leg.name and any(
                        r.startswith(wide) for r in raw[other])
                    for other in raw):
                scopes.add(root)
            else:
                scopes.add(wide)
        out[leg.name] = scopes
    return out


def _touches(changed: Iterable[str], scopes: Iterable[str]) -> List[str]:
    scopes = list(scopes)
    return sorted(
        f for f in changed
        if any(f == s or (s.endswith("/") and f.startswith(s))
               for s in scopes))


def plan(mode: str, changed: Sequence[str] = ()) -> List[Planned]:
    """The legs a mode runs, in workflow order.

    ``all``: every pytest leg of every commit-triggered workflow.
    ``default``: ``contract-guards`` plus the other legs (the reversion
    meta-guard aside) whose scopes a changed file lies under.
    """
    legs = _pytest_legs()
    _require_roles(legs)
    if mode == "all":
        return [Planned(leg.name, leg.workflow, _commands(leg), "--all")
                for leg in legs]
    scopes = leg_scopes(legs)
    out: List[Planned] = []
    for leg in legs:
        if leg.job_id == ALWAYS_JOB:
            out.append(Planned(leg.name, leg.workflow, _commands(leg),
                               "always"))
            continue
        if leg.job_id == REVERSION_JOB:
            continue
        hit = _touches(changed, scopes[leg.name])
        if hit:
            more = f" (+{len(hit) - 1} more)" if len(hit) > 1 else ""
            out.append(Planned(leg.name, leg.workflow, _commands(leg),
                               f"diff touches {hit[0]}{more}"))
    return out


# ---------------------------------------------------------------------------
# Scoped reversion meta-guard
# ---------------------------------------------------------------------------

_DECLARES_REVERSIONS = re.compile(r"^REVERSIONS\s*[:=]", re.MULTILINE)

#: Meta-guard tests that are not per-reversion cases but check every
#: declared reversion up front (nodeid resolution, import failures).
#: Cheap, and exactly the ones that catch a malformed new entry.
_META_PREFLIGHTS = (
    "test_every_reversion_names_a_test_that_exists",
    "test_discovery_imported_every_module_it_walked",
)


def _meta_guard_path() -> str:
    legs = [leg for leg in _pytest_legs() if leg.job_id == REVERSION_JOB]
    files = sorted(r for leg in legs for inv in leg.invocations()
                   for r in inv.covered if r.endswith(".py"))
    if len(files) != 1:
        raise SystemExit(
            f"check.py: expected the {REVERSION_JOB} job to name exactly "
            f"one file, found {files}")
    return files[0]


def _meta_guard_packages(meta: str) -> Tuple[str, ...]:
    """The meta-guard's own ``_PACKAGES``, read from its source."""
    tree = ast.parse((ROOT / meta).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_PACKAGES"
                for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit(f"check.py: {meta} no longer defines _PACKAGES")


def changed_guard_modules(changed: Sequence[str]) -> List[str]:
    """Module names of changed files the meta-guard discovers that
    declare ``REVERSIONS``."""
    meta = _meta_guard_path()
    packages = _meta_guard_packages(meta)
    out = []
    for f in changed:
        p = Path(f)
        if p.parent.as_posix() not in packages or not p.name.startswith(
                "test_") or p.suffix != ".py" or f == meta:
            continue
        path = ROOT / f
        if path.is_file() and _DECLARES_REVERSIONS.search(
                path.read_text(encoding="utf-8", errors="replace")):
            out.append(p.stem)
    return sorted(out)


def _collect_meta_nodeids(meta: str) -> List[str]:
    """Every nodeid of the meta-guard, repo-relative, from a collection pass."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", meta],
        cwd=str(ROOT), capture_output=True, text=True, env=_env())
    # Collected nodeids are relative to pytest's rootdir (``jaato-server/``
    # here, not the repo root), so they are re-anchored on ``meta``.
    marker = Path(meta).name + "::"
    ids = [meta + "::" + line.strip().split(marker, 1)[1]
           for line in proc.stdout.splitlines() if marker in line]
    if proc.returncode != 0 or not ids:
        raise SystemExit(
            f"check.py: collecting {meta} failed (exit {proc.returncode}):\n"
            + (proc.stdout + proc.stderr)[-2000:])
    return ids


def scoped_reversion_plan(changed: Sequence[str]) -> Optional[Planned]:
    """The meta-guard limited to the cases of changed guard modules.

    No hook in the meta-guard is needed: its parametrized ids are
    ``<module>::<test>``, so the nodeids are selected from a collection
    pass.  ``None`` when no changed module declares REVERSIONS.
    """
    modules = changed_guard_modules(changed)
    if not modules:
        return None
    meta = _meta_guard_path()
    ids = _collect_meta_nodeids(meta)
    wanted = [i for i in ids if any(f"[{m}::" in i for m in modules)]
    missing = [t for t in _META_PREFLIGHTS
               if f"{meta}::{t}" not in ids]
    if missing:
        raise SystemExit(
            f"check.py: {meta} no longer has {missing}; update "
            f"_META_PREFLIGHTS in scripts/check.py")
    nodeids = [f"{meta}::{t}" for t in _META_PREFLIGHTS] + wanted
    run = "python -m pytest -q " + " ".join(
        "'" + n.replace("'", "'\\''") + "'" for n in nodeids)
    workflow = next(leg.workflow for leg in _pytest_legs()
                    if leg.job_id == REVERSION_JOB)
    return Planned(
        f"{REVERSION_JOB} (scoped: {', '.join(modules)})", workflow,
        [Command("Every guard detects its own reversion (scoped)", "", run)],
        f"{len(wanted)} case(s) from changed guard module(s)")


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def _git(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                          text=True, check=check)


def changed_files(base: str) -> List[str]:
    """Files that differ from the merge base with *base*: commits on this
    branch, uncommitted edits and untracked files alike -- the check is
    meant to run before the commit as well as after."""
    mb = _git("merge-base", "HEAD", base)
    if mb.returncode != 0:
        raise SystemExit(f"check.py: no merge base with {base}: "
                         f"{mb.stderr.strip()}")
    files = set(_git("diff", "--name-only", mb.stdout.strip()
                     ).stdout.split())
    files |= set(_git("ls-files", "--others", "--exclude-standard"
                      ).stdout.split())
    return sorted(files)


def preflight(base: str, fetch: bool, show_commit: bool) -> List[str]:
    """Print the base report; return the warnings it raised."""
    warnings: List[str] = []
    if fetch:
        remote, _, branch = base.partition("/")
        f = subprocess.run(["git", "fetch", "--quiet", remote, branch],
                           cwd=str(ROOT), capture_output=True, text=True,
                           timeout=120)
        if f.returncode != 0:
            warnings.append(f"could not fetch {base} ({f.stderr.strip()}); "
                            f"comparing against the last fetched copy")
    behind = _git("rev-list", "--count", f"HEAD..{base}")
    ahead = _git("rev-list", "--count", f"{base}..HEAD")
    if behind.returncode == 0:
        n = int(behind.stdout.strip() or 0)
        print(f"base: {base}  (HEAD is {ahead.stdout.strip()} ahead, "
              f"{n} behind)")
        if n:
            warnings.append(
                f"HEAD is {n} commit(s) behind {base}; rebase (or merge) "
                f"before pushing, or CI will test a different tree")
    mt = _git("merge-tree", "--write-tree", "--name-only", "--no-messages",
              "HEAD", base)
    if mt.returncode == 1:
        files = mt.stdout.splitlines()[1:]
        warnings.append(
            f"merging {base} would CONFLICT in: "
            + ", ".join(f for f in files if f) )
    elif mt.returncode != 0:
        warnings.append(f"could not test-merge with {base}: "
                        f"{mt.stderr.strip()} (needs git >= 2.38)")
    if show_commit:
        log = _git("log", "--reverse", "--stat",
                   "--format=%n--- %h %s%n%n%b", f"{base}..HEAD")
        print("\ncommits on this branch (message beside what they change):")
        print(log.stdout.rstrip() or "  (none)")
        dirty = _git("status", "--short").stdout.rstrip()
        if dirty:
            print("\nuncommitted (NOT in any commit above):\n" + dirty)
    for w in warnings:
        print(f"WARNING: {w}")
    return warnings


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _env() -> dict:
    env = dict(os.environ)
    bindir = str(Path(sys.executable).parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    return env


def run_leg(p: Planned) -> int:
    """Run a leg's commands as GitHub runs a ``run:`` step
    (``bash --noprofile --norc -eo pipefail``); return the first non-zero
    exit status, or 0."""
    print(f"\n==> {p.name}   [{p.why}]", flush=True)
    for c in p.commands:
        print(f"--- {c.step}" + (f"  (in {c.cwd})" if c.cwd else ""),
              flush=True)
        start = time.monotonic()
        rc = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", c.run],
            cwd=str(ROOT / c.cwd), env=_env()).returncode
        print(f"--- exit {rc} after {time.monotonic() - start:.0f}s",
              flush=True)
        if rc != 0:
            return rc
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run locally what CI runs (commands read from "
                    ".github/workflows/).")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true",
                      help="every pytest leg, as CI runs it")
    ap.add_argument("--reversions", action="store_true",
                    help="also run the reversion meta-guard, limited to the "
                         "REVERSIONS of changed guard modules")
    ap.add_argument("--list", action="store_true",
                    help="print the pre-flight and the plan; run nothing")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help=f"branch to diff and compare against "
                         f"(default {DEFAULT_BASE})")
    ap.add_argument("--no-fetch", action="store_true",
                    help="do not fetch the base first")
    ap.add_argument("--show-commit", action="store_true",
                    help="print each commit's message beside its --stat, so "
                         "a message describing changes not in the diff "
                         "is visible before pushing")
    args = ap.parse_args(argv)

    preflight(args.base, fetch=not args.no_fetch,
              show_commit=args.show_commit)
    changed = changed_files(args.base)
    print(f"changed vs merge base: {len(changed)} file(s)")

    planned = plan("all" if args.all else "default", changed)
    if args.reversions and not args.all:
        scoped = scoped_reversion_plan(changed)
        if scoped is None:
            print("--reversions: no changed guard module declares "
                  "REVERSIONS; nothing to scope")
        else:
            planned.append(scoped)

    _report_unrun_jobs(changed)
    _print_plan(planned, commands=args.list)
    if args.list:
        return 0
    return _run_all(planned)


def _report_unrun_jobs(changed: Sequence[str]) -> None:
    """Name the jobs with no pytest step that check the code (the Node
    jobs); a gate or a deploy step with no working directory is skipped."""
    for leg in cw.legs():
        if leg.is_pytest_leg:
            continue
        dirs = {s.working_directory.rstrip("/") + "/"
                for s in leg.steps if s.working_directory}
        if not dirs:
            continue
        hit = _touches(changed, dirs)
        note = (f"  <- the diff touches {hit[0]}; run its steps in "
                f"{', '.join(sorted(dirs))}" if hit else "")
        print(f"not run here: {leg.name} (no pytest step){note}")


def _print_plan(planned: Sequence[Planned], commands: bool) -> None:
    print("\nplan:")
    for p in planned:
        print(f"  {p.name:<40} {p.why}")
    if not commands:
        return
    for p in planned:
        print(f"\n# {p.name}")
        for c in p.commands:
            print((f"(cd {c.cwd}) " if c.cwd else "") + c.run.rstrip())


def _run_all(planned: Sequence[Planned]) -> int:
    """Run each leg in order; stop at the first failure and name it."""
    results: List[Tuple[str, float]] = []
    for p in planned:
        start = time.monotonic()
        rc = run_leg(p)
        if rc != 0:
            print(f"\nFAILED: {p.name}    ({p.workflow}; exit {rc})")
            for name, secs in results:
                print(f"  passed: {name} ({secs:.0f}s)")
            return rc
        results.append((p.name, time.monotonic() - start))
    print("\nsummary:")
    for name, secs in results:
        print(f"  passed: {name} ({secs:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
