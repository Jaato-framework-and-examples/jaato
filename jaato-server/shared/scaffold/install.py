"""``jaato-scaffold install`` — put framework-shipped assets where tools look.

Today that means one asset: the ``jaato-sdk`` skill, the agent-facing guide to
building on this SDK.

WHY THIS VERB EXISTS.  The skill used to be a file in a git repo, so the only
way to get it was to copy it by hand — and hand-copies drift.  A survey of one
org found the same skill living in four repos at four different lengths, and
two large skills whose user-global installs were 2.5 months behind the repo
originals they were copied from.  Nothing detected any of it.

The cure is structural rather than procedural: the skill ships as package data
of the distribution it documents, so an installed copy cannot describe a
different framework than the one running, and every copy this verb writes
carries a stamp naming the version it came from.  ``jaato-doctor`` compares
that stamp against the installed framework and says so when they part company.

Scope is explicit because it decides who sees it:

    --user       ~/.claude/skills/<name>/      every repo on this machine (default)
    --workspace  DIR/.claude/skills/<name>/    that project only

``--user`` is the default because a skill about the framework is not a property
of any one project, and per-project copies are the pattern that produced the
drift above.
"""
from __future__ import annotations

import filecmp
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STAMP = ".installed-from"
"""Filename of the provenance stamp written beside an installed asset.

Read by ``jaato-doctor``.  Its presence is what makes a stale copy detectable
instead of merely wrong.
"""


def framework_version() -> str:
    """The version of the distribution this asset ships with."""
    try:
        from importlib.metadata import version
        return version("jaato-server")
    except Exception:      # noqa: BLE001 — a source checkout may not be installed
        return "unknown"


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "skills"


def available() -> List[str]:
    """Asset names this build ships."""
    root = _source_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())


def target_dir(name: str, *, user: bool, workspace: Optional[str]) -> Path:
    base = Path.home() if user else Path(workspace or ".").resolve()
    return base / ".claude" / "skills" / name


def read_stamp(installed: Path) -> Dict[str, str]:
    """The provenance of an installed copy, or ``{}`` when it has none.

    A copy with no stamp predates this verb — it was hand-copied — which is
    worth reporting rather than treating as absent.
    """
    f = installed / STAMP
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:      # noqa: BLE001 — a corrupt stamp is a missing stamp
        return {}


def compare(name: str, installed: Path) -> Tuple[str, str]:
    """``(state, detail)`` for an installed copy against what this build ships.

    States: ``absent``, ``current``, ``stale`` (a different framework version),
    ``modified`` (same version, edited on disk), ``unstamped``.
    """
    src = _source_root() / name
    if not installed.is_dir():
        return "absent", str(installed)
    stamp = read_stamp(installed)
    if not stamp:
        return "unstamped", "installed by hand — provenance unknown"
    got, want = stamp.get("version", "?"), framework_version()
    if got != want:
        return "stale", f"installed from {got}, framework is {want}"
    if not src.is_dir():
        return "current", got
    diff = filecmp.dircmp(str(src), str(installed))
    changed = list(diff.diff_files) + list(diff.left_only)
    if changed:
        return "modified", f"{len(changed)} file(s) differ from {got}"
    return "current", got


def install(name: str, dest: Path, *, force: bool = False,
            dry_run: bool = False) -> Tuple[bool, List[str]]:
    """Copy ``name`` to ``dest``; return ``(changed, lines)``.

    Refuses to overwrite an existing copy without ``--force``, and says which
    state it found — a local edit and a stale version want different answers
    from the operator, so the message names which one it is.
    """
    src = _source_root() / name
    if not src.is_dir():
        return False, [f"unknown asset '{name}' — this build ships: "
                       f"{', '.join(available()) or '(none)'}"]

    state, detail = compare(name, dest)
    if state != "absent" and not force:
        return False, [f"{dest} already exists ({state}: {detail})",
                       "pass --force to overwrite, or --dry-run to see what would change"]

    files = sorted(p.relative_to(src).as_posix()
                   for p in src.rglob("*") if p.is_file() and p.name != STAMP)
    if dry_run:
        return False, [f"would write {dest}/"] + [f"  + {f}" for f in files]

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    (dest / STAMP).write_text(json.dumps(
        {"asset": name, "version": framework_version(), "source": str(src)},
        indent=2) + "\n", encoding="utf-8")
    return True, [f"installed {name} -> {dest}  (from jaato-server {framework_version()})"] \
        + [f"  + {f}" for f in files]
