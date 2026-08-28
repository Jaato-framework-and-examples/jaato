"""Fixture materialisation — one hermetic workspace per arm.

Every arm gets a fresh copy of the task's fixture tree.  Without this, N
repeats of the same task contaminate each other (the second run finds the
first run's edits already applied and "passes" trivially), and a sweep
across profile sets measures the order the arms happened to run in.

The workspace is the *mutable* half.  The task definition (``.jaato/``)
stays where it is and is passed as ``config_root``, read-only from the
agent's point of view.  That split is what makes the environment
reproducible: the agent cannot edit the profiles that configure it.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

#: Names never copied out of a fixture tree.  A fixture carrying its own
#: ``.git`` would make every arm a repo and confuse workspace tooling.
_EXCLUDE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".DS_Store")


class FixtureError(RuntimeError):
    """Raised when a workspace cannot be materialised.

    Always a harness fault, never a fault of the agent under test — the
    caller must translate it into a BLOCKED verdict, not a FAIL.
    """


@dataclass(frozen=True)
class Workspace:
    """A materialised scratch workspace for exactly one arm.

    Attributes:
        path: Root of the copied fixture.  Sent to the daemon as
            ``workspace_path``; graders inspect it after the run.
        env_file: The ``.env`` written into ``path``.  Carries
            ``JAATO_PROFILE_SET`` for this arm, which is how the
            model/provider axis of the sweep reaches profile discovery.
    """

    path: Path
    env_file: Path


def materialise(fixture: Path, dest: Path, *,
                profile_set: Optional[str] = None) -> Workspace:
    """Copy ``fixture`` to ``dest`` and write the arm's ``.env``.

    Args:
        fixture: Source tree, left untouched.
        dest: Target directory.  Must not already exist — refusing to
            overwrite is deliberate: silently reusing a dirty workspace is
            the contamination this module exists to prevent.
        profile_set: Written as ``JAATO_PROFILE_SET``.  Selects
            ``<config_root>/profiles/<set>/`` at profile discovery.  This
            is the ONLY thing the engine writes into an arm's ``.env``,
            because it is the one value the engine owns: it is the sweep's
            axis, not a property of the task.

            An ``env`` parameter used to sit here, offering arbitrary extra
            pairs and citing ``VLLM_HOST`` as the case.  Nothing ever
            passed it, and it should not have existed: ``env`` is a
            ``SubagentProfile`` field, so a self-hosted arm declares its
            host in the profile that binds that provider — beside the
            model and the base URL it belongs with, rather than in a second
            place the engine would have to merge.

    Returns:
        The materialised :class:`Workspace`.

    Raises:
        FixtureError: if the fixture is missing or ``dest`` already exists.
    """
    if not fixture.is_dir():
        raise FixtureError(f"fixture is not a directory: {fixture}")
    if dest.exists():
        raise FixtureError(
            f"workspace already exists: {dest} — refusing to reuse a dirty "
            "workspace, since a stale edit would make the next arm pass for "
            "the wrong reason")

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(fixture, dest, ignore=_EXCLUDE, symlinks=True)
    except OSError as exc:
        raise FixtureError(f"could not copy {fixture} -> {dest}: {exc}") from exc

    lines = []
    if profile_set:
        lines.append(f"JAATO_PROFILE_SET={profile_set}")

    env_file = dest / ".env"
    # An arm with no profile_set and no extra env still gets an empty
    # .env: open_session(env_file=".env") resolves it relative to the
    # workspace, and a missing file there is not distinguishable from an
    # empty one at the point where it would matter.
    env_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    return Workspace(path=dest, env_file=env_file)


def discard(workspace: Workspace) -> None:
    """Remove a materialised workspace.

    Best-effort: a workspace that cannot be removed (a file left locked by
    a tool the agent spawned) is not worth failing an otherwise-complete
    arm over.  Callers that need the workspace preserved for inspection
    simply do not call this.
    """
    shutil.rmtree(workspace.path, ignore_errors=True)
