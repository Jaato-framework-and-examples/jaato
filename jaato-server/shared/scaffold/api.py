"""Stable extension API for third-party (e.g. premium) ``jaato-scaffold`` verbs.

``jaato-scaffold`` ships three built-in verbs (``explain`` / ``validate`` /
``new``).  External packages can contribute additional verbs — the canonical
case being the premium ``compile`` verb (the Daruma invariant compiler) — by
registering a :class:`ScaffoldVerb` under the ``jaato.scaffold_verbs`` entry-point
group.  The public CLI discovers and mounts them at startup; a verb whose package
is not installed simply does not appear (mirrors the ``jaato.premium`` /
``jaato.premium_reactors`` convention used elsewhere in the framework).

This module is the **primary** stable surface for an external verb — the seam
(the CLI mount + the generic emit/validate plumbing).  Keep it stable: internals
under :mod:`shared.scaffold` may churn, but what is re-exported here is the
contract.  ``SCAFFOLD_EXTENSION_API`` is bumped when that contract changes so an
external verb can declare the minimum it needs.

It is **not** necessarily the *only* jaato surface a verb touches, and honesty
about that matters for the compat/version story.  A verb that generates or
validates jaato *assets* also depends on the framework's own stable asset
contracts it targets — those are legitimate, not a leak:

* the **evaluator contract**, :mod:`shared.plugins.permission.evaluator`
  (``PolicyDecision`` / ``EvalResult`` / ``load_evaluators`` / ``run_evaluator``):
  a generated permission evaluator *must* import the real ``PolicyDecision`` /
  ``EvalResult`` (that is the runtime contract of an evaluator), and a verb that
  validates its output loads it through the real ``load_evaluators`` /
  ``run_evaluator``;
* the **script loader**, :func:`shared.script_loader.load_script_symbol`, used to
  load an emitted script (evaluator / processor / reactor action) through the
  framework the way the daemon would.

So the real compatibility surface for such a verb is *this facade* **plus**
whichever framework asset contracts it emits or validates.  Pin against
``SCAFFOLD_EXTENSION_API`` for the seam, and against the jaato-server version that
carries those asset contracts for the rest.

What a verb gets *here* (the seam):

* :class:`ScaffoldVerb` — the protocol the CLI expects (name / help / configure /
  run).
* :class:`GeneratedFile` + :func:`write_files` + :func:`emit_then_validate` — the
  generic emit-then-validate plumbing (no domain logic), so an external generator
  can write a tree and run it straight back through the framework validator,
  exactly like the built-in ``new`` verb.
* :func:`validate_workspace` / :class:`Diagnostic` and the :mod:`introspect`
  module — the reusable validation + framework-introspection internals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple, runtime_checkable

# Re-exported internals (the reusable, stable-enough surface) --------------- #
from . import introspect  # noqa: F401  (re-export)
from .validate import Diagnostic, validate_workspace  # noqa: F401  (re-export)

#: Bump on any backwards-incompatible change to the names re-exported here.
#: An external verb can compare against this to fail loud on a version skew.
SCAFFOLD_EXTENSION_API = "1.0"

#: The entry-point group the CLI scans for external verbs.
VERB_ENTRY_POINT_GROUP = "jaato.scaffold_verbs"


@runtime_checkable
class ScaffoldVerb(Protocol):
    """The contract a CLI-mounted verb must satisfy.

    An entry point in :data:`VERB_ENTRY_POINT_GROUP` loads to either an instance
    or a zero-arg class/factory producing one.  The CLI reads :attr:`name` /
    :attr:`help`, calls :meth:`configure` to let the verb register its own
    arguments on a fresh subparser, then dispatches :meth:`run`.

    Attributes:
        name: The subcommand name (e.g. ``"compile"``).
        help: One-line help shown in ``jaato-scaffold --help``.
    """

    name: str
    help: str

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Register this verb's arguments on its subparser."""
        ...

    def run(self, args: argparse.Namespace) -> int:
        """Execute the verb; return a process exit code."""
        ...


@dataclass(frozen=True)
class GeneratedFile:
    """One emitted artifact: a path relative to the output root + its content."""

    path: str
    content: str


def write_files(
    files: List[GeneratedFile], out_dir: str | Path, *, force: bool = False,
) -> List[str]:
    """Write emitted files under ``out_dir``; return the relative paths written.

    Raises :class:`FileExistsError` if a target exists and ``force`` is False.
    """
    out = Path(out_dir)
    written: List[str] = []
    for f in files:
        dest = out / f.path
        if dest.exists() and not force:
            raise FileExistsError(f"{dest} exists (use force=True to overwrite)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f.content, encoding="utf-8")
        written.append(f.path)
    return written


def emit_then_validate(
    files: List[GeneratedFile],
    out_dir: str | Path,
    *,
    force: bool = False,
    profile_set: Optional[str] = None,
) -> Tuple[List[str], List[Diagnostic]]:
    """Write ``files`` then run the framework validator over the result.

    The same emit-then-validate discipline the ``new`` verb uses: whatever a verb
    emits is validated by construction.  Returns ``(written_paths, diagnostics)``;
    inspect ``[d for d in diagnostics if d.severity == "error"]`` for failures.
    """
    written = write_files(files, out_dir, force=force)
    diags = validate_workspace(str(Path(out_dir).resolve()), profile_set=profile_set)
    return written, diags
