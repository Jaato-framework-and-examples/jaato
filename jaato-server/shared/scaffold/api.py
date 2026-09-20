"""Stable extension API for third-party (e.g. premium) ``jaato-scaffold`` verbs.

``jaato-scaffold`` ships three built-in verbs (``explain`` / ``validate`` /
``new``).  External packages can contribute additional verbs — the canonical
case being the premium ``compile`` verb (the Daruma invariant compiler) — by
registering a :class:`ScaffoldVerb` under the ``jaato.scaffold_verbs`` entry-point
group.  The public CLI discovers and mounts them at startup; a verb whose package
is not installed simply does not appear (mirrors the ``jaato.premium`` and
``jaato.extensions`` convention used elsewhere in the framework; there is
no ``jaato.premium_reactors`` group — see ``__main__.py``).

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
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

# Re-exported internals (the reusable, stable-enough surface) --------------- #
from . import introspect  # noqa: F401  (re-export)
from .validate import Diagnostic, validate_workspace  # noqa: F401  (re-export)

#: Bump on any backwards-incompatible change to the names re-exported here.
#: An external verb can compare against this to fail loud on a version skew.
#:
#: ``1.1`` added the ``explain`` TOPIC seam beside the verb seam:
#: :data:`TOPIC_ENTRY_POINT_GROUP`, :class:`ExplainTopic`, :class:`TopicRequest`
#: and :data:`Rendered`.  Additive — every 1.0 verb is unchanged.
SCAFFOLD_EXTENSION_API = "1.1"

#: The entry-point group the CLI scans for external verbs.
VERB_ENTRY_POINT_GROUP = "jaato.scaffold_verbs"

#: The entry-point group the CLI scans for external ``explain`` topics.
#:
#: A VERB is a new subcommand; a TOPIC is a new (or extended) answer from the
#: one subcommand an agent actually reads.  They are separate groups because a
#: package that contributes an engine extension — premium's reactors, gossip,
#: pseudonymization — has something to SAY without having anything to RUN, and
#: the verb seam could only ever have given it a subcommand nobody would think
#: to type.
TOPIC_ENTRY_POINT_GROUP = "jaato.scaffold_topics"

#: What every ``explain`` renderer returns: ``(structured_data, human_text)``.
#: ``--json`` prints the first, a terminal prints the second, and a topic that
#: returns only one of them is only half an answer — an agent reads the JSON.
Rendered = Tuple[Dict[str, Any], str]


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
class TopicRequest:
    """Everything an ``explain`` renderer is handed, whatever its shape.

    ONE signature for every topic — a new one and one that extends a built-in
    — because the alternative is the caller having to know each topic's
    calling convention before it can call it.  The built-in table needs five
    (``simple`` / ``filter`` / ``named`` / ``workspace`` / ``optional_named``)
    for historical reasons; an external topic reads the fields it cares about
    and ignores the rest, and gains nothing to update when a sixth is added.

    Attributes:
        topic: The topic asked for.  For an extension this is the BUILT-IN's
            name (what the reader typed), not the extension's own — an
            extension that appends to two topics needs to know which one it is
            answering.
        name: The topic argument, or ``None``.  A topic that REQUIRES one says
            so by returning a data dict carrying an ``error`` key; the CLI
            turns that into a stderr message and exit 2, the same contract a
            built-in ``named`` scope has, so a reader who typo'd a name never
            mistakes the miss for documentation.
        workspace: The ``--workspace`` value, defaulted to ``"."``.  Always
            present, so a topic that reads the workspace never has to ask
            whether it was given one.
    """

    topic: str
    name: Optional[str] = None
    workspace: str = "."


@runtime_checkable
class ExplainTopic(Protocol):
    """The contract an external ``explain`` topic must satisfy.

    An entry point in :data:`TOPIC_ENTRY_POINT_GROUP` loads to either an
    instance or a zero-arg class/factory producing one, exactly like
    :class:`ScaffoldVerb`.

    Two shapes, decided by :attr:`extends`:

    * ``extends = ""`` — a topic of its OWN.  ``jaato-scaffold explain <name>``
      renders it, and it appears in the overview banner, in ``--help`` and in
      the unknown-scope error, because all three are derived from the same
      merged table that dispatches it.  A name a built-in already holds is
      REFUSED with a warning (the verb seam's rule: the framework's own answer
      about its own subject cannot be replaced by a package that happens to be
      installed).
    * ``extends = "<built-in topic>"`` — an attributed SECTION appended to that
      topic.  The built-in renders first and unchanged; the section's text
      follows under a header naming its contributor, and its data lands at
      ``data["extensions"][<name>]`` — never merged into the built-in's own
      keys, so a contributed section cannot silently redefine what a documented
      key means.  This is the half that matters for a subsystem whose FILES
      live in a tree the built-in already describes: premium's reactor rules
      are read from ``<workspace>/.jaato/reactors/``, and the place a reader
      looks for that is ``explain paths``, not a topic they have not heard of.

    Attributes:
        name: The topic name (own topic), or the section's identity (extension).
        help: The one-line ``# ...`` blurb the overview banner prints beside an
            own topic.  Empty for a topic whose name already says what it is.
        arg: The argument hint rendered beside an own topic (``"<name>"``,
            ``"[<filter>]"``, ...).  Empty when the topic takes none.
        extends: The built-in topic this appends to, or ``""`` for an own topic.
        reads_workspace: Whether the banner appends ``[--workspace DIR]`` to
            this topic's line.  Declared rather than inferred: every topic is
            HANDED the workspace, so nothing but the topic itself knows whether
            it reads it.
    """

    name: str
    help: str
    arg: str
    extends: str
    reads_workspace: bool

    def render(self, request: TopicRequest) -> Rendered:
        """Render this topic; return ``(structured_data, human_text)``."""
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
            raise FileExistsError(f"{dest} exists")
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
