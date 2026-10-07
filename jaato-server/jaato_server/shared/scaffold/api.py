"""Stable extension API for third-party (e.g. premium) ``jaato-scaffold`` verbs.

``jaato-scaffold`` ships in jaato-sdk with two built-in verbs (``new`` /
``integration``); jaato-server contributes four more (``explain`` /
``validate`` / ``dependencies`` / ``releases``) through the same seam this
module describes (#1267).  External packages can contribute additional verbs — the canonical
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
* :class:`ScaffoldValidator` / :class:`ValidationRequest` — the seam a package
  uses to add its own ``validate`` findings without contributing a verb.
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
#:
#: ``1.2`` added the ``validate`` VALIDATOR seam (#1306):
#: :data:`VALIDATOR_ENTRY_POINT_GROUP`, :class:`ScaffoldValidator`,
#: :class:`ValidationRequest`, and ``Diagnostic.source``.  Additive — a 1.1
#: verb or topic is unchanged, and ``Diagnostic.as_dict()`` gains the
#: ``source`` key only on a contributed finding.
SCAFFOLD_EXTENSION_API = "1.2"

#: The entry-point group the CLI scans for external verbs.  Defined by the
#: shell that scans it, in jaato-sdk (#1267); re-exported here because this
#: facade is where external verbs have always imported it from.
from jaato_sdk.scaffold.cli import VERB_ENTRY_POINT_GROUP  # noqa: E402,F401

#: The entry-point group the CLI scans for external ``explain`` topics.
#:
#: A VERB is a new subcommand; a TOPIC is a new (or extended) answer from the
#: one subcommand an agent actually reads.  They are separate groups because a
#: package that contributes an engine extension — premium's reactors, gossip,
#: pseudonymization — has something to SAY without having anything to RUN, and
#: the verb seam could only ever have given it a subcommand nobody would think
#: to type.
TOPIC_ENTRY_POINT_GROUP = "jaato.scaffold_topics"

#: The entry-point group ``validate`` scans for contributed validators (#1306).
#:
#: The third seam, and the one a package needs when it ships an ASSET — a file
#: an author writes, which the framework's own checks know nothing about.
#: premium's reactor rules are the case that motivated it: ``explain reactors``
#: tells an author how to write ``reactors.json``, and until this group existed
#: nothing checked the file they then wrote.
VALIDATOR_ENTRY_POINT_GROUP = "jaato.scaffold_validators"

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
class ValidationRequest:
    """Everything a contributed validator is handed, once per ``validate`` run.

    WORKSPACE-scoped (#1306): one call per run, not one per profile.  The asset
    that motivated the seam — premium's reactor rules under
    ``<workspace>/.jaato/reactors/`` — belongs to no profile, and a validator
    that does care about profiles has them all here and iterates itself.  The
    reverse would not work: a profile-scoped call cannot see a file that no
    profile names.

    The fields are what the framework's own checks already had in hand, so a
    contributor judges the same inputs and pays for no second discovery.

    Attributes:
        workspace: Absolute workspace root.
        config_root: Absolute config root (``<workspace>/.jaato`` unless a
            caller overrode it, as the doctor may).
        profile_set: The set being validated (``--set``), or ``None``.
        only: The one profile asked for (``--profile``), or ``None`` for all.
            A contributor that reports per profile should honour it; one that
            checks workspace files may ignore it.
        profiles: The RESOLVED profiles, keyed by name — ``inherits:`` and the
            set overlay already applied, exactly what the daemon would load.
        providers: ``introspect.providers()``, as the framework's checks saw it.
        plugins: ``introspect.plugins()``, likewise.
        gc_names: Installed GC strategy names.
    """

    workspace: str
    config_root: str
    profile_set: Optional[str]
    only: Optional[str]
    profiles: Dict[str, Any]
    providers: Dict[str, Any]
    plugins: Dict[str, Any]
    gc_names: List[str]


@runtime_checkable
class ScaffoldValidator(Protocol):
    """The contract a contributed ``validate`` check must satisfy (#1306).

    An entry point in :data:`VALIDATOR_ENTRY_POINT_GROUP` loads to an instance
    or a zero-arg class/factory producing one, like :class:`ScaffoldVerb`.

    Its findings are merged into the run after the framework's own, each
    stamped ``source = "<distribution>:<name>"`` — the stamp is the
    framework's, not the contributor's, so a finding can never be passed off
    as one of the framework's.  A validator that fails to load, raises, or
    returns something that is not a list of findings is reported as a
    ``validator_unavailable`` / ``validator_failed`` WARNING rather than
    dropped: a check that did not run must not read as a pass.

    Attributes:
        name: Stable identity, unique across installed validators.

    ``validate`` returns a list of :class:`Diagnostic` (anything with
    ``severity`` in ``error`` / ``warn`` / ``info``, a string ``code`` and a
    string ``message``; ``profile`` / ``where`` / ``tier`` optional, ``tier``
    defaulting to ``workspace``).  An ``error`` fails the run like any other.
    """

    name: str

    def validate(self, request: ValidationRequest) -> List[Diagnostic]:
        """Check the workspace; return findings (``[]`` when all is well)."""
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
