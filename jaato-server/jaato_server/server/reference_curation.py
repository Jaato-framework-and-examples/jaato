"""Promote or dismiss a reference CLAIM: the half of the write path an agent cannot do.

An agent proposes a reference with ``proposeReference``, which writes a
claim under ``<workspace>/.jaato/references-claims/``
(:mod:`jaato_server.shared.plugins.references.claims`).  It never writes the
catalog: every AppArmor body is ``audit deny ... wlk`` on
``<workspace>/.jaato/references/**``.  This module is the other half, run by
the DAEMON for the person on a connection (``reference.promote`` /
``reference.dismiss``, protocol 1.33), and it is the only in-tree code that
turns a claim into a catalog entry.

What a promotion does, in order, and why each step is here:

1. **The owner gate.**  :func:`~.memory_verbs.may_curate` -- the rule the
   memory rail uses (#1232): the workspace owner may, anyone may on an
   unowned workspace, a connection with no identity may not on an owned one.
   The identity is the transport's, never the request's.
2. **The claim is re-read, not trusted.**  The claims directory is
   model-writable, so the file may not be what ``proposeReference`` wrote.
   It is read without following a link out of the workspace
   (:func:`~.contained_write.contained_dir`, a symlinked claim file is
   refused), re-checked with :func:`~...claims.is_claim`, and its entry is
   run back through :func:`~...claims.build_proposed_reference` -- the one
   door a proposal passed -- which re-checks the id, the tags, that a
   ``path`` still resolves to a file INSIDE the workspace, and that the id
   is not already in the catalog.
3. **The origin is re-stamped where the daemon can observe it.**
   ``generated_by`` is kept as the claim recorded it (it is what the
   curator saw in the listing, and the daemon cannot re-derive a model
   binding for a turn that is over).  ``created_by`` is NOT kept: it is
   derived again from the session the claim names, and only when that
   session is known to have run in this workspace
   (:meth:`SessionManager.creator_in_workspace`); otherwise it is absent.
   ``curated_by`` is the person the transport authenticated, and ``at`` is
   the instant the reference arrived in the catalog.
4. **The write never goes through a link.**  ``write_contained`` (#1386)
   writes ``<workspace>/.jaato/references/<id>.json``; a destination that
   already exists is a collision, never an overwrite.
5. **The claim is removed** once the entry is written.  A claim that cannot
   be removed afterwards is reported, and the next promotion of it answers
   ``collision`` rather than writing a second copy.

**A revision claim** (``revises``, #1437) is promoted by
:func:`_promote_revision` instead: the catalog file is found the way the
loader reads the catalog, its digest compared with the one the claim
recorded (``stale`` on a mismatch, nothing written), and the file replaced
in place through ``write_contained`` -- origin, mode and every other key
kept, a ``revisions[]`` record appended -- in whichever bundle it lives,
whose index is then reconciled.

A local ``path`` is stored in the claim workspace-relative and rewritten
relative to the catalog file (``../../docs/x.md``), because the catalog
loader resolves a relative path against the reference file's own
directory.

**Into a named bundle, and its index.**  A promotion may name a
workspace-tier sub-bundle (``bundle``): the entry is written into that
bundle's directory instead of the catalog root.  Either way, when the
destination bundle declares a vector index (``embedding_config.json``), the
new entry has no row in it yet, so the daemon reconciles that index
(:func:`reconcile_destination`).  The daemon does the writing because on a
confined host it is the only process that may write ``.jaato/references/**``
-- every runner body denies it, the base profile included, and in-process
tools run in the base profile.  The embedding MODEL is in the runner, so
the vectors come from the caller's session over ``session.embed_texts``
(:class:`RunnerEmbeddingProvider`).  The reference is placed whatever the
reconcile's outcome, which is reported, never silent.  This split is a
stopgap: once in-process tools run in a real ``tool_hat`` and the base
profile can let the plugin write its own catalog, the reconcile belongs
back in the runner (#1422, which lists what to remove).

Stdlib plus the references plugin's own helpers; no daemon state.  The
caller hands in the owner, the identity and a creator lookup.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from jaato_server.shared.plugins.references.bundle import (
    BUNDLE_TIER_WORKSPACE,
    REFERENCE_NON_SOURCE_FILENAMES,
    ROOT_BUNDLE_NAME,
    ReferenceBundle,
    discover_bundles,
    load_reference_bundle,
)
from jaato_server.shared.plugins.references.claims import (
    CLAIMS_DIRNAME,
    INLINE_CLAIM_MAX_CHARS,
    REVISES_KEY,
    build_claim_entry,
    build_proposed_reference,
    claim_as_args,
    claim_tags,
    current_fields,
    is_claim,
    is_revision,
    link_warnings,
    pending_claim_ids,
    rendered_stamp_now,
    revision_staleness,
    revision_target,
    valid_id,
)
from jaato_server.shared.plugins.references.config_loader import discover_references
from jaato_server.shared.plugins.references.embedding_types import EmbeddingResult
from jaato_server.shared.plugins.references.models import (
    ORIGIN_AGENT,
    ReferenceOrigin,
)

from .contained_write import PathLeavesRoot, contained_dir, write_contained
from jaato_server.shared.workspace_ownership import inherit_owner_tree
from .memory_verbs import may_curate

logger = logging.getLogger(__name__)

#: The two verbs, by the command name a client sends.
CURATION_COMMANDS: Dict[str, str] = {
    "reference.promote": "promote",
    "reference.dismiss": "dismiss",
}

#: Where the catalog lives, relative to the workspace (the workspace tier
#: ``discover_references`` reads by default).
CATALOG_REL = ".jaato/references"

#: Where claims live, relative to the workspace.
CLAIMS_REL = f".jaato/{CLAIMS_DIRNAME}"

#: ``texts -> answer`` -- the caller's session embedding texts
#: (``JaatoServer.embed_texts``): ``{"ok", "model", "dimensions", "vectors"}``
#: or ``{"ok": False, "category", "error"}``.
Embed = Callable[[List[str]], Dict[str, Any]]

#: What :func:`reconcile_destination` reports, for ``reconcile`` on the
#: answer: ``none`` (the bundle has no vector index), ``updated``,
#: ``clean``, ``busy`` (another reconcile holds the lock), ``unavailable``
#: (no session, no provider, a model that differs from the index's, or no
#: numpy here), ``error``.
RECONCILE_OUTCOMES = ("none", "updated", "clean", "busy", "unavailable", "error")


@dataclass
class CurationOutcome:
    """What one curation verb did.

    Attributes:
        ok: Whether the verb did what was asked.
        action: ``promote`` or ``dismiss``.
        claim_id: The claim acted on, as the caller named it.
        category: ``""`` on success, else one of ``invalid_request``,
            ``not_owner``, ``unknown_bundle``, ``not_found``,
            ``invalid_claim``, ``collision``, ``stale`` (a revision written
            against a version that has since changed), ``ambiguous`` (the
            revised id is in two catalog files), ``unsafe_path``,
            ``io_error``.  A client branches on this, not on ``error``.
        error: The reason, for a person.
        reference_id: The catalog id promoted (promote only).
        reference_file: The workspace-relative catalog file written.
        warnings: Things that happened beside success (the claim file could
            not be removed after a promotion; the index was not updated).
        bundle: The bundle promoted into, ``""`` for the catalog root.
        reconcile: The destination index's reconcile outcome
            (:data:`RECONCILE_OUTCOMES`), ``""`` when nothing was promoted.
        reconcile_detail: Why, when it is not ``updated`` / ``clean`` /
            ``none``.
        revised: ``True`` when the claim was a revision and the catalog
            file was replaced in place rather than created.
    """

    ok: bool
    action: str
    claim_id: str
    category: str = ""
    error: str = ""
    reference_id: str = ""
    reference_file: str = ""
    warnings: List[str] = field(default_factory=list)
    bundle: str = ""
    reconcile: str = ""
    reconcile_detail: str = ""
    revised: bool = False


def curator_stamp(user_id: Optional[str]) -> Dict[str, Any]:
    """The ``curated_by`` a promotion records.

    The person the transport authenticated.  ``user`` is omitted, not set
    to ``None``, when the connection carries no identity (an unowned
    workspace promoted over an identity-less connection), so the stamp
    claims nothing it did not observe.
    """
    stamp: Dict[str, Any] = {"kind": "human", "via": "reference.promote"}
    if user_id:
        stamp["user"] = user_id
    return stamp


def catalog_ids(root: str) -> Set[str]:
    """Every ``id`` in the workspace catalog, sub-bundles included.

    Walks ``<root>/.jaato/references`` without following links (a link out
    of the workspace names nothing the catalog loader would load here).  An
    unreadable file contributes nothing: a promotion that collides with an
    id nobody can read is refused later by the existence check on the
    destination file, which is the one collision that would overwrite.
    """
    try:
        base = contained_dir(root, CATALOG_REL, create=False)
    except PathLeavesRoot:
        return set()
    ids: Set[str] = set()
    if base is None:
        return ids
    for dirpath, _dirnames, filenames in os.walk(base, followlinks=False):
        for name in filenames:
            if not name.endswith(".json") or name in REFERENCE_NON_SOURCE_FILENAMES:
                continue
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and isinstance(data.get("id"), str):
                ids.add(data["id"])
    return ids


def _read_claim(root: str, claim_id: str) -> "tuple[Optional[str], Any, str, str]":
    """``(claim_path, data, category, error)`` -- the claim, or why not.

    The claims directory is resolved component by component inside the
    workspace, and a claim FILE that is a symlink is refused: it is a file
    a model could have planted to point the daemon's read (and its later
    ``unlink``) anywhere.
    """
    try:
        directory = contained_dir(root, CLAIMS_REL, create=False)
    except PathLeavesRoot as exc:
        return None, None, "unsafe_path", str(exc)
    path = os.path.join(directory, f"{claim_id}.json") if directory else ""
    if not directory or not os.path.lexists(path):
        return None, None, "not_found", f"no claim '{claim_id}' in {CLAIMS_REL}"
    if os.path.islink(path) or not os.path.isfile(path):
        return None, None, "unsafe_path", f"{CLAIMS_REL}/{claim_id}.json is not a regular file"
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, None, "invalid_claim", f"claim '{claim_id}' is unreadable: {exc}"
    if not is_claim(data) or data.get("claim_id") != claim_id:
        return None, None, "invalid_claim", f"claim '{claim_id}' is not a proposed-reference claim"
    return path, data, "", ""


def promoted_origin(
    claim: Dict[str, Any], *, root: str, user_id: Optional[str],
    creator_in_workspace: Callable[[str, str], Optional[str]], at: str,
    entry: Optional[Dict[str, Any]] = None,
) -> ReferenceOrigin:
    """The origin a promoted entry carries (step 3 of the module docstring).

    ``created_by`` is re-derived from the session named in ``generated_by``
    and never copied from the claim, whose file is model-writable.
    ``generated_by`` and ``witnessed_by`` are carried AS RECORDED: the
    daemon holds no record to check either against, and the curator
    promoting the claim has read both; ``curated_by`` is the stamp this
    step makes itself.  ``rendered_from`` is carried with its
    ``edited_after_render`` re-decided against ``entry``'s file now.
    """
    recorded = ReferenceOrigin.from_dict(claim.get("origin"))
    generated_by = recorded.generated_by if recorded else None
    session_id = (generated_by or {}).get("session_id")
    created_by = None
    if isinstance(session_id, str) and session_id:
        created_by = creator_in_workspace(session_id, root)
    return ReferenceOrigin(
        kind=ORIGIN_AGENT, at=at, generated_by=generated_by,
        created_by=created_by, claim_id=claim["claim_id"],
        curated_by=curator_stamp(user_id),
        witnessed_by=recorded.witnessed_by if recorded else None,
        rendered_from=rendered_stamp_now(root, entry or {}, recorded.rendered_from)
        if recorded else None,
    )


def destination_rel(bundle: str) -> str:
    """The workspace-relative directory a promotion into ``bundle`` writes to."""
    return f"{CATALOG_REL}/{bundle}" if bundle else CATALOG_REL


def _catalog_entry(entry: Dict[str, Any], root: str, origin: ReferenceOrigin,
                   dest_rel: str = CATALOG_REL) -> Dict[str, Any]:
    """The entry as written to the catalog: path re-anchored, origin attached."""
    out = dict(entry)
    if out.get("type") == "local":
        target = os.path.join(root, out["path"])
        out["path"] = os.path.relpath(target, os.path.join(root, dest_rel)).replace(os.sep, "/")
    out["origin"] = origin.to_dict()
    return out


#: Keys of a catalog entry a revision replaces: the document's.  Everything
#: else in the file (``mode``, ``origin``, ``contents``, a fetch hint, earlier
#: ``revisions``) is kept as the file has it.
_DOCUMENT_KEYS = ("type", "path", "content")


def revision_stamp(origin: ReferenceOrigin) -> Dict[str, Any]:
    """One ``revisions[]`` record: who wrote the revision, who promoted it, when.

    The promotion origin's fields without ``kind`` (a revision is not an
    arrival): ``claim_id``, ``at``, ``curated_by`` (the daemon's own stamp),
    and, as the claim recorded them, ``generated_by`` / ``witnessed_by`` /
    ``rendered_from``, plus a re-derived ``created_by``.
    """
    stamp = origin.to_dict()
    stamp.pop("kind", None)
    return stamp


def revised_entry(current: Dict[str, Any], entry: Dict[str, Any], *, root: str,
                  rel_file: str, stamp: Dict[str, Any]) -> Dict[str, Any]:
    """The catalog file after a revision: ``current`` with ``entry``'s fields.

    Replaced: name, description, tags, the document (``type`` + ``path`` /
    ``content``, a path re-anchored to the catalog file's directory), and
    ``links`` only when the revision carried them (``[]`` removes them).
    Kept: the id, ``origin`` (where the reference arrived), ``mode`` and
    every other key.  ``stamp`` is appended to ``revisions``.
    """
    out = {k: v for k, v in current.items() if k not in _DOCUMENT_KEYS}
    out["name"], out["description"], out["tags"] = (
        entry["name"], entry["description"], list(entry["tags"]))
    out["type"] = entry["type"]
    if entry["type"] == "local":
        out["path"] = os.path.relpath(
            os.path.join(root, entry["path"]),
            os.path.join(root, os.path.dirname(rel_file))).replace(os.sep, "/")
    else:
        out["content"] = entry["content"]
    if "links" in entry:
        if entry["links"]:
            out["links"] = entry["links"]
        else:
            out.pop("links", None)
    prior = current.get("revisions")
    out["revisions"] = (list(prior) if isinstance(prior, list) else []) + [stamp]
    return out


def _promote_revision(
    root: str, claim_path: str, claim: Dict[str, Any], *, user_id: Optional[str],
    creator_in_workspace: Callable[[str, str], Optional[str]],
    outcome: CurationOutcome, embed: Optional[Embed] = None,
) -> CurationOutcome:
    """Promote a REVISION claim: replace the catalog file in place.

    The file is found the way the loader reads the catalog
    (``claims.revision_target``), and its bytes must still have the digest
    the claim recorded -- otherwise ``stale``, nothing written: promoting
    would undo whatever changed it (another revision promoted first, an
    edge edited, a hand edit).  The claim's entry is re-validated through
    ``build_revision``, which refuses an id change or an ``origin``.  The
    revision is written where the reference lives; a ``bundle`` naming
    another one is refused.  Then the destination index is reconciled,
    because the embedding text (name, description, tags) may have changed.
    """
    stale, reason, target = revision_staleness(claim, root)
    if target is None:
        revised_id = (claim.get(REVISES_KEY) or {}).get("id")
        _t, category, error = revision_target(root, str(revised_id))
        if category == "ambiguous":
            return _fail(outcome, "ambiguous", f"{error}; remove the duplicate first")
        if category == "not_revisable":
            return _fail(outcome, "invalid_claim", error)
        return _fail(outcome, "stale", reason)
    if stale:
        return _fail(outcome, "stale", reason)
    if outcome.bundle and outcome.bundle != target["bundle"]:
        return _fail(outcome, "invalid_request",
                     f"a revision is written where '{target['id']}' lives "
                     f"({destination_rel(target['bundle'])}), not into another bundle")
    outcome.bundle = target["bundle"]
    entry, errors = build_claim_entry(claim_as_args(claim), workspace=root,
                                      catalog_ids=catalog_ids(root))
    if entry is None:
        return _fail(outcome, "invalid_claim", "; ".join(errors))
    origin = promoted_origin(claim, root=root, user_id=user_id,
                             creator_in_workspace=creator_in_workspace,
                             at=datetime.now(timezone.utc).isoformat(), entry=entry)
    rel_file = target["file"]
    data = json.dumps(revised_entry(target["data"], entry, root=root, rel_file=rel_file,
                                    stamp=revision_stamp(origin)),
                      indent=2, ensure_ascii=False)
    try:
        write_contained(root, rel_file, (data + "\n").encode("utf-8"))
    except PathLeavesRoot as exc:
        return _fail(outcome, "unsafe_path", str(exc))
    except OSError as exc:
        return _fail(outcome, "io_error", f"could not write {rel_file}: {exc}")
    return _after_write(root, claim_path, entry["id"], rel_file, outcome, embed,
                        revised=True)


def _after_write(root: str, claim_path: str, ref_id: str, rel_file: str,
                 outcome: CurationOutcome, embed: Optional[Embed], *,
                 revised: bool = False) -> CurationOutcome:
    """What every successful promotion does next: drop the claim, reconcile."""
    outcome.ok, outcome.reference_id, outcome.reference_file = True, ref_id, rel_file
    outcome.revised = revised
    try:
        os.unlink(claim_path)
    except OSError as exc:
        outcome.warnings.append(f"promoted, but the claim file could not be removed: {exc}")
    outcome.reconcile, outcome.reconcile_detail = reconcile_destination(
        root, outcome.bundle, embed, ref_id)
    if outcome.reconcile not in ("none", "updated", "clean"):
        outcome.warnings.append(
            f"promoted, but the bundle's vector index was not updated "
            f"({outcome.reconcile}: {outcome.reconcile_detail}); similarity "
            f"matching will not find '{ref_id}' until it is reconciled")
    return outcome


def _promote(
    root: str, claim_path: str, claim: Dict[str, Any], *, user_id: Optional[str],
    creator_in_workspace: Callable[[str, str], Optional[str]],
    outcome: CurationOutcome, embed: Optional[Embed] = None,
) -> CurationOutcome:
    """Steps 2 (re-validation) to 5 of the module docstring, then the index."""
    if is_revision(claim):
        return _promote_revision(root, claim_path, claim, user_id=user_id,
                                 creator_in_workspace=creator_in_workspace,
                                 outcome=outcome, embed=embed)
    ids = catalog_ids(root)
    ref_id = claim["reference"]["id"]
    dest_rel = destination_rel(outcome.bundle)
    rel_file = f"{dest_rel}/{ref_id}.json"
    if ref_id in ids or os.path.lexists(os.path.join(root, rel_file)):
        return _fail(outcome, "collision",
                     f"'{ref_id}' is already in the catalog; dismiss the claim and "
                     f"propose it again with revises='{ref_id}' to replace the "
                     "existing reference in place")
    entry, errors = build_proposed_reference(claim_as_args(claim), workspace=root,
                                             catalog_ids=ids)
    if entry is None:
        return _fail(outcome, "invalid_claim", "; ".join(errors))
    origin = promoted_origin(claim, root=root, user_id=user_id,
                             creator_in_workspace=creator_in_workspace,
                             at=datetime.now(timezone.utc).isoformat(), entry=entry)
    data = json.dumps(_catalog_entry(entry, root, origin, dest_rel), indent=2,
                      ensure_ascii=False)
    try:
        write_contained(root, rel_file, (data + "\n").encode("utf-8"))
    except PathLeavesRoot as exc:
        return _fail(outcome, "unsafe_path", str(exc))
    except OSError as exc:
        return _fail(outcome, "io_error", f"could not write {rel_file}: {exc}")
    return _after_write(root, claim_path, ref_id, rel_file, outcome, embed)


class RunnerEmbeddingProvider:
    """An embedding provider whose vectors come from the caller's session.

    Satisfies what :func:`~...references.reconcile.reconcile_bundle` uses of
    ``EmbeddingProviderProtocol`` -- ``available``, ``model_name``,
    ``dimensions``, ``embed_batch``, ``embed_text`` -- by forwarding to
    ``embed`` (``JaatoServer.embed_texts``, i.e. ``session.embed_texts`` on
    the runner).  A failed answer, or one from a model other than
    ``model_name``, raises, which ``reconcile_bundle`` records per text as
    skipped rather than writing a vector from the wrong model.
    """

    def __init__(self, embed: Embed, model_name: str, dimensions: int) -> None:
        self._embed = embed
        self.model_name = model_name
        self.dimensions = dimensions
        self.available = True

    def load_model(self) -> bool:
        return True

    def embed_batch(self, texts: List[str]) -> List[Optional[EmbeddingResult]]:
        answer = self._embed(list(texts))
        if not answer.get("ok"):
            raise RuntimeError(answer.get("error") or answer.get("category") or "embedding failed")
        if answer.get("model") != self.model_name:
            raise RuntimeError(f"the session's model changed to {answer.get('model')!r}")
        vectors = list(answer.get("vectors") or [])
        vectors += [None] * (len(texts) - len(vectors))
        return [EmbeddingResult(embedding=v, model=self.model_name, dimensions=len(v))
                if isinstance(v, list) else None for v in vectors[:len(texts)]]

    def embed_text(self, text: str) -> Optional[EmbeddingResult]:
        return self.embed_batch([text])[0]


_RECONCILE_STATUS = {"updated": "updated", "clean": "clean",
                     "skipped_busy": "busy", "unavailable": "unavailable",
                     "error": "error"}


def _destination_bundle(root: str, bundle: str) -> Optional[ReferenceBundle]:
    """The workspace-tier bundle ``bundle`` names (``""`` = catalog root), or ``None``."""
    try:
        directory = contained_dir(root, destination_rel(bundle), create=False)
    except PathLeavesRoot:
        return None
    if not directory:
        return None
    return load_reference_bundle(Path(directory), name=bundle or ROOT_BUNDLE_NAME,
                                 tier=BUNDLE_TIER_WORKSPACE)


def reconcile_destination(
    root: str, bundle: str, embed: Optional[Embed], ref_id: str,
) -> "tuple[str, str]":
    """Bring the destination bundle's vector index up to date: ``(outcome, detail)``.

    ``none`` when the bundle declares no index -- a bundle of definitions
    has nothing to reconcile.  Otherwise the session is asked for vectors
    once with no texts, to learn its model; a model other than the index's
    is ``unavailable``, because vectors from two models are not comparable.
    Then :func:`~...references.reconcile.reconcile_bundle` runs here, in
    the daemon, with :class:`RunnerEmbeddingProvider` -- the one code path
    that writes an index, unchanged.  ``updated`` means ``ref_id`` got its
    row; a reconcile that ran and skipped it is ``error``, with the reason.
    """
    from jaato_server.shared.plugins.references.reconcile import reconcile_bundle

    dest = _destination_bundle(root, bundle)
    if dest is None or not dest.has_index:
        return "none", ""
    if embed is None:
        return "unavailable", "no session is attached in this workspace to embed with"
    probe = embed([])
    if not probe.get("ok"):
        return "unavailable", str(probe.get("error") or probe.get("category") or "no answer")
    if probe.get("model") != dest.embedding_model:
        return "unavailable", (f"the session embeds with {probe.get('model')!r} and the "
                               f"index was built with {dest.embedding_model!r}")
    sources = discover_references(str(dest.directory), base_path=str(dest.directory.parent),
                                  project_root=root)
    for source in sources:
        source.bundle_name = dest.name
    provider = RunnerEmbeddingProvider(embed, dest.embedding_model, dest.embedding_dimensions)
    result = reconcile_bundle(dest, sources, provider)
    # The index files (sidecar, manifest, rewritten references) were written
    # on the daemon's account; the workspace's owner reads and rebuilds them.
    inherit_owner_tree(str(dest.directory), root)
    outcome = _RECONCILE_STATUS.get(result.status.value, "error")
    skipped = dict(result.skipped)
    if ref_id in skipped:
        return "error", f"'{ref_id}' was not embedded: {skipped[ref_id]}"
    return outcome, result.error or ""


def workspace_bundles(root: str) -> List[Dict[str, Any]]:
    """The workspace-tier sub-bundles a promotion may name, for the listing.

    ``[{"name", "indexed", "model"}]`` -- ``model`` only when ``indexed``.
    The catalog root is not listed: it is the default destination.
    """
    try:
        base = contained_dir(root, CATALOG_REL, create=False)
    except PathLeavesRoot:
        return []
    if not base:
        return []
    rows: List[Dict[str, Any]] = []
    for b in discover_bundles([(Path(base), BUNDLE_TIER_WORKSPACE)]):
        if b.name == ROOT_BUNDLE_NAME:
            continue
        row: Dict[str, Any] = {"name": b.name, "indexed": bool(b.has_index)}
        if b.has_index:
            row["model"] = b.embedding_model
        rows.append(row)
    return sorted(rows, key=lambda r: r["name"])


@dataclass
class ClaimsListing:
    """The claims in one workspace, as the curator's listing shows them.

    Attributes:
        ok: Whether the claims directory could be read at all.
        category: ``""``, or ``unsafe_path`` when the directory resolves out
            of the workspace.
        error: The reason, for a person.
        claims: One row per well-formed claim, oldest first
            (:func:`claim_row`).
        unreadable: File names in the claims directory that are not a claim
            this listing can show -- a symlink, a non-file, unreadable JSON,
            or a record that is not a proposed-reference claim.
        bundles: The workspace-tier sub-bundles a promotion may name
            (:func:`workspace_bundles`).
    """

    ok: bool = True
    category: str = ""
    error: str = ""
    claims: List[Dict[str, Any]] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    bundles: List[Dict[str, Any]] = field(default_factory=list)


def claim_row(
    claim: Dict[str, Any], *, root: str, ids: Set[str],
    pending: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """One claim as the listing shows it to a PERSON.

    Name, description and inline content are passed as they are -- they are
    model-written, and the client renders them as text -- and ``problems``
    is what :func:`build_proposed_reference` would refuse today, so the
    curator sees why Promote would fail before pressing it.  ``links`` are
    the claim's declared edges (notes included, as text) and ``warnings``
    what the curator should know that does not block Promote -- today, an
    edge whose target this workspace's catalog does not hold, naming the
    pending claim (``pending``: reference id -> claim id) when the target is
    one, so promoting both is visibly what resolves it.
    """
    ref = claim["reference"]
    row: Dict[str, Any] = {
        "claim_id": claim["claim_id"],
        "id": ref["id"],
        "name": ref.get("name") if isinstance(ref.get("name"), str) else "",
        "description": ref.get("description") if isinstance(ref.get("description"), str) else "",
        "tags": claim_tags(claim),
        "type": ref.get("type") if ref.get("type") in ("local", "inline") else "",
    }
    if row["type"] == "local" and isinstance(ref.get("path"), str):
        row["path"] = ref["path"]
    if row["type"] == "inline" and isinstance(ref.get("content"), str):
        row["content"] = ref["content"][:INLINE_CLAIM_MAX_CHARS]
    origin = ReferenceOrigin.from_dict(claim.get("origin"))
    if origin is not None:
        origin.rendered_from = rendered_stamp_now(root, ref, origin.rendered_from)
        row["origin"] = origin.to_dict()
    entry, problems = build_claim_entry(claim_as_args(claim), workspace=root,
                                        catalog_ids=ids)
    row["problems"] = problems
    if is_revision(claim):
        row.update(_revision_fields(claim, root))
    if entry is not None and entry.get("links"):
        row["links"] = entry["links"]
        row["warnings"] = link_warnings(entry["links"], ids, pending)
    return row


def _revision_fields(claim: Dict[str, Any], root: str) -> Dict[str, Any]:
    """What a revision row adds, so the curator sees a diff, not a new page.

    ``revises`` (the id), ``revises_file`` (where it lives now), ``current``
    (its fields as they are, in the claim's own terms --
    :func:`~...claims.current_fields`), ``links_replaced`` (whether the
    revision sets the edges or leaves them), and ``stale`` with
    ``stale_reason`` -- decided now, against the file's current bytes,
    because a revision written against an older version cannot be promoted.
    """
    stale, reason, target = revision_staleness(claim, root)
    fields: Dict[str, Any] = {
        "revises": claim[REVISES_KEY]["id"],
        "stale": stale,
        "links_replaced": "links" in claim["reference"],
    }
    if stale:
        fields["stale_reason"] = reason
    if target is not None:
        fields["revises_file"] = target["file"]
        fields["current"] = current_fields(target, root)
    return fields


def _load_claim_file(path: str) -> Any:
    """The JSON in ``path``, or ``None`` for a link, a non-file or bad JSON."""
    if not os.path.isfile(path) or os.path.islink(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def list_claims(workspace: str) -> ClaimsListing:
    """Every reference claim in ``workspace``, read the way a promotion reads one.

    The claims directory is resolved inside the workspace
    (:func:`~.contained_write.contained_dir`), a claim FILE that is a
    symlink is not followed, and each record is re-checked with
    :func:`~...claims.is_claim` -- the directory is model-writable.  A
    workspace with no claims directory lists nothing, which is not an error.
    Writes nothing; the owner gate applies to promoting, not to looking.
    """
    root = os.path.realpath(workspace)
    try:
        directory = contained_dir(root, CLAIMS_REL, create=False)
    except PathLeavesRoot as exc:
        return ClaimsListing(ok=False, category="unsafe_path", error=str(exc))
    listing = ClaimsListing(bundles=workspace_bundles(root))
    if not directory:
        return listing
    ids = catalog_ids(root)
    claims: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        data = _load_claim_file(os.path.join(directory, name))
        if is_claim(data) and name == f"{data['claim_id']}.json":
            claims.append(data)
        else:
            listing.unreadable.append(name)
    pending = pending_claim_ids(claims)
    listing.claims = [claim_row(c, root=root, ids=ids, pending=pending) for c in claims]
    return listing


def _fail(outcome: CurationOutcome, category: str, error: str) -> CurationOutcome:
    outcome.ok, outcome.category, outcome.error = False, category, error
    return outcome


def curate_claim(
    workspace: str, action: str, claim_id: str, *, owner: Optional[str],
    user_id: Optional[str],
    creator_in_workspace: Callable[[str, str], Optional[str]],
    bundle: str = "", embed: Optional[Embed] = None,
) -> CurationOutcome:
    """Promote or dismiss ``claim_id`` in ``workspace``.

    Args:
        workspace: The caller's workspace (``resolve_caller_workspace``).
        action: ``promote`` or ``dismiss``.
        claim_id: The claim to act on.
        owner: The workspace's qualified owner, ``None`` when unowned.
        user_id: The identity the transport authenticated on this
            connection, ``None`` when it authenticated nobody.
        creator_in_workspace: ``(session_id, workspace) -> created_by``, the
            daemon's own record of who a session was created for, answering
            ``None`` for a session it cannot place in that workspace.
        bundle: Promote into this workspace-tier sub-bundle instead of the
            catalog root (``unknown_bundle`` when there is none by that
            name).  Ignored by ``dismiss``.
        embed: The caller's session embedding texts, for the destination
            bundle's vector index; ``None`` when no session in this
            workspace can be asked (the index is then reported
            ``unavailable``, and the reference is placed anyway).

    Returns:
        A :class:`CurationOutcome`; this never raises for a refusal.
    """
    outcome = CurationOutcome(ok=False, action=action, claim_id=claim_id,
                              bundle=bundle if action == "promote" else "")
    if action not in CURATION_COMMANDS.values() or not valid_id(claim_id) or (
            outcome.bundle and not valid_id(outcome.bundle)):
        return _fail(outcome, "invalid_request",
                     "usage: reference.promote <claim_id> [--bundle <name>] | "
                     "reference.dismiss <claim_id>")
    if not may_curate(owner, user_id):
        return _fail(outcome, "not_owner",
                     "only the workspace owner may curate its references")
    root = os.path.realpath(workspace)
    if outcome.bundle and _destination_bundle(root, outcome.bundle) is None:
        return _fail(outcome, "unknown_bundle",
                     f"no bundle '{outcome.bundle}' in {CATALOG_REL}")
    claim_path, claim, category, error = _read_claim(root, claim_id)
    if claim_path is None:
        return _fail(outcome, category, error)
    if action == "promote":
        return _promote(root, claim_path, claim, user_id=user_id,
                        creator_in_workspace=creator_in_workspace, outcome=outcome,
                        embed=embed)
    try:
        os.unlink(claim_path)
    except OSError as exc:
        return _fail(outcome, "io_error", f"could not remove the claim: {exc}")
    outcome.ok = True
    return outcome


#: The typable bundle verb (1.36, #1478).
BUNDLE_CREATE_COMMAND = "reference.bundle.create"

#: Names a bundle may not take: the catalog root's spellings.
_RESERVED_BUNDLE_NAMES = frozenset({"root", "(root)", ROOT_BUNDLE_NAME})


@dataclass
class BundleCreateOutcome:
    """What one :func:`create_bundle` did; mirrors ``ReferenceBundleCreateResultEvent``."""
    ok: bool = False
    category: str = ""
    error: str = ""
    bundle: str = ""
    indexed: bool = False
    bundles: List[Dict[str, Any]] = field(default_factory=list)


def create_bundle(
    workspace: str, name: str, *, owner: Optional[str], user_id: Optional[str],
) -> BundleCreateOutcome:
    """Create the workspace-tier sub-bundle ``name``, unindexed (#1478).

    Writes only ``<workspace>/.jaato/references/<name>/bundle.json`` (the
    marker ``write_bundle_manifest`` writes, same body), never an
    ``embedding_config.json``: the daemon holds no embedding model, and a
    bundle without an index is a bundle -- tag lookup and selection work on
    it as on the root, and a promotion into it reports ``reconcile: none``.
    ``references bundle index <name>`` adds the index later from a session
    whose workspace has an embedding provider.

    The name is one id token (:func:`valid_id`, the rule a promotion's
    ``bundle`` already obeys: one flat component, no traversal), not a root
    alias.  Any existing directory by that name -- a bundle or not -- is a
    ``collision``; nothing is overwritten.  The write goes through
    :func:`write_contained`, so a link planted on the path is refused.
    Gated by the owner rule promotion uses.  Never raises for a refusal.
    """
    outcome = BundleCreateOutcome(bundle=name if isinstance(name, str) else "")
    root = os.path.realpath(workspace)

    def done(category: str = "", error: str = "") -> BundleCreateOutcome:
        outcome.ok, outcome.category, outcome.error = not category, category, error
        outcome.bundles = workspace_bundles(root)
        return outcome

    if not valid_id(name) or name in _RESERVED_BUNDLE_NAMES:
        return done("invalid_request",
                    f"usage: {BUNDLE_CREATE_COMMAND} <name> -- one id token "
                    "(letters, digits, '.', '_', '-'), not 'root'")
    allowed = may_curate(owner, user_id)
    if not allowed:
        return done("not_owner", "only the workspace owner may create its reference bundles")
    rel = f"{CATALOG_REL}/{name}"
    try:
        catalog = contained_dir(root, CATALOG_REL, create=False)
    except PathLeavesRoot as exc:
        return done("unsafe_path", str(exc))
    target = os.path.join(catalog, name) if catalog else ""
    if target and os.path.lexists(target):
        what = ("a bundle" if os.path.isfile(os.path.join(target, "bundle.json"))
                else "an entry")
        return done("collision", f"{what} named '{name}' already exists in {CATALOG_REL}")
    body = json.dumps({"name": name, "description": "references bundle"},
                      indent=2, ensure_ascii=False) + "\n"
    try:
        write_contained(root, f"{rel}/bundle.json", body.encode("utf-8"))
    except PathLeavesRoot as exc:
        return done("unsafe_path", str(exc))
    except OSError as exc:
        return done("io_error", f"could not create the bundle: {exc}")
    return done()
