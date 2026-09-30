"""Promote or dismiss a reference CLAIM: the half of the write path an agent cannot do.

An agent proposes a reference with ``proposeReference``, which writes a
claim under ``<workspace>/.jaato/references-claims/``
(:mod:`jaato_server.shared.plugins.references.claims`).  It never writes the
catalog: a model-called tool body runs in ``tool_hat``, which is ``audit
deny ... wlk`` on ``<workspace>/.jaato/references/**`` (#1422).  This
module is the other half, run by the DAEMON for the person on a connection
(``reference.promote`` / ``reference.dismiss``, protocol 1.33): it decides
what a claim becomes, and a writer (below) puts it in the catalog.

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
4. **The write never goes through a link.**  The writer
   (:func:`~.reference_catalog_write.write_catalog_file`) uses
   ``write_contained`` (#1386) for ``<workspace>/.jaato/references/<id>.json``;
   a destination that already exists is a collision, never an overwrite.
5. **The claim is removed** once the entry is written.  A claim that cannot
   be removed afterwards is reported, and the next promotion of it answers
   ``collision`` rather than writing a second copy.

A local ``path`` is stored in the claim workspace-relative and rewritten
relative to the catalog file (``../../docs/x.md``), because the catalog
loader resolves a relative path against the reference file's own
directory.

**Into a named bundle, and its index.**  A promotion may name a
workspace-tier sub-bundle (``bundle``): the entry is written into that
bundle's directory instead of the catalog root.  When the destination
bundle declares a vector index (``embedding_config.json``), the new entry
has no row in it yet, so the index is reconciled.

**Who writes.**  This module decides WHAT is written; the bytes go to a
``Writer`` the caller supplies.  The daemon's writer asks the runner of a
session in this workspace (``session.write_reference``), which writes in
its base profile and reconciles with its own embedding provider
(:mod:`.reference_catalog_write`).  Before #1422 the daemon wrote and
reconciled here, with vectors fetched from the runner, because every
runner body denied the catalog; template v44 moved that deny out of base.
With no session to ask, the daemon writes itself and an indexed bundle is
reported ``unavailable``.  The reference is placed whatever the reconcile's
outcome, which is reported, never silent.

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
    discover_bundles,
)
from jaato_server.shared.plugins.references.claims import (
    CLAIMS_DIRNAME,
    INLINE_CLAIM_MAX_CHARS,
    build_proposed_reference,
    claim_as_args,
    claim_tags,
    is_claim,
    link_warnings,
    valid_id,
)
from jaato_server.shared.plugins.references.models import (
    ORIGIN_AGENT,
    ReferenceOrigin,
)

from .contained_write import PathLeavesRoot, contained_dir
from .memory_verbs import may_curate
from .reference_catalog_write import (
    CATALOG_REL,
    destination_bundle,
    destination_rel,
    write_catalog_file,
)

logger = logging.getLogger(__name__)

#: The two verbs, by the command name a client sends.
CURATION_COMMANDS: Dict[str, str] = {
    "reference.promote": "promote",
    "reference.dismiss": "dismiss",
}

#: Where claims live, relative to the workspace.
CLAIMS_REL = f".jaato/{CLAIMS_DIRNAME}"

#: ``args -> answer`` -- writes one catalog file:
#: ``args = {"rel_file", "data", "replace", "bundle", "ref_id", "reconcile"}``
#: answered ``{"ok", "category", "error", "reconcile", "reconcile_detail"}``
#: (:func:`~.reference_catalog_write.write_catalog_file`'s shape).  The
#: daemon's writer is the runner of a session in the workspace
#: (``JaatoServer.write_reference``); :func:`local_writer` is the fallback.
Writer = Callable[[Dict[str, Any]], Dict[str, Any]]

#: What the reconcile reports, for ``reconcile`` on the answer: ``none``
#: (the bundle has no vector index), ``updated``, ``clean``, ``busy``
#: (another reconcile holds the lock), ``unavailable`` (no session, no
#: provider, or a model that differs from the index's), ``error``.
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
            ``invalid_claim``, ``collision``, ``unsafe_path``, ``io_error``.  A client branches on this, not
            on ``error``.
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
) -> ReferenceOrigin:
    """The origin a promoted entry carries (step 3 of the module docstring).

    ``created_by`` is re-derived from the session named in ``generated_by``
    and never copied from the claim, whose file is model-writable.
    ``generated_by`` and ``witnessed_by`` are carried AS RECORDED: the
    daemon holds no record to check either against, and the curator
    promoting the claim has read both; ``curated_by`` is the stamp this
    step makes itself.
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
    )


def local_writer(root: str) -> Writer:
    """Write in THIS process, with no embedding provider.

    For when no session in the workspace can be asked: the daemon is not
    confined, so the entry is placed, and an indexed bundle is reported
    ``unavailable`` (nothing here can embed).
    """
    def _write(args: Dict[str, Any]) -> Dict[str, Any]:
        answer = write_catalog_file(root, provider=None, **args)
        if answer.get("reconcile") == "unavailable":
            answer["reconcile_detail"] = (
                "no session is attached in this workspace to embed with")
        return answer
    return _write


def _catalog_entry(entry: Dict[str, Any], root: str, origin: ReferenceOrigin,
                   dest_rel: str = CATALOG_REL) -> Dict[str, Any]:
    """The entry as written to the catalog: path re-anchored, origin attached."""
    out = dict(entry)
    if out.get("type") == "local":
        target = os.path.join(root, out["path"])
        out["path"] = os.path.relpath(target, os.path.join(root, dest_rel)).replace(os.sep, "/")
    out["origin"] = origin.to_dict()
    return out


def _promote(
    root: str, claim_path: str, claim: Dict[str, Any], *, user_id: Optional[str],
    creator_in_workspace: Callable[[str, str], Optional[str]],
    outcome: CurationOutcome, write: Writer,
) -> CurationOutcome:
    """Steps 2 (re-validation) to 5 of the module docstring, then the index."""
    ids = catalog_ids(root)
    ref_id = claim["reference"]["id"]
    dest_rel = destination_rel(outcome.bundle)
    rel_file = f"{dest_rel}/{ref_id}.json"
    if ref_id in ids:
        return _fail(outcome, "collision",
                     f"'{ref_id}' is already in the catalog; dismiss the claim, or "
                     "revise the existing reference")
    entry, errors = build_proposed_reference(claim_as_args(claim), workspace=root,
                                             catalog_ids=ids)
    if entry is None:
        return _fail(outcome, "invalid_claim", "; ".join(errors))
    origin = promoted_origin(claim, root=root, user_id=user_id,
                             creator_in_workspace=creator_in_workspace,
                             at=datetime.now(timezone.utc).isoformat())
    data = json.dumps(_catalog_entry(entry, root, origin, dest_rel), indent=2,
                      ensure_ascii=False)
    written = write({"rel_file": rel_file, "data": data + "\n", "replace": False,
                     "bundle": outcome.bundle, "ref_id": ref_id, "reconcile": True})
    if not written.get("ok"):
        return _fail(outcome, written.get("category") or "io_error",
                     written.get("error") or f"could not write {rel_file}")
    outcome.ok, outcome.reference_id, outcome.reference_file = True, ref_id, rel_file
    try:
        os.unlink(claim_path)
    except OSError as exc:
        outcome.warnings.append(f"promoted, but the claim file could not be removed: {exc}")
    outcome.reconcile = written.get("reconcile") or "none"
    outcome.reconcile_detail = written.get("reconcile_detail") or ""
    if outcome.reconcile not in ("none", "updated", "clean"):
        outcome.warnings.append(
            f"promoted, but the bundle's vector index was not updated "
            f"({outcome.reconcile}: {outcome.reconcile_detail}); similarity "
            f"matching will not find '{ref_id}' until it is reconciled")
    return outcome


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


def claim_row(claim: Dict[str, Any], *, root: str, ids: Set[str]) -> Dict[str, Any]:
    """One claim as the listing shows it to a PERSON.

    Name, description and inline content are passed as they are -- they are
    model-written, and the client renders them as text -- and ``problems``
    is what :func:`build_proposed_reference` would refuse today, so the
    curator sees why Promote would fail before pressing it.  ``links`` are
    the claim's declared edges (notes included, as text) and ``warnings``
    what the curator should know that does not block Promote -- today, an
    edge whose target this workspace's catalog does not hold.
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
        row["origin"] = origin.to_dict()
    entry, problems = build_proposed_reference(claim_as_args(claim), workspace=root,
                                               catalog_ids=ids)
    row["problems"] = problems
    if entry is not None and entry.get("links"):
        row["links"] = entry["links"]
        row["warnings"] = link_warnings(entry["links"], ids)
    return row


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
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        data = _load_claim_file(os.path.join(directory, name))
        if is_claim(data) and name == f"{data['claim_id']}.json":
            listing.claims.append(claim_row(data, root=root, ids=ids))
        else:
            listing.unreadable.append(name)
    return listing


def _fail(outcome: CurationOutcome, category: str, error: str) -> CurationOutcome:
    outcome.ok, outcome.category, outcome.error = False, category, error
    return outcome


def curate_claim(
    workspace: str, action: str, claim_id: str, *, owner: Optional[str],
    user_id: Optional[str],
    creator_in_workspace: Callable[[str, str], Optional[str]],
    bundle: str = "", write: Optional[Writer] = None,
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
        write: Writes the entry and reconciles the destination bundle's
            index (the runner of a session in this workspace);
            ``None`` writes here with no embedding provider
            (:func:`local_writer`), which reports an indexed bundle
            ``unavailable`` and places the reference anyway.

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
    if outcome.bundle and destination_bundle(root, outcome.bundle) is None:
        return _fail(outcome, "unknown_bundle",
                     f"no bundle '{outcome.bundle}' in {CATALOG_REL}")
    claim_path, claim, category, error = _read_claim(root, claim_id)
    if claim_path is None:
        return _fail(outcome, category, error)
    if action == "promote":
        return _promote(root, claim_path, claim, user_id=user_id,
                        creator_in_workspace=creator_in_workspace, outcome=outcome,
                        write=write or local_writer(root))
    try:
        os.unlink(claim_path)
    except OSError as exc:
        return _fail(outcome, "io_error", f"could not remove the claim: {exc}")
    outcome.ok = True
    return outcome
