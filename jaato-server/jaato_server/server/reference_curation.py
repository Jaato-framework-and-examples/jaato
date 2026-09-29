"""Promote or dismiss a reference CLAIM: the half of the write path an agent cannot do.

An agent proposes a reference with ``proposeReference``, which writes a
claim under ``<workspace>/.jaato/references-claims/``
(:mod:`jaato_server.shared.plugins.references.claims`).  It never writes the
catalog: every AppArmor body is ``audit deny ... wlk`` on
``<workspace>/.jaato/references/**``.  This module is the other half, run by
the DAEMON for the person on a connection (``reference.promote`` /
``reference.dismiss``, protocol 1.32), and it is the only in-tree code that
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

A local ``path`` is stored in the claim workspace-relative and rewritten
relative to the catalog file (``../../docs/x.md``), because the catalog
loader resolves a relative path against the reference file's own
directory.

Stdlib plus the references plugin's own helpers; no daemon state.  The
caller hands in the owner, the identity and a creator lookup.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from jaato_server.shared.plugins.references.bundle import (
    REFERENCE_NON_SOURCE_FILENAMES,
)
from jaato_server.shared.plugins.references.claims import (
    CLAIMS_DIRNAME,
    build_proposed_reference,
    claim_as_args,
    is_claim,
    valid_id,
)
from jaato_server.shared.plugins.references.models import (
    ORIGIN_AGENT,
    ReferenceOrigin,
)

from .contained_write import PathLeavesRoot, contained_dir, write_contained
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


@dataclass
class CurationOutcome:
    """What one curation verb did.

    Attributes:
        ok: Whether the verb did what was asked.
        action: ``promote`` or ``dismiss``.
        claim_id: The claim acted on, as the caller named it.
        category: ``""`` on success, else one of ``invalid_request``,
            ``not_owner``, ``not_found``, ``invalid_claim``, ``collision``,
            ``unsafe_path``, ``io_error``.  A client branches on this, not
            on ``error``.
        error: The reason, for a person.
        reference_id: The catalog id promoted (promote only).
        reference_file: The workspace-relative catalog file written.
        warnings: Things that happened beside success (the claim file could
            not be removed after a promotion).
    """

    ok: bool
    action: str
    claim_id: str
    category: str = ""
    error: str = ""
    reference_id: str = ""
    reference_file: str = ""
    warnings: List[str] = field(default_factory=list)


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
    )


def _catalog_entry(entry: Dict[str, Any], root: str, origin: ReferenceOrigin) -> Dict[str, Any]:
    """The entry as written to the catalog: path re-anchored, origin attached."""
    out = dict(entry)
    if out.get("type") == "local":
        target = os.path.join(root, out["path"])
        out["path"] = os.path.relpath(target, os.path.join(root, CATALOG_REL)).replace(os.sep, "/")
    out["origin"] = origin.to_dict()
    return out


def _promote(
    root: str, claim_path: str, claim: Dict[str, Any], *, user_id: Optional[str],
    creator_in_workspace: Callable[[str, str], Optional[str]],
    outcome: CurationOutcome,
) -> CurationOutcome:
    """Steps 2 (re-validation) to 5 of the module docstring."""
    ids = catalog_ids(root)
    ref_id = claim["reference"]["id"]
    rel_file = f"{CATALOG_REL}/{ref_id}.json"
    if ref_id in ids or os.path.lexists(os.path.join(root, rel_file)):
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
    data = json.dumps(_catalog_entry(entry, root, origin), indent=2, ensure_ascii=False)
    try:
        write_contained(root, rel_file, (data + "\n").encode("utf-8"))
    except PathLeavesRoot as exc:
        return _fail(outcome, "unsafe_path", str(exc))
    except OSError as exc:
        return _fail(outcome, "io_error", f"could not write {rel_file}: {exc}")
    outcome.ok, outcome.reference_id, outcome.reference_file = True, ref_id, rel_file
    try:
        os.unlink(claim_path)
    except OSError as exc:
        outcome.warnings.append(f"promoted, but the claim file could not be removed: {exc}")
    return outcome


def _fail(outcome: CurationOutcome, category: str, error: str) -> CurationOutcome:
    outcome.ok, outcome.category, outcome.error = False, category, error
    return outcome


def curate_claim(
    workspace: str, action: str, claim_id: str, *, owner: Optional[str],
    user_id: Optional[str],
    creator_in_workspace: Callable[[str, str], Optional[str]],
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

    Returns:
        A :class:`CurationOutcome`; this never raises for a refusal.
    """
    outcome = CurationOutcome(ok=False, action=action, claim_id=claim_id)
    if action not in CURATION_COMMANDS.values() or not valid_id(claim_id):
        return _fail(outcome, "invalid_request",
                     "usage: reference.promote <claim_id> | reference.dismiss <claim_id>")
    if not may_curate(owner, user_id):
        return _fail(outcome, "not_owner",
                     "only the workspace owner may curate its references")
    root = os.path.realpath(workspace)
    claim_path, claim, category, error = _read_claim(root, claim_id)
    if claim_path is None:
        return _fail(outcome, category, error)
    if action == "promote":
        return _promote(root, claim_path, claim, user_id=user_id,
                        creator_in_workspace=creator_in_workspace, outcome=outcome)
    try:
        os.unlink(claim_path)
    except OSError as exc:
        return _fail(outcome, "io_error", f"could not remove the claim: {exc}")
    outcome.ok = True
    return outcome
