"""A person reads the workspace reference catalog and edits a reference's links.

Typed links (``links: [{to, rel, note?}]``, ``shared/plugins/references/
links.py``) reach the catalog two ways: written by hand into a reference's
JSON, or carried in by the promotion of an agent's proposal
(``reference_curation``).  After that there was no way to change them short
of editing the file on the host.  This module is the DAEMON's half of two
verbs (protocol 1.33); the file itself is written by the runner of a session
in the workspace (``session.write_reference``, base profile, #1422), or here
when there is none:

* :func:`list_catalog` -- ``ReferenceCatalogRequest`` -> ``ReferenceCatalogEvent``:
  every reference in the workspace catalog, sub-bundles included, with its
  declared links (a dangling one marked) and the links pointing at it.
* :func:`update_links` -- ``ReferenceLinksUpdateRequest`` ->
  ``ReferenceLinksUpdateResultEvent``: replace one reference's links.

Rules, each a way it could go wrong:

1. **The owner gate** (:func:`~.memory_verbs.may_curate`, the promotion's):
   changing an edge changes what other requests get -- a ``supersedes``
   reroutes every request for its target -- so it is a curator's act.
2. **The links are validated, not trusted**: :func:`~...links.link_errors`
   refuses an unknown ``rel``, a target that is not an id, an edge to the
   reference itself, and unknown keys, before anything is written.
3. **What a person should know is said, and does not block**: an edge to an
   id this catalog does not hold is kept (the loader marks it dangling), and
   a ``supersedes`` another reference also declares leaves the older one
   routed nowhere; both are ``warnings``.
4. **Only the ``links`` key changes.**  The file is re-read, the key
   replaced (removed when the list is empty) and the rest written back as
   it was, through ``write_contained`` (#1386, in
   :func:`~.reference_catalog_write.write_catalog_file`), so a link planted in the
   catalog cannot carry the write out of the workspace.  A reference whose
   id appears in two files is ``ambiguous`` and is not edited: which one
   the loader keeps is not this module's to guess.
5. **The vector index is untouched**: an embedding is made from the name,
   description, tags and fetch hint (``bundle.embedding_text``), never the
   links, so no reconcile is needed.

A running session sees the change at its next catalog reload.  Stdlib plus
the references plugin's own helpers; no daemon state.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from jaato_server.shared.plugins.bundle_common.bundle import BUNDLE_MANIFEST_FILENAME
from jaato_server.shared.plugins.references.bundle import REFERENCE_NON_SOURCE_FILENAMES
from jaato_server.shared.plugins.references.claims import link_warnings, valid_id
from jaato_server.shared.plugins.references.links import (
    REL_SUPERSEDES,
    LinkIndex,
    link_errors,
    parse_links,
)

from .contained_write import PathLeavesRoot, contained_dir
from .memory_verbs import may_curate
from .reference_curation import CATALOG_REL, local_writer

logger = logging.getLogger(__name__)


@dataclass
class CatalogListing:
    """The workspace catalog, as the curator's view shows it.

    Attributes:
        ok: Whether the catalog directory could be read at all.
        category: ``""``, or ``unsafe_path`` when it resolves out of the
            workspace.
        error: The reason, for a person.
        references: One row per reference file (:func:`_row`), by id.
        unreadable: Workspace-relative files under the catalog that are not
            a reference this view can show (a link, bad JSON, no ``id``).
    """

    ok: bool = True
    category: str = ""
    error: str = ""
    references: List[Dict[str, Any]] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)


@dataclass
class LinksOutcome:
    """What one links update did.

    Attributes:
        ok: Whether the links were written.
        reference_id: The reference, as the caller named it.
        category: ``""`` on success, else ``invalid_request``, ``not_owner``,
            ``invalid_links``, ``not_found``, ``ambiguous``, ``unsafe_path``
            or ``io_error``.
        error: The reason, for a person.
        reference_file: The workspace-relative file written.
        links: The links as written (normalised).
        warnings: What does not block the write: a dangling target, a
            ``supersedes`` another reference also declares.
    """

    ok: bool
    reference_id: str
    category: str = ""
    error: str = ""
    reference_file: str = ""
    links: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _load(path: str) -> Optional[Dict[str, Any]]:
    """The JSON object in ``path``, or ``None`` for a link, a non-file, bad JSON."""
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _catalog_dirs(base: str) -> List[Tuple[str, str]]:
    """``[(bundle, directory)]`` the loader reads: the root, then each sub-bundle.

    A sub-bundle is an immediate subdirectory carrying ``bundle.json``
    (``bundle_common.discover_bundles``); any other directory is not part
    of the catalog, and a linked one is never followed.
    """
    dirs = [("", base)]
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if (not os.path.islink(path) and os.path.isdir(path)
                and os.path.isfile(os.path.join(path, BUNDLE_MANIFEST_FILENAME))):
            dirs.append((name, path))
    return dirs


def catalog_files(root: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[str]]:
    """``([(relative_file, data)], unreadable)`` for every reference in the catalog.

    Reads the ``*.json`` files the reference loader reads
    (``config_loader.discover_references``): directly in
    ``<root>/.jaato/references`` and in each sub-bundle directory, never a
    link, never a bundle manifest.  Raises :class:`PathLeavesRoot` when the
    catalog directory itself resolves out of the workspace.
    """
    base = contained_dir(root, CATALOG_REL, create=False)
    entries: List[Tuple[str, Dict[str, Any]]] = []
    unreadable: List[str] = []
    if not base:
        return entries, unreadable
    for _bundle, directory in _catalog_dirs(base):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json") or name in REFERENCE_NON_SOURCE_FILENAMES:
                continue
            path = os.path.join(directory, name)
            if not os.path.islink(path) and os.path.isdir(path):
                continue
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            data = _load(path)
            if data is None or not isinstance(data.get("id"), str) or not data["id"]:
                unreadable.append(rel)
                continue
            entries.append((rel, data))
    return entries, unreadable


def _index(entries: List[Tuple[str, Dict[str, Any]]]) -> LinkIndex:
    return LinkIndex(SimpleNamespace(id=d["id"], links=parse_links(d.get("links")))
                     for _rel, d in entries)


def _row(rel: str, data: Dict[str, Any], index: LinkIndex, duplicate: bool) -> Dict[str, Any]:
    """One reference as the curator's view shows it.

    ``name`` and ``description`` are catalog text: a client shows them as
    text.  ``bundle`` is the sub-bundle directory (``""`` for the catalog
    root).  ``links`` are its declared edges, each ``dangling`` when the
    target is absent; ``linked_from`` the edges pointing at it.
    """
    parts = rel.split("/")
    bundle = parts[2] if len(parts) > 3 else ""
    row: Dict[str, Any] = {
        "id": data["id"],
        "name": data.get("name") if isinstance(data.get("name"), str) else "",
        "description": data.get("description") if isinstance(data.get("description"), str) else "",
        "bundle": bundle,
        "file": rel,
        "links": index.links_of(data["id"]),
        "linked_from": index.linked_from(data["id"]),
    }
    if duplicate:
        row["duplicate_id"] = True
    return row


def list_catalog(workspace: str) -> CatalogListing:
    """Every reference in ``workspace``'s catalog, with its links both ways.

    Writes nothing; the owner gate applies to editing, not to looking.
    """
    root = os.path.realpath(workspace)
    try:
        entries, unreadable = catalog_files(root)
    except PathLeavesRoot as exc:
        return CatalogListing(ok=False, category="unsafe_path", error=str(exc))
    index = _index(entries)
    counts: Dict[str, int] = {}
    for _rel, data in entries:
        counts[data["id"]] = counts.get(data["id"], 0) + 1
    rows = [_row(rel, data, index, counts[data["id"]] > 1) for rel, data in entries]
    return CatalogListing(references=sorted(rows, key=lambda r: (r["id"], r["file"])),
                          unreadable=unreadable)


def _supersedes_warnings(ref_id: str, links: List[Dict[str, Any]], index: LinkIndex) -> List[str]:
    """A ``supersedes`` another reference also declares routes the older one nowhere."""
    out = []
    for link in links:
        if link["rel"] != REL_SUPERSEDES:
            continue
        others = sorted(s for s in index.successors.get(link["to"], []) if s != ref_id)
        if others:
            out.append(f"'{', '.join(others)}' also supersedes '{link['to']}'; with two "
                       f"successors, requests for '{link['to']}' are routed to neither")
    return out


def _fail(outcome: LinksOutcome, category: str, error: str) -> LinksOutcome:
    outcome.ok, outcome.category, outcome.error = False, category, error
    return outcome


def _request_refusal(
    reference_id: Any, links: Any, owner: Optional[str], user_id: Optional[str],
) -> Optional[Tuple[str, str]]:
    """``(category, error)`` when the request itself is refused, before any read.

    Checked in order: the request's shape, the owner gate, then the links.
    """
    if not valid_id(reference_id) or not isinstance(links, list):
        return ("invalid_request",
                "usage: reference.links <reference_id> with a list of {to, rel, note?}")
    if not may_curate(owner, user_id):
        return ("not_owner", "only the workspace owner may change its references' links")
    errors = link_errors(links, source_id=reference_id)
    if errors:
        return ("invalid_links", "; ".join(errors))
    return None


def _match_refusal(
    reference_id: str, matches: List[Tuple[str, Dict[str, Any]]],
) -> Optional[Tuple[str, str]]:
    """``(category, error)`` unless exactly one catalog file declares the id."""
    if not matches:
        return ("not_found", f"no reference '{reference_id}' in {CATALOG_REL}")
    if len(matches) > 1:
        return ("ambiguous",
                f"'{reference_id}' is defined in {', '.join(r for r, _ in matches)}; "
                "remove the duplicate first")
    return None


def update_links(
    workspace: str, reference_id: str, links: Any, *,
    owner: Optional[str], user_id: Optional[str],
    write: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> LinksOutcome:
    """Replace ``reference_id``'s declared links in ``workspace``'s catalog.

    Args:
        workspace: The caller's workspace (``resolve_caller_workspace``).
        reference_id: The catalog id to edit.
        links: The complete new list, ``[{to, rel, note?}]``; ``[]`` removes
            every declared edge.
        owner: The workspace's qualified owner, ``None`` when unowned.
        user_id: The identity the transport authenticated on this
            connection.
        write: Writes the file (the runner of a session in this workspace,
            ``session.write_reference``, #1422); ``None`` writes here.  A
            links edit changes no embedding, so nothing is reconciled.

    Returns:
        A :class:`LinksOutcome`; this never raises for a refusal.
    """
    outcome = LinksOutcome(ok=False, reference_id=reference_id)
    refusal = _request_refusal(reference_id, links, owner, user_id)
    if refusal:
        return _fail(outcome, *refusal)
    root = os.path.realpath(workspace)
    try:
        entries, _unreadable = catalog_files(root)
    except PathLeavesRoot as exc:
        return _fail(outcome, "unsafe_path", str(exc))
    matches = [(rel, data) for rel, data in entries if data["id"] == reference_id]
    refusal = _match_refusal(reference_id, matches)
    if refusal:
        return _fail(outcome, *refusal)
    rel, data = matches[0]
    normalised = [link.to_dict() for link in parse_links(links)]
    updated = dict(data)
    if normalised:
        updated["links"] = normalised
    else:
        updated.pop("links", None)
    body = json.dumps(updated, indent=2, ensure_ascii=False) + "\n"
    written = (write or local_writer(root))(
        {"rel_file": rel, "data": body, "replace": True, "bundle": "",
         "ref_id": reference_id, "reconcile": False})
    if not written.get("ok"):
        return _fail(outcome, written.get("category") or "io_error",
                     written.get("error") or f"could not write {rel}")
    index = _index([(r, updated if r == rel else d) for r, d in entries])
    outcome.ok, outcome.reference_file, outcome.links = True, rel, normalised
    outcome.warnings = (link_warnings(normalised, index.ids)
                        + _supersedes_warnings(reference_id, normalised, index))
    return outcome
