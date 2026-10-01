"""Reference CLAIMS: what an agent proposes, before anybody promotes it.

An agent that wrote a document others should be able to select calls
``proposeReference``.  The result is a **claim** -- one JSON file under
``<workspace>/.jaato/references-claims/`` -- and never a catalog entry:

* the confined runner is write-denied on ``.jaato/references/**`` in every
  AppArmor body, so an agent cannot edit the catalog and is not given a way
  around that;
* the catalog is AUTHORED state (``scaffold/gitignore.py``) and a claim is
  runtime state, so the two are kept apart on disk as well as in trust.

A claim file carries the catalog entry it proposes (the shape
``validate_reference_file`` accepts, so promotion is a copy) and an
``origin`` of kind :data:`~.models.ORIGIN_AGENT`, stamped from the
proposing SESSION -- never from the tool's arguments.  ``mode`` is always
``selectable``: an ``auto`` reference is injected into every system prompt,
which is a curator's decision and not the writing agent's.

**What a claim's origin is worth.**  On a confined host only this tool
writes the claims directory: the file tools and ``cli`` refuse
``.jaato/...``, and the template write-denies it in ``//child`` (v43), so
no subprocess the model drives can write one by hand.  On an unconfined
host, or from the flat isolated sub-runner profile, a script still can,
with an origin of its own choosing.  That is why a claim is listed as
*unreviewed* and fenced as untrusted content (``listReferences``), and
why the stamp promotion makes (``curated_by``) is the daemon's own.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import wrap_untrusted_content

from jaato_server.shared.call_witness import current_call_witness

from .config_loader import validate_reference_file
from .links import LINK_RELS, link_errors, parse_links
from .models import ORIGIN_AGENT, ReferenceOrigin

#: Directory under ``<workspace>/.jaato/`` holding claim files.
CLAIMS_DIRNAME = "references-claims"

#: The one status a claim has before a curator acts on it.
CLAIM_STATUS_PROPOSED = "proposed"

#: Inline content above this many characters is refused: write the document
#: to a workspace file and propose its ``path`` instead, so the claim stays
#: small enough to list and the document stays editable.
INLINE_CLAIM_MAX_CHARS = 32 * 1024

#: A reference id: one token a later ``selectReferences`` can name, a
#: filename component when promoted, and a whole word the transitive matcher
#: can find in another document.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: A tag: one token.  Enforced here rather than by ``validate_reference_file``
#: (which accepts any string) because a claim's tags are shown OUTSIDE the
#: untrusted fence, so they must not be able to carry prose.
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


def valid_id(value: Any) -> bool:
    """Whether ``value`` is one id token (a reference id or a claim id)."""
    return isinstance(value, str) and bool(_ID_RE.match(value))


def claims_dir(workspace: str) -> Path:
    """``<workspace>/.jaato/references-claims``."""
    return Path(workspace) / ".jaato" / CLAIMS_DIRNAME


def proposing_origin(session: Any, *, claim_id: str, at: str) -> ReferenceOrigin:
    """The ``agent`` origin for a claim proposed by ``session``.

    Everything is read off the session the call is running in:
    ``generated_by`` from ``JaatoSession._model_provenance`` (the one
    definition memory and model media also use), ``created_by`` from
    ``_client_user_id`` -- set from ``SessionInitEnvelope.created_by``, the
    ``EventSink.get_client_user`` chain, which reads no environment.  With no
    session in context both are ``None``: provenance unknown, never invented.

    ``witnessed_by`` is the call's own permission verdict
    (:func:`~jaato_server.shared.call_witness.current_call_witness`): set only
    when a person was ASKED at the prompt and approved, absent whenever the
    policy decided -- including the default, where ``proposeReference`` is
    auto-approved.
    """
    generated_by: Optional[Dict[str, Any]] = None
    resolver = getattr(session, "_model_provenance", None)
    if callable(resolver):
        try:
            generated_by = resolver() or None
        except Exception:  # noqa: BLE001 -- a stamp must not fail a proposal
            generated_by = None
    created_by = getattr(session, "_client_user_id", None)
    return ReferenceOrigin(
        kind=ORIGIN_AGENT,
        at=at,
        generated_by=generated_by if isinstance(generated_by, dict) else None,
        created_by=created_by if isinstance(created_by, str) and created_by else None,
        claim_id=claim_id,
        witnessed_by=current_call_witness(),
    )


def _tag_list(value: Any) -> Optional[List[str]]:
    """``value`` as a list of tag tokens, ``[]`` when absent, ``None`` if malformed."""
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(v, str) and _TAG_RE.match(v)
                                       for v in value):
        return list(value)
    return None


def _resolve_document(
    args: Dict[str, Any], workspace: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The ``type`` + ``path`` / ``content`` half of the entry, or an error.

    A path must resolve (symlinks followed) to a regular file INSIDE the
    workspace, and is stored workspace-relative: a claim is about a document
    the agent could itself read, and an absolute path would not survive the
    workspace moving.
    """
    path, content = args.get("path"), args.get("content")
    if bool(path) == bool(content):
        return None, "Provide exactly one of 'path' (a workspace file) or 'content'."
    if content:
        if not isinstance(content, str):
            return None, "'content' must be a string."
        if len(content) > INLINE_CLAIM_MAX_CHARS:
            return None, (f"'content' exceeds {INLINE_CLAIM_MAX_CHARS} characters; "
                          "write the document to a workspace file and propose its 'path'.")
        return {"type": "inline", "content": content}, None
    if not isinstance(path, str):
        return None, "'path' must be a string."
    root = Path(workspace).resolve()
    target = Path(path) if os.path.isabs(path) else root / path
    target = target.resolve()
    if target != root and root not in target.parents:
        return None, f"'path' must be inside the workspace: {path}"
    if not target.is_file():
        return None, f"'path' is not a file: {path}"
    return {"type": "local", "path": target.relative_to(root).as_posix()}, None


def _proposed_links(
    value: Any, *, ref_id: str,
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    """A proposal's declared edges, normalised, or why they are refused.

    Only SHAPE is refused (``links.link_errors``, the rule a catalog file
    obeys): an unknown ``rel``, a target that is not one id, a self-edge.
    A well-formed edge to an id the catalog does not hold is KEPT.  An
    agent writing several related pages proposes them one call at a time,
    often in parallel, so each page's siblings are not in the catalog, and
    may not even be claims yet, when its edges are checked.  Refusing them
    made cross-links between pages proposed together impossible.  What an
    unknown target might be (a sibling, a typo) is reported instead: to the
    agent by :func:`forward_links`, to the curator by :func:`link_warnings`,
    and the loader marks the edge dangling until its target arrives.
    """
    errors = link_errors(value, source_id=ref_id)
    if errors:
        return None, errors
    return [link.to_dict() for link in parse_links(value)], []


def pending_claim_ids(claims: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """``{reference id: claim id}`` for the claims in ``claims``, oldest first wins.

    What a link target not in the catalog may still be: a page proposed
    and not yet promoted.  ``claims`` are already checked (``is_claim``).
    """
    pending: Dict[str, str] = {}
    for claim in claims:
        pending.setdefault(claim["reference"]["id"], claim["claim_id"])
    return pending


def forward_links(
    entry_links: Any, ids: Iterable[str], pending: Mapping[str, str],
) -> List[Dict[str, str]]:
    """The edges of a proposal whose target is not in the catalog yet.

    Each is ``{to, rel}`` plus ``claim_id`` when the target is another
    pending claim.  An edge with neither is a page not proposed yet, or a
    typo; the agent is told which ones so it can tell the two apart.
    """
    known = set(ids)
    out: List[Dict[str, str]] = []
    for link in parse_links(entry_links):
        if link.to in known:
            continue
        item = {"to": link.to, "rel": link.rel}
        if link.to in pending:
            item["claim_id"] = pending[link.to]
        out.append(item)
    return out


def link_warnings(
    entry_links: Any, ids: Iterable[str],
    pending: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """What a curator should know about a claim's edges before promoting it.

    One thing: an edge whose target is not in ``ids``.  When the target is
    another pending claim (``pending``) the warning names it, so the
    curator knows promoting both resolves the edge.  Either way the edge is
    kept and marked dangling in the catalog, so this is a warning, never a
    reason Promote is refused.
    """
    known = set(ids)
    pending = pending or {}
    out: List[str] = []
    for link in parse_links(entry_links):
        if link.to in known:
            continue
        if link.to in pending:
            out.append(f"links to '{link.to}' ({link.rel}), proposed in claim "
                       f"{pending[link.to]} and not promoted yet; the edge "
                       "resolves when that claim is promoted")
        else:
            out.append(f"links to '{link.to}' ({link.rel}), which is not in this "
                       "workspace's catalog")
    return out


def build_proposed_reference(
    args: Dict[str, Any], *, workspace: str, catalog_ids: Iterable[str],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """The catalog entry a claim proposes, or the reasons it cannot be one.

    Returns ``(entry, [])`` on success and ``(None, errors)`` otherwise.  The
    entry passes ``validate_reference_file``, so promotion copies it as is.
    ``links`` (typed edges, ``links.py``) are carried when well-formed,
    whether or not their targets are in the catalog yet (see
    :func:`_proposed_links`).
    """
    ref_id = args.get("id")
    if not isinstance(ref_id, str) or not _ID_RE.match(ref_id):
        return None, ["'id' must be one token: letters, digits, '.', '_' or '-'."]
    if ref_id in set(catalog_ids):
        return None, [f"'{ref_id}' is already in the catalog; propose a different "
                      "id, or ask a curator to revise the existing reference."]
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, ["'name' is required."]
    description = args.get("description") or ""
    tags = _tag_list(args.get("tags"))
    if not isinstance(description, str) or tags is None:
        return None, ["'description' must be a string and 'tags' a list of "
                      "single-token strings (letters, digits, '.', '_', ':', '-')."]
    document, error = _resolve_document(args, workspace)
    if document is None:
        return None, [error or "invalid document"]
    links, link_problems = _proposed_links(args.get("links"), ref_id=ref_id)
    if links is None:
        return None, link_problems
    entry = {"id": ref_id, "name": name.strip(), "description": description,
             "mode": "selectable", "tags": tags, **document}
    if links:
        entry["links"] = links
    ok, errors, _warnings = validate_reference_file(entry)
    return (entry, []) if ok else (None, errors)


def new_claim(entry: Dict[str, Any], session: Any,
              rendered: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wrap a proposed entry in a claim record with its stamped origin.

    ``rendered`` is :func:`rendered_from`'s answer for the entry's file.
    """
    now = datetime.now(timezone.utc)
    claim_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    origin = proposing_origin(session, claim_id=claim_id, at=now.isoformat())
    origin.rendered_from = rendered or None
    return {"claim_id": claim_id, "status": CLAIM_STATUS_PROPOSED,
            "reference": entry, "origin": origin.to_dict()}


def file_digest(path: Path) -> Optional[str]:
    """SHA-256 of ``path``'s bytes, or ``None`` when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def rendered_from(
    entry: Mapping[str, Any], workspace: str,
    lookup: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """The ``origin.rendered_from`` stamp for a proposed ``entry``, or ``None``.

    ``lookup`` is the template plugin's ``render_record``: it answers what
    catalog template this session rendered a file from.  Only a ``local``
    entry (a ``path`` proposal) can have been rendered.  When the file no
    longer matches the render the stamp still names the template, with
    ``edited_after_render: true``, because the curator should know both
    that it started from a template and that it was changed since.
    """
    if not callable(lookup) or entry.get("type") != "local":
        return None
    rel = entry.get("path")
    if not isinstance(rel, str) or not rel:
        return None
    target = Path(workspace).resolve() / rel
    try:
        record = lookup(str(target))
    except Exception:  # noqa: BLE001 -- a stamp must not fail a proposal
        return None
    if not isinstance(record, dict) or not record.get("digest"):
        return None
    stamp = {k: record[k] for k in ("template", "template_id", "digest", "at") if record.get(k)}
    if file_digest(target) != record["digest"]:
        stamp["edited_after_render"] = True
    return stamp


def rendered_stamp_now(root: str, entry: Mapping[str, Any],
                       rendered: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """``rendered`` with ``edited_after_render`` re-decided from the file now.

    The recorded flag says what the proposing session saw; a page edited
    after the proposal would still read as unedited.  When the file can be
    compared, the answer replaces the recorded one; when it cannot, the
    recorded stamp is kept as it was.
    """
    if not rendered:
        return None
    stamp = dict(rendered)
    changed = rendered_file_changed(root, entry, rendered)
    if changed is True:
        stamp["edited_after_render"] = True
    elif changed is False:
        stamp.pop("edited_after_render", None)
    return stamp


def rendered_file_changed(root: str, entry: Mapping[str, Any],
                          rendered: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """Whether ``entry``'s file differs from the render its claim recorded.

    ``None`` when there is nothing to compare: no ``rendered_from`` stamp
    with a digest, not a ``local`` entry, or a file that cannot be read.
    The daemon asks this when it lists and promotes a claim, so a page
    edited after the proposal is flagged too.
    """
    digest = (rendered or {}).get("digest")
    rel = entry.get("path")
    if not isinstance(digest, str) or entry.get("type") != "local" or not isinstance(rel, str):
        return None
    now = file_digest(Path(root).resolve() / rel)
    if now is None:
        return None
    return now != digest


def write_claim(workspace: str, claim: Dict[str, Any]) -> Path:
    """Write ``claim`` atomically (temp file + ``os.replace``) and return its path.

    A claim is never rewritten in place: a second proposal is a second file,
    and reconciling two claims is the curator's job, not a write race.
    """
    directory = claims_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{claim['claim_id']}.json"
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".claim-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(claim, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return target


def load_claims(workspace: Optional[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Every readable, well-formed claim, oldest first, and the files skipped.

    A file that cannot be read, is not JSON, or does not carry a proposed
    entry is skipped and NAMED rather than dropped in silence, so a caller
    can say that a claim exists and could not be shown.
    """
    if not workspace:
        return [], []
    directory = claims_dir(workspace)
    if not directory.is_dir():
        return [], []
    claims: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped.append(path.name)
            continue
        if is_claim(data):
            claims.append(data)
        else:
            skipped.append(path.name)
    return claims, skipped


def is_claim(data: Any) -> bool:
    """A claim file this module would have written, checked on READ.

    The directory is model-writable, so shape is re-checked here rather
    than trusted: an id or claim id that is not one token would reach the
    listing outside the untrusted fence.
    """
    if not isinstance(data, dict) or data.get("status") != CLAIM_STATUS_PROPOSED:
        return False
    ref, claim_id = data.get("reference"), data.get("claim_id")
    return valid_id(claim_id) and isinstance(ref, dict) and valid_id(ref.get("id"))


def claim_tags(claim: Dict[str, Any]) -> List[str]:
    """The claim's tags that are single tokens; anything else is dropped."""
    tags = claim["reference"].get("tags")
    if not isinstance(tags, list):
        return []
    return [t for t in tags if isinstance(t, str) and _TAG_RE.match(t)]


def claim_as_args(claim: Dict[str, Any]) -> Dict[str, Any]:
    """The ``proposeReference`` arguments a claim's entry corresponds to.

    Promotion re-validates a claim by running it back through
    :func:`build_proposed_reference` -- the one door a proposal passed --
    because the claims directory is model-writable and the file may no
    longer be what ``proposeReference`` wrote.
    """
    ref = claim["reference"]
    args = {k: ref.get(k) for k in ("id", "name", "description", "tags")}
    if ref.get("links") is not None:
        args["links"] = ref.get("links")
    if ref.get("type") == "inline":
        args["content"] = ref.get("content")
    else:
        args["path"] = ref.get("path")
    return args


def listing_entry(claim: Dict[str, Any]) -> Dict[str, Any]:
    """How ``listReferences`` shows one claim to a model.

    The claim's free text -- name and description, written by a model and
    reviewed by nobody -- is shown only inside the untrusted-content
    boundary (``wrap_untrusted_content``), so an instruction planted in a
    proposal reads as data.  What is shown outside it is either one token
    re-checked on read (id, claim id, tags) or a path, and ``origin`` is the
    record the claim file carries, which the listing labels for what it is.
    """
    ref = claim["reference"]
    claim_id = claim["claim_id"]
    name = ref.get("name") if isinstance(ref.get("name"), str) else ""
    desc = ref.get("description") if isinstance(ref.get("description"), str) else ""
    text = f"{name}\n{desc}".strip()
    entry: Dict[str, Any] = {
        "claim_id": claim_id,
        "id": ref["id"],
        "status": CLAIM_STATUS_PROPOSED,
        "tags": claim_tags(claim),
        "unreviewed": wrap_untrusted_content(text, source=f"reference-claim:{claim_id}"),
        "claim_file": f".jaato/{CLAIMS_DIRNAME}/{claim_id}.json",
    }
    if ref.get("type") == "local" and isinstance(ref.get("path"), str):
        entry["path"] = ref["path"]
    links = claim_links(claim)
    if links:
        entry["links"] = links
    origin = ReferenceOrigin.from_dict(claim.get("origin"))
    if origin is not None:
        entry["origin"] = origin.to_dict()
    return entry


def claim_links(claim: Dict[str, Any]) -> List[Dict[str, str]]:
    """The claim's edges a MODEL is shown: ``{to, rel}``, the target re-checked.

    Outside the untrusted fence, so only what is one token is kept: a
    target that is not a valid id is dropped, and the free-text ``note`` is
    never shown here (the curator's listing carries it, rendered as text).
    """
    return [{"to": link.to, "rel": link.rel}
            for link in parse_links(claim["reference"].get("links"))
            if valid_id(link.to) and link.rel in LINK_RELS]
