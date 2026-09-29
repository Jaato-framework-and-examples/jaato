"""Declared, TYPED edges between references (the wikiLLM brainstorm, §5 Seam 3).

A reference already pulls in its neighbourhood: ``_resolve_transitive_references``
walks the ids and paths each body MENTIONS.  A mention is an untyped edge
nobody authored: renaming an id silently deletes every inbound one, and
"mentioned" is the only relation there is.  A reference may now also DECLARE
edges, beside its tags::

    "links": [{"to": "adr-004", "rel": "supersedes", "note": "why"}]

The vocabulary is closed, and each ``rel`` decides the expansion policy --
which is the payoff, since it makes traversal cheaper and better at once:

=============  ===========================================================
``rel``        what a selection does with it
=============  ===========================================================
depends-on     expanded: the source is not comprehensible without the target
elaborates     NOT expanded; offered as a hint on the selection (``related``)
supersedes     the older target is routed to the source (declared on the NEWER
               reference): a selection or expansion reaching it gets the
               newer one instead, never as well
contradicts    never expanded and never hinted to a working agent; listed
=============  ===========================================================

Inference is KEPT beside declaration, and the declared edge wins for its
pair: a mention of B in A that A declares ``elaborates`` does not expand B.
A declared ``depends-on`` expands even a source with no readable body (a URL
or MCP reference), which inference never could.

A dangling edge -- one whose target is not in the catalog -- is kept and
marked, never dropped: losing it silently is the defect the declaration
exists to end.  The index is computed from the catalog when asked; the plugin
reassigns its catalog in many places, and a cached index would go stale.

Stdlib only; no plugin state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

REL_DEPENDS_ON = "depends-on"
REL_ELABORATES = "elaborates"
REL_SUPERSEDES = "supersedes"
REL_CONTRADICTS = "contradicts"

#: The closed vocabulary, in the order it is documented.
LINK_RELS = (REL_DEPENDS_ON, REL_ELABORATES, REL_SUPERSEDES, REL_CONTRADICTS)

#: Relations whose target a selection pulls in.
EXPANDING_RELS = frozenset({REL_DEPENDS_ON})

#: Longest ``to`` / ``note`` accepted; an edge is a pointer, not a document.
MAX_LINK_TARGET_CHARS = 256
MAX_LINK_NOTE_CHARS = 1000


@dataclass(frozen=True)
class ReferenceLink:
    """One declared edge from the reference that carries it.

    Attributes:
        to: The target reference's id.
        rel: One of :data:`LINK_RELS`.
        note: Why the edge exists, for a reader; optional.
    """

    to: str
    rel: str
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"to": self.to, "rel": self.rel}
        if self.note:
            out["note"] = self.note
        return out


def _is_target(value: Any) -> bool:
    return (isinstance(value, str) and bool(value.strip())
            and len(value) <= MAX_LINK_TARGET_CHARS
            and not any(c in value for c in "\r\n\t"))


def link_errors(links: Any, *, source_id: Any = None) -> List[str]:
    """Why ``links`` (a reference file's ``links`` value) is malformed, or ``[]``.

    ``None`` (absent) is fine.  An unknown ``rel`` is an error, not a
    mention: a relation nobody can act on is one a reader would trust.
    """
    if links is None:
        return []
    if not isinstance(links, list):
        return ["'links' must be an array"]
    errors: List[str] = []
    for i, link in enumerate(links):
        where = f"links[{i}]"
        if not isinstance(link, dict):
            errors.append(f"{where} must be an object with 'to' and 'rel'")
            continue
        if not _is_target(link.get("to")):
            errors.append(f"{where}.to must be a reference id")
        elif link.get("to") == source_id:
            errors.append(f"{where}.to names the reference itself")
        if link.get("rel") not in LINK_RELS:
            errors.append(f"{where}.rel must be one of: {', '.join(LINK_RELS)}")
        note = link.get("note")
        if note is not None and (not isinstance(note, str) or len(note) > MAX_LINK_NOTE_CHARS):
            errors.append(f"{where}.note must be a string of at most {MAX_LINK_NOTE_CHARS} characters")
        unknown = set(link) - {"to", "rel", "note"}
        if unknown:
            errors.append(f"{where} has unknown keys: {', '.join(sorted(unknown))}")
    return errors


def parse_links(value: Any) -> List[ReferenceLink]:
    """The well-formed edges in ``value``; a malformed entry is left out.

    Loading is lenient (a bad edge must not cost the reference);
    :func:`link_errors` is where a malformed one is reported.
    """
    if not isinstance(value, list):
        return []
    out: List[ReferenceLink] = []
    for link in value:
        if not isinstance(link, dict) or link.get("rel") not in LINK_RELS:
            continue
        if not _is_target(link.get("to")):
            continue
        note = link.get("note")
        out.append(ReferenceLink(to=link["to"], rel=link["rel"],
                                 note=note if isinstance(note, str) and note else None))
    return out


class LinkIndex:
    """The declared edges of one catalog, both directions.

    Built from any iterable of objects carrying ``id`` and ``links``
    (``ReferenceSource``).  A source repeated under one id keeps the last
    copy's edges, the way the catalog itself resolves a duplicate id.
    """

    def __init__(self, sources: Iterable[Any]) -> None:
        self.ids: Set[str] = set()
        self.outbound: Dict[str, List[ReferenceLink]] = {}
        for source in sources:
            self.ids.add(source.id)
            self.outbound[source.id] = list(getattr(source, "links", None) or [])
        self.inbound: Dict[str, List[Dict[str, str]]] = {}
        #: older id -> the ids that declare they supersede it
        self.successors: Dict[str, List[str]] = {}
        for src in sorted(self.outbound):
            for link in self.outbound[src]:
                self.inbound.setdefault(link.to, []).append({"from": src, "rel": link.rel})
                if link.rel == REL_SUPERSEDES:
                    self.successors.setdefault(link.to, []).append(src)

    def declared_targets(self, source_id: str) -> Set[str]:
        """Every id ``source_id`` declares an edge to, whatever the ``rel``."""
        return {link.to for link in self.outbound.get(source_id, [])}

    def expanding_targets(self, source_id: str) -> Set[str]:
        """The targets a selection pulls in: declared ``depends-on``, in the catalog."""
        return {link.to for link in self.outbound.get(source_id, [])
                if link.rel in EXPANDING_RELS and link.to in self.ids}

    def current_version(self, ref_id: str) -> str:
        """``ref_id``, or the reference that supersedes it, followed to the end.

        Stops at an id nobody supersedes, at one superseded by more than one
        reference (ambiguous: routed nowhere rather than guessed), and on a
        cycle, where ``ref_id`` itself is returned: a chain that loops names
        no current version, and routing to whichever member was reached
        last would depend on where the request entered it.
        """
        seen = {ref_id}
        current = ref_id
        while True:
            newer = [s for s in self.successors.get(current, []) if s in self.ids]
            if len(newer) != 1:
                return current
            if newer[0] in seen:
                return ref_id  # a cycle names no current version
            current = newer[0]
            seen.add(current)

    def links_of(self, source_id: str) -> List[Dict[str, Any]]:
        """``source_id``'s declared edges, each marked ``dangling`` when its target is absent."""
        out = []
        for link in self.outbound.get(source_id, []):
            entry = link.to_dict()
            if link.to not in self.ids:
                entry["dangling"] = True
            out.append(entry)
        return out

    def linked_from(self, target_id: str) -> List[Dict[str, str]]:
        """The declared edges pointing AT ``target_id`` (the reverse index)."""
        return list(self.inbound.get(target_id, []))

    def dangling(self) -> List[Dict[str, str]]:
        """Every declared edge whose target is not in the catalog."""
        return [{"from": src, "to": link.to, "rel": link.rel}
                for src in sorted(self.outbound) for link in self.outbound[src]
                if link.to not in self.ids]

    def related(self, selected_ids: Iterable[str]) -> List[Dict[str, str]]:
        """``elaborates`` targets of the selection that the selection does not hold."""
        chosen = set(selected_ids)
        out = []
        for src in sorted(chosen):
            for link in self.outbound.get(src, []):
                if link.rel == REL_ELABORATES and link.to in self.ids and link.to not in chosen:
                    out.append({"id": link.to, "rel": link.rel, "from": src})
        return out


def expansion_neighbours(
    source_id: str, mentions: Optional[Set[str]], index: LinkIndex,
) -> Optional[Set[str]]:
    """What one node expands to: its mentions, retyped by what it declares.

    * a mentioned id the node DECLARES an edge to follows the declared
      ``rel`` (only ``depends-on`` expands) -- the declared edge wins;
    * a declared ``depends-on`` expands whether or not it is mentioned, and
      whether or not the node's body could be read;
    * every neighbour is routed to its current version (``supersedes``).

    ``None`` (the node's body could not be read, ``mentions is None``) stays
    ``None`` only when the node declares nothing to expand either.
    """
    expanding = index.expanding_targets(source_id)
    if mentions is None and not expanding:
        return None
    base = (set(mentions or ()) - index.declared_targets(source_id)) | expanding
    return {index.current_version(n) for n in base}


#: How :func:`rank_frontier` orders one depth's candidates, in words, for the
#: truncation record: a reader of a cut neighbourhood should know what the
#: cut preferred.
FRONTIER_RANKING = (
    "declared depends-on edges first, then references more of the previous "
    "depth points to, then id"
)


def rank_frontier(
    candidates: Mapping[str, Set[str]], index: LinkIndex,
) -> List[str]:
    """One depth's newly reached references, most wanted first.

    ``candidates`` maps each id reached at this depth to the ids at the
    previous depth that reached it.  The order matters only when
    ``max_transitive_references`` cuts the depth -- it decides which
    references survive -- and it is:

    1. how many of those parents DECLARE ``depends-on`` to it: an author
       said the parent is not comprehensible without it, which no mention
       says;
    2. how many parents reached it at all: a reference several selected
       documents point to is more central to the selection than one only
       one of them names;
    3. the id, so the order is total and reproducible (it reaches the
       prompt-cache prefix, and a set's order varies across processes).

    A declared target is counted by its current version, the same routing
    :func:`expansion_neighbours` applies, so the two agree about which id
    a declaration names.
    """
    def declared_votes(cid: str) -> int:
        return sum(
            1 for parent in candidates[cid]
            if cid in {index.current_version(t) for t in index.expanding_targets(parent)}
        )

    return sorted(candidates, key=lambda cid: (-declared_votes(cid), -len(candidates[cid]), cid))
