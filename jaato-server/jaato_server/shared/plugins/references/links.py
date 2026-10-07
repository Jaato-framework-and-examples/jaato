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
see-also       NOT expanded; offered as a hint on the selection (``related``):
               a neighbouring concern useful alongside (coexisting
               alternatives, two sides of one topic), neither detailing the
               other (#1472)
supersedes     the older target is routed to the source (declared on the NEWER
               reference): a selection or expansion reaching it gets the
               newer one instead, never as well
contradicts    never expanded and never hinted to a working agent; listed
=============  ===========================================================

Inference is KEPT beside declaration, and the declared edge wins for its
pair: a mention of B in A that A declares ``elaborates`` (or ``see-also``)
does not expand B.
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
REL_SEE_ALSO = "see-also"

#: Every relation: its meaning and what a selection does with it.  The ONE
#: table; ``LINK_RELS``, the tool schema's description and ``explain plugin
#: references`` are all derived from it, so they cannot disagree.
REL_DOCS: Dict[str, Dict[str, Any]] = {
    REL_DEPENDS_ON: {
        "meaning": "a reader of the source needs the target too",
        "selection": "expanded: the target is selected with the source",
        "expands": True},
    REL_ELABORATES: {
        "meaning": "the target goes deeper into the source's subject",
        "selection": "not expanded; offered as an optional neighbour (related)",
        "expands": False},
    REL_SEE_ALSO: {
        "meaning": "the target covers a neighbouring concern useful alongside "
                   "(a coexisting alternative, the other side of one topic)",
        "selection": "not expanded; offered as an optional neighbour (related)",
        "expands": False},
    REL_SUPERSEDES: {
        "meaning": "the source replaces the target (declared on the newer one)",
        "selection": "a request for the target gets the source instead",
        "expands": False},
    REL_CONTRADICTS: {
        "meaning": "the source disagrees with the target",
        "selection": "never expanded and never offered; listed for a curator",
        "expands": False},
}

#: The closed vocabulary, in the order it is documented.
LINK_RELS = tuple(REL_DOCS)

#: Relations whose target a selection pulls in.
EXPANDING_RELS = frozenset(r for r, d in REL_DOCS.items() if d["expands"])

#: Relations whose target a selection OFFERS (``related``), never pulls in.
OFFERED_RELS = frozenset({REL_ELABORATES, REL_SEE_ALSO})


def rel_summary() -> str:
    """One line per relation, ``'rel': meaning (selection effect)``, for prose."""
    return " ".join(f"'{r}': {d['meaning']} ({d['selection']})." for r, d in REL_DOCS.items())

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
        """Offered targets (``elaborates`` / ``see-also``) the selection does not hold."""
        chosen = set(selected_ids)
        out = []
        for src in sorted(chosen):
            for link in self.outbound.get(src, []):
                if link.rel in OFFERED_RELS and link.to in self.ids and link.to not in chosen:
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

FRONTIER_RANKING_SIMILARITY = (
    "declared depends-on edges first, then similarity to the selection, then "
    "references more of the previous depth points to, then id"
)


def rank_frontier(
    candidates: Mapping[str, Set[str]],
    index: LinkIndex,
    similarity: Optional[Mapping[str, float]] = None,
) -> List[str]:
    """One depth's newly reached references, most wanted first.

    ``candidates`` maps each id reached at this depth to the ids at the
    previous depth that reached it.  The order matters only when
    ``max_transitive_references`` cuts the depth -- it decides which
    references survive -- and it is:

    1. how many of those parents DECLARE ``depends-on`` to it: an author
       said the parent is not comprehensible without it, which neither a
       mention nor a vector says;
    2. with ``similarity``, how close its vector is to the selection's: the
       nearest references rather than the best-linked ones.  The caller
       passes scores only when EVERY candidate has one (see
       ``ReferencesPlugin._similarity_to_selection``): a candidate with no
       vector cannot be compared, and ranking it below every indexed one
       would prefer a reference for being in an indexed bundle;
    3. how many parents reached it at all: a reference several selected
       documents point to is more central to the selection than one only
       one of them names;
    4. the id, so the order is total and reproducible (it reaches the
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

    def nearness(cid: str) -> float:
        return -similarity[cid] if similarity is not None else 0.0

    return sorted(candidates, key=lambda cid: (-declared_votes(cid), nearness(cid), -len(candidates[cid]), cid))
