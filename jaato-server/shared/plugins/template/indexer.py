"""Template indexer for tag-based prompt and tool-result enrichment.

Models the same matching contract used by the memory and references plugins
so all three surfaces share one mental model: tags drive contextual surfacing
during conversation via ``shared.tag_coherence``.

Templates are tagged upstream by the ``gen-references`` agent and the tags
arrive on each :class:`TemplateIndexEntry` via the persisted ``index.json``.
Runtime-discovered templates (scanned from referenced directories without an
index file) carry no tags and are not surfaced contextually — they remain
accessible via ``listAvailableTemplates`` and ``renderTemplateToFile``.

This module deliberately omits the lifecycle complexity that ``memory`` has
(maturity states) and the dual-store split (workspace vs global): templates
form a single flat catalog whose only filter is presence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from .plugin import TemplateIndexEntry


class TemplateIndexer:
    """Tag index over template entries, mirroring :class:`MemoryIndexer`.

    Each entry's tags are stored under their canonical (lowercased)
    whole-string key.  Sub-token expansion is applied only at query time
    via :func:`shared.tag_coherence.tag_coherent_in_segments`, so the
    indexer stays cheap to rebuild and adding new tag schemes does not
    require reindexing.
    """

    def __init__(self) -> None:
        # Canonical tag → list of template names carrying it.
        self._tag_index: Dict[str, List[str]] = {}
        # Template name → entry (back-pointer for result hydration).
        self._entries: Dict[str, "TemplateIndexEntry"] = {}

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, entries: List["TemplateIndexEntry"]) -> None:
        """Replace the index contents with the supplied entries.

        Re-indexing is cheaper than incremental updates for the typical
        catalog size (low hundreds of templates), so callers re-build
        whenever the underlying catalog changes (template added/removed
        in a referenced directory, gen-references regenerated).
        """
        self._tag_index.clear()
        self._entries.clear()
        for entry in entries:
            self.index_entry(entry)

    def index_entry(self, entry: "TemplateIndexEntry") -> None:
        """Add a single template to the index.

        Untagged entries are kept in ``_entries`` (so the indexer's view
        matches the on-disk catalog) but contribute nothing to
        ``_tag_index`` and therefore never surface via tag matching.
        """
        self._entries[entry.name] = entry
        for tag in entry.tags:
            key = tag.lower()
            bucket = self._tag_index.setdefault(key, [])
            if entry.name not in bucket:
                bucket.append(entry.name)

    def remove_entry(self, name: str) -> None:
        """Drop a template from both indices.  Idempotent."""
        entry = self._entries.pop(name, None)
        if entry is None:
            return
        for tag in entry.tags:
            bucket = self._tag_index.get(tag.lower())
            if not bucket:
                continue
            try:
                bucket.remove(name)
            except ValueError:
                pass
            if not bucket:
                self._tag_index.pop(tag.lower(), None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find_matches_in_text(
        self,
        text: str,
        limit: int = 5,
    ) -> List["TemplateIndexEntry"]:
        """Return templates whose tags are coherent with the supplied text.

        For every canonical tag in the index, runs the shared
        :func:`shared.tag_coherence.tag_coherent_in_segments` check
        against the segments of *text*.  A template surfaces if at least
        one of its tags is coherent in any segment.  Results are ranked
        by number of matching tags (more = more relevant), with stable
        order across calls so the agent sees consistent listings turn
        over turn.

        Args:
            text: Free-form text to analyse — a user prompt or a tool
                result (the same matcher is used on both surfaces).
            limit: Cap on returned entries.

        Returns:
            List of :class:`TemplateIndexEntry` (already-resolved
            entries from the catalog), most relevant first.
        """
        from shared.tag_coherence import text_segments, tag_coherent_in_segments

        if not self._tag_index:
            return []

        segments = text_segments(text)
        if not segments:
            return []

        # Tally tag hits per template name.
        hits: Dict[str, int] = {}
        for tag_key, names in self._tag_index.items():
            if not tag_coherent_in_segments(tag_key, segments):
                continue
            for name in names:
                hits[name] = hits.get(name, 0) + 1

        # Score = (tag-match count desc, name asc) for stable ordering.
        scored = [
            (count, self._entries[name])
            for name, count in hits.items()
            if name in self._entries
        ]
        scored.sort(key=lambda pair: pair[1].name)
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def triggering_tags(self, text: str, entries: List["TemplateIndexEntry"]) -> List[str]:
        """Return the tags from *entries* that actually triggered against *text*.

        Used by enrichment to display only the tags that drove each
        match (not every tag attached to the matched entries) — same
        truthful-hint convention used by the memory plugin.
        """
        from shared.tag_coherence import text_segments, tag_coherent_in_segments

        segments = text_segments(text)
        if not segments:
            return []

        seen: Set[str] = set()
        triggered: List[str] = []
        for entry in entries:
            for tag in entry.tags:
                key = tag.lower()
                if key in seen:
                    continue
                if tag_coherent_in_segments(tag, segments):
                    seen.add(key)
                    triggered.append(tag)
        return triggered

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_all_tags(self) -> List[str]:
        """Return all unique canonical tags currently indexed."""
        return list(self._tag_index.keys())

    def get_template_count(self) -> int:
        """Return total number of indexed templates (tagged or otherwise)."""
        return len(self._entries)
