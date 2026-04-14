"""Indexer for memory plugin keyword extraction and matching.

The indexer maintains lightweight in-memory data structures for fast
tag-based lookup.  It is maturity-aware: ``find_matches`` defaults to
returning only *active* memories (``raw`` and ``validated``), filtering
out ``escalated`` and ``dismissed`` entries so that prompt enrichment
never surfaces memories that have graduated to references or been rejected.
"""

import re
from typing import Dict, List, Optional, Set

from .models import ACTIVE_MATURITIES, Memory, MemoryMetadata


# Common English stopwords to filter out
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "can", "about", "how", "what", "when", "where",
    "which", "who", "why", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "them", "their", "my", "your", "our"
}


class MemoryIndexer:
    """Keyword extraction and tag indexing for efficient memory lookup.

    The indexer maintains in-memory data structures for fast matching:
    - tag_index: Maps tags to memory IDs
    - memories: Maps memory IDs to metadata (lightweight, no full content)

    Maturity filtering is applied at query time (``find_matches``) rather
    than at indexing time so that the full index is always available for
    administrative queries (e.g. listing all tags including escalated ones).
    """

    def __init__(self):
        """Initialize empty index."""
        self._tag_index: Dict[str, List[str]] = {}  # tag -> [memory_id, ...]
        self._memories: Dict[str, MemoryMetadata] = {}  # id -> metadata

    def build_index(self, memories: List[Memory]) -> None:
        """Build index from existing memories.

        Args:
            memories: List of Memory objects to index
        """
        for mem in memories:
            self.index_memory(mem)

    def index_memory(self, memory: Memory) -> None:
        """Add a single memory to the index.

        All memories are indexed regardless of maturity so that
        administrative tools (``list_memory_tags``, ``memory list``) see
        the complete picture.  Maturity filtering happens at query time.

        Each tag is indexed under its **whole canonical string only**
        (lowercased).  Sub-token expansion is applied at query time
        through paragraph-coherence matching — see
        ``find_matches_in_text``.

        Args:
            memory: Memory object to index
        """
        # Store lightweight metadata (now includes maturity, confidence, scope)
        metadata = MemoryMetadata(
            id=memory.id,
            description=memory.description,
            tags=memory.tags,
            timestamp=memory.timestamp,
            maturity=memory.maturity,
            confidence=memory.confidence,
            scope=memory.scope,
        )
        self._memories[memory.id] = metadata

        # Index by canonical tag (whole, lowercased).
        for tag in memory.tags:
            key = tag.lower()
            bucket = self._tag_index.setdefault(key, [])
            if memory.id not in bucket:
                bucket.append(memory.id)

    @staticmethod
    def _tag_components(tag: str) -> List[str]:
        """Split a tag into its ≥3-char sub-tokens (lowercased).

        ``workspace-baseline`` → ``["workspace", "baseline"]``
        ``agent:main`` → ``["agent", "main"]``
        ``api`` → ``["api"]``
        ``plan-v2`` → ``["plan"]`` (``v2`` dropped as too short)

        Returns an empty list if the tag has no usable components, in
        which case the caller should fall back to whole-string matching.
        """
        parts = re.split(r"[^a-z0-9]+", tag.lower())
        return [p for p in parts if len(p) >= 3]

    @staticmethod
    def _paragraph_word_sets(text: str) -> List[Set[str]]:
        """Split text into paragraphs, return a set of words per paragraph.

        Paragraph separator is one or more blank lines (``\\n\\s*\\n+``).
        Single newlines (e.g. inside a bullet list) keep lines in the
        same paragraph — they're still one topical block.

        Each paragraph's word set includes both whole tokens (with
        internal hyphens/underscores/colons/dots) AND alphanumeric
        sub-words, so a paragraph mentioning ``workspace-baseline``
        contributes ``workspace-baseline``, ``workspace``, ``baseline``.
        """
        paragraphs = re.split(r"\n\s*\n+", text)
        result: List[Set[str]] = []
        for p in paragraphs:
            p_lower = p.lower()
            whole = set(re.findall(r"[a-z0-9][a-z0-9\-_:.]*[a-z0-9]", p_lower))
            sub = set(re.findall(r"\b\w+\b", p_lower))
            result.append(whole | sub)
        return result

    @classmethod
    def _tag_coherent_in_paragraphs(
        cls,
        tag: str,
        paragraphs: List[Set[str]],
    ) -> bool:
        """Check if a tag is topically present in any paragraph.

        Matching rules:

        - **Whole-tag match** — if the full tag string (lowercased)
          appears verbatim in any paragraph, it matches.  Covers
          short/atomic tags like ``api`` or ``gc`` and also catches
          users who typed the whole compound tag.
        - **Component coherence** — split the tag into ≥3-char
          components.  For tags with 1–2 components, **all** must
          appear in the same paragraph.  For 3+ components, a
          **majority** (``ceil(n/2)``) suffices, since long compound
          tags rarely have every part repeated together.

        Word-distance is irrelevant — paragraph membership is the
        cohesion signal.
        """
        tag_lower = tag.lower()
        if any(tag_lower in p for p in paragraphs):
            return True

        components = cls._tag_components(tag)
        if not components:
            return False

        n = len(components)
        threshold = n if n <= 2 else (n + 1) // 2  # all if ≤2, majority if ≥3

        for paragraph in paragraphs:
            present = sum(1 for c in components if c in paragraph)
            if present >= threshold:
                return True
        return False

    def extract_keywords(self, prompt: str) -> List[str]:
        """Extract potential keywords from a prompt.

        Two extraction passes so compound tags are reachable both as
        whole strings and as their parts:

        1. **Whole tokens** including non-alphanumerics —
           ``workspace-baseline`` or ``agent:main`` become a single
           keyword.
        2. **Sub-words** from the same token — ``workspace-baseline``
           also contributes ``workspace`` and ``baseline``.

        Both are lowercased, filtered against stopwords, and capped to
        ``≥ 3 chars`` (matches the sub-token minimum in
        ``_tag_index_keys`` so exact lookups line up).

        Args:
            prompt: User prompt text

        Returns:
            De-duplicated list of keywords (whole tokens first).
        """
        prompt_lower = prompt.lower()

        # Whole tokens: alphanumerics with internal hyphens/underscores/colons/dots
        whole = re.findall(r"[a-z0-9][a-z0-9\-_:.]*[a-z0-9]", prompt_lower)
        # Sub-words: just alphanumeric runs
        sub = re.findall(r"\b\w+\b", prompt_lower)

        seen = set()
        keywords: List[str] = []
        for w in whole + sub:
            if w in seen:
                continue
            if w in STOPWORDS:
                continue
            if len(w) < 3:
                continue
            seen.add(w)
            keywords.append(w)
        return keywords

    def find_matches_in_text(
        self,
        text: str,
        limit: int = 5,
        *,
        active_only: bool = True,
    ) -> List[MemoryMetadata]:
        """Find memories whose tags are coherent in any paragraph of text.

        For every canonical tag in the index, runs
        ``_tag_coherent_in_paragraphs``.  A memory surfaces if any of
        its tags is coherent in at least one paragraph of the prompt
        or tool result.

        Results are ranked by **number of matching tags** (more matches
        = more relevant), then by recency.

        Cost is O(unique_tags × paragraphs × tag_components) per call,
        which for hundreds of tags and short prompts is sub-millisecond.

        Args:
            text: Free-form text to analyse (user prompt, tool result, …).
            limit: Maximum number of matches to return.
            active_only: When True (default), restrict to memories with
                ``raw`` or ``validated`` maturity.

        Returns:
            List of ``MemoryMetadata`` (lightweight, no full content),
            ordered by relevance and recency.
        """
        paragraphs = self._paragraph_word_sets(text)
        if not paragraphs or not any(paragraphs):
            return []

        # tag (canonical) → list of memory IDs that have this tag
        # We need to know which tags matched per memory to score by tag count.
        memory_tag_hits: Dict[str, int] = {}
        for tag_key, memory_ids in self._tag_index.items():
            if not self._tag_coherent_in_paragraphs(tag_key, paragraphs):
                continue
            for mid in memory_ids:
                memory_tag_hits[mid] = memory_tag_hits.get(mid, 0) + 1

        # Build (tag_count, MemoryMetadata) tuples, applying maturity filter.
        scored = []
        for mid, count in memory_tag_hits.items():
            meta = self._memories.get(mid)
            if meta is None:
                continue
            if active_only and meta.maturity not in ACTIVE_MATURITIES:
                continue
            scored.append((count, meta))

        # Sort by recency first, then stable-sort by tag count (count wins).
        scored.sort(key=lambda pair: pair[1].timestamp, reverse=True)
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [meta for _, meta in scored[:limit]]

    def find_matches(
        self,
        keywords: List[str],
        limit: int = 5,
        *,
        active_only: bool = True,
        min_overlap: int = 1,
    ) -> List[MemoryMetadata]:
        """Find memories whose tags exactly match any of the keywords.

        **Exact whole-tag matching only** — a keyword must equal a
        canonical tag (case-insensitive).  This is the legacy path
        useful when the caller has already extracted distinct keywords
        and wants pure tag-set intersection (e.g. tests, programmatic
        callers).

        For free-form text (prompts, tool results), use
        :meth:`find_matches_in_text`, which applies paragraph-coherence
        matching so compound tags like ``workspace-baseline`` surface
        only when their components co-occur in a paragraph.

        Results are ranked by overlap count, then by recency.

        Args:
            keywords: List of keywords to compare against canonical tags.
            limit: Maximum number of matches to return.
            active_only: Restrict to ``raw`` / ``validated`` maturity
                (default).
            min_overlap: Minimum number of tag matches per memory
                (default 1).

        Returns:
            List of ``MemoryMetadata`` ordered by relevance.
        """
        # Normalize keywords to a set for O(1) lookup
        keywords_set = {kw.lower() for kw in keywords}

        # Count exact tag matches per memory
        overlap_counts: Dict[str, int] = {}
        for kw in keywords_set:
            if kw in self._tag_index:
                for memory_id in self._tag_index[kw]:
                    overlap_counts[memory_id] = overlap_counts.get(memory_id, 0) + 1

        # Filter by min_overlap, maturity, and build result list
        matches = []
        for mid, count in overlap_counts.items():
            if count < min_overlap:
                continue
            meta = self._memories.get(mid)
            if meta is None:
                continue
            if active_only and meta.maturity not in ACTIVE_MATURITIES:
                continue
            matches.append((count, meta))

        # Sort by overlap count descending, then by recency (newest first)
        matches.sort(key=lambda pair: pair[1].timestamp, reverse=True)  # recency first
        matches.sort(key=lambda pair: pair[0], reverse=True)  # stable: overlap count wins

        return [meta for _, meta in matches[:limit]]

    def get_all_tags(self, *, active_only: bool = False) -> List[str]:
        """Return all unique tags in the index.

        Args:
            active_only: When True, only include tags from memories with
                active maturity states.  Defaults to False (all tags).

        Returns:
            List of all tags (lowercase)
        """
        if not active_only:
            return list(self._tag_index.keys())

        # Collect tags only from active memories
        active_tags: Set[str] = set()
        for tag, mem_ids in self._tag_index.items():
            for mid in mem_ids:
                meta = self._memories.get(mid)
                if meta and meta.maturity in ACTIVE_MATURITIES:
                    active_tags.add(tag)
                    break
        return list(active_tags)

    def get_memory_count(self, *, active_only: bool = False) -> int:
        """Return total number of indexed memories.

        Args:
            active_only: When True, only count memories with active
                maturity states.

        Returns:
            Number of memories in index
        """
        if not active_only:
            return len(self._memories)
        return sum(
            1 for m in self._memories.values()
            if m.maturity in ACTIVE_MATURITIES
        )

    def clear(self) -> None:
        """Clear all index data."""
        self._tag_index.clear()
        self._memories.clear()
