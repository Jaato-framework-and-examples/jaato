"""Indexer for memory plugin keyword extraction and matching.

The indexer maintains lightweight in-memory data structures for fast
tag-based lookup.  It is maturity-aware: ``find_matches`` defaults to
returning only *active* memories (``raw`` and ``validated``), filtering
out ``escalated`` and ``dismissed`` entries so that prompt enrichment
never surfaces memories that have graduated to references or been rejected.
"""

import re
from typing import Dict, List, Set

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

        Each tag is indexed under its **whole string** AND under each
        of its sub-tokens (split on non-alphanumeric characters).  This
        makes compound tags like ``workspace-baseline`` reachable from
        a prompt that only mentions ``workspace`` while still preserving
        the exact-match discipline in ``find_matches``.

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

        # Index by tags — both the full tag string and its sub-tokens.
        for tag in memory.tags:
            for key in self._tag_index_keys(tag):
                bucket = self._tag_index.setdefault(key, [])
                if memory.id not in bucket:
                    bucket.append(memory.id)

    @staticmethod
    def _tag_index_keys(tag: str) -> List[str]:
        """Return all index keys for a tag: the whole string plus sub-tokens.

        Sub-tokens are extracted by splitting on non-alphanumeric
        characters (hyphens, underscores, colons, dots), so
        ``workspace-baseline`` produces ``["workspace-baseline",
        "workspace", "baseline"]`` and ``agent:main`` produces
        ``["agent:main", "agent", "main"]``.

        Short sub-tokens (< 3 chars) are excluded to avoid noise from
        parts like ``"v2"`` or ``"ws"`` matching too broadly, unless
        the whole tag itself is that short (then it's kept as-is so
        intentionally-short tags like ``"gc"`` remain reachable via
        exact-match).
        """
        tag_lower = tag.lower()
        keys = [tag_lower]
        parts = re.split(r"[^a-z0-9]+", tag_lower)
        for p in parts:
            if p and p != tag_lower and len(p) >= 3 and p not in keys:
                keys.append(p)
        return keys

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

    def find_matches(
        self,
        keywords: List[str],
        limit: int = 5,
        *,
        active_only: bool = True,
        min_overlap: int = 1,
    ) -> List[MemoryMetadata]:
        """Find memories with tags matching the provided keywords.

        Uses **exact tag-index matching** — a keyword must exactly equal
        a key in the tag index (case-insensitive).  The index keys are
        the whole tag strings plus their sub-tokens (see
        ``_tag_index_keys``), so ``workspace`` in the prompt matches a
        memory tagged ``workspace-baseline`` through the sub-token
        expansion.  This keeps query-time logic O(1) per keyword while
        making compound tags reachable from single-word prompts.

        Results are ranked by overlap count (most matches first), then
        by recency within the same overlap count.

        Args:
            keywords: List of keywords to match against tag-index keys.
            limit: Maximum number of matches to return.
            active_only: When True (default), only return memories whose
                maturity is in ``ACTIVE_MATURITIES`` (raw, validated).
                Set to False to include all maturity states.
            min_overlap: Minimum number of tag matches required for a
                memory to be considered relevant (default: 1).  Single
                matches are acceptable — the specificity of the tags
                (and the length filter applied to extracted keywords)
                prevents noise.

        Returns:
            List of MemoryMetadata objects (lightweight, no full content)
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
