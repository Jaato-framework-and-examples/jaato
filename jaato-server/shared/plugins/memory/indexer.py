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

        # Index by tags
        for tag in memory.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = []
            if memory.id not in self._tag_index[tag_lower]:
                self._tag_index[tag_lower].append(memory.id)

    def extract_keywords(self, prompt: str) -> List[str]:
        """Extract potential keywords from a prompt.

        Simple approach:
        1. Extract words (alphanumeric sequences)
        2. Convert to lowercase
        3. Filter out stopwords
        4. Filter out very short words (< 4 chars)

        Args:
            prompt: User prompt text

        Returns:
            List of extracted keywords
        """
        # Extract words (including underscores for technical terms)
        words = re.findall(r'\b\w+\b', prompt.lower())

        # Filter stopwords and short words
        keywords = [
            w for w in words
            if w not in STOPWORDS and len(w) >= 4
        ]

        return keywords

    def find_matches(
        self,
        keywords: List[str],
        limit: int = 5,
        *,
        active_only: bool = True,
        min_overlap: int = 2,
    ) -> List[MemoryMetadata]:
        """Find memories with tags matching the provided keywords.

        Uses **exact tag matching only** — a keyword must exactly equal
        a tag (case-insensitive) to count as a match.  Memories must
        have at least ``min_overlap`` matching tags to be returned.
        This prevents false positives from substring matches against
        large prompts where short tags like ``"test"`` would match
        thousands of unrelated words.

        Results are ranked by overlap count (most matches first), then
        by recency within the same overlap count.

        Args:
            keywords: List of keywords to match against tags.
            limit: Maximum number of matches to return.
            active_only: When True (default), only return memories whose
                maturity is in ``ACTIVE_MATURITIES`` (raw, validated).
                Set to False to include all maturity states.
            min_overlap: Minimum number of tag matches required for a
                memory to be considered relevant (default: 2).

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
