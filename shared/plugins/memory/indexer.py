"""Indexer for memory plugin keyword extraction and matching."""

import re
from typing import Dict, List, Set

from .models import Memory, MemoryMetadata


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

        Args:
            memory: Memory object to index
        """
        # Store lightweight metadata
        metadata = MemoryMetadata(
            id=memory.id,
            description=memory.description,
            tags=memory.tags,
            timestamp=memory.timestamp
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

    def find_matches(self, keywords: List[str], limit: int = 5) -> List[MemoryMetadata]:
        """Find memories with tags matching the provided keywords.

        Matching strategy:
        1. Exact tag matches (keyword == tag)
        2. Partial matches (keyword in tag or tag in keyword)

        Results are sorted by recency (most recent first).

        Args:
            keywords: List of keywords to match against tags
            limit: Maximum number of matches to return

        Returns:
            List of MemoryMetadata objects (lightweight, no full content)
        """
        matched_ids: Set[str] = set()

        # Normalize keywords to lowercase for matching
        keywords_lower = [kw.lower() for kw in keywords]

        # Exact tag matches first
        for kw in keywords_lower:
            if kw in self._tag_index:
                matched_ids.update(self._tag_index[kw])

        # Partial matches (substring matching)
        for tag in self._tag_index:
            for kw in keywords_lower:
                # Check if keyword is substring of tag or vice versa
                if kw in tag or tag in kw:
                    matched_ids.update(self._tag_index[tag])
                    break  # Don't need to check other keywords for this tag

        # Get metadata and sort by recency (newest first)
        matches = [
            self._memories[mid]
            for mid in matched_ids
            if mid in self._memories
        ]
        matches.sort(key=lambda m: m.timestamp, reverse=True)

        return matches[:limit]

    def get_all_tags(self) -> List[str]:
        """Return all unique tags in the index.

        Returns:
            List of all tags (lowercase)
        """
        return list(self._tag_index.keys())

    def get_memory_count(self) -> int:
        """Return total number of indexed memories.

        Returns:
            Number of memories in index
        """
        return len(self._memories)

    def clear(self) -> None:
        """Clear all index data."""
        self._tag_index.clear()
        self._memories.clear()
