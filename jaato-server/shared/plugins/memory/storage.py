"""Storage backend for memory plugin.

Handles JSONL persistence with backward compatibility for memories created
before the knowledge-curation lifecycle fields (maturity, confidence, scope,
evidence, source_agent, source_session) were added.  Old records missing
these fields are loaded with sensible defaults.
"""

import json
from dataclasses import asdict, fields as dc_fields
from pathlib import Path
from typing import Iterable, List, Optional, Set

from .models import (
    ACTIVE_MATURITIES,
    MATURITY_RAW,
    Memory,
)


# Fields that exist on the Memory dataclass.  Used to silently drop unknown
# keys from old or hand-edited JSONL lines rather than crashing on
# ``TypeError: __init__() got an unexpected keyword argument``.
_MEMORY_FIELD_NAMES: Set[str] = {f.name for f in dc_fields(Memory)}


def _memory_from_dict(data: dict) -> Memory:
    """Construct a ``Memory`` from a raw JSON dict.

    Provides backward compatibility by:
    - Dropping unknown keys that may exist in hand-edited files.
    - Letting dataclass defaults fill in any missing new fields so that
      memories created before the knowledge-curation fields were added
      load without error.
    """
    filtered = {k: v for k, v in data.items() if k in _MEMORY_FIELD_NAMES}
    return Memory(**filtered)


class MemoryStorage:
    """JSONL-based storage for memories.

    Each memory is stored as a JSON line in the file.
    This format allows for easy appending and sequential reading.

    Supports the knowledge-curation lifecycle by providing maturity-based
    query methods used by the advisor agent during curation.
    """

    def __init__(self, path: str):
        """Initialize storage with file path.

        Directory creation is deferred to the first write operation
        (save/update/delete) so that read-only usage against a
        not-yet-existing path does not create stale directories
        (e.g., before set_workspace_path corrects the storage location).

        Args:
            path: Path to JSONL file for storing memories
        """
        self.path = Path(path)

    def _ensure_parent_dir(self) -> None:
        """Create parent directory if it does not exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, memory: Memory) -> None:
        """Append memory to JSONL file.

        Args:
            memory: Memory object to store
        """
        self._ensure_parent_dir()
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(memory)) + '\n')

    def load_all(self) -> List[Memory]:
        """Load all memories from file.

        Backward-compatible: old JSONL lines missing the new lifecycle
        fields are loaded with their dataclass defaults (``maturity="raw"``,
        ``confidence=0.5``, ``scope="project"``, etc.).

        Returns:
            List of Memory objects, or empty list if file doesn't exist
        """
        if not self.path.exists():
            return []

        memories = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        memories.append(_memory_from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError) as e:
                        # Log but continue - don't let one bad line break everything
                        print(f"[MemoryStorage] Warning: Skipping invalid line: {e}")
                        continue
        return memories

    # ------------------------------------------------------------------
    # Tag-based search
    # ------------------------------------------------------------------

    def search_by_tags(
        self,
        tags: List[str],
        limit: int = 3,
        *,
        active_only: bool = True,
    ) -> List[Memory]:
        """Find memories matching any of the provided tags.

        Memories are scored by tag overlap and sorted by:
        1. Number of matching tags (descending)
        2. Recency (most recent first)

        Args:
            tags: List of tags to search for.
            limit: Maximum number of memories to return.
            active_only: When True (default), only return memories whose
                maturity is in ``ACTIVE_MATURITIES`` (raw, validated).
                Set to False to search across all maturity states.

        Returns:
            List of Memory objects matching the tags, sorted by relevance
        """
        all_memories = self.load_all()

        # Score by tag overlap
        scored = []
        for mem in all_memories:
            if active_only and mem.maturity not in ACTIVE_MATURITIES:
                continue
            overlap = len(set(mem.tags) & set(tags))
            if overlap > 0:
                scored.append((overlap, mem))

        # Sort by score desc, then by recency (timestamp desc)
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        return [mem for _, mem in scored[:limit]]

    # ------------------------------------------------------------------
    # Maturity-based queries (used by advisor / curation workflow)
    # ------------------------------------------------------------------

    def search_by_maturity(
        self,
        maturities: Iterable[str],
        limit: int = 50,
    ) -> List[Memory]:
        """Return memories whose maturity is in the given set.

        Args:
            maturities: Maturity values to include (e.g. ``{"raw"}``).
            limit: Maximum number of memories to return.

        Returns:
            Matching memories sorted by recency (newest first).
        """
        target = set(maturities)
        matches = [m for m in self.load_all() if m.maturity in target]
        matches.sort(key=lambda m: m.timestamp, reverse=True)
        return matches[:limit]

    def get_pending_curation(self, limit: int = 50) -> List[Memory]:
        """Return raw memories awaiting advisor review.

        Convenience wrapper over ``search_by_maturity`` for the most
        common curation query.

        Args:
            limit: Maximum number of memories to return.

        Returns:
            Raw memories sorted by recency (newest first).
        """
        return self.search_by_maturity({MATURITY_RAW}, limit=limit)

    def count_by_maturity(self) -> dict:
        """Return a dict of ``{maturity: count}`` for all stored memories.

        Useful for curation dashboards and threshold-based triggers.
        """
        counts: dict = {}
        for mem in self.load_all():
            counts[mem.maturity] = counts.get(mem.maturity, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # CRUD (single-record)
    # ------------------------------------------------------------------

    def update(self, memory: Memory) -> None:
        """Update an existing memory.

        This is inefficient for JSONL (requires rewriting entire file),
        but acceptable for small memory stores. For larger stores,
        consider using a proper database.

        Args:
            memory: Memory object with updated fields
        """
        all_memories = self.load_all()

        # Find and update the memory
        updated = False
        for i, mem in enumerate(all_memories):
            if mem.id == memory.id:
                all_memories[i] = memory
                updated = True
                break

        if not updated:
            # Memory not found - append it
            all_memories.append(memory)

        # Rewrite entire file
        self._ensure_parent_dir()
        with open(self.path, 'w', encoding='utf-8') as f:
            for mem in all_memories:
                f.write(json.dumps(asdict(mem)) + '\n')

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: Unique identifier of the memory

        Returns:
            Memory object if found, None otherwise
        """
        for mem in self.load_all():
            if mem.id == memory_id:
                return mem
        return None

    def count(self) -> int:
        """Return total number of stored memories.

        Returns:
            Number of memories in storage
        """
        return len(self.load_all())

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: Unique identifier of the memory to delete

        Returns:
            True if memory was deleted, False if not found
        """
        all_memories = self.load_all()

        # Find and remove the memory
        original_count = len(all_memories)
        all_memories = [mem for mem in all_memories if mem.id != memory_id]

        if len(all_memories) == original_count:
            return False  # Memory not found

        # Rewrite entire file
        self._ensure_parent_dir()
        with open(self.path, 'w', encoding='utf-8') as f:
            for mem in all_memories:
                f.write(json.dumps(asdict(mem)) + '\n')

        return True
