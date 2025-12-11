"""Storage backend for memory plugin."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .models import Memory


class MemoryStorage:
    """JSONL-based storage for memories.

    Each memory is stored as a JSON line in the file.
    This format allows for easy appending and sequential reading.
    """

    def __init__(self, path: str):
        """Initialize storage with file path.

        Args:
            path: Path to JSONL file for storing memories
        """
        self.path = Path(path)
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, memory: Memory) -> None:
        """Append memory to JSONL file.

        Args:
            memory: Memory object to store
        """
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(memory)) + '\n')

    def load_all(self) -> List[Memory]:
        """Load all memories from file.

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
                        memories.append(Memory(**json.loads(line)))
                    except (json.JSONDecodeError, TypeError) as e:
                        # Log but continue - don't let one bad line break everything
                        print(f"[MemoryStorage] Warning: Skipping invalid line: {e}")
                        continue
        return memories

    def search_by_tags(self, tags: List[str], limit: int = 3) -> List[Memory]:
        """Find memories matching any of the provided tags.

        Memories are scored by tag overlap and sorted by:
        1. Number of matching tags (descending)
        2. Recency (most recent first)

        Args:
            tags: List of tags to search for
            limit: Maximum number of memories to return

        Returns:
            List of Memory objects matching the tags, sorted by relevance
        """
        all_memories = self.load_all()

        # Score by tag overlap
        scored = []
        for mem in all_memories:
            overlap = len(set(mem.tags) & set(tags))
            if overlap > 0:
                scored.append((overlap, mem))

        # Sort by score desc, then by recency (timestamp desc)
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        return [mem for _, mem in scored[:limit]]

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
