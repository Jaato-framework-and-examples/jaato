"""Data models for memory plugin."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Memory:
    """A stored memory with full content and metadata.

    Attributes:
        id: Unique identifier for this memory
        content: Full explanation/content to be stored
        description: Brief summary of what this memory contains
        tags: Keywords for retrieval and matching
        timestamp: ISO format timestamp when memory was created
        usage_count: Number of times this memory has been retrieved
        last_accessed: ISO format timestamp of last retrieval (optional)
    """
    id: str
    content: str
    description: str
    tags: List[str]
    timestamp: str
    usage_count: int = 0
    last_accessed: Optional[str] = None


@dataclass
class MemoryMetadata:
    """Lightweight metadata for prompt enrichment.

    Used during prompt enrichment to provide hints without loading full content.

    Attributes:
        id: Unique identifier for this memory
        description: Brief summary of what this memory contains
        tags: Keywords for retrieval and matching
        timestamp: ISO format timestamp when memory was created
    """
    id: str
    description: str
    tags: List[str]
    timestamp: str
