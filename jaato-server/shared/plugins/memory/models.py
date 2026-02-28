"""Data models for memory plugin.

Memories go through a maturity lifecycle managed by the knowledge curation
system ("The School"):

    raw → validated → escalated
                   ↘ dismissed

- **raw**: Fresh from an agent session, not yet reviewed by the advisor.
- **validated**: Advisor reviewed and confirmed valuable; retained as a memory.
- **escalated**: Promoted to a reference entry; no longer surfaced in prompt
  enrichment (the reference takes over).
- **dismissed**: Advisor reviewed and rejected (incorrect, trivial, or
  superseded); kept for audit trail but not surfaced.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# Valid maturity states for the knowledge curation lifecycle.
MATURITY_RAW = "raw"
MATURITY_VALIDATED = "validated"
MATURITY_ESCALATED = "escalated"
MATURITY_DISMISSED = "dismissed"

VALID_MATURITIES = frozenset({
    MATURITY_RAW,
    MATURITY_VALIDATED,
    MATURITY_ESCALATED,
    MATURITY_DISMISSED,
})

# Valid scope values indicating how broadly a memory applies.
SCOPE_PROJECT = "project"
SCOPE_UNIVERSAL = "universal"

VALID_SCOPES = frozenset({SCOPE_PROJECT, SCOPE_UNIVERSAL})

# Maturity states that should be surfaced in prompt enrichment hints.
# Escalated memories are represented by their reference entry instead;
# dismissed memories are hidden from the model entirely.
ACTIVE_MATURITIES = frozenset({MATURITY_RAW, MATURITY_VALIDATED})


@dataclass
class Memory:
    """A stored memory with full content and metadata.

    Lifecycle:
        Memories are created with ``maturity="raw"`` by working agents during
        sessions.  The advisor agent later reviews raw memories and transitions
        them to ``validated`` (keep), ``escalated`` (promoted to reference), or
        ``dismissed`` (rejected).

    Attributes:
        id: Unique identifier for this memory.
        content: Full explanation/content to be stored.
        description: Brief summary of what this memory contains.
        tags: Keywords for retrieval and matching.
        timestamp: ISO format timestamp when memory was created.
        usage_count: Number of times this memory has been retrieved.
        last_accessed: ISO format timestamp of last retrieval (optional).
        maturity: Lifecycle stage — one of ``raw``, ``validated``,
            ``escalated``, ``dismissed``.  Defaults to ``raw``.
        confidence: Agent's self-assessed confidence in the accuracy of this
            memory, from 0.0 (uncertain) to 1.0 (certain).
        scope: How broadly this memory applies — ``project`` (specific to this
            codebase) or ``universal`` (generalizable).
        evidence: What triggered this learning — error messages, tool results,
            or other observations that substantiate the memory.
        source_agent: Name or profile of the agent that created this memory.
        source_session: Session ID where this memory was created.
    """
    id: str
    content: str
    description: str
    tags: List[str]
    timestamp: str
    usage_count: int = 0
    last_accessed: Optional[str] = None
    maturity: str = MATURITY_RAW
    confidence: float = 0.5
    scope: str = SCOPE_PROJECT
    evidence: Optional[str] = None
    source_agent: Optional[str] = None
    source_session: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Whether this memory should be surfaced in prompt enrichment.

        Escalated and dismissed memories are not surfaced — escalated ones
        are represented by their reference entry, dismissed ones are hidden.
        """
        return self.maturity in ACTIVE_MATURITIES


@dataclass
class MemoryMetadata:
    """Lightweight metadata for prompt enrichment.

    Used during prompt enrichment to provide hints without loading full content.
    Only memories with active maturity states (``raw``, ``validated``) are
    included in enrichment hints.

    Attributes:
        id: Unique identifier for this memory.
        description: Brief summary of what this memory contains.
        tags: Keywords for retrieval and matching.
        timestamp: ISO format timestamp when memory was created.
        maturity: Lifecycle stage for filtering during enrichment.
        confidence: Agent's self-assessed confidence (0.0–1.0).
        scope: ``project`` or ``universal``.
    """
    id: str
    description: str
    tags: List[str]
    timestamp: str
    maturity: str = MATURITY_RAW
    confidence: float = 0.5
    scope: str = SCOPE_PROJECT
