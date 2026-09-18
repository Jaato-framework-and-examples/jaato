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
from typing import Any, Dict, List, Optional


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
#: Maturities that count a memory as ACTIVE — eligible to be surfaced.
#: Answers "should this be used?", and RAW is deliberately in it: a raw
#: memory is usable, it just has not been vetted.
ACTIVE_MATURITIES = frozenset({MATURITY_RAW, MATURITY_VALIDATED})

#: Maturities that mean a memory has LEFT the raw queue — the curator has
#: made a decision about it.
#:
#: A SEPARATE set from :data:`ACTIVE_MATURITIES` on purpose, because it
#: answers a different question.  ``MemoryStore.update`` used the active set
#: for this test, and since RAW is active, ANY update of a raw memory --
#: including a usage-counter bump on retrieval -- moved it into the curated
#: store still marked raw.  The curator's own discovery call emptied the queue
#: it was reading, and an unvetted memory became enrichment material by being
#: LOOKED AT.  Measured: 18 raw against 1 validated in the curated store.
#:
#: One set answering two questions is the same defect as a value meaning two
#: things; the fix is a second name, not a smarter condition.
PROMOTES_OUT_OF_RAW = frozenset({MATURITY_VALIDATED, MATURITY_ESCALATED})

#: Maturities that mean a curator APPROVED the memory, so ``curated_by``
#: should carry their stamp (#1123).
#:
#: A THIRD set, for the reason the comment above gives about the second:
#: it answers its own question.  ``PROMOTES_OUT_OF_RAW`` asks *has the
#: curator decided about this* -- which a DISMISSAL also answers, and a
#: dismissal is precisely not an approval.  Reusing it would stamp
#: ``curated_by`` on memories the curator rejected, and
#: ``require_curation`` reads that field as permission to surface one.
#:
#: The two sets happen to have the same members today.  That is not a
#: reason to share a name: they would diverge the moment a maturity is
#: added that is a decision and not an approval, and the sharing would
#: make the divergence silent.
CURATED_MATURITIES = frozenset({MATURITY_VALIDATED, MATURITY_ESCALATED})


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
        generated_by: WHICH MODEL wrote it -- ``{"kind": "ai", "provider",
            "model", "session_id", "agent_id"}``, the shape
            :func:`jaato_sdk.events.ai_generated_by` mints (#1123).
            jaato's learning loop is this plugin: the model writes
            memories during a session and they are re-injected into later
            sessions' prompts, which is what Art. 15(4) means by a system
            that "continues to learn after being placed on the market".
            A memory that does not record which model wrote it cannot be
            audited when that model turns out to have been wrong.
            **Stamped by the PLUGIN, never by the model** -- provenance a
            subject asserts about itself is not provenance -- and ``None``
            on a record written before #1123, which reads as *provenance
            unknown* and never as human-authored.
        curated_by: WHO APPROVED it -- the curator's own stamp, set when a
            raw memory is promoted.  A SECOND field rather than an
            overwrite of ``generated_by``: who wrote it and who approved
            it are two facts, and collapsing them loses the one an
            auditor asks for.  ``None`` = never curated, which is what
            ``plugin_configs.memory.require_curation`` gates on.
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
    generated_by: Optional[Dict[str, Any]] = None
    curated_by: Optional[Dict[str, Any]] = None

    @property
    def is_curated(self) -> bool:
        """Whether a curator has approved this memory (Art. 15(4), #1123).

        Read by the ``require_curation`` gate.  Deliberately NOT derived
        from ``maturity``: ``validated`` and ``escalated`` are the
        curator's own vocabulary and a deployment may set them by hand or
        by script, while this asks the narrower question *did the
        curation step run and leave its mark*.  Absent is not curated.
        """
        return bool(self.curated_by)

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
