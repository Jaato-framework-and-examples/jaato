"""
Instruction Budget Tracking

Tracks token consumption by instruction source to help understand context budget
allocation and enable intelligent garbage collection based on source importance.

See docs/design-instruction-source-tracking.md for the full design.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any


class InstructionSource(Enum):
    """The 4 tracked instruction source layers."""
    SYSTEM = "system"           # System instructions (children: base, client, framework)
    PLUGIN = "plugin"           # Plugin instructions (children: per-tool)
    ENRICHMENT = "enrichment"   # Prompt enrichment pipeline additions
    CONVERSATION = "conversation"  # Message history (children: per-turn)


class GCPolicy(Enum):
    """Garbage collection policy for instruction sources."""
    LOCKED = "locked"           # Never GC - essential for operation
    PRESERVABLE = "preservable" # Prefer to keep, GC only under extreme pressure
    PARTIAL = "partial"         # Some parts GC-able (container with mixed children)
    EPHEMERAL = "ephemeral"     # Can be fully GC'd


# UI indicators for each policy
GC_POLICY_INDICATORS: Dict[GCPolicy, str] = {
    GCPolicy.LOCKED: "🔒",
    GCPolicy.PRESERVABLE: "◑",
    GCPolicy.PARTIAL: "◐",
    GCPolicy.EPHEMERAL: "○",
}


# Default GC policies per source
DEFAULT_SOURCE_POLICIES: Dict[InstructionSource, GCPolicy] = {
    InstructionSource.SYSTEM: GCPolicy.LOCKED,  # All children are LOCKED
    InstructionSource.PLUGIN: GCPolicy.PARTIAL,
    InstructionSource.ENRICHMENT: GCPolicy.EPHEMERAL,
    InstructionSource.CONVERSATION: GCPolicy.PARTIAL,
}


class SystemChildType(Enum):
    """Types of SYSTEM instruction children with their default GC policies."""
    BASE = "base"           # User-provided .jaato/system_instructions.md - LOCKED
    CLIENT = "client"       # Programmatic system_instructions param - LOCKED
    FRAMEWORK = "framework" # Task completion, parallel tool guidance - LOCKED


# Default GC policies per system child type
DEFAULT_SYSTEM_POLICIES: Dict[SystemChildType, GCPolicy] = {
    SystemChildType.BASE: GCPolicy.LOCKED,
    SystemChildType.CLIENT: GCPolicy.LOCKED,
    SystemChildType.FRAMEWORK: GCPolicy.LOCKED,
}


class ConversationTurnType(Enum):
    """Types of conversation turns with their default GC policies."""
    ORIGINAL_REQUEST = "original_request"   # User's initial request - LOCKED
    CLARIFICATION_Q = "clarification_q"     # Model's clarification question - PRESERVABLE
    CLARIFICATION_A = "clarification_a"     # User's clarification answer - PRESERVABLE
    TURN_SUMMARY = "turn_summary"           # Turn overview/conclusion - PRESERVABLE
    WORKING = "working"                     # Verbose working output - EPHEMERAL


# Default GC policies per conversation turn type
DEFAULT_TURN_POLICIES: Dict[ConversationTurnType, GCPolicy] = {
    ConversationTurnType.ORIGINAL_REQUEST: GCPolicy.LOCKED,
    ConversationTurnType.CLARIFICATION_Q: GCPolicy.PRESERVABLE,
    ConversationTurnType.CLARIFICATION_A: GCPolicy.PRESERVABLE,
    ConversationTurnType.TURN_SUMMARY: GCPolicy.PRESERVABLE,
    ConversationTurnType.WORKING: GCPolicy.EPHEMERAL,
}


class PluginToolType(Enum):
    """Types of plugin tools with their default GC policies."""
    CORE = "core"               # Always loaded, essential - LOCKED
    DISCOVERABLE = "discoverable"  # On-demand, can be re-discovered - EPHEMERAL


# Default GC policies per plugin tool type
DEFAULT_TOOL_POLICIES: Dict[PluginToolType, GCPolicy] = {
    PluginToolType.CORE: GCPolicy.LOCKED,
    PluginToolType.DISCOVERABLE: GCPolicy.EPHEMERAL,
}


@dataclass
class SourceEntry:
    """A single instruction source with its token count and GC policy.

    Attributes:
        source: The instruction source type.
        tokens: Token count for this entry (excluding children).
        gc_policy: Garbage collection policy for this entry.
        label: Display label (e.g., tool name, turn description).
        children: Child entries (e.g., per-tool for PLUGIN, per-turn for CONVERSATION).
        metadata: Optional metadata for richer context.
        created_at: Unix timestamp when this entry was created (for GC recency tracking).
        message_ids: List of message IDs associated with this entry (for GC history sync).
    """
    source: InstructionSource
    tokens: int
    gc_policy: GCPolicy
    label: Optional[str] = None
    children: Dict[str, "SourceEntry"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # GC integration fields
    created_at: Optional[float] = field(default_factory=time.time)
    message_ids: List[str] = field(default_factory=list)

    def total_tokens(self) -> int:
        """Total tokens including children."""
        if self.children:
            return sum(child.total_tokens() for child in self.children.values())
        return self.tokens

    def gc_eligible_tokens(self) -> int:
        """Tokens that can be reclaimed by GC."""
        if self.gc_policy == GCPolicy.LOCKED:
            return 0
        if self.gc_policy == GCPolicy.EPHEMERAL:
            return self.total_tokens()
        # PARTIAL or PRESERVABLE with children - recurse
        if self.children:
            return sum(child.gc_eligible_tokens() for child in self.children.values())
        # PRESERVABLE leaf - only under extreme pressure
        if self.gc_policy == GCPolicy.PRESERVABLE:
            return 0  # Not counted as easily reclaimable
        return 0

    def locked_tokens(self) -> int:
        """Tokens that can never be GC'd."""
        if self.gc_policy == GCPolicy.LOCKED:
            return self.total_tokens()
        if self.children:
            return sum(child.locked_tokens() for child in self.children.values())
        return 0

    def preservable_tokens(self) -> int:
        """Tokens that are preservable (GC only under extreme pressure)."""
        if self.gc_policy == GCPolicy.PRESERVABLE:
            return self.total_tokens()
        if self.children:
            return sum(child.preservable_tokens() for child in self.children.values())
        return 0

    def effective_gc_policy(self) -> GCPolicy:
        """Determine effective GC policy considering children.

        If no children, returns the stored gc_policy.
        If children exist, aggregates their effective policies:
        - All same policy → use that policy
        - Mixed policies → PARTIAL
        """
        if not self.children:
            return self.gc_policy

        # Collect effective policies from all children (recursive)
        child_policies = {
            child.effective_gc_policy() for child in self.children.values()
        }

        # All children share the same policy → use that
        if len(child_policies) == 1:
            return child_policies.pop()

        # Mixed policies → PARTIAL
        return GCPolicy.PARTIAL

    def indicator(self) -> str:
        """Get the UI indicator for this entry's effective GC policy."""
        return GC_POLICY_INDICATORS.get(self.effective_gc_policy(), "?")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "source": self.source.value,
            "tokens": self.tokens,
            "total_tokens": self.total_tokens(),
            "gc_policy": self.effective_gc_policy().value,
            "gc_eligible_tokens": self.gc_eligible_tokens(),
            "indicator": self.indicator(),
        }
        if self.label:
            result["label"] = self.label
        if self.metadata:
            result["metadata"] = self.metadata
        if self.created_at is not None:
            result["created_at"] = self.created_at
        if self.message_ids:
            result["message_ids"] = self.message_ids
        if self.children:
            result["children"] = {
                key: child.to_dict() for key, child in self.children.items()
            }
        return result


@dataclass
class InstructionBudget:
    """Tracks token usage by instruction source for an agent.

    Attributes:
        session_id: Server-managed session ID (umbrella that groups all agents).
                    This is what clients connect to, not JaatoSession.
        agent_id: Identifier for this agent within the session ("main", "explore-1", etc.)
        agent_type: Type of agent ("main", "explore", "plan", etc.) for display purposes.
        entries: Token usage broken down by instruction source.
        context_limit: Model's context window size.
    """
    session_id: str = ""
    agent_id: str = "main"
    agent_type: Optional[str] = None
    entries: Dict[InstructionSource, SourceEntry] = field(default_factory=dict)
    context_limit: int = 128_000  # Model's context window

    def total_tokens(self) -> int:
        """Total tokens across all sources."""
        return sum(entry.total_tokens() for entry in self.entries.values())

    def gc_eligible_tokens(self) -> int:
        """Total tokens that can be reclaimed by GC."""
        return sum(entry.gc_eligible_tokens() for entry in self.entries.values())

    def locked_tokens(self) -> int:
        """Total tokens that can never be GC'd."""
        return sum(entry.locked_tokens() for entry in self.entries.values())

    def preservable_tokens(self) -> int:
        """Total tokens that are preservable."""
        return sum(entry.preservable_tokens() for entry in self.entries.values())

    def utilization_percent(self) -> float:
        """Context window utilization as percentage."""
        if self.context_limit == 0:
            return 0.0
        return (self.total_tokens() / self.context_limit) * 100

    def available_tokens(self) -> int:
        """Tokens still available in the context window."""
        return max(0, self.context_limit - self.total_tokens())

    def gc_headroom_percent(self) -> float:
        """
        Percentage of context that could be freed by GC.
        Higher = more room to reclaim if needed.
        """
        if self.context_limit == 0:
            return 0.0
        return (self.gc_eligible_tokens() / self.context_limit) * 100

    # --- Entry Management ---

    def set_entry(
        self,
        source: InstructionSource,
        tokens: int,
        gc_policy: Optional[GCPolicy] = None,
        label: Optional[str] = None,
        children: Optional[Dict[str, SourceEntry]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SourceEntry:
        """Set or replace an entry for a source."""
        policy = gc_policy or DEFAULT_SOURCE_POLICIES.get(source, GCPolicy.EPHEMERAL)
        entry = SourceEntry(
            source=source,
            tokens=tokens,
            gc_policy=policy,
            label=label,
            children=children or {},
            metadata=metadata or {},
        )
        self.entries[source] = entry
        return entry

    def get_entry(self, source: InstructionSource) -> Optional[SourceEntry]:
        """Get the entry for a source, if it exists."""
        return self.entries.get(source)

    def update_tokens(self, source: InstructionSource, tokens: int) -> None:
        """Update the token count for an existing source."""
        if source in self.entries:
            self.entries[source].tokens = tokens

    def add_child(
        self,
        source: InstructionSource,
        child_key: str,
        tokens: int,
        gc_policy: GCPolicy,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
        message_ids: Optional[List[str]] = None,
    ) -> Optional[SourceEntry]:
        """Add a child entry to a source (e.g., tool to PLUGIN, turn to CONVERSATION).

        Args:
            source: The parent source type.
            child_key: Unique key for this child within the parent.
            tokens: Token count for this child entry.
            gc_policy: Garbage collection policy.
            label: Display label (defaults to child_key).
            metadata: Optional metadata dict.
            created_at: Creation timestamp (defaults to current time).
            message_ids: List of message IDs associated with this entry.
        """
        parent = self.entries.get(source)
        if not parent:
            return None
        child = SourceEntry(
            source=source,  # Inherits parent's source type
            tokens=tokens,
            gc_policy=gc_policy,
            label=label or child_key,
            metadata=metadata or {},
            created_at=created_at if created_at is not None else time.time(),
            message_ids=message_ids or [],
        )
        parent.children[child_key] = child
        return child

    def remove_child(self, source: InstructionSource, child_key: str) -> bool:
        """Remove a child entry from a source."""
        parent = self.entries.get(source)
        if parent and child_key in parent.children:
            del parent.children[child_key]
            return True
        return False

    # --- Serialization ---

    def snapshot(self) -> Dict[str, Any]:
        """Serializable snapshot for UI events."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "context_limit": self.context_limit,
            "total_tokens": self.total_tokens(),
            "gc_eligible_tokens": self.gc_eligible_tokens(),
            "locked_tokens": self.locked_tokens(),
            "preservable_tokens": self.preservable_tokens(),
            "utilization_percent": round(self.utilization_percent(), 2),
            "available_tokens": self.available_tokens(),
            "gc_headroom_percent": round(self.gc_headroom_percent(), 2),
            "entries": {
                source.value: entry.to_dict()
                for source, entry in self.entries.items()
            },
        }

    def get_conversation_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get serializable snapshot of conversation budget for persistence.

        Only saves the CONVERSATION source, as other sources (SYSTEM, PLUGIN,
        ENRICHMENT) are automatically recreated when the session is restored.
        """
        conv_entry = self.entries.get(InstructionSource.CONVERSATION)
        if not conv_entry:
            return None
        return conv_entry.to_dict()

    def restore_conversation_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore conversation budget from a saved snapshot.

        Recreates the CONVERSATION source entry with its children (turn entries)
        from a previously saved snapshot.
        """
        if not snapshot:
            return

        def parse_entry(data: Dict[str, Any]) -> SourceEntry:
            """Recursively parse a SourceEntry from serialized dict."""
            gc_policy = GCPolicy(data.get("gc_policy", "ephemeral"))
            children: Dict[str, SourceEntry] = {}
            if "children" in data:
                for child_key, child_data in data["children"].items():
                    children[child_key] = parse_entry(child_data)
            return SourceEntry(
                source=InstructionSource.CONVERSATION,
                tokens=data.get("tokens", 0),
                gc_policy=gc_policy,
                label=data.get("label"),
                children=children,
                metadata=data.get("metadata", {}),
                created_at=data.get("created_at"),
                message_ids=data.get("message_ids", []),
            )

        # Parse and restore the conversation entry
        entry = parse_entry(snapshot)
        self.entries[InstructionSource.CONVERSATION] = entry

    # --- Convenience Factories ---

    @classmethod
    def create_default(
        cls,
        session_id: str,
        agent_id: str = "main",
        agent_type: Optional[str] = None,
        context_limit: int = 128_000,
    ) -> "InstructionBudget":
        """Create a budget with default empty entries for all sources."""
        budget = cls(
            session_id=session_id,
            agent_id=agent_id,
            agent_type=agent_type or agent_id,
            context_limit=context_limit,
        )
        for source in InstructionSource:
            budget.set_entry(source, tokens=0)
        return budget


# --- Helper Functions ---

def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count from text length.

    Default assumes ~4 characters per token (reasonable for English text).
    For more accurate counts, use the model's actual tokenizer.
    """
    if not text:
        return 0
    return int(len(text) / chars_per_token)
