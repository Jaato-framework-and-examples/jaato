# GC Plugins and Budget Backend Integration Design

## Overview

This document describes the integration between GC (Garbage Collection) plugins and the InstructionBudget backend, ensuring budget information stays consistent with context compaction and enabling policy-aware garbage collection.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Granularity** | Finer (source/child level) | Budget tracks per-tool and per-turn, GC should match |
| **Synchronization** | Option C: GC returns removal list | Clean separation - GC decides what, session executes |
| **Ephemeral recency** | Add timestamps to SourceEntry | Enables "keep most recent ephemeral" logic |
| **Target threshold** | Add `gc_target_percent` | Defines where to land after GC, not just when to trigger |

## Changes Required

### 1. SourceEntry: Add Timestamp

```python
# shared/instruction_budget.py

@dataclass
class SourceEntry:
    source: InstructionSource
    tokens: int
    gc_policy: GCPolicy
    label: Optional[str] = None
    children: Dict[str, "SourceEntry"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = None  # NEW: Unix timestamp for recency tracking

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
```

Update `to_dict()` and `restore_conversation_from_snapshot()` to include `created_at`.

### 2. GCResult: Add Removal List

```python
# shared/plugins/gc/base.py

@dataclass
class GCRemovalItem:
    """Describes a single item to be removed from budget."""
    source: InstructionSource
    child_key: Optional[str] = None  # None = remove entire source, str = specific child
    tokens_freed: int = 0
    reason: str = ""  # e.g., "ephemeral", "oldest_turn", "summarized"


@dataclass
class GCResult:
    success: bool
    items_collected: int
    tokens_before: int
    tokens_after: int
    plugin_name: str
    trigger_reason: GCTriggerReason
    notification: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # NEW: Structured removal list for budget synchronization
    removal_list: List[GCRemovalItem] = field(default_factory=list)
```

### 3. GCConfig: Add Target Threshold

```python
# shared/plugins/gc/base.py

@dataclass
class GCConfig:
    # Existing trigger settings
    threshold_percent: float = 80.0  # When to trigger GC

    # NEW: Target after GC
    target_percent: float = field(
        default_factory=lambda: float(os.getenv('JAATO_GC_TARGET', '60.0'))
    )
    """Target context usage after GC. GC will try to reach this level."""

    # NEW: Pressure threshold for touching PRESERVABLE content
    pressure_percent: float = field(
        default_factory=lambda: float(os.getenv('JAATO_GC_PRESSURE', '90.0'))
    )
    """When usage exceeds this, PRESERVABLE content may be collected."""

    # Existing settings...
    max_turns: Optional[int] = None
    auto_trigger: bool = True
    check_before_send: bool = True
    preserve_recent_turns: int = 5
    pinned_turn_indices: List[int] = field(default_factory=list)
    plugin_config: Dict[str, Any] = field(default_factory=dict)
```

### 4. GCPlugin Protocol: Accept Budget

Update the `collect()` signature to optionally accept budget:

```python
# shared/plugins/gc/base.py

@runtime_checkable
class GCPlugin(Protocol):
    # ... existing methods ...

    def collect(
        self,
        history: List[Message],
        context_usage: Dict[str, Any],
        config: GCConfig,
        reason: GCTriggerReason,
        budget: Optional[InstructionBudget] = None  # NEW: Optional budget for policy-aware GC
    ) -> Tuple[List[Message], GCResult]:
        """Perform garbage collection.

        Args:
            history: Current conversation history.
            context_usage: Current context window usage stats.
            config: GC configuration.
            reason: Why this collection was triggered.
            budget: Optional budget for policy-aware decisions.

        Returns:
            Tuple of (new_history, result) where result includes removal_list.
        """
        ...
```

### 5. JaatoSession: Handle Removal List

```python
# shared/jaato_session.py

def _execute_gc(self, reason: GCTriggerReason) -> Optional[GCResult]:
    """Execute garbage collection and sync budget."""

    # Call GC plugin with budget
    new_history, result = self._gc_plugin.collect(
        history=self._history,
        context_usage=self.get_context_usage(),
        config=self._gc_config,
        reason=reason,
        budget=self._instruction_budget  # Pass budget
    )

    if result.success:
        # Update history
        self._history = new_history

        # Sync budget from removal list
        self._apply_gc_removal_list(result.removal_list)

        # Emit budget update
        self._emit_instruction_budget_update()

    return result

def _apply_gc_removal_list(self, removal_list: List[GCRemovalItem]) -> None:
    """Apply removal list to budget."""
    if not self._instruction_budget:
        return

    for item in removal_list:
        if item.child_key:
            # Remove specific child
            self._instruction_budget.remove_child(item.source, item.child_key)
        else:
            # Clear entire source (rare)
            entry = self._instruction_budget.get_entry(item.source)
            if entry:
                entry.tokens = 0
                entry.children.clear()
```

---

## New Plugin: gc_budget (Policy-Aware)

### Priority Order

The gc_budget plugin respects GC policies in this priority order:

```
Removal Priority (first to go):
1a. ENRICHMENT (bulk clear)
    - All enrichment content removed at once
    - Regenerated each turn anyway

1b. Other EPHEMERAL entries (oldest first)
    - Working output in conversation
    - Discoverable tool schemas

2. PARTIAL entries (conversation turns, oldest first)
   - Respect preserve_recent_turns
   - Respect pinned_turn_indices

3. PRESERVABLE entries (only under extreme pressure > pressure_percent)
   - Clarification Q&A
   - Turn summaries

4. LOCKED entries (never touched)
   - System instructions
   - Original user request
   - Core tool schemas
```

### Algorithm

```python
def collect(self, history, context_usage, config, reason, budget):
    """Policy-aware garbage collection."""

    if not budget:
        # Fall back to simple truncation
        return self._fallback_truncate(history, context_usage, config, reason)

    current_tokens = budget.total_tokens()
    target_tokens = int(budget.context_limit * config.target_percent / 100)
    tokens_to_free = current_tokens - target_tokens

    if tokens_to_free <= 0:
        return history, GCResult(success=True, items_collected=0, ...)

    removal_list = []
    tokens_freed = 0

    # Phase 1a: Bulk clear ENRICHMENT (always ephemeral, regenerated each turn)
    enrichment_entry = budget.get_entry(InstructionSource.ENRICHMENT)
    if enrichment_entry and enrichment_entry.total_tokens() > 0:
        removal_list.append(GCRemovalItem(
            source=InstructionSource.ENRICHMENT,
            child_key=None,  # Bulk clear entire source
            tokens_freed=enrichment_entry.total_tokens(),
            reason="enrichment_bulk_clear"
        ))
        tokens_freed += enrichment_entry.total_tokens()

    # Phase 1b: Remove other EPHEMERAL entries (oldest first, keep most recent)
    if tokens_freed < tokens_to_free:
        ephemeral_candidates = self._get_ephemeral_candidates(budget, exclude_enrichment=True)
        # Sort by created_at (oldest first)
        sorted_candidates = sorted(ephemeral_candidates, key=lambda e: e.created_at)

        for entry in sorted_candidates:
            if tokens_freed >= tokens_to_free:
                break
            removal_list.append(GCRemovalItem(
                source=entry.source,
                child_key=entry.label,
                tokens_freed=entry.total_tokens(),
                reason="ephemeral"
            ))
            tokens_freed += entry.total_tokens()

    # Phase 2: Remove old PARTIAL turns
    if tokens_freed < tokens_to_free:
        turn_candidates = self._get_partial_turn_candidates(budget, config)
        for turn_key, entry in turn_candidates:
            if tokens_freed >= tokens_to_free:
                break
            removal_list.append(GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key=turn_key,
                tokens_freed=entry.total_tokens(),
                reason="partial_turn"
            ))
            tokens_freed += entry.total_tokens()

    # Phase 3: PRESERVABLE (only if usage > pressure_percent)
    if tokens_freed < tokens_to_free and context_usage['percent_used'] >= config.pressure_percent:
        preservable_candidates = self._get_preservable_candidates(budget)
        for entry in preservable_candidates:
            if tokens_freed >= tokens_to_free:
                break
            removal_list.append(GCRemovalItem(
                source=entry.source,
                child_key=entry.label,
                tokens_freed=entry.total_tokens(),
                reason="preservable_under_pressure"
            ))
            tokens_freed += entry.total_tokens()

    # Apply removals to history
    new_history = self._apply_removals_to_history(history, removal_list, budget)

    return new_history, GCResult(
        success=True,
        items_collected=len(removal_list),
        tokens_before=current_tokens,
        tokens_after=current_tokens - tokens_freed,
        plugin_name=self.name,
        trigger_reason=reason,
        removal_list=removal_list,
        details={
            "enrichment_cleared": any(r.reason == "enrichment_bulk_clear" for r in removal_list),
            "ephemeral_removed": sum(1 for r in removal_list if r.reason == "ephemeral"),
            "partial_removed": sum(1 for r in removal_list if r.reason == "partial_turn"),
            "preservable_removed": sum(1 for r in removal_list if r.reason == "preservable_under_pressure"),
        }
    )
```

### Configuration

```python
# .jaato/gc.json
{
    "plugin": "gc_budget",
    "config": {
        "preserve_recent_turns": 5,
        "target_percent": 60.0,
        "pressure_percent": 90.0,
        "notify_on_gc": true
    }
}
```

---

## Updating Existing Plugins

### gc_truncate

Minimal changes - add `removal_list` to result:

```python
def collect(self, history, context_usage, config, reason, budget=None):
    # ... existing logic ...

    removal_list = []
    for turn in turns:
        if turn.index not in preserved_indices:
            removal_list.append(GCRemovalItem(
                source=InstructionSource.CONVERSATION,
                child_key=f"turn_{turn.index}",
                tokens_freed=turn.estimated_tokens,
                reason="truncated"
            ))

    result = GCResult(
        # ... existing fields ...
        removal_list=removal_list
    )
    return new_history, result
```

### gc_summarize

Similar - track what was summarized:

```python
def collect(self, history, context_usage, config, reason, budget=None):
    # ... existing logic ...

    removal_list = []
    for turn in summarized_turns:
        removal_list.append(GCRemovalItem(
            source=InstructionSource.CONVERSATION,
            child_key=f"turn_{turn.index}",
            tokens_freed=turn.estimated_tokens,
            reason="summarized"
        ))

    # Add the summary as a new entry (handled by session)
    result.details["summary_tokens"] = estimate_tokens(summary_text)

    return new_history, result
```

### gc_hybrid

Combine both approaches:

```python
def collect(self, history, context_usage, config, reason, budget=None):
    # ... existing logic ...

    removal_list = []

    # Ancient turns (truncated)
    for turn in ancient_turns:
        removal_list.append(GCRemovalItem(
            source=InstructionSource.CONVERSATION,
            child_key=f"turn_{turn.index}",
            tokens_freed=turn.estimated_tokens,
            reason="ancient_truncated"
        ))

    # Middle turns (summarized)
    for turn in middle_turns:
        removal_list.append(GCRemovalItem(
            source=InstructionSource.CONVERSATION,
            child_key=f"turn_{turn.index}",
            tokens_freed=turn.estimated_tokens,
            reason="middle_summarized"
        ))

    return new_history, result
```

---

## Budget Sync After Summary

When gc_summarize or gc_hybrid create a summary, the session should add a new budget entry:

```python
# In JaatoSession._apply_gc_removal_list()

if "summary_tokens" in result.details:
    # Add summary as a new conversation child
    self._instruction_budget.add_child(
        source=InstructionSource.CONVERSATION,
        child_key="gc_summary",
        tokens=result.details["summary_tokens"],
        gc_policy=GCPolicy.PRESERVABLE,  # Summaries are worth keeping
        label="Context Summary",
        metadata={"created_by": result.plugin_name}
    )
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JAATO_GC_THRESHOLD` | 80.0 | Trigger GC when usage exceeds this % |
| `JAATO_GC_TARGET` | 60.0 | Target usage % after GC |
| `JAATO_GC_PRESSURE` | 90.0 | Start touching PRESERVABLE above this % |

---

## Migration Path

1. **Phase 1**: Add timestamp to SourceEntry, add removal_list to GCResult
2. **Phase 2**: Update existing plugins to populate removal_list
3. **Phase 3**: Update JaatoSession to apply removal_list to budget
4. **Phase 4**: Implement gc_budget plugin
5. **Phase 5**: Add target_percent and pressure_percent to GCConfig

---

## Testing Strategy

### Unit Tests

- `test_source_entry_timestamp`: Verify timestamps are set and serialized
- `test_gc_result_removal_list`: Verify removal list structure
- `test_gc_truncate_removal_list`: Verify truncate populates removal list
- `test_gc_budget_priority_order`: Verify ephemeral → partial → preservable order
- `test_gc_budget_respects_target`: Verify stops when target reached

### Integration Tests

- `test_session_applies_removal_list`: Verify session syncs budget after GC
- `test_budget_consistent_after_gc`: Verify budget tokens match actual history
- `test_gc_budget_with_mixed_policies`: Full flow with realistic budget

---

## Design Decisions (Resolved)

1. **Enrichment GC**: **Bulk clear**
   - Enrichments are regenerated each turn, so remove all at once when GC triggers

2. **Tool schema notification**: **No notification needed**
   - The model receives full accumulated context each turn; it won't see removed content anyway

3. **Multiple summaries**: **Keep separate** (`gc_summary_1`, `gc_summary_2`, etc.)
   - Preserves GC history and allows selective removal of older summaries

4. **History-to-budget mapping**: **Track message IDs in both places**
   - Add `message_id` field to Messages
   - Reference `message_id` in budget entry metadata
   - Enables precise mapping between history and budget entries

5. **Timestamp precision**: `time.time()` (float seconds)
   - Simple, JSON-serializable, sufficient precision

---

## Message ID Tracking

To enable precise history-to-budget mapping, we add message IDs:

### Message Enhancement

```python
# shared/plugins/model_provider/types.py

@dataclass
class Message:
    role: Role
    parts: List[Part]
    message_id: Optional[str] = None  # NEW: Unique identifier

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
```

### Budget Entry Reference

```python
# shared/instruction_budget.py

@dataclass
class SourceEntry:
    source: InstructionSource
    tokens: int
    gc_policy: GCPolicy
    label: Optional[str] = None
    children: Dict[str, "SourceEntry"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = None
    message_ids: List[str] = field(default_factory=list)  # NEW: Associated message IDs
```

### GC Removal by Message ID

```python
# In gc_budget plugin

def _apply_removals_to_history(self, history, removal_list, budget):
    """Remove messages from history based on removal list."""

    # Collect all message IDs to remove
    ids_to_remove = set()
    for item in removal_list:
        entry = budget.get_child(item.source, item.child_key)
        if entry and entry.message_ids:
            ids_to_remove.update(entry.message_ids)

    # Filter history
    return [msg for msg in history if msg.message_id not in ids_to_remove]
```

This ensures:
- Precise removal of exactly the messages tracked by budget entries
- No reliance on position matching or content heuristics
- Clean separation between GC decision (budget-based) and execution (history-based)
