# Design: Instruction Source Tracking

## Overview

Track token consumption by instruction source to help users understand where their context budget is being spent, and enable intelligent garbage collection based on source importance.

---

## Visual Design

### Panel Layout

A dedicated panel in the rich client (toggled, replaces output panel) showing token usage per source with drill-down capability.

```
╭─ Token Usage (8230 / 128K = 6.4%) ────────────────╮
│                                                   │
│  Source         Tokens  GC   ▏ Usage              │
│  ──────────────────────────────────────────────── │
│  System            890  🔒   ▏████░░░░░░░░░░░░░░  │
│  Session           200  🔒   ▏█░░░░░░░░░░░░░░░░░  │
│  Plugin           1840  ◐    ▏████████░░░░░░░░░░  │
│  Enrichment        300  ○    ▏█░░░░░░░░░░░░░░░░░  │
│  Conversation     5000  ◐    ▏██████████████████  │
│                                                   │
│  🔒 = locked  ◐ = partial  ○ = ephemeral          │
╰───────────────────────────────────────────────────╯
 [Total] [Main] [explore-1] [subagent-2]       TAB →
```

### Interaction

- **TAB / SHIFT-TAB**: Cycle through views (Total → Main agent → Subagent 1 → ...)
- **ENTER on row**: Drill down into children (e.g., Plugin → per-tool breakdown)
- **ESC**: Return to parent view / close panel

### Drill-Down Views

**Plugin drill-down (per-tool):**
```
╭─ Plugin Breakdown (1840 tokens) ──────────────────╮
│                                                   │
│  Tool              Tokens  GC   ▏ Usage           │
│  ──────────────────────────────────────────────── │
│  run_shell_command    800  🔒   ▏████████████░░░  │
│  edit_file            700  🔒   ▏██████████░░░░░  │
│  glob_files           150  ○    ▏██░░░░░░░░░░░░░  │
│  grep_content         120  ○    ▏█░░░░░░░░░░░░░░  │
│  web_search            70  ○    ▏█░░░░░░░░░░░░░░  │
│                                                   │
│  🔒 = core (always loaded)  ○ = discoverable      │
╰───────────────────────────────────────────────────╯
 [← Back]                                      ESC
```

**Conversation drill-down (per-turn):**
```
╭─ Conversation Breakdown (5000 tokens) ────────────╮
│                                                   │
│  Turn               Tokens  GC   ▏ Usage          │
│  ──────────────────────────────────────────────── │
│  turn_0 (user)          50  🔒   ▏░░░░░░░░░░░░░░  │
│  turn_1 (clarify Q)    200  ◑    ▏█░░░░░░░░░░░░░  │
│  turn_2 (clarify A)     30  ◑    ▏░░░░░░░░░░░░░░  │
│  turn_3 (model)       2200  ○    ▏████████░░░░░░  │
│  turn_3 (summary)      150  ◑    ▏░░░░░░░░░░░░░░  │
│  turn_4 (user)          80  ○    ▏░░░░░░░░░░░░░░  │
│  turn_5 (model)       2140  ○    ▏████████░░░░░░  │
│  turn_5 (summary)      150  ◑    ▏░░░░░░░░░░░░░░  │
│                                                   │
│  🔒 = original  ◑ = preservable  ○ = ephemeral    │
╰───────────────────────────────────────────────────╯
 [← Back]                                      ESC
```

---

## Data Model

### Enums

```python
from enum import Enum

class InstructionSource(Enum):
    """The 5 tracked instruction source layers"""
    SYSTEM = "system"           # Base + framework constants (task completion, parallel, sandbox, permission)
    SESSION = "session"         # Programmatic system_instructions param
    PLUGIN = "plugin"           # Plugin instructions (children: per-tool)
    ENRICHMENT = "enrichment"   # Prompt enrichment pipeline additions
    CONVERSATION = "conversation"  # Message history (children: per-turn)


class GCPolicy(Enum):
    """Garbage collection policy for instruction sources"""
    LOCKED = "locked"           # 🔒 Never GC - essential for operation
    PRESERVABLE = "preservable" # ◑  Prefer to keep, GC only under extreme pressure
    PARTIAL = "partial"         # ◐  Some parts GC-able (container with mixed children)
    EPHEMERAL = "ephemeral"     # ○  Can be fully GC'd
```

### Core Classes

```python
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class SourceEntry:
    """A single instruction source with its token count and GC policy"""
    source: InstructionSource
    tokens: int
    gc_policy: GCPolicy
    label: Optional[str] = None  # Display label (e.g., tool name, turn description)
    children: Dict[str, "SourceEntry"] = field(default_factory=dict)


@dataclass
class InstructionBudget:
    """Tracks token usage by instruction source for an agent.

    Identity Model:
    - session_id: Server-managed session (umbrella that groups all agents).
                  This is what clients connect/reconnect to.
    - agent_id: This agent's identity within the session ("main", "explore-1", etc.)
    - agent_type: Type for display purposes ("main", "explore", "plan", etc.)

    One InstructionBudget per agent (JaatoSession). When client reconnects to a
    session_id, server provides budgets for ALL agents in that session.
    """
    session_id: str = ""           # Server session (umbrella)
    agent_id: str = "main"         # This agent within the session
    agent_type: Optional[str] = None  # For display
    entries: Dict[InstructionSource, SourceEntry] = field(default_factory=dict)
    context_limit: int = 128_000   # Model's context window
```

### Session/Agent Relationship

```
┌─────────────────────────────────────────────────────┐
│  Server Session (session_id="abc123")               │
│  - What client connects/reconnects to               │
│  - Managed by SessionManager                        │
│  - Persists across client disconnects               │
│                                                     │
│  ┌───────────────┐  ┌───────────────┐              │
│  │ Main Agent    │  │ Subagent      │  ...         │
│  │ agent_id=main │  │ agent_id=     │              │
│  │               │  │ explore-1     │              │
│  │ JaatoSession  │  │ JaatoSession  │              │
│  │ + Budget      │  │ + Budget      │              │
│  └───────────────┘  └───────────────┘              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

On client reconnect:
- Server gathers all `InstructionBudget` where `session_id` matches
- Client receives budgets for main + all subagents (active or completed)
- UI shows: `[Total] [Main] [explore-1] [subagent-2] ...`

---

## Default GC Policies

| Source | Default Policy | Rationale |
|--------|----------------|-----------|
| SYSTEM | LOCKED | Framework essentials, always needed |
| SESSION | LOCKED | User-defined behavior, intentional |
| PLUGIN | PARTIAL | Core tools locked, discoverable tools ephemeral |
| ENRICHMENT | EPHEMERAL | Can be re-enriched on next turn |
| CONVERSATION | PARTIAL | Original request + clarifications preserved, working turns ephemeral |

### Plugin Children Policies

| Tool Type | Policy | Rationale |
|-----------|--------|-----------|
| Core tools (cli, file_edit, etc.) | LOCKED | Always needed for agentic work |
| Discoverable tools | EPHEMERAL | Can be re-discovered via introspection |

### Conversation Children Policies

| Turn Type | Policy | Rationale |
|-----------|--------|-----------|
| Original user request (turn 0) | LOCKED | The task definition |
| Clarification questions (model) | PRESERVABLE | Important context |
| Clarification answers (user) | PRESERVABLE | Important context |
| Turn summaries/conclusions | PRESERVABLE | High-value compressed context |
| Working turns (verbose output) | EPHEMERAL | Can be discarded if summary exists |

**GC Strategy for Conversation:**
When GC needs to reclaim tokens from CONVERSATION, it should:
1. First, discard EPHEMERAL working turns that have an associated summary
2. Then, summarize remaining working turns (creating PRESERVABLE summaries)
3. Only under extreme pressure, consider PRESERVABLE content

---

## Integration Points

### JaatoSession

```python
class JaatoSession:
    instruction_budget: InstructionBudget

    def configure(self, ...):
        # After assembling system instructions, populate budget
        self._populate_instruction_budget()

    def _populate_instruction_budget(self):
        # Count tokens per source and create entries
        ...
```

### Server Events

New event for UI updates:

```python
@dataclass
class InstructionBudgetEvent:
    """Emitted when instruction budget changes"""
    session_id: str
    agent_id: str
    budget_snapshot: Dict  # From InstructionBudget.snapshot()
```

### GC Plugin Integration

GC plugins can query the budget to make intelligent decisions:

```python
class GCPlugin:
    def collect(self, session: JaatoSession, target_tokens: int) -> int:
        budget = session.instruction_budget

        # Prioritize ephemeral sources first
        for source in [InstructionSource.ENRICHMENT, InstructionSource.CONVERSATION]:
            entry = budget.entries.get(source)
            if entry and entry.gc_eligible_tokens() > 0:
                # GC this source's ephemeral children
                ...
```

---

## Open Questions

1. **Token counting**: Use model's tokenizer or approximate (chars/4)?
2. **Update frequency**: Update budget on every turn or only when panel is shown?
3. **Multi-agent aggregation**: How to aggregate "Total" view across agents with different context limits?

---

## Future Considerations

- **Cost tracking**: Extend to show estimated cost per source
- **Historical trends**: Track budget over conversation lifetime
- **Recommendations**: Suggest which plugins to disable if budget is tight
- **TokenLedger deprecation**: Migrate remaining ledger functionality to OTel spans
