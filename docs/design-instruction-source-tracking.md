# Design: Instruction Source Tracking

## Overview

Track token consumption by instruction source to help users understand where their context budget is being spent, and enable intelligent garbage collection based on source importance.

---

## Implementation Status

### Phase 1: Data Model - COMPLETE

| Component | File | Status |
|-----------|------|--------|
| `InstructionSource` enum | `shared/instruction_budget.py` | Done |
| `GCPolicy` enum | `shared/instruction_budget.py` | Done |
| `SourceEntry` dataclass | `shared/instruction_budget.py` | Done |
| `InstructionBudget` dataclass | `shared/instruction_budget.py` | Done |
| `ConversationTurnType` enum | `shared/instruction_budget.py` | Done |
| `PluginToolType` enum | `shared/instruction_budget.py` | Done |
| `estimate_tokens()` helper | `shared/instruction_budget.py` | Done |
| Unit tests (35 tests) | `shared/tests/test_instruction_budget.py` | Done |

### Phase 2: Session Integration - COMPLETE

| Component | File | Status |
|-----------|------|--------|
| `InstructionBudgetEvent` | `server/events.py` | Done |
| `_instruction_budget` attribute | `shared/jaato_session.py` | Done |
| `instruction_budget` property | `shared/jaato_session.py` | Done |
| `_populate_instruction_budget()` | `shared/jaato_session.py` | Done |
| `_update_conversation_budget()` | `shared/jaato_session.py` | Done |
| `_emit_instruction_budget_update()` | `shared/jaato_session.py` | Done |
| `set_instruction_budget_callback()` | `shared/jaato_session.py` | Done |
| Wire up in `configure()` | `shared/jaato_session.py` | Done |
| Wire up on turn completion | `shared/jaato_session.py` | Done |

### Phase 3: Server Integration - COMPLETE

| Component | File | Status |
|-----------|------|--------|
| Import `InstructionBudgetEvent` | `server/core.py` | Done |
| Wire up callback in server | `server/core.py` | Done |
| Add `on_agent_instruction_budget_updated` hook | `shared/plugins/subagent/ui_hooks.py` | Done |
| Implement hook in `ServerAgentHooks` | `server/core.py` | Done |
| Emit budgets in `emit_current_state()` | `server/core.py` | Done |
| Emit budgets in `_emit_subagent_state()` | `server/core.py` | Done |
| Add `GetInstructionBudgetRequest` | `server/events.py` | Done |
| Handle request in `SessionManager` | `server/session_manager.py` | Done |

### Phase 4: Rich Client UI - COMPLETE

| Component | File | Status |
|-----------|------|--------|
| Token usage panel widget | `rich-client/budget_panel.py` | Done |
| Drill-down navigation | `rich-client/budget_panel.py:158-179` | Done |
| Multi-agent view tabs | `rich-client/budget_panel.py:142-156, 423-446` | Done |
| Keybinding for panel toggle | `rich-client/keybindings.py:149, 322` | Done |
| Event handling | `rich-client/rich_client.py:3554-3557` | Done |
| PTDisplay integration | `rich-client/pt_display.py:462-463, 1426-1451, 1680-1696` | Done |

### Decisions Made

| Question | Decision |
|----------|----------|
| Token counting | Using `estimate_tokens()` (chars/4 approximation) initially |
| Update frequency | Update after `configure()` and after each turn completes |
| Multi-agent aggregation | Show raw token counts only; omit limit/percentage since they're meaningless across different context windows |

### Open Questions (Remaining)

1. **Enrichment tracking**: How to accurately track enrichment pipeline token contributions?

---

## Visual Design

### Panel Layout

A dedicated panel in the rich client (toggled, replaces output panel) showing token usage per source with drill-down capability.

**Per-agent view** (shows limit and percentage):
```
╭─ Token Usage (8030 / 128K = 6.3%) ────────────────╮
│                                                   │
│  Source         Tokens  GC   ▏ Usage              │
│  ──────────────────────────────────────────────── │
│  System           1090  🔒 ▸ ▏████░░░░░░░░░░░░░░  │
│  Plugin           1840  ◐  ▸ ▏████████░░░░░░░░░░  │
│  Enrichment        300  ○    ▏█░░░░░░░░░░░░░░░░░  │
│  Conversation     5000  ◐  ▸ ▏██████████████████  │
│                                                   │
│  🔒 = locked  ◐ = partial  ○ = ephemeral  ▸ = drill │
╰───────────────────────────────────────────────────╯
 [Total] [Main] [explore-1] [subagent-2]       TAB →
```

**Total view** (aggregated across agents - no limit/percentage since agents may have different context windows):
```
╭─ Token Usage (Total: 12050 tokens) ───────────────╮
│                                                   │
│  Source         Tokens  GC   ▏ Distribution       │
│  ──────────────────────────────────────────────── │
│  System           1600  🔒 ▸ ▏███░░░░░░░░░░░░░░░  │
│  Plugin           2850  ◐  ▸ ▏██████░░░░░░░░░░░░  │
│  Enrichment        500  ○    ▏█░░░░░░░░░░░░░░░░░  │
│  Conversation     7500  ◐  ▸ ▏██████████████████  │
│                                                   │
│  🔒 = locked  ◐ = partial  ○ = ephemeral  ▸ = drill │
╰───────────────────────────────────────────────────╯
 [Total] [Main] [explore-1] [subagent-2]       TAB →
```

### Interaction

- **TAB / SHIFT-TAB**: Cycle through views (Total → Main agent → Subagent 1 → ...)
- **ENTER on row**: Drill down into children (e.g., Plugin → per-tool breakdown)
- **ESC**: Return to parent view / close panel

### Drill-Down Views

**System drill-down (base, client, framework):**
```
╭─ System Breakdown (1090 tokens) ──────────────────╮
│                                                   │
│  Component          Tokens  GC   ▏ Usage          │
│  ──────────────────────────────────────────────── │
│  Base Instructions     500  🔒   ▏█████████░░░░░  │
│  Client Instructions   200  🔒   ▏███░░░░░░░░░░░  │
│  Framework             390  🔒   ▏███████░░░░░░░  │
│                                                   │
│  All components are locked (essential)            │
╰───────────────────────────────────────────────────╯
 [← Back]                                      ESC
```

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
    """The 4 tracked instruction source layers"""
    SYSTEM = "system"           # System instructions (children: base, client, framework)
    PLUGIN = "plugin"           # Plugin instructions (children: per-tool)
    ENRICHMENT = "enrichment"   # Prompt enrichment pipeline additions
    CONVERSATION = "conversation"  # Message history (children: per-turn)


class SystemChildType(Enum):
    """Types of SYSTEM instruction children"""
    BASE = "base"           # User-provided .jaato/instructions/*.md (or legacy single file)
    CLIENT = "client"       # Programmatic system_instructions param
    FRAMEWORK = "framework" # Task completion, parallel tool guidance


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

### JaatoSession (Implemented)

```python
class JaatoSession:
    _instruction_budget: Optional[InstructionBudget] = None
    _on_instruction_budget_updated: Optional[Callable[[Dict], None]] = None

    @property
    def instruction_budget(self) -> Optional[InstructionBudget]:
        return self._instruction_budget

    def set_instruction_budget_callback(self, callback: Callable[[Dict], None]) -> None:
        """Set callback for budget updates (used by server to emit events)."""
        self._on_instruction_budget_updated = callback

    def configure(self, system_instructions: Optional[str] = None, ...):
        # ... existing configuration ...
        self._populate_instruction_budget(session_instructions=system_instructions)

    def _populate_instruction_budget(self, session_instructions: Optional[str] = None):
        """Populate budget with token counts from all source layers."""
        # Creates InstructionBudget with entries for:
        # - SYSTEM: Framework constants (task completion, parallel tools)
        # - SESSION: User-provided system_instructions
        # - PLUGIN: Per-plugin instructions (with children per-tool)
        # - ENRICHMENT: Pipeline additions (placeholder)
        # - CONVERSATION: Message history (initially 0)
        self._emit_instruction_budget_update()

    def _update_conversation_budget(self):
        """Update CONVERSATION entry from current history (called after each turn)."""
        # Iterates history, counts tokens per message, adds children per-turn
        self._emit_instruction_budget_update()

    def _emit_instruction_budget_update(self):
        """Emit budget update via callback if registered."""
        if self._on_instruction_budget_updated and self._instruction_budget:
            self._on_instruction_budget_updated(self._instruction_budget.snapshot())
```

### Server Events (Implemented)

```python
class EventType(str, Enum):
    INSTRUCTION_BUDGET_UPDATED = "instruction_budget.updated"

@dataclass
class InstructionBudgetEvent(Event):
    """Emitted when instruction budget changes."""
    type: EventType = field(default=EventType.INSTRUCTION_BUDGET_UPDATED)
    agent_id: str = ""
    budget_snapshot: Dict[str, Any] = field(default_factory=dict)
```

### Server Core (Implemented)

```python
# In server/core.py - callback for main session:
def instruction_budget_callback(snapshot: dict):
    server.emit(InstructionBudgetEvent(
        agent_id=snapshot.get('agent_id', 'main'),
        budget_snapshot=snapshot,
    ))
session.set_instruction_budget_callback(instruction_budget_callback)

# In ServerAgentHooks - hook for all agents (main + subagents):
def on_agent_instruction_budget_updated(self, agent_id, budget_snapshot):
    server.emit(InstructionBudgetEvent(
        agent_id=agent_id,
        budget_snapshot=budget_snapshot,
    ))

# In emit_current_state() - emit on client reconnect:
if session and session.instruction_budget:
    emit(InstructionBudgetEvent(
        agent_id=session.agent_id,
        budget_snapshot=session.instruction_budget.snapshot(),
    ))
```

### AgentUIHooks Protocol (Implemented)

```python
# In shared/plugins/subagent/ui_hooks.py:
def on_agent_instruction_budget_updated(
    self,
    agent_id: str,
    budget_snapshot: Dict[str, Any]
) -> None:
    """Called when agent's instruction budget is updated."""
    ...
```

### On-Demand Request (Implemented)

Clients can request the current instruction budget at any time:

```python
# Client sends:
GetInstructionBudgetRequest(agent_id="main")  # or specific subagent ID

# Server responds with:
InstructionBudgetEvent(agent_id="main", budget_snapshot={...})
```

The request is handled by `SessionManager._dispatch_to_session()`, which:
1. For `agent_id="main"` (or None): Gets budget from `server._jaato.get_session().instruction_budget`
2. For subagents: Gets budget from `SubagentPlugin._active_sessions[agent_id].session.instruction_budget`

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

1. ~~**Token counting**: Use model's tokenizer or approximate (chars/4)?~~
   **Decision**: Using `estimate_tokens()` with chars/4 approximation. Can be upgraded to model tokenizer later.

2. ~~**Update frequency**: Update budget on every turn or only when panel is shown?~~
   **Decision**: Update after `configure()` and after each turn completes. Callback mechanism allows lazy UI updates.

3. ~~**Multi-agent aggregation**: How to aggregate "Total" view across agents with different context limits?~~
   **Decision**: Show raw token counts only; omit limit/percentage since they're meaningless across different context windows.

---

## Provider Token Counting

### History Handling by Provider

Different providers handle conversation history differently, which affects how token counting works:

| Provider | Mode | History Handling | Pre-request `count_tokens()` |
|----------|------|------------------|------------------------------|
| **Google GenAI** | Stateless | Full history sent each request via `chats.create()` SDK | ✅ Actual (API call) |
| **Anthropic** | Stateless | Full history sent each request via `messages.create()` | ✅ Actual (API call) |
| **GitHub Models** | Stateless | Full history sent each request via `complete()` | ❌ Estimate (`len // 4`) |
| **Antigravity** | Stateless | Full history sent each request | ❌ Estimate (`len // 4`) |
| **Ollama** | Stateless | Inherits from Anthropic provider | ❌ Estimate (`len // 4`) |
| **Claude CLI** | **Stateful** | Uses `--resume {session_id}` - only new content sent | ❌ Estimate (`len // 4`) |

### Stateless Providers

For stateless providers (all except Claude CLI), each API request includes the **full context**:
- System instructions (sent with every request)
- Tool definitions (sent with every request)
- Full conversation history (all previous messages)
- Current new message

The `response.usage.prompt_tokens` reflects the total input context for that request.

### Stateful Provider: Claude CLI

Claude CLI maintains **server-side session state**:
- First call: Full context sent → CLI returns `session_id` in SystemMessage
- Subsequent calls: `--resume {session_id}` → only new message sent

For Claude CLI, `prompt_tokens` in subsequent responses reflects **only the incremental content**, not the full history.

### Implications for Token Budget Tracking

1. **Pre-request breakdown**: For providers with `count_tokens()` APIs (Google, Anthropic), we can get accurate per-piece counts by calling `provider.count_tokens()` on each component separately.

2. **Post-response total**: All providers return `response.usage.prompt_tokens` after each request, but this is only a single total—no breakdown by source.

3. **Claude CLI special case**: Token tracking needs different logic since subsequent turns only report incremental tokens.

### Current Implementation

Currently using `estimate_tokens()` (chars/4 approximation) for all providers. This could be improved by:
- Using `provider.count_tokens()` where APIs are available (Google, Anthropic)
- Falling back to estimates for providers without token counting APIs

---

## Future Considerations

- **Cost tracking**: Extend to show estimated cost per source
- **Historical trends**: Track budget over conversation lifetime
- **Recommendations**: Suggest which plugins to disable if budget is tight
- **TokenLedger deprecation**: Migrate remaining ledger functionality to OTel spans
