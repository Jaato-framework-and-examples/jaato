# JAATO Instruction Sources & Priority Hierarchy

## Executive Summary

JAATO (Just Another Agentic Tool Orchestrator) assembles model instructions from multiple sources in a carefully layered architecture. Instructions are combined at runtime in a specific priority order, while runtime messages (prompts) follow a separate priority queue for agent communication.

---

## Part 1: System Instruction Assembly Order

When a session is configured, system instructions are assembled in the following order (from first to last in the final prompt):

### 📊 Token Budget Overview

| Configuration | Estimated Tokens | Use Case |
|--------------|------------------|----------|
| **Minimal** | ~500-800 | Single simple plugin (e.g., web_search only) |
| **Typical** | ~2,000-2,500 | Standard setup with 4-5 common plugins |
| **Full** | ~3,500-4,500 | All plugins + sandbox + extensive base instructions |

*Note: These estimates use ~4 characters per token. Actual counts vary by model tokenizer.*

*⚠️ With `JAATO_DEFERRED_TOOLS=true` (the default), initial context is smaller — only "core" tools are loaded upfront. The model discovers other tools via introspection as needed, so actual per-request token usage may be lower than these estimates.*

### Layer 1: Base System Instructions
**Source:** `.jaato/system_instructions.md`
**Priority:** HIGHEST (appears first)
**Estimated Tokens:** 0-500+ (user-defined, highly variable)
**Lookup Order:**
1. `{CWD}/.jaato/system_instructions.md`
2. `~/.jaato/system_instructions.md`

**Purpose:** Defines base behavioral rules that apply to ALL agents (main and subagents), such as transparency requirements and operational constraints.

---

### Layer 2: Additional (Session-Specific) Instructions
**Source:** `system_instructions` parameter passed to `session.configure()`
**Priority:** HIGH
**Estimated Tokens:** 0-1,000+ (programmatic, varies by use case)

**Purpose:** Session-specific instructions provided programmatically. Can customize behavior for specific use cases without modifying base instructions.

---

### Layer 3: Plugin-Specific Instructions
**Source:** Each plugin's `get_system_instructions()` method
**Priority:** MEDIUM
**Estimated Tokens:** 200-3,000+ (depends on plugins enabled)

**Plugin Token Breakdown (approximate):**

| Plugin | Tokens | Description |
|--------|--------|-------------|
| `cli` | ~800 | Shell access, extensive examples, backgrounding |
| `file_edit` | ~700 | CRUD operations, multiFileEdit, backups |
| `subagent` | ~875 | Complex orchestration, event handling |
| `filesystem_query` | ~300 | glob_files, grep_content |
| `web_search` | ~225 | Simple search capability |
| `web_fetch` | ~200 | URL fetching |
| `mcp` | ~400 | Model Context Protocol servers |
| `memory` | ~250 | Session memory management |
| `references` | ~150 | @file reference handling |
| `permission` | ~100 | Permission system rules |

**Aggregation:** Instructions from all exposed plugins are concatenated with `\n\n` separator.

**Includes:**
- Tool-specific usage instructions
- Capability descriptions
- Usage constraints and guidelines

**Special Processing:**
- Registry runs enrichment pipeline on combined plugin instructions
- Enrichment plugins can modify/augment these instructions

---

### Layer 4: Permission Plugin Instructions
**Source:** Permission plugin's `get_system_instructions()`
**Priority:** MEDIUM-LOW
**Estimated Tokens:** ~100-200

**Purpose:** Informs the model about permission requirements and constraints for tool usage.

---

### Layer 5: Framework-Level Task Completion Instruction
**Source:** `_TASK_COMPLETION_INSTRUCTION` constant
**Priority:** LOW (always included)
**Estimated Tokens:** ~30 (fixed)

**Content:**
```
"After each action, continue working until the request is truly fulfilled. 
Pause only for permissions or clarifications—never from uncertainty."
```

**Purpose:** Encourages agentic behavior - continuing work autonomously without unnecessary pauses.

---

### Layer 6: Parallel Tool Guidance (Conditional)
**Source:** `_PARALLEL_TOOL_GUIDANCE` constant
**Priority:** LOWEST
**Condition:** Only included when `JAATO_PARALLEL_TOOLS=true` (default)
**Estimated Tokens:** ~60 (fixed, when enabled)

**Content:**
```
"When you need to perform multiple independent operations (e.g., reading 
several files, searching multiple patterns, fetching multiple URLs), issue 
all tool calls in a single response rather than one at a time. Independent 
operations will execute in parallel, significantly reducing latency."
```

---

### Layer 7: Sandbox Guidance (Conditional)
**Source:** `_get_sandbox_guidance()` function
**Condition:** Only included when `JAATO_WORKSPACE_ROOT` or `workspaceRoot` env var is set
**Estimated Tokens:** ~70 (fixed, when enabled)

**Purpose:** Informs model about file operation restrictions in sandboxed environments.

---

## Part 2: Prompt Enrichment Pipeline

User prompts are processed through an enrichment pipeline before being sent to the model. Plugins subscribe to this pipeline and are called **in priority order** (lower numbers first).

### Enrichment Priority Table

| Priority | Plugin | Purpose |
|----------|--------|---------|
| 20 | `references` | Injects MODULE.md and other @reference content |
| 40 | `template` | Extracts embedded templates from injected content |
| 50 | (default) | Any plugin without explicit priority |
| 60 | `multimodal` | Handles @image references |
| 80 | `memory` | Adds memory hints based on prompt content |
| 90 | `session` | Adds session description hints |

**Pipeline Flow:**
```
User Prompt → references (20) → template (40) → ... → session (90) → Final Prompt
```

Each plugin receives the output of the previous plugin, enabling chained transformations.

---

## Part 3: Runtime Message Priority (Agent Communication)

During runtime, messages between agents are queued with priority-based processing:

### Source Types & Priority

| Source Type | Priority Level | Processing Mode |
|-------------|----------------|-----------------|
| `PARENT` | HIGH | Mid-turn (interrupt) |
| `USER` | HIGH | Mid-turn (interrupt) |
| `SYSTEM` | HIGH | Mid-turn (interrupt) |
| `CHILD` | LOW | Process when idle |

### Processing Rules

**High-Priority Messages (PARENT/USER/SYSTEM):**
- Processed immediately, even mid-turn
- Can interrupt ongoing model generation
- Used for controller instructions, user input, system alerts

**Low-Priority Messages (CHILD):**
- Queued and processed when agent becomes idle
- Used for status updates from subagents
- FIFO ordering within priority group

---

## Part 4: Visual Hierarchy Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEM INSTRUCTION ASSEMBLY                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. BASE INSTRUCTIONS                                        │    │
│  │     .jaato/system_instructions.md                            │    │
│  │     (CWD first, then ~/.jaato)                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  2. ADDITIONAL (SESSION) INSTRUCTIONS                        │    │
│  │     session.configure(system_instructions="...")             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  3. PLUGIN INSTRUCTIONS                                      │    │
│  │     plugin.get_system_instructions() for each exposed plugin │    │
│  │     + Enrichment Pipeline Processing                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  4. PERMISSION INSTRUCTIONS                                  │    │
│  │     permission_plugin.get_system_instructions()              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  5. FRAMEWORK INSTRUCTIONS                                   │    │
│  │     • Task Completion (always)                               │    │
│  │     • Parallel Tool Guidance (if enabled)                    │    │
│  │     • Sandbox Guidance (if configured)                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    PROMPT ENRICHMENT PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Prompt                                                         │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌─────────┐         │
│  │ refs    │ → │ template │ → │ multimodal │ → │ memory  │ → ...   │
│  │ (20)    │   │ (40)     │   │ (60)       │   │ (80)    │         │
│  └─────────┘   └──────────┘   └────────────┘   └─────────┘         │
│       │                                                              │
│       ▼                                                              │
│  Enriched Prompt                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    RUNTIME MESSAGE PRIORITY                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  HIGH PRIORITY (Interrupt Mid-Turn)                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PARENT │ USER │ SYSTEM                                      │    │
│  │  Controller instructions, user input, system alerts          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  LOW PRIORITY (Process When Idle)                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  CHILD                                                       │    │
│  │  Subagent status updates, deferred messages                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Key Configuration Points

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `JAATO_PARALLEL_TOOLS` | Enable parallel tool guidance | `true` |
| `JAATO_DEFERRED_TOOLS` | Load only core tools initially (others via introspection) | `true` |
| `JAATO_WORKSPACE_ROOT` | Sandbox workspace path | (none) |

*Note: Both `JAATO_PARALLEL_TOOLS` and `JAATO_DEFERRED_TOOLS` default to `true` for optimal token economy.*

### Plugin Configuration

Plugins can define:
- `get_system_instructions()` → Static instructions about capabilities
- `get_enrichment_priority()` → Priority in enrichment pipeline (default: 50)
- `subscribes_to_prompt_enrichment()` → Opt into prompt processing
- `subscribes_to_system_instruction_enrichment()` → Opt into instruction processing

---

## Part 6: Summary Table

| Source | Type | When Applied | Order | Est. Tokens |
|--------|------|--------------|-------|-------------|
| `.jaato/system_instructions.md` | Static | Session creation | 1st | 0-500+ |
| Session `system_instructions` param | Dynamic | Session creation | 2nd | 0-1,000+ |
| Plugin instructions | Dynamic | Session creation | 3rd | 200-3,000+ |
| Permission instructions | Dynamic | Session creation | 4th | ~100-200 |
| Task completion instruction | Static | Session creation | 5th | ~30 |
| Parallel tool guidance | Conditional | Session creation | 6th | ~60 |
| Sandbox guidance | Conditional | Session creation | 7th | ~70 |
| Prompt enrichment | Dynamic | Per-turn | Runtime | varies |
| Message queue priority | Dynamic | Per-message | Runtime | N/A |

### Total System Instruction Token Ranges

```
┌──────────────────────────────────────────────────────────────────┐
│                     TOKEN BUDGET SCENARIOS                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  MINIMAL (~500-800 tokens)                                        │
│  ├─ Base instructions:     ~100 tokens                            │
│  ├─ Simple plugin (1):     ~225 tokens (e.g., web_search)         │
│  ├─ Task completion:       ~30 tokens                             │
│  └─ Parallel guidance:     ~60 tokens                             │
│                                                                   │
│  TYPICAL (~2,000-2,500 tokens)                                    │
│  ├─ Base instructions:     ~200 tokens                            │
│  ├─ CLI plugin:            ~800 tokens                            │
│  ├─ File Edit plugin:      ~700 tokens                            │
│  ├─ Filesystem Query:      ~300 tokens                            │
│  ├─ Framework constants:   ~90 tokens                             │
│  └─ Other plugins:         ~200-400 tokens                        │
│                                                                   │
│  FULL (~3,500-4,500+ tokens)                                      │
│  ├─ Base instructions:     ~300-500 tokens                        │
│  ├─ All plugins:           ~3,000+ tokens                         │
│  ├─ Subagent:              ~875 tokens (adds complexity)          │
│  ├─ Framework constants:   ~160 tokens (incl. sandbox)            │
│  └─ Session-specific:      ~200-500 tokens                        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Cost Implications:**
- At $0.25/1M input tokens (typical Gemini pricing), a "typical" configuration costs ~$0.0005 per request just for system instructions
- `JAATO_DEFERRED_TOOLS=true` is the **default** — only "core" tools are loaded initially, with others discoverable via introspection. This already optimizes token usage.
- Setting `JAATO_DEFERRED_TOOLS=false` loads all tools upfront, increasing initial context size but avoiding introspection overhead

---

## Budget Panel Categories

The budget panel (Ctrl+B) groups instruction sources differently for a cleaner display:

| Budget Category | Contains | Drill-Down |
|----------------|----------|------------|
| **System** | Base instructions + Client (session) instructions + Framework constants | Yes (3 children) |
| **Plugin** | Plugin-specific instructions | Yes (per-tool) |
| **Enrichment** | Enrichment pipeline additions | No |
| **Conversation** | Message history | Yes (per-turn) |

This grouping consolidates all "system-level" instructions (base, session-specific, and framework) under a single "System" category, making it easier to understand context budget allocation at a glance.

---

## Color Coding Suggestion for Infographic

- **🔵 Blue:** Static/Base instructions (always present)
- **🟢 Green:** Dynamic/Plugin instructions (configurable)
- **🟡 Yellow:** Conditional instructions (environment-dependent)
- **🔴 Red:** High-priority runtime messages
- **🟠 Orange:** Low-priority runtime messages
- **⬜ Gray:** Processing pipelines/flows
