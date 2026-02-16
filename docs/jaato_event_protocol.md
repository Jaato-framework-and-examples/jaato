# JAATO Server-Client Event Protocol

## Executive Summary

JAATO uses a **server-first architecture** where the server runs as a daemon and clients connect via IPC (Unix domain sockets) or WebSocket. All communication between server and client flows through a typed **event protocol** — a set of 40+ JSON-serializable dataclasses that represent every meaningful state change in the system. Events flow bidirectionally: the server emits lifecycle, output, and status events to clients, while clients send requests (messages, permission responses, commands) back to the server. This design decouples the agentic runtime from UI concerns, enabling multiple clients to observe the same session simultaneously and allowing different UI implementations (TUI, web, headless) to consume the same event stream.

---

## Part 1: Why an Event Protocol?

### The Problem

Without an event protocol, the UI must be tightly coupled to the runtime:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│    Without Events:                 With Events:                      │
│                                                                      │
│    ┌─────────┐                     ┌─────────────────────────────┐   │
│    │ Runtime │◄──► UI              │       JAATO SERVER          │   │
│    │         │  (coupled)          │       (daemon)              │   │
│    └─────────┘                     │                             │   │
│                                    │   Runtime → Event Emitter   │   │
│    - Single client only            └──────────┬──────────────────┘   │
│    - UI blocks runtime                        │                      │
│    - No remote clients             ┌──────────┴──────────┐           │
│    - No reconnection               │    Event Stream     │           │
│                                    │   (JSON over IPC    │           │
│                                    │    or WebSocket)    │           │
│                                    └──────┬──────┬───────┘           │
│                                           │      │                   │
│                                    ┌──────┴┐  ┌──┴──────┐           │
│                                    │ TUI   │  │ Web UI  │           │
│                                    │Client │  │ Client  │           │
│                                    └───────┘  └─────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | How the Protocol Achieves It |
|-----------|------------------------------|
| **Decoupling** | Server emits semantic events; clients decide how to render them |
| **Multi-client** | Session manager broadcasts events to all attached clients |
| **Reconnection** | `emit_current_state()` replays full state on reconnect |
| **Forward compatibility** | Unknown fields are filtered during deserialization |
| **Thread safety** | Events queued via `call_soon_threadsafe()` from model threads |

---

## Part 2: Event Architecture

### Base Event Structure

Every event is a Python dataclass inheriting from `Event`:

```python
@dataclass
class Event:
    type: EventType          # Enum identifying the event kind
    timestamp: str           # ISO 8601 UTC timestamp (auto-generated)
```

Events serialize to JSON for transmission:

```json
{
  "type": "agent.output",
  "timestamp": "2025-01-15T10:30:00.123456",
  "agent_id": "main",
  "source": "model",
  "text": "Let me read that file.",
  "mode": "write"
}
```

### Event Direction

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVENT DIRECTION                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SERVER → CLIENT (Notifications)                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Agent lifecycle, output streaming, tool execution,          │    │
│  │  permission/clarification prompts, context updates,          │    │
│  │  plan updates, session info, system messages, errors         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  CLIENT → SERVER (Requests)                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Send message, permission response, clarification response,  │    │
│  │  stop, command execution, history request, tool management,  │    │
│  │  workspace management, client configuration                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  BIDIRECTIONAL FLOWS (Request-Response Patterns)                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Permission: Requested → InputMode ← Response → Resolved    │    │
│  │  Clarification: Requested → Question → InputMode            │    │
│  │                 ← Response → Resolved                        │    │
│  │  Reference Selection: Requested ← Response → Resolved       │    │
│  │  Workspace Mismatch: Requested ← Response → Resolved        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Event Categories

The 40+ event types are organized into functional categories:

### Category Overview

| Category | Event Count | Direction | Purpose |
|----------|-------------|-----------|---------|
| **Connection** | 2 | S→C | Client connect/disconnect lifecycle |
| **Agent Lifecycle** | 4 | S→C | Agent creation, output, status, completion |
| **Tool Execution** | 3 | S→C | Tool start, end, live output |
| **Permission Flow** | 4 | S↔C | Permission request, input, response, resolution |
| **Clarification Flow** | 5 | S↔C | Multi-question clarification sessions |
| **Reference Selection** | 3 | S↔C | User selects which references to include |
| **Workspace Mismatch** | 3 | S↔C | Resolve workspace path conflicts |
| **Plan Management** | 2 | S→C | Plan creation, updates, completion |
| **Context & Tokens** | 4 | S→C | Token usage, budget, turn progress |
| **System Messages** | 5 | S→C | Info, errors, help, init progress, retries |
| **Session Management** | 3 | S→C | Session list, info snapshot, description |
| **Mid-Turn Prompts** | 4 | S→C | Queue, inject, interrupt, recovery |
| **Client Requests** | 8 | C→S | Messages, commands, config, history |
| **Workspace Config** | 8 | C↔S | Workspace list, create, select, configure |

---

## Part 4: Server → Client Events (Detailed)

### 4.1 Connection Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONNECTION LIFECYCLE                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Client connects (IPC or WebSocket)                                  │
│       │                                                              │
│       ▼                                                              │
│  ConnectedEvent                                                      │
│  ├─ protocol_version: "1.0"                                         │
│  └─ server_info: {capabilities, version, ...}                       │
│       │                                                              │
│       ▼                                                              │
│  SessionInfoEvent (full state snapshot)                               │
│  ├─ session_id, session_name, model_provider, model_name            │
│  ├─ sessions: [{id, name, model, is_loaded, client_count}, ...]     │
│  ├─ tools: [{name, description, enabled, plugin}, ...]              │
│  ├─ models: ["gemini-2.5-flash", "claude-sonnet-4-5", ...]         │
│  └─ user_inputs: ["previous prompt 1", "previous prompt 2", ...]   │
│       │                                                              │
│       ▼                                                              │
│  (Client is fully initialized, ready for interaction)                │
│                                                                      │
│  ...                                                                 │
│                                                                      │
│  DisconnectedEvent                                                   │
│  └─ (Client removed from broadcast list)                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

| Event | Key Fields | When Emitted |
|-------|------------|--------------|
| `ConnectedEvent` | `protocol_version`, `server_info` | Client establishes connection |
| `SessionInfoEvent` | `session_id`, `sessions`, `tools`, `models`, `user_inputs` | On connect/attach; full state snapshot for client initialization |

**Client Reaction (Rich Client):**
- Stores sessions/tools/models for tab completion
- Restores command history from `user_inputs`
- Updates status bar with model and session info
- If reconnecting: clears connection status, shows "Session restored!"

---

### 4.2 Agent Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT LIFECYCLE                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  AgentCreatedEvent                                                   │
│  ├─ agent_id: "main" | "subagent-abc123"                            │
│  ├─ agent_type: "main" | "subagent"                                 │
│  ├─ profile_name: "researcher" (optional)                           │
│  ├─ parent_agent_id: null | "main"                                  │
│  └─ icon_lines: ["  🔍  ", " Search"] (optional)                   │
│       │                                                              │
│       ▼                                                              │
│  AgentStatusChangedEvent ─── (repeats as status changes)            │
│  ├─ status: "active" | "idle" | "done" | "error"                   │
│  └─ error: "..." (only when status="error")                        │
│       │                                                              │
│       ├──► AgentOutputEvent(s) ─── (streaming text chunks)          │
│       │    ├─ source: "model" | "tool" | "system" | plugin_name    │
│       │    ├─ text: "Let me read that file."                        │
│       │    └─ mode: "write" (new block) | "append" (continue)      │
│       │                                                              │
│       ├──► ToolCallStartEvent / ToolCallEndEvent (tool activity)    │
│       │                                                              │
│       └──► ... (more output, tool calls, etc.)                      │
│       │                                                              │
│       ▼                                                              │
│  AgentCompletedEvent                                                 │
│  ├─ success: true/false                                             │
│  ├─ token_usage: {prompt: N, output: M, total: T}                  │
│  └─ turns_used: 5                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

| Event | Key Fields | When Emitted |
|-------|------------|--------------|
| `AgentCreatedEvent` | `agent_id`, `agent_type`, `profile_name`, `parent_agent_id` | New agent (main or subagent) is created |
| `AgentOutputEvent` | `agent_id`, `source`, `text`, `mode` | Each streaming text chunk from model/tool/system |
| `AgentStatusChangedEvent` | `agent_id`, `status`, `error` | Agent transitions between active/idle/done/error |
| `AgentCompletedEvent` | `agent_id`, `success`, `token_usage`, `turns_used` | Agent task finishes |

**Client Reaction (Rich Client):**

| Event | UI Element | Reaction |
|-------|-----------|----------|
| `AgentCreated` | Agent tabs | Registers agent, shows help text for main agent |
| `AgentOutput` | Output panel | Appends/extends text in agent's output buffer |
| `AgentStatusChanged(active)` | Status bar | Starts spinner animation, auto-selects agent tab |
| `AgentStatusChanged(done)` | Status bar | Stops spinner |
| `AgentCompleted` | Agent registry | Marks agent as completed |

---

### 4.3 Tool Execution

```
┌─────────────────────────────────────────────────────────────────────┐
│  TOOL EXECUTION EVENTS                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model requests tool call                                            │
│       │                                                              │
│       ▼                                                              │
│  ToolCallStartEvent                                                  │
│  ├─ tool_name: "readFile"                                           │
│  ├─ tool_args: {"path": "src/main.py"}                              │
│  └─ call_id: "tc-abc123" (for parallel tool correlation)            │
│       │                                                              │
│       ├──► ToolOutputEvent(s) ─── (live output, tail -f style)      │
│       │    ├─ call_id: "tc-abc123"                                  │
│       │    └─ chunk: "Building... 45% complete\n"                   │
│       │                                                              │
│       ▼                                                              │
│  ToolCallEndEvent                                                    │
│  ├─ tool_name: "readFile"                                           │
│  ├─ call_id: "tc-abc123"                                            │
│  ├─ success: true                                                   │
│  ├─ duration_seconds: 0.23                                          │
│  └─ error_message: null (or "File not found" if failed)             │
│                                                                      │
│                                                                      │
│  PARALLEL EXECUTION EXAMPLE:                                         │
│                                                                      │
│  ToolCallStartEvent (readFile, call_id="tc-1")    ───┐              │
│  ToolCallStartEvent (run, call_id="tc-2")         ───┤ concurrent   │
│  ToolCallStartEvent (glob_files, call_id="tc-3")  ───┘              │
│       │                                                              │
│       ├── ToolOutputEvent(call_id="tc-2", chunk="npm: installing")  │
│       ├── ToolCallEndEvent(call_id="tc-1", duration=0.05s)          │
│       ├── ToolOutputEvent(call_id="tc-2", chunk="npm: done")        │
│       ├── ToolCallEndEvent(call_id="tc-3", duration=0.12s)          │
│       └── ToolCallEndEvent(call_id="tc-2", duration=1.45s)          │
│                                                                      │
│  (call_id correlates start/output/end for each concurrent tool)     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Client Reaction (Rich Client):**

| Event | UI Element | Reaction |
|-------|-----------|----------|
| `ToolCallStart` | Tool tree | Creates tool entry with name+args, auto-scrolls |
| `ToolOutput` | Tool tree | Appends live output preview under tool entry |
| `ToolCallEnd` | Tool tree | Marks tool completed with duration/error, grays out |

---

### 4.4 Permission Flow

The permission flow is a **request-response cycle** involving three server events and one client response:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PERMISSION EVENT SEQUENCE                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Server                                               Client         │
│    │                                                    │            │
│    │  PermissionRequestedEvent                          │            │
│    │  ├─ tool_name: "updateFile"                       │            │
│    │  ├─ tool_args: {path: "src/main.py", ...}        │            │
│    │  ├─ prompt_lines: ["@@ -10,5 +10,7 @@", ...]    │            │
│    │  ├─ format_hint: "diff"                           │            │
│    │  ├─ warnings: "Modifies production code"          │            │
│    │  └─ response_options: [{key:"y", label:"yes"},    │            │
│    │                         {key:"n", label:"no"}, ...] │           │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                          Renders permission     │
│    │                                          panel with diff        │
│    │                                                    │            │
│    │  PermissionInputModeEvent                          │            │
│    │  ├─ request_id: "perm-001"                        │            │
│    │  ├─ call_id: "tc-abc123"                          │            │
│    │  └─ response_options: [{key, label, action}, ...]  │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                          Switches input to      │
│    │                                          permission mode        │
│    │                                          Shows y/n/a/t/i/all   │
│    │                                                    │            │
│    │  PermissionResponseRequest           (CLIENT → SERVER)          │
│    │  ├─ request_id: "perm-001"                        │            │
│    │  └─ response: "y"                                 │            │
│    │ ◄───────────────────────────────────────────────── │            │
│    │                                                    │            │
│    │  PermissionResolvedEvent                           │            │
│    │  ├─ request_id: "perm-001"                        │            │
│    │  ├─ granted: true                                 │            │
│    │  └─ method: "user_approved"                       │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                          Clears permission      │
│    │                                          panel, resumes normal  │
│    │                                                    │            │
└─────────────────────────────────────────────────────────────────────┘
```

**Client Reaction (Rich Client):**

| Event | UI Element | Reaction |
|-------|-----------|----------|
| `PermissionRequested` | Output panel | Renders permission content with diff highlighting |
| `PermissionInputMode` | Input field | Switches to permission mode, shows response options |
| `PermissionResolved` | Input field, Tool tree | Returns to normal input; shows grant/deny in tool tree |

---

### 4.5 Clarification Flow

A multi-question dialog where the model asks the user for information before proceeding:

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLARIFICATION EVENT SEQUENCE                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Server                                               Client         │
│    │                                                    │            │
│    │  ClarificationRequestedEvent                       │            │
│    │  ├─ request_id: "clar-001"                        │            │
│    │  ├─ tool_name: "request_clarification"            │            │
│    │  ├─ context_lines: ["Before proceeding..."]       │            │
│    │  └─ total_questions: 3                            │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                                    │            │
│    │  ClarificationQuestionEvent  (repeated per question)            │
│    │  ├─ question_index: 0                             │            │
│    │  ├─ question_type: "single_choice"                │            │
│    │  ├─ question_text: "Which auth method?"           │            │
│    │  └─ options: [{label: "JWT"}, {label: "OAuth"}]   │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                                    │            │
│    │  ClarificationInputModeEvent                       │            │
│    │  ├─ question_index: 0                             │            │
│    │  └─ total_questions: 3                            │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                          Shows question 1/3     │
│    │                                          Switches input mode    │
│    │                                                    │            │
│    │  ClarificationResponseRequest       (CLIENT → SERVER)           │
│    │  ├─ question_index: 0                             │            │
│    │  └─ response: "JWT"                               │            │
│    │ ◄───────────────────────────────────────────────── │            │
│    │                                                    │            │
│    │  ... (repeat for questions 1, 2) ...               │            │
│    │                                                    │            │
│    │  ClarificationResolvedEvent                        │            │
│    │  ├─ request_id: "clar-001"                        │            │
│    │  └─ qa_pairs: [["Which auth?","JWT"],             │            │
│    │                 ["Token expiry?","1h"], ...]       │            │
│    │ ─────────────────────────────────────────────────► │            │
│    │                                          Shows Q&A summary      │
│    │                                          Returns to normal      │
│    │                                                    │            │
└─────────────────────────────────────────────────────────────────────┘
```

**Client Reaction (Rich Client):**

| Event | UI Element | Reaction |
|-------|-----------|----------|
| `ClarificationRequested` | Tool tree | Shows "Q#/## awaiting response" under tool |
| `ClarificationQuestion` | Output panel | Content flows through `AgentOutputEvent` |
| `ClarificationInputMode` | Input field | Switches to clarification mode, shows progress |
| `ClarificationResolved` | Tool tree, Input | Shows Q&A pairs; returns to normal input |

---

### 4.6 Reference Selection & Workspace Mismatch

Two additional request-response flows follow the same pattern:

| Flow | Trigger | Server Events | Client Response |
|------|---------|---------------|-----------------|
| **Reference Selection** | Model calls `selectReferences` | `Requested` → `Resolved` | User picks references (e.g., "1,3,4" or "all") |
| **Workspace Mismatch** | Client attaches to session from different path | `Requested` → `Resolved` | User picks "switch" / "new session" / "cancel" |

---

### 4.7 Plan Management

```
┌─────────────────────────────────────────────────────────────────────┐
│  PLAN EVENTS                                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PlanUpdatedEvent (repeats as steps change status)                   │
│  ├─ agent_id: "main"                                                │
│  ├─ plan_name: "Refactor auth module"                               │
│  └─ steps:                                                          │
│     ├─ {content: "Read existing auth code",                         │
│     │   status: "completed", active_form: null}                     │
│     ├─ {content: "Extract JWT logic",                               │
│     │   status: "in_progress", active_form: "Extracting JWT..."}   │
│     ├─ {content: "Update imports",                                  │
│     │   status: "pending", blocked_by: ["step-2"]}                 │
│     └─ {content: "Run tests",                                       │
│         status: "pending", depends_on: ["step-3"]}                 │
│       │                                                              │
│       ▼                                                              │
│  PlanClearedEvent                                                    │
│  └─ agent_id: "main"                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Client Reaction (Rich Client):**

| Event | UI Element | Reaction |
|-------|-----------|----------|
| `PlanUpdated` | Plan panel (sticky top) | Shows step progression with progress bar, dependency indicators |
| `PlanCleared` | Plan panel | Hides/removes plan panel |

---

### 4.8 Context & Token Tracking

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONTEXT & TOKEN EVENTS                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  During turn execution:                                              │
│                                                                      │
│  TurnProgressEvent(s) ─── (incremental, real-time updates)          │
│  ├─ total_tokens: 15,230                                            │
│  ├─ prompt_tokens: 12,100                                           │
│  ├─ output_tokens: 3,130                                            │
│  ├─ context_limit: 128,000                                          │
│  ├─ percent_used: 11.9%                                             │
│  └─ pending_tool_calls: 2                                           │
│                                                                      │
│  After turn completes:                                               │
│                                                                      │
│  TurnCompletedEvent                                                  │
│  ├─ turn_number: 3                                                  │
│  ├─ prompt_tokens: 12,100                                           │
│  ├─ output_tokens: 3,130                                            │
│  ├─ duration_seconds: 4.7                                           │
│  ├─ function_calls: [{name, args, result}, ...]                    │
│  └─ formatted_text: "..." (optional post-processed output)         │
│                                                                      │
│  ContextUpdatedEvent ─── (cumulative session-wide usage)            │
│  ├─ total_tokens: 45,600                                            │
│  ├─ percent_used: 35.6%                                             │
│  ├─ tokens_remaining: 82,400                                       │
│  ├─ turns: 3                                                        │
│  ├─ gc_threshold: 80.0                                              │
│  ├─ gc_strategy: "hybrid"                                           │
│  └─ gc_continuous_mode: false                                       │
│                                                                      │
│  InstructionBudgetEvent ─── (per-source token breakdown)            │
│  └─ budget_snapshot:                                                │
│     ├─ context_limit: 128,000                                       │
│     ├─ total_tokens: 45,600                                         │
│     ├─ utilization_percent: 35.6%                                   │
│     └─ entries:                                                     │
│        ├─ system: 200 tokens                                        │
│        ├─ session: 150 tokens                                       │
│        ├─ plugin: 1,800 tokens (per-tool breakdown)                 │
│        ├─ enrichment: 260 tokens                                    │
│        └─ conversation: 43,190 tokens (per-turn breakdown)          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Client Reaction (Rich Client):**

| Event | UI Element | Reaction |
|-------|-----------|----------|
| `TurnProgress` | Status bar | Real-time token counter updates during generation |
| `TurnCompleted` | Output panel | Flushes pending content, stops spinner |
| `ContextUpdated` | Status bar | Updates context % display, GC threshold indicator |
| `InstructionBudget` | Budget panel (Ctrl+B) | Shows per-source token breakdown |

---

### 4.9 System Messages

| Event | Key Fields | When Emitted | Client Reaction |
|-------|------------|--------------|-----------------|
| `SystemMessageEvent` | `message`, `style` (info/warning/error/success) | System notifications | Appends styled message to output panel |
| `ErrorEvent` | `error`, `error_type`, `recoverable` | Exceptions | Shows error in bold red style |
| `HelpTextEvent` | `lines` (list of (text, style) tuples) | `help` commands | Opens pager with formatted help |
| `InitProgressEvent` | `step`, `status`, `step_number`, `total_steps` | Session initialization | Shows step-by-step progress with OK/error indicators |
| `RetryEvent` | `attempt`, `max_attempts`, `delay`, `error_type` | API transient errors | Shows retry countdown in warning style |

```
┌─────────────────────────────────────────────────────────────────────┐
│  INIT PROGRESS EXAMPLE                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  InitProgressEvent(step="Loading plugins", status="running", 1/4)   │
│    → "Loading plugins..."                                            │
│                                                                      │
│  InitProgressEvent(step="Loading plugins", status="done", 1/4)      │
│    → "Loading plugins... OK"  (updates in place)                    │
│                                                                      │
│  InitProgressEvent(step="Connecting MCP", status="running", 2/4)    │
│    → "Connecting MCP servers..."                                     │
│                                                                      │
│  InitProgressEvent(step="Connecting MCP", status="error", 2/4)      │
│    → "Connecting MCP servers... FAILED: timeout"                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 4.10 Mid-Turn Prompts

When a user sends input while the model is still processing, the message is queued rather than rejected:

```
┌─────────────────────────────────────────────────────────────────────┐
│  MID-TURN PROMPT FLOW                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model is generating...                                              │
│       │                                                              │
│  User types: "Actually, use TypeScript instead"                      │
│       │                                                              │
│       ▼                                                              │
│  MidTurnPromptQueuedEvent                                            │
│  ├─ text: "Actually, use TypeScript instead"                        │
│  └─ position_in_queue: 0                                            │
│       │                                                              │
│       ▼  (Client shows pending prompt indicator)                    │
│  ┌───────────────────────────────────────┐                          │
│  │  Status:  Model generating... [||||||||||...]                    │
│  │  Queued:  "Actually, use TypeScript instead"                     │
│  └───────────────────────────────────────┘                          │
│       │                                                              │
│       ├──► OPTION A: Natural pause point reached                    │
│       │    MidTurnPromptInjectedEvent                                │
│       │    └─ text: "Actually, use TypeScript instead"              │
│       │    (Client removes from pending bar, model processes it)    │
│       │                                                              │
│       └──► OPTION B: Prompt arrives during streaming                │
│            MidTurnInterruptEvent                                     │
│            ├─ partial_response_chars: 340                           │
│            └─ user_prompt_preview: "Actually, use Type..."          │
│            (Client shows "[Pivoting to your input...]")             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 4.11 Session Recovery

```
┌─────────────────────────────────────────────────────────────────────┐
│  SESSION RECOVERY                                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Client reconnects after server restart or network drop              │
│       │                                                              │
│       ▼                                                              │
│  Server calls emit_current_state():                                  │
│  ├─ SessionInfoEvent (full state snapshot)                          │
│  ├─ AgentCreatedEvent (for each tracked agent)                      │
│  ├─ AgentStatusChangedEvent (if non-idle)                           │
│  ├─ InstructionBudgetEvent (for each agent)                         │
│  └─ Clears stale pending permission/clarification requests          │
│       │                                                              │
│       ├──► If turn was interrupted:                                  │
│       │    InterruptedTurnRecoveredEvent                              │
│       │    ├─ recovered_calls: 3 (pending tool calls recovered)     │
│       │    └─ action_taken: "synthetic_error"                       │
│       │                                                              │
│       └──► Client shows "Session restored!" success message         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Client → Server Events (Requests)

### Request Types

| Event | Key Fields | Purpose |
|-------|------------|---------|
| `SendMessageRequest` | `text`, `attachments` | Send user prompt to model |
| `PermissionResponseRequest` | `request_id`, `response` | Respond to permission prompt (y/n/a/t/i/all) |
| `ClarificationResponseRequest` | `request_id`, `question_index`, `response` | Answer a clarification question |
| `ReferenceSelectionResponseRequest` | `request_id`, `response` | Select references to include |
| `WorkspaceMismatchResponseRequest` | `request_id`, `response` | Resolve workspace path conflict |
| `StopRequest` | `agent_id` (optional) | Cancel current operation |
| `CommandRequest` | `command`, `args` | Execute a command (model, reset, permissions, etc.) |
| `ClientConfigRequest` | `trace_log_path`, `terminal_width`, `working_dir`, `env_file` | Send client configuration to server |
| `GetInstructionBudgetRequest` | `agent_id` | Request instruction budget breakdown |
| `HistoryRequest` | `agent_id` | Request conversation history |
| `ToolDisableRequest` | `tool_name` | Disable a specific tool |

---

## Part 6: Transport Layers

Events travel between server and client over two transport options:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSPORT ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                      ┌─────────────────┐                            │
│                      │  JAATO SERVER   │                            │
│                      │   (daemon)      │                            │
│                      └────────┬────────┘                            │
│                               │                                      │
│                     emit(event: Event)                               │
│                               │                                      │
│                ┌──────────────┴──────────────┐                      │
│                │      SESSION MANAGER        │                      │
│                │  (routes to attached clients)│                      │
│                └──────┬─────────────┬────────┘                      │
│                       │             │                                │
│          ┌────────────┴───┐   ┌────┴────────────┐                   │
│          │    IPC Layer    │   │  WebSocket Layer │                   │
│          │                │   │                  │                   │
│          │  Unix Domain   │   │   ws://host:port │                   │
│          │  Socket        │   │                  │                   │
│          │                │   │  Native WS text  │                   │
│          │  Length-prefix  │   │  frames          │                   │
│          │  framing:      │   │                  │                   │
│          │  [4B len][JSON] │   │  [JSON message]  │                   │
│          │                │   │                  │                   │
│          │  Max: 10 MB    │   │  Standard WS     │                   │
│          │  per message   │   │  limits          │                   │
│          └────────┬───────┘   └────────┬─────────┘                   │
│                   │                    │                              │
│              ┌────┴────┐          ┌────┴────┐                       │
│              │  Local  │          │ Remote  │                       │
│              │ Clients │          │ Clients │                       │
│              └─────────┘          └─────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Transport Comparison

| Aspect | IPC (Unix Socket) | WebSocket |
|--------|-------------------|-----------|
| **Protocol** | Length-prefixed JSON | WS text frames |
| **Framing** | 4-byte big-endian u32 + UTF-8 payload | Native WebSocket framing |
| **Max message** | 10 MB | Standard WS limits |
| **Scope** | Local machine only | Local or remote |
| **Queuing** | Per-client `asyncio.Queue` (unbounded) | Shared `asyncio.Queue`, fan-out |
| **Thread safety** | `call_soon_threadsafe()` | `run_coroutine_threadsafe()` |
| **Disconnection** | Skips send silently | Removes from client dict |

### Ordering Guarantees

| Guarantee | Description |
|-----------|-------------|
| **Per-client FIFO** | Events sent to a specific client maintain order |
| **Broadcast consistency** | All clients receive the same event in the same order |
| **No batching** | Each event is serialized and transmitted individually |
| **At-most-once delivery** | Disconnected clients miss events (recovered via `emit_current_state`) |

---

## Part 7: Event Emission Pipeline (Server Side)

How events originate inside the server and reach clients:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVENT EMISSION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. SOURCE (Runtime Hooks)                                           │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  Permission hooks:    on_requested(), on_resolved()      │     │
│     │  Clarification hooks: on_requested(), on_resolved()      │     │
│     │  Output callback:     on_output(source, text, mode)      │     │
│     │  Usage callback:      usage_update_callback(usage)       │     │
│     │  Retry callback:      retry_callback(attempt, delay)     │     │
│     │  Interrupt callback:  mid_turn_interrupt_callback(...)   │     │
│     └────────────────────────────────┬────────────────────────┘     │
│                                      │                               │
│  2. EVENT CONSTRUCTION                │                               │
│     ┌────────────────────────────────▼────────────────────────┐     │
│     │  server.emit(PermissionRequestedEvent(                   │     │
│     │      agent_id="main",                                    │     │
│     │      request_id="perm-001",                              │     │
│     │      tool_name="updateFile",                             │     │
│     │      tool_args={...},                                    │     │
│     │      format_hint="diff",                                 │     │
│     │  ))                                                      │     │
│     └────────────────────────────────┬────────────────────────┘     │
│                                      │                               │
│  3. SERIALIZATION                    │                               │
│     ┌────────────────────────────────▼────────────────────────┐     │
│     │  event.to_json()                                         │     │
│     │  → Enum values converted to strings                      │     │
│     │  → Dataclass fields serialized via asdict()              │     │
│     │  → json.dumps() produces UTF-8 string                   │     │
│     └────────────────────────────────┬────────────────────────┘     │
│                                      │                               │
│  4. ROUTING                          │                               │
│     ┌────────────────────────────────▼────────────────────────┐     │
│     │  Session Manager                                         │     │
│     │  ├─ Updates in-memory state (descriptions, turn tracking)│     │
│     │  └─ Broadcasts to all clients attached to session        │     │
│     └────────────────────────────────┬────────────────────────┘     │
│                                      │                               │
│  5. TRANSPORT                        │                               │
│     ┌────────────────────────────────▼────────────────────────┐     │
│     │  IPC: queue_event(client_id, event)                      │     │
│     │       → call_soon_threadsafe() into event loop           │     │
│     │       → Per-client asyncio.Queue                         │     │
│     │       → Broadcast loop: dequeue → write_message()        │     │
│     │       → length_prefix + payload → socket.drain()         │     │
│     │                                                          │     │
│     │  WebSocket: run_coroutine_threadsafe(queue.put(event))   │     │
│     │       → Shared asyncio.Queue                             │     │
│     │       → Broadcast loop: dequeue → send to all clients    │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 8: Client UI Element Mapping

How the TUI client maps events to visual elements:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLIENT UI STRUCTURE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  SESSION BAR                                                 │    │
│  │  SessionInfo, SessionDescriptionUpdated                      │    │
│  │  Shows: session ID, description, workspace path              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌───────────────────────────────────┬─────────────────────────┐    │
│  │  AGENT TABS                       │  PLAN PANEL (sticky)    │    │
│  │  AgentCreated,                    │  PlanUpdated,           │    │
│  │  AgentStatusChanged               │  PlanCleared            │    │
│  │  Shows: agent list, active tab    │  Shows: step list,      │    │
│  │  with spinner                     │  progress bar           │    │
│  ├───────────────────────────────────┴─────────────────────────┤    │
│  │                                                              │    │
│  │  OUTPUT PANEL                                                │    │
│  │  AgentOutput, SystemMessage, Error, Retry,                   │    │
│  │  InitProgress, MidTurnInterrupt                              │    │
│  │                                                              │    │
│  │  ┌────────────────────────────────────────────────────────┐  │    │
│  │  │  TOOL TREE (inline in output)                          │  │    │
│  │  │  ToolCallStart, ToolCallEnd, ToolOutput,               │  │    │
│  │  │  PermissionInputMode, PermissionResolved,              │  │    │
│  │  │  ClarificationInputMode, ClarificationResolved         │  │    │
│  │  │                                                        │  │    │
│  │  │  Shows: tool name, args, duration, live output,        │  │    │
│  │  │  approval status, Q&A pairs                            │  │    │
│  │  └────────────────────────────────────────────────────────┘  │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PENDING PROMPTS BAR                                         │    │
│  │  MidTurnPromptQueued, MidTurnPromptInjected                  │    │
│  │  Shows: queued prompts with preview text                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  INPUT FIELD                                                 │    │
│  │  Mode switches based on:                                     │    │
│  │  • Normal: SendMessageRequest (user typing)                  │    │
│  │  • Permission: PermissionInputMode → PermissionResponse      │    │
│  │  • Clarification: ClarificationInputMode → ClarificationResp│    │
│  │  • Reference: ReferenceSelectionRequested → ReferenceResp    │    │
│  │  • Mismatch: WorkspaceMismatchRequested → MismatchResp       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  STATUS BAR (bottom)                                         │    │
│  │  ContextUpdated, TurnProgress, SessionInfo                   │    │
│  │  Shows: token %, model name, GC threshold, turn count        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  PAGER VIEW (full-screen overlay, temporary)                        │
│  HelpText, SessionList, ToolStatus, History                         │
│  Shows: formatted content with scrolling                            │
│                                                                      │
│  BUDGET PANEL (Ctrl+B overlay)                                      │
│  InstructionBudget                                                   │
│  Shows: per-source token breakdown                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete Event-to-UI Mapping

| UI Element | Events That Affect It | Nature of Change |
|-----------|----------------------|------------------|
| **Session Bar** | `SessionInfo`, `SessionDescriptionUpdated` | Session ID, description, workspace |
| **Agent Tabs** | `AgentCreated`, `AgentStatusChanged` | Agent list, active/spinner state |
| **Plan Panel** | `PlanUpdated`, `PlanCleared` | Step list, progress %, dependencies |
| **Output Panel** | `AgentOutput`, `SystemMessage`, `Error`, `Retry`, `InitProgress`, `MidTurnInterrupt` | Streaming text, styled messages, in-place updates |
| **Tool Tree** | `ToolCallStart`, `ToolCallEnd`, `ToolOutput`, `Permission*`, `Clarification*` | Tool entries, duration, live output, approval/Q&A |
| **Pending Prompts** | `MidTurnPromptQueued`, `MidTurnPromptInjected` | Queued prompt indicators |
| **Input Field** | `PermissionInputMode`, `ClarificationInputMode`, `ReferenceSelectionRequested`, `WorkspaceMismatchRequested`, all `*Resolved` | Mode switching (normal ↔ response) |
| **Status Bar** | `ContextUpdated`, `TurnProgress`, `SessionInfo` | Token %, model info, GC status |
| **Pager** | `HelpText`, `SessionList`, `ToolStatus`, `History` | Full-screen formatted content |
| **Budget Panel** | `InstructionBudget` | Per-source token breakdown |
| **Spinner** | `AgentStatusChanged(active/done)` | Animation on/off |
| **Completion** | `CommandList`, `SessionInfo` | Tab-completion entries |

---

## Part 9: Complete Turn Lifecycle (Event Sequence)

A full turn from user prompt to model completion, showing every event emitted:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE TURN EVENT SEQUENCE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  USER INPUT                                                          │
│  ◄── SendMessageRequest(text="Add logging to auth module")          │
│       │                                                              │
│       ▼                                                              │
│  AgentStatusChangedEvent(status="active")                            │
│  │   → Client starts spinner                                        │
│  │                                                                   │
│  ▼                                                                   │
│  AgentOutputEvent(source="model", text="Let me read the...",        │
│  │                mode="write")                                      │
│  AgentOutputEvent(source="model", text=" authentication files.",    │
│  │                mode="append")                                     │
│  │   → Client streams text to output panel                          │
│  │                                                                   │
│  ▼                                                                   │
│  ToolCallStartEvent(tool_name="readFile",                            │
│  │                   tool_args={path:"src/auth.py"}, call_id="tc-1")│
│  │   → Client shows tool in tree                                    │
│  │                                                                   │
│  ▼                                                                   │
│  ToolCallEndEvent(tool_name="readFile", call_id="tc-1",              │
│  │                 success=true, duration=0.05)                      │
│  │   → Client marks tool completed                                  │
│  │                                                                   │
│  ▼                                                                   │
│  TurnProgressEvent(total_tokens=8200, percent_used=6.4%)             │
│  │   → Client updates status bar counters                           │
│  │                                                                   │
│  ▼                                                                   │
│  AgentOutputEvent(source="model", text="I'll add logging...",       │
│  │                mode="write")                                      │
│  │                                                                   │
│  ▼                                                                   │
│  ToolCallStartEvent(tool_name="updateFile", call_id="tc-2")         │
│  │   → Client shows tool in tree                                    │
│  │                                                                   │
│  ▼                                                                   │
│  PermissionRequestedEvent(tool_name="updateFile",                    │
│  │    prompt_lines=["@@ -10,5 +10,7 @@",...], format_hint="diff")  │
│  │   → Client renders diff in output                                │
│  │                                                                   │
│  ▼                                                                   │
│  PermissionInputModeEvent(request_id="perm-001", call_id="tc-2")    │
│  │   → Client switches input to permission mode                     │
│  │                                                                   │
│  ◄── PermissionResponseRequest(request_id="perm-001", response="y") │
│  │   → User approves                                                │
│  │                                                                   │
│  ▼                                                                   │
│  PermissionResolvedEvent(granted=true, method="user_approved")       │
│  │   → Client returns to normal input, updates tool tree            │
│  │                                                                   │
│  ▼                                                                   │
│  ToolCallEndEvent(tool_name="updateFile", call_id="tc-2",            │
│  │                 success=true, duration=0.12)                      │
│  │                                                                   │
│  ▼                                                                   │
│  AgentOutputEvent(source="model", text="Done. I've added ...",      │
│  │                mode="write")                                      │
│  │                                                                   │
│  ▼                                                                   │
│  TurnCompletedEvent(turn_number=1, prompt_tokens=8200,               │
│  │                   output_tokens=1450, duration=12.3)              │
│  │   → Client flushes output                                        │
│  │                                                                   │
│  ▼                                                                   │
│  ContextUpdatedEvent(total_tokens=9650, percent_used=7.5%,           │
│  │                    turns=1)                                        │
│  │   → Client updates status bar                                    │
│  │                                                                   │
│  ▼                                                                   │
│  AgentStatusChangedEvent(status="idle")                               │
│  │   → Client stops spinner                                         │
│  │                                                                   │
│  ▼                                                                   │
│  (Session idle, waiting for next user input)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 10: Display Refresh Strategy

The TUI client uses a debounced refresh mechanism to balance responsiveness with rendering efficiency:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DISPLAY REFRESH STRATEGY                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Event arrives from server                                           │
│       │                                                              │
│       ▼                                                              │
│  Event handler updates internal state                                │
│  (agent registry, output buffer, pending requests)                   │
│       │                                                              │
│       ▼                                                              │
│  display.refresh()                                                   │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────┐                │
│  │  DEBOUNCE (during streaming)                     │                │
│  │                                                  │                │
│  │  Raw events:  ||||||||||||||||||||||||||||||||    │                │
│  │               50+ refresh calls per second       │                │
│  │                                                  │                │
│  │  After debounce: |   |   |   |   |   |   |      │                │
│  │                  ~20 actual renders per second    │                │
│  └─────────────────────────────────────────────────┘                │
│       │                                                              │
│       ▼                                                              │
│  app.invalidate()                                                    │
│  (schedules redraw in prompt_toolkit event loop)                    │
│  (thread-safe: events arrive from server's model thread)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

| Refresh Type | When Used | Rate |
|-------------|-----------|------|
| **Debounced** | During streaming (`AgentOutput` bursts) | ~20/sec |
| **Immediate** | Mode switches (permission, clarification) | On event |
| **None** | Lightweight updates (`TurnProgress`) | Status bar auto-refreshes |

---

## Part 11: Visual Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JAATO EVENT PROTOCOL OVERVIEW                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                   ┌─────────────────────────┐                       │
│                   │     JAATO SERVER         │                       │
│                   │      (daemon)            │                       │
│                   │                          │                       │
│                   │  ┌──────────────────┐    │                       │
│                   │  │ Runtime Hooks    │    │                       │
│                   │  │ • on_output      │    │                       │
│                   │  │ • on_permission  │    │                       │
│                   │  │ • on_usage       │    │                       │
│                   │  │ • on_retry       │    │                       │
│                   │  └────────┬─────────┘    │                       │
│                   │           │               │                       │
│                   │  ┌────────▼─────────┐    │                       │
│                   │  │  Event Emitter   │    │                       │
│                   │  │  emit(Event)     │    │                       │
│                   │  └────────┬─────────┘    │                       │
│                   │           │               │                       │
│                   │  ┌────────▼─────────┐    │                       │
│                   │  │ Session Manager  │    │                       │
│                   │  │ (broadcast to    │    │                       │
│                   │  │  all clients)    │    │                       │
│                   │  └────────┬─────────┘    │                       │
│                   └───────────┼───────────────┘                       │
│                               │                                      │
│              ┌────────────────┴────────────────┐                    │
│              │         EVENT STREAM             │                    │
│              │     (40+ typed events)           │                    │
│              │     JSON over IPC / WebSocket    │                    │
│              └───────┬────────────────┬─────────┘                    │
│                      │                │                               │
│             ┌────────▼───────┐ ┌──────▼─────────┐                   │
│             │   TUI CLIENT    │ │  OTHER CLIENTS  │                   │
│             │                │ │  (web, headless) │                   │
│             │ Event Handler  │ │                  │                   │
│             │      │         │ │  Same events,    │                   │
│             │      ▼         │ │  different UI    │                   │
│             │ ┌──────────┐   │ │                  │                   │
│             │ │ Output   │   │ └──────────────────┘                   │
│             │ │ Buffer   │   │                                       │
│             │ │ + Tool   │   │                                       │
│             │ │ Tree     │   │                                       │
│             │ └──────────┘   │                                       │
│             │      │         │                                       │
│             │      ▼         │                                       │
│             │ ┌──────────┐   │                                       │
│             │ │ Terminal  │   │                                       │
│             │ │ Display   │   │                                       │
│             │ └──────────┘   │                                       │
│             └────────────────┘                                       │
│                                                                      │
│  ════════════════════════════════════════════════════════════════════│
│                                                                      │
│   THE EVENT PROTOCOL ENABLES:                                        │
│                                                                      │
│   Daemon Architecture    →    Clients connect/disconnect freely      │
│   ────────────────────        ──────────────────────────────         │
│   • Server runs as daemon     • Reconnection with full state replay │
│   • Multiple simultaneous     • Different UI implementations        │
│     clients per session       • Forward-compatible serialization     │
│   • Persistent across         • Thread-safe, async-native           │
│     client restarts           • FIFO ordering guaranteed            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 12: Related Documentation

| Document | Focus |
|----------|-------|
| [jaato_model_harness.md](jaato_model_harness.md) | Instructions, tools, and permissions layers |
| [jaato_instruction_sources.md](jaato_instruction_sources.md) | System instruction assembly and enrichment |
| [jaato_tool_system.md](jaato_tool_system.md) | Tool architecture, discoverability, execution |
| [jaato_permission_system.md](jaato_permission_system.md) | Permission evaluation, channels, suspension |
| [architecture.md](architecture.md) | Server-first architecture overview |
| [sequence-diagram-architecture.md](sequence-diagram-architecture.md) | Client-server interaction flows |

---

## Part 13: Color Coding Suggestion for Infographic

- **Blue:** Server → Client events (notifications, state updates)
- **Green:** Agent lifecycle events (created, output, status, completed)
- **Orange:** Tool execution events (start, output, end)
- **Red:** Permission/safety flow events (request, input mode, response, resolved)
- **Yellow:** Clarification and reference selection flows (user interaction)
- **Purple:** Context and token tracking events (budget, progress, GC)
- **Gray:** Transport layers and serialization infrastructure
- **Cyan:** Client → Server requests (messages, commands, responses)
- **Pink:** Mid-turn prompt events (queue, inject, interrupt)
