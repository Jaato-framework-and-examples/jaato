# Design Philosophy

This document captures jaato's opinionated design decisions and the reasoning behind them. These principles guide architectural choices and help maintain consistency across the codebase.

---

## 1. Model-First Decision Making

**Principle**: Trust the model to make contextual decisions rather than forcing users to specify intent explicitly.

### Example: Mid-Turn Prompt Injection

When a user sends a message while the model is actively running, jaato queues it and lets the model decide how to handle it:

- **Steering instruction** ("Stop, that's wrong!") → Model interrupts and addresses immediately
- **Continuation** ("Also check X when done") → Model acknowledges and continues current task
- **Status query** ("What's your progress?") → Model can answer inline

**Why not explicit user controls?**

Some systems (e.g., pi-mono) require users to choose between "steering" (Enter) vs "follow-up" (Alt+Enter) keys. We reject this because:

1. **Context awareness** - The model knows what it's doing and can assess urgency better than the user predicting from outside
2. **Lower cognitive load** - Users don't need to learn semantic differences between key combinations
3. **Natural interaction** - Matches how you'd communicate with a human collaborator
4. **Fewer wrong choices** - Users can't accidentally interrupt when they meant to queue, or vice versa

**Implementation**: `shared/message_queue.py` uses a priority queue with `SourceType` (PARENT, USER, SYSTEM, CHILD), but all user messages are treated as high-priority. The model receives them at natural pause points and decides the appropriate response.

---

## 2. Server-First Architecture

**Principle**: The server is the source of truth; clients are thin presentation layers.

### Why Daemon Mode?

Jaato runs as a daemon with clients connecting via IPC or WebSocket. This enables:

- **Multi-client support** - Multiple UIs (TUI, web, IDE) can connect to the same session
- **Session persistence** - Server maintains state across client reconnections
- **Resource sharing** - Single runtime for multiple agents, shared token ledger
- **Clean separation** - Business logic in server, presentation in clients

**Contrast**: Many CLI tools run as single processes where the UI and logic are intertwined. This makes it hard to add new client types or maintain state across sessions.

---

## 3. Separation of Data and Presentation

**Principle**: Pipeline produces structured data; clients choose optimal presentation.

### The Pipeline-Presentation Split

| Layer | Responsibility |
|-------|----------------|
| **Pipeline** (`shared/plugins/`, `server/`) | Emit semantic events with structured data |
| **Client** (`rich-client/`) | Choose UX presentation based on context |

**Example - Clarification Plugin**:
```
Pipeline: on_resolved(tool_name, qa_pairs=[(question, answer), ...])
Client:   Receives qa_pairs, decides: table? stacked? inline?
          Applies theme, calculates widths, handles wrapping
```

**Benefits**:
- Pipeline code remains testable without UI dependencies
- Multiple clients can present the same data differently
- Presentation can evolve without changing pipeline logic
- Terminal size, theme, and context inform display choices

---

## 4. First-Class MCP Support

**Principle**: MCP is a valuable integration layer, not a token-wasting overhead.

### Why Built-In MCP?

Some systems (e.g., pi-mono) deliberately exclude MCP, arguing it wastes 7-9% of context on tool descriptions. We disagree:

1. **Ecosystem access** - MCP provides immediate access to a growing ecosystem of servers
2. **Standardization** - Common protocol means tools work across different agents
3. **Token efficiency via deferred loading** - Jaato's `discoverability` attribute lets tools be "discoverable" (loaded on-demand) vs "core" (always present)

**Mitigation**: Rather than rejecting MCP, we address token overhead through:
- Deferred tool loading (`JAATO_DEFERRED_TOOLS=true`)
- `list_tools()` → `get_tool_schemas()` workflow for progressive discovery
- Multi-server management with `MCPClientManager`

---

## 5. Parallel Tool Execution by Default

**Principle**: When the model requests multiple tools, execute them concurrently unless there are dependencies.

### Implementation

- Thread pool executes up to 8 concurrent tools per turn
- Thread-safe callbacks via thread-local storage
- Enabled by default (`JAATO_PARALLEL_TOOLS=true`)

**Why parallel by default?**

Many tool calls are independent (e.g., reading multiple files, searching different directories). Sequential execution wastes time. The model implicitly signals independence by requesting multiple tools in one response.

---

## 6. Proactive Garbage Collection

**Principle**: Monitor context usage and trigger GC automatically rather than waiting for failures.

### Threshold-Based Triggering

```python
gc_config = GCConfig(
    threshold_percent=80.0,    # Trigger when context is 80% full
    preserve_recent_turns=5,   # Keep last 5 turns
    auto_trigger=True,
)
```

**Why proactive?**

- Prevents context overflow errors mid-conversation
- Multiple strategies (truncate, summarize, hybrid) for different needs
- Preserves recent context while compacting older history

---

## 7. Plugin Auto-Wiring

**Principle**: Plugins should just work without manual dependency injection.

### Automatic Lifecycle

| Method | When Called | By |
|--------|-------------|-----|
| `set_plugin_registry(registry)` | During `expose_tool()` | PluginRegistry |
| `set_session(session)` | During `configure()` | JaatoSession |
| `set_workspace_path(path)` | After `expose_all()` | PluginRegistry |

**Benefits**:
- Plugin authors don't need to understand wiring internals
- Reduces boilerplate in plugin implementations
- Consistent initialization across all plugin types

---

## 8. Subagents Share Runtime, Not State

**Principle**: Subagents should be lightweight to spawn but isolated in conversation state.

### Architecture

```
JaatoRuntime (shared)
├── Provider config
├── Plugin registry
├── Permissions
└── Token ledger

JaatoSession (per-agent)
├── Conversation history
├── Cancel token
└── Message queue
```

**Why this split?**

- **Fast spawning** - `create_session()` is lightweight; no redundant connections
- **Resource efficiency** - Registry, permissions, ledger shared across agents
- **State isolation** - Each agent has independent history and control flow
- **Parent-child communication** - Via injection queue, not shared state

---

## 9. Permission Responses Are Granular

**Principle**: Give users fine-grained control over permission grants, not just yes/no.

### Response Options

| Response | Meaning |
|----------|---------|
| `y` / `yes` | Allow this one time |
| `n` / `no` | Deny this one time |
| `a` / `always` | Whitelist permanently |
| `t` / `turn` | Allow until model finishes this response |
| `i` / `idle` | Allow until session goes idle |
| `once` | Allow this specific invocation only |
| `never` | Blacklist permanently |
| `all` | Allow all pending permissions |

**Why granular?**

- `turn` is perfect for "let it run this batch of file edits"
- `idle` handles multi-turn autonomous work
- `always`/`never` reduce future interruptions
- Users maintain control without constant prompting

---

## 10. Streaming with Cancellation Tokens

**Principle**: Long-running operations should be cancellable at any point.

### Implementation

```python
class CancelToken:
    """Thread-safe cancellation signaling."""

class CancelledException:
    """Raised when operation is cancelled."""

class FinishReason(Enum):
    CANCELLED = "cancelled"
```

**Cancellation points**:
- `client.stop()` / `session.request_stop()`
- Checked during streaming, tool execution, subagent work
- Propagates through the call stack via exception

---

## 11. Waypoints with Shared Ownership

**Principle**: Both user and model can create checkpoints, with an ownership model that protects user work while giving the model agency to manage its own.

### The Ownership Model

Waypoints form a navigable tree structure where both parties can participate:

| Owner | Created By | Model Can... |
|-------|------------|--------------|
| **User** | `waypoint create` command | View, restore (with permission), NOT delete |
| **Model** | `create_waypoint` tool | View, restore, delete freely |

All waypoints use sequential IDs (w1, w2, w3...) regardless of owner. Ownership is tracked separately.

### Why Model Agency?

The model can proactively protect against mistakes:

```
1. create_waypoint("before auth refactor")  → w1 (model-owned)
2. Make risky changes across multiple files
3. If changes fail: restore_waypoint("w1")  → reverts all files
4. If changes work: delete_waypoint("w1")   → cleanup
```

This is more powerful than user-only checkpoints because:

1. **Proactive safety** - Model creates checkpoints before risky operations without user intervention
2. **Self-correction** - Model can undo its own mistakes without asking permission
3. **Clean workspace** - Model can delete its own checkpoints when no longer needed

### Why Ownership Boundaries?

User-created waypoints are protected:

- Model cannot delete user waypoints (prevents accidental loss of user safety nets)
- Model needs permission to restore to user waypoints (user controls their checkpoints)
- User can always restore to any waypoint regardless of owner

### Tree Structure with Auto-Preservation

Waypoints form a tree, not a linear list:

```
w0 "session start" ◀ current
├── w1 "before auth refactor" [user]
│   └── w2 "pre-optimization" [model]
└── w3 "alternate approach" [model]
```

**Auto-save on restore**: When restoring to a previous waypoint with uncommitted file changes, a "ceiling" waypoint is automatically created. This ensures you never lose work when navigating the tree.

### What's Captured

Unlike pure conversation branching, waypoints capture:

- **File state** - All modified files tracked via backup system
- **Conversation metadata** - Message count, turn index, preview
- **Ownership** - Who created it and what permissions apply

**Implementation**: `shared/plugins/waypoint/` with `WaypointManager` tracking the tree structure and `BackupManager` integration for file state.

---

## Future Additions

Document new design decisions here as they emerge. Each entry should include:

1. **Principle** - The opinionated stance
2. **Rationale** - Why this approach over alternatives
3. **Implementation** - How it's realized in code
4. **Contrast** - What other systems do differently (if relevant)
