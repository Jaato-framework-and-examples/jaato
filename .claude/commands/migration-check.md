# Migration Check Command

Analyze a branch for compatibility with the server-first architecture introduced in the refactor.

## Usage

```
/migration-check [branch-name]
```

If no branch is provided, analyze uncommitted changes and recent commits on the current branch.

## Instructions

When this command is invoked, perform the following analysis:

### 1. Identify Changed Files

First, determine what files to analyze:
- If a branch name is provided: `git diff main...<branch> --name-only` (or compare against the branch)
- If no branch: Check `git status` for modified files and `git log --oneline -20` for recent commits

Focus on files in:
- `jaato-tui/`
- `shared/`
- Any new client implementations

### 2. Check for Legacy Callback Patterns

Search for these callback patterns that should now use server events:

| Legacy Pattern | Should Use Event |
|----------------|------------------|
| `on_output=` callback | `AgentOutputEvent` |
| `on_permission_requested=` | `PermissionRequestedEvent` |
| `on_tool_start=` | `ToolExecutionStartedEvent` |
| `on_tool_end=` | `ToolExecutionCompletedEvent` |
| `on_plan_updated=` | `PlanUpdatedEvent` |
| `on_gc_threshold=` | `GCThresholdEvent` |
| `set_retry_callback()` | `RetryEvent` |
| `on_agent_output=` | `AgentOutputEvent` |
| `on_status_update=` | `StatusUpdateEvent` |

Report each occurrence with file path and line number.

### 3. Check for Direct Client Usage in UI Code

Look for patterns that bypass the server architecture:

- `JaatoClient()` instantiation in `jaato-tui/` files (should use `IPCClient` or `JaatoServer`)
- `from shared.jaato_client import` in client code
- Direct `jaato.send_message()` calls in UI code (should send via IPC/events)

### 4. Check for Deprecated CLI Flags

Search for references to removed/renamed flags:

| Deprecated | Replaced By |
|------------|-------------|
| `--headless` | `--ipc-socket` (on server) |
| `--expose-server` | `--web-socket` (on server) |
| `--port` (on client) | `--connect` |
| `--host` (on client) | `--connect` |

### 5. Check for Missing Event Handlers

If the branch adds new functionality, verify it handles relevant events from `server/events.py`:

- `PermissionRequestedEvent` - must respond with `PermissionResponseRequest`
- `AgentOutputEvent` - display to user
- `ToolExecutionStartedEvent` / `ToolExecutionCompletedEvent` - tool status
- `ErrorEvent` - error display
- `SessionListEvent` - session management

### 6. Check for Synchronous Blocking Patterns

Look for patterns that should be async in client code:

- `while True:` polling loops that should use async event streams
- `time.sleep()` in UI code
- Blocking `input()` calls mixed with server communication

### 7. Report Format

Generate a structured report:

```
## Migration Check Report

### Summary
- Files analyzed: N
- Issues found: N
- Severity: [High/Medium/Low]

### Legacy Callbacks (High Priority)
These callback patterns should be migrated to event handlers:

1. `jaato-tui/foo.py:123` - `on_output=` callback
   → Migrate to handle `AgentOutputEvent` from event stream

### Direct Client Usage (Medium Priority)
...

### Deprecated Flags (Low Priority)
...

### Recommendations
1. ...
2. ...
```

### 8. Additional Checks

Also look for:

- **Thread-based concurrency** in client code that should use async/await
- **Direct socket creation** (`socket.socket()`) in client code
- **Hardcoded paths** like `/tmp/jaato.sock` that should use config
- **Missing error handling** for `ConnectionRefusedError`, `BrokenPipeError`
- **State stored in client** that should be server-side (session state, history)

## Context

The server-first architecture (introduced in commits after `914bd84`) changes how clients interact with the framework:

**Before (embedded):**
```python
client = JaatoClient()
client.connect(...)
response = client.send_message("Hello", on_output=my_callback)
```

**After (server-first):**
```python
ipc = IPCClient("/tmp/jaato.sock")
await ipc.connect()
await ipc.send_message("Hello")
async for event in ipc.events():
    if isinstance(event, AgentOutputEvent):
        handle_output(event)
```

Key files for reference:
- `server/events.py` - All event types
- `server/core.py` - JaatoServer (replaces direct JaatoClient in server context)
- `jaato-tui/ipc_client.py` - Client connection pattern
