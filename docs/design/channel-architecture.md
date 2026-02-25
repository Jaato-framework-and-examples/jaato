# Channel Architecture: Agent-to-User vs Agent-to-Agent Communication

The framework uses a **channel abstraction** to handle both user-facing and inter-agent communication at the plugin level. While the plugin API remains the same in both cases, the underlying mechanics differ substantially.

## Overview

Two plugins use the channel pattern for interactive communication:

- **Permission plugin** (`shared/plugins/permission/`) — intercepts tool execution and requests approval
- **Clarification plugin** (`shared/plugins/clarification/`) — lets the model ask the user structured questions

Both plugins are **singletons** shared across all sessions. Thread-local storage isolates per-session channel state so subagent threads don't interfere with the main agent.

Other plugins (todo, references) have their own channel-like abstractions (reporters, selection channels) but don't follow the same thread-local/ParentBridgedChannel pattern described here.

## The Channel Pattern

### Permission Channels

| Scenario | Channel Type | Source |
|----------|-------------|--------|
| User (console) | `ConsoleChannel` | `permission/channels.py:359` |
| User (TUI / server) | `QueueChannel` | `permission/channels.py:910` |
| Subagent → Parent | `ParentBridgedChannel` | `permission/channels.py:1140` |
| External system | `WebhookChannel` | `permission/channels.py:698` |
| File-based automation | `FileChannel` | `permission/channels.py:801` |

### Clarification Channels

| Scenario | Channel Type | Source |
|----------|-------------|--------|
| User (console) | `ConsoleChannel` | `clarification/channels.py:59` |
| User (TUI / server) | `QueueChannel` | `clarification/channels.py:363` |
| Subagent → Parent | `ParentBridgedChannel` | `clarification/channels.py:576` |
| Automated / test | `AutoChannel` | `clarification/channels.py:313` |

Note: The permission plugin has no `AutoChannel` equivalent. The clarification `AutoChannel` automatically selects defaults or first available choices — useful for non-interactive and test scenarios.

### Thread-Local Channel Isolation

Both plugins use identical thread-local storage patterns (`permission/plugin.py:61`, `clarification/plugin.py:39`):

```python
# Both permission/plugin.py and clarification/plugin.py
_thread_local = threading.local()

def _get_channel(self):
    thread_channel = getattr(self._thread_local, 'channel', None)
    return thread_channel if thread_channel is not None else self._channel
```

- `self._channel` — default channel for the main agent (set during `initialize()` or `set_channel()`)
- `self._thread_local.channel` — per-thread override (set during `configure_for_subagent()`)

Since each subagent runs in its own thread, this ensures channel isolation without requiring separate plugin instances.

## Agent-to-User Flow

This is the direct human-in-the-loop path, using either `ConsoleChannel` (plain terminal) or `QueueChannel` (TUI/server).

### Permission Flow

1. Model calls a tool (e.g., `cli_based_tool`)
2. `PermissionPlugin.check_permission()` evaluates the request against policy (`permission/plugin.py:897`)
3. If policy decision is `ASK_CHANNEL`, the channel is invoked
4. **ConsoleChannel**: Formats the request with ANSI colors, reads from stdin via `input()`
5. **QueueChannel**: Emits output via callback, waits on a `queue.Queue` that the TUI input handler feeds into
6. User sees a formatted prompt and responds (y, n, always, turn, idle, once, never, all, comment, edit)
7. Channel returns a `ChannelResponse` with the decision
8. Plugin handles the response: whitelist/blacklist updates, suspension state changes, edit loops

### Clarification Flow

1. Model calls `request_clarification` tool
2. `ClarificationPlugin._execute_clarification()` parses the request (`clarification/plugin.py:336`)
3. Channel presents questions one at a time
4. **ConsoleChannel**: Renders questions with ANSI formatting, reads from stdin
5. **QueueChannel**: Emits each question via output callback, waits on the TUI input queue
6. User answers each question (free text, single choice, or multiple choice)
7. Channel returns a `ClarificationResponse` with all answers

### Key Characteristics

- **Blocking on real user input** — stdin or TUI queue
- **UI lifecycle hooks fire** — `on_permission_requested`, `on_permission_resolved`, `on_clarification_requested`, `on_clarification_resolved`, `on_question_displayed`, `on_question_answered`
- **Events emitted to clients** — `PermissionRequestedEvent`, `ClarificationInputModeEvent`, etc.
- **User has full control** — can cancel, skip optional questions, choose scoped approvals (`always`, `never`, `turn`, `idle`)
- **Cancellation support** — `QueueChannel` checks a `CancelToken` every 100ms during input polling

## Agent-to-Agent Flow (ParentBridgedChannel)

When a subagent needs permission or clarification, it cannot talk to the user directly. Requests are bridged through the parent agent's session.

### Setup

During subagent creation, both plugins are configured for the subagent's thread (`permission/plugin.py:1436`, `clarification/plugin.py:524`):

```python
def configure_for_subagent(self, session):
    from .channels import ParentBridgedChannel
    channel = ParentBridgedChannel()
    channel.set_session(session)
    # Store in thread-local storage, not the shared instance
    self._thread_local.channel = channel
```

### Communication Flow

```
Subagent thread                          Parent session
───────────────                          ──────────────
Model calls tool
  │
  ▼
Plugin intercepts
  │
  ▼
_get_channel() → ParentBridgedChannel
  │
  ▼
_format_request_for_parent()
  → XML with request_id:
    <permission_request request_id="abc123">
      <tool_name>cli_based_tool</tool_name>
      <arguments>
        <arg name="command">git status</arg>
      </arguments>
      <options>
        <option short="y" full="yes">Allow this tool execution</option>
        ...
      </options>
    </permission_request>
  │
  ▼
_forward_to_parent() on session ─────────→  Parent receives as CHILD message
  (SourceType.CHILD)                        via inject_prompt()
  │                                         │
  ▼                                         ▼
_wait_for_response(request_id)              Parent decides (or forwards to user)
  polls _injection_queue             ◄───── Response injected back
  matches by request_id                     via inject_prompt()
  │
  ▼
_parse_response_from_parent()
  → ChannelResponse (ALLOW/DENY)
  │
  ▼
Tool executes (or denied)
```

### Message Format

**Permission requests** use XML with `request_id` correlation (`permission/channels.py:1175`):

```xml
<permission_request request_id="abc123">
  <tool_name>cli_based_tool</tool_name>
  <arguments>
    <arg name="command">git status</arg>
  </arguments>
  <options>
    <option short="y" full="yes">Allow this tool execution</option>
    <option short="n" full="no">Deny this tool execution</option>
    <!-- ... -->
  </options>
</permission_request>
```

**Clarification requests** use a similar XML structure (`clarification/channels.py:608`):

```xml
<clarification_request request_id="a1b2c3d4">
  <context>Need deployment preferences</context>
  <question index="1" type="single_choice" required="true">
    <text>Which environment?</text>
    <choices>
      <choice index="1">Development</choice>
      <choice index="2" default="true">Staging</choice>
    </choices>
  </question>
</clarification_request>
```

### Response Matching

The `_wait_for_response()` method polls the session's `_injection_queue` and matches responses by `request_id` (`permission/channels.py:1246`, `clarification/channels.py:722`):

1. Dequeue a message from `_injection_queue` with a 100ms timeout
2. Check if the message contains the target `request_id`
3. If yes: put held messages back, return the response
4. If no: hold the message and continue polling
5. On timeout or cancellation: put all held messages back, return `None`

For permission, simple responses (`y`, `yes`, `n`, `no`, `a`, `always`, etc.) are also accepted when there's a single pending request — no XML wrapping required (`permission/channels.py:1311`).

### Key Characteristics

- **No direct user access** — everything goes through the parent session
- **Structured XML format** with `request_id` for request-response correlation
- **Injection queue** (`_injection_queue`) is the transport — the same queue used for all parent-child communication
- **UI hooks are suppressed** — `is_subagent_mode` check skips firing hooks that would confuse the parent's UI (`permission/plugin.py:946`, `clarification/plugin.py:358`)
- **Timeout defaults to 300s** — configurable via `JAATO_PERMISSION_TIMEOUT` / `JAATO_CLARIFICATION_TIMEOUT` environment variables; 0 or negative means wait forever
- **Non-matching messages are held and re-queued** — the subagent may receive messages not meant for it
- **Cancellation-aware** — checks `_cancel_token` during polling

### Forwarding Mechanism

The `_forward_to_parent()` method on `JaatoSession` (`jaato_session.py:626`) wraps the message with subagent metadata and injects it into the parent:

```python
def _forward_to_parent(self, event_type, content):
    # Skip verbose progress events (MODEL_OUTPUT, TOOL_CALL, TOOL_OUTPUT)
    if event_type in ("MODEL_OUTPUT", "TOOL_CALL", "TOOL_OUTPUT"):
        return

    message = f"[SUBAGENT agent_id={self._agent_id} event={event_type}]\n{content}"
    self._parent_session.inject_prompt(
        message,
        source_id=self._agent_id,
        source_type=SourceType.CHILD
    )
```

If `_forward_to_parent` is not available, `ParentBridgedChannel` falls back to direct injection into the parent session.

## Summary of Key Differences

| Aspect | Agent-to-User | Agent-to-Agent (Subagent) |
|--------|--------------|--------------------------|
| **Channel** | `ConsoleChannel` or `QueueChannel` | `ParentBridgedChannel` |
| **Channel storage** | `self._channel` (shared) | `self._thread_local.channel` (per-thread) |
| **Transport** | stdin / TUI input queue | Session injection queue (`_injection_queue`) |
| **Message format** | Human-readable text prompts | Structured XML with `request_id` |
| **Response matching** | Implicit (one pending at a time) | Explicit `request_id` correlation |
| **UI hooks fire?** | Yes (events emitted to client) | No (suppressed via `is_subagent_mode`) |
| **Timeout behavior** | Configurable, defaults to deny/cancel | 300s default, configurable, defaults to deny/cancel |
| **Fallback on parse failure** | Re-prompt the user | Deny (permission) / Cancel (clarification) |
| **Who decides?** | Human directly | Parent agent (which may itself ask the human) |
| **Message source type** | `SourceType.USER` | `SourceType.CHILD` / `SourceType.PARENT` |

## The Layered Trust Model

This creates a delegation chain:

```
User  ◄──►  Main Agent  ◄──►  Subagent
            (QueueChannel)     (ParentBridgedChannel)
```

The main agent acts as a **proxy**: it receives the subagent's permission/clarification requests and can either:

1. **Forward to the user** — the typical case, showing the prompt in the UI
2. **Decide autonomously** — if the parent agent has enough context to approve/deny

The subagent never directly interacts with the user. Unrecognized responses always default to the safe choice (deny for permissions, cancel for clarifications). This keeps the security boundary clean while allowing the parent to mediate all subagent interactions.

## Thread-Local Channels and Parallel Tool Execution

### The Problem

The thread-local isolation pattern works correctly when tools execute **sequentially** on the subagent's own thread — the thread where `configure_for_subagent()` set `_thread_local.channel`. However, when the subagent's model returns **2+ tool calls in a single turn**, `_execute_function_calls_parallel()` creates a `ThreadPoolExecutor` with fresh worker threads (`jaato_session.py:3654`).

These worker threads **do not inherit** the spawning thread's `threading.local()` values. When a tool on a worker thread triggers a permission or clarification check:

1. `_get_channel()` finds no `_thread_local.channel` on the worker thread
2. Falls back to `self._channel` — the **main agent's** channel (e.g. `QueueChannel`)
3. `isinstance(channel, ParentBridgedChannel)` → `False`
4. `is_subagent_mode` is `False` → UI hooks fire
5. `PermissionRequestedEvent` / `PermissionInputModeEvent` emitted to the TUI
6. The TUI captures the input field, forcing the user to answer a prompt that should have been routed to the parent agent

```
Subagent thread (spawning)              Worker thread (pool)
──────────────────────────              ────────────────────
_thread_local.channel = ParentBridged   _thread_local.channel = (empty)
                                         │
configure_for_subagent() ✓               _get_channel() → self._channel
                                         │                  (main agent's QueueChannel!)
                                         ▼
                                        is_subagent_mode = False  ← BUG
                                        hooks fire → TUI input captured
```

The existing propagation loop in `_execute_single_tool_for_parallel` (lines 3974-3978) only covers **exposed tool plugins** via `list_exposed()` and `set_session()`. The permission plugin is **not** in the exposed list (it's a special singleton on `JaatoRuntime._permission_plugin`), and neither plugin has a `set_session()` method — so neither is reached by this loop.

### The Fix: Capture and Restore

The fix snapshots the spawning thread's channel references before creating the thread pool, then restores them on each worker thread:

```python
# In _execute_function_calls_parallel (runs on subagent's thread):
captured_channels = self._capture_interactive_channels()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_fc = {
        executor.submit(
            self._execute_single_tool_for_parallel, fc, captured_channels
        ): fc
        for fc in function_calls
    }

# In _execute_single_tool_for_parallel (runs on worker thread):
if captured_channels:
    self._restore_interactive_channels(captured_channels)
```

`_capture_interactive_channels()` reads the current thread's channel from:
- `self._runtime.permission_plugin._get_channel()` — permission plugin (singleton on runtime)
- `self._runtime.registry.get_plugin('clarification')._get_channel()` — clarification plugin (in registry)

`_restore_interactive_channels()` writes those references into the worker thread's `_thread_local.channel` on both plugins.

After the fix, worker threads see the same `ParentBridgedChannel` that the spawning thread had, so `is_subagent_mode` evaluates correctly and hooks are suppressed.

### Why Not Store the Channel on the Session?

An alternative design would store the channel directly on `JaatoSession` and have `_get_channel()` check the session as a fallback. This was considered but rejected because:

1. **The plugins don't hold session references** — the permission plugin is a singleton that doesn't receive `set_session()`, so it would need a new mechanism to look up the "current session"
2. **Thread-local is the right abstraction** — the channel must vary per-thread (main agent thread vs subagent thread vs worker thread), which is exactly what `threading.local()` provides
3. **Capture/restore is minimal and explicit** — it follows the existing pattern (`set_session()` propagation loop) and makes the propagation visible in the parallel execution code

## Source File Reference

| File | Contents |
|------|----------|
| `shared/plugins/permission/channels.py` | Permission channel hierarchy (Console, Queue, Webhook, File, ParentBridged) |
| `shared/plugins/permission/plugin.py` | Permission plugin with thread-local pattern, `is_subagent_mode` checks |
| `shared/plugins/clarification/channels.py` | Clarification channel hierarchy (Console, Queue, Auto, ParentBridged) |
| `shared/plugins/clarification/plugin.py` | Clarification plugin with thread-local pattern, `is_subagent_mode` checks |
| `shared/jaato_session.py` | `inject_prompt()`, `_forward_to_parent()`, `_injection_queue`, `_capture_interactive_channels()`, `_restore_interactive_channels()` |
| `shared/message_queue.py` | `SourceType` enum (PARENT, CHILD, USER, SYSTEM) |
