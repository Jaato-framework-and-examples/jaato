# Host-Provided (Client-Side) Tools — Complete Reference

> Scope: How WS clients register tools that the model can call, and how the daemon routes execution back to the client.

## Table of Contents

1. [What Are Host-Provided Tools?](#1-what-are-host-provided-tools)
2. [Architecture Overview](#2-architecture-overview)
3. [Registration Protocol](#3-registration-protocol)
4. [Execution Protocol](#4-execution-protocol)
5. [Server-Side Implementation](#5-server-side-implementation)
6. [Event Definitions](#6-event-definitions)
7. [Tool Definition Schema](#7-tool-definition-schema)
8. [Runtime Internals](#8-runtime-internals)
9. [Source Code Map](#9-source-code-map)

## 1. What Are Host-Provided Tools?

Host-provided tools (called **client-provided tools** in the source code) are tools whose
execution happens on the WebSocket client, not on the daemon server. The model sees them
as regular tools in its function-calling schema, but when it calls one, the daemon
forwards the call to the connected WS client and waits for a result.

This mechanism enables the frontend/browser to provide tools that require browser
capabilities (DOM manipulation, screen capture, user interaction) or client-side
resources that the server cannot access.

The key distinction: **plugin tools** execute server-side via Python callables.
**Client-provided tools** execute client-side via WS message round-trip.

## 2. Architecture Overview

```
Model                    Daemon (jaato-server)                   WS Client (browser)
  │                            │                                   │
  │  function_call(tool)       │                                   │
  ├───────────────────────────►│                                   │
  │                            │  ToolExecuteRequestEvent            │
  │                            ├──────────────────────────────────►│
  │                            │          (wait on thread)           │
  │                            │                                   │  execute()
  │                            │                                   │
  │                            │  ToolExecuteResultEvent             │
  │                            │◄──────────────────────────────────┤
  │  function_call_result       │                                   │
  │◄───────────────────────────┤                                   │
```

**Components involved:**

| Component | File | Role |
|-----------|------|------|
| Event definitions | `jaato_sdk/events.py` | `ToolsRegisterClientRequest`, `ToolExecuteRequestEvent`, `ToolExecuteResultEvent`, `ToolDisableRequest` |
| WS message dispatch | `server/websocket.py` | Routes `tools.register_client` and `tool.execute_result` events |
| Tool registration | `server/websocket.py` (`_register_client_tools`) | Creates proxy executors in the session registry |
| Tool schema | `jaato_sdk/plugins/model_provider/types.py` (`ToolSchema`) | Provider-agnostic tool declaration |
| Tool registry | `shared/plugins/registry.py` (`register_core_tool`) | Stores schemas and executors |
| Runtime refresh | `server/core.py` | Updates the model's tool list after registration |
| Tool ID registry | `server/core.py` (`_emit_tool_id_registry_from_schemas`) | Emits hash ID → name mappings to clients |

## 3. Registration Protocol

### 3.1 Client Sends Registration Request

The WS client sends a `tools.register_client` event containing tool definitions:

```json
{
  "type": "tools.register_client",
  "tools": [
    {
      "name": "browser_click",
      "description": "Click an element on the page",
      "parameters": {
        "type": "object",
        "properties": {
          "selector": {"type": "string", "description": "CSS selector"},
          "button": {"type": "string", "enum": ["left", "right", "middle"]}
        },
        "required": ["selector"]
      },
      "category": "browser",
      "timeout": 30000,
      "auto_approve": true
    }
  ],
  "categories": {
    "browser": "Browser automation tools provided by the web client"
  }
}
```

### 3.2 Server Processes Registration

When the server receives `ToolsRegisterClientRequest` (in `websocket.py`):

1. **If no session exists yet** (client connected but hasn't sent `session.new`):
   - Tools are buffered in `_pending_client_tools[client_id]`
   - When `session.new` is later processed, the buffered tools are registered
   - This handles the common case where the web client connects and registers
     tools before creating a session

2. **If session exists**:
   - Calls `_register_client_tools(client_id, tools, categories)` immediately

### 3.3 What _register_client_tools Does

For each tool definition in the request:

1. Creates a **proxy executor** — a Python callable that:
   - Generates a unique `call_id` (8-char UUID prefix)
   - Creates a `threading.Event` and result holder in `_client_tool_waiters[call_id]`
   - Sends `ToolExecuteRequestEvent` to all WS clients attached to the session
   - Waits for the client to respond (with timeout)
   - Returns `{"result": ...}` or `{"error": ...}`

2. Creates a `ToolSchema` with the tool's name, description, parameters, and category.
   The description is appended with ` [client-provided]` to distinguish it from
   server-side tools.

3. Calls `registry.register_core_tool(schema, executor, auto_approved=auto_approve)`
   to register the tool in the session's tool registry alongside server-side tools.

4. Registers any category descriptions via `registry.register_category()`.

5. Refreshes the runtime's `_all_tool_schemas` list so the model sees the new tools
   in its next request.

6. Emits an updated `tool_id_registry` event so clients can resolve hash-based
   tool IDs to tool names.

### 3.4 Timing: Before vs After Session Creation

The registration supports both ordering:

- **Tools before session.new** (common for web client): Tools are buffered, then
  registered when `session.new` arrives. The `session.new` handler in `_handle_message`
  checks `_pending_client_tools` and calls `_register_client_tools` after session creation.

- **Tools after session.new** (e.g., dynamically added tools): Tools are registered
  immediately into the existing session.

## 4. Execution Protocol

### 4.1 Model Calls a Client Tool

When the model generates a function call for a client-provided tool:

1. The model provider creates a `FunctionCall(name="browser_click", args={"selector": "#btn"})`
2. The daemon's tool execution pipeline resolves the tool name to its executor
3. The executor (the proxy created during registration) runs:
   - Generates `call_id`
   - Sends `ToolExecuteRequestEvent` to all clients attached to the session
   - Blocks the execution thread waiting for a response

### 4.2 Client Receives and Executes

The WS client receives:

```json
{
  "type": "tool.execute_request",
  "call_id": "a1b2c3d4",
  "agent_id": "",
  "tool_name": "browser_click",
  "tool_args": {"selector": "#btn"}
}
```

The client executes the tool using its own runtime (browser API, native code, etc.)
and sends back the result:

```json
{
  "type": "tool.execute_result",
  "call_id": "a1b2c3d4",
  "result": "{"clicked": true, "element": "button"}",
  "error": ""
}
```

### 4.3 Server Routes Result Back

`_handle_tool_execute_result(client_id, event)`:
1. Looks up `call_id` in `_client_tool_waiters`
2. Pops the `(waiter, result_holder)` entry
3. Sets `result_holder['result']` and `result_holder['error']`
4. Calls `waiter.set()` to unblock the executor thread
5. The executor returns `{"result": ...}` or `{"error": ...}` to the model pipeline

### 4.4 Timeout and Error Handling

- If no client is connected when the tool is called, the executor polls every
  1 second until a client connects or the timeout expires.
- If the timeout expires, returns `{"error": "Client tool X timed out after Ys"}`.
- If no client ever connects, returns `{"error": "No client connected to receive tool call X"}`.

## 5. Server-Side Implementation

### 5.1 Proxy Executor Creation

The proxy executor is created inside `_register_client_tools()` using a closure
factory (`make_executor`):

```python
def make_executor(tname, tout):
    def executor(args):
        call_id = str(uuid.uuid4())[:8]
        waiter = threading.Event()
        result_holder = {'result': None, 'error': None}
        ws_server._client_tool_waiters[call_id] = (waiter, result_holder)
        # Send ToolExecuteRequestEvent to session clients
        # ... (sends to all clients attached to the session)
        # Wait for result with remaining timeout
        remaining = max(0, deadline - time.time())
        if waiter.wait(timeout=remaining):
            if result_holder['error']:
                return {'error': result_holder['error']}
            return {'result': result_holder['result']}
        else:
            return {'error': f'Client tool {tname} timed out after {tout}s'}
    return executor
```

Key design decisions:
- **Per-call UUID** — avoids collisions between concurrent tool calls
- **Threading Event** — the model execution thread blocks; the WS event loop
  runs in a separate asyncio thread
- **Broadcast to all session clients** — not tied to the registering client_id.
  Any client attached to the session can respond. This means if the registering
  client disconnects and another connects, tool calls still work.
- **asyncio.run_coroutine_threadsafe** — bridges the threading (executor) and
  asyncio (WS server) worlds

### 5.2 register_core_tool Integration

Client tools are registered via `registry.register_core_tool()`, the same mechanism
used for framework tools like `dismiss_stream` or `shell_spawn`. This means:

- They appear in `get_exposed_tool_schemas()` alongside plugin tools
- They appear in `get_exposed_executors()` alongside plugin executors
- They respect the `auto_approved` flag for permission bypass
- They're included in the model's tool list on the next request

### 5.3 Tool ID Registry Refresh

After registration, the server emits an updated `tool_id_registry` event so
clients can resolve the hash-based tool IDs that appear in `ToolCallStartEvent` and
`ToolCallEndEvent` messages back to tool names. This is handled by:

```python
session.server._emit_tool_id_registry_from_schemas()
```

## 6. Event Definitions

### ToolsRegisterClientRequest

```python
@dataclass
class ToolsRegisterClientRequest(Event):
    type: EventType = field(default=EventType.TOOLS_REGISTER_CLIENT)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    categories: Dict[str, str] = field(default_factory=dict)
```

**Direction:** Client → Server

**Fields:**
- `tools`: List of tool definition dicts (see Section 7)
- `categories`: Optional mapping of category name → description for categories
  introduced by client tools

### ToolExecuteRequestEvent

```python
@dataclass
class ToolExecuteRequestEvent(Event):
    type: EventType = field(default=EventType.TOOL_EXECUTE_REQUEST)
    call_id: str = ""
    agent_id: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
```

**Direction:** Server → Client

**Fields:**
- `call_id`: Unique identifier for correlating request with response
- `agent_id`: The agent that initiated the tool call (empty for main agent)
- `tool_name`: Name of the tool to execute
- `tool_args`: Arguments to pass to the tool

### ToolExecuteResultEvent

```python
@dataclass
class ToolExecuteResultEvent(Event):
    type: EventType = field(default=EventType.TOOL_EXECUTE_RESULT)
    call_id: str = ""
    result: str = ""   # JSON-encoded result
    error: str = ""    # Error message if execution failed
```

**Direction:** Client → Server

**Fields:**
- `call_id`: Must match the `call_id` from the request
- `result`: JSON-encoded string of the tool's return value
- `error`: Error message if the tool failed (empty string for success)

### ToolDisableRequest

```python
@dataclass
class ToolDisableRequest(Event):
    type: EventType = field(default=EventType.TOOL_DISABLE_REQUEST)
    tool_name: str = ""
```

**Direction:** Client → Server

Directly calls `registry.disable_tool()` without generating response events.
Used by headless mode to disable tools before starting event handling.

### ToolCallStartEvent / ToolCallEndEvent

These events are emitted for ALL tool calls (both server-side and client-side)
as part of the tool tree rendering. They use hash-based tool IDs that clients
resolve using the `tool_id_registry`.

```python
@dataclass
class ToolCallStartEvent(Event):
    type: EventType = field(default=EventType.TOOL_CALL_START)
    tool_name: str = ""

@dataclass
class ToolCallEndEvent(Event):
    type: EventType = field(default=EventType.TOOL_CALL_END)
    tool_name: str = ""
```

## 7. Tool Definition Schema

Each tool in the `tools` array of `ToolsRegisterClientRequest` is a dict with these fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | — | Unique tool name. Must not collide with existing server/plugin tools. |
| `description` | `str` | No | `""` | Human-readable description. Appended with `[client-provided]` on the server. |
| `parameters` | `dict` | No | `{}` | JSON Schema object describing parameters (type: "object", properties, required). |
| `category` | `str` | No | `None` | Tool category for organization. Standard: filesystem, code, search, memory, planning, system, web, communication. Custom allowed. |
| `timeout` | `number` | No | `30000` | Execution timeout in milliseconds. Converted to seconds on the server. |
| `auto_approve` | `bool` | No | `true` | If true, tool calls bypass the permission prompt. |

### Parameters JSON Schema

The `parameters` field follows standard JSON Schema format:

```json
{
  "type": "object",
  "properties": {
    "selector": {
      "type": "string",
      "description": "CSS selector of the target element"
    },
    "action": {
      "type": "string",
      "enum": ["click", "type", "hover"],
      "description": "Action to perform"
    }
  },
  "required": ["selector"]
}
```

## 8. Runtime Internals

### 8.1 Threading Model

Client tool execution uses a **threading-based blocking model**:

- The model's tool execution happens on a thread (the model provider's thread pool)
- The proxy executor blocks that thread with `threading.Event.wait()`
- The WS server runs on the asyncio event loop
- `_send_to_session_clients()` uses `asyncio.run_coroutine_threadsafe()` to bridge
  from the executor thread to the event loop

This means:
- Multiple concurrent tool calls are possible (each gets its own thread + call_id)
- The executor thread is blocked but doesn't hold the GIL (if the tool does I/O)
- The WS event loop continues processing other events while waiting

### 8.2 Client Tool Waiters

`ws_server._client_tool_waiters` is a `Dict[str, Tuple[threading.Event, Dict]]`:

- Key: `call_id` (8-char UUID prefix)
- Value: `(waiter_event, result_holder_dict)`
- Entries are created when a tool call is initiated
- Entries are consumed (popped) when the result arrives
- Stale entries (from timed-out calls) are cleaned up by the pop

### 8.3 Session Attachment (Not Client Binding)

Tool calls are routed to **all clients attached to the session**, not just the
registering client. This is a deliberate design choice:

```python
# The executor sends the request to whichever clients are
# currently attached to the session (not a hardcoded client_id).
for cid, csid in ws_server._event_sink_adapter._client_sessions.items():
    if csid == sid:
        # send to this client
```

This means:
- If the registering client disconnects, tool calls still work when another client connects
- Multiple clients can handle tool calls (first to respond wins, per call_id)
- The `call_id` mechanism prevents duplicate responses

### 8.4 Runtime Schema Refresh

After registering client tools, the runtime's tool schema list must be updated:

```python
if session.server and session.server._jaato:
    runtime = session.server._jaato.get_runtime()
    if runtime and hasattr(runtime, '_all_tool_schemas'):
        existing = {s.name for s in runtime._all_tool_schemas}
        for name, schema in registry._core_tools.items():
            if name not in existing:
                runtime._all_tool_schemas.append(schema)
```

This ensures the model sees the new tools on its next API call. Without this refresh,
the model would not know the tools exist even though they're in the registry.

### 8.5 Category Registration

Client tools can define custom categories. The server registers these via
`registry.register_category(name, description)` so that `list_tools` shows
descriptions for categories introduced by client-side tools, not empty strings.

### 8.6 Tool Disable

The `ToolDisableRequest` event directly calls `registry.disable_tool(tool_name)`
without generating response events. This is used by headless mode to disable tools
before starting event handling — a lightweight way to prevent certain tools from
being available.

## 9. Source Code Map

| File | What It Contains |
|------|-----------------|
| `jaato_sdk/events.py` | `ToolsRegisterClientRequest`, `ToolExecuteRequestEvent`, `ToolExecuteResultEvent`, `ToolDisableRequest`, `ToolCallStartEvent`, `ToolCallEndEvent`, `ToolIdRegistryEvent`, `ToolStatusEvent` |
| `server/websocket.py` | `_register_client_tools()` (proxy creation, lines ~1395-1540), `_handle_tool_execute_result()` (result routing), `ToolsRegisterClientRequest` dispatch (lines ~1245-1260), pending tool buffering |
| `server/core.py` | `_build_tool_id_mappings()`, `_emit_tool_id_registry_from_schemas()`, `_current_tool_agent_id` tracking |
| `jaato_sdk/plugins/model_provider/types.py` | `ToolSchema` dataclass (line 161), `FunctionCall` dataclass |
| `shared/plugins/registry.py` | `register_core_tool()` (line 751), `register_category()` (line 668), `disable_tool()`, `_core_tools`, `_core_executors`, `_core_auto_approved` |
| `shared/jaato_session.py` | Session-level tool registration (lines ~1349-1362), session.new handling with pending tools |
