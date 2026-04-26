# jaato-sdk

Python client SDK for connecting to a [jaato](https://github.com/Jaato-framework-and-examples/jaato) server. Provides the wire protocol, async IPC client, and an auto-reconnecting recovery client.

## Installation

```bash
pip install --extra-index-url https://test.pypi.org/simple/ jaato-sdk
```

## Quick Start

```python
import asyncio
from jaato_sdk import IPCRecoveryClient
from jaato_sdk.events import AgentOutputEvent, ToolCallStartEvent

async def main():
    client = IPCRecoveryClient()  # default: /tmp/jaato.sock (Windows: \\.\pipe\jaato)
    await client.connect()
    await client.create_session()
    await client.send_message("Hello!")

    async for event in client.events():
        if isinstance(event, AgentOutputEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallStartEvent):
            print(f"\n[tool: {event.tool_name}]")

asyncio.run(main())
```

## Core Concepts

### Server-first architecture

Unlike a local agent loop, the agent itself runs in a separate **jaato server** process (a daemon). The SDK is purely a transport layer — it ships JSON-encoded events over a Unix domain socket (or Windows named pipe) and yields them back to your code as Python dataclasses.

```
your code  ──►  IPCRecoveryClient  ──►  /tmp/jaato.sock  ──►  jaato server (agent loop)
                                                                    │
your code  ◄──  IPCRecoveryClient  ◄──  /tmp/jaato.sock  ◄──────────┘
                       (events)
```

If no server is running and `auto_start=True` (the default), the client launches `python -m server --daemon` for you.

### Two clients

| Client | Use when |
|---|---|
| `IPCClient` | You want a thin, transparent connection. No retries — if the server goes away, your iterator ends. |
| `IPCRecoveryClient` | You want automatic reconnection with exponential backoff and session reattachment. Recommended for most apps. |

`IPCRecoveryClient` wraps `IPCClient` with a state machine and a configurable retry policy. It exposes the same request methods plus connection-lifecycle hooks.

### Events vs requests

Everything on the wire is an `Event` dataclass.

- **Server → Client events** describe what the agent is doing: `AgentOutputEvent`, `ToolCallStartEvent`, `PermissionRequestedEvent`, `PlanUpdatedEvent`, `TurnCompletedEvent`, `ErrorEvent`, …
- **Client → Server requests** are the same `Event` shape but flow the other way: `SendMessageRequest`, `PermissionResponseRequest`, `StopRequest`, `CommandRequest`, …

You never construct request events directly in normal usage — the client provides typed methods like `send_message()`, `respond_to_permission()`, `stop()`. Construct the request dataclasses only when you need to send something the convenience methods don't cover (use `client.execute_command()` for that).

## Event Flow

### connect() sequence

```
client.connect()
├─ open socket / pipe
├─ wait for ConnectedEvent           # carries client_id + server_version
├─ send CommandRequest(set_workspace) # client cwd
└─ send ClientConfigRequest          # env file + PresentationContext
```

After `connect()` returns `True`, the server has accepted the connection but no session is attached yet. Either call `create_session()` to spawn a new one or `attach_session(id)` to resume an existing one.

### send_message() sequence

```
client.send_message("Read config.json")
├─ SendMessageRequest                  # → server
│
├─ AgentOutputEvent     {source: "model", text: "I'll read..."}
├─ AgentOutputEvent     {source: "model", text: " the file."}
├─ ToolCallStartEvent   {tool_name: "read", tool_args: {...}, call_id: "..."}
├─ ToolOutputEvent      {chunk: "..."}                      # if the tool streams
├─ ToolCallEndEvent     {call_id: "...", success: true}
├─ AgentOutputEvent     {source: "model", text: "The file..."}
└─ TurnCompletedEvent   {token_usage: {...}}
```

`AgentOutputEvent.mode` is `"write"` for a new block of output and `"append"` for streaming continuation chunks. `source` is one of `"model"`, `"tool"`, `"system"`, or a plugin name.

### Permission flow

When a tool needs approval, the server pauses and emits a permission request. The client responds with one of the offered keys.

```
ToolCallStartEvent
PermissionRequestedEvent   { request_id, tool_name, tool_args, response_options, prompt_lines }
PermissionInputModeEvent   { request_id }                  # signal: take input now
        │
        │ client.respond_to_permission(request_id, "y")    # → server
        ▼
PermissionResolvedEvent    { request_id, response, granted }
ToolCallEndEvent           { ... }                          # tool runs
```

Permission response keys (returned in `response_options`):

| Key | Meaning |
|-----|---------|
| `y` | yes, this once |
| `n` | no |
| `a` | always allow this tool |
| `t` | allow for the rest of this turn |
| `i` | allow until the session goes idle |
| `never` | blacklist this tool |
| `e` | edit the arguments and re-prompt (pass `edited_arguments=...`) |

### Cancellation

`await client.stop()` sends a `StopRequest`. The server cancels in-flight tool calls and the streaming model call; expect to see an `AgentStatusChangedEvent(status="error")` or a `TurnCompletedEvent` with cancellation metadata, then the iterator continues normally.

## Client Options

```python
client = IPCRecoveryClient(
    socket_path="/tmp/jaato.sock",   # Unix socket or Windows pipe name
    config=RecoveryConfig(...),       # see "Auto-reconnection" below
    auto_start=True,                  # spawn server daemon if not running
    env_file=".env",                  # client env forwarded to server (relative to workspace)
    workspace_path=Path.cwd(),        # what the server sees as the working directory
    on_status_change=lambda s: ...,   # ConnectionStatus callback
)
```

`IPCClient` takes the same parameters minus `config` and `on_status_change`.

## Client State

```python
client.is_connected         # bool
client.is_reconnecting      # bool (recovery client only)
client.is_closed            # bool (recovery client only)
client.state                # ConnectionState enum (recovery client only)
client.session_id           # currently attached session, or None
client.client_id            # assigned by the server on connect
client.server_version       # server package version, or None on pre-0.2.28 servers
client.get_status()         # → ConnectionStatus dataclass (recovery client only)
```

`ConnectionState` values: `DISCONNECTED`, `CONNECTING`, `CONNECTED`, `RECONNECTING`, `DISCONNECTING`, `CLOSED`.

The recovery client refuses to send while reconnecting and raises `ReconnectingError`. Once `state == CLOSED` (max attempts exceeded, or `close()` called), it raises `ConnectionClosedError` and cannot be revived — construct a new instance.

## Methods

### Lifecycle

```python
await client.connect(timeout=5.0)
await client.disconnect()       # graceful, can be reconnected
await client.close()            # permanent (recovery client only)
```

### Sessions

```python
await client.create_session(
    name="my-session",
    profile="researcher",       # name of a profile under .jaato/profiles/
    agent="reviewer",           # agent name; its rendered markdown becomes system instructions
    agent_params={"focus": "security"},
)
await client.attach_session(session_id)
await client.get_default_session()
await client.list_sessions()           # response arrives as SessionListEvent
await client.list_profiles()           # response arrives as SessionProfilesEvent
```

`create_session()` returns the new session id when no event iterator is active; otherwise it is fire-and-forget and the id arrives via the event stream as a `SessionInfoEvent`.

### Messages and replies

```python
await client.send_message("Build the README", attachments=[...])
await client.respond_to_permission(request_id, "y")
await client.respond_to_permission(request_id, "e", edited_arguments={"path": "..."})
await client.respond_to_clarification(request_id, "use json")
await client.respond_to_reference_selection(request_id, "1,3,4")
await client.stop()
```

### Commands and metadata

```python
await client.execute_command("model", ["claude-sonnet-4-5"])
await client.request_command_list()    # response: CommandListEvent
await client.request_history()         # response: HistoryEvent
await client.disable_tool("bash")
```

### Event stream

```python
async for event in client.events():
    handle(event)
```

The iterator exits cleanly on disconnect. With `IPCRecoveryClient`, it survives reconnects: events from the new connection are yielded transparently after the gap.

For a callback-style API:

```python
client.set_event_callback(handle)
await client.receive_events()
```

## Auto-reconnection

`IPCRecoveryClient` retries with exponential backoff plus jitter and reattaches to the previous session on success.

```python
from jaato_sdk import RecoveryConfig

config = RecoveryConfig(
    enabled=True,
    max_attempts=10,
    base_delay=1.0,         # seconds
    max_delay=60.0,
    jitter_factor=0.3,      # ±30% random jitter
    connection_timeout=5.0,
    reattach_session=True,  # call attach_session() with the previous id after reconnect
)
```

Status callback:

```python
from jaato_sdk import ConnectionState

def on_status(status):
    if status.state == ConnectionState.RECONNECTING:
        print(f"reconnecting {status.attempt}/{status.max_attempts} "
              f"in {status.next_retry_in:.1f}s ({status.last_error})")

client = IPCRecoveryClient(on_status_change=on_status)
```

The recovery loop classifies errors as **transient** (retried — `ConnectionRefusedError`, `ConnectionResetError`, timeouts) or **permanent** (not retried — `IncompatibleServerError`, `FileNotFoundError`, permission/auth failures). Permanent errors transition straight to `CLOSED`.

### Server version mismatch

Each client can pin a minimum server version. If the connected server is older, `connect()` raises `IncompatibleServerError` (which the recovery client classifies as permanent — no retries). Catch it and prompt the user to upgrade.

## Configuration

The recovery client picks up settings from these places, highest precedence first:

1. Environment variables
2. `<workspace>/.jaato/client.json`
3. `~/.jaato/client.json`
4. Built-in defaults

```python
from jaato_sdk.client import load_client_config, get_recovery_config

config = load_client_config(workspace_path=Path.cwd())
recovery = get_recovery_config(workspace_path=Path.cwd())
```

| Environment variable | RecoveryConfig field |
|---|---|
| `JAATO_IPC_AUTO_RECONNECT` | `enabled` |
| `JAATO_IPC_RETRY_MAX_ATTEMPTS` | `max_attempts` |
| `JAATO_IPC_RETRY_BASE_DELAY` | `base_delay` |
| `JAATO_IPC_RETRY_MAX_DELAY` | `max_delay` |
| `JAATO_IPC_RETRY_JITTER` | `jitter_factor` |
| `JAATO_IPC_CONNECTION_TIMEOUT` | `connection_timeout` |
| `JAATO_IPC_REATTACH_SESSION` | `reattach_session` |

## Presentation Context

Every client tells the server what its display surface looks like so the agent can adapt its output (avoid wide tables on a phone, skip mermaid on a TUI, etc.). The default is a generic terminal — override it before `connect()` if you're building a different kind of client.

```python
from jaato_sdk import PresentationContext, ClientType, CommunicationStyle

presentation = PresentationContext(
    content_width=72,
    client_type=ClientType.CHAT,             # TERMINAL | WEB | CHAT | API
    supports_tables=False,
    supports_expandable_content=True,        # client wraps overflow itself
    communication_style=CommunicationStyle.CONVERSATIONAL,
)
```

`ClientType` describes the kind of surface, not a specific app. Telegram, Slack and WhatsApp bots are all `CHAT`. `CommunicationStyle.CONVERSATIONAL` tells the model to send short, frequent updates; `NARRATIVE` tells it to deliver one well-structured response at the end. When `communication_style` is left `None`, `CHAT` defaults to conversational and everything else to narrative.

## Building a Custom Client

Everything you need to drive the server yourself:

```python
from jaato_sdk import IPCClient
from jaato_sdk.events import (
    # Server → Client
    ConnectedEvent,
    AgentOutputEvent,
    ToolCallStartEvent, ToolCallEndEvent, ToolOutputEvent,
    PermissionRequestedEvent, PermissionInputModeEvent, PermissionResolvedEvent,
    ClarificationRequestedEvent, ReferenceSelectionRequestedEvent,
    PlanUpdatedEvent, PlanStepUpdatedEvent, PlanClearedEvent,
    ContextUpdatedEvent, TurnCompletedEvent, TurnProgressEvent,
    SystemMessageEvent, ErrorEvent, RetryEvent, InitProgressEvent,
    SessionInfoEvent, SessionListEvent, SessionProfilesEvent,

    # Client → Server
    SendMessageRequest, PermissionResponseRequest, ClarificationResponseRequest,
    ReferenceSelectionResponseRequest, StopRequest, CommandRequest,
    HistoryRequest, ClientConfigRequest, ToolDisableRequest,
)
```

A reference TUI implementation lives at [`jaato-tui`](../jaato-tui/) in the same repo.

## Low-Level API

Every event is a serializable dataclass. If you need to bypass the client (for example, embedding the protocol in a different transport), you can drive serialization directly:

```python
from jaato_sdk.events import serialize_event, deserialize_event, SendMessageRequest

wire = serialize_event(SendMessageRequest(text="hi"))   # → JSON string
event = deserialize_event(wire)                          # → typed dataclass
```

Frames on the socket are a 4-byte big-endian length prefix followed by the JSON payload. Max message size is 10 MiB. The Unix-socket variant uses `asyncio.open_unix_connection`; on Windows the SDK uses `loop.create_pipe_connection` against `\\.\pipe\<name>`.

## Tracing

The SDK ships a small tracing helper that the server picks up via `JAATO_TRACE_LOG` and `PROVIDER_TRACE_LOG`:

```python
from jaato_sdk import trace, provider_trace, trace_write, resolve_trace_path
```

These write JSONL records to per-agent files under the configured trace directory. Useful for offline replay and debugging — leave them off in production unless you need them.

## Requirements

- Python 3.10+
- A reachable jaato server (auto-started by default)
- `python-dotenv` (the only runtime dependency)

## License

BUSL-1.1
