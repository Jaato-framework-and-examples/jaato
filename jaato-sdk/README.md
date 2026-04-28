# jaato-sdk

Python client SDK for connecting to a [jaato](https://github.com/Jaato-framework-and-examples/jaato) server. Provides the wire protocol, async IPC client, and an auto-reconnecting recovery client.

## Installation

```bash
pip install --extra-index-url https://test.pypi.org/simple/ jaato-sdk
```

## Quick Start

```python
import asyncio
from jaato_sdk import IPCRecoveryClient, EventType

async def main():
    client = IPCRecoveryClient()  # default: /tmp/jaato.sock (Windows: \\.\pipe\jaato)

    # Typed event handlers — register before connect() to capture
    # the inaugural ConnectedEvent.
    client.subscribe(EventType.AGENT_OUTPUT, lambda e: print(e.text, end=""))
    client.subscribe(EventType.TOOL_CALL_START, lambda e: print(f"\n[tool: {e.tool_name}]"))

    await client.connect()
    await client.create_session()
    await client.send_message("Hello!")

    # Drive the event loop so the dispatcher fires.  Either iterate
    # client.events() (legacy style) or await client.drain_events()
    # to let your subscribers do the work.
    await client.drain_events()

asyncio.run(main())
```

## Core Concepts

### Server-first architecture

Unlike a local agent loop, the agent itself runs in a separate **jaato server** process (a daemon). The SDK is purely a transport layer — it ships JSON-encoded events to the server and yields them back to your code as Python dataclasses.

```
your code  ──►  IPCRecoveryClient  ──►  /tmp/jaato.sock  ──►  jaato server (agent loop)
                                                                    │
your code  ◄──  IPCRecoveryClient  ◄──  /tmp/jaato.sock  ◄──────────┘
                       (events)
```

If no server is running and `auto_start=True` (the default), the client launches `python -m server --daemon` for you.

### Transports

The server speaks the same wire protocol over two transports:

| Transport | Endpoint | What's shipped in this SDK |
|---|---|---|
| **IPC** (Unix domain socket / Windows named pipe) | `/tmp/jaato.sock` or `\\.\pipe\jaato` | `IPCClient` and `IPCRecoveryClient` — Python async clients with auto-start, framing, and reconnection. |
| **WebSocket** | `ws://host:port` (or `wss://`) | Just the protocol — `Event` dataclasses, `serialize_event` / `deserialize_event`. No Python WS client. |

The IPC client is for processes living on the same machine as the daemon (TUI, scripts, in-process tooling). The WebSocket transport is meant for remote and browser clients — the reference web UI talks to the server over WS, as can any JavaScript or non-Python client.

To start the server with WebSocket enabled:

```bash
python -m server --ipc-socket /tmp/jaato.sock --web-socket :8080 --daemon
```

WS clients authenticate with a bearer token (auto-generated to `~/.jaato/ws.token` on first start) sent either as `Authorization: Bearer <token>` on the upgrade request or as `?token=<token>` for browsers that can't set headers. The server stores only the SHA-256 digest and rejects bad tokens with WS close code 1008 before any session work happens.

**If you're writing a non-Python WS client**, the [Events vs requests](#events-vs-requests) protocol below is the same — frame each event as one WS text message containing the JSON returned by `to_json()` (no length prefix; WS already frames). Adding a Python WebSocket client to this SDK is a possible future addition; until then, point your WS client library at the same `Event` schema.

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
└─ TurnCompletedEvent   {usage: UsageBreakdown(...), duration_seconds: 1.5}
```

`AgentOutputEvent.mode` is `"write"` for a new block of output and `"append"` for streaming continuation chunks. `source` is one of `"model"`, `"tool"`, `"system"`, or a plugin name.

The three usage-bearing events (`TurnCompletedEvent`, `TurnProgressEvent`, `ContextUpdatedEvent`) all carry the same `UsageBreakdown` shape — token counts, cache hits, reasoning/thinking tokens, and `cost_usd` populated when the daemon can derive it. Cost resolution: provider-reported (e.g. claude_cli) wins over pricing-table computed from `.jaato/pricing.json`; otherwise `None` (never silently zero). See [docs/sdk-pricing.md](../docs/sdk-pricing.md) for the full pricing contract.

GC configuration is its own event (`GCConfigEvent`) since v1.0 — subscribe to that for status-bar GC display rather than reading from `ContextUpdatedEvent`.

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
| `y` | allow this tool execution |
| `n` | deny this tool execution |
| `a` / `always` | allow and whitelist the tool for this session |
| `t` / `turn` | allow remaining tool calls this turn |
| `i` / `idle` | allow until the session goes idle |
| `once` | allow once without remembering |
| `all` | allow all future requests in this session |
| `never` | deny and blacklist the tool for this session |
| `c:<text>` | deny **with feedback** the model sees as the tool result |
| `yc:<text>` | allow **with feedback** the model sees alongside the tool result |
| `e` | edit the arguments and re-prompt (pass `edited_arguments=...`); only offered when the request has editable content |

The two comment variants let you steer the model without simply rejecting the call. Pass them to `respond_to_permission` as a single string with the prefix and the text:

```python
await client.respond_to_permission(request_id, "c:please check the file size first")
await client.respond_to_permission(request_id, "yc:ok but write the result to /tmp/audit.log")
```

The server strips the `c:` / `yc:` prefix and forwards the comment to the model alongside the deny/allow decision. Empty text after the prefix falls back to plain `n` / `y`.

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
# By profile name — references .jaato/profiles/<name>.json on the server
await client.create_session(
    name="my-session",
    profile="researcher",
    agent="reviewer",
    agent_params={"focus": "security"},
)

# By inline spec — same shape as a profile JSON, no disk file needed
await client.create_session(
    name="ops-task",
    profile={
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "plugins": ["cli", "web_search"],
        "system_instructions": "You are an operations engineer.",
        # Any other field a profile JSON accepts: plugin_configs, gc,
        # env, max_turns, runtime_limits, model_tiers, ...
    },
)

await client.attach_session(session_id)
await client.get_default_session()
await client.list_sessions()           # response arrives as SessionListEvent
await client.list_profiles()           # response arrives as SessionProfilesEvent
```

#### Profile picker — SessionProfilesEvent shape

`list_profiles()` triggers a `SessionProfilesEvent` with a stable, versioned shape — pin against `schema_version` if you build a profile-picker UI:

```python
event.schema_version       # "1.0" — bumped only on breaking shape changes
event.profiles             # List[ProfileSummary]
event.parse_errors         # List[ProfileParseError] — broken files surface here, not in `profiles`
```

`ProfileSummary` exposes the safe-to-display subset of a profile (full field list in `jaato_sdk/events.py`):

| Field | Purpose |
|---|---|
| `name`, `description` | identity |
| `plugins`, `preloaded_plugins`, `plugin_configs` | capabilities |
| `model`, `provider`, `max_turns`, `model_tiers` | runtime |
| `gc`, `runtime_limits`, `completion_payload_schema` | structural config (dicts, expose as-is) |
| `env_var_names` | **names only** — env values never leave the daemon |

Deliberately **not** exposed: `system_instructions` (deprecated, now lives in agents), `icon_name` (deprecated), `inherits` (resolved during discovery), env values (sensitive). Profile-author secrets should always go through `${VAR}` indirection in `env`.

#### `profile` parameter polymorphism

The `profile` parameter is polymorphic:

- **`str`** → references a profile JSON on the server's disk under `.jaato/profiles/`. Use this when an operator has curated profiles for human users.
- **`dict`** → inline spec with the same shape. Use this when you're an orchestrator with your own governance layer and don't want to depend on disk files.

The two forms are mutually exclusive — pass one or the other. The server validates inline specs and rejects them with an `ErrorEvent` if `model` is missing (no silent default fallback). `agent` and `agent_params` are independent of `profile` and compose with either form: profile decides *capabilities* (model, plugins, GC), agent decides *persona* (system instructions / personality).

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

There are two ways to consume events: typed subscriptions (recommended) or the raw async iterator. They cooperate — subscribers always fire, and the iterator yields the same events.

#### Typed subscriptions

```python
from jaato_sdk import EventType

# One handler per type — only fires for that event type.
unsub = client.subscribe(EventType.PERMISSION_REQUESTED, on_perm)

# Fire once, then auto-unsubscribe.
unsub = client.subscribe_once(EventType.AGENT_COMPLETED, on_done)

# Catchall (every event regardless of type).
unsub = client.subscribe_all(lambda e: log(e))

# Register many in one call; unsub_all() removes them atomically.
unsub_all = client.subscribe_many({
    EventType.PERMISSION_REQUESTED: on_perm,
    EventType.TOOL_CALL_START:      on_tool_start,
    EventType.AGENT_COMPLETED:      on_done,
})
```

Handlers may be sync (`def`) or async (`async def`). Async handlers are scheduled fire-and-forget on the current event loop — order of *delivery* is FIFO, but order of *completion* is not guaranteed. Exceptions and rejections are logged and swallowed; one bad handler never breaks the stream or affects others. Subscribing during dispatch only takes effect for the next event (the handler list is snapshotted before iterating).

For the dispatcher to actually fire handlers, your code must drive the loop:

```python
# Option A — let subscribers do all the work
await client.drain_events()

# Option B — iterate and react to specific events directly
async for event in client.events():
    ...
```

The async iterator exits cleanly on disconnect. With `IPCRecoveryClient`, it survives reconnects: events from the new connection are yielded transparently after the gap, and subscribed handlers continue firing without re-registration.

#### Migration from `set_event_callback`

The old single-callback API was removed in jaato-sdk 0.4.0 — replace it with `subscribe_all`:

```python
# before
client.set_event_callback(handle)
await client.receive_events()

# after
client.subscribe_all(handle)
await client.drain_events()
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

### Protocol version mismatch

Each client pins a minimum **wire-protocol** version (`MIN_PROTOCOL_VERSION = "1.0"` on `IPCClient`, overridable per-instance via the `min_protocol_version=` constructor arg). On `connect()` the SDK reads `ConnectedEvent.protocol_version` from the daemon and runs a semver-flavoured compat check:

- Server major must equal client major (otherwise wire shapes are incompatible)
- Server minor must be ≥ client's required minor (otherwise daemon is missing fields the client expects)
- Server with newer minor is fine — additive optional fields the client will ignore

Mismatch raises `IncompatibleServerError` carrying both `server_protocol` and `min_protocol`, with a hint in the message about *why* (major mismatch vs missing minor). The recovery client classifies it as permanent — no retries. The daemon's package version (`server_version`) is reported for diagnostics but **not** used for the compat check; pin against `protocol_version` so a daemon bug-fix release doesn't require every client to re-pin.

See [docs/sdk-protocol-versioning.md](../docs/sdk-protocol-versioning.md) for the bump policy and the CHANGELOG of past wire versions.

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
    UsageBreakdown, GCConfigEvent,
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

Framing differs by transport:

- **IPC** — each frame is a 4-byte big-endian length prefix followed by the JSON payload. Max message size is 10 MiB. The Unix-socket variant uses `asyncio.open_unix_connection`; on Windows the SDK uses `loop.create_pipe_connection` against `\\.\pipe\<name>`.
- **WebSocket** — one event per WS text frame, no length prefix (WS frames itself). Authenticate with a bearer token on the upgrade request or via `?token=...`.

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
