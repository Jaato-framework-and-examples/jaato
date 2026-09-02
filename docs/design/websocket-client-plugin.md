# WebSocket Client Plugin — Outbound Real-Time Connections

## Overview

The WebSocket Client plugin enables agent sessions to **connect to external
WebSocket endpoints** and maintain persistent, bidirectional communication
channels. Where the Webhook plugin is an inbound HTTP listener (external
services push events *to* the agent), this plugin is an outbound connector
(the agent reaches out *to* external services).

Incoming messages are published to `TaskEventBus` (same as the webhook
plugin). The model subscribes via `subscribeToTasks` and receives messages
as inline events — no polling. It connects once, subscribes once, then goes
fully idle. When a message arrives, `subscribeToTasks`'s callback calls
`inject_prompt()` to wake the model for a new turn.

```
Agent Session                  WebSocket Client Plugin           External Service
(daemon mode)                  (outbound connections)            (Slack, Binance, K8s, ...)
        │                                │                              │
        │  subscribeToTasks(             │                              │
        │    event_types=["external_event"])                            │
        │  ws_connect(url="wss://...")    │                              │
        ├───────────────────────────────►│                              │
        │                                │  TCP + TLS + WS handshake   │
        │                                ├─────────────────────────────►│
        │                                │  101 Switching Protocols     │
        │                                │◄─────────────────────────────┤
        │  {connection_id: "ws-1"}       │                              │
        │◄───────────────────────────────┤                              │
        │                                │                              │
        │         (session is idle)      │◄────── message ─────────────┤
        │                                │  → TaskEventBus.publish()   │
        │                                │  → subscribeToTasks callback │
        │◄───────────────────────────────┤  → inject_prompt()          │
        │  (new turn triggered)          │                              │
        │  model processes message       │                              │
        │                                │                              │
        │  ws_send(data={...})           │                              │
        ├───────────────────────────────►│  ────── message ────────────►│
        │                                │                              │
        │         (session idle again)   │                              │
        │              ...sleeps...      │                              │
        │                                │◄────── message ─────────────┤
        │                                │  → TaskEventBus → inject    │
        │◄───────────────────────────────┤                              │
        │  (new turn triggered)          │                              │
```

## Motivation

The Webhook plugin solved **inbound event ingestion** — external services push
HTTP POSTs to the agent. But many real-time APIs are **WebSocket-first**:

| Service | Protocol | Webhook Alternative? |
|---------|----------|---------------------|
| Slack RTM API | WebSocket | Events API (webhooks) exists but RTM is richer |
| Discord Gateway | WebSocket | No webhook equivalent for real-time presence |
| Binance/Coinbase streams | WebSocket | REST polling only (latency, rate limits) |
| Kubernetes watch API | WebSocket / HTTP streaming | No webhook for cluster events |
| Home Assistant | WebSocket | No webhook equivalent |
| GraphQL subscriptions | WebSocket (`graphql-ws`) | No equivalent |
| Firebase Realtime DB | WebSocket | Cloud Functions (webhook-like) |
| MQTT over WebSocket | WebSocket | Native MQTT (different transport) |

These services require the **client to initiate** the connection. The agent
can't just listen on a port and wait — it needs to reach out, authenticate,
and maintain a persistent connection.

### Complementary to Webhooks

| Dimension | Webhook Plugin | WebSocket Client Plugin |
|-----------|---------------|------------------------|
| **Direction** | Inbound (agent listens) | Outbound (agent connects) |
| **Transport** | HTTP POST (stateless) | WebSocket (persistent, bidirectional) |
| **Initiator** | External service pushes | Agent connects and subscribes |
| **Auth model** | HMAC signature on each request | Headers on handshake + protocol-level auth |
| **Bidirectional** | No (receive only) | Yes (send and receive) |
| **Connection lifecycle** | Per-request | Long-lived, needs reconnection logic |
| **Endpoint requirement** | Agent needs a public/reachable URL | Agent only needs outbound access |
| **Delivery model** | Event bus (`subscribeToTasks` push / `pollForTasks` poll) + `webhook_poll` fallback | Event bus (`subscribeToTasks` push) |
| **Dependency** | stdlib only (`http.server`) | `websockets` (optional) |

The two plugins together cover the vast majority of real-time integration
patterns an agent might need.

## Distribution: Public Plugin, Premium Profiles

Following the Webhook plugin's distribution model:

The **plugin itself** lives in the public codebase (`shared/plugins/ws_client/`).
It's an outbound WebSocket connector with a single optional dependency — no
proprietary logic. Gating it would be inconsistent with the Webhook plugin
(inbound HTTP) being public.

**Premium ships curated profiles** — production-ready daemon session profiles
with battle-tested system prompts, authentication flows, and reconnection
strategies for specific integrations:

- `slack-rtm-bot` — Slack RTM connection, channel monitoring, message responses
- `discord-bot` — Discord Gateway, presence tracking, slash command responses
- `market-watcher` — Binance/Coinbase streams, price alerts, portfolio tracking
- `k8s-watcher` — Kubernetes watch API, pod events, deployment monitoring

## Design Principles

1. **Push, don't poll.** Incoming messages are delivered to the model via
   `inject_prompt()`. The model goes fully idle between messages — no
   wasted turns on empty poll loops. It wakes up only when there's
   something to process.
2. **Model drives outbound.** The model calls `ws_connect` and `ws_send`.
   The plugin never sends data without the model's explicit instruction.
   Inbound delivery is the only automatic behavior.
3. **Connections are explicit.** The model connects to specific URLs with
   specific parameters. No magic auto-connections. Pre-configured named
   connections in config are a convenience, not a requirement.
4. **Reconnection is opt-in.** Dropped connections surface as injected
   system messages. Auto-reconnect is configurable per-connection but defaults to
   off — the model should understand when connections drop.
5. **Optional dependency.** The plugin requires `websockets` but degrades
   gracefully — importing the plugin without the dependency installed
   produces a clear error message, not a crash.

## Message Delivery: Event Bus + `subscribeToTasks`

### The Core Problem: Waking an Idle Model

A WebSocket connection might be quiet for seconds, minutes, or hours between
messages (e.g., a Slack bot waiting for mentions). The delivery mechanism must:

1. **Let the model sleep** — no wasted turns or tokens while nothing happens.
2. **Wake the model** — when a message arrives, start a new turn automatically.
3. **Batch naturally** — if messages arrive in bursts while the model is busy,
   deliver them together.

### How `inject_prompt()` Wakes Idle Sessions

`inject_prompt()` can start a new model turn on an idle session — but only
when the **continuation callback** is set. This callback is configured by
`JaatoServer.initialize()` (`server/core.py:2284`) and calls
`_start_model_thread()` to kick off a new turn:

```python
# jaato_session.py:680-689 — inject_prompt() on idle session
if (
    self._activity_phase == ActivityPhase.IDLE
    and not self._is_running
    and self._on_continuation_needed    # ← set by server only
):
    self._on_continuation_needed(text)  # → _start_model_thread()
```

In **daemon mode** (the primary use case for WebSocket connections), this
callback is always set. In standalone/interactive mode, `inject_prompt()`
on an idle session just queues the message — but standalone mode is not a
meaningful scenario for this plugin (users provide input directly).

Note: `subscribeToTasks` (todo plugin) also relies on `inject_prompt()` for
its push delivery — the same mechanism, same daemon-mode requirement.

### Delivery Model: Event Bus or Direct `inject_prompt()`?

The webhook plugin follows a clean separation: it publishes events to the
`TaskEventBus`, and `subscribeToTasks` handles the `inject_prompt()` delivery.
The plugin doesn't need to know about session wakeup — it just publishes.
Cross-agent fan-out comes for free (multiple sessions subscribe to the same
events).

The WebSocket plugin could follow the same pattern:

1. Reader thread publishes each message to `TaskEventBus` as an
   `EXTERNAL_EVENT` with `source_agent="websocket:<connection_id>"`
2. Model calls `subscribeToTasks(event_types=["external_event"])` to receive
   them (same as webhook)
3. `subscribeToTasks` handles `inject_prompt()` delivery

**Advantages over direct `inject_prompt()`:**
- Consistent with the webhook plugin's pattern
- Cross-agent fan-out built-in (multiple sessions process the same stream)
- Plugin doesn't own the delivery concern — cleaner separation
- Model uses the same `subscribeToTasks` pattern for both webhook and
  WebSocket events

**Trade-off:**
- Requires the model to call `subscribeToTasks` as a setup step after
  `ws_connect` (one extra tool call)
- Messages route through the bus before reaching the session (minimal overhead)

**Recommendation:** Follow the webhook pattern. The one extra `subscribeToTasks`
call is negligible, and consistency across event-driven plugins is worth more
than saving one tool call. The `ws_connect` tool description and system
instructions should guide the model to subscribe immediately after connecting.

The event bus is also used for cross-agent fan-out regardless — so the
plugin publishes to the bus either way. The question is only whether the
*owning session* receives via bus subscription or via direct `inject_prompt()`.
Using the bus for both keeps things uniform.

### Prerequisite: Extract Event Bus from Todo Plugin

Currently `TaskEventBus`, `subscribeToTasks`, `pollForTasks`, and the event
types (`TaskEvent`, `TaskEventType`, `EventFilter`) all live inside the todo
plugin. This is an architectural smell — the event bus is a cross-cutting
concern used by webhook, WebSocket, and subagent coordination, not a todo
feature.

Before implementing the WebSocket plugin, the event bus should be extracted
into a shared component:

```
# Current (coupled to todo)
shared/plugins/todo/event_bus.py          → TaskEventBus
jaato_sdk/plugins/todo/models.py          → TaskEvent, TaskEventType, EventFilter
shared/plugins/todo/plugin.py             → subscribeToTasks, pollForTasks tools

# Proposed (shared infrastructure)
shared/event_bus.py                       → EventBus (renamed from TaskEventBus)
jaato_sdk/events.py or jaato_sdk/event_bus_types.py → Event, EventType, EventFilter
shared/plugins/event_bus/                 → subscribeToEvents, pollForEvents tools
  ├── __init__.py                         # PLUGIN_KIND = "tool"
  ├── plugin.py                           # EventBusPlugin with subscribe/poll tools
  └── tests/
```

**What changes:**
- `TaskEventBus` → `EventBus` (not task-specific)
- `TaskEvent` → `Event` (generic)
- `TaskEventType` gains `EXTERNAL_EVENT` as a first-class type (not bolted on)
- `subscribeToTasks` → `subscribeToEvents` (generic name)
- `pollForTasks` → `pollForEvents` (generic name)
- Todo plugin imports from the shared event bus, not the reverse
- Webhook and WebSocket plugins import from the shared event bus directly

**Backward compatibility:** The todo plugin can re-export the old names
(`subscribeToTasks` as an alias for `subscribeToEvents`) during a transition
period. The event bus singleton is shared — all plugins see the same events
regardless of which tool name the model uses.

This extraction is a prerequisite for the WebSocket plugin — without it, the
plugin would need to import from `shared/plugins/todo/event_bus.py`, creating
a dependency on the todo plugin that makes no semantic sense.

### Why Not `StreamManager`?

The framework's `StreamingCapable` + `StreamManager` infrastructure handles
incremental delivery well during active turns — but it's designed for
**finite streams** (search results trickling in over seconds), not
**infinite connections** that live for hours.

The critical limitation is in the streaming continuation loop
(`jaato_session.py:3586-3642`):

```python
while self._has_active_streams() and attempts < 20:
    updates = self._wait_for_streaming_updates()  # blocks ~5s
    if not updates:
        break  # ← EXIT: turn ends, model goes fully idle
```

After the model processes a batch of WebSocket messages and no new ones arrive
within ~5 seconds, the loop breaks and `send_message()` returns. The model is
now truly idle — no active turn, nothing listening. If a Slack message arrives
30 seconds later, the `StreamChunk` goes into `StreamManager` but **nobody
picks it up**. There is no mechanism to wake the model from true idle.

### The Right Tool: `TaskEventBus` + `subscribeToTasks`

The combination of `TaskEventBus.publish()` (plugin side) and
`subscribeToTasks` (model side) provides exactly what's needed:

1. Plugin publishes → bus distributes → subscription callback calls
   `inject_prompt()` → session wakes
2. Multiple sessions can subscribe → fan-out for free
3. Natural batching: messages that arrive during a busy turn queue up
   in the session's message queue and are delivered together

**What the model sees** (delivered as `[SUBAGENT event=external_event]` inline
messages, same format as webhook events):

```
[SUBAGENT agent_id=websocket:ws-1 event=external_event]
{"source": "websocket", "connection_id": "ws-1", "message_type": "text",
 "data": "{\"e\":\"trade\",\"s\":\"BTCUSDT\",\"p\":\"67420.50\"}"}
```

The model processes the message and can respond with `ws_send` if needed.
When it finishes its turn, the session goes idle. When the next WebSocket
message arrives, the cycle repeats.

### Natural Batching

High-volume streams (market data, log tails) could produce hundreds of
messages per second. The `subscribeToTasks` → `inject_prompt()` → session
message queue chain provides natural batching:

1. First message arrives → `inject_prompt()` → new turn starts.
2. While model is processing, more messages arrive → each triggers
   `inject_prompt()` → messages queue up in the session's message queue.
3. Model finishes its turn → drains the queue → sees all queued messages.
4. Model processes the batch → finishes → goes idle.
5. Next message arrives → cycle repeats.

The model's own processing time acts as the batching window. Fast streams
batch more, slow streams deliver individually. No custom debounce timers
or batch size configs needed.

### First-Message Auth Bootstrapping

Some protocols (Discord Gateway, GraphQL subscriptions) require the model to
read the first message from the server and respond with an auth payload.

The `ws_connect` tool result includes **initial messages** captured during a
brief startup window before the reader thread starts publishing to the bus:

```python
def _execute_connect(self, args):
    """Connect and return initial messages from the server.

    After the WebSocket handshake completes, waits up to
    initial_read_timeout (default 3s) for any server-initiated
    messages (hello, capabilities, auth challenge). These are
    returned directly in the tool result so the model can respond
    with ws_send in the same turn.

    After the initial read, the reader thread starts publishing
    subsequent messages to TaskEventBus.
    """
    conn = self._create_connection(args)
    conn.connect()

    # Capture initial messages (hello, auth challenge, etc.)
    initial_messages = conn.drain_initial(timeout=3.0)

    # Start reader thread — messages published to TaskEventBus
    conn.start_reader(on_message=self._on_message_received)

    return {
        "connection_id": conn.connection_id,
        "status": "connected",
        "url": conn.url,
        "initial_messages": [self._format_message(m) for m in initial_messages],
        "message": f"Connected. {len(initial_messages)} initial message(s). "
                   "Subsequent messages delivered via event subscription.",
    }
```

**Discord example flow:**

```
Agent: subscribeToTasks(event_types=["external_event"])
→ {subscription_id: "sub-1", message: "Subscribed to 1 event type(s)."}

Agent: ws_connect(url="wss://gateway.discord.gg/?v=10&encoding=json")
→ {
    connection_id: "ws-1",
    initial_messages: [
      {type: "text", data: '{"op":10,"d":{"heartbeat_interval":41250}}'}
    ],
    message: "Connected. 1 initial message(s). Subsequent messages delivered via event subscription."
  }

Agent: ws_send(connection_id="ws-1", data='{"op":2,"d":{"token":"...","intents":513}}')
→ {status: "sent"}

    ... session goes idle ...

    [SUBAGENT agent_id=websocket:ws-1 event=external_event]    ← subscribeToTasks callback
    {"source":"websocket","connection_id":"ws-1","data":"{\"op\":0,\"t\":\"READY\",...}"}

Agent: (processes READY event, session goes idle again)

    ... waits for next message ...
```

## Configuration

### Config File: `websocket.json`

```json
{
  "max_connections": 4,
  "connections": {
    "slack": {
      "url": "wss://wss-primary.slack.com/link/${SLACK_WS_URL_TOKEN}",
      "headers": {
        "Authorization": "Bearer ${SLACK_BOT_TOKEN}"
      },
      "subprotocols": [],
      "reconnect": true,
      "reconnect_max_attempts": 10,
      "reconnect_base_delay": 1.0,
      "reconnect_max_delay": 60.0,
      "ping_interval": 30,
      "ping_timeout": 10
    },
    "binance": {
      "url": "wss://stream.binance.com:9443/ws",
      "reconnect": true,
      "ping_interval": 180
    }
  }
}
```

### Config Precedence

```
1. Profile plugin_configs.ws_client   (highest — per-session override)
2. <workspace>/.jaato/websocket.json  (project-level)
3. ~/.jaato/websocket.json            (user-level)
4. Built-in defaults                  (lowest)
```

Each layer is merged, not replaced — a profile can override just
`max_connections` while inheriting named connections from workspace config.

### Built-in Defaults

| Key | Default | Description |
|-----|---------|-------------|
| `max_connections` | `4` | Maximum concurrent WebSocket connections per session |
| `max_buffer_size` | `1000` | Per-connection message buffer (FIFO eviction) |
| `connections` | `{}` | Named pre-configured connections |

### Per-Connection Config Defaults

| Key | Default | Description |
|-----|---------|-------------|
| `url` | *(required)* | WebSocket URL (`ws://` or `wss://`) |
| `headers` | `{}` | Extra headers sent during handshake |
| `subprotocols` | `[]` | WebSocket subprotocols to negotiate |
| `reconnect` | `false` | Auto-reconnect on connection loss |
| `reconnect_max_attempts` | `5` | Max consecutive reconnect attempts before giving up |
| `reconnect_base_delay` | `1.0` | Initial reconnect delay (seconds, exponential backoff) |
| `reconnect_max_delay` | `30.0` | Maximum reconnect delay (seconds) |
| `ping_interval` | `20` | Seconds between keepalive pings (`0` = disabled) |
| `ping_timeout` | `20` | Seconds to wait for pong before considering connection dead |
| `initial_read_timeout` | `3.0` | Seconds to wait for server hello after handshake |

### Environment Variable Support

Config values support `${VAR}` expansion (via existing `expand_variables()`):

- `${SLACK_BOT_TOKEN}` — Slack bot OAuth token
- `${DISCORD_BOT_TOKEN}` — Discord bot token
- `${SLACK_WS_URL_TOKEN}` — Dynamic portion of Slack RTM URL

## Plugin Architecture

### Module Structure

```
shared/plugins/ws_client/
├── __init__.py          # PLUGIN_KIND = "tool", create_plugin()
├── plugin.py            # WebSocketClientPlugin — connection mgmt + event bus publishing
├── connection.py        # WebSocketConnection — per-connection state + reader thread
├── config.py            # WebSocketClientConfig dataclass, config loading/merging
└── tests/
    ├── test_plugin.py
    ├── test_connection.py
    └── test_config.py
```

### Plugin Class

```python
class WebSocketClientPlugin:
    """Outbound WebSocket client plugin for real-time bidirectional communication.

    Manages persistent WebSocket connections to external services. Each
    connection runs a reader thread that publishes incoming messages to
    TaskEventBus. The model subscribes via subscribeToTasks to receive
    them as inline events (pushed via inject_prompt by the subscription
    callback).

    Delivery model:
        - Inbound: reader thread → TaskEventBus → subscribeToTasks →
          inject_prompt() → model wakes on message arrival
        - Outbound: explicit via ws_send tool calls
        - The model never polls — it goes fully idle between messages
        - Natural batching: messages that arrive during a busy turn queue
          up in the session's message queue and are delivered together

    Lifecycle:
        1. initialize(config) — load and merge config
        2. Model calls subscribeToTasks(event_types=["external_event"])
        3. ws_connect tool call — establish connection, return initial
           messages, start reader thread publishing to TaskEventBus
        4. Reader thread publishes each message as EXTERNAL_EVENT
        5. subscribeToTasks callback → inject_prompt() → model processes
        6. Model responds with ws_send if needed → goes idle → waits
        7. ws_close or shutdown() — close connections, stop reader threads

    Requires: `websockets` package (optional dependency).
    """

    PLUGIN_KIND = "tool"

    @property
    def name(self) -> str:
        return "ws_client"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load config with standard precedence."""
        ...

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for WebSocket operations."""
        ...

    def get_executors(self) -> Dict[str, Any]:
        """Return tool executors."""
        ...

    def shutdown(self) -> None:
        """Close all connections, stop reader threads, clean up."""
        ...
```

### Dependency Gating

```python
# ws_client/__init__.py

PLUGIN_KIND = "tool"

_MISSING_DEP = None
try:
    import websockets  # noqa: F401
except ImportError:
    _MISSING_DEP = "websockets"


def create_plugin():
    if _MISSING_DEP:
        from .plugin import WebSocketClientPlugin
        plugin = WebSocketClientPlugin()
        plugin._missing_dependency = _MISSING_DEP
        return plugin
    from .plugin import WebSocketClientPlugin
    return WebSocketClientPlugin()
```

When the dependency is missing, all tool executors return a clear error:

```json
{
  "error": "WebSocket client requires the 'websockets' package. Install with: pip install websockets"
}
```

This follows the same pattern as `interactive_shell` with `pexpect`/`wexpect`.

## Tool Schemas

The plugin exposes four tools, all `discoverability="discoverable"`:

### `ws_connect`

Connect to a WebSocket endpoint. Returns a connection ID and any initial
server messages (hello, auth challenge). Subsequent messages are published
to `TaskEventBus` — use `subscribeToTasks` before connecting to receive them.

```json
{
  "name": "ws_connect",
  "description": "Connect to a WebSocket endpoint. Returns a connection_id and any initial server messages. Incoming messages are published to TaskEventBus as external_event — use subscribeToTasks before connecting to receive them. Use ws_send to send messages back.",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string",
        "description": "WebSocket URL (wss:// or ws://). Required unless 'name' is provided."
      },
      "name": {
        "type": "string",
        "description": "Name of a pre-configured connection from websocket.json. Overrides url."
      },
      "headers": {
        "type": "object",
        "description": "Extra headers for the handshake (e.g., Authorization). Merged with config headers."
      },
      "subprotocols": {
        "type": "array",
        "items": { "type": "string" },
        "description": "WebSocket subprotocols to negotiate (e.g., ['graphql-ws'])."
      }
    },
    "required": []
  }
}
```

**Returns:**
```json
{
  "connection_id": "ws-1",
  "status": "connected",
  "url": "wss://gateway.discord.gg/?v=10&encoding=json",
  "negotiated_subprotocol": null,
  "initial_messages": [
    {
      "type": "text",
      "data": "{\"op\":10,\"d\":{\"heartbeat_interval\":41250}}"
    }
  ],
  "message": "Connected. 1 initial message(s). Subsequent messages delivered via streaming updates."
}
```

**Returns (no initial messages):**
```json
{
  "connection_id": "ws-1",
  "status": "connected",
  "url": "wss://stream.binance.com:9443/ws",
  "initial_messages": [],
  "message": "Connected. Incoming messages published to event bus as external_event."
}
```

### `ws_send`

Send a message through an open connection.

```json
{
  "name": "ws_send",
  "description": "Send a message through an open WebSocket connection.",
  "parameters": {
    "type": "object",
    "properties": {
      "connection_id": {
        "type": "string",
        "description": "Connection ID from ws_connect"
      },
      "data": {
        "type": "string",
        "description": "Message to send (text). For JSON payloads, pass the JSON string."
      }
    },
    "required": ["connection_id", "data"]
  }
}
```

**Returns:**
```json
{
  "connection_id": "ws-1",
  "status": "sent",
  "bytes_sent": 85
}
```

**Error (disconnected):**
```json
{
  "connection_id": "ws-1",
  "error": "Connection is not open (status: disconnected). Reconnect or create a new connection."
}
```

### `ws_close`

Close a WebSocket connection gracefully.

```json
{
  "name": "ws_close",
  "description": "Close a WebSocket connection. Sends a close frame and waits for graceful shutdown.",
  "parameters": {
    "type": "object",
    "properties": {
      "connection_id": {
        "type": "string",
        "description": "Connection ID from ws_connect"
      },
      "code": {
        "type": "integer",
        "description": "WebSocket close code (default 1000 = normal closure)"
      },
      "reason": {
        "type": "string",
        "description": "Close reason string (optional, max 123 bytes)"
      }
    },
    "required": ["connection_id"]
  }
}
```

**Returns:**
```json
{
  "connection_id": "ws-1",
  "status": "closed",
  "close_code": 1000,
  "messages_received_total": 4523,
  "messages_sent_total": 12,
  "uptime_seconds": 1800
}
```

### `ws_status`

List active connections and their state. Auto-approved (read-only).

```json
{
  "name": "ws_status",
  "description": "Show active WebSocket connections, their state, and message statistics.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

**Returns:**
```json
{
  "connections": [
    {
      "connection_id": "ws-1",
      "url": "wss://stream.binance.com:9443/ws",
      "status": "connected",
      "connected_at": "2026-03-08T14:00:00Z",
      "uptime_seconds": 1800,
      "messages_received": 4523,
      "messages_sent": 12,
      "messages_delivered": 4520,
      "messages_dropped": 3,
      "last_message_at": "2026-03-08T14:29:58Z",
      "reconnect_count": 0
    }
  ],
  "available_connections": ["slack", "binance"],
  "max_connections": 4
}
```

## Connection Architecture

### WebSocketConnection Class

Each connection is represented by a `WebSocketConnection` object that owns a
reader thread and a message delivery callback:

```python
class WebSocketConnection:
    """A single WebSocket connection with a background reader thread.

    Lifecycle:
        1. connect(url, headers, subprotocols) — handshake, start reader
        2. drain_initial(timeout) — capture server hello messages
        3. start_reader(on_message) — begin recv loop, call back on each message
        4. send(data) — send message through the connection
        5. close() — send close frame, stop reader, clean up

    The reader thread runs until the connection closes (remotely or locally).
    On each received message, it calls the on_message callback (which the
    plugin wires to inject_prompt()). On unexpected close, it calls the
    on_status_change callback and optionally triggers reconnection.

    Thread safety:
        - _lock (threading.Lock) protects status transitions
        - _send_lock (threading.Lock) serializes writes
    """

    def __init__(self, connection_id, config, on_status_change=None):
        self.connection_id = connection_id
        self.url = None
        self.status = "idle"            # idle → connecting → connected → disconnected → closed
        self._ws = None                 # websockets.sync.client.ClientConnection
        self._reader_thread = None
        self._on_message = None         # Callable[[str, IncomingMessage], None]
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stats = ConnectionStats()
        self._config = config
        self._on_status_change = on_status_change
        self._reconnect_count = 0
        self._stop_requested = False
```

### Reader Thread

Each connection runs a daemon reader thread that loops on `recv()` and
delivers messages via the callback:

```python
def _reader_loop(self):
    """Background thread: read messages and deliver via callback.

    Runs until connection closes or stop is requested. Each message
    is passed to the on_message callback (which the plugin wires to
    inject_prompt()). On unexpected close with reconnect enabled,
    attempts to re-establish the connection with exponential backoff.
    """
    while not self._stop_requested:
        try:
            raw = self._ws.recv(timeout=1.0)  # Short timeout for stop checking
            msg = IncomingMessage(
                message_id=self._next_message_id(),
                timestamp=datetime.utcnow().isoformat() + "Z",
                type="binary" if isinstance(raw, bytes) else "text",
                data=raw if isinstance(raw, str) else base64.b64encode(raw).decode(),
            )
            if self._on_message:
                self._on_message(self.connection_id, msg)
            self._stats.messages_received += 1

        except websockets.exceptions.ConnectionClosed as exc:
            self._handle_disconnect(exc.code, exc.reason)
            if self._should_reconnect():
                self._attempt_reconnect()
            else:
                break

        except TimeoutError:
            continue  # Normal — just checking _stop_requested
```

### Connection State Machine

```
         ws_connect()
  idle ──────────────► connecting
                           │
                    handshake ok │ handshake fail
                           │         │
                           ▼         ▼
                      connected    error
                       │     │
            remote close │     │ ws_close()
                       │     │
                       ▼     ▼
                   disconnected
                    │        │
        reconnect=true │  reconnect=false │
                    │        │
                    ▼        ▼
               connecting   closed
```

Status transitions are protected by `_lock`. The `on_status_change` callback
notifies the plugin for system message injection into the session.

### System Messages

Connection lifecycle events are published to `TaskEventBus` as
`EXTERNAL_EVENT`s with a `"system"` message type, so they arrive through the
same `subscribeToTasks` channel as regular messages:

| Event | Event Data |
|-------|-----------|
| Disconnected | `{"source":"websocket","connection_id":"ws-1","message_type":"system","data":"Connection closed by remote (code=1006). Reconnecting..."}` |
| Reconnected | `{"source":"websocket","connection_id":"ws-1","message_type":"system","data":"Reconnected successfully (was disconnected for 8.2s)"}` |
| Reconnect failed | `{"source":"websocket","connection_id":"ws-1","message_type":"system","data":"Reconnection failed after 5 attempts. Use ws_connect to reconnect."}` |
| Ping timeout | `{"source":"websocket","connection_id":"ws-1","message_type":"system","data":"Connection appears dead (ping timeout). Closing."}` |

These flow through the same event bus pipeline — no special delivery path for
system messages.

## Event Bus Integration

Following the webhook plugin's pattern, the reader thread publishes each
incoming WebSocket message to `TaskEventBus`. The owning session (and any
other sessions) receive events via `subscribeToTasks`:

```python
def _on_message_received(self, connection_id: str, message: IncomingMessage):
    """Called by reader thread when a WebSocket message arrives.

    Publishes to TaskEventBus as an EXTERNAL_EVENT. The owning session
    receives it via subscribeToTasks → inject_prompt() (set up during
    ws_connect). Other sessions can also subscribe for fan-out.
    """
    try:
        bus = TaskEventBus.get_instance()
        event = TaskEvent.create(
            event_type=TaskEventType.EXTERNAL_EVENT,
            source_agent=f"websocket:{connection_id}",
            data={
                "source": "websocket",
                "connection_id": connection_id,
                "message_type": message.type,
                "data": message.data,
                "formatted": self._format_message(connection_id, message),
            }
        )
        bus.publish(event)
    except Exception:
        logger.debug("TaskEventBus unavailable, message not delivered")
```

**Delivery flow:**

```
Reader thread → TaskEventBus.publish()
                    │
                    ├─→ subscribeToTasks callback (owning session)
                    │       └─→ inject_prompt() → new turn
                    │
                    └─→ subscribeToTasks callback (other sessions)
                            └─→ inject_prompt() → new turn
```

**Model setup** (guided by system instructions and `ws_connect` tool result):

```
1. subscribeToTasks(event_types=["external_event"])   ← receive events
2. ws_connect(url="wss://...")                         ← start connection
3. Messages arrive as [SUBAGENT event=external_event] inline messages
```

This is the same pattern the webhook plugin uses — consistent across all
event-driven plugins.

## Reconnection Strategy

WebSocket connections are inherently fragile. The plugin provides configurable
auto-reconnection per connection:

### Exponential Backoff

```python
def _attempt_reconnect(self):
    """Try to reconnect with exponential backoff.

    Attempts: delay = min(base_delay * 2^attempt, max_delay) + jitter
    Jitter: ±25% to prevent thundering herd on shared endpoints.

    System messages about reconnection progress are published to the
    event bus so the model stays informed via its subscription.
    """
    for attempt in range(self._config.reconnect_max_attempts):
        if self._stop_requested:
            return

        delay = min(
            self._config.reconnect_base_delay * (2 ** attempt),
            self._config.reconnect_max_delay,
        )
        jitter = delay * 0.25 * (2 * random.random() - 1)
        actual_delay = max(0.1, delay + jitter)

        self._push_system_message(
            f"Reconnecting (attempt {attempt + 1}/{self._config.reconnect_max_attempts}, "
            f"next retry in {actual_delay:.1f}s)..."
        )
        time.sleep(actual_delay)

        try:
            self._do_connect()
            self._reconnect_count += 1
            self._push_system_message(
                f"Reconnected successfully (attempt {attempt + 1})"
            )
            return  # Success — resume reader loop
        except Exception as exc:
            logger.warning("Reconnect attempt %d failed: %s", attempt + 1, exc)

    self._push_system_message(
        f"Reconnection failed after {self._config.reconnect_max_attempts} attempts. "
        "Use ws_connect to reconnect manually."
    )
    self._set_status("closed")
```

### When Reconnection Applies

| Scenario | Reconnect? | Rationale |
|----------|-----------|-----------|
| Remote close (1000 Normal) | No | Server intentionally closed |
| Remote close (1001 Going Away) | Yes | Server restarting |
| Abnormal close (1006) | Yes | Network issue |
| Ping timeout | Yes | Connection dead |
| `ws_close()` by model | No | Model intentionally closed |
| Plugin `shutdown()` | No | Session ending |
| Handshake failure (initial) | No | Bad URL/auth, not transient |

## Authentication Patterns

WebSocket authentication varies by service. The plugin supports the common
patterns without imposing a specific auth flow:

### Pattern 1: Headers on Handshake

Most services accept Bearer tokens or API keys in the initial HTTP handshake.

```json
{
  "connections": {
    "slack": {
      "url": "wss://wss-primary.slack.com/link/...",
      "headers": {
        "Authorization": "Bearer ${SLACK_BOT_TOKEN}"
      }
    }
  }
}
```

The model can also pass headers dynamically via `ws_connect`:
```
ws_connect(url="wss://...", headers={"Authorization": "Bearer xoxb-..."})
```

### Pattern 2: Query String Tokens

Some services embed tokens in the URL.

```
ws_connect(url="wss://stream.example.com/v1?token=${API_TOKEN}")
```

### Pattern 3: First-Message Auth

Services like Discord and some GraphQL endpoints require an authentication
message after the WebSocket handshake completes. The `initial_messages` in the
`ws_connect` response give the model the server's hello/challenge, and the
model responds with `ws_send` in the same turn:

```
Agent: ws_connect(url="wss://gateway.discord.gg/?v=10&encoding=json")
→ {
    connection_id: "ws-1",
    initial_messages: [
      {type: "text", data: '{"op":10,"d":{"heartbeat_interval":41250}}'}
    ]
  }

Agent: ws_send(connection_id="ws-1", data='{"op":2,"d":{"token":"...","intents":513}}')
→ {status: "sent"}

    ... session goes idle, push delivery begins ...

    [WebSocket ws-1] 1 message received:
    1. {"op":0,"t":"READY","d":{...}}

    (model processes READY, goes idle again)
```

No special auth abstraction needed — the model reads the protocol docs and
drives the handshake via `ws_send`.

### Pattern 4: Two-Phase URL (Slack RTM)

Slack's RTM API requires an HTTP call to get a WebSocket URL, then connecting:

```
Agent: cli(command="curl -s https://slack.com/api/rtm.connect -d token=$SLACK_TOKEN")
→ {"ok": true, "url": "wss://cerberus-xxxx.lb.slack-msgs.com/websocket/..."}

Agent: ws_connect(url="wss://cerberus-xxxx.lb.slack-msgs.com/websocket/...")
→ {connection_id: "ws-1", status: "connected", initial_messages: [...]}
```

The model orchestrates multi-step auth using existing tools. The WebSocket
plugin doesn't need to know about Slack's API.

## Daemon Session Profile

### Example: `.jaato/profiles/slack-rtm-bot.json`

```json
{
  "name": "slack-rtm-bot",
  "description": "Daemon session connected to Slack via RTM WebSocket",
  "model": "gemini-2.5-flash",
  "provider": "google_genai",
  "plugins": ["ws_client(preload)", "cli", "todo"],
  "plugin_configs": {
    "ws_client": {
      "max_connections": 2,
      "connections": {
        "slack": {
          "url": "${SLACK_RTM_URL}",
          "reconnect": true,
          "reconnect_max_attempts": 10,
          "reconnect_max_delay": 60.0,
          "ping_interval": 30
        }
      }
    }
  },
  "system_instructions": "You are a Slack bot connected via RTM WebSocket.\n\nOn startup:\n1. Call subscribeToTasks(event_types=['external_event']) to receive events.\n2. Call ws_connect(name=\"slack\") to connect.\n3. If the initial_messages contain a hello, you're ready.\n4. Incoming Slack messages arrive as inline event notifications.\n5. Parse each message JSON and respond using ws_send.\n6. If you receive a disconnect event, wait — auto-reconnect is enabled.\n\nYou do not need to poll. Messages arrive as event notifications.",
  "max_turns": 0,
  "gc": {
    "type": "budget",
    "threshold_percent": 75.0,
    "preserve_recent_turns": 3
  }
}
```

### Example: `.jaato/profiles/market-watcher.json`

```json
{
  "name": "market-watcher",
  "description": "Daemon session monitoring cryptocurrency price feeds",
  "model": "gemini-2.5-flash",
  "provider": "google_genai",
  "plugins": ["ws_client(preload)", "todo"],
  "plugin_configs": {
    "ws_client": {
      "debounce_ms": 500,
      "max_batch_size": 100,
      "connections": {
        "binance": {
          "url": "wss://stream.binance.com:9443/ws",
          "reconnect": true,
          "ping_interval": 180
        }
      }
    }
  },
  "system_instructions": "You are a market data monitor.\n\nOn startup:\n1. subscribeToTasks(event_types=['external_event'])\n2. ws_connect(name=\"binance\")\n3. ws_send to subscribe: {\"method\":\"SUBSCRIBE\",\"params\":[\"btcusdt@trade\",\"ethusdt@trade\"],\"id\":1}\n4. Trade data arrives as event notifications — no polling needed.\n5. Analyze price movements and maintain a running summary.\n6. Alert on significant moves (>2% in 5 minutes).",
  "max_turns": 0,
  "gc": {
    "type": "budget",
    "threshold_percent": 80.0,
    "preserve_recent_turns": 5
  }
}
```

## Security Considerations

### URL Validation

The plugin validates WebSocket URLs before connecting:

- Must use `ws://` or `wss://` scheme (no `http://`, `file://`, etc.)
- `wss://` (TLS) is strongly recommended; `ws://` connections log a warning
- No local file access, no SSRF via URL scheme abuse

### Credential Handling

- Headers containing tokens are never logged at INFO level
- Connection URLs are logged with query parameters redacted
- Config secrets use `${VAR}` expansion — never hardcoded
- `ws_status` redacts the URL query string and auth headers

### Network Boundaries

- The plugin only makes **outbound** connections — no listening ports opened
- Proxy support: respects `HTTPS_PROXY` / `HTTP_PROXY` / `NO_PROXY` env vars
  (via `websockets` library's proxy support or manual `CONNECT` tunneling)
- Corporate environments: works through HTTP proxies with CONNECT method for
  `wss://` (TLS) connections

### Resource Limits

- `max_connections` prevents connection sprawl (default 4)
- `max_buffer_size` prevents memory growth per connection (default 1000, FIFO)
- Session message queue naturally batches — messages that arrive during a busy
  turn are delivered together when the turn completes
- Reconnection attempts are bounded (`reconnect_max_attempts`)
- Ping timeouts detect dead connections

### Message Identification

WebSocket events published to the event bus include
`source_agent="websocket:<connection_id>"`, which `subscribeToTasks` renders
as `[SUBAGENT agent_id=websocket:ws-1 event=external_event]`. The model can
distinguish WebSocket messages from webhook events, subagent results, and
other event sources by the `source_agent` prefix.

The event data includes the raw WebSocket message content as-is — no escaping
or sanitization. This is intentional: the model needs to see the exact data
to parse protocol-specific formats (JSON APIs, binary-encoded payloads).

## Comparison: WebSocket Client vs Interactive Shell

The `interactive_shell` plugin also maintains persistent bidirectional sessions.
Key differences:

| Dimension | Interactive Shell | WebSocket Client |
|-----------|------------------|-----------------|
| **Target** | Local processes (PTY) | Remote services (network) |
| **Transport** | Pseudo-terminal (stdin/stdout) | WebSocket (TCP + TLS) |
| **Output format** | Raw terminal output (ANSI stripped) | Structured messages (JSON, text) |
| **Delivery** | Must use `shell_read`/`shell_input` | Push via event bus (`subscribeToTasks`) |
| **Auth** | OS-level (user permissions) | Application-level (tokens, headers) |
| **Reconnection** | N/A (process is dead) | Auto-reconnect with backoff |
| **Use case** | REPLs, debuggers, SSH | APIs, event streams, chat bots |

There is no overlap — they serve completely different integration patterns.

**Note:** Both plugins use the same delivery model: publish to `TaskEventBus`,
receive via `subscribeToTasks`. The webhook plugin's system instructions have
been updated to recommend `subscribeToTasks` over `pollForTasks` (see commit
in this branch), aligning both plugins on a consistent push-based pattern.

## Dependency Choice: `websockets`

### Why `websockets`

| Criterion | `websockets` | `aiohttp` | stdlib-only |
|-----------|-------------|-----------|-------------|
| Sync client API | Yes (since v12.0) | No (async only) | N/A |
| Maturity | 10+ years, widely used | Mature but heavier | No WS in stdlib |
| Dependencies | None (pure Python) | multidict, yarl, etc. | N/A |
| Ping/pong | Built-in | Built-in | Manual |
| Close handshake | Correct per RFC 6455 | Correct | Manual |
| Size | ~15 files, focused | Large framework | ~500 lines to write |
| Proxy support | Since v14.0 | Yes | Manual |

`websockets` is the clear choice: mature, pure Python (no C extensions needed),
provides a synchronous client API that fits the threaded plugin architecture,
and has zero transitive dependencies.

### Sync API Usage

```python
from websockets.sync.client import connect

ws = connect("wss://example.com/ws", additional_headers=headers)
msg = ws.recv(timeout=5.0)  # Blocks up to 5s
ws.send("hello")
ws.close()
```

The sync API runs in a regular thread — no asyncio event loop needed. This
keeps the plugin consistent with the threaded architecture used by the Webhook
and Interactive Shell plugins.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `websockets` not installed | All tools return install instruction error |
| Handshake failure (bad URL) | `ws_connect` returns error with diagnostic |
| Handshake failure (auth rejected) | `ws_connect` returns 401/403 details |
| Connection timeout on handshake | `ws_connect` returns timeout error |
| Remote close during operation | System message injected, auto-reconnect if enabled |
| Send on closed connection | `ws_send` returns error suggesting reconnect |
| Unknown connection_id | Error result listing valid connection IDs |
| Buffer overflow | Oldest messages evicted (FIFO), `messages_dropped` counter incremented |
| Max connections reached | `ws_connect` returns error listing active connections |
| Reader thread crashes | Logged, status changes to `"error"`, system message injected |
| TaskEventBus unavailable | Messages lost, logged as warning |
| Busy turn during delivery | subscribeToTasks → inject_prompt() queues in session message queue, delivered after current turn |

## Lifecycle & Cleanup

1. **Plugin initialize** — config loaded and merged, no connections opened.
2. **Model subscribes** — `subscribeToTasks(event_types=["external_event"])`.
3. **`ws_connect` call** — connection established, initial messages captured and
   returned, reader thread started publishing to `TaskEventBus`.
4. **Session active** — connections running, messages published to event bus,
   delivered via `subscribeToTasks` → `inject_prompt()`, model responding
   with `ws_send` as needed.
5. **Session GC** — old turns with processed messages are garbage-collected;
   connections persist and continue delivering new messages.
6. **`ws_close` call** — individual connection closed gracefully.
7. **Session stop / shutdown** — `shutdown()` closes all connections, stops all
   reader threads.

Reader threads are daemon threads — they don't prevent process exit if the
session crashes without calling `shutdown()`.

## Testing Strategy

### Unit Tests

- **Config loading** — precedence merging, variable expansion, named connections.
- **Connection state machine** — status transitions, lock safety.
- **Message formatting** — injected prompt format, multi-message batches,
  binary message base64 encoding.
- **Tool executors** — connect/send/close/status return correct structures.
- **Dependency gating** — correct error when `websockets` is not installed.

### Integration Tests

- **Echo server** — start a local WebSocket echo server (using `websockets`
  `serve`), connect, verify initial messages returned.
- **Event bus delivery** — connect to echo server, send message, verify it's
  published to `TaskEventBus` as `EXTERNAL_EVENT`.
- **Natural batching** — send 100 rapid messages while session is busy,
  verify they queue up and arrive together when turn completes.
- **Idle wakeup** — session is idle, message arrives, verify
  `subscribeToTasks` callback triggers a new turn via `inject_prompt()`.
- **Reconnection** — connect, kill server, verify reconnect with backoff,
  restart server, verify push delivery resumes.
- **Concurrent connections** — open multiple connections, verify independent
  delivery.
- **Binary messages** — send/receive binary frames, verify base64 encoding.
- **Subprotocol negotiation** — verify `graphql-ws` negotiation.
- **Connection limit** — attempt to exceed `max_connections`, verify error.

### Test Echo Server

Tests use `websockets.sync.server` (or `websockets.serve` with a thread) as
a local echo server — no external dependencies, no network access needed:

```python
from websockets.sync.server import serve

def echo_handler(ws):
    for msg in ws:
        ws.send(msg)

with serve(echo_handler, "127.0.0.1", 0) as server:
    port = server.socket.getsockname()[1]
    # Run tests against ws://127.0.0.1:{port}
```

## Future Extensions

- **Message filtering** — JSONPath or jq-like expressions to filter messages
  before delivery (reduce noise for high-volume streams).
- **Binary protocol support** — structured binary message parsing (Protocol
  Buffers, MessagePack) with schema registration.
- **Connection groups** — connect to multiple related endpoints with a single
  merged delivery stream.
- **Shared event bus tools** — extract `subscribeToTasks` / `pollForTasks`
  from the todo plugin into a standalone event bus plugin, so any plugin
  can use them without depending on the todo plugin.
- **Shared connections** — multiple sessions sharing a single WebSocket
  connection (via event bus fan-out) to avoid duplicate connections to the same
  endpoint.
- **Rate-limited sending** — configurable send rate limits to avoid being
  banned by external services.
- **Adaptive buffering** — dynamically adjust `max_buffer_size` based on
  stream velocity and model processing speed.
