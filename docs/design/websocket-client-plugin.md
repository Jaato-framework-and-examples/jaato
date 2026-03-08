# WebSocket Client Plugin — Outbound Real-Time Connections

## Overview

The WebSocket Client plugin enables agent sessions to **connect to external
WebSocket endpoints** and maintain persistent, bidirectional communication
channels. Where the Webhook plugin is an inbound HTTP listener (external
services push events *to* the agent), this plugin is an outbound connector
(the agent reaches out *to* external services).

Incoming messages are **pushed directly into the session** via
`JaatoSession.inject_prompt()` — the model doesn't poll. It connects once,
then goes fully idle. When a message arrives, `inject_prompt()` wakes the
model for a new turn. Messages that arrive while the model is busy queue
up naturally and are delivered together when the current turn completes.

```
Agent Session                  WebSocket Client Plugin           External Service
(daemon or interactive)        (outbound connections)            (Slack, Binance, K8s, ...)
        │                                │                              │
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
        │                                │◄────── message ─────────────┤
        │                                │                              │
        │                    inject_prompt(batched messages)            │
        │◄───────────────────────────────┤                              │
        │  (new turn triggered)          │                              │
        │  model processes messages      │                              │
        │                                │                              │
        │  ws_send(data={...})           │                              │
        ├───────────────────────────────►│  ────── message ────────────►│
        │                                │                              │
        │         (session idle again)   │                              │
        │              ...sleeps...      │                              │
        │                                │◄────── message ─────────────┤
        │◄───────────────────────────────┤  inject_prompt()            │
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
| **Delivery model** | Event bus (`subscribeToTasks` push / `pollForTasks` poll) + `webhook_poll` fallback | Push via `inject_prompt()` |
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

## Message Delivery: `inject_prompt()` with Natural Batching

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

### Why Direct `inject_prompt()` Instead of the Event Bus?

The `TaskEventBus` already supports push delivery via `subscribeToTasks` —
the todo plugin's subscription callback uses `inject_prompt()` to push events
into the subscriber's session (see `todo/plugin.py:1616`). The webhook plugin
publishes to this bus, so a model *could* use `subscribeToTasks` to receive
webhook events via push.

However, for the WebSocket plugin, going through the event bus adds unnecessary
indirection. The plugin already has a reader thread per connection — it can
call `inject_prompt()` directly without publishing to an intermediate bus and
requiring the model to set up a subscription. The event bus is still used as
a **secondary** delivery channel for cross-agent fan-out, but primary delivery
is direct `inject_prompt()`.

This also avoids coupling to the todo plugin's `subscribeToTasks` tool — the
WebSocket plugin works even if the todo plugin is not loaded.

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

### The Right Tool: `inject_prompt()`

`inject_prompt()` is designed exactly for this: waking a dormant session and
starting a new turn from outside the normal chat loop. It's what subagents use
to send results back to parent sessions, and it's what the WebSocket plugin
uses to deliver messages:

```python
def _on_message_received(self, connection_id: str, message: IncomingMessage):
    """Called by reader thread when a WebSocket message arrives.

    If the session is idle, inject_prompt() triggers a new model turn
    immediately. If the session is busy (mid-turn), the message is
    queued and delivered when the current turn completes.

    Messages that arrive in rapid succession while the session is busy
    are naturally batched — they queue up and the model sees them all
    together when the current turn finishes.
    """
    formatted = self._format_message(connection_id, message)
    self._session.inject_prompt(
        text=formatted,
        source_id=f"ws_client:{connection_id}",
        source_type=SourceType.SYSTEM,
    )
```

**What the model sees** (injected as a system-like message):

```
[WebSocket ws-1 (wss://stream.binance.com)] 3 messages received:

1. {"e":"trade","s":"BTCUSDT","p":"67420.50","q":"0.123","T":1709910600123}
2. {"e":"trade","s":"BTCUSDT","p":"67421.00","q":"0.456","T":1709910600456}
3. {"e":"trade","s":"ETHUSDT","p":"3421.80","q":"1.200","T":1709910600789}
```

The model processes the messages and can respond with `ws_send` if needed.
When it finishes its turn, the session goes idle. If more WebSocket messages
arrive later, `inject_prompt()` wakes the model for a new turn.

### Natural Batching (No Custom Batcher Needed)

High-volume streams (market data, log tails) could produce hundreds of
messages per second. But `inject_prompt()` + the session's message queue
provide natural batching without any custom timer logic:

1. First message arrives → `inject_prompt()` → new turn starts.
2. While model is processing, more messages arrive → each calls
   `inject_prompt()` → messages queue up in the session's message queue.
3. Model finishes its turn → drains the queue → sees all queued messages
   together.
4. Model processes the batch → finishes → goes idle.
5. Next message arrives → `inject_prompt()` → cycle repeats.

**Behavior by stream velocity:**

| Stream Type | Messages/sec | Behavior |
|-------------|-------------|----------|
| Chat (Slack) | 0.1–1 | Near-instant delivery (single messages) |
| Moderate events | 1–10 | Small natural batches (2–5 messages) |
| Market data | 10–100 | Larger batches accumulate during model processing |
| Firehose (logs) | 100+ | Large batches, oldest dropped if queue exceeds `max_buffer_size` |

The key insight: the model's own processing time acts as the batching window.
Fast streams batch more, slow streams deliver individually. No debounce
timers or batch size configs needed.

### Why Not a Custom `MessageBatcher`?

An earlier version of this design proposed a custom `MessageBatcher` with
debounce timers, `max_batch_size`, and `max_wait_ms`. This is unnecessary
because the session's message queue already provides the right batching
behavior. Adding custom timers on top would just add latency to low-volume
streams (the 200ms debounce delay) while providing no benefit for high-volume
streams (where the model's processing time is already the batching window).

### First-Message Auth Bootstrapping

Some protocols (Discord Gateway, GraphQL subscriptions) require the model to
read the first message from the server and respond with an auth payload.

The `ws_connect` tool result includes **initial messages** captured during a
brief startup window before switching to `inject_prompt()` delivery:

```python
def _execute_connect(self, args):
    """Connect and return initial messages from the server.

    After the WebSocket handshake completes, waits up to
    initial_read_timeout (default 3s) for any server-initiated
    messages (hello, capabilities, auth challenge). These are
    returned directly in the tool result so the model can respond
    with ws_send in the same turn.

    After the initial read, the reader thread switches to
    inject_prompt() delivery for all subsequent messages.
    """
    conn = self._create_connection(args)
    conn.connect()

    # Capture initial messages (hello, auth challenge, etc.)
    initial_messages = conn.drain_initial(timeout=3.0)

    # Start reader thread — messages delivered via inject_prompt()
    conn.start_reader(on_message=self._on_message_received)

    return {
        "connection_id": conn.connection_id,
        "status": "connected",
        "url": conn.url,
        "initial_messages": [self._format_message(m) for m in initial_messages],
        "message": f"Connected. {len(initial_messages)} initial message(s). "
                   "Subsequent messages delivered automatically.",
    }
```

**Discord example flow:**

```
Agent: ws_connect(url="wss://gateway.discord.gg/?v=10&encoding=json")
→ {
    connection_id: "ws-1",
    initial_messages: [
      {type: "text", data: '{"op":10,"d":{"heartbeat_interval":41250}}'}
    ],
    message: "Connected. 1 initial message(s). Subsequent messages delivered automatically."
  }

Agent: ws_send(connection_id="ws-1", data='{"op":2,"d":{"token":"...","intents":513}}')
→ {status: "sent"}

    ... session goes idle ...

    [WebSocket ws-1] 1 message received:    ← inject_prompt() triggers new turn
    1. {"op":0,"t":"READY","d":{...}}

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
├── plugin.py            # WebSocketClientPlugin — connection mgmt + inject_prompt() delivery
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
    connection runs a reader thread that delivers incoming messages to the
    session via inject_prompt(). The model can send messages through open
    connections using ws_send.

    Delivery model:
        - Inbound: push via inject_prompt() — model wakes on message arrival
        - Outbound: explicit via ws_send tool calls
        - The model never polls — it goes fully idle between messages
        - Natural batching: messages that arrive during a busy turn queue up
          and are delivered together when the turn completes

    Lifecycle:
        1. initialize(config) — load and merge config
        2. set_session(session) — receive session reference for inject_prompt()
        3. ws_connect tool call — establish connection, return initial messages,
           start reader thread in inject_prompt() delivery mode
        4. Reader thread calls inject_prompt() on each message
        5. Session wakes (if idle) or queues (if busy) → model processes
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

    def set_session(self, session: "JaatoSession") -> None:
        """Receive session reference for inject_prompt() delivery.

        Called automatically by plugin auto-wiring during configure().
        The session reference is required for push delivery — without it,
        the plugin falls back to buffer-only mode (messages accumulate
        but are never delivered).
        """
        self._session = session

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
server messages (hello, auth challenge). All subsequent messages are
delivered automatically via `inject_prompt()`.

```json
{
  "name": "ws_connect",
  "description": "Connect to a WebSocket endpoint. Returns a connection_id and any initial server messages. After this call, incoming messages are delivered to you automatically — no polling needed. Use ws_send to send messages back.",
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
  "message": "Connected. Incoming messages will arrive as streaming updates."
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

Connection lifecycle events are delivered to the session via
`inject_prompt()` as system messages, so the model sees them as new turns:

| Event | Injected Message |
|-------|-----------------|
| Disconnected | `[WebSocket ws-1] Connection closed by remote (code=1006). Reconnecting...` |
| Reconnected | `[WebSocket ws-1] Reconnected successfully (was disconnected for 8.2s)` |
| Reconnect failed | `[WebSocket ws-1] Reconnection failed after 5 attempts. Use ws_connect to reconnect.` |
| Ping timeout | `[WebSocket ws-1] Connection appears dead (ping timeout). Closing.` |

These are injected with `SourceType.SYSTEM` — they're queued if the session is
busy and delivered when the session becomes idle.

## Event Bus Integration

Like the Webhook plugin, WebSocket events are published to `TaskEventBus` for
cross-agent consumption. The reader thread does both in the message callback:

```python
def _on_message_received(self, connection_id: str, message: IncomingMessage):
    """Called by reader thread when a WebSocket message arrives."""
    # 1. Inject into owning session (primary delivery — wakes idle model)
    formatted = self._format_message(connection_id, message)
    self._session.inject_prompt(
        text=formatted,
        source_id=f"ws_client:{connection_id}",
        source_type=SourceType.SYSTEM,
    )

    # 2. Publish to event bus (secondary — for cross-agent fan-out)
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
            }
        )
        bus.publish(event)
    except Exception:
        logger.debug("TaskEventBus unavailable, skipping cross-agent delivery")
```

This means other sessions can subscribe to WebSocket events via the event bus
(using `pollForTasks`), enabling fan-out patterns (e.g., one connection shared
across multiple agent sessions that each process different message types).

## Reconnection Strategy

WebSocket connections are inherently fragile. The plugin provides configurable
auto-reconnection per connection:

### Exponential Backoff

```python
def _attempt_reconnect(self):
    """Try to reconnect with exponential backoff.

    Attempts: delay = min(base_delay * 2^attempt, max_delay) + jitter
    Jitter: ±25% to prevent thundering herd on shared endpoints.

    System messages about reconnection progress are injected into the
    session via inject_prompt() so the model stays informed.
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
  "system_instructions": "You are a Slack bot connected via RTM WebSocket.\n\nOn startup:\n1. Call ws_connect(name=\"slack\") to connect.\n2. If the initial_messages contain a hello, you're ready.\n3. Incoming Slack messages will be delivered to you automatically.\n4. Parse each message JSON and respond using ws_send.\n5. If you receive a disconnect system message, wait — auto-reconnect is enabled.\n\nYou do not need to poll. Messages arrive as new turns.",
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
  "system_instructions": "You are a market data monitor.\n\nOn startup:\n1. ws_connect(name=\"binance\")\n2. ws_send to subscribe: {\"method\":\"SUBSCRIBE\",\"params\":[\"btcusdt@trade\",\"ethusdt@trade\"],\"id\":1}\n3. Trade data will be delivered to you in batches automatically.\n4. Analyze price movements and maintain a running summary.\n5. Alert on significant moves (>2% in 5 minutes).\n\nYou do not need to poll. Batches of trades arrive as new turns.",
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

### Injection Safety

Messages injected via `inject_prompt()` are prefixed with
`[WebSocket <connection_id>]` to clearly identify their source. The model
can distinguish WebSocket messages from user messages or subagent results.

The formatted injection includes the raw message data as-is — no escaping or
sanitization of the WebSocket payload content. This is intentional: the model
needs to see the exact data to parse protocol-specific formats. The
`[WebSocket ...]` prefix prevents confusion with other message sources.

## Comparison: WebSocket Client vs Interactive Shell

The `interactive_shell` plugin also maintains persistent bidirectional sessions.
Key differences:

| Dimension | Interactive Shell | WebSocket Client |
|-----------|------------------|-----------------|
| **Target** | Local processes (PTY) | Remote services (network) |
| **Transport** | Pseudo-terminal (stdin/stdout) | WebSocket (TCP + TLS) |
| **Output format** | Raw terminal output (ANSI stripped) | Structured messages (JSON, text) |
| **Delivery** | Must use `shell_read`/`shell_input` | Push via `inject_prompt()` |
| **Auth** | OS-level (user permissions) | Application-level (tokens, headers) |
| **Reconnection** | N/A (process is dead) | Auto-reconnect with backoff |
| **Use case** | REPLs, debuggers, SSH | APIs, event streams, chat bots |

There is no overlap — they serve completely different integration patterns.

**Note:** The webhook plugin already supports push delivery via
`subscribeToTasks(event_types=["external_event"])`, which uses
`inject_prompt()` under the hood. However, the webhook plugin's own system
instructions still direct the model to use `pollForTasks`. A future
enhancement could update the webhook plugin to recommend `subscribeToTasks`
as the primary delivery mechanism (or add direct `inject_prompt()` delivery
like this plugin does), fully unifying the push model across both plugins.

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
| Session reference missing | Messages buffer but are never delivered (logged as warning) |
| inject_prompt() during busy turn | Queued by session's message queue, delivered after current turn |

## Lifecycle & Cleanup

1. **Plugin initialize** — config loaded and merged, no connections opened.
2. **set_session()** — session reference stored for `inject_prompt()` delivery.
3. **`ws_connect` call** — connection established, initial messages captured and
   returned, reader thread started in `inject_prompt()` delivery mode.
4. **Session active** — connections running, messages pushed via `inject_prompt()`,
   model responding with `ws_send` as needed.
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
- **Push delivery** — connect to echo server, send message, verify it's
  delivered via mock `inject_prompt()` call.
- **Natural batching** — send 100 rapid messages while session is busy,
  verify they queue up and arrive together when turn completes.
- **Idle wakeup** — session is idle, message arrives, verify
  `inject_prompt()` triggers a new turn.
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
- **Webhook plugin direct push** — add direct `inject_prompt()` delivery to
  the webhook plugin (like this plugin), bypassing the event bus indirection
  for single-session use cases.
- **Shared connections** — multiple sessions sharing a single WebSocket
  connection (via event bus fan-out) to avoid duplicate connections to the same
  endpoint.
- **Rate-limited sending** — configurable send rate limits to avoid being
  banned by external services.
- **Adaptive buffering** — dynamically adjust `max_buffer_size` based on
  stream velocity and model processing speed.
