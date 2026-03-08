# WebSocket Client Plugin — Outbound Real-Time Connections

## Overview

The WebSocket Client plugin enables agent sessions to **connect to external
WebSocket endpoints** and maintain persistent, bidirectional communication
channels. Where the Webhook plugin is an inbound HTTP listener (external
services push events *to* the agent), this plugin is an outbound connector
(the agent reaches out *to* external services).

Incoming messages are delivered through the framework's existing **streaming
tool infrastructure** (`StreamingCapable` + `StreamManager`). The model calls
`ws_connect` which starts an infinite stream — incoming WebSocket messages
become `StreamChunk`s that the `StreamManager` delivers when the model is idle.
No polling, no custom batching — just the same mechanism that `grep_content:stream`
and `glob_files:stream` already use.

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
        │  {connection_id, initial_msgs} │                              │
        │◄───────────────────────────────┤                              │
        │                                │  register stream with        │
        │                                │  StreamManager               │
        │                                │                              │
        │         (model idle)           │◄────── message ─────────────┤
        │                                │◄────── message ─────────────┤
        │                                │  → StreamChunk → StreamManager
        │                                │                              │
        │  <streaming_updates>           │                              │
        │  [ws-1] 2 messages             │                              │
        │◄───────────────────────────────┤  (idle-time delivery)       │
        │  model processes messages      │                              │
        │                                │                              │
        │  ws_send(data={...})           │                              │
        ├───────────────────────────────►│  ────── message ────────────►│
        │                                │                              │
        │         (model idle again)     │◄────── message ─────────────┤
        │                                │  → StreamChunk               │
        │  <streaming_updates>           │                              │
        │◄───────────────────────────────┤  (idle-time delivery)       │
        │  (model processes)             │                              │
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
| **Delivery model** | Event bus (`pollForTasks`) + `webhook_poll` fallback | Push via `StreamManager` (idle-time delivery) |
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

1. **Reuse the streaming infrastructure.** The framework's `StreamingCapable`
   protocol and `StreamManager` already solve incremental delivery to the
   model — idle-time chunk batching, background collection, `dismiss_stream`.
   A WebSocket connection is just an infinite stream. No custom delivery
   layer needed.
2. **Model drives outbound.** The model calls `ws_connect` and `ws_send`.
   The plugin never sends data without the model's explicit instruction.
   Inbound delivery is the only automatic behavior.
3. **Connections are explicit.** The model connects to specific URLs with
   specific parameters. No magic auto-connections. Pre-configured named
   connections in config are a convenience, not a requirement.
4. **Reconnection is opt-in.** Dropped connections surface as stream
   chunks. Auto-reconnect is configurable per-connection but defaults to
   off — the model should understand when connections drop.
5. **Optional dependency.** The plugin requires `websockets` but degrades
   gracefully — importing the plugin without the dependency installed
   produces a clear error message, not a crash.

## Streaming Delivery via `StreamManager`

### Why Not the Webhook Plugin's Delivery Model?

The webhook plugin delivers events via `webhook_subscribe` → `TaskEventBus` →
`pollForTasks` (with `webhook_poll` as a direct fallback). Both paths require
the model to actively poll — either the event bus or the subscription buffer.
This works for discrete HTTP events but has drawbacks for high-frequency
persistent streams:

- **Wasted turns.** Empty polls consume model turns and tokens for nothing.
- **Latency.** Messages wait in a buffer until the model's next poll call.
- **Fragile loop.** If the model forgets to poll (e.g., after a GC cycle that
  summarizes away the loop instructions), messages pile up silently.
- **Unnatural for streams.** WebSocket connections are continuous — the model
  should be doing application work (processing messages), not infrastructure
  work (polling in a loop).

### Why Not a Custom Batcher?

An earlier version of this design proposed a custom `MessageBatcher` with
debounce timers and `inject_prompt()`. But the framework already has
`StreamManager` + `StreamingCapable` — the same infrastructure that powers
`grep_content:stream` and `glob_files:stream`. A WebSocket connection is just
an infinite stream of chunks. Building a custom delivery layer would duplicate:

- Idle-time chunk delivery (already in `StreamManager`)
- Background collection in a daemon thread (already in `StreamManager`)
- Batching of multiple chunks per delivery (already in `StreamManager`)
- A dismiss mechanism for the model to stop receiving (already `dismiss_stream`)
- Thread-safe chunk tracking (`StreamState.chunks_delivered`)

### How It Works

The WebSocket client plugin implements `StreamingCapable`. When the model calls
`ws_connect`, the plugin:

1. Establishes the WebSocket connection
2. Captures initial server messages (hello, auth challenge) and returns them
   directly in the tool result
3. Registers an **infinite stream** with `StreamManager`
4. The reader thread yields `StreamChunk`s as WebSocket messages arrive

The `StreamManager` handles the rest — collecting chunks in the background,
delivering them when the model is idle via `<hidden><streaming_updates>`,
and supporting `dismiss_stream` to stop.

```python
class WebSocketClientPlugin(StreamingCapable):
    """Implements StreamingCapable for WebSocket message delivery.

    Each active connection registers as a named stream with StreamManager.
    The stream's async generator bridges the synchronous WebSocket recv
    loop to async chunk yields via a threading-to-asyncio queue.
    """

    def supports_streaming(self, tool_name: str) -> bool:
        return tool_name == "ws_connect"

    async def execute_streaming(
        self, tool_name: str, arguments: Dict[str, Any],
        on_chunk: Optional[ChunkCallback] = None
    ) -> AsyncIterator[StreamChunk]:
        """Async generator that yields WebSocket messages as StreamChunks.

        This generator runs for the lifetime of the connection. It bridges
        the synchronous reader thread (which calls ws.recv()) to the async
        StreamManager via an asyncio.Queue.

        Lifecycle:
            1. Handshake + initial messages returned via tool result
            2. Reader thread starts, puts messages into async queue
            3. This generator awaits queue.get() and yields StreamChunks
            4. On disconnect: yields a system chunk, then returns (or
               waits for reconnect and resumes yielding)
            5. On dismiss_stream or ws_close: generator returns
        """
        conn = self._connections[arguments["connection_id"]]
        queue = conn._async_queue  # asyncio.Queue bridging reader thread

        while conn.status in ("connected", "reconnecting"):
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield StreamChunk(
                    content=self._format_ws_message(conn.connection_id, msg),
                    chunk_type="ws_message",
                    metadata={
                        "connection_id": conn.connection_id,
                        "message_type": msg.type,
                    },
                )
            except asyncio.TimeoutError:
                continue  # Check connection status
            except asyncio.CancelledError:
                break  # dismiss_stream or shutdown

    def get_streaming_tool_names(self) -> List[str]:
        return ["ws_connect"]
```

**What the model sees** (delivered by `StreamManager` when idle):

```xml
<hidden><streaming_updates>
<stream id="ws-1" tool="ws_connect" status="streaming" new_chunks="3">
[ws-1] {"e":"trade","s":"BTCUSDT","p":"67420.50","q":"0.123"}
[ws-1] {"e":"trade","s":"BTCUSDT","p":"67421.00","q":"0.456"}
[ws-1] {"e":"trade","s":"ETHUSDT","p":"3421.80","q":"1.200"}
</stream>
</streaming_updates></hidden>
```

The model processes the messages and can respond with `ws_send` if needed.
When it finishes its turn, the session goes idle. If more WebSocket messages
have arrived, `StreamManager` delivers the next batch.

### Streaming Lifecycle for WebSocket

The key difference from finite streams (grep, glob) is that a WebSocket stream
**never completes on its own**. It continues yielding chunks until:

| Trigger | Stream Status | What Happens |
|---------|--------------|--------------|
| Remote close | `COMPLETED` | Final system chunk, generator returns |
| `ws_close()` by model | `COMPLETED` | Generator returns |
| `dismiss_stream` by model | `DISMISSED` | Generator cancelled, connection stays open |
| Plugin `shutdown()` | `FAILED` | Generator cancelled, connection closed |
| Reconnect failure | `FAILED` | Final system chunk with error |

**`dismiss_stream` vs `ws_close`:** The model can dismiss the stream (stop
receiving updates) while keeping the connection open for sending. This is
useful for connections where the model only needs to send, not receive (e.g.,
after initial setup). To fully disconnect, use `ws_close`.

### Natural Batching via `StreamManager`

`StreamManager` already handles batching naturally through its idle-time
delivery model:

1. Chunks accumulate while the model is busy (processing a turn, calling tools)
2. When the model goes idle, `StreamManager` delivers all pending chunks at once
3. The streaming continuation loop runs up to 20 iterations per idle period
4. For daemon sessions (`max_turns=0`), this creates a natural
   process → idle → receive → process cycle

This means high-volume streams (market data, logs) are automatically batched —
if 50 messages arrive while the model processes the previous batch, they're
all delivered together in the next idle window. No custom debounce timers
needed.

### First-Message Auth Bootstrapping

Some protocols (Discord Gateway, GraphQL subscriptions) require the model to
read the first message from the server and respond with an auth payload.

The `ws_connect` tool result includes **initial messages** captured during a
brief startup window before the stream is registered with `StreamManager`:

```python
def _execute_connect(self, args):
    """Connect, capture initial messages, then register stream.

    After the WebSocket handshake completes, waits up to
    initial_read_timeout (default 3s) for any server-initiated
    messages (hello, capabilities, auth challenge). These are
    returned directly in the tool result so the model can respond
    with ws_send in the same turn.

    After the initial read, registers the connection as a stream
    with StreamManager for ongoing delivery.
    """
    conn = self._create_connection(args)
    conn.connect()

    # Capture initial messages (hello, auth challenge, etc.)
    initial_messages = conn.drain_initial(timeout=3.0)

    # Start reader thread — messages go to async queue
    conn.start_reader()

    # Register infinite stream with StreamManager
    self._stream_manager.start_stream(
        stream_id=conn.connection_id,
        tool_name="ws_connect",
        generator=self._create_stream_generator(conn),
    )

    return {
        "connection_id": conn.connection_id,
        "status": "connected",
        "url": conn.url,
        "initial_messages": [self._format_message(m) for m in initial_messages],
        "message": f"Connected. {len(initial_messages)} initial message(s). "
                   "Subsequent messages delivered via streaming updates.",
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
    message: "Connected. 1 initial message(s). Subsequent messages delivered via streaming updates."
  }

Agent: ws_send(connection_id="ws-1", data='{"op":2,"d":{"token":"...","intents":513}}')
→ {status: "sent"}

    ... model goes idle, StreamManager delivers next batch ...

    <streaming_updates>
    <stream id="ws-1" tool="ws_connect" status="streaming" new_chunks="1">
    [ws-1] {"op":0,"t":"READY","d":{...}}
    </stream>
    </streaming_updates>

Agent: (processes READY event, goes idle again)

    ... StreamManager delivers when next messages arrive ...
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
| `max_buffer_size` | `1000` | Per-connection async queue depth (FIFO eviction) |
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
├── plugin.py            # WebSocketClientPlugin (StreamingCapable) — connection mgmt + streaming
├── connection.py        # WebSocketConnection — per-connection state + reader thread
├── config.py            # WebSocketClientConfig dataclass, config loading/merging
└── tests/
    ├── test_plugin.py
    ├── test_connection.py
    └── test_config.py
```

### Plugin Class

```python
class WebSocketClientPlugin(StreamingCapable):
    """Outbound WebSocket client plugin for real-time bidirectional communication.

    Manages persistent WebSocket connections to external services. Each
    connection runs a reader thread that feeds messages into an async queue.
    The plugin implements StreamingCapable, so StreamManager collects chunks
    from the async queue and delivers them to the model during idle windows.

    Delivery model:
        - Inbound: StreamManager delivers chunks when model is idle
        - Outbound: explicit via ws_send tool calls
        - The model never polls — chunks arrive as streaming updates
        - Model can call dismiss_stream to stop receiving from a connection

    Lifecycle:
        1. initialize(config) — load and merge config
        2. set_session(session) — receive session reference (auto-wired)
        3. ws_connect tool call — establish connection, return initial messages,
           register infinite stream with StreamManager
        4. Reader thread yields StreamChunks via async queue → StreamManager
        5. StreamManager delivers chunks when model is idle
        6. Model processes messages, optionally calls ws_send
        7. ws_close or shutdown() — close connections, cancel streams

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
        """Receive session reference (auto-wired during configure())."""
        self._session = session

    def supports_streaming(self, tool_name: str) -> bool:
        """ws_connect is the only streaming tool."""
        return tool_name == "ws_connect"

    def get_streaming_tool_names(self) -> List[str]:
        return ["ws_connect"]

    async def execute_streaming(
        self, tool_name: str, arguments: Dict[str, Any],
        on_chunk: Optional[ChunkCallback] = None
    ) -> AsyncIterator[StreamChunk]:
        """Async generator yielding WebSocket messages as StreamChunks.

        Bridges the synchronous reader thread to the async StreamManager
        via an asyncio.Queue. Runs for the lifetime of the connection.
        """
        ...

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for WebSocket operations."""
        ...

    def get_executors(self) -> Dict[str, Any]:
        """Return tool executors."""
        ...

    def shutdown(self) -> None:
        """Close all connections, cancel streams, stop reader threads."""
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
  "description": "Connect to a WebSocket endpoint. Returns a connection_id and any initial server messages. After this call, incoming messages are delivered as streaming updates — no polling needed. Use ws_send to send messages back. Use dismiss_stream to stop receiving.",
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
reader thread and an async queue for bridging to `StreamManager`:

```python
class WebSocketConnection:
    """A single WebSocket connection with a background reader thread.

    Lifecycle:
        1. connect(url, headers, subprotocols) — handshake, start reader
        2. drain_initial(timeout) — capture server hello messages
        3. start_reader() — begin recv loop, push to async queue
        4. StreamManager's async generator awaits queue.get() → StreamChunk
        5. send(data) — send message through the connection
        6. close() — send close frame, stop reader, clean up

    The reader thread puts messages into an asyncio.Queue. The plugin's
    execute_streaming() async generator awaits that queue and yields
    StreamChunks to StreamManager. This bridges sync (WebSocket recv)
    to async (StreamManager collection).

    Thread safety:
        - _async_queue (asyncio.Queue) is thread-safe for put_nowait/get
        - _lock (threading.Lock) protects status transitions
        - _send_lock (threading.Lock) serializes writes
    """

    def __init__(self, connection_id, config, async_queue, on_status_change=None):
        self.connection_id = connection_id
        self.url = None
        self.status = "idle"            # idle → connecting → connected → disconnected → closed
        self._ws = None                 # websockets.sync.client.ClientConnection
        self._reader_thread = None
        self._async_queue = async_queue # asyncio.Queue — bridge to StreamManager
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
pushes messages into the async queue:

```python
def _reader_loop(self):
    """Background thread: read messages and push to async queue.

    Runs until connection closes or stop is requested. Messages are
    put into the asyncio.Queue where the plugin's execute_streaming()
    generator awaits them. On unexpected close with reconnect enabled,
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
            try:
                self._async_queue.put_nowait(msg)
            except asyncio.QueueFull:
                self._stats.messages_dropped += 1  # Buffer overflow
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

Connection lifecycle events are pushed into the async queue as system-type
messages, so they appear alongside regular WebSocket messages in the
streaming updates:

| Event | StreamChunk Content |
|-------|-------------------|
| Disconnected | `[ws-1] ⚠ Connection closed by remote (code=1006). Reconnecting...` |
| Reconnected | `[ws-1] ✓ Reconnected successfully (was disconnected for 8.2s)` |
| Reconnect failed | `[ws-1] ✗ Reconnection failed after 5 attempts. Use ws_connect to reconnect.` |
| Ping timeout | `[ws-1] ⚠ Connection appears dead (ping timeout). Closing.` |

These are delivered through the same `StreamManager` pipeline as regular
messages — the model sees them in `<streaming_updates>` blocks.

## Event Bus Integration

In addition to `StreamManager` delivery to the owning session, WebSocket
messages are published to `TaskEventBus` for cross-agent consumption. This
happens in the `execute_streaming()` generator as each chunk is yielded:

```python
async def execute_streaming(self, tool_name, arguments, on_chunk=None):
    conn = self._connections[arguments["connection_id"]]

    while conn.status in ("connected", "reconnecting"):
        try:
            msg = await asyncio.wait_for(conn._async_queue.get(), timeout=1.0)

            # Publish to event bus for cross-agent fan-out
            self._publish_to_event_bus(conn.connection_id, msg)

            yield StreamChunk(
                content=self._format_ws_message(conn.connection_id, msg),
                chunk_type="ws_message",
            )
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
```

This means other sessions can subscribe to WebSocket events via the event bus,
enabling fan-out patterns (e.g., one connection shared across multiple agent
sessions that each process different message types).

## Reconnection Strategy

WebSocket connections are inherently fragile. The plugin provides configurable
auto-reconnection per connection:

### Exponential Backoff

```python
def _attempt_reconnect(self):
    """Try to reconnect with exponential backoff.

    Attempts: delay = min(base_delay * 2^attempt, max_delay) + jitter
    Jitter: ±25% to prevent thundering herd on shared endpoints.

    System messages about reconnection progress are pushed into the
    async queue so they appear in streaming updates.
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
  "system_instructions": "You are a Slack bot connected via RTM WebSocket.\n\nOn startup:\n1. Call ws_connect(name=\"slack\") to connect.\n2. If the initial_messages contain a hello, you're ready.\n3. Incoming Slack messages arrive as streaming updates — no polling needed.\n4. Parse each message JSON and respond using ws_send.\n5. If you see a disconnect system message in streaming updates, wait — auto-reconnect is enabled.",
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
  "system_instructions": "You are a market data monitor.\n\nOn startup:\n1. ws_connect(name=\"binance\")\n2. ws_send to subscribe: {\"method\":\"SUBSCRIBE\",\"params\":[\"btcusdt@trade\",\"ethusdt@trade\"],\"id\":1}\n3. Trade data arrives as streaming updates — no polling needed.\n4. Analyze price movements and maintain a running summary.\n5. Alert on significant moves (>2% in 5 minutes).",
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
- `max_buffer_size` limits async queue depth per connection (default 1000, FIFO)
- `StreamManager` naturally batches chunks — no single delivery overwhelms the model
- Reconnection attempts are bounded (`reconnect_max_attempts`)
- Ping timeouts detect dead connections
- `dismiss_stream` lets the model stop receiving without closing the connection

### Injection Safety

Streaming updates are delivered inside `<hidden><streaming_updates>` tags
with a `<stream>` wrapper identifying the connection. The `[ws-<id>]` prefix
on each chunk clearly identifies the message source.

The chunk content includes the raw WebSocket message data as-is — no escaping
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
| **Delivery** | Must use `shell_read`/`shell_input` | Push via `StreamManager` |
| **Auth** | OS-level (user permissions) | Application-level (tokens, headers) |
| **Reconnection** | N/A (process is dead) | Auto-reconnect with backoff |
| **Use case** | REPLs, debuggers, SSH | APIs, event streams, chat bots |

There is no overlap — they serve completely different integration patterns.

**Note:** The streaming delivery model used here could retroactively benefit
the Webhook plugin as well. Currently the webhook plugin delivers via
`TaskEventBus` + `pollForTasks` (with `webhook_poll` as fallback) — both
require the model to actively poll. A future enhancement could make the
Webhook plugin implement `StreamingCapable`, adding `StreamManager` delivery
as a third option — unifying the delivery model across both plugins and
eliminating the need for poll loops entirely.

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
| Remote close during operation | System chunk pushed to queue, auto-reconnect if enabled |
| Send on closed connection | `ws_send` returns error suggesting reconnect |
| Unknown connection_id | Error result listing valid connection IDs |
| Async queue full | Oldest messages dropped (FIFO), `messages_dropped` counter incremented |
| Max connections reached | `ws_connect` returns error listing active connections |
| Reader thread crashes | Logged, status changes to `"error"`, system chunk pushed |
| Model busy during delivery | `StreamManager` holds chunks until model is idle (built-in) |
| `dismiss_stream` called | Stream generator cancelled, connection stays open for `ws_send` |

## Lifecycle & Cleanup

1. **Plugin initialize** — config loaded and merged, no connections opened.
2. **set_session()** — session reference stored (auto-wired).
3. **`ws_connect` call** — connection established, initial messages returned,
   reader thread started, infinite stream registered with `StreamManager`.
4. **Session active** — reader thread pushes to async queue, `StreamManager`
   delivers chunks when model is idle, model responds with `ws_send`.
5. **Session GC** — old turns with processed streaming updates are garbage-collected;
   connections and streams persist.
6. **`dismiss_stream`** — model stops receiving from a connection (connection
   stays open for `ws_send`).
7. **`ws_close` call** — connection closed, stream completed.
8. **Session stop / shutdown** — `shutdown()` closes all connections, cancels
   all streams, stops reader threads.

Reader threads are daemon threads — they don't prevent process exit if the
session crashes without calling `shutdown()`.

## Testing Strategy

### Unit Tests

- **Config loading** — precedence merging, variable expansion, named connections.
- **Connection state machine** — status transitions, lock safety.
- **Async queue bridge** — sync put_nowait from reader thread, async get from
  streaming generator.
- **StreamChunk formatting** — connection prefix, binary base64, system messages.
- **Tool executors** — connect/send/close/status return correct structures.
- **Dependency gating** — correct error when `websockets` is not installed.

### Integration Tests

- **Echo server** — start a local WebSocket echo server (using `websockets`
  `serve`), connect, verify initial messages returned.
- **Streaming delivery** — connect to echo server, send message, verify
  `StreamChunk` yielded by `execute_streaming()`.
- **Natural batching** — send 100 rapid messages, verify `StreamManager`
  delivers them in batches when model goes idle (no custom batcher needed).
- **Reconnection** — connect, kill server, verify reconnect with backoff,
  restart server, verify stream resumes yielding chunks.
- **Concurrent connections** — open multiple connections, verify independent
  streams.
- **Binary messages** — send/receive binary frames, verify base64 encoding.
- **Subprotocol negotiation** — verify `graphql-ws` negotiation.
- **Connection limit** — attempt to exceed `max_connections`, verify error.
- **dismiss_stream** — dismiss stream, verify generator cancelled, verify
  connection still open for `ws_send`.

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
  merged stream.
- **Webhook plugin streaming mode** — make the Webhook plugin implement
  `StreamingCapable` to replace `webhook_poll` with `StreamManager` delivery.
- **Shared connections** — multiple sessions sharing a single WebSocket
  connection (via event bus fan-out) to avoid duplicate connections to the same
  endpoint.
- **Rate-limited sending** — configurable send rate limits to avoid being
  banned by external services.
- **Adaptive streaming** — dynamically adjust `StreamManager` delivery
  frequency based on stream velocity and model processing speed.
