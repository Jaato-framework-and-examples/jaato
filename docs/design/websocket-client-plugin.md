# WebSocket Client Plugin — Outbound Real-Time Connections

## Overview

The WebSocket Client plugin enables agent sessions to **connect to external
WebSocket endpoints** and maintain persistent, bidirectional communication
channels. Where the Webhook plugin is an inbound HTTP listener (external
services push events *to* the agent), this plugin is an outbound connector
(the agent reaches out *to* external services).

A daemon session using this plugin connects to one or more WebSocket endpoints,
receives real-time messages, and can send messages back — enabling use cases
like chat bots, market data monitoring, IoT control, and log tailing.

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
        │  ws_poll(connection_id="ws-1") │                              │
        ├───────────────────────────────►│  (reader thread buffers      │
        │                                │   incoming frames)           │
        │                                │◄────── message ─────────────┤
        │                                │◄────── message ─────────────┤
        │  {messages: [...]}             │                              │
        │◄───────────────────────────────┤                              │
        │                                │                              │
        │  ws_send(data={...})           │                              │
        ├───────────────────────────────►│  ────── message ────────────►│
        │                                │                              │
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

1. **Same mental model as webhooks.** The model already knows
   subscribe → poll → process → poll. WebSocket tools follow the same
   pattern: connect → poll → process → poll. Bidirectional send is the
   only new concept.
2. **Model drives the loop.** The plugin doesn't inject messages or force
   turns. The model calls `ws_connect`, then loops on `ws_poll`, and
   decides when to `ws_send`. The system prompt instructs the behavior.
3. **Connections are explicit.** The model connects to specific URLs with
   specific parameters. No magic auto-connections. Pre-configured named
   connections in config are a convenience, not a requirement.
4. **Reconnection is opt-in.** Dropped connections surface as events the
   model can see. Auto-reconnect is configurable per-connection but
   defaults to off — the model should understand when connections drop.
5. **Optional dependency.** The plugin requires `websockets` but degrades
   gracefully — importing the plugin without the dependency installed
   produces a clear error message, not a crash.

## Configuration

### Config File: `websocket.json`

```json
{
  "max_connections": 4,
  "max_buffer_size": 1000,
  "default_read_timeout": 30,
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
| `default_read_timeout` | `30` | Default `ws_poll` timeout in seconds |
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
├── plugin.py            # WebSocketClientPlugin — tool plugin with connection mgmt
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
    connection runs a reader thread that buffers incoming messages for the
    model to poll. The model can also send messages through open connections.

    Lifecycle:
        1. initialize(config) — load and merge config
        2. set_workspace_path(path) — resolve workspace-relative config paths
        3. ws_connect tool call — establish WebSocket connection, start reader
        4. ws_poll / ws_send — model-driven interaction loop
        5. ws_close or shutdown() — close connections, stop reader threads

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

The plugin exposes five tools, all `discoverability="discoverable"`:

### `ws_connect`

Connect to a WebSocket endpoint. Returns a connection ID for subsequent
operations.

```json
{
  "name": "ws_connect",
  "description": "Connect to a WebSocket endpoint. Use a named connection from config or provide a raw URL. Returns a connection_id for polling and sending.",
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
  "url": "wss://stream.binance.com:9443/ws",
  "negotiated_subprotocol": null,
  "message": "Connected to wss://stream.binance.com:9443/ws"
}
```

**Error (dependency missing):**
```json
{
  "error": "WebSocket client requires the 'websockets' package. Install with: pip install websockets"
}
```

**Error (max connections):**
```json
{
  "error": "Maximum connections (4) reached. Close an existing connection first.",
  "active_connections": ["ws-1", "ws-2", "ws-3", "ws-4"]
}
```

### `ws_poll`

Poll for messages on an open connection. Blocks up to `timeout` seconds.
The model calls this in a loop.

```json
{
  "name": "ws_poll",
  "description": "Poll for new messages on a WebSocket connection. Blocks up to timeout seconds. Call in a loop after connecting.",
  "parameters": {
    "type": "object",
    "properties": {
      "connection_id": {
        "type": "string",
        "description": "Connection ID from ws_connect"
      },
      "timeout": {
        "type": "number",
        "description": "Max seconds to wait (1-30, default 15)"
      },
      "max_messages": {
        "type": "integer",
        "description": "Maximum messages to return per poll (default 50, max 200)"
      }
    },
    "required": ["connection_id"]
  }
}
```

**Returns (messages received):**
```json
{
  "connection_id": "ws-1",
  "status": "connected",
  "messages": [
    {
      "message_id": "msg_001",
      "timestamp": "2026-03-08T14:30:00.123Z",
      "type": "text",
      "data": "{\"e\":\"trade\",\"s\":\"BTCUSDT\",\"p\":\"67420.50\"}"
    },
    {
      "message_id": "msg_002",
      "timestamp": "2026-03-08T14:30:00.456Z",
      "type": "text",
      "data": "{\"e\":\"trade\",\"s\":\"BTCUSDT\",\"p\":\"67421.00\"}"
    }
  ],
  "cursor": "msg_002",
  "buffer_size": 0
}
```

**Returns (timeout, no messages):**
```json
{
  "connection_id": "ws-1",
  "status": "connected",
  "messages": [],
  "cursor": null,
  "buffer_size": 0
}
```

**Returns (connection lost):**
```json
{
  "connection_id": "ws-1",
  "status": "disconnected",
  "messages": [
    {
      "message_id": "sys_001",
      "timestamp": "2026-03-08T14:35:00Z",
      "type": "system",
      "data": "Connection closed by remote (code=1006, reason='')"
    }
  ],
  "cursor": "sys_001",
  "reconnecting": true,
  "reconnect_attempt": 2
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
      "buffer_size": 0,
      "last_message_at": "2026-03-08T14:29:58Z",
      "reconnect_count": 0
    }
  ],
  "available_connections": ["slack", "binance"],
  "max_connections": 4
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

## Connection Architecture

### WebSocketConnection Class

Each connection is represented by a `WebSocketConnection` object that owns a
reader thread and a message buffer:

```python
class WebSocketConnection:
    """A single WebSocket connection with a background reader thread.

    Lifecycle:
        1. connect(url, headers, subprotocols) — handshake, start reader
        2. Reader thread runs recv loop, buffers messages
        3. poll(timeout) — drain buffer or block on wake event
        4. send(data) — send message through the connection
        5. close() — send close frame, stop reader, clean up

    The reader thread runs until the connection closes (remotely or locally).
    On unexpected close, it sets status to 'disconnected' and optionally
    triggers reconnection.

    Thread safety:
        - _buffer (deque) is thread-safe for append/popleft
        - _wake_event (threading.Event) coordinates reader→poller
        - _lock (threading.Lock) protects status transitions
        - send() acquires _send_lock to serialize writes
    """

    def __init__(self, connection_id, config, on_status_change=None):
        self.connection_id = connection_id
        self.url = None
        self.status = "idle"            # idle → connecting → connected → disconnected → closed
        self._ws = None                 # websockets.sync.client.ClientConnection
        self._reader_thread = None
        self._buffer = deque(maxlen=config.max_buffer_size)
        self._wake_event = threading.Event()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stats = ConnectionStats()
        self._config = config
        self._on_status_change = on_status_change
        self._reconnect_count = 0
        self._stop_requested = False
```

### Reader Thread

Each connection runs a daemon reader thread that loops on `recv()`:

```python
def _reader_loop(self):
    """Background thread: read messages and buffer them.

    Runs until connection closes or stop is requested. On unexpected
    close with reconnect enabled, attempts to re-establish the
    connection with exponential backoff.
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
            self._buffer.append(msg)
            self._wake_event.set()  # Wake up any blocked poller
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
notifies the plugin (for TaskEventBus publishing and system message injection).

### System Messages

Connection lifecycle events are injected into the message buffer as system
messages (type `"system"`) so the model sees them during `ws_poll`:

| Event | System Message |
|-------|---------------|
| Connected | `"Connected to wss://... (negotiated subprotocol: graphql-ws)"` |
| Disconnected | `"Connection closed by remote (code=1006, reason='')"` |
| Reconnecting | `"Reconnecting (attempt 2/5, next retry in 4.0s)..."` |
| Reconnected | `"Reconnected successfully (was disconnected for 8.2s)"` |
| Reconnect failed | `"Reconnection failed after 5 attempts. Use ws_connect to reconnect manually."` |
| Ping timeout | `"Connection appears dead (ping timeout). Closing."` |

This keeps the model informed without requiring it to call `ws_status` after
every poll.

## Event Bus Integration

Like the Webhook plugin, WebSocket events are published to `TaskEventBus` for
cross-agent consumption:

```python
def _publish_to_event_bus(self, connection_id, message):
    """Publish incoming WebSocket message to TaskEventBus."""
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
        logger.debug("TaskEventBus unavailable, using buffer-only delivery")
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

        self._inject_system_message(
            f"Reconnecting (attempt {attempt + 1}/{self._config.reconnect_max_attempts}, "
            f"next retry in {actual_delay:.1f}s)..."
        )
        time.sleep(actual_delay)

        try:
            self._do_connect()
            self._reconnect_count += 1
            self._inject_system_message(
                f"Reconnected successfully (attempt {attempt + 1})"
            )
            return  # Success — resume reader loop
        except Exception as exc:
            logger.warning("Reconnect attempt %d failed: %s", attempt + 1, exc)

    self._inject_system_message(
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
message after the WebSocket handshake completes. The model handles this
naturally:

```
Agent: ws_connect(url="wss://gateway.discord.gg/?v=10&encoding=json")
→ {connection_id: "ws-1", status: "connected"}

Agent: ws_poll(connection_id="ws-1")
→ {messages: [{type: "text", data: '{"op":10,"d":{"heartbeat_interval":41250}}'}]}

Agent: ws_send(connection_id="ws-1", data='{"op":2,"d":{"token":"${DISCORD_TOKEN}","intents":513,...}}')
→ {status: "sent"}

Agent: ws_poll(connection_id="ws-1")
→ {messages: [{type: "text", data: '{"op":0,"t":"READY","d":{...}}'}]}
```

No special auth abstraction needed — the model reads the protocol docs and
drives the handshake via `ws_send`.

### Pattern 4: Two-Phase URL (Slack RTM)

Slack's RTM API requires an HTTP call to get a WebSocket URL, then connecting:

```
Agent: cli(command="curl -s https://slack.com/api/rtm.connect -d token=$SLACK_TOKEN")
→ {"ok": true, "url": "wss://cerberus-xxxx.lb.slack-msgs.com/websocket/..."}

Agent: ws_connect(url="wss://cerberus-xxxx.lb.slack-msgs.com/websocket/...")
→ {connection_id: "ws-1", status: "connected"}
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
  "system_instructions": "You are a Slack bot connected via RTM WebSocket.\n\nOn startup:\n1. Call ws_connect(name=\"slack\") to connect.\n2. Loop forever calling ws_poll(connection_id=<id>).\n3. For each message, parse the JSON and respond appropriately.\n4. Use ws_send to reply in channels.\n5. If you see a system message about disconnection, wait for auto-reconnect.\n\nNever stop polling. After processing each batch, immediately poll again.",
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
      "connections": {
        "binance": {
          "url": "wss://stream.binance.com:9443/ws",
          "reconnect": true,
          "ping_interval": 180
        }
      }
    }
  },
  "system_instructions": "You are a market data monitor.\n\nOn startup:\n1. ws_connect(name=\"binance\")\n2. ws_send to subscribe: {\"method\":\"SUBSCRIBE\",\"params\":[\"btcusdt@trade\",\"ethusdt@trade\"],\"id\":1}\n3. Loop on ws_poll, analyzing price movements.\n4. Maintain a running summary of price trends.\n5. Alert on significant moves (>2% in 5 minutes).\n\nNever stop polling.",
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
- Reconnection attempts are bounded (`reconnect_max_attempts`)
- Ping timeouts detect dead connections

## Comparison: WebSocket Client vs Interactive Shell

The `interactive_shell` plugin also maintains persistent bidirectional sessions.
Key differences:

| Dimension | Interactive Shell | WebSocket Client |
|-----------|------------------|-----------------|
| **Target** | Local processes (PTY) | Remote services (network) |
| **Transport** | Pseudo-terminal (stdin/stdout) | WebSocket (TCP + TLS) |
| **Output format** | Raw terminal output (ANSI stripped) | Structured messages (JSON, text) |
| **Idle detection** | 500ms silence heuristic | Explicit framing (WS messages) |
| **Auth** | OS-level (user permissions) | Application-level (tokens, headers) |
| **Reconnection** | N/A (process is dead) | Auto-reconnect with backoff |
| **Use case** | REPLs, debuggers, SSH | APIs, event streams, chat bots |

There is no overlap — they serve completely different integration patterns.

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
| Remote close during operation | System message in buffer, auto-reconnect if enabled |
| Send on closed connection | `ws_send` returns error suggesting reconnect |
| Poll on unknown connection_id | Error result listing valid connection IDs |
| Buffer overflow | Oldest messages evicted (FIFO), counter incremented |
| Max connections reached | `ws_connect` returns error listing active connections |
| Reader thread crashes | Logged, status changes to `"error"`, system message buffered |
| Plugin shutdown during active poll | Wake event set, poll returns immediately |

## Lifecycle & Cleanup

1. **Plugin initialize** — config loaded and merged, no connections opened.
2. **`ws_connect` call** — connection established, reader thread started.
3. **Session active** — connections running, model polling and sending.
4. **Session GC** — old turns with processed messages are garbage-collected;
   connections and buffers persist.
5. **`ws_close` call** — individual connection closed gracefully.
6. **Session stop / shutdown** — `shutdown()` closes all connections, stops all
   reader threads, wakes blocked pollers, clears buffers.

Reader threads are daemon threads — they don't prevent process exit if the
session crashes without calling `shutdown()`.

## Testing Strategy

### Unit Tests

- **Config loading** — precedence merging, variable expansion, named connections.
- **Connection state machine** — status transitions, lock safety.
- **Message buffering** — append, drain, overflow eviction, cursor logic.
- **Tool executors** — connect/poll/send/close/status return correct structures.
- **Dependency gating** — correct error when `websockets` is not installed.

### Integration Tests

- **Echo server** — start a local WebSocket echo server (using `websockets`
  `serve`), connect, send, receive, verify round-trip.
- **Reconnection** — connect, kill server, verify reconnect with backoff,
  restart server, verify reconnect succeeds.
- **Concurrent connections** — open multiple connections, poll all, verify
  independent buffers.
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
  before buffering (reduce noise for high-volume streams).
- **Binary protocol support** — structured binary message parsing (Protocol
  Buffers, MessagePack) with schema registration.
- **Connection groups** — connect to multiple related endpoints and poll all at
  once (e.g., multiple Binance streams).
- **Outbound-only mode** — `ws_connect` + `ws_send` without reader thread, for
  services where the agent only pushes data.
- **Health check endpoint** — expose connection health via the server's existing
  WebSocket transport for monitoring dashboards.
- **Shared connections** — multiple sessions sharing a single WebSocket
  connection (via event bus fan-out) to avoid duplicate connections to the same
  endpoint.
- **Rate-limited sending** — configurable send rate limits to avoid being
  banned by external services.
