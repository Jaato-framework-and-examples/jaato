# JAATO IPC & WebSocket Transport Clients — Complete Reference

> Scope: How to write client programs that connect to a Jaato server daemon via IPC (Unix domain socket / Windows named pipe) or WebSocket, send messages, receive events, handle reconnection, and manage sessions.

## Table of Contents

1. [What Are Transport Clients?](#1-what-are-transport-clients)
2. [Protocol Overview — The Event Layer](#2-protocol-overview--the-event-layer)
3. [IPC Client (IPCClient)](#3-ipc-client-ipcclient)
4. [IPC Recovery Client (IPCRecoveryClient)](#4-ipc-recovery-client-ipcrecoveryclient)
5. [WebSocket Client (JaatoWSServer)](#5-websocket-client-jaatowsserver)
6. [Event Protocol — Serialization & Deserialization](#6-event-protocol--serialization--deserialization)
7. [Client Configuration (RecoveryConfig)](#7-client-configuration-recoveryconfig)
8. [Backend Abstraction (Backend / IPCBackend)](#8-backend-abstraction-backend--ipcbackend)
9. [Event Bus (Internal Cross-Agent Events)](#9-event-bus-internal-cross-agent-events)
10. [Presentation Context](#10-presentation-context)
11. [Configuration / Schema Reference](#11-configuration--schema-reference)
12. [Runtime Internals](#12-runtime-internals)
13. [Source Code Map](#13-source-code-map)

---

## 1. What Are Transport Clients?

Jaato uses a **client-server architecture** where a long-running daemon process (`jaato-server`) manages AI sessions, plugins, and tool execution. Clients connect to this daemon over one of two transports:

| Transport | Use Case | Platform | Security Model |
|-----------|----------|----------|----------------|
| **IPC** (Unix domain socket / Windows named pipe) | Local TUI, CLI tools, IDE extensions | Local machine only | Filesystem permissions |
| **WebSocket** (ws:// or wss://) | Web dashboards, remote access, browser integrations | Any host (local or remote) | TLS (wss://), optional SSO |

Both transports share the **same event protocol** — the wire format is identical. The only difference is the framing and transport layer:

- **IPC**: 4-byte big-endian length prefix + JSON payload (framed stream)
- **WebSocket**: JSON text frames (the `websockets` library handles framing)

### Architecture Layers

```
┌─────────────────────────────────────────────────┐
│  UI Layer (rich_client.py, web dashboard)        │
├─────────────────────────────────────────────────┤
│  Recovery Layer (IPCRecoveryClient)              │  ← auto-reconnect, session reattach
├─────────────────────────────────────────────────┤
│  Backend Abstraction (Backend / IPCBackend)      │  ← mode-agnostic API
├─────────────────────────────────────────────────┤
│  Transport Client (IPCClient / JaatoWSServer)    │  ← framing, connect/disconnect
├─────────────────────────────────────────────────┤
│  Event Protocol (serialize_event / deserialize)   │  ← JSON dataclass serialization
└─────────────────────────────────────────────────┘
```

---

## 2. Protocol Overview — The Event Layer

All communication between client and server uses **JSON-serialized dataclass events**. Every event has a `type` field (an `EventType` enum value) and a `timestamp`.

### Event Categories

| Direction | Category | Examples |
|-----------|----------|---------|
| Server → Client | Connection lifecycle | `ConnectedEvent`, `ErrorEvent` |
| Server → Client | Agent lifecycle | `AgentCreatedEvent`, `AgentOutputEvent`, `AgentCompletedEvent` |
| Server → Client | Tool execution | `ToolCallStartEvent`, `ToolCallEndEvent`, `ToolOutputEvent` |
| Server → Client | Permission flow | `PermissionInputModeEvent`, `PermissionResolvedEvent` |
| Server → Client | Clarification flow | `ClarificationBatchEvent`, `ClarificationResolvedEvent` |
| Server → Client | Plan/context updates | `PlanUpdatedEvent`, `ContextUpdatedEvent`, `TurnCompletedEvent` |
| Server → Client | Session management | `SessionInfoEvent`, `SessionListEvent`, `SessionProfilesEvent` |
| Client → Server | Message sending | `SendMessageRequest` |
| Client → Server | Permission response | `PermissionResponseRequest` |
| Client → Server | Clarification response | `ClarificationResponseRequest`, `ClarificationBatchResponseEvent` |
| Client → Server | Commands | `CommandRequest`, `StopRequest` |
| Client → Server | Configuration | `ClientConfigRequest` |
| Client → Server | Workspace management | `WorkspaceListRequest`, `WorkspaceSelectRequest` |
| Bidirectional | External events | `ExternalEventRequest` (client→server) |
| Server ↔ Server | Peer gossip | `PeerHeartbeatEvent`, `PeerSpawnRequestEvent` |

### Serialization

Events are serialized via `serialize_event(event)` and deserialized via `deserialize_event(json_str)`:

```python
from jaato_sdk.events import SendMessageRequest, serialize_event, deserialize_event

# Client → Server
request = SendMessageRequest(text="Hello, world!")
json_str = serialize_event(request)
# → '{"type": "message.send", "timestamp": "...", "text": "Hello, world!", "attachments": []}'

# Server → Client
event = deserialize_event(json_str)
assert isinstance(event, SendMessageRequest)
```

The `_EVENT_CLASSES` registry maps `EventType` string values to dataclass types. Unknown event types raise `ValueError`. Unknown fields in incoming JSON are silently dropped (forward compatibility).

---

## 3. IPC Client (IPCClient)

**File**: `jaato-sdk/jaato_sdk/client/ipc.py`

The `IPCClient` is the low-level transport client for connecting to a Jaato server daemon via IPC. It handles platform-specific socket/pipe connections, message framing, and the initial handshake.

### Constructor

```python
client = IPCClient(
    socket_path="/tmp/jaato.sock",   # Unix: socket path; Windows: pipe name
    auto_start=True,                  # Auto-start server if not running
    env_file=".env",                  # .env file for auto-started server
    workspace_path="/path/to/project", # Working directory sent to server
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `socket_path` | `str` | Platform-dependent | Unix: `/tmp/jaato.sock`; Windows: `jaato` (becomes `\\.\pipe\jaato`) |
| `auto_start` | `bool` | `True` | Launch `python -m server --daemon` if server not running |
| `env_file` | `str` | `.env` | Path to `.env` file (resolved relative to `workspace_path`) |
| `workspace_path` | `Optional[str]` | `None` | Working directory; falls back to `os.getcwd()` |

### Platform-Specific Defaults

```python
# Defined in ipc.py
if sys.platform == "win32":
    DEFAULT_SOCKET_PATH = "jaato"          # → \\.\pipe\jaato
    DEFAULT_PID_FILE = "<temp>/jaato.pid"
else:
    DEFAULT_SOCKET_PATH = "/tmp/jaato.sock"
    DEFAULT_PID_FILE = "/tmp/jaato.pid"
```

### Connection Flow

```
Client                          Server
  │                               │
  │──── open_unix_connection() ───►│  (or create_pipe_connection on Windows)
  │                               │
  │◄─── ConnectedEvent ───────────│  {client_id, transport: "ipc", server_version}
  │                               │
  │──── CommandRequest("set_workspace", [cwd]) ──►│
  │──── ClientConfigRequest(env_file, presentation) ──►│
  │                               │
  │     ... session management & event streaming ...
```

### Key Methods

#### `async connect(timeout=5.0) -> bool`

Establishes the IPC connection and performs the handshake:

1. Connects to the socket/pipe (with auto-start retry logic)
2. Reads the `ConnectedEvent` (extracts `client_id` and `server_version`)
3. Sends `set_workspace` command with the client's working directory
4. Sends `ClientConfigRequest` with env file path and presentation context

On connection failure with `auto_start=True`, the client:
- Checks for a running server via PID file (Unix) or pipe probe (Windows)
- Launches `python -m server --daemon --ipc-socket <path>`
- Waits for the socket/pipe to appear (up to 10s)
- Retries the connection with backoff

**Important Windows note**: The client must NOT use short per-attempt timeouts inside `asyncio.wait_for()` when retrying after auto-start. If `create_pipe_connection()` has already established a transport at the OS level, cancelling the coroutine leaks a "ghost client" on the server. Only retry on errors that prove no connection was established (`ConnectionRefused`, `FileNotFound`, `OSError`).

#### `async disconnect() -> None`

Closes the writer, clears all state (reader, writer, session_id, client_id, server_version).

#### `async send_message(text, attachments=None) -> None`

Sends a `SendMessageRequest` event.

#### `async events() -> AsyncIterator[Event]`

Async generator that yields events from the server until the connection is closed. Handles:
- **Buffered events**: Events consumed during `create_session()` are replayed first
- **Session ID tracking**: Automatically updates `_session_id` from `SessionInfoEvent`
- **Callback**: Calls `_on_event` callback if set via `set_event_callback()`
- **Clean exit**: Stops yielding on connection loss (no exception raised to caller)

The `_events_active` flag prevents concurrent readers on the `StreamReader` — when `events()` is active, `create_session()` falls back to fire-and-forget mode.

#### `async create_session(name=None, profile=None, agent=None, agent_params=None, timeout=60.0) -> Optional[str]`

Sends a `session.new` command and waits for `SessionInfoEvent`. When `events()` is already consuming the socket, operates in fire-and-forget mode (the `SessionInfoEvent` will arrive via `events()`).

#### Session Management

```python
await client.attach_session(session_id)       # Attach to existing session
await client.get_default_session()            # Get or create default
await client.list_sessions()                  # Request session list
await client.list_profiles()                  # Request agent profiles
await client.request_history(agent_id="main") # Request conversation history
await client.request_command_list()           # Request available commands
await client.disable_tool(tool_name)          # Disable a tool
```

#### Response Methods

```python
await client.respond_to_permission(request_id, response, edited_arguments=None)
await client.respond_to_clarification(request_id, response)
await client.respond_to_reference_selection(request_id, response)
await client.stop()                           # Cancel current generation
await client.execute_command(command, args)   # Execute a command
```

### Message Framing

```
┌──────────────────┬──────────────────────────┐
│ 4 bytes (uint32) │     JSON payload         │
│ big-endian       │     UTF-8 encoded        │
│ = payload length │                          │
└──────────────────┴──────────────────────────┘
```

Constants:
- `HEADER_SIZE = 4`
- `MAX_MESSAGE_SIZE = 10 * 1024 * 1024` (10 MB)

Reading uses `readexactly()` for reliable framed reading. Writing uses `struct.pack(">I", len(payload))` for the header.

### Windows Named Pipe Handling

The `_get_pipe_path()` method normalizes various user inputs to the canonical `\\.\pipe\<name>` form:
- Accepts bare names: `"jaato"` → `\\.\pipe\jaato`
- Accepts partial prefixes: `\.\pipe\jaato` → `\\.\pipe\jaato`
- Accepts mangled MSYS2 input: `\.pipejaato` → `\\.\pipe\jaato`
- Accepts full canonical: `\\.\pipe\jaato` → `\\.\pipe\jaato`

The `_check_pipe_exists()` method uses `WaitNamedPipeW` with a 1ms timeout to probe for pipe existence without consuming a pipe instance. Distinguishes "pipe exists but busy" (`ERROR_SEM_TIMEOUT = 121`) from "pipe not found".

### IncompatibleServerError

Raised when the server version is below the client's minimum requirement. This error is **non-retryable** — an old server will not become newer on retry. The `IPCRecoveryClient` classifies this as a permanent error and stops reconnection.

### ClientConfigRequest

After connecting, the client sends its configuration to the server:

```python
await client._send_client_config()
```

This sends:
- **env_file**: Absolute path to the client's `.env` file (server loads it for session creation)
- **working_dir**: Client's working directory (for finding config files like `.lsp.json`)
- **trace_log_path** and **provider_trace_log**: Trace log paths from environment
- **presentation**: `PresentationContext` dict describing display capabilities (content_width, client_type, etc.)

---

## 4. IPC Recovery Client (IPCRecoveryClient)

**File**: `jaato-sdk/jaato_sdk/client/recovery.py`

Wraps `IPCClient` with automatic connection recovery. When the server becomes unavailable, the recovery client attempts to reconnect with exponential backoff and can reattach to the previous session.

### Constructor

```python
from jaato_sdk.client import IPCRecoveryClient, ConnectionState
from jaato_sdk.client.config import get_recovery_config

client = IPCRecoveryClient(
    socket_path="/tmp/jaato.sock",
    config=get_recovery_config(),          # Loads from files + env
    auto_start=True,
    env_file=".env",
    workspace_path=Path.cwd(),
    on_status_change=lambda status: print(status.state),
)
```

### State Machine

```
DISCONNECTED ──connect()──► CONNECTING ──success──► CONNECTED
     ▲                          │                      │
     │                     failure               connection lost
     │                          ▼                      ▼
     │                     DISCONNECTED          RECONNECTING
     │                          ▲                      │
     │                     give_up()          max_attempts │
     │                          │                 or permanent error
     │                          ▼                      ▼
     └──────────────────── CLOSED ◄─────────────── CLOSED
```

| State | Description |
|-------|-------------|
| `DISCONNECTED` | Initial state, or recovery gave up |
| `CONNECTING` | Attempting initial connection |
| `CONNECTED` | Active connection, events flowing |
| `RECONNECTING` | Connection lost, auto-recovery in progress |
| `DISCONNECTING` | Graceful disconnect initiated |
| `CLOSED` | Terminal state — no more reconnection attempts |

### ConnectionStatus

```python
@dataclass
class ConnectionStatus:
    state: ConnectionState
    attempt: int = 0               # Current reconnection attempt
    max_attempts: int = 0           # Maximum attempts configured
    next_retry_in: Optional[float]  # Seconds until next retry
    last_error: Optional[str]       # Last error description
    session_id: Optional[str]       # Attached session ID
    client_id: Optional[str]        # Server-assigned client ID
```

### Reconnection Logic

When the connection is lost (the `events()` iterator exits):

1. Transitions to `RECONNECTING`
2. Creates a new `IPCClient` instance (with `auto_start=False`)
3. Attempts connection with exponential backoff
4. On success: reattaches to previous session (if `reattach_session=True` and session_id is known)
5. On max attempts or permanent error: transitions to `CLOSED`

**Exponential backoff formula**:
```
delay = min(base_delay * 2^(attempt-1), max_delay) ± jitter
jitter = delay * jitter_factor * random.uniform(-1, 1)
```

Default: 1s base, 60s max, 0.3 jitter factor → [0.7s, 1.3s], [1.4s, 2.6s], [2.8s, 5.2s], ...

### Error Classification

| Classification | Error Types | Action |
|---------------|-------------|--------|
| **Transient** | `ConnectionRefusedError`, `ConnectionResetError`, `TimeoutError` | Retry |
| **Permanent** | `IncompatibleServerError`, `FileNotFoundError`, "permission denied", "authentication" | Stop retrying |

### Event Stream with Reconnection

```python
async for event in client.events():
    if isinstance(event, AgentOutputEvent):
        print(event.text)
    # Connection loss is handled transparently —
    # the iterator blocks during reconnection
```

### Guard Methods

```python
client._check_can_send()  # Raises ReconnectingError or ConnectionClosedError
```

All send methods (`send_message`, `respond_to_permission`, etc.) call `_check_can_send()` first and raise:
- `ConnectionClosedError` if the connection is permanently closed
- `ReconnectingError` if reconnection is in progress
- `ConnectionError` if not connected

---

## 5. WebSocket Client (JaatoWSServer)

**File**: `jaato-server/server/websocket.py`

The WebSocket server enables remote clients (web dashboards, browser integrations) to connect to Jaato. It wraps `JaatoServer` and provides multi-client support with optional TLS, workspace provisioning, and AppArmor isolation.

### Constructor

```python
server = JaatoWSServer(
    host="localhost",
    port=8080,
    workspace_root="/data/jaato-workspaces",  # Enables workspace management
    apparmor=True,                              # Auto-detect AppArmor availability
    default_template="default",                 # Template for auto-provisioning
    workspace_max_age=86400,                    # Reap after 24h
    ssl_context=load_tls_context(),             # Optional TLS
)
```

### Operating Modes

**Standalone mode** (no `CommandRouter`): The WS server creates its own `JaatoServer` per workspace and handles messages directly.

**Daemon mode** (with `CommandRouter`): The WS server delegates all session/command events to the `CommandRouter`, which manages multiple sessions across transports (IPC + WS).

### Key Features

#### Connection Interceptors
```python
server.set_connection_interceptor(
    check=lambda ws: ws.request.headers.get("X-Peer") == "true",
    handler=handle_peer_connection,
)
```
Interceptors are evaluated before normal client handling. Used for peer gossip clustering.

#### Extension Message Handlers
```python
server.register_message_handler("reconnect.snapshot", handle_snapshot)
```
Custom WS message types for daemon extensions.

#### Client-Side Tool Execution
WS clients can register tools that execute on the client (browser):
1. Client sends `ToolsRegisterClientRequest` with tool definitions
2. Server creates proxy tools in the session registry
3. When the model calls a proxy tool, server sends `ToolExecuteRequestEvent` to the client
4. Client executes and returns `ToolExecuteResultEvent`

#### Workspace Auto-Provisioning
When a WS client creates a session without a workspace:
1. Server provisions a new workspace from a template
2. Copies `.env` and `.jaato/profiles/` from the template
3. Optionally applies AppArmor confinement
4. Materializes inline `staged_files` (base64-encoded files sent in the WS envelope)

#### TLS Configuration
Loaded from `~/.jaato/servers.json`:
```json
{
  "tls": {
    "cert": "/path/to/cert.pem",
    "key": "/path/to/key.pem",
    "ca_cert": "/path/to/ca.pem"
  }
}
```

#### WSEventSinkAdapter
Bridges the async WebSocket world with the synchronous `EventSink` protocol used by `CommandRouter` and `SessionManager`. Uses `asyncio.run_coroutine_threadsafe()` for thread-safe scheduling.

### Message Handling Flow

```
Client sends JSON message
    │
    ├── Intercepted? → Custom handler
    │
    ├── deserialize_event() success?
    │   ├── Workspace request? → _handle_workspace_event()
    │   ├── External event? → _handle_external_event()
    │   ├── Daemon mode? → _handle_message_daemon() → CommandRouter
    │   └── Standalone mode? → Direct JaatoServer routing
    │
    └── Deserialize failed?
        ├── Extension handler matches? → Custom handler
        └── Otherwise → ErrorEvent("Unknown message type")
```

---


## 5.1 WebSocket Authentication (Keycloak / Optional)

### Overview

WebSocket authentication is **optional**. The `JaatoWSServer` does not require auth for
local or trusted deployments. When enabled, authentication provides **inter-user session
isolation** (user A cannot attach/snapshot/delete user B's sessions) but does not gate
individual commands.

### Message Interchange

The WS handshake itself is unauthenticated. After the TCP connection is established,
the following interchange occurs:

```
1. Client  ─── WS handshake (HTTP 101) ────→  Server
2. Server  ─── ConnectedEvent ────────────→  Client   (automatic, always sent)
3. Client  ─── {"type": "auth.token",       (if auth desired)
               "token": "<Keycloak JWT>"}
                                                    →  Server
4. Server  ─── {"type": "auth.token",              →  Client
               "user_id": "<username>"}
   OR
4. Server  ─── {"type": "auth.token",              →  Client
               "error": "invalid or expired token"}
```

**Steps 1-2 always happen.** Step 3 is client-initiated and optional. If the client
never sends an `auth.token` frame, the server proceeds normally with no user
association.

### Token Acquisition (client_credentials grant)

Clients obtain a Keycloak access token via the standard OAuth 2.0 `client_credentials`
grant:

```
POST {keycloak_base_url}/realms/{realm}/protocol/openid-connect/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
&client_id=<client_id>
&client_secret=<client_secret>
```

Response:
```json
{
  "access_token": "<JWT>",
  "token_type": "Bearer",
  "expires_in": 300
}
```

### Keycloak Client Setup

Create a service account in Keycloak for programmatic access:

```bash
# Create client (via admin API)
curl -sk -X POST "https://localhost:8180/admin/realms/jaato/clients"   -H "Authorization: Bearer $ADMIN_TOKEN"   -H "Content-Type: application/json"   -d '{
    "clientId": "telegram-bot",
    "enabled": true,
    "publicClient": false,
    "serviceAccountsEnabled": true,
    "directAccessGrantsEnabled": true,
    "clientAuthenticatorType": "client-secret"
  }'

# Get the generated secret
curl -sk "https://localhost:8180/admin/realms/jaato/clients/$CLIENT_UUID/client-secret"   -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Important:** `clientAuthenticatorType` must be `"client-secret"`, not
`"client-secret-post"` or `"client-secret-jwt"`. Other types will cause
`client_credentials` grant to fail with "Invalid client".

### Server-Side Validation

Handled by `jaato_premium/session_reconnect/extension.py`:

1. Receives the `auth.token` message with a JWT
2. Fetches JWKS from the issuer in `~/.jaato/servers.json` auth section
   (e.g. `https://localhost:8180/realms/jaato`)
3. Decodes + validates the JWT using `authlib.jose.jwt.decode(token, jwks)`
4. Extracts `preferred_username` (or `sub`) and calls
   `ws_server.set_client_user(client_id, username)`

### Ownership Check

The session ownership check (extension.py line 247) short-circuits when either side
is `None`:

```python
if user_id and journal.created_by and journal.created_by != user_id:
    # reject — user owns the session, not the requester
```

Consequences:
- **No auth sent** → `user_id = None` → check short-circuits → full access
- **No auth on session creation** → `created_by = None` → check short-circuits → any client can attach
- **Both authenticated** → ownership enforced

### Running Without Auth

Three options, from least to most invasive:

1. **Don't send `auth.token`** — simplest, works today. The server treats the
   connection as anonymous.
2. **Remove auth section from `~/.jaato/servers.json`** — the extension's
   `_get_auth_config()` returns `None`, tokens get rejected, but unauthenticated
   clients still work. Requires daemon restart.
3. **Don't load the extension** — remove the `session_reconnect` entry from
   `[project.entry-points."jaato.extensions"]` in `jaato-premium/pyproject.toml`
   (or don't install `jaato-premium`). No `auth.token` handler exists at all.

### Security Considerations

Auth provides **inter-user isolation** only. It does not:
- Prevent anonymous connections from creating sessions
- Prevent anonymous connections from reading other anonymous sessions
- Gate individual commands by role/scope
- Gate `message.send` or `tool.execute_request`

For local/VPN deployments behind a firewall, running auth-less is fine.
For exposed deployments, auth-less is actively unsafe — anyone who can open
a TCP connection to the WS port can drive the agent (which has shell access).

### Credentials Storage (pass: resolver)

Service account credentials can be stored using the Unix `pass` password manager
and resolved via the `pass://` URI scheme:

```
pass://jaato/keycloak/server-url      → https://localhost:8180
pass://jaato/keycloak/realm            → jaato
pass://jaato/keycloak/telegram-bot/client-id    → telegram-bot
pass://jaato/keycloak/telegram-bot/client-secret → <secret>
pass://jaato/keycloak/telegram-bot/token         → <JWT>
```


## 6. Event Protocol — Serialization & Deserialization

**File**: `jaato-sdk/jaato_sdk/events.py` (1748 lines, ~72 KB)

### Event Base Class

```python
@dataclass
class Event:
    type: EventType
    timestamp: str  # ISO 8601 UTC

    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self) -> str: ...
```

### EventType Enum

The `EventType` enum defines all event types as string values (e.g., `"connected"`, `"message.send"`, `"agent.output"`). See the source file for the complete list of ~80 event types.

### Serialization

```python
def serialize_event(event: Event) -> str:
    """Event → JSON string"""
    return event.to_json()

def deserialize_event(json_str: str) -> Event:
    """JSON string → Event (raises ValueError for unknown types)"""
    data = json.loads(json_str)
    event_type = data.get("type")
    event_class = _EVENT_CLASSES[event_type]
    # Remove unknown fields for forward compatibility
    known_fields = {f.name for f in event_class.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in known_fields}
    return event_class(**filtered_data)
```

### Forward Compatibility

Unknown fields in incoming JSON are silently dropped. This allows server and client to evolve independently — a newer server can add fields to events without breaking older clients.

### Factory Function

```python
event = create_event(EventType.CONNECTED, protocol_version="1.0", server_info={...})
```

---

## 7. Client Configuration (RecoveryConfig)

**File**: `jaato-sdk/jaato_sdk/client/config.py`

### RecoveryConfig

```python
@dataclass
class RecoveryConfig:
    enabled: bool = True           # Enable automatic reconnection
    max_attempts: int = 10         # Max reconnection attempts
    base_delay: float = 1.0        # Initial backoff (seconds)
    max_delay: float = 60.0        # Maximum backoff (seconds)
    jitter_factor: float = 0.3     # Random jitter range (±30%)
    connection_timeout: float = 5.0 # Per-attempt timeout (seconds)
    reattach_session: bool = True  # Auto-reattach after reconnect
```

**Validation rules** (enforced in `__post_init__`):
- `max_attempts >= 1`
- `base_delay >= 0.1`
- `max_delay >= base_delay`
- `0.0 <= jitter_factor <= 1.0`
- `connection_timeout >= 1.0`

### Configuration Precedence (highest wins)

1. **Environment variables** (`JAATO_IPC_*`)
2. **Project config** (`.jaato/client.json` in workspace)
3. **User config** (`~/.jaato/client.json`)
4. **Built-in defaults** (dataclass defaults)

### Environment Variables

| Variable | Config Path | Type | Default |
|----------|-------------|------|---------|
| `JAATO_IPC_AUTO_RECONNECT` | `recovery.enabled` | bool | `true` |
| `JAATO_IPC_RETRY_MAX_ATTEMPTS` | `recovery.max_attempts` | int | `10` |
| `JAATO_IPC_RETRY_BASE_DELAY` | `recovery.base_delay` | float | `1.0` |
| `JAATO_IPC_RETRY_MAX_DELAY` | `recovery.max_delay` | float | `60.0` |
| `JAATO_IPC_RETRY_JITTER` | `recovery.jitter_factor` | float | `0.3` |
| `JAATO_IPC_CONNECTION_TIMEOUT` | `recovery.connection_timeout` | float | `5.0` |
| `JAATO_IPC_REATTACH_SESSION` | `recovery.reattach_session` | bool | `true` |

### Example client.json

```json
{
  "recovery": {
    "enabled": true,
    "max_attempts": 10,
    "base_delay": 1.0,
    "max_delay": 60.0,
    "jitter_factor": 0.3,
    "connection_timeout": 5.0,
    "reattach_session": true
  }
}
```

---

## 8. Backend Abstraction (Backend / IPCBackend)

**File**: `jaato-tui/backend.py`

The `Backend` abstract class provides a mode-agnostic async API for the RichClient TUI. `IPCBackend` is the concrete implementation that delegates to `IPCRecoveryClient`.

### Backend Interface (Abstract)

| Method | Description |
|--------|-------------|
| `connect(project_id, location, model)` | Connect to the backend |
| `disconnect()` | Disconnect |
| `send_message(text, on_output, attachments)` | Send a message |
| `stop()` | Stop current processing |
| `get_user_commands()` | Get available commands |
| `execute_user_command(name, args)` | Execute a command |
| `get_history()` | Get conversation history |
| `get_context_usage()` | Get token counts |
| `reset_session()` | Clear history |
| `set_ui_hooks(hooks)` | Set agent lifecycle callbacks |
| `configure_tools(registry, permission, ledger)` | Configure tools |
| `set_gc_plugin(plugin, config)` | Set GC plugin |
| `set_session_plugin(plugin, config)` | Set session plugin |

---

## 9. Event Bus (Internal Cross-Agent Events)

**File**: `jaato-sdk/jaato_sdk/event_bus.py`, `jaato-sdk/jaato_sdk/event_payloads.py`

The `EventBus` is an internal pub/sub system for cross-agent and cross-plugin coordination. It is separate from the wire protocol events.

### Key Types

```python
@dataclass
class Event:
    event_id: str
    event_type: EventType  # event_bus.EventType, NOT events.EventType
    timestamp: str
    source_agent: str
    payload: Dict[str, Any]
```

### Subscription

```python
@dataclass
class Subscription:
    subscription_id: str
    subscriber_name: str
    filter: EventFilter  # agent_id, plan_id, step_id, event_types
    action_type: str     # "callback", "unblock_step", "inject_message"
```

### Payload Schemas

`event_payloads.py` defines `TypedDict` schemas for each event type's payload. These are the single source of truth for what keys a bus event contains. A sync test verifies these stay aligned with the corresponding server event dataclass fields.

---

## 10. Presentation Context

**File**: `jaato-sdk/jaato_sdk/events.py` (within `PresentationContext` dataclass)

Clients send a `PresentationContext` to describe their display capabilities:

```python
@dataclass
class PresentationContext:
    content_width: int = 80
    content_height: Optional[int] = None
    supports_markdown: bool = True
    supports_tables: bool = True
    supports_code_blocks: bool = True
    supports_images: bool = False
    supports_rich_text: bool = True
    supports_unicode: bool = True
    supports_mermaid: bool = False
    supports_expandable_content: bool = False
    client_type: ClientType = ClientType.TERMINAL
    communication_style: Optional[CommunicationStyle] = None
```

### ClientType

| Value | Description |
|-------|-------------|
| `TERMINAL` | TUI / CLI (rich text, fixed-width) |
| `WEB` | Browser-based UI (HTML, responsive) |
| `CHAT` | Messaging platform (Telegram, Slack, WhatsApp) |
| `API` | Headless / programmatic (plain text) |

### CommunicationStyle

| Value | Description |
|-------|-------------|
| `CONVERSATIONAL` | Short, frequent messages (inferred for `CHAT` client type) |
| `NARRATIVE` | Thorough, well-structured responses (inferred for all other types) |

The `to_system_instruction()` method generates a compact display-context block (~30-80 tokens) appended to the model's system prompt.

---

## 11. Configuration / Schema Reference

### IPCClient Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_connected` | `bool` | Whether connected to server |
| `connection_state` | `str` | `"connected"`, `"closing"`, or `"disconnected"` |
| `session_id` | `Optional[str]` | Current session ID |
| `client_id` | `Optional[str]` | Server-assigned client ID |
| `server_version` | `Optional[str]` | Server's package version (after connect) |

### IPCRecoveryClient Properties

| Property | Type | Description |
|----------|------|-------------|
| `socket_path` | `str` | IPC endpoint path |
| `config` | `RecoveryConfig` | Recovery configuration |
| `state` | `ConnectionState` | Current connection state |
| `is_connected` | `bool` | `state == CONNECTED` |
| `is_reconnecting` | `bool` | `state == RECONNECTING` |
| `is_closed` | `bool` | `state == CLOSED` |
| `session_id` | `Optional[str]` | Current session ID |
| `client_id` | `Optional[str]` | Server-assigned client ID |
| `server_version` | `Optional[str]` | Delegated to underlying IPCClient |

### JaatoWSServer Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"localhost"` | Bind address |
| `port` | `int` | `8080` | Bind port |
| `workspace_root` | `Optional[str]` | `None` | Root for workspace management |
| `apparmor` | `Optional[bool]` | `None` | `True`=required, `False`=disabled, `None`=auto |
| `default_template` | `str` | `"default"` | Template for auto-provisioning |
| `workspace_max_age` | `int` | `86400` | Max workspace age (seconds) |
| `ssl_context` | `Optional[SSLContext]` | `None` | TLS context for wss:// |

### Constants

| Constant | Value | Location |
|----------|-------|----------|
| `HEADER_SIZE` | `4` | Both `ipc.py` files |
| `MAX_MESSAGE_SIZE` | `10_485_760` (10 MB) | Both `ipc.py` files |
| `WINDOWS_PIPE_PREFIX` | `\\\\.\\pipe\\` | Both `ipc.py` files |

---

## 12. Runtime Internals

### Thread Safety

- **IPC Server**: Uses `asyncio.Lock` for client dict mutations and `call_soon_threadsafe` for event queue operations from thread pool executors.
- **WS Server**: Uses `asyncio.Lock` for client dict mutations. `WSEventSinkAdapter` uses `run_coroutine_threadsafe` for thread-safe event delivery from model/session threads.
- **Event Bus**: Uses `threading.Lock` for subscription management.

### Concurrent Reader Prevention

The `IPCClient._events_active` flag prevents concurrent reads on the `asyncio.StreamReader`. When `events()` is actively iterating:
- `create_session()` operates in fire-and-forget mode (sends command but doesn't read response)
- The `SessionInfoEvent` arrives via the `events()` iterator instead

### Connection Ghost Prevention (Windows)

On Windows, cancelling `asyncio.wait_for()` after `create_pipe_connection()` has established a transport leaks a ghost client. The client avoids this by:
1. Only retrying on errors that prove no transport was created
2. Not using short per-attempt timeouts during retry loops
3. Using `WaitNamedPipeW` to probe pipe existence without consuming instances

### Server Auto-Start

The IPC client's `_start_server()` method:
1. Checks for a running server via PID file and pipe probe
2. On Unix: cleans up stale socket files from crashed servers
3. On Windows: passes the resolved pipe path (`\\.\pipe\jaato`) to avoid MSYS2 backslash mangling
4. Launches `python -m server --ipc-socket <path> --daemon`
5. Waits up to 10 seconds for the IPC endpoint to appear

### Event Buffering

During request-response operations (e.g., `create_session()`), events that arrive before the target response are buffered in `_buffered_events`. When `events()` starts, it drains this buffer first, ensuring no events are lost.

### Daemon Mode vs Standalone Mode

In daemon mode, the WS server delegates to a `CommandRouter` which manages sessions via a `SessionManager`. In standalone mode, the WS server creates its own `JaatoServer` instance. The `_handle_message()` method checks `self._command_router` to determine which path to use.

### Staged Files (WS Session Creation)

The `<jaato-task>` web component can ship files inline in the WS `session.new` envelope as base64-encoded `staged_files`. This avoids the HTTP `/api/task/artifacts` upload roundtrip and its SSO cookie dependency. The server materializes these files with path-traversal protections matching the HTTP upload route.

---

## 13. Source Code Map

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| `jaato-sdk/jaato_sdk/events.py` | 1748 | 72 KB | All event dataclasses, EventType enum, serialization |
| `jaato-sdk/jaato_sdk/event_payloads.py` | 423 | 14 KB | TypedDict payload schemas for EventBus |
| `jaato-sdk/jaato_sdk/event_bus.py` | 249 | 9 KB | Internal pub/sub event bus types |
| `jaato-sdk/jaato_sdk/client/ipc.py` | 1178 | 47 KB | IPCClient — low-level IPC transport |
| `jaato-sdk/jaato_sdk/client/recovery.py` | 961 | 34 KB | IPCRecoveryClient — auto-reconnect wrapper |
| `jaato-sdk/jaato_sdk/client/config.py` | 413 | 14 KB | RecoveryConfig, config loading |
| `jaato-sdk/jaato_sdk/client/__init__.py` | 15 | 475 B | Public exports |
| `jaato-server/server/ipc.py` | 710 | 26 KB | JaatoIPCServer — IPC server |
| `jaato-server/server/websocket.py` | 1787 | 74 KB | JaatoWSServer — WebSocket server |
| `jaato-server/shared/jaato_client.py` | 1088 | 40 KB | JaatoClient — server-side facade |
| `jaato-tui/backend.py` | 569 | 18 KB | Backend abstraction for TUI |
| `jaato-tui/rich_client.py` | 2888 | 135 KB | RichClient TUI implementation |
| `examples/client.json` | 29 | 1 KB | Example client configuration |

### Related Documentation

| Document | Purpose |
|----------|---------|
| `docs/ipc-recovery.md` | IPC connection recovery design |
| `docs/design/websocket-client-plugin.md` | Outbound WebSocket client plugin (agent → external services) |
| `docs/design/websocket-workspace-isolation.md` | WS workspace provisioning and AppArmor isolation |
| `docs/web-client-design.md` | Web client design document |
