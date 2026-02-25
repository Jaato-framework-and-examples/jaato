# IPC Connection Recovery

This document describes the automatic connection recovery mechanism for IPC-connected clients in jaato.

## Overview

When a client connects to a jaato server via IPC (Unix domain socket or Windows named pipe), the connection may be interrupted due to:

- Server restart (e.g., upgrade, configuration change)
- Server crash
- Network issues (for remote connections via SSH tunnels)
- System sleep/wake cycles

Without recovery, the client would lose its connection and require manual reconnection. The IPC recovery mechanism provides **automatic reconnection with exponential backoff**, preserving the user's session and conversation state.

## Architecture

### Component Diagram

```
┌──────────────────┐     ┌───────────────────┐     ┌─────────────┐
│  rich_client.py  │◄───►│ IPCRecoveryClient │◄───►│  IPCClient  │
│    (UI Layer)    │     │ (Recovery Layer)  │     │(Connection) │
└──────────────────┘     └───────────────────┘     └─────────────┘
         │                        │
         │                        ▼
         │               ┌───────────────────┐
         │               │  RecoveryConfig   │◄──── client.json
         │               │  (Configuration)  │◄──── Environment
         │               └───────────────────┘
         │
         ▼
┌──────────────────┐
│    PTDisplay     │◄──── Connection status bar
│  (Status UI)     │
└──────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `IPCRecoveryClient` | `jaato-tui/ipc_recovery.py` | Wraps IPCClient with reconnection logic |
| `RecoveryConfig` | `jaato-tui/client_config.py` | Configuration dataclass |
| `load_client_config()` | `jaato-tui/client_config.py` | Loads config from files + env |
| `ConnectionState` | `jaato-tui/ipc_recovery.py` | State machine enum |
| `ConnectionStatus` | `jaato-tui/ipc_recovery.py` | Status for UI display |

## Connection State Machine

```
                                      ┌────────────────┐
                                      │                │
                                      ▼                │
┌──────────────┐  connect()   ┌─────────────┐  connection  │
│ DISCONNECTED │─────────────►│ CONNECTING  │◄─── lost ────┘
└──────────────┘              └─────────────┘
       ▲                            │
       │                     success│
       │                            ▼
       │                     ┌─────────────┐
       │  give_up()          │  CONNECTED  │────────┐
       └─────────────────────┴─────────────┘        │
       │                            │               │
       │                     detach/close      connection
       │                            │              lost
       │                            ▼               │
       │                     ┌─────────────┐        │
       └─────────────────────│DISCONNECTING│        │
                             └─────────────┘        │
                                    │               │
                                    ▼               ▼
                             ┌─────────────┐ ┌─────────────┐
                             │   CLOSED    │ │RECONNECTING │
                             └─────────────┘ └─────────────┘
```

### State Descriptions

| State | Description |
|-------|-------------|
| `DISCONNECTED` | Initial state, or recovery gave up |
| `CONNECTING` | Attempting initial connection |
| `CONNECTED` | Active connection, events flowing |
| `RECONNECTING` | Connection lost, auto-recovery in progress |
| `DISCONNECTING` | Graceful disconnect initiated |
| `CLOSED` | Terminal state, no more attempts |

### State Transitions

| From | To | Trigger |
|------|-----|---------|
| `DISCONNECTED` | `CONNECTING` | `connect()` called |
| `CONNECTING` | `CONNECTED` | Successful handshake |
| `CONNECTING` | `DISCONNECTED` | Connection failed |
| `CONNECTED` | `RECONNECTING` | Connection lost |
| `RECONNECTING` | `CONNECTING` | Backoff timer fires |
| `RECONNECTING` | `CLOSED` | Max attempts exceeded |
| `*` | `CLOSED` | `close()` called |

## Configuration

Recovery behavior is configurable via JSON files and/or environment variables.

### Configuration Precedence

Configuration is loaded and merged in this order (highest precedence wins):

1. **Environment variables** - Quick overrides
2. **Project config** - `.jaato/client.json` in workspace
3. **User config** - `~/.jaato/client.json`
4. **Built-in defaults** - Sensible defaults

### JSON Configuration

Create `.jaato/client.json` in your project or `~/.jaato/client.json` for user-wide settings:

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

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable automatic reconnection |
| `max_attempts` | int | `10` | Maximum reconnection attempts before giving up |
| `base_delay` | float | `1.0` | Initial backoff delay in seconds |
| `max_delay` | float | `60.0` | Maximum backoff delay (caps exponential growth) |
| `jitter_factor` | float | `0.3` | Random jitter range (0.3 = ±30% variation) |
| `connection_timeout` | float | `5.0` | Timeout for each connection attempt |
| `reattach_session` | bool | `true` | Auto-reattach to previous session after reconnect |

### Environment Variables

Environment variables override file-based configuration:

| Variable | Config Key | Example |
|----------|------------|---------|
| `JAATO_IPC_AUTO_RECONNECT` | `recovery.enabled` | `true` |
| `JAATO_IPC_RETRY_MAX_ATTEMPTS` | `recovery.max_attempts` | `10` |
| `JAATO_IPC_RETRY_BASE_DELAY` | `recovery.base_delay` | `1.0` |
| `JAATO_IPC_RETRY_MAX_DELAY` | `recovery.max_delay` | `60.0` |
| `JAATO_IPC_RETRY_JITTER` | `recovery.jitter_factor` | `0.3` |
| `JAATO_IPC_CONNECTION_TIMEOUT` | `recovery.connection_timeout` | `5.0` |
| `JAATO_IPC_REATTACH_SESSION` | `recovery.reattach_session` | `true` |

### Example: Faster Recovery for Development

For local development where the server restarts frequently:

```json
{
  "recovery": {
    "max_attempts": 20,
    "base_delay": 0.5,
    "max_delay": 10.0
  }
}
```

### Example: Conservative Recovery for Production

For production where you want to avoid hammering a struggling server:

```json
{
  "recovery": {
    "max_attempts": 5,
    "base_delay": 5.0,
    "max_delay": 120.0
  }
}
```

## Exponential Backoff Algorithm

The recovery mechanism uses exponential backoff with jitter to avoid thundering herd problems.

### Delay Calculation

```
delay = min(max_delay, base_delay * 2^(attempt - 1))
jitter = delay * jitter_factor * random(-1, 1)
final_delay = max(0.1, delay + jitter)
```

### Example Delays (Default Config)

| Attempt | Base Delay | With Jitter (±30%) |
|---------|------------|-------------------|
| 1 | 1.0s | 0.7s - 1.3s |
| 2 | 2.0s | 1.4s - 2.6s |
| 3 | 4.0s | 2.8s - 5.2s |
| 4 | 8.0s | 5.6s - 10.4s |
| 5 | 16.0s | 11.2s - 20.8s |
| 6 | 32.0s | 22.4s - 41.6s |
| 7+ | 60.0s (capped) | 42.0s - 78.0s |

## User Interface

### Status Bar

During reconnection, the status bar shows:

```
Connection lost. Reconnecting in 4s... (attempt 3/10)
```

After successful reconnection:

```
Connection restored!
```

After max attempts exceeded:

```
Connection lost permanently: Max retries exceeded
```

After connecting to an incompatible (too-old) server:

```
Error: Server version 0.2.10 is not supported by this client (requires >= 0.2.27). Please upgrade the server.
```

`IncompatibleServerError` is classified as a **permanent** error — the recovery client will not retry, since an old server won't become newer on reconnection.

### User Actions During Recovery

| Action | Behavior |
|--------|----------|
| Wait | Automatic reconnection continues |
| Type input | Queued until reconnected |
| Ctrl+C | Cancel reconnection, prompt for exit |
| `quit` | Cancel reconnection, exit cleanly |

## Session Preservation

### What's Preserved

- **Session ID** - Tracked for reattachment
- **Conversation history** - Persisted on server disk
- **Subagent state** - Registry + per-agent conversation history (dedicated persistence)
- **TODO plans** - Agent-plan mapping + plan files (dedicated persistence)
- **Plugin session state** - Generic mechanism for all other plugins (see below)
- **Turn accounting** - Token usage per turn
- **Budget state** - Conversation budget snapshot
- **Interrupted turns** - Pending tool calls (recovered with synthetic errors)

### Plugin State Persistence

SessionManager persists plugin in-memory state via two mechanisms:

**1. Generic mechanism** — Iterates all exposed plugins, calls `get_persistence_state()` on each that implements it, stores results in `metadata['plugin_states']` inside the session JSON. On restore, calls `restore_persistence_state(state)` on each.

Currently integrated:

| Plugin | What's Persisted | What's Restored |
|--------|-----------------|-----------------|
| `service_connector` | Discovered service alias list | In-memory cache pre-warmed from SchemaStore YAML |
| `reliability` | Tool trust states, turn index, session settings, escalation overrides | Full tool escalation state rebuilt |

**2. Dedicated mechanism** — Plugins with complex multi-file state or non-standard restore signatures use custom save/load orchestration in SessionManager:

| Plugin | Why Dedicated | Storage |
|--------|---------------|---------|
| `subagent` | Per-agent state files (MB+), restore needs runtime instance | `sessions/<id>/subagents/*.json` |
| `todo` | Plan files managed by storage backend, restore re-registers event bus dependencies | `sessions/<id>/plans/` |

See [Plugin README — Session Persistence](../shared/plugins/README.md#plugin-with-session-persistence) for implementation guide.

### What's Lost

- **Active IPC connection** - Replaced by new connection
- **In-flight requests** - Pending permission responses, etc.
- **Real-time event stream** - Restarted after reconnect
- **Interactive shell sessions** - PTY processes cannot be serialized
- **MCP server-side state** - Servers are reconnected, but any in-process server state is server-dependent

### Session Reattachment

After successful reconnection:

1. Client sends `session.attach` command with stored session ID
2. Server loads session from disk if evicted from memory
3. Server sends `SessionInfoEvent` with session state
4. Client continues normal operation

If session is not found (deleted, expired):

- Client logs warning
- User may need to create new session

## Error Classification

The recovery mechanism classifies errors to determine retry behavior:

### Transient Errors (Will Retry)

- `ConnectionRefusedError` - Server may be restarting
- `ConnectionResetError` - Connection dropped
- `asyncio.TimeoutError` - Network hiccup
- Socket timeout
- "Connection refused" messages

### Permanent Errors (No Retry)

- `FileNotFoundError` - Socket file deleted
- "Permission denied" - Auth issues
- "Authentication failed" - Credential issues

## Implementation Details

### Thread Safety

- State transitions protected by `asyncio.Lock`
- Status callbacks invoked synchronously
- Event queue is `asyncio.Queue` (thread-safe)

### Cancellation

Recovery can be cancelled by:

- Calling `close()` on the client
- Setting `should_exit` flag in rich_client
- Ctrl+C interrupt

### Resource Cleanup

On each reconnection attempt:

1. Old client is disconnected
2. Socket/pipe is closed
3. New client is created
4. Fresh connection established

## Troubleshooting

### Recovery Not Working

1. Check if recovery is enabled:
   ```bash
   echo $JAATO_IPC_AUTO_RECONNECT
   cat .jaato/client.json
   ```

2. Check server status:
   ```bash
   .venv/bin/python -m server --status
   ```

3. Check socket file exists:
   ```bash
   ls -la /tmp/jaato.sock
   ```

### Recovery Too Slow

Reduce delays in configuration:

```json
{
  "recovery": {
    "base_delay": 0.5,
    "max_delay": 10.0
  }
}
```

### Recovery Gives Up Too Quickly

Increase max attempts:

```json
{
  "recovery": {
    "max_attempts": 20
  }
}
```

### Session Not Reattaching

1. Check session still exists:
   ```bash
   ls -la ~/.jaato/sessions/
   ```

2. Verify session ID is tracked (check logs)

3. Ensure `reattach_session` is enabled

## API Reference

### IPCRecoveryClient

```python
class IPCRecoveryClient:
    def __init__(
        self,
        socket_path: str,
        config: Optional[RecoveryConfig] = None,
        auto_start: bool = True,
        env_file: str = ".env",
        workspace_path: Optional[Path] = None,
        on_status_change: Optional[StatusCallback] = None,
    ): ...

    # Properties
    @property
    def state(self) -> ConnectionState: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def is_reconnecting(self) -> bool: ...
    @property
    def session_id(self) -> Optional[str]: ...

    # Connection
    async def connect(self, timeout: float = 5.0) -> bool: ...
    async def disconnect(self) -> None: ...
    async def close(self) -> None: ...

    # Session
    def set_session_id(self, session_id: str) -> None: ...
    async def attach_session(self, session_id: str) -> bool: ...

    # Events
    async def events(self) -> AsyncIterator[Event]: ...

    # Status
    def get_status(self) -> ConnectionStatus: ...
```

### RecoveryConfig

```python
@dataclass
class RecoveryConfig:
    enabled: bool = True
    max_attempts: int = 10
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.3
    connection_timeout: float = 5.0
    reattach_session: bool = True
```

### ConnectionStatus

```python
@dataclass
class ConnectionStatus:
    state: ConnectionState
    attempt: int = 0
    max_attempts: int = 0
    next_retry_in: Optional[float] = None
    last_error: Optional[str] = None
    session_id: Optional[str] = None
    client_id: Optional[str] = None
```

### Loading Configuration

```python
from client_config import load_client_config, get_recovery_config

# Load full client config
config = load_client_config(workspace_path=Path.cwd())
print(config.recovery.max_attempts)

# Or just recovery config
recovery = get_recovery_config(workspace_path=Path.cwd())
print(recovery.enabled)
```

## Related Documentation

- [Architecture Overview](architecture.md) - Server-first architecture
- [Sequence Diagrams](sequence-diagram-architecture.md) - Client-server interactions
