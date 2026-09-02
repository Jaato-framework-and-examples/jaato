# Daemon Extensions

Daemon extensions are the mechanism by which external packages (like `jaato-premium`)
add runtime functionality to the jaato server without modifying the public codebase.

## Architecture

The public repo provides **four generic extension points**. None are specific to any
particular feature — they are general-purpose hooks that any extension can use.

```
┌────────────────────────────────────────────────────────┐
│  jaato-server (public)                                 │
│                                                        │
│  ┌──────────────┐   ┌──────────────┐                   │
│  │ __main__.py  │   │ websocket.py │                   │
│  │              │   │              │                   │
│  │ 1. Extension │   │ 2. WS        │                   │
│  │    Lifecycle │   │    Intercept  │                   │
│  └──────┬───────┘   └──────────────┘                   │
│         │                                              │
│  ┌──────┴───────┐   ┌──────────────┐                   │
│  │session_mgr.py│   │ environment/ │                   │
│  │              │   │  plugin.py   │                   │
│  │ 3. Session   │   │              │                   │
│  │    Hooks     │   │ 4. Custom    │                   │
│  └──────────────┘   │    Aspects   │                   │
│                     └──────────────┘                   │
│                     ┌──────────────┐                   │
│                     │  subagent/   │                   │
│                     │  plugin.py   │                   │
│                     │              │                   │
│                     │ 5. Remote    │                   │
│                     │    Handler   │                   │
│                     └──────────────┘                   │
└────────────────────────────────────────────────────────┘
          ▲
          │ jaato.extensions entry point
          │
┌─────────┴──────────────────────────────────────────────┐
│  jaato-premium (or any extension package)              │
│                                                        │
│  extension.py                                          │
│  ├── __init__(context) — create infrastructure         │
│  ├── start() — register hooks, start services          │
│  └── stop()  — shut down services                      │
└────────────────────────────────────────────────────────┘
```

## Extension Point 1: Daemon Extensions (`__main__.py`)

**Entry point group:** `jaato.extensions`

Extensions are discovered from Python entry points at daemon startup. Each entry
point must resolve to a factory function:

```python
def create_extension(context: _ExtensionContext) -> Extension
```

### Extension Context

The factory receives an `_ExtensionContext` with these attributes:

| Attribute         | Type                    | Description                          |
|-------------------|-------------------------|--------------------------------------|
| `session_manager`   | `SessionManager`        | The daemon's session manager         |
| `ws_server`         | `JaatoWSServer \| None` | WebSocket server (None if disabled)  |
| `web_socket`        | `str \| None`           | Raw `--web-socket` CLI arg           |
| `ipc_socket`        | `str \| None`           | Raw `--ipc-socket` CLI arg           |
| `server_name`       | `str \| None`           | `--server-name` CLI arg              |
| `dashboard_port`    | `int \| None`           | `--dashboard-port` CLI arg           |
| `available_plugins`    | `frozenset[str]`        | Tool/enrichment plugin names            |
| `plugin_registry`      | `PluginRegistry`        | Registry for config schema queries      |
| `available_gc_plugins` | `frozenset[str]`        | GC plugin names (e.g. `gc_truncate`)    |
| `gc_plugin_factories`  | `dict[str, Callable]`   | GC plugin factories for introspection   |

### Extension Protocol

The returned object must implement:

- `async start()` — Called after transport servers are listening. Register hooks here.
- `async stop()` — Called before daemon shutdown. Clean up resources here.

Extensions start in discovery order, stop in reverse order.

### Registration (pyproject.toml)

```toml
[project.entry-points."jaato.extensions"]
my_extension = "my_package.ext:create_extension"
```

## Extension Point 2: WebSocket Connection Interceptors (`websocket.py`)

**Method:** `ws_server.set_connection_interceptor(check, handler)`

Intercepts incoming WebSocket connections before normal client handling.

| Parameter | Type                          | Description                              |
|-----------|-------------------------------|------------------------------------------|
| `check`   | `(websocket) -> bool`         | Returns True if this interceptor handles |
| `handler` | `async (websocket) -> None`   | Takes over the connection lifecycle      |

Interceptors are evaluated in registration order. First match wins.

```python
# Example: route peer connections
ws_server.set_connection_interceptor(
    check=lambda ws: (
        ws.request
        and ws.request.headers.get("X-My-Header") == "true"
    ),
    handler=self._handle_special_connection,
)
```

## Extension Point 3: Session Hooks (`session_manager.py`)

**Method:** `session_manager.add_session_hook(hook)`

Registers a callback invoked after each session is fully initialized (plugins
loaded, provider connected, tools configured).

| Parameter | Type                           | Description                     |
|-----------|--------------------------------|---------------------------------|
| `hook`    | `(server: JaatoServer) -> None` | Called with the session's server |

If a hook raises, it is logged and subsequent hooks still run.

```python
# Example: wire plugins in each new session
def _on_session_ready(self, server):
    registry = server.registry
    env = registry.get_plugin("environment")
    if env and hasattr(env, 'register_aspect'):
        env.register_aspect("my_data", self._handler, description="my data")

ctx.session_manager.add_session_hook(self._on_session_ready)
```

## Extension Point 4: Custom Environment Aspects (`environment/plugin.py`)

**Method:** `env_plugin.register_aspect(name, handler, description="")`

Adds a dynamic aspect to the `get_environment` tool at runtime.

| Parameter     | Type           | Description                                |
|---------------|----------------|--------------------------------------------|
| `name`        | `str`          | Aspect name (e.g., `"cluster_topology"`)   |
| `handler`     | `() -> dict`   | Returns the aspect data when queried       |
| `description` | `str`          | Human-readable description for tool schema |

Custom aspects are:
- **Excluded from `"all"`** — must be queried explicitly by name
- **Auto-added to VALID_ASPECTS** and the tool schema enum
- **Described in the tool schema** using the `description` argument

```python
env_plugin.register_aspect(
    "jaato_agentic_servers",
    lambda: self._get_cluster_topology(),
    description="cluster topology: this server + peers",
)
```

## Extension Point 5: Remote Spawn Handler (`subagent/plugin.py`)

**Method:** `subagent_plugin.register_remote_handler(handler)`

Enables the `server` parameter on `spawn_subagent` for remote delegation.

The handler receives these keyword arguments:

| Argument        | Type            | Description                    |
|-----------------|-----------------|--------------------------------|
| `server`        | `str`           | Target peer server name        |
| `task`          | `str`           | The prompt/task                |
| `profile_name`  | `str`           | Profile name (empty for inline)|
| `context`       | `Any`           | Context string or dict         |
| `inline_config` | `dict \| None`  | Optional inline overrides      |
| `custom_name`   | `str`           | Optional custom agent name     |

Returns a dict with `success` (bool) and either `error` (str) on failure or
`subagent_id`, `status`, `remote_server`, `message` on success.

Without this handler, the `server` parameter returns a clear error:
*"Remote subagent delegation requires jaato-premium."*

## Typical Extension Flow

1. Daemon discovers extension via `jaato.extensions` entry point
2. Factory creates extension with `_ExtensionContext`
3. `start()` is called after transport servers are up:
   - Registers WS interceptor (if needed)
   - Registers session hook
   - Starts background services
4. For each new session, session hook fires:
   - Registers custom aspects on environment plugin
   - Registers remote handler on subagent plugin
5. `stop()` is called before shutdown:
   - Stops background services
   - Cleans up resources

## Testing Without Premium

When no extensions are installed:
- `_load_extensions()` finds no entry points and returns immediately
- Session hooks list is empty; `_run_session_hooks()` is a no-op
- No custom aspects registered; built-in aspects work normally
- `server` param on `spawn_subagent` returns a clear error message
- Zero gossip-related imports or code paths execute
