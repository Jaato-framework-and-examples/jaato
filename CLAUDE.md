# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**jaato** ("just another agentic tool orchestrator") is a framework for:
- Multi-provider AI SDK integration (Google GenAI, Anthropic, etc.)
- Function calling patterns with LLMs
- Tool orchestration (CLI tools and MCP servers)

## Commands

### Environment Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Running the Server (Multi-Client Mode)
```bash
# Start server as daemon with IPC socket
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon

# Start server with both IPC and WebSocket
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --web-socket :8080 --daemon

# Check server status
.venv/bin/python -m server --status

# Stop server
.venv/bin/python -m server --stop

# Connect TUI client to running server
.venv/bin/python rich-client/rich_client.py --connect /tmp/jaato.sock
```

### Running Tests
```bash
.venv/bin/pytest                          # All tests
.venv/bin/pytest shared/tests/            # Core tests
.venv/bin/pytest shared/plugins/cli/tests/ # Plugin tests
.venv/bin/pytest -v                       # Verbose output
```

Test organization:
- Core tests: `shared/tests/`
- Plugin tests: `shared/plugins/<plugin>/tests/`
- Provider tests: `shared/plugins/model_provider/<provider>/tests/`

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed diagrams and component interactions.

### Server Components (`server/`)

The framework uses a server-first architecture where the server runs as a daemon and clients connect via IPC or WebSocket.

- **`server/__main__.py`**: Entry point with daemon mode, PID management
  - `--ipc-socket PATH`: Unix domain socket for local clients
  - `--web-socket [HOST:]PORT`: WebSocket for remote clients
  - `--daemon`: Run as background process
  - `--status`/`--stop`: Server management

- **`server/core.py`**: `JaatoServer` - UI-agnostic core logic
  - Wraps `JaatoClient` with event emission instead of callbacks
  - Handles permission requests, tool execution, streaming

- **`server/events.py`**: Event protocol (25+ typed events)
  - Server→Client: `AgentOutputEvent`, `PermissionRequestedEvent`, `PlanUpdatedEvent`, etc.
  - Client→Server: `SendMessageRequest`, `PermissionResponseRequest`, `StopRequest`, etc.

- **`server/session_manager.py`**: Multi-session orchestration with disk persistence
- **`server/ipc.py`**: Unix domain socket server (length-prefixed framing)
- **`server/websocket.py`**: WebSocket server for remote clients

### Core Components (`shared/`)

- **jaato_client.py**: `JaatoClient` - Backwards-compatible facade wrapping `JaatoRuntime` + `JaatoSession`
  - `connect()`, `configure_tools()`, `send_message()` - core methods
  - `get_runtime()` - access shared runtime for subagent creation
  - `get_session()` - access main session

- **jaato_runtime.py**: `JaatoRuntime` - Shared environment
  - Manages provider config, plugin registry, permissions, ledger
  - `create_session(model, tools, system_instructions)` - spawn lightweight sessions

- **jaato_session.py**: `JaatoSession` - Per-agent conversation state
  - `send_message()`, `get_history()`, `reset_session()` - conversation methods
  - Sessions share runtime resources but maintain isolated state

- **ai_tool_runner.py**: `ToolExecutor` - Registry mapping tool names to callables with permission checking

- **mcp_context_manager.py**: `MCPClientManager` - Multi-server MCP client manager
  - Auto-discovers tools from connected servers
  - Supports `call_tool_auto()` to find which server has a tool

- **token_accounting.py**: `TokenLedger` - Token usage tracking with rate-limit retries

### Plugin System (`shared/plugins/`)

Three plugin types:

**Tool Plugins** - Provide tools the model can invoke:
- `PluginRegistry`: Discovers and manages tool plugins
- `cli/`: Shell commands | `mcp/`: MCP servers | `permission/`: Permission control
- `file_edit/`, `todo/`, `web_search/`, `filesystem_query/`, etc.

**GC Plugins** - Context garbage collection strategies:
- `gc_truncate/`: Simple truncation
- `gc_summarize/`: Summarization-based
- `gc_hybrid/`: Combined approach (recent preserved, middle summarized, ancient truncated)

**Model Provider Plugins** - SDK abstraction for multi-provider support:
- `model_provider/types.py`: Provider-agnostic types (`ToolSchema`, `Message`, `ProviderResponse`)
- `model_provider/google_genai/`: Google GenAI/Vertex AI
- `model_provider/anthropic/`: Anthropic Claude API
- `model_provider/claude_cli/`: Claude Code CLI wrapper (uses subscription, not API credits)
- `model_provider/github_models/`: GitHub Models API (uses `azure-ai-inference` SDK)
- `model_provider/antigravity/`: Google Antigravity IDE backend (Gemini 3, Claude via Google OAuth)

### Tool Execution Flow

1. Create `JaatoClient` and connect: `jaato.connect(project, location, model)`
2. Configure tools: `jaato.configure_tools(registry, permission_plugin)`
3. Send message with callback:
   ```python
   response = jaato.send_message(prompt, on_output=lambda source, text, mode: print(f"[{source}]: {text}"))
   ```
   Callback receives `(source, text, mode)` for each output chunk.
4. SDK chat API handles function calling loop until model returns text without function calls
5. Access history: `jaato.get_history()` | Reset: `jaato.reset_session()`

### Parallel Tool Execution

When model returns multiple function calls, jaato executes them in parallel using a thread pool.
- Enabled by default (`JAATO_PARALLEL_TOOLS=true`)
- Set `JAATO_PARALLEL_TOOLS=false` to disable
- Maximum 8 concurrent tools per turn
- Thread-safe callbacks via thread-local storage

### Subagent Architecture

Subagents share the parent's `JaatoRuntime` but get their own `JaatoSession`:
- **No redundant connections** - subagents share provider config
- **Fast spawning** - `create_session()` is lightweight
- **Resource sharing** - registry, permissions, ledger shared

### MCP Server Configuration

MCP servers are configured in `.mcp.json`:
```json
{
  "mcpServers": {
    "Atlassian": { "type": "stdio", "command": "mcp-atlassian" }
  }
}
```

### Streaming & Cancellation

Key types in `shared/plugins/model_provider/types.py`:
- `CancelToken`: Thread-safe cancellation signaling
- `CancelledException`: Raised when operation is cancelled
- `FinishReason.CANCELLED`: Indicates cancelled generation

Session/client methods:
- `client.stop()` / `session.request_stop()`: Request cancellation
- `client.is_processing` / `session.is_running`: Check if message in progress
- `client.set_streaming_enabled(bool)`: Toggle streaming mode

### Proactive Garbage Collection

The framework monitors token usage during streaming and automatically triggers GC when thresholds are exceeded:

```python
from shared.plugins.gc import GCConfig

gc_config = GCConfig(
    threshold_percent=80.0,    # Trigger when context is 80% full
    preserve_recent_turns=5,   # Keep last 5 turns
    auto_trigger=True,
)
client.set_gc_plugin(gc_plugin, gc_config)
```

### Deferred Tool Loading

Tools have a `discoverability` attribute: `"core"` (always loaded) or `"discoverable"` (on-demand).
Model uses `list_tools()` → `get_tool_schemas()` workflow to discover tools.

- Enabled by default (`JAATO_DEFERRED_TOOLS=true`)
- Core tools: introspection, file_edit, cli, filesystem_query, todo, clarification

### Plugin Auto-Wiring

Plugins are automatically wired during initialization - no manual wiring needed:

| Method | When Called | By |
|--------|-------------|-----|
| `set_plugin_registry(registry)` | During `expose_tool()` | PluginRegistry |
| `set_session(session)` | During `configure()` | JaatoSession |
| `set_workspace_path(path)` | After `expose_all()` | PluginRegistry |

## Key Environment Variables

### Google GenAI / Vertex AI
| Variable | Purpose |
|----------|---------|
| `PROJECT_ID` | GCP project ID |
| `LOCATION` | Vertex AI region (e.g., `us-central1`, `global`) |
| `MODEL_NAME` | Gemini model (e.g., `gemini-2.5-flash`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key JSON |

### GitHub Models
| Variable | Purpose |
|----------|---------|
| `GITHUB_TOKEN` | GitHub PAT with `models: read` permission |
| `JAATO_GITHUB_ORGANIZATION` | Organization for billing attribution |

### Anthropic Claude
| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (uses API credits) |
| `ANTHROPIC_AUTH_TOKEN` | OAuth token for Claude Pro/Max subscription |

**Authentication Options (in priority order):**
1. **PKCE OAuth Login** (recommended for subscription): `oauth_login()` from `shared.plugins.model_provider.anthropic`
2. **OAuth Token** (`sk-ant-oat01-...`): From `claude setup-token`
3. **API Key** (`sk-ant-api03-...`): Uses API credits

Configuration options via `ProviderConfig.extra`:
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_caching` | bool | False | Enable prompt caching (90% cost reduction) |
| `enable_thinking` | bool | False | Enable extended thinking |
| `thinking_budget` | int | 10000 | Max thinking tokens when enabled |
| `cache_history` | bool | True | Cache historical messages |

### Claude CLI Provider
| Variable | Purpose |
|----------|---------|
| `JAATO_CLAUDE_CLI_PATH` | Path to claude CLI (default: from PATH) |
| `JAATO_CLAUDE_CLI_MODE` | `delegated` (CLI handles tools) or `passthrough` (jaato handles) |
| `JAATO_CLAUDE_CLI_MAX_TURNS` | Maximum agentic turns |

Requirements: `npm install -g @anthropic-ai/claude-code` + `claude login`

Benefits:
- Uses Claude Pro/Max subscription without API credits
- Leverages CLI's built-in tools (Read, Write, Edit, Bash, etc.)
- Automatic prompt caching by CLI

### Antigravity (Google IDE Backend)
| Variable | Purpose |
|----------|---------|
| `JAATO_ANTIGRAVITY_QUOTA` | `antigravity` (default) or `gemini-cli` |
| `JAATO_ANTIGRAVITY_THINKING_LEVEL` | Gemini 3: `minimal`/`low`/`medium`/`high` |
| `JAATO_ANTIGRAVITY_THINKING_BUDGET` | Claude thinking budget (default: 8192) |
| `JAATO_ANTIGRAVITY_AUTO_ROTATE` | Enable multi-account rotation (default: `true`) |

Auth: `oauth_login()` from `shared.plugins.model_provider.antigravity`

Available Models:
- Antigravity quota: `antigravity-gemini-3-pro/flash`, `antigravity-claude-sonnet-4-5[-thinking]`
- Gemini CLI quota: `gemini-2.5-flash/pro`, `gemini-3-flash/pro-preview`

### General
| Variable | Purpose |
|----------|---------|
| `AI_USE_CHAT_FUNCTIONS` | Enable function calling mode (`1`/`true`) |
| `LEDGER_PATH` | Output path for token accounting JSONL |
| `JAATO_GC_THRESHOLD` | GC trigger threshold % (default: 80.0) |
| `JAATO_PARALLEL_TOOLS` | Enable parallel tool execution (default: `true`) |
| `JAATO_DEFERRED_TOOLS` | Enable deferred tool loading (default: `true`) |
| `JAATO_AMBIGUOUS_WIDTH` | Width for East Asian Ambiguous chars in tables (`1` default, `2` for CJK terminals) |

### Rate Limiting
| Variable | Purpose |
|----------|---------|
| `AI_REQUEST_INTERVAL` | Minimum seconds between requests (default: 0) |
| `AI_RETRY_ATTEMPTS` | Max retry attempts (default: 5) |
| `AI_RETRY_BASE_DELAY` | Initial retry delay seconds (default: 1.0) |
| `AI_RETRY_MAX_DELAY` | Maximum retry delay seconds (default: 30.0) |

## Rich Client Commands

### Authentication Commands
```
anthropic-auth login/logout/status     # Anthropic OAuth
antigravity-auth login/logout/status   # Google OAuth
```

### Session Commands
```
reset                       # Reset conversation history
model <name>                # Switch to a different model
keybindings reload          # Reload keybindings from config
```

### Permission Commands
```
permissions [show|whitelist|blacklist|suspend|resume|status]
```

Permission responses: `y`(yes), `n`(no), `a`(always), `t`(turn), `i`(idle), `once`, `never`, `all`

- **turn**: Approval lasts until model finishes responding
- **idle**: Approval persists across consecutive turns until session goes idle

### Vision Capture (TUI Screenshots)
```
screenshot [nosend|format F|auto|interval N|help]
```
Captures TUI as SVG/PNG to `$JAATO_VISION_DIR` (default: `/tmp/jaato_vision`).

## Rich Client Keybindings

Config files: `.jaato/keybindings.json` (project) or `~/.jaato/keybindings.json` (user)

Key syntax (prompt_toolkit): `enter`, `c-c` (Ctrl+C), `f1`, `pageup`, `["escape", "enter"]`

Default keybindings: `submit`=enter, `cancel`=c-c, `exit`=c-d, `toggle_plan`=c-p, `toggle_tools`=c-t

## Rich Client Theming

Built-in themes: `dark` (default), `light`, `high-contrast`

Switch: `/theme [dark|light|high-contrast|reload]`

Custom theme: Create `theme.json` in `.jaato/` or `~/.jaato/` with `colors` object containing: `primary`, `secondary`, `success`, `warning`, `error`, `muted`, `background`, `surface`, `text`, `text_muted`

## Telemetry (OpenTelemetry)

See [docs/opentelemetry-design.md](docs/opentelemetry-design.md) for comprehensive design.

```bash
.venv/bin/pip install -r requirements-telemetry.txt
export JAATO_TELEMETRY_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Span hierarchy: `jaato.turn` → `jaato.tool` → `jaato.permission`

Key attributes:
- Turn: `session_id`, `agent_type`, `turn_index`, `streaming`, `cancelled`
- Tool: `tool.name`, `tool.plugin_type`, `tool.success`, `tool.duration_seconds`

## Additional Documentation

- [Architecture Overview](docs/architecture.md) - Server-first architecture, event protocol, component diagrams
- [Sequence Diagrams](docs/sequence-diagram-architecture.md) - Client-server interaction, tool execution flows
- [OpenTelemetry Design](docs/opentelemetry-design.md) - Comprehensive OTel tracing integration
- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
