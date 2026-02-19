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
.venv/bin/pip install -e jaato-sdk/. -e "jaato-server/.[all]" -e "jaato-tui/.[all]"
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
.venv/bin/python jaato-tui/rich_client.py --connect /tmp/jaato.sock
```

### Running Tests
```bash
.venv/bin/pytest                                        # All tests
.venv/bin/pytest jaato-server/shared/tests/             # Core tests
.venv/bin/pytest jaato-server/shared/plugins/cli/tests/ # Plugin tests
.venv/bin/pytest -v                                     # Verbose output
```

Test organization:
- Core tests: `jaato-server/shared/tests/`
- Plugin tests: `jaato-server/shared/plugins/<plugin>/tests/`
- Provider tests: `jaato-server/shared/plugins/model_provider/<provider>/tests/`

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed diagrams and component interactions.

### Server Components (`jaato-server/server/`)

The framework uses a server-first architecture where the server runs as a daemon and clients connect via IPC or WebSocket.

- **`server/__main__.py`**: Entry point with daemon mode, PID management
  - `--ipc-socket PATH`: Unix domain socket for local clients
  - `--web-socket [HOST:]PORT`: WebSocket for remote clients
  - `--socket-mode MODE`: Octal file permissions for the IPC socket (default: `666`). Use `660` to restrict to owner and group only.
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

### Core Components (`jaato-server/shared/`)

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

### Plugin System (`jaato-server/shared/plugins/`)

Three plugin types:

**Tool Plugins** - Provide tools the model can invoke:
- `PluginRegistry`: Discovers and manages tool plugins
- `cli/`: Shell commands | `mcp/`: MCP servers | `permission/`: Permission control
- `interactive_shell/`: Interactive PTY sessions (REPLs, password prompts, wizards, debuggers)
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
- `model_provider/ollama/`: Ollama local models (Anthropic-compatible API)

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

### Tool Traits

Tools can declare semantic **traits** on their `ToolSchema` via the `traits` field (a `FrozenSet[str]`). Traits drive cross-cutting behavior without hardcoding tool names in session or plugin code.

**Currently defined traits:**

| Constant | Value | Contract |
|----------|-------|----------|
| `TRAIT_FILE_WRITER` | `"file_writer"` | Tool writes/modifies files. Result must include `path` (str), `files_modified` (list), or `changes[].file`. Triggers full-JSON enrichment (LSP diagnostics, artifact tracking). |

**How it works:**
1. Tool schemas declare traits: `traits=frozenset({TRAIT_FILE_WRITER})`
2. Session queries `registry.get_tool_traits(tool_name)` to decide enrichment strategy
3. Enrichment plugins (LSP, artifact_tracker) extract file paths generically from the result dict

**Adding a trait to a new tool:**
1. Import the constant: `from ..model_provider.types import TRAIT_FILE_WRITER`
2. Add to the `ToolSchema`: `traits=frozenset({TRAIT_FILE_WRITER})`
3. Ensure the tool result dict includes the required keys (`path`, `files_modified`, or `changes`)

**Defining a new tool trait:**
1. Add a `TRAIT_*` constant in `shared/plugins/model_provider/types.py` with a docstring documenting the contract
2. Update consumers (session, plugins) to query `get_tool_traits()` for the new trait

### Plugin-Level Traits

Plugins themselves can declare **plugin-level traits** via a `plugin_traits` class attribute (`FrozenSet[str]`). These work like tool traits but identify *plugin* capabilities rather than individual tool behaviors.

**Currently defined plugin traits:**

| Constant | Value | Contract |
|----------|-------|----------|
| `TRAIT_AUTH_PROVIDER` | `"auth_provider"` | Plugin provides interactive authentication for a model provider. Must also expose `provider_name` property identifying which provider. |

**How it works:**
1. Plugin declares: `plugin_traits = frozenset({TRAIT_AUTH_PROVIDER})`
2. Server filters plugins by trait: `TRAIT_AUTH_PROVIDER in plugin.plugin_traits`
3. Among matching plugins, server reads `provider_name` to select the right one

**Adding a plugin trait to a new plugin:**
1. Import the constant: `from shared.plugins.base import TRAIT_AUTH_PROVIDER`
2. Add class attribute: `plugin_traits = frozenset({TRAIT_AUTH_PROVIDER})`
3. Implement the contract (e.g., `provider_name` property for auth plugins)

**Defining a new plugin trait:**
1. Add a `TRAIT_*` constant in `shared/plugins/base.py` with a docstring documenting the contract
2. Update consumers (server, daemon) to query `getattr(plugin, 'plugin_traits', frozenset())`

### Interactive Shell Sessions (`shared/plugins/interactive_shell/`)

The `interactive_shell` plugin lets the model drive any user-interactive command by spawning persistent PTY sessions. Unlike `cli/` (which uses `subprocess` and can only run non-interactive commands), this plugin uses `pexpect` to provide a real pseudo-terminal where the model can read output and send input back and forth.

**Design:** No expect patterns. The plugin uses idle-based output detection — it reads until the process stops producing output (~500ms of silence), then returns whatever appeared. The model reads the raw output, understands what the program is asking (password prompt, menu, REPL prompt, etc.), and decides what to type next.

**Tools** (all `discoverability="discoverable"`):

| Tool | Purpose |
|------|---------|
| `shell_spawn` | Start a new interactive process. Called **once** per command. Returns `session_id` + initial output. |
| `shell_input` | Send text to an **existing** session (by `session_id`). Used for **all** subsequent interactions after spawn. |
| `shell_read` | Read pending output without sending input. For checking on long-running operations. |
| `shell_control` | Send control keys: `c-c` (interrupt), `c-d` (EOF), `c-z` (suspend), `c-l` (clear). |
| `shell_close` | Terminate a session (EOF → SIGTERM → SIGKILL). Returns exit status. |
| `shell_list` | List all active sessions with status, command, and age. Auto-approved. |

**Key distinction:** `shell_spawn` starts a new process; `shell_input` sends input to an already-running one. The model must never call `shell_spawn` to send input to an existing session.

**Architecture:**
- `session.py`: `ShellSession` wraps `pexpect.spawn` with `read_until_idle()` — the idle detection algorithm
- `ansi.py`: ANSI escape sequence stripping (CSI, OSC, CR, backspace overprint) for clean model-readable output
- `plugin.py`: `InteractiveShellPlugin` with session dict, reaper thread (cleans up expired/idle/dead sessions), and tool executors

**Session lifecycle:** Sessions have configurable max lifetime (default 600s) and max idle time (default 300s). A background reaper thread periodically closes expired sessions. Max concurrent sessions defaults to 8.

**Use cases:** Database REPLs (`psql`, `mysql`), SSH sessions, debuggers (`gdb`, `pdb`), package manager wizards (`npm init`), interactive installers, language REPLs (`python`, `node`), container shells (`docker exec -it`).

### UI Rendering Architecture (Separation of Concerns)

The UI rendering follows a strict separation between data production and presentation:

**Pipeline Layer** (`shared/plugins/`, `server/`):
- Produces **structured data** (e.g., Q&A pairs, tool results, plan steps)
- Emits **lifecycle events** with semantic content
- Is UI-agnostic - no formatting, colors, or layout decisions

**Client Presentation Layer** (`jaato-tui/output_buffer.py`):
- Receives structured data from pipeline
- Chooses **optimal UX presentation** based on terminal size, theme, context
- Handles formatting, truncation, tables, colors, layout
- May adapt presentation dynamically (e.g., compact vs expanded based on space)

**Example - Clarification Plugin:**
```
Pipeline (clarification/plugin.py):
  → Emits: on_resolved(tool_name, qa_pairs=[(question, answer), ...])

Client (output_buffer.py):
  → Receives qa_pairs, decides: table? stacked? inline?
  → Applies theme colors, calculates column widths, handles wrapping
```

This separation ensures:
- Pipeline code remains testable without UI dependencies
- Multiple clients can present the same data differently
- Presentation can evolve without changing pipeline logic

### Presentation Context (Agent Display Awareness)

The model receives display constraints via `PresentationContext` (defined in
`jaato-sdk/jaato_sdk/events.py`) so it adapts its output format.

**Data flow:**
```
Client → ClientConfigRequest.presentation (dict)
  → SessionManager._apply_client_config()
  → JaatoServer.set_presentation_context()
  → JaatoClient → JaatoSession._presentation_context
  → get_system_instructions(presentation_context=...) → system prompt
```

**Key fields:** `content_width`, `supports_tables`, `supports_code_blocks`,
`supports_images`, `supports_expandable_content`, `client_type`.

`client_type` is a `ClientType` enum (`terminal`, `web`, `chat`, `api`) —
values describe the presentation surface category, not specific apps.

When `supports_expandable_content=True`, the model outputs freely and the
**client** wraps overflow in its native expand/collapse widget (Telegram inline
buttons, HTML `<details>`, TUI scrollable panel). When `False`, the model is
asked to use compact formats for narrow displays.

See [Agent Presentation Awareness](docs/design/agent-presentation-awareness.md).

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

**Authentication Options (in priority order):**
1. **Device Code OAuth** (recommended): `github-auth login` - browser-based authorization
2. **Personal Access Token** (`ghp_...` or `github_pat_...`): Set `GITHUB_TOKEN` env var

The device code flow uses GitHub Copilot's OAuth client ID and doesn't require creating a PAT manually.

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

### Ollama (Local Models)
| Variable | Purpose |
|----------|---------|
| `OLLAMA_HOST` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Default model name |
| `OLLAMA_CONTEXT_LENGTH` | Override context length for models |

Requirements: Ollama v0.14.0+ (for Anthropic API compatibility)

Setup:
```bash
# Install Ollama: https://ollama.com/download
ollama serve                    # Start server
ollama pull qwen3:32b          # Pull a model
```

Configuration options via `ProviderConfig.extra`:
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | str | `http://localhost:11434` | Ollama server URL |
| `context_length` | int | 32768 | Context window size |

Benefits:
- Run models locally without API costs
- Privacy - data never leaves your machine
- Use any model Ollama supports (Qwen, Llama, Mistral, etc.)

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
| `JAATO_SESSION_LOG_DIR` | Per-session log directory, relative to workspace (default: `.jaato/logs`) |

### Rate Limiting
| Variable | Purpose |
|----------|---------|
| `AI_REQUEST_INTERVAL` | Minimum seconds between requests (default: 0) |
| `AI_RETRY_ATTEMPTS` | Max retry attempts (default: 5) |
| `AI_RETRY_BASE_DELAY` | Initial retry delay seconds (default: 1.0) |
| `AI_RETRY_MAX_DELAY` | Maximum retry delay seconds (default: 30.0) |

### Proxy Configuration
| Variable | Purpose |
|----------|---------|
| `HTTPS_PROXY` / `HTTP_PROXY` | Standard proxy URL (e.g., `http://proxy:8080`) |
| `NO_PROXY` | Standard no-proxy hosts (suffix matching) |
| `JAATO_NO_PROXY` | Exact host matching for no-proxy (e.g., `github.com,api.github.com`) |
| `JAATO_KERBEROS_PROXY` | Enable Kerberos/SPNEGO proxy auth (`true`/`false`) |
| `JAATO_SSL_VERIFY` | SSL certificate verification (`true`/`false`, default: `true`). Set to `false` to disable — escape hatch for SSL-intercepting proxies. |

**Kerberos Proxy Authentication:**
For corporate proxies requiring SPNEGO/Negotiate authentication:
```bash
export HTTPS_PROXY=http://proxy.corp.com:8080
export JAATO_KERBEROS_PROXY=true
# Ensure you have valid Kerberos tickets (kinit on Linux/Mac, Windows domain login)
```

Requires `pyspnego` package (`pip install pyspnego`) on Linux/macOS. On Windows, a native SSPI fallback via `secur32.dll` is used automatically when `pyspnego` is not installed.

## Rich Client Commands

### Authentication Commands
```
anthropic-auth login/logout/status     # Anthropic OAuth (PKCE flow)
antigravity-auth login/logout/status   # Google OAuth (PKCE flow)
github-auth login/poll/logout/status   # GitHub OAuth (device code flow)
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

Default keybindings: `submit`=enter, `cancel`=c-c, `exit`=c-d, `toggle_plan`=c-p, `toggle_tools`=c-t, `open_editor`=c-g, `search`=c-f

The `open_editor` keybinding (Ctrl+G) opens the current input in your external editor (`$EDITOR` or `$VISUAL`, defaults to `vi`). Useful for composing complex multi-line prompts.

The `workspace_open_file` keybinding (Enter by default, when workspace panel is open) opens the file at the cursor in your external editor (`$EDITOR` or `$VISUAL`, defaults to `vi`). The workspace panel must be visible and the input buffer empty for this keybinding to activate.

The `search` keybinding (Ctrl+F) opens search mode to find text in session output. When in search mode: Enter=next match, Ctrl+P=previous match, Escape=close search.

Large pastes (>10 lines or >1000 chars) are automatically replaced with placeholders like `[paste #1: +50 lines]` to prevent UI freezing. The original content is stored and expanded when you submit the prompt.

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

## Coding Policies

### Docstring Maintenance

Whenever you read or modify code, check that the docstrings on the classes, methods, and functions you touch are **present, accurate, and complete**. If they are missing, outdated, or misleading, update them as part of the same change. Specifically:

- **Lifecycle and state transitions** must be documented on the class that holds the state (e.g., which methods transition between states, what each state means, where the object lives at each stage).
- **Non-obvious parameters** like `finalized`, `backgrounded`, or boolean flags that change rendering/behavior must explain *when* and *why* they are set.
- **Relationships between classes** (e.g., `ActiveToolCall` living in `_active_tools` vs being deep-copied into a `ToolBlock`) must be documented on both sides.
- **Rendering methods** must document what visual output they produce and under which conditions they are called.

This is not optional cleanup — treat missing or inaccurate docstrings as a defect to fix alongside the feature work.

## Additional Documentation

- [Architecture Overview](docs/architecture.md) - Server-first architecture, event protocol, component diagrams
- [Sequence Diagrams](docs/sequence-diagram-architecture.md) - Client-server interaction, tool execution flows
- [Design Philosophy](docs/design-philosophy.md) - Opinionated design decisions and rationale
- [Path Boundary Pattern](docs/path-boundary-pattern.md) - MSYS2/Windows path handling for new components
- [OpenTelemetry Design](docs/opentelemetry-design.md) - Comprehensive OTel tracing integration
- [Reliability Policies Config](docs/reliability-policies-config.md) - JSON schema, per-tool thresholds, prerequisite policies, usage examples
- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
