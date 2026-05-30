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
  - `--ws-token TOKEN` / `--ws-token-file PATH`: bearer token clients must present in the WS Upgrade. Token-file mode 0600 enforced. When neither flag is passed (and `--web-socket` is set), the daemon reads `~/.jaato/ws.token`; if the file doesn't exist, it generates a 32-byte token and persists it there with mode 0600. Local clients can read the same default path for zero-config auth.
  - `--ws-unsafe-no-auth`: explicit opt-out of WS bearer auth (legacy open-accept). Logs a startup WARNING. Required to keep the historical behaviour.
  - `--daemon`: Run as background process
  - `--status`/`--stop`: Server management

  **WS auth contract:** clients send `Authorization: Bearer <token>` on the Upgrade request (Python/curl/proxies) or pass `?token=<token>` as a query parameter (browsers, which can't set custom headers from `new WebSocket()`). The server stores only the SHA-256 digest and compares with `hmac.compare_digest`. Auth runs after connection-interceptors but before any session work, so a bad token is closed with WS code 1008 immediately. The `set_client_user()` hook for jaato-premium SSO is unchanged — premium can still attach an identity after the bearer check passes.

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

Four plugin types:

**Tool Plugins** - Provide tools the model can invoke (`PLUGIN_KIND = "tool"`, implements `ToolPlugin`):
- `PluginRegistry`: Discovers and manages tool plugins
- `cli/`: Shell commands | `mcp/`: MCP servers | `permission/`: Permission control
- `interactive_shell/`: Interactive PTY sessions (REPLs, password prompts, wizards, debuggers)
- `file_edit/`, `todo/`, `web_search/`, `filesystem_query/`, etc.

**Enrichment Plugins** - Enrich prompts/instructions/results without providing tools (`PLUGIN_KIND = "enrichment"`, implements `EnrichmentPlugin`):
- Lightweight alternative to `ToolPlugin` for plugins that only participate in the enrichment pipeline
- No `get_tool_schemas()`, `get_executors()`, or command methods needed
- Automatically registered as enrichment-only by the registry
- Discovered alongside tool plugins during `registry.discover()`

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
- `model_provider/lmstudio/`: LM Studio local models (OpenAI-compat chat + native load-control)
- `model_provider/nim/`: NVIDIA NIM (OpenAI-compatible API, hosted + self-hosted)
- `model_provider/tensorrt_llm/`: NVIDIA TensorRT-LLM via `trtllm-serve` (OpenAI-compatible, self-hosted GPU inference)
- `model_provider/openrouter/`: OpenRouter (unified gateway over 300+ models, OpenAI-compatible)

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

### Agent Profiles

Sessions can be created with a predefined agent profile that configures model, provider, plugins, system instructions, and GC strategy. Profiles are JSON files in `.jaato/profiles/`.

**Profile schema** (same as `SubagentProfile` in `shared/plugins/subagent/config.py`):
```json
{
  "name": "researcher",
  "description": "Deep research profile",
  "model": "claude-sonnet-4-20250514",
  "provider": "anthropic",
  "plugins": ["cli", "web_search", "memory", "todo(preload)"],
  "plugin_configs": {},
  "system_instructions": "You are a research analyst...",
  "gc": { "type": "budget", "threshold_percent": 80.0 }
}
```

**SDK API:**
```python
# List available profiles
await client.list_profiles()  # → SessionProfilesEvent

# Create session with a profile
await client.create_session(profile="researcher")
```

**IPC command protocol:**
- `session.new [name] --profile <name>` — create session from profile
- `session.profiles` — list available profiles (→ `SessionProfilesEvent`)

**Flow:** Client sends `session.new --profile researcher` → server discovers profiles from `.jaato/profiles/` → resolves `SubagentProfile` → `JaatoServer` applies profile overrides (model, provider, plugins, system_instructions, plugin_configs, GC) during `initialize()`.

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

### Server Version Check

The server includes its package version (`server_version`) in the `ConnectedEvent`'s `server_info` dict, read from `importlib.metadata` at runtime. The SDK exposes it as `IPCClient.server_version` (and `IPCRecoveryClient.server_version`) after connect.

Each client declares its own minimum — e.g., the TUI sets `MIN_SERVER_VERSION = "0.2.27"` and refuses to connect if the server is older. If a client doesn't declare a minimum, no check is performed. `IncompatibleServerError` is classified as permanent by the recovery client (no retries).

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

### Pre-warm Runner Pool

Sessions consume a pre-warm runner subprocess from a pool instead of cold-spawning one each time.  Cuts per-session bootstrap from ~30s (with full plugin discovery + imports) to ~7s on cascade workloads.

Architecture: daemon spawns a **template subprocess** at startup that imports all runner-tier plugin modules.  N pre-warm **pool slots** fork from the template (no exec), inheriting the warm imports.  When a session arrives, daemon claims a pool slot and dispatches `session.bootstrap` to it via the same `RunnerRPCClient` it would use for a cold-spawned runner.  Slot self-confines to the session's AppArmor profile in bootstrap step 1c via `aa_change_profile` (main-thread dispatch so subsequently-spawned worker threads inherit the confined cred).

**Operational properties:**
- **Subreaper**: daemon calls `prctl(PR_SET_CHILD_SUBREAPER, 1)` at startup so orphaned descendants (slots whose template died) re-parent to the daemon.
- **Watchdog**: `PoolManager` replenishment thread detects template death + auto-respawns + refills pool.
- **READY handshake**: template sends `"READY\n"` after plugin discovery completes; daemon's `TemplateManager.spawn` blocks for it (30s timeout) instead of a fixed sleep.
- **Telemetry**: `PoolManager.get_telemetry()` exposes counters (`pool_slot_acquired_total`, `pool_acquire_miss_total`, `pool_replenish_success_total`, `pool_replenish_failures_total`, `template_respawn_attempts_total`, `template_respawn_failures_total`).

**Configuration:**
- Enabled by default (`JAATO_RUNNER_POOL_ENABLED=true`).  Disable with `=false` / `0` / `no` / `off`.
- Pool size via `JAATO_RUNNER_POOL_SIZE` (default 2).

**Pool routing gates** (`spawn_session_runner`): pool is consulted iff `pool_manager` wired AND env flag enabled AND `cgroup_attach is None` (cgroup migration mid-life is a follow-up).  Apparmor opt-in sessions ARE eligible (slot self-confines to the per-session profile).

See `docs/design/runner_prewarm_pool_plan.md` for the full multi-PR plan + decision log.

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

### Webhook Plugin (`shared/plugins/webhook/`)

The webhook plugin provides an inbound HTTP listener for receiving external webhooks (GitHub, Slack, Jira, etc.) and delivering them to agent sessions via subscribe/poll tools. Enables long-running daemon sessions that react to external events.

**Tools** (all `discoverability="discoverable"`):

| Tool | Purpose |
|------|---------|
| `webhook_subscribe` | Subscribe to webhook events, starts HTTP listener lazily. Returns subscription ID + endpoints. |
| `webhook_poll` | Long-poll for events on a subscription. Blocks up to timeout seconds. |
| `webhook_status` | Check listener status, routes, and event statistics. Auto-approved. |

**Configuration** (`.jaato/webhook.json`):
```json
{
  "port": 9100,
  "host": "127.0.0.1",
  "secret": "${WEBHOOK_SECRET}",
  "tls": { "enabled": true, "certfile": "...", "keyfile": "...", "ca_certfile": "..." },
  "allowed_ips": ["10.0.0.0/8"],
  "rate_limit_per_second": 50,
  "routes": {
    "github": {
      "path": "/webhook/github",
      "secret_header": "X-Hub-Signature-256",
      "secret_algo": "hmac-sha256",
      "event_type_header": "X-GitHub-Event"
    }
  }
}
```

**Corporate hardening** (all stdlib, no external deps):
- **TLS/SSL**: HTTPS with optional mutual TLS (client certificate verification)
- **IP allowlisting**: CIDR-aware, IPv4/IPv6, IPv4-mapped-IPv6 normalization
- **Rate limiting**: Per-IP token-bucket algorithm

**Architecture:** HTTP server runs in a daemon thread using `http.server.HTTPServer`. Per-subscription event buffers (`deque(maxlen=1000)`) with `threading.Event`-based long-poll wakeup. Server starts lazily on first subscribe call.

See [Webhook Plugin Design](docs/design/webhook-plugin.md) for full design doc.

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

**Profile knobs** (under `plugin_configs.anthropic`) — namespaced into
three layers since server 0.6.24 (no `routing` layer because Anthropic's
API has no gateway routing extension):

```yaml
plugin_configs:
  anthropic:
    # Top-level — auth / identity (rarely set per-profile; usually env vars)
    api_key: "sk-ant-..."          # overrides env / OAuth
    oauth_token: "sk-ant-oat01-..." # OAuth token for subscription

    # api_params — Anthropic Messages API request body fields
    api_params:
      temperature: 0.0             # 0.0-1.0 (server default 1.0)
      top_p: 0.95
      top_k: 40
      max_tokens: 4096             # overrides framework default
      enable_thinking: true        # extended reasoning
      thinking_budget: 10000       # max thinking tokens

    # framework_overrides — rare escape hatches (none defined today;
    # reserved for future use like context_length overrides)
```

| Layer | Keys | Purpose |
|-------|------|---------|
| top-level | `api_key`, `oauth_token` | auth / identity |
| `api_params` | `temperature`, `top_p`, `top_k`, `max_tokens`, `enable_thinking`, `thinking_budget` | Anthropic Messages API body fields. Sampling params are omitted from the request when unset, letting Anthropic apply its server-side defaults. Setting `temperature: 0.0` is the framework's determinism knob. |
| `framework_overrides` | (reserved) | Future escape hatches |

(Prompt caching is managed by the `cache_anthropic` plugin, not via
`api_params`.)

**Backward compatibility:** the same keys are also accepted at the
legacy flat position (`temperature:` directly under `anthropic:`) with
a one-time deprecation warning per key.  Flat-key support will be
removed in a future server release.

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

### LM Studio (Local Models)
| Variable | Purpose |
|----------|---------|
| `LMSTUDIO_HOST` | LM Studio server URL (default: `http://localhost:1234`) |
| `LMSTUDIO_MODEL` | Default model name |
| `LMSTUDIO_CONTEXT_LENGTH` | Override context window size |
| `LMSTUDIO_API_TOKEN` | Optional bearer token (only when LM Studio requires it) |

Chat uses LM Studio's OpenAI-compatible `/v1/chat/completions`.  Model
catalog comes from the native `/api/v0/models` endpoint (which reports
each model's real `max_context_length`).

**Load-control**: when the session profile supplies a `load` dict under
`plugin_configs["lmstudio"]`, the provider POSTs it to
`POST /api/v1/models/load` before the first chat, reconfiguring the
in-memory model with context length, GPU offload, KV-cache placement,
etc.  Without a `load` dict the provider is passive — it uses whatever
model the user has already loaded in LM Studio's UI or via `lms load`.

Configuration options via `ProviderConfig.extra` (typically set from the
session profile — see "Profile schema" below for the plumbing):
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | str | `http://localhost:1234` | LM Studio server URL |
| `context_length` | int | discovered from catalog | Context window override |
| `api_token` | str | None | Bearer token for auth-required servers |
| `load` | dict | None | Passthrough body for `/api/v1/models/load` |

`load` keys (passed through to LM Studio unchanged):
- `context_length`, `eval_batch_size`, `flash_attention`,
  `num_experts`, `offload_kv_cache_to_gpu`, `echo_load_config`
- Any future LM Studio load param — the provider does not validate keys.

**Profile example:**
```json
{
  "name": "local-gpt-oss",
  "model": "openai/gpt-oss-20b",
  "provider": "lmstudio",
  "plugin_configs": {
    "lmstudio": {
      "host": "http://localhost:1234",
      "load": {
        "context_length": 16384,
        "flash_attention": true,
        "offload_kv_cache_to_gpu": true,
        "eval_batch_size": 512
      }
    }
  }
}
```

### NVIDIA NIM
| Variable | Purpose |
|----------|---------|
| `JAATO_NIM_API_KEY` | API key for hosted NIM (`nvapi-...` from build.nvidia.com) |
| `JAATO_NIM_BASE_URL` | Endpoint (default: `https://integrate.api.nvidia.com/v1`) |
| `JAATO_NIM_MODEL` | Default model name |
| `JAATO_NIM_CONTEXT_LENGTH` | Override context window size |

**Authentication Options (in priority order):**
1. **Environment variable**: Set `JAATO_NIM_API_KEY`
2. **Stored credentials**: `nim-auth key <api_key>` — validates and stores securely
3. **Self-hosted**: Set `JAATO_NIM_BASE_URL` to a local endpoint (no key needed)

Available models include Llama 3.3/3.1, DeepSeek-R1, Nemotron, and other NIM catalog models.

### NVIDIA TensorRT-LLM (`trtllm-serve`)
| Variable | Purpose |
|----------|---------|
| `TENSORRT_LLM_HOST` | trtllm-serve URL (default: `http://localhost:8000`) |
| `TENSORRT_LLM_MODEL` | Default model name (matches the engine's `id` in `/v1/models`) |
| `TENSORRT_LLM_CONTEXT_LENGTH` | Override context window size (trtllm-serve does not surface `max_seq_len` in `/v1/models`) |
| `TENSORRT_LLM_API_TOKEN` | Optional bearer token (only when fronted by an auth proxy — trtllm-serve has no built-in API key mechanism) |

Talks to a `trtllm-serve` instance the user has already launched. Each `trtllm-serve` process hosts exactly one engine, built out-of-band with `trtllm-build`. Provider is **passive** — no in-band load endpoint analogous to LM Studio's `/api/v1/models/load`.

Profile knobs under `plugin_configs.tensorrt_llm`:

| Key | Type | Description |
|-----|------|-------------|
| `host` | str | Override `TENSORRT_LLM_HOST` |
| `context_length` | int | Context window override (required for long-context engines) |
| `api_token` | str | Bearer token override |

Quick start:
```bash
trtllm-serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000
export TENSORRT_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export TENSORRT_LLM_CONTEXT_LENGTH=131072  # match the engine's max_seq_len
```

Benefits:
- Maximum throughput on NVIDIA GPUs (FP8/INT4 quant, in-flight batching, KV-cache reuse, speculative decoding)
- Self-hosted — no API costs, data never leaves your hardware
- DIY counterpart to NIM: NIM is essentially TensorRT-LLM productized; this provider serves users who build their own engines

### OpenRouter
| Variable | Purpose |
|----------|---------|
| `JAATO_OPENROUTER_API_KEY` | API key (`sk-or-...` from https://openrouter.ai/settings/keys) |
| `JAATO_OPENROUTER_BASE_URL` | Endpoint (default: `https://openrouter.ai/api/v1`) |
| `JAATO_OPENROUTER_MODEL` | Default model name |
| `JAATO_OPENROUTER_CONTEXT_LENGTH` | Override context window size |
| `JAATO_OPENROUTER_HTTP_REFERER` | App-attribution: `HTTP-Referer` header (required for OpenRouter app rankings) |
| `JAATO_OPENROUTER_APP_TITLE` | App-attribution: `X-OpenRouter-Title` header (display name) |
| `JAATO_OPENROUTER_APP_CATEGORIES` | App-attribution: `X-OpenRouter-Categories` header (comma-separated; default `cli-agent`) |

**Authentication Options (in priority order):**
1. **Environment variable**: Set `JAATO_OPENROUTER_API_KEY`
2. **Stored credentials**: `openrouter-auth key <api_key>` — validates against `GET /api/v1/key` and stores securely

OpenRouter is a unified gateway exposing 300+ models from many vendors (OpenAI,
Anthropic, Google, Meta, Mistral, DeepSeek, ...) behind a single OpenAI-compatible
API. Models use the `vendor/model` form, e.g. `anthropic/claude-3.5-sonnet`,
`openai/gpt-4o`, `meta-llama/llama-3.3-70b-instruct`. Use `openrouter/auto` to let
OpenRouter pick the best model per request.

`list_models()` queries `GET /api/v1/models` (no auth required) and `connect()`
caches per-model `context_length` from that catalog. The OpenAI SDK's
`default_headers` carry the optional `HTTP-Referer` and `X-OpenRouter-Title`
attribution headers automatically.

**Profile knobs** (under `plugin_configs.openrouter`) — namespaced into
four layers since server 0.6.23:

```yaml
plugin_configs:
  openrouter:
    # Top-level — auth / identity
    api_key: "sk-or-..."           # overrides env / stored credentials
    http_referer: "https://..."    # HTTP-Referer header for app rankings
    app_title: "MyApp"             # X-OpenRouter-Title header
    app_categories: ["cli-agent"]  # X-OpenRouter-Categories header
                                   # (marketplace categories for jaato's
                                   # placement in OpenRouter rankings;
                                   # default ["cli-agent"], pass [] to
                                   # opt out of category attribution).
                                   # See https://openrouter.ai/docs/app-attribution
    extra_headers:                 # arbitrary additional headers (str→str)
      x-anthropic-beta: "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14"
                                   # OpenRouter forwards supported beta
                                   # headers to upstreams; see
                                   # https://openrouter.ai/docs/features/provider-routing#provider-specific-headers

    # api_params — OpenAI Chat Completions request body fields
    api_params:
      temperature: 0.55            # sampling
      top_p: 1.0
      top_k: 40
      max_tokens: 8192             # cap on response size
      models:                      # cross-model fallback list (sibling of `model`)
        - anthropic/claude-sonnet-4.5
        - openai/gpt-5-mini
        - google/gemini-3-flash-preview
                                   # OpenRouter walks candidates on outage /
                                   # context-limit / safety failures of `model`.
                                   # Pairs with routing.sort.partition="none"
                                   # to find the best provider across all
                                   # candidate models.
      enable_thinking: true        # extended-reasoning request + extraction
      thinking_budget: 16384       # → reasoning.max_tokens
      thinking_level: "high"       # → reasoning.effort (low/medium/high)
      cache_prompt: "auto"         # "auto" (default) / true / false —
                                   # stamps cache_control breakpoints on the
                                   # system block and last tool definition
                                   # (Anthropic / Gemini upstreams).  See
                                   # https://openrouter.ai/docs/features/prompt-caching
      cache_ttl: "5m"              # "5m" (default) or "1h" (2x write
                                   # premium, no mid-session cache miss)

    # routing — OpenRouter `provider` extension; forwarded via extra_body.
    # Opaque pass-through: any field from
    # https://openrouter.ai/docs/features/provider-routing works.
    routing:
      sort: "price"                # "price" | "throughput" | "latency"
                                   # OR {by: "...", partition: "model"|"none"}
      data_collection: "deny"      # "allow" (default) | "deny"
      ignore: ["Groq"]             # provider slugs to skip
      only: ["azure"]              # allowlist (mutex with ignore)
      order: ["openai", "together"]  # try these first, then fall back
      require_parameters: true     # only upstreams supporting every param
      allow_fallbacks: true        # false → fail rather than try others
      quantizations: ["fp8"]       # int4/int8/fp4/fp6/fp8/fp16/bf16/fp32
      zdr: true                    # Zero Data Retention endpoints only
      enforce_distillable_text: true   # only distillable-text-allowed models
      max_price: {prompt: 1, completion: 2, request: 0.01, image: 0.001}
      preferred_min_throughput: {p90: 50}  # number for p50, or {p50,p75,p90,p99}
      preferred_max_latency: {p50: 1, p90: 3, p99: 5}

    # framework_overrides — rare escape hatches
    framework_overrides:
      context_length: 32768        # override catalog-reported window
      base_url: "https://..."      # endpoint override
```

| Layer | Keys | Purpose |
|-------|------|---------|
| top-level | `api_key`, `http_referer`, `app_title`, `app_categories`, `extra_headers` | auth / identity. `app_categories` (`List[str]`) is jaato's hook into [OpenRouter's app marketplace](https://openrouter.ai/docs/app-attribution) — emitted as the `X-OpenRouter-Categories` header. Default is `["cli-agent"]` (jaato is a terminal-driven agentic tool orchestrator); pass `[]` to opt out of category attribution entirely. Validated strictly: lowercase hyphen-separated slugs, ≤30 chars each, ≤5 entries; unrecognized categories are silently dropped server-side. `extra_headers` (`Dict[str,str]`) is the hook for OpenRouter's [provider-specific beta headers](https://openrouter.ai/docs/features/provider-routing#provider-specific-headers) — Anthropic `x-anthropic-beta` is the canonical case (`fine-grained-tool-streaming-2025-05-14`, `interleaved-thinking-2025-05-14`, `structured-outputs-2025-11-13`). Both merge into the OpenAI client's `default_headers`; profile values win on key collisions. |
| `api_params` | `temperature`, `top_p`, `top_k`, `max_tokens`, `models`, `enable_thinking`, `thinking_budget`, `thinking_level`, `cache_prompt`, `cache_ttl`, `strict_tools` | OpenAI Chat Completions body fields. `models` is OpenRouter's request-level cross-model fallback list (sibling of `model`; OpenRouter walks candidates on failure). `thinking_*` keys mirror Anthropic / Antigravity; when both `thinking_level` and `thinking_budget` are set, `level` wins (more portable across upstreams). `cache_prompt: "auto"` (default) places `cache_control: {type: ephemeral}` breakpoints on the system block and last tool definition for explicit-cache upstreams (Anthropic, Gemini 1.5+/2.5+/3+); other upstreams (OpenAI, DeepSeek, Grok) cache automatically and need no client annotation. Response-side parsing of `prompt_tokens_details.cached_tokens` / `cache_creation_input_tokens` / `cost` is unconditional. `strict_tools: true` (server 0.6.118+) emits `"strict": true` as a sibling of `parameters` in each tool definition; OpenRouter forwards to supported upstreams (Sonnet 4.5 / Opus 4.1+, GPT-4o+, Gemini, OSS, Fireworks per [structured outputs list](https://openrouter.ai/docs/guides/features/structured-outputs)) for grammar-constrained tool-arg sampling. Required for cascade-determinism use cases (see `feedback_cascade_completion_schemas_require_strict_model_support` memory); the framework does NOT auto-rewrite schemas to satisfy OpenAI's strict-mode requirements (kb authors own schema shape — `additionalProperties: false` on every object, exhaustive `required` arrays, no `oneOf`/`anyOf` if you enable strict). |
| `routing` | any [provider routing](https://openrouter.ai/docs/features/provider-routing) key (`order`, `allow_fallbacks`, `require_parameters`, `data_collection`, `ignore`, `only`, `quantizations`, `sort` (string or `{by, partition}`), `zdr`, `enforce_distillable_text`, `max_price`, `preferred_min_throughput`, `preferred_max_latency`, ...) | constrains which upstream host serves a request. Composes with `model: "openrouter/auto"` (auto picks model, routing constrains hosts) and `api_params.models` (cross-model fallback list, routing constrains providers across all of them). Opaque pass-through — new routing keys land automatically. |
| `framework_overrides` | `context_length`, `base_url` | rare escape hatches; normally context length is discovered from the OpenRouter catalog at connect time. |

**Backward compatibility:** the same keys are also accepted at the
legacy flat position (`temperature:` / `provider:` / `context_length:`
directly under `openrouter:`) with a one-time deprecation warning per
key.  Flat-key support will be removed in a future server release.

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
| `JAATO_RUNNER_POOL_ENABLED` | Enable pre-warm runner pool routing (default: `true`).  Sessions consume pre-warm pool slots instead of cold-spawning a runner subprocess.  Set to `false` / `0` / `no` / `off` to disable.  See `docs/design/runner_prewarm_pool_plan.md`. |
| `JAATO_RUNNER_POOL_SIZE` | Number of pre-warm pool slots to keep idle (default: 2).  Raise for cascade workloads that spawn many sessions in tight succession. |
| `JAATO_AMBIGUOUS_WIDTH` | Width for East Asian Ambiguous chars in tables (`1` default, `2` for CJK terminals) |
| `JAATO_SESSION_LOG_DIR` | Per-session log directory, relative to workspace (default: `.jaato/logs`) |
| `JAATO_CGROUPS_ROOT` | Parent cgroup v2 directory for the WS server's per-session cgroup tree (default: `/sys/fs/cgroup/jaato`). Override when the host has subtree_control delegated under a different path. Must already exist with `memory`, `pids`, `cpu` in `cgroup.subtree_control`. |

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
nim-auth login/key/logout/status       # NVIDIA NIM API key
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

The `workspace_open_file` keybinding (Enter by default, when workspace panel is open) opens the file at the cursor in your external editor (`$EDITOR` or `$VISUAL`, defaults to `vi`). The workspace panel must be visible and the input buffer empty for this keybinding to activate.  The companion `workspace_diff` keybinding (`d` by default) opens the same file in an external diff viewer instead — it resolves the `diff` action from `openers.json` and is a no-op if no pattern defines one.

**Per-extension openers**: the launched program can be customized per file pattern via `.jaato/openers.json` (project) or `~/.jaato/openers.json` (user). Maps fnmatch globs to either a single command string (the `raw` action — opens in editor) or an object of `{action: command}` entries (currently `raw` and `diff`). Project entries override user entries **per action**, so a user-level `diff` opener survives a project that only redefines `raw`. `$EDITOR` and `$VISUAL` are valid placeholders (both resolve to the default editor). Longest matching pattern wins; on a tie, basename match beats path match. **Per-action fallthrough**: if the most-specific matching pattern doesn't define the requested action, the resolver walks the next-most-specific match — so a catch-all `"*"` entry can supply defaults.

```json
{
  "*.md":       { "raw": "glow -p", "diff": "git diff HEAD --" },
  "*.markdown": "glow -p",
  "*.png":      { "raw": "chafa" },
  "docs/*":     "less",
  "*":          { "raw": "$EDITOR", "diff": "git diff HEAD --" }
}
```

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
- [Daemon Extensions](docs/design/daemon-extensions.md) - Extension points for external packages (session hooks, WS interceptors, custom aspects, remote handlers)
- [Payload-Schema Conventions](docs/design/payload-schema-conventions.md) - Symmetric authoring guide for `spawn_payload_schema` (input boundary) and `completion_payload_schema` (output boundary). Mirror prefetch required-keys; always carry `warnings[]` / `errors[]` escape hatches; persona ↔ schema consistency check; canonical-hash strip rules; `agent_params` interaction with agent-continuity (§6).
- [Agent Continuity Pattern](docs/design/agent-continuity.md) - `{{continuity_scope}}` + memory plugin enrichment + raw/curated lifecycle: persona-level continuity across sessions composed from existing primitives, no new framework code. Reference impl in `jaato-knowledge-manager/.jaato.example/`.
- [AppArmor Setup](docs/apparmor-setup.md) - Kernel-enforced workspace isolation. WS deployments confine automatically when AppArmor is available; IPC clients opt in via `IPCClient(..., apparmor=True)` (defaults to `False`).
- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
