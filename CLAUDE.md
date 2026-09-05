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
  - `--socket-mode MODE`: Octal file permissions for the IPC socket (default: `660`, owner and group only). The IPC transport is unauthenticated, so any principal that can open the socket can fully drive the agent. Pass `666` to opt into world-accessible (e.g. cross-user containers on a trusted host).
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
  - `create_session(model, plugins=..., system_instructions=..., ...)` - spawn lightweight sessions (`tools=` is a deprecated alias for `plugins=`; #292)

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
- `model_provider/chrome_ai/`: Chrome built-in AI — the Gemini Nano on-device model via the browser's Prompt API (`LanguageModel` global), driven over the Chrome DevTools Protocol; zero cost, no credentials, tiny context (~6-9k)
- `model_provider/lmstudio/`: LM Studio local models (OpenAI-compat chat + native load-control)
- `model_provider/nim/`: NVIDIA NIM (OpenAI-compatible API, hosted + self-hosted)
- `model_provider/tensorrt_llm/`: NVIDIA TensorRT-LLM via `trtllm-serve` (OpenAI-compatible, self-hosted GPU inference)
- `model_provider/vllm/`: vLLM via `vllm.entrypoints.openai.api_server` (OpenAI-compatible, self-hosted GPU inference)
- `model_provider/openrouter/`: OpenRouter (unified gateway over 300+ models, OpenAI-compatible)
- `model_provider/nebius/`: Nebius Token Factory (serverless open-model inference, OpenAI-compatible; `/v1/models` catalog auto-detects context window + input modalities)
- `model_provider/ovhcloud/`: OVHcloud AI Endpoints (serverless open-model inference on OVHcloud's EU cloud, OpenAI-compatible unified gateway; catalog auto-detects context window when reported, manual knobs otherwise; opt-in keyless free tier)
- `model_provider/doubleword/`: Doubleword (serverless open-model inference priced by delivery window, OpenAI-compatible; `api_params.service_tier: flex` opts into the discounted async tier — queued work, ~1 min to first token — on the same chat endpoint; `context_length` must be set — the catalog reports no per-model window)

**Model Quirks** — per-model workarounds a profile opts into via `quirks:`
(injected into `config.extra["quirks"]`; each provider declares the names it
honors in its `PROVIDER_QUIRKS` contract):

| Quirk | Honored by | Effect |
|-------|-----------|--------|
| `prose_tool_calls` | all OpenAI-compat providers (nim, nebius, ovhcloud, lmstudio, tensorrt_llm, triton, vllm, zhipuai_openai) + openrouter | Prose-emulated tool calling for upstream models that cannot emit native tool calls: the `tools` array is withheld, schemas are prompt-injected (hashed wire ids, model picks by description), tool traffic in history is replayed as text, and fenced ` ```tool_call ` JSON blocks in the response are parsed back into `FunctionCall` parts. Reliability tier below native tool calling (hallucinated ids surface as recoverable unknown-tool errors; malformed blocks stay visible in text). Shared machinery in `model_provider/_prose_tools.py` — the same protocol `chrome_ai` uses unconditionally. |
| `coerce_typed_tool_args`, `force_tool_choice_for_lifecycle`, `force_narration_between_tools`, `auto_finalize_on_complete` | vllm | Small-model tool-calling workarounds; see `vllm/provider.py` |

```yaml
# profile example: a cheap OpenRouter model that answers in prose
provider: openrouter
model: some-vendor/cheap-model
quirks:
  prose_tool_calls: true
```

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

### Application Identity (naming the app, not the framework)

Anything built on the SDK used to introduce itself upstream as **jaato** —
the framework's name and repo were hardcoded as OpenRouter's app-attribution
headers, so every integrator's harness collapsed into one row on the
dashboard. `shared/app_identity.py` separates the two: `AppIdentity` is the
*application*, and the framework rides along in a `(powered by jaato)` suffix.

```bash
export JAATO_APP_NAME="Acme Copilot"
export JAATO_APP_URL="https://acme.example"
export JAATO_APP_VERSION="1.4.0"
export JAATO_APP_CATEGORIES="chat-bot"   # optional; no marketplace listing without it
# → X-OpenRouter-Title:      Acme Copilot (powered by jaato)
# → HTTP-Referer:            https://acme.example
# → X-OpenRouter-Categories: chat-bot
```

```python
from shared.app_identity import AppIdentity
runtime = JaatoRuntime(app_identity=AppIdentity(name="Acme Copilot",
                                                url="https://acme.example",
                                                version="1.4.0"))
```

Precedence, highest first:

| # | Surface | Scope |
|---|---------|-------|
| 1 | `plugin_configs.openrouter.app_title` / `http_referer` / `app_categories` | one session |
| 2 | `JAATO_OPENROUTER_APP_TITLE` / `_HTTP_REFERER` / `_APP_CATEGORIES` | provider-specific env |
| 3 | `JaatoRuntime(app_identity=...)` | the embedding process |
| 4 | `JAATO_APP_*` | deployment (process env, workspace `.env`, a profile's `env:` map) |

With none of them set the identity is the framework's own and the provider
config is byte-identical to before — an unconfigured checkout still reports as
`jaato`. `JAATO_APP_POWERED_BY=false` drops the suffix; every field is sanitised
(CR/LF stripped, length-capped) because these strings become HTTP headers.
`AppIdentity.user_agent()` (`Acme-Copilot/1.4.0 (powered by jaato/0.7.0)`) is
the general form for providers that gain a `User-Agent` later. Categories are
the one value that does **not** fall back to the framework's: an app filed
under jaato's `cli-agent` by default would be mis-filed, so a named app sends
none until it declares its own. Full rationale
— including why the env vars are `host`-scoped and why there is no typed
profile block — in [Application Identity](docs/design/app-identity.md).

### Agent Profiles

Sessions can be created with a predefined agent profile that configures model, provider, plugins, and GC strategy. Profiles are YAML files in `.jaato/profiles/` (preferred; JSON is also accepted).

**Profile schema** (same as `SubagentProfile` in `shared/plugins/subagent/config.py`):
```yaml
name: researcher
description: Deep research profile
model: claude-sonnet-4-20250514
provider: anthropic
plugins:
  - cli
  - web_search
  - memory
  - todo(preload)
plugin_configs: {}
# Agent identity and instructions belong in .jaato/agents/<name>.md (persona)
# layered on top of .jaato/instructions/ base instructions.
# system_instructions: DEPRECATED — use agents instead.
# suppress_base_instructions: drop framework-injected instruction layers
#   (persona + plugin instructions are ALWAYS kept). Accepts a bool or a
#   granular map over three pieces:
#     - disk      — the .jaato/instructions/*.md base layer
#     - constants — framework prompt constants (task-completion/verification,
#                   parallel/batching, turn-summary; incl. jaato-premium overrides)
#     - security  — the untrusted-content boundary (indirect-prompt-injection defense)
#   `true` ≡ {disk: true, constants: true} — the security boundary is KEPT
#   (drop it only by naming it explicitly). Absent key = keep. Examples:
#     suppress_base_instructions: true                    # drop disk + constants
#     suppress_base_instructions: {constants: true}       # keep disk + security
#     suppress_base_instructions: {disk: true, constants: true, security: true}
#   Inheritance merges by UNION (a piece any layer drops stays dropped).
gc:
  type: budget
  threshold_percent: 80.0
# trace: typed diagnostic log paths — the validated sibling of the
#   JAATO_TRACE_LOG / JAATO_PROVIDER_TRACE env vars, which remain the
#   lower-precedence default (the block outranks both the workspace .env
#   and this profile's own `env:` map).  Absolute = one file shared by
#   every session using the profile; relative = one file per session,
#   resolved against the workspace by jaato_sdk.trace.  Refuses a switch
#   written into a path field — `env: {JAATO_PROVIDER_TRACE: '1'}` is a
#   valid str and wrote every session's trace to a file named `1` (#775).
trace:
  provider_log: .jaato/logs/provider_trace.jsonl
  session_log: .jaato/logs/session_trace.jsonl
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

**Flow:** Client sends `session.new --profile researcher` → server discovers profiles from `.jaato/profiles/` → resolves `SubagentProfile` → `JaatoServer` applies profile overrides (model, provider, plugins, plugin_configs, GC) during `initialize()`.

### Session Revive (waking a persisted session)

A session woken from disk — `session.wake`, a reattach, anything reaching
`SessionManager._load_session` — comes back with **what it persisted**, not
with what the files on disk say today (issue #787):

| What | Persisted as | Restored via |
|------|--------------|--------------|
| the resolved profile | `SessionState.profile_snapshot` (`profile_to_snapshot`) | `profile_from_snapshot` → `BootstrapEnvelope.profile` |
| the rendered system instruction | `SessionState.rendered_instructions` (snapshotted at the end of `JaatoSession.configure()`) | `BootstrapEnvelope.system_instruction_override` |
| the creation `agent_params` | `SessionState.agent_params` | `BootstrapEnvelope.agent_params` |

Record version 2.8+. Restoring the render means a revive does **not** re-run
the persona's `{{!py:...}}` prefetch scripts — which is what made a session
whose prefetch reads `context.agent_params` impossible to wake at all (the
params were not persisted, so the script was handed an empty dict and
aborted session-prep, blaming the task definition). It also makes a prefetch
run **once**, as `explain prefetch` documents, and stops a revived session's
prompt from silently diverging from the one its own history was produced
under.

Two env knobs (`JAATO_REVIVE_PROFILE`, `JAATO_REVIVE_PERSONA` — see the
General env table) opt back into re-deriving either half; both default to
`persisted`, and both fall back to re-deriving automatically when nothing
was persisted, so records written before 2.8 revive exactly as before. The
rationale for these being env vars rather than profile keys, and the matrix
of which combination each workflow needs, live in `server/revive_policy.py`.

Both are **per-process, not per-invocation**: they are resolved once when the
`SessionManager` is constructed and held for the life of the daemon, so
changing the posture means restarting the daemon and the new posture then
applies to *every* session that revives until the next restart. Freezing is
also what makes their `host` scope true — read live they would be settable
process-wide from any single workspace's `.env`, because
`JaatoServer._with_session_env` overlays every key of it onto the daemon's
`os.environ` for the duration of a turn.

**Contract for persona authors: never pass a credential as an
`agent_param`.** They are substituted into the persona by `resolve_agent`,
so they already reach the model in its system prompt — and the rendered
persona is now a persisted artifact. Secrets belong in the profile's `env:`
as a `pass://` / `vault://` URI, which stays unresolved on disk and is
resolved daemon-side at spawn.

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

### Binary Media Chunks (delivery)

Binary content (audio, images, PDFs) moves in three directions, and they are
NOT the same path: **inbound** (content the model looks at), **outbound** (the
model emits speech), and **tool -> client** (a tool produces bytes a *person*
consumes; the model may never see them). See
[Binary Media Chunks](docs/design/binary-media-chunks.md).

**One chunk primitive.** `StreamChunk` (`shared/plugins/streaming/protocol.py`)
carries text, bytes, or both. `inline_data` mirrors `Part.inline_data`
(`{mime_type, data}`) so one shape serves inbound parts, tool attachments and
chunks alike. New fields are appended, so positional construction and every
existing producer are untouched.

**Audience is data, not policy.** `Audience` (`MODEL` / `CLIENT` / `BOTH`,
default `MODEL`) selects whether a chunk enters *this session's history* — not
whether the event is published. Every chunk still reaches all three
subscription surfaces; a parent agent watching a child sees a `CLIENT` chunk
and is subject to its own modality gate. Tool streaming was previously
hardcoded to "for the model, hidden from the user"; media inverts that.

| Audience | Enters history | Delivered to clients |
|----------|----------------|----------------------|
| `MODEL` (default) | yes (`<hidden>`, as before) | no |
| `CLIENT` | **never** | yes |
| `BOTH` | yes | yes |

**The gate routes, it no longer shreds.** `_gate_one_tool_result` used to
*destroy* attachments the active model can't consume. Content the model cannot
consume is exactly what a viewer may want, so those attachments are now emitted
as `CLIENT` media (correlated by the result's `call_id`) before being stripped
from the model's copy. The model-facing withheld-note is unchanged.

**Delivery.** `ToolOutputEvent` gains `stream_id`, `sequence`, `mime_type`,
`data_b64`, `final` — widened rather than joined by a rival event, so the SDK
client, the `subscribeToEvents` agent tool and the `EventBus` all light up with
no new API. A whole-blob delivery is a single-chunk stream (`sequence=0,
final=True`).

> **Binary bypasses the formatter.** `server/core.py` `on_tool_output` runs text
> through `agent_pipeline.process_chunk()` for highlighting and marker
> transformation. That pipeline reflows its input and would corrupt bytes, so a
> media chunk skips it entirely.

**Backpressure.** The per-client IPC queue was unbounded, making the `QueueFull`
branch dead code — a slow consumer grew it without limit (cosmetic for text,
unbounded memory plus permanent audio drift for media). It is now bounded
(`JAATO_IPC_EVENT_QUEUE_MAX`, default 2048) with a per-class policy: tool-output
chunks are **lossy** (oldest evicted, media before text, because recency beats
completeness for a stream), everything else is **essential** and is queued past
the bound rather than desynchronising the client. `dropped_chunk_count()`
reports what was lost.

**Client renderability.** `PresentationContext.renderable_media` declares the
MIME types a viewer can play (`can_render_media()` honours `type/*` wildcards
and ignores parameters). This is the CLIENT axis and is kept strictly apart
from the MODEL axis (`model_tiers.<tier>.modalities`) — different owners,
different lifetimes.

**Model-emitted media (outbound).** `StreamingCallback` is
`Callable[[Union[str, MediaDelta]], None]`: `str` is a text token as always,
`MediaDelta` is model-generated bytes. One callback rather than two, so
text/audio ordering is preserved by construction. **Consumers must branch on
the type** — a `MediaDelta` is not text. Model media is `CLIENT`-audience by
construction (replaying the model's own audio into its history would be
meaningless) and is delivered on the tool-output channel under the reserved
`call_id` `"model-output"`.

Providers declare emission with `output_modalities()` /
`supports_output_modality(kind, model=)` — deliberately *not* named
`modalities()`, which is framework-wide for **input**. The tier startup check
probes `supports_output_modality` by name, so defining it turns outbound tier
roles from unverified declarations into enforced ones. Text-only is the floor;
`ProviderCapabilities.output_media` marks an adapter that actually delivers it.

> **Naming collision.** OpenAI's request field `modalities` means **OUTPUT**
> (`["text","audio"]` with `audio: {voice, format}`); a jaato tier's
> `modalities` key means **INPUT**. Both appear in one profile —
> `api_params.modalities` vs `model_tiers.<tier>.modalities` — and the layer
> disambiguates them. Both `modalities` and `audio` are now in
> `_FORWARDED_API_PARAMS` (`_openai_compat/base.py`); they were previously
> dropped, which made audio output unrequestable through any OpenAI-compatible
> provider. While streaming, OpenAI emits **only pcm16** (24 kHz mono s16le,
> headerless), which is why `STREAM_AUDIO_MIME` spells the parameters out.

### Tool Traits

Tools can declare semantic **traits** on their `ToolSchema` via the `traits` field (a `FrozenSet[str]`). Traits drive cross-cutting behavior without hardcoding tool names in session or plugin code.

**Currently defined traits:**

| Constant | Value | Contract |
|----------|-------|----------|
| `TRAIT_FILE_WRITER` | `"file_writer"` | Tool writes/modifies files. Result must include `path` (str), `files_modified` (list), or `changes[].file`. Triggers full-JSON enrichment (LSP diagnostics, artifact tracking). |
| `TRAIT_GREPPABLE_CONTENT` | `"greppable_content"` | Tool returns bulk content eligible for result-rewriting. Routes the tool's **full JSON result** through the same full-dict enrichment path as `TRAIT_FILE_WRITER`, so result-rewriter plugins (`result_grep`) can inspect/shrink structured payloads the text-field path never sees (e.g. `call_service.body`/`headers`). Marks eligibility only — filtering is performed by whichever rewriter is subscribed/active. |

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

### Entry-point Plugin Trust

Out-of-tree plugins are installed as distributions declaring
`[project.entry-points."jaato.plugins"]` (also `jaato.enrichment_plugins`,
`jaato.gc_plugins`, `jaato.cache_plugins`).  `PluginRegistry.discover()`
runs entry points **first**, then the directory scan — and the directory
scan skips any name already registered.  Left unguarded, that made every
built-in overridable by any distribution sharing the venv, silently
(#684).

The policy lives in `shared/plugins/entry_point_trust.py` and is applied
by `PluginRegistry._gate_entry_point`:

| Rule | Effect |
|------|--------|
| **Built-in names are reserved** | The reserved set is the module listing of `shared/plugins/` (read with `pkgutil.iter_modules` — a directory listing, no imports). A foreign entry point claiming one is refused. |
| **Refusal precedes `ep.load()`** | Every decision is made from the entry point's metadata (`ep.name` / `ep.value` / `ep.dist`), so a refused claim never has its module imported. `ep.load()` executes code — being installed must not be enough to run it. |
| **The framework's own declaration is exempt** | jaato-server publishes its built-ins through the same groups; an entry point targeting `shared.plugins.*` is the framework, not a claim. |
| **A security-critical subset is never shadowable** | `permission`, `cli`, `file_edit`, `mcp`, `sandbox_manager`, `interactive_shell` — refused even with the opt-in below. |
| **Operator opt-in** | `JAATO_PLUGIN_ALLOW_SHADOW=<name>[,<name>]` lets a distribution replace a non-critical built-in. The substitution is announced at WARNING, never silent. |
| **Optional distribution allowlist** | `JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST=<dist>[,<dist>]` narrows which distributions may contribute plugins at all, so a transitive dependency nobody chose stops participating. Names compare under PEP 503 normalisation. |
| **Collisions are named, not skipped** | First writer still wins, but the loser is logged at WARNING with both providers named — including the directory scan skipping a built-in because something else holds its name. Re-discovery by the same module stays quiet. |

**Provenance.**  The registry records a `PluginOrigin` for every plugin it
registers (`get_plugin_source(name)` / `get_plugin_sources()`), so a
shadow is visible without reading logs.  `jaato-scaffold plugins` marks
any plugin not supplied by the built-in package with
`<- <distribution> (<module>)`.

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

    # Top-level — prompt-cache control (delivered by the cache_anthropic
    # plugin, NOT Messages-API body fields, so not under api_params)
    enable_caching: true           # unset resolves JAATO_ANTHROPIC_ENABLE_CACHING
    cache_ttl: "1h"                # 5m (default) | 1h (2x write premium)
    cache_history: true            # place BP3 on history, not just system+tools
    cache_exclude_recent_turns: 2  # BP3 fallback when no InstructionBudget
    cache_min_tokens: true         # enforce the minimum cacheable size

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
| top-level (cache) | `enable_caching`, `cache_ttl`, `cache_history`, `cache_exclude_recent_turns`, `cache_min_tokens` | Prompt-cache control, consumed by the `cache_anthropic` plugin (explicit `cache_control` breakpoints on system / tools / history). These are not Messages-API body fields, so they sit at top level rather than under `api_params`. `enable_caching` unset falls back to `JAATO_ANTHROPIC_ENABLE_CACHING` (default off). **Server 0.7.1+**: before the `_wire_cache_plugin` fix these keys were silently ignored — the plugin was built from an always-empty config. See [Model Tiers × Prompt Caching](docs/design/model-tier-prompt-cache.md) §4. |
| `api_params` | `temperature`, `top_p`, `top_k`, `max_tokens`, `enable_thinking`, `thinking_budget` | Anthropic Messages API body fields. Sampling params are omitted from the request when unset, letting Anthropic apply its server-side defaults. Setting `temperature: 0.0` is the framework's determinism knob. |
| `framework_overrides` | (reserved) | Future escape hatches |

(Prompt caching is delivered by the `cache_anthropic` plugin rather than
by the provider, which is why its knobs sit at top level and not under
`api_params`.  Google's equivalents — `enable_caching` (default off) and
`cache_ttl` (Google duration format, e.g. `"3600s"`) — sit at top level
under `plugin_configs.google_genai`.  OpenRouter caches internally
instead, via `api_params.cache_prompt` / `api_params.cache_ttl`.  The
divergence between those three surfaces, and a proposed common `cache:`
profile field, are assessed in
[Model Tiers × Prompt Caching](docs/design/model-tier-prompt-cache.md) §7.)

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

### Chrome Built-in AI (Gemini Nano, Local)
| Variable | Purpose |
|----------|---------|
| `JAATO_CHROME_AI_BINARY` | Browser binary path (default: search PATH / well-known locations for Google Chrome, then Microsoft Edge) |
| `JAATO_CHROME_AI_CDP_URL` | Attach to an already-running browser (`http://host:port` DevTools endpoint or `ws://` URL) instead of launching one |
| `JAATO_CHROME_AI_USER_DATA_DIR` | Persistent profile directory (default: `~/.jaato/chrome_ai/profile`; the model download is profile-bound) |
| `JAATO_CHROME_AI_HEADLESS` | Run headless (default: `true`; the model must already be downloaded in the profile) |
| `JAATO_CHROME_AI_CONTEXT_LENGTH` | Manual context-window override (normally detected from the session quota) |
| `JAATO_CHROME_AI_MODEL` | Nominal model name (default: `gemini-nano` — the Prompt API has no model selection) |
| `JAATO_CHROME_AI_PAGE_URL` | Page hosting the Prompt API calls (default: `about:blank`) |

Drives the on-device LLM embedded in Google Chrome (Gemini Nano) through the
built-in AI **Prompt API** (`LanguageModel` global; stable for web pages since
Chrome 148, extensions since 138), bridged over the Chrome DevTools Protocol.
No credentials, no API costs, no new Python dependencies (the CDP transport
reuses the core `websockets` package). Microsoft Edge (which ships the
same-shaped API backed by Phi-4-mini / Aion-1.0-Instruct) works via the same
provider.

Requirements & limits:
- **Branded Google Chrome or Edge only** — plain Chromium has no on-device
  model (it's a Google-proprietary component).
- The model (~2-4 GB component; ~22 GB free disk required by Chrome) must be
  downloaded into the browser profile. Set `auto_download: true` to let the
  provider trigger it at `connect()`, or run once headed and evaluate
  `await LanguageModel.create()` in DevTools. On older/gated builds enable
  `chrome://flags/#prompt-api-for-gemini-nano` and
  `chrome://flags/#optimization-guide-on-device-model` (BypassPerfRequirement).
- The context window is tiny (~6-9k tokens, shared input+output, detected via
  the session quota) — pair with an aggressive GC strategy.
- Tool calling is prompt-injected (`tool_call` fenced blocks, parsed by the
  provider — the Prompt API's native `tools` option isn't on stable Chrome);
  expect small-model reliability. Structured output uses the API's native
  `responseConstraint` (JSON Schema) and is comparatively strong.

Profile knobs under `plugin_configs.chrome_ai`:

| Key | Type | Description |
|-----|------|-------------|
| `binary` | str | Browser binary override |
| `cdp_url` | str | Attach to a running browser instead of launching |
| `user_data_dir` | str | Persistent profile dir |
| `headless` | bool | `--headless=new` (default true) |
| `page_url` | str | Page hosting the API calls (point at an https origin if the build gates the API) |
| `reuse_page` | bool | Attach to an already-open tab whose URL == `page_url` instead of creating a dedicated one, and leave it open on teardown (default false). Anchors the Prompt API onto a real https tab the user already has open; falls back to creating a tab when none matches. The page helper is re-installed per turn, so the session self-heals if the anchored tab navigates. |
| `extra_args` | list | Additional Chrome CLI switches |
| `auto_download` | bool | Trigger the model component download at connect (default false) |
| `download_timeout` / `connect_timeout` / `turn_timeout` | int | Seconds: model download / launch+attach / mid-turn silence before abort |
| `context_length` | int | Manual context-window override |
| `warmup` | bool | Run one throwaway generation at `connect()` to absorb the model cold-start (default true; see below). Set false for fastest connect. |
| `api_params.temperature`, `api_params.top_k` | float / int | `LanguageModel.create()` sampling options (unset = browser defaults) |

Performance (measured on real Gemini Nano, Chrome 149, consumer GPU;
fully on-device, zero network/token cost):
- **Cold start dominates the first turn.** The first inference after the
  model is provisioned pays a one-time compile/load cost — **~11s to first
  token** — and can return an empty completion. The `warmup` knob (default
  **on**) runs one throwaway generation at the end of `connect()` so that
  cost lands in setup, not on the caller's first real turn; it's
  best-effort (a warmup failure never fails `connect()`) and skipped under
  `warmup: false` or `skip_model_test`.
- **Warm steady-state is sub-second:** `connect()` ~180ms; a tool-call
  turn ~930ms (ttft ~155ms, ~22 tok/s); a plain-prose turn ~480ms
  (ttft ~135ms, ~42 tok/s). Structured/`tool_call` decoding is ~2× costlier
  per token than free prose — budget for it in tool-heavy loops.

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
| `TENSORRT_LLM_HOST` | trtllm-serve URL (**required** — e.g. `http://localhost:8000`; no localhost fallback) |
| `TENSORRT_LLM_MODEL` | Default model name (matches the engine's `id` in `/v1/models`) |
| `TENSORRT_LLM_CONTEXT_LENGTH` | Context window size (**required** — trtllm-serve does not surface `max_seq_len` in `/v1/models`) |
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

### vLLM (`vllm serve`)
| Variable | Purpose |
|----------|---------|
| `VLLM_HOST` | vLLM server URL (**required** — e.g. `http://localhost:8000`; no localhost fallback) |
| `VLLM_MODEL` | Default model name (matches the model's `id` in `/v1/models`) |
| `VLLM_CONTEXT_LENGTH` | Context window size (**optional override** — current vLLM surfaces `max_model_len` in `/v1/models`, which the provider auto-detects at connect via `resolve_context_window` (post-#281). Set this only to override the detected value, or as a fallback for older vLLM builds that don't report it.) |
| `VLLM_API_TOKEN` | Optional bearer token (only when the server was launched with `--api-key <token>` — vLLM's native bearer auth — or fronted by an auth proxy) |

Talks to a vLLM OpenAI-compatible server (`vllm.entrypoints.openai.api_server`) the user has already launched. Provider is **passive** — no in-band model load endpoint; model choice (`--model`), context length (`--max-model-len`), and the tool-call parser (`--enable-auto-tool-choice --tool-call-parser <name>`) all live at server-launch boundary.

Profile knobs under `plugin_configs.vllm`:

| Key | Type | Description |
|-----|------|-------------|
| `host` | str | Override `VLLM_HOST` |
| `context_length` | int | Context window override (required for long-context engines) |
| `api_token` | str | Bearer token override (when server uses `--api-key`) |
| `max_tokens` | int | Cap on per-request output budget; forwarded as OpenAI `max_tokens`. Omit to let vLLM apply its own default (bounded by `--max-model-len` minus prompt). |

Quick start:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export VLLM_CONTEXT_LENGTH=32768  # optional — overrides the auto-detected max_model_len
```

For tool-calling, also pass `--enable-auto-tool-choice --tool-call-parser <name>` matching the model family (e.g. `hermes` for Qwen2.5, `mistral` for Mistral Instruct, `llama3_json` for Llama 3.1, `pythonic` for Llama 3.2 / 4, `granite` for IBM Granite, ...). See the vLLM Tool-Calling docs for the full parser list.

Benefits:
- High-throughput batched inference on NVIDIA GPUs (PagedAttention, continuous batching, prefix caching)
- Self-hosted — no API costs, data never leaves your hardware
- 200+ supported model architectures; LoRA adapter hot-loading; structured outputs (`json_schema`, guided decoding via `extra_body`)

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
| `JAATO_OPENROUTER_REQUEST_TIMEOUT` | Byte-level per-request deadline in seconds (default 600; `0` disables) |
| `JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT` | Streaming payload idle deadline in seconds (default 300; `0` disables) |

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
      service_tier: "flex"         # auto|default|flex|priority|scale —
                                   # OpenAI-style processing tier forwarded
                                   # to tier-supporting upstreams (flex =
                                   # ~50% off, slower; priority = faster,
                                   # premium).  See
                                   # https://openrouter.ai/docs/guides/features/service-tiers
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
      connect_timeout: 15          # seconds; TCP + TLS handshake
      request_timeout: 600         # seconds; httpx read/write/pool (0 = none)
      stream_idle_timeout: 300     # seconds; payload idle deadline (0 = none)
```

| Layer | Keys | Purpose |
|-------|------|---------|
| top-level | `api_key`, `http_referer`, `app_title`, `app_categories`, `extra_headers` | auth / identity. `http_referer` / `app_title` are the highest tier of app attribution: unset, they fall back to `JAATO_OPENROUTER_*` and then to the resolved [application identity](#application-identity-naming-the-app-not-the-framework) (`JAATO_APP_NAME` → `"<app> (powered by jaato)"`), and finally to jaato's own name. `app_categories` (`List[str]`) is jaato's hook into [OpenRouter's app marketplace](https://openrouter.ai/docs/app-attribution) — emitted as the `X-OpenRouter-Categories` header. Default is `["cli-agent"]` (jaato is a terminal-driven agentic tool orchestrator); pass `[]` to opt out of category attribution entirely. Validated strictly: lowercase hyphen-separated slugs, ≤30 chars each, ≤5 entries; unrecognized categories are silently dropped server-side. `extra_headers` (`Dict[str,str]`) is the hook for OpenRouter's [provider-specific beta headers](https://openrouter.ai/docs/features/provider-routing#provider-specific-headers) — Anthropic `x-anthropic-beta` is the canonical case (`fine-grained-tool-streaming-2025-05-14`, `interleaved-thinking-2025-05-14`, `structured-outputs-2025-11-13`). Both merge into the OpenAI client's `default_headers`; profile values win on key collisions. |
| `api_params` | `temperature`, `top_p`, `top_k`, `max_tokens`, `models`, `service_tier`, `enable_thinking`, `thinking_budget`, `thinking_level`, `cache_prompt`, `cache_ttl`, `strict_tools` | OpenAI Chat Completions body fields. `models` is OpenRouter's request-level cross-model fallback list (sibling of `model`; OpenRouter walks candidates on failure). `service_tier` (`auto` / `default` / `flex` / `priority` / `scale`) is the OpenAI-style processing-tier selector, forwarded to tier-supporting upstreams (OpenAI, Gemini, ...) per [service tiers](https://openrouter.ai/docs/guides/features/service-tiers) — `flex` trades latency for ~50% off, `priority` the reverse; the response reports the tier actually used. `thinking_*` keys mirror Anthropic / Antigravity; when both `thinking_level` and `thinking_budget` are set, `level` wins (more portable across upstreams). `cache_prompt: "auto"` (default) places `cache_control: {type: ephemeral}` breakpoints on the system block and last tool definition for explicit-cache upstreams (Anthropic, Gemini 1.5+/2.5+/3+); other upstreams (OpenAI, DeepSeek, Grok) cache automatically and need no client annotation. Response-side parsing of `prompt_tokens_details.cached_tokens` / `cache_creation_input_tokens` / `cost` is unconditional. `strict_tools: true` (server 0.6.118+) emits `"strict": true` as a sibling of `parameters` in each tool definition; OpenRouter forwards to supported upstreams (Sonnet 4.5 / Opus 4.1+, GPT-4o+, Gemini, OSS, Fireworks per [structured outputs list](https://openrouter.ai/docs/guides/features/structured-outputs)) for grammar-constrained tool-arg sampling. Required for cascade-determinism use cases (see `feedback_cascade_completion_schemas_require_strict_model_support` memory); the framework does NOT auto-rewrite schemas to satisfy OpenAI's strict-mode requirements (kb authors own schema shape — `additionalProperties: false` on every object, exhaustive `required` arrays, no `oneOf`/`anyOf` if you enable strict). |
| `routing` | any [provider routing](https://openrouter.ai/docs/features/provider-routing) key (`order`, `allow_fallbacks`, `require_parameters`, `data_collection`, `ignore`, `only`, `quantizations`, `sort` (string or `{by, partition}`), `zdr`, `enforce_distillable_text`, `max_price`, `preferred_min_throughput`, `preferred_max_latency`, ...) | constrains which upstream host serves a request. Composes with `model: "openrouter/auto"` (auto picks model, routing constrains hosts) and `api_params.models` (cross-model fallback list, routing constrains providers across all of them). Opaque pass-through — new routing keys land automatically. |
| `framework_overrides` | `context_length`, `base_url`, `connect_timeout`, `request_timeout`, `stream_idle_timeout` | rare escape hatches; normally context length is discovered from the OpenRouter catalog at connect time. The three deadlines are what bounds a single request (#732) — before them, a stalled upstream left the provider waiting forever and delegated the timeout to whoever sat above it. `connect_timeout` (15s) and `request_timeout` (600s, httpx read/write/pool) are byte-level; `stream_idle_timeout` (300s) is *payload*-level, enforced by a watchdog around the streaming chunk loop, because OpenRouter's `: OPENROUTER PROCESSING` keep-alive comments reset httpx's read clock while no chunk is ever yielded. On expiry the provider closes the transport and raises `StallTimeoutError` (a retryable `InfrastructureError`), so `with_retry` handles it like any other transient. Each accepts `0` to disable. The OpenAI SDK's own `max_retries` is pinned to 0 — the framework owns retries, and a hidden second budget would triple every deadline. |

**Backward compatibility:** the same keys are also accepted at the
legacy flat position (`temperature:` / `provider:` / `context_length:`
directly under `openrouter:`) with a one-time deprecation warning per
key.  Flat-key support will be removed in a future server release.

### Nebius Token Factory
| Variable | Purpose |
|----------|---------|
| `JAATO_NEBIUS_API_KEY` | API key (jaato namespace, highest priority) |
| `NEBIUS_API_KEY` | API key (the vendor's own documented variable; honored so users who already set it for the Nebius/OpenAI SDK work with no extra config) |
| `JAATO_NEBIUS_BASE_URL` | Endpoint (default: `https://api.tokenfactory.nebius.com/v1`) |
| `JAATO_NEBIUS_MODEL` | Default model name (e.g. `deepseek-ai/DeepSeek-R1`, `meta-llama/Llama-3.3-70B-Instruct`) |
| `JAATO_NEBIUS_CONTEXT_LENGTH` | Override the catalog-detected context window |

**Authentication Options (in priority order):**
1. **Environment variable**: `JAATO_NEBIUS_API_KEY`, then the vendor's `NEBIUS_API_KEY`
2. **Stored credentials**: `nebius-auth` (validates against the OpenAI-compatible `/chat/completions` endpoint and stores securely)

Nebius Token Factory is a hosted **serverless** inference service for open
models (Llama, Qwen, DeepSeek-R1, Mistral, ...) behind a single
OpenAI-compatible API (`https://api.tokenfactory.nebius.com/v1`). Models use
the `vendor/model` form, e.g. `deepseek-ai/DeepSeek-R1`,
`meta-llama/Llama-3.3-70B-Instruct`.

`list_models()` queries `GET /v1/models` (the **RichModel** catalog). At
`connect()` the provider **bootstraps** the active model's metadata from that
catalog — the per-model `context_length` is the PRIMARY context-window tier
(then profile knob `plugin_configs.nebius.context_length` / env, else
fail-loud), and `architecture.modality` (OpenRouter-style `input->output`,
e.g. `text->text` or `text+image->text`) drives input-modality detection for
the multimodal tier system (catalog → `plugin_configs.nebius.modalities` knob
→ text floor). No hardcoded fallback.

Profile knobs under `plugin_configs.nebius`:

| Key | Type | Description |
|-----|------|-------------|
| `base_url` | str | Override `JAATO_NEBIUS_BASE_URL` (e.g. a local proxy) |
| `context_length` | int | Manual context-window override (used when the catalog lacks the model) |
| `modalities` | list[str] | Assert/correct input modalities (e.g. `["text","image"]`) for a model the catalog doesn't classify |

**Self-deployed / fine-tuned models** are supported out of the box — they
run on the *same* serverless endpoint, addressed by name. After you register
a fine-tune with Token Factory (`POST /v0/models` → a custom `name` such as
`legislation-qa-private`, a management step done out-of-band), just point the
profile at it:

```yaml
provider: nebius
model: legislation-qa-private   # your deployed fine-tune's name
```

`connect()` passes the name straight through as the OpenAI `model`.
Catalog-based context/modality auto-detection works for custom models too,
because the provider's `GET /v1/models` fetch is **authenticated** (sends the
Bearer key) and that listing is account-scoped — your deployed fine-tunes
appear there alongside the public catalog, so the per-model `context_length`
(inherited from the `base_model`) is detected. If a custom model isn't listed,
set `plugin_configs.nebius.context_length` (the provider fails loud telling you
so). The deploy/register step itself is a management workflow, out of scope for
this provider.

> **Note (dedicated endpoints):** Distinct from the above, Token Factory also
> offers *dedicated endpoints* (a control-plane API that provisions GPUs and
> exposes a region-specific data-plane URL + routing key). This provider
> implements the **serverless** path only (including serverless custom/
> fine-tuned models); dedicated-endpoint provisioning is out of scope (it
> incurs GPU cost and is managed out-of-band via the Nebius dashboard/CLI).

### OVHcloud AI Endpoints
| Variable | Purpose |
|----------|---------|
| `JAATO_OVHCLOUD_API_KEY` | API key (jaato namespace, highest priority) |
| `OVH_AI_ENDPOINTS_ACCESS_TOKEN` | API key (the vendor's own documented variable; honored so users who already set it for OVHcloud's OpenAI SDK examples work with no extra config) |
| `JAATO_OVHCLOUD_BASE_URL` | Endpoint (default: `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1`) |
| `JAATO_OVHCLOUD_MODEL` | Default model name (e.g. `gpt-oss-120b`, `Meta-Llama-3_3-70B-Instruct`) |
| `JAATO_OVHCLOUD_CONTEXT_LENGTH` | Override / supply the context window when the catalog doesn't report it |
| `JAATO_OVHCLOUD_ALLOW_ANONYMOUS` | Opt into the keyless rate-limited free tier (`1`/`true`/`yes`/`on`; evaluation only — never a silent fallback) |

**Authentication Options (in priority order):**
1. **Environment variable**: `JAATO_OVHCLOUD_API_KEY`, then the vendor's `OVH_AI_ENDPOINTS_ACCESS_TOKEN`
2. **Stored credentials**: `ovhcloud-auth` (validates against the OpenAI-compatible `/chat/completions` endpoint and stores securely)
3. **Anonymous free tier**: explicit opt-in via `JAATO_OVHCLOUD_ALLOW_ANONYMOUS` / the `allow_anonymous` knob (heavily rate-limited)

OVHcloud AI Endpoints is a hosted **serverless** inference service for open
models (Llama, Mistral, Qwen, gpt-oss, DeepSeek distills, ...) running in
OVHcloud's European data centers, behind a single OpenAI-compatible unified
gateway (`https://oai.endpoints.kepler.ai.cloud.ovh.net/v1`). Model IDs are
**case-sensitive** catalog names, e.g. `gpt-oss-120b`,
`Meta-Llama-3_3-70B-Instruct`, `Qwen2.5-Coder-32B-Instruct` — browse them at
https://endpoints.ai.cloud.ovh.net/catalog or via `list_models()`.

`list_models()` queries `GET /v1/models`. At `connect()` the provider
bootstraps the active model's context window from that catalog when it
reports one (the lookup tolerates the common key spellings:
`context_length`, `max_model_len`, `max_context_length`), then falls back to
the profile knob `plugin_configs.ovhcloud.context_length` / env, else
fail-loud. Input modalities resolve catalog → `plugin_configs.ovhcloud.
modalities` knob → text floor (assert vision models the catalog doesn't
classify, e.g. `Qwen2.5-VL-72B-Instruct`, via the knob). No hardcoded
fallback.

Profile knobs under `plugin_configs.ovhcloud`:

| Key | Type | Description |
|-----|------|-------------|
| `base_url` | str | Override `JAATO_OVHCLOUD_BASE_URL` (e.g. a local proxy, or a legacy per-model `*.endpoints.kepler.ai.cloud.ovh.net` endpoint) |
| `context_length` | int | Manual context-window override (used when the catalog doesn't report the model's window) |
| `modalities` | list[str] | Assert/correct input modalities (e.g. `["text","image"]`) for a model the catalog doesn't classify |
| `allow_anonymous` | bool | Opt into the keyless rate-limited free tier (evaluation only) |

### Doubleword
| Variable | Purpose |
|----------|---------|
| `JAATO_DOUBLEWORD_API_KEY` | API key (from https://app.doubleword.ai/api-keys) |
| `JAATO_DOUBLEWORD_BASE_URL` | Endpoint (default: `https://api.doubleword.ai/v1`) |
| `JAATO_DOUBLEWORD_MODEL` | Default model name (e.g. `deepseek-ai/DeepSeek-V4-Pro`) |
| `JAATO_DOUBLEWORD_CONTEXT_LENGTH` | Context window (**required in practice** — Doubleword's catalog reports no per-model window; see below) |
| `JAATO_DOUBLEWORD_SERVICE_TIER` | Inference tier: `flex` (discounted async) or `priority` (realtime); the profile knob wins when both are set |

**Authentication (in priority order):**
1. **Environment variable**: `JAATO_DOUBLEWORD_API_KEY`
2. **Stored credentials**: `doubleword-auth` (validates against the OpenAI-compatible `/chat/completions` endpoint and stores securely)

Doubleword (https://doubleword.ai) is a hosted **serverless** inference
service for open models (DeepSeek, Qwen, GLM, Kimi, gpt-oss, Nemotron, ...)
that prices by **delivery window** on one OpenAI-compatible API
(`https://api.doubleword.ai/v1`).  The same `/chat/completions` endpoint
serves the realtime tier and — via the `service_tier: "flex"` request-body
field — the discounted **async** tier: work is queued and guaranteed to
start within ~1 minute (minutes-level latency, ~1 min to first token) at a
fraction of realtime pricing.  Suits background agents and fan-out
workloads where each turn tolerates a short queue delay.  Model IDs are
vendor-prefixed catalog names, e.g. `deepseek-ai/DeepSeek-V4-Pro`,
`Qwen/Qwen3.5-35B-A3B` — browse them at https://doubleword.ai/models or via
`list_models()`.

`list_models()` queries `GET /v1/models` (**authenticated** — the listing
is account-scoped).

> **You must set a context window.**  Doubleword's catalog serves bare
> OpenAI-shaped entries — verified live 2026-07-19, every one of the 25
> listed models reports only `{id, object, created, owned_by}`, with no
> context-length or modality field.  So
> `plugin_configs.doubleword.context_length` (or
> `JAATO_DOUBLEWORD_CONTEXT_LENGTH`) is in practice **required**: without
> it `connect()` fails loud rather than guessing.  Per-model windows are
> listed at https://doubleword.ai/models.

`connect()` still consults the catalog **first** (tolerating the common key
spellings `context_length` / `max_model_len` / `max_context_length`), so
the manual knob becomes redundant automatically if Doubleword ever
enriches the listing — but today that tier never fires.  Resolution order
is catalog → profile knob → env → fail-loud.  Input modalities resolve
catalog → `plugin_configs.doubleword.modalities` knob → text floor; the
catalog tier is likewise dormant today, so assert vision models (e.g.
`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`) via the knob.  No hardcoded
fallback.

Profile knobs under `plugin_configs.doubleword`:

| Key | Type | Description |
|-----|------|-------------|
| `base_url` | str | Override `JAATO_DOUBLEWORD_BASE_URL` (e.g. a local proxy) |
| `context_length` | int | Context window. **Required in practice** — the catalog reports none, so `connect()` fails loud without it |
| `modalities` | list[str] | Assert input modalities (e.g. `["text","image"]`). Required for vision models — the catalog classifies none |
| `api_params.service_tier` | str | `flex` (discounted async tier) or `priority` (realtime); forwarded verbatim, so future tier names work without a provider release |

```yaml
# profile example: a background agent on the discounted async tier
provider: doubleword
model: deepseek-ai/DeepSeek-V4-Pro
plugin_configs:
  doubleword:
    context_length: 131072      # required — the catalog reports no window
    api_params:
      service_tier: flex
```

> **Note (batch tier):** Doubleword's deepest-discount **batch** tier
> (JSONL file upload + `/batches` jobs with a 24h completion window) is a
> different interaction shape (submit → poll → collect) and is not part of
> this provider; background-job polling and batch-job support are a
> follow-up.

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

Every env var the installed tree reads is tagged with a **scope** in
`jaato-server/shared/env_scope.py` — `session` (a knob two sessions on one host
may legitimately differ on), `host` (process-scoped; a per-session value would
be a lie), `ambient` (the host environment being read, not a knob) or `internal`
(a framework-to-framework handoff) — together with the typed profile key that
covers it, where one exists. `jaato-scaffold explain env` renders the tags;
`explain env untyped` lists the session-scoped knobs that still have none, each with the key proposed for it. See
[Env Vars vs Profile Keys](docs/design/env-vars-vs-profile-keys.md).

| Variable | Purpose |
|----------|---------|
| `JAATO_APP_NAME` | Display name of the **application** built on the framework, used for upstream app attribution (today: OpenRouter's `X-OpenRouter-Title`). Unset means jaato attributes as itself, exactly as before. See [Application Identity](#application-identity-naming-the-app-not-the-framework). |
| `JAATO_APP_URL` | The application's own site/repo — becomes the attributed `HTTP-Referer`. Falls back to the framework's repository. |
| `JAATO_APP_VERSION` | The application's own version (not the framework's); used by `AppIdentity.user_agent()`. |
| `JAATO_APP_POWERED_BY` | Whether attribution appends `(powered by jaato)` (default `true`). Set `false` for a white-labelled product. |
| `JAATO_APP_CATEGORIES` | Comma-separated marketplace categories the application claims (OpenRouter's `X-OpenRouter-Categories`). An app that names itself does **not** inherit jaato's `cli-agent` — declare your own or send none. |
| `AI_USE_CHAT_FUNCTIONS` | Enable function calling mode (`1`/`true`) |
| `LEDGER_PATH` | Output path for token accounting JSONL |
| `JAATO_GC_THRESHOLD` | GC trigger threshold % (default: 80.0) |
| `JAATO_PARALLEL_TOOLS` | Enable parallel tool execution (default: `true`) |
| `JAATO_DEFERRED_TOOLS` | Enable deferred tool loading (default: `true`) |
| `JAATO_RUNNER_POOL_ENABLED` | Enable pre-warm runner pool routing (default: `true`).  Sessions consume pre-warm pool slots instead of cold-spawning a runner subprocess.  Set to `false` / `0` / `no` / `off` to disable.  See `docs/design/runner_prewarm_pool_plan.md`. |
| `JAATO_RUNNER_POOL_SIZE` | Number of pre-warm pool slots to keep idle (default: 2).  Raise for cascades that fan out stages **concurrently** (each simultaneous stage needs its own warm slot).  Sequential/back-to-back stages do NOT need a larger pool — they reuse one warm slot via the `slot.settled` handoff (the next stage is spawned on slot-availability), so pool size >1 only helps parallel fan-out. |
| `JAATO_IPC_EVENT_QUEUE_MAX` | Per-client IPC event-queue bound (default 2048). Beyond it, lossy tool-output chunks are evicted oldest-first (media before text); essential lifecycle events are queued past the bound rather than dropped, because losing one desynchronises the client permanently. A non-numeric or non-positive value falls back to the default — "unbounded" is the bug this exists to fix. See [Binary Media Chunks](docs/design/binary-media-chunks.md). |
| `JAATO_AMBIGUOUS_WIDTH` | Width for East Asian Ambiguous chars in tables (`1` default, `2` for CJK terminals) |
| `JAATO_SESSION_LOG_DIR` | Per-session log directory, relative to workspace (default: `.jaato/logs`) |
| `JAATO_CGROUPS_ROOT` | Parent cgroup v2 directory for the WS server's per-session cgroup tree (default: `/sys/fs/cgroup/jaato`). Override when the host has subtree_control delegated under a different path. Must already exist with `memory`, `pids`, `cpu` in `cgroup.subtree_control`. |
| `JAATO_REQUIRE_APPARMOR` | Require kernel-enforced AppArmor confinement (`1`/`true`/`yes`). Promotes the WS server's auto-detect mode to *required*: if confinement is unavailable the server refuses to start instead of silently degrading to directory-sandbox-only isolation. Equivalent to the WS `--apparmor` flag; combining it with `--no-apparmor` is a contradiction the server rejects at startup. When unset (auto), unavailability is logged at WARNING with the specific failing precondition and the server degrades. |
| `JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC` | Opt into in-process execution of model-authored notebook cells (`1`/`true`/`yes`). The `notebook` plugin's `local` backend runs cells via `exec`/`eval` in the host interpreter, so by default it **fails closed** unless a kernel-enforced AppArmor profile is active (the production confined-runner path). Set this (or notebook plugin config `allow_inprocess_exec: true`) to accept in-process execution on unconfined hosts (e.g. trusted single-user dev). Logs a one-time WARNING when execution runs unconfined via this opt-in. |
| `JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST` | Comma-separated distribution names allowed to contribute plugins through the `jaato.*` entry-point groups. Unset (the default) means every installed distribution participates. When set, an entry point from any other distribution is refused **before** `ep.load()` — so its module is never imported — with a WARNING naming it. The built-in package is always honoured and never needs listing. See [Entry-point plugin trust](#entry-point-plugin-trust). |
| `JAATO_REVIVE_PROFILE` | Where a REVIVED session's profile comes from: `persisted` (default — the resolved recipe the session froze at creation) or `disk` (re-resolve `profile_name` against the profile files as they stand now). Set `disk` to interrogate a finished session under a different contract, where a `JAATO_PROFILE_SET` switch must actually take effect. |
| `JAATO_REVIVE_PERSONA` | Where a REVIVED session's system instruction comes from: `persisted` (default — the exact prompt rendered at session-prep, prefetch output included) or `disk` (re-render from the agent markdown, **re-running** the persona's `{{!py:...}}` prefetch scripts against the session's original `agent_params`). The default is what makes a prefetch run once as documented; `disk` may execute side effects. |
| `JAATO_PLUGIN_ALLOW_SHADOW` | Comma-separated built-in plugin names an out-of-tree distribution IS allowed to replace. Built-in names are reserved by default; a foreign entry point claiming one is refused. Names in the never-shadowable set (`permission`, `cli`, `file_edit`, `mcp`, `sandbox_manager`, `interactive_shell`) are refused even when listed here. An honoured shadow still logs a WARNING naming the distribution that won. |

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

Spans follow **OpenInference** semantic conventions (`openinference.span.kind`,
`llm.token_count.*`, `llm.model_name`), so they render natively in Arize
Phoenix, Langfuse, and other OpenInference-compatible backends. The LLM span
carries per-call cost as `gen_ai.usage.cost` (Langfuse) and `llm.cost.total`
(Phoenix), resolved in the same precedence as `UsageBreakdown`:
provider-reported `TokenUsage.cost_usd` → operator pricing table
(`.jaato/pricing.json`, computed from model + token counts) → none (backend may
still estimate). Resolution happens in `jaato_session._resolve_span_cost` while
the span is open.

**Langfuse backend:** set `JAATO_TELEMETRY_ENABLED=true` +
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (+ optional `LANGFUSE_HOST`). The
`langfuse` backend (`LangfusePlugin`, an `OTelPlugin` subclass) derives the
`/api/public/otel` endpoint, `http/protobuf` transport (Langfuse is HTTP-only;
the generic exporter is gRPC-first), and Basic-auth header from the keys. It's
auto-selected when a Langfuse public key is set and no
`OTEL_EXPORTER_OTLP_ENDPOINT` is configured; force with
`JAATO_TELEMETRY_BACKEND=langfuse` (`=otel` to opt out). For a generic OTLP
collector, set `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` (or the `protocol`
config key) yourself. See
[docs/opentelemetry-design.md §12.1](docs/opentelemetry-design.md).

## Coding Policies

### Cyclomatic Complexity

New functions must score **15 or below** under radon. The gate is
`jaato-server/shared/tests/test_cyclomatic_complexity_audit.py`, which runs in the
required `contract-guards` CI job.

It is a **ratchet, not a threshold**. The tree already carried 416 functions over
15 when the guard went in, so those are frozen in a `BASELINE` dict with their
scores. Three rules follow:

- a function over 15 that is **not** in `BASELINE` fails — split it, or add a
  baseline entry with a comment justifying it;
- a **baselined** function may not grow past its recorded score;
- a baselined function that gets **simpler or is deleted** fails as stale — lower
  the number or drop the line. This is how the baseline shrinks.

Regenerate the whole baseline (deliberate re-freeze only) with:

```bash
python jaato-server/shared/tests/test_cyclomatic_complexity_audit.py
```

Note that radon counts `and`/`or` and comprehensions as decision points, so a run
of defensive `x.get(k) or ""` defaults can push an otherwise flat function over
the line. The ceiling is 15 rather than 10 precisely to leave room for that; see
the test module's docstring for the measurements behind the choice.

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
- [Path Boundary Pattern](docs/path-boundary-pattern.md) - MSYS2/Windows path handling for new components, and the cross-process rule: a **relative path never crosses the daemon boundary** — client-supplied `workspace_path` / `config_root` / `env_file` / trace-log paths are REJECTED, not resolved against the daemon's cwd (#742)
- [OpenTelemetry Design](docs/opentelemetry-design.md) - Comprehensive OTel tracing integration
- [Reliability Policies Config](docs/reliability-policies-config.md) - JSON schema, per-tool thresholds, prerequisite policies, usage examples
- [Daemon Extensions](docs/design/daemon-extensions.md) - Extension points for external packages (session hooks, WS interceptors, custom aspects, remote handlers)
- [Application Identity](docs/design/app-identity.md) - Naming the application an integrator built, rather than reporting every SDK-based harness upstream as "jaato". `AppIdentity` + the four-tier precedence (provider knob → provider env → `JaatoRuntime(app_identity=)` → `JAATO_APP_*`), the `(powered by jaato)` suffix, header-safety sanitisation, and why the env vars are `host`-scoped.
- [Env Vars vs Profile Keys](docs/design/env-vars-vs-profile-keys.md) - Which of the 186 env vars earned a typed profile/`plugin_configs` key, and which are correctly env-only. The tagged catalog lives in `shared/env_scope.py` (scope: `session` / `host` / `ambient` / `internal`, plus the typed key where one exists) and is enforced by `test_env_scope_catalog.py`; 38 session-scoped knobs with no typed key sit in a may-only-shrink ratchet, each carrying a tier and a **proposed** key (`explain env untyped` prints both). Includes the credential policy for the three providers whose peers expose an `api_key` knob and they don't.
- [Payload-Schema Conventions](docs/design/payload-schema-conventions.md) - Symmetric authoring guide for `spawn_payload_schema` (input boundary) and `completion_payload_schema` (output boundary). Mirror prefetch required-keys; always carry `warnings[]` / `errors[]` escape hatches; persona ↔ schema consistency check; canonical-hash strip rules; `agent_params` interaction with agent-continuity (§6).
- [Competitor Memory Systems](docs/design/competitor-memory-systems.md) - Survey of nine agent-memory products, sorted by what a *framework* owes: pattern (nothing) / seam (an extension point) / fidelity (a fix) / not ours. Records which items were already expressible as cascade patterns, which memory hot paths are not pluggable, and why the pattern corpus needs `certify/`-style contract tests run against `main`.
- [Agent Continuity Pattern](docs/design/agent-continuity.md) - `{{continuity_scope}}` + memory plugin enrichment + raw/curated lifecycle: persona-level continuity across sessions composed from existing primitives, no new framework code. Reference impl in `jaato-knowledge-manager/.jaato.example/`.
- [Model Tiers × Prompt Caching](docs/design/model-tier-prompt-cache.md) - What `enter_tier` costs when prompt caching is on: cache is keyed per model, so an in-place tier switch re-reads the whole prefix cold (break-even ~6 consecutive calls at the new tier). Covers the `_wire_cache_plugin` gap that made profile cache knobs inert, the system-block tier line that invalidates BP1, and the per-provider knob divergence + proposed common `cache:` field.
- [AppArmor Setup](docs/apparmor-setup.md) - Kernel-enforced workspace isolation. WS deployments confine automatically when AppArmor is available; IPC clients opt in via `IPCClient(..., apparmor=True)` (defaults to `False`).
- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
