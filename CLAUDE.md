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
  - `--ws-token TOKEN` / `--ws-token-file PATH`: bearer token clients must present in the WS Upgrade. Token-file mode 0600 enforced. When neither flag is passed (and `--web-socket` is set), the daemon reads `~/.jaato/ws.token`; if the file doesn't exist, it generates a 32-byte token and persists it there with mode 0600. Local clients can read the same default path for zero-config auth. **Prefer `--ws-token-file`, or neither flag.** A token passed as `--ws-token TOKEN` sits in the daemon's `argv` and is therefore served by `/proc/<daemon_pid>/cmdline` to anything on the host that can read it. AppArmor template v30 denies that read from inside a confined session (#712), but the exposure to everything else on the box is a property of the flag, not of the profile.
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
- `model_provider/openai/`: OpenAI, natively — **two wires in one plugin**: Chat Completions (the shared `_openai_compat` transport, with PDF `file` blocks and `input_audio` enabled because OpenAI's own endpoint carries them) and the **Responses API** (`api: responses` — flat `input` items, typed SSE events, reasoning summaries). `context_length` must be set: OpenAI's catalog reports no window for any model
- `model_provider/azure_openai/`: Azure OpenAI — deployment-name routing (`model:` carries the *deployment*, not the model id), a required `api_version`, and resource-key **or** Microsoft Entra ID auth (`auth: aad`, via `azure-identity`); `context_length` must be set
- `model_provider/bedrock/`: **AWS Bedrock** — Anthropic, Amazon Nova, Meta, Mistral, Cohere, AI21 and DeepSeek behind ONE message-shaped API (`Converse` / `ConverseStream`) in the customer's own AWS account. The only provider here that resolves **no credential of its own**: SigV4 signing is botocore's, so `initialize()` builds a `boto3.Session` and asks it what it found (env / profile / SSO / instance role). `context_length` must be set — Bedrock's catalog reports no capacity
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
- `model_provider/minimax/`: MiniMax (hosted, OpenAI-compatible; `MiniMax-M3` 1M window with adaptive thinking + image input, `MiniMax-M2.7` 200K always-thinking; `reasoning_split` requested on every call, reasoning replayed as `reasoning_content` + `reasoning_details`; `tool_choice` folded to `none`/`auto`; `max_completion_tokens` always sent; built-in context table beneath the `context_length` knob; region-bound keys, `.cn` platform via `base_url`)
- `model_provider/kimi/`: Moonshot AI Kimi (hosted, OpenAI-compatible; `kimi-k3` 1M, `kimi-k2.7-code` / `kimi-k2.6` 256K; `GET /v1/models` reports context length **and** image/reasoning flags, so both are catalog-detected; `thinking_level` → K3 `reasoning_effort`, `enable_thinking` / `thinking_keep` → K2.6's `thinking` object; sampling parameters are a 400 on this wire and are not forwarded; tools stamped `strict: false`; the quota-exhausted 429 stops the retry loop; Kimi Code plan via `base_url`)
- `model_provider/mimo/`: Xiaomi MiMo (hosted, OpenAI-compatible; `mimo-v2.5-pro` text / `mimo-v2.5` omnimodal, both 1M; thinking toggle `enable_thinking` → `thinking: {type}`; **reasoning replay is mandatory** — the vendor answers 400 to a tool loop that omits `reasoning_content`; `tool_choice` is `auto` only; not available in the EU, UK or Korea)

All three set `replay_reasoning = True` — see **Reasoning Replay** below.

**Model Quirks** — per-model workarounds a profile opts into via `quirks:`
(injected into `config.extra["quirks"]`; each provider declares the names it
honors in its `PROVIDER_QUIRKS` contract):

| Quirk | Honored by | Effect |
|-------|-----------|--------|
| `prose_tool_calls` | all OpenAI-compat providers (nim, nebius, ovhcloud, doubleword, minimax, kimi, mimo, lmstudio, tensorrt_llm, triton, vllm, zhipuai_openai) + openrouter | Prose-emulated tool calling for upstream models that cannot emit native tool calls: the `tools` array is withheld, schemas are prompt-injected (hashed wire ids, model picks by description), tool traffic in history is replayed as text, and fenced ` ```tool_call ` JSON blocks in the response are parsed back into `FunctionCall` parts. Reliability tier below native tool calling (hallucinated ids surface as recoverable unknown-tool errors; malformed blocks stay visible in text). Shared machinery in `model_provider/_prose_tools.py` — the same protocol `chrome_ai` uses unconditionally. |
| `coerce_typed_tool_args`, `force_tool_choice_for_lifecycle`, `force_narration_between_tools`, `auto_finalize_on_complete` | vllm | Small-model tool-calling workarounds; see `vllm/provider.py` |

```yaml
# profile example: a cheap OpenRouter model that answers in prose
provider: openrouter
model: some-vendor/cheap-model
quirks:
  prose_tool_calls: true
```

### Reasoning Replay (interleaved thinking)

The thinking models of MiniMax, Kimi and MiMo ask the client to send the
assistant's `reasoning_content` **back** on the next request of a tool-call
loop — MiMo returns `400` without it, Kimi K3 wants the assistant message
back "as-is", MiniMax measures a large quality drop. Before the seam the
session dropped thought parts from history and every OpenAI-shaped converter
ignored `Part.thought` on replay, so no provider could satisfy that rule.

The seam is opt-in per provider via `OpenAICompatProvider.replay_reasoning`
(default `False`; every pre-existing inheritor's wire is byte-identical) and
declared to the capability contract as `reasoning_replay`:

| Touch | Where | Effect |
|-------|-------|--------|
| the turn's reasoning becomes a leading `Part.thought` | `_openai_compat/base.py` (streaming loop + `_finish_batch_response`) | history has something to replay; `ProviderResponse.thinking` still feeds the UI |
| the session keeps thought parts in history | `JaatoSession._add_model_response_to_history`, gated `provider.replay_reasoning is True` | a mock or a non-opted provider changes nothing |
| the converter replays them | `message_to_openai(..., reasoning_fields=)` → `{"reasoning_content": text}` by default, `content: ""` next to `tool_calls` | a vendor with a second field overrides `_reasoning_replay_fields` (MiniMax adds `reasoning_details`) |
| GC sizes them | `gc/utils.estimate_message_tokens` | replayed reasoning is context, and a K3 turn at max effort carries tens of thousands of tokens of it |
| persistence round-trips them | `serialize_message` / `deserialize_message` (already did) | a revived session replays what it replayed live |

Reasoning is read off streaming deltas through `_reasoning_from_delta`, so a
wire that streams it under another field (MiniMax's `reasoning_details[]`)
overrides one method. Full design and the cost argument (auto-caching makes
the replayed prefix cheaper, not dearer): [MiniMax, Kimi and MiMo providers](docs/design/minimax-kimi-mimo-providers.md) §3.

The same PR gave the base three more inert hooks the three providers share:
`_THINKING_KNOBS` + `_apply_thinking_knobs` (profile thinking keys consumed
rather than warned about), `_thinking_request_fields` (merged beneath the
profile's `extra_body`), `_tool_choice_vocabulary` + `_narrow_tool_choice`
(a value the vendor rejects becomes `auto` **with a warning** — never a
silent drop, never a 400), `_MAX_TOKENS_WIRE_NAME` (`max_completion_tokens`
where the vendor deprecated `max_tokens`), `_wire_tools` and
`_map_finish_reason`.

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
- Width: 8 concurrent tools per turn by default, capped per session by
  `runtime_limits.max_parallel_tools` (#862)
- Thread-safe callbacks via thread-local storage

**Whether vs how wide.** `JAATO_PARALLEL_TOOLS` is an on/off switch and was
the ONLY lever: the width was the literal `8`, written twice in
`jaato_session.py` (tool execution, and the background token-count fan-out).
Eight is a reasonable desktop default and a poor fit for two deployment shapes
this tree already supports — a confined runner whose profile deliberately set a
small `pids_max` (eight simultaneous `cli` subprocesses hit the cgroup ceiling
and fail non-deterministically) and a rate-limited internal service behind
`service_connector` (a burst of eight is the wrong shape whatever the memory
ceiling says). Both are `runtime_limits` questions, so:

```yaml
runtime_limits:
  pids_max: 64
  max_parallel_tools: 2      # 1..256; unset = 8
```

`max_parallel_tools` is application-enforced by `JaatoSession` itself rather
than by the subprocess plugins, which makes it the one `runtime_limits` field
an **in-process** subagent can honour — the kernel-enforced trio is refused
there (`ConfinementUnavailableError`), because there is no `fork()/exec()`
boundary to confine. It reaches a runner-served session on the **envelope**
(v6), not by env var: `JAATO_RUNNER_MAX_OUTPUT_CHARS` and friends are set at
cold spawn, and a pre-warm pool slot is forked before the session exists, so a
knob delivered that way would be inert exactly where the default path runs.
**Its two subprocess siblings now ride the same envelope** (v7) for a stronger
version of that reason — see below.

Inheritance is **most-restrictive-wins** — the minimum across every layer that
declares it, like `max_turns` and `budget_control.limits`, and unlike the rest
of `runtime_limits`, which is child-REPLACES. A child may narrow the pool,
never widen it; two parents differing only in the width are resolved by `min()`
rather than reported as a conflict. `jaato-scaffold explain runtime` prints the
whole block with the effective value.

### A Cap Nobody Was Wearing (#735)

`explain runtime` said `tool_timeout_seconds → cli/shell →
subprocess.run(timeout=) per tool call`, and `cli/plugin.py` does exactly that
— when it has been handed a `RuntimeLimits`. Nothing ever handed it one. On a
real daemon, a profile declaring `tool_timeout_seconds: 2` ran a `sleep 60`
for **60.02 s**, and `max_output_bytes: 1000` truncated at the compile-time
50 000:

| path | cap declared | observed |
|---|---|---|
| pool-served (the default) | `tool_timeout_seconds: 2` | tool ran **60.02 s** |
| cold-spawn | `tool_timeout_seconds: 2` | tool ran **60.02 s** |
| both | `max_output_bytes: 1000` | truncated at **50 000** |
| control: `plugin_configs.cli.max_output_chars: 1000` | — | truncated at **1 000** ✅ |

The control run is what locates the defect: the enforcement point was always
fine, and only the **delivery** of `runtime_limits` was broken.

**The env pair was never the session's enforcement path.**
`JAATO_RUNNER_TOOL_TIMEOUT_SECONDS` / `JAATO_RUNNER_MAX_OUTPUT_CHARS` arrive
correctly — the runner's own startup line prints them — and configure
`server/runner/tool_executor.ToolExecutor`, the **Phase-2, cli-only**
`execute_fn`. `RunnerRPC._dispatch_method` uses that object only as a fallback
and routes to `host.session._executor` whenever a session host exists, which
is on every path that dispatches `session.bootstrap` — all of them. A
session's tools run through `shared.ai_tool_runner.ToolExecutor`, whose
`set_runtime_limits` had **no non-test caller**, so `CliPlugin._runtime_limits`
was `None` everywhere. The issue's own table marked cold-spawn ✅ for
forwarding the values; it forwarded them to an executor the session does not
use.

**One vehicle, one application point.**

| Seat | What it does |
|------|--------------|
| `build_session_envelope` + the isolated sub-runner builder | stamp the whole resolved block onto `SessionInitEnvelope.runtime_limits` (**envelope v7**), outside any spawn branch, so pool-served and cold-spawned sessions carry the same thing |
| `server/runner/session.py` | `_runtime_limits_from_envelope` re-parses it; a block this runner cannot parse degrades to "nobody declared limits" with a WARNING rather than refusing the bootstrap |
| `JaatoSession.configure()` | `_apply_runtime_limits` calls `executor.set_runtime_limits(None, limits)` **after** `set_registry` — the forwarding loop walks `registry.list_exposed()`, so a caller that ran earlier would arm nothing |

`configure()` rather than the runner bootstrap is what makes it *one*
mechanism: the in-process lead and in-process subagents reach the same code
with no second call site. Two properties the implementation holds to:

- **`limits=None` is a no-op, not a clear.** Sessions on one runtime SHARE the
  plugin registry, so a limitless in-process subagent calling
  `set_runtime_limits(None, None)` would strip the cap off its parent's tools.
- **`attach_callback` is `None` deliberately.** The runner PROCESS is migrated
  into its cgroup at fork time and its children inherit it, so a per-plugin
  `preexec_fn` would be redundant. The kernel trio still rides the cgroup, not
  the envelope.

**The env pair is kept, and relabelled.** It is the only configuration the
Phase-2 executor has, and that surface is still live for a runner with no
session host (cli-only runners, harnesses, tests); deleting it would silently
drop those back to compile-time defaults — the same class of regression, in
the same direction. Both surfaces read the one source of truth
(`server._profile.runtime_limits`), so they cannot disagree about a number;
they bound different executors. What changed is that the comments no longer
claim the env pair enforces a session's caps — including the slot-mode
docstring in `runner/__main__.py` and the `JaatoServer.set_runtime_limits`
stub, both of which asserted a mechanism that did not exist.

**A cap that silently does not apply is worse than no cap**, so `configure()`
logs the effective block and **names the plugins that took it**
(`runtime_limits armed: tool_timeout=120.0s max_output_bytes=8192
max_parallel_tools=None; receivers=['cli', 'interactive_shell']`). An empty
receiver list beside a declared timeout is the visible form of "this profile
enables no subprocess plugin, so the cap bounds nothing".

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
# default_agent: the persona THIS profile spawns with when the caller names
#   no `agent` (#944).  A profile supplies plugins, an agent supplies
#   instructions — this binds the two, so spawn_subagent(profile=...) alone
#   yields a subagent that has both.  An explicit `agent=` still wins.
default_agent: researcher
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
# runtime_limits: per-session resource caps.  memory/pids/cpu are
#   kernel-enforced (cgroup v2); tool_timeout_seconds / max_output_bytes /
#   max_parallel_tools are application-enforced.  max_parallel_tools (#862)
#   is the width of the tool thread pool (default 8) and inherits
#   most-restrictive-wins; the rest of the block is child-replaces.
runtime_limits:
  pids_max: 64
  max_parallel_tools: 2
# scrub_secret_env: secret env vars stripped from every model-driven
#   subprocess (cli / interactive_shell / mcp).  ON by default (#863) —
#   absent = the framework set; `none` opts out (announced at WARNING);
#   `[default, '!GH_TOKEN']` keeps one tool's token while the provider
#   key stays out of the shell.  plugin_configs.<surface>.scrub_secret_env
#   overrides it for one surface.  See "Secret Env Scrubbing" below.
scrub_secret_env: default
gc:
  type: budget
  threshold_percent: 80.0
  # media accounting + consumed-media eviction (#850); omit to keep defaults
  media_bytes_threshold: 8388608     # bytes of binary payload; 0 disables
  evict_consumed_media: true         # purge audio once its turn completed
  media_evict_mime_prefixes: ["audio/"]
# trace: typed diagnostic log paths — the validated sibling of the
#   JAATO_TRACE_LOG / JAATO_PROVIDER_TRACE env vars, which remain the
#   lower-precedence default (the block outranks both the workspace .env
#   and this profile's own `env:` map).  Absolute = one file shared by
#   every session using the profile; relative = one file per session,
#   resolved against the workspace by jaato_sdk.trace.  Takes ${VAR}
#   expansion like `env:` does, and the per-agent {agent} /
#   {agent_suffix} placeholders.  Refuses a switch written into a path
#   field — `env: {JAATO_PROVIDER_TRACE: '1'}` wrote every session's
#   trace to a file named `1` (#775), and is now refused on BOTH routes.
#   `jaato-scaffold explain profile` prints the whole vocabulary.
trace:
  provider_log: .jaato/logs/provider{agent_suffix}.jsonl
  session_log: .jaato/logs/session_trace.jsonl
# completion_processors: kb Python that gates signal_completion — the
#   OUTPUT-side script hook (the input-side one is the persona's
#   `{{!py:...}}` prefetch).  A `validate` returning errors blocks the
#   completion and hands the agent every string, so it fixes and signals
#   again within max_turns, which IS the retry budget.
#   `max_refusals:` bounds how many times THIS GATE may block — without it
#   the loop does not terminate (the processor refuses, the agent
#   re-claims, forever); `on_exhausted:` says what happens at the ceiling.
#   Full contract: `jaato-scaffold explain completion`, and
#   docs/design/completion-gate.md.
completion_processors:
  - script: scripts/processors/acceptance.py
    name: acceptance          # stable identity for suppress_inherited_processors
    on_error: fail_completion # fail_completion (default) | warn
    phase: finalization       # finalization (default) | completeness
    max_refusals: 3           # unset = unbounded (the pre-#768 behaviour)
    on_exhausted: allow       # allow (default) | fail
# max_completion_nudges: the OTHER direction — how many times the framework
#   re-prompts an agent that ended its loop without calling
#   signal_completion at all, before giving up with NudgeExhausted.  Where
#   `max_refusals` bounds how many times the gate may BLOCK a completion,
#   this bounds how many chances the model gets to CALL it (#919).
#   Unset = 2, which is right for a strong tool-caller and is unchanged.
max_completion_nudges: 4
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

### The Completion-Nudge Budget (#919, #934)

A session whose surface carries `signal_completion` is expected to call it
before its loop ends. When the loop settles without that call the framework
re-prompts the model — a **nudge** — and re-enters the loop; the budget bounds
how many times, before it gives up and emits `NudgeExhausted`.

That budget was a function-local `MAX_COMPLETION_NUDGES = 2` in **three**
files — `server/core.py` (the daemon's top-level guard),
`jaato_embedded/client.py` (the in-process lead) and
`shared/plugins/subagent/plugin.py` (the subagent loop). Nothing kept the three
equal, and none was reachable from a profile — which made it the one bound in
the completion path a deployment could not express:

| bound | configurable? |
|---|---|
| completion-processor refusals | `max_refusals` + `on_exhausted`, per processor |
| turns before the (sub)agent returns | `max_turns` |
| session resource caps | `runtime_limits` |
| **completion nudges** | **`max_completion_nudges`** |

`try_completion_nudge(max_nudges)` always took the bound as an argument, so the
plumbing was already there; what was missing was a value to pass.

```yaml
# .jaato/profiles/<agent>.yaml
max_completion_nudges: 4      # default 2, unchanged when unset
```

**Two stays the default.** For a strong tool-caller it is right, and raising it
globally would make weak models loop longer for everyone. The number belongs to
the deployment: a voice agent whose audio tier hands off, writes to memory,
narrates the write and never calls the tool routinely burns one of its two
nudges on a redundant `enter_tier` (`already_at_tier`), leaving exactly one real
attempt — announcing a tool instead of invoking it is a documented weakness of
that model class, not something persona prose fixes.

- **Per TURN** — how many retries ONE turn gets, not how many turns a
  conversation may have. See below; it used to be per session.
- **Positive integer.** `0` is refused at load, not read as "never nudge": the
  give-up predicate is `nudges_fired >= max`, so a budget of 0 would report
  `NudgeExhausted` on sessions that completed **cleanly**. A deployment that
  wants no nudging keeps `signal_completion` out of the surface.
- **Inheritance follows `max_turns`**: child overrides outright, else the
  minimum across the parents that declared one.
- **One definition.** `shared/completion_nudge.py` owns
  `DEFAULT_MAX_COMPLETION_NUDGES` and the resolver every site now calls, so the
  three paths cannot drift again. A profile predating the field — an older
  session snapshot, or no profile at all — resolves to the default rather than
  raising, so an unconfigured deployment behaves exactly as before.
  `jaato_eval.sign_off` restates the number deliberately (the eval engine must
  not import `server.*` / `shared.*`); that copy is a reporting ceiling, and is
  stale in exactly one direction against a profile that raised its own.

**Whose turn spent it (#934).** The budget lives in one counter,
`_completion_nudges_fired`, and the only question about it is when a turn start
clears it. Both simple answers are wrong, and each was shipped:

| Reset | Bounds | Breaks |
|-------|--------|--------|
| every turn | nothing — a nudge RE-PROMPTS, so the turn the nudge created handed back the token it had just spent. Observed: "nudge 1/2" logged three times in one session, 735 turns in 40 seconds against a live daemon, and the subagent loop's `while ... < MAX_COMPLETION_NUDGES` could not terminate at all (#767) | the runaway guard |
| never | the SESSION | the conversation (#934) |
| **the turns a nudge did not create** | one turn's retries | — |

Never resetting was correct while one clause held: *a completion-gated session
is one-shot by construction*, so a session-lifetime budget cost nothing.
**#913 / #915 made that false** — recording `signal_completion`'s tool result
is precisely what lets a completed session be driven again, and #845 / #914
lets the next turn arrive with an attachment. A bound written to stop a runaway
retry loop *inside one turn* had become a ceiling on how many turns a
conversation may have.

It shows up wherever the nudge is load-bearing on **every** turn rather than
being an exception path — measured on a voice agent across 43 sessions, whose
model announces `signal_completion` instead of invoking it (a documented
weakness of that model class, not something persona prose fixes):

```
 1 model text  ->   2 NUDGE  ->   3 signal_completion   ok
13 model text  ->  14 NUDGE  ->  15 signal_completion   ok
18 model text  ->  19 NUDGE  ->  20 signal_completion   ok
29 model text  ->  30 NUDGE  ->  31 signal_completion   ok
38 model text  ->  (no budget left)                     never completes
```

Deterministic, and every later turn ends `NudgeExhausted`. Raising the knob only
moves the wall: 2 dies at turn 3, 40 at turn 41.

The distinction #767 actually needs is not *never reset* but *do not let a
nudge refund itself*, so the reset asks **who started this turn**.
`try_completion_nudge` latches `_completion_nudge_turn_pending` in the same
call that spends a token; `_begin_turn_completion_state` consumes the latch and
KEEPS the counter, so the nudge loop terminates exactly as before. Any other
turn — a user message, a `session.wake`, a parent's `send_to_subagent` — is
caller-originated and starts with a full budget. Consequences:

- **Every nudge site spends through `try_completion_nudge`.** The subagent loop
  used to bump `_completion_nudges_fired` in place; that re-prompt now reads as
  caller-originated, the reset refills the budget, and #767 is back. The method
  is the only writer.
- **Exhaustion is a verdict on the turn that failed**, not on every turn after
  it. `NudgeExhausted` still terminates the session at the ceiling.
- **Nothing is persisted**, so a revived session begins with a full budget —
  the same answer the reset gives its first caller-originated turn.

### Session Revive (waking a persisted session)

A session woken from disk — `session.wake`, a reattach, anything reaching
`SessionManager._load_session` — comes back with **what it persisted**, not
with what the files on disk say today (issue #787):

| What | Persisted as | Restored via |
|------|--------------|--------------|
| the resolved profile | `SessionState.profile_snapshot` (`profile_to_snapshot`) | `profile_from_snapshot` → `BootstrapEnvelope.profile` |
| the rendered system instruction | `SessionState.rendered_instructions` (snapshotted at the end of `JaatoSession.configure()`) | `BootstrapEnvelope.system_instruction_override` |
| the creation `agent_params` | `SessionState.agent_params` | `BootstrapEnvelope.agent_params` |
| the authenticated creator (#859, record 2.9+) | `SessionState.created_by` | `BootstrapEnvelope.created_by` → `SessionInitEnvelope.created_by` → `set_client_user_id` |

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

**A shared registry is a shared mutable object (#938).** The last bullet is
also a concurrency contract: `spawn_subagent` exposes the child's plugins —
`self._exposed.add(...)` on the `PluginRegistry` — from the spawning thread
while the parent's model thread is part-way through a read that walks the same
set. `registry.get_plugin_for_tool` iterated it live, so a spawn could raise
`RuntimeError: Set changed size during iteration` inside the parent's model
loop and terminate the parent's turn — reaching the caller as an opaque
`RunnerCallError`, indistinguishable from a provider failure.

Two things made it routine rather than theoretical: it is the **cache-miss**
path, which a profile that subsets tools (`plugins: ["memory(tools:[...])"]`)
is in constantly because it is resolving names the cache has not seen, and
`_apply_tool_scopes` runs on **every** provider call — so the parent is in that
loop for the whole round trip that follows the `spawn_subagent` result.

The registry takes **no lock**: every read path calls into plugin code, and
holding a lock across those callbacks invites deadlock. The invariant is
cheaper — **every read path iterates a snapshot** (`list(self._exposed)`,
`list(self._plugins.items())`, ...), never the live container, and looks each
name up inside the `try`/`except` that already wraps these loops, which is what
absorbs a plugin that vanished between the snapshot and the lookup. Removals
use `dict.pop(key, None)` rather than `del` for the same reason. A snapshot
buys consistency-of-iteration, not a consistent view: a reader may see a plugin
being unexposed or miss one being exposed — both were already true of any
unsynchronized read here, and both are recoverable where a lost turn is not.
`test_registry_iteration_snapshots.py` carries an AST guard over `registry.py`
so the next read path cannot reintroduce the shape silently.

### Spawning Requires a Profile (#944)

`spawn_subagent` let the model omit `profile`. The subagent then inherited the
parent's **entire** plugin set with **no system instructions**, and the call
returned `success: true` — indistinguishable, from the caller's side, from a
correct delegation. A voice agent told to delegate document-writing to a
`documentalista` profile spawned that instead: no `file_edit`, no persona,
nothing written, and the person was told a document was on its way.

The knob against it, `allow_inline`, was declared, documented, and **advertised
to the model** by `list_subagent_profiles` — and read nowhere. Setting
`allow_inline: false` changed nothing, so a profile could tell an agent "inline
is not allowed" and then allow it. The dataclass said `True`; the config schema
every consumer reads said `False`. The advertised contract was already the
right one — what was missing was the implementation.

**`allow_inline` now defaults to `false`, and enabling it is the explicit,
WARNING-announced act** — the posture `scrub_secret_env` (#863) and
`--ws-unsafe-no-auth` already take. A workspace that genuinely spawns inline
sets `plugin_configs.subagent.allow_inline: true`; every other workspace gets
the safe default without editing anything.

Four surfaces answer to the knob, and the **schema is the load-bearing one**:

| Surface | With inline denied (default) |
|---------|------------------------------|
| `spawn_subagent`'s `required` | `["task", "profile"]` — the bad call is *unrepresentable*, enforced by the provider's function-calling validator, not corrected after the fact |
| the tool + `profile` descriptions | stop offering "EITHER a profile OR a descriptive name"; models read descriptions, and leaving that in place while `required` says otherwise reproduces the same mismatch one layer up |
| `inline_config` | absent from the wire body, and refused at execution |
| the executor's gate | an omitted `profile` is an error listing what IS available — the wording a *wrong* profile name has always produced |

A runtime error alone would be a retry loop that spends turns and can exhaust
the completion-nudge budget; a `required` field is a contract the model cannot
step outside of. The schema is rebuilt **per exposure** — subagents share the
parent's `PluginRegistry`, so a memoised one would leak one agent's knob into
another's tool list.

The gate sits **before** the remote-spawn branch, which forwards
`profile_name or ''` and returns: unprofiled means the same thing on a peer as
it does here.

**`inline_allowed_plugins` binds both inline paths.** Its only enforcement site
was nested inside `if inline_config:` → `if 'plugins' in inline_config:`, so a
profile saying "inline subagents may hold only `cli` and `todo`" delivered the
parent's whole set to any spawn that simply did not mention `inline_config` —
the restriction bound only an agent that opted into being restricted.

**A profile can name its own persona: `default_agent:`.** `profile` supplies
plugins; `agent` supplies instructions, and requiring only the first yields a
correctly-tooled subagent with nothing to tell it what to do. The binding
belongs in the profile that already knows which persona is its own, rather than
in every caller:

```yaml
# .jaato/profiles/documentalista.yaml
plugins: [file_edit, memory]
default_agent: documentalista      # .jaato/agents/documentalista.md
```

`spawn_subagent(profile="documentalista")` then resolves that persona through
the same `_resolve_agent` path an explicit `agent=` uses — and an explicit
`agent=` still wins, so the profile's binding is a default, not a ceiling. A
`default_agent` naming a file that is not on disk fails by blaming the
*profile*, not the caller who passed no agent, and `jaato-scaffold validate`
reports it as `default_agent_missing` (**error**) before any spawn — by
locating the file, never by rendering it, since rendering runs the persona's
`{{!py:...}}` prefetch and `validate` is side-effect free.

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
    # Media is a SECOND denominator, in bytes (#850) — a voice session can
    # sit far below its token threshold while carrying megabytes of audio.
    media_bytes_threshold=8 * 1024 * 1024,   # 0 disables
    evict_consumed_media=True,               # purge bytes after the turn
    media_evict_mime_prefixes=("audio/",),   # images survive by default
)
client.set_gc_plugin(gc_plugin, gc_config)
```

The same three keys are settable per session from a profile's `gc:` block and
from `.jaato/gc.json`; both layers pass a key only when it is present, so
omitting one leaves the framework default (and `JAATO_GC_MEDIA_BYTES`) in
charge rather than silently overriding it.

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
- **Telemetry**: `PoolManager.get_telemetry()` exposes counters (`pool_slot_acquired_total`, `pool_acquire_miss_total`, `pool_replenish_success_total`, `pool_replenish_failures_total`, `template_respawn_attempts_total`, `template_respawn_failures_total`, `pool_slots_over_cap_total`, `pool_stale_reservation_evicted_total`, `pool_replenish_ceiling_blocked_total`).

**Configuration:**
- Enabled by default (`JAATO_RUNNER_POOL_ENABLED=true`).  Disable with `=false` / `0` / `no` / `off`.
- Pool size via `JAATO_RUNNER_POOL_SIZE` (default 2).
- Ceiling via `JAATO_RUNNER_POOL_MAX_SIZE` (default `2 x` the pool size) — see below.

**A reservation is not capacity (#898).** Slots carry a `cascade_id` and
cross-cascade reuse is forbidden by design (warm plugin state belongs to the
original cascade), so a cascade-affined idle slot is capacity for exactly ONE
tenant. Both capacity sites counted it as capacity for everybody, and a pool at
capacity could therefore be empty from the point of view of every tenant but
one: with two idle slots affined to cascade B and `target_size=2`,
`acquire_slot(cascade=A)` returned `None` (no affine match, no pure-idle) while
replenishment read `idle_count() == 2 >= 2` and never forked. Nothing to run on,
and nothing in the system that would ever create one, until B released — 31 s
past A's 60 s client budget.

The floor and the ceiling now count different things:

| Knob | Counts | Means |
|------|--------|-------|
| `JAATO_RUNNER_POOL_SIZE` (`target_size`, default 2) | **unreserved** idle slots — those with no cascade affinity | how many warm slots ANY arriving session may take.  What replenishment tops up. |
| `JAATO_RUNNER_POOL_MAX_SIZE` (`max_size`, default `2 x target_size`) | **all** idle slots, reservations included | the memory ceiling.  A slot is 129–187 MB, so one reservation per live cascade cannot accumulate unbounded. |

Reservations sit ON TOP of the floor rather than consuming it, which is what
lets a second tenant's arrival grow the pool instead of starving behind the
first. Eviction at the ceiling follows from the same principle — **liveness
outranks warmth**: a returning slot whose cascade is demonstrably mid-run
displaces the **stalest reservation** (the one the 300 s cascade-idle sweep was
going to reap anyway), and only when there is no reservation to spend does it
displace a pure-idle resident, as before. A cascade that loses its reservation
still RUNS — it falls through to an unreserved slot and pays a cold plugin
bootstrap. Warm state for one tenant is negotiable; capacity for every tenant is
not.

Nothing here depends on a tenant declaring "cascade finished" — there is no such
call, and the crash case is when a pinned slot hurts most. The 300 s
cascade-idle sweep remains the backstop. Two counters are the sizing signal:
`pool_stale_reservation_evicted_total` (reservations spent at the ceiling) and
`pool_replenish_ceiling_blocked_total` (replenishment wanted an unreserved slot
and `max_size` refused) — nonzero on a multi-tenant daemon means raise
`JAATO_RUNNER_POOL_MAX_SIZE`.

**Pool routing gates** (`spawn_session_runner`): pool is consulted iff `pool_manager` wired AND env flag enabled AND `cgroup_attach is None` (cgroup migration mid-life is a follow-up).  Apparmor opt-in sessions ARE eligible (slot self-confines to the per-session profile).

See `docs/design/runner_prewarm_pool_plan.md` for the full multi-PR plan + decision log.

### Slot-scoped Plugin Lifetime (#890)

A pool slot serves several sessions of one cascade in turn, and
`reset_for_next_session()` is the hook at that boundary — several plugins
answer it with "keep everything, the next stage benefits". They were keeping
state on an object nobody would read: every `session.bootstrap` built a fresh
`PluginRegistry` and re-ran `discover()`, which calls `create_plugin()`, and
nothing shut the outgoing registry down. Dropping a reference is not freeing a
resource, so `lsp` started a language server per stage and left the previous
one running with no owner — three live jdtls (~2.3 GB) on a 5-subphase run,
each stage paying the cold start the preserved state existed to avoid.

**`TRAIT_SLOT_SCOPED` is what makes the hook reachable.** A plugin declaring it
is carried across the boundary as an INSTANCE:

| When | What happens |
|------|--------------|
| `session.end` (warm — slot returns to the pool) | `reset_for_next_session()` on every plugin as before, then `slot_plugins.park_from`: slot-scoped instances move into a process-level store, **every other initialized plugin gets `shutdown()`** |
| next `session.bootstrap` on that slot | `slot_plugins.adopt_into` registers the parked instances **before** `discover()` — both discovery paths skip a name already registered, so no rival is constructed. `expose_all` still calls `initialize()`; the plugin's own `_initialized` guard is what makes it a no-op and preserves the warm resource |
| `session.shutdown` (cold — slot is being reaped) | everything is shut down, parked and current alike; a graceful `shutdown()` beats the SIGKILL-only process-exit backstop |

Declared today by `lsp` (connected LSP clients + the thread that owns them) and
`todo` (the per-agent plan map a later stage reads). **A no-op
`reset_for_next_session()` is not on its own a reason to declare it** —
`references` has one because it has nothing to clear, while its state
(sources, selected ids, preselected paths) is emphatically per-session, and
carrying that instance would leak one stage's context into the next. The trait
means "everything I hold is deliberately cross-session".

**Reuse is conditional.** A slot returns to the pool and may be handed
unrelated work, so an instance is adopted only when the arriving session
matches on cascade (`cascade_driver_id`), workspace, config root, and that
plugin's declared config — session-identity keys (`session_id`, `agent_name`)
excluded, since counting them would make every comparison a miss. Any
mismatch shuts the parked instance down rather than leaving it running. A
standalone session (no cascade) parks nothing, and neither does a boundary
whose reset sweep raised — the daemon won't pool that slot, so parking for a
next session that never arrives is only a slower leak.

`PluginRegistry` gains `adopt_plugin()` / `is_adopted()` / `shutdown_all()`,
and `set_session_id()` now **broadcasts** to plugins implementing it: a carried
instance's `initialize()` early-returns, so nothing else refreshes the session
identity it logs under.

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

**Inbound attachments dispatch on mime (they are not all images).**
`model_provider/_attachments.py` is the inbound counterpart of
`_media_deltas.py` — a plain module, imported rather than inherited, for the
same reason: the providers speaking OpenAI's format are not the set that
inherits `_openai_compat`. It owns the one question every OpenAI-shaped
converter must answer for an `inline_data` part or a tool-result
`Attachment`: *does this wire carry this mime?* `image/*` becomes an
`image_url` block; `application/pdf` becomes a `file` block **only** where
the wire declares it (`pdf_as_file=True`, i.e. `openrouter`, which declares
`pdf_input=True`); `audio/*` becomes an `input_audio` block **only** where
the wire declares it (`audio_as_input_audio=True`, likewise `openrouter`,
which declares `audio_input=True`); everything else — video, and a part with
**no** declared mime — is withheld, logged at WARNING, and reported to the
model with the same `[Attachment withheld: ...]` note the modality gate uses.
That note now also names what *this* wire accepts, because telling a model
refused audio to retry as "text, or an image" on a wire that carries PDFs
and audio wastes a turn on advice that was never true.

**The ears (#830).** Outbound audio was complete — #824 delivers
model-emitted media to a client, #828 sources `final` from the provider's own
end-of-audio marker — and there was no inbound path at all: `input_audio`
appeared **nowhere** in the tree, so an `audio/*` part was withheld by the
clause that (correctly) withholds video, however loudly a model's catalog
entry declared `audio` as an input modality. The gates deciding whether audio
*may* be sent already existed on both the tier path
(`_validate_modality_tier_capabilities`) and the tool-result path
(`_gate_one_tool_result`); what was missing was somewhere to send the bytes.

| Wire | Shape | Vocabulary |
|------|-------|-----------|
| `openrouter` (and any OpenAI-shaped wire that opts in) | `{"type":"input_audio","input_audio":{"data":"<b64>","format":"wav"}}` | closed: `wav`, `mp3`, `aiff`, `aac`, `ogg`, `flac`, `m4a`, `pcm16` |
| `google_genai` | `Blob(mime_type=..., data=...)` — already carried any mime | the model's own |

The `format` field is the wire's vocabulary, not ours, so a container it does
not name (`audio/opus`, `audio/webm`) is **withheld with a note**, never
relabelled to one it does name — that would be #829 in new clothes. Raw PCM
is the one mime whose whole shape lives in its parameters, and `pcm16` is an
assertion about them (s16le / mono / 24 kHz): a parameter that *contradicts*
it withholds, an absent one is agreement, and `audio/L16` is refused outright
because RFC 2586 makes it big-endian and relabelled big-endian samples are
noise. The framework's own `STREAM_AUDIO_MIME` — what a speaking model emits
— maps to `pcm16`, so the ears accept what the mouth produces.

Gemini needed no wire work at all: `_part_to_google` has always marshalled
any `inline_data` into a `Blob` with the part's own mime. Its
`MODEL_INPUT_MODALITIES` table simply omitted `audio`, and that table is what
the tool-result gate and the tier validator read — so the framework declined
content the wire beneath it would have delivered.

**An attachment no longer means "do not stream" (#837).** The ears and the
voice could not be used in the same turn. A user message carrying an
attachment is routed to `_run_chat_loop_with_parts`, which called the
**batched** `provider.complete()` unconditionally — `_use_streaming` was
never consulted there — and OpenAI emits audio only while streaming, so a
`modalities: {audio: bidirectional}` tier answered the first turn that used
both directions with `400 Audio output requires stream: true`. The path
predates media output and was built for images, where a batched vision turn
is perfectly reasonable; audio input is the first attachment kind whose
*reply* may itself be audio. Both chat loops now take the streaming decision
from `JaatoSession._resolve_use_streaming`, and a streamed parts turn stops
emitting each response's assembled text on top of the chunks it already
delivered. A provider that reports `supports_streaming() == False` still gets
the batched call it always got. See
[Binary Media Chunks §8](docs/design/binary-media-chunks.md).

**An attachment IS content (#838).** One step earlier, the voice turn still
did nothing. `SessionManager.handle_request` decided whether a
`SendMessageRequest` becomes a model turn by reading the message **text** and
nothing else, so a blank-text send returned with a synthetic
`TurnCompletedEvent` and never called `server.send_message` — while
`event.attachments` sat on the same object, read twenty lines later on the
path that branch had already returned from. The same 88 KB `audio/wav`
attachment *with* text reached the provider (and was refused by it, per #837);
with `text=""` it was dropped **before** the wire, and the caller was told the
turn completed. For an image, blank text is unusual — there is normally a
question about the picture; for **audio it is the normal case**, since the
attachment *is* the message, so `session.complete("", attachments=[utterance])`
was exactly the request that silently did nothing. Every layer below already
handled it (`_parts_from_user_message` documents the no-text parts list, the
runner RPC accepts `""` as a valid `str` prompt, and the standalone-WS handler
dispatches such a send with no emptiness check at all), which is what
identifies this one site as the defect. The remaining blank branch —
`SessionManager._close_contentless_message` — now distinguishes its two
arrivals: a solely-`%name --help` message closes quietly because the help
*was* the answer, and a request with no text and no attachments is refused by
name (`ErrorEvent(error_type="EmptyMessageError")`), because a bare
`TurnCompletedEvent` is indistinguishable from a turn that ran and produced
nothing. See [Binary Media Chunks §9](docs/design/binary-media-chunks.md).

**What a model was GIVEN is replayed to whatever model comes next (#847).**
A session that had *heard* audio could not `enter_tier` into a text model.
The utterance stays in history — the audio tier needs it next turn — and
history is replayed on every later request, so the text tier's first
request carried `input_audio` and OpenRouter answered `404 No endpoints
found that support input audio`. Not a missing model: a refused request.
The modality gate existed and covered one direction only
(`_gate_tool_results_for_active_modalities`, tool results), so
`duet`-style outbound-audio profiles were fine and the failure appeared
only once #830 made *inbound* audio possible.
`JaatoSession._gate_history_for_active_modalities` is the other half, and
sits where `docs/design/multimodal-model-support.md` always said the gate
belonged — the send path, right before history→provider conversion — so
every `provider.complete()` call site now reads
`_history_for_provider()` instead of `SessionHistory.messages`. It is
**per-request, never destructive**: it filters a copy, the stored bytes
stay, and switching back restores them; a fix that stripped history would
repair the planner by permanently deafening the session. Withheld content
leaves a note (`_build_withheld_attachment_note`, now taking a
`retry_action` because "re-run this tool" is not the remedy when nothing
needs re-running), since a planner handed a silently-emptied user turn
answers as though the caller said nothing. Tool results already in history
are gated too — `_gate_one_tool_result` ran against the model active when
the result was produced. Gating above the converter rather than teaching
`openrouter/converters.py` to ask `self.modalities()` answers the same
latent hardcode in every OpenAI-shaped converter (`_openai_compat` emits
`image_url` whatever the model declares) with the framework's one answer,
`provider.supports_modality()`. Consequence: for the providers inheriting
the text-only floor from `ModalityCapabilityMixin`, a user-message image
now meets the same withhold their tool-result images always have. See
[Binary Media Chunks §10](docs/design/binary-media-chunks.md).

**How long anyone sees it (#850).** #847 fixed *which* model sees an
utterance; media still had a lifecycle in one direction only. Outbound was
right — model media is `CLIENT`-audience so it never enters history, and
`ensure_spoken_part` leaves the *transcript* in its place. Inbound got
neither: a heard utterance stayed in history verbatim and rode every later
request. Five questions on one helpdesk call measured ~2.8 MB of accumulated
audio and a final request carrying all of it (~3.8 MB base64), growing with
every turn.

**And GC could not see it.** `grep -rn "inline_data" shared/plugins/gc_*/`
returned nothing: `estimate_message_tokens` walked text, function calls and
function responses and never looked at `inline_data`, so a 600 KB utterance
was sized at the one-token floor and no threshold could fire on the payload
it exists to bound. Two channels now, deliberately different in kind —
`estimate_media_tokens` puts media in the **token** denominator (so a
media-carrying turn is sized, and eviction can prefer it), and
`context_usage["media_bytes"]` is a second denominator in **bytes** that
`media_pressure_reason` compares against `GCConfig.media_bytes_threshold`
(all four strategies consult it). Bytes, because the thing that killed the
session was request *size* and an operator bounding it thinks in megabytes;
laundering them through a token estimate would hide the quantity that
matters behind a guess.

**Purging cannot simply delete.** In a call-retention-regulated domain "the
agent handled a claim from audio nobody can produce" is the audit finding.
An inbound attachment carried **no id** (`{mime_type, data, display_name}`)
while outbound media has carried `stream_id`/`sequence` since #824 — so the
identifier is minted at *ingest* (`jaato_sdk/media_identity.py`), as a
SHA-256 digest of the payload rather than a uuid, because the archive side
must be able to **recompute** it from the recording rather than look it up
in a mapping somebody kept. The SDK client mints it (the sender is what
archives the file); `_parts_from_user_message` back-fills the same value for
clients that send none, and never overwrites one the caller supplied.

`JaatoSession._evict_consumed_media` then replaces the bytes with a marker
naming that id, duration and mime. It runs at the **start** of a turn, from
both chat loops: everything in history then belongs to a completed turn, and
a turn that died before the model saw its audio keeps the bytes for a retry.
Unlike `_gate_history_for_active_modalities` (a per-request *copy*), this is
destructive — the accepted trade, since re-hearing a recording yields the
understanding the conversation already records in words. Audio only by
default; an image is routinely re-examined across turns, a recording is not.
Shape **A** of the two the issue names; shape **B** (substitute a
transcript, mirroring `ensure_spoken_part`) needs a transcript source that
chat-completions does not provide for inbound audio, and composes on top.
Knobs: profile `gc:` / `.jaato/gc.json` `evict_consumed_media`,
`media_evict_mime_prefixes`, `media_bytes_threshold`; env
`JAATO_GC_MEDIA_BYTES`. See
[Binary Media Chunks §11](docs/design/binary-media-chunks.md).

**What was said reaches the client (#869).** Every spoken turn produced a
transcript inside the provider and no client could obtain it: the decoder
put the words in a caller-owned sink, and the sink became a history Part
(`ensure_spoken_part`) only after the stream closed, past every point that
emits to a client. A voice client saw 5.45 s of audio in 14 chunks and an
`ask()` returning `''`; a call log could record how long the agent spoke but
not what it said. The `pending` one-slot buffer already holds the last chunk
back to mark it `final`, and the transcript is complete at that moment, so
the chunk released as `final` now carries the whole utterance in
`MediaDelta.transcript` — which `_deliver_model_media` already forwarded into
`ToolOutputEvent.chunk`. Intermediate chunks stay wordless; a client reads
the words exactly once, off the event that also ends playback. Under the
same rule history follows: a turn that wrote its own text sends no
transcript, because those words already went out as `AGENT_OUTPUT`. One
predicate, `model_wrote_text(parts, accumulated_text)`, answers for both
destinations, and both streaming loops pass it as a callable read *at the
marker* rather than a flag read at the start. See
[Binary Media Chunks §12](docs/design/binary-media-chunks.md).

**A session that ended can be driven with bytes (#845).** Everything above
concerns one live turn. There are two ways to drive an **existing** session
— `session.wake` and `inject_prompt` — and both were text-only, while
`attachments` sat on `send_message`, the *live-session* path. A
completion-gated session is designed to END (`signal_completion` releases the
runner) and the documented way back in is `session.wake`; for a voice agent
the next input is a spoken utterance and there was no field to put it in, so
the resume path was closed to exactly the sessions #830 made possible. Both
verbs now take the same `attachments` `send_message` accepts, normalised by
the same `IPCClient._normalize_attachments`, plus typed
`IPCClient.wake_session(...)` / `wakeSession(...)` so the field is visible
from the API surface a reader starts at. Protocol **1.5**.

Three rules the fix holds to:

| Rule | Why |
|------|-----|
| the untrusted boundary is **stated beside** the bytes, not inherited | a wake payload is untrusted (a webhook, a cron, a public comment), and an audio part has no marker to defang. `_wrap_wake_content` names each attachment INSIDE the wrapper — mime, display name, ingest id, never the payload — so a **spoken** instruction is not weighed differently from the identical typed one |
| an attachment-bearing **inject is idle-only** | only the drive branch can carry bytes; a queued message is folded into the running turn as TEXT (a tool result's `model_suffix`, or `Message.from_text`) and has nowhere to put an `inline_data` part. So `deliver_prompt_to_session` forces `require_idle`: a busy target answers `BUSY` with **nothing enqueued** rather than accepting the message and dropping the payload that WAS the message. A text-only inject is unchanged |
| an old daemon is **refused, not degraded** | an additive optional field normally degrades harmlessly — true of #620's `request_id`, false of bytes: the degraded call is a turn driven without the audio, and for a blank-text utterance an empty turn reported as a success. The SDK raises below `MIN_ATTACHMENT_RESUME_PROTOCOL` |

An attachment IS content here too (#838): a wake carrying only an utterance
is valid; one carrying neither text nor bytes is refused by name. See
[Binary Media Chunks §13](docs/design/binary-media-chunks.md).

**A question can be answered with media (#989).** #845 made an existing
session drivable with bytes; this is the direction the agent itself opens.
`request_clarification` blocks the turn until someone answers, and the
answer was text-only — so a user asked "what's your name?" who replies with
a voice note has nowhere to put it, and submitting the audio as an ordinary
turn is stashed silently behind the pending clarification. **A clarification
answer is a tool result, not a user message**, which decides the mechanism:
the bytes ride `ToolResult.attachments`, where all three converter families
already marshal them (`input_audio` included, since #830) and where
`_gate_one_tool_result` already gates them against the active model. The
ferry the issue first proposed — `_parts_from_user_message` — would put a
user turn in history the user never sent and leave the tool result claiming
it carried nothing.

**Attachments are orthogonal to the answer's TYPE.** "Choice answers stay
ordinal, attachments N/A" is wrong:

```
How should we design this?
  1. You attach a screenshot of the design
  2. We discuss the design
```

Choosing (1) and attaching it is `selected_choices=[1]` **plus** an image —
the ordinal says which branch, the attachment says what content. So
`attachments` is a field of `Answer`, beside `selected_choices` /
`free_text` / `skipped`, and the question-side declaration
`Choice.expects_attachment` is per-CHOICE, because in that example option 1
wants a file and option 2 does not. `QuestionType` stays three values; the
choice flag is advisory (a client renders an attach control, nothing
enforces it). The model spells it as `attachment_choices: [1]`, a sibling
array of 1-based indices — `choices` stays an array of strings, because this
schema is ordinal throughout (`default_choice` is the same shape) and
promoting every choice to an object would charge every clarification that
ceremony for a rarely-used flag.

Protocol **1.6**: `ClarificationBatchResponseEvent.answer_attachments` is a
PARALLEL map (1-based question index -> the canonical `{mime_type, data,
display_name, attachment_id}` dicts `send_message` already takes), never a
widening of `answers` into a union — that list is positional in the TUI, the
TS SDK and the web store. #845's rule 3 transfers verbatim: an old daemon
ignores the field and answers with the media gone, which for a voice answer
is a BLANK answer reported as a success (`_parse_answer` reads an empty
response as `free_text=""`), so both SDKs REFUSE below 1.6. Rule 1 transfers
as **labelling only** — today's typed answer is inserted verbatim, so
defanging just the spoken one would weigh it differently from the identical
typed one; each attachment is instead NAMED beside the answer it belongs to
(mime, display name, ingest id — never the payload), which is also what
makes a multi-question batch attributable.

| Refused at submit, clarification left OPEN | Why there |
|---|---|
| an index naming no question; an undecodable payload | the daemon holds the batch it emitted |
| a batch over **6 MiB** of payload | over it the RPC response frame is never written (`FrameTooLargeError`), the runner's call never resolves, and the turn hangs behind a clarification nobody can answer. A 120 s utterance is ~5.12 MB serialised: one fits, two do not, and the multi-question voice policy is still open |
| attachments with no relay waiting | the daemon-local `QueueChannel` carries strings and has nowhere to put bytes |

**Not** refused at submit: whether the active model can consume the mime.
The daemon does not hold the runner's provider and the tier can change
between question and answer (#847), so that stays with
`_gate_one_tool_result`, which answers it against the model the bytes
actually reach and routes what it withholds to the client.

**The GC prerequisite (#850, one part-shape over).** `message_media_bytes`,
`estimate_media_tokens` and the eviction walk read `part.inline_data`
exclusively; `function_response` appeared in `gc/utils.py` only for message
grouping. So a tool result's attachments were sized at the one-token floor
and immune to `_evict_consumed_media` — pre-existing for the `_multimodal`
image tools, and a per-clarification voice answer is where it recurs every
turn. `part_media_views` is now the one answer to "what binary payload does
this part carry". **No new knob**: `media_evict_mime_prefixes` already
defaults to `("audio/",)`, which is exactly right here — the screenshot in a
design discussion survives the turns that discuss it, the voice note
answering "what's your name" does not.

Scoped out, explicitly: subagents (`ParentBridgedChannel` parses answers out
of injected TEXT and has no representation for bytes), and advertising the
accepted mimes at ask time (`ClarificationBatchEvent.accepted_attachment_mimes`
— worth doing, and an unpopulated field would only lie by omission). See
[Binary Media Chunks §14](docs/design/binary-media-chunks.md).

Two shapes were available for #830 and only one is implemented here: audio as
an **input modality** (above), not **transcription as a step**. A transcriber
is a different animal — `microsoft/mai-transcribe-2` is served on
`/api/v1/audio/transcriptions` (multipart), not chat-completions, and every
provider in this tree is a chat provider — so whether it is a tool plugin or a
new provider kind stays an open design question rather than something settled
in passing.

> `_openai_compat/converters.py` previously sent *every* `inline_data` part
> as `image_url` and defaulted a missing mime to `image/png` (#829), so a PDF
> reached the wire as `data:application/pdf;base64,...` inside an image block
> and an audio part as `data:audio/wav;...` — for nim, vllm, lmstudio,
> tensorrt_llm, zhipuai_openai, triton, nebius, ovhcloud and doubleword, every
> one of which declares `pdf_input=False`. The capability declaration and the
> converter now agree. Silently mislabelling was the one outcome worse than
> either carrying the bytes or declining them.

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

**Bytes on the runner RPC wire (#920).** The daemon ↔ runner channel is
length-prefixed JSON, and JSON has no `bytes`. The runner encoded with
`json.dumps(payload, default=str)`, so a binary payload was serialised as its
**Python repr** — every non-printable byte becoming `\xNN`, which JSON then
escaped again. A 120 s utterance (3.84 MB, what `MAX_UTTERANCE_SECONDS`
legally produces) crossed as a **16.13 MB** frame: over the 10 MB
`MAX_MESSAGE_SIZE`, so the peer refused it, closed the transport, and every
in-flight call died with it — including the `session.send_message` that WAS
the turn. Size was the lesser half: `str(b'\x00\xff')` does not round-trip,
so a payload small enough to pass the cap (an image, a short clip, a PDF)
delivered a repr string nothing would ever decode. The cap failing loudly on
size before it could fail quietly on content was luck, not design.

Three fixes, one per layer:

| Layer | Before | Now |
|-------|--------|-----|
| encoding | `default=str` reached bytes | `server/runner/json_codec.py` — bytes become `{"__bytes_b64__": ...}` and decode back to bytes; `str` stays the fallback for genuinely diagnostic objects (a datetime, an enum). Used by **both** ends, so daemon→runner bytes stop raising `TypeError` too |
| the payload that broke | `agent_history_updated` handed raw `Message` objects to that encoder, which stringified each whole message | serialised with the canonical session serializer — the shape `session.get_history` already used (1.33x, and `AgentState.history` holds `Message`s again instead of repr strings the reconnect replay reads `msg.role` off) |
| the blast radius | an oversized frame was written, and the reader — which consumes the length prefix but not the body — could only close the channel | oversized frames are refused at the **write** side: the runner answers that one call with `FrameTooLargeError` and keeps serving, the daemon raises it to that one caller before anything reaches the socket |

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

**The contract is universal; the wire format is not.** Four hooks live on
`ModalityCapabilityMixin`, which every provider already has, so media support is
*implemented or left unimplemented* per provider rather than assumed of all:

| Hook | Default | Meaning |
|------|---------|---------|
| `output_modalities(model=)` | `{text}` floor, raised by `_output_modalities_knob` | what the model can EMIT |
| `supports_output_modality(kind, model=)` | derived | probed by name by the tier startup check |
| `emit_media_delta(delta, on_chunk, seq)` | no-op returning `seq` | decode model media from one streaming delta |
| `request_output_modalities(kinds)` | no-op | ask the model to emit these on later turns |

Deliberately *not* named `modalities()`, which is framework-wide for **input**.

The **OpenAI wire decoder** lives in `model_provider/_media_deltas.py` — a plain
module mirroring `_prose_tools.py`, imported rather than inherited. This matters:
ten providers speak OpenAI's chat-completions format but only five inherit
`_openai_compat.OpenAICompatProvider` (openrouter, lmstudio, vllm, tensorrt_llm
and triton each own their streaming loop), so machinery parked on that base class
is unavailable to half the fleet that speaks its format — which is exactly how
model audio was unreachable through OpenRouter. `delta.audio.data` as base64
pcm16 is one vendor's shape (Google delivers model media as `inlineData` on
parts, Anthropic emits none), so it must not sit in the universal contract either.

Wiring a provider is three touches: add `OpenAIMediaOutputMixin` to its bases,
call `self.emit_media_delta(...)` in its streaming loop, and call
`self.apply_requested_output_modalities(kwargs)` where it assembles the request.
Done for `_openai_compat` (nim, nebius, ovhcloud, doubleword, zhipuai_openai) and
`openrouter`; lmstudio, vllm, tensorrt_llm and triton are the same three touches
when someone needs them.

**A tier's outbound role reaches the wire.** `_connect_tier_entry` (every tier
switch) and the initial-provider build (a session that *starts* in a speaking
tier) both call `request_output_modalities(entry.outbound_modalities)`, which an
OpenAI-shaped provider turns into `modalities: ["text","audio"]` +
`audio: {voice, format}`. The empty set is an instruction, not an absence of one —
it is how leaving a speaking tier stops requesting audio. The **tier says what**
to emit, the **profile says how**: `api_params.audio` always wins, because the
tier stamp uses `setdefault`.

Since no catalog in this tree reports output modalities, an operator assertion is
the only source of truth — hence the `output_modalities` knob (the counterpart of
the input `modalities` knob). Without it the floor stays text-only and the startup
check refuses any outbound role. `ProviderCapabilities.output_media` marks an
adapter proven to deliver media on the wire — declared today by `openrouter`
(verified end to end: a spoken answer reaches a separate client process and
plays) and the five `_openai_compat` inheritors that share its streaming loop.
`jaato-scaffold validate` reads that capability rather than asserting: an
outbound role is flagged INERT only when the named provider does not declare
it, so writing a working speaking tier no longer produces a warning saying it
does nothing.

**Knowing a spoken turn ENDED is its own problem.** OpenRouter reports no
terminal event for a request that asked for audio: measured across 19
consecutive streams of `openai/gpt-audio-mini` — speaking turns and
tool-call-only turns alike — not one named a finish reason, and OpenRouter's
own generation record confirms it at the source (`"finish_reason": null`,
`"native_finish_reason": null`, `"cancelled": false`, `"status": 200`,
`"tokens_completion": 27` — a completed, billed generation naming no reason).
Left unhandled the turn is discarded as a fragment by the #687 truncation
guard. `stream_terminated()` therefore accepts three signals, each evidence
the upstream REACHED THE END: a named finish reason, decoded media, or a
**usage frame** (`stream_options.include_usage` is set, and that frame is the
last of a finished stream, so a severed connection never delivers one).
Deliberately NOT "a completed tool call arrived" — accumulated calls are
exactly what a stream cut mid-`arguments` also leaves behind, and reading them
as completion is how a severed turn becomes an executed one.

```yaml
# a tier that speaks, on OpenRouter
provider: openrouter
plugin_configs:
  openrouter:
    framework_overrides:
      output_modalities: [text, audio]   # assert the model can EMIT audio
    api_params:
      audio: {voice: cedar, format: pcm16}   # optional: pin the voice
model_tiers:
  planner:
    model: openai/gpt-audio
    modalities: {audio: outbound}
```

> **Naming collision.** OpenAI's request field `modalities` means **OUTPUT**
> (`["text","audio"]` with `audio: {voice, format}`); a jaato tier's
> `modalities` key means **INPUT**. Both appear in one profile —
> `api_params.modalities` vs `model_tiers.<tier>.modalities` — and the layer
> disambiguates them. Both `modalities` and `audio` are now in
> `_FORWARDED_API_PARAMS` (`_openai_compat/base.py`); they were previously
> dropped, which made audio output unrequestable through any OpenAI-compatible
> provider. While streaming, OpenAI emits **only pcm16** (24 kHz mono s16le,
> headerless), which is why `STREAM_AUDIO_MIME` spells the parameters out.

### Tool IDs on the Wire and in the Trace (#873)

Tool names reach the model as hashed ids (`t_<8 hex>`, `shared/tool_id_map.py`)
because upstreams enforce `^[a-zA-Z0-9_-]{1,128}$` and MCP names like
`mcp.server.tool` do not pass. The reverse map is an **in-process dict**
populated as a side effect of hashing, not a function of the id — so a
provider trace record that named a call only by `name=<wire id>` was
unreadable in any later process, and the tool inventory needed to re-hash
candidates varies per session and is not recorded.

Every streaming provider that hashes (`_openai_compat` and its inheritors,
`openrouter`, `github_models`, `anthropic`) now renders its tool-call trace
records through `wire_name_trace_fields`, which writes
`name=<wire> tool_name=<resolved>`: `name` keeps meaning "what the wire
said" so existing readers do not change meaning, and `tool_name` is the
resolution made in the only process that can make it. Resolution is never
destructive — a hallucinated id the process never issued is recorded as
`tool_name='t_deadbeef'`, evidence of the invention rather than a
resolved-looking name hiding it. The OpenAI-shaped loops also emit a
`TOOL_CALL_END` record at flush, because an upstream may send the name on a
later delta than the one that opens the call, leaving the `TOOL_CALL_START`
record honestly nameless.

### What a Session Spent, and Which Model Spent It

`get_environment(aspect="context")` has always reported how FULL the context
window is. Nothing reported what the session had **spent** — and nothing
could, because spend was attributed to no model anywhere in the tree:
`_turn_accounting` carries no model stamp and neither does the token
ledger's `response` record. So on a `model_tiers` session "what did the
voice tier cost me" was not a question the stored data could answer. A
measurement gap, not a reporting one.

`aspect="consumption"` is the answer, and the two aspects are deliberately
separate because they are not two views of one number:

| Aspect | Question | Denominator |
|--------|----------|-------------|
| `context` | how full is the window right now | the shared history — belongs to no particular model |
| `consumption` | what has been spent, and on what | the responses each binding served |

A session that calls `enter_tier` has **one history and several bills**.

**The unit of segregation is the BINDING** — `(provider, model, tier)` —
not the tier name. A budget-control degrade rung **rebinds a tier's model
in place** (`planner: opus → flash`) leaving the tier's name unchanged, so
a tier-keyed total would merge two models' spend under one row precisely in
the session someone is reading it because of; `switch_tier` short-circuits
on the resolved entry rather than the name for the same reason. Tiers may
also name different providers, so the model alone is not a key either. A
single-model session reports a list of one with `tier: null`, so no
consumer branches on whether the session happened to be tiered.

**Measured once per response, where spend already accumulates.**
`_accumulate_turn_tokens` is the only hook that runs exactly once per
response on every path — which is why the `spend_*` keys live there — and
`_observe_binding_usage` rides it. Per response rather than per turn
because a turn is not a unit that belongs to one model: `enter_tier` can
fire mid-turn. The per-CHUNK streaming hook (`_track_streaming_usage`)
contributes nothing, under the same rule that forbids it a `spend_` key.

```yaml
get_environment(aspect="consumption")                 # totals + active binding
get_environment(aspect="consumption", detail="full")  # + per-binding rows
```

Four rules the payload holds to, each attached to a way it could mislead:

- **The three input buckets are disjoint and named for what they are.**
  `TokenUsage.prompt_tokens` is the NEW, uncached input and excludes both
  cache counts (#758), so the aspect reports `uncached_input_tokens`,
  `cache_read_tokens`, `cache_creation_tokens` and the derived
  `input_tokens_total`. Passing `prompt_tokens` through under its own name,
  beside `cache_read_tokens`, invites the model reading it to count the
  same tokens twice.
- **Absent is not zero.** A dimension nothing reported is OMITTED, not
  rendered `null` or `0`: a provider with no prompt cache must not read as
  a cache that never hits, and a session with no pricing table must not
  read as free. A reported `0` is a measurement and is shown.
- **A cost says where it came from.** `cost_source` is `provider` (billed),
  `pricing_table` (computed from `.jaato/pricing.json`) or `mixed`. Nothing
  in the tree distinguished them before — `cost_usd` arrived as a bare
  float whose meaning depended on which provider produced it —
  so `_resolve_span_cost` now delegates to `_resolve_cost_with_source`,
  one ladder walked once, because a cost and a provenance derived
  separately is a bug waiting for a provider that reports cost sometimes.
- **Every figure is SPEND.** Never the end-of-turn context size, which
  lives under `active` where it belongs to the history. `turns` per binding
  counts the turns it served in, so a turn crossing an `enter_tier` is
  counted by both; `totals.turns` is the exact distinct count, taken from
  the ledger's own set of turn indices — summing the per-binding column
  double-counts that turn and taking the max under-counts two bindings that
  served different turns.

Two blocks appear only when they mean something. **`budget`** (when the
profile declares `budget_control`) reports the declared dimensions, the
fraction used, and `next_rung` — the lowest rung not yet passed, which is
the deadline the agent can act on where the rungs behind it are history;
an absent key is how "unbounded" is said (#947). **`completion`** (when
`signal_completion` is on the surface) reports the nudge budget:
`nudges_fired_this_turn` is the per-TURN counter the guard actually reads
(#934), `nudges_fired_total` is a lifetime figure that **decides nothing** —
a second meaning layered onto the per-turn counter is how #767's unbounded
nudge loop returns. `max_nudges_source` is `observed` once a nudge has been
considered and `framework_default` before that, because the budget is
resolved from the profile by the caller that nudges and does not reach the
session on the init envelope: `framework_default` means "not observed yet",
not "your `max_completion_nudges` was ignored". Carrying it on the envelope
would remove the caveat and is a wire version bump, deliberately not taken
here.

**Own session only.** A subagent runs its own `JaatoSession` and reports
its own spend; a parent's numbers never silently absorb a child's. The
aggregate has an owner already (`CascadeBudgetPool`), and a second, quieter
total competing with it is how two answers start disagreeing.

**`detail` bounds the feedback loop.** `summary` (the default) is totals
plus the active binding; `full` adds the rows and the declared tier ladder
— including tiers never entered, since a vision tier the agent was given
and never used looks identical, in a spend report alone, to one it was
never given. Under `aspect="all"` the detail is FORCED to summary whatever
is passed: `all` is the eager default a model reaches for when it wants the
OS name, and the result enters the history of the very session it measures.
Asking what you have spent is itself spending.

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

### Tool-Result Enrichment Reaches Every Dict Result (#922)

`enrich_tool_result` speaks **strings**, and most tools return a **dict**, so
something has to decide what text an enricher sees. That decision used to be
a guess: the session enriched fields named one of six well-known names
(`result`, `content`, `stdout`, `output`, `text`, `data`) and only from 100
characters up. Both filters were invisible, and `store_memory` fails both —
its text lives in `message`, and the message measured **83 characters**. So
`memory` and `references`, the only two plugins that implement tool-result
enrichment, never ran on the pairing they exist for (*"the agent just wrote
down something about X; surface what we know about X"*). A catalogued
reference matching a memory's tags 3/3 was never offered, and there was no
error, no warning and not one trace line to say why — the plugin looked
correctly written and simply never fired. More generally, every dict-returning
tool had to happen to name its text one of six ways or be exempt forever.

The session no longer guesses which key holds "the text":

| Result shape | What the chain receives |
|--------------|-------------------------|
| a **string** result | the string, verbatim (unchanged) |
| a dict from a tool declaring `TRAIT_FILE_WRITER` / `TRAIT_GREPPABLE_CONTENT` | the whole JSON (unchanged) |
| **any other dict** | a *text view*: one `key: value` line per scalar field, then the **anchor** field's raw value |

The **anchor** is the field an enriched view is written back to — the first
present name from the conventional six (so tools already using one keep
receiving hints exactly where they did), else the **wordiest** string field.
Wordiest rather than longest is what stops a one-word `status: "success"`
out-weighing a short sentence. `tool_result_text_view` /
`apply_text_view_enrichment` (`shared/tool_result_builder.py`) are the two
halves, and the write-back is what makes one text view serve both enricher
shapes: whatever still carries the header is the anchor's new value, so an
appended hint block (`memory`) and an in-place `@ref-id` expansion
(`references`) both land on the field the tool actually used, and the header
fields are context for matching only — never written back. A dict with no
string field at all takes its addition under `_enrichment`.

Three properties follow, each attached to a way the old path went wrong:

- **No length floor.** A tag match does not need 100 characters of context to
  be valid; the floor was aimed at semantic matching.
- **One chain invocation per result.** The old loop ran once per matching
  field, so a dict with both `content` and `output` collected two hint blocks.
- **A skip is audible.** A dict carrying no text at all traces `ENRICH_SKIP`
  naming its keys, and a run traces `ENRICH` with the anchor and the plugins
  that contributed. Silence was half the defect.

Nested payloads are deliberately not rendered into the view — handing an
enricher a whole structured result is what the two traits above are for.

### Plugin-Level Traits

Plugins themselves can declare **plugin-level traits** via a `plugin_traits` class attribute (`FrozenSet[str]`). These work like tool traits but identify *plugin* capabilities rather than individual tool behaviors.

**Currently defined plugin traits:**

| Constant | Value | Contract |
|----------|-------|----------|
| `TRAIT_AUTH_PROVIDER` | `"auth_provider"` | Plugin provides interactive authentication for a model provider. Must also expose `provider_name` property identifying which provider. |
| `TRAIT_SESSION_PERSISTENT` | `"session_persistent"` | Plugin state must outlive an unload/reload of the SAME session. Must implement `get_persistence_state()` / `restore_persistence_state()`; `SessionManager` snapshots into `metadata['plugin_states'][<name>]`. |
| `TRAIT_SLOT_SCOPED` | `"slot_scoped"` | Plugin INSTANCE survives the cascade session boundary — the runner carries it across sessions served by the same pool slot instead of constructing a new one. `shutdown()` then means slot teardown, not session teardown. See [Slot-scoped plugin lifetime](#slot-scoped-plugin-lifetime-890). |

The three answer different questions and compose freely: `session_persistent`
is "survives THIS session being unloaded and reloaded",  `slot_scoped` is
"survives the NEXT session of the same cascade starting", `auth_provider` is a
capability rather than a lifetime.

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

### Out-of-Tree Plugin Authoring (#917, #918)

Passing the trust gate is not the same as being loaded, and building
against the SDK is not the same as being able to read a credential.
Both gaps hit exactly one audience — a third party writing a plugin
against the documented entry-point surface, which is what that
extension point is *for* — and both failed silently.

**`PLUGIN_TIER` decides whether the session ever sees the plugin.**
Discovery is tier-filtered and the filter is not the same on both
sides: the runner, the runner's `__main__` and `jaato_embedded` pass
`tier_filter="runner"`; the daemon-side registry and
`shared/scaffold/introspect.py` pass none.  A plugin whose package
declares no `PLUGIN_TIER` is excluded under **any** filter — the
deliberate "annotate or be excluded" contract — so the split ran
straight through the diagnostic: the author installed the
distribution, ran `jaato-scaffold plugins`, saw the plugin listed with
its provenance line, wrote `plugins: [m365]` in a profile, and the
session came up without the tools.  No error, no warning, one debug
`_trace`.  `test_plugin_tier_partition` fails the build on this, but
its walk is an AST scan of `shared/plugins/` and cannot see a
distribution outside that path; nothing in the third party's own repo
knows the rule exists.  Worse than a trust refusal, which at least
avoids executing code: `ep.load()` has already run when the tier is
read, so the module is imported and *then* discarded.

Three surfaces now say it, and each distinguishes the mistake from the
mechanism working — a **mismatched** tier (`daemon` under a `runner`
filter) is correct partitioning and stays at debug; only a **missing**
one is announced:

| Surface | Signal |
|---|---|
| `PluginRegistry` | `logger.WARNING` naming the entry point, its `ep.value`, the filter, and the package `__init__.py` to edit — the same PR #171 promotion the protocol-gap check got, for the same audience |
| `jaato-scaffold explain plugins` | the row is marked `[no PLUGIN_TIER - will not load in the runner]`, with a footer naming the fix.  This walk must keep discovering unfiltered (it is an inventory, not a session); what it must not do is imply the plugin works |
| `jaato-scaffold validate` | `plugin_missing_tier`, severity **error** — the profile is valid by every other measure and the session is already broken |

The entry-point path never reads `PLUGIN_KIND` either, so a package
missing **both** constants is registered by that path and dropped by
every filter.  The in-tree `calculator` plugin shipped in exactly that
state, listed in this repo's own
`[project.entry-points."jaato.plugins"]` table and reachable from no
session; it is annotated now, and the gate covers every package that
table names rather than only those declaring `PLUGIN_KIND`.

**`get_session_env` is the credential read, and it is on the SDK
surface.**  The plugin contract is otherwise cleanly SDK-shaped —
`ToolPlugin` / `UserCommand` / `TRAIT_*` from `jaato_sdk.plugins.base`,
`ToolSchema` from `jaato_sdk.plugins.model_provider.types`, and
`jaato-sdk` never imports `shared` — with one hole, sitting where a
connector's credential handling goes.  The session-scoped read lived
only in `shared/session_context.py`, so a third-party plugin either
took a hard dependency on jaato-server or wrote `os.environ.get(...)`.

That is not a missed abstraction.  `JaatoServer._with_session_env()`
overlays each session's `env:` map onto the daemon's `os.environ` for
the duration of a turn, so on a daemon serving two tenants a plain read
can return **another session's token** — non-deterministically, with no
error.  It works on the author's machine and in every single-session
test.

```python
from jaato_sdk.session_env import get_session_env    # also jaato_sdk,
                                                     # jaato_sdk.plugins.base
token = get_session_env("GRAPH_CLIENT_SECRET")
```

`shared/session_context.py` imports the three functions back, so every
in-tree caller keeps its import path and — the part that makes the fix
real rather than cosmetic — there is exactly **one `ContextVar`
object**: the one the daemon sets is the one the plugin reads.  A
second var declared in the SDK would read empty, fall through to
`os.environ`, and reproduce the leak wearing the fix as a disguise, so
the test asserts object identity rather than behaviour.
`get_current_session` is deliberately **not** exported — it hands back
a `JaatoSession`, and a plugin reaching into `session._runtime` is not
something to make easier from out of tree.

### A Knob's Value, Not Only Its Name (#925)

`jaato-scaffold validate` already read every plugin's `get_config_schema()`,
and already kept each knob's declared `type` — then checked the knob **names**
and threw the rest away. So a knob violating the plugin's own declared `enum`
validated clean and exited 0:

```yaml
plugin_configs:
  todo:
    storage_type: sqlite        # declared enum: [memory, file, hybrid]
```

Nothing fails at runtime either, which is the point: `create_storage` raises,
`todo/plugin.py` catches, prints two lines to daemon stdout, and installs
`InMemoryStorage()`. An operator who asked for persistent storage gets none,
and the only signal is a bare `print()` nobody is reading. That is exactly the
silent-ignore class `validate` exists to catch — rename the knob to
`storage_typo` and `unknown_knob` fired immediately, so the schema *was* being
read.

`ConfigSetting` now carries `enum` alongside `type` (JSON Schema spells the
closed set `enum`, the `PluginSetting` object form spells it `choices`; both
land on the one field), and `_validate_plugin_knobs` reads
`config_settings` rather than `config_keys`:

| Finding | Severity | Why that severity |
|---------|----------|-------------------|
| `invalid_knob_value` | **error** | a plugin that spells out `["memory","file","hybrid"]` has left no incompleteness to be generous about |
| `knob_type_mismatch` | **warn** | YAML scalar typing is easy to trip over, a plugin may coerce, and a declared type can be an incomplete summary |
| `unknown_knob` | warn (unchanged) | a plugin may accept free-form keys it did not enumerate |

The generosity that makes an unknown **name** a warning is a claim about
schema *completeness*, and it does not carry over to a **value** the schema
explicitly closed.

Three properties, each attached to a way the check could go wrong:

- **A deferred value is not judged.** `${VAR}` and `pass://` / `vault://` are
  resolved later, against an environment the validator does not have, and
  their literal form is a `str` whatever the knob declares — so
  `timeout: ${HTTP_TIMEOUT}` is not a type error and
  `lookup_strategy: ${STRATEGY}` is not an enum violation. `None` is "unset",
  not "wrongly typed".
- **`True` is not an integer.** Python makes `bool` a subclass of `int`, so
  `timeout: true` would otherwise satisfy a knob declared `integer` on a
  technicality — the same silent shape the check exists to catch. Both type
  vocabularies are understood (JSON Schema's `integer`/`boolean`/`array`, the
  object form's `int`/`bool`/`dict`), a union renders as `string|array` rather
  than a Python repr, and a token in **neither** table asserts nothing: the
  table is a source of findings, never of guesses.
- **It reaches out of tree.** `_PLUGIN_VALUE_CHECKS` is a hardcoded
  jaato-server dict keyed by plugin name, so a third-party distribution had
  **no** route to value validation even though its `get_config_schema()`
  already declared the constraint in machine-readable form. The generic path
  needs no registration. That dict stays for genuinely *structural* knobs
  (`template.file_conventions`), which no declared type can describe.

Still not descended: nested and free-form sub-structures (`permission.policy`,
`permission.evaluators`) — only a top-level knob's own shape is judged.
`jaato-scaffold explain plugins <name>` now prints the permitted set beside
each knob, because it is the set `validate` enforces.

### A Ceiling Nobody Declared, and One That Cannot Stop (#947)

`budget_control` is complete — five dimensions of `limits`, a `degrade`
ladder, `CascadeBudgetPool` — fully wired into the profile, parsed at load,
and validated for internal consistency. That validation runs **only when the
key is present**. Absent, there was nothing: no warning, no default, no
mention in `validate` output. So the failure mode was silent by construction
— an unbudgeted profile behaves identically to a budgeted one right up until
something loops.

Something did. A `documentalista` subagent whose `file_edit` plugin had
failed to initialise retried `writeNewFile` **127 times** over four and a
half minutes, ~800 tokens heavier each cycle (each rejected attempt is
appended to the conversation) until every request carried ~57k tokens. Its
parent had ended two minutes in:

```
Subagent plugin shutdown (running subagents preserved)
```

which is the *correct* behaviour for a backgrounded subagent, and is exactly
what makes an unbudgeted one dangerous — the loop outlived the session that
could have noticed, and stopping it took `kill -TERM` on the pool slot pid.

Two findings, and the second is the one that makes the first honest:

| Code | Severity | Fires when |
|------|----------|-----------|
| `budget_control_absent` | warn | no `budget_control` at all — unbounded on `usd`, `tokens`, `seconds`, `tool_calls`, `turns`. The message names all five, because "missing `budget_control`" teaches nothing to an author who has never seen the knob |
| `budget_limits_without_abort` | warn | `limits` declared and no rung stops the run |

**`limits` are observed, never enforced.** `BudgetTracker` accumulates
against them and `usage_fraction()` turns them into a percentage — and the
`degrade` ladder is the *only* consumer of that percentage. A profile
declaring `limits` and no ladder crosses 100%, 200%, 1000% in silence. Of
the three terminal actions only `abort` reaches `request_stop`; `finalize`
and `escalate` are latched on `_budget_terminal_action` for a layer above,
which is advice a looping model can decline — and did, through 35
consecutive failures without ever emitting text.

That is why the second check is load-bearing rather than a refinement:
without it the first one is **actively misleading**. An author told "you
have no budget" writes `limits: {usd: 5}`, the warning clears, and the loop
is still unbounded. The pair only works together:

```yaml
budget_control:
  limits: {tool_calls: 200, usd: 5.0}
  degrade:
    - at: 95
      action: finalize     # advice
    - at: 100
      action: abort        # the ceiling
```

**Warnings, not errors.** An unbudgeted profile is a legitimate choice for a
short-lived local agent, and an error would fail every existing workspace at
once. Surfacing a knob must not break the people who need it — the same
posture `unknown_knob` takes.

**The danger is not uniform, and the discriminator is used one way.** A
profile spawned as a subagent outlives the thing that would have noticed, so
those deserve a stronger message. Every discovered profile is reachable by
name through `spawn_subagent`, which makes "is it spawnable" useless as a
separator; a profile that names its own `default_agent` is one built to be
spawned by profile name alone (#944). So `default_agent` **strengthens** a
message that fires regardless, and never weakens or suppresses one — its
absence proves nothing.

Not addressed here: `limits` still do not enforce themselves at runtime, and
the archetypes `jaato-scaffold new` emits still carry no `budget_control`
(picking ceilings for someone else's workload is the author's call, which is
what the warning now asks them to make).

### A Ceiling That Only Counted at Turn End (#955)

The runtime half of the ladder above was wired for subagents and still did
not hold. A `documentalista` subagent made **196 tool calls** under
`tool_calls: 100` with `degrade: [{at: 100, action: abort}]` — the very
ladder `validate` prescribes — and nothing aborted, degraded or logged. The
profile reached the session (its `gc` block demonstrably applied) and
`spawn_subagent` passed `budget_control` through; what was missing was the
**moment of observation**. `_budget_observe_turn` was the only writer of
`tool_calls`, `seconds` and `turns`, and it runs in the turn's `finally`.
Every one of the 196 calls happened inside one `send_message`, so the
tracker was handed the count only when that turn ended — and a runaway loop
is precisely the turn that does not end. `tokens` and `usd` were exempt
because they are fed per *response*, which is why only a `usd` ceiling
small enough to cross within one turn had ever been seen to trip.

| Dimension | Was observed | Now observed |
|-----------|--------------|--------------|
| `tokens`, `usd` | per response | unchanged |
| `tool_calls` | turn end | per completed call (sequential + parts loops), per batch (parallel loop) |
| `seconds` | turn end | with every tool-call observation, plus the tail at turn end |
| `turns` | turn end | unchanged — a turn is the unit |

`JaatoSession._budget_observe_tool_calls` is called from every path that
records a call in `turn_data['function_calls']`, and an AST guard
(`test_budget_mid_turn_955.py`) fails the build if a new path records
without observing. `_budget_observe_turn` stays as the **closing entry**: it
settles whatever was not observed mid-turn and only that, so nothing is
counted twice and a budget is exact whichever loop produced the turn. An
`abort` mid-turn cancels the session's token; the sequential loop checks it
before the next call and the main chat loop before the next model round-trip,
so the overshoot is one call, or one parallel batch. The parts loop
(attachment-carrying turns) finishes its batch and is cancelled by the
provider on the first chunk of the next response. A session at its
ceiling is also no longer completion-nudged — the re-prompt would be refused
at turn start, and each refusal spent a nudge on a turn that could not run.

**A ladder that never logs is indistinguishable from one that is not wired**
(the issue's third question). Every ceiling crossing now traces
`BUDGET CEILING dim=... used=... limit=...` once per dimension, rung or no
rung, and every fired rung traces `BUDGET RUNG at=...% action=...`
(`RUNG_SKIPPED`, `EXHAUSTED`) — on the per-agent provider trace and on the
application trace (`trace.session_log`), beside the permission DECISION lines
(#951) an operator correlates them against.

Still true: a profile with `limits` and no `abort` rung crosses in silence
except for that trace line; `finalize` remains advice, and the subagent
that outlives its parent is bounded only by what its own profile declares.

### Configuring a Plugin and Enabling It Are Two Decisions (#950)

`plugin_configs.<name>` and `plugins:` answer different questions — *how does
this plugin behave* and *which tools reach the model* — and `permission` is
the case that separates them: `PermissionPlugin.get_tool_schemas()` returns
`[]` **on purpose**, so it is never in a `plugins:` list, and its whole
configuration is a `policy` block.

`JaatoSession.configure` collapsed the two:

```python
if plugins is None or plugin_name in plugins:      # the block, or nothing
```

A top-level session never noticed — its configs reach the plugins through
`expose_all(plugin_configs)` at bootstrap, and `permission` gets an explicit
merge on both the daemon (`server/core.py`) and runner
(`server/runner/session.py` Step 8) paths. **A subagent reuses the parent's
already-bootstrapped registry**, so that loop was the only place its own
profile's configs could land. A `documentalista` profile whose
`plugin_configs.permission` whitelisted `writeNewFile` therefore produced 55
permission ASKs on that pre-approved tool — under a headless `ClientType.API`
driver an ASK has no channel to reach, so 56 write attempts at one path wrote
nothing, silently. Adding the toolless `permission` to `plugins:` was the
entire difference.

Three other places in the tree already stated the opposite intent —
`SessionInitEnvelope.plugin_configs` ("carries configs for **all** plugins …
including ones the runner auto-loads without them appearing in `plugins`"),
the runner's Phase 4 §C merge, and `PluginRegistry._ALWAYS_INITIALIZE_PLUGINS`
("`permission` … is wired even when not in profile.plugins"). The gate was the
one dissenter, and the runner docstring claiming the overrides "aren't
currently in the envelope" had been stale since §C shipped.

The loop now applies a config for every plugin **the registry knows**, and two
properties make that safe:

| Property | Why it is load-bearing |
|----------|------------------------|
| names the registry does not know are **skipped, not attempted** | `plugin_configs` also carries the PROVIDER sections (`openrouter`, `anthropic`, …), read by `create_provider` and by no plugin. `expose_tool` raises `ValueError` on each; the old gate filtered them out only as a side effect of them never being in `plugins:` |
| the model's tool surface is **untouched** | bootstrap's `expose_all` already exposed every discovered plugin; what the model sees is filtered per session from `plugins`, which this loop does not write |

Worth knowing, and unchanged in kind: the registry — and so each plugin
INSTANCE — is shared with the parent and its sibling subagents, so a config
applied here applies for all of them. That was already true of every plugin a
subagent *did* list; this widens it to the ones it only configures.

**`permission` is the one plugin that loop does not re-initialize.** The
route above — `registry.expose_tool(name, config)`, a `shutdown()` +
`initialize()` on the registry's instance — is the wrong one for the
enforcer, on every path: see [the next section](#a-subagent-is-judged-by-its-own-profiles-policy-957).
A session's `plugin_configs.permission` block is stashed by `configure()`
and installed as a policy of the session's own instead.

**The validator says the other half.** With the config reaching the plugin,
"unreachable" is no longer the finding; what survives is narrower and still
worth saying — the plugin is configured and **none of its tools are on the
wire**:

| Finding | Severity | Fires when |
|---------|----------|-----------|
| `plugin_config_without_plugin` | warn | `plugin_configs.<X>` for an installed, tool-bearing plugin absent from `plugins:` |

Warn rather than error, for the reason the whole silent-config family (#910,
#925, #947) warns: a base profile in an `inherits` chain may legitimately
carry a config its children enable, and validation runs on every discovered
profile including those bases. Three exemptions keep it from being noise:
a plugin that exposes **no tools at all** (`permission`, `sandbox_manager`) is
configured-only by construction; `introspection`'s tools are core and reach
every wire whatever `plugins:` says; and a plugin whose tools are not
statically knowable (`mcp`) reports none offline and is read the same way —
a false negative, and the right one, since the alternative is asserting a
missing surface the validator cannot see.

### A Subagent Is Judged by Its Own Profile's Policy (#957)

The permission plugin is a registry-shared singleton: a parent and every
subagent it spawns are gated by ONE object, and until #957 by one **policy** —
`_policy`, seeded at bootstrap from the ROOT profile's block. A subagent
profile that declared its own `plugin_configs.permission` was judged by the
parent's policy anyway, so a `documentalista` that whitelisted exactly the two
tools it needed, and worked standalone, was denied `method=default` 14 times
when spawned. The child's profile was byte-identical throughout; the only
edit that changed the verdict was adding `writeNewFile` to the **parent's**
whitelist. The tools that *did* work (`selectReferences`, `retrieve_memories`)
are auto-approved by their own plugins and resolve `whitelist` under any
policy, which is what let the profile look correct until it reached a tool
that was genuinely gated.

The block had two routes, and both were wrong:

| Path | Enforcer | What `registry.expose_tool("permission", block)` did |
|------|----------|------------------------------------------------------|
| daemon (`server/core.py`), runner (`runner/session.py` Step 8) | constructed separately, seeded from the root block | re-initialized the registry's **unread copy**; the enforcer kept the parent's policy — #957 as reported |
| in-process (`jaato_embedded`) | the registry's instance | `shutdown()` + `initialize(child block)` on the enforcer: the parent was now judged by the **child's** policy, and its in-process ASK channel was replaced by a console one — the maintainer's in-tree repro "found the opposite" because it was the same defect from the other side |

**Now: one enforcer, one policy per session.** `PermissionPlugin` keeps
`_policy` as the runtime-wide policy and gains `_scoped_policies`, one
`PermissionPolicy` per session that declared a block, keyed by a
`permission_scope` the session minted at construction. The key travels in the
executor's per-session permission context — the same dict #951 reads the
caller's identity from — so `_resolve_policy(context)` picks the session's own
policy when it installed one and the runtime policy otherwise. Every read and
every session-level mutation in `_check_permission_impl` goes to the resolved
object: a subagent's `always` answer whitelists the tool for the subagent, not
for its parent. The DECISION line names it — `policy=session` /
`policy=runtime` — the distinction the issue could only infer from the verdict
changing when the parent's whitelist was edited.

Where the session installs it decides who gets one:

- **A subagent, at `set_agent_context("subagent", …)`** — the point at which a
  session *becomes* one, called by both spawn paths right after
  `create_session` and before the first turn. `configure()` runs inside
  `create_session`, when the session is still `"main"`, so it only stashes the
  block; a session that is already a subagent when configured (a revive)
  installs from `configure()` itself.
- **Never the root.** The runtime policy IS the root's, seeded from the same
  block at bootstrap on every path, and it is the object the operator's
  `permissions allow|deny|default` commands mutate. A scoped copy would detach
  the root from those commands for no gain.
- **A subagent with no block installs nothing** and is judged by the runtime
  policy, as before — inheriting the parent's posture is the right default for
  a profile that declared none. A block that defines no policy (the
  `agent_name` injection alone) installs nothing either.

Two consequences that were easy to get wrong:

- **Auto-approved tools reach a scoped policy installed later.** `configure()`
  whitelists the lifecycle tools (`signal_completion`) and each plugin's
  `get_auto_approved_tools()` on the runtime policy *before* the session
  becomes a subagent. `add_whitelist_tools` now records every name it is
  handed and `set_scoped_policy` seeds from that set, or a `defaultPolicy:
  deny` child would deny its own completion. These names are plugin-declared
  and identical for every session on the registry; they were never something
  one session's policy could withhold from another.
- **`askPermission` fetches the context itself.** The executor dispatches it
  ungated, so nothing handed it a scope and the model's pre-check answered
  from the runtime policy while the real call was judged by the child's. It
  now reads `JaatoSession.permission_context()` off the current session.

A scoped policy is released by `close_session()`, and every scoped policy is
dropped at `reset_for_next_session()` / `shutdown()`, so a subagent that ends
without closing is bounded by the slot boundary.

**The validator (issue ask 2).** With the child's block governing the child,
"does the spawner's whitelist cover the spawnee's gated tools?" stops being a
question — the cross-profile check the issue floated would now assert a
relationship that no longer holds. Not added.

### What the Authoring Surface Would Not Say

Six findings from one workspace bring-up, each the same shape: the framework
already held the answer and no surface a session reads would state it. The
fix in every case is to make the existing fact reachable, never to invent a
new source of truth.

**`explain agents` and `explain services` — two directories nothing named.**
`.jaato/agents/` (the PERSONA layer) appeared in no `explain` scope, while
`explain profile` listed `system_instructions` — marked DEPRECATED, and the
only instruction-shaped key on the page. An author who never found the agents
directory reached for the deprecated key, which works, so nothing corrected
them. `.jaato/services/` had the same problem with a different fallback: the
`service_connector` caches OpenAPI specs and auth there and calls an API by
alias, and unnamed, the path of least resistance is a raw URL with the base
URL, auth header and pagination re-derived by hand on every call. Both topics
READ their search order from the runtime's own helpers (`agent_search_dirs`,
the schema store's tier constants) rather than restating it — a documented
order that disagrees with the loaded one is worse than none — and both report
what is actually on disk in the workspace. They are the first entries in a
`_WORKSPACE_SCOPES` table, which is what turned `sets`' lone `elif` into the
table this CLI's docstring already asked for.

**A nested parameter is not describable by a signature.**
`explain plugin service_connector` rendered `configure_service_auth(service,
auth)` — accurate, and useless, because `auth` is an object whose five accepted
shapes ARE the tool. `--json` carried the schema all along; the human page
showed neither the shapes nor the `*_env` field each wants, so the only route
to a correct call was the plugin source. The page now expands any parameter
carrying an `enum`, and any `object` parameter's own properties, **one level
deep** — deeper is a schema dump, which is what `--json` is for. The tool's own
description was the other half: *"Credentials are read from environment
variables"* is true and does not say that `bearer` wants `token_env` while
`apiKey` wants `in` + `name` + `value_env`. The model reads that string.

**`PROVIDER_NOTES` — a caveat no contract field could carry.** On Azure
OpenAI, a profile's `model:` is the DEPLOYMENT name from your resource, not a
catalog model id; get it wrong and the failure is a runtime `DeploymentNotFound`
with nothing earlier naming the cause. The provider module's docstring said so
and `explain provider azure_openai` did not. `PROVIDER_NOTES` is a tuple of
prose declared beside `PROVIDER_CAPABILITIES` / `PROVIDER_KNOBS` — co-located so
it cannot drift from the provider it describes — rendered as a `read this
first:` block ABOVE the knob table, because a knob table read under the wrong
premise is still read wrong. Empty for a provider with nothing unusual to say.

**Where the `api_params` check stops.** `validate` checks an `api_params` key
against the PROVIDER's allow-list, which is the only thing it can check: which
values a given MODEL accepts is declared nowhere in this tree and moves whenever
a vendor ships. So `temperature: 0.0` validates clean and is a `400` on a
reasoning model that accepts only its default. Every provider page with an
`api_params` layer now says so, and says the thing that is always true —
omitting a parameter is never the cause of a 400. Deliberately NOT a per-model
incompatibility table: a stale row would reject what the vendor accepts or pass
what it rejects, and either beats silence only by accident. `jaato-scaffold new
profile-set` stopped emitting `temperature: 0.0` live for the same reason; it is
a commented example now, **header included**, because a live `api_params:` over
nothing but comments parses as a YAML null that `validate` correctly reports as
`unknown_knob` — the generator would have failed the file it just wrote.

**Two things a profile says about ITSELF.** `description` is required
(no default on `SubagentProfile`, `(required)` in `explain profile`) and the
loader is lenient, so a missing key becomes `""`. Inheritance does not rescue
it — the merge takes `description=child.description`, so a tier-2 set profile
that omits it OVERRIDES its base's with the empty string — and every profile
`new profile-set` emitted was in that state. What breaks is the one line
`spawn_subagent` advertises to the model, `- worker:  (tools: cli)`: the prose a
delegate is chosen from. `system_instructions` is the second, deprecated and
unflagged. Both are `missing_description` / `deprecated_system_instructions`,
**warn** — either profile loads and runs, and an error would fail existing
workspaces wholesale, the posture `unknown_knob` and `budget_control_absent`
already take.

**A skip that names the module but not the install.** `pexpect` IS declared —
`jaato-server[interactive]`, an extra named after neither the plugin nor the
module — and the registry's skip line said only *"install it to enable this
plugin"*, so the reliable next move was `pip install pexpect` into whatever
interpreter was nearest, and the gap read as an undeclared dependency. The line
now resolves the install target from installed distribution metadata (the same
index `explain <unit> dependencies` reads), so a new extra needs no edit and a
stale one cannot outlive `pyproject.toml`. Best-effort by construction: it runs
inside discovery's error path, and a diagnostic that raises is worse than a
vague one.

### Five Ways a Session Came Up Wrong and Said Nothing

The findings above are about surfaces that would not TELL you something. These
are about a running session that was already broken and reported success. Each
was traced from a live cascade that returned `None`.

**A plugin the profile named that did not load.** `expose_tool` deliberately
refuses to let one broken plugin take the session down: it logs, records the
failure in `PluginRegistry._failed_plugins`, and carries on. That recovery had
no audience — `_failed_plugins` was written in four places and read in **none**
— so the session came up looking healthy with a plugin the profile asked for
simply absent from the model's surface. `get_failed_plugins()` is the read, and
`expose_all` now names, at WARNING, every REQUESTED plugin that failed. A
failure in a plugin the session did **not** request stays quiet: only the
profile's own list is a promise to the author.

**`config_root` was documented as defaulting and did not.** The contract is
written down three times — the SDK parameter's own docstring,
`shared/config_resolver.py`, `explain paths` — and applied in one place: the
in-process client, whose comment names the reason ("config-rooted plugins like
`file_edit` fail to init without a config_root"). The daemon transports left it
`None`, so the same driver got a different session depending on how it
connected, and every scaffolded driver was on the wrong side of that.

The asymmetry that hid it is worth stating, because it is why the failure looks
unrelated to its cause. `config_root` has two consumers and only one falls
back:

| Consumer | With the value unset |
|---|---|
| the config SEARCH PATH (`resolve_config_search_path`) | appends `<workspace>/.jaato` anyway — profiles, agents, schemas all resolve, and the session looks fine |
| a plugin that WRITES under the root | reads the VALUE. `file_edit` puts backups in `<config_root>/sessions/<id>/backups/` and raises at `initialize()` without one |

So the session started, `writeNewFile` was gone, and the only trace was one
daemon-side ERROR. `IPCClient` (and thus `WSClient` / both recovery clients)
now derives `<workspace_path>/.jaato` when a workspace is given. Derived, not
required: an explicit value still wins, so rooting config elsewhere to keep it
out of the agent's filesystem tools works exactly as before, and with no
workspace there is nothing to derive from.

**A completion asset that resolved nowhere.** Nothing checked that a profile's
`completion_payload_schema` or `completion_processors[].script` exists. An
unresolvable one is a WARNING in the runner log and nothing else — and the
consequence is total: with no schema `_should_hide_signal_completion` removes
`signal_completion` from the surface entirely, so the agent hunts for it
through `list_tools`, the framework spends its nudge budget re-prompting, and
the driver gets `None` from a session that looked like it ran. `validate` now
reports `completion_asset_missing` (**error**, matching
`prefetch_script_missing`) and locates paths without loading them, since
importing a processor would execute it.

**...usually because the path carried the prefix the resolver adds.** Every
relative reference is joined onto the config root, so
`.jaato/completion_schemas/x.json` resolves to
`<ws>/.jaato/.jaato/completion_schemas/x.json`. It is an easy mistake — every
other path an author writes is spelled from the workspace root — and it gets
its own code, `redundant_config_root_prefix`, so the message names the fix
instead of sending someone to look on disk for a file that is exactly where
they put it.

**A hidden `signal_completion` that could not be told from an intentional
one.** `_should_hide_signal_completion` reads `_payload_schema is None`, which
is true both when a profile declared no schema (the documented way to opt out)
and when it declared one that failed to resolve (a mistake). `LifecycleTools`
now distinguishes them at construction and logs the second by name — the
runtime backstop for a session that never went through `validate`.

### `configure_service_auth` Configured Auth That Never Reached the Wire

Reported from a live cascade: an `apiKey`/header scheme was configured, the
call returned `env_vars_present: ["GITLAB_TOKEN"]`, and `preview_request`
showed the request going out with no `PRIVATE-TOKEN` header. Every reasonable
auth spelling was tried; the workaround was passing the header by hand on every
`call_service`. Three defects in one chain, and a fourth that only became
reachable once they were fixed:

1. **`preview_request` had the config precedence backwards.** `call_service`
   reads the stored `<service>/_service.yaml` first and falls back to the
   in-memory discovered cache — with a comment explaining why. `preview_request`
   did the opposite. Since `configure_service_auth` writes to disk, the preview
   kept showing the auth the OpenAPI spec was PARSED with: an invented
   `<scheme>_API_KEY` env var, because a spec declares the header name and never
   the credential. The two verbs disagreed about the request, and the one that
   lied was the one an agent uses to check its work.
2. **The in-memory entry was never refreshed**, so that stale config outlived
   the call meant to replace it, for the rest of the session.
3. **`build_request` swallowed the resulting `AuthError`** — "for preview, we
   can skip auth errors" — and returned a request with no auth header, which
   reads as *this endpoint needs none*: the one answer a caller acts on and the
   one that is wrong. A preview still must not raise, so it now carries
   `auth_unresolved` naming the env var that did not resolve.
4. **The credential then reaches a preview that is returned to the MODEL**, and
   `redact_headers` matched a hardcoded four names (`authorization`,
   `x-api-key`, `api-key`, `apikey`). No operator-chosen header is in that list
   and none could be — an `apiKey` scheme's header name belongs to the API.
   Redaction is now by **provenance**: the caller passes the header names the
   auth manager actually resolved a credential into on this request, and the
   name list stays for headers a caller supplied by hand, which have no
   provenance to read.

`explain plugin service_connector` now prints the whole `auth` object (see
above), and the tool's own description names the fields each `type` needs — the
thing a model reads before it guesses.

### The MCP SDK Moved Its Decode Seam (mcp 2.x)

`mcp[cli]` is an unpinned dependency, so a fresh `pip install` resolves
whatever is current — and on any `mcp>=2` install, EVERY MCP server in the
workspace was unreachable. `_ensure_mcp_patch` read
`types.JSONRPCMessage.model_validate_json` to install the filter that keeps a
server's own stdout log lines from being decoded as JSON-RPC; mcp 2.x turned
`JSONRPCMessage` into a PEP 604 `UnionType`, which has no such method, and the
`AttributeError` escaped the MCP thread's `run_until_complete`. The only
evidence was a traceback in that thread's log — printed, incidentally, on top
of every `jaato-scaffold` invocation in the same environment.

The client beneath it was never the problem: driven against a real mcp 2.x
stdio server, `connect` → `list_tools` → `call_tool` all work unchanged. So the
fix is to find the seam, not to pin the SDK back:

| Generation | `JSONRPCMessage` | Decode seam | Silenced by |
|------------|------------------|-------------|-------------|
| mcp 1.x | Pydantic model | `JSONRPCMessage.model_validate_json` | wrapping `traceback` / `builtins.print` (1.x *prints* the failure) |
| mcp 2.x | PEP 604 union | `types.jsonrpc_message_adapter.validate_json` | a logging filter on `mcp.client.stdio`'s own logger |

`detect_jsonrpc_seam(mcp_types)` is module-level and separate from installing
anything, so `jaato-doctor` can ASK the question cold — the `mcp sdk` check
reports the version and the seam in use, and WARNs when a build exposes
neither. Three properties follow from what went wrong:

- **`SkipMessage` is a `ValueError`.** mcp 2.x's `_parse_line` catches exactly
  `ValueError` and hands it to the session as a value; a sentinel outside that
  hierarchy escapes the stdout reader and takes the connection down — the
  opposite of what the filter is for. 1.x catches `Exception` there, so the
  narrower base serves both.
- **No seam is not a failure.** The filter is a convenience and MCP works
  without it, so an unrecognised shape is announced once at WARNING and left
  unfiltered. The install is wrapped too: this method used to be able to kill
  the MCP thread, and no future SDK shape may cost an operator their servers
  again.
- **The 2.x silencer is scoped to one logger object.** A filter installed on
  an ancestor logger is not consulted for records propagating up from a child,
  and every parse failure that is not our sentinel still reaches the operator.

### Secret Env Scrubbing (#863)

The runner legitimately holds secrets in its own `os.environ` — the
provider key, the tokens `web_fetch` expands into headers.  A shell
command, PTY session or MCP server the *model* drives inherits that
environment, so `env` or `echo $GITHUB_TOKEN` hands every credential to
model-controlled code.  `shared/secret_scrub.py` removes a set of secret
names from the environment *given to the subprocess* (never from the
runner's own), at three surfaces: the `cli` subprocess boundary, every
`interactive_shell` spawn, and MCP stdio spawn.

**Scrubbing is on by default.**  Until #863 it was opt-in: the module
defined `DEFAULT_SECRET_ENV_PATTERNS` and applied it nowhere, so a profile
with `cli` or `mcp` and no scrub configuration passed the daemon's full
environment, provider keys included, to every command the model ran.  The
reason was real — `gh`, `git push`, cloud CLIs need their tokens — but the
failure was silent and the default was the unsafe one.  Now the default set
applies when nothing is declared, and opting *out* is the explicit,
WARNING-announced act, like `--ws-unsafe-no-auth`.

One grammar, accepted at the profile top level (`scrub_secret_env:`, which
covers all three surfaces) and per surface
(`plugin_configs.<cli|interactive_shell|mcp>.scrub_secret_env`, which wins
for that surface):

| Value | Meaning |
|-------|---------|
| absent / `default` | the framework set: `*_API_KEY`, `*_APIKEY`, `*_TOKEN`, `*_SECRET`, `*_SECRET_KEY`, `*_PASSWORD`, `*_PASSWD`, `*_ACCESS_KEY`, `*_ACCESS_KEY_ID`, `*_SECRET_ACCESS_KEY`, `*_PRIVATE_KEY`, `*_CREDENTIALS`, `ANTHROPIC_AUTH_TOKEN`, `GH_TOKEN`, `AWS_SESSION_TOKEN` |
| `none` | scrub nothing — the developer-desktop opt-out, logged at WARNING by each surface that applies it. The **only** spelling that disables: `[]`, `""`, a boolean, or a list holding only `!` exemptions are rejected (`invalid_scrub_secret_env`, fail closed), because in this codebase `plugins: []` means "the minimal set", not "off", and an author writing `[]` or `["!GH_TOKEN"]` to mean "nothing beyond the default" must not land on the leaky posture |
| `"*_TOKEN"` | one glob (a lone string is one pattern, never split into characters) |
| `[glob, ...]` | an explicit list; the entry `default` expands to the framework set in place |
| `"!NAME"` in a list | an **exemption**: a variable matching it survives whatever else matches — `[default, '!GH_TOKEN']` keeps `gh` working while the provider key stays out of the shell |

```yaml
# a developer-desktop profile: gh and the AWS CLI keep their credentials,
# everything else in the framework set is still scrubbed
plugins: [cli, interactive_shell, mcp]
scrub_secret_env: [default, "!GH_TOKEN", "!AWS_*"]
plugin_configs:
  mcp:
    scrub_secret_env: default     # the MCP servers get no exemption
```

Rules the implementation holds to:

- **A malformed value fails closed.**  The plugin applies the default set,
  logs an ERROR, and `jaato-scaffold validate` reports
  `invalid_scrub_secret_env`.  The only outcome worse than a broken
  workflow is a silently leaked credential.
- **An explicit grant is not scrubbed.**  A secret in an MCP server's own
  `env` (in `.mcp.json`) or in the `env=` an interactive-shell caller
  passes reaches that process; only the *inherited* `os.environ` is
  filtered.  A profile's `env:` map is **not** a grant — it is where the
  provider key usually lives.
- **The primitives stay policy-free.**  `run_command(scrub_env=None)`,
  `ShellSession(scrub_env=None)` and `ServerConfig.scrub_secret_env=()`
  still mean "scrub nothing"; the default lives in the three plugins
  (`resolve_scrub_patterns`), so it applies however a session was built,
  profile or not.  The profile key is folded into the surfaces the profile
  enables by `inject_scrub_secret_env` at every profile-to-session site
  (runner envelope, in-process root session, both subagent spawn paths),
  beneath an explicit per-surface knob.
- **Inheritance is scalar-override**: a child's value replaces the
  parents' outright, so `none` in one leaf profile does not leak into its
  siblings.  The value is persisted raw in the session snapshot (#787).
- **The leaky posture is announced three times**: `jaato-scaffold
  validate` warns `secret_scrub_disabled` per enabled surface (and
  `scrub_secret_env_inert` when the key names no enabled surface),
  `jaato-doctor` preflight WARNs naming each such profile in the
  workspace, and the plugin logs at WARNING when it applies the opt-out.

**The scrub's one bypass now has a backstop (#712).**  The paragraph above
is precise about what the scrub does *not* cover: the runner's own
`os.environ` stays intact by design, so anything that can read
`/proc/<pid>/environ` reads the unscrubbed store — every provider key and
OAuth token the session holds — and the more the scrub is relied on, the
more that one path matters.  Two layers close it, deliberately with
different reach:

| Layer | Covers | Applies when |
|-------|--------|--------------|
| AppArmor template **v30** — `audit deny` on `/proc/*/{environ,mem,pagemap,auxv,cmdline}` and each `task/<tid>/` twin, in base, `tool_hat`, `//child` and the isolated sub-runner | every read from a confined process, in-process tool or subprocess alike | AppArmor is available and the session is confined |
| `is_sensitive_proc_path` in `shared/plugins/sandbox_utils.py` — the same set plus `maps` / `smaps`, minus `cmdline` | a path handed to a model-driven **file** tool (`readFile`, `glob_files`, `file_edit`) | always, including `workspace_root` unset and the degraded posture of #504 |

The rules are written `/proc/*/...` and never `/proc/self/...`: AppArmor
resolves that symlink to `/proc/<pid>/` **before** matching, so a
`/proc/self/environ` deny would never fire against the read it exists to
stop — the same finding the v15 template note records, and the reason the
old `/proc/self/** r,` grant was inert rather than generous.  The two
layers' sets differ on purpose.  `cmdline` is denied only at the kernel
layer (the `--ws-token TOKEN` exposure; a file-tool read of it is a
narrower leak than the `ps` workflows a denial would break), and
`maps` / `smaps` only at the application layer (they leak address layout
rather than credentials, and the distro's `abstractions/base` may grant
them for the C library's own use — a deny there would override an
abstraction on hosts this change could not be tested against).

Known cost of the `cmdline` deny: `ps` / `top` / `pgrep` run by a confined
agent show empty command columns, and **there is no fragment-level escape
hatch** — a deny beats an allow at any specificity, including one granted
from `~/.jaato/apparmor-fragments/*.rules`.  `JAATO_APPARMOR_COMPLAIN=1`
is the diagnostic route.

Not covered here: the eventual TLS-terminating broker (#505) that keeps a
credential out of the runner environment entirely, and `/proc/*/fd/*`,
which stays readable for any pid because CPython's `close_fds` path
enumerates it at every subprocess spawn and AppArmor has no rule form for
"my own pid only".

### Approver Identity (#859)

`PermissionResolvedEvent` said HOW a decision was reached (`method`) and
nothing about WHO reached it; the authenticated user the daemon knew
(`set_client_user()`) reached only the telemetry `user.id` span attribute,
so an auditor joined approvals against spans by timestamp, and a keyless
deployment had nothing to join against.  Three things now carry identity,
all optional so unauthenticated IPC sessions are unchanged:

| Where | Field | Meaning |
|-------|-------|---------|
| `PermissionResolvedEvent` | `user_id` | the identity the daemon authenticated for the client that answered — stamped by the transport that received the `PermissionResponseRequest` (`get_client_user`), never taken from the request body, and carried to the runner on `PromptResponse.user_id` |
| `PermissionResolvedEvent` | `approver` | the name an external approval system attached to its webhook / file response (`"approver": "..."`); asserted, recorded as claimed |
| ledger `permission-check` record | `user_id` / `approver` | the same two, on the token ledger |
| ledger `response` record | `user_id` | the session's user — the same id the telemetry `user.id` attribute carries |
| session record header | `created_by` | the session's user, persisted (record version 2.9) and restored onto the daemon `Session` and the revived runner session |

Both event fields are `None` for a policy decision (whitelist, evaluator,
suspension), so "nobody was asked" stays distinguishable from "somebody
answered".  The session's own user reaches the runner-side `JaatoSession`
on `SessionInitEnvelope.created_by` — `set_client_user_id` previously had
no caller, so runner-tier telemetry was anonymous too.  Related: #507 is
the integrity half (tamper evidence); this is the identity half.

### Observable Permission Decisions (#951)

`check_permission` traced two things: that a check had **started**, and — on
the ASK branch only — that it was about to prompt. Every terminal decision went
to `_log_decision`, which appends to `_execution_log`, an in-memory list that
reaches no file and no event. So an ALLOW and a DENY were **byte-identical in
every log an operator can read**: one `check_permission: tool=X` line, then
silence.

A silent denial is invisible to the model too — no error to react to, so it
re-issues the same call. #951 reports a subagent whose `writeNewFile` was
checked 38 times and never ran; re-running the same profile with
`defaultPolicy: deny` instead of `ask` produced *the same* logs and the same
~10-token tool result. A genuine policy DENY and a call vanishing after the
gate were indistinguishable, from the operator's side and the model's, so the
symptom could be described exactly and the verdict could not be named. (#947 is
the cost: 127 attempts at ~57k tokens each, outliving the parent session.)

Two stages now say what they did.

**The gate.** `check_permission` is a thin **single-exit wrapper** around
`_check_permission_impl`, which keeps its twenty-odd rule-specific exits:

```
[PERMISSION] check_permission: tool=writeNewFile call_id=call_1 agent=subagent:documentalista session=20260910_115239
[PERMISSION] check_permission: DECISION tool=writeNewFile call_id=call_1 agent=subagent:documentalista allowed=False method=default reason='Denied by default policy'
```

Structural rather than per-branch, deliberately: a branch added later is traced
whether or not its author remembers to, which is the one property per-branch
tracing does not have. `test_decision_observability_951.py` carries an AST
guard that the wrapper stays single-exit, and a second that no exit of the impl
returns a verdict without recording one — the two that did
(`not_initialized`, `unknown`) are closed. A **raise** is traced too: it is a
third outcome the old logging could not express, and `ToolExecutor` converts it
into a fail-closed denial that reads downstream like a policy decision.

`agent=` comes from the **caller's** per-session context, never from
`self._agent_name`: the plugin is a registry-shared singleton whose
`_agent_name` is whatever initialized it last, so a subagent's spawn re-labels
the parent's own later decisions. (#951's traces show exactly that — the
subagent's lines carrying the parent's `@escriba`.) The audit entry gains
`method` / `call_id` / the caller, stamped at the single exit the way approver
identity already was (#859), so `EvalContext.execution_log` can be reasoned
over; the ledger's `permission-check` record gains the same three.

**The stage after it.** `ToolExecutor._execute_impl` has three exits that end a
tool call without running its body, and none left a mark — which is why "the
call disappears between the permission gate and `file_edit`'s executor" had
nowhere to look:

```
[TOOL_RUNNER] permission: tool=writeNewFile call_id=call_1 verdict=DENY method=default reason='Denied by default policy'
[TOOL_RUNNER] resolve: tool=writeNewFile call_id=call_1 executor=shared.plugins.file_edit.plugin._execute_write_new_file
[TOOL_RUNNER] result: tool=writeNewFile call_id=call_1 ok=True dict(keys=_permission,_telemetry,lines,path,size,success)
```

| Line | Says |
|------|------|
| `resolve: ... executor=MISSING — refused before the permission check` | no executor resolvable, so the gate was never consulted (`No executor registered for X` — roughly the same ten tokens a denial is) |
| `permission: ... verdict=DENY\|ALLOW method=...` | the **consumer's** record of the verdict, so a policy plugin that traces nothing of its own (a stub, a wrapper, an out-of-tree engine) is still recorded where its decision takes effect |
| `auto-background: ... — execution leaves this path` | approved, and its body runs elsewhere |
| `resolve: ... executor=MISSING` after an ALLOW | the shape that reads, from outside, as a call vanishing after the gate |
| `result: ... ok=... dict(keys=...)` | what the model was handed — **keys and the error string only**; a trace file is not the place for the file a tool just wrote |

Both stages write to `JAATO_TRACE_LOG` / the profile's `trace.session_log`.

**The line is a contract, and it says whether anybody was asked (#968).**
The trace is the ONE artefact every deployment gets: the ledger's
`permission-check` row needs a ledger, `_execution_log` never leaves the
process, and the event (below) is opt-in. So the DECISION line is
machine-readable by construction — scalar `key=value` fields first, the
free-text `reason=` **last**, read back by
`shared.plugins.permission.plugin.parse_decision_trace`:

```
[PERMISSION] check_permission: DECISION tool=writeNewFile call_id=call_1 \
  agent=subagent:documentalista session=20260910_115239 allowed=True \
  method=allow_all asked=False policy=runtime reason='Pre-approved all requests'
```

`asked=` is the field `method` could not supply, and the one #797 needed:
`allow_all`, `turn_suspension` and `idle_suspension` are each produced BOTH
by a pre-approval short-circuit that consulted nobody and by a human
answering `a` / `t` / `i` at a prompt. It is recorded where the call
actually reaches the channel, so it is a fact rather than an inference.
`user_id=` / `approver=` (#859) are written **only when present** — their
absence is how "nobody was asked" stays distinguishable from "somebody
answered", so they are never rendered as `None`.

**The event carries every decision only if you ask for it.** Most terminal
decisions announce nothing on `PermissionResolvedEvent` — suspensions,
`allow_all`, the trusted bridge, an `askPermission` grant, an evaluator's
early exit, an uninitialized plugin — and in subagent mode *nothing at
all*, which is #951's blind spot. `plugin_configs.permission.emit_decision_events:
true` makes `check_permission`'s single exit emit one for any decision no
branch announced (never a second for one that did):

```yaml
plugin_configs:
  permission:
    emit_decision_events: true      # default false
    policy: {defaultPolicy: ask}
```

Off by default because it is per-tool-call cost on the hot path, and
measured rather than assumed: the record's own bookkeeping is **~0.7 µs**
per decision against a ~36 µs baseline the trace write dominates, while the
fallback emission adds **~9 µs** with a *no-op* hook — and the daemon's real
hook emits two events (`PermissionResolvedEvent` plus the
`PermissionStatusEvent` `emit_permission_status()` appends) and serialises
both to every connected client. A trace line is cheap; an event is not.

### A Boundary the Notebook Did Not Have (#710)

`cli` contains the paths a model names: every path token in a command goes
through `check_path_with_jaato_containment` and the command is refused when one
falls outside the workspace. `notebook_execute` had nothing equivalent, and it
is the cheapest surface a model can reach (eager, core, and measured at ~93% of
tool calls in #717). Same daemon, same workspace, two sessions:

| tool | asked for | result |
|---|---|---|
| `cli_based_tool` | `cat /etc/hostname` | **refused** |
| `notebook_execute` | `open('/etc/hostname').read()` | the hostname |
| `notebook_execute` | `subprocess.run(['cat','/etc/hostname'])` | the hostname |

The third row is the point: the notebook spawned the very command `cli` had
just refused. Any workspace-confinement claim made about `cli` was void while
`notebook` was exposed, and nothing had to be jailbroken to get there — the
reported session was asked to study a sibling checkout and simply browsed it.

**`cwd=workspace_root` was never containment**, and the backend's own docstring
said as much: it exists so `os.getcwd()` and *relative* paths resolve
in-workspace. Absolute paths were untouched, and `_set_pdeathsig` is lifecycle,
not isolation.

**The in-process gate is a different question, and it stays.**
`_inprocess_exec_allowed` (`backends/local.py`) asks *may model-authored code
run in the host interpreter at all* — where a cell can reach the runtime, the
tool executor and other plugins' memory — and fails closed without AppArmor or
`allow_inprocess_exec`. It bounds the PROCESS. It never bounded the
FILESYSTEM, and it is on the `local` backend while `subprocess` is the default,
so `CLAUDE.md` describing notebook execution as failing closed described one
backend of it. Both halves exist now, and neither replaces the other.

**Three tiers, strongest first, decided in the kernel before any cell arrives:**

| Tier | What bounds the cell |
|------|----------------------|
| AppArmor | the runner's per-session profile, inherited by the kernel through the profile's `ix` exec rule. A real kernel boundary; nothing else is installed on this path |
| audit hook | `kernel_sandbox` installs a PEP 578 hook applying the same containment `cli` applies — to `open`, the `os.*` mutators, `listdir`/`scandir`, every spawn, and `ctypes.dlopen` |
| nothing | the kernel answers every cell with a refusal naming the knob, and stays up so the refusal reads as a cell error rather than a dead kernel |

The hook sees **actual I/O**, so it does not care how a path was spelled,
built or obfuscated — the standing weakness of the static `code_analyzer`
scan, which reads source text and (in the default `warn` mode) blocks nothing.
Spawns are checked on their arguments, because a child process runs in another
interpreter where no hook of ours is installed; `ctypes.dlopen` of anything
outside the interpreter installation is refused for the same reason, one layer
down — native code's own `open(2)` raises no audit event.

**Be clear about what an in-process hook is.** It is not a kernel boundary. A
cell that *sets out* to escape can attack the hook itself — rebind the module's
policy, reach it through `gc`. It closes the accidental and the opportunistic
case, which is what #710 measured, and makes a deliberate escape an act that
has to defeat a declared boundary rather than walk through an open door. For a
hostile-model threat model the answer is AppArmor, and the layering says so.

**The gate belongs to notebook execution, not to one backend.** Every backend
answers `NotebookBackend.execution_boundary()`, `NotebookPlugin` asks before
dispatching a cell, and the base class's default is a **refusal** — a backend
added later is contained once its author has decided how, rather than
inheriting "allowed" and reproducing this defect. `local` answers with its
in-process gate, `subprocess` with the tiers above, `kaggle` with "the code
never touches this host".

**And it is asked on both dispatch paths.** `supports_streaming` is `True` for
`notebook_execute`, so the streaming path is the live one in a daemon — and it
ran neither the boundary check nor the static analyzer. `sandbox_mode: strict`
therefore blocked nothing in a real session. One `_guard_execution` now serves
both paths; a gate that covers one of two dispatch paths is the same shape as a
gate that covers one of two backends.

**Escape hatches, narrow before wide:**

```yaml
plugin_configs:
  notebook:
    allow_read_paths: ["/srv/corpus"]   # extra readable roots
    allow_uncontained_exec: false       # the whole boundary, off (WARNING)
```

`sandbox add <path>` is the operator's live equivalent and is honoured too: a
kernel outlives many cells, so the session's authorized and denied paths ride
**every** `execute` frame rather than the kernel's argv — a grant made after
the kernel spawned takes effect on the next cell, and a revocation likewise. A
denial outranks every allowance, the workspace included.

Not addressed here: an AppArmor child profile for the kernel on an unconfined
host (there is no profile to transition from), and the in-process backend's
memory-reach, which #710's grooming correctly separates and which the deferred
kernel + tool-RPC redesign owns.

### A Cancel the Daemon Wrote and the Runner Threw Away (#988)

`client.stop()` / `session.request_stop()` reach a runner-served tool as a
`kind: "cancel"` frame on the daemon ↔ runner socket. The daemon wrote it, the
runner decoded it, and under load the tool ran to completion and answered
`ok=True` — measured from a plain `asyncio.run()` driver with no pytest
involved: token tripped at 0.201 s, frame written at 0.206 s with no
exception, result at 10.971 s.

**The frame was not lost on the wire; it lost a race inside the runner.**
`serve()` reads frames on one thread and dispatches each request to a pool
worker, and the worker's FIRST act was to register the call:

```python
self._pool.submit(self._handle_request, env)   # serve, reader thread
...
def _handle_request(self, env):                # pool worker
    token = CancelToken()
    self._active_calls[env.id] = _ActiveCall(cancel_token=token)
```

so `serve` went straight back to reading while the worker was still being
scheduled. A cancel frame already sitting in the socket buffer — which is the
normal case, because a runner takes ~0.5 s to boot and both frames are queued
before it reads either — reached `_handle_cancel` first, found no
`_active_calls[id]`, logged at DEBUG and returned. Nothing re-checked. The
token the worker then created was a **different object**, never tripped, and
the `cli` executor polled it ~200 times without seeing anything.

| Layer | Verdict |
|-------|---------|
| the runner never reads the frame | **no** — `_handle_cancel` ran, every time |
| `_handle_cancel` finds no `_active_calls[id]` | **yes** — 12/12 losses under load, 0/10 idle |
| the executor does not observe a tripped token | **no** — the token was never tripped |

**Registration moved to the reader thread** (`_register_call`, called from
`serve` before the next frame is decoded; the token is then handed to
`_handle_request`, which publishes that same object to its thread-local). The
daemon only cancels an id whose request frame it has already written, and the
reader handles frames in wire order, so by the time a cancel for id *N* is
read the entry for *N* is either present (trip it) or gone (already finished).
*Not registered yet* is no longer reachable. This is shared RPC machinery, so
the **session-hosted path was affected identically** — `session.send_message`
takes the same work-lane branch — and is fixed by the same change.

**Why the loss was invisible, and what says it now.** Two swallows on one
path made a lost cancel indistinguishable from "no cancel was requested".
Both are promoted, and each miss is told apart from the others rather than
lumped together, because one of them is routine and the other never is:

| Site | Event | Now |
|------|-------|-----|
| `RunnerRPCClient._send_cancel` | the write FAILED — the runner is alive and never heard | **WARNING**, counted in `cancel_write_failures()` |
| same | the channel is already closed — the call dies with it | DEBUG |
| `RunnerRPC._handle_cancel` | id ≤ the highest registered — the call finished first | DEBUG, counted `late` |
| same | id > the highest registered — a cancel for a call this runner was never asked to run (#856's signature) | **WARNING**, counted `unknown` |

`RunnerRPC.cancel_stats()` reports `received / tripped / late / unknown`, and
`received == tripped + late + unknown` by construction — so a nonzero
`unknown` is the one number that says *a cancel reached this runner and
nothing was cancelled*.

**The test that hid it.** `test_cancel_token_trips_runner_cancel` is a timing
bet, and a previous attempt to stabilise it widened the command from 200 to
2000 iterations on the theory that the event loop was starved. It was not, and
the extra iterations bought nothing: a dropped cancel is dropped
*permanently* — the frame is consumed and no later cancel-check can recover it
— measured at 2000 iterations under the same load, still `ok=True`, at 105.1 s
instead of 10.9 s. The count is back at 200, and
`server/runner/tests/test_cancel_before_worker_registers_988.py` reproduces
the loaded case **with no clock at all**: one work-lane worker, held by a
gated call, so the cancelled call is provably still queued; wire ordering is
established by a control-lane probe rather than by polling. It fails
`unknown=1, tripped=0` against the old registration site, deterministically.

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

#### Path Containment (#722, absorbing the `cwd` half of #503)

`cli` inspects the paths in a command and refuses those outside the
workspace. `interactive_shell` — which spawns a real PTY and is strictly more
capable — refused **nothing**: its only spatial notion was `cwd`, which sets
where relative paths start and says nothing about absolute ones. So a session
that could not `cat /etc/hostname` through `cli` could `shell_spawn` a shell
and read anything the daemon user could. Whichever tool enforces containment,
the model routes around it through one that does not.

Three layers now answer, and they are deliberately **not** equal:

| Layer | What it covers | On a string it cannot parse |
|-------|----------------|-----------------------------|
| `shell_spawn`'s `command` | the workspace check `cli` applies, on the same analyzer | **refuses** — this string IS a shell command |
| `shell_input`'s `text` | the same check, best effort | **allows** — typed text is whatever the running program reads (Python, SQL, a password), so a shell-grammar failure is the normal case, not an evasion |
| the AppArmor child profile | everything, kernel-enforced | n/a |

Only the third is a boundary in the strict sense, which is why the string
layers are stated as what they are rather than sold as one: a live PTY's
working directory drifts under `cd`, and the analyzer does not descend into
the quoted argument of `sh -c` (a blind spot inherited from `cli`, whose
analyzer this is). The check refuses the direct attempt, in a vocabulary the
model can act on, and claims nothing more.

**The enforcement point is one module, not two.** `cli`'s classification and
workspace test moved to `shared/plugins/command_containment.py`
(`classify_command_paths`, `path_within_workspace`, `first_denied_path`); `cli`
keeps its result-shaping (it refuses by mimicking "No such file or directory")
and delegates the analysis. Fail-closed is the caller's decision and
`first_denied_path` makes each caller state it — `on_parse_error="deny"` for a
command, `"allow"` for typed input.

**Unconfined is announced, not inherited.** When the runner installed no
AppArmor child-profile transition, the plugin logs **once per session at
WARNING** that the kernel boundary is absent — the posture `scrub_secret_env:
none` and `--ws-unsafe-no-auth` already take. `require_confinement: true`
refuses every spawn instead, the shape `notebook`'s in-process-exec gate has.
The default is `false` because a PTY child is a separate process, the same risk
class as a `cli` subprocess, which does not fail closed either.

```yaml
plugin_configs:
  interactive_shell:
    require_confinement: true     # default false
```

**The spawn `cwd` is verified where the process starts** (the remainder of
#503): `ShellSession` takes the workspace root beside the `cwd` and refuses one
that resolves outside it, both sides canonicalised so a symlinked `cwd` is
judged by its target. The plugin passes its own workspace root, so the check is
a tautology on today's only caller — which is the point: the invariant holds at
the seam that spawns, rather than being a property of one call site a later
caller could drop.

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
    },
    "gitlab": {
      "path": "/webhook/gitlab",
      "secret_header": "X-Gitlab-Token",
      "secret_algo": "token",
      "event_type_header": "X-Gitlab-Event"
    }
  }
}
```

**Route auth: two modes, and they are not peers (#930).** `secret_algo` names
how the route's shared secret is checked:

| Mode | Header carries | Property |
|------|----------------|----------|
| `hmac-sha256` | an HMAC digest over the request **body** | the secret never travels; a captured request cannot be replayed against another payload |
| `token` | the shared secret **verbatim**, compared with `hmac.compare_digest` | **weaker**: readable by anything that terminates TLS, and replayable against any payload |

`token` exists because a large class of producers signs nothing — GitLab sends
the configured secret in `X-Gitlab-Token` and expects an equality check. Before
it, the *only* configuration that ingested such a webhook was
`allow_unauthenticated: true`: a secret sitting in the request, thrown away, on
the flag whose whole purpose is to be unreachable by omission.

Three properties keep the wider vocabulary from becoming a softer posture:

- **A pair, still fail-closed.** `secret_header` without `secret_algo` (or the
  reverse) is a 500, and an `secret_algo` outside `SECRET_ALGOS` is a hard
  config-validation error *and* a 500 at request time — never a downgrade to
  unsigned. Widening the vocabulary widens what `secret_algo` may **say**, never
  what it may omit.
- **No cross-mode leniency.** `token` strips no `sha256=` prefix and reads no
  body; a valid HMAC digest does not authenticate a `token` route, and the
  reverse. Either transformation would accept a secret nobody configured.
- **Announced, not silent.** Every `token` route logs a WARNING at listener
  startup naming route and header, and a louder one when TLS is off (the secret
  is then sent in the clear). Same posture as `--ws-unsafe-no-auth` and
  `scrub_secret_env: none` — so `token` cannot become the quiet path of least
  resistance for a producer that *does* sign bodies. Pair it with TLS.

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

### OpenAI (native — Chat Completions and the Responses API)
| Variable | Purpose |
|----------|---------|
| `JAATO_OPENAI_API_KEY` | API key (jaato namespace, highest priority) |
| `OPENAI_API_KEY` | API key (the vendor's own documented variable; honored so a machine already set up for the OpenAI SDK works with no extra config) |
| `JAATO_OPENAI_BASE_URL` / `OPENAI_BASE_URL` | Endpoint (default: `https://api.openai.com/v1`) |
| `JAATO_OPENAI_MODEL` | Default model name (e.g. `gpt-5.1`) |
| `JAATO_OPENAI_CONTEXT_LENGTH` | Context window (**required in practice** — see below) |
| `JAATO_OPENAI_ORG_ID` / `OPENAI_ORG_ID` | `OpenAI-Organization` header (billing attribution) |
| `JAATO_OPENAI_PROJECT_ID` / `OPENAI_PROJECT_ID` | `OpenAI-Project` header; a project-scoped key needs it |
| `JAATO_OPENAI_API` | Wire selector: `chat` (default) or `responses` |

**Authentication (in priority order):**
1. `plugin_configs.openai.api_key` (may carry a `pass://` / `vault://` URI)
2. `JAATO_OPENAI_API_KEY`, then the vendor's `OPENAI_API_KEY`
3. The stored `openai_auth.json` (`config_root` → `<workspace>/.jaato/` → `~/.jaato/`)

**Two wires, one plugin.**  jaato already reached OpenAI's models through
OpenRouter and through nine OpenAI-*compatible* gateways.  What it had no
way to speak was the endpoint OpenAI itself serves — and, in particular,
the **Responses API**, which no compatible gateway offers and which is
where OpenAI ships first.  `plugin_configs.openai.api` selects:

| `api:` | Transport | Shape |
|--------|-----------|-------|
| `chat` (default) | the shared `_openai_compat` streaming loop | `messages`, choice deltas, `prompt_tokens`/`completion_tokens` |
| `responses` | `openai/responses.py` | a flat `input` item list, top-level `function_call` items keyed by `call_id`, typed SSE events, `input_tokens`/`output_tokens` |

The two disagree about the name of the most-used knob
(`max_output_tokens` vs `max_tokens`), so the `api_params` allow-list is
**per wire**: a chat-only key on the Responses wire is dropped with a
warning rather than forwarded into a 400.  The Responses transport streams
text and reasoning deltas for the UX but builds the parts that become
history from the `response` object on the terminal event — the API's own
account of what it produced.  Every request is sent `store: false` with the
full `input`: jaato owns the history and its GC decides what the model
sees, so a server-side thread keyed by `previous_response_id` would
silently diverge from it (`api_params.store: true` opts back in).

**You must set a context window.**  OpenAI's `GET /v1/models` serves bare
entries — `{id, object, created, owned_by}`, no capacity for any model — so
`plugin_configs.openai.context_length` (or `JAATO_OPENAI_CONTEXT_LENGTH`)
is required and `connect()` fails loud without it.  No per-model table is
hardcoded: GC sizes the whole history against that number, and a stale
table would truncate a session or over-fill a request without saying so.
Input *modalities* are a different question and do carry a table (a wrong
entry withholds an attachment, visibly; it does not corrupt the history),
with `plugin_configs.openai.modalities` above it.

```yaml
# profile example: the Responses wire, with reasoning
provider: openai
model: gpt-5.1
plugin_configs:
  openai:
    api: responses
    context_length: 400000      # required — the catalog reports none
    project: proj_abc123        # a project-scoped key needs its header
    api_params:
      reasoning: {effort: high, summary: auto}
      max_output_tokens: 16384
```

`pip install 'jaato-server[openai]'`.

**Smoke test.**  `examples/provider_smoke_openai_azure.py` drives this
provider on both wires, and the Azure one, against a local stand-in —
no account, no key, no spend — and prints the requests that reached it,
so deployment routing and the `api-version` query string are visible as
facts.  `--live-openai` / `--live-azure` run the *same* assertions
against the real endpoints.  A green mock run means the framework is not
the problem; only a live run proves the vendor agrees.

### Azure OpenAI
| Variable | Purpose |
|----------|---------|
| `JAATO_AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_ENDPOINT` | Resource URL, `https://<resource>.openai.azure.com` (**required**) |
| `JAATO_AZURE_OPENAI_API_VERSION` / `AZURE_OPENAI_API_VERSION` | The `api-version` date, e.g. `2024-10-21` (**required**) |
| `JAATO_AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_API_KEY` | Resource key (not needed under Entra auth) |
| `JAATO_AZURE_OPENAI_DEPLOYMENT` / `AZURE_OPENAI_DEPLOYMENT` | Default deployment name when a profile names none |
| `JAATO_AZURE_OPENAI_CONTEXT_LENGTH` | Context window (**required in practice** — see below) |
| `JAATO_AZURE_OPENAI_AUTH` | Credential kind: `key` (default) or `aad` |

Azure serves OpenAI's models on the same request and response shapes, so
the transport is the shared `_openai_compat` one unchanged.  Everything
*around* the request is Azure's, and each piece is load-bearing:

- **Deployment-name routing.**  `model:` carries the DEPLOYMENT name your
  subscription chose, not a model id; the SDK's `AzureOpenAI` client turns
  it into `/openai/deployments/<name>/chat/completions`.  The same name can
  be repointed at a different model version without changing, which is also
  why no per-model context table could be right here.
- **A pinned `api-version`.**  Required, with no default: the date decides
  which request fields exist, so a framework-chosen default would silently
  decide what your deployment accepts and change under you when it moved.
- **Key or Microsoft Entra ID.**  `auth: aad` mints a bearer token per
  request from whatever identity the host already has (managed identity,
  workload identity, `az login`) and stores no secret at all — which is why
  "no key found" is not the end of the credential search for this provider.
  Needs `pip install 'jaato-server[azure-openai]'` for `azure-identity`.
- **Wire extensions are not assumed.**  PDFs (`file` blocks) and audio
  (`input_audio`) are gated on Azure by api-version *and* by what the
  deployment points at, so this provider declares `pdf_input=False` /
  `audio_input=False` and uses the shared images-only converter.  The
  native `openai` provider declares both because its one endpoint carries
  them unconditionally; here it would be a guess about someone's resource.

```yaml
# profile example: Entra ID auth, no secret anywhere
provider: azure_openai
model: prod-gpt4o            # the DEPLOYMENT name, not "gpt-4o"
plugin_configs:
  azure_openai:
    endpoint: https://my-resource.openai.azure.com
    api_version: "2024-10-21"
    auth: aad
    context_length: 128000   # required — Azure reports no capacity
    model_name: gpt-4o       # what the deployment serves, so vision is detected
```

Check the wiring before spending anything:
`python examples/provider_smoke_openai_azure.py` (mock, then
`--live-azure --deployment <name>` for the real resource).

### AWS Bedrock
| Variable | Purpose |
|----------|---------|
| `JAATO_BEDROCK_REGION` | AWS region (**required** — via this, `AWS_REGION`, `AWS_DEFAULT_REGION`, the named profile, or the `region` knob) |
| `AWS_REGION` | The vendor's documented region variable. Read by jaato because Python's botocore maps `region` to `AWS_DEFAULT_REGION` **alone**, so a host configured the documented way resolves nothing through boto3 |
| `JAATO_BEDROCK_MODEL` | Default model id / inference-profile id |
| `JAATO_BEDROCK_CONTEXT_LENGTH` | Context window (**required in practice** — see below) |
| `JAATO_BEDROCK_PROFILE` | Named AWS profile for this session (the per-session override of boto3's `AWS_PROFILE`) |
| `JAATO_BEDROCK_ENDPOINT_URL` | `bedrock-runtime` endpoint override (a VPC endpoint) |

**Authentication: jaato resolves nothing.**  Every other provider in this
tree resolves an API key.  SigV4 signing belongs to botocore, and so does
the chain behind it, so `initialize()` builds a `boto3.Session` and reads
`get_credentials()` — env vars, a named profile, IAM Identity Center, an
ECS/EKS task role, an EC2 instance role, in boto3's own order.  There is no
`api_key` knob and no interactive login (AWS's own is `aws sso login`, run
outside the process).  The practical consequence: **a confined runner on EC2
or EKS needs no provider credential in its environment at all** — which also
means `scrub_secret_env` has nothing of Bedrock's to strip.

Bedrock is one endpoint over many vendors, so the wire is Amazon's rather
than OpenAI's, and three things follow from what its catalog does *not*
report:

- **You must set a context window.**  `ListFoundationModels` describes
  modalities and streaming support, never capacity, so
  `plugin_configs.bedrock.context_length` (or the env var) is required and
  `connect()` fails loud without it — as with the native OpenAI and Azure
  providers.  A guessed window truncates a session without saying so.
- **Input modalities come from a documented per-family table**, not from
  detection.  The catalog's vocabulary is `TEXT | IMAGE | EMBEDDING`, which
  cannot express the `document` and `audio` blocks Converse plainly carries;
  an incomplete source wearing the authoritative *detect* tier would withhold
  every PDF from a model that reads PDFs.  `plugin_configs.bedrock.modalities`
  is above the table for a model it does not name.  Cross-region inference
  profiles (`us.` / `eu.` / `apac.` prefixes) are matched with the prefix
  stripped: it routes the request and says nothing about the model.
- **Every media block names a format from a CLOSED vocabulary** — `png`,
  `jpeg`, `gif`, `webp` for images; `pdf`, `csv`, `doc`, `docx`, `xls`,
  `xlsx`, `html`, `txt`, `md` for documents; a fixed audio list; a fixed
  video list.  A mime outside a vocabulary is **withheld with a note**, never
  relabelled into one that is inside it (that is #829).  Raw PCM is accepted
  only when its parameters agree with Bedrock's `pcm` (s16le); `audio/L16` is
  refused outright, because RFC 2586 makes it big-endian.

**Extended thinking is extraction-only by default, and deliberately.**
Converse's `reasoningContent` block carries the upstream's `signature` beside
the text, and the Anthropic-family models require that block back — signature
included — on the next request of a tool-call loop.  `Part.thought` has
nowhere to put a signature, so replaying the text alone would be rejected by
the very models that produced it.  The provider therefore reads reasoning out
(it reaches the UI and `ProviderResponse.thinking`) and does not write it
back; `api_params.enable_thinking` exists and **WARNS at connect** that a
thinking turn which also calls tools may be refused.  Wiring it properly
needs a signature-carrying part — the reasoning-replay seam's next step,
which is why `reasoning_replay` is declared `False` rather than fudged.

```yaml
# profile example: Claude on Bedrock, cross-region, cached, no secret anywhere
provider: bedrock
model: us.anthropic.claude-sonnet-4-5-20250929-v1:0   # an inference profile
plugin_configs:
  bedrock:
    region: us-east-1
    context_length: 200000        # required — the catalog reports none
    enable_caching: true          # Converse cachePoint blocks
    cache_ttl: "1h"               # 5m | 1h; anything else is dropped
    api_params:
      max_tokens: 8192
      temperature: 0.0
```

`pip install 'jaato-server[bedrock]'`.

> **Not covered here:** the `InvokeModel` wire (Converse supersedes it for
> every text model), Bedrock Agents / Knowledge Bases (a different service
> shape), and the batch inference API (submit → poll → collect, the same
> follow-up shape Doubleword's batch tier is).

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
| top-level (cache) | `enable_caching`, `cache_ttl`, `cache_history`, `cache_exclude_recent_turns`, `cache_min_tokens` | Prompt-cache control, consumed by the `cache_anthropic` plugin (explicit `cache_control` breakpoints on system / tools / history). These are not Messages-API body fields, so they sit at top level rather than under `api_params`. `enable_caching` unset falls back to `JAATO_ANTHROPIC_ENABLE_CACHING` (default off). **Server 0.7.0+**: before the `_wire_cache_plugin` fix these keys were silently ignored — the plugin was built from an always-empty config. See [Model Tiers × Prompt Caching](docs/design/model-tier-prompt-cache.md) §4. |
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
      tool_choice: "required"      # auto|required|none, or a dict naming
                                   # one tool.  Forwarded since the
                                   # provider gained the parameter its
                                   # own contract declared; dropped on a
                                   # turn that sends no tools, and a
                                   # name-bearing choice is mapped to the
                                   # hashed wire id.
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
| `api_params` | `temperature`, `top_p`, `top_k`, `max_tokens`, `models`, `service_tier`, `enable_thinking`, `thinking_budget`, `thinking_level`, `cache_prompt`, `cache_ttl`, `strict_tools`, `tool_choice` | OpenAI Chat Completions body fields. `models` is OpenRouter's request-level cross-model fallback list (sibling of `model`; OpenRouter walks candidates on failure). `service_tier` (`auto` / `default` / `flex` / `priority` / `scale`) is the OpenAI-style processing-tier selector, forwarded to tier-supporting upstreams (OpenAI, Gemini, ...) per [service tiers](https://openrouter.ai/docs/guides/features/service-tiers) — `flex` trades latency for ~50% off, `priority` the reverse; the response reports the tier actually used. `thinking_*` keys mirror Anthropic / Antigravity; when both `thinking_level` and `thinking_budget` are set, `level` wins (more portable across upstreams). `cache_prompt: "auto"` (default) places `cache_control: {type: ephemeral}` breakpoints on the system block and last tool definition for explicit-cache upstreams (Anthropic, Gemini 1.5+/2.5+/3+); other upstreams (OpenAI, DeepSeek, Grok) cache automatically and need no client annotation. Response-side parsing of `prompt_tokens_details.cached_tokens` / `cache_creation_input_tokens` / `cost` is unconditional. `tool_choice` (`auto` / `required` / `none`, or a dict naming one tool) selects whether the model MAY, MUST or MUST NOT call a tool this turn. OpenRouter declared `tool_choice_forwarding=False` and had no such parameter at all — the implementation lived on `_openai_compat`, which it does not inherit, the same gap that made model audio unreachable through it. It is dropped when the turn sends no tools (OpenAI rejects it without `tools`) and a name-bearing choice is mapped through `tool_choice_to_wire`, because tool names are hashed on the wire. **`required` is not free on a speaking tier**: measured against `openai/gpt-audio-mini`, it does force a single-generation native call and remove the completion nudge, and it produces zero audio — the model satisfies the constraint by calling the tool and saying nothing. `strict_tools: true` (server 0.6.118+) emits `"strict": true` as a sibling of `parameters` in each tool definition; OpenRouter forwards to supported upstreams (Sonnet 4.5 / Opus 4.1+, GPT-4o+, Gemini, OSS, Fireworks per [structured outputs list](https://openrouter.ai/docs/guides/features/structured-outputs)) for grammar-constrained tool-arg sampling. Required for cascade-determinism use cases (see `feedback_cascade_completion_schemas_require_strict_model_support` memory); the framework does NOT auto-rewrite schemas to satisfy OpenAI's strict-mode requirements (kb authors own schema shape — `additionalProperties: false` on every object, exhaustive `required` arrays, no `oneOf`/`anyOf` if you enable strict). |
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

### MiniMax
| Variable | Purpose |
|----------|---------|
| `JAATO_MINIMAX_API_KEY` | API key (jaato namespace, highest priority) — an API key from https://platform.minimax.io or a Token Plan subscription key (`sk-cp-...`), which works on `/v1` unchanged |
| `MINIMAX_API_KEY` | API key (the vendor's own variable; honoured beneath the jaato one) |
| `JAATO_MINIMAX_BASE_URL` | Endpoint (default: `https://api.minimax.io/v1`; China: `https://api.minimax.cn/v1` — keys are region-bound, a China key is `401` on `.io`) |
| `JAATO_MINIMAX_MODEL` | Default model name (e.g. `MiniMax-M3`, `MiniMax-M2.7`) |
| `JAATO_MINIMAX_CONTEXT_LENGTH` | Override the built-in per-model window |

**Authentication (in priority order):** `JAATO_MINIMAX_API_KEY`, then `MINIMAX_API_KEY`, then `minimax-auth key <key>` (validated against `GET /v1/models`, stored as `minimax_auth.json`).

| Model | Context | Thinking | Input |
|-------|---------|----------|-------|
| `MiniMax-M3` | 1,000,000 | `thinking: {type: adaptive\|disabled}` via `api_params.enable_thinking`; default on | text + image |
| `MiniMax-M2.7`, `-highspeed` (+ legacy M2 / M2.1 / M2.5) | 204,800 | always on (`enable_thinking: false` is logged and ignored) | text |

The catalog is bare, so the window comes from a built-in table beneath
`plugin_configs.minimax.context_length`. `reasoning_split: true` is sent on
every call so reasoning never arrives as `<think>` inside the text (a block
that still leaks is moved to the reasoning channel); replay echoes both
`reasoning_content` and `reasoning_details`. `tool_choice` accepts `none` /
`auto` (anything else is folded to `auto` with a warning). `max_completion_tokens`
is always sent — the vendor default truncates tool-call JSON — defaulting to
the model's recommended cap (M3 131072, M2.x 65536). `presence_penalty` /
`frequency_penalty` are unsupported and `response_format` is silently ignored
upstream for M2.x, so neither is forwarded. `api_params.service_tier: priority`
is 1.5× price. Error code `2056` (Token Plan 5-hour window, names the reset
time) and `1008` (balance) are non-transient quota errors; `1026` / `1027` are
the content filter.

```yaml
provider: minimax
model: MiniMax-M3
plugin_configs:
  minimax:
    api_params:
      enable_thinking: true      # M3 only
      max_tokens: 131072         # sent as max_completion_tokens
```

### Moonshot AI Kimi
| Variable | Purpose |
|----------|---------|
| `JAATO_KIMI_API_KEY` | API key (jaato namespace, highest priority) from https://platform.kimi.ai/console/api-keys (K3 unlocks after a first top-up) |
| `MOONSHOT_API_KEY` | API key (the vendor's own variable; honoured beneath the jaato one) |
| `JAATO_KIMI_BASE_URL` | Endpoint (default: `https://api.moonshot.ai/v1`; China `https://api.moonshot.cn/v1`; Kimi Code plan `https://api.kimi.com/coding/v1` with the plan's own ids `k3`, `kimi-for-coding`) |
| `JAATO_KIMI_MODEL` | Default model name (e.g. `kimi-k3`) |
| `JAATO_KIMI_CONTEXT_LENGTH` | Manual override when the catalog lacks the model |

**Authentication (in priority order):** `JAATO_KIMI_API_KEY`, then `MOONSHOT_API_KEY`, then `kimi-auth key <key>` (validated against `GET /v1/users/me/balance`, stored as `kimi_auth.json`).

`GET /v1/models` reports `context_length`, `supports_image_in` and
`supports_reasoning` per model, so the window, input modalities and thinking
support are **catalog-detected at connect** (catalog → knob → fail-loud, no
table). Every id before `kimi-k2.6` was retired on 2026-08-31 and answers
`404`.

| Model | Context | Thinking control | `tool_choice` |
|-------|---------|------------------|---------------|
| `kimi-k3` | 1,048,576 | always on; `api_params.thinking_level: low\|high\|max` → `reasoning_effort` (changing it mid-session restarts the prefix cache) | full set |
| `kimi-k2.7-code`, `-highspeed` | 262,144 | forced `{type: enabled, keep: all}` | `auto` / `none` |
| `kimi-k2.6` | 262,144 | `enable_thinking` → `thinking.type`; `thinking_keep: all` → `thinking.keep` | `auto` / `none` |

**Sampling parameters are a 400 on this wire** (`temperature`, `top_p`, `n`,
penalties are fixed per model), so the forwarded allow-list is exactly what
the request schema names: `max_tokens` (→ `max_completion_tokens`),
`tool_choice`, `stop`, `response_format`, `prompt_cache_key`. Tool
definitions are stamped `strict: false` (Kimi defaults strict **on** and the
framework's schemas are not authored for it; `api_params.strict_tools: true`
opts in). Cache hits are read from top-level `usage.cached_tokens`. `429` is
split by `error.type`: `engine_overloaded_error` and `rate_limit_reached_error`
back off, `exceeded_current_quota_error` (balance) stops the retry loop.

```yaml
provider: kimi
model: kimi-k3
plugin_configs:
  kimi:
    api_params:
      thinking_level: high
      prompt_cache_key: my-task-42   # required on the Kimi Code plan
```

### Xiaomi MiMo
| Variable | Purpose |
|----------|---------|
| `JAATO_MIMO_API_KEY` | API key (jaato namespace, highest priority) from https://platform.xiaomimimo.com/#/console/api-keys |
| `MIMO_API_KEY` | API key (the vendor's own variable; honoured beneath the jaato one) |
| `JAATO_MIMO_BASE_URL` | Endpoint (default: `https://api.xiaomimimo.com/v1`; Token Plan keys `tp-...` only work against `https://token-plan-{cn,sgp,ams}.xiaomimimo.com/v1`) |
| `JAATO_MIMO_MODEL` | Default model name (`mimo-v2.5-pro` or `mimo-v2.5`) |
| `JAATO_MIMO_CONTEXT_LENGTH` | Override the built-in per-model window |

**Authentication (in priority order):** `JAATO_MIMO_API_KEY`, then `MIMO_API_KEY`, then `mimo-auth key <key>` (validated against `GET /v1/models`, stored as `mimo_auth.json`). Not available in the EU, the UK or Korea (`403`).

| Model | Context | Input | Thinking |
|-------|---------|-------|----------|
| `mimo-v2.5-pro` | 1,048,576 | text | `api_params.enable_thinking` → `thinking: {type: enabled\|disabled}`; default on |
| `mimo-v2.5` | 1,048,576 | text + image (the model also takes video and audio; neither wire is probed yet) | same |

The V2 series was deprecated on 2026-06-30 and is not in the table. **In
thinking mode the vendor refuses (`400`) the next request of a tool loop
unless the assistant message carries its `reasoning_content`** — the case
the reasoning-replay seam exists for. While thinking is on the vendor forces
`temperature=1.0` / `top_p=0.95` whatever is sent. `tool_choice` accepts
`auto` only. `finish_reason: repetition_truncation` maps to `MAX_TOKENS`.
`402` (balance), `403` (region / key) and `421` (content filter) are
non-transient.

```yaml
provider: mimo
model: mimo-v2.5
plugin_configs:
  mimo:
    api_params:
      enable_thinking: false     # the vendor's own advice for tool-heavy work
      response_format: {type: json_object}
```

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

### Substitution in Config Values (two vocabularies, two resolution times)

A value an author writes into a profile may carry placeholders. There are
**two** sets, they use different syntax, and the distinction that matters is
**when** each is resolved — a token resolved daemon-side cannot carry a value
that does not exist until a subagent thread writes a line.

| Syntax | Resolved by | When | Honoured in |
|--------|-------------|------|-------------|
| `${workspaceRoot}` `${cwd}` `${jdtlsStateRoot}` `${HOME}` `${USER}` `${ANY_ENV_VAR}`, plus `pass://` / `vault://` secret URIs | `expand_variables` (daemon) | once, while the profile resolves | `env:`, `plugin_configs:`, `trace:`, and the plugins that expand (lsp, webhook, web_fetch, service_connector, references) |
| `{agent}` `{agent_suffix}` | `jaato_sdk.trace` (the writer) | per line written | `trace.session_log` / `trace.provider_log` and their env vars |

The two are told apart by the `$`: `${agent}` is an env var nobody sets,
`{agent}` is the per-agent token. `jaato-scaffold explain profile` and
`explain env` render both tables from the live registries
(`EXPANSION_CONTEXT_VARS`, `TRACE_PATH_PLACEHOLDERS`), so a token added to
either appears in the docs without anyone writing it down.

**The `trace:` block used to be the one typed route that expanded nothing.**
A profile's `env:` map has always been run through `expand_variables`, while
`TraceProfileConfig.as_env()` was applied verbatim — so `${HOME}/t.log` in
the *validated* block was not absolute, was joined onto the workspace, and
`trace_write`'s `os.makedirs` created a directory literally named `${HOME}`
inside it. That is the #775 shape inside the block written to stop #775, and
the asymmetry is the defect: the route that outranks the other must not
understand less than it.

**A provider trace splits per agent whether or not you ask.** With no
placeholder the agent id is appended before the extension
(`provider.jsonl` → `provider_subagent_1.jsonl`) — unchanged, and what every
existing deployment gets. Naming a placeholder puts the id where you want it
instead (`logs/{agent}/provider.jsonl`), suppresses the implicit suffix, and
is the **only** way to split a *session* trace, which never splits on its own.
For the main agent the two forms differ deliberately: implicit leaves the path
alone, explicit renders `{agent}` as `main`, because an author who asked for
the id wants it on every file rather than one anonymous file among named
siblings.

**What is refused, where.** A path-valued knob given a *switch* is refused at
profile **load** on both routes — the typed block always did, and the `env:`
map (the spelling that actually caused #775, since it was the only route that
existed then) now does too, for `JAATO_TRACE_LOG`, `JAATO_PROVIDER_TRACE` and
`JAATO_SESSION_LOG_DIR`. Closing one and leaving the other open made the block
a suggestion: an author who hit the refusal satisfied it by moving the same
value one key over. The workspace `.env` is deliberately **not** covered — it
is the operator's own file, lower precedence, and outside the validated
surface. An unknown `{token}` is refused too, because unresolved it becomes a
literal directory.

`jaato-scaffold validate` reports the cases that load fine and are still
probably wrong:

| Finding | Severity | Fires when |
|---------|----------|-----------|
| `trace_path_placeholder_unknown` | error | a `{token}` nothing substitutes, reaching via the `env:` route (the `trace:` route is refused at load) |
| `trace_path_daemon_scoped_var` | warn | `${workspaceRoot}` / `${cwd}` in a trace path — on the main-session path these expand to the **daemon's** workspace, so every session shares one file. A relative path is the per-session idiom |
| `trace_path_unexpanded_var` | warn | a `${VAR}` nothing in the workspace defines — an undefined name stays **literal**, so the path gains a `${VAR}` directory |
| `trace_env_shadowed` | warn | both routes set for one variable: the block wins, so the `env:` value is dead |

Before this, `validate.py` contained the string `trace` **zero** times — the
one profile block whose silent-ignore failures had no reporter, the family
#910 / #925 / #947 / #950 each closed for a different knob.

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
| `JAATO_GC_MEDIA_BYTES` | Binary payload a history may carry before GC triggers, in **bytes** (default 8 MiB; `0` disables). The second GC denominator: a voice session can sit far below its token threshold while carrying megabytes of audio, which is the state GC could not see at all before #850. Typed sibling: `gc.media_bytes_threshold`. |
| `JAATO_PARALLEL_TOOLS` | Enable parallel tool execution (default: `true`) |
| `JAATO_DEFERRED_TOOLS` | Enable deferred tool loading (default: `true`) |
| `JAATO_RUNNER_POOL_ENABLED` | Enable pre-warm runner pool routing (default: `true`).  Sessions consume pre-warm pool slots instead of cold-spawning a runner subprocess.  Set to `false` / `0` / `no` / `off` to disable.  See `docs/design/runner_prewarm_pool_plan.md`. |
| `JAATO_RUNNER_POOL_SIZE` | Number of **unreserved** pre-warm pool slots to keep idle (default: 2) — slots any arriving session may take.  Raise for cascades that fan out stages **concurrently** (each simultaneous stage needs its own warm slot).  Sequential/back-to-back stages do NOT need a larger pool — they reuse one warm slot via the `slot.settled` handoff (the next stage is spawned on slot-availability), so pool size >1 only helps parallel fan-out.  Cascade-affined idle slots (reservations) are **not** counted here (#898): they are capacity for one tenant only, and counting them as pool capacity starved everybody else. |
| `JAATO_RUNNER_POOL_MAX_SIZE` | Hard ceiling on **total** idle pool slots, reservations included (default: `2 x JAATO_RUNNER_POOL_SIZE`).  This is the memory bound — a slot is 129–187 MB — on the growth that per-tenant reservations imply.  Raise it when `pool_replenish_ceiling_blocked_total` is nonzero: some tenant is being served by cold-spawn while reservations hold the ceiling. |
| `JAATO_IPC_EVENT_QUEUE_MAX` | Per-client IPC event-queue bound (default 2048). Beyond it, lossy tool-output chunks are evicted oldest-first (media before text); essential lifecycle events are queued past the bound rather than dropped, because losing one desynchronises the client permanently. A non-numeric or non-positive value falls back to the default — "unbounded" is the bug this exists to fix. See [Binary Media Chunks](docs/design/binary-media-chunks.md). |
| `JAATO_AMBIGUOUS_WIDTH` | Width for East Asian Ambiguous chars in tables (`1` default, `2` for CJK terminals) |
| `JAATO_SESSION_LOG_DIR` | Per-session log directory, relative to workspace (default: `.jaato/logs`) |
| `JAATO_CGROUPS_ROOT` | Parent cgroup v2 directory for the WS server's per-session cgroup tree (default: `/sys/fs/cgroup/jaato`). Override when the host has subtree_control delegated under a different path. Must already exist with `memory`, `pids`, `cpu` in `cgroup.subtree_control`. |
| `JAATO_REQUIRE_APPARMOR` | Require kernel-enforced AppArmor confinement (`1`/`true`/`yes`). Promotes the WS server's auto-detect mode to *required*: if confinement is unavailable the server refuses to start instead of silently degrading to directory-sandbox-only isolation. Equivalent to the WS `--apparmor` flag; combining it with `--no-apparmor` is a contradiction the server rejects at startup. When unset (auto), unavailability is logged at WARNING with the specific failing precondition and the server degrades. |
| `JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC` | Opt into in-process execution of model-authored notebook cells (`1`/`true`/`yes`). The `notebook` plugin's `local` backend runs cells via `exec`/`eval` in the host interpreter, so by default it **fails closed** unless a kernel-enforced AppArmor profile is active (the production confined-runner path). Set this (or notebook plugin config `allow_inprocess_exec: true`) to accept in-process execution on unconfined hosts (e.g. trusted single-user dev). Logs a one-time WARNING when execution runs unconfined via this opt-in. Bounds the PROCESS, not the filesystem — see the row below and [A Boundary the Notebook Did Not Have](#a-boundary-the-notebook-did-not-have-710). |
| `JAATO_NOTEBOOK_ALLOW_UNCONTAINED_EXEC` | Opt into notebook cells reaching **outside the workspace** (`1`/`true`/`yes`; profile key `plugin_configs.notebook.allow_uncontained_exec`). Cells are otherwise contained to the workspace and `/tmp` — the same boundary `cli` applies — by an AppArmor profile where one is enforced and by the kernel's own audit hook otherwise (#710). Announced at WARNING on every kernel that takes it. `plugin_configs.notebook.allow_read_paths` and the operator's `sandbox add` are the narrow alternatives. |
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
minimax-auth login/key/logout/status   # MiniMax API key
kimi-auth login/key/logout/status      # Moonshot AI Kimi API key
mimo-auth login/key/logout/status      # Xiaomi MiMo API key
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

Default keybindings: `submit`=enter, `cancel`=c-c, `exit`=c-d, `toggle_plan`=c-p, `toggle_tools`=c-t, `toggle_thinking`=c-r, `open_editor`=c-g, `search`=c-f

**Reasoning blocks** (`toggle_thinking`, Ctrl+R by default): a model's
reasoning reaches the TUI as its own output source (`thinking`) ahead of the
answer. It renders **collapsed by default** as one summary line
(`▸ Internal thinking (14 lines, 412 words)  ───  Ctrl+R to expand`), the
sibling of the collapsed tool tree behind Ctrl+T; the toggle expands every
reasoning block in the active buffer into the bordered `Internal thinking`
box. The session bar shows a `Reasoning: ▶ collapsed [Ctrl+R]` indicator once
the buffer holds any. Note that most sessions will show **no** reasoning at
all: providers discard it unless the profile asks for it
(`plugin_configs.<provider>.api_params.enable_thinking: true`), because
reasoning costs output tokens. Reasoning is streamed like text — the first
delta is a `write`, the rest `append` — so it lands in one block; before #755
every delta was a `write` and each rendered on its own line.

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

### The Telemetry Block a Profile Wrote and Nothing Read (#858)

`plugin_configs.telemetry` is a documented surface — `env_scope.py` named a
typed key for all five `JAATO_TELEMETRY_*` vars, and `OTelPlugin.initialize()`
reads every one of them. The factory in between assembled its own config dict
from the environment and passed **that**:

```python
redact = os.environ.get("JAATO_TELEMETRY_REDACT_CONTENT", "true") ...
plugin.initialize({"enabled": True, "exporter": exporter,
                   "redact_content": redact})
```

So the whole block was inert. Not a missing key — a **missing argument**, and
the same silent-ignore family as #910 / #925 / #947 / #950.

`redact_content` is why this was P1 rather than a tidy-up: it is a **privacy**
control. An operator who sets it in the typed place and reads `explain profile`
has every reason to believe prompt and response content is withheld from the
collector, and it is being exported. That the env default happens to be the
safe one does not make the ignore correct — setting it explicitly to `false`
was ignored just as completely, and nothing, at any severity, said so.

**The whole block, not one key.** `exporter` had the identical defect and
`file_path` / `backend` / `enabled` were each one line away from it, so
`create_plugin(config)` carries the block through to `initialize()` and layers
the env vars **beneath** it with `setdefault`. A key the profile did not supply
is left ABSENT rather than filled in, because `initialize()` already resolves
`config.get(key, os.environ[...])` per key — forcing an env-derived value in
would make the profile's silence outrank the env var it is the default for.

| Key | Reaches | Precedence beneath it |
|-----|---------|----------------------|
| `enabled` | the gate that returns `NullTelemetryPlugin` | `JAATO_TELEMETRY_ENABLED` |
| `backend` | `otel` vs `langfuse` | `JAATO_TELEMETRY_BACKEND`, then Langfuse auto-detect |
| `exporter` | `initialize()` | `JAATO_TELEMETRY_EXPORTER` |
| `redact_content` | `initialize()` | `JAATO_TELEMETRY_REDACT_CONTENT` (default **true**) |
| `file_path`, `endpoint`, `headers`, `protocol`, `service_name`, `instance_id`, `sample_rate`, `batch_export`, `public_key`, `secret_key`, `host` | `initialize()` unchanged | each key's own env fallback inside `initialize()` |

**Backend selection is not widened by accident.** A profile carrying no
`telemetry` block selects exactly the backend it always did. The Langfuse
auto-detect gains the block's own `public_key` / `endpoint` as signals under the
*same* rule it applies to `LANGFUSE_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_ENDPOINT`,
because the profile outranks the environment everywhere else and a Langfuse
setup expressed purely in a profile would otherwise be served by the generic
backend.

**Where it is plumbed.** Telemetry is **runtime**-scoped — one plugin per
`JaatoRuntime`, shared by the main session and every in-process subagent, built
in `JaatoRuntime.__init__` before any session exists. It is not a registry
plugin, so `JaatoSession._apply_plugin_configs` (#950) cannot serve it and the
block has to arrive at construction: `JaatoRuntime(telemetry_config=...)`, fed
from `SessionInitEnvelope.plugin_configs` (runner-served sessions, the default
path), from `JaatoServer._profile.plugin_configs` (the daemon's in-process
runtime), and from `JaatoClient.set_telemetry_config()` before `connect()` (the
embedded path, whose runtime factory is a documented test seam that could not
take a fourth argument).

**The safe default is unchanged.** With no block and no env vars: telemetry
off; switched on by env alone: `exporter=otlp`, `redact_content=True`. A
misspelled boolean resolves to the default rather than to `False`, so a typo
cannot quietly disable redaction. All five vars leave the `AWAITING_TYPED_KEY`
ratchet for a resolving `typed_key`.

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

### Comparison and Design Docs Are Checked Against the Tree

`docs/compare-*.md` and the multimodal design doc carry claims an evaluator
quotes — the licence, whether AppArmor is free, which wires carry images —
and both comparison docs had drifted once before (#866). The guard is
`jaato-server/shared/tests/test_docs_do_not_contradict_the_tree.py`, in the
required `contract-guards` job:

- no line a comparison doc attributes to jaato may call it MIT or open source
  (the identifier is read from `jaato-server/pyproject.toml`; a competitor's
  licence on its own line or in its own column is fine);
- no comparison-doc line may call AppArmor premium while `server/apparmor.py`
  ships in the free package;
- the "Where jaato is now" table in `docs/design/multimodal-model-support.md`
  must name exactly the providers whose `PROVIDER_CAPABILITIES` declare the
  row's capability, so adding `pdf_input` to a provider means updating that
  row.

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
- [The Self-Bounding Completion Gate](docs/design/completion-gate.md) - What `completion_processors` is for and the seven rules a working one had to get right, each attached to the incident that produced it. Covers `max_refusals:` / `on_exhausted:` (the gate's own refusal ceiling, distinct from `max_turns`, which is and remains the retry budget), the `faults[]` channel that keeps an unfixable environment fault from burning the retry budget, why a broken gate must never read as a passing one, and the load-once-per-session caching the counter used to depend on as folklore. Start from `jaato-scaffold explain completion` and `jaato-scaffold new processor` — both are computed from the framework, so they cannot drift the way the prose can. §9 covers why the gate is three files rather than one: `jaato-scaffold new sweep` emits the checks (`acceptance.sh`, shared with the post-hoc graders), the processor, and the profile's `completion_processors:` + `completion_payload_schema:` as ONE set (`--no-gate` opts out), because a profile carrying processors and no schema has no lenient gate — `_should_hide_signal_completion` removes `signal_completion` entirely, so the agent cannot signal and the gate never runs. §11 covers why a session that completed is still drivable: `signal_completion` ends the TURN, and the continuation it skips was also the only writer of that batch's results into history, so a completed conversation used to end on a `tool_calls` block nothing answered and every later request — `send_message` and `session.wake` alike — was rejected by the provider (#913). `_record_terminal_tool_results` writes them without the round-trip, which is what makes "complete every turn to enforce a contract, then keep talking" usable.
- [Payload-Schema Conventions](docs/design/payload-schema-conventions.md) - Symmetric authoring guide for `spawn_payload_schema` (input boundary) and `completion_payload_schema` (output boundary) — symmetric in everything but the type system: a completion payload is JSON the model emitted, a spawn payload crosses the IPC wire as `key=value` argv tokens, so **every spawn property is a `string`** (`pattern` carries the shape, the consumer parses). #883 ratified that rather than reopening the transport, and both spawn boundaries now validate the same string view — the in-process `spawn_subagent` call used to accept a typed value the wire could never deliver, so one profile meant two things. A refusal caused by the schema names the profile; `jaato-scaffold validate` catches it before any spawn as `spawn_schema_type_unreachable`. Mirror prefetch required-keys; always carry `warnings[]` / `errors[]` escape hatches; persona ↔ schema consistency check; canonical-hash strip rules; `agent_params` interaction with agent-continuity (§6).
- [Competitor Memory Systems](docs/design/competitor-memory-systems.md) - Survey of nine agent-memory products, sorted by what a *framework* owes: pattern (nothing) / seam (an extension point) / fidelity (a fix) / not ours. Records which items were already expressible as cascade patterns, which memory hot paths are not pluggable, and why the pattern corpus needs `certify/`-style contract tests run against `main`.
- [Agent Continuity Pattern](docs/design/agent-continuity.md) - `{{continuity_scope}}` + memory plugin enrichment + raw/curated lifecycle: persona-level continuity across sessions composed from existing primitives, no new framework code. Reference impl in `jaato-knowledge-manager/.jaato.example/`.
- [Model Tiers × Prompt Caching](docs/design/model-tier-prompt-cache.md) - What `enter_tier` costs when prompt caching is on: cache is keyed per model, so an in-place tier switch re-reads the whole prefix cold (break-even ~6 consecutive calls at the new tier). Covers the `_wire_cache_plugin` gap that made profile cache knobs inert, the system-block tier line that invalidates BP1, and the per-provider knob divergence + proposed common `cache:` field.
- [MiniMax, Kimi and MiMo providers](docs/design/minimax-kimi-mimo-providers.md) - Design for three first-party OpenAI-compatible providers (`minimax`, `kimi`, `mimo`) and the framework prerequisite they share: **reasoning replay** — sending an assistant turn's `reasoning_content` back on the next request of a tool-call loop, which the session currently drops from history and every OpenAI-shaped converter ignores. Covers the surface decision (chat completions, not the Anthropic shims), per-vendor thinking-control dialects, tool-choice vocabularies, catalog vs table context resolution, error taxonomies, and the registration checklist.
- [AppArmor Setup](docs/apparmor-setup.md) - Kernel-enforced workspace isolation. WS deployments confine automatically when AppArmor is available; IPC clients opt in via `IPCClient(..., apparmor=True)` (defaults to `False`).
- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
