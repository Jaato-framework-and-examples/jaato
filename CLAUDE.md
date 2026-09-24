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

Or with [uv](https://docs.astral.sh/uv/) — `uv venv` creates `.venv` and
`uv pip` resolves it, so the `.venv/bin/` prefix is not needed:
```bash
uv venv
uv pip install -e jaato-sdk/. -e "jaato-server/.[all]" -e "jaato-tui/.[all]"
```

### Running the Server (Multi-Client Mode)
```bash
# Start server as daemon with IPC socket
.venv/bin/python -m jaato_server --ipc-socket /tmp/jaato.sock --daemon

# Start server with both IPC and WebSocket
.venv/bin/python -m jaato_server --ipc-socket /tmp/jaato.sock --web-socket :8080 --daemon

# Check server status
.venv/bin/python -m jaato_server --status

# Stop server
.venv/bin/python -m jaato_server --stop

# Connect TUI client to running server
.venv/bin/python jaato-tui/rich_client.py --connect /tmp/jaato.sock
```

### Running Tests
```bash
.venv/bin/pytest                                        # All tests
.venv/bin/pytest jaato-server/jaato_server/shared/tests/             # Core tests
.venv/bin/pytest jaato-server/jaato_server/shared/plugins/cli/tests/ # Plugin tests
.venv/bin/pytest -v                                     # Verbose output
```

Test organization:
- Core tests: `jaato-server/jaato_server/shared/tests/`
- Plugin tests: `jaato-server/jaato_server/shared/plugins/<plugin>/tests/`
- Provider tests: `jaato-server/jaato_server/shared/plugins/model_provider/<provider>/tests/`

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed diagrams and component interactions.

### Server Components (`jaato-server/jaato_server/server/`)

The framework uses a server-first architecture where the server runs as a daemon and clients connect via IPC or WebSocket.

- **`jaato_server/server/__main__.py`**: Entry point with daemon mode, PID management
  - `--ipc-socket PATH`: Unix domain socket for local clients
  - `--web-socket [HOST:]PORT`: WebSocket for remote clients
  - `--socket-mode MODE`: Octal file permissions for the IPC socket (default: `660`, owner and group only). Pass `666` to opt into world-accessible (e.g. cross-user containers on a trusted host). The socket's mode is still the only thing deciding WHO may connect; what the daemon does with a connection it accepted is bounded by the peer check below.
  - `--ipc-trust-peer-paths`: opt out of that check (also `JAATO_IPC_TRUST_PEER_PATHS=1`). Announced at WARNING the first time it takes effect.
  - `--ws-token TOKEN` / `--ws-token-file PATH`: bearer token clients must present in the WS Upgrade. Token-file mode 0600 enforced. When neither flag is passed (and `--web-socket` is set), the daemon reads `~/.jaato/ws.token`; if the file doesn't exist, it generates a 32-byte token and persists it there with mode 0600. Local clients can read the same default path for zero-config auth. **Prefer `--ws-token-file`, or neither flag.** A token passed as `--ws-token TOKEN` sits in the daemon's `argv` and is therefore served by `/proc/<daemon_pid>/cmdline` to anything on the host that can read it. AppArmor template v30 denies that read from inside a confined session (#712), but the exposure to everything else on the box is a property of the flag, not of the profile.
  - `--ws-unsafe-no-auth`: explicit opt-out of WS bearer auth (legacy open-accept). Logs a startup WARNING. Required to keep the historical behaviour.
  - `--ws-app-credentials PATH`: opt into per-user **connect tickets** (#1074). A JSON object mapping an application id to that application's long-lived credential, mode 0600 enforced. Each entry authorises one WS connection to call `ticket.bind` / `ticket.revoke` and **nothing else** — it cannot open a session. Omit the flag and WS auth is byte-identical to what it has always been. See [Identity at Connect](#identity-at-connect-1074).
  - `--daemon`: Run as background process
  - `--status`/`--stop`: Server management

  **WS auth contract:** clients send `Authorization: Bearer <token>` on the Upgrade request (Python/curl/proxies) or pass `?token=<token>` as a query parameter (browsers, which can't set custom headers from `new WebSocket()`). The server stores only the SHA-256 digest and compares with `hmac.compare_digest`. Auth runs after connection-interceptors but before any session work, so a bad token is closed with WS code 1008 immediately. The `set_client_user()` hook for jaato-premium SSO is unchanged — premium can still attach an identity after the bearer check passes. Since #1074 the same check also resolves an **application credential** and a **user ticket**, in that order, from the same presented value; with neither configured it is the one digest comparison it always was.

- **`jaato_server/server/core.py`**: `JaatoServer` - UI-agnostic core logic
  - Wraps `JaatoClient` with event emission instead of callbacks
  - Handles permission requests, tool execution, streaming

- **`jaato_server/server/events.py`**: Event protocol (25+ typed events)
  - Server→Client: `AgentOutputEvent`, `PermissionRequestedEvent`, `PlanUpdatedEvent`, etc.
  - Client→Server: `SendMessageRequest`, `PermissionResponseRequest`, `StopRequest`, etc.

- **`jaato_server/server/session_manager.py`**: Multi-session orchestration with disk persistence
- **`jaato_server/server/ipc.py`**: Unix domain socket server (length-prefixed framing)
- **`jaato_server/server/websocket.py`**: WebSocket server for remote clients

### Core Components (`jaato-server/jaato_server/shared/`)

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

- **cdp.py**: Minimal Chrome DevTools Protocol client — browser launch with
  race-free port discovery, attach-to-running-browser, thread-safe
  request/response plus an event pump. Provider-neutral (raises
  `CDPConnectionError`), no new dependencies: the transport reuses the core
  `websockets` package, discovery is stdlib. The `chrome_ai` provider is its
  first consumer, not its owner — it lived under that provider until the
  [WebMCP assessment](docs/design/webmcp.md) found that reaching a browser
  required importing from a model provider.

### Plugin System (`jaato-server/jaato_server/shared/plugins/`)

Four plugin types:

**Tool Plugins** - Provide tools the model can invoke (`PLUGIN_KIND = "tool"`, implements `ToolPlugin`):
- `PluginRegistry`: Discovers and manages tool plugins
- `cli/`: Shell commands | `mcp/`: MCP servers | `permission/`: Permission control
- `interactive_shell/`: Interactive PTY sessions (REPLs, password prompts, wizards, debuggers)
- `courier/`: peer-to-peer messaging between SESSIONS — `send_to_session` / `list_group_sessions` (any-to-any within a group, waking a cold peer) and the relocated `send_to_sibling` / `list_siblings` (cascade-only, never wake). Cross-tier (`daemon_callable`); see [Session Group Messaging](#session-group-messaging-the-courier-plugin)
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
declares it, like `budget_control.limits`, and unlike the rest
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
`jaato_server/server/runner/tool_executor.ToolExecutor`, the **Phase-2, cli-only**
`execute_fn`. `RunnerRPC._dispatch_method` uses that object only as a fallback
and routes to `host.session._executor` whenever a session host exists, which
is on every path that dispatches `session.bootstrap` — all of them. A
session's tools run through `jaato_server.shared.ai_tool_runner.ToolExecutor`, whose
`set_runtime_limits` had **no non-test caller**, so `CliPlugin._runtime_limits`
was `None` everywhere. The issue's own table marked cold-spawn ✅ for
forwarding the values; it forwarded them to an executor the session does not
use.

**One vehicle, one application point.**

| Seat | What it does |
|------|--------------|
| `build_session_envelope` + the isolated sub-runner builder | stamp the whole resolved block onto `SessionInitEnvelope.runtime_limits` (**envelope v7**), outside any spawn branch, so pool-served and cold-spawned sessions carry the same thing |
| `jaato_server/server/runner/session.py` | `_runtime_limits_from_envelope` re-parses it; a block this runner cannot parse degrades to "nobody declared limits" with a WARNING rather than refusing the bootstrap |
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
(`jaato_server.server._profile.runtime_limits`), so they cannot disagree about a number;
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
dashboard. `jaato_server/shared/app_identity.py` separates the two: `AppIdentity` is the
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
from jaato_server.shared.app_identity import AppIdentity
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

**Profile schema** (same as `SubagentProfile` in `jaato_server/shared/plugins/subagent/config.py`):
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
#   max_session_seconds / max_orphan_seconds (#812) are the two WALL-CLOCK
#   bounds, enforced daemon-side by the session-lifetime watchdog so they
#   still apply when the client that created the session has died.  Both
#   inherit most-restrictive-wins; 0 = explicitly unbounded.
#   max_orphan_seconds is one of two fields here with a framework default
#   (900s).  The other is unload_grace_seconds (#1106, 60s): how long the
#   daemon keeps an unwatched session LOADED before unloading it, so a
#   browser reload or a network blip costs nothing instead of a full
#   teardown + respawn.  It inherits most-restrictive-wins too, but 0 is
#   its TIGHTEST value ("no grace", the pre-#1106 behaviour) where 0 on the
#   two bounds means "unbounded".
runtime_limits:
  pids_max: 64
  max_parallel_tools: 2
  max_orphan_seconds: 300
  unload_grace_seconds: 60
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
#   again.  THE RETRY BUDGET IS `budget_control` (a degrade rung
#   whose action is abort); there is no second attempts knob.
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
- `session.orphans` — list LOADED sessions with no client attached
  (→ `SessionListEvent`; see [A Session Nobody Was Watching](#a-session-nobody-was-watching-812))
- `session.stop <id>` — stop ANY loaded session by id, not just the caller's own
- `session.reload_env [id]` — re-resolve a LIVE session's `.env` and credentials and rebuild its provider (see [A Credential Stored After the Runner Booted](#a-credential-stored-after-the-runner-booted))
- `workspace.ignore <path>` — toggle one exact entry in the caller's workspace `.gitignore` (→ `WorkspaceIgnoreResultEvent`; protocol 1.12, see [A Key the Web Files Panel Did Not Have](#a-key-the-web-files-panel-did-not-have))
- `scaffold.explain [topic] [name]` — render one `jaato-scaffold explain` topic **on the daemon**, so a CLI whose own virtualenv lacks the extension contributing it can still be told (→ `ScaffoldExplainEvent`; protocol 1.18, see [A Topic the CLI Could Not Answer and the Daemon Could](#a-topic-the-cli-could-not-answer-and-the-daemon-could))
- `workspace.delete` (a `WorkspaceDeleteRequest`, WS only) — delete a workspace the caller may see: its directory, its sessions, its registry row (→ `WorkspaceDeletedEvent`; protocol 1.13, see [A Workspace Everyone Could See](#a-workspace-everyone-could-see))
- `workspace.file.fetch` (a `WorkspaceFileFetchRequest`, WS only) — download one file from the caller's workspace (→ `WorkspaceFileContentEvent` + one binary frame; protocol 1.20, see [A File That Could Go In and Not Come Out](#a-file-that-could-go-in-and-not-come-out))

**Flow:** Client sends `session.new --profile researcher` → server discovers profiles from `.jaato/profiles/` → resolves `SubagentProfile` → `JaatoServer` applies profile overrides (model, provider, plugins, plugin_configs, GC) during `initialize()`.

### The Completion-Nudge Budget (#919, #934)

A session whose surface carries `signal_completion` is expected to call it
before its loop ends. When the loop settles without that call the framework
re-prompts the model — a **nudge** — and re-enters the loop; the budget bounds
how many times, before it gives up and emits `NudgeExhausted`.

That budget was a function-local `MAX_COMPLETION_NUDGES = 2` in **three**
files — `jaato_server/server/core.py` (the daemon's top-level guard),
`jaato_embedded/client.py` (the in-process lead) and
`jaato_server/shared/plugins/subagent/plugin.py` (the subagent loop). Nothing kept the three
equal, and none was reachable from a profile — which made it the one bound in
the completion path a deployment could not express:

| bound | configurable? |
|---|---|
| completion-processor refusals | `max_refusals` + `on_exhausted`, per processor |
| turns a session may take | `budget_control.limits.turns` + an `abort` rung |
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
- **Inheritance**: child overrides outright, else the minimum across the
  parents that declared one.
- **One definition.** `jaato_server/shared/completion_nudge.py` owns
  `DEFAULT_MAX_COMPLETION_NUDGES` and the resolver every site now calls, so the
  three paths cannot drift again. A profile predating the field — an older
  session snapshot, or no profile at all — resolves to the default rather than
  raising, so an unconfigured deployment behaves exactly as before.
  `jaato_eval.sign_off` restates the number deliberately (the eval engine must
  not import `jaato_server.server.*` / `jaato_server.shared.*`); that copy is a reporting ceiling, and is
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

### A Tier Binds (provider, model), and Half of It Did Not Take

Reported as *"two tiers of the same profile do not share history; each
tier keeps its own."* The premise is false, and saying so is what locates
the real defects: a session has ONE `_history`, `switch_tier` never
touches it, and every request is built from `_history_for_provider()`,
whose only per-tier filter withholds **binary** content. Driven end to
end, the tier entered second is handed the first tier's turns verbatim —
`test_a_tier_binds_provider_and_window.py` measures that on the wire
rather than leaving it as folklore.

What a tier *does* bind is a **pair**, and the session applied one half of
it in two places.

**The initial tier's provider was dropped.** `configure()` overrode
`self._model_name` from the initial tier and left
`self._provider_name_override` at the profile's top-level `provider:` —
`None` when the profile declares none, which is the normal shape when
every tier declares its own. So turn 0 ran the initial tier's MODEL on
somebody else's PROVIDER (the runtime default, or a top-level value that
disagrees), and the binding the profile declared did not take effect
until the first `enter_tier`.

The second consequence is the one the report is about.
`_active_provider_name` — what `_connect_tier_entry` compares
`entry.provider` against to decide whether to SWAP — was that same wrong
value, and `_ensure_provider` seeded the per-provider cache only when the
override was non-`None`. Entering a tier that named the provider the
session was **already running** therefore compared unequal and built a
SECOND instance of it:

| Provider | What a duplicate instance costs |
|----------|--------------------------------|
| any stateless one (all but one) | a wasted handshake |
| `claude_cli` | **a second conversation.** It sends `messages[-1]` and nothing else, leaving the transcript to the CLI's own `--resume` session (`_cli_session_id`), so the new instance starts empty and accumulates only the turns taken while its tier holds the wheel |

That last row is the one shape in this tree that genuinely produces
"each tier keeps its own history", and it is a property of that provider
rather than of `switch_tier`: even on ONE instance, a `claude_cli` tier
never receives the turns another tier took. Two fixes, because they cover
different cases and neither subsumes the other —
`_apply_initial_tier_binding` overrides the provider alongside the model,
and `_ensure_provider` resolves the name from the runtime
(`cfg['provider_name'] or runtime.provider_name`) when the override is
absent, since `None` there never meant "no provider", it meant "whichever
one the runtime is configured for" — which is exactly what
`create_provider` had just resolved. A tier declaring no `provider` still
leaves the choice alone: that is what "use the session's main provider"
means.

**The window did not follow the model.**
`InstructionBudget.context_limit` is the denominator for the after-turn GC
threshold, the pre-send refusal guard and `get_context_usage` (so
`aspect="context"` and every client's readout). It was stamped ONCE, when
the provider was lazily created, and never again — while
`get_context_limit()` answers live for whatever provider is active.
Measured: a session that booted on a 200k tier and entered an 8k one
reported `get_context_limit() == 8000` against a budget still saying
`200000`.

The direction that hurts is entering a **smaller** window: GC cannot fire
before the request overflows, the guard lets it through, the upstream
rejects it, and `_try_gc_for_context_recovery` then trims history in the
**store** — destructive, and shared, so the tier that had the room loses
the conversation too. Which is how a stale denominator ends up looking
like the isolation that was reported.
`_refresh_context_limit_from_provider` is now the one definition, called
from both points where the binding changes (first materialisation, and
every tier connect), so the two cannot disagree about where the number
comes from. It joins the post-connect block that must not raise, and
therefore joins its rule: a failure is counted onto
`jaato.tier.context_limit_refresh_failures`, beside its two siblings —
best-effort blocks are not the problem, unobservable ones are.

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
| the runner that ran it (#812, record 2.10+) | `SessionState.runner_identity` | restored onto `Session.runner_identity` as **`stale=True`** — the pid named is from a previous process lifetime, so it is evidence, never a handle |

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
of which combination each workflow needs, live in `jaato_server/server/revive_policy.py`.

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

### A Credential Stored After the Runner Booted

A session's environment is resolved ONCE. `JaatoServer._resolve_session_env`
reads the workspace `.env`, the profile's `env:` map, the typed `trace:`
block and the post-auth overrides, decodes secret URIs (the daemon is the
only process that can exec `pass` / `vault`), and ships the dict on the
bootstrap envelope; the runner applies it to `os.environ` once, and the
provider resolves its credential once, in `initialize()`, and caches the
client. Every one of those is right for a session whose configuration is
settled, and together they close the door on the one being configured from
the prompt: `zhipuai-auth key <k>` stores the key in
`<workspace>/.jaato/zhipuai_auth.json` and the post-auth flow writes
`JAATO_PROVIDER` / `MODEL_NAME` to the `.env`, both **after** the runner
booted, so the open session keeps whatever it resolved at startup — a
daemon-wide variable inherited from the service environment, or nothing —
and every turn fails on it until a new session is created. Measured on a
live daemon: `Found Zhipu AI API key (env ZHIPUAI_API_KEY)` on a session
whose workspace held a valid stored key, 401 on every completion.

**`session.reload_env` is the refresh.** The daemon drops the
once-only flag, resolves again from the same four sources in the same
order, and calls the runner's `session.reload_env` with the WHOLE dict; the
runner runs it through `apply_session_env` — the one writer of the slot's
session-scoped environment, shared with bootstrap, so a re-application is a
REPLACEMENT (a key the previous dict set and the new one does not is gone,
which is what lets a reload retract a credential as well as supply one) —
and then `JaatoSession.reload_provider()` forgets the live provider and the
per-provider tier cache, re-arms the lazy-creation config from the binding
the session is currently on, and creates the new provider eagerly, so a
credential that does not resolve fails in the reload's answer rather than on
the next turn.

| Property | Why |
|---|---|
| **refused mid-turn, with nothing changed** (`stage="busy"`) | swapping the environment under a streaming provider call is a race; the daemon checks `is_processing` before paying the RPC, the runner checks `is_running` again |
| **env applied BEFORE the provider rebuild, and left applied when it fails** (`stage="provider"`) | the answer names the provider failure; the next `send_message`'s lazy creation still sees the new environment |
| **the answer names the credential source** | `auth_info` is the provider's own account (`API key from …/zhipuai_auth.json`), the line an operator compares against what they just stored |
| **the daemon fires it itself** after `<provider>-auth login\|key` | gated three ways: only those two actions (never `status`), only when the caller has a live session, only when that session runs the plugin's provider |

Precedence is **unchanged**: a provider still resolves config knob, then
environment, then the stored file, so a daemon-wide `JAATO_<P>_API_KEY` in
the service environment outranks a workspace's stored key however often it
is reloaded — the fix for that is removing the variable. Protocol **1.11**;
both SDKs refuse the verb below it (the 1.7 rule: a missing verb is ignored
silently, and "reloaded" would be reported about a session still on its
old credential). `IPCClient.reload_session_env()` / `reloadSessionEnv()`;
from a prompt, `session reload_env`.

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

**And WHICH profile, not only that there is one (#1052).** `profile` stayed an
unconstrained string, so the schema said a profile is mandatory and said
nothing about which ones exist. A production bot's model invented
`profile="summarizer"` — a name present in no profiles tier — read the
not-found result as retryable, re-worded the task and spawned again, once per
permission prompt, until the operator denied the tool to stop it. Same shape as
#944 one layer in: the valid set was already computed
(`_available_profile_names`, for the error message the model had already
failed to act on) and was never put where the model reads before it chooses.
`_spawn_profile_enum` now stamps it as the parameter's `enum`.

**It is a strong default, not a contract, and the difference matters here.**
`required` is enforced by the provider's function-calling validator on every
wire; an `enum` is *enforced* only under grammar-constrained decoding
(`strict: true` — opt-in via `api_params.strict_tools`, which the framework
deliberately does not turn on for you). Everywhere else it is read as part of
the description, and under the `prose_tool_calls` quirk the whole parameter
schema is prompt-injected text — the tier whose models are likeliest to invent
a name. Nothing in this framework validates tool arguments against the schema
before dispatch either (`jsonschema` here serves `signal_completion` payloads,
not tool args), so **the runtime not-found refusal remains the only layer that
actually refuses one**, and it stays. The issue's "cannot be emitted" is true of
one configuration and of no other.

Two cases withhold the enum rather than narrowing it:

| State | Schema |
|---|---|
| `allow_inline: true` | no enum — hung off `_inline_allowed()`, the predicate `required` and `inline_config` already read, so the surfaces cannot disagree about the knob |
| no profile available | no enum. `enum: []` makes every value invalid and some providers reject it outright; that state is reported in prose by `_inline_spawn_denial` and `list_subagent_profiles`. The `profile` PROPERTY stays either way — it is in `required` when inline is disallowed, and a required property absent from `properties` is an unsatisfiable schema |

The list is **sorted**, not in discovery order: `_scan_profiles_dir` builds the
profile dict from an unsorted `iterdir()`, and the tool schema sits in the
prompt-cache prefix, so a per-host order would re-read the whole prefix for
nothing. It is rebuilt per exposure for #944's reason — a shared registry means
a cached schema leaks one agent's profile set into another's tool list.

**Remote spawn is NOT exempted, and that is the one known cost.**
`spawn_subagent(server=...)` forwards the name verbatim to a PEER, which
resolves it against the peer's own `config_root`; the enum is built from
`self._config.profiles`, which is local. No predicate available at schema-build
time decides whether a peer is reachable, and both candidates are unsound
rather than merely imperfect:

| Candidate | Why it fails |
|---|---|
| enum iff `_remote_spawn_handler is None` | jaato-premium registers that handler on the **daemon-side** instance through a post-initialization session hook, while this schema is built **runner-side**, where the attribute is `None` even when remote spawn works (the runner→daemon bridge carries the call). It reads `None` on every path — an unconditional enum wearing a comment that claims otherwise — and is silently wrong on the one path that is live |
| no enum where a `runner_rpc_client` bridge exists | true on every runner-served session, i.e. the default and the deployment in the incident. Fixes nothing |
| a JSON Schema conditional (`oneOf` / `if`) | breaks strict mode and several providers |

So the constraint is documented instead of branched on: under
`strict_tools: true`, a spawn naming a **peer-only** profile becomes
schema-invalid. Under every other configuration it still executes — nothing
validates here, and the remote branch returns *before* profile resolution — the
model is merely steered away from it. The workaround is exact: declare a local
profile of that name. The stub satisfies the schema and changes nothing about
what the peer runs. `spawn_subagent`'s own `server` parameter description says
so, so the model reads it where it chooses.

**`inherit` is the one reserved `profile` value (#1198), and it does NOT
reopen inline spawning.** `spawn_subagent(profile="inherit")` spawns a
subagent from a **frozen snapshot of the parent's profile at spawn time** —
the parent's plugin set and the parent's assembled system instruction (carried
as `create_session(system_instruction_override=...)` so the child runs the
parent's exact framing rather than re-assembling, and doubling, its own). It
succeeds under `allow_inline: false` (the headline point: the "spawn a child
like me" pattern was only reachable through the opt-in #944 switched off),
because `inherit` NAMES a profile and passes the profile-first gate. It is
added to `_spawn_profile_enum` whenever the enum is offered and slots in
alphabetically under `sorted`; the discovery scan REFUSES a profile file that
would load as `inherit` (`RESERVED_PROFILE_NAMES` in `config.py`, rejected in
`_parse_profile_file`) so a workspace cannot shadow the reserved value.

**The self-replication guard is the triage's smaller, local mitigation, not a
depth bound.** An `inherit` snapshot includes the persona that makes the parent
delegate, so an `inherit` child is primed to call
`spawn_subagent(profile="inherit")` itself — and there is no spawn-depth bound
(#680), so leaving it in would make unbounded self-replication the path of least
resistance. So `_build_inherit_profile` strips the `subagent` plugin from the
inherited set: an `inherit` child can neither spawn nor message siblings, and a
caller that needs a spawning child names a real profile. (`_parent_plugins` is
already `subagent`-free upstream, so the strip is a property of THIS code rather
than an accident of how that list is populated.) "Frozen" is a deep copy via the
#787 snapshot round-trip, so later parent mutation cannot reach the child;
`agent=` / `default_agent` are ignored (the override is authoritative); and
`inherit` is refused — with a clear message — on the remote `server=` path (the
snapshot is local), under the isolated-runner opt-in (the override cannot cross
that boundary), and when there is no parent session to snapshot. Guard:
`shared/tests/test_inherit_profile_snapshot_1198.py`, five reversions.

### Session Group Messaging: the `courier` Plugin

`send_to_sibling` reached a peer by name inside ONE cascade and refused a
cold one on purpose; `session.wake` woke anything by id and checked
nothing about who asked. Neither let a session reach *another session of
the same user*, in another workspace, that had unloaded. The requirement —
any-to-any within a group, wake the target whatever state it is in — is
met by composing the two, with one new fact in between:

**A group is derived, never declared** (`server/session_groups.py`,
stdlib-only). Two sessions share a group when their key sets intersect:
`cid:<cascade_driver_id>` (record 2.10) and `user:<created_by>` (record
2.9, already `app:user`-qualified so a user group never crosses an
application). `None` never matches `None` — two anonymous IPC sessions form
no daemon-wide group — and an empty string is an absent fact.

| Piece | Where |
|---|---|
| the predicate | `server/session_groups.py` — `group_keys`, `same_group` |
| the cold half of "who is in my group" | a `membership` section on the `SessionWorkspaceIndex` (owner, cascade, sibling name), written beside the workspace mapping on every save; a per-workspace listing cannot answer "which sessions does this user own", and `SessionInfo` carries `created_by` so a cold record answers the predicate too |
| the verb | `SessionManager.deliver_group_message`: resolve (id first, name second — a name matching several members of a user group is `ambiguous` with the ids, never delivered to the first match), refuse unless `same_group`, the sibling grammar/size/cap checks, wrap as untrusted `peer:<sender>` with the #845 attachment manifest, then **loaded** → `deliver_prompt_to_session` on the idle-only `SIBLING` tier, **cold** → `resume_session` + `send_message_to_session` (no deferred-turn gate: a peer is not a client and none will attach, so the woken target runs headless). `event_id` shares `wake_session`'s dedup LRU. One `GROUP_DELIVERY:` daemon-log line per attempt |
| the model surface | the `courier` plugin, `PLUGIN_TIER = "daemon_callable"` with every executor forwarded — the cross-tier pattern in full, where `subagent` had wrapped two of nine. All four tools are `category="coordination"`; only the two listings are auto-approved; knobs `wake_cold`, `max_message_bytes`, `max_pending_per_target`, `max_exchanges_per_group` |
| the client surface | `session.message` (protocol **1.23**) → one `SessionMessageResultEvent` carrying the receipt and the caller's `request_id`; `IPCClient.send_session_message` / `sendSessionMessage`, refused below `MIN_SESSION_MESSAGE_PROTOCOL` (the 1.7 missing-verb rule) |

Three things are deliberate:

- **A target in another group answers `no_such_session`, the same words an
  unknown id gets**, so the verb is not an existence oracle across groups.
- **The sibling tools moved.** `send_to_sibling` / `list_siblings` left
  `subagent` for `courier` in the same change, contracts intact (cid-scoped,
  cold refused, the §8 caps); `subagent` returns to a plain runner-tier
  shape with no `DaemonForwardingMixin` and no `set_session_manager`. A
  profile whose persona names them and whose `plugins:` carries no `courier`
  gets `sibling_tools_moved` (**error**) from `validate`.
- **No per-turn visibility gate.** The runner-side session does not carry
  its cascade id, so a predicate there would hide the tools from exactly the
  cascade members who need them; a session in no group gets a refusal
  saying so instead.

Not built here (Phase 2/3 of the design): the durable inbox that survives an
unload between queue and drain, `file_refs` / `text_attachments`, and
cross-workspace file copy. Design and rollout:
[Session Group Messaging](docs/design/session-group-messaging.md).

### A Failure the Framework Was Told Was a Success (#1053)

`ToolExecutor` hands the reliability plugin one flag — `ok` — and derives it
from the executor's return SHAPE: `(ok, payload)` says what it means, and
`_normalize_executor_return` reads anything else as a success. The subagent
plugin returned a **bare dict** on 26 of its 33 failure paths:

```python
return SubagentResult(success=False, error="Profile 'x' not found…").to_dict()
```

so `on_tool_result` was told `success=True`, `PatternDetector` recorded it as
a success, and `_check_error_retry_loop` — which only walks entries with
`success is False` — could not see it. The plugin's own circuit-breaker and
retry policies were blind to five of its seven executors.

**The precedent was in the same file.** `_execute_send_to_sibling` and
`_execute_list_siblings` had already been converted, with a comment giving
this exact argument (*"making a failing tool invisible to anything watching
the event stream"*). Five executors were left behind — `spawn_subagent` (11
failure returns), `cancel_subagent` (6), `send_to_subagent` (4),
`_dispatch_isolated_spawn` (3, returned directly by `spawn_subagent`) and
`close_subagent` (2).

It is why #1052's invented-profile spawn loop ran unchecked: the detector's
similarity test is on argument **keys** (`patterns.py:543`), the loop reworded
only the values, and the default threshold is 3 — it matches, and never
reached the check.

**Only the flag changed; the payload did not.** `normalize_result_dict`
reshapes on `not ok` in exactly one case — an `error`-ONLY dict collapses to a
bare string — and `SubagentResult.to_dict` always emits `success` and
`turns_used` beside it, so the collapse cannot fire and the model reads the
same dict either way. What does change is correct and was already true of the
144 tuple-contract failures elsewhere: enrichment skips the result, the
telemetry span marks `is_error`, and the #951 `TOOL_RUNNER` trace records
`ok=False`.

| Property | Why |
|---|---|
| **failures explicit, successes left bare** | `split_executor_result` documents a bare value as `ok=True`, and 144 sites rely on it; converting successes is churn with no signal |
| **the guard is an AST scan, not a list of executors** | a failure path added later is covered whether or not its author remembers the contract |
| **`reliability` is opt-in** | it is not in `_ALWAYS_INITIALIZE_PLUGINS`, so this makes the failures VISIBLE to a detector a deployment must still enable. The flag is right regardless — telemetry and tracing read it too |

Not done here: `_normalize_executor_return` learning to read `success: False`
out of a bare dict. That shape survives at 9 more sites across `telepathy`,
`background` and `streaming`; changing the boundary would reclassify every
result whose `success` key means something else, and wants those enumerated
first.

### Marking Generated Output (#1117)

`ToolOutputEvent.generated_by` (protocol 1.14) is the machine-readable
marking Article 50(2) asks for, and it travels with the **delivery event**.
The moment a client writes those bytes down, the fact that a model produced
them is gone — and the Article is about the output, not about the event that
carried it. `TRAIT_OUTPUT_MARKER` is the seam that puts the marking *in or
beside* the payload so it survives leaving jaato.

| Piece | Where |
|---|---|
| the trait | `jaato_sdk/plugins/base.py`, beside `TRAIT_AUTH_PROVIDER` — a plugin capability, **not** a `ToolSchema` trait |
| the contract | `jaato_sdk/output_marking.py`: `OutputPayload` in, `MarkResult` out |
| the dispatcher | `JaatoSession._mark_generated_output`, called from `_deliver_model_media` and `_emit_withheld_attachments_to_clients` |
| the one in-tree marker | `jaato_server/shared/plugins/output_marker/` — `<file>.provenance.json` |

**Three rules the FRAMEWORK enforces, so a marker cannot get them wrong.**

- **A relayed payload is never marked.** The gate is `generated_by`, in the
  dispatcher, once — a fetched image is not AI-generated because an agent
  fetched it, and a false provenance record on somebody else's file is worse
  than none. Putting the gate in each marker is how two seams start meaning
  different things by "relayed".
- **A marker that fails must not lose the payload.** Every call is wrapped;
  a raising marker is traced and the ORIGINAL bytes continue. Refusing to
  deliver would be a stronger posture than the Article asks for and would
  take down every voice session on a transient disk error.
- **Which marker ran is recorded.** Each outcome traces `OUTPUT_MARKER: …`.
  A marker that looked and **declined** and **no marker configured** are
  different facts; a deployment that cannot tell them apart believes its
  output is marked because a marker is installed.

**The sidecar is C2PA-*shaped* and unsigned, and says so twice.** It carries
the IPTC `trainedAlgorithmicMedia` digital-source token and a C2PA-style
actions assertion, so a reader who knows C2PA recognises it; and it carries
`"conformance": "c2pa-shaped-unsigned"` with `"signature": null`, because
signing needs a certificate the framework cannot hold for you. The file is
`<file>.provenance.json` rather than `<file>.c2pa.json` for the same reason
— a name that asserts the standard would be read as a manifest by anything
looking for one, rejected by every verifier, and would meanwhile tell a
deployer their output is C2PA-marked.

**Audio is deliberately outside the in-tree marker's set.** A sidecar beside
a streamed utterance marks nothing anybody will read, and there is no
dependency-free watermarker; the state of the art moves faster than a
release. The trait is what an out-of-tree plugin implements, and the
declined marking is traced rather than silent.

**Nothing in tree produces an AI-generated file yet — that is the point.**
The only in-tree `Attachment` constructor is `clarification`: a person's
voice note answering a question, which must never be stamped. So the
deliverable is the contract plus an AST guard
(`test_every_attachment_producer_either_stamps_or_is_a_declared_relay`) over
an explicit relays-only allow-**list**: a producer the guard does not
recognise fails and must be classified, because the failure being guarded
against is a producer nobody thought about. A guard written when the first
image-generation plugin ships is a guard written after the first unmarked
output.

**Text is not marked**, and `jaato-scaffold explain oversight` says so under
`MARKING GENERATED OUTPUT` — computed from the tree (the protocol version
carrying the stamp, the marker plugins this build has) rather than written
down beside it. `AgentOutputEvent.source` attributes text at the event layer
and stops at the client; no watermark ships until the Article 50(7) code of
practice names one; and Article 50(4)'s publishing disclosure is a decision
the framework cannot see. A "generated by AI" prefix is refused on the
sharper ground that it would certify what it did not find — a reader seeing
it on some text and not other text concludes the unmarked text is
human-written, which the framework has no basis to say. Full argument:
[EU AI Act §4.3](docs/design/eu-ai-act.md).

### MCP Server Configuration

MCP servers are configured in `.mcp.json`:
```json
{
  "mcpServers": {
    "Atlassian": { "type": "stdio", "command": "mcp-atlassian" }
  }
}
```

**stdio is the only transport implemented.** `MCPClientManager` imports
`mcp.client.stdio` and nothing else, and `ServerConfig` carries
`command`/`args`/`env` with no URL field — so every server is launched as a
subprocess. The `"type"` key above is accepted for compatibility but read by
nothing. Remote servers (SSE / streamable HTTP) are **not** supported; a
URL-based entry will not connect. (The `mcp` help text advertised an `sse`
transport that never existed — corrected, since a user following it wrote
config that could not work.)

**A server's schema text is untrusted.** An MCP server authors its own tool
names, descriptions, and parameter descriptions, and those land in the
*trusted* region of the prompt — the schema block and the system
instructions — where the model reads instructions as legitimate. The plugin
therefore declares `TRAIT_UNTRUSTED_SCHEMA` and routes every schema through
`sanitize_untrusted_schema()`, and fences its per-server listing in the
system instructions with `wrap_untrusted_content()`. See the Tool Traits
table above.

### Streaming & Cancellation

Key types in `jaato_server/shared/plugins/model_provider/types.py`:
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

### A Release Nobody Was Told About

jaato ships through two channels and, until now, asked neither anything:

| channel | index | what a version there means |
|---|---|---|
| `pypi` | `https://pypi.org` | a **production release** — what `pip install -U <pkg>` gives you |
| `testpypi` | `https://test.pypi.org` | a **release candidate** — a staging build of a release that has not shipped yet |

So a production release or a staged candidate could sit on an index with
nothing in the framework that would ever mention it; the only way to learn one
existed was to open the project page. `jaato_server/shared/scaffold/dependencies.py` knew
every installed jaato distribution and its version, and compared it only
against the SOURCE TREE beside it (`dependency coherence`, the editable-install
skew check) — never against what had been published.

`jaato_sdk/release_channels.py` is the one place that asks. Two surfaces
render it, and they are **drawings of one report, never two opinions about
what is newest**:

| Surface | Form |
|---|---|
| `jaato-doctor` | one preflight line — `package releases`, WARN when something newer is published |
| `jaato-scaffold explain releases` | every channel's answer per package, with the commands that install each |

**Each channel carries a `pip` and a `uv` command, and the second is not a
rename of the first.** Both renderers loop over `Channel.install_commands`
rather than naming the installers, so they cannot document different sets.
The candidate channel is where the translation is load-bearing — measured
2026-09-18, with `jaato-sdk` 0.22.0 on PyPI and 0.23.0rc4 on TestPyPI:

```
uv pip install -U --prerelease allow \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ jaato-sdk
  -> jaato-sdk==0.22.0        the PyPI STABLE, not the candidate
```

It runs cleanly and installs the wrong package, which is the worst shape a
documented command can have. Two flags differ, not one: `--pre` is
`--prerelease allow`, and uv gives `--extra-index-url` priority **over**
`--index-url` (pip's precedence is the reverse) while defaulting to
`--index-strategy first-index`, so the first index holding the name wins
outright. `--index-strategy unsafe-best-match` restores pip's rule —
consider every index, take the best version — and with it both commands
resolve `jaato-sdk==0.23.0rc4` and the same seven packages. The flags are
therefore spelled out per channel rather than derived from the pip string:
they are not a transformation of it, and a helper that pretended otherwise
would re-introduce exactly that wrong-package failure.

**The index's own `latest` is the wrong answer, on the channel that matters.**
PyPI pins a project's "latest" to the newest **stable** version whenever one
exists — correct for `pip install`, and the reason the publish workflow stages
every TestPyPI build as a pre-release in the first place. Measured 2026-09-18,
with `jaato-sdk` 0.23.0rc4 published:

```
test.pypi.org   jaato-sdk   info.version = 0.21.0      actually newest: 0.23.0rc4
```

Reading that field reports the release-candidate channel as two releases
*behind* the candidate it is carrying — silently, about the one channel the
feature exists for. Versions are therefore computed from the release listing
and ordered per channel, and a release with no files or with every file
**yanked** is excluded: neither is installable, so offering either as "newer
is available" sends a reader to a command that no-ops.

**PEP 440 ordering is implemented here rather than imported.** jaato-sdk
depends on `python-dotenv` and `pydantic`; `packaging` is not a dependency, so
importing it would make the check work on the machines that happen to expose
pip's vendored copy and silently do nothing on the others. A capability
fallback is worse still — two orderings that can disagree about which release
is newer is the bug the check would then be shipping. There is one
implementation, and a version string outside PEP 440 is **refused** rather than
approximated: `parse_version` returns `None`, the string is carried on
`unparseable` and named in the output, never sorted as "older", which is how a
notifier starts hiding the release it was asked about.

Four rules, each attached to a way a version notifier stops being read:

| Rule | Why |
|---|---|
| **an index that did not answer is `unknown`** | absence of evidence is not currency. A notifier that reads silence as "up to date" answers the question wrongly instead of declining to, and the reader now believes something false. WARN, in the wording `check_mcp_sdk` already uses for "cannot check" |
| **a build newer than both channels is `ahead`, not `current`** | the normal state of a checkout of this repository. Telling a contributor on an unreleased build that they are up to date is how a check stops being read |
| **never FAIL** | a release is news, not a defect, and `jaato-doctor` is documented as usable as a CI gate — exiting non-zero because somebody shipped would break every harness using it the documented way |
| **a dead index is asked once per RUN, not once per package** | the deadline is per request; with several distributions and two channels each, re-learning "the network is down" per package is one deadline against eight. A connection-level failure retires its channel for the run; an HTTP answer (a 404 for one unpublished package) does not, because it says nothing about the next one |

**On by default, because a notification nobody enables is a notification
nobody gets** — which is the complaint this answers. It is bounded (a 3s
per-index deadline), cached for 6h at `~/.jaato/release_check.json`, and
switched off by `JAATO_RELEASE_CHECK=off` (also `0`/`no`/`false`/`none`/
`never`). A value that is set and *unrecognised* reads as ON: the two failure
directions are not equal, since a typo that silently disables the notifier
reproduces the state being fixed and is invisible, while a typo that leaves it
on costs one bounded request. `jaato-doctor` also takes
`--no-release-check`, `--release-check-timeout SECONDS` and
`--refresh-release-check` (for "I just published — is it visible?").

**`explain releases` is its own topic rather than a section of `explain
dependencies`.** That verb is an offline introspection of the installed tree,
and quietly giving it an egress would change what running it means. What the
two DO share is the distribution set: `installed_jaato_dists()` now delegates
to `release_channels.installed_distributions()`, so "which distributions are
ours" has one definition rather than two applying the same jaato-prefix rule —
#966's measured-not-hardcoded rule, which is also what lets a separately
shipped package (`jaato-premium` today) be release-checked with no edit here.

`JAATO_RELEASE_CHECK` is read in **exactly one place**, the SDK module, and
the `shared` surfaces honour it by calling that module rather than growing a
second reader. That is also why it is absent from `jaato_server/shared/env_scope.py`: that
catalog is re-derived by an AST scan of `jaato_server/server/` and `jaato_server/shared/`, so an
SDK-only entry would be reported as stale in both directions.

Tests: `jaato-sdk/jaato_sdk/tests/test_release_channels.py` (ordering
cross-checked pairwise against `packaging` where it is installed, channel
semantics, cache, offline degradation — no test touches the network or the
real cache), `.../test_doctor_package_releases.py` (the preflight rendering),
`jaato-server/jaato_server/shared/scaffold/tests/test_explain_releases.py`, and
`jaato-server/jaato_server/shared/tests/test_release_check_notices_a_new_version.py`, which
carries the four `REVERSIONS` — trusting `info.version`, reading unreachable
as current, ordering versions as strings, and a check `run_checks` does not
call.

### Proactive Garbage Collection

The framework monitors token usage during streaming and automatically triggers GC when thresholds are exceeded:

```python
from jaato_server.shared.plugins.gc import GCConfig

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

### A Strategy Resolved, Carried, Rendered — and Installed on Nobody (#1133)

`JaatoSession._gc_plugin` has two writers, both inside `set_gc_plugin` /
`remove_gc_plugin`, and its callers were the **embedded** client and
**in-process subagents**. Nothing under `jaato_server/server/runner/` called it, and nothing
there read `envelope.gc` either — so on the **runner-served path, the default**,
a session had no GC plugin whatever its profile declared. Every collection site
opens `if not self._gc_plugin or not self._gc_config: return`, so nothing
collected; meanwhile `core.py` kept resolving the same config to fill the
`gc_threshold` / `gc_strategy` readouts in the TUI status bar and the web rail.
A strategy displayed and never run — worse than showing nothing, because the
readout is the thing an operator checks.

Measured before the fix: `_build_session` received **17 kwargs** and `gc` was
none of them. The control — handing that same session a plugin the way the
embedded path does — showed it immediately, so `None` was the absence of an
install rather than a blind probe.

**The producer was broken too, and that is what makes it two fixes.**

```python
gc_config = getattr(gc_obj, "config", None) or {}   # GCProfileConfig has no .config
gc_dict   = {"type": gc_type, **dict(gc_config)}    # → always just {"type": ...}
```

So the envelope could only ever carry the strategy NAME. Wiring the consumer
alone would have installed `budget` at **framework defaults** and discarded
every number the profile declared — the same silent-ignore shape one layer up,
wearing the fix as a disguise.

Three write-sides for one dataclass existed and each had drifted differently:
the producer (1 field of 12), the session-snapshot serializer (10 of 12,
missing `target_percent` and `pressure_percent`, so a revived session lost how
far a collection goes and when PRESERVABLE may be touched), and `from_dict`
(12 — the only complete one, and the only one with no symmetric partner).

| Change | Where |
|--------|-------|
| `GCProfileConfig.to_dict()` — derived from the dataclass fields, so it cannot be edited out of date | `subagent/config.py` |
| the producer and both halves of the snapshot serializer call it | `runner_spawn.py`, `subagent/serializer.py` |
| `_install_gc` resolves and installs at bootstrap | `jaato_server/server/runner/session.py` |

`_install_gc` adds a **caller, not a second definition**: precedence is the
daemon's — profile `gc:` first, then `<workspace>/.jaato/gc.json` — through the
same `gc_profile_to_plugin_config` an in-process subagent uses and the same
`load_gc_from_file` the daemon uses. The runner resolves rather than receiving
a built plugin because a plugin is not serializable: the envelope carries the
declaration and the side that will own the object constructs it.

It is **best-effort and audible**. A session that fails to install GC is every
pre-#1133 session, so raising would turn a silent degradation into a refused
bootstrap; the failure logs at WARNING instead, because a GC strategy that does
not install is exactly what this issue is about and must not become invisible
twice. A successful install logs the strategy and its thresholds.

The guard is `jaato_server/server/tests/test_envelope_carries_gc.py`, third in the family
with `test_envelope_carries_budget_control.py` and
`test_envelope_carries_runtime_limits.py`. Its consumer test asserts the **call
site** by AST walk, not the installer's behaviour: a test that imports
`_install_gc` and calls it passes on the broken tree, since the broken tree's
defect was precisely that the resolution existed and nothing invoked it. The
reversion meta-guard caught that weakness in the first draft of this very file.

### A Workspace With No GC, and a File That Overrode What It Did Not Say

Two halves of one question — *what does a session get when nobody chose a GC
strategy* — and both answered wrongly, in opposite directions.

**A created workspace had no strategy at all.** A session's GC comes from its
profile's `gc:` block, else `<workspace>/.jaato/gc.json`, else
`~/.jaato/gc.json`; when none of the three answers, `JaatoServer.initialize`
leaves `gc_result` at `None` and the session runs with **no context garbage
collection whatever** — the history grows until the pre-send guard refuses it
or the upstream does. `WorkspaceManager.create_workspace` made `.jaato/`,
touched an empty `.env` and stopped, and a session driven against such a
workspace is a bare `session.new` over that `.env`'s `JAATO_PROVIDER` /
`MODEL_NAME` pair — no profile, so no `gc:` block. Every workspace the web
client creates was in exactly that state.

It is fixed by writing `.jaato/gc.json`, **not** by a line in `.env`, which is
where one reaches first: there is no `JAATO_GC_TYPE`. The four `JAATO_GC_*`
variables are read by `GCConfig`'s field defaults, and that object is
constructed only once a strategy has been selected — so a threshold written
into `.env` with nothing selecting a strategy configures nothing, silently.
Choosing the strategy is the load-bearing act.

`budget` rather than `truncate` because it dominates it: with an
`InstructionBudget` it removes by GC policy (enrichment first, never
`LOCKED`), and without one `BudgetGCPlugin.collect` falls back to the same
turn-based truncation `gc_truncate` would have done. There is no state in
which it is the worse choice. The generated file carries `type` and nothing
else, for the reason the other half of this section is about. Existing
workspaces are not migrated — the file is a starting point its owner is meant
to edit, and a workspace that predates this may have been deliberately left
without one.

**And a `gc.json` decided keys it never mentioned.** `load_gc_from_file` built
its `GCConfig` with `data.get(key, <literal>)` for the trigger keys, so
`threshold_percent` and `target_percent` were fixed at 80.0 / 60.0 for any
session that had a `gc.json` **at all** — while those fields' own docstrings
promise *"Can be overridden via `JAATO_GC_THRESHOLD`"* / `JAATO_GC_TARGET`.
`_media_settings` had already fixed precisely this for the three media keys
and written the rule down — *omission has to mean "the dataclass decides", not
"the default I happened to type"* — and the trigger keys were left behind.

`pressure_percent` was worse, because its default is not a number.
`data.get('pressure_percent')` answers `None` for a file that omits it, `None`
is how `GCConfig` spells **continuous mode** (GC after every turn above
`target_percent`, `threshold_percent` ignored), and passing it explicitly beat
the env-derived 90.0 — so omitting one key silently selected a different
operating *mode*. The `== 0` test sitting immediately below that read is the
evidence it was never meant to: that line exists to make a literal `0` mean
continuous, which is only worth writing if *absent* does not. The profile
route into the same dataclass never had any of this (`GCProfileConfig`
carries real dataclass defaults, `pressure_percent = 90.0` among them), so
**the two routes disagreed about what an omitted key means**. `_scalar_settings`
is `_media_settings`' rule applied to the rest of the file; a declared `0` or
`null` still opts into continuous mode, because that is the documented opt-in.

The two halves are one change because the first is unsafe without the second:
a generated `{"type": "budget"}` under the old loader would have put every new
workspace into continuous GC and made the `JAATO_GC_*` knobs in its own `.env`
inert.

### One Invariant, Enforced Where History Leaves (#674)

Every provider here requires that an assistant turn's function calls and
the results answering them stay **paired**: one `ToolResult` per
`FunctionCall` id, no result before its call, no empty content block. Break
it and the upstream rejects the *whole* request — several turns after the
damage, which is what makes the class expensive.

**Five subsystems edit history independently and none can see the others**:
the four GC strategies, cancellation (`CancelToken` mid-batch, widened by
8-wide parallel execution), `rewind`, subagent history sharing, and the
**wire** — an OpenAI-compatible endpoint that streams a tool call with no
id puts a call into history no result can ever match. Enforcing at each site
means auditing five subsystems plus every future GC plugin; enforcing at the
boundary is one place.

`jaato_server/shared/history_invariant.py` is that place. `validate_history` reports
defects (`unmatched_call`, `orphan_result`, `missing_call_id`,
`duplicate_call_id`, `empty_content`); `repair_history` fixes them in four
ordered passes — mint missing ids, drop orphan results, **answer** unmatched
calls, drop empty content — and is called from
`JaatoSession._history_for_provider`, the seam every `provider.complete()`
call site already reads (#847).

| Property | Why it is load-bearing |
|---|---|
| **repairs the per-request COPY, never stored history** | the #847 precedent, inverted: a turn cancelled mid-batch must keep its unanswered calls on disk, because the session may still execute them and real results would collide with synthetic ones written into the store |
| **an unanswered call is ANSWERED, never deleted** | deleting the assistant message is the obvious repair and it strips `Part.thought` — which `replay_reasoning` providers require back (MiMo answers **400** without `reasoning_content`, Kimi K3 wants the message as-is). That trades a pairing 400 for a reasoning 400. No MODEL message is ever removed |
| **the synthesised result is byte-CONSTANT** | `repair_history` runs on every request built from the same history, so a timestamp or random id in the payload would change the request prefix each turn and cost a full prompt-cache re-read on Anthropic/Gemini upstreams |
| **a healthy history is returned unchanged** (same list object) | the overwhelmingly common case allocates nothing |
| **every repair traces `HISTORY_INVARIANT: ...`** | a silent repair hides the subsystem producing bad histories, which was half the original defect |

**Empty content blocks are MORE reachable now, not less.** *"Text content
blocks must be non-empty"* after a thought-only turn is exactly the shape
reasoning replay makes routine: a `replay_reasoning` session deliberately
keeps thought parts in history. The empty text beside the thought is removed
and the thought is kept.

**The wire seam is fixed too, not only the backstop.**
`synthetic_tool_call_id(index, nonce)` mints an id at accumulation time in
all four streaming loops that key tool-call deltas by index
(`_openai_compat` — so nim/nebius/ovhcloud/doubleword/zhipuai_openai — plus
`openrouter`, `vllm` and `github_models`). The **nonce is required, not
defaulted**: `index` is unique only *within* a response, so index alone
would put two different calls from two turns under one id in the same
history — the defect wearing the fix as a disguise. An upstream-supplied id
is never replaced.

**Measured, and recorded because it is a negative result.** On a seeded
corpus of multi-call histories all four GC strategies are *already*
pair-safe — they cut on turn boundaries (`split_into_turns` /
`flatten_turns` keep an assistant turn and its results together), removing
~500 messages across 40 seeds without orphaning a call. So those tests are a
regression detector, guarded by `test_gc_collection_is_not_vacuous`.

**The defect was in the repair, not the strategies.**
`ensure_tool_call_integrity` — the GC-path repair, named for the invariant —
*produced* the violation it exists to prevent, on the shape #674 is about:

```
MODEL calls {A, B}  ->  TOOL answers A  ->  USER turn
```

Pass 1 kept the TOOL message (`A` is valid); pass 2 reached the USER branch
with `B` still pending and deleted the **MODEL** message, leaving `A`'s
result answering a call present nowhere. Every case with at least one real
result — 14 of 18 at widths 2/3/5/8. And its output is **stored**, so a
session GC'd into that state and then persisted came back corrupt on revive.

So the producer is fixed at source: `ensure_tool_call_integrity` now
delegates to `repair_history`, giving one policy in one place. Writing
synthetic results into the store is safe there because all four call sites
run **between** turns — `_maybe_collect_after_turn` and
`_maybe_collect_before_send` sit on turn boundaries, `manual_gc` is
operator-driven, and `_try_gc_for_context_recovery` pops the trailing MODEL
message with pending calls before calling it. The boundary validator remains,
now as a backstop for a *fixed* producer rather than the sole defence — it
covers what GC never sees (cancellation, rewind, subagent sharing, the wire).
Its complexity baseline entry dropped from **41** to gone (the function is
now 1).

**Already-corrupt records are not migrated**, deliberately. Such a history is
never *sent* corrupt (every request goes through `_history_for_provider`), and
the first GC pass in a revived session heals the stored record — but an
untouched record stays corrupt on disk until one of those happens. Rewriting
persisted session records is a migration with its own failure modes; the two
paths above cover every route this tree reads history by.

Tests: `jaato_server/shared/tests/test_history_invariant.py` (unit per defect kind,
seeded property tests over all four GC strategies / cancel-at-turn-N / every
suffix / every partial-batch width, plus the **prose** wire — `_prose_tools`
carries the pairing in text, so a repair that is pair-safe on the Anthropic
wire is not automatically pair-safe there; both pairing survival *and*
byte-stability are asserted on it, since the prose form re-serialises the
whole result including its `(call <id>)` label),
`jaato_server/shared/plugins/gc/tests/test_utils.py` (the stored-history half — the
`{A,B}`/answer-`A`/USER regression, and the renamed tests that used to
encode the deletion policy) and
`jaato_server/shared/plugins/model_provider/_openai_compat/tests/test_streamed_tool_call_without_id_674.py`
(drives the real streaming loop of three providers; verified to fail 9/15
when the mint is neutralised, because a test that cannot fail proves
nothing).

### Deferred Tool Loading

Tools have a `discoverability` attribute: `"core"` (always loaded) or `"discoverable"` (on-demand).
Model uses `list_tools()` → `get_tool_schemas()` workflow to discover tools.

- Enabled by default (`JAATO_DEFERRED_TOOLS=true`)
- Core tools: introspection, file_edit, cli, filesystem_query, todo, clarification

### Pre-warm Runner Pool

Sessions consume a pre-warm runner subprocess from a pool instead of cold-spawning one each time.  Cuts per-session bootstrap from ~30s (with full plugin discovery + imports) to ~7s on cascade workloads.

Architecture: daemon spawns a **template subprocess** at startup that imports all runner-tier plugin modules.  N pre-warm **pool slots** fork from the template (no exec), inheriting the warm imports.  When a session arrives, daemon claims a pool slot and dispatches `session.bootstrap` to it via the same `RunnerRPCClient` it would use for a cold-spawned runner.  A slot that has never been confined self-confines to the session's AppArmor profile in bootstrap step 1c via `aa_change_profile` (main-thread dispatch so subsequently-spawned worker threads inherit the confined cred).  A slot that already wears a profile does **not** transition — it is only ever handed to a session that wants the profile it has, which is what the reuse key enforces (#1033, below).

**Operational properties:**
- **Subreaper**: daemon calls `prctl(PR_SET_CHILD_SUBREAPER, 1)` at startup so orphaned descendants (slots whose template died) re-parent to the daemon.
- **Watchdog**: `PoolManager` replenishment thread detects template death + auto-respawns + refills pool.
- **READY handshake**: template sends `"READY\n"` after plugin discovery completes; daemon's `TemplateManager.spawn` blocks for it (30s timeout) instead of a fixed sleep.
- **Telemetry**: `PoolManager.get_telemetry()` exposes counters (`pool_slot_acquired_total`, `pool_acquire_miss_total`, `pool_replenish_success_total`, `pool_replenish_failures_total`, `template_respawn_attempts_total`, `template_respawn_failures_total`, `pool_slots_over_cap_total`, `pool_stale_reservation_evicted_total`, `pool_replenish_ceiling_blocked_total`, `pool_profile_mismatch_skips_total`, `pool_dead_slot_evicted_total`, `pool_duplicate_return_refused_total`).
- **Liveness**: a slot whose `RunnerRPCClient` has died is never handed out and is reaped rather than skipped — see [A Slot the Pool Kept Offering After Its Channel Died](#a-slot-the-pool-kept-offering-after-its-channel-died-1058).

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

**Pool routing gates** (`spawn_session_runner`): pool is consulted iff `pool_manager` wired AND env flag enabled AND `cgroup_attach is None` (cgroup migration mid-life is a follow-up).  Apparmor opt-in sessions ARE eligible — but **not** because the slot re-confines itself per session; see the next section for what actually makes that true.

### A Slot the Pool Kept Offering After Its Channel Died (#1058)

`PoolManager` had **no notion of RPC liveness**. `_closed` is set in exactly
two places — `RunnerRPCClient.close` and the read loop's `finally` — and the
second runs long after the session that created the client ended, while the
slot sits in `_idle_slots`. Nothing told the pool. `acquire_slot` handed back
whatever was at the head of the list, and the next session discovered the
corpse by failing its own bootstrap:

```
19:00:25,826  spawn_session_runner: session ... served by pool slot pid=5307
19:00:25,826  spawn_session_runner: session ... reused slot's rpc client
19:00:25,828  [ERROR] runner session.bootstrap FAILED:
              error_type=RunnerCallError error=RunnerRPCClient is closed
```

**The one reaper the pool had was exempt from looking.** `_sweep_cascade_idle`
skips `cascade_id is None`, which is right — a PURE IDLE slot has no cascade
affinity to *time out* — and a standalone session returns its slot as PURE IDLE
(#1033). So the reported slots sat in the one state nothing ever inspected.
Warmth has a timeout; liveness is not warmth, and `_sweep_dead_slots` exempts
nobody.

**A poll, not a callback.** The read loop already knows and already sets the
flag; what was missing was anybody asking. `slot_rpc_death` is asked at the two
points where a slot's liveness is load-bearing — `acquire_slot`, and the
replenish sweep — because a poll at the point of use cannot silently stop
working, where a callback registered at slot construction is inert the moment
one of the two construction paths forgets it (#735). Only **positive evidence**
counts, the posture #1014 and #1023 take about confinement labels: no client is
alive (nothing has died), and a client that cannot answer `is_closed` is alive
(reading absence as death would empty the pool).

**Skipping would have been cheaper and wrong twice.** A corpse left in the list
leaks its runner process, and — the sharper half — keeps counting toward
`unreserved_idle_count` and `max_size`, so replenishment reads the pool as full
and never forks the replacement. That is #898's failure in a new costume:
capacity no arriving session can use, counted as capacity for everybody. Dead
slots go to `_pending_teardown`, which the existing drain reaps.

**Layer 2 does more than check a flag.** `spawn_session_runner` runs OFF the
daemon loop while the read loop that sets the flag runs on it, so the reuse
fast-path can still be handed a client that died since the acquire. What it
does about it depends on **how** the client died, which is why the cause is now
recorded (`close_reason`):

| `close_reason` | What survives | Action |
|---|---|---|
| `cancelled` | the runner, the socket, the transport — only the READER task is gone | start a **fresh read task** on the surviving transport (`revive_read_loop`) |
| `eof` | nothing — the peer is gone | discard the slot, cold-spawn |
| `explicit-close` | nothing — `close()` reaped the runner | discard, cold-spawn |
| `oversized-frame` / `malformed-frame` | the stream is desynchronised mid-frame | discard, cold-spawn |

A task cannot be un-cancelled, so clearing the boolean would hand back a channel
nobody is reading — a hang instead of an error. Re-**adopting** the socket is
never an option: it is already owned by the dead client's asyncio transport, and
a second `connect_accepted_socket` on it is the PR #173 failure the shared-client
design exists to avoid. `close()` can never produce `cancelled` — it stamps
`explicit-close` *before* cancelling the read task and the first cause wins — so
that reason is a statement of provenance: something **outside** this client
cancelled it.

**Two nearby defects the same investigation turned up, both fixed:**

- **A read task cancelled before its first step never runs its `finally`**, so
  `_closed` stayed `False` and the channel reported itself HEALTHY with no
  reader: writes land, replies are never consumed, every call waits out its
  deadline. Worse than the reported failure, because the liveness poll answers
  "alive". A done-callback records it, narrowly — only when the task was
  cancelled and nothing else has recorded a cause.
- **`return_slot_after_session` logged "returned to pool" for a slot it did not
  pool.** At capacity with a pure-idle returner, the returner is the one
  dropped — and the line still said it was pooled. The incident's key log line
  (`slot pid=5307 returned to pool (idle_count=4/4)`) is *exactly* that branch's
  shape, and reading it as proof the slot entered the pool cost two rounds of
  diagnosis. The method now returns whether the slot was retained, and both its
  own line and `JaatoServer.shutdown`'s say which happened.

**One slot, one entry.** `return_slot_after_session` refuses (by identity) a
slot already in `_idle_slots`, at ERROR, counted. `JaatoServer.shutdown`
captures and nulls `_runner_rpc` / `_spawned_runner` / `_pool_manager_ref`
**without a lock** and is called from session unload, `session.stop`, the #812
orphan sweep and daemon shutdown — so two concurrent callers can both see the
live triple and both return the same slot. The two entries share one `rpc`:
tear either down and the other is a corpse in the pool that no teardown line
names. Guarded at the pool rather than at the four teardown paths, the #674
argument — the pool owns the list, so one boundary check covers callers that do
not exist yet. It does not *fix* the double shutdown; it makes it audible.

**What killed the reported client is still unknown, and is now answerable.**
Both read-loop exit lines named no channel (`RunnerRPCClient: runner closed
connection`, `read loop crashed`), so an operator's grep for a pid could not
detect a read-loop exit at all — which is why three rounds of evidence went into
distinguishing lines that could have named themselves. Both carry
`pid=` now, `close_reason` records which of the five endings it was, and the
pool's eviction WARNING prints it. Counters: `pool_dead_slot_evicted_total`
(**not** a success metric — each unit is a runner that died unasked) and
`pool_duplicate_return_refused_total`.

Latent, not the cause, and worth its own change: `MCPClientManager.__aexit__`
cancels **every task on the running loop** (`asyncio.all_tasks()`, unscoped).
Today it runs on a dedicated loop in a dedicated thread in the runner process,
so it cannot reach a daemon-side read task — but it is one `async with` on the
daemon loop away from doing exactly what this issue describes.

### A Key That Said "Reusable" and a Name That Said "New" (#1033)

The slot reuse key was `(cascade_id, config_root)` and the AppArmor profile
was named `jaato-ws-{session_id}` — **the one property guaranteed to differ
on every reuse**. Both statements cannot be true once confinement is
per-task, and #1023 is what made that visible: `aa_change_profile` confines
the CALLING TASK, the kernel refuses `current != task` (`-EACCES`), and
#1026 can retire only the two RPC executor lanes. `OtelBatchSpanRe` and the
reader/`Thread-N` set are not executors, so on every **reused** slot the
thread population straddled two profiles and `verify_thread_confinement`
correctly refused the bootstrap.

Measured on the maintainer's enforcing host: five consecutive session
creations failed with `RunnerCallError`, each naming four of five threads
still labelled with the PRIOR session's profile. Deterministic per slot — a
fresh slot bootstrapped, a reused one refused — which is exactly why it read
as intermittent.

**The guard was right; the naming was wrong.** So the profile is named after
the BOUNDARY (`jaato_server/server/confinement_id.py`), and the key contains it:

| property the slot carries | in the key |
|---|---|
| warm plugin state from the config tree | `config_root` |
| warm plugin instances, tenant isolation | `cascade_id` |
| **the AppArmor profile its threads wear** | **`workspace_root` + `profile_name`** |
| the cgroup (`runtime_limits` kernel trio) | not needed — `cgroup_attach is not None` routes the session away from the pool entirely |
| session `env:` / secrets | not needed — overlaid per turn, never baked into the slot |

The rule that generates it: **the key must contain every property of the
slot that the next session cannot change.** With the name derived from the
key, "reused slot" and "same profile" are one statement, so a reused slot
takes `_maybe_self_confine`'s **idempotent** path — no `aa_change_profile`,
nothing to diverge.

`confinement_id` = a workspace slug plus a digest of the **rendered profile
body**, which folds in the config root, the env file, the composed fragments'
contents and the plugin-contributed rules. Rendering rather than hashing the
inputs is what stops a second "what goes into a profile" implementation
drifting from the renderer. Including the body is not belt-and-braces: two
CONCURRENT sessions of one cascade — a narrow stage and a broad one, each on
its own slot — would otherwise share a name, and provisioning the broad one
would reload the profile the narrow one is confined to. A silent widening of
a live boundary is worse than the failure being fixed.

Three consequences, each handled rather than inherited:

- **The gate is on BOTH acquire paths.** A standalone session (no cascade)
  returns its slot as PURE IDLE — profile and all — and path (2) hands it to
  the next arrival of any kind. The cascade key never covered that path, so
  every unrelated pair of workspaces on one daemon reproduced the same
  straddle. `SlotKey.accepts_unaffined` is that gate: a never-confined slot
  fits anybody, a confined one fits only the profile it already wears.
  Counter: `pool_profile_mismatch_skips_total`.
- **Profile teardown is per-SLOT, not per-session.** A boundary-derived
  profile outlives its session, so unloading it at session end would strip
  the boundary off a live runner. `PoolManager._reap_slot_profile` unloads
  when the last slot wearing it dies; `AppArmorManager.teardown_profile`
  refuses to unload an id a live session still holds. Unbounded accumulation
  goes with the per-session name: boundary-derived names are bounded by the
  number of distinct boundaries a deployment has.
- **The profile name no longer identifies the session.** It stays
  human-readable (`jaato-ws-my-repo-3f2a9c1b7d4e`), and #812's
  `runner_identity` is what records which runner ran which session.

**Two standalone sessions in one workspace and config root now share a slot
and a profile.** With `cascade_id = None` the key collapses to the boundary,
which is identical for both, and #890 parks nothing for a standalone
session — so there is no warm state to leak, and the kernel boundary they
share is the one each would have been given separately.

Costs, stated: a cascade whose stages have genuinely different boundaries now
uses a slot per boundary instead of one slot and a transition, and a
boundary's AppArmor **reference fragments** (`selectReferences` grants) are
shared by every session on it rather than starting empty per session. Both
are the direct price of a slot being unable to change the profile it wears.

Not verified here: no kernel. This container carries no AppArmor LSM, so the
tests exercise the naming, the key and the lifetime with `is_available()` and
`apparmor_parser` stubbed — they prove the framework stops ASKING the kernel
to do what #1023 says it cannot, not that the kernel then behaves.

See `docs/design/runner_prewarm_pool_plan.md` for the full multi-PR plan + decision log.

### "Never Confined" Is Not "Never Served" (#1100)

#1033 put the boundary in the reuse key. The gate that reads it asked the
wrong question about half of it:

```python
return not slot.profile_name or slot.profile_name == self.profile_name
```

`SlotKey.build` folds `""` to `None` so unconfined has ONE spelling — right
for a key — so a slot that **served an unconfined session** stamps
`profile_name=None` and is indistinguishable from a **virgin** slot, while
carrying that session's `unconfined` threads. Handed next to a confined
session, the main thread transitions, the two RPC lanes are recycled, and
#1023's per-thread verification correctly refuses the bootstrap for the
leftovers — which cannot be confined, only retired.

Confirmed on a live daemon, one slot:

```
12:06:57  slot pid=95942  profile=(none)                     confined=False  -> ran fine
12:11:25  slot pid=95942  returned to pool
12:17:44  slot pid=95942  profile=jaato-ws-test-hola-3-...   confined=True   -> REFUSED
          "6 of 7 scanned threads report otherwise (tid=95982 label='unconfined', ...)"
```

It **escalates with uptime** — 2 of 3 threads one day, 6 of 7 the next —
because the survivors accumulate per session on a long-lived slot. It fails
CLOSED, which is the safe direction and the whole reason the guard is not
what changed.

`PoolSlot.has_served` is the missing property, and the gate becomes:

```python
if not slot.has_served:
    return True
return (slot.profile_name or None) == self.profile_name
```

This is #1033's own generating rule — *the key must contain every property
of the slot that the next session cannot change* — applied to the one it
missed. `has_served` **cannot be derived** from the key, because every one
of its four fields is legitimately `None` for an unconfined standalone
session; that is the defect, not an implementation detail of it. It is
raised in `SlotKey.stamp` (the one place a slot stops being virgin) and
never lowered, and it is deliberately NOT a fifth key field: a key says
what an arriving SESSION wants, and path (1) compares the whole key for
equality, where a served slot must still match a session wanting exactly
its boundary.

**Cost, and it needs no new instrumentation.** On a daemon mixing postures,
a slot that ran unconfined is no longer offered to a confined session, so
that arrival cold-spawns (~7 s). Already counted by
`pool_profile_mismatch_skips_total` beside `pool_acquire_miss_total` — the
skip branch does not care WHY the boundary did not fit — and remedied by
raising `JAATO_RUNNER_POOL_MAX_SIZE`. The same cost #1033 accepted for the
multi-boundary case. Unconfined→unconfined reuse is untouched, and on a host
with no AppArmor every profile name is empty, so the second clause is a
tautology and nothing changes.

**The refusal now names the threads.** `_tids_from_threading` had each
`Thread` object in hand, read `native_id` and discarded `.name`, so the
message was `tid=… label=…` and two separate live incidents were
investigated — one by correlating daemon-log timestamps against slot pids —
without ever establishing what the divergent threads were.
`ThreadProfileScan.names` carries the map (read at scan time, because a
thread that exits before the message is rendered has already left
`threading.enumerate`) and the message reads
`tid=95982 name='runner-rpc-work_0' label='unconfined'`. A tid the
interpreter does not know — always possible on the complete `task_dir`
route — renders `(unknown)`; `/proc/<tid>/comm` is not a substitute, since
every runner thread's `comm` is `"python"` on CPython 3.11.

The name is **advisory and nothing else**. It never decides whether a thread
is divergent: an allow-list of thread names is an allow-list of unconfined
code, and it is the one change here that could not be validated without an
enforcing host.

**`TelemetryPlugin.shutdown()` had no caller anywhere in the tree**, and
that is a plain thread leak independent of AppArmor — the most likely
member of the surviving population. Telemetry is RUNTIME-scoped and a
runner builds a fresh `JaatoRuntime` on **every** `session.bootstrap`, so
the outgoing plugin is unreachable after session end; it is not a registry
plugin, so `_handle_session_end`'s sweep over `registry.list_available()`
never saw it. Dropping a reference is not freeing a resource: an OTel
`BatchSpanProcessor` owns a live export thread, and a pool slot accrued one
per session it served. `RunnerRPC` now calls it on both boundaries —
`session.end` (warm, slot returns to the pool) and `session.shutdown`
(cold) — because a runner reaches one or the other, never both.
`OTelPlugin.reset_for_next_session` still keeps the provider, and its
docstring no longer claims `shutdown()` does the teardown "at slot end":
that was true only of an instance that survived the slot, which the
per-bootstrap runtime makes false. A warm-path failure joins `errors`,
which stops the daemon pooling that slot — deliberate, since a shutdown
that raised is exactly when the export thread may still be running.

Not verified here: no kernel, as with #1023 and #1033. Every confinement
fact is exercised against a fabricated `attr/current` tree and in-memory
pool state. Nothing proves the kernel behaves as #1023 describes; it proves
the framework stops handing the kernel a slot it cannot re-confine, and
that when it refuses one it says which threads it refused.

Also not addressed: the comment on #1100 records that the WS path
**provisions the AppArmor profile ~370 ms AFTER** the first runner has
already spawned unconfined, so a session that asked for confinement gets an
unconfined first runner by construction — which is what poisons the slot on
a deployment that believes it runs confined throughout. The admission gate
is still the right place for the fix, because it protects against ANY
unconfined session sharing a daemon; whether the first runner should wait
for provisioning is its own change.

### A Tmpdir Two Modules Named Differently (#1171)

`RunnerSpawner` decided where a runner's temp files go; `AppArmorManager`
rendered the rule that grants them. One fact, two modules, and after #1037
they stopped agreeing:

| side | keyed on | value |
|---|---|---|
| `TMPDIR` (`runner_spawner.py:319`) | the **session** | `/tmp/jaato-20260921_053346` |
| the profile's tmp rule (`apparmor.py:623`, rendered with `confinement_id_of`) | the **boundary** | `/tmp/jaato-runtime-cdd58a0cee68` |

They agreed only while the profile was named after the session, which #1033
/ #1037 deliberately stopped doing. So a confined runner's first temp-dir
resolution was denied, every fallback (`/tmp`, `/var/tmp`, `/usr/tmp`, `/`)
was denied by default, and the bootstrap died with `[Errno 2] No usable
temporary directory found` — with the kernel logging the other half as
`apparmor="DENIED" operation="mknod"`.

**No release caused it.** `apparmor.py` and `runner_spawner.py` are
byte-identical from 0.16.0 through the fix; `_session_tmpdir(session_id)`
predates 0.16.0 and so does `confinement_id_of`. What changed is how often
a session reaches the path where the disagreement bites.

**The pool was hiding it, and hiding a second defect underneath.**
`jaato_server/shared/plugins/sandbox_utils.py` resolves `tempfile.gettempdir()` at
**module scope**, reached through `cli` / `file_edit` / `filesystem_query`
during `registry.discover(tier_filter="runner")`. `gettempdir()` PROBES —
it creates and deletes a file — and caches the winner in a module global:

| path | when the probe runs | consequence |
|---|---|---|
| **pool slot** | in the **template**, unconfined, before the fork | resolves `/tmp`, every slot inherits the cache, and `TMPDIR` is **never read again** |
| **cold spawn** | after `runner/__main__.py` calls `aa_change_profile` | resolves under the profile — the reported crash |

So the session-scoped `TMPDIR` was **inert on the default path and wrong on
the other**: it works nowhere. And the surviving path was latently broken
too, because a slot's inherited `/tmp` is a directory the base profile does
not grant either (the broad `/tmp/jaato-*` grants are in the **isolated
sub-runner** profile, not the base) — it survived only for as long as
nothing in a confined session wrote a real temp file.

**Both halves were then measured on the enforcing host** (#1171), with
`aa-exec` into the live profile — which is the one thing this repository's
CI cannot do:

```
aa-exec -p jaato-ws-runtime-cdd58a0cee68 -- python -c 'import tempfile; tempfile.NamedTemporaryFile()'
  -> FileNotFoundError ... ['/tmp', '/var/tmp', '/usr/tmp', '/root']
  with TMPDIR=/tmp/jaato-20260921_060114
  -> FileNotFoundError ... ['/tmp/jaato-20260921_060114', '/tmp', ...]
```

So the pool path's inherited `/tmp` is denied, the session-scoped `TMPDIR`
is denied, and the session directory had never been created at all. The
surviving configuration was surviving on the absence of a temp write.

**Four changes, and the first two are both load-bearing.**

| # | Change | Why alone it is not enough |
|---|---|---|
| 1 | `jaato_server/server/confinement_id.session_tmpdir` — `/tmp/jaato-<confinement_id>/<session_id>` | fixes cold spawn; the pool path has already cached `/tmp` |
| 2 | `_pin_session_tmpdir` assigns `tempfile.tempdir` in the runner, after confinement, before any plugin import | fixes the pool path; without (1) it would pin a path nothing grants |
| 3 | `sandbox_utils` tolerates a failing resolution | it classifies paths and never writes one — a probe should not decide whether a module can be imported |
| 4 | the daemon creates the directory in `spawn_session_runner`, before either branch | `RunnerSpawner.spawn` mkdirs only on the branch that calls it, and the confined runner cannot make the boundary directory itself |

**Nesting rather than flattening.** Keying `TMPDIR` on the confinement id
alone would restore agreement and throw away what Phase 5 bought — two
concurrent sessions of one cascade would share a directory. The session
keeps one of its own, nested; the existing `/tmp/jaato-{id}/** rw` rule
already covers it, so the template is untouched. The alternative of granting
`/tmp/jaato-*` on the base profile is refused: it would open every session's
tmpdir to every other session on the host to fix a naming disagreement.
An unconfined runner keeps the pre-#1171 path exactly.

**The pin sets the module global rather than the env var**, because an
assignment cannot be denied where a probe can, and because `TMPDIR` reaches
a pool slot through no channel at all — it is set at `execvpe` and a slot is
forked. `os.environ` is set too, so the subprocesses the model drives
inherit the same answer instead of resolving one of their own.

**The guard needs no kernel, and that is the point.** Every confinement test
in this tree stubs `is_available()` and `apparmor_parser` because CI has no
AppArmor LSM, so nothing in CI can observe an AVC. But the agreement is a
property of two strings: `test_runner_tmpdir_matches_the_profile_grant_1171`
renders a profile for a boundary, asks the spawner for that session's
`TMPDIR`, and checks the second is covered by the first — with the pre-#1171
path asserted **not** granted, so the test cannot pass vacuously. It would
have failed the day #1037 landed. The old guard,
`test_runner_session_tmpdir.py`, names this exact failure in its own
docstring and asserts only one side of it.

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

### A Profile the Process Wore and Two of Its Threads Did Not (#1023)

`aa_change_profile` confines the **calling task**, not the process.
`/proc/self/attr/current` resolves to `/proc/<pid>/attr/current`, which
reports the label of the task whose tid == pid — the **main thread**. Every
confinement check in this tree read exactly that path:

| Reader | Saw |
|---|---|
| the post-transition readback (`runner/bootstrap.py`) | `(enforce)` ✅ |
| the idempotency check (`runner/session.py`) | `(enforce)` ✅ |
| `sandbox_mode` in the session record | `"apparmor"` ✅ |
| an operator running `cat /proc/<pid>/attr/current` | `(enforce)` ✅ |

So a worker thread created **before** the transition kept its `unconfined`
cred for the life of the pool slot, and nothing in the framework could see
it. Confirmed live on a fully enforcing host: **2 of 5 running session
runners**, each with a `runner-rpc-work` thread in the unconfined set.

`session.bootstrap` running synchronously on the reader thread was
necessary and **not sufficient**, which the dispatch site's own comment had
stated as a premise (*"main confines BEFORE any worker spawns"*).
`ThreadPoolExecutor` spawns workers lazily on first submit, and a slot
fields RPCs before its first bootstrap — `session.end` from the previous
session of a cascade, a dispatch-watchdog probe — so the population is
routine. Executors reuse workers and `TRAIT_SLOT_SCOPED` instances are
deliberately carried across session boundaries, so it is durable and
**re-accumulates** after any manual clear.

The exposure is not the notebook, which is only the canary (the one
subprocess surface that inherits passively, so the one whose failure is
visible — and its failure was the audit-tier fail-safe working). It is
**every in-process tool** dispatched onto such a thread: `readFile`,
`file_edit`, `glob_files`, the notebook `local` backend, any in-process
plugin. AppArmor restricts tasks, so those run bounded only by the
application-layer containment heuristic, leave no AVC, and the session
record asserts `sandbox_mode: apparmor` throughout.

**Two changes, and the second is the one that makes the first checkable.**

| | What | Where |
|---|---|---|
| retire | `RunnerRPC.recycle_worker_pools` replaces both executor OBJECTS, so every later RPC spawns a fresh worker under the confined cred | `runner/rpc.py` |
| verify | `verify_thread_confinement` walks `/proc/self/task/*/attr/current` and refuses the bootstrap on divergence | `runner/bootstrap.py` |

An existing thread cannot be repaired — the kernel enforces
`current != task -> -EACCES` on an `attr/current` write, so no thread can
confine another — it can only be retired. Both run on the **idempotent**
path too, which is deliberate: a slot serving its second session of a
cascade under the same profile takes that path, and a worker created before
the slot's *first* bootstrap is unconfined on it.

**Nothing in flight is dropped.** The recycle runs on the reader thread
inside the synchronous bootstrap handler, which is the only submitter, so
no *new* work can arrive while it runs. What may still be executing is a
pre-bootstrap RPC, so the old executors are shut down with `wait=False` and
**without** `cancel_futures`: cancelling would leave the daemon waiting for
a response nobody writes, and waiting could deadlock against a task blocked
on an `outgoing_call` that only the reader thread can resolve. A worker
draining its last task is therefore briefly alive holding the old cred,
which is why verification re-scans across a short grace window before
calling divergence durable.

**Divergence fails the bootstrap, and only positive evidence counts.** A
label read successfully that names a different profile is proof that code
in this process runs outside the boundary the record claims. A label that
could not be read proves nothing — a restricted `/proc`, an unusual
container, a profile template predating the task-dir grant — so that is
logged and the session continues. Failing closed on *absence of evidence*
would take down every session on a host whose `/proc` it merely could not
read. The alternative for real divergence — log an ERROR and proceed — is
the incident state itself, and #1013's proposed `require` mode would
certify such a runner. An operator who cannot tolerate the refusal already
has an honest opt-out: an empty `profile_name` runs the session unconfined
and the record then claims nothing.

**Template v32** adds `/proc/*/task/ r,` to the base and isolated
sub-runner bodies — the *directory listing*; the per-tid `attr/current`
reads inside it have been granted since v15. It reveals tids and nothing
else (the v30 `audit deny` on per-tid `environ` / `mem` / `pagemap` /
`auxv` / `cmdline` is unaffected, a deny beating an allow at any
specificity). Without it the walk still runs — it falls back to
`threading.enumerate()`, which needs no grant because it reads no file and
sees every thread the interpreter created, i.e. every thread this defect is
known to produce — so a runner confined by an older profile is checked
rather than unchecked, and pays one denial AVC per bootstrap. The scan
reports which route it took, because that bounds what it could have seen.

**Checking a deployment** (any runner whose threads are not uniformly
labelled is exposed; `/proc/<pid>/attr/current` alone will say it is fine):

```bash
for p in $(pgrep -f 'server\.runner'); do
  proc=$(tr -d '\0' < /proc/$p/attr/current)
  for t in /proc/$p/task/*; do
    lbl=$(tr -d '\0' < "$t/attr/current")
    [ "$lbl" = "$proc" ] || echo "DIVERGENT pid=$p tid=${t##*/} proc='$proc' thread='$lbl'"
  done
done
```

Not done here: giving the notebook kernel an explicit `//child` transition
(the issue's fix 4). It is defence in depth against a *future* passive-`ix`
spawn path, it does not address the in-process exposure above, and it
changes what every notebook cell runs under on confined hosts — which
cannot be exercised without an enforcing kernel. It belongs in its own
change.

### A Profile Attached and a Kernel Enforcing Nothing (#1014)

#1023 asked **which task** is confined. This is the other axis of the same
readback: **which mode** the kernel is applying. `JAATO_APPARMOR_COMPLAIN=1`
stamps `flags=(complain)` on the whole profile chain — base, `tool_hat` and
`//child` alike — and a complain-mode profile **logs each denial and allows
the syscall**. There is a profile and there is no boundary.

Four of the five places that asked "is this session confined?" answered
without consulting the mode — two by prefix-matching the profile NAME, one
from the fact that provisioning succeeded, one from the presence of a
callback:

| Reader | Under complain |
|---|---|
| the post-transition readback (`runner/bootstrap.py`) | **"confined"** — the comment even read `# e.g. "jaato-ws-... (enforce)"` while the match accepted `(complain)` |
| the idempotency check (`runner/session.py`) | **"already confined"** |
| `sandbox_mode` in the session record | **`"apparmor"`** — mode never consulted |
| `interactive_shell.require_confinement: true` | **satisfied** — the transition into `//child (complain)` succeeds |
| `notebook/kernel_sandbox.apparmor_enforced_profile` | correctly reports no boundary |

The fourth row is the sharpest: `require_confinement` is the strictest
fail-closed knob in the tree — *"Refuse to spawn at all when no AppArmor…"* —
and it passed while the kernel blocked nothing. The fifth is why the only
visible symptom was a notebook cell failing to `import numpy`.

And it was **silent**: grepping `complain` against `logger|warn` in
`jaato_server/server/apparmor.py` returned nothing — no WARNING at generation, at
provisioning or at startup, while every other weakened boundary here
announces itself (`scrub_secret_env: none`, `--ws-unsafe-no-auth`,
`notebook.allow_uncontained_exec`).

**One definition, in one module.** `jaato_server/shared/apparmor_label.py` parses an
`attr/current` value into profile + mode and is the only place either
question is answered. Pure stdlib with zero jaato imports, the shape
`jaato_server/shared/runtime_limits.py` already has, because `jaato_server/shared/` cannot import
`jaato_server/server/` and `jaato_server.server.runner.bootstrap` must stay importable before plugin
discovery — that is the bootstrap's one documented cross-import, and a
guard test pins the module's import set. `apparmor_enforced_profile()` is
unchanged in behaviour and now delegates; it was the one right answer, and
lifting it out is what stops the other four carrying a fifth and sixth
opinion.

**Two questions, deliberately kept apart.** `profile_name_ignoring_mode` is
mode-TOLERANT and named so it cannot be read as an enforcement assertion.
Its callers ask *which profile is this task in* — #1023's per-thread
comparison, and the bootstrap idempotency skip — where a complain-mode
label is not divergence, and reading it as one would make the diagnostic
unusable. `AppArmorLabel.enforced` is the only predicate that may stand
behind a claim that a boundary exists. A label with **no** `(mode)`
annotation is not enforced: absence of evidence is not a boundary.

| Ask | Change |
|---|---|
| one predicate | `confine_to_profile` classifies the mode; the idempotency skip stays mode-tolerant and says so; `interactive_shell` and the notebook resolve through the same helper |
| record the mode | `sandbox_mode` gains **`apparmor-complain`** beside `apparmor` and `soft` |
| announce it | complain-mode generation WARNs once per daemon, naming the env var; `confine_to_profile` and the idempotency path WARN per session |
| `require_confinement` means enforce | an installed child transition is necessary and no longer sufficient — the spawning thread's label must be `(enforce)` |

**A value, not a field.** `apparmor-complain` widens an existing string's
vocabulary rather than adding a record key, so the record version is not
bumped. The record's `version` does have a reader — `deserialize_session_state`
— and it gates on the **major** (`1.x` / `2.x`), which is exactly what a field
gaining a value is indifferent to; 2.10 → 2.11 would have been cosmetic. An
older reader comparing `sandbox_mode == "apparmor"` reads it as "not
confined", which is TRUE and is the safe direction. In-tree readers ask through
`sandbox_mode_is_apparmor` ("was a profile provisioned at all" — the revive
gate, which re-arms confinement for either mode because the mode is
re-decided at the next provisioning) or `sandbox_mode_is_enforced` ("was
there a boundary" — the question the field could not answer before).

**Complain mode is announced, not refused.** It is a documented diagnostic
(harvest the missing-rule set in one cascade run, then ship targeted
grants), and refusing to start under it would delete the diagnostic. What
the default does instead is stop lying: the readback's leading words no
longer say "confined", which is what made #1014's evidence unfindable — the
truth sat in the parenthetical of `runner confined to AppArmor profile X
(kernel reports: X (complain))`, and nobody greps a parenthetical.

**The hook #1013 needs.** `confine_to_profile(..., require_enforce=True)`
raises `ConfinementModeError`. #1013 names this readback as where
`JAATO_APPARMOR_BEHAVIOR=require` would assert that the kernel applied the
profile; built on the mode-blind version, `require` would have been
satisfied by the exact posture it exists to refuse. That is why ask 1 is a
prerequisite rather than a parallel cleanup — the flag itself is not wired
to an env knob here, which is #1013's own change.

Not done here: no kernel-side verification. This container carries no
AppArmor LSM, so every site is exercised from a fabricated `attr/current`
value — which is the point of routing the decision through one parser, and
is also the limit of what was checked. `JAATO_APPARMOR_COMPLAIN` stays a
`host`-scoped env var with no typed profile key; it is a whole-daemon
diagnostic, not a per-session knob.

### Binary Media Chunks (delivery)

Binary content (audio, images, PDFs) moves in three directions, and they are
NOT the same path: **inbound** (content the model looks at), **outbound** (the
model emits speech), and **tool -> client** (a tool produces bytes a *person*
consumes; the model may never see them). See
[Binary Media Chunks](docs/design/binary-media-chunks.md).

**One chunk primitive.** `StreamChunk` (`jaato_server/shared/plugins/streaming/protocol.py`)
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
`TurnCompletedEvent` and never called `jaato_server.server.send_message` — while
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

**A tier gets what it ASKED for, not what its model can read (#1001).**
#847's gate had one bound — `provider.supports_modality(kind)`, the active
model's catalog INPUT capability — and a tier declares a second one nobody
consulted. A voice bot heard a Telegram voice note in its `executor` tier
(`google/gemini-2.5-flash`, which accepts audio) and then
`enter_tier("voz")` to *speak* the answer. `voz` is
`openai/gpt-audio` with `modalities: {audio: outbound}` — it speaks, it was
never asked to listen — and `gpt-audio`'s catalog lists audio INPUT
(verified live: `input: ['text', 'audio']`), so the gate kept the recording
and the upstream refused the request. The tier that only needed the text to
vocalise was handed the bytes of the question.

**This is the level at which the question is answerable.** The refusal
named a container (`Invalid value: 'ogg'`), which invites fixing the
*format* — and the framework cannot know which containers a given model
accepts, so a jaato-side allowlist could only ever be stale in one of two
directions, the argument `api_params` already makes about per-model tables.
The tier's declared role is knowable, local, and written by the author.
`_gate_history_for_active_modalities` is where the two bounds meet.

| The active tier | Inbound bound |
|---|---|
| no tier config, or no active tier | the model's capability alone — **unchanged** |
| declares no role of its own | the model's capability alone — **unchanged** |
| a `vision` tier carrying only its IMPLICIT `{image}` role | the model's capability alone — the shim exists to keep pre-`modalities` profiles working, so it must not arm a gate |
| declares any role, either direction | the model's capability **∩** its own `inbound_modalities` |

`ModelTierConfig.gating_inbound_modalities` answers that question (`None` =
"no opinion", distinct from `frozenset()` = "accepts no non-text input"),
and `JaatoSession._modality_refusal` intersects it with the model —
**narrow, never widen**, the most-restrictive-wins shape
`runtime_limits.max_parallel_tools` uses: a tier declaring `audio: inbound`
on a text-only model still withholds.

**The note names the bound that refused.** A model on `openai/gpt-audio`
told "the active model can't view audio content" has a true fact to
contradict and spends a turn contradicting it, so a tier refusal gets its
own note (`_build_tier_role_withheld_note`) saying the TIER declares no
inbound role, that the model itself can read it, and which tier does
declare it — while a model refusal keeps #847's wording unchanged. The
trace line names both sets for the same reason.

Deliberately NOT done: evicting consumed media at tier entrance. That is a
context-SIZE concern (#850's territory), it is destructive to stored
history where this gate filters a per-request copy, and per-tier
consumption tracking is state nobody has a failing case for — the gate
above is what fixes the reported failure.

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

> **Binary bypasses the formatter.** `jaato_server/server/core.py` `on_tool_output` runs text
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
| encoding | `default=str` reached bytes | `jaato_server/server/runner/json_codec.py` — bytes become `{"__bytes_b64__": ...}` and decode back to bytes; `str` stays the fallback for genuinely diagnostic objects (a datetime, an enum). Used by **both** ends, so daemon→runner bytes stop raising `TypeError` too |
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

Tool names reach the model as hashed ids (`t_<8 hex>`, `jaato_server/shared/tool_id_map.py`)
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

### A Turn Ends Because It ENDED, Not Because It Was Billed (#881)

A completion-gated session did its work, the agent called `signal_completion`,
the payload was accepted and delivered as `AgentCompletedEvent` — and then **no
terminal event was ever emitted**. `Session.complete()` / `.ask()` / `.stream()`
waited out their own timeouts. Nothing was logged on either side: the daemon was
content, the files were written, and only the driver was stuck.

`_turn_accounting` is a **usage ledger**. A turn lands in it only when the
provider reported tokens, and every consumer of `len()` reads it that way — the
`turns` figure in `get_context_usage`, `get_consumption`'s unattributed-turn
reconciliation, the persisted `turn_count`. `rpc._forward_post_turn_hooks` —
the **single site** that fires both halves of the terminus
(`on_agent_turn_completed` → `TurnCompletedEvent`, `flush_session_quiescent()` →
`SessionTerminatedEvent`) — gated on that ledger growing. So a **usage** fact
stood in for the **lifecycle** fact "a turn ran":

| `plugin_configs.echo.usage` | outcome |
|---|---|
| present | `TurnCompletedEvent` immediately; `complete()` returns |
| absent | `AGENT_COMPLETED`; **no** `TURN_COMPLETED`, **no** `SESSION_TERMINATED` |

**The gate was not buying what its docstring claimed.** It was justified as
refused-turn suppression, and a refused turn *is* suppressed — but by
`send_message` returning at the budget gate **before** the chat loop, so no
`turn_data` is ever built and the append site is never reached. The one thing
`total > 0` uniquely suppressed was a turn that ran and was not metered.

So the two facts are separated rather than merged. `JaatoSession._record_turn_ran`
is the one place that closes a turn: it always bumps `_turns_ran` and stamps
`_last_turn_ran`, and appends to the usage ledger only when tokens were
reported. `get_turns_ran()` / `get_last_turn_ran()` are what the fan-out reads.
Option 2 of the issue — drop the gate, append every turn — is deliberately **not**
taken: it would silently change what all three `len()` consumers mean. Option 3
— move only the quiescence flush — is deliberately not taken either: it fixes
`complete()` and leaves `ask()` / `stream()`, whose only terminus is
`TURN_COMPLETED`, still hanging.

Three properties, each attached to a way it could go wrong:

- **`turn_number` follows the lifecycle count.** `jaato-tui/agent_registry.py`
  indexes a list by it, so sourcing it from the ledger gave every unmetered turn
  the same ordinal and the second overwrote the first.
- **A session without the accessors gets the PRE-#881 behaviour**, not an
  exception and not an unguarded double emission. `RunnerRPC._turns_ran_snapshot`
  falls back to the ledger length for a duck-typed double or an out-of-tree
  session class; a real `JaatoSession` never takes that path.
- **The in-process path (`jaato_server/shared/jaato_client.py`) already fired
  unconditionally**, with a comment saying why ("Some providers report 0 tokens;
  skipping the hook would leave buffered content stuck in the pipeline"). The
  two paths disagreed, and the runner path — the default — was the wrong one.
  That path is otherwise unchanged here; it still re-emits the previous turn on
  a *refused* send, which the runner path has always suppressed.

**Nothing about this is `echo`-specific.** Any provider that completes a turn
without reporting usage reaches the same state — a stream that never delivers a
usage frame, a gateway that strips the field, a zero-cost cached turn. `echo` is
where it is *guaranteed*, and it is what every new harness reaches for first
because it is credential-free and deterministic.

**And the suite could not see it.** `jaato_sdk/conformance/` stayed green for
the whole life of the defect precisely because every conformance profile passed
`usage=TURN_USAGE`; a suite whose every
profile is metered is structurally unable to catch this class however many
scenarios it runs. `conformance-unmetered` is the fifth profile, and the only
one that differs from another by a **missing** key.

**The silence is now announced (#688 item 3).** A turn that carries no tokens
logs a WARNING **once per session**, naming the provider and model, and saying
what stops working: the consumption report reads empty and a `budget_control`
ceiling on `tokens` / `usd` is fed zero, so a run that looks capped is uncapped.
Deliberately worded as *this turn carried no tokens* rather than *the provider
reported nothing* — nothing in the tree can yet tell a reported zero from an
unreported one (`TokenUsage` starts at all-zeros and is only overwritten when a
frame arrives, so the two share a value), and asserting the stronger claim would
be a statement the data does not support. Giving them separate representations,
and deciding what `budget_control` should do with "unknown", is **#688 items 1
and 2** — done in the section below, which is what lets that warning's wording
finally be strengthened.

### Unreported Usage Is Not Zero Spend (#688)

`TokenUsage` started all-zero and was only overwritten when a usage block
arrived, so **two different facts shared one value**: a provider that measured
the call at zero, and a provider — or a proxy in front of it — that sent no
`usage` at all. `budget_control` enforces its `usd` / `tokens` ceilings from
exactly that data, so an unmetered upstream **silently disabled spend
enforcement**: the tracker was fed zero, no dimension advanced, no rung fired,
and a run that looked capped was uncapped. Failing open on a spend control is
the wrong direction, and it failed open quietly.

The exposure is wide by construction — `nim`, `nebius`, `ovhcloud`,
`doubleword`, `lmstudio`, `tensorrt_llm`, `triton`, `vllm`, `zhipuai_openai`,
plus `openrouter` fronting 300+ upstreams and any corporate gateway in front of
those: precisely the "approximately OpenAI-compatible" endpoints where usage
reporting is least reliable.

**`TokenUsage.reported` is the distinction**, and the precedent was already in
the same dataclass: `cache_read_tokens` documents `None` as *"provider reported
nothing", distinct from a reported zero*.

**It defaults `True`, deliberately.** A seam that has not been migrated — an
out-of-tree provider, jaato-premium, a third-party adapter — behaves exactly as
it did before the field existed, rather than being marked unknown and having a
policy applied that its author never saw. The danger was never the default but
a **half**-migration, so all 30 in-tree placeholders are migrated in the same
change: the default protects strangers, not this repository. Each converter
already had the right shape — construct, early-return on absent usage, fill —
so the seam is `TokenUsage(reported=False)` at the top and `reported = True`
after the guard. `anthropic` is the one provider whose `message_delta` route
MUTATES the accumulator rather than replacing it, and it is marked on both
routes; that is exactly the shape the issue cites two upstream fixes for.

**What an unmeasured turn costs is the profile's choice** —
`budget_control.on_unmetered`:

| Policy | Effect |
|--------|--------|
| `estimate` (default) | charge `tokens` from a local estimate; leave `usd` **untouched** |
| `halt` | stop the session, the same stop an `abort` rung uses |
| `ignore` | the pre-#688 behaviour, chosen explicitly |

The asymmetry is the house rule `_budget_observe_response`'s own docstring
already stated — *a budget must never hard-stop on a number it invented*. The
two dimensions are not alike: GC already estimates token counts for its own
threshold, so that quantity is one the framework routinely computes, while a
dollar figure derived from guessed tokens is a price nobody quoted. The
estimator is deliberately the **same** one GC uses; a budget and a GC threshold
disagreeing about the size of one turn is very hard to see from either side.

**`halt` is opt-in, not the default.** As a default it would, on upgrade, take
down every deployment sitting behind a usage-dropping proxy. Stated cost of
that choice: a profile whose ONLY ceiling is `usd` is still unenforceable
against an unmetered provider, because `usd` is never fed an estimate —
`halt` is the answer there, and is why the knob exists. A test asserts that
limitation rather than leaving it implied.

**One check, one door.** The vocabulary is validated in `__post_init__` only.
An earlier draft also validated inside the `from_dict` helper, and the
meta-guard correctly reported **both** reversions as decorative: each copy
caught what the other would have let through, so neither could be shown to do
anything. Duplicated validation is not defence in depth when it makes every
copy individually unreachable.

Tests: `jaato_server/shared/tests/test_unreported_usage_is_not_zero_688.py` — 28 cases,
six REVERSIONS. The wire-level cases drive the **real** streaming loop of three
providers against a usage-omitting stream, which is what could not be exercised
when this was first sized (`openai` was not installed then). The three-way
assertion is the point: no usage, a genuine zero, and real numbers must produce
three distinguishable results.

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
  read as free. A reported `0` is a measurement and is shown — which no
  OpenAI-shaped provider could actually say until `reported_cache_count`
  replaced the per-seam `and value > 0` gates that folded a reported
  `cached_tokens: 0` into `None` (`_openai_compat`, `openrouter`, `nebius`,
  the OpenAI Responses wire, `kimi`, `google_genai`). `TokenUsage` and
  `compute_cache_hit_percent` had both documented the distinction for
  longer than anything could feed it.
- **And a TOTAL is where that rule is easiest to break by arithmetic.**
  A pooled cache-hit rate sums the numerator over the bindings that HAVE a
  cache and the denominator over all of them, so a binding reporting no
  cache dimension contributes zero hits and its whole uncached input.
  Measured on a two-tier voice session: a caching `executor`
  (`google/gemini-2.5-flash`, 76.41%) pooled with a non-reporting `voz`
  (`openai/gpt-audio`, 224,423 uncached input tokens) to **54.66%** — a
  session-wide "efficiency" figure no binding had, which moves with the
  tier mix rather than with anything a reader can act on. So `totals`
  **withholds** `cache_hit_percent` whenever part of its denominator is
  unmeasured and publishes `cache_hit_basis` instead: how many bindings
  were measured, how much uncached input could not be, and the rate over
  the subset that could. A homogeneous pool is unchanged, and a pool that
  measured nothing explains nothing — `cache_read_tokens` is already
  absent there, and a basis would explain the absence of a number nobody
  expected.
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
| `TRAIT_UNTRUSTED_CONTENT` | `"untrusted_content"` | Tool **result** carries content from the open internet or a third party (`web_fetch`, `web_search`, `subagent`, MCP servers). The session marks the result and the provider converter wraps the model-facing text in the `⟦UNTRUSTED-EXTERNAL-CONTENT⟧` boundary, so injected instructions in a payload read as data. Defense-in-depth, complementing egress allowlisting and permission gating. |
| `TRAIT_UNTRUSTED_SCHEMA` | `"untrusted_schema"` | The tool's **own declaration** — name, description, and the `description` fields nested in `parameters` — was authored by a third party rather than the framework. Independent of the trait above: `web_fetch` returns untrusted content but its description is framework text, while an MCP server authors both. Matters because a description lands in the *trusted* region of the system prompt. A plugin declaring this **must** pass its schemas through `sanitize_untrusted_schema()` (wraps the description, defangs nested ones, forces the name onto `[A-Za-z0-9_-]{1,64}`). Enforced by `test_untrusted_schema_is_sanitized.py`. |

**How it works:**
1. Tool schemas declare traits: `traits=frozenset({TRAIT_FILE_WRITER})`
2. Session queries `registry.get_tool_traits(tool_name)` to decide enrichment strategy
3. Enrichment plugins (LSP, artifact_tracker) extract file paths generically from the result dict

**Adding a trait to a new tool:**
1. Import the constant: `from ..model_provider.types import TRAIT_FILE_WRITER`
2. Add to the `ToolSchema`: `traits=frozenset({TRAIT_FILE_WRITER})`
3. Ensure the tool result dict includes the required keys (`path`, `files_modified`, or `changes`)

**Defining a new tool trait:**
1. Add a `TRAIT_*` constant in `jaato_server/shared/plugins/model_provider/types.py` with a docstring documenting the contract
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
`apply_text_view_enrichment` (`jaato_server/shared/tool_result_builder.py`) are the two
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

### A Guard That Only Binds When Nothing Needs Bounding

`references` expands a selection transitively: `selectReferences` runs a BFS
from the chosen ids, discovering edges from each node's **body text** (a
catalog id mentioned as a whole word, or a relative path that resolves to
another LOCAL source). What it returns is the **reachable set** — the
transitive closure's row — not a neighbourhood, and its only bound was
`MAX_TRANSITIVE_DEPTH = 10`.

Depth is a **logarithmic control over an exponential quantity**: a frontier
of out-degree `d` reaches `d**k` nodes at depth `k`, so depth 10 binds only
on catalogs larger than `d**10` — 59,049 at `d=3`. Measured on a 200-entry
catalog, varying only how many other topics each document mentions:

| mentions per document | resolved from ONE selection | depth actually binds? |
|---|---|---|
| 1 | 4 | yes |
| 2 | 147 | no (saturates past 10) |
| 3 | **200 — the whole catalog** | no (saturates at 9) |
| 4 | 200 | no (saturates at 5) |

**Three mentions per document is the cliff**, and that is a "see also"
naming three siblings. The guard binds at out-degree 1 — a pure chain,
which is the one shape that never needed bounding. Every resolved
reference is manifested to the model *and* path-authorized, so this is a
context and authorization-surface question, not only latency.

An edge is a **mention**, not a link: nobody authors it. An index page, a
changelog or a table of contents gives one node an out-degree of the whole
catalog, and a DIRECTORY reference inherits every id mentioned anywhere
beneath it (`_get_reference_content` rglobs and concatenates).

**Three changes, and only the third alters behaviour.**

| # | Change | Effect |
|---|---|---|
| 1 | `sorted()` on the frontier and on each node's discoveries | reproducible expansion |
| 2 | one-pass id matching | 130x faster, identical results |
| 3 | `max_transitive_references` + a truncation record | a bound that binds |

**Sorting had to land before the bound, not with it.** `pending` and
`new_mentions` are sets, so iteration order varies across processes
(string hash randomisation). Unbounded that only shuffled the manifest;
with a cap it decides **which** references survive — measured on the same
catalog capped at 25, **six of the 25 differed** between two
`PYTHONHASHSEED` values. Same argument this file already makes for sorting
the `spawn_subagent` profile enum: the output reaches the prompt-cache
prefix.

**The matcher was O(catalog x content).** `_find_referenced_ids` built a
word-boundary regex per catalog id and searched the whole body with each.
On one 386,029-char directory reference against a 200-entry catalog:
**1.5544s -> 0.0119s**, identical result sets. It now tokenises the content
once on the pattern's own separator class and intersects. An id that
*contains* a separator (`foo bar`, `a(b)`) is unreachable by tokenising and
keeps the original per-id matcher — dropping those would silently narrow
the graph for catalogs that use such ids. Equivalence is pinned against the
original implementation, and by 400 randomised trials over an alphabet that
includes the separator characters.

**The bound defaults to UNBOUNDED, and is announced instead.** Capping by
default would silently cut neighbourhoods every existing workspace relies
on; the safe-by-default posture `scrub_secret_env` takes is right for a
credential and wrong for a relevance heuristic nobody has measured against
a real catalog. So the default is unchanged behaviour plus a WARNING, once
per session, naming the knob when one expansion resolves 50+ references:

```yaml
plugin_configs:
  references:
    max_transitive_references: 25   # unset or 0 = unbounded
```

A malformed value falls back to unbounded, never to an invented ceiling —
silently applying a limit nobody configured would cut a neighbourhood for a
reason no operator could find.

**A cut neighbourhood says so.** `selectReferences` returns a `truncated`
record naming the limit, the count resolved and the depth it stopped at —
and deliberately **no "dropped" figure**, because the walk stops early and
how many more it would have found is unknown. A fabricated count is worse
than an absent one. The note is the load-bearing part: a model handed a
silently-cut neighbourhood reads absence as *"no such reference exists"*,
which on a knowledge graph is exactly the wrong conclusion and is
unfalsifiable from its side.

**Deliberately not done.** Ranking the frontier — keep the *nearest* 25
rather than the first 25. The machinery nearly exists (`score_sources`
takes a vector and does not care where it came from), but it is reachable
only where a vector index has been generated, and `initialize()` skips the
embedding provider entirely otherwise. A bound has to work for everyone;
ranking is a refinement for workspaces that have embeddings. Also not done:
typed edges, which would let `rel` decide what counts as adjacency rather
than "the string appeared".

### Plugin-Level Traits

Plugins themselves can declare **plugin-level traits** via a `plugin_traits` class attribute (`FrozenSet[str]`). These work like tool traits but identify *plugin* capabilities rather than individual tool behaviors.

**Currently defined plugin traits:**

| Constant | Value | Contract |
|----------|-------|----------|
| `TRAIT_AUTH_PROVIDER` | `"auth_provider"` | Plugin provides interactive authentication for a model provider. Must also expose `provider_name` property identifying which provider. |
| `TRAIT_SESSION_PERSISTENT` | `"session_persistent"` | Plugin state must outlive an unload/reload of the SAME session. Must implement `get_persistence_state()` / `restore_persistence_state()`; `SessionManager` snapshots into `metadata['plugin_states'][<name>]`. |
| `TRAIT_SLOT_SCOPED` | `"slot_scoped"` | Plugin INSTANCE survives the cascade session boundary — the runner carries it across sessions served by the same pool slot instead of constructing a new one. `shutdown()` then means slot teardown, not session teardown. See [Slot-scoped plugin lifetime](#slot-scoped-plugin-lifetime-890). |
| `TRAIT_OUTPUT_MARKER` | `"output_marker"` | Plugin MARKS AI-generated output (Art. 50(2)). Implements `mark_output(OutputPayload) -> MarkResult`; invoked by `JaatoSession._mark_generated_output` at both delivery seams. The framework enforces the two rules a marker must not get wrong: an **unstamped** payload is never offered (relaying is not generating), and a marker that **raises** loses nothing (traced, bytes delivered unmarked). One in-tree implementation, `output_marker`: a `<file>.provenance.json` sidecar. See [Marking Generated Output](#marking-generated-output-1117). |

The four answer different questions and compose freely: `session_persistent`
is "survives THIS session being unloaded and reloaded",  `slot_scoped` is
"survives the NEXT session of the same cascade starting", and `auth_provider`
and `output_marker` are capabilities rather than lifetimes.

**How it works:**
1. Plugin declares: `plugin_traits = frozenset({TRAIT_AUTH_PROVIDER})`
2. Server filters plugins by trait: `TRAIT_AUTH_PROVIDER in plugin.plugin_traits`
3. Among matching plugins, server reads `provider_name` to select the right one

**Adding a plugin trait to a new plugin:**
1. Import the constant: `from shared.plugins.base import TRAIT_AUTH_PROVIDER`
2. Add class attribute: `plugin_traits = frozenset({TRAIT_AUTH_PROVIDER})`
3. Implement the contract (e.g., `provider_name` property for auth plugins)

**Defining a new plugin trait:**
1. Add a `TRAIT_*` constant in `jaato_server/shared/plugins/base.py` with a docstring documenting the contract
2. Update consumers (server, daemon) to query `getattr(plugin, 'plugin_traits', frozenset())`

### A Topic the CLI Could Not Answer and the Daemon Could

`jaato-scaffold explain` introspects the framework installed in the **calling
process**. That is the whole answer while the CLI and the daemon share a
virtualenv, and silently wrong the moment they do not — which is the normal
shape of a deployed application: `jaato-sdk` (and `jaato-server`) in the
application's own `.venv`, driving a daemon owned by a different user over
IPC. There are then TWO installs, and the CLI was answering about the one that
is not serving the sessions.

Measured on exactly that pair, with `jaato-server 0.17.0` present and
`jaato_premium` absent from the CLI's venv:

```
$ jaato-scaffold explain reactors
unknown explain scope 'reactors' — one of: plugins | plugin <name> | ...
```

`reactors` is contributed through `jaato.scaffold_topics` by jaato-premium,
which is installed in the DAEMON's venv. The refusal is indistinguishable from
*no such topic exists*, so it sends a reader looking for a feature they already
have — the failure the entry-point seam exists to remove, one process over.
**The seam is not the gap**: it works, in the process that has the package.
What was missing was a way to ask the other process.

**One dispatch, asked over a socket.** `scaffold.explain` (protocol **1.18**,
answered by one `ScaffoldExplainEvent`) calls
`jaato_server.shared.scaffold.__main__.render_topic` — the SAME function the CLI calls,
contributed topics and contributed `extends` sections included. A second
dispatch daemon-side would be free to disagree with the CLI's about what a
topic answers, which is this defect reproduced over a socket rather than fixed;
the guard is an AST scan, not a behavioural probe, because a second dispatch
agrees right up until somebody edits one of them.

| When | What happens |
|---|---|
| a topic this venv HAS | rendered locally, **no socket touched** |
| a topic it does not, daemon reachable | the daemon's rendering, plus a line saying whose install produced it |
| a topic neither has | **the LOCAL refusal**, plus a note that the daemon was asked and does not serve it either |
| a topic it does not, no daemon | the local refusal, unchanged |
| `--connect [SOCKET]` | that daemon, always — including its refusal, and an unreachable one is this command's failure |

**The second row is a correction, and the first draft had it wrong.** Letting
the daemon's refusal replace the local one looks symmetrical and is a WRONG
ANSWER: the two refusals list different topic sets, and the daemon's omits
every topic the reader's own install has. A reader who typos on a machine that
happens to run a daemon was shown that list and would conclude a topic they can
use does not exist. So on the fallback the local list prints — it is the one
they can act on without a socket — and the note keeps *asked and not there
either* distinguishable from *nobody looked*, which is the `reached`/`ok`
distinction one layer out. `--connect` is the deliberate exception: there the
reader named that daemon, so its list is the one they asked about. It surfaced
as a pre-existing test whose outcome changed depending on whether a daemon
happened to be running on the machine — the suite is now green with one running
and with none.

**A working local answer never grows an egress.** Quietly giving an offline
introspection a network call changes what running it means — the argument
`explain releases` already makes about being its own topic rather than a facet
of `explain dependencies` — so the fallback is gated on
`_scope_renderer(scope) is None`. A usage error (a topic that needs a name,
given none) is about the caller's own command line and is never taken to a
daemon: it would answer a question nobody posed. Measured: three
locally-answerable topics add **zero** `scaffold.explain` lines to the daemon
log; one unknown topic adds exactly one.

Four properties, each attached to a way asking could mislead:

- **`reached` is not `ok`.** *The daemon answered and does not have it either*
  and *no daemon could be asked* are different facts — one says the topic does
  not exist, the other says nobody looked — and collapsing them reproduces the
  confusion this removes. `RemoteAnswer` carries both, and `error` stays empty
  when nothing was said rather than manufacturing a refusal nobody made.
- **Every answer names the install that produced it.** On this deployment the
  two installs are two machines' worth of different packages, and a reader who
  cannot tell them apart cannot tell which one to change. The byline is never
  omitted; a daemon that named no version degrades it to `unknown version`
  rather than dropping it.
- **The probe never STARTS a daemon** (`auto_start=False`). A report about a
  process the report just created is a report about the wrong process — and on
  this deployment that process is not even the same user's. Verified: with no
  daemon, the command exits 2 with the local refusal and creates no socket.
- **An old daemon is refused, not waited out.** The 1.7 rule applied to a verb:
  a daemon below 1.18 ignores `scaffold.explain` silently, the caller would
  wait out its deadline and report the topic missing — the original defect
  arriving through the fix. Both the refusal and its wording name what the
  daemon *speaks*, because the remedy is upgrading a process the reader may not
  own.

**`ScaffoldExplainEvent.data` is deliberately not typed as an object.** The
in-tree `profile` topic renders an ARRAY of field rows, and wrapping it to
satisfy a narrower field would make the daemon's `--json` differ from the same
command's local `--json` — two installs disagreeing about one topic, which is
what the event exists to stop. The daemon re-encodes with `default=str` (the
CLI's own fallback) so a `Path` or an enum in a rendering cannot produce a
frame the serialiser refuses and a caller waiting for a reply that never comes.
`topics` — the catalog THIS daemon can render — rides **every** outcome,
bound once in the handler's one `answer` door rather than per call site,
because "which topics does the daemon have" is precisely the question a failed
lookup raises and the caller's own catalog is by construction the wrong one.

**There is no `workspace` parameter**, deliberately. A workspace-reading topic
reads the caller's own workspace, resolved daemon-side from the session it is
attached to or the workspace it declared — both entitlement-checked at the
handshake ([Two Principals on One Socket](#two-principals-on-one-socket)).
Letting a read-only report name a directory would add a second, unchecked path
ingress for the sake of a diagnostic.

Guards: `jaato_server/shared/tests/test_explain_asks_the_daemon_that_has_the_topic.py`
(four reversions) and
`jaato-sdk/jaato_sdk/tests/test_explain_topic_refuses_an_old_daemon.py`. Two
drafts of the `auto_start` guard were decorative and the reversion meta-guard
caught both, which is worth recording because they failed differently:
`inspect.getsource` reads the module the editable install pins — the real
checkout, whatever the interpreter's cwd — so it never saw the sabotage; and a
substring test for `auto_start=False` was then satisfied by the *comment* two
lines above the call. It asserts the call by AST now. **A guard on prose is a
guard on nothing.**

Not addressed here: `validate` has no extension seam at all, so a package that
contributes a topic still cannot contribute a *finding* — `jaato-scaffold
validate . --set drive` reports nothing about reactor rules however wrong they
are, in either venv. That is its own change, and the entry-point group
`jaato.premium_reactors` the report names has never existed.

### An Integration Declares Its Own Paths, and Its Own Harness

`jaato-scaffold integration <name>` installs the `jaato-sdk` skill where
another tool looks for skills. Everything tool-specific lives in that
integration's `integration.json` — there is no `if name == "claude-code"`
anywhere in the code, which is what lets a new harness be added without
touching the generic module.

**`target` is one key with two forms.** A string when a harness uses the same
relative path at both scopes; an object when they differ:

```json
"target": ".claude/skills/jaato-sdk"

"target": {"user":      ".pi/agent/skills/jaato-sdk",
           "workspace": ".pi/skills/jaato-sdk"}
```

The rejected alternative was `target` **plus** `user_target` /
`workspace_target` companions. Three keys cost a reader the question of which
are alternatives and which are siblings, put two nulls in every `listing()`
row (and flipped *which* two per integration, so every consumer handled both
shapes), and — the sharp one — let an author declare one scope, forget the
other, and get a **silently wrong path** for the missing one.

**A manifest that cannot say raises.** The old code resolved a missing
`target` to `.jaato-integration-<name>` and its docstring said the caller
reported it. No caller did: the string occurred exactly once in the tree, at
the site that built it, with no reader anywhere — so a forgotten key installed
a real payload to a plausible-looking wrong path. `IntegrationManifestError`
now covers a missing target, an object missing a scope, a non-string path, and
an absolute one (targets are joined onto `$HOME` or the workspace, so absolute
would escape the scope asked for). `listing()` reports such an integration as
`invalid` rather than raising — the bare verb is how an operator finds out
something is wrong, so it must survive the thing being wrong.

**`detect` says how to know the harness is installed, and only its author
knows.** `jaato-doctor` warned once per shipped-but-unapplied integration with
no test of whether that tool exists on the machine. With one integration
shipped that was invisible; with two, every user of the first gets a warning
they cannot clear — the only way to satisfy it is to install a skill for a
harness they do not use, and the noise grows with every integration added.

```json
"detect": {
  "commands": ["claude"],
  "paths": ["~/.claude/projects", "~/.claude/sessions", "~/.claude.json"],
  "why": "Claude Code writes these as it runs; jaato creates only
          ~/.claude/skills/jaato-sdk, so none can come from installing us."
}
```

| Property | Why it is load-bearing |
|---|---|
| **`None` is not `False`** | an integration declaring no `detect` asserted nothing, and still warns exactly as before. Suppression follows an *assertion*, never an inference — absence of evidence is not evidence of absence |
| **the skip is gated on `state == "absent"`** | detection is a heuristic, so the most it may ever do is withhold an optional suggestion. A copy that EXISTS is reported whatever detection says, which keeps `stale` / `edited` / `diverged` drift visible on a machine whose harness was removed after the skill was applied |
| **a `detect.paths` entry may not be an ancestor of the integration's own target** | jaato creates those. Measured: on a host with *neither* harness, installing only our skills brings `~/.claude`, `~/.claude/skills`, `~/.pi` and `~/.pi/agent` into existence — so `~/.pi` would answer "Pi is here" on a machine that has never had Pi, silently restoring the noise the key exists to remove |

That last row is the one error a machine can check
(`manifest_detect_problems`). Whether a path is *truly* harness-owned cannot
be checked here — it is what the author asserts, and `why` is how a reviewer
who does not use that harness judges the claim. `commands` cannot be
contaminated that way (we never put a binary on `PATH`) but is not absolute
either: a harness outside this process's `PATH` reads as absent, which loses a
nudge rather than inventing noise — the safer direction to be wrong in.

**A guard with `REVERSIONS` lives in `jaato-server/jaato_server/shared/tests/`**, whatever
package its subject is in — the meta-suite walks only `jaato_server/shared/tests` and
`jaato_server/server/tests`, so a reversion declared elsewhere is silently unexercised.
`test_doctor_detects_checkout_skew_823.py` is the precedent: it sits there and
targets `jaato-sdk/jaato_sdk/doctor.py`. Widening that walk is not a
mechanical change — `_collect_nodeids` keys by BASENAME and a collision
already exists across the wider set (`test_completion_processors.py`), so
widening would reintroduce the ambiguity #1084 fixed.

### Entry-point Plugin Trust

Out-of-tree plugins are installed as distributions declaring
`[project.entry-points."jaato.plugins"]` (also `jaato.enrichment_plugins`,
`jaato.gc_plugins`, `jaato.cache_plugins`).  `PluginRegistry.discover()`
runs entry points **first**, then the directory scan — and the directory
scan skips any name already registered.  Left unguarded, that made every
built-in overridable by any distribution sharing the venv, silently
(#684).

The policy lives in `jaato_server/shared/plugins/entry_point_trust.py` and is applied
by `PluginRegistry._gate_entry_point`:

| Rule | Effect |
|------|--------|
| **Built-in names are reserved** | The reserved set is the module listing of `jaato_server/shared/plugins/` (read with `pkgutil.iter_modules` — a directory listing, no imports). A foreign entry point claiming one is refused. |
| **Refusal precedes `ep.load()`** | Every decision is made from the entry point's metadata (`ep.name` / `ep.value` / `ep.dist`), so a refused claim never has its module imported. `ep.load()` executes code — being installed must not be enough to run it. |
| **The framework's own declaration is exempt** | jaato-server publishes its built-ins through the same groups; an entry point targeting `jaato_server.shared.plugins.*` is the framework, not a claim. |
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
`jaato_server/shared/scaffold/introspect.py` pass none.  A plugin whose package
declares no `PLUGIN_TIER` is excluded under **any** filter — the
deliberate "annotate or be excluded" contract — so the split ran
straight through the diagnostic: the author installed the
distribution, ran `jaato-scaffold plugins`, saw the plugin listed with
its provenance line, wrote `plugins: [m365]` in a profile, and the
session came up without the tools.  No error, no warning, one debug
`_trace`.  `test_plugin_tier_partition` fails the build on this, but
its walk is an AST scan of `jaato_server/shared/plugins/` and cannot see a
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
only in `jaato_server/shared/session_context.py`, so a third-party plugin either
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

`jaato_server/shared/session_context.py` imports the three functions back, so every
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
  on_unmetered: estimate   # estimate (default) | halt | ignore -- what to do
                           # when the provider reports no usage at all (#688)
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

### A Bound That Was Declared Everywhere and Enforced Nowhere (#1068)

`SubagentProfile.max_turns` was a complete field in every respect but the one
that mattered. It was **declared** (`int = field(default=10)`), **validated**
(non-int and `<= 0` refused), **inherited** most-restrictive-wins — the same
treatment `budget_control.limits` and `max_parallel_tools` get — **serialized**
into and out of session snapshots and the runner RPC payload, exposed on the
wire as `ProfileSummary.max_turns`, rendered by `explain profile`, and
**advertised to the model** by two tool descriptions:

```
'While sessions auto-close after max_turns, explicit closure is preferred
 to free resources immediately.'
```

It was compared against a turn counter **nowhere**. `grep -c max_turns
jaato-server/jaato_server/shared/jaato_session.py` returned **0** — the class that owns the
turn loop had never heard of it — and the four `turns >= config.max_turns`
comparisons in the tree are all `GCConfig.max_turns`, a garbage-collection
trigger that happens to share the name. The four reads of
`profile.max_turns` outside validation, inheritance and serialization were
each stuffing it into a dict for `list_subagents` to report.

So a parent agent that declined to `close_subagent` — reasoning, correctly per
its own instructions, that the session would auto-close — leaked it; #947's
`documentalista` retrying `writeNewFile` **127 times** had `max_turns`
declared and it bounded nothing; and `explain completion` stated in capitals
that **`max_turns` IS the retry budget** for a gate whose retry loop was in
fact unbounded.

**The field is removed rather than implemented**, which is the part worth
recording. The obvious fix — compare `_turns_ran` against it in
`JaatoSession._record_turn_ran` — builds a *second* mechanism beside one that
already works: `budget_control.limits.turns` is fed `turns=1` per turn by
`_budget_observe_turn` on every path, and a `degrade` rung whose `action` is
`abort` reaches `request_stop()`. Two counters for one quantity is the "one
check, one door" failure the reversion meta-guard caught twice while #688 and
#1069 were being written, and it would have been worse here: a `max_turns`
that finally enforced its **default of 10** would stop sessions that
legitimately run longer — #732 measured 44 round-trips in a session declaring
`max_turns: 15`, because nothing had ever stopped them.

| | before | after |
|---|---|---|
| present on every profile | yes, defaulting to 10 | no — `budget_control` is opt-in |
| enforced | **no** | yes, via an `abort` rung |
| `limits` alone | — | observed, never enforced (#947's `budget_limits_without_abort`) |

What replaces it in the docs is the truth: a session takes as many turns as it
takes, and the way to bound one is `budget_control`. A profile that declares
neither a ceiling nor `max_refusals` has an **unbounded** retry loop, which
`jaato-scaffold validate` already warns about and which the removed field
disguised.

Three consequences handled rather than inherited:

- **`removed_profile_key` is its own finding**, not a bare `unknown_profile_key`.
  A profile on disk keeps loading — construction is keyword-explicit, so the key
  is simply ignored — and the warning names the replacement, the posture
  `deprecated_system_instructions` already takes.
- **Protocol 1.9, not 2.0.** Removing `ProfileSummary.max_turns` is the first
  field removal from a versioned wire shape. Both directions still parse (an
  older client's model declares the field with a default, so an absent key fills
  it; a newer client's `extra='ignore'` drops an older daemon's value), while a
  MAJOR bump hard-refuses every client — `server_major != client_major` is an
  unconditional refuse. The rule the entry establishes: removing a field that
  carries a **default** is a MINOR, removing a required one is a MAJOR.
- **The runner RPC allow-list drops the key outright.** `PROFILE_PAYLOAD_ALLOWED_KEYS`
  rejects unknown fields by design, so tolerating a dead one would be a
  permanently inert entry in a security allow-list; both ends of that wire ship in
  the same package, so there is no version skew to tolerate.

### A Rung a Client Could See and Not Read (#1069)

#955 made the ladder observable **in the trace**. This is the same argument
one layer out, and it starts by correcting the premise it was reported under:
*"degrade rungs are observable only in the trace log — no client-visible event
fires when one applies"*. A fired rung has always reached the client.
`_apply_budget_rungs` calls `_surface_budget_event` on both paths, and that
method emits `AgentOutputEvent(source="system")`:

```
[budget[self-enforced] tokens 85%: degraded planner opus -> flash]
```

So the gap is not *a signal*. It is **a signal a client can branch on**, and
one that is not mixed into the stream the client renders as what the agent
said. A bot that already decorates tier switches and memory stores cannot
decorate this without string-matching `[budget[`, and a client that renders
agent output verbatim shows the framework's cost machinery to users as the
agent talking. That is a sharper argument than "nothing is emitted" was.

**`BudgetRungFiredEvent`** (protocol **1.8**) is the typed sibling. The prose
channel is unchanged and still fires — it has consumers, and its docstring
records that it was already broken once (routed through `_ui_hooks`, never
set on the runner path, so every budget decision was silently dropped).

| Field | Notes |
|-------|-------|
| `at_percent`, `action`, `pressure` | the rung and what drove it |
| `origin` | `self-enforced` / `cascade-pushed` — the MECHANISM, carried rather than dropped at the boundary because `_apply_budget_rungs` argues it is the distinction a consumer needs: *I hit my own ceiling* invites a narrower retry, *the shared pot ran out* means the run is winding down |
| `usage`, `driving_dimension` | per-dimension fractions. **Present only when `origin == "self-enforced"`** — a cascade-pushed rung was crossed by the POOL, and publishing this child's own fractions beside the pool's pressure is the exact contradiction the prose line already avoids (*"degrading at 50% (tokens 32%)"*). Absent means "not measured here", never zero |
| `tier_changes` | `{tier: "old -> new"}`, the shape `overlay_tier_table` returns — whose own docstring already named this event as its consumer. What the overlay **did**, not what the rung declared: a tier already bound to that model contributes nothing, and a session with no tier config contributes none |

A rung **skipped** by the backwards-rebind guard emits nothing: it changed
nothing, and telling a user the model was downgraded when it was not is worse
than silence.

**`action: notify`** is a rung that only emits — no rebind, no latch.

```yaml
budget_control:
  limits: {usd: 15, tool_calls: 1500}
  degrade:
    - {at: 60,  action: notify}    # checkpoint: changes nothing
    - {at: 80,  action: notify}
    - {at: 95,  action: finalize}  # advice
    - {at: 100, action: abort}     # the ceiling
```

The issue floated an alternative — treat a rung with neither `model_tiers`
nor an action as emit-only — and it is not available: `DegradeRung.from_dict`
refuses such a rung outright (*"degrade[N] does nothing"*), so the vocabulary
had to grow rather than the bare form being reinterpreted. Naming it is the
better half of that anyway: a bare rung is ambiguous between *checkpoint* and
*author forgot the action*.

Three properties, each attached to a way it could go wrong:

- **`notify` does not latch, and exactly one thing decides that.**
  `TERMINAL_ACTIONS` (`finalize`/`abort`/`escalate`) is what
  `_budget_terminal_action` is gated on. An earlier draft ALSO relied on the
  `notify` branch short-circuiting before the latch, and the reversion
  meta-guard correctly called both copies decorative — each masked the other,
  so neither could be shown to do anything. That is #688's *one check, one
  door* in a second place. Membership is the test rather than "not notify",
  so a future non-terminal action is excluded by default instead of being
  latched until someone remembers to add a branch.
- **`has_abort_rung` is untouched**, which is what #947's
  `budget_limits_without_abort` finding reads: a ladder of pure checkpoints
  must not start looking like one that stops the run.
- **A checkpoint surfaces on the prose channel too.** A rung visible only
  through an event type shipped in this same change would be invisible in
  exactly the deployments asking for it. Opt-in by construction — you get
  that line only by writing `action: notify`.

**A new EVENT is the third degradation shape in the protocol changelog**, and
it degrades unlike both an additive field and a missing verb.
`deserialize_event` RAISES on an unrecognised `type`, but the SDK reader
wraps it, logs and continues — so an older client on a 1.8 daemon with a
ladder configured loses the event and logs a line per rung rather than
dropping the connection. Bounded, noisy in exactly the deployment that
configured a ladder, hence a version bump rather than a silent addition. No
SDK refusal: the direction is inverted from 1.5/1.6 (a NEW daemon emitting to
an OLD client, which cannot opt out), so a minimum to refuse below would fail
the wrong party.

**Overlap with #675, stated because both are open.** That issue names budget
degradation as subsystem 1 of 3 that changes the model with nothing emitting.
The **brownout** case is inside it; `finalize`/`escalate` is not (nothing
about the model changes) and `notify` is not (it changes nothing at all).
They are different events — *a rung fired* versus *the model changed* — and a
brownout fires both from one code path, so #675 should subscribe to that site
rather than add a second emission.

### A Session Nobody Was Watching (#812)

An eval sweep created session `20260903_084517` and its client process was
stopped ~40 s later. The session ran for **seven more minutes and spent
$2.52**, executing tools and marking plan steps complete against files that
no longer existed. Nothing graded it and no result row was ever written, so
the work is unrecoverable. It stopped only because its profile happened to
declare `budget_control` with a `degrade: at 100 -> abort` rung — **a profile
omitting `budget_control` would not have been stopped by anything.**

It also could not be stopped from outside. Every ceiling that existed was
held by something the session could outlive:

| Ceiling | Held by | Why it did not fire |
|---|---|---|
| `jaato_eval --arm-timeout` | the client's own runner loop | that process is the one that died |
| the task pool's `seconds` | reconciled when a session **ends** | a session that never ends never consumes it |
| `budget_control` | the session itself | worked — and is opt-in |

**Identity: what the daemon knew and never wrote down.** `spawn_session_runner`
builds a `SpawnedRunner` carrying the pid, and the `PoolSlot` when pool-served,
and hands it to `set_runner_rpc`. It then went nowhere an outside observer
could read: `~/.jaato/session_workspace_index.json` mapped the id to a
workspace and recorded no runner, pid or slot; the session record had no
runner-, slot- or pid-shaped key; the per-session logs named only an IPC
connection number (`client_ipc_14`). So the choice was killing a
circumstantially-identified process on a daemon shared with another live
session, or waiting for the budget to burn.

`jaato_server/server/session_identity.py` is the record — runner pid, `pool_served` + slot
pid, cascade, AppArmor profile — written to three places because they answer
different questions: `Session.runner_identity` (live), the session record
(**version 2.10**), and the daemon-owned workspace index (the cross-workspace
lookup for someone holding only an id). Reusable rather than eval-specific:
#806 needs the same fact to say which runner holds which language server.

Three properties, each attached to a way it could mislead:

- **A restored record is `stale=True`.** After a reload the pid belonged to a
  previous process lifetime, so it is *evidence* about what ran the session and
  never a handle. Nothing in the framework acts on a stale record. It is kept
  rather than cleared because "which process last ran this" is exactly what a
  post-mortem wants.
- **`None` means "could not read one", never "there is none".** The save-path
  refresher never clears a good record, or a transient read would destroy the
  evidence the field exists to preserve.
- **The refresh is on the SAVE path, not per spawn site.** There are two spawn
  call sites (IPC, and the WS pre-init hook) and a third could be added;
  instrumenting each is the shape that leaves one path armed and one silently
  not — the #735 failure. The explicit post-spawn stamp is kept for
  *immediacy* (identifiable from the first tool call, not the first save).

**Surface and stop.** `session.orphans` lists the LOADED sessions with no
client — how long each has been orphaned, its effective bounds, whether it is
`is_processing` (spending, right now), and the runner executing it.
`session.stop <id>` stops any one of them. Shaped after the verbs that already
exist (`session.list`'s own `SessionListEvent`, `cascade.cancel`'s confirmation
pair) rather than as a parallel surface, and distinct from both neighbours:
`session.end` stops the CALLER's session and `cascade.cancel` stops a whole
cascade, so neither could stop the one session an operator was looking at.

**Stopping is cancellation, not a kill.** Nothing signals the runner pid —
killing a pool-served runner destroys a slot other stages of the same cascade
expect to reuse, and stopping by id lets the daemon unwind its own bookkeeping.
It reaches `jaato_server.server.stop()`, the same cancel token `budget_control`'s `abort`
rung trips, so a mid-turn session stops at its next check point and is then
saved to disk.

**The bound.** `runtime_limits` gains two wall-clock fields, enforced
DAEMON-side by a `SessionManager` sweep (`jaato_server/server/session_lifetime.py`) — the
odd ones out in *where* they apply, which is the whole point: they are held by
the one process that is still there when the client dies, and they do not wait
for the session to end.

| Field | Measures | Default |
|---|---|---|
| `max_session_seconds` | total wall-clock LOADED | unbounded (opt-in) |
| `max_orphan_seconds` | continuous time with **no consumer** | **900 s** |

```yaml
runtime_limits:
  max_orphan_seconds: 120     # tighter than the default
  max_session_seconds: 3600   # opt-in total ceiling
```

`max_session_seconds` is opt-in because an interactive session left open over
a lunch break is not a defect and a default would kill it.
`max_orphan_seconds` is **the one field in the block with a framework
default**, because the session it exists for is precisely the one whose
profile declared nothing — a bound you must remember to write would have left
this session running exactly as long. `0` is the explicit opt-out, the
0-disables spelling `gc.media_bytes_threshold` and the OpenRouter deadlines
already use. Inheritance is **most-restrictive-wins** (`min()` across every
declaring layer, like `max_parallel_tools`), and `0` cannot win
that `min()` — a child may disable a bound no ancestor set, and may not
disable one an ancestor did.

**"Orphaned" is TWO conditions, and the second is the load-bearing one.** A
session qualifies only when it is loaded with no attached clients **and** the
sweep has previously *observed* it carrying one:

| Detached shape | Orphaned? |
|---|---|
| a completion-gated session that ended | invisible — it is UNLOADED, and the sweep only walks `_sessions` |
| `session.wake` → `resume_session` (a **cold revive**) | no — never observed attached, so clause 2 fails |
| a cascade stage mid-run | no — carries its driver's client |
| an attached interactive session | no — clause 1 fails |
| an idle orphan | usually unloaded by `_maybe_unload_session` before the grace expires |
| **a client that existed and went away, mid-turn** | **yes** — the #812 state |

The first draft had only clause 1 and justified it with an invariant —
*every path that drives a session attaches a client id* — which is **false in
this tree**. `_load_session_impl` uses its `client_id` for config, env and
progress events and never attaches it; `wake_session` branches on exactly that
("revived cold, no client — DEFERRED"). So a cold revive driving a long turn
would have been cancelled by a bound written for a different situation. An AST
guard policing the invariant would have failed on `main`, and one exempting the
revive path would assert almost nothing — so the *dependency* was removed
instead: the bound now rests on a fact the sweep MEASURES. It fails safe, since
a session never seen attached is never stopped by the orphan bound (an explicit
`max_session_seconds` still applies).

The orphan clock is **derived by the sweep** from `attached_clients` rather
than stamped at the ten-odd sites that mutate it: a bound that silently does
not apply is #735, and instrumenting every mutation is exactly the shape that
lets one new call site disarm it. It measures CONTINUOUS orphanhood, so a
reconnect renews the session's claim on being wanted.
`test_orphan_bound_observes_attachment_812.py` carries two AST guards — one
that `_orphan_since` has a single writer, one that the clock is started
*inside* an `if` consulting `_ever_attached` — each verified to fail on its own
reversion. The sibling precedents are `test_budget_mid_turn_955.py` and
`test_registry_iteration_snapshots.py`.

**It logs what it armed.** `start_lifetime_watchdog` is called by the daemon
(not from `__init__`, so a `SessionManager` in a test grows no thread) and logs
the sweep period and the *effective* defaults — #735's rule, since the declared
and effective values differ whenever a field is omitted or set to `0`.

Protocol **1.7** for `IPCClient.stop_session` / `list_orphan_sessions`. An
additive FIELD degrades harmlessly; a missing VERB does not — an older daemon
ignores `session.stop` silently, and its caller concludes a runaway session was
stopped. So the SDK refuses below `MIN_SESSION_STOP_PROTOCOL`, the #845 verdict
applied to a command rather than a payload.

Not addressed here: nothing grades a session whose harness died (the results
have no consumer by construction — #812's first ask, "terminate on client
loss", is deliberately **not** implemented as written), and a session that
survives its stop because its model thread is wedged is re-judged on the next
sweep rather than escalated.

### A Session Torn Down Because the Browser Blinked (#1106)

#812 asked when the daemon should **stop** a session nobody is watching.
This is the other verb on the same state: when it should **unload** one. The
answer was *immediately*, and nothing on that path had a time dimension at
all.

A WebSocket close — a tab reload, a network blip, a laptop waking, a phone
backgrounding the page — reaches `command_router.handle_client_disconnect` →
`SessionManager.detach_client` → `_maybe_unload_session`, which had exactly
two gates:

```python
if session.attached_clients:  return     # still has clients
if session.server._model_running:  return  # defer mid-turn
```

Both correct, neither a grace. The second is the only thing that had ever
saved a session from a blip, and only by accident — it holds the session
while a turn is in flight and the turn-tracking handler unloads it the moment
the model goes `done`. An **idle** session with a blinking client was torn
down on the spot: saved, log handlers closed, workspace monitor stopped,
isolated subagents torn down, `jaato_server.server.shutdown()`, pool slot returned. The
browser came back holding an id the daemon no longer had in memory and every
send was answered `[SessionError] Session not found:`.

Nothing was ever lost — `attach_session` revives from disk, and #1104 taught
the page to re-attach — so what is wrong is the **price**: a full teardown and
a full respawn for a two-second network event.

**The line, and it is a line rather than a carve-out:** *a terminal ENDS a
session; a disconnect only removes its audience.* Three of the four callers
of `_maybe_unload_session` mean the second thing and now defer; the fourth
passes `immediate=True`.

| Call site | Means | Grace |
|---|---|---|
| `detach_client` | a client went away | **yes** |
| attach switching away | a client left for another session | **yes** — left, not ended |
| the agent-done re-check | a deferred unload resumes | yes, measured from when it became clientless, so a long turn CONSUMES the grace |
| `_apply_default_cascade_policy` | a `SessionTerminatedEvent` | **no** |

That last row is the regression that matters most. Its own docstring measures
what a delay costs: every headless handoff returns its slot in ~250 ms, while
one discovery slot that stayed pinned stalled a cascade for **6m43s**. A 60 s
grace there would reintroduce that on every stage of every cascade.

**The attach-switch row has a cost, and it is stated rather than waved at.**
Without the grace, clicking through five sessions in a UI holds exactly one
at a time; with it, up to five, for one decaying 60 s window. What the grace
does NOT do is cause loads that would not otherwise happen — attaching
already loads a session — so the worst case is the transient working set of
someone browsing their own sessions, which is precisely the set they may
click back into. The alternative is paying a teardown and a respawn per
click, which is #1106's own complaint arriving through a different verb. A
deployment that disagrees has the knob.

```yaml
runtime_limits:
  unload_grace_seconds: 0     # no grace — the pre-#1106 behaviour
```

**Where the knob lives, and why not an env var.** `runtime_limits`, beside
the two wall-clock bounds, with a framework default of **60 s** — the exact
argument `max_orphan_seconds` used to earn its own default, because the
session that needs this most is the one whose profile declared nothing. An
env var would be a new `host`-scoped knob with no typed key, which is what
`docs/design/env-vars-vs-profile-keys.md` and the `AWAITING_TYPED_KEY` ratchet
exist to discourage; the profile key gets `explain runtime`, validation,
snapshotting and inheritance for free.

**60 s, because that is where the two costs cross.** A grace pins a runner
(129–187 MB) and a pool slot for its whole window, and buys nothing past the
point where disk revival is the right answer: a tab reload is ~2 s, a network
blip tens of seconds, a lid-close or a backgrounded phone **minutes** — which
no grace worth paying for would cover. It is also far below
`DEFAULT_MAX_ORPHAN_SECONDS` (900 s), so the ordering of the two verdicts is
unambiguous and the watchdog stays the outer bound.

**`0` means no grace, and is the TIGHTEST value here** — the opposite reading
from the two bounds in the same block, where `0` means "never stop this" and
is the *least* restrictive thing a layer can say. Both are min-wins across an
`inherits:` chain, so the two readings need two rules
(`_MIN_WINS_ZERO_TIGHTEST_FIELDS`, `jaato_server/shared/plugins/subagent/config.py`):
routing the grace through the 0-as-infinity branch would let a parent's 60 s
overrule a child that asked for none.

**The clock is DERIVED, and the issue's proposed source does not work.**
#1106 suggested reusing `_orphan_since`. It cannot be: that clock is written
only for a session the sweep has seen ATTACHED (`_ever_attached`), because the
bound it feeds *stops* a session and must not reach a cold `session.wake`
revive — which has an empty `attached_clients` by construction. Such a session
would have **no** clock, "no clock" reads as "defer", and it would never be
unloaded at all: the grace would be a leak rather than a delay. So
`_clientless_since` is a sibling dict with no such gate, written by two
readers that both derive the fact from an empty `attached_clients` at the
moment they observe it — `_observe_session_lifetimes` for every loaded session
each sweep, and `_note_clientless` from the unload gate itself, so the clock
starts at the disconnect INSTANT rather than up to one sweep interval later.
Never stamped where `attached_clients` is mutated: that is #735's shape, and a
future call site that forgets both writers still defers and is picked up by
the sweep.

**The sweep is the level trigger.** All four callers are edge-triggered and
none fires again just because time passed, so `_sweep_unload_grace` — in the
same pass that derives the clock — re-drives each deferral through
`_maybe_unload_session`, which re-applies every gate. It runs BEFORE the stop
verdicts: a session eligible for both should be unloaded (saved, revivable)
rather than stopped (cancelled).

**The grace is armed with the thread that carries it out.**
`_unload_grace_armed` is set by `start_lifetime_watchdog` and cleared by
`stop_lifetime_watchdog`, and both halves read it. A `SessionManager` built in
a test or an embedding process grows no watchdog on purpose and must not
silently acquire a mechanism whose other half is missing — a deferral nobody
acts on is #735 in its worst form, because the symptom is sessions quietly
accumulating rather than an error. Disarmed, the pre-#1106 behaviour is exact.
The arming line names all three fields with their effective values, and
`session.orphans` rows carry `unload_grace_seconds` /
`unload_grace_remaining`, so "kept for a client that may return" is
distinguishable from "nothing has got round to unloading it".

**A re-attach inside the grace costs nothing** — not "the unload aborts", but
the unload thread is never started, so there is no save, no handler close, no
`jaato_server.server.shutdown` and no slot return to undo. `_do_session_unload`'s existing
under-the-lock re-check remains, now as a backstop rather than the only
defence.

**The grace is deliberately NOT scoped by `client_type`.** The tempting rule
was "exempt `api`", on the grounds that a program's socket closing means the
program exited. `ClientType` is a **presentation** field — its docstring says
the values "describe the *kind* of display surface, not specific apps" — it is
client-declared and optional, and it is wrong in both directions: a localhost
`web` UI blinks least of anything and a chat bot on a mobile network blinks
most. Deciding how long the daemon holds a session from what the client's
screen looks like is the substitution #881 is about. The case it would buy is
narrower than it looks, too: a cascade-stamped or headless session is already
exempt through the terminal row above, so what remains is a **non-cascade
programmatic** driver that simply disconnects — and that deployment has a
visible knob (`unload_grace_seconds: 0`) where a `client_type` rule would be a
second, silent lifetime policy nobody validates. `test_unload_grace_1106.py`
pins this both behaviourally and at the source level, because a behavioural
test alone can be satisfied by a rule that reads the field and happens to
agree.

Out of scope, per the issue: terminate-on-client-loss (#812 records the
decision not to have it), `proxy` mode in `jaato-web-coder-server`, and the
§11.5 logout gap.

### A Session Waiting on a Human, Invisible from Every Other One (#1138)

#812 asked which session nothing is consuming. This is the opposite state:
a session that is blocked because it is consuming **you**. One session
raises a permission ASK or a `request_clarification` while you work in
another, and the turn stays blocked until you happen to attach to it and
find the prompt sitting there.

No client could fix it. Prompt events go to `session.attached_clients` and
`_client_to_session` is **1:1**, so a browser attached to A is not in B's
set and B's `PermissionRequestedEvent` never reaches it;
`SessionManager.broadcast_event` is not the escape hatch either — its
docstring reserves it for events that are *not* tied to a specific session.

So the fact rides the listing every client already polls. `session.list`
rows gain **`awaiting`** (`permission` / `clarification` / absent) and
**`awaiting_since`** (ISO-8601 UTC). Protocol **1.17**.

**`is_processing` could not carry it.** A session blocked on a prompt is
still processing; telling *working* from *waiting on you* is the whole
point, and one boolean cannot.

**The issue named the wrong two fields, and implemented literally the
change would have been inert on the default path.** It reads
`JaatoServer._pending_permission_request_id` /
`_pending_clarification_request_id` and calls this "the same read, one
field over" from `is_processing`. Those are written by the **daemon-side**
hooks in `_setup_permission_hooks` / `_setup_clarification_hooks`, and
`permission`, `clarification` and `references` are all
`PLUGIN_TIER = "runner"` — so on a runner-served session the plugin that
raises the prompt is in the runner process and the daemon-side hook is not
in the loop. `PromptOperatorHandler`'s own comments say so: *"that path is
dead post-§7c since the runner-side permission plugin is the one in the
loop"*. Measured: with a relayed ASK in flight, the relay holds the request
id and `_pending_permission_request_id` is `None`.

| Path | Where the prompt is pending |
|---|---|
| runner-served (**the default**) | `_prompt_operator_handler` / `_clarification_relay_handler` |
| embedded, standalone-WS, legacy Path 2 | `_pending_*_request_id` + their `_since` twins |

`JaatoServer.awaiting_prompt()` reads **both**, so the field is not a
strategy resolved and installed on nobody (#1133) or a cap that silently
does not apply (#735).

Four properties, each attached to a way it could mislead:

- **The pair describes the OLDEST unanswered prompt.** Both kinds can be in
  flight at once — tool execution is 8-wide — and the pair has room for
  one. Oldest-first is the question a reader is actually asking, and it
  needs no invented ranking between the kinds; the vocabulary order is the
  tiebreak only, for prompts the clock cannot separate.
- **`awaiting_since` is a SECOND field, never a widening of `awaiting`.**
  The degradation argument depends on `awaiting` staying a scalar an older
  client can ignore. The clock is not a nicety: the listing is a **poll**,
  so without it a client can only date the wait from when *it* first saw
  the flag — which under-reports every wait that predates the client and
  resets to zero on every reconnect. Wall clock, not monotonic, because its
  only consumer is a browser subtracting it from its own.
- **In-memory branch only.** A persisted-only row has no `server`, and a
  session that is not loaded cannot be waiting on anything.
- **The vocabulary is closed AT THE BOUNDARY.** `awaiting_of` accepts any
  duck-typed server (the `_turns_ran_snapshot` precedent, #881), so
  `awaiting_fields` drops a kind outside `AWAITING_KINDS` and a `since`
  that is not a real number. A listing that raised because one session's
  server answered oddly would be worse than the fact it was reporting.

**No poll cadence knob.** `session.list` is client-initiated, so a
daemon-side interval would be advice a client can ignore — the
`finalize`/`escalate` shape — and the cost that actually bounds the cadence
is `list_sessions`' parse, which is #1137's to fix. The client has a better
trigger than a timer anyway: refresh on window focus.

**The protocol bump, where #812's `orphaned` / `runner` took none on this
same free-form dict.** The difference is what a client does with the field:
those two are diagnostics a human reads, this one gates whether a client
interrupts a person. "Can this daemon tell me?" is a question a client will
ask, `ConnectedEvent.protocol_version` is the only way to ask it, and a
client that cannot distinguish *no session is waiting* from *this daemon
never says* reports the first when the truth is the second. Additive in
both directions, so no SDK minimum to refuse below.

**Not widened to a third kind.** `_pending_reference_selection_request_id`
is a fourth holder of the same shape, and `references` has **no** relay
handler — so on the runner path a `selectReferences` prompt has no daemon
side at all. That is its own gap; adding `"reference"` here would widen a
vocabulary the issue closed while fixing nothing on the path that matters.

The client half is #1135's Sessions rail: a second marker beside the
`●`/`○` glyph, never a recolouring of it — that glyph is `is_loaded` and
the two facts are orthogonal. The TUI has the same blind spot and now the
same field; it renders `SessionListEvent` already, so what it needs is a
render change rather than a daemon one.

Guard: `jaato_server/shared/tests/test_a_session_waiting_on_a_human_is_visible_1138.py`,
six reversions. The relay cases drive the real `handle()` coroutine rather
than poking the handlers' dicts — a case that registered the future by hand
would pass against a handler that never stamped.

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
merge on both the daemon (`jaato_server/server/core.py`) and runner
(`jaato_server/server/runner/session.py` Step 8) paths. **A subagent reuses the parent's
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
| daemon (`jaato_server/server/core.py`), runner (`runner/session.py` Step 8) | constructed separately, seeded from the root block | re-initialized the registry's **unread copy**; the enforcer kept the parent's policy — #957 as reported |
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

### A Plan the Profile Names, Not the Model (#1195)

A session can start with its plan already in place instead of asking the
model to write one:

```yaml
plugins: [todo, cli]
plugin_configs:
  todo:
    initial_plan_name: onboard-service   # an id, never a path
```

The id names `<config_root>/plans/<id>.yaml` — an **authored** asset beside
`profiles/` and `agents/`, in the `.gitignore` `AUTHORED` set and write-denied
to a confined runner (template **v33**, all four profile bodies). Not the
`{base_path}/plans/{plan_id}/` directory `FileReporter` writes: that is a
write-only progress log nothing reads back, and pointing an authored input at
it would have made the plugin's own output a place to author plans.

For the session that declared it, and **only** that session:

| Effect | Mechanism |
|---|---|
| the plan is its active plan before turn 1, overriding a plan an earlier cascade stage of the same agent left in the per-agent map | `TodoPlugin.preload_plan`, called from `JaatoSession.configure()` |
| its system prompt carries *"You were invoked with a predefined plan. Use the available TODO tools to read it and organize your work to comply with it."* | the plugin's instruction contribution, only for a scope whose plan LOADED |
| `createPlan` is off its surface | the session's own `_tool_scopes["todo"]` — every other TODO tool stays |

**Per session, never per instance — the #944 lesson.** The todo instance is
shared by a parent and its in-process subagents (one registry) and carried
across stages on a pool slot (#890), so gating the instance would strip
`createPlan` from every session sharing it. The mechanism is #957's: the
session mints `JaatoSession.plugin_scope`, the plugin files per-session state
under it (`_scoped_plan_ids`, `_preloaded_scopes`) and resolves the CALLING
session's scope at call time, and the gate is the per-session tool-scope
filter rather than the plugin's schema. `TodoPlugin.initialize()` ignores the
knob by design. Two consequences handled rather than inherited:

- **A subagent's preload never lands in the per-agent map.** An in-process
  subagent is configured while its `agent_id` is still the default `main` —
  its parent's key — so the preload is bound into that map lazily, on the
  first tool call of a session whose `agent_type` is not `subagent`. A root
  cascade stage therefore still hands its plan to the next stage of its
  agent; a child cannot overwrite its parent's.
- **`is_tool_visible` asks the session that is actually on the wire.**
  `_get_tools_for_provider` now sets the session ContextVar before consulting
  the predicates, so the plan-required tools are visible on turn 1 of a
  preloaded session rather than decided against whatever session last wrote
  this thread's thread-local.

**The authored file is only read** (`yaml.safe_load`). What is stored is a
copy with a **fresh `plan_id`**, kept by the ordinary storage backend — so two
sessions preloading one file never share one stored plan, and even a file
storage rooted at `.jaato/plans/` writes beside the authored file, never over
it. The document mirrors `TodoPlan.to_dict()` (the todo README's
`<!-- initial-plan-example -->` block, loaded verbatim by a test); only
`title` and `steps[].description` are required, and `started` defaults to
**true** because the authored file is the approval `startPlan` otherwise asks
for.

**A missing or malformed plan refuses the session.** A plugin
`initialize()` exception is swallowed by `expose_tool` (a WARNING, and the
session up without the plugin), so the knob is read by the session instead:
`configure()` raises `InitialPlanError`, which on the runner is a bootstrap
failure the daemon turns into a non-recoverable `RunnerBootstrapFailed`
refusal naming the file (#1033), for an in-process subagent a failed spawn,
and in-process an exception from `create_session`. A profile declaring the
knob without enabling `todo` is refused the same way. `jaato-scaffold
validate` reports the same defects first, through the same resolver:
`initial_plan_name_invalid`, `initial_plan_missing`, `initial_plan_invalid`
(all **error**).

**Todo persistence is YAML only.** `FileStorage` writes `{plan_id}.yaml` /
`todo_plans.yaml` (the `TODO_STORAGE_PATH` default is `./todo_plans.yaml`)
and `FileReporter` writes `plan.yaml`, `progress.yaml`, `events/NNN_*.yaml`
and `latest.yaml`. Deliberately **no** fallback reader and no warning: plans
an older release saved as `.json` are not read. `todo.json`, the plugin's own
CONFIG file, is configuration rather than persistence and is unchanged.

Not done: a revived session re-runs `configure()` and so re-preloads a fresh
copy rather than resuming the one it had — the per-session state is not
persisted, which matches what a revive does for every other todo plan today.

Guard: `jaato_server/shared/tests/test_a_predefined_plan_is_per_session_1195.py`, eleven
reversions — among them the instance-wide gate (a sibling session must still
see `createPlan`), the hint recorded before the load, the preload losing to a
carried plan, a subagent binding into its parent's key, and `safe_load`
replaced by a loader that constructs objects.

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

### Six Things a Session Log Said the Authoring Surface Still Would Not

A 2026-09 transcript of an assistant bringing up a workspace from scratch —
venv, a simple client, then a two-stage GitLab MR cascade — records eight
corrections. Three were already closed (`deprecated_system_instructions`,
`explain services`, the `configure_service_auth` chain), one had since been
*inverted* (#950 made `plugin_configs.<plugin>` apply whether or not the
plugin is in `plugins:`, so the transcript's own correction is now false),
and what survived was six gaps of one shape: **the author wrote something
that had no effect and nothing said so.**

That family already has four entries — #910, #925, #947, #950 — and the gap
each of these closes is one layer out from where those stopped.

**A profile's own top-level keys were never checked.** `unknown_knob` covers
`plugin_configs.<plugin>.<knob>`, `trace:` refuses its own unknown keys,
`runtime_limits` parks them in `extra` — and the outermost layer had no
reporter at all. `SubagentProfile` construction is keyword-explicit (the
snapshot-version note in `config.py` says so), so a key outside
`PROFILE_FILE_KEYS` is read by nobody, silently.

The transcript contains a live instance nobody noticed: told *"te falta
activar el config_root"*, the author added `config_root: .jaato` to two base
profiles. `config_root` is a session/SDK parameter and **not a profile key**,
so the line did nothing; what actually fixed the session was the duplicated
`.jaato/` path prefix removed in the same edit (now
`redundant_config_root_prefix`). The session ended successfully with a wrong
belief baked into the workspace.

| Finding | Severity | Fires when |
|---------|----------|-----------|
| `unknown_profile_key` | warn | a top-level key the loader does not read; a near-miss from the accepted set is named (`plugins_configs` → `plugin_configs`) |
| `derived_profile_key` | warn | the key IS a `SubagentProfile` field and is DERIVED, not read from the file |

The second exists because `explain profile` renders
`dataclasses.fields(SubagentProfile)`, and two of those fields cannot be
written into a file at all: `preloaded_plugins` and `tool_scopes` come out of
the `plugins:` list's own modifiers (`todo(preload)`, `memory(tools:[a,b])`).
The page was **advertising two keys a file may not set**; it now says so
beside each, reading the same `PROFILE_DERIVED_FIELDS` constant the finding
does.

Warn rather than error, the posture the whole family takes: an `inherits`
base may carry a key a later version reads, and a snapshot may be newer than
the installation reading it. `PROFILE_FILE_KEYS` is a declared constant
because the loader reads four of its keys through block parsers and one
(`env`) through a parameter default, so no scan alone could produce it —
and `test_profile_file_keys.py` AST-scans the six loader functions and fails
the build if any literal key read is missing from it. 26 of 27 keys are
covered that way.

**`explain plugin permission` rendered the whole policy vocabulary as one
line.** `policy  object  Permission policy rules` — while
`get_config_schema()` has always declared `defaultPolicy` with its
`allow`/`deny`/`ask` enum, `whitelist.tools`, `blacklist.patterns` and the
four-deep `sanitization.path_scope.*` tree. So the only honest route to the
shape was `jaato_server/shared/plugins/permission/policy.py`, and the transcript took it —
for a page whose entire job is to make that unnecessary.

`ConfigSetting` gains `children` and `free_form`, and both surfaces descend:
`explain` prints the tree at every declared depth, `validate` checks names
and values at every declared depth. Nesting is rendered **fully**, unlike a
tool PARAMETER, which stops at one level (#1020): the reader's need is the
opposite — a nested parameter is one call's argument, while `policy` IS the
plugin's whole configuration surface. Descent stops at
`additionalProperties` (`permission.evaluators` maps tool names to scripts,
so every key there is authored and none is a typo); that frontier is
declared by the plugin, marked in the render, and is where `validate` stops
reporting unknown names. `permission.policy.defaultPolicy: denied` is now
`invalid_knob_value`, an error; it used to validate clean.

**A declared `properties` block is NOT a completeness claim**, and the first
version of this descent read it as one — which inverts the whole family's
thesis. `PermissionPolicy.from_config` reads `cwd`,
`sanitization.custom_blocked_commands` and `path_scope.resolve_symlinks`,
and the schema declared none of them, so all three were reported as typos:
a knob that *does* something, told it does not, whose remedy is deleting a
working line. Measured across the plugin, the schema omits **sixteen** names
its own reader consumes, so the completeness claim would be wrong far more
often than right.

So the nested finding is **evidence-driven**, exactly as `_report_undeclared_name`
is one layer up: a read site → `undeclared_knob` quoting it, no site →
`unknown_knob`, not scanned → the absence of evidence stated as such. The
evidence needs a third scanner (`plugin_nested_config_read_sites`), because a
nested key is read off a LOCAL — `ps_cfg.get("resolve_symlinks")` — and
`_TOP_LEVEL_CONFIG_RECEIVERS` excludes those on purpose (a nested dict's
inner names are not `plugin_configs` knobs). Widening that set would have
been the wrong fix twice over. The three keys above are now declared too, so
`explain plugin permission` shows them.

**And so are the five an author actually writes.** `agent_name`,
`config_path`, `workspace_path`, `channel_type` and `channel_config` are read
straight off `initialize(config)`, and someone configuring a webhook approval
channel had no route to their shape but the source — which is the sentence
this whole section opens with. `channel_config` is declared
`additionalProperties`, because the names inside it (`endpoint`, `headers`,
`auth_token`, `timeout`, …) belong to the CHANNEL and nothing in this plugin
should judge them.

**The gap is a measurement before it is a rule.** Reconciling every in-tree
plugin's schema against its own reader gives 123 undeclared reads across 26
plugins — and the number is not one number: 33 are the keys
`PluginRegistry._augment_plugin_config` INJECTS into every plugin's config
(`workspace_path` / `config_root` / `session_id` / `agent_name`, `setdefault`
so an author may still override one), and the rest span the block an author
writes, a nested dict with its own owner, and *a different file the block
points at* — `permission.config_path` names a permissions JSON whose own keys
(`version`, `channel`) are not `plugin_configs.permission` keys at all.

So `scripts/plugin_schema_census.py` ships as a **census, not a guard**, with
its output committed at [Plugin schema census](docs/design/plugin-schema-census.md)
and `permission` worked through site by site as the one audited row. A
ratchet needs a definition before it needs a baseline: seeded today it would
freeze that four-way ambiguity as though it were a fact, and its
stale-entry rule would then charge every unrelated PR that declares a knob
with updating a number nobody can re-derive. If it becomes one, the thing to
count is the block an author writes — keys read off `initialize(config)` —
said so in its docstring, with the other three surfaces a documented
exclusion.

**The permission whitelist was unchecked against the tool inventory
`tool_scopes` has been checked against since the validator shipped.** One key
over, and the key where being wrong is expensive: under `defaultPolicy: deny`
a name matching nothing is a permanent denial of a tool the author believes
they approved, and the runtime symptom is the one #951 exists to make
legible — the call reaches the gate and vanishes. It is also the cheapest
thing in a profile to misspell, because nothing else in the file repeats the
name.

| Finding | Severity | Both lists? |
|---------|----------|-------------|
| `unknown_tool` | warn | yes — the name is exposed by no installed plugin and is not a framework session tool |
| `permission_rule_without_plugin` | warn | **whitelist only** — the tool exists and its plugin is absent from `plugins:`, so the rule governs a tool that never reaches the wire |

The blacklist is deliberately exempt from the second: denying a tool the
profile does not enable is defence in depth, and a profile that adds the
plugin later keeps the protection it already wrote. Silent by design: a
profile whose `plugins:` list is EMPTY (an abstract base declares no
surface — the carve-out `missing_model` already uses), a name of MCP shape
(`mcp__server__tool` / `mcp.server.tool`, whose inventory comes from the
servers a LIVE session connects to), the framework's own session tools
(`signal_completion`, `askPermission`), and a tool belonging to
`PluginRegistry._ALWAYS_INITIALIZE_PLUGINS` — `introspection`'s `list_tools`
/ `get_tool_schemas` are core and reach every wire whatever `plugins:` says,
the exemption `plugin_config_without_plugin` already applies and this check
initially dropped. The set is read from the registry, not re-spelled.

"What the author declared" and "what the session will hold" are two
variables for that reason: folding the framework's always-initialized set
into the first would make every abstract base look like it declared a
surface.

**`validate` never asked whether the provider's SDK was installed.** It did
not import `jaato_server/shared/scaffold/dependencies.py` at all, while both halves of the
answer lived there: the AST closure of the provider package, and the
import-name → extra index that turns a missing module into the `pip install`
line. So `provider: azure_openai` with no `openai` validated clean and died
at `connect()` with an ImportError several layers from anything the author
wrote — the transcript's first runtime error.

```
[warn] provider 'azure_openai' needs azure, openai, which are not installed
here — the profile is valid and the session will fail at connect() with an
ImportError.  Run: pip install 'jaato-server[azure-openai]'
```

`provider_import_gaps` probes with `find_spec`, importing nothing, because
`validate` is required to be side-effect free and a provider SDK's import can
register handlers, read environment or open a config file (`_health`, which
serves an operator-requested report, still imports; a subprocess test asserts
the difference, with a control run that would notice an import). Every tier's
provider is checked too — a tier binds a (provider, model) PAIR (#1036), and
the second provider is the one nobody notices until `enter_tier`. **Warn**,
not error: validating a workspace from a machine that is not the one that
will run it is legitimate.

**And the message asserts no runtime consequence.** Its first wording said
the session "will fail at `connect()` with an ImportError", which a static
import closure cannot know and which is false for a guarded import:
`azure_openai`'s closure includes `azure`, and `azure_identity_available()`
wraps that import in `try/except ImportError` on a path only `auth: aad`
takes — so a key-auth profile was told it would fail, and it would not. Same
class as #937, in this change's own new code. What it says now is what it
measured: the package imports these names, they are not installed here, and
whether a session reaches them depends on which path its configuration takes.

Only TOP-LEVEL names are probed, so a namespace
package whose submodule is absent reads as present (`google` resolves from
`google-api-core`); that blind spot is `_health`'s too, and it is the safe
direction.

**Processors declared behind a tool that is not on the wire.**
`_should_hide_signal_completion` gate 1 hides `signal_completion` whenever no
`completion_payload_schema` is declared, so a profile carrying
`completion_processors` and no schema has a gate that can never run — the
agent hunts for the tool through `list_tools`, the nudge budget drains, and
the driver is handed `None` by a session that looked like it ran. That is
exactly the state the transcript spent an afternoon in, and the fact was
already written down: `docs/design/completion-gate.md` §9 is why
`jaato-scaffold new sweep` emits the schema, the processor and the profile
keys as ONE set. Nothing enforced it for a profile written by hand, which is
the only way to reach the state. `completion_processors_without_schema`,
**error**, matching `completion_asset_missing` — the two are the same defect
by different routes. Silent for a profile binding neither `model` nor
`model_tiers`: processors are inherited, so every concrete descendant is
checked in resolved form.

**And the OTHER gate on `signal_completion` is not a profile key at all.** A
root session on an interactive client (`terminal` / `web` / `chat`) hides the
tool; `api` keeps it. So one profile completes under a headless driver and
cannot complete under the TUI, with the same files — the one gate an author
cannot see by reading their own workspace, and `explain` stated it only as
hand-written prose on one page. It is now **probed**
(`introspect.client_gate()`, which exercises `LifecycleTools` once per
`ClientType`) and rendered on both `explain completion` and `explain plugin
lifecycle`, so it cannot drift from the gate it describes.

**`new client` always emitted the inline spec.** The template's comment said
*"Inline spec so this runs before you have a profile. Swap for
profile=…"* — and the generator always wrote the spec, whatever the workspace
already had in it, while the only message an author saw when they supplied no
flags was `missing required --provider / --model`: the inline answer, pointing
away from the one form that can carry plugins, a persona, GC, ceilings and a
completion schema. The transcript's first correction was *"en lugar de un
.env, usa un profile"*.

```bash
jaato-scaffold new client --workspace . --profile collector
```

`--profile` is mutually exclusive with `--provider`/`--model` (two bindings
for one session, where the profile wins at runtime, so the flags would
decide nothing), refused on `--transport in_process` where the embedded
client IS the binding, and **refused by name on an archetype that opens no
session it can bind**. That last one is derived from the template rather
than tabulated — the `__SESSION_BINDING__` placeholder is what receives the
binding — because a hardcoded list is how the flag came to be accepted where
it could not be honoured: `new cascade --profile worker` resolved the name,
accepted it, and emitted the `"<profile-name>"` placeholder, byte-identical
to passing nothing. A cascade's stages each name their own profile, which is
the point of them.

The name is resolved through the framework's own resolver under the set the
workspace actually selects — `JAATO_PROFILE_SET` from its `.env`, or
`--set` — so a profile that resolves only through `inherits` counts, and one
that exists only inside an unselected set is reported as that rather than as
missing. Four properties:

- **`[]` and `None` are different answers.** `[]` is "this workspace declares
  no profiles", which `--profile` can be refused against; `None` is "I could
  not look", which must not become "your profile does not exist".
- **The workspace `.env` is never rewritten for a `--profile` client**, even
  under `--force`: that file is where `JAATO_PROFILE_SET` lives, and this
  archetype's template carries a provider/model pair the profile supersedes.
- **With no `--profile`, nothing changes.** An empty workspace gets the same
  message and the same inline-spec client it always did; the `--profile`
  suggestion appears only when the workspace demonstrably has profiles to
  name.
- **The set that made resolution succeed is persisted, and the banner
  carries the flag.** `--profile X --set Y` resolved X only because Y was
  forced, and the generated client resolves its profile from the workspace
  `.env` — so `JAATO_PROFILE_SET` is written there (never retargeting one
  already present). `_provenance` names `--profile` too: its whole claim is
  to be copy-paste reproducible, and without the flag the printed command
  re-ran to `missing required --provider / --model`.

### A Workspace That Committed Its Sessions, or Lost Its Profiles

`<workspace>/.jaato/` mixes two things a version-control rule must treat
oppositely: **authored assets** (profiles, agents, instructions, the two
payload schemas, prefetch scripts and completion processors, service
specs, the template catalog) and **runtime state** (`sessions/`, `logs/`,
`memories/`, caches, language-server data, and the `<provider>_auth.json`
a stored credential lands in). The split was already written down — once,
as the `audit deny .../.jaato/<x> wlk` rules in `jaato_server/server/apparmor.py` that
name the "user-authored config subpaths" a confined runner may not
rewrite, and in prose on `explain paths` — and expressed nowhere a
repository could read. `new profile-set` ignored `.env` and nothing else,
so a workspace either committed its session records and stored keys, or
ignored `.jaato/` wholesale and lost the profiles its sessions ran under;
the TUI's own `.jaato.example/README.md` documented the second posture as
the default, with a hand-typed re-include list as the remedy.

`jaato_server/shared/scaffold/gitignore.py` declares the split as data (`AUTHORED`,
each entry carrying what it holds and whether the template write-denies
it), and three consumers read it so they cannot disagree:

| Consumer | What it does |
|---|---|
| every `new` archetype that writes under `.jaato/` (`profile-set`, `processor`, `sweep`) | merges the block into the workspace `.gitignore` — in the SAME write as the `.env` rule, one file and one plan entry, so `--dry-run` reports it truthfully instead of two helpers each reporting `create` |
| `new gitignore --workspace DIR` | the block on its own, for a workspace written by hand; it reads the file back through the daemon's parser and fails if the block did not take |
| `validate` | judges an existing `.gitignore` by **effect**, never by spelling: `gitignore_missing` (a repository root with none), `gitignore_hides_jaato_assets` (authored entries ignored), `gitignore_leaks_jaato_state` (state probes not ignored — a stored credential is named when it is one of them). All `warn`, each naming the command |

```gitignore
!.jaato/
.jaato/*
!.jaato/profiles/
!.jaato/agents/
!.jaato/completion_schemas/
!.jaato/scripts/
# … one re-include per authored entry …
.jaato/**/__pycache__/
```

Three properties, each attached to a way it could go wrong:

- **Ignored unless named.** The block is `.jaato/*` plus re-includes, never
  a list of state directories to ignore, so state a later release adds —
  or a credential file nobody anticipated — is never one `git add -A` from
  a remote. The failure direction of an incomplete list is an asset that
  needs `git add -f`, not a leak.
- **A wholesale rule is neutralised, not edited.** Git cannot re-include
  beneath an excluded directory, so on top of an author's `.jaato/` (or
  `.*`) line every `!.jaato/<x>/` would be inert. The block opens with
  `!.jaato/`, which un-excludes the directory before `.jaato/*` excludes
  its children — last match wins, and the author's line stays where it
  was. Verified against `git check-ignore` for each prior shape, and the
  reason `validate` stays read-only (it is required to be side-effect
  free): it reports, `new gitignore` writes.
- **The two declarations of "authored" are guarded.**
  `test_gitignore_authored_set_tracks_apparmor.py` reads the template's
  write-deny rules out of its source and checks them against `AUTHORED`
  in both directions, so a directory becomes committable the release it
  is protected.

**Judging by effect needed a parser that answers like git.** The daemon's
`GitignoreParser` — what the workspace monitor's file panel reads — was an
approximation in three ways that all bite on this one file: `*` crossed
`/`, so `.jaato/*` swallowed every file beneath and no re-include could
reach them; a multi-segment directory pattern was compared against each
path SEGMENT on its own, so `!.jaato/profiles/` matched nothing at all;
and a negated directory pattern un-ignored everything beneath the
directory, where git un-excludes only the directory itself. It now
compiles each rule to git's model (`*` stops at `/`, directory-only rules
match directories, an excluded ancestor excludes everything beneath it)
and agrees with `git check-ignore` on every probe in
`test_gitignore_parser_honours_multi_segment_dirs.py`. `gitignore_missing`
fires only at a repository root: a nested workspace with no file of its
own sits under rules the validator cannot see, and reading their absence
as "unignored" would warn on every eval workspace under an ignored
parent.

### When the Introspection Tools Misreport Their Own Environment (#966, #823)

Two findings of one shape, and it is the one thing a diagnostic must not do:
`jaato-doctor` and `jaato-scaffold explain` describing an environment that is
not the one they are running in.

**An incomplete install list is worse than none (#966).** `explain
dependencies` enumerated extras over a hardcoded `("jaato-sdk",
"jaato-server")` pair — written twice, as `EXTRA_DISTS` and again as an inline
literal inside `framework_picture()`, which had drifted from the constant
beside it. `jaato-premium` was in `FIRST_PARTY` and in `JAATO_DISTS` and in
neither copy, so its nine extras were invisible: an operator was shown a list
of what to install and the list was short. The same scan feeds the
missing-import → extra index, so `presidio_analyzer` could not be traced back
to `pip install 'jaato-premium[pseudonymization]'` the way `pexpect` is traced
to `jaato-server[interactive]`.

Adding `"jaato-premium"` to the tuple would fix the one distribution that had
already shipped and leave the next one equally invisible — the same defect
wearing the fix as a disguise. So the set is **measured**:
`installed_jaato_dists()` reads installed metadata for every distribution whose
normalised name starts with `jaato-`, `framework_dists()` unions that with the
names this repo expects (so "not installed" stays sayable about the ones it
knows), and one enumeration — `_extras_by_label()` — serves both the rendered
listing and the index that inverts it. A distribution this repository cannot
see at authoring time participates with no edit here.

Not installed is **silence, not a finding**: `_requires` answers `[]` for an
uninstalled distribution, so a workspace without premium is told nothing about
premium's extras. What it is still told is that `jaato-premium` is not
installed, which is the honest report and was already there. Deliberately not
added: a note that `[pseudonymization]` also wants a spaCy model. It is true
and it is a hardcoded fact about one extra of one package — the thing this
change exists to stop.

**The daemon is not running the code you are reading (#823).** A client and its
daemon can resolve `jaato_sdk` from different checkouts with no mistake made by
anyone: the daemon launched with a `PYTHONPATH` at a feature branch, the client
launched without one and inheriting the venv's editable install. Both are
installed correctly and they speak different event shapes, so the daemon sends
`ToolOutputEvent.mime_type`, the client's class has nowhere to put it, pydantic
drops it on ingest, and a handler dies with `AttributeError` several frames
from anything the reader wrote. It bites hardest while the event protocol is
being extended — the case where a field is new is the case where one side lacks
it.

`check_checkout_skew` sits beside the HOME comparison because it is the same
class of defect (an invisible property of the daemon PROCESS that changes
behaviour) answered from the same `/proc/<pid>/environ` block. The daemon's
resolution is derived the way CPython would: its own `PYTHONPATH` entries in
order, then the installed package.

| Both sides resolve | Verdict |
|---|---|
| one directory | PASS |
| two paths, at least one a source **checkout** | **FAIL** — never intentional in a dev loop |
| two paths, both unpacked **installs** | WARN — what a rolling upgrade looks like; both versions are named |

The discriminator is measured, not declared: a checkout has the distribution's
`pyproject.toml` beside the package directory, an install in `site-packages`
does not.

**The false PASS it must never produce.** With no daemon `PYTHONPATH`, "the
installed package" means installed for the DAEMON's interpreter. Resolving the
caller's own `site-packages` instead would compare a path against itself and
report PASS about two environments that were never compared — so
`/proc/<pid>/exe` is read beside the environ, and when the interpreters
demonstrably differ the check says it cannot tell. Every probe here is
best-effort and returns rather than raises: `_proc_exe` uses `readlink` (not
`realpath`, which answers with the path it was handed when nothing is there,
reporting an interpreter that does not exist), and a `PYTHONPATH` entry that
cannot be walked is skipped.

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
`jaato_server/shared/config_resolver.py`, `explain paths` — and applied in one place: the
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
model-controlled code.  `jaato_server/shared/secret_scrub.py` removes a set of secret
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
| `is_sensitive_proc_path` in `jaato_server/shared/plugins/sandbox_utils.py` — the same set plus `maps` / `smaps`, minus `cmdline` | a path handed to a model-driven **file** tool (`readFile`, `glob_files`, `file_edit`) | always, including `workspace_root` unset and the degraded posture of #504 |

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

### Two Principals on One Socket

`EventSink.get_client_user` returned a hardcoded `None` on IPC, with the
comment *"IPC connections are local and unauthenticated — user identity is a
WS/SSO concept."* True of a daemon serving one human, which is what
`--socket-mode`'s `0o660` default encodes. False of the deployment the same
flag documents — *"pass `666` to opt into world-accessible (e.g. cross-user
containers on a trusted host)"* — where several OS accounts share one socket
and the daemon could not tell them apart.

**Two principals, and they are orthogonal.** The account the daemon RUNS as
(owner of `~/.jaato`, the pooled provider credentials, `ws.token`) and the
accounts that CONNECT are different axes, as they are for any shared Unix
service. The framework already supports the second at the configuration
layer: `resolve_config_search_path` puts a CLIENT-SUPPLIED `config_root` (or
`<workspace>/.jaato`) at the primary tier and the daemon's `~/.jaato` only as
a fallback, so each connecting user can bring their own profiles, agents —
and their own `<provider>_auth.json`. State writes are anchored at
`<workspace>/.jaato` by `workspace_state_path`, which never honours
`config_root`, so session records and logs land in the user's own tree. Per
user configuration and per user credentials were already expressible.

**What was missing is the binding.** Nothing tied *which workspace or
config_root you may name* to *who you are*. Both arrive as plain strings
(`CommandRouter._handle_set_workspace` reads `args[0]`; `ClientConfigRequest`
carries `working_dir` / `config_root` / `env_file`) and the only validation
on the way down is that they be absolute — #742's anti-ambiguity guard, not
an access check. The daemon then acts on them with ITS credential, and so
does the runner: `RunnerSpawner._exec_runner` is `fork()` + `os.execvpe` with
no `setuid` anywhere on the path, so a session runs in a different PROCESS
under the same UID. A peer who cannot read another user's tree could have the
agent read it for them — a confused deputy with the service account as the
amplifier.

**Every boundary below the transport is workspace-shaped**, so none of them
could answer it: the AppArmor profile is keyed on `workspace_root` plus the
rendered profile body (#1033), `check_path_with_jaato_containment` tests
against the session's own root. Point a session at somebody else's tree and
they all do their job perfectly — they lock the runner INTO that tree. The
only place the question is answerable is before the path is accepted, where
the peer credential exists and the session does not.

`jaato_server/shared/peer_identity.py` is that place: stdlib-only, the shape
`jaato_server/shared/apparmor_label.py` already has, so the transport can import it before
plugin discovery and so the answer cannot be derived twice.

| Half | What it does |
|------|--------------|
| attribution | `SO_PEERCRED` → `PeerCredentials`, rendered by `get_client_user`. `Session.created_by`, the ledger's `response` / `permission-check` `user_id` and the telemetry `user.id` populate on IPC for the first time — everything above `EventSink` is transport-agnostic, so #859's plumbing lights up with no further change |
| entitlement | `unreachable_client_paths` refuses a `workspace_path` / `config_root` / `env_file` / trace path the connecting account could not reach, at `_handle_set_workspace` and `_reject_unentitled_client_paths` (beside #742's relative-path guard, and all-or-nothing for the same reason: a half-applied handshake is its own silent-wrong-directory bug) |

**The reachability rule, and why each half is what it is.** An existing path
needs `r-x` — deliberately NOT `w`, because an org-wide `config_root` of
shared profiles under `/opt`, readable by everyone and writable by none of
them, is a legitimate and desirable shape in exactly this deployment. A path
that does not exist needs `-wx` on the nearest existing ancestor, since the
daemon provisions workspaces and refusing every not-yet-created directory
would refuse the normal case. Every ancestor needs `--x`, which is what makes
a `0700` home directory protect what is under it. Symlinks are resolved
first, so a link planted in a world-writable directory is judged by its
target. ACLs and MAC labels are not consulted, so the answer can be stricter
than the kernel and never looser — the safe direction, since the cost is a
visible refusal naming the path rather than a silent grant.

**It arms itself, per CONNECTION.** There is no mode to declare and nothing
classifies a deployment: the check is skipped when `peer.uid == os.getuid()`
and runs otherwise. So one daemon skips its owner's client and checks a
colleague's, and a daemon under a dedicated service account checks every
human connection there is. Skipping the daemon's own uid is not a
single-user carve-out — that account can already `ptrace` the daemon, read
its memory and read `~/.jaato`, so a refusal would deny nothing it cannot
obtain more simply; what the skip buys is that the common single-login case
pays no `stat`. A control nobody remembers to enable is a control nobody
has, so there is no knob to switch it on — only
`--ipc-trust-peer-paths` to switch it off, announced at WARNING.

**Positive evidence only**, the posture #1014 and #1023 take about
confinement labels — with the direction chosen per question. `None` from
`peer_credentials` means *this transport cannot tell me* (a Windows pipe, a
non-Linux socket, WS), and the guards read it as **not applicable**: that
transport's own access control is what applies, and a denial there would
break every deployment the check was never about. `None` from
`path_reachable_by` means *I could not determine this*, and there the caller
**refuses** — granting on ignorance is the failure being fixed.

**What it is not.** An ENTITLEMENT check, not a sandbox. The session still
runs as the daemon's uid, so within a tree the peer can read, the daemon's
own rights still apply. Closing that residue means dropping privileges
between `fork()` and `exec()`, which needs a privileged daemon and a
uid-keyed slot pool — a uid is a property the next session cannot change, so
`SlotKey` would have to carry it by that class's own stated rule. That is a
decision about the process model; this is what makes the current one honest.

Unchanged for everyone else, by construction: a WS deployment (no peer to
read), a Windows pipe, and any connection from the daemon's own account.

Tests: `jaato_server/shared/tests/test_peer_identity.py` (the rule) and
`jaato_server/server/tests/test_ipc_peer_entitlement.py` (the wiring — the three places a
correct mechanism could still be inert). Every deny case is paired with the
same call one `chmod` apart, because a refusal that would have happened
anyway proves nothing; the fixture roots at `/tmp` rather than using
`tmp_path`, whose `0700` parent would make every refusal pass for a reason
unrelated to the code. Verified non-vacuous: with the check neutralised,
exactly the five enforcement cases fail and the not-applicable ones still
pass.

### A Workspace Name That Left the Workspace Root

The section above is about a transport that could not tell its callers
apart. This is the other transport's version of the same question, and it
starts by saying what is NOT true of it: a WS client does not name a path
on the normal path at all. `session.new` from a client with no workspace
**auto-provisions** one under `{workspace_root}/sessions/{session_id}/`
from a template, so the tenant serving those clients decides where they
run. The `startswith(workspace_root + os.sep)` tests in `websocket.py` are
not the boundary either — they are a ROUTING gate deciding which sessions
get an AppArmor profile and a cgroup, and a session that fails one is
skipped with a debug line so IPC and user-CWD sessions pass through the
same hook.

Reuse is the opt-in, `workspace.select <name>`, and there the name was
joined onto the root with nothing in between. **A join contains nothing by
itself**, which is the whole finding:

| name | `workspace_root / name` |
|---|---|
| `../../etc` | `<root>/../../etc` — `..` is kept verbatim |
| `/etc/passwd` | `/etc/passwd` — pathlib DISCARDS the left operand |
| `..` | the root's PARENT, with no separator involved |

`create_workspace` refused `/` and `\` and so was contained by accident;
`select_workspace` validated nothing, checked `path.exists()`, and then
`_analyze_workspace`'d whatever it found — reading that directory's `.env`
and reporting its provider and model back to the client through
`get_config_status`. So the escape was also an oracle, and
`_save_registry` **persisted** the out-of-root row into
`~/.jaato/workspaces.json`, where `get_workspace_path`'s registry branch
would return it again across restarts.

`WorkspaceManager._resolve_under_root` is the one rule, and every site that
turns a NAME into a PATH goes through it — there were five, and containing
`select_workspace` alone would have left the other four as the next
route in. It resolves symlinks BEFORE comparing, which is not
belt-and-braces: provisioned session workspaces live under this same root
and the agent's own file tools write into them, so a link planted under the
root is model-reachable.

| Property | Why |
|---|---|
| **containment, not a separator check** | `..` carries no separator and resolves to the root's parent, so a character check misses exactly the cheapest escape — and misses symlinks entirely |
| **checked BEFORE `exists()`** | a refusal must not double as an oracle for what exists outside the root; a present and an absent target answer alike |
| **the verbs raise, the accessors answer** | `select`/`create` raise `WorkspaceContainmentError`, a `ValueError` **subclass** so the WS handlers' existing `except ValueError` reports it unchanged; `get_workspace_path` / `get_config_status` return their existing "no such workspace" answers, because an accessor that starts raising breaks callers that never expected it |
| **the stored path is checked, not trusted** | one accepted selection used to persist, so the registry row is re-checked rather than read back as authority |
| **`create` keeps BOTH checks** | they catch different things and neither masks the other: `".."` passes the naming check and containment refuses it, `"a/b"` passes containment and the naming check refuses it. That is not the duplicated-validation shape the reversion meta-guard flags |

**Two costs, stated rather than hidden.** A workspace an operator
deliberately symlinked into the root is now refused and must be moved or
bind-mounted — and such a workspace was *already* running unconfined,
because the AppArmor gate resolves both sides the same way and skipped it.
And containment bounds the ROOT, not the tenant: every tenant's
provisioned workspace is a sibling under one server-wide `workspace_root`,
so this stops a name leaving the root and says nothing about which
workspace inside it a client may select. Per-tenant roots are not
expressible today, and are the same shape as the other WS singletons (one
bearer-token digest, one `SSOAuth` realm, one cookie secret).

Tests: `jaato_server/server/tests/test_workspace_name_containment.py`. Every deny case
points at a directory that EXISTS, because the refusal has to come from the
containment check rather than from the `exists()` test one line below it —
against the unfixed code those selections SUCCEEDED, so a test using an
absent target would pass either way. Verified non-vacuous: with
`_resolve_under_root` reduced to the bare join, exactly the ten enforcement
cases fail and all six controls still pass.

### A Root Daemon Writes Root-Owned Files, and Nothing Said So (#1168)

The section above is about *which* paths a caller may name. This is about
**who the files under them end up belonging to**, and it is the same answer
one layer down: the daemon acts with ITS credential, and so does the runner.
`RunnerSpawner.spawn` is `os.fork()` + `os.execvpe` with no `setuid` /
`setgid` / `initgroups` anywhere on the path — tree-wide, nothing drops
privileges. So on a **root** daemon everything the agent writes into a user's
workspace is root-owned, and not one tool but all of them, because they all
run in that one process:

| What | Why it lands root-owned |
|---|---|
| `writeNewFile` / `file_edit` backups | runner-tier plugins, in the runner process |
| the directories under them | `mkdir(parents=True)`, and the result payload never names them — which is why **delete** fails, not only overwrite |
| completion processors | loaded in-process via `spec_from_file_location`, so not special in any way |
| anything a `cli` subprocess creates | it runs with `cwd=<workspace_root>`, so one `git clone` drops a whole tree in there |

The two symptoms have different causes, and the difference is what made the
cheap fix findable: **overwrite** fails because the file is `644 root:root` —
a permissions problem; **delete** fails only when the parent directory is
root-owned too.

**And the daemon never noticed.** `grep -rn 'geteuid'` across `jaato_server/server/`
returned the egress proxy's sudo decision and a cgroups writability message,
and nothing else — while every other weakened posture in this tree announces
itself at WARNING (`scrub_secret_env: none`, `--ws-unsafe-no-auth`,
complain-mode AppArmor, `interactive_shell` without `require_confinement`).
The deployment guides are already written around a service user
(`docs/apparmor-setup.md` and `docs/runtime-limits-setup.md` both
`chown jaato:jaato`), so a root daemon is off the documented path; it simply
was not *said*. `jaato_server/server/process_posture.py` says it, once per daemon process
— naming the consequence, the fix (run as a service user) and the mitigation
below, in that order.

**`--umask` / `JAATO_UMASK` is the mitigation, and it is a mitigation rather
than a fix.** `grep -rn 'os.umask'` returned nothing tree-wide, so a
shared-group deployment could not be expressed at all. `umask 002` plus a
setgid workspace makes those root-written files **group-writable**, which
fixes the reported pain — overwrite and delete without `sudo` — **without
touching ownership**. Measured on a live root daemon: pid file `644` → `664`,
and `/proc/<pid>/status` reports `Umask: 0002` on the daemon, on the pre-warm
template forked from it and on both pool slots forked from that.

```bash
python -m jaato_server --ipc-socket /tmp/jaato.sock --umask 002   # or JAATO_UMASK=002
chgrp jaato /srv/workspaces/mine && chmod g+ws /srv/workspaces/mine
```

Four properties, each attached to a way it could go wrong:

- **Host-scoped, and that is not a preference.** `os.umask` is a property of
  the PROCESS and the daemon serves every session from one process, so a
  per-session value could not be applied without racing whatever turn is
  already running — unimplementable, not merely undesirable. It would also
  miss the files the **daemon itself** puts in a workspace (session records,
  `.jaato/logs`, the provisioned tree), which no session-scoped knob reaches.
  The catalog entry in `jaato_server/shared/env_scope.py` says so, and there is no
  `AWAITING_TYPED_KEY` row because `host` knobs do not want one.
- **Applied at step 0 of `start()`, before anything forks.** The pre-warm
  template is forked from this process and every pool slot forks from the
  template, so a umask set later would not reach the slots that serve the
  **default** path. In `start()` rather than `main()` because `--daemon`
  double-forks on Unix and re-execs on Windows, and `start()` is the one
  function the surviving process runs either way.
- **Unset means the inherited umask, never a framework default.** Supplying
  one would change the mode of every file on every existing deployment, which
  is the opposite of what an opt-in mitigation may do. A **malformed** value
  is refused at ERROR and the inherited umask is kept, for the sharper version
  of the same reason: silently applying an invented mask changes every file's
  mode for a reason no operator can find in their own configuration.
- **It round-trips `--restart`.** A flag that decided the mode of every file
  the agent wrote, silently dropped on restart, is the silent-posture-change
  shape this tree announces rather than performs. A value that came from
  `JAATO_UMASK` is not persisted — the restarted daemon re-reads it from its
  own environment, as every other env knob here behaves.

**Deliberately NOT done here: privilege dropping.** The issue measures it on
an enforcing AppArmor host and finds an order that works (drop, *then*
`aa_change_profile` — which needs nothing added to the template, so the
confined runner keeps zero capabilities and cannot `setuid` afterwards). It
is still its own change: `SlotKey` must carry the uid (a slot that has
dropped is permanently that uid — #1033's own generating rule), *which* uid
needs a policy, and **WS has no answer at all**, since `get_client_peer`
returns `None` there by design (#1074). Two unknowns are also recorded
unmeasured: the result was taken with
`kernel.apparmor_restrict_unprivileged_unconfined = 0`, and an unprivileged
*already-confined* re-transition — what a reused pool slot does — was not
probed. Also not done, and for a stated reason: **chowning after write**. It
cannot be made complete (it misses the intermediate directories, every file a
subprocess writes, and anything an out-of-tree plugin writes), which is #735's
shape — a mechanism handed to each write site where one path forgets — whereas
a umask is applied once and inherited by everything downstream.

Daemon-tier artifacts stay root-owned under either mitigation, deliberately:
session records, `~/.jaato/session_workspace_index.json` and the daemon log
are written by the daemon process and are not the agent's output.

Guard: `jaato_server/server/tests/test_a_root_daemon_says_so_1168.py`, six reversions. The
uid is substituted in **both** directions — patched to 0 for the warning
cases and to an ordinary uid for the control — because patching only one side
makes the verdict depend on the uid the suite happens to run under: on a
normal runner an unconditional warning sails through the control, and in a
root container a never-firing one sails through the warning cases. The two
call-site tests are the load-bearing ones: everything else exercises
`process_posture` directly, which says the mechanism works and says nothing
about whether anything invokes it — the #1133 shape exactly.

### A Refresh Token That Rotates, and Two Sessions Refreshing It (#683)

An OAuth refresh token **rotates**: the response replaces the token that
bought it, and the old one is void the moment the provider answers. So

```
load -> stale? -> refresh -> save
```

is a read-modify-write on a shared file, and two of them at once produce
two refreshes, each rotating the other's token away. Last-write-wins
decides which superseded credential lands on disk. The loser does not
fail then — it fails at its **next** refresh, and the user is logged out
with nothing in the failure naming the cause. That opacity is why a
small defect is a P1: the symptom is an unexplained re-login.

**A daemon is more exposed than a one-process CLI, and the pool makes it
worse.** Claude Code is one process per session and still needed a
cross-process lock for this. Here, many sessions share one process,
runner subprocesses refresh independently of the daemon, and pool slots
`fork()` from a template that has already imported the auth plugins. A
cascade fanning out after an idle period is N stages waking at once
against one expired token — the normal shape, not an edge case. So an
in-process `threading.Lock` is **not sufficient**: it does not exist
across the runner boundary, and a lock object held at fork time is
inherited in a state the child cannot reason about.

`jaato_server/shared/credential_lock.py` is the one mechanism, the shape
`jaato_server/shared/completion_nudge.py` and `jaato_server/shared/apparmor_label.py` already have:
one definition, so four auth plugins cannot drift apart on it.

**A lock alone is not the fix.** It converts the race into a queue — N
sessions still perform N refreshes, politely one at a time, and the last
one still wins for reasons nobody can see. What collapses the queue into
a single refresh is **re-reading the credential after acquiring**, so a
caller that blocked adopts the token the winner just wrote. The guard
asserts `refresh_count == 1`, not "the writes did not interleave": with
the lock intact and only the re-read removed, both race tests fail `2 ==
1`.

| Ask | Where | Note |
|-----|-------|------|
| lock spans read-check-refresh-write, cross-process | `credential_lock()` — an `flock` on a sibling `<cred>.lock` | a sibling, not the credential itself: `os.replace` swaps the inode and would detach the lock from what it guards |
| re-read after acquiring | `refresh_under_lock()` | the half that makes it a fix rather than a queue |
| refresh before expiry, with a margin | `with_margin()` / `JAATO_OAUTH_REFRESH_MARGIN` | the margin was **already there** (a hardcoded 300s in each provider); it is now one definition and configurable |
| a transient failure is not a logout | `TransientRefreshError` vs `InvalidGrantError` | only an explicit OAuth error code reads as a dead credential |

**One primitive covers threads and processes.** `fcntl.flock` attaches to
the open file *description*, and each acquisition opens its own — so two
threads of one daemon contend exactly as two processes do. That is why
there is no second in-process lock: two locks are how a deadlock is
built. The corollary is that the lock is **not reentrant**, so call sites
stay flat — a helper called from inside the lock must not take it again.

**Nothing is held at import time**, which is the fork answer: the
descriptor lives inside one call, so a pool slot forked from the template
inherits no lock state. For the residual case — a `fork()` landing while
another thread holds it, where the child inherits a descriptor holding a
lock it never took — live descriptors are closed in the child by an
`os.register_at_fork` hook.

**The margin does not disperse a herd, and the docs no longer imply it
does.** Every process computes the same threshold from the same
`expires_at`, so N sessions cross it at the same instant exactly as they
crossed real expiry — the margin only moves the instant earlier. Its real
job is to make the refresh happen while the old token is still **valid**,
which is what gives a transient failure something to fall back on: a
token inside the margin is stale but still accepted, so a network blip
there returns the stored token instead of raising.

**The asymmetry in classifying a failure runs one way, deliberately.**
Mistaking a dead grant for a transient costs one wasted retry; mistaking
a transient for a dead grant costs the user their session — the same
symptom the lock exists to prevent, arriving by a different route. So
only `invalid_grant` / `invalid_client` / `unauthorized_client` /
`invalid_token` in the body is read as a dead credential. A bare 401 from
a proxy, a 429, a 502, a connect timeout: all transient. Both subclass
`RuntimeError`, so existing callers that catch it are unaffected.

**Which plugins actually share the pattern.** The issue names four; the
tree says three, and the third is different in kind:

| Plugin | Rotates? | Wired |
|--------|----------|-------|
| `anthropic_auth` | yes — `data.get("refresh_token", refresh_token)` | `get_valid_access_token` under the lock |
| `antigravity_auth` | yes (Google) | `get_valid_access_token` **and** the provider's per-request `_refresh_token_if_needed`, which had its own unlocked refresh and was the hot path |
| `github_auth` | **no** — the stored device-flow token has no refresh token at all | the *Copilot* exchange is locked anyway: N sessions each made that rate-limited call, and `save_copilot_token` is a read-modify-write of the file that also holds the OAuth section |
| `zhipuai_auth` | **no refresh path exists** — a static API key | nothing to wire |

**`save_accounts` rewrites the whole account file**, so Antigravity's
locked helper writes back the manager it read *under the lock* rather
than the long-lived one the provider holds — saving a stale manager
republishes every other account's stale tokens over whatever another
process just refreshed. Not fixed here, and worth knowing: the remaining
`save_accounts(self._account_manager)` calls in `shutdown()` and
`_rotate_account_on_rate_limit` are still whole-file last-write-wins
across accounts. That is a merge problem, not a locking one, and wants
its own change.

### Identity at Connect (#1074)

A WS client was attributed to a user by a **message** — the `auth.token`
frame jaato-premium's `session_reconnect` extension validates against the one
`auth.issuer` in `~/.jaato/servers.json` — and two things follow that need
not:

1. **one daemon serves one realm.** A second application with its own
   userbase cannot share it: its users' tokens fail signature validation
   against the configured realm's JWKS.
2. **identity is opt-in, so declining to present one is the permissive
   path.** `client.user_id` is `None` until `auth.token` succeeds, and the
   ownership guards read `if user_id and journal.created_by and ...` — so a
   client that completes the bearer handshake and never sends `auth.token`
   short-circuits all three.

> The premium half of that second claim is asserted by the issue and was
> **not verifiable here**: `jaato_premium` is not in this checkout. What this
> change does is make the shape unreachable on its own route, not patch
> premium's.

**"Teach the daemon about realms" is the wrong repair.** A `tenants:` block,
one `SSOAuth` per realm, per-tenant JWKS — it puts an identity provider's
domain model inside the transport, and identity still arrives as a message
that can be omitted. The application already authenticated the user; it is
the authority on who they are, and the daemon re-deriving that from a JWT is
second-guessing the party that already knew.

So there are **two credential kinds**, both presented exactly where the
bearer token is presented today (`Authorization: Bearer` on the Upgrade, or
`?token=` for browsers, which cannot set headers from `new WebSocket()`):

| | held by | lifetime | authorises |
|---|---|---|---|
| **app credential** | the application's backend | long-lived, configured | `ticket.bind` / `ticket.revoke` |
| **user ticket** | one user's browser | minted per login, short, single-use | opening ONE attributed connection |

The user authenticates in the application's own realm (Keycloak, Auth0, SAML,
LDAP, an internal session store — the daemon never learns which); the
backend calls `ticket.bind`, gets a ticket, hands it to that user's client;
the client connects with it; the daemon resolves it **during the Upgrade**
and stamps the identity on the `ClientConnection`. No JWKS, no `aud`/`iss`
validation, no per-tenant `SSOAuth` — the realm never enters the daemon, and
`jaato_server/server/ws_tickets.py` is stdlib-only with no realm vocabulary in it.

**`_check_ws_token` becomes a lookup, and costs nothing extra.** It already
computed `sha256(presented)`; `_resolve_connection_auth` hashes once and asks
three tiers in order — the shared digest (`hmac.compare_digest`, byte-identical
to before), the app-credential store, the ticket registry — returning an
identity instead of a bool. Neither store holds a plaintext credential: `bind`
hands the ticket to its caller and keeps the digest, and the credentials file
is hashed at load. A dict lookup is not constant-time and is the right
primitive anyway: what it compares is a **digest**, and a timing signal about
one is not a timing signal about the credential that produced it.

**The daemon qualifies the identity itself.** `app_id` comes from the
credential that called `bind` — there is deliberately no such field on the
request — and what reaches `Session.created_by` is `BoundIdentity.qualified`,
`f"{app_id}:{user}"`. `preferred_username` is unique only within a realm, so
two applications each holding an `alice` would otherwise collide and the
ownership guards would silently pass *across* the boundary. An `app_id`
containing `:` is refused at load for the same reason: the qualified form is
a concatenation, so `a:b` + `c` and `a` + `b:c` are one string for two
identities.

**Both verbs are request/result PAIRS carrying a `request_id`** — the
protocol-1.3 shape — because one long-lived bind connection serves many
concurrent logins, and a result that cannot be attributed to a request is
useless to the backend that sent it. Protocol **1.10**.

Four properties, each attached to a way it could go wrong:

- **The ticket is spent at connect.** A single-use ticket peeked at rather
  than consumed opens any number of connections, which is the difference
  between a connect credential and a short bearer token. `_check_ws_token`
  survives as the **non-consuming** predicate, because a bool-returning
  helper that silently spent a credential would be a trap for its second
  caller.
- **An app credential binds and nothing else.** It authorises minting
  identities; letting it also drive a session would make it a super-user of
  every workspace on the daemon, attributed to no person. Enforced in
  `_dispatch_client_message` on the connection KIND rather than on a list of
  verbs, so a verb added later is covered whether or not its author
  remembers. The privilege is not transitive either — a ticket connection is
  a user, and its `ticket.bind` is denied.
- **Revocation is scoped to the binding application**, on both routes
  (one ticket, or every ticket of one user — the logout path). One
  application must not be able to log out another's `alice`. A ticket
  belonging to another application answers `not_found`, the **same** answer
  an unknown ticket gives, so the verb is not an existence oracle across the
  boundary.
- **A user the daemon would read as unauthenticated is refused at the door.**
  `created_by=""` is falsy, so an empty `user` reproduces the very fail-open
  this mechanism exists to close, arriving through the front door. Control
  characters are refused too: the value is logged and persisted.

**With no app credentials configured, behaviour is byte-identical to
before** — a hard requirement, not a preference. `_app_credentials` falsy
means no connection can ever be an app-credential connection, so no ticket
can be bound, so the ticket tier is unreachable; `_resolve_connection_auth`
collapses to the single shared-digest comparison, and both verbs answer
`denied` naming the flag. `--ws-app-credentials` without `--web-socket`, or
with `--ws-unsafe-no-auth`, is refused at startup rather than accepted into a
posture where it authorises nothing.

**Four open decisions, left open deliberately:**

| Decision | What the code does today | How to change it |
|---|---|---|
| may an app credential open a session? | **no** — the issue's own "fail-closed suggests no" | one predicate, `_dispatch_client_message`'s `AUTH_KIND_APP` gate |
| where bindings live | **in memory**, one `TicketRegistry` per daemon. A restart invalidates outstanding tickets (users re-login); a ticket bound on one node does not resolve on another, which matters where premium's gossip clustering is in play | `TicketRegistry` is the whole persistence surface — a shared store substitutes there and nowhere else |
| the config surface | a JSON object mapping `app_id` to credential, one shape | `load_app_credentials` is the whole format surface; a richer per-application form (`workspace_root`, `config_root`) is additive to it alone |
| degradation | opt-in, byte-identical when absent | — (the hard requirement) |

**What it does not do.** Every session still runs as the daemon's uid. This
segregates identity, attribution and — with per-app `workspace_root` /
`config_root` — configuration and filesystem confinement; not the OS
principal. Separating *that* needs a privileged daemon and a uid-keyed slot
pool, and is a decision about the process model rather than about the
transport. [Two Principals on One Socket](#two-principals-on-one-socket)
reaches the same line from the other transport: `SO_PEERCRED` tells the daemon
which OS account opened the socket and binds the paths that account may name,
and the session still runs as the daemon's uid. Two transports, two ways of
learning who is calling, one process model neither of them changes — and the
two identities are not interchangeable, which is why `get_client_peer` answers
`None` on WS however firmly a ticket has established a user.

**Not addressed here:** premium's JWT route keeps working unchanged and is
not on the critical path any more, but the three defects the issue attributes
to it (`claims.validate()` with no `claims_options`, `verify=False` on the
discovery and JWKS fetches, an unchecked OIDC `nonce`) are in a package this
checkout does not contain and were neither confirmed nor fixed. Neither was
`gossip/ws_auth_proxy.py`, the cookie route, which needs its own answer to
"which application is this".

### Both Transports Authenticate, and One Path Threw It Away

Two transports now learn who is calling — IPC from `SO_PEERCRED`, WS from a
bound ticket — and each has a suite proving its own `get_client_user` returns
the right string. A third family proves `created_by` round-trips once
something has set it. **Nothing tested the joint**, and the joint is where the
defect was.

The chain is one spine with two heads, and `CommandRouter` is the single
consumer that asks:

```
IPC  SO_PEERCRED ──▶ JaatoIPCServer.get_client_user()  ──┐
WS   ticket bind ──▶ JaatoWSServer.get_client_user()   ──┤
                                                         ▼
                        CommandRouter (transport-agnostic)
                 ├─ create_session(created_by=…)
                 └─ handle_request(user_id=…)
                                                         ▼
      Session.created_by ─▶ record 2.9+ ─▶ envelope ─▶ set_client_user_id()
                         ├─ ledger `response` / `permission-check` user_id
                         ├─ the #951 DECISION line's `user_id=`
                         └─ the OpenInference `user.id` span attribute
```

`session.default` did not ask. `CommandRouter._handle_session_default` →
`SessionManager.get_or_create_default` → `create_session(client_id,
workspace_path=…)`, with no `created_by` — while the two siblings twelve lines
away in the same file both read the sink. `_create_session_impl` declares
`created_by: Optional[str] = None`, so the omission was not a `TypeError`, not
a warning, and invisible in the resulting session: the record simply carries no
creator and every downstream consumer OMITS the key rather than reporting an
absence. `IPCClient.get_default_session()` is its only caller and is a public
SDK method, so a session opened that way was anonymous in all four artefacts on
**both** transports, however well each had authenticated its client.

**Attribution applies to the create branch only.** `get_or_create_default` has
two attach branches, and they leave `created_by` alone: it records who brought
a session into existence, so re-stamping someone else's session with whoever
attached next replaces a true fact with a plausible one — the rule
`_emit_to_client`'s session stamper already states one field over.

**The identity is read at the router, never below it.** `SessionManager` holds
an event *callback*, not an `EventSink`, so it cannot ask; and the event body
must never be able to claim an identity the transport is the only thing that
knows. That is why the new parameter is threaded down rather than resolved in
place.

Two guards, because they answer different questions:

| Guard | Asks |
|---|---|
| `test_attribution_reaches_both_transports.py` | does the value survive the joint, on each transport, on each of the three consumer paths |
| `test_every_session_creation_is_attributed.py` | is there a FOURTH path — an AST scan over every `SessionManager.create_session` call site |

The first builds **real** sinks: a `JaatoIPCServer` holding a fabricated peer
and a `JaatoWSServer` holding a ticket resolved through the real
`_resolve_connection_auth`, so the string under assertion is the one each
transport actually derives rather than one the test wrote down. What it fakes
is `SessionManager`, which is the right half to fake — the record/ledger/trace
end is already covered by the round-trip suites, and what was missing is
whether the value ever ARRIVES. A behavioural test cannot cover the second
question at all: it would have to know about a call site to exercise it, and
the failure being guarded against is a call site nobody thought about.

**The AST guard uses a receiver ALLOW-list, not a skip-list.**
`create_session` is also a method of `JaatoRuntime` — a different call with no
`created_by` parameter — so a guard that skipped receivers it did not recognise
would silently stop covering the daemon the day someone renames
`_session_manager`. An unrecognised receiver fails and must be classified.
`create_headless_session` is the one exemption, with its reason recorded: its
client is the synthetic `_HEADLESS_CLIENT_ID`, so no sink can answer and a
value there would be invented rather than authenticated.

**Open, and deliberately not decided here:** whether a reactor-spawned headless
stage should INHERIT its cascade driver's creator, the idiom
`_create_subagent_session` already uses (`created_by=self._creator_of(parent)`).
It is a real question — a cascade stage does belong to whoever drove the
cascade — and it changes attribution for every reactor-spawned session, so it
wants its own change rather than riding this one.

### A Reactor an IPC Deployment Could Not Trigger (#1167)

`ExternalEventRequest` is how a host pokes a running session from outside —
`{name, data, timestamp}` becomes an `EXTERNAL_EVENT` on the session's
`EventBus`, reaching every agent that called
`subscribeToEvents(event_types=['external_event'])` and sinking onward to
`SessionManager.reactor_event_bus`. It was handled in exactly **one** place in
the tree: `JaatoWSServer._handle_external_event`. `session_manager.py`,
`command_router.py` and `ipc.py` contained zero occurrences.

So an IPC client's request deserialized correctly — it is in
`deserialize_event`'s registry, so it never took `ipc.py`'s unknown-type
branch — fell through every `isinstance` arm of
`SessionManager.handle_request`, and hit the final `else`:

```
ErrorEvent(error="Unknown request type: ExternalEventRequest",
           error_type="RequestError")
```

**Loud, not silent**, which is the good direction and bounds the severity:
this is a missing feature that announces itself rather than a member of the
silent-ignore family (#910 / #925 / #947 / #950 / #1133). What it cost is
still real — that WS publish was the only `EXTERNAL_EVENT` producer under
`jaato_server/server/`, so an IPC-only deployment ran a daemon-wide reactor engine nothing
could externally trigger. `session.wake` is not a substitute: it drives a turn
on one session and publishes no bus event. The one workaround, the `webhook`
plugin's listener (the tree's other producer, transport-independent), costs a
bound port, TLS, an allowlist and a signed route — to deliver an event from a
process already holding an authenticated socket to the same daemon.

**The dispatch is what was missing; the translation is now shared.** Adding an
IPC branch alone would have given the tree TWO answers to *what an external
event looks like on the bus* — the shape this file already names as a defect
in its own right, because each copy masks the other and neither can be shown
to do anything (#688's `on_unmetered` validation, which the reversion
meta-guard correctly called decorative). `jaato_server/server/external_event.py` is the one
answer; each transport keeps only what is genuinely its own, which is
resolving WHICH session the caller is driving.

| Surface | |
|---|---|
| `publish_external_event(server, *, name, data, timestamp, source)` | `jaato_server/server/external_event.py`. Resolves `jaato_server.server._runtime.event_bus`, builds the bus event, publishes. Returns an `ExternalEventDelivery` — never raises for a missing bus, because both callers answer their client and a transport handler that raised would cost the connection |
| `SessionManager._handle_external_event_request` | the IPC arm, `source="ipc"` |
| `JaatoWSServer._handle_external_event` | unchanged behaviour, now a caller, `source="websocket"` |
| `IPCClient.send_external_event()` / `JaatoClient.sendExternalEvent()` | the SDK methods |

**Both SDKs gained a method, because neither had one.** `ExternalEventRequest`
existed as a TYPE in each and as a METHOD in neither, so the only producer in
practice was an out-of-tree web component hand-rolling the JSON frame — which
reframes the gap: it was not only "IPC lacks what WS has" but "one component
is the sole client of a protocol message the SDKs describe and cannot send".
An empty `name` is refused client-side (it matches no `event_names` filter and
renders a blank `Type:` to the model), and an absent `data` is sent as `{}`
rather than `None` — a name with no payload is a legitimate ping.

**`source` names the transport, and that is checkable rather than assumed.**
It is rendered to the model as `Source: <x>` by
`jaato_server.shared.event_bus_tools._format_event_notification`, so the parameter has no
default: a module that guessed one would be making a provenance claim on every
caller's behalf. `SessionManager.handle_request` serves BOTH transports and
cannot ask which one it is on — it may say `ipc` only because
`JaatoWSServer._handle_message` intercepts and returns *before* delegating to
the `CommandRouter`. That ordering is an invariant, and
`jaato_server/server/tests/test_external_event_over_ipc_1167.py` fails if the interception
is removed, rather than letting WS traffic quietly start arriving at the IPC
arm and being labelled `ipc`.

**No protocol bump.** No event type and no field is added — only a dispatch
for a message that already existed, already deserialized and already had a
registry entry — so `PROTOCOL_VERSION` stays at 1.18 and neither SDK declares
a floor. The 1.7 missing-verb rule is what would have forced one, and it does
not apply: an older daemon answers that named `ErrorEvent` on the event stream
rather than ignoring the request. A floor would also be **wrong in the other
direction**, since the version is transport-agnostic: it would refuse against
every WS daemon where the same request has always worked.

**A cost, stated.** The complexity ratchet frozen `handle_request` at 101 and
a baselined function may not grow, so the branchiest of the six permission
arms — the only one with a three-way `target` switch — was lifted into
`_handle_permission_remove` to pay for the new branch. Same move as #812's and
#1069's; the chain is shorter than it was (101 → 93) and the arm's behaviour
is unchanged.

### EU AI Act Mechanisms

Regulation (EU) 2024/1689 addresses the **provider** and **deployer** of an
AI system; a jaato *application* (profile + persona + tools + model binding)
is the system and the framework is a component supplier. The assessment is
[docs/design/eu-ai-act.md](docs/design/eu-ai-act.md). Thirteen mechanisms
exist in the tree, each the smallest shape that makes an obligation
expressible:

| Obligation | Mechanism |
|---|---|
| 6(4) document the risk determination | `regulatory:` profile block — `{intended_purpose, risk_class: minimal\|limited\|high, annex_iii, provider: {name, contact}, interacts_with_persons, disclosure_text}`. Declared, never inferred. `risk_class` inherits most-restrictive-wins, the rest child-replaces. Rides every ingress incl. the isolated-runner payload |
| 50(1) say you are an AI | the `disclosure` instruction piece (`jaato_server/shared/ai_disclosure.py`), fourth beside `disk` / `constants` / `security`; kept by the blanket `suppress_base_instructions: true`, dropped only by name and then announced at WARNING (`ANNOUNCED_PIECES`) |
| 50(1) *before* the first turn | the announcement `announcement_for()` decides and `SessionManager._announce_ai_interaction` emits once at session creation — `AgentOutputEvent(source="system")` plus `SessionInfoEvent.disclosure_announcement` (protocol 1.15) for a client that owns a medium the framework cannot reach. `PresentationContext.client_discloses_ai` is the "unless this is obvious" suppression, asserted by the only party that can see the screen |
| 50(1) *prove* they were told | the `announcement` audit event (#1157), in the LEDGER because that is the store that chains: `text` as delivered, `client_type` + `locale` off the client's `PresentationContext` (`locale` is new, BCP 47, declared by the client and never defaulted), the `provider` / `model` pair, `session_id`, `created_by`. Written by `jaato_server/shared/ai_disclosure.py::announcement_record` through `JaatoServer.record_disclosure_announcement` on **every** outcome: `delivered: true` with the text, or `delivered: false` with a `withheld_reason` from a closed vocabulary — `client_discloses` (the client took the obligation), `headless` (no transport serves the client id, so the emit reached no person), `decision_failed` (the predicate raised, now at WARNING rather than collapsing to "declared nothing"), `wake` / `reattach` (a revive, no new emit, and WHICH kind — a long-lived session accumulates many reattaches and no wakes). Absent is not a record of anything; a profile that declared nothing gets no row. A row with no file to land in is announced at WARNING naming `trace.ledger` / `LEDGER_PATH`, and `validate` reports the same profile as `disclosure_unrecorded` (warn; error under `risk_class: high`) before any session exists. The daemon appends it at creation, before the first turn, to the same file the runner's ledger continues (#1120: the chain belongs to the file), so it precedes the first `response`; `event_index` is per writer, the chain is the order. Found while writing it: `PresentationContext.to_dict` / `from_dict` carried neither `client_discloses_ai` nor `renderable_media`, so the #1116 suppression never held over the wire — both ride now |
| 50(2) mark generated output, on the wire | `ToolOutputEvent.generated_by` (protocol 1.14): `{"kind": "ai", provider, model, session_id, agent_id}` on the model's own media, stamped in `_deliver_model_media`; `Attachment.generated_by` for a producer's claim; nothing on a relayed file |
| 50(2) mark it so it SURVIVES the wire | `TRAIT_OUTPUT_MARKER` + `JaatoSession._mark_generated_output` at both delivery seams; the in-tree `output_marker` plugin writes `<file>.provenance.json`. The seam walks `registry.list_enabled()`, **never `list_exposed()`** — a marker provides no tools, so it is an ENRICHMENT plugin and can never be in the tool-bearing set; reading that one made the mechanism inert in the only configuration that uses it. An AST guard fails any future `Attachment` producer that neither stamps nor is a declared relay |
| 12 keep the logs | `TokenLedger` appends per record to `LEDGER_PATH`; `trace.ledger` is the typed key (relative = per session); `write_ledger` flushes only what was not appended — and the runner holds a ledger of its own (see *A ledger the runner never held*, below) |
| 12 / 13(3)(f) say WHAT is logged | `jaato_sdk.audit.AUDIT_SCHEMA` — the events, their fields, the store each lands in — rendered by `jaato-scaffold explain audit [<profile>]` and enforced by a guard that walks the writers it names. Not a sixth store: a contract over the five that already record. `docs/audit-log.md` |
| 73(6) prove they were not altered | `record_keeping.integrity: sha256-chain` — each record carries `prev_digest` + `digest` over canonical bytes; `jaato-doctor --audit-verify <path>` walks a file with nothing but the stdlib. It proves the file was not edited IN PLACE and **not** who wrote it, which the verifier's own output says. An unchained file reports as unchained, never as intact. The chain belongs to the FILE: every chained append takes an exclusive lock, reads the tail digest back off disk and links to THAT, so a restart, a shared absolute `trace.ledger` and two concurrent appenders continue one chain rather than each starting a rival one — holding the pointer in memory alone made the mechanism accuse its own normal deployment. Stated cost: a chained file cannot be pruned from the front, so retention rotates whole files |
| 19 / 26(6) keep them long enough | `record_keeping: {retention_days, conversation_retention_days, integrity}` — two clocks, because the audit record and the conversation are kept for different reasons, and they resolve differently across profiles: `retention_days` governs the files a profile NAMES, so **each profile's files are judged under its own clock** (pooling them let one profile's 30 days unlink a sibling's `retention_days: 0` record), while a session record is named by no profile, so the workspace has ONE conversation clock and the only safe reading of several is the longest. A trace path is EXPANDED, not taken literally — the provider channel splits per agent, so the siblings are globbed. Declared, never defaulted: it changes what DELETE MEANS. `workspace.delete` refuses a held record; the #812 watchdog runs an hourly pass that removes audit files past `retention_days` and session records past `conversation_retention_days`, never a loaded session's own |
| 15(4) don't feed bias back | `Memory.generated_by` (which model wrote it — stamped by the PLUGIN, never from the tool's arguments) + `curated_by` (who approved it: a second field, because who wrote it and who approved it are two facts, stamped on EVERY promotion path — the model's `update_memory` and the human curator's `%memory edit`, both through one helper — and CLEARED on demotion, since an approval that was withdrawn must not keep reading as one; an AST guard fails any future writer of `maturity` that does not stamp, because a promotion that skips it is withheld by `require_curation` with no error anywhere) + `plugin_configs.memory.require_curation`, which withholds uncurated memories from BOTH retrieval paths and says how many. `validate` warns when the knob closes the learning loop instead of mitigating it |
| 72 / 73 know what to report | `IncidentEvent` (protocol 1.16) + an `INCIDENT:` line in the application trace, raised by `jaato_server.shared.incidents.raise_incident` from the five sites that already knew; `jaato-doctor --incidents [--since 15d]` renders all three Art. 73 windows beside each row. It carries **no severity**: whether an entry is a serious incident under Art. 3(49) is a determination about consequences no log line holds. Unreadable is reported as unread, never as "none" |
| 14(4) human oversight | `jaato-scaffold explain oversight [<profile>]` — the two stop verbs, the permission gate, the built-in constraints, reversibility, read from their enforcers; `jaato-doctor` prints the exact `jaato-server --stop` for the running daemon |
| 11 / 13 draw up the documentation | `jaato-scaffold new dossier --profile <p>` — the Annex IV skeleton, computed from the same helpers `explain` reads so it cannot disagree with them; **every one of the nine headings is present**, because a section dropped for having nothing to say reads as *nothing to declare*. `--component` emits the Art. 25(4) pack for jaato itself, committed at [docs/jaato-component-pack.md](docs/jaato-component-pack.md) as the first versioned instance |
| 15(3) / 9(8) declare the accuracy | `--eval-results <file>` fills Annex IV §4 from a `jaato-eval` run: pass rate per (task, profile set), blocked arms out of the denominator, a **TODO** threshold on every row. The harness's own caveats ride IN the results file (`results_version`, `caveats`; [docs/eval-results.md](docs/eval-results.md)) and are rendered verbatim, so a number cannot be separated from the limits of the instrument that produced it. A file whose declared format this reader does not know is refused BY NAME — an absent version is an unknown version |

**`validate` reads the block.** `disclosure_absent` (warn) for a
persona-bound profile that declares nothing about interaction; under
`risk_class: high` the codes in `HIGH_RISK_ESCALATED_CODES`
(`budget_control_absent`, `budget_limits_without_abort`,
`secret_scrub_disabled`, `missing_description`,
`permission_rule_without_plugin`, `unknown_tool`, `disclosure_absent`)
become **errors**, and five high-risk-only errors name the obligation each
covers: `high_risk_without_intended_purpose`,
`high_risk_without_oversight_policy` (no `plugin_configs.permission.policy`),
`high_risk_without_record_keeping` (no `trace.session_log`),
`high_risk_disclosure_suppressed`, `high_risk_shell_unconfined`
(`interactive_shell` without `require_confinement: true`). A profile that
declares no class validates exactly as before: absent is `minimal` for
validation and *undeclared* for documentation, because a framework that
printed `minimal` for it would be asserting a determination nobody made.

**The authoring surface knows the keys, all three verbs of it.** `explain
profile` documented `regulatory:`, `trace.ledger` and `record_keeping:` and
`validate` checked them once declared — and `new profile-set` emitted none of
them, so a workspace scaffolded the documented way got a clean bill from
`validate` (only `budget_control_absent`) while declaring nothing under the
Act: `disclosure_absent` needs a persona-bound profile and the `high_risk_*`
errors a declared class. Now the tier-1 base carries the three blocks
**commented out** (a live `regulatory:` with no fields is a determination
nobody made; a live `record_keeping:` changes what DELETE means), `validate`
says once per workspace that nothing declares it (`regulatory_undeclared`,
warn — the `budget_control_absent` posture, and never inferring `minimal`),
and the Claude Code integration skill lists `explain oversight`, `explain
audit` and the `dossier` archetype. Guard:
`jaato_server/shared/tests/test_scaffold_surfaces_know_the_compliance_keys.py`, which also
checks that every topic and archetype the skill lists is one the CLI has.

**A ledger the runner never held.** Everything above about the ledger was
true of the in-process path and false of the default one. The runner's
bootstrap passed `ledger=None` to `configure_plugins` — on the reading that
token accounting is daemon-tier (§4.2 of the runner design) — while the
daemon never receives a runner session's usage into *its* ledger. So a
runner-served session wrote no `response` and no `permission-check` record
anywhere, and `explain audit <profile>` said the ledger was written: a key
parsed, validated, rendered and enforced by nothing, the #735 shape, found
by driving a live daemon for the evidence manual (#1139). The runner now
constructs its own `TokenLedger` (Step 9 of `jaato_server/server/runner/session.py`); the
path and the integrity posture are read per record through the session-scoped
env the bootstrap already applied, so it is one file per session as
documented. Guard: `jaato_server/shared/tests/test_runner_session_holds_a_ledger.py`.

**And the controls are driven live in CI, not only unit-guarded.** Every
mechanism above has a unit guard with a reversion, and #1139 shows what that
layer cannot see: the ledger key was parsed, validated and rendered
correctly, and no record ever reached disk on the default path. So
`jaato_sdk/conformance/test_eu_ai_act_controls.py` (the `conformance`
marker, in the "SDK + scaffold + conformance + eval" job) drives a real
`echo` daemon through the SDK and asserts the artefacts a deployer would
show an auditor: the announcement before the first turn (and its absence
for a profile declaring nothing), the `disclosure` piece in the rendered
prompt, a chained ledger that `verify` accepts intact and refuses edited,
a budget stop reaching the incident register, and a memory record carrying
`generated_by` behind a curation gate that holds. It is the layer that
would have caught #1139 the day the runner path shipped.

The two documents are **skeletons**, not compliance documents, and say so on
their first line: what the framework can compute is a small part of Annex IV,
and every part it cannot is left marked in the Article's own words rather than
omitted. The component pack lists what jaato does NOT guarantee at greater
length than what it does — a limitations section a reader finishes quickly is
one that leaves them unable to tell what was considered from what was
forgotten.

Nothing on `docs/design/eu-ai-act.md`'s list is now unbuilt; what remains is
named there as a decision rather than a gap (a text watermark waits on the
Art. 50(7) code of practice, and signing an audit chain is a key-management
problem this tree does not take on).

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
`jaato_server.shared.plugins.permission.plugin.parse_decision_trace`:

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

### A Prompt the Runner Rendered and Never Sent

A permission ASK on a runner-served session (the default) reached every
client with **options and no question**. The permission plugin resolves a
`PermissionDisplayInfo` from the tool's own plugin — the summary, the
unified diff for a file edit, the analyzer warnings — and parks it in
`request.context["display_info"]` before calling the channel. On the
daemon-local path the daemon's `on_permission_requested` hook rendered
that into an `AgentOutputEvent(source="permission")`. On the runner path
`RunnerRPCChannel.request_permission` forwarded tool name, args and
options and dropped the display info, so `PermissionRequestedEvent`
arrived with `prompt_lines=None` and `warnings=None` — although
`PromptPayload` and the event both declare the fields, and
`docs/jaato_permission_system.md` draws the diff on the event. The daemon
hook that used to render it is registered on the daemon-side plugin, which
is not in the loop for a runner session, and the runner never arms it.

| Client | What it showed on the default path |
|---|---|
| web (`jaato-web-coder-ui`) | a grid of raw tool arguments — the whole new file for a write, no diff, no warning |
| TUI | `🔒 Permission required` and the options bar, nothing between them |

**Rendered once, at the seam that has the information.**
`prompt_fields_from_request` (`runner_rpc_channel.py`) mirrors
`_build_prompt_lines(include_options=False)` — summary, details line by
line, `Tool:`/`Args:` when a plugin renders no display info — and the
payload carries the four fields; the daemon-side `PromptOperatorHandler`
already copied them onto the event. Details are included whatever the
`format_hint`: the daemon-local hook withheld `code` details so its output
pipeline could highlight them, and there is no pipeline on this path.

**The TUI reads the event.** `jaato-tui/permission_prompt.py` renders a
`PermissionRequestedEvent` into the same text the daemon-local hook emits
— the `<security-warning level="…">` block the buffer already parses, then
the lines — and appends it under the `permission` source, so
`set_tool_awaiting_approval` attaches it to the tool exactly as before.
Called **unconditionally** for every event rather than as a branch of
`handle_events`' `isinstance` chain, which is frozen at the top of the
complexity ratchet. No double render: a daemon-local session emits the
output event and never the requested event; a runner session the reverse.

**The mock spoke the card's vocabulary, again.** The web mock's `permit`
scenario sent `prompt_lines` and a warning on `permission.requested` — a
shape only the daemon-local path produced — so the e2e suite certified
diff rendering the default path never exercised, the same pattern that hid
the clarification defect (the card read `question_text`/`options` while
the batch wire carried `text`/`choices`). Both mock scenarios now emit
what the daemon emits on the runner path (options are
`{key, label, description}` there — no `action`), and `permit-bare` is the
same ASK from a plugin with no display info, pinning the tool-arguments
fallback.

### A Workspace Everyone Could See

[A Workspace Name That Left the Workspace Root](#a-workspace-name-that-left-the-workspace-root)
ends on a stated cost: *containment bounds the ROOT, not the tenant*. Every
WS client saw every workspace under the root, could select any of them, and
`session.list` returned every session on the daemon. That was the honest
state while a WS connection had no identity. #1074 gave it one — a bound
ticket stamps `app:user` on the connection — and nothing read it for
workspaces.

**A workspace belongs to whoever created it.** `WorkspaceInfo.owner` is
stamped from `get_client_user` at `workspace.create`, persisted in the
registry, and **preserved across re-discovery** — which rebuilds every other
field from the directory, so a `_analyze_workspace` that forgot it would
silently un-own every workspace on the first listing after a restart and
make the rule cosmetic. One predicate, `WorkspaceManager.visible_to`, is
read by the list and by both verbs, so the list can never show a workspace
`select` then refuses:

| Connection | Sees |
|---|---|
| no identity (shared bearer, no tickets configured) | everything — what it always saw |
| `app:alice` | her own, and the **unowned** ones |

Unowned is the state of every pre-existing workspace and of one created by
an identity-less connection. Deliberately **not adopted on select**: a
first-come claim on a shared directory is how a colleague's workspace
disappears from their list. Migration is a registry edit.
`WorkspaceOwnershipError` is a `ValueError` (the handlers' existing catch)
and is worded so it cannot be read as "does not exist" — that wording
invites creating it.

**Sessions follow the same boundary.** The transport reports it through
`EventSink.visible_workspace_paths` (`None` = no scoping: IPC, or a WS
connection with no identity — the `client_peer` tolerance shape, so an
out-of-tree sink contributes "no scoping" rather than raising).
`CommandRouter._sessions_visible_to` keeps a session that runs **in** one of
those workspaces or that this user **created** (`created_by`, #859), and
`session.attach` admits exactly that set — refusing the rest by name as
`ErrorEvent(error_type="SessionError")`, which is what settles a client's
`ask()` rather than hanging it (#1007). Persisted-only sessions carry no
creator in their listing, so for them the workspace rule is the whole rule.

**And a workspace can be removed.** `workspace.delete` (protocol **1.13**)
answers with one `WorkspaceDeletedEvent` whatever happened. It removes the
directory — persisted sessions included — and the registry row, and refuses:
a name that leaves the root or names the root; another user's workspace; a
workspace with **loaded** sessions (resolved by the WS server through the
session manager and handed in, because a directory a runner is confined to
is not deleted, it is a session failure with a delayed cause); a workspace
another client currently has selected. The deleting client's own selection
is cleared on both the manager and the sink adapter. The web workspace list
confirms inline before sending.

Stated cost, unchanged in kind: the session still runs as the daemon's uid,
so this is an entitlement boundary at the verbs, not a filesystem one.

### A Session You Deleted, and a Listing That Was Not Yours

Reported from a deployed client, as one symptom: the web rail's Sessions
section said **`1 of 169 noted`** collapsed and **`1 of 2 noted`** expanded,
and the single "other" row was a session its owner had deleted — still
listed, still carrying the note they had written about it. Two independent
defects, one behind each number, and a third that only the first two were
hiding.

**The 2 that should have been 1: a cold delete landed nowhere.**
`SessionManager.delete_session` reads `workspace_path` off the in-memory
`Session` it pops. A session that is not LOADED has no in-memory object, so
the field stayed `None`, `storage_dir` stayed `None`, and
`FileSessionPlugin.delete`'s `storage_dir or self._storage_path` fell back
to the plugin's own relative path instead of
`<workspace>/.jaato/sessions/`. Measured against the real class: the record
survived and the call returned `False`, so the daemon answered
`Session '<id>' not found.` — and the session was back in the next listing.

The daemon knew the workspace the whole time.
`jaato_server/server/session_workspace_index.py` exists for exactly this and its own
first paragraph says so — *"locating a COLD (unloaded) session's record
requires knowing its workspace"* — and it was consulted here only to
`forget` the entry, twenty lines BELOW the delete that needed it. An
AMBIGUOUS id (one second-granularity timestamp, two workspaces) still
resolves to `None` and still fails, deliberately: deleting the wrong
session is unrecoverable, which is the index's own stated reason for
refusing to guess.

**The 169 that should have been 2: the snapshot bypassed the boundary.**
`SessionInfoEvent.sessions` and `SessionListEvent.sessions` answer one
question — which sessions are there — and disagreed.
`_handle_session_list` renders `_sessions_visible_to(client_id)`; the state
snapshot was built from `list_sessions()`, every session on the daemon. So
the **wider** answer was the one every client received at attach and at
create, each row naming another user's session id, workspace path,
provider/model and its model-written description.

`SessionManager` holds an event CALLBACK rather than an `EventSink` and
cannot ask a transport about a client — the #1138 finding — so rather than
grow a second copy of the rule there (two definitions of *may this client
see this session* is the defect one layer down), it takes the router's own
method as a resolver. `CommandRouter.__init__` is the one place holding
both halves, and wires it in a line.

| Property | Why |
|---|---|
| **scoped per RECIPIENT, not per event** | `_emit_session_info_to_attached` builds one snapshot per attached client, because a listing bounded by one recipient's entitlement must not decide what another is shown. It also withholds the snapshot from cascade observers, which is the point: a state snapshot of somebody else's session, scoped to nobody, is what this exists to stop sending |
| **no client and no resolver both mean UNSCOPED** | IPC, an embedding process and every in-process caller keep the answer they always had |
| **a resolver that RAISES sends none** | failing open on an entitlement decision is the defect; an empty listing costs a client its completions until the next `session.list` |

**And the verb that destroys was outside the boundary.** #1113 wrote it in
terms of the two verbs that READ (`session.list` renders the set,
`session.attach` admits only members of it). `session.delete` took no gate
at all. That mattered less while the snapshot handed every id to every
client and cold deletes silently did nothing — with both fixed, an id is
the only thing between one user and another user's record, and an id here
is `YYYYMMDD_HHMMSS`. `_refuse_foreign_session` now takes the verb it is
refusing for, so the two share one rule and one wording.

**The note outlived the session, and pruning on absence is not the fix.**
A note (§ *A Note the Model Never Sees*) is keyed by session id and stored
where the daemon cannot see it, so nothing removed one when its session
went away. The obvious repair — drop notes whose session is absent from the
listing — is wrong, and *this very defect is the demonstration*: that
listing is entitlement-scoped and it NARROWS (a workspace deselected, an
identity not yet resolved, a daemon that answered 2 where the snapshot said
169), so pruning on absence destroys the only copy of what you meant to do
next because a workspace went momentarily out of view. **A note is
forgotten when the daemon SAYS the session is gone, and at no other time.**

`app/sessionDelete.ts` is that one place, because there were two routes and
only one touched the note:

| Route | Before | Now |
|---|---|---|
| `End session` on the exit plate | removed the note BEFORE the daemon was asked | deletes, waits, forgets on the answer |
| `session delete <id>` in the composer | forgot nothing | the same verb |

Removing it pre-emptively was wrong twice over: a delete the daemon now
REFUSES would have taken the note and left the session, and the other route
had no forget at all. `deleted` and `missing` both mean the session is gone
and both forget; `refused` and `silent` change nothing, which is what makes
those two the load-bearing cases in the guard.

### A Key Typed Once Per Workspace

The web client's configure form asked for the provider's API key on every
new workspace, because the daemon keeps it where `config.update` puts it:
in that workspace's `.env`. The fix is **application state in the BFF**
(`jaato-web-coder-server/src/credentials.ts`), deliberately not a daemon
vault and not an SDK verb — the daemon knows users only as `app:user`, an
application's concept, and the TUI would carry a verb it never calls. The
signed-in user's keys are stored per OIDC `sub`, encrypted at rest
(AES-256-GCM, key from a 0600 file, owner bound into the AAD), listed by
label and hint, revealed on a same-origin `POST`, and forwarded by the page
as `config.update`'s `api_key` exactly as a typed key travels. `pass` was
rejected for the unattended path: the daemon resolves `pass://` as its own
uid against its own GnuPG store, and gpg-agent's `max-cache-ttl` is
absolute, so an unattended store eventually blocks on a pinentry nobody
answers. With no `credentials:` block nothing changes, and the daemon's
`~/.jaato/<provider>_auth.json` tiers stay as they are for mono-user
installs. Design: [web-server-bff.md §12](docs/design/web-server-bff.md).

### A Note the Model Never Sees

Running several sessions at once, you lose track of what each one still
needs from **you**. A day later the picker says what the session *is* and
nothing about what you were going to do next; *"waiting on an answer about
the grace period, then re-run the e2e suite"* lives in your head or nowhere.
Two session fields exist and neither is this: `Session.name` is free text and
**create-only** (there is no rename verb anywhere in the tree), and
`Session.description` is **model-written** — it arrives as
`description_updated` from the runner and is what the picker shows as the row
name, so a note put there is overwritten by the next turn. The picker tells
you what the *agent* thought the session was about; what was missing is what
*you* meant to do about it.

**Application state in the BFF**, the call [A Key Typed Once Per
Workspace](#a-key-typed-once-per-workspace) already made. A note is per
signed-in user, never read by the model and never read by the daemon, so a
`session.note` verb would sit in the protocol for one client and the TUI
would carry it and never call it — **no daemon change, no protocol bump**.
Being BFF-side is also what makes "for the human" mean what it says: a note
in the workspace tree would be `readFile`-reachable by the agent, so it would
only be *not injected into the prompt* rather than unreachable.

`jaato-web-coder-server/src/notes.ts` reuses the credential envelope rather
than introducing a second storage posture in one process — AES-256-GCM, HKDF
from the same 0600 secret file under its own info string, owner and session
id bound into the AAD, atomic temp-file-plus-rename at mode 0600. The owner
is the OIDC **`sub`**, not the display claim (`subject_claim` is
configurable, `sub` is stable), with the consequence worth stating: a note on
a session you did not create is *your* note about their session. The note's
text is never logged. **The session id is opaque to the BFF** — it stores
text under a key, and the browser joins that key against the `session.list`
listing it already holds; nothing there models a session.

**Without a backend it still works, and says so.** A local
`npx @jaato/web-coder-ui` against a daemon has no server at all, so notes go
to `localStorage`. That is worth having and worth SAYING — they are then per
browser, and "next time I look" from another device loses them — so
`NotesApi.scope` is `backend` or `local` and the rail renders the difference.
A silent fallback is a promise the storage does not keep.

**One editor, four mount points.** A note is reachable from the exit prompt,
the resume picker's row, the rail for **this** session and the same rail for
another one, so the debounce, the save, the cap, the error rendering and the
placeholder live in `app/useNote.ts` + `components/panels/NoteEditor.tsx`
rather than in each — otherwise there are four behaviours, four bugs, and
`saved 12:04` in three dialects. The placeholder is the most valuable string
in the feature (*"What should you pick up next time?"*): an empty box gets
skipped, a question gets answered.

**The exit prompt is where a note actually gets written**, because that is
the instant you know what needs doing next. A textarea inside a plate whose
whole interaction model is keystrokes collides four ways, and each is decided
rather than left to chance:

| Key | |
|---|---|
| a letter (`d`/`e`/`r`) | free — the composer forwards keys only while IT has focus |
| `Tab` | already guarded by `inField`; cycling options from inside the field would be a trap |
| `Escape` | **guarded too.** It answered `r` unconditionally, so dismissing the field discarded a half-typed note *and* left the session. First Escape blurs, second returns |
| `Enter` | inserts a newline, so this plate is never `as="form"` |

And the one that outranks them: **the save completes before the exit action
runs.** A failed `PUT` while the client detaches anyway loses the note
silently, which is the single outcome this feature cannot have — so a failure
keeps the plate open with the reason, and the draft is never cleared on
failure because what was typed is then the only copy left. `End session`
neither saves nor deletes here — a draft about a session being deleted has
nowhere to go, and the stored note is forgotten once the daemon CONFIRMS the
delete (§ *A Session You Deleted, and a Listing That Was Not Yours*). It used
to be removed on the spot, which was wrong twice: it ran before the daemon was
asked, and it left the other delete route with no forget at all.

**The rail's Sessions section supersedes a Notes section.** `SessionScreen`
shows its picker only when no session is held, and `session list` renders
inert transcript text that scrolls away carrying no affordances — so there
was no surface on which to survey your sessions while working in one, which
is exactly the reported situation. The listing is cross-workspace, so a row
carries its workspace: two `20260917_090428`-shaped ids from different trees
would otherwise read as the same session. The note gets its **own** line
(`✎ …`) rather than a fourth `·` segment, because that line is already the
agent's voice and telling yours apart from its is the whole point; the pencil
is always drawn, never hover-gated, the lesson the Files panel's `hide` /
`ignore` actions already taught.

**And adding it cost the row its click.** The row used to BE a `<button>`,
with `Attach` a decorative `<span>` inside it — so clicking the chip worked by
bubbling rather than by being a control. A button cannot nest inside a button,
so the pencil forced the row out of a `button`, the chip became a SIBLING of
the clickable element, and it kept its `btn btn-steel` styling while doing
nothing at all. What still worked was clicking the row's text, which nothing
advertised; what every user reached for was the chip, which `.btn` renders
uppercase as `ATTACH` and with `cursor: pointer`. `Attach` is a real `button`
now and the text is inert, and its accessible name CONTAINS its visible word
(WCAG 2.5.3) — a control reading `Attach` that announces itself as "Resume" is
unreachable to a voice user asking for what they can see.

**The e2e could not have caught it**, which is the part worth keeping. It
resumes by ROLE and accessible name, and that name lived on the row's text
while the chip was `aria-hidden` and therefore had no role to match: the suite
exercised the path that worked and was structurally blind to the one that did
not. Same shape as the mock speaking the client's vocabulary, one layer in —
the test was correct and could not see the defect. The guard is
`SessionRow.test.tsx`, which asserts *where the click goes* rather than what
the row looks like: two cases that fail against the reverted chip, and a
control (the rail, which offers no `Attach` at all) that passes either way.

**A route that existed one way only.** The same picker had no way back to
the workspace list. `WorkspaceScreen` routes FORWARD into it
(`setScreen("session")`), and the only two routes back were **ending a
session** and **disconnecting** — so a workspace opened by mistake could be
left only by leaving the daemon, and the one visible escape offered to
leave sessions behind entirely (`Go to the prompt without a session`). The
button is gated on the workspace MODE rather than on a selection, because
the list's own *server-provisioned workspace* link arrives with nothing
selected and is equally worth backing out of, and it is bordered rather
than a `.link` for the reason above — what was reported is that there is
no button to find.

Going back selects nothing and destroys nothing, so the daemon's
per-connection selection survives it. That is **not** new: `endSession`
already reached the list the same way (`resetSessionState` does not touch
`workspace`), so the one residue — the list's provisioned-workspace link
opening in the still-selected workspace rather than a fresh one — was
reachable before this button and is left where it was.

**And the daemon answers the other half of that question.** #1138's `awaiting`
/ `awaiting_since` (protocol 1.17) is the only way a client working in session
A learns that B is blocked on a permission ASK or a clarification — prompt
events go to that session's attached clients, and a client is attached to one
at a time — so the row carries it as a `⚠` and a `waiting 4 min: permission`
line, and the section header counts **what needs a person** ahead of what
carries a note. Two rules the renderer holds to, both of them the protocol's
own wording: an absent `awaiting_since` is *not measured*, so the duration is
dropped rather than rendered as "just now"; and an absent `awaiting` is
"nothing is waiting as far as this daemon says", never a positive no — an
unloaded session is never reported, and a daemon below 1.17 sends nothing.

Tests: `test/notes.test.ts` + `test/routes.test.ts` (BFF), `app/notes.test.ts`
(both stores, and the text normalisation the two sides must agree on), and
`components/prompts/ExitPrompt.test.tsx`, which is deliberately two cases and
a control — **Escape inside the field does not answer the prompt** and **a
failed save keeps the plate open** — because everything else about that
component is markup, and a test asserting markup pins only the markup. Both
were verified to fail against their own reversion. Not done, deliberately:
`session.rename` (framework-side, its own issue) and sorting the picker by
"has a note". Design: [web-server-bff.md §13](docs/design/web-server-bff.md).

### The Web Client on a Blueprint

The web client was a faithful port of the terminal UI: rounded cards, one
accent, panels that appear and disappear, a status bar that reads like a
log line. The redesign (Claude Design, *Jaato Web UI Redesign*, proposal
01c, the light face) keeps every behaviour and every word of the routing
model and changes the structure it is drawn on: square hairline **plates**
with registration marks (`components/layout/Plate.tsx`), Barlow Condensed
for what the interface says and monospace for what the daemon said, and one
**steel** interface accent while the theme's own colours shrink to state
glyphs. It is one layer over the theme variables: `themes.ts` derives
`--c-steel` from the theme's ground (full steel on a light one, a lighter
steel on a dark one) and departs from a theme file in exactly one place
(`WEB_OVERRIDES`: the light theme's ground is the design's paper, the dark
theme's plates are `#202223`; state colours are never overridden), so
`theme dark` and the other four still draw the same structure. `light` is
the web client's default now; the store's and the loader's defaults agree.

What each screen became: the connect plate in two columns; workspaces as a
**table** (Open / Configure / Delete per row, the configure form a plate
under it); new-session with resume and start side by side; the session with
its identity in a 46px **header** (brand, agent tabs — always rendered,
`main` included — and `ws` / `model` / `ctx` on the right), tool calls as
**rows** (glyph, name in the chrome face, arguments in monospace, duration
at the edge, the output in a ground plate under the name), user turns
numbered `T<n>` in the gutter by the pane, one **persistent rail** whose
Plan / Budget / Files sections open and close on the same `ui.show*` flags
the shortcuts and the status bar toggle, and a 26px status bar; and the
permission request as a full-width warning plate with the diff at full
measure, the focused option solid, the refusals apart at the right edge.

Two things the port turned up. The picker's silent `session.list` request
fires twice under React's development double-effect, and
`sessionListSilent` was a **flag** the first reply cleared — so the second
printed a listing nobody typed. It is a count of replies owed now. And the
e2e suite's one assertion on a button's text (`/^y yes$/`) encoded the old
key-then-label order; the design puts the key after the label, and the
test says so. Fonts are self-hosted from `@fontsource/barlow` and
`@fontsource/barlow-condensed` (latin subsets in the bundle, ~180 KB of
woff2), so a deployment behind a corporate proxy needs no font CDN.

### Three Rows the Web Client Drew That Nobody Sent

Reported from a live daemon in one evening: every `workspace.create` added
a row named after the workspace ROOT's own directory (`workspaces`) that
`select` and `delete` then refused; saving a provider left the configure
form saying `missing: provider` over an empty dropdown; and every prompt
appeared twice, the second time as agent output under a `USER` header.
Three defects of the shape this file already names — **the mock spoke the
client's vocabulary, not the daemon's** — and one daemon defect the first
of them exposed.

| Wire | The daemon sends | The client read |
|---|---|---|
| `workspace.created` | `WorkspaceCreatedEvent(workspace=...)` — a field the SDK model **did not declare**, dropped on ingest by `extra='ignore'`, so the event arrived as `{name: "", path: ""}` | a row named `""`; clicking it selected `""` |
| `config.updated` | `workspace`, `provider`, `model`, `success` — what was WRITTEN, no status field | as a `config.status`: `configured=false`, an empty `available_providers`, and `missing: provider` for the provider it had just saved |
| `agent.output` | every prompt echoed with `source: "user"` (on send, and again on a replay to an attaching client) | a text block, rendered as agent output |

`WorkspaceCreatedEvent` now carries `workspace` (the row, as the list
renders it) beside `name` / `path`; `WorkspaceListEvent.root` is finally
sent. The store merges `config.updated` over the status it holds and
updates the table row; and a `user`- (or `parent`-) sourced output line is
the user's turn — it confirms the bubble the composer already drew when
the texts match, and becomes a user bubble of its own otherwise (a replay
after attach). The mock now emits all three in the daemon's shape.

**And the root is not a workspace.** Selecting `""` reached
`_resolve_under_root("")`, which is the root itself, and `_is_under_root`
accepted it (`path == root or ...`). `_analyze_workspace` named the root by
its basename, the cache held it under the key `""`, and the registry got a
row nothing could act on — recreated on every click, which is why the
stale-row prune one section up did not catch it (the root IS a directory).
`_is_under_root` is strictly beneath now, so an empty name, `.` and
`mine/..` are refused as `WorkspaceContainmentError` wherever a NAME is
resolved — `select`, `delete`, `get_config_status`, and the registry-path
branch of `get_workspace_path`. The naming rule `create` always applied
(`_check_name`: one flat component) binds `select` and `delete` too, and a
verb that resolved a name passes it to `_analyze_workspace`, so the cache
key and `WorkspaceInfo.name` cannot disagree for a symlinked entry either.
Containment is still checked first, so a traversal is refused as one and
before existence. Tests:
`jaato_server/server/tests/test_workspace_root_is_not_a_workspace.py`.

### A Plan Nobody Was Watching, and a Step That Was Not a Failure

Two more from the same evening, one on each side of the tool row.

**`createPlan` completed and every client said "no plan yet".** `todo` is
runner-tier, so on the default path the plugin reports into the RUNNER's
instance, whose reporter was the bootstrap's `MemoryReporter` (events
stored, read by nobody), while the daemon's `_setup_plan_hooks` armed a
`LivePlanReporter` on the DAEMON's instance, which no runner-served session
calls. Not one `PlanUpdatedEvent` crossed the wire for a runner session; the
TUI's Ctrl+P panel and the web rail were fed by the same absence. It is the
description-callback gap (`description_updated`) with a different plugin,
closed the same way: `RunnerRPC._install_plan_reporter` swaps the reporter
per turn for one whose callbacks emit `plan_updated` / `plan_step_updated` /
`plan_cleared` / `plan_output` frames (the reporter's own dicts, unconverted),
hands the same reporter to the subagent plugin, and restores both on exit;
the daemon's `_PURE_NOTIFICATION_EVENTS` table turns the four frames into
the plan events through `_plan_updated_event` and its siblings, which are
now the one place a reporter's `description` becomes the event's `content`,
so the in-process and runner paths cannot disagree about a step. The table's
builders take the server too, because a profile name resolves to an agent id
through `_agents`, which no payload carries. `_setup_plan_hooks` stays for
the embedded and standalone-WS sessions that are its actual audience, and
its docstring now says so.

**A completed step drew as a failed call.** `setStepStatus` answered with
`"error": step.error`, which is `None` for a step just marked completed, and
`tool_result_is_error` read `"error" in result` — so `{"error": None,
"result": "Proyecto creado correctamente"}` was `is_error_result=True`: a red
✗ in every client, an error in the reliability plugin's ledger, `is_error`
on the telemetry span. A null error is the ABSENCE of one, and the helper
now says `result.get("error") is not None`; the `background` plugin answers
with the same shape on success and is covered by the same line. The todo
plugin also stops spelling a step's own failure as the tool's: a step the
model marked `failed` is the tool doing what it was asked, so its text
travels as `step_error`. Tests:
`jaato_server/server/runner/tests/test_plan_reporter_bridge.py`,
`jaato_sdk/tests/test_tool_result_is_error.py`.

**Two web-client touches from a tablet.** The Files panel's `hide` / `ignore`
actions appeared on hover only, and a touch screen has no hover, so on the
tablet the panel was first tried on nothing could be hidden or ignored; they
are always drawn now, dimmed until the row is hovered. And the rail has a
drag handle on its left edge (`components/layout/RailResizer.tsx`): a
`separator` that resizes by pointer — mouse, pen or finger, `touch-action:
none` — and by arrow keys, clamped to 220–720px and remembered per browser
(`ui.railWidth`, `localStorage`).

### A Policy You Could Read and Not Change

The status bar's `permissions ask` segment reported the effective default
and named the command in its tooltip, and was inert text. So the reading
and the doing were in different places, and only the reading was on
screen — you learned the session's posture from the foot of the page and
then had to know a command to act on it.

It is a button now, opening a plate over the status bar with the
`permissions` verbs **whose arguments are a closed set**:

| | |
|---|---|
| default | `ask` / `allow` / `deny`, the one in force marked `aria-pressed` |
| suspend | `--turn` or until idle — replaced by **Resume** once suspended |
| show, clear | `permissions show`, `permissions clear` |

**What is absent is the design.** `allow`, `deny` and `check` take a tool
NAME — an open set the composer already completes from the daemon's own
inventory — so the plate names them in prose and sends you there rather
than building a second, staler picker. The split is *closed set clicks,
open set types*.

**Every action goes through `submitInput`**, the path a typed command
takes. A button speaking to the daemon directly would change the
session's permission posture with no record in the transcript of who
asked for it, and would be a second expression of `permissions` free to
drift from the first.

**Suspension outranks the default in the rendering**, because it outranks
it in the daemon: while prompting is suspended no policy is being
consulted, so none is drawn as in force and the plate says why. Drawing
`ask` as current there would be a true field rendered as a false claim.

**And the test found a real defect, which is the reason to record how.**
The plate's trigger sits OUTSIDE it, so the outside-click listener fires
on the trigger's `mousedown`, closes, and the trigger's own `click` —
which arrives after — reopens: a button that cannot be clicked shut. It
was invisible to the first draft of the test because `fireEvent.click`
dispatches **no `mousedown`**, so the listener was never exercised by the
opening click at all. A first attempt at defending it (deferring the
listener by a tick) survived its own reversion, which is what exposed
that the test could not see the mechanism. Modelling a real browser click
— `mousedown` then `click` — made both the defect and the fix visible:
the listener treats the anchor as inside, and the deferral is gone as
something that could not be shown to do anything.

Guard: `components/prompts/PermissionsPlate.test.tsx`, nine cases;
dropping the anchor exclusion fails exactly the toggle case.

### A Budget Panel That Showed Something Else

Reported as *"this is not the same budget panel as the TUI"*, and it was
not. The rail's **Budget** section rendered the TUI's `context` readout —
how FULL the window is — under a heading that says Budget. What a budget
answers is the other question, *what is the window spent ON*, and the TUI
keeps the two apart: Ctrl+B opens the instruction budget, `context` is a
command.

**The data was on the wire the whole time.** `InstructionBudgetEvent`
carries `InstructionBudget.snapshot()` — per-source tokens with each
layer's GC policy — and the daemon emits it from six sites. Neither
`INSTRUCTION_BUDGET_UPDATED` nor `budget_snapshot` appeared anywhere under
the web client's `src/`. A typed event, declared in the TS SDK, read by
nobody: the same shape as the plan reporter and the subagent hooks, one
layer out — the mechanism complete except for the consumer.

The section now renders both, budget first:

| Heading | Question | Source |
|---|---|---|
| **Instructions** | what the window is spent on, by source layer, with each layer's GC policy and a drill-down into its children | `InstructionBudgetEvent` |
| **Context** | how full the window is | `ContextUpdatedEvent` / `TurnCompletedEvent` — unchanged |

Four rules, each attached to a way a readout starts lying:

- **The snapshot is kept verbatim, not reshaped.** It is one dict and
  several clients read it; a client-side flattening is how two readers
  start disagreeing about what a source layer costs.
- **An empty snapshot is IGNORED, not applied.** Replacing a populated
  breakdown with an empty one blanks the panel mid-turn, which reads as
  "nothing is using the window".
- **The glyph is the daemon's.** `SourceEntry.to_dict()` ships
  `indicator`; the local policy table is a fallback for a snapshot that
  carries none, never an override — a client inventing its own mapping is
  a second opinion about what `partial` means.
- **A source the table does not name is still shown**, after the ones it
  does. The snapshot decides WHAT exists; a hardcoded list deciding it
  would hide a layer added later.
- **A missing budget says which readout is missing.** The daemon reports
  this only once a session has an `InstructionBudget`, and an unexplained
  gap above a populated Context block is worse than a line saying so.

Bars are drawn against `context_limit` when the daemon reported one, so a
row reads as a share of the WINDOW rather than of the tracked total; with
no limit they fall back to the largest row, which at least keeps them
comparable with each other.

Guard: `components/panels/BudgetPanel.test.tsx`. Five of its seven cases
fail with the store handler removed and the two that do not are the
Context-side controls; the empty-snapshot case fails on its own reversion
(dropping the `length === 0` clause), which is what makes it a rule rather
than a comment.

Not done here: `BudgetRungFiredEvent` (#1069, the degrade ladder) is also
unread by this client and is also arguably "budget" — but it is an
episodic notification rather than a standing readout, so where it belongs
is a separate question from this one.

### A Subagent Nobody Could See (#1179)

`spawn_subagent` succeeded, the agent said *"Subagent spawned (id:
`subagent_1`)"*, and the only tab on screen stayed `MAIN AGENT`. The TUI
has the same blind spot — both clients render `AgentCreatedEvent`, and on
the default path nothing emitted one.

**The web client was not at fault**, which is what located the defect:
`store.ts` handles `AGENT_CREATED` fully, and `ensureAgent()` is a second
chance invoked from `AGENT_OUTPUT`, so a tab would appear if *either*
event arrived. Neither did. There is even an e2e case — *"subagents get
their own tab"* — that was green throughout, and this time for a good
reason rather than the usual one: the mock emits `agent.created` in the
DAEMON's shape, so the test was correct and the client it tested was
correct. What no web test can reach is the daemon's runner path, which is
where the event was not being produced.

`subagent` is `PLUGIN_TIER = "runner"`, so the plugin the model drives
lives in the runner process, and `SubagentPlugin._ui_hooks` is the slot
every `if self._ui_hooks:` in `subagent/plugin.py` reads. The daemon arms
its OWN instance (`_setup_agent_hooks` → `subagent_plugin.set_ui_hooks`),
which no runner-served session calls. The runner installs
`_AgentUIHooksNotificationShim` — on `session._ui_hooks`, **a different
object**. Measured against the real classes:

```
session._ui_hooks      : _AgentUIHooksNotificationShim
todo._reporter         : LivePlanReporter
subagent._plan_reporter: LivePlanReporter
subagent._ui_hooks     : NoneType        <- the slot on_agent_created reads
```

The first three lines are what makes it findable: the same install already
reaches into the registry for the `todo` and `subagent` plugins to hand
them a plan reporter (*[A Plan Nobody Was
Watching](#a-plan-nobody-was-watching-and-a-step-that-was-not-a-failure)*),
and stops one attribute short. Meanwhile the shim's own `on_agent_created`
docstring reads *"Called by the subagent plugin (PLUGIN_TIER='runner')"*
and the class docstring claims *"every `self._ui_hooks.on_X` call on the
runner side hits this shim"* — a forwarder written, tested, and wired to
a caller that was never handed it. The #735 shape: the mechanism is
complete except for the delivery.

**One install, a whole family.** `_install_subagent_ui_hooks` mirrors
`_install_plan_reporter` — per turn, save/restore, best-effort — and
unlocks more than the tab, because the plugin propagates its own hooks to
each child session (`session.set_ui_hooks(self._ui_hooks, agent_id)`): the
subagent's status, context, turn accounting and **tool activity** all
travel that slot, and every notification frame in the family already
carries an `agent_id`. The plugin's `set_ui_hooks` is a plain setter,
unlike the session's, which also overwrites `_agent_id` — that asymmetry
is why the session is assigned directly and this one is not, and for the
CHILD session overwriting `_agent_id` is exactly right.

**Restoring is not tidiness.** The plugin is registry-scoped and outlives
the call; a shim left behind keeps emitting frames under a request id that
has already been answered.

**`on_agent_output` is the second half.** It was a no-op, on the reasoning
that *"the runner-side session uses the `on_output` kwarg path (stream
frames)"* — true of the ROOT session and false of a subagent, whose output
has no other route. It is the same misclassification the comment directly
above it apologises for. A stream frame could not have carried it either:
`StreamFrame` has `source`, `text` and `mode` and **no agent id**, so a
subagent's words would arrive attributed to whoever owns the stream. A
notification frame is the only shape that can say whose output this is.
Forwarding it cannot double-emit, because on the runner path the subagent
plugin is this method's only caller — `JaatoSession` never calls it, and
`JaatoClient` (which does) is not on the runner's send path.

**Paid for at the ratchet.** The daemon demuxer's `_handle` is baselined
and a baselined function may not grow, so the seven `agent_*` forwards —
already a commented group — moved into `_forward_agent_notification`
(membership, a name test answered *before* the hooks are looked up, so it
cannot depend on whether they are wired yet) plus `_dispatch_agent_hook`
(the unpacking). The `_wire_str` / `_wire_int` / `_wire_float` readers put
"absent and null both mean the default" in one place instead of an `or` on
every field. `_handle` **91 → 58**, and both new functions are under the
ceiling.

Not measured here, deliberately: whether a subagent's output should ALSO
keep reaching the parent's stream, as it does today by a separate route.
It is a display question, the two are distinguishable at the client by
`agent_id`, and answering it means deciding what a parent tab should show
about its children.

Guard: `jaato_server/server/tests/test_a_subagent_nobody_could_see.py`, five reversions.
It asserts the plugin's slot rather than the session's, because filling the
session's is precisely what the broken tree did.

### A Policy the Enforcer Did Not Hold

Reported as *"the permission was to become a button, and now I do not even
see it"*: the status bar's `permissions ask` segment, absent entirely. It
is gated on having been **told** the policy (`{permStatus && ...}`), and
the daemon had never told that client. Driving a real daemon over IPC found
three facts, of which the report is only the first:

| | measured on a live daemon |
|---|---|
| **A** | `session.new` → `PERMISSION_STATUS ('ask', None)` |
| **B** | `session.attach` → **nothing at all** |
| **C** | after `permissions default deny`, the command's own answer says `deny (session override, was: ask)` and the `PermissionStatusEvent` beside it says **`ask`** |

**B is the reported symptom.** `emit_current_state` is the one door for
*tell a client arriving mid-session what the state is* — it replays the
agents, the conversation, the statuses, the instruction budget, the
subagents and the tool-id registry — and the policy was not among them. So a
client that ATTACHED rather than created (a reconnect, a session switch, a
resume from the picker) never learned it, and the web client resets that
field on attach, so the segment vanished and did not come back.

**C is worse, and it is why this is not a one-line emit.** There are two
`PermissionPlugin` objects on a runner-served session — the default — and
only one of them decides anything. The daemon builds its own at
`initialize()` and seeds it from the profile; the RUNNER's is the plugin
`check_permission` consults and the one a `permissions` command mutates.
`emit_permission_status` read the daemon's, so the value was true until
somebody changed the policy and wrong from then on. Emitting *that* on
attach would have made a stale fact arrive more reliably — the defect
wearing the fix as a disguise.

The segment is a **control** since the plate landed: it marks which default
is in force and offers Suspend or Resume from this value. A control whose
readout disagrees with the thing it controls is worse than one that shows
nothing — the argument `test_envelope_carries_gc` (#1133) makes about a GC
strategy displayed and never run.

So the runner is asked — `session.get_permission_status`, a **control-lane**
verb because an attach can land mid-turn and must not queue behind it — and
**a failed ask reports nothing**. Falling back to the daemon's copy is
reading the stale value this exists to stop reading, and the client then
keeps what it last knew rather than being handed a new claim. The fallback
applies only where there is no runner at all — the embedded client,
standalone WS, the legacy daemon-local path — and there the daemon's plugin
IS the enforcer, so it is the right answer rather than a tolerated one. A
runner session whose runtime carries no permission plugin is likewise
**refused, not defaulted**: `ask` invented there is the same lie one process
over.

Measured after the fix, same daemon, same probe:

```
A. create                         -> [('ask',  None)]
B. attach (was: nothing)          -> [('ask',  None)]
C. after 'default deny' (was ask) -> [('deny', None)]
D. attach again                   -> [('deny', None)]
E. after 'suspend --turn'         -> [('deny', 'turn')]
```

**The mock was right and the daemon was not**, which inverts this tree's
usual failure. Forty-three e2e tests passed throughout because
`mock/daemon.ts` emitted `permission.status` on attach — the shape the
daemon was *supposed* to have — so the suite was certifying an agreement
that only one side kept. What it genuinely could not see is the loop: the
mock advertised `permissions status` as a command and handled none, so no
test could ask whether clicking a default in the plate changes what the bar
reads. It applies the verb and re-emits now, and one case asserts the round
trip (verified to fail with that re-emit removed).

**Paid for at the ratchet.** `_dispatch_method` is baselined at the top of
the complexity table and a baselined function may not grow, so the
argument-free session READS — `get_auth_info`, `get_user_commands` and the
new one — fold into `_SESSION_READS` and one `_dispatch_session_read`
branch. Each still says what it answers, beside its handler name; the table
maps to NAMES because these are instance methods and the table is a class
attribute, and the names are literals in that file, so nothing a peer sends
can steer the lookup. 54 → **53**. `test_rpc_lane_classification` reads the
new table alongside the two routes it already read, so the served set stays
derived from the tree rather than restated.

**NOT closed here, and stated rather than implied:** an `a` / `t` / `i`
answer to a prompt re-emits only on the daemon-local path
(`on_permission_resolved`), which is dead on a runner-served session. The
runner applies the suspension asynchronously *after* `resolve_response`
hands the answer over, so a re-emit at that seam would race the change it is
reporting — and reporting the pre-change value is this section's own defect
in a third place. The honest fix is for the runner-side plugin to announce
its own policy change, which is a notification frame rather than a pull, and
its own change.

Guard: `jaato_server/server/tests/test_a_policy_the_enforcer_did_not_hold.py`, four
reversions. The `emit_current_state` case is an **AST walk of the call
sites** rather than a drive of the method: it reaches a dozen subsystems, so
a test that stubbed enough of a server to run it would be asserting the
stubs — and what the defect was is a missing call site, which is exactly
what a walk of the call sites can answer.

### A Memory Store Nobody Could See From the Browser (#1232)

The web client had no view of the session's memories. The only route was
the `memory` command, which prints a listing into the transcript, and
`memory edit`, which opens `$EDITOR` on the daemon's host — a browser
cannot drive that. The rail now has a **Memories** section, and the daemon
has four quiet verbs behind it (protocol **1.22**):

| Request | Answer | What it does |
|---|---|---|
| `MemoryListRequest` | `MemoryListEvent` (+ `request_id`, `ok`, `category`, `source`, `may_curate`) | both tiers, raw and curated; rows carry `tier`, timestamps, `usage_count`, `generated_by`, `curated_by`, source agent/session and two this-session flags, and no content |
| `MemoryGetRequest` | `MemoryGetResultEvent` | one memory with its content and evidence |
| `MemoryUpdateRequest` | `MemoryUpdateResultEvent` | a structured edit (description, content, tags) or a maturity change: approve is `validated`, dismiss is `dismissed` |
| `MemoryDeleteRequest` | `MemoryDeleteResultEvent` | the plugin's own `delete_memory` path |

Each answer echoes the caller's `request_id`. Both SDKs refuse the verbs
below 1.22 (`list_memories` and friends / `listMemories` and friends): an
older daemon answers "Unknown request type" and never the result.

**The answer comes from the runner's copy.** `memory` is
`PLUGIN_TIER = "runner"`, and the `memory` command used to run on the
runner and then fill its `MemoryListEvent` from the DAEMON's copy of the
plugin — a store no runner-served session writes to (the #1179 class).
`JaatoServer.memory_op` asks the runner (`session.memory`, a named
control-lane handler, so a refresh answers during a turn). A failed ask
answers `ok=False, category="runner_unreachable"`, never an empty list and
never the daemon's copy. The daemon's plugin answers only when there is no
runner at all (embedded, standalone WS), where it is the store. The
`memory` command's push and the `SessionInfoEvent.memories` snapshot now go
through the same method, and are withheld when the read failed.
`jaato_server/shared/plugins/memory/verbs.py` holds the one definition of what each
verb does, so the runner and the no-runner path cannot disagree.

**Only the workspace owner may change memories.** `memory_verbs.may_curate`
is the one predicate: the owner may, anyone may on an unowned workspace,
and an identity-less connection on an owned workspace may look but not
change. The identity comes from the transport, never from the request. A
refused mutation answers `not_owner` without reaching the runner, and the
list carries `may_curate` so the rail hides buttons the daemon would refuse.
Stated cost: an IPC client on a workspace the WS server records as owned is
refused, because an IPC identity is an OS account and never equals an
`app:user` owner; it still has the `memory` command.

**An approval records who approved it.** Approve and dismiss go through
`_stamp_curation`, the one writer of `curated_by`. A rail action has no
model in context, so the daemon passes the person the transport
authenticated (`{"kind": "human", "via": "memory.update", "user": …}`).
Moving back out of a curated maturity clears the stamp. Dismissing a raw
memory unlinks it from the queue, so it is gone from the next list.
Deleting from a tier now rebuilds that tier's index too.

The rail section (`app/memories.ts`, `components/panels/MemoriesPanel.tsx`)
lists the whole store by default, with a "this session only" toggle. Rows
written or retrieved in this session are highlighted, and a raw memory
says **unvetted** in words. The header reads `12 memories · 3 unvetted`.
Expanding a row fetches its content. The list is refreshed on attach and
on every `session.info`, on a successful `store_memory` / `update_memory` /
`delete_memory` in any agent, on the `memory` command's push, on window
focus, and after each rail action. A failed read is shown and keeps the
rows it had.

Guard: `jaato_server/shared/tests/test_memory_rail_1232.py`, six reversions (the daemon
answering from its own copy, the owner predicate, the gate not applied, an
approval with no stamp, the human curator dropped, and the
`handle_request` arm removed). `handle_request` stayed on its ratchet by
lifting the instruction-budget arm into `_handle_instruction_budget_request`
(93 → 80).

### A File the Browser Could Not Put in the Workspace

The premium `<jaato-task>` component (and the knowledge-manager client
built on it) ships files to the daemon two ways: inline base64
`staged_files` on the `session.new` envelope, and the canonical
`StageFilesRequest` — one TEXT frame naming the files, one BINARY frame
per file, one `StageFilesEvent` back, into the connection's selected or
provisioned workspace (`docs/sdk-file-staging.md`). The TS SDK already
carried `stageFiles`; the web coder used neither, so a browser session
had no way to hand the agent a file.

The web coder now uses the canonical verb for **both** moments, which is
what the SDK method was written for and what the docstring on the legacy
envelope field asks new clients to do:

| Where | When it stages | Why that order |
|---|---|---|
| the composer (drop, paste, **Attach**) | at once, into the session's workspace | the agent's tools read it on the next turn; the message sent next ends with a line naming the staged paths |
| the session picker, workspace selected | **before** `session.new` | the session starts with the files on disk |
| the session picker, no workspace yet | after the daemon's `session.info` | a daemon that provisions the workspace **as part of** `session.new` has nowhere to put them earlier; still ahead of the first turn |

Three properties, each attached to a way it went wrong while being built:

- **The workspace is a fact learned from the daemon, not sampled at attach
  time.** The picker is on screen the moment `workspace.select` is *sent*,
  and the store's `selected` is written when its `config.status` reply is
  reduced, a round-trip later — so a file attached in that window read as
  "no workspace" and sat queued until a profile was picked. The staging
  module subscribes to the store and stages the moment a workspace or a
  session appears. Measured against a real daemon: the picker's file is on
  disk before `session.new`, the composer's file lands under the folder
  chosen in the strip.
- **What the daemon would refuse is refused before any bytes are sent**,
  in the daemon's own words — a name that climbs or is absolute, a file
  over `DEFAULT_STAGE_PER_FILE_LIMIT`, a batch over
  `DEFAULT_STAGE_TOTAL_LIMIT` (`src/protocol/attachments.ts` mirrors the
  numbers). The daemon still checks; the client just does not stream 11 MB
  to hear "no".
- **One request per drop, requests in order.** The SDK correlates a
  `StageFilesEvent` to a `stageFiles` call by *order*, so the module runs
  every call through one promise chain; a directory dropped beside real
  files fails alone (its `File` cannot be read) rather than failing the
  batch.

The mock daemon speaks the multi-frame protocol (`mock/daemon.ts`,
`finishStaging`), including the up-front refusals, so the e2e suite drives
the real frames. Not done: `send_message`'s inline `attachments` (model
context, #838) — a file the model should *see* rather than have on disk
is a different feature with a different cost, and the composer does not
yet offer it.

### A File Staged Where the Session Was Not

Reported from a deployed client, on a workspace the user had created and
named: every indicator said the attachment was in the workspace — the chip
went `staged`, the transcript said *"Staged into the workspace: X.jpg"*,
the user turn carried *"Attached files, staged in the workspace: X.jpg"* —
and the model could not read it, at the relative path or at the
workspace's own absolute path, while the FILES panel showed only the
session's `.jaato/` and no new file.

**Neither client was lying**, and establishing that is what located the
defect. `staging.ts` marks a chip `staged` only for a name the daemon
echoed back in `StageFilesEvent.staged`, and the daemon appends a name
only after `_write_staged_payload` returned. The bytes really were
written — somewhere else.

**Two definitions of "this client's workspace", and staging used the
wrong one.** `CommandRouter.resolve_caller_workspace` is the daemon's
answer — the attached session's workspace, else the transport's session,
else what the client declared — and it already recorded why staging needs
exactly that order: *a session's path outranks a declared one because the
session's tree is what `WorkspaceMonitor` watches and what the panel
shows.* `_resolve_staging_workspace` answered separately, reading
`_client_provisioned` first: a map stamped when a `session.new` arrives
from a client with no workspace (the daemon auto-provisions one) and
cleared only on DISCONNECT, never by a later `workspace.select`.

Measured over the real WS wire, three orders on one connection:

| order | session runs in | file lands in |
|---|---|---|
| `select, new` | named | named |
| `new, select` | provisioned | provisioned |
| `new, select, new` | **named** | **provisioned** |

Only the third diverges, which is why it reads as intermittent — and it
is an ordinary flow: create a session, end it (in workspace mode that
returns to the workspace list), open a named workspace, create a session,
attach a file.

**Reordering the two stores would have fixed that row and broken the
second**, which is the whole reason this delegates rather than growing a
third ordering: after `new, select` the SESSION is still in the
provisioned workspace, and a file attached to it belongs there, not in
whatever the client selected afterwards. Both directions were measured,
and both are pinned by the guard.

The explicit-id caller was broken too, and loudly: staging with
`workspace_id="named"` compared that name against the PROVISIONED
directory's basename and refused with `workspace_not_found` for a
workspace the client had selected and the session was running in.

`_client_provisioned` stays as the LAST fallback, for a caller that
reached `provision_workspace()` directly — the one path that stamps it
without telling the adapter or the router.

Guard: `jaato_server/server/tests/test_a_file_staged_where_the_session_is_not.py`,
three reversions. It drives the real `CommandRouter.resolve_caller_workspace`
rather than a stub of it, because a stub would assert the test's opinion
of the ordering instead of the daemon's.

**A harness note worth keeping.** The first three runs of the wire probe
measured nothing: every `session.new` was refused with `envelope.model_name
is empty`, so no session existed, `attached_session` was `None` in all
three orders, and the resolver fell through to the declared workspace —
which made the fix look like it had broken `new, select`. A session's
provider comes from its WORKSPACE `.env`, not the daemon's environment,
so an auto-provisioned workspace needs the provisioner's own
`templates/default/.env` seeded. A probe whose sessions do not start
reports on the no-session path and says so nowhere.

### A File That Could Go In and Not Come Out

A remote client could put a file INTO a workspace (`StageFilesRequest`)
and take none OUT, so an asset the agent produced in a server-provisioned
workspace was reachable only by somebody with a shell on the host.
`workspace.file.fetch` (protocol **1.20**, WS only) is the download, and
it is the staging protocol in reverse: one TEXT
`WorkspaceFileContentEvent` header and, on success, ONE raw binary frame
of exactly `size` bytes. Wire details are in
[SDK file staging](docs/sdk-file-staging.md).

The web client uses it in two places:

| Where | What |
|---|---|
| the Files panel | a file's name is a button that downloads it; a deleted file, or any file against an older daemon, stays plain text |
| `offer_download`, a host tool the client registers | the model draws a download button in the chat when the user asks for a file, or when it produced one worth keeping |

**The binary frame is the next frame after its header, and that is the
whole protocol.** The header carries no id the frame could be matched by,
so `_send_to_client_with_binary` writes both under `self._lock` (the lock
every other send takes), and the TS transport holds a successful header
until its frame arrives and delivers the two as one event. `request_id`
correlates the ANSWER, so several fetches may be in flight.

**What may leave is the daemon's decision, in one module**
(`server/workspace_download.py`), because a link is something the MODEL
can propose:

| Rule | Why |
|---|---|
| the workspace is the one staging writes into (`_resolve_staging_workspace`, i.e. the router's `resolve_caller_workspace`) | one definition of "this client's workspace", as the staging fix established |
| containment is judged on the RESOLVED path, checked before existence | a link the agent planted inside the workspace cannot carry out a file from outside it, and a refusal is not an oracle for what exists there |
| `.env` (at any depth) and `.jaato/*_auth.json` are refused as `credential` | that is where `config.update` and `<provider>-auth key` put a provider key. `.env.example` and a `*_auth.json` outside `.jaato/` are the user's own files and download normally |
| 50 MB cap (`too_large`), the staging total cap | the file is read whole and sent as one frame; the read runs off the event loop |

**`offer_download` checks, it does not send.** The tool does a
`metadata_only` fetch and answers the model with what it offered, or
throws the refusal back as the tool's error, so the model is told "holds
credentials" instead of offering a button that fails. The bytes move when
the person clicks. The button sits under the tool's row and is always
visible, because it IS the tool's output and cannot hide behind the
expand toggle. It is auto-approved: it only offers, and every byte still
passes the daemon's rules. Registration is per SESSION on the daemon,
so the client re-registers on every `session.info` naming a new session,
including the re-attach after a reconnect, and never against a daemon
below 1.20.

A missing verb (the 1.7 rule): an older daemon answers `ErrorEvent
("Unknown message type")` and never the header, so the TS SDK refuses below
`MIN_FILE_FETCH_PROTOCOL` rather than wait. `_handle_message` is baselined
in the complexity ratchet, so the upload and download dispatch share one
helper, `_dispatch_workspace_file_transfer`.

Not done: directories (a zip would be a new verb and a new memory bound),
the Python SDK (an IPC client is on the daemon's host and reads the
filesystem itself), and files over the cap.

Guards: `server/tests/test_a_file_the_user_could_not_download.py` (four
reversions: climbing out, a symlink judged by its name, the `.env` rule,
and the frame order), the TS SDK's `fetchWorkspaceFile` cases (three fail
with the transport's pairing removed), and two e2e cases against a mock
daemon that sends the header and frame in the daemon's shape.

### Markup That Reached the Transcript as Tags (#1191, #1193)

Reported as two web-client defects: `<j-table>` tags showing as text inside
a code block, and a notebook cell arriving wrapped in literal
`<nb-row …>` / `</nb-row>` with its traceback as unstyled prose. Driving the
daemon's own formatter pipeline put two of the three causes **upstream of
every client** — the TUI showed the first one too.

**#1191 — a table inside a fence.** `table_formatter` (priority 25) runs
before `code_block_formatter` (40) and knew nothing about fences, so a
markdown table QUOTED in a fence — a ```` ```markdown ```` example, a cell's
fenced output — was rewritten into `<j-table>`, and the code-block formatter
then escaped that markup into a `<j-code>` block. Every client renders a
`<j-code>` block faithfully: monospace, line-numbered, and the literal
`&lt;j-table&gt;` the screenshot shows. The table formatter now leaves the
inside of a fence alone, and decides "inside" with the code-block
formatter's OWN two patterns (`FENCE_OPEN_RE` / `FENCE_CLOSE_RE`, imported,
never restated), so the two cannot disagree about which lines are code.

Two properties the streaming shape forces:

- **A fence opener can arrive without its newline.** Partial lines are
  passed through immediately for latency, so the completed line is judged
  with the head that already went out (`_fence_line_prefix`); judging only
  the tail means the fence never opens.
- **Chunking must not change the output.** The guard formats each fixture
  whole, one character at a time, and at twenty random cut sets, and asserts
  byte equality.

**#1193, the server half — an error cell with no execution count.** Every
early exit on the notebook's streaming path (no code, a refusal by the
containment boundary, a notebook that could not be created or does not
exist) emits `<notebook-cell type="error">` with no `exec`, and
`notebook_output_formatter` required one. Unmatched, the raw marker went to
every client — and the refusals are the messages a person most needs to
read. `exec` is optional now and such a cell is labelled `Err:`. The guard
reads every `<notebook-cell` literal out of the notebook plugin's AST rather
than listing shapes, because the defect was an emitter nobody checked
against the formatter.

**#1193, the client half — the web client had no `<nb-row>` renderer.**
The TUI has had one since the notebook shipped. `src/protocol/nbmarkup.ts`
is the parser and is bounded to its own tags, as `jmarkup.ts` is to `<j-*>`
(the boundary the TUI pins with `test_nb_row_is_not_j_markup`): a row's body
is handed back raw so an input cell's `<j-code>` still highlights. A cell
renders as a two-column grid, label beside body, with `error` / `stderr`
rows in the error tone and program output verbatim — the server does not
escape it, so a traceback's `File "<cell>"` must reach the screen as
written. **An unterminated row stays text**, as an unterminated `<j-code>`
does: no producer the web client sees splits a row, while a model quoting
the tag in prose is entirely possible. And the tool-row gate asked only
`includes("<j-")`, so a `print(42)` — a cell whose whole output is one
stdout row — went to the raw `<pre>` with its tags showing; `hasServerMarkup`
asks about both families.

The fixtures in `JMarkup.test.tsx` and the mock's two notebook scenarios are
the daemon's real output (the plugin's `_format_*_cell` emitters run through
the pipeline), not hand-written: a hand-written fixture is how the mock ends
up speaking the client's vocabulary.

Not fixed, and stated: `notebook_output_formatter` passes a `<notebook-cell`
marker through untouched if it is split across two chunks. Nothing emits it
that way today (tool output is formatted per chunk and the three emitters
write whole markers), so it is a latent gap rather than a live one. The
"FOLLOW INPUT" control in the #1193 screenshot is the tool row's existing
live-output **Follow** toggle, not a notebook feature.

Guards: `jaato_server/shared/tests/test_markup_that_leaked_into_the_transcript.py` (four
reversions) and the web client's `nbmarkup.test.ts` / `JMarkup.test.tsx`
plus two e2e cases.

### Prose Drawn as a Code Block, Because a Fence Never Closed

Reported with a web-client screenshot: the second half of an ordinary reply
(the model's headings, its questions to the user, its closing line) rendered
monospaced, line-numbered and syntax-coloured as one `<j-code>` block
running to the end of the message. Clients draw a `<j-code>` block
faithfully, so the defect is where the block is decided:
`code_block_formatter`.

Its opener was `` ```(\w*)\n `` and its closer any line that STARTED with
three backticks. So an opener the pattern did not recognise was passed
through as text, and ITS closer then opened a block that nothing closed:

| Input | What went wrong |
|---|---|
| `` ```c++ ``, `` ```shell-session ``, `` ```text `` (trailing space), `` ```python title="x" `` | the opener was not seen |
| a four-backtick fence quoting a three-backtick one | the inner fence closed the outer, and every later fence was read the wrong way round |
| a closer indented inside a list item | never seen at all |

The rule is CommonMark's now, in one place (`open_fence` / `Fence` in
`code_block_formatter/plugin.py`), and the table formatter imports it, as
#1191 requires:

- an opener at a line start takes any info string, and the language is its
  first word;
- a closer is a line holding only a run of the **same** character, at least
  as long as the opener's run;
- both may be indented, and the opener's indent is removed from the code;
- `~~~` fences work.

The old mid-line opener (`` text ```py ``) is kept, with its old narrow info
string, because models write it. The text held back while streaming is
unchanged: only an incomplete line that could still turn out to be an
opener waits for its newline. One behaviour changes: a closer now needs its
line to end, so a block completes when the closer's newline arrives, or at
`flush`, rather than on the closer's backticks alone.

Guard: `shared/tests/test_prose_drawn_as_a_code_block.py`, three
reversions. Every fixture is also streamed one character at a time and in
random pieces.

### A Reset the Next Reconnect Undid (#1189)

The TUI's workspace panel has `workspace_clear` (Delete): empty the list so
only files that change from now on appear. It is a reset of the starting
point, not a hide — a file the agent touches again after the reset comes
back. The web Files panel had no equivalent, and after a long session it
listed thousands of entries with no way to clear them short of a reload.

**Kept only in the client, a reset does not survive a reconnect.** An
attaching client gets a `WorkspaceFilesSnapshotEvent` it applies wholesale,
and a snapshot entry is `{path, status}` — nothing says *when* it changed.
The TUI rarely reattaches; a browser does it routinely (a tablet sleeps, a
network blips), so a naive port works in testing and stops working in use.

**The daemon numbers changes** (protocol **1.19**). The workspace monitor
stamps every flushed batch with `seq`, one more than the last, and each
tracked path remembers its latest `seq`:

| Event | Carries |
|---|---|
| `WorkspaceFilesChangedEvent` | `seq`, `epoch` for the batch |
| `WorkspaceFilesSnapshotEvent` | `seq` (latest), `epoch`, `seqs` (path → seq) |

A counter rather than a clock: nothing to skew between daemon and browser,
and ordering is all the filter needs. `seqs` is a parallel map rather than a
third key on each `files` entry because those entries are `Dict[str, str]`
and an older client validates them as such — an integer there fails its
whole event, while an unknown top-level field is ignored. The monitor hands
its callback a `ChangeBatch`, a `list` subclass carrying `seq` / `epoch`, so
every existing callback keeps receiving exactly what it did.

**The epoch is what keeps it from failing silently.** A session reload
rebuilds the monitor, which counts from 0 again; a mark of 500 compared
against the new counter hides every new change and empties the panel with
nothing saying why. `epoch` names the monitor instance and is not
persisted, so a mark from before a reload is recognisably void — dropped,
the full list shown, and the panel says so. Restored entries carry no
number of their own and read as 0. The snapshot is now sent even when
empty: it is the only way a reconnecting client learns the epoch changed.

| Client | What the reset does |
|---|---|
| web (`store/workspaceView.ts`) | keeps the FULL list plus the numbers, filters past the mark — so **show everything** is possible. Per viewer, in memory, like hide |
| TUI (`workspace_panel.py`) | `clear()` records the mark; a reattach's snapshot keeps only entries past it |
| either, against a daemon below 1.19 | changes numbered locally between snapshots; a snapshot drops the mark (the web client says why) — the old behaviour, stated |

**Not reset on the daemon**, deliberately: the monitor is per session and
shared by every attached client, so a daemon-side reset would empty the list
for everyone else watching.

Guards: `server/tests/test_a_reset_that_survives_a_reconnect_1189.py` (three
reversions), `jaato-tui/tests/test_workspace_clear_survives_a_reattach_1189.py`,
`src/store/workspaceView.test.ts`, and an e2e case that resets, touches a
listed file again, drops the connection and checks the reset held. A first
draft cleared the restored entries' numbers inside `restore()`; the
reversion meta-guard reported it decorative — unreachable, since only paths
this monitor numbered are reported — and it would have erased a number the
new monitor genuinely assigned, so it went.

### A Category Id Nobody Could Read

A discovery call reached the web transcript as
`LIST_TOOLS  category_id=c_bbc5e661`. Tool names reach the MODEL as
hash-derived ids (`t_<8 hex>`, `c_<8 hex>`; `shared/tool_id_map.py`), so a
call's arguments carry them, and the agreement is that anything user-facing
shows the name a person knows — the TUI has done so since the ids shipped
(`ui_utils.resolve_tool_ids`). The daemon already sent the mapping for
exactly this, twice: `tools.id_registry` (`ToolIdRegistryEvent`, the full
set each time) and `session.info`'s `tool_id_mappings`. The web client read
neither.

`src/protocol/toolIds.ts` is the TUI's function, and the store keeps the map
(`toolIdNames`), replaced wholesale on each receive. Two properties:

- **Resolved at render time, not at reduce time.** The registry is sent after
  tool configuration and again when deferred tools activate, so it can arrive
  after the call that used an id; a row rendered from the stored arguments
  picks the name up whenever it lands. The e2e sends the mapping after the
  call for that reason, and fails with the resolution removed.
- **Only a value that IS an id is replaced.** An id the map does not name is
  left as it is — showing the id is honest, inventing a name is not — and a
  string merely containing one is untouched.

Both argument displays use it: the tool row and the permission card's
argument grid.

### When GC Last Ran, What It Freed, and Which Policy (#1190)

The Instructions panel showed what each layer held and each layer's GC
glyph, and nothing about collection itself: not when a pass last ran, not
what it freed, not which policy was in force. All three were already on the
wire — `GCConfigEvent` (strategy, threshold, target, continuous) at
initialisation and on reconfigure, `GCEvent` for each phase of a pass with
`tokens_freed` on `completed` — and the web client read neither.

**Neither was replayed, and that is the daemon half.** `emit_current_state`
is the one door for "tell a client arriving mid-session what the state is",
and it sent neither event. A tab that attached after a pass — a second tab,
a session switch, a resume from the picker — read "no GC" about a session
that had collected an hour ago, and never learned the strategy at all.
`JaatoServer._emit_gc_state` replays both:

- **The last completed pass is kept as the event that announced it**
  (`_last_gc_pass`), so the replay carries the pass's ORIGINAL timestamp.
  "When did GC last run" answered with the attach time would be a readout
  that lies. Not persisted: a revived session starts with no record, which
  the panel reports as "no GC pass reported yet", never as "never collected".
- **The policy is replayed whatever it is, including no strategy.** A session
  with no GC is the state an operator most needs to see (#1133 is what it
  cost when invisible); the panel renders it as a warning line.

The panel (`src/protocol/gc.ts` for the wording) adds two lines under the
heading, subordinate to the tracked total: `◷ last GC 12 min ago · freed
14.2k tokens` (hover: absolute time, trigger, before → after; `collecting…`
between `started` and `completed`; a failed pass in the error tone) and
`GC: budget · runs at 80% · down to 60%` (or `after every turn above N%`
when continuous). They show even before any usage is reported — the policy
is known first. The legend's glyphs carry one-line tooltips, and a caption
says the icons describe what GC **may** reclaim from each layer, not what
it recently did — `never collected` read as "data was never collected".

Guards: `server/tests/test_the_gc_state_reaches_a_late_client_1190.py` (two
reversions; its call-site check parses `core.py` from beside the test
rather than via `inspect.getsource`, which the reversion meta-guard's
sandbox cannot see through), `BudgetPanel.test.tsx` / `gc.test.ts`, and an
e2e case in which a SECOND tab attaches after the pass and must be told —
verified to fail with the replay removed. A reconnect of the same tab could
not have proved it: it keeps what the tab already knew.

### A Popup That Floated Off the Page

Reported with a screenshot: the live tool-output popup of a running
`cli_based_tool` was cut off on the left edge of the transcript. The
component was correct (`absolute right-5 bottom-[104px]` inside a `relative`
`<main>`); the stylesheet was not. `.plate { position: relative }` in
`theme.css` was an **unlayered** rule, and in Tailwind v4 an unlayered rule
outranks every utility whatever its specificity. So each floating plate lost
its `absolute`. The popup sat in normal flow, and `right-5` then shifted it
20px past the left edge. The command-proposal list had the same defect: laid
out in flow, it grew the composer strip upward and shrank the transcript by
its own height every time a proposal appeared.

The rule now lives in `@layer components`. A plate is still positioned by
default (its corner marks need it), and a caller's utility can override
that. Two e2e cases measure the result rather than the styling: the popup's
box lies inside the transcript column, above the input and against its right
edge; and the composer strip's top does not move when proposals open. Both
fail against the unlayered rule. Measuring the input's own position would
not have caught the second case, because it is pinned to the bottom. The
mock gained a `live` turn that keeps a tool running with output until
`session.stop`, because the popup only exists in that window.

### An Exit That Never Asked

The TUI's `exit` is a question before it is an action: a session lives on
the daemon, so leaving it means one of three things — **detach** and keep
it for `session attach` later, **end** it (`session.delete`), or, with a
turn in flight, **cancel** the turn and detach — and the TUI asks which
(`[d/e/r]`, or `[c/d/e/r]` mid-turn) before doing anything. The web
client's `exit` command and status-bar Exit took the first reading
unconditionally. Safe, and the only reading the button offered: a session
someone wanted gone stayed loaded on the daemon until the orphan sweep or
a `session delete <id>` typed from memory.

The question is ported as a plate (`components/prompts/ExitPrompt.tsx`),
drawn like the permission plate so the two read as one kind of prompt,
with the TUI's option sets and letters (`app/exitChoice.ts`). The store
holds the open question (`exitChoice`); the composer forwards a typed key
to it **before** a pending permission prompt, as the TUI's pending exit
confirmation takes the line first; Tab cycles the buttons, Enter answers
the focused one, Escape and any unlisted key are Return.

Two decisions the TUI never has to make, because it is a process and
"end" is also "quit":

| Answer | Where it lands |
|---|---|
| Detach, Cancel task and exit | disconnect, the connect screen — the exit command as it was |
| End session, workspace mode | **the workspace list**, connection kept |
| End session, single-workspace daemon | disconnect, the connect screen |

`SessionManager.delete_session` removes the session's memory and disk
record and never touches the directory it ran in, so after End the
workspace is exactly where the person left it, and the list is where they
pick it — or another — again. End also **waits for the daemon's answer**
before leaving: `session.delete` is confirmed by a `system.message`
(`Session '<id>' deleted.` / `not found.`, and `Session deleted: <name>`
to attached clients), and leaving on the send alone would report a
deletion nobody confirmed. A daemon that says nothing gets a bounded grace.

The e2e mock gained `session.delete` in the daemon's shape and a `hang`
turn that runs until `session.stop`: the suite runs with `MOCK_SPEED=0`,
so a turn "long enough to press Exit during" cannot be a sleep, and the
first draft's timed turn had already ended by the time the button was
clicked. The pre-existing `Disconnect` on the workspace list is also why
the End-session test asserts the connect button by `exact` name — a
substring match counts it.

### State That Outlived the Connection It Belonged To

Three reports from one deployed session on a tablet, and one cause behind
all of them: **the daemon keeps per-connection state, the browser keeps its
own copy, and nothing reconciled the two.**

A reconnect is a NEW client on the daemon. The disconnect path calls
`remove_client` on the workspace manager and on the event-sink adapter,
dropping this connection's `workspace.select`, and detaches the client from
its session. The store keeps both, so the screen went on naming a workspace
and a session the live connection did not have. What the person saw:

| Reported | Mechanism |
|---|---|
| a staged file refused with `No workspace selected for client client_6 (workspace_id='')` while the picker's own header named the workspace | `_resolve_staging_workspace` asks this connection's selection; the store's answer came from the previous one |
| a session that came up on `RunnerBootstrapFailed: envelope.model_name is empty` | `session.new` with no workspace resolves no `.env`, so the envelope carries no model. The same loss, one verb over, with no error naming it |
| the chat pane carrying a previous attempt's errors above a healthy session's lines | the pane was never cleared by the verb that binds it to a new session |

**The client re-asserts what it believes, in the order the daemon needs it.**
`reassertAfterReconnect` (`sdk/connection.ts`) fires on the
reconnecting→connected transition: the workspace FIRST, because
`session.attach` compares the session's workspace against the client's and
opens a mismatch prompt when they differ, and a client that has just
reconnected has none; then the session, with the SDK verb alone. Not the
app's `attachSession`, which resets the pane and re-requests history: that is
right when switching sessions and wrong here, where the transcript on screen
is already this session's and a bare attach replays nothing. The SDK's own
opt-in `autoReattachSessionId` is left off for the other half of that reason
— it sends the attach without the workspace that has to precede it.

A batch already on the wire when the socket dropped is answered by the new
connection, so `staging.ts` retries a `workspace_not_found` refusal **once**,
after re-asserting. Once, so a daemon that genuinely has no workspace still
reports it.

**Leaving the daemon drops the session state, because that is what it
describes.** `sessionId` means "this client is attached to that session", and
a client that closed its socket is attached to nothing. Keeping it left the
screen rendering a dead session's transcript with a live composer — and
`SessionScreen` shows its picker only when no session is held, so after a
Detach and a reconnect there was no way back to the picker at all: the next
connection opened straight into the transcript of the session it had just
left. `disconnect()` is the one place that can say it, so it says it there.

**And a new session starts on an empty pane.** `attachSession` has always
reset; creating was the one verb that bound the screen to a different session
and cleared nothing, so a failed attempt's errors were still at the top when
the next attempt's "Session created" line arrived. Queued attachments survive
both resets by construction — `uploads` lives outside `emptySessionState`,
which is what lets the picker stage files into the session it is about to
open.

**The mock refused staging on a rule the daemon does not have**: "no session
and not workspace mode", so in workspace mode it never refused, whatever the
connection had selected. That is why 36 e2e tests could not see any of this.
It now resolves the workspace the way `_resolve_staging_workspace` does —
this connection's selection, else one provisioned for it at `session.new` —
keeps its sessions across connections so a reconnecting client can attach to
the one it had, and drops the socket on `mock-drop` with `terminate()` (a
close frame carrying 1006 is one the library refuses to send, and a reserved
code is what a dropped socket reports). Verified non-vacuous: with the
re-assert neutralised the reconnect test fails on the daemon's refusal, and
with the create reset neutralised two of the three unit cases fail.

### A Spinner Keyed on a Word Nobody Says

Reported as *"the thinking indicator does not follow what happens in
reality, sometimes it does not even show"*. Three defects, one in each
layer, and the middle one is why the suite could not see the first.

**The client tested for a status vocabulary the daemon does not have.**
`AgentStatusChangedEvent.status` is `active` | `idle` | `done` | `error`
(the docstring) plus `cancelled` from the subagent plugin — five words,
emitted from nine sites. The web store computed its `processing` flag as
`status === "processing" || status === "running"`. Nothing upstream
produces either word. The only producer was **the client itself**, which
dispatched a fabricated `AgentStatusChangedEvent` at its own store before
awaiting `sendMessage`, and because the assignment is unconditional the
daemon's real `active` — emitted immediately before `_start_model_thread`
— evaluated false and switched the indicator back **off**. So it lived for
exactly one round trip per turn, and a turn the composer did not start (an
attach to a running session, a subagent, `session.wake`, an injected
prompt, a reconnect mid-turn) never lit it at all.

Two more consumers read the same flag, so both were wrong in the same
direction and neither announced it: **Ctrl+C was dead** whenever the flag
was wrongly false (`useKeyboardShortcuts` gates `stop()` on it), and the
exit question offered the wrong option set (`exitChoice` reads it to
decide whether to lead with *Cancel task and exit*). `AgentTabs` keyed its
glyph map on `processing` / `running` / `awaiting_permission` / `finished`
— four invented words — so every agent tab fell through to the idle glyph
whatever the agent was doing.

**And the mock spoke the client's vocabulary.** `mock/daemon.ts` emitted
`status: "processing"`, so 38 e2e tests certified an indicator that could
not work against a real daemon. Same shape as the clarification, permission
and budget-panel defects before it, which is why the mock is now the
daemon's word and an e2e case asserts a tab's own title.

**The flag is replaced by a derived phase, not corrected.** A second copy
of "is it busy", written from four places and read from four more, is what
fell out of step; `store/phase.ts` computes the phase from facts the store
already holds for other reasons:

| Phase | Derived from | Priority |
|---|---|---|
| `waiting` | a pending permission / clarification / reference for that agent | a person is blocked — outranks everything |
| `tool` | an open `tool.call_start` with no end, oldest first, plus the batch count | names what is running |
| `thinking` | the daemon's `active` | |
| `sending` | `busySince` stamped by the composer, no daemon word yet | the one optimistic piece |
| `idle` | — | |

`busySince` is the busy predicate rather than the status string: it is
stamped by the two events that START a turn and dropped by every event
that ENDS one (a non-active status, `turn.completed`, `agent.completed`,
`agent.error`). That is what makes it **self-healing** where a flag was
not — a tool call whose `call_end` was lost to a reconnect leaves a
`running` block in the transcript and cannot pin the indicator, because
the `tool` phase is reachable only inside a turn the daemon still owns.
The client now invents no status of its own: `Agent.status` is the
daemon's word verbatim, the locally-written `awaiting_permission` /
`processing` / `finished` are gone, and the indicator carries the phase,
the tool's name and an elapsed clock (`Running cli_based_tool — 2:14` is
the difference between a session that is working and one that is wedged).

**The third defect is the daemon's, and it is the other half of "does not
even show" (#1139).** `AgentState.status` has exactly one reader — the
attach replay, which tells a client arriving mid-session what each agent
is doing — and the MAIN agent's six status emits called `emit()`
directly, never touching the field, so it stayed at the `"idle"` it is
constructed with for the life of the session. The subagent path had it
right (`on_agent_status_changed` stamps before emitting), which is why
only the main agent went dark. A second browser tab, a reconnect, a
re-attach after a detach: told `idle` about an agent mid-turn, and told it
until the turn ended. `JaatoServer.emit_agent_status` is the one door —
record, then emit — and an AST guard fails the build on an
`AgentStatusChangedEvent` constructed around it, because the defect is a
call site nobody thought of and a behavioural test can only exercise one
somebody did. The replay is excluded by construction: it passes
`status=agent.status`, a read of the record rather than a new claim about
it.

Verified non-vacuous on both sides: restoring the old vocabulary in
`phase.ts` fails 6 of the 11 new store cases, and the two declared
reversions in `test_an_agent_status_the_record_did_not_keep.py` fail their
named tests.

### A Key the Web Files Panel Did Not Have

The TUI's workspace panel (Ctrl+W) binds two keys to the entry under the
cursor: `h` **hides** it — a per-session, client-side set, with a
show-hidden toggle that brings the set back dimmed with an `H` marker so an
entry can be unhidden — and `i` **toggles its line in the workspace's
`.gitignore`**, which the TUI does by writing the file itself, because it
runs on the host. The web Files panel had neither, and could not have had
the second: a browser client has no file to write.

Hide is client state and is reproduced as such (`workspaceHidden`, the same
entry ids — a directory carries its trailing `/` and hides its subtree). The
`.gitignore` half becomes a daemon verb, **`workspace.ignore <path>`**
(protocol **1.12**), answered by one `WorkspaceIgnoreResultEvent` whatever
happened — a panel has to render *something* for the press. Three
properties:

| Property | Why |
|---|---|
| **one text transform, in `jaato_sdk.gitignore_toggle`** | the TUI's key and the daemon's verb both call it, so one press means one edit whichever client made it: exact-match toggle of ONE line, a glob already covering the path neither matched nor touched |
| **the SESSION's workspace, then the client's declared one** | the session's tree is what `WorkspaceMonitor` watches — and it reloads its parser on this very write, so the pattern binds every later file event. Entries already shown are **not** pruned; that is what hide is for |
| **the path is a pattern, not a path the daemon resolves** | so #742's relative-path rule does not apply; what is refused is anything that is not a workspace entry — empty, a line break, absolute (the panel's sandbox-monitored entries lie outside the tree `.gitignore` covers, the TUI's own no-op), or a leading `#` / `!`, which git would read as a comment or a negation and the toggle would then report a state the file does not have |

A missing VERB again (the 1.7 rule): an older daemon ignores the command and
"added to .gitignore" would describe a file nobody changed, so both SDKs
refuse below `MIN_WORKSPACE_IGNORE_PROTOCOL` (`toggle_workspace_ignore` /
`toggleWorkspaceIgnore`). The result event is client-initiated, the 1.10
shape, so an old client never receives it unprompted.

**And the panel read the wrong key.** `WorkspaceFilesChangedEvent.changes`
is `[{path, status}]`; the web store read `change`, so on a real daemon every
entry rendered `~` and a deleted file was never removed — while the mock sent
`change` and the e2e suite was green. The same shape as the clarification,
permission and budget-panel defects before it: the mock spoke the client's
vocabulary, not the daemon's. The mock now sends `status`.

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

#### The audit tier cannot import `ctypes`, and so cannot import numpy (#1011)

The row above says `ctypes.dlopen` of anything **outside the interpreter
installation** is refused. `dlopen(None)` — "this process's own symbols" — is
refused too, and that one is paid by code that never mentions `ctypes`:
CPython's `ctypes/__init__.py` runs `pythonapi = PyDLL(None)` at module import,
so **`import ctypes` raises the event by itself**. `numpy/_core/_internal.py`
does `import ctypes` guarded by `except ImportError`, and
`NotebookContainmentError` is a `PermissionError`, so the guard does not catch
it and the whole numpy import dies. With it go pandas, scipy, scikit-learn,
matplotlib, shapely-via-geopandas, osmnx and torch. **On the audit tier a cell
cannot run the scientific stack at all.**

The compiled extension modules were never the problem — they load through the
import machinery (`open` events) and site-packages is a read root via
`sys.path`. Measured: with `dlopen(None)` permitted and every other rule
intact, `import numpy` succeeds and `(numpy.arange(10)**2).sum()` is `285`.
Only the stdlib's own initialization line fails.

**This is not a sloppy check and relaxing it is not the fix.** `dlopen(None)`
returns a handle to the whole process symbol table, so the object the stdlib
builds *is* the escape: `ctypes.pythonapi.system(b"...")` resolves **and calls**
libc `system(3)` — verified, it ran a shell command. The only event that path
raises is `ctypes.dlsym`, which this hook does not audit, so a frame-gated
exemption permitting just `ctypes/__init__.py`'s own `PyDLL(None)` would hand a
cell arbitrary libc with nothing left to stop it. Auditing `dlsym` as well would
not close it either: once the module is imported, `memmove` over a
`from_buffer` view and `string_at` at a raw address give in-process memory
read/write with no library load at all — the `_POLICY`-rebinding limit the
module docstring already names. Under the audit tier you genuinely cannot have
both `import ctypes` and this boundary.

**The AppArmor tier has no such restriction**, because `establish_containment`
installs no hook at all when `/proc/self/attr/current` reports an enforced
profile — there, `import ctypes` and numpy work normally. Neither does a
**subprocess**: a spawned child is bounded by the OS and nothing else, so the
same import runs through `cli` or `!python`, which is also why `!pip install X`
works. Those are the remedies, and they are all *other surfaces* — which is the
whole reason the next section exists.

#### Which boundary is in force, said before the first cell (#1012)

The tier materially changes what a cell can do and the model was told none of
it: it discovered the tier by hitting a refusal mid-task and reverse-engineering
the boundary from one error string. A live session hit the `ctypes` refusal,
read its old wording ("loading native code would bypass the notebook's
filesystem boundary") as *the sandbox forbids native code*, told the user so
three times with increasing confidence, and steered them to an external API. It
had `cli` throughout, where one `python -c "import numpy, osmnx"` would have
falsified the theory in a single call. It never occurred to it that the two
surfaces have different containment.

Both shapes the issue offers are implemented, because neither covers the other:

| Surface | Carries | Why this one too |
|---------|---------|------------------|
| the plugin's **instruction contribution** | the active tier + its two consequences | the standing fact, in the system prompt, read BEFORE the first cell rather than after a refusal |
| the **first `notebook_execute` result** of each kernel | the tier the KERNEL reported on its READY frame | a kernel that respawned under a different posture — or whose interpreter could install no hook — contradicts the prompt visibly, in the result the model is already reading |

Both render through `kernel_sandbox.boundary_notice`, the **one** place a
tier's consequences are written down, so the standing fact and the per-kernel
one cannot disagree. Nothing derives a second answer: the tier comes from
`establish_containment` in the kernel and travels on the READY frame (where the
kind was already sent and was being discarded), and the pre-spawn expectation is
`SubprocessKernelBackend.boundary_kind()` — the ladder `execution_boundary` used
to write out inline, now named once and worded from there. A backend whose
containment is not one of the four tiers answers `None`, which renders as **no
claim** rather than a plausible wrong one.

Two things the text gets right, both of them the distinction the failing session
could not make:

- **refused-by-containment is not not-installed.** The audit notice says so in
  those words, and adds that other compiled extension modules import normally.
- **a subprocess is not audited.** The same import succeeds through `cli`, and
  `!pip install X` works — the single fact that would have ended the incident.

It is paid on every request (it lands in the prompt-cache prefix), so only the
**active** tier's lines are rendered, never a table of four, and it stops at two
consequences. The answer is stable within a session — each backend's
`boundary_kind` derives from AppArmor state, the operator's opt-out and whether
a workspace resolved, none of which changes under a running daemon — so the
prefix is stable; the respawn case is exactly what the per-result announcement
covers, and it is spent **once per kernel**, re-armed by `_spawn`.

**A refusal now names its tier.** Every message the hook raises is prefixed
`notebook containment (audit tier)`, and the `ctypes` one names the culprit,
denies the general claim, and lists the remedies. The hook is installed on the
audit tier and on **no other**, so that prefix is also an operator signal: seeing
it on a deployment that believes AppArmor is enforced is evidence of a silent
degrade, where previously the only trace was a runner log line nobody reads.
(`JAATO_REQUIRE_APPARMOR=1` remains the way to make that a refusal to start;
this only stops the evidence being invisible.)

### A Bootstrap That Failed, Discovered at an Unrelated Verb (#1033)

`session.new` began failing intermittently on a live daemon with

```
RunnerCallError: session.get_context_usage failed:
ToolError: session not bootstrapped on this runner
```

`session.get_context_usage` is a read-only toolbar RPC. It is not what
failed; it is what happened to ask first.

**`dispatch_bootstrap_envelope` does not propagate a bootstrap failure** —
deliberately: it emits `SessionTerminatedEvent` so cascade observers see the
death, and it calls `mark_runner_ready()` so a warm pool slot cannot strand a
client-tool push on a readiness timeout. Both are right and both still happen.
What did not happen is anyone being **told**, so session creation carried
straight on into `jaato_server.server.initialize()` — which asks the runner for its context
usage with no guard, two dozen lines below the one call in that function that
IS guarded, whose comment names this exact contingency (*"the runner-side
handler may return `stage="no_session"` if `session.bootstrap` RPC hasn't
completed"*). Every bootstrap failure class — an AppArmor confinement
mismatch, a provider connect failure, a secret that would not resolve, the
30 s bootstrap deadline — therefore reached the client as a toolbar read,
with the real cause in a daemon-side WARNING nobody was reading.

**It is not a race, and the lane split is not involved.** `session.bootstrap`
runs SYNCHRONOUSLY on the runner's reader thread, so no frame can even be
DECODED while it runs; and the daemon's own order is strictly sequential —
spawn, dispatch bootstrap, then initialize, all in one function. A control-lane
call cannot be served mid-bootstrap on this path. `no_host` was the honest
answer to a question that should not have been asked.

| Half | Where |
|------|-------|
| the outcome is recorded where it is known | `dispatch_bootstrap_envelope` → `JaatoServer.note_runner_bootstrap_outcome` (a summary on every failure path, `None` on success, so the field means "the outcome of THIS bootstrap" rather than "one failed once") |
| it is consulted where session creation is decided | `session_manager.initialize_or_refuse` — a free function, because it reaches nothing on the manager: it is a precondition of the SERVER |

Three properties:

- **`_require_ready_session` is untouched.** Making it wait, retry or answer
  optimistically would convert a visible refusal into a session reporting
  usage nobody measured. The predicate is right; the dispatch was wrong.
- **The refusal names the bootstrap failure**, as a non-recoverable
  `ErrorEvent(error_type="RunnerBootstrapFailed")` through the in-init sink —
  the shape `core.py`'s own in-init refusals use, followed by #882's
  correlated `session.new` answer.
- **`mark_runner_ready()` is still called on failure, and no longer claims
  what it used to.** Readiness and success were one question in that
  docstring ("the daemon-side JaatoSession remains authoritative"), naming a
  §7c rollout window that has since closed: `initialize()` itself now reads
  the runner's session. The log line moves WARNING → ERROR for the same
  reason — it is the one place the real cause is recorded.

**A bootstrap failure is a session failure either way; what changed is that it
says so.** A server that dispatched no bootstrap at all (the embedded path,
standalone WS) records nothing and initializes exactly as before.

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
`jaato_server/server/runner/tests/test_cancel_before_worker_registers_988.py` reproduces
the loaded case **with no clock at all**: one work-lane worker, held by a
gated call, so the cancelled call is provably still queued; wire ordering is
established by a control-lane probe rather than by polling. It fails
`unknown=1, tripped=0` against the old registration site, deterministically.

### A Failure While Reporting a Failure, Discarded (#1077)

The daemon's model thread wound its turn down inside a `finally` holding
**four `return` statements**. A `return` in a `finally` discards whatever
exception is in flight, so the wind-down could complete having thrown away
the thing that explained the failure. Python 3.14 makes the shape a
`SyntaxWarning` (PEP 765), which is how it was found — four warnings on
import.

**Most of what PEP 765 warns about was already handled here**, and saying
so is what locates the real exposure: the two `except` clauses below catch
`Exception` and `KeyboardInterrupt`, and the thread target's return value
is read by nobody. Two cases genuinely lost information:

| | |
|---|---|
| a `BaseException` that is neither of the two caught | `SystemExit`, `GeneratorExit`, an injected `CancelledError` |
| **an exception raised INSIDE either `except` handler** | a failure while reporting a failure — in the daemon's model thread |

It is narrower still than that table: the swallow only happens when the
wind-down actually *reaches* one of its four exits, so both cases
additionally need a terminal error, a stashed continuation, a drained user
send, or a pending nudge. A wind-down that fell off the end always
propagated.

**The fix moves the body, not the logic.** The 355-line wind-down is lifted
**verbatim** into a nested `_finish_turn()` declared before the `try`; the
`finally` is one call. Its four exits are now returns from `_finish_turn`,
which is exactly what they meant — the `try` is the last statement in
`model_thread`, so falling off the end and returning were already the same
thing. `try` body and both handlers are byte-identical. A nested closure
rather than a method because the body reads six enclosing names, and
threading them through would have meant editing it.

**`terminal_error` is read through the closure deliberately.** The handlers
assign it, a cell resolves at call time, and it is bound to `None` before
the `try` — so the wind-down cannot be handed a stale value, and the
handler-raises case leaves it `None` (the two cannot co-occur).

**Stated cost:** on the paths that previously swallowed, an in-flight
exception now escapes the thread target and is reported by
`threading.excepthook`. That is the intent, and it is new output on those
paths.

Complexity: `model_thread` 40 → **15**, so its ratchet entry is *removed*
rather than lowered and the function is held to the ceiling like any
un-baselined one; `_finish_turn` enters at 26, irreducible here by
construction — rewriting 300 lines of wind-down in the same change would
have made the one behavioural difference unreviewable.

Guard: `jaato_server/server/tests/test_no_return_in_finally_1077.py`. Its AST scan needs
two exclusions to be satisfiable, and both are pinned by a discrimination
test: a `return` inside a function *declared in* the `finally` is the shape
of the fix, and a `break` bound to a loop inside the `finally` transfers
control within it (PEP 765 does not warn about that either). An AST sweep
found these four were the only such sites in the tree, and none after.

### A Turn That Ended the Session, Handed Back as a Turn (#1007)

#988 and #856 are a call that never returns. This is the other half of the
same family: a call that **returns, on time, saying nothing true**.

`Session.ask` settled on first-of `{TURN_COMPLETED, SESSION_TERMINATED}`. The
daemon emits both from one thread, and the turn event comes **first** — 2-3 ms
first, measured on the issue's credential-free `echo` repro. So a session its
`budget_control` ceiling had just ended answered `''`, indistinguishable from a
model that said nothing, and `_raise_if_needed` raised only for
`reason == "error"`. **This half needs no cascade at all.**

On a **cascade-stamped** session it then hangs. The terminal also unloads the
session (`_apply_default_cascade_policy` detaches every IPC client), so the
next send is answered at once with `ErrorEvent("Session not found: …")` —
which `ask` did not subscribe to. `timeout=` converts the wait into
`TurnTimeout`; without one it waits forever. A real run blocked **12+ minutes**
in `pipe_read` and left no line in the daemon log, because that reply was not
logged either.

**`complete` already did the right thing, and that is the whole lever.** It
settles on the daemon's confirmation (#767) and on the terminal, so a
completion-gated stage over its ceiling returns cleanly. The facade had **two**
settle rules; one of them was wrong, and nothing kept them in agreement.

**One rule now, in `_TurnWatch`, parameterised by SPAN** — which is the entire
difference between what a turn verb and a session verb want from the same
events:

| | `span="turn"` (`ask` / `stream`) | `span="session"` (`complete`) |
|---|---|---|
| `TURN_COMPLETED` (latched agent) | proposes a terminus | proposes a terminus |
| that agent's next `AGENT_STATUS_CHANGED` | **any** status confirms — the turn is over either way | `active` **withdraws** (a nudge, a drained send); `idle`/`done` confirms |
| `SESSION_TERMINATED` | unconditional; the only thing that sets `session_ended` | same |
| `ErrorEvent(error_type="SessionError")` | the session is not there — fail, do not wait | same |
| never latched (no opening `active`) | first turn settles, as pre-#767 | same |
| latched, never confirmed | settle the proposal after `TERMINUS_GRACE` | after `SETTLE_GRACE` |

**Two graces, because they bound different questions.** `SETTLE_GRACE` (10 s)
asks *is this agent going to take another turn* — about the AGENT, which may
legitimately pause. `TERMINUS_GRACE` (1 s) asks *did the daemon emit a terminal
in the same breath as this turn event* — about ONE emit sequence on ONE thread,
answered in milliseconds or not at all. **Its expiry costs information, never
correctness**: the call settles on the turn and returns the turn's text, which
is exactly the pre-#1007 behaviour. Measured after the fix, the confirming
status arrives 4-9 ms after the turn event on every plain turn and the grace
never fires; `ask` costs ~7 ms more per turn than it did.

**What reaches the caller.**

| | turn verbs (`ask` / `stream`) | `complete` |
|---|---|---|
| the turn was **cut short** by the session ending | raises **`SessionEnded`** (`reason`, `details`) | returns the payload; records it |
| the session ended **cleanly** (`natural`, `client_request`, `stopped`) | returns the turn; records it | same |
| the daemon says the session is gone | raises `SessionEnded(reason="not_found")` | same — a request that reached nothing is not a terminus anybody drove to |
| `reason == "error"` | `AgentError`, as always — the richer type wins | same |
| either way | `Session.terminus` (`reason`, `details`, `session_ended`) | same |

A turn verb promises one turn **with the session still there for the next
one**; a terminal that cut the turn short breaks that promise and every later
call on the handle is dead, so it raises. `CLEAN_TERMINAL_REASONS` is the
exception and it has a consumer in this tree: a scaffolded `client` drives a
completion-gated profile with `ask`, and its one successful turn ends the
session with `natural` — raising there would fail every such run
(`jaato_server/shared/scaffold/tests/test_client_template_completion_wait.py`, which is
what caught it). It is an **allow-list**, so a reason that does not exist yet
raises rather than passing quietly: #1007 *is* a new reason arriving and going
unconsidered, and a deny-list would let the next one do it again. The ending
reaches the caller on `terminus` either way.

`complete` promises to drive the session to its terminus,
so a terminal is its success condition — but "no payload" was a caller's whole
account of a budget stop, which is why `terminus` exists. The partial text is
not lost: chunks reach `on_media` and any `s.client` listener as they arrive,
and `stream` yields everything it had before raising.

**`ERROR` is narrow on purpose.** Only `error_type == "SessionError"` — the
daemon's four "this request could not reach a session" replies — settles a
call; an ordinary mid-turn error still belongs to the turn. `recoverable` is
**not** the discriminator: the "Session not found" reply leaves it at its
`True` default. An event naming a different session is ignored; one naming
none is accepted, because a daemon predating the stamp sends it unattributed
and degrading to "wait forever" is the defect.

**Two daemon-side consequences, both of them why the run left no trace.**
`handle_request` now logs the refusal at WARNING, naming the request type, and
stamps the target `session_id` on the `ErrorEvent` — `_emit_to_client` stamps
from `_client_to_session`, and on this exact path the client has already been
detached, so the stamp found nothing. And
`_apply_default_cascade_policy`'s docstring said "all four current reasons
(`natural`, `error`, `stopped`, `client_request`)". There are **six**;
`budget_exhausted` is one of the two it did not name and is the one this policy
turns into a hang. The policy never reads the reason at all, which is what the
docstring says now.

**A claim that used to stand in `complete`'s docstring and does not.** "The
daemon closes every turn of the main agent with exactly one
`AGENT_STATUS_CHANGED`" is false on a cascade session: the status emitted after
the unload reaches nobody. `complete` survives only because the terminal comes
first, so **nothing may be built on the status event alone arriving** — which
is why `SESSION_TERMINATED` settles unconditionally in both spans.

**Side effect of one rule: `ask` inherits #767's agent latch.** A subagent's
`TURN_COMPLETED` no longer settles the parent's `ask`, which it did before.

Tests: `jaato-sdk/jaato_sdk/tests/test_a_turn_that_ended_the_session_says_so.py`
(deterministic — `ScriptedClient` delivers one event per loop tick, so "did it
return before the terminal arrived?" is a counter, not a clock; 17 of its 24
fail on the parent commit) and
`jaato-server/jaato_server/shared/tests/test_one_settle_rule_1007.py` (an AST guard that no
verb decides a terminus outside `_TurnWatch`, and that the watch still wires
all four settle events — dropping `ERROR` restores the hang exactly).

### A Request the Daemon Wrote and the Runner Does Not Have (#856)

#988 is a cancel that lost a race inside the runner. This is the turn
itself. A session's second turn was registered in the daemon's
`_in_flight` and never happened; sixty-eight minutes later the client was
still blocked in `session.ask()` and py-spy showed **every thread in all
three processes idle** — no lock held, no read outstanding, the pipe
healthy, the runner alive with an empty work queue. The daemon's entire
record was four lines:

```
[RPC_DIAG] daemon _in_flight SET id=30 client=129050257488880
[RPC_DIAG] daemon _in_flight SET id=31 ...
```

which are emitted ~25 lines BEFORE the write, so they say what the daemon
INTENDED and nothing about what became of it.

**Three failure modes, and one had no name and no bound:**

| what happened | how it surfaces | session survives? |
|---|---|---|
| the runner died (#851) | `RunnerCallError` — the read loop sees EOF and fails every future | no — terminal |
| the runner is slow | `RunnerAnswerTimeout` | yes |
| **the dispatch is lost** | **`RunnerDispatchLost`** | yes |

`call()`'s `return await fut` had no deadline, and the two calls that
matter most default to `timeout=None` (`session.send_message`,
`call_threadsafe`) because a turn legitimately runs for minutes. So a
dead runner reached a person as an error and a lost request reached them
as the agent apparently thinking.

**The bound is a reconciliation, not a wall clock.** A wall-clock cap on
an RPC would kill healthy long turns, which is worse than the hang. What
is bounded is how long the daemon will believe, *unchecked*, that a
request it wrote is being worked on: after a window with no frame bearing
that id, it ASKS the runner — over the **control lane**, so the answer
arrives while the work lane is busy with the very turn in question.

| verdict | the runner says | action | may have run? |
|---|---|---|---|
| `running` | the id is in `active_call_ids` | another full window, indefinitely | — |
| `finished` | it left `active_call_ids`, still in the window | fail — the RESPONSE was lost | **yes** |
| `never_received` | not in a window that has evicted nothing | fail — **the bug** | no |
| `indeterminate` | the window has rolled past the id | fail | **yes** |
| `unreachable` | the probe itself did not come back | fail | **yes** |

`session.health_check` carries the transport view (`active_call_ids`,
`known_request_ids`, `highest_request_id`, `seen_window_capacity`),
reported whether or not a session host exists because it describes the
CHANNEL, not the session.

**The last column is the whole point, and it is a TYPE.** `finished` and
`never_received` have opposite consequences: one says the work did not
happen and retrying is safe; the other says it RAN — history advanced,
tools executed, files were written, a provider was billed — and only the
daemon's view of the result was lost, so retrying re-executes it. A
single exception hid that, which is the shape
`jaato_sdk.client.errors.SessionNotConfirmed` already names: *"the one
failure where the correct action depends on something the caller cannot
see, which is why it is a distinct type rather than a detail in a
message"*.

| exception | verdict(s) | `may_have_run` |
|---|---|---|
| `RunnerDispatchNotReceived` | `never_received` | `False` — retry is safe |
| `RunnerResultLost` | `finished` | `True` — retry re-executes |
| `RunnerDispatchUnknown` | `indeterminate`, `unreachable` | `True` — nobody could say |

All three subclass `RunnerDispatchLost`, so `except RunnerDispatchLost`
still catches every one. `may_have_run` is the axis
`SessionCreateFailed.may_exist` established, `verdict` is the
machine-readable mechanism token its `cause` established, and both **fail
safe**: the base defaults to `may_have_run = True`, so a verdict added
later cannot inherit "safe to retry" by forgetting to decide.

It reaches a client on **`ErrorEvent.details`** — the field documented as
"what a driver branches on", with `error` left as the human sentence.
Deliberately **not** `recoverable`: in this tree that flag means *this
session can continue* (every `recoverable=False` site is a config or
provider-connect failure that ends initialisation), and sparing a
`RunnerRPCTimeout` here is precisely what keeps the session alive.
Encoding retry-safety there would assert something false about session
viability to every consumer that reads it the documented way.

**The window's FULLNESS, not its size, is the correctness bound.**
`known_request_ids` is a `deque(maxlen=...)`, so it has evicted something
only once it is FULL; until then "not in the window" proves "never
registered". A full window whose floor sits above the id is answered
`indeterminate`, never `never_received` — that one misclassification is
exactly the unsafe direction, and raising 256 to a larger number would
only make it rarer, not impossible. The runner reports its own
`seen_window_capacity` rather than the daemon assuming the constant,
because a number asserted on one side of a wire about the other is how
that distinction goes stale unnoticed.

Two counters, because the two need different operator responses:
`dispatch_lost_count()` is every lost dispatch,
`dispatch_lost_may_have_run_count()` the subset whose side effects are
already in the world. The first is logged at WARNING, the second at
ERROR.

**The predicate is a window, not a high-water mark.** The probe is itself
a request, registered on the runner's reader thread *before* its own
handler runs — so `highest_request_id` at answer time is always the
probe's own id, and comparing against it would report every id as
received, including the ones that never arrived: the one answer that
makes the whole mechanism worthless. `SEEN_REQUEST_ID_MEMORY` (256) only
has to outlive the ack window. This is the daemon-side half of #988's
`unknown` cancel counter — both count a peer disagreeing about what is in
flight — and `dispatch_lost_count()` is its counter.

**`RunnerDispatchLost` subclasses `RunnerRPCTimeout`**, which decides the
blast radius: `core.py`'s model thread terminates the session for
anything it catches EXCEPT a `RunnerRPCTimeout`, so the TURN fails and
the SESSION lives. That is also what makes it distinguishable from #851
in the client-visible error, since `ErrorEvent.error_type` is
`type(exc).__name__`. The message names the id, the method and the
session, because the incident's whole record named none of them.

**Two exemptions, each for a runner that cannot answer:**

- `session.health_check` — it IS the probe, and watching it would have
  the watchdog answer its own question, recursively.
- any window in which `session.bootstrap` is outstanding. It runs on the
  runner's MAIN thread (synchronously, so `aa_change_profile` confines
  the thread that later spawns the workers), so the reader thread is
  inside it and NEITHER lane answers; a probe would time out and read as
  a lost dispatch. The 120s default also sits clear above bootstrap's own
  30s deadline, so two independent things stop it.

**And the dispatch is logged.** `_dispatched[id] = method` is set AFTER
the frame reaches the wire, beside a `[RPC_DIAG] daemon DISPATCHED
id=N method=...` line. An `_in_flight` entry with no `_dispatched` entry
is "registered, never written" — the state the four original log lines
could not be told apart from a healthy in-flight turn.

Knob: `JAATO_RUNNER_ACK_TIMEOUT` (host-scoped — it bounds the channel,
which a pool slot shares across several sessions in turn). `0` disables,
following the provider-deadline convention; a negative or unparseable
value falls back to the default rather than disabling, because
"unbounded" is the bug this exists to fix.

**What the triage established, and what it did not.** The request-write
path was ruled OUT as a silent drop: `_write_frame_json` awaits
`writer.drain()` and `call()` catches only `FrameTooLargeError`, which it
re-raises after deregistering, so any other write failure reaches the
caller. There is no queue on that path to drop from. A runner-side
dispatch-table miss was ruled out too — an unknown method returns an
error RESPONSE (`unknown method: ...`), and `serve()` wraps its loop body
in no broad `except`, so an unexpected reader-thread exception closes the
channel and becomes #851 rather than this.

What WAS reachable, and demonstrated in
`test_the_response_side_drop_that_produced_this_state_is_closed`, is a
response-side drop. #920 refuses to write an oversized frame and
substitutes a small typed error for the call — guarded by `if not ok or
self._closed: return`, so the substitution happened for a SUCCESS
response and not for an ERROR one, on the reasoning that an error frame
that did not fit will not fit again. But the substitute is small by
construction; what did not fit is the original, whose `result` dict
carries megabytes of tool output beside the traceback. Measured: `ok=True`
answered the caller in 221 bytes, `ok=False` wrote **nothing at all** and
left `_closed` **False** — the channel open, the runner idle, the
daemon waiting forever. Not a claim about the reported incident, which
would need the runner's own log; a demonstration that the class is
reachable, and it is precisely what the `finished` verdict answers.

**That producer is closed (#999).** Both paths substitute now, and
`_emit_response`'s guard is `if self._closed` alone — the only half of it
that was ever true. The substitute also carries the original failure's
`type` and a bounded prefix of its `message`, because the runner held the
reason and telling the caller only "too large" threw it away; the
traceback is deliberately not carried, being usually what made the frame
oversized. `OVERSIZE_MESSAGE_CHARS` (2000, against a 10 MB cap) makes
"small by construction" an enforced property rather than the unchecked
claim the old guard rested on, and a second, quotation-free attempt is
written if even that is refused.

The deadline is unaffected and is still what matters here: it bounds the
CLASS rather than any one producer, and the `finished` verdict remains
the right answer whenever a response is lost — a closed channel, a
severed peer, or whatever drops one next. An oversized STREAM chunk is
still dropped with no substitute, now as a stated decision: a chunk is
display output rather than an answer, so the call still ends in a
response, and a per-chunk error would arrive interleaved with real output
as though the model had said it.

### Interactive Shell Sessions (`jaato_server/shared/plugins/interactive_shell/`)

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
workspace test moved to `jaato_server/shared/plugins/command_containment.py`
(`classify_command_paths`, `path_within_workspace`, `first_denied_path`); `cli`
keeps its result-shaping (an explicit `cli containment (workspace boundary):`
refusal naming `plugin_configs.cli.extra_paths` — until #1202 it mimicked "No
such file or directory", and a session read that as binaries vanishing) and
delegates the analysis. `cli` alone also asks for the executable position to be
classified `exec`: a program named by path (`/usr/bin/git`) is allowed iff its
directory is an entry of the subprocess PATH `_build_subprocess_env` builds —
the one PATH that both judges and runs — so it gets the verdict its bare name
gets; the same path as an argument (`cat /usr/bin/git`) is still data. Fail-closed is the caller's decision and
`first_denied_path` makes each caller state it — `on_parse_error="deny"` for a
command, `"allow"` for typed input.

**Unconfined is announced, not inherited.** When the runner installed no
AppArmor child-profile transition, the plugin logs **once per session at
WARNING** that the kernel boundary is absent — the posture `scrub_secret_env:
none` and `--ws-unsafe-no-auth` already take. `require_confinement: true`
refuses every spawn instead, the shape `notebook`'s in-process-exec gate has.
The default is `false` because a PTY child is a separate process, the same risk
class as a `cli` subprocess, which does not fail closed either.

**`require_confinement` means ENFORCE (#1014).** An installed child
transition is necessary and not sufficient: under `JAATO_APPARMOR_COMPLAIN`
the transition into `//child (complain)` succeeds and the kernel blocks
nothing, so the knob used to pass on a profile containing nothing. It now
also reads the **spawning thread's** own label
(`/proc/thread-self/attr/current` — `fork()` inherits the cred of the thread
that calls it, and per #1023 a worker can differ from the main thread) and
refuses anything that is not `(enforce)`, with wording distinct from the
no-transition refusal because the operator's next move is different. The
label is read only when the knob is set, so the default posture gains no
per-spawn `/proc` read.

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

### What a Model Finds When It Reaches for a Tool (#1273, #1274)

The `jaato-sdk` skill tells a model to run `jaato-doctor` and `jaato-scaffold`
"from the same Python environment as the daemon". In a live web session
`which jaato-doctor` found nothing. The daemon ran as
`<venv>/bin/python -m jaato_server` without its venv activated, so its console
scripts were on no `PATH` a subprocess inherits. The model could not name them
by path either: #1202 allows a program by path only when its directory is on
that same `PATH`.

Four kinds of tool, four sources, and the daemon's own venv is never one of
them:

| The model runs | Resolves from |
|---|---|
| `jaato-doctor`, `jaato-scaffold` | an allow-listed symlink directory in the session tmpdir (`jaato_tools_path.py`), APPENDED to the `PATH` |
| `python`, `pip`, `uv pip` | the workspace venv (`VIRTUAL_ENV` + its `bin` prepended), which daemon-managed workspaces now get by default |
| what `uv tool` / `pipx` / `pip --user` install | `<home>/.local/bin`, APPENDED wherever the workspace HOME is applied |
| `git`, `gh`, `node`, a host `uv` | the host `PATH`, i.e. provisioning; `extra_paths` for unusual locations |

**Why not `<venv>/bin`.** It would fix `jaato-doctor` and also expose the
daemon venv's `python` and `pip` wherever nothing earlier on the `PATH` shadows
them, so a model's `pip install` would modify the daemon's own environment, as
root on a root daemon. The allow-list is two names. `jaato-server` (which
starts and stops the daemon) and `jaato` (the TUI) are excluded.

**Why symlinks, and why in tmp.** AppArmor mediates exec on the RESOLVED
path. The targets are `{venv_path}/bin/*`, which the runner profile already
grants `ix`, while the tmpdir is `rw` without exec, so a wrapper script there
would be refused. The directory is rebuilt if the session tmpdir is reaped.
**Known limitation:** a symlink cannot reset the environment, so a tool started
this way sees the workspace venv's site-packages on `PYTHONPATH`, and a
conflicting package installed there could shadow one of jaato's dependencies.

**The managed venv (#1274) is the #1225 rule applied to `workspace_venv`.**
Every workspace the web client creates is profile-less (a bare `.env`), so no
`plugin_configs` channel reached it. Without a venv, `pip install` ran the
host's pip: as root against the system Python on a root daemon, or into PEP
668's refusal and then `--break-system-packages`. Now a workspace under the WS
server's `workspace_root` gets `.jaato/tool-venv` on `cli`, `interactive_shell`
and `notebook`:

| Property | Where |
|---|---|
| an explicit `plugin_configs.cli.workspace_venv` wins; `""` opts out | `effective_workspace_venv` |
| an explicit per-surface value is kept; an explicit `cli` value is NOT mirrored (its pre-#1274 meaning) | `inject_workspace_venv` |
| the envelope carries it | `build_session_envelope`, beside `inject_workspace_home` |
| the AppArmor rules follow it (venv `bin/*` `ix`, home `.local/bin/*` and `.local/share/**/bin/*` `ix`) | `resolve_plugin_apparmor_rules(..., managed_workspace_root=)` folds both defaults |
| kept out of the Files panel | `_WORKSPACE_HOME_IGNORE` in `workspace_monitor.py` |

A user's own checkout (IPC, user-CWD) is unchanged: no implicit venv.
`resolve_plugin_apparmor_rules` used to return `None` for any profile-less
session. On a managed workspace it no longer does, so those sessions also get
the rules of the plugins the runner loads either way.

Not verified here: no AppArmor kernel. The exec grants and the resolved-path
reasoning are exercised as rendered strings, not against an enforcing host.

### WebMCP Plugin (`jaato_server/shared/plugins/webmcp/`)

Invokes the tools a **web page** declares for agents via
[WebMCP](https://github.com/webmachinelearning/webmcp)
(`document.modelContext`) — so the model drives a web app through its own
declared operations instead of scraping and clicking. WebMCP is *not* an MCP
transport (no server, no JSON-RPC); it is an in-page JS API, so the plugin
drives a browser over `jaato_server/shared/cdp.py`.

**Not in the default plugin set** — it drives a browser. Enable it explicitly
in a profile's `plugins:` list.

| Tool | Purpose |
|------|---------|
| `webmcp_list_tools` | Harvest the open page's currently-declared tools (name, description, parsed `input_schema`, `origin`). Auto-approved (read-only). |
| `webmcp_call` | Invoke one page tool by name. **Not** auto-approved — it runs the page's own code and can post, delete, or buy on the user's behalf. |

**Why two tools and not one schema per page tool.** A page's toolset changes on
navigation and with app state, while the registry exposes schemas once at
configure time — so page tools are *discovered* through this pair rather than
registered. The security consequence is the larger half: arriving as a tool
**result** rather than as `ToolSchema` objects, page-authored names and
descriptions never enter the trusted schema block, so the existing
`TRAIT_UNTRUSTED_CONTENT` boundary covers them and neither
`TRAIT_UNTRUSTED_SCHEMA` nor `sanitize_untrusted_schema` is needed. Each entry
is labelled with the `origin` Chrome reports for it.

**The shipped browser API differs from the published explainer in six places**
(measured on Chrome for Testing 153): the API is on `document` not `navigator`;
there is no `unregisterTool` or `provideContext` (unregistration is via
`AbortSignal`); `inputSchema` and the call arguments and the result are all
**JSON strings**; and `executeTool` requires a live `RegisteredTool`, not a
name. Chrome also does **not validate arguments** — a missing `required`
property reaches the page as `undefined`. See
[the WebMCP assessment](docs/design/webmcp.md) §1.1 for the full table.

| Config key (`plugin_configs.webmcp`) | Env | Purpose |
|---|---|---|
| `page_url` | `JAATO_WEBMCP_PAGE_URL` | Page to drive; an already-open tab with this URL is preferred over creating one |
| `cdp_url` | `JAATO_WEBMCP_CDP_URL` | Attach to a running browser instead of launching (left running on shutdown) |
| `binary` | `JAATO_WEBMCP_BINARY` | Browser binary when launching |
| `user_data_dir`, `headless`, `extra_args`, `connect_timeout`, `call_timeout` | — | Launch and deadline knobs |

Requires Chrome/Edge 149+ (origin trial), or `chrome://flags/#enable-webmcp-testing`.

### Webhook Plugin (`jaato_server/shared/plugins/webhook/`)

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
    },
    "slack": {
      "path": "/webhook/slack",
      "secret_header": "X-Slack-Signature",
      "secret_algo": "hmac-sha256",
      "signature_scheme": "slack-v0",
      "timestamp_header": "X-Slack-Request-Timestamp",
      "max_age_seconds": 300
    }
  }
}
```

**Route auth: two modes, and they are not peers (#930).** `secret_algo` names
how the route's shared secret is checked:

| Mode | Header carries | Property |
|------|----------------|----------|
| `hmac-sha256` | an HMAC digest keyed by the secret | the secret never travels; a captured request cannot be replayed against another payload — but it **can** be replayed against the same one, see below |
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

#### A Signature That Never Expired (#713)

`secret_algo` says how the credential is CHECKED. It cannot say **what was
signed** — and a digest over the request body alone binds no time, so it
authenticates the same bytes forever. Whoever observes one delivery (a proxy
log, a mirrored port, a misrouted retry) replays it verbatim, indefinitely, and
it authenticates every time. Driven against the tree at `9abaffa5`: the original
ACCEPTED, the replay ACCEPTED, the replay a year later ACCEPTED.

Here that is not a duplicate row. The plugin exists to let external events drive
long-running agent sessions (`webhook_subscribe` → `webhook_poll` → the agent
acts), so a replayed webhook is a **re-triggered turn** — tool calls, ledger
spend, whatever the persona authorises. The hardening already present (TLS/mTLS,
CIDR allowlists, token buckets) constrains *who may connect*; none of it
constrains replaying a request that was legitimately signed. And #930's `token`
mode made it load-bearing for two modes rather than one: that mode is documented
as *"replayable against any payload"* and the mitigation its weakness implies
did not exist.

**`signature_scheme` is the missing axis**, orthogonal to `secret_algo`:

| `signature_scheme` | Signed payload | Timestamp from | Sender |
|---|---|---|---|
| `body` (default) | the request body | — binds none | GitHub, GitLab, most internal senders |
| `slack-v0` | `v0:{ts}:{body}` | `timestamp_header` | Slack |
| `stripe-v1` | `{ts}.{body}` | `t=` inside the signature header | Stripe |

Both constructions are taken from the vendors' published documentation, and
Slack's **published worked example** (secret, timestamp, body, digest) is a test
vector in the suite — a signature this tree builds a different way cannot
satisfy it.

Two mechanisms, complements rather than alternatives. **A freshness window**
(`max_age_seconds`, default 300, two-sided so a future-dated capture is refused
too) bounds how long a captured delivery stays useful, and by construction
catches nothing *inside* the window. **A bounded, TTL'd replay cache**
(`replay_cache_size`, default 10 000 keys) catches the copy that arrives inside
it, answering **409** — distinct from 403 so a repeat and a forgery are
distinguishable in the access log.

Three orderings are the security argument rather than style:

- **Signature first, freshness second.** In both timestamped schemes the
  timestamp is *part of the signed payload*, so it is evidence of nothing until
  the signature holds — which is also what makes the window unbypassable:
  rewriting the header to refresh a stale capture breaks the digest.
- **The cache is written last**, only for a delivery that passed both. Recording
  on arrival would let anyone who can reach the port register a guessed delivery
  id and have the *genuine* delivery refused as a replay — a denial of service
  built out of the anti-replay control.
- **The cache TTL is the window.** Past it the entry is gone and the delivery is
  also no longer fresh, so it is refused as stale rather than accepted; the two
  hand off with no gap.

**What each route shape gets**, and the backward-compatibility choice stated
rather than inherited:

| Route | Window | Replay cache |
|---|---|---|
| `slack-v0` / `stripe-v1` | on by default (300s) | keyed on the signature — automatic |
| `body` + `replay_key_header` | none available | keyed on the delivery id |
| **`body`, no delivery id** | **none available** | **none — a replay is accepted** |
| `token` + `replay_key_header` | none available | keyed on the delivery id |
| `token`, no delivery id | none available | none |

The bold row is where **every route configured before #713 sits**, and it could
not be fixed by choosing a braver default: the sender signs no timestamp, so
there is nothing to check, and the credential is a pure function of the body, so
two genuine deliveries of one payload are indistinguishable from a replay. The
posture #863 took — make the default safe, make opting out the explicit act — is
applied exactly where the material exists: a `slack-v0` / `stripe-v1` route gets
a window and a cache with **no** extra keys, and `max_age_seconds: 0` is the
announced opt-out. Where it does not, what is switched on instead is **saying
so**: each such route logs a WARNING at listener startup naming itself and both
remedies (`signature_scheme`, or `replay_key_header` — GitHub's
`X-GitHub-Delivery`, GitLab's `X-Gitlab-Event-UUID`). **Stated cost:** the
weakness persists for a deployment that reads no logs and changes nothing. The
alternative was to break every existing route.

**For `token` routes, `replay_key_header` is the whole answer.** There is no
signed payload to bind a timestamp into, and the credential is byte-identical
every time — so a cache keyed on it would refuse the second *legitimate*
delivery. A delivery id refuses a verbatim replay within the TTL; it does not
stop an attacker who *holds* the token from minting fresh requests, which is
credential compromise rather than replay, and which that mode concedes by design.

**The pair rule is preserved and extended.** Each of these is a
`validate_config` **error** and a **500** at request time, never a fall-through:
a `signature_scheme` outside the vocabulary (a typo does not degrade to `body`);
`slack-v0` with no `timestamp_header`; `stripe-v1` *with* one (its timestamp is
in the signature header, and an unsigned second source could win the window
check); `timestamp_header` on `body`, where the value is not covered by the
signature and whoever replays the request simply rewrites it — a window checking
an attacker-controlled value reads like protection and is not; and a
timestamp-bound scheme paired with `secret_algo: token`. An unparseable
timestamp is a refusal, not a skipped check. No cross-mode leniency: `slack-v0`
rejects a bare hex digest and an HMAC over the body alone, and `stripe-v1`
ignores Stripe's legacy `v0` test-mode signature per Stripe's own downgrade
guidance.

**Constant-time comparison confirmed** (the issue's fourth ask): every
credential comparison — both `secret_algo` modes and both schemes — goes through
one helper over `hmac.compare_digest`, which encodes to UTF-8 first so a crafted
non-ASCII header cannot turn a verification failure into a 500. A source-level
guard fails the build if a `==` on a digest appears beside it.

Also confirmed clean in the same pass, as the issue reported: the unbounded
pre-auth body read. `do_POST` compares `Content-Length` against
`config.max_body_size` **before** `self.rfile.read(content_length)`.

Tests: `jaato_server/shared/plugins/webhook/tests/test_replay_protection_713.py` — the
vendor vectors, a replayed request refused and a fresh one accepted on each
scheme, an expired and a future-dated timestamp, a tampered body and a tampered
timestamp, the cache-poisoning refusal, and the bounded-cache eviction counter.
No test sleeps: both the window and the TTL take `now` as a parameter, so a test
states the instant it means rather than betting on a clock (#996).

**Corporate hardening** (all stdlib, no external deps):
- **TLS/SSL**: HTTPS with optional mutual TLS (client certificate verification)
- **IP allowlisting**: CIDR-aware, IPv4/IPv6, IPv4-mapped-IPv6 normalization
- **Rate limiting**: Per-IP token-bucket algorithm
- **Replay refusal**: per-route freshness window + a bounded, TTL'd delivery cache

**Architecture:** HTTP server runs in a daemon thread using `http.server.HTTPServer`. Per-subscription event buffers (`deque(maxlen=1000)`) with `threading.Event`-based long-poll wakeup. Server starts lazily on first subscribe call.

See [Webhook Plugin Design](docs/design/webhook-plugin.md) for full design doc.

### UI Rendering Architecture (Separation of Concerns)

The UI rendering follows a strict separation between data production and presentation:

**Pipeline Layer** (`jaato_server/shared/plugins/`, `jaato_server/server/`):
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
1. **PKCE OAuth Login** (recommended for subscription): `oauth_login()` from `jaato_server.shared.plugins.model_provider.anthropic`
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

Auth: `oauth_login()` from `jaato_server.shared.plugins.model_provider.antigravity`

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
`jaato-server/jaato_server/shared/env_scope.py` — `session` (a knob two sessions on one host
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
| `JAATO_RUNNER_ACK_TIMEOUT` | Seconds a dispatched runner RPC may go with NO frame bearing its id before the daemon stops assuming and asks the runner what it actually has (default 120; `0` disables). **Not** a cap on how long an RPC may take — a turn legitimately runs for minutes, and a runner that claims the id buys another full window. What it bounds is an unbounded WAIT: before it, a request the daemon wrote and the runner does not have hung the caller forever with every thread idle (#856). Host-scoped, because it bounds the channel, which a pool slot shares across several sessions in turn. A negative or unparseable value falls back to the default — "unbounded" is the bug this exists to fix. |
| `JAATO_RUNNER_POOL_MAX_SIZE` | Hard ceiling on **total** idle pool slots, reservations included (default: `2 x JAATO_RUNNER_POOL_SIZE`).  This is the memory bound — a slot is 129–187 MB — on the growth that per-tenant reservations imply.  Raise it when `pool_replenish_ceiling_blocked_total` is nonzero: some tenant is being served by cold-spawn while reservations hold the ceiling. |
| `JAATO_IPC_TRUST_PEER_PATHS` | Switch OFF the IPC peer-entitlement check, so the daemon acts on whatever `workspace_path` / `config_root` a client names. Host-scoped: it is a property of the SOCKET, and a session must not be able to widen the transport's own trust posture. Announced at WARNING the first time it applies. See [Two Principals on One Socket](#two-principals-on-one-socket). |
| `JAATO_UMASK` | Octal umask for the daemon PROCESS, inherited by the pre-warm template, every pool slot forked from it and every runner — so it governs the mode of every file the agent writes into a workspace. Unset (the default) leaves the umask the daemon inherited, which is what every existing deployment gets. Host-scoped because `os.umask` is a process attribute and the daemon serves all of its sessions from one process: a per-session value could not be applied without racing whatever turn is already running, and would in any case miss the files the *daemon* puts in a workspace (session records, `.jaato/logs`, the provisioned tree). CLI twin `--umask`, which outranks it; a malformed value is refused at ERROR and the inherited umask is kept rather than an invented one applied. See [A Root Daemon Writes Root-Owned Files](#a-root-daemon-writes-root-owned-files-1168). |
| `JAATO_IPC_EVENT_QUEUE_MAX` | Per-client IPC event-queue bound (default 2048). Beyond it, lossy tool-output chunks are evicted oldest-first (media before text); essential lifecycle events are queued past the bound rather than dropped, because losing one desynchronises the client permanently. A non-numeric or non-positive value falls back to the default — "unbounded" is the bug this exists to fix. See [Binary Media Chunks](docs/design/binary-media-chunks.md). |
| `JAATO_CREDENTIAL_LOCK_TIMEOUT` | Seconds a caller waits for another process to finish refreshing a rotating OAuth credential before giving up (default 60). Host-scoped for the reason `JAATO_RUNNER_ACK_TIMEOUT` is: what is bounded is contention on a FILE, and the contenders — daemon, runner subprocesses, pool slots — serve sessions that have no say in each other's timeouts. A non-numeric or non-positive value falls back to the default; "unbounded" is the bug this exists to fix. See [A Refresh Token That Rotates](#a-refresh-token-that-rotates-and-two-sessions-refreshing-it-683). |
| `JAATO_OAUTH_REFRESH_MARGIN` | Seconds before real expiry at which an OAuth access token is treated as stale and refreshed (default 300 — the value each provider previously hardcoded). Host-scoped because every process sharing one credential file must agree on when that file's token is stale. Note what it does **not** do: a fixed margin does not disperse a thundering herd (every process crosses it at the same instant), it makes the refresh happen while the old token is still valid — which is what lets a transient failure fall back on it instead of logging the user out. |
| `JAATO_RELEASE_CHECK` | Set to `off` (also `0`/`no`/`false`/`none`/`never`) to stop `jaato-doctor` and `jaato-scaffold explain releases` asking PyPI / TestPyPI whether a newer jaato package is published. On by default — a notification nobody enables is a notification nobody gets — and bounded: a 3s per-index deadline, cached 6h at `~/.jaato/release_check.json`, never a FAIL, and an index that does not answer is reported as UNKNOWN rather than as currency. An unrecognised value reads as ON, because a typo that silently disables the notifier reproduces the state it exists to fix. Read only in `jaato_sdk/release_channels.py`, which is why it is not in `jaato_server/shared/env_scope.py` (that catalog is derived by a scan of `jaato_server/server/` and `jaato_server/shared/`). See [A Release Nobody Was Told About](#a-release-nobody-was-told-about). |
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
nebius-auth login/key/logout/status    # Nebius Token Factory API key
ovhcloud-auth login/key/logout/status  # OVHcloud AI Endpoints API key
doubleword-auth login/key/logout/status # Doubleword API key
```

Every provider whose `PROVIDER_AUTH_RESOLUTION` names a `stored` command has
a plugin registering it, and
`jaato_server/shared/tests/test_a_named_auth_command_exists.py` derives both halves from
the tree so a new provider cannot advertise a command nothing provides
(#888). A `stored` entry naming a FILE instead — `openai_auth.json`,
`~/.aws/credentials`, `GOOGLE_APPLICATION_CREDENTIALS` — has no command and
is not expected to: Bedrock resolves through botocore's chain and Google
through ADC.

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
.venv/bin/pip install -r requirements-telemetry.txt   # uv: uv pip install -r requirements-telemetry.txt
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
`jaato-server/jaato_server/shared/tests/test_cyclomatic_complexity_audit.py`, which runs in the
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
python jaato-server/jaato_server/shared/tests/test_cyclomatic_complexity_audit.py
```

Note that radon counts `and`/`or` and comprehensions as decision points, so a run
of defensive `x.get(k) or ""` defaults can push an otherwise flat function over
the line. The ceiling is 15 rather than 10 precisely to leave room for that; see
the test module's docstring for the measurements behind the choice.

### Comparison and Design Docs Are Checked Against the Tree

`docs/compare-*.md` and the multimodal design doc carry claims an evaluator
quotes — the licence, whether AppArmor is free, which wires carry images —
and both comparison docs had drifted once before (#866). The guard is
`jaato-server/jaato_server/shared/tests/test_docs_do_not_contradict_the_tree.py`, in the
required `contract-guards` job:

- no line a comparison doc attributes to jaato may call it MIT or open source
  (the identifier is read from `jaato-server/pyproject.toml`; a competitor's
  licence on its own line or in its own column is fine);
- no comparison-doc line may call AppArmor premium while `jaato_server/server/apparmor.py`
  ships in the free package;
- the "Where jaato is now" table in `docs/design/multimodal-model-support.md`
  must name exactly the providers whose `PROVIDER_CAPABILITIES` declare the
  row's capability, so adding `pdf_input` to a provider means updating that
  row.

### The Declared Python Floor Is One CI Runs (#1076)

Every workflow here pins a single interpreter — `python-version: "3.12"`,
no matrix — while the four published distributions declared
`requires-python = ">=3.10"`. So pip installed them on 3.10, 3.11, 3.13 and
3.14 and **nothing in CI had ever run a test on any of them**. The dev venv
was 3.11, so it did not reproduce in development either: the metadata was a
promise the build did not check, and the class of defect it hides is the one
whose only trigger is a version nobody runs.

#1076 offers two honest resolutions — test the declared range, or declare
the tested range. This tree takes the second. The floor is **`>=3.12`** in
all four published `pyproject.toml` files, the 3.10 / 3.11 classifiers are
gone, and `jaato-server/jaato_server/shared/tests/test_declared_python_floor_is_tested.py`
(in the required `contract-guards` job) is what stops the two drifting apart
again. It asserts four things and nothing more:

| Property | Why it is separate |
|---|---|
| the four distributions declare **one** floor | they install into one interpreter, so a disagreement is not a range — it is the highest of them, silently |
| that floor is installed by a **commit-triggered** job that runs `pytest` | `workflow_dispatch`-only publish workflows do not count: a version only a manual run touches is in practice never run (#736) |
| every version **any** workflow installs satisfies the floor | the reverse drift, and the worse one — a floor above the CI pin makes pip refuse the repo's own editable installs |
| no classifier names a version below the floor, and the floor's own is named | pip does not read classifiers, so that is where a stale claim outlives a corrected `requires-python`, on the PyPI project page |

**The ceiling is still a promise nobody checks, and that is said out loud
rather than implied.** `>=3.12` has no upper bound, so it still claims 3.13
and 3.14 and CI runs neither. Closing that needs a test matrix or an upper
bound; asserting an upper bound the project has not chosen would be a test
file inventing policy. So #1076's floor half is closed here and its ceiling
half is not.

**A consequence for contributors: the dev venv must be 3.12 or newer.** On
3.11 `pip install -e jaato-server/` is now refused by pip, which is the
point — the refusal is the promise being kept.

`out-of-tree-plugins/moon-phase` is deliberately outside the check. It is an
example of a third party's package, it is never published from here, and its
floor is its author's to choose.

### The Reversion Meta-Guard Never Writes to Your Checkout (#995)

`jaato-server/jaato_server/shared/tests/test_every_guard_detects_its_own_reversion.py`
proves each contract guard is not decorative by putting its defect **back** —
rewriting a source file, running the one guard test that must fail, and
restoring. Until #995 it rewrote the file **in the working tree** and restored
it in a per-case `finally`, which is an exception handler and not a
crash-safety mechanism: SIGKILL, a CI job timeout, a container restart or an
interrupt between the write and the restore leaves the sabotage on disk,
looking exactly like deliberate work in progress. Measured in one session: a
dirty tree in **seven of nine** runs, and one occasion where a `git add -A`
committed the suite's live deletion of `__repr__ = secret_safe_repr("api_key")`
— #721's protection against an API key reaching a log through a default
`repr`. It was caught only by diffing failing test **IDs** against a baseline.

The suite now copies the checkout once per session and does every sabotage,
every guard subprocess and every restore **inside the copy**. No code path in
it opens a file under the repository root for writing, so there is no signal,
timeout or interrupt that can leave your tree modified — the failure is
unreachable rather than recoverable. Three things hold that up: the one path
helper every write goes through refuses anything resolving outside the sandbox
or inside the checkout (symlinks resolved before comparing), each case asserts
the real file's bytes are unchanged afterwards, and one test states the
property directly on a real reversion.

A green run is itself evidence the copy is being read: if the subprocess
resolved the real tree instead it would see **unsabotaged** source, the guard
would pass, and the case would fail as "decorative". There is no configuration
in which a broken copy reports success.

Consequences worth knowing:

- The sandbox is a **snapshot** taken at session start. Editing the tree while
  the suite runs means the guards report on the source as it was, and the
  per-case tripwire says so by name rather than blaming the suite.
- The folklore that this suite must never run under `pytest -n` or alongside
  anything else, and that `git status` is untrustworthy near it, was a
  consequence of the in-place design and no longer applies. Each xdist worker
  builds its own sandbox, which costs disk rather than correctness.
- Cost is one copy of ~2.4k files plus a `compileall` pass — a couple of
  seconds once per session, against ~5s per case. A case runs marginally
  *faster* in the sandbox than in the checkout.
- **It has its own CI job**, `reversion-guard`, and runs in no other (#1080).
  It used to run **twice** — named by `contract-guards` and again by the
  `suite (shared/tests)` leg, which runs that whole directory — so that leg
  now `--ignore`s it. The duplication was expensive rather than merely
  wasteful because this is not a CPU-bound suite: each case spawns a pytest
  **subprocess** against the sandboxed copy, so its cost is process creation
  and I/O. Measured: the other fifteen files in `contract-guards` take
  **30-38s together**, while this one file was **20m38s of that job's 21m16s
  — 97%**.
  Being I/O-bound also makes it the most *variable* suite in the tree — the
  same block timed at **1320s and 1896s**, a 1.43x swing with no code
  difference to explain it — and wedged inside another job that variance was
  charged to whatever else that job gated, with a timeout reading as an
  unexplained red rather than as *a guard stopped detecting its own
  reversion*. Alone, it gets a cap sized for its own variance and the
  required `contract-guards` check is seconds again.
- **Only pytest's `TESTS_FAILED` counts as detection (#1065).** The verdict
  was `assert code != 0`, so *every* non-zero exit read as "the guard noticed
  its defect" — including `USAGE_ERROR` (4), which is what pytest returns for
  a nodeid it cannot resolve, having run no test body at all. A typo in a
  `test` field therefore certified a guard that was exercising nothing,
  silently and permanently, and it failed in the unsafe direction: there is no
  output on a passing case, so a malformed nodeid looks exactly like a working
  guard. **23 of the 227 in-tree reversions were in that state** when this was
  fixed — 22 naming a class-nested test without its class, and one carrying
  the whole repo-relative path in `test`. Now `1` is detection, `0` is
  decorative, and anything else is `BLOCKED` naming the exit code and quoting
  pytest's own complaint, which `_run_guard` no longer discards. A pre-flight
  (`test_every_reversion_names_a_test_that_exists`) resolves every `test`
  against one collection pass **before** anything is sabotaged, so the whole
  corpus is answered at once at the point an author can act on it.
- **Discovery no longer drops a guard in silence.** `_guard_modules` swallows
  an `ImportError` and moves on, which is right for a module's own test run
  and wrong here — its reversions vanish while this suite still reports
  success. It now records the failure when the module's *source* declares
  `REVERSIONS`, and a test surfaces it. The source is read rather than the
  module imported because the question is only asked about a module that
  already failed to import; and it is narrowed to declaring modules because
  these two packages hold 384 test modules and only 82 contribute cases.
- **An `--ignore` binds to the invocation that carries it.**
  `test_ci_runs_every_test_file.py` used to union `covered` and `ignored`
  across every invocation in every workflow and let the union of ignores win,
  so the ignore above would have made the meta-guard read as *unrun* although
  `contract-guards` names it — the coverage guard failing a repository that
  had just stopped running a file twice. Coverage is existential: one leg
  running a file is coverage, however many other legs exclude it.

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
- [Session Group Messaging](docs/design/session-group-messaging.md) - Assessment and design for any-to-any messaging between sessions that share a group (`created_by` owner or `cascade_driver_id`), waking an idle, detached, or unloaded target to process the message, with a payload of text, file references and text/binary attachments. Inventories the primitives that already exist against the requirement, names the eight missing blocks and proposes a three-phase rollout that composes rather than duplicates. **Phase 1 is shipped**: `server/session_groups.py`, the index's `membership` section, `SessionManager.deliver_group_message`, the `courier` plugin (`send_to_session` / `list_group_sessions` plus the relocated `send_to_sibling` / `list_siblings`), the `session.message` verb (protocol 1.23) and both SDK methods — see [Session Group Messaging](#session-group-messaging-the-courier-plugin). Phases 2 (durable inbox) and 3 (file references, cross-workspace copy) remain design.
- [Daemon Extensions](docs/design/daemon-extensions.md) - Extension points for external packages (session hooks, WS interceptors, custom aspects, remote handlers)
- [Application Identity](docs/design/app-identity.md) - Naming the application an integrator built, rather than reporting every SDK-based harness upstream as "jaato". `AppIdentity` + the four-tier precedence (provider knob → provider env → `JaatoRuntime(app_identity=)` → `JAATO_APP_*`), the `(powered by jaato)` suffix, header-safety sanitisation, and why the env vars are `host`-scoped.
- [Env Vars vs Profile Keys](docs/design/env-vars-vs-profile-keys.md) - Which of the 186 env vars earned a typed profile/`plugin_configs` key, and which are correctly env-only. The tagged catalog lives in `jaato_server/shared/env_scope.py` (scope: `session` / `host` / `ambient` / `internal`, plus the typed key where one exists) and is enforced by `test_env_scope_catalog.py`; 38 session-scoped knobs with no typed key sit in a may-only-shrink ratchet, each carrying a tier and a **proposed** key (`explain env untyped` prints both). Includes the credential policy for the three providers whose peers expose an `api_key` knob and they don't.
- [The Self-Bounding Completion Gate](docs/design/completion-gate.md) - What `completion_processors` is for and the seven rules a working one had to get right, each attached to the incident that produced it. Covers `max_refusals:` / `on_exhausted:` (the gate's own refusal ceiling, distinct from `budget_control`, which is the retry budget since #1068 removed the `max_turns` field that used to claim the role), the `faults[]` channel that keeps an unfixable environment fault from burning the retry budget, why a broken gate must never read as a passing one, and the load-once-per-session caching the counter used to depend on as folklore. Start from `jaato-scaffold explain completion` and `jaato-scaffold new processor` — both are computed from the framework, so they cannot drift the way the prose can. §9 covers why the gate is three files rather than one: `jaato-scaffold new sweep` emits the checks (`acceptance.sh`, shared with the post-hoc graders), the processor, and the profile's `completion_processors:` + `completion_payload_schema:` as ONE set (`--no-gate` opts out), because a profile carrying processors and no schema has no lenient gate — `_should_hide_signal_completion` removes `signal_completion` entirely, so the agent cannot signal and the gate never runs. §11 covers why a session that completed is still drivable: `signal_completion` ends the TURN, and the continuation it skips was also the only writer of that batch's results into history, so a completed conversation used to end on a `tool_calls` block nothing answered and every later request — `send_message` and `session.wake` alike — was rejected by the provider (#913). `_record_terminal_tool_results` writes them without the round-trip, which is what makes "complete every turn to enforce a contract, then keep talking" usable.
- [Payload-Schema Conventions](docs/design/payload-schema-conventions.md) - Symmetric authoring guide for `spawn_payload_schema` (input boundary) and `completion_payload_schema` (output boundary) — symmetric in everything but the type system: a completion payload is JSON the model emitted, a spawn payload crosses the IPC wire as `key=value` argv tokens, so **every spawn property is a `string`** (`pattern` carries the shape, the consumer parses). #883 ratified that rather than reopening the transport, and both spawn boundaries now validate the same string view — the in-process `spawn_subagent` call used to accept a typed value the wire could never deliver, so one profile meant two things. A refusal caused by the schema names the profile; `jaato-scaffold validate` catches it before any spawn as `spawn_schema_type_unreachable`. Mirror prefetch required-keys; always carry `warnings[]` / `errors[]` escape hatches; persona ↔ schema consistency check; canonical-hash strip rules; `agent_params` interaction with agent-continuity (§6).
- [Competitor Memory Systems](docs/design/competitor-memory-systems.md) - Survey of nine agent-memory products, sorted by what a *framework* owes: pattern (nothing) / seam (an extension point) / fidelity (a fix) / not ours. Records which items were already expressible as cascade patterns, which memory hot paths are not pluggable, and why the pattern corpus needs `certify/`-style contract tests run against `main`.
- [Agent Continuity Pattern](docs/design/agent-continuity.md) - `{{continuity_scope}}` + memory plugin enrichment + raw/curated lifecycle: persona-level continuity across sessions composed from existing primitives, no new framework code. Reference impl in `jaato-knowledge-manager/.jaato.example/`.
- [Model Tiers × Prompt Caching](docs/design/model-tier-prompt-cache.md) - What `enter_tier` costs when prompt caching is on: cache is keyed per model, so an in-place tier switch re-reads the whole prefix cold (break-even ~6 consecutive calls at the new tier). Covers the `_wire_cache_plugin` gap that made profile cache knobs inert, the system-block tier line that invalidates BP1, and the per-provider knob divergence + proposed common `cache:` field.
- [MiniMax, Kimi and MiMo providers](docs/design/minimax-kimi-mimo-providers.md) - Design for three first-party OpenAI-compatible providers (`minimax`, `kimi`, `mimo`) and the framework prerequisite they share: **reasoning replay** — sending an assistant turn's `reasoning_content` back on the next request of a tool-call loop, which the session currently drops from history and every OpenAI-shaped converter ignores. Covers the surface decision (chat completions, not the Anthropic shims), per-vendor thinking-control dialects, tool-choice vocabularies, catalog vs table context resolution, error taxonomies, and the registration checklist.
- [Plugin schema census](docs/design/plugin-schema-census.md) - Which config keys each plugin READS that its `get_config_schema()` does not DECLARE, measured tree-wide by `scripts/plugin_schema_census.py`. A census, deliberately **not** a guard: the raw count spans four surfaces (framework-injected keys, the block an author writes, a nested dict with its own owner, and a separate file the block points at), and a ratchet seeded before those are separated would freeze the ambiguity as a fact. `permission` is worked through site by site as the one audited row.
- [The audit log](docs/audit-log.md) - Which of the five stores records what, the three rules a reader of them must apply, and what `record_keeping:` changes about DELETE. Plus the tamper-evidence contract: a `sha256-chain` proves no edit in place, never authorship.
- [The jaato-eval results contract](docs/eval-results.md) - What a reader outside `jaato-eval` may rely on in a results file: `results_version` (and why an absent one is an unknown one, refused by name), `caveats` (rendered verbatim, because a second copy of a caveat is the copy that rots), and the three reader rules that keep the numbers from misleading.
- [jaato as a component](docs/jaato-component-pack.md) - The Article 25(4) information pack, generated by `jaato-scaffold new dossier --component` and committed as the first versioned instance: what the framework guarantees with the thing that enforces each, what it does not, and the versioned surfaces a written agreement can cite.
- [EU AI Act evidence manual](docs/eu-ai-act-manual.md) - One section per control, with a capture of each doing its job on a live daemon. Regenerated, never edited: `python scripts/eu_ai_act_evidence.py` rebuilds every picture under `docs/eu-ai-act-manual/evidence/` from a real run (echo provider, private socket), so a mechanism that goes inert shows up as a picture that says so — which is how #1139 was found.
- [EU AI Act](docs/design/eu-ai-act.md) - What Regulation (EU) 2024/1689 asks of a jaato *application* (the AI system is the profile + persona + tools + model binding; jaato is a component supplier under Art. 25(4), and BUSL-1.1 is not a free and open-source licence, so neither Art. 2(12) nor the 25(4) carve-out applies), which obligations bind when after the Digital Omnibus (Art. 50 disclosure and marking since 2 Aug 2026; Annex III high-risk from 2 Dec 2027), and the mechanisms in order. Every mechanism it names is shipped: the `regulatory:` profile block, the `disclosure` piece and the first-interaction announcement, `generated_by` plus the `TRAIT_OUTPUT_MARKER` hook, one audit-record contract with `record_keeping:` retention and a sha256 chain, the incident register, memory provenance, and the Annex IV dossier generator with its `jaato-eval` accuracy section. What remains is recorded there as a decision rather than a gap. See [EU AI Act Mechanisms](#eu-ai-act-mechanisms).
- [Per-User GitHub Credentials](docs/design/per-user-github-credentials.md) - Proposed (#1225–#1228): how a multi-user web deployment on a root daemon gives each session its WUI user's GitHub token. The BFF holds the grant (GitHub App, refresh token encrypted per OIDC `sub`) and binds an account per workspace; the workspace `.env` carries only a reference (`GH_TOKEN=app://github`), which the daemon resolves at every spawn by asking the owning application over its bind channel, so cascade, wake and revived sessions get it too and nothing resolved is persisted. Includes the per-workspace `.home/` for model-driven subprocesses.
- [GitHub Workspace Guidance](docs/design/github-workspace-guidance.md) - Proposed (#1240, docs-only, application-scoped): how the web coder ships the "use `gh` safely in a shared workspace" rule-set into every workspace it binds a GitHub account to, as application-managed files (no daemon change). The BFF writes `.jaato/instructions/40-github.md` at bind time beside the `.env`/`.gitconfig` it already seeds, so cascade/wake/revive sessions get the rules as they get the token; the UI refreshes on session start. Recommends BFF-as-primary-writer, one gitconfig source of commit identity, a force-push permission blacklist (enforced) plus prose (judgement), helper+prose worktree cleanup, and a generic managed-file mechanism GitLab can later reuse.
- [AppArmor Setup](docs/apparmor-setup.md) - Kernel-enforced workspace isolation. WS deployments confine automatically when AppArmor is available; IPC clients opt in via `IPCClient(..., apparmor=True)` (defaults to `False`).
- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
