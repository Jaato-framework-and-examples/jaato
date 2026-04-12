# Jaato vs. LangChain Deep Agents — Feature Comparison

## Overview

|  | **Jaato** | **Deep Agents** |
|---|---|---|
| **Tagline** | "Just another agentic tool orchestrator" | "Batteries-included agent harness" |
| **Creator** | Independent project | LangChain (langchain-ai) |
| **Language** | Python | Python (+ separate JS/TS variant) |
| **License** | — | MIT |
| **Architecture** | Server-first daemon with IPC/WebSocket clients | LangGraph `StateGraph` compiled agent |
| **Foundation** | Custom runtime from scratch | Built on LangChain + LangGraph |
| **Default model** | Configurable (no hard default) | `claude-sonnet-4-6` |

**Key philosophical difference:** Jaato is a **standalone server** — a daemon
that clients connect to from anywhere (local IPC or remote WebSocket). Deep
Agents is a **library** — you call `create_deep_agent()` and get a compiled
LangGraph graph you can embed, deploy, or run in a CLI.

---

## Model Provider Support

| Provider | **Jaato** | **Deep Agents** |
|----------|-----------|-----------------|
| Anthropic Claude | Yes (API key + OAuth PKCE) | Yes (default, with prompt caching) |
| Google GenAI / Vertex AI | Yes | Yes (via langchain-google-genai) |
| OpenAI / OpenAI-compatible | Yes (via NIM, ZhipuAI-OpenAI, and GitHub Models providers — all use the `openai` SDK; any OpenAI-compatible endpoint works by setting the base URL) | Yes (native, via LangChain `init_chat_model()`) |
| GitHub Models | Yes (device code OAuth) | No |
| NVIDIA NIM | Yes | Yes (partner extra) |
| Ollama (local) | Yes | Yes (partner extra) |
| Claude CLI (subscription) | Yes (unique — uses Claude Pro/Max without API credits) | No |
| Google Antigravity | Yes (Gemini 3, Claude via Google) | No |
| ZhipuAI | Yes | No |
| OpenRouter | No | Yes |
| Any `BaseChatModel` | No (must write a provider plugin) | Yes (pass any LangChain model) |

**Verdict:** Jaato has more *niche* providers (Claude CLI, Antigravity, ZhipuAI,
GitHub Models). Deep Agents wins on *breadth* via LangChain's
`init_chat_model()` — any LangChain-compatible model works out of the box.
Jaato's NIM provider serves as a general-purpose OpenAI-compatible client for
any endpoint exposing the OpenAI chat completions API (vLLM, LiteLLM, LocalAI,
text-generation-inference, etc.).

---

## Tool Systems

| Capability | **Jaato** | **Deep Agents** |
|---|---|---|
| **Total plugins/tools** | 55+ plugins | ~9 built-in tools |
| **File operations** | `readFile`, `writeNewFile`, `updateFile`, `removeFile`, `moveFile`, `undoFileChange` | `read_file`, `write_file`, `edit_file` |
| **Shell execution** | `cli` plugin (configurable per profile) | `execute` tool (opt-in, requires sandbox) |
| **Interactive shells (PTY)** | Yes (`interactive_shell` — REPLs, SSH, debuggers) | No |
| **File search** | `filesystem_query` (glob, depth, size filters) | `ls`, `glob`, `grep` |
| **AST-level code search** | Yes (`ast_search` — Python, JS, TS, Go, Rust, Java, C/C++) | No |
| **LSP integration** | Yes (diagnostics, enrichment) | No |
| **Jupyter notebooks** | Yes (`notebook` plugin) | No |
| **Web search** | Yes (DuckDuckGo) | Yes (CLI only) |
| **Web fetch** | Yes | No |
| **Calculator** | Yes | No |
| **MCP servers** | Yes (first-class, `.mcp.json` config) | Yes (via `mcp.json` config) |
| **Todo/planning** | Yes (`todo` plugin) | Yes (`write_todos` tool) |
| **Vision/multimodal** | Yes (`multimodal`, `vision_capture`) | No |
| **Custom skills** | Yes (`prompt_library` plugin) | Yes (`SKILL.md` files with YAML frontmatter) |
| **Deferred tool loading** | Yes (core vs. discoverable) | No |
| **Tool traits** | Yes (semantic traits like `TRAIT_FILE_WRITER`) | No |
| **Parallel tool execution** | Yes (8 concurrent, thread pool) | Not documented |
| **Background tasks** | Yes (`background` plugin — auto-backgrounds long tool calls) | No |
| **Webhooks (inbound)** | Yes (HTTP listener, HMAC, TLS, rate limiting, IP allowlisting) | No |

**Verdict:** Jaato has a significantly richer tool ecosystem — interactive
shells, AST search, LSP, notebooks, vision, background tasks, webhooks, and
deferred loading are all absent from Deep Agents. Deep Agents keeps it minimal
by design.

---

## Subagent / Multi-Agent Orchestration

| Capability | **Jaato** | **Deep Agents** |
|---|---|---|
| **Async spawning** | Yes — `spawn_subagent` returns immediately, runs in thread pool | Yes — 3 types (sync `SubAgent`, compiled `CompiledSubAgent`, remote `AsyncSubAgent`) |
| **Parallel execution** | Up to 4 concurrent subagents | No hard cap for AsyncSubAgent (remote server scales) |
| **Shared resources** | Provider config, plugin registry, permissions, ledger | Inherits parent tools/permissions (can override) |
| **Isolated state** | Per-session conversation history | Per-subagent middleware stack |
| **Event-driven results** | Yes — `COMPLETED`, `IDLE`, `ERROR`, `CANCELLED`, `CLARIFICATION_REQUESTED`, `PERMISSION_REQUESTED` events injected into parent prompt | Via LangGraph API polling for AsyncSubAgent |
| **Bidirectional communication** | Yes — `send_to_subagent` (parent→child) + `share_context` (child→parent, structured: files, findings, notes) | No mid-flight communication |
| **Activity phase introspection** | Yes — `idle`, `waiting_for_llm`, `streaming`, `executing_tool` | No |
| **Session reuse** | Yes — send more work to idle subagents without respawning | No |
| **Remote delegation** | Yes — `server` parameter delegates to peer jaato servers | Yes — `AsyncSubAgent` communicates with remote LangGraph servers |
| **Agent profiles** | Yes — JSON profiles in `.jaato/profiles/` with model, provider, plugins, system instructions, GC | Via subagent config (name, description, system_prompt, tools) |
| **Parameterized agents** | Yes — `.jaato/agents/` definitions with `{{param}}` placeholders | No |
| **Per-subagent GC config** | Yes — configurable strategy and threshold per subagent | No |
| **Cancellation** | Yes — `cancel_subagent` with graceful lifecycle | No explicit cancel (AsyncSubAgent has cancel tool) |
| **Auto-inserted default** | No | Yes (general-purpose subagent auto-inserted) |

**Verdict:** Both frameworks support async subagent spawning. Jaato has richer
lifecycle control (bidirectional communication, activity phases, session reuse,
structured context sharing, per-subagent GC). Deep Agents' `AsyncSubAgent`
provides remote-first execution via LangGraph Platform with potentially higher
concurrency limits. Jaato achieves remote delegation natively via its
client-server architecture and the `server` parameter for peer-to-peer
delegation.

---

## Architecture: Execution Location & Client-Server Model

| Aspect | **Jaato** | **Deep Agents** |
|---|---|---|
| **Default execution model** | Client-server: daemon runs execution, clients connect from anywhere | In-process library; CLI wraps it |
| **Local clients** | IPC (Unix domain socket) | Direct function call |
| **Remote clients** | WebSocket server (built-in) | Requires LangGraph Platform deployment |
| **Web clients** | `jaato-task.js` web component (premium) with `ExternalEventRequest` protocol for bidirectional host↔agent communication | Separate `deep-agents-ui` repo |
| **Multi-session** | Native — `SessionManager` with disk persistence | Via LangGraph checkpointers |
| **Multi-client** | Native — IPC + WebSocket simultaneously, multiple clients per session | Via LangGraph Platform |
| **Multi-tenant isolation** | AppArmor per-session kernel-enforced sandboxing | Sandbox backends (Daytona, Modal, Runloop) |
| **Workspace provisioning** | Server-provisioned per WS client, auto-cleanup (24h reaper) | Manual or via sandbox provider |
| **Session persistence** | Built-in disk persistence | Via LangGraph `BaseStore` / checkpointers |
| **Auto-reconnection** | Yes (`IPCRecoveryClient` with state recovery) | Not built-in |

**Key distinction:** Jaato was designed as a remote server from the start — the
client is always a separate process (or host). Subagents, tool execution, and
model calls all run on the server. This means "remote execution" is not a
special feature; it's the default architecture. Deep Agents is an in-process
library that requires LangGraph Platform to get equivalent client-server
separation.

---

## Context Management / GC

| Capability | **Jaato** | **Deep Agents** |
|---|---|---|
| **Strategies** | 4 plugins: truncate, summarize, hybrid, budget | Summarization middleware + filesystem offloading |
| **Budget-aware GC** | Yes (5-tier priority: ENRICHMENT → EPHEMERAL → PARTIAL → PRESERVABLE → LOCKED) | No |
| **Proactive triggering** | Yes (during streaming, pre-send checks) | Not documented |
| **Threshold config** | Yes (default 80% context usage) | Not documented |
| **Continuous mode** | Yes (per-turn trimming in `gc_budget`) | No |
| **Preserve recent turns** | Yes (configurable, default 5) | Not documented |
| **Memory files** | Yes (`memory` plugin with semantic matching) | Yes (`AGENTS.md` files via `MemoryMiddleware`) |

**Verdict:** Jaato has far more sophisticated context management with 4
pluggable GC strategies, budget-aware removal with 5-tier priority, and
proactive triggering. Deep Agents relies primarily on summarization and
filesystem offloading.

---

## Permissions & Security

| Capability | **Jaato** | **Deep Agents** |
|---|---|---|
| **Permission granularity** | 8 scopes: `yes`/`no`/`once`/`always`/`never`/`turn`/`idle`/`all` | Human-in-the-loop approval (opt-in) |
| **Evaluation order** | Deterministic: suspensions → sanitization → blacklist → whitelist | Declaration order (first match wins) |
| **Shell injection prevention** | Yes (pattern-based sanitization) | Shell execution is opt-in |
| **Workspace sandboxing** | AppArmor (kernel-enforced) + directory containment | Virtual mode path restriction + sandbox backends |
| **Remote sandboxes** | Peer delegation via `server` param | Yes (LangSmith, Daytona, Modal, Runloop) |
| **Shell availability** | Configurable per profile — omit `cli` plugin for no shell access | No shell by default (`StateBackend`) |
| **Threat model documented** | No formal document | Yes (`THREAT_MODEL.md` with 9 identified threats) |

**Verdict:** Both frameworks allow controlling tool availability through
configuration (jaato profiles vs. Deep Agents backends). Jaato adds
significantly more granular runtime permissions (8 approval scopes,
sanitization, deterministic evaluation) and kernel-level sandboxing via
AppArmor. Deep Agents has a wider selection of third-party sandbox backends
and a formal threat model document.

---

## Client / UI

| Capability | **Jaato** | **Deep Agents** |
|---|---|---|
| **TUI** | Rich TUI (custom, prompt_toolkit-based) | CLI (Textual-based) |
| **Headless / CI mode** | Yes (`headless_mode.py` — auto-approve, file output, session isolation) | Yes (non-interactive mode) |
| **Themes** | 4 built-in + custom JSON themes | Not documented |
| **Plan panel** | Yes (toggle with Ctrl+P) | Not documented |
| **Tools panel** | Yes (toggle with Ctrl+T) | Not documented |
| **Custom keybindings** | Yes (project + user level) | Not documented |
| **External editor** | Yes (Ctrl+G opens `$EDITOR`) | Not documented |
| **Search in output** | Yes (Ctrl+F) | Not documented |
| **Large paste handling** | Yes (auto-placeholder for >10 lines) | Not documented |
| **Screenshot capture** | Yes (SVG/PNG for vision models) | No |
| **Web UI** | `jaato-task.js` web component (premium, WebSocket) | Yes (`deep-agents-ui`, separate repo) |
| **Editor integration** | No | Yes (ACP protocol for Zed) |
| **GitHub Action** | No | Yes (`action.yml` with 13 inputs, memory scoping) |

**Verdict:** Jaato has a more feature-rich TUI with panels, themes,
keybindings, search, and vision capture. Both have headless modes for
CI/CD automation. Deep Agents has a pre-built GitHub Action and editor
protocol (ACP for Zed) that jaato lacks.

---

## Observability & Telemetry

| Capability | **Jaato** | **Deep Agents** |
|---|---|---|
| **OpenTelemetry** | Yes (full span hierarchy, semantic conventions) | Via LangSmith (LangChain tracing) |
| **Token accounting** | Yes (`TokenLedger`, shared across sessions) | Via LangSmith |
| **Evaluation suite** | No | Yes (`deepagents-evals`, `deepagents-harbor`) |
| **Structured logging** | Yes (per-session logs) | Via LangSmith |

**Verdict:** Jaato has native OpenTelemetry with detailed span hierarchies
(`jaato.turn` → `jaato.tool` → `jaato.permission`). Deep Agents leverages
LangSmith for observability and adds a dedicated evaluation/benchmarking
suite that jaato lacks.

---

## Enrichment & Formatting

| Capability | **Jaato** | **Deep Agents** |
|---|---|---|
| **Enrichment pipeline** | Yes (8 formatter plugins) | No |
| **Diff rendering** | Yes (unified/side-by-side, word-level) | No |
| **Code block formatting** | Yes (language-aware truncation) | No |
| **Mermaid diagrams** | Yes (rendering hints) | No |
| **Table formatting** | Yes (ASCII tables) | No |
| **Presentation context** | Yes (`PresentationContext` — adapts output to client type and terminal size) | No |

**Verdict:** Jaato has a unique enrichment pipeline that adapts output to
client capabilities. Deep Agents has no equivalent.

---

## Extensibility

| Capability | **Jaato** | **Deep Agents** |
|---|---|---|
| **Extension model** | Plugin system (55+ plugins, 6 plugin kinds) | Middleware stack (strict-ordered chain) |
| **Plugin kinds** | `tool`, `enrichment`, `gc`, `session`, `model_provider`, `cache` | Middleware only |
| **Entry points** | Python `pyproject.toml` entry points | Not documented |
| **Daemon extensions** | Yes (session hooks, WS interceptors, custom aspects) | No (not a daemon) |
| **Provider profiles** | Plugin + provider config | `_HarnessProfile` per provider |
| **Plugin config schema** | Yes (`get_config_schema()` with `PluginSetting` introspection) | Not documented |

---

## Summary: When to Choose Which

| Use Case | **Better Choice** | **Why** |
|---|---|---|
| **Quick start with any LLM** | Deep Agents | `create_deep_agent()` with any LangChain model |
| **LangChain ecosystem** | Deep Agents | Native LangGraph, LangSmith, LangServe |
| **Editor integration (Zed)** | Deep Agents | ACP protocol |
| **GitHub Actions CI/CD** | Deep Agents | Pre-built `action.yml` |
| **Evaluation & benchmarking** | Deep Agents | Built-in eval suite |
| **Production daemon / server** | Jaato | Built-in daemon, multi-session, IPC/WebSocket |
| **Remote/web clients** | Jaato | Native client-server architecture, `jaato-task.js` web component |
| **Rich terminal experience** | Jaato | Panels, themes, keybindings, search, vision |
| **Context window management** | Jaato | 4 GC strategies, budget-aware, proactive triggering |
| **Fine-grained permissions** | Jaato | 8 permission scopes, sanitization, AppArmor |
| **Multi-tenant deployment** | Jaato | Kernel-enforced per-session isolation |
| **Interactive processes (REPLs, SSH)** | Jaato | PTY-based interactive shell plugin |
| **Code intelligence (AST, LSP)** | Jaato | AST search + LSP diagnostics |
| **Subagent lifecycle control** | Jaato | Bidirectional comms, session reuse, activity phases |
| **Webhook-driven workflows** | Jaato | Built-in webhook listener with corporate hardening |
| **Tool ecosystem depth** | Jaato | 55+ plugins vs. ~9 built-in tools |

### Bottom Line

**Jaato** is a more feature-complete, self-contained system — a full server
with 55+ plugins, sophisticated context management, kernel-level security, and
a rich TUI. Its client-server architecture makes remote execution the default,
not a special mode.

**Deep Agents** is a leaner, ecosystem-integrated harness — simpler to start
with, model-agnostic via LangChain, and plugged into LangGraph Platform for
deployment, LangSmith for observability, and a growing partner ecosystem for
sandboxing. It trades depth of features for breadth of integrations and
ecosystem reach.
