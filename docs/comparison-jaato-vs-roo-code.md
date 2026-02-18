# Jaato vs Roo Code: Feature Set & Architectural Comparison

## Executive Summary

**Jaato** is a Python-based, server-first agentic tool orchestrator with a
plugin-driven architecture, multi-provider support, and a standalone TUI client.
It runs as a daemon process that clients connect to via IPC or WebSocket.

**Roo Code** is a TypeScript-based VS Code extension (forked from Cline) that
embeds AI agents directly into the editor. It uses a mode-based system where
specialized AI personas handle different tasks with scoped tool permissions.

They solve overlapping problems — orchestrating LLM-driven tool use — but from
fundamentally different architectural positions.

---

## 1. Architecture

| Aspect | Jaato | Roo Code |
|--------|-------|----------|
| **Language** | Python | TypeScript |
| **Runtime model** | Standalone daemon (server-first) | VS Code extension (editor-embedded) |
| **Client connectivity** | Unix IPC socket + WebSocket; multiple clients can connect simultaneously | Single VS Code instance; Roo Code Cloud for remote/team use |
| **Separation of concerns** | Server (logic) / SDK (events) / TUI (presentation) — three distinct packages | Extension core + webview UI — tightly coupled to VS Code |
| **Session model** | `JaatoRuntime` (shared) + `JaatoSession` (per-agent, isolated state) | Task-based sessions within the extension; checkpoint system for history |
| **Event protocol** | 25+ typed events (JSON-serializable) between server and clients | Internal VS Code message passing; webview ↔ extension host |
| **Deployment** | Any terminal; no IDE dependency | VS Code required (JetBrains port exists but is secondary) |

### Architectural Implications

Jaato's daemon architecture means the AI backend survives client disconnects,
supports multiple simultaneous clients (e.g., TUI + web dashboard), and enables
headless/CI use. The cost is more operational complexity (PID management, socket
permissions, daemon lifecycle).

Roo Code's editor-embedded architecture provides zero-config setup (install
extension, add API key, go) and deep VS Code integration (inline diffs,
workspace awareness, terminal access). The cost is tight coupling to VS Code.

---

## 2. Model Provider Support

| Provider | Jaato | Roo Code |
|----------|-------|----------|
| **Anthropic Claude** | Native SDK with PKCE OAuth + API key | Via API key or API proxy |
| **Google Gemini / Vertex AI** | Native SDK with ADC, service account, impersonation | Via API key or OpenRouter |
| **OpenAI / GPT** | Not directly supported | Via API key |
| **GitHub Models** | Native SDK with device code OAuth | Via GitHub Copilot integration |
| **Ollama (local)** | Native plugin (Anthropic-compatible API) | Via OpenAI-compatible endpoint |
| **Claude CLI** | Wraps `claude` CLI (uses subscription, no API credits) | Not supported |
| **Google Antigravity** | Native plugin (Gemini 3, Claude via Google OAuth) | Not supported |
| **OpenRouter** | Not directly supported | First-class support (model marketplace) |
| **AWS Bedrock** | Not directly supported | Supported |
| **Azure OpenAI** | Not directly supported | Supported |
| **Provider abstraction** | `ModelProviderPlugin` protocol with type converters per provider | API adapter layer with model-specific cost calculators |

Jaato has deeper integration with Google and GitHub ecosystems (ADC, service
accounts, device code OAuth). Roo Code has broader coverage of commercial API
providers (OpenAI, Azure, Bedrock, OpenRouter).

---

## 3. Tool System

### Built-in Tools

| Category | Jaato | Roo Code |
|----------|-------|----------|
| **File read** | `filesystem_query` (glob, grep, read) | `read_file` (up to 5 files at once) |
| **File write** | `file_edit` (diff-based, with backup/undo) | `write_to_file`, `apply_diff`, `insert_content`, `search_and_replace` |
| **Shell execution** | `cli` (non-interactive subprocess) | `execute_command` (terminal) |
| **Interactive shell** | `interactive_shell` (persistent PTY via pexpect) | Not available as a distinct tool |
| **Web search** | `web_search` (DuckDuckGo) | Not built-in (via MCP) |
| **Web fetch** | `web_fetch` (HTTP + markdown conversion) | Not built-in (via MCP) |
| **Browser automation** | Not built-in | `browser_action` (requires Puppeteer/Playwright) |
| **Code search** | `ast_search` (AST-level, multi-language) | `codebase_search` (semantic via embeddings), `list_code_definition_names` |
| **Calculator** | `calculator` (configurable precision) | Not built-in |
| **Memory** | `memory` plugin (self-curated persistent knowledge) | Via Memory Bank MCP Server (structured markdown files) |
| **Task planning** | `todo` plugin (plan registration + progress tracking) | Not a distinct tool; handled by mode switching |
| **Clarification** | `clarification` plugin (single/multi-choice, free text) | `ask_followup_question` |
| **Subagent delegation** | `subagent` plugin (spawn agents with own tools/instructions) | `new_task` + `switch_mode` (mode-based delegation) |
| **Multimodal** | `multimodal` plugin (image handling) | Image support in chat |
| **Environment info** | `environment` plugin | Not a distinct tool |
| **Tool count** | 43+ plugins, many with multiple tools each | ~15 built-in tools + MCP extensions |

### Tool Discovery & Loading

| Feature | Jaato | Roo Code |
|---------|-------|----------|
| **Deferred loading** | Core tools always loaded; others on-demand via `list_tools()` → `get_tool_schemas()` | All tools sent in system prompt per request (on-demand loading proposed in issue #5373) |
| **Tool traits** | Semantic classification (`TRAIT_FILE_WRITER`) driving enrichment pipelines | Mode-based tool scoping (modes define which tools are available) |
| **Introspection** | Model can discover tools at runtime via introspection plugin | Tool list is static per mode configuration |

Jaato's deferred loading is a meaningful token optimization — only core tools
appear in the initial context, and the model discovers others on demand. Roo Code
currently sends all tool specifications in every prompt, which is a known
cost/context issue being addressed.

---

## 4. MCP (Model Context Protocol) Support

| Feature | Jaato | Roo Code |
|---------|-------|----------|
| **Configuration** | `.mcp.json` at project root | Global `mcp_settings.json` + project `.roo/mcp.json` |
| **Transport** | stdio servers | stdio + Streamable HTTP + legacy SSE |
| **Multi-server** | `MCPClientManager` manages multiple servers; `call_tool_auto()` finds the right server | Multiple servers supported; per-server enable/disable |
| **Tool discovery** | Auto-discovers tools; integrates into deferred loading pipeline | Runtime discovery; full specs sent in every prompt |
| **Configuration scope** | Project-level | Global + project-level (project overrides global) |
| **MCP server creation** | Not a built-in feature | AI can create MCP servers autonomously (via `fetch_instructions`) |

---

## 5. Agent & Task Orchestration

### Jaato: Subagent Architecture

- Subagents are full `JaatoSession` instances sharing the parent's `JaatoRuntime`
- Each subagent can have its own model, tool set, and system instructions
- Cross-provider subagents are possible (parent on Gemini, child on Claude)
- Profile system (`.jaato/profiles/`) for predefined agent configurations
- Parallel execution with `background=true`
- Shared state via thread-safe inter-agent communication
- Mid-turn prompt injection — messages can be queued during model execution

### Roo Code: Mode-Based Delegation

- Modes define AI personas with scoped tool permissions and instructions
- 5 built-in modes: Code, Architect, Ask, Debug, Custom
- `new_task` tool delegates work to a specific mode
- Boomerang pattern: orchestrator mode dispatches subtasks and collects results
- Community Mode Gallery for sharing configurations
- Modes are defined in `.mode.md` files or via UI
- Modes can request switching to another mode when stepping outside their scope

### Comparison

Jaato's subagent model is more flexible at the infrastructure level — different
models, different providers, different tool subsets, parallel execution, shared
runtime resources. This is useful for complex multi-step workflows where
subtasks have genuinely different resource requirements.

Roo Code's mode system is more opinionated and user-facing — modes are named
personas with clear responsibilities, making it easier to understand what the AI
is doing and why. The mode gallery creates a community ecosystem around agent
configurations.

---

## 6. Context Management & Garbage Collection

| Feature | Jaato | Roo Code |
|---------|-------|----------|
| **GC strategies** | 4 plugins: truncate, summarize, hybrid (generational), budget-aware | Context condensing (AI-summarized) + sliding window truncation |
| **Trigger mechanism** | Proactive during streaming (configurable threshold, default 80%) | Triggers at ~80% of context window |
| **Configuration** | `GCConfig` with threshold, preserved turns, auto-trigger | Settings-based; context condensing toggle |
| **Token budget tracking** | `InstructionBudget` tracks usage by source (base, plugins, history, current) | Token counting per request with debounced UI updates |
| **History preservation** | Configurable number of recent turns preserved | Original messages preserved internally; restored on checkpoint rewind |
| **Prompt caching** | Anthropic prompt caching (90% cost reduction), configurable | Anthropic prompt caching; OpenRouter "middle-out" compression |

Jaato has significantly more sophisticated GC — four strategies including a
generational hybrid approach and budget-aware tracking that prevents fixed-cost
instructions from being garbage collected. The `InstructionBudget` system that
tracks token usage by source (base instructions vs. plugins vs. history) is
architecturally distinctive.

Roo Code's approach is simpler (condense or truncate) but benefits from
checkpoint-based history restoration, which is a different kind of safety net.

---

## 7. Permission & Security Model

| Feature | Jaato | Roo Code |
|---------|-------|----------|
| **Granular control** | Per-tool policies, blacklists, whitelists, interactive editing of parameters | Per-tool auto-approve settings, mode-scoped tool access |
| **Approval strategies** | `y/n/a/t/i/once/never/all` — including turn-scoped and idle-scoped | Approve/reject per action; auto-approve rules |
| **Configuration** | `permissions.json` + interactive runtime control | VS Code settings UI |
| **Parameter editing** | Can edit tool parameters at approval time | Cannot edit parameters before approval |
| **Output sanitization** | Redaction and masking of sensitive data in tool outputs | Not documented |
| **Policy engine** | Reliability policies with per-tool thresholds and prerequisites | Not documented |

Jaato's permission system is more granular — turn-scoped and idle-scoped
approvals, parameter editing at approval time, and output sanitization are
features that matter in security-conscious environments.

---

## 8. Streaming & Cancellation

| Feature | Jaato | Roo Code |
|---------|-------|----------|
| **Streaming granularity** | Token-level with usage callbacks during streaming | Streaming with 2-second debounced token updates |
| **Cancellation model** | `CancelToken` (cooperative, propagates to subagents) | Task abort via UI |
| **Mid-stream recovery** | Incomplete function calls recovered on cancellation | Grace retry for tool errors |
| **Activity phases** | Tracked (IDLE, WAITING_FOR_LLM, STREAMING, EXECUTING_TOOL) | Task state tracking |

---

## 9. UI & Presentation

| Feature | Jaato | Roo Code |
|---------|-------|----------|
| **Primary UI** | Rich TUI (Prompt Toolkit + Rich library) | VS Code sidebar webview |
| **Layout** | Multi-pane: input, output, plan, budget, workspace panels | Chat interface with inline diffs and terminal |
| **Theming** | Built-in (dark, light, high-contrast) + custom JSON themes | VS Code theme integration |
| **Presentation context** | `PresentationContext` lets model adapt output to client capabilities | Not documented as a model-facing feature |
| **Multi-client** | Multiple clients can render the same session differently | Single VS Code instance per session |
| **IDE integration** | None (standalone terminal tool) | Deep VS Code integration (inline diffs, workspace, terminal, problems panel) |
| **Browser IDE** | WebSocket support enables web-based clients | Roo Code Cloud for remote agents |

---

## 10. Unique Strengths

### Jaato Distinguishing Features

1. **Server-first daemon architecture** — survives client disconnects, supports
   multiple simultaneous clients, enables headless/CI use
2. **Interactive shell sessions** — persistent PTY sessions for REPLs, debuggers,
   SSH, package wizards (not just one-shot command execution)
3. **Deferred tool loading** — reduces token cost by only loading core tools
   initially; model discovers others on demand
4. **Four GC strategies** — including generational hybrid and budget-aware
   approaches with per-source token tracking
5. **Cross-provider subagents** — parent on Gemini, child on Claude, sharing
   runtime resources without redundant connections
6. **Mid-turn prompt injection** — queue messages while model is executing; model
   decides how to handle them
7. **Permission parameter editing** — modify tool parameters at approval time
8. **Presentation context protocol** — model adapts output format based on
   client capabilities

### Roo Code Distinguishing Features

1. **VS Code native** — inline diffs, workspace awareness, terminal integration,
   problems panel — all without leaving the editor
2. **Mode system with community gallery** — named personas with scoped
   permissions; community-shared configurations
3. **Zero-config onboarding** — install extension, add API key, start coding
4. **Browser automation** — built-in `browser_action` tool for web testing
5. **Broader provider coverage** — OpenAI, Azure, Bedrock, OpenRouter
   out of the box
6. **Checkpoint system** — rewind to earlier conversation states with full
   history restoration
7. **Codebase indexing** — semantic search via configurable embedding providers
   and vector databases
8. **Roo Code Cloud** — team features, GitHub integration, remote agents,
   SOC 2 compliance

---

## 11. Architectural Trade-offs Summary

| Trade-off | Jaato's Position | Roo Code's Position |
|-----------|------------------|---------------------|
| **Coupling** | Loosely coupled (server/client/SDK separation) | Tightly coupled to VS Code |
| **Extensibility** | Plugin system (43+ plugins, trait-based) | MCP + custom modes |
| **Complexity** | Higher operational complexity (daemon, sockets, PID management) | Lower (install extension, done) |
| **Token efficiency** | Deferred loading + 4 GC strategies + budget tracking | Full tool specs per request (optimization in progress) |
| **IDE integration** | None — trades IDE features for portability | Deep — trades portability for IDE features |
| **Multi-model** | Cross-provider subagents (different models per agent) | Different models per mode (same conversation) |
| **Team/Enterprise** | Self-hosted daemon, open protocol | Roo Code Cloud, SOC 2, GitHub App integration |
| **User experience** | Power-user terminal workflow | Visual, editor-integrated workflow |

---

## 12. When to Choose Which

**Choose Jaato when:**
- You need a headless/CI-compatible AI tool orchestrator
- You work across multiple terminals or need multi-client sessions
- You need interactive shell sessions (REPLs, debuggers, SSH)
- Token cost optimization is critical (deferred loading, sophisticated GC)
- You need cross-provider subagent orchestration
- You prefer terminal-based workflows

**Choose Roo Code when:**
- You work primarily in VS Code and want deep editor integration
- You want zero-config setup with broad provider support
- You need browser automation as part of your workflow
- You want a community ecosystem for agent configurations (Mode Gallery)
- You need team features and enterprise compliance (Roo Cloud, SOC 2)
- You prefer a visual, chat-based interaction model
