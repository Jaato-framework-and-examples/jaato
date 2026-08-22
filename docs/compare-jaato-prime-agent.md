# Jaato vs. Prime Agent — Feature Comparison

> Compared against `PrimeIntellect-ai/prime-agent` @ `e319a66` (v0.8.0, 2026-08-21)
> and jaato `0.7.0` / jaato-premium `main` @ `a57fcf2`.

## Overview

|  | **jaato (+ premium)** | **Prime Agent** |
|---|---|---|
| **Tagline** | "Just another agentic tool orchestrator" | "A Self-Improving RLM Agent" |
| **Creator** | apanoia (independent) | Prime Intellect (decentralised-training / RL lab) |
| **Language** | Python (TUI, server, SDK); TypeScript SDK + React web client | TypeScript / Node ≥22.8 (+ a small Python runtime package) |
| **License** | BSL 1.1 (Change Date 2030-09-01 → Apache-2.0); premium is All Rights Reserved | MIT |
| **Lineage** | Built from scratch | Fork/derivative of [`earendil-works/pi`](https://github.com/earendil-works/pi) (MIT, Mario Zechner) |
| **Size** | ~486k LoC Python (jaato) + ~44k (premium) | ~360k LoC TS + ~4.4k LoC Python runtime |
| **Tests** | 596 Python test modules + 62 premium | 440 `*.test.ts` files |
| **Architecture** | Daemon + per-session confined **runner** subprocesses; IPC/WebSocket multi-client | Daemon supervisor + per-root-session **worker** processes; local socket, single-user |
| **Model-facing tool surface** | ~60 plugins → many typed tools (with deferred loading) | **One** built-in tool: `ipython` |
| **Distribution** | `pip install -e` from source; Docker compose | `curl … install.sh` → signed versioned binary release |

**Key philosophical difference.** Prime Agent collapses the entire tool surface
into *one* tool — a persistent IPython kernel — and makes everything else
(file edits, shell, subagents, MCP, skills, web search) a **Python call inside
that kernel**. jaato does the opposite: a broad, typed, permission-gated tool
catalogue delivered by a plugin registry, with the notebook/REPL as *one plugin
among many*. Prime Agent optimises for a model that can *program*; jaato
optimises for an operator who must *govern* what the model does.

The second axis is trust. Prime Agent states plainly and repeatedly that its
process boundaries are **not** a security sandbox — workers and kernels run with
the user's full OS permissions and there is no built-in approval gate. jaato's
entire per-session confined-runner arc (AppArmor, cgroups, egress allowlist
proxy, sandbox manager, permission evaluators) exists precisely to make that
boundary kernel-enforced. These are not competing implementations of the same
idea; they are different products.

---

## 1. Execution Model

| Capability | **jaato** | **Prime Agent** |
|---|---|---|
| Primary model tool surface | Typed schemas per plugin (`cli`, `file_edit`, `filesystem_query`, `ast_search`, `web_search`, `notebook`, …) | `ipython` only (`--tools ipython`; built-in tool list is literally one entry) |
| Persistent code state across turns | Yes — `notebook` plugin (local Jupyter / subprocess kernel / Kaggle GPU backend) | Yes — the core design; one IPython kernel per session, survives compaction and kernel restart via snapshot |
| Shell | `cli` plugin (subprocess) + `interactive_shell` (real PTY: REPLs, SSH, `gdb`, wizards, password prompts) | `%%bash` magics inside the kernel (subshell per cell); `bash` tool exists in source but is not in the default tool set |
| Tools callable *from code* | Yes — `notebook` **Tool Bindings**: `generate_tools_module()` injects a `tools` module so cells call `tools.web_search(...)` through the normal executor (permissions still apply) | Yes — but inverted: code *is* the tool surface; skills are importable Python packages |
| Parallel tool execution | Yes — thread pool, 8 concurrent per turn (`JAATO_PARALLEL_TOOLS`) | Kernel executions are serialised (one namespace); parallelism comes from spawning child agents |
| Deferred / progressive tool loading | Yes — `core` vs `discoverable` discoverability, `list_tools()` → `get_tool_schemas()` | N/A (one tool); skills use progressive disclosure — only metadata in the prompt, `SKILL.md` loaded on demand |
| Prose-emulated tool calls for weak models | Yes — `prose_tool_calls` quirk on every OpenAI-compat provider + unconditional on `chrome_ai` | No — the single-tool design sidesteps it; small models must still emit a valid `ipython` call |

**Verdict.** Prime Agent's RLM model is genuinely elegant and reduces
schema-bloat in the prompt to near zero. It also assumes a model strong enough
to write correct Python for every operation, and it forfeits per-operation
governance: you cannot allow `read` but deny `rm` when both are `ipython`.
jaato's `notebook` plugin already delivers the "call tools from code" half of
the RLM idea while keeping every call routed through the permission pipeline —
that is the architecturally interesting overlap.

---

## 2. Model Providers

| | **jaato** | **Prime Agent** |
|---|---|---|
| Provider plugins / adapters | 19 (`anthropic`, `google_genai`, `antigravity`, `claude_cli`, `github_models`, `ollama`, `chrome_ai`, `lmstudio`, `nim`, `tensorrt_llm`, `triton`, `vllm`, `openrouter`, `nebius`, `ovhcloud`, `doubleword`, `zhipuai`, `zhipuai_openai`, plus the `echo` test double and a shared `_openai_compat` base) | 32 provider entries over 5 wire APIs (OpenAI Completions, OpenAI Responses, Anthropic Messages, Google Generative AI, Bedrock Converse) |
| Baked-in model catalogue | No — context window / modalities auto-detected from each provider's `/v1/models`, with profile knob + fail-loud fallback | Yes — `models.generated.ts`, ~22k lines, regenerated per release (ids, pricing, context windows, modalities) |
| Subscription auth (no API credits) | Claude Pro/Max (PKCE OAuth), Claude CLI wrapper, Antigravity (Google OAuth), GitHub Models device-code | ChatGPT Plus/Pro (Codex), Claude Pro/Max, GitHub Copilot |
| Self-hosted GPU inference | vLLM, TensorRT-LLM, Triton, Ollama, LM Studio — each a first-class plugin with load-control / catalog knobs | Via `models.json` custom providers (Ollama, LM Studio, vLLM "or anything speaking a supported API") |
| Browser on-device model | `chrome_ai` (Gemini Nano over CDP, zero cost) | No |
| Capability conformance | **`PROVIDER_CAPABILITIES` matrix + CI guard** asserting wire-level behaviour (images, PDF, `tool_choice`, thinking, caching, streaming, cancellation) | No equivalent; model metadata is declarative in the generated catalogue |
| Provider extensibility | Write a provider plugin (Python) | `models.json` for API-compatible endpoints; a TS **extension** for custom APIs/OAuth |
| Per-model quirks | Yes — declared `quirks:` in a profile, honoured per provider (`prose_tool_calls`, `coerce_typed_tool_args`, …) | Thinking-level maps and per-model overrides in `models.json` |

**Verdict.** Prime Agent has more *named* providers and ships pricing/context
metadata for all of them out of the box — that is real day-one convenience, and
its Cloudflare AI Gateway / Vercel AI Gateway / Bedrock coverage is broader than
jaato's. jaato has deeper *per-provider* engineering: catalog-driven context
detection, a wire-level capability conformance guard in CI, LM Studio load
control, per-model quirks, prose-emulated tool calling, and providers Prime
Agent has nothing for (Antigravity, Chrome built-in AI, Nebius, OVHcloud,
Doubleword, TensorRT-LLM, Triton).

---

## 3. Subagents and Multi-Agent Orchestration

| Capability | **jaato** | **Prime Agent** |
|---|---|---|
| Spawn mechanism | `spawn_subagent` tool (+ `send_to_subagent`, `close_subagent`, `cancel_subagent`, `list_active_subagents`) | `await rlm("prompt", name=…)` inside the kernel — a Python call |
| Resource model | Shared `JaatoRuntime` (provider configs, registry, permissions, ledger), isolated `JaatoSession` per agent | Child `AgentSession` per call; inherits parent model, provider, skills, tools, retry policy, resource loader |
| Result delivery | Structured completion payloads with **`completion_payload_schema`** (typed, validated handoffs) | **Never returned by the call.** `rlm()` returns an admission handle only; results arrive via explicit `agent_message` replies or files |
| Recursion depth | Configurable; cascades are a first-class pattern | Default max depth **1** (root → children, no grandchildren) unless raised |
| Agent-to-agent messaging | Parent-bridged; `telepathy` plugin (`share_context`) pushes structured context child→parent; reactors route completions | First-class: `agent_message.send(..., receiver_role="parent"/"child"/"sibling", mode="auto"/"steer"/"follow_up")`, plus `prime-agent send <agent> "…"` from the shell and a family roster |
| Cross-machine delegation | **Yes** — jaato-premium gossip clustering: peer discovery, health, remote subagent delegation on peer servers | No — all workers are local to one host |
| Profiles / specialisation | YAML/JSON `SubagentProfile` (model, provider, plugins, plugin_configs, GC, quirks, instruction suppression, inheritance) | Child inherits the parent; only `name`, `model`, `thinking` are overridable per spawn |
| Usage attribution | Shared `TokenLedger` aggregated across agents | Child usage folded into the parent assistant turn; `child_usage_attributed` entries reconcile the context tree |
| Registry durability | Session persistence + reactor-driven respawn | Parent-scoped registry survives compaction, kernel restart, and parent restore; completed daemon children are rehydrated and remain addressable |
| Warm-start | **Pre-warm runner pool** (fork-from-template, ~30s → ~7s per session) | Kernel is lazily provisioned; no pool |

**Verdict.** A genuine split. Prime Agent's messaging layer is better: named
agents, sibling/parent/child roles, three delivery modes with receipts, a
CLI `send` verb, and rate/size limits enforced by the daemon. jaato's
*configuration* layer is better: profiles let each child differ in model,
provider, plugin set, GC strategy and permissions, and typed completion
payload schemas make handoffs machine-checkable rather than prose. jaato is
also the only one of the two that delegates across hosts.

Note Prime Agent's deliberate constraint — `rlm()` never returns the child's
answer — which forces an explicit message-passing discipline and keeps the
parent's context small. jaato's cascade design reaches the same goal through
reactors and typed payloads instead.

---

## 4. Long-Running / Autonomous Operation

This is Prime Agent's headline area, and where jaato has the clearest gaps.

| Capability | **jaato** | **Prime Agent** |
|---|---|---|
| Detach / reattach a live session | Yes — daemon + IPC/WS clients; session survives client disconnect; `IPCRecoveryClient` reconnect | Yes — `prime-agent attach <agent>`, `list`, `stop`, `rename`; closing the TUI only detaches |
| Cron / one-time scheduling | **No built-in scheduler** | Yes — `prime-agent schedule add <agent> "0 9 * * 1-5" -- "…"`, persisted per session, ticks claimed before delivery so a crash cannot replay |
| Recurring self-prompt (heartbeat) | No | Yes — `/heartbeat every 10m …` (user-owned) *and* `rlm_heartbeat.create(...)` (agent-owned, multiple, labelled, pausable) |
| Persistent goal across turns | Partial — `todo`/plan plugin tracks steps; no host-enforced objective loop | Yes — `/goal`, `--goal-token-budget`, state records tokens/elapsed/continuations; only `goal.complete()` ends it |
| Bounded autonomous continuation | Partial — reactors can inject prompts on events; premium auto-steering re-injects hints every N turns | Yes — `/autonomous`, with `--autonomous-gate "npm run check"`, gate retries/timeouts, and hard limits on continuations / turns / tokens / wall-clock |
| Quality gates before "done" | Reliability policies + completion-payload validation + premium validator profiles | Shell gate commands that must pass before the run may finish; failed-gate output is fed back for repair |
| External event wake-up | **Yes** — daemon-tier wake ingress (Ed25519/RSA-signed bodies, replay window, per-session trust keys) + `webhook` plugin (HMAC, mTLS, IP allowlist, rate limit) | No inbound HTTP listener; external triggers must go through the CLI or a custom extension |
| Event-driven autonomy | **Yes** — reactor engine (premium): JMESPath match on the session event bus → action scripts → spawn/fork/inject/webhook, hot-reloaded | No equivalent; extensions can subscribe to events in-process but there is no declarative rules file |
| Background tool execution | Yes — `background` plugin auto-backgrounds long tool calls | Detached bash trees are tracked and reaped by the daemon; no model-facing background API |

**Verdict.** Prime Agent wins the *unattended-operation* column outright.
Goals, heartbeats, cron schedules, and gate-checked autonomous mode are four
distinct, documented, host-enforced mechanisms with explicit budgets — jaato
has no direct equivalent for any of them. jaato answers the same need from the
other direction: signed external wakes and a declarative reactor rules engine,
which is stronger for *event-driven* pipelines and weaker for *time-driven* or
*self-driven* ones.

**Concrete borrow candidates for jaato:** (1) a per-session persisted scheduler
(`scheduled-jobs.json` with claim-before-deliver semantics and tick
coalescing); (2) host-enforced continuation budgets with shell quality gates;
(3) agent-owned recurring prompts distinct from user-owned ones.

---

## 5. Context Management

| Capability | **jaato** | **Prime Agent** |
|---|---|---|
| Strategies | Four pluggable GC plugins: `gc_truncate`, `gc_summarize`, `gc_hybrid`, `gc_budget` | One: LLM summarisation compaction (+ branch summarisation for `/tree`) |
| Policy model | `gc_budget` five-tier priority (ENRICHMENT → EPHEMERAL → PARTIAL → PRESERVABLE → LOCKED) with continuous per-turn collection | Token-threshold cut point; `keepRecentTokens` (20k default), `reserveTokens` (16k default) |
| Trigger | Proactive threshold monitoring **during streaming**, pre-send checks, post-collection budget resync | `contextTokens > contextWindow - reserveTokens`, or `/compact [instructions]` |
| Turn-boundary handling | Priority-tier aware | Explicit split-turn handling: two summaries (history + turn prefix) merged when one turn exceeds the budget |
| User-steered summarisation | Per-strategy config | `/compact focus on the auth refactor…` — instructions persisted on the `CompactionEntry` and shown in the TUI |
| Cache-aware | Yes — `cache_anthropic` / `cache_google_genai` / `cache_zhipuai` plugins, cache-aware GC design | Provider-level prompt caching; `PI_CACHE_RETENTION=long` |
| Custom compaction | Configure or write a GC plugin | Extension can replace compaction entirely (`CompactionEntry.details` is free-form JSON) |
| State that survives compaction | Conversation only | **Kernel namespace survives** — variables, imports, handles remain valid after compaction |

**Verdict.** jaato's GC system is more sophisticated as a *policy engine*;
Prime Agent's is more sophisticated as a *summariser* (split turns, iterative
summaries, user instructions) and benefits structurally from the kernel: it can
throw away conversation aggressively because the *work* lives in Python
variables, not in the transcript. The `/compact <instructions>` affordance is a
cheap, high-value borrow.

---

## 6. Security, Permissions, and Isolation

This is jaato's decisive advantage, and Prime Agent does not contest it — the
docs say so in five separate places.

| Capability | **jaato** | **Prime Agent** |
|---|---|---|
| Built-in permission gate | **Yes** — every tool call routed through the permission plugin | **No.** `permission-gate.ts` / `confirm-destructive.ts` are *example extensions* you must install |
| Approval scopes | 8: `once`, `yes`, `no`, `turn`, `idle`, `always`, `never`, `all`, with a defined widening hierarchy | N/A |
| Declarative policy | `permissions.json` whitelist/blacklist with deterministic precedence (blacklist > whitelist, session > static) | N/A |
| Programmatic policy | **Permission evaluators** — Python scripts returning `ALLOW`/`DENY`/`FALLBACK`, with argument/time/history/environment awareness | Extension `on("tool_call")` handler can block |
| Kernel-enforced confinement | **Yes** — per-session AppArmor profile; the runner self-confines in bootstrap so every tool subprocess inherits it. `JAATO_REQUIRE_APPARMOR` makes it mandatory | **Explicitly no.** "Workers and kernels are separate processes for lifecycle and failure containment, **not** security sandboxes" |
| Resource limits | cgroup v2 per session (memory, pids, cpu) under `JAATO_CGROUPS_ROOT` | None |
| Network egress control | **Yes** — per-session CONNECT allowlist proxy (deny-by-default hostname/port policy, injected via `HTTPS_PROXY`) | None |
| Filesystem policy at runtime | `sandbox_manager` three-tier paths (global / workspace / session) with `sandbox add|remove|list` | Kernel runs with full user permissions |
| Code-execution safety | `notebook` **fails closed** unless AppArmor is active; `JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC` is an explicit opt-out; `CodeAnalyzer` risk classification | Documented warning: "use a disposable clone"; run untrusted work in an external sandbox |
| Prompt-injection boundary | Untrusted-content security layer in base instructions, retained even when other layers are suppressed | Not a documented framework concern |
| Multi-tenancy | Workspace provisioning + per-session confined runners + WS bearer auth (`--ws-token`, SHA-256 + `compare_digest`), socket mode control | Single OS user; per-worker tokens are process coordination, not authorization |
| Secret handling | Premium secret resolvers: Vault, AWS SM, SOPS, `pass`, OS keyring, Infisical | `auth.json` (0600) with `!command` / env-var / literal key resolution |
| PII handling | Premium session-scoped pseudonymization (AEAD-at-rest placeholder tables, sealed-box audit spans) | None |

**Verdict.** If the agent runs on a developer's own laptop against their own
repo, Prime Agent's stance is a defensible trade — fewer prompts, faster work,
and the user can review the diff. If the agent runs on shared infrastructure,
touches customer data, or is driven by anyone other than its operator, jaato is
in a different category. There is no configuration of Prime Agent that closes
this gap; it would need to be built.

---

## 7. Sessions and Persistence

| Capability | **jaato** | **Prime Agent** |
|---|---|---|
| Format | Session plugin, `.jaato/sessions/`; JSONL event log | JSONL transcript + `session-artifacts/<id>/` (kernel snapshot, scheduled jobs, harness state, child dirs) |
| Resume | Yes — session plugin + `IPCRecoveryClient` | `-c/--continue`, `-r/--resume [path|id]`, `--fork`, `--no-session` |
| Branching | `waypoint` (mark/restore file state), `rewind-with-hint` design | `/tree` navigation with branch summarisation, `/fork`, `/clone` |
| Concurrency safety | Session-scoped workspaces; workspace lease/index | Process-safe lease per canonical JSONL path; concurrent open returns `session_already_active` |
| Crash recovery | Runner respawn, pool replenishment, template watchdog | Command journal keyed `clientId+commandId`; uncertain mutations reported, never replayed; recovery marker appended to the transcript |
| Export / share | Bundle plugin (composite pack/unpack across references, agents, profiles, services) | `session export <file> [output]` → HTML; opt-in trace upload to Prime Intellect |
| Multi-client on one session | **Yes** — multiple IPC/WS clients attach to the same session concurrently | Yes — supervisor fans out to attached clients with generation-aware cursors, snapshot streaming, per-attachment backpressure |

Prime Agent's daemon protocol work here is genuinely strong: `{generation,
sequence}` event cursors, begin/chunk/end snapshot streaming at 512 KiB, file-backed
transcript caches above 4 MiB, idempotency journals, and two-phase coordinated
updates. jaato's IPC recovery story is comparable in intent but less formalised
in one place; the generation-aware cursor and the idempotency journal are worth
studying against `docs/ipc-recovery.md`.

---

## 8. Extensibility

| Surface | **jaato** | **Prime Agent** |
|---|---|---|
| Primary unit | **Plugin** (Python) — four kinds: tool, enrichment, GC, model-provider; auto-discovered and auto-wired | **Extension** (TypeScript) — event subscriber that can register tools, commands, providers, UI |
| Registration | `PluginRegistry.discover()`; `set_plugin_registry` / `set_session` / `set_workspace_path` auto-wiring | `pi.on()`, `pi.registerTool()`, `pi.registerCommand()`, `pi.sendMessage()`, `pi.appendEntry()`; hot-reload with `/reload` |
| Custom UI from an extension | Client-side: web components (`<jaato-task>`, `<jaato-profile>`), host-provided tools | **Yes** — full TUI components with keyboard input via `ctx.ui.custom()`, overlays, custom footers/headers/editors |
| Packaged capability units | Profiles, agents (personas), references/knowledge bundles, prompt library, reactors | **Skills** (Agent Skills standard + Python-backed superset), prompt templates, themes, packages |
| Skills interop | Prompt library / references | Reads Claude Code and Codex skill directories directly (`~/.claude/skills`, `~/.codex/skills`) |
| Package manager | pip / entry points (`jaato.extensions`, `jaato.scaffold_verbs`, `jaato.embedding`) | `prime-agent package install <npm|git|path>` with gallery metadata and convention directories |
| MCP | First-class: `.mcp.json`, `MCPClientManager`, multi-server, `call_tool_auto()`; MCP tools appear as normal permission-gated tools | Deliberately **not** tools — each MCP server is a Python-backed skill imported in the kernel (`await linear.list_issues(...)`); built-in Linear/Notion integrations gated by OAuth login |
| Daemon extension points | Documented (`docs/design/daemon-extensions.md`): session hooks, WS interceptors, custom aspects, remote handlers | Supervisor is internal infrastructure; extensions attach at the session/agent layer |
| Editor / IDE protocol | No ACP; WS + web components instead | **ACP mode** (Zed and other ACP clients), plus JSON and RPC modes |

**Verdict.** Prime Agent's extension API is the richer *client-side* surface —
being able to render arbitrary TUI overlays and register tools at runtime from a
single TS file is a real productivity difference, and the ACP/RPC/JSON triple
gives it editor integration jaato lacks. jaato's plugin system is the richer
*runtime* surface — four plugin kinds, enrichment pipelines, tool traits, GC
strategies, and provider plugins are all things a Prime Agent extension cannot
express.

Prime Agent's MCP decision is worth flagging: routing MCP through the kernel
means MCP tool calls are **not individually gated or schema-surfaced** — the
model discovers tools at runtime with `list_tools()` and calls them as Python.
jaato treats MCP tools as first-class permission-gated tools.

---

## 9. Knowledge, Memory, and Self-Improvement

| Capability | **jaato (+ premium)** | **Prime Agent** |
|---|---|---|
| Persistent memory | `memory` plugin — two-phase (enrichment hints → model-driven retrieval), `.jaato/memories.jsonl` | Harness memories in `harness_state.json` (session-local) / `~/.prime/agent/harness/` (global) |
| Curated knowledge | `references` plugin + knowledge bundles + semantic matching (premium embeddings via sentence-transformers) | `SKILL.md` progressive disclosure; reference docs loaded on demand |
| Instruction layering | `.jaato/instructions/` base + `.jaato/agents/<name>.md` personas + plugin instructions + framework constants + security boundary, each independently suppressible via `suppress_base_instructions` | Immutable base system prompt + `AGENTS.md`/`CLAUDE.md` context files + `--append-system-prompt` |
| Self-modifying harness | Premium auto-steering re-injects drift hints; `finetuner-closed-loop` design (assessor → applies reliability rules to profiles) is **designed, not built** | **`/refine`** — reviews the trajectory and applies small evidence-backed create/update/delete edits to supplemental prompts, memories, skill descriptions, and subagent specs, with before/after snapshots for rollback. Base prompt stays immutable |
| Drift detection | Premium `drift_monitor` reactor — embedding similarity of turn text vs. active plan-step goal, emits `drift.measured`, injects nudges | None |
| Skill authoring by the agent | Scaffold verbs (`jaato-scaffold compile`, Daruma invariant compiler → profiles, evaluators, processors, reactors, host tools, emit-then-validate) | Built-in `skill-creator` skill: the agent packages recurring workflows into markdown or Python-backed skills |
| Training-data flywheel | Premium `modlog_training_pipeline`; fine-tuner closed loop (design) | Opt-in trace upload to Prime Intellect (`PRIME_AGENT_TRACES_API_KEY`), feeding the same org's `verifiers` / `prime-rl` stack |

**Verdict.** The "Continual Harness" is Prime Agent's most distinctive idea and
it is *shipped*, not designed: durable supplemental prompt state that the agent
refines with reviewable, rollback-able edits, explicitly walled off from the
immutable base prompt. jaato's equivalents are split across memory,
auto-steering, drift monitoring and the (unbuilt) fine-tuner loop — richer in
aggregate, but no single reviewable ledger with rollback. The refinement-ledger
shape is the strongest single borrow candidate in this whole comparison.

Conversely, jaato's instruction-layering model is far more precise than Prime
Agent's, and `drift_monitor` measures something Prime Agent does not measure at
all.

---

## 10. Observability

| Capability | **jaato** | **Prime Agent** |
|---|---|---|
| Tracing | OpenTelemetry with **OpenInference** semantic conventions; span tree `jaato.turn → jaato.tool → jaato.permission`; renders natively in Phoenix / Langfuse | Pseudonymous product analytics to `api.primeintellect.ai` (agent started / command used / run completed / session ended) |
| Cost attribution | Per-call `gen_ai.usage.cost` / `llm.cost.total`, resolved provider-reported → `.jaato/pricing.json` → none | Per-model pricing baked into the model catalogue; child usage attributed to the parent turn; context-tree reporting reconciles own vs. aggregate |
| Token accounting | `TokenLedger` (JSONL via `LEDGER_PATH`), rate-limit retries, budget panel in the TUI | Usage on each assistant message; `/context` tree view |
| Session logs | `JAATO_SESSION_LOG_DIR` per-session logs; env report; health check endpoint | Rotating logs; `prime-agent status` / `doctor [--fix]` |
| Backend integrations | Any OTLP collector; Langfuse auto-selected from keys | Prime Intellect analytics + optional trace sharing |

**Verdict.** Not close. jaato is instrumented for production observability
(vendor-neutral OTel with a recognised semantic convention); Prime Agent is
instrumented for product analytics and RL data collection. If you need agent
traces in your existing Phoenix/Langfuse/Datadog pipeline, Prime Agent has
nothing to offer today.

---

## 11. Clients and Interfaces

| Surface | **jaato** | **Prime Agent** |
|---|---|---|
| Terminal UI | `rich_client` (prompt_toolkit): themes, keybindings, panes, plan panel, workspace panel, budget panel, search, external editor, per-extension openers, vision capture | Rich TUI with themes, keybindings, overlays, `/tree` browser, agents view, extension-rendered components |
| Web | React web client + premium cluster dashboard + `<jaato-task>` / `<jaato-profile>` web components | None |
| Remote access | **WebSocket with bearer auth** (token file, SHA-256 digest, `?token=` for browsers) — remote clients are a first-class transport | Local Unix socket only; remote use means SSH/tmux |
| Headless | Headless mode + JSON/event protocol over IPC/WS; `cascade` driver clients | `-p/--print`, `--mode json`, `--mode rpc` (LF-delimited JSONL), piped stdin |
| Editor integration | None | **ACP** (Zed etc.) |
| SDKs | Python SDK (`jaato-sdk`) + TypeScript SDK (`jaato-sdk-ts`) | TypeScript SDK (`createAgentSession`, `AgentSessionRuntime`, run modes) |
| Client-side tools | **Host-provided tools** — a WS client registers tools the daemon routes back to it (browser DOM, screen capture, user interaction) | Extensions run in-process; no remote tool-execution boundary |
| Presentation awareness | **`PresentationContext`** — the model is told content width, table/image/expandable support and client type, and adapts its output | Terminal-only assumptions |

**Verdict.** jaato is built as a *service* with many client shapes; Prime Agent
is built as a *terminal program* with excellent headless modes. If you need a
browser UI, a remote agent, or tools executing on the client, jaato is the only
option. If you need Zed integration or an RPC-driven local automation, Prime
Agent is ahead.

---

## 12. Summary Matrix

| Dimension | Winner | Margin |
|---|---|---|
| Tool-surface minimalism / programmatic composition | **Prime Agent** | Decisive — the RLM design is the product |
| Tool breadth and typed schemas | **jaato** | Large — ~60 plugins vs. 1 tool |
| Permissions and policy | **jaato** | Total — Prime Agent has none built in |
| Kernel-enforced isolation (AppArmor / cgroups / egress) | **jaato** | Total — Prime Agent explicitly disclaims it |
| Unattended operation (goals, schedules, heartbeats, gates) | **Prime Agent** | Large — jaato has no scheduler or continuation budget |
| Event-driven autonomy (reactors, signed wakes, webhooks) | **jaato** | Large — Prime Agent has no declarative rules engine |
| Subagent messaging ergonomics | **Prime Agent** | Moderate — roles, modes, receipts, CLI verb |
| Subagent configurability (per-child model/plugins/GC/permissions) | **jaato** | Large |
| Cross-host / clustered agents | **jaato** | Total (premium gossip) |
| Context GC as a policy engine | **jaato** | Moderate |
| Compaction quality (split turns, steered summaries, kernel survival) | **Prime Agent** | Moderate |
| Self-improving harness with reviewable rollback | **Prime Agent** | Large — `/refine` is shipped; jaato's equivalent is designed |
| Instruction layering precision | **jaato** | Moderate |
| Model-provider depth (capability guard, quirks, catalog detection) | **jaato** | Moderate |
| Model-provider breadth + baked-in pricing metadata | **Prime Agent** | Moderate |
| Observability (OTel / OpenInference) | **jaato** | Total |
| Remote + web + multi-client access | **jaato** | Total |
| Editor integration (ACP) and local RPC automation | **Prime Agent** | Total |
| Daemon protocol formalism (cursors, snapshots, idempotency) | **Prime Agent** | Moderate |
| Extension-authored TUI components | **Prime Agent** | Large |
| Install / first-run experience | **Prime Agent** | Large — one-line signed binary install |
| Openness | **Prime Agent** | MIT vs. BSL 1.1 + closed premium |

---

## 13. What Each Should Steal

**jaato → from Prime Agent**

1. **Persisted per-session scheduler** — cron + one-shot prompts with
   claim-before-deliver ticks and coalescing. Closes jaato's biggest
   long-running gap and composes cleanly with the existing reactor engine
   (`schedule.fired` becomes just another bus event).
2. **Host-enforced continuation budgets with shell quality gates** —
   `--autonomous-gate "pytest"` plus hard continuation/turn/token/time limits.
   jaato has the reliability primitives; it lacks the loop that enforces them.
3. **A single reviewable refinement ledger** with before/after snapshots and
   rollback, sitting above memory / auto-steering / references rather than
   beside them.
4. **`/compact <instructions>`** — user-steered summarisation focus, persisted
   on the compaction record. Cheap to add to `gc_summarize` / `gc_hybrid`.
5. **Agent-to-agent messaging ergonomics** — named agents, sibling addressing,
   `steer` / `follow_up` / `auto` delivery modes with receipts, and a CLI
   `send` verb.
6. **Split-turn compaction** — jaato's GC assumes turn boundaries; a single
   oversized turn is a real failure mode.
7. **Reading foreign skill directories** (`~/.claude/skills`, `~/.codex/skills`)
   into the prompt library.

**Prime Agent → from jaato** (for context on where the moat is)

Permission scopes and evaluators, kernel-enforced per-session confinement,
egress allowlisting, OpenTelemetry/OpenInference tracing, per-child profiles,
a provider capability conformance guard, remote/WS multi-client access, and
presentation-aware output. None of these are extension-shaped; they are
architecture.

---

## 14. Positioning

They are not really competitors.

- **Prime Agent** is a *single-user research and coding agent* optimised for
  long unattended runs on a trusted machine, with a research agenda (RLM +
  continual harness) and an RL data flywheel behind it. Its trust model is
  "you own the machine and you review the diff."
- **jaato** is *agent infrastructure* — a governed, observable, multi-client,
  multi-tenant orchestration server whose distinguishing work is in the parts
  Prime Agent explicitly declines to build: permissions, confinement, egress
  control, tracing, and per-agent configuration.

The interesting overlap is jaato's `notebook` plugin with Tool Bindings: it
already offers RLM-style programmatic tool composition *inside* the permission
pipeline. Investing there — plus a scheduler and a refinement ledger — would
close most of the substantive gap without adopting Prime Agent's trust model.
