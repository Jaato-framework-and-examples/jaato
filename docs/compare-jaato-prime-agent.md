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

### 3.1 Agent messaging, in detail

One of the two places (with ACP, §11.1) where Prime Agent is ahead on a check
rather than on first impression.

| | jaato | Prime Agent |
|---|---|---|
| Topology | parent ↔ child only | parent / **sibling** / child (`AgentFamilyRelationship`) |
| Addressing | opaque `subagent_id` from `spawn_subagent` | **names**, with availability checks — human-typeable and restart-stable |
| Delivery choice | one behaviour: *"processed at the next yield point"* | caller picks `auto` / `steer` / `follow_up` |
| Result | no delivery status | **receipt**: `deliveryStatus: delivered\|queued`, plus `deliveredAt` / `queuedAt` |
| Receiver context | — | `fromRelationship` — who the sender is *from the receiver's point of view* |
| Discovery | `list_active_subagents` — **own direct children only**: the executor filters `_active_sessions` by `owner_id = id(self._parent_session)`, so a subagent calling it sees its own children (usually none), never its parent or peers | `list_agents()` roster with `running` / `idle` / `inactive`, spanning parent, siblings and children |
| Broadcast | — | `send("all", …)` within the family roster |
| From a shell | — | `prime-agent send <agent> "…"` |
| Backpressure | — | 16 384 chars, 20 pending per session, token bucket (3 / 1 s) |

A roster does exist in jaato — but one tier up. `session.list` is a **client**
command, and `cascade_events(cid)` lets a driver observe every session stamped
with its cascade id. So an orchestrating client can enumerate and route; the
*agent* cannot see its peers. (Premium gossip adds cluster topology — reachable
peer *servers* with health, for remote spawn — which is a different axis again.)

That tier split is consistent with jaato's architecture rather than an oversight:
the driver orchestrates, and agents do not self-organise. It becomes a limitation
only for the peer-to-peer shape Prime Agent is built around, where a stage talks
directly to another stage without the driver in the loop.

Three of those matter more than the rest. **Sibling addressing**: a jaato cascade
stage that needs something from a peer must route through the parent, a reactor,
or a file. **Named addressing** is what makes a CLI verb possible at all — you
cannot type a UUID you never saw. **Delivery mode as a caller decision** is a
real semantic: a course correction wants `steer`, a data handoff wants
`follow_up`, and jaato has only the latter.

jaato is not simply behind here, though — it is typed where Prime Agent is
textual:

- **`share_context` (telepathy)** transfers *structured* findings child→parent.
  Prime Agent's agent messages are strings; it has no equivalent, and its own
  tool description tells the model to use messages for conversation and files
  for data.
- **`completion_payload_schema`** makes stage handoffs validated objects rather
  than prose.
- **`session.wake`** carries an actual trust model — Ed25519/RSA signatures over
  the raw body, per-session trust keys, replay window — where Prime Agent's
  messaging is same-OS-user trust with the daemon deriving sender identity.
- **Gossip** delegates across hosts; Prime Agent's roster is one machine.

So the split is ergonomics and topology (Prime Agent) against typing and trust
(jaato). Prime Agent optimises for a conversational mesh of peer agents; jaato
for a governed pipeline with validated handoffs. The borrow is the addressing
layer — names, roles, modes, receipts — which would sit on top of jaato's
existing routing without disturbing the typed paths.

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

**Verdict.** Prime Agent wins the *time-driven* half of this column outright
and does **not** win the rest — the four mechanisms are less novel against
jaato than their marketing implies. Sorting them by what they actually
correct:

| Failure being corrected | jaato | Prime Agent |
|---|---|---|
| **Direction** — agent works, but on the wrong thing | `drift_monitor` (cosine similarity of turn text vs. active plan-step goal embedding, `HARD_THRESHOLD = 0.30`, emits `drift.measured` + injects a nudge); `auto_steering` (open-loop hint re-injection); reliability nudges | **nothing** — Prime Agent measures no drift signal |
| **Termination** — agent says "done" and isn't | `completion_payload_schema` structural floor + `completeness`-phase processors contributing `incomplete[]` / `errors[]` to a composite `is_complete` verdict; `on_error: fail_completion` returns a `validation_failed` shape so the model retries | `--autonomous-gate "npm run check"` — shell exit code as the verdict |
| **Runaway** — agent never stops | `budget_control`: ceilings on `usd` / `tokens` / `seconds` / `tool_calls` / `turns`, plus an ordered **degradation ladder** that rebinds model tiers at percentage thresholds (brownout, not blackout) — and per-profile `max_turns` | hard limits on continuations / turns / tokens / wall-clock; stop only, no degradation |
| **Self-report** — objective survives across turns | — | `/goal`, re-presented after each turn until `goal.complete()` |
| **Time-driven re-entry** — nothing is happening and something should | **nothing** | cron schedules + user/agent heartbeats |

So on three of five rows jaato is level or ahead, and on one row (drift) Prime
Agent has no answer at all. `budget_control`'s degradation ladder in particular
is strictly richer than Prime Agent's limits, which can only halt.

The real gap is narrower than "unattended operation": it is **time-driven
re-entry**. jaato can be woken by an event (signed wake, webhook, reactor) but
not by a clock, and nothing re-enters a session on its own initiative. Two
narrower gaps sit alongside it: (a) Prime Agent's gates are *subprocesses*
with timeouts and process-tree kill, so "does the build pass" is a valid
termination verdict, whereas jaato's `completeness` phase is explicitly
constrained to cheap payload inspection (no subprocess) — a finalization-phase
validator can shell out, but that only runs at `signal_completion`; and (b)
jaato's completion gate exists only for schema-carrying agents — `signal_completion`
is hidden for terminal roots and schema-less sessions, so an ordinary
interactive session has no done-ness check at all.

### Are these "just skills"?

Worth being precise, because the model-facing surface is misleading. Two of the
four are **not skills at all**, and the other two are skill-shaped facades over
host machinery:

| Mechanism | Kernel-side skill | Host-side implementation |
|---|---|---|
| Cron / one-shot schedules | **none** | `src/core/cron-jobs.ts` — 1735 lines: `AgentCronJobStore`, `AgentCronScheduler`, cron parsing, `scheduled-jobs.json` persistence, claim-before-deliver ticks. Surfaced as the `prime-agent schedule` CLI verb |
| Autonomous mode | **none** | `src/core/autonomous.ts` — 593 lines, wired into `AgentSession` as `_autonomousState` with continuation-suppression tracking. Gate subprocesses, budget counters and continuation injection all sit *around* the model, which cannot see them |
| Goals | `skills/goal` — **52 lines** of `await host_request("goal.get")` | `src/core/goals.ts` — 290 lines owning state, persistence, token/elapsed accounting and continuation prompting; handlers at `agent-session.ts:2935` |
| Heartbeats | `skills/rlm-heartbeat` — **102 lines** of the same RPC shim | `/heartbeat` is a host slash command; `rlm_heartbeat.*` handlers at `agent-session.ts:3084` back onto the same cron store |

So ~154 lines of Python facade against ~2600 lines of TypeScript enforcement.
The bundled skills are a *typed API veneer over host RPC* — an interesting
pattern in itself (jaato's analogue would be a tool schema), but they carry no
policy.

This matters for the borrow assessment: the hard part is precisely the part
that isn't a skill — crash-safe tick claiming so an uncertain prompt is never
replayed, coalescing missed ticks instead of accumulating a backlog, budget
accounting the model cannot misreport, gate subprocess lifecycle with timeout
and process-tree kill, and injection into the session prompt queue. None of
that is reachable by writing prompt files; it is a scheduler and a policy loop
in the daemon.

### The dual completion schema: jaato's better-shaped answer

Prime Agent's `rlm_heartbeat` is a **tool the model may call at will**. jaato has
a shape available to it that Prime Agent does not, because jaato already gates
completion on an operator-authored schema: make the *suspend* a branch of the
completion payload.

```yaml
# completion_payload_schema (sketch) — discriminated on `outcome`
outcome: finished | suspended
# finished  → result fields, artifacts, …
# suspended → wake_at, continuation_note, poll handle (job id / URL), …
```

The agent then ends its turn the only way it already knows how — `signal_completion`
— but says *"paused, here is my state, wake me at T"* instead of *"done"*. The
routing half needs no new code: `LifecycleTools._execute_signal_completion`
already fires `hooks.on_agent_completed(payload=...)` with the **validated**
payload, and a reactor rule discriminates on it with an ordinary `where` clause.
Continuity is free too — `wake_session` cold-revives via `resume_session` before
driving, so the resumed agent keeps its history and the continuation note
becomes the wake text.

Why this is better than a heartbeat tool, for jaato specifically:

- **Deferral becomes a granted capability, not a discovered one.** Whether an
  agent may suspend is a property of the schema its operator authored. Prime
  Agent's model decides for itself.
- **No new model-facing surface** — no tool, no permission entry, no extra
  discoverable schema. Only a payload variant on a call the agent already makes.
- **The pause is validated and atomic.** A `completeness` processor can *require*
  that a `suspended` payload carries the handle its continuation will need; a
  half-set timer is unrepresentable. Prime Agent's heartbeat instruction is
  unvalidated free text.

It does **not** remove the store or the sweeper: a reactor action script runs on
a worker thread and can no more sleep until 09:00 than a background task can. It
changes *who writes the due-row* — a reactor action instead of a model tool —
which is a real simplification, but something must still persist and sweep it.

**Three hazards this design has to answer:**

1. **A suspend is not a completion.** `signal_completion` is "the terminal tool
   by contract" and terminates the turn loop; downstream consumers read
   `agent.completed` as *finished*. A suspend riding the same event means the
   cascade driver could advance the pipeline, budget accounting could close out
   the stage, and — most concretely — **`finalization`-phase processors would
   run and write final artifacts for work that is not done**. `phase` currently
   selects *when* a processor runs, not *conditional on which payload branch
   validated*; suppressing finalization on a suspend branch is the one piece of
   framework work this design actually requires.
2. **It makes the `DEFERRED` wrinkle mandatory to fix.** A suspended cascade
   agent is precisely "revived cold, no client attached, cid present" — the exact
   condition under which `wake_session` holds the turn pending an attach. Left
   as-is, every suspend/resume stalls.
3. **Suspend loops need a ceiling.** An agent that keeps emitting "wake me in ten
   minutes" is a runaway with no natural terminator, and unlike Prime Agent's
   autonomous mode nothing is counting continuations. This is where
   `budget_control` earns its keep: count suspend-resumes against the profile's
   `turns` / `seconds` / `usd` ceilings, and let the degradation ladder respond
   rather than only hard-stopping.

Handled, this is strictly stronger than what Prime Agent ships: a governed,
schema-validated, operator-granted pause with typed continuation state, against
Prime Agent's ungoverned free-text timer.

### Why the model sets its own timer: the no-blocking rule

The agent-owned heartbeat is not a convenience — it is the other half of a
discipline Prime Agent enforces in its system prompt (`src/core/prompts/rlm.ts`):

> For slow or independently completing work, use a nonblocking control loop:
> start the work, record its handle or output location, **then end your turn**.
> Read the result on a later turn or when a reply arrives.
>
> **Do not keep the turn open by polling with `time.sleep()` or shell `sleep`**,
> and do not replace polling with a long blocking `await`. Await only the short
> operation needed to start work or inspect a result that is already available;
> otherwise end the turn.

So the model is told never to wait inside a turn. That splits waiting into two
cases by whether the awaited work can announce itself:

- **It can notify** — an RLM child. The child sends `agent_message` on
  completion, which arrives as an ordinary message and re-enters the parent.
  No timer is involved; this is the "or when a reply arrives" branch.
- **It cannot notify** — a test run, a deployment, a training job. Nothing will
  ever send a message, so "read the result on a later turn" requires *a later
  turn to exist at all*. `rlm_heartbeat` manufactures exactly that turn.

That is the whole purpose: the heartbeat is the yield-side counterpart to the
no-blocking rule. Without it, "end your turn" applied to non-notifying work
means the work is silently abandoned until a human happens to type something.

The delivery-mode default confirms the session is *not* assumed idle in between.
`steer` (the default) interrupts an in-flight turn, because the common shape is
start the long job → set the heartbeat → end the turn → do something else → be
interrupted mid-task when the tick comes due. An idle or cold session instead
gets a fresh turn (revived from disk if necessary); `shouldDeferHeartbeatCronJob`
suppresses the tick entirely in states where neither is safe.

**The consequence for jaato is a sequencing constraint.** This prompt discipline
and the wake mechanism are a matched pair, and the mechanism must land first:
telling a model "end your turn, you will be resumed" is only safe once
resumption actually exists. jaato today has no affordance for this at all — an
agent facing a long external job must block (burning context and wall-clock),
background it (which cannot represent work jaato does not own, and dies with the
runner), or hand back to the user. Adding the guidance before the wake path
would convert every long external wait into silently dropped work.

It also re-prices the heartbeat spend noted above: ending the turn makes the
*wait* free, and moves the entire cost into one turn per tick — which is exactly
why an agent-owned timer needs a ceiling, and why routing it through
`budget_control` rather than leaving it unbounded is the better version.

### What jaato would actually have to build

Less than "a scheduler". `SessionManager.wake_session()` already **is** the hard
half, and its own docstring names the missing caller:

> The client-agnostic wake primitive (``session.wake``): any authenticated
> caller — IPC, WS, an HTTP webhook shim, **cron**, a peer — can drive a fresh
> turn on a session with NO client attached.

It revives cold/unloaded sessions from disk (`resume_session` → `send_message_to_session`),
resolves the workspace server-side from `SessionWorkspaceIndex` so an
authenticated-but-untrusted caller cannot point revival at a weaker sandbox
root, wraps the payload via `wrap_untrusted_content`, dedups on `event_id`, and
already accepts a `cascade_driver_id` so cascade observers are notified. Every
capability that makes Prime Agent's scheduler load-bearing — session revival,
crash-safe idempotency, durable targeting — exists here already.

What is missing is only **the clock**. The composition:

```
daemon-tier job store (due-time, prompt, target session, cid)
  → single timer on the nearest due time, claim-before-advance
    → wake_session(session_id, text, source="schedule", event_id=..., cascade_driver_id=...)
      → session revives → reactor sees the bus events
        → cascade agent spawns → completion schema + completeness processors gate done-ness
```

Everything from `wake_session` rightward is unchanged and already shipped. The
scheduler is a *clock-driven wake source* sitting beside the existing
*externally-driven* one: wake ingress is Stage A transport hygiene + Stage B
signature verification, and for a clock trigger Stage B collapses to nothing
because the caller is our own daemon. Reusing the `event_id` dedup claim gives
idempotent redelivery for free.

**How small is the store?** Smaller than Prime Agent's. Its `AgentCronJobStore`
carries dispatch rows because it has no idempotency layer downstream; jaato
does — `wake_session` already dedups on `event_id`. Deriving a deterministic
`event_id` from `(job_id, scheduled_for)` therefore buys crash-safe
at-most-once delivery *and* tick coalescing from machinery that already exists,
leaving the store as a plain table: job id, schedule expression, prompt, target
session, optional cid, TTL.

**Where it lives is a placement decision, not a requirement.** The only real
constraint is that the clock must outlive the *target session's runner*. That
admits three hosts, and the daemon is not the obvious winner:

| Host | Cost | Fits |
|---|---|---|
| **OS scheduler + a thin SDK client** — `crontab` / systemd timer / k8s CronJob calling `session.wake` | ~zero jaato code; inherits decades of scheduler hardening; no timer to write | operator-driven recurring prompts |
| **External always-on cascade-client** — owns its own store and timer, holds the cid, subscribes to cascade events | zero daemon code; swappable, testable, per-tenant | pipeline / cascade drivers |
| **Daemon-tier store** | new code, but in-band and lifecycle-coupled | agent-owned self-timers |

For the operator case the first row is plainly best — there is no reason to
write a scheduler when the OS ships one, and jaato's contribution reduces to a
CLI verb for ergonomics.

**What actually earns a daemon-side store is the narrow case: agent-owned
schedules** — Prime Agent's `rlm_heartbeat`, where the *model* sets its own
timer. That needs an in-band tool, storage that outlives the session, lifecycle
coupling so the schedule dies with its session, and a guarantee that session A
cannot schedule wakes into session B.

Which is precisely the shape jaato already built for `wake_binding_registry`:
daemon-owned storage under `~/.jaato/`, holding session-owned content, written
through an owner-guarded command that *"runs AS the caller's session, so a
caller can only bind ITSELF — hijack-proof by construction"*, carrying a TTL as
the safety net for a forgotten unbind, durable because *"a wake may arrive days
after the bind"*. A binding says "wake me about `wake_ref` if someone signs with
`trust_keys`"; a schedule says "wake me at time T". Same write path, same
durability rationale, same TTL, same daemon-owns-the-sandbox /
session-owns-the-invitation split. **It is another column on an existing table,
not a new subsystem.**

The one genuinely new moving part either way is the sweep: registry expiry today
is *lazy* (checked at resolve — `existing.expires_at > now`), so nothing
currently scans for due rows. That is also the one piece the OS-cron route lets
you skip entirely.

**The leg that does not work in any of the three placements is a backgrounded
"cron" task.** Three reasons, and jaato already settled the argument for itself:

1. **Wrong lifetime.** A background task runs in the runner, and the runner is
   precisely what unloads — a timer living inside the thing it is supposed to
   wake is circular. `server/wake_ingress.py` states the identical conclusion
   verbatim for the same reason: *"It lives at the DAEMON tier — NOT the
   runner-tier webhook plugin — because it must survive session unload (a wake
   can arrive days after the session went idle-detached; the runner-bound
   webhook listener would be gone)."* That argument rules the runner out; it
   does not by itself select the daemon over the two external placements above.
2. **Wrong cost.** `BackgroundCapableMixin` runs a `SafeThreadPoolExecutor` with
   `max_workers=4` and `default_timeout=300.0`. A task sleeping until 09:00
   tomorrow holds a worker slot and is killed after five minutes anyway. A
   future point in time wants a persisted due-time plus one timer — Prime
   Agent's `scheduleNext()` keeps exactly one `setTimeout` aimed at the nearest
   due job — not a parked thread per pending job.
3. **No durability.** The background plugin persists nothing, so a daemon
   restart would silently drop every pending schedule.

**One wrinkle worth designing around up front.** `wake_session` returns
`DEFERRED` when a session is revived cold with no attached client *and* a
`cascade_driver_id` is present: host (client-side) tools would have no client to
dispatch to, so it emits `SessionWokenEvent` to the cid's observers and holds
the turn until a client re-attaches. That is exactly the state a scheduled wake
into a detached cascade lands in — so the turn would wait for an attach that may
never come. `attach_session`'s comment shows the resolution is already
conditional (*"the transport layer drives the pending wake AFTER wiring the
client's tools, or immediately when the client has none"*), so the fix is to
extend that "client has none" reasoning to the target profile: a profile
declaring no host tools should drive immediately even under a cid. Without
that, scheduled cascade wakes stall.

### Isn't jaato's `background` plugin the same thing?

No — they solve adjacent problems, and being exact about the difference narrows
what actually needs building.

| | jaato `background` | Prime Agent scheduler |
|---|---|---|
| Who executes the long work | **jaato** — a `SafeThreadPoolExecutor` thread in the runner wrapping one of our own tool calls | **nobody here** — the job only injects a prompt; the real work runs elsewhere |
| Creates a new turn | **No** | **Yes** |
| Completion discovery | model polls `getBackgroundTaskStatus` / `getBackgroundTaskResult` | daemon re-enters the session unprompted |
| Survives turn end | task keeps running, but delivery does not — nothing tells the model | yes |
| Survives session unload / daemon restart | **No** — in-process threads, no persistence of any kind | yes — `scheduled-jobs.json`, lockfile + fsync, and it revives dormant sessions from disk |
| Horizon | `default_timeout = 300.0` s | unbounded |

Put plainly: **backgrounding is intra-turn concurrency; scheduling is inter-turn
re-entry.** Backgrounding lets one agent do two things at once inside a turn it
is already having. Scheduling lets an agent *exist at a moment when nothing is
happening*. Once the turn ends, backgrounding has stopped being a mechanism —
the thread may still run, but no path exists to tell anyone it finished.

Ownership matters as much as timing. `startBackgroundTask` wraps a jaato tool.
It cannot represent "the CI run finishes in 40 minutes" — pointing it at that
burns a runner thread on a sleep loop and hits the 300 s timeout anyway. Prime
Agent's scheduler never executes the external job either; it just guarantees
somebody asks about it later.

**A cheaper fix falls out of the comparison.** The background plugin's real hole
is not the missing scheduler — it is that a task completing after the turn ends
stays invisible until the user types again. It emits nothing: no bus event, no
callback. Yet `JaatoRuntime.event_bus` is already reachable from a plugin — the
`webhook` plugin does exactly this (`self._event_bus = _sess._runtime.event_bus`,
then `_publish_to_event_bus(...)`) and reactors consume it. Having background
tasks publish `background.completed` would let a reactor inject the result and
resume the agent, closing the gap for all work **jaato itself owns**, with no
new subsystem. The scheduler would then be needed only for its irreducible
case: polling external state jaato does not own and cannot be notified about.

### How Prime Agent's scheduler actually works

Worth recording, since it is the one primitive jaato lacks outright.

**Shape.** One `AgentCronJobStore` per session (`session-artifacts/<id>/scheduled-jobs.json`,
`proper-lockfile`-guarded, atomic rename + fsync) and one `AgentCronScheduler`
per worker. Three schedule kinds — `once` / `cron` / `interval`; three sources —
`cron` (user CLI), `heartbeat` (user `/heartbeat`), `rlm_heartbeat` (agent-owned);
two runtime kinds — `top-level` / `subagent`.

**Payload is a prompt string.** It is not a job runner: a due job injects text
into the session prompt queue and the model reads it on its next turn. Every
tick therefore costs a full turn.

**The load-bearing part is session revival.** `getOrCreateCronJobSession()` falls
through to `createRuntime({ type: "create", sessionPath: job.sessionFile })` —
a due job **loads a dormant session from disk and brings it back**, including
restoring an RLM subagent runtime for `rlm_heartbeat` jobs whose
`runtimeKind === "subagent"`. So a schedule is not "run while attached"; it is
"revive this session at 09:00 and hand it this instruction". That is the same
capability jaato's wake ingress provides for *external* events and nothing
provides for the clock.

**Crash safety.** `claimDueInState()` advances `nextRunAt` **at claim time**, before
dispatch, and records a dispatch row. A crash mid-dispatch therefore cannot
replay an uncertain prompt — `recoverInterruptedDispatches()` marks it
interrupted and resumes at the *next* tick. A tick landing on an
already-claimed job stamps `lastSkippedAt` instead of queueing, which is the
coalescing: a session busy for an hour accrues one pending heartbeat, not twelve.

**Delivery discipline.** Dispatches are serialised per `activeSessionId` lane.
`shouldDeferHeartbeatCronJob()` skips outright while compacting, retrying,
running bash, holding pending session work, or mid agent-message — regardless
of delivery mode. Otherwise `steer` interrupts the current turn and `follow_up`
waits for it.

**What it is for.** Three surfaces over one store: `prime-agent schedule add`
for operator-driven recurrence (`"0 9 * * 1-5" -- "Review open work"`),
`/heartbeat` for one visible user-owned poll loop, and `rlm_heartbeat.create(...)`
for agent-owned timers the model sets itself (multiple, labelled, pausable).
The animating use case is Prime Intellect's own: start a long evaluation or
training run, have the agent set a 5-minute check-back on it, detach the
terminal. It exists to poll **external state that has no callback** — a GPU
job, a CI run, a benchmark — which is exactly the case jaato's event-driven
stack cannot cover, since there is nothing to subscribe to.

The honest limit: heartbeats carry no budget of their own. An idle session with
a 5-minute `rlm_heartbeat` burns a turn every five minutes indefinitely; the
deferral rules bound *stacking*, not *spend*. Wiring the equivalent into
jaato's `budget_control` ceilings would be a strict improvement.

**Concrete borrow candidates for jaato:** (1) a per-session persisted scheduler
(`scheduled-jobs.json` with claim-before-deliver semantics and tick
coalescing) — the one genuinely missing primitive; (2) agent-owned recurring
prompts, distinct from user-owned ones, so the model can set its own check-back
timers; (3) subprocess quality gates with timeout and process-tree kill as a
termination verdict, wired to the existing `budget_control` ceilings rather
than to a new budget system; (4) extending the done-ness gate to schema-less
sessions.

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
| Self-modifying harness | Premium auto-steering re-injects drift hints; `finetuner-closed-loop` design (assessor → applies reliability rules to profiles) is **designed, not built** | **`/refine`, and it runs automatically by default** — every 25 assistant turns and on every compaction (20-min cooldown), gated by a cheap LLM reviewer before the expensive edit pass. Emits create/update/delete edits over four entry kinds (`prompt`/`memory`/`skill`/`subagent`) in two scopes (session-local default, global on explicit request). Every applied edit stores `before`/`after` entry snapshots, so rollback is a pure function of the ledger, not a second LLM call. Base prompt immutable. See §9.1 |
| Drift detection | Premium `drift_monitor` reactor — embedding similarity of turn text vs. active plan-step goal, emits `drift.measured`, injects nudges | None |
| Skill authoring by the agent | Scaffold verbs (`jaato-scaffold compile`, Daruma invariant compiler → profiles, evaluators, processors, reactors, host tools, emit-then-validate) | Built-in `skill-creator` skill: the agent packages recurring workflows into markdown or Python-backed skills |
| Training-data flywheel | **`kb-stage-agent-LoRA-training` — built and proven.** Teacher (GLM-5) runs the cascade; trajectories are harvested per stage; LoRA adapters are trained on Nebius Token Factory and served on self-hosted vLLM; each is scored on held-out specs against the teacher as gold. Plus premium `modlog_training_pipeline`. The operator trains adapters for their own cascade, on their own account | Opt-in trace upload to Prime Intellect (`/traces`, `PRIME_AGENT_TRACES_API_KEY`). The repo documents the mechanism but **states no purpose** for the uploaded traces; that they feed the same org's `verifiers` / `prime-rl` stack is a reasonable inference from the branding and sibling repos, not a documented claim. What is certain is the direction: full session traces leave the machine, and nothing returns as a model the operator owns |

### 9.0 What refinement is *for*

The mechanism only makes sense against the architecture it serves. Prime Agent
is deliberately **context-minimal**: one model tool, skills that stay
metadata-only until loaded, subagents whose answers never enter the parent's
context, and compaction that discards conversation aggressively *because the
real work lives in Python variables rather than in the transcript*.

That buys a small, cheap context — and creates the problem refinement exists to
solve: **everything the agent learns evaporates.** In an agent whose transcript
*is* its memory, a lesson survives by sitting in history. Here history is
summarised away by design, subagent context is isolated by design, and sessions
end.

So refinement is the **write-back path for an architecture that keeps throwing
context away**. Its own system prompt says so directly:

> This is similar in spirit to context compaction, but instead of summarizing
> the conversation you emit precise Create, Update, or Delete edits to reusable
> state.

It is compaction's inverse. Compaction asks *what can I discard from this
trajectory?*; refinement asks *what should I extract before it goes?* — which is
why one of its two automatic triggers is compaction itself
(`AutoRefineReason = "turn_interval" | "compact"`). It fires precisely when
context is about to be lost.

The four entry kinds are four things worth surviving that loss:

| Kind | Saves you from |
|---|---|
| `memory` | re-deriving a fact, decision or outcome, or re-asking the user a preference |
| `prompt` | the user repeating the same behavioural correction every session |
| `skill` | re-improvising a procedure already performed several times |
| `subagent` | re-describing a delegation role in prose each time it recurs |

So the intended payoff is narrow and practical: **don't re-learn, don't re-ask,
don't re-improvise.** Not "the agent gets smarter" — the model is unchanged;
only its durable notes accumulate.

Whether that payoff is achieved is unmeasured (§9.3). The guardrails — immutable
base prompt, bounded overview injection, session-local default, per-edit
rollback — bound the *damage* of a bad refinement rather than verify the
*benefit* of a good one. Read alongside the "Continual Harness" being one of
the project's two headline abstractions, refinement reads as a research bet:
that durable editable harness state is a route to improvement without touching
weights. jaato takes the other route (§9.3's layer table), which is why it needs
no equivalent as urgently — its context GC preserves by policy, its knowledge is
human-curated, and its improvement loop reaches for weights or enforced policy
instead of prompt notes.

### 9.1 How `/refine` actually works

Worth the detail, because the shape is more copyable than the idea.

**Data model.** `HarnessEntry` × 4 kinds × 3 actions × 2 scopes. Kinds are
`prompt` (supplemental notes only), `memory` (durable facts, decisions,
failures, preferences), `skill` (an installed Python REPL skill, which must
carry a `reference` object naming its import and call pattern plus an
`arguments` contract), and `subagent` (a reusable delegation spec). Entries are
versioned with `created_at` / `updated_at` / `source`. State lives in
`harness/harness_state.json` — under the session artifact directory for `local`,
under `~/.prime/agent/harness/` for `global`.

**It is automatic, not just a slash command.** `getAutoRefineSettings()` defaults
to `enabled: true`, `turnInterval: 25`, `compact: true`, `cooldownMs: 20 min`.
So refinement fires on its own every 25 assistant turns and at every compaction.
`/refine` and the kernel-side `await refine.run(...)` are manual entry points to
the same machinery.

**Two stages, deliberately asymmetric in cost.** A cheap review gate
(`AUTO_REFINE_REVIEW_SYSTEM_PROMPT`, 4 096 output tokens) returns
`{shouldRefine, rationale, instructions?}` and is told to *"reject one-off noise,
unsupported hypotheses, and transient tool outputs."* Only on approval does the
expensive pass run (`REFINEMENT_SYSTEM_PROMPT`, 32 000 output tokens) and emit
the edit proposal. You never pay for the big pass without a judge finding
evidence first.

**Rollback is mechanical, and this is the part worth stealing.** Every applied
edit persists `before` and `after` `HarnessEntry` snapshots, so
`rollbackProposal()` walks the applied edits **in reverse** and inverts each one
from its own snapshots — restore `before` where it existed, delete where only
`after` did. No model is involved in undoing a refinement; it is a pure function
of the ledger. History is appended to `refinements.jsonl`, and malformed lines
are skipped explicitly *"so a single bad append cannot break rollback."*

**Scope discipline is enforced in the prompt, not just by convention.** During a
local refinement, global entries are read-only context: the refiner is told never
to propose update or delete edits against them, and to create a local override
instead. `mergeHarnessStates()` layers global then local, disambiguating an id
collision by prefixing the key — and the prompt tells the model those
`local:` / `global:` prefixes are display-only, so edits must use the bare id.

**What reaches the prompt is an overview, not the content.** Six entries per
kind, five recent refinements, 180 characters each, framed explicitly as
*"compact summaries, not full descriptions … routing/context hints; inspect the
underlying entry only when detail matters."* Progressive disclosure applied to
the harness itself, so the standing prompt cost stays bounded however much
harness accumulates.

**Ordering guarantees.** Refinement never runs mid-cell: `refine.run()` returns
`{"scheduled": true}` immediately, the pass runs at turn end, applies edits,
**rebuilds the system prompt**, and resumes the agent automatically. A refine
started before a `/tree` branch switch is discarded via a branch-version check,
and a failed review stamps the cooldown so a persistent failure (bad auth,
unparseable output) cannot retry a full review every turn.

**The guardrails, collected:** the base system prompt is immutable and may not be
rewritten; refinement may never edit source files directly; local is the default
and global requires an explicit request; and the refiner is pushed toward the
*smallest* component that fits — memory for declarative facts, skill for
repeatable procedures, prompt for narrow behavioural policy, subagent for a
repeated delegation role.

### 9.2 Isn't `waypoint` already this?

Nearly, and it is worth being exact — jaato's `waypoint` plugin is the stronger
primitive of the two *in its own domain*, and Prime Agent has no answer to it
(`/tree` branches the conversation, not the workspace).

Waypoint captures a workspace file snapshot **plus** a
`session_state_snapshot` — JSON from `JaatoSession.get_all_session_state()`, so
registered state providers are invoked live. Waypoints form a **tree** via
`parent_id`, carry an `owner` (`user` or `model`, so the agent can mark its own),
and on restore an enrichment tells the model which files were reverted so its
context does not go stale. Prime Agent has nothing comparable.

Two differences keep it from covering the same ground:

**1. Different object.** Waypoint versions the *work product* — the workspace and
session runtime state. The refinement ledger versions the *agent's own persisted
configuration*: prompt notes, memories, skill descriptions, subagent specs.
jaato has no equivalent because it has no self-modifying configuration to
protect — `.jaato/instructions/`, references and profiles are human-authored, and
the one surface that **is** model-authored, the `memory` plugin
(`.jaato/memories.jsonl`), has no undo of any kind.

**2. Point-in-time restore vs per-change inversion.** This is the structural one.
A waypoint restores state *as of time T*; its snapshot is whole-state, so
returning to `w3` discards `w4` and `w5` with it. Refinement rollback stores
`before` / `after` snapshots **per edit**, so `rollbackProposal()` can invert
refinement B alone while A and C stand. It is the difference between
`git reset --hard <commit>` and `git revert <commit>` — and only the second is
usable when a change three edits back turns out to be the bad one.

So the borrow is narrower than "snapshot and restore", which jaato already does
better. It is: **a per-change inversion ledger over agent-authored state**, whose
natural home in jaato is the `memory` plugin (model-written, currently
irreversible) and the not-yet-built fine-tuner loop that would write reliability
rules into profiles. `register_session_state_provider` is the existing seam a
memory ledger could hang from.

### 9.3 Is refinement the auto-tune pattern?

No — and the gap is the part that defines tuning. Set the two loops side by side:

| Step | jaato `finetuner-closed-loop` (designed) | Prime Agent `/refine` (shipped) |
|---|---|---|
| **1. Signal** | a failure pattern detected from the **OpenTelemetry stream** of another running session — structured, post-hoc, cross-session | an LLM reading the current conversation trajectory in-context |
| **2. Proposal** | a validated reliability patch against the profile's `plugin_configs.reliability`, returned as a unified diff | create/update/delete edits over harness entries |
| **3. Verification** | **fork-replay the failing turn against the patched profile** | *nothing* |
| **4. Accept / reject** | apply if the failure no longer reproduces; discard and escalate to a human if it does | always applies; rollback exists but is a later, separately-initiated act |

Prime Agent has steps 1 and 2 plus an undo. It emits an `expectedOutcome` field
described in its own prompt as *"what should improve and how to validate it"* —
and nothing ever reads it back. There is no metric, no replay, no convergence
test. So refinement is **open-loop self-editing with undo**, not tuning: it never
learns whether an edit helped.

Three further differences that follow from that:

- **Reflexive vs second-party.** `/refine` edits the session it runs inside, on
  evidence it produced itself. The fine-tuner is an external observer analysing a
  *different* session's telemetry, and can fork that session read-only to
  interrogate the model about its decisions. A judge that is also the defendant
  is a weaker instrument.
- **Advice vs enforced policy.** A harness memory ("always check git status
  first") is guidance the model may ignore on any given turn. A reliability
  policy is applied by the framework at tool-call time. Same intent, entirely
  different blast radius.
- **Where it lands.** Reliability patches are written to the profile — on disk,
  diffable, reviewable in version control, travelling with the agent definition.
  Harness state lives in `harness_state.json`, session-local by default and
  outside VCS.

It is not auto-tune in the hyperparameter sense either: nothing searches a
parameter space against an objective.

**Correction: jaato's closed loop is built, at a deeper layer than the design
doc describes.** `kb-stage-agent-LoRA-training` distils a GLM-5 teacher running
the kb-enablement cascade into **per-stage LoRA adapters**, trained on Nebius and
served on self-hosted vLLM, each scored on held-out specs with the teacher as
gold — discovery 0.53 → 0.78 precision, build_judge `error_count` 0/3 → 3/3,
codegen template recall 0.10 → 0.75. The *reliability-rule-patching* variant in
`finetuner-closed-loop.md` remains unbuilt as specified, but the pattern it
describes — measure, intervene, verify, accept or reject — is running at the
weight level.

`pipeline/tool_tutor.py` is the sharpest piece: STaR / hint-augmented rejection
sampling that drives a student through a stage in isolation, validates its
structured tool call, injects Socratic error-grounded hints until it emits a
schema-conforming call, and harvests the corrected trajectory as training data.
Its second role turned out to matter more than its first — as a **diagnostic**
it adds real cascade conditions back one variable at a time to localise a
failure across model / prompt / serving / wiring. A 2×2 factorial
(`pipeline/twobytwo.py`) isolated an *interaction*: the failure needed **both**
sampling **and** the real conditions (~42K of tool-doc bloat plus the
`prepare_completion` accumulator), neither alone. That reassigned the blame from
"weak students" to two framework bugs — vllm temperature threading (#381) and the
accumulator's `floor_met` (#386). A training tool that proved training was not
what was needed, and a README that records having falsified its own earlier
conclusion.

So the two projects sit at different layers, and only one of them closes:

| Intervention layer | jaato | Prime Agent |
|---|---|---|
| Prompt notes / memories | `memory`, `auto_steering`, `references` | `/refine` — automatic, but open-loop |
| Declarative policy in profiles | `finetuner-closed-loop` (designed; fork-replay verification) | — |
| **Model weights** | **`kb-stage-agent-LoRA-training` (built, measured)** | — |

What remains worth taking from Prime Agent is narrower than "the refinement
pattern": the **plumbing for continuous, cheap self-editing** — an automatic
cadence, a cheap review gate in front of an expensive writer, a per-edit
inversion ledger, bounded overview injection. jaato's loops are rigorous and
operator-driven; Prime Agent's is shallow but runs unattended every 25 turns.
Neither project currently has both.

### 9.4 The closest jaato analogue: the memory curator

Refinement's purpose (§9.0) is jaato's **memory curator** pattern
(`docs/design/agent-continuity.md`), and the two are close enough that the
differences are the interesting part.

Both extract durable lessons from a finished stretch of work, both gate what is
allowed to persist, and both re-inject the result as **bounded hints** rather
than full content — jaato's tag-coherent `💡 Available Memories`, Prime Agent's
6-per-kind / 180-character overview. Both fire automatically.

Where jaato's is the better-designed of the two:

- **Second-party, not reflexive.** The curator is a *separate headless session*
  running its own `memory-advisor` profile, spawned by a reactor on
  `agent.completed`, reviewing another agent's residue. Prime Agent's refiner
  judges its own trajectory from inside the session that produced it. This is
  the same second-party advantage §9.3 credits to the fine-tuner, and jaato has
  it here too.
- **A quarantine tier.** Working agents write memories at `maturity="raw"`, and
  enrichment indexes **`curated.jsonl` only** — raw is explicitly *"the
  curator's queue"* and never auto-surfaces. So an agent-written memory is inert
  until an independent reviewer promotes it. Prime Agent has no such tier:
  a refinement applies to the live store, rebuilds the system prompt, and
  resumes the agent immediately.
- **Expressive scoping.** `{{continuity_scope}}` takes any operator-chosen id —
  project, A2A `contextId`, ticket — against Prime Agent's fixed local/global.

Where Prime Agent's is ahead:

- **More surfaces.** Four entry kinds against the curator's memories alone,
  though `prompt` notes are the riskiest of them and `skill` entries are only
  descriptions.
- **Mechanical rollback.** Per-edit `before`/`after` inversion; the raw→curated
  tier bounds what reaches the prompt but offers no revert of a bad promotion.
#### Why Prime Agent must fire at compaction, and jaato need not

The compaction trigger looks at first like a jaato gap. It is the opposite —
a mitigation for a weakness the two architectures do not share.

| | jaato | Prime Agent |
|---|---|---|
| When the insight is captured | **immediately** — the working agent calls `store_memory`, which atomically writes `memories/raw/{id}.json` (tempfile + rename, one file per writer, no contention) | **not captured at the time**; nothing is written when the insight occurs |
| What the later pass reads | the raw queue **on disk** | whatever survives of the **conversation trajectory** |
| Effect of context loss | none — the file already exists | decisive — the refiner can only extract what is still in context |
| Why fire at compaction | no reason to | **necessary**: it is racing context loss |

jaato memories are not conversation content, so GC and compaction never touch
them. Prime Agent's refinement is *trajectory-dependent extraction*; jaato's is
*eager write, later curation* — two independent decisions, neither of which
needs the history intact. jaato also captures the persist decision at the moment
of insight, when the agent knows why it mattered, rather than reconstructing it
later from a compacted trace.

The trade jaato accepts in exchange is real but small: an agent that never calls
`store_memory` records nothing, whereas Prime Agent's sweep needs no in-turn
discipline. Over-eager storing costs nothing, since raw is quarantined; forgetting
costs the memory outright.

**Verdict.** The "Continual Harness" is Prime Agent's headline abstraction and is
genuinely shipped, but it is not a capability jaato lacks. The memory curator
answers the same need with a second-party reviewer and a quarantine tier Prime
Agent has no equivalent for, and it is immune to the context loss Prime Agent has
to schedule around. The one thing jaato does not have is the **per-edit inversion
ledger**. That alone — not "the refinement pattern" — is the borrow, and §9.2
places it on `memory` rather than beside it.

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
> **Two separate data flows, worth not conflating.** Prime Agent's *analytics*
> (`telemetry.enabled`, **default true**, opt-out via settings,
> `PRIME_AGENT_TELEMETRY=0`, `DO_NOT_TRACK=1`, or `--offline`) sends only
> pseudonymous aggregates — version, OS category, execution mode, TTFT/latency,
> prompt and turn counts, token usage, tool success counts, retries,
> compactions — behind a random installation id, and the docs enumerate what is
> excluded: prompts, responses, thinking, tool arguments and results, command
> text, filenames, paths, repository information, environment variables,
> credentials, raw error messages, hostnames, usernames, emails, hardware ids.
> That is a carefully scoped policy. *Trace sharing* is the separate,
> genuinely opt-in path that uploads whole session traces; it is off unless a
> key with the `agent_traces` scope is configured.

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

### 11.1 ACP: what jaato is actually missing

jaato has no ACP implementation — the term appears nowhere in the tree outside
these comparison documents. This is the clearest single-feature gap in the
comparison, and it is worth being precise about what it buys.

ACP ([agentclientprotocol.com](https://agentclientprotocol.com)) is JSON-RPC 2.0
over newline-delimited JSON on stdin/stdout. `prime-agent --mode acp` implements
`initialize`, `session/new`, `session/prompt`, `session/cancel`, `session/close`,
and streams activity as `session/update` notifications — assistant text as
`agent_message_chunk`, reasoning as `agent_thought_chunk`, tool starts as
`tool_call`, completions as `tool_call_update`.

The point is not the transport. jaato has three already (IPC, WebSocket, and the
headless event stream), all of them richer. The point is that **they are jaato's
own**. An editor wanting to drive jaato must implement jaato's protocol; an
editor that already speaks ACP drives Prime Agent with no Prime-Agent-specific
work at all. That is the difference between having an API and speaking a protocol
someone else already implements — Zed and other ACP clients arrive free.

Two details worth copying if jaato ever adds it:

- **Extensions ride in a reverse-domain `_meta` envelope**
  (`ai.primeintellect.prime-agent`), carrying what ACP has no field for —
  subagents, gate attempts, goals, heartbeats, refinement. A standard client
  ignores `_meta` and still works. *"Nothing non-standard is ever added to an ACP
  object root, which the protocol reserves for future fields."* Disciplined
  protocol citizenship, and the pattern jaato would need for permissions,
  reactors and cascade state.
- **Refusal over silent degradation.** One session per connection, because the
  underlying session is fixed at process startup — a second `session/new` is
  *refused* rather than silently sharing conversation, cwd and model. A concurrent
  `session/prompt` is refused too, and a client-supplied `cwd` that differs from
  the real one is reported back in `_meta` instead of being ignored.

The open question for a jaato ACP mode is **permissions**: jaato's defining
feature is a gate that can block a tool call pending approval, and mapping that
onto ACP's session lifecycle needs design rather than a straight adapter. The
streaming and tool-call mappings themselves would be thin over the existing event
protocol.

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
| Time-driven re-entry (cron schedules, heartbeats) | **Prime Agent** | Total — jaato has no scheduler and no self-initiated re-entry |
| Termination gating (done-ness verdicts) | **Even** | jaato: typed schema + completeness processors; PA: shell exit codes. Different verdict sources, comparable strength |
| Budget ceilings and degradation | **jaato** | Moderate — `budget_control` is multi-dimensional with a tier-rebinding brownout ladder; PA can only hard-stop |
| Drift detection and steering | **jaato** | Total — PA measures no drift signal |
| Event-driven autonomy (reactors, signed wakes, webhooks) | **jaato** | Large — Prime Agent has no declarative rules engine |
| Subagent messaging ergonomics | **Prime Agent** | Moderate — roles, modes, receipts, CLI verb |
| Subagent configurability (per-child model/plugins/GC/permissions) | **jaato** | Large |
| Cross-host / clustered agents | **jaato** | Total (premium gossip) |
| Context GC as a policy engine | **jaato** | Moderate |
| Compaction quality (split turns, steered summaries, kernel survival) | **Prime Agent** | Moderate |
| Continuous, unattended self-editing of harness state | **Prime Agent** | Moderate — `/refine` runs automatically with a gate and an inversion ledger; jaato has no equivalent cadence |
| Closed-loop agent improvement with measured verification | **jaato** | Large — `kb-stage-agent-LoRA-training` trains and evaluates per-stage LoRA adapters against held-out metrics; Prime Agent's `/refine` never measures whether an edit helped, and its traces improve the vendor's models rather than yours |
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

1. **A clock in front of `wake_session`** — a daemon-tier persisted job store
   plus one timer, with Prime Agent's claim-before-advance and tick coalescing.
   Not a scheduler subsystem: `wake_session` already revives cold sessions,
   resolves the workspace server-side, dedups on `event_id`, and notifies
   cascade observers — its docstring already lists `cron` as an intended
   caller. For operator-driven schedules an OS cron plus a thin SDK client
   needs no jaato code at all; a daemon-side store earns its place only for
   agent-owned self-timers, and there it is a column on `wake_binding_registry`
   rather than a new subsystem. Never a runner-tier background task. See §4 for
   the placement trade, the composition, and the `DEFERRED` wrinkle.
2. **Subprocess quality gates as a termination verdict** — `--autonomous-gate
   "pytest"` with timeout and process-tree kill. jaato already has the ceilings
   (`budget_control`) and the gate slot (completion processors); what it lacks
   is a gate that may shell out, and a done-ness check for schema-less
   sessions.
3. **`background.completed` on the runtime event bus** — not a Prime Agent
   feature, but exposed by the comparison: background tasks emit nothing today,
   so a task finishing after the turn ends is invisible. Publishing to
   `JaatoRuntime.event_bus` (the `webhook` plugin's existing pattern) lets a
   reactor resume the agent. Cheaper than the scheduler, and it covers every
   case where jaato owns the work.
4. **A per-edit inversion ledger for curated memories** — the one piece of the
   refinement machinery jaato has no counterpart for. Everything else in §9
   survived checking: the memory curator already does eager-write + second-party
   curation with a quarantine tier (§9.4), `waypoint` already does
   snapshot-and-restore better (§9.2), and `kb-stage-agent-LoRA-training`
   already closes a measured loop Prime Agent does not attempt (§9.3). What is
   missing is `git revert` semantics over a bad curation — undoing one promotion
   while keeping later ones.
5. **`/compact <instructions>`** — user-steered summarisation focus, persisted
   on the compaction record. Cheap to add to `gc_summarize` / `gc_hybrid`.
6. **Agent-to-agent messaging ergonomics** — named agents, sibling addressing,
   `steer` / `follow_up` / `auto` delivery modes with receipts, and a CLI
   `send` verb.
7. **Split-turn compaction** — jaato's GC assumes turn boundaries; a single
   oversized turn is a real failure mode.
8. **Reading foreign skill directories** (`~/.claude/skills`, `~/.codex/skills`)
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
