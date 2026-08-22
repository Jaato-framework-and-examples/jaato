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
4. **A single reviewable refinement ledger** with before/after snapshots and
   rollback, sitting above memory / auto-steering / references rather than
   beside them.
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
