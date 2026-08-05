# Reliability Plugin → Event-Driven Reactor: Migration Analysis

**Status:** Proposed (analysis only — not implemented)
**Date:** 2026-06-19
**Author:** advisor (framework) — for Daniel's decision
**Related:** [[project_reliability_plugin_built_but_unwired]], `docs/design/agent-presentation-awareness.md` (reactor precedent: `jaato_premium/drift_monitor/`)

---

## 1. Why this migration

The `reliability` plugin (`jaato-server/shared/plugins/reliability/`) is fully built but **never runs in the shipped daemon**. The original design (pre-dating the event bus) wired it through bespoke imperative hooks the session/executor call directly. The event bus now exists, and the `drift_monitor` premium reactor proves the same class of job — *observe the turn stream, steer via `inject_prompt`* — is done cleanly as an event-subscribing reactor with **one** activation seam (an entry point) instead of several.

### It isn't just unwired once — it's TRIPLE-dead

Verified 2026-06-19 (jaato `87ddd31c`). Even if you filled the one slot everyone talks about, two more seams are also never called outside tests:

| Seam | Activation call | Live call sites (non-test) |
|------|-----------------|----------------------------|
| Plugin attach | `configure_plugins(…, reliability_plugin=…)` (`jaato_runtime.py:727`) | **none** (daemon/runner/client pass 3 args) |
| Nudge sinks | `set_nudge_callbacks(...)` (`plugin.py:1124`) → `NudgeInjector.set_injection_callbacks` (`nudge.py:289`) | **none** (only `tests/test_reliability.py`) |
| Forced approval | `wrap_permission_plugin(...)` → `ReliabilityPermissionWrapper` (`plugin.py:3110/3296`) | **none** (only tests) |

So the "one wiring line" framing undersells it: the current plugin needs **three** bespoke wirings to come alive, and the nudge path (`_inject_nudge`, `nudge.py:381`) silently no-ops today because its four callbacks are `None`. A reactor collapses the detect+steer wirings into **one** entry point and replaces the dead nudge callbacks with `ctx.inject_prompt` — which *is* wired.

---

## 2. The load-bearing split: synchronicity

Reliability divides cleanly along one axis — **does the behavior have to run synchronously, in-path, before a tool executes?**

### 2a. Asynchronous / observational half — migrates 1:1 to a reactor
~85% of the code, and **all** of the value that is reachable today (nudges are dead anyway):

- **Adaptive tool-trust ledger** — `on_tool_result` (`plugin.py:1202`) → `FailureKey` (`types.py:104`) → `EscalationRule` (`types.py:349`) → `TrustState` transitions (`types.py:36`). Pure state mutation off observed `(tool, args, success, result)`.
- **Behavioral pattern detection** — `PatternDetector` (`patterns.py:26`): REPETITIVE_CALLS, INTROSPECTION_LOOP, READ_ONLY_LOOP, ANNOUNCE_NO_ACTION, ERROR_RETRY_LOOP, PREREQUISITE_VIOLATED.
- **Steering by nudge** — emit a corrective message into the next turn's context. Maps directly to `ctx.inject_prompt` (`action_context.py:170`), exactly as `drift_monitor` does (`reactor_logic.py:265`).
- **Observability** — emit `reliability.escalated` / `reliability.pattern_detected` via `ctx.emit_event` (`action_context.py:832`), mirroring `drift.measured`.

### 2b. Synchronous half — CANNOT be a reactor
- **Forced approval** (`ReliabilityPermissionWrapper`, `plugin.py:3110`): intercepts `check_permission`, and for an escalated tool **mutates the permission policy's whitelist** (`plugin.py:3221-3223`) so the inner check returns "ask" *before the tool runs*. A reactor reacting to `tool.call_started` is async/observational and cannot retroactively gate a call.
- **Pre-execution interrupt**: pattern checks fire on `on_tool_called` *before* the tool runs (`patterns.py:157` ← `ai_tool_runner.py:1101`). A reactor degrades this to a **next-turn nudge** rather than a same-call block.

This split is the whole decision. Everything in 2a is a faithful reactor port; everything in 2b needs an in-path seam or gets reframed/dropped.

---

## 3. Event mapping + payload feasibility

Reactor subscribes via rule `match.event_type` against **bus** enum values (`jaato_sdk/event_bus.py`) — note the bus uses `tool.call_started`/`tool.call_completed`, NOT the `events.py` `tool.call_start/end` strings.

| Current hook | Bus event | Payload carries what's needed? |
|--------------|-----------|-------------------------------|
| `on_tool_called(tool, args)` `plugin.py:1023` | `tool.call_started` | ✅ `tool_args` + `call_id` (`event_payloads.py:194-197`) |
| `on_tool_result(tool, args, success, result, …)` `plugin.py:1202` | `tool.call_completed` | ⚠️ `success` + `error_message` + `call_id` only — **no `tool_args`, no raw `result`** (`event_payloads.py:200-214`) |
| `on_model_text(text)` `plugin.py:1044` | `agent.output` | ✅ `text` |
| `on_turn_start/end` `plugin.py:998/1013` | `turn.completed` (+ turn-start tick) | ✅ lifecycle |
| `register_prerequisite_policies` `plugin.py:717` | — | config: load from `.jaato/reliability-policies.json` at reactor init |

### Two payload gaps to close (SDK change, small)
1. **`call_id` correlation** — `tool.call_completed` lacks `tool_args`. The reactor holds a per-session `call_id → args` map (populated on `tool.call_started`) and joins on completion to rebuild `FailureKey`. *No SDK change — reactor-side bookkeeping.*
2. **Lost `_is_error_result` cases** — today `on_tool_result` flags failures even when `success=True` if the result is an error dict (`{"error":…}` / `status_code≥400`, `plugin.py:1229/1253`). `tool.call_completed` carries only the `success` bool + `error_message`, **not the raw result**. To preserve this, enrich `ToolCallCompletedPayload` with either the result dict or a framework-computed `is_error_result` flag. *Requires a one-field SDK payload addition.* Without it, the reactor under-counts failures for tools that "succeed" with an error body.

---

## 4. The reactor design (mirror `drift_monitor`)

New premium package `jaato_premium/reliability/` — 4 files, plus one `pyproject.toml` entry-point line. Contract verified against `drift_monitor`:

```
jaato_premium/reliability/
├── reactor_logic.py   # _session_states: Dict[session_id, _Entry] (import-cache survives script reload);
│                      # handle_event(params, event, ctx) dispatch by event_type
├── state.py           # ReliabilityState — trust ledger + pattern window + nudge cooldowns (port of the
│                      # plugin's instance dicts; framework-agnostic)
├── registration.py    # _SCRIPT_SOURCE shim (execute→handle_event), _RULES (5 subscriptions), get_reactor_definition()
└── __init__.py
```
```toml
[project.entry-points."jaato.premium_reactors"]
reliability = "jaato_premium.reliability.registration:get_reactor_definition"
```

- **State**: ports `_tool_states` (trust ledger), the `PatternDetector` per-turn window, and `NudgeInjector` cooldowns into `_session_states[ctx.session_id]` (import-cache, `reactor_logic.py:81`). ⚠️ **Semantic change**: today `_tool_states` is *global* across sessions (`plugin.py:75`); per-session keying loses cross-session/persisted trust (`_persistence`). Decide whether per-session is correct for the cascade use case, or whether to keep a global tier.
- **Steering**: nudges → `ctx.inject_prompt(message)`; escalations/patterns → `ctx.emit_event("reliability.escalated"|"reliability.pattern_detected", {...})`. The new event types must be added to the SDK `EventType` enum (like `DRIFT_MEASURED`) or `emit_event` drops them (`action_context.py:846`).
- **Subscriptions (`_RULES`)**: `tool.call_started`, `tool.call_completed`, `agent.output`, `turn.completed`, `plan.step_updated` — all real bus types, no engine allowlist gate.

---

## 5. Presentation-aware escalation (not "drop it" — branch it)

The original framing treated forced-approval as a synchronous residue to keep or drop. Better: the **escalation *decision* is the reactor's** (event-driven, both session types); only the **escalation *action* branches on session type**. The framework already knows the type — `PresentationContext.client_type` (`ClientType` enum, `events.py:1850`): `TERMINAL`/`WEB`/`CHAT` are interactive (a human can answer a prompt), `API`/`_headless` is headless (no human in the loop).

**The hard invariant this bakes in:** a headless session must NEVER synchronously block on a human signal. That is *exactly* the §7c bug we just fixed (`run_ephemeral_session` blocked on a terminal that never came). A forced permission prompt in a cascade would hang the same way. So the headless branch must be non-blocking by construction.

| | Interactive (`TERMINAL`/`WEB`/`CHAT`) | Headless / cascade (`API`/`_headless`) |
|---|---|---|
| **On escalation** | synchronous forced prompt — human decides | NEVER a blocking prompt (would hang) |
| **Mechanism** | thin in-path permission read-hook (reactor writes "escalated tools" set → permission check returns "ask"), gated on interactive `client_type` | reactor reacts to `reliability.escalated` / `permission.requested` and steers **async** |
| **Net** | keeps the strongest safety feature where it works | turns the gate into an out-of-band / non-blocking signal |

**Headless steer tiers** — all compose from *existing* reactor `ActionContext` primitives, no new framework:
- **T1 — deny + nudge (non-blocking):** auto-deny the escalated tool and `ctx.inject_prompt` a corrective nudge ("tool X failed N times — stop / reconsider"). The cascade continues; cheap; the obvious first cut. (`inject_prompt:170`)
- **T2 — out-of-band notify (your email idea):** `ctx.post_webhook(...)` to an email/Slack/webhook endpoint so a human reviewer who is NOT in-session learns of the escalation. Pairs with T1 (deny + FYI). (`post_webhook:785`)
- **T3 — async approval gate:** `ctx.gate(...)` SUSPENDS the cascade (suspend, not thread-block) + notifies the human (T2); the human's out-of-band approval releases the gate and the cascade resumes. Human-in-the-loop **without a live session** — the headless analog of the interactive prompt. (`gate:106`)

This is strictly better than "drop forced-approval": the interactive capability is kept, and the headless path becomes a *new* capability (async/out-of-band review) that is impossible today — today reliability is unwired AND a synchronous prompt would hang a cascade. Everything is event-driven + existing primitives; the only in-path code is the interactive read-hook (gated on `client_type`).

---

## 6. Phased rollout (not A-vs-drop — one design, staged)

The presentation-aware design (§5) makes the old "keep vs drop forced-approval" question moot: it's one design, shipped in phases.

### Phase 1 — Reactor: detect + universal non-blocking steer
Port all of §2a to a `jaato_premium/reliability/` reactor. On escalation/pattern, the steer is a `ctx.inject_prompt` nudge — for BOTH session types (always non-blocking, so headless is safe by construction). No permission interception yet.
- **Pros:** one entry-point lights up detect+steer that is currently *triple-dead*; nudges actually fire (`inject_prompt` is wired, unlike `set_nudge_callbacks`); unifies with `drift_monitor`; deletes the bespoke hook protocol + executor `set_reliability_plugin` wiring; pure reactor, zero in-path code. **Strictly better than today** (today: nothing runs).
- **Cons:** no *forced* prompt yet (interactive) and no out-of-band notify (headless) — those are Phase 2. Escalation is a strong nudge, not yet an enforced gate.

### Phase 2 — Presentation-aware enforcement
Add the §5 branch on `client_type`:
- **Interactive:** thin in-path permission read-hook (reactor writes an "escalated tools" set into session-attached state; the permission check reads it and returns "ask"). Restores the synchronous forced prompt — gated on interactive `client_type` so it can never fire in a cascade.
- **Headless:** wire the async tiers — T1 deny+nudge (trivial), then T2 `ctx.post_webhook` notify, then optionally T3 `ctx.gate` async-approval.
- **Cons / design work:** the cross-package escalation-state channel (premium reactor → public permission check); the headless notify/gate need a webhook/email endpoint + (for T3) gate semantics in a cascade.

### Recommendation
**Phase 1 now, Phase 2 staged.** Phase 1 is pure upside (the currently-dead detect+steer finally runs, validated end-to-end) and carries zero headless risk because the only steer is a non-blocking nudge. Phase 2 adds the interactive forced-prompt and the headless out-of-band review (T1→T2→T3) as separately-reviewed increments — each independently valuable, none coupling the easy 85% to the harder cross-package + endpoint work. This bakes in the §7c invariant from day one: a headless session never synchronously blocks on a human.

---

## 7. Work items (if greenlit)

**Phase 1 (first PR):**
1. SDK: add `EventType.RELIABILITY_ESCALATED` / `RELIABILITY_PATTERN_DETECTED` (+ payloads).
2. SDK: enrich `ToolCallCompletedPayload` with `is_error_result` (or `result`) — §3 gap 2.
3. premium: `jaato_premium/reliability/` 4-file reactor + entry point; port `state.py` from the plugin's `_tool_states`/`PatternDetector`/cooldowns; escalation/pattern → `ctx.inject_prompt` nudge (both session types).
4. Decide per-session vs global trust ledger (§4).
5. Deprecate the dead plugin seams (or keep the plugin for its config schema + types, which the reactor reuses).

**Phase 2 (staged follow-ups):**
6. Interactive: cross-package escalation-state channel (reactor → permission check) + the thin read-hook, gated on interactive `client_type`.
7. Headless T1: auto-deny escalated tool + nudge.
8. Headless T2: `ctx.post_webhook` notify (email/Slack) on escalation.
9. Headless T3 (optional): `ctx.gate` async-approval (suspend cascade → notify → resume on out-of-band approval).

---

## 8. Decisions (settled 2026-06-19, Daniel)
1. **Phase 1 scope** — **YES.** Ship the detect+steer reactor (nudge-only, both session types) as the first increment.
2. **Headless escalation depth** — **T3.** Build through to the async-approval gate (T1 deny+nudge → T2 `post_webhook` notify → T3 `gate` suspend/resume), T3 the eventual target. Staged in that order.
3. **Trust ledger scope** — **per-session** (reactor-native, keyed by `ctx.session_id`; no global/persisted tier).
4. **Payload enrichment** — **YES.** Enrich `tool.call_completed` with `is_error_result` so success=True-but-error-body failures are still counted.

### Consequence to confirm — reactor home / premium-gating
The reactor engine + entry point (`jaato.premium_reactors`, `jaato_premium/reactors/`) is **premium-only** — there is no public reactor mechanism. So the reliability *reactor* (the activation/wiring) lives in **jaato-premium**, which makes reliability-**active** a premium capability (like `drift_monitor`). This is NOT a regression of a working feature (reliability is dead/unwired today), and the **logic stays public**: the reactor imports the public `shared/plugins/reliability` module (`FailureKey`/`EscalationRule`/`PatternDetector`/state). A third party can still self-wire the public plugin via the original `configure_plugins(reliability_plugin=…)` seam; the premium reactor is just the supported, maintained activation path. Confirm this productization split is intended.

---

## 9. T3 — headless async-approval gate (semantics sketch)

The headless analog of an interactive permission prompt: an escalated tool in a cascade is **held for an out-of-band human decision**, without a live session and without ever blocking a turn. Built entirely on existing primitives — `HandoffGate` (`reactors/gates/gate.py`), the `gate.released` bus event (`events.py:286`), `ctx.post_webhook`, the inbound `webhook` plugin, `ctx.inject_prompt`, and session-attached state.

### The load-bearing constraint
A turn cannot freeze waiting for a human (that's the §7c hang). So T3 is **deny-now + resume-on-approval**, NOT "pause mid-execution":
- the escalated tool is **denied in the current turn** (the model gets a denial result: *"tool X held for approval, request `<gate>`"*) — turn proceeds/ends normally;
- the **cascade parks** (the stage-transition does not advance);
- approval (or a timeout) later triggers a **fresh turn** that retries the tool, now allowed.

### Sequence
1. **Escalation (reactor, headless session).** On `reliability.escalated`, the reactor:
   - `gate = ctx.gate(f"reliability:{session_id}:{tool}", ttl_seconds=T)` → `lease = gate.try_acquire(owner=session_id)` (GREEN→RED);
   - `gate.announce(lease, {tool, args, reason, session_id, failures})` — records what's pending (emits `gate.announced`);
   - stashes `lease` in **session-attached state** (so the completer/bridge can release it);
   - `ctx.post_webhook(notify_url, {gate, tool, reason, approve_url, deny_url})` — the email/Slack to a human reviewer;
   - denies the tool for this turn + `ctx.inject_prompt(session_id, "tool X held for human approval; do not retry until notified")`.
2. **Park.** The cascade's stage-transition is gated on the gate: the completer/advance reactor checks `gate.is_red()` (or simply waits for `gate.released`) and does not advance while RED.
3. **Human decides, out-of-band.** The approve/deny link hits the inbound **`webhook` plugin** listener; a small route handler loads the stashed `lease` and calls `gate.release(lease, {"decision": "approved"|"denied", "by": <user>})` → **`gate.released`** fires (with the `outcome`).
4. **Resume (completer reactor on `gate.released`).**
   - *approved* → write `tool` into the session's **approved-tools** set (the same session-attached state the Phase-2 permission read-hook consults — "approved" overrides "escalated") + `ctx.inject_prompt(session_id, "approved — retry tool X")`; the cascade advances; the retried call is allowed.
   - *denied* → `ctx.inject_prompt(session_id, "denied — proceed without tool X / abort")`; the cascade advances down the denied branch.
5. **Non-hang guarantee.** The gate's **TTL watchdog auto-releases** after `ttl_seconds` of RED → `gate.released` with a default `outcome` (e.g. `{"decision": "denied", "by": "timeout"}`). A human who never answers cannot hang the cascade — it auto-resolves on the deadline. This *is* the §7c invariant, enforced by construction.

### How the parked agent survives — it does NOT stay live
The agent is **not** kept alive during the wait (that would be the §7c hang plus a held runner/thread for the whole TTL). The parked agent's "state" **is its conversation history snapshot**, and resume is **exactly the `session_ops.interrogate_session` primitive**: snapshot the `Message` list → `create_headless_session(initial_history=…)` (`session_manager.py:4696`; the same call `session_ops/plugin.py:561` and the reactor `fork_from_*` helpers use) → a fresh forked session seeded with the parked conversation, which then continues. T3-resume and interrogation differ only in intent *after* the fork — interrogation probes-then-discards; T3 injects "approved, retry tool X" and **continues** the cascade.

The gate holds only coordination (RED/approval) + a **pointer** to the snapshot — not the agent. So there is no new state-preservation machinery; T3 rides the fork-from-history path that interrogation / premium-handoff / waypoint already use.

**The one ephemeral-specific addition.** Where the snapshot comes from differs by session type:
- **Persistent session:** history is already on disk (SessionPlugin) → fork from it whenever approval lands (literally interrogating a persisted record).
- **Ephemeral (gossip remote-spawn subagent):** explicitly *not persisted* (`session_manager.py:7712`) and torn down at turn end. So T3 must **capture `get_history()` at the park point** and stash it durably (gate `announce` intent or a sidecar) *before* teardown — there is no live/persisted session to fork from later. This snapshot-at-park is the *entire* delta T3 adds over the existing interrogate/fork path.

### What is reused vs new
- **Reused (no framework change):** `HandoffGate` + lease/announce/release + TTL watchdog; `gate.released` event + payload; `ctx.post_webhook`; the inbound `webhook` plugin; `ctx.inject_prompt`; session-attached state; **the `interrogate_session` fork-from-history primitive (`create_headless_session(initial_history=…)`) for resume.**
- **New wiring (small):**
  - a **webhook route → `gate.release`** handler (maps an approve/deny POST to the stashed lease) — the external→gate bridge;
  - the **completer reactor** subscribing to `gate.released` (resume/advance);
  - the **stage-park** check (advance reactor gates on the gate) — shape depends on the orchestrator;
  - the Phase-2 permission read-hook must read **both** "escalated" (→ deny/ask) and "approved" (→ allow) from session-attached state;
  - **snapshot-at-park for the ephemeral case** (`get_history()` → durable stash before teardown) — the only new state-handling, and only because the ephemeral isn't persisted (a normal session forks straight from its disk record).

### Open T3 sub-decisions
- **Timeout default:** auto-**deny** (safe) vs auto-**escalate-louder** (re-notify) vs operator-configurable per policy.
- **Granularity:** approve this `tool+params` once, vs the tool for the rest of the session, vs a standing allow.
- **Where the lease lives** for the bridge (session-attached state vs a gate-registry lookup by name) — affects whether the webhook handler needs the session id.
- **Reply channel:** the inbound `webhook` plugin (links in the email) vs the premium dashboard's approve button vs an IPC command — any can call `gate.release`.
