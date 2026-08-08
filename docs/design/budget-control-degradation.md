# Budget Control & Graceful Degradation — Design Note

**Status**: IMPLEMENTED end-to-end (config + discoverability + runtime).
A profile may declare `budget_control`; it is parsed, inherited,
validated, surfaced by `jaato-scaffold`, carried to the runner on the
session envelope (v5), and ENFORCED at runtime — spend accumulates from
the session's existing accounting, degrade rungs rebind `model_tiers`
bindings (the brownout), and `abort` stops the run. Remaining follow-ups
are listed in §8.
**Origin**: discussion comparing jaato against the "advanced agentic
harness" pattern (typed tools / plan DAG / tiered memory / verification
hierarchy / budgets / tracer). Every primitive in that pattern already
maps onto an existing jaato surface **except** multi-dimensional budget
enforcement with graceful degradation, which the cascade-as-client
design explicitly deferred (see
[`cascade-as-client.md`](cascade-as-client.md) §7: "Cascade-level
resource limits (memory / cpu / token budget tracked per cid)").
**Scope**: a per-profile `budget_control` block that caps resource
consumption and, as ceilings approach, **degrades** by rebinding the
models behind the session's declared `model_tiers` — a brownout, not a
blackout.

---

## 1. Executive Summary

jaato already **measures** every dimension a budget would cap (tokens,
dollars, wall-clock, tool-calls, turns) and already has the **runtime
mechanism** to swap the model backing a tier mid-session
(`JaatoSession.switch_tier`, cross-provider capable). What it lacks is
the thin layer that (a) accumulates those measurements against declared
ceilings and (b) reacts as a ceiling approaches by **reducing cost per
unit of work** rather than hard-stopping.

The design adds one profile field, `budget_control`, with two halves:

- **`limits`** — multi-dimensional ceilings (`usd`, `tokens`,
  `seconds`, `tool_calls`, `turns`). Any dimension may be omitted
  (unbounded).
- **`degrade`** — an ordered list of rungs. Each rung fires once when
  *any* dimension crosses its `at` threshold (a percentage of that
  dimension's limit) and **overlays a new `model_tiers` binding table**:
  the tier vocabulary and the model's cognitive role are untouched; only
  the model each tier *points at* changes. A terminal rung can force
  `finalize` (graceful) or `abort` (ends the session — see §5.1).

The key design decision — arrived at by rejecting two earlier shapes —
is that **degradation rebinds tier→model mappings; it does not move the
agent between tiers**. This keeps the *role* axis (owned by the model
via `enter_tier`) fully orthogonal to the *cost* axis (owned by the
budget), and it removes any need to assume a cost ordering over the
semantic tier labels (see §4).

---

## 2. Background — what already exists

| Concern | Existing surface |
|---|---|
| Token usage + context % | `TokenLedger` (`shared/token_accounting.py`); `turn.progress` payload carries `percent_used`; `context.updated` carries `total_tokens` / `percent_used`. |
| Dollar cost | `UsageBreakdown.cost_usd`, resolved provider-reported → `pricing.json` estimate → `None` (`server/core.py:_build_usage_breakdown` / `shared/pricing.py`). Same precedence the telemetry span cost uses. |
| Wall-clock, tool-calls, turns | `turn.completed` (`duration_seconds`, `function_calls`), `tool.call_completed` (`duration_seconds`), turn counter (`max_turns` is today's only hard cap). |
| Threshold-crossing reactions | The reactor engine already dispatches actions on bus events with JMESPath `where` clauses (see [`reactor-implementation.md`](../reactor-implementation.md)). |
| Model vocabulary + per-turn switch | `model_tiers` profile field + `ModelTierConfig` (`shared/model_tiers.py`); the model moves between tiers via the `enter_tier` lifecycle tool. |
| Runtime model swap (incl. cross-provider) | `JaatoSession.switch_tier` (`jaato_session.py:9357`) → `provider.connect(model, skip_model_test=True)`, with a per-provider instance cache (`_provider_for_tier`, `jaato_session.py:9326`) keyed by `provider_name`. History is provider-neutral (`Message`/`Part`), so it flows across a swap. |

So the mechanism is all present. What is missing is a **`BudgetTracker`**
(accumulate the dimensions per scope, emit threshold events) and the
**overlay-application** step that mutates the tier table when a rung
fires.

---

## 3. Schema

```yaml
# .jaato/profiles/<name>.yaml
model_tiers:
  planner:    { model: anthropic/claude-opus-4,    provider: openrouter }
  dispatcher: { model: anthropic/claude-sonnet-4,  provider: openrouter }
  executor:   { model: anthropic/claude-haiku-4.5, provider: openrouter }
  initial:    dispatcher
  fallback:   dispatcher

budget_control:
  limits:                      # omit a dimension to leave it unbounded
    usd:        3.00           # ← UsageBreakdown.cost_usd (reported → pricing.json)
    tokens:     300000         # ← total tokens (the one dim GC already enforces)
    seconds:    480            # ← summed turn.completed / tool.call_completed durations
    tool_calls: 40             # ← counted from tool.call_completed
    turns:      30             # ← turn counter (max_turns is the hard cap)

  degrade:                     # ordered; each rung fires ONCE, latched, cumulative
    - at: 70%
      model_tiers:             # sparse overlay on the base table; unspecified tiers unchanged
        planner: { model: google/gemini-flash, provider: openrouter }
    - at: 90%
      model_tiers:
        planner:    { model: google/gemini-flash, provider: openrouter }
        dispatcher: { model: google/gemini-flash, provider: openrouter }
    - at: 100%
      action: finalize         # graceful terminal: inject "wrap up and answer now"
      # alternative terminals: `abort` (ends the session: cancels the
      #   in-flight turn AND refuses further turns, §5.1)
      #   | `escalate` (hand to cascade owner)
```

**Field notes:**

- A `degrade[].model_tiers` overlay entry is a **tier-entry value** —
  the identical `str` | `{model, provider}` grammar the base
  `model_tiers` already parses via `_normalize_tier_entry`
  (`model_tiers.py:139`). No new syntax; overlays validate through the
  same path. `provider` is the plugin name (`openrouter`, `anthropic`,
  `ollama`, …); for OpenRouter the `vendor/model` form lives inside
  `model`.
- Overlays may **only** reference the officially declared tier names
  (`VALID_TIER_NAMES` = `planner` / `dispatcher` / `executor` /
  `vision`). They introduce no ad-hoc labels, so no widening of the tier
  vocabulary or the `enter_tier` tool schema is required.

### 3.0 Authoring a budgeted profile — four traps

Found while the first PoC was being built; none of them is caught by
`jaato-scaffold validate`, so they are documented rather than enforced.

1. **`max_turns` must exceed `limits.turns`.** If both are `4` the run
   stops at turn 4 either way and the abort is *unattributable* — a
   budget stop is indistinguishable from the turn cap. Set `max_turns`
   comfortably above the turn ceiling so stopping is attributable only
   to the budget.
2. **Preload the tools the demo depends on.** Most tool plugins are
   discovery-gated (`cli` included): the model must call
   `list_tools` / `get_tool_schemas` before it can use them. Under a
   turn budget that discovery *burns the budget*, so rungs fire during
   discovery instead of during the work, and the run stops being
   reproducible. Use `cli(preload)` (the validator emits an
   informational `discovery_gated_tools` line listing what is deferred).
3. **Every model in a ladder must support what the agent actually does.**
   A rung that degrades to a model which cannot make tool calls does not
   degrade the agent, it kills it — and it fails at the worst moment,
   mid-run, on a rung that only fires under pressure. Worse than a bad id,
   which at least fails on turn 1.
   Checking a catalogue's `supported_parameters` is **necessary but not
   sufficient**: that field is an aggregate over the providers serving a
   model, so it tells you the model can do tools *somewhere*, not that the
   provider you are routed to will. A gateway can advertise `tools: true`
   and still route to an upstream that rejects them. Prefer models already
   exercised in the same deployment over cheaper unproven ones.
4. **Model ids are not validated.** The validator checks tier names and
   that a named provider is installed — it does NOT check that the model
   id exists on that provider. A typo validates clean and fails at
   `connect`. Check the provider's catalog (for OpenRouter,
   `GET /api/v1/models`, public) — note e.g. the OpenRouter id is
   `anthropic/claude-haiku-4.5` (dot), while the Anthropic-native
   spelling is `claude-haiku-4-5-20251001`. This bites hardest on a
   `fallback` tier, which may not be entered until late in a run.

Also note `system_instructions` is deprecated at profile level — put the
persona in `.jaato/agents/<name>.md`.

### 3.1 No `scope` field (deliberate)

An earlier draft carried `scope: agent | cascade`. It was dropped: a
profile is a reusable **template**, but a cascade budget is a **runtime
aggregate** over a live `cid` spanning many sessions — putting a
cascade-wide cap on a leaf profile is a category error (which profile
owns the number when three cascades each spawn the profile? when two
profiles in one cascade both declare a cascade cap?).

Therefore: **a profile-level `budget_control` is always the envelope of
the one agent instance created from it.** Cascade-/subtree-wide
budgeting is a *separate surface* declared at cascade launch, on the
cascade-as-client **owner** (which already holds lifecycle authority for
the cid). The two compose by **min-wins down the spawn tree**:

```
effective_agent_limit[dim] = min(profile.budget_control.limits[dim],
                                 cascade.remaining[dim])
```

Two axes stay distinct: **profile inheritance** (parent profiles →
min-wins on the template) vs **spawn-tree propagation** (cascade
remaining → clamps descendants at runtime). Conflating them under one
`scope` enum was the smell. The cascade-launch surface is out of scope
for this note (§8) but the min-wins composition rule is the contract it
must honor.

---

## 4. Why degradation rebinds tiers instead of switching them

Two rejected shapes and why the third wins:

**Rejected A — "clamp the tier cap."** Degrade by forcing the agent
down to a cheaper tier (`planner` → `executor`). Problem: the tier
labels are a **cognitive/role** axis, not a cost axis, and
`model_tiers.py:64` says so explicitly — "order is conceptual … but
doesn't enforce ordering on the model assignments; operators are free to
wire them however the provider's pricing makes sense." An operator may
map `planner → haiku` and `executor → opus`. So "degrade = go to
executor" reads cost meaning into labels that carry none, and it *also*
yanks the model's role identity (it thinks it's now "executing" when it
was "planning").

**Rejected B — "budget owns an independent ordered model list."**
Decouples cost from role, but now two things (the model's chosen tier
and the budget's forced rung) drive the single `_active_tier` /
`_model_name`, requiring a reconciliation rule when they disagree.

**Chosen — rebind the tier→model table.** At 70%, `planner` still means
"planner" and the model still calls `enter_tier(planner)` when it wants
to plan — but `planner` now *resolves to* `gemini-flash`. The role axis
is untouched; the budget mutates only the binding beneath it. This:

- needs **no cost ordering** over tiers (each tier's replacement is
  declared independently and explicitly — fully sidesteps the
  `model_tiers.py:64` caveat);
- introduces **no new labels** (references only declared tiers);
- never fights the model's `enter_tier` choices (orthogonal axes).

A **brownout, not a blackout**: every "room" dims to a cheaper bulb;
none are switched off.

---

## 5. Runtime semantics

1. **Trigger — first dimension wins.** A rung fires when *any* declared
   dimension crosses its `at` percentage. Blowing the dollar budget
   while tokens are fine still fires the rung. This is the
   "multi-dimensional" property.
2. **Latched.** Once a rung applies, its overlay stays applied even if a
   later measurement recovers below the threshold. Required because GC
   *lowers* `percent_used`; without latching the token dimension would
   flap the model between opus and flash on every GC cycle.
3. **Cumulative, sparse overlay.** Each rung is a patch keyed by tier;
   unspecified tiers keep their current binding; later rungs win on
   collision. A rung only restates a tier whose binding changes at that
   threshold.
4. **Application = mutate the tier table, then re-resolve the active
   tier.** Applying a rung overlays its entries onto the session's
   `_tier_config.tiers`, then re-points the *currently active* tier at
   its (possibly new) model. New turns resolve from the mutated table
   automatically.

---

### 5.0 Verifying a brownout actually took effect

The cheapest independent witness is **per-turn wall-clock**, not the
`budget:` log lines or the system notice — both of those are *reports*
that the swap happened; duration is an observable *consequence* of it.

From the first live PoC (`turns: 4`, rung at 50% rebinding `planner`
opus-4 → gemini-2.5-flash-lite):

| turn | model | duration |
|---|---|---|
| 0 | opus-4 | 7.411 s |
| 1 | opus-4 | 6.415 s |
| 2 | flash-lite | 1.319 s |
| 3 | flash-lite | 1.350 s |

A ~5× collapse landing exactly on the rung boundary. If the rebind had
not reached the provider, the durations would not move. This is worth
asserting on in any brownout demo: it needs no log access, no telemetry
backend, and it cannot be faked by a notice that fires without the swap
behind it.

---

### 5.2 What each dimension actually measures

Sizing a budget from the wrong mental model is the easiest way to get a
useless one. Two traps, both observed on real runs:

**`tokens` is prompt-inclusive, so it is a CONTEXT-PRESSURE budget, not a
work budget.** It accumulates `usage.total_tokens` per *response*, and
`total_tokens` includes the whole prompt — which is re-sent every turn.
The dimension therefore grows superlinearly with conversation length
rather than tracking incremental work. This is the correct measure of
*spend* (you are billed for the prompt on every call), but anyone sizing
a token budget by asking "how much work should this agent do" will
undershoot badly.

And the number most people reach for to measure it is **not** the number
the budget counts. From a real calibration run (`limits: {tokens: 9000}`,
3 turns, 5 responses):

| source | values | sum |
|---|---|---|
| per **response** — what the tracker accumulates | 2150, 2504, 1685, 1572, 1594 | **9505** |
| per **turn** — `turn.completed.total_tokens` | 2504, 1572, 1594 | 5670 |

The tracker's own verdict at abort was `tokens 106%` of 9000, and
9505/9000 = 105.6% — so the per-response sum matches it exactly, while
the per-turn figure accounts for **59%** of it. A turn with a tool call
has >=2 billed responses, and `turn_data['total']` *assigns* rather than
accumulates, so only the last survives. Fixed additively — `spend_total`
accumulates alongside `total`, because `total` is legitimately the
end-of-turn CONTEXT SIZE that GC and the context displays read.

Note responses 3-5 (1685/1572/1594) are *lower* than 1-2 (2150/2504):
a degrade rung swapped in a cheaper model and the context accounting
shrank. Spend went down after the brownout — the brownout working.

**`usd` only advances when a cost is actually KNOWN.** Resolution is
provider-reported → `.jaato/pricing.json` → nothing. With neither source
the dimension stays at `0` and no `usd` rung can ever fire — deliberate
(§9: never hard-stop on an invented number), but the failure mode is
silent: the budget looks configured and is inert. A live OpenRouter run
with no pricing table reported `cost_usd: None` on every turn. **Verify
`usd` advances before relying on it**; `tokens` / `turns` / `tool_calls`
are always exact.

---

### 5.1 What `abort` means (settled by the first live run)

`abort` ends the **session**, not merely the turn: it cancels the
in-flight turn via the cooperative `request_stop`, **and** latches
`_budget_exhausted_reason` so every subsequent `send_message` is refused.

The latch is not belt-and-braces — without it the ceiling does not
exist. `request_stop` only cancels the turn in flight, and rungs are
**latched**, so the 100% rung never fires again; a client that simply
sends another message then runs completely unbudgeted. The first live PoC
ran a `limits: {turns: 4}` profile to **8 turns — 200% of budget** on
exactly this path, with three real tool calls after the abort. A ceiling
that only cancels one turn is not a ceiling, and "hard stop" in the
earlier draft of this note was wrong about its own mechanism.

This also bounds the cascade pool (§8): if pool exhaustion pushes `abort`
to each live session and `abort` did not refuse later turns, the pool's
overshoot would be unbounded for as long as any client kept sending —
not the "one in-flight turn per child" the design claims.

`finalize` deliberately does **not** latch a refusal: it is the graceful
terminal, and blocking further turns is `abort`'s job.

---

## 6. Mechanism against real surfaces

Two components, both thin:

### 6.1 `BudgetTracker`

A per-session object that subscribes to events already on the bus and
accumulates the dimensions:

- `usd` ← `UsageBreakdown.cost_usd` off `turn.completed` /
  `context.updated` (prefer provider-reported; fall back to
  `pricing.json` — never guess when both are `None`, just don't advance
  the `usd` dimension).
- `tokens` ← running total (already tracked).
- `seconds` ← sum of `turn.completed.duration_seconds` (+
  `tool.call_completed.duration_seconds` if tool wall-clock is counted
  separately).
- `tool_calls` ← count of `tool.call_completed`.
- `turns` ← turn counter.

After each update it computes `max(dim_used / dim_limit)` over declared
dimensions and, when a not-yet-fired rung's `at` is crossed, applies
that rung. Emits a `budget.threshold_crossed` bus event (source_agent
convention as per reactor infinite-loop prevention) so reactors /
observers can also react.

### 6.2 Overlay application — the one real blocker

Applying an overlay to the **currently active** tier must survive the
`switch_tier` short-circuit. `switch_tier` today compares **tier
names** (`jaato_session.py:9395`):

```python
actual_tier, entry = self._tier_config.model_for(requested_tier)
if actual_tier == self._active_tier:
    return {"status": "already_at_tier", ...}   # ← no re-connect
```

So if the agent is in `planner` when the 70% rung rebinds
`planner: opus → flash`, calling `switch_tier("planner")` would
short-circuit and **never re-`connect` the new model** — the rebind
wouldn't take effect until the agent happened to leave and re-enter
`planner`. The fix is a re-resolve path that compares the **resolved
entry**, not the tier name:

- After overlaying onto `_tier_config.tiers`, re-run `model_for(active)`
  and, if `entry.model` / `entry.provider` differs from what
  `self._provider` is currently connected to, run the
  swap/`connect(entry.model, skip_model_test=True)` even though
  `_active_tier` is unchanged.

The cross-provider swap itself is already handled: `_provider_for_tier`
(`jaato_session.py:9326`) caches provider instances **keyed by
`provider_name`**, and a same-provider model change is just
`self._provider.connect(new_model)`. Because that cache is keyed by
provider name (not by model), correctness for a *same-provider* rebind
(opus → flash, both `openrouter`) depends entirely on re-running
`connect(entry.model)` after the overlay — which the re-resolve path
above provides. A *cross-provider* rebind (e.g. degrade to
`provider: ollama`) swaps the cached instance by name in O(1), already
supported.

**This short-circuit is the single implementation risk to verify;
everything else reuses existing, tested paths.**

### 6.3 Scaffold discoverability (explain / validate / build)

`jaato-scaffold` introspects the **installed** framework — a new profile
block is invisible to `explain`, and silently accepted by `validate`,
unless wired at the same three points `model_tiers` is. The
implementation **must** follow this pattern (it is not optional polish —
a block that skips it is a silent-ignore footgun):

1. **`explain profile`** renders `introspect.profile_schema()`
   (`scaffold/introspect.py:274`), which iterates
   `dataclasses.fields(SubagentProfile)` and reads each field's
   `metadata["description"]`. → `budget_control` must be a real
   dataclass field on `SubagentProfile`
   (`shared/plugins/subagent/config.py`) with a `default_factory` and
   `metadata={"description": …}`. Without the field,
   `SubagentProfile.from_dict` silently drops a `budget_control:` YAML
   key (it reads only known keys) and `explain` never shows it.

2. **Constraint surfacing** — `introspect._profile_field_constraints()`
   (`scaffold/introspect.py:254`) resolves allowed-value hints bounded
   by a framework constant, exactly as it does for `model_tiers`
   (`VALID_TIER_NAMES` / `RESERVED_KEYS`). → add a `budget_control`
   entry surfacing the valid dimension names
   (`usd` / `tokens` / `seconds` / `tool_calls` / `turns`), the
   `degrade[].at` percentage form, and the terminal action names
   (`finalize` / `abort` / `escalate`) — so an author sees the real
   shape without reading source.

3. **`validate`** — `_validate_profile` (`scaffold/validate.py`) walks
   the **resolved** `SubagentProfile` and carries a bespoke branch per
   structured block (`model_tiers` at `validate.py:116-132`). →
   `budget_control` needs its own branch that:
   - checks `limits` keys ∈ the known dimension set;
   - validates each `degrade[].model_tiers` overlay by **reusing the
     identical `VALID_TIER_NAMES` + `introspect.resolve_provider`
     checks** the `model_tiers` branch already applies — an overlay *is*
     a tier table, so it inherits the same tier-name-typo and
     uninstalled-cross-provider defects;
   - errors on an overlay rung in a profile that declares **no**
     `model_tiers` (Risk §9 — overlay is meaningless in single-model
     mode);
   - checks the terminal `action` ∈ {`finalize`, `abort`, `escalate`}.
   Without this branch a mistyped `budget_control` is **silently
   ignored** — the precise failure mode `_validate_profile` exists to
   prevent (it calls this out explicitly for `plugin_configs` knobs).

4. **`build` / `new`** — `emit_then_validate`
   (`scaffold/api.py` / `build.py`) runs every generated profile back
   through the validator, so once (1)–(3) land a scaffolded profile
   carrying `budget_control` is validated for free; no build-specific
   work beyond emitting the block.

Net pattern: **dataclass field + `metadata` description + constraint
entry + bespoke `validate` branch**, mirroring `model_tiers`
end-to-end. A test analogous to those under `scaffold/tests/` should
assert the field appears in `profile_schema()` and that a malformed
`budget_control` produces a diagnostic (guarding against silent
regression to the ignore path).

---

## 7. Inheritance

`budget_control` follows the profile-inheritance conventions in
`shared/plugins/subagent/config.py`:

- **`limits`: min-wins.** A child may only *tighten* a dimension:
  `effective[dim] = min(child[dim], parent[dim])`. A child can never
  grant itself a larger ceiling than a parent — mirrors the safe
  direction of the existing `suppress_base_instructions` union merge
  (a restriction any layer imposes stays imposed).
- **`degrade`: scalar-override** (child's whole ladder wins if set,
  else inherit), matching how `model_tiers` itself merges
  (`config.py:1768`, scalar-override).

---

## 8. Out of scope (deferred)

- **Cascade-launch budget surface** — the owner-side aggregate cap and
  its min-wins propagation down the spawn tree (§3.1). This note fixes
  the composition contract; the surface itself is a follow-up on the
  cascade-as-client owner.
- **Per-cid resource tracking** (memory / cpu) — the broader
  `cascade-as-client.md` §7 deferral.
- **`finalize` / `escalate` agent-facing effect.** Both are latched on
  `JaatoSession._budget_terminal_action` and surfaced to the client, but
  neither injects anything into the model's context. That is deliberate:
  the reactor layer already owns agent-directed actions (`inject_prompt`,
  `fork_from_originating`, `delete_session`), so the right shape is a
  typed `budget.threshold_crossed` bus event a reactor rule matches on —
  which means adding an `EventType` + payload (an SDK/TS-codegen protocol
  change) and is its own change. `abort` needed no such wiring because
  cooperative cancel is a session-local primitive (`request_stop`).
- **Cascade-level budgets** — per-cid aggregate ceilings and the
  min-wins propagation down the spawn tree (§3.1). The per-profile
  envelope is what landed; the owner-side cascade cap is still open.

---

## 8b. Implementation status

Landed (config + discoverability):

| Piece | Where |
|---|---|
| `BudgetControlConfig` / `DegradeRung` / `merge_limits`, parsing + validation | `shared/budget_control.py` (new; modelled on `shared/runtime_limits.py`) |
| Profile field + all 4 loaders + JSON-validation helper | `shared/plugins/subagent/config.py` |
| Inheritance (`limits` min-wins, `degrade` scalar-override) | `config._merge_budget_control` |
| `explain profile` constraint surfacing (§6.3.1–2) | `shared/scaffold/introspect.py` |
| Validator branch (§6.3.3) — uninstalled overlay provider, `budget_overlay_without_tiers`, `budget_overlay_undeclared_tier` | `shared/scaffold/validate.py` |
| 45 tests | `shared/tests/test_budget_control.py`, `shared/tests/test_scaffold_budget_control.py` |

Deliberate parse-time invariants (fail loud at profile load, not at
session start): unknown dimension; non-positive limit; `at` outside
`(0, 100]`; unknown action; overlay naming a non-tier or a control key
(`initial`/`fallback`); a rung that does nothing; **`degrade` without
`limits`** (`at` is a percentage *of* a limit, so no rung could fire);
non-strictly-increasing thresholds.

Landed (runtime):

| Piece | Where |
|---|---|
| `BudgetTracker` + `BudgetUsage` + `overlay_tier_table` (§6.1) | `shared/budget_control.py` — pure logic, no session coupling |
| §6.2 resolved-entry re-resolve + extracted `_is_connected_to` / `_connect_tier_entry` | `shared/jaato_session.py:switch_tier` |
| Observation hooks (tokens+usd per response; turns+seconds+tool_calls per turn) | `jaato_session._budget_observe_response` / `_budget_observe_turn`, folded into the EXISTING `_record_token_usage` and turn-end accounting — no new measurement path |
| Rung application (brownout + `abort`) | `jaato_session._apply_budget_rungs` / `_reconnect_active_tier_if_rebound` |
| Wire: envelope v5 `budget_control` + `to_dict`/`from_dict` round-trip | `shared/session_envelope.py`, `server/session_manager.py` |
| Plumbing: profile → session | `server/core.py`, `server/runner/session.py`, `jaato_runtime.create_session`, `JaatoSession.configure` |
| 17 runtime tests | `shared/tests/test_budget_runtime.py` |

Cost resolution reuses `_resolve_span_cost` (provider-reported → pricing
table → `None`), so the budget and the telemetry span always agree, and
an unknown cost leaves the `usd` dimension **unadvanced** rather than
advancing on a guess. All budget work is wrapped so a guardrail failure
can never break a live turn.

---

## 8c. Validated end-to-end (two live PoCs)

**PoC #1 — per-session brownout + ceiling.** A `turns: 4` profile with a
50% rung rebinding `planner` opus-4 → gemini-2.5-flash-lite: the rebind
and the active-tier re-connect land **1 ms apart** (§6.2 holds in
production, not only in unit tests); exactly four tool calls then none;
four subsequent sends refused; every decision visible client-side as
`AGENT_OUTPUT source='system'`. Per-turn duration collapsed ~5× at the
rung boundary — the independent witness of §5.0.

**PoC #2 — cascade cap, linear 3-stage chain.** Pool 12000, three stages
of `tokens: 9000`, **none of which mentions the cascade**:

| | capped | uncapped baseline |
|---|---|---|
| stage 2 | 3654 tokens, 1 turn — aborted at 132% of its **clamped** 2773 | ~9300, 2–4 turns |
| stage 3 | never spawned, refused with a machine-readable reason | ~9300, 2 turns |

Pool depletion 12000 → 2773 → 0, with the client's `cascade.budget.get`
and the daemon's clamp line agreeing at every boundary. The uncapped
baseline is what makes each difference attributable to the cap rather
than to stage variance.

### Why the pool reconciles against the tracker — measured

In the PoC #2 run the event stream was **inflated by 33%** by the
duplicate-emission bug (§ open issues):

| source | stage-1 tokens |
|---|---|
| `turn.progress` raw | 12261 |
| `turn.progress` de-duplicated | 9227 |
| **pool contribution** | **9227.0** — exact |

Had the pool still accumulated from that stream it would have believed
stage 1 spent 12261 and refused stage 2 outright. It reconciles against
the per-session tracker's absolute totals instead, so the inflation
never reached it. The event stream has now been shown wrong in **both**
directions on real runs — inflated by re-emission, deflated by a dropped
final-turn event — which is the case for never deriving a ceiling from
it.

### Reading a pool: use `usage_fraction`, not differences of `remaining`

`remaining()` floors at zero, so differencing it across stages
*understates* the last one (a stage that really spent 3654 shows 2773).
`usage_fraction()` does not floor: `1.0734 × 12000 = 12881`, which is
both stages' real spend to the token. For per-stage accounting, read
`usage_fraction`.

---

### 8e. Whose degradation policy applies to a child

A child in a cascade always runs against the CLAMPED limits. What differs
is whose *policy* governs it, and the rule turns on whether the child's
author expressed one at all:

| the child's profile | limits | degrade ladder |
|---|---|---|
| declares `budget_control` **with** a ladder | clamped | **its own** |
| declares `budget_control`, **limits only** | clamped | **none — taken literally** |
| declares **no** `budget_control` | clamped to the cascade remainder | **inherits the cascade's** |

The middle row is the deliberate one. A block with `limits` and no
`degrade` is an author saying "cap me but do not degrade me", and the
cascade is not entitled to override it — **the cascade constrains
ceilings, never policy**. Degrading a profile whose author did not ask for
degradation would be the framework substituting its judgement for theirs.

The third row exists because the alternative is worse than it looks: a
profileless child previously received a ceiling with **no behaviour
attached**. Its tracker accumulated, crossed the limit, and nothing fired
— "budgeted" only in the sense that a number had been written down, with a
best-effort push the sole thing that could degrade it.

Note the inherited ladder's thresholds are percentages of the **child's
clamped limit**, not the pool's. The policy *shape* ("brown out at half,
stop at full") applies at whatever scale the child was allocated, which is
what makes it meaningful for a child holding a slice rather than the whole
pool.

**Interaction with the aggregate ceiling.** Inheritance means a
profileless child SELF-ENFORCES — it reaches its own clamped ceiling and
fires the rungs itself, no push involved. That removes the push from the
critical path for those children. It does NOT close the aggregate hole:
the clamp is still a read, so N concurrent children each clamped to the
full remainder will each self-enforce correctly and still sum past the
cap. Self-enforcement bounds each child; only a reservation bounds the
sum.

---

### 8f. Scope: "cascade" is one grouping, not the general case

The aggregate ceiling is currently keyed on ``cascade_driver_id`` and
lives in ``SessionManager._sessions``. That was a convenience — a cid was
the only pre-existing way to name a group of related agents — and it left
the naming, and the implementation, narrower than the problem.

The relationship that actually matters is **parent → children**. A cascade
is one way to have it. A plain main agent calling ``spawn_subagent`` is
another, and it is the more common one.

**What that currently means, stated plainly:**

| how the child is created | in `SessionManager._sessions` | carries a cid | pool / clamp / push |
|---|---|---|---|
| `session.new` over IPC with a `cascade_driver_id` | yes | yes | **applies** |
| `session.new` without one | yes | no | no aggregate |
| `spawn_subagent` (subagent plugin) | **no** — runtime-level session | no | **none of it** |

A subagent is a `JaatoSession` created by ``runtime.create_session()``, not
a daemon session. It never enters ``_sessions``, so the pool cannot see its
spend, the spawn-time clamp never runs for it, and a mid-flight push cannot
reach it. Its own profile's ``budget_control`` is the only budget it can
have — which is why that had to be forwarded (it was not, and a subagent
was silently unbudgeted regardless of what its profile declared).

**Consequence for a non-cascade parent.** A main agent with a strict
``budget_control`` bounds only *its own* session. Spawning ten subagents
does not touch that ceiling, because the parent's session is not the one
spending. There is no aggregate over the family today unless the family
happens to be a cascade.

**Direction, not yet built.** The pool should be keyed on a **spend
scope** — an identifier a cascade *or* a parent session can own — rather
than on `cascade_driver_id` specifically, with children drawing from the
scope they were spawned into. That generalisation and the reservation
question in §8 are the same decision at different scopes, and are best
made together rather than retrofitting the cid-keyed mechanism twice.

Until then, read every "cascade" in §8/§8b-§8e as "cascade specifically",
not "any parent" — the mechanism is real but its reach is narrower than
the vocabulary suggests.

---

### 8d. When a cascade rung actually takes effect

A pushed rung does **not** take effect when the pool crosses it. It takes
effect at each child's **next turn boundary**.

A child inside a model call does not service the RPC until its turn ends,
so a push aimed at a busy child does not ack within the daemon's timeout.
That timeout is the daemon giving up waiting, **not** a rejection — the
rung still lands, and rungs latched while a child was busy are applied
together at the next boundary. Nothing is lost; delivery is *delayed by up
to one turn*.

Two consequences worth designing around:

* This delay **is** the `cap + (N x one turn)` overshoot. The bound is not
  a safety margin bolted on — it is the direct arithmetic of every live
  child being able to finish the turn it is in before a rung reaches it.
* A timeout in the logs is expected traffic under load, not a fault. It is
  logged at INFO for that reason; a genuine delivery failure (an
  unreachable runner) is the WARNING, and is the only case where a child
  may never receive the rung.

### The aggregate ceiling is conditional on the push

Worth stating plainly because it bounds what §8c's validation claims. The
spawn-time clamp cannot bound a fan-out: N children spawning concurrently
each legitimately see the same full remaining, so each may be granted the
entire pool. Under concurrency the aggregate ceiling therefore rests
**entirely** on the mid-flight push, and the push is best-effort.

Measured on the same harness: with the push working, total spend settled
at ~158% of cap (inside the N-turn bound). With the push broken, ~307% of
cap and no ceiling at all. Nothing in the current design sits between
those two outcomes — a per-child *reservation* at spawn would provide that
floor and demote the push to an optimisation. Not built; see §8.

---

## 9. Risks

- **`switch_tier` name-compare short-circuit** (§6.2) — the correctness
  hinge for rebinding the active tier. Must be addressed or the overlay
  silently no-ops until the next `enter_tier`.
- **Silent-ignore if scaffold wiring is skipped** (§6.3) — a
  `budget_control` block with no `SubagentProfile` dataclass field +
  `validate` branch is dropped by `from_dict` and never flagged. The
  block *looks* configured while doing nothing. This is the highest-
  likelihood regression and the reason §6.3 is a hard requirement, not
  polish.
- **`usd` accuracy** — provider-reported cost is exact; `pricing.json`
  drifts. The tracker must prefer reported cost and simply not advance
  the dollar dimension when neither source has a number (never
  estimate-then-enforce a hard stop on a guess).
- **Latch vs. GC interaction** — mandated latching (§5.2) is what keeps
  GC-driven `percent_used` recovery from flapping the model binding.
- **Single-model profiles** — `budget_control.degrade` with
  `model_tiers` overlays is meaningful only when the profile declares
  `model_tiers`. For single-model profiles, only non-tier rung actions
  (`finalize` / `abort`) are applicable; overlay rungs should be
  rejected at profile-load with a clear error.
