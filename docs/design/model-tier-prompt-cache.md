# Model Tiers × Prompt Caching — Assessment & Plan

**Status**: ASSESSMENT, with every identified defect fixed. CLOSED: the
config gap (§4), the mutable tier line in the cached system block (§5.1),
the tier-switch re-wire (§5.2), the model-invalidation half of Google's
`CachedContent` binding (§5.3), the invisibility of the miss (§5.4), and
the reliability attribution (§5.5). STILL OPEN: the Google mismatch
guard, the common `cache:` profile field (§7), and everything in §6.

§6 is where the original cost question actually gets answered, and it is
now answerable: with §5.4 landed, a real tiered session reports what its
switches cost instead of hiding it. Everything in §6 remains a hypothesis
until that measurement exists — including the break-even in §3, which is
arithmetic, not data.
**Origin**: the question "our profiles can declare a model tier per task
type and `enter_tier` hands the task to the most suitable one — how does
that impact cache usage?", which had never been assessed.
**Scope**: the interaction between `model_tiers` / `enter_tier`
(`shared/model_tiers.py`, `JaatoSession.switch_tier`) and prompt caching
(`shared/plugins/cache*/`, plus the provider-internal caching in
`openrouter`).

---

## 1. Executive Summary

Tier switching and prompt caching are, today, mutually unaware. The
`enter_tier` tool re-points the session's provider at a different model
**while keeping the whole conversation history**, which is exactly the
operation prompt caching cannot survive: every provider's cache is
keyed per model, so each switch re-reads the entire accumulated prefix
uncached at the new model.

On top of that structural cost, three self-inflicted problems and one
measurement gap:

- the session appended a tier-identity line to the **system block**, the
  root of every cached prefix (§5.1) — **fixed**;
- the cache plugin was wired once and never re-wired or re-informed on a
  switch, so a cross-provider tier ran with no cache plugin at all
  (§5.2) — **fixed**;
- Google's `CachedContent` is created bound to one model while its reuse
  test omits the model (§5.3) — **the invalidation half is fixed**, the
  defensive guard is not;
- per-turn cache figures *replaced* rather than accumulated, so a turn
  containing a switch reported only its last leg (§5.4) — **fixed**, and
  it was the one blocking measurement of any of this;
- a second subsystem attributed its records to the boot model after a
  switch (§5.5) — **fixed**, same shape as §5.4.

The reason none of this has hurt yet is the subject of §4: the framework's
own caching was **not reachable from a profile at all**. That is now
fixed, which makes the rest of this document actionable — and urgent, in
the sense that turning caching on is the thing that makes the tier cost
real.

---

## 2. The mechanism under review

A profile declares tiers (`shared/model_tiers.py`):

```yaml
model_tiers:
  planner:    claude-opus-4-7
  dispatcher: claude-sonnet-4-6
  executor:   claude-haiku-4-5
  initial:    dispatcher
  fallback:   dispatcher
```

The model calls `enter_tier(name)`
(`LifecycleTools._enter_tier_schema`, `shared/lifecycle_tools.py:520`),
which reaches `JaatoSession.switch_tier`
(`shared/jaato_session.py:10231`) → `_connect_tier_entry`
(`:10205`) → `provider.connect(model, skip_model_test=True)`, or a swap
to a cached per-provider instance when the tier declares its own
`provider`.

The defining property: **one session, one history, a different model
underneath.** `_connect_tier_entry`'s own docstring states it — "history
is provider-neutral (Message/Part), so the conversation flows across the
swap". That is the feature. It is also the whole problem.

`enter_tier` is an ordinary tool, so a switch lands **mid-turn**: the
next `provider.complete()` in the same turn runs against the new model
with the full accumulated history.

---

## 3. The structural cost

Prompt caches are keyed per model. A switch is therefore a guaranteed
100% miss on the entire prefix, at the new model.

Break-even, in rate multipliers of base input (cache write 1.25×, cache
read 0.1×), for a prefix of size `P`, `n` consecutive model calls at the
new tier, old-tier base rate `A` and new-tier base rate `B`:

```
stay at old tier:  n · P · 0.1·A
switch to new:     P · 1.25·B  +  (n-1) · P · 0.1·B
```

With a "cheap" tier at roughly a third the price (`B = A/3`), the two
sides cross at **`n ≈ 6`**. Below that, the cheap tier costs *more per
call* than staying on the expensive one warm — the rate advantage does
not cover one cold read of the prefix.

Consequences worth stating plainly:

- **A one-shot hop is always a loss.** Switching to `executor` for a
  single mechanical tool call and back is strictly worse than not
  switching.
- **The `vision` tier is the worst case by construction.** Its documented
  usage — switch in, view an image, switch back — is two full prefix
  misses for one or two calls.
- **The advertised cost is wrong.** `enter_tier`'s description tells the
  model *"Switching is cheap (no network round-trip; just re-points the
  active provider)"* (`lifecycle_tools.py:557`). True for latency,
  actively misleading for spend, and it is the model's primary guidance
  for when to switch.
- **The economics invert as context grows.** Early in a session the
  prefix is small and hops are nearly free; by the time the agent has
  accumulated a large history — precisely when it most wants to drop to a
  cheap tier — each hop is at its most expensive.

This is intrinsic to in-place switching. It cannot be optimised away,
only amortised (§6).

---

## 4. The wiring gap (FIXED)

`_wire_cache_plugin` built the cache plugin's config from
`runtime._provider_config.extra` alone. That object is assigned exactly
once — `ProviderConfig(project=..., location=...)` in
`JaatoRuntime.connect` (`shared/jaato_runtime.py:616`) — with an empty
`extra` that nothing ever writes to. The profile's
`plugin_configs[<provider>]` merge happens inside
`JaatoRuntime.create_provider` (`:1206-1221`), via `dataclasses.replace`
into a **local** config that is never stored back.

Every cache plugin was therefore handed `{}`:

| knob | intended position | actually reached the plugin? |
|---|---|---|
| `anthropic.enable_caching` | `plugin_configs.anthropic` | no — survived only via the `JAATO_ANTHROPIC_ENABLE_CACHING` env default inside `initialize()` |
| `anthropic.cache_ttl` / `cache_history` / `cache_exclude_recent_turns` / `cache_min_tokens` | same | no route at all |
| `google_genai.enable_caching` / `cache_ttl` | `plugin_configs.google_genai` | no — and its default is a hard `False` with no env fallback, so the `CachedContent` path was unreachable from anywhere |
| `openrouter.api_params.cache_prompt` / `cache_ttl` | `plugin_configs.openrouter.api_params` | **yes** — openrouter bypasses the cache-plugin system and reads its own knobs off the live per-session config |

Belt and braces: neither `anthropic/__init__.py` nor
`google_genai/__init__.py` declared a cache knob in `PROVIDER_KNOBS`, so
`scaffold validate` flagged `enable_caching` as `unknown_knob`
(`shared/scaffold/validate.py:267-277`) — correctly, since the runtime
ignored it too. The validator and the runtime agreed the knob did not
exist; only the docstrings claimed otherwise.

That shape is worth naming, because it generalises past this bug: **two
independent sources agreed, the documentation was the outlier, and nobody
checked which was right.** The prose was the least authoritative artifact
in the repository and it was the only one anyone read. Where a knob is
concerned, `PROVIDER_KNOBS` and the read site are the evidence; a
docstring is a claim about them.

**The fix.** The merge is now one function,
`jaato_runtime.resolve_provider_extra`, called by **both**
`create_provider` (building the `ProviderConfig` the provider is
initialized with) and `JaatoSession._cache_plugin_config` (building the
config for the cache plugin attached to that same provider). It folds
`plugin_configs[<provider>]` onto the runtime base, child-wins, and
promotes `api_key` out to the `ProviderConfig.api_key` field.

The two callers share the **function**, not the **result** — and that is
forced, not lazy. `plugin_configs` is a per-session argument, while
`runtime._provider_configs` is runtime-level and shared by every session
on that provider. Writing the merged config back there would leak one
session's profile knobs, credentials included, into every other session
using the same provider. So the merge is necessarily recomputed per
caller, and the only defence against the callers drifting apart is that
there is exactly one implementation of it.

That defence is checked rather than asserted.
`TestTheTwoMergesAgree` drives the *real* `create_provider` and
`_cache_plugin_config` over the same inputs and compares the resulting
extras. Its input is derived from the provider's declared
`PROVIDER_KNOBS` top-level layer rather than hand-enumerated, because a
hand-written case only notices a newly promoted field if it happens to
mention that field — verified by sabotage: making `create_provider` pop a
second key passed a hand-enumerated suite and fails the derived one.

The profile lookup is keyed on `_active_provider_name` (the name the
provider was *registered* under), not `provider.name`; they are not
interchangeable, since zhipuai subclasses the Anthropic provider and
reports the parent's name.

The cache knobs are now declared in both providers' `PROVIDER_KNOBS`, at
`top_level` — the position the read sites actually use. Regression
coverage: `shared/tests/test_cache_plugin_profile_knobs.py`, which
declares a `REVERSIONS` entry so the meta-guard
(`test_every_guard_detects_its_own_reversion`, #665) proves the guard
still notices the defect being put back.

> **Implication for anyone reading old measurements**: prior to this fix,
> any jaato session not on openrouter and without
> `JAATO_ANTHROPIC_ENABLE_CACHING` set was running with framework caching
> off. Cost baselines gathered before it are not comparable.

---

## 5. Open defects

### 5.1 The tier line mutates the system block — FIXED

`_get_effective_system_instruction` appended
`"You are currently operating in the \`<tier>\` tier."` to the system
prompt on every turn. The Anthropic provider folds
`system_instruction` into a **single** text block
(`anthropic/provider.py:777`), and the cache plugin places BP1 on that
block (`cache_anthropic/plugin.py:289`). So the tier line sits inside the
cached anchor, and changing it invalidates BP1 and, transitively, BP2
(tools) and BP3 (history).

The method's docstring claims this keeps "the assembled instruction a
stable cache anchor". That holds for `_system_instruction` in memory, not
for what goes on the wire.

Cost: even cases that *should* hit are missed — returning to a tier the
session already used, two tiers sharing one model, and every
implicit-prefix-caching upstream (OpenAI, DeepSeek, Grok, Gemini) reached
through openrouter's default `cache_prompt: "auto"`.

**The fix.** The system block now carries tier **protocol** instead of
tier **state** — a line that is byte-identical for the life of the
session:

> This session runs in multi-tier mode and started in the `dispatcher`
> tier. Your active tier changes only when you call `enter_tier`, which
> reports the tier you land in.

`initial_tier` is a config value that never changes (a budget rung
rebinds a tier's *model*, not which tier is initial), so BP1 survives
every switch, and with it BP2 and BP3.

The current tier stays derivable — start point plus the `enter_tier`
results already in history — without restating mutable state in the one
place that must not carry any. The model does not need it in order to
*decide*: `enter_tier` is chosen by the work about to be done rather than
by where it currently is, and entering the active tier is a documented
no-op. What it loses is a per-turn restatement; the earlier fallback idea
of moving that to the message *tail* stays available if the loss ever
shows up in practice, and would touch message assembly rather than the
prefix.

The invariant is now asserted directly — the system instruction must be
byte-identical across two switches
(`test_the_system_instruction_does_not_change_across_a_switch`, with a
`REVERSIONS` entry) — rather than left as a property of the text.

**This is what made §5.3 load-bearing.** While the tier line moved, a
Google `CachedContent`'s system+tools hash changed on every switch and
forced a rebuild, which accidentally covered the fact that the hash has
no notion of the model. With the block stable the hash no longer changes,
so the only thing between a tier switch and a cache bound to the wrong
model is §5.3's `set_model_name` discard. The two fixes were made in
separate commits and each reads as safe alone, so the combination is
asserted rather than reasoned about:
`TestTheGoogleCacheFollowsTheModel` drives the real plugin through a
session tier switch with an unchanged hash and requires the cache to be
dropped.

### 5.2 The cache plugin is never re-wired on a switch — FIXED

`_wire_cache_plugin` was called from exactly one place —
`_ensure_provider` — once per session, and `_provider_for_tier` builds
cross-provider tier providers without wiring one. So:

- a **cross-provider tier** targeting anthropic or google_genai ran with
  no cache plugin for the rest of the session — caching silently off, no
  warning (an openrouter tier was unaffected; it caches internally); and
- a **same-provider tier** left the plugin's model pinned to the boot
  model. `AnthropicCachePlugin.set_model_name()` existed with **no caller
  anywhere**, so the minimum-cacheable-size threshold was chosen for the
  wrong model after every switch.

**The fix.** `_connect_tier_entry` re-wires after connecting, which
covers both routes into a tier change: model-driven (`enter_tier` →
`switch_tier`) and framework-driven (a budget-control degrade rung
rebinding the active tier in place, via
`_reconnect_active_tier_if_rebound` — a path where the tier *name* never
changes, so nothing else could catch it).

Three properties worth stating, because each was a decision:

- **Plugin instances are cached per provider**
  (`_cache_plugins_by_provider`), keyed on the **registration** name, the
  same key `_provider_cache` uses — not `provider.name`, which zhipuai
  inherits from anthropic and which would collide two tiers onto one
  plugin built from the wrong `plugin_configs` section. A switch back is
  then O(1) and keeps the metrics and prefix state that provider
  accumulated; re-discovery would rescan entry points on every hop.
- **A provider with no cache plugin clears `_cache_plugin`** rather than
  leaving the previous one attached. That slot drives budget forwarding,
  usage extraction and telemetry, so a stale one would book openrouter's
  cache traffic against anthropic's counters.
- **Re-wiring is best-effort.** The connect still raises — a session
  pointed at the wrong model is not something to continue from — but a
  cache plugin that cannot be attached means running uncached, not
  failing the switch.

Coverage: `shared/tests/test_cache_plugin_follows_tier_switch.py`
(with a `REVERSIONS` entry), plus the model-rebinding tests in
`cache_google_genai/tests/test_plugin.py`.

### 5.3 Google's CachedContent ignores the model — PARTLY FIXED

`GoogleGenAICachePlugin._model_name` was set only in `initialize()` and
the `CachedContent` is created bound to it
(`cache_google_genai/plugin.py:399`), while the reuse test covers system +
tools only (`_compute_content_hash`) — **no model in the key**.

**What §5.2 forced.** Pushing the active model into the plugin (§5.2)
is unsafe on its own here: a switch that left system and tools untouched
would find a matching hash and hand the new model a cache name bound to
the old one. Today the tier line (§5.1) perturbs the system text on every
switch, so the hash always changes and hides it — but correctness must
not rest on an unrelated prompt-assembly detail that §5.1 is about to
remove. So the plugin's new `set_model_name` **discards the cached
content when the model actually changes**, which is the model-invalidation
half of this item expressed at the setter rather than in the hash. It
deletes rather than forgets: the cache is server-side and billed for its
TTL, so an orphan keeps costing until it expires. A no-op when the model
is unchanged, because the session pushes the model on every wire.

**Still open**: a defensive guard refusing to emit a `cached_content`
name whose bound model does not match the request's. The invalidation
above closes the path we know of; the guard closes the class.

### 5.4 The miss is invisible — FIXED

`_accumulate_turn_tokens` (`shared/jaato_session.py:8107-8111`) **sums**
prompt and output tokens but **replaces** `cache_read` and
`cache_creation`. A turn containing a tier switch therefore reports only
the last leg's cache numbers, in the turn record and in
`on_turn_progress`. Cache-plugin telemetry is a session-level scalar
(`_total_cache_read_tokens`) with no per-model or per-tier split.

The per-response OTel path is correct — `llm.model_name` comes from
`self._model_name`, which `switch_tier` updates (`:7403`) — but carries
no tier attribute, so tier must be inferred from the model name, which
breaks when two tiers share a model or a budget rung rebinds one
(`budget_control` degradation does exactly that).

**The fix.** Both halves.

*Accumulation.* `_accumulate_turn_tokens` now also keeps
`spend_cache_read` / `spend_cache_creation`, summed per response, exactly
as `spend_total` and `cost_usd` already were and for the same reason —
every response in a turn is separately billed. The level readings
(`cache_read` / `cache_creation`) are kept as well, because they are what
the streaming usage-callback writes and that callback fires per usage
*chunk*, so it must not sum. Two shapes, two questions: "how big is the
context now" and "what did this turn cost".

The spend pair rides the same five links `cost_usd` does — session →
`on_agent_turn_completed` → runner RPC payload → daemon unpack →
`_build_usage` → `UsageBreakdown` — and the guard checks each link at its
own exit shape (a keyword in a *named* call, a key in the wire dict, a
`payload.get` on the far side). Checking "the name appears as a call
keyword somewhere in the file" was not enough twice over: `rpc.py` and
`core.py` each have two exits, and deleting one left the other matching.

A third blind spot in the same commit, found by a reviewer aiming at it:
the guard on the streaming callback collected `ast.Assign` targets, so
`turn_data['spend_cache_creation'] += 1` — an `ast.AugAssign`, and the
form a double-count would most naturally be written in — passed it, along
with the whole suite. It now asserts that no `spend_`-prefixed string
literal appears anywhere in the callback, which covers `=`, `+=`,
`.update` and `.setdefault` alike, with a parametrised test over all four
so the guard's own coverage is asserted rather than assumed.

A fourth, from the next review round: the shape check reads one function
body, so an ordinary extract-a-helper refactor moves the write out of
view and the guard goes quiet — confirmed with a genuine per-chunk
double-count passing the whole file. No cleverer AST walk fixes that;
only an **effect** assertion does. The streaming tracking is now a named
method (`_track_streaming_usage`) rather than an anonymous closure, so
the test can drive the real code with two chunks and assert the `spend_`
keys did not move — which catches a write anywhere below it, however
many helpers deep. The closure that remains is pinned by *whitelist* to
exactly two statements, because it is the one body the effect test does
not reach and a blacklist there is dodged by a plain call.

Every one of the four was caught by running a sabotage, never by reading
the guard. The rule worth keeping, since the four found it from both
directions:

> A **shape** check reads one body, so any indirection defeats it. An
> **effect** check runs the real thing, so indirection is irrelevant —
> but the body has to be **reachable**. Extracting
> `_track_streaming_usage` from its closure was not a tidy-up; it was the
> minimal change that made the effect check possible at all.

Shape plus effect covers what neither does alone, and the shape check
still earns its place for the message it gives when it fires. One caveat
stated rather than defended: the effect test drives two chunks, so a
write conditional on a third would evade it. That is not a mistake a
refactor produces, and a guard aimed at an adversary rather than at a
mistake buys nothing.

*Attribution.* The LLM span carries `jaato.tier`,
`jaato.tier.switches`, `jaato.tier.cache_rewire_failures` and
`jaato.tier.reliability_retarget_failures` whenever tier mode is active,
and nothing when it is not. The tier is what makes a span's cache figures readable — a miss
after a switch is expected, a miss without one is not.
`jaato.tier.switches` counts real binding changes only (an `enter_tier`
to the active tier short-circuits before it), so it is the multiplier on
what tier mode costs. Both routes increment it, including the
budget-control rebind where the tier *name* never changes.

Deriving the tier from `llm.model_name` instead does not work: two tiers
may share a model, and a degrade rung rebinds a tier's model underneath
it.

The two `*_failures` counters exist because §5.2's post-connect
bookkeeping cannot be allowed to raise — the provider is already
re-pointed by then, so an exception leaves the switch half-applied. That
made two real regressions invisible: a cache plugin that fails to
re-attach leaves the session running **uncached** (a cost regression),
and a failed reliability retarget judges patterns against the **wrong
model** (a correctness one). Three best-effort blocks is not the problem;
three *unobservable* ones is. Both counters are emitted even when zero,
so `> 0` is a queryable condition and a healthy span is distinguishable
from an older build's.

Landing them exposed a second layer of hiding:
`_retarget_reliability_model` had its own `try/except` *inside* the
caller's, so it ate the exception before the counter could see it — the
span would have reported a healthy session while every pattern was filed
against the wrong model. The helper now raises and the caller's block is
the single place that decides. Two layers of swallowing is one layer of
hiding.

Coverage: `shared/tests/test_cache_spend_survives_a_tier_switch.py`
(with a `REVERSIONS` entry).

### 5.5 The other uncalled model setters

`AnthropicCachePlugin.set_model_name` having no caller was the tell that
found §5.2 — a setter nobody calls means a whole path was never wired.
The tree has two more `set_model_name` methods with zero callers, and
they are **not** the same finding. Recording the audit so the next reader
who greps for them does not have to redo it.

**`PatternDetector.set_model_name`** (`reliability/patterns.py:111`) —
**same class, lower stakes, FIXED.** Its `_model_name` is stamped
into every emitted `BehavioralPattern` (six construction sites), and it
is captured once when the detector is built, from
`ReliabilityPlugin._current_model` — the boot model.
`set_session_context` is called without a model name, so nothing updates
it afterwards. After a tier switch, patterns detected while running the
executor tier were attributed to the model that started the session, and
the record could not say which tier misbehaved.  Attribution silently
wrong is worse than absent: absent prompts a question, wrong ends one.

Fixed by making `ReliabilityPlugin.set_model_context` forward to its own
detector, and having the session call it from the same post-connect point
as the cache re-wire — so both routes into a tier change are covered.
Coverage: `shared/tests/test_reliability_model_follows_the_tier.py` (with
a `REVERSIONS` entry).

**`PluginRegistry.set_model_name`** (`registry.py:825`) — **not a gap.**
Its `_model_name` gates plugin *exposure*: a plugin declaring
`get_model_requirements` is skipped when the active model does not match
(`registry.py:1080-1085`; today only `multimodal` declares any). That
gate runs at `expose_tool` time, while the session's tool surface is
being built — before any tier switch can happen, and the surface is then
fixed and already described to the model. Re-running it mid-session would
change tool schemas underneath a model mid-conversation. Construction
time is the correct lifetime here, and per-tier tool surfaces would be a
new feature rather than a repair. The setter is reachable API that
nothing currently needs.

---

## 6. Where to go after that

Ordered by confidence, not by effort.

1. **Measure first.** §5.4 plus the §4 fix is the minimum needed to
   answer "what do tiers actually cost us" with data rather than
   arithmetic. Everything below is a guess until that lands.

2. **Tell the model the truth.** Replace `enter_tier`'s "switching is
   cheap" with the break-even rule, and report the realised miss in the
   tool result ("this switch re-read 58k uncached tokens"). The model is
   the entity making the switch decision; it is currently making it on
   bad information.

3. **Minimum dwell / hysteresis.** Refuse or discourage a switch that
   immediately reverses. Cheap to implement, directly targets the
   one-shot hop that §3 shows is always a loss.

4. **Prefer handoff over in-place switching.** A cheap model is cheap
   because it receives a *small* task brief, not because of its rate;
   handing it the planner's full history defeats the point. Routing
   cheap-tier work to a subagent (which already gets its own
   `JaatoSession` over a shared `JaatoRuntime`) gives each model an
   independently cacheable prefix sized to its job. This is the
   structurally correct answer and it needs no new primitive — but it
   changes the ergonomics of tiers from "switch" to "delegate", so it is
   a design decision, not a fix.

5. **Longer TTL when hopping is frequent.** Anthropic's `1h` TTL costs a
   2× write premium instead of 1.25× but keeps a tier's prefix alive
   across excursions to another tier. Worth evaluating once §5.4 shows
   the real switch frequency; not worth guessing at now.

---

## 7. A common cache knob

Separate from the tier question, and surfaced by §4: caching is the one
cross-cutting concern with no first-class home in a profile.

`ProviderCapabilities.prompt_caching`
(`model_provider/base.py:321,351`) is already a canonical, CI-guarded
declaration of *whether* a provider can cache — `anthropic`,
`google_genai` and `openrouter` declare `True`. There is no matching
declaration of *whether it should*. Instead:

| | anthropic | google_genai | openrouter |
|---|---|---|---|
| on/off key | `enable_caching` (bool) | `enable_caching` (bool) | `cache_prompt` (`auto`/true/false) |
| layer | top-level | top-level | `api_params` |
| TTL units | `5m` / `1h` | `3600s` | `5m` / `1h` |
| default | off | off | **on** (`auto`) |
| delivery | cache plugin | cache plugin | provider-internal |

The mechanisms genuinely differ — breakpoints, a server-side object, and
a gateway annotation are not the same thing. But "on or off", "how long",
and "how much history" are the same three questions everywhere, and the
`CachePlugin` protocol is already the per-provider translation layer.

Proposed shape, a sibling of the existing first-class `gc:` and
`model_tiers:` profile fields (`plugins/subagent/config.py:1006-1127`):

```yaml
cache:
  enabled: auto     # auto (capability + provider default) | true | false
  ttl: 5m           # normalised duration; each plugin maps to its own units
  history: true     # cache the conversation prefix, not just system+tools
```

`auto` is well-defined because `prompt_caching` already gates it, so a
provider that cannot cache degrades to a no-op rather than an error.
Per-provider `plugin_configs` knobs stay as the escape hatch for
mechanism-specific tuning (breakpoint counts, `CachedContent` TTL
format), with the profile field winning where they overlap.

Caching is also a *consumer* of the GC policies it reads
(`plugins/cache/base.py:5-8`), which is a second argument for placing it
at the same altitude as `gc:` rather than one level down inside a
provider's config.

---

## 8. Follow-ups

- [x] `_wire_cache_plugin` reads the session's merged provider config (§4)
- [x] one merge function shared by `create_provider` and the cache-plugin
      config, with an executable agreement guard (§4)
- [x] cache knobs declared in `PROVIDER_KNOBS` for anthropic + google_genai (§4)
- [x] drop the tier line from the system block (§5.1)
- [x] re-wire the cache plugin on tier switch; push the model name (§5.2)
- [x] Google's `CachedContent` discarded when the model changes (§5.3)
- [ ] Google mismatch guard: never emit a name bound to another model (§5.3)
- [x] `PatternDetector` model attribution follows the tier (§5.5)
- [x] accumulate cache tokens per turn; `jaato.tier` span attribute (§5.4)
- [ ] measure a real tiered session, then revisit §6
- [ ] first-class `cache:` profile field (§7)
