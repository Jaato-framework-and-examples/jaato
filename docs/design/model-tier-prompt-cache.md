# Model Tiers × Prompt Caching — Assessment & Plan

**Status**: ASSESSMENT, with one fix landed. The wiring gap in §4 is
CLOSED (`_wire_cache_plugin` now reads the session's merged provider
config, and the cache knobs are declared in `PROVIDER_KNOBS`).
Everything in §5 and §6 is analysis and proposal, not yet implemented.
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

- the session appends a tier-identity line to the **system block**, which
  is the root of every cached prefix (§5.1);
- the cache plugin is wired once and never re-wired or re-informed on a
  switch, so a cross-provider tier runs with no cache plugin at all
  (§5.2);
- Google's `CachedContent` is created bound to the boot model but its
  invalidation hash omits the model, so with tiers it is both re-billed
  on every switch and referenced against the wrong model (§5.3);
- per-turn cache figures *replace* rather than accumulate, so a turn
  containing a switch reports only its last leg (§5.4).

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

**The fix.** `JaatoSession._cache_plugin_config` now reproduces the
`ProviderConfig.extra` the *active* provider was built with: the runtime
base layer, then `plugin_configs[<provider>]` on top, child-wins, in the
same order `create_provider` uses. Two details are deliberate:

- The profile lookup is keyed on `_active_provider_name` (the name the
  provider was *created* under), not `provider.name`. They are not
  interchangeable — zhipuai subclasses the Anthropic provider and reports
  the parent's name — and only the creation name selects the right
  `plugin_configs` section.
- `api_key` is dropped, because `create_provider` promotes it to the
  top-level `ProviderConfig.api_key` field and it never lands in `extra`.
  The rule is "match the provider's own config view", not "guess at
  secrets", so `oauth_token` (which *is* left in `extra` there) passes
  through.

The cache knobs are now declared in both providers' `PROVIDER_KNOBS`, at
`top_level` — the position the read sites actually use. Regression
coverage: `shared/tests/test_cache_plugin_profile_knobs.py`.

> **Implication for anyone reading old measurements**: prior to this fix,
> any jaato session not on openrouter and without
> `JAATO_ANTHROPIC_ENABLE_CACHING` set was running with framework caching
> off. Cost baselines gathered before it are not comparable.

---

## 5. Open defects

### 5.1 The tier line mutates the system block

`_get_effective_system_instruction` (`shared/jaato_session.py:10141`)
appends `"You are currently operating in the \`<tier>\` tier."` to the
system prompt on every turn. The Anthropic provider folds
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

**Proposal**: remove the line from the system block. It is redundant —
`switch_tier` already returns `active_tier` in the tool result the model
reads. If a per-turn reminder is judged necessary, it belongs at the
*tail* of the message list (after the last cache breakpoint), never in
the prefix.

### 5.2 The cache plugin is never re-wired on a switch

`_wire_cache_plugin` is called from exactly one place —
`_ensure_provider` (`shared/jaato_session.py:2767`) — once per session.
`_provider_for_tier` (`:10160`) builds cross-provider tier providers and
never wires one.

- A **cross-provider tier** targeting anthropic or google_genai runs with
  no cache plugin for the rest of the session: caching silently off, no
  warning. (An openrouter tier is unaffected — it caches internally.)
- `AnthropicCachePlugin.set_model_name()` exists
  (`cache_anthropic/plugin.py:200`) and **has no caller anywhere**, so
  the minimum-cacheable-size threshold stays pinned to the boot model
  across every switch.

**Proposal**: call `_wire_cache_plugin()` from `_connect_tier_entry`,
with the plugin cached per provider name alongside `_provider_cache` so a
switch back is O(1); and push the new model name into the plugin on every
switch.

### 5.3 Google's CachedContent ignores the model

`GoogleGenAICachePlugin._model_name` is set only in `initialize()`
(`cache_google_genai/plugin.py:129`) and the `CachedContent` is created
bound to it (`:399`), while the invalidation hash covers system + tools
only (`_compute_content_hash`, `:360`) — **no model in the key**.

With tiers and `enable_caching: true`, every switch changes the system
text (§5.1), so the hash changes, so the plugin deletes a cache it has
already paid to create, creates another, and then hands the new name to a
request running against a *different* model. This was latent while §4
made the knob unreachable; the fix makes it live.

**Proposal**: include the active model in the content hash, and refuse to
emit a `cached_content` name whose bound model does not match the request's.

### 5.4 The miss is invisible

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

**Proposal**: accumulate cache tokens like every other dimension; add
`jaato.tier` and a tier-switch counter to the LLM span.

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
- [x] cache knobs declared in `PROVIDER_KNOBS` for anthropic + google_genai (§4)
- [ ] drop the tier line from the system block (§5.1)
- [ ] re-wire the cache plugin on tier switch; push the model name (§5.2)
- [ ] model in Google's `CachedContent` hash + mismatch guard (§5.3)
- [ ] accumulate cache tokens per turn; `jaato.tier` span attribute (§5.4)
- [ ] measure a real tiered session, then revisit §6
- [ ] first-class `cache:` profile field (§7)
