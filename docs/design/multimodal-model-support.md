# Multimodal Model Support — vision (and audio/…) via model-tier roles

**Status:** v1 **implemented** (PRs #297 capability primitive, #298 honest
provider vision tables, #299 `vision` tier + content gate, #300 config-time
vision-tier validation). Scope: **v1 = input vision via the tier system**;
output/generation and audio/PDF breadth are explicit later scopes (§9).

## Goal

Let a jaato agent actually *use* a multimodal model — to start, **see images**
(diagrams, screenshots, PDFs). The design principle is **multimodal by
composition**: you don't make every provider multimodal; you compose a
multimodal agent from role-specialized providers via the existing
**model-tier** mechanism. A text-only executor (even a local model) plus a
`vision` tier mapped to a vision-capable provider = a vision-capable agent, with
the executor's provider untouched.

## Where jaato is today (grounded)

| Piece | State |
|-------|-------|
| Type model (`Part.inline_data {mime_type,data}`, `ToolResult.attachments`) | ✅ already models arbitrary binary content |
| Image-input conversion in adapters | 🟡 4/13 providers (anthropic, google_genai, antigravity, openrouter); the OpenAI-compat + local fleet = 0 |
| Input source | 🟡 `readFile` MIME-detects images → multimodal content. No paste/URL/drag-drop |
| Modality breadth | 🔴 image-only (mostly PNG on the wire); no PDF/audio/video conversion |
| Output (model→media) | 🔴 `ProviderResponse` carries no model-generated media |
| Capability awareness | 🔴 no `supports_vision`/modality declaration — the framework can't tell which model accepts images |
| Multi-model-by-role | ✅ `model_tiers.py`: `ModelTierConfig`, tiers `{planner,dispatcher,executor}` + reserved `initial`/`fallback`, each a model-name string or `{"model","provider"}`, switched mid-session via the `enter_tier` lifecycle tool. **V1 invariant: all tiers same provider** (`_validate_same_provider_v1`) |

The plumbing types and the role/tier mechanism **exist**; the gaps are
capability-awareness, honest gating, and wiring modality content to the
role provider.

## Approach — modality roles in the tier system ("Pattern 1")

1. **Add `vision` (later `audio`, …) to `VALID_TIER_NAMES`** (`model_tiers.py`).
   The `ModelTierConfig` validator already gates the profile's `model_tiers`
   dict on this set. The dict is **single-level** — tier→model entries
   (keys in `VALID_TIER_NAMES`) mixed with the reserved control keys `initial`
   and `fallback`. Each tier value is either the **shorthand** (a model-name
   string, which resolves to the session's main provider) or the **rich form**
   `{"model": ..., "provider": ...}`. So a profile on OpenRouter can declare:

   ```yaml
   provider: openrouter
   model_tiers:
     executor: openai/gpt-5-mini              # text-only is fine
     vision:   google/gemini-3-pro            # OpenRouter vision model
     initial:  executor
     fallback: executor
   ```

   Both tiers use the shorthand, so both resolve to the profile's `openrouter`
   provider — **same-provider, valid under V1** (see *V1 same-provider
   constraint* below). The rich form is only needed when you want to pin a tier
   to an explicit provider; under V1 that provider must still match the others:

   ```yaml
   provider: openrouter
   model_tiers:
     executor: {model: openai/gpt-5-mini,   provider: openrouter}
     vision:   {model: google/gemini-3-pro, provider: openrouter}
     initial:  executor
     fallback: executor
   ```

2. **Per-active-model capability** — a `modalities()` set on the provider,
   resolved for the *currently-connected* model exactly like
   `get_context_limit()` (see §Capability). `supports_modality("image")` =
   `"image" in modalities()`.

3. **The agent switches to the role for the modality** — `enter_tier("vision")`
   (the existing tool) to view an image, then `enter_tier(executor)` back.
   Single `_active_tier`, so this is a brief whole-session swap; fine because
   vision-tier models are usually capable generalists.

4. **Content already flows to the active-tier provider** — `readFile`'s
   `Part.inline_data` reaches whichever provider the active tier resolved to;
   the 4 vision adapters already convert it. So once in the `vision` tier, it
   just works.

## V1 same-provider constraint (the scope boundary)

`model_tiers.py` enforces a hard **V1 invariant**: *all tiers must use the same
provider* (`ModelTierConfig._validate_same_provider_v1`, raising
`ModelTierConfigError` "V1 supports only same-provider tier switching"). The
per-tier `provider` field is **forward-compat for V2** — present in the schema
(`TierEntry.provider`) but rejected at construction if tiers disagree.

This is not an obstacle for v1 multimodal — it *defines the v1 shape*:

- **In scope (v1): multimodal within one gateway provider.** A provider that
  serves both text-only and vision models behind one endpoint — **OpenRouter**
  is the canonical case (its catalog carries 339 models, vision and text-only
  side by side) — composes a multimodal agent with **zero constraint change**.
  `executor: openai/gpt-5-mini` + `vision: google/gemini-3-pro`, both on
  `openrouter`, is same-provider-valid today. The agent `enter_tier("vision")`s
  to a vision model and back, all within the one provider the session connected.
- **Out of scope (V2): cross-provider tiers.** A *local* text executor (vLLM /
  Ollama) plus a *remote* vision tier (Anthropic / Google) is the genuinely
  cross-provider case, and it's exactly what `_validate_same_provider_v1`
  forbids. Lifting it is a tracked V2 follow-on (drop the validate call + add
  provider-swap handling at the session layer, per `model_tiers.py` lines
  22-27) — orthogonal to this feature's capability/gating work and deferred.

So v1 ships multimodal-by-composition **inside a gateway provider**; the
cross-provider lift rides the existing V2 tier roadmap, not this design.

## Capability — `resolve_modalities` (sibling of `resolve_context_window`)

Multimodality is **per-model** (a gateway serves both vision and text-only
models), so capability answers for the *active* model, resolved by the same
precedence as the context-window work — a sibling helper in
`shared/plugins/model_provider/base.py`:

```python
def resolve_modalities(
    *,
    detect: Optional[Callable[[], Optional[Set[str]]]] = None,  # catalog/endpoint
    profile_value: Optional[Iterable[str]] = None,              # plugin_configs knob
    table_value: Optional[Iterable[str]] = None,                # static per-model map
) -> Optional[Set[str]]:
    """Resolve a model's INPUT modality set ({"text","image",...}).
    Precedence: detect (live, authoritative, self-updating) → profile knob →
    static table → None.  None = "unknown" — the caller (tier validation /
    content gate) raises rather than guessing.  No hardcoded fallback."""
```

**Tiers, per provider:**

1. **Detect (catalog/endpoint) — tier 1, authoritative + self-updating.**
   **Verified:** OpenRouter's public `GET /api/v1/models` carries
   `architecture.input_modalities` *and* `output_modalities` for **all 339**
   models (`["image","text"]` for vision, `["text"]` for text-only). Likely also
   github_models (its catalog already gives `context_window`) and Gemini's
   models API. These get modality **nearly free** — same catalog fetch already
   used for context length.
2. **Profile knob — tier 2, explicit override / escape hatch.**
   `plugin_configs.<provider>.modalities: ["text","image"]` for a model the
   catalog doesn't list yet, or to assert/correct.
3. **Static per-model table — tier 3, documented constants.** A `MODEL_VISION`
   map (mirrors `MODEL_CONTEXT_LIMITS`) for closed providers with no live
   modality endpoint — anthropic (claude-3+), some Gemini, gpt-4o, etc.
4. **Unknown → fail-loud** (§Validation / §Gating). Never a silent guess.

Provider mapping: OpenRouter/github_models → detect; anthropic/google → static
table; local/self-hosted (vllm/ollama/lmstudio/nim/tensorrt) → knob (operator
knows their engine); fail-loud otherwise.

**Output bonus:** OpenRouter's `output_modalities` is right there, so the detect
tier built now is *also* the foundation the future generation path (§9) queries.
One investment, two scopes.

## Provider contract

On `ModelProviderPlugin` (base.py), alongside `supports_thinking()` /
`get_context_limit()`:

```python
def modalities(self) -> Set[str]:
    """INPUT modalities the CURRENTLY-CONNECTED model accepts
    ({"text"} at minimum).  Resolved via resolve_modalities(...).  Raises a
    provider 'not configured' error only when consulted for an unresolved
    model (mirrors get_context_limit)."""

def supports_modality(self, kind: str) -> bool:
    return kind in self.modalities()
```

A `modalities()` **set** (not a bare `supports_vision` bool) future-proofs
audio/video/PDF with zero new methods — the gate, the validator, and the
agent-error message all read the same set.

## Config validation (fail-loud at config, not at first image)

When `ModelTierConfig` resolves a `vision` tier to a `{model, provider}`, the
framework verifies `provider.supports_modality("image")`. If it can't be
confirmed (modalities unresolved / no `image`), session creation **fails loud**:

> `vision` tier maps to `<model>` (`<provider>`), which does not declare image
> input. Map it to a vision model, or set
> `plugin_configs.<provider>.modalities: ["text","image"]` to assert it.

## Content-boundary gate (the load-bearing piece)

The correctness of the whole feature hinges here. When a turn carries
`Part.inline_data` of an image and the **active** tier's provider lacks `image`
in `modalities()`, the framework returns a **clear, actionable error** instead
of sending bytes a model can't see:

> This message contains an image, but the active `executor` model
> (`<model>`) can't view images. Call `enter_tier("vision")` first, then retry.

This converts the agent's mistake (readFile-an-image while in a text-only tier)
from a silent failure into a loud, self-correcting signal — same philosophy as
the recovery-event work. Gate location: the session's send path, right before
history→provider conversion (where the active provider + the outgoing `Part`s
are both in scope).

## Agent guidance

`enter_tier`'s tool description + the tier system-instructions teach: *"To view
an image or PDF, call `enter_tier("vision")` first; switch back with
`enter_tier("executor")` when done."* The content-gate error (above) is the
backstop when the agent forgets.

## Decisions (resolved for v1)

- **Single tier-set, not a separate axis.** `vision` joins
  `VALID_TIER_NAMES`; `_active_tier` stays single-valued, so `vision` and
  `executor` are mutually exclusive. Fine because vision-tier models are
  usually capable generalists. A two-axis active state (cognitive tier *and*
  modality role simultaneously) is a bigger change — deferred until
  "weak-executor + vision-glance" is a real need.
- **Content-to-wrong-tier → error, not auto-switch.** Keeps v1 agent-driven and
  predictable; auto-routing is Pattern 3 (§9), layerable later on the same
  capability flag with no rework.
- **Capability is per-active-model** (method on the provider, resolved like
  `get_context_limit`), not a global per-provider flag — gateways serve mixed
  fleets.

## Non-goals (explicit later scopes)

- **Output / generation** (model emits an image/audio) — `ProviderResponse`
  carries no model-generated media; adapters don't parse it; clients don't
  render it. Separate scope (Scope B). The detect tier here is its foundation.
- **Modality breadth** — PDF/audio/video converters in whichever providers fill
  those roles. (`modalities()` already generalizes; the *converters* don't.)
- **Ingestion UX** — paste/URL/drag-drop is client-side, downstream of this.
- **Backfilling image conversion to all 13 providers** — unnecessary under
  composition; only the providers chosen to *fill* a modality role need it.
- **Cross-provider tiers** (text executor on provider A + vision tier on
  provider B) — blocked by the V1 same-provider invariant
  (`_validate_same_provider_v1`); it's the existing V2 tier-roadmap item, not
  this feature. v1 multimodal lives *inside* one gateway provider (§*V1
  same-provider constraint*).

## Test plan

- `resolve_modalities` precedence (detect > knob > table > None) — unit.
- `modalities()` per provider: OpenRouter detect (catalog), anthropic static
  table, a knob override; unknown → raises.
- `ModelTierConfig`: a `vision` tier mapped to a text-only model fails config
  validation with the actionable message; a vision model passes.
- Content gate: an image `Part` to a non-vision active provider → the
  enter_tier-guidance error; to a vision active provider → passes through.
- `enter_tier("vision")` switches the active provider and the image converts.

## Open questions

1. **`modalities()` return shape** — bare `Set[str]` of input modalities, or a
   richer `{"input": {...}, "output": {...}}` (carries OpenRouter's
   output_modalities for Scope B from day one)? Leaning richer, input-gated now.
2. **vLLM/local detect** — can the served model's HF config reliably flag
   multimodal architecture, or do local providers stay knob-only? Needs a probe.
3. **Where the static `MODEL_VISION` table lives** — per-provider (like
   `MODEL_CONTEXT_LIMITS`) vs a shared map. Per-provider matches the existing
   pattern.

## Decision log

- **Tiers over per-provider-multimodality** — composition (text executor +
  vision tier) beats forcing every adapter multimodal; reuses the proven
  `model_tiers` + `enter_tier` mechanism.
- **`resolve_modalities` mirrors `resolve_context_window`** — same per-model
  metadata problem; reuse the precedence, no new architecture; catalog
  providers come nearly free.
- **Fail-loud on unknown / wrong-tier** — no silent send of unviewable bytes;
  the agent's mistake is surfaced and self-correcting (recovery-event
  philosophy, no-hardcoded-fallback rule).
- **v1 scoped to within-gateway composition** — the V1 same-provider invariant
  (`_validate_same_provider_v1`) makes "OpenRouter text executor + OpenRouter
  vision tier" the v1 shape with zero constraint change. Cross-provider tiers
  (local executor + remote vision) ride the existing V2 tier lift, keeping this
  design orthogonal to that roadmap item.
