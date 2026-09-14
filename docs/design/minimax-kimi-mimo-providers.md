# MiniMax, Kimi and MiMo providers — brainstorm and design

Status: **implemented on this branch** (seam, then `mimo`, `kimi`, `minimax`,
each its own commit); §10 records where the code diverged from the design.
Written 2026-09-08 against the tree at `7306154` and the three vendors' public documentation as of that date
(sources in §11). Every wire fact below that came from a vendor page rather
than a live probe is marked *(docs)*; the few that were probed are marked
*(probed)*.

## 0. Summary

Three first-party providers, all on the vendors' **OpenAI-compatible** chat
surface, all thin subclasses of `_openai_compat.OpenAICompatProvider` in
the shape `doubleword` / `nebius` / `zhipuai_openai` already have:

| Provider dir | Vendor | Base URL (intl) | Current models | Catalog reports context? |
|---|---|---|---|---|
| `minimax` | MiniMax (platform.minimax.io) | `https://api.minimax.io/v1` | `MiniMax-M3`, `MiniMax-M2.7[-highspeed]` (+ legacy M2 / M2.1 / M2.5) | no → table + knob |
| `kimi` | Moonshot AI (platform.kimi.ai) | `https://api.moonshot.ai/v1` | `kimi-k3`, `kimi-k2.7-code[-highspeed]`, `kimi-k2.6` | **yes** (`context_length` + modality flags) |
| `mimo` | Xiaomi MiMo (mimo.mi.com) | `https://api.xiaomimimo.com/v1` | `mimo-v2.5-pro`, `mimo-v2.5` | no → table + knob |

They share one prerequisite the framework does not have today and which
none of the three can be used seriously without: **reasoning replay**.
Every thinking model in the three lineups asks the client to send the
assistant's `reasoning_content` back on the next request of a tool-call
loop. MiMo returns `400` when it is missing; Kimi K3's docs say the
assistant message must go back "as-is, including `reasoning_content` and
`tool_calls`"; MiniMax measured an 87 → 64 drop on Tau² without it. The
session today **drops thinking from history** (`_add_model_response_to_history`
keeps only text and function-call parts) and every OpenAI-shaped
converter **ignores** `Part.thought` on replay. §3 designs that seam once,
framework-side; the three providers then opt into it with one class
attribute each.

Recommended order: the replay seam first (one PR, tested against a fake
upstream), then the three providers, each its own PR, MiMo first because it
is the one whose API *refuses* to work without replay and therefore proves
the seam on a real wire.

## 1. Why first-party providers at all

All three lineups are already reachable through providers in the tree:

- **OpenRouter** lists `minimax/minimax-m3`, `moonshotai/kimi-k3`,
  `xiaomi/mimo-v2.5-pro` and their siblings, and the `openrouter` provider
  already handles `reasoning` / `reasoning_content` extraction, `tool_choice`
  forwarding, cache accounting and catalog context windows.
- **vLLM / SGLang** serve the open weights (MiniMax-M3, Kimi K2.x, every
  MiMo model is MIT on Hugging Face) and the `vllm` provider is passive over
  any of them; MiMo needs `--reasoning-parser mimo --tool-call-parser mimo`
  (aliases of the Qwen3 parsers in vLLM ≥ 0.21), which is a documentation
  item, not code.

What a first-party provider adds, and what justifies three directories:

| Concern | Through OpenRouter / vLLM | First-party |
|---|---|---|
| Vendor keys and plans | not usable: OpenRouter bills its own key; the vendors' subscription keys (MiniMax Token Plan `sk-cp-…`, Kimi Code `api.kimi.com/coding`, MiMo Token Plan `tp-…`) only work against the vendor's hosts | `base_url` + key, the `*-auth` plugin, region hosts (`.cn`) |
| Thinking control | OpenRouter's generic `reasoning` object; the vendor-specific toggles (`thinking: {type}`, `reasoning_effort` low/high/max, `reasoning_split`) are not all mapped | the exact wire field per model family |
| Reasoning replay | OpenRouter replays `reasoning_details` when *it* is the client; our converter drops it either way (§3) | the seam in §3, exercised on the vendor wire |
| Catalog | OpenRouter's | Kimi's own catalog is richer than OpenRouter's for Kimi (per-model `supports_image_in` / `supports_video_in` / `supports_reasoning`) |
| Error taxonomy | OpenRouter's flattened errors | Kimi's three-way `429` (overloaded / rate-limited / **quota exhausted, not retryable**), MiniMax's numbered codes, MiMo's `421` content filter |
| Cost | OpenRouter margin | vendor list price; cache-hit input at 1/10 (MiniMax), 1/10 (Kimi), 1/100 (MiMo) |

The `zhipuai` precedent applies directly: Z.AI is also on OpenRouter and
still earned a first-party provider for exactly these reasons.

## 2. The one decision that shapes all three: which surface

Each vendor exposes three surfaces: OpenAI Chat Completions, an Anthropic
Messages shim (for Claude Code), and an OpenAI Responses API. The tree has
two bases to inherit from — `OpenAICompatProvider` (chat completions) and
`AnthropicProvider` (as `zhipuai` and `ollama` do).

**Decision: OpenAI Chat Completions, one provider per vendor.** Reasons:

1. **The Anthropic shims are the weaker surfaces.** MiniMax's `/anthropic`
   rejects `Authorization: Bearer` and wants `X-Api-Key` only *(probed)*,
   and ignores `top_k` / `stop_sequences`; Kimi's accepts no `cache_control`
   and enumerates only `kimi-k3`; MiMo's ignores `cache_control` and
   `thinking.budget_tokens` and reports no cache usage. The OpenAI surface is
   the documented primary for Kimi and MiMo, and MiniMax's "recommended"
   Anthropic label is about Claude Code, not about capability.
2. **The subscription keys do not need the Anthropic surface.** MiniMax's
   Token Plan key is used exactly like an API key on `/v1` *(docs: Zed,
   OpenCode, LangChain examples)*; Kimi Code exposes
   `https://api.kimi.com/coding/v1` (OpenAI-shaped, its own model ids);
   MiMo's Token Plan hosts serve `/v1`. So a plan user is a `base_url` knob
   away, not a second provider away. (Kimi Code's terms forbid spoofing the
   client User-Agent; jaato sends its own identity — see the
   `AppIdentity` doc — and must not pretend to be Claude Code.)
3. **The framework's Anthropic converter has the same replay gap.**
   `anthropic/converters.py` returns `None` for `thinking` blocks, so the
   Anthropic path would need its own replay work (with `signature`
   round-tripping) on top of §3. Not worth carrying twice.
4. **Sharing the streaming loop is how the guards stay honest.**
   `_openai_compat/base.py` stands in for eight providers in
   `test_a_dead_stream_is_not_a_finished_turn` and
   `test_truncation_is_not_reported_as_tool_use`; three more inheritors add
   zero new end-of-stream decision points.

A `*_anthropic` sibling (the `zhipuai` / `zhipuai_openai` pattern, in
reverse) can be added later if a Claude-Code-parity use case appears. Not
in scope.

The Responses API is out of scope for the same reason it is everywhere
else in the tree: nothing inherits from a Responses-shaped base.

## 3. Prerequisite: reasoning replay (interleaved thinking)

### 3.1 The gap, precisely

Today the OpenAI-compat streaming loop accumulates `delta.reasoning_content`
into `ProviderResponse.thinking` (a string the session forwards to
`on_output("thinking", …)`) and **not** into `response.parts`. Then:

```python
# jaato_session.py — _add_model_response_to_history
history_parts = [p for p in response.parts
                 if p.text is not None or p.function_call is not None]
```

so even a provider that emitted `Part(thought=…)` (google_genai does, in
one path) has it stripped before the message reaches history. And on the
way back out, `_openai_compat/converters.message_to_openai` reads only
`text`, `function_call` and `function_response`; `_prose_tools` says it
out loud: `elif part.thought is not None: continue  # internal reasoning is
never replayed to the model`.

That was the right default for DeepSeek-R1-era models, where the reasoning
of a finished turn is noise. It is wrong for these three:

| Vendor | Rule *(docs)* | Consequence of dropping |
|---|---|---|
| MiMo (`mimo-v2.5*`) | "must completely pass back the `reasoning_content` field … or the API returns a 400 error" | hard failure on the second request of every tool loop (corroborated by XiaomiMiMo/MiMo#44) |
| Kimi (`kimi-k3`, `kimi-k2.7-code`, `kimi-k2.6` with `keep: "all"`) | "requires the complete assistant message returned by the API to be passed back to `messages` as-is, including `reasoning_content` and `tool_calls`" | K2.6 with `keep: null` ignores *historical* reasoning but still wants the current turn's during a tool loop |
| MiniMax (`M2.x` always-on, `M3` adaptive) | "the complete model response (i.e., the assistant message) must be appended to the conversation history"; with `reasoning_split: true` keep `reasoning_details` | quality: Tau² 87 → 64 |

### 3.2 The seam — five touches, none provider-specific

1. **The streaming loop emits a thought Part.** In
   `OpenAICompatProvider._stream_completion` (and the non-streaming
   `response_from_openai`), when `self.replay_reasoning` is true, prepend
   `Part(thought="".join(accumulated_thinking))` to `parts` in addition to
   setting `ProviderResponse.thinking` (which the UI keeps reading). Thought
   first, text/tool calls after — the vendors emit reasoning before content
   and replay it in that order.
2. **The session keeps it, gated by the provider.**
   `_add_model_response_to_history` keeps `p.thought is not None` parts iff
   `getattr(self._provider, "replay_reasoning", False)`. Gating on the
   provider, not on the part, means google_genai's incidental thought parts
   keep being dropped and no existing provider changes behaviour.
3. **The converter replays it.** `message_to_openai` gains a
   `reasoning_fields: Optional[Callable[[str], Dict[str, Any]]]` argument
   (threaded through `history_to_openai`); when a `Role.MODEL` message
   carries thought parts and the callable is set, the assistant dict is
   updated with its result. The default callable is
   `lambda t: {"reasoning_content": t}`. Per-vendor deviations are one
   override each: MiniMax adds
   `reasoning_details: [{"type": "reasoning.text", "id": "reasoning-text-1",
   "format": "MiniMax-response-v1", "index": 0, "text": t}]`.
   An assistant message whose only parts are thought + tool calls must go
   out as `content: ""` rather than `content: None` — MiMo#44's failing
   payload had `null`, and `""` is what the vendors' own examples send.
4. **GC sees it.** `estimate_message_tokens` in the gc plugins must count
   `part.thought` the way #850 taught it to count `inline_data`; a Kimi K3
   turn at `reasoning_effort: max` can carry tens of thousands of reasoning
   tokens that are now *replayed context*, and a threshold that cannot see
   them is the #850 bug again in text. `gc_summarize` / `gc_hybrid` drop
   thought parts from summarised turns (a summary of the reasoning is the
   summary); `gc_truncate` keeps whatever it keeps.
5. **Persistence already works.** Every converter's `serialize_message` /
   `deserialize_message` round-trips `{"type": "thought"}` parts, so a
   revived session (#787) replays the same reasoning it would have replayed
   live. `_gate_history_for_active_modalities` (#847) is text-only for
   thought parts and needs no change.

`replay_reasoning` is a class attribute on `OpenAICompatProvider`, default
`False`; the three new providers set it `True`. `openrouter` does not
inherit the base and gets nothing here (it would need the same three
touches plus OpenRouter's `reasoning_details` shape; a follow-up, not a
blocker).

### 3.3 Should this be a declared capability?

Yes, but as a follow-up field: `ProviderCapabilities.reasoning_replay`
("assistant reasoning from history reaches the wire on the next request").
Adding a field fails every provider's structural guard until each declares
it (§"Adding a capability" in `provider-capability-contract.md`), which is
a mechanical but 18-file change; it should ride the seam PR so that the
doc table shows exactly which wires replay. The conformance guard can
assert it behaviourally the same way it asserts images: build a `MODEL`
message with a thought part, run it through the converter, require the
text to appear under `reasoning_content`.

### 3.4 What it costs, and why it is still right

Replaying reasoning grows the prompt on every tool-loop turn. Three things
bound it:

- **All three vendors auto-cache prefixes** (§4–6), and a replayed
  assistant message is a stable prefix by construction — the cache-hit rate
  goes *up* with replay, and cache-hit input is priced at 1/10 to 1/100 of
  a miss.
- **GC counts it** (touch 4), so the budget strategy fires on it.
- **Kimi K2.6 has a server-side opt-out** (`thinking.keep: null` — the
  server ignores historical reasoning); the provider exposes it as
  `api_params.thinking_keep` and the default follows the vendor default.

The alternative — replaying only the *most recent* assistant turn's
reasoning, which is the minimum MiMo needs to not 400 — is tempting and
wrong: Kimi says the whole history "as-is", MiniMax's measured degradation
is over the loop, and a half-replay is exactly the kind of quiet
divergence #787 was about.

## 4. `minimax`

### 4.1 Identity, endpoints, auth

| | |
|---|---|
| Provider name | `minimax` |
| Base URL | `https://api.minimax.io/v1` (intl). China: `https://api.minimax.cn/v1` (`api.minimaxi.com` serves the same routes *(probed)*). Keys are **region-bound**: a China key is `401` on `.io`. Knob `base_url`, env `JAATO_MINIMAX_BASE_URL`. |
| Auth | `Authorization: Bearer <key>` on `/v1` (the SDK default). |
| Env | `JAATO_MINIMAX_API_KEY`, then the ecosystem's `MINIMAX_API_KEY` (litellm, OpenClaw). No Group ID anywhere in current docs — do not add one. |
| Stored | `minimax-auth key <key>` → `minimax_auth.json` (config_root → workspace → `~/.jaato`), validated by a one-token chat call (the `doubleword-auth` pattern; `GET /v1/models` is Bearer too and cheaper — prefer it). |
| Plan keys | Token Plan subscription keys (`sk-cp-…`) work on `/v1` unchanged. Exhaustion is `HTTP 429`, code `2056`, message carrying an ISO reset time → `RateLimitError(retry_after=<reset − now>)`. |

### 4.2 Models and context

`GET /v1/models` exists (Bearer) and returns bare `{id, object, created,
owned_by}` *(docs)* — no window. So: the `doubleword` catalog lookup kept
for the day it is enriched, a `MODEL_CONTEXT_LIMITS` table (longest-prefix,
the `zhipuai_openai` shape) as the working tier, `plugin_configs.minimax.
context_length` / `JAATO_MINIMAX_CONTEXT_LENGTH` as the override, and
fail-loud for an id none of them names.

| Model | Context | Max output (recommended) | Thinking | Vision |
|---|---|---|---|---|
| `MiniMax-M3` | 1,000,000 | 524,288 (131,072) | `thinking: {type: adaptive \| disabled}`; **default ON on `/v1`** | image + video |
| `MiniMax-M2.7`, `-highspeed` | 204,800 | 204,800 (65,536) | always on, cannot disable | — |
| `MiniMax-M2.5`, `M2.1`, `M2` (+`-highspeed`) | 204,800 | same | always on | — |

`REASONING_CAPABLE_MODELS = ["minimax-m"]` (prefix, case-insensitive).
`modalities()`: `{text, image}` for `MiniMax-M3` (video is declared by the
vendor but no wire in the tree carries video; declaring it would trip the
cross-rule), `{text}` otherwise, `modalities` knob layered on top.

### 4.3 Thinking control

- **Always send `reasoning_split: true`** (an `extra_body` field). Without
  it reasoning arrives *inside* `content` as `<think>…</think>`, which
  would pollute `AGENT_OUTPUT` and be replayed as text. With it, streaming
  delivers `delta.reasoning_details[]` (and, per some reports, no
  `delta.reasoning_content`) — so the loop's reasoning extraction becomes a
  hook, `_reasoning_from_delta(delta) -> Optional[str]`, that the base
  implements as `reasoning_content` and MiniMax overrides to also read
  `reasoning_details[].text`.
- `enable_thinking` (framework convention) → `thinking: {type: "disabled"}`
  when `False` **and** the model is M3; on M2.x a `False` is logged and
  ignored (the vendor cannot turn it off). `thinking_level` is not a
  MiniMax concept on chat completions (only on Responses) → rejected with a
  clear error rather than silently dropped.
- Defensive: a `strip_think_tags` pass on assembled text, on by default,
  for the M2.x paths that still leak `<think>` into `content`.
- Known wire quirk *(docs, MiniMax-M2.5#2)*: chunks with
  `"delta": {"role": "", "reasoning_content": ""}`. The OpenAI SDK's
  models are constructed without validation, so this is harmless to us;
  pin it with a test so an SDK upgrade that starts validating is caught.

### 4.4 Tool calling

- `tool_choice`: only `none` / `auto` documented. Forward those; map
  `required` / a named choice to `auto` with a WARNING naming the model
  (never a silent drop, never a 400). Declare
  `tool_choice_forwarding=True` — the parameter *is* forwarded; the
  restriction is a vocabulary the vendor imposes, as with Kimi K2.x.
- `parallel_tool_calls` undocumented but multi-call responses are normal
  — no special handling; the base already emits one `tool` message per
  result.
- **Always send `max_completion_tokens`** (`max_tokens` is deprecated on
  this surface): the vendor's default when omitted is small and truncates
  tool-call JSON *(hermes-agent#37151)*. `api_params.max_tokens` is renamed
  at apply time; when the profile sets nothing, the provider sends the
  model's recommended value from the table above.
- `n` must be 1; `presence_penalty`, `frequency_penalty`, `logit_bias`
  unsupported → dropped from `_FORWARDED_API_PARAMS` (a profile naming them
  gets the base's "ignoring unsupported key" warning). `temperature` 0–2.
- `service_tier: "priority"` is 1.5× price — allowlisted like doubleword's.
- `response_format` is silently ignored upstream for M2.x → not
  allowlisted; a profile asking for JSON mode should hear that from us.

### 4.5 Caching, errors, capabilities

- Caching is **automatic** (≥ 512 tokens), reported in
  `usage.prompt_tokens_details.cached_tokens` — the base's
  `_extract_cache_tokens` already reads it. `prompt_caching=False` per the
  contract (no breakpoints emitted).
- Errors: real HTTP status with an OpenAI-ish envelope and the MiniMax
  numeric code appended to the message *(probed)*: `1004`/`2049` auth,
  `1002`/`1041`/`2045` rate limit, `1008` balance, `1039` token limit,
  `2056` plan window, `1026`/`1027` sensitive content. The base maps by SDK
  exception class; `minimax/errors.py` refines `RateLimitError.retry_after`
  from the `2056` reset time and classifies `1008` (no balance) as
  non-transient. No `Retry-After` header was seen *(probed)*.
- `PROVIDER_CAPABILITIES`: images ✅ (M3), pdf —, audio —,
  tool_choice ✅, thinking ✅, prompt_caching —, streaming ✅,
  cancellation ✅, output_media ✅ (inherits the wired loop, as
  doubleword declares).

### 4.6 Knobs

```yaml
plugin_configs:
  minimax:
    api_key: pass://minimax/key        # top-level; JAATO_MINIMAX_API_KEY fallback
    base_url: https://api.minimax.cn/v1   # region / plan hosts
    context_length: 1000000           # override when the table lacks the model
    modalities: [text, image]         # assert; M3 is table-detected
    api_params:
      max_tokens: 131072              # sent as max_completion_tokens
      temperature: 1.0
      top_p: 0.95
      tool_choice: auto
      service_tier: priority
      enable_thinking: true           # M3: adaptive|disabled; M2.x: ignored with a log
    extra_body: {}                    # reasoning_split is set by the provider, not here
```

## 5. `kimi`

### 5.1 Identity, endpoints, auth

| | |
|---|---|
| Provider name | `kimi` (the brand users look for; the company is Moonshot). Directory `kimi`, class `KimiProvider`. |
| Base URL | `https://api.moonshot.ai/v1` (intl), `https://api.moonshot.cn/v1` (China). The docs moved to platform.kimi.ai; the API hosts did not. Kimi Code plan: `https://api.kimi.com/coding/v1` with its own ids (`k3`, `k3-256k`, `kimi-for-coding[-highspeed]`). Knob `base_url`, env `JAATO_KIMI_BASE_URL`. |
| Auth | `Authorization: Bearer <key>` (the only scheme in the OpenAPI spec, Anthropic surface included). |
| Env | `JAATO_KIMI_API_KEY`, then the vendor's own `MOONSHOT_API_KEY` (the name in every official snippet and the Vercel SDK). |
| Stored | `kimi-auth key <key>` → `kimi_auth.json`; validate with `GET /v1/users/me/balance` (cheap, authenticated, and it also tells the user their balance — K3 is unlocked only after a ≥ $1 top-up). |

### 5.2 Models and context — the best catalog of the three

`GET /v1/models` returns `context_length`, `supports_image_in`,
`supports_video_in`, `supports_reasoning` per model *(docs, spec)*. So Kimi
gets the **nebius** shape: catalog PRIMARY at `connect()` for the window
*and* for modalities (`supports_image_in` → `image`; video is declared but
not carried — see §4.2), then the `context_length` / `modalities` knobs,
then fail-loud. No table except a comment listing today's ids.

| Model | Context | Max output (default) | Thinking | Sampling | `tool_choice` |
|---|---|---|---|---|---|
| `kimi-k3` | 1,048,576 | up to 1,048,576 (131,072) | always on; `reasoning_effort: low \| high \| max` (default max) | **fixed** | auto / none / required / named |
| `kimi-k2.7-code`, `-highspeed` | 262,144 | 32,768 | always on; `thinking.type` must be `enabled`, `keep` forced `all` | fixed | auto / none |
| `kimi-k2.6` | 262,144 | 32,768 | `thinking: {type: enabled \| disabled, keep: "all" \| null}` | fixed per mode | auto / none while thinking |

Every pre-K2.6 id (`moonshot-v1-*`, `kimi-latest`, `kimi-k2*`,
`kimi-k2.5`) returns `404 resource_not_found_error` as of 2026-08-31 — the
provider hard-codes nothing about them. `REASONING_CAPABLE_MODELS =
["kimi-"]`; `supports_thinking()` reads the catalog's `supports_reasoning`
when the model is listed.

### 5.3 Sampling parameters are a 400

`temperature`, `top_p`, `n`, `presence_penalty`, `frequency_penalty` are
**absent from the request schema**; sending them is rejected *(docs)*. So
`_FORWARDED_API_PARAMS` for Kimi is the narrow set the spec names:
`max_tokens` (renamed `max_completion_tokens`), `tool_choice`, `stop`
(≤ 5 strings ≤ 32 bytes), `response_format` (`text` / `json_object` /
`json_schema`), `prompt_cache_key`, `reasoning_effort`, plus the media
params the base carries. A profile that sets `temperature` gets the base's
"ignoring unsupported key" warning — which is the truth, and cheaper than
a 400 on every turn.

### 5.4 Thinking control — two dialects, one framework knob

- `kimi-k3`: `thinking_level` (framework convention, cf. openrouter /
  antigravity) → `reasoning_effort`, vocabulary `low | high | max`
  (validated; `medium` is not a K3 value). `enable_thinking: false` is
  impossible on K3 → logged and ignored. Changing effort mid-session breaks
  the prefix cache *(docs)* → `set_thinking_config` at runtime logs that.
- `kimi-k2.6`: `enable_thinking` → `thinking: {type: enabled|disabled}`;
  `thinking_keep` (`"all"` | `null`, default vendor's `null`) → `thinking.keep`.
- `kimi-k2.7-code`: `thinking: {type: enabled, keep: "all"}` is the only
  legal shape; the provider sends it and ignores knobs that contradict it.
- `thinking` and `reasoning_effort` are non-standard body fields → carried
  via `extra_body`, merged with the profile's own `extra_body` (profile wins
  on a collision, and the collision is logged).
- Replay: §3, default `reasoning_content` shape.

### 5.5 Tool calling

- `tool_choice`: `required` / named **only on K3**, and a named choice is
  "incompatible with thinking (400)". The provider knows the model it is on:
  on K2.x, `required` / named → `auto` + WARNING (§4.4 rule).
- `strict` defaults to **true** on Kimi's tool schemas *(spec)*; our
  schemas are not authored for strict mode (no `additionalProperties:
  false`, permissive `required`). Send `"strict": false` explicitly on
  every tool definition unless `api_params.strict_tools: true` (the
  openrouter knob) says otherwise. This is the one converter-level
  deviation and it is a wrapper around `tool_schemas_to_openai`, not a
  fork of it.
- Tool names: `^[a-zA-Z_][a-zA-Z0-9-_]{0,127}$` — the hashed wire ids the
  framework already sends satisfy it.
- `$web_search` builtin and K3 dynamic tool loading (`{role: system,
  tools: […]}` mid-history) are vendor features with no framework
  counterpart; out of scope, noted in the provider docstring.
- Streaming terminates on `data: [DONE]`; the base already treats a usage
  frame and a named finish reason as terminal, and `stream_options.
  include_usage` is supported → set it.

### 5.6 Caching, errors, capabilities

- Caching is **automatic** (prior request > 256 prompt tokens). Hits are
  reported in **`usage.cached_tokens` at the top level**, not under
  `prompt_tokens_details` → override `_extract_cache_tokens`.
  `prompt_cache_key` improves hit rate and is "required for Kimi Code
  Plan" → when the profile sets none, the provider sends the session's
  agent id (`set_agent_context` already delivers it), so every session gets
  its own stable key without configuration.
- Errors *(docs)*: `429` splits three ways by `error.type`:
  `engine_overloaded_error` (honour `Retry-After`, transient),
  `rate_limit_reached_error` (transient), **`exceeded_current_quota_error`
  (balance — not transient)**. The base's `_handle_api_error` maps every
  `openai.RateLimitError` to a transient `RateLimitError`; Kimi's override
  reads `error.type` from the response body first and raises a
  non-transient `QuotaExhaustedError` (subclass of the auth error family so
  `with_retry` stops) for the third. `400 invalid_request_error` with
  "Input token length too long" / "prompt tokens + max_tokens exceeds" →
  `ContextLimitError` (the base already substring-matches; add these
  phrases). `504` after 900 s → the streaming default covers it.
- `PROVIDER_CAPABILITIES`: images ✅, pdf —, audio —, tool_choice ✅,
  thinking ✅, prompt_caching —, streaming ✅, cancellation ✅,
  output_media ✅.

### 5.7 Knobs

```yaml
plugin_configs:
  kimi:
    api_key: pass://kimi/key           # JAATO_KIMI_API_KEY → MOONSHOT_API_KEY fallback
    base_url: https://api.kimi.com/coding/v1   # Kimi Code plan (its own model ids)
    context_length: 262144             # only when the catalog lacks the model
    modalities: [text, image]
    api_params:
      max_tokens: 65536                # → max_completion_tokens; ≥16k advised for thinking
      tool_choice: auto
      thinking_level: high             # K3 → reasoning_effort
      enable_thinking: true            # K2.6 → thinking.type
      thinking_keep: all               # K2.6 → thinking.keep
      prompt_cache_key: my-task-42     # default: the session's agent id
      strict_tools: false
      response_format: {type: json_object}
```

## 6. `mimo`

### 6.1 Identity, endpoints, auth

| | |
|---|---|
| Provider name | `mimo` (directory `mimo`, class `MiMoProvider`); "Xiaomi" stays in the display name. |
| Base URL | `https://api.xiaomimimo.com/v1` (one host for both regions; keys and balances are per-region). Token Plan: `https://token-plan-{cn,sgp,ams}.xiaomimimo.com/v1` (`tp-…` keys, `401` on the pay-as-you-go host, and the plan's terms restrict them to "programming tools"). Knob `base_url`, env `JAATO_MIMO_BASE_URL`. **Not available in the EU, UK or Korea** *(docs)* — `403` there. |
| Auth | `Authorization: Bearer <key>` or `api-key: <key>`; use Bearer (SDK default). |
| Env | `JAATO_MIMO_API_KEY`, then the vendor's `MIMO_API_KEY`. |
| Stored | `mimo-auth key <key>` → `mimo_auth.json`; validate with `GET /v1/models` (authenticated). |

### 6.2 Models and context

`GET /v1/models` returns `{id, object, owned_by}` only *(docs)* — the
MiniMax situation, so the same shape: catalog lookup kept dormant, table
as the working tier, knob override, fail-loud.

| Model | Context | Max output (default) | Input | Thinking |
|---|---|---|---|---|
| `mimo-v2.5-pro` | 1,048,576 | 131,072 (131,072) | text | on by default |
| `mimo-v2.5` | 1,048,576 | 131,072 (32,768) | text, image, video, audio | on by default |
| `mimo-v2-flash`, `-pro`, `-omni` | — | — | — | **deprecated 2026-06-30**, auto-routed to v2.5; not in the table |

`modalities()`: `{text, image}` for `mimo-v2.5`, `{text}` for the pro.
The vendor accepts audio on `mimo-v2.5`, and the tree now has an
`input_audio` wire (#830, `audio_as_input_audio`) — but whether MiMo
accepts the OpenAI `input_audio` block shape is unverified, so `audio`
is a follow-up: probe it, then flip `audio_input=True` and the modality
in one change (the cross-rule forbids declaring one without the other).

### 6.3 Thinking control

- Toggle is `thinking: {type: enabled | disabled}` (non-standard, via
  `extra_body`), default enabled; **no** `reasoning_effort`, budget or
  `enable_thinking` on chat completions *(docs; Alibaba's hosted copy lists
  them as unsupported)*. `enable_thinking` maps to it; `thinking_level` /
  `thinking_budget` are rejected with a clear error.
- In thinking mode the vendor **forces** `temperature = 1.0`, `top_p =
  0.95` whatever is sent — forward the profile's values anyway (they apply
  when thinking is off) and say so in the knob description.
- Output is split into `reasoning_content` (base already extracts it);
  `usage.completion_tokens_details.reasoning_tokens` → `TokenUsage.
  reasoning_tokens`.
- **Replay is mandatory** (§3): missing `reasoning_content` on an assistant
  message that carries `tool_calls` is a `400 Param Incorrect` on the next
  request.
- Vendor-admitted instability: the model sometimes emits tool calls
  *inside* the reasoning and returns `tool_calls: null`; the FAQ's advice is
  to disable thinking for tool-heavy work. That is a profile decision, and
  the provider docstring says so; the framework's existing "text-only
  reply, no tool call" nudge handles the symptom.

### 6.4 Tool calling

- `tool_choice`: **`auto` only**. Anything else → `auto` + WARNING (§4.4).
  `parallel_tool_calls` undocumented; multi-call responses occur.
- `finish_reason` adds `repetition_truncation` → map to
  `FinishReason.MAX_TOKENS` (a truncation, not an unknown; leaving it
  `UNKNOWN` would still count as terminal but would misreport the turn).
- `response_format: {type: json_object}` works on both models; no
  `json_schema`. Allowlisted with that note.
- Function names `[A-Za-z0-9_-]`, ≤ 64 chars — hashed wire ids fit.
- Built-in `{"type": "web_search"}` tool: out of scope.

### 6.5 Caching, errors, capabilities

- Caching automatic; `usage.prompt_tokens_details.cached_tokens` (base
  reads it). Cache-hit input ≈ 1/100 of a miss.
- Errors *(docs)*: `402` insufficient balance (non-transient), `403`
  region / risk control (non-transient, message names the region rule),
  `404` "model lacks image input" (→ a modality error naming the knob, not
  `ModelNotFoundError`), **`421` content filter** (the OpenAI SDK raises a
  generic `APIStatusError` for 421 → map to a non-transient
  `ContentFilteredError`), `429` (transient; a Token Plan quota exhaustion
  is the same status and indistinguishable by status alone — read the body
  message), `503` overloaded (transient). No documented `Retry-After`.
- `PROVIDER_CAPABILITIES`: images ✅ (`mimo-v2.5`), pdf —, audio — (until
  §6.2's probe), tool_choice ✅ (forwarded; vocabulary is `auto`),
  thinking ✅, prompt_caching —, streaming ✅, cancellation ✅,
  output_media ✅.

### 6.6 Knobs

```yaml
plugin_configs:
  mimo:
    api_key: pass://mimo/key           # JAATO_MIMO_API_KEY → MIMO_API_KEY fallback
    base_url: https://token-plan-sgp.xiaomimimo.com/v1
    context_length: 1048576
    modalities: [text, image]          # mimo-v2.5 is table-detected
    api_params:
      max_tokens: 131072               # → max_completion_tokens
      temperature: 0.7                 # forced to 1.0 while thinking
      top_p: 0.95
      enable_thinking: true            # → thinking.type
      response_format: {type: json_object}
```

## 7. The shared skeleton

Each provider directory mirrors `doubleword/`:

```
model_provider/<name>/
  __init__.py     PROVIDER_CAPABILITIES / PROVIDER_KNOBS / PROVIDER_QUIRKS /
                  PROVIDER_AUTH_RESOLUTION — the scaffold reads these, nothing
                  else registers the provider
  env.py          JAATO_<NAME>_API_KEY / _BASE_URL / _MODEL / _CONTEXT_LENGTH
                  + the vendor's own key var; resolve_*; is_self_hosted;
                  get_checked_credential_locations(config=)
  errors.py       <Name>Error tree: APIKeyNotFoundError, AuthenticationError,
                  RateLimitError(retry_after), ModelNotFoundError,
                  ContextLimitError, InfrastructureError (+ the vendor's
                  extra: QuotaExhaustedError for kimi, ContentFilteredError
                  for mimo)
  auth.py         credential file (project → home), validate_api_key,
                  try_load_credentials_with_reason
  provider.py     the subclass; see below
  tests/          test_auth.py, test_<name>_provider.py,
                  test_stream_cancel_close.py, test_tool_choice_mapping.py
                  (copied from doubleword), + the vendor-specific ones in §8
plugins/<name>_auth/plugin.py   TRAIT_AUTH_PROVIDER, provider_name,
                  credential_env_vars, get_default_models — the nim_auth shape
```

Three base-class additions serve all three providers and stay inert for
the existing eight inheritors (each is `False` / identity by default):

| Hook | Default | minimax | kimi | mimo |
|---|---|---|---|---|
| `replay_reasoning: bool` | `False` | `True` | `True` | `True` |
| `_reasoning_replay_fields(text) -> dict` | `{"reasoning_content": text}` | + `reasoning_details` | default | default |
| `_reasoning_from_delta(delta) -> Optional[str]` | `delta.reasoning_content` | + `reasoning_details[].text` | default | default |
| `_tool_choice_vocabulary(model) -> set` | `{auto, none, required, named}` | `{auto, none}` | K3 full; K2.x `{auto, none}` | `{auto}` |
| `_max_tokens_wire_name` | `"max_tokens"` | `"max_completion_tokens"` | `"max_completion_tokens"` | `"max_completion_tokens"` |
| `_thinking_request_fields() -> dict` (merged into `extra_body`) | `{}` | `reasoning_split` + `thinking` (M3) | `reasoning_effort` (K3) / `thinking` (K2.x) | `thinking` |

`_tool_choice_vocabulary` folds into `_apply_api_params`, which is the one
place `tool_choice` is written; a value outside the vocabulary becomes
`auto` with a WARNING that names the model and the value. That rule is
worth stating once in the base docstring: **a vendor-imposed vocabulary
narrows the forwarded value, it never silently drops the parameter and it
never lets a 400 through** — the doubleword `test_tool_choice_mapping.py`
gets a parametrised sibling per provider.

### 7.1 Registration checklist (what the doubleword PR touched outside its directory)

| Site | Change |
|---|---|
| `shared/env_scope.py` | `JAATO_<NAME>_API_KEY` / `_BASE_URL` / `_CONTEXT_LENGTH` / `_MODEL` (SESSION, typed key); the vendor key vars (`MINIMAX_API_KEY`, `MOONSHOT_API_KEY`, `MIMO_API_KEY`) as SESSION with the same typed key, like `NEBIUS_API_KEY` |
| `tests/test_provider_capability_conformance.py` | `_CONVERTERS["<name>"] = ("_openai_compat/converters.py", "message_to_openai")` |
| `plugins/model_provider/tests/test_modalities.py` | the `(pkg, Class)` list |
| `plugins/model_provider/tests/test_profile_api_key_location.py` | the parametrised env-module list |
| `tests/test_a_dead_stream_is_not_a_finished_turn.py`, `tests/test_truncation_is_not_reported_as_tool_use.py` | comment only — the count for `_openai_compat/base.py` is unchanged; the docstring list of inheritors grows |
| `plugins/model_provider/_attachments.py`, `_openai_compat/converters.py`, `tests/test_cache_hit_percent_against_invoice.py` | the docstring lists of "providers sharing this converter" |
| `docs/model-provider-capabilities.md` | regenerate (`python -m shared.tests.test_provider_capabilities`) |
| `docs/design/multimodal-model-support.md` | the "Where jaato is now" `output_media` row must name the new inheritors — `test_docs_do_not_contradict_the_tree` enforces it |
| `CLAUDE.md` | a provider bullet under "Model Provider Plugins" and an env table per vendor; the design-doc bullet |
| `README.md` | the provider table (+3 rows) and the "16 providers" count |
| `tests/test_cyclomatic_complexity_audit.py` | nothing, if every new function stays ≤ 15 — the `_knob`-style closures in `zhipuai` are the pattern to avoid |

## 8. Test plan

Unit (no network, the `mock_client_class` fixture the doubleword tests
use):

- **Replay seam** (framework PR): a MODEL message with thought + tool
  calls converts to an assistant dict with `reasoning_content` and
  `content: ""`; without `replay_reasoning` the dict is byte-identical to
  today's; the session keeps thought parts only when the provider says so;
  `estimate_message_tokens` counts them; serialize/deserialize round-trips
  them.
- **Streaming**: reasoning arrives as `delta.reasoning_content` (kimi,
  mimo) or `delta.reasoning_details` (minimax) and lands both in
  `ProviderResponse.thinking` and in a leading thought Part; the MiniMax
  `{"role": ""}` chunk does not break assembly; `<think>` in content is
  stripped.
- **Thinking mapping**: every (model, knob) cell of §4.3 / §5.4 / §6.3
  produces the documented body, and the impossible cells log rather than
  send.
- **Tool-choice vocabulary**: per provider and per model family.
- **Sampling stripping** (kimi): `temperature` never reaches the wire.
- **Cache accounting**: `usage.cached_tokens` (kimi) vs
  `prompt_tokens_details.cached_tokens` (minimax, mimo) → `cache_read_tokens`.
- **Errors**: kimi's three `429` types → the right transient flag; MiniMax
  `2056` → `retry_after` from the ISO time; MiMo `421` → non-transient.
- **Context**: catalog-first for kimi (a fake `/v1/models` with
  `context_length` beats the knob); table-then-knob-then-fail for the other
  two; an unknown model fails loud naming the knob and the env var.

Live (opt-in, keyed, the `smoke/` convention `ollama` uses): one tool-loop
turn per provider against the real endpoint, asserting the second request
carried `reasoning_content` — the one test that proves §3 on a real wire,
and for MiMo the one that would have been a `400` before it.

## 9. Sequencing and open questions

1. **PR 0 — reasoning replay seam** (§3): base-class hooks, session gate,
   converter argument, GC counting, capability field, tests with a fake
   upstream. No provider yet, so nothing user-visible changes.
2. **PR 1 — `mimo`**: the smallest surface (two models, one toggle,
   `auto`-only tool choice) and the one whose vendor *refuses* a request
   without replay — the live smoke test is the seam's proof.
3. **PR 2 — `kimi`**: the richest (catalog modalities, two thinking
   dialects, three-way 429, `strict: false`, `prompt_cache_key`).
4. **PR 3 — `minimax`**: `reasoning_split`, `reasoning_details` echo, the
   `<think>` guard, numbered error codes.

PR 1–3 are independent of each other and can land in any order once PR 0
is in.

Open questions, each with the default this design takes:

- **Video input.** All three vendors accept `video_url` on at least one
  model; no wire in the tree carries video and `_attachments.py` withholds
  it with a note. Default: declare `{text, image}` only; video is a
  framework feature, not a provider one.
- **MiMo audio input.** Default: text + image until the `input_audio`
  block is probed against `mimo-v2.5` (§6.2).
- **Region as a knob.** MiniMax and Kimi both have `.cn` hosts and
  region-bound keys. Default: `base_url` only, hosts documented — one knob
  fewer, consistent with every other provider; revisit if users hit the
  wrong-region `401` often.
- **A `thinking_keep` framework knob.** Only Kimi K2.6 has it. Default: a
  kimi-only `api_params` key, not a framework convention.
- **`openrouter` replay.** Same gap, different wire shape
  (`reasoning_details`). Default: follow-up after PR 0, not part of this
  work.
- **Pricing tables.** `.jaato/pricing.json` is operator-owned; ship the
  list prices from §4–6 as an example, not as defaults.

## 10. Implementation notes (where the code diverged)

- **Knob outranks table.** §4.2 / §6.2 said catalog → table → knob. The
  built-in table is the provider's *guess*, so the operator's
  `context_length` knob now beats it: catalog → knob → table → fail-loud
  (`resolve_context_window` for the first two, the table after). Kimi is
  unchanged: catalog → knob, no table.
- **MiniMax `2056` is a quota error, not a rate limit.** §4.1 proposed
  `RateLimitError(retry_after=<reset − now>)`; a Token Plan window resets
  hours later and the retry ladder tops out at seconds, so it is a
  non-transient `QuotaExhaustedError(resets_at=…)` alongside `1008`.
- **No automatic `prompt_cache_key`** (§5.6). The provider has no
  per-session identity to key on — `_agent_id` is `"main"` for every main
  session, so a default would have shared one cache key across sessions.
  It stays a profile knob.
- **Auth plugins shipped** (`minimax_auth`, `kimi_auth`, `mimo_auth`, the
  `nim_auth` shape). The three newest providers before this work
  (`nebius`, `ovhcloud`, `doubleword`) document a `*-auth` command that
  no plugin provides; these three do provide it.
- **The base gained the §7 hooks as written**, plus `_wire_tools` (Kimi's
  `strict: false`), `_map_finish_reason` (MiMo's `repetition_truncation`)
  and `_finish_batch_response` (which also fills `reasoning_tokens` from
  `completion_tokens_details` on every OpenAI-compat provider).
- **The MiniMax `{"role": ""}` chunk** is not pinned by a test: the
  OpenAI SDK constructs stream chunks without validation, and a
  MagicMock-based test would prove nothing about that.

## 11. Sources

MiniMax: platform.minimax.io/docs (`api-reference/text-openai-api`,
`text-chat-openai.md`, `text-prompt-caching.md`, `models/openai/list-models.md`,
`guides/rate-limits.md`, `guides/pricing-paygo.md`, `token-plan/*`,
`api-reference/errorcode.md`, `guides/text-m3-function-call.md`);
minimax.io/news "why-is-interleaved-thinking-important-for-m2";
MiniMax-AI/MiniMax-M2.5#2, MiniMax-M2.7#36; live probes of the three hosts.

Kimi: platform.kimi.ai/docs (`openapi.json`, `models.md`,
`platform-changelog.md`, `api/chat.md`, `api/list-models.md`, `api/errors.md`,
`pricing/limits.md`, `pricing/chat-k3.md`, `guide/use-thinking-models.md`,
`guide/use-reasoning-effort.md`, `guide/use-tool-choice.md`,
`guide/use-context-caching-feature-of-kimi-api.md`, `guide/claude-code-kimi.md`,
`guide/product-plans.md`); kimi.com/code/docs.

MiMo: mimo.mi.com/docs/en-US (`api/chat/openai-api`, `api/chat/anthropic-api`,
`api/model/list-models`, `api/guidance/error-codes`, `api/guidance/rate-limit`,
`quick-start/usage-guide/text-generation/deep-thinking`, `updates/deprecate`,
`price/pay-as-you-go`, `quick-start/faq/api-integration`);
XiaomiMiMo/MiMo#44, #53, #55; recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5;
vLLM `reasoning/__init__.py`, `tool_parsers/__init__.py` (the `mimo` alias).
