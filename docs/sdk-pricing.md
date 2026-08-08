# SDK pricing & cost reporting

`UsageBreakdown.cost_usd` is populated on three events
(`TurnCompletedEvent`, `TurnProgressEvent`, `ContextUpdatedEvent`)
when the daemon can derive a cost. There is no hardcoded pricing in
jaato — the operator decides which models have prices and what the
prices are.

## How a cost ends up on the wire

Two sources, in this precedence:

1. **Provider-reported.** Some providers expose a real cost number
   (e.g. `claude_cli` reads `total_cost_usd` from the underlying
   `claude` CLI output and forwards it to `TokenUsage.cost_usd`).
   That value is the fiscal truth and beats any computed estimate.
2. **Pricing-table computed.** When the provider doesn't report
   cost, the daemon multiplies token counts by per-model rates from
   `.jaato/pricing.json` (workspace) or `~/.jaato/pricing.json`
   (user-level). Workspace overrides user.

If neither source has a number, `cost_usd` stays `None`. **Never
zero, never silently filled with a default.** Consumers must not
interpret `None` as "free".

## Pricing file format

Litellm-compatible. Drop any subset of Litellm's
`model_prices_and_context_window.json` straight in:

```json
{
  "claude-sonnet-4-5": {
    "input_cost_per_token":             3e-06,
    "output_cost_per_token":            15e-06,
    "cache_read_input_token_cost":      3e-07,
    "cache_creation_input_token_cost":  3.75e-06
  },
  "gpt-4o": {
    "input_cost_per_token":  2.5e-06,
    "output_cost_per_token": 1e-05
  }
}
```

Recognised fields:

| Key | Multiplied by |
|-----|---------------|
| `input_cost_per_token` | `prompt_tokens` (new uncached input) |
| `output_cost_per_token` | `output_tokens` |
| `cache_read_input_token_cost` | `cache_read_tokens` (cache hits) |
| `cache_creation_input_token_cost` | `cache_creation_tokens` (cache writes) |

Unknown keys in the JSON are ignored. Token counts that reach the
daemon as `None` (the provider doesn't track that dimension)
contribute nothing rather than crashing.

## File precedence and discovery

```
<workspace>/.jaato/pricing.json   # highest precedence
~/.jaato/pricing.json             # falls back to user
```

Earlier paths win on key conflicts. Missing files are skipped
silently. Malformed JSON is logged at WARN and skipped — a broken
pricing file never breaks session creation.

## Model name matching

Exact match only. There is no fuzzy matching, no version stripping,
no provider prefix logic. If your tracking distinguishes
`claude-sonnet-4-5` and `claude-sonnet-4-5-20250514`, both keys
must exist in the JSON. This keeps the math auditable — operators
control the map.

## What gets emitted

The three events all carry the same `UsageBreakdown` shape:

```python
class UsageBreakdown(BaseModel):
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None       # OpenAI o-series
    thinking_tokens: Optional[int] = None        # Anthropic / Gemini
    cost_usd: Optional[float] = None             # populated per above
```

A consumer wanting to display "this turn cost $0.0024" reads
`event.usage.cost_usd` and falls back to "?" or hides the line when
it's `None`.

## When to use provider-side vs computed cost

Default: leave the precedence as-is. Provider-side wins because
it's authoritative.

Cases where you might *want* computed even when provider gives a
number:

- Cross-provider comparison ("what would Sonnet have cost?"). Drop
  pricing for the comparison model in the JSON and consumers can
  compute it client-side from the token breakdown.
- Audit trail with operator-controlled rates (e.g. internal
  charge-back model with custom margins). Provider-side is the
  external truth, but you may want to record both. The current
  framework only emits one cost — fork the math at the consumer if
  you need both.

## Disabling cost computation

Don't ship a `pricing.json`. The daemon emits `cost_usd: None`
universally, providers that report cost themselves still populate
it (claude_cli stays correct), and the consumer interface is
unchanged. There's no "off switch" because the absence of a file
*is* the off switch.

## Migration notes

Pre-1.0 events carried token counts as flat fields directly on the
event (`event.prompt_tokens`, `event.cache_read_tokens`, ...). v1.0+
nests them under `event.usage`:

```python
# before (pre-1.0)
event.prompt_tokens
event.cache_read_tokens

# after (v1.0+)
event.usage.prompt_tokens
event.usage.cache_read_tokens
event.usage.cost_usd                   # new
event.usage.reasoning_tokens           # new
event.usage.thinking_tokens            # new
```

The `compute_cache_hit_percent` helper in both SDKs reads from
`event.usage.*` automatically — consumers using the helper require
no code changes beyond the version bump.

GC configuration also moved out of `ContextUpdatedEvent` into a
dedicated `GCConfigEvent` in v1.0+; subscribe to that event for
status-bar configuration display.
