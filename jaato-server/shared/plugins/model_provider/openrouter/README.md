# OpenRouter Provider

Runs jaato against [OpenRouter](https://openrouter.ai) — a unified gateway
exposing 300+ models from many vendors (OpenAI, Anthropic, Google, Meta,
Mistral, DeepSeek, …) behind a single OpenAI-compatible API.

Chat goes through `POST /api/v1/chat/completions`; model catalog lookups
go through the public `GET /api/v1/models`; auth introspection uses
`GET /api/v1/key`.

---

## When to use this provider

| You want… | Use |
|---|---|
| One API key for many vendors / models | **OpenRouter** |
| Native Anthropic auth (OAuth subscription, direct billing) | [`anthropic`](../anthropic/) |
| Local inference (no API spend) | [`ollama`](../ollama/), [`lmstudio`](../lmstudio/) |
| Per-request **provider routing** (price/throughput/region constraints) | OpenRouter — `routing` knob |
| **Prompt caching** with a single profile across Claude / Gemini / GPT | OpenRouter — `cache_prompt` knob |

---

## Requirements

- An OpenRouter API key (`sk-or-...`) from
  <https://openrouter.ai/settings/keys>.
- Python deps already present in jaato (`openai`, `httpx`).

---

## Quick start

```bash
export JAATO_OPENROUTER_API_KEY=sk-or-...
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon
.venv/bin/python jaato-tui/rich_client.py --connect /tmp/jaato.sock
# In TUI: session.new --profile openrouter   (or any profile that sets provider=openrouter)
```

Or store the key persistently:

```bash
openrouter-auth key sk-or-...
```

---

## Configuration

Two sources, merged at runtime — environment provides defaults, the
session profile overrides per session.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `JAATO_OPENROUTER_API_KEY` | — | API key (`sk-or-…`) |
| `JAATO_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint |
| `JAATO_OPENROUTER_MODEL` | — | Default model |
| `JAATO_OPENROUTER_CONTEXT_LENGTH` | catalog | Context-window override |
| `JAATO_OPENROUTER_HTTP_REFERER` | the application's URL | [App attribution](https://openrouter.ai/docs/app-attribution): site URL (required for rankings) |
| `JAATO_OPENROUTER_APP_TITLE` | the application's name | App attribution: display name |
| `JAATO_OPENROUTER_APP_CATEGORIES` | `cli-agent` | App attribution: comma-separated marketplace categories |
| `JAATO_OPENROUTER_REQUEST_TIMEOUT` | `600` | Byte-level per-request deadline, seconds (`0` disables) |
| `JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT` | `300` | Streaming payload idle deadline, seconds (`0` disables) |

### Naming your application

The two attribution defaults above are **the application's**, not the
framework's. Set `JAATO_APP_NAME` (plus optionally `JAATO_APP_URL` /
`JAATO_APP_VERSION`) and every product built on the SDK reports under its own
name instead of collapsing into jaato's row on the OpenRouter dashboard:

```bash
export JAATO_APP_NAME="Acme Copilot"
export JAATO_APP_URL="https://acme.example"
# → X-OpenRouter-Title: Acme Copilot (powered by jaato)
# → HTTP-Referer:       https://acme.example
```

Set `JAATO_APP_POWERED_BY=false` to drop the suffix, or
`JaatoRuntime(app_identity=AppIdentity(...))` to declare it in code. The
per-session knobs below still outrank all of it. With nothing set, the
headers are jaato's own, exactly as before. See
[Application Identity](../../../../docs/design/app-identity.md).

### Profile `plugin_configs["openrouter"]`

Four layers under `plugin_configs.openrouter` (server 0.6.23+):

```yaml
plugin_configs:
  openrouter:
    # Top-level — auth / identity
    api_key: "sk-or-..."           # overrides env / stored credentials
    http_referer: "https://..."    # HTTP-Referer header (outranks JAATO_APP_URL)
    app_title: "MyApp"             # X-OpenRouter-Title header (outranks JAATO_APP_NAME)
    app_categories: ["cli-agent"]  # X-OpenRouter-Categories header
                                   # (marketplace categories for rankings;
                                   # pass [] to opt out of category attribution)
    extra_headers:                 # arbitrary HTTP headers (e.g. beta opt-ins)
      x-anthropic-beta: "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14"

    # api_params — OpenAI Chat Completions request body fields
    api_params:
      temperature: 0.55
      top_p: 1.0
      top_k: 40
      max_tokens: 8192
      models:                      # cross-model fallback list (sibling of `model`)
        - anthropic/claude-sonnet-4.5
        - openai/gpt-5-mini
        - google/gemini-3-flash-preview
      enable_thinking: true        # extended-reasoning request + extraction
      thinking_budget: 16384       # → reasoning.max_tokens
      thinking_level: "high"       # → reasoning.effort (low/medium/high)
      cache_prompt: "auto"         # "auto" (default) | true | false
      cache_ttl: "5m"              # "5m" (default) | "1h"

    # routing — OpenRouter provider routing extension, passed verbatim.
    # Any key from https://openrouter.ai/docs/features/provider-routing
    # works without code changes — the dict is opaque pass-through.
    routing:
      sort: "price"                # "price" | "throughput" | "latency"
                                   # OR {by: "...", partition: "model"|"none"}
      data_collection: "deny"      # "allow" (default) | "deny"
      ignore: ["Groq"]             # provider slugs to skip
      only: ["azure"]              # allowlist (mutex with ignore)
      order: ["openai", "together"]  # try these first, then fall back
      require_parameters: true     # only upstreams that support every
                                   # param in the request (e.g. tools, JSON)
      allow_fallbacks: true        # set false to fail rather than try others
      quantizations: ["fp8"]       # filter by quant level
      zdr: true                    # Zero Data Retention endpoints only
      enforce_distillable_text: true  # only models that allow text distillation
      max_price:                   # caps; request fails if no provider qualifies
        prompt: 1
        completion: 2
        request: 0.01
        image: 0.001
      preferred_min_throughput:    # number for p50, or per-percentile object
        p90: 50                    # ≥50 tokens/sec at p90 over 5-min window
      preferred_max_latency:
        p50: 1
        p90: 3
        p99: 5

    # framework_overrides — rare escape hatches
    framework_overrides:
      context_length: 32768
      base_url: "https://..."
      connect_timeout: 15         # seconds; TCP + TLS
      request_timeout: 600        # seconds; httpx read/write/pool (0 disables)
      stream_idle_timeout: 300    # seconds; payload idle deadline (0 disables)
```

| Layer | Keys | Purpose |
|---|---|---|
| top-level | `api_key`, `http_referer`, `app_title`, `app_categories`, `extra_headers` | auth / identity. `app_categories` (`List[str]`) opts your profile into OpenRouter's [marketplace rankings](https://openrouter.ai/docs/app-attribution) via the `X-OpenRouter-Categories` header (jaato defaults to `["cli-agent"]`; pass `[]` to opt out). Validated strictly: lowercase hyphen-separated, ≤30 chars each, ≤5 entries. `extra_headers` carries OpenRouter's [provider-specific beta headers](https://openrouter.ai/docs/features/provider-routing#provider-specific-headers) (e.g. `x-anthropic-beta`). |
| `api_params` | `temperature`, `top_p`, `top_k`, `max_tokens`, `models`, `enable_thinking`, `thinking_budget`, `thinking_level`, `cache_prompt`, `cache_ttl` | OpenAI Chat Completions body fields; `models` is OpenRouter's request-level cross-model fallback list |
| `routing` | any [provider routing](https://openrouter.ai/docs/features/provider-routing) key (`order`, `allow_fallbacks`, `require_parameters`, `data_collection`, `ignore`, `only`, `quantizations`, `sort`, `zdr`, `enforce_distillable_text`, `max_price`, `preferred_min_throughput`, `preferred_max_latency`, ...) | constrains which upstream serves each request; opaque pass-through, so new routing keys work without a framework release |
| `framework_overrides` | `context_length`, `base_url`, `connect_timeout`, `request_timeout`, `stream_idle_timeout` | rare escape hatches. The three deadlines are what bounds a single request — see [Request deadlines](#request-deadlines). |

**Backward compatibility:** every nested key is also accepted at the
legacy flat position (`temperature:` directly under `openrouter:`,
`provider:` instead of `routing:`) with a one-time deprecation warning
per key. Flat-key support will be removed in a future release.

---

## Prompt caching

OpenRouter's [prompt-caching feature](https://openrouter.ai/docs/features/prompt-caching)
splits into two flavours depending on the upstream:

- **Automatic caching** — OpenAI, DeepSeek, Grok, Moonshot, … cache
  stable prefixes server-side without any client-side annotation.
  Savings happen regardless.
- **Explicit caching** — Anthropic Claude and Google Gemini Pro/Flash
  require the client to stamp
  `cache_control: {"type": "ephemeral"}` breakpoints onto the system
  block and the last tool definition. **Without these, no caching
  occurs.**

This provider does both:

| `cache_prompt` setting | Behaviour |
|---|---|
| `"auto"` (default) | Stamps breakpoints **only** when the model id matches `anthropic/*` or `google/gemini-1.5-*` / `2.5-*` / `3*`. Other models cache automatically. |
| `true` / `"on"` | Always stamps breakpoints, regardless of model. Useful when OpenRouter adds a new explicit-cache vendor before this module learns about it. |
| `false` / `"off"` | Never stamps. Useful for always-fresh prompts that don't benefit. |

| `cache_ttl` setting | Behaviour |
|---|---|
| `"5m"` (default) | 5-minute ephemeral cache, 1.25× write premium (Anthropic). |
| `"1h"` | Extended 1-hour cache, 2× write premium. Avoids mid-session cache misses on hour-long agentic runs. |

### What gets cached

Two breakpoints per request:

1. **System block.** Converted from `content: "<text>"` to
   `content: [{"type": "text", "text": ..., "cache_control": ...}]`.
2. **Last tool definition.** Tools are sorted by their (hashed) wire
   name so the cache prefix stays stable across turns; `cache_control`
   is stamped on the last element.

A third "history breakpoint" (analogous to BP3 in the
`cache_anthropic` plugin) is intentionally out of scope — it requires
budget-aware placement that the OpenAI-shaped wire makes awkward, and
the two breakpoints we do place cover the highest-value cacheable
content (large system prompts + tool catalogs).

### Response-side accounting

The provider always sends `usage: {"include": true}` so OpenRouter
returns its detailed usage block. The following fields land in
`TokenUsage`:

| OpenRouter field | `TokenUsage` field |
|---|---|
| `prompt_tokens_details.cached_tokens` | `cache_read_tokens` |
| `cache_creation_input_tokens` | `cache_creation_tokens` |
| `cost` | `cost_usd` |

These flow into the daemon's ledger / token-accounting telemetry, so
savings are visible regardless of whether caching was explicit or
automatic.

### Example: Claude profile with 1-hour cache

```json
{
  "name": "claude-cached",
  "model": "anthropic/claude-3.5-sonnet",
  "provider": "openrouter",
  "plugin_configs": {
    "openrouter": {
      "api_params": {
        "cache_prompt": "auto",
        "cache_ttl": "1h"
      }
    }
  }
}
```

---

## Authentication

Three sources, checked in order:

1. **`api_key` in the profile** (`plugin_configs.openrouter.api_key`)
2. **`JAATO_OPENROUTER_API_KEY`** environment variable
3. **Stored credentials** from `openrouter-auth key <api_key>` (mode 0600)

`openrouter-auth status` shows which source is active.

---

## Provider routing

OpenRouter's killer feature: constrain which upstream serves each
request. Composes with `model: "openrouter/auto"` (auto picks model,
routing constrains hosts):

```yaml
routing:
  sort: "price"            # cheapest upstream
  data_collection: "deny"  # exclude upstreams that train on your data
  ignore: ["Groq"]
  require_parameters: true
  allow_fallbacks: true
```

Any [provider routing](https://openrouter.ai/docs/features/provider-routing)
key works (`order`, `quantizations`, …) — passed through verbatim via
`extra_body`.

---

## Error types

Defined in `errors.py`:

| Exception | When raised |
|---|---|
| `APIKeyNotFoundError` | No credentials in env / profile / stored file |
| `AuthenticationError` | API rejected the key (401) |
| `RateLimitError` | 429 — surfaces `Retry-After` if present |
| `ModelNotFoundError` | Model id not in OpenRouter catalog |
| `ContextLimitError` | Prompt exceeds upstream's context window |
| `InfrastructureError` | 5xx / connection error — retryable |
| `StallTimeoutError` | No response payload inside `stream_idle_timeout` — subclasses `InfrastructureError`, so retryable |
| `UpstreamFinishError` | Turn ended with `finish_reason: "error"` and no error payload — names the `native_finish_reason`; subclasses `InfrastructureError`, so retryable |

`RateLimitError` and `InfrastructureError` are classified as transient
by the reliability layer.

### The two shapes of a mid-stream error

OpenRouter reports an upstream failure mid-turn in two shapes, and
before #766 only one of them survived:

| Shape | Wire | Raised as |
|---|---|---|
| 1 | top-level `error` object **+** `finish_reason: "error"` | `InfrastructureError` carrying the upstream's own message |
| 2 | `finish_reason: "error"` **alone**, cause in `native_finish_reason` | `UpstreamFinishError` naming that reason |

Shape 2 used to resolve to `FinishReason.ERROR` and travel back as an
ordinary response, which `JaatoSession` turned into a terminal
`RuntimeError("Provider returned an error")` — one string for every
cause, and fatal where the identical upstream condition in shape 1 was
retried. Eleven sweep arms died with that string. The cause was
knowable throughout: Gemini's `MALFORMED_FUNCTION_CALL`, a function
call the model's own serialiser rejected, sitting in a field the
provider already parsed for #745.

`native_finish_reason` is now traced on **every** streamed turn
(`*_NATIVE_FINISH_REASON`), not only failing ones — an
`UNEXPECTED_TOOL_CALL` on the generation before a
`MALFORMED_FUNCTION_CALL` is the kind of adjacency that shortens the
next investigation, and `FinishReason` has nowhere to carry it.

---

## Request deadlines

Nothing inside the provider used to bound a single request (#732). An
agentic session could stop mid-tool-loop and never resume — no
exception, no `finish_reason`, no retry — sitting on an ESTABLISHED
socket with zero bytes queued until something *outside* the provider
(a harness arm-timeout, a budget ceiling) tore it down. Two layers now
bound it:

| Layer | Knob | Default | Enforced by | Bounds |
|---|---|---|---|---|
| connect | `connect_timeout` | 15s | httpx | TCP + TLS handshake |
| byte | `request_timeout` | 600s | httpx (read/write/pool) | a socket that has gone silent entirely |
| payload | `stream_idle_timeout` | 300s | `stall.StreamStallGuard` | a stream that produces no chunks |

The payload layer exists because the byte layer cannot see this
failure: OpenRouter keeps a stalled stream fed with
`: OPENROUTER PROCESSING` SSE comments, and the OpenAI SDK's decoder
drops comment lines without yielding an event — so those bytes reset
httpx's read clock while the consumer's chunk loop never ticks. The
guard measures silence in *payload*: it is pinged by every chunk the
SDK yields, and on expiry it closes the stream and the client's httpx
pool (which is what unparks the blocked read, and stops upstream
generation and billing per OpenRouter's stream-cancellation spec).
The consumer then raises `StallTimeoutError`.

Each knob accepts `0` to disable that deadline, restoring the
pre-#732 unbounded wait — a legitimate long generation and a dead
socket look identical from here, so the bound is configurable rather
than assumed.

The OpenAI SDK's own `max_retries` is set to **0**: its default of 2
silently multiplies every deadline by three and is invisible in the
daemon log. Retries belong to `retry_utils.with_retry`, which
classifies via `classify_error`, applies exponential backoff, and
honours the `Retry-After` hint OpenRouter puts in the response body
(#720). A stall carries no such hint — the upstream never answered —
so it falls through to the standard backoff.

---

## Architecture

```
jaato/jaato-server/shared/plugins/model_provider/openrouter/
├── __init__.py       # Package docstring, exports OpenRouterProvider + create_provider
├── _lazy.py          # Lazy import of the openai SDK
├── auth.py           # Stored-credential file handling
├── cache.py          # Prompt-caching helpers — model detection, cache_control dict
├── converters.py     # ToolSchema / Message ↔ OpenAI chat dict; cache_control stamping
├── env.py            # JAATO_OPENROUTER_* env-var resolution
├── errors.py         # APIKey / Auth / RateLimit / ModelNotFound / Context / Infra
├── provider.py       # OpenRouterProvider — the main class
├── stall.py          # StreamStallGuard — the payload-idle watchdog (#732)
├── tests/
│   ├── test_auth.py
│   ├── test_openrouter_provider.py
│   ├── test_prompt_caching.py
│   └── test_stall_deadline.py
└── README.md         # This file
```

### Profile → Provider wiring

```
SubagentProfile.plugin_configs["openrouter"]
    ↓  JaatoServer._build_session_kwargs_from_profile()
JaatoRuntime.create_session(plugin_configs=...)
    ↓  JaatoSession.configure()
JaatoRuntime.create_provider(plugin_configs=...)
    ↓  merged into ProviderConfig.extra
OpenRouterProvider.initialize(config)
    ↓  reads four-layer namespacing (top-level / api_params / routing / framework_overrides)
```

---

## Limitations

- **No history breakpoint.** Only two cache breakpoints are placed
  (system + last tool). A budget-aware history breakpoint analogous to
  BP3 in `cache_anthropic` is deferred — OpenAI-shaped messages make
  the placement awkward.
- **Token counting is heuristic.** OpenRouter doesn't expose a
  tokenizer endpoint and tokenizer choice depends on the upstream.
  `count_tokens()` uses ~4 chars / token. For tight budget
  enforcement, set `framework_overrides.context_length` conservatively.
- **No structured-output schema.** `response_format: json_object` is
  forwarded but no per-schema validation happens here — upstream
  capabilities vary.
- **Tool name hashing.** Tool names are hashed to `t_<8hex>` before
  the wire to dodge strict-validator upstreams. The mapping is
  deterministic; reverse-lookups go through `shared.tool_id_map`.

---

## See also

- [OpenRouter docs — Prompt caching](https://openrouter.ai/docs/features/prompt-caching)
- [OpenRouter docs — Provider routing](https://openrouter.ai/docs/features/provider-routing)
- [`anthropic/`](../anthropic/) — native Anthropic provider (cache plugin lives there)
- [`cache_anthropic/`](../../cache_anthropic/) — explicit-breakpoint caching for the native Anthropic provider
- [`lmstudio/`](../lmstudio/), [`ollama/`](../ollama/) — local-inference alternatives
