# LM Studio Provider

Runs jaato against a local [LM Studio](https://lmstudio.ai) server —
GPT-OSS, Qwen, Gemma, Llama, and any other model LM Studio hosts — with
optional **load-control**: the provider can reconfigure the model in
memory (context length, GPU offload, KV-cache placement, flash attention,
MoE expert count, …) from the session profile before the first chat.

Chat goes through LM Studio's OpenAI-compatible `/v1/chat/completions`
endpoint; load-control goes through its native `POST /api/v1/models/load`
endpoint.

---

## When to use this provider

| You want… | Use |
|---|---|
| Run an already-loaded LM Studio model | **passive mode** (omit `load`) |
| Reconfigure context length / GPU offload per session | **active mode** (set `load`) |
| Remote local-inference (e.g. a GPU box on LAN) | set `LMSTUDIO_HOST` / `host` |
| Switch between dozens of hosted models via one API | use the upcoming OpenRouter provider instead |
| CPU-only laptop inference with smallest footprint | `ollama` is lighter weight |

---

## Requirements

- **LM Studio ≥ 0.3.5** (earlier versions don't expose `/api/v1/models/load`).
- **At least one model downloaded** — use LM Studio's Discover tab or
  the `lms` CLI: `lms get openai/gpt-oss-20b`.
- **Local server started** — Developer tab → *Start Server* (default port
  1234). `lms server start` works headlessly.
- Python deps: already present in jaato (`openai`, `httpx`).

---

## Quick start

### 1. Start LM Studio's local server

From LM Studio UI → Developer → Start Server.  Or headless:

```bash
lms server start
```

### 2. Verify it's up

```bash
curl http://localhost:1234/api/v0/models
```

### 3. Passive mode — use whatever's loaded

```bash
export LMSTUDIO_MODEL=openai/gpt-oss-20b
.venv/bin/jaato-server --ipc-socket /tmp/jaato.sock --daemon
.venv/bin/python jaato-tui/rich_client.py --connect /tmp/jaato.sock
# In TUI: session.new --profile lmstudio
```

### 4. Active mode — jaato loads the model with a specific config

Create `.jaato/profiles/local-gpt-oss.json`:

```json
{
  "name": "local-gpt-oss",
  "description": "Local GPT-OSS 20B with 16K context, flash attention on",
  "model": "openai/gpt-oss-20b",
  "provider": "lmstudio",
  "plugin_configs": {
    "lmstudio": {
      "host": "http://localhost:1234",
      "load": {
        "context_length": 16384,
        "flash_attention": true,
        "offload_kv_cache_to_gpu": true,
        "eval_batch_size": 512
      }
    }
  }
}
```

Start a session with it:

```
session.new --profile local-gpt-oss
```

The provider will POST the `load` dict to `/api/v1/models/load` and then
start chatting against the freshly-loaded model.

---

## Configuration

There are two ways to configure the provider.  They merge: environment
variables provide defaults, the session profile overrides per-session.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LMSTUDIO_HOST` | `http://localhost:1234` | Server URL |
| `LMSTUDIO_MODEL` | — | Default model name |
| `LMSTUDIO_CONTEXT_LENGTH` | — | Manual context-window override |
| `LMSTUDIO_API_TOKEN` | — | Bearer token (only when LM Studio has *Require API Token* enabled) |

### Profile `plugin_configs["lmstudio"]`

Providers are plugins in jaato's plugin model (`PLUGIN_KIND =
"model_provider"`).  Their session-level configuration lives under
`plugin_configs` keyed by the provider's `name` — in this case,
`"lmstudio"`.  The runtime merges this dict into `ProviderConfig.extra`
before `initialize()`.

| Key | Type | Description |
|---|---|---|
| `host` | str | Override `LMSTUDIO_HOST` |
| `context_length` | int | Context window override (bypasses catalog) |
| `api_token` | str | Bearer token override |
| `load` | dict | Passthrough body for `POST /api/v1/models/load` (see below) |

---

## Passive vs active mode

### Passive mode (`load` not set)

Provider only talks to the chat endpoint.  The user is responsible for
loading the model into LM Studio via UI or `lms load`.  This is the
right default when:

- You're happy with LM Studio's default load settings
- You manage model loading externally (GUI, CI script, `lms` CLI)
- Multiple sessions share a single loaded model and shouldn't be
  reloading it with different configs

### Active mode (`load` is a dict)

Provider POSTs `load` verbatim (with `model` injected from `connect()`'s
argument) to `POST /api/v1/models/load` during `connect()`.  LM Studio
loads or reloads the model with the supplied configuration.  This is the
right choice when:

- Different sessions need different context windows / GPU settings
- You want the profile to fully describe the inference environment
- Reproducibility matters — the profile is the single source of truth

**Important:** active mode affects the server globally.  If two sessions
with different `load` configs run concurrently, the second reload will
interrupt the first.  If that matters, use one LM Studio server per
active-mode profile (different `host` ports).

---

## Load parameters

The `load` dict is passed to LM Studio unchanged.  No whitelist, no
translation — whatever keys LM Studio accepts today (or adds tomorrow)
are forwarded.  These are the parameters LM Studio documents:

| Key | Type | Purpose |
|---|---|---|
| `context_length` | int | Maximum tokens the model will consider per turn |
| `eval_batch_size` | int | Tokens processed per batch during prompt eval |
| `flash_attention` | bool | Enable FlashAttention for supported models |
| `num_experts` | int | Experts used during inference (MoE models only) |
| `offload_kv_cache_to_gpu` | bool | Keep KV cache in VRAM instead of system RAM |
| `echo_load_config` | bool | Include resolved config in the load response |

Advanced parameters that appear in LM Studio's UI (CPU thread pool,
RoPE base/scale, seed, `use_mmap`, `num_gpu_layers`, unified KV cache)
are **not currently in LM Studio's REST API** — they're only accessible
via `lmstudio-python`/`lmstudio-js` SDK calls.  Tune those through
LM Studio's preset system or GUI; the provider will pick up whatever was
configured out-of-band.

---

## Authentication

LM Studio runs unauthenticated on localhost by default, so most users
need to do nothing.

If you enable *Require API Token* in LM Studio:

1. Copy the token from the Developer tab.
2. Either:
   - Export `LMSTUDIO_API_TOKEN=<token>`, or
   - Set `plugin_configs["lmstudio"]["api_token"]` in the profile.

The token is sent as `Authorization: Bearer <token>` on both chat and
native API calls.

---

## Error types

Defined in `errors.py`:

| Exception | When raised |
|---|---|
| `LMStudioConnectionError` | Server not reachable (connection refused, timeout, network error) |
| `LMStudioModelNotFoundError` | `connect()` called with a model ID missing from the catalog |
| `LMStudioLoadError` | `POST /api/v1/models/load` returned a non-2xx (usually config incompatible with model/hardware) |
| `LMStudioAuthenticationError` | Bearer token required or rejected |

`LMStudioConnectionError` is classified as transient (retryable by the
reliability layer); `LMStudioLoadError` is deterministic and not retried
— change the load config instead.

---

## Troubleshooting

### `Cannot connect to LM Studio server at http://localhost:1234`

The local server isn't running.  Open LM Studio → Developer tab → *Start
Server*, or `lms server start`.

### `Model 'foo/bar' not found in LM Studio. Available models: …`

The model isn't downloaded.  Use LM Studio's Discover tab or
`lms get foo/bar`.  The error message lists the models that *are*
available.

### `LM Studio failed to load 'foo' (HTTP 400)`

The `load` config exceeds what the model or your hardware supports.
Common causes:

- `context_length` larger than the model's training context
- `offload_kv_cache_to_gpu: true` with insufficient VRAM
- `num_experts` higher than the model actually has

Check the response body (included in the exception) for LM Studio's
specific error, then narrow the failing parameter.

### `LM Studio rejected the bearer token`

*Require API Token* is enabled but `LMSTUDIO_API_TOKEN` is wrong or
unset.  Regenerate or copy the token from LM Studio's Developer tab.

### Chat works but the model ignores `context_length`

You're probably in **passive mode** — jaato didn't load the model.
LM Studio's currently-loaded model still has its old context length.
Move to active mode by adding `load` to `plugin_configs["lmstudio"]`,
or reload the model in LM Studio's UI with the context you want.

---

## Architecture

```
jaato/jaato-server/shared/plugins/model_provider/lmstudio/
├── __init__.py       # Package docstring, exports LMStudioProvider + create_provider
├── env.py            # LMSTUDIO_* env-var resolution
├── errors.py         # LMStudio{Connection,ModelNotFound,Load,Authentication}Error
├── provider.py       # LMStudioProvider — the main class
├── tests/
│   └── test_provider.py   # Unit tests (passive/active mode, errors, env, catalog)
└── README.md         # This file
```

### Reuse

The provider reuses two pieces from the `nim/` provider (since LM Studio
speaks the same OpenAI wire format):

- `nim._lazy` — lazy import of the `openai` SDK.
- `nim.converters` — `history_to_openai`, `tool_schemas_to_openai`,
  `response_from_openai`, `map_finish_reason`, tool-name
  sanitization.

When LM Studio adds behaviour that diverges from OpenAI's wire format
(e.g., custom reasoning fields), copy those converters here and adapt
rather than mutating `nim`.

### Profile → Provider wiring

Profile `plugin_configs["lmstudio"]` reaches the provider through:

```
SubagentProfile.plugin_configs["lmstudio"]
    ↓  JaatoServer._build_session_kwargs_from_profile()
JaatoRuntime.create_session(plugin_configs=...)
    ↓  JaatoSession.configure()
JaatoRuntime.create_provider(plugin_configs=...)
    ↓  merged into ProviderConfig.extra
LMStudioProvider.initialize(config)
    ↓  reads config.extra["host"|"load"|"api_token"|"context_length"]
```

The merge happens in `jaato_runtime.create_provider()` — only the entry
matching the effective provider name is extracted, the rest are ignored.

---

## Limitations

- **Load-control is process-global.**  LM Studio doesn't support
  multiple concurrent load configurations for the same model.  If you
  need session-level isolation, run multiple LM Studio servers on
  different ports.
- **No thinking/reasoning surface.**  LM Studio's OpenAI-compatible API
  doesn't expose reasoning deltas, so `supports_thinking()` returns
  `False` even for models that have it internally.
- **No tokenizer endpoint.**  `count_tokens()` uses a heuristic
  (~4 chars/token).  If you need exact token counts for budget
  enforcement, set `context_length` conservatively.
- **Advanced load parameters (CPU threads, RoPE, mmap, `num_gpu_layers`,
  seed, unified KV cache)** are only reachable through LM Studio's SDK,
  not its REST API.  Use LM Studio presets to configure those.

---

## See also

- [LM Studio REST API docs](https://lmstudio.ai/docs/developer/rest)
- [`ollama/`](../ollama/) — similar local-inference provider, Anthropic-compat instead
- [`nim/`](../nim/) — OpenAI-compatible base whose converters are reused here
- [`zhipuai_openai/`](../zhipuai_openai/) — sister OpenAI-compat provider this was forked from
