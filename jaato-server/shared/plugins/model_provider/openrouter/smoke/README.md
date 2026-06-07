# openrouter provider smoke harness

Three end-to-end smokes for the `openrouter` provider, layered from simplest to most demanding:

| # | Smoke | What it validates | Profile | Harness |
|---|---|---|---|---|
| 1 | **Chat** | Provider wire — daemon can reach the OpenRouter cloud gateway and round-trip `/v1/chat/completions`. **No tools, no `signal_completion`**: the profile declares no `completion_payload_schema` so `signal_completion` is hidden (2026-06-07 schema gate). The turn ends naturally when the model emits text without function calls. | `openrouter-chat` | `smoke_chat.py` |
| 2 | **signal_completion** | Lifecycle — schema-driven completion contract. Profile declares a non-trivial 3-field schema (summary + status + word_count, 2 required). Model must acknowledge in text then call `signal_completion` with a schema-valid payload. Tests both the OpenAI tools wire AND `jsonschema.validate` end-to-end (OpenRouter forwards tool config to the selected upstream). | `openrouter-signal_completion` | `smoke_signal_completion.py` |
| 3 | **Tools** | Full tool-calling — `cli` plugin (one shell tool call) followed by `signal_completion` (same schema as #2). Exercises the OpenAI tools shape with multiple tool surfaces. | `openrouter-tools` | `smoke_tools.py` |

Run them in order. If **chat** is red, the wire is broken (key, network,
or gateway) — fix that first. If **signal_completion** is red but chat
is green, the issue is either OpenRouter's upstream forwarding for the
selected model or a model-fidelity gap on structured output. If
**tools** is red but the first two are green, the issue is the `cli`
tool's schema or chained tool-result handling.

Pick a model that's strong at tool-calling for tests 2-3
(`anthropic/claude-sonnet-4.5`, `openai/gpt-4o`, Gemini 2.5). Weaker
tool-callers may pass test 1 but fail tests 2-3 from model-fidelity
limits, not framework bugs.

These are **not** unit tests — they require a live daemon and a live
OpenRouter API key (with credits). Unit tests for the provider live in
`../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (API key):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile YAML (literal) | Model choice IS the profile choice. To target a different model, edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile YAML | Pure profile concern. |
| `api_key` | workspace `.env` as `JAATO_OPENROUTER_API_KEY`, referenced from profile as `${JAATO_OPENROUTER_API_KEY}` | The key is the per-user / per-deployment secret, not a property of the profile. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`). |

The profiles ship with `anthropic/claude-sonnet-4.5` baked in. OpenRouter
also accepts `openrouter/auto` if you want the gateway to pick the model
for you, or any other `vendor/model` slug from
https://openrouter.ai/models. Edit the profile to switch.

## What's in here

```
smoke/
├── README.md                                # this file
├── bootstrap.sh                             # one-shot workspace install
├── smoke_chat.py                            # #1 — pure text round-trip
├── smoke_signal_completion.py               # #2 — lifecycle, schema-driven
├── smoke_tools.py                           # #3 — cli + signal_completion
├── .env.example                             # workspace env template
└── .jaato.example/                          # workspace .jaato/ template
    ├── profiles/
    │   ├── openrouter-chat.yaml             # no schema → signal_completion hidden
    │   ├── openrouter-signal_completion.yaml # 3-field schema, plugins:[]
    │   └── openrouter-tools.yaml            # 3-field schema, plugins:[cli]
    └── agents/
        ├── openrouter-chat.md               # text-only responder
        ├── openrouter-signal_completion.md  # acknowledge + payload
        └── openrouter-tools.md              # tool call + acknowledge + payload
```

## Prerequisites

- An OpenRouter API key (`sk-or-v1-...`) from
  https://openrouter.ai/settings/keys with credits on the account.
- Network reachability to `https://openrouter.ai` from the jaato host.
- The model slug in the profile is one your account can access — visit
  https://openrouter.ai/models to verify, or set the profile's `model`
  to `openrouter/auto` to let the gateway pick.
- Daemon listening on `/tmp/jaato.sock`:
  `jaato-server --ipc-socket /tmp/jaato.sock --daemon`

## Quick path: `bootstrap.sh`

The included bash helper sets up a self-contained smoke install at the
target workspace (default `/tmp/jaato-openrouter-smoke`):

```bash
./jaato-server/shared/plugins/model_provider/openrouter/smoke/bootstrap.sh
```

After it completes, the workspace looks like:

```
/tmp/jaato-openrouter-smoke/
├── smoke_chat.py                ← copied from the repo
├── smoke_signal_completion.py   ← copied from the repo
├── smoke_tools.py               ← copied from the repo
├── .env                         ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/                ← templates (Claude Sonnet 4.5 baked in, api_key → ${JAATO_OPENROUTER_API_KEY})
    └── agents/
```

### Configure the deployment

Edit `<workspace>/.env` and set `JAATO_OPENROUTER_API_KEY` to your key:

```bash
$EDITOR /tmp/jaato-openrouter-smoke/.env
# JAATO_OPENROUTER_API_KEY=sk-or-v1-your-key-here
# or, recommended:
# JAATO_OPENROUTER_API_KEY=pass://providers/openrouter/api_key
```

A `pass://` URI is resolved daemon-side and keeps the literal token out
of the workspace tree — see the project CLAUDE.md "Workspace .env values
resolve secret URIs" note.

### Run the smokes

Either run from the workspace:

```bash
cd /tmp/jaato-openrouter-smoke
<repo>/.venv/bin/python smoke_chat.py               # 1: pure text round-trip
<repo>/.venv/bin/python smoke_signal_completion.py  # 2: signal_completion w/ schema
<repo>/.venv/bin/python smoke_tools.py              # 3: cli tool + signal_completion
```

…or re-invoke `bootstrap.sh` with `--run chat` / `--run signal_completion` /
`--run tools` to do the bootstrap + run step in one command:

```bash
./bootstrap.sh --run chat
./bootstrap.sh --run signal_completion
./bootstrap.sh --run tools
```

`bootstrap.sh` is idempotent — re-running never clobbers an existing
workspace `.env` (your edits survive). The harness scripts and profile
templates get re-copied each time.

The script does not manage daemon lifecycle. Start the daemon
separately before running the smoke.

## Manual path

If you want full control over each step (or you're scripting the
bootstrap into a larger setup), here's what `bootstrap.sh` is doing
under the hood.

### 1. Create the workspace dirs

```bash
WS=/tmp/jaato-openrouter-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/openrouter/smoke
cp -f "$SMOKE/smoke_chat.py" "$SMOKE/smoke_signal_completion.py" \
    "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.yaml "$WS/.jaato/profiles/"
cp -f "$SMOKE/.jaato.example/agents/"*.md "$WS/.jaato/agents/"
cp -f "$SMOKE/.env.example" "$WS/.env"      # only if you don't already have .env
```

### 3. Configure the deployment

Edit `$WS/.env`:

```
JAATO_OPENROUTER_API_KEY=sk-or-v1-...
```

### 4. Run the chat smoke first

```bash
cd "$WS"
<repo>/.venv/bin/python smoke_chat.py
```

Expected: one sentence of model output, exit 0.

### 5. Then the signal_completion smoke

```bash
cd "$WS"
<repo>/.venv/bin/python smoke_signal_completion.py
```

Expected: a brief acknowledgement followed by a `signal_completion` call
with a schema-valid payload, exit 0.

### 6. Finally the tools smoke

```bash
cd "$WS"
<repo>/.venv/bin/python smoke_tools.py
```

Expected: a `cli_based_tool` call running `ls /tmp`, followed by a
one-sentence summary, followed by a `signal_completion` call, exit 0.
The full conversation including the tool call streams to stdout via
the SDK's output events.

If the model loops or refuses to call the tool, that's a **model
fidelity** issue, not a provider bug — try a stronger tool-calling
model (Claude Sonnet 4.5, GPT-4o, and Gemini 2.5 work well).

### Other knobs

- **Model** — edit `model` in the profile to any slug from
  https://openrouter.ai/models. Use `openrouter/auto` to let the gateway
  pick.
- **`context_length`** — usually unnecessary; the provider discovers it
  from the OpenRouter model catalog. Override under
  `plugin_configs.openrouter.framework_overrides.context_length` if a
  specific upstream reports the wrong window.
- **Routing** — `plugin_configs.openrouter.routing` configures the
  OpenRouter `provider` extension (preferred upstreams, quantization,
  price caps, etc.); see the project CLAUDE.md "OpenRouter" section.
- **App attribution** — `JAATO_OPENROUTER_HTTP_REFERER` /
  `JAATO_OPENROUTER_APP_TITLE` / `JAATO_OPENROUTER_APP_CATEGORIES` work
  in `.env` if you want to override the defaults.

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `APIKeyNotFoundError` | `JAATO_OPENROUTER_API_KEY` not set or not visible to the daemon. Confirm the workspace `.env` has it set; if you used a `pass://` URI, `pass show <path>` must succeed under the daemon's gpg-agent. |
| 1 | `AuthenticationError` | Key was rejected. Test the resolved bearer with `curl -H "Authorization: Bearer <key>" https://openrouter.ai/api/v1/key` before touching framework code. |
| 1 | `ModelNotFoundError` | `model` in the profile is a slug your account can't access. Edit the profile (default `anthropic/claude-sonnet-4.5`); list options at https://openrouter.ai/models. |
| 1 | `ContextLimitError` | Conversation grew past the model's window. Reset, shorten the prompt, or switch to a larger-window model. |
| 1 | `InfrastructureError` | Transient 5xx / network. The provider retries automatically; if it still surfaces, OpenRouter or your network is degraded. |
| 1 | `RateLimitError` | Per-key throughput exceeded. Wait for the retry window, top up credits, or change models. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | Upstream model is slow or hung. Bump `TURN_TIMEOUT_SECONDS` in the harness (the tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool` — usually a fidelity issue with smaller models. Try Claude Sonnet 4.5, GPT-4o, or Gemini 2.5. |
| 1 with `APIStatusError 402: Prompt tokens limit exceeded: N > M` | Your OpenRouter account's remaining prompt-token credit (M) is below the framework's irreducible prompt-token floor (~15K for a `ClientType.API` session, even with `suppress_base_instructions: true` and `plugins: []` — the floor is the tool-schemas array: `signal_completion` + the always-on introspection tools `list_tools` / `get_tool_schemas` etc. — see saved-lesson `feedback_introspection_tools_in_array_by_design`). Top up credits at https://openrouter.ai/settings/credits and retry. |
| 1 with `APIStatusError 402: You requested up to N tokens, but can only afford M` | The provider was asking for too many output tokens. The profiles cap `api_params.max_tokens: 256` which avoids this — if you still see it, you're running an older daemon that pre-dates the `api_params.max_tokens` wiring fix (PR #233). |
| 1 with `NudgeExhausted: Agent loop exhausted N completion nudges` | The model responded with text but didn't call `signal_completion` to end the turn. **The wire worked** — a coherent text reply proves provider connectivity. NudgeExhausted on a weak tool-caller (small models on local providers) is a **model-fidelity result, not a smoke failure**. Claude Sonnet 4.5 / GPT-4o / Gemini 2.5 follow the persona's `signal_completion` instruction cleanly. |
