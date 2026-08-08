# ollama provider smoke harness

Three end-to-end smokes for the `ollama` provider, layered from simplest to most demanding:

| # | Smoke | What it validates | Profile | Harness |
|---|---|---|---|---|
| 1 | **Chat** | Provider wire — daemon can reach the local Ollama server and round-trip `/v1/messages` via the Anthropic-compatible API (Ollama v0.14.0+). **No tools, no `signal_completion`**: the profile declares no `completion_payload_schema` so `signal_completion` is hidden (2026-06-07 schema gate). The turn ends naturally when the model emits text without function calls. | `ollama-chat` | `smoke_chat.py` |
| 2 | **signal_completion** | Lifecycle — schema-driven completion contract. Profile declares a non-trivial 3-field schema (summary + status + word_count, 2 required). Model must acknowledge in text then call `signal_completion` with a schema-valid payload. Tests both the Anthropic-compatible tools wire AND `jsonschema.validate` end-to-end. | `ollama-signal_completion` | `smoke_signal_completion.py` |
| 3 | **Tools** | Full tool-calling — `cli` plugin (one shell tool call) followed by `signal_completion` (same schema as #2). Exercises the Anthropic tools shape with multiple tool surfaces. | `ollama-tools` | `smoke_tools.py` |

Run them in order. If **chat** is red, the wire is broken — fix that first.
If **signal_completion** is red but chat is green, the issue is likely a
model-fidelity gap on structured output (small ollama models tend to be
weak tool-callers). If **tools** is red but the first two are green,
the issue is the `cli` tool's schema or chained tool-result handling.

Pick a model that's strong at tool-calling for tests 2-3 (`qwen3:8b`,
`qwen3:32b`, `llama3.1:8b-instruct`). Tiny / base / non-instruct models
may pass test 1 but fail tests 2-3 from model-fidelity limits, not
framework bugs.

These are **not** unit tests — they require a live daemon and a live
local Ollama server (≥v0.14.0). Unit tests for the provider live in
`../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (host):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile YAML (literal) | Model choice IS the profile choice. To target a different model, edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile YAML | Pure profile concern. |
| `host` | workspace `.env` as `OLLAMA_HOST`, referenced from profile as `${OLLAMA_HOST}` | Endpoint detail varies per deployment (local vs LAN box vs proxy-fronted host), not per profile. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`). |

The profiles ship with `qwen3:8b` baked in (a small instruct-tuned model
that runs comfortably on a single consumer GPU and exercises tool calls
well). If you serve a different model on your Ollama install, edit the
profile — anything `ollama list` reports works (e.g. `qwen3:32b`,
`llama3.1:8b-instruct`, `mistral:7b-instruct`).

Ollama runs locally and does **not** require an API key, so there's no
key knob in `.env` — only the host.

## What's in here

```
smoke/
├── README.md                            # this file
├── bootstrap.sh                         # one-shot workspace install
├── smoke_chat.py                        # #1 — pure text round-trip
├── smoke_signal_completion.py           # #2 — lifecycle, schema-driven
├── smoke_tools.py                       # #3 — cli + signal_completion
├── .env.example                         # workspace env template
└── .jaato.example/                      # workspace .jaato/ template
    ├── profiles/
    │   ├── ollama-chat.yaml             # no schema → signal_completion hidden
    │   ├── ollama-signal_completion.yaml # 3-field schema, plugins:[]
    │   └── ollama-tools.yaml            # 3-field schema, plugins:[cli]
    └── agents/
        ├── ollama-chat.md               # text-only responder
        ├── ollama-signal_completion.md  # acknowledge + payload
        └── ollama-tools.md              # tool call + acknowledge + payload
```

## Prerequisites

- A reachable Ollama server (≥v0.14.0 for the Anthropic-compatible
  `/v1/messages` endpoint). See the parent provider's docstring
  (`../provider.py`) and the project `CLAUDE.md` for env vars.
- `ollama serve` running on the host (or a remote Ollama box reachable
  from the jaato host).
- `curl http://<host>:<port>/api/tags` returns 200 and lists the model
  that matches the profile's `model` field (default `qwen3:8b` —
  `ollama pull qwen3:8b` if absent).
- Daemon listening on `/tmp/jaato.sock`:
  `jaato-server --ipc-socket /tmp/jaato.sock --daemon`

## Quick path: `bootstrap.sh`

The included bash helper sets up a self-contained smoke install at the
target workspace (default `/tmp/jaato-ollama-smoke`):

```bash
./jaato-server/shared/plugins/model_provider/ollama/smoke/bootstrap.sh
```

After it completes, the workspace looks like:

```
/tmp/jaato-ollama-smoke/
├── smoke_chat.py                ← copied from the repo
├── smoke_signal_completion.py   ← copied from the repo
├── smoke_tools.py               ← copied from the repo
├── .env                         ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/                ← templates (qwen3:8b baked in, host → ${OLLAMA_HOST})
    └── agents/
```

### Configure the deployment

Edit `<workspace>/.env` and set `OLLAMA_HOST` to your endpoint:

```bash
$EDITOR /tmp/jaato-ollama-smoke/.env
# OLLAMA_HOST=http://192.168.1.50:11434
```

For a default local install, the shipped `http://localhost:11434` already
works — no edit needed.

### Run the smokes

Either run from the workspace:

```bash
cd /tmp/jaato-ollama-smoke
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
WS=/tmp/jaato-ollama-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/ollama/smoke
cp -f "$SMOKE/smoke_chat.py" "$SMOKE/smoke_signal_completion.py" \
    "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.yaml "$WS/.jaato/profiles/"
cp -f "$SMOKE/.jaato.example/agents/"*.md "$WS/.jaato/agents/"
cp -f "$SMOKE/.env.example" "$WS/.env"      # only if you don't already have .env
```

### 3. Configure the deployment

Edit `$WS/.env`:

```
OLLAMA_HOST=http://<your-host>:<your-port>
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
model (Qwen3 and Llama 3.1 Instruct families work well; tiny 1B models
and base completions models tend to be weak on tool calls).

### Other knobs

- **`context_length`** (default 32768) — most modern Ollama models
  support 32K+. The provider falls back to this when the model's actual
  context isn't reported.
- **Different model** — edit `model` in the profile to whatever
  `ollama list` reports. Match the tag exactly (e.g. `qwen3:32b`,
  not just `qwen3`).
- **Remote Ollama** — set `OLLAMA_HOST` in workspace `.env` to the
  remote URL. No API key needed (Ollama trusts the bearer).

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `OllamaConnectionError` | Network / Ollama-not-running: `curl http://<host>:<port>/api/tags` from the jaato host. Confirm `OLLAMA_HOST` in workspace `.env` matches and `ollama serve` is up. |
| 1 | `OllamaModelNotFoundError` | `model` in the profile doesn't match `ollama list`. Either edit the profile or `ollama pull <model>`. |
| 1 | "404 / page not found" wrapped in RuntimeError | Ollama version < 0.14.0 (the Anthropic-compatible API requires 0.14.0+). Check with `curl http://<host>:<port>/api/version`. |
| 1 | "Not enough memory" wrapped in RuntimeError | The model is too large for available RAM/VRAM. Try a smaller variant (`qwen3:8b` instead of `qwen3:32b`). |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | Model is loading from disk (cold start), or the model is hung. Check `ollama ps` on the remote. Bump `TURN_TIMEOUT_SECONDS` in the harness if cold-start exceeds 120s (the tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool` — usually a fidelity issue with smaller / non-instruct models. Try a stronger tool-calling model. |
| 1 with `NudgeExhausted: Agent loop exhausted N completion nudges` | The model responded with text but didn't call `signal_completion` to end the turn. **The wire worked** — the smoke validates provider connectivity, and a coherent text reply proves the wire end-to-end. NudgeExhausted on a weak tool-caller (small ollama models) is a **model-fidelity result, not a smoke failure**. Capable models (Claude Sonnet 4.5, GPT-4o) follow the persona's `signal_completion` instruction cleanly. The persona pattern (instruct one sentence + signal_completion call) is the canonical shape; smaller models may need richer pattern (front-load imperative, forbid alternatives) but example payloads can backfire — weak models echo the example as natural text. |
