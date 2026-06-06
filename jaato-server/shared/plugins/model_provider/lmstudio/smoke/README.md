# lmstudio provider smoke harness

Two end-to-end smokes for the `lmstudio` provider:

| Smoke | What it validates | Profile | Harness |
|---|---|---|---|
| **Chat** | Provider wire — daemon can reach the local LM Studio server and round-trip `/v1/chat/completions`. No tools involved. | `lmstudio-smoke` | `smoke.py` |
| **Tools** | OpenAI tools shape — schema serialization, tool-call argument parsing, tool-result round-trip. Exercises the `cli` plugin. The `permission` plugin is server-wired automatically (no need to list it in `plugins`), but its policy is set via `plugin_configs.permission`. | `lmstudio-tools` | `smoke_tools.py` |

Run the **chat** smoke first. If it's red, the wire is broken and tool-shape
results would be meaningless. Once chat is green, the tools smoke tells you
whether the OpenAI tools path is intact (and gives the model a fair chance
to demonstrate tool-calling fidelity — pick a model that's good at it, e.g.
`openai/gpt-oss-20b`, `qwen2.5-7b-instruct`, or a Llama-3.1-8B variant).

These are **not** unit tests — they require a live daemon and a live local
LM Studio server. Unit tests for the provider live in `../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (host, optional bearer token):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile YAML (literal) | Model choice IS the profile choice. To target a different model, edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile YAML | Pure profile concern. |
| `host` | workspace `.env` as `LMSTUDIO_HOST`, referenced from profile as `${LMSTUDIO_HOST}` | Endpoint detail varies per deployment, not per profile. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`). |
| `api_token` (optional) | workspace `.env` as `LMSTUDIO_API_TOKEN` | Only required when LM Studio is configured with "Require API Token". Most local-dev installs don't need it; the .env.example leaves it commented out. |

The profiles ship with `openai/gpt-oss-20b` baked in. If you have a
different model loaded in LM Studio, edit the profile.

## What's in here

```
smoke/
├── README.md                              # this file
├── bootstrap.sh                           # one-shot workspace install
├── smoke.py                               # chat-only harness
├── smoke_tools.py                         # tool-calling harness
├── .env.example                           # workspace env template
└── .jaato.example/                        # workspace .jaato/ template
    ├── profiles/
    │   ├── lmstudio-smoke.yaml            # pure chat, no tools, no GC
    │   └── lmstudio-tools.yaml            # cli plugin, default-allow permission
    └── agents/
        ├── lmstudio-smoke.md              # one-sentence-responder persona
        └── lmstudio-tools.md              # tool-using-then-summarize persona
```

## Prerequisites

- A running LM Studio server (Developer tab → Start Server, or
  `lms server start`). Default URL is `http://localhost:1234`.
- The model named in the profile is **loaded** in LM Studio (either
  via the UI's Chat tab or `lms load openai/gpt-oss-20b`). The
  lmstudio provider is **passive** — it does not auto-load the model
  unless the profile supplies a `plugin_configs.lmstudio.load` dict.
- `curl http://<host>:<port>/api/v0/models` returns the model `id` that
  matches the profile's `model` field.
- Daemon listening on `/tmp/jaato.sock`:
  `jaato-server --ipc-socket /tmp/jaato.sock --daemon`

## Quick path: `bootstrap.sh`

The included bash helper sets up a self-contained smoke install at the
target workspace (default `/tmp/jaato-lmstudio-smoke`):

```bash
./jaato-server/shared/plugins/model_provider/lmstudio/smoke/bootstrap.sh
```

After it completes, the workspace looks like:

```
/tmp/jaato-lmstudio-smoke/
├── smoke.py             ← copied from the repo
├── smoke_tools.py       ← copied from the repo
├── .env                 ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/        ← templates (openai/gpt-oss-20b baked in, host → ${LMSTUDIO_HOST})
    └── agents/
```

### Configure the deployment

Edit `<workspace>/.env` and set `LMSTUDIO_HOST` to your endpoint:

```bash
$EDITOR /tmp/jaato-lmstudio-smoke/.env
# LMSTUDIO_HOST=http://localhost:1234
# LMSTUDIO_API_TOKEN=<only if LM Studio has Require API Token enabled>
```

### Run the smoke

Either run from the workspace:

```bash
cd /tmp/jaato-lmstudio-smoke
<repo>/.venv/bin/python smoke.py         # chat smoke
<repo>/.venv/bin/python smoke_tools.py   # tools smoke
```

…or re-invoke `bootstrap.sh` with `--run chat` / `--run tools` to do the
bootstrap + run step in one command:

```bash
./bootstrap.sh --run chat
./bootstrap.sh --run tools
```

`bootstrap.sh` is idempotent — re-running never clobbers an existing
workspace `.env` (your edits survive). The harness scripts and profile
templates get re-copied each time.

The script does not manage daemon lifecycle. Start the daemon
separately before running the smoke. It also does not manage LM Studio
itself — make sure the server is running and the target model is loaded
before invoking the smoke.

## Manual path

If you want full control over each step (or you're scripting the
bootstrap into a larger setup), here's what `bootstrap.sh` is doing
under the hood.

### 1. Create the workspace dirs

```bash
WS=/tmp/jaato-lmstudio-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/lmstudio/smoke
cp -f "$SMOKE/smoke.py" "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.yaml "$WS/.jaato/profiles/"
cp -f "$SMOKE/.jaato.example/agents/"*.md "$WS/.jaato/agents/"
cp -f "$SMOKE/.env.example" "$WS/.env"      # only if you don't already have .env
```

### 3. Configure the deployment

Edit `$WS/.env`:

```
LMSTUDIO_HOST=http://localhost:1234
# LMSTUDIO_API_TOKEN=<bearer-token>   # only if LM Studio requires it
```

### 4. Run the chat smoke first

```bash
cd "$WS"
<repo>/.venv/bin/python smoke.py
```

Expected: one sentence of model output, exit 0.

### 5. Then run the tools smoke

```bash
cd "$WS"
<repo>/.venv/bin/python smoke_tools.py
```

Expected: a `cli_based_tool` call running `ls /tmp`, followed by a
one-sentence summary, exit 0. The full conversation including the tool
call streams to stdout via the SDK's output events.

If the model loops or refuses to call the tool, that's a **model
fidelity** issue, not a provider bug — try a stronger tool-calling
model. `openai/gpt-oss-20b` and Qwen2.5-7B-Instruct work well on
LM Studio; smaller / older models may be weak on tool calls.

### Other knobs

- **`context_length`** — LM Studio's `/api/v0/models` surfaces each
  model's real `max_context_length`, so the provider discovers it
  automatically. Override via `plugin_configs.lmstudio.context_length`
  only when you need to clamp it lower (e.g. memory constraints).
- **`plugin_configs.lmstudio.api_token`** — only when LM Studio's
  server is configured with "Require API Token". A `pass://` URI
  resolves daemon-side; a literal token also works but don't commit
  it. The .env path (`LMSTUDIO_API_TOKEN`) is usually simpler.
- **`plugin_configs.lmstudio.load`** — optional dict POSTed to
  `/api/v1/models/load` before the first chat. Use this to reconfigure
  the in-memory model (context length, GPU offload, KV-cache placement,
  etc.). The default smoke profiles omit this — the user is expected to
  pre-load the model. See the project `CLAUDE.md` for the keys.
- **Different model** — edit `model` in the profile to whatever
  `/api/v0/models` reports.

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `LMStudioConnectionError` | LM Studio's local server isn't running. Open the Developer tab → Start Server, or `lms server start`. Confirm `LMSTUDIO_HOST` in workspace `.env` matches the URL LM Studio reports. |
| 1 | `LMStudioModelNotFoundError` | The model in the profile isn't loaded / downloaded. `curl http://<host>:<port>/api/v0/models` to see what LM Studio actually reports, then either load the right model (`lms load <id>`) or edit the profile. |
| 1 | `LMStudioAuthenticationError` | LM Studio has "Require API Token" enabled but the token is missing or wrong. Copy the token from LM Studio's Developer tab and set `LMSTUDIO_API_TOKEN` in the workspace `.env`. Test it with `curl -H "Authorization: Bearer <token>" http://<host>:<port>/api/v0/models` before touching framework code. |
| 1 | `LMStudioLoadError` | Only fires when the profile supplies a `plugin_configs.lmstudio.load` dict and LM Studio rejects it (context length too large, GPU offload impossible, etc.). The error body usually says exactly which key tripped it. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | The model is loading on first request (cold start), or generation is unusually slow on CPU-only / small GPU. Watch LM Studio's terminal logs to see if generation is actually progressing. Bump `TURN_TIMEOUT_SECONDS` in the harness if cold-start exceeds 120s (the tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool` — usually a fidelity issue with smaller models. Try `openai/gpt-oss-20b` or a Qwen2.5-7B-Instruct variant. |
| 1 with `NudgeExhausted: Agent loop exhausted N completion nudges` | The model responded with text but didn't call `signal_completion` to end the turn. **The wire worked** — the smoke validates provider connectivity, and a coherent text reply proves the wire end-to-end. NudgeExhausted on a weak tool-caller (Qwen-7B on LM Studio, small ollama models, etc.) is a **model-fidelity result, not a smoke failure**. Capable models (Claude Sonnet 4.5, GPT-4o) follow the persona's `signal_completion` instruction cleanly. The persona pattern (instruct one sentence + signal_completion call) is the canonical shape; smaller models may need richer pattern (front-load imperative, forbid alternatives) but example payloads can backfire — weak models echo the example as natural text. |
