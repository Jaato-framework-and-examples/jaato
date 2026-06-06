# ollama provider smoke harness

Two end-to-end smokes for the `ollama` provider:

| Smoke | What it validates | Profile | Harness |
|---|---|---|---|
| **Chat** | Provider wire — daemon can reach the local Ollama server and round-trip `/v1/messages` via the Anthropic-compatible API (Ollama v0.14.0+). No tools involved. | `ollama-smoke` | `smoke.py` |
| **Tools** | Anthropic tools shape — schema serialization, tool-call argument parsing, tool-result round-trip. Exercises the `cli` plugin. The `permission` plugin is server-wired automatically (no need to list it in `plugins`), but its policy is set via `plugin_configs.permission`. | `ollama-tools` | `smoke_tools.py` |

Run the **chat** smoke first. If it's red, the wire is broken and tool-shape
results would be meaningless. Once chat is green, the tools smoke tells you
whether the Anthropic tools path is intact (and gives the model a fair chance
to demonstrate tool-calling fidelity — pick a model that's good at it, e.g.
`qwen3:8b` or `llama3.1:8b-instruct`).

These are **not** unit tests — they require a live daemon and a live
local Ollama server (≥v0.14.0). Unit tests for the provider live in
`../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (host):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile JSON (literal) | Model choice IS the profile choice. To target a different model, edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile JSON | Pure profile concern. |
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
├── README.md                         # this file
├── bootstrap.sh                      # one-shot workspace install
├── smoke.py                          # chat-only harness
├── smoke_tools.py                    # tool-calling harness
├── .env.example                      # workspace env template
└── .jaato.example/                   # workspace .jaato/ template
    ├── profiles/
    │   ├── ollama-smoke.json         # pure chat, no tools, no GC
    │   └── ollama-tools.json         # cli plugin, default-allow permission
    └── agents/
        ├── ollama-smoke.md           # one-sentence-responder persona
        └── ollama-tools.md           # tool-using-then-summarize persona
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
├── smoke.py             ← copied from the repo
├── smoke_tools.py       ← copied from the repo
├── .env                 ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/        ← templates (qwen3:8b baked in, host → ${OLLAMA_HOST})
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

### Run the smoke

Either run from the workspace:

```bash
cd /tmp/jaato-ollama-smoke
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
cp -f "$SMOKE/smoke.py" "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.json "$WS/.jaato/profiles/"
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
