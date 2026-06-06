# nim provider smoke harness

Two end-to-end smokes for the `nim` provider:

| Smoke | What it validates | Profile | Harness |
|---|---|---|---|
| **Chat** | Provider wire — daemon can reach the NIM endpoint (hosted or self-hosted) and round-trip `/v1/chat/completions`. No tools involved. | `nim-smoke` | `smoke.py` |
| **Tools** | OpenAI tools shape — schema serialization, tool-call argument parsing, tool-result round-trip. Exercises the `cli` plugin. The `permission` plugin is server-wired automatically (no need to list it in `plugins`), but its policy is set via `plugin_configs.permission`. | `nim-tools` | `smoke_tools.py` |

Run the **chat** smoke first. If it's red, the wire is broken and tool-shape
results would be meaningless. Once chat is green, the tools smoke tells you
whether the OpenAI tools path is intact (and gives the model a fair chance
to demonstrate tool-calling fidelity — pick a model that's good at it, e.g.
Llama-3.1-8B/70B-Instruct or Nemotron variants).

These are **not** unit tests — they require a live daemon and a live NIM
endpoint. Unit tests for the provider live in `../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (auth, endpoint):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile JSON (literal) | Model choice IS the profile choice. To target a different model, edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` (except `api_key`) | profile JSON | Pure profile concern. |
| `api_key` | workspace `.env` as `JAATO_NIM_API_KEY`, referenced from profile as `${JAATO_NIM_API_KEY}` | Credential, varies per deployment. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`), then promoted to `ProviderConfig.api_key` (`jaato_runtime.py` PR-149). |
| `base_url` | workspace `.env` as `JAATO_NIM_BASE_URL` (optional) | Endpoint varies per deployment. Picked up directly by the provider via `resolve_base_url()`; leave unset to use NVIDIA's hosted API (`https://integrate.api.nvidia.com/v1`). |

The profiles ship with `meta/llama-3.1-8b-instruct` baked in (a NIM
catalog standard with reliable tool-calling). If you want a different
model, edit the profile.

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
    │   ├── nim-smoke.json                 # pure chat, no tools, no GC
    │   └── nim-tools.json                 # cli plugin, default-allow permission
    └── agents/
        ├── nim-smoke.md                   # one-sentence-responder persona
        └── nim-tools.md                   # tool-using-then-summarize persona
```

## Prerequisites

- An API key from <https://build.nvidia.com/> (sign in, click "Get API
  Key" on any model card) — required for the hosted endpoint. For a
  self-hosted NIM container, set `JAATO_NIM_BASE_URL` to the container
  endpoint and the key becomes optional.
- `curl https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $JAATO_NIM_API_KEY"`
  returns the model catalog (sanity check before involving the daemon).
- Daemon listening on `/tmp/jaato.sock`:
  `jaato-server --ipc-socket /tmp/jaato.sock --daemon`

## Quick path: `bootstrap.sh`

The included bash helper sets up a self-contained smoke install at the
target workspace (default `/tmp/jaato-nim-smoke`):

```bash
./jaato-server/shared/plugins/model_provider/nim/smoke/bootstrap.sh
```

After it completes, the workspace looks like:

```
/tmp/jaato-nim-smoke/
├── smoke.py             ← copied from the repo
├── smoke_tools.py       ← copied from the repo
├── .env                 ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/        ← templates (Llama-3.1-8B baked in, api_key → ${JAATO_NIM_API_KEY})
    └── agents/
```

### Configure the deployment

Edit `<workspace>/.env` and set `JAATO_NIM_API_KEY`:

```bash
$EDITOR /tmp/jaato-nim-smoke/.env
# JAATO_NIM_API_KEY=nvapi-...
# (Optionally uncomment JAATO_NIM_BASE_URL for a self-hosted container.)
```

### Run the smoke

Either run from the workspace:

```bash
cd /tmp/jaato-nim-smoke
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
WS=/tmp/jaato-nim-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/nim/smoke
cp -f "$SMOKE/smoke.py" "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.json "$WS/.jaato/profiles/"
cp -f "$SMOKE/.jaato.example/agents/"*.md "$WS/.jaato/agents/"
cp -f "$SMOKE/.env.example" "$WS/.env"      # only if you don't already have .env
```

### 3. Configure the deployment

Edit `$WS/.env`:

```
JAATO_NIM_API_KEY=nvapi-...
# JAATO_NIM_BASE_URL=http://localhost:8000/v1   # self-hosted container only
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
fidelity** issue, not a provider bug — try a larger model
(`meta/llama-3.1-70b-instruct` is more reliable for tool calls than
the 8B variant).

### Other knobs

- **`context_length`** (default 8192) — both profiles ship with this
  cap. Bump it for long-context Llama / Nemotron models.
- **Different model** — edit `model` in the profile to any ID from
  the NIM catalog (`meta/llama-3.1-70b-instruct`,
  `nvidia/llama-3.1-nemotron-70b-instruct`,
  `deepseek-ai/deepseek-r1`, etc.).
- **Self-hosted NIM container** — set `JAATO_NIM_BASE_URL` to the
  container endpoint (e.g. `http://localhost:8000/v1`). The provider
  treats self-hosted endpoints (localhost / 192.168.* / 10.*) as
  not requiring an API key, so `JAATO_NIM_API_KEY` can stay unset.

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `APIKeyNotFoundError` | `JAATO_NIM_API_KEY` is not set in the workspace `.env`, or stored credentials are missing. Run `nim-auth key <nvapi-...>` to store one, or set the env var. |
| 1 | `AuthenticationError` | Key is set but rejected. Test it before touching framework code: `curl -H "Authorization: Bearer $JAATO_NIM_API_KEY" https://integrate.api.nvidia.com/v1/models`. Regenerate at build.nvidia.com if needed. |
| 1 | `ModelNotFoundError` | `model` in the profile is not in the NIM catalog (or not loaded in your self-hosted container). Edit the profile to a valid ID (`meta/llama-3.1-8b-instruct`, `meta/llama-3.1-70b-instruct`, ...). |
| 1 | `ContextLimitError` | Prompt + history exceeds the model's window. Bump `context_length` in the profile to the model's real limit. |
| 1 | `RateLimitError` | NVIDIA's hosted API throttled you. Wait and retry, or switch to a self-hosted container. |
| 1 | `InfrastructureError` | Transient 5xx / network error. The framework retries automatically; persistent failures point to a real outage. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | Endpoint is slow or hung. Bump `TURN_TIMEOUT_SECONDS` in the harness if cold-start exceeds 120s (the tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool` — usually a fidelity issue with smaller models. Try `meta/llama-3.1-70b-instruct`. |
