# anthropic provider smoke harness

Two end-to-end smokes for the `anthropic` provider:

| Smoke | What it validates | Profile | Harness |
|---|---|---|---|
| **Chat** | Provider wire — daemon can reach `api.anthropic.com` and round-trip a Messages API call. No tools involved. | `anthropic-smoke` | `smoke.py` |
| **Tools** | Anthropic tool-use shape — schema serialization, tool-call argument parsing, tool-result round-trip. Exercises the `cli` plugin. The `permission` plugin is server-wired automatically (no need to list it in `plugins`), but its policy is set via `plugin_configs.permission`. | `anthropic-tools` | `smoke_tools.py` |

Run the **chat** smoke first. If it's red, the wire is broken (bad key,
network, or model id) and tool-shape results would be meaningless. Once
chat is green, the tools smoke tells you whether the Anthropic
tool-use path is intact.

These are **not** unit tests — they require a live daemon and a live
Anthropic credential. Unit tests for the provider live in `../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (credential):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile YAML (literal) | Model choice IS the profile choice. To target a different model, edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile YAML | Pure profile concern. |
| API key | workspace `.env` as `ANTHROPIC_API_KEY`, referenced from profile as `${ANTHROPIC_API_KEY}` | Credential value varies per developer / per environment, not per profile. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`), then promoted to `ProviderConfig.api_key` by `jaato_runtime.py:1149-1162`. |

Unlike self-hosted providers (tensorrt_llm, ollama, lmstudio), there is
no `host` to vary — the anthropic provider always talks to
`api.anthropic.com`. The only deployment knob is the credential itself.

The profiles ship with `claude-sonnet-4-6` baked in. To target a
different Claude model, edit the profile.

### API key vs OAuth token

The `.env.example` defaults to `ANTHROPIC_API_KEY` (uses API credits).
If you have a Claude Pro/Max subscription instead, generate an OAuth
token with `claude setup-token`, set `ANTHROPIC_AUTH_TOKEN` in the
workspace `.env`, and change the profile's
`plugin_configs.anthropic.api_key` to
`plugin_configs.anthropic.oauth_token: "${ANTHROPIC_AUTH_TOKEN}"`. The
provider handles both auth modes; the profile just selects which one
the workspace exposes.

## What's in here

```
smoke/
├── README.md                            # this file
├── bootstrap.sh                         # one-shot workspace install
├── smoke.py                             # chat-only harness
├── smoke_tools.py                       # tool-calling harness
├── .env.example                         # workspace env template
└── .jaato.example/                      # workspace .jaato/ template
    ├── profiles/
    │   ├── anthropic-smoke.yaml         # pure chat, no tools, no GC
    │   └── anthropic-tools.yaml         # cli plugin, default-allow permission
    └── agents/
        ├── anthropic-smoke.md           # one-sentence-responder persona
        └── anthropic-tools.md           # tool-using-then-summarize persona
```

## Prerequisites

- An Anthropic API key (or OAuth token) — see CLAUDE.md "Anthropic
  Claude" section for env vars and `oauth_login()` flow.
- `curl https://api.anthropic.com/v1/models -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"`
  returns 200 from the jaato host (network reachable, key valid).
- The model `id` returned by `/v1/models` matches the profile's `model`
  field (default `claude-sonnet-4-6`).
- Daemon listening on `/tmp/jaato.sock`:
  `jaato-server --ipc-socket /tmp/jaato.sock --daemon`

## Quick path: `bootstrap.sh`

The included bash helper sets up a self-contained smoke install at the
target workspace (default `/tmp/jaato-anthropic-smoke`):

```bash
./jaato-server/shared/plugins/model_provider/anthropic/smoke/bootstrap.sh
```

After it completes, the workspace looks like:

```
/tmp/jaato-anthropic-smoke/
├── smoke.py             ← copied from the repo
├── smoke_tools.py       ← copied from the repo
├── .env                 ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/        ← templates (claude-sonnet-4-6 baked in, key → ${ANTHROPIC_API_KEY})
    └── agents/
```

### Configure the deployment

Edit `<workspace>/.env` and set `ANTHROPIC_API_KEY` to your credential:

```bash
$EDITOR /tmp/jaato-anthropic-smoke/.env
# ANTHROPIC_API_KEY=sk-ant-api03-...
```

A `pass://` URI works too — the daemon resolves it before shipping to
the runner, and the literal never touches disk in plain text.

### Run the smoke

Either run from the workspace:

```bash
cd /tmp/jaato-anthropic-smoke
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
WS=/tmp/jaato-anthropic-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/anthropic/smoke
cp -f "$SMOKE/smoke.py" "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.yaml "$WS/.jaato/profiles/"
cp -f "$SMOKE/.jaato.example/agents/"*.md "$WS/.jaato/agents/"
cp -f "$SMOKE/.env.example" "$WS/.env"      # only if you don't already have .env
```

### 3. Configure the deployment

Edit `$WS/.env`:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
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
fidelity** issue, not a provider bug — Claude Sonnet/Opus models are
generally strong on tool use, so a loop here usually means the
schema-serialization path regressed.

### Other knobs

- **`plugin_configs.anthropic.oauth_token`** — use a Claude Pro/Max
  OAuth token instead of an API key (see "API key vs OAuth token" above).
- **`plugin_configs.anthropic.api_params.max_tokens`** — cap the
  response size.  Defaults to the provider's `DEFAULT_MAX_TOKENS`
  (8192).
- **Different model** — edit `model` in the profile. Available ids
  surface from `/v1/models` on the Anthropic API.

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `APIKeyNotFoundError` | `${ANTHROPIC_API_KEY}` didn't resolve. Check the workspace `.env` exists, contains the key, and the daemon read it (look at `~/.jaato/logs/` for the env-load line). |
| 1 | `APIKeyInvalidError` | The key was sent but the API rejected it. Test the resolved bearer with `curl -H "x-api-key: <key>" -H "anthropic-version: 2023-06-01" https://api.anthropic.com/v1/models` before touching framework code. |
| 1 | `ModelNotFoundError` | The `model` in the profile doesn't match the API. Edit the profile to use a Claude model id `/v1/models` reports. |
| 1 | `RateLimitError` / `OverloadedError` | Transient — retry. If persistent, check your usage at `console.anthropic.com`. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | API is slow or hung. Bump `TURN_TIMEOUT_SECONDS` in the harness (tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool`. Rare for Claude Sonnet/Opus — usually means the request was answerable from priors. Stronger prompt usually fixes it. |
| 1 with `NudgeExhausted: Agent loop exhausted N completion nudges` | The model responded with text but didn't call `signal_completion` to end the turn. **The wire worked** — the smoke validates provider connectivity, and a coherent text reply proves the wire end-to-end. NudgeExhausted on a weak tool-caller (Qwen-7B on anthropic, small ollama models, etc.) is a **model-fidelity result, not a smoke failure**. Capable models (Claude Sonnet 4.5, GPT-4o) follow the persona's `signal_completion` instruction cleanly. The persona pattern (instruct one sentence + signal_completion call) is the canonical shape; smaller models may need richer pattern (front-load imperative, forbid alternatives) but example payloads can backfire — weak models echo the example as natural text. |
