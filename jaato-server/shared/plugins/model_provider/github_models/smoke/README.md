# github_models provider smoke harness

Three end-to-end smokes for the `github_models` provider, layered from simplest to most demanding:

| # | Smoke | What it validates | Profile | Harness |
|---|---|---|---|---|
| 1 | **Chat** | Provider wire — daemon can reach the managed GitHub Models endpoint and round-trip `/chat/completions`. **No tools, no `signal_completion`**: the profile declares no `completion_payload_schema` so `signal_completion` is hidden (2026-06-07 schema gate). The turn ends naturally when the model emits text without function calls. | `github-models-chat` | `smoke_chat.py` |
| 2 | **signal_completion** | Lifecycle — schema-driven completion contract. Profile declares a non-trivial 3-field schema (summary + status + word_count, 2 required). Model must acknowledge in text then call `signal_completion` with a schema-valid payload. Tests both the OpenAI tools wire AND `jsonschema.validate` end-to-end. | `github-models-signal_completion` | `smoke_signal_completion.py` |
| 3 | **Tools** | Full tool-calling — `cli` plugin (one shell tool call) followed by `signal_completion` (same schema as #2). Exercises the OpenAI tools shape with multiple tool surfaces. The `permission` plugin is server-wired automatically (no need to list it in `plugins`), but its policy is set via `plugin_configs.permission`. | `github-models-tools` | `smoke_tools.py` |

Run them in order. If **chat** is red, the wire is broken — fix that
first. If **signal_completion** is red but chat is green, the issue is
either the OpenAI tools serialization path or a model-fidelity gap on
structured output. If **tools** is red but the first two are green,
the issue is the `cli` tool's schema or chained tool-result handling.

Pick a model that's strong at tool-calling for tests 2-3 (e.g.
`openai/gpt-4o`, `anthropic/claude-3.5-sonnet`). Smaller / open-weights
catalog entries may pass test 1 but fail tests 2-3 from model-fidelity
limits, not framework bugs.

These are **not** unit tests — they require a live daemon and live
GitHub credentials. Unit tests for the provider live in `../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (`GITHUB_TOKEN`):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile YAML (literal) | Model choice IS the profile choice. To target a different model (e.g. `anthropic/claude-3.5-sonnet`, `meta/Llama-3.3-70B-Instruct`), edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile YAML | Pure profile concern. |
| `GITHUB_TOKEN` | workspace `.env`, referenced from profile as `${GITHUB_TOKEN}` in `plugin_configs.github_models.api_key` | Credential varies per deployment, not per profile. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`). |

The profiles ship with `openai/gpt-4o` baked in. To target a different
model from the GitHub Models catalog, edit the `model` field.

GitHub Models is a **managed cloud endpoint** — there is no host knob.
The token is the only deployment-time variable.

## What's in here

```
smoke/
├── README.md                                    # this file
├── bootstrap.sh                                 # one-shot workspace install
├── smoke_chat.py                                # #1 — pure text round-trip
├── smoke_signal_completion.py                   # #2 — lifecycle, schema-driven
├── smoke_tools.py                               # #3 — cli + signal_completion
├── .env.example                                 # workspace env template
└── .jaato.example/                              # workspace .jaato/ template
    ├── profiles/
    │   ├── github-models-chat.yaml              # no schema → signal_completion hidden
    │   ├── github-models-signal_completion.yaml # 3-field schema, plugins:[]
    │   └── github-models-tools.yaml             # 3-field schema, plugins:[cli]
    └── agents/
        ├── github-models-chat.md                # text-only responder
        ├── github-models-signal_completion.md   # acknowledge + payload
        └── github-models-tools.md               # tool call + acknowledge + payload
```

## Prerequisites

- A valid GitHub credential. Two options:
  - **OAuth (recommended for Copilot)**: run `github-auth login` once; the
    provider picks up the stored OAuth token automatically and `GITHUB_TOKEN`
    in the workspace `.env` can be left as the placeholder.
  - **Personal Access Token**: create a PAT at
    https://github.com/settings/tokens with the `models: read` permission
    (fine-grained PATs are auto-authorized for SSO orgs; classic PATs
    require per-org SSO authorization). Set `GITHUB_TOKEN` in the
    workspace `.env`.
- Network reachability to `https://models.github.ai/inference` (or
  `https://api.githubcopilot.com` for OAuth).
- Optional: `JAATO_GITHUB_ORGANIZATION` for org-attributed billing.
- Daemon listening on `/tmp/jaato.sock`:
  `jaato-server --ipc-socket /tmp/jaato.sock --daemon`

## Quick path: `bootstrap.sh`

The included bash helper sets up a self-contained smoke install at the
target workspace (default `/tmp/jaato-github-models-smoke`):

```bash
./jaato-server/shared/plugins/model_provider/github_models/smoke/bootstrap.sh
```

After it completes, the workspace looks like:

```
/tmp/jaato-github-models-smoke/
├── smoke_chat.py                ← copied from the repo
├── smoke_signal_completion.py   ← copied from the repo
├── smoke_tools.py               ← copied from the repo
├── .env                         ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/                ← templates (gpt-4o baked in, token → ${GITHUB_TOKEN})
    └── agents/
```

### Configure the deployment

Either run `github-auth login` (preferred, no `.env` edit needed), or
edit `<workspace>/.env` and set `GITHUB_TOKEN`:

```bash
$EDITOR /tmp/jaato-github-models-smoke/.env
# GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

### Run the smoke

Either run from the workspace:

```bash
cd /tmp/jaato-github-models-smoke
<repo>/.venv/bin/python smoke_chat.py               # 1: pure text, no tools
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
WS=/tmp/jaato-github-models-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/github_models/smoke
cp -f "$SMOKE/smoke_chat.py" "$SMOKE/smoke_signal_completion.py" "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.yaml "$WS/.jaato/profiles/"
cp -f "$SMOKE/.jaato.example/agents/"*.md "$WS/.jaato/agents/"
cp -f "$SMOKE/.env.example" "$WS/.env"      # only if you don't already have .env
```

### 3. Configure the deployment

Either run `github-auth login` (uses Copilot device-code OAuth, no `.env`
edit needed), or edit `$WS/.env`:

```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

### 4. Run the chat smoke first

```bash
cd "$WS"
<repo>/.venv/bin/python smoke_chat.py
```

Expected: one sentence of model output, exit 0.

### 5. Then run the signal_completion smoke

```bash
cd "$WS"
<repo>/.venv/bin/python smoke_signal_completion.py
```

Expected: a one-sentence acknowledgement followed by a
`signal_completion` tool call with `{summary, status, word_count}`,
exit 0.

### 6. Then run the tools smoke

```bash
cd "$WS"
<repo>/.venv/bin/python smoke_tools.py
```

Expected: a `cli_based_tool` call running `ls /tmp`, followed by a
one-sentence summary, followed by a `signal_completion` call with the
same 3-field payload, exit 0. The full conversation including the tool
call streams to stdout via the SDK's output events.

If the model loops or refuses to call the tool, that's a **model
fidelity** issue, not a provider bug — try a stronger tool-calling
model like `openai/gpt-4o` or `anthropic/claude-3.5-sonnet`.

### Other knobs

- **Different model** — edit `model` in the profile to any catalog id
  (e.g. `anthropic/claude-3.5-sonnet`, `meta/Llama-3.3-70B-Instruct`,
  `mistral/Mistral-large-2411`).
- **`JAATO_GITHUB_ORGANIZATION`** — set in workspace `.env` to attribute
  billing to an org. The provider switches to the org-scoped endpoint.
- **`JAATO_GITHUB_ENDPOINT`** — override the API endpoint URL (very
  rare; defaults to `https://models.github.ai/inference`).

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `TokenNotFoundError` | No credential resolved. Run `github-auth status` to see what the provider found; either run `github-auth login` or set `GITHUB_TOKEN` in the workspace `.env`. |
| 1 | `TokenInvalidError` | Token exists but was rejected. Test it: `curl -H "Authorization: Bearer $GITHUB_TOKEN" https://models.github.ai/catalog/models`. If 401, regenerate the PAT (and re-authorize SSO for classic PATs). |
| 1 | `TokenPermissionError` | Token is valid but lacks `models: read`. Recreate as a fine-grained PAT with that permission. |
| 1 | `ModelsDisabledError` | GitHub Models is disabled at the org/enterprise level. An admin must enable it — see the error message for the exact settings URL. |
| 1 | `ModelNotFoundError` | `model` in the profile doesn't match any catalog id. Check `GET https://models.github.ai/catalog/models` and edit the profile. |
| 1 | `RateLimitError` | Too many requests. Wait or upgrade the Copilot plan / enable paid usage. |
| 1 | `InfrastructureError` (5xx) | Transient GitHub-side. The runtime retries automatically; if it persists, check https://www.githubstatus.com/. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | Cold-start latency or the model is hung. Bump `TURN_TIMEOUT_SECONDS` in the harness (the tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool` — usually a fidelity issue with weaker models. Try `openai/gpt-4o` or `anthropic/claude-3.5-sonnet`. |
| 1 with `NudgeExhausted: Agent loop exhausted N completion nudges` | The model responded with text but didn't call `signal_completion` to end the turn. **The wire worked** — the smoke validates provider connectivity, and a coherent text reply proves the wire end-to-end. NudgeExhausted on a weak tool-caller (smaller github_models entries, etc.) is a **model-fidelity result, not a smoke failure**. Capable models (Claude Sonnet 4.5, GPT-4o) follow the persona's `signal_completion` instruction cleanly. |
