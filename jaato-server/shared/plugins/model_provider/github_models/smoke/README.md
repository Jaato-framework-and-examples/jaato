# github_models provider smoke harness

Two end-to-end smokes for the `github_models` provider:

| Smoke | What it validates | Profile | Harness |
|---|---|---|---|
| **Chat** | Provider wire — daemon can reach the managed GitHub Models endpoint and round-trip `/chat/completions`. No tools involved. | `github-models-smoke` | `smoke.py` |
| **Tools** | OpenAI tools shape — schema serialization, tool-call argument parsing, tool-result round-trip. Exercises the `cli` plugin. The `permission` plugin is server-wired automatically (no need to list it in `plugins`), but its policy is set via `plugin_configs.permission`. | `github-models-tools` | `smoke_tools.py` |

Run the **chat** smoke first. If it's red, the wire is broken and tool-shape
results would be meaningless. Once chat is green, the tools smoke tells you
whether the OpenAI tools path is intact (and gives the model a fair chance
to demonstrate tool-calling fidelity — `openai/gpt-4o` and
`anthropic/claude-3.5-sonnet` are both strong here).

These are **not** unit tests — they require a live daemon and live
GitHub credentials. Unit tests for the provider live in `../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (`GITHUB_TOKEN`):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile JSON (literal) | Model choice IS the profile choice. To target a different model (e.g. `anthropic/claude-3.5-sonnet`, `meta/Llama-3.3-70B-Instruct`), edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile JSON | Pure profile concern. |
| `GITHUB_TOKEN` | workspace `.env`, referenced from profile as `${GITHUB_TOKEN}` in `plugin_configs.github_models.api_key` | Credential varies per deployment, not per profile. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`). |

The profiles ship with `openai/gpt-4o` baked in. To target a different
model from the GitHub Models catalog, edit the `model` field.

GitHub Models is a **managed cloud endpoint** — there is no host knob.
The token is the only deployment-time variable.

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
    │   ├── github-models-smoke.json       # pure chat, no tools, no GC
    │   └── github-models-tools.json       # cli plugin, default-allow permission
    └── agents/
        ├── github-models-smoke.md         # one-sentence-responder persona
        └── github-models-tools.md         # tool-using-then-summarize persona
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
├── smoke.py             ← copied from the repo
├── smoke_tools.py       ← copied from the repo
├── .env                 ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/        ← templates (gpt-4o baked in, token → ${GITHUB_TOKEN})
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
WS=/tmp/jaato-github-models-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/github_models/smoke
cp -f "$SMOKE/smoke.py" "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.json "$WS/.jaato/profiles/"
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
