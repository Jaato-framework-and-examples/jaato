# tensorrt_llm provider smoke harness

Three end-to-end smokes for the `tensorrt_llm` provider, layered from simplest to most demanding:

| # | Smoke | What it validates | Profile | Harness |
|---|---|---|---|---|
| 1 | **Chat** | Provider wire — daemon can reach the remote `trtllm-serve` (or Triton OpenAI frontend) and round-trip `/v1/chat/completions`. **No tools, no `signal_completion`**: the profile declares no `completion_payload_schema` so `signal_completion` is hidden (2026-06-07 schema gate). The turn ends naturally when the model emits text without function calls. | `tensorrt-llm-chat` | `smoke_chat.py` |
| 2 | **signal_completion** | Lifecycle — schema-driven completion contract. Profile declares a non-trivial 3-field schema (summary + status + word_count, 2 required). Model must acknowledge in text then call `signal_completion` with a schema-valid payload. Tests both the OpenAI tools wire AND `jsonschema.validate` end-to-end. | `tensorrt-llm-signal_completion` | `smoke_signal_completion.py` |
| 3 | **Tools** | Full tool-calling — `cli` plugin (one shell tool call) followed by `signal_completion` (same schema as #2). Exercises the OpenAI tools shape with multiple tool surfaces. | `tensorrt-llm-tools` | `smoke_tools.py` |

Run them in order. If **chat** is red, the wire is broken — fix that first.
If **signal_completion** is red but chat is green, the issue is either
the trtllm engine build (engines built without tool-calling parsing
cannot satisfy this smoke) or a model-fidelity gap on structured output.
If **tools** is red but the first two are green, the issue is the `cli`
tool's schema or chained tool-result handling.

Pick a model that's strong at tool-calling for tests 2-3
(Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, or larger). Small / weak
tool-callers (Mistral-7B-v0.3, base completions models) may pass test 1
but fail tests 2-3 from model-fidelity limits, not framework bugs.

These are **not** unit tests — they require a live daemon and a live
remote endpoint. Unit tests for the provider live in `../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (host):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile YAML (literal) | Model choice IS the profile choice. To target a different model, edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile YAML | Pure profile concern. |
| `host` | workspace `.env` as `TENSORRT_LLM_HOST`, referenced from profile as `${TENSORRT_LLM_HOST}` | Endpoint detail varies per deployment, not per profile. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`). |

The profiles ship with `Qwen/Qwen2.5-7B-Instruct` baked in. If you serve
a different model on your endpoint, edit the profile.

## What's in here

```
smoke/
├── README.md                                  # this file
├── bootstrap.sh                               # one-shot workspace install
├── smoke_chat.py                              # #1 — pure text round-trip
├── smoke_signal_completion.py                 # #2 — lifecycle, schema-driven
├── smoke_tools.py                             # #3 — cli + signal_completion
├── .env.example                               # workspace env template
└── .jaato.example/                            # workspace .jaato/ template
    ├── profiles/
    │   ├── tensorrt-llm-chat.yaml             # no schema → signal_completion hidden
    │   ├── tensorrt-llm-signal_completion.yaml # 3-field schema, plugins:[]
    │   └── tensorrt-llm-tools.yaml            # 3-field schema, plugins:[cli]
    └── agents/
        ├── tensorrt-llm-chat.md               # text-only responder
        ├── tensorrt-llm-signal_completion.md  # acknowledge + payload
        └── tensorrt-llm-tools.md              # tool call + acknowledge + payload
```

## Prerequisites

- A reachable `trtllm-serve` instance (or Triton with the OpenAI
  frontend). See the parent provider's docstring (`../provider.py`)
  and the project `CLAUDE.md` for env vars.
- `nvidia-smi` confirmed on the remote host.
- `curl http://<host>:<port>/health` returns 200 from the jaato host
  (firewall, WSL network mode, etc. all working).
- `curl http://<host>:<port>/v1/models` returns the model `id` that
  matches the profile's `model` field (default `Qwen/Qwen2.5-7B-Instruct`).
- Daemon listening on `/tmp/jaato.sock`:
  `jaato-server --ipc-socket /tmp/jaato.sock --daemon`

## Quick path: `bootstrap.sh`

The included bash helper sets up a self-contained smoke install at the
target workspace (default `/tmp/jaato-tensorrt-smoke`):

```bash
./jaato-server/shared/plugins/model_provider/tensorrt_llm/smoke/bootstrap.sh
```

After it completes, the workspace looks like:

```
/tmp/jaato-tensorrt-smoke/
├── smoke_chat.py                ← copied from the repo
├── smoke_signal_completion.py   ← copied from the repo
├── smoke_tools.py               ← copied from the repo
├── .env                         ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/                ← templates (Qwen baked in, host → ${TENSORRT_LLM_HOST})
    └── agents/
```

### Configure the deployment

Edit `<workspace>/.env` and set `TENSORRT_LLM_HOST` to your endpoint:

```bash
$EDITOR /tmp/jaato-tensorrt-smoke/.env
# TENSORRT_LLM_HOST=http://192.168.1.50:8000
```

### Run the smokes

Either run from the workspace:

```bash
cd /tmp/jaato-tensorrt-smoke
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
WS=/tmp/jaato-tensorrt-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/tensorrt_llm/smoke
cp -f "$SMOKE/smoke_chat.py" "$SMOKE/smoke_signal_completion.py" \
    "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.yaml "$WS/.jaato/profiles/"
cp -f "$SMOKE/.jaato.example/agents/"*.md "$WS/.jaato/agents/"
cp -f "$SMOKE/.env.example" "$WS/.env"      # only if you don't already have .env
```

### 3. Configure the deployment

Edit `$WS/.env`:

```
TENSORRT_LLM_HOST=http://<your-host>:<your-port>
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
model (Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct work well;
Mistral-7B-v0.3 is weak on tool calls).

### Other knobs

- **`context_length`** (default 8192) — match the engine's `max_seq_len`.
  `trtllm-serve`'s `/v1/models` does not surface this, so set it
  explicitly for long-context engines.
- **`plugin_configs.tensorrt_llm.api_token`** — only if the endpoint
  is fronted by an auth proxy. A `pass://` URI resolves daemon-side; a
  literal token also works but don't commit it.
- **Different model** — edit `model` in the profile to whatever
  `/v1/models` reports.

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `TensorRTLLMConnectionError` | Network / firewall: `curl http://<host>:<port>/health` from the jaato host. Confirm `TENSORRT_LLM_HOST` in workspace `.env` matches. |
| 1 | `TensorRTLLMModelNotFoundError` | `model` in the profile doesn't match `/v1/models`. Edit the profile. |
| 1 | `TensorRTLLMAuthenticationError` | Token missing or wrong. Test the resolved bearer with `curl -H "Authorization: Bearer <token>" ...` before touching framework code. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | Engine is loading on the remote host (cold start), or the model is hung. Check `trtllm-serve` logs on the remote. Bump `TURN_TIMEOUT_SECONDS` in the harness if cold-start exceeds 120s (the tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool` — usually a fidelity issue with smaller models, OR a trtllm engine built without tool-calling parsing. Try Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct first, then check the engine build. |
| 1 with `NudgeExhausted: Agent loop exhausted N completion nudges` | The model responded with text but didn't call `signal_completion` to end the turn. **The wire worked** — the smoke validates provider connectivity, and a coherent text reply proves the wire end-to-end. NudgeExhausted on a weak tool-caller (Qwen-7B on trtllm-serve, small ollama models, etc.) is a **model-fidelity result, not a smoke failure**. Capable models (Claude Sonnet 4.5, GPT-4o) follow the persona's `signal_completion` instruction cleanly. The persona pattern (instruct one sentence + signal_completion call) is the canonical shape; smaller models may need richer pattern (front-load imperative, forbid alternatives) but example payloads can backfire — weak models echo the example as natural text. |
