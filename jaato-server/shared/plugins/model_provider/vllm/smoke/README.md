# vllm provider smoke harness

Three end-to-end smokes for the `vllm` provider, layered from simplest to most demanding:

| # | Smoke | What it validates | Profile | Harness |
|---|---|---|---|---|
| 1 | **Chat** | Provider wire — daemon can reach the remote endpoint and round-trip `/v1/chat/completions`. **No tools, no `signal_completion`**: the profile declares no `completion_payload_schema` so `signal_completion` is hidden (2026-06-07 schema gate). The turn ends naturally when the model emits text without function calls. | `vllm-chat` | `smoke_chat.py` |
| 2 | **signal_completion** | Lifecycle — schema-driven completion contract. Profile declares a non-trivial 3-field schema (summary + status + word_count, 2 required). Model must acknowledge in text then call `signal_completion` with a schema-valid payload. Tests both the OpenAI tools wire AND `jsonschema.validate` end-to-end. | `vllm-signal_completion` | `smoke_signal_completion.py` |
| 3 | **Tools** | Full tool-calling — `cli` plugin (one shell tool call) followed by `signal_completion` (same schema as #2). Exercises the OpenAI tools shape with multiple tool surfaces. Requires `--enable-auto-tool-choice --tool-call-parser <name>` at server launch. | `vllm-tools` | `smoke_tools.py` |

Run them in order. If **chat** is red, the wire is broken — fix that first.
If **signal_completion** is red but chat is green, the issue is either the
tool-calling parser at the vLLM server (missing `--enable-auto-tool-choice`)
or a model-fidelity gap on structured output. If **tools** is red but the
first two are green, the issue is the `cli` tool's schema or chained
tool-result handling.

Pick a model that's strong at tool-calling for tests 2-3 (Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, or larger). Small / heavily-quantized models
(< 7B, INT4 AWQ) may pass test 1 but fail tests 2-3 from model-fidelity
limits, not framework bugs.

These are **not** unit tests — they require a live daemon and a live
remote endpoint. Unit tests for the provider live in `../tests/`.

## Configuration model

The smoke separates **profile knobs** (model, plugins, GC) from
**deployment knobs** (host):

| Knob | Lives in | Why |
|---|---|---|
| `model` | profile YAML (literal) | Model choice IS the profile choice. To target a different model, edit the profile or copy it to a new file. |
| `plugins`, `plugin_configs.*` | profile YAML | Pure profile concern. |
| `host` | workspace `.env` as `VLLM_HOST`, referenced from profile as `${VLLM_HOST}` | Endpoint detail varies per deployment, not per profile. Resolved at profile-load time via the framework's `${VAR}` substitution chain (`shared/plugins/subagent/config.py:_expand_string`). |

The profiles ship with `Qwen/Qwen2.5-7B-Instruct` baked in. If you serve
a different model on your endpoint, edit the profile.

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
    │   ├── vllm-chat.yaml               # no schema → signal_completion hidden
    │   ├── vllm-signal_completion.yaml  # 3-field schema, plugins:[]
    │   └── vllm-tools.yaml              # 3-field schema, plugins:[cli]
    └── agents/
        ├── vllm-chat.md                 # text-only responder
        ├── vllm-signal_completion.md    # acknowledge + payload
        └── vllm-tools.md                # tool call + acknowledge + payload
```

## Prerequisites

- A reachable vLLM OpenAI-compatible server, launched with the model
  you want to use. Minimal command:
  ```bash
  vllm serve Qwen/Qwen2.5-7B-Instruct \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 8192
  ```
  For the tools smoke, also include `--enable-auto-tool-choice` and
  pick a parser appropriate for the model family — see the vLLM
  Tool-Calling docs (https://docs.vllm.ai/en/stable/features/tool_calling)
  for the parser name (`hermes` for Qwen2.5, `mistral` for Mistral
  Instruct, `llama3_json` for Llama 3.1 Instruct, etc.).
- `nvidia-smi` confirmed on the remote host.
- `curl http://<host>:<port>/health` returns 200 from the jaato host
  (firewall, WSL network mode, etc. all working).
- `curl http://<host>:<port>/v1/models` returns the model `id` that
  matches the profile's `model` field (default `Qwen/Qwen2.5-7B-Instruct`).
- Daemon listening on `/tmp/jaato.sock`:
  `jaato-server --ipc-socket /tmp/jaato.sock --daemon`

## Quick path: `bootstrap.sh`

The included bash helper sets up a self-contained smoke install at the
target workspace (default `/tmp/jaato-vllm-smoke`):

```bash
./jaato-server/shared/plugins/model_provider/vllm/smoke/bootstrap.sh
```

After it completes, the workspace looks like:

```
/tmp/jaato-vllm-smoke/
├── smoke.py             ← copied from the repo
├── smoke_tools.py       ← copied from the repo
├── .env                 ← created from .env.example (only if absent)
└── .jaato/
    ├── profiles/        ← templates (Qwen baked in, host → ${VLLM_HOST})
    └── agents/
```

### Configure the deployment

Edit `<workspace>/.env` and set `VLLM_HOST` to your endpoint:

```bash
$EDITOR /tmp/jaato-vllm-smoke/.env
# VLLM_HOST=http://192.168.1.50:8000
```

### Run the smoke

Either run from the workspace:

```bash
cd /tmp/jaato-vllm-smoke
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
WS=/tmp/jaato-vllm-smoke
mkdir -p "$WS/.jaato/profiles" "$WS/.jaato/agents"
```

### 2. Copy the templates

```bash
SMOKE=jaato-server/shared/plugins/model_provider/vllm/smoke
cp -f "$SMOKE/smoke.py" "$SMOKE/smoke_tools.py" "$WS/"
cp -f "$SMOKE/.jaato.example/profiles/"*.yaml "$WS/.jaato/profiles/"
cp -f "$SMOKE/.jaato.example/agents/"*.md "$WS/.jaato/agents/"
cp -f "$SMOKE/.env.example" "$WS/.env"      # only if you don't already have .env
```

### 3. Configure the deployment

Edit `$WS/.env`:

```
VLLM_HOST=http://<your-host>:<your-port>
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
fidelity** issue, not a provider bug — check that vLLM was launched
with `--enable-auto-tool-choice --tool-call-parser <name>` matching
the model family (parser names: `hermes` / `mistral` / `llama3_json` /
`pythonic` / `granite` / `xlam` / ... — see vLLM tool-calling docs),
and try a stronger tool-calling model.

### Other knobs

- **`context_length`** (default 8192) — match the engine's
  `--max-model-len`. vLLM's `/v1/models` does not surface this
  (verified against vLLM stable docs 2026-06-07 via context7), so set
  it explicitly for long-context engines.
- **`plugin_configs.vllm.api_token`** — only if the vLLM server was
  launched with `--api-key <token>` or the endpoint is fronted by an
  auth proxy. A `pass://` URI resolves daemon-side; a literal token
  also works but don't commit it.
- **Different model** — edit `model` in the profile to whatever
  `/v1/models` reports.

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `VLLMConnectionError` | Network / firewall: `curl http://<host>:<port>/health` from the jaato host. Confirm `VLLM_HOST` in workspace `.env` matches. |
| 1 | `VLLMModelNotFoundError` | `model` in the profile doesn't match `/v1/models`. Edit the profile. |
| 1 | `VLLMAuthenticationError` | Token missing or wrong. Test the resolved bearer with `curl -H "Authorization: Bearer <token>" ...` before touching framework code. |
| 1 | `VLLMMidStreamError` | Connection dropped mid-stream after the server already committed HTTP 200. Check the vLLM server log on the host running the engine — the failure is named there (prompt-too-long under `--max-model-len`, KV-cache exhaustion, CUDA OOM, engine-internal exception, ...). The framework cannot see the cause from the wire by definition. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | Engine is loading on the remote host (cold start), or the model is hung. Check the `vllm serve` logs on the remote. Bump `TURN_TIMEOUT_SECONDS` in the harness if cold-start exceeds 120s (the tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool` — usually means either the model isn't strong at tool-calling, or the server was launched without `--enable-auto-tool-choice --tool-call-parser <name>` matching the model family. Confirm the launch flags first. |
| 1 with `NudgeExhausted: Agent loop exhausted N completion nudges` | The model responded with text but didn't call `signal_completion` to end the turn. **The wire worked** — the smoke validates provider connectivity, and a coherent text reply proves the wire end-to-end. NudgeExhausted on a weak tool-caller (small models on vLLM, etc.) is a **model-fidelity result, not a smoke failure**. Capable models (Claude Sonnet 4.5, GPT-4o) follow the persona's `signal_completion` instruction cleanly. |

## Sub-decisions baked into this plugin (flag for human review)

These are calls that weren't pinned by the tensorrt_llm template;
listed here so a reviewer can challenge them.

| Call | Made | Alternative considered |
|---|---|---|
| Guided decoding (`guided_json`/`guided_choice`/`guided_regex`/`guided_grammar`) NOT exposed as profile knobs | The provider only surfaces `response_format={"type":"json_object"}` via the existing `response_schema=` arg, mirroring trtllm/lmstudio. | Could add `plugin_configs.vllm.api_params.guided_json` etc. as opaque pass-through into `extra_body`. Skipped because the symmetric trtllm/lmstudio providers don't expose it either, and adding cross-provider asymmetry costs more in cognitive load than it earns in capability for cascade workloads. Operators wanting strict guided decoding compose `extra_body` at the harness layer. |
| `verify_auth()` is network-free | Matches the canonical `AnthropicProvider.verify_auth` + `TensorRTLLMProvider.verify_auth` shape — credentials availability only, no live `/v1/models` probe. | Could live-probe `/v1/models` for early failure surfacing. Skipped because the CLAUDE.md contract is explicit that `verify_auth` runs on a fresh, uninitialized instance and must NOT touch the network; the live probe lives in `initialize()` (`/health`) and `connect()` (`/v1/models`). |
| Tool-call-parser is operator-launch concern, NOT a profile knob | The parser flags (`--enable-auto-tool-choice --tool-call-parser <name>`) shape vLLM's PARSING of model output — they belong at server-launch boundary, not in the client profile. Documented in `README.md` + `__init__.py` docstring. | Could add `plugin_configs.vllm.tool_call_parser` as informational metadata. Skipped — the framework cannot enforce a server-launch flag from the client, and surfacing the knob in the profile invites the impression that setting it has an effect. |
| Reasoning channel (`message.reasoning` from `--reasoning-parser`) NOT extracted | `supports_thinking()` returns False so the framework does not silently drop reasoning content into the main text path. | Could extract `message.reasoning` when present and route it to `on_thinking`. Skipped for the initial PR — none of the trtllm/triton/lmstudio siblings extract reasoning either, and the symmetry matters more than the capability gain for the cascade workloads that motivated this provider. Easy follow-up if a user with a DeepSeek-R1 vLLM deployment files a request. |
