# tensorrt_llm provider smoke harness

Two end-to-end smokes for the `tensorrt_llm` provider:

| Smoke | What it validates | Profile | Harness |
|---|---|---|---|
| **Chat** | Provider wire — daemon can reach the remote endpoint and round-trip `/v1/chat/completions`. No tools involved. | `tensorrt-llm-smoke` | `smoke.py` |
| **Tools** | OpenAI tools shape — schema serialization, tool-call argument parsing, tool-result round-trip. Exercises the `cli` plugin. | `tensorrt-llm-tools` | `smoke_tools.py` |

Run the **chat** smoke first. If it's red, the wire is broken and tool-shape
results would be meaningless. Once chat is green, the tools smoke tells you
whether the OpenAI tools path is intact (and gives the model a fair chance
to demonstrate tool-calling fidelity — pick a model that's good at it, e.g.
Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct).

These are **not** unit tests — they require a live daemon and a live
remote endpoint. Unit tests for the provider live in `../tests/`.

## What's in here

```
smoke/
├── README.md                              # this file
├── smoke.py                               # chat-only harness
├── smoke_tools.py                         # tool-calling harness
└── .jaato.example/                        # workspace artifact templates
    ├── profiles/
    │   ├── tensorrt-llm-smoke.json        # pure chat, no tools, no GC
    │   └── tensorrt-llm-tools.json        # cli + permission, default-allow
    └── agents/
        ├── tensorrt-llm-smoke.md          # one-sentence-responder persona
        └── tensorrt-llm-tools.md          # tool-using-then-summarize persona
```

The `.jaato.example/` tree mirrors the structure of a real workspace's
`.jaato/` dir. Templates are copied into the workspace before running —
see step 2 below.

## Prerequisites

- A reachable `trtllm-serve` instance (or Triton with the OpenAI
  frontend). See the parent provider's docstring (`../provider.py`)
  and the project `CLAUDE.md` for env vars.
- `nvidia-smi` confirmed on the remote host.
- `curl http://REMOTE_HOST:PORT/health` returns 200 from the jaato
  host (firewall, WSL network mode, etc. all working).
- `curl http://REMOTE_HOST:PORT/v1/models` returns the model `id` you
  intend to test.

## Run it

### 1. Copy the templates into your workspace

From the workspace root:

```bash
mkdir -p .jaato/profiles .jaato/agents
cp jaato-server/shared/plugins/model_provider/tensorrt_llm/smoke/.jaato.example/profiles/*.json .jaato/profiles/
cp jaato-server/shared/plugins/model_provider/tensorrt_llm/smoke/.jaato.example/agents/*.md .jaato/agents/
```

### 2. Fill in the placeholders in **both** profiles

Edit `.jaato/profiles/tensorrt-llm-smoke.json` and
`.jaato/profiles/tensorrt-llm-tools.json`:

| Placeholder | Replace with |
|---|---|
| `REPLACE_WITH_MODEL_ID_FROM_v1_models` | The exact `id` field returned by `GET /v1/models` on the remote endpoint. |
| `http://REMOTE_HOST:8000` | The remote host's LAN address + port. |

Optional knobs (apply to both profiles):

- `context_length` (default 8192) — match the engine's `max_seq_len`.
  `trtllm-serve`'s `/v1/models` does not surface this, so you must
  set it explicitly for long-context engines.
- `plugin_configs.tensorrt_llm.api_token` — only if the endpoint is
  fronted by an auth proxy. A `pass://` URI resolves daemon-side; a
  literal token also works but don't commit it.

### 3. Start the daemon

```bash
.venv/bin/jaato-server --ipc-socket /tmp/jaato.sock --daemon
.venv/bin/jaato-server --status
```

### 4. Run the chat smoke first

```bash
.venv/bin/python jaato-server/shared/plugins/model_provider/tensorrt_llm/smoke/smoke.py
```

Expected: one sentence of model output, exit 0.

### 5. Then run the tools smoke

```bash
.venv/bin/python jaato-server/shared/plugins/model_provider/tensorrt_llm/smoke/smoke_tools.py
```

Expected: a `cli_based_tool` call running `ls /tmp`, followed by a one-sentence
summary, exit 0. The full conversation including the tool call streams to stdout
via the SDK's output events.

If the model loops or refuses to call the tool, that's a **model fidelity**
issue, not a provider bug — try a stronger tool-calling model (Qwen2.5-7B-Instruct
and Llama-3.1-8B-Instruct work well; Mistral-7B-v0.3 is weak on tool calls).

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `TensorRTLLMConnectionError` | Network / firewall: `curl http://REMOTE_HOST:PORT/health` from the jaato host. |
| 1 | `TensorRTLLMModelNotFoundError` | `model` in the profile doesn't match `/v1/models`. Copy the exact `id`. |
| 1 | `TensorRTLLMAuthenticationError` | Token missing or wrong. Test the resolved bearer with `curl -H "Authorization: Bearer <token>" ...` before touching framework code. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | Engine is loading on the remote host (cold start), or the model is hung. Check `trtllm-serve` logs on the remote. Bump `TURN_TIMEOUT_SECONDS` in the harness if cold-start exceeds 120s (the tools smoke uses 180s by default since tool round-trips take longer). |
| 0 but no tool call (`smoke_tools.py`) | The model answered without calling `cli_based_tool` — usually a fidelity issue with smaller models. Try Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct. |
