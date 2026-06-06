# tensorrt_llm provider smoke harness

A minimal end-to-end smoke for the `tensorrt_llm` provider. Validates the
daemon can reach a remote `trtllm-serve` (or Triton OpenAI frontend)
instance and round-trip a chat completion over the provider wire.

This is **not** a unit test — it requires a live daemon and a live
remote endpoint. Unit tests for the provider live in `../tests/`.

## What's in here

```
smoke/
├── README.md                       # this file
├── smoke.py                        # the harness
└── .jaato.example/                 # workspace artifact templates
    ├── profiles/
    │   └── tensorrt-llm-smoke.json # pure chat profile (no tools, no GC)
    └── agents/
        └── tensorrt-llm-smoke.md   # one-sentence-responder persona
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

### 2. Fill in the placeholders in the profile

Edit `.jaato/profiles/tensorrt-llm-smoke.json`:

| Placeholder | Replace with |
|---|---|
| `REPLACE_WITH_MODEL_ID_FROM_v1_models` | The exact `id` field returned by `GET /v1/models` on the remote endpoint. |
| `http://REMOTE_HOST:8000` | The remote host's LAN address + port. |

Optional knobs:

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

### 4. Run the harness

```bash
.venv/bin/python jaato-server/shared/plugins/model_provider/tensorrt_llm/smoke/smoke.py
```

Expected: one sentence of model output, exit 0.

## Failure-class triage

| Exit | Symptom | First thing to check |
|---|---|---|
| 1 | `TensorRTLLMConnectionError` | Network / firewall: `curl http://REMOTE_HOST:PORT/health` from the jaato host. |
| 1 | `TensorRTLLMModelNotFoundError` | `model` in the profile doesn't match `/v1/models`. Copy the exact `id`. |
| 1 | `TensorRTLLMAuthenticationError` | Token missing or wrong. Test the resolved bearer with `curl -H "Authorization: Bearer <token>" ...` before touching framework code. |
| 2 | "connect failed" | Daemon isn't listening on `/tmp/jaato.sock` — re-run `jaato-server --status`. |
| 3 | Timeout, no output | Engine is loading on the remote host (cold start), or the model is hung. Check `trtllm-serve` logs on the remote. Bump `TURN_TIMEOUT_SECONDS` in the harness if cold-start exceeds 120s. |

## Next step

Once chat is green, the natural follow-up smoke is **tool-calling**: copy
the profile to `tensorrt-llm-tools.json`, add `plugins: ["cli", "permission"]`
and `max_turns: 4`, and ask the model to `ls /tmp`. That exercises the
OpenAI tools shape against TRT-LLM's parser — which is where the next
class of bugs usually surfaces. Note: tool-calling fidelity depends
heavily on the model — Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct
work well; Mistral-7B-v0.3 is weak on tool calls.
