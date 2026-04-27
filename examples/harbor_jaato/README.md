# harbor-jaato

A [Harbor](https://www.harborframework.com) `BaseAgent` that drives a
[jaato-server](https://github.com/Jaato-framework-and-examples/jaato)
running inside the evaluation environment.

## How it works

Both the daemon and the SDK live **inside the Harbor container**.
The host-side `BaseAgent` is a thin shim that:

1. `setup()` — installs `jaato-server` + `jaato-sdk` into a `/opt/jaato`
   venv, uploads this package's source to `/opt/harbor_jaato`, and
   writes `.jaato/profiles/harbor.json` pinning the model, provider,
   and plugin set.
2. `run()` — uploads the instruction text and execs
   `python -m harbor_jaato.harness` in the container. The harness opens
   an `IPCRecoveryClient` against `/tmp/jaato.sock` (the SDK
   auto-starts the daemon), creates a session against the `harbor`
   profile, sends the instruction, and drains events until a terminal
   status. After every `TurnCompletedEvent` it writes
   `result.json` atomically. The host downloads that file and copies
   token counts and the trajectory onto Harbor's `AgentContext`.

No WebSocket, no port forwarding, no bearer-token plumbing. The IPC
socket and the harness share a process tree.

## Usage

```bash
pip install -e examples/harbor_jaato
harbor run \
    --dataset terminal-bench@2.0 \
    --agent jaato \
    --model anthropic/claude-opus-4-7 \
    --n-concurrent 4
```

The agent is registered with Harbor through the `harbor.agents` entry
point in `pyproject.toml`, so `--agent jaato` is enough.

## Provider credentials

The host forwards a small set of provider env vars into the container
via `environment.exec(env=...)`. See `PROVIDER_ENV_KEYS` in
`agent.py` — extend that table for any provider you need.

## Permissions

The harness auto-responds `"a"` (always) to the first
`PermissionRequestedEvent` per tool, promoting it to the session
whitelist. The sandbox is the boundary; per-call approvals don't add
safety inside a Harbor eval container.

## Trajectory

`AgentContext.metadata["trajectory"]` carries an event-by-event log
(model output, tool calls, tool results). Set `SUPPORTS_ATIF = True`
on `JaatoAgent` only after mapping that log to Harbor's
[ATIF](https://www.harborframework.com/docs/agents/trajectory-format)
format.

## Layout

```
examples/harbor_jaato/
├── pyproject.toml
├── README.md
└── src/harbor_jaato/
    ├── __init__.py
    ├── agent.py     # BaseAgent (host-side)
    ├── harness.py   # python -m harbor_jaato.harness (in-container)
    └── result.py    # shared dataclass written/read across the boundary
```
