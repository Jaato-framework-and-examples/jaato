# jaato-server

The runtime and daemon behind [jaato](https://github.com/Jaato-framework-and-examples/jaato) — a multi-provider agentic tool orchestrator. This package holds the session runtime, the plugin system, ~24 model-provider adapters, and the daemon that clients connect to over a Unix socket or WebSocket.

You install this on the machine where agents should *run*. To talk to it from Python, install [`jaato-sdk`](https://pypi.org/project/jaato-sdk/); for a terminal client, [`jaato-tui`](https://pypi.org/project/jaato-tui/).

## Installation

```bash
pip install jaato-server
# or:  uv pip install jaato-server
```

The base install carries the daemon, the session runtime and the plugins that need no third-party SDK. Provider SDKs and the heavier tool plugins are extras, so you install only the ones you use:

| Extra | Brings |
|-------|--------|
| `google` | Google GenAI / Vertex AI |
| `openai` | OpenAI (Chat Completions + the Responses API) |
| `azure-openai` | Azure OpenAI, including Microsoft Entra ID auth |
| `bedrock` | AWS Bedrock (`boto3`) |
| `openrouter`, `nim`, `ovhcloud` | OpenAI-compatible gateways |
| `github-models` | GitHub Models (`azure-ai-inference`) |
| `web` | `web_fetch` / `web_search` — extraction, PDF and search backends |
| `interactive` | `interactive_shell` — PTY sessions (`pexpect` / `wexpect`) |
| `ast` | `ast_search` — structural code search |
| `notebook`, `diagrams`, `templates`, `kaggle` | the matching tool plugins |
| `telemetry` | OpenTelemetry tracing |
| `kerberos` | SPNEGO/Negotiate proxy auth |
| `all` | everything above, plus the dev tooling |

```bash
pip install 'jaato-server[all]'
```

Anthropic's SDK, `mcp`, `websockets` and `jaato-sdk` are base dependencies — Claude, MCP servers and the WebSocket transport work out of the box. Requires Python 3.12+.

## Quick start

Start the daemon:

```bash
# Local clients over a Unix domain socket
jaato-server --ipc-socket /tmp/jaato.sock --daemon

# Add a WebSocket port for remote or browser clients.  A bearer token is
# generated at ~/.jaato/ws.token on first WS start if you supply none.
jaato-server --ipc-socket /tmp/jaato.sock --web-socket :8080 --daemon

jaato-server --status
jaato-server --stop
```

Then drive it from Python:

```python
import asyncio
import jaato          # from jaato-sdk

async def main():
    async with jaato.session(mode="ipc", profile="researcher") as s:
        print(await s.ask("What changed in this repo last week?"))

asyncio.run(main())
```

The same code runs against an embedded runtime (`mode="in_process"`, no daemon), a local daemon (`"ipc"`), or a remote one (`"ws"`).

## Model providers

One session interface over every backend below; a profile picks the provider and model, and `enter_tier` can switch bindings mid-session.

| | |
|---|---|
| **Hosted APIs** | Anthropic, OpenAI, Azure OpenAI, Google GenAI / Vertex AI, AWS Bedrock, GitHub Models, Zhipu AI, MiniMax, Moonshot Kimi, Xiaomi MiMo |
| **Gateways** | OpenRouter (300+ models), NVIDIA NIM, Nebius Token Factory, OVHcloud AI Endpoints, Doubleword |
| **Self-hosted** | Ollama, LM Studio, vLLM, NVIDIA TensorRT-LLM, Triton |
| **On-device** | Chrome built-in AI (Gemini Nano, over the DevTools Protocol) |
| **Subscription-backed** | Claude Code CLI, Google Antigravity |

Several ship an interactive auth command (`anthropic-auth login`, `github-auth login`, `openrouter-auth key …`) so a credential need never be pasted into a shell profile.

## Plugins

Tools reach the model through plugins, discovered from this package and from any installed distribution declaring a `jaato.plugins` entry point. Built-ins include:

- **Execution** — `cli` (shell), `interactive_shell` (PTY sessions for REPLs, debuggers, wizards), `notebook`, `mcp` (any stdio MCP server)
- **Code and files** — `file_edit`, `filesystem_query`, `ast_search`, `lsp`, `references`
- **Web** — `web_fetch`, `web_search`, `webmcp`, `service_connector` (call an OpenAPI service by alias), `webhook` (inbound events)
- **Orchestration** — `subagent`, `todo`, `clarification`, `waypoint`, `reliability`, `template`
- **Context management** — four GC strategies (`gc_truncate`, `gc_summarize`, `gc_hybrid`, `gc_budget`) plus provider prompt-caching plugins
- **Control** — `permission`, `sandbox_manager`, `telemetry`

## Agent profiles

A profile is a YAML file in `.jaato/profiles/` binding a model, a provider, a plugin set, a GC strategy and resource ceilings:

```yaml
name: researcher
description: Deep research profile
provider: anthropic
model: claude-sonnet-4-20250514
plugins: [cli, web_search, memory, todo(preload)]
default_agent: researcher        # the persona in .jaato/agents/researcher.md
runtime_limits:
  max_parallel_tools: 2
  max_orphan_seconds: 300
budget_control:
  limits: {usd: 5.0, tool_calls: 200}
  degrade:
    - {at: 95,  action: finalize}
    - {at: 100, action: abort}
```

`jaato-scaffold` interrogates the *installed* framework rather than a copy of the docs, so it cannot go stale:

```bash
jaato-scaffold explain profile         # every profile key, with its vocabulary
jaato-scaffold explain providers       # providers, their knobs and caveats
jaato-scaffold explain plugins         # what is installed, and where it came from
jaato-scaffold validate                # check a workspace before you run it
jaato-scaffold new client --profile researcher
```

`explain` covers 20-odd scopes — `env`, `tiers`, `completion`, `runtime`, `paths`, `services` among them.

## Isolation and secrets

Sessions run in a subprocess runner, and the daemon confines it where the host allows:

- **AppArmor** — a per-boundary profile confines the runner to its workspace when the LSM is available. Opt in per client, or require it with `JAATO_REQUIRE_APPARMOR=1` so an unconfined host refuses to start rather than degrading silently.
- **cgroup v2** — `runtime_limits` caps memory, pids and CPU; tool timeouts, output size, tool-pool width and two wall-clock bounds are enforced by the framework itself.
- **Secret scrubbing** — provider keys and tokens are stripped from the environment handed to every model-driven subprocess by default; `scrub_secret_env` narrows or waives it, and waiving is announced.
- **Permissions** — every tool call passes a policy that can allow, deny or ask, per session, with the decision recorded.

## Configuration

Workspace configuration lives under `.jaato/` (profiles, agents, GC settings, permissions) with `~/.jaato/` as the fallback tier. MCP servers are declared in `.mcp.json`. Credentials resolve from a profile knob, then the environment, then a stored auth file — `pass://` and `vault://` URIs are resolved daemon-side so a secret never sits on disk in the clear.

`jaato-scaffold explain env` prints every environment variable the installed tree reads, each tagged with its scope and the typed profile key that supersedes it.

## Documentation

- [Project README](https://github.com/Jaato-framework-and-examples/jaato/blob/main/README.md)
- [Architecture](https://github.com/Jaato-framework-and-examples/jaato/blob/main/docs/architecture.md) · [Sequence diagrams](https://github.com/Jaato-framework-and-examples/jaato/blob/main/docs/sequence-diagram-architecture.md)
- [Design philosophy](https://github.com/Jaato-framework-and-examples/jaato/blob/main/docs/design-philosophy.md)
- [Permission system](https://github.com/Jaato-framework-and-examples/jaato/blob/main/docs/jaato_permission_system.md)
- [AppArmor setup](https://github.com/Jaato-framework-and-examples/jaato/blob/main/docs/apparmor-setup.md)
- [OpenTelemetry](https://github.com/Jaato-framework-and-examples/jaato/blob/main/docs/opentelemetry-design.md)
- [Issues](https://github.com/Jaato-framework-and-examples/jaato/issues)

## License

BUSL-1.1. See [LICENSE](https://github.com/Jaato-framework-and-examples/jaato/blob/main/LICENSE).
