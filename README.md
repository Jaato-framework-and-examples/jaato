# jaato

<p align="center">
  <img src="docs/jaato-logo.png" alt="jaato logo" width="200"/>
</p>

<p align="center">
  <strong>A production-grade framework for building agentic AI applications</strong>
</p>

<p align="center">
  Multi-provider AI integration &bull; 55+ plugins &bull; Server-first architecture &bull; MCP &amp; CLI tool orchestration
</p>

<p align="center">
  <a href="https://jaato-framework-and-examples.github.io/jaato/web/index.html">Documentation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="https://jaato-framework-and-examples.github.io/jaato/web/api-reference/plugins/index.html">Plugin Reference</a> &bull;
  <a href="docs/architecture.md">Architecture</a>
</p>

## Demo

![jaato Demo](demo.svg)

## Overview

jaato is a framework for building agentic AI applications with LLM function calling, tool orchestration, and an extensible plugin architecture. It runs as a daemon with a typed event protocol, allowing multiple clients (TUI, web, headless) to connect simultaneously via IPC or WebSocket.

**Core capabilities:**

- **8 Model Providers** - Google GenAI/Vertex AI, Anthropic Claude, Claude CLI, GitHub Models, Google Antigravity, Ollama, ZhipuAI, and NVIDIA NIM through a unified provider abstraction
- **55+ Plugins** - File editing, shell execution, interactive PTY sessions, MCP servers, subagent delegation, AST search, LSP diagnostics, memory, web search, and more
- **Server-First Architecture** - Daemon mode with IPC (Unix socket) and WebSocket transports, multi-session orchestration, and disk persistence
- **Parallel Tool Execution** - Concurrent tool calls with thread-safe callbacks (up to 8 tools per turn)
- **Context Management** - Three garbage collection strategies (truncation, summarization, hybrid generational) with proactive threshold-based triggering
- **Subagent Architecture** - Lightweight session spawning with shared runtime resources (provider config, plugin registry, permissions, token ledger)
- **OpenTelemetry Observability** - Structured tracing with span hierarchy (`jaato.turn` > `jaato.tool` > `jaato.permission`)

### Etymology

While "jaato" serves as an acronym (**j**ust **a**nother **a**gentic **t**ool **o**rchestrator), the name carries deeper meaning. In the Himalayan region (Nepal, Sikkim, Darjeeling, and Bhutan), a **jaato** (जाँतो) is a traditional rotary hand-quern used to mill grains. This ancient tool consists of two round stones with a wooden handle (*hato*) used to turn the top stone in a circular motion.

The metaphor is intentional: just as a traditional jaato grinds raw grains into refined flour, this orchestrator processes raw inputs through LLM tools to produce refined outputs.

## Architecture

jaato uses a server-first design where the server is the source of truth and clients are thin presentation layers.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  TUI Client │  │  Web Client │  │   Headless   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │ IPC            │ WebSocket       │ IPC
       └────────┬───────┴─────────┬───────┘
          ┌─────┴─────────────────┴─────┐
          │        jaato Server          │
          │  ┌───────────────────────┐   │
          │  │    Session Manager    │   │
          │  │  ┌─────┐  ┌─────┐    │   │
          │  │  │Ses. 1│  │Ses. 2│   │   │
          │  │  └─────┘  └─────┘    │   │
          │  └───────────────────────┘   │
          │  ┌───────────────────────┐   │
          │  │    Shared Runtime     │   │
          │  │  Providers │ Plugins  │   │
          │  │  Registry  │ Ledger   │   │
          │  └───────────────────────┘   │
          └──────────────────────────────┘
```

**Key design decisions:**
- **Multi-client support** - Multiple UIs connect to the same running server
- **Session persistence** - State survives client disconnections and restarts
- **Resource sharing** - Single runtime for multiple agents with a shared token ledger
- **Pipeline-presentation split** - Server emits structured events; clients choose how to render them

See [Architecture Overview](docs/architecture.md) for detailed diagrams and [Design Philosophy](docs/design-philosophy.md) for rationale.

## Provider Support

jaato abstracts model providers behind a unified interface. Switch providers by changing a configuration value — no code changes required.

| Provider | Models | Authentication |
|----------|--------|----------------|
| **Google GenAI / Vertex AI** | Gemini 2.5 Flash, Gemini 2.5 Pro | Service account JSON or Application Default Credentials |
| **Anthropic Claude** | Claude Opus, Sonnet, Haiku | PKCE OAuth (subscription) or API key |
| **Claude CLI** | Claude via CLI subscription | `claude login` (uses subscription, not API credits) |
| **GitHub Models** | Models via GitHub API | Device code OAuth or Personal Access Token |
| **Google Antigravity** | Gemini 3, Claude (via Google OAuth) | PKCE OAuth flow |
| **Ollama** | Any Ollama-supported model (Qwen, Llama, Mistral, etc.) | Local — no auth required |
| **ZhipuAI** | ZhipuAI models | API key |
| **NVIDIA NIM** | Llama, DeepSeek-R1, Nemotron (hosted + self-hosted) | API key or self-hosted (no auth) |

## Plugin Ecosystem

jaato ships with **55+ built-in plugins** organized by function. Plugins are auto-discovered and auto-wired — no manual registration needed.

### Tool Execution
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/web/assets/images/plugins/plugin-cli.png" width="32"> | **cli** | Execute shell commands with intelligent auto-backgrounding for long-running processes |
| <img src="docs/web/assets/images/plugins/plugin-mcp.png" width="32"> | **mcp** | Connect to Model Context Protocol servers for external tool integrations |
| <img src="docs/web/assets/images/plugins/plugin-background.png" width="32"> | **background** | Orchestrate parallel background tasks across all BackgroundCapable plugins |
| | **interactive_shell** | Drive interactive processes (REPLs, debuggers, SSH, wizards) via persistent PTY sessions |
| | **environment** | Query execution environment (OS, shell, architecture) for platform-appropriate commands |

### File & Code Operations
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/web/assets/images/plugins/plugin-file-edit.png" width="32"> | **file_edit** | File operations with diff-based approval, automatic backups, and undo support |
| | **filesystem_query** | File system search, traversal, and glob-based discovery |
| | **ast_search** | AST-based code search across Python, JavaScript, TypeScript, and more |
| | **lsp** | Language Server Protocol integration for diagnostics and code intelligence |
| | **notebook** | Jupyter notebook cell execution and management |
| <img src="docs/web/assets/images/plugins/plugin-artifact-tracker.png" width="32"> | **artifact_tracker** | Track file artifacts produced during agent sessions |

### Memory & State
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/web/assets/images/plugins/plugin-memory.png" width="32"> | **memory** | Model self-curated persistent knowledge across sessions |
| <img src="docs/web/assets/images/plugins/plugin-session.png" width="32"> | **session** | Save and resume conversations across restarts |
| <img src="docs/web/assets/images/plugins/plugin-todo.png" width="32"> | **todo** | Plan registration with progress tracking and workflow enforcement |
| | **waypoint** | Checkpoint and restore conversation state at named points |

### User Interaction
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/web/assets/images/plugins/plugin-permission.png" width="32"> | **permission** | Control tool execution with policies, blacklists, and interactive approval |
| <img src="docs/web/assets/images/plugins/plugin-clarification.png" width="32"> | **clarification** | Request user input with single/multiple choice and free text responses |
| | **prompt_library** | Reusable prompt templates for common workflows |

### Context Management (GC)
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/web/assets/images/plugins/plugin-gc-truncate.png" width="32"> | **gc_truncate** | Simple turn-based garbage collection via truncation |
| <img src="docs/web/assets/images/plugins/plugin-gc-summarize.png" width="32"> | **gc_summarize** | Compression-based GC via summarization |
| <img src="docs/web/assets/images/plugins/plugin-gc-hybrid.png" width="32"> | **gc_hybrid** | Generational approach: recent preserved, middle summarized, oldest truncated |
| | **gc_budget** | Token budget management and threshold monitoring |

### Specialized Capabilities
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/web/assets/images/plugins/plugin-web-search.png" width="32"> | **web_search** | Web search integration for current information |
| | **web_fetch** | Fetch and process web page content |
| <img src="docs/web/assets/images/plugins/plugin-subagent.png" width="32"> | **subagent** | Delegate tasks to specialized subagents with isolated sessions and custom tools |
| <img src="docs/web/assets/images/plugins/plugin-calculator.png" width="32"> | **calculator** | Mathematical calculation tools with configurable precision |
| <img src="docs/web/assets/images/plugins/plugin-references.png" width="32"> | **references** | Inject documentation sources into model context (auto or user-selected) |
| <img src="docs/web/assets/images/plugins/plugin-multimodal.png" width="32"> | **multimodal** | Handle images via @file references with lazy-loading |
| | **vision_capture** | Capture TUI screenshots as SVG/PNG for vision model input |
| | **thinking** | Extended thinking / chain-of-thought support for compatible models |

### Infrastructure
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/web/assets/images/plugins/plugin-model-provider.png" width="32"> | **model_provider** | Provider-agnostic abstraction layer (7 providers) |
| <img src="docs/web/assets/images/plugins/plugin-registry.png" width="32"> | **registry** | Plugin discovery, lifecycle management, and tool exposure control |
| | **introspection** | Runtime self-inspection for tool and plugin discovery |
| | **streaming** | Token-level streaming with cancellation support |
| | **telemetry** | OpenTelemetry tracing integration |
| | **reliability** | Per-tool reliability policies with configurable thresholds |
| | **sandbox_manager** | Sandboxed execution environments for untrusted tools |
| | **service_connector** | External service integration (APIs, databases) |

Plus additional plugins for caching (per-provider), output formatting (code blocks, diffs, tables, Mermaid, notebooks), content filtering, and authentication (per-provider OAuth flows).

For the complete reference, see the **[Plugin Documentation](https://jaato-framework-and-examples.github.io/jaato/web/api-reference/plugins/index.html)**. For plugin development, see [Plugin Development Guide](jaato-server/shared/plugins/README.md).

## Quick Start

### Prerequisites

- Python 3.10+
- An AI provider account (any of the 7 supported providers)

### Installation

jaato is structured as three packages:
- **jaato-sdk** - Lightweight client library and event protocol for building custom clients
- **jaato-server** - Runtime daemon with all plugins, providers, and core logic
- **jaato-tui** - Feature-rich terminal user interface client

```bash
git clone https://github.com/Jaato-framework-and-examples/jaato.git
cd jaato

# Create virtual environment
python3 -m venv .venv

# For contributors: install all packages in development mode
.venv/bin/pip install -e jaato-sdk/. -e "jaato-server/.[all]" -e "jaato-tui/.[all]"

# For SDK users: just the lightweight client library
.venv/bin/pip install jaato-sdk/

# Server with dev tools
.venv/bin/pip install "jaato-server/.[dev]"

# TUI with all optional dependencies
.venv/bin/pip install "jaato-tui/.[all]"
```

### Configuration

1. **Set up your AI provider** - Configure credentials for your chosen provider (see [Provider Setup Guides](https://jaato-framework-and-examples.github.io/jaato/web/api-reference/providers/index.html))
2. **Configure environment** - Copy `.env.example` to `.env` and edit with your credentials
3. **Optional: Add MCP servers** - Configure external tool integrations in `.mcp.json`

## Usage

### Starting the Server

```bash
# Start server as daemon with IPC socket
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon

# Start with both IPC and WebSocket (for remote/web clients)
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --web-socket :8080 --daemon

# Server management
.venv/bin/python -m server --status    # Check if running
.venv/bin/python -m server --stop      # Stop the daemon
```

### Connecting Clients

```bash
# TUI client (interactive)
.venv/bin/python jaato-tui/rich_client.py --connect /tmp/jaato.sock

# Headless mode (scripting)
.venv/bin/python jaato-tui/rich_client.py --connect /tmp/jaato.sock --cmd "What time is it?"
```

### TUI Features

- **Multi-turn conversations** with full context preservation
- **Permission system** with granular approval controls (yes, no, always, per-turn, idle-based)
- **Plan tracking** with the TODO plugin for complex multi-step tasks
- **Session persistence** for saving and resuming conversations across restarts
- **Theming** with built-in dark, light, and high-contrast themes plus custom theme support
- **Configurable keybindings** with external editor integration (Ctrl+G)
- **Search mode** (Ctrl+F) for finding text in session output
- **Subagent delegation** with tabbed agent UI
- **Authentication commands** for Anthropic, GitHub, and Google OAuth flows

### Interactive Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `tools` | List all registered tools |
| `reset` | Clear conversation history and start fresh |
| `model <name>` | Switch to a different model |
| `history` | Display conversation history |
| `context` | Show context window usage |
| `export [file]` | Export session to YAML for replay |
| `plan` | Show current task plan |
| `save` / `resume` | Save or resume sessions |
| `sessions` | List all saved sessions |
| `permissions` | Manage tool permission policies |
| `backtoturn <id>` | Revert conversation to a specific turn |
| `screenshot` | Capture TUI as SVG/PNG |

### Session Export for Replay

Export interactive sessions to YAML for reproducible demos, testing, and sharing:

```
You> List the Python files in the current directory
Model> [executes cli_execute tool...]

You> export my_session.yaml
[Session exported to: my_session.yaml]
  Replay with: python demo-scripts/run_demo.py my_session.yaml
```

```bash
# Replay a session
python demo-scripts/run_demo.py my_session.yaml

# Record as SVG animation
termtosvg -c "python demo-scripts/run_demo.py my_session.yaml" -g 100x40 my_demo.svg
```

See [demo-scripts/README.md](demo-scripts/README.md) for the complete YAML script format.

## Project Structure

```
jaato/
├── jaato-sdk/                     # Client SDK (events, protocol, trace)
│   └── jaato_sdk/
│       ├── events.py              # 25+ typed event definitions
│       ├── client/                # Client connection logic
│       └── plugins/               # Plugin type definitions
├── jaato-server/                  # Server runtime
│   ├── server/                    # Daemon, IPC, WebSocket
│   │   ├── __main__.py            # Entry point with daemon mode
│   │   ├── core.py                # JaatoServer (UI-agnostic core)
│   │   ├── session_manager.py     # Multi-session orchestration
│   │   ├── ipc.py                 # Unix domain socket transport
│   │   └── websocket.py           # WebSocket transport
│   └── shared/                    # Core library
│       ├── jaato_client.py        # JaatoClient facade
│       ├── jaato_runtime.py       # Shared runtime (providers, registry)
│       ├── jaato_session.py       # Per-agent conversation state
│       ├── ai_tool_runner.py      # Tool execution with permissions
│       ├── token_accounting.py    # Token ledger with rate-limit retries
│       ├── mcp_context_manager.py # Multi-server MCP management
│       └── plugins/               # 55+ plugins (see above)
├── jaato-tui/                     # Terminal UI client
│   ├── rich_client.py             # Entry point
│   ├── output_buffer.py           # Output rendering engine
│   ├── pt_display.py              # Prompt toolkit display layer
│   └── backend.py                 # IPC/WebSocket client backend
├── web-client/                    # Web client (React/Vite/Tailwind)
├── docs/                          # Comprehensive documentation (45+ docs)
├── examples/                      # Usage examples
├── out-of-tree-plugins/           # Third-party plugin template
├── cli_vs_mcp/                    # CLI vs MCP benchmarking harness
├── demo-scripts/                  # YAML-driven demo recording
└── scripts/                       # Utility scripts
```

## Environment Variables

### Provider Configuration

| Variable | Provider | Description |
|----------|----------|-------------|
| `PROJECT_ID` | Google GenAI | GCP project ID |
| `LOCATION` | Google GenAI | Vertex AI region (e.g., `us-central1`) |
| `MODEL_NAME` | Google GenAI | Model name (e.g., `gemini-2.5-flash`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google GenAI | Path to service account JSON key |
| `ANTHROPIC_API_KEY` | Anthropic | API key (uses API credits) |
| `ANTHROPIC_AUTH_TOKEN` | Anthropic | OAuth token (uses subscription) |
| `GITHUB_TOKEN` | GitHub Models | GitHub PAT with `models: read` permission |
| `OLLAMA_HOST` | Ollama | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Ollama | Default model name |

### Runtime Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_USE_CHAT_FUNCTIONS` | Enable function calling mode | `1` |
| `LEDGER_PATH` | Token accounting JSONL output path | `token_events_ledger.jsonl` |
| `JAATO_GC_THRESHOLD` | GC trigger threshold % | `80.0` |
| `JAATO_PARALLEL_TOOLS` | Enable parallel tool execution | `true` |
| `JAATO_DEFERRED_TOOLS` | Enable deferred tool loading | `true` |

### Retry & Rate Limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_RETRY_ATTEMPTS` | Max retry attempts for transient errors | `5` |
| `AI_RETRY_BASE_DELAY` | Base delay (seconds) for exponential backoff | `1.0` |
| `AI_RETRY_MAX_DELAY` | Maximum delay (seconds) between retries | `30.0` |
| `AI_REQUEST_INTERVAL` | Minimum seconds between requests | `0` |

### Proxy & Network

| Variable | Description | Default |
|----------|-------------|---------|
| `HTTPS_PROXY` / `HTTP_PROXY` | Standard proxy URL | — |
| `NO_PROXY` | No-proxy hosts (suffix matching) | — |
| `JAATO_KERBEROS_PROXY` | Enable Kerberos/SPNEGO proxy auth | `false` |
| `JAATO_SSL_VERIFY` | SSL certificate verification | `true` |

### OpenTelemetry

| Variable | Description | Default |
|----------|-------------|---------|
| `JAATO_TELEMETRY_ENABLED` | Enable OTel tracing | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | — |

## Tooling

### CLI vs MCP Harness

Benchmark token usage between CLI and MCP tool approaches:

```bash
.venv/bin/python cli_vs_mcp/cli_mcp_harness.py \
  --domain github \
  --scenarios list_issues \
  --domain-params '{"owner": "your-org", "repo": "your-repo"}' \
  --verbose
```

Supported domains: GitHub, Confluence. See `cli_vs_mcp/` for full documentation.

### Sequence Diagram Generator

Generate sequence diagrams from execution traces:

```bash
.venv/bin/python sequence-diagram-generator/trace_to_sequence.py \
  --trace traces/trace.json -o diagram.pdf
```

Supports PDF, PlantUML (`--export-plantuml`), and Mermaid (`--export-mermaid`) output.

### Demo Recording

Record terminal demos from YAML-driven scripts:

```bash
termtosvg -c "python demo-scripts/run_demo.py demo.yaml" -g 100x40 demo.svg
```

## Documentation

**[Full Documentation](https://jaato-framework-and-examples.github.io/jaato/web/index.html)** - Complete reference with examples, guides, and API documentation.

| Resource | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture.md) | Server-first architecture, event protocol, component diagrams |
| [Design Philosophy](docs/design-philosophy.md) | Opinionated design decisions and rationale |
| [Plugin Reference](https://jaato-framework-and-examples.github.io/jaato/web/api-reference/plugins/index.html) | All built-in plugins with configuration and examples |
| [Plugin Development](jaato-server/shared/plugins/README.md) | Guide for creating custom plugins |
| [Provider Setup](https://jaato-framework-and-examples.github.io/jaato/web/api-reference/providers/index.html) | Configuration guides for each model provider |
| [GCP/Vertex AI Setup](docs/gcp-setup.md) | Google Cloud Platform setup walkthrough |
| [OpenTelemetry Design](docs/opentelemetry-design.md) | Tracing integration architecture |
| [Reliability Policies](docs/reliability-policies-config.md) | Per-tool thresholds and retry configuration |

## License

MIT License - See [LICENSE](LICENSE) for details.
