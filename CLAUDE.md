# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**jaato** ("just another agentic tool orchestrator") is an experimental project for exploring:
- Multi-provider AI SDK integration (Google GenAI, Anthropic, etc.)
- Function calling patterns with LLMs
- Tool orchestration (CLI tools and MCP servers)

This is not intended to be a production tool, but a sandbox for experimentation.

## Commands

### Environment Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Running the CLI vs MCP Harness
```bash
.venv/bin/python cli_vs_mcp/cli_mcp_harness.py \
  --env-file .env \
  --scenarios get_page \
  --page-id 12345 \
  --trace --verbose
```

### Running the ModLog Training Set Generator
```bash
.venv/bin/python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --out training_data.jsonl \
  --mode full-stream
```

### Running the Vertex AI Connectivity Test
```bash
.venv/bin/python test_vertex.py
```

### Running the Server (Multi-Client Mode)
```bash
# Start server as daemon with IPC socket
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --daemon

# Start server with both IPC and WebSocket
.venv/bin/python -m server --ipc-socket /tmp/jaato.sock --web-socket :8080 --daemon

# Check server status
.venv/bin/python -m server --status

# Stop server
.venv/bin/python -m server --stop

# Connect TUI client to running server
.venv/bin/python rich-client/rich_client.py --connect /tmp/jaato.sock
```

## Architecture

### Server-First Architecture (`server/`)

The framework uses a server-first architecture where the server runs as a daemon and clients connect via IPC or WebSocket:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         JaatoDaemon                                  │
│  python -m server --ipc-socket /tmp/jaato.sock --web-socket :8080   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ IPC Server  │    │  WS Server  │    │    SessionManager       │  │
│  │ (Unix sock) │    │ (WebSocket) │    │  ┌─────────────────┐    │  │
│  └──────┬──────┘    └──────┬──────┘    │  │ Session "main"  │    │  │
│         │                  │           │  │ ┌─────────────┐ │    │  │
│         └──────────────────┴───────────│  │ │ JaatoServer │ │    │  │
│                    │                   │  │ └─────────────┘ │    │  │
│                    ▼                   │  └─────────────────┘    │  │
│         ┌──────────────────────┐       │  ┌─────────────────┐    │  │
│         │    Event Router      │◄──────┤  │ Session "dev"   │    │  │
│         │  (broadcast events)  │       │  └─────────────────┘    │  │
│         └──────────────────────┘       └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         ▲              ▲              ▲
         │              │              │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │   TUI   │    │   Web   │    │   IDE   │
    │ Client  │    │ Client  │    │ Plugin  │
    └─────────┘    └─────────┘    └─────────┘
```

**Server Components:**

- **`server/__main__.py`**: Entry point with daemon mode, PID management
  - `--ipc-socket PATH`: Unix domain socket for local clients
  - `--web-socket [HOST:]PORT`: WebSocket for remote clients
  - `--daemon`: Run as background process
  - `--status`: Check if server is running
  - `--stop`: Stop running daemon

- **`server/core.py`**: `JaatoServer` - UI-agnostic core logic
  - Wraps `JaatoClient` with event emission instead of callbacks
  - Emits typed events for all UI interactions
  - Handles permission requests, tool execution, streaming

- **`server/events.py`**: Event protocol (25+ typed events)
  - Server→Client: `AgentOutputEvent`, `PermissionRequestedEvent`, `PlanUpdatedEvent`, etc.
  - Client→Server: `SendMessageRequest`, `PermissionResponseRequest`, `StopRequest`, etc.
  - `serialize_event()`, `deserialize_event()` for JSON transport

- **`server/session_manager.py`**: Multi-session orchestration
  - Manages multiple named sessions with on-demand loading
  - Integrates with `SessionPlugin` for disk persistence
  - Tracks client→session mappings for event routing

- **`server/ipc.py`**: Unix domain socket server
  - Length-prefixed framing (4-byte header + JSON)
  - Fast, secure local communication

- **`server/websocket.py`**: WebSocket server
  - Remote client support via `websockets` library
  - Same event protocol as IPC

**Client Connection:**

```python
from rich_client.ipc_client import IPCClient

client = IPCClient("/tmp/jaato.sock", auto_start=True)
await client.connect()

# Send message
await client.send_message("Hello, world!")

# Receive events
async for event in client.events():
    if isinstance(event, AgentOutputEvent):
        print(event.text)
```

### Core Components (`shared/`)

- **jaato_client.py**: Core client (facade) for the framework
  - `JaatoClient`: Backwards-compatible facade wrapping `JaatoRuntime` + `JaatoSession`
  - `connect()`, `configure_tools()`, `send_message()` - core methods (unchanged API)
  - `get_runtime()` - access shared runtime for subagent session creation
  - `get_session()` - access main session for direct manipulation

- **jaato_runtime.py**: Shared environment (resources used across agents)
  - `JaatoRuntime`: Manages provider config, plugin registry, permissions, ledger
  - `connect(project, location)` - establish provider configuration
  - `configure_plugins(registry, permission_plugin, ledger)` - setup shared resources
  - `create_session(model, tools, system_instructions)` - spawn lightweight sessions

- **jaato_session.py**: Per-agent conversation state
  - `JaatoSession`: Isolated session with history, model, tool subset
  - `send_message()`, `get_history()`, `reset_session()` - conversation methods
  - `set_agent_context(agent_type, agent_name)` - for permission context
  - Sessions share runtime resources but maintain isolated conversation state

- **ai_tool_runner.py**: Tool execution infrastructure
  - `ToolExecutor`: Registry mapping tool names to callables with permission checking and auto-backgrounding

- **plugins/**: Plugin system with three plugin types:
  - **Tool Plugins**: Provide tools the model can invoke
    - `PluginRegistry`: Discovers and manages tool plugins
    - `cli/`: CLI tool plugin for shell commands
    - `mcp/`: MCP tool plugin for Model Context Protocol servers
    - `permission/`: Permission control for tool execution
    - `file_edit/`, `todo/`, `web_search/`, etc.
  - **GC Plugins**: Garbage collection strategies for context management
    - `gc_truncate/`: Simple truncation strategy
    - `gc_summarize/`: Summarization-based strategy
    - `gc_hybrid/`: Combined approach
  - **Model Provider Plugins**: SDK abstraction for multi-provider support
    - `model_provider/`: Provider-agnostic types and protocol
    - `model_provider/google_genai/`: Google GenAI/Vertex AI implementation
    - `model_provider/github_models/`: GitHub Models API (GPT, Claude, Gemini via GitHub)

- **plugins/model_provider/**: Provider abstraction layer
  - `types.py`: Provider-agnostic types (`ToolSchema`, `Message`, `ProviderResponse`)
  - `base.py`: `ModelProviderPlugin` protocol definition
  - `google_genai/`: Google GenAI/Vertex AI implementation
  - `github_models/`: GitHub Models API implementation (uses `azure-ai-inference` SDK)
  - `anthropic/`: Anthropic Claude implementation (uses `anthropic` SDK)

- **mcp_context_manager.py**: Multi-server MCP client manager
  - `MCPClientManager`: Manages persistent connections to multiple MCP servers
  - Auto-discovers tools from connected servers
  - Supports `call_tool_auto()` to find which server has a tool

- **token_accounting.py**: Token usage tracking and retry logic
  - `TokenLedger`: Records prompt/output tokens, handles rate-limit retries with exponential backoff
  - Writes events to JSONL ledger files

### Tool Execution Flow

1. Create `JaatoClient` and connect: `jaato.connect(project, location, model)`
2. Configure tools from plugin registry: `jaato.configure_tools(registry, permission_plugin)`
3. Send message with callback for real-time output:
   ```python
   response = jaato.send_message(prompt, on_output=lambda source, text, mode: print(f"[{source}]: {text}"))
   ```
   The callback receives `(source, text, mode)` for each output:
   - `source`: "model" for model responses, plugin name for plugin output
   - `text`: The output text
   - `mode`: "write" for new block, "append" to continue
   Returns only the final response text.
4. Internally, SDK chat API handles function calling loop:
   - Model returns function calls → executor runs them → results fed back
   - Intermediate text responses trigger the callback
   - Loop continues until model returns text without function calls
5. Access history when needed: `history = jaato.get_history()`
6. Reset session: `jaato.reset_session()` or `jaato.reset_session(modified_history)`

### Subagent Architecture

Subagents share the parent's `JaatoRuntime` but get their own `JaatoSession`:

```
┌─────────────────────────────────────────────────────────┐
│                    JaatoClient (facade)                 │
│  • Backwards-compatible API for existing code           │
│  • get_runtime() → access shared environment            │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────┐           ┌──────────────────┐
│   JaatoRuntime   │           │   JaatoSession   │
│  • Provider cfg  │◄─────────►│  (main agent)    │
│  • Registry      │           │  • History       │
│  • Permissions   │           │  • Model         │
│  • Ledger        │           │  • Tools         │
└──────────────────┘           └──────────────────┘
          │
          │ create_session() - lightweight
          ▼
┌──────────────────┐
│   JaatoSession   │
│   (subagent)     │
│  • Own history   │
│  • Own model     │
│  • Tool subset   │
└──────────────────┘
```

Benefits:
- **No redundant connections** - subagents share provider config
- **Fast spawning** - `create_session()` is lightweight
- **Resource sharing** - registry, permissions, ledger shared

### MCP Server Configuration

MCP servers are configured in `.mcp.json`:
```json
{
  "mcpServers": {
    "Atlassian": {
      "type": "stdio",
      "command": "mcp-atlassian"
    }
  }
}
```

### Streaming & Cancellation

The framework supports streaming responses and mid-turn cancellation:

```python
import threading

# Streaming is enabled by default
client.set_streaming_enabled(True)  # or False for batched

# Start message in background
def run():
    response = client.send_message("Tell me a long story")

thread = threading.Thread(target=run)
thread.start()

# Cancel mid-response
if client.is_processing:
    client.stop()  # Returns partial text + "[Generation cancelled]"

thread.join()
```

Key cancellation types in `shared/plugins/model_provider/types.py`:
- `CancelToken`: Thread-safe cancellation signaling
- `CancelledException`: Raised when operation is cancelled
- `FinishReason.CANCELLED`: Indicates cancelled generation

Provider streaming methods:
- `supports_streaming()`: Check if provider supports streaming
- `supports_stop()`: Check if provider supports mid-turn cancellation
- `send_message_streaming(message, on_chunk, cancel_token)`: Stream with cancellation
- `send_tool_results_streaming(results, on_chunk, cancel_token)`: Stream tool responses

Session/client methods:
- `client.stop()` / `session.request_stop()`: Request cancellation
- `client.is_processing` / `session.is_running`: Check if message in progress
- `client.supports_stop` / `session.supports_stop`: Check if stop is supported
- `client.set_streaming_enabled(bool)`: Toggle streaming mode
- `session.set_retry_callback(callback)`: Route retry notifications through custom handler

Retry callback signature: `(message: str, attempt: int, max_attempts: int, delay: float) -> None`

When using subagents, the retry callback is automatically propagated to subagent sessions
via `SubagentPlugin.set_retry_callback()`, ensuring retry messages from all agents route
through the same channel (e.g., rich client's output panel) instead of printing to console.

### Proactive Garbage Collection

The framework supports proactive context garbage collection that monitors token usage
during streaming and automatically triggers GC when thresholds are exceeded:

```python
from shared.plugins.gc_truncate import create_plugin
from shared.plugins.gc import GCConfig

# Setup GC plugin with threshold
gc_plugin = create_plugin()
gc_plugin.initialize({"preserve_recent_turns": 5, "notify_on_gc": True})

gc_config = GCConfig(
    threshold_percent=80.0,    # Trigger when context is 80% full
    preserve_recent_turns=5,   # Keep last 5 turns
    auto_trigger=True,         # Enable automatic triggering
    check_before_send=True     # Check before each send_message()
)

client.set_gc_plugin(gc_plugin, gc_config)

# Optional: Get notified when threshold is crossed during streaming
def on_gc_threshold(percent_used: float, threshold: float) -> None:
    print(f"Warning: Context at {percent_used:.1f}%, GC will run after turn")

response = client.send_message(
    "Long prompt...",
    on_gc_threshold=on_gc_threshold  # Optional callback
)
# GC automatically runs after turn if threshold was crossed
```

Key GC types in `shared/plugins/gc/`:
- `GCConfig`: Configuration for GC behavior (thresholds, triggers)
- `GCPlugin`: Protocol for GC strategy plugins
- `GCResult`: Result of a GC operation (tokens freed, items collected)
- `GCTriggerReason`: Why GC was triggered (THRESHOLD, MANUAL, TURN_LIMIT)

GC strategies:
- `gc_truncate`: Simple removal of oldest turns
- `gc_summarize`: Compress old turns via summarization
- `gc_hybrid`: Generational approach (recent preserved, middle summarized, ancient truncated)

Proactive GC flow:
1. Token usage is monitored during streaming via `on_usage_update` callback
2. When usage crosses `threshold_percent`, a flag is set
3. After the turn completes, GC is automatically triggered
4. Optional `on_gc_threshold` callback notifies the application

### Plugin Type System

The project uses provider-agnostic types throughout the plugin system:

```python
# Tool declarations use ToolSchema (not SDK-specific types)
from shared.plugins.model_provider.types import ToolSchema

class MyPlugin:
    def get_tool_schemas(self) -> List[ToolSchema]:
        return [ToolSchema(
            name='my_tool',
            description='Does something useful',
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"]
            }
        )]
```

```python
# Conversation history uses Message (not types.Content)
from shared.plugins.model_provider.types import Message, Role

history: List[Message] = client.get_history()
for msg in history:
    print(f"{msg.role}: {msg.text}")
```

Key types in `shared/plugins/model_provider/types.py`:
- `ToolSchema`: Provider-agnostic function declaration
- `Message`: Conversation message with role and parts
- `Part`: Message content (text, function_call, function_response)
- `ProviderResponse`: Unified response from any provider
- `FunctionCall`, `ToolResult`: Function calling types

## Key Environment Variables

### Google GenAI / Vertex AI
| Variable | Purpose |
|----------|---------|
| `PROJECT_ID` | GCP project ID |
| `LOCATION` | Vertex AI region (e.g., `us-central1`, `global`) |
| `MODEL_NAME` | Gemini model (e.g., `gemini-2.5-flash`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key JSON |

### GitHub Models
| Variable | Purpose |
|----------|---------|
| `GITHUB_TOKEN` | GitHub PAT with `models: read` permission |
| `JAATO_GITHUB_ORGANIZATION` | Organization for billing attribution |
| `JAATO_GITHUB_ENTERPRISE` | Enterprise name (for context) |
| `JAATO_GITHUB_ENDPOINT` | Override API endpoint URL |

### Anthropic Claude
| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (uses API credits) |
| `ANTHROPIC_AUTH_TOKEN` | OAuth token for Claude Pro/Max subscription (experimental) |
| `CLAUDE_CODE_OAUTH_TOKEN` | Alternative OAuth token env var (Claude Code CLI) |

**Authentication Options:**
- **API Key** (`sk-ant-api03-...`): Uses API credits from console.anthropic.com
- **OAuth Token** (`sk-ant-oat01-...`): Attempts to use Claude Pro/Max subscription (experimental)

> **⚠️ OAuth Token Warning:** OAuth tokens are currently restricted by Anthropic to
> only work with official Claude Code clients. Third-party tools may receive:
> `"This credential is only authorized for use with Claude Code"`
>
> The OAuth support is included for future compatibility if Anthropic relaxes this
> restriction, or if the correct request format is discovered.

To get an OAuth token for your subscription:
```bash
# Install Claude Code CLI
npm install -g @anthropic/claude-code

# Generate OAuth token (valid for 1 year)
claude setup-token

# Set the token
export ANTHROPIC_AUTH_TOKEN='sk-ant-oat01-...'
```

Configuration options via `ProviderConfig.extra`:
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `oauth_token` | str | None | OAuth token (alternative to env var) |
| `enable_caching` | bool | False | Enable prompt caching (90% cost reduction) |
| `enable_thinking` | bool | False | Enable extended thinking (reasoning traces) |
| `thinking_budget` | int | 10000 | Max thinking tokens when enabled |

### General
| Variable | Purpose |
|----------|---------|
| `AI_USE_CHAT_FUNCTIONS` | Enable function calling mode (`1`/`true`) |
| `AI_EXECUTE_TOOLS` | Allow generic tool execution (`1`/`true`) |
| `LEDGER_PATH` | Output path for token accounting JSONL |
| `JAATO_GC_THRESHOLD` | GC trigger threshold percentage (default: 80.0) |

### Rate Limiting
| Variable | Purpose |
|----------|---------|
| `AI_REQUEST_INTERVAL` | Minimum seconds between API requests (default: 0 = disabled) |
| `AI_RETRY_ATTEMPTS` | Max retry attempts for rate limits (default: 5) |
| `AI_RETRY_BASE_DELAY` | Initial retry delay in seconds (default: 1.0) |
| `AI_RETRY_MAX_DELAY` | Maximum retry delay in seconds (default: 30.0) |
| `AI_RETRY_LOG_SILENT` | Suppress retry logging (`1`/`true`/`yes`) |

### Clipboard
| Variable | Purpose |
|----------|---------|
| `JAATO_COPY_MECHANISM` | Clipboard provider: `osc52` (default) |
| `JAATO_COPY_SOURCES` | Sources to include: `model` (default), or `model&user&tool` |

## Rich Client Keybindings

The rich client supports customizable keybindings via:
1. **Project config**: `.jaato/keybindings.json`
2. **User config**: `~/.jaato/keybindings.json`
3. **Environment variables**: `JAATO_KEY_<ACTION>=<key>`

Priority: Environment variables > Project config > User config > Defaults

### Configuration File Format

```json
{
  "submit": "enter",
  "newline": ["escape", "enter"],
  "clear_input": ["escape", "escape"],
  "cancel": "c-c",
  "exit": "c-d",
  "scroll_up": "pageup",
  "scroll_down": "pagedown",
  "scroll_top": "home",
  "scroll_bottom": "end",
  "nav_up": "up",
  "nav_down": "down",
  "pager_quit": "q",
  "pager_next": "space",
  "toggle_plan": "c-p",
  "toggle_tools": "c-t",
  "cycle_agents": "c-a",
  "yank": "c-y",
  "view_full": "v"
}
```

### Key Syntax (prompt_toolkit)

- Simple keys: `enter`, `space`, `tab`, `q`, `v`
- Control: `c-c`, `c-d`, `c-p` (Ctrl+C, Ctrl+D, Ctrl+P)
- Function keys: `f1`, `f2`, `f12`
- Special: `pageup`, `pagedown`, `home`, `end`, `up`, `down`
- Multi-key sequences: `["escape", "enter"]` or `"escape enter"`

### Reloading Keybindings

Use the `/keybindings reload` command to reload keybindings without restarting:
```
/keybindings reload
```

## Additional Documentation

- [Architecture Overview](docs/architecture.md) - Server-first architecture, event protocol, component diagrams
- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
- [ModLog Training README](modlog-training-set-test/README.md) - COBOL training set generation
