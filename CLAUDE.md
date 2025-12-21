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

## Architecture

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

## Additional Documentation

- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
- [ModLog Training README](modlog-training-set-test/README.md) - COBOL training set generation
