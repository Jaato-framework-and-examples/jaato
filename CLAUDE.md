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
    - Future: `model_provider/anthropic/` for Claude API

- **plugins/model_provider/**: Provider abstraction layer
  - `types.py`: Provider-agnostic types (`ToolSchema`, `Message`, `ProviderResponse`)
  - `base.py`: `ModelProviderPlugin` protocol definition
  - `google_genai/provider.py`: Google GenAI implementation
  - `google_genai/converters.py`: Type conversion utilities

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

| Variable | Purpose |
|----------|---------|
| `PROJECT_ID` | GCP project ID |
| `LOCATION` | Vertex AI region (e.g., `us-central1`, `global`) |
| `MODEL_NAME` | Gemini model (e.g., `gemini-2.5-flash`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key JSON |
| `AI_USE_CHAT_FUNCTIONS` | Enable function calling mode (`1`/`true`) |
| `AI_EXECUTE_TOOLS` | Allow generic tool execution (`1`/`true`) |
| `AI_RETRY_ATTEMPTS` | Max retry attempts for rate limits (default: 5) |
| `LEDGER_PATH` | Output path for token accounting JSONL |

## Additional Documentation

- [GCP Setup Guide](docs/gcp-setup.md) - Setting up GCP project for Vertex AI
- [ModLog Training README](modlog-training-set-test/README.md) - COBOL training set generation
