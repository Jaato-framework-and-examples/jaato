# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**jaato** ("just another agentic tool orchestrator") is an experimental project for exploring:
- Vertex AI SDK integration with Gemini models
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

- **jaato_client.py**: Core client for the framework
  - `JaatoClient`: Unified client using SDK chat API for multi-turn conversations
  - `connect()`, `configure_tools()`, `send_message()` - core methods
  - `get_history()`, `reset_session()` - history access and control
  - SDK manages conversation history internally

- **ai_tool_runner.py**: Central orchestrator for function-calling loops with Vertex AI
  - `ToolExecutor`: Registry mapping tool names to callables
  - `run_function_call_loop()`: Legacy function for direct API usage (JaatoClient handles this internally now)

- **plugins/**: Tool plugin system
  - `PluginRegistry`: Discovers and manages tool plugins
  - `cli.py`: CLI tool plugin for shell commands
  - `mcp.py`: MCP tool plugin for Model Context Protocol servers
  - `permission/`: Permission control for tool execution

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
3. Send message with callback for intermediate responses:
   ```python
   response = jaato.send_message(prompt, on_intermediate_response=lambda text: print(text))
   ```
   The callback receives each intermediate text response during the function-calling loop
   (for real-time display). Returns only the final response text.
4. Internally, SDK chat API handles function calling loop:
   - Model returns function calls → executor runs them → results fed back
   - Intermediate text responses trigger the callback
   - Loop continues until model returns text without function calls
5. Access history when needed: `history = jaato.get_history()`
6. Reset session: `jaato.reset_session()` or `jaato.reset_session(modified_history)`

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
