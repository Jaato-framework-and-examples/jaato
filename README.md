# jaato

<p align="center">
  <img src="docs/jaato-logo.png" alt="jaato logo" width="200"/>
</p>

**j**ust **a**nother **a**gentic **t**ool **o**rchestrator

A provider-agnostic framework for building agentic AI applications with LLM function calling, tool orchestration, and extensible plugin architecture.

**Currently supports:** Google Vertex AI (Gemini models) | **Coming soon:** Anthropic Claude, OpenAI

## Demo

![jaato Demo](demo.svg)

### Etymology

While "jaato" serves as an acronym, the name also carries deeper meaning. In the Himalayan region (Nepal, Sikkim, Darjeeling, and Bhutan), a **jaato** (जाँतो) is a traditional rotary hand-quern or grinder used to mill grains. This ancient tool consists of two round stones with a wooden handle (*hāto*) used to turn the top stone in a circular motion.

The metaphor is intentional: just as a traditional jaato grinds raw grains into refined flour, this orchestrator processes raw inputs through LLM tools to produce refined outputs.

## Overview

jaato is a future-proof framework for building agentic AI applications, featuring:

- **Multi-Provider Support** - Provider-agnostic architecture (currently: Vertex AI/Gemini)
- **Function Calling** - Multi-turn tool execution loops with automatic result feeding
- **Tool Orchestration** - Unified interface for CLI tools and MCP (Model Context Protocol) servers
- **Plugin System** - 14+ built-in plugins for file ops, memory, web search, and more
- **Token Accounting** - Detailed tracking of prompt/output tokens with retry logic

> **Note**: This project is currently experimental, but is being developed with the intention of becoming a production-ready orchestration framework. It follows best engineering practices and patterns, with AI-assisted development applied judiciously rather than relying on "vibe coding."

## Features

- **Plugin Architecture** - Extensible system for adding new tool types
- **Token Ledger** - JSONL logging of all API calls with token counts
- **Rate Limit Handling** - Exponential backoff retry for transient errors
- **Prompt Templates** - Domain-specific templates for different use cases
- **Trace Visualization** - Generate sequence diagrams from execution traces

### Available Plugins

jaato includes **20 built-in plugins** organized into categories for tool orchestration, file operations, session management, and more.

#### Tool Execution
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/api/assets/images/plugins/plugin-cli.png" width="32"> | **cli** | Execute shell commands with intelligent auto-backgrounding for long-running processes |
| <img src="docs/api/assets/images/plugins/plugin-mcp.png" width="32"> | **mcp** | Connect to Model Context Protocol servers for external tool integrations |
| <img src="docs/api/assets/images/plugins/plugin-background.png" width="32"> | **background** | Orchestrate parallel background tasks across all BackgroundCapable plugins |

#### File & Content Management
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/api/assets/images/plugins/plugin-file-edit.png" width="32"> | **file_edit** | File operations with diff-based approval, automatic backups, and undo support |
| <img src="docs/api/assets/images/plugins/plugin-references.png" width="32"> | **references** | Inject documentation sources into model context (auto or user-selected) |
| <img src="docs/api/assets/images/plugins/plugin-multimodal.png" width="32"> | **multimodal** | Handle images via @file references with lazy-loading |
| <img src="docs/api/assets/images/plugins/plugin-slash-command.png" width="32"> | **slash_command** | Process /commands with template variable substitution |

#### Memory & State
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/api/assets/images/plugins/plugin-memory.png" width="32"> | **memory** | Model self-curated persistent knowledge across sessions |
| <img src="docs/api/assets/images/plugins/plugin-session.png" width="32"> | **session** | Save and resume conversations across restarts |
| <img src="docs/api/assets/images/plugins/plugin-todo.png" width="32"> | **todo** | Plan registration with progress tracking and workflow enforcement |

#### User Interaction
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/api/assets/images/plugins/plugin-permission.png" width="32"> | **permission** | Control tool execution with policies, blacklists, and interactive approval |
| <img src="docs/api/assets/images/plugins/plugin-clarification.png" width="32"> | **clarification** | Request user input with single/multiple choice and free text responses |

#### Context Management (GC)
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/api/assets/images/plugins/plugin-gc-truncate.png" width="32"> | **gc_truncate** | Simple turn-based garbage collection via truncation |
| <img src="docs/api/assets/images/plugins/plugin-gc-summarize.png" width="32"> | **gc_summarize** | Compression-based GC via summarization |
| <img src="docs/api/assets/images/plugins/plugin-gc-hybrid.png" width="32"> | **gc_hybrid** | Generational approach combining truncation and summarization |

#### Specialized Capabilities
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/api/assets/images/plugins/plugin-web-search.png" width="32"> | **web_search** | DuckDuckGo web search integration for current information |
| <img src="docs/api/assets/images/plugins/plugin-subagent.png" width="32"> | **subagent** | Delegate tasks to specialized subagents with custom tool configurations |
| <img src="docs/api/assets/images/plugins/plugin-calculator.png" width="32"> | **calculator** | Mathematical calculation tools with configurable precision |

#### Infrastructure
| | Plugin | Description |
|:--:|--------|-------------|
| <img src="docs/api/assets/images/plugins/plugin-model-provider.png" width="32"> | **model_provider** | Provider-agnostic abstraction layer for multi-provider AI support |
| <img src="docs/api/assets/images/plugins/plugin-registry.png" width="32"> | **registry** | Plugin discovery, lifecycle management, and tool exposure control |

📖 **[View full plugin reference with examples →](https://apanoia.github.io/jaato/api/api-reference/plugins/index.html)**

For plugin development, see [shared/plugins/README.md](shared/plugins/README.md). For plugin icon design guidelines, see [docs/plugin-icons.md](docs/plugin-icons.md).

## Quick Start

### Prerequisites

- Python 3.10+
- AI provider account (currently supports Google Vertex AI)

### Installation

```bash
git clone https://github.com/apanoia/jaato.git
cd jaato
pip install .
```

📖 **[Installation guides →](https://apanoia.github.io/jaato/api/index.html#installation)** - Detailed instructions for users, plugin developers, and contributors

### Configuration

jaato uses environment variables and configuration files for setup:

1. **Set up your AI provider** - Configure your model provider (Google Vertex AI, etc.)
2. **Configure environment** - Copy `.env.example` to `.env` and edit with your credentials
3. **Optional: Add MCP servers** - Configure in `.mcp.json` for external tool integrations

📖 **[Provider setup guides →](https://apanoia.github.io/jaato/api/api-reference/providers/index.html)** | **[GCP/Vertex AI setup →](docs/gcp-setup.md)** | **[Environment variables →](#environment-variables)**

## Usage

The primary way to use jaato is through the **Interactive Client**, which provides a full-featured conversational interface for interacting with LLMs and tools.

```bash
# Start interactive session
.venv/bin/python simple-client/interactive_client.py

# Start with an initial prompt, then continue interactively
.venv/bin/python simple-client/interactive_client.py -i "List files in current directory"

# Run a single prompt and exit (non-interactive)
.venv/bin/python simple-client/interactive_client.py -p "What time is it?"
```

### Features

- **Multi-turn conversations** with full context preservation
- **Permission prompts** for tool execution approval (approve, deny, or allow-all)
- **Plan tracking** with the TODO plugin for complex multi-step tasks
- **Session persistence** for saving and resuming conversations across restarts
- **Readline history** with ↑/↓ navigation and persistent history
- **Subagent delegation** for specialized task handling
- **Web search** integration for current information

### Interactive Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `tools` | List all registered tools |
| `reset` | Clear conversation history and start fresh |
| `history` | Display conversation history |
| `context` | Show context window usage |
| `export [file]` | Export session to YAML for replay |
| `plan` | Show current task plan (TODO plugin) |
| `save` | Save current session for later resumption |
| `resume` | Resume a previously saved session |
| `sessions` | List all saved sessions |
| `backtoturn <id>` | Revert conversation to a specific turn |
| `quit` | Exit the session |

### Session Export for Replay

The interactive client can export your conversation session to a YAML file that can be replayed later using the demo runner. This is useful for:

- **Creating reproducible demos** from real interactions
- **Testing and regression** by replaying sessions
- **Sharing workflows** with others

#### Exporting a Session

During an interactive session, use the `export` command:

```
You> List the Python files in the current directory
Model> [executes cli_execute tool...]

You> Show me the git status
Model> [executes cli_execute tool...]

You> export my_session.yaml
[Session exported to: my_session.yaml]
  Steps: 2 interaction(s) + quit
  Replay with: python demo-scripts/run_demo.py my_session.yaml
```

The exported YAML captures your prompts and permission decisions:

```yaml
name: Session Export [2025-12-06 14:30]
timeout: 120
steps:
  - type: List the Python files in the current directory
    permission: y
  - type: Show me the git status
    permission: a
  - type: quit
    delay: 0.08
```

#### Replaying an Exported Session

Use the demo runner to replay the session:

```bash
# Simple replay
python demo-scripts/run_demo.py my_session.yaml

# Record as SVG animation
termtosvg -c "python demo-scripts/run_demo.py my_session.yaml" -g 100x40 my_demo.svg
```

See [demo-scripts/README.md](demo-scripts/README.md) for the complete YAML script format and options.

See [simple-client/README.md](simple-client/README.md) for full documentation.

## Other Tools

### Simple Connectivity Test

Verify your Vertex AI setup is working correctly:

```bash
.venv/bin/python simple-connectivity-test/simple-connectivity-test.py
```

### CLI vs MCP Harness

A benchmarking tool to compare token usage between CLI and MCP tool approaches:

```bash
.venv/bin/python cli_vs_mcp/cli_mcp_harness.py \
  --domain github \
  --scenarios list_issues \
  --domain-params '{"owner": "your-org", "repo": "your-repo"}' \
  --verbose
```

#### CLI Path

Use `--cli-path` to specify the location of a CLI binary if it's not in your PATH:

```bash
.venv/bin/python cli_vs_mcp/cli_mcp_harness.py \
  --domain github \
  --cli-path /usr/local/bin/gh \
  --scenarios list_issues \
  --domain-params '{"owner": "your-org", "repo": "your-repo"}'
```

#### Domain Parameters

The `--domain-params` argument accepts a JSON object with parameters specific to each domain and scenario. The harness substitutes these values into prompt templates.

**GitHub Domain** (`--domain github`)

| Scenario | Parameters | Description |
|----------|------------|-------------|
| `list_issues` | `owner`*, `repo`*, `limit` | List repository issues |
| `get_issue` | `owner`*, `repo`*, `issue_number`* | Get a specific issue |
| `search_issues` | `owner`*, `repo`*, `search_query`, `limit`, `top_n` | Search issues with query |

*Required parameters

Examples:
```bash
# List issues
--domain-params '{"owner": "anthropics", "repo": "claude-code", "limit": 10}'

# Get specific issue
--domain-params '{"owner": "anthropics", "repo": "claude-code", "issue_number": 42}'

# Search issues
--domain-params '{"owner": "anthropics", "repo": "claude-code", "search_query": "bug label:urgent", "limit": 20}'
```

**Confluence Domain** (`--domain confluence`)

| Scenario | Parameters | Description |
|----------|------------|-------------|
| `get_page` | `page_id`* | Retrieve a page by ID |
| `search` | `cql_query`, `limit`, `top_n` | Search using CQL |
| `list_children` | `parent_page_id`*, `limit` | List child pages |
| `update_page` | `page_id`*, `current_title`, `current_body_file`, `change_request` | Update a page |

*Required parameters

Examples:
```bash
# Get page
--domain-params '{"page_id": "123456789"}'

# Search pages
--domain-params '{"cql_query": "space=DEV and type=page", "limit": 10}'

# List children
--domain-params '{"parent_page_id": "123456789", "limit": 20}'
```

### COBOL ModLog Training Generator

Generate training data from COBOL modification logs:

```bash
.venv/bin/python modlog-training-set-test/generate_training_set.py \
  --source modlog-training-set-test/sample_cobol.cbl \
  --out training_data.jsonl \
  --mode full-stream
```

### Sequence Diagram Generator

Generate PDF sequence diagrams from trace files to visualize interactions between the client, orchestrator, LLM, and tools:

```bash
# First, run the harness with tracing enabled
.venv/bin/python cli_vs_mcp/cli_mcp_harness.py \
  --domain github \
  --scenarios list_issues \
  --domain-params '{"owner": "your-org", "repo": "your-repo"}' \
  --trace --trace-dir cli_vs_mcp/traces

# Generate sequence diagram from trace
.venv/bin/python sequence-diagram-generator/trace_to_sequence.py \
  --trace cli_vs_mcp/traces/cli_list_issues_run1.trace.json \
  -o sequence_diagram.pdf
```

You can also export to PlantUML or Mermaid formats:

```bash
# Export PlantUML source
.venv/bin/python sequence-diagram-generator/trace_to_sequence.py \
  --trace traces/trace.json --export-plantuml diagram.puml

# Export Mermaid source
.venv/bin/python sequence-diagram-generator/trace_to_sequence.py \
  --trace traces/trace.json --export-mermaid diagram.mmd
```

### Demo Recording

Record terminal demos of plugins using YAML-driven scripts with `termtosvg`:

```bash
# Record a specific plugin demo
cd demo-scripts
termtosvg -c "python run_demo.py ../shared/plugins/cli/demo.yaml" -g 100x40 ../shared/plugins/cli/demo.svg

# Record all plugin demos
./record_all.sh
```

Demo scripts are defined in YAML format and live in each plugin's directory. See [demo-scripts/README.md](demo-scripts/README.md) for the script format and available options.

## Project Structure

```
jaato/
├── shared/                     # Core library
│   ├── jaato_client.py         # Core client (JaatoClient)
│   ├── ai_tool_runner.py       # Function calling loop orchestrator
│   ├── token_accounting.py     # Token usage tracking
│   ├── mcp_context_manager.py  # MCP server connection manager
│   ├── plugins/                # Tool plugin system
│   │   ├── cli.py              # CLI tool plugin
│   │   ├── mcp.py              # MCP tool plugin
│   │   ├── registry.py         # Plugin discovery & lifecycle
│   │   └── permission/         # Permission control plugin
│   └── prompt_templates/       # Domain-specific prompts
├── simple-client/              # Interactive console client
├── cli_vs_mcp/                 # CLI vs MCP comparison harness
├── demo-scripts/               # YAML-driven demo recording scripts
├── sequence-diagram-generator/ # Trace visualization tool
├── simple-connectivity-test/   # Basic Vertex AI test
├── modlog-training-set-test/   # COBOL training data generator
└── docs/                       # Additional documentation
```

## Environment Variables

### Core Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ID` | GCP project ID | Required |
| `LOCATION` | Vertex AI region | `us-central1` |
| `MODEL_NAME` | Gemini model name | `gemini-2.5-flash` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON key | Required |

### Function Calling

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_USE_CHAT_FUNCTIONS` | Enable function calling mode | `0` (disabled) |
| `AI_FC_MAX_TURNS` | Max iterations for function call loop | `2` |
| `AI_EXECUTE_TOOLS` | Allow generic/dynamic tool execution | `0` (disabled) |

### Retry & Rate Limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_RETRY_ATTEMPTS` | Max retry attempts for transient errors | `5` |
| `AI_RETRY_BASE_DELAY` | Base delay (seconds) for exponential backoff | `1.0` |
| `AI_RETRY_MAX_DELAY` | Maximum delay (seconds) between retries | `30.0` |
| `AI_RETRY_LOG_SILENT` | Suppress retry log messages | `0` (show logs) |

### Logging & Output

| Variable | Description | Default |
|----------|-------------|---------|
| `VERBOSE` | Enable verbose console output | `1` (enabled) |
| `LEDGER_PATH` | Output path for token accounting JSONL | `token_events_ledger.jsonl` |

### SSL/TLS Certificates

| Variable | Description | Default |
|----------|-------------|---------|
| `REQUESTS_CA_BUNDLE` | Custom CA certificate bundle path | System default |
| `SSL_CERT_FILE` | SSL certificate file path | System default |
| `ENV_VALIDATE_CA` | Validate CA paths exist on startup | `0` (disabled) |

### Debug

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_TOOL_RUNNER_DEBUG` | Debug logging for tool execution | `0` (disabled) |

## Documentation

📚 **[Full API Documentation →](https://apanoia.github.io/jaato/api/index.html)** - Complete reference with examples

**Guides:**
- [GCP Setup Guide](docs/gcp-setup.md) - Setting up your GCP project
- [Plugin Reference](https://apanoia.github.io/jaato/api/api-reference/plugins/index.html) - All built-in plugins with configuration options
- [Plugin Development](shared/plugins/README.md) - Creating custom tool plugins
- [ModLog Training](modlog-training-set-test/README.md) - COBOL training data generation
- [Sequence Diagrams](sequence-diagram-generator/README.md) - Trace visualization

## License

MIT License - See [LICENSE](LICENSE) for details.
