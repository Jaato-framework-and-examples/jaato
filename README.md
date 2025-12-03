# jaato

<p align="center">
  <img src="docs/jaato_(traditional grinder)_logo.jpg" alt="jaato logo" width="200"/>
</p>

**j**ust **a**nother **a**gentic **t**ool **o**rchestrator

An experimental project for exploring LLM function calling patterns with Google's Vertex AI and Gemini models.

### Etymology

While "jaato" serves as an acronym, the name also carries deeper meaning. In the Himalayan region (Nepal, Sikkim, Darjeeling, and Bhutan), a **jaato** (जाँतो) is a traditional rotary hand-quern or grinder used to mill grains. This ancient tool consists of two round stones with a wooden handle (*hāto*) used to turn the top stone in a circular motion.

The metaphor is intentional: just as a traditional jaato grinds raw grains into refined flour, this orchestrator processes raw inputs through LLM tools to produce refined outputs.

## Overview

jaato is a sandbox for experimenting with:

- **Vertex AI Integration** - Using the `google-genai` SDK with Gemini models
- **Function Calling** - Multi-turn tool execution loops with automatic result feeding
- **Tool Orchestration** - Unified interface for CLI tools and MCP (Model Context Protocol) servers
- **Token Accounting** - Detailed tracking of prompt/output tokens with retry logic

> **Note**: This is an experimental project, not intended for production use.

## Features

- **Plugin Architecture** - Extensible system for adding new tool types
- **CLI Tool Execution** - Run local command-line tools via subprocess
- **MCP Server Support** - Connect to multiple MCP servers and auto-discover their tools
- **Token Ledger** - JSONL logging of all API calls with token counts
- **Rate Limit Handling** - Exponential backoff retry for transient errors
- **Prompt Templates** - Domain-specific templates for different use cases
- **Trace Visualization** - Generate sequence diagrams from execution traces

## Prerequisites

- Python 3.10+
- Google Cloud Platform account with Vertex AI enabled
- Service account with Vertex AI permissions

## Installation

```bash
# Clone the repository
git clone https://github.com/apanoia/jaato.git
cd jaato

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration:
   ```bash
   PROJECT_ID=your-gcp-project-id
   LOCATION=us-central1
   MODEL_NAME=gemini-2.5-flash
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

3. (Optional) Configure MCP servers in `.mcp.json`:
   ```json
   {
     "mcpServers": {
       "GitHub": {
         "type": "stdio",
         "command": "mcp-server-github",
         "env": {
           "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
         }
       }
     }
   }
   ```

## Usage

### CLI vs MCP Harness

Compare token usage between CLI and MCP tool approaches:

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

### Simple Connectivity Test

Verify your Vertex AI setup:

```bash
.venv/bin/python simple-connectivity-test/simple-connectivity-test.py
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

- [GCP Setup Guide](docs/gcp-setup.md) - Setting up your GCP project
- [Plugin System](shared/plugins/README.md) - Creating custom tool plugins
- [ModLog Training](modlog-training-set-test/README.md) - COBOL training data generation
- [Sequence Diagrams](sequence-diagram-generator/README.md) - Trace visualization

## License

MIT License - See [LICENSE](LICENSE) for details.
