# Environment Plugin

The environment plugin provides the `get_environment` tool for querying execution environment details in the jaato framework.

## Overview

This plugin enables models to understand both the external execution environment (OS, shell, architecture) and internal context (token usage, GC thresholds). It exposes all information through a single tool with optional aspect filtering.

## Tool Declaration

The plugin exposes a single tool:

| Tool | Description |
|------|-------------|
| `get_environment` | Query execution environment details |

### Parameters

```json
{
  "aspect": "all"
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `aspect` | string | No | `"all"` | Which aspect to query: `os`, `shell`, `arch`, `cwd`, `terminal`, `context`, or `all` |

### Response

When `aspect="all"` (default):
```json
{
  "os": {
    "type": "linux",
    "name": "Linux",
    "release": "5.15.0",
    "friendly_name": "Linux"
  },
  "shell": {
    "default": "bash",
    "current": "bash",
    "path": "/bin/bash",
    "path_separator": ":",
    "dir_separator": "/"
  },
  "arch": {
    "machine": "x86_64",
    "processor": "x86_64",
    "normalized": "x86_64"
  },
  "cwd": "/home/user/project",
  "terminal": {
    "term": "xterm-256color",
    "term_program": "iTerm.app",
    "colorterm": "truecolor",
    "multiplexer": null,
    "color_depth": "24bit",
    "emulator": "iTerm.app"
  },
  "context": {
    "model": "gemini-2.5-flash",
    "context_limit": 1048576,
    "total_tokens": 15234,
    "prompt_tokens": 12000,
    "output_tokens": 3234,
    "tokens_remaining": 1033342,
    "percent_used": 1.45,
    "turns": 5,
    "gc": {
      "threshold_percent": 80.0,
      "auto_trigger": true,
      "preserve_recent_turns": 5
    }
  }
}
```

When querying a single aspect (e.g., `aspect="os"`), the response is flattened:
```json
{
  "type": "linux",
  "name": "Linux",
  "release": "5.15.0",
  "friendly_name": "Linux"
}
```

## Usage

### Basic Setup

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover()
registry.expose_tool("environment")
```

### With JaatoClient

```python
from shared import JaatoClient, PluginRegistry

client = JaatoClient()
client.connect(project_id, location, model_name)

registry = PluginRegistry()
registry.discover()
registry.expose_tool("environment")

client.configure_tools(registry)
response = client.send_message("What OS am I running on?")
```

### Enabling Context Aspect

The `context` aspect requires session injection to access token usage. Use `set_session_plugin()`:

```python
from shared.plugins.environment import EnvironmentPlugin

plugin = EnvironmentPlugin()
client.set_session_plugin(plugin)  # Injects session reference

# Now the model can query context usage
response = client.send_message("How many tokens have I used?")
```

### Querying Specific Aspects

The tool supports filtering by aspect to reduce response size:

```python
# Get only OS info
get_environment(aspect="os")

# Get only shell info
get_environment(aspect="shell")

# Get only architecture
get_environment(aspect="arch")

# Get only working directory
get_environment(aspect="cwd")

# Get terminal emulation info
get_environment(aspect="terminal")

# Get token usage and GC thresholds
get_environment(aspect="context")

# Get everything (default)
get_environment()
get_environment(aspect="all")
```

## Aspect Details

### OS (`aspect="os"`)

| Field | Description | Example Values |
|-------|-------------|----------------|
| `type` | Lowercase OS identifier | `"linux"`, `"darwin"`, `"windows"` |
| `name` | Platform name from Python | `"Linux"`, `"Darwin"`, `"Windows"` |
| `release` | OS release/version | `"5.15.0"`, `"22.1.0"`, `"10"` |
| `friendly_name` | Human-readable name | `"Linux"`, `"macOS"`, `"Windows"` |

### Shell (`aspect="shell"`)

| Field | Description | Example Values |
|-------|-------------|----------------|
| `default` | Default shell name | `"bash"`, `"zsh"`, `"cmd"` |
| `current` | Currently detected shell | `"bash"`, `"zsh"`, `"powershell"` |
| `path` | Full path to shell (Unix) | `"/bin/bash"`, `"/bin/zsh"` |
| `path_separator` | PATH delimiter | `":"` (Unix), `";"` (Windows) |
| `dir_separator` | Directory separator | `"/"` (Unix), `"\\"` (Windows) |
| `powershell_available` | PowerShell present (Windows) | `true`, `false` |
| `pwsh_available` | PowerShell Core present (Windows) | `true`, `false` |

### Architecture (`aspect="arch"`)

| Field | Description | Example Values |
|-------|-------------|----------------|
| `machine` | Raw machine type | `"x86_64"`, `"arm64"`, `"AMD64"` |
| `processor` | Processor description | `"x86_64"`, `"arm"` |
| `normalized` | Standardized architecture | `"x86_64"`, `"arm64"`, `"x86"` |

### Working Directory (`aspect="cwd"`)

Returns a string (not an object) with the absolute path to the current working directory.

### Terminal (`aspect="terminal"`)

| Field | Description | Example Values |
|-------|-------------|----------------|
| `term` | TERM environment variable | `"xterm-256color"`, `"screen"`, `"dumb"` |
| `term_program` | Terminal application | `"iTerm.app"`, `"Apple_Terminal"`, `"vscode"` |
| `colorterm` | Color capability hint | `"truecolor"`, `"24bit"`, `null` |
| `multiplexer` | Terminal multiplexer in use | `"tmux"`, `"screen"`, `null` |
| `color_depth` | Detected color support | `"24bit"`, `"256"`, `"basic"`, `"none"` |
| `emulator` | Terminal emulator type | `"iTerm.app"`, `"xterm-compatible"`, `"linux-console"` |

### Context (`aspect="context"`)

Returns token usage and garbage collection settings. Requires session injection via `set_session()`.

| Field | Description | Example Values |
|-------|-------------|----------------|
| `model` | Model name | `"gemini-2.5-flash"` |
| `context_limit` | Maximum context window size | `1048576` |
| `total_tokens` | Total tokens used so far | `15234` |
| `prompt_tokens` | Input tokens used | `12000` |
| `output_tokens` | Output tokens generated | `3234` |
| `tokens_remaining` | Tokens available before limit | `1033342` |
| `percent_used` | Percentage of context used | `1.45` |
| `turns` | Number of conversation turns | `5` |
| `gc` | Garbage collection settings (if configured) | See below |

#### GC Settings

| Field | Description | Example Values |
|-------|-------------|----------------|
| `threshold_percent` | Trigger GC at this usage % | `80.0` |
| `auto_trigger` | Whether GC triggers automatically | `true` |
| `preserve_recent_turns` | Turns to always keep | `5` |
| `max_turns` | Maximum turns before GC (if set) | `100` |

## Use Cases

1. **Generating shell commands**: Query shell aspect to determine correct syntax
   - Unix: `ls -la`, `cat file.txt`
   - Windows: `dir`, `type file.txt`

2. **Constructing file paths**: Query shell aspect for correct separators
   - Unix: `/home/user/file.txt`
   - Windows: `C:\Users\user\file.txt`

3. **Detecting available tools**: Query OS to know which utilities exist
   - Linux/macOS: `grep`, `sed`, `awk`
   - Windows: `findstr`, PowerShell cmdlets

4. **Architecture-specific decisions**: Query arch for binary selection
   - Download correct binaries for x86_64 vs arm64

5. **Terminal capabilities**: Query terminal aspect for output formatting
   - Use colors/formatting only if supported
   - Detect tmux/screen for session awareness
   - Adjust output width based on terminal type

6. **Context management**: Query context aspect to monitor token usage
   - Proactively summarize or trim context before hitting GC threshold
   - Track cost/usage during long conversations

## Auto-Approval

This plugin's tool is auto-approved (no permission prompt required) because it only reads environment information and has no side effects.

## Configuration

This plugin requires no configuration. For context aspect functionality, inject the session via `set_session_plugin()`.
