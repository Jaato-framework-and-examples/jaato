# Environment Plugin

The environment plugin provides the `get_environment` tool for querying local execution environment details in the jaato framework.

## Overview

This plugin enables models to understand the execution environment before generating platform-specific commands, paths, or configurations. It exposes OS type, shell, architecture, and working directory information through a single tool with optional aspect filtering.

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
| `aspect` | string | No | `"all"` | Which aspect to query: `os`, `shell`, `arch`, `cwd`, or `all` |

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
  "cwd": "/home/user/project"
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

## Auto-Approval

This plugin's tool is auto-approved (no permission prompt required) because it only reads environment information and has no side effects.

## Configuration

This plugin requires no configuration.
