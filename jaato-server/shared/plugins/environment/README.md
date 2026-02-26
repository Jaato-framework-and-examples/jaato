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
| `aspect` | string | No | `"all"` | Which aspect to query: `os`, `shell`, `arch`, `cwd`, `terminal`, `context`, `session`, `datetime`, `network`, or `all` |

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
  },
  "network": {
    "proxy": {
      "http_proxy": "http://proxy.corp.com:8080",
      "https_proxy": "http://proxy.corp.com:8080",
      "configured": true
    },
    "proxy_auth": {
      "type": "kerberos",
      "kerberos_enabled": true
    },
    "ssl": {
      "verify": true,
      "requests_ca_bundle": "/etc/pki/tls/certs/corp-ca-bundle.crt"
    },
    "no_proxy": {
      "no_proxy": "localhost,127.0.0.1,.corp.com",
      "jaato_no_proxy": "github.com,api.github.com"
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

# Get network connectivity configuration
get_environment(aspect="network")

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

### Network (`aspect="network"`)

Reports proxy settings, proxy authentication method, SSL/TLS verification config, and no-proxy rules. Credentials embedded in proxy URLs are automatically masked.

#### Proxy Settings

| Field | Description | Example Values |
|-------|-------------|----------------|
| `proxy.http_proxy` | HTTP proxy URL (credentials masked) | `"http://proxy.corp.com:8080"`, `null` |
| `proxy.https_proxy` | HTTPS proxy URL (credentials masked) | `"http://***:***@proxy.corp.com:8080"`, `null` |
| `proxy.configured` | Whether any proxy is configured | `true`, `false` |

Reads from `HTTPS_PROXY`/`https_proxy` and `HTTP_PROXY`/`http_proxy` environment variables (uppercase takes precedence).

#### Proxy Authentication

| Field | Description | Example Values |
|-------|-------------|----------------|
| `proxy_auth.type` | Detected authentication method | `"none"`, `"basic"`, `"kerberos"` |
| `proxy_auth.kerberos_enabled` | Kerberos/SPNEGO enabled (only present when true) | `true` |

- `"kerberos"`: `JAATO_KERBEROS_PROXY=true` is set
- `"basic"`: Proxy URL contains embedded `user:password@`
- `"none"`: No proxy authentication detected

#### SSL / TLS

| Field | Description | Example Values |
|-------|-------------|----------------|
| `ssl.verify` | Whether SSL certificate verification is enabled | `true`, `false` |
| `ssl.ssl_cert_file` | Custom CA certificate file (if `SSL_CERT_FILE` is set) | `"/etc/ssl/custom-ca.pem"` |
| `ssl.ssl_cert_dir` | Custom CA certificate directory (if `SSL_CERT_DIR` is set) | `"/etc/ssl/certs"` |
| `ssl.requests_ca_bundle` | CA bundle for requests library (if `REQUESTS_CA_BUNDLE` is set) | `"/etc/pki/tls/certs/ca.crt"` |
| `ssl.curl_ca_bundle` | CA bundle for curl (if `CURL_CA_BUNDLE` is set) | `"/etc/pki/tls/certs/ca.crt"` |

`ssl.verify` defaults to `true`; set `JAATO_SSL_VERIFY=false` to disable (escape hatch for SSL-intercepting proxies).

#### No-Proxy Rules

| Field | Description | Example Values |
|-------|-------------|----------------|
| `no_proxy.no_proxy` | Standard no-proxy hosts (`NO_PROXY`) | `"localhost,127.0.0.1,.internal.corp"` |
| `no_proxy.jaato_no_proxy` | Exact-match no-proxy hosts (`JAATO_NO_PROXY`) | `"github.com,api.github.com"` |

Returns `null` when no rules are configured.

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

6. **Network configuration**: Query network aspect to understand connectivity constraints
   - Detect whether a proxy is required and which host/port to use
   - Determine proxy authentication method (Kerberos, basic credentials)
   - Check if SSL verification is relaxed (corporate SSL-intercepting proxies)
   - Find custom CA bundle paths for HTTPS connections
   - Identify no-proxy bypass rules for internal hosts

7. **Context management**: Query context aspect to monitor token usage
   - Proactively summarize or trim context before hitting GC threshold
   - Track cost/usage during long conversations

## Auto-Approval

This plugin's tool is auto-approved (no permission prompt required) because it only reads environment information and has no side effects.

## Configuration

This plugin requires no configuration. For context aspect functionality, inject the session via `set_session_plugin()`.
