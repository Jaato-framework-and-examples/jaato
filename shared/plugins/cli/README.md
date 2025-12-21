# CLI Tool Plugin

The CLI plugin provides the `cli_based_tool` function for executing local shell commands in the jaato framework.

## Demo

The demo below shows the CLI plugin executing shell commands: first listing Python files in the current directory, then showing git status. Each command requires permission approval before execution.

![CLI Plugin Demo](demo.svg)

## Overview

This plugin allows models to run shell commands on the local machine via subprocess. Simple commands are executed without a shell for safety, while commands requiring shell features (pipes, redirections, command chaining) are automatically detected and executed through the shell.

## Tool Declaration

The plugin exposes a single tool:

| Tool | Description |
|------|-------------|
| `cli_based_tool` | Execute a local CLI command |

### Parameters

```json
{
  "command": "Full command string (e.g., 'git status', 'ls -la')",
  "args": ["Optional", "array", "of", "arguments"]
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `command` | string | Yes | Full command string or executable name |
| `args` | array | No | Additional arguments (if `command` is just the executable) |

### Response

```json
{
  "stdout": "Command standard output",
  "stderr": "Command standard error",
  "returncode": 0
}
```

On error:
```json
{
  "error": "Error message",
  "hint": "Optional hint for resolution"
}
```

## Usage

### Basic Setup

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover()
registry.expose_all()  # CLI plugin is exposed by default
```

### With Extra Paths

```python
registry.expose_all({
    "cli": {"extra_paths": ["/usr/local/bin", "/opt/custom/bin"]}
})
```

The `extra_paths` configuration adds directories to the PATH environment variable when resolving and executing commands.

### With JaatoClient

```python
from shared import JaatoClient, PluginRegistry

client = JaatoClient()
client.connect(project_id, location, model_name)

registry = PluginRegistry()
registry.discover()
registry.expose_all()

client.configure_tools(registry)
response = client.send_message("List files in the current directory")
```

## Command Execution

### How Commands Are Parsed

1. **Full command string** (recommended):
   ```python
   cli_based_tool(command="git status")
   cli_based_tool(command="ls -la /tmp")
   ```

2. **Executable + args**:
   ```python
   cli_based_tool(command="git", args=["status"])
   cli_based_tool(command="ls", args=["-la", "/tmp"])
   ```

### Executable Resolution

The plugin uses `shutil.which()` to resolve executables via PATH:
- Finds executables in standard PATH directories
- Supports Windows PATH resolution (.exe, .bat, .cmd)
- Respects `extra_paths` configuration

If an executable is not found:
```json
{
  "error": "cli_based_tool: executable 'foo' not found in PATH",
  "hint": "Configure extra_paths or provide full path to the executable."
}
```

### Shell Commands

The plugin automatically detects when a command requires shell interpretation and switches to shell mode. This happens when the command contains:

| Shell Feature | Example | Description |
|---------------|---------|-------------|
| Pipes | `ls \| grep foo` | Pass output between commands |
| Redirections | `echo hello > file.txt` | Redirect input/output |
| Command chaining | `cd /tmp && ls` | Run commands in sequence |
| OR chaining | `cmd1 \|\| cmd2` | Run cmd2 if cmd1 fails |
| Semicolon | `echo a; echo b` | Run commands sequentially |
| Command substitution | `echo $(date)` | Embed command output |
| Background | `sleep 10 &` | Run in background |

**Examples:**
```python
# Find oldest file (uses pipe)
cli_based_tool(command="ls -t | tail -1")

# Filter output (uses pipe)
cli_based_tool(command="ls -la | grep '.py'")

# Chain commands (uses &&)
cli_based_tool(command="cd /tmp && ls -la")

# Redirect to file (uses >)
cli_based_tool(command="echo 'hello' > /tmp/test.txt")
```

## Security Considerations

1. **Shell auto-detection**: Simple commands run without shell (safer), shell is only used when required for pipes/redirections
2. **No automatic approval**: Plugin returns empty `get_auto_approved_tools()` - all executions require permission
3. **PATH isolation**: Only configured paths are searched for executables
4. **Permission plugin integration**: Use the permission plugin to whitelist/blacklist specific commands

### Recommended: Use with Permission Plugin

```python
from shared.plugins.permission import PermissionPlugin

permission_plugin = PermissionPlugin()
permission_plugin.initialize({
    "policy": {
        "defaultPolicy": "ask",
        "blacklist": {"patterns": ["rm -rf *", "sudo *"]},
        "whitelist": {"patterns": ["git *", "ls *"]}
    }
})

client.configure_tools(registry, permission_plugin)
```

## Auto-Backgrounding

Commands that exceed a configurable threshold (default: 10 seconds) are automatically moved to background execution. This prevents long-running operations from blocking the model.

### How It Works

When a command exceeds the threshold, instead of the normal response, you receive:

```json
{
  "auto_backgrounded": true,
  "task_id": "abc-123",
  "threshold_seconds": 10.0,
  "message": "Task exceeded 10.0s threshold, continuing in background..."
}
```

### Known Slow Commands

The plugin estimates duration for common slow operations:

| Command Pattern | Estimated Duration |
|-----------------|-------------------|
| `mvn install` | 60s |
| `gradle build` | 45s |
| `cargo build` | 60s |
| `npm install` | 30s |
| `pip install` | 20s |
| `pytest` | 30s |
| `docker build` | 60s |

### Monitoring Background Tasks

Use the background plugin's tools to monitor auto-backgrounded commands:

```python
# List all running background tasks
listBackgroundTasks()

# Check status of a specific task
getBackgroundTaskStatus(task_id="abc-123")
# Returns: {"status": "RUNNING", "plugin": "cli", "tool": "cli_based_tool", ...}

# Get result once complete
getBackgroundTaskResult(task_id="abc-123")
# Returns: {"stdout": "...", "stderr": "...", "returncode": 0}
```

### Example Workflow

```
1. Model: cli_based_tool(command="mvn clean install")
2. Response: {"auto_backgrounded": true, "task_id": "xyz-789", ...}
3. Model: getBackgroundTaskStatus(task_id="xyz-789")
4. Response: {"status": "RUNNING", ...}
5. (Model does other work or waits)
6. Model: getBackgroundTaskStatus(task_id="xyz-789")
7. Response: {"status": "COMPLETED", ...}
8. Model: getBackgroundTaskResult(task_id="xyz-789")
9. Response: {"stdout": "BUILD SUCCESS...", "returncode": 0}
```

### Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_background_threshold` | float | `10.0` | Seconds before auto-backgrounding |
| `background_max_workers` | int | `4` | Max concurrent background tasks |

```python
registry.expose_all({
    "cli": {
        "auto_background_threshold": 30.0,  # Increase threshold to 30s
        "background_max_workers": 8
    }
})
```

## System Instructions

The plugin provides comprehensive system instructions to the model covering:
- Filesystem operations (create, read, write, delete, move files)
- Search and filtering (find, grep, pipes)
- Version control (git commands)
- Running programs and tests
- **Auto-backgrounding behavior** and how to monitor background tasks
- Error handling guidance

See `get_system_instructions()` in `plugin.py` for the full text.

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `extra_paths` | list[str] | `[]` | Additional directories to add to PATH |
| `max_output_chars` | int | `50000` | Maximum characters to return from stdout/stderr |
| `auto_background_threshold` | float | `10.0` | Seconds before auto-backgrounding |
| `background_max_workers` | int | `4` | Max concurrent background tasks |
