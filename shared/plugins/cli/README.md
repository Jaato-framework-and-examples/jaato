# CLI Tool Plugin

The CLI plugin provides the `cli_based_tool` function for executing local shell commands in the jaato framework.

## Overview

This plugin allows models to run shell commands on the local machine via subprocess. Commands are executed without a shell (using `subprocess.run` with a list), which ensures proper argument handling and avoids shell injection vulnerabilities.

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
registry.expose_tool("cli")
```

### With Extra Paths

```python
registry.expose_tool("cli", {
    "extra_paths": ["/usr/local/bin", "/opt/custom/bin"]
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
registry.expose_tool("cli")

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

## Security Considerations

1. **No shell execution**: Commands run via subprocess list, not shell string
2. **No automatic approval**: Plugin returns empty `get_auto_approved_tools()` - all executions require permission
3. **PATH isolation**: Only configured paths are searched for executables

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

## System Instructions

The plugin provides these system instructions to the model:

```
You have access to `cli_based_tool` which executes local shell commands.

Use it to run commands like `ls`, `cat`, `grep`, `find`, `git`, `gh`, etc.

Example usage:
- List files: cli_based_tool(command="ls -la")
- Read a file: cli_based_tool(command="cat /path/to/file")
- Check git status: cli_based_tool(command="git status")
- Search for text: cli_based_tool(command="grep -r 'pattern' /path")

The tool returns stdout, stderr, and returncode from the executed command.
```

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `extra_paths` | list[str] | `[]` | Additional directories to add to PATH |
