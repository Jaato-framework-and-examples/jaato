# CLI Tool Plugin

The CLI plugin provides the `cli_based_tool` function for executing local shell commands in the jaato framework.

## Demo

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

Shell features like pipes and redirections are supported:
- Filter output: cli_based_tool(command="ls -la | grep '.py'")
- Chain commands: cli_based_tool(command="cd /tmp && ls -la")
- Redirect output: cli_based_tool(command="echo 'hello' > /tmp/test.txt")
- Find oldest file: cli_based_tool(command="ls -t | tail -1")

The tool returns stdout, stderr, and returncode from the executed command.
```

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `extra_paths` | list[str] | `[]` | Additional directories to add to PATH |
