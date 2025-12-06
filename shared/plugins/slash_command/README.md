# Slash Command Plugin

The slash command plugin enables users to invoke commands from `.jaato/commands/` directory using `/command_name` syntax. When users type a slash command, it's sent to the model which calls the `processCommand` tool to read and follow the instructions in the command file.

## How It Works

1. User types `/command_name` (e.g., `/summarize`)
2. The interactive client provides autocomplete for available commands
3. The command is sent to the model
4. The model recognizes the `/` prefix and calls `processCommand(command_name="summarize")`
5. The tool reads the command file and returns its contents
6. The model follows the instructions in the command file

## Setup

Create a `.jaato/commands/` directory in your project root and add command files:

```bash
mkdir -p .jaato/commands
```

### Command File Format

Command files are plain text files. The first line is used as the description shown in autocomplete. The rest of the file contains instructions for the model.

Example `.jaato/commands/summarize`:
```
# Summarize the conversation
Please provide a concise summary of our conversation so far, highlighting:
1. Key topics discussed
2. Decisions made
3. Outstanding questions or tasks
```

Example `.jaato/commands/review`:
```
# Review code for issues
Please review any code we've discussed and check for:
- Security vulnerabilities
- Performance issues
- Code style and best practices
- Potential bugs
```

## Usage

In the interactive client:

```
You> /summarize
```

The model will read `.jaato/commands/summarize` and follow its instructions.

### Autocomplete

Type `/` and available commands will appear in the autocomplete dropdown with their descriptions (from the first line of each file).

## Model Tool

The plugin exposes one tool:

### processCommand

Reads a command file from the `.jaato/commands/` directory.

**Parameters:**
- `command_name` (string, required): Name of the command file to read (without leading `/`)

**Returns:**
- `command_name`: The name of the command
- `content`: The full contents of the command file
- `file_path`: Absolute path to the command file
- `size`: File size in bytes

**Errors:**
- Command file not found
- Invalid command name (path traversal attempted)
- File too large (>100KB)
- Commands directory not found

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `commands_dir` | `.jaato/commands` | Path to commands directory |

Example configuration in the interactive client:
```python
plugin_configs = {
    "slash_command": {
        "commands_dir": "/custom/commands/path"
    }
}
registry.expose_all(plugin_configs)
```

## Auto-Approval

The `processCommand` tool is auto-approved by default since it only reads files from a designated directory. No permission prompt is shown when the model calls this tool.

## Security

- Command names are validated to prevent directory traversal (no `..`, `/`, or `\`)
- Only files within the configured commands directory can be read
- Maximum file size is limited to 100KB
