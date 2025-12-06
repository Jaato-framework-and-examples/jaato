# Slash Command Plugin

The slash command plugin enables users to invoke commands from `.jaato/commands/` directory using `/command_name [args...]` syntax. When users type a slash command, it's sent to the model which calls the `processCommand` tool to read the command file, substitute parameters, and follow the instructions.

## How It Works

1. User types `/command_name arg1 arg2` (e.g., `/summarize file.py`)
2. The interactive client provides autocomplete for available commands
3. The command with arguments is sent to the model
4. The model recognizes the `/` prefix and calls `processCommand(command_name="summarize", args=["file.py"])`
5. The tool reads the command file and substitutes `{{$1}}` with `file.py`
6. The model follows the instructions with the substituted content

## Setup

Create a `.jaato/commands/` directory in your project root and add command files:

```bash
mkdir -p .jaato/commands
```

### Command File Format

Command files are plain text files. The first line is used as the description shown in autocomplete. The rest of the file contains instructions for the model, optionally with parameter placeholders.

#### Basic Command (no parameters)

`.jaato/commands/summarize`:
```
# Summarize the conversation
Please provide a concise summary of our conversation so far, highlighting:
1. Key topics discussed
2. Decisions made
3. Outstanding questions or tasks
```

#### Command with Parameters

`.jaato/commands/review`:
```
# Review a file for issues
Please review the file {{$1}} and check for:
- Security vulnerabilities
- Performance issues
- Code style and best practices
- Potential bugs
```

Usage: `/review src/main.py`

#### Command with Default Values

`.jaato/commands/explain`:
```
# Explain code at a given detail level
Please explain the code in {{$1}} at a {{$2:beginner}} level.
Focus on the main concepts and how it works.
```

Usage:
- `/explain utils.py` → explains at "beginner" level (default)
- `/explain utils.py expert` → explains at "expert" level

#### Using All Arguments

`.jaato/commands/compare`:
```
# Compare multiple files
Please compare the following files and highlight their differences:
{{$0}}

Focus on structural differences and different approaches used.
```

Usage: `/compare file1.py file2.py file3.py`
- `{{$0}}` expands to `file1.py file2.py file3.py`

## Template Syntax

| Syntax | Description | Example |
|--------|-------------|---------|
| `{{$1}}` | First positional argument | `/cmd foo` → `foo` |
| `{{$2}}` | Second positional argument | `/cmd foo bar` → `bar` |
| `{{$N}}` | Nth positional argument | Any position |
| `{{$1:default}}` | With default if not provided | `/cmd` → `default` |
| `{{$0}}` | All arguments joined with spaces | `/cmd a b c` → `a b c` |

## Usage

In the interactive client:

```
You> /summarize
You> /review src/main.py
You> /explain utils.py expert
You> /compare old.py new.py
```

The model will read the command file, substitute any parameters, and follow the instructions.

### Autocomplete

Type `/` and available commands will appear in the autocomplete dropdown with their descriptions (from the first line of each file).

## Model Tool

The plugin exposes one tool:

### processCommand

Reads a command file from the `.jaato/commands/` directory and substitutes parameters.

**Parameters:**
- `command_name` (string, required): Name of the command file to read (without leading `/`)
- `args` (array of strings, optional): Positional arguments to substitute into the template

**Returns:**
- `command_name`: The name of the command
- `content`: The command file contents with parameters substituted
- `file_path`: Absolute path to the command file
- `size`: File size in bytes
- `args_provided`: The arguments that were passed (if any)
- `missing_parameters`: List of required parameters that were not provided (if any)

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
- Arguments are substituted as-is (no shell interpretation)
