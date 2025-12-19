# Filesystem Query Plugin

The filesystem query plugin provides read-only tools for exploring the filesystem in the jaato framework. It offers fast, safe, auto-approved tools for finding files and searching content.

## Overview

This plugin enables models to be self-sufficient when locating files without requiring users to provide exact paths. It provides two complementary tools:

| Tool | Description |
|------|-------------|
| `glob_files` | Find files matching glob patterns (e.g., `**/*.py`, `src/**/*.ts`) |
| `grep_content` | Search file contents using regular expressions |

Both tools:
- Return structured JSON output
- Support configurable exclusion patterns
- Auto-approve without permission prompts (read-only operations)
- Support background execution for large searches

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FilesystemQueryPlugin                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    BackgroundCapableMixin                          │  │
│  │  • Thread pool for long-running searches                          │  │
│  │  • Auto-background after timeout threshold                        │  │
│  │  • Task status tracking                                           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────┐    ┌─────────────────────────────────────────┐ │
│  │   glob_files        │    │   grep_content                          │ │
│  │   ────────────      │    │   ────────────                          │ │
│  │   • Glob patterns   │    │   • Regex search                        │ │
│  │   • File metadata   │    │   • Context lines                       │ │
│  │   • Result limits   │    │   • File type filtering                 │ │
│  └─────────────────────┘    └─────────────────────────────────────────┘ │
│                                      │                                   │
│                                      ▼                                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   FilesystemQueryConfig                           │  │
│  │  • exclude_patterns (extend/replace defaults)                     │  │
│  │  • include_patterns (force-include overrides)                     │  │
│  │  • max_results, max_file_size_kb, timeout_seconds                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │     Configuration Sources       │
                    │  ───────────────────────────── │
                    │  1. Runtime config (highest)    │
                    │  2. .jaato/filesystem_query.json│
                    │  3. Environment variable        │
                    │  4. Hardcoded defaults (lowest) │
                    └─────────────────────────────────┘
```

## Tool Declarations

### glob_files

Find files matching a glob pattern.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pattern` | string | Yes | Glob pattern (e.g., `**/*.py`, `src/**/*.ts`) |
| `root` | string | No | Root directory (default: cwd) |
| `max_results` | integer | No | Maximum files to return (default: 500) |
| `include_hidden` | boolean | No | Include hidden files (default: false) |

**Response:**

```json
{
  "files": [
    {
      "path": "src/main.py",
      "absolute_path": "/home/user/project/src/main.py",
      "size": 1234,
      "modified": "2025-01-15T10:30:00"
    }
  ],
  "total": 42,
  "returned": 42,
  "truncated": false,
  "root": "/home/user/project",
  "pattern": "**/*.py"
}
```

### grep_content

Search file contents using a regular expression.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pattern` | string | Yes | Regex pattern to search for |
| `path` | string | No | File or directory to search (default: cwd) |
| `file_glob` | string | No | Only search files matching this glob |
| `context_lines` | integer | No | Lines of context (default: 2) |
| `case_sensitive` | boolean | No | Case-sensitive search (default: true) |
| `max_results` | integer | No | Maximum matches (default: 500) |

**Response:**

```json
{
  "matches": [
    {
      "file": "src/main.py",
      "absolute_path": "/home/user/project/src/main.py",
      "line": 42,
      "column": 8,
      "text": "    def hello():",
      "match": "def hello",
      "context_before": ["", "class Greeter:"],
      "context_after": ["        print('hi')"]
    }
  ],
  "total_matches": 15,
  "files_with_matches": 3,
  "files_searched": 200,
  "truncated": false,
  "pattern": "def\\s+hello",
  "path": "/home/user/project",
  "file_glob": "*.py"
}
```

## Usage

### Basic Setup

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover()
registry.expose_tool("filesystem_query")
```

### With Configuration

```python
registry.expose_tool("filesystem_query", config={
    "exclude_patterns": ["vendor", "third_party"],
    "include_patterns": ["node_modules/@company"],
    "max_results": 1000,
})
```

### With JaatoClient

```python
from shared import JaatoClient, PluginRegistry

client = JaatoClient()
client.connect(project_id, location, model_name)

registry = PluginRegistry()
registry.discover()
registry.expose_tool("filesystem_query")

client.configure_tools(registry)
response = client.send_message("Find all Python test files in this project")
```

## Configuration

### Configuration File

Create `.jaato/filesystem_query.json`:

```json
{
  "version": "1.0",
  "exclude_patterns": ["vendor", "third_party"],
  "exclude_mode": "extend",
  "include_patterns": ["node_modules/@mycompany"],
  "max_results": 500,
  "max_file_size_kb": 1024,
  "timeout_seconds": 30,
  "context_lines": 2
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `exclude_patterns` | list | `[]` | Additional patterns to exclude |
| `exclude_mode` | string | `"extend"` | `"extend"` adds to defaults, `"replace"` overrides |
| `include_patterns` | list | `[]` | Force-include patterns (override exclusions) |
| `max_results` | integer | `500` | Maximum files/matches returned |
| `max_file_size_kb` | integer | `1024` | Skip files larger than this (KB) |
| `timeout_seconds` | integer | `30` | Auto-background threshold |
| `context_lines` | integer | `2` | Default context lines for grep |

### Default Exclusions

The following patterns are excluded by default:

```
.git, .svn, .hg, .bzr                    # Version control
node_modules, bower_components           # JavaScript
__pycache__, .venv, venv, .tox          # Python
.mypy_cache, .pytest_cache, .ruff_cache  # Python tools
dist, build, out, target, _build         # Build artifacts
.idea, .vscode, .eclipse                 # IDEs
*.min.js, *.min.css                      # Minified files
package-lock.json, yarn.lock, etc.       # Lock files
```

Use `exclude_mode: "replace"` to disable all defaults, or `include_patterns` to force-include specific paths.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FILESYSTEM_QUERY_CONFIG_PATH` | Path to config file |

## Auto-Approval

Both `glob_files` and `grep_content` are automatically approved and do not require user permission since they are read-only operations that pose no security risk to the local system.

## Background Execution

Large searches may take time. The plugin supports background execution:

- Auto-backgrounds after `timeout_seconds` threshold (default: 30s)
- Returns a task handle for status checking
- Uses thread pool with configurable workers

## System Instructions

The plugin provides these system instructions to the model:

```
You have access to filesystem query tools for exploring the codebase:

## glob_files
Find files by name pattern. Use glob syntax:
- `**/*.py` - all Python files recursively
- `src/**/*.ts` - TypeScript files in src/
- `**/test_*.py` - test files anywhere
- `*.json` - JSON files in current directory

## grep_content
Search file contents with regex:
- `def\\s+function_name` - find function definitions
- `class\\s+ClassName` - find class definitions
- `import\\s+module` - find imports
- `TODO|FIXME|HACK` - find code comments

Tips:
- Use glob_files first to locate files, then grep_content to search within them
- Both tools respect .gitignore-style exclusions (node_modules, __pycache__, etc.)
- Use file_glob parameter in grep_content to limit search to specific file types
- Results are limited to prevent overwhelming output; adjust max_results if needed
```

## Examples

### Find all Python files

```python
# Model calls:
glob_files(pattern="**/*.py")

# Returns files sorted by modification time (newest first)
```

### Search for function definitions

```python
# Model calls:
grep_content(
    pattern="def\\s+process_",
    file_glob="*.py",
    context_lines=3
)

# Returns matches with surrounding context
```

### Find configuration files

```python
# Model calls:
glob_files(pattern="**/*.{json,yaml,yml,toml}")
```

### Search with case-insensitive matching

```python
# Model calls:
grep_content(
    pattern="error|exception|fail",
    case_sensitive=False,
    file_glob="*.log"
)
```
