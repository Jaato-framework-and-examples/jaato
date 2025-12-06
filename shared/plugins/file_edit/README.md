# File Edit Plugin

The file_edit plugin provides tools for reading, modifying, and managing files with integrated permission approval (showing diffs) and automatic backups.

## Overview

This plugin enables the model to perform file operations with safety features:
- **Diff preview**: File modifications show a unified diff for approval before execution
- **Automatic backups**: Updates and deletions create backups that can be restored
- **Colorized display**: Console approval shows colorized diffs (green for additions, red for deletions)

## Tool Declarations

The plugin exposes five tools:

| Tool | Description | Auto-approved |
|------|-------------|---------------|
| `readFile` | Read file contents | Yes |
| `updateFile` | Update an existing file | No (shows diff) |
| `writeNewFile` | Create a new file | No (shows content) |
| `removeFile` | Delete a file | No (shows confirmation) |
| `undoFileChange` | Restore from backup | Yes |

### readFile

Read the contents of a file.

**Parameters:**
```json
{
  "path": "Path to the file to read"
}
```

**Response:**
```json
{
  "path": "/path/to/file.txt",
  "content": "File contents...",
  "size": 1234,
  "lines": 50
}
```

### updateFile

Update an existing file with new content. Shows a diff for approval and creates a backup before modifying.

**Parameters:**
```json
{
  "path": "Path to the file to update",
  "new_content": "The new content to write to the file"
}
```

**Response:**
```json
{
  "success": true,
  "path": "/path/to/file.txt",
  "size": 1234,
  "lines": 50,
  "backup": ".jaato/backups/_path_to_file.txt_2025-12-06T14-30-00.bak"
}
```

### writeNewFile

Create a new file. Shows the content for approval. Fails if the file already exists.

**Parameters:**
```json
{
  "path": "Path where the new file should be created",
  "content": "Content to write to the new file"
}
```

**Response:**
```json
{
  "success": true,
  "path": "/path/to/newfile.txt",
  "size": 500,
  "lines": 20
}
```

### removeFile

Delete a file. Creates a backup before deletion so it can be restored.

**Parameters:**
```json
{
  "path": "Path to the file to delete"
}
```

**Response:**
```json
{
  "success": true,
  "path": "/path/to/file.txt",
  "deleted": true,
  "backup": ".jaato/backups/_path_to_file.txt_2025-12-06T14-30-00.bak"
}
```

### undoFileChange

Restore a file from its most recent backup.

**Parameters:**
```json
{
  "path": "Path to the file to restore"
}
```

**Response:**
```json
{
  "success": true,
  "path": "/path/to/file.txt",
  "restored_from": ".jaato/backups/_path_to_file.txt_2025-12-06T14-30-00.bak",
  "message": "File restored from backup"
}
```

## Usage

### Basic Setup

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover()
registry.expose_all()  # file_edit plugin is exposed by default
```

### With Custom Backup Directory

```python
registry.expose_all({
    "file_edit": {"backup_dir": "/custom/backup/path"}
})
```

### With JaatoClient

```python
from shared import JaatoClient, PluginRegistry
from shared.plugins.permission import PermissionPlugin

client = JaatoClient()
client.connect(project_id, location, model_name)

registry = PluginRegistry()
registry.discover()
registry.expose_all()

# Important: Set registry on permission plugin for diff display
permission_plugin = PermissionPlugin()
permission_plugin.initialize()
permission_plugin.set_registry(registry)

client.configure_tools(registry, permission_plugin)
response = client.send_message("Update config.json to add a new setting")
```

## Permission Integration

The file_edit plugin integrates with the permission system to show formatted diffs when requesting approval:

```
============================================================
[askPermission] Main agent requesting tool execution:
  Update file: src/config.py (+5, -2 lines)

--- a/src/config.py
+++ b/src/config.py
@@ -10,7 +10,10 @@
 DEFAULT_TIMEOUT = 30
-MAX_RETRIES = 3
+MAX_RETRIES = 5
+ENABLE_CACHE = True
+CACHE_TTL = 3600

============================================================

Options: [y]es, [n]o, [a]lways, [never], [once], [all]
```

The plugin implements the optional `format_permission_request()` method to provide custom display formatting.

## Backup System

### Backup Location

Backups are stored in `.jaato/backups/` with the naming convention:
```
{path_with_underscores}_{ISO_timestamp}.bak
```

Example:
```
.jaato/backups/
├── _home_user_project_src_main.py_2025-12-06T14-30-00.bak
├── _home_user_project_src_main.py_2025-12-06T14-35-22.bak
└── _home_user_project_config.json_2025-12-06T14-32-11.bak
```

### Backup Retention

The number of backups kept per file is controlled by the `JAATO_FILE_BACKUP_COUNT` environment variable (default: 5). When a new backup is created, old backups exceeding this limit are automatically pruned.

### Gitignore Integration

On initialization, the plugin automatically adds `.jaato` to `.gitignore` if the file exists and the entry is not already present.

## System Instructions

The plugin provides these system instructions to the model:

```
You have access to file editing tools:

- `readFile(path)`: Read file contents. Safe operation, no approval needed.
- `updateFile(path, new_content)`: Update an existing file. Shows diff for approval and creates backup.
- `writeNewFile(path, content)`: Create a new file. Shows content for approval. Fails if file exists.
- `removeFile(path)`: Delete a file. Creates backup before deletion.
- `undoFileChange(path)`: Restore a file from its most recent backup.

File modifications (updateFile, writeNewFile, removeFile) will show you a preview
and require approval before execution. Backups are automatically created for
updateFile and removeFile operations.
```

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backup_dir` | str | `.jaato/backups` | Directory for storing backups |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JAATO_FILE_BACKUP_COUNT` | 5 | Maximum number of backups to keep per file |
