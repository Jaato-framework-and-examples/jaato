# References Plugin

The References plugin manages documentation and reference source injection into the model's context. It maintains a catalog of reference sources and handles user selection via configurable communication protocols.

## Demo

![References Plugin Demo](demo.svg)

## Overview

Reference sources can be configured in two modes:

- **AUTO**: Automatically included in system instructions at startup. The model is instructed to fetch these sources immediately.
- **SELECTABLE**: Available on-demand. The user chooses which sources to include when the model requests them via the `selectReferences` tool.

The model is responsible for fetching content using existing tools (CLI for local files, MCP tools, URL fetch, etc.). This plugin only manages the catalog metadata and user interaction.

## Configuration

### Config File Location

The plugin searches for the config file in the following order:

1. Path specified via `REFERENCES_CONFIG_PATH` environment variable
2. `./references.json` (current working directory)
3. `./.references.json` (hidden file in current working directory)
4. `~/.config/jaato/references.json` (user config directory)

The first file found is used. If no file is found, the plugin initializes with an empty source list.

### Config File Format

Create a `references.json` file:

```json
{
  "version": "1.0",
  "sources": [
    {
      "id": "api-spec",
      "name": "API Specification",
      "description": "OpenAPI spec for the REST API",
      "type": "local",
      "path": "./docs/openapi.yaml",
      "mode": "auto",
      "tags": ["api", "endpoints", "REST"]
    },
    {
      "id": "auth-guide",
      "name": "Authentication Guide",
      "description": "OAuth2 and JWT authentication documentation",
      "type": "mcp",
      "server": "Confluence",
      "tool": "get_page",
      "args": {"page_id": "12345"},
      "mode": "selectable",
      "tags": ["auth", "oauth", "jwt", "security"]
    },
    {
      "id": "coding-standards",
      "name": "Coding Standards",
      "description": "Team coding conventions and style guide",
      "type": "url",
      "url": "https://wiki.internal/standards.md",
      "mode": "selectable",
      "tags": ["standards", "style", "conventions"]
    },
    {
      "id": "project-rules",
      "name": "Project Rules",
      "description": "Key rules for this project",
      "type": "inline",
      "content": "1. Always use TypeScript\n2. Tests required for all features\n3. No direct database access from controllers",
      "mode": "auto",
      "tags": ["rules"]
    }
  ],
  "actor": {
    "type": "console",
    "timeout": 60
  }
}
```

### Source Types

The `type` field is **required** for each source. It determines which additional fields are needed:

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `local` | Local file on filesystem | `type`, `path` |
| `url` | HTTP/HTTPS URL | `type`, `url` |
| `mcp` | MCP server tool call | `type`, `server`, `tool`, optionally `args` |
| `inline` | Content embedded in config | `type`, `content` |

### Injection Modes

| Mode | Description |
|------|-------------|
| `auto` | Included in system instructions; model fetches at startup |
| `selectable` | User selects when model calls `selectReferences` |

### Actor Configuration

The actor handles user interaction for selecting references. Three protocols are supported:

#### Console Actor (default)
```json
{
  "actor": {
    "type": "console",
    "timeout": 60
  }
}
```

Interactive terminal prompts for reference selection.

#### Webhook Actor
```json
{
  "actor": {
    "type": "webhook",
    "endpoint": "https://approval-service.internal/references",
    "timeout": 300
  }
}
```

Sends selection requests to an HTTP endpoint. Expected response:
```json
{
  "selected_ids": ["api-spec", "auth-guide"]
}
```

#### File Actor
```json
{
  "actor": {
    "type": "file",
    "base_path": "/tmp/jaato-references",
    "timeout": 300
  }
}
```

Writes requests to `{base_path}/requests/{request_id}.json` and polls for responses at `{base_path}/responses/{request_id}.json`.

## Tools Exposed

### selectReferences

Triggers user selection of additional reference sources.

```json
{
  "context": "Need API documentation for implementing the /users endpoint",
  "filter_tags": ["api", "auth"]
}
```

**Parameters:**
- `context` (optional): Explains why references are needed, helping users make better selections
- `filter_tags` (optional): Only show sources with matching tags

**Returns:**
```json
{
  "status": "success",
  "selected_count": 2,
  "message": "The user has selected the following reference sources...",
  "sources": "### API Specification\n*OpenAPI spec...*\n\n### Auth Guide\n..."
}
```

### listReferences

Lists all available reference sources.

```json
{
  "filter_tags": ["api"],
  "mode": "selectable"
}
```

**Parameters:**
- `filter_tags` (optional): Filter by tags
- `mode` (optional): Filter by mode (`all`, `auto`, `selectable`)

**Returns:**
```json
{
  "sources": [
    {
      "id": "api-spec",
      "name": "API Specification",
      "description": "OpenAPI spec for the REST API",
      "type": "local",
      "mode": "auto",
      "tags": ["api", "endpoints"],
      "selected": false,
      "access": "File: ./docs/openapi.yaml"
    }
  ],
  "total": 4,
  "selected_count": 0
}
```

## Tags and Proactive Reference Access

Each reference source can have tags describing its topic. The model is instructed to:

1. Note available tags when the session starts
2. When encountering topics matching these tags during conversation, consider calling `selectReferences` with relevant `filter_tags`
3. Fetch and incorporate the selected references before proceeding

This enables context-aware documentation injection based on the current task.

## Usage

### With Plugin Registry

```python
from shared.plugins import PluginRegistry

registry = PluginRegistry()
registry.discover()
registry.expose_all()  # References plugin is exposed by default

# Get declarations and executors
declarations = registry.get_exposed_declarations()
executors = registry.get_exposed_executors()
instructions = registry.get_system_instructions()
```

### Standalone

```python
from shared.plugins.references import create_plugin

plugin = create_plugin()
plugin.initialize({
    "config_path": "./references.json",
    "actor_type": "console",
})

# Execute tools directly
result = plugin.get_executors()["selectReferences"]({
    "context": "Need auth documentation"
})

# Access sources programmatically
sources = plugin.get_sources()
selected = plugin.get_selected_ids()
```

### Inline Configuration

```python
plugin = create_plugin()
plugin.initialize({
    "sources": [
        {
            "id": "readme",
            "name": "README",
            "description": "Project overview",
            "type": "local",
            "path": "./README.md",
            "mode": "auto",
            "tags": ["overview"]
        }
    ],
    "actor_type": "console",
})
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `REFERENCES_CONFIG_PATH` | Path to references.json file |

## Console Selection Example

When the model calls `selectReferences`, the console actor displays:

```
============================================================
REFERENCE SELECTION
============================================================

Context: Need API documentation for implementing authentication

Available references:

  [1] API Specification
      OpenAPI spec for the REST API
      Type: local | Tags: api, endpoints, REST

  [2] Authentication Guide
      OAuth2 and JWT authentication documentation
      Type: mcp | Tags: auth, oauth, jwt, security

  [3] Coding Standards
      Team coding conventions and style guide
      Type: url | Tags: standards, style, conventions

Enter selection:
  - Numbers separated by commas (e.g., '1,3,4')
  - 'all' to select all
  - 'none' or empty to skip

> 1,2

Selected 2 reference source(s). Instructions provided to model.
```

## Security Considerations

- This plugin is user-triggered and all tools are auto-approved (no permission prompts)
- The plugin only provides metadata; actual content fetching uses existing tools with their own permission controls
- Webhook endpoints should be properly authenticated
- File actor paths should have appropriate filesystem permissions
