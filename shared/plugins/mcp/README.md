# MCP Tool Plugin

The MCP plugin provides integration with Model Context Protocol (MCP) servers, allowing models to use tools exposed by external MCP-compliant services.

## Overview

This plugin connects to MCP servers defined in `.mcp.json` configuration files and automatically discovers and exposes their tools to the AI model. It runs a background thread with an asyncio event loop to handle the async MCP protocol.

## Configuration

MCP servers are configured in `.mcp.json` files. The plugin searches for configuration in:
1. `.mcp.json` in the current working directory
2. `~/.mcp.json` in the user's home directory

### Configuration Format

```json
{
  "mcpServers": {
    "ServerName": {
      "type": "stdio",
      "command": "mcp-server-command",
      "args": ["--option", "value"],
      "env": {
        "API_KEY": "${ENV_VAR_NAME}"
      }
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Connection type (currently `stdio` supported) |
| `command` | string | Command to start the MCP server |
| `args` | array | Arguments to pass to the command |
| `env` | object | Environment variables (supports `${VAR}` expansion) |

### Example: GitHub MCP Server

```json
{
  "mcpServers": {
    "GitHub": {
      "type": "stdio",
      "command": "mcp-server-github",
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Example: Atlassian MCP Server

```json
{
  "mcpServers": {
    "Atlassian": {
      "type": "stdio",
      "command": "mcp-atlassian",
      "env": {
        "CONFLUENCE_URL": "${CONFLUENCE_URL}",
        "CONFLUENCE_USERNAME": "${CONFLUENCE_USERNAME}",
        "CONFLUENCE_API_TOKEN": "${CONFLUENCE_API_TOKEN}"
      }
    }
  }
}
```

## Usage

### Basic Setup

```python
from shared.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover()
registry.expose_tool("mcp")
```

### With JaatoClient

```python
from shared import JaatoClient, PluginRegistry

client = JaatoClient()
client.connect(project_id, location, model_name)

registry = PluginRegistry()
registry.discover()
registry.expose_tool("mcp")

client.configure_tools(registry)
response = client.send_message("Search for issues in the repository")
```

## Tool Discovery

When initialized, the plugin:
1. Loads configuration from `.mcp.json`
2. Starts a background thread with an asyncio event loop
3. Connects to each configured MCP server
4. Discovers available tools from each server
5. Caches tool declarations for use with the model

Tools are automatically exposed with their original names and descriptions from the MCP server.

## Response Format

MCP tool responses are returned in this format:

```json
{
  "result": {
    "tool": "tool_name",
    "isError": false,
    "structured": null,
    "content": ["Tool output text"]
  }
}
```

On error:
```json
{
  "error": "Error message"
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCPToolPlugin                         │
│                                                         │
│  ┌─────────────┐      ┌──────────────────────────────┐ │
│  │ Main Thread │      │     Background Thread         │ │
│  │             │      │                              │ │
│  │ _execute()  │─────▶│  asyncio event loop          │ │
│  │             │ queue│  └─ MCPClientManager         │ │
│  │             │◀─────│      └─ MCP Server 1         │ │
│  │             │      │      └─ MCP Server 2         │ │
│  └─────────────┘      │      └─ ...                  │ │
│                       └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

The plugin uses:
- A background thread running an asyncio event loop
- Request/response queues for cross-thread communication
- `MCPClientManager` for managing MCP server connections

## Schema Cleaning

MCP tool schemas may contain JSON Schema fields not supported by Vertex AI. The plugin automatically removes:
- `$schema`
- `$id`
- `$ref`
- `$defs`
- `definitions`
- `additionalItems`

## Security Considerations

1. **No automatic approval**: Plugin returns empty `get_auto_approved_tools()` - all executions require permission
2. **Environment variable isolation**: Server credentials are passed via environment variables
3. **Connection timeouts**: Background thread has a 10-second timeout for tool discovery

### Recommended: Use with Permission Plugin

```python
from shared.plugins.permission import PermissionPlugin

permission_plugin = PermissionPlugin()
permission_plugin.initialize({
    "policy": {
        "defaultPolicy": "ask"
    }
})

client.configure_tools(registry, permission_plugin)
```

## System Instructions

The plugin generates system instructions listing all discovered tools:

```
You have access to the following MCP (Model Context Protocol) tools:

From 'GitHub' server:
  - search_issues: Search for issues in a repository
  - get_issue: Get details of a specific issue
  - create_issue: Create a new issue

From 'Atlassian' server:
  - confluence_get_page: Retrieve a Confluence page
  - confluence_search: Search Confluence content
```

## Troubleshooting

### No tools discovered

1. Check that `.mcp.json` exists and is valid JSON
2. Verify the MCP server command is installed and in PATH
3. Check environment variables are set correctly
4. Look for connection errors in stderr output

### Tool execution timeout

The plugin has a 30-second timeout for tool execution. If tools consistently timeout:
1. Check MCP server responsiveness
2. Consider increasing timeout in the plugin code
3. Verify network connectivity for remote MCP servers

### JSON-RPC validation errors

The plugin includes a patch to filter non-JSON-RPC messages from MCP server output. If you see validation errors:
1. Check MCP server isn't outputting debug/log messages to stdout
2. Update to the latest MCP server version
