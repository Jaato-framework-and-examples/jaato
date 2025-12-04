# JAATO Framework Architecture

This document describes the architecture of the jaato framework and how a generic client uses it.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GENERIC CLIENT APPLICATION                         │
│  (Your code that wants to use AI with tools)                                │
└─────────────────────────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JaatoClient                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Public API:                                                         │    │
│  │  • connect(project, location, model)                                │    │
│  │  • configure_tools(registry, permission_plugin, ledger)             │    │
│  │  • send_message(message) → response                                 │    │
│  │  • get_history() / reset_session()                                  │    │
│  │  • get_context_usage()                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                              │                               │
│              ┌───────────────────────────────┼───────────────────────────┐   │
│              │                               │                           │   │
│              ▼                               ▼                           ▼   │
│  ┌───────────────────┐       ┌───────────────────┐       ┌──────────────┐   │
│  │   ToolExecutor    │       │  PluginRegistry   │       │ TokenLedger  │   │
│  │  (Runs functions) │◄──────│  (Manages tools)  │       │ (Accounting) │   │
│  └─────────┬─────────┘       └───────────────────┘       └──────────────┘   │
│            │                          │                                      │
│            ▼                          ▼                                      │
│  ┌───────────────────┐       ┌───────────────────┐                          │
│  │PermissionPlugin   │       │   Tool Plugins    │                          │
│  │ (Access control)  │       │ (CLI, MCP, etc.)  │                          │
│  └───────────────────┘       └───────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────┐
                               │     Vertex AI / Gemini   │
                               │   (LLM with function     │
                               │    calling support)      │
                               └──────────────────────────┘
```

## Component Diagram (Mermaid)

```mermaid
flowchart TB
    subgraph Client["Generic Client Application"]
        App[Your Application]
    end

    subgraph Framework["JAATO Framework (shared/)"]
        JC[JaatoClient<br/>jaato_client.py]

        subgraph Core["Core Components"]
            TE[ToolExecutor<br/>ai_tool_runner.py]
            TL[TokenLedger<br/>token_accounting.py]
        end

        subgraph Plugins["Plugin System"]
            PR[PluginRegistry<br/>plugins/registry.py]

            subgraph Available["Available Plugins"]
                CLI[CLIToolPlugin<br/>plugins/cli/]
                MCP[MCPToolPlugin<br/>plugins/mcp/]
                PERM[PermissionPlugin<br/>plugins/permission/]
            end
        end

        MCM[MCPClientManager<br/>mcp_context_manager.py]
    end

    subgraph External["External Services"]
        VAI[Vertex AI<br/>Gemini Models]
        MCPS[MCP Servers<br/>.mcp.json config]
        Shell[Local Shell<br/>CLI commands]
    end

    App --> JC
    JC --> TE
    JC --> TL
    JC --> PR
    PR --> CLI
    PR --> MCP
    PR --> PERM
    TE --> PERM
    MCP --> MCM
    MCM --> MCPS
    CLI --> Shell
    JC --> VAI
```

## Message Flow Sequence

```mermaid
sequenceDiagram
    participant App as Generic Client
    participant JC as JaatoClient
    participant SDK as Vertex AI SDK
    participant TE as ToolExecutor
    participant Plugin as Tool Plugin
    participant TL as TokenLedger

    App->>JC: send_message("List files")
    JC->>SDK: chat.send_message()
    SDK->>SDK: Model generates response

    alt Response has function_calls
        SDK-->>JC: function_call: cli_based_tool
        JC->>TE: execute("cli_based_tool", args)
        TE->>Plugin: call registered executor
        Plugin-->>TE: result
        TE-->>JC: function response
        JC->>SDK: send function response
        SDK->>SDK: Model processes result
    end

    SDK-->>JC: final text response
    JC->>TL: record token usage
    JC-->>App: "Here are the files: ..."
```

## Plugin System Architecture

```mermaid
classDiagram
    class ToolPlugin {
        <<Protocol>>
        +name: str
        +get_function_declarations() List
        +get_executors() Dict
        +initialize(config)
        +shutdown()
        +get_system_instructions() str
        +get_auto_approved_tools() List
    }

    class PluginRegistry {
        -_available: Dict
        -_exposed: Dict
        +discover()
        +expose_tool(name, config)
        +unexpose_tool(name)
        +get_exposed_declarations()
        +get_exposed_executors()
    }

    class CLIToolPlugin {
        +name = "cli"
        -extra_paths: List
        +cli_based_tool(command)
    }

    class MCPToolPlugin {
        +name = "mcp"
        -manager: MCPClientManager
        +call_mcp_tool(server, tool, args)
    }

    class PermissionPlugin {
        +name = "permission"
        -policy: PermissionPolicy
        -actor: PermissionActor
        +check_permission(tool, args)
        +askPermission(tool)
    }

    ToolPlugin <|.. CLIToolPlugin
    ToolPlugin <|.. MCPToolPlugin
    ToolPlugin <|.. PermissionPlugin
    PluginRegistry o-- ToolPlugin
```

## Typical Client Usage

```python
from shared import JaatoClient, PluginRegistry, TokenLedger

# 1. Create and configure registry
registry = PluginRegistry()
registry.discover()
registry.expose_tool('cli')
registry.expose_tool('mcp')

# 2. Create client and connect
client = JaatoClient()
client.connect(
    project_id='my-gcp-project',
    location='us-central1',
    model='gemini-2.0-flash'
)

# 3. Configure tools
ledger = TokenLedger()
client.configure_tools(registry, ledger=ledger)

# 4. Have a conversation (history managed automatically)
response1 = client.send_message("What's in my current directory?")
response2 = client.send_message("Show me the git log")

# 5. Check usage
usage = client.get_context_usage()
print(f"Context: {usage['percent_used']:.1f}% used")
```

## Data Flow Summary

```
┌────────────────┐
│ Generic Client │
└───────┬────────┘
        │ send_message()
        ▼
┌───────────────────────────────────────────────────────────────┐
│                        JaatoClient                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Internal Function Call Loop                 │  │
│  │                                                          │  │
│  │   ┌──────────┐    ┌──────────────┐    ┌─────────────┐   │  │
│  │   │  Model   │───▶│ Function     │───▶│ ToolExecutor│   │  │
│  │   │ Response │    │ Call Detected│    │   .execute()│   │  │
│  │   └──────────┘    └──────────────┘    └──────┬──────┘   │  │
│  │        ▲                                      │          │  │
│  │        │              ┌───────────────────────┘          │  │
│  │        │              ▼                                  │  │
│  │   ┌────┴─────┐   ┌──────────────┐                       │  │
│  │   │  Model   │◀──│   Function   │                       │  │
│  │   │ Continue │   │   Response   │                       │  │
│  │   └──────────┘   └──────────────┘                       │  │
│  │                                                          │  │
│  │   Loop until model returns text without function calls   │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────┐
│ Final Response │
└────────────────┘
```

## File Structure

```
shared/
├── __init__.py              # Package exports
├── jaato_client.py          # Main client (entry point)
├── ai_tool_runner.py        # ToolExecutor & function loop
├── token_accounting.py      # TokenLedger for usage tracking
├── mcp_context_manager.py   # MCP server connections
└── plugins/
    ├── __init__.py
    ├── base.py              # ToolPlugin protocol
    ├── registry.py          # PluginRegistry
    ├── cli/
    │   └── plugin.py        # CLIToolPlugin
    ├── mcp/
    │   └── plugin.py        # MCPToolPlugin
    └── permission/
        └── plugin.py        # PermissionPlugin
```
