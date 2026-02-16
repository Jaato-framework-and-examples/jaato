# Jaato Architecture - Detailed Sequence Diagrams

This document provides ultra-detailed sequence diagrams showing the complete interaction flow between the rich client, server, sessions, agents, tool plugins, and output rendering pipeline.

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Client-Server Connection Flow](#2-client-server-connection-flow)
3. [Message Processing Flow](#3-message-processing-flow)
4. [Streaming Output Pipeline](#4-streaming-output-pipeline)
5. [Tool Execution Flow](#5-tool-execution-flow)
6. [Permission Request Flow](#6-permission-request-flow)
7. [Multi-Agent / Subagent Flow](#7-multi-agent--subagent-flow)
8. [Session Management Flow](#8-session-management-flow)
9. [Complete End-to-End Flow](#9-complete-end-to-end-flow)
10. [Component Reference](#10-component-reference)

---

## 1. System Overview

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  RichClient (TUI)          │  WebClient (future)  │  IDE Plugin (future) │    │
│  │  jaato-tui/rich_client.py │                      │                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                          ┌───────────┴───────────┐                              │
│                          │   Backend Abstraction │                              │
│                          │   backend.py          │                              │
│                          └───────────┬───────────┘                              │
│                                      │                                           │
│                    ┌─────────────────┴─────────────────┐                        │
│                    ▼                                   ▼                        │
│           ┌────────────────┐                  ┌────────────────┐                │
│           │  DirectBackend │                  │   IPCBackend   │                │
│           │ (embedded mode)│                  │  (daemon mode) │                │
│           └────────────────┘                  └────────┬───────┘                │
└────────────────────────────────────────────────────────│────────────────────────┘
                                                         │
                                            IPC Socket / WebSocket
                                            (/tmp/jaato.sock)
                                                         │
┌────────────────────────────────────────────────────────│────────────────────────┐
│                              TRANSPORT LAYER           │                         │
│  ┌─────────────────────────────────────────────────────┴───────────────────┐    │
│  │                        JaatoIPCServer / JaatoWSServer                    │    │
│  │                        server/ipc.py, server/websocket.py                │    │
│  │  • Length-prefixed framing (4-byte header + JSON)                       │    │
│  │  • Async event queues per client                                         │    │
│  │  • Event serialization/deserialization                                   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SERVER LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          JaatoDaemon                                     │    │
│  │                          server/__main__.py                              │    │
│  │  • Daemon process management (PID file, signals)                        │    │
│  │  • Routes requests to SessionManager                                     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        SessionManager                                    │    │
│  │                        server/session_manager.py                         │    │
│  │  • Multi-session orchestration                                          │    │
│  │  • Client → Session mapping                                              │    │
│  │  • Session persistence via SessionPlugin                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│            ┌─────────────────────────┼─────────────────────────┐                │
│            ▼                         ▼                         ▼                │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐        │
│  │  JaatoServer     │     │  JaatoServer     │     │  JaatoServer     │        │
│  │  (session: main) │     │  (session: dev)  │     │  (session: N)    │        │
│  │  server/core.py  │     │  server/core.py  │     │  server/core.py  │        │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CORE LAYER                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          JaatoClient (Facade)                            │    │
│  │                          shared/jaato_client.py                          │    │
│  │  • Backwards-compatible API                                              │    │
│  │  • Wraps Runtime + Session                                               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                    ┌─────────────────┴─────────────────┐                        │
│                    ▼                                   ▼                        │
│  ┌─────────────────────────────┐       ┌─────────────────────────────┐         │
│  │      JaatoRuntime           │       │      JaatoSession           │         │
│  │   (shared resources)        │◄──────│   (per-agent state)         │         │
│  │   jaato_runtime.py          │       │   jaato_session.py          │         │
│  │                             │       │                             │         │
│  │  • Provider configuration   │       │  • Conversation history     │         │
│  │  • PluginRegistry           │       │  • Tool configuration       │         │
│  │  • PermissionPlugin         │       │  • System instructions      │         │
│  │  • TokenLedger              │       │  • CancelToken (stop)       │         │
│  │  • create_session()         │       │  • ToolExecutor             │         │
│  └─────────────────────────────┘       └─────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PLUGIN LAYER                                        │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐           │
│  │  Tool Plugins     │  │  GC Plugins       │  │  Model Providers  │           │
│  │  • cli/           │  │  • gc_truncate/   │  │  • anthropic/     │           │
│  │  • mcp/           │  │  • gc_summarize/  │  │  • google_genai/  │           │
│  │  • file_edit/     │  │  • gc_hybrid/     │  │  • github_models/ │           │
│  │  • web_search/    │  │                   │  │                   │           │
│  │  • permission/    │  │                   │  │                   │           │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘           │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          ToolExecutor                                    │    │
│  │                          shared/ai_tool_runner.py                        │    │
│  │  • Tool registration and lookup                                          │    │
│  │  • Permission checking                                                   │    │
│  │  • Auto-backgrounding for long tasks                                     │    │
│  │  • Output callback streaming                                             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Client-Server Connection Flow

### Initial Connection Sequence

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant RichClient as RichClient<br/>(rich_client.py)
    participant IPCClient as IPCClient<br/>(ipc_client.py)
    participant Socket as Unix Socket<br/>(/tmp/jaato.sock)
    participant IPCServer as JaatoIPCServer<br/>(ipc.py)
    participant Daemon as JaatoDaemon<br/>(__main__.py)
    participant SessionMgr as SessionManager<br/>(session_manager.py)

    User->>RichClient: Launch CLI<br/>python rich_client.py --connect /tmp/jaato.sock

    RichClient->>IPCClient: IPCClient(socket_path, auto_start=True)

    RichClient->>IPCClient: await connect()

    alt Socket does not exist AND auto_start=True
        IPCClient->>IPCClient: _start_server()
        IPCClient->>Daemon: subprocess: python -m server<br/>--ipc-socket /tmp/jaato.sock --daemon
        Daemon->>Daemon: Write PID file
        Daemon->>IPCServer: start()
        IPCServer->>Socket: Create Unix domain socket
        IPCServer->>IPCServer: asyncio.start_unix_server()
        Note over Daemon: Server listening...
    end

    IPCClient->>Socket: asyncio.open_unix_connection()
    Socket->>IPCServer: Connection accepted

    IPCServer->>IPCServer: Assign client_id = "ipc_1"
    IPCServer->>IPCServer: Create asyncio.Queue for client
    IPCServer->>IPCServer: Start _broadcast_to_client() task

    IPCServer->>IPCClient: ConnectedEvent(client_id="ipc_1")
    Note over IPCClient,IPCServer: 4-byte length prefix + JSON payload

    IPCClient->>RichClient: Connection established

    RichClient->>IPCClient: send_event(ClientConfigRequest)
    IPCClient->>IPCServer: ClientConfigRequest(workspace_path, env)

    IPCServer->>SessionMgr: get_or_create_default(client_id, workspace_path)
    SessionMgr->>SessionMgr: Load or create "main" session
    SessionMgr->>IPCServer: session_id = "main"

    IPCServer->>IPCClient: SessionInfoEvent(session_id="main", ...)

    IPCClient->>RichClient: Ready for input
    RichClient->>User: Display prompt
```

### Message Framing Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                    IPC Message Frame                             │
├─────────────┬───────────────────────────────────────────────────┤
│  4 bytes    │              Variable length                       │
│  (uint32    │              (JSON payload)                        │
│   big-end)  │                                                    │
├─────────────┼───────────────────────────────────────────────────┤
│  Length N   │  {"type": "SendMessageRequest", "text": "...",    │
│             │   "session_id": "main", "attachments": [...]}     │
└─────────────┴───────────────────────────────────────────────────┘

Max message size: 10 MB
```

---

## 3. Message Processing Flow

### Complete Message Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant RichClient as RichClient<br/>(TUI)
    participant IPCClient as IPCClient
    participant IPCServer as JaatoIPCServer
    participant SessionMgr as SessionManager
    participant JaatoServer as JaatoServer<br/>(core.py)
    participant JaatoClient as JaatoClient<br/>(facade)
    participant JaatoSession as JaatoSession
    participant Provider as ModelProviderPlugin<br/>(anthropic/google)
    participant LLM as LLM API

    User->>RichClient: Type message + Enter
    RichClient->>RichClient: Capture input text

    RichClient->>IPCClient: send_message(text, attachments)
    IPCClient->>IPCClient: Create SendMessageRequest
    IPCClient->>IPCServer: serialize_event() → JSON frame

    Note over IPCServer: _handle_message()
    IPCServer->>IPCServer: deserialize_event() → SendMessageRequest
    IPCServer->>SessionMgr: handle_request(client_id, session_id, event)

    SessionMgr->>SessionMgr: Get session for client
    SessionMgr->>JaatoServer: handle_request(SendMessageRequest)

    Note over JaatoServer: Process in thread pool
    JaatoServer->>JaatoServer: Create ServerAgentHooks callbacks
    JaatoServer->>JaatoClient: send_message(text, on_output=hooks.on_agent_output)

    JaatoClient->>JaatoSession: send_message(text, on_output, attachments)

    Note over JaatoSession: Message processing loop
    JaatoSession->>JaatoSession: Create CancelToken
    JaatoSession->>JaatoSession: Add user message to history
    JaatoSession->>JaatoSession: Build provider request

    JaatoSession->>Provider: send_message_streaming(messages, on_chunk, cancel_token)

    Provider->>LLM: API request (streaming)

    loop Streaming response chunks
        LLM-->>Provider: Text chunk / Function call

        alt Text chunk
            Provider->>JaatoSession: on_chunk(text)
            JaatoSession->>JaatoClient: on_output("model", text, "append")
            JaatoClient->>JaatoServer: hooks.on_agent_output(agent_id, "model", text, "append")
            JaatoServer->>JaatoServer: emit(AgentOutputEvent)
            JaatoServer->>SessionMgr: _emit_to_client(client_id, event)
            SessionMgr->>IPCServer: queue_event(client_id, event)
            IPCServer->>IPCClient: Async queue → JSON frame
            IPCClient->>RichClient: event received
            RichClient->>RichClient: PTDisplay.add_output(text)
            RichClient->>User: Display streaming text
        end

        alt Function call
            Note over Provider: See Tool Execution Flow (Section 5)
        end
    end

    LLM-->>Provider: FinishReason.STOP
    Provider->>JaatoSession: ProviderResponse (final)

    JaatoSession->>JaatoSession: Add assistant message to history
    JaatoSession->>JaatoSession: Update token accounting
    JaatoSession->>JaatoClient: hooks.on_agent_turn_completed()

    JaatoClient->>JaatoServer: emit(TurnCompletedEvent)
    JaatoServer->>JaatoServer: emit(ContextUpdatedEvent)

    JaatoServer->>IPCServer: Queue events
    IPCServer->>IPCClient: TurnCompletedEvent, ContextUpdatedEvent
    IPCClient->>RichClient: Turn complete
    RichClient->>User: Display token stats, ready for input
```

---

## 4. Streaming Output Pipeline

### Output Callback Chain

```mermaid
sequenceDiagram
    autonumber
    participant Provider as ModelProviderPlugin
    participant Session as JaatoSession
    participant Executor as ToolExecutor
    participant Formatter as FormatterPipeline
    participant Hooks as ServerAgentHooks
    participant Server as JaatoServer
    participant IPC as JaatoIPCServer
    participant Client as RichClient
    participant Display as PTDisplay

    Note over Provider,Display: MODEL OUTPUT PATH

    Provider->>Session: on_chunk(text_chunk)
    Session->>Hooks: on_output("model", text, "append")

    Hooks->>Formatter: process(source="model", text, mode)

    alt Regular text (no code block)
        Formatter->>Formatter: Pass through immediately
        Formatter->>Hooks: formatted_text
    else Code block detected
        Formatter->>Formatter: Buffer until block complete
        Note over Formatter: DiffFormatter (priority 20)<br/>CodeValidationFormatter (priority 35)<br/>CodeBlockFormatter (priority 40)
        Formatter->>Hooks: Buffered (wait for flush)
    end

    Hooks->>Server: emit(AgentOutputEvent(source, text, mode))
    Server->>IPC: queue_event(client_id, event)
    IPC->>Client: AgentOutputEvent

    Client->>Client: _process_agent_output_event()
    Client->>Display: add_output(text, source, mode)

    alt mode == "write"
        Display->>Display: Clear current block
        Display->>Display: Start new block
    else mode == "append"
        Display->>Display: Append to current block
    end

    Display->>Display: Scroll if needed
    Display->>Display: Invalidate layout

    Note over Provider,Display: TOOL OUTPUT PATH

    Executor->>Executor: Execute tool with OutputCallback
    Executor->>Session: on_tool_output(tool_name, output_text)
    Session->>Hooks: on_output(tool_name, output_text, "write")

    Note over Hooks: Tool output bypasses formatter

    Hooks->>Server: emit(ToolOutputEvent(tool_name, output_text))
    Server->>IPC: queue_event(client_id, event)
    IPC->>Client: ToolOutputEvent

    Client->>Display: add_tool_output(tool_name, output_text)

    Note over Provider,Display: TURN COMPLETION (flush formatter)

    Session->>Hooks: on_agent_turn_completed()
    Hooks->>Formatter: flush()

    Formatter->>Formatter: Emit all buffered code blocks
    Formatter->>Hooks: remaining formatted text

    Hooks->>Server: emit(TurnCompletedEvent)
    Server->>IPC: queue_event(client_id, event)
```

### Output Sources

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Output Source Taxonomy                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  source: "model"                                                                │
│  ├─ LLM-generated text responses                                               │
│  ├─ Processed through FormatterPipeline                                        │
│  └─ Triggers: code highlighting, diff rendering, LSP diagnostics               │
│                                                                                  │
│  source: "tool" or tool_name (e.g., "cli", "file_edit", "web_search")          │
│  ├─ Direct tool execution output                                               │
│  ├─ Bypasses formatter (raw output)                                            │
│  └─ Displayed in tool output section                                           │
│                                                                                  │
│  source: "system"                                                               │
│  ├─ Framework messages (initialization, errors, warnings)                      │
│  └─ Styled differently in UI                                                   │
│                                                                                  │
│  source: plugin_name (e.g., "mcp", "gc_plugin")                                │
│  ├─ Plugin-generated messages                                                  │
│  └─ Plugin-specific styling                                                    │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  mode: "write"  → Start new output block (replace current)                     │
│  mode: "append" → Continue current output block (streaming)                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Tool Execution Flow

### Function Calling Loop

```mermaid
sequenceDiagram
    autonumber
    participant Provider as ModelProviderPlugin
    participant Session as JaatoSession
    participant Executor as ToolExecutor
    participant Permission as PermissionPlugin
    participant Plugin as ToolPlugin<br/>(cli/mcp/file_edit)
    participant Hooks as ServerAgentHooks
    participant Server as JaatoServer
    participant IPC as IPCServer
    participant Client as RichClient

    Note over Provider,Client: LLM returns function_call

    Provider->>Session: ProviderResponse(function_calls=[...])

    loop For each function_call
        Session->>Session: Extract tool_name, arguments

        Session->>Executor: execute(tool_name, arguments)

        Note over Executor: Permission Check
        Executor->>Permission: check_tool_permission(tool_name, context)

        alt Permission granted (whitelist/cached)
            Permission->>Executor: PermissionResult.APPROVED
        else Permission needed
            Permission->>Executor: PermissionResult.NEEDS_APPROVAL
            Note over Executor: See Permission Flow (Section 6)
        else Permission denied
            Permission->>Executor: PermissionResult.DENIED
            Executor->>Session: ToolResult(error="Permission denied")
            Session->>Provider: send_tool_results([error_result])
        end

        Note over Executor: Tool Execution
        Executor->>Hooks: emit(ToolCallStartEvent(tool_name, args))
        Hooks->>Server: emit event
        Server->>IPC: queue_event
        IPC->>Client: ToolCallStartEvent
        Client->>Client: Update tool tree UI

        Executor->>Executor: Lookup tool plugin

        alt Auto-background needed (long-running)
            Executor->>Executor: Submit to ThreadPoolExecutor
            Note over Executor: Execute in background
        else Synchronous execution
            Executor->>Plugin: tool_fn(arguments, output_callback)
        end

        loop Tool produces output
            Plugin->>Executor: output_callback(text)
            Executor->>Hooks: emit(ToolOutputEvent(tool_name, text))
            Hooks->>Server: emit event
            Server->>IPC: queue_event
            IPC->>Client: ToolOutputEvent
            Client->>Client: Display tool output
        end

        Plugin->>Executor: return result

        Executor->>Hooks: emit(ToolCallEndEvent(tool_name, result))
        Hooks->>Server: emit event
        Server->>IPC: queue_event
        IPC->>Client: ToolCallEndEvent
        Client->>Client: Mark tool complete in UI

        Executor->>Session: ToolResult(output=result)
    end

    Session->>Session: Collect all ToolResults
    Session->>Session: Add function_response to history

    Session->>Provider: send_tool_results(results)

    Note over Provider: Provider continues conversation loop
    Provider->>Provider: May return more function_calls<br/>or final text response
```

### Tool Plugin Registration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Tool Plugin Lifecycle                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  1. DISCOVERY (PluginRegistry.discover_plugins())                               │
│     ├─ Scan shared/plugins/ directories                                         │
│     ├─ Load plugin modules with __init__.py                                     │
│     └─ Call create_plugin() factory function                                    │
│                                                                                  │
│  2. INITIALIZATION (plugin.initialize(config))                                  │
│     ├─ Plugin-specific setup                                                    │
│     ├─ MCP: Connect to MCP servers defined in .mcp.json                        │
│     └─ CLI: Setup shell environment                                             │
│                                                                                  │
│  3. SCHEMA EXPORT (plugin.get_tool_schemas())                                   │
│     ├─ Return List[ToolSchema] for LLM function definitions                    │
│     └─ ToolSchema: name, description, parameters (JSON Schema)                 │
│                                                                                  │
│  4. REGISTRATION (ToolExecutor.register(name, fn))                              │
│     ├─ Map tool names to executor functions                                     │
│     └─ Configure permissions, output callbacks                                  │
│                                                                                  │
│  5. EXECUTION (tool_fn(args, output_callback))                                  │
│     ├─ Receive arguments from LLM function_call                                │
│     ├─ Execute tool logic                                                       │
│     ├─ Stream output via output_callback                                        │
│     └─ Return result string                                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Permission Request Flow

### Interactive Permission Dialog

```mermaid
sequenceDiagram
    autonumber
    participant Executor as ToolExecutor
    participant Permission as PermissionPlugin
    participant Hooks as ServerAgentHooks
    participant Server as JaatoServer
    participant IPC as IPCServer
    participant Queue as Event Queue
    participant Client as RichClient
    participant User

    Executor->>Permission: check_tool_permission(tool_name, context)

    Permission->>Permission: Check policy cache

    alt Policy: ALWAYS_ALLOW
        Permission->>Executor: PermissionResult.APPROVED
        Note over Executor: Continue execution
    else Policy: NEVER_ALLOW
        Permission->>Executor: PermissionResult.DENIED
        Note over Executor: Return error to LLM
    else Policy: ASK or NO_POLICY
        Permission->>Permission: Generate request_id
        Permission->>Hooks: emit(PermissionRequestedEvent)

        Hooks->>Server: emit(PermissionRequestedEvent)
        Note over Server: request_id, tool_name, arguments, context

        Server->>IPC: queue_event(client_id, event)
        IPC->>Queue: Enqueue event
        Queue->>Client: PermissionRequestedEvent

        Client->>Client: _process_permission_requested_event()
        Client->>User: Display permission dialog<br/>"Allow 'cli' to run 'rm -rf ...'?<br/>[y]es / [n]o / [a]lways / ne[v]er"

        User->>Client: User types response (y/n/a/v)

        Client->>Client: Parse response
        Client->>IPC: PermissionResponseRequest(request_id, response)

        IPC->>Server: Handle PermissionResponseRequest
        Server->>Permission: resolve_permission(request_id, response)

        alt response == "always"
            Permission->>Permission: Update policy: ALWAYS_ALLOW
            Permission->>Executor: PermissionResult.APPROVED
        else response == "yes"
            Permission->>Permission: Cache: approved for this call
            Permission->>Executor: PermissionResult.APPROVED
        else response == "never"
            Permission->>Permission: Update policy: NEVER_ALLOW
            Permission->>Executor: PermissionResult.DENIED
        else response == "no"
            Permission->>Permission: Cache: denied for this call
            Permission->>Executor: PermissionResult.DENIED
        end

        Server->>IPC: emit(PermissionResolvedEvent(request_id, approved))
        IPC->>Client: PermissionResolvedEvent
        Client->>Client: Update UI (permission resolved)

        Note over Executor: Resume or abort tool execution
    end
```

### Permission Context

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Permission Context Fields                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PermissionContext:                                                              │
│  ├─ agent_type: str          # "main", "subagent", "review"                     │
│  ├─ agent_name: str          # Human-readable agent identifier                  │
│  ├─ tool_name: str           # Tool being invoked                               │
│  ├─ arguments: Dict          # Tool arguments (for display)                     │
│  ├─ workspace_path: str      # Current working directory                        │
│  ├─ file_paths: List[str]    # Files being accessed (if applicable)            │
│  └─ risk_level: str          # "low", "medium", "high", "critical"             │
│                                                                                  │
│  Risk Levels:                                                                    │
│  ├─ low: Read-only operations (file read, search)                              │
│  ├─ medium: Local modifications (file write, git commit)                       │
│  ├─ high: System operations (shell commands, network)                          │
│  └─ critical: Destructive operations (rm -rf, git push --force)                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Multi-Agent / Subagent Flow

### Subagent Creation and Communication

```mermaid
sequenceDiagram
    autonumber
    participant MainSession as JaatoSession<br/>(main agent)
    participant Runtime as JaatoRuntime<br/>(shared resources)
    participant SubSession as JaatoSession<br/>(subagent)
    participant SubProvider as ModelProviderPlugin<br/>(subagent's model)
    participant Hooks as ServerAgentHooks
    participant Server as JaatoServer
    participant IPC as IPCServer
    participant Client as RichClient

    Note over MainSession,Client: Main agent decides to spawn subagent

    MainSession->>MainSession: Process function_call: create_subagent
    MainSession->>Runtime: create_session(model, tools, system_instructions)

    Note over Runtime: Lightweight session creation
    Runtime->>Runtime: Reuse: registry, permissions, ledger
    Runtime->>SubSession: new JaatoSession(runtime_resources)

    SubSession->>SubSession: Configure: own history, tools, instructions
    SubSession->>SubSession: set_agent_context("subagent", "code-reviewer")

    Runtime->>MainSession: return subagent_session

    MainSession->>Hooks: emit(AgentCreatedEvent)
    Note over Hooks: agent_id, agent_type, parent_agent_id
    Hooks->>Server: emit event
    Server->>IPC: queue_event
    IPC->>Client: AgentCreatedEvent
    Client->>Client: Register agent in AgentRegistry
    Client->>Client: Update agent tree UI

    Note over MainSession,Client: Subagent executes task

    MainSession->>SubSession: send_message(task_description, on_output)

    loop Subagent processing
        SubSession->>SubProvider: send_message_streaming()
        SubProvider->>SubSession: on_chunk(text)
        SubSession->>MainSession: on_output("subagent:code-reviewer", text, "append")
        MainSession->>Hooks: emit(AgentOutputEvent(agent_id=subagent_id, ...))
        Hooks->>Server: emit event
        Server->>IPC: queue_event
        IPC->>Client: AgentOutputEvent (from subagent)
        Client->>Client: Display subagent output<br/>(indented/styled differently)
    end

    SubSession->>MainSession: return result

    MainSession->>Hooks: emit(AgentCompletedEvent(agent_id=subagent_id))
    Hooks->>Server: emit event
    Server->>IPC: queue_event
    IPC->>Client: AgentCompletedEvent
    Client->>Client: Mark subagent complete in registry

    Note over MainSession: Main agent continues with subagent result
    MainSession->>MainSession: Incorporate result into response
```

### Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Agent Hierarchy Model                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                    ┌─────────────────────────────────┐                          │
│                    │        JaatoRuntime             │                          │
│                    │  (Shared across all agents)     │                          │
│                    │                                 │                          │
│                    │  • Provider configuration       │                          │
│                    │  • PluginRegistry               │                          │
│                    │  • PermissionPlugin             │                          │
│                    │  • TokenLedger (aggregated)     │                          │
│                    └──────────────┬──────────────────┘                          │
│                                   │                                              │
│              ┌────────────────────┼────────────────────┐                        │
│              │                    │                    │                        │
│              ▼                    ▼                    ▼                        │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐           │
│  │   JaatoSession    │  │   JaatoSession    │  │   JaatoSession    │           │
│  │   (main agent)    │  │   (subagent A)    │  │   (subagent B)    │           │
│  │                   │  │                   │  │                   │           │
│  │  • Own history    │  │  • Own history    │  │  • Own history    │           │
│  │  • All tools      │  │  • Tool subset    │  │  • Tool subset    │           │
│  │  • gemini-2.5-pro │  │  • gemini-flash   │  │  • claude-sonnet  │           │
│  │  • Full instruct  │  │  • Review focus   │  │  • Code focus     │           │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘           │
│           │                      │                      │                       │
│           │                      │                      │                       │
│           │         ┌────────────┴────────────┐         │                       │
│           │         │     Parent manages      │         │                       │
│           └─────────┤  • Spawn/terminate      ├─────────┘                       │
│                     │  • Inject messages      │                                  │
│                     │  • Collect results      │                                  │
│                     └─────────────────────────┘                                  │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Benefits:                                                                       │
│  • No redundant provider connections (shared runtime)                           │
│  • Fast spawning (lightweight session creation)                                 │
│  • Isolated conversations (separate histories)                                  │
│  • Shared permissions (consistent policy across agents)                         │
│  • Aggregated token accounting (single ledger)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Agent Communication Patterns

```mermaid
sequenceDiagram
    autonumber
    participant Main as Main Agent
    participant Runtime as JaatoRuntime
    participant SubA as Subagent A<br/>(reviewer)
    participant SubB as Subagent B<br/>(coder)
    participant Queue as Injection Queue

    Note over Main,Queue: Pattern 1: Sequential Delegation
    Main->>Runtime: create_session("reviewer")
    Runtime->>SubA: new session
    Main->>SubA: send_message("Review this code...")
    SubA->>Main: return review_result
    Main->>Main: Process review

    Note over Main,Queue: Pattern 2: Parallel Delegation
    Main->>Runtime: create_session("coder")
    Runtime->>SubB: new session

    par Parallel execution
        Main->>SubA: send_message("Review PR #123")
        Main->>SubB: send_message("Write tests for X")
    end

    SubA->>Main: review_result
    SubB->>Main: test_code
    Main->>Main: Combine results

    Note over Main,Queue: Pattern 3: Async Injection
    Main->>SubA: Start long task

    Note over Main: Main continues other work

    Main->>Queue: inject_message(SubA, "Stop and summarize")
    Queue->>SubA: Message injected
    SubA->>SubA: Handle injection
    SubA->>Main: return partial_result
```

---

## 8. Session Management Flow

### Multi-Session Orchestration

```mermaid
sequenceDiagram
    autonumber
    participant Client1 as Client 1<br/>(ipc_1)
    participant Client2 as Client 2<br/>(ipc_2)
    participant IPC as JaatoIPCServer
    participant SessionMgr as SessionManager
    participant SessionPlugin as SessionPlugin<br/>(persistence)
    participant ServerMain as JaatoServer<br/>(session: main)
    participant ServerDev as JaatoServer<br/>(session: dev)
    participant Storage as Disk Storage<br/>(.jaato/sessions/)

    Note over Client1,Storage: Client 1 connects and uses "main" session

    Client1->>IPC: ClientConfigRequest(workspace="/project")
    IPC->>SessionMgr: get_or_create_default("ipc_1", "/project")

    SessionMgr->>SessionPlugin: load_session("main")
    SessionPlugin->>Storage: Read sessions/main.json
    Storage->>SessionPlugin: Session state (history, config)
    SessionPlugin->>SessionMgr: SessionState

    SessionMgr->>ServerMain: new JaatoServer(session_state)
    ServerMain->>ServerMain: initialize()
    SessionMgr->>SessionMgr: Map: ipc_1 → "main"

    IPC->>Client1: SessionInfoEvent(session_id="main")

    Note over Client1,Storage: Client 2 connects and creates "dev" session

    Client2->>IPC: ClientConfigRequest(workspace="/project")
    Client2->>IPC: CommandRequest("/session create dev")

    IPC->>SessionMgr: create_session("ipc_2", "dev", "/project")

    SessionMgr->>ServerDev: new JaatoServer(empty_state)
    ServerDev->>ServerDev: initialize()
    SessionMgr->>SessionMgr: Map: ipc_2 → "dev"

    IPC->>Client2: SessionInfoEvent(session_id="dev")

    Note over Client1,Storage: Client 1 switches to "dev" session

    Client1->>IPC: CommandRequest("/session attach dev")
    IPC->>SessionMgr: attach_session("ipc_1", "dev", "/project")

    SessionMgr->>SessionMgr: Check workspace compatibility
    SessionMgr->>SessionMgr: Map: ipc_1 → "dev"

    IPC->>Client1: SessionInfoEvent(session_id="dev")

    Note over Client1,Storage: Both clients now share "dev" session

    Client1->>IPC: SendMessageRequest("Hello from Client 1")
    IPC->>SessionMgr: handle_request("ipc_1", "dev", event)
    SessionMgr->>ServerDev: handle_request()
    ServerDev->>SessionMgr: AgentOutputEvent
    SessionMgr->>IPC: Broadcast to ipc_1 AND ipc_2

    IPC->>Client1: AgentOutputEvent
    IPC->>Client2: AgentOutputEvent

    Note over Client1,Storage: Session auto-save on turn completion

    ServerDev->>SessionMgr: TurnCompletedEvent
    SessionMgr->>SessionPlugin: save_session("dev", state)
    SessionPlugin->>Storage: Write sessions/dev.json
```

### Session State Persistence

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Session State Structure                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  .jaato/sessions/main.json:                                                     │
│  {                                                                               │
│    "session_id": "main",                                                        │
│    "created_at": "2025-01-07T10:00:00Z",                                        │
│    "last_activity": "2025-01-07T14:30:00Z",                                     │
│    "workspace_path": "/home/user/project",                                      │
│    "model": "gemini-2.5-pro",                                                   │
│    "system_instructions": "You are a helpful assistant...",                     │
│    "history": [                                                                  │
│      {                                                                           │
│        "role": "user",                                                          │
│        "parts": [{"text": "Hello"}]                                             │
│      },                                                                          │
│      {                                                                           │
│        "role": "assistant",                                                      │
│        "parts": [{"text": "Hi! How can I help?"}]                               │
│      }                                                                           │
│    ],                                                                            │
│    "tool_config": {                                                              │
│      "enabled_tools": ["cli", "file_edit", "web_search"],                       │
│      "permissions": {                                                            │
│        "cli": "ask",                                                            │
│        "file_edit": "always_allow"                                              │
│      }                                                                           │
│    },                                                                            │
│    "token_usage": {                                                              │
│      "total_prompt_tokens": 15000,                                              │
│      "total_output_tokens": 8000                                                │
│    },                                                                            │
│    "metadata": {                                                                 │
│      "gc_runs": 2,                                                              │
│      "turns_count": 25                                                          │
│    }                                                                             │
│  }                                                                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Complete End-to-End Flow

### Full Request Lifecycle with Tool Execution

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant TUI as RichClient TUI<br/>(PTDisplay)
    participant Client as IPCClient
    participant IPC as JaatoIPCServer
    participant SessionMgr as SessionManager
    participant Server as JaatoServer
    participant JClient as JaatoClient
    participant Session as JaatoSession
    participant Provider as ModelProviderPlugin
    participant Executor as ToolExecutor
    participant Permission as PermissionPlugin
    participant Tool as ToolPlugin (cli)
    participant LLM as LLM API

    User->>TUI: Type: "List files in /tmp"
    TUI->>TUI: Capture input
    TUI->>Client: send_message("List files in /tmp")

    Client->>Client: Create SendMessageRequest
    Client->>IPC: JSON frame (4-byte len + payload)

    IPC->>IPC: deserialize_event()
    IPC->>SessionMgr: handle_request(client_id, session_id, event)

    SessionMgr->>Server: handle_request(SendMessageRequest)

    Server->>JClient: send_message(text, on_output=hooks)
    JClient->>Session: send_message(text, on_output)

    Session->>Session: Add user message to history
    Session->>Provider: send_message_streaming(messages)

    Provider->>LLM: API Request
    LLM->>Provider: Streaming: "I'll list the files..."

    Provider->>Session: on_chunk("I'll list the files...")
    Session->>Server: on_output("model", chunk)
    Server->>IPC: AgentOutputEvent
    IPC->>Client: Event
    Client->>TUI: Display text
    TUI->>User: See: "I'll list the files..."

    LLM->>Provider: FunctionCall: cli(command="ls /tmp")
    Provider->>Session: ProviderResponse(function_calls)

    Session->>Executor: execute("cli", {command: "ls /tmp"})

    Executor->>Permission: check_tool_permission("cli", context)
    Permission->>IPC: PermissionRequestedEvent
    IPC->>Client: Event
    Client->>TUI: Show permission dialog
    TUI->>User: "Allow cli: ls /tmp? [y/n/a/v]"

    User->>TUI: Type: "y"
    TUI->>Client: respond_to_permission(request_id, "yes")
    Client->>IPC: PermissionResponseRequest
    IPC->>Permission: resolve_permission()
    Permission->>Executor: APPROVED

    Executor->>Server: ToolCallStartEvent
    Server->>IPC: Event
    IPC->>Client: Event
    Client->>TUI: Update tool tree

    Executor->>Tool: cli(command="ls /tmp", output_callback)
    Tool->>Tool: Execute: ls /tmp
    Tool->>Executor: output_callback("file1.txt\nfile2.txt\n...")
    Executor->>Server: ToolOutputEvent
    Server->>IPC: Event
    IPC->>Client: Event
    Client->>TUI: Display tool output
    TUI->>User: See file listing

    Tool->>Executor: return result
    Executor->>Server: ToolCallEndEvent
    Server->>IPC: Event
    IPC->>Client: Event
    Client->>TUI: Mark tool complete

    Executor->>Session: ToolResult
    Session->>Session: Add function_response to history
    Session->>Provider: send_tool_results([result])

    Provider->>LLM: API Request with tool result
    LLM->>Provider: "Here are the files in /tmp: ..."

    Provider->>Session: on_chunk("Here are the files...")
    Session->>Server: on_output("model", chunk)
    Server->>IPC: AgentOutputEvent
    IPC->>Client: Event
    Client->>TUI: Display final response
    TUI->>User: See: "Here are the files in /tmp: file1.txt, file2.txt..."

    Provider->>Session: FinishReason.STOP
    Session->>Session: Add assistant message to history
    Session->>Server: on_agent_turn_completed()
    Server->>IPC: TurnCompletedEvent
    Server->>IPC: ContextUpdatedEvent
    IPC->>Client: Events
    Client->>TUI: Update token stats
    TUI->>User: Ready for next input
```

---

## 10. Component Reference

### Key Classes and Files

| Component | File | Purpose |
|-----------|------|---------|
| **RichClient** | `jaato-tui/rich_client.py` | TUI application, event handling |
| **IPCClient** | `jaato-tui/ipc_client.py` | Client-side IPC connection |
| **PTDisplay** | `jaato-tui/pt_display.py` | prompt_toolkit display layout |
| **Backend** | `jaato-tui/backend.py` | Backend abstraction (Direct/IPC) |
| **JaatoDaemon** | `server/__main__.py` | Daemon process, PID management |
| **JaatoIPCServer** | `server/ipc.py` | Server-side IPC handling |
| **JaatoWSServer** | `server/websocket.py` | WebSocket server |
| **SessionManager** | `server/session_manager.py` | Multi-session orchestration |
| **JaatoServer** | `server/core.py` | Per-session request handler |
| **Events** | `server/events.py` | Event types, serialization |
| **JaatoClient** | `shared/jaato_client.py` | Core facade |
| **JaatoRuntime** | `shared/jaato_runtime.py` | Shared resources |
| **JaatoSession** | `shared/jaato_session.py` | Per-agent state |
| **ToolExecutor** | `shared/ai_tool_runner.py` | Tool execution |
| **TokenLedger** | `shared/token_accounting.py` | Token tracking |
| **PluginRegistry** | `shared/plugins/registry.py` | Plugin discovery |
| **PermissionPlugin** | `shared/plugins/permission/` | Permission control |
| **ModelProviderPlugin** | `shared/plugins/model_provider/` | Provider abstraction |

### Event Types Quick Reference

| Event | Direction | Purpose |
|-------|-----------|---------|
| `ConnectedEvent` | S→C | Connection established |
| `SessionInfoEvent` | S→C | Session metadata |
| `AgentCreatedEvent` | S→C | New agent spawned |
| `AgentOutputEvent` | S→C | Streaming text output |
| `AgentStatusChangedEvent` | S→C | Agent state change |
| `AgentCompletedEvent` | S→C | Agent finished |
| `ToolCallStartEvent` | S→C | Tool execution started |
| `ToolCallEndEvent` | S→C | Tool execution completed |
| `ToolOutputEvent` | S→C | Tool output chunk |
| `PermissionRequestedEvent` | S→C | Permission needed |
| `PermissionResolvedEvent` | S→C | Permission resolved |
| `TurnCompletedEvent` | S→C | Turn finished |
| `ContextUpdatedEvent` | S→C | Token usage update |
| `PlanUpdatedEvent` | S→C | Plan step progress |
| `SendMessageRequest` | C→S | User message |
| `PermissionResponseRequest` | C→S | Permission response |
| `StopRequest` | C→S | Cancel processing |
| `CommandRequest` | C→S | Execute command |
| `HistoryRequest` | C→S | Get history |
| `ClientConfigRequest` | C→S | Client configuration |

---

## Appendix: Mermaid Diagram Rendering

These diagrams use Mermaid syntax. To render them:

1. **GitHub**: Mermaid diagrams render automatically in markdown files
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Use [Mermaid Live Editor](https://mermaid.live)
4. **CLI**: Use `mmdc` (Mermaid CLI) to export as PNG/SVG

```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Export diagram
mmdc -i docs/sequence-diagram-architecture.md -o diagram.png
```
