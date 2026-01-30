# JAATO Tool System & Discoverability Architecture

## Executive Summary

JAATO provides tools to AI models through a sophisticated **deferred loading** architecture. Rather than loading all 100+ tools upfront (consuming valuable context tokens), JAATO exposes only a small set of **core** tools initially. The model can then **discover** additional tools on-demand through introspection, activating them only when needed. This design significantly reduces initial context overhead while maintaining full capability access.

---

## Part 1: Discoverability Modes

Tools in JAATO have a `discoverability` attribute that controls when they appear in the model's context:

### Mode Definitions

| Mode | Behavior | When to Use |
|------|----------|-------------|
| **`core`** | Always loaded in initial context | Essential tools the model needs immediately |
| **`discoverable`** | Loaded on-demand via introspection | Specialized tools for specific use cases |

**Default:** All tools default to `"discoverable"` unless explicitly marked as `"core"`.

### Why This Matters

```
Traditional Approach (All Tools Upfront):
┌─────────────────────────────────────────────────────────────────┐
│  Initial Context: ~8,000-15,000 tokens for tool schemas         │
│  100+ tools × ~80-150 tokens per schema = significant overhead  │
└─────────────────────────────────────────────────────────────────┘

JAATO Deferred Approach:
┌─────────────────────────────────────────────────────────────────┐
│  Initial Context: ~1,500-2,500 tokens for core tools            │
│  ~15 core tools × ~100 tokens = minimal overhead                │
│  + Introspection tools enable discovery of 85+ more tools       │
└─────────────────────────────────────────────────────────────────┘
```

**Token Savings:** 70-85% reduction in initial tool schema tokens.

---

## Part 2: Core Tools (Always Available)

Core tools are loaded immediately when a session starts. They represent the **minimum viable toolkit** for agentic operation.

### Core Tool Registry

| Plugin | Tool | Purpose | Why Core? |
|--------|------|---------|-----------|
| **introspection** | `list_tools` | Discover available tool categories | Gateway to all other tools |
| **introspection** | `get_tool_schemas` | Get detailed schemas for tools | Required to use discoverable tools |
| **file_edit** | `readFile` | Read file contents | Essential for code understanding |
| **cli** | `run` | Execute shell commands | Fundamental system interaction |
| **environment** | `get_environment` | Query environment info | Context awareness |
| **todo** | `add_todo` | Create task items | Task tracking |
| **todo** | `list_todos` | View existing tasks | Task management |
| **todo** | `update_todo_step` | Update task progress | Progress tracking |
| **todo** | `get_todo` | Get task details | Task inspection |
| **todo** | `delete_todo` | Remove tasks | Cleanup |
| **todo** | `mark_todo_complete` | Complete tasks | Task completion |
| **todo** | `clear_todos` | Clear all tasks | Reset capability |
| **todo** | *(additional)* | Various TODO operations | Comprehensive task mgmt |

### Why These Are Core

**Introspection Tools (2):** The model cannot discover or use any other tools without these. They are the foundation of the entire deferred loading system.

**File Reading (1):** Code assistants spend most time reading code. Making `readFile` core ensures the model can immediately start understanding the codebase without extra discovery steps.

**Shell Access (1):** Many tasks require running commands. Having `run` available immediately enables the model to execute builds, tests, and other operations from the start.

**Environment Info (1):** Models need context about their operating environment (OS, working directory, etc.) to make appropriate decisions.

**TODO System (8+):** Task tracking is essential for complex multi-step operations. Core availability ensures the model can organize work from the first turn.

### Token Budget (Core Tools)

```
┌────────────────────────────────────────────────────────────────┐
│  CORE TOOL TOKEN ESTIMATES                                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Introspection (2 tools):           ~250 tokens                │
│  ├─ list_tools:                     ~120 tokens                │
│  └─ get_tool_schemas:               ~130 tokens                │
│                                                                 │
│  File Operations (1 tool):          ~100 tokens                │
│  └─ readFile:                       ~100 tokens                │
│                                                                 │
│  CLI (1 tool):                      ~150 tokens                │
│  └─ run:                            ~150 tokens                │
│                                                                 │
│  Environment (1 tool):              ~80 tokens                 │
│  └─ get_environment:                ~80 tokens                 │
│                                                                 │
│  TODO System (8+ tools):            ~600 tokens                │
│  ├─ add_todo:                       ~80 tokens                 │
│  ├─ list_todos:                     ~60 tokens                 │
│  └─ ... (6+ more):                  ~460 tokens                │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  TOTAL CORE TOOLS:                  ~1,180 tokens              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Discoverable Tools (On-Demand)

Discoverable tools are NOT loaded initially. The model must explicitly request their schemas via `get_tool_schemas()` before using them.

### Available Discoverable Tools by Category

#### Filesystem Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **file_edit** | `updateFile` | Modify existing files |
| **file_edit** | `writeNewFile` | Create new files |
| **file_edit** | `removeFile` | Delete files |
| **file_edit** | `moveFile` | Move/rename files |
| **file_edit** | `copyFile` | Copy files |
| **file_edit** | `createDirectory` | Create directories |
| **file_edit** | `multiFileEdit` | Batch file operations |
| **filesystem_query** | `glob_files` | Pattern-based file search |
| **filesystem_query** | `list_directory` | Directory listing |
| **filesystem_query** | `get_file_info` | File metadata |

#### Search Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **filesystem_query** | `grep_content` | Search file contents |
| **web_search** | `web_search` | Internet search |
| **ast_search** | `ast_search` | Code structure search |

#### Web Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **web_fetch** | `fetch_url` | HTTP GET requests |
| **web_fetch** | `fetch_url_post` | HTTP POST requests |

#### Code Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **lsp** | `lsp_query` | Language Server Protocol |
| **notebook** | `execute_cell` | Run notebook cells |
| **notebook** | `edit_notebook` | Modify notebooks |

#### Coordination Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **subagent** | `delegate` | Spawn subagents |
| **background** | `start_background` | Background tasks |
| **background** | `check_background` | Task status |
| **background** | `stop_background` | Cancel tasks |
| **waypoint** | `save_waypoint` | Session checkpoints |
| **waypoint** | `restore_waypoint` | Restore checkpoints |

#### Communication Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **clarification** | `request_clarification` | Ask user questions |

#### Memory Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **template** | `list_templates` | View templates |
| **template** | `get_template` | Retrieve template |
| **template** | `create_template` | New template |
| **template** | `delete_template` | Remove template |

#### System Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **thinking** | `extended_thinking` | Deep reasoning |
| **sandbox_manager** | `manage_sandbox` | Sandbox control |

#### Authentication Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **anthropic_auth** | `anthropic_auth_login` | Anthropic OAuth |
| **anthropic_auth** | `anthropic_auth_logout` | Logout |
| **github_auth** | `github_auth_login` | GitHub OAuth |
| **github_auth** | `github_auth_logout` | Logout |

#### MCP Category
| Plugin | Tool | Purpose |
|--------|------|---------|
| **mcp** | *(dynamic)* | External MCP server tools |

### Why These Are Discoverable

1. **Specialization:** Most tasks don't need all tools. A code review task rarely needs `web_fetch` or `anthropic_auth_login`.

2. **Token Economy:** Loading 85+ tool schemas would consume 7,000-12,000 tokens of context. Deferring them saves this overhead.

3. **Dynamic MCP Tools:** MCP servers may provide dozens of tools. Loading them all upfront would be wasteful.

4. **Write Operations:** File modification tools (`updateFile`, `writeNewFile`, etc.) are only needed when making changes, not for read-only analysis.

---

## Part 4: The Introspection Mechanism

The introspection system is the bridge between core and discoverable tools.

### Tool Discovery Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOOL DISCOVERY WORKFLOW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐                                                │
│  │  Session Start   │                                                │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  Load Core Tools                                  │                │
│  │  • introspection (list_tools, get_tool_schemas)  │                │
│  │  • file_edit (readFile)                          │                │
│  │  • cli (run)                                     │                │
│  │  • environment (get_environment)                 │                │
│  │  • todo (8+ tools)                               │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  Model needs web_search capability               │                │
│  │  "I should search the web for this info..."      │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  Model calls: list_tools(category="search")      │                │
│  │                                                  │                │
│  │  Response:                                       │                │
│  │  • grep_content - Search file contents           │                │
│  │  • web_search - Search the internet              │                │
│  │  • ast_search - Search code by structure         │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  Model calls: get_tool_schemas(names=["web_search"]) │            │
│  │                                                  │                │
│  │  Response:                                       │                │
│  │  {                                               │                │
│  │    "name": "web_search",                         │                │
│  │    "description": "Search the web...",           │                │
│  │    "parameters": { ... }                         │                │
│  │  }                                               │                │
│  │                                                  │                │
│  │  ★ ACTIVATION: Tool is now added to provider's  │                │
│  │    declared tools list                           │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  Model can now call: web_search(query="...")     │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Introspection Tool Details

#### `list_tools`

**Purpose:** Discover what tools are available, organized by category.

**Usage Patterns:**

```python
# Get all categories with tool counts
list_tools()
# Returns: {"filesystem": 10, "search": 3, "web": 2, ...}

# Get tools in a specific category
list_tools(category="filesystem")
# Returns: [
#   {"name": "readFile", "description": "Read file contents", "enabled": true},
#   {"name": "updateFile", "description": "Modify files", "enabled": true},
#   ...
# ]
```

**Categories Available:**
- `filesystem` - File operations
- `code` - Code editing, LSP, analysis
- `search` - File and web search
- `memory` - Persistent memory, templates
- `coordination` - Subagents, background tasks, TODO
- `system` - Shell, environment
- `web` - URL fetching, external APIs
- `communication` - User interaction
- `MCP` - Model Context Protocol servers

#### `get_tool_schemas`

**Purpose:** Retrieve detailed parameter schemas for specific tools AND activate them for use.

**Usage:**

```python
# Get schemas for specific tools
get_tool_schemas(names=["web_search", "web_fetch"])
# Returns: [
#   {
#     "name": "web_search",
#     "description": "Search the internet for information",
#     "parameters": {
#       "type": "object",
#       "properties": {
#         "query": {"type": "string", "description": "Search query"}
#       },
#       "required": ["query"]
#     }
#   },
#   ...
# ]
```

**Critical Side Effect:** Calling `get_tool_schemas()` automatically **activates** the requested tools. They are added to the provider's declared tool list and become callable.

---

## Part 5: Tool Schema Structure

Every tool is defined by a `ToolSchema` dataclass:

```python
@dataclass
class ToolSchema:
    name: str                              # Unique identifier
    description: str                       # What the tool does
    parameters: Dict[str, Any]            # JSON Schema format
    category: Optional[str] = None        # Organizational grouping
    discoverability: str = "discoverable" # "core" or "discoverable"
```

### JSON Schema Parameters

Tool parameters follow JSON Schema specification:

```python
{
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file"
        },
        "content": {
            "type": "string",
            "description": "Content to write"
        },
        "create_directories": {
            "type": "boolean",
            "description": "Create parent dirs if needed",
            "default": false
        }
    },
    "required": ["path", "content"]
}
```

---

## Part 6: Tool Execution Flow

Once a tool is available (core or activated), execution follows this pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐                                                │
│  │  Model Request   │  "Call updateFile with path=... content=..."  │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  1. PERMISSION CHECK                             │                │
│  │     └─ PermissionPlugin.check_permission()       │                │
│  │        ├─ Auto-approved? → Continue              │                │
│  │        ├─ Whitelisted? → Continue                │                │
│  │        ├─ Blacklisted? → Return error            │                │
│  │        └─ Ask user → Wait for response           │                │
│  └────────┬────────────────────────────────────────┘                │
│           │ (if allowed)                                             │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  2. AUTO-BACKGROUND CHECK                        │                │
│  │     └─ If tool is BackgroundCapable:             │                │
│  │        ├─ Execution time estimate > threshold?   │                │
│  │        └─ Auto-background and return task handle │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  3. EXECUTOR LOOKUP                              │                │
│  │     ├─ Check registered executor map             │                │
│  │     ├─ Fallback: Query registry for plugin       │                │
│  │     └─ Get callable for tool                     │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  4. EXECUTE                                      │                │
│  │     ├─ Call executor(arguments)                  │                │
│  │     ├─ Handle streaming if supported             │                │
│  │     └─ Capture result                            │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  5. RESULT ENRICHMENT                            │                │
│  │     └─ Run through result enrichment pipeline    │                │
│  │        (plugins can modify/augment results)      │                │
│  └────────┬────────────────────────────────────────┘                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────┐                │
│  │  6. RETURN TO MODEL                              │                │
│  │     └─ (success: bool, result: str/dict)         │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Parallel Tool Execution

When the model requests multiple independent tools in one turn, they execute concurrently:

```
Model: "Read config.json, check git status, and list directory"
       └─ 3 tool calls in single response

       ┌──────────────────────────────────────────────────┐
       │  ThreadPoolExecutor (max 8 workers)              │
       │                                                  │
       │  ┌──────────┐ ┌──────────┐ ┌──────────────┐     │
       │  │ readFile │ │   run    │ │ list_directory │   │
       │  │ config   │ │ git stat │ │     ./       │     │
       │  └────┬─────┘ └────┬─────┘ └──────┬───────┘     │
       │       │            │              │              │
       │       ▼            ▼              ▼              │
       │    result1      result2        result3          │
       │       └────────────┴──────────────┘              │
       │                    │                             │
       │                    ▼                             │
       │            Combined response                     │
       └──────────────────────────────────────────────────┘
```

**Configuration:** `JAATO_PARALLEL_TOOLS=true` (default)

---

## Part 7: Plugin System Architecture

Tools are provided by **plugins**, which are discovered and managed by the **PluginRegistry**.

### Plugin Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PLUGIN LIFECYCLE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DISCOVERY                                                        │
│     registry.discover(plugin_kind='tool')                           │
│     ├─ Scan entry points (installed packages)                       │
│     └─ Scan shared/plugins/ directory                               │
│                                                                      │
│  2. EXPOSURE                                                         │
│     registry.expose_tool('file_edit', config)                       │
│     ├─ Call plugin.initialize(config)                               │
│     ├─ Auto-wire: set_plugin_registry(registry)                     │
│     ├─ Auto-wire: set_session(session)                              │
│     ├─ Register tool schemas                                        │
│     └─ Register executors                                           │
│                                                                      │
│  3. OPERATION                                                        │
│     ├─ get_tool_schemas() → Return ToolSchema list                  │
│     ├─ get_executors() → Return {name: callable} map                │
│     ├─ get_system_instructions() → Return prompt content            │
│     └─ execute() → Run tool logic                                   │
│                                                                      │
│  4. SHUTDOWN                                                         │
│     registry.unexpose_tool('file_edit')                             │
│     └─ Call plugin.shutdown()                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Plugin Interface (ToolPlugin Protocol)

```python
class ToolPlugin(Protocol):
    # Required
    def name(self) -> str: ...
    def get_tool_schemas(self) -> List[ToolSchema]: ...
    def get_executors(self) -> Dict[str, Callable]: ...
    def initialize(self, config: Optional[Dict]) -> None: ...
    def shutdown(self) -> None: ...

    # Optional
    def get_system_instructions(self) -> Optional[str]: ...
    def get_auto_approved_tools(self) -> List[str]: ...
    def get_model_requirements(self) -> Optional[List[str]]: ...
```

### Defining Discoverability in Plugins

```python
class MyPlugin:
    def get_tool_schemas(self) -> List[ToolSchema]:
        return [
            ToolSchema(
                name="essential_tool",
                description="Always needed",
                parameters={...},
                category="system",
                discoverability="core"  # ← Always loaded
            ),
            ToolSchema(
                name="specialized_tool",
                description="Sometimes needed",
                parameters={...},
                category="system",
                discoverability="discoverable"  # ← On-demand (default)
            ),
        ]
```

---

## Part 8: Complete Tool Inventory

### Summary Statistics

| Category | Core Tools | Discoverable Tools | Total |
|----------|------------|-------------------|-------|
| **Introspection** | 2 | 0 | 2 |
| **Filesystem** | 1 | 9+ | 10+ |
| **System** | 1 | 2+ | 3+ |
| **Coordination** | 8+ | 5+ | 13+ |
| **Search** | 0 | 3+ | 3+ |
| **Web** | 0 | 2+ | 2+ |
| **Code** | 0 | 3+ | 3+ |
| **Communication** | 0 | 1 | 1 |
| **Memory** | 0 | 4+ | 4+ |
| **Authentication** | 0 | 4+ | 4+ |
| **MCP** | 0 | dynamic | dynamic |
| **TOTAL** | ~14 | ~85+ | ~100+ |

### Plugin Inventory

| Plugin | Status | Core | Discoverable | Primary Purpose |
|--------|--------|------|--------------|-----------------|
| introspection | Essential | 2 | 0 | Tool discovery |
| file_edit | Essential | 1 | 7 | File CRUD |
| cli | Essential | 1 | 0 | Shell execution |
| environment | Essential | 1 | 0 | Env queries |
| todo | Essential | 8+ | 0 | Task management |
| filesystem_query | Common | 0 | 3+ | File search |
| web_search | Common | 0 | 1 | Internet search |
| web_fetch | Common | 0 | 2 | HTTP requests |
| subagent | Advanced | 0 | 1 | Spawn subagents |
| background | Advanced | 0 | 5 | Background tasks |
| clarification | Common | 0 | 1 | User questions |
| template | Utility | 0 | 4 | Template mgmt |
| lsp | Advanced | 0 | 1 | Language Server |
| notebook | Specialized | 0 | 2 | Jupyter |
| mcp | Dynamic | 0 | * | External servers |
| *(12 more)* | Various | 0 | ~50+ | Various |

---

## Part 9: Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `JAATO_DEFERRED_TOOLS` | Enable on-demand tool loading | `true` |
| `JAATO_PARALLEL_TOOLS` | Enable parallel tool execution | `true` |

### Disabling Deferred Loading

Set `JAATO_DEFERRED_TOOLS=false` to load all tools upfront. Use this when:
- Working with simple sessions that use most tools
- Debugging tool availability issues
- Token budget is not a concern

### Enabling/Disabling Individual Tools

```python
# Via registry
registry.disable_tool('web_search')
registry.enable_tool('web_search')

# Check status
if registry.is_tool_enabled('web_search'):
    # Tool is available
```

---

## Part 10: Visual Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JAATO TOOL SYSTEM OVERVIEW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                        ┌───────────────────┐                        │
│                        │   MODEL CONTEXT    │                        │
│                        │   (Token Budget)   │                        │
│                        └─────────┬─────────┘                        │
│                                  │                                   │
│              ┌───────────────────┴───────────────────┐              │
│              ▼                                       ▼              │
│  ┌─────────────────────────┐          ┌─────────────────────────┐  │
│  │      CORE TOOLS          │          │   DISCOVERABLE TOOLS    │  │
│  │   (Always Loaded)        │          │     (On-Demand)         │  │
│  │                          │          │                         │  │
│  │  • list_tools       ●────┼──────────┼─────────────────────►   │  │
│  │  • get_tool_schemas ●────┼──────────┼─────────────────────►   │  │
│  │  • readFile              │          │  • updateFile           │  │
│  │  • run                   │          │  • writeNewFile         │  │
│  │  • get_environment       │          │  • web_search           │  │
│  │  • add_todo              │          │  • web_fetch            │  │
│  │  • list_todos            │          │  • delegate             │  │
│  │  • (8 more TODO tools)   │          │  • (80+ more tools)     │  │
│  │                          │          │                         │  │
│  │  ~1,200 tokens           │          │  ~8,000+ tokens if all  │  │
│  └─────────────────────────┘          └─────────────────────────┘  │
│              │                                   ▲                   │
│              │                                   │                   │
│              │         INTROSPECTION             │                   │
│              │     ┌───────────────────┐         │                   │
│              └────►│ 1. list_tools()   │─────────┘                   │
│                    │ 2. get_tool_schemas()                           │
│                    │    ↓ (ACTIVATES)  │                             │
│                    │ 3. Tool now usable │                            │
│                    └───────────────────┘                             │
│                                                                      │
│  ══════════════════════════════════════════════════════════════════ │
│                                                                      │
│   BENEFITS:                                                          │
│   ✓ 70-85% reduction in initial context tokens                      │
│   ✓ Model discovers tools as needed                                 │
│   ✓ Specialized tools don't waste context                           │
│   ✓ MCP tools scale without upfront cost                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 11: Color Coding Suggestion for Infographic

- **Blue:** Core tools (always loaded, essential)
- **Green:** Discoverable tools (on-demand, activated via introspection)
- **Yellow:** Introspection tools (the bridge between core and discoverable)
- **Gray:** Token budget representations
- **Orange:** Activation flow arrows
- **Purple:** Plugin registry / infrastructure components
