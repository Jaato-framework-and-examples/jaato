# Background Plugin

The background plugin enables long-running tool executions to run asynchronously, allowing the model to continue working on other tasks while waiting for slow operations to complete.

## Features

- **Explicit backgrounding** - Model can proactively start tasks in background
- **Auto-backgrounding** - Tasks exceeding a plugin-defined threshold are automatically backgrounded
- **Task management** - Start, check status, get results, cancel, and list tasks
- **Plugin-scoped control** - Each plugin controls its own background thresholds

## Tools

| Tool | Description | Auto-approved |
|------|-------------|---------------|
| `startBackgroundTask` | Start a tool execution in background | No |
| `getBackgroundTaskStatus` | Check current status of a task | Yes |
| `getBackgroundTaskResult` | Get the result of a completed task | Yes |
| `cancelBackgroundTask` | Cancel a running task | No |
| `listBackgroundTasks` | List all active background tasks | Yes |
| `listBackgroundCapableTools` | List tools that support backgrounding | Yes |

## User Commands

| Command | Description |
|---------|-------------|
| `/tasks` | List all active background tasks |

## Usage

### Explicit Background Execution

The model can explicitly start a task in background when it anticipates long execution:

```
User: "Install dependencies and then run tests"

Model: startBackgroundTask(
    plugin_name="cli",
    tool_name="cli_based_tool",
    arguments={"command": "npm install"}
)

# Returns task handle immediately, model can continue...

Model: "I've started the installation in background (task: abc-123).
       While that runs, I'll prepare the test command..."

# Later...
Model: getBackgroundTaskStatus(task_id="abc-123")
Model: getBackgroundTaskResult(task_id="abc-123")
```

### Auto-Backgrounding

When a tool execution exceeds its configured threshold, it's automatically converted to a background task:

```
User: "Run the full test suite"

Model: cli_based_tool(command="npm test -- --coverage")

# After 10 seconds (threshold exceeded)...

Tool returns: {
    "auto_backgrounded": true,
    "task_id": "xyz-789",
    "threshold_seconds": 10.0,
    "background_reader_available": true,
    "reader_tool": "getBackgroundTask",
    "message": "Task exceeded 10.0s threshold, continuing in background.
                Use getBackgroundTask(task_id='xyz-789') to check status and output."
}

Model: "The tests are taking longer than expected and have moved to
       background. I'll monitor progress..."
```

### Auto-backgrounding requires this plugin

The *capability* comes from `BackgroundCapableMixin`, which a plugin such as
`cli` inherits directly — loading `cli` does **not** load `background`. The
tools that accept a `task_id` (`getBackgroundTask`, `cancelBackgroundTask`,
`listBackgroundTasks`) live here.

So the runner checks for `getBackgroundTask` before issuing a receipt
(`ToolExecutor._background_reader_tool`), and behaves differently when it is
absent (issue #804):

| `background` loaded | Behaviour past the threshold |
|---|---|
| yes | Returns `auto_backgrounded: true` with a message naming `getBackgroundTask(task_id=…)`. |
| no | Keeps waiting — up to `DEFAULT_NO_READER_TIMEOUT_SECONDS` (300s), capped by `RuntimeLimits.tool_timeout_seconds` — and returns the tool's real result. Only if the task outlives *that* does it return `auto_backgrounded: true`, with `background_reader_available: false` and **`success=False`**, so the caller sees an error rather than a handle it cannot redeem. |

Nothing reaps the task left running in that last case. `shutdown()` calls
`ThreadPoolExecutor.shutdown(wait=False)`, which refuses new work but does not
cancel a task already running, so it runs to its own completion and goes when
the runner process does — or with `cgroup.kill` at teardown, on a session
confined to a cgroup. That is deliberate: killing a half-finished install or
build is the worse failure. A deployment that needs a hard bound sets
`RuntimeLimits.tool_timeout_seconds`, which is both the cap on the wait and the
layer that actually enforces a deadline on the command.

The failure this closes was silent: an agent whose profile omitted `background`
was handed nine `task_id`s it had no tool for, `is_error` was `false` every
time, and it burned roughly a quarter of its run inventing a `nohup` + `sleep`
+ `cat` polling protocol — at one point guessing the task id was a path under
`/tmp`.

## Making Plugins Background-Capable

Plugins can implement the `BackgroundCapable` protocol to support background execution.

### Using the Mixin (Recommended)

```python
from shared.plugins.background import BackgroundCapableMixin

class MyPlugin(BackgroundCapableMixin):
    def __init__(self):
        super().__init__(max_workers=4)

    @property
    def name(self) -> str:
        return "my_plugin"

    def supports_background(self, tool_name: str) -> bool:
        return tool_name in ["slow_tool", "another_slow_tool"]

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        if tool_name == "slow_tool":
            return 10.0  # Auto-background after 10 seconds
        return None  # No auto-background for other tools

    def estimate_duration(self, tool_name: str, arguments: dict) -> Optional[float]:
        # Help model make decisions
        if tool_name == "slow_tool" and "large" in arguments.get("type", ""):
            return 60.0
        return None
```

### Implementing the Protocol Directly

For more control, implement `BackgroundCapable` directly:

```python
from shared.plugins.background.protocol import BackgroundCapable, TaskHandle, TaskStatus

class MyCustomPlugin:
    # ... implement all protocol methods ...

    def supports_background(self, tool_name: str) -> bool: ...
    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]: ...
    def estimate_duration(self, tool_name: str, arguments: dict) -> Optional[float]: ...
    def start_background(self, tool_name: str, arguments: dict, timeout: Optional[float]) -> TaskHandle: ...
    def get_status(self, task_id: str) -> TaskStatus: ...
    def get_result(self, task_id: str, wait: bool) -> TaskResult: ...
    def cancel(self, task_id: str) -> bool: ...
    def list_tasks(self) -> List[TaskHandle]: ...
    def cleanup_completed(self, max_age_seconds: float) -> int: ...
    def register_running_task(self, future: Future, tool_name: str, arguments: dict) -> TaskHandle: ...
```

## Task Lifecycle

```
┌─────────┐     ┌─────────┐     ┌───────────┐
│ PENDING │────>│ RUNNING │────>│ COMPLETED │
└─────────┘     └────┬────┘     └───────────┘
                     │
                     ├──────────>┌────────┐
                     │           │ FAILED │
                     │           └────────┘
                     │
                     └──────────>┌───────────┐
                                 │ CANCELLED │
                                 └───────────┘
```

## Configuration

The background plugin discovers capable plugins automatically. No configuration is required.

Plugins control their own behavior:
- `max_workers` - Maximum concurrent background tasks (in mixin constructor)
- `get_auto_background_threshold()` - Per-tool thresholds for auto-backgrounding

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         JaatoClient                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      ToolExecutor                         │  │
│  │  - Checks for auto-background threshold                   │  │
│  │  - Executes with timeout                                  │  │
│  │  - Converts to background on timeout                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    PluginRegistry                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │  │
│  │  │  CLIPlugin  │  │  MCPPlugin  │  │ BackgroundPlugin  │  │  │
│  │  │ (capable)   │  │ (capable)   │  │  (orchestrator)   │  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────┬─────────┘  │  │
│  │         │                │                   │             │  │
│  │         └────────────────┼───────────────────┘             │  │
│  │                          │                                 │  │
│  │              ┌───────────▼───────────┐                     │  │
│  │              │   BackgroundCapable   │                     │  │
│  │              │       Protocol        │                     │  │
│  │              └───────────────────────┘                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Design Document

For detailed design information, see [Background Task Processing Design](../../../docs/design/background-task-processing.md).
