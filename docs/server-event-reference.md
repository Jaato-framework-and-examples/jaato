# Server Event Reference

Complete reference of all events in the jaato server event protocol. Events are JSON-serializable dataclasses defined in `jaato-sdk/jaato_sdk/events.py`, dispatched via `server.emit()`, and delivered to clients over IPC/WebSocket.

All events inherit from `Event`:
```
Event
├── type: EventType       # Enum string identifier
└── timestamp: str        # ISO 8601 UTC timestamp (auto-generated)
```

---

## Connection & Session Management

### `ConnectedEvent`
**Wire type:** `connected`
**Dispatch source:** IPC server (`ipc.py`) / WebSocket server (`websocket.py`)
**Trigger:** Client connects

| Field | Type | Description |
|-------|------|-------------|
| `protocol_version` | `str` | Protocol version (default `"1.0"`) |
| `server_info` | `Dict[str, Any]` | Server metadata (includes `server_version`) |

---

### `SessionInfoEvent`
**Wire type:** `session.info`
**Dispatch source:** `JaatoServer.initialize()`, `handle_model_command()`
**Trigger:** Session initialization complete, or model switched

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Current session ID |
| `session_name` | `str` | Session display name |
| `model_provider` | `str` | Active provider name |
| `model_name` | `str` | Active model name |
| `profile_name` | `Optional[str]` | Agent profile used to create session |
| `sessions` | `List[Dict]` | All available sessions (`{id, name, model_provider, model_name, is_loaded, client_count, turn_count}`) |
| `tools` | `List[Dict]` | All tools with status (`{name, description, enabled, plugin}`) |
| `models` | `List[str]` | Available model names |
| `user_inputs` | `List[str]` | Command history for prompt restoration |
| `memories` | `List[Dict]` | Memory entries (`{id, description, tags}`) |
| `sandbox_paths` | `List[Dict]` | Sandbox paths (`{path, description}`) |
| `services` | `List[Dict]` | Discovered services (`{name, methods}`) |

---

### `SessionListEvent`
**Wire type:** `session.list`
**Dispatch source:** Daemon (`__main__.py`)
**Trigger:** Client sends `session list` command

| Field | Type | Description |
|-------|------|-------------|
| `sessions` | `List[Dict]` | Session list (`{id, name, created_at, last_active, ...}`) |

---

### `SessionProfilesEvent`
**Wire type:** `session.profiles`
**Dispatch source:** Daemon (`__main__.py`)
**Trigger:** Client sends `session profiles` command

| Field | Type | Description |
|-------|------|-------------|
| `profiles` | `List[Dict]` | Profile summaries (`{name, description, model, provider, icon_name, plugins}`) |

---

### `SessionDescriptionUpdatedEvent`
**Wire type:** `session.description_updated`
**Dispatch source:** `JaatoServer` callback on `session_describe` tool
**Trigger:** Model calls `session_describe` tool

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Session ID |
| `description` | `str` | New description |

---

### `InitProgressEvent`
**Wire type:** `init.progress`
**Dispatch source:** `JaatoServer._emit_init_progress()`
**Trigger:** During server initialization steps

| Field | Type | Description |
|-------|------|-------------|
| `step` | `str` | Step name (e.g., `"Loading plugins"`) |
| `status` | `str` | `"running"`, `"done"`, or `"error"` |
| `message` | `str` | Optional details (e.g., error message) |
| `step_number` | `int` | Current step (1-based) |
| `total_steps` | `int` | Total number of steps |

---

## Agent Lifecycle

### `AgentCreatedEvent`
**Wire type:** `agent.created`
**Dispatch source:** `ServerAgentHooks.on_agent_created()`, `SessionManager._restore_subagent_states()`
**Trigger:** New agent/subagent spawned, or subagent restored after session recovery

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Unique ID (`"main"`, `"subagent_1"`, `"parent.child"`) |
| `agent_name` | `str` | Display name |
| `agent_type` | `str` | `"main"` or `"subagent"` |
| `profile_name` | `Optional[str]` | Profile name if subagent |
| `parent_agent_id` | `Optional[str]` | Parent's ID if nested subagent |
| `icon_lines` | `Optional[List[str]]` | Custom ASCII art icon (3 lines) |
| `created_at` | `Optional[str]` | ISO creation timestamp |

---

### `AgentOutputEvent`
**Wire type:** `agent.output`
**Dispatch source:** `ServerAgentHooks.on_agent_output()`, `on_tool_call_start()` (intent args), `_setup_plan_hooks()`, `send_message()` (user echo)
**Trigger:** Model streaming text, non-model output, formatter flush, plan reporter text, user message echo

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent produced output |
| `source` | `str` | `"model"`, `"tool"`, `"system"`, or plugin name |
| `text` | `str` | Output text content |
| `mode` | `str` | `"write"` (new block) or `"append"` (continue previous) |

---

### `AgentStatusChangedEvent`
**Wire type:** `agent.status_changed`
**Dispatch source:** `ServerAgentHooks.on_agent_status_changed()`, `_on_stop_request()`, `SessionManager._recover_interrupted_turn()`
**Trigger:** Agent status transitions, user stop request, interrupted turn recovery

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `status` | `str` | `"active"`, `"idle"`, `"done"`, or `"error"` |
| `error` | `Optional[str]` | Error message if status is `"error"` |

---

### `AgentCompletedEvent`
**Wire type:** `agent.completed`
**Dispatch source:** `ServerAgentHooks.on_agent_completed()`
**Trigger:** Agent finishes all work

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent completed |
| `completed_at` | `str` | ISO completion timestamp |
| `success` | `bool` | Whether agent succeeded |
| `token_usage` | `Optional[Dict[str, int]]` | `{prompt_tokens, output_tokens, total_tokens}` |
| `turns_used` | `Optional[int]` | Number of conversation turns used |

---

## Tool Execution

### `ToolCallStartEvent`
**Wire type:** `tool.call_start`
**Dispatch source:** `ServerAgentHooks.on_tool_call_start()`
**Trigger:** Tool begins execution

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent initiated the call |
| `tool_name` | `str` | Name of the tool |
| `tool_args` | `Dict[str, Any]` | Arguments passed to tool |
| `call_id` | `Optional[str]` | Unique ID for parallel execution correlation |

---

### `ToolCallEndEvent`
**Wire type:** `tool.call_end`
**Dispatch source:** `ServerAgentHooks.on_tool_call_end()`
**Trigger:** Tool completes execution

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `tool_name` | `str` | Tool name |
| `call_id` | `Optional[str]` | Correlation ID |
| `success` | `bool` | Whether execution succeeded |
| `duration_seconds` | `float` | Execution time |
| `error_message` | `Optional[str]` | Error details if failed |
| `backgrounded` | `bool` | True if auto-backgrounded (still producing output) |
| `continuation_id` | `Optional[str]` | Session ID for interactive tool continuations (e.g., shell) |
| `show_output` | `Optional[bool]` | Whether to render output in main panel (`None` = default True) |
| `show_popup` | `Optional[bool]` | Whether to track tool output popup (`None` = default True) |

---

### `ToolOutputEvent`
**Wire type:** `tool.output`
**Dispatch source:** `ServerAgentHooks.on_tool_output()`
**Trigger:** Live output chunk from running tool

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent's tool |
| `call_id` | `str` | **Required** — correlates with `ToolCallStartEvent` |
| `chunk` | `str` | Output text chunk (may contain newlines) |

---

### `ToolStatusEvent`
**Wire type:** `tools.status`
**Dispatch source:** Daemon (`__main__.py`)
**Trigger:** `tools list`, `tools enable`, `tools disable` commands

| Field | Type | Description |
|-------|------|-------------|
| `tools` | `List[Dict]` | Tool list (`{name, description, enabled, plugin}`) |
| `message` | `str` | Optional result message for enable/disable operations |

---

## Plan Management

### `PlanUpdatedEvent`
**Wire type:** `plan.updated`
**Dispatch source:** `_setup_plan_hooks()` update_callback (via `LivePlanReporter`)
**Trigger:** Plan created or steps updated (from todo/subagent plugin)

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent created/updated the plan |
| `plan_name` | `str` | Display name of the plan |
| `steps` | `List[Dict]` | Step list (see schema below) |

**Step schema:**
```json
{
  "content": "Step description",
  "status": "pending | in_progress | completed",
  "active_form": "optional active form indicator",
  "step_id": "optional step identifier",
  "blocked_by": "optional cross-agent dependency",
  "depends_on": "optional dependency info",
  "received_outputs": "optional received outputs from dependencies"
}
```

---

### `PlanClearedEvent`
**Wire type:** `plan.cleared`
**Dispatch source:** `_setup_plan_hooks()` clear_callback (via `LivePlanReporter`)
**Trigger:** Plan completed/cleared

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent completed the plan |

---

## Context & Token Tracking

### `ContextUpdatedEvent`
**Wire type:** `context.updated`
**Dispatch source:** `ServerAgentHooks.on_agent_context_updated()`, `on_agent_gc_config()`, `SessionManager._restore_session_state()`
**Trigger:** Token usage changes, GC config set, context restored from disk

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `total_tokens` | `int` | Total tokens used |
| `prompt_tokens` | `int` | Prompt tokens |
| `output_tokens` | `int` | Output tokens |
| `context_limit` | `int` | Maximum context window |
| `percent_used` | `float` | Percentage of context used |
| `tokens_remaining` | `int` | Tokens still available |
| `turns` | `int` | Number of conversation turns |
| `gc_threshold` | `Optional[float]` | GC trigger threshold % |
| `gc_strategy` | `Optional[str]` | GC strategy name (e.g., `"truncate"`, `"hybrid"`) |
| `gc_target_percent` | `Optional[float]` | Target usage after GC |
| `gc_continuous_mode` | `bool` | True if GC runs after every turn |

---

### `TurnCompletedEvent`
**Wire type:** `turn.completed`
**Dispatch source:** `ServerAgentHooks.on_agent_turn_completed()`
**Trigger:** Conversation turn finishes (all tool calls done)

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `turn_number` | `int` | Turn index (0-based) |
| `prompt_tokens` | `int` | Tokens consumed by prompt |
| `output_tokens` | `int` | Tokens generated in response |
| `total_tokens` | `int` | Sum of prompt + output |
| `duration_seconds` | `float` | Time taken for the turn |
| `function_calls` | `List[Dict]` | Calls made during turn (`{name, duration_seconds}`) |
| `formatted_text` | `Optional[str]` | Syntax-highlighted result text |
| `cache_read_tokens` | `Optional[int]` | Tokens served from prompt cache |
| `cache_creation_tokens` | `Optional[int]` | Tokens written into cache |

---

### `TurnProgressEvent`
**Wire type:** `turn.progress`
**Dispatch source:** `ServerAgentHooks.on_turn_progress()`
**Trigger:** Incremental progress during turn execution

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `total_tokens` | `int` | Current total tokens used |
| `prompt_tokens` | `int` | Current prompt tokens |
| `output_tokens` | `int` | Current output tokens |
| `context_limit` | `int` | Maximum context window |
| `percent_used` | `float` | Percentage of context used |
| `tokens_remaining` | `int` | Tokens still available |
| `pending_tool_calls` | `int` | Tool calls remaining |
| `cache_read_tokens` | `Optional[int]` | Tokens from prompt cache |
| `cache_creation_tokens` | `Optional[int]` | Tokens written to cache |

---

### `InstructionBudgetEvent`
**Wire type:** `instruction_budget.updated`
**Dispatch source:** `ServerAgentHooks.on_agent_instruction_budget_updated()`, instruction budget callback, `SessionManager._restore_session_state()`
**Trigger:** Budget snapshot changes, budget restored from disk

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `budget_snapshot` | `Dict[str, Any]` | Full budget breakdown (see below) |

**Budget snapshot schema:**
```json
{
  "session_id": "...",
  "agent_id": "...",
  "agent_type": "main | subagent",
  "context_limit": 128000,
  "total_tokens": 45000,
  "utilization_percent": 35.2,
  "gc_eligible_tokens": 30000,
  "locked_tokens": 15000,
  "preservable_tokens": 5000,
  "available_tokens": 83000,
  "gc_headroom_percent": 44.8,
  "entries": [
    {"source": "system", "tokens": 5000, "children": [...]},
    {"source": "plugin", "tokens": 8000, "children": [...]},
    {"source": "conversation", "tokens": 32000, "children": [...]}
  ]
}
```

---

## Permission Flow

### `PermissionRequestedEvent`
**Wire type:** `permission.requested`
**Dispatch source:** `_setup_permission_hooks()` on_permission_requested
**Trigger:** Tool needs user permission (includes pre-formatted diff for file edits)

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent is requesting |
| `request_id` | `str` | Unique request ID |
| `tool_name` | `str` | Tool requesting permission |
| `tool_args` | `Dict[str, Any]` | Arguments passed to tool |
| `response_options` | `List[Dict]` | Valid responses (`{key, label, action, description?}`) |
| `prompt_lines` | `Optional[List[str]]` | Pre-formatted prompt (with diff) |
| `format_hint` | `Optional[str]` | `"diff"` for colored diff display |
| `warnings` | `Optional[str]` | Security/analysis warnings |
| `warning_level` | `Optional[str]` | `"info"`, `"warning"`, or `"error"` |

---

### `PermissionInputModeEvent`
**Wire type:** `permission.input_mode`
**Dispatch source:** `_setup_permission_hooks()` on_permission_requested
**Trigger:** After permission content emitted — signals client to enter input mode

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent is requesting |
| `request_id` | `str` | Unique request ID |
| `tool_name` | `str` | Tool name |
| `call_id` | `Optional[str]` | For matching tool call (parallel execution) |
| `response_options` | `List[Dict]` | Valid responses (`{key, label, action, description?}`) |
| `tool_args` | `Optional[Dict]` | Tool arguments for client-side editing |
| `editable_metadata` | `Optional[Dict]` | `{parameters: [...], format: "yaml"|"json"|"text"}` |

---

### `PermissionResolvedEvent`
**Wire type:** `permission.resolved`
**Dispatch source:** `_setup_permission_hooks()` on_permission_resolved
**Trigger:** User grants or denies permission

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `request_id` | `str` | Request ID |
| `tool_name` | `str` | Tool name |
| `granted` | `bool` | Whether permission was granted |
| `method` | `str` | `"user"`, `"whitelist"`, `"blacklist"`, or `"default"` |

---

### `PermissionStatusEvent`
**Wire type:** `permission.status`
**Dispatch source:** `JaatoServer.emit_permission_status()`
**Trigger:** Permission policy changed (after permission commands)

| Field | Type | Description |
|-------|------|-------------|
| `effective_default` | `str` | `"allow"`, `"deny"`, or `"ask"` |
| `suspension_scope` | `Optional[str]` | `"turn"`, `"idle"`, `"session"`, or `None` |

---

## Clarification Flow

### `ClarificationRequestedEvent`
**Wire type:** `clarification.requested`
**Dispatch source:** `_setup_clarification_hooks()`
**Trigger:** Model starts a clarification session

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `request_id` | `str` | Request ID |
| `tool_name` | `str` | Tool that triggered clarification |
| `context_lines` | `List[str]` | Context text lines |
| `total_questions` | `int` | Number of questions to ask |

---

### `ClarificationQuestionEvent`
**Wire type:** `clarification.question`
**Dispatch source:** `_setup_clarification_hooks()`
**Trigger:** Individual clarification question displayed

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `request_id` | `str` | Request ID |
| `question_index` | `int` | Question index (0-based) |
| `total_questions` | `int` | Total questions in session |
| `question_type` | `str` | `"single_choice"`, `"multiple_choice"`, or `"free_text"` |
| `question_text` | `str` | The question |
| `options` | `Optional[List[Dict]]` | Options for choice questions (`{key, label}`) |

---

### `ClarificationInputModeEvent`
**Wire type:** `clarification.input_mode`
**Dispatch source:** `_setup_clarification_hooks()` on_question_displayed
**Trigger:** After question content emitted — signals client to accept answer

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `request_id` | `str` | Request ID |
| `tool_name` | `str` | Tool name |
| `question_index` | `int` | Current question index |
| `total_questions` | `int` | Total questions |

---

### `ClarificationResolvedEvent`
**Wire type:** `clarification.resolved`
**Dispatch source:** `_setup_clarification_hooks()` on_clarification_resolved
**Trigger:** All clarification questions answered

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `request_id` | `str` | Request ID |
| `tool_name` | `str` | Tool name |
| `qa_pairs` | `List[List[str]]` | List of `[question_text, answer_text]` pairs |

---

## Reference Selection Flow

### `ReferenceSelectionRequestedEvent`
**Wire type:** `reference_selection.requested`
**Dispatch source:** `_setup_reference_selection_hooks()` on_selection_requested
**Trigger:** Ambiguous reference needs user disambiguation

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `request_id` | `str` | Request ID |
| `tool_name` | `str` | Tool name |
| `prompt_lines` | `List[str]` | Pre-formatted selection prompt |

---

### `ReferenceSelectionResolvedEvent`
**Wire type:** `reference_selection.resolved`
**Dispatch source:** `_setup_reference_selection_hooks()` on_selection_resolved
**Trigger:** User picks an option

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent |
| `request_id` | `str` | Request ID |
| `tool_name` | `str` | Tool name |
| `selected_ids` | `List[str]` | Selected reference IDs |

---

## Workspace Mismatch Flow

### `WorkspaceMismatchRequestedEvent`
**Wire type:** `workspace_mismatch.requested`
**Dispatch source:** Session manager during attach
**Trigger:** Client's workspace path differs from session's workspace

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Request ID |
| `session_id` | `str` | Session ID |
| `session_workspace` | `str` | Session's workspace path |
| `client_workspace` | `str` | Client's workspace path |
| `response_options` | `List[Dict]` | Valid responses (`{key, label, action, description?}`) |
| `prompt_lines` | `List[str]` | Pre-formatted prompt |

---

### `WorkspaceMismatchResolvedEvent`
**Wire type:** `workspace_mismatch.resolved`
**Dispatch source:** Session manager
**Trigger:** User resolves workspace mismatch

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Request ID |
| `session_id` | `str` | Session ID |
| `action` | `str` | `"switch"`, `"new_session"`, or `"cancel"` |
| `new_session_id` | `Optional[str]` | Set if action is `"new_session"` |

---

## Post-Auth Setup Flow

### `PostAuthSetupEvent`
**Wire type:** `auth.setup`
**Dispatch source:** Daemon after auth command succeeds
**Trigger:** Successful authentication — offers session setup wizard

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Request ID |
| `provider_name` | `str` | e.g., `"zhipuai"` |
| `provider_display_name` | `str` | e.g., `"Zhipu AI (Z.AI)"` |
| `available_models` | `List[Dict]` | `{name, description}` per model |
| `has_active_session` | `bool` | Whether a session is already running |
| `current_provider` | `str` | Only if `has_active_session` |
| `current_model` | `str` | Only if `has_active_session` |
| `workspace_path` | `str` | Current workspace |

---

## Mid-Turn Interaction

### `MidTurnPromptQueuedEvent`
**Wire type:** `mid_turn_prompt.queued`
**Dispatch source:** `JaatoServer.send_message()`
**Trigger:** User sends message while model is running

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The queued message |
| `position_in_queue` | `int` | 0-based queue position |

---

### `MidTurnPromptInjectedEvent`
**Wire type:** `mid_turn_prompt.injected`
**Dispatch source:** `JaatoServer._on_mid_turn_prompt()`
**Trigger:** Queued prompt injected into session at natural pause point

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The injected message |

---

### `MidTurnInterruptEvent`
**Wire type:** `mid_turn_prompt.interrupt`
**Dispatch source:** `JaatoServer._on_interrupt_request()`
**Trigger:** Streaming interrupted to process user prompt

| Field | Type | Description |
|-------|------|-------------|
| `partial_response_chars` | `int` | Characters generated before interrupt |
| `user_prompt_preview` | `str` | First 100 chars of user's prompt |

---

## System & Error

### `SystemMessageEvent`
**Wire type:** `system.message`
**Dispatch source:** Various methods in `JaatoServer`, `SessionManager`
**Trigger:** Auth status, recovery info, command results, setup messages

| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | Message text |
| `style` | `str` | `"info"`, `"warning"`, `"error"`, `"success"`, or `"dim"` |

---

### `ErrorEvent`
**Wire type:** `error`
**Dispatch source:** Various error handlers in `JaatoServer`
**Trigger:** Init failures, provider errors, missing config

| Field | Type | Description |
|-------|------|-------------|
| `error` | `str` | Error message |
| `error_type` | `str` | Exception class name |
| `recoverable` | `bool` | Whether the error is recoverable |

---

### `HelpTextEvent`
**Wire type:** `help.text`
**Dispatch source:** Daemon (`__main__.py`)
**Trigger:** `tools help` or similar help commands

| Field | Type | Description |
|-------|------|-------------|
| `lines` | `List[tuple]` | List of `(text, style)` tuples for pager display |

---

### `RetryEvent`
**Wire type:** `retry`
**Dispatch source:** `JaatoServer._on_retry_request()`
**Trigger:** API transient error with exponential backoff retry

| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | Human-readable retry message |
| `attempt` | `int` | Current attempt number (1-indexed) |
| `max_attempts` | `int` | Maximum attempts configured |
| `delay` | `float` | Delay in seconds before next attempt |
| `error_type` | `str` | `"rate_limit"` or `"transient"` |

---

## Session Recovery

### `InterruptedTurnRecoveredEvent`
**Wire type:** `session.interrupted_turn_recovered`
**Dispatch source:** `SessionManager._recover_interrupted_turn()`
**Trigger:** Server restart recovers pending tool calls

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Session ID |
| `agent_id` | `str` | Agent ID |
| `recovered_calls` | `int` | Number of tool calls recovered |
| `action_taken` | `str` | e.g., `"synthetic_error"` |

---

## Workspace File Monitoring

### `WorkspaceFilesChangedEvent`
**Wire type:** `workspace.files_changed`
**Dispatch source:** `SessionManager._on_workspace_files_changed()`
**Trigger:** File system changes in workspace (debounced)

| Field | Type | Description |
|-------|------|-------------|
| `changes` | `List[Dict]` | `{path: str, status: "created"|"modified"|"deleted"}` |

---

### `WorkspaceFilesSnapshotEvent`
**Wire type:** `workspace.files_snapshot`
**Dispatch source:** `SessionManager._emit_to_client()`
**Trigger:** Client reconnects — full state snapshot

| Field | Type | Description |
|-------|------|-------------|
| `files` | `List[Dict]` | `{path: str, status: "created"|"modified"|"deleted"}` |
| `total` | `int` | Count of non-deleted entries |

---

## Workspace Management

### `WorkspaceListEvent`
**Wire type:** `workspace.list_response`
**Dispatch source:** WebSocket handler
**Trigger:** `workspace list` command

| Field | Type | Description |
|-------|------|-------------|
| `root` | `str` | Absolute path to workspace root |
| `workspaces` | `List[Dict]` | `{name, configured, provider?, model?, last_accessed?}` |

---

### `WorkspaceCreatedEvent`
**Wire type:** `workspace.created`
**Dispatch source:** WebSocket handler
**Trigger:** `workspace create` command

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Relative path from workspace root |
| `path` | `str` | Absolute path |

---

### `ConfigStatusEvent`
**Wire type:** `config.status`
**Dispatch source:** WebSocket handler
**Trigger:** `workspace select` command

| Field | Type | Description |
|-------|------|-------------|
| `workspace` | `str` | Workspace name |
| `configured` | `bool` | Has valid provider config |
| `provider` | `Optional[str]` | Current provider |
| `model` | `Optional[str]` | Current model |
| `available_providers` | `List[str]` | Providers that can be configured |
| `missing_fields` | `List[str]` | Fields needed to complete config |

---

### `ConfigUpdatedEvent`
**Wire type:** `config.updated`
**Dispatch source:** WebSocket handler
**Trigger:** `config update` command

| Field | Type | Description |
|-------|------|-------------|
| `workspace` | `str` | Workspace name |
| `provider` | `str` | New provider |
| `model` | `Optional[str]` | New model |
| `success` | `bool` | Whether update succeeded |
| `error` | `Optional[str]` | Error message if failed |

---

## Command Discovery

### `CommandListEvent`
**Wire type:** `command.list`
**Dispatch source:** IPC server
**Trigger:** Client requests available commands

| Field | Type | Description |
|-------|------|-------------|
| `commands` | `List[Dict]` | `{name, description, subcommands?}` |

---

### `CommandListRefreshEvent`
**Wire type:** `command.list_refresh`
**Dispatch source:** `JaatoServer` after state-changing commands
**Trigger:** Commands that change completion state (e.g., `references select/unselect`)

*(No additional fields — signal-only event)*

---

## Memory & Metadata

### `MemoryListEvent`
**Wire type:** `memory.list`
**Dispatch source:** `JaatoServer.execute_command()`
**Trigger:** `memory` command

| Field | Type | Description |
|-------|------|-------------|
| `memories` | `List[Dict]` | `{id, description, tags}` |

---

### `SandboxPathsEvent`
**Wire type:** `sandbox.paths`
**Dispatch source:** `JaatoServer.execute_command()`
**Trigger:** Sandbox add/remove commands

| Field | Type | Description |
|-------|------|-------------|
| `paths` | `List[Dict]` | `{path, description}` |

---

### `ServiceListEvent`
**Wire type:** `service.list`
**Dispatch source:** `ServerAgentHooks.on_tool_call_end()` (after `discover_service`), `execute_command()` (`services list`)
**Trigger:** Service discovery or listing

| Field | Type | Description |
|-------|------|-------------|
| `services` | `List[Dict]` | `{name, methods}` |

---

## History

### `HistoryEvent`
**Wire type:** `history`
**Dispatch source:** Daemon (`__main__.py`)
**Trigger:** `history` command

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent's history (default `"main"`) |
| `history` | `List[Dict]` | Serialized `Message` objects |
| `turn_accounting` | `List[Dict]` | `{prompt, output, total}` per turn |

---

## Peer Channel Events (Server-to-Server)

### `PeerHeartbeatEvent`
**Wire type:** `peer.heartbeat`
**Dispatch source:** Peer channel at configurable interval
**Trigger:** Periodic heartbeat between peer servers

| Field | Type | Description |
|-------|------|-------------|
| `server_id` | `str` | Server identifier |
| `server_name` | `str` | Server display name |
| `server_version` | `str` | Server version |
| `active_sessions` | `int` | Running session count |
| `active_agents` | `int` | Running agent count |
| `available_providers` | `List[str]` | Configured providers |
| `available_models` | `List[str]` | Available models |
| `tags` | `List[str]` | Server tags |
| `cpu_percent` | `float` | CPU usage |
| `memory_percent` | `float` | Memory usage |
| `uptime_seconds` | `float` | Server uptime |
| `trust_state` | `str` | Reliability self-report (default `"trusted"`) |
| `success_rate_1h` | `float` | 1-hour success rate |
| `escalated_tools` | `int` | Escalated tool count |

---

### `PeerSpawnRequestEvent`
**Wire type:** `peer.spawn_request`
**Dispatch source:** Origin server when model calls `spawn_subagent(server=...)`
**Trigger:** Remote subagent delegation

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Spawn lifecycle correlation ID |
| `origin_server` | `str` | Requesting server ID |
| `agent_name` | `str` | Subagent name |
| `task` | `str` | Task description |
| `context` | `str` | Context for the subagent |
| `profile_json` | `str` | Serialized `SubagentProfile` |
| `inline_config_json` | `str` | Inline config overrides |
| `workspace_git_url` | `str` | Git URL for workspace replication |
| `workspace_branch` | `str` | Git branch |
| `workspace_commit` | `str` | Git commit SHA |
| `workspace_temp_branch` | `str` | Temporary branch for replication |

---

### `PeerSpawnAcceptedEvent`
**Wire type:** `peer.spawn_accepted`
**Dispatch source:** Remote peer after accepting spawn
**Trigger:** Remote peer created ephemeral session

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Correlation ID |
| `remote_agent_id` | `str` | Agent ID on remote server |

---

### `PeerSpawnRejectedEvent`
**Wire type:** `peer.spawn_rejected`
**Dispatch source:** Remote peer
**Trigger:** Remote peer rejects spawn (capacity, missing provider, etc.)

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Correlation ID |
| `reason` | `str` | Human-readable rejection reason |

---

### `PeerAgentOutputEvent`
**Wire type:** `peer.agent_output`
**Dispatch source:** Remote server during subagent execution
**Trigger:** Remote subagent produces output

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Correlation ID |
| `remote_agent_id` | `str` | Agent ID on remote server |
| `text` | `str` | Output text |
| `source` | `str` | `"model"` or `"tool"` |

---

### `PeerAgentCompletedEvent`
**Wire type:** `peer.agent_completed`
**Dispatch source:** Remote server
**Trigger:** Remote subagent finished

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Correlation ID |
| `remote_agent_id` | `str` | Agent ID on remote server |
| `success` | `bool` | Whether completed normally |
| `summary` | `str` | Brief result description |
| `error` | `str` | Error message (only when `success=False`) |
| `workspace_modified` | `bool` | Whether workspace was changed |

---

### `PeerStopRequestEvent`
**Wire type:** `peer.stop_request`
**Dispatch source:** Origin server
**Trigger:** Parent session wants to cancel remote subagent

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Correlation ID |
| `remote_agent_id` | `str` | Agent to stop |

---

### `PeerStopAcknowledgedEvent`
**Wire type:** `peer.stop_acknowledged`
**Dispatch source:** Remote peer
**Trigger:** Remote peer processed stop request

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Correlation ID |
| `remote_agent_id` | `str` | Agent that was stopped |

---

## Client → Server Requests

These events flow from client to server and are **not** emitted via `server.emit()`.

### `SendMessageRequest`
**Wire type:** `message.send`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Message text |
| `attachments` | `List[Dict]` | `{type: "file", path: "..."}` or `{type: "image", data: "base64..."}` |

### `PermissionResponseRequest`
**Wire type:** `permission.response`

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Matching request ID |
| `response` | `str` | `"y"`, `"n"`, `"a"`, `"never"`, etc. |
| `edited_arguments` | `Optional[Dict]` | Edited tool args (when response is `"e"`) |

### `ClarificationResponseRequest`
**Wire type:** `clarification.response`

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Matching request ID |
| `question_index` | `int` | Which question |
| `response` | `str` | User's answer |

### `ReferenceSelectionResponseRequest`
**Wire type:** `reference_selection.response`

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Matching request ID |
| `response` | `str` | Selection (e.g., `"1,3,4"`, `"all"`, `"none"`) |

### `WorkspaceMismatchResponseRequest`
**Wire type:** `workspace_mismatch.response`

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Matching request ID |
| `response` | `str` | `"s"` (switch), `"n"` (new session), `"c"` (cancel) |

### `PostAuthSetupResponse`
**Wire type:** `auth.setup_response`

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Matching request ID |
| `connect` | `bool` | Whether to create/switch session |
| `model_name` | `str` | Selected model (if `connect=True`) |
| `persist_env` | `bool` | Whether to save provider/model to `.env` |

### `StopRequest`
**Wire type:** `session.stop`

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `Optional[str]` | Agent to stop (`None` = all agents) |

### `CommandRequest`
**Wire type:** `command.execute`

| Field | Type | Description |
|-------|------|-------------|
| `command` | `str` | Command name |
| `args` | `List[str]` | Command arguments |

### `GetInstructionBudgetRequest`
**Wire type:** `instruction_budget.request`

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `Optional[str]` | Agent to query (`None` = main agent) |

### `CommandListRequest`
**Wire type:** `command.list_request`

*(No additional fields)*

### `ToolDisableRequest`
**Wire type:** `tools.disable`

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Tool to disable |

### `HistoryRequest`
**Wire type:** `history.request`

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Which agent's history (default `"main"`) |

### `WorkspaceListRequest`
**Wire type:** `workspace.list`

*(No additional fields)*

### `WorkspaceCreateRequest`
**Wire type:** `workspace.create`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name for the new workspace |

### `WorkspaceSelectRequest`
**Wire type:** `workspace.select`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Workspace name |

### `ConfigUpdateRequest`
**Wire type:** `config.update`

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Provider name |
| `model` | `Optional[str]` | Model name |
| `api_key` | `Optional[str]` | API key (for non-OAuth providers) |

### `ClientConfigRequest`
**Wire type:** `client.config`

| Field | Type | Description |
|-------|------|-------------|
| `trace_log_path` | `Optional[str]` | `JAATO_TRACE_LOG` path |
| `provider_trace_log` | `Optional[str]` | Provider trace log path |
| `working_dir` | `Optional[str]` | Client's working directory |
| `env_file` | `Optional[str]` | Path to client's `.env` file |
| `presentation` | `Optional[Dict]` | `PresentationContext` as dict |

---

## Dispatch Architecture Summary

All server→client events funnel through `server.emit()`, which dispatches to connected clients via IPC (length-prefixed framing) or WebSocket.

**Dispatch sources by category:**

| Source | Events | Location |
|--------|--------|----------|
| `ServerAgentHooks` (inner class) | Agent lifecycle, output, tools, context, budget | `core.py` lines 1419–1750 |
| Permission hooks | Permission request/resolve/status | `core.py` `_setup_permission_hooks()` |
| Clarification hooks | Clarification request/question/resolve | `core.py` `_setup_clarification_hooks()` |
| Reference selection hooks | Selection request/resolve | `core.py` `_setup_reference_selection_hooks()` |
| `LivePlanReporter` callbacks | Plan updated/cleared | `core.py` `_setup_plan_hooks()` |
| `JaatoServer` direct | Mid-turn, errors, system messages, command results | Various methods in `core.py` |
| `SessionManager` | Session restore, file monitoring, recovery | `session_manager.py` |
| Daemon (`__main__.py`) | Session list, profiles, history, tool status | Command handlers |
| IPC/WebSocket servers | Connected, command list | `ipc.py`, `websocket.py` |
