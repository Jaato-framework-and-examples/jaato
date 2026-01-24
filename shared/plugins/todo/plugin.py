"""TODO plugin for plan registration and progress reporting.

This plugin enables LLMs to:
1. Register execution plans with ordered steps
2. Report progress on individual steps
3. Query plan status
4. Complete/fail/cancel plans

Progress is reported through configurable transport protocols
(console, webhook, file) matching the permissions plugin pattern.
"""

import os
import tempfile
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .models import (
    PlanStatus, StepStatus, TodoPlan, TodoStep,
    TaskEventType, TaskEvent, TaskRef, EventFilter, Subscription
)
from ..model_provider.types import ToolSchema
from .storage import TodoStorage, create_storage, InMemoryStorage
from .channels import TodoReporter, ConsoleReporter, create_reporter
from .config_loader import load_config, TodoConfig
from .event_bus import TaskEventBus, get_event_bus
from ..base import UserCommand


# Thread-local storage for per-agent context
# This allows each agent (running in its own thread) to have its own agent_name
_thread_local = threading.local()


class TodoPlugin:
    """Plugin that provides plan registration and progress tracking.

    This plugin exposes tools for the LLM to:
    - createPlan: Register a new execution plan with steps
    - updateStep: Report progress on a specific step
    - getPlanStatus: Query current plan state
    - completePlan: Mark a plan as finished

    Progress is reported through configurable reporters (console, webhook, file)
    using the same transport protocol patterns as the permissions plugin.
    """

    def __init__(self):
        self._config: Optional[TodoConfig] = None
        self._storage: Optional[TodoStorage] = None
        self._reporter: Optional[TodoReporter] = None
        self._initialized = False
        # Track current plan per agent (agent_name -> plan_id)
        # This allows multiple agents to have their own active plans
        self._current_plan_ids: Dict[Optional[str], str] = {}

        # Note: session is stored in thread-local storage via set_session()
        # This prevents subagent sessions from overwriting the parent's reference

        # Event bus for cross-agent task collaboration
        self._event_bus: Optional[TaskEventBus] = None
        # Track subscriptions created by this agent (for cleanup)
        self._agent_subscriptions: Dict[Optional[str], List[str]] = {}

    @property
    def _agent_name(self) -> Optional[str]:
        """Get the agent name/ID for the current context.

        Uses thread-local session reference to get agent_id. This ensures
        each thread (main agent vs subagent) sees its own session context.
        """
        session = getattr(_thread_local, 'session', None)
        if session is not None:
            return session.agent_id
        # Fall back to legacy thread-local agent_name
        return getattr(_thread_local, 'agent_name', None)

    @_agent_name.setter
    def _agent_name(self, value: Optional[str]) -> None:
        """Set the agent name for the current thread (legacy)."""
        _thread_local.agent_name = value

    def set_session(self, session: Any) -> None:
        """Set the session for agent context in thread-local storage.

        This is called by the plugin wiring system. Using thread-local storage
        ensures that when subagents (running in different threads) call this,
        they don't overwrite the parent agent's session reference.

        Args:
            session: The JaatoSession instance.
        """
        _thread_local.session = session

    @property
    def name(self) -> str:
        return "todo"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [TODO{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TODO plugin.

        Args:
            config: Optional configuration dict. If not provided, loads from
                   file specified by TODO_CONFIG_PATH or default locations.

                   Config options:
                   - config_path: Path to todo.json file
                   - reporter_type: Type of reporter ("console", "webhook", "file")
                   - reporter_config: Configuration for the reporter
                   - storage_type: Type of storage ("memory", "file", "hybrid")
                   - storage_path: Path for file-based storage
        """
        config = config or {}

        # Extract agent name for trace logging
        self._agent_name = config.get("agent_name")

        # Try to load from file first
        config_path = config.get("config_path")
        try:
            self._config = load_config(config_path)
        except FileNotFoundError:
            # Use defaults
            self._config = TodoConfig()

        # Initialize storage (preserve existing if not explicitly overridden)
        # This allows plans from other agents to persist when a new agent
        # re-initializes the shared plugin
        storage_type_explicit = config.get("storage_type")
        if storage_type_explicit or not self._storage:
            storage_type = storage_type_explicit or self._config.storage_type
            storage_path = config.get("storage_path") or self._config.storage_path
            use_directory = config.get("storage_use_directory", self._config.storage_use_directory)

            try:
                self._storage = create_storage(
                    storage_type=storage_type,
                    path=storage_path,
                    use_directory=use_directory,
                )
            except (ValueError, OSError) as e:
                print(f"Warning: Failed to initialize storage: {e}")
                print("Falling back to in-memory storage")
                self._storage = InMemoryStorage()
        else:
            storage_type = "preserved"

        # Initialize reporter (preserve existing if not explicitly overridden)
        # This allows the LivePlanReporter set by rich_client to survive
        # re-initialization when subagents configure their agent_name
        #
        # Priority:
        # 1. Injected reporter from parent (for subagents inheriting UI)
        # 2. Explicit reporter_type in config
        # 3. Existing reporter (preserve)
        # 4. Create from config/defaults
        injected_reporter = config.get("_injected_reporter")
        reporter_type = config.get("reporter_type")

        if injected_reporter:
            # Use reporter injected by parent (e.g., subagent inheriting LivePlanReporter)
            self._reporter = injected_reporter
            self._trace(f"initialize: using injected reporter {type(injected_reporter).__name__}")
        elif reporter_type or not self._reporter:
            # Only create new reporter if explicitly requested or none exists
            reporter_type = reporter_type or self._config.reporter_type
            reporter_config = config.get("reporter_config") or self._config.to_reporter_config()

            try:
                self._reporter = create_reporter(reporter_type, reporter_config)
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Failed to initialize {reporter_type} reporter: {e}")
                print("Falling back to console reporter")
                self._reporter = ConsoleReporter()

        # Initialize event bus for cross-agent collaboration
        self._event_bus = get_event_bus()
        # Register dependency resolver callback
        self._event_bus.set_dependency_resolver(self._on_dependency_resolved)

        self._initialized = True
        self._trace(f"initialize: storage={storage_type}, reporter={reporter_type}")

    def shutdown(self) -> None:
        """Shutdown the TODO plugin.

        Note: We preserve _current_plan_ids and _storage across shutdown/
        re-initialization because this plugin is shared between multiple
        agents. Each agent's plan data and tracking should persist even
        when another agent re-initializes the plugin with different config.
        """
        self._trace("shutdown: cleaning up (preserving plan tracking and storage)")
        # Don't shutdown reporter - it's shared and may still be in use
        # self._reporter will be replaced in initialize() if needed
        self._initialized = False
        # Do NOT clear _current_plan_ids - it tracks plans per agent
        # Do NOT clear _storage - it contains plans that should persist

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for TODO tools."""
        return [
            ToolSchema(
                name="createPlan",
                description="Step 1: Register a new execution plan with ordered steps. "
                           "Think carefully before calling - only propose plans you can actually "
                           "achieve with available tools. IMPORTANT: Before including any tool "
                           "in a step, verify that tool is available and you are allowed to use it. "
                           "Each step must be specific and actionable. "
                           "After calling this, you MUST call startPlan to get user approval.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Brief summary of the plan (e.g., 'Refactor auth module')"
                        },
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ordered list of specific, actionable step descriptions"
                        }
                    },
                    "required": ["title", "steps"]
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="startPlan",
                description="Step 2: Request user approval to begin executing the plan. "
                           "This MUST be called after createPlan and BEFORE any updateStep calls. "
                           "If the user denies: call completePlan with status='cancelled', "
                           "do NOT create another plan and retry.",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Optional message explaining why this plan is proposed"
                        }
                    },
                    "required": []
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="updateStep",
                description="Step 3: Update the status of a step. Can only be called AFTER "
                           "startPlan has been approved. Use this to report progress as you work.",
                parameters={
                    "type": "object",
                    "properties": {
                        "step_id": {
                            "type": "string",
                            "description": "ID of the step to update (from createPlan response)"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["in_progress", "completed", "failed", "skipped"],
                            "description": "New status for the step"
                        },
                        "result": {
                            "type": "string",
                            "description": "Optional outcome or notes for the step"
                        },
                        "error": {
                            "type": "string",
                            "description": "Error message if step failed"
                        }
                    },
                    "required": ["step_id", "status"]
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="getPlanStatus",
                description="Query current plan state and progress. Can be called at any time.",
                parameters={
                    "type": "object",
                    "properties": {
                        "plan_id": {
                            "type": "string",
                            "description": "ID of the plan (optional, defaults to current plan)"
                        }
                    },
                    "required": []
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="completePlan",
                description="Step 4: Mark the plan as finished. Use 'completed' or 'failed' only "
                           "if the plan was started. Use 'cancelled' if the user rejected startPlan.",
                parameters={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["completed", "failed", "cancelled"],
                            "description": "Final status: 'completed'/'failed' require started plan, "
                                         "'cancelled' for rejected plans"
                        },
                        "summary": {
                            "type": "string",
                            "description": "Optional summary of the outcome"
                        }
                    },
                    "required": ["status"]
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="addStep",
                description="Add a new step to the plan during execution. Can only be called "
                           "AFTER startPlan has been approved.",
                parameters={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of the new step"
                        },
                        "after_step_id": {
                            "type": "string",
                            "description": "Insert after this step ID. If omitted, appends to end."
                        }
                    },
                    "required": ["description"]
                },
                category="planning",
                discoverability="core",
            ),
            # === Cross-agent collaboration tools ===
            ToolSchema(
                name="subscribeToTasks",
                description=(
                    "Subscribe to task events from other agents. "
                    "CALL THIS BEFORE spawning subagents to enable coordination.\n\n"
                    "RECOMMENDED: Subscribe early in your workflow:\n"
                    "  subscribeToTasks(event_types=['plan_created', 'step_completed'])\n\n"
                    "This enables you to:\n"
                    "- See subagent plan structures when they call createPlan\n"
                    "- React when subagents complete steps with outputs\n"
                    "- Add dependent steps that link to their work\n\n"
                    "Events arrive as [SUBAGENT event=X] messages. Don't poll - just wait."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Agent to subscribe to. Use '*' for any agent. Omit for any."
                        },
                        "plan_id": {
                            "type": "string",
                            "description": "Specific plan ID to filter (optional)"
                        },
                        "step_id": {
                            "type": "string",
                            "description": "Specific step ID to filter (optional)"
                        },
                        "event_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "plan_created", "plan_started", "plan_completed",
                                    "plan_failed", "plan_cancelled",
                                    "step_added", "step_started", "step_completed",
                                    "step_failed", "step_skipped",
                                    "step_blocked", "step_unblocked"
                                ]
                            },
                            "description": "Event types to subscribe to"
                        }
                    },
                    "required": ["event_types"]
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="addDependentStep",
                description=(
                    "Add a step that waits for tasks from other agents. The step will be "
                    "BLOCKED until ALL dependencies complete, then auto-unblocks.\n\n"
                    "WHEN TO USE: After receiving a [SUBAGENT event=plan_created] message "
                    "showing subagent step IDs, call this to link your plan to their work.\n\n"
                    "WORKFLOW:\n"
                    "1. Subscribe to events (subscribeToTasks)\n"
                    "2. Spawn subagents\n"
                    "3. When you see plan_created events with step IDs, call addDependentStep\n"
                    "4. Wait - your step auto-unblocks when dependencies complete\n"
                    "5. Check getPlanStatus to see received_outputs from subagents\n\n"
                    "Example (single dependency):\n"
                    "  addDependentStep(\n"
                    "    description='Review implementation results',\n"
                    "    depends_on=[{agent_id: 'implementer', step_id: 'final_step'}]\n"
                    "  )\n\n"
                    "Example (multiple dependencies - all must complete):\n"
                    "  addDependentStep(\n"
                    "    description='Check all validations passed',\n"
                    "    depends_on=[\n"
                    "      {agent_id: 'type_checker', step_id: 'check'},\n"
                    "      {agent_id: 'test_runner', step_id: 'run'},\n"
                    "      {agent_id: 'linter', step_id: 'lint'}\n"
                    "    ]\n"
                    "  )"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of the step"
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent_id": {
                                        "type": "string",
                                        "description": "Agent that owns the dependency"
                                    },
                                    "plan_id": {
                                        "type": "string",
                                        "description": "Plan ID (optional - matches latest if omitted)"
                                    },
                                    "step_id": {
                                        "type": "string",
                                        "description": "Step ID to depend on"
                                    }
                                },
                                "required": ["agent_id", "step_id"]
                            },
                            "description": "Tasks this step depends on"
                        },
                        "after_step_id": {
                            "type": "string",
                            "description": "Insert after this step (optional)"
                        },
                        "provides": {
                            "type": "string",
                            "description": "Named output key for this step (optional)"
                        }
                    },
                    "required": ["description", "depends_on"]
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="completeStepWithOutput",
                description=(
                    "Complete a step AND pass structured data to tasks depending on it. "
                    "This is how subagents report results back to parent agents.\n\n"
                    "SUBAGENTS: Use this for your FINAL step to return structured results:\n"
                    "  completeStepWithOutput(\n"
                    "    step_id='final',\n"
                    "    output={passed: true, results: [...], errors: []}\n"
                    "  )\n\n"
                    "The parent agent's dependent step will:\n"
                    "- Auto-unblock when this completes\n"
                    "- Receive your output in received_outputs (visible in getPlanStatus)\n"
                    "- Be able to make decisions based on your data (e.g., iterate on failures)\n\n"
                    "VALIDATION PATTERN: Return pass/fail status so parent can iterate:\n"
                    "  completeStepWithOutput(\n"
                    "    step_id='validation',\n"
                    "    output={passed: false, errors: ['Type error in foo.ts:42']}\n"
                    "  )\n\n"
                    "Use regular updateStep(status='completed') for steps that don't need "
                    "to pass data to other agents."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "step_id": {
                            "type": "string",
                            "description": "ID of the step to complete"
                        },
                        "output": {
                            "type": "object",
                            "description": "Structured output to pass to dependent steps"
                        },
                        "result": {
                            "type": "string",
                            "description": "Optional text result/notes"
                        }
                    },
                    "required": ["step_id", "output"]
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="getBlockedSteps",
                description=(
                    "See which of your steps are waiting on other agents.\n\n"
                    "USE THIS when you want to check coordination status:\n"
                    "- What steps are blocked and why\n"
                    "- Which dependencies have resolved (received_outputs)\n"
                    "- Which are still pending (blocked_by)\n\n"
                    "Steps auto-unblock when all dependencies complete - you don't need to poll."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "plan_id": {
                            "type": "string",
                            "description": "Plan ID (optional, defaults to current plan)"
                        }
                    },
                    "required": []
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="getTaskEvents",
                description=(
                    "Review recent cross-agent activity. Shows plan/step events from all agents.\n\n"
                    "USE THIS to:\n"
                    "- See subagent progress after spawning them\n"
                    "- Debug why a dependency hasn't resolved\n"
                    "- Review what happened while you were working\n\n"
                    "If subscribed to events, you'll see them inline - this is for history review."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Filter by source agent (optional)"
                        },
                        "event_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by event types (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum events to return (default: 20)"
                        }
                    },
                    "required": []
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="listSubscriptions",
                description="List your active event subscriptions. Shows what events you're listening for.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                category="planning",
                discoverability="core",
            ),
            ToolSchema(
                name="unsubscribe",
                description="Remove an event subscription.",
                parameters={
                    "type": "object",
                    "properties": {
                        "subscription_id": {
                            "type": "string",
                            "description": "ID of the subscription to remove"
                        }
                    },
                    "required": ["subscription_id"]
                },
                category="planning",
                discoverability="core",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executors for TODO tools."""
        return {
            "createPlan": self._execute_create_plan,
            "startPlan": self._execute_start_plan,
            "updateStep": self._execute_update_step,
            "getPlanStatus": self._execute_get_plan_status,
            "completePlan": self._execute_complete_plan,
            "addStep": self._execute_add_step,
            # User command alias for getPlanStatus
            "plan": self._execute_get_plan_status,
            # Cross-agent collaboration tools
            "subscribeToTasks": self._execute_subscribe_to_tasks,
            "addDependentStep": self._execute_add_dependent_step,
            "completeStepWithOutput": self._execute_complete_step_with_output,
            "getBlockedSteps": self._execute_get_blocked_steps,
            "getTaskEvents": self._execute_get_task_events,
            "listSubscriptions": self._execute_list_subscriptions,
            "unsubscribe": self._execute_unsubscribe,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the TODO plugin."""
        return (
            "You have access to plan tracking and cross-agent coordination tools.\n\n"
            "# WHEN TO USE PLANS\n"
            "- Use plans when the user explicitly requests one, OR when coordinating subagents\n"
            "- For simple single-agent tasks, just do the work directly\n"
            "- When spawning subagents, ALWAYS use plans to coordinate their work\n\n"
            "# PLAN WORKFLOW (Single Agent)\n"
            "1. createPlan → 2. startPlan (wait for approval) → 3. updateStep → 4. completePlan\n\n"
            "# CROSS-AGENT COORDINATION (Multi-Agent)\n\n"
            "When delegating work to subagents, use this pattern:\n\n"
            "**Step 1: Subscribe to events BEFORE spawning subagents**\n"
            "```\n"
            "subscribeToTasks(event_types=['plan_created', 'step_completed'])\n"
            "```\n"
            "This lets you see subagent plans and react when they complete work.\n\n"
            "**Step 2: Create your master plan with placeholder steps**\n"
            "```\n"
            "createPlan(title='Main Task', steps=[\n"
            "  'Analyze requirements and spawn workers',\n"
            "  'Await and integrate results',  // Will become dependent step\n"
            "  'Validate and iterate'\n"
            "])\n"
            "```\n\n"
            "**Step 3: Spawn subagents for parallel work**\n"
            "```\n"
            "spawn_subagent(name='implementer', task='...')\n"
            "spawn_subagent(name='validator', task='...')\n"
            "```\n\n"
            "**Step 4: When you receive plan_created events, add dependent steps**\n"
            "After receiving `[SUBAGENT event=plan_created]` with step IDs:\n"
            "```\n"
            "addDependentStep(\n"
            "  description='Integrate implementation results',\n"
            "  depends_on=[{agent_id: 'implementer', step_id: '<final_step>'}]\n"
            ")\n"
            "```\n"
            "This step will be BLOCKED until the subagent completes that step.\n\n"
            "**Step 5: Subagents use completeStepWithOutput for structured results**\n"
            "Subagents should complete their final steps with:\n"
            "```\n"
            "completeStepWithOutput(step_id='...', output={results: [...], passed: true})\n"
            "```\n"
            "This output flows to dependent steps via received_outputs.\n\n"
            "**Step 6: Your blocked steps auto-unblock with data**\n"
            "When dependencies complete, your step unblocks and you can:\n"
            "- Check getPlanStatus() to see received_outputs\n"
            "- Access the structured data from subagents\n"
            "- Make decisions based on their results (e.g., iterate on failures)\n\n"
            "# VALIDATION & ITERATION PATTERN\n\n"
            "For tasks requiring validation:\n"
            "1. Spawn implementation subagent\n"
            "2. Spawn validation subagents (type checker, test runner, linter)\n"
            "3. Add step depending on ALL validators:\n"
            "   ```\n"
            "   addDependentStep(\n"
            "     description='Check all validations passed',\n"
            "     depends_on=[\n"
            "       {agent_id: 'type_checker', step_id: 'final'},\n"
            "       {agent_id: 'test_runner', step_id: 'final'},\n"
            "       {agent_id: 'linter', step_id: 'final'}\n"
            "     ]\n"
            "   )\n"
            "   ```\n"
            "4. When unblocked, check received_outputs for pass/fail\n"
            "5. If any failed: send_to_subagent with fixes, re-run validators\n"
            "6. Repeat until all validators report {passed: true}\n\n"
            "# KEY BEHAVIORS\n\n"
            "- **Don't poll** - blocked steps auto-unblock when dependencies complete\n"
            "- **Use structured outputs** - completeStepWithOutput passes typed data\n"
            "- **Check getBlockedSteps()** to see what's waiting on what\n"
            "- **Check getTaskEvents()** to review cross-agent activity\n"
            "- **Subagents see each other's events** - they can coordinate peer-to-peer\n\n"
            "# STEP STATUS RULES\n\n"
            "- 'completed': ONLY if fully accomplished\n"
            "- 'failed': Could not achieve goal (errors, limitations, partial)\n"
            "- 'skipped': Step became unnecessary\n"
            "- 'blocked': Waiting on cross-agent dependencies (automatic)\n"
            "- Be honest - failed steps are more valuable than false completions\n\n"
            "# RULES\n\n"
            "- MUST call startPlan after createPlan and wait for approval\n"
            "- CANNOT updateStep until startPlan is approved\n"
            "- If startPlan denied: call completePlan(status='cancelled')\n"
            "- When using completeStepWithOutput, the output is passed to dependent steps"
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return TODO tools as auto-approved (no security implications).

        Note: startPlan is intentionally excluded - it requires user permission
        to confirm they want the model to proceed with the proposed plan.

        The 'plan' user command is also included since it's just a status query.
        """
        return [
            # Core plan management
            "createPlan", "updateStep", "getPlanStatus", "completePlan", "addStep", "plan",
            # Cross-agent collaboration (read/write plan state, no side effects)
            "subscribeToTasks", "addDependentStep", "completeStepWithOutput",
            "getBlockedSteps", "getTaskEvents", "listSubscriptions", "unsubscribe",
        ]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for direct invocation.

        These commands can be typed directly by the user (human or agent)
        to check plan status without model mediation.

        The plan command output is NOT shared with the model (share_with_model=False)
        since it's purely for user visibility into progress.
        """
        return [
            UserCommand("plan", "Show current or most recent plan status", share_with_model=False),
        ]

    def _execute_create_plan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the createPlan tool."""
        title = args.get("title", "")
        steps = args.get("steps", [])
        self._trace(f"createPlan: title={title!r}, steps={len(steps)}")

        if not title:
            return {"error": "title is required"}

        if not steps or not isinstance(steps, list):
            return {"error": "steps must be a non-empty array"}

        if not all(isinstance(s, str) for s in steps):
            return {"error": "all steps must be strings"}

        # Create plan
        plan = TodoPlan.create(title=title, step_descriptions=steps)

        # Save to storage
        if self._storage:
            self._storage.save_plan(plan)

        # Set as current plan for this agent
        self._current_plan_ids[self._agent_name] = plan.plan_id

        # Report creation
        if self._reporter:
            self._reporter.report_plan_created(plan, agent_id=self._agent_name)

        # Publish plan_created event for cross-agent collaboration
        self._publish_event(
            TaskEventType.PLAN_CREATED,
            plan,
            payload={
                "steps": [
                    {
                        "step_id": s.step_id,
                        "sequence": s.sequence,
                        "description": s.description,
                    }
                    for s in plan.steps
                ]
            }
        )

        return {
            "plan_id": plan.plan_id,
            "title": plan.title,
            "status": plan.status.value,
            "steps": [
                {
                    "step_id": s.step_id,
                    "sequence": s.sequence,
                    "description": s.description,
                    "status": s.status.value,
                }
                for s in plan.steps
            ],
            "progress": plan.get_progress(),
        }

    def _execute_start_plan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the startPlan tool.

        This tool requires permission - when the user approves, it signals
        that they agree with the proposed plan and the model can proceed.
        """
        message = args.get("message", "")
        self._trace(f"startPlan: message={message!r}")

        # Get current plan
        plan = self._get_current_plan()
        if not plan:
            return {"error": "No active plan. Create a plan first with createPlan."}

        if plan.started:
            return {"error": "Plan already started. Proceed with updateStep."}

        # Mark plan as started (user approved)
        plan.started = True
        plan.started_at = datetime.utcnow().isoformat() + "Z"

        # Save to storage
        if self._storage:
            self._storage.save_plan(plan)

        # Publish plan_started event
        self._publish_event(TaskEventType.PLAN_STARTED, plan)

        return {
            "approved": True,
            "plan_id": plan.plan_id,
            "title": plan.title,
            "message": message or "Plan approved by user. You may proceed with execution.",
            "steps": [
                {
                    "sequence": s.sequence,
                    "description": s.description,
                }
                for s in sorted(plan.steps, key=lambda x: x.sequence)
            ],
        }

    def _execute_update_step(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the updateStep tool."""
        step_id = args.get("step_id", "")
        status_str = args.get("status", "")
        result = args.get("result")
        error = args.get("error")
        self._trace(f"updateStep: step_id={step_id}, status={status_str}, agent_name={self._agent_name}")

        if not step_id:
            return {"error": "step_id is required"}

        if not status_str:
            return {"error": "status is required"}

        # Validate status
        try:
            new_status = StepStatus(status_str)
        except ValueError:
            return {"error": f"Invalid status: {status_str}. "
                           f"Must be one of: in_progress, completed, failed, skipped"}

        # Get current plan
        plan = self._get_current_plan()
        if not plan:
            # Enhanced debugging: show what plan IDs exist
            known_agents = list(self._current_plan_ids.keys())
            self._trace(f"updateStep ERROR: No plan for agent={self._agent_name}, known_agents={known_agents}")
            return {"error": f"No active plan. Create a plan first with createPlan. (agent={self._agent_name})"}

        if not plan.started:
            return {"error": "Plan not started. Call startPlan first to get user approval."}

        # Find step
        step = plan.get_step_by_id(step_id)
        if not step:
            # Enhanced debugging: show what steps exist in the plan
            existing_step_ids = [s.step_id for s in plan.steps]
            self._trace(f"updateStep ERROR: Step {step_id} not in plan {plan.plan_id}, existing={existing_step_ids}")
            return {"error": f"Step not found: {step_id}"}

        # Update step status
        if new_status == StepStatus.IN_PROGRESS:
            step.start()
            plan.current_step = step.sequence
        elif new_status == StepStatus.COMPLETED:
            step.complete(result)
        elif new_status == StepStatus.FAILED:
            step.fail(error)
        elif new_status == StepStatus.SKIPPED:
            step.skip(result)

        # Save to storage
        if self._storage:
            self._storage.save_plan(plan)

        # Report update
        if self._reporter:
            self._reporter.report_step_update(plan, step, agent_id=self._agent_name)

        # Publish step event for cross-agent collaboration
        event_type_map = {
            StepStatus.IN_PROGRESS: TaskEventType.STEP_STARTED,
            StepStatus.COMPLETED: TaskEventType.STEP_COMPLETED,
            StepStatus.FAILED: TaskEventType.STEP_FAILED,
            StepStatus.SKIPPED: TaskEventType.STEP_SKIPPED,
        }
        if new_status in event_type_map:
            payload = {"result": step.result, "error": step.error}
            if step.output is not None:
                payload["output"] = step.output
            if step.provides:
                payload["provides"] = step.provides
            self._publish_event(event_type_map[new_status], plan, step, payload)

        # Build response with continuation prompt
        progress = plan.get_progress()
        response = {
            "step_id": step.step_id,
            "sequence": step.sequence,
            "description": step.description,
            "status": step.status.value,
            "result": step.result,
            "error": step.error,
            "progress": progress,
        }

        # Add continuation prompt based on status to prevent model from stopping
        if new_status == StepStatus.IN_PROGRESS:
            response["instruction"] = (
                f"Step is now in progress. Proceed immediately with: {step.description}"
            )
        elif new_status == StepStatus.COMPLETED:
            if progress["pending"] > 0:
                # Find next pending step
                next_step = next(
                    (s for s in sorted(plan.steps, key=lambda x: x.sequence)
                     if s.status == StepStatus.PENDING),
                    None
                )
                if next_step:
                    response["instruction"] = (
                        f"Step completed. Proceed immediately to next step: {next_step.description}"
                    )
                else:
                    response["instruction"] = "Step completed. Continue with remaining work."
            else:
                response["instruction"] = "All steps completed. Call completePlan to finalize."
        elif new_status == StepStatus.FAILED:
            if progress["pending"] > 0:
                response["instruction"] = (
                    "Step failed. Decide whether to retry, skip, or proceed to remaining steps."
                )

        return response

    def _execute_get_plan_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the getPlanStatus tool."""
        plan_id = args.get("plan_id")
        self._trace(f"getPlanStatus: plan_id={plan_id}, agent_name={self._agent_name}")

        # Get plan by explicit ID or current plan for this agent
        if plan_id and self._storage:
            plan = self._storage.get_plan(plan_id)
        else:
            # Only get this agent's current plan - don't fall back to other agents' plans
            plan = self._get_current_plan()

        if not plan:
            # Provide helpful context about why no plan was found
            agent_context = f" (agent: {self._agent_name})" if self._agent_name else ""
            return {
                "error": f"No plan found for this agent{agent_context}. "
                         f"Create a plan first with createPlan."
            }

        return {
            "plan_id": plan.plan_id,
            "title": plan.title,
            "status": plan.status.value,
            "created_at": plan.created_at,
            "completed_at": plan.completed_at,
            "summary": plan.summary,
            "current_step": plan.current_step,
            "steps": [
                {
                    "step_id": s.step_id,
                    "sequence": s.sequence,
                    "description": s.description,
                    "status": s.status.value,
                    "started_at": s.started_at,
                    "completed_at": s.completed_at,
                    "result": s.result,
                    "error": s.error,
                }
                for s in plan.steps
            ],
            "progress": plan.get_progress(),
        }

    def _execute_complete_plan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the completePlan tool."""
        status_str = args.get("status", "")
        summary = args.get("summary")
        self._trace(f"completePlan: status={status_str}")

        if not status_str:
            return {"error": "status is required"}

        # Validate status
        if status_str not in ("completed", "failed", "cancelled"):
            return {"error": f"Invalid status: {status_str}. "
                           f"Must be one of: completed, failed, cancelled"}

        # Get current plan
        plan = self._get_current_plan()
        if not plan:
            return {"error": "No active plan. Create a plan first with createPlan."}

        # Guard: can only complete/fail a plan that was started
        # Cancelling is allowed even if not started (user rejected the plan)
        if not plan.started and status_str in ("completed", "failed"):
            return {"error": f"Cannot mark plan as '{status_str}' - plan was never started. "
                           f"Use 'cancelled' if the plan was rejected."}

        # Update plan status
        if status_str == "completed":
            plan.complete_plan(summary)
        elif status_str == "failed":
            plan.fail_plan(summary)
        else:
            plan.cancel_plan(summary)

        # Save to storage
        if self._storage:
            self._storage.save_plan(plan)

        # Report completion
        if self._reporter:
            self._reporter.report_plan_completed(plan, agent_id=self._agent_name)

        # Publish plan completion event
        event_type_map = {
            "completed": TaskEventType.PLAN_COMPLETED,
            "failed": TaskEventType.PLAN_FAILED,
            "cancelled": TaskEventType.PLAN_CANCELLED,
        }
        if status_str in event_type_map:
            self._publish_event(
                event_type_map[status_str],
                plan,
                payload={"summary": plan.summary, "progress": plan.get_progress()}
            )

        # Clear current plan for this agent
        self._current_plan_ids.pop(self._agent_name, None)

        return {
            "plan_id": plan.plan_id,
            "title": plan.title,
            "status": plan.status.value,
            "completed_at": plan.completed_at,
            "summary": plan.summary,
            "progress": plan.get_progress(),
        }

    def _execute_add_step(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the addStep tool."""
        description = args.get("description", "")
        after_step_id = args.get("after_step_id")
        self._trace(f"addStep: description={description!r}, after={after_step_id}")

        if not description:
            return {"error": "description is required"}

        # Get current plan
        plan = self._get_current_plan()
        if not plan:
            return {"error": "No active plan. Create a plan first with createPlan."}

        if not plan.started:
            return {"error": "Plan not started. Call startPlan first to get user approval."}

        # Add the step
        new_step = plan.add_step(description, after_step_id)

        # Save to storage
        if self._storage:
            self._storage.save_plan(plan)

        # Report the addition
        if self._reporter:
            self._reporter.report_step_update(plan, new_step, agent_id=self._agent_name)

        # Publish step_added event
        self._publish_event(TaskEventType.STEP_ADDED, plan, new_step)

        return {
            "step_id": new_step.step_id,
            "sequence": new_step.sequence,
            "description": new_step.description,
            "status": new_step.status.value,
            "total_steps": len(plan.steps),
            "progress": plan.get_progress(),
        }

    def _get_current_plan(self) -> Optional[TodoPlan]:
        """Get the current active plan for this agent."""
        plan_id = self._current_plan_ids.get(self._agent_name)
        if not plan_id or not self._storage:
            return None
        return self._storage.get_plan(plan_id)

    def _get_most_recent_plan(self) -> Optional[TodoPlan]:
        """Get the most recently created plan from storage."""
        if not self._storage:
            return None
        all_plans = self._storage.get_all_plans()
        if not all_plans:
            return None
        # Sort by created_at descending and return the most recent
        return max(all_plans, key=lambda p: p.created_at)

    # Convenience methods for programmatic access

    def create_plan(
        self,
        title: str,
        steps: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> TodoPlan:
        """Create a new plan programmatically.

        Args:
            title: Plan title
            steps: List of step descriptions
            context: Optional context data

        Returns:
            Created TodoPlan instance
        """
        plan = TodoPlan.create(title=title, step_descriptions=steps, context=context)

        if self._storage:
            self._storage.save_plan(plan)

        self._current_plan_ids[self._agent_name] = plan.plan_id

        if self._reporter:
            self._reporter.report_plan_created(plan, agent_id=self._agent_name)

        return plan

    def update_step(
        self,
        step_id: str,
        status: StepStatus,
        result: Optional[str] = None,
        error: Optional[str] = None
    ) -> Optional[TodoStep]:
        """Update a step programmatically.

        Args:
            step_id: ID of the step
            status: New status
            result: Optional result/notes
            error: Optional error message

        Returns:
            Updated TodoStep or None if not found
        """
        plan = self._get_current_plan()
        if not plan:
            return None

        step = plan.get_step_by_id(step_id)
        if not step:
            return None

        if status == StepStatus.IN_PROGRESS:
            step.start()
            plan.current_step = step.sequence
        elif status == StepStatus.COMPLETED:
            step.complete(result)
        elif status == StepStatus.FAILED:
            step.fail(error)
        elif status == StepStatus.SKIPPED:
            step.skip(result)

        if self._storage:
            self._storage.save_plan(plan)

        if self._reporter:
            self._reporter.report_step_update(plan, step, agent_id=self._agent_name)

        return step

    def get_current_plan(self) -> Optional[TodoPlan]:
        """Get the current active plan."""
        return self._get_current_plan()

    def get_all_plans(self) -> List[TodoPlan]:
        """Get all stored plans."""
        if not self._storage:
            return []
        return self._storage.get_all_plans()

    # Interactivity protocol methods

    def supports_interactivity(self) -> bool:
        """TODO plugin has interactive features for progress reporting.

        Returns:
            True - TODO plugin has interactive progress reporting.
        """
        return True

    def get_supported_channels(self) -> List[str]:
        """Return list of channel types supported by TODO plugin.

        Returns:
            List of supported channel types matching available reporters.
        """
        return ["console", "webhook", "file"]

    def set_channel(
        self,
        channel_type: str,
        channel_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the interaction channel for progress reporting.

        Args:
            channel_type: One of: console, webhook, file
            channel_config: Optional channel-specific configuration
                For "console": output_callback for rich client integration

        Raises:
            ValueError: If channel_type is not supported
        """
        if channel_type not in self.get_supported_channels():
            raise ValueError(
                f"Channel type '{channel_type}' not supported. "
                f"Supported: {self.get_supported_channels()}"
            )

        # Create reporter with config
        from .channels import create_reporter
        reporter_config = channel_config or {}
        self._reporter = create_reporter(channel_type, reporter_config)

    # === Cross-agent collaboration executors ===

    def _execute_subscribe_to_tasks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the subscribeToTasks tool."""
        event_types_raw = args.get("event_types", [])
        agent_id = args.get("agent_id")
        plan_id = args.get("plan_id")
        step_id = args.get("step_id")

        self._trace(f"subscribeToTasks: events={event_types_raw}, agent={agent_id}")

        if not event_types_raw:
            return {"error": "event_types is required"}

        # Parse event types
        event_types = []
        for et in event_types_raw:
            try:
                event_types.append(TaskEventType(et))
            except ValueError:
                return {"error": f"Invalid event type: {et}"}

        if not self._event_bus:
            return {"error": "Event bus not initialized"}

        # Create filter
        filter = EventFilter(
            agent_id=agent_id,
            plan_id=plan_id,
            step_id=step_id,
            event_types=event_types
        )

        # Create callback that stores events for the agent
        def on_event(event: TaskEvent) -> None:
            # Store event in agent's queue for later retrieval
            # The event is already in the bus history, but we trace it
            self._trace(
                f"Event received: {event.event_type.value} from {event.source_agent}"
            )

        # Subscribe
        sub_id = self._event_bus.subscribe(
            subscriber_agent=self._agent_name or "main",
            filter=filter,
            callback=on_event
        )

        # Track subscription for this agent
        if self._agent_name not in self._agent_subscriptions:
            self._agent_subscriptions[self._agent_name] = []
        self._agent_subscriptions[self._agent_name].append(sub_id)

        return {
            "subscription_id": sub_id,
            "filter": filter.to_dict(),
            "message": f"Subscribed to {len(event_types)} event type(s)"
        }

    def _execute_add_dependent_step(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the addDependentStep tool."""
        description = args.get("description", "")
        depends_on_raw = args.get("depends_on", [])
        after_step_id = args.get("after_step_id")
        provides = args.get("provides")

        self._trace(f"addDependentStep: description={description!r}, deps={len(depends_on_raw)}")

        if not description:
            return {"error": "description is required"}

        if not depends_on_raw:
            return {"error": "depends_on is required and must not be empty"}

        # Get current plan
        plan = self._get_current_plan()
        if not plan:
            return {"error": "No active plan. Create a plan first with createPlan."}

        if not plan.started:
            return {"error": "Plan not started. Call startPlan first to get user approval."}

        # Parse dependencies
        depends_on = []
        for dep in depends_on_raw:
            if not dep.get("agent_id") or not dep.get("step_id"):
                return {"error": "Each dependency must have agent_id and step_id"}
            depends_on.append(TaskRef.from_dict(dep))

        # Add the step with dependencies
        new_step = plan.add_step(
            description=description,
            after_step_id=after_step_id,
            depends_on=depends_on,
            provides=provides
        )

        # Register dependencies with the event bus
        if self._event_bus:
            for dep in depends_on:
                self._event_bus.register_dependency(
                    dependency_ref=dep,
                    waiting_agent=self._agent_name or "main",
                    waiting_plan_id=plan.plan_id,
                    waiting_step_id=new_step.step_id
                )

        # Save to storage
        if self._storage:
            self._storage.save_plan(plan)

        # Report the addition
        if self._reporter:
            self._reporter.report_step_update(plan, new_step, agent_id=self._agent_name)

        # Publish step_blocked event
        if self._event_bus:
            event = TaskEvent.create(
                event_type=TaskEventType.STEP_BLOCKED,
                agent_id=self._agent_name or "main",
                plan=plan,
                step=new_step,
                payload={
                    "blocked_by": [ref.to_dict() for ref in new_step.blocked_by]
                }
            )
            self._event_bus.publish(event)

        return {
            "step_id": new_step.step_id,
            "sequence": new_step.sequence,
            "description": new_step.description,
            "status": new_step.status.value,
            "blocked_by": [ref.to_dict() for ref in new_step.blocked_by],
            "total_steps": len(plan.steps),
            "progress": plan.get_progress(),
            "message": f"Step added with {len(depends_on)} dependencies (BLOCKED)"
        }

    def _execute_complete_step_with_output(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the completeStepWithOutput tool."""
        step_id = args.get("step_id", "")
        output = args.get("output")
        result = args.get("result")

        self._trace(f"completeStepWithOutput: step_id={step_id}, output_keys={list(output.keys()) if output else []}")

        if not step_id:
            return {"error": "step_id is required"}

        if output is None:
            return {"error": "output is required"}

        # Get current plan
        plan = self._get_current_plan()
        if not plan:
            return {"error": "No active plan. Create a plan first with createPlan."}

        if not plan.started:
            return {"error": "Plan not started. Call startPlan first to get user approval."}

        # Find step
        step = plan.get_step_by_id(step_id)
        if not step:
            return {"error": f"Step not found: {step_id}"}

        # Complete the step with output
        step.complete(result=result, output=output)

        # Save to storage
        if self._storage:
            self._storage.save_plan(plan)

        # Report update
        if self._reporter:
            self._reporter.report_step_update(plan, step, agent_id=self._agent_name)

        # Publish step_completed event with output
        if self._event_bus:
            event = TaskEvent.create(
                event_type=TaskEventType.STEP_COMPLETED,
                agent_id=self._agent_name or "main",
                plan=plan,
                step=step,
                payload={
                    "output": output,
                    "result": result,
                    "provides": step.provides
                }
            )
            self._event_bus.publish(event)

        return {
            "step_id": step.step_id,
            "sequence": step.sequence,
            "description": step.description,
            "status": step.status.value,
            "output": output,
            "result": result,
            "progress": plan.get_progress(),
            "message": "Step completed with output. Dependent steps will be notified."
        }

    def _execute_get_blocked_steps(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the getBlockedSteps tool."""
        plan_id = args.get("plan_id")

        self._trace(f"getBlockedSteps: plan_id={plan_id}")

        # Get plan by explicit ID or current plan
        if plan_id and self._storage:
            plan = self._storage.get_plan(plan_id)
        else:
            plan = self._get_current_plan()

        if not plan:
            return {"error": "No plan found. Create a plan first with createPlan."}

        blocked_steps = plan.get_blocked_steps()

        if not blocked_steps:
            return {
                "blocked_steps": [],
                "message": "No blocked steps"
            }

        return {
            "blocked_steps": [
                {
                    "step_id": s.step_id,
                    "sequence": s.sequence,
                    "description": s.description,
                    "blocked_by": [ref.to_dict() for ref in s.blocked_by],
                    "depends_on": [ref.to_dict() for ref in s.depends_on],
                    "received_outputs": list(s.received_outputs.keys())
                }
                for s in blocked_steps
            ],
            "count": len(blocked_steps)
        }

    def _execute_get_task_events(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the getTaskEvents tool."""
        agent_id = args.get("agent_id")
        event_types_raw = args.get("event_types", [])
        limit = args.get("limit", 20)

        self._trace(f"getTaskEvents: agent={agent_id}, types={event_types_raw}, limit={limit}")

        if not self._event_bus:
            return {"error": "Event bus not initialized"}

        # Parse event types
        event_types = None
        if event_types_raw:
            event_types = []
            for et in event_types_raw:
                try:
                    event_types.append(TaskEventType(et))
                except ValueError:
                    pass  # Skip invalid

        events = self._event_bus.get_recent_events(
            agent_id=agent_id,
            event_types=event_types,
            limit=limit
        )

        return {
            "events": [e.to_dict() for e in events],
            "count": len(events)
        }

    def _execute_list_subscriptions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the listSubscriptions tool."""
        self._trace("listSubscriptions")

        if not self._event_bus:
            return {"error": "Event bus not initialized"}

        subs = self._event_bus.get_subscriptions(agent_id=self._agent_name)

        return {
            "subscriptions": [s.to_dict() for s in subs],
            "count": len(subs)
        }

    def _execute_unsubscribe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the unsubscribe tool."""
        sub_id = args.get("subscription_id", "")

        self._trace(f"unsubscribe: sub_id={sub_id}")

        if not sub_id:
            return {"error": "subscription_id is required"}

        if not self._event_bus:
            return {"error": "Event bus not initialized"}

        success = self._event_bus.unsubscribe(sub_id)

        if success:
            # Remove from agent's tracked subscriptions
            if self._agent_name in self._agent_subscriptions:
                if sub_id in self._agent_subscriptions[self._agent_name]:
                    self._agent_subscriptions[self._agent_name].remove(sub_id)

        return {
            "success": success,
            "subscription_id": sub_id,
            "message": "Subscription removed" if success else "Subscription not found"
        }

    def _on_dependency_resolved(
        self,
        waiting_agent: str,
        waiting_plan_id: str,
        waiting_step_id: str,
        completion_event: TaskEvent
    ) -> None:
        """Callback when a dependency is resolved.

        Called by the event bus when a step that another step depends on
        completes. Updates the waiting step's blocked_by and received_outputs.

        Args:
            waiting_agent: Agent that was waiting.
            waiting_plan_id: Plan containing the waiting step.
            waiting_step_id: Step that was blocked.
            completion_event: The step_completed event with output.
        """
        self._trace(
            f"Dependency resolved: {waiting_agent}:{waiting_step_id} <- "
            f"{completion_event.source_agent}:{completion_event.source_step_id}"
        )

        # Get the waiting plan
        if not self._storage:
            return

        plan = self._storage.get_plan(waiting_plan_id)
        if not plan:
            return

        step = plan.get_step_by_id(waiting_step_id)
        if not step:
            return

        # Extract output from completion event
        output = completion_event.payload.get("output")
        provides_name = completion_event.payload.get("provides")

        # Create ref for the completed step
        completed_ref = TaskRef(
            agent_id=completion_event.source_agent,
            plan_id=completion_event.source_plan_id,
            step_id=completion_event.source_step_id or ""
        )

        # Resolve the dependency
        is_unblocked = step.resolve_dependency(completed_ref, output, provides_name)

        # Save updated plan
        self._storage.save_plan(plan)

        # If unblocked, publish event
        if is_unblocked and self._event_bus:
            event = TaskEvent.create(
                event_type=TaskEventType.STEP_UNBLOCKED,
                agent_id=waiting_agent,
                plan=plan,
                step=step,
                payload={
                    "received_outputs": list(step.received_outputs.keys()),
                    "unblocked_by": completed_ref.to_dict()
                }
            )
            self._event_bus.publish(event)

            # Report if reporter is available
            if self._reporter:
                self._reporter.report_step_update(plan, step, agent_id=waiting_agent)

    # === Event publishing helpers ===

    def _publish_event(
        self,
        event_type: TaskEventType,
        plan: TodoPlan,
        step: Optional[TodoStep] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publish an event to the event bus.

        Args:
            event_type: Type of event.
            plan: The plan this event relates to.
            step: Optional step this event relates to.
            payload: Additional event data.
        """
        if not self._event_bus:
            return

        event = TaskEvent.create(
            event_type=event_type,
            agent_id=self._agent_name or "main",
            plan=plan,
            step=step,
            payload=payload
        )
        self._event_bus.publish(event)


def create_plugin() -> TodoPlugin:
    """Factory function to create the TODO plugin instance."""
    return TodoPlugin()
