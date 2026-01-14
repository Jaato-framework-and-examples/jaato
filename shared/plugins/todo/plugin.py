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

from .models import PlanStatus, StepStatus, TodoPlan, TodoStep
from ..model_provider.types import ToolSchema
from .storage import TodoStorage, create_storage, InMemoryStorage
from .channels import TodoReporter, ConsoleReporter, create_reporter
from .config_loader import load_config, TodoConfig
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
        # Note: agent_name is stored in thread-local storage, not instance

    @property
    def _agent_name(self) -> Optional[str]:
        """Get the agent name for the current thread.

        Uses thread-local storage so each agent session (running in its
        own thread) has its own agent_name context.
        """
        return getattr(_thread_local, 'agent_name', None)

    @_agent_name.setter
    def _agent_name(self, value: Optional[str]) -> None:
        """Set the agent name for the current thread."""
        _thread_local.agent_name = value

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
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the TODO plugin."""
        return (
            "You have access to plan tracking tools.\n\n"
            "WHEN TO USE:\n"
            "- Only use these tools if the user explicitly requests a plan\n"
            "- Do NOT automatically create plans for every task\n"
            "- If the user does not ask for a plan, just do the task directly\n\n"
            "**IMPORTANT**: When you decide to present a plan to the user, you MUST use the `createPlan` tool. "
            "Do NOT write the plan as text in your response - always use the tool instead. "
            "This ensures proper plan tracking and user approval workflow.\n\n"
            "BEFORE CREATING A PLAN:\n"
            "- Think carefully about what steps are actually needed to achieve the goal\n"
            "- Break down the task into minimal, concrete steps you can realistically complete\n"
            "- Consider what tools and information you have available\n"
            "- Do NOT propose a plan you cannot achieve - be trustworthy\n"
            "- Each step should be specific and actionable, not vague\n"
            "- If a step will use a specific tool, include the tool name and command in the description\n"
            "  Example: 'List files sorted by date using cli_based_tool: ls -lt'\n"
            "  Example: 'Search for Python files using cli_based_tool: find . -name \"*.py\"'\n"
            "- This helps the user understand exactly what will be executed before approving\n\n"
            "VALIDATE TOOL AVAILABILITY:\n"
            "- BEFORE including any tool in a plan step, you MUST verify:\n"
            "  1. The tool exists and is available in the current session\n"
            "  2. You are allowed/permitted to use that tool\n"
            "  3. The tool can perform the action described in the step\n"
            "- Do NOT assume a tool exists - check your available tools first\n"
            "- Do NOT include steps that rely on tools you cannot access or are not permitted to use\n"
            "- If unsure whether a tool is available, either check first or choose an alternative approach\n"
            "- Steps that fail due to unavailable tools waste user time and erode trust\n\n"
            "WORKFLOW:\n"
            "1. createPlan - Register your execution plan with ordered steps\n"
            "2. startPlan - Request user approval (REQUIRED before any execution)\n"
            "3. updateStep - Report progress on each step (only after startPlan approved)\n"
            "4. addStep - Add new steps if needed during execution\n"
            "5. completePlan - Mark plan as finished\n\n"
            "RULES:\n"
            "- You MUST call startPlan after createPlan and wait for approval\n"
            "- You CANNOT execute ANY other tools until startPlan is approved, unless necessary to compose the plan\n"
            "- You CANNOT call updateStep or addStep until startPlan is approved\n"
            "- Only use status='completed' or 'failed' for plans that were started\n"
            "- Use status='cancelled' for plans the user rejected\n\n"
            "STEP STATUS RULES (be honest about outcomes):\n"
            "- 'completed': ONLY if the step was FULLY accomplished with the expected outcome\n"
            "- 'failed': If you could NOT achieve the step's goal for ANY reason (errors, limitations, partial results)\n"
            "- 'skipped': If the step became unnecessary or was intentionally bypassed\n"
            "- DO NOT mark a step as 'completed' if the outcome was partial, unsuccessful, or had errors\n"
            "- If you say 'could not', 'unable to', 'failed to', or similar in the result, the status MUST be 'failed'\n"
            "- Being honest about failures is more valuable than false completion reports\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "- When a step produces data/results, you MUST include the actual output in your response\n"
            "- Do NOT just report 'completed' - show what was found/created/computed\n"
            "- If the user asked for a table, list, or specific information, your final response MUST contain it\n"
            "- Saying 'the table above shows...' when no table was shown is WRONG - include the actual table\n"
            "- The plan completion message is NOT a substitute for showing results to the user\n\n"
            "WHEN startPlan IS DENIED:\n"
            "- Immediately call completePlan with status='cancelled' on the current plan\n"
            "- Do NOT create a new plan and retry - the user does not want plan tracking\n"
            "- Ask the user how they would like to proceed instead\n"
            "- You may proceed with the task without plan tracking if appropriate"
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return TODO tools as auto-approved (no security implications).

        Note: startPlan is intentionally excluded - it requires user permission
        to confirm they want the model to proceed with the proposed plan.

        The 'plan' user command is also included since it's just a status query.
        """
        return ["createPlan", "updateStep", "getPlanStatus", "completePlan", "addStep", "plan"]

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


def create_plugin() -> TodoPlugin:
    """Factory function to create the TODO plugin instance."""
    return TodoPlugin()
