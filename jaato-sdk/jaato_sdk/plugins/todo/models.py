"""Data models for the TODO plugin.

Defines the core data structures for plans, steps, and progress tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class StepStatus(Enum):
    """Possible statuses for a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"  # Waiting on cross-agent dependencies


class TaskEventType(Enum):
    """Types of task events for cross-agent collaboration.

    These events are published to the TaskEventBus when plan/step
    state changes, enabling other agents to react and coordinate.
    """
    # Plan lifecycle events
    PLAN_CREATED = "plan_created"
    PLAN_STARTED = "plan_started"
    PLAN_COMPLETED = "plan_completed"
    PLAN_FAILED = "plan_failed"
    PLAN_CANCELLED = "plan_cancelled"

    # Step lifecycle events
    STEP_ADDED = "step_added"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_SKIPPED = "step_skipped"

    # Collaboration events
    STEP_BLOCKED = "step_blocked"      # Step waiting on dependencies
    STEP_UNBLOCKED = "step_unblocked"  # All dependencies satisfied


class PlanStatus(Enum):
    """Possible statuses for an overall plan."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskRef:
    """Reference to a specific task (step) in any agent's plan.

    Used for:
    - Defining dependencies: "my step depends on this task"
    - Subscribing to events: "notify me when this task completes"
    - Querying status: "is this task done?"

    Examples:
        TaskRef(agent_id="researcher", step_id="final_analysis")
        TaskRef(agent_id="subagent_1", plan_id="abc123", step_id="step_2")
    """
    agent_id: str            # "main", "subagent_1", "researcher", etc.
    step_id: str             # Step ID within the plan
    plan_id: Optional[str] = None  # Optional - if None, matches latest plan

    # Optional display metadata (for richer UI presentation)
    agent_name: Optional[str] = None      # Human-friendly agent name
    step_sequence: Optional[int] = None   # Step sequence number (1, 2, 3...)
    step_description: Optional[str] = None  # Step description text

    def to_uri(self) -> str:
        """Convert to URI format: agent:plan/step or agent:*/step"""
        plan_part = self.plan_id or "*"
        return f"{self.agent_id}:{plan_part}/{self.step_id}"

    @classmethod
    def from_uri(cls, uri: str) -> 'TaskRef':
        """Parse from URI format (agent:plan/step or agent:*/step)."""
        if ":" not in uri or "/" not in uri:
            raise ValueError(f"Invalid TaskRef URI: {uri}")
        agent_part, rest = uri.split(":", 1)
        plan_part, step_part = rest.split("/", 1)
        return cls(
            agent_id=agent_part,
            plan_id=None if plan_part == "*" else plan_part,
            step_id=step_part
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskRef':
        """Create from dict (e.g., from tool args)."""
        return cls(
            agent_id=data.get("agent_id", ""),
            step_id=data.get("step_id", ""),
            plan_id=data.get("plan_id"),
            agent_name=data.get("agent_name"),
            step_sequence=data.get("step_sequence"),
            step_description=data.get("step_description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "agent_id": self.agent_id,
            "step_id": self.step_id,
        }
        if self.plan_id:
            result["plan_id"] = self.plan_id
        # Include display metadata if available
        if self.agent_name:
            result["agent_name"] = self.agent_name
        if self.step_sequence is not None:
            result["step_sequence"] = self.step_sequence
        if self.step_description:
            result["step_description"] = self.step_description
        return result

    def matches(self, agent_id: str, plan_id: str, step_id: str) -> bool:
        """Check if this ref matches the given task.

        Args:
            agent_id: The agent ID to match against.
            plan_id: The plan ID to match against.
            step_id: The step ID to match against.

        Returns:
            True if this reference matches the given task.
        """
        if self.agent_id != agent_id:
            return False
        if self.plan_id and self.plan_id != plan_id:
            return False
        return self.step_id == step_id

    def __hash__(self):
        return hash((self.agent_id, self.plan_id, self.step_id))

    def __eq__(self, other):
        if not isinstance(other, TaskRef):
            return False
        return (self.agent_id == other.agent_id and
                self.plan_id == other.plan_id and
                self.step_id == other.step_id)


@dataclass
class EventFilter:
    """Filter for subscribing to specific task events.

    All fields are optional - None means "match any".
    Empty event_types list means "match all event types".
    """
    agent_id: Optional[str] = None       # None or "*" = any agent
    plan_id: Optional[str] = None        # None = any plan
    step_id: Optional[str] = None        # None = any step
    event_types: List[TaskEventType] = field(default_factory=list)  # Empty = all

    def matches(self, event: 'TaskEvent') -> bool:
        """Check if an event matches this filter.

        Args:
            event: The TaskEvent to check.

        Returns:
            True if the event matches all specified filter criteria.
        """
        # Agent filter
        if self.agent_id and self.agent_id != "*":
            if event.source_agent != self.agent_id:
                return False

        # Plan filter
        if self.plan_id and event.source_plan_id != self.plan_id:
            return False

        # Step filter
        if self.step_id and event.source_step_id != self.step_id:
            return False

        # Event type filter
        if self.event_types and event.event_type not in self.event_types:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "event_types": [e.value for e in self.event_types],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventFilter':
        """Create from dictionary."""
        event_types = []
        for et in data.get("event_types", []):
            try:
                event_types.append(TaskEventType(et))
            except ValueError:
                pass  # Skip invalid event types
        return cls(
            agent_id=data.get("agent_id"),
            plan_id=data.get("plan_id"),
            step_id=data.get("step_id"),
            event_types=event_types,
        )


@dataclass
class Subscription:
    """A subscription to task events.

    When events matching the filter are published, the subscription's
    action is executed.
    """
    subscription_id: str
    subscriber_agent: str    # Agent that created the subscription
    filter: EventFilter

    # Action when event matches
    action_type: str = "callback"  # "callback", "unblock_step", "inject_message"
    action_target: Optional[str] = None  # step_id to unblock, callback name, etc.

    # Lifecycle
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z")
    expires_after: Optional[int] = None   # Auto-remove after N matches (None = persistent)
    match_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subscription_id": self.subscription_id,
            "subscriber_agent": self.subscriber_agent,
            "filter": self.filter.to_dict(),
            "action_type": self.action_type,
            "action_target": self.action_target,
            "created_at": self.created_at,
            "expires_after": self.expires_after,
            "match_count": self.match_count,
        }


@dataclass
class TaskEvent:
    """Event published when task state changes.

    These events are published to the TaskEventBus and delivered to
    matching subscribers. They enable cross-agent coordination.
    """
    event_id: str
    event_type: TaskEventType
    timestamp: str  # ISO8601

    # Source identification
    source_agent: str
    source_plan_id: str
    source_plan_title: str
    source_step_id: Optional[str] = None
    source_step_description: Optional[str] = None
    source_step_sequence: Optional[int] = None

    # Event payload - varies by event type:
    # - plan_created: {steps: [{step_id, description, sequence}, ...]}
    # - step_completed: {output: {...}, result: "..."}
    # - step_blocked: {blocked_by: [{agent_id, step_id}, ...]}
    # - step_unblocked: {received_outputs: {...}}
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: TaskEventType,
        agent_id: str,
        plan: 'TodoPlan',
        step: Optional['TodoStep'] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> 'TaskEvent':
        """Create a new TaskEvent.

        Args:
            event_type: Type of event.
            agent_id: ID of the agent that owns the plan.
            plan: The plan this event relates to.
            step: Optional step this event relates to.
            payload: Additional event-specific data.

        Returns:
            A new TaskEvent instance.
        """
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            source_agent=agent_id,
            source_plan_id=plan.plan_id,
            source_plan_title=plan.title,
            source_step_id=step.step_id if step else None,
            source_step_description=step.description if step else None,
            source_step_sequence=step.sequence if step else None,
            payload=payload or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "source_plan_id": self.source_plan_id,
            "source_plan_title": self.source_plan_title,
            "source_step_id": self.source_step_id,
            "source_step_description": self.source_step_description,
            "source_step_sequence": self.source_step_sequence,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskEvent':
        """Create from dictionary."""
        try:
            event_type = TaskEventType(data.get("event_type", "step_completed"))
        except ValueError:
            event_type = TaskEventType.STEP_COMPLETED

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat() + "Z"),
            source_agent=data.get("source_agent", ""),
            source_plan_id=data.get("source_plan_id", ""),
            source_plan_title=data.get("source_plan_title", ""),
            source_step_id=data.get("source_step_id"),
            source_step_description=data.get("source_step_description"),
            source_step_sequence=data.get("source_step_sequence"),
            payload=data.get("payload", {}),
        )


@dataclass
class TodoStep:
    """A single step within a plan.

    Extended with cross-agent collaboration fields for task dependencies.

    Validation enforcement:
        Steps with ``validation_required=True`` cannot be marked as completed
        via ``setStepStatus`` unless the step has ``received_outputs`` from a
        subagent (proving a validator actually ran and returned results). This
        prevents the model from bypassing delegated validation by manually
        marking validation steps as completed.

        The flag is set automatically by ``createPlan`` when step descriptions
        match validation-related patterns (e.g., "validate", "verify",
        "tier-N validation"), or explicitly via the ``validation_required``
        parameter in ``createPlan`` step objects.

        To complete a validation step, the model must:
        1. Spawn a validator subagent
        2. Use ``addDependentStep`` to link the validation step to the subagent
        3. Wait for the subagent to call ``completeStepWithOutput``
        4. The step auto-unblocks with ``received_outputs`` as evidence
    """

    step_id: str
    sequence: int  # 1-based ordering
    description: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[str] = None  # ISO8601
    completed_at: Optional[str] = None  # ISO8601
    result: Optional[str] = None  # Outcome or notes
    error: Optional[str] = None  # Error message if failed

    # === Validation enforcement ===

    # When True, setStepStatus(status='completed') is rejected unless the step
    # has received_outputs from a subagent (proving a validator actually ran).
    # Set automatically from description patterns or explicitly by the caller.
    validation_required: bool = False

    # === Cross-agent collaboration fields ===

    # Dependencies on other agents' tasks
    depends_on: List[TaskRef] = field(default_factory=list)

    # Named output key (for dependent tasks to reference by name)
    provides: Optional[str] = None

    # Structured output when completed (passed to dependent steps)
    output: Optional[Dict[str, Any]] = None

    # Current blocking state (subset of depends_on that are unmet)
    blocked_by: List[TaskRef] = field(default_factory=list)

    # Outputs received from completed dependencies
    # Key format: "agent_id:step_id" or the 'provides' name
    received_outputs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        sequence: int,
        description: str,
        depends_on: Optional[List[TaskRef]] = None,
        provides: Optional[str] = None
    ) -> 'TodoStep':
        """Create a new step with auto-generated ID.

        Args:
            sequence: The step's position in the plan (1-based).
            description: Human-readable description of the step.
            depends_on: Optional list of TaskRefs this step depends on.
            provides: Optional name for this step's output.

        Returns:
            A new TodoStep instance.
        """
        step = cls(
            step_id=str(uuid.uuid4()),
            sequence=sequence,
            description=description,
            provides=provides,
        )
        if depends_on:
            step.depends_on = depends_on.copy()
            step.blocked_by = depends_on.copy()
            step.status = StepStatus.BLOCKED
        return step

    def is_blocked(self) -> bool:
        """Check if step is waiting on unmet dependencies."""
        return len(self.blocked_by) > 0

    def add_dependency(self, ref: TaskRef) -> None:
        """Add a dependency on another agent's task.

        Args:
            ref: Reference to the task this step depends on.
        """
        # Avoid duplicates
        for existing in self.depends_on:
            if existing == ref:
                return

        self.depends_on.append(ref)
        self.blocked_by.append(ref)

        # Mark as blocked if not already completed/failed/skipped
        if self.status in (StepStatus.PENDING, StepStatus.BLOCKED):
            self.status = StepStatus.BLOCKED

    def resolve_dependency(
        self,
        ref: TaskRef,
        output: Optional[Dict[str, Any]] = None,
        provides_name: Optional[str] = None
    ) -> bool:
        """Mark a dependency as resolved.

        Called when a dependency completes. Stores its output and
        removes it from blocked_by.

        Args:
            ref: The TaskRef that completed.
            output: Optional structured output from the completed step.
            provides_name: Optional 'provides' name from the completed step.

        Returns:
            True if this step is now unblocked (all dependencies met).
        """
        # Find and remove from blocked_by
        removed = False
        for i, blocked in enumerate(self.blocked_by):
            if blocked.matches(ref.agent_id, ref.plan_id or "", ref.step_id):
                self.blocked_by.pop(i)
                removed = True
                break

        if not removed:
            # Not in our blocked_by list
            return len(self.blocked_by) == 0

        # Store output if provided
        if output is not None:
            # Use provides_name as key if given, otherwise use agent:step_id
            if provides_name:
                self.received_outputs[provides_name] = output
            key = f"{ref.agent_id}:{ref.step_id}"
            self.received_outputs[key] = output

        # Check if now unblocked
        is_unblocked = len(self.blocked_by) == 0
        if is_unblocked and self.status == StepStatus.BLOCKED:
            self.status = StepStatus.PENDING

        return is_unblocked

    def get_blocking_refs(self) -> List[TaskRef]:
        """Get list of unmet dependencies."""
        return self.blocked_by.copy()

    def start(self) -> None:
        """Mark step as in progress."""
        self.status = StepStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc).isoformat() + "Z"

    def complete(
        self,
        result: Optional[str] = None,
        output: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark step as completed.

        Args:
            result: Optional text result/notes.
            output: Optional structured output to pass to dependent steps.
        """
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
        if result:
            self.result = result
        if output is not None:
            self.output = output

    def fail(self, error: Optional[str] = None) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
        if error:
            self.error = error

    def skip(self, reason: Optional[str] = None) -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
        if reason:
            self.result = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "step_id": self.step_id,
            "sequence": self.sequence,
            "description": self.description,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }
        # Include validation_required if set
        if self.validation_required:
            result["validation_required"] = True
        # Include collaboration fields if present
        if self.depends_on:
            result["depends_on"] = [ref.to_dict() for ref in self.depends_on]
        if self.provides:
            result["provides"] = self.provides
        if self.output is not None:
            result["output"] = self.output
        if self.blocked_by:
            result["blocked_by"] = [ref.to_dict() for ref in self.blocked_by]
        if self.received_outputs:
            result["received_outputs"] = self.received_outputs
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TodoStep':
        """Create from dictionary."""
        status_str = data.get("status", "pending")
        try:
            status = StepStatus(status_str)
        except ValueError:
            status = StepStatus.PENDING

        # Parse collaboration fields
        depends_on = [
            TaskRef.from_dict(ref) for ref in data.get("depends_on", [])
        ]
        blocked_by = [
            TaskRef.from_dict(ref) for ref in data.get("blocked_by", [])
        ]

        return cls(
            step_id=data.get("step_id", str(uuid.uuid4())),
            sequence=data.get("sequence", 0),
            description=data.get("description", ""),
            status=status,
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
            validation_required=data.get("validation_required", False),
            depends_on=depends_on,
            provides=data.get("provides"),
            output=data.get("output"),
            blocked_by=blocked_by,
            received_outputs=data.get("received_outputs", {}),
        )


@dataclass
class TodoPlan:
    """A plan consisting of ordered steps."""

    plan_id: str
    created_at: str  # ISO8601
    title: str
    steps: List[TodoStep] = field(default_factory=list)
    current_step: Optional[int] = None  # Current sequence number
    status: PlanStatus = PlanStatus.ACTIVE
    started: bool = False  # True after user approves via startPlan
    started_at: Optional[str] = None  # ISO8601 - when startPlan was approved
    completed_at: Optional[str] = None  # ISO8601
    summary: Optional[str] = None  # Final outcome summary
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        title: str,
        step_descriptions: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> 'TodoPlan':
        """Create a new plan with auto-generated ID and steps."""
        steps = [
            TodoStep.create(i + 1, desc)
            for i, desc in enumerate(step_descriptions)
        ]
        return cls(
            plan_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat() + "Z",
            title=title,
            steps=steps,
            context=context or {},
        )

    def get_step_by_id(self, step_id: str) -> Optional[TodoStep]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_step_by_sequence(self, sequence: int) -> Optional[TodoStep]:
        """Get a step by its sequence number (1-based)."""
        for step in self.steps:
            if step.sequence == sequence:
                return step
        return None

    def get_current_step(self) -> Optional[TodoStep]:
        """Get the current step being worked on."""
        if self.current_step:
            return self.get_step_by_sequence(self.current_step)
        return None

    def get_next_pending_step(self) -> Optional[TodoStep]:
        """Get the next pending step in sequence order.

        Note: Blocked steps are not considered pending.
        """
        for step in sorted(self.steps, key=lambda s: s.sequence):
            if step.status == StepStatus.PENDING:
                return step
        return None

    def get_blocked_steps(self) -> List[TodoStep]:
        """Get all steps that are blocked waiting on dependencies."""
        return [s for s in self.steps if s.status == StepStatus.BLOCKED]

    def get_step_by_provides(self, provides_name: str) -> Optional[TodoStep]:
        """Get a step by its 'provides' name."""
        for step in self.steps:
            if step.provides == provides_name:
                return step
        return None

    def resolve_dependencies_from(
        self,
        completed_agent: str,
        completed_plan_id: str,
        completed_step_id: str,
        output: Optional[Dict[str, Any]] = None,
        provides_name: Optional[str] = None
    ) -> List[TodoStep]:
        """Resolve dependencies on a completed step from another agent.

        Called when a cross-agent dependency completes. Updates all
        blocked steps that were waiting on this dependency.

        Args:
            completed_agent: Agent ID of the completed step.
            completed_plan_id: Plan ID of the completed step.
            completed_step_id: Step ID that completed.
            output: Optional output from the completed step.
            provides_name: Optional 'provides' name from the completed step.

        Returns:
            List of steps that became unblocked.
        """
        unblocked = []
        ref = TaskRef(
            agent_id=completed_agent,
            plan_id=completed_plan_id,
            step_id=completed_step_id
        )

        for step in self.steps:
            if step.status == StepStatus.BLOCKED:
                if step.resolve_dependency(ref, output, provides_name):
                    unblocked.append(step)

        return unblocked

    def add_step(
        self,
        description: str,
        after_step_id: Optional[str] = None,
        depends_on: Optional[List[TaskRef]] = None,
        provides: Optional[str] = None
    ) -> TodoStep:
        """Add a new step to the plan.

        Args:
            description: Description of the new step.
            after_step_id: If provided, insert after this step. Otherwise append to end.
            depends_on: Optional list of cross-agent dependencies.
            provides: Optional name for this step's output.

        Returns:
            The newly created TodoStep.
        """
        if after_step_id:
            # Find the step to insert after
            after_step = self.get_step_by_id(after_step_id)
            if after_step:
                insert_sequence = after_step.sequence + 1
                # Re-sequence all steps at or after the insert position
                for step in self.steps:
                    if step.sequence >= insert_sequence:
                        step.sequence += 1
            else:
                # Step not found, append to end
                insert_sequence = len(self.steps) + 1
        else:
            # Append to end
            insert_sequence = len(self.steps) + 1

        new_step = TodoStep.create(
            sequence=insert_sequence,
            description=description,
            depends_on=depends_on,
            provides=provides
        )
        self.steps.append(new_step)
        # Sort steps by sequence for consistent ordering
        self.steps.sort(key=lambda s: s.sequence)
        return new_step

    def get_progress(self) -> Dict[str, Any]:
        """Get progress statistics for the plan."""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in self.steps if s.status == StepStatus.SKIPPED)
        in_progress = sum(1 for s in self.steps if s.status == StepStatus.IN_PROGRESS)
        pending = sum(1 for s in self.steps if s.status == StepStatus.PENDING)
        blocked = sum(1 for s in self.steps if s.status == StepStatus.BLOCKED)

        percent = (completed / total * 100) if total > 0 else 0

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "in_progress": in_progress,
            "pending": pending,
            "blocked": blocked,
            "percent": round(percent, 1),
        }

    def complete_plan(self, summary: Optional[str] = None) -> None:
        """Mark the plan as completed."""
        self.status = PlanStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
        if summary:
            self.summary = summary

    def fail_plan(self, summary: Optional[str] = None) -> None:
        """Mark the plan as failed."""
        self.status = PlanStatus.FAILED
        self.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
        if summary:
            self.summary = summary

    def cancel_plan(self, summary: Optional[str] = None) -> None:
        """Mark the plan as cancelled."""
        self.status = PlanStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
        if summary:
            self.summary = summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "title": self.title,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "status": self.status.value,
            "started": self.started,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TodoPlan':
        """Create from dictionary."""
        status_str = data.get("status", "active")
        try:
            status = PlanStatus(status_str)
        except ValueError:
            status = PlanStatus.ACTIVE

        steps = [
            TodoStep.from_dict(s)
            for s in data.get("steps", [])
        ]

        return cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat() + "Z"),
            title=data.get("title", ""),
            steps=steps,
            current_step=data.get("current_step"),
            status=status,
            started=data.get("started", False),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            summary=data.get("summary"),
            context=data.get("context", {}),
        )


@dataclass
class ProgressEvent:
    """An event representing progress in a plan.

    Used for reporting to channels.
    """

    event_id: str
    timestamp: str  # ISO8601
    event_type: str  # plan_created, step_started, step_completed, etc.
    plan_id: str
    plan_title: str
    step: Optional[TodoStep] = None
    progress: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: str,
        plan: TodoPlan,
        step: Optional[TodoStep] = None
    ) -> 'ProgressEvent':
        """Create a new progress event."""
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            event_type=event_type,
            plan_id=plan.plan_id,
            plan_title=plan.title,
            step=step,
            progress=plan.get_progress(),
            context=plan.context,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "plan_id": self.plan_id,
            "plan_title": self.plan_title,
            "step": self.step.to_dict() if self.step else None,
            "progress": self.progress,
            "context": self.context,
        }
