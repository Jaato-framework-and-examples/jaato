"""Data models for the TODO plugin.

Defines the core data structures for plans, steps, and progress tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class StepStatus(Enum):
    """Possible statuses for a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(Enum):
    """Possible statuses for an overall plan."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TodoStep:
    """A single step within a plan."""

    step_id: str
    sequence: int  # 1-based ordering
    description: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[str] = None  # ISO8601
    completed_at: Optional[str] = None  # ISO8601
    result: Optional[str] = None  # Outcome or notes
    error: Optional[str] = None  # Error message if failed

    @classmethod
    def create(cls, sequence: int, description: str) -> 'TodoStep':
        """Create a new step with auto-generated ID."""
        return cls(
            step_id=str(uuid.uuid4()),
            sequence=sequence,
            description=description,
        )

    def start(self) -> None:
        """Mark step as in progress."""
        self.status = StepStatus.IN_PROGRESS
        self.started_at = datetime.utcnow().isoformat() + "Z"

    def complete(self, result: Optional[str] = None) -> None:
        """Mark step as completed."""
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        if result:
            self.result = result

    def fail(self, error: Optional[str] = None) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        if error:
            self.error = error

    def skip(self, reason: Optional[str] = None) -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        if reason:
            self.result = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "sequence": self.sequence,
            "description": self.description,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TodoStep':
        """Create from dictionary."""
        status_str = data.get("status", "pending")
        try:
            status = StepStatus(status_str)
        except ValueError:
            status = StepStatus.PENDING

        return cls(
            step_id=data.get("step_id", str(uuid.uuid4())),
            sequence=data.get("sequence", 0),
            description=data.get("description", ""),
            status=status,
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
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
            created_at=datetime.utcnow().isoformat() + "Z",
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
        """Get the next pending step in sequence order."""
        for step in sorted(self.steps, key=lambda s: s.sequence):
            if step.status == StepStatus.PENDING:
                return step
        return None

    def add_step(self, description: str, after_step_id: Optional[str] = None) -> TodoStep:
        """Add a new step to the plan.

        Args:
            description: Description of the new step.
            after_step_id: If provided, insert after this step. Otherwise append to end.

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

        new_step = TodoStep.create(sequence=insert_sequence, description=description)
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

        percent = (completed / total * 100) if total > 0 else 0

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "in_progress": in_progress,
            "pending": pending,
            "percent": round(percent, 1),
        }

    def complete_plan(self, summary: Optional[str] = None) -> None:
        """Mark the plan as completed."""
        self.status = PlanStatus.COMPLETED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        if summary:
            self.summary = summary

    def fail_plan(self, summary: Optional[str] = None) -> None:
        """Mark the plan as failed."""
        self.status = PlanStatus.FAILED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        if summary:
            self.summary = summary

    def cancel_plan(self, summary: Optional[str] = None) -> None:
        """Mark the plan as cancelled."""
        self.status = PlanStatus.CANCELLED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
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
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
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

    Used for reporting to actors.
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
            timestamp=datetime.utcnow().isoformat() + "Z",
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
