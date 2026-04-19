"""Payload schemas for EventBus events.

Each ``TypedDict`` defines the canonical payload schema for one or more
``EventType`` values.  These are the **single source of truth** for what
keys a bus event's ``payload`` dict contains.

Groups:

1. **Plan/step lifecycle** — produced by the todo plugin via
   ``TaskEvent.create()``.  All include ``plan_id`` and ``plan_title``;
   step-level events add ``step_id``, ``step_description``, ``step_sequence``.

2. **Bridged from server events** — produced by ``_server_event_to_bus_event()``
   in ``server/core.py``, which flattens a server event dataclass into a
   payload dict (all fields except ``type`` and ``timestamp``).
   A sync test verifies these TypedDicts stay aligned with their corresponding
   server event dataclass fields.

3. **Plugin-originated** — published directly by plugins (webhook, drift monitor).

Usage::

    from jaato_sdk.event_payloads import StepCompletedPayload

    def on_event(event: Event) -> None:
        payload: StepCompletedPayload = event.payload
        step_id = payload["step_id"]  # IDE completion works
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, NotRequired


# =============================================================================
# Group 1: Plan/step lifecycle (todo plugin → EventBus)
#
# TaskEvent.create() always adds: plan_id, plan_title.
# If a step is provided: step_id, step_description, step_sequence.
# Callers add custom keys via the payload= parameter.
# =============================================================================

class _PlanBase(TypedDict):
    """Common fields for all plan-level events."""
    plan_id: str
    plan_title: str


class PlanCreatedPayload(_PlanBase):
    """EventType.PLAN_CREATED — new plan created.

    ``steps`` contains the initial step list (id, sequence, description).
    """
    steps: List[Dict[str, Any]]


class PlanStartedPayload(_PlanBase):
    """EventType.PLAN_STARTED — plan execution approved by user."""


class _PlanTerminalBase(_PlanBase):
    """Common fields for plan completion/failure/cancellation."""
    summary: NotRequired[Optional[str]]
    progress: NotRequired[Optional[Dict[str, Any]]]


class PlanCompletedPayload(_PlanTerminalBase):
    """EventType.PLAN_COMPLETED — plan finished successfully."""


class PlanFailedPayload(_PlanTerminalBase):
    """EventType.PLAN_FAILED — plan failed."""


class PlanCancelledPayload(_PlanTerminalBase):
    """EventType.PLAN_CANCELLED — plan cancelled by user or agent."""


class _StepBase(_PlanBase):
    """Common fields for all step-level events."""
    step_id: str
    step_description: str
    step_sequence: int


class StepAddedPayload(_StepBase):
    """EventType.STEP_ADDED — new step added to an existing plan."""


class StepStartedPayload(_StepBase):
    """EventType.STEP_STARTED — step transitioned to in_progress."""


class StepCompletedPayload(_StepBase):
    """EventType.STEP_COMPLETED — step transitioned to completed."""
    result: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]
    output: NotRequired[Optional[Any]]
    provides: NotRequired[Optional[str]]


class StepFailedPayload(_StepBase):
    """EventType.STEP_FAILED — step failed."""
    result: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]


class StepSkippedPayload(_StepBase):
    """EventType.STEP_SKIPPED — step skipped."""
    result: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]


class StepBlockedPayload(_StepBase):
    """EventType.STEP_BLOCKED — step waiting on cross-agent dependencies."""
    blocked_by: List[Dict[str, Any]]


class StepUnblockedPayload(_StepBase):
    """EventType.STEP_UNBLOCKED — all dependencies resolved."""
    received_outputs: List[str]
    unblocked_by: Dict[str, Any]


# =============================================================================
# Group 2: Bridged from server events (server.emit → EventBus)
#
# These mirror the server event dataclass fields (minus type/timestamp).
# A sync test in test_event_payloads.py verifies they stay aligned.
#
# Source server event class is noted in each docstring.
# =============================================================================

class AgentCreatedPayload(TypedDict):
    """EventType.AGENT_CREATED — new agent spawned.

    Source: ``AgentCreatedEvent``
    """
    agent_id: str
    agent_name: str
    agent_type: str
    profile_name: NotRequired[Optional[str]]
    parent_agent_id: NotRequired[Optional[str]]
    created_at: NotRequired[Optional[str]]


class AgentOutputPayload(TypedDict):
    """EventType.AGENT_OUTPUT — streaming text from an agent.

    Source: ``AgentOutputEvent``
    """
    agent_id: str
    source: str
    text: str
    mode: str


class AgentStatusChangedPayload(TypedDict):
    """EventType.AGENT_STATUS_CHANGED — agent status transition.

    Source: ``AgentStatusChangedEvent``
    """
    agent_id: str
    status: str
    error: NotRequired[Optional[str]]


class AgentCompletedPayload(TypedDict):
    """EventType.AGENT_COMPLETED — agent finished its task.

    Source: ``AgentCompletedEvent``

    The ``payload`` field carries the validated typed payload from
    ``signal_completion`` when the agent's profile declared a
    ``completion_payload_schema``. Absent (or ``None``) when the
    profile uses the legacy untyped ``summary`` parameter.
    """
    agent_id: str
    completed_at: str
    success: bool
    token_usage: NotRequired[Optional[Dict[str, int]]]
    turns_used: NotRequired[Optional[int]]
    error: NotRequired[str]
    payload: NotRequired[Optional[Dict[str, Any]]]


class ToolCallStartedPayload(TypedDict):
    """EventType.TOOL_CALL_STARTED — tool execution began.

    Source: ``ToolCallStartEvent``
    """
    agent_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    call_id: NotRequired[Optional[str]]


class ToolCallCompletedPayload(TypedDict):
    """EventType.TOOL_CALL_COMPLETED — tool execution finished.

    Source: ``ToolCallEndEvent``
    """
    agent_id: str
    tool_name: str
    call_id: NotRequired[Optional[str]]
    success: bool
    duration_seconds: float
    error_message: NotRequired[Optional[str]]
    backgrounded: NotRequired[bool]
    continuation_id: NotRequired[Optional[str]]
    show_output: NotRequired[Optional[bool]]
    show_popup: NotRequired[Optional[bool]]


class ToolOutputPayload(TypedDict):
    """EventType.TOOL_OUTPUT — live output chunk from a running tool.

    Source: ``ToolOutputEvent``
    """
    agent_id: str
    call_id: str
    chunk: str


class PlanStepUpdatedPayload(TypedDict):
    """EventType.PLAN_STEP_UPDATED — lean delta for a single step status change.

    Source: ``PlanStepUpdatedEvent``
    """
    agent_id: str
    step_id: str
    sequence: int
    content: str
    status: str
    result: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]
    blocked_by: NotRequired[Optional[List[Dict[str, Any]]]]
    depends_on: NotRequired[Optional[List[Dict[str, Any]]]]
    received_outputs: NotRequired[Optional[Dict[str, Any]]]


class TurnCompletedPayload(TypedDict):
    """EventType.TURN_COMPLETED — conversation turn finished.

    Source: ``TurnCompletedEvent``
    """
    agent_id: str
    turn_number: int
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    duration_seconds: float
    function_calls: List[Dict[str, Any]]
    formatted_text: NotRequired[Optional[str]]
    cache_read_tokens: NotRequired[Optional[int]]
    cache_creation_tokens: NotRequired[Optional[int]]


class TurnProgressPayload(TypedDict):
    """EventType.TURN_PROGRESS — incremental progress during turn.

    Source: ``TurnProgressEvent``
    """
    agent_id: str
    total_tokens: int
    prompt_tokens: int
    output_tokens: int
    context_limit: int
    percent_used: float
    tokens_remaining: int
    pending_tool_calls: int
    cache_read_tokens: NotRequired[Optional[int]]
    cache_creation_tokens: NotRequired[Optional[int]]


class ContextUpdatedPayload(TypedDict):
    """EventType.CONTEXT_UPDATED — context window usage changed.

    Source: ``ContextUpdatedEvent``
    """
    agent_id: str
    total_tokens: int
    prompt_tokens: int
    output_tokens: int
    context_limit: int
    percent_used: float
    tokens_remaining: int
    turns: int
    gc_threshold: NotRequired[Optional[float]]
    gc_strategy: NotRequired[Optional[str]]
    gc_target_percent: NotRequired[Optional[float]]
    gc_continuous_mode: NotRequired[bool]


class PermissionRequestedPayload(TypedDict):
    """EventType.PERMISSION_REQUESTED — permission input mode entered.

    Source: ``PermissionInputModeEvent``
    """
    agent_id: str
    request_id: str
    tool_name: str
    call_id: NotRequired[Optional[str]]
    response_options: List[Dict[str, str]]
    tool_args: NotRequired[Optional[Dict[str, Any]]]
    editable_metadata: NotRequired[Optional[Dict[str, Any]]]


class PermissionResolvedPayload(TypedDict):
    """EventType.PERMISSION_RESOLVED — permission granted or denied.

    Source: ``PermissionResolvedEvent``
    """
    agent_id: str
    request_id: str
    tool_name: str
    granted: bool
    method: str
    comment: str


# =============================================================================
# Group 3: Plugin-originated events
# =============================================================================

class ExternalEventPayload(TypedDict):
    """EventType.EXTERNAL_EVENT — inbound webhook or external event.

    Published by the webhook plugin. ``plan_id`` and ``step_id`` are set
    for EventFilter compatibility (plan_id = route name, step_id = event_id).
    """
    source: str
    event_type: str
    headers: Dict[str, str]
    payload: Any
    plan_id: str
    plan_title: str
    step_id: str
    step_description: str


class DriftMeasuredPayload(TypedDict):
    """EventType.DRIFT_MEASURED — drift measurement from drift monitor plugin.

    Published by the drift monitor plugin at step completion boundaries.
    Compares intent (step description + pre-tool model text) against
    achievement (tool calls + post-tool model text) via embedding similarity.
    """
    step_id: str
    step_description: str
    drift_score: float
    drift_flagged: bool
    drift_flag_type: NotRequired[Optional[str]]
    strategic_drift: NotRequired[Optional[Dict[str, Any]]]


# =============================================================================
# Mapping from EventType to payload TypedDict
# =============================================================================

# Import EventType here to avoid circular imports at module level.
# This mapping is used by the sync test and for documentation.
def get_payload_schema_map() -> Dict[str, type]:
    """Return a mapping from EventType value to payload TypedDict class.

    Deferred import of EventType to avoid circular dependency
    (event_bus.py imports from here would create a cycle).
    """
    from jaato_sdk.event_bus import EventType

    return {
        EventType.PLAN_CREATED.value: PlanCreatedPayload,
        EventType.PLAN_STARTED.value: PlanStartedPayload,
        EventType.PLAN_COMPLETED.value: PlanCompletedPayload,
        EventType.PLAN_FAILED.value: PlanFailedPayload,
        EventType.PLAN_CANCELLED.value: PlanCancelledPayload,
        EventType.STEP_ADDED.value: StepAddedPayload,
        EventType.STEP_STARTED.value: StepStartedPayload,
        EventType.STEP_COMPLETED.value: StepCompletedPayload,
        EventType.STEP_FAILED.value: StepFailedPayload,
        EventType.STEP_SKIPPED.value: StepSkippedPayload,
        EventType.STEP_BLOCKED.value: StepBlockedPayload,
        EventType.STEP_UNBLOCKED.value: StepUnblockedPayload,
        EventType.AGENT_CREATED.value: AgentCreatedPayload,
        EventType.AGENT_OUTPUT.value: AgentOutputPayload,
        EventType.AGENT_STATUS_CHANGED.value: AgentStatusChangedPayload,
        EventType.AGENT_COMPLETED.value: AgentCompletedPayload,
        EventType.TOOL_CALL_STARTED.value: ToolCallStartedPayload,
        EventType.TOOL_CALL_COMPLETED.value: ToolCallCompletedPayload,
        EventType.TOOL_OUTPUT.value: ToolOutputPayload,
        EventType.PLAN_STEP_UPDATED.value: PlanStepUpdatedPayload,
        EventType.TURN_COMPLETED.value: TurnCompletedPayload,
        EventType.TURN_PROGRESS.value: TurnProgressPayload,
        EventType.CONTEXT_UPDATED.value: ContextUpdatedPayload,
        EventType.PERMISSION_REQUESTED.value: PermissionRequestedPayload,
        EventType.PERMISSION_RESOLVED.value: PermissionResolvedPayload,
        EventType.EXTERNAL_EVENT.value: ExternalEventPayload,
        EventType.DRIFT_MEASURED.value: DriftMeasuredPayload,
    }


# =============================================================================
# Mapping from bridged payload TypedDicts to their source server event classes
#
# Used by the sync test to verify field alignment.
# =============================================================================

# Maps payload TypedDict class → server event class name (string to avoid
# importing server event classes here — the test resolves them).
BRIDGED_PAYLOAD_SOURCES: Dict[type, str] = {
    AgentCreatedPayload: "AgentCreatedEvent",
    AgentOutputPayload: "AgentOutputEvent",
    AgentStatusChangedPayload: "AgentStatusChangedEvent",
    AgentCompletedPayload: "AgentCompletedEvent",
    ToolCallStartedPayload: "ToolCallStartEvent",
    ToolCallCompletedPayload: "ToolCallEndEvent",
    ToolOutputPayload: "ToolOutputEvent",
    PlanStepUpdatedPayload: "PlanStepUpdatedEvent",
    TurnCompletedPayload: "TurnCompletedEvent",
    TurnProgressPayload: "TurnProgressEvent",
    ContextUpdatedPayload: "ContextUpdatedEvent",
    PermissionRequestedPayload: "PermissionInputModeEvent",
    PermissionResolvedPayload: "PermissionResolvedEvent",
}
