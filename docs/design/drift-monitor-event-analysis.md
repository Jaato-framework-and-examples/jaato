# Drift Monitor Event Contract Analysis

Analysis of the todo plugin's current event infrastructure against the drift
monitor's requirements. The drift monitor measures whether the agent is staying
on-task by embedding step descriptions and agent actions into a shared vector
space and computing cosine similarity each turn.

## Event Contract Requirements

The drift monitor requires four event types:

| Event | Trigger | Key Payload Fields |
|-------|---------|-------------------|
| `todo:step_started` | Step → IN_PROGRESS | step_id, step_description, step_index (0-based), plan_description |
| `todo:step_completed` | Step → COMPLETED | step_id, step_description, step_index, outcome_summary |
| `todo:step_changed` | Description modified while IN_PROGRESS | step_id, old_description, new_description |
| `todo:plan_created` | Plan first established | plan_id, plan_description, steps[] |

## Current State Assessment

### 1. Event Bus — Fully Supported

The todo plugin emits structured `TaskEvent` instances (a `@dataclass`) on a
shared `EventBus` via `_publish_event()` (`plugin.py:1732`). The bus supports
subscription filtering via `EventFilter` (by event type, agent, plan, step) and
fire-and-forget semantics — the publisher does not wait for subscribers.

The drift monitor can subscribe with zero changes to the todo plugin:

```python
from jaato_sdk.event_bus import EventType, EventFilter

filter = EventFilter(event_types=[
    EventType.PLAN_CREATED,
    EventType.STEP_STARTED,
    EventType.STEP_COMPLETED,
])
event_bus.subscribe(filter, agent_id="drift_monitor")
```

Callbacks receive `Event` dataclass instances with `Dict[str, Any]` payloads —
no framework coupling.

### 2. Step Identity (`step_id`) — Fully Supported

Every `TodoStep` receives a stable UUID via `uuid.uuid4()` at creation time.
This ID is immutable — it persists across status transitions, reordering, and
step insertions. The `sequence` field (1-based) may shift, but `step_id` never
changes.

Events include `source_step_id` on the `TaskEvent` envelope.

### 3. Plan-Level Description — Fully Supported

`TodoPlan.title` stores the plan-level description and is included in every
event as `source_plan_title` on the `TaskEvent` envelope. The `PLAN_CREATED`
event payload also includes the full steps array.

**Mapping to drift monitor fields:**

| Drift Monitor | Current | Source |
|---------------|---------|--------|
| `plan_description` | `source_plan_title` | `TaskEvent` envelope |
| `plan_id` | `source_plan_id` | `TaskEvent` envelope |
| `step_description` | `source_step_description` | `TaskEvent` envelope |
| `step_id` | `source_step_id` | `TaskEvent` envelope |
| `step_index` (0-based) | `source_step_sequence` (1-based) | `TaskEvent` envelope |

### 4. Outcome Tracking — Fully Supported

When a step is completed, `step.result` captures the outcome text (set via
`setStepStatus(result='...')` or `completeStepWithOutput(result='...')`). The
`STEP_COMPLETED` event payload already includes `"result": step.result`.

Maps to `outcome_summary` → `payload["result"]`. Can be `None`, matching the
drift monitor's `str | None` contract.

### 5. Step Mutation Detection — NOT Supported (New Capability)

Step descriptions are **immutable after creation**. There is no
`update_description()` method, no mutation tracking, and no `STEP_CHANGED`
event type in the `EventType` enum.

To add this:
1. Add `STEP_CHANGED = "step_changed"` to `EventType` in `jaato_sdk/event_bus.py`
2. Add a description update mechanism (new tool `updateStep` or parameter on
   `setStepStatus`)
3. Emit the event with `{old_description, new_description}` in the payload

Since descriptions are currently immutable, this event would never fire today.
It only becomes relevant if/when step-editing capability is added.

### 6. Integration Surface — Well-Suited

The drift monitor subscribes as a standard `EventBus` subscriber. No access to
todo plugin internals needed. The `EventBus` is available via the
`TaskEventBus` wrapper (set on the plugin during initialization) or directly
from the shared runtime.

## Summary Matrix

| Requirement | Status | Work Needed |
|-------------|--------|-------------|
| Event bus (fire-and-forget publish) | **Supported** | None |
| `todo:plan_created` with steps array | **Supported** | None |
| `todo:step_started` with step metadata | **Supported** | None |
| `todo:step_completed` with outcome | **Supported** | `payload["result"]` = outcome_summary |
| Stable `step_id` (UUID) | **Supported** | None |
| `plan_description` (top-level goal) | **Supported** | `source_plan_title` on every event |
| `step_index` (0-based) | **Minor addition** | `source_step_sequence` is 1-based |
| `outcome_summary` on completion | **Supported** | Already in `payload["result"]` |
| `todo:step_changed` (mutation detection) | **Not supported** | New EventType, new tool, new emission |
| Plain dict/dataclass payloads | **Supported** | `Event` is `@dataclass`, payload is dict |

### Already Supported (Out of the Box)

- Full event bus infrastructure with subscription filtering
- `PLAN_CREATED`, `STEP_STARTED`, `STEP_COMPLETED` events at correct lifecycle
  hooks
- Stable UUIDs for step identity
- Plan title on every event (`source_plan_title`)
- Outcome tracking via `step.result` in completion payloads
- Fire-and-forget semantics
- Dataclass events with dict payloads (no framework coupling)

### Minor Additions

- **0-based `step_index`**: `source_step_sequence` is 1-based. Either add
  `step_index` to event payloads (`step.sequence - 1`) or document that
  consumers subtract 1. Recommend adding to payload only (not the envelope) to
  avoid disrupting the 1-based convention used throughout the codebase.
- **`plan_description` in step payloads**: Already on the `TaskEvent` envelope
  as `source_plan_title`. If the drift monitor reads the full `Event` dataclass,
  no change needed. If it only reads `payload`, add `plan_description` to step
  event payloads.

### New Capability Required

- **`todo:step_changed`**: Step descriptions are immutable today. Adding mutation
  detection requires a new `EventType`, a mechanism to update descriptions, and
  event emission with old/new values. This is the only design-level gap.

## Concerns and Constraints

1. **1-based sequence convention**: The 1-based `sequence` is used consistently
   throughout storage, display, and APIs. Adding a parallel 0-based `step_index`
   in event payloads is fine; changing the envelope field would break existing
   subscribers.

2. **Step immutability is intentional**: Descriptions are immutable to simplify
   cross-agent dependency tracking (descriptions are part of the inter-agent
   contract). Adding mutability needs careful design around dependency
   references. If the drift monitor is the sole consumer of `step_changed`,
   weigh the complexity cost.

3. **Thread safety**: The event bus is thread-safe (`threading.Lock`).
   Subscription callbacks are invoked on the publishing thread. The drift
   monitor should offload embedding computation to its own thread to avoid
   blocking todo plugin state transitions.

4. **`plan_description` vs original user request**: `TodoPlan.title` is set by
   the model when calling `createPlan(title=...)`. If the drift monitor needs
   the raw user request (before the model interpreted it), that comes from the
   session's message history, not the todo plugin.
