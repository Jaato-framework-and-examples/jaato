"""Event Protocol for Jaato Server.

This module defines all events for client-server communication.
Events are pydantic models serialised as JSON over WebSocket / IPC.

Event Flow:
    Server -> Client: Status updates, output streaming, permission requests
    Client -> Server: Messages, permission responses, commands

Protocol Version: 1.0

Pydantic was chosen over plain ``@dataclass`` so the same model
definitions can drive JSON Schema export (``Event.model_json_schema``)
for the upcoming TypeScript SDK codegen — single source of truth for
both languages.  Wire format is preserved byte-for-byte versus the
previous dataclass implementation; see
``jaato_sdk/tests/test_events_wire_format.py`` for the snapshot
baselines that gate any drift.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Protocol version
# =============================================================================

# Wire-protocol version, semver-style ``"MAJOR.MINOR"``.
#
# - **MAJOR** bumps when a wire field is removed, renamed, or retyped
#   (anything an existing client cannot ignore-and-keep-working through).
# - **MINOR** bumps when a new optional field is added that older
#   clients can safely ignore (pydantic ``extra='ignore'`` already
#   handles this on the read side).
#
# Clients carry their own ``MIN_PROTOCOL_VERSION`` and refuse to
# connect when the server's major differs from theirs, or when the
# server's minor is below their minimum.  Server's minor *above* the
# client's minimum is fine — the client just won't see the newer
# fields.
#
# DO NOT confuse with the ``server_version`` carried in
# ``ConnectedEvent.server_info`` — that's the daemon package version,
# kept for diagnostics only ("which build is the daemon?").  Compat
# is checked against ``protocol_version`` exclusively from v1.0
# onwards.
#
# See ``docs/sdk-protocol-versioning.md`` for the bump policy and the
# CHANGELOG of past versions.
# 1.1 (2026-08-24): additive optional ``request_id`` on SessionInfoEvent
# and ErrorEvent, so a client can tell WHICH session.new a given answer
# belongs to.  Minor bump per the compat rule -- same major, additive
# optional fields -- so clients declaring 1.0 still connect.
# 1.3 (2026-08-26): additive optional ``request_id`` on InjectPromptRequest
# plus the new ``inject_prompt.result`` event, so an inject can report whether
# the target will ACT on the message rather than only that it was accepted
# into a queue.  Older clients that send no ``request_id`` get the previous
# fire-and-forget behaviour unchanged.
# 1.4 (2026-09-05): additive optional ``stream_id`` / ``sequence`` /
# ``mime_type`` / ``data_b64`` / ``final`` on ToolOutputEvent, carrying
# BINARY media -- a tool's attachments and the model's own speech -- on
# the existing tool-output channel rather than a rival event.  Older
# clients ignore the fields and see the text stream exactly as before,
# so the compat rule holds.  A client that needs to RECEIVE media must
# declare ``min_protocol_version="1.4"``: against a 1.3 daemon the
# fields are simply never sent, which is indistinguishable from a model
# that chose not to speak.
PROTOCOL_VERSION = "1.4"


# =============================================================================
# Event Types
# =============================================================================

class EventType(str, Enum):
    """All event types in the protocol."""

    # Connection lifecycle
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

    # Agent lifecycle (Server -> Client)
    AGENT_CREATED = "agent.created"
    AGENT_OUTPUT = "agent.output"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_COMPLETED = "agent.completed"
    # Recoverable terminal error: emitted when an agent hits a terminal error
    # AFTER the framework's automatic management (with_retry / nudge) is
    # exhausted or never applied.  Gives a reactor first refusal to recover the
    # stage (re-spawn / reroute / escalate) before the session's teardown
    # SessionTerminatedEvent(reason="error") lands.  See
    # docs/design/agent-error-recovery-event.md.
    AGENT_ERROR = "agent.error"

    # Session lifecycle (Server -> Client)
    # Fires when the session has fully wound down — emitted spontaneously
    # after natural completion drains, OR in response to session.end.
    # Replaces the [SESSION_TERMINATED] string-based marker.
    SESSION_TERMINATED = "session.terminated"

    # Cascade stage settled: emitted once per cascade stage at the END of
    # session teardown — on ALL paths (warm slot returned, slot torn down on
    # error, or cold-spawned) — so a reactor can gate the next stage's spawn on
    # one universal, stall-proof event (no timeout).  ``was_warm`` reports
    # whether the next spawn reuses a warm slot.  Cascade sessions only.
    SLOT_SETTLED = "slot.settled"

    # Session lifecycle: emitted on first client-attach to a session
    # that was loaded from disk (Phase 3 §3.12 disk-restore +
    # peer-review M5/N1).  Carries the count of pending tool calls
    # that the daemon held during defer-and-flush so the client can
    # surface a "this session was restored — N pending tool calls
    # to review" prompt.  ``pending_tool_call_count == 0`` is the
    # common case (clean restore with no in-flight work); the event
    # still fires so clients can distinguish a fresh-attach from a
    # restored-attach for telemetry / UX purposes.
    SESSION_RESTORED = "session.restored"

    # Tool execution (Server -> Client)
    TOOL_CALL_START = "tool.call_start"
    TOOL_CALL_END = "tool.call_end"
    TOOL_OUTPUT = "tool.output"  # Live output chunk from running tool

    # Permission flow (Server <-> Client)
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_INPUT_MODE = "permission.input_mode"  # Signal client to enter permission input mode
    PERMISSION_RESOLVED = "permission.resolved"
    PERMISSION_RESPONSE = "permission.response"  # Client -> Server
    PERMISSION_STATUS = "permission.status"  # Server -> Client (status bar update)

    # Clarification flow (Server <-> Client)
    CLARIFICATION_REQUESTED = "clarification.requested"
    CLARIFICATION_INPUT_MODE = "clarification.input_mode"  # Signal client to enter clarification input mode
    CLARIFICATION_QUESTION = "clarification.question"
    CLARIFICATION_RESOLVED = "clarification.resolved"
    CLARIFICATION_RESPONSE = "clarification.response"  # Client -> Server
    CLARIFICATION_BATCH = "clarification.batch"  # Server -> Client: all questions at once (WS only)
    CLARIFICATION_BATCH_RESPONSE = "clarification.batch_response"  # Client -> Server: all answers at once

    # Reference selection flow (Server <-> Client)
    REFERENCE_SELECTION_REQUESTED = "reference_selection.requested"
    REFERENCE_SELECTION_RESOLVED = "reference_selection.resolved"
    REFERENCE_SELECTION_RESPONSE = "reference_selection.response"  # Client -> Server

    # Workspace mismatch flow (Server <-> Client)
    WORKSPACE_MISMATCH_REQUESTED = "workspace_mismatch.requested"
    WORKSPACE_MISMATCH_RESOLVED = "workspace_mismatch.resolved"
    WORKSPACE_MISMATCH_RESPONSE = "workspace_mismatch.response"  # Client -> Server

    # Plan updates (Server -> Client)
    PLAN_UPDATED = "plan.updated"
    PLAN_STEP_UPDATED = "plan.step_updated"  # Lean delta for a single step status change
    PLAN_CLEARED = "plan.cleared"

    # Context/token updates (Server -> Client)
    CONTEXT_UPDATED = "context.updated"
    TURN_COMPLETED = "turn.completed"
    TURN_PROGRESS = "turn.progress"
    INSTRUCTION_BUDGET_UPDATED = "instruction_budget.updated"
    GC_CONFIG = "gc.config"
    GC = "gc"                       # GC lifecycle (phase-switched)

    # Instruction budget (Client <-> Server)
    INSTRUCTION_BUDGET_REQUEST = "instruction_budget.request"  # Client -> Server

    # System messages (Server -> Client)
    SYSTEM_MESSAGE = "system.message"
    HELP_TEXT = "help.text"  # Detailed help output for commands
    ERROR = "error"
    INIT_PROGRESS = "init.progress"  # Initialization step progress
    RETRY = "retry"  # API retry with exponential backoff

    # Session management (Server -> Client)
    SESSION_LIST = "session.list"  # For user display (updates local cache too)
    SESSION_INFO = "session.info"  # Full state snapshot on connect/attach
    SESSION_DESCRIPTION_UPDATED = "session.description_updated"  # Description changed

    # Memory management (Server -> Client)
    MEMORY_LIST = "memory.list"  # Memory list for completion cache and pager display

    # Sandbox management (Server -> Client)
    SANDBOX_PATHS = "sandbox.paths"  # Sandbox allowed paths for @@ completion cache

    # Service management (Server -> Client)
    SERVICE_LIST = "service.list"  # Service list for completion cache

    # Client requests (Client -> Server)
    SEND_MESSAGE = "message.send"
    STOP = "session.stop"
    COMMAND = "command.execute"
    COMMAND_LIST_REQUEST = "command.list_request"

    # Command list (Server -> Client)
    COMMAND_LIST = "command.list"
    COMMAND_LIST_REFRESH = "command.list_refresh"

    # Tool status (Server -> Client)
    TOOL_STATUS = "tools.status"
    TOOL_ID_REGISTRY = "tools.id_registry"  # Hash-derived ID → name mapping for display

    # Tool management (Client -> Server)
    TOOL_DISABLE_REQUEST = "tools.disable"

    # Client-side tool execution (Client <-> Server)
    TOOLS_REGISTER_CLIENT = "tools.register_client"   # Client -> Server
    TOOL_EXECUTE_REQUEST = "tool.execute_request"      # Server -> Client
    TOOL_EXECUTE_RESULT = "tool.execute_result"        # Client -> Server

    # History (Client <-> Server)
    HISTORY_REQUEST = "history.request"
    HISTORY = "history"

    # Client configuration (Client -> Server)
    CLIENT_CONFIG = "client.config"

    # Mid-turn prompts (Server -> Client)
    MID_TURN_PROMPT_QUEUED = "mid_turn_prompt.queued"
    MID_TURN_PROMPT_INJECTED = "mid_turn_prompt.injected"
    MID_TURN_INTERRUPT = "mid_turn_prompt.interrupt"  # Streaming interrupted for user prompt

    # Session recovery (Server -> Client)
    INTERRUPTED_TURN_RECOVERED = "session.interrupted_turn_recovered"  # Turn recovered after reconnect

    # Post-auth setup flow (Server <-> Client)
    POST_AUTH_SETUP = "auth.setup"  # Server -> Client: offer session setup after auth
    POST_AUTH_SETUP_RESPONSE = "auth.setup_response"  # Client -> Server: user's choices

    # Workspace management (Client <-> Server)
    WORKSPACE_LIST_REQUEST = "workspace.list"  # Client -> Server
    WORKSPACE_LIST = "workspace.list_response"  # Server -> Client
    WORKSPACE_CREATE_REQUEST = "workspace.create"  # Client -> Server
    WORKSPACE_CREATED = "workspace.created"  # Server -> Client
    WORKSPACE_SELECT_REQUEST = "workspace.select"  # Client -> Server
    CONFIG_STATUS = "config.status"  # Server -> Client (response to workspace.select)
    CONFIG_UPDATE_REQUEST = "config.update"  # Client -> Server
    CONFIG_UPDATED = "config.updated"  # Server -> Client

    # File staging into a workspace (Client <-> Server, WS only).
    # Multi-frame protocol: client sends one TEXT frame with this request
    # carrying file metadata, then N raw BINARY frames (one per file) in
    # declared order.  Server responds with WORKSPACE_FILES_STAGED.  See
    # docs/sdk-file-staging.md for the wire protocol.
    WORKSPACE_FILES_STAGE_REQUEST = "workspace.files.stage_request"  # Client -> Server
    WORKSPACE_FILES_STAGED = "workspace.files.staged"  # Server -> Client

    # Agent profiles (Client <-> Server)
    SESSION_PROFILES = "session.profiles"  # Server -> Client: available profiles

    # Workspace file monitoring (Server -> Client)
    WORKSPACE_FILES_CHANGED = "workspace.files_changed"  # Incremental delta
    WORKSPACE_FILES_SNAPSHOT = "workspace.files_snapshot"  # Full state on reconnect

    # External events (Client -> Server, from web components)
    EVENT_EXTERNAL = "event.external"

    # SDK feature parity (Client -> Server) — typed verbs over the
    # public-side primitives JaatoSession.inject_prompt /
    # replay_messages / resolve_fork_point.  The premium
    # ``session_ops`` plugin builds higher-level model-callable tools
    # on top of these same primitives; the WS verbs let SDK consumers
    # reach the primitives directly.  See
    # ``project_backlog_sdk_feature_parity.md``.
    INJECT_PROMPT_REQUEST = "inject_prompt.request"   # Client -> Server
    INJECT_PROMPT_RESULT = "inject_prompt.result"     # Server -> Client
    REPLAY_MESSAGES_REQUEST = "replay_messages.request"  # Client -> Server
    REPLAY_MESSAGES_RESULT = "replay_messages.result"    # Server -> Client
    RESOLVE_FORK_POINT_REQUEST = "resolve_fork_point.request"  # Client -> Server
    RESOLVE_FORK_POINT_RESULT = "resolve_fork_point.result"    # Server -> Client

    # Wake primitive: result of session.bind_wake / session.unbind_wake
    WAKE_BIND_RESULT = "session.wake_bind_result"              # Server -> Client
    # Wake primitive: a wake arrived for a session with no attached client —
    # revived server-side; the woken turn is DEFERRED until a client re-attaches.
    SESSION_WOKEN = "session.woken"                            # Server -> Client

    # SDK feature parity — typed permission-policy verbs replacing
    # stringly-typed CommandRequest("permissions", [...]) for SDK
    # consumers.  CLI command path stays for actual users.
    PERMISSION_ADD_WHITELIST_REQUEST = "permission.add_whitelist"
    PERMISSION_ADD_BLACKLIST_REQUEST = "permission.add_blacklist"
    PERMISSION_REMOVE_REQUEST = "permission.remove"
    PERMISSION_CLEAR_REQUEST = "permission.clear"
    PERMISSION_SET_DEFAULT_REQUEST = "permission.set_default"
    PERMISSION_POLICY_SNAPSHOT_REQUEST = "permission.policy_snapshot.request"  # Client -> Server
    PERMISSION_POLICY_SNAPSHOT = "permission.policy_snapshot"                  # Server -> Client

    # Event subscription notifications (Server -> Client)
    EVENTS_SUBSCRIBED = "events.subscribed"

    # Peer channel events (server-to-server gossip).
    #
    # NOTE: "peer" HERE means another jaato SERVER — this is the
    # federation/gossip channel consumed by the premium gossip
    # extension (jaato_premium/gossip/).  It is NOT the sibling-session
    # coordination surface (``list_peers`` / ``send_to_peer``, which are
    # TOOLS scoped to one cascade_driver_id and emit no events).  The two
    # never share an identifier, but a reader running ``explain events``
    # while building session coordination lands here first.
    PEER_HEARTBEAT = "peer.heartbeat"
    PEER_SPAWN_REQUEST = "peer.spawn_request"
    PEER_SPAWN_ACCEPTED = "peer.spawn_accepted"
    PEER_SPAWN_REJECTED = "peer.spawn_rejected"
    PEER_AGENT_OUTPUT = "peer.agent_output"
    PEER_AGENT_COMPLETED = "peer.agent_completed"
    PEER_STOP_REQUEST = "peer.stop_request"
    PEER_STOP_ACKNOWLEDGED = "peer.stop_acknowledged"

    # HandoffGate events — emitted by the jaato-premium reactor framework
    # when a daemon-side gate transitions state.  See docs/sdk/gate-events.md
    # for the public wire contract; full design lives in
    # jaato-premium/docs/design/handoff-gate-api.md.  Wire types are
    # pre-registered here in the public SDK so any client can deserialize
    # them; production of these events is gated on premium being installed.
    GATE_ANNOUNCED = "gate.announced"
    GATE_RELEASED = "gate.released"
    GATES_SNAPSHOT = "gates.snapshot"


# =============================================================================
# Base Event
# =============================================================================

class Event(BaseModel):
    """Base class for all events.

    ``model_config['extra'] = 'ignore'`` mirrors the previous
    ``deserialize_event`` behaviour of silently dropping unknown
    fields — preserves forward compatibility when an older client
    receives an event from a newer server with extra keys.
    """

    model_config = ConfigDict(extra='ignore')

    type: EventType
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # WHICH SESSION THIS EVENT IS ABOUT (protocol 1.2+).
    #
    # Stamped centrally by ``SessionManager._emit_to_session`` — the one
    # fan-out chokepoint that knows the session and feeds BOTH the
    # direct-attach clients and the cascade-observer dispatch — so no
    # emit site has to remember, and an observer can attribute every
    # event it receives regardless of type.
    #
    # Before this, 12 of 112 event types declared their own
    # ``session_id`` and the rest had none.  The split was not random:
    # LIFECYCLE events (created / woken / restored / terminated) carried
    # it; ACTIVITY events (turn, tool, agent output) did not.  So a
    # cascade observer could watch sessions appear, sleep and die — but
    # never watch them WORK, which is most of what an observer is for.
    # Two siblings running under one cid were indistinguishable on the
    # bus, because ``agent_id`` is ``"main"`` for every top-level
    # session.
    #
    # Reading it with ``getattr(ev, "session_id", "")`` — the idiom the
    # framework's own client scaffold used to teach — could not tell
    # "this event type has no such field" from "the field is blank".
    # Declaring it here collapses that to ONE meaning: empty means the
    # event did not travel a routed path.
    #
    # An emitter may set it explicitly, and the router NEVER overwrites a
    # populated value: an event ABOUT another session (e.g.
    # ``SlotSettledEvent``, "the session that just ended") must keep its
    # own subject rather than being relabelled with whoever emitted it.
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Uses ``mode='json'`` so str-Enum subclasses (``EventType``,
        ``ClientType``, etc.) serialise as their string values
        rather than as enum instances — matching the prior
        ``asdict`` + manual enum coercion behaviour.
        """
        return self.model_dump(mode='json')

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()


# =============================================================================
# Server -> Client Events
# =============================================================================

class ConnectedEvent(Event):
    """Sent when client connects successfully."""
    type: EventType = Field(default=EventType.CONNECTED)
    protocol_version: str = "1.0"
    server_info: Dict[str, Any] = Field(default_factory=dict)


class AgentCreatedEvent(Event):
    """Sent when a new agent (main or subagent) is created.

    Attributes:
        agent_id: Logical agent identifier (e.g. ``"main"`` or a
            subagent slot id).  This is the agent NAME slot, NOT the
            daemon's session_id.
        agent_name: Human-readable agent display name (typically the
            agent's persona name from ``.jaato/agents/<name>.md``).
        agent_type: ``"main"`` or ``"subagent"``.
        profile_name: Optional profile name resolved at spawn time.
        parent_agent_id: Optional logical id of the spawning agent
            (None for top-level / main agents).
        created_at: Optional ISO-8601 timestamp.
        session_id: Daemon-side session identifier (server 0.6.175+).
            Populated by every constructor site via the same parent-
            walk resolution used by ``RenderContext.session_id``
            (server 0.6.172+).  Subagent emit sites fall back to the
            parent's session_id when the immediate session has no
            ``_daemon_session_id`` of its own yet (subagent
            JaatoSession instances inherit the root agent's session
            via ``_parent_session``).  Empty string when no ancestor
            in the parent chain has a session_id set
            (e.g. ``main_agent_id`` emit during bootstrap before
            ``set_daemon_session_id`` fires).  Cascade observers
            use this for per-stage session_id correlation without
            having to maintain their own ``agent_id → session_id``
            map.
    """
    type: EventType = Field(default=EventType.AGENT_CREATED)
    agent_id: str = ""
    agent_name: str = ""
    agent_type: str = ""  # "main" or "subagent"
    profile_name: Optional[str] = None
    parent_agent_id: Optional[str] = None
    created_at: Optional[str] = None
    session_id: str = ""


class AgentOutputEvent(Event):
    """Streaming text output from an agent."""
    type: EventType = Field(default=EventType.AGENT_OUTPUT)
    agent_id: str = ""
    source: str = ""  # "model", "tool", "system", plugin name
    text: str = ""
    mode: str = "write"  # "write" (new block) or "append" (continue)


class AgentStatusChangedEvent(Event):
    """Agent status change (active, idle, done, error)."""
    type: EventType = Field(default=EventType.AGENT_STATUS_CHANGED)
    agent_id: str = ""
    status: str = ""  # "active", "idle" (waiting for input), "done", "error"
    error: Optional[str] = None


class AgentCompletedEvent(Event):
    """Agent has completed its task.

    The ``payload`` field carries the validated typed payload from
    ``signal_completion`` when the agent's profile declared a
    ``completion_payload_schema``. Reactor consumers should prefer
    ``payload`` (structured fields) over ``summary`` (free text). When
    the profile did not declare a schema, ``payload`` is ``None`` and
    consumers fall back to reading the legacy ``summary`` field on the
    associated tool result.
    """
    type: EventType = Field(default=EventType.AGENT_COMPLETED)
    agent_id: str = ""
    completed_at: str = ""
    success: bool = True
    token_usage: Optional[Dict[str, int]] = None
    turns_used: Optional[int] = None
    error: str = ""  # Cancellation reason or error message
    payload: Optional[Dict[str, Any]] = None  # Validated typed payload from signal_completion


class AgentErrorEvent(Event):
    """An agent hit a terminal error that the framework could not self-resolve.

    This is the **recovery contract**: it fires when the framework's automatic
    management (``with_retry`` for retryable provider errors, the completion
    nudge loop) is **exhausted** or never applied — i.e. the framework is out of
    moves. It gives a reactor *first refusal* to recover the failed stage
    (re-spawn, reroute to another model/provider, escalate) via the existing
    ``create_session`` path, BEFORE the session's terminal
    ``SessionTerminatedEvent(reason="error")`` lands.

    Emit order on the wire is **always** ``AgentErrorEvent`` first, then
    ``SessionTerminatedEvent(reason="error")``. A reactor that recovers should
    mark the ``session_id`` handled so the (back-compat) terminated handler
    no-ops; a cascade with no ``AGENT_ERROR`` handler ignores this event and the
    terminated event drives the legacy abort — fully back-compatible.

    Recovery is **decoupled from transience**: a non-transient error is still
    stage-recoverable (reroute/escalate). The framework offers the recovery
    *point*; the reactor's policy decides what to do.

    Fields:
        agent_id: The failed agent / cascade stage.
        session_id: The failed session (dedupe / handled-marking key).
        error_type: Exception class name (``"APIError"``, ``"RunnerCallError"``,
            ``"NudgeExhausted"``, ...). Same value carried on the subsequent
            ``SessionTerminatedEvent.error_type``.
        error_summary: Human-readable cause.
        request_id: Provider request id (e.g. OpenAI ``req_…``) when the
            underlying exception carries one; ``None`` otherwise. For
            observability / support correlation.
        attempt: The **reactor-level** re-spawn count for this logical stage,
            echoed verbatim from the spawn's ``agent_params["attempt"]`` (a
            string on the wire). This is NOT ``with_retry``'s internal
            per-request attempt count (which is never surfaced). ``"0"`` /
            absent on the first spawn. The reactor owns the cap.
        classification: Optional COARSE shape hint — ``"transient_provider"`` /
            ``"fatal_contract"`` / ``"unknown"``. **Advisory only**: it never
            gates whether this event fires. ``None`` when unclassified.
        framework_retries_exhausted: Optional informational count of automatic
            retries the framework already burned before giving up. ``None`` when
            not applicable.
        occurred_at: Emit timestamp (epoch seconds).
    """
    type: EventType = Field(default=EventType.AGENT_ERROR)
    agent_id: str = ""
    session_id: str = ""
    error_type: str = ""
    error_summary: str = ""
    request_id: Optional[str] = None
    attempt: str = "0"  # reactor-level re-spawn count, echoed from agent_params (wire is str)
    classification: Optional[str] = None  # coarse hint, non-gating
    framework_retries_exhausted: Optional[int] = None
    occurred_at: Optional[float] = None


class SessionTerminatedEvent(Event):
    """Session has fully wound down — safe to disconnect or
    ``delete_session``.

    Fires in two scenarios:

    1. **Natural completion**: emitted spontaneously after the
       agent's terminal completion (``AgentCompletedEvent``) AND
       the framework's post-completion wrap-up has drained
       (``_is_running`` returned False, plugin-on-end hooks ran,
       journal flushed).  Test harnesses can subscribe to this
       instead of the legacy "subscribe AGENT_COMPLETED + wait
       10s for TURN_COMPLETED" heuristic.

    2. **Client-requested**: emitted in response to ``session.end``
       after the daemon has stopped any in-flight activity and run
       cleanup.  Replaces the legacy
       ``SystemMessageEvent("[SESSION_TERMINATED]")`` string-based
       marker.

    The ``reason`` field distinguishes the two paths so consumers
    can handle them differently if needed.

    When ``reason="error"``, the framework populates
    ``error_summary`` + ``error_type`` from the underlying
    ``Exception`` at the emit site (server 0.6.159+ / SDK 0.14.1+).
    Cascade observers can read these to surface the failure cause
    without grepping the daemon log — e.g.
    ``error_type="AnthropicAPIError"`` +
    ``error_summary="402 Payment Required ..."``.  Both fields stay
    ``None`` for the non-error reasons (``natural`` /
    ``client_request`` / ``stopped``).

    When ``reason="budget_exhausted"``, the session hit a budget
    ceiling and REFUSES all further turns -- exhaustion means "this
    session is done", not "cancel this turn"
    (:meth:`JaatoSession._refuse_if_budget_exhausted`).  ``details``
    carries the refusal prose and the per-dimension usage.  Emitted
    because the refusal short-circuits before any turn runs, so no
    turn-completion notification fires and a wake-driven driver would
    otherwise wait out its full timeout and report a generic failure --
    a ceiling stop indistinguishable from a break.

    Canonical pattern (test harness):

        client.subscribe_once(EventType.SESSION_TERMINATED, on_done)
        sid = await client.create_session(...)
        await client.send_message(...)
        await on_done.wait()
        # Session has fully wound down.  Optionally delete_session(sid).
    """
    type: EventType = Field(default=EventType.SESSION_TERMINATED)
    session_id: str = ""
    agent_id: Optional[str] = None
    reason: str = "natural"  # "natural" | "client_request" | "stopped" | "error" | "cascade_cancelled" | "budget_exhausted"
    error_summary: Optional[str] = None  # populated when reason="error"
    error_type: Optional[str] = None     # Python exception class name (e.g. "AnthropicAPIError")
    # Machine-readable evidence for reasons whose cause is structured.
    # Populated for ``reason="budget_exhausted"`` with the refusal prose
    # and the per-dimension usage, so a driver can exit on the ceiling
    # without substring-matching the output stream.  ``None`` otherwise.
    details: Optional[Dict[str, Any]] = None


class SlotSettledEvent(Event):
    """A cascade stage's session has fully settled — its runner/slot has
    returned to the pool (warm) or been torn down (cold) — and the next stage
    is safe to spawn.

    Emitted by the daemon at the END of ``JaatoServer.shutdown`` for **every**
    cascade session (``cascade_driver_id`` set), on ALL teardown paths:
    pool-slot-returned, pool-slot-torn-down-on-error, and cold-spawned.  This
    universality is the point — a cascade reactor can gate the next stage's
    spawn on this single event with NO timeout and NO stall risk, because it
    fires exactly once per stage regardless of how the stage's runner ended.

    ``was_warm`` reports whether a warm pre-warm-pool slot was returned (so the
    next stage's spawn will reuse it, ≈30s→7s bootstrap) vs. a cold/torn-down
    teardown (next stage cold-spawns).  It is observability for the reactor —
    the spawn happens either way; ``was_warm`` just says whether it'll be fast.

    Replaces the earlier warm-only ``SlotReusableEvent`` (which did not fire for
    cold-spawned stages — common for the early cascade stages — and so could
    stall a pure-reactor handoff).  Correlate by ``cascade_driver_id``; route
    per-stage by ``agent_id``.  Distinct from :class:`SessionTerminatedEvent`,
    which fires EARLIER (before the slot returns) so spawning on it races the
    slot and cold-spawns.
    """
    type: EventType = Field(default=EventType.SLOT_SETTLED)
    session_id: str = ""                      # the session that just ended
    agent_id: Optional[str] = None            # stage's primary agent name
                                              # (e.g. "discovery", "codegen") —
                                              # route per-stage via
                                              # where: agent_id == <stage>
    cascade_driver_id: Optional[str] = None   # cascade affinity (always set here)
    was_warm: bool = False                    # True = warm slot returned (next
                                              # spawn reuses it); False = cold/
                                              # torn-down (next spawn is cold)
    pool_slot_pid: int = 0                     # the warm slot's PID (0 if cold)
    terminal_reason: Optional[str] = None     # how the settled session ended:
                                              # mirrors SessionTerminatedEvent.reason
                                              # ("error"/"stopped"/"cascade_cancelled"),
                                              # or None for natural completion.  Lets
                                              # a stage-advance reactor SKIP advancement
                                              # on an error-terminated session (the
                                              # recovery path re-spawns it instead) —
                                              # race-free, straight from the event.
                                              # See docs/design/agent-error-recovery-event.md.


class SessionRestoredEvent(Event):
    """Session was loaded from disk and the first client just attached.

    Phase 3 §3.12 disk-restore + peer-review M5/N1: when a session
    is restored from disk (daemon restart / cold attach), the
    daemon may have held in-flight tool calls during the
    no-client window using the defer-and-flush posture (vs
    denying outright per the pre-§3.12 behaviour).  This event
    fires on the first client-attach so the client can surface a
    "this session was restored — N pending tool calls to review"
    prompt; the operator drains the queue (each held ASK relays
    through the now-attached ``client.prompt_operator`` channel
    as if it had just landed) and the
    ``Session.restored_pending_attach`` flag clears.

    ``pending_tool_call_count`` is 0 for clean restores with no
    in-flight work; the event still fires in that case so clients
    can distinguish a fresh-attach from a restored-attach for
    telemetry / UX purposes.
    """

    type: EventType = Field(default=EventType.SESSION_RESTORED)
    session_id: str = ""
    pending_tool_call_count: int = 0


class ToolCallStartEvent(Event):
    """Tool execution has started."""
    type: EventType = Field(default=EventType.TOOL_CALL_START)
    agent_id: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    call_id: Optional[str] = None


class ToolCallEndEvent(Event):
    """Tool execution has completed."""
    type: EventType = Field(default=EventType.TOOL_CALL_END)
    agent_id: str = ""
    tool_name: str = ""
    call_id: Optional[str] = None
    success: bool = True
    is_error_result: bool = False  # computed deeper error check — success=True but error body; distinct from `success`
    # The tool result's own `status` string, copied verbatim when it declares
    # one (`send_to_sibling` → accepted/queued/refused/sibling_cold/…).  None
    # when the tool says nothing — which is NOT an outcome, just silence.
    # `success`/`is_error_result` answer "did it fail"; this answers "how",
    # without the consumer having to match on `error_message` prose.
    result_status: Optional[str] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    backgrounded: bool = False  # True when tool was auto-backgrounded (still producing output)
    continuation_id: Optional[str] = None  # Session ID for continuation grouping (e.g., interactive shell)
    show_output: Optional[bool] = None  # Whether to render output_lines in the main panel (None = default True)
    show_popup: Optional[bool] = None  # Whether to track/update the tool output popup (None = default True)


#: Reserved ``ToolOutputEvent.call_id`` for media the MODEL produced, as
#: opposed to media a tool returned under its own call id.  Defined here,
#: on the CLIENT side of the wire, because it is the clients that must
#: read it: the daemon writes one value, every consumer compares against
#: it, and a literal copied into each consumer is a shared constant with
#: no single owner.
MODEL_MEDIA_CALL_ID = "model-output"


class ToolOutputEvent(Event):
    """Live output chunk from a running tool (tail -f style).

    Carries text, binary media, or both.  This event is widened rather
    than joined by a rival media event because it already correlates by
    ``call_id``, is already mapped onto the in-process bus, and clients
    already subscribe to it -- so widening the payload lights up all
    three subscription surfaces (SDK client, ``subscribeToEvents`` agent
    tool, ``EventBus``) at once, with no new API on any of them.

    A whole-blob delivery -- a tool returning one finished WAV -- is just
    a single-chunk stream: ``sequence=0, final=True``.

    Attributes:
        agent_id: Which agent produced the chunk.
        call_id: Correlates the chunk with a specific tool call.
        chunk: Output text (may contain newlines).  Empty for a
            pure-media chunk.
        stream_id: Correlates chunks belonging to one media stream.
            Empty for unstreamed text, preserving existing frames.
        sequence: Ordering, passed through from
            :attr:`StreamChunk.sequence` rather than re-counted here --
            a second counter would be a second source of truth.
        mime_type: Tags the ``data_b64`` payload (e.g. ``"audio/wav"``).
        data_b64: Base64-encoded binary payload (+33% over the raw
            bytes; the frame is UTF-8 JSON).
        final: Last chunk of this stream, so a client can close its
            playback buffer or finish writing the file without waiting
            on a separate completion event.

    Note:
        When ``mime_type``/``data_b64`` are set the chunk MUST bypass the
        text formatter pipeline -- see ``server/core.py`` ``on_tool_output``.
        A formatter that reflows text corrupts bytes.
    """
    type: EventType = Field(default=EventType.TOOL_OUTPUT)
    agent_id: str = ""
    call_id: str = ""  # Required to correlate with specific tool call
    chunk: str = ""  # Output text chunk (may contain newlines)
    stream_id: str = ""  # Correlates chunks of one media stream
    sequence: Optional[int] = None  # From StreamChunk.sequence
    mime_type: Optional[str] = None  # Tags the data_b64 payload
    data_b64: Optional[str] = None  # Base64 binary payload
    final: bool = False  # Last chunk of this stream

    def is_media(self) -> bool:
        """Whether this event carries a binary payload.

        The predicate routing code should use, so "has bytes" is defined
        once rather than re-derived at each call site.
        """
        return bool(self.mime_type and self.data_b64)

    def is_model_speech(self) -> bool:
        """Whether these bytes are the MODEL's own output, not a tool's.

        Both travel on this event; :data:`MODEL_MEDIA_CALL_ID` is what
        separates them.  Offered here because every client needs the
        distinction — audio the model produced is played, a tool's
        attachment is saved or shown — and without it each one
        rediscovers the literal ``"model-output"``.
        """
        return self.is_media() and self.call_id == MODEL_MEDIA_CALL_ID


class PermissionResponseOption(BaseModel):
    """A valid response option for permission prompts."""
    key: str  # Single char like "y", "n", "a"
    label: str  # Display label like "yes", "no", "always"
    action: str  # Action type: "allow", "deny", "whitelist", "blacklist"
    description: Optional[str] = None


class PermissionRequestedEvent(Event):
    """Permission is requested for a tool execution.

    Includes pre-formatted prompt lines (with diff for file edits) when available.
    """
    type: EventType = Field(default=EventType.PERMISSION_REQUESTED)
    agent_id: str = ""  # Which agent is requesting permission
    request_id: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    response_options: List[Dict[str, str]] = Field(default_factory=list)
    # ^ List of {key, label, action, description?}
    prompt_lines: Optional[List[str]] = None  # Pre-formatted prompt (with diff)
    format_hint: Optional[str] = None  # "diff" for colored diff display
    warnings: Optional[str] = None  # Security/analysis warnings to display separately
    warning_level: Optional[str] = None  # "info", "warning", "error"


class PermissionInputModeEvent(Event):
    """Signal client to enter permission input mode.

    Sent AFTER permission content has been emitted via AgentOutputEvent.
    This lightweight control event separates content delivery from input control.
    """
    type: EventType = Field(default=EventType.PERMISSION_INPUT_MODE)
    agent_id: str = ""  # Which agent is requesting permission
    request_id: str = ""
    tool_name: str = ""
    call_id: Optional[str] = None  # Unique ID for matching tool call (parallel execution)
    response_options: List[Dict[str, str]] = Field(default_factory=list)
    # ^ List of {key, label, action, description?}
    # Tool arguments for client-side editing (when edit option is available)
    tool_args: Optional[Dict[str, Any]] = None
    # Editable content metadata: {parameters: [...], format: "yaml"|"json"|"text"}
    editable_metadata: Optional[Dict[str, Any]] = None


class PermissionResolvedEvent(Event):
    """Permission has been resolved (granted or denied)."""
    type: EventType = Field(default=EventType.PERMISSION_RESOLVED)
    agent_id: str = ""  # Which agent's permission was resolved
    request_id: str = ""
    tool_name: str = ""
    granted: bool = False
    method: str = ""  # "user", "whitelist", "blacklist", "default"
    comment: str = ""  # Advisory comment (from yc: or ALLOW_WITH_COMMENT evaluator)


class PermissionStatusEvent(Event):
    """Permission status update for client toolbar display.

    Emitted after permission commands (default/suspend/resume) and
    permission resolutions that change the effective policy.
    """
    type: EventType = Field(default=EventType.PERMISSION_STATUS)
    effective_default: str = "ask"  # "allow", "deny", or "ask"
    suspension_scope: Optional[str] = None  # "turn", "idle", "session", or None


class ClarificationRequestedEvent(Event):
    """Clarification session has started."""
    type: EventType = Field(default=EventType.CLARIFICATION_REQUESTED)
    agent_id: str = ""  # Which agent is requesting clarification
    request_id: str = ""
    tool_name: str = ""
    context_lines: List[str] = Field(default_factory=list)
    total_questions: int = 0


class ClarificationQuestionEvent(Event):
    """A single clarification question to answer."""
    type: EventType = Field(default=EventType.CLARIFICATION_QUESTION)
    agent_id: str = ""  # Which agent is asking the question
    request_id: str = ""
    question_index: int = 0
    total_questions: int = 0
    question_type: str = ""  # "single_choice", "multiple_choice", "free_text"
    question_text: str = ""
    options: Optional[List[Dict[str, str]]] = None  # For choice questions


class ClarificationInputModeEvent(Event):
    """Signal client to enter clarification input mode.

    Sent AFTER clarification content has been emitted via AgentOutputEvent.
    This lightweight control event separates content delivery from input control.
    """
    type: EventType = Field(default=EventType.CLARIFICATION_INPUT_MODE)
    agent_id: str = ""  # Which agent is requesting clarification
    request_id: str = ""
    tool_name: str = ""
    question_index: int = 0
    total_questions: int = 0


class ClarificationResolvedEvent(Event):
    """All clarification questions have been answered."""
    type: EventType = Field(default=EventType.CLARIFICATION_RESOLVED)
    agent_id: str = ""  # Which agent's clarification was resolved
    request_id: str = ""
    tool_name: str = ""
    qa_pairs: List[List[str]] = Field(default_factory=list)
    # ^ List of [question_text, answer_text] pairs for overview display


class ClarificationBatchEvent(Event):
    """All clarification questions sent at once for batch answering.

    Emitted on two distinct paths, told apart by ``batch_only``:

    * **Daemon-local sessions** (``batch_only=False``) — emitted before the
      QueueChannel loop so a client that can render every question at once
      (a tabbed panel, say) does not have to wait for them to trickle in.
      The per-question flow still follows: an ``AgentOutputEvent`` carrying
      the question text plus a ``ClarificationInputModeEvent`` for each
      question in turn.  A client that prefers the per-question flow may
      ignore this event entirely.
    * **Runner-tier sessions** (``batch_only=True``) — emitted by
      ``server.runner_rpc_handlers.clarification_relay``, which relays the
      whole batch from the runner and awaits the whole answer set.  Nothing
      else follows: no ``AgentOutputEvent``, no
      ``ClarificationInputModeEvent``, no ``ClarificationResolvedEvent``.
      A client that ignores this event leaves the tool call — and with it
      the turn — blocked forever (#704), so handling it is mandatory.

    Either way the reply is a single :class:`ClarificationBatchResponseEvent`.
    """
    type: EventType = Field(default=EventType.CLARIFICATION_BATCH)
    agent_id: str = ""
    request_id: str = ""
    tool_name: str = ""
    context: str = ""
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of {index, text, question_type, required, choices: [{text, default?}]}
    batch_only: bool = False
    # ^ True when this event is the ONLY delivery of the questions and the
    #   only way to answer them.  False means the per-question
    #   ClarificationInputModeEvent flow follows and a client may use either.


class ClarificationBatchResponseEvent(Event):
    """Client responds with all answers at once (batch mode).

    ``cancelled=True`` abandons the clarification instead of answering it:
    the tool returns ``{"cancelled": True}`` to the model and the turn
    continues.  It is the only way out of a ``batch_only`` clarification
    the user cannot or will not answer — without it, an unanswerable
    question blocks the turn indefinitely.  ``answers`` is ignored when
    ``cancelled`` is set.
    """
    type: EventType = Field(default=EventType.CLARIFICATION_BATCH_RESPONSE)
    request_id: str = ""
    answers: List[str] = Field(default_factory=list)
    # ^ Ordered list of answers, one per question (by index)
    cancelled: bool = False
    # ^ True to cancel the clarification outright (answers ignored).


class ReferenceSelectionRequestedEvent(Event):
    """Reference selection has been requested.

    Sent when the model calls selectReferences and the user needs to choose
    which references to include.
    """
    type: EventType = Field(default=EventType.REFERENCE_SELECTION_REQUESTED)
    agent_id: str = ""  # Which agent is requesting reference selection
    request_id: str = ""
    tool_name: str = ""
    prompt_lines: List[str] = Field(default_factory=list)


class ReferenceSelectionResolvedEvent(Event):
    """Reference selection has been completed."""
    type: EventType = Field(default=EventType.REFERENCE_SELECTION_RESOLVED)
    agent_id: str = ""  # Which agent's reference selection was resolved
    request_id: str = ""
    tool_name: str = ""
    selected_ids: List[str] = Field(default_factory=list)


class WorkspaceMismatchResponseOption(BaseModel):
    """A valid response option for workspace mismatch prompts."""
    key: str  # Single char like "s", "n"
    label: str  # Display label like "switch", "new session"
    action: str  # Action type: "switch", "new_session", "cancel"
    description: Optional[str] = None


class WorkspaceMismatchRequestedEvent(Event):
    """Workspace mismatch detected when attaching to a session.

    Sent when a client tries to attach to a session that was created
    with a different workspace path. The client must choose to either
    switch to the session's workspace or create a new session.
    """
    type: EventType = Field(default=EventType.WORKSPACE_MISMATCH_REQUESTED)
    request_id: str = ""
    session_id: str = ""
    session_workspace: str = ""  # The session's current workspace
    client_workspace: str = ""   # The client's workspace
    response_options: List[Dict[str, str]] = Field(default_factory=list)
    # ^ List of {key, label, action, description?}
    prompt_lines: List[str] = Field(default_factory=list)


class WorkspaceMismatchResolvedEvent(Event):
    """Workspace mismatch has been resolved."""
    type: EventType = Field(default=EventType.WORKSPACE_MISMATCH_RESOLVED)
    request_id: str = ""
    session_id: str = ""
    action: str = ""  # "switch", "new_session", "cancel"
    new_session_id: Optional[str] = None  # Set if action is "new_session"


class PostAuthSetupEvent(Event):
    """Offer session setup after successful authentication.

    Emitted by daemon after an auth command succeeds. The client renders a
    multi-step wizard and sends back a single PostAuthSetupResponse.
    """
    type: EventType = Field(default=EventType.POST_AUTH_SETUP)
    request_id: str = ""
    provider_name: str = ""        # e.g., "zhipuai"
    provider_display_name: str = ""  # e.g., "Zhipu AI (Z.AI)"
    available_models: List[Dict[str, str]] = Field(default_factory=list)
    # ^ [{name: "zhipuai/glm-4.7", description: "..."}, ...]
    has_active_session: bool = False
    current_provider: str = ""     # Only set if has_active_session
    current_model: str = ""        # Only set if has_active_session
    workspace_path: str = ""


class PostAuthSetupResponse(Event):
    """User's response to post-auth session setup prompt."""
    type: EventType = Field(default=EventType.POST_AUTH_SETUP_RESPONSE)
    request_id: str = ""
    connect: bool = False          # Whether to create/switch session
    model_name: str = ""           # Selected model (if connect=True)
    persist_env: bool = False      # Whether to save provider/model to .env


class PlanStepData(BaseModel):
    """A single step in a plan."""
    content: str
    status: str  # "pending", "in_progress", "completed"
    active_form: Optional[str] = None


class PlanUpdatedEvent(Event):
    """Plan has been created or updated."""
    type: EventType = Field(default=EventType.PLAN_UPDATED)
    agent_id: str = ""
    plan_name: str = ""
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of {content, status, active_form?, blocked_by?, depends_on?, received_outputs?}


class PlanStepUpdatedEvent(Event):
    """Single step status change within a plan.

    Lean delta event — carries only the changed step's data, not the
    full plan snapshot. The client maintains local plan state and applies
    this delta to update the specific step.

    Sent for status-only changes (started, completed, failed, skipped,
    blocked, unblocked). Structural changes (plan created, steps added,
    plan completed) use ``PlanUpdatedEvent`` with the full snapshot.
    """
    type: EventType = Field(default=EventType.PLAN_STEP_UPDATED)
    agent_id: str = ""
    step_id: str = ""
    sequence: int = 0
    content: str = ""  # Step description
    status: str = ""   # "pending", "in_progress", "completed", "failed", "skipped", "blocked"
    result: Optional[str] = None
    error: Optional[str] = None
    blocked_by: Optional[List[Dict[str, Any]]] = None
    depends_on: Optional[List[Dict[str, Any]]] = None
    received_outputs: Optional[Dict[str, Any]] = None


class PlanClearedEvent(Event):
    """Plan has been cleared/completed."""
    type: EventType = Field(default=EventType.PLAN_CLEARED)
    agent_id: str = ""


class UsageBreakdown(BaseModel):
    """Provider-agnostic per-turn usage shape carried by Context/Turn events.

    Single source of truth for token + cost reporting on the wire.  All
    optional fields default to ``None`` so a provider that doesn't
    report a given dimension simply omits it instead of zero-padding
    (which would make a ``0`` cache hit indistinguishable from "no
    caching support").

    ``cost_usd`` is populated by the daemon when either:
    - the provider reports a real cost (e.g. ``claude_cli`` exposes
      ``total_cost_usd`` from the underlying CLI), or
    - the operator has loaded a pricing table at
      ``.jaato/pricing.json`` (Litellm-compatible) and the model name
      is found there.

    When neither source has a number, ``cost_usd`` is ``None`` —
    consumers must not assume zero means free.
    """
    model_config = ConfigDict(extra='ignore')

    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Prompt-cache token counts (None when provider does not support caching)
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    # Reasoning tokens (OpenAI o-series) — billed as output
    reasoning_tokens: Optional[int] = None
    # Thinking tokens (Anthropic / Gemini extended thinking)
    # Subset of output_tokens; useful for UI breakdowns
    thinking_tokens: Optional[int] = None
    # Cost in USD; None when neither provider nor pricing table knows
    cost_usd: Optional[float] = None
    # Tokens BILLED across the turn — the sum over the turn's responses.
    # Distinct from ``total_tokens``, which is the LAST response's total and
    # therefore the end-of-turn CONTEXT SIZE for a prompt-inclusive provider.
    # A turn with a tool call has >=2 billed responses, so summing
    # ``total_tokens`` across turns undercounts spend (measured: 59% of
    # actual).  ``None`` on per-response usage, where spend == total.
    spend_total_tokens: Optional[int] = None
    # The same split, BILLED across the turn.  ``prompt_tokens`` and
    # ``output_tokens`` above are the LAST response's figures, so a consumer
    # summing them across turns hits the identical undercount measured for
    # ``total_tokens``: a turn with a tool call has >=2 billed responses and
    # only the last one is visible there.  The session has accumulated both
    # per response all along (``spend_prompt`` / ``spend_output``); until
    # jaato #802 neither reached the wire, so ``jaato-eval`` summed the
    # level pair for want of anything else to record.  ``None`` on
    # per-response usage, where spend == the response's own figure.
    spend_prompt_tokens: Optional[int] = None
    spend_output_tokens: Optional[int] = None
    # Cache traffic BILLED across the turn — summed over the turn's
    # responses, the same shape as ``spend_total_tokens``.  Distinct from
    # ``cache_read_tokens`` / ``cache_creation_tokens``, which are the LAST
    # response's figures.  The distinction is load-bearing for a session
    # using ``model_tiers``: a mid-turn tier switch re-reads the whole
    # prefix cold at the new model, and the last-response figures hide
    # exactly that miss.  ``None`` when the provider reports no cache usage.
    spend_cache_read_tokens: Optional[int] = None
    spend_cache_creation_tokens: Optional[int] = None


class GCConfigEvent(Event):
    """GC configuration snapshot for the active session.

    Emitted on session init and whenever the GC plugin is reconfigured.
    Carries only configuration — actual usage lives in
    ``ContextUpdatedEvent``. Splitting these two concerns avoids the
    pre-1.0 hack where ``ContextUpdatedEvent`` doubled as a status-bar
    config carrier.
    """
    type: EventType = Field(default=EventType.GC_CONFIG)
    agent_id: str = ""
    threshold: Optional[float] = None       # GC trigger threshold percentage
    strategy: Optional[str] = None          # "truncate" | "summarize" | "hybrid" | "budget"
    target_percent: Optional[float] = None  # Target usage after GC
    continuous_mode: bool = False           # True if GC runs after every turn


class GCEvent(Event):
    """Garbage collection lifecycle — one event, switched on ``phase``.

    DISTINCT from :class:`GCConfigEvent`, which carries configuration only
    (threshold / strategy / target) at init and reconfigure.  This is the
    lifecycle: GC is about to run, is running, has finished.

    Before this existed there was NO lifecycle signal on the bus.  The
    framework opened an OpenTelemetry span with the trigger reason and the
    strategy, and clients got either prose -- a ``SystemMessageEvent`` reading
    "Context usage (84.2%) exceeds threshold (80%). GC will run after this
    turn." -- or nothing at all.  A client wanting to show "compacting..." had
    to substring-match that sentence for the start and guess at the end, which
    is the parse-the-log shape typed events exist to replace.

    Phases:

    ``about_to_run``
        The threshold was crossed; GC will run after this turn.  Carries
        ``percent_used`` / ``threshold``.  Announces a FUTURE pass -- the
        session keeps serving the current turn.
    ``started``
        A pass is beginning now.  Carries ``trigger_reason`` / ``strategy``
        plus the "before" figures.
    ``completed``
        The pass finished.  Carries ``success``, ``items_collected``,
        ``tokens_freed``, ``tokens_before`` / ``tokens_after``, and ``error``
        when it failed.

    "Ongoing" is the interval BETWEEN ``started`` and ``completed`` -- a
    client renders its spinner there.  ``collect()`` is atomic, so there is no
    sub-pass progress to report and none is invented.

    Every GC pass emits ``started`` + ``completed``, including a failed one:
    the failure is the case an operator most needs.  ``about_to_run`` fires
    only for threshold-triggered passes, since the other triggers (manual,
    context-limit recovery) have no advance warning by nature.
    """
    type: EventType = Field(default=EventType.GC)
    agent_id: str = ""
    phase: str = ""                          # about_to_run | started | completed
    trigger_reason: Optional[str] = None     # threshold | manual | context_limit | ...
    strategy: Optional[str] = None           # GC plugin name
    # Context framing (about_to_run / started)
    percent_used: Optional[float] = None
    threshold: Optional[float] = None
    context_limit: Optional[int] = None
    # Outcome (completed)
    success: Optional[bool] = None
    items_collected: Optional[int] = None
    tokens_before: Optional[int] = None
    tokens_after: Optional[int] = None
    tokens_freed: Optional[int] = None
    error: Optional[str] = None


class ContextUpdatedEvent(Event):
    """Context window usage has changed.

    Carries a typed ``usage`` (``UsageBreakdown``) shared with
    ``TurnCompletedEvent`` and ``TurnProgressEvent`` so consumers can
    treat the three events uniformly. Context-window framing fields
    (``context_limit``, ``percent_used``, ``tokens_remaining``,
    ``turns``) stay on this event because they describe the *window*,
    not the most recent generation.

    GC configuration moved to ``GCConfigEvent`` in v1.0 — query that
    event (or read it from session init) for status-bar display.
    """
    type: EventType = Field(default=EventType.CONTEXT_UPDATED)
    agent_id: str = ""
    usage: UsageBreakdown = Field(default_factory=UsageBreakdown)
    context_limit: int = 0
    percent_used: float = 0.0
    tokens_remaining: int = 0
    turns: int = 0


class InstructionBudgetEvent(Event):
    """Instruction budget has been updated.

    Provides detailed breakdown of token usage by instruction source layer.
    Sent after session configuration and when budget changes significantly.

    The budget_snapshot contains:
    - session_id, agent_id, agent_type: Identity
    - context_limit, total_tokens, utilization_percent: Overall usage
    - gc_eligible_tokens, locked_tokens, preservable_tokens: GC info
    - entries: Per-source breakdown (system, session, plugin, enrichment, conversation)
    """
    type: EventType = Field(default=EventType.INSTRUCTION_BUDGET_UPDATED)
    agent_id: str = ""
    budget_snapshot: Dict[str, Any] = Field(default_factory=dict)


class TurnCompletedEvent(Event):
    """A conversation turn has completed.

    The ``usage`` field carries the provider-agnostic
    ``UsageBreakdown`` (token counts, cache hits, reasoning/thinking
    tokens, cost when known).  Treat it as the canonical per-turn
    usage record — it's the same shape ``TurnProgressEvent`` and
    ``ContextUpdatedEvent`` use.
    """
    type: EventType = Field(default=EventType.TURN_COMPLETED)
    agent_id: str = ""
    turn_number: int = 0
    usage: UsageBreakdown = Field(default_factory=UsageBreakdown)

    #: Why this turn produced no typed completion, when one was expected.
    #:
    #: ``None`` on every normal turn, including every turn of a session that
    #: never had a completion schema and every turn that legitimately has
    #: more work to do.
    #:
    #: ``"not_signalled_after_nudges"`` means the framework EXPECTED a
    #: completion here and gave up asking: ``signal_completion`` was in the
    #: session's tool surface, the model ended its loop without calling it,
    #: and the nudge budget (``MAX_COMPLETION_NUDGES``) is spent.
    #:
    #: This is the only signal a consumer gets in that state. No
    #: ``AgentCompletedEvent`` fires (only ``signal_completion`` produces
    #: one) and no ``SessionTerminatedEvent`` fires (quiescence is gated on
    #: ``signal_completion`` having been called), so a cascade driver
    #: otherwise sees a turn end and must INVENT a reason for the missing
    #: payload -- typically blaming the schema, which is the one thing that
    #: was fine.
    completion_gap: Optional[str] = None
    duration_seconds: float = 0.0
    function_calls: List[Dict[str, Any]] = Field(default_factory=list)
    # Formatted output text (with syntax highlighting, validation, etc.)
    # Client can use this to replace raw streaming output with formatted version
    formatted_text: Optional[str] = None
    # The provider's finish reason for the turn's terminal response, as the
    # lowercase ``FinishReason`` enum value: ``"stop"`` (normal completion),
    # ``"max_tokens"`` (output-token limit — response truncated), ``"safety"``
    # (safety filter), ``"error"`` (provider error), plus ``"tool_use"`` /
    # ``"cancelled"`` / ``"unknown"`` for completeness.  Defaults to ``"stop"``.
    #
    # DO NOT BRANCH ON ``finish_reason != "stop"`` TO DETECT TRUNCATION.
    # This comment used to recommend exactly that, and it is wrong for every
    # schema-driven profile.  A profile with a ``completion_payload_schema``
    # ends by calling ``signal_completion``, which terminates the session
    # INSIDE a tool-use turn — so a COMPLETE run's terminal turn reports
    # ``"tool_use"`` and no later turn ever says ``"stop"``.  A consumer
    # following the old advice blocked every schema-driven arm as truncated,
    # with the finished artefact sitting on disk beside the verdict.
    #
    # The correct rule needs the SESSION's outcome as well as the turn's, so
    # it does not fit in this field and is not this field's to state:
    # ``jaato_sdk.helpers.truncation_reason()`` — ``termination_reason``
    # outranks a payload, a payload outranks ``finish_reason``, and anything
    # else is named rather than guessed.
    #
    # This field remains the machine-readable companion to the human-readable
    # ``source="system"`` ``AgentOutputEvent`` banner emitted alongside
    # abnormal finishes; it is one input to the rule, not the rule.
    finish_reason: str = "stop"


class TurnProgressEvent(Event):
    """Incremental progress during turn execution.

    Emitted after each model response within a turn, enabling
    real-time token tracking before the turn completes.  The
    ``usage`` field is the same provider-agnostic shape used by
    ``TurnCompletedEvent`` and ``ContextUpdatedEvent``.
    """
    type: EventType = Field(default=EventType.TURN_PROGRESS)
    agent_id: str = ""
    usage: UsageBreakdown = Field(default_factory=UsageBreakdown)
    context_limit: int = 0
    percent_used: float = 0.0
    tokens_remaining: int = 0
    pending_tool_calls: int = 0  # How many tool calls remain


class SystemMessageEvent(Event):
    """System message (info, warning, status)."""
    type: EventType = Field(default=EventType.SYSTEM_MESSAGE)
    message: str = ""
    style: str = ""  # "info", "warning", "error", "success", "dim"


class HelpTextEvent(Event):
    """Detailed help text for commands.

    Sent in response to 'help' subcommands to display formatted help
    using the pager. Each line is a (text, style) tuple.
    """
    type: EventType = Field(default=EventType.HELP_TEXT)
    lines: List[tuple] = Field(default_factory=list)  # List of (text, style) tuples


class InitProgressEvent(Event):
    """Initialization progress update.

    Sent during session initialization to show progress on each step.
    Steps are shown in sequence with their status.
    """
    type: EventType = Field(default=EventType.INIT_PROGRESS)
    step: str = ""  # Step name (e.g., "Loading plugins")
    status: str = "running"  # "running", "done", "error"
    message: str = ""  # Optional details (e.g., error message)
    step_number: int = 0  # Current step (1-based)
    total_steps: int = 0  # Total number of steps


class ErrorEvent(Event):
    """Error occurred."""
    type: EventType = Field(default=EventType.ERROR)
    error: str = ""
    error_type: str = ""  # Exception class name
    recoverable: bool = True
    # Optional machine-readable evidence for errors whose cause is
    # structured.  ``error`` stays the human-readable sentence; this is what
    # a driver branches on.  Canonical case: a cascade budget refusal, which
    # carries the exhausted dimensions and BOTH inputs to the min() that
    # produced it — so a client can distinguish "out of budget" from
    # "daemon hung" without parsing prose or reading the server's log.
    details: Optional[Dict[str, Any]] = None
    # Echoes the originating request's correlation id when there was one, so a
    # failure is attributable to the call that caused it.  Without it, two
    # creates in flight and one refused meant the SUCCEEDING caller could
    # observe the other's failure.
    request_id: Optional[str] = None


class RetryEvent(Event):
    """API retry notification with exponential backoff.

    Sent when a transient error (rate limit, server error) is encountered
    and the system is retrying the request.
    """
    type: EventType = Field(default=EventType.RETRY)
    message: str = ""  # Human-readable retry message
    attempt: int = 0  # Current attempt number (1-indexed)
    max_attempts: int = 0  # Maximum attempts configured
    delay: float = 0.0  # Delay in seconds before next attempt
    error_type: str = ""  # Type of error (rate_limit, transient)


class SessionListEvent(Event):
    """List of available sessions - for user display."""
    type: EventType = Field(default=EventType.SESSION_LIST)
    sessions: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of {id: str, name: str, created_at: str, last_active: str, ...}


class MemoryListEvent(Event):
    """List of available memories - for completion cache and pager display."""
    type: EventType = Field(default=EventType.MEMORY_LIST)
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of {id: str, description: str, tags: List[str]}


class SandboxPathsEvent(Event):
    """List of sandbox-allowed paths - for @@ completion cache.

    Emitted after sandbox add/remove commands to refresh the client's
    completion list for @@ (sandbox path) references.
    """
    type: EventType = Field(default=EventType.SANDBOX_PATHS)
    paths: List[Dict[str, str]] = Field(default_factory=list)
    # ^ List of {path: str, description: str}


class ServiceListEvent(Event):
    """List of discovered services - for completion cache.

    Emitted after services commands to refresh the client's
    completion list for service names and HTTP methods.
    """
    type: EventType = Field(default=EventType.SERVICE_LIST)
    services: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of {name: str, methods: List[str]}


class SessionInfoEvent(Event):
    """Session state snapshot - sent on connect/attach with all data client needs.

    Includes current session info plus lists for completion/display:
    - sessions: All available sessions (for session commands)
    - tools: All available tools with enabled status (for tools commands)
    - models: Available model names (for model command)

    Client stores this locally and uses it for both completion and display.
    Server pushes updates when state changes.
    """
    type: EventType = Field(default=EventType.SESSION_INFO)
    # Echoes ``CommandRequest.payload["request_id"]`` when the client supplied
    # one, so a caller can tell WHICH create this describes.  Without it
    # ``_await_session_info`` matched on SHAPE -- any SessionInfoEvent with a
    # session_id -- and since the client's event buffer is drained into each
    # new subscription, a stale event from an EARLIER create satisfied a LATER
    # wait and handed the caller an id it had not created.
    request_id: Optional[str] = None
    # Current session
    session_id: str = ""
    session_name: str = ""
    model_provider: str = ""
    model_name: str = ""
    profile_name: Optional[str] = None  # Agent profile used to create this session
    # State snapshot for local use
    sessions: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ [{id, name, model_provider, model_name, is_loaded, client_count, turn_count}, ...]
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ [{name, description, enabled, plugin}, ...]
    models: List[str] = Field(default_factory=list)
    # ^ ["gemini-2.5-flash", "gemini-2.5-pro", ...]
    user_inputs: List[str] = Field(default_factory=list)
    # ^ Command history for prompt restoration on reconnect
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ [{id, description, tags}, ...] for memory command completions
    sandbox_paths: List[Dict[str, str]] = Field(default_factory=list)
    # ^ [{path, description}, ...] for @@ sandbox completion
    services: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ [{name, methods}, ...] for services command completions
    tool_id_mappings: Dict[str, str] = Field(default_factory=dict)
    # ^ {hash_id: human_name, ...} for resolving opaque tool/category IDs in display


class SessionDescriptionUpdatedEvent(Event):
    """Session description was updated (by model calling session_describe)."""
    type: EventType = Field(default=EventType.SESSION_DESCRIPTION_UPDATED)
    session_id: str = ""
    description: str = ""


class ProfileSummary(BaseModel):
    """Stable summary of a profile, safe to expose to external clients.

    Versioned by the global ``ConnectedEvent.protocol_version`` —
    breaking changes to this shape bump the protocol's MAJOR; additive
    optional fields bump the MINOR.  Sensitive material is intentionally
    omitted: env *values* are summarised by name only;
    ``system_instructions``, ``icon_name`` and ``inherits`` are not
    exposed (deprecated or already resolved during discovery).
    Structural config (``plugin_configs``, ``model_tiers``,
    ``runtime_limits``, ``gc``) is exposed as-is — profile authors are
    expected to use ``${VAR}`` indirection for secrets and put the
    actual values in ``env`` (which is summarised by key).
    """
    model_config = ConfigDict(extra='ignore')

    # Identity
    name: str
    description: str = ""

    # Capabilities
    plugins: List[str] = Field(default_factory=list)
    preloaded_plugins: List[str] = Field(default_factory=list)
    plugin_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Runtime
    model: Optional[str] = None
    provider: Optional[str] = None
    max_turns: int = 10
    model_tiers: Dict[str, Any] = Field(default_factory=dict)
    # Profile-declared budget_control, re-serialised (shared/budget_control.py).
    # None = unbudgeted.  Declared explicitly because pydantic SILENTLY DROPS
    # unknown kwargs — without the field the producer's value vanishes with no
    # error (same class as the #540 turn_accounting drop).
    budget_control: Optional[Dict[str, Any]] = None
    gc: Optional[Dict[str, Any]] = None
    runtime_limits: Optional[Dict[str, Any]] = None
    completion_payload_schema: Optional[Union[str, Dict[str, Any]]] = None

    # Env: variable NAMES the profile expects to find at runtime —
    # values never leave the daemon
    env_var_names: List[str] = Field(default_factory=list)


class ProfileParseError(BaseModel):
    """Profile file that failed to parse during discovery.

    Carried in ``SessionProfilesEvent.parse_errors`` (a separate field
    from ``profiles``) so a picker can surface broken files distinctly
    rather than treating them as unusable entries in the main list.
    """
    model_config = ConfigDict(extra='ignore')

    name: str   # filename stem of the broken file
    error: str  # human-readable parse-error message


class SessionProfilesEvent(Event):
    """List of available agent profiles for session creation.

    Sent in response to a ``session.profiles`` command.  Each profile
    is a typed ``ProfileSummary`` carrying enough metadata for a
    profile picker UI without leaking secrets.

    Profiles that failed to parse during discovery are reported in
    ``parse_errors`` rather than mixed into ``profiles`` — a picker
    can surface them separately or hide them entirely.

    The shape of this event (and of nested ``ProfileSummary``) is
    versioned by the global ``ConnectedEvent.protocol_version``.  Pre-
    1.0 versions of the SDK had a per-event ``schema_version`` field
    here; that was promoted to the global protocol version in v1.0 and
    removed from this event.
    """
    type: EventType = Field(default=EventType.SESSION_PROFILES)
    profiles: List[ProfileSummary] = Field(default_factory=list)
    parse_errors: List[ProfileParseError] = Field(default_factory=list)


# =============================================================================
# Workspace Management Events (Server -> Client)
# =============================================================================

class WorkspaceInfo(BaseModel):
    """Information about a single workspace."""
    name: str  # Relative path from workspace root (e.g., "project-a")
    configured: bool  # Has valid .env with provider
    provider: Optional[str] = None  # Provider if configured
    model: Optional[str] = None  # Model if configured
    last_accessed: Optional[str] = None  # ISO timestamp


class WorkspaceListEvent(Event):
    """Response to workspace.list - list of available workspaces."""
    type: EventType = Field(default=EventType.WORKSPACE_LIST)
    root: str = ""  # Absolute path to workspace root
    workspaces: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of WorkspaceInfo as dicts


class WorkspaceCreatedEvent(Event):
    """Response to workspace.create - new workspace created."""
    type: EventType = Field(default=EventType.WORKSPACE_CREATED)
    name: str = ""  # Relative path from workspace root
    path: str = ""  # Absolute path


class ConfigStatusEvent(Event):
    """Response to workspace.select - configuration status of selected workspace."""
    type: EventType = Field(default=EventType.CONFIG_STATUS)
    workspace: str = ""  # Workspace name (relative path)
    configured: bool = False  # Has valid provider config
    provider: Optional[str] = None  # Current provider if set
    model: Optional[str] = None  # Current model if set
    available_providers: List[str] = Field(default_factory=list)  # Providers that can be configured
    missing_fields: List[str] = Field(default_factory=list)  # What's needed to complete config


class ConfigUpdatedEvent(Event):
    """Response to config.update - configuration was updated."""
    type: EventType = Field(default=EventType.CONFIG_UPDATED)
    workspace: str = ""  # Workspace name
    provider: str = ""  # New provider
    model: Optional[str] = None  # New model if set
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Workspace File Monitoring Events (Server -> Client)
# =============================================================================

class WorkspaceFilesChangedEvent(Event):
    """Incremental workspace file change notification.

    Emitted in real-time (debounced) whenever files in the workspace are
    created, modified, or deleted during the session.  Each entry carries
    a ``status`` indicating the nature of the change relative to the
    session baseline.

    Statuses:
        ``"created"``  – file did not exist at session start.
        ``"modified"`` – file existed at session start and was changed.
        ``"deleted"``  – file was previously tracked and is now gone.
    """
    type: EventType = Field(default=EventType.WORKSPACE_FILES_CHANGED)
    changes: List[Dict[str, str]] = Field(default_factory=list)
    # ^ List of {"path": str, "status": "created"|"modified"|"deleted"}


class WorkspaceFilesSnapshotEvent(Event):
    """Complete workspace file state snapshot.

    Sent on client reconnect / initial attach so the client can rebuild
    its local mirror of the session's file tracking state without
    replaying individual deltas.
    """
    type: EventType = Field(default=EventType.WORKSPACE_FILES_SNAPSHOT)
    files: List[Dict[str, str]] = Field(default_factory=list)
    # ^ List of {"path": str, "status": "created"|"modified"|"deleted"}
    total: int = 0
    # ^ Convenience: count of non-deleted entries


# =============================================================================
# Client -> Server Events (Requests)
# =============================================================================

class SendMessageRequest(Event):
    """Send a message to the model."""
    type: EventType = Field(default=EventType.SEND_MESSAGE)
    text: str = ""
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of {type: "file", path: "..."} or {type: "image", data: "base64..."}
    parallel_tools: Optional[bool] = None
    # ^ Per-call override of the JAATO_PARALLEL_TOOLS env default.
    #   ``None`` (default) keeps the env-configured behaviour.
    #   ``True`` / ``False`` forces parallel / sequential tool
    #   execution for this turn only.


class PermissionResponseRequest(Event):
    """Respond to a permission request."""
    type: EventType = Field(default=EventType.PERMISSION_RESPONSE)
    request_id: str = ""
    response: str = ""  # "y", "n", "a", "never", etc.
    # Edited tool arguments (set when response is "e" and client handled editing)
    edited_arguments: Optional[Dict[str, Any]] = None


class ClarificationResponseRequest(Event):
    """Respond to a clarification question."""
    type: EventType = Field(default=EventType.CLARIFICATION_RESPONSE)
    request_id: str = ""
    question_index: int = 0
    response: str = ""  # User's answer


class ReferenceSelectionResponseRequest(Event):
    """Respond to a reference selection request."""
    type: EventType = Field(default=EventType.REFERENCE_SELECTION_RESPONSE)
    request_id: str = ""
    response: str = ""  # User's selection (e.g., "1,3,4", "all", "none")


class WorkspaceMismatchResponseRequest(Event):
    """Respond to a workspace mismatch request."""
    type: EventType = Field(default=EventType.WORKSPACE_MISMATCH_RESPONSE)
    request_id: str = ""
    response: str = ""  # "s" (switch), "n" (new session), "c" (cancel)


class StopRequest(Event):
    """Stop current operation (cancel generation)."""
    type: EventType = Field(default=EventType.STOP)
    agent_id: Optional[str] = None  # None = all agents


class ExternalEventRequest(Event):
    """External event injected by the host page via the web component.

    Published on the session's ``EventBus`` as an ``external_event``
    so that agents subscribed via ``subscribeToEvents`` are notified.
    """
    type: EventType = Field(default=EventType.EVENT_EXTERNAL)
    name: str = ""        # Event name (e.g., "order.placed")
    data: Dict[str, Any] = Field(default_factory=dict)  # Arbitrary payload
    timestamp: str = ""   # ISO 8601 timestamp from the client


class EventsSubscribedEvent(Event):
    """Notification that an agent has subscribed to external events.

    Sent to WS clients so the host page knows which external event
    names the agent is listening for.  ``["*"]`` means all external
    events.
    """
    type: EventType = Field(default=EventType.EVENTS_SUBSCRIBED)
    agent_id: str = ""
    event_names: List[str] = Field(default_factory=list)


class CommandRequest(Event):
    """Execute a command (like 'model', 'save', 'resume', etc.).

    ``args`` carries CLI-style positional/flag arguments — the same
    array a TUI user would type after the command name.

    ``payload`` is an opt-in escape hatch for SDK consumers that need
    to pass structured data argv can't ergonomically carry (e.g.
    ``session.new`` with an inline profile spec dict).  Commands that
    accept ``payload`` document their expected keys in the relevant
    server handler; the wire stays generic.

    The TUI never produces ``payload`` (its input is always argv-shaped),
    so commands that *only* read from ``payload`` are SDK-only by
    construction.
    """
    type: EventType = Field(default=EventType.COMMAND)
    command: str = ""
    args: List[str] = Field(default_factory=list)
    payload: Optional[Dict[str, Any]] = None


class GetInstructionBudgetRequest(Event):
    """Request current instruction budget for an agent.

    Server responds with InstructionBudgetEvent containing the budget snapshot.
    If agent_id is None or empty, returns budget for main agent.
    """
    type: EventType = Field(default=EventType.INSTRUCTION_BUDGET_REQUEST)
    agent_id: Optional[str] = None  # None = main agent


class CommandListRequest(Event):
    """Request list of available commands from server."""
    type: EventType = Field(default=EventType.COMMAND_LIST_REQUEST)


class CommandListEvent(Event):
    """List of available commands from server/plugins."""
    type: EventType = Field(default=EventType.COMMAND_LIST)
    commands: List[Dict[str, str]] = Field(default_factory=list)
    # ^ List of {name, description, ?subcommands}


class CommandListRefreshEvent(Event):
    """Signal that the command list should be refreshed.

    Emitted by core.py after commands that change completion state
    (e.g., references select/unselect). The IPC client handles this
    by re-requesting the full command list from the daemon.
    """
    type: EventType = Field(default=EventType.COMMAND_LIST_REFRESH)


class ToolStatusEvent(Event):
    """Tool status information for client display."""
    type: EventType = Field(default=EventType.TOOL_STATUS)
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of {name, description, enabled, plugin}
    message: str = ""  # Optional result message (for enable/disable operations)


class ToolIdRegistryEvent(Event):
    """Hash-derived ID → human-readable name mapping for client display.

    Sent after tool configuration and when deferred tools are activated.
    Clients use this to resolve opaque tool/category IDs in tool arguments
    and model output without pattern-matching or reverse engineering.

    The mapping is cumulative — each event carries the full current set,
    not a delta. Clients should replace their local lookup on each receive.
    """
    type: EventType = Field(default=EventType.TOOL_ID_REGISTRY)
    mappings: Dict[str, str] = Field(default_factory=dict)


class ToolDisableRequest(Event):
    """Client request to disable a tool.

    Directly calls registry.disable_tool() without generating response events.
    Used by headless mode to disable tools before starting event handling.
    """
    type: EventType = Field(default=EventType.TOOL_DISABLE_REQUEST)
    tool_name: str = ""  # Tool to disable


class ToolsRegisterClientRequest(Event):
    """Register client-side tools that the browser/frontend can execute.

    The server creates proxy tools in the session's registry. When the model
    calls one, the server routes execution to the WS client via
    ``tool.execute_request`` and waits for ``tool.execute_result``.
    """
    type: EventType = Field(default=EventType.TOOLS_REGISTER_CLIENT)
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of {name, description, parameters: {type:'object', properties, required}, timeout}
    categories: Dict[str, str] = Field(default_factory=dict)
    # ^ Optional mapping of category name → description for categories
    # introduced by client-side tools.  Registered via
    # registry.register_category() so list_tools shows descriptions.


class ToolExecuteRequestEvent(Event):
    """Server requests the WS client to execute a client-registered tool."""
    type: EventType = Field(default=EventType.TOOL_EXECUTE_REQUEST)
    call_id: str = ""
    agent_id: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class ToolExecuteResultEvent(Event):
    """Client returns the result of a client-side tool execution."""
    type: EventType = Field(default=EventType.TOOL_EXECUTE_RESULT)
    call_id: str = ""
    result: str = ""  # JSON-encoded result
    error: str = ""   # Error message if execution failed


class WakeBindResultEvent(Event):
    """Server returns the result of ``session.bind_wake`` / ``session.unbind_wake``.

    ``outcome`` is the ``BindOutcome`` value (``ok`` / ``unauthorized`` /
    ``malformed_key`` / ``too_many_keys`` / ``no_keys`` / ``no_session`` /
    ``unknown``); route on it, not ``detail``.  On a successful ``bind_wake``,
    ``wake_ref`` echoes the (session-supplied) ref and ``expires_at`` is the
    binding's Unix expiry — the values the caller's waker keys on.
    """
    type: EventType = Field(default=EventType.WAKE_BIND_RESULT)
    wake_ref: str = ""
    outcome: str = ""
    detail: str = ""
    expires_at: float = 0.0
    # The daemon's operator-declared PUBLIC wake endpoint (wake.json
    # ``public_url``), so a binding session can embed it as the relay's routing
    # marker WITHOUT any bot-side URL config.  Empty when the operator hasn't
    # declared one (the ingress may still be reachable by other means).
    endpoint: str = ""


class SessionWokenEvent(Event):
    """A wake arrived for a session with NO attached client; the daemon revived
    it and DEFERRED the turn until a client re-attaches.

    Routed to the session's cascade observers (a connected-but-detached client
    that registered ``cascade.register(cid, "observer", ["SessionWokenEvent"])``
    — the cascade filter matches on the event's CLASS NAME
    (``type(event).__name__``), NOT the ``EventType`` value ``"session.woken"``;
    registering the value string silently never matches), so a bot whose session
    went cold can learn it must re-attach to serve the
    woken turn's host tools + render.  Re-emitted whenever an observer
    (re)registers for the cid while a wake is still pending, so a reconnecting
    bot is re-nudged.

    Filter client-side by ``session_id`` (map it to your chat / attach target).
    ``wake_ref`` names the matter (e.g. the PR); ``source`` is the provenance
    tag.  The wake TEXT is NOT here — it stays inside the deferred turn (the
    notification is a signal to attach, not the untrusted payload).
    """
    type: EventType = Field(default=EventType.SESSION_WOKEN)
    session_id: str = ""
    wake_ref: str = ""
    source: str = ""


class HistoryRequest(Event):
    """Client request for conversation history."""
    type: EventType = Field(default=EventType.HISTORY_REQUEST)
    agent_id: str = "main"  # Which agent's history to get


class HistoryEvent(Event):
    """Conversation history from server."""
    type: EventType = Field(default=EventType.HISTORY)
    agent_id: str = "main"
    history: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ List of serialized Message objects
    turn_accounting: List[Dict[str, Any]] = Field(default_factory=list)
    # ^ Per-turn accounting dicts.  The value type is Any (not int): besides
    # the int token counts ({prompt, output, total}) each entry may carry the
    # richer fields the runner records — ``duration_seconds`` (float) and
    # ``function_calls`` (list) — mirroring TurnCompletedEvent.  A strict
    # ``Dict[str, int]`` here rejected those and took the whole HistoryEvent
    # down (pydantic ValidationError), so a disk-restored session's
    # history.request / attach-replay / snapshot all failed on the accounting
    # alone even though the messages validated fine.


# =============================================================================
# SDK Feature Parity — Session-primitive verbs (Client <-> Server)
#
# Typed WS verbs over the public-side primitives
# ``JaatoSession.inject_prompt`` / ``replay_messages`` /
# ``resolve_fork_point``.  Premium's ``session_ops`` plugin builds
# higher-level model-callable tools (``interrogate_session``,
# ``setup_replay_workspace``, ``replay_in_workspace``,
# ``discard_replay_workspace``) on top of these same primitives;
# the WS verbs let SDK consumers reach the primitives directly
# without going through the model loop.  See
# ``project_backlog_sdk_feature_parity.md``.
# =============================================================================

class InjectPromptRequest(Event):
    """Inject a prompt into a session's message queue.

    Maps to :meth:`JaatoSession.inject_prompt`.  ``source_type``
    selects the queue priority:

    * ``"user"`` — USER priority (mid-turn "steer", interrupts the
      model at the next safe point).
    * ``"child"`` — CHILD priority (queued behind in-flight work; runs
      when the agent would otherwise stop, the "follow-up" pattern).
    * ``"system"`` / ``"event"`` / ``"parent"`` — other priority
      tiers from :class:`SourceType` for reactor / hook callers.

    Single verb covers both pi-agent's ``steer`` and ``followUp``
    patterns via the priority dimension.
    """
    type: EventType = Field(default=EventType.INJECT_PROMPT_REQUEST)
    text: str = ""
    source_type: str = "user"  # "user" | "child" | "system" | "event" | "parent"
    source_id: Optional[str] = None  # caller identifier for telemetry / logs
    #: Correlates this inject with the :class:`InjectPromptResultEvent` that
    #: answers it.  ``None`` (the default, and what every pre-1.3 client
    #: sends) keeps the historical fire-and-forget behaviour: the daemon
    #: still routes the prompt, it just emits no result event.
    request_id: Optional[str] = None


class InjectPromptResultEvent(Event):
    """Server's response to :class:`InjectPromptRequest`.

    Answers the only question an injecting caller actually has: **after
    this call, will the target act on the message?**  The pre-1.3 verb
    could not answer it — the runner's ``{"ok": True}`` was discarded by
    the daemon and the SDK method returned ``None`` — so a driver got the
    same silence whether its target was busy, idle, stranded, or dead.

    ``status`` is one of the constants in ``shared.message_delivery``:

    * ``"accepted"``    — the target was idle, so a turn was STARTED on it.
    * ``"queued"``      — the target is mid-turn; its running turn will
      drain the message.
    * ``"terminated"``  — the target is loaded but terminal and will run no
      further turns.  Reported from the target's own terminal stamp, never
      inferred from silence.
    * ``"no_session"``  — no session with that id is loaded.
    * ``"unreachable"`` — loaded and live, but NOTHING WAS SENT: no server
      attached, no runner channel, a runner too old to accept the offer verb,
      or a drive that failed.  A transport fault, not a decision by the
      target.  **Re-sending is safe** — nothing was enqueued, so it cannot
      duplicate — though it will keep failing until the path is restored.
    * ``"not_confirmed"`` — an offer WAS made and its answer was lost (the
      call raised or timed out).  The message may be in the target's queue
      right now, or may never have arrived; from here those are
      indistinguishable.  **Re-sending may deliver it twice.**

    A consumer that only checks membership of the delivered set stays correct
    when a word is added here -- which is how ``not_confirmed`` was split out
    of ``unreachable`` without touching a caller.  Branch on the set, not on
    the individual failure words, unless you are choosing whether to retry.

    Only ``accepted`` and ``queued`` mean the message will be acted on
    (``shared.message_delivery.DELIVERED``).  The rest are failures and must
    not be read as success: a caller that assumes delivery and is wrong gets
    a silent stall it cannot attribute, which is the expensive direction to
    be wrong in.

    ``detail`` carries a human-readable elaboration when one exists.  It is
    **omitted rather than filled with a placeholder** when there is nothing
    to say — a reader of ``"unknown"`` is back where they started, so
    absence is left checkable instead of forgeable.
    """
    type: EventType = Field(default=EventType.INJECT_PROMPT_RESULT)
    request_id: str = ""
    status: str = ""
    detail: Optional[str] = None


class ReplayMessagesRequest(Event):
    """Re-run the model loop against an explicit message list.

    Maps to :meth:`JaatoSession.replay_messages`.  When ``messages``
    is omitted, replays the session's current ``get_history()`` —
    semantically equivalent to "continue from the current state with
    no new user input" (pi-agent's ``continue()`` shape).

    Acquires exclusive provider access internally so concurrent
    in-flight turn calls are serialised.  Does NOT mutate session
    history or turn accounting.  Use when you want a one-shot
    completion against an arbitrary message list — fork/interrogate
    flows compose this with ``resolve_fork_point``.
    """
    type: EventType = Field(default=EventType.REPLAY_MESSAGES_REQUEST)
    request_id: str = ""
    messages: Optional[List[Dict[str, Any]]] = None
    # ^ When None, replay uses the session's current get_history().
    #   When supplied, must be the serialized form of List[Message]
    #   (same shape as HistoryEvent.history).
    timeout_seconds: float = 120.0


class ReplayMessagesResultEvent(Event):
    """Server's response to :class:`ReplayMessagesRequest`."""
    type: EventType = Field(default=EventType.REPLAY_MESSAGES_RESULT)
    request_id: str = ""
    response_text: str = ""  # The model's text response (empty on error)
    error: str = ""           # Populated when the replay failed


class ResolveForkPointRequest(Event):
    """Resolve a fork point in the session's history to a message index.

    Maps to :meth:`JaatoSession.resolve_fork_point`.  Exactly one of
    ``after_message`` / ``after_tool_call`` / ``after_timestamp``
    should be supplied; if none are given, the server returns the
    last message index (full-history fork).  The session's current
    ``get_history()`` is used as the search space — clients don't
    pass history over the wire.

    Composes with :class:`ReplayMessagesRequest` so a client can
    resolve a fork point, snapshot history up to that point, edit,
    and replay — the same shape premium's ``interrogate_session``
    tool uses internally.
    """
    type: EventType = Field(default=EventType.RESOLVE_FORK_POINT_REQUEST)
    request_id: str = ""
    after_message: Optional[int] = None
    # ^ Direct message index specifier.
    after_tool_call: Optional[str] = None
    # ^ Tool call ID specifier — server scans for the message
    #   carrying this ``FunctionCall.id`` or the corresponding
    #   ``ToolResult``.
    after_timestamp: Optional[str] = None
    # ^ HH:MM:SS or ISO timestamp — server returns the index of the
    #   last message at or before this time (best-effort, based on
    #   session turn accounting).


class ResolveForkPointResultEvent(Event):
    """Server's response to :class:`ResolveForkPointRequest`."""
    type: EventType = Field(default=EventType.RESOLVE_FORK_POINT_RESULT)
    request_id: str = ""
    fork_index: int = -1     # -1 if no suitable point exists
    error: str = ""


# =============================================================================
# SDK Feature Parity — Permission policy verbs (Client <-> Server)
#
# Typed verbs replacing stringly-typed
# ``CommandRequest("permissions", [...])`` for SDK consumers.  The
# CLI command path stays for actual users typing.  See
# ``project_backlog_sdk_feature_parity.md`` "Broader pattern
# (audit follow-up)" — these are the first concrete instance of
# typed-WS-verbs-for-plugin-APIs.
# =============================================================================

class PermissionAddWhitelistRequest(Event):
    """Add tools / patterns to the session's permission whitelist.

    Maps to :meth:`PermissionPlugin.add_whitelist_tools` for tools
    and :meth:`PermissionPolicy.add_session_whitelist` for patterns.
    Tools and patterns can be supplied together — both lists are
    additive.
    """
    type: EventType = Field(default=EventType.PERMISSION_ADD_WHITELIST_REQUEST)
    tools: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)


class PermissionAddBlacklistRequest(Event):
    """Add tools / patterns to the session's permission blacklist.

    Maps to :meth:`PermissionPolicy.add_session_blacklist` for both
    tools and patterns.  Tools and patterns supplied together —
    both lists are additive.
    """
    type: EventType = Field(default=EventType.PERMISSION_ADD_BLACKLIST_REQUEST)
    tools: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)


class PermissionRemoveRequest(Event):
    """Remove tools / patterns from a permission list.

    ``target`` selects which list: ``"whitelist"`` or
    ``"blacklist"``.  Empty lists are no-ops.
    """
    type: EventType = Field(default=EventType.PERMISSION_REMOVE_REQUEST)
    target: str = "whitelist"  # "whitelist" | "blacklist"
    tools: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)


class PermissionClearRequest(Event):
    """Clear the session-level permission lists.

    ``target`` selects which list to clear: ``"whitelist"``,
    ``"blacklist"``, or ``"all"`` (clears both).  Does NOT affect
    the base policy declared in ``permissions.json``; only the
    session-level overrides.
    """
    type: EventType = Field(default=EventType.PERMISSION_CLEAR_REQUEST)
    target: str = "all"  # "whitelist" | "blacklist" | "all"


class PermissionSetDefaultRequest(Event):
    """Set the session-level default permission policy.

    ``policy`` is one of ``"allow"`` | ``"deny"`` | ``"ask"``.
    Maps to ``PermissionPolicy.session_default_policy`` — overrides
    the base default for this session only.
    """
    type: EventType = Field(default=EventType.PERMISSION_SET_DEFAULT_REQUEST)
    policy: str = "ask"


class PermissionPolicySnapshotRequest(Event):
    """Request a structured snapshot of the current permission policy."""
    type: EventType = Field(default=EventType.PERMISSION_POLICY_SNAPSHOT_REQUEST)
    request_id: str = ""


class PermissionPolicySnapshotEvent(Event):
    """Structured permission policy snapshot.

    Returned in response to :class:`PermissionPolicySnapshotRequest`.
    Carries the full policy state — base policy + session overrides
    — so clients can build introspection UIs without going through
    the stringly-typed ``permissions check`` command.
    """
    type: EventType = Field(default=EventType.PERMISSION_POLICY_SNAPSHOT)
    request_id: str = ""
    default_policy: str = "ask"          # base default (from permissions.json)
    session_default_policy: Optional[str] = None  # session override, when set
    whitelist_tools: List[str] = Field(default_factory=list)
    whitelist_patterns: List[str] = Field(default_factory=list)
    blacklist_tools: List[str] = Field(default_factory=list)
    blacklist_patterns: List[str] = Field(default_factory=list)
    session_whitelist: List[str] = Field(default_factory=list)
    session_blacklist: List[str] = Field(default_factory=list)


# =============================================================================
# Workspace Management Requests (Client -> Server)
# =============================================================================

class WorkspaceListRequest(Event):
    """Client requests list of available workspaces."""
    type: EventType = Field(default=EventType.WORKSPACE_LIST_REQUEST)


class WorkspaceCreateRequest(Event):
    """Client requests creation of a new workspace."""
    type: EventType = Field(default=EventType.WORKSPACE_CREATE_REQUEST)
    name: str = ""  # Name for the new workspace (becomes subdirectory name)


class WorkspaceSelectRequest(Event):
    """Client selects a workspace to use for the session."""
    type: EventType = Field(default=EventType.WORKSPACE_SELECT_REQUEST)
    name: str = ""  # Workspace name (relative path from root)


class ConfigUpdateRequest(Event):
    """Client updates workspace configuration (provider, model, API key)."""
    type: EventType = Field(default=EventType.CONFIG_UPDATE_REQUEST)
    provider: str = ""  # Provider name (anthropic, google, github, etc.)
    model: Optional[str] = None  # Model name (optional, uses provider default)
    api_key: Optional[str] = None  # API key (optional, for non-OAuth providers)


class StagedFileSpec(BaseModel):
    """Per-file metadata sent inside a :class:`StageFilesRequest`.

    The ``size`` is the exact byte length of the binary frame that will
    follow for this file. The server validates the matching binary
    frame's length against this value; a mismatch is fatal for the
    whole staging operation.

    ``content_type`` is informational — the server uses it for logging
    and may reflect it in the response. It does not gate writing.

    ``mode`` is reserved for future POSIX permission bits (e.g. 0o755
    for executables in the workspace). Currently ignored by the server;
    files are written with the daemon's umask. Add the field now so
    clients don't need a protocol bump later.
    """
    model_config = ConfigDict(extra='ignore')

    name: str = ""  # Workspace-relative path; rejected if absolute or contains ".."
    size: int = 0   # Exact byte length of the upcoming binary frame
    content_type: Optional[str] = None
    mode: Optional[int] = None


class StageFilesRequest(Event):
    """Stage files into a workspace via a multi-frame WS protocol.

    **Wire protocol:**

    1. Client sends *this* event as one TEXT WS frame.  ``files``
       declares the names, sizes, and (optional) content types of the
       payloads that will follow.
    2. Client immediately sends ``len(files)`` raw BINARY WS frames in
       the **same order** as ``files``.  Each frame's byte length must
       equal the corresponding ``files[i].size``.
    3. Server responds with a TEXT frame carrying
       :class:`StageFilesEvent` summarising what was written.

    The handler reads the binary frames inline (the per-connection
    receive loop preserves frame order, so other event types cannot
    interleave between the request and its blobs).

    ``workspace_id`` identifies the target workspace.  Clients learn
    valid IDs from :class:`WorkspaceCreatedEvent` and
    :class:`SessionInfoEvent`.  An empty value targets the connection's
    currently-selected workspace (the WS server tracks one
    selected workspace per client).

    Caps (server-enforced, configurable per deployment):

    - Per-file ``size`` cap (default 10 MB)
    - Sum of ``size`` values cap (default 50 MB)

    Limits exceeded → server emits a :class:`StageFilesEvent` with the
    failure recorded; no binary frames are read.  Clients should check
    the response before considering files staged.

    Compared to the legacy ``staged_files`` field on the
    ``session.new`` envelope (kept for premium back-compat), this is
    the canonical SDK primitive: workspace-scoped (not session-scoped),
    binary-framed (no base64 inflation), and supports staging into an
    already-existing workspace mid-session.
    """
    type: EventType = Field(default=EventType.WORKSPACE_FILES_STAGE_REQUEST)
    workspace_id: str = ""
    files: List[StagedFileSpec] = Field(default_factory=list)
    # Pydantic coerces nested raw dicts to ``StagedFileSpec`` instances
    # automatically during validation; the ``__post_init__`` shim that
    # used to do this manually under ``@dataclass`` is no longer needed.


class StageFilesEvent(Event):
    """Server's response to :class:`StageFilesRequest`.

    Reports which files were written (by ``name``, in declared order)
    and which failed.  Failures are surfaced per-file so clients can
    retry just the ones that didn't make it.

    Possible per-file error categories:

    - ``"unsafe_path"`` — name is absolute, contains ``..``, or escapes
      the workspace root.
    - ``"size_mismatch"`` — declared ``size`` did not match the binary
      frame length.
    - ``"size_limit_per_file"`` — file exceeded the per-file cap.
    - ``"size_limit_total"`` — sum of declared sizes exceeded the
      total payload cap.
    - ``"workspace_not_found"`` — ``workspace_id`` doesn't match a
      known workspace for this client.
    - ``"io_error"`` — write failed (disk full, permission denied,
      AppArmor refusal, ...).  ``error`` carries the OS message.
    """
    type: EventType = Field(default=EventType.WORKSPACE_FILES_STAGED)
    workspace_id: str = ""
    staged: List[str] = Field(default_factory=list)  # successfully written, by name
    failed: List[Dict[str, str]] = Field(default_factory=list)  # [{"name", "category", "error"}]


class ClientType(str, Enum):
    """Presentation-layer categories for PresentationContext.

    Values describe the *kind* of display surface, not specific apps.
    A Telegram bot and a WhatsApp bot are both ``CHAT``; a browser-based
    UI is ``WEB``; a headless integration is ``API``.
    """
    TERMINAL = "terminal"  # TUI / CLI (rich text, fixed-width)
    WEB = "web"            # Browser-based UI (HTML, responsive)
    CHAT = "chat"          # Messaging platform (Telegram, Slack, WhatsApp, …)
    API = "api"            # Headless / programmatic (plain text)


class CommunicationStyle(str, Enum):
    """How the model should pace and structure its output.

    ``CONVERSATIONAL``: Emit short, frequent messages as work progresses —
    each message should be a self-contained update.  Ideal for chat platforms
    where the user expects back-and-forth interaction.

    ``NARRATIVE``: Produce thorough, well-structured responses once the work
    is complete.  Suited for terminals, web UIs, and programmatic consumers
    that can display longer content at once.
    """
    CONVERSATIONAL = "conversational"
    NARRATIVE = "narrative"


class PresentationContext(BaseModel):
    """Display capabilities and constraints of the connected client.

    Assembled by each client at connection time and transmitted to the server
    via ``ClientConfigRequest.presentation``.  The server stores it on the
    ``JaatoSession`` so that:

    1. **System instructions** include a compact display-context block, letting
       the model adapt its output format (e.g. vertical key-value lists instead
       of wide tables on a narrow mobile screen).
    2. **Formatter pipelines** can use the context for any client-side
       reformatting they choose to implement.

    Clients that support expandable/collapsible UI (Telegram inline keyboards,
    HTML ``<details>`` blocks, TUI scrollable panels) should set
    ``supports_expandable_content = True``.  When enabled, the model is *not*
    asked to avoid wide content — instead the client is expected to collapse
    overflow and offer a "show more" affordance.

    Attributes:
        content_width: Available content width in characters.
        content_height: Available content height in lines (None = unlimited scroll).
        supports_markdown: Whether the client renders markdown.
        supports_tables: Whether markdown pipe-tables render acceptably.
        supports_code_blocks: Whether fenced code blocks are rendered.
        supports_images: Whether inline images can be displayed.
        supports_rich_text: Whether bold/italic/links are rendered.
        supports_unicode: Whether wide characters and emoji are safe.
        supports_mermaid: Whether Mermaid diagrams are rendered.
        supports_expandable_content: Whether the client can collapse overflow
            behind an expand/click affordance (e.g. Telegram inline buttons,
            HTML details, TUI panels).
        renderable_media: MIME types this client can present to a person,
            e.g. ``["image/png", "audio/*"]``.  Wildcards of the form
            ``type/*`` are honoured by :meth:`can_render_media`.  Empty
            (the default) means the client can present none -- the honest
            answer for a plain terminal.
        client_type: The kind of client (see ``ClientType`` enum).
    """

    # ── Dimensions ──────────────────────────────────────────────
    content_width: int = 80
    content_height: Optional[int] = None

    # ── Format capabilities ─────────────────────────────────────
    supports_markdown: bool = True
    supports_tables: bool = True
    supports_code_blocks: bool = True
    supports_images: bool = False
    supports_rich_text: bool = True
    supports_unicode: bool = True
    supports_mermaid: bool = False
    supports_expandable_content: bool = False

    # ── Playable / renderable media ─────────────────────────────
    # MIME types (or ``type/*`` wildcards) this client can present to a
    # person.  A TUI declares none, a web client image+audio, a voice
    # client audio only.  Empty is the honest default: a client that has
    # not said it can play something cannot.
    #
    # This is the CLIENT axis and is kept strictly apart from the MODEL
    # axis (``model_tiers.<tier>.modalities``, consumed by the content
    # gate).  Conflating them is the easiest mistake here: they have
    # different owners and different lifetimes -- what a model can consume
    # is fixed by the tier, what a viewer can play changes per connection.
    renderable_media: List[str] = Field(default_factory=list)

    # ── Client hint ─────────────────────────────────────────────
    client_type: ClientType = ClientType.TERMINAL

    # ── Communication style ────────────────────────────────────
    # When None, inferred from client_type: CHAT → CONVERSATIONAL,
    # all others → NARRATIVE.  Clients may override explicitly.
    communication_style: Optional['CommunicationStyle'] = None

    # ──────────────────────────────────────────────────────────

    def can_render_media(self, mime_type: Optional[str]) -> bool:
        """Whether this client declares it can present ``mime_type``.

        Matches an exact type first, then a ``type/*`` wildcard.  Any
        parameters on the supplied type are ignored for matching, so a
        client declaring ``"audio/pcm"`` still matches a payload tagged
        ``"audio/pcm;rate=24000;channels=1"`` -- the parameters describe
        how to play it, not whether it can be played.

        A falsy ``mime_type`` is not renderable: absence of a type is not
        a claim about one.
        """
        if not mime_type:
            return False
        base = mime_type.split(";", 1)[0].strip().lower()
        if not base:
            return False
        declared = {m.strip().lower() for m in self.renderable_media if m}
        if base in declared:
            return True
        top = base.split("/", 1)[0]
        return f"{top}/*" in declared

    def to_system_instruction(self) -> str:
        """Generate a compact display-context block for system instructions.

        The returned string is appended to the model's system prompt so it
        can make intelligent formatting decisions.  The instruction is kept
        deliberately short (30-80 tokens) to minimise overhead.

        Returns:
            A multi-line instruction string describing display constraints.
        """
        lines: List[str] = [
            "## Display Context",
            f"Output width: {self.content_width} characters.",
        ]

        # ── Width-dependent guidance ────────────────────────────
        if self.supports_expandable_content:
            lines.append(
                "The client can collapse wide or long content behind an "
                "expandable control. You may use full-width tables and "
                "detailed output freely."
            )
        elif self.content_width < 60:
            lines.append(
                "This is a NARROW display. Avoid markdown tables — "
                "use vertical key: value lists instead. "
                f"Keep lines under {self.content_width} characters."
            )
        elif self.content_width < 100:
            lines.append(
                "Prefer compact tables (3-4 columns max). "
                "For wider data, use vertical key: value format."
            )

        # ── Capability restrictions ─────────────────────────────
        if not self.supports_tables:
            lines.append(
                "Markdown tables are NOT supported. "
                "Use bullet lists or indented key: value pairs."
            )

        if not self.supports_code_blocks:
            lines.append(
                "Fenced code blocks are NOT supported. "
                "Indent code with 4 spaces instead."
            )

        if not self.supports_markdown:
            lines.append("Markdown is NOT supported. Use plain text only.")

        if self.supports_images:
            lines.append("Inline images are supported.")

        # ── Communication style guidance ──────────────────────
        effective_style = self.communication_style
        if effective_style is None:
            effective_style = (
                CommunicationStyle.CONVERSATIONAL
                if self.client_type == ClientType.CHAT
                else CommunicationStyle.NARRATIVE
            )

        if effective_style == CommunicationStyle.CONVERSATIONAL:
            lines.append(
                "Communication style: conversational. "
                "The user sees messages in a chat interface. "
                "Send short, frequent updates as you work rather than "
                "one large response at the end. Each message should be "
                "a self-contained progress update. "
                "Prefer multiple brief messages over a single long one."
            )
        else:
            lines.append(
                "Communication style: narrative. "
                "Provide thorough, well-structured responses."
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for event transport."""
        return {
            "content_width": self.content_width,
            "content_height": self.content_height,
            "supports_markdown": self.supports_markdown,
            "supports_tables": self.supports_tables,
            "supports_code_blocks": self.supports_code_blocks,
            "supports_images": self.supports_images,
            "supports_rich_text": self.supports_rich_text,
            "supports_unicode": self.supports_unicode,
            "supports_mermaid": self.supports_mermaid,
            "supports_expandable_content": self.supports_expandable_content,
            "client_type": self.client_type.value,
            "communication_style": self.communication_style.value if self.communication_style else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PresentationContext':
        """Create from a dict (e.g. deserialized from ClientConfigRequest)."""
        return cls(
            content_width=data.get("content_width", 80),
            content_height=data.get("content_height"),
            supports_markdown=data.get("supports_markdown", True),
            supports_tables=data.get("supports_tables", True),
            supports_code_blocks=data.get("supports_code_blocks", True),
            supports_images=data.get("supports_images", False),
            supports_rich_text=data.get("supports_rich_text", True),
            supports_unicode=data.get("supports_unicode", True),
            supports_mermaid=data.get("supports_mermaid", False),
            supports_expandable_content=data.get("supports_expandable_content", False),
            client_type=ClientType(data.get("client_type", "terminal")),
            communication_style=(
                CommunicationStyle(data["communication_style"])
                if data.get("communication_style")
                else None
            ),
        )


class ClientConfigRequest(Event):
    """Client sends its configuration to the server.

    Sent after connection to apply client-specific settings like trace paths
    and display capabilities.  The ``presentation`` dict is deserialized into
    a ``PresentationContext`` on the server side.
    """
    type: EventType = Field(default=EventType.CLIENT_CONFIG)
    # Environment overrides from client's .env
    trace_log_path: Optional[str] = None  # JAATO_TRACE_LOG
    provider_trace_log: Optional[str] = None  # client: PROVIDER_TRACE_LOG → server: JAATO_PROVIDER_TRACE
    # Client's working directory (for finding config files like .lsp.json)
    working_dir: Optional[str] = None
    # Optional override for the read-only framework-config search root.
    # When unset (the default), the daemon scans
    # ``<working_dir>/.jaato/`` for profiles, agents, prompts,
    # references, completion_schemas, instructions, scripts, services,
    # etc.  When set, it scans ``<config_root>`` instead — letting
    # clients decouple the agent's filesystem (``working_dir`` /
    # workspace) from where the framework reads its config.  The user
    # tier (``~/.jaato/``) is always honored regardless.  See
    # ``shared/config_resolver.py``.
    config_root: Optional[str] = None
    # Path to client's .env file - server loads this for session creation
    # This provides all provider-related env vars (PROJECT_ID, JAATO_PROVIDER, etc.)
    env_file: Optional[str] = None
    # Client display capabilities (PresentationContext as dict).
    # Keys: content_width, content_height, supports_markdown, supports_tables,
    #        supports_code_blocks, supports_images, supports_rich_text,
    #        supports_unicode, supports_mermaid, supports_expandable_content,
    #        client_type.
    presentation: Optional[Dict[str, Any]] = None
    # Permission timeout override (seconds). 0 = wait forever.
    # WS clients typically set 0 since the user may not be watching.
    permission_timeout: Optional[int] = None
    # Opt-in AppArmor confinement for this client's sessions.
    #
    # ``False`` (the default) preserves the long-standing IPC behavior:
    # sessions run unconfined because the local user already has full
    # filesystem access.  ``True`` asks the daemon to provision a
    # per-session AppArmor profile (same machinery used for WS-
    # provisioned workspaces) confining the session's tool plugins to
    # ``working_dir`` (rw), ``config_root`` (read-only), the standard
    # user-tier ``~/.jaato/`` config, and the venv / source tree.
    #
    # Useful for orchestrator-driven test harnesses that want
    # kernel-enforced isolation of the agent's filesystem even though
    # they connect over IPC — the threat model there is the LLM going
    # off-script, not the local user.
    #
    # When AppArmor is unavailable on the host (non-Linux, kernel
    # module not loaded, ``apparmor_parser`` missing) the session
    # falls back to running unconfined — but the daemon always emits
    # a ``SystemMessageEvent`` describing the outcome (``[apparmor]
    # confinement applied (...)`` for info, ``[apparmor] requested
    # but ...`` for warnings) so the client can surface it to the
    # user.  See ``docs/apparmor-setup.md`` for the prerequisites.
    apparmor: bool = False


# =============================================================================
# Mid-Turn Prompt Events
# =============================================================================

class MidTurnPromptQueuedEvent(Event):
    """Sent when a user prompt is queued during model processing.

    Instead of returning an error when the user sends a message while the model
    is running, the message is queued and will be injected at the next natural
    pause point (between tool executions, after subagent completion, etc.).
    """
    type: EventType = Field(default=EventType.MID_TURN_PROMPT_QUEUED)
    text: str = ""
    position_in_queue: int = 0  # 0-based position (usually 0, can be >0 if multiple queued)


class MidTurnPromptInjectedEvent(Event):
    """Sent when a queued prompt is injected into the conversation.

    This notifies the client that the queued prompt is now being processed
    by the model.
    """
    type: EventType = Field(default=EventType.MID_TURN_PROMPT_INJECTED)
    text: str = ""


class MidTurnInterruptEvent(Event):
    """Sent when streaming is interrupted to process a mid-turn user prompt.

    This notifies the client that the model's current generation was interrupted
    because a user prompt arrived and needs to be processed immediately.
    The partial response is preserved and the user's prompt is being processed.
    """
    type: EventType = Field(default=EventType.MID_TURN_INTERRUPT)
    partial_response_chars: int = 0  # How much of the response was generated before interrupt
    user_prompt_preview: str = ""  # First 100 chars of the user's prompt


class InterruptedTurnRecoveredEvent(Event):
    """Sent when the server recovers from an interrupted turn after reconnection.

    This event notifies the client that a turn was interrupted (e.g., by server
    restart) and has been recovered with synthetic error responses injected
    for any pending tool calls.
    """
    type: EventType = Field(default=EventType.INTERRUPTED_TURN_RECOVERED)
    session_id: str = ""
    agent_id: str = ""
    recovered_calls: int = 0  # Number of tool calls that were recovered
    action_taken: str = ""  # What action was taken (e.g., "synthetic_error")


# =============================================================================
# Peer Channel Events (Server-to-Server Gossip)
# =============================================================================

class PeerHeartbeatEvent(Event):
    """Heartbeat sent between peer servers at a configurable interval.

    Contains server identity, workload metrics, and health data used by the
    PeerRegistry to track peer liveness and by the environment aspect (Phase 2)
    to expose cluster state to the model.
    """
    type: EventType = Field(default=EventType.PEER_HEARTBEAT)
    server_id: str = ""
    server_name: str = ""
    server_version: str = ""
    active_sessions: int = 0
    active_agents: int = 0
    available_providers: List[str] = Field(default_factory=list)
    available_models: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    uptime_seconds: float = 0.0
    # Server-level reliability self-report (Phase 4).
    # Primitives only — no enum imports in SDK.
    trust_state: str = "trusted"
    success_rate_1h: float = 1.0
    escalated_tools: int = 0


# =============================================================================
# Peer Spawn Events (Server-to-Server Remote Subagent Delegation)
# =============================================================================

class PeerSpawnRequestEvent(Event):
    """Request to spawn a subagent on a remote peer server.

    Sent from the origin server (where the model called
    ``spawn_subagent(server=...)``) to the remote peer that should execute
    the subagent.  The ``request_id`` correlates all subsequent events in
    this spawn lifecycle.
    """
    type: EventType = Field(default=EventType.PEER_SPAWN_REQUEST)
    request_id: str = ""
    origin_server: str = ""
    agent_name: str = ""
    task: str = ""
    context: str = ""
    profile_json: str = ""
    inline_config_json: str = ""
    # Workspace replication fields (Phase 5)
    workspace_git_url: str = ""
    workspace_branch: str = ""
    workspace_commit: str = ""
    workspace_temp_branch: str = ""


class PeerSpawnAcceptedEvent(Event):
    """Confirmation that a remote peer accepted the spawn request.

    Sent back to the origin server once the remote has created the
    ephemeral session and is about to start processing.
    """
    type: EventType = Field(default=EventType.PEER_SPAWN_ACCEPTED)
    request_id: str = ""
    remote_agent_id: str = ""


class PeerSpawnRejectedEvent(Event):
    """Notification that a remote peer rejected the spawn request.

    The ``reason`` field contains a human-readable explanation (e.g.
    capacity limits, missing provider, unknown profile).
    """
    type: EventType = Field(default=EventType.PEER_SPAWN_REJECTED)
    request_id: str = ""
    reason: str = ""


class PeerAgentOutputEvent(Event):
    """Streamed output chunk from a remote subagent.

    Sent from the remote server back to the origin as the subagent
    produces output.  The origin's ``RemoteSpawnHandler`` forwards these
    to the parent session via ``inject_prompt``.
    """
    type: EventType = Field(default=EventType.PEER_AGENT_OUTPUT)
    request_id: str = ""
    remote_agent_id: str = ""
    text: str = ""
    source: str = ""  # "model" or "tool"


class PeerAgentCompletedEvent(Event):
    """Signal that a remote subagent has finished execution.

    ``success`` indicates whether the subagent completed normally.
    ``summary`` contains a brief result description; ``error`` is
    populated only when ``success`` is False.
    """
    type: EventType = Field(default=EventType.PEER_AGENT_COMPLETED)
    request_id: str = ""
    remote_agent_id: str = ""
    success: bool = True
    summary: str = ""
    error: str = ""
    # Workspace replication field (Phase 5)
    workspace_modified: bool = False


# =============================================================================
# HandoffGate events (jaato-premium reactor framework)
# =============================================================================

class GateState(BaseModel):
    """Snapshot of a single gate's state.

    Used as the payload-level shape inside ``GatesSnapshotEvent`` and
    accessible as a typed property on the live events.  Public/private
    intent split is enforced server-side via ``public_intent_fields`` —
    cross-tenant subscribers receive only the public keys; same-tenant
    subscribers receive the full intent.

    See ``jaato-premium/docs/design/handoff-gate-api.md`` §3.4 for the canonical
    intent shape.
    """
    gate_name: str = ""
    tenant_id: str = ""
    state: str = "green"  # "green" | "red"
    owner: Optional[str] = None              # service-identity ID (when RED)
    intent: Optional[Dict[str, Any]] = None  # populated when RED+announced
    acquired_at: Optional[str] = None        # ISO 8601
    expires_at: Optional[str] = None         # ISO 8601 (acquired_at + ttl)


class GateAnnouncedEvent(Event):
    """A reactor producer announced its intent on a held HandoffGate.

    Fired after a producer reactor calls ``gate.try_acquire(...)`` and
    then ``gate.announce(intent)``.  When ``intent.session_id`` is set,
    subscribers can ``client.attach_session(intent['session_id'])`` to
    observe the spawned session's events.
    """
    type: EventType = Field(default=EventType.GATE_ANNOUNCED)
    gate_name: str = ""
    tenant_id: str = ""
    owner: str = ""                          # service-identity ID
    intent: Dict[str, Any] = Field(default_factory=dict)
    announced_at: str = ""                   # ISO 8601


class GateReleasedEvent(Event):
    """A held HandoffGate was released (work completed, failed, or timed out).

    ``was_announced=False`` indicates the producer crashed or errored
    between ``try_acquire`` and ``announce`` — subscribers that
    auto-attached on the announce event simply have nothing to detach.
    ``outcome.status='timeout'`` indicates the watchdog auto-released
    on TTL expiry.
    """
    type: EventType = Field(default=EventType.GATE_RELEASED)
    gate_name: str = ""
    session_id: str = ""                     # originating session (from the
                                             # gate's announce intent); empty
                                             # for a gate with no session
                                             # association. Lets bus subscribers
                                             # (reactors) target the parked
                                             # session without parsing gate_name.
    tenant_id: str = ""
    owner: str = ""
    outcome: Optional[Dict[str, Any]] = None
    released_at: str = ""                    # ISO 8601
    was_announced: bool = True


class GatesSnapshotEvent(Event):
    """All currently-RED gates, sent on subscribe so late subscribers catch up.

    Mirrors ``SessionInfoEvent`` for sessions: rather than forcing
    every subscriber to track gate state externally across reconnects,
    the registry replays the live state once at subscription time.
    """
    type: EventType = Field(default=EventType.GATES_SNAPSHOT)
    gates: List[GateState] = Field(default_factory=list)
    snapshot_at: str = ""                    # ISO 8601


class PeerStopRequestEvent(Event):
    """Request to cancel a running remote subagent.

    Sent from the origin server when the parent session wants to stop
    a previously spawned remote subagent.
    """
    type: EventType = Field(default=EventType.PEER_STOP_REQUEST)
    request_id: str = ""
    remote_agent_id: str = ""


class PeerStopAcknowledgedEvent(Event):
    """Confirmation that a remote peer received and processed the stop request."""
    type: EventType = Field(default=EventType.PEER_STOP_ACKNOWLEDGED)
    request_id: str = ""
    remote_agent_id: str = ""


# =============================================================================
# Serialization Helpers
# =============================================================================

# Map of event type -> event class
_EVENT_CLASSES: Dict[str, type] = {
    EventType.CONNECTED.value: ConnectedEvent,
    EventType.AGENT_CREATED.value: AgentCreatedEvent,
    EventType.AGENT_OUTPUT.value: AgentOutputEvent,
    EventType.AGENT_STATUS_CHANGED.value: AgentStatusChangedEvent,
    EventType.AGENT_COMPLETED.value: AgentCompletedEvent,
    EventType.AGENT_ERROR.value: AgentErrorEvent,
    EventType.SESSION_TERMINATED.value: SessionTerminatedEvent,
    EventType.SESSION_RESTORED.value: SessionRestoredEvent,
    EventType.SLOT_SETTLED.value: SlotSettledEvent,
    EventType.TOOL_CALL_START.value: ToolCallStartEvent,
    EventType.TOOL_CALL_END.value: ToolCallEndEvent,
    EventType.TOOL_OUTPUT.value: ToolOutputEvent,
    EventType.PERMISSION_REQUESTED.value: PermissionRequestedEvent,
    EventType.PERMISSION_INPUT_MODE.value: PermissionInputModeEvent,
    EventType.PERMISSION_RESOLVED.value: PermissionResolvedEvent,
    EventType.PERMISSION_STATUS.value: PermissionStatusEvent,
    EventType.CLARIFICATION_REQUESTED.value: ClarificationRequestedEvent,
    EventType.CLARIFICATION_INPUT_MODE.value: ClarificationInputModeEvent,
    EventType.CLARIFICATION_QUESTION.value: ClarificationQuestionEvent,
    EventType.CLARIFICATION_RESOLVED.value: ClarificationResolvedEvent,
    EventType.CLARIFICATION_BATCH.value: ClarificationBatchEvent,
    EventType.CLARIFICATION_BATCH_RESPONSE.value: ClarificationBatchResponseEvent,
    EventType.REFERENCE_SELECTION_REQUESTED.value: ReferenceSelectionRequestedEvent,
    EventType.REFERENCE_SELECTION_RESOLVED.value: ReferenceSelectionResolvedEvent,
    EventType.REFERENCE_SELECTION_RESPONSE.value: ReferenceSelectionResponseRequest,
    EventType.WORKSPACE_MISMATCH_REQUESTED.value: WorkspaceMismatchRequestedEvent,
    EventType.WORKSPACE_MISMATCH_RESOLVED.value: WorkspaceMismatchResolvedEvent,
    EventType.WORKSPACE_MISMATCH_RESPONSE.value: WorkspaceMismatchResponseRequest,
    EventType.POST_AUTH_SETUP.value: PostAuthSetupEvent,
    EventType.POST_AUTH_SETUP_RESPONSE.value: PostAuthSetupResponse,
    EventType.PLAN_UPDATED.value: PlanUpdatedEvent,
    EventType.PLAN_STEP_UPDATED.value: PlanStepUpdatedEvent,
    EventType.PLAN_CLEARED.value: PlanClearedEvent,
    EventType.CONTEXT_UPDATED.value: ContextUpdatedEvent,
    EventType.GC_CONFIG.value: GCConfigEvent,
    EventType.INSTRUCTION_BUDGET_UPDATED.value: InstructionBudgetEvent,
    EventType.TURN_COMPLETED.value: TurnCompletedEvent,
    EventType.TURN_PROGRESS.value: TurnProgressEvent,
    EventType.SYSTEM_MESSAGE.value: SystemMessageEvent,
    EventType.HELP_TEXT.value: HelpTextEvent,
    EventType.INIT_PROGRESS.value: InitProgressEvent,
    EventType.ERROR.value: ErrorEvent,
    EventType.RETRY.value: RetryEvent,
    EventType.SESSION_LIST.value: SessionListEvent,
    EventType.GC.value: GCEvent,
    EventType.SESSION_INFO.value: SessionInfoEvent,
    EventType.MEMORY_LIST.value: MemoryListEvent,
    EventType.SANDBOX_PATHS.value: SandboxPathsEvent,
    EventType.SERVICE_LIST.value: ServiceListEvent,
    EventType.SESSION_DESCRIPTION_UPDATED.value: SessionDescriptionUpdatedEvent,
    EventType.SESSION_PROFILES.value: SessionProfilesEvent,
    EventType.SEND_MESSAGE.value: SendMessageRequest,
    EventType.PERMISSION_RESPONSE.value: PermissionResponseRequest,
    EventType.CLARIFICATION_RESPONSE.value: ClarificationResponseRequest,
    EventType.STOP.value: StopRequest,
    EventType.EVENT_EXTERNAL.value: ExternalEventRequest,
    EventType.EVENTS_SUBSCRIBED.value: EventsSubscribedEvent,
    EventType.COMMAND.value: CommandRequest,
    EventType.INSTRUCTION_BUDGET_REQUEST.value: GetInstructionBudgetRequest,
    EventType.COMMAND_LIST_REQUEST.value: CommandListRequest,
    EventType.COMMAND_LIST.value: CommandListEvent,
    EventType.COMMAND_LIST_REFRESH.value: CommandListRefreshEvent,
    EventType.TOOL_STATUS.value: ToolStatusEvent,
    EventType.TOOL_ID_REGISTRY.value: ToolIdRegistryEvent,
    EventType.TOOL_DISABLE_REQUEST.value: ToolDisableRequest,
    EventType.TOOLS_REGISTER_CLIENT.value: ToolsRegisterClientRequest,
    EventType.TOOL_EXECUTE_REQUEST.value: ToolExecuteRequestEvent,
    EventType.TOOL_EXECUTE_RESULT.value: ToolExecuteResultEvent,
    EventType.HISTORY_REQUEST.value: HistoryRequest,
    EventType.HISTORY.value: HistoryEvent,
    EventType.CLIENT_CONFIG.value: ClientConfigRequest,
    EventType.MID_TURN_PROMPT_QUEUED.value: MidTurnPromptQueuedEvent,
    EventType.MID_TURN_PROMPT_INJECTED.value: MidTurnPromptInjectedEvent,
    EventType.MID_TURN_INTERRUPT.value: MidTurnInterruptEvent,
    EventType.INTERRUPTED_TURN_RECOVERED.value: InterruptedTurnRecoveredEvent,
    # Workspace management
    EventType.WORKSPACE_LIST_REQUEST.value: WorkspaceListRequest,
    EventType.WORKSPACE_LIST.value: WorkspaceListEvent,
    EventType.WORKSPACE_CREATE_REQUEST.value: WorkspaceCreateRequest,
    EventType.WORKSPACE_CREATED.value: WorkspaceCreatedEvent,
    EventType.WORKSPACE_SELECT_REQUEST.value: WorkspaceSelectRequest,
    EventType.CONFIG_STATUS.value: ConfigStatusEvent,
    EventType.CONFIG_UPDATE_REQUEST.value: ConfigUpdateRequest,
    EventType.CONFIG_UPDATED.value: ConfigUpdatedEvent,
    # Workspace file monitoring
    EventType.WORKSPACE_FILES_CHANGED.value: WorkspaceFilesChangedEvent,
    EventType.WORKSPACE_FILES_SNAPSHOT.value: WorkspaceFilesSnapshotEvent,
    # Workspace file staging (multi-frame: TEXT request + N BINARY blobs)
    EventType.WORKSPACE_FILES_STAGE_REQUEST.value: StageFilesRequest,
    EventType.WORKSPACE_FILES_STAGED.value: StageFilesEvent,
    # Peer channel
    EventType.PEER_HEARTBEAT.value: PeerHeartbeatEvent,
    EventType.PEER_SPAWN_REQUEST.value: PeerSpawnRequestEvent,
    EventType.PEER_SPAWN_ACCEPTED.value: PeerSpawnAcceptedEvent,
    EventType.PEER_SPAWN_REJECTED.value: PeerSpawnRejectedEvent,
    EventType.PEER_AGENT_OUTPUT.value: PeerAgentOutputEvent,
    EventType.PEER_AGENT_COMPLETED.value: PeerAgentCompletedEvent,
    EventType.PEER_STOP_REQUEST.value: PeerStopRequestEvent,
    EventType.PEER_STOP_ACKNOWLEDGED.value: PeerStopAcknowledgedEvent,
    # HandoffGate (jaato-premium reactor framework)
    EventType.GATE_ANNOUNCED.value: GateAnnouncedEvent,
    EventType.GATE_RELEASED.value: GateReleasedEvent,
    EventType.GATES_SNAPSHOT.value: GatesSnapshotEvent,
    # SDK feature parity — session-primitive verbs
    EventType.INJECT_PROMPT_REQUEST.value: InjectPromptRequest,
    EventType.INJECT_PROMPT_RESULT.value: InjectPromptResultEvent,
    EventType.REPLAY_MESSAGES_REQUEST.value: ReplayMessagesRequest,
    EventType.REPLAY_MESSAGES_RESULT.value: ReplayMessagesResultEvent,
    EventType.RESOLVE_FORK_POINT_REQUEST.value: ResolveForkPointRequest,
    EventType.RESOLVE_FORK_POINT_RESULT.value: ResolveForkPointResultEvent,
    EventType.WAKE_BIND_RESULT.value: WakeBindResultEvent,
    EventType.SESSION_WOKEN.value: SessionWokenEvent,
    # SDK feature parity — permission policy verbs
    EventType.PERMISSION_ADD_WHITELIST_REQUEST.value: PermissionAddWhitelistRequest,
    EventType.PERMISSION_ADD_BLACKLIST_REQUEST.value: PermissionAddBlacklistRequest,
    EventType.PERMISSION_REMOVE_REQUEST.value: PermissionRemoveRequest,
    EventType.PERMISSION_CLEAR_REQUEST.value: PermissionClearRequest,
    EventType.PERMISSION_SET_DEFAULT_REQUEST.value: PermissionSetDefaultRequest,
    EventType.PERMISSION_POLICY_SNAPSHOT_REQUEST.value: PermissionPolicySnapshotRequest,
    EventType.PERMISSION_POLICY_SNAPSHOT.value: PermissionPolicySnapshotEvent,
}


def serialize_event(event: Event) -> str:
    """Serialize an event to JSON string."""
    return event.to_json()


def deserialize_event(json_str: str) -> Event:
    """Deserialize a JSON string to an event object.

    Args:
        json_str: JSON string representing an event.

    Returns:
        The deserialized event object.

    Raises:
        ValueError: If the event type is unknown.
        json.JSONDecodeError: If the JSON is invalid.
    """
    data = json.loads(json_str)
    event_type = data.get("type")

    if event_type not in _EVENT_CLASSES:
        raise ValueError(f"Unknown event type: {event_type}")

    event_class = _EVENT_CLASSES[event_type]

    # Pydantic ``model_config['extra'] = 'ignore'`` on the base Event
    # silently drops unknown fields, so an older client deserialising an
    # event from a newer server doesn't crash on extra keys — same
    # forward-compat semantics as the previous manual filtering.
    return event_class.model_validate(data)


def create_event(event_type: EventType, **kwargs) -> Event:
    """Factory function to create an event by type.

    Args:
        event_type: The type of event to create.
        **kwargs: Event-specific fields.

    Returns:
        The created event object.
    """
    event_class = _EVENT_CLASSES.get(event_type.value)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class(**kwargs)
