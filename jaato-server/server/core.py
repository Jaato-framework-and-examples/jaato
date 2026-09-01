"""JaatoServer - Core logic for multi-client support.

This module extracts the non-UI logic from RichClient into a reusable
server that can be driven by different frontends (TUI, WebSocket, HTTP).

The server emits events for all state changes, allowing clients to
subscribe and render appropriately.
"""

import contextlib
import logging
import os
import re
import sys
import pathlib
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — types only
    from server.runner_rpc_client import RunnerRPCClient
    from server.runner_spawner import SpawnedRunner

logger = logging.getLogger(__name__)

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIMPLE_CLIENT = ROOT / "simple-client"
if str(SIMPLE_CLIENT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_CLIENT))

RICH_CLIENT = ROOT / "jaato-tui"
if str(RICH_CLIENT) not in sys.path:
    sys.path.insert(0, str(RICH_CLIENT))

from dotenv import load_dotenv

from shared import (
    JaatoRuntime,
    TokenLedger,
    PluginRegistry,
    PermissionPlugin,
    TodoPlugin,
    active_cert_bundle,
)
from shared.dynamic_instructions import DynamicInstructionsError
from shared.instruction_suppression import normalize_suppression
from shared.instruction_token_cache import InstructionTokenCache
from shared.message_queue import SourceType
from shared.plugins.session import create_plugin as create_session_plugin, load_session_config
from jaato_sdk.plugins.base import parse_command_args, HelpLines
from shared.plugins.gc import load_gc_from_file
from shared.bootstrap_timing import BootstrapTimer

# Formatter pipeline for server-side output formatting
from shared.plugins.formatter_pipeline import FormatterRegistry, create_registry

# Plan reporter from SDK (callback-based TodoReporter)
from jaato_sdk.plugins.todo.channels import create_live_reporter

# Import events from SDK
from jaato_sdk.event_bus import (
    EventType as BusEventType,
    Event as BusEvent,
)
from jaato_sdk.events import (
    Event,
    EventType,
    ConnectedEvent,
    AgentCreatedEvent,
    AgentOutputEvent,
    AgentStatusChangedEvent,
    AgentCompletedEvent,
    AgentErrorEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolOutputEvent,
    PermissionInputModeEvent,
    PermissionResolvedEvent,
    PermissionStatusEvent,
    ClarificationInputModeEvent,
    ClarificationResolvedEvent,
    ClarificationBatchEvent,
    ReferenceSelectionRequestedEvent,
    ReferenceSelectionResolvedEvent,
    ReferenceSelectionResponseRequest,
    PlanUpdatedEvent,
    PlanStepUpdatedEvent,
    PlanClearedEvent,
    ContextUpdatedEvent,
    GCConfigEvent,
    InstructionBudgetEvent,
    TurnCompletedEvent,
    TurnProgressEvent,
    UsageBreakdown,
    SystemMessageEvent,
    HelpTextEvent,
    InitProgressEvent,
    ErrorEvent,
    RetryEvent,
    SessionInfoEvent,
    SessionDescriptionUpdatedEvent,
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    StopRequest,
    CommandRequest,
    MidTurnPromptQueuedEvent,
    MidTurnPromptInjectedEvent,
    MidTurnInterruptEvent,
    MemoryListEvent,
    SandboxPathsEvent,
    ServiceListEvent,
    serialize_event,
    deserialize_event,
)


# Type alias for event callback
EventCallback = Callable[[Event], None]

# Mapping from server EventType to bus EventType.
# Events not in this mapping bypass the bus and go directly to clients
# (init progress, errors, session list, help text, etc.).
_SERVER_TO_BUS: Dict[EventType, BusEventType] = {
    EventType.AGENT_CREATED: BusEventType.AGENT_CREATED,
    EventType.AGENT_OUTPUT: BusEventType.AGENT_OUTPUT,
    EventType.AGENT_STATUS_CHANGED: BusEventType.AGENT_STATUS_CHANGED,
    EventType.AGENT_COMPLETED: BusEventType.AGENT_COMPLETED,
    EventType.AGENT_ERROR: BusEventType.AGENT_ERROR,
    EventType.TOOL_CALL_START: BusEventType.TOOL_CALL_STARTED,
    EventType.TOOL_CALL_END: BusEventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_OUTPUT: BusEventType.TOOL_OUTPUT,
    EventType.PLAN_STEP_UPDATED: BusEventType.PLAN_STEP_UPDATED,
    EventType.TURN_COMPLETED: BusEventType.TURN_COMPLETED,
    EventType.TURN_PROGRESS: BusEventType.TURN_PROGRESS,
    EventType.CONTEXT_UPDATED: BusEventType.CONTEXT_UPDATED,
    EventType.PERMISSION_INPUT_MODE: BusEventType.PERMISSION_REQUESTED,
    EventType.PERMISSION_RESOLVED: BusEventType.PERMISSION_RESOLVED,
    # Server 0.6.162+ (Bug C fix): bridge SessionTerminatedEvent to
    # the bus so reactor rules matching event_type=session.terminated
    # actually fire.  Pre-0.6.162 the event bypassed the bus entirely
    # (went directly to IPC/WS clients via _on_event), which meant
    # the reactor engine — which subscribes via bus.subscribe — never
    # saw terminal events.  premium 0.1.188's build_merged_view fix
    # (Bug A) made `reason` JMESPath-visible, but the event wasn't
    # reaching the matcher at all.  See PR for diagnosis trace.
    EventType.SESSION_TERMINATED: BusEventType.SESSION_TERMINATED,
    # Cascade stage settled — bridged so cascade reactors can gate next-stage
    # spawn on this universal per-stage event (SlotSettledEvent, cascade only).
    EventType.SLOT_SETTLED: BusEventType.SLOT_SETTLED,
    # HandoffGate release — bridged so reactor rules matching
    # event_type=gate.released fire (premium reliability's T3 gate-park
    # resume). Same bug+fix shape as SESSION_TERMINATED above: pre-bridge the
    # event went only to client sinks via the registry's daemon-wide
    # broadcast_event, so the reactor engine (bus subscriber) never saw it.
    # The GateRegistry now routes gate.released through the parked session's
    # server.emit() (premium); gate.announced stays a daemon-wide client
    # broadcast. See jaato-premium docs/design/gate-released-bus-delivery.md.
    EventType.GATE_RELEASED: BusEventType.GATE_RELEASED,
}


def _server_event_to_bus_event(server_event: Event) -> Optional[BusEvent]:
    """Convert a server event to a bus event.

    Extracts the server event's dataclass fields into the bus event payload.
    Returns None for unmapped event types (those go directly to clients).
    """
    bus_type = _SERVER_TO_BUS.get(server_event.type)
    if bus_type is None:
        return None

    # Flatten dataclass fields to payload dict, excluding base Event fields
    payload = {
        k: v for k, v in server_event.to_dict().items()
        if k not in ("type", "timestamp")
    }

    # Hoist a nested typed ``payload`` (e.g. AgentCompletedEvent's validated
    # signal_completion payload) to the bus-event top level.  ``to_dict()``
    # leaves the typed payload nested under a ``payload`` key, but reactor
    # consumers read it via ``build_merged_view``'s single
    # ``view.update(event.payload)`` hoist — one level too shallow to reach
    # the nested fields.  Without this hoist a cascade's ``event.get("facts")``
    # returned None despite a validated typed payload (the
    # ``AgentCompletedEvent.payload`` contract was honoured on the raw event
    # but lost on the bus hop the reactor actually receives).  ``setdefault``
    # so the typed fields never clobber envelope identity (agent_id, success,
    # ...); the nested ``payload`` key is preserved for back-compat.
    typed_payload = payload.get("payload")
    if isinstance(typed_payload, dict):
        for k, v in typed_payload.items():
            payload.setdefault(k, v)

    return BusEvent.create(
        event_type=bus_type,
        source_agent=payload.get("agent_id", "server"),
        payload=payload,
    )


def _extract_provider_request_id(exc: Optional[BaseException]) -> Optional[str]:
    """Best-effort provider request-id extraction for ``AgentErrorEvent``.

    Provider SDK exceptions expose the upstream request id under various
    attributes (OpenAI/OpenRouter: ``request_id``; some wrappers stash it on
    ``response.headers["x-request-id"]``).  Returns the first non-empty value
    found, else ``None``.  Walks the ``__cause__`` chain once because the
    framework often re-raises a jaato-typed error wrapping the provider's.
    Purely informational — never raises.
    """
    seen = 0
    cur: Optional[BaseException] = exc
    while cur is not None and seen < 4:
        rid = getattr(cur, "request_id", None)
        if isinstance(rid, str) and rid:
            return rid
        resp = getattr(cur, "response", None)
        headers = getattr(resp, "headers", None)
        if headers is not None:
            try:
                hid = headers.get("x-request-id") or headers.get("X-Request-Id")
            except Exception:
                hid = None
            if isinstance(hid, str) and hid:
                return hid
        cur = getattr(cur, "__cause__", None)
        seen += 1
    return None


class AgentState:
    """Tracks state for a single agent."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        profile_name: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.profile_name = profile_name
        self.parent_agent_id = parent_agent_id
        self.status = "idle"  # idle, active, done, error
        # Set by the completion-nudge guard when the framework gives up
        # asking for signal_completion; read and CLEARED by the next
        # on_agent_turn_completed so it rides exactly one event.
        self.completion_gap: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.history: List[Any] = []
        self.turn_accounting: List[Dict] = []
        self.context_usage: Dict[str, Any] = {}
        # GC configuration (set when agent is created with GC)
        self.gc_threshold: Optional[float] = None
        self.gc_strategy: Optional[str] = None
        self.gc_target_percent: Optional[float] = None
        self.gc_continuous_mode: bool = False
        # Per-agent formatter pipeline for output formatting
        # Initialized lazily via JaatoServer._get_agent_pipeline()
        self.formatter_pipeline: Optional[Any] = None
        # Pending formatter feedback for auto-continuation
        self.pending_formatter_feedback: Optional[str] = None


from shared.model_tiers import bound_model_for_profile


def _profile_binds_a_model(profile: Any) -> bool:
    """Does *profile* bind a model for session start, by EITHER route?

    Thin wrapper over :func:`shared.model_tiers.bound_model_for_profile`,
    which ``runner_spawn`` also uses to fill ``envelope.model_name``.  They
    MUST agree: when the gate said yes and the envelope said empty, the
    caller got a dropped IPC connection and "session not bootstrapped on this
    runner" instead of a configuration error.
    """
    return bound_model_for_profile(profile) is not None


class JaatoServer:
    """Core server logic for Jaato - UI-agnostic.

    This class manages:
    - JaatoClient and plugins
    - Agent lifecycle and state
    - Message processing
    - Permission/clarification flows
    - Event emission for clients

    Clients subscribe to events via the `on_event` callback and send
    requests via the public methods.
    """

    #: Declared at class scope so the ``session_id`` property can honour its
    #: own ``Optional[str]`` annotation.  ``__init__`` always assigns it, but
    #: fifteen test modules construct this class via ``__new__`` to exercise
    #: one method without standing up a session -- and on those, reading the
    #: property raised ``AttributeError`` instead of returning ``None``.
    #:
    #: Not a fallback: it makes ``__init__``'s assignment an override of a
    #: declared default rather than the attribute's only creation, which is
    #: what the annotation already claimed.
    _session_id: Optional[str] = None

    def __init__(
        self,
        env_file: str = ".env",
        provider: Optional[str] = None,
        on_event: Optional[EventCallback] = None,
        workspace_path: Optional[str] = None,
        session_id: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        instruction_token_cache: Optional[InstructionTokenCache] = None,
        profile: Optional[Any] = None,
        system_instruction_override: Optional[str] = None,
        suppress_base_instructions: Any = False,
        agent_name: Optional[str] = None,
    ):
        """Initialize the server.

        Args:
            env_file: Path to .env file.
            provider: Model provider override (e.g., 'google_genai').
            on_event: Callback for emitting events to clients.
            workspace_path: Client's working directory for file operations.
                           If provided, the server will chdir to this path
                           when processing requests.
            session_id: Unique identifier for this session (used in logs).
            env_overrides: Optional dict of env vars that take precedence over
                          the .env file (e.g., from post-auth wizard).
            instruction_token_cache: Optional shared cache for instruction
                token counts, passed from ``SessionManager`` so cached counts
                survive across session creates/restores within a daemon.
            profile: Optional ``SubagentProfile`` instance to apply during
                initialization. When set, the profile's model, provider,
                plugins, system_instructions, plugin_configs, and GC
                settings override the session defaults.
            system_instruction_override: If provided, replaces the
                fully-assembled system instruction for this session.  Plugin
                enrichment and the agent prompt are still computed (for
                budget accounting side effects) but their output is
                discarded in favour of this string.  Pass ``""`` to send
                the model no system message at all — the only viable option
                when the model's context window is too small for the
                assembled prompt (e.g. an 8K model with a 30K+ enriched
                instruction).  Forwarded to ``JaatoSession.configure``.
            suppress_base_instructions: Partial-suppression flag — drop
                only the BASE layer (``.jaato/instructions/*.md`` + premium
                baseline) while keeping the agent content, plugin
                instructions, and framework constants.  The intended
                default for small-context model sessions that still want
                the agent's own prompt and the tool-specific hints.
                Forwarded to ``JaatoSession.configure``.  Ignored when
                ``system_instruction_override`` is also set.
            agent_name: Optional agent identifier resolved from
                ``--agent <name>``.  When set, this becomes the main
                agent's ``agent_id`` (instead of the default ``"main"``)
                so reactor rules and other event consumers can route on
                the agent's logical identity.  When ``None``, the main
                agent's id remains ``"main"`` for backwards compatibility.
        """
        self.env_file = env_file
        self._env_overrides = env_overrides or {}
        self._provider = provider
        self._profile = profile
        self._system_instruction_override = system_instruction_override
        # Canonical frozenset of framework instruction pieces to drop
        # (accepts bool / dict / list / frozenset; see instruction_suppression).
        self._suppress_base_instructions = normalize_suppression(
            suppress_base_instructions
        )
        # Client-provided ("host") tools registered via the WS/IPC protocol
        # (websocket._register_client_tools).  name -> schema dict.  Read by
        # spawn_session_runner to seed envelope.client_tools so the RUNNER-tier
        # model sees them in list_tools (registering only on self.registry left
        # the runner model blind — the #344-sibling daemon-vs-runner split).
        self.client_tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._on_event = on_event or (lambda e: None)
        self._on_auth_complete: Optional[Callable[[], None]] = None

        # Identity of this server's primary ("main") agent.  Defaults to
        # the literal ``"main"`` when no ``--agent <name>`` was supplied.
        # When an agent_name is given, that name is used as the agent_id
        # so consumers (reactor rules, event subscribers) can match on
        # the agent's logical identity.  All emit/lookup sites that
        # previously hardcoded ``"main"`` should reference this attribute.
        self._main_agent_id: str = agent_name or "main"
        self._main_agent_display_name: Optional[str] = agent_name

        # Plug-in transformer chain for outbound events (seat 3 of the
        # four-seat pseudonymization design — see
        # docs/design/daemon-extensions.md and
        # project_backlog_pseudonymization_plugin_surface.md).
        # ``emit()`` runs every transformer in registration order
        # before publishing to the EventBus and forwarding to the
        # client transport.  Both internal subscribers (plugins,
        # activity detector) and external clients (IPC/WS) see the
        # same (transformed) view — the chokepoint is intentionally
        # one stage upstream of both delivery paths so the canonical
        # view is the transformed one.  Empty list = no-op.
        self._outbound_event_transformers: List[
            Callable[[Event], Event]
        ] = []
        self._workspace_path = workspace_path
        self._config_root: Optional[str] = None
        self._session_id = session_id

        # Core components
        # Phase 3 §7c step 6.6.4.5e: ``self._jaato`` field removed.
        # Every dependency redirected to either ``self._runtime`` (5a,
        # 5d) or ``self._runner_rpc`` (5b, 5c.1-5c.5).  Seat-flip
        # complete — daemon no longer constructs JaatoClient.
        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Optional[PermissionPlugin] = None
        self.todo_plugin: Optional[TodoPlugin] = None
        self.ledger = TokenLedger()

        # AppArmor pre-init confine-context factory (server 0.6.50+).
        # Stashed by ``set_pre_init_confine_context`` from the WS
        # pre-initialize hook; propagated onto :class:`JaatoRuntime`
        # during ``initialize()`` so sessions created on this runtime
        # can wrap their dynamic-instructions expansion in the
        # session's confinement.  None = no confinement applies.
        #
        # Phase 2 (confined runner): no caller installs this any more
        # — the daemon-side per-thread confinement was removed in
        # task 2.1, and Phase 3 will absorb the prefetch confinement
        # into the runner.  The slot stays in place for any in-tree
        # consumer still reading it; Phase 6 cleanup deletes both.
        self._pre_init_confine_context_factory: Optional[Callable] = None

        # Phase 2 confined runner: per-session RPC client to the
        # spawned runner subprocess (see server.runner_spawner +
        # server.runner_rpc_client).  Set by the IPC apparmor pre-init hook
        # AFTER ``RunnerSpawner.spawn`` returns and BEFORE
        # ``initialize()`` runs, so plugins discovering the registry
        # via ``set_plugin_registry`` see ``registry.runner_rpc`` at
        # configure time.  ``None`` for sessions that don't get a
        # runner (Phase 2: only apparmor-enabled IPC sessions spawn
        # one; Phase 3 makes it always-runner across all four session
        # bootstrap paths — see plan §"Non-IPC bootstrap path
        # deferral").
        #
        # Phase 3 §7b.1 audit (see
        # docs/design/per_session_confined_runner_phase3_3c_rpc_surface.md
        # §10 "audit appendix"): every ``self._jaato.X`` site reachable
        # from inside ``initialize()`` or anywhere it calls into has
        # ``self._runner_rpc`` available — set_runner_rpc fires from
        # within ``runner_spawn.spawn_session_runner`` BEFORE
        # ``server.initialize()`` runs.  ``__init__`` is the only
        # truly-pre-runner site; everything else can dispatch to the
        # runner.  See appendix for the per-site bucket table
        # (DONE / NOW / DAEMON / INTERNAL / WIRING / §7b.2 / TRUTHINESS).
        self._runner_rpc: Optional["RunnerRPCClient"] = None
        self._spawned_runner: Optional["SpawnedRunner"] = None
        # Signals the runner has finished ``session.bootstrap`` and can service
        # this session's RPCs.  CLEARED when the rpc handle is wired
        # (set_runner_rpc) and SET by ``mark_runner_ready`` after
        # ``dispatch_bootstrap_envelope`` — readiness is bootstrap-complete, NOT
        # rpc-handle-live, because a reused warm pool slot's handle is live the
        # instant it's claimed yet can't service the session until bootstrap
        # finishes.  Both the send path AND the mid-session client-tool push gate
        # on this (attach has no synchronous ready-gate like session.new, and the
        # §7c seat-flip forwards both to the runner).  Cleared on teardown.
        self._runner_ready: threading.Event = threading.Event()
        # Phase 2 cascade-sharing (server 0.6.144+): pool manager
        # reference for the cascade-aware teardown path in shutdown().
        # When the runner was served from the pool AND the cascade
        # session_end RPC succeeds, shutdown returns the slot to the
        # pool instead of closing the transport.  ``None`` for sessions
        # spawned cold (no pool reuse possible).  Set by
        # ``runner_spawn.spawn_session_runner`` alongside the
        # ``SpawnedRunner.pool_slot`` field.
        self._pool_manager_ref: Optional[Any] = None
        # Phase 3 §7c Step 7.1: daemon-side ``client.prompt_operator``
        # handler — relays runner-fired permission ASKs to the
        # connected client via emit(PermissionRequestedEvent) and
        # awaits the client's response via resolve_response (called
        # from JaatoServer.respond_to_permission post-Step-7.3
        # rewire).  Set in :meth:`set_runner_rpc`; torn down in
        # :meth:`shutdown`.
        self._prompt_operator_handler: Optional[
            "PromptOperatorHandler"
        ] = None

        # Clarification relay handler (symmetric with the prompt-operator
        # handler) — relays runner-fired clarification batches to the
        # connected client via emit(ClarificationBatchEvent) and awaits the
        # client's answers via resolve_response (called from
        # :meth:`respond_to_clarification_batch`).  Set in
        # :meth:`set_runner_rpc`; torn down in :meth:`shutdown`.
        self._clarification_relay_handler: Optional[
            "ClarificationRelayHandler"
        ] = None

        # Path E (cycle 6) §7c step 6.6.4.5b race fix: cached
        # context_limit avoids in-band ``session_get_context_limit``
        # RPCs from notification handlers + aspect callbacks that
        # fire DURING the runner's active ``send_message``.  The
        # in-band RPC raced against the runner's processing and
        # timed out (Layer 5 of the post-§7c send-message chain).
        # Cache populated at end of ``initialize()`` (off-band) and
        # invalidated on ``/model`` command.  ``None`` means
        # uninitialized — readers fall back to 0 / payload value.
        self._cached_context_limit: Optional[int] = None

        # Path F (cycle 7) §7c streaming-response chain: cached
        # ServerAgentHooks instance.  Populated in
        # ``_setup_agent_hooks`` during initialize().  Read by the
        # send_message notification demuxer to re-emit runner-side
        # ``tool_call_*`` / ``tool_output`` / ``turn_progress``
        # events through the same daemon-side path the pre-§7c
        # in-process flow used.  Pre-Path-F the runner-side
        # ``_ui_hooks`` was None and these events silently dropped.
        self._agent_hooks: Optional[Any] = None

        # Phase 3 §7c step 4: direct daemon-side reference to the
        # ``JaatoRuntime`` (provider config + auth + plugin registry +
        # ledger).  Populated after ``connect()`` returns; aliased to
        # ``self._jaato._runtime`` during the §7c rollout so all
        # introspection sites can read it without going through the
        # ``self._jaato.get_runtime()`` indirection.
        #
        # Post-step-6 (when ``self._jaato`` is removed) this becomes
        # the sole daemon-side runtime handle.  Per §4.2,
        # ``JaatoRuntime`` stays daemon-side (model_provider plugins
        # are daemon-tier); only ``JaatoSession`` moves runner-side.
        self._runtime: Optional["JaatoRuntime"] = None

        # Phase 3 §3.13: the ``_planned_sandbox_mode`` slot was
        # removed.  Phase 2's IPC apparmor pre-init hook used it as
        # a transitional channel to communicate the planned mode
        # to ``_create_session_impl``'s Session-record assembly.
        # After §3.13's relocation, the apparmor opt-in lookup lives
        # inline in ``SessionManager._provision_ipc_apparmor_and_spawn_runner``
        # which returns the mode directly to ``_bootstrap_session``;
        # the disk-restore path passes its known mode via
        # ``BootstrapEnvelope.sandbox_mode``.  Neither needs a
        # server-side stash anymore.

        # Pricing table — loaded lazily on first use; populates
        # UsageBreakdown.cost_usd on emitted Context/Turn events when
        # the active model is known to the table.  Empty when no
        # .jaato/pricing.json is present, so cost stays None and
        # consumers know "I don't have a number" vs. "free".
        from shared.pricing import PricingTable
        self._pricing: PricingTable = PricingTable.empty()
        self._pricing_loaded: bool = False

        # Agent tracking
        self._agents: Dict[str, AgentState] = {}
        self._selected_agent_id: str = self._main_agent_id

        # Track original inputs for session export
        self._original_inputs: List[Dict] = []

        # Queue for permission/clarification responses
        self._channel_input_queue: queue.Queue[str] = queue.Queue()
        self._waiting_for_channel_input: bool = False
        self._pending_permission_request_id: Optional[str] = None
        # Edited arguments from client-side editing (set before "e" is put in queue)
        self._pending_edited_arguments: Optional[Dict[str, Any]] = None
        self._pending_clarification_request_id: Optional[str] = None
        self._pending_reference_selection_request_id: Optional[str] = None

        # Track which agent is currently executing a tool (for permission/clarification routing)
        self._current_tool_agent_id: str = self._main_agent_id


        # Background model thread
        self._model_thread: Optional[threading.Thread] = None
        self._model_running: bool = False
        # How the most recent turn terminated: "error" when a terminal error
        # (provider exhaustion / NudgeExhausted) ended the turn, else None
        # (natural).  Reset per-turn at model-thread start; read at the
        # SlotSettledEvent emit so a cascade stage-advance reactor can SKIP
        # advancement on an error-terminated session (the recovery path
        # re-spawns it).  See docs/design/agent-error-recovery-event.md.
        self._terminal_reason: Optional[str] = None
        # Texts stashed for the model thread's ``finally`` to turn into the
        # next turn: continuations drained from child/sibling messages, and
        # sends that arrived for an idle SESSION while THIS thread was still
        # unwinding its previous turn.
        #
        # A LIST, not a slot.  #620 added the second writer above and kept the
        # single-slot assignment, so N messages landing in one wind-down window
        # overwrote each other and only the last became a turn -- silent loss,
        # reproduced live at 4 deliveries -> 2 turn inputs.  Before #620 that
        # path went to the runner's ``_message_queue``, which is a real queue;
        # the regression was replacing a queue with a variable.
        self._pending_continuations: List[str] = []
        #: Guards :attr:`_pending_continuations`.  The stash is written from
        #: the RPC client's asyncio READ LOOP (notification dispatch, see
        #: ``runner_rpc_client._read_loop``) and from ``send_message`` on a
        #: caller thread, but READ on the MODEL thread.  A prior comment here
        #: asserted writer and reader "both run on the same model_thread, so
        #: no race condition" -- that is false for every notification-driven
        #: write, and the lost update it allows is silent: the stashed text
        #: simply never becomes a turn.
        self._pending_continuation_lock = threading.Lock()

        # Model info
        self._model_provider: str = ""
        self._model_name: str = ""

        # Auth state
        self._auth_pending: bool = False
        self._auth_plugin_command: Optional[str] = None  # Command name for pending auth plugin

        # Terminal width for formatting (default 80)
        self._terminal_width: int = 80

        # Presentation context describing client display capabilities.
        # Set via set_presentation_context() when the client sends
        # ClientConfigRequest with a presentation dict.
        self._presentation_context: Optional['PresentationContext'] = None

        # Instruction token cache (shared across sessions in daemon mode)
        self._instruction_token_cache = instruction_token_cache

        # Session-specific environment variables (isolated per session)
        # These are loaded from the session's .env file and NOT applied to
        # global os.environ, keeping each session's configuration isolated.
        self._session_env: Dict[str, str] = {}
        # Phase 4 §B: idempotency flag for _resolve_session_env so the
        # SessionManager can call it pre-spawn (giving the runner-fork
        # access to resolved secret URIs via inherited os.environ)
        # without forcing initialize() step 1 to re-do the work.
        self._session_env_resolved: bool = False

        # Formatter pipeline for server-side output formatting
        # Initialized in _setup_formatter_pipeline() after registry is available
        # The pipeline handles buffering internally for streaming
        self._formatter_pipeline = None

    # =========================================================================
    # Workspace Management
    # =========================================================================

    @property
    def session_id(self) -> Optional[str]:
        """Get the session identifier for this server instance.

        Returns ``None`` on a server whose ``__init__`` has not run -- which
        is a real shape: fifteen test modules build ``JaatoServer`` via
        ``__new__`` to exercise one method without standing up a session.

        THE ANNOTATION WAS ALREADY ``Optional[str]`` AND COULD NOT HONOUR IT.
        Without the class-level default below, this could return a ``str`` or
        RAISE ``AttributeError`` -- never ``None``.  The signature promised a
        value the code could not produce, so every caller that trusted it was
        wrong in a way no type checker could see.  Adding a log line that read
        this property turned that latent lie into two failing tests and
        thirteen more waiting.

        This is a declaration, not a fallback: ``_session_id`` is genuinely
        ``Optional`` and ``__init__``'s assignment is now an override of a
        declared default rather than the attribute's only creation.
        """
        return self._session_id

    @property
    def event_bus(self):
        """Get the session's EventBus for cross-agent event delivery.

        Available after ``initialize()`` completes.  Returns ``None``
        if the server is not yet initialized or has no session.

        Phase 3 §7c step 6.2: read directly from ``self._runtime.event_bus``.
        Pre-§7c-step-6.2 the property reached through
        ``self._jaato.get_session()._runtime.event_bus`` — three
        levels of indirection where the runtime is already daemon-
        side per §4.2.  Mirrors the migration in ``_get_event_bus``
        (server/core.py:1004) shipped during §7c step 4 first pass.
        """
        if self._runtime is None:
            return None
        try:
            return self._runtime.event_bus
        except AttributeError:
            return None

    @property
    def workspace_path(self) -> Optional[str]:
        """Get the client's workspace path."""
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path: Optional[str]) -> None:
        """Set the client's workspace path."""
        self._workspace_path = path
        # Propagate to plugins that need workspace awareness
        self._update_plugin_workspace(path)

    @property
    def config_root(self) -> Optional[str]:
        """Get the read-only framework-config root override.

        When ``None`` (the default), the daemon falls back to
        ``<workspace_path>/.jaato/`` for read-only config discovery
        (profiles, agents, prompts, references, completion_schemas,
        instructions, scripts, services).  When set, that workspace-
        anchored search is replaced with this path.  See
        ``shared/config_resolver.py`` for the resolver chain.
        """
        return self._config_root

    @config_root.setter
    def config_root(self, path: Optional[str]) -> None:
        """Set the read-only framework-config root override.

        Mirrors :attr:`workspace_path` propagation: pushes the new
        value through to the registry / runtime / session so plugins
        that consult :func:`shared.config_resolver.resolve_config_search_path`
        pick it up on their next discovery.
        """
        self._config_root = path
        # Propagate to plugins that need config-root awareness; mirrors
        # how _update_plugin_workspace pushes workspace_path through.
        self._update_plugin_config_root(path)

    @property
    def terminal_width(self) -> int:
        """Get the terminal width for formatting."""
        return self._terminal_width

    @terminal_width.setter
    def terminal_width(self, width: int) -> None:
        """Set the terminal width for formatting.

        This affects enrichment notification formatting to properly
        wrap and align text for the terminal.

        Phase 3 §7c step 6.3: daemon-side leg dropped.  The
        runner-side ``session.set_terminal_width`` RPC is now the
        only source of truth for the runner's enrichment-
        notification width.  ``self._terminal_width`` is still
        tracked daemon-side for the formatter-pipeline propagation
        below (daemon-tier formatting concern); daemon-side
        ``JaatoSession`` state stays orphan post-§7b.2.
        """
        self._terminal_width = width
        rpc = self._runner_rpc
        if rpc is not None:
            forwarder = getattr(rpc, "session_set_terminal_width_threadsafe", None)
            if callable(forwarder):
                try:
                    forwarder(width, timeout=2.0)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.debug(
                        "set_terminal_width: runner RPC propagation "
                        "failed (%s)", exc,
                    )
        # Propagate to main formatter pipeline if initialized
        if self._formatter_pipeline:
            self._formatter_pipeline.set_console_width(width)
        # Propagate to all agent-specific pipelines
        for agent in self._agents.values():
            if agent.formatter_pipeline:
                agent.formatter_pipeline.set_console_width(width)

    def set_presentation_context(self, ctx: 'PresentationContext') -> None:
        """Set the presentation context and propagate to session components.

        The context is stored on the runner-side ``JaatoSession``
        so that the model's system prompt can adapt to the
        client's display capabilities (e.g. avoid wide tables on
        narrow mobile screens).  It is *not* propagated to the
        formatter pipeline: the pipeline always emits client-
        agnostic semantic markup (``<j-code>``, ``<j-table>``),
        which every attached client renders natively — so there is
        no shared rendering state for heterogeneous clients to
        fight over.

        Phase 3 §7c step 6.3: daemon-side leg dropped.  The
        runner-side ``session.set_presentation_context`` RPC is
        now the only source of truth for the system-prompt
        display-context block.  ``self._presentation_context`` is
        still tracked daemon-side for the ``terminal_width =
        ctx.content_width`` sync below (daemon-tier formatter
        concern).

        Args:
            ctx: Presentation context from the connected client.
        """
        self._presentation_context = ctx
        # Keep terminal_width in sync (property setter propagates
        # to formatter pipelines daemon-side and forwards
        # ``set_terminal_width`` to the runner via the property's
        # own RPC forward).
        self.terminal_width = ctx.content_width
        rpc = self._runner_rpc
        if rpc is not None:
            forwarder = getattr(rpc, "session_set_presentation_context_threadsafe", None)
            if callable(forwarder):
                try:
                    forwarder(ctx, timeout=2.0)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.debug(
                        "set_presentation_context: runner RPC propagation "
                        "failed (%s)", exc,
                    )

    def set_apparmor_confinement(
        self,
        confine_context: Callable,
    ) -> None:
        """No-op since Phase 3 §7c step 6.2 (kept for back-compat).

        Pre-§7c this method wired a thread-level AppArmor
        confinement context onto the daemon-side ``ToolExecutor``,
        so every tool call ran inside the context manager and
        inherited the confinement via fork+exec for spawned
        subprocesses.  The daemon's executor is dead post-§7b.2
        (tool execution flows through the runner subprocess, which
        is already process-confined via AppArmor at spawn time —
        see ``server/runner_spawn.py`` + ``server/runner/__main__.py``);
        the daemon-side thread-level wiring this method installed
        had no effect post-§7b.2 even when called.

        The method is preserved as a no-op rather than removed
        because external integrations (jaato-premium, etc.) may
        still call it.  No live caller exists in the OSS tree; if
        a caller is re-introduced, ``server/runner_spawner.py``'s
        invariant guard will detect it (see the §3.13 / §4.6
        re-introduction check at runner_spawner.py:210).

        Args:
            confine_context: Ignored.  Kept for ABI stability.
        """
        # Intentionally no-op.  See docstring for rationale.
        if confine_context is not None:
            logger.debug(
                "set_apparmor_confinement called (no-op since §7c step 6.2; "
                "runner subprocess is process-confined at spawn time)"
            )

    def set_pre_init_confine_context(
        self,
        confine_context_factory: Optional[Callable],
    ) -> None:
        """Stash an AppArmor confine-context factory BEFORE
        ``initialize()`` runs (server 0.6.50+).

        Called from the WS pre-initialize hook so that ``configure()``
        — which expands ``{{!py:...}}`` placeholders during
        ``initialize()`` — can wrap the expansion in the session's
        confinement.  Without this, prefetch scripts ran unconfined
        and could bypass deny rules in the AppArmor profile (notably
        ``deny .jaato/** w`` once R3 lands).

        The factory is propagated onto :class:`JaatoRuntime` during
        connect, then onto each :class:`JaatoSession` created on that
        runtime.  Subagent sessions inherit it automatically.

        Distinct from :meth:`set_apparmor_confinement` (post-init):
        - Pre-init: stash factory for configure-time work.
        - Post-init: wire factory onto the executor for tool calls.

        Both refer to the same factory in normal WS flow — the
        pre-init hook calls this; the post-init hook calls
        ``set_apparmor_confinement`` with the same factory.

        Args:
            confine_context_factory: Zero-arg callable returning a
                context manager, or ``None`` to clear.
        """
        self._pre_init_confine_context_factory = confine_context_factory
        # If runtime exists already (rare — pre-init hook usually fires
        # before connect()), propagate immediately.  Otherwise the
        # factory is read from self._pre_init_confine_context_factory
        # during initialize() right before configure_tools.
        #
        # Phase 3 §7c step 4: read directly from ``self._runtime``
        # (populated by the connect() call site) instead of going
        # through ``self._jaato.get_runtime()``.
        if self._runtime is not None:
            self._runtime.set_confine_context_factory(confine_context_factory)

    def set_runtime_limits(
        self,
        attach_callback: Optional[Callable[[], None]],
        limits: Optional[Any] = None,
        event_reader: Optional[Callable[[], Optional[Any]]] = None,
    ) -> None:
        """Install per-session cgroup attach + app-layer caps + event reader.

        Mirrors :meth:`set_apparmor_confinement` on the runtime-limits
        axis: AppArmor controls *what's reachable*, this controls *how
        much can be consumed*.

        Called by the WebSocket server after :class:`CgroupsManager`
        provisions the session's cgroup.  Subprocess-launching plugins
        (cli, interactive_shell) read attach + limits via the executor's
        accessors:

        * ``attach_callback`` is passed as ``Popen(preexec_fn=...)``,
          migrating each forked child into the session's cgroup before
          ``exec``.
        * ``limits`` carries application-layer caps
          (``tool_timeout_seconds``, ``max_output_bytes``) that have no
          cgroup equivalent — plugins apply them at the Python layer.
        * ``event_reader`` is consumed by ``ToolExecutor.execute`` (not
          forwarded to plugins) — snapshots ``cgroup.events`` before /
          after each tool call and injects deltas into the result's
          ``_telemetry`` dict, where the session's tool span picks
          them up as OTel attributes.

        Args:
            attach_callback: Zero-arg callable for ``preexec_fn``, or
                ``None`` when no kernel-enforced limits are configured.
            limits: :class:`shared.runtime_limits.RuntimeLimits` with
                the app-layer caps.  ``None`` means "no app caps either".
            event_reader: Zero-arg callable returning a
                ``cgroup.events`` snapshot dict, or ``None`` when
                cgroups are unavailable.
        """
        # Phase 3 §7c step 6.2: this method is now a no-op kept
        # for back-compat with WS callers (websocket.py:721) that
        # still invoke it on every WS-provisioned session.  The
        # daemon-side ``ToolExecutor`` is dead post-§7b.2 (tool
        # execution flows through the runner subprocess); the
        # runner-side cgroup attach is set via env vars at
        # spawn time (``JAATO_RUNNER_CGROUP_PATH`` etc., see
        # ``server/runner_spawner.py``), and the runner-side
        # executor reads its app-layer ``RuntimeLimits`` from the
        # bootstrap envelope's ``env_overrides`` mechanism +
        # provider config — neither of which travels through this
        # method.  The pre-§7c daemon-side wiring this method
        # installed had no effect post-§7b.2 even when called.
        #
        # Future cleanup: §7d (cgroup attach migration) may
        # introduce a runner-RPC for streaming live-cgroup-events
        # back to daemon for OTel; until then the
        # ``event_reader`` argument simply isn't consumed by
        # anyone post-seat-flip.
        if attach_callback is not None or limits is not None:
            logger.debug(
                "set_runtime_limits called (no-op since §7c step 6.2; "
                "runner subprocess gets cgroup attach + RuntimeLimits at "
                "spawn time, not via this daemon-side method)"
            )

    def set_reference_authorizer(self, authorizer) -> None:
        """Install the per-session AppArmor reference-fragment authorizer.

        Mirrors :meth:`set_apparmor_confinement` for the
        ``selectReferences`` flow: when a confined WS session selects a
        reference whose ``resolved_path`` lies outside the workspace,
        the references plugin calls ``authorizer.authorize(ref_id, path)``
        to write a per-reference AppArmor fragment so the kernel actually
        permits the subsequent ``readFile`` syscall.  Without this,
        application-layer ``sandbox_manager`` would whitelist the path
        but the kernel would still EACCES at ``open()``.

        Called by ``JaatoWSServer`` after AppArmor confinement is
        enabled for the session.  ``None`` is treated as "no authorizer
        available" — the references plugin then operates with the
        in-process allowlist alone.

        Phase 3 §7c step 6.1: ALSO forwards a bool flag
        (``authorizer is not None``) to the runner-side session via
        the new ``session.set_reference_authorizer`` RPC.  The
        Python ``ReferenceAuthorizer`` object itself can't cross
        the RPC (holds a daemon-side ``AppArmorManager`` reference);
        the runner-side references plugin (post-migration) uses the
        existing ``apparmor.add_reference_fragment`` runner→daemon
        RPC to authorize paths, gated on the bool flag.

        Best-effort forwarding: failures log at DEBUG but don't
        block the daemon-side state update — the daemon-side
        references plugin (still active during the §7c rollout
        window) keeps using the Python authorizer object directly.
        """
        # Phase 3 §7c step 6.6.4.5e: daemon-side leg dropped.
        # ``set_reference_authorizer`` is forwarded to the runner via
        # the ``session.set_reference_authorizer`` RPC below; the
        # runner-side references plugin (post-§3.x sub-track migration)
        # uses the existing ``apparmor.add_reference_fragment``
        # runner→daemon RPC to authorize paths.
        rpc = self._runner_rpc
        if rpc is not None:
            forwarder = getattr(
                rpc, "session_set_reference_authorizer_threadsafe", None,
            )
            if callable(forwarder):
                try:
                    forwarder(authorizer is not None, timeout=2.0)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.debug(
                        "set_reference_authorizer: runner RPC propagation "
                        "failed (%s) — daemon-side state still updated",
                        exc,
                    )

    @property
    def auth_pending(self) -> bool:
        """Check if authentication is pending."""
        return self._auth_pending

    @contextlib.contextmanager
    def _in_workspace(self):
        """Context manager to set workspace identity for the current scope.

        Server 0.6.68+: sets ``shared.session_context._workspace_root``
        and ``_config_root`` ContextVars — per-asyncio-task storage that
        is race-free across concurrent sessions.  Pre-0.6.68 this used
        ``os.environ`` mutation, which is process-global and clobbered
        across overlapping sessions (jaato_session_manager_cross_client_workspace_leak).

        ``os.environ`` is still mutated here as a compat fallback for
        third-party code (provider SDKs, user scripts under
        ``.jaato/scripts/``) that may read ``JAATO_WORKSPACE_ROOT``
        directly.  Jaato-side code should use
        ``shared.session_context.get_workspace_root()`` /
        ``get_config_root()`` instead — those read the ContextVar first
        and fall back to ``os.environ`` for daemon-startup callers.

        Does NOT call ``os.chdir()`` — that is process-global and not
        thread-safe.  All workspace-dependent jaato code uses explicit
        absolute paths.
        """
        if not self._workspace_path:
            yield
            return

        from shared.session_context import (
            set_workspace_root, reset_workspace_root,
            set_config_root, reset_config_root,
        )

        # Per-task ContextVar (race-free, asyncio-aware) — the canonical
        # source of truth for jaato-side reads.
        ws_token = set_workspace_root(self._workspace_path)
        cr_token = (
            set_config_root(self._config_root) if self._config_root else None
        )

        # Process-global os.environ — kept for third-party / user-script
        # readers that import ``os`` directly.  The race against
        # concurrent sessions still exists for these readers, but
        # jaato-side code is now race-free via the ContextVar.
        original_workspace_env = os.environ.get("JAATO_WORKSPACE_ROOT")
        original_config_root_env = os.environ.get("JAATO_CONFIG_ROOT")
        try:
            os.environ["JAATO_WORKSPACE_ROOT"] = self._workspace_path
            if self._config_root:
                os.environ["JAATO_CONFIG_ROOT"] = self._config_root
            yield
        finally:
            if original_workspace_env is not None:
                os.environ["JAATO_WORKSPACE_ROOT"] = original_workspace_env
            elif "JAATO_WORKSPACE_ROOT" in os.environ:
                del os.environ["JAATO_WORKSPACE_ROOT"]
            if original_config_root_env is not None:
                os.environ["JAATO_CONFIG_ROOT"] = original_config_root_env
            elif "JAATO_CONFIG_ROOT" in os.environ:
                del os.environ["JAATO_CONFIG_ROOT"]
            reset_workspace_root(ws_token)
            if cr_token is not None:
                reset_config_root(cr_token)

    def _resolve_session_env(self) -> None:
        """Populate ``self._session_env`` from the four session-env sources.

        Precedence, lowest to highest: the workspace ``.env`` file, the
        profile's ``env:`` map, the profile's typed ``trace:`` block, and
        the post-auth ``_env_overrides``.  The typed block outranks the
        stringly-typed map deliberately -- it is the only one of the two
        whose values were validated (issue #775).

        Idempotent — returns immediately on second call.  Designed so
        :class:`SessionManager` can invoke it BEFORE the runner-spawn
        fork in ``_construct_and_initialize_server``, then wrap the
        spawn in :meth:`_with_session_env` so the resolved values reach
        the runner subprocess via inherited ``os.environ``.

        Phase 4 §B fix for the §7c env-propagation gap.  Pre-fix the
        env resolution lived inline at the top of :meth:`initialize`
        step 1, which runs AFTER the spawn (the spawn is wired through
        ``SessionManager._provision_ipc_apparmor_and_spawn_runner``
        which fires before ``server.initialize()``).  As a result,
        workspace `.env` values like
        ``JAATO_OPENROUTER_API_KEY=pass://jaato/openrouter/api-key``
        stayed unresolved in the daemon's ``os.environ`` at fork-time,
        and the runner subprocess inherited the literal `pass://` URI
        without the runtime resolver state that would let it resolve
        the value itself.

        After this method runs daemon-side, ``self._session_env``
        contains the fully-resolved env.  The runner-side
        ``JaatoServer.initialize()`` will call this method again; the
        idempotency flag makes that a cheap no-op (preserves the
        ``self._session_env`` populated by the daemon, which transited
        the envelope via ``env_overrides`` *and* via ``os.environ``
        inheritance).
        """
        if self._session_env_resolved:
            return

        from dotenv import dotenv_values
        from shared.plugins.subagent.config import expand_variables

        # PR #91 retrospective (Shape 3 PR 1 attempt + Y fix):
        # The daemon resolves session env — including workspace .env —
        # because the daemon (unconfined) is the only process with
        # access to the user's password store via ``pass`` /
        # ``vault`` / etc. exec.  The AppArmor-confined runner
        # subprocess correctly cannot exec those tools.
        #
        # PR #91 tried to move workspace .env reading to the runner
        # (Shape 3's "per-workspace state belongs to the per-workspace
        # process" principle), which surfaced this AppArmor
        # constraint: the runner's resolver init exits 126
        # (CalledProcessError on ``pass version``), no scheme
        # registers, ``pass://`` URIs survive as literal strings.
        #
        # Y fix: daemon stays the secret-resolution authority.
        # ``build_session_envelope`` reads ``self._session_env`` (the
        # fully-resolved dict this method produces) and ships it via
        # ``envelope.session_env``.  The runner applies it to
        # ``os.environ`` unchanged — no resolver discovery,
        # no ``pass`` exec, no AppArmor-blocked subprocess.  Same
        # trust posture as pre-PR-91 (cold-spawn fork-inherit):
        # runner ends up with specific session secrets; cannot
        # enumerate the password store.
        raw_session_env = dotenv_values(self.env_file) if self.env_file else {}
        raw_filtered = {k: v for k, v in raw_session_env.items() if v is not None}
        # ``${VAR}`` cross-references within .env resolve against
        # sibling entries; secret URIs (pass://, vault://, awssm://,
        # sops://, keyring://) resolve via the registered
        # SecretResolver.
        self._session_env = expand_variables(raw_filtered, context=raw_filtered)

        # Profile env: block — higher precedence than .env, supports
        # ${VAR} expansion and secret URI resolution.
        if self._profile and self._profile.env:
            expanded_env = expand_variables(self._profile.env)
            self._session_env.update(expanded_env)

        # Typed `trace:` block (issue #775) — outranks both the
        # workspace .env and the profile's own `env:` map, because it is
        # the only one of the three whose values were validated.  The
        # literal ``"1"`` that started this cannot reach here:
        # ``TraceProfileConfig.from_dict`` refused it at profile-parse
        # time as a switch written into a path field.  An author who
        # still reaches for `env:` gets the old stringly-typed
        # behaviour; the env vars stay the lower-precedence default and
        # nothing downstream changed.
        if self._profile and getattr(self._profile, 'trace', None):
            self._session_env.update(self._profile.trace.as_env())

        # Highest precedence — post-auth wizard overrides everything.
        if self._env_overrides:
            self._session_env.update(self._env_overrides)

        # Family IV (PR-217): compute + provision the sibling jdtls state
        # directory, export its path as ``JAATO_JDTLS_STATE_DIR`` so
        # runner-side LSP / .lsp.json can resolve ``-data`` against it.
        # Symmetric with the ``${jdtlsStateRoot}`` apparmor template var
        # registered in ``shared.plugins.subagent.config.expand_variables``
        # — both paths derive from the same algorithm so apparmor
        # grants and the actual jdtls spawn-arg point at the same
        # absolute path.
        #
        # Provisioned (mkdir -p) here, BEFORE apparmor profile
        # composition AND BEFORE runner spawn, so:
        # - apparmor composer's grant pattern resolves to an existing dir
        # - LSP plugin's jdtls spawn at first model use finds the dir
        # - sibling lifecycle is framework-owned (sanity-fail at
        #   bootstrap if write doesn't survive apparmor; catches
        #   missing-grant misconfigurations audibly rather than
        #   silently dropping diagnostics like the pre-Family-IV bug
        #   surfaced by 2026-06-04 smoke).
        if self._workspace_path:
            from shared.plugins.subagent.config import _compute_jdtls_state_root
            jdtls_state_root = _compute_jdtls_state_root(self._workspace_path)
            if jdtls_state_root:
                try:
                    os.makedirs(jdtls_state_root, exist_ok=True)
                    # Sanity probe: confirm the dir is writable.
                    # Catches apparmor-grant misconfigurations at
                    # bootstrap rather than at first diagnostic poll
                    # (which would silently return zero diagnostics —
                    # the failure mode that motivated PR-217).
                    _probe = os.path.join(jdtls_state_root, ".write-probe")
                    with open(_probe, "w") as _f:
                        _f.write("")
                    os.unlink(_probe)
                    self._session_env["JAATO_JDTLS_STATE_DIR"] = jdtls_state_root
                except OSError as exc:  # noqa: BLE001
                    # Fail-loud rather than silently degrade — LSP
                    # diagnostics depend on this dir being writable.
                    logger.error(
                        "Family IV: failed to provision jdtls state root "
                        "%r for session workspace %r: %s.  LSP "
                        "diagnostics for tool-written files WILL NOT "
                        "surface this session; jdtls bootstrap will "
                        "either fail to write -data or silently drop "
                        "publishDiagnostics.  Operator action: verify "
                        "the per-session AppArmor profile grants r/w "
                        "to ${jdtlsStateRoot} (see "
                        "_base_codegen.yaml apparmor_extra_rules).",
                        jdtls_state_root,
                        self._workspace_path,
                        exc,
                    )

        self._session_env_resolved = True

    @contextlib.contextmanager
    def _with_session_env(self):
        """Context manager to apply session environment variables.

        Two mechanisms are used in parallel:

        1. **ContextVar** (``session_context.set_session_env``) — race-free,
           per-context storage that propagates to ``ThreadPoolExecutor``
           workers in Python 3.12+.  Jaato's own code (e.g. the
           ``service_connector`` auth manager) reads this via
           ``get_session_env()``.

        2. **os.environ** — still mutated for third-party code that reads
           the process environment directly (provider SDKs, proxy libs).
           This remains subject to races between concurrent sessions, but
           those affect only external libraries, not jaato's credential
           handling.

        On exit, the ContextVar is cleared and os.environ is restored.
        """
        from shared.session_context import set_session_env, clear_session_env

        # Set the race-free ContextVar first
        set_session_env(self._session_env)

        # Still set os.environ for third-party code
        saved: dict[str, str | None] = {}
        for key, value in self._session_env.items():
            if value is not None:
                saved[key] = os.environ.get(key)  # None if absent
                os.environ[key] = value
        try:
            yield
        finally:
            clear_session_env()
            for key, previous in saved.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

    def create_registry_and_discover(self) -> None:
        """Create the plugin registry + discover plugins, early.

        Server 0.6.131+ (PR-148) structural fix.  The per-session
        AppArmor profile composition runs in
        ``SessionManager._provision_ipc_apparmor_and_spawn_runner``
        BEFORE :meth:`initialize` — but pre-PR-148, the registry was
        created INSIDE :meth:`initialize`.  When the composer at
        :func:`server.apparmor.resolve_plugin_apparmor_rules` ran,
        ``server.registry`` was ``None``, the entire
        ``profile.plugins`` iteration was silently skipped, and ZERO
        plugin-contributed AppArmor rules ever landed in the rendered
        profile — for ANY plugin (``memory``, ``references``,
        ``subagent``, ``prompt_library``, ``service_connector``,
        ``file_edit``).  The bug was latent for plugins whose typical
        writes didn't get exercised under cascade-spawned AppArmor
        confinement; ``file_edit``'s backup writes (PR-144 anchor on
        ``<config_root>/sessions/<id>/backups/``) first surfaced it
        in v126/v128.

        Fix: this method extracts the registry-creation + discover
        steps from :meth:`initialize`'s "load_plugins" stage and
        makes them callable independently from the apparmor
        provisioning path.  :meth:`initialize` is idempotent on
        registry presence — calling it after this method runs is a
        no-op for the registry-setup step.

        Also pre-populates the registry with framework-known values
        (``workspace_path``, ``config_root``, ``session_id``,
        ``agent_name``) so :func:`resolve_plugin_apparmor_rules` can
        query plugins under a properly-configured registry context —
        mirrors what PR-146's pre-init injection does inside
        :meth:`initialize`'s expose_all flow, just earlier.

        Idempotent: safe to call multiple times; subsequent calls
        skip the create + discover steps when the registry is
        already populated.
        """
        if self.registry is not None and self.registry._plugins:
            return  # Already done; idempotent.

        model_name = (
            self._model_name
            or self.get_session_env("MODEL_NAME")
            or os.environ.get("MODEL_NAME")
        )
        if self.registry is None:
            self.registry = PluginRegistry(model_name=model_name)
        if not self.registry._plugins:
            self.registry.discover()

        # Pre-populate registry with framework-known values so the
        # apparmor composer (called next from
        # _provision_ipc_apparmor_and_spawn_runner) can query
        # plugins with the right context AND so the eventual
        # plugin.initialize calls inherit them via PR-146's
        # _augment_plugin_config helper.
        if self._workspace_path:
            self.registry.set_workspace_path(self._workspace_path)
        if self._config_root:
            self.registry.set_config_root(self._config_root)
        if self._session_id:
            self.registry.set_session_id(self._session_id)
        agent_name = getattr(self, "_main_agent_id", None) or "main"
        if agent_name:
            self.registry.set_agent_name(agent_name)

    def get_session_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a session-specific environment variable.

        Only reads from the session's own env dict (populated from its
        .env file). Does NOT fall back to os.environ — env vars are
        per-session exclusively.

        Args:
            key: Environment variable name.
            default: Default value if not found in session env.

        Returns:
            Value from session env, or the default value.
        """
        return self._session_env.get(key) or default

    def get_all_session_env(self) -> Dict[str, str]:
        """Get all session-specific environment variables.

        Returns:
            Copy of session environment dictionary.
        """
        return dict(self._session_env)

    def _update_plugin_workspace(self, path: Optional[str]) -> None:
        """Update workspace-aware plugins with the new workspace path.

        This notifies plugins like LSP, MCP, file_edit, and CLI that need
        to find config files relative to the client's working directory.

        Uses registry.set_workspace_path() which broadcasts to all plugins
        implementing set_workspace_path().
        """
        if not path or not hasattr(self, 'registry') or not self.registry:
            return

        self.registry.set_workspace_path(path)
        logger.debug(f"Broadcast workspace_path to plugins: {path}")

    def _update_plugin_config_root(self, path: Optional[str]) -> None:
        """Update config-root-aware plugins with the new override.

        Mirrors :meth:`_update_plugin_workspace`.  Plugins that perform
        read-only config discovery (profile loader, references config
        loader, completion-schema loader, instructions loader, etc.)
        consult :func:`shared.config_resolver.resolve_config_search_path`
        — passing ``None`` falls back to today's
        ``<workspace>/.jaato/`` chain, while a non-``None`` value
        overrides the workspace tier.  Plugins that don't implement
        ``set_config_root`` are simply skipped (parity with
        ``set_workspace_path``).
        """
        if not hasattr(self, 'registry') or not self.registry:
            return

        self.registry.set_config_root(path)
        logger.debug(f"Broadcast config_root to plugins: {path}")

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _build_usage(
        self,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        cost_usd_override: Optional[float] = None,
        spend_total_tokens: Optional[int] = None,
        spend_cache_read_tokens: Optional[int] = None,
        spend_cache_creation_tokens: Optional[int] = None,
    ) -> "UsageBreakdown":
        """Construct a ``UsageBreakdown`` and populate ``cost_usd``.

        Cost-resolution precedence:
        1. ``cost_usd_override`` — caller-supplied (e.g. ``claude_cli``
           reports ``total_cost_usd`` from the CLI; that's the fiscal
           truth and beats any computed estimate).
        2. ``self._pricing`` table lookup — operator-loaded
           ``.jaato/pricing.json``; computed from rates and counts.
        3. ``None`` — no source knew, consumer must not assume zero.
        """
        # Lazy-load pricing on first call so we don't read the JSON
        # for sessions that never check cost.
        if not self._pricing_loaded:
            from shared.pricing import load_pricing
            self._pricing = load_pricing(self._workspace_path)
            self._pricing_loaded = True

        if cost_usd_override is not None:
            cost: Optional[float] = cost_usd_override
        elif self._model_name and self._pricing.has(self._model_name):
            cost = self._pricing.cost_for_usage(
                self._model_name,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
            )
        else:
            cost = None

        return UsageBreakdown(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            reasoning_tokens=reasoning_tokens,
            thinking_tokens=thinking_tokens,
            cost_usd=cost,
            spend_total_tokens=spend_total_tokens,
            spend_cache_read_tokens=spend_cache_read_tokens,
            spend_cache_creation_tokens=spend_cache_creation_tokens,
        )

    def _schedule_context_limit_fill(self) -> None:
        """Heal ``_cached_context_limit`` WITHOUT blocking, from any thread.

        The miss path in the notification hooks used to call
        ``session_get_context_limit_threadsafe`` inline.  Those hooks run on
        the RPC read loop's thread, where every ``*_threadsafe`` call
        self-deadlocks (the thread blocks on a coroutine only it could run,
        broken only by the 10s timeout).  Worse, the timeout's exception path
        left the cache unhealed, so the ONE cold miss repeated on every
        streaming-progress notification forever -- a 10s loop stall per
        event, diagnosed by #631's watchdog in its first ninety seconds.

        ``run_coroutine_threadsafe`` WITHOUT blocking on the future is safe
        from any thread including the loop's own: it only enqueues.  The
        fill lands the cache for the NEXT notification; the current one
        emits with the limit unknown (0), which is the established
        honest-unknown semantics (#541) rather than a new state.

        Single-flight: a stampede of notifications during one miss schedules
        one fill, not one per event.

        Returns:
            Whether a fill was actually scheduled by THIS call.  The caller
            logs the miss only when it did, so a stampede produces one line
            rather than one per notification.

        IT USED TO SAY NOTHING AT ALL.  Every path out of here was silent --
        three early returns and a bare ``except: pass`` on the heal -- so the
        branch could not be observed from outside the daemon.  A consumer
        verifying #633 could establish that no bad outcome occurred, but not
        that the code under test had run: zero honest-unknown readings looks
        identical whether the heal beat the first notification or the miss
        path never executed.  That is not a limit of external observation, it
        is this function declining to testify.

        Worse, the silent ``except`` hid the state that used to be permanent.
        A heal that keeps failing leaves the cache cold forever, and the only
        symptom is ``percent_used=0`` on every event with nothing saying why
        -- the same self-sustaining shape #633 fixed, one notch quieter.
        """
        if getattr(self, "_context_limit_fill_inflight", False):
            return False            # a fill is already on its way; not a miss to report
        rpc = self._runner_rpc
        if rpc is None:
            # Expected during bootstrap and the ONLY reason the cache is cold
            # on a healthy session -- but it is also indistinguishable from a
            # runner that died, so it says which it is rather than nothing.
            logger.info(
                "CONTEXT_LIMIT_HEAL_SKIPPED session=%s reason=no_runner_rpc "
                "-- emitting with the limit unknown until the runner is wired",
                self.session_id,
            )
            return False
        loop = getattr(rpc, "_loop", None)
        if loop is None or not loop.is_running():
            logger.warning(
                "CONTEXT_LIMIT_HEAL_SKIPPED session=%s reason=no_running_loop "
                "-- the cache stays cold and percent_used stays 0 until it is "
                "healed by another path",
                self.session_id,
            )
            return False
        self._context_limit_fill_inflight = True

        async def _fill() -> None:
            try:
                limit = await rpc.session_get_context_limit(timeout=5.0)
                if limit:
                    self._cached_context_limit = int(limit)
                    logger.info(
                        "CONTEXT_LIMIT_HEALED session=%s limit=%s "
                        "source=off_band_fill",
                        self.session_id, limit,
                    )
                else:
                    # NOT a failure: a provider that reports 0 is honestly
                    # saying it does not know (#541), and caching that would
                    # turn an honest unknown into a wrong denominator.
                    logger.info(
                        "CONTEXT_LIMIT_HEAL_EMPTY session=%s -- the provider "
                        "reports no context window; percent_used stays 0 by "
                        "design, not from a stale cache",
                        self.session_id,
                    )
            except Exception as exc:  # noqa: BLE001 — best-effort heal
                # Local import, matching this module's existing style at the
                # offer_message boundary; core.py has no module-level import.
                from shared.utils.errors import exc_message
                # WARNING, and it names the exception TYPE: this is the path
                # that leaves the cache cold, and it used to ``pass``.
                # ``exc_message`` because str(TimeoutError()) is the empty
                # string.
                logger.warning(
                    "CONTEXT_LIMIT_HEAL_FAILED session=%s (%s: %s) -- the "
                    "cache stays cold, so the next notification misses again "
                    "and reschedules",
                    self.session_id, type(exc).__name__, exc_message(exc),
                )
            finally:
                self._context_limit_fill_inflight = False

        import asyncio
        asyncio.run_coroutine_threadsafe(_fill(), loop)
        return True

    def emit(self, event: Event) -> None:
        """Emit an event to all subscribed clients and to the EventBus.

        Mapped events are published to the EventBus first, then forwarded
        to clients via the callback. Unmapped events (init, error, help,
        session list, etc.) go directly to clients without touching the bus.

        When outbound transformers are registered (via
        :meth:`register_outbound_event_transformer`), the chain runs
        before both bus publish and client emission so the canonical
        outbound view is consistently transformed.  Empty chain = no-op,
        identical to pre-transformer behaviour.
        """
        # Apply outbound transformer chain (seat 3 of the four-seat
        # pseudonymization design).  Runs once at the top so both the
        # bus and the client transport see the same transformed view.
        for fn in self._outbound_event_transformers:
            event = fn(event)

        # Publish to EventBus for internal subscribers (plugins, activity detector)
        bus = self._get_event_bus()
        if bus:
            bus_event = _server_event_to_bus_event(event)
            if bus_event:
                bus.publish(bus_event)

        # Forward to clients (IPC/WebSocket)
        self._on_event(event)

    def register_outbound_event_transformer(
        self, fn: Callable[[Event], Event]
    ) -> None:
        """Register a transformer for every outbound Event.

        Plug-in surface for redaction / un-redaction / audit /
        content-filter consumers that need to inspect or rewrite events
        before they leave the daemon (toward IPC/WS clients) or hit
        the EventBus.  Multiple transformers stack — registered in
        order, applied as a chain.

        The transformer must return an Event of the same type
        (returning ``None`` would silently drop the event for both
        internal subscribers and external clients — V1 doesn't support
        that semantics).  For premium's pseudonymization use case the
        canonical purpose is un-redaction on the user-display path,
        but the API is generic.

        Premium typically calls this from a session hook
        (:meth:`SessionManager.add_session_hook`) so the transformer is
        wired before any event is emitted.
        """
        self._outbound_event_transformers.append(fn)

    def _get_event_bus(self):
        """Get the EventBus from the runtime, if available.

        Returns None during early init before the runtime is created,
        or if no JaatoClient is connected yet.

        Phase 3 §7c step 4: read directly from ``self._runtime``
        (set by the connect() site) instead of
        ``self._jaato.get_runtime()``.  ``is_connected`` check is
        preserved via the runtime's own state.
        """
        if self._runtime is not None and self._runtime.is_connected:
            return self._runtime.event_bus
        return None

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for clients."""
        self._on_event = callback

    def set_auth_complete_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when authentication completes.

        This is called when a session that was in auth-pending state
        successfully completes authentication.
        """
        self._on_auth_complete = callback

    def emit_current_state(
        self,
        emit_fn: Optional[EventCallback] = None,
        skip_session_info: bool = False,
        clear_stale_pending_requests: bool = False
    ) -> None:
        """Emit current agent state to a specific client or all clients.

        This is useful when a client attaches to an existing session and needs
        to receive the current agent state that was emitted before they connected.

        Args:
            emit_fn: Optional callback to emit to a specific client.
                     If None, uses the default event callback (broadcast).
            skip_session_info: If True, skip emitting SessionInfoEvent (caller will send it).
            clear_stale_pending_requests: If True, emit "resolved" events for permission/
                clarification if no request is pending on the server. This clears stale
                client state after session recovery.
        """
        logger.info(f"emit_current_state called, emit_fn={emit_fn is not None}, agents={list(self._agents.keys())}")
        emit = emit_fn or self._on_event

        # Emit session info with model details (unless caller is sending its own)
        if not skip_session_info:
            logger.info(f"  emitting SessionInfoEvent")
            emit(SessionInfoEvent(
                session_id="",  # Will be set by SessionManager if needed
                session_name="",
                model_provider=self._model_provider,
                model_name=self._model_name,
            ))

        # Emit AgentCreatedEvent for all existing agents (from _agents dict)
        for agent_id, agent in self._agents.items():
            emit(AgentCreatedEvent(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                agent_type=agent.agent_type,
                profile_name=agent.profile_name,
                parent_agent_id=agent.parent_agent_id,
                created_at=agent.created_at,
                session_id=self._session_id or "",
            ))

        # Replay conversation history as output events so reconnecting clients
        # can populate their output panels with the conversation content.
        # This must happen after AgentCreatedEvent (so client buffers exist)
        # and before status events (so tool trees get finalized properly).
        #
        # SKIP for chat-type clients: their conversation is already
        # persistently on-screen (Telegram, Slack, …), so replaying it as
        # OUTPUT events makes the client render the whole history as the answer
        # to the user's NEXT message.  Replay is a redraw concept — it applies
        # only to ephemeral display surfaces (terminal / web).  Gated on the
        # session's presentation ``client_type`` (set via ClientConfigRequest);
        # when the presentation context is unknown, default to replaying (the
        # prior behavior, correct for the TUI).
        from jaato_sdk.events import ClientType
        pres = self._presentation_context
        if pres is None or pres.client_type != ClientType.CHAT:
            self._emit_conversation_replay(emit)
        else:
            logger.info(
                "  skipping conversation replay for chat-type client "
                "(persistent history)"
            )

        # Emit agent status. For idle agents, this triggers stop_spinner() on
        # the client which finalizes any replayed tool trees. For non-idle
        # agents, the client knows tools may still be running.
        for agent_id, agent in self._agents.items():
            emit(AgentStatusChangedEvent(
                agent_id=agent.agent_id,
                status=agent.status,
            ))

        # Emit instruction budget for main agent.
        #
        # Phase 3 §7c step 6.2: read the runner-side session's
        # instruction-budget snapshot via the
        # ``session.snapshot_instruction_budget`` RPC (added in
        # §7c step 6.1 (2/3) at commit 1043bfde) instead of
        # reaching into the daemon-side ``_jaato.get_session()
        # .instruction_budget``.  Returns None when the runner-
        # side budget hasn't been populated yet (pre-configure)
        # — same skip-emit semantics as the pre-§7c
        # ``if session.instruction_budget:`` guard.
        rpc = self._runner_rpc
        if rpc is not None:
            forwarder = getattr(
                rpc, "session_snapshot_instruction_budget_threadsafe", None,
            )
            if callable(forwarder):
                try:
                    snapshot = forwarder(timeout=2.0)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.debug(
                        "emit_current_state: snapshot_instruction_budget "
                        "RPC failed (%s) — skipping budget emit",
                        exc,
                    )
                    snapshot = None
                if snapshot is not None:
                    # ``agent_id`` is a top-level key in the snapshot
                    # dict (see InstructionBudget.snapshot()), so we
                    # don't need a separate read.
                    emit(InstructionBudgetEvent(
                        agent_id=snapshot.get("agent_id", self._main_agent_id),
                        budget_snapshot=snapshot,
                    ))

        # Emit restored subagent state from SubagentPlugin
        # This handles subagents that were restored from persistence but not yet
        # tracked in _agents (since they're managed by SubagentPlugin._active_sessions)
        self._emit_subagent_state(emit)

        # Emit tool ID registry so clients can resolve hash IDs
        self._emit_tool_id_registry_from_schemas(emit_fn=emit)

        # Clear stale pending requests on client if requested
        # This is used after session recovery when the server has no pending requests
        # but the client might still have stale UI state from before the crash
        if clear_stale_pending_requests:
            self._emit_clear_stale_requests(emit)

    def _emit_subagent_state(self, emit: EventCallback) -> None:
        """Emit state for subagents from SubagentPlugin._active_sessions.

        This is called by emit_current_state() to ensure reconnecting clients
        see all active subagents, including those restored from persistence.

        Args:
            emit: Event callback to use for emission.
        """
        if not self.registry:
            return

        subagent_plugin = self.registry.get_plugin("subagent")
        if not subagent_plugin or not hasattr(subagent_plugin, '_active_sessions'):
            return

        from datetime import datetime, timezone

        for agent_id, info in subagent_plugin._active_sessions.items():
            # Skip if already emitted via _agents dict
            if agent_id in self._agents:
                continue

            profile = info.get('profile')
            created_at = info.get('created_at')
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()

            display_name = info.get('display_name') or (profile.name if profile else agent_id)
            emit(AgentCreatedEvent(
                agent_id=agent_id,
                agent_name=display_name,
                agent_type="subagent",
                profile_name=profile.name if profile else "",
                parent_agent_id=self._main_agent_id,
                created_at=created_at,
                session_id=self._session_id or "",
            ))

            # Emit context update for the subagent
            session = info.get('session')
            if session:
                usage = session.get_context_usage()
                context_limit = session.get_context_limit()
                emit(ContextUpdatedEvent(
                    agent_id=agent_id,
                    usage=self._build_usage(
                        prompt_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('output_tokens', 0),
                        total_tokens=usage.get('total_tokens', 0),
                    ),
                    context_limit=context_limit,
                    percent_used=usage.get('percent_used', 0.0),
                    tokens_remaining=max(0, context_limit - usage.get('total_tokens', 0)),
                    turns=usage.get('turns', 0),
                ))

                # Emit instruction budget for the subagent
                if hasattr(session, 'instruction_budget') and session.instruction_budget:
                    emit(InstructionBudgetEvent(
                        agent_id=agent_id,
                        budget_snapshot=session.instruction_budget.snapshot(),
                    ))

    def _emit_conversation_replay(self, emit: EventCallback) -> None:
        """Replay conversation history as output events for reconnecting clients.

        Iterates over stored conversation history for each agent and emits
        AgentOutputEvent, ToolCallStartEvent, and ToolCallEndEvent events so
        the client's output buffer gets populated with the full conversation
        content from before the reconnect.

        The events are emitted in chronological order matching the original
        conversation flow:
        - User messages → AgentOutputEvent(source="user")
        - Model text → AgentOutputEvent(source="model")
        - Model thinking → AgentOutputEvent(source="thinking")
        - Tool calls → ToolCallStartEvent (all) then ToolCallEndEvent (all)
        - Tool response messages are skipped (shown via tool tree)

        Args:
            emit: Event callback to use for emission.
        """
        for agent_id in list(self._agents.keys()):
            # Read via get_history so the MAIN agent's transcript comes from
            # the runner (authoritative post-seat-flip) — the daemon-side
            # agent.history is empty for a cold-restored runner session, which
            # is why reconnecting clients saw a blank panel.
            history = self.get_history(agent_id)
            if not history:
                continue

            logger.info(
                f"  replaying {len(history)} history messages "
                f"for agent {agent_id}"
            )

            for msg in history:
                role = msg.role
                # Compare by value to avoid import dependency on Role enum
                role_value = role.value if hasattr(role, 'value') else str(role)

                if role_value == "user":
                    # Emit user prompt text
                    text = msg.text  # Message.text property concatenates text parts
                    if text:
                        emit(AgentOutputEvent(
                            agent_id=agent_id,
                            source="user",
                            text=text,
                            mode="write",
                        ))

                elif role_value == "model":
                    # Emit text and thinking parts first
                    for part in (msg.parts or []):
                        if part.thought:
                            emit(AgentOutputEvent(
                                agent_id=agent_id,
                                source="thinking",
                                text=part.thought,
                                mode="write",
                            ))
                        elif part.text:
                            emit(AgentOutputEvent(
                                agent_id=agent_id,
                                source="model",
                                text=part.text,
                                mode="write",
                            ))

                    # Emit tool calls as start+end pairs (they're already completed)
                    function_calls = [
                        p.function_call
                        for p in (msg.parts or [])
                        if p.function_call
                    ]
                    if function_calls:
                        # Start all tools first (mirrors parallel execution)
                        for fc in function_calls:
                            emit(ToolCallStartEvent(
                                agent_id=agent_id,
                                tool_name=fc.name,
                                tool_args=fc.args or {},
                                call_id=fc.id,
                            ))
                        # Then complete all tools
                        for fc in function_calls:
                            emit(ToolCallEndEvent(
                                agent_id=agent_id,
                                tool_name=fc.name,
                                call_id=fc.id,
                                success=True,
                            ))

                # Skip "tool" role messages — their content is shown via tool tree

    def _emit_clear_stale_requests(self, emit: EventCallback) -> None:
        """Emit "resolved" events to clear stale pending requests on clients.

        After session recovery, the client might still have UI state for a pending
        permission or clarification request that no longer exists on the server.
        This method emits resolved events with method="session_restored" to tell
        clients to clear their stale state.

        Args:
            emit: Event callback to use for emission.
        """
        # If no permission request is pending on server, emit a clear event
        # The client will ignore this if it has no pending request
        if not self._pending_permission_request_id:
            emit(PermissionResolvedEvent(
                agent_id=self._main_agent_id,
                request_id="",  # Empty - client clears any pending request
                tool_name="",
                granted=False,
                method="session_restored",  # Special method indicating recovery
            ))
            logger.debug("Emitted PermissionResolvedEvent to clear stale client state")

        # Same for clarification requests
        if not self._pending_clarification_request_id:
            emit(ClarificationResolvedEvent(
                agent_id=self._main_agent_id,
                request_id="",
                tool_name="",
                qa_pairs=[],
            ))
            logger.debug("Emitted ClarificationResolvedEvent to clear stale client state")

        # Same for reference selection requests
        if not self._pending_reference_selection_request_id:
            emit(ReferenceSelectionResolvedEvent(
                agent_id=self._main_agent_id,
                request_id="",
                tool_name="",
                selected_ids=[],
            ))
            logger.debug("Emitted ReferenceSelectionResolvedEvent to clear stale client state")

    # =========================================================================
    # Tool ID Registry
    # =========================================================================

    def _build_tool_id_mappings(self) -> Dict[str, str]:
        """Build the complete tool/category ID → name mapping from current schemas.

        Iterates session tools and the full registry to cover both active
        and deferred tools. Calls ``name_to_id`` eagerly so the reverse map
        is populated as a side effect.

        Phase 3 §7c step 3b: replaced the private ``session._tools``
        read with the public :meth:`JaatoClient.get_tool_schemas`
        accessor (which forwards to
        :meth:`JaatoSession.get_tool_schemas`).
        """
        from shared.tool_id_map import name_to_id
        mappings: Dict[str, str] = {}
        # Phase 3 §7c step 6.6.4.5c.5: route through runner-RPC.
        # Daemon wrapper reconstructs ToolSchema NamedTuples so
        # ``.name`` and ``.category`` attr access works unchanged.
        # ``session_get_tool_schemas_threadsafe`` is a _threadsafe RPC
        # (run_coroutine_threadsafe + future.result()).  Invoked ON the event
        # loop it SELF-DEADLOCKS — the loop blocks on .result() for a coro only
        # the loop can pump (the runner replies in ~1ms but the loop can't
        # deliver the reply) -> 15s timeout.  SAME re-entrancy class as the
        # register-RPC stall.  Gate it OFF-loop only: on-loop emits
        # (emit_current_state / initialize / _register_client_tools) map
        # daemon-tier names from the registry walk below; the OFF-loop re-emits
        # (runner-ready + client-tool _push) invoke this RPC to add runner-tier.
        import asyncio as _asyncio
        try:
            _asyncio.get_running_loop()
            on_loop_thread = True
        except RuntimeError:
            on_loop_thread = False
        if self._runner_rpc is not None and not on_loop_thread:
            try:
                schemas = self._runner_rpc.session_get_tool_schemas_threadsafe()
            except Exception:  # noqa: BLE001 — best-effort registry build
                schemas = []
            for schema in schemas:
                mappings[name_to_id(schema.name)] = schema.name
                if schema.category:
                    mappings[name_to_id(schema.category, prefix="c")] = schema.category
        if self.registry:
            # ALWAYS exclude runner-tier: the daemon-side registry walk must
            # NEVER invoke a runner-tier plugin's get_tool_schemas
            # (prompt_library's filesystem discovery) — a tier violation that,
            # on the event-loop thread, blocked the loop ~15s on re-attach and
            # self-blocked the register-RPC send.  Runner-tier names come from
            # the runner (session_get_tool_schemas, above), reachable only via
            # the OFF-loop re-emits (post-bootstrap runner-ready re-emit in
            # runner_spawn + the client-tool _push re-emit).  On-loop emits map
            # daemon-tier names immediately; off-loop re-emits add runner-tier.
            for schema in self.registry.get_exposed_tool_schemas(
                exclude_runner_tier=True,
            ):
                mappings[name_to_id(schema.name)] = schema.name
                if schema.category:
                    mappings[name_to_id(schema.category, prefix="c")] = schema.category
        return mappings

    def _emit_tool_id_registry_from_schemas(
        self,
        emit_fn: Optional[EventCallback] = None,
    ) -> None:
        """Emit the tool ID registry to clients.

        Called from ``initialize()`` (new sessions), ``emit_current_state()``
        (reconnect/re-attach), the mid-session client-tool push, and the
        post-bootstrap runner-ready re-emit.

        The daemon-side registry walk in ``_build_tool_id_mappings`` ALWAYS
        excludes runner-tier plugins, so this never runs a runner-tier plugin's
        filesystem discovery (``prompt_library``) on the event-loop thread —
        the re-attach self-block.  Runner-tier names come from the runner
        (``session_get_tool_schemas``), which only runs off-loop, so they are
        supplied by the OFF-loop callers: the post-bootstrap runner-ready
        re-emit (``runner_spawn.dispatch_bootstrap_envelope``) and the
        client-tool ``_push`` re-emit.  On-loop callers map daemon-tier names
        immediately; the off-loop re-emits add runner-tier.
        """
        from jaato_sdk.events import ToolIdRegistryEvent
        mappings = self._build_tool_id_mappings()
        if mappings:
            emit = emit_fn or self._on_event
            emit(ToolIdRegistryEvent(mappings=mappings))

    # =========================================================================
    # Initialization
    # =========================================================================

    def _emit_init_progress(
        self,
        step: str,
        status: str,
        step_number: int,
        total_steps: int,
        message: str = ""
    ) -> None:
        """Emit an initialization progress event."""
        self.emit(InitProgressEvent(
            step=step,
            status=status,
            step_number=step_number,
            total_steps=total_steps,
            message=message,
        ))

    def initialize(self) -> bool:
        """Initialize the server.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        self._bootstrap_timer = BootstrapTimer()
        total_steps = 6

        # Step 1: Load configuration
        self._emit_init_progress("Loading configuration", "running", 1, total_steps)

        _timer = self._bootstrap_timer

        # Read session's env file into session-specific storage (NOT global os.environ)
        # This keeps each session's configuration isolated from other sessions.
        with _timer.stage("load_config") as _s1:
            # Phase 4 §B: env resolution hoisted into _resolve_session_env
            # so SessionManager can call it pre-spawn (giving the runner
            # subprocess access to resolved secret URIs via inherited
            # os.environ).  The method is idempotent — daemon-side
            # pre-spawn call populates self._session_env; this call is a
            # no-op when reached for the second time.  When initialize()
            # runs without a pre-spawn call (test paths, headless flows
            # bypassing SessionManager), this is the first call and does
            # the actual resolution work.
            with _s1.sub("dotenv_values"):
                self._resolve_session_env()

            def get_config(key: str) -> Optional[str]:
                """Get config value from session env only (no os.environ fallback)."""
                return self._session_env.get(key)

            with _s1.sub("ssl_cert_bundle"):
                active_bundle = active_cert_bundle(verbose=False)

            model_name = get_config("MODEL_NAME")

        try:
            if not model_name and not _profile_binds_a_model(self._profile):
                self.emit(ErrorEvent(
                    error=(
                        "No model bound: set MODEL_NAME, or give the profile a "
                        "'model', or declare 'model_tiers' with an initial tier"
                    ),
                    error_type="ConfigurationError",
                    recoverable=False,
                ))
                return False

            # Get provider from session env (takes precedence over constructor arg)
            session_provider = get_config("JAATO_PROVIDER")
            provider_to_use = session_provider or self._provider

            # Apply agent profile overrides for model and provider.
            # Use the SAME binder the gate above used: reading
            # ``profile.model`` alone left ``self._model_name`` None for a
            # tiers-only profile, so ``SessionInfoEvent(model_name=None)``
            # failed pydantic validation inside _create_session_impl and the
            # caller saw a dropped IPC connection -- the third time this
            # mismatch surfaced as "spawn refused".
            if self._profile:
                _bound = bound_model_for_profile(self._profile)
                if _bound:
                    model_name = _bound
                if self._profile.provider:
                    provider_to_use = self._profile.provider

            # Get provider-specific settings (may be None for non-Google providers)
            project_id = get_config("PROJECT_ID")
            location = get_config("LOCATION")
            self._emit_init_progress("Loading configuration", "done", 1, total_steps)

            # Steps 2 & 3 run in parallel: connecting to the model provider
            # and discovering/loading plugins are independent operations.
            # Running them concurrently saves ~100-200ms during bootstrap.
            from concurrent.futures import ThreadPoolExecutor

            _connect_error: Optional[Exception] = None
            _plugins_error: Optional[Exception] = None

            def _run_connect_provider() -> None:
                """Stage 2: Construct JaatoRuntime and connect to provider.

                Phase 3 §7c step 6.6.4.5e: dropped the transitional
                ``JaatoClient(...) + .connect()`` calls from this stage.
                Daemon now constructs ``JaatoRuntime`` directly.  Keeps
                the executor-thread scope (~100-200ms concurrent with
                plugin load).
                """
                nonlocal _connect_error
                try:
                    self._emit_init_progress(
                        "Connecting to model provider", "running", 2, total_steps
                    )
                    with _timer.stage("connect_provider") as _s2:
                        with self._with_session_env():
                            with _s2.sub("create_runtime"):
                                from pathlib import Path as _Path
                                _ws = (
                                    _Path(self._workspace_path)
                                    if self._workspace_path else None
                                )
                                self._runtime = JaatoRuntime(
                                    provider_name=provider_to_use,
                                    workspace_path=_ws,
                                    config_root=self._config_root,
                                    instruction_token_cache=self._instruction_token_cache,
                                )
                            with _s2.sub("runtime_connect"):
                                self._runtime.connect(project_id, location)
                            # Propagate the pre-init AppArmor confine-context
                            # factory onto the runtime now that it exists.
                            # Server 0.6.50+; sessions created on this
                            # runtime read it during configure() to wrap
                            # dynamic-instructions expansion.
                            if (
                                self._pre_init_confine_context_factory
                                is not None
                                and self._runtime is not None
                            ):
                                self._runtime.set_confine_context_factory(
                                    self._pre_init_confine_context_factory,
                                )
                except Exception as e:
                    _connect_error = e

            def _run_load_plugins() -> None:
                """Stage 3: Discover and configure plugins."""
                nonlocal _plugins_error
                try:
                    self._emit_init_progress(
                        "Loading plugins", "running", 3, total_steps
                    )
                    with _timer.stage("load_plugins") as _s3:
                        # Server 0.6.131+ (PR-148): registry + discover
                        # may have run earlier via
                        # :meth:`create_registry_and_discover` (called
                        # by ``SessionManager._construct_and_initialize_server``
                        # BEFORE ``_provision_ipc_apparmor_and_spawn_runner``
                        # so the apparmor composer can query plugins
                        # for their ``get_apparmor_rules`` contributions).
                        # Both steps are idempotent — re-entering this
                        # block when registry already exists is a no-op.
                        with _s3.sub("create_registry"):
                            if self.registry is None:
                                self.registry = PluginRegistry(model_name=model_name)
                        with _s3.sub("discover"):
                            if not self.registry._plugins:
                                self.registry.discover()

                        plugin_configs = {
                            "todo": {
                                "reporter_type": "memory",
                                "storage_type": "memory",
                            },
                            "references": {
                                "channel_type": "queue",
                                "workspace_path": self._workspace_path,
                            },
                            "clarification": {
                                "channel_type": "queue",
                            },
                            "lsp": {
                                "workspace_path": self._workspace_path,
                                "session_id": self._session_id,
                            },
                            "mcp": {
                                "workspace_path": self._workspace_path,
                                "session_id": self._session_id,
                            },
                            # file_edit needs workspace_path + config_root
                            # at init time so the BackupManager anchors on
                            # the correct location BEFORE any updateFile
                            # fires.  Pre-PR-145 these came only from the
                            # post-init ``set_workspace_path`` / ``set_config_root``
                            # broadcasts (line 1781-1797 below), which
                            # triggered the WARN at every session bootstrap
                            # — the broadcasts did re-init the backup
                            # manager correctly, but the init-time WARN was
                            # cosmetic noise.  Threading these here mirrors
                            # the pattern used by references / lsp / mcp /
                            # auth plugins above.
                            "file_edit": {
                                "session_id": self._session_id,
                                "workspace_path": self._workspace_path,
                                "config_root": self._config_root,
                            },
                            "waypoint": {
                                "session_id": self._session_id,
                            },
                            "sandbox_manager": {
                                "session_id": self._session_id,
                            },
                            # Auth plugins need workspace_path to store
                            # credentials in the session workspace
                            "anthropic_auth": {
                                "workspace_path": self._workspace_path,
                            },
                            "github_auth": {
                                "workspace_path": self._workspace_path,
                            },
                            "zhipuai_auth": {
                                "workspace_path": self._workspace_path,
                            },
                            "antigravity_auth": {
                                "workspace_path": self._workspace_path,
                            },
                            "nim_auth": {
                                "workspace_path": self._workspace_path,
                            },
                        }
                        if self._profile and self._profile.plugin_configs:
                            profile_sandbox_config = self._profile.plugin_configs.get(
                                "sandbox_manager"
                            )
                            if profile_sandbox_config:
                                plugin_configs["sandbox_manager"].update(
                                    profile_sandbox_config
                                )

                        def _on_plugin_progress(plugin_name: str) -> None:
                            self._emit_init_progress(
                                "Loading plugins",
                                "running",
                                3,
                                total_steps,
                                message=plugin_name,
                            )

                        # Server 0.6.129+ structural fix: register
                        # framework-known values on the registry
                        # BEFORE ``expose_all`` fires, so the registry's
                        # ``_augment_plugin_config`` helper injects them
                        # into each plugin's config at init time.  Pre-fix
                        # these were threaded per-plugin into the
                        # ``plugin_configs`` dict above AND broadcast
                        # post-init via ``set_workspace_path`` /
                        # ``set_config_root`` — the "12 sites per
                        # framework concept" pattern.  The single-layer
                        # injection collapses the class.  See
                        # ``shared/plugins/registry.py:_augment_plugin_config``.
                        if self._workspace_path:
                            self.registry.set_workspace_path(self._workspace_path)
                        if self._config_root:
                            self.registry.set_config_root(self._config_root)
                        if self._session_id:
                            self.registry.set_session_id(self._session_id)
                        agent_name = getattr(self, "_main_agent_id", None) or "main"
                        if agent_name:
                            self.registry.set_agent_name(agent_name)

                        with _s3.sub("expose_all"):
                            self.registry.expose_all(
                                plugin_configs, on_progress=_on_plugin_progress
                            )
                        self.todo_plugin = self.registry.get_plugin("todo")

                        with _s3.sub("set_workspace_path"):
                            # Post-init broadcast — idempotent refresh
                            # given pre-init injection above, BUT still
                            # needed to fire the ``set_workspace_path``
                            # / ``set_config_root`` hooks on plugins
                            # that update derived state (e.g.
                            # ``file_edit._reinit_backup_manager`` per
                            # PR-144).  Also propagates mid-session
                            # workspace changes if the daemon ever
                            # mutates ``self._workspace_path`` post-init.
                            if self._workspace_path:
                                self.registry.set_workspace_path(self._workspace_path)
                            if self._config_root:
                                self.registry.set_config_root(self._config_root)

                        with _s3.sub("permission_init"):
                            self.permission_plugin = PermissionPlugin()
                            permission_init_config: Dict[str, Any] = {
                                "channel_type": "queue",
                                "channel_config": {"use_colors": False},
                                "workspace_path": self._workspace_path,
                                "policy": {
                                    "defaultPolicy": "ask",
                                    "whitelist": {"tools": [], "patterns": []},
                                    "blacklist": {"tools": [], "patterns": []},
                                },
                            }
                            if self._profile and self._profile.plugin_configs:
                                profile_perm_config = (
                                    self._profile.plugin_configs.get("permission")
                                )
                                if profile_perm_config:
                                    permission_init_config.update(profile_perm_config)
                            self.permission_plugin.initialize(permission_init_config)
                except Exception as e:
                    _plugins_error = e

            with ThreadPoolExecutor(max_workers=2) as pool:
                pool.submit(_run_connect_provider)
                pool.submit(_run_load_plugins)
                # ThreadPoolExecutor.__exit__ waits for all futures

            # Check for errors from either stage
            if _connect_error is not None:
                e = _connect_error
                self._emit_init_progress(
                    "Connecting to model provider", "error", 2, total_steps,
                    str(e),
                )
                self.emit(ErrorEvent(
                    error=f"Failed to connect: {e}",
                    error_type=type(e).__name__,
                    recoverable=False,
                ))
                return False

            if _plugins_error is not None:
                e = _plugins_error
                self._emit_init_progress(
                    "Loading plugins", "error", 3, total_steps, str(e),
                )
                self.emit(ErrorEvent(
                    error=f"Failed to load plugins: {e}",
                    error_type=type(e).__name__,
                    recoverable=False,
                ))
                return False

            # Phase 3 §7c step 6.6.4.5d: configure the daemon-direct
            # runtime with the plugins loaded by ``_run_load_plugins``.
            # JaatoClient also configures its own internal runtime via
            # ``_jaato.configure_tools()`` at line 1912 — both runtimes
            # get the same plugin/permission/ledger references (shared
            # object identity, no duplication).  Daemon-side reads on
            # ``self._runtime.registry`` (e.g. session_manager.py:3365's
            # ``runtime.registry.get_plugin("prompt_library")``) work
            # against the daemon-direct runtime post-5d.
            if self._runtime is not None:
                self._runtime.configure_plugins(
                    self.registry,
                    self.permission_plugin,
                    self.ledger,
                )

        except Exception as e:
            self._emit_init_progress("Connecting to model provider", "error", 2, total_steps,
                                     str(e))
            self.emit(ErrorEvent(
                error=f"Failed to connect: {e}",
                error_type=type(e).__name__,
                recoverable=False,
            ))
            return False

        # Phase 3 §7c step 6.5: read directly from the daemon-side
        # ``self._runtime`` instead of through ``self._jaato``.
        # ``model_name`` was redundantly OR'd with ``self._jaato.model_name``
        # — JaatoClient's ``model_name`` property returns
        # ``self._model_name`` set at ``connect()`` time to the same
        # ``model`` arg the daemon passed in, so the OR was always
        # equivalent to just the param.  ``provider_name`` is exposed
        # on JaatoRuntime per §4.2 (model_provider plugins are
        # daemon-tier).
        self._model_name = model_name
        self._model_provider = (
            self._runtime.provider_name if self._runtime is not None else None
        )
        # Phase 3 §7c step 6.3: post-init terminal_width sync goes
        # straight to the runner-side JaatoSession (the only
        # source of truth post-step-6.3).  The runner spawn
        # happens BEFORE this line (the §3.13 inline call in
        # ``_bootstrap_session`` fires from
        # ``_construct_and_initialize_server`` BEFORE
        # ``server.initialize()``), so ``self._runner_rpc`` is
        # already attached when present.  Best-effort: failures
        # log but don't block the daemon-side init.
        rpc = self._runner_rpc
        if rpc is not None:
            forwarder = getattr(
                rpc, "session_set_terminal_width_threadsafe", None,
            )
            if callable(forwarder):
                try:
                    forwarder(self._terminal_width, timeout=2.0)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.debug(
                        "initialize: runner RPC terminal_width "
                        "post-init sync failed (%s)", exc,
                    )
        self._emit_init_progress("Connecting to model provider", "done", 2, total_steps)
        self._emit_init_progress("Loading plugins", "done", 3, total_steps)

        # Set up formatter pipeline for server-side output formatting
        with _timer.stage("formatter_pipeline"):
            self._setup_formatter_pipeline()

        # Step 4: Verify authentication (may trigger interactive login via plugin)
        self._emit_init_progress("Verifying authentication", "running", 4, total_steps)
        self._trace(f"[auth] Starting verify_auth for provider: {self._model_provider}")

        self._auth_pending = False  # Track if auth is still needed
        self._auth_plugin_command = None

        def auth_message(msg: str) -> None:
            """Send auth status messages to the client."""
            self._trace(f"[auth] {msg}")
            self.emit(SystemMessageEvent(message=msg, style="info"))

        try:
            # Use session env context and workspace directory so auth can access
            # session-specific credentials and save tokens to the right location.
            #
            # Profile plugin_configs are forwarded so providers that resolve
            # credentials from profile-level knobs (e.g. LM Studio's optional
            # bearer token under plugin_configs['lmstudio']['api_token'], a
            # custom NIM base_url) see the same view they will see during
            # initialize().  Without this, verify_auth fell back to
            # environment-only resolution and profile-supplied credentials
            # were invisible at verify time.
            with _timer.stage("verify_auth") as _s4:
                with self._with_session_env(), self._in_workspace():
                    profile_plugin_configs = (
                        self._profile.plugin_configs if self._profile else None
                    )
                    # Phase 3 §7c step 6.5: read directly from
                    # ``self._runtime`` instead of through ``self._jaato``.
                    # JaatoClient.verify_auth is a thin forwarder to
                    # ``self._runtime.verify_auth(...)``; runtime is daemon-
                    # tier per §4.2.
                    auth_ok = self._runtime.verify_auth(
                        allow_interactive=True,
                        on_message=auth_message,
                        plugin_configs=profile_plugin_configs,
                    )

            if not auth_ok:
                # Credentials not found - try to use provider-specific auth plugin
                auth_plugin = self._get_auth_plugin_for_provider(self._model_provider)

                if auth_plugin:
                    self._trace(f"[auth] Using {auth_plugin.name} plugin for interactive login")
                    self._auth_pending = True
                    # Store the auth command name for provider-agnostic completion check
                    auth_commands = auth_plugin.get_user_commands()
                    self._auth_plugin_command = auth_commands[0].name if auth_commands else None

                    # Set up output callback for plugin messages
                    def plugin_output(source: str, text: str, mode: str) -> None:
                        self._trace(f"[auth][{source}] {text.rstrip()}")
                        self.emit(SystemMessageEvent(message=text.rstrip(), style="info"))

                    auth_plugin.set_output_callback(plugin_output)

                    # Run the login command in workspace context so tokens are saved
                    # to the correct location. This is a blocking call that waits
                    # for OAuth flow to complete.
                    with self._in_workspace():
                        auth_plugin.execute_user_command(auth_plugin.get_user_commands()[0].name, {"action": "login"})

                    # Check if auth completed during the plugin execution
                    # (the plugin blocks until OAuth flow completes or times out)
                    self._check_auth_completion()

                    # If still pending after check, emit waiting status
                    if self._auth_pending:
                        self._emit_init_progress("Verifying authentication", "pending", 4, total_steps,
                                                 "Waiting for authentication")
                else:
                    self._emit_init_progress("Verifying authentication", "error", 4, total_steps,
                                             "No credentials found")
                    self.emit(ErrorEvent(
                        error="Authentication failed: no credentials found and no auth plugin available",
                        error_type="AuthenticationError",
                        recoverable=False,
                    ))
                    return False

        except Exception as e:
            self._emit_init_progress("Verifying authentication", "error", 4, total_steps, str(e))
            self.emit(ErrorEvent(
                error=f"Authentication failed: {e}",
                error_type=type(e).__name__,
                recoverable=False,
            ))
            return False

        if not self._auth_pending:
            self._trace("[auth] verify_auth completed successfully")
            self._emit_init_progress("Verifying authentication", "done", 4, total_steps)
        else:
            # Auth pending — skip remaining steps until auth completes.
            # Phase 3 §7c step 6.6.4.5e: ``_jaato.configure_plugins_only``
            # call dropped.  The daemon-direct runtime is already
            # configured with plugins via ``self._runtime.configure_plugins``
            # (5d, post-threadpool-join); the JaatoClient session-creation
            # half of ``configure_plugins_only`` is dead-weight post-seat-
            # flip.
            self._trace("[auth] Auth pending — skipping provider session")
            return True

        # Step 5: Configure tools (only if auth is complete)
        # Phase 3 §7c step 6.6.4.5e: ``_jaato.configure_tools(...)`` call
        # dropped.  Daemon-direct runtime is already configured with
        # plugins via ``self._runtime.configure_plugins(...)`` (5d post-
        # threadpool-join); the JaatoClient session-creation half is
        # dead-weight post-seat-flip.  ``DynamicInstructionsError`` from
        # dynamic-instructions expansion now surfaces from the runner-
        # side session bootstrap via its own error-reporting path
        # (the bootstrap RPC).
        self._emit_init_progress("Configuring tools", "running", 5, total_steps)
        with _timer.stage("configure_tools") as _s5:
            with self._with_session_env(), self._in_workspace():
                # Wire formatter pipeline into runtime so output formatters can
                # contribute system instructions (e.g., mermaid rendering hints).
                # Phase 3 §7c step 4: read directly from ``self._runtime``.
                if self._formatter_pipeline and self._runtime is not None:
                    self._runtime.set_formatter_pipeline(self._formatter_pipeline)

                # Agent profile GC takes precedence over file-based GC
                with _s5.sub("gc_config"):
                    gc_result = None
                    if self._profile and self._profile.gc:
                        from shared.plugins.subagent.config import gc_profile_to_plugin_config
                        gc_result = gc_profile_to_plugin_config(self._profile.gc)
                    if not gc_result:
                        gc_result = load_gc_from_file(workspace_root=self._workspace_path)

            gc_threshold = None
            gc_strategy = None
            gc_target_percent = None
            gc_continuous_mode = False
            if gc_result:
                gc_plugin, gc_config = gc_result
                # Phase 3 §7c step 6.6.4.4: ``self._jaato.set_gc_plugin(...)``
                # WIRING deleted.  GC trigger path is now runner-side post-
                # 6.6.4.3b (the daemon-side _session is no longer the live
                # one for the model loop), so propagating the GC plugin to
                # the daemon-side session was dead-weight.  The runner's
                # SessionInitEnvelope already carries the GC plugin spec
                # for runner-side install at bootstrap time.  Daemon-side
                # ``gc_threshold`` / ``gc_strategy`` / ``gc_target_percent``
                # / ``gc_continuous_mode`` reads below stay daemon-tier
                # (they feed AgentState UI fields, not the GC trigger path).
                gc_threshold = gc_config.threshold_percent
                gc_target_percent = gc_config.target_percent
                gc_continuous_mode = gc_config.continuous_mode
                gc_strategy = getattr(gc_plugin, 'name', 'gc')
                if gc_strategy.startswith('gc_'):
                    gc_strategy = gc_strategy[3:]  # Remove 'gc_' prefix

            # Phase 3 §7c step 6.6.4.3b: deleted ``_event_bus_tools._on_subscribed``
            # wiring + ``set_instruction_budget_callback`` wiring.  Both
            # collapse into runner-side notification emissions consumed
            # by the ``_build_send_message_notification_handler`` demuxer
            # via the §7c step 6.6.4.1 NotificationFrame protocol
            # (commit 6e31d375) + §7c step 6.6.4.2 install machinery
            # (commit 973923c6).  The initial-budget snapshot emit
            # below stays daemon-side — it's a one-shot after configure_tools(),
            # not a recurring callback, so notification-frame routing
            # would be overkill.
            with _s5.sub("instruction_budget"):
                # Phase 3 §7c step 6.6.4.5b: read budget snapshot via the
                # ``session.snapshot_instruction_budget`` RPC (added §7c
                # step 6.1) instead of reaching into
                # ``self._jaato.get_session().instruction_budget.snapshot()``.
                # ``agent_id`` is carried in the snapshot dict itself.
                #
                # Phase 3 post-Step-7 regression fix: defensive try/except
                # wrap — the runner-side handler may return ``stage="no_session"``
                # if ``session.bootstrap`` RPC hasn't completed by the
                # time this fires (initialize-time timing race exposed by
                # Step 7's set_runner_rpc changes).  The handler contract
                # explicitly supports this case; the wrapper raises
                # ``RunnerCallError`` on ``ok=False`` envelopes, so the
                # caller must catch and treat as "snapshot not yet
                # available — skip emit" (mirrors the
                # ``emit_current_state`` site at core.py:1177-1185).
                snapshot = None
                if self._runner_rpc is not None:
                    try:
                        snapshot = (
                            self._runner_rpc
                            .session_snapshot_instruction_budget_threadsafe()
                        )
                    except Exception as exc:  # noqa: BLE001 — best-effort
                        logger.debug(
                            "initialize: snapshot_instruction_budget RPC "
                            "failed (%s) — skipping initial budget emit "
                            "(runner-side bootstrap may not be complete)",
                            exc,
                        )
                if snapshot:
                    self.emit(InstructionBudgetEvent(
                        agent_id=snapshot.get("agent_id", "main"),
                        budget_snapshot=snapshot,
                    ))

        self._emit_init_progress("Configuring tools", "done", 5, total_steps)

        # Emit tool ID registry to the client so it can resolve hash IDs
        # from the first turn onwards. Built eagerly from configured schemas.
        self._emit_tool_id_registry_from_schemas()

        # Step 6: Set up session
        self._emit_init_progress("Setting up session", "running", 6, total_steps)
        with _timer.stage("setup_session") as _s6:
            with _s6.sub("session_plugin"):
                self._setup_session_plugin()
            with _s6.sub("agent_hooks"):
                self._setup_agent_hooks()
            with _s6.sub("permission_hooks"):
                self._setup_permission_hooks()
            with _s6.sub("clarification_hooks"):
                self._setup_clarification_hooks()
            with _s6.sub("reference_selection_hooks"):
                self._setup_reference_selection_hooks()
            with _s6.sub("plan_hooks"):
                self._setup_plan_hooks()
            with _s6.sub("queue_channels"):
                self._setup_queue_channels()
            with _s6.sub("create_main_agent"):
                self._create_main_agent()
        # Store GC config in main agent state
        if self._main_agent_id in self._agents and gc_threshold is not None:
            self._agents[self._main_agent_id].gc_threshold = gc_threshold
            self._agents[self._main_agent_id].gc_strategy = gc_strategy
            self._agents[self._main_agent_id].gc_target_percent = gc_target_percent
            self._agents[self._main_agent_id].gc_continuous_mode = gc_continuous_mode

        # Emit initial context update so toolbar shows correct usage at startup
        # This must happen after _create_main_agent() so client has the agent registered.
        # Phase 3 §7c step 6.6.4.5b: route through runner-RPC instead of
        # the daemon-side JaatoClient indirection.
        if self._runner_rpc is not None:
            usage = self._runner_rpc.session_get_context_usage_threadsafe()
            context_limit = (
                usage.get('context_limit')
                or self._runner_rpc.session_get_context_limit_threadsafe()
            )
            # Path E (cycle 6) E.2: cache for in-band aspect callbacks
            # + notification handlers that must NOT call back into the
            # runner during active send_message (race with the
            # runner's own message processing).  Stable across the
            # session lifetime; invalidated on /model command.
            if context_limit:
                self._cached_context_limit = int(context_limit)
            self.emit(ContextUpdatedEvent(
                agent_id=self._main_agent_id,
                usage=self._build_usage(
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0),
                    total_tokens=usage.get('total_tokens', 0),
                ),
                context_limit=context_limit,
                percent_used=usage.get('percent_used', 0.0),
                tokens_remaining=usage.get('tokens_remaining', context_limit),
                turns=usage.get('turns', 0),
            ))
            # GC config is emitted as its own event in v1.0+; see GCConfigEvent.
            self.emit(GCConfigEvent(
                agent_id=self._main_agent_id,
                threshold=gc_threshold,
                strategy=gc_strategy,
                target_percent=gc_target_percent,
                continuous_mode=gc_continuous_mode,
            ))

        self._emit_init_progress("Setting up session", "done", 6, total_steps)

        # Emit initial permission status so clients show the correct policy
        # from the start (profile may have set a non-default policy).
        self.emit_permission_status()

        # Phase 3 §7c step 6.6.4.5c.1: route through runner-RPC.  Best-
        # effort: a transport error here just means no auth-info suffix
        # in the display message; don't propagate the failure.
        try:
            auth_info = (
                self._runner_rpc.session_get_auth_info_threadsafe()
                if self._runner_rpc is not None else ""
            )
        except Exception:  # noqa: BLE001 — display-only, fall back to ""
            auth_info = ""
        auth_suffix = f" ({auth_info})" if auth_info else ""
        self.emit(SystemMessageEvent(
            message=f"Connected to {self._model_provider}/{self._model_name}{auth_suffix}",
            style="info",
        ))

        # Emit bootstrap timing report to session trace log
        _timer.finish()
        if os.environ.get("JAATO_BOOTSTRAP_TIMING", "").lower() in ("1", "true", "yes"):  # env: print a session bootstrap timing report (with per-plugin breakdown) to the trace log
            import io as _io
            _buf = _io.StringIO()
            _timer.report(file=_buf)
            # Append per-plugin breakdown
            plugin_timings = self.registry.get_bootstrap_timings()
            if plugin_timings:
                _buf.write("\n  PER-PLUGIN BREAKDOWN (sorted by total time):\n")
                _buf.write("  " + "-" * 68 + "\n")
                sorted_plugins = sorted(
                    plugin_timings.items(),
                    key=lambda x: x[1].get("total_ms", 0),
                    reverse=True,
                )
                for pname, ptiming in sorted_plugins:
                    total = ptiming.get("total_ms", 0)
                    if total < 1.0:
                        continue
                    imp = ptiming.get("import_ms", 0)
                    create = ptiming.get("create_ms", 0)
                    init = ptiming.get("init_ms", 0)
                    _buf.write(
                        f"    {pname:<30} total={total:>7.1f}ms  "
                        f"import={imp:>6.1f}  create={create:>6.1f}  init={init:>7.1f}\n"
                    )
                _buf.write("\n")
            # Route to session trace log (not stderr, which would break the TUI)
            logger.info("Bootstrap timing report:\n%s", _buf.getvalue())
        else:
            # Always log at DEBUG level
            logger.debug("Bootstrap completed in %.0f ms", _timer.total_elapsed * 1000)

        return True

    def _build_profile_session_kwargs(self) -> Optional[Dict[str, Any]]:
        """Build ``create_session()`` kwargs from the agent profile.

        Always returns a dict when ``system_instruction_override`` is set
        (even without a profile), so the override reaches the session
        regardless of whether the caller used ``--profile``.

        Returns:
            Dict of kwargs for ``JaatoRuntime.create_session()``, or None
            if neither a profile nor an override is set.
        """
        kwargs: Dict[str, Any] = {}

        if self._profile:
            from shared.plugins.subagent.config import expand_plugin_configs

            # ALWAYS pass ``profile.plugins`` through — including the
            # empty-list case.  Pre-fix this branch used a falsy check
            # (``if self._profile.plugins:``) that conflated explicit
            # ``plugins: []`` with absent.  When the condition was
            # false, ``tools`` was never set on kwargs, which caused
            # ``JaatoRuntime.create_session`` to fall back to its
            # ``tools=None`` semantic (use ALL exposed plugins from
            # the registry).  Surfaced 2026-06-07 by the vLLM smoke:
            # ``plugins: []`` in the profile produced ~30 tools on the
            # wire (todo, references, memory, notebook, event_bus,
            # clarification, environment, introspection — every
            # registered tool plugin).  Empirical evidence captured
            # via VLLM_WIRE_PROBE in the vLLM provider.
            #
            # Paired with the ``plugins:`` required-key check in
            # ``shared/plugins/subagent/config.py``: profile authors
            # now must explicitly write ``plugins: []`` for the
            # minimal framework set OR list the plugin names they
            # want.  ``profile.plugins`` is therefore guaranteed to
            # be a list (possibly empty) post-validation; the
            # unconditional pass-through here honors that.
            kwargs["tools"] = self._profile.plugins
            if self._profile.preloaded_plugins:
                kwargs["preloaded_plugins"] = self._profile.preloaded_plugins

            if self._profile.system_instructions:
                kwargs["system_instructions"] = self._profile.system_instructions

            if self._profile.plugin_configs:
                expanded = expand_plugin_configs(
                    self._profile.plugin_configs,
                    workspace_root_override=self._workspace_path,
                )
                kwargs["plugin_configs"] = expanded

            if self._profile.provider:
                kwargs["provider_name"] = self._profile.provider

            # Note: ``profile.quirks`` injection lives in
            # ``server/runner_spawn.py:build_session_envelope`` (the
            # LIVE production path) and
            # ``shared/plugins/subagent/plugin.py`` (subagent spawn).
            # Not injected here because this entire method is dead
            # code (referenced only by tests; no production caller —
            # see ``server/runner/session.py:~1016`` for the
            # diagnostic comment + PR #240 history).

            if self._profile.completion_payload_schema is not None:
                kwargs["completion_payload_schema"] = (
                    self._profile.completion_payload_schema
                )

            if self._profile.completion_processors:
                kwargs["completion_processors"] = (
                    self._profile.completion_processors
                )

        # Per-turn model-tier config: profile-declared model_tiers win;
        # otherwise env vars (JAATO_TIER_*) are consulted; otherwise
        # the session stays in single-model mode (no enter_tier tool,
        # no system-prompt augmentation).  Resolved here so the
        # env-var fallback works regardless of whether a profile is
        # set at all.
        from shared.model_tiers import ModelTierConfig
        try:
            tier_config = ModelTierConfig.resolve(
                profile_model_tiers=(
                    self._profile.model_tiers
                    if self._profile and self._profile.model_tiers
                    else None
                ),
            )
        except Exception as exc:
            logger.warning(
                "Tier config rejected for session (falling back to "
                "single-model mode): %s", exc,
            )
            tier_config = None
        if tier_config is not None:
            kwargs["tier_config"] = tier_config

        # Budget control (profile-declared; already parsed + validated at
        # profile load).  Absent => the session runs unbudgeted.
        if self._profile is not None and getattr(
                self._profile, "budget_control", None) is not None:
            kwargs["budget_control"] = self._profile.budget_control

        # Apply the per-session system-instruction knobs last so they
        # win over any profile-supplied system_instructions.  Distinct
        # from None (which means "no override") — the empty string is a
        # legitimate value meaning "send no system message at all".
        if self._system_instruction_override is not None:
            kwargs["system_instruction_override"] = self._system_instruction_override
        if self._suppress_base_instructions:
            # Pass the canonical frozenset through; configure() normalizes it
            # (idempotent) and gates each framework layer accordingly.
            kwargs["suppress_base_instructions"] = self._suppress_base_instructions

        return kwargs or None

    @property
    def profile_name(self) -> Optional[str]:
        """Name of the agent profile used for this session, if any."""
        return self._profile.name if self._profile else None

    @property
    def main_agent_id(self) -> str:
        """Identity of the primary ("main") agent for this session.

        Returns the value of ``--agent <name>`` when one was supplied at
        session creation, otherwise the literal ``"main"``.  Callers
        outside ``JaatoServer`` (notably ``SessionManager``) consult this
        when matching incoming events or restoring per-agent state, so
        their bookkeeping stays in sync with the agent_id the session
        actually emits on the wire.
        """
        return self._main_agent_id

    def _create_main_agent(self) -> None:
        """Create the main agent entry + emit ``AgentCreatedEvent``.

        Path I (cycle 11) Layer 9: emits ``AgentCreatedEvent``
        directly from this daemon-side bootstrap path.  Pre-Path-I
        the event was emitted via ``ServerAgentHooks.on_agent_created``
        when ``set_ui_hooks()`` was called on the daemon-side
        ``JaatoClient`` — a path that became dead at §7c.

        Why the hook-driven path doesn't work post-§7c: the runner-
        side ``JaatoSession._ui_hooks`` is None during bootstrap.
        Path F installed the ``_AgentUIHooksNotificationShim``
        inside ``_handle_session_send_message`` (per-RPC-request
        scope) so the shim only captures callbacks DURING active
        send_message — bootstrap-time ``on_agent_created`` calls
        fire BEFORE the shim is installed and silently drop.

        Without ``AgentCreatedEvent`` the TUI has no agent registry,
        and every subsequent agent-keyed event
        (``ToolCallStartEvent``, ``PermissionRequestedEvent``)
        references an unknown agent → silently discarded.  This
        was the cycle-11 root cause for the persistent TUI-shows-
        nothing symptom.

        The agent_id used here is ``self._main_agent_id`` — either
        the literal ``"main"`` or the ``--agent <name>`` value
        supplied to ``__init__``.  Hook-registered AgentState
        entries use the same id, so the duplicate-creation guard
        below works in both modes.
        """
        logger.debug("  _create_main_agent: creating AgentState...")

        # Check if agent was already created by hooks
        if self._main_agent_id in self._agents:
            logger.debug(
                "  _create_main_agent: %r already exists (created by hooks), skipping",
                self._main_agent_id,
            )
            return

        display_name = (
            self._main_agent_display_name
            or (self._profile.name if self._profile else "Main Agent")
        )
        profile_name = self._profile.name if self._profile else None
        agent = AgentState(
            agent_id=self._main_agent_id,
            name=display_name,
            agent_type="main",
            profile_name=profile_name,
            parent_agent_id=None,
        )
        self._agents[self._main_agent_id] = agent
        self._selected_agent_id = self._main_agent_id
        logger.debug("  _create_main_agent: agent state created")

        # Path I (cycle 11) Layer 9: emit AgentCreatedEvent daemon-
        # side at bootstrap time.  Mirrors the payload that
        # ``ServerAgentHooks.on_agent_created`` (core.py:2711) would
        # have emitted pre-§7c.  Bootstrap-scope event — fires
        # BEFORE the runner's first send_message, so it can't go
        # through the Path F per-RPC-request shim.
        #
        # Dedup interaction (server 0.6.176+): the runner subsequently
        # echoes the agent-creation back via ``on_agent_created`` once
        # it boots.  That hook now skips its own emit when ``agent_id``
        # is ALREADY in ``server._agents`` (set at line 2529 below).
        # Pre-PR-205 both emits carried ``session_id=""`` so the dup
        # was cosmetic-invisible to cascade observers; post-PR-205
        # (session_id populated) the duplicate became visible
        # (double ``↳ session <id>`` lines in cascade_develop.py's
        # walker, kb-side report 2026-06-03).  The bootstrap-side
        # emit here is authoritative for the main agent; the hook-
        # side emit at 2728 remains load-bearing for SUBAGENTS
        # (which are NOT pre-registered in ``_agents`` at hook
        # time — only the main agent is).
        self.emit(AgentCreatedEvent(
            agent_id=self._main_agent_id,
            agent_name=display_name,
            agent_type="main",
            profile_name=profile_name,
            parent_agent_id=None,
            created_at=agent.created_at,
            session_id=self._session_id or "",
        ))

    def _setup_formatter_pipeline(self) -> None:
        """Set up the formatter pipeline for server-side output formatting.

        Uses FormatterRegistry for dynamic formatter discovery and configuration.
        Loads config from .jaato/formatters.json if present, otherwise uses defaults.

        Formatters that need tool plugins (like code_validation_formatter needing
        LSP) will wire themselves automatically via wire_dependencies().
        """
        # Create formatter registry and discover available formatters
        formatter_registry = create_registry()
        formatter_registry.discover()

        # Give formatters access to tool plugins for self-wiring
        if self.registry:
            formatter_registry.set_tool_registry(self.registry)

        # Try to load config from project directory (workspace), an
        # explicit config_root override, or the user directory.  The
        # config_root path takes precedence over the workspace tier so
        # the daemon can read formatters even when the agent's sandbox
        # filesystem doesn't expose ``.jaato/``.
        if self._config_root:
            project_config = os.path.join(self._config_root, "formatters.json")
        elif self._workspace_path:
            project_config = os.path.join(self._workspace_path, ".jaato/formatters.json")
        else:
            project_config = ".jaato/formatters.json"

        config_loaded = (
            formatter_registry.load_config(project_config) or
            formatter_registry.load_config(os.path.expanduser("~/.jaato/formatters.json"))
        )

        if not config_loaded:
            formatter_registry.use_defaults()
            self._trace("Using default formatter configuration")
        else:
            self._trace("Loaded formatter configuration from file")

        # Create pipeline from registry (formatters wire themselves)
        self._formatter_pipeline = formatter_registry.create_pipeline(self._terminal_width)

        # Propagate workspace path so formatters can resolve artifact dirs
        if self._workspace_path:
            self._formatter_pipeline.set_workspace_path(self._workspace_path)

        self._trace(f"Formatter pipeline initialized with {len(self._formatter_pipeline.list_formatters())} formatters")

    def _get_agent_pipeline(self, agent_id: str) -> Optional[Any]:
        """Get the formatter pipeline for a specific agent.

        Each agent has its own formatter pipeline to prevent cross-contamination
        of buffered content when multiple agents are active.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            The agent's formatter pipeline, or None if agent not found.
        """
        if agent_id not in self._agents:
            return None

        agent = self._agents[agent_id]
        if agent.formatter_pipeline is None and self._formatter_pipeline:
            # Lazily create a new pipeline for this agent using the same config
            # We clone the main pipeline's configuration
            from shared.plugins.formatter_pipeline import create_registry
            formatter_registry = create_registry()
            formatter_registry.discover()
            if self.registry:
                formatter_registry.set_tool_registry(self.registry)
            # Use same config loading as main pipeline
            # Use workspace_path if available for project config
            if self._workspace_path:
                project_config = os.path.join(self._workspace_path, ".jaato/formatters.json")
            else:
                project_config = ".jaato/formatters.json"
            config_loaded = (
                formatter_registry.load_config(project_config) or
                formatter_registry.load_config(os.path.expanduser("~/.jaato/formatters.json"))
            )
            if not config_loaded:
                formatter_registry.use_defaults()
            agent.formatter_pipeline = formatter_registry.create_pipeline(self._terminal_width)
            if self._workspace_path:
                agent.formatter_pipeline.set_workspace_path(self._workspace_path)
            self._trace(f"Created formatter pipeline for agent {agent_id}")
        return agent.formatter_pipeline

    def _setup_session_plugin(self) -> None:
        """Set up session persistence plugin.

        Each JaatoServer has its own session plugin instance for tool operations.
        SessionManager has a separate plugin instance for persistence operations.

        Phase 3 §7c step 6.6.4.5e: ``if not self._jaato: return`` guard
        dropped (always-true branch — daemon-direct ``self._runtime``
        is populated synchronously by ``_run_connect_provider``).
        """

        try:
            logger.debug("  _setup_session_plugin: loading session config...")
            session_config = load_session_config()
            logger.debug("  _setup_session_plugin: creating session plugin...")
            session_plugin = create_session_plugin()
            logger.debug("  _setup_session_plugin: initializing session plugin...")
            session_plugin.initialize({'storage_path': session_config.storage_path})
            # Phase 3 §7c step 6.6.4.4: ``self._jaato.set_session_plugin(...)``
            # WIRING deleted.  Propagating the session_plugin to the
            # daemon-side ``_jaato._session`` is dead-weight post-6.6.4.3b
            # — the daemon-side session no longer runs the enrichment
            # pipeline.  The runner's ``SessionInitEnvelope`` carries
            # the session_plugin spec for runner-side install at bootstrap
            # time.
            #
            # Phase 4 §4.4 (Finding 2 closure): the previous deferred-fix
            # comment-block here described the daemon-side
            # ``set_description_callback`` wiring as "wired on the wrong
            # instance — fix planned via NotificationFrame extension".
            # That fix landed.  The session plugin is now runner-tier
            # (§4.4 sub-action A); runner-side install lives in
            # _install_session_notification_callbacks
            # (§4.4 sub-action B); daemon-side emit lives in the
            # ``description_updated`` demuxer branch (§4.4 sub-action C).
            # The daemon-side callback wiring previously here has been
            # deleted as dead code (§4.4 sub-action D).

            # Set session ID on plugin so it knows the current session
            if self._session_id and hasattr(session_plugin, 'set_session_id'):
                session_plugin.set_session_id(self._session_id)
                logger.debug(f"  _setup_session_plugin: session_id set to {self._session_id}")

            if self.registry:
                logger.debug("  _setup_session_plugin: registering session plugin with registry...")
                self.registry.register_plugin(session_plugin, enrichment_only=True)

            if self.permission_plugin and hasattr(session_plugin, 'get_auto_approved_tools'):
                auto_approved = session_plugin.get_auto_approved_tools()
                if auto_approved:
                    logger.debug(f"  _setup_session_plugin: adding {len(auto_approved)} auto-approved tools")
                    self.permission_plugin.add_whitelist_tools(auto_approved)

            logger.debug("  _setup_session_plugin: completed successfully")
        except Exception as e:
            logger.warning(f"  _setup_session_plugin: exception: {e}")
            pass  # Session plugin is optional

    def _setup_agent_hooks(self) -> None:
        """Set up agent lifecycle hooks.

        Phase 3 §7c step 6.6.4.5e: ``if not self._jaato: return`` guard
        dropped (always-true branch post-seat-flip).
        """
        logger.debug("  _setup_agent_hooks: entering...")
        logger.debug("  _setup_agent_hooks: defining ServerAgentHooks class...")
        server = self

        class ServerAgentHooks:
            """Agent hooks that emit events."""

            def on_agent_created(self, agent_id, agent_name, agent_type, profile_name,
                                 parent_agent_id, created_at, **_kwargs):
                # Dedup guard (server 0.6.176+).  ``_create_main_agent``
                # at core.py:2539 emits ``AgentCreatedEvent`` daemon-side
                # at bootstrap (Path I cycle 11 Layer 9 §7c).  When the
                # runner subsequently echoes back the agent-creation via
                # this hook, the agent_id is ALREADY in ``server._agents``
                # because ``_create_main_agent`` (called by the daemon
                # before any runner traffic) inserted it at line 2529.
                # Pre-PR-205 both emits had ``session_id=""`` so the
                # duplicate was invisible to cascade observers (they
                # had no way to correlate the dup pair).  Post-PR-205
                # (session_id populated) the dup became visible to
                # walkers like cascade_develop.py and produced double
                # ``↳ session <id>`` prints per main-agent bootstrap.
                # Subagents created mid-run reach this hook BEFORE any
                # daemon-side bootstrap-emit (the daemon-side path only
                # fires for the main agent), so ``agent_id`` is NOT
                # already in ``_agents`` for subagents — the emit still
                # fires for them (load-bearing).
                was_already_registered = agent_id in server._agents
                agent = AgentState(
                    agent_id=agent_id,
                    name=agent_name,
                    agent_type=agent_type,
                    profile_name=profile_name,
                    parent_agent_id=parent_agent_id,
                )
                if created_at:
                    # Convert datetime to isoformat string if needed
                    if hasattr(created_at, 'isoformat'):
                        agent.created_at = created_at.isoformat()
                    else:
                        agent.created_at = str(created_at)
                server._agents[agent_id] = agent

                if not was_already_registered:
                    server.emit(AgentCreatedEvent(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        agent_type=agent_type,
                        profile_name=profile_name,
                        parent_agent_id=parent_agent_id,
                        created_at=agent.created_at,
                        session_id=server._session_id or "",
                    ))

            def on_agent_output(self, agent_id, source, text, mode):
                server._trace(f"ON_AGENT_OUTPUT agent={agent_id} source={source} len={len(text)} mode={mode}")
                # Get agent-specific formatter pipeline to prevent cross-contamination
                agent_pipeline = server._get_agent_pipeline(agent_id)
                # For model output with streaming formatter pipeline
                if source == "model" and agent_pipeline:
                    # Process chunk through streaming pipeline
                    # Pipeline buffers code blocks, passes through regular text
                    for output in agent_pipeline.process_chunk(text):
                        if output:
                            server.emit(AgentOutputEvent(
                                agent_id=agent_id,
                                source=source,
                                text=output,
                                mode=mode,
                            ))
                else:
                    # Non-model output: strip <hidden>...</hidden> content
                    # These are mid-turn prompts that may contain internal tags
                    filtered_text = re.sub(r'<hidden>.*?</hidden>', '', text, flags=re.DOTALL)

                    # Flush mode: flush the formatter pipeline to emit buffered content
                    # BEFORE tool events, ensuring text appears in correct order
                    if mode == "flush" and agent_pipeline:
                        for output in agent_pipeline.flush():
                            if output:
                                server.emit(AgentOutputEvent(
                                    agent_id=agent_id,
                                    source="model",
                                    text=output,
                                    mode="append",
                                ))

                    # For other modes, only emit if content remains after filtering
                    if filtered_text.strip():
                        server.emit(AgentOutputEvent(
                            agent_id=agent_id,
                            source=source,
                            text=filtered_text,
                            mode=mode,
                        ))

            def on_agent_status_changed(self, agent_id, status, error=None):
                if agent_id in server._agents:
                    server._agents[agent_id].status = status
                server.emit(AgentStatusChangedEvent(
                    agent_id=agent_id,
                    status=status,
                    error=error,
                ))

            def on_agent_completed(self, agent_id, completed_at, success,
                                   token_usage=None, turns_used=None, error="",
                                   payload=None):
                # Convert datetime to isoformat string if needed
                completed_at_str = completed_at
                if completed_at and hasattr(completed_at, 'isoformat'):
                    completed_at_str = completed_at.isoformat()
                elif completed_at:
                    completed_at_str = str(completed_at)

                if agent_id in server._agents:
                    server._agents[agent_id].completed_at = completed_at_str

                server.emit(AgentCompletedEvent(
                    agent_id=agent_id,
                    completed_at=completed_at_str,
                    success=success,
                    token_usage=token_usage,
                    turns_used=turns_used,
                    error=error,
                    payload=payload,
                ))

            def on_agent_error(self, agent_id, error_type, error_summary, *,
                               session_id, request_id=None, attempt="0",
                               classification=None,
                               framework_retries_exhausted=None,
                               occurred_at=None):
                """Emit AgentErrorEvent — the recovery contract.

                Fires from the terminal-error sites AFTER the framework's
                automatic management is exhausted, BEFORE the teardown
                SessionTerminatedEvent.  See
                docs/design/agent-error-recovery-event.md.
                """
                if agent_id in server._agents:
                    server._agents[agent_id].status = "error"
                server.emit(AgentErrorEvent(
                    agent_id=agent_id,
                    session_id=session_id,
                    error_type=error_type,
                    error_summary=error_summary,
                    request_id=request_id,
                    attempt=attempt,
                    classification=classification,
                    framework_retries_exhausted=framework_retries_exhausted,
                    occurred_at=occurred_at,
                ))

            def on_session_quiescent(self, agent_id, reason="natural"):
                """Emit SessionTerminatedEvent after natural completion.

                Called from JaatoSession after the turn that contained
                ``signal_completion`` has fully wrapped up.  By the time
                this fires, ``_is_running`` has gone False, the cancel
                token is cleared, and the session is genuinely safe for
                a client to ``end_session`` / ``delete_session`` /
                ``disconnect`` without racing.
                """
                from jaato_sdk.events import SessionTerminatedEvent
                server.emit(SessionTerminatedEvent(
                    session_id=server.session_id or "",
                    agent_id=agent_id,
                    reason=reason,
                ))

            def on_agent_turn_completed(self, agent_id, turn_number, prompt_tokens,
                                        output_tokens, total_tokens, duration_seconds,
                                        function_calls, cache_read_tokens=None,
                                        cache_creation_tokens=None,
                                        spend_total_tokens=None,
                                        spend_cache_read_tokens=None,
                                        spend_cache_creation_tokens=None,
                                        cost_usd=None,
                                        finish_reason="stop"):
                # Flush any remaining buffered content from the agent's formatter pipeline
                agent_pipeline = server._get_agent_pipeline(agent_id)
                if agent_pipeline:
                    for output in agent_pipeline.flush():
                        if output:
                            server.emit(AgentOutputEvent(
                                agent_id=agent_id,
                                source="model",
                                text=output,
                                mode="append",
                            ))

                    # Collect turn feedback from formatters for auto-continuation
                    agent_pipeline.collect_turn_feedback()
                    feedback = agent_pipeline.get_pending_feedback()
                    if feedback and agent_id in server._agents:
                        server._agents[agent_id].pending_formatter_feedback = feedback

                    # Reset pipeline for next turn
                    agent_pipeline.reset()

                if agent_id in server._agents:
                    turn_entry = {
                        'turn': turn_number,
                        'prompt': prompt_tokens,
                        'output': output_tokens,
                        'total': total_tokens,
                        'duration_seconds': duration_seconds,
                        'function_calls': function_calls,
                    }
                    if cache_read_tokens is not None:
                        turn_entry['cache_read'] = cache_read_tokens
                    if cache_creation_tokens is not None:
                        turn_entry['cache_creation'] = cache_creation_tokens
                    server._agents[agent_id].turn_accounting.append(turn_entry)

                # Read AND CLEAR: the gap describes the turn that just ended,
                # so it must ride exactly one event.  Leaving it set would
                # re-report the same give-up on every later turn of a session
                # that went on to do more work.
                _gap = None
                _agent_rec = server._agents.get(agent_id)
                if _agent_rec is not None:
                    _gap = _agent_rec.completion_gap
                    _agent_rec.completion_gap = None

                server.emit(TurnCompletedEvent(
                    agent_id=agent_id,
                    turn_number=turn_number,
                    completion_gap=_gap,
                    usage=server._build_usage(
                        # THE PROVIDER'S OWN FIGURE, when it reported one.
                        # Without it the event carried None for every provider
                        # that reports a real cost, while the SAME measurement
                        # survived on the telemetry-span path -- two readers of
                        # one number, one of them empty.
                        cost_usd_override=cost_usd,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_creation_tokens=cache_creation_tokens,
                        spend_total_tokens=spend_total_tokens,
                        spend_cache_read_tokens=spend_cache_read_tokens,
                        spend_cache_creation_tokens=(
                            spend_cache_creation_tokens),
                    ),
                    duration_seconds=duration_seconds,
                    function_calls=function_calls,
                    finish_reason=finish_reason,
                ))

            def on_agent_context_updated(self, agent_id, total_tokens, prompt_tokens,
                                         output_tokens, turns, percent_used):
                if agent_id in server._agents:
                    server._agents[agent_id].context_usage = {
                        'total_tokens': total_tokens,
                        'prompt_tokens': prompt_tokens,
                        'output_tokens': output_tokens,
                        'turns': turns,
                        'percent_used': percent_used,
                    }
                # Path E (cycle 6) E.2: read from cache.  Pre-Path-E
                # this site called ``session_get_context_limit_threadsafe``
                # in-band, racing against the runner's active
                # send_message.  Cache populated post-initialize and
                # invalidated on /model command.  Fallback to off-band
                # RPC only if cache miss (uninitialized — first
                # callback before initialize completed).
                context_limit = getattr(server, "_cached_context_limit", None) or 0
                if context_limit == 0:
                    # MISS: emit with the limit unknown (honest-unknown, #541)
                    # and heal the cache OFF-BAND for the next notification.
                    # This callback runs on the RPC read loop's thread, where
                    # a ``*_threadsafe`` round-trip self-deadlocks -- the old
                    # inline fetch here was a 10s loop stall per streaming
                    # notification, forever, because its own timeout kept the
                    # cache from healing.
                    #
                    # Logged only when a fill was actually SCHEDULED, so a
                    # stampede of notifications during one cold period makes
                    # one line rather than one per event -- and so the branch
                    # can be OBSERVED at all.  Without it, zero honest-unknown
                    # readings looks identical whether the heal beat the first
                    # notification or this path never ran, which is exactly the
                    # ambiguity a consumer verifying #633 hit.
                    if server._schedule_context_limit_fill():
                        logger.info(
                            "CONTEXT_LIMIT_MISS session=%s hook=%s -- emitting "
                            "with the limit unknown (0, the #541 semantics) and "
                            "healing off-band for the next notification",
                            server.session_id, "agent_context_updated",
                        )
                # Pull cache tokens from the most recent turn entry so the
                # usage matches Turn{Completed,Progress}Event in expressivity.
                # The protocol callback doesn't carry them, but we have the
                # turn accounting at hand.
                cache_read_tokens = None
                cache_creation_tokens = None
                if agent_id in server._agents:
                    accounting = server._agents[agent_id].turn_accounting
                    if accounting:
                        last_turn = accounting[-1]
                        cache_read_tokens = last_turn.get('cache_read')
                        cache_creation_tokens = last_turn.get('cache_creation')
                server.emit(ContextUpdatedEvent(
                    agent_id=agent_id,
                    usage=server._build_usage(
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_creation_tokens=cache_creation_tokens,
                    ),
                    context_limit=context_limit,
                    percent_used=percent_used,
                    tokens_remaining=max(0, context_limit - total_tokens),
                    turns=turns,
                ))

            def on_agent_gc_config(self, agent_id, threshold, strategy, target_percent=None, continuous_mode=False):
                # Store GC config in agent state
                if agent_id in server._agents:
                    server._agents[agent_id].gc_threshold = threshold
                    server._agents[agent_id].gc_strategy = strategy
                    server._agents[agent_id].gc_target_percent = target_percent
                    server._agents[agent_id].gc_continuous_mode = continuous_mode
                # GC config is its own concern, emitted on its own event.
                # Pre-1.0 versions piggy-backed it onto ContextUpdatedEvent,
                # which conflated context-usage with GC configuration; the
                # split lets clients render a status bar from one event and
                # configure GC from another.
                server.emit(GCConfigEvent(
                    agent_id=agent_id,
                    threshold=threshold,
                    strategy=strategy,
                    target_percent=target_percent,
                    continuous_mode=continuous_mode,
                ))

            def on_agent_history_updated(self, agent_id, history):
                if agent_id in server._agents:
                    server._agents[agent_id].history = history

            def on_tool_call_start(self, agent_id, tool_name, tool_args, call_id=None):
                # Track current agent for permission/clarification routing
                server._current_tool_agent_id = agent_id

                # Flush any buffered model output before starting the tool
                # This ensures model text appears BEFORE the tool tree
                # Use agent-specific pipeline to prevent cross-contamination
                agent_pipeline = server._get_agent_pipeline(agent_id)
                if agent_pipeline:
                    for output in agent_pipeline.flush():
                        if output:
                            server.emit(AgentOutputEvent(
                                agent_id=agent_id,
                                source="model",
                                text=output,
                                mode="append",
                            ))
                    agent_pipeline.reset()

                # Extract "message" or similar intent arguments and emit as model text
                # This shows the model's intent before the tool block, not collapsed in it
                # We keep the arg in tool_args so the tool still receives it
                intent_arg_names = ("message", "summary", "intent", "rationale")
                intent_text = None
                if tool_args:
                    for arg_name in intent_arg_names:
                        if arg_name in tool_args:
                            val = tool_args[arg_name]
                            if val and isinstance(val, str) and val.strip():
                                intent_text = val.strip()
                                break  # Use first found intent arg

                if intent_text:
                    server.emit(AgentOutputEvent(
                        agent_id=agent_id,
                        source="model",
                        text=intent_text,
                        mode="write",
                    ))

                # Enrich tool_args with a TUI-only display name for
                # hashed identifiers (server 0.6.120+).  ``template_id``
                # hashes by design (so the LLM can't pattern-match on the
                # filename — see PR #136); the TUI needs the human name
                # for the user-facing rendering.  Underscore prefix marks
                # the field as UI-metadata, not a real tool argument.
                # The executor never reads ``_template_name_display``.
                #
                # Same pattern would extend to any future hashed-id
                # surface (subagent_id → display name, etc.) — keep the
                # naming convention.
                if tool_args and isinstance(tool_args, dict):
                    template_id = tool_args.get("template_id")
                    if (
                        isinstance(template_id, str)
                        and template_id.startswith("tpl_")
                        and "_template_name_display" not in tool_args
                    ):
                        from shared.tool_id_map import id_to_name as _id_to_name
                        resolved = _id_to_name(template_id)
                        # When the id is unknown to this process,
                        # ``id_to_name`` round-trips the id unchanged —
                        # skip the enrichment in that case so the TUI
                        # falls back to showing the raw id rather than
                        # showing a "name" that's actually the id.
                        if resolved != template_id:
                            tool_args = {
                                **tool_args,
                                "_template_name_display": resolved,
                            }

                server.emit(ToolCallStartEvent(
                    agent_id=agent_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    call_id=call_id,
                ))

            def on_tool_call_end(self, agent_id, tool_name, success, duration_seconds,
                                 error_message=None, call_id=None, backgrounded=False,
                                 continuation_id=None, show_output=None, show_popup=None,
                                 is_error_result=False, result_status=None):
                server.emit(ToolCallEndEvent(
                    agent_id=agent_id,
                    tool_name=tool_name,
                    call_id=call_id,
                    success=success,
                    is_error_result=is_error_result,
                    result_status=result_status,
                    duration_seconds=duration_seconds,
                    error_message=error_message,
                    backgrounded=backgrounded,
                    continuation_id=continuation_id,
                    show_output=show_output,
                    show_popup=show_popup,
                ))

                # After discover_service succeeds, refresh the client's
                # service completion cache so 'services list' reflects the
                # newly discovered service immediately.
                if tool_name == "discover_service" and success:
                    svc_plugin = server._find_plugin_for_command("services")
                    if svc_plugin and hasattr(svc_plugin, 'get_service_metadata'):
                        server.emit(ServiceListEvent(
                            services=svc_plugin.get_service_metadata(),
                        ))

            def on_tool_output(self, agent_id, call_id, chunk):
                # Process tool output through formatter pipeline for syntax highlighting
                # and marker transformation (e.g., <notebook-cell> → <nb-row>)
                # Use agent-specific pipeline to prevent cross-contamination
                agent_pipeline = server._get_agent_pipeline(agent_id)
                if agent_pipeline:
                    formatted_parts = []
                    for output in agent_pipeline.process_chunk(chunk):
                        formatted_parts.append(output)
                    for output in agent_pipeline.flush():
                        formatted_parts.append(output)
                    agent_pipeline.reset()
                    chunk = "".join(formatted_parts)

                server.emit(ToolOutputEvent(
                    agent_id=agent_id,
                    call_id=call_id,
                    chunk=chunk,
                ))

            def on_agent_instruction_budget_updated(self, agent_id, budget_snapshot):
                server.emit(InstructionBudgetEvent(
                    agent_id=agent_id,
                    budget_snapshot=budget_snapshot,
                ))

            def on_turn_progress(self, agent_id, total_tokens, prompt_tokens,
                                 output_tokens, percent_used, pending_tool_calls,
                                 cache_read_tokens=None, cache_creation_tokens=None):
                # Path E (cycle 6) E.2: read from cache.  Same race
                # shape as on_agent_context_updated above — pre-Path-E
                # this site called ``session_get_context_limit_threadsafe``
                # in-band, racing against the runner's active
                # send_message.
                context_limit = getattr(server, "_cached_context_limit", None) or 0
                if context_limit == 0:
                    # MISS: emit with the limit unknown (honest-unknown, #541)
                    # and heal the cache OFF-BAND for the next notification.
                    # This callback runs on the RPC read loop's thread, where
                    # a ``*_threadsafe`` round-trip self-deadlocks -- the old
                    # inline fetch here was a 10s loop stall per streaming
                    # notification, forever, because its own timeout kept the
                    # cache from healing.
                    #
                    # Logged only when a fill was actually SCHEDULED, so a
                    # stampede of notifications during one cold period makes
                    # one line rather than one per event -- and so the branch
                    # can be OBSERVED at all.  Without it, zero honest-unknown
                    # readings looks identical whether the heal beat the first
                    # notification or this path never ran, which is exactly the
                    # ambiguity a consumer verifying #633 hit.
                    if server._schedule_context_limit_fill():
                        logger.info(
                            "CONTEXT_LIMIT_MISS session=%s hook=%s -- emitting "
                            "with the limit unknown (0, the #541 semantics) and "
                            "healing off-band for the next notification",
                            server.session_id, "turn_progress",
                        )
                server.emit(TurnProgressEvent(
                    agent_id=agent_id,
                    usage=server._build_usage(
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_creation_tokens=cache_creation_tokens,
                    ),
                    context_limit=context_limit,
                    percent_used=percent_used,
                    tokens_remaining=max(0, context_limit - total_tokens),
                    pending_tool_calls=pending_tool_calls,
                ))

        logger.debug("  _setup_agent_hooks: class defined, creating instance...")
        hooks = ServerAgentHooks()
        # Path F (cycle 7): cache the hooks instance so the daemon-
        # side notification demuxer (``_build_send_message_notification_handler``)
        # can route runner-emitted ``tool_call_start`` /
        # ``tool_call_end`` / ``tool_output`` / ``turn_progress``
        # frames through the same hooks that pre-§7c fired in-process.
        # Without this cache the demuxer would need to walk
        # ``self.registry.get_plugin("subagent")._ui_hooks`` per call.
        self._agent_hooks = hooks
        # Propagate the resolved agent identity to the JaatoClient/JaatoSession
        # BEFORE set_ui_hooks runs.  ``set_ui_hooks`` reads ``self._agent_id``
        # both to register the AgentState (via ``on_agent_created``) and to
        # forward the id to the session (via ``session.set_ui_hooks``).
        # Setting it here means every downstream consumer — most importantly
        # ``LifecycleTools._execute_signal_completion`` and the reactor that
        # matches on ``AgentCompletedEvent.agent_id`` — sees the agent's
        # logical identity (e.g. ``"coordinator"``) instead of the
        # default ``"main"``.
        #
        # Phase 3 §7c step 6.6.4.5e: ``_jaato.set_agent_identity()`` and
        # ``_jaato.set_ui_hooks()`` calls dropped per the §7c step
        # 6.6.4.5c.0 missing-method audit (commit a88676ca).  Both were
        # daemon-side JaatoClient state mutations with no meaningful
        # runner-side propagation:
        # - ``set_agent_identity`` only mutated JaatoClient._agent_id /
        #   _agent_name; equivalent state already on
        #   ``self._main_agent_id`` / ``self._main_agent_display_name``.
        # - ``set_ui_hooks`` propagated to JaatoSession, but the runner-
        #   side session's ``_ui_hooks`` is None (Audit Finding 3 — see
        #   docs/design/project_backlog_runner_ui_hooks_gap.md).  The
        #   ``AgentUIHooks`` callable object isn't serializable across
        #   the wire anyway.  Daemon-side hooks live on JaatoServer +
        #   subagent_plugin (registered separately below).

        # Register with subagent plugin
        if self.registry:
            logger.debug("  _setup_agent_hooks: getting subagent plugin...")
            subagent_plugin = self.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_ui_hooks'):
                logger.debug("  _setup_agent_hooks: calling subagent.set_ui_hooks...")
                subagent_plugin.set_ui_hooks(hooks)
                logger.debug("  _setup_agent_hooks: subagent.set_ui_hooks done")
        logger.debug("  _setup_agent_hooks: completed")

    def _get_ui_hooks(self) -> Optional[Any]:
        """Return the daemon-side ``ServerAgentHooks`` instance, or
        ``None`` if ``_setup_agent_hooks`` hasn't run yet.

        Path F (cycle 7).  The send_message notification demuxer
        reads this to route ``tool_call_*`` / ``tool_output`` /
        ``turn_progress`` events through the daemon-side hooks,
        re-using their formatting + state-mutation logic without
        duplicating it inside the demuxer.

        Defensive: returns ``None`` rather than raising so a
        notification arriving before hooks are wired (rare — would
        require runner activity pre-initialize) drops cleanly.
        """
        return getattr(self, "_agent_hooks", None)

    def _read_spawn_attempt(self) -> str:
        """Read the reactor-level ``attempt`` from the session's spawn params.

        The kb passes ``agent_params={"attempt": "<n>"}`` at ``create_session``
        (echoed verbatim onto ``AgentErrorEvent.attempt``).  The envelope types
        agent_params as ``Dict[str, str]``; we return the string as-is, falling
        back to ``"0"`` when absent / unreadable.  Never raises.

        This is the REACTOR-level re-spawn count — NOT ``with_retry``'s internal
        per-request attempts (which are framework-only and never surfaced).
        See docs/design/agent-error-recovery-event.md.
        """
        try:
            jaato = getattr(self, "_jaato", None)
            session = jaato.get_session() if jaato is not None else None
            params = getattr(session, "_agent_params", None) or {}
            val = params.get("attempt")
            if isinstance(val, str) and val:
                return val
            if val is not None:
                return str(val)
        except Exception:
            pass
        return "0"

    def _emit_agent_error(
        self,
        *,
        error_type: str,
        error_summary: str,
        request_id: Optional[str] = None,
        classification: Optional[str] = None,
        framework_retries_exhausted: Optional[int] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Emit ``AgentErrorEvent`` for a terminal error, BEFORE teardown.

        The recovery contract (docs/design/agent-error-recovery-event.md):
        called at the terminal-error sites — model-thread terminal, nudge
        exhaustion, bootstrap failure — each of which is, by construction, a
        point where the framework's automatic management (``with_retry`` /
        nudge) is exhausted or never applied.  Routes through the daemon-side
        ``ServerAgentHooks.on_agent_error`` so the event reaches the bus +
        clients.  No-ops (logs) if hooks aren't wired yet.  Never raises into
        the caller's teardown path.

        ``session_id`` / ``agent_id`` default to the server's own state but may
        be overridden (bootstrap-failure path, where the server's session state
        isn't fully wired yet — the caller knows the ids).
        """
        try:
            hooks = self._get_ui_hooks()
            if hooks is None:
                return
            import time as _time
            sid = session_id if session_id is not None else (getattr(self, "session_id", None) or "")
            aid = agent_id if agent_id is not None else (getattr(self, "_main_agent_id", None) or "main")
            hooks.on_agent_error(
                agent_id=aid,
                error_type=error_type,
                error_summary=error_summary,
                session_id=sid,
                request_id=request_id,
                attempt=self._read_spawn_attempt(),
                classification=classification,
                framework_retries_exhausted=framework_retries_exhausted,
                occurred_at=_time.time(),
            )
        except Exception:
            logger.warning("Failed to emit AgentErrorEvent", exc_info=True)

    def _emit_agent_error_from_exc(
        self,
        exc: BaseException,
        *,
        classification: Optional[str] = None,
        framework_retries_exhausted: Optional[int] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Convenience wrapper: emit ``AgentErrorEvent`` from a caught
        exception, deriving ``error_type`` / ``error_summary`` / ``request_id``.
        """
        self._emit_agent_error(
            error_type=type(exc).__name__,
            error_summary=str(exc),
            request_id=_extract_provider_request_id(exc),
            classification=classification,
            framework_retries_exhausted=framework_retries_exhausted,
            session_id=session_id,
            agent_id=agent_id,
        )

    def _emit_error_termination(
        self,
        *,
        error_type: str,
        error_summary: str,
        request_id: Optional[str] = None,
        classification: Optional[str] = None,
        framework_retries_exhausted: Optional[int] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """THE single chokepoint for an error-terminated session.

        Emits, in order: (1) ``AgentErrorEvent`` — the recovery first-refusal,
        and (2) ``SessionTerminatedEvent(reason="error")`` — the teardown
        signal; and stamps ``_terminal_reason="error"`` so the later
        ``SlotSettledEvent`` carries it.

        This makes the invariant **structural, not conventional**: "an
        ``AgentErrorEvent`` precedes EVERY ``SessionTerminatedEvent(reason=
        "error")``" holds because no error-termination path constructs the
        terminal event directly — they all route here, so the recovery offer
        can never be skipped.  Downstream that lets a cascade safely retire the
        redundant ``on session.terminated reason=error`` abort reactor (which
        otherwise races the recovery reactor's ``mark_handled``).  A guard test
        (``test_error_termination_single_chokepoint``) fails the build if any
        ``reason="error"`` terminal emit appears outside this method.  See
        docs/design/agent-error-recovery-event.md.
        """
        from jaato_sdk.events import SessionTerminatedEvent
        sid = session_id if session_id is not None else (getattr(self, "session_id", None) or "")
        aid = agent_id if agent_id is not None else (getattr(self, "_main_agent_id", None) or "main")
        # Stamp before teardown so SlotSettledEvent (emitted in shutdown) carries
        # terminal_reason="error" → stage-advance reactor skips, recovery re-spawns.
        self._terminal_reason = "error"
        # 1. Recovery first refusal (fires only after Layer-1 exhaustion).
        self._emit_agent_error(
            error_type=error_type,
            error_summary=error_summary,
            request_id=request_id,
            classification=classification,
            framework_retries_exhausted=framework_retries_exhausted,
            session_id=session_id,
            agent_id=agent_id,
        )
        # 2. Terminal teardown signal (back-compat; carries the cause).
        self.emit(SessionTerminatedEvent(
            session_id=sid,
            agent_id=aid,
            reason="error",
            error_summary=error_summary,
            error_type=error_type,
        ))

    def _emit_error_termination_from_exc(
        self,
        exc: BaseException,
        *,
        classification: Optional[str] = None,
        framework_retries_exhausted: Optional[int] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Convenience wrapper: :meth:`_emit_error_termination` from a caught
        exception, deriving ``error_type`` / ``error_summary`` / ``request_id``.
        """
        self._emit_error_termination(
            error_type=type(exc).__name__,
            error_summary=str(exc),
            request_id=_extract_provider_request_id(exc),
            classification=classification,
            framework_retries_exhausted=framework_retries_exhausted,
            session_id=session_id,
            agent_id=agent_id,
        )

    def _setup_permission_hooks(self) -> None:
        """Set up permission lifecycle hooks."""
        if not self.permission_plugin:
            return

        server = self

        def on_permission_requested(tool_name: str, request_id: str,
                                    tool_args: dict, response_options: list,
                                    call_id: Optional[str] = None):
            server._pending_permission_request_id = request_id
            server._waiting_for_channel_input = True

            # Convert response options to dicts
            options_dicts = []
            for opt in response_options:
                opt_dict = {
                    "key": getattr(opt, 'short', getattr(opt, 'key', str(opt))),
                    "label": getattr(opt, 'full', getattr(opt, 'label', str(opt))),
                    "action": getattr(opt, 'decision', getattr(opt, 'action', 'unknown')),
                }
                # Convert enum to string if needed
                if hasattr(opt_dict["action"], 'value'):
                    opt_dict["action"] = opt_dict["action"].value
                if hasattr(opt, 'description') and opt.description:
                    opt_dict["description"] = opt.description
                options_dicts.append(opt_dict)

            # Get formatted prompt from permission plugin
            prompt_lines = None
            format_hint = None
            warnings = None
            warning_level = None
            if hasattr(server.permission_plugin, 'get_formatted_prompt'):
                try:
                    result = server.permission_plugin.get_formatted_prompt(
                        tool_name, tool_args or {}, "ipc"
                    )
                    # Handle both old (4-tuple) and new (6-tuple) return formats
                    if len(result) >= 6:
                        prompt_lines, format_hint, language, raw_details, warnings, warning_level = result
                    else:
                        prompt_lines, format_hint, language, raw_details = result

                    # Use agent-specific pipeline to prevent cross-contamination
                    agent_pipeline = server._get_agent_pipeline(server._current_tool_agent_id)
                    if agent_pipeline:
                        # First, flush any buffered model output and emit it separately
                        # This prevents model text from leaking into the permission prompt
                        for output in agent_pipeline.flush():
                            if output:
                                server.emit(AgentOutputEvent(
                                    agent_id=server._current_tool_agent_id,
                                    source="model",
                                    text=output,
                                    mode="append",
                                ))
                        agent_pipeline.reset()

                        # Build permission content for unified output flow
                        content_parts = []

                        # When format_hint is "code", include code block first with syntax highlighting
                        if format_hint == "code" and language and raw_details:
                            code_block = f"```{language}\n{raw_details}\n```\n"
                            # Format through pipeline for syntax highlighting
                            formatted_code = []
                            for output in agent_pipeline.process_chunk(code_block):
                                formatted_code.append(output)
                            for output in agent_pipeline.flush():
                                formatted_code.append(output)
                            agent_pipeline.reset()
                            if formatted_code:
                                content_parts.append("".join(formatted_code))

                        # Add security warnings with special markers for client styling
                        if warnings:
                            # Use XML-style markers that client can parse and style separately
                            level_marker = warning_level or "warning"
                            warnings_block = f"<security-warning level=\"{level_marker}\">\n{warnings}\n</security-warning>\n"
                            content_parts.append(warnings_block)

                        # Format the permission prompt summary + options
                        if prompt_lines:
                            formatted_lines = []
                            for line in prompt_lines:
                                for output in agent_pipeline.process_chunk(line + "\n"):
                                    formatted_lines.extend(output.rstrip("\n").split("\n"))
                            for output in agent_pipeline.flush():
                                formatted_lines.extend(output.rstrip("\n").split("\n"))
                            agent_pipeline.reset()
                            content_parts.append("\n".join(formatted_lines))

                        # Emit content as AgentOutputEvent (flows through main output area)
                        if content_parts:
                            full_content = "\n".join(content_parts)
                            server.emit(AgentOutputEvent(
                                agent_id=server._current_tool_agent_id,
                                source="permission",
                                text=full_content,
                                mode="write",
                            ))

                except Exception:
                    pass  # Content formatting failed, tool tree will show minimal status

            # Check if edit option is available (indicates editable tool)
            has_edit = any(opt.get("action") == "edit" for opt in options_dicts)
            editable_metadata = None
            if has_edit and server.permission_plugin and hasattr(server.permission_plugin, '_get_tool_schema'):
                try:
                    schema = server.permission_plugin._get_tool_schema(tool_name)
                    if schema and schema.editable:
                        editable_metadata = {
                            "parameters": schema.editable.parameters if hasattr(schema.editable, 'parameters') else [],
                            "format": schema.editable.format if hasattr(schema.editable, 'format') else "yaml",
                        }
                except Exception:
                    pass

            # Emit control event to signal input mode (lightweight, no content)
            server.emit(PermissionInputModeEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                call_id=call_id,
                response_options=options_dicts,
                tool_args=tool_args if has_edit else None,
                editable_metadata=editable_metadata,
            ))

        def on_permission_resolved(tool_name: str, request_id: str,
                                   granted: bool, method: str,
                                   comment: str = ""):
            # Only clear pending-prompt state when the resolution targets
            # the currently-displayed prompt. Whitelist/blacklist auto-
            # decisions fire this hook with an empty request_id and can
            # race with a parallel tool's pending channel prompt — without
            # this guard they would clobber the pending id and the user's
            # response to the visible prompt would surface as a
            # StateError ("Unknown permission request: ...").
            if request_id and server._pending_permission_request_id == request_id:
                server._pending_permission_request_id = None
                server._waiting_for_channel_input = False

            # Resolution status is shown in the tool tree (e.g., "✓ [once]")
            # No need to emit separate output text

            server.emit(PermissionResolvedEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                granted=granted,
                method=method,
                comment=comment,
            ))

            # Emit updated permission status (a/t/i responses change the policy)
            server.emit_permission_status()

        self.permission_plugin.set_permission_hooks(
            on_requested=on_permission_requested,
            on_resolved=on_permission_resolved,
        )

    def emit_permission_status(self) -> None:
        """Emit current permission status for client toolbar updates."""
        if not self.permission_plugin:
            return
        status = self.permission_plugin.get_permission_status()
        self.emit(PermissionStatusEvent(
            effective_default=status.get("effective_default", "ask"),
            suspension_scope=status.get("suspension_scope"),
        ))

    def _setup_clarification_hooks(self) -> None:
        """Set up clarification lifecycle hooks."""
        if not self.registry:
            return

        clarification_plugin = self.registry.get_plugin("clarification")
        if not clarification_plugin or not hasattr(clarification_plugin, 'set_clarification_hooks'):
            return

        server = self

        def on_clarification_requested(tool_name: str, prompt_lines: list):
            request_id = f"clarify_{datetime.now(timezone.utc).timestamp()}"
            server._pending_clarification_request_id = request_id
            server._waiting_for_channel_input = True

            # Emit context content as AgentOutputEvent (flows through main output)
            if prompt_lines:
                content = "\n".join(prompt_lines)
                server.emit(AgentOutputEvent(
                    agent_id=server._current_tool_agent_id,
                    source="clarification",
                    text=content,
                    mode="write",
                ))

        def on_clarification_resolved(tool_name: str, qa_pairs: list):
            request_id = server._pending_clarification_request_id or ""
            server._pending_clarification_request_id = None
            server._waiting_for_channel_input = False
            # Convert qa_pairs from list of tuples to list of lists for JSON serialization
            qa_pairs_serializable = [[q, a] for q, a in qa_pairs] if qa_pairs else []
            server.emit(ClarificationResolvedEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                qa_pairs=qa_pairs_serializable,
            ))

        def on_question_displayed(tool_name: str, question_index: int,
                                  total_questions: int, question_lines: list):
            # Emit question content as AgentOutputEvent (flows through main output)
            if question_lines:
                content = "\n".join(question_lines)
                server.emit(AgentOutputEvent(
                    agent_id=server._current_tool_agent_id,
                    source="clarification",
                    text=content,
                    mode="write",
                ))

            # Emit control event to signal input mode (lightweight, no content)
            server.emit(ClarificationInputModeEvent(
                agent_id=server._current_tool_agent_id,
                request_id=server._pending_clarification_request_id or "",
                tool_name=tool_name,
                question_index=question_index,
                total_questions=total_questions,
            ))

        def on_question_answered(tool_name: str, question_index: int, answer_summary: str):
            # Question answered, waiting for next or resolution
            pass

        def on_batch_requested(tool_name: str, request, *_args):
            """Emit ClarificationBatchEvent with all questions, as a preview.

            Fires before the QueueChannel loop so that a client which can
            render every question at once (a tabbed panel, say) does not
            have to wait for them to trickle in.  It is emitted WITHOUT
            ``batch_only``: the per-question flow follows it here
            (AgentOutputEvent + ClarificationInputModeEvent, one question
            at a time, via the on_question_displayed hook), so a client
            may use either and must not prompt for both.  The runner-tier
            relay in ``runner_rpc_handlers.clarification_relay`` emits the
            same event with ``batch_only=True``, where it is the only
            delivery and answering it is mandatory (#704).
            """
            request_id = server._pending_clarification_request_id or ""
            questions_payload = []
            for i, q in enumerate(request.questions, 1):
                q_data = {
                    "index": i,
                    "text": q.text,
                    "question_type": q.question_type.value,
                    "required": q.required,
                }
                if q.choices:
                    choices_list = []
                    for j, c in enumerate(q.choices, 1):
                        choice_entry = {"text": c.text}
                        if q.default_choice == j:
                            choice_entry["default"] = True
                        choices_list.append(choice_entry)
                    q_data["choices"] = choices_list
                if q.default_choice:
                    q_data["default_choice"] = q.default_choice
                questions_payload.append(q_data)

            server.emit(ClarificationBatchEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                context=request.context or "",
                questions=questions_payload,
            ))

        clarification_plugin.set_clarification_hooks(
            on_requested=on_clarification_requested,
            on_resolved=on_clarification_resolved,
            on_question_displayed=on_question_displayed,
            on_question_answered=on_question_answered,
            on_batch_requested=on_batch_requested,
        )

    def _setup_reference_selection_hooks(self) -> None:
        """Set up reference selection lifecycle hooks."""
        if not self.registry:
            return

        references_plugin = self.registry.get_plugin("references")
        if not references_plugin or not hasattr(references_plugin, 'set_selection_hooks'):
            return

        server = self

        def on_selection_requested(tool_name: str, prompt_lines: list):
            request_id = f"ref_selection_{datetime.now(timezone.utc).timestamp()}"
            server._pending_reference_selection_request_id = request_id
            server._waiting_for_channel_input = True
            server.emit(ReferenceSelectionRequestedEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                prompt_lines=prompt_lines,
            ))

        def on_selection_resolved(tool_name: str, selected_ids: list):
            request_id = server._pending_reference_selection_request_id or ""
            server._pending_reference_selection_request_id = None
            server._waiting_for_channel_input = False
            server.emit(ReferenceSelectionResolvedEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                selected_ids=selected_ids,
            ))

        references_plugin.set_selection_hooks(
            on_requested=on_selection_requested,
            on_resolved=on_selection_resolved,
        )

    def _setup_plan_hooks(self) -> None:
        """Set up plan update hooks."""
        if not self.todo_plugin:
            return

        server = self

        def _get_agent_id(agent_name: Optional[str]) -> str:
            """Get agent ID from agent name."""
            agent_id = server._main_agent_id if agent_name is None else agent_name
            for aid, agent in server._agents.items():
                if agent.profile_name == agent_name:
                    agent_id = aid
                    break
            return agent_id

        def update_callback(plan_data: dict, agent_name: Optional[str] = None):
            """Emit PlanUpdatedEvent from plan data."""
            agent_id = _get_agent_id(agent_name)
            steps = []
            for step in plan_data.get('steps', []):
                step_data = {
                    'content': step.get('description', ''),
                    'status': step.get('status', 'pending'),
                    'active_form': step.get('active_form'),
                    'step_id': step.get('step_id', ''),
                    'result': step.get('result'),
                    'error': step.get('error'),
                }
                # Include cross-agent dependency info for blocked steps
                if step.get('blocked_by'):
                    step_data['blocked_by'] = step['blocked_by']
                if step.get('depends_on'):
                    step_data['depends_on'] = step['depends_on']
                if step.get('received_outputs'):
                    step_data['received_outputs'] = step['received_outputs']
                steps.append(step_data)
            server.emit(PlanUpdatedEvent(
                agent_id=agent_id,
                plan_name=plan_data.get('title', 'Plan'),
                steps=steps,
            ))

        def clear_callback(agent_name: Optional[str] = None):
            """Emit PlanClearedEvent."""
            agent_id = _get_agent_id(agent_name)
            server.emit(PlanClearedEvent(agent_id=agent_id))

        def step_update_callback(step_data: dict, agent_name: Optional[str] = None):
            """Emit PlanStepUpdatedEvent for lean step status deltas."""
            agent_id = _get_agent_id(agent_name)
            server.emit(PlanStepUpdatedEvent(
                agent_id=agent_id,
                step_id=step_data.get('step_id', ''),
                sequence=step_data.get('sequence', 0),
                content=step_data.get('content', ''),
                status=step_data.get('status', 'pending'),
                result=step_data.get('result'),
                error=step_data.get('error'),
                blocked_by=step_data.get('blocked_by'),
                depends_on=step_data.get('depends_on'),
                received_outputs=step_data.get('received_outputs'),
            ))

        def output_callback(source: str, text: str, mode: str):
            """Emit AgentOutputEvent for plan messages."""
            server.emit(AgentOutputEvent(
                agent_id=server._main_agent_id,
                source=source,
                text=text,
                mode=mode,
            ))

        # Reuse LivePlanReporter from jaato-tui with event-emitting callbacks
        reporter = create_live_reporter(
            update_callback=update_callback,
            step_update_callback=step_update_callback,
            clear_callback=clear_callback,
            output_callback=output_callback,
        )

        if hasattr(self.todo_plugin, '_reporter'):
            self.todo_plugin._reporter = reporter

        # Also set for subagent plugin
        if self.registry:
            subagent_plugin = self.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_plan_reporter'):
                subagent_plugin.set_plan_reporter(reporter)

    def _setup_queue_channels(self) -> None:
        """Set up queue-based channels for permission/clarification."""
        server = self

        # Phase 3 §7c step 6.6.3.6: the legacy in-process cancel-
        # token closure has been deleted.  The daemon-side
        # ``_jaato.get_session()._cancel_token`` reach was a
        # vestige of the pre-§7b.2 daemon-side message-processing
        # path.  Post-§7b.2 cancellation routes through
        # ``self._runner_rpc.session_request_stop_threadsafe(...)``
        # via :meth:`JaatoServer.stop` (§7b.1 8cbb8ba2), which
        # is the authoritative cancel surface.  The
        # CancelTokenProxy presented here is now always
        # not-cancelled — preserved as a no-op stub for
        # back-compat with the channels API contract; channel
        # consumers should call ``server.stop()`` for actual
        # cancellation.
        class CancelTokenProxy:
            @property
            def is_cancelled(self):
                return False

        cancel_token_proxy = CancelTokenProxy()

        # Output callback for channels
        def output_callback(source: str, text: str, mode: str):
            server.emit(AgentOutputEvent(
                agent_id=server._main_agent_id,
                source=source,
                text=text,
                mode=mode,
            ))

        def on_prompt_state_change(waiting: bool):
            server._waiting_for_channel_input = waiting

        # Set callbacks on clarification plugin
        if self.registry:
            clarification_plugin = self.registry.get_plugin("clarification")
            if clarification_plugin and hasattr(clarification_plugin, '_channel'):
                channel = clarification_plugin._channel
                if hasattr(channel, 'set_callbacks'):
                    channel.set_callbacks(
                        output_callback=output_callback,
                        input_queue=self._channel_input_queue,
                        prompt_callback=on_prompt_state_change,
                        cancel_token=cancel_token_proxy,
                    )

            # References plugin
            references_plugin = self.registry.get_plugin("references")
            if references_plugin and hasattr(references_plugin, '_channel'):
                channel = references_plugin._channel
                if hasattr(channel, 'set_callbacks'):
                    channel.set_callbacks(
                        output_callback=output_callback,
                        input_queue=self._channel_input_queue,
                        prompt_callback=on_prompt_state_change,
                    )

        # Permission plugin
        if self.permission_plugin and hasattr(self.permission_plugin, '_channel'):
            channel = self.permission_plugin._channel
            if channel and hasattr(channel, 'set_callbacks'):
                channel.set_callbacks(
                    output_callback=output_callback,
                    input_queue=self._channel_input_queue,
                    prompt_callback=on_prompt_state_change,
                    cancel_token=cancel_token_proxy,
                    edit_callback=self._create_edit_callback(),
                )

    def _create_edit_callback(self) -> Callable:
        """Create edit callback for permission plugin in server mode.

        In server mode, editing happens on the client side. The client opens
        the external editor and sends back the edited arguments via
        PermissionResponseRequest.edited_arguments. This callback retrieves
        those pre-stored edited arguments.

        Returns:
            Callback that returns client-provided edited arguments.
        """
        server = self

        def edit_callback(arguments: Dict[str, Any], editable: Any) -> Optional[Dict[str, Any]]:
            """Return edited arguments provided by the client."""
            edited = server._pending_edited_arguments
            server._pending_edited_arguments = None  # Consume
            return edited

        return edit_callback

    # =========================================================================
    # Client Request Handlers
    # =========================================================================

    def send_message(self, text: str, attachments: Optional[List[Dict]] = None) -> None:
        """Send a message to the model.

        Args:
            text: The message text.
            attachments: Optional list of attachments.
        """
        # Phase 3 §7c step 6.6.4.5e: ``if not self._jaato: emit error;
        # return`` guard dropped (always-true branch post-seat-flip;
        # daemon-direct ``self._runtime`` is populated synchronously by
        # initialize()).
        # ASK THE SESSION WHETHER A TURN WILL DRAIN THIS.
        #
        # This used to read ``self._model_running`` -- a daemon-side REPLICA
        # of the session's ``_is_running`` that clears strictly later (only
        # once ``session.send_message`` returns and this thread unwinds).  A
        # send arriving in that window was queued as a mid-turn prompt
        # against a turn that had already ended and already run its final
        # drain, so nothing collected it.  That is the defect
        # ``try_drain_pending_user`` was added to rescue after the fact.
        #
        # ``session.offer_message`` puts the decision where the state lives
        # and makes the check-and-enqueue atomic there.
        outcome = "needs_turn"
        rpc = self._runner_rpc
        if rpc is not None:
            offerer = getattr(rpc, "session_offer_message_threadsafe", None)
            if callable(offerer):
                try:
                    outcome = offerer(
                        text,
                        source_id="user",
                        source_type=SourceType.USER.value,
                        timeout=2.0,
                    )
                except Exception as exc:  # noqa: BLE001 — boundary
                    # ``exc_message``: the likeliest exception is the 2.0s
                    # timeout, and ``str(TimeoutError())`` is the EMPTY
                    # STRING -- this rendered as "RPC failed () -- falling
                    # back", naming no cause at all.
                    from shared.utils.errors import exc_message
                    logger.warning(
                        "offer_message RPC failed (%s: %s) -- falling back to "
                        "starting a turn, which is the safe direction: a "
                        "duplicate turn is visible, a swallowed message is "
                        "not",
                        type(exc).__name__, exc_message(exc),
                    )

        if outcome == "queued":
            self.emit(MidTurnPromptQueuedEvent(
                text=text,
                position_in_queue=0,
            ))
            return

        # ``needs_turn``: the SESSION is idle.  This daemon-side model thread
        # may nonetheless still be unwinding its previous turn, and starting a
        # second one would race it -- so hand the text to the thread that is
        # already finishing.  ``_model_running`` is read here for what it
        # actually IS (is MY thread alive), not as a proxy for session state.
        with self._pending_continuation_lock:
            if self._model_running:
                self._pending_continuations.append(text)
                # INFO, not debug.  This and its partner below are the ONLY
                # witnesses that #623's accumulate path ran, and #623 shipped
                # on inspection with no live reproduction -- so at debug the
                # evidence for a fix nobody has observed working is itself
                # unobservable on a default daemon.  Low volume by
                # construction: fires only when a send lands in the
                # wind-down window, which is the rare case.
                logger.info(
                    "SEND_WHILE_UNWINDING: stashed %d chars for the model "
                    "thread's finally to pick up (%d now waiting)",
                    len(text), len(self._pending_continuations),
                )
                return

        # Track input
        self._original_inputs.append({"text": text, "local": False})

        # Emit user message as output
        self.emit(AgentOutputEvent(
            agent_id=self._main_agent_id,
            source="user",
            text=text,
            mode="write",
        ))

        # Signal main agent is active
        self.emit(AgentStatusChangedEvent(
            agent_id=self._main_agent_id,
            status="active",
        ))

        # Start model in background.  Attachments (client-expanded base64 dicts)
        # ride the first send to the runner session's multimodal path.
        self._start_model_thread(text, attachments=attachments)

    def _emit_gc_phase_event(self, payload: Any) -> None:
        """Re-emit a runner ``gc_phase`` notification as a typed ``GCEvent``.

        The lifecycle counterpart to the ``gc_threshold`` handler below, which
        renders a human sentence ("Context usage (84.2%) exceeds threshold
        (80%). GC will run after this turn.").  That sentence stays -- it is
        good for humans -- but it was the ONLY GC signal on the bus, so a
        client wanting to show "compacting..." had to substring-match it for
        the start and guess at the end.

        Fields are read positionally from the payload rather than
        reconstructed, so a field added in ``gc_support.run_gc`` reaches
        clients without a change here.
        """
        from jaato_sdk.events import GCEvent
        if not isinstance(payload, dict):
            return
        phase = str(payload.get("phase") or "")
        if not phase:
            return
        self.emit(GCEvent(
            agent_id=self._main_agent_id or "",
            phase=phase,
            trigger_reason=payload.get("trigger_reason"),
            strategy=payload.get("strategy"),
            percent_used=payload.get("percent_used"),
            threshold=payload.get("threshold"),
            context_limit=payload.get("context_limit"),
            success=payload.get("success"),
            items_collected=payload.get("items_collected"),
            tokens_before=payload.get("tokens_before"),
            tokens_after=payload.get("tokens_after"),
            tokens_freed=payload.get("tokens_freed"),
            error=payload.get("error"),
        ))

    def _emit_budget_refusal_if_exhausted(self, result: Any) -> bool:
        """Emit ``SessionTerminatedEvent(reason="budget_exhausted")`` when the
        send result reports a ceiling refusal.  Returns True if emitted.

        A budget refusal never runs a turn: ``JaatoSession.send_message``
        short-circuits with a log line, a PROSE output chunk and a string
        return, so no turn-completion notification is produced and the normal
        ``on_session_quiescent`` path is never reached.  A wake-driven client
        therefore had nothing to wait for -- it waited out its full timeout and
        reported a generic failure, making a correct ceiling stop
        indistinguishable from a break.

        ``SessionTerminatedEvent`` rather than a new event type or an
        ``ErrorEvent``: it is the terminal event drivers already subscribe to
        (the canonical wait pattern in its own docstring), ``reason`` is an
        open vocabulary, and exhaustion genuinely means the SESSION is done --
        it refuses all further turns, not just this one.  Filing it as an
        error would mischaracterise a working ceiling as a failure.

        Extracted so it can be tested by CALLING it: inline in the model
        thread it is reachable only through a live runner.
        """
        if not isinstance(result, dict) or not result.get("budget_exhausted"):
            return False
        from jaato_sdk.events import SessionTerminatedEvent
        reason_text = result.get("budget_exhausted_reason") or "budget exhausted"
        self._terminal_reason = "budget_exhausted"
        logger.info(
            "session %s stopped at its budget ceiling: %s",
            self.session_id or "", reason_text,
        )
        self.emit(SessionTerminatedEvent(
            session_id=self.session_id or "",
            agent_id=self._main_agent_id,
            reason="budget_exhausted",
            details={
                "reason": reason_text,
                "usage": dict(result.get("budget_usage") or {}),
            },
        ))
        return True

    def _build_send_message_notification_handler(self):
        """Build the per-call ``on_notification`` demuxer used by
        ``_start_model_thread``'s runner-RPC ``session.send_message``.

        Phase 3 §7c step 6.6.4.3b.  Replaces 9 daemon-side
        callback wirings (7 ``set_*_callback`` setters at sites
        1996/2011/3396/3420/3435/3445/4323 + 2 per-call kwargs
        ``on_usage_update`` / ``on_gc_threshold`` at 3511-3512 /
        3534-3535) with one demuxer that branches on the
        NotificationFrame ``event_type``.  Runner-side session
        emits the frames during ``send_message``; this handler
        runs daemon-side off the read-loop thread.

        Each branch mirrors the pre-§7c-step-6.6.4.3b daemon-side
        callback body — same emit-events, same trace logs, same
        side effects (e.g. ``_pending_continuations`` stash, recursive
        ``_start_model_thread`` for parent-idle continuation).
        """
        server = self

        def _handle(event_type: str, payload: Dict[str, Any]) -> None:
            try:
                if event_type == "instruction_budget_updated":
                    snapshot = payload.get("snapshot") or {}
                    server.emit(InstructionBudgetEvent(
                        agent_id=snapshot.get("agent_id", "main"),
                        budget_snapshot=snapshot,
                    ))
                    return

                if event_type == "prompt_injected":
                    text = payload.get("text", "") or ""
                    server.emit(MidTurnPromptInjectedEvent(text=text))
                    return

                if event_type == "continuation_needed":
                    child_messages = payload.get("child_messages", "") or ""
                    if not child_messages:
                        return
                    if not server._model_running:
                        # Normal path: parent is idle between turns.
                        server._trace(
                            f"CONTINUATION: Child messages drained "
                            f"({len(child_messages)} chars), triggering new turn",
                        )
                        server.emit(AgentStatusChangedEvent(
                            agent_id=server._main_agent_id,
                            status="active",
                        ))
                        server._start_model_thread(child_messages)
                    else:
                        # Stash for the model_thread finally block to pick up.
                        with server._pending_continuation_lock:
                            server._pending_continuations.append(child_messages)
                        server._trace(
                            f"CONTINUATION: Stashed {len(child_messages)} "
                            f"chars (model still running)",
                        )
                    return

                if event_type == "retry":
                    message = payload.get("message", "") or ""
                    error_type = (
                        "rate_limit"
                        if "rate-limit" in message.lower()
                        else "transient"
                    )
                    server.emit(RetryEvent(
                        message=message,
                        attempt=int(payload.get("attempt", 0)),
                        max_attempts=int(payload.get("max_attempts", 0)),
                        delay=float(payload.get("delay", 0.0)),
                        error_type=error_type,
                    ))
                    return

                if event_type == "mid_turn_interrupt":
                    partial_chars = int(payload.get("partial_chars", 0))
                    prompt_preview = payload.get("prompt_preview", "") or ""
                    server._trace(
                        f"MID_TURN_INTERRUPT: partial={partial_chars}, "
                        f"preview={prompt_preview[:50]}...",
                    )
                    server.emit(MidTurnInterruptEvent(
                        partial_response_chars=partial_chars,
                        user_prompt_preview=prompt_preview,
                    ))
                    return

                if event_type == "events_subscribed":
                    from jaato_sdk.events import EventsSubscribedEvent
                    server.emit(EventsSubscribedEvent(
                        agent_id=payload.get("agent_id", "") or "",
                        event_names=list(payload.get("event_names") or []),
                    ))
                    return

                if event_type == "usage_update":
                    total_tokens = int(payload.get("total_tokens", 0))
                    if total_tokens == 0:
                        return
                    # Path E (cycle 6) E.1: context_limit + turns now
                    # come from the runner-side shim's payload (batched
                    # alongside the usage figures).  Pre-Path-E this
                    # handler called back into the runner via 2
                    # blocking RPCs DURING active send_message — a
                    # race that timed out and dropped the
                    # ContextUpdatedEvent (TUI never rendered the
                    # response).  Fallback chain: payload → cached
                    # value → 0.  ``0`` keeps the event well-formed
                    # while signaling unknown-limit downstream.
                    payload_limit = int(payload.get("context_limit", 0) or 0)
                    # E.1 HEALS THE CACHE.  The /model invalidation comment
                    # always claimed it did; it never wrote it -- one of the
                    # two promised recovery paths did not exist, and the
                    # other (the hooks' inline re-fetch) self-deadlocked.
                    #
                    # AND IT SAYS SO.  #637 gave every OTHER outcome a token
                    # and left this write silent, which turned out to leave
                    # the interesting case unnamed: after a mid-session
                    # ``/model`` invalidation only two writers can refill the
                    # cache -- the off-band fill (which announces itself) and
                    # this one.  A consumer verifying the heal observed the
                    # cache going from None to non-zero with NO token
                    # explaining it, and correctly deduced this line by
                    # elimination rather than by reading a log.  Deduction by
                    # elimination is what a missing log line costs.
                    if payload_limit and not getattr(
                            server, "_cached_context_limit", None):
                        server._cached_context_limit = payload_limit
                        logger.info(
                            "CONTEXT_LIMIT_HEALED session=%s limit=%s "
                            "source=usage_payload",
                            server.session_id, payload_limit,
                        )
                    context_limit = (
                        payload_limit
                        or (getattr(server, "_cached_context_limit", None) or 0)
                    )
                    percent_used = (
                        (total_tokens / context_limit * 100)
                        if context_limit > 0 else 0
                    )
                    turns = int(payload.get("turns", 0) or 0)
                    server.emit(ContextUpdatedEvent(
                        agent_id=server._main_agent_id,
                        usage=server._build_usage(
                            prompt_tokens=int(payload.get("prompt_tokens", 0)),
                            output_tokens=int(payload.get("output_tokens", 0)),
                            total_tokens=total_tokens,
                            cache_read_tokens=payload.get("cache_read_tokens"),
                            cache_creation_tokens=payload.get(
                                "cache_creation_tokens",
                            ),
                            reasoning_tokens=payload.get("reasoning_tokens"),
                            thinking_tokens=payload.get("thinking_tokens"),
                            cost_usd_override=payload.get("cost_usd"),
                        ),
                        context_limit=context_limit,
                        percent_used=percent_used,
                        tokens_remaining=max(0, context_limit - total_tokens),
                        turns=turns,
                    ))
                    return

                if event_type == "gc_phase":
                    server._emit_gc_phase_event(payload)
                    return

                if event_type == "gc_threshold":
                    percent_used = float(payload.get("percent_used", 0.0))
                    threshold = float(payload.get("threshold", 0.0))
                    server.emit(SystemMessageEvent(
                        message=(
                            f"Context usage ({percent_used:.1f}%) exceeds "
                            f"threshold ({threshold}%). GC will run after "
                            f"this turn."
                        ),
                        style="warning",
                    ))
                    return

                # Path F (cycle 7) F.3: AgentUIHooks bridge.  Each
                # branch routes the notification payload through the
                # existing ``ServerAgentHooks`` instance so daemon-side
                # formatting + state-mutation logic stays in one
                # place (no divergence from the pre-§7c-step-6.6.4.3b
                # in-process flow).  Hooks live on the subagent
                # plugin per _setup_agent_hooks; ``server._agents``
                # gives us direct access to the same instance.
                if event_type == "tool_call_start":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_tool_call_start(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            tool_name=payload.get("tool_name", ""),
                            tool_args=payload.get("tool_args") or {},
                            call_id=payload.get("call_id"),
                        )
                    return

                if event_type == "tool_call_end":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_tool_call_end(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            tool_name=payload.get("tool_name", ""),
                            success=bool(payload.get("success", False)),
                            duration_seconds=float(
                                payload.get("duration_seconds", 0.0) or 0.0
                            ),
                            error_message=payload.get("error_message"),
                            call_id=payload.get("call_id"),
                            backgrounded=bool(payload.get("backgrounded", False)),
                            continuation_id=payload.get("continuation_id"),
                            show_output=payload.get("show_output"),
                            show_popup=payload.get("show_popup"),
                            is_error_result=bool(payload.get("is_error_result", False)),
                            result_status=payload.get("result_status"),
                        )
                    return

                if event_type == "tool_output":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_tool_output(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            call_id=payload.get("call_id", ""),
                            chunk=payload.get("chunk", ""),
                        )
                    return

                if event_type == "turn_progress":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_turn_progress(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            total_tokens=int(payload.get("total_tokens", 0) or 0),
                            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
                            output_tokens=int(payload.get("output_tokens", 0) or 0),
                            percent_used=float(
                                payload.get("percent_used", 0.0) or 0.0
                            ),
                            pending_tool_calls=int(
                                payload.get("pending_tool_calls", 0) or 0
                            ),
                            cache_read_tokens=payload.get("cache_read_tokens"),
                            cache_creation_tokens=payload.get("cache_creation_tokens"),
                        )
                    return

                # Path F regression fix (2026-05-12): bridge the
                # runner-side ``on_agent_completed`` notification to
                # the daemon-side ``ServerAgentHooks.on_agent_completed``,
                # which fires ``AgentCompletedEvent`` into the
                # reactor engine + event-bus subscribers.  Pre-fix
                # the runner-side shim's ``on_agent_completed`` was
                # a ``pass`` no-op, so the event was dropped before
                # crossing the wire and the daemon-side reactor
                # never saw it (cascade rules keying on
                # ``AgentCompletedEvent.agent_id`` silently missed).
                if event_type == "agent_completed":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_agent_completed(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            completed_at=payload.get("completed_at"),
                            success=bool(payload.get("success", True)),
                            token_usage=payload.get("token_usage"),
                            turns_used=payload.get("turns_used"),
                            error=payload.get("error", "") or "",
                            payload=payload.get("payload"),
                        )
                    return

                # Same fix shape for ``on_session_quiescent``, which
                # fires ``SessionTerminatedEvent`` to attached
                # clients after the quiescent turn wraps up.
                if event_type == "session_quiescent":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_session_quiescent(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            reason=payload.get("reason", "natural") or "natural",
                        )
                    return

                # Path F sweep (2026-05-12): wire the remaining 6
                # ``ServerAgentHooks`` methods that the original
                # Path F audit misclassified as "covered daemon-
                # side".  Each branch unpacks the payload and
                # forwards to the daemon-side hook (which fires the
                # corresponding SDK event into the event-bus +
                # reactor engine).
                if event_type == "agent_created":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_agent_created(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            agent_name=payload.get("agent_name", "") or "",
                            agent_type=payload.get("agent_type", "") or "",
                            profile_name=payload.get("profile_name"),
                            parent_agent_id=payload.get("parent_agent_id"),
                            created_at=payload.get("created_at"),
                        )
                    return

                if event_type == "agent_status_changed":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_agent_status_changed(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            status=payload.get("status", "") or "",
                            error=payload.get("error"),
                        )
                    return

                if event_type == "agent_turn_completed":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        # ``function_calls`` is the per-call timing list
                        # (``TurnCompletedEvent.function_calls`` is typed
                        # ``List[Dict[str, Any]]``).  Pass it through as a
                        # list — pre-2026-06-07 this coerced to ``int``,
                        # which crashed the shim for any turn that actually
                        # contained tool calls and silently dropped the
                        # event.  See ``project_pr_turn_completed_event_*``
                        # for the diagnosis.
                        fc_payload = payload.get("function_calls")
                        if not isinstance(fc_payload, list):
                            fc_payload = []
                        hooks.on_agent_turn_completed(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            turn_number=int(payload.get("turn_number", 0) or 0),
                            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
                            output_tokens=int(payload.get("output_tokens", 0) or 0),
                            total_tokens=int(payload.get("total_tokens", 0) or 0),
                            duration_seconds=float(
                                payload.get("duration_seconds", 0.0) or 0.0
                            ),
                            function_calls=fc_payload,
                            cache_read_tokens=payload.get("cache_read_tokens"),
                            cache_creation_tokens=payload.get("cache_creation_tokens"),
                            spend_total_tokens=payload.get("spend_total_tokens"),
                            spend_cache_read_tokens=payload.get(
                                "spend_cache_read_tokens"),
                            spend_cache_creation_tokens=payload.get(
                                "spend_cache_creation_tokens"),
                            # ``.get`` without a default: absent and null both
                            # mean the provider reported no cost, and a 0.0
                            # default would claim it reported free.
                            cost_usd=payload.get("cost_usd"),
                            finish_reason=payload.get("finish_reason", "stop"),
                        )
                    return

                if event_type == "agent_context_updated":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_agent_context_updated(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            total_tokens=int(payload.get("total_tokens", 0) or 0),
                            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
                            output_tokens=int(payload.get("output_tokens", 0) or 0),
                            turns=int(payload.get("turns", 0) or 0),
                            percent_used=float(
                                payload.get("percent_used", 0.0) or 0.0
                            ),
                        )
                    return

                if event_type == "agent_gc_config":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_agent_gc_config(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            threshold=float(payload.get("threshold", 0.0) or 0.0),
                            strategy=payload.get("strategy", "") or "",
                            target_percent=payload.get("target_percent"),
                            continuous_mode=bool(
                                payload.get("continuous_mode", False)
                            ),
                        )
                    return

                if event_type == "agent_history_updated":
                    hooks = server._get_ui_hooks()
                    if hooks is not None:
                        hooks.on_agent_history_updated(
                            agent_id=payload.get("agent_id") or server._main_agent_id,
                            history=payload.get("history"),
                        )
                    return

                # Phase 4 §4.4 (Finding 2 closure): bridge the runner-
                # side session-plugin description-callback to the
                # daemon's SessionDescriptionUpdatedEvent stream.
                # Pre-§4.4 the daemon-side _setup_session_plugin wired
                # this callback on a daemon-side instance whose
                # set_description was never invoked post-§7c — the
                # event never fired.  Runner-side install lives in
                # _install_session_notification_callbacks (§4.4
                # sub-action B); this is the demuxer's mirror.
                if event_type == "description_updated":
                    server.emit(SessionDescriptionUpdatedEvent(
                        session_id=str(payload.get("session_id", "") or ""),
                        description=str(payload.get("description", "") or ""),
                    ))
                    return

                # Unknown event_type — log and drop.  Forward-compat
                # for runner-side additions the daemon hasn't been
                # taught about yet.
                server._trace(
                    f"NOTIFICATION_UNKNOWN: event_type={event_type!r} "
                    f"dropped (daemon doesn't recognize)",
                )
            except Exception:
                logger.exception(
                    "send_message notification handler raised for "
                    "event_type=%r — event dropped, model loop continues",
                    event_type,
                )

        return _handle

    def _start_model_thread(
        self, prompt: str, attachments: Optional[List[Dict]] = None
    ) -> None:
        """Start the model call in a background thread.

        ``attachments`` (user-message multimodal: ``[{mime_type, data:
        base64-str, display_name}, ...]``) ride only the FIRST send to the
        runner session; continuation sends (formatter feedback, nudges, child
        messages) are text-only.

        Phase 3 §7c step 6.6.4.3b: switched from
        ``server._jaato.send_message(...)`` (daemon-side
        JaatoSession) to
        ``server._runner_rpc.session_send_message_threadsafe(...)``
        (runner-RPC).  The 7 daemon-side ``set_*_callback`` wirings
        (4 init-time at sites 1996/2011/4313 + 3 per-call here)
        delete; runner-side session emits NotificationFrames; the
        ``on_notification`` demuxer below fans out by event_type.
        Closes the audit-caught 7→9 callback miss by also
        wiring ``on_usage_update`` + ``on_gc_threshold`` per-call
        kwargs runner-side as notification shims.
        """
        server = self

        def output_callback(source: str, text: str, mode: str) -> None:
            # Path F (cycle 7): emit ``AgentOutputEvent`` for stream
            # frames the runner-side ``on_output`` produces.  Pre-§7c
            # this was a no-op because the daemon-side ``_ui_hooks``
            # fired ``on_agent_output`` in-process; post-§7c the
            # runner-side ``_ui_hooks`` is None and the stream-frame
            # path is the only route for text chunks back to the
            # daemon.  Route through ``ServerAgentHooks.on_agent_output``
            # so the formatter pipeline + ``<hidden>`` filtering
            # logic stays in one place.
            hooks = server._get_ui_hooks()
            if hooks is not None:
                try:
                    hooks.on_agent_output(
                        server._main_agent_id, source, text, mode,
                    )
                except Exception:  # noqa: BLE001 — never let an emit
                    # failure crash the streaming read-loop callback
                    logger.exception(
                        "output_callback on_agent_output raised "
                        "(source=%r mode=%r)", source, mode,
                    )

        notification_handler = server._build_send_message_notification_handler()

        # Capture logging context for propagation into model thread
        from server.session_logging import (
            get_logging_context, set_logging_context, clear_logging_context,
        )
        _log_ctx = get_logging_context()

        def model_thread():
            # Propagate session logging context so plugin logger calls
            # are routed to per-session log files.
            if _log_ctx.get('session_id') and _log_ctx.get('workspace_path'):
                set_logging_context(
                    session_id=_log_ctx['session_id'],
                    client_id=_log_ctx.get('client_id'),
                    workspace_path=_log_ctx['workspace_path'],
                    session_env=_log_ctx.get('session_env'),
                )
            server._model_running = True
            # Tracks whether the try block escaped via except — used to
            # short-circuit the nudge / continuation logic in finally so
            # provider errors (quota, auth, context, etc.) don't get
            # retried by COMPLETION_NUDGE.  See PR fixing "Fix #2e:
            # COMPLETION_NUDGE fires on provider exceptions".
            terminal_error: Optional[Exception] = None
            # Per-turn clean slate — only an error-terminated turn flips this to
            # "error" (read at the SlotSettledEvent emit).  Reset here so warm
            # slot reuse can't leak a prior session's terminal reason.
            server._terminal_reason = None
            try:
                # A fresh attach to a restored session may still be (re)spawning
                # its runner asynchronously (attach has no synchronous ready-gate
                # like session.new).  Await readiness rather than deref a None
                # ``_runner_rpc`` — the reported NoneType crash.  Bounded; raise a
                # clean error on timeout (caught below as a terminal error)
                # instead of a hard AttributeError.
                # Readiness is now bootstrap-complete (mark_runner_ready), not
                # rpc-handle-live — so wait whenever it's unset.  Covers BOTH the
                # attach re-spawn (rpc None) AND a reused warm pool slot whose
                # handle is live but whose bootstrap for this session hasn't
                # finished yet (same window the client-tool-push stall hit).
                if not server._runner_ready.is_set():
                    server._runner_ready.wait(timeout=30.0)
                _rpc = server._runner_rpc
                if _rpc is None or not server._runner_ready.is_set():
                    raise RuntimeError(
                        "session runner not ready: (re)spawn + bootstrap did not "
                        "complete within 30s"
                    )
                # Run in workspace context so file operations use client's CWD
                # Also apply session env so provider/tools can access session-specific config
                with server._with_session_env(), server._in_workspace():
                    # The runner puts a TYPED budget signal on the send result
                    # (rpc.py).  Capture it: a budget refusal short-circuits
                    # before any turn runs, so no turn-completion notification
                    # fires and a driver waiting on a terminal event would sit
                    # out its whole timeout and then report a generic failure.
                    _send_result: Dict[str, Any] = {}
                    _rpc.session_send_message_threadsafe(
                        prompt,
                        on_output=output_callback,
                        on_notification=notification_handler,
                        attachments=attachments,
                        on_result=_send_result.update,
                    )
                    server._emit_budget_refusal_if_exhausted(_send_result)

                    # Auto-continuation for formatter feedback
                    # When formatters detect errors in model text output (syntax errors,
                    # validation failures), the model needs to see the feedback eagerly —
                    # not wait for the next user prompt. Loop here to inject feedback
                    # as a hidden prompt and let the model self-correct.
                    max_feedback_continuations = 2
                    for _attempt in range(max_feedback_continuations):
                        main_agent = server._agents.get(server._main_agent_id)
                        if not main_agent or not main_agent.pending_formatter_feedback:
                            break
                        feedback = main_agent.pending_formatter_feedback
                        main_agent.pending_formatter_feedback = None
                        server._trace(f"FORMATTER_FEEDBACK_CONTINUATION: attempt {_attempt + 1}, {len(feedback)} chars")
                        feedback_prompt = (
                            f"<hidden>[Formatter Feedback]\n{feedback}</hidden>"
                        )
                        server._runner_rpc.session_send_message_threadsafe(
                            feedback_prompt,
                            on_output=output_callback,
                            on_notification=notification_handler,
                        )

                    # Update context usage
                    # Phase 3 §7c step 6.6.4.5b: route through runner-RPC.
                    if server._runner_rpc is not None:
                        usage = server._runner_rpc.session_get_context_usage_threadsafe()
                        context_limit = (
                            server._runner_rpc.session_get_context_limit_threadsafe()
                        )
                        server.emit(ContextUpdatedEvent(
                            agent_id=server._main_agent_id,
                            usage=server._build_usage(
                                prompt_tokens=usage.get('prompt_tokens', 0),
                                output_tokens=usage.get('output_tokens', 0),
                                total_tokens=usage.get('total_tokens', 0),
                            ),
                            context_limit=context_limit,
                            percent_used=usage.get('percent_used', 0),
                            tokens_remaining=usage.get('tokens_remaining', 0),
                            turns=usage.get('turns', 0),
                        ))

            except KeyboardInterrupt as e:
                server.emit(SystemMessageEvent(
                    message="Interrupted",
                    style="warning",
                ))
                terminal_error = e
            except Exception as e:
                # Permanent INFO-level log of the wrapped error text at
                # emit time.  Lets consumers verify end-to-end that the
                # client-facing ErrorEvent payload carries the
                # vendor-correct message (Fix #1a in PR #118) without
                # needing to parse the binary IPC frame separately.
                # Greppable token: MODEL_THREAD_TERMINAL_ERROR.
                #
                # RUNNER-SIDE FRAMES, when the failure came from across the
                # RPC boundary.  The runner sanitizes and ships them in
                # ``ErrorPayload.traceback``; ``RunnerCallError`` now carries
                # them here.  Without this the crash reached every consumer
                # as ONE SANITIZED LINE -- exception type and message intact,
                # frames gone -- and the line reads like a finished error, so
                # a reader assumes they have the wrong log rather than that
                # the frames were dropped.
                #
                # They go to BOTH witnesses on purpose: the log for whoever
                # is on the machine, ``details`` for a client that is not.
                # OUR PLUMBING FAILING IS NOT THE AGENT FAILING.
                #
                # This handler terminates the session for anything it catches.
                # A ``RunnerRPCTimeout`` is the daemon's own transport --
                # typically its event loop not scheduling a coroutine -- and
                # the session behind it is healthy.  Terminating for one
                # killed a cascade half mid-run, twice on two builds: the
                # cascade policy unloads on reason=error, the session goes
                # cold, and a cold sibling is not woken by a sibling message,
                # so the surviving half sent into a corpse for the rest of the
                # run.
                #
                # Enumerating what must NOT terminate, rather than what must:
                # a framework-internal type nobody listed here still dies (the
                # status quo), whereas listing what must terminate would let
                # an unlisted PROVIDER error survive and COMPLETION_NUDGE
                # cycle on it -- the bug this terminal path exists to stop.
                from server.runner_rpc_client import RunnerRPCTimeout

                if isinstance(e, RunnerRPCTimeout):
                    logger.warning(
                        "MODEL_THREAD_TRANSPORT_ERROR error_type=%s error=%s "
                        "-- the TURN failed; the SESSION stays loaded. This "
                        "is daemon-side plumbing, not the agent.",
                        type(e).__name__, str(e),
                    )
                    server.emit(ErrorEvent(
                        error=str(e),
                        error_type=type(e).__name__,
                        recoverable=True,
                    ))
                    # RETURN.  Without it the terminal path below runs anyway:
                    # ``terminal_error = e`` is reached unconditionally and the
                    # finally takes the termination branch, so the session dies
                    # exactly as before with a better log line above it.  #628
                    # shipped that way -- the comment described the control
                    # flow and nothing implemented it -- and a cascade half
                    # still died 3.5 minutes in, WARNING and INFO one
                    # millisecond apart on the same exception.
                    #
                    # ``return`` from inside ``except`` still runs the
                    # ``finally``, which is the point: the turn winds down its
                    # ordinary way (pending continuation, status) with
                    # ``terminal_error`` left None, so the session stays
                    # loaded.  Re-raising instead would run the finally and
                    # then escape the thread target unhandled.
                    return

                _runner_tb = getattr(e, "traceback_text", None)
                logger.info(
                    "MODEL_THREAD_TERMINAL_ERROR error_type=%s error=%s",
                    type(e).__name__, str(e),
                )
                if _runner_tb:
                    logger.error(
                        "MODEL_THREAD_TERMINAL_ERROR runner traceback:\n%s",
                        _runner_tb,
                    )
                server.emit(ErrorEvent(
                    error=str(e),
                    error_type=type(e).__name__,
                    details=({"runner_traceback": _runner_tb}
                             if _runner_tb else None),
                ))
                terminal_error = e
            finally:
                server._model_running = False
                server._model_thread = None

                # When the try block escaped via except — e.g. a non-
                # transient provider error (UsageLimitError, quota,
                # APIKeyInvalidError, ContextLimitError) propagated out
                # of with_retry — emit a SessionTerminatedEvent and exit
                # the model_thread cleanly.  Without this guard the
                # finally would fall through to the status check and
                # fire COMPLETION_NUDGE, which would restart the model
                # thread and immediately hit the same provider error
                # again.  That cycle masked the actual error (the v83-v91
                # cascade investigation chased an imaginary regression
                # for 4 days because Zhipu 5h-quota APIStatusError kept
                # nudging instead of terminating).
                #
                # The ErrorEvent has already been emitted above; the
                # SessionTerminatedEvent here gives the cascade
                # orchestrator / TUI a terminal signal so they don't sit
                # waiting for AGENT_COMPLETED that will never arrive.
                if terminal_error is not None:
                    # Single chokepoint: emits AgentErrorEvent (recovery first
                    # refusal — this point is post-with_retry-exhaustion by
                    # construction) THEN SessionTerminatedEvent(reason=error)
                    # (teardown, carrying the cause for log-grep-free
                    # surfacing), and stamps _terminal_reason.  The invariant
                    # "AgentErrorEvent precedes every reason=error" is structural
                    # because the terminal event is never constructed here
                    # directly.  See docs/design/agent-error-recovery-event.md.
                    server._emit_error_termination_from_exc(terminal_error)
                    clear_logging_context()
                    return

                # Process continuation stashed during _drain_child_messages()
                # or by a send that arrived while this thread was unwinding.
                #
                # The stash is written from the RPC read loop and from caller
                # threads, and read here on the model thread -- so it is taken
                # under ``_pending_continuation_lock``.  The comment this
                # replaces claimed writer and reader "both run on the same
                # model_thread, so no race condition", which was true only of
                # the original in-process path and false for every
                # notification-driven write since.  The lost update it allowed
                # was silent: the stashed text simply never became a turn.
                with server._pending_continuation_lock:
                    stashed = server._pending_continuations
                    server._pending_continuations = []
                # ALL of them, joined -- the same shape
                # ``_drain_child_messages`` uses for a batch it collected.
                # Taking only one would re-introduce the loss with extra steps.
                pending = "\n\n".join(stashed) if stashed else None
                if pending:
                    # INFO for the same reason as SEND_WHILE_UNWINDING above.
                    # ``count>1`` is the case that USED to lose messages: the
                    # stash was a single slot until #623, so every message but
                    # the last was silently overwritten.  Named in the line so
                    # one grep separates "the fix ran" from "the fix mattered".
                    logger.info(
                        "CONTINUATION: Processing %d stashed message(s), "
                        "%d chars%s",
                        len(stashed), len(pending),
                        "  <- MULTIPLE: pre-#623 this lost all but the last"
                        if len(stashed) > 1 else "",
                    )
                    server.emit(AgentStatusChangedEvent(
                        agent_id=server._main_agent_id,
                        status="active",
                    ))
                    server._start_model_thread(pending)
                    clear_logging_context()
                    return  # new thread handles idle/done status

                # Multi-turn deadlock fix: drain a high-priority (USER) send
                # that raced into this turn's wind-down.  ``turn.completed``
                # reaches the client (which sends its next turn) BEFORE this
                # finally cleared ``_model_running`` above, so the daemon gate
                # forwarded that send as an ``inject_prompt``; the runner-side
                # session — idle, with its per-RPC continuation callback
                # already restored to None — queued it with no drainer (see
                # ``JaatoSession.inject_prompt`` / ``try_drain_pending_user``).
                # Atomically pop it and start a fresh turn.  Mirrors the
                # ``_pending_continuations`` drain above; runner-tier only
                # (daemon-local sessions have no ``_runner_rpc``).
                if server._runner_rpc is not None:
                    drained = None
                    try:
                        drained = (
                            server._runner_rpc
                            .session_try_drain_pending_user_threadsafe()
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Best-effort — a transport error just means no drain
                        # this turn; don't tear down the model thread.
                        server._trace(
                            f"DRAIN_PENDING_USER_RPC: try_drain_pending_user "
                            f"raised {type(exc).__name__}: {exc} — skipping "
                            f"drain this turn",
                        )
                    if drained:
                        server._trace(
                            f"DRAIN_PENDING_USER: starting fresh turn for a "
                            f"send that raced the turn wind-down "
                            f"({len(drained)} chars)",
                        )
                        server.emit(AgentStatusChangedEvent(
                            agent_id=server._main_agent_id,
                            status="active",
                        ))
                        server._start_model_thread(drained)
                        clear_logging_context()
                        return  # new thread handles idle/done status

                # Determine whether the main agent is truly finished or just
                # paused waiting for external input.
                #   "idle"  – waiting for user input or subagent results
                #   "done"  – nothing left to do, session can exit
                has_active_subagents = any(
                    info.agent_id != server._main_agent_id and info.completed_at is None
                    for info in server._agents.values()
                )
                if server._waiting_for_channel_input or has_active_subagents:
                    status = "idle"
                else:
                    status = "done"

                # Completion-nudge guard for top-level sessions.  When
                # the loop is about to terminate (status="done") AND
                # the agent never called ``signal_completion``, inject
                # a framework reminder via ``_pending_continuations``
                # and restart the model thread — the existing pending
                # path above will pick it up next iteration.  Bounded
                # by ``MAX_COMPLETION_NUDGES`` so a model that keeps
                # ignoring the reminder eventually halts naturally.
                # Mirrors the subagent-side guard in
                # ``shared.plugins.subagent.plugin._run_subagent_async``.
                #
                # Server 0.6.61+: skip the nudge when ``signal_completion``
                # isn't in the session's tool surface (interactive root
                # sessions filter it out per LifecycleTools).  Without
                # this check the nudge text instructs the model to call
                # a tool it can't see in its schema — providers don't
                # strictly enforce schema membership, so the model
                # would dutifully emit the call from cached knowledge,
                # and the executor (still registered) would terminate
                # the session.  Skipping the nudge entirely when the
                # tool was filtered preserves the user's expected
                # contract: TUI / web / chat sessions stay alive across
                # turns until the user disconnects.
                MAX_COMPLETION_NUDGES = 2
                # Phase 3 §7c step 6.6.4.3b: completion-nudge
                # guard now goes through the runner-RPC
                # ``session.try_completion_nudge`` handler (shipped
                # in §7c step 6.6.4.3a at commit 68abe7c8).  The
                # one round-trip atomically reads
                # ``_signal_completion_called``, reads
                # ``_completion_nudges_fired``, and increments
                # the latter when a nudge should fire — replacing
                # the pre-§7c-step-6.6.4.3b 3 private-attr
                # reaches into ``server._jaato.get_session()``.
                #
                # ``signal_completion_in_surface`` stays daemon-
                # side: tool-schemas already routed through the
                # daemon-side ``JaatoClient.get_tool_schemas()``
                # (§7c step 6.6.3.6 at commit 89f0c001), and
                # daemon-tier filtering is unchanged.
                # Phase 3 §7c step 6.6.4.5c.5: route through runner-RPC.
                # Daemon wrapper reconstructs ToolSchema NamedTuples so
                # ``getattr(t, 'name', None)`` works unchanged.
                if server._runner_rpc is not None:
                    try:
                        tool_schemas = (
                            server._runner_rpc
                            .session_get_tool_schemas_threadsafe()
                        )
                    except Exception:  # noqa: BLE001 — nudge guard is best-effort
                        tool_schemas = []
                else:
                    tool_schemas = []
                signal_completion_in_surface = any(
                    getattr(t, 'name', None) == 'signal_completion'
                    for t in tool_schemas
                )
                should_nudge = False
                nudges_fired = 0
                if (
                    status == "done"
                    and signal_completion_in_surface
                    and server._runner_rpc is not None
                ):
                    try:
                        should_nudge, nudges_fired = (
                            server._runner_rpc
                                .session_try_completion_nudge_threadsafe(
                                    MAX_COMPLETION_NUDGES,
                                )
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Nudge guard is best-effort — a transport
                        # error here just means no nudge fires this
                        # turn.  Don't tear down the model thread.
                        server._trace(
                            f"COMPLETION_NUDGE_RPC: try_completion_nudge "
                            f"raised {type(exc).__name__}: {exc} — "
                            f"skipping nudge this turn",
                        )
                # The framework asked and gave up.  ``should_nudge`` is
                # also False when the agent DID signal, so the count is what
                # separates them: it is only at the ceiling when nudges
                # actually fired and were ignored.
                if (
                    status == "done"
                    and signal_completion_in_surface
                    and not should_nudge
                    and nudges_fired >= MAX_COMPLETION_NUDGES
                ):
                    _agent = server._agents.get(server._main_agent_id)
                    if _agent is not None:
                        _agent.completion_gap = "not_signalled_after_nudges"
                    server._trace(
                        f"COMPLETION_GAP: agent ended without "
                        f"signal_completion after "
                        f"{nudges_fired}/{MAX_COMPLETION_NUDGES} nudges — "
                        f"no terminal event will fire for this session"
                    )

                if should_nudge:
                    server._trace(
                        f"COMPLETION_NUDGE: agent ended its loop without "
                        f"signal_completion (nudge "
                        f"{nudges_fired}/{MAX_COMPLETION_NUDGES}) "
                        f"— re-prompting"
                    )
                    nudge = (
                        "Your session is about to end without calling "
                        "`signal_completion`. The loop cannot close cleanly "
                        "until you either continue the work with another "
                        "tool call, or call `signal_completion` per your "
                        "profile's payload schema with the appropriate "
                        "decision and evidence. Please proceed with one of "
                        "those two paths."
                    )
                    server.emit(AgentStatusChangedEvent(
                        agent_id=server._main_agent_id,
                        status="active",
                    ))
                    server._start_model_thread(nudge)
                    clear_logging_context()
                    return  # new thread handles idle/done status

                # PR #179 (Finding D, 2026-05-21): nudge-exhaust
                # detection.  When the agent has been nudged
                # MAX_COMPLETION_NUDGES times and still didn't call
                # ``signal_completion``, the session has effectively
                # failed.  Pre-fix the only event was
                # AgentStatusChangedEvent(status="done") — which fires
                # ON NORMAL COMPLETION TOO — so cascade observers
                # couldn't distinguish success from nudge-exhaust
                # failure.  Surfaced by kb-orchestrator
                # v152-retry-12 cascade 2026-05-21: transform
                # step 5 looped 201 turns + exhausted nudges +
                # observer saw no terminal signal + 90-min poll
                # timeout.
                #
                # Distinguishing condition: ``nudges_fired >=
                # MAX_COMPLETION_NUDGES`` AND signal_completion was
                # in the surface (so the agent COULD have called it
                # but didn't).  Emit ErrorEvent (for clients
                # subscribed to errors) + SessionTerminatedEvent
                # (for cascade-clients per Phase 1 default policy
                # — auto-unloads headless / cascade-owned sessions).
                #
                # AgentStatusChangedEvent(status="done") still fires
                # below for backward compat with consumers that
                # don't watch SessionTerminatedEvent.
                if (
                    nudges_fired >= MAX_COMPLETION_NUDGES
                    and signal_completion_in_surface
                ):
                    from jaato_sdk.events import ErrorEvent as _ErrorEvent
                    server._trace(
                        f"NUDGE_EXHAUSTED: agent looped "
                        f"{nudges_fired}/{MAX_COMPLETION_NUDGES} "
                        f"nudges without calling signal_completion — "
                        f"emitting terminal events"
                    )
                    nudge_exhaust_summary = (
                        f"Agent loop exhausted "
                        f"{MAX_COMPLETION_NUDGES} completion nudges "
                        f"without calling signal_completion"
                    )
                    server.emit(_ErrorEvent(
                        error=nudge_exhaust_summary,
                        error_type="NudgeExhausted",
                    ))
                    # Single chokepoint: AgentErrorEvent (recovery first refusal
                    # — nudge exhaustion is the framework's automatic-management
                    # giving up) THEN SessionTerminatedEvent(reason=error) so
                    # nudge-exhaust is distinguishable from a provider error
                    # without log-grep, plus _terminal_reason.  Structural
                    # invariant — see _emit_error_termination /
                    # docs/design/agent-error-recovery-event.md.
                    server._emit_error_termination(
                        error_type="NudgeExhausted",
                        error_summary=nudge_exhaust_summary,
                    )

                server.emit(AgentStatusChangedEvent(
                    agent_id=server._main_agent_id,
                    status=status,
                ))
                clear_logging_context()

        self._model_thread = threading.Thread(target=model_thread, daemon=True)
        self._model_thread.start()

    def respond_to_permission(self, request_id: str, response: str,
                              edited_arguments: Optional[Dict[str, Any]] = None) -> None:
        """Respond to a permission request.

        Phase 3 §7c Step 7.3: tries two resolution paths.

        1. **Runner-fired ASK (post-seat-flip)**: the runner-side
           permission plugin's ``RunnerRPCChannel`` relayed the ASK
           via the ``client.prompt_operator`` RPC; the daemon's
           ``PromptOperatorHandler`` holds the pending future keyed
           by ``request_id``.  ``resolve_response`` returns True if
           a future was pending — typical post-§7c.
        2. **Daemon-fired ASK (legacy / fallback)**: the daemon-side
           channel set ``_pending_permission_request_id`` and is
           reading from ``_channel_input_queue``.  Falls through to
           pushing the response into the queue.

        When neither path resolves, emit an "Unknown permission
        request" ErrorEvent — the request_id doesn't match any
        pending state.

        Args:
            request_id: The permission request ID.
            response: The response (y, n, a, never, etc.).
            edited_arguments: Optional edited tool arguments (when response is "e"
                and the client handled editing locally).
        """
        # Path 1: try the runner-RPC handler first.
        prompt_handler = getattr(self, "_prompt_operator_handler", None)
        if prompt_handler is not None:
            if prompt_handler.resolve_response(
                request_id, response, edited_arguments=edited_arguments,
            ):
                # Runner-fired ASK resolved.  No need to touch the
                # daemon-side queue or ``_pending_edited_arguments``;
                # the runner-side permission plugin gets the
                # PromptResponse directly from its outgoing_call
                # await.
                return

        # Path 2: daemon-fired ASK fallback (legacy).
        if self._pending_permission_request_id == request_id:
            # Store edited arguments before putting response in queue so the
            # edit_callback can retrieve them synchronously
            if edited_arguments is not None:
                self._pending_edited_arguments = edited_arguments
            self._channel_input_queue.put(response)
            return

        # Neither path resolved — unknown request.
        self.emit(ErrorEvent(
            error=f"Unknown permission request: {request_id}",
            error_type="StateError",
        ))

    def respond_to_clarification(self, request_id: str, response: str) -> None:
        """Respond to a clarification question.

        Args:
            request_id: The clarification request ID.
            response: The user's answer.
        """
        if self._pending_clarification_request_id != request_id:
            self.emit(ErrorEvent(
                error=f"Unknown clarification request: {request_id}",
                error_type="StateError",
            ))
            return

        self._channel_input_queue.put(response)

    def respond_to_clarification_batch(
        self,
        request_id: str,
        answers: List[str],
        cancelled: bool = False,
    ) -> None:
        """Respond to a batch clarification request with all answers at once.

        Runner-tier sessions resolve the ``ClarificationRelayHandler``
        future directly (the runner-side relay channel is awaiting it);
        daemon-local sessions feed the legacy ``_channel_input_queue`` so
        the QueueChannel loop picks the answers up one by one.

        Args:
            request_id: The clarification request ID.
            answers: Ordered list of answer strings, one per question.
            cancelled: Abandon the clarification instead of answering it.
                The relay resolves its future as cancelled; the legacy
                path feeds the QueueChannel the ``cancel`` sentinel it
                already understands.  Either way ``request_clarification``
                returns ``{"cancelled": True}`` and the turn continues,
                which is what keeps an unanswerable question from blocking
                a turn forever (#704).  ``answers`` is ignored.
        """
        # Runner→daemon relay path (post-seat-flip runner sessions) — mirror
        # of respond_to_permission's prompt_operator_handler.resolve_response.
        relay = getattr(self, "_clarification_relay_handler", None)
        if relay is not None and relay.resolve_response(
            request_id, answers, cancelled=cancelled
        ):
            return

        # Legacy daemon-local QueueChannel path.
        if self._pending_clarification_request_id != request_id:
            self.emit(ErrorEvent(
                error=f"Unknown clarification request: {request_id}",
                error_type="StateError",
            ))
            return

        if cancelled:
            # QueueChannel treats a literal "cancel" as an abort and stops
            # reading, so one entry ends the whole request.
            self._channel_input_queue.put("cancel")
            return

        for answer in answers:
            self._channel_input_queue.put(answer)

    def respond_to_reference_selection(self, request_id: str, response: str) -> None:
        """Respond to a reference selection request.

        Args:
            request_id: The reference selection request ID.
            response: The user's selection (e.g., "1,3,4", "all", "none").
        """
        if self._pending_reference_selection_request_id != request_id:
            self.emit(ErrorEvent(
                error=f"Unknown reference selection request: {request_id}",
                error_type="StateError",
            ))
            return

        self._channel_input_queue.put(response)

    def _find_plugin_for_command(self, command: str) -> Any:
        """Find the plugin that provides a user command.

        Args:
            command: The command name to find.

        Returns:
            The plugin instance or None if not found.

        Phase 3 §7c step 4: read directly from ``self._runtime``
        instead of ``self._jaato.get_runtime()``.
        """
        if self._runtime is None:
            return None

        registry = self._runtime.registry
        if not registry:
            return None

        # Search exposed plugins for the command
        for plugin_name in registry.list_exposed():
            plugin = registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, 'get_user_commands'):
                for cmd in plugin.get_user_commands():
                    if cmd.name == command:
                        return plugin

        # Also check permission plugin
        perm = self._runtime.permission_plugin
        if perm and hasattr(perm, 'get_user_commands'):
            for cmd in perm.get_user_commands():
                if cmd.name == command:
                    return perm

        return None

    def _get_sandbox_paths(self) -> list[dict[str, str]]:
        """Build the list of sandbox-allowed paths for @@ completion.

        Returns:
            List of {path, description} dicts for the client's completion cache.

        Phase 3 §7c step 4: read directly from ``self._runtime``
        instead of ``self._jaato.get_runtime()``.
        """
        paths = []
        if self._runtime is None:
            return paths

        registry = self._runtime.registry
        if not registry:
            return paths

        # Workspace root
        workspace = registry.get_workspace_path()
        if workspace:
            paths.append({"path": workspace, "description": "workspace"})

        # Authorized external paths from sandbox manager / plugins
        try:
            import os
            authorized = registry.list_authorized_paths()
            for auth_path, source in authorized.items():
                # Skip if same as workspace (already added)
                if workspace and os.path.realpath(auth_path) == os.path.realpath(workspace):
                    continue
                paths.append({"path": auth_path, "description": f"allowed ({source})"})
        except Exception:
            pass

        # System temp directory
        paths.append({"path": "/tmp", "description": "system temp"})

        return paths

    def stop(self) -> bool:
        """Stop current operation.

        Returns:
            True iff a cancel was actually issued (False when no
            message running).

        Phase 3 §7c step 6.3: daemon-side leg dropped.  The
        runner-side ``session.request_stop`` RPC is now the only
        source of truth for cancellation.  Pre-step-6.3 the
        daemon-side ``_jaato.stop()`` call mirrored the cancel
        onto an in-process ``JaatoSession`` whose message-
        processing loop was orphan post-§7b.2 (no message
        processing happens daemon-side).
        """
        rpc = self._runner_rpc
        if rpc is None:
            return False
        forwarder = getattr(rpc, "session_request_stop_threadsafe", None)
        if not callable(forwarder):
            return False
        try:
            return bool(forwarder(reason="user_stop", timeout=2.0))
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "stop: runner RPC propagation failed (%s)", exc,
            )
            return False

    def execute_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """Execute a command.

        Args:
            command: Command name (e.g., 'model', 'save', 'resume').
            args: Command arguments.

        Returns:
            Command result dict.

        Phase 3 §7c step 6.6.4.5e: ``if not self._jaato`` guard
        dropped (always-true branch post-seat-flip).
        """
        # Phase 3 §7c step 6.6.4.5c.2: route through runner-RPC
        # (dict-shape-only wire format reconstructed daemon-side
        # into UserCommand NamedTuples — full ``parse_command_args``
        # surface preserved).
        if self._runner_rpc is None:
            return {"error": "Client not initialized"}
        user_commands = self._runner_rpc.session_get_user_commands_threadsafe()
        if command not in user_commands:
            return {"error": f"Unknown command: {command}"}

        cmd = user_commands[command]
        raw_args = " ".join(args)
        parsed_args = parse_command_args(cmd, raw_args)

        # Special handling for save command
        if command.lower() == "save":
            parsed_args["user_inputs"] = self._original_inputs.copy()

        # Find and configure plugin output callback for real-time output.
        # User commands run outside agent context, so we buffer _emit() output
        # and send as a SystemMessageEvent (not AgentOutputEvent).
        plugin = self._find_plugin_for_command(command)
        output_parts = []
        if plugin and hasattr(plugin, 'set_output_callback'):
            def output_callback(source: str, text: str, mode: str) -> None:
                output_parts.append(text)
            plugin.set_output_callback(output_callback)

        try:
            # Phase 3 §7c step 6.6.4.5c.3: route through runner-RPC.
            # Per-type reconstruction (Path A bounded to HelpLines /
            # dict / str) preserves the structured-access invariants
            # the downstream code depends on (`isinstance(result,
            # HelpLines)` at 4073, `isinstance(result, dict)` +
            # `result.get("success")` at 4078, `isinstance(result,
            # dict)` at the IPC-return fallback).
            result, shared = (
                self._runner_rpc.session_execute_user_command_threadsafe(
                    command, parsed_args,
                )
            )

            # Send accumulated _emit() output as a single system message
            if output_parts:
                combined = "".join(output_parts).rstrip("\n")
                if combined:
                    self.emit(SystemMessageEvent(
                        message=combined,
                        style="info",
                    ))

            # After memory commands, push updated memory list for completion cache
            # (must run before HelpLines early return so memory list/help also refresh)
            if command.lower() == "memory":
                mem_plugin = self._find_plugin_for_command("memory")
                if mem_plugin and hasattr(mem_plugin, 'get_memory_metadata'):
                    self.emit(MemoryListEvent(memories=mem_plugin.get_memory_metadata()))

            # After sandbox commands, push updated sandbox paths for @@ completion cache
            if command.lower() == "sandbox":
                self.emit(SandboxPathsEvent(paths=self._get_sandbox_paths()))

            # After services commands, push updated service list for completion cache
            if command.lower() == "services":
                svc_plugin = self._find_plugin_for_command("services")
                if svc_plugin and hasattr(svc_plugin, 'get_service_metadata'):
                    self.emit(ServiceListEvent(services=svc_plugin.get_service_metadata()))

            # Handle HelpLines result - emit HelpTextEvent for pager display
            if isinstance(result, HelpLines):
                self.emit(HelpTextEvent(lines=result.lines))
                return {"_pager": True}

            # Handle model change
            if command.lower() == "model" and isinstance(result, dict):
                if result.get("success") and result.get("current_model"):
                    self._model_name = result["current_model"]
                    # Path E (cycle 6) E.3: invalidate cached
                    # context_limit — different model can have a
                    # different context window.  Healed by the next
                    # ``usage_update`` notification that carries a limit
                    # (E.1 now actually writes the cache), or by the
                    # non-blocking off-band fill the hooks schedule on a
                    # miss.  (An earlier version of this comment promised
                    # an in-band re-fetch that in fact self-deadlocked,
                    # and an E.1 heal that did not exist.)
                    self._cached_context_limit = None
                    self.emit(SystemMessageEvent(
                        message=f"Model changed to: {self._model_name}",
                        style="info",
                    ))
                    # Push updated model info so client toolbar refreshes
                    self.emit(SessionInfoEvent(
                        model_provider=self._model_provider,
                        model_name=self._model_name,
                    ))

            # Handle permission status change
            if command.lower() == "permissions":
                self.emit_permission_status()

            # Handle auth completion - if auth was pending and user ran the matching auth command
            if self._auth_pending and self._auth_plugin_command and command.lower() == self._auth_plugin_command.lower():
                self._check_auth_completion()

            return result if isinstance(result, dict) else {"result": str(result)}

        except Exception as e:
            return {"error": str(e)}

        finally:
            # Clear output callback
            if plugin and hasattr(plugin, 'set_output_callback'):
                plugin.set_output_callback(None)

    def clear_history(self) -> None:
        """Clear conversation history.

        Phase 3 §7c step 6.3: daemon-side leg dropped.  The
        runner-side ``session.reset`` RPC is now the only source
        of truth for conversation-history state.  Pre-step-6.3
        the daemon-side ``_jaato.reset_session()`` call mirrored
        the reset onto an in-process ``JaatoSession`` whose state
        was orphan post-§7b.2 (no message processing happens
        daemon-side).
        """
        rpc = self._runner_rpc
        if rpc is not None:
            forwarder = getattr(rpc, "session_reset_threadsafe", None)
            if callable(forwarder):
                try:
                    forwarder(timeout=2.0)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.debug(
                        "clear_history: runner RPC propagation "
                        "failed (%s) — daemon-side AgentState still cleared",
                        exc,
                    )
        self._original_inputs = []
        if self._main_agent_id in self._agents:
            main_state = self._agents[self._main_agent_id]
            main_state.history = []
            main_state.turn_accounting = []
            main_state.context_usage = {}

        self.emit(SystemMessageEvent(
            message="History cleared",
            style="info",
        ))

    # =========================================================================
    # Getters
    # =========================================================================

    @property
    def is_processing(self) -> bool:
        """Check if model is currently processing."""
        return self._model_running

    # Phase 3 §7c step 6.6.3.6: ``JaatoServer.get_session()``
    # was removed.  Pre-§7c-step-6.6.3.6 it returned the
    # underlying daemon-side ``JaatoSession`` instance for
    # direct API access by session-manipulation tools (e.g.
    # session_ops's interrogate_session).  Post-seat-flip the
    # session lives runner-side and is not directly accessible
    # from the daemon process; consumers should use the
    # runner-RPC surface instead:
    #
    #   - ``server._runner_rpc.session_get_history_threadsafe()``
    #     for history reads (§3.3c precursor).
    #   - ``server._runner_rpc.session_replay_messages_threadsafe(messages)``
    #     for replay (§7c step 6.6.3.4 at commit 24ed6c0f).
    #   - ``server._runner_rpc.session_resolve_fork_point_threadsafe(...)``
    #     for fork-point resolution (§7c step 6.6.3.5 at e4eddc0e).
    #   - ``server._runner_rpc.session_inject_prompt_threadsafe(...)``
    #     for prompt injection (§7c step 6.1 (3/3) at 14e57709).
    #   - ``server._runner_rpc.session_set_initial_history_threadsafe(...)``
    #     for history seeding (§7c step 6.6.1.1 at 3f859e3a).
    #   - ``server._runner_rpc.session_append_history_message_threadsafe(...)``
    #     for synthetic-message append (§7c step 6.6.3.1 at aa9059ec).
    #   - ``server._runner_rpc.session_set_session_state_threadsafe(...)``
    #     for state injection (§3.3c precursor).
    #   - ``server._runner_rpc.session_snapshot_instruction_budget_threadsafe()``
    #     / ``session_snapshot_conversation_budget_threadsafe()`` for
    #     budget reads (§7c step 6.1 (2/3) + 6.6.3.2).
    #   - ``server._runner_rpc.session_restore_*_threadsafe(...)`` for
    #     persistence-restore paths (§7c step 6.6.1 trio + 6.6.3.2).

    @property
    def is_waiting_for_input(self) -> bool:
        """Check if waiting for permission/clarification input."""
        return self._waiting_for_channel_input

    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self._model_name

    @property
    def model_provider(self) -> str:
        """Get current model provider."""
        return self._model_provider

    def get_agents(self) -> Dict[str, AgentState]:
        """Get all tracked agents."""
        return self._agents.copy()

    def get_history(self, agent_id: Optional[str] = None) -> List[Any]:
        """Get conversation history for an agent.

        ``agent_id=None`` resolves to the main agent's id (``main`` by
        default, or the ``--agent <name>`` value when one was supplied).

        Post-seat-flip the session and its AUTHORITATIVE history live
        runner-side; the daemon-side ``_agents[*].history`` is NOT
        maintained for a runner-based session — it is empty after a cold
        disk-restore and stale after new turns (``on_agent_history_updated``
        only fires in the in-process path).  So for the MAIN agent, when a
        runner is attached, read from the runner via the RPC surface the
        architecture mandates (see the ``get_session`` removal note above).
        Without this, disk-restored WS sessions replayed an empty transcript
        and ``history.request`` returned nothing even though the runner had
        the turns.  Subagents (no per-agent runner history RPC) and
        in-process sessions read the daemon-side copy as before.  A runner
        read failure falls back to daemon-side so a transient RPC blip
        degrades rather than raises.
        """
        if agent_id is None:
            agent_id = self._main_agent_id
        if self._runner_rpc is not None and agent_id == self._main_agent_id:
            fetched = self._runner_history_or_none()
            if fetched is not None:
                return fetched
        if agent_id in self._agents:
            return self._agents[agent_id].history
        return []

    def _runner_history_or_none(self) -> Optional[List[Any]]:
        """Fetch + deserialize the runner-side main history, or None on any
        failure (so ``get_history`` can fall back to the daemon-side copy).

        The runner serializes each message with the canonical session
        serializer (``_serialize_message_for_wire`` → ``serialize_message``),
        so ``deserialize_history`` round-trips it back to ``Message`` objects
        — the shape every ``get_history`` consumer already expects.
        """
        rpc = self._runner_rpc
        if rpc is None:
            return None
        try:
            dicts = rpc.session_get_history_threadsafe(timeout=10.0)
        except Exception:  # noqa: BLE001 — read boundary; degrade to daemon-side
            logger.warning(
                "get_history: runner fetch failed; falling back to daemon-side",
                exc_info=True)
            return None
        try:
            from shared.plugins.session.serializer import deserialize_history
            return deserialize_history(dicts)
        except Exception:  # noqa: BLE001
            logger.warning(
                "get_history: runner history deserialize failed", exc_info=True)
            return None

    def get_turn_accounting(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get turn accounting for an agent.

        ``agent_id=None`` resolves to the main agent's id.
        """
        if agent_id is None:
            agent_id = self._main_agent_id
        if agent_id in self._agents:
            return self._agents[agent_id].turn_accounting
        return []

    def get_context_usage(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get context usage for an agent.

        ``agent_id=None`` resolves to the main agent's id.
        """
        if agent_id is None:
            agent_id = self._main_agent_id
        if agent_id in self._agents:
            return self._agents[agent_id].context_usage
        return {}

    def get_available_commands(self) -> Dict[str, str]:
        """Get available commands with descriptions."""
        # Phase 3 §7c step 6.6.4.5c.2: route through runner-RPC.
        if self._runner_rpc is None:
            return {}
        try:
            user_commands = (
                self._runner_rpc.session_get_user_commands_threadsafe()
            )
        except Exception:  # noqa: BLE001 — display-only, fall back to {}
            return {}
        return {name: cmd.description for name, cmd in user_commands.items()}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools with status."""
        if not self.registry:
            return []
        return self.registry.get_tool_status()

    def get_tool_status(self) -> List[Dict[str, Any]]:
        """Get tool status for state snapshot.

        Returns list of {name, description, enabled, plugin}.
        """
        if not self.registry:
            return []
        # Use registry's tool status which includes enabled/disabled info
        return self.registry.get_tool_status()

    def get_available_models(self) -> List[str]:
        """Get available model names for completion.

        Returns list of model name strings.
        """
        # Phase 3 §7c step 6.6.4.5c.4: route through runner-RPC.
        # Get model completions for the "select" subcommand to get
        # actual model names (calling with [] returns subcommands
        # like "list", "select" instead).
        if self._runner_rpc is None:
            return []
        try:
            completions = (
                self._runner_rpc.session_get_model_completions_threadsafe(
                    ["select"],
                )
            )
            return [c.value if hasattr(c, 'value') else str(c) for c in completions]
        except Exception:
            return []

    # =========================================================================
    # Confined runner (Phase 2 §4.6)
    # =========================================================================

    def mark_runner_ready(self) -> None:
        """Signal that the per-session runner has finished ``session.bootstrap``
        and can service mid-session RPCs (the client-tool push) + the send path.

        Readiness is bootstrap-complete, NOT rpc-handle-live (see
        :meth:`set_runner_rpc`).  Called from ``dispatch_bootstrap_envelope``
        after the bootstrap RPC settles (success OR the daemon-authoritative
        failure path) so a reused warm pool slot doesn't strand the push/send on
        a readiness timeout.  Idempotent.
        """
        self._runner_ready.set()

    def set_runner_rpc(
        self,
        rpc_client: Optional["RunnerRPCClient"],
        spawned: Optional["SpawnedRunner"],
    ) -> None:
        """Stash the daemon-side RPC handle for the per-session runner.

        Called by :class:`SessionManager` after
        :meth:`RunnerSpawner.spawn` returns and BEFORE this server's
        :meth:`initialize` runs.  The cli plugin's daemon-side stub
        reads ``registry.runner_rpc`` at configure time (the
        registry-attribute injection pattern picked in plan §5.4),
        so the registry must already carry the reference when the
        plugin's ``set_plugin_registry`` hook fires.

        Args:
            rpc_client: The :class:`RunnerRPCClient` started against
                the runner's parent socket.  ``None`` means this
                session runs without a runner (Phase 2 falls back to
                in-process execution for non-apparmor sessions).
            spawned: The :class:`SpawnedRunner` handle (pid + socket).
                Stored for ``shutdown()`` so we can reap the runner
                if it doesn't exit cleanly on socket close.
        """
        self._runner_rpc = rpc_client
        self._spawned_runner = spawned
        # Runner readiness is BOOTSTRAP-complete, NOT rpc-handle-live.  A reused
        # warm pool slot's rpc handle is live the instant it's claimed, but the
        # slot can't service THIS session until its ``session.bootstrap``
        # completes — so wiring the handle CLEARS readiness; ``mark_runner_ready``
        # (called from ``dispatch_bootstrap_envelope`` after the bootstrap RPC)
        # sets it.  This closes the re-attach stall where the mid-session
        # client-tool push and the send-path gate raced ahead of the reused
        # slot's async bootstrap and hit a 15s push TimeoutError.
        self._runner_ready.clear()
        # Plumb onto the registry so plugins can consume via the
        # registry-attribute pattern (§5.4 of the Phase 2 plan).
        if self.registry is not None and rpc_client is not None:
            setattr(self.registry, "runner_rpc", rpc_client)
        # Phase 3 §7c Step 7.1: instantiate + register the
        # ``client.prompt_operator`` handler.  The daemon-side
        # infrastructure (RunnerRPCServer + bidirectional read-loop
        # dispatch) is already wired in RunnerRPCClient — Step 7.1's
        # missing piece is just creating + registering the handler.
        # The handler relays runner→daemon ASKs by emitting a
        # PermissionRequestedEvent (bound to ``self.emit``) and
        # awaiting the matching response via :meth:`resolve_response`
        # (called from :meth:`respond_to_permission` post-Step-7.3
        # rewire).
        if rpc_client is not None:
            from server.runner_rpc_handlers.prompt_operator import (
                PromptOperatorHandler,
                register as register_prompt_operator,
            )
            self._prompt_operator_handler = PromptOperatorHandler(
                emit_event=self.emit,
            )
            register_prompt_operator(
                rpc_client.rpc_server, self._prompt_operator_handler,
            )
            # Clarification relay — symmetric with prompt_operator.  Lets a
            # runner-tier (confined / pool) session deliver clarifications to
            # the connected client (the runner-side QueueChannel has no
            # daemon-wired input_queue).  See
            # server/runner_rpc_handlers/clarification_relay.py.
            from server.runner_rpc_handlers.clarification_relay import (
                ClarificationRelayHandler,
                register as register_clarification_relay,
            )
            self._clarification_relay_handler = ClarificationRelayHandler(
                emit_event=self.emit,
            )
            register_clarification_relay(
                rpc_client.rpc_server, self._clarification_relay_handler,
            )
            # Phase 4 §4.3.2: register the
            # ``subagent.spawn_isolated_runner`` handler.  Stub body
            # for now — returns "not yet implemented" with stage=spawn
            # until §4.3.3-§4.3.7 fill in the actual spawn machinery
            # (helper, sub-AppArmor profile, sub-cgroup, cross-runner
            # forwarding).  Registering early so §4.3.3-§4.3.7 land
            # against a stable surface.  Guarded by ``_session_id``
            # because the handler's confused-deputy check requires a
            # non-empty parent session id; bootstrap paths that don't
            # carry a session id (very early init / test fakes that
            # bypass ``__init__`` via ``__new__``) skip registration
            # rather than crashing.  ``getattr`` defense mirrors the
            # shutdown() pattern below.
            session_id = getattr(self, "_session_id", None)
            if session_id:
                from server.runner_rpc_handlers.spawn_isolated_runner import (
                    SpawnIsolatedRunnerHandler,
                    register as register_spawn_isolated_runner,
                )
                self._spawn_isolated_runner_handler = (
                    SpawnIsolatedRunnerHandler(
                        parent_session_id=session_id,
                    )
                )
                register_spawn_isolated_runner(
                    rpc_client.rpc_server,
                    self._spawn_isolated_runner_handler,
                )
            else:
                self._spawn_isolated_runner_handler = None
            # Register the ``daemon.plugin_execute`` handler — the
            # reverse-dispatch verb for cross-tier (``PLUGIN_TIER =
            # "daemon_callable"``) plugins.  Runner-side tool stubs
            # built via DaemonForwardingMixin route execution back
            # through this handler so the daemon-side plugin
            # instance (with its daemon-only state, e.g.
            # session_ops's SessionManager reference) runs the body.
            # See ``shared/plugins/daemon_forwarding.py`` for the
            # mixin + ``shared/plugins/CLAUDE.md`` for the full
            # cross-tier pattern.  Unguarded by session_id — the
            # handler binds to ``self`` (the server's registry) not
            # to a session_id, so it's safe even on bootstrap paths
            # that lack a session id (test fakes via ``__new__``).
            from server.runner_rpc_handlers.daemon_plugin_execute import (
                DaemonPluginExecuteHandler,
                register as register_daemon_plugin_execute,
            )
            self._daemon_plugin_execute_handler = (
                DaemonPluginExecuteHandler(server=self)
            )
            register_daemon_plugin_execute(
                rpc_client.rpc_server,
                self._daemon_plugin_execute_handler,
            )
        else:
            self._prompt_operator_handler = None
            self._clarification_relay_handler = None
            self._spawn_isolated_runner_handler = None
            self._daemon_plugin_execute_handler = None

    @property
    def runner_rpc(self) -> Optional["RunnerRPCClient"]:
        """Read accessor for the runner RPC handle."""
        return self._runner_rpc

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.registry:
            self.registry.unexpose_all()
        if self.permission_plugin:
            self.permission_plugin.shutdown()
        # Phase 3 §7c Step 7.1: tear down the prompt-operator handler
        # before the runner-RPC transport closes.  ``shutdown()``
        # cancels in-flight prompts with a clean error so any
        # runner-side awaiter sees a typed failure (not a transport
        # disconnect).  ``getattr`` for forward-compat with tests
        # that bypass ``__init__`` via ``JaatoServer.__new__``.
        prompt_handler = getattr(self, "_prompt_operator_handler", None)
        if prompt_handler is not None:
            try:
                prompt_handler.shutdown()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.exception(
                    "JaatoServer.shutdown: prompt_operator_handler "
                    "shutdown raised",
                )
            self._prompt_operator_handler = None
        # Tear down the clarification relay handler (symmetric with the
        # prompt-operator teardown above) — fails in-flight clarifications
        # with a clean error so the runner-side awaiter sees a typed cancel.
        clarif_relay = getattr(self, "_clarification_relay_handler", None)
        if clarif_relay is not None:
            try:
                clarif_relay.shutdown()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.exception(
                    "JaatoServer.shutdown: clarification_relay_handler "
                    "shutdown raised",
                )
            self._clarification_relay_handler = None
        # Phase 4 §4.3.2: tear down the spawn_isolated_runner handler.
        # Stub body holds no in-flight state (just a closed flag) so
        # this is a no-op beyond marking the handler closed; §4.3.6
        # will extend ``shutdown()`` to cascade through in-flight
        # spawn tracking.  Same ``getattr`` defense as the prompt
        # handler for test fakes.
        spawn_handler = getattr(
            self, "_spawn_isolated_runner_handler", None,
        )
        if spawn_handler is not None:
            try:
                spawn_handler.shutdown()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.exception(
                    "JaatoServer.shutdown: "
                    "spawn_isolated_runner_handler shutdown raised",
                )
            self._spawn_isolated_runner_handler = None
        # Tear down the daemon.plugin_execute handler.  Holds no
        # in-flight state beyond the closed flag (worker-thread
        # executor calls finish naturally; the plugin instances
        # they touch are owned by the server's registry, which
        # tears down on its own path).
        plugin_exec_handler = getattr(
            self, "_daemon_plugin_execute_handler", None,
        )
        if plugin_exec_handler is not None:
            try:
                plugin_exec_handler.shutdown()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.exception(
                    "JaatoServer.shutdown: "
                    "daemon_plugin_execute_handler shutdown raised",
                )
            self._daemon_plugin_execute_handler = None
        # Tear down the runner subprocess if one was spawned.  The
        # close ladder (parent EOF → wait → SIGTERM → SIGKILL) lives
        # inside ``RunnerRPCClient.close``; we run it on the daemon's
        # main loop via run_coroutine_threadsafe so this synchronous
        # ``shutdown`` doesn't block.
        rpc = self._runner_rpc
        spawned = self._spawned_runner
        pool_manager = self._pool_manager_ref
        self._runner_rpc = None
        self._runner_ready.clear()  # runner torn down — send path must await respawn
        self._spawned_runner = None
        self._pool_manager_ref = None

        # Phase 2 cascade-sharing: when the runner was served from
        # the pool, attempt to return the slot instead of closing the
        # transport.  Flow per docs/design/runner-cascade-sharing.md §4.2:
        #
        #   1. Call session.end RPC — runner fires
        #      reset_for_next_session() on every initialized plugin.
        #   2. If errors == [] — slot is fully reset; return to pool.
        #      The runner keeps running, ready for the next session
        #      of the same cascade.
        #   3. If errors != [] — slot is in undefined state; fall
        #      through to the cold close path (close transport +
        #      reap the runner).
        #
        # Only fires for pool-served sessions (``spawned.pool_slot``
        # set) AND when the pool_manager ref is wired.  Standalone /
        # cold-spawned sessions take the existing close path.
        pool_slot = getattr(spawned, "pool_slot", None) if spawned else None
        cascade_returned = False
        if rpc is not None and pool_slot is not None and pool_manager is not None:
            session_end = getattr(rpc, "session_end_threadsafe", None)
            if callable(session_end):
                try:
                    result = session_end(timeout=10.0)
                    errors = result.get("errors") if isinstance(result, dict) else None
                    if errors == []:
                        # Phase 3 cascade-sharing: stamp the session_id
                        # we just served so the NEXT session to acquire
                        # this slot can apparmor_parser --remove our
                        # apparmor profile after its own transition.
                        pool_slot.last_session_id = self._session_id
                        # Phase 3 hotfix (server 0.6.150+): the rpc
                        # client lives on the slot (asyncio transport
                        # binds the socket exclusively; can't create
                        # a fresh rpc per session).  Clear per-session
                        # state on the rpc but KEEP the transport
                        # bound to the socket.  Next session reuses
                        # this same rpc client.  See PR #173.
                        reset_method = getattr(
                            rpc, "reset_for_slot_reuse", None,
                        )
                        if callable(reset_method):
                            try:
                                reset_method()
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "JaatoServer.shutdown: rpc."
                                    "reset_for_slot_reuse raised %s — "
                                    "slot may still be usable; "
                                    "monitoring", exc,
                                )
                        pool_slot.rpc = rpc  # idempotent — first
                                              # session stashed it here;
                                              # subsequent sessions
                                              # re-affirm the binding.
                        pool_manager.return_slot_after_session(pool_slot)
                        cascade_returned = True
                        logger.info(
                            "JaatoServer.shutdown: pool slot pid=%d "
                            "returned to pool after session_end "
                            "(plugins_reset=%d cascade=%s last_session=%s; "
                            "rpc reset, transport preserved)",
                            pool_slot.pid,
                            result.get("plugins_reset", 0),
                            pool_slot.cascade_id or "(standalone)",
                            self._session_id,
                        )
                    else:
                        logger.warning(
                            "JaatoServer.shutdown: pool slot pid=%d "
                            "session_end returned errors %r — slot will "
                            "be torn down (not returned to pool)",
                            pool_slot.pid, errors,
                        )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "JaatoServer.shutdown: session_end RPC failed "
                        "(%s); falling through to runner close",
                        exc,
                    )

        # SlotSettledEvent (cascade warm-reuse handoff): emit ONCE per cascade
        # stage HERE — after the pool-return decision (``cascade_returned``
        # known) but BEFORE the warm-path early-return + the cold-close — so it
        # fires on EVERY teardown path (warm-return / pool-torn-down / cold-
        # spawn).  Gated on the session's cascade affinity.  ``was_warm`` tells
        # the reactor whether the next stage's spawn reuses the warm slot.
        # Universal + stall-proof: a cascade reactor gates the next stage on
        # this single event with no timeout.
        cascade_driver_id = getattr(self, "_cascade_driver_id", None)
        if cascade_driver_id:
            try:
                from jaato_sdk.events import SlotSettledEvent
                self.emit(SlotSettledEvent(
                    session_id=self._session_id or "",
                    agent_id=self._main_agent_id,
                    cascade_driver_id=cascade_driver_id,
                    was_warm=cascade_returned,
                    pool_slot_pid=(
                        pool_slot.pid
                        if (pool_slot is not None and cascade_returned)
                        else 0
                    ),
                    # Discriminator for the slot.settled-vs-recovery collision:
                    # an error-terminated session's stage is re-spawned by the
                    # recovery reactor, so the stage-advance reactor must SKIP it.
                    terminal_reason=getattr(self, "_terminal_reason", None),
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JaatoServer.shutdown: SlotSettledEvent emit raised %s",
                    exc,
                )

        if cascade_returned:
            # Slot is back in the pool serving the next session.  Do
            # NOT close the transport / waitpid the runner — both
            # belong to the slot now.
            return

        if rpc is not None:
            # Phase 3 §3.3c precursor: call session.shutdown FIRST so
            # the runner-side host calls close_session on the
            # bootstrapped JaatoSession (firing on_session_end hooks)
            # BEFORE we close the transport + SIGTERM the runner
            # process.  Without this, plugin teardown ran AFTER
            # process termination — file flushes / network closes
            # raced against SIGKILL.
            #
            # session_shutdown is best-effort: if no session was
            # bootstrapped (Phase 2 cli-only path) it returns the
            # empty session_id; if it raises (transport already
            # closed, runner crashed mid-call) we log + proceed to
            # close().  Keeping shutdown robust matters more than
            # the graceful-teardown improvement.
            shutdown_method = getattr(rpc, "session_shutdown_threadsafe", None)
            if callable(shutdown_method):
                try:
                    sid = shutdown_method(timeout=5.0)
                    if sid:
                        logger.debug(
                            "JaatoServer.shutdown: runner-side "
                            "session.shutdown for %s succeeded",
                            sid,
                        )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "JaatoServer.shutdown: session.shutdown RPC "
                        "failed (%s); proceeding to transport close",
                        exc,
                    )
            try:
                import asyncio
                loop = getattr(rpc, "_loop", None)
                if loop is None or not loop.is_running():
                    return
                fut = asyncio.run_coroutine_threadsafe(rpc.close(), loop)
                # Bound the wait so a stuck runner doesn't wedge
                # session shutdown — close() escalates to SIGKILL
                # internally within ~7s.
                fut.result(timeout=10.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JaatoServer.shutdown: runner-rpc close failed: %s",
                    exc, exc_info=True,
                )

    def _trace(self, msg: str) -> None:
        """Write trace message for debugging (goes to daemon log)."""
        logger.debug(msg)

    def _get_auth_plugin_for_provider(self, provider_name: str):
        """Get the authentication plugin for a provider via trait-based discovery.

        Discovery uses the plugin-level trait system:

        1. Filter plugins that declare ``TRAIT_AUTH_PROVIDER`` in their
           ``plugin_traits`` — this identifies them as auth plugins.
        2. Among those, match by ``provider_name`` property to find the
           one serving the requested provider.

        This avoids a hardcoded provider-to-plugin mapping, so adding a new
        auth plugin never requires editing server code.

        Args:
            provider_name: Provider name (e.g., 'anthropic', 'zhipuai')

        Returns:
            The auth plugin instance, or None if no matching plugin found.
        """
        from jaato_sdk.plugins.base import TRAIT_AUTH_PROVIDER

        if not self.registry:
            return None

        for name in self.registry.list_available():
            plugin = self.registry.get_plugin(name)
            if plugin is None:
                continue
            traits = getattr(plugin, 'plugin_traits', frozenset())
            if TRAIT_AUTH_PROVIDER in traits and getattr(plugin, 'provider_name', None) == provider_name:
                return plugin

        return None

    def _check_auth_completion(self) -> None:
        """Check if auth has been completed and finish initialization if so."""
        if not self._auth_pending:
            return

        self._trace(f"[auth] Checking if auth is now complete... (workspace={self._workspace_path})")

        # Try to verify auth again (use session env and workspace context for credentials)
        try:
            with self._with_session_env(), self._in_workspace():
                self._trace(f"[auth] Workspace path: {self._workspace_path}")
                # Phase 3 §7c step 6.5: read directly from ``self._runtime``.
                auth_ok = self._runtime.verify_auth(allow_interactive=False)
            if auth_ok:
                self._trace("[auth] Auth completed successfully, finishing initialization...")
                self._auth_pending = False
                self._auth_plugin_command = None

                # Complete the remaining initialization steps that were skipped
                self._emit_init_progress("Verifying authentication", "done", 4, 6)

                # Step 5: Configure tools (use session env and workspace context for plugin config)
                # Phase 3 §7c step 6.6.4.5e: ``_jaato.configure_tools()`` call
                # dropped (mirror of initialize() site).  Daemon-direct
                # runtime is already configured with plugins via 5d's post-
                # threadpool-join ``self._runtime.configure_plugins(...)``;
                # the JaatoClient session-creation half is dead-weight.
                self._emit_init_progress("Configuring tools", "running", 5, 6)
                with self._with_session_env(), self._in_workspace():
                    gc_result = load_gc_from_file(workspace_root=self._workspace_path)
                gc_threshold = None
                gc_strategy = None
                gc_target_percent = None
                gc_continuous_mode = False
                if gc_result:
                    gc_plugin, gc_config = gc_result
                    # Phase 3 §7c step 6.6.4.4: ``set_gc_plugin`` WIRING
                    # deleted (mirror of initialize() site).  GC trigger
                    # path is runner-side post-6.6.4.3b; daemon-side
                    # propagation is dead-weight.
                    gc_threshold = gc_config.threshold_percent
                    gc_target_percent = gc_config.target_percent
                    gc_continuous_mode = gc_config.continuous_mode
                    gc_strategy = getattr(gc_plugin, 'name', 'gc')
                    if gc_strategy.startswith('gc_'):
                        gc_strategy = gc_strategy[3:]

                # Phase 3 §7c step 6.6.4.3b: ``set_instruction_budget_callback``
                # wiring deleted (mirror of the initialize() site).  Recurring
                # budget updates flow runner→daemon via NotificationFrames
                # consumed by ``_build_send_message_notification_handler``.
                # Initial-budget snapshot emit below stays daemon-side — it's
                # a one-shot after configure_tools().
                # Phase 3 §7c step 6.6.4.5b: read budget snapshot via the
                # ``session.snapshot_instruction_budget`` RPC (mirror of
                # initialize() site).
                #
                # Phase 3 post-Step-7 regression fix: defensive try/except
                # wrap matching the initialize() site.  Handler may return
                # ``stage="no_session"`` during the auth-completion
                # initialize-finishing path; wrapper raises
                # ``RunnerCallError`` on ``ok=False``.
                snapshot = None
                if self._runner_rpc is not None:
                    try:
                        snapshot = (
                            self._runner_rpc
                            .session_snapshot_instruction_budget_threadsafe()
                        )
                    except Exception as exc:  # noqa: BLE001 — best-effort
                        logger.debug(
                            "auth-completion: snapshot_instruction_budget "
                            "RPC failed (%s) — skipping initial budget "
                            "emit (runner-side bootstrap may not be complete)",
                            exc,
                        )
                if snapshot:
                    self.emit(InstructionBudgetEvent(
                        agent_id=snapshot.get("agent_id", "main"),
                        budget_snapshot=snapshot,
                    ))

                self._emit_init_progress("Configuring tools", "done", 5, 6)

                # Step 6: Set up session
                self._emit_init_progress("Setting up session", "running", 6, 6)
                self._setup_session_plugin()
                self._setup_agent_hooks()
                self._setup_permission_hooks()
                self._setup_clarification_hooks()
                self._setup_reference_selection_hooks()
                self._setup_plan_hooks()
                self._setup_queue_channels()
                self._create_main_agent()
                if self._main_agent_id in self._agents and gc_threshold is not None:
                    main_state = self._agents[self._main_agent_id]
                    main_state.gc_threshold = gc_threshold
                    main_state.gc_strategy = gc_strategy
                    main_state.gc_target_percent = gc_target_percent
                    main_state.gc_continuous_mode = gc_continuous_mode

                # Emit initial context update so toolbar shows correct usage.
                # Phase 3 §7c step 6.6.4.5b: route through runner-RPC.
                if self._runner_rpc is not None:
                    usage = self._runner_rpc.session_get_context_usage_threadsafe()
                    context_limit = (
                        usage.get('context_limit')
                        or self._runner_rpc.session_get_context_limit_threadsafe()
                    )
                    self.emit(ContextUpdatedEvent(
                        agent_id=self._main_agent_id,
                        usage=self._build_usage(
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            output_tokens=usage.get('output_tokens', 0),
                            total_tokens=usage.get('total_tokens', 0),
                        ),
                        context_limit=context_limit,
                        percent_used=usage.get('percent_used', 0.0),
                        tokens_remaining=usage.get('tokens_remaining', context_limit),
                        turns=usage.get('turns', 0),
                    ))
                    self.emit(GCConfigEvent(
                        agent_id=self._main_agent_id,
                        threshold=gc_threshold,
                        strategy=gc_strategy,
                        target_percent=gc_target_percent,
                        continuous_mode=gc_continuous_mode,
                    ))

                self._emit_init_progress("Setting up session", "done", 6, 6)

                self.emit(SystemMessageEvent(
                    message="Authentication successful. Session is now ready.",
                    style="success",
                ))
                # Phase 3 §7c step 6.6.4.5c.1: route through runner-RPC.  Best-
                # effort: a transport error here just means no auth-info
                # suffix in the display message; don't propagate the failure.
                try:
                    auth_info = (
                        self._runner_rpc.session_get_auth_info_threadsafe()
                        if self._runner_rpc is not None else ""
                    )
                except Exception:  # noqa: BLE001 — display-only, fall back to ""
                    auth_info = ""
                auth_suffix = f" ({auth_info})" if auth_info else ""
                self.emit(SystemMessageEvent(
                    message=f"Connected to {self._model_provider}/{self._model_name}{auth_suffix}",
                    style="info",
                ))

                # Notify session_manager to emit session info
                if self._on_auth_complete:
                    self._on_auth_complete()
            else:
                self._trace("[auth] Auth still pending")
        except Exception as e:
            self._trace(f"[auth] Auth check failed: {e}")
            # Emit error so user knows what happened
            self.emit(SystemMessageEvent(
                message=f"Auth verification failed: {e}",
                style="error",
            ))
