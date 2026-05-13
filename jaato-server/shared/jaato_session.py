"""JaatoSession - Per-agent conversation session.

Provides isolated conversation state for an agent (main or subagent),
while sharing resources from the parent JaatoRuntime.
"""

import json
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, TYPE_CHECKING

from .message_queue import MessageQueue, QueuedMessage, SourceType
from .session_history import SessionHistory

logger = logging.getLogger(__name__)

from .ai_tool_runner import ToolExecutor
from .session_context import set_current_session
from .retry_utils import with_retry, RequestPacer, RetryCallback, RetryConfig, is_context_limit_error
from .token_accounting import TokenLedger
from jaato_sdk.plugins.base import HelpLines, UserCommand, OutputCallback
from .plugins.gc import GCConfig, GCPlugin, GCRemovalItem, GCResult, GCTriggerReason
from .plugins.gc.utils import ensure_tool_call_integrity, estimate_history_tokens
from .instruction_budget import (
    InstructionBudget,
    InstructionSource,
    estimate_tokens,
    SystemChildType,
    DEFAULT_SYSTEM_POLICIES,
    GCPolicy,
    PluginToolType,
    DEFAULT_TOOL_POLICIES,
)
from .instruction_token_cache import InstructionTokenCache
from .plugins.session import SessionPlugin, SessionConfig, SessionState, SessionInfo
from .plugins.streaming import StreamManager, StreamingCapable, StreamChunk, StreamUpdate
from .plugins.model_provider.base import UsageUpdateCallback, GCThresholdCallback
from jaato_sdk.plugins.model_provider.types import (
    Attachment,
    CancelledException,
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ThinkingConfig,
    TokenUsage,
    ToolResult,
    ToolSchema,
    TurnOutcome,
    TurnResult,
)

if TYPE_CHECKING:
    from .jaato_runtime import JaatoRuntime
    from .plugins.model_provider.base import ModelProviderPlugin
    from .plugins.subagent.ui_hooks import AgentUIHooks
    from .plugins.telemetry import TelemetryPlugin
    from .plugins.thinking import ThinkingPlugin
    from .model_tiers import ModelTierConfig

# Import framework instruction for tool result injection
from .jaato_runtime import _TASK_COMPLETION_INSTRUCTION

# Pattern to match @references in prompts
AT_REFERENCE_PATTERN = re.compile(r'@([\w./\-]+(?:\.\w+)?)')

# Rewind-with-hint budget.  How many consecutive rewinds we allow
# per logical operation before giving up and surfacing the failure
# normally.  The counter resets on any successful tool execution.
# Keep small: the point is to unstick the model once, not to loop.
# See ``docs/design/rewind-with-hint.md`` for rationale.
REWIND_BUDGET_PER_OPERATION = 2


class ActivityPhase(Enum):
    """Activity phases for tracking what a session is doing.

    Used to help parent agents/UIs understand whether a session is
    actively working vs genuinely idle.
    """
    IDLE = "idle"                        # Waiting for input, ready to receive messages
    WAITING_FOR_LLM = "waiting_for_llm"  # Request sent, awaiting cloud response
    STREAMING = "streaming"              # Receiving tokens from LLM
    EXECUTING_TOOL = "executing_tool"    # Running a tool


@dataclass
class _ToolExecutionResult:
    """Result of executing a single tool, used for parallel execution."""
    fc: FunctionCall
    executor_result: Tuple[bool, Any]
    start_time: datetime
    end_time: datetime
    success: bool
    error_message: Optional[str]
    plugin_type: str


@dataclass
class _PinnedReference:
    """A preselected reference whose content has been read and pinned.

    When the model reads a file matching a preselected reference's
    resolved_path, the content is captured here and appended to the
    system instruction.  This ensures the reference content survives
    garbage collection — the original tool result in conversation history
    can be freely GC'd (EPHEMERAL/WORKING) while the pinned copy lives
    in the system instruction (LOCKED under SYSTEM.SELECTED_REFERENCES).

    Attributes:
        ref_id: Reference source ID from the catalog.
        ref_name: Human-readable reference name.
        content: The captured file content (tool result text).
        pinned_at: Unix timestamp when the content was pinned.
    """
    ref_id: str
    ref_name: str
    content: str
    pinned_at: float


@dataclass
class _TokenCountRequest:
    """A pending token-count request for a single instruction text.

    Used during two-phase instruction budget population: Phase 1 resolves
    counts from cache or estimates, Phase 2 refines cache misses via
    background ``provider.count_tokens()`` calls.
    """
    text: str
    source: InstructionSource
    child_key: str
    gc_policy: GCPolicy
    label: str
    token_count: int = 0
    is_estimate: bool = False


@dataclass
class _CancellationResult:
    """Result of a cancellation/mid-turn interrupt check.

    Replaces the out-of-band ``_mid_turn_continuation_response`` mutable
    field pattern. Callers inspect ``action`` to decide their next step:

    - ``"continue"``: Not cancelled; keep processing normally.
    - ``"end_turn"``: Turn should end; ``turn_result`` has the outcome.
    - ``"switch_response"``: A mid-turn interrupt produced a new response;
      continue the loop with ``new_response``.

    Attributes:
        action: What the caller should do next.
        turn_result: Present when ``action == "end_turn"``. The
            ``TurnResult`` to return from the chat loop.
        new_response: Present when ``action == "switch_response"``. The
            fresh ``ProviderResponse`` from the mid-turn prompt.
    """
    action: Literal["continue", "end_turn", "switch_response"]
    turn_result: Optional[TurnResult] = None
    new_response: Optional[ProviderResponse] = None


class JaatoSession:
    """Per-agent conversation session.

    A session represents an isolated conversation with its own:
    - Model selection
    - Tool configuration (can be a subset of runtime's tools)
    - Conversation history
    - System instructions
    - Turn accounting

    Sessions share the runtime's resources (registry, permissions, ledger)
    but maintain independent state.

    Usage:
        # Created via runtime.create_session()
        session = runtime.create_session(
            model="gemini-2.5-flash",
            tools=["cli", "web_search"],
            system_instructions="You are a research assistant."
        )

        # Use the session
        response = session.send_message("Search for Python tutorials")
        history = session.get_history()
    """

    def __init__(
        self,
        runtime: 'JaatoRuntime',
        model: str,
        provider_name: Optional[str] = None,
        agent_id: str = "main",
    ):
        """Initialize a session.

        Note: Use runtime.create_session() instead of calling this directly.

        Args:
            runtime: Parent JaatoRuntime providing shared resources.
            model: Model name to use for this session.
            provider_name: Optional provider override for cross-provider sessions.
                          If specified, uses a different AI provider than the runtime default.
            agent_id: Logical agent identifier (e.g. ``"main"``,
                ``"discovery"``, ``"coordinator"``).  Sets
                ``self._agent_id`` at construction time so consumers
                that key on agent identity (``AgentCompletedEvent``,
                reactor where-clauses, telemetry spans) see the
                correct value.  Defaults to ``"main"`` for backward
                compat with callers that don't yet thread the field.
                Post-construction the only mutator is
                :meth:`set_ui_hooks` (overwrites with the daemon's
                resolved id when ui_hooks attach).
        """
        self._runtime = runtime
        self._model_name = model
        self._provider_name_override = provider_name
        # Session-level workspace path override.  When set, this session
        # operates against a different workspace than the runtime's
        # default (e.g. a worktree snapshot for fork-replay).
        self._workspace_path: Optional[str] = None
        # Shape 3 PR 1: per-session resolved env (workspace ``.env`` +
        # profile env + overrides, expanded and secret-URI-resolved).
        # Populated by runner-side ``bootstrap_session`` AFTER the
        # session is constructed.  Mirrors the daemon-side
        # ``JaatoServer._session_env`` attribute — the runner-side
        # analog where the resolution naturally belongs once Shape 3
        # PR 4 removes the daemon-side surface.  Empty dict before
        # population; readers use :meth:`get_session_env`.
        self._session_env: Dict[str, str] = {}
        # AppArmor confine-context factory (server 0.6.50+).  Set by
        # ``JaatoRuntime.create_session`` from
        # ``runtime._confine_context_factory``.  When set, ``configure()``
        # wraps dynamic-instructions expansion in the returned context
        # manager so prefetch scripts run inside the session's
        # confinement (closes the policy-write-leak on ``.jaato`` for
        # prefetch).  ``None`` = no confinement applies.
        self._confine_context_factory: Optional[Callable] = None
        # Profile-declared JSON Schema for signal_completion's payload parameter.
        # Either an inline dict or a string path resolved via
        # .jaato/completion_schemas/. ``LifecycleTools`` consults this field at
        # construction time; when present the legacy ``summary: str`` parameter
        # is replaced with a typed ``payload: <schema>``. None = legacy untyped.
        self._completion_payload_schema: Optional[Any] = None

        # Profile-declared output artefacts.  Each entry is a
        # ``CompletionArtifact`` (renderer / output / on_error) — when
        # ``signal_completion`` validates against
        # ``_completion_payload_schema``, ``LifecycleTools`` runs each
        # renderer in turn, passing the validated payload, and writes
        # the result to the templated output path.  Empty list = legacy
        # behaviour (agents call ``writeNewFile`` themselves).
        # ``CompletionArtifact`` typed as ``Any`` here to avoid a top-
        # level subagent-config import; concrete type is
        # ``shared.plugins.subagent.config.CompletionArtifact``.
        self._completion_artifacts: List[Any] = []

        # Completion lifecycle tracking — flipped True by
        # ``LifecycleTools._execute_signal_completion`` on the first
        # successful invocation.  The completion-nudge guard reads this
        # at loop-exit (top-level: ``core.py`` model_thread finally;
        # subagent: end of ``_run_subagent_async``) to decide whether
        # to inject a nudge prompt back into the session asking the
        # agent to call ``signal_completion`` before terminating.
        # ``_completion_nudges_fired`` bounds the retry budget.
        self._signal_completion_called: bool = False
        self._completion_nudges_fired: int = 0

        # Per-turn model-tier config.  ``_tier_config`` is the resolved
        # view (built from profile.tiers or env vars).  ``_active_tier``
        # tracks which tier the session is currently operating in;
        # mutated by the ``enter_tier`` lifecycle tool, consulted by
        # provider model selection and by system-instruction assembly.
        # Both ``None`` means single-model mode — no ``enter_tier`` tool
        # is registered, no system-prompt augmentation, the provider
        # uses the legacy ``self._model_name``.
        self._tier_config: Optional['ModelTierConfig'] = None
        self._active_tier: Optional[str] = None

        # Spawn-time parameters passed to this session by the caller
        # (typically ``spawn_subagent(agent_params={...})``).  Carried
        # through to dynamic-instructions render scripts as
        # ``RenderContext.agent_params`` so they can read forwarded
        # ``case_data`` and other per-spawn fields without parsing
        # the prompt text.  Empty for top-level sessions whose prompt
        # carries case data inline.  See
        # ``shared/dynamic_instructions.py``.
        self._agent_params: Dict[str, Any] = {}

        # Provider for this session.  Lazy-initialized on first
        # model use via :meth:`_ensure_provider` (deferred from
        # ``configure()`` per the 2026-05-13 bootstrap-latency design
        # at ``docs/design/runner_prewarm_pool_plan.md`` §3.5).  The
        # 9s zhipuai INIT / multi-second anthropic INIT shifts off
        # the bootstrap RPC critical path; first model call wears
        # the cost, which is invisible under the existing streaming
        # spinner.  ``None`` after configure() finishes; populated
        # on first ``send_message`` / first BUDGET_BG token-count
        # refinement attempt, whichever fires first.
        self._provider: Optional['ModelProviderPlugin'] = None
        # Pending provider-creation args, stashed by ``configure()``
        # for ``_ensure_provider()`` to consume on first use.  Set
        # to ``None`` for skip_provider (auth-pending) mode where
        # the provider truly never gets created here.
        self._provider_lazy_pending: Optional[Dict[str, Any]] = None
        # Serializes concurrent first-use _ensure_provider() calls
        # (e.g., BUDGET_BG thread + send_message racing on a fresh
        # session).  Once the provider exists, the lock's hot path
        # is just an "already initialized" check.
        self._provider_init_lock = threading.Lock()
        # True iff ``configure()`` finished its work successfully.
        # Decoupled from ``_provider is not None`` because the
        # provider is now lazy; ``is_configured`` checks this flag
        # instead.
        self._configured: bool = False

        # Canonical conversation history owned by the session.
        # Phase 1: synced from provider after each provider operation.
        # Phase 2+: session is sole owner; provider receives messages
        # as parameters to stateless complete().
        self._history = SessionHistory()

        # Session-attached state — opaque per-key storage that
        # extensions persist alongside the journal and that fork
        # primitives carry across to the new session.  Values must be
        # JSON-serialisable (extensions encrypt before attach if they
        # need confidentiality — the framework treats values as
        # opaque).  Two write modes coexist:
        #
        # ``_session_state`` — explicit values pushed via
        # ``set_session_state(key, value)``.  Right shape for static
        # or rarely-mutated state (audit chain head, version markers).
        #
        # ``_state_providers`` — callbacks registered via
        # ``register_session_state_provider(key, fn)`` that the
        # framework invokes at journal-save / waypoint-snapshot /
        # fork-snapshot time to obtain the current value.  Right
        # shape for incrementally-mutated state (e.g.
        # pseudonymization lookup table that grows turn-by-turn) where
        # forcing the consumer to re-attach after every mutation
        # would scatter the persistence concern across every mutation
        # site.  A registered provider takes precedence over any
        # value previously set via ``set_session_state`` for the
        # same key.
        self._session_state: Dict[str, Any] = {}
        self._state_providers: Dict[str, Callable[[], Any]] = {}

        # Session always owns history and uses stateless provider.complete().
        # Legacy send_message()/send_tool_results() path removed in Phase 4.

        # Tool configuration
        self._executor: Optional[ToolExecutor] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._system_instruction: Optional[str] = None
        # Per-session AppArmor reference-fragment authorizer.  Set by
        # JaatoServer.set_reference_authorizer() after WS provisions an
        # AppArmor profile.  ``None`` means no kernel layer to mutate
        # — the references plugin operates at the application layer
        # (sandbox_manager) only.  Plugins access this via
        # get_reference_authorizer() rather than touching the slot.
        self._reference_authorizer = None
        # Phase 3 §7c step 6.1: bool flag mirror of the authorizer
        # for the runner-side seat.  When the daemon-side _jaato is
        # removed (step 6.6), the runner-side session reads this
        # via :meth:`is_reference_authorization_enabled` and uses
        # the ``apparmor.add_reference_fragment`` runner→daemon RPC
        # to authorize paths.  Set via
        # :meth:`set_reference_authorization_enabled` (called by
        # the new ``session.set_reference_authorizer`` RPC handler).
        self._reference_authorization_enabled: bool = False
        # The active override (passed via configure()) — None means the
        # assembled pipeline output is sent on the wire; "" means no
        # system message at all; non-empty replaces the assembly entirely.
        # Stored so _populate_instruction_budget can compute an honest
        # budget without having to be re-passed the value through every
        # call site.
        self._system_instruction_override: Optional[str] = None
        # Partial-suppression flag — drop the BASE layer only.  Keeps
        # agent / plugin / framework layers in play.  Ignored when
        # _system_instruction_override is set (override wins).
        self._suppress_base_instructions: bool = False
        self._tool_plugins: Optional[List[str]] = None  # Plugin names for this session

        # Per-turn token accounting
        self._turn_accounting: List[Dict[str, int]] = []

        # Instruction budget tracking (token usage by source layer)
        self._instruction_budget: Optional[InstructionBudget] = None

        # User commands for this session
        self._user_commands: Dict[str, UserCommand] = {}

        # Context garbage collection
        self._gc_plugin: Optional[GCPlugin] = None
        self._gc_config: Optional[GCConfig] = None
        self._gc_history: List[GCResult] = []

        # Cache control plugin (provider-specific caching strategy)
        self._cache_plugin: Optional[Any] = None  # CachePlugin protocol

        # Thinking mode
        self._thinking_plugin: Optional['ThinkingPlugin'] = None

        # Session persistence
        self._session_plugin: Optional[SessionPlugin] = None
        self._session_config: Optional[SessionConfig] = None

        # Agent type context (for permission checks)
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._telemetry_spans_started: bool = False

        # UI hooks for agent lifecycle events
        self._ui_hooks: Optional['AgentUIHooks'] = None
        # Logical agent identifier — sourced from the constructor
        # arg so the envelope's resolved agent_id (``"discovery"``,
        # ``"coordinator"``, ...) lands at construction.  Previously
        # hardcoded ``"main"`` here and only mutated by
        # ``set_ui_hooks``, but the runner-side bootstrap bypasses
        # ``set_ui_hooks`` (installs a notification-shim via direct
        # attribute write at rpc.py:3178-3185).  Result pre-fix: every
        # runner-side session carried ``_agent_id="main"`` regardless
        # of the daemon's ``--agent <name>`` resolution.
        self._agent_id: str = agent_id
        self._daemon_session_id: Optional[str] = None  # Session manager ID for telemetry correlation

        # Retry notification callback (client-configurable)
        self._on_retry: Optional[RetryCallback] = None

        # Request pacing (proactive rate limiting)
        # Reads AI_REQUEST_INTERVAL from env (default: 0 = disabled)
        self._pacer = RequestPacer()

        # Cancellation support
        self._cancel_token: Optional[CancelToken] = None
        self._parent_cancel_token: Optional[CancelToken] = None  # For parent→child propagation
        self._is_running: bool = False
        self._use_streaming: bool = True  # Enable streaming by default if provider supports it

        # Activity phase tracking (for parent agents/UIs to understand what we're doing)
        self._activity_phase: ActivityPhase = ActivityPhase.IDLE
        self._phase_started_at: Optional[datetime] = None
        # Disable model notifications about cancellation by default - they cause
        # the model to hallucinate "interruptions" on subsequent turns
        self._notify_model_on_cancel: bool = False

        # Proactive GC tracking
        self._gc_threshold_crossed: bool = False  # Set when threshold crossed during streaming

        # Token count cache for conversation budget updates.
        # Maps message_id -> token count. Since message content is immutable
        # once added to history, cached counts never go stale. This avoids
        # O(N) network API calls to count_tokens on every budget rebuild.
        self._msg_token_cache: Dict[str, int] = {}
        self._gc_threshold_callback: Optional[GCThresholdCallback] = None

        # Terminal width for formatting (used by enrichment notifications)
        self._terminal_width: int = 80

        # Presentation context describing the client's display constraints
        # and capabilities (width, markdown support, expandable content, etc.).
        # Injected into system instructions so the model adapts its output.
        # Set via set_presentation_context() when the client connects.
        self._presentation_context: Optional['PresentationContext'] = None

        # Tracks plugin names whose system instructions were deferred because
        # they had no core tools at session start.  When the model activates
        # a tool from one of these plugins, the instructions are injected into
        # the system prompt and the budget.
        self._deferred_plugin_instructions: Set[str] = set()
        self._preloaded_plugins: set = set()

        # Priority-aware message queue for agent communication
        # Uses double-linked list for efficient mid-queue removal of parent messages
        # Parent/user messages: processed mid-turn (high priority)
        # Child messages: processed when idle (lower priority)
        self._message_queue: MessageQueue = MessageQueue()

        # Parent session for output forwarding (subagent -> parent visibility)
        # When set, all output events are forwarded to parent's injection queue
        self._parent_session: Optional['JaatoSession'] = None

        # Callback when a prompt is injected (processed from queue)
        # Used by server to emit MidTurnPromptInjectedEvent
        self._on_prompt_injected: Optional[Callable[[str], None]] = None

        # Callback when streaming is interrupted for mid-turn prompt
        # Used by server to emit MidTurnInterruptEvent
        # Callback receives (partial_response_chars, user_prompt_preview)
        self._on_mid_turn_interrupt: Optional[Callable[[int, str], None]] = None

        # Callback when continuation is needed (child messages received while idle)
        # Used by server to trigger a new turn when subagent sends messages
        # Callback receives the collected child message text as argument
        self._on_continuation_needed: Optional[Callable[[str], None]] = None

        # Callback when the session transitions between idle and non-idle.
        # Fires on the first non-IDLE phase after IDLE (is_active=True) and
        # when returning to IDLE from a non-IDLE phase (is_active=False).
        # Used by the subagent plugin to drive AgentStatusChangedEvents
        # so that the UI tab bar spinner starts/stops automatically.
        self._on_running_state_changed: Optional[Callable[[bool], None]] = None

        # Current output callback for this turn (used by enrichment to route notifications)
        # Stored here so _enrich_tool_result_dict can pass it to registry.enrich_tool_result()
        # This ensures enrichment notifications go to the correct agent panel even when
        # multiple sessions share the same registry (e.g., subagents)
        self._current_output_callback: Optional['OutputCallback'] = None

        # Current turn span — set while a turn is in progress so that
        # _enrich_tool_result_dict can emit enrichment telemetry events on it.
        self._current_turn_span = None

        # Callback when instruction budget is updated
        # Used by server to emit InstructionBudgetEvent
        # Callback receives the budget snapshot dict
        self._on_instruction_budget_updated: Optional[Callable[[Dict[str, Any]], None]] = None

        # Turn counter for telemetry
        self._turn_index: int = 0

        # Turn complexity tracking for GC policy classification
        # Tracks whether the current turn is "complex" (multiple model responses with tool calls)
        self._turn_model_response_count: int = 0
        self._turn_had_tool_calls: bool = False

        # Rewind-with-hint state.  Counter increments each time the
        # rewind detector fires for a MAX_TOKENS-truncated tool call
        # (see ``shared/rewind.py``); resets on any successful tool
        # execution so a healthy session is never starved of future
        # rewinds.  Capped at ``REWIND_BUDGET_PER_OPERATION`` to prevent
        # a persistently-failing model from looping.
        self._rewind_count: int = 0

        # Background thread for Phase 2 instruction token counting.
        # Set by _start_background_token_counting(), joined before GC.
        self._budget_counting_thread: Optional[threading.Thread] = None

        # Pinned preselected references: content captured when the model
        # reads a file matching a preselected reference's resolved_path.
        # Keyed by ref_id.  Pinned content is appended to the system
        # instruction (LOCKED under SYSTEM.SELECTED_REFERENCES) so it
        # survives GC, while the original tool result in conversation
        # history remains EPHEMERAL and can be freely collected.
        self._pinned_references: Dict[str, _PinnedReference] = {}

        # Provider exclusion for replay_messages() — serializes
        # provider.complete() calls so an external replay/fork cannot run
        # concurrently with the session's own provider call.
        # _fork_gate is open by default (session runs freely);
        # replay_messages() clears it to block the session's next provider call.
        # _provider_idle is set when no provider call is in flight;
        # replay_messages() waits on it before running its own call.
        self._fork_gate = threading.Event()
        self._fork_gate.set()
        self._provider_idle = threading.Event()
        self._provider_idle.set()

        # Observation pause — blocks new send_message() calls while an
        # external observer is inspecting the session's workspace or
        # snapshotting its state.  Open (set) by default.
        self._observation_lock = threading.Event()
        self._observation_lock.set()
        # Fires whenever a turn finishes (_is_running goes False).
        # Used by pause_for_observation to wait for an in-progress turn.
        self._turn_complete = threading.Event()
        self._turn_complete.set()  # No turn in progress initially.

        # Streaming tool support
        self._stream_manager: Optional[StreamManager] = None
        # Timeout for waiting on streaming updates when model is idle (seconds)
        self._streaming_wait_timeout: float = 5.0

    @property
    def _telemetry(self) -> 'TelemetryPlugin':
        """Get telemetry plugin from runtime."""
        return self._runtime.telemetry

    def _ensure_telemetry_spans(self) -> None:
        """Lazily start the session and agent telemetry spans.

        Called on the first turn rather than in ``set_agent_context``
        because the parent session reference may not be set yet when
        ``set_agent_context`` runs (subagents call it before
        ``set_parent_session``).

        Uses the main session's ``_agent_id`` as the root session span
        so that all agents (main + subagents) share one session tree.
        """
        if self._telemetry_spans_started:
            return
        self._telemetry_spans_started = True

        telemetry = self._telemetry
        # Determine the root session ID: use the main session's agent_id
        # so all agents tree under one session span.
        parent = self._parent_session
        session_root = self._agent_id
        while parent is not None:
            session_root = parent._agent_id
            parent = getattr(parent, '_parent_session', None)

        # Resolve daemon session ID: walk up to root session to find it.
        daemon_sid = self._daemon_session_id
        if not daemon_sid:
            p = self._parent_session
            while p is not None:
                if p._daemon_session_id:
                    daemon_sid = p._daemon_session_id
                    break
                p = getattr(p, '_parent_session', None)

        extra_attrs = {}
        if daemon_sid:
            extra_attrs["jaato.session_id"] = daemon_sid

        telemetry.begin_session(session_root, attributes=extra_attrs or None)
        telemetry.begin_agent(
            session_id=session_root,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            agent_type=self._agent_type,
            attributes=extra_attrs or None,
        )

    def set_terminal_width(self, width: int) -> None:
        """Set the terminal width for formatting.

        This affects enrichment notification formatting.

        Args:
            width: Terminal width in columns.
        """
        self._terminal_width = width

    def set_confine_context_factory(
        self, factory: Optional[Callable],
    ) -> None:
        """Set the AppArmor confine-context factory (server 0.6.50+).

        ``configure()`` wraps the dynamic-instructions expansion
        (``{{!py:...}}`` / ``{{!py?:...}}``) in the returned context
        manager so prefetch scripts run inside the session's
        AppArmor profile.  Closes the gap where prefetch ran
        unconfined and could write to ``.jaato`` regardless of the
        deny rules in the profile (R2 of the option-2-phased sandbox
        refactor).

        Set by :meth:`JaatoRuntime.create_session` from the runtime's
        own confine-context factory.  ``None`` clears (no confinement
        applies).

        Args:
            factory: Zero-arg callable returning a context manager,
                or ``None``.
        """
        self._confine_context_factory = factory

    def set_presentation_context(self, ctx: 'PresentationContext') -> None:
        """Set the presentation context describing client display capabilities.

        Updates both the stored context and ``_terminal_width`` (for backwards
        compatibility with code that reads the width directly).  The context is
        used by ``get_system_instructions()`` to inject a display-context block
        into the model's system prompt.

        Args:
            ctx: Presentation context from the connected client.
        """
        from jaato_sdk.plugins.model_provider.types import PresentationContext  # noqa: F811
        self._presentation_context = ctx
        self._terminal_width = ctx.content_width
        # Server 0.6.62+: re-apply the lifecycle-tools interactive-root
        # filter now that ``_presentation_context`` is known.  At
        # configure() time the context wasn't yet set (it arrives via
        # ``_apply_client_config_to_server`` AFTER configure runs), so
        # the filter defaulted to "expose" — leaking
        # ``signal_completion`` into the tool surface for what we now
        # know is an interactive root session.  This call removes it
        # post-hoc when the filter says so; for non-interactive
        # contexts (api root, subagent), the call is a no-op.
        self._reapply_lifecycle_tool_filter()

    def _reapply_lifecycle_tool_filter(self) -> None:
        """Re-apply the LifecycleTools interactive-root filter post-config.

        Server 0.6.62+: ``configure()`` registers signal_completion + its
        executor + auto-approval whitelist BEFORE
        ``_apply_client_config_to_server`` runs, which means
        ``_presentation_context`` is not yet known at registration time
        and the filter's "expose by default when unknown" path lets
        signal_completion through.  After ``set_presentation_context``
        sets the context, we re-run the filter and remove
        signal_completion from each surface (schema, executor,
        permission whitelist) when the filter now says it should be
        hidden.

        No-op when the filter says signal_completion should remain
        exposed (api-client root sessions, subagents).
        """
        if self._lifecycle_tools is None:
            return
        if not self._lifecycle_tools._should_hide_signal_completion():
            return
        # Filter wants signal_completion hidden — strip it from the
        # two surfaces that determine accessibility:
        #   (1) self._tools (the model's visible tool schema list)
        #   (2) self._executor._map (the dispatch table)
        # Permission whitelist isn't actively pruned — the tool not
        # being in the schema or executor is sufficient; a stale
        # whitelist entry for a tool that doesn't exist anywhere is
        # harmless.
        self._tools = [t for t in self._tools if t.name != "signal_completion"]
        if "signal_completion" in getattr(self._executor, '_map', {}):
            del self._executor._map["signal_completion"]

    def _get_trace_prefix(self) -> str:
        """Get the trace prefix including agent context."""
        if self._agent_type == "main":
            return "session:main"
        elif self._agent_name:
            return f"session:subagent:{self._agent_name}"
        else:
            return f"session:subagent:{self._agent_id}"

    def _trace(self, msg: str) -> None:
        """Write trace message to the provider trace log.

        Uses ``provider_trace()`` which applies per-agent routing via
        ContextVar so subagent session traces go to agent-specific files
        (e.g. ``provider_trace_subagent_1.log``).
        """
        from shared.trace import provider_trace
        prefix = self._get_trace_prefix()
        provider_trace(prefix, msg)

    @property
    def model_name(self) -> Optional[str]:
        """Get the model name for this session."""
        return self._model_name

    @property
    def runtime(self) -> 'JaatoRuntime':
        """Get the parent runtime."""
        return self._runtime

    @property
    def is_configured(self) -> bool:
        """Check if session is configured and ready.

        Decoupled from ``self._provider`` since 2026-05-13 (deferred
        provider INIT design).  A configured session may not yet have
        constructed its provider — that happens lazily on first model
        use via :meth:`_ensure_provider`.  ``is_configured`` reflects
        whether ``configure()`` finished its work.
        """
        return self._configured

    @property
    def agent_id(self) -> str:
        """Get the agent ID for this session.

        Returns:
            The unique agent ID (e.g., "main", "subagent_1", etc.)
        """
        return self._agent_id

    @property
    def instruction_budget(self) -> Optional[InstructionBudget]:
        """Get the instruction budget for this session.

        Returns:
            The instruction budget tracking token usage by source layer,
            or None if not yet populated.
        """
        return self._instruction_budget

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None
    ) -> None:
        """Set the agent context for permission checks and trace identification.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent (e.g., profile name).
        """
        self._agent_type = agent_type
        self._agent_name = agent_name

        # Update executor permission context if already configured
        if self._executor and self._runtime.permission_plugin:
            context = {"agent_type": agent_type, "session_id": self._daemon_session_id}
            if agent_name:
                context["agent_name"] = agent_name
            self._executor.set_permission_plugin(
                self._runtime.permission_plugin,
                context=context
            )

        # Propagate agent context to provider for trace identification
        if self._provider and hasattr(self._provider, 'set_agent_context'):
            self._provider.set_agent_context(
                agent_type=agent_type,
                agent_name=agent_name,
                agent_id=self._agent_id
            )

        # Telemetry session/agent spans are started lazily on the first
        # turn (see _ensure_telemetry_spans) because the parent session
        # reference may not be set yet when set_agent_context is called.

    def set_daemon_session_id(self, session_id: str) -> None:
        """Set the daemon session manager ID for telemetry correlation.

        This ID (e.g. ``"20260328_204308"``) is emitted as the
        ``jaato.session_id`` span attribute so Phoenix traces can be
        correlated back to jaato session manager sessions.

        Args:
            session_id: The session manager's session ID.
        """
        self._daemon_session_id = session_id

    def set_ui_hooks(
        self,
        hooks: 'AgentUIHooks',
        agent_id: str
    ) -> None:
        """Set UI hooks for agent lifecycle events.

        This enables rich terminal UIs to track tool execution and other
        lifecycle events for this session.

        Args:
            hooks: Implementation of AgentUIHooks protocol.
            agent_id: Unique identifier for this agent (e.g., "main", "subagent_1").
        """
        self._ui_hooks = hooks
        self._agent_id = agent_id

        # Update instruction budget's agent_id if it was already created
        # (configure() creates the budget with default "main" agent_id)
        if self._instruction_budget:
            self._instruction_budget.agent_id = agent_id

    def set_retry_callback(self, callback: Optional[RetryCallback]) -> None:
        """Set callback for retry notifications.

        Clients can use this to control how retry messages are delivered:
        - Simple interactive client: Don't set (uses console print)
        - Rich client: Set callback to route to queue/status bar/etc.

        Args:
            callback: Function called on each retry attempt.
                Signature: (message: str, attempt: int, max_attempts: int, delay: float) -> None
                Set to None to revert to console output.

        Example:
            # Route retries to a queue for non-disruptive display
            session.set_retry_callback(
                lambda msg, att, max_att, delay: status_queue.put(msg)
            )
        """
        self._on_retry = callback

    def set_instruction_budget_callback(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> None:
        """Set callback for instruction budget updates.

        Called when the instruction budget changes (e.g., after configure(),
        on conversation changes).

        Args:
            callback: Function called with the budget snapshot dict.
                Set to None to disable notifications.
        """
        self._on_instruction_budget_updated = callback

    def set_parent_session(self, parent: Optional['JaatoSession']) -> None:
        """Set parent session for output forwarding.

        When set, all output events from this session are forwarded to the
        parent session's injection queue. This enables parent agents to
        monitor and react to their subagents' activities in real-time.

        The forwarding is one level only - each parent sees only its
        direct children, not grandchildren.

        Args:
            parent: The parent session to forward output to, or None to disable.

        Example:
            # In SubagentPlugin when creating a subagent session:
            subagent_session.set_parent_session(self._parent_session)
        """
        self._parent_session = parent

    def set_prompt_injected_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback for when a prompt is processed from the injection queue.

        This callback is invoked when a queued prompt is about to be processed
        (injected into the conversation). The server uses this to emit
        MidTurnPromptInjectedEvent to notify the client UI.

        Args:
            callback: Function called with the prompt text when injected.
        """
        self._on_prompt_injected = callback

    def set_continuation_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback for when continuation is needed after child messages.

        This callback is invoked when child messages are drained while the
        session is idle. The server uses this to trigger a new turn so the
        model can react to subagent completion/error events.

        Args:
            callback: Function called with collected child message text.
        """
        self._on_continuation_needed = callback

    def set_running_state_callback(
        self,
        callback: Optional[Callable[[bool], None]]
    ) -> None:
        """Set callback for when the session transitions between idle and non-idle.

        The callback fires when ``_set_activity_phase`` moves the session from
        ``IDLE`` to any working phase (``is_active=True``) or back to ``IDLE``
        (``is_active=False``).  The subagent plugin uses this to emit
        ``AgentStatusChangedEvent`` so the UI tab-bar spinner starts/stops
        automatically whenever the session processes a message.

        Args:
            callback: Function called with ``True`` when the session starts
                      processing and ``False`` when it becomes idle.
        """
        self._on_running_state_changed = callback

    def set_mid_turn_interrupt_callback(
        self,
        callback: Optional[Callable[[int, str], None]]
    ) -> None:
        """Set callback for when streaming is interrupted for mid-turn prompt.

        This callback is invoked when the model's streaming generation is
        interrupted because a user prompt arrived. The server uses this to
        emit MidTurnInterruptEvent to notify the client UI.

        Args:
            callback: Function called with (partial_response_chars, user_prompt_preview)
                      when streaming is interrupted.
        """
        self._on_mid_turn_interrupt = callback

    def get_auth_info(self) -> str:
        """Return a description of the credential source the session's
        provider is using.

        Phase 3 §7c step 6.6.4.5c.1.  Public wrapper that surfaces
        the underlying ``ModelProviderPlugin.get_auth_info()`` value
        without daemon-side callers reaching into the private
        ``self._provider`` attr — required for the runner-RPC seat-
        flip where the JaatoSession lives in a separate process.

        Returns a human-readable string like ``"API key from
        ~/.jaato/zhipuai_auth.json"`` or ``"PKCE OAuth"``.  Empty
        string when no provider is attached (pre-:meth:`configure`)
        or the provider doesn't implement ``get_auth_info``.

        Closes the missing-method gap caught by the §7c step
        6.6.4.5c.0 audit (commit a88676ca).
        """
        if self._provider and hasattr(self._provider, 'get_auth_info'):
            try:
                return str(self._provider.get_auth_info() or "")
            except Exception:  # noqa: BLE001 — best-effort display string
                return ""
        return ""

    def try_completion_nudge(self, max_nudges: int) -> Tuple[bool, int]:
        """Atomic check-and-increment for the completion-nudge guard.

        Phase 3 §7c step 6.6.4.3a.  Collapses three private-state
        reaches (``_signal_completion_called`` read,
        ``_completion_nudges_fired`` read, ``_completion_nudges_fired``
        increment) into one method so daemon-side callers don't need
        direct private-attr access — required for the runner-RPC
        seat-flip in §7c step 6.6.4.3b where the JaatoSession lives
        in a separate process.

        Decision: returns ``(True, n+1)`` when a nudge should fire
        (agent didn't call ``signal_completion`` AND the budget isn't
        exhausted) — also bumps the counter atomically so callers
        don't need to do it.  Returns ``(False, current)`` otherwise
        (no counter change).

        Args:
            max_nudges: Bound on ``_completion_nudges_fired``.
                Caller's nudge-budget knob (the existing daemon-side
                site uses ``MAX_COMPLETION_NUDGES = 2``).  Must be
                non-negative; values <= 0 always yield
                ``(False, current)``.

        Returns:
            ``(should_nudge, nudges_fired_after_this_call)``.
            ``nudges_fired_after_this_call`` reflects the
            post-increment value when ``should_nudge`` is True;
            otherwise it's the unchanged current count.

        Thread-safety: matches the existing private-attr access
        pattern — the model_thread is the sole writer at the loop
        boundary.  No additional locking added (would be a behavior
        change).
        """
        if (
            not getattr(self, "_signal_completion_called", False)
            and getattr(self, "_completion_nudges_fired", 0) < max_nudges
        ):
            self._completion_nudges_fired += 1
            return True, self._completion_nudges_fired
        return False, getattr(self, "_completion_nudges_fired", 0)

    def inject_prompt(
        self,
        text: str,
        source_id: Optional[str] = None,
        source_type: Optional[SourceType] = None
    ) -> None:
        """Inject a prompt into this agent's queue.

        The prompt will be processed based on priority:
        - Parent/user/system messages: processed mid-turn (high priority)
        - Child messages: processed when agent becomes idle (lower priority)

        This method is thread-safe and can be called from any thread.

        Args:
            text: The prompt text to inject.
            source_id: ID of the sender (defaults to "unknown").
            source_type: Type of sender for priority (defaults to USER for
                        backward compatibility with existing callers).

        Example:
            # User input to main agent
            session.inject_prompt("What's the status?", source_id="user", source_type=SourceType.USER)

            # Parent sending guidance to subagent
            subagent_session.inject_prompt(
                "Focus on the authentication module",
                source_id="main",
                source_type=SourceType.PARENT
            )

            # Subagent returning result to parent
            parent_session.inject_prompt(
                "[SUBAGENT agent_id=researcher event=COMPLETED]\\nFound 3 issues",
                source_id="researcher",
                source_type=SourceType.CHILD
            )
        """
        # Default to USER for backward compatibility
        actual_source_id = source_id or "unknown"
        actual_source_type = source_type or SourceType.USER

        self._trace(
            f"INJECT_PROMPT: agent_id={self._agent_id}, "
            f"queue_size_before={len(self._message_queue)}, "
            f"source_id={actual_source_id}, source_type={actual_source_type.value}, "
            f"text={text[:50]}..."
        )
        # If this session is IDLE and we have a continuation callback, trigger it
        # directly instead of queuing. This applies to:
        # - CHILD messages: subagent status updates (COMPLETED, IDLE, etc.)
        # - PARENT messages: instructions from parent agent (send_to_subagent)
        # - USER messages: direct user input while idle
        if (
            self._activity_phase == ActivityPhase.IDLE
            and not self._is_running
            and self._on_continuation_needed
        ):
            self._trace(f"INJECT_PROMPT: Session is idle, triggering continuation for {actual_source_type.value} message")
            # Notify for tracing (UI visibility)
            if self._on_prompt_injected:
                self._on_prompt_injected(text)
            self._on_continuation_needed(text)
        else:
            # Session is busy - queue the message for later processing
            # High-priority (PARENT/USER/SYSTEM/EVENT) → processed mid-turn
            # Low-priority (CHILD) → processed when becoming idle
            self._message_queue.put(text, actual_source_id, actual_source_type)
            self._trace(f"INJECT_PROMPT: queue_size_after={len(self._message_queue)}")

    def _forward_to_parent(self, event_type: str, content: str) -> None:
        """Forward an event to the parent session.

        Only forwards essential events that require parent action. Progress events
        (MODEL_OUTPUT, TOOL_CALL, TOOL_OUTPUT) are NOT forwarded to avoid cluttering
        the parent's context and causing the model to echo them.

        These messages are queued with CHILD priority, meaning they will be
        processed when the parent becomes idle (not mid-turn). This prevents
        status updates from interrupting the parent's current work.

        Args:
            event_type: Type of event:
                - MODEL_OUTPUT: (NOT forwarded) Text the subagent is generating
                - TOOL_CALL: (NOT forwarded) Tool the subagent is calling
                - TOOL_OUTPUT: (NOT forwarded) Output from subagent's tool execution
                - COMPLETED: Subagent finished its task
                - IDLE: Subagent is idle and ready for input
                - ERROR: Subagent encountered an error
                - CANCELLED: Subagent was cancelled
                - CLARIFICATION_REQUESTED: Subagent needs clarification from parent
                - PERMISSION_REQUESTED: Subagent needs permission approval from parent
            content: Event content/payload.
        """
        if not self._parent_session:
            return

        # Skip verbose progress events - parent doesn't need to see these
        # and forwarding them causes the model to echo them in its output
        if event_type in ("MODEL_OUTPUT", "TOOL_CALL", "TOOL_OUTPUT"):
            return

        message = f"[SUBAGENT agent_id={self._agent_id} event={event_type}]\n{content}"
        self._parent_session.inject_prompt(
            message,
            source_id=self._agent_id,
            source_type=SourceType.CHILD
        )

    def _has_active_streams(self) -> bool:
        """Check if there are active streaming tools.

        Returns:
            True if there are streaming tools that may produce more output.
        """
        if not self._stream_manager:
            return False
        return self._stream_manager.has_active_streams()

    def _wait_for_streaming_updates(self) -> List[StreamUpdate]:
        """Wait for updates from active streaming tools.

        Blocks until updates are available or timeout is reached.

        Returns:
            List of StreamUpdate objects (may be empty).
        """
        if not self._stream_manager:
            return []
        return self._stream_manager.wait_for_updates(timeout=self._streaming_wait_timeout)

    def _format_streaming_updates(self, updates: List[StreamUpdate]) -> str:
        """Format streaming updates for injection into the conversation.

        Args:
            updates: List of StreamUpdate objects.

        Returns:
            Formatted message string for model consumption.
        """
        if not updates:
            return ""

        # Wrap in <hidden> so user doesn't see raw streaming data, only model sees it
        parts = ["<hidden><streaming_updates>"]
        for update in updates:
            parts.append(f"\n[Stream: {update.tool_name} (stream_id={update.stream_id})]")
            if update.new_chunks:
                parts.append(f"New results ({len(update.new_chunks)} items):")
                for chunk in update.new_chunks:
                    parts.append(f"  - {chunk.content}")
            if update.is_complete:
                parts.append(f"Stream completed. Total results: {update.total_chunks}")
                if update.final_result:
                    # Only include final result if it's different from chunks
                    parts.append(f"Final result summary available.")
        parts.append("\n</streaming_updates>")
        parts.append(
            "\nYou can continue acting on these results. "
            "Call dismiss_stream(stream_id='*') when you have enough results from all streams."
        )
        parts.append("</hidden>")

        return "".join(parts)

    def _drain_child_messages(self, on_output: Optional[OutputCallback] = None) -> str:
        """Process all pending messages when becoming idle.

        This drains both:
        - Child messages: Status updates from subagents (COMPLETED, IDLE, etc.)
        - High-priority messages: USER, PARENT, SYSTEM messages queued while busy

        All queued messages are collected and returned so the caller can
        send them to the model as the next prompt.

        This method is called:
        - In the finally block of send_message() before going idle
        - In inject_prompt() when messages arrive while already idle

        Args:
            on_output: Optional callback for logging/tracing.

        Returns:
            Collected message text (empty string if no messages).
        """
        drained_count = 0
        collected_messages: List[str] = []

        # First drain high-priority messages (USER, PARENT, SYSTEM)
        # These take precedence - if the user/parent sends a message, it should
        # be processed before subagent status updates
        while True:
            msg = self._message_queue.pop_first_parent_message()
            if msg is None:
                break

            drained_count += 1
            collected_messages.append(msg.text)
            self._trace(
                f"DRAIN_PRIORITY_MESSAGE: agent_id={self._agent_id}, "
                f"source_type={msg.source_type.value}, source_id={msg.source_id}, "
                f"text={msg.text[:100]}..."
            )

            # Log the message for tracing (UI visibility)
            if self._on_prompt_injected:
                self._on_prompt_injected(msg.text)

        # Then drain child messages (subagent status updates)
        # These are lower priority and processed after user/parent messages
        while True:
            msg = self._message_queue.pop_first_child_message()
            if msg is None:
                break

            drained_count += 1
            collected_messages.append(msg.text)
            self._trace(
                f"DRAIN_CHILD_MESSAGE: agent_id={self._agent_id}, "
                f"source_id={msg.source_id}, text={msg.text[:100]}..."
            )

            # Log the child message for tracing (UI visibility)
            if self._on_prompt_injected:
                self._on_prompt_injected(msg.text)

        collected_text = "\n\n".join(collected_messages)

        if drained_count > 0:
            self._trace(f"DRAIN_MESSAGES: Processed {drained_count} messages total")

            # If we're idle and drained messages, the model needs to react
            # Trigger continuation callback so server can start a new turn
            if (
                self._activity_phase == ActivityPhase.IDLE
                and not self._is_running
                and self._on_continuation_needed
            ):
                self._trace(f"DRAIN_MESSAGES: Triggering continuation callback with {len(collected_text)} chars")
                self._on_continuation_needed(collected_text)

        return collected_text

    # ==================== Public Accessors ====================

    @property
    def workspace_path(self) -> Optional[str]:
        """Return the workspace path for this session.

        If a session-level override was provided via ``configure()``
        (e.g. for a fork-replay worktree), returns that.  Otherwise
        falls back to the runtime's registry workspace path.
        """
        if self._workspace_path is not None:
            return self._workspace_path
        if self._runtime and self._runtime.registry:
            return getattr(self._runtime.registry, '_workspace_path', None)
        return None

    def get_session_env(
        self, key: str, default: Optional[str] = None,
    ) -> Optional[str]:
        """Read a per-session env value (Shape 3 PR 1).

        Lookup order:

        1. ``self._session_env`` — the resolved workspace ``.env`` +
           profile env + overrides populated by runner-side
           ``bootstrap_session``.  Set after the session is
           constructed; empty dict if Shape 3 PR 1's wiring hasn't
           populated it (older daemons, test stubs, etc.).
        2. ``os.environ`` — process-global fallback.  Runner-side
           bootstrap also writes the resolved env here so third-party
           code reading the process env sees the same values.

        Mirrors ``shared.session_context.get_session_env`` — but reads
        from the session-attached attribute instead of the
        daemon-side ContextVar.  Use this accessor when you have a
        :class:`JaatoSession` in hand; use the session_context
        helper from code that runs outside a session method (e.g.
        plugin discovery / initialize).
        """
        if self._session_env and key in self._session_env:
            return self._session_env[key]
        import os as _os
        return _os.environ.get(key, default)

    def get_system_instruction(self) -> Optional[str]:
        """Return the materialised system instruction for this session.

        After ``configure()`` this is the fully-assembled prompt that
        the provider receives as ``system_instruction`` on every
        ``provider.complete()`` call.  It incorporates base instructions,
        plugin contributions, formatter guidance, and any
        ``system_instruction_override`` that was supplied at configure time.

        External tools (e.g. session interrogation, fork-replay) read
        this to let the fine-tuner inspect or edit the materialised
        prompt without having to reverse-engineer the assembly pipeline.
        """
        return self._system_instruction

    # ==================== Cancellation Support ====================

    @property
    def is_running(self) -> bool:
        """Check if a message is currently being processed.

        Returns:
            True if send_message() is in progress, False otherwise.
        """
        return self._is_running

    @property
    def activity_phase(self) -> ActivityPhase:
        """Get the current activity phase.

        Returns:
            The current ActivityPhase (IDLE, WAITING_FOR_LLM, STREAMING, EXECUTING_TOOL).
        """
        return self._activity_phase

    @property
    def phase_started_at(self) -> Optional[datetime]:
        """Get when the current activity phase started.

        Returns:
            Datetime when the current phase began, or None if IDLE.
        """
        return self._phase_started_at

    @property
    def phase_duration_seconds(self) -> Optional[float]:
        """Get how long we've been in the current phase.

        Returns:
            Duration in seconds, or None if IDLE (no active phase).
        """
        if self._phase_started_at is None:
            return None
        return (datetime.now() - self._phase_started_at).total_seconds()

    def _set_activity_phase(self, phase: ActivityPhase) -> None:
        """Set the current activity phase (internal use).

        Fires ``_on_running_state_changed`` when the session crosses the
        idle/non-idle boundary (i.e. ``IDLE → WAITING_FOR_LLM`` or
        ``STREAMING → IDLE``), so external listeners like the subagent plugin
        can drive UI status updates automatically.

        Args:
            phase: The new activity phase.
        """
        previous_phase = self._activity_phase
        self._activity_phase = phase
        self._phase_started_at = datetime.now() if phase != ActivityPhase.IDLE else None

        # Notify when the running status changes (idle ↔ non-idle)
        was_idle = previous_phase == ActivityPhase.IDLE
        is_idle = phase == ActivityPhase.IDLE
        if was_idle != is_idle and self._on_running_state_changed:
            self._on_running_state_changed(not is_idle)

        # Clear permission suspensions on phase transitions
        if phase == ActivityPhase.IDLE and self._runtime and self._runtime.permission_plugin:
            # Clear idle suspension when session goes idle
            self._runtime.permission_plugin.clear_idle_suspension()
            # Also clear turn suspension (turn has ended)
            self._runtime.permission_plugin.clear_turn_suspension()

    @contextmanager
    def _provider_access(self):
        """Context manager that serializes access to the provider.

        Waits for ``_fork_gate`` (blocks while an external replay owns the
        provider), then signals ``_provider_idle`` clear/set around the
        body.  Used by every ``provider.complete()`` call site in the turn
        loop so that ``replay_messages()`` can safely pause the session
        and run its own provider call without concurrent access.
        """
        self._fork_gate.wait()
        self._provider_idle.clear()
        try:
            yield
        finally:
            self._provider_idle.set()

    @contextmanager
    def pause_for_observation(self, timeout: float = 30.0):
        """Pause the session for external observation.

        Waits for any in-progress turn to complete, then blocks new
        ``send_message()`` calls and provider access until the context
        manager exits.  Used by workspace snapshot, session
        interrogation, and fork-replay primitives that need a stable
        view of session state.

        Within the paused window, the session's history, workspace
        files, and plugin state are guaranteed not to change from
        session-driven activity (the user can still type — their input
        queues behind the observation lock and proceeds when the
        context manager exits).

        Args:
            timeout: Maximum seconds to wait for the current turn
                to finish before raising ``TimeoutError``.

        Raises:
            TimeoutError: If the session does not become idle within
                *timeout* seconds.
        """
        if not self._turn_complete.wait(timeout=timeout):
            raise TimeoutError(
                f"Session still running after {timeout}s — cannot pause"
            )
        self._observation_lock.clear()
        self._fork_gate.clear()
        try:
            yield
        finally:
            self._fork_gate.set()
            self._observation_lock.set()

    def request_stop(self, reason: str = "") -> bool:
        """Request cancellation of the current message processing.

        If a message is being processed, signals the cancel token to stop.
        The message loop will check this token and exit gracefully.

        Args:
            reason: Why the stop was requested. Included in the
                ``[Generation cancelled (reason)]`` output message.
                Defaults to ``"user_cancelled"``.

        Returns:
            True if a cancellation was requested (message was running),
            False if no message was running.

        Note:
            Cancellation is cooperative - it may not be immediate.
            The current streaming chunk will complete before stopping.
        """
        if self._cancel_token and self._is_running:
            self._cancel_token.cancel(reason=reason or "user_cancelled")
            return True
        return False

    def set_streaming_enabled(self, enabled: bool) -> None:
        """Enable or disable streaming mode.

        When enabled (default), the session uses streaming APIs for
        real-time output and better cancellation support.

        Args:
            enabled: True to use streaming, False for batched responses.
        """
        self._use_streaming = enabled

    def set_reference_authorizer(self, authorizer) -> None:
        """Install the AppArmor reference-fragment authorizer for this session.

        Called by ``JaatoServer.set_reference_authorizer()`` after WS
        provisions a confined profile.  Plugins read it via
        :meth:`get_reference_authorizer`.  Passing ``None`` clears it.
        """
        self._reference_authorizer = authorizer

    def get_reference_authorizer(self):
        """Return the AppArmor reference-fragment authorizer, or ``None``.

        Used by the references plugin to grant kernel-level readonly
        access to selected reference paths.  ``None`` means the session
        is not running under AppArmor confinement, so the application-
        layer ``sandbox_manager`` allowlist is the only authorization
        layer the plugin needs to touch.
        """
        return self._reference_authorizer

    def set_reference_authorization_enabled(self, enabled: bool) -> None:
        """Set the bool flag indicating AppArmor reference-fragment
        authorization is available for this session.

        Phase 3 §7c step 6.1.  This is the **runner-side counterpart**
        of :meth:`set_reference_authorizer`.

        Pre-§7c the daemon called :meth:`set_reference_authorizer`
        with a Python ``ReferenceAuthorizer`` instance, which the
        daemon-side references plugin consumed via
        :meth:`get_reference_authorizer`.  The Python object can't
        cross the RPC boundary (it holds a daemon-side
        ``AppArmorManager`` reference), so post-§7c the daemon
        forwards a bool flag instead.

        When the references plugin migrates runner-side, it reads
        :meth:`is_reference_authorization_enabled` and uses the
        existing ``apparmor.add_reference_fragment`` runner→daemon
        RPC (Phase 3 §3.2.2) to authorize paths.  The session_id
        for the RPC call is already known runner-side via the
        bootstrap envelope.

        Args:
            enabled: ``True`` if the daemon-side AppArmor manager
                successfully provisioned a profile for this session
                (i.e. ``ReferenceAuthorizer is not None`` daemon-
                side).  ``False`` for unconfined sessions.
        """
        self._reference_authorization_enabled = bool(enabled)

    def is_reference_authorization_enabled(self) -> bool:
        """Read the AppArmor reference-fragment authorization flag.

        Returns ``False`` by default (pre-set, or when the daemon
        forwards ``enabled=False``).  Used by the runner-side
        references plugin (post-migration) to decide whether to
        invoke the ``apparmor.add_reference_fragment`` runner→daemon
        RPC when admitting an external reference path.

        Phase 3 §7c step 6.1.
        """
        return getattr(self, "_reference_authorization_enabled", False)

    def set_parent_cancel_token(self, token: CancelToken) -> None:
        """Set a parent cancel token for cancellation propagation.

        When set, this session will check both its own cancel token
        and the parent token. If the parent is cancelled, this session
        will also stop - enabling automatic parent→child propagation.

        Args:
            token: The parent session's cancel token.
        """
        self._parent_cancel_token = token

    def _is_cancelled(self) -> bool:
        """Check if this session or its parent has been cancelled.

        Returns:
            True if either this session's token or parent token is cancelled.
        """
        if self._cancel_token and self._cancel_token.is_cancelled:
            return True
        if self._parent_cancel_token and self._parent_cancel_token.is_cancelled:
            return True
        return False

    @property
    def supports_stop(self) -> bool:
        """Check if the current provider supports mid-turn cancellation.

        Stop capability requires both streaming support and provider
        implementation of cancellation handling.

        Returns:
            True if stop is supported, False otherwise.
        """
        if not self._provider:
            return False
        # Check if provider has supports_stop method and it returns True
        if hasattr(self._provider, 'supports_stop'):
            return self._provider.supports_stop()
        # Fallback: if streaming is supported, stop is supported
        if hasattr(self._provider, 'supports_streaming'):
            return self._provider.supports_streaming()
        return False

    def configure(
        self,
        tools: Optional[List[str]] = None,
        system_instructions: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        skip_provider: bool = False,
        preloaded_plugins: Optional[set] = None,
        skip_model_test: bool = False,
        system_instruction_override: Optional[str] = None,
        suppress_base_instructions: bool = False,
        workspace_path: Optional[str] = None,
        completion_payload_schema: Optional[Any] = None,
        tier_config: Optional['ModelTierConfig'] = None,
        agent_params: Optional[Dict[str, Any]] = None,
        completion_artifacts: Optional[List[Any]] = None,
    ) -> None:
        """Configure the session with tools and instructions.

        Args:
            tools: Optional list of plugin names to expose. If None, uses all
                   exposed plugins from the runtime's registry.
            system_instructions: Optional additional system instructions.
            plugin_configs: Optional per-plugin configuration overrides.
                           Plugins will be re-initialized with these configs.
            skip_provider: If True, skip provider creation (for auth-pending mode).
                          User commands will be available but model calls won't work.
            preloaded_plugins: Optional set of plugin names that should bypass
                              deferred tool loading. All their tools (including
                              discoverable) are loaded into the initial context.
            skip_model_test: If True, skip the network call that verifies the
                model responds during provider creation.
            system_instruction_override: If provided, replaces the assembled
                system instruction entirely — the multi-layer pipeline
                (base + additional + plugins + formatters + presentation) is
                still run (for side effects like instruction budget accounting)
                but its result is discarded in favour of this string.  Used by
                session-manipulation tools that want to replay a session with
                an edited version of the materialised prompt.
            suppress_base_instructions: Partial suppression — drop only the
                BASE layer (``.jaato/instructions/*.md`` + premium baseline)
                from the assembled prompt while keeping the agent/session
                instructions, plugin instructions, pinned references, and
                framework constants.  Intended for small-context models where
                the framework baseline is the single biggest token consumer.
                Ignored when ``system_instruction_override`` is set (the full
                override supersedes any partial suppression).
            workspace_path: If provided, overrides the runtime's workspace
                path for this session.  Used by fork-replay to point a
                temp session at a worktree snapshot without affecting other
                sessions sharing the same runtime.
        """
        import time as _time
        _t_configure_start = _time.perf_counter()

        # Session-level workspace override
        if workspace_path is not None:
            self._workspace_path = workspace_path

        # Profile-declared completion payload schema (raw — resolved by
        # LifecycleTools at construction time using session.workspace_path)
        if completion_payload_schema is not None:
            self._completion_payload_schema = completion_payload_schema

        # Spawn-time agent_params (forwarded case_data etc.) for
        # dynamic-instructions render scripts.  See ``_agent_params`` doc
        # in __init__.
        if agent_params is not None:
            self._agent_params = dict(agent_params)

        # Profile-declared output artefacts (rendered after
        # signal_completion validates).  See ``_completion_artifacts``
        # doc in __init__.
        if completion_artifacts is not None:
            self._completion_artifacts = list(completion_artifacts)

        # Tier mode: when a tier_config is supplied, the session's
        # initial model is overridden by the initial tier's model so the
        # provider connects to the right model from turn 0.  The active
        # tier is set to the config's initial_tier.  When None, the
        # session stays in single-model mode (legacy behaviour).
        if tier_config is not None:
            self._tier_config = tier_config
            self._active_tier = tier_config.initial_tier
            initial_model = tier_config.tiers[tier_config.initial_tier].model
            if self._model_name and self._model_name != initial_model:
                logger.info(
                    "Tier mode active: overriding session model %s with "
                    "initial tier %s's model %s",
                    self._model_name, tier_config.initial_tier, initial_model,
                )
            self._model_name = initial_model

        # Store preloaded plugins for use in deferred instruction collection
        self._preloaded_plugins = preloaded_plugins or set()
        # Store tool plugin names
        self._tool_plugins = tools

        # Re-initialize plugins with session-specific configs if provided
        if plugin_configs and self._runtime.registry:
            for plugin_name, config in plugin_configs.items():
                if tools is None or plugin_name in tools:
                    try:
                        # Inject agent_name into plugin config for trace logging
                        if self._agent_name and "agent_name" not in config:
                            config = {**config, "agent_name": self._agent_name}
                        # expose_tool with new config will re-initialize
                        self._runtime.registry.expose_tool(plugin_name, config)
                    except Exception as e:
                        print(f"Warning: Failed to configure plugin '{plugin_name}': {e}")

        # Stash provider-creation args for lazy use by ``_ensure_provider``.
        # Pre-2026-05-13 the eager ``self._provider = self._runtime.create_provider(...)``
        # call here added 9s (zhipuai) / 2-3s (anthropic) to the bootstrap
        # RPC critical path because ``create_provider`` calls
        # ``provider.initialize(config)`` which does the network handshake.
        # Deferring shifts that cost to first model call, where the user
        # is already waiting under the streaming spinner.  See
        # ``docs/design/runner_prewarm_pool_plan.md`` §3.5 + §4 PR 1.
        # ``skip_provider`` (auth-pending mode) keeps the existing
        # "never create the provider here" semantics — the post-auth
        # handler triggers the lazy path the same way send_message does.
        if not skip_provider:
            self._provider_lazy_pending = {
                'model_name': self._model_name,
                'provider_name': self._provider_name_override,
                'skip_model_test': skip_model_test,
                'plugin_configs': plugin_configs,
            }

        # Create executor
        self._executor = ToolExecutor(ledger=self._runtime.ledger)

        # Get tool schemas and executors from runtime
        self._tools = self._runtime.get_tool_schemas(tools, preloaded_plugins=self._preloaded_plugins)
        executors = self._runtime.get_executors(tools)

        # Register executors
        for name, fn in executors.items():
            self._executor.register(name, fn)

        # Set registry for auto-background support
        if self._runtime.registry:
            self._executor.set_registry(self._runtime.registry)

        # Initialize stream manager for streaming tool support
        self._stream_manager = StreamManager()
        if self._runtime.registry:
            self._stream_manager.set_registry(self._runtime.registry)

            # Register streaming control tools (e.g., dismiss_stream) as core tools
            # This makes them visible to introspection and includes them in tool schemas
            auto_approved = self._stream_manager.get_auto_approved_tools()
            for schema in self._stream_manager.get_tool_schemas():
                executor = self._stream_manager.get_executors().get(schema.name)
                if executor:
                    is_auto_approved = schema.name in auto_approved
                    self._runtime.registry.register_core_tool(schema, executor, is_auto_approved)

            # Register event bus subscription tools as core tools.
            # These allow the model to subscribe to events (task lifecycle,
            # external ingress) and receive them via inject_prompt().
            from .event_bus_tools import EventBusTools
            self._event_bus_tools = EventBusTools(self)
            ebt_auto = self._event_bus_tools.get_auto_approved_tools()
            ebt_executors = self._event_bus_tools.get_executors()
            for schema in self._event_bus_tools.get_tool_schemas():
                executor = ebt_executors.get(schema.name)
                if executor:
                    is_auto = schema.name in ebt_auto
                    self._runtime.registry.register_core_tool(schema, executor, is_auto)
            # Also register backward-compat aliases (subscribeToTasks, getTaskEvents)
            # These map to the same executors but aren't in get_tool_schemas()
            # (no separate schema — the model uses them by name from old prompts).
            for alias_name, executor in ebt_executors.items():
                if alias_name not in [s.name for s in self._event_bus_tools.get_tool_schemas()]:
                    self._executor.register(alias_name, executor)

            # Refresh runtime's tool cache to include the newly registered core tools
            self._runtime.refresh_tool_cache()

            # Merge newly-registered core tools (stream, event_bus) into the
            # carefully-built _tools — additive only.  Re-assigning
            # ``self._tools`` from a fresh ``get_tool_schemas`` call without
            # ``preloaded_plugins=`` silently re-applies the discoverability
            # filter and drops preloaded plugins' discoverable tools.
            existing_names = {s.name for s in self._tools}
            refreshed_schemas = self._runtime.get_tool_schemas(
                tools, preloaded_plugins=self._preloaded_plugins
            )
            for schema in refreshed_schemas:
                if schema.name not in existing_names:
                    self._tools.append(schema)
                    existing_names.add(schema.name)
            for name, fn in self._runtime.get_executors(tools).items():
                self._executor.register(name, fn)

            # Register lifecycle tools (signal_completion) directly on this
            # session.  These are model-facing tools that must be visible in
            # schemas regardless of profile plugin lists — unlike core tools
            # (stream controls, event bus) which are internal infrastructure.
            from .lifecycle_tools import LifecycleTools
            self._lifecycle_tools = LifecycleTools(self)
            existing_names = {s.name for s in self._tools}
            exposed_lifecycle_names = set()
            for schema in self._lifecycle_tools.get_tool_schemas():
                if schema.name not in existing_names:
                    self._tools.append(schema)
                exposed_lifecycle_names.add(schema.name)
            # Server 0.6.61+: only register executors for lifecycle
            # tools that survived the schema filter.  Without this
            # gate, ``signal_completion`` would still be callable
            # even when filtered from the tool surface (interactive
            # root sessions): providers don't strictly enforce schema
            # membership, so a model emitting the call from cached
            # knowledge would still hit the executor and terminate
            # the session — the exact failure the schema filter was
            # supposed to close.  See LifecycleTools.get_tool_schemas
            # for the filter rationale.
            for name, fn in self._lifecycle_tools.get_executors().items():
                if name in exposed_lifecycle_names:
                    self._executor.register(name, fn)
            # Auto-approve so no permission prompt — same gating: only
            # whitelist lifecycle tools that are actually exposed.
            if self._runtime.permission_plugin:
                approved = [
                    t for t in self._lifecycle_tools.get_auto_approved_tools()
                    if t in exposed_lifecycle_names
                ]
                if approved:
                    self._runtime.permission_plugin.add_whitelist_tools(approved)

        # Set permission plugin with agent context
        if self._runtime.permission_plugin:
            context = {"agent_type": self._agent_type, "session_id": self._daemon_session_id}
            if self._agent_name:
                context["agent_name"] = self._agent_name
            self._executor.set_permission_plugin(
                self._runtime.permission_plugin,
                context=context
            )

        # Set reliability plugin for tool failure tracking
        if self._runtime.reliability_plugin:
            self._executor.set_reliability_plugin(self._runtime.reliability_plugin)
            # Set session context for pattern tracking
            self._runtime.reliability_plugin.set_session_context(
                session_id=self._agent_id,
            )
            # Set model context
            if self._model_name:
                available = self._runtime.list_available_models(
                    provider_name=self._provider_name_override
                )
                self._runtime.reliability_plugin.set_model_context(self._model_name, available)
            # Collect and register prerequisite policies from all plugins
            if self._runtime.registry:
                policies = self._runtime.registry.collect_prerequisite_policies()
                if policies:
                    self._runtime.reliability_plugin.register_prerequisite_policies(policies)
                    self._trace(f"configure: registered {len(policies)} prerequisite policies with reliability plugin")

        # Set this session as parent for subagent plugin (for cancellation propagation)
        if self._runtime.registry:
            subagent_plugin = self._runtime.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_parent_session'):
                subagent_plugin.set_parent_session(self)

        # Wire the waypoint plugin with session-state accessors so it can
        # capture conversation history snapshots when a waypoint is created.
        # Without this the plugin's history-capture path is dormant: every
        # waypoint is saved with history_snapshot=None and downstream
        # consumers (waypoint_info metadata, premium handoff
        # fork_from_waypoint) get nothing to work with.  The serializer
        # adapter wraps shared.plugins.session.serialize_history (which
        # returns List[Dict]) in json.dumps to match the plugin's
        # Callable[[List[Message]], str] contract — bridges the asymmetry
        # without changing the plugin's signature or Waypoint's schema.
        if self._runtime.registry:
            waypoint_plugin = self._runtime.registry.get_plugin("waypoint")
            if waypoint_plugin and hasattr(waypoint_plugin, 'set_session_callbacks'):
                from .plugins.session.serializer import serialize_history
                waypoint_plugin.set_session_callbacks(
                    get_history=self.get_history,
                    serialize_history=lambda msgs: json.dumps(
                        serialize_history(msgs)
                    ),
                    get_turn_index=lambda: self._turn_index,
                    # Snapshot session-attached state alongside history so
                    # a fork-from-waypoint primitive can carry extension-
                    # owned state across the fork (premium pseudonymization
                    # lookup table, audit chain head, etc.).  Routes
                    # through get_all_session_state so registered providers
                    # are invoked — the snapshot reflects live values, not
                    # stale set-state pushes.
                    get_session_state=self.get_all_session_state,
                )

        # Auto-wire plugins that need session access
        # Any plugin with set_session() will receive this session reference.
        # When the session has an explicit plugin list (agent profile), only
        # wire plugins that are in that list (plus introspection which is
        # always essential).  This prevents plugins like notebook from
        # injecting instructions when the profile didn't include them.
        if self._runtime.registry:
            import threading
            self._trace(f"configure: wiring plugins with session, thread_id={threading.current_thread().ident}")
            set_current_session(self)
            wire_set = None
            if tools is not None:
                wire_set = set(tools)
                wire_set.add("introspection")
            for plugin_name in self._runtime.registry._exposed:
                if wire_set is not None and plugin_name not in wire_set:
                    continue
                plugin = self._runtime.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'set_session'):
                    plugin.set_session(self)

        # Remember both knobs so _populate_instruction_budget (called
        # below) can produce an honest budget reflecting the wire prompt.
        self._system_instruction_override = system_instruction_override
        self._suppress_base_instructions = suppress_base_instructions

        # Build system instructions.
        #
        # Full-override path: skip assembly entirely — no disk I/O, no
        # enrichment churn.  Plugin state was already initialised earlier
        # in configure_tools(), so tool functionality is intact; only the
        # would-be-discarded enrichment text is skipped.
        #
        # Otherwise: assemble normally.  ``include_base=False`` drops just
        # the BASE layer (framework baseline) while keeping the agent
        # prompt, plugin instructions, and framework constants — the
        # partial-suppression path for small-context models.  Base is
        # also lazy-loaded on first use, so sessions that always suppress
        # it never touch the disk.
        if system_instruction_override is not None:
            self._system_instruction = system_instruction_override
        else:
            self._system_instruction = self._runtime.get_system_instructions(
                plugin_names=tools,
                additional=system_instructions,
                presentation_context=self._presentation_context,
                include_base=not suppress_base_instructions,
            )

        # Dynamic-instructions expansion ({{!py:script.py}}).  Walks
        # the assembled system_instruction for placeholders and
        # substitutes each with the output of the named user script.
        # Scripts run on the framework's authority with a RenderContext
        # carrying session/runtime/registry/workspace_path/config_root/
        # agent_params handles, plus an os.environ snapshot.  This is
        # the input-side symmetric counterpart to reactor actions:
        # the agent never sees these scripts as choices to make, only
        # their output as content already present in its prompt.
        # See ``shared/dynamic_instructions.py`` and
        # ``project_backlog_dynamic_instructions`` (2026-04-30 addendum).
        if self._system_instruction and (
            "{{!py:" in self._system_instruction
            or "{{!py?:" in self._system_instruction
        ):
            from .dynamic_instructions import (
                expand_py_placeholders,
                build_render_context,
            )
            ctx = build_render_context(self, agent_params=self._agent_params)
            # Server 0.6.50+: when an AppArmor confine-context factory
            # is set, wrap the dynamic-instructions expansion in the
            # session's confinement so prefetch scripts and any
            # ``{{!py:...}}`` callables run with the same kernel-level
            # filesystem isolation as tools.  Closes the gap where
            # prefetch could write to ``.jaato`` regardless of the
            # deny rules in the profile.  Without a factory (IPC, no
            # AppArmor), the expansion runs as before.
            if self._confine_context_factory is not None:
                with self._confine_context_factory():
                    self._system_instruction = expand_py_placeholders(
                        self._system_instruction, ctx,
                    )
            else:
                self._system_instruction = expand_py_placeholders(
                    self._system_instruction, ctx,
                )

        # Store user commands
        if self._runtime.registry:
            self._user_commands = {}
            for cmd in self._runtime.registry.get_exposed_user_commands():
                self._user_commands[cmd.name] = cmd

        # Add reliability plugin user commands and executors
        if self._runtime.reliability_plugin:
            for cmd in self._runtime.reliability_plugin.get_user_commands():
                self._user_commands[cmd.name] = cmd
            # Register reliability command executor
            for name, fn in self._runtime.reliability_plugin.get_executors().items():
                self._executor.register(name, fn)

        # Register built-in model command
        self._register_model_command()

        # Register built-in telepathy tool (share_context)
        self._register_telepathy_tool()

        # Initialize empty session history (skip if in auth-pending mode)
        if not skip_provider:
            self._history.clear()

        # Populate instruction budget after all configuration is complete.
        # ``_count_tokens`` is already defensive against ``self._provider is
        # None`` — synchronous-phase falls back to the chars/4 estimate; the
        # background-refinement phase short-circuits via ``has_count_tokens``
        # when no provider exists.  Budget refinement triggers lazy provider
        # creation if/when needed (see ``_start_background_token_counting``).
        self._populate_instruction_budget(session_instructions=system_instructions)

        # Note: cache plugin wiring moved from here into ``_ensure_provider``
        # since 2026-05-13 — wiring needs a constructed provider, which is
        # now lazy.  The cache plugin attaches when the provider first
        # materializes; for the configure-then-first-message gap (where
        # there's no provider yet), there's nothing to cache against either.

        self._configured = True

        _configure_ms = (_time.perf_counter() - _t_configure_start) * 1000
        if _configure_ms > 10.0:
            self._trace(f"configure: completed in {_configure_ms:.1f}ms")

    def _ensure_provider(self) -> Optional['ModelProviderPlugin']:
        """Lazy-create the session's provider on first model use.

        Idempotent + thread-safe.  Pre-2026-05-13 the provider was
        eagerly created in ``configure()`` — that added 9s (zhipuai) or
        2-3s (anthropic) to the bootstrap RPC critical path because
        ``create_provider`` calls ``provider.initialize(config)`` which
        does the network handshake.  Deferring shifts that cost to first
        model call where the user is already waiting under the streaming
        spinner.  See ``docs/design/runner_prewarm_pool_plan.md`` §3.5.

        Returns:
            The constructed provider, or ``None`` if this session is
            in skip_provider mode (auth-pending — caller is responsible
            for re-running this once auth completes).
        """
        # Fast-path: already initialized.  Avoid lock contention on the
        # hot path.  Memory visibility is fine: ``self._provider`` is
        # only written under the lock, so a non-None read here implies
        # the writer's lock-release happens-before this read.
        if self._provider is not None:
            return self._provider

        with self._provider_init_lock:
            # Re-check under the lock (double-checked locking).
            if self._provider is not None:
                return self._provider
            if self._provider_lazy_pending is None:
                # skip_provider (auth-pending) mode — provider truly
                # never created here; post-auth path triggers a new
                # _ensure_provider call after stashing the pending args.
                return None
            cfg = self._provider_lazy_pending
            self._provider = self._runtime.create_provider(
                cfg['model_name'],
                provider_name=cfg['provider_name'],
                skip_model_test=cfg['skip_model_test'],
                plugin_configs=cfg['plugin_configs'],
            )
            # Propagate agent context to provider for trace identification.
            if hasattr(self._provider, 'set_agent_context'):
                self._provider.set_agent_context(
                    agent_type=self._agent_type,
                    agent_name=self._agent_name,
                    agent_id=self._agent_id,
                )
            # Wire cache plugin now that the provider exists.  Pre-defer
            # this fired at the end of configure() unconditionally.
            self._wire_cache_plugin()
            # Consume the stashed args — repeated calls become no-ops
            # (the fast-path above returns the cached provider).
            self._provider_lazy_pending = None
            return self._provider

    def _wire_cache_plugin(self) -> None:
        """Discover and attach the cache plugin matching the active provider.

        The cache plugin is selected by matching the provider's ``name``
        property against available cache plugins' ``provider_name``.
        When found:
        - The plugin is initialized with provider config extras
        - The current InstructionBudget is set on the plugin
        - The plugin is attached to the provider via ``set_cache_plugin()``

        This is a Variant A integration (provider delegates to plugin).
        """
        if not self._provider:
            return

        try:
            from shared.plugins.cache import load_cache_plugin_for_provider
        except ImportError:
            # Cache plugin infrastructure not installed
            return

        provider_name = getattr(self._provider, 'name', None)
        if not provider_name:
            return

        # Build config from provider config extras
        config = {}
        if self._runtime and self._runtime._provider_config:
            config = dict(self._runtime._provider_config.extra)
        # Include model name for threshold selection
        model_name = getattr(self._provider, 'model_name', None)
        if model_name:
            config['model_name'] = model_name

        cache_plugin = load_cache_plugin_for_provider(provider_name, config)

        if cache_plugin:
            # Set the budget so the plugin can make policy-aware decisions
            if self._instruction_budget:
                cache_plugin.set_budget(self._instruction_budget)

            # Attach to provider (Variant A: provider delegates to plugin)
            if hasattr(self._provider, 'set_cache_plugin'):
                self._provider.set_cache_plugin(cache_plugin)

            self._cache_plugin = cache_plugin
            self._trace(
                f"CACHE_PLUGIN: Attached {cache_plugin.name} for provider "
                f"{provider_name}"
            )

    def _unwrap_turn_result(self, turn_result: 'TurnResult') -> 'ProviderResponse':
        """Extract the ``ProviderResponse`` from a ``TurnResult``.

        ``provider.complete()`` now returns ``TurnResult``.  Call sites that
        previously received a raw ``ProviderResponse`` use this helper to
        unwrap the result, raising on fatal errors that the provider could
        not recover from.

        Args:
            turn_result: Result returned by ``provider.complete()`` (via
                ``with_retry``).

        Returns:
            The inner ``ProviderResponse``.

        Raises:
            RuntimeError: If the outcome is ``ERROR`` and no response is
                available (the exception stored in the ``TurnResult`` is
                re-raised).
        """
        if turn_result.outcome == TurnOutcome.ERROR:
            if turn_result.error:
                raise turn_result.error
            raise RuntimeError(turn_result.error_message or "Provider returned an error")
        # SUCCESS, TOOL_USE, CANCELLED, MAX_TOKENS — all have a response
        return turn_result.response

    def _add_model_response_to_history(self, response: 'ProviderResponse') -> None:
        """Add the model's response to session history.

        Called after ``provider.complete()`` returns successfully. Filters
        response parts to only text and function_call (excludes
        function_response parts which belong to user/tool messages).

        Args:
            response: The ProviderResponse from the provider.
        """
        history_parts = [
            p for p in response.parts
            if p.text is not None or p.function_call is not None
        ]
        if history_parts:
            self._history.append(Message(
                role=Role.MODEL,
                parts=history_parts,
                model=self._model_name,
                provider=self._provider.name if self._provider else None,
            ))

    def _get_tools_for_provider(self) -> Optional[List['ToolSchema']]:
        """Get the tool list to pass to the provider.

        Checks whether the provider manages its own tools (e.g. Claude CLI
        in delegated mode). If so, returns an empty list.

        Returns:
            Tools to pass, or empty list if provider manages its own.
        """
        uses_external = getattr(self._provider, 'uses_external_tools', lambda: True)()
        return self._tools if uses_external else []

    def _count_tokens(self, text: str) -> int:
        """Count tokens using cache, provider, or estimate (in that order).

        Lookup order:
        1. ``InstructionTokenCache`` — instant, shared across sessions.
        2. ``provider.count_tokens()`` — accurate HTTP call; result is
           stored in the cache for future hits.
        3. ``estimate_tokens()`` — chars/4 approximation fallback.

        Args:
            text: The text to count tokens for.

        Returns:
            Token count (actual or estimated).
        """
        if not text:
            return 0

        # 1. Check instruction token cache
        cache = self._runtime.instruction_token_cache
        provider_name = self._provider_name_override or self._runtime.provider_name
        cached = cache.get(provider_name, text)
        if isinstance(cached, int):
            return cached

        # 2. Try provider API
        if self._provider and hasattr(self._provider, 'count_tokens'):
            try:
                result = self._provider.count_tokens(text)
                # Ensure we got an int (handles mocked providers returning MagicMock)
                if isinstance(result, int):
                    cache.put(provider_name, text, result)
                    return result
                else:
                    self._trace(
                        f"count_tokens returned non-int ({type(result).__name__}), "
                        f"falling back to estimate"
                    )
            except Exception as e:
                self._trace(
                    f"count_tokens FAILED ({type(e).__name__}: {e}), "
                    f"falling back to estimate (text length: {len(text)} chars)"
                )

        # 3. Estimate fallback
        est = estimate_tokens(text)
        self._trace(f"count_tokens: using estimate={est} (from {len(text)} chars)")
        return est

    # ------------------------------------------------------------------
    # Two-phase instruction budget population
    # ------------------------------------------------------------------

    def _populate_instruction_budget(
        self,
        session_instructions: Optional[str] = None
    ) -> None:
        """Populate instruction budget with token counts by source layer.

        Uses a two-phase approach for fast session creation:

        **Phase 1 (synchronous, instant):** Build budget structure using
        cached counts (from ``InstructionTokenCache``) or ``estimate_tokens()``
        (chars/4).  Emit initial budget event.  Session is immediately usable.

        **Phase 2 (background threads):** For cache misses only, fire
        ``provider.count_tokens()`` calls in a ``ThreadPoolExecutor``.  Once
        all futures complete, update budget entries with accurate counts and
        emit a refined budget event.

        Args:
            session_instructions: The user-provided system_instructions from configure().
        """
        # Get context limit from provider.  By this point the provider has
        # already connect()'ed and resolved the limit (e.g. from model metadata
        # or a static lookup), so this is a cheap in-memory read.
        context_limit = 128_000  # Default
        if self._provider and hasattr(self._provider, 'get_context_limit'):
            try:
                context_limit = self._provider.get_context_limit()
            except Exception:
                pass  # keep default

        # Get session_id - use runtime's session ID or generate placeholder
        # The server will assign proper session_id when session is registered
        session_id = getattr(self._runtime, 'session_id', '') or ''

        # Create budget with default entries
        self._instruction_budget = InstructionBudget.create_default(
            session_id=session_id,
            agent_id=self._agent_id,
            agent_type=self._agent_type,
            context_limit=context_limit,
        )

        # --- Collect phase: gather all texts that need counting ---
        # Pass the override and suppress-base flag so the budget
        # reflects what's actually on the wire — a single OVERRIDE entry
        # (or nothing) on full override, or the assembly minus BASE on
        # partial suppression — rather than the assembly pipeline's
        # would-be-full output.
        requests = self._collect_instruction_texts(
            session_instructions,
            system_instruction_override=self._system_instruction_override,
            suppress_base=self._suppress_base_instructions,
        )

        # --- Resolve phase: use cache or estimate for each request ---
        cache = self._runtime.instruction_token_cache
        provider_name = self._provider_name_override or self._runtime.provider_name
        cache_misses: List[_TokenCountRequest] = []

        for req in requests:
            cached = cache.get(provider_name, req.text)
            if isinstance(cached, int):
                req.token_count = cached
                req.is_estimate = False
            else:
                req.token_count = estimate_tokens(req.text)
                req.is_estimate = True
                cache_misses.append(req)

        # --- Apply phase: build budget from resolved counts ---
        self._apply_instruction_counts(requests, context_limit)

        # --- Background phase: refine cache misses with provider ---
        has_count_tokens = (
            self._provider is not None
            and hasattr(self._provider, 'count_tokens')
        )
        if cache_misses and has_count_tokens:
            self._start_background_token_counting(
                cache_misses, provider_name, context_limit
            )
        else:
            if cache_misses:
                self._trace(
                    f"BUDGET_CALC: {len(cache_misses)} cache misses but no "
                    f"count_tokens API — estimates are final"
                )

    def _collect_instruction_texts(
        self,
        session_instructions: Optional[str],
        system_instruction_override: Optional[str] = None,
        suppress_base: bool = False,
    ) -> List['_TokenCountRequest']:
        """Collect all instruction texts that need token counting.

        When ``system_instruction_override`` is set the wire-level system
        message is exactly that string (or empty), regardless of what
        the assembly pipeline produced.  In that case the budget must
        reflect what *actually* reaches the model — we emit a single
        ``OVERRIDE`` entry (or nothing for an empty override) instead of
        walking BASE/CLIENT/FRAMEWORK/plugin texts that never make it to
        the wire.  This avoids the silent lie where the budget showed
        thousands of tokens of premium instructions while the model was
        receiving an empty system message.

        When ``system_instruction_override`` is ``None`` the assembly
        pipeline is the authoritative source and we walk SYSTEM children
        (base, client, framework, pinned references) plus PLUGIN
        children (per-plugin, per-formatter) into a flat list of
        ``_TokenCountRequest`` objects.

        Args:
            session_instructions: The user-provided system_instructions from configure().
            system_instruction_override: When not ``None``, supplants the
                whole assembly — see above.

        Returns:
            List of ``_TokenCountRequest`` — one per instruction text.
        """
        from .jaato_runtime import (
            _TASK_COMPLETION_INSTRUCTION,
            _PARALLEL_TOOL_GUIDANCE,
            _TURN_SUMMARY_INSTRUCTION,
            _is_parallel_tools_enabled,
        )

        # Override-aware short-circuit: the wire system message is the
        # override, period.  No need to walk the assembly pipeline (and
        # — paired with the lazy base loader — no disk I/O fires for
        # ``runtime.get_base_system_instructions()`` on this session).
        if system_instruction_override is not None:
            if system_instruction_override == "":
                return []
            return [_TokenCountRequest(
                text=system_instruction_override,
                source=InstructionSource.SYSTEM,
                child_key=SystemChildType.OVERRIDE.value,
                gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.OVERRIDE],
                label="Override (system_instruction_override)",
            )]

        requests: List[_TokenCountRequest] = []

        # --- SYSTEM children ---

        # 1. Base instructions from .jaato/instructions/ (or legacy single
        #    file) — lazy-loaded the first time any session asks.  Skipped
        #    entirely when suppress_base is set; other SYSTEM children
        #    (CLIENT / FRAMEWORK / pinned refs) and PLUGIN children below
        #    still contribute, so the agent keeps its agent-.md content
        #    and tool instructions.
        if not suppress_base and self._runtime:
            base_instructions = self._runtime.get_base_system_instructions()
        else:
            base_instructions = None
        if base_instructions:
            requests.append(_TokenCountRequest(
                text=base_instructions,
                source=InstructionSource.SYSTEM,
                child_key=SystemChildType.BASE.value,
                gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.BASE],
                label="Base Instructions",
            ))

        # 2. Client-provided session instructions (programmatic)
        if session_instructions:
            requests.append(_TokenCountRequest(
                text=session_instructions,
                source=InstructionSource.SYSTEM,
                child_key=SystemChildType.CLIENT.value,
                gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.CLIENT],
                label="Client Instructions",
            ))

        # 3. Framework constants (concatenated into one request)
        framework_parts = [_TASK_COMPLETION_INSTRUCTION]
        if _is_parallel_tools_enabled():
            framework_parts.append(_PARALLEL_TOOL_GUIDANCE)
        framework_parts.append(_TURN_SUMMARY_INSTRUCTION)
        framework_text = "\n\n".join(framework_parts)
        requests.append(_TokenCountRequest(
            text=framework_text,
            source=InstructionSource.SYSTEM,
            child_key=SystemChildType.FRAMEWORK.value,
            gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.FRAMEWORK],
            label="Framework",
        ))

        # 4. Pinned preselected references (content read by the model and
        #    promoted to system instruction for GC protection)
        for ref_id, pinned in getattr(self, '_pinned_references', {}).items():
            child_key = f"{SystemChildType.SELECTED_REFERENCES.value}:{ref_id}"
            requests.append(_TokenCountRequest(
                text=pinned.content,
                source=InstructionSource.SYSTEM,
                child_key=child_key,
                gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.SELECTED_REFERENCES],
                label=f"ref: {pinned.ref_name}",
            ))

        # --- PLUGIN children ---
        # When deferred tool loading is enabled, only include system
        # instructions from plugins that have at least one core tool.
        # Instructions for discoverable-only plugins are deferred until
        # the model activates one of their tools via get_tool_schemas.

        from .jaato_runtime import _is_deferred_tools_enabled
        deferred_enabled = _is_deferred_tools_enabled()

        if self._runtime.registry:
            for plugin_name in self._runtime.registry._exposed:
                if deferred_enabled and plugin_name not in self._preloaded_plugins and not self._runtime.registry.plugin_has_core_tools(plugin_name):
                    # Remember this plugin's instructions are deferred so we
                    # can inject them when the model discovers its tools.
                    # Exception: preloaded plugins always include instructions.
                    plugin = self._runtime.registry.get_plugin(plugin_name)
                    if plugin and hasattr(plugin, 'get_system_instructions'):
                        instr = plugin.get_system_instructions()
                        if instr:
                            self._deferred_plugin_instructions.add(plugin_name)
                    continue
                plugin = self._runtime.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'get_system_instructions'):
                    instr = plugin.get_system_instructions()
                    if instr:
                        requests.append(_TokenCountRequest(
                            text=instr,
                            source=InstructionSource.PLUGIN,
                            child_key=plugin_name,
                            gc_policy=DEFAULT_TOOL_POLICIES[PluginToolType.CORE],
                            label=plugin_name,
                        ))

        # Formatter pipeline instructions (output rendering capabilities)
        formatter_pipeline = getattr(self._runtime, '_formatter_pipeline', None)
        if formatter_pipeline and hasattr(formatter_pipeline, '_formatters'):
            for formatter in formatter_pipeline._formatters:
                if hasattr(formatter, 'get_system_instructions'):
                    instr = formatter.get_system_instructions()
                    if instr:
                        requests.append(_TokenCountRequest(
                            text=instr,
                            source=InstructionSource.PLUGIN,
                            child_key=formatter.name,
                            gc_policy=GCPolicy.PRESERVABLE,
                            label=formatter.name,
                        ))

        return requests

    def _apply_instruction_counts(
        self,
        requests: List['_TokenCountRequest'],
        context_limit: int,
    ) -> None:
        """Build budget children and parent totals from resolved token counts.

        Called once in Phase 1 (with estimates/cached values) and again after
        Phase 2 completes (with accurate counts for previously-estimated entries).

        Args:
            requests: List of resolved ``_TokenCountRequest`` objects.
            context_limit: Context window size for percentage logging.
        """
        # Group by source to compute parent totals
        source_totals: Dict[InstructionSource, int] = {}

        for req in requests:
            source_totals.setdefault(req.source, 0)
            source_totals[req.source] += req.token_count

            # Check if child already exists (Phase 2 update path)
            parent_entry = self._instruction_budget.get_entry(req.source)
            existing = parent_entry.children.get(req.child_key) if parent_entry else None
            if existing is not None:
                existing.tokens = req.token_count
            else:
                if req.token_count > 0:
                    self._instruction_budget.add_child(
                        req.source,
                        req.child_key,
                        req.token_count,
                        req.gc_policy,
                        label=req.label,
                    )

        # Update parent totals
        for source, total in source_totals.items():
            self._instruction_budget.update_tokens(source, total)

        # ENRICHMENT and CONVERSATION start at 0
        self._instruction_budget.update_tokens(InstructionSource.ENRICHMENT, 0)
        self._instruction_budget.update_tokens(InstructionSource.CONVERSATION, 0)

        # Log summary
        total_initial = sum(source_totals.values())
        estimate_count = sum(1 for r in requests if r.is_estimate)
        try:
            pct = (total_initial / context_limit * 100) if context_limit else 0
            self._trace(
                f"BUDGET_CALC: Budget {'updated' if any(not r.is_estimate for r in requests) else 'initial'} — "
                f"total={total_initial} tokens ({pct:.1f}% of {context_limit}), "
                f"estimates={estimate_count}/{len(requests)}"
            )
        except (TypeError, ValueError):
            self._trace(
                f"BUDGET_CALC: Budget applied — total={total_initial} tokens, "
                f"estimates={estimate_count}/{len(requests)}"
            )

        # Emit budget update event
        self._emit_instruction_budget_update()

    def _start_background_token_counting(
        self,
        cache_misses: List['_TokenCountRequest'],
        provider_name: str,
        context_limit: int,
    ) -> None:
        """Fire background threads to get accurate token counts for cache misses.

        Creates a ``ThreadPoolExecutor`` inside a daemon thread.  Each worker
        calls ``provider.count_tokens(text)`` and stores the result in the
        ``InstructionTokenCache``.  After all futures complete, updates budget
        entries with accurate counts and emits a refined budget event.

        Args:
            cache_misses: Requests whose counts are currently estimates.
            provider_name: Provider name for cache keying.
            context_limit: Context window size (for logging).
        """
        self._trace(
            f"BUDGET_CALC: Starting background token counting for "
            f"{len(cache_misses)} cache misses"
        )

        provider = self._provider
        cache = self._runtime.instruction_token_cache

        def _background_count() -> None:
            max_workers = min(len(cache_misses), 8)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                def _count_one(req: _TokenCountRequest) -> None:
                    try:
                        result = provider.count_tokens(req.text)
                        if isinstance(result, int):
                            cache.put(provider_name, req.text, result)
                            req.token_count = result
                            req.is_estimate = False
                        else:
                            self._trace(
                                f"BUDGET_BG: count_tokens for '{req.child_key}' "
                                f"returned non-int ({type(result).__name__}), "
                                f"keeping estimate"
                            )
                    except Exception as e:
                        self._trace(
                            f"BUDGET_BG: count_tokens for '{req.child_key}' "
                            f"failed ({type(e).__name__}: {e}), keeping estimate"
                        )

                futures = [pool.submit(_count_one, req) for req in cache_misses]
                for f in futures:
                    f.result()  # propagate exceptions to log

            # Update budget entries for refined counts
            refined = [r for r in cache_misses if not r.is_estimate]
            if refined:
                self._trace(
                    f"BUDGET_BG: Refined {len(refined)}/{len(cache_misses)} counts, "
                    f"updating budget entries"
                )
                # Update each refined child entry directly and recompute parent totals
                for req in refined:
                    parent_entry = self._instruction_budget.get_entry(req.source)
                    if parent_entry and req.child_key in parent_entry.children:
                        parent_entry.children[req.child_key].tokens = req.token_count

                # Recompute parent tokens from children for affected sources
                affected_sources = {r.source for r in refined}
                for source in affected_sources:
                    entry = self._instruction_budget.get_entry(source)
                    if entry and entry.children:
                        new_total = sum(c.tokens for c in entry.children.values())
                        entry.tokens = new_total

                self._emit_instruction_budget_update()
            else:
                self._trace(
                    f"BUDGET_BG: No counts refined (all provider calls failed), "
                    f"keeping estimates"
                )

        thread = threading.Thread(
            target=_background_count,
            name=f"budget-count-{self._agent_id}",
            daemon=True,
        )
        self._budget_counting_thread = thread
        thread.start()

    def _emit_instruction_budget_update(self) -> None:
        """Emit instruction budget update via callback and/or UI hooks.

        Also forwards the updated budget to the cache plugin (if attached)
        so it can adjust breakpoint placement based on current GC policies.
        """
        if not self._instruction_budget:
            return

        # Forward budget to cache plugin for policy-aware decisions
        cache_plugin = getattr(self, '_cache_plugin', None)
        if cache_plugin and hasattr(cache_plugin, 'set_budget'):
            try:
                cache_plugin.set_budget(self._instruction_budget)
            except Exception as e:
                logger.debug(f"Cache plugin set_budget failed: {e}")

        try:
            snapshot = self._instruction_budget.snapshot()

            # Direct callback (for main session in server)
            if self._on_instruction_budget_updated:
                self._on_instruction_budget_updated(snapshot)

            # UI hooks (for both main and subagent sessions)
            if self._ui_hooks and hasattr(self._ui_hooks, 'on_agent_instruction_budget_updated'):
                self._ui_hooks.on_agent_instruction_budget_updated(
                    agent_id=self._agent_id,
                    budget_snapshot=snapshot,
                )
        except Exception as e:
            logger.warning(f"Failed to emit instruction budget update: {e}")

    def _get_framework_enrichments(self, text: str) -> list[str]:
        """Detect and identify framework-injected enrichment content.

        Framework enrichments are automatically injected by plugins and include:
        - System reminders (<system-reminder> tags) - external (Claude CLI)
        - System notices ([System: ...]) - gc, cancellation, multimodal, session
        - Memory injection (💡 **Available Memories**) - memory plugin
        - Hidden content (<hidden> tags) - streaming, waypoint, nudge

        Args:
            text: The text content to check.

        Returns:
            List of enrichment type names found in the text.
        """
        if not text:
            return []

        enrichments = []

        # Check for system-reminder (external, e.g., Claude CLI)
        if "<system-reminder>" in text:
            enrichments.append("system-reminder")

        # Check for [System: ...] notices - identify specific source
        if "[System:" in text:
            # Extract content after [System: to identify source
            if "cancelled" in text or "canceled" in text:
                enrichments.append("cancellation")
            elif "image files" in text or "viewImage" in text:
                enrichments.append("multimodal")
            elif "conversation has been ongoing" in text or "session_describe" in text:
                enrichments.append("session")
            else:
                enrichments.append("gc")

        # Check for memory injection
        if "💡 **Available Memories**" in text:
            enrichments.append("memory")

        # Check for hidden content - identify specific source by inner content
        if "<hidden>" in text:
            # Extract content inside <hidden> tags to identify source
            import re
            hidden_matches = re.findall(r'<hidden>(.*?)</hidden>', text, re.DOTALL)
            hidden_types_found = set()
            for hidden_content in hidden_matches:
                if "<streaming_updates>" in hidden_content or hidden_content.startswith("["):
                    hidden_types_found.add("streaming")
                elif "<waypoint-restore>" in hidden_content:
                    hidden_types_found.add("waypoint")
                else:
                    hidden_types_found.add("nudge")
            enrichments.extend(sorted(hidden_types_found))

        return enrichments

    def _update_conversation_budget(self) -> None:
        """Update CONVERSATION entry in instruction budget from current history."""
        if not self._instruction_budget:
            return

        history = self.get_history()
        conversation_tokens = 0
        conv_entry = self._instruction_budget.get_entry(InstructionSource.CONVERSATION)
        if conv_entry:
            conv_entry.children.clear()  # Reset children

        # Determine if the just-completed turn was complex
        # Complex turn = multiple model responses AND had tool calls
        # The final model response in a complex turn contains the summary (per framework guidance)
        is_complex_turn = self._turn_model_response_count > 1 and self._turn_had_tool_calls

        # Find the index of the last MODEL message with text-only content (no tool calls)
        # This is the turn summary candidate if the turn was complex
        last_model_text_only_idx = -1
        if is_complex_turn:
            for i in range(len(history) - 1, -1, -1):
                msg = history[i]
                if msg.role == Role.MODEL:
                    # Check if this message has text but no function calls
                    has_text = any(p.text for p in msg.parts)
                    has_function_calls = any(p.function_call for p in msg.parts)
                    if has_text and not has_function_calls:
                        last_model_text_only_idx = i
                        break

        # Track actual turn numbers - a turn starts with each USER message
        current_turn = 0
        for i, msg in enumerate(history):
            # Increment turn number when we see a USER message
            if msg.role == Role.USER:
                current_turn += 1

            # Count tokens for this message and detect content types.
            # Use cached count when available — message content is immutable,
            # so the token count for a given message_id never changes.
            has_tool_result = False
            has_text = False
            text_content = ""
            tool_names = []
            mid = msg.message_id
            cached = self._msg_token_cache.get(mid)
            if cached is not None:
                msg_tokens = cached
                # Still need metadata (has_text, tool_names, etc.) for labelling
                for part in msg.parts:
                    if hasattr(part, 'text') and part.text:
                        has_text = True
                        text_content += part.text
                    elif hasattr(part, 'function_response') and part.function_response:
                        has_tool_result = True
                        if part.function_response.name:
                            tool_names.append(part.function_response.name)
            else:
                msg_tokens = 0
                for part in msg.parts:
                    if hasattr(part, 'text') and part.text:
                        msg_tokens += self._count_tokens(part.text)
                        has_text = True
                        text_content += part.text
                    elif hasattr(part, 'function_response') and part.function_response:
                        # Tool results (function_response is a ToolResult)
                        tr = part.function_response
                        result_text = str(tr.result) if tr.result else ''
                        msg_tokens += self._count_tokens(result_text)
                        has_tool_result = True
                        if tr.name:
                            tool_names.append(tr.name)
                self._msg_token_cache[mid] = msg_tokens

            conversation_tokens += msg_tokens

            # Add as child for per-turn drill-down
            if conv_entry:
                from .instruction_budget import ConversationTurnType, DEFAULT_TURN_POLICIES, GCPolicy
                # Determine turn type based on position and turn complexity
                if i == 0 and msg.role == Role.USER:
                    # First user message is the original request - LOCKED
                    turn_type = ConversationTurnType.ORIGINAL_REQUEST
                elif i == last_model_text_only_idx:
                    # Final text-only model response in a complex turn - TURN_SUMMARY (PRESERVABLE)
                    turn_type = ConversationTurnType.TURN_SUMMARY
                else:
                    # Everything else is working output - EPHEMERAL
                    turn_type = ConversationTurnType.WORKING
                gc_policy = DEFAULT_TURN_POLICIES[turn_type]

                # Determine descriptive label based on role and content type
                if msg.role == Role.MODEL:
                    role_label = "output (model)"
                elif has_tool_result:
                    # Handle tool results regardless of USER or TOOL role
                    if tool_names:
                        tools_str = ", ".join(tool_names)
                        role_label = f"input (tool = {tools_str})"
                    else:
                        role_label = "input (tool)"
                elif msg.role == Role.USER:
                    enrichments = self._get_framework_enrichments(text_content)
                    if enrichments:
                        enrichments_str = ", ".join(enrichments)
                        role_label = f"input (framework = {enrichments_str})"
                    else:
                        role_label = "input (external)"
                else:
                    role_label = msg.role.value if msg.role else "unknown"

                # Use message index i for unique key, but display actual turn number
                self._instruction_budget.add_child(
                    InstructionSource.CONVERSATION,
                    f"msg_{i}",  # Unique key using message index
                    msg_tokens,
                    gc_policy,
                    label=f"turn_{current_turn} {role_label}",  # Display turn number and type
                    message_ids=[msg.message_id],
                )

        self._instruction_budget.update_tokens(InstructionSource.CONVERSATION, conversation_tokens)

        # Emit budget update event
        self._emit_instruction_budget_update()

    def _update_thinking_budget(self, thinking_tokens: int) -> None:
        """Update THINKING entry in instruction budget with cumulative thinking tokens."""
        if not self._instruction_budget:
            return

        entry = self._instruction_budget.get_entry(InstructionSource.THINKING)
        if entry:
            entry.tokens += thinking_tokens
        else:
            self._instruction_budget.set_entry(
                InstructionSource.THINKING,
                tokens=thinking_tokens,
                label="Thinking",
            )
        self._emit_instruction_budget_update()

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Read accessor for the session's resolved tool schemas.

        Returns the list of :class:`ToolSchema` instances the session
        has activated for the current model — this is the resolved
        subset (preloaded plugins + on-demand activations), not the
        registry's full exposed set.

        Returns an empty list when ``configure()`` hasn't run yet
        (``self._tools is None``) — callers can iterate the result
        unconditionally without a None check.

        Phase 3 §7c step 3b: replaces daemon-side reads of the
        private ``self._tools`` attribute (e.g. core.py's
        ``_build_tool_id_mappings``).  Pre-§7c-step-3b the daemon
        reached into the private list directly.
        """
        return list(self._tools) if self._tools else []

    def refresh_tools(self) -> None:
        """Refresh tools from the runtime.

        Call this after enabling/disabling tools in the registry to update
        the session's tool configuration. Preserves conversation history.
        """
        if not self._provider or not self._executor:
            return

        # Refresh runtime's cache first
        self._runtime.refresh_tool_cache()

        # Get updated tool schemas and executors from runtime
        self._tools = self._runtime.get_tool_schemas(self._tool_plugins)
        executors = self._runtime.get_executors(self._tool_plugins)

        # Clear and re-register executors
        self._executor.clear_executors()
        for name, fn in executors.items():
            self._executor.register(name, fn)

        # Re-register the model command executor
        self._executor.register("model", self._execute_model_command)

        # Re-register session plugin executors if available
        if self._session_plugin and hasattr(self._session_plugin, 'get_executors'):
            for name, fn in self._session_plugin.get_executors().items():
                self._executor.register(name, fn)

        # Add session plugin tool schemas if available
        if self._session_plugin and hasattr(self._session_plugin, 'get_tool_schemas'):
            session_schemas = self._session_plugin.get_tool_schemas()
            if session_schemas:
                self._tools = list(self._tools) if self._tools else []
                self._tools.extend(session_schemas)

    def activate_discovered_tools(self, tool_names: List[str]) -> List[str]:
        """Activate discovered tools so the model can call them.

        When deferred tool loading is enabled, discoverable tools are not
        initially sent to the provider. When the model discovers tools via
        get_tool_schemas, this method activates them by adding their schemas
        to the provider's declared tools.

        If the newly-activated tool belongs to a plugin whose system
        instructions were deferred (because it had no core tools), those
        instructions are injected into ``self._system_instruction`` and
        tracked in the instruction budget at this point — not before.

        Args:
            tool_names: Names of tools to activate.

        Returns:
            List of tool names that were actually activated (not already active).
        """
        if not self._provider or not self._runtime.registry:
            return []

        # Get current tool names for dedup
        current_tool_names = {t.name for t in (self._tools or [])}
        activated = []

        # Build allowed plugin set from session's tool_plugins (profile filter)
        allowed_plugins: Optional[set] = None
        if self._tool_plugins is not None:
            allowed_plugins = set(self._tool_plugins)
            allowed_plugins.add("introspection")  # Always allowed

        # Get schemas for requested tools from registry
        all_schemas = self._runtime.registry.get_exposed_tool_schemas()
        schema_map = {s.name: s for s in all_schemas}

        for tool_name in tool_names:
            if tool_name in current_tool_names:
                continue  # Already active
            if tool_name not in schema_map:
                continue  # Tool doesn't exist

            # Enforce profile plugin filter: only activate tools from
            # plugins that the profile explicitly lists.
            if allowed_plugins is not None:
                plugin = self._runtime.registry.get_plugin_for_tool(tool_name)
                if plugin and plugin.name not in allowed_plugins:
                    self._trace(
                        f"activate_discovered_tools: skipping '{tool_name}' "
                        f"(plugin '{plugin.name}' not in profile)"
                    )
                    continue

            schema = schema_map[tool_name]
            if self._tools is None:
                self._tools = []
            self._tools.append(schema)
            current_tool_names.add(tool_name)
            activated.append(tool_name)

        # --- Update budget and system instructions for activated tools ---
        if activated and self._runtime.registry:
            self._track_activated_tools_in_budget(activated, schema_map)

        if activated:
            self._trace(f"Activating discovered tools: {activated}")
            self._emit_instruction_budget_update()

        return activated

    def _track_activated_tools_in_budget(
        self,
        activated: List[str],
        schema_map: Dict[str, 'ToolSchema'],
    ) -> None:
        """Track newly-activated tools in the instruction budget.

        Each tool's schema tokens are accumulated under its owning plugin's
        budget entry.  If the plugin had its system instructions deferred
        (because it had no core tools), those instructions are injected into
        ``self._system_instruction`` and the budget on first discovery.

        This keeps the budget panel clean: one entry per plugin, never
        per-tool entries.

        Args:
            activated: Tool names that were just activated.
            schema_map: Mapping of tool name to ToolSchema.
        """
        import json
        from .instruction_budget import GCPolicy

        registry = self._runtime.registry

        # Group activated tools by their owning plugin.
        plugin_tools: Dict[str, List[str]] = {}
        for tool_name in activated:
            plugin = registry.get_plugin_for_tool(tool_name)
            if plugin:
                plugin_tools.setdefault(plugin.name, []).append(tool_name)

        for plugin_name, tool_names_in_plugin in plugin_tools.items():
            plugin = registry.get_plugin(plugin_name)

            # --- Inject deferred system instructions (once per plugin) ---
            if plugin_name in self._deferred_plugin_instructions:
                self._deferred_plugin_instructions.discard(plugin_name)
                if plugin and hasattr(plugin, 'get_system_instructions'):
                    instr = plugin.get_system_instructions()
                    if instr:
                        if self._system_instruction:
                            self._system_instruction = self._system_instruction + "\n\n" + instr
                        else:
                            self._system_instruction = instr
                        self._trace(
                            f"Injected deferred system instructions for plugin: "
                            f"{plugin_name}"
                        )

            # --- Accumulate tool schema tokens under the plugin entry ---
            if not self._instruction_budget:
                continue

            # Sum schema tokens for all tools activated in this batch
            batch_tokens = 0
            for tool_name in tool_names_in_plugin:
                schema = schema_map.get(tool_name)
                if not schema:
                    continue
                try:
                    schema_dict = {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.parameters,
                    }
                    schema_json = json.dumps(schema_dict, indent=2)
                    batch_tokens += self._count_tokens(schema_json)
                except Exception:
                    pass

            if batch_tokens == 0:
                continue

            # Check if the plugin already has a budget entry (e.g. from
            # initial core tools, or a previous discovery batch).
            plugin_entry = self._instruction_budget.get_entry(InstructionSource.PLUGIN)
            existing = plugin_entry.children.get(plugin_name) if plugin_entry else None

            if existing is not None:
                # Accumulate into the existing entry
                existing.tokens += batch_tokens
            else:
                # First time this plugin appears in the budget — create
                # entry with instructions tokens (if any) + schema tokens.
                instr_tokens = 0
                if plugin and hasattr(plugin, 'get_system_instructions'):
                    instr = plugin.get_system_instructions()
                    if instr:
                        instr_tokens = self._count_tokens(instr)

                try:
                    self._instruction_budget.add_child(
                        InstructionSource.PLUGIN,
                        plugin_name,
                        instr_tokens + batch_tokens,
                        DEFAULT_TOOL_POLICIES[PluginToolType.CORE],
                        label=plugin_name,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to track plugin {plugin_name} in budget: {e}"
                    )

    def _register_model_command(self) -> None:
        """Register the built-in model command for listing and switching models."""
        from jaato_sdk.plugins.base import CommandParameter

        # Define the command with subcommand parameter
        model_cmd = UserCommand(
            name="model",
            description="Manage models: list, select <name>",
            share_with_model=False,
            parameters=[
                CommandParameter(
                    name="subcommand",
                    description="Subcommand: list, select",
                    required=False
                ),
                CommandParameter(
                    name="model_name",
                    description="Model name (for select)",
                    required=False
                )
            ]
        )

        # Register command
        self._user_commands["model"] = model_cmd

        # Register executor
        if self._executor:
            self._executor.register("model", self._execute_model_command)

    def _execute_model_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the model command.

        Subcommands:
            list   - Show available models and current model
            select - Switch to a different model

        Args:
            args: Command arguments with 'subcommand' and optionally 'model_name'.

        Returns:
            Dict with command result.
        """
        subcommand = args.get("subcommand", "").lower()
        model_name = args.get("model_name")

        # No subcommand - show current model and brief usage
        if not subcommand:
            return {
                "current_model": self._model_name,
                "subcommands": {
                    "list": "Show available models",
                    "select <name>": "Switch to a different model",
                    "help": "Show detailed help"
                }
            }

        # Help subcommand
        if subcommand == "help":
            return {
                "help": """Model Command

Switch between AI models during a session. The model command allows you to
list available models and switch to a different one without losing context.

USAGE
    model [subcommand] [args]

SUBCOMMANDS
    (none)            Show current model and available subcommands

    list              List all available models for the current provider
                      Shows which model is currently active

    select <name>     Switch to a different model
                      Preserves conversation history

    help              Show this help message

EXAMPLES
    model                         Show current model
    model list                    List available models
    model select gpt-4            Switch to gpt-4
    model select claude-3-opus    Switch to Claude Opus

NOTES
    - Switching models preserves your conversation history
    - Available models depend on your configured provider
    - Some models may have different capabilities or costs
    - Use 'model list' to see all available options"""
            }

        # List subcommand
        if subcommand == "list":
            # Use session's provider if available (faster, no new API connection)
            if self._provider and hasattr(self._provider, 'list_models'):
                models = self._provider.list_models()
            else:
                models = self._runtime.list_available_models(
                    provider_name=self._provider_name_override
                )
            return {
                "current_model": self._model_name,
                "available_models": models
            }

        # Select subcommand
        if subcommand == "select":
            if not model_name:
                return {
                    "error": "Model name required",
                    "usage": "model select <name>",
                    "hint": "Use 'model list' to see available models"
                }

            available = self._runtime.list_available_models(
                provider_name=self._provider_name_override
            )
            if model_name not in available:
                return {
                    "error": f"Model '{model_name}' not found",
                    "available_models": available
                }

            # Preserve current history
            history = self.get_history()

            # Update model name
            old_model = self._model_name
            self._model_name = model_name

            # Create new provider for the new model (preserving provider override)
            self._provider = self._runtime.create_provider(
                model_name,
                provider_name=self._provider_name_override
            )

            # Propagate agent context to new provider for trace identification
            if hasattr(self._provider, 'set_agent_context'):
                self._provider.set_agent_context(
                    agent_type=self._agent_type,
                    agent_name=self._agent_name,
                    agent_id=self._agent_id
                )

            # Update reliability plugin with new model context
            if self._runtime.reliability_plugin:
                available = self._runtime.list_available_models(
                    provider_name=self._provider_name_override
                )
                self._runtime.reliability_plugin.set_model_context(model_name, available)

            return {
                "success": True,
                "previous_model": old_model,
                "current_model": model_name,
                "history_preserved": True,
                "message": f"Switched from {old_model} to {model_name}"
            }

        # Unknown subcommand
        return {
            "error": f"Unknown subcommand: {subcommand}",
            "valid_subcommands": ["list", "select"]
        }

    def _register_telepathy_tool(self) -> None:
        """Register the built-in share_context tool for agent communication.

        This tool allows any agent (main or subagent) to share structured
        context with its parent agent. It's a native session capability,
        not tied to any specific plugin.
        """
        # Only register if we have a parent session (subagents can share with parent)
        # Main agent can also use this to share with subagents via the subagent plugin
        share_context_schema = ToolSchema(
            name='share_context',
            description=(
                'Share context from your memory with your parent agent. '
                'Use this to transfer knowledge without the parent needing to '
                're-read files or re-execute tools.\n\n'
                'CRITICAL: Share the COMPLETE file content, not summaries or excerpts. '
                'The parent needs the full content to work with it. Never omit content '
                '"for brevity" - that defeats the purpose of this tool.\n\n'
                'IMPORTANT: Do NOT re-read files before sharing. Use your memory of files '
                'you have already read. Copy the full content from your context.\n\n'
                'Use this to:\n'
                '- Share complete file contents you have already read\n'
                '- Share your analysis or findings\n'
                '- Share relevant facts you have discovered'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "object",
                        "description": (
                            "Files to share from your memory. Keys are file paths, "
                            "values are the COMPLETE file content from your context. "
                            "Do NOT summarize or omit content - share the full text."
                        ),
                        "additionalProperties": {"type": "string"}
                    },
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Key findings, facts, or conclusions to share. "
                            "These should be insights from your analysis."
                        )
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Free-form context, analysis, guidance, or explanation "
                            "to help the parent agent understand the shared context."
                        )
                    }
                },
                "required": []
            }
        )

        # Add tool schema to session tools
        if self._tools is None:
            self._tools = []
        self._tools = list(self._tools)
        self._tools.append(share_context_schema)

        # Register executor
        if self._executor:
            self._executor.register("share_context", self._execute_share_context)

    def _execute_share_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the share_context tool.

        Shares structured context with the parent agent.

        Args:
            args: Tool arguments containing:
                - files: Optional dict of path -> content from memory
                - findings: Optional list of key findings
                - notes: Optional free-form notes

        Returns:
            Status dict indicating success or error.
        """
        files = args.get('files', {})
        findings = args.get('findings', [])
        notes = args.get('notes', '')

        # Check if there's anything to share
        if not files and not findings and not notes:
            return {
                'success': False,
                'error': 'No context to share. Provide at least one of: files, findings, notes.'
            }

        # Check if we have a parent to share with
        if not self._parent_session:
            return {
                'success': False,
                'error': 'No parent session available. This agent may be the main agent.'
            }

        # Format the context
        formatted_context = self._format_shared_context(files, findings, notes)

        try:
            # Use same pattern as subagent communication: inject if busy, send if idle
            # CHILD source type - will be processed when parent is idle
            if self._parent_session.is_running:
                # Parent is busy - queue for idle processing
                self._parent_session.inject_prompt(
                    formatted_context,
                    source_id=self._agent_id,
                    source_type=SourceType.CHILD
                )
                return {
                    'success': True,
                    'status': 'queued',
                    'message': 'Context queued for parent. Will be processed when parent is idle.',
                    'shared': {
                        'files': list(files.keys()) if files else [],
                        'findings_count': len(findings) if findings else 0,
                        'has_notes': bool(notes)
                    }
                }

            # Parent is idle - this shouldn't normally happen (subagent runs while parent waits)
            # But handle it gracefully by injecting anyway
            self._parent_session.inject_prompt(
                formatted_context,
                source_id=self._agent_id,
                source_type=SourceType.CHILD
            )
            return {
                'success': True,
                'status': 'sent',
                'message': 'Context sent to parent.',
                'shared': {
                    'files': list(files.keys()) if files else [],
                    'findings_count': len(findings) if findings else 0,
                    'has_notes': bool(notes)
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to share context: {str(e)}'
            }

    def _format_shared_context(
        self,
        files: Dict[str, str],
        findings: List[str],
        notes: str
    ) -> str:
        """Format shared context for injection into parent's conversation.

        Args:
            files: Dict of file path -> content from memory
            findings: List of key findings
            notes: Free-form notes

        Returns:
            Formatted string for injection with instructions.
        """
        parts = []

        # Add instruction prefix so the receiving agent knows to use this content
        if files:
            parts.append(
                "CONTEXT FROM SUBAGENT: The following files and findings are shared from the subagent's memory. "
                "DO NOT re-read these files - use the content provided below directly."
            )
            parts.append("")

        parts.append('<shared_context from_agent="subagent">')

        if files:
            parts.append('<files>')
            for path, content in files.items():
                parts.append(f'<file path="{path}">')
                parts.append(content)
                parts.append('</file>')
            parts.append('</files>')

        if findings:
            parts.append('<findings>')
            for finding in findings:
                parts.append(f'  - {finding}')
            parts.append('</findings>')

        if notes:
            parts.append('<notes>')
            parts.append(notes)
            parts.append('</notes>')

        parts.append('</shared_context>')
        return '\n'.join(parts)

    def get_model_completions(self, args: List[str]) -> List['CommandCompletion']:
        """Get completions for the model command.

        Args:
            args: Arguments typed so far.

        Returns:
            List of CommandCompletion objects.
        """
        from jaato_sdk.plugins.base import CommandCompletion

        # No args yet - show subcommands
        if not args:
            return [
                CommandCompletion(value="list", description="Show available models"),
                CommandCompletion(value="select", description="Switch to a model"),
                CommandCompletion(value="help", description="Show detailed help"),
            ]

        subcommand = args[0].lower() if args else ""

        # Completing subcommand
        if len(args) == 1:
            subcommands = [
                ("list", "Show available models"),
                ("select", "Switch to a model"),
                ("help", "Show detailed help"),
            ]
            return [
                CommandCompletion(value=cmd, description=desc)
                for cmd, desc in subcommands
                if cmd.startswith(subcommand)
            ]

        # Completing model name for 'select' subcommand
        if subcommand == "select" and len(args) >= 2:
            prefix = args[1] if len(args) > 1 else ""
            models = self._runtime.list_available_models(
                provider_name=self._provider_name_override
            )
            if prefix:
                models = [m for m in models if m.startswith(prefix)]
            return [CommandCompletion(value=m, description="") for m in sorted(models)]

        return []

    def send_message(
        self,
        message: str,
        on_output: Optional[OutputCallback] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_gc_threshold: Optional[GCThresholdCallback] = None
    ) -> str:
        """Send a message to the model.

        Args:
            message: The user's message text.
            on_output: Optional callback for real-time output.
                Signature: (source: str, text: str, mode: str) -> None
            on_usage_update: Optional callback for real-time token usage.
                Signature: (usage: TokenUsage) -> None
            on_gc_threshold: Optional callback when GC threshold is crossed.
                Signature: (percent_used: float, threshold: float) -> None

        Returns:
            The final model response text.

        Raises:
            RuntimeError: If session is not configured.
        """
        if not self._configured:
            raise RuntimeError("Session not configured. Call configure() first.")

        # Lazy-init the provider on first model use (deferred-provider-INIT
        # design 2026-05-13).  The 9s/2-3s INIT cost happens here on first
        # send_message instead of during configure(), shifting it off the
        # bootstrap RPC critical path.  Idempotent + thread-safe.
        self._ensure_provider()
        if not self._provider:
            # skip_provider (auth-pending) mode and auth still hasn't
            # completed — fall through to the existing error path so
            # the caller sees a clear failure.
            raise RuntimeError(
                "Session has no provider — auth-pending mode and auth "
                "has not completed yet, OR _ensure_provider() returned "
                "without setting one (check configure() succeeded)"
            )

        # Block while an observation pause is active.  The observer
        # holds the lock briefly (snapshot, interrogation, etc.) and
        # releases it when done — the turn then proceeds normally.
        self._observation_lock.wait()

        self._trace(f"SESSION_SEND_MESSAGE len={len(message)} streaming={self._use_streaming}")

        # Increment turn counter
        self._turn_index += 1

        # Update permission context with turn index so evaluators can access it
        if self._executor:
            self._executor.update_permission_context(
                turn_index=self._turn_index,
                model_preamble=None,  # Reset at turn start
            )

        # Notify reliability plugin of turn start
        if self._runtime.reliability_plugin:
            self._runtime.reliability_plugin.on_turn_start(self._turn_index)

        # Ensure session/agent spans exist before the first turn
        self._ensure_telemetry_spans()

        # Wrap entire turn with telemetry span
        # Determine parent session ID for graph visualization
        _parent_sid = None
        if self._parent_session is not None:
            _parent_sid = getattr(self._parent_session, '_agent_id', None)

        with self._telemetry.turn_span(
            session_id=self._agent_id,
            agent_type=self._agent_type,
            agent_name=self._agent_name,
            turn_index=self._turn_index,
            parent_session_id=_parent_sid,
        ) as turn_span:
            self._current_turn_span = turn_span
            # Reset per-turn token accumulators for aggregating on the turn span
            self._turn_prompt_tokens = 0
            self._turn_completion_tokens = 0

            # Check and perform GC if needed (pre-send)
            if self._gc_plugin and self._gc_config and self._gc_config.check_before_send:
                self._maybe_collect_before_send()

            # Reset proactive GC tracking for this turn
            self._gc_threshold_crossed = False
            self._gc_threshold_callback = on_gc_threshold

            # Store output callback for this turn so enrichment can use it directly
            # This avoids the race condition where concurrent sessions overwrite
            # the shared registry callback
            self._current_output_callback = on_output

            # Wrap usage callback to check GC threshold
            wrapped_usage_callback = self._wrap_usage_callback_with_gc_check(on_usage_update)

            # Set output callback on registry BEFORE prompt enrichment
            # so enrichment notifications are visible to the user
            if self._runtime.registry and on_output:
                self._runtime.registry.set_output_callback(on_output, self._terminal_width)

            # Run prompt enrichment if registry is available
            processed_message = self._enrich_and_clean_prompt(message, turn_span=turn_span)

            try:
                response = self._run_chat_loop(processed_message, on_output, wrapped_usage_callback)
                turn_span.set_status_ok()
            except Exception as e:
                turn_span.record_exception(e)
                turn_span.set_status_error(str(e))
                raise

            # Record turn completion metadata
            turn_metadata = {}
            if self._is_cancelled():
                turn_metadata["cancelled"] = True
            turn_metadata["streaming"] = self._use_streaming
            if turn_metadata:
                turn_span.set_metadata(turn_metadata)

            # Proactive GC: if threshold was crossed during streaming, trigger GC now
            if self._gc_threshold_crossed and self._gc_plugin and self._gc_config:
                self._trace("PROACTIVE_GC: Threshold crossed during streaming, triggering post-turn GC")
                self._maybe_collect_after_turn()

            # Notify session plugin
            self._notify_session_turn_complete()

            # Notify reliability plugin of turn end
            if self._runtime.reliability_plugin:
                self._runtime.reliability_plugin.on_turn_end()

            self._current_turn_span = None
            return response

    def _wrap_usage_callback_with_gc_check(
        self,
        on_usage_update: Optional[UsageUpdateCallback]
    ) -> Optional[UsageUpdateCallback]:
        """Wrap usage callback for GC threshold check and cache usage tracking.

        When a cache plugin is attached, also forwards cache metrics
        from each usage update via ``extract_cache_usage()``.
        """
        _cache = getattr(self, '_cache_plugin', None)
        if not self._gc_plugin or not self._gc_config:
            # Even without GC, we may still need cache tracking
            if _cache:
                def cache_only_callback(usage: TokenUsage) -> None:
                    try:
                        _cache.extract_cache_usage(usage)
                    except Exception:
                        pass
                    if on_usage_update:
                        on_usage_update(usage)
                return cache_only_callback
            return on_usage_update

        def wrapped_callback(usage: TokenUsage) -> None:
            # Forward cache metrics to cache plugin
            if _cache:
                try:
                    _cache.extract_cache_usage(usage)
                except Exception:
                    pass
            # Check if threshold crossed
            if not self._gc_threshold_crossed and usage.total_tokens > 0:
                context_limit = self.get_context_limit()
                if context_limit > 0:
                    percent_used = (usage.total_tokens / context_limit) * 100
                    threshold = self._gc_config.threshold_percent if self._gc_config else 80.0

                    if percent_used >= threshold:
                        self._gc_threshold_crossed = True
                        self._trace(f"PROACTIVE_GC: Threshold crossed ({percent_used:.1f}% >= {threshold}%)")

                        # Notify via callback if provided
                        if self._gc_threshold_callback:
                            self._gc_threshold_callback(percent_used, threshold)

            # Call original callback if provided
            if on_usage_update:
                on_usage_update(usage)

        return wrapped_callback

    def _maybe_collect_after_turn(self) -> Optional[GCResult]:
        """Perform GC after turn if threshold was crossed during streaming."""
        if not self._gc_plugin or not self._gc_config:
            return None

        # Ensure background token counting is complete before GC so
        # eviction decisions use accurate counts, not estimates.
        if self._budget_counting_thread and self._budget_counting_thread.is_alive():
            self._trace("PROACTIVE_GC: Waiting for background token counting to finish...")
            self._budget_counting_thread.join(timeout=5.0)
            if self._budget_counting_thread.is_alive():
                self._trace("PROACTIVE_GC: Background counting still running after 5s, proceeding with estimates")

        context_usage = self.get_context_usage()
        history = self.get_history()

        # Build pre-GC telemetry attributes for span context
        gc_attrs = self._build_gc_span_attributes(
            context_usage, pre_collect=True,
        )

        with self._telemetry.gc_span(
            trigger_reason=GCTriggerReason.THRESHOLD.value,
            strategy=self._gc_plugin.name,
            attributes=gc_attrs,
        ) as gc_span:
            # Use THRESHOLD as the reason since it was triggered by threshold crossing
            new_history, result = self._gc_plugin.collect(
                history, context_usage, self._gc_config, GCTriggerReason.THRESHOLD,
                budget=self._instruction_budget,
            )

            if result.success:
                if result.items_collected == 0:
                    # GC ran but collected nothing - this is often surprising to users
                    self._trace(
                        f"PROACTIVE_GC: WARNING - GC triggered but collected 0 items. "
                        f"Check preserve_recent_turns setting vs actual turn count. "
                        f"Details: {result.details}"
                    )
                else:
                    self._trace(
                        f"PROACTIVE_GC: Collected {result.items_collected} items, "
                        f"freed {result.tokens_freed} tokens"
                    )
                new_history = ensure_tool_call_integrity(
                    new_history, trace_fn=lambda m: self._trace(f"PROACTIVE_GC: {m}"),
                )
                self._history.replace(new_history)
                self._gc_history.append(result)

                # Sync budget with GC changes (publishes cache invalidation events
                # on the active gc_span via on_gc_result callback)
                self._apply_gc_removal_list(result, gc_span=gc_span)
                self._emit_instruction_budget_update()

            # Populate post-GC span attributes from the result
            self._populate_gc_span_result(gc_span, result)

        return result

    def _build_llm_span_attributes(self) -> Dict[str, Any]:
        """Build the attribute dict to attach to an LLM telemetry span.

        Combines per-turn context (turn index) with cache plugin state
        (anchor, BP3 strategy, totals) so external observers can
        correlate LLM calls with the GC ↔ cache coordination dance.
        """
        attrs: Dict[str, Any] = {
            "jaato.turn_index": int(self._turn_index),
        }
        cache = getattr(self, "_cache_plugin", None)
        if cache and hasattr(cache, "get_telemetry_attributes"):
            try:
                cache_attrs = cache.get_telemetry_attributes() or {}
                attrs.update(cache_attrs)
            except Exception as e:
                self._trace(f"LLM_TELEMETRY: cache attr fetch failed: {e}")
        return attrs

    def _classify_cache_outcome(
        self,
        prompt_tokens: int,
        cache_read_tokens: Optional[int],
        cache_creation_tokens: Optional[int],
    ) -> str:
        """Classify a request's cache hit/miss outcome.

        ``prompt_tokens`` is the *new* (uncached) input only — matches
        Anthropic's ``input_tokens`` semantics, which jaato normalizes
        other providers to (see ``model_provider/anthropic/converters.py``).
        Total input therefore = ``cache_read_tokens + prompt_tokens``,
        and the hit ratio is ``cache_read_tokens / total_input`` — which
        naturally caps at 1.0.  Dividing by ``prompt_tokens`` alone
        produces ratios above 1.0 on cache-warm turns (e.g. 36.97 from
        26580 / 719) and misclassifies them as anomalies.

        Returns:
            "hit"   — most input tokens served from cache (>= 80%)
            "partial" — some input tokens served from cache (10-80%)
            "warm"  — cache was being written but not read (creation only)
            "miss"  — no cache reads, no creation
            "unknown" — usage data missing
        """
        read = cache_read_tokens or 0
        creation = cache_creation_tokens or 0
        new_input = prompt_tokens or 0
        total_input = read + new_input
        if total_input <= 0:
            return "unknown"
        ratio = read / total_input
        if ratio >= 0.8:
            return "hit"
        if ratio >= 0.1:
            return "partial"
        if creation > 0:
            return "warm"
        return "miss"

    def _populate_llm_span_outcome(
        self,
        llm_span: Any,
        response: Optional['ProviderResponse'],
    ) -> None:
        """Populate an LLM span with cache outcome derived from the response.

        Called after ``provider.complete()`` returns.  Extracts the
        cache hit/miss classification from response.usage and sets it
        as a span attribute.
        """
        if not llm_span or not response or not getattr(response, "usage", None):
            return
        try:
            usage = response.usage
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            read = getattr(usage, "cache_read_tokens", None)
            creation = getattr(usage, "cache_creation_tokens", None)
            llm_span.set_attribute("cache.read_tokens", int(read or 0))
            llm_span.set_attribute("cache.creation_tokens", int(creation or 0))
            llm_span.set_attribute(
                "cache.outcome",
                self._classify_cache_outcome(prompt, read, creation),
            )
        except Exception as e:
            self._trace(f"LLM_TELEMETRY: failed to populate cache outcome: {e}")

    def _build_gc_span_attributes(
        self, context_usage: Dict[str, Any], pre_collect: bool = True,
    ) -> Dict[str, Any]:
        """Build the initial attribute dict for a GC telemetry span.

        Captures budget state, cache anchor (if a cache plugin is active),
        and context usage at the moment GC is about to run.  These are
        the "before" values; ``_populate_gc_span_result`` adds the "after"
        values once GC completes.

        Args:
            context_usage: Output of ``get_context_usage()``.
            pre_collect: Reserved for future divergence between pre/post
                attribute sets.  Currently always True.

        Returns:
            Dict of OTel-friendly attributes.
        """
        attrs: Dict[str, Any] = {
            "gc.percent_used": float(context_usage.get("percent_used", 0)),
            "gc.tokens_total": int(context_usage.get("total_tokens", 0)),
            "gc.context_limit": int(context_usage.get("context_limit", 0)),
        }
        if self._instruction_budget:
            try:
                attrs["gc.tokens_before"] = int(self._instruction_budget.total_tokens())
            except Exception:
                pass
        # Cache anchor (if any cache plugin exposes it)
        cache = getattr(self, "_cache_plugin", None)
        if cache and hasattr(cache, "get_cache_anchor_message_id"):
            try:
                anchor = cache.get_cache_anchor_message_id()
                if anchor:
                    attrs["gc.cache_anchor_message_id"] = anchor
            except Exception:
                pass
        return attrs

    def _populate_gc_span_result(self, gc_span: Any, result: 'GCResult') -> None:
        """Populate a GC span with attributes derived from the GC result.

        Called after ``gc_plugin.collect()`` returns.  The span receives
        per-phase counts and aggregate metrics so external observers can
        correlate GC operations with subsequent cache hit/miss outcomes.

        Args:
            gc_span: The active OTel span (or no-op span when telemetry
                is disabled).
            result: The ``GCResult`` from ``gc_plugin.collect()``.
        """
        if not gc_span:
            return
        try:
            gc_span.set_attribute("gc.success", bool(result.success))
            gc_span.set_attribute("gc.items_collected", int(result.items_collected))
            gc_span.set_attribute("gc.tokens_freed", int(result.tokens_freed))
            gc_span.set_attribute("gc.tokens_after", int(result.tokens_after))
            # Per-phase counts come from result.details
            details = result.details or {}
            for key in (
                "ephemeral_removed",
                "partial_removed",
                "preservable_removed",
                "enrichment_cleared",
                "tokens_to_free",
                "target_tokens",
            ):
                if key in details:
                    val = details[key]
                    # bool first to avoid being treated as int
                    if isinstance(val, bool):
                        gc_span.set_attribute(f"gc.{key}", val)
                    elif isinstance(val, (int, float)):
                        gc_span.set_attribute(f"gc.{key}", val)
        except Exception as e:
            self._trace(f"GC_TELEMETRY: failed to populate span attrs: {e}")

    def _apply_gc_removal_list(
        self, result: GCResult, gc_span: Any = None,
    ) -> None:
        """Apply GC removal list to instruction budget.

        This synchronizes the budget with the actual history changes made by GC.
        Must be called after a successful GC operation.

        Args:
            result: The GCResult containing the removal_list.
            gc_span: Optional active GC telemetry span; passed to the
                cache plugin's ``on_gc_result`` so it can emit cache
                invalidation events on the same span.
        """
        if not self._instruction_budget or not result.removal_list:
            return

        for item in result.removal_list:
            if item.child_key:
                # Remove specific child entry
                self._instruction_budget.remove_child(item.source, item.child_key)
            else:
                # Bulk clear entire source (e.g., ENRICHMENT)
                entry = self._instruction_budget.get_entry(item.source)
                if entry:
                    entry.tokens = 0
                    entry.children.clear()

        # If summary was created (summarize/hybrid plugins), add summary entry
        summary_tokens = result.details.get("summary_tokens")
        if summary_tokens and summary_tokens > 0:
            # Find or create a unique summary key
            conv_entry = self._instruction_budget.get_entry(InstructionSource.CONVERSATION)
            if conv_entry:
                # Count existing summaries to generate unique key
                summary_count = sum(
                    1 for key in conv_entry.children.keys()
                    if key.startswith("gc_summary_")
                )
                summary_key = f"gc_summary_{summary_count + 1}"
                self._instruction_budget.add_child(
                    source=InstructionSource.CONVERSATION,
                    child_key=summary_key,
                    tokens=summary_tokens,
                    gc_policy=GCPolicy.PRESERVABLE,
                    label=f"Context Summary #{summary_count + 1}",
                    metadata={"created_by": result.plugin_name},
                )

        self._trace(
            f"GC_BUDGET_SYNC: Applied {len(result.removal_list)} removals to budget"
        )

        # Notify cache plugin about GC so it can track prefix invalidation.
        # The cache plugin may emit a 'cache.prefix_invalidated' event on
        # the active gc_span (when provided) so the GC↔cache coordination
        # is visible in the trace.
        _cache = getattr(self, '_cache_plugin', None)
        if _cache and hasattr(_cache, 'on_gc_result'):
            try:
                # Try the span-aware signature first; fall back to legacy
                # call if the cache plugin only accepts the result.
                try:
                    _cache.on_gc_result(result, gc_span=gc_span)
                except TypeError:
                    _cache.on_gc_result(result)
            except Exception as e:
                self._trace(f"CACHE_PLUGIN: on_gc_result failed: {e}")

    def _enrich_and_clean_prompt(self, prompt: str, turn_span=None) -> str:
        """Run prompt through enrichment pipeline and strip @references.

        Args:
            prompt: The user prompt to enrich.
            turn_span: Optional span context for emitting enrichment telemetry
                events.  When provided, any ``_telemetry`` dicts found in the
                enrichment metadata are forwarded as span events.
        """
        enriched_prompt = prompt

        # Run through plugin enrichment pipeline
        if self._runtime.registry:
            result = self._runtime.registry.enrich_prompt(prompt)
            enriched_prompt = result.prompt

            # Forward enrichment telemetry as span events on the turn span
            if turn_span and result.metadata:
                for plugin_name, meta in result.metadata.items():
                    if isinstance(meta, dict):
                        telem = meta.get('_telemetry')
                        if isinstance(telem, dict):
                            turn_span.add_event(
                                f'enrichment.prompt.{plugin_name}',
                                telem,
                            )

        # Strip @references
        return AT_REFERENCE_PATTERN.sub(r'\1', enriched_prompt)

    # -- TurnResult helpers -----------------------------------------------
    #
    # These methods consolidate the three previously distinct error
    # mechanisms (exceptions, finish-reason checks, boolean tuples)
    # into a single TurnResult-based flow.  They are used exclusively
    # by ``_run_chat_loop`` and its sub-methods.

    def _maybe_rewind(self, response: ProviderResponse) -> bool:
        """Attempt a rewind-with-hint recovery for truncated tool calls.

        Detects the ``MAX_TOKENS``-truncated tool call pathology (see
        ``shared/rewind.py``), rewrites the last assistant message to
        preserve its narration text while dropping the half-serialized
        ``tool_use`` part, and appends a synthetic user-role hint
        naming the specific tool and the truncation reason.  On a hit,
        the caller should ``continue`` its request loop to re-request
        the provider with the rewritten history.

        Bounded by :data:`REWIND_BUDGET_PER_OPERATION` — after the cap
        is reached the pathology is allowed to surface normally via
        the existing abnormal-termination path in
        :meth:`_classify_finish_reason`.  The counter resets on any
        successful tool execution (see the tool-call completion path).

        Telemetry: on fire, sets ``jaato.rewind.reason``,
        ``jaato.rewind.tool``, and ``jaato.rewind.count`` on the active
        turn span.

        Args:
            response: The just-received :class:`ProviderResponse` whose
                function calls will be inspected.

        Returns:
            True when the detector fired AND the rewind budget
            allowed it AND narration text was successfully preserved —
            caller should re-request the provider.  False otherwise
            (no detection, budget exhausted, or nothing to preserve).
        """
        from .rewind import (
            detect_truncated_tool_call,
            find_truncated_call_name,
        )

        tool_schemas = (
            self._runtime.registry.get_exposed_tool_schemas()
            if self._runtime and self._runtime.registry else []
        )
        reason = detect_truncated_tool_call(response, tool_schemas)
        if reason is None:
            return False

        if self._rewind_count >= REWIND_BUDGET_PER_OPERATION:
            self._trace(
                f"REWIND_BUDGET_EXHAUSTED reason={reason} "
                f"count={self._rewind_count}, letting failure surface"
            )
            return False

        bad_call_name = (
            find_truncated_call_name(response, tool_schemas) or "unknown"
        )
        preserved = self._history.rewrite_last_dropping_tool_use()
        if preserved is None:
            # No narration to anchor the hint on — fall through.
            self._trace(
                f"REWIND_SKIP reason={reason} tool={bad_call_name}: "
                "no narration preserved"
            )
            return False

        self._history.append(
            Message.from_text(Role.USER, self._build_rewind_hint_text(
                bad_call_name, reason, preserved,
            )),
        )
        self._rewind_count += 1

        span = getattr(self, '_current_turn_span', None)
        if span is not None:
            try:
                span.set_attribute("jaato.rewind.reason", reason)
                span.set_attribute("jaato.rewind.tool", bad_call_name)
                span.set_attribute("jaato.rewind.count", self._rewind_count)
            except Exception as exc:
                logger.debug(f"Failed to set rewind telemetry: {exc}")

        self._trace(
            f"REWIND fired reason={reason} tool={bad_call_name} "
            f"count={self._rewind_count} preserved_chars={len(preserved)}"
        )
        return True

    def _build_rewind_hint_text(
        self,
        tool_name: str,
        reason: str,
        preserved_narration: str,
    ) -> str:
        """Compose the synthetic user-role hint injected after a rewind.

        The hint references the model's own preserved narration, names
        the specific tool, explains the detection reason, and offers a
        concrete recipe for working around the truncation.  Keeping the
        message conversational ("yes, and here's the right way…") is
        deliberate — a bare "try again" would read as an unexplained
        reset to the model.
        """
        # Trim narration to keep the hint compact; the model has its own
        # narration in the preceding assistant turn anyway — we just
        # need enough to anchor the correction.
        if len(preserved_narration) > 200:
            preserved_narration = preserved_narration[:197] + "…"

        reason_explanation = {
            "max_tokens_empty_args": (
                "the completion budget truncated your arguments — the "
                "tool call arrived with empty arguments."
            ),
            "max_tokens_missing_required": (
                "the completion budget truncated your arguments — the "
                "tool call is missing required fields."
            ),
        }.get(reason, "your tool call arguments were truncated.")

        return (
            f"Before that `{tool_name}` call lands: {reason_explanation} "
            f"Your narration just before the call was: "
            f"\"{preserved_narration}\". "
            f"Please write this in smaller pieces instead — for a large "
            f"file, start with a skeleton call (e.g. outline or placeholder "
            f"content), then append each section with follow-up calls. "
            f"For shell or CLI tools with long inputs, split the work "
            f"across multiple invocations. Try again now."
        )

    def _classify_finish_reason(
        self,
        response: ProviderResponse,
    ) -> Optional[TurnResult]:
        """Classify a provider response's finish reason.

        Returns a ``TurnResult`` for abnormal terminations (``SAFETY``,
        ``MAX_TOKENS``, ``ERROR``) and ``None`` for reasons that the
        chat loop should continue processing (``STOP``, ``UNKNOWN``,
        ``TOOL_USE``, ``CANCELLED``).

        ``CANCELLED`` is handled separately by ``_handle_cancellation``
        because it requires additional logic (mid-turn interrupts,
        UI notification, model notification).
        """
        if response.finish_reason in (
            FinishReason.STOP,
            FinishReason.UNKNOWN,
            FinishReason.TOOL_USE,
            FinishReason.CANCELLED,
        ):
            return None
        logger.warning(f"Model stopped with finish_reason={response.finish_reason}")
        return TurnResult.from_finish_reason(
            response.finish_reason, response.get_text()
        )

    def _handle_cancellation(
        self,
        response: ProviderResponse,
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any],
        cancellation_notified: bool,
        accumulated_text: Optional[List[str]] = None,
        context: str = "",
    ) -> _CancellationResult:
        """Check for cancellation or mid-turn interrupt and return a result.

        This unifies the ~11 cancellation check sites that previously
        existed throughout ``_run_chat_loop``.

        Returns a ``_CancellationResult`` whose ``action`` field tells
        the caller what to do:

        - ``"continue"``: Not cancelled; keep processing normally.
        - ``"end_turn"``: Turn should end; use ``turn_result``.
        - ``"switch_response"``: A mid-turn interrupt produced a new
          response; continue with ``new_response``.

        Args:
            response: The most recent provider response.
            use_streaming: Whether streaming is enabled.
            on_output: Output callback for UI notifications.
            wrapped_usage_callback: Token usage callback.
            turn_data: Mutable turn accounting dict.
            cancellation_notified: Whether the UI has already been told
                about cancellation (prevents duplicate messages).
            accumulated_text: Text accumulated so far in this turn.
            context: Human-readable description of where in the loop
                cancellation was detected (for tracing).

        Returns:
            A ``_CancellationResult`` describing the action to take.
        """
        if not self._is_cancelled() and response.finish_reason != FinishReason.CANCELLED:
            return _CancellationResult(action="continue")

        partial_text = response.get_text()

        # --- Mid-turn interrupt path ---
        # Distinguish user-initiated cancellation from a mid-turn interrupt
        # (a parent/user message that arrived while the model was streaming).
        # The streaming callbacks encode the reason on the cancel token so we
        # don't need a separate boolean flag.
        is_mid_turn_interrupt = (
            self._cancel_token is not None
            and self._cancel_token.cancel_reason == "mid_turn_interrupt"
        )
        if is_mid_turn_interrupt:
            self._trace(
                f"MID_TURN_INTERRUPT: Processing user prompt "
                f"({context}, partial: {len(partial_text) if partial_text else 0} chars)"
            )
            self._cancel_token = CancelToken()

            # Peek at the pending prompt for the callback
            pending_prompts = self._message_queue.peek_all()
            user_prompt_preview = ""
            for msg in pending_prompts:
                if msg.source_type in (SourceType.USER, SourceType.PARENT, SourceType.SYSTEM):
                    user_prompt_preview = msg.text[:100] if msg.text else ""
                    break

            if self._on_mid_turn_interrupt:
                self._on_mid_turn_interrupt(
                    len(partial_text) if partial_text else 0,
                    user_prompt_preview,
                )

            mid_turn_response = self._check_and_handle_mid_turn_prompt(
                use_streaming, on_output, wrapped_usage_callback, turn_data
            )
            if mid_turn_response:
                return _CancellationResult(
                    action="switch_response",
                    new_response=mid_turn_response,
                )
            else:
                self._trace(
                    f"MID_TURN_INTERRUPT: No prompt in queue ({context}), returning partial"
                )
                return _CancellationResult(
                    action="end_turn",
                    turn_result=TurnResult.success(partial_text or ""),
                )

        # --- Normal cancellation path ---
        reason = self._cancel_token.cancel_reason if self._cancel_token else ""
        reason_suffix = f" ({reason})" if reason else ""
        cancel_msg = f"[Generation cancelled{reason_suffix}]"
        if on_output and not cancellation_notified:
            self._trace(f"CANCEL_NOTIFY: {cancel_msg} ({context})")
            on_output("system", cancel_msg, "write")

        # Merge any accumulated text with partial text from the response
        if accumulated_text:
            all_text = ''.join(accumulated_text)
        else:
            all_text = partial_text

        self._notify_model_of_cancellation(cancel_msg, all_text)
        return _CancellationResult(
            action="end_turn",
            turn_result=TurnResult.cancelled(all_text, context=context),
        )

    def _emit_text_parts(
        self,
        response: ProviderResponse,
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        accumulated_text: List[str],
    ) -> None:
        """Emit text parts from a response to UI and accumulate them.

        This deduplicates the pattern that appears ~6 times in the chat
        loop where we iterate over response parts, emit non-streaming
        text, forward to parent, notify reliability, and append to
        the accumulated text list.

        Args:
            response: The provider response to extract text from.
            use_streaming: Whether streaming is enabled (text already
                emitted via callback in streaming mode).
            on_output: Output callback for non-streaming text emission.
            accumulated_text: Mutable list to append text parts to.
        """
        for part in response.parts:
            if part.text:
                if not use_streaming:
                    if on_output:
                        on_output("model", part.text, "write")
                    self._forward_to_parent("MODEL_OUTPUT", part.text)
                    if self._runtime.reliability_plugin:
                        self._runtime.reliability_plugin.on_model_text(part.text)
                accumulated_text.append(part.text)

    def _nudge_for_tool_use(
        self,
        response: ProviderResponse,
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any],
        max_attempts: int = 3,
        context: str = "",
    ) -> ProviderResponse:
        """Nudge the model when it indicates TOOL_USE but emits no function calls.

        Some providers emit ``finish_reason=TOOL_USE`` without including
        function-call parts, or return empty responses with ``UNKNOWN``
        finish.  This helper injects a hidden prompt to push the model to
        execute the tool call it intended.

        Returns the (possibly updated) ``ProviderResponse``.  If nudging
        fails or is not needed, returns the original *response* unchanged.
        """
        for attempt in range(1, max_attempts + 1):
            has_fc = response.has_function_calls()
            is_empty = not response.parts or all(
                not p.text and not p.function_call for p in response.parts
            )
            needs_nudge = (
                (not has_fc and response.finish_reason == FinishReason.TOOL_USE)
                or (response.finish_reason == FinishReason.UNKNOWN and is_empty)
            )
            if not needs_nudge:
                break

            reason = (
                "TOOL_USE without function call"
                if response.finish_reason == FinishReason.TOOL_USE
                else "UNKNOWN with empty response"
            )
            self._trace(
                f"NUDGE_REQUIRED: {reason} ({context}, "
                f"attempt {attempt}/{max_attempts})"
            )
            nudge_prompt = (
                "<hidden>Your previous response was incomplete or empty. "
                "You were in the middle of a task. Continue executing your plan. "
                "Do NOT describe or re-read files. Execute the next tool call directly. "
                "Your next response MUST continue the task, not restart or summarize.</hidden>"
            )
            self._message_queue.put(nudge_prompt, "system", SourceType.SYSTEM)
            nudge_response = self._check_and_handle_mid_turn_prompt(
                use_streaming, on_output, wrapped_usage_callback, turn_data
            )
            if nudge_response:
                response = nudge_response
                self._trace(
                    f"NUDGE_RESULT: has_fc={response.has_function_calls()} ({context})"
                )
            else:
                self._trace(f"NUDGE_NO_RESPONSE: ({context})")
                break
        return response

    def _execute_tools_and_continue(
        self,
        fc_group: List[FunctionCall],
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any],
        cancellation_notified: bool,
        accumulated_text: Optional[List[str]] = None,
        context: str = "",
        check_mid_turn: bool = True,
    ) -> Tuple[Optional[ProviderResponse], Optional[TurnResult], bool]:
        """Execute a tool group, send results, and classify the continuation.

        This consolidates the repeated pattern of:

        1. Execute the tool group
        2. Check simple cancellation
        3. Send results to provider for continuation
        4. Check cancellation / mid-turn interrupt on the continuation
        5. Classify the finish reason for abnormal stops
        6. Nudge if TOOL_USE without function calls
        7. Optionally check for mid-turn prompts

        Args:
            fc_group: Function calls to execute.
            use_streaming: Whether streaming is enabled.
            on_output: Output callback.
            wrapped_usage_callback: Token usage callback.
            turn_data: Mutable turn accounting dict.
            cancellation_notified: Whether cancellation has been shown.
            accumulated_text: Text accumulated so far (for cancel messages).
            context: Tracing context string.
            check_mid_turn: Whether to check for queued mid-turn prompts
                after getting the continuation.  Set to ``False`` when
                called from within the mid-turn drain loop.

        Returns:
            ``(response, None, False)`` — processing should continue with the
            new response.
            ``(None, result, False)`` — the turn should end with the given
            ``TurnResult``.
            ``(response, None, True)`` — a mid-turn interrupt switched the
            response. The caller should stop iterating any remaining parts
            from the previous response and call
            ``_inject_synthetic_cancelled_results`` for unexecuted tool calls.
        """
        # Update permission context with model preamble so evaluators can
        # inspect what the model said before calling tools.
        if self._executor and accumulated_text:
            self._executor.update_permission_context(
                model_preamble=''.join(accumulated_text),
            )

        # 1. Execute the tool group
        tool_results = self._execute_function_call_group(
            fc_group, turn_data, on_output, cancellation_notified
        )

        # 2. Simple cancellation check after execution
        if self._is_cancelled():
            reason = self._cancel_token.cancel_reason if self._cancel_token else ""
            reason_suffix = f" ({reason})" if reason else ""
            cancel_msg = f"[Generation cancelled{reason_suffix}]"
            if on_output and not cancellation_notified:
                on_output("system", cancel_msg, "write")
            partial = ''.join(accumulated_text) if accumulated_text else ""
            self._notify_model_of_cancellation(cancel_msg, partial)
            return None, TurnResult.cancelled(partial, context=f"after tool execution ({context})"), False

        # 2.5. GC check between tool execution and next model call.
        # In agentic mode, send_message() is called once and the session
        # loops internally (tool → model → tool → ...) without returning.
        # The check_before_send at the top of send_message only runs once.
        # This intra-turn check ensures GC fires as context grows.
        if self._gc_plugin and self._gc_config and self._gc_config.check_before_send:
            # Refresh conversation budget so the GC sees current token usage
            self._update_conversation_budget()
            self._maybe_collect_before_send()

        # 3. Send results and get continuation
        response = self._send_tool_results_and_continue(
            tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
        )

        # 4. Handle cancellation / mid-turn interrupt on continuation
        cr = self._handle_cancellation(
            response, use_streaming, on_output, wrapped_usage_callback,
            turn_data, cancellation_notified, accumulated_text,
            context=f"after tool results ({context})",
        )
        if cr.action == "end_turn":
            return None, cr.turn_result, False
        if cr.action == "switch_response":
            # Mid-turn interrupt: caller must not process remaining old parts
            return cr.new_response, None, True

        # 5. Classify finish reason for abnormal stops
        abnormal = self._classify_finish_reason(response)
        if abnormal is not None:
            return None, abnormal, False

        # 6. Nudge if TOOL_USE without function calls
        response = self._nudge_for_tool_use(
            response, use_streaming, on_output, wrapped_usage_callback,
            turn_data, context=context,
        )

        # 7. Optionally check mid-turn prompts
        if check_mid_turn and not response.has_function_calls():
            mid_turn_response = self._check_and_handle_mid_turn_prompt(
                use_streaming, on_output, wrapped_usage_callback, turn_data
            )
            if mid_turn_response:
                response = mid_turn_response
                # Check cancellation on the mid-turn response
                cr = self._handle_cancellation(
                    response, use_streaming, on_output, wrapped_usage_callback,
                    turn_data, cancellation_notified, accumulated_text,
                    context=f"after mid-turn ({context})",
                )
                if cr.action == "end_turn":
                    return None, cr.turn_result, False
                if cr.action == "switch_response":
                    response = cr.new_response

        return response, None, False

    def _run_chat_loop(
        self,
        message: str,
        on_output: Optional[OutputCallback],
        on_usage_update: Optional[UsageUpdateCallback] = None
    ) -> str:
        """Internal function calling loop with streaming and cancellation support.

        Args:
            message: The user's message text.
            on_output: Optional callback for real-time output.
            on_usage_update: Optional callback for real-time token usage updates.

        Returns:
            The final response text.
        """
        # Set output callback on executor
        if self._executor:
            self._executor.set_output_callback(on_output)

        # Set output callback on registry for enrichment notifications
        if self._runtime.registry and on_output:
            self._runtime.registry.set_output_callback(on_output, self._terminal_width)

        # Initialize cancellation support
        self._cancel_token = CancelToken()
        self._is_running = True
        self._turn_complete.clear()
        cancellation_notified = False  # Track if we've already shown cancellation message
        terminal_event_sent = False  # Track if abnormal termination (CANCELLED/ERROR) occurred

        # Reset turn complexity tracking
        self._turn_model_response_count = 0
        self._turn_had_tool_calls = False

        # Track tokens and timing
        turn_start = datetime.now()
        turn_data = {
            'prompt': 0,
            'output': 0,
            'total': 0,
            'start_time': turn_start.isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'function_calls': [],
        }
        response: Optional[ProviderResponse] = None

        # Wrap usage callback to also update turn_data during streaming
        # This ensures we capture token values even if streaming is cancelled
        # Always enabled for internal turn tracking, regardless of external callback
        def usage_callback_with_turn_tracking(usage: TokenUsage) -> None:
            if usage.total_tokens > 0:
                turn_data['prompt'] = usage.prompt_tokens
                turn_data['output'] = usage.output_tokens
                turn_data['total'] = usage.total_tokens
            # Cache tokens: capture when present (streaming path)
            if usage.cache_read_tokens is not None:
                turn_data['cache_read'] = usage.cache_read_tokens
            if usage.cache_creation_tokens is not None:
                turn_data['cache_creation'] = usage.cache_creation_tokens
            if on_usage_update:
                on_usage_update(usage)

        wrapped_usage_callback = usage_callback_with_turn_tracking

        # Determine if we should use streaming
        use_streaming = (
            self._use_streaming and
            self._provider and
            hasattr(self._provider, 'supports_streaming') and
            self._provider.supports_streaming()
        )

        try:
            # Check for cancellation before starting (including parent)
            if self._is_cancelled():
                msg = "[Cancelled before start]"
                if on_output:
                    on_output("system", msg, "write")
                return msg

            # Proactive rate limiting: wait if needed before request
            self._pacer.pace()

            # Set activity phase: we're about to wait for LLM response
            self._set_activity_phase(ActivityPhase.WAITING_FOR_LLM)

            # Append user message to session history before provider call.
            # The message stays in history across retries (correct: the user DID send it).
            # Rolled back in the outer except block if all retries fail.
            self._history.append(Message.from_text(Role.USER, message))

            # Send message (streaming or batched) with telemetry.
            #
            # Wrapped in a ``while True`` retry loop to support the
            # rewind-with-hint recovery path for MAX_TOKENS-truncated
            # tool calls.  Normal runs iterate once and ``break``.  A
            # rewind detection rewrites history, injects a user-role
            # hint, and ``continue``s to re-request the provider —
            # bounded by ``REWIND_BUDGET_PER_OPERATION``.  See
            # ``docs/design/rewind-with-hint.md`` and
            # ``shared/rewind.py``.
            while True:
              with self._telemetry.llm_span(
                model=self._model_name or "unknown",
                provider=self._provider.name if self._provider else "unknown",
                streaming=use_streaming,
                attributes=self._build_llm_span_attributes(),
              ) as llm_telemetry:
                self._record_input_messages_telemetry(llm_telemetry)
                if use_streaming:
                    # Track whether we've sent the first chunk (to use "write" vs "append")
                    first_chunk_sent = False
                    # Track accumulated text for mid-turn interrupt preservation
                    accumulated_streaming_text: List[str] = []

                    # Streaming callback that routes to on_output and forwards to parent
                    def streaming_callback(chunk: str) -> None:
                        nonlocal first_chunk_sent
                        # Accumulate text for potential mid-turn interrupt preservation
                        accumulated_streaming_text.append(chunk)

                        # Notify reliability plugin of model text for pattern detection
                        if self._runtime.reliability_plugin:
                            self._runtime.reliability_plugin.on_model_text(chunk)

                        # Check for pending mid-turn prompts during streaming
                        # This allows user input to interrupt the current generation
                        if self._message_queue.has_parent_messages():
                            self._trace("MID_TURN_INTERRUPT: Detected pending user prompt during streaming")
                            if self._cancel_token:
                                self._cancel_token.cancel(reason="mid_turn_interrupt")
                            # Don't return - let the current chunk be processed first

                        # Transition to STREAMING phase on first chunk
                        if not first_chunk_sent:
                            self._set_activity_phase(ActivityPhase.STREAMING)
                        if on_output:
                            # First chunk uses "write" to start block, subsequent use "append"
                            mode = "append" if first_chunk_sent else "write"
                            self._trace(f"SESSION_OUTPUT mode={mode} len={len(chunk)} preview={repr(chunk[:50])}")
                            on_output("model", chunk, mode)
                            first_chunk_sent = True
                        # Forward model output to parent for real-time visibility
                        self._forward_to_parent("MODEL_OUTPUT", chunk)

                    self._trace(f"STREAMING on_usage_update={'set' if wrapped_usage_callback else 'None'}")

                    # Create thinking callback to emit thinking BEFORE text
                    def thinking_callback(thinking: str) -> None:
                        if on_output:
                            self._trace(f"SESSION_THINKING_CALLBACK len={len(thinking)}")
                            on_output("thinking", thinking, "write")

                    with self._provider_access():
                        turn_result, _retry_stats = with_retry(
                            lambda: self._provider.complete(
                                self._history.messages,
                                system_instruction=self._get_effective_system_instruction(),
                                tools=self._get_tools_for_provider(),
                                on_chunk=streaming_callback,
                                cancel_token=self._cancel_token,
                                on_usage_update=wrapped_usage_callback,
                                on_thinking=thinking_callback,
                                # Note: on_function_call is intentionally NOT used here.
                                # The SDK may deliver function calls before preceding text,
                                # which would cause tool trees to appear in wrong positions.
                                # Tool trees are displayed during parts processing instead.
                            ),
                            context="complete_streaming",
                            on_retry=self._on_retry,
                            cancel_token=self._cancel_token,
                            provider=self._provider
                        )
                else:
                    with self._provider_access():
                        turn_result, _retry_stats = with_retry(
                            lambda: self._provider.complete(
                                self._history.messages,
                                system_instruction=self._get_effective_system_instruction(),
                                tools=self._get_tools_for_provider(),
                            ),
                            context="complete",
                            on_retry=self._on_retry,
                            cancel_token=self._cancel_token,
                            provider=self._provider
                        )
                response = self._unwrap_turn_result(turn_result)

                # Record model response in session history
                self._add_model_response_to_history(response)
                self._record_token_usage(response)
                self._accumulate_turn_tokens(response, turn_data)
                # Track model response count for turn complexity
                self._turn_model_response_count += 1
                # Record token usage to telemetry span
                self._record_token_telemetry(llm_telemetry, response)
              self._trace(f"SESSION_STREAMING_COMPLETE parts_count={len(response.parts)} finish={response.finish_reason}")

              # Rewind-with-hint hook: detect MAX_TOKENS-truncated tool
              # calls, rewrite history to preserve narration, inject a
              # corrective user-role hint, and re-request the provider.
              # ``_maybe_rewind`` returns True when the rewind fired and
              # the budget allowed it — in which case we loop to
              # re-request.  Otherwise we break and fall through to the
              # existing abnormal-termination classifier.
              if self._maybe_rewind(response):
                continue
              break

            # Emit turn progress after initial response
            pending_calls = len([p for p in response.parts if p.function_call])
            self._emit_turn_progress(turn_data, pending_tool_calls=pending_calls)

            # Emit thinking content if present (non-streaming only).
            # For streaming, the provider emits thinking via on_thinking callback
            # before text starts, so we don't need to emit it again here.
            if not use_streaming and on_output and response.thinking:
                self._trace(f"SESSION_THINKING_OUTPUT len={len(response.thinking)}")
                on_output("thinking", response.thinking, "write")

            # Check finish_reason for abnormal termination
            abnormal = self._classify_finish_reason(response)
            if abnormal is not None:
                return abnormal.text

            # Check for cancellation / mid-turn interrupt after initial message
            cr = self._handle_cancellation(
                response, use_streaming, on_output, wrapped_usage_callback,
                turn_data, cancellation_notified,
                context="after initial message",
            )
            if cr.action == "end_turn":
                return cr.turn_result.text
            if cr.action == "switch_response":
                response = cr.new_response

            # Handle function calling loop - process parts in order to support interleaved text/tools
            accumulated_text: List[str] = []
            self._trace(f"SESSION_PARTS_PROCESSING parts_count={len(response.parts)}")

            # Nudge if initial response indicates tool use but has no function calls
            response = self._nudge_for_tool_use(
                response, use_streaming, on_output, wrapped_usage_callback,
                turn_data, context="after initial message",
            )

            while any(p.function_call for p in response.parts if p.function_call):
                # Check for cancellation before processing tools
                if self._is_cancelled():
                    cancel_msg = "[Cancelled during tool execution]"
                    if on_output and not cancellation_notified:
                        self._trace(f"CANCEL_NOTIFY: {cancel_msg} (before processing tools)")
                        on_output("system", cancel_msg, "write")
                    all_text = ''.join(accumulated_text) if accumulated_text else ''
                    self._notify_model_of_cancellation(cancel_msg, all_text)
                    return TurnResult.cancelled(all_text, context="before processing tools").text

                # Process parts in order - emit text, collect function calls into groups
                # When text appears between function calls, execute the preceding group first.
                # Snapshot parts before iteration: a mid-turn interrupt may switch `response`
                # mid-loop, and we need the original list to find orphaned tool calls.
                current_fc_group: List[FunctionCall] = []
                parts_snapshot = list(response.parts)
                interrupted_at_idx: Optional[int] = None
                for idx, part in enumerate(parts_snapshot):
                    text_info = "empty" if part.text == "" else bool(part.text) if part.text else None
                    fc_info = part.function_call.name if part.function_call else None
                    self._trace(f"SESSION_PART[{idx}] text={text_info} fc={fc_info}")
                    if part.text:
                        # Before emitting text, execute any pending function calls
                        if current_fc_group:
                            new_response, turn_result, was_interrupted = self._execute_tools_and_continue(
                                current_fc_group, use_streaming, on_output,
                                wrapped_usage_callback, turn_data, cancellation_notified,
                                accumulated_text, context="interleaved tools",
                            )
                            if turn_result is not None:
                                return turn_result.text
                            response = new_response
                            current_fc_group = []
                            if was_interrupted:
                                interrupted_at_idx = idx
                                break  # Stop iterating stale parts from old response

                        # Emit text (only in non-streaming mode)
                        if not use_streaming:
                            if on_output:
                                on_output("model", part.text, "write")
                            self._forward_to_parent("MODEL_OUTPUT", part.text)
                            if self._runtime.reliability_plugin:
                                self._runtime.reliability_plugin.on_model_text(part.text)
                        accumulated_text.append(part.text)

                    elif part.function_call:
                        current_fc_group.append(part.function_call)

                # If a mid-turn interrupt fired during iteration, inject synthetic
                # cancelled results for any tool calls in parts we never executed.
                if interrupted_at_idx is not None:
                    orphaned_fcs = [
                        p.function_call
                        for p in parts_snapshot[interrupted_at_idx + 1:]
                        if p.function_call
                    ]
                    self._inject_synthetic_cancelled_results(orphaned_fcs)
                    current_fc_group = []  # Already cleared above, defensive

                # Execute remaining function calls at end of parts
                if current_fc_group:
                    new_response, turn_result, _was_interrupted = self._execute_tools_and_continue(
                        current_fc_group, use_streaming, on_output,
                        wrapped_usage_callback, turn_data, cancellation_notified,
                        accumulated_text, context="end of parts",
                    )
                    if turn_result is not None:
                        return turn_result.text
                    response = new_response

            # Collect any remaining text from the final response (no more tool calls)
            self._emit_text_parts(response, use_streaming, on_output, accumulated_text)

            # Final check for mid-turn prompts before completing the turn
            # This handles prompts that arrived while the model was generating its final response
            self._trace(f"FINAL_MID_TURN_CHECK: Starting drain loop, queue_size={len(self._message_queue)}")
            while True:
                mid_turn_response = self._check_and_handle_mid_turn_prompt(
                    use_streaming, on_output, wrapped_usage_callback, turn_data
                )
                if not mid_turn_response:
                    break

                # Check cancellation on the drain-loop response
                cr = self._handle_cancellation(
                    mid_turn_response, use_streaming, on_output,
                    wrapped_usage_callback, turn_data, cancellation_notified,
                    accumulated_text, context="drain loop",
                )
                if cr.action == "end_turn":
                    return cr.turn_result.text
                if cr.action == "switch_response":
                    mid_turn_response = cr.new_response

                # Emit text from the mid-turn response
                self._emit_text_parts(mid_turn_response, use_streaming, on_output, accumulated_text)

                # If the mid-turn response triggered function calls, execute them
                mid_turn_fc = [p.function_call for p in mid_turn_response.parts if p.function_call]
                if mid_turn_fc:
                    new_response, turn_result, _was_interrupted = self._execute_tools_and_continue(
                        mid_turn_fc, use_streaming, on_output,
                        wrapped_usage_callback, turn_data, cancellation_notified,
                        accumulated_text, context="drain loop tools",
                        check_mid_turn=False,
                    )
                    if turn_result is not None:
                        return turn_result.text
                    response = new_response

                    # Process any further tool calls from the continuation
                    while any(p.function_call for p in response.parts if p.function_call):
                        fc_group = [p.function_call for p in response.parts if p.function_call]
                        self._emit_text_parts(response, use_streaming, on_output, accumulated_text)
                        new_response, turn_result, _was_interrupted = self._execute_tools_and_continue(
                            fc_group, use_streaming, on_output,
                            wrapped_usage_callback, turn_data, cancellation_notified,
                            accumulated_text, context="drain loop nested tools",
                            check_mid_turn=False,
                        )
                        if turn_result is not None:
                            return turn_result.text
                        response = new_response

                    # Collect final text from the last response in the chain
                    self._emit_text_parts(response, use_streaming, on_output, accumulated_text)

                # Continue to check for more queued prompts
                continue

            # Safety check: process any prompts that might have been added during the final iteration
            # This handles the race condition where prompts arrive just as the drain loop exits
            final_queue_size = len(self._message_queue)
            if final_queue_size > 0:
                self._trace(f"FINAL_MID_TURN_CHECK: Queue not empty after drain loop! size={final_queue_size}, processing remaining")
                safety_iterations = 0
                max_safety_iterations = 10  # Prevent livelock
                while safety_iterations < max_safety_iterations:
                    safety_iterations += 1
                    remaining_response = self._check_and_handle_mid_turn_prompt(
                        use_streaming, on_output, wrapped_usage_callback, turn_data
                    )
                    if not remaining_response:
                        break
                    self._emit_text_parts(remaining_response, use_streaming, on_output, accumulated_text)
                    # Note: We don't process function calls here to avoid complexity
                    if any(p.function_call for p in remaining_response.parts):
                        self._trace("FINAL_MID_TURN_CHECK: Safety loop response had function calls (not processed)")

            # Check for active streaming tools before completing
            # If there are active streams, wait for updates and continue the loop
            streaming_continuation_attempts = 0
            max_streaming_continuations = 20  # Prevent infinite loops
            while self._has_active_streams() and streaming_continuation_attempts < max_streaming_continuations:
                streaming_continuation_attempts += 1
                self._trace(f"STREAMING_CONTINUATION: Active streams detected, waiting for updates (attempt {streaming_continuation_attempts})")

                # Check for cancellation
                if self._is_cancelled():
                    self._trace("STREAMING_CONTINUATION: Cancelled, exiting streaming loop")
                    break

                # Wait for streaming updates
                updates = self._wait_for_streaming_updates()

                if not updates:
                    self._trace("STREAMING_CONTINUATION: No updates received, timeout")
                    break

                # Format and inject streaming updates
                update_message = self._format_streaming_updates(updates)
                self._trace(f"STREAMING_CONTINUATION: Injecting {len(updates)} updates")

                # Notify UI about streaming updates
                if on_output:
                    on_output("streaming", f"Streaming updates received ({len(updates)} streams)", "write")

                # Inject the update message into the conversation
                self._message_queue.put(update_message, "streaming", SourceType.SYSTEM)

                # Process the injected message to let model react
                streaming_response = self._check_and_handle_mid_turn_prompt(
                    use_streaming, on_output, wrapped_usage_callback, turn_data
                )

                if streaming_response:
                    # Collect text from the streaming response
                    self._emit_text_parts(streaming_response, use_streaming, on_output, accumulated_text)

                    # Process any tool calls from the streaming response
                    streaming_fc = [p.function_call for p in streaming_response.parts if p.function_call]
                    if streaming_fc:
                        self._trace(f"STREAMING_CONTINUATION: Model called {len(streaming_fc)} tools")
                        new_response, turn_result, _was_interrupted = self._execute_tools_and_continue(
                            streaming_fc, use_streaming, on_output,
                            wrapped_usage_callback, turn_data, cancellation_notified,
                            accumulated_text, context="streaming continuation",
                            check_mid_turn=False,
                        )
                        if turn_result is not None:
                            return turn_result.text
                        response = new_response
                        self._emit_text_parts(response, use_streaming, on_output, accumulated_text)
                else:
                    self._trace("STREAMING_CONTINUATION: No response to streaming updates")

            if streaming_continuation_attempts >= max_streaming_continuations:
                self._trace("STREAMING_CONTINUATION: Max attempts reached, completing")

            # Forward completion to parent
            final_response = ''.join(accumulated_text) if accumulated_text else ''
            self._forward_to_parent("COMPLETED", final_response)
            # Note: Do NOT set terminal_event_sent here - COMPLETED is a normal completion
            # and should be followed by IDLE to signal the subagent is ready for more work

            return final_response

        except CancelledException:
            # Handle explicit cancellation exception
            # Note: Don't send on_output here - the explicit checks above already do
            reason = self._cancel_token.cancel_reason if self._cancel_token else ""
            reason_suffix = f" ({reason})" if reason else ""
            cancel_msg = f"Generation cancelled{reason_suffix}"
            self._forward_to_parent("CANCELLED", cancel_msg)
            terminal_event_sent = True
            return f"[{cancel_msg}]"

        except Exception as exc:
            # Route provider errors through output callback before re-raising
            # This ensures errors appear in the UI (queue channel) instead of raw console
            exc_name = type(exc).__name__
            exc_module = type(exc).__module__

            # Check if this is a known provider error (from model_provider plugins)
            is_provider_error = 'model_provider' in exc_module or exc_name in (
                # Anthropic errors
                'AnthropicProviderError', 'APIKeyNotFoundError', 'APIKeyInvalidError',
                'RateLimitError', 'ContextLimitError', 'ModelNotFoundError',
                'OverloadedError', 'UsageLimitError',
                # GitHub Models errors
                'GitHubModelsError', 'TokenNotFoundError', 'TokenInvalidError',
                'TokenPermissionError', 'ModelsDisabledError',
                # Google GenAI errors
                'JaatoAuthError', 'CredentialsNotFoundError', 'CredentialsInvalidError',
                'CredentialsPermissionError', 'ProjectConfigurationError',
            )

            if is_provider_error and on_output:
                # Format error message nicely for the UI
                error_msg = f"[Error] {exc_name}: {str(exc)}"
                on_output("error", error_msg, "write")
                self._trace(f"PROVIDER_ERROR routed to callback: {exc_name}")

            # Forward error to parent for visibility
            self._forward_to_parent("ERROR", f"{exc_name}: {str(exc)}")
            terminal_event_sent = True

            # Re-raise so caller can also handle if needed
            raise

        finally:
            # Record turn end time
            turn_end = datetime.now()
            turn_data['end_time'] = turn_end.isoformat()
            turn_data['duration_seconds'] = (turn_end - turn_start).total_seconds()

            if turn_data['total'] > 0:
                self._turn_accounting.append(turn_data)

            # Update instruction budget with conversation tokens
            self._update_conversation_budget()

            # Clean up cancellation state and activity phase
            self._is_running = False
            self._turn_complete.set()
            self._cancel_token = None
            self._set_activity_phase(ActivityPhase.IDLE)

            # Session quiescence hook (server 0.6.27+).  When the agent
            # called ``signal_completion`` during this turn AND the
            # turn has now fully wrapped up, the session is quiescent
            # and safe to terminate.  Notify hooks so the JaatoServer
            # adapter can emit ``SessionTerminatedEvent`` to attached
            # clients — replaces the legacy "subscribe AGENT_COMPLETED
            # + wait 10s for TURN_COMPLETED" heuristic.
            if (
                getattr(self, "_signal_completion_called", False)
                and not getattr(self, "_session_quiescent_emitted", False)
            ):
                self._session_quiescent_emitted = True
                hooks = getattr(self, "_ui_hooks", None) or getattr(
                    self, "_callbacks", None
                )
                if hooks is not None and hasattr(hooks, "on_session_quiescent"):
                    try:
                        hooks.on_session_quiescent(
                            agent_id=self._agent_id,
                            reason="natural",
                        )
                    except Exception as exc:
                        logger.warning(
                            "on_session_quiescent hook raised: %s — "
                            "event emission skipped, session will still "
                            "wind down correctly", exc,
                        )

            # Notify parent that this subagent is now idle
            # IDLE should be sent after COMPLETED (subagent ready for more work/cleanup),
            # but NOT after CANCELLED or ERROR (abnormal termination states).
            # The terminal_event_sent flag is True for CANCELLED/ERROR, False for COMPLETED.
            if not terminal_event_sent:
                self._forward_to_parent("IDLE", f"Subagent {self._agent_id} is now idle and ready for input.")

            # Self-drain: Process any pending child messages now that we're idle
            # Child messages are status updates from subagents that were queued
            # while we were busy. Process them before truly becoming idle.
            self._drain_child_messages(on_output)

    def _execute_function_call_group(
        self,
        function_calls: List[FunctionCall],
        turn_data: Dict[str, Any],
        on_output: Optional[OutputCallback],
        cancellation_notified: bool
    ) -> List[ToolResult]:
        """Execute a group of function calls and return their results.

        When multiple independent function calls are requested, they are executed
        in parallel using a thread pool. This significantly reduces latency when
        the model requests multiple tools in a single turn.

        Parallel execution is enabled by default but can be disabled via the
        JAATO_PARALLEL_TOOLS environment variable (set to 'false' or '0').
        """
        # Set activity phase: we're executing tools
        self._set_activity_phase(ActivityPhase.EXECUTING_TOOL)

        # Track that this turn has tool calls (for turn complexity classification)
        self._turn_had_tool_calls = True

        # Check if parallel execution is enabled.  Per-call override
        # via SendMessageRequest.parallel_tools wins over env; cleared
        # after one consultation so it only affects the current turn.
        override = getattr(self, '_parallel_tools_override', None)
        if override is not None:
            parallel_enabled = bool(override)
            self._parallel_tools_override = None
        else:
            parallel_enabled = os.environ.get(
                'JAATO_PARALLEL_TOOLS', 'true'
            ).lower() not in ('false', '0', 'no')

        # Use parallel execution for multiple calls, sequential for single call
        if parallel_enabled and len(function_calls) > 1:
            return self._execute_function_calls_parallel(
                function_calls, turn_data, on_output
            )
        else:
            return self._execute_function_calls_sequential(
                function_calls, turn_data, on_output
            )

    def _execute_function_calls_sequential(
        self,
        function_calls: List[FunctionCall],
        turn_data: Dict[str, Any],
        on_output: Optional[OutputCallback]
    ) -> List[ToolResult]:
        """Execute function calls sequentially (original behavior)."""
        tool_results: List[ToolResult] = []

        for fc in function_calls:
            # Check for cancellation before each tool (including parent)
            if self._is_cancelled():
                break

            result = self._execute_single_tool(fc, on_output)

            # Record timing in turn_data
            fc_duration = (result.end_time - result.start_time).total_seconds()
            turn_data['function_calls'].append({
                'name': fc.name,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'duration_seconds': fc_duration,
            })

            # Build ToolResult
            tool_result = self._build_tool_result(fc, result.executor_result)
            tool_results.append(tool_result)

        return tool_results

    def _execute_function_calls_parallel(
        self,
        function_calls: List[FunctionCall],
        turn_data: Dict[str, Any],
        on_output: Optional[OutputCallback]
    ) -> List[ToolResult]:
        """Execute function calls in parallel using a thread pool.

        All function calls are started concurrently. Results are collected
        and returned in the original order.
        """
        # Signal UI to flush before starting parallel tools
        if self._ui_hooks and on_output:
            on_output("system", "", "flush")

        # Emit tool start hooks for all tools before execution
        # This allows UI to show all pending tools at once
        for fc in function_calls:
            self._forward_to_parent("TOOL_CALL", f"{fc.name}({json.dumps(fc.args)})")
            if self._ui_hooks:
                self._trace(f"SESSION_TOOL_START name={fc.name} call_id={fc.id}")
                self._ui_hooks.on_tool_call_start(
                    agent_id=self._agent_id,
                    tool_name=fc.name,
                    tool_args=fc.args,
                    call_id=fc.id
                )

        # Execute all tools in parallel
        results: Dict[str, _ToolExecutionResult] = {}
        max_workers = min(len(function_calls), 8)  # Cap at 8 concurrent tools

        # Capture interactive plugin channels from the spawning thread.
        # Thread-local channels (set by configure_for_subagent) are only
        # visible on this thread.  Worker threads in the pool below won't
        # inherit them, so we snapshot them here and pass them explicitly.
        captured_channels = self._capture_interactive_channels()

        # Capture the current OTel context so worker threads can attach it.
        # Without this, tool spans created in the thread pool become orphans
        # because OTel context (stored in contextvars) doesn't propagate
        # automatically to ThreadPoolExecutor workers.
        otel_ctx = self._telemetry.capture_context()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_fc = {
                executor.submit(
                    self._execute_single_tool_for_parallel, fc, captured_channels, otel_ctx
                ): fc
                for fc in function_calls
            }

            # Collect results as they complete
            for future in as_completed(future_to_fc):
                fc = future_to_fc[future]
                try:
                    result = future.result()
                    results[fc.id or fc.name] = result
                except Exception as e:
                    # Handle unexpected errors
                    results[fc.id or fc.name] = _ToolExecutionResult(
                        fc=fc,
                        executor_result=(False, {"error": f"Parallel execution error: {e}"}),
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        success=False,
                        error_message=str(e),
                        plugin_type="unknown"
                    )

                # Emit tool end hook as each completes
                result = results[fc.id or fc.name]
                # Rewind-with-hint budget reset: a successful tool
                # execution ends the "logical operation" that the
                # rewind budget is gating.  Future MAX_TOKENS-truncated
                # calls in this session start fresh instead of being
                # starved by an exhausted counter.  See
                # ``_maybe_rewind`` for the counter's consumer side.
                if result.success and self._rewind_count > 0:
                    self._trace(
                        f"REWIND_BUDGET_RESET on successful {fc.name} "
                        f"(was count={self._rewind_count})"
                    )
                    self._rewind_count = 0
                fc_duration = (result.end_time - result.start_time).total_seconds()
                # Check if tool was auto-backgrounded or has continuation
                fc_auto_bg = False
                fc_continuation_id = None
                fc_show_output = None
                fc_show_popup = None
                if isinstance(result.executor_result, tuple) and len(result.executor_result) == 2:
                    er = result.executor_result[1]
                    if isinstance(er, dict):
                        fc_auto_bg = er.get('auto_backgrounded', False)
                        fc_continuation_id = er.get('continuation_id')
                        fc_show_output = er.get('show_output')
                        fc_show_popup = er.get('show_popup')
                if self._ui_hooks:
                    self._ui_hooks.on_tool_call_end(
                        agent_id=self._agent_id,
                        tool_name=fc.name,
                        success=result.success,
                        duration_seconds=fc_duration,
                        error_message=result.error_message,
                        call_id=fc.id,
                        backgrounded=fc_auto_bg,
                        continuation_id=fc_continuation_id,
                        show_output=fc_show_output,
                        show_popup=fc_show_popup,
                    )

        # Build results in original order
        tool_results: List[ToolResult] = []
        for fc in function_calls:
            result = results.get(fc.id or fc.name)
            if result:
                fc_duration = (result.end_time - result.start_time).total_seconds()
                turn_data['function_calls'].append({
                    'name': fc.name,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'duration_seconds': fc_duration,
                })
                tool_result = self._build_tool_result(fc, result.executor_result)
                tool_results.append(tool_result)

        return tool_results

    def _is_streaming_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a streaming variant."""
        if not self._runtime.registry:
            return False
        return self._runtime.registry.is_streaming_tool(tool_name)

    def _execute_streaming_tool(
        self,
        fc: FunctionCall,
        on_output: Optional[OutputCallback]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute a streaming tool via the StreamManager.

        Args:
            fc: The function call (with -stream suffix).
            on_output: Optional callback for UI updates.

        Returns:
            Tuple of (success, result_dict) where result_dict contains
            stream_id, initial_chunks, and status.
        """
        if not self._stream_manager or not self._runtime.registry:
            return (False, {"error": "Streaming not available"})

        # Get base tool name and streaming plugin
        base_name = self._runtime.registry.get_base_tool_name(fc.name)
        streaming_plugin = self._runtime.registry.get_streaming_plugin(base_name)

        if not streaming_plugin:
            return (False, {"error": f"Tool {base_name} does not support streaming"})

        # Get plugin name for handle
        plugin_name = "unknown"
        plugin = self._runtime.registry.get_plugin_for_tool(base_name)
        if plugin:
            plugin_name = getattr(plugin, 'name', type(plugin).__name__)

        # Create chunk callback for UI - wrapped in hidden tags so only model sees content
        def on_chunk(chunk: StreamChunk) -> None:
            if on_output:
                # Wrap in <hidden> so the hidden_content_filter strips it from user view
                # but the model still receives the streaming results
                on_output("streaming", f"<hidden>[{base_name}] {chunk.content}</hidden>", "append")

        try:
            # Start the streaming execution
            handle = self._stream_manager.start_stream(
                plugin=streaming_plugin,
                plugin_name=plugin_name,
                tool_name=base_name,
                arguments=fc.args,
                call_id=fc.id or "",
                on_ui_chunk=on_chunk,
            )

            # Format initial chunks for model
            initial_content = []
            for chunk in handle.initial_chunks:
                initial_content.append(chunk.content)

            return (True, {
                "stream_id": handle.stream_id,
                "tool_name": base_name,
                "status": handle.status.value,
                "initial_results": initial_content,
                "initial_count": len(handle.initial_chunks),
                "message": (
                    f"Streaming started. Received {len(handle.initial_chunks)} initial results. "
                    f"More results will be automatically provided as they become available. "
                    f"Call dismiss_stream(stream_id='{handle.stream_id}') when you have enough results."
                ),
            })
        except Exception as e:
            return (False, {"error": f"Streaming execution failed: {str(e)}"})

    def _execute_single_tool(
        self,
        fc: FunctionCall,
        on_output: Optional[OutputCallback]
    ) -> _ToolExecutionResult:
        """Execute a single tool call with full UI hooks and telemetry.

        Used for sequential execution where we want tool-by-tool UI updates.
        """
        import threading
        name = fc.name
        args = fc.args

        self._trace(f"_execute_single_tool: name={name}, thread_id={threading.current_thread().ident}")

        # Ensure session is set in thread-local and ContextVar for plugins
        # This handles cases where tool execution might be in a different thread
        # context than where configure() was called
        set_current_session(self)
        if self._runtime.registry:
            for plugin_name in self._runtime.registry.list_exposed():
                plugin = self._runtime.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'set_session'):
                    plugin.set_session(self)

        # Forward tool call to parent for visibility
        self._forward_to_parent("TOOL_CALL", f"{name}({json.dumps(args)})")

        # Emit hook: tool starting
        if self._ui_hooks:
            if on_output:
                self._trace(f"SESSION_OUTPUT_FLUSH before tool {name}")
                on_output("system", "", "flush")
            self._trace(f"SESSION_TOOL_START name={name} call_id={fc.id}")
            self._ui_hooks.on_tool_call_start(
                agent_id=self._agent_id,
                tool_name=name,
                tool_args=args,
                call_id=fc.id
            )

        fc_start = datetime.now()

        # Determine plugin type for telemetry
        plugin_type = "unknown"
        if self._runtime.registry:
            plugin = self._runtime.registry.get_plugin_for_tool(name)
            if plugin:
                plugin_type = getattr(plugin, 'plugin_type', type(plugin).__name__)

        # Wrap tool execution with telemetry span
        with self._telemetry.tool_span(
            tool_name=name,
            call_id=fc.id or "",
            plugin_type=plugin_type,
        ) as tool_span:
            # Check if this is a streaming tool (name ends with -stream)
            # Set tool input
            tool_span.set_attribute("input.value", json.dumps(args) if args else "{}")
            tool_span.set_attribute("input.mime_type", "application/json")

            if self._is_streaming_tool(name):
                # Route to streaming execution
                executor_result = self._execute_streaming_tool(fc, on_output)
            elif self._executor:
                # Set up tool output callback for streaming output during execution
                def tool_output_callback(chunk: str, _call_id=fc.id, _name=name) -> None:
                    if self._ui_hooks and _call_id:
                        self._ui_hooks.on_tool_output(
                            agent_id=self._agent_id,
                            call_id=_call_id,
                            chunk=chunk
                        )
                    self._forward_to_parent("TOOL_OUTPUT", f"[{_name}] {chunk}")
                self._executor.set_tool_output_callback(tool_output_callback)

                # Set up done callback for auto-backgrounded tasks.
                # Fires when the background task eventually completes, triggering
                # a deferred on_tool_call_end to finalize the UI.
                def task_done_callback(
                    task_id: str, success: bool, error: 'Optional[str]',
                    duration: 'Optional[float]',
                    _call_id=fc.id, _name=name
                ) -> None:
                    if self._ui_hooks:
                        self._ui_hooks.on_tool_call_end(
                            agent_id=self._agent_id,
                            tool_name=_name,
                            success=success,
                            duration_seconds=duration or 0.0,
                            error_message=error,
                            call_id=_call_id,
                        )
                self._executor.set_task_done_callback(task_done_callback)

                executor_result = self._executor.execute(name, args, call_id=fc.id, cancel_token=self._cancel_token)

                self._executor.set_tool_output_callback(None)
                self._executor.set_task_done_callback(None)
            else:
                executor_result = (False, {"error": f"No executor registered for {name}"})

            fc_end = datetime.now()

            # Determine success and error message
            fc_success = True
            fc_error_message = None
            fc_auto_backgrounded = False
            fc_continuation_id = None
            fc_show_output = None
            fc_show_popup = None
            if isinstance(executor_result, tuple) and len(executor_result) == 2:
                fc_success = executor_result[0]
                if not fc_success and isinstance(executor_result[1], dict):
                    fc_error_message = executor_result[1].get('error')
                # Check if tool was auto-backgrounded or has continuation
                if isinstance(executor_result[1], dict):
                    fc_auto_backgrounded = executor_result[1].get('auto_backgrounded', False)
                    fc_continuation_id = executor_result[1].get('continuation_id')
                    fc_show_output = executor_result[1].get('show_output')
                    fc_show_popup = executor_result[1].get('show_popup')

            # Record telemetry
            fc_duration = (fc_end - fc_start).total_seconds()
            if fc_error_message:
                tool_span.set_attribute("exception.message", fc_error_message)
                tool_span.set_status_error(fc_error_message)
            else:
                tool_span.set_status_ok()

            # Set tool output
            result_dict_for_output = None
            if isinstance(executor_result, tuple) and len(executor_result) == 2:
                result_dict_for_output = executor_result[1]
            elif isinstance(executor_result, dict):
                result_dict_for_output = executor_result
            if result_dict_for_output is not None:
                tool_span.set_attribute("output.value", json.dumps(result_dict_for_output))
                tool_span.set_attribute("output.mime_type", "application/json")

            # Pack jaato-specific tool metadata
            tool_meta = {"duration_seconds": fc_duration, "success": fc_success}
            if self._is_streaming_tool(name):
                tool_meta["streaming"] = True
            tool_span.set_metadata(tool_meta)

            # Convention-based telemetry enrichment: if the executor result
            # dict contains a '_telemetry' key mapping to a dict of
            # {attr_name: value}, forward them as span attributes.  This
            # lets plugins emit domain-specific telemetry without coupling
            # to the telemetry plugin directly.
            if isinstance(executor_result, tuple) and len(executor_result) == 2:
                result_dict = executor_result[1]
                if isinstance(result_dict, dict):
                    telem = result_dict.get('_telemetry')
                    if isinstance(telem, dict):
                        for attr_key, attr_val in telem.items():
                            tool_span.set_attribute(attr_key, attr_val)
            elif isinstance(executor_result, dict):
                telem = executor_result.get('_telemetry')
                if isinstance(telem, dict):
                    for attr_key, attr_val in telem.items():
                        tool_span.set_attribute(attr_key, attr_val)

        # Emit hook: tool ended
        if self._ui_hooks:
            self._ui_hooks.on_tool_call_end(
                agent_id=self._agent_id,
                tool_name=name,
                success=fc_success,
                duration_seconds=fc_duration,
                error_message=fc_error_message,
                call_id=fc.id,
                backgrounded=fc_auto_backgrounded,
                continuation_id=fc_continuation_id,
                show_output=fc_show_output,
                show_popup=fc_show_popup,
            )

        return _ToolExecutionResult(
            fc=fc,
            executor_result=executor_result,
            start_time=fc_start,
            end_time=fc_end,
            success=fc_success,
            error_message=fc_error_message,
            plugin_type=plugin_type
        )

    def _capture_interactive_channels(self) -> Dict[str, Any]:
        """Snapshot interactive plugin channels from the current thread.

        Both the permission and clarification plugins use
        ``threading.local()`` to isolate per-session channels (see
        ``configure_for_subagent``).  When we spawn a
        ``ThreadPoolExecutor`` for parallel tool execution the worker
        threads don't inherit these thread-local values, so the plugins
        fall back to ``self._channel`` — the main agent's channel.

        For subagents this means permission prompts escape to the user
        instead of being routed through ``ParentBridgedChannel``.

        This method captures the *current* thread's channel references so
        they can be passed to ``_restore_interactive_channels`` on each
        worker thread.

        Returns:
            Dict with ``permission_channel`` and ``clarification_channel``
            keys (values may be ``None`` when no override is active).
        """
        channels: Dict[str, Any] = {
            'permission_channel': None,
            'clarification_channel': None,
        }

        # Permission plugin — lives on the runtime, not in the registry
        perm = self._runtime.permission_plugin if self._runtime else None
        if perm and hasattr(perm, '_get_channel'):
            channels['permission_channel'] = perm._get_channel()

        # Clarification plugin — lives in the registry
        if self._runtime and self._runtime.registry:
            clari = self._runtime.registry.get_plugin('clarification')
            if clari and hasattr(clari, '_get_channel'):
                channels['clarification_channel'] = clari._get_channel()

        return channels

    def _restore_interactive_channels(self, channels: Dict[str, Any]) -> None:
        """Restore captured interactive channels into the current thread.

        Called on each worker thread in the parallel tool pool to ensure
        that permission and clarification requests use the same channel
        that was active on the spawning thread (e.g.
        ``ParentBridgedChannel`` for subagents).

        Args:
            channels: Dict produced by ``_capture_interactive_channels``.
        """
        perm_channel = channels.get('permission_channel')
        clari_channel = channels.get('clarification_channel')

        # Permission plugin
        if perm_channel is not None:
            perm = self._runtime.permission_plugin if self._runtime else None
            if perm and hasattr(perm, '_thread_local'):
                perm._thread_local.channel = perm_channel

        # Clarification plugin
        if clari_channel is not None and self._runtime and self._runtime.registry:
            clari = self._runtime.registry.get_plugin('clarification')
            if clari and hasattr(clari, '_thread_local'):
                clari._thread_local.channel = clari_channel

    def _execute_single_tool_for_parallel(
        self,
        fc: FunctionCall,
        captured_channels: Optional[Dict[str, Any]] = None,
        otel_ctx: Optional[Any] = None,
    ) -> _ToolExecutionResult:
        """Execute a single tool for parallel execution.

        Similar to _execute_single_tool but:
        - Uses thread-local callback (not instance-level)
        - Does not emit start/end hooks (handled by caller)
        - Includes telemetry for this thread
        - Propagates session to worker thread's thread-local storage
        - Restores interactive plugin channels captured from spawning thread
        - Attaches captured OTel context so tool spans parent correctly

        Args:
            fc: The function call to execute.
            captured_channels: Channel references captured from the spawning
                thread by ``_capture_interactive_channels()``.  Restored into
                this worker thread's thread-local storage so that permission
                and clarification requests route through the correct channel
                (e.g. ``ParentBridgedChannel`` for subagents).
            otel_ctx: OTel context captured from the spawning thread via
                ``telemetry.capture_context()``.  Attached here so that
                tool spans become children of the active turn span instead
                of being orphaned.
        """
        name = fc.name
        args = fc.args

        fc_start = datetime.now()

        # Propagate session to this worker thread's ContextVar and thread-local
        # This is critical for plugins (like TODO) that use thread-local to
        # identify the current agent context. Without this, parallel tools
        # would see agent_name=None and fail to find the correct plan.
        set_current_session(self)
        if self._runtime.registry:
            for plugin_name in self._runtime.registry.list_exposed():
                plugin = self._runtime.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'set_session'):
                    plugin.set_session(self)

        # Restore interactive channels that were captured from the spawning
        # thread.  Without this, worker threads fall back to the main
        # agent's default channel, causing subagent permission/clarification
        # requests to surface as user-facing prompts (the input-capture bug).
        if captured_channels:
            self._restore_interactive_channels(captured_channels)

        # Determine plugin type for telemetry
        plugin_type = "unknown"
        if self._runtime.registry:
            plugin = self._runtime.registry.get_plugin_for_tool(name)
            if plugin:
                plugin_type = getattr(plugin, 'plugin_type', type(plugin).__name__)

        # Attach the OTel context captured from the spawning thread so that
        # tool spans created here become children of the active turn span
        # instead of being orphaned.  Without this, ThreadPoolExecutor
        # workers start with an empty OTel context.
        with self._telemetry.attach_context(otel_ctx):
            # Wrap tool execution with telemetry span
            with self._telemetry.tool_span(
                tool_name=name,
                call_id=fc.id or "",
                plugin_type=plugin_type,
            ) as tool_span:
                # Set tool input
                tool_span.set_attribute("input.value", json.dumps(args) if args else "{}")
                tool_span.set_attribute("input.mime_type", "application/json")

                # Check if this is a streaming tool (name ends with -stream)
                if self._is_streaming_tool(name):
                    # Route to streaming execution
                    executor_result = self._execute_streaming_tool(fc, None)
                elif self._executor:
                    # Create callback that captures this tool's call_id
                    def tool_output_callback(chunk: str, _call_id=fc.id, _name=name) -> None:
                        if self._ui_hooks and _call_id:
                            self._ui_hooks.on_tool_output(
                                agent_id=self._agent_id,
                                call_id=_call_id,
                                chunk=chunk
                            )
                        self._forward_to_parent("TOOL_OUTPUT", f"[{_name}] {chunk}")

                    # Set up done callback for auto-backgrounded tasks (parallel path)
                    def task_done_callback(
                        task_id: str, success: bool, error: 'Optional[str]',
                        duration: 'Optional[float]',
                        _call_id=fc.id, _name=name
                    ) -> None:
                        if self._ui_hooks:
                            self._ui_hooks.on_tool_call_end(
                                agent_id=self._agent_id,
                                tool_name=_name,
                                success=success,
                                duration_seconds=duration or 0.0,
                                error_message=error,
                                call_id=_call_id,
                            )
                    self._executor.set_task_done_callback(task_done_callback)

                    # Pass callback and cancel token directly - executor will set them in thread-local
                    executor_result = self._executor.execute(
                        name, args, tool_output_callback=tool_output_callback, call_id=fc.id,
                        cancel_token=self._cancel_token,
                    )

                    self._executor.set_task_done_callback(None)
                else:
                    executor_result = (False, {"error": f"No executor registered for {name}"})

                fc_end = datetime.now()

                # Determine success and error message
                fc_success = True
                fc_error_message = None
                if isinstance(executor_result, tuple) and len(executor_result) == 2:
                    fc_success = executor_result[0]
                    if not fc_success and isinstance(executor_result[1], dict):
                        fc_error_message = executor_result[1].get('error')

                # Record telemetry
                fc_duration = (fc_end - fc_start).total_seconds()
                if fc_error_message:
                    tool_span.set_attribute("exception.message", fc_error_message)
                    tool_span.set_status_error(fc_error_message)
                else:
                    tool_span.set_status_ok()

                # Set tool output
                result_dict_for_output = None
                if isinstance(executor_result, tuple) and len(executor_result) == 2:
                    result_dict_for_output = executor_result[1]
                elif isinstance(executor_result, dict):
                    result_dict_for_output = executor_result
                if result_dict_for_output is not None:
                    tool_span.set_attribute("output.value", json.dumps(result_dict_for_output))
                    tool_span.set_attribute("output.mime_type", "application/json")

                # Pack jaato-specific tool metadata
                tool_meta = {
                    "duration_seconds": fc_duration,
                    "success": fc_success,
                    "parallel": True,
                }
                if self._is_streaming_tool(name):
                    tool_meta["streaming"] = True
                tool_span.set_metadata(tool_meta)

                # Convention-based telemetry enrichment (parallel path)
                if isinstance(executor_result, tuple) and len(executor_result) == 2:
                    result_dict = executor_result[1]
                    if isinstance(result_dict, dict):
                        telem = result_dict.get('_telemetry')
                        if isinstance(telem, dict):
                            for attr_key, attr_val in telem.items():
                                tool_span.set_attribute(attr_key, attr_val)
                elif isinstance(executor_result, dict):
                    telem = executor_result.get('_telemetry')
                    if isinstance(telem, dict):
                        for attr_key, attr_val in telem.items():
                            tool_span.set_attribute(attr_key, attr_val)

        return _ToolExecutionResult(
            fc=fc,
            executor_result=executor_result,
            start_time=fc_start,
            end_time=fc_end,
            success=fc_success,
            error_message=fc_error_message,
            plugin_type=plugin_type
        )

    def _send_tool_results_and_continue(
        self,
        tool_results: List[ToolResult],
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any]
    ) -> ProviderResponse:
        """Send tool results back to the model and get the continuation response."""
        # with_retry is already imported at module level from .retry_utils

        # Inject task completion spur into last tool result
        if tool_results:
            last = tool_results[-1]
            result_text = str(last.result) if last.result is not None else ""
            spurred_result = f"{result_text}\n\n<hidden>{_TASK_COMPLETION_INSTRUCTION}</hidden>"
            tool_results = tool_results[:-1] + [
                ToolResult(
                    call_id=last.call_id,
                    name=last.name,
                    result=spurred_result,
                    is_error=last.is_error,
                    attachments=last.attachments
                )
            ]

        # Check for queued mid-turn prompts to inject between tool executions.
        # This ensures user prompts are processed during tool-calling chains,
        # not just after the model finishes all tool calls.
        # The prompt is appended to the last tool result to maintain the
        # tool_use → tool_result protocol required by providers.
        injected_prompts: List[str] = []
        while True:
            msg = self._message_queue.pop_first_parent_message()
            if msg is None:
                break
            self._trace(
                f"MID_TURN_PROMPT_PIGGYBACK: Injecting prompt from "
                f"{msg.source_type.value}:{msg.source_id}: {msg.text[:100]}..."
            )
            # Notify callback for UI (removes from pending bar)
            if self._on_prompt_injected:
                self._on_prompt_injected(msg.text)
            # Emit the prompt as user output so UI shows it
            if on_output:
                source = "parent" if msg.source_type == SourceType.PARENT else "user"
                on_output(source, msg.text, "write")
            injected_prompts.append(msg.text)

        if injected_prompts and tool_results:
            combined_prompt = "\n\n".join(injected_prompts)
            last = tool_results[-1]
            result_text = str(last.result) if last.result is not None else ""
            tool_results = tool_results[:-1] + [
                ToolResult(
                    call_id=last.call_id,
                    name=last.name,
                    result=(
                        f"{result_text}\n\n"
                        f"<user_message>{combined_prompt}</user_message>\n"
                        f"The user has sent a new message during your tool execution. "
                        f"Please address their input in your next response."
                    ),
                    is_error=last.is_error,
                    attachments=last.attachments
                )
            ]
            self._trace(
                f"MID_TURN_PROMPT_PIGGYBACK: Injected {len(injected_prompts)} prompt(s) "
                f"into last tool result"
            )

        # Proactive rate limiting
        self._pacer.pace()

        # Set activity phase: we're waiting for LLM response again
        self._set_activity_phase(ActivityPhase.WAITING_FOR_LLM)

        try:
            return self._do_send_tool_results(
                tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
            )
        except Exception as e:
            if not is_context_limit_error(e):
                raise

            # Extract token counts from error message
            # Look for patterns like "373112 exceeds the limit of 128000" or
            # "token count of 373112 exceeds ... limit of 128000"
            import re
            error_str = str(e).replace(',', '')

            # Try to find "X exceeds ... limit of Y" pattern
            match = re.search(r'(\d{4,})\s+exceeds.*?limit.*?(\d{4,})', error_str, re.I)
            if match:
                current_tokens = int(match.group(1))
                limit_tokens = int(match.group(2))
            else:
                # Fallback: find all large numbers (>1000) and assume first two are current/limit
                large_numbers = [int(n) for n in re.findall(r'\d+', error_str) if int(n) > 1000]
                current_tokens = large_numbers[0] if len(large_numbers) >= 1 else 0
                limit_tokens = large_numbers[1] if len(large_numbers) >= 2 else 0

            self._trace(
                f"CONTEXT_LIMIT_RECOVERY: {type(e).__name__}: "
                f"current={current_tokens}, limit={limit_tokens}"
            )

            # Step 1: Try GC first to free up space (GC plugin decides if feasible)
            gc_helped = self._try_gc_for_context_recovery(on_output)

            if gc_helped:
                # GC freed some space - retry the original request.
                # The provider already appended these tool results to its history
                # before the failed API call, and they survived GC (preserve_recent_turns).
                # Remove them so _do_send_tool_results can re-append them cleanly.
                self._remove_tool_results_from_history(len(tool_results))
                self._trace("CONTEXT_LIMIT_RECOVERY: GC freed space, retrying original request")
                try:
                    return self._do_send_tool_results(
                        tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                    )
                except Exception as retry_e:
                    if not is_context_limit_error(retry_e):
                        raise
                    # GC helped but still not enough - fall through to truncation
                    self._trace("CONTEXT_LIMIT_RECOVERY: GC helped but still exceeded, proceeding to truncation")

            # Step 2: Truncate tool results to fit within 80% of the model's limit
            # This ensures we have headroom and don't hit the limit again immediately
            truncated_results = self._truncate_results_to_fit(
                tool_results, current_tokens, limit_tokens
            )

            # Check if any result was actually modified
            any_modified = any(
                orig.result != trunc.result
                for orig, trunc in zip(tool_results, truncated_results)
            )
            if not any_modified:
                self._trace("CONTEXT_LIMIT_RECOVERY: No results were truncated — re-raising")
                raise

            # Notify output callback about the recovery action
            if on_output:
                truncated_names = [
                    orig.name
                    for orig, trunc in zip(tool_results, truncated_results)
                    if orig.result != trunc.result
                ]
                if truncated_names:
                    names_str = ", ".join(truncated_names)
                    on_output(
                        "system",
                        f"[Context limit exceeded — truncated tool results for: {names_str}. Retrying.]",
                        "write",
                    )

            self._trace("CONTEXT_LIMIT_RECOVERY: Retrying with truncated results")

            # Remove the original tool results from provider history
            # (they were added before the API call that failed)
            self._remove_tool_results_from_history(len(tool_results))

            # Update instruction budget to reflect the reduced content
            self._sync_budget_after_truncation(tool_results, truncated_results)

            return self._do_send_tool_results(
                truncated_results, use_streaming, on_output, wrapped_usage_callback, turn_data
            )

    def _try_gc_for_context_recovery(
        self,
        on_output: Optional[OutputCallback],
    ) -> bool:
        """Attempt garbage collection to free context space during limit recovery.

        This is called when the model rejects a request due to context limit exceeded.
        The GC plugin decides whether it's feasible to collect anything at this point.

        During context limit recovery from send_tool_results, the provider has already
        rolled back the tool result messages, leaving the trailing MODEL message (with
        function_calls) without matching tool results. This MODEL message must be
        preserved through GC because the caller will retry sending the tool results.
        Without this preservation, ensure_tool_call_integrity() would remove the
        "unpaired" MODEL message, and the retry would fail because the tool results
        would reference tool_call_ids absent from the history.

        Args:
            on_output: Optional callback for UI notifications.

        Returns:
            True if GC freed any space, False otherwise.
        """
        if not self._gc_plugin or not self._gc_config:
            self._trace("CONTEXT_LIMIT_RECOVERY: No GC plugin configured, skipping GC attempt")
            return False

        self._trace("CONTEXT_LIMIT_RECOVERY: Attempting GC before truncation")

        context_usage = self.get_context_usage()
        history = self.get_history()

        # Save trailing MODEL message with pending tool calls before GC.
        # When send_tool_results fails with context limit, the provider rolls back
        # the tool result messages but the MODEL message (with function_calls) remains
        # at the end of history without matching responses. ensure_tool_call_integrity()
        # would remove this as "unpaired", but we need it for the retry.
        trailing_model_msg = None
        if (history and history[-1].role == Role.MODEL
                and history[-1].function_calls):
            trailing_model_msg = history.pop()
            self._trace(
                f"CONTEXT_LIMIT_RECOVERY: Saved trailing MODEL message with "
                f"{len(trailing_model_msg.function_calls)} pending tool call(s) "
                f"before GC"
            )

        new_history, result = self._gc_plugin.collect(
            history,
            context_usage,
            self._gc_config,
            GCTriggerReason.CONTEXT_LIMIT,
            budget=self._instruction_budget,
        )

        if result.success and result.tokens_freed > 0:
            self._trace(
                f"CONTEXT_LIMIT_RECOVERY: GC collected {result.items_collected} items, "
                f"freed {result.tokens_freed} tokens"
            )
            new_history = ensure_tool_call_integrity(
                new_history,
                trace_fn=lambda m: self._trace(f"CONTEXT_LIMIT_RECOVERY: {m}"),
            )

            # Re-append the trailing MODEL message with pending tool calls.
            # This ensures the retry's tool results have a matching assistant message.
            if trailing_model_msg is not None:
                new_history.append(trailing_model_msg)
                self._trace(
                    "CONTEXT_LIMIT_RECOVERY: Re-appended trailing MODEL message "
                    "with pending tool calls after GC"
                )

            self._history.replace(new_history)
            self._gc_history.append(result)

            # Sync budget with GC changes
            self._apply_gc_removal_list(result)
            self._emit_instruction_budget_update()

            if on_output:
                on_output(
                    "system",
                    f"[Context limit exceeded — GC freed {result.tokens_freed:,} tokens. Retrying.]",
                    "write",
                )
            return True
        else:
            self._trace(
                f"CONTEXT_LIMIT_RECOVERY: GC did not free any space "
                f"(items_collected={result.items_collected}, tokens_freed={result.tokens_freed})"
            )
            return False

    def _remove_tool_results_from_history(self, count: int) -> None:
        """Remove the last N tool result messages from session history.

        Called during context limit recovery to remove the original (too-large)
        tool results before retrying with truncated versions.
        """
        # Operate on session's canonical history directly
        messages = self._history.messages_ref
        removed = 0
        while removed < count and messages:
            last_msg = messages[-1]
            # Check if it's a tool result message
            is_tool_result = (
                last_msg.role == Role.TOOL or
                any(p.function_response is not None for p in last_msg.parts)
            )
            if is_tool_result:
                self._history.pop_last()
                removed += 1
                self._trace(f"CONTEXT_LIMIT_RECOVERY: Removed tool result from history ({removed}/{count})")
            else:
                # Hit a non-tool message, stop
                break

    _TRUNCATION_PRESERVE_LINES = 20  # Lines to keep from the start of truncated results
    _TRUNCATION_PRESERVE_CHARS = 2000  # Minimum characters to keep when using char-based truncation
    _TRUNCATION_NOTICE = (
        "\n\n[NOTICE: This tool result was automatically truncated because it caused "
        "the prompt to exceed the model's context window. Only the first {kept} "
        "of {total} are shown above ({removed_tokens} estimated tokens removed). "
        "If you need more content, re-invoke the tool with offset/limit parameters "
        "to read in smaller chunks.]"
    )

    # Target 80% of context limit to leave headroom after truncation
    _TRUNCATION_TARGET_PERCENT = 0.80

    def _truncate_results_to_fit(
        self, tool_results: List[ToolResult], current_tokens: int, limit_tokens: int
    ) -> List[ToolResult]:
        """Truncate tool results to reduce token count, preserving first lines.

        Strategy:
        - Targets 80% of the model's context limit to leave headroom.
        - Targets the largest results first (they are the most likely culprits).
        - Preserves the first N lines of content so the model retains useful context.
        - Appends a notice informing the model about the truncation.
        - Never removes the tool result itself (models expect one response per call).
        - Continues truncating multiple tool results until target is reached.

        Args:
            tool_results: The original tool results.
            current_tokens: Current total tokens as reported by the model error.
            limit_tokens: Maximum allowed tokens as reported by the model error.

        Returns:
            A new list of tool results with large ones truncated.
        """
        # Estimate size of each result
        result_sizes = []
        for i, tr in enumerate(tool_results):
            result_str = str(tr.result) if tr.result is not None else ""
            estimated_tokens = len(result_str) / 4  # ~4 chars per token
            result_sizes.append((i, estimated_tokens, result_str))

        total_result_tokens = sum(size for _, size, _ in result_sizes)

        # Calculate target: reduce to 80% of limit to leave headroom
        # target_removal = how many tokens we need to remove from current
        target_context = int(limit_tokens * self._TRUNCATION_TARGET_PERCENT)
        target_removal = current_tokens - target_context

        self._trace(
            f"CONTEXT_LIMIT_RECOVERY: truncate called with current={current_tokens}, "
            f"limit={limit_tokens}, target_context={target_context} (80%), "
            f"target_removal={target_removal}, total_result_tokens={total_result_tokens}, "
            f"num_results={len(tool_results)}"
        )

        # If we couldn't extract valid token counts, be aggressive: cut 50% of results
        if target_removal <= 0:
            target_removal = int(total_result_tokens * 0.5)
            self._trace(f"CONTEXT_LIMIT_RECOVERY: using aggressive default target_removal={target_removal}")

        # Sort indices by size descending to truncate largest first
        sized_indices = sorted(
            range(len(result_sizes)),
            key=lambda j: result_sizes[j][1],
            reverse=True,
        )

        truncated = list(tool_results)  # shallow copy
        tokens_removed = 0.0
        preserve_lines = self._TRUNCATION_PRESERVE_LINES

        for j in sized_indices:
            if tokens_removed >= target_removal:
                break

            idx, size, result_str = result_sizes[j]
            tr = tool_results[idx]

            # Skip small results (< 200 tokens estimated) — not worth truncating
            if size < 200:
                self._trace(f"CONTEXT_LIMIT_RECOVERY: skipping result {idx} (size={size} < 200)")
                continue

            # Split into lines and try line-based truncation first
            lines = result_str.split('\n')

            # Calculate how much content to keep (in characters)
            # Keep enough to preserve context but remove overflow + safety margin
            chars_to_remove = int(target_removal * 4)  # tokens -> chars
            chars_to_keep = max(2000, len(result_str) - chars_to_remove)  # Keep at least 2000 chars

            if len(lines) > preserve_lines:
                # Line-based truncation: keep first N lines
                kept_lines = lines[:preserve_lines]
                kept_text = '\n'.join(kept_lines)
                truncation_unit = "lines"
                truncation_kept = preserve_lines
                truncation_total = len(lines)
            elif len(result_str) > chars_to_keep:
                # Character-based truncation: content has few lines but is large
                # Keep first chars_to_keep characters
                kept_text = result_str[:chars_to_keep]
                # Try to break at a word boundary
                last_space = kept_text.rfind(' ', max(0, chars_to_keep - 200))
                if last_space > chars_to_keep // 2:
                    kept_text = kept_text[:last_space]
                truncation_unit = "characters"
                truncation_kept = len(kept_text)
                truncation_total = len(result_str)
                self._trace(
                    f"CONTEXT_LIMIT_RECOVERY: using char-based truncation for result {idx} "
                    f"(lines={len(lines)}, chars={len(result_str)} -> {len(kept_text)})"
                )
            else:
                self._trace(
                    f"CONTEXT_LIMIT_RECOVERY: skipping result {idx} "
                    f"(lines={len(lines)}, chars={len(result_str)} — already small enough)"
                )
                continue

            kept_tokens = len(kept_text) / 4
            removed_tokens = size - kept_tokens

            if removed_tokens <= 0:
                continue

            # Build the truncated content with notice
            notice = self._TRUNCATION_NOTICE.format(
                kept=f"{truncation_kept} {truncation_unit}",
                total=f"{truncation_total} {truncation_unit}",
                removed_tokens=f"{int(removed_tokens):,}",
            )
            truncated_content = kept_text + notice

            truncated[idx] = ToolResult(
                call_id=tr.call_id,
                name=tr.name,
                result=truncated_content,
                is_error=tr.is_error,
                attachments=None,  # Drop attachments to reduce size
            )
            tokens_removed += removed_tokens

        return truncated

    def _cap_tool_results(self, tool_results: List[ToolResult]) -> List[ToolResult]:
        """Proactively cap tool results before they enter history.

        Estimates the aggregate token size of all results and, if they
        would push the context beyond 80% of the model's limit, truncates
        the largest results with a hard character cap.

        Uses a direct cap approach (not the removal-based math in
        ``_truncate_results_to_fit()``) because a single oversized result
        can be many times larger than the entire context window — the
        removal formula underflows in that case.

        Args:
            tool_results: The tool results about to be appended to history.

        Returns:
            The original list (unchanged) if results fit, or a new list
            with large results truncated.
        """
        budget = self._instruction_budget
        if not budget or budget.context_limit == 0:
            return tool_results

        # Estimate per-result sizes
        result_sizes = []
        total_result_tokens = 0
        for tr in tool_results:
            result_str = str(tr.result) if tr.result is not None else ""
            tokens = len(result_str) / 4  # ~4 chars per token
            result_sizes.append((tr, result_str, tokens))
            total_result_tokens += tokens

        # Cap: available space to reach 80% of context limit
        target = int(budget.context_limit * self._TRUNCATION_TARGET_PERCENT)
        cap_tokens = max(0, target - budget.total_tokens())

        if total_result_tokens <= cap_tokens:
            self._trace(
                f"PROACTIVE_CAP: result_tokens={int(total_result_tokens)}, "
                f"cap_tokens={int(cap_tokens)}, action=passed"
            )
            return tool_results

        self._trace(
            f"PROACTIVE_CAP: result_tokens={int(total_result_tokens)}, "
            f"cap_tokens={int(cap_tokens)}, action=truncating"
        )

        # Hard cap: each result gets at most cap_tokens (divided equally
        # if multiple, but in practice one result dominates).
        n_results = len(tool_results)
        per_result_cap_tokens = max(
            self._TRUNCATION_PRESERVE_CHARS // 4,
            cap_tokens // max(1, n_results),
        )
        per_result_cap_chars = int(per_result_cap_tokens * 4)

        truncated = []
        for tr, result_str, tokens in result_sizes:
            if tokens <= per_result_cap_tokens:
                truncated.append(tr)
                continue

            # Truncate to hard character cap
            kept_text = result_str[:per_result_cap_chars]

            # Try to break at a word or line boundary
            last_newline = kept_text.rfind('\n', max(0, per_result_cap_chars - 500))
            if last_newline > per_result_cap_chars // 2:
                kept_text = kept_text[:last_newline]
            else:
                last_space = kept_text.rfind(' ', max(0, per_result_cap_chars - 200))
                if last_space > per_result_cap_chars // 2:
                    kept_text = kept_text[:last_space]

            # Determine units for the notice
            original_lines = result_str.count('\n') + 1
            kept_lines = kept_text.count('\n') + 1
            if original_lines > 1:
                unit_kept = f"{kept_lines} lines"
                unit_total = f"{original_lines} lines"
            else:
                unit_kept = f"{len(kept_text):,} characters"
                unit_total = f"{len(result_str):,} characters"

            removed_tokens = int(tokens - len(kept_text) / 4)
            notice = self._TRUNCATION_NOTICE.format(
                kept=unit_kept,
                total=unit_total,
                removed_tokens=f"{removed_tokens:,}",
            )

            self._trace(
                f"PROACTIVE_CAP: truncated result '{tr.name}' from "
                f"{int(tokens)} to {int(len(kept_text)/4)} tokens "
                f"(cap={per_result_cap_tokens})"
            )

            truncated.append(ToolResult(
                call_id=tr.call_id,
                name=tr.name,
                result=kept_text + notice,
                is_error=tr.is_error,
                attachments=None,  # Drop attachments to reduce size
            ))

        return truncated

    def _sync_budget_after_truncation(
        self,
        original_results: List[ToolResult],
        truncated_results: List[ToolResult],
    ) -> None:
        """Update instruction budget to reflect token savings from truncation.

        Adjusts the CONVERSATION source entry by the difference in estimated
        tokens between original and truncated results.
        """
        if not self._instruction_budget:
            return

        original_tokens = sum(
            len(str(tr.result)) / 4 if tr.result is not None else 0
            for tr in original_results
        )
        truncated_tokens = sum(
            len(str(tr.result)) / 4 if tr.result is not None else 0
            for tr in truncated_results
        )
        saved_tokens = int(original_tokens - truncated_tokens)

        if saved_tokens <= 0:
            return

        # Note: We don't adjust the budget here — total_tokens() returns
        # sum(children) when children exist, so adjusting conv_entry.tokens
        # has no effect. The budget rebuilds from actual history at turn-end
        # via _update_conversation_budget().
        self._trace(
            f"CONTEXT_LIMIT_RECOVERY: Truncation saved ~{saved_tokens} tokens "
            f"(budget will sync at turn-end)"
        )

        # Record truncation event in ledger
        if self._runtime.ledger:
            self._runtime.ledger._record('context-limit-truncation', {
                'original_tokens': int(original_tokens),
                'truncated_tokens': int(truncated_tokens),
                'saved_tokens': saved_tokens,
                'results_affected': sum(
                    1 for o, t in zip(original_results, truncated_results)
                    if o.result != t.result
                ),
            })

        self._emit_instruction_budget_update()

    def _do_send_tool_results(
        self,
        tool_results: List[ToolResult],
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any]
    ) -> ProviderResponse:
        """Send tool results to the provider via ``complete()``.

        Appends tool results to session history as a TOOL message, then
        calls ``provider.complete()`` with the full history.
        """
        # Proactive size guard: cap results before they enter history
        tool_results = self._cap_tool_results(tool_results)
        # Append tool results to session history
        tool_result_parts = [Part(function_response=r) for r in tool_results]
        self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

        with self._telemetry.llm_span(
            model=self._model_name or "unknown",
            provider=self._provider.name if self._provider else "unknown",
            streaming=use_streaming,
            attributes=self._build_llm_span_attributes(),
        ) as llm_telemetry:
            self._record_input_messages_telemetry(llm_telemetry)
            if use_streaming:
                # Track first chunk to use "write" for new block, "append" for continuation
                first_chunk_after_tools = [False]  # Use list to allow mutation in closure

                def streaming_callback(chunk: str) -> None:
                    # Check for pending mid-turn prompts during tool result streaming
                    # This mirrors the interrupt detection in the initial streaming callback
                    if self._message_queue.has_parent_messages():
                        self._trace("MID_TURN_INTERRUPT: Detected pending user prompt during tool result streaming")
                        if self._cancel_token:
                            self._cancel_token.cancel(reason="mid_turn_interrupt")

                    if on_output:
                        # First chunk after tool results starts a new block
                        mode = "append" if first_chunk_after_tools[0] else "write"
                        self._trace(f"SESSION_TOOL_RESULT_OUTPUT mode={mode} len={len(chunk)} preview={repr(chunk[:50])}")
                        on_output("model", chunk, mode)
                        first_chunk_after_tools[0] = True

                # Create thinking callback to emit thinking BEFORE text
                def thinking_callback(thinking: str) -> None:
                    if on_output:
                        self._trace(f"SESSION_TOOL_RESULT_THINKING_CALLBACK len={len(thinking)}")
                        on_output("thinking", thinking, "write")

                with self._provider_access():
                    turn_result, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._get_effective_system_instruction(),
                            tools=self._get_tools_for_provider(),
                            on_chunk=streaming_callback,
                            cancel_token=self._cancel_token,
                            on_usage_update=wrapped_usage_callback,
                            on_thinking=thinking_callback,
                        ),
                        context="complete_tool_results_streaming",
                        on_retry=self._on_retry,
                        cancel_token=self._cancel_token,
                        provider=self._provider
                    )
            else:
                with self._provider_access():
                    turn_result, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._get_effective_system_instruction(),
                            tools=self._get_tools_for_provider(),
                        ),
                        context="complete_tool_results",
                        on_retry=self._on_retry,
                        cancel_token=self._cancel_token,
                        provider=self._provider
                    )
            response = self._unwrap_turn_result(turn_result)

            # Record model response in session history
            self._add_model_response_to_history(response)

            # Emit thinking content if present (non-streaming only).
            # For streaming, the provider emits thinking via on_thinking callback
            # before text starts, so we don't need to emit it again here.
            if not use_streaming and on_output and response.thinking:
                self._trace(f"SESSION_TOOL_RESULT_THINKING_OUTPUT len={len(response.thinking)}")
                on_output("thinking", response.thinking, "write")

            self._record_token_usage(response)
            self._accumulate_turn_tokens(response, turn_data)
            # Track model response count for turn complexity
            self._turn_model_response_count += 1
            # Record token usage to telemetry span
            self._record_token_telemetry(llm_telemetry, response)

            # Emit turn progress after tool result handling
            pending_calls = len([p for p in response.parts if p.function_call])
            self._emit_turn_progress(turn_data, pending_tool_calls=pending_calls)

            return response

    def _check_and_handle_mid_turn_prompt(
        self,
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any]
    ) -> Optional[ProviderResponse]:
        """Check for and handle a pending mid-turn prompt.

        This is called at natural pause points during message processing
        (e.g., after tool execution, after receiving model response).

        Mid-turn processing only handles HIGH PRIORITY messages:
        - User input (SourceType.USER)
        - Parent agent guidance (SourceType.PARENT)
        - System messages (SourceType.SYSTEM)
        - Subscribed events (SourceType.EVENT)

        Child messages (SourceType.CHILD) - passive subagent status updates -
        are left in the queue and processed when the agent becomes idle via
        _drain_child_messages().

        If a high-priority prompt is pending, this method:
        1. Emits the prompt as user output
        2. Sends it to the model as a new user message
        3. Returns the model's response

        Args:
            use_streaming: Whether to use streaming for the model call.
            on_output: Callback for output events.
            wrapped_usage_callback: Callback for usage updates.
            turn_data: Current turn's data for token tracking.

        Returns:
            The model's response if a prompt was handled, None otherwise.
        """
        # Only process high-priority messages mid-turn (parent/user/system)
        # Child messages (subagent status updates) wait until we're idle
        msg = self._message_queue.pop_first_parent_message()
        if msg is None:
            self._trace("MID_TURN_PROMPT: No high-priority messages, returning None")
            return None

        prompt = msg.text
        self._trace(
            f"MID_TURN_PROMPT: Handling prompt from {msg.source_type.value}:{msg.source_id}: "
            f"{prompt[:100]}..."
        )

        # Notify that prompt is being injected (for UI to remove from pending bar)
        if self._on_prompt_injected:
            self._on_prompt_injected(prompt)

        # Emit the prompt as user/parent output so UI shows it
        if on_output:
            # Use "parent" source if message came from parent agent,
            # otherwise "user" for user input
            source = "parent" if msg.source_type == SourceType.PARENT else "user"
            self._trace(f"MID_TURN_PROMPT: Emitting with source={source}")
            on_output(source, prompt, "write")

        # Proactive rate limiting
        self._pacer.pace()

        self._trace(f"MID_TURN_PROMPT: About to call provider, cancel_token.is_cancelled={self._cancel_token.is_cancelled if self._cancel_token else 'None'}")

        # Append user message to session history
        self._history.append(Message.from_text(Role.USER, prompt))

        # Send the prompt to the model with telemetry
        with self._telemetry.llm_span(
            model=self._model_name or "unknown",
            provider=self._provider.name if self._provider else "unknown",
            streaming=use_streaming,
            attributes=self._build_llm_span_attributes(),
        ) as llm_telemetry:
            self._record_input_messages_telemetry(llm_telemetry)
            if use_streaming:
                first_chunk_sent = [False]

                def streaming_callback(chunk: str) -> None:
                    if on_output:
                        mode = "append" if first_chunk_sent[0] else "write"
                        self._trace(f"MID_TURN_RESPONSE mode={mode} len={len(chunk)}")
                        on_output("model", chunk, mode)
                        first_chunk_sent[0] = True

                # Create thinking callback to emit thinking BEFORE text
                def thinking_callback(thinking: str) -> None:
                    if on_output:
                        self._trace(f"MID_TURN_THINKING_CALLBACK len={len(thinking)}")
                        on_output("thinking", thinking, "write")

                self._trace("MID_TURN_PROMPT: Calling with_retry for streaming...")
                with self._provider_access():
                    turn_result, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._get_effective_system_instruction(),
                            tools=self._get_tools_for_provider(),
                            on_chunk=streaming_callback,
                            cancel_token=self._cancel_token,
                            on_usage_update=wrapped_usage_callback,
                            on_thinking=thinking_callback,
                        ),
                        context="complete_mid_turn_streaming",
                        on_retry=self._on_retry,
                        cancel_token=self._cancel_token,
                        provider=self._provider
                    )
                response = self._unwrap_turn_result(turn_result)
                self._trace(f"MID_TURN_PROMPT: Provider returned, finish_reason={response.finish_reason if response else 'None'}")
            else:
                with self._provider_access():
                    turn_result, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._get_effective_system_instruction(),
                            tools=self._get_tools_for_provider(),
                        ),
                        context="complete_mid_turn",
                        on_retry=self._on_retry,
                        cancel_token=self._cancel_token,
                        provider=self._provider
                    )
                response = self._unwrap_turn_result(turn_result)

                # Emit thinking content if present
                if on_output and response.thinking:
                    on_output("thinking", response.thinking, "write")

                # Emit response text if not streaming
                if on_output and response.get_text():
                    on_output("model", response.get_text(), "write")

            # Record model response in session history
            self._add_model_response_to_history(response)

            self._record_token_usage(response)
            self._accumulate_turn_tokens(response, turn_data)
            # Record token usage to telemetry span
            self._record_token_telemetry(llm_telemetry, response)

            return response

    def _build_tool_result(
        self,
        fc: FunctionCall,
        executor_result: Any
    ) -> ToolResult:
        """Build a ToolResult from executor output.

        Handles three result shapes from executors:
        1. ``(ok, result_data)`` tuple – explicit success/error flag.
        2. Plain dict – treated as success.
        3. Plain string – passed through **without** wrapping in a dict so
           that provider converters send it as-is (avoiding JSON escaping of
           file content, which breaks subsequent ``updateFile`` calls).
        """
        # Executor returns (ok, result_dict) tuple
        if isinstance(executor_result, tuple) and len(executor_result) == 2:
            ok, result_data = executor_result
        else:
            ok = True
            result_data = executor_result

        # Check for multimodal result
        attachments: Optional[List[Attachment]] = None
        if isinstance(result_data, dict) and result_data.get('_multimodal'):
            attachments = self._extract_multimodal_attachments(result_data)
            result_data = {k: v for k, v in result_data.items()
                          if not k.startswith('_multimodal') and k not in ('image_data',)}

        # String results pass through directly so converters never
        # JSON-encode them (which would escape quotes, backslashes, etc.).
        if isinstance(result_data, str):
            # Run string-level enrichment (template extraction, etc.)
            if ok and self._runtime.registry:
                enrichment = self._runtime.registry.enrich_tool_result(
                    fc.name,
                    result_data,
                    output_callback=self._current_output_callback,
                    terminal_width=self._terminal_width,
                    tool_args=fc.args
                )
                result_data = enrichment.result
                # Check for preselected reference pinning signal
                self._check_and_pin_reference(enrichment.metadata, result_data)

            return ToolResult(
                call_id=fc.id,
                name=fc.name,
                result=result_data,
                is_error=not ok,
                attachments=attachments
            )

        # Build result dict
        if isinstance(result_data, dict):
            result_dict = result_data
        else:
            result_dict = {"result": result_data}

        # Inject advisory comment from permission evaluator (ALLOW_WITH_COMMENT)
        # before stripping internal metadata.  The comment becomes a visible
        # field so the model sees the feedback alongside the tool result.
        perm_meta = result_dict.get('_permission')
        if isinstance(perm_meta, dict) and perm_meta.get('comment'):
            result_dict['_permission_note'] = perm_meta['comment']

        # Strip internal metadata keys (prefixed with '_') before sending
        # to the model.  These carry scaffolding like _permission, _multimodal
        # flags, etc. that are not meaningful to the model.  The
        # _permission_note is intentionally kept (renamed below).
        permission_note = result_dict.pop('_permission_note', None)
        result_dict = {
            k: v for k, v in result_dict.items()
            if not k.startswith('_')
        }
        if permission_note:
            result_dict['permission_note'] = permission_note

        # For error results, extract a clean error string so provider
        # converters don't double-wrap a dict inside {"error": str(dict)}.
        # This ensures the model receives a readable message (e.g.,
        # "Tool not executed. User comment: ...") rather than a repr of
        # internal scaffolding.
        if not ok and 'error' in result_dict:
            error_msg = result_dict['error']
            # If 'error' is the only remaining key, pass the string directly
            # so converters don't JSON-encode a single-key dict.
            if len(result_dict) == 1:
                result_dict = error_msg

        # Run tool result enrichment (e.g., template extraction)
        if ok and self._runtime.registry:
            result_dict = self._enrich_tool_result_dict(
                fc.name, result_dict, tool_args=fc.args
            )

        return ToolResult(
            call_id=fc.id,
            name=fc.name,
            result=result_dict,
            is_error=not ok,
            attachments=attachments
        )

    def _inject_synthetic_cancelled_results(self, fcs: List[FunctionCall]) -> None:
        """Append synthetic cancelled tool results to history for unexecuted tool calls.

        When a mid-turn interrupt fires after executing some tool groups but
        before executing others, the model's response in history contains
        tool_use blocks that have no matching tool_result entry. Providers that
        require a 1:1 tool_use ↔ tool_result correspondence (e.g. Anthropic)
        will reject the next API call without these synthetic entries.

        This method writes ``{"error": "cancelled"}`` results for each orphaned
        function call directly into history, without making an additional
        provider API call.

        Args:
            fcs: Function calls whose tool_use blocks are already in history but
                 whose results were never sent because the turn was interrupted.
        """
        if not fcs:
            return
        tool_results = [
            ToolResult(call_id=fc.id, name=fc.name, result={"error": "cancelled"}, is_error=True)
            for fc in fcs
        ]
        tool_result_parts = [Part(function_response=r) for r in tool_results]
        self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))
        self._trace(
            f"INJECT_SYNTHETIC: {len(fcs)} orphaned tool calls cancelled: "
            f"{[fc.name for fc in fcs]}"
        )

    def _enrich_tool_result_dict(
        self,
        tool_name: str,
        result_dict: Dict[str, Any],
        tool_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run tool result enrichment on tool results.

        Two enrichment modes:
        1. For file-writing tools (writeNewFile, updateFile): Pass the full JSON
           result so enrichers can extract file paths and run diagnostics.
        2. For other tools with large text fields: Enrich individual text fields.

        Also checks enrichment metadata for preselected reference pinning
        signals and delegates to ``_check_and_pin_reference`` when detected.

        Note: Passes the session's output callback to enrich_tool_result() so that
        enrichment notifications are routed to the correct agent panel. This is
        critical for concurrent sessions (e.g., subagents running in parallel with
        the parent) that share the same registry.

        Args:
            tool_name: Name of the tool that produced the result.
            result_dict: The result dictionary to enrich.
            tool_args: Optional tool call arguments for context-aware enrichment.

        Returns:
            Enriched result dictionary.
        """
        enriched_dict = result_dict.copy()

        # Tools declaring the file_writer trait get full-JSON enrichment
        # (LSP diagnostics, artifact tracking, etc.)
        from jaato_sdk.plugins.model_provider.types import TRAIT_FILE_WRITER
        tool_traits = self._runtime.registry.get_tool_traits(tool_name)

        if TRAIT_FILE_WRITER in tool_traits:
            # Pass full result as JSON so LSP can extract file paths
            import json
            result_json = json.dumps(result_dict)
            # Pass session's callback to route notifications to correct agent panel
            enrichment = self._runtime.registry.enrich_tool_result(
                tool_name,
                result_json,
                output_callback=self._current_output_callback,
                terminal_width=self._terminal_width,
                tool_args=tool_args
            )
            if enrichment.result != result_json:
                try:
                    enriched_dict = json.loads(enrichment.result)
                except json.JSONDecodeError:
                    # If enrichment broke JSON, keep original and append as text
                    enriched_dict['_lsp_diagnostics'] = enrichment.result
            self._check_and_pin_reference(enrichment.metadata, result_json)
            self._emit_enrichment_telemetry(enrichment.metadata, 'tool_result')
            return enriched_dict

        # For other tools: enrich large text fields
        text_fields = ('result', 'content', 'stdout', 'output', 'text', 'data')
        min_length = 100

        for field in text_fields:
            if field in enriched_dict:
                value = enriched_dict[field]
                if isinstance(value, str) and len(value) >= min_length:
                    # Pass session's callback to route notifications to correct agent panel
                    enrichment = self._runtime.registry.enrich_tool_result(
                        tool_name,
                        value,
                        output_callback=self._current_output_callback,
                        terminal_width=self._terminal_width,
                        tool_args=tool_args
                    )
                    if enrichment.result != value:
                        enriched_dict[field] = enrichment.result
                    # Check for pinning signal (only need first match)
                    self._check_and_pin_reference(enrichment.metadata, value)
                    self._emit_enrichment_telemetry(enrichment.metadata, 'tool_result')

        return enriched_dict

    def _emit_enrichment_telemetry(
        self,
        metadata: Dict[str, Any],
        enrichment_type: str
    ) -> None:
        """Forward ``_telemetry`` dicts from enrichment metadata as span events.

        Enrichment plugins can include a ``_telemetry`` key in their metadata
        dict.  This method checks each plugin's metadata entry and emits a
        span event on the current turn span when found.

        Args:
            metadata: Combined enrichment metadata keyed by plugin name.
            enrichment_type: ``"prompt"`` or ``"tool_result"`` — used in the
                event name prefix.
        """
        turn_span = self._current_turn_span
        if not turn_span or not metadata:
            return
        for plugin_name, meta in metadata.items():
            if isinstance(meta, dict):
                telem = meta.get('_telemetry')
                if isinstance(telem, dict):
                    turn_span.add_event(
                        f'enrichment.{enrichment_type}.{plugin_name}',
                        telem,
                    )

    def _check_and_pin_reference(
        self,
        enrichment_metadata: Dict[str, Any],
        content: str
    ) -> None:
        """Check enrichment metadata for a preselected reference pinning signal.

        When the references plugin detects that a tool result contains content
        from a preselected reference file, it sets ``pinned_reference`` metadata.
        This method captures that signal, stores the content in
        ``_pinned_references``, and appends it to the system instruction so it
        survives garbage collection.

        The pinned content is tracked in the instruction budget under
        ``SYSTEM.SELECTED_REFERENCES`` with LOCKED GC policy, ensuring it is
        never garbage-collected.  The original tool result in conversation
        history remains EPHEMERAL and can be freely collected.

        Args:
            enrichment_metadata: Combined metadata from all enrichment plugins,
                keyed by plugin name (e.g., ``{"references": {"pinned_reference": {...}}}``).
            content: The tool result content to pin.
        """
        if not enrichment_metadata:
            return

        # Look for pinning signal from the references plugin
        refs_meta = enrichment_metadata.get("references", {})
        pin_info = refs_meta.get("pinned_reference")
        if not pin_info:
            return

        ref_id = pin_info.get("ref_id")
        ref_name = pin_info.get("ref_name", ref_id)

        if not ref_id:
            return

        import time as _time

        # For directory references the model reads multiple files, each
        # triggering a pin signal with the same ref_id.  Accumulate content
        # so the budget reflects the total reference cost.
        if ref_id in self._pinned_references:
            existing = self._pinned_references[ref_id]
            existing.content += f"\n\n{content}"

            # Also append the new file content to the system instruction
            pinned_block = (
                f"\n<!-- pinned_ref_id={ref_id} (continued) -->\n"
                f"{content}"
            )
            self._system_instruction = (self._system_instruction or "") + pinned_block

            self._update_pinned_references_budget()

            self._trace(
                f"PIN_REF: Appended to existing reference '{ref_id}' "
                f"({ref_name}), new_content_len={len(content)}, "
                f"total_len={len(existing.content)}"
            )
            return

        pinned = _PinnedReference(
            ref_id=ref_id,
            ref_name=ref_name,
            content=content,
            pinned_at=_time.time(),
        )
        self._pinned_references[ref_id] = pinned

        # Append to system instruction so the content persists through GC
        pinned_block = (
            f"\n\n## Selected Reference: {ref_name}\n"
            f"<!-- pinned_ref_id={ref_id} -->\n"
            f"{content}"
        )
        self._system_instruction = (self._system_instruction or "") + pinned_block

        # Update instruction budget with the new pinned reference
        self._update_pinned_references_budget()

        self._trace(
            f"PIN_REF: Pinned reference '{ref_id}' ({ref_name}), "
            f"content_len={len(content)}"
        )

    def _update_pinned_references_budget(self) -> None:
        """Update the instruction budget with pinned reference entries.

        Adds or updates SYSTEM.SELECTED_REFERENCES children in the budget
        for each pinned reference, using LOCKED GC policy.  Token counts
        are estimated from content length (accurate counts are obtained
        in the next budget refresh cycle).
        """
        if not self._pinned_references:
            return

        for ref_id, pinned in self._pinned_references.items():
            child_key = f"{SystemChildType.SELECTED_REFERENCES.value}:{ref_id}"
            gc_policy = DEFAULT_SYSTEM_POLICIES[SystemChildType.SELECTED_REFERENCES]

            # Estimate tokens (will be refined in next budget cycle)
            tokens = estimate_tokens(pinned.content)

            # Check if child already exists
            parent = self._instruction_budget.get_entry(InstructionSource.SYSTEM)
            if parent:
                existing = parent.children.get(child_key)
                if existing is not None:
                    existing.tokens = tokens
                else:
                    self._instruction_budget.add_child(
                        InstructionSource.SYSTEM,
                        child_key,
                        tokens,
                        gc_policy,
                        label=f"ref: {pinned.ref_name}",
                    )

        self._emit_instruction_budget_update()

    def _remove_pinned_from_system_instruction(self) -> None:
        """Remove pinned reference blocks from the system instruction.

        Called during a true fresh reset to strip all
        ``## Selected Reference: ...`` blocks that were appended when
        references were pinned.  Each block is delimited by a
        ``<!-- pinned_ref_id=... -->`` comment for reliable matching.
        """
        if not self._system_instruction:
            return
        import re as _re
        # Remove blocks starting with "\n\n## Selected Reference: ..."
        # up to (but not including) the next "\n\n## Selected Reference:" or end.
        self._system_instruction = _re.sub(
            r'\n\n## Selected Reference: [^\n]*\n<!-- pinned_ref_id=[^\n]* -->\n'
            r'(?:(?!\n\n## Selected Reference: )[\s\S])*',
            '',
            self._system_instruction,
        )

    def _extract_multimodal_attachments(
        self,
        result: Dict[str, Any]
    ) -> Optional[List[Attachment]]:
        """Extract multimodal attachments from a result dict."""
        multimodal_type = result.get('_multimodal_type', 'image')

        if multimodal_type == 'image':
            image_data = result.get('image_data')
            if not image_data:
                return None

            mime_type = result.get('mime_type', 'image/png')
            display_name = result.get('display_name', 'image')

            return [Attachment(
                mime_type=mime_type,
                data=image_data,
                display_name=display_name
            )]

        return None

    def _accumulate_turn_tokens(
        self,
        response: ProviderResponse,
        turn_tokens: Dict[str, int]
    ) -> None:
        """Update token counts from provider response.

        Note: We REPLACE (not sum) because each API response's prompt_tokens
        already includes ALL previous history. The final API call in a turn
        has the complete context usage.

        However, we only replace if values are non-zero, to preserve good values
        when streaming is cancelled mid-turn (which may return zero tokens).

        Cache token fields (cache_read, cache_creation) are replaced alongside
        prompt/output/total so the final API call's values propagate to
        turn_accounting and ultimately to TurnCompletedEvent.
        """
        if response.usage.total_tokens > 0:
            turn_tokens['prompt'] = response.usage.prompt_tokens
            turn_tokens['output'] = response.usage.output_tokens
            turn_tokens['total'] = response.usage.total_tokens

        # Cache tokens: replace when present (same semantics as prompt/output)
        if response.usage.cache_read_tokens is not None:
            turn_tokens['cache_read'] = response.usage.cache_read_tokens
        if response.usage.cache_creation_tokens is not None:
            turn_tokens['cache_creation'] = response.usage.cache_creation_tokens

        # Accumulate thinking tokens (these are summed, not replaced)
        if response.usage.thinking_tokens:
            turn_tokens['thinking'] = turn_tokens.get('thinking', 0) + response.usage.thinking_tokens
            self._update_thinking_budget(response.usage.thinking_tokens)

    def _emit_turn_progress(self, turn_data: Dict[str, Any], pending_tool_calls: int) -> None:
        """Emit turn progress event with current token state.

        Called after each model response within a turn to provide real-time
        token tracking before the turn completes.  Includes cache token
        fields when the provider reports them.
        """
        if not self._ui_hooks:
            return

        context_usage = self.get_context_usage()
        self._ui_hooks.on_turn_progress(
            agent_id=self._agent_id,
            total_tokens=turn_data.get('total', 0),
            prompt_tokens=turn_data.get('prompt', 0),
            output_tokens=turn_data.get('output', 0),
            percent_used=context_usage.get('percent_used', 0.0),
            pending_tool_calls=pending_tool_calls,
            cache_read_tokens=turn_data.get('cache_read'),
            cache_creation_tokens=turn_data.get('cache_creation'),
        )

        # Update conversation budget and emit for budget panel
        # This ensures the budget snapshot includes current turn's conversation tokens
        self._update_conversation_budget()

    def _record_token_usage(self, response: ProviderResponse) -> None:
        """Record token usage to ledger if available."""
        if not self._runtime.ledger:
            return

        self._runtime.ledger._record('response', {
            'prompt_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.output_tokens,
            'total_tokens': response.usage.total_tokens,
        })

    def _record_token_telemetry(self, span, response: ProviderResponse) -> None:
        """Record OpenInference token count and response attributes on a telemetry span.

        Sets ``llm.token_count.prompt`` and ``llm.token_count.completion``
        (which auto-computes ``llm.token_count.total`` via _SpanWrapper),
        plus optional cache and reasoning detail attributes.

        Also records ``llm.output_messages.*`` (OpenInference indexed attributes)
        from the model response, and ``gen_ai.response.finish_reasons``.

        Accumulates the counts on the current turn span so the root
        AGENT span carries aggregate token usage for the entire turn.

        Args:
            span: The LLM span context to set attributes on.
            response: Provider response containing usage data and content.
        """
        # Record finish reason (OpenInference convention)
        if response.finish_reason is not None:
            span.set_attribute(
                "gen_ai.response.finish_reasons",
                [response.finish_reason.value],
            )

        # Record output messages (OpenInference indexed attributes)
        output_msgs = self._response_to_openinference(response)
        if output_msgs:
            span.set_output_messages(output_msgs)

        if not response.usage:
            return

        usage = response.usage
        if usage.prompt_tokens is not None:
            span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
        if usage.output_tokens is not None:
            span.set_attribute("llm.token_count.completion", usage.output_tokens)
        if usage.cache_read_tokens is not None:
            span.set_attribute("llm.token_count.prompt_details.cache_read", usage.cache_read_tokens)
        if usage.cache_creation_tokens is not None:
            span.set_attribute("llm.token_count.prompt_details.cache_write", usage.cache_creation_tokens)
        if usage.reasoning_tokens is not None:
            span.set_attribute("llm.token_count.completion_details.reasoning", usage.reasoning_tokens)

        # Cache outcome classification (hit/partial/warm/miss/unknown)
        # so external observers can correlate cache behavior with the
        # GC ↔ cache coordination dance.
        try:
            outcome = self._classify_cache_outcome(
                int(usage.prompt_tokens or 0),
                usage.cache_read_tokens,
                usage.cache_creation_tokens,
            )
            span.set_attribute("cache.outcome", outcome)
        except Exception as e:
            self._trace(f"LLM_TELEMETRY: cache outcome classification failed: {e}")

        # Accumulate on turn span so the root AGENT span shows totals
        turn_span = self._current_turn_span
        if turn_span:
            if usage.prompt_tokens is not None:
                self._turn_prompt_tokens = getattr(self, '_turn_prompt_tokens', 0) + usage.prompt_tokens
                turn_span.set_attribute("llm.token_count.prompt", self._turn_prompt_tokens)
            if usage.output_tokens is not None:
                self._turn_completion_tokens = getattr(self, '_turn_completion_tokens', 0) + usage.output_tokens
                turn_span.set_attribute("llm.token_count.completion", self._turn_completion_tokens)

    def _record_input_messages_telemetry(self, span) -> None:
        """Record OpenInference input messages on a telemetry span.

        Converts the current session history (messages being sent to the
        provider) into OpenInference ``llm.input_messages.*`` indexed
        attributes on the LLM span.

        Args:
            span: The LLM span context to set attributes on.
        """
        input_msgs = self._history_to_openinference()
        if input_msgs:
            span.set_input_messages(input_msgs)

    @staticmethod
    def _response_to_openinference(response: ProviderResponse) -> List[Dict[str, Any]]:
        """Convert a ProviderResponse to OpenInference output message dicts.

        Returns a list with a single assistant message containing text content
        and any tool calls from the response parts.

        Args:
            response: The provider response to convert.

        Returns:
            List of message dicts with 'role', 'content', and optional
            'tool_calls' suitable for ``span.set_output_messages()``.
        """
        text = response.get_text()
        function_calls = response.get_function_calls()

        if not text and not function_calls:
            return []

        msg: Dict[str, Any] = {"role": "assistant", "content": text or ""}
        if function_calls:
            msg["tool_calls"] = [
                {
                    "name": fc.name,
                    "arguments": json.dumps(fc.args) if fc.args else "{}",
                }
                for fc in function_calls
            ]
        return [msg]

    def _history_to_openinference(self) -> List[Dict[str, Any]]:
        """Convert the current session history to OpenInference input message dicts.

        Maps jaato ``Message`` objects to the dict format expected by
        ``span.set_input_messages()``: each dict has 'role' and 'content'.

        Returns:
            List of message dicts suitable for ``span.set_input_messages()``.
        """
        result = []
        for msg in self._history.messages:
            # Map jaato roles to OpenInference roles
            role = msg.role.value  # "user", "model", "tool"
            if role == "model":
                role = "assistant"

            # Extract text content from parts
            texts = [p.text for p in msg.parts if p.text]
            content = "".join(texts) if texts else ""

            entry: Dict[str, Any] = {"role": role, "content": content}

            # Include tool calls from model messages
            if msg.role == Role.MODEL:
                tool_calls = [
                    {
                        "name": p.function_call.name,
                        "arguments": json.dumps(p.function_call.args)
                            if p.function_call.args else "{}",
                    }
                    for p in msg.parts
                    if p.function_call
                ]
                if tool_calls:
                    entry["tool_calls"] = tool_calls

            result.append(entry)
        return result

    def get_history(self) -> List[Message]:
        """Get current conversation history.

        Returns the session's canonical copy of the history.  When an
        inbound history transformer is registered (e.g. a
        pseudonymization consumer), this returns the **transformed**
        view — the form that lives in the canonical container.  Trusted
        callers that need the un-transformed form should use
        :meth:`get_history_raw` instead.

        The session is the sole owner of conversation state; providers
        receive messages as parameters to ``complete()``.
        """
        return self._history.messages

    def get_history_raw(self) -> List[Message]:
        """Get the trusted-caller view of conversation history.

        When a raw-view transformer is registered on the underlying
        :class:`SessionHistory`, this returns the result of running
        each stored Message through that transformer (typically the
        un-pseudonymized form for premium's user-display swap-back
        path).  When no transformer is registered, this returns the
        same data as :meth:`get_history`.

        Trusted callers (user-display renderer, audit logger, the
        narrow set of components that legitimately need raw values)
        should call this accessor explicitly so the trust grant is
        visible at the call site.
        """
        return self._history.messages_raw

    def set_history_inbound_transformer(
        self, fn: Optional[Callable[[Message], Message]]
    ) -> None:
        """Register an inbound transformer on the session's history.

        Plug-in surface for pseudonymization / redaction / audit /
        content-filter consumers that need to transform every Message
        before it lands in the canonical container.  See
        :meth:`SessionHistory.set_inbound_transformer` for semantics.

        Premium typically calls this from a session hook
        (:meth:`SessionManager.add_session_hook`) so the transformer
        is wired before any user message arrives.
        """
        self._history.set_inbound_transformer(fn)

    def set_history_raw_view_transformer(
        self, fn: Optional[Callable[[Message], Message]]
    ) -> None:
        """Register a raw-view transformer on the session's history.

        Plug-in surface paired with :meth:`set_history_inbound_transformer`
        — when the inbound transformer redacts, the raw-view transformer
        un-redacts for trusted callers via :meth:`get_history_raw`.
        See :meth:`SessionHistory.set_raw_view_transformer` for
        semantics.
        """
        self._history.set_raw_view_transformer(fn)

    # ─── Session-attached state ──────────────────────────────────────
    #
    # Generic facility for extensions to attach opaque state to a
    # session.  The framework owns the storage container, the
    # persistence path (via the session journal), and the fork-carry
    # plumbing (via the ``initial_session_state`` kwarg on
    # ``SessionManager.create_*_session`` and the
    # ``session_state_snapshot`` field on ``Waypoint``).  Encryption,
    # schema validation, and cross-session sharing stay out of scope
    # — extensions handle those.

    def set_session_state(self, key: str, value: Any) -> None:
        """Attach opaque state under a string key.

        Right shape for **static or rarely-mutated state** (audit chain
        head, version markers, telemetry counters the consumer updates
        explicitly).  For **incrementally-mutated state** (e.g. a
        lookup table that grows turn-by-turn) prefer
        :meth:`register_session_state_provider` — pushing on every
        mutation is brittle.

        ``value`` must be JSON-serialisable; non-serialisable values
        raise ``TypeError`` at attach time so the failure surfaces at
        the call site rather than at journal-save.

        Setting a key that has a registered provider does not unregister
        the provider — providers always win for both reads and
        snapshot-for-persistence.  The pushed value is retained as a
        fallback that surfaces only if the provider is later
        unregistered.
        """
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"set_session_state({key!r}): value must be JSON-serialisable "
                f"(the framework persists session_state as JSON; encrypt before "
                f"attach if confidentiality is needed). Underlying error: {exc}"
            ) from exc
        self._session_state[key] = value

    def register_session_state_provider(
        self, key: str, fn: Callable[[], Any]
    ) -> None:
        """Register a callback returning the current value for ``key``.

        Plug-in surface for **incrementally-mutated state** (e.g. the
        pseudonymization lookup table that grows whenever a new
        sensitive value is encountered).  ``fn`` is invoked by the
        framework at journal-save, waypoint-snapshot, and fork-snapshot
        time (i.e. inside :meth:`get_all_session_state` /
        :meth:`get_session_state` for this key) and must return a
        JSON-serialisable value.  Encryption (if needed) happens inside
        ``fn`` — the framework treats the return value as opaque JSON.

        At most one provider per key — registering a second time
        replaces the prior registration.  A registered provider takes
        precedence over any value previously set via
        :meth:`set_session_state` for the same key (for both
        :meth:`get_session_state` reads and snapshot-for-persistence).
        """
        if not callable(fn):
            raise TypeError(
                f"register_session_state_provider({key!r}): fn must be callable"
            )
        self._state_providers[key] = fn

    def get_session_state(self, key: str, default: Any = None) -> Any:
        """Read the current value for ``key``, or ``default`` if absent.

        If a provider is registered for ``key``, returns the provider's
        current value (so the read reflects live state, not whatever
        was last pushed via :meth:`set_session_state`).  Otherwise
        returns the value last set via :meth:`set_session_state`, or
        ``default`` if the key has neither.
        """
        provider = self._state_providers.get(key)
        if provider is not None:
            return provider()
        if key in self._session_state:
            return self._session_state[key]
        return default

    def get_all_session_state(self) -> Dict[str, Any]:
        """Snapshot of all currently-attached state.

        Invokes every registered provider once (so the snapshot
        reflects live values at call time, not whatever was last
        pushed) and merges with set-state values; provider values win
        on key collision.  Returns a copy — mutation of the returned
        dict doesn't propagate back into the session.

        This is the right call for fork primitives and the journal
        save path: it materialises the current state into a plain
        dict that can be carried across to a new session via
        ``initial_session_state`` or persisted on disk.
        """
        snapshot: Dict[str, Any] = dict(self._session_state)
        for key, fn in self._state_providers.items():
            snapshot[key] = fn()
        return snapshot

    def get_turn_accounting(self) -> List[Dict[str, Any]]:
        """Get token usage and timing per turn."""
        return list(self._turn_accounting)

    def restore_turn_accounting(
        self, turns: List[Dict[str, Any]],
    ) -> None:
        """Replace the per-turn token-usage / timing list.

        Pre-§7c-step-6.6.1.0 the daemon's persistence-restore
        path (``server/session_manager.py:2558``) reached into
        the private ``self._turn_accounting`` attribute directly:

            jaato_session._turn_accounting = list(state.turn_accounting)

        That violated the same encapsulation discipline §7c step
        3a (set_agent_identity) + step 3b (get_tool_schemas)
        established.  This public method replaces the
        private-attr write with a stable surface that the
        upcoming ``session.restore_turn_accounting`` runner-RPC
        (§7c step 6.6.1.2) can wrap.

        Args:
            turns: List of per-turn dicts from a
                :class:`SessionState` snapshot.  Caller owns the
                list; a shallow copy is taken to isolate
                in-session state from caller mutation.
        """
        self._turn_accounting = list(turns)

    def restore_conversation_budget(
        self, snapshot: Dict[str, Any],
    ) -> None:
        """Restore the CONVERSATION budget entry from a saved snapshot.

        Pre-§7c-step-6.6.1.0 the daemon's persistence-restore
        path (``server/session_manager.py:2592-2593``) reached
        through the session into the underlying
        :class:`InstructionBudget`:

            jaato_session.instruction_budget.restore_conversation_from_snapshot(
                state.budget_state)

        The :meth:`InstructionBudget.restore_conversation_from_snapshot`
        method exists, but JaatoSession had no public wrapper.
        This method exposes the operation as a stable
        JaatoSession-level surface for the upcoming
        ``session.restore_conversation_budget`` runner-RPC
        (§7c step 6.6.1.3).

        No-op when ``self._instruction_budget`` is None
        (pre-:meth:`configure`); matches the daemon caller's
        existing ``if jaato_session.instruction_budget:`` guard.

        Args:
            snapshot: Conversation-source snapshot dict from a
                :class:`SessionState`'s ``budget_state``.  Format
                is opaque here; the underlying InstructionBudget
                method validates + reconstructs the entry tree.
        """
        if self._instruction_budget is None:
            return
        self._instruction_budget.restore_conversation_from_snapshot(snapshot)

    def set_parallel_tools_override(self, enabled: bool) -> None:
        """Stash a per-turn override for parallel-tool execution.

        Pre-§7c-step-6.6.3.0 the daemon's SDK request handler
        (``server/session_manager.py:4096``) reached into the
        private attribute directly:

            jaato_session._parallel_tools_override = event.parallel_tools

        That violated the same encapsulation discipline §7c
        step 3a / 3b / 6.1 (1/3) / 6.6.1.0 / 6.6.3.0 / 6.6.3.1 /
        6.6.3.2 established.  This public method replaces the
        private-attr write with a stable surface that the
        upcoming ``session.set_parallel_tools_override``
        runner-RPC (§7c step 6.6.3.3) can wrap.

        Semantic: the override wins over ``JAATO_PARALLEL_TOOLS``
        env-var consultation for the current turn ONLY.  The
        session's tool-execution branch reads the override at
        line 4886-4889 and clears it after one read — i.e. each
        ``set_parallel_tools_override(True)`` call affects
        exactly the next turn that consults the override.

        Args:
            enabled: True to force parallel-tool execution for the
                next turn; False to disable.  Caller passes the
                raw bool from the SDK request; daemon's existing
                ``if event.parallel_tools is not None:`` guard
                prevents passing None (no-override) through this
                method.
        """
        self._parallel_tools_override = bool(enabled)

    def snapshot_conversation_budget(self) -> Optional[Dict[str, Any]]:
        """Return a serializable snapshot of the CONVERSATION budget
        entry for persistence.

        Inverse of :meth:`restore_conversation_budget` (added in
        §7c step 6.6.1.0).  Pre-§7c-step-6.6.3.0 the daemon's
        persistence-save path
        (``server/session_manager.py:2986``) reached through the
        session into the underlying :class:`InstructionBudget`:

            jaato_session.instruction_budget.get_conversation_snapshot()

        The :meth:`InstructionBudget.get_conversation_snapshot`
        method exists (instruction_budget.py:390), but
        JaatoSession had no public wrapper.  This method exposes
        the operation as a stable JaatoSession-level surface for
        the upcoming ``session.snapshot_conversation_budget``
        runner-RPC (§7c step 6.6.3.2).

        Returns ``None`` when ``self._instruction_budget`` is
        None (pre-:meth:`configure`); matches the daemon caller's
        existing ``if jaato_session.instruction_budget:`` guard
        semantic.

        Returns:
            Conversation-source snapshot dict (JSON-native), or
            ``None`` when budget unavailable / no conversation
            entry exists.
        """
        if self._instruction_budget is None:
            return None
        return self._instruction_budget.get_conversation_snapshot()

    def append_history_message(self, message: Message) -> None:
        """Append a single message to the session's history.

        Pre-§7c-step-6.6.3.0 the daemon's interrupted-tool-call
        recovery path (``server/session_manager.py:2855``) did
        the get-modify-reset dance manually:

            current_history = jaato_session.get_history()
            current_history.append(synthetic_message)
            jaato_session.reset_session(current_history)

        That worked but was awkward — three calls for one
        operation, and `reset_session` clears
        ``_turn_accounting`` as a side effect (which the
        recovery path actually wants, since the interrupted
        turn's accounting is mid-flight).  This method
        preserves the existing semantic exactly: appends the
        message + clears turn_accounting (via the underlying
        ``reset_session`` call).

        Phase 3 §7c step 6.6.3.0 (encapsulation cleanup,
        prerequisite for §7c step 6.6.3.1's
        ``session.append_history_message`` runner-RPC).

        Args:
            message: A :class:`Message` instance to append.
        """
        current_history = self.get_history()
        current_history.append(message)
        self.reset_session(current_history)

    def get_context_limit(self) -> int:
        """Get the context window limit for the current model."""
        if not self._provider:
            return 1_048_576
        return self._provider.get_context_limit()

    def get_context_usage(self) -> Dict[str, Any]:
        """Get context window usage statistics.

        Uses InstructionBudget as the single source of truth for token accounting.
        This includes system instructions, plugin schemas, enrichment, and conversation
        tokens - providing accurate context usage from startup through all turns.
        """
        # Use InstructionBudget as the single source of truth
        if self._instruction_budget:
            total_tokens = self._instruction_budget.total_tokens()
            context_limit = self._instruction_budget.context_limit
            percent_used = self._instruction_budget.utilization_percent()
            tokens_remaining = self._instruction_budget.available_tokens()
        else:
            # Fallback if budget not initialized
            total_tokens = 0
            context_limit = self.get_context_limit()
            percent_used = 0.0
            tokens_remaining = context_limit

        # Get turn count from turn_accounting for backward compatibility
        turn_accounting = self.get_turn_accounting()

        return {
            'model': self._model_name or 'unknown',
            'context_limit': context_limit,
            'total_tokens': total_tokens,
            'prompt_tokens': total_tokens,  # InstructionBudget tracks total, not split
            'output_tokens': 0,  # Output tokens are included in conversation total
            'turns': len(turn_accounting),
            'percent_used': percent_used,
            'tokens_remaining': tokens_remaining,
        }

    def reset_session(self, history: Optional[List[Message]] = None) -> None:
        """Reset the chat session, clearing turn accounting and optionally restoring history.

        When history is provided (e.g. after GC), the token count cache is
        preserved because restored Message objects keep their original
        message_id, so cached counts remain valid.  The cache is only
        cleared on a true fresh reset (no history).

        Args:
            history: Optional initial history for the new session.
        """
        if history:
            logger.info(f"[session:{self._agent_id}] reset_session: restoring {len(history)} messages")
            self._history.replace(history)
        else:
            logger.info(f"[session:{self._agent_id}] reset_session: starting fresh (no history)")
            self._history.clear()
        self._turn_accounting = []
        if not history:
            self._msg_token_cache.clear()
            # On true fresh reset, clear pinned references and remove their
            # content from the system instruction.  GC resets (history provided)
            # preserve pinned references — they stay in the system instruction.
            if self._pinned_references:
                self._remove_pinned_from_system_instruction()
                self._pinned_references.clear()
            # Notify enrichment plugins (memory, references, template, ...)
            # so they can clear per-session dedup tracking.  Otherwise hints
            # that were surfaced in the wiped conversation would never be
            # re-emitted in the fresh one.
            if self._runtime and self._runtime.registry:
                try:
                    self._runtime.registry.broadcast_history_cleared()
                except Exception as exc:
                    logger.debug(
                        f"[session:{self._agent_id}] "
                        f"broadcast_history_cleared failed: {exc}"
                    )

    def get_turn_boundaries(self) -> List[int]:
        """Get indices where each turn starts in the history."""
        history = self.get_history()
        boundaries = []

        for i, msg in enumerate(history):
            if msg.role == Role.USER and msg.parts and msg.parts[0].text:
                boundaries.append(i)

        return boundaries

    def revert_to_turn(self, turn_id: int) -> Dict[str, Any]:
        """Revert the conversation to a specific turn."""
        boundaries = self.get_turn_boundaries()
        total_turns = len(boundaries)

        if turn_id < 1:
            raise ValueError(f"Turn ID must be >= 1, got {turn_id}")

        if turn_id > total_turns:
            raise ValueError(f"Turn {turn_id} does not exist. Current session has {total_turns} turn(s).")

        if turn_id == total_turns:
            return {
                'success': True,
                'turns_removed': 0,
                'new_turn_count': total_turns,
                'message': f"Already at turn {turn_id}, no changes made."
            }

        history = self.get_history()

        if turn_id < total_turns:
            truncate_at = boundaries[turn_id]
        else:
            truncate_at = len(history)

        truncated_history = list(history[:truncate_at])
        turns_removed = total_turns - turn_id

        if turn_id <= len(self._turn_accounting):
            self._turn_accounting = self._turn_accounting[:turn_id]

        self._history.replace(truncated_history)

        if self._session_plugin and hasattr(self._session_plugin, 'set_turn_count'):
            self._session_plugin.set_turn_count(turn_id)

        return {
            'success': True,
            'turns_removed': turns_removed,
            'new_turn_count': turn_id,
            'message': f"Reverted to turn {turn_id} (removed {turns_removed} turn(s))."
        }

    def get_user_commands(self) -> Dict[str, UserCommand]:
        """Get available user commands."""
        return dict(self._user_commands)

    def execute_user_command(
        self,
        command_name: str,
        args: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, bool]:
        """Execute a user command."""
        if command_name not in self._user_commands:
            raise ValueError(f"Unknown user command: {command_name}")

        if not self._executor:
            raise RuntimeError("Executor not configured.")

        cmd = self._user_commands[command_name]
        args = args or {}

        _ok, result = self._executor.execute(command_name, args)

        if cmd.share_with_model and self._provider:
            self._inject_command_into_history(command_name, args, result)

        return result, cmd.share_with_model

    def _inject_command_into_history(
        self,
        command_name: str,
        args: Dict[str, Any],
        result: Any
    ) -> None:
        """Inject a user command execution into conversation history."""
        # HelpLines is display-only (rendered via pager, not serializable) — skip
        if isinstance(result, HelpLines):
            return

        user_message = Message(
            role=Role.USER,
            parts=[Part.from_text(f"[User executed command: {command_name}]")]
        )

        result_dict = result if isinstance(result, dict) else {"result": result}
        model_message = Message(
            role=Role.MODEL,
            parts=[Part.from_function_response(ToolResult(
                call_id="",
                name=command_name,
                result=result_dict
            ))]
        )

        self._history.append(user_message)
        self._history.append(model_message)

    def _notify_model_of_cancellation(self, cancel_msg: str, partial_text: str = '') -> None:
        """Inject cancellation notice into history so model has context.

        This adds a user message noting the cancellation, so on the next turn
        the model understands why the previous response was cut short.

        NOTE: This feature is disabled by default (_notify_model_on_cancel=False)
        because it causes the model to hallucinate "interruptions" on subsequent
        turns, even when the cancellation was internal or expected.

        Args:
            cancel_msg: The cancellation message shown to user.
            partial_text: Any partial response text before cancellation.
        """
        # Skip notification if disabled (default) - prevents model hallucinations
        if not self._notify_model_on_cancel:
            self._trace(f"CANCEL_NOTIFY_SKIP: notifications disabled")
            return

        if not self._provider:
            return

        # Create a note for the model about what happened
        if partial_text:
            note = f"[System: Your previous response was cancelled by the user after: \"{partial_text[:100]}{'...' if len(partial_text) > 100 else ''}\"]"
        else:
            note = "[System: Your previous response was cancelled by the user before any output was generated.]"

        user_message = Message(
            role=Role.USER,
            parts=[Part.from_text(note)]
        )

        self._history.append(user_message)

    def generate(self, prompt: str) -> str:
        """Simple one-shot generation without tools or history.

        Uses ``provider.complete()`` with a single user message and no tools.
        Does not modify or use session history.
        """
        if not self._configured:
            raise RuntimeError("Session not configured.")
        self._ensure_provider()
        if not self._provider:
            raise RuntimeError("Session has no provider (skip_provider mode + auth incomplete)")

        messages = [Message.from_text(Role.USER, prompt)]
        with self._provider_access():
            turn_result = self._provider.complete(messages)
        response = self._unwrap_turn_result(turn_result)
        return response.get_text() or ''

    def send_message_with_parts(
        self,
        parts: List[Part],
        on_output: OutputCallback
    ) -> str:
        """Send a message with custom Part objects."""
        if not self._configured:
            raise RuntimeError("Session not configured.")
        self._ensure_provider()
        if not self._provider:
            raise RuntimeError("Session has no provider (skip_provider mode + auth incomplete)")

        return self._run_chat_loop_with_parts(parts, on_output)

    def _run_chat_loop_with_parts(
        self,
        parts: List[Part],
        on_output: OutputCallback
    ) -> str:
        """Internal function calling loop for multi-part messages."""
        if self._executor:
            self._executor.set_output_callback(on_output)

        turn_start = datetime.now()
        turn_data = {
            'prompt': 0,
            'output': 0,
            'total': 0,
            'start_time': turn_start.isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'function_calls': [],
        }
        response: Optional[ProviderResponse] = None

        try:
            # Proactive rate limiting: wait if needed before request
            self._pacer.pace()

            # Append user message to session history
            self._history.append(Message(role=Role.USER, parts=list(parts)))

            with self._telemetry.llm_span(
                model=self._model_name or "unknown",
                provider=self._provider.name if self._provider else "unknown",
                streaming=False,
                attributes=self._build_llm_span_attributes(),
            ) as llm_telemetry:
                self._record_input_messages_telemetry(llm_telemetry)
                with self._provider_access():
                    turn_result, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._get_effective_system_instruction(),
                            tools=self._get_tools_for_provider(),
                        ),
                        context="complete_with_parts",
                        on_retry=self._on_retry,
                        provider=self._provider
                    )
                response = self._unwrap_turn_result(turn_result)

                # Record model response in session history
                self._add_model_response_to_history(response)

                self._record_token_usage(response)
                self._accumulate_turn_tokens(response, turn_data)
                # Record token usage to telemetry span
                self._record_token_telemetry(llm_telemetry, response)

            from jaato_sdk.plugins.model_provider.types import FinishReason
            if response.finish_reason not in (FinishReason.STOP, FinishReason.UNKNOWN, FinishReason.TOOL_USE):
                logger.warning(f"Model stopped with finish_reason={response.finish_reason}")
                response_text = response.get_text()
                if response_text:
                    return f"{response_text}\n\n[Model stopped: {response.finish_reason}]"
                else:
                    return f"[Model stopped unexpectedly: {response.finish_reason}]"

            function_calls = list(response.function_calls) if response.function_calls else []
            while function_calls:
                response_text = response.get_text()
                if response_text and on_output:
                    on_output("model", response_text, "write")

                tool_results: List[ToolResult] = []

                for fc in function_calls:
                    name = fc.name
                    args = fc.args

                    # Emit hook: tool starting
                    if self._ui_hooks:
                        self._ui_hooks.on_tool_call_start(
                            agent_id=self._agent_id,
                            tool_name=name,
                            tool_args=args,
                            call_id=fc.id
                        )

                    fc_start = datetime.now()
                    if self._executor:
                        # Set up tool output callback for streaming output during execution
                        if self._ui_hooks and fc.id:
                            def tool_output_callback(chunk: str, _call_id=fc.id) -> None:
                                self._ui_hooks.on_tool_output(
                                    agent_id=self._agent_id,
                                    call_id=_call_id,
                                    chunk=chunk
                                )
                            self._executor.set_tool_output_callback(tool_output_callback)

                            # Done callback for auto-backgrounded tasks (legacy path)
                            def task_done_callback(
                                task_id: str, success: bool, error: 'Optional[str]',
                                duration: 'Optional[float]',
                                _call_id=fc.id, _name=name
                            ) -> None:
                                if self._ui_hooks:
                                    self._ui_hooks.on_tool_call_end(
                                        agent_id=self._agent_id,
                                        tool_name=_name,
                                        success=success,
                                        duration_seconds=duration or 0.0,
                                        error_message=error,
                                        call_id=_call_id,
                                    )
                            self._executor.set_task_done_callback(task_done_callback)

                        executor_result = self._executor.execute(name, args, call_id=fc.id, cancel_token=self._cancel_token)

                        # Clear the callbacks after execution
                        self._executor.set_tool_output_callback(None)
                        self._executor.set_task_done_callback(None)
                    else:
                        executor_result = (False, {"error": f"No executor registered for {name}"})
                    fc_end = datetime.now()

                    # Determine success and error message from executor result
                    fc_success = True
                    fc_error_message = None
                    fc_auto_backgrounded = False
                    fc_continuation_id = None
                    fc_show_output = None
                    fc_show_popup = None
                    if isinstance(executor_result, tuple) and len(executor_result) == 2:
                        fc_success = executor_result[0]
                        # Extract error message if tool failed
                        if not fc_success and isinstance(executor_result[1], dict):
                            fc_error_message = executor_result[1].get('error')
                        # Check if tool was auto-backgrounded or has continuation
                        if isinstance(executor_result[1], dict):
                            fc_auto_backgrounded = executor_result[1].get('auto_backgrounded', False)
                            fc_continuation_id = executor_result[1].get('continuation_id')
                            fc_show_output = executor_result[1].get('show_output')
                            fc_show_popup = executor_result[1].get('show_popup')

                    # Emit hook: tool ended
                    fc_duration = (fc_end - fc_start).total_seconds()
                    if self._ui_hooks:
                        self._ui_hooks.on_tool_call_end(
                            agent_id=self._agent_id,
                            tool_name=name,
                            success=fc_success,
                            duration_seconds=fc_duration,
                            error_message=fc_error_message,
                            call_id=fc.id,
                            backgrounded=fc_auto_backgrounded,
                            continuation_id=fc_continuation_id,
                            show_output=fc_show_output,
                            show_popup=fc_show_popup,
                        )

                    turn_data['function_calls'].append({
                        'name': name,
                        'start_time': fc_start.isoformat(),
                        'end_time': fc_end.isoformat(),
                        'duration_seconds': fc_duration,
                    })

                    tool_result = self._build_tool_result(fc, executor_result)
                    tool_results.append(tool_result)

                # Send tool results back (with retry for rate limits)
                self._pacer.pace()  # Proactive rate limiting

                # Proactive size guard: cap results before they enter history
                tool_results = self._cap_tool_results(tool_results)
                # Append tool results to session history
                tool_result_parts = [Part(function_response=r) for r in tool_results]
                self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

                with self._telemetry.llm_span(
                    model=self._model_name or "unknown",
                    provider=self._provider.name if self._provider else "unknown",
                    streaming=False,
                    attributes=self._build_llm_span_attributes(),
                ) as llm_telemetry:
                    self._record_input_messages_telemetry(llm_telemetry)
                    with self._provider_access():
                        turn_result, _retry_stats = with_retry(
                            lambda: self._provider.complete(
                                self._history.messages,
                                system_instruction=self._get_effective_system_instruction(),
                                tools=self._get_tools_for_provider(),
                            ),
                            context="complete_tool_results_parts",
                            on_retry=self._on_retry,
                            provider=self._provider
                        )
                    response = self._unwrap_turn_result(turn_result)

                    # Record model response in session history
                    self._add_model_response_to_history(response)

                    self._record_token_usage(response)
                    self._accumulate_turn_tokens(response, turn_data)
                    # Record token usage to telemetry span
                    self._record_token_telemetry(llm_telemetry, response)
                function_calls = list(response.function_calls) if response.function_calls else []

            final_text = response.get_text()
            if final_text and on_output:
                on_output("model", final_text, "write")

            return final_text or ''

        except Exception as exc:
            # Route provider errors through output callback before re-raising
            exc_name = type(exc).__name__
            exc_module = type(exc).__module__

            is_provider_error = 'model_provider' in exc_module or exc_name in (
                'AnthropicProviderError', 'APIKeyNotFoundError', 'APIKeyInvalidError',
                'RateLimitError', 'ContextLimitError', 'ModelNotFoundError',
                'OverloadedError', 'UsageLimitError',
                'GitHubModelsError', 'TokenNotFoundError', 'TokenInvalidError',
                'TokenPermissionError', 'ModelsDisabledError',
                'JaatoAuthError', 'CredentialsNotFoundError', 'CredentialsInvalidError',
                'CredentialsPermissionError', 'ProjectConfigurationError',
            )

            if is_provider_error and on_output:
                error_msg = f"[Error] {exc_name}: {str(exc)}"
                on_output("error", error_msg, "write")
                self._trace(f"PROVIDER_ERROR routed to callback: {exc_name}")

            raise

        finally:
            turn_end = datetime.now()
            turn_data['end_time'] = turn_end.isoformat()
            turn_data['duration_seconds'] = (turn_end - turn_start).total_seconds()

            if turn_data['total'] > 0:
                self._turn_accounting.append(turn_data)

    # ==================== Context Garbage Collection ====================

    def set_gc_plugin(
        self,
        plugin: GCPlugin,
        config: Optional[GCConfig] = None
    ) -> None:
        """Set the GC plugin for context management."""
        self._gc_plugin = plugin
        self._gc_config = config or GCConfig()

    def remove_gc_plugin(self) -> None:
        """Remove the GC plugin."""
        if self._gc_plugin:
            self._gc_plugin.shutdown()
        self._gc_plugin = None
        self._gc_config = None

    def manual_gc(self) -> GCResult:
        """Manually trigger garbage collection."""
        if not self._gc_plugin:
            raise RuntimeError("No GC plugin configured.")
        if not self._gc_config:
            self._gc_config = GCConfig()

        history = self.get_history()
        context_usage = self.get_context_usage()
        self._trace(
            f"MANUAL_GC: triggering manual GC (usage={context_usage.get('percent_used', 0):.1f}%)"
        )

        new_history, result = self._gc_plugin.collect(
            history, context_usage, self._gc_config, GCTriggerReason.MANUAL,
            budget=self._instruction_budget,
        )

        if result.success:
            if result.items_collected == 0:
                self._trace(
                    f"MANUAL_GC: WARNING - GC ran but collected 0 items. "
                    f"Check preserve_recent_turns setting vs actual turn count. "
                    f"Details: {result.details}"
                )
            else:
                self._trace(
                    f"MANUAL_GC: collected {result.items_collected} items, "
                    f"freed {result.tokens_freed} tokens"
                )
            new_history = ensure_tool_call_integrity(
                new_history, trace_fn=lambda m: self._trace(f"MANUAL_GC: {m}"),
            )
            self._history.replace(new_history)
            self._gc_history.append(result)

            # Sync budget with GC changes
            self._apply_gc_removal_list(result)
            self._emit_instruction_budget_update()

        return result

    def get_gc_history(self) -> List[GCResult]:
        """Get history of GC operations."""
        return list(self._gc_history)

    def _maybe_collect_before_send(self) -> Optional[GCResult]:
        """Check and perform GC if needed before sending."""
        if not self._gc_plugin or not self._gc_config:
            logger.info(
                "GC_CHECK: skipped — plugin=%s config=%s",
                self._gc_plugin is not None, self._gc_config is not None,
            )
            return None

        context_usage = self.get_context_usage()
        logger.info(
            "GC_CHECK: plugin=%s usage=%.1f%% threshold=%.1f%% target=%.1f%% continuous=%s",
            type(self._gc_plugin).__name__,
            context_usage.get('percent_used', 0),
            self._gc_config.threshold_percent,
            self._gc_config.target_percent,
            getattr(self._gc_config, 'continuous_mode', '?'),
        )
        should_gc, reason = self._gc_plugin.should_collect(context_usage, self._gc_config)
        logger.info(
            "GC_CHECK: should_gc=%s reason=%s",
            should_gc, reason.value if reason else None,
        )

        if should_gc and reason:
            self._trace(
                f"GC_BEFORE_SEND: triggering GC (reason={reason.value}, "
                f"usage={context_usage.get('percent_used', 0):.1f}%)"
            )
            history = self.get_history()

            # Build pre-GC telemetry attributes
            gc_attrs = self._build_gc_span_attributes(
                context_usage, pre_collect=True,
            )

            with self._telemetry.gc_span(
                trigger_reason=reason.value,
                strategy=self._gc_plugin.name,
                attributes=gc_attrs,
            ) as gc_span:
                new_history, result = self._gc_plugin.collect(
                    history, context_usage, self._gc_config, reason,
                    budget=self._instruction_budget,
                )

                if result.success:
                    if result.items_collected == 0:
                        # GC ran but collected nothing - this is often surprising to users
                        self._trace(
                            f"GC_BEFORE_SEND: WARNING - GC triggered but collected 0 items. "
                            f"Check preserve_recent_turns setting vs actual turn count. "
                            f"Details: {result.details}"
                        )
                    else:
                        self._trace(
                            f"GC_BEFORE_SEND: collected {result.items_collected} items, "
                            f"freed {result.tokens_freed} tokens"
                        )
                    new_history = ensure_tool_call_integrity(
                        new_history, trace_fn=lambda m: self._trace(f"GC_BEFORE_SEND: {m}"),
                    )
                    self._history.replace(new_history)
                    self._gc_history.append(result)

                    # Sync budget with GC changes (notifies cache plugin
                    # via on_gc_result with the active span attached)
                    self._apply_gc_removal_list(result, gc_span=gc_span)
                    self._emit_instruction_budget_update()

                # Populate post-GC span attributes from the result
                self._populate_gc_span_result(gc_span, result)

            return result

        return None

    # ==================== Cache Control ====================

    def set_cache_plugin(self, plugin: Any) -> None:
        """Set the cache control plugin for this session.

        Attaches the plugin and wires it to the provider and budget.

        Args:
            plugin: A CachePlugin instance (duck-typed).
        """
        self._cache_plugin = plugin

        # Forward current budget
        if self._instruction_budget and hasattr(plugin, 'set_budget'):
            plugin.set_budget(self._instruction_budget)

        # Attach to provider
        if self._provider and hasattr(self._provider, 'set_cache_plugin'):
            self._provider.set_cache_plugin(plugin)

    def remove_cache_plugin(self) -> None:
        """Remove the cache control plugin."""
        if self._cache_plugin and hasattr(self._cache_plugin, 'shutdown'):
            self._cache_plugin.shutdown()
        self._cache_plugin = None

        # Detach from provider
        if self._provider and hasattr(self._provider, 'set_cache_plugin'):
            self._provider.set_cache_plugin(None)

    # ==================== Thinking Mode ====================

    def set_thinking_plugin(self, plugin: 'ThinkingPlugin') -> None:
        """Set the thinking plugin for controlling reasoning modes.

        The thinking plugin provides user commands for controlling extended
        thinking capabilities (e.g., Anthropic extended thinking, Gemini
        thinking mode).

        Args:
            plugin: The ThinkingPlugin instance.
        """
        self._thinking_plugin = plugin

        # Give plugin access to this session
        if hasattr(plugin, 'set_session'):
            plugin.set_session(self)

        # Register user commands
        if hasattr(plugin, 'get_user_commands'):
            for cmd in plugin.get_user_commands():
                self._user_commands[cmd.name] = cmd

        # Register executors
        if hasattr(plugin, 'get_executors') and self._executor:
            for name, fn in plugin.get_executors().items():
                self._executor.register(name, fn)

    def remove_thinking_plugin(self) -> None:
        """Remove the thinking plugin."""
        if self._thinking_plugin:
            if hasattr(self._thinking_plugin, 'shutdown'):
                self._thinking_plugin.shutdown()
        self._thinking_plugin = None

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set thinking mode configuration directly on the provider.

        This is a convenience method that bypasses the plugin and sets
        the thinking configuration directly on the provider.

        Args:
            config: ThinkingConfig with enabled flag and budget.
        """
        if self._provider and hasattr(self._provider, 'set_thinking_config'):
            self._provider.set_thinking_config(config)

    def get_thinking_config(self) -> Optional[ThinkingConfig]:
        """Get current thinking configuration from the plugin.

        Returns:
            Current ThinkingConfig if plugin is set, None otherwise.
        """
        if self._thinking_plugin and hasattr(self._thinking_plugin, 'get_current_config'):
            return self._thinking_plugin.get_current_config()
        return None

    def supports_thinking(self) -> bool:
        """Check if the current provider supports thinking mode.

        Returns:
            True if thinking is supported, False otherwise.
        """
        if self._provider and hasattr(self._provider, 'supports_thinking'):
            return self._provider.supports_thinking()
        return False

    # ==================== Session Persistence ====================

    def set_session_plugin(
        self,
        plugin: SessionPlugin,
        config: Optional[SessionConfig] = None
    ) -> None:
        """Set the session plugin for persistence."""
        self._session_plugin = plugin
        self._session_config = config or SessionConfig()

        if hasattr(plugin, 'set_session'):
            plugin.set_session(self)

        if hasattr(plugin, 'get_user_commands'):
            for cmd in plugin.get_user_commands():
                self._user_commands[cmd.name] = cmd

        if hasattr(plugin, 'get_executors') and self._executor:
            for name, fn in plugin.get_executors().items():
                self._executor.register(name, fn)

        if hasattr(plugin, 'get_tool_schemas'):
            session_schemas = plugin.get_tool_schemas()
            if session_schemas:
                current_tools = list(self._tools) if self._tools else []
                current_tools.extend(session_schemas)
                self._tools = current_tools

        if self._session_config.auto_resume_last:
            state = self._session_plugin.on_session_start(self._session_config)
            if state:
                self._restore_session_state(state)

    def remove_session_plugin(self) -> None:
        """Remove the session plugin."""
        if self._session_plugin:
            self._session_plugin.shutdown()
        self._session_plugin = None
        self._session_config = None

    def save_session(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> str:
        """Save the current session."""
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured.")

        state = self._get_session_state(session_id, user_inputs)
        self._session_plugin.save(state)

        if hasattr(self._session_plugin, 'set_current_session_id'):
            self._session_plugin.set_current_session_id(state.session_id)

        return state.session_id

    def resume_session(self, session_id: str) -> SessionState:
        """Resume a previously saved session."""
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured.")

        state = self._session_plugin.load(session_id)
        self._restore_session_state(state)
        return state

    def list_sessions(self) -> List[SessionInfo]:
        """List all available sessions."""
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured.")
        return self._session_plugin.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session."""
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured.")
        return self._session_plugin.delete(session_id)

    def _get_session_state(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> SessionState:
        """Build a SessionState from current state."""
        if not session_id:
            if (self._session_plugin and
                    hasattr(self._session_plugin, 'get_current_session_id')):
                session_id = self._session_plugin.get_current_session_id()
            if not session_id:
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        now = datetime.now()
        turn_accounting = self.get_turn_accounting()

        description = None
        if self._session_plugin and hasattr(self._session_plugin, '_session_description'):
            description = self._session_plugin._session_description

        # Snapshot session-attached state at save time.  Invokes every
        # registered provider so the snapshot reflects live values
        # (extension-owned incrementally-mutated structures stay
        # correct without push-on-every-mutation).  Empty dict
        # collapses to None so old persistence files round-trip
        # unchanged when nothing has been attached.
        attached_state = self.get_all_session_state()
        return SessionState(
            session_id=session_id,
            history=self.get_history(),
            created_at=now,
            updated_at=now,
            turn_count=len(turn_accounting),
            turn_accounting=turn_accounting,
            user_inputs=user_inputs or [],
            project=self._runtime.project,
            location=self._runtime.location,
            model=self._model_name,
            description=description,
            session_state=attached_state if attached_state else None,
        )

    def _restore_session_state(self, state: SessionState) -> None:
        """Restore session state from a SessionState."""
        self.reset_session(state.history)
        self._turn_accounting = list(state.turn_accounting)
        # Re-attach session-state values from the persisted snapshot.
        # Routes through set_session_state so the JSON-serialisability
        # check fires (defensive: persisted state should already be
        # serialisable, but a corrupted file shouldn't silently
        # inject a non-JSON value into the live container).
        # Consumer hooks fire via the normal session-creation path
        # (when the persisted session is loaded by SessionManager) and
        # can re-register providers / instantiate runtime structures
        # from the restored values.
        if state.session_state:
            for key, value in state.session_state.items():
                self.set_session_state(key, value)

    def _notify_session_turn_complete(self) -> None:
        """Notify session plugin that a turn completed."""
        if not self._session_plugin or not self._session_config:
            return

        state = self._get_session_state()

        if hasattr(self._session_plugin, 'increment_turn_count'):
            self._session_plugin.increment_turn_count()

        self._session_plugin.on_turn_complete(state, self._session_config)

    # ------------------------------------------------------------------
    # Conversation fork
    # ------------------------------------------------------------------

    def replay_messages(
        self,
        messages: List[Message],
        *,
        timeout: float = 120.0,
    ) -> str:
        """Run a one-shot completion against an arbitrary message list.

        Public *capability primitive* for session-manipulation tools that
        live outside the session (fork/interrogate/replay).  Acquires
        exclusive provider access (so concurrent in-flight turn calls are
        serialized — important for providers with strict concurrency
        limits), runs ``provider.complete()`` against the supplied
        ``messages``, and returns the model's text response.

        Does **not** modify session history, ``_turn_accounting``, or any
        other turn-loop state.  Callers are expected to have already
        constructed the message list they want to replay (typically by
        snapshotting ``get_history()``, slicing at a point returned by
        ``resolve_fork_point()``, and appending whatever new messages
        the replay should contain).

        Provider exclusion: if the session is mid-turn, this waits for
        the current provider call to finish, pauses the session's next
        call, runs the replay, then resumes the session.

        Args:
            messages: The full message list to send to the provider.
                Caller owns construction; this method does not snapshot
                or mutate session history.
            timeout: Maximum seconds to wait for exclusive provider
                access.

        Returns:
            The model's text response (empty string if the response had
            no text content).

        Raises:
            TimeoutError: If the provider is not available within
                *timeout* seconds.
            RuntimeError: If the session has no configured provider.
        """
        if not self._provider:
            raise RuntimeError("Session not configured — cannot replay.")

        self._fork_gate.clear()
        if not self._provider_idle.wait(timeout=timeout):
            self._fork_gate.set()
            raise TimeoutError(
                f"Target session provider busy for >{timeout}s"
            )

        try:
            result = self._provider.complete(
                messages,
                system_instruction=self._get_effective_system_instruction(),
            )
            response = self._unwrap_turn_result(result)
            return response.get_text() or ""
        finally:
            self._fork_gate.set()

    def _get_effective_system_instruction(self) -> Optional[str]:
        """System instruction to send to the provider on this turn.

        Equal to the assembled :attr:`_system_instruction` plus a single
        line naming the current tier when tier mode is active.
        Recomputed dynamically (not stored on ``_system_instruction``)
        so tier switches take effect immediately without re-assembling
        the whole prompt — and so the assembled instruction stays a
        stable cache anchor for providers that key prompt cache on it.
        """
        if self._active_tier is None:
            return self._system_instruction
        tier_line = (
            f"You are currently operating in the `{self._active_tier}` tier."
        )
        if self._system_instruction:
            return self._system_instruction + "\n\n" + tier_line
        return tier_line

    def switch_tier(self, requested_tier: str) -> Dict[str, Any]:
        """Switch the session's active model tier.

        Called by the ``enter_tier`` lifecycle tool.  Resolves the
        requested tier through ``_tier_config.model_for`` (which routes
        to the configured fallback when the tier isn't declared),
        re-points the active provider at the new tier's model via
        ``provider.connect(model, skip_model_test=True)`` (cheap — no
        network round-trip, just sets ``self._model_name`` on the
        provider), updates ``_active_tier``, and returns a structured
        result the tool surfaces back to the model.

        The returned dict tells the model exactly what happened:
            * ``status``: ``"switched"`` (tier changed),
              ``"already_at_tier"`` (idempotent no-op), or
              ``"fallback_used"`` (requested tier wasn't declared, fell
              back to ``tier_config.tier_fallback``).
            * ``active_tier``: the actual tier the session is now in.
            * ``requested_tier``: what the model asked for (helps the
              model self-correct when fallback fired).
            * ``model``: the model name now in use.

        Raises:
            RuntimeError: If tier mode isn't active for this session
                (the ``enter_tier`` tool shouldn't be registered then,
                so this is a programmer-error guard).
            ValueError: If ``requested_tier`` isn't one of the framework's
                three valid names; raised through to surface as an
                ``error`` in the tool result.
        """
        if self._tier_config is None or self._active_tier is None:
            raise RuntimeError(
                "switch_tier called but session is in single-model mode "
                "(no tier config); enter_tier tool should not be registered"
            )

        actual_tier, entry = self._tier_config.model_for(requested_tier)

        if actual_tier == self._active_tier:
            return {
                "status": "already_at_tier",
                "active_tier": actual_tier,
                "requested_tier": requested_tier,
                "model": entry.model,
            }

        if self._provider is not None:
            try:
                self._provider.connect(entry.model, skip_model_test=True)
            except Exception as exc:
                logger.warning(
                    "switch_tier: provider.connect(%s) failed: %s",
                    entry.model, exc,
                )
                raise

        previous_tier = self._active_tier
        self._active_tier = actual_tier
        self._model_name = entry.model

        logger.info(
            "Tier switch: %s → %s (model %s)",
            previous_tier, actual_tier, entry.model,
        )

        return {
            "status": (
                "fallback_used"
                if actual_tier != requested_tier
                else "switched"
            ),
            "active_tier": actual_tier,
            "requested_tier": requested_tier,
            "model": entry.model,
        }

    def set_initial_history(self, messages: List[Message]) -> None:
        """Seed an empty session with replayed conversation history.

        Pre-turn-loop primitive used by spawn-from-snapshot callers
        (premium handoff via ``fork_session_from_history``, waypoint
        fork-to-session, test harnesses).  The session must be **idle**
        and its history must be **empty** at call time — both are true
        for a freshly created session before any user/agent turn.

        Replayed messages may include tool-use / tool-result blocks for
        tools the new session's profile doesn't expose.  These remain
        as inert context for the model to read; we deliberately do not
        strip them, so handoff preserves the source agent's reasoning
        trace.  If a provider chokes on unknown tool references at
        sampling time, that's a provider-specific concern surfaced via
        the normal tool-result reconciliation path on the first new
        turn.

        The session's system instruction is NOT touched — it comes
        from the new session's own agent/profile, independently of the
        replayed user/assistant turns.

        Args:
            messages: The conversation history to seed.  Caller owns
                the list; a shallow copy is taken.

        Raises:
            RuntimeError: If the session is not idle or its history
                already contains messages.  This is a defensive guard
                — the right place to call this method is between
                ``server.initialize()`` and any ``inject_prompt`` /
                ``handle_request``.
        """
        if self._is_running:
            raise RuntimeError(
                "set_initial_history requires an idle session; this one "
                "is mid-turn."
            )
        if self._history.messages_ref:
            raise RuntimeError(
                "set_initial_history requires an empty history; this "
                f"session already has {len(self._history.messages_ref)} "
                "messages."
            )
        self._history.replace(messages)

    def resolve_fork_point(
        self,
        history: List[Message],
        after_message: Optional[int] = None,
        after_tool_call: Optional[str] = None,
        after_timestamp: Optional[str] = None,
    ) -> int:
        """Resolve a fork point specifier to a message index.

        Exactly one specifier should be provided.  If none are given,
        returns the last message index (full history fork).

        Args:
            history: The message list to search.
            after_message: Direct message index.
            after_tool_call: Tool call ID — returns the index of the
                message containing this ``FunctionCall.id`` or the
                corresponding ``ToolResult``.
            after_timestamp: HH:MM:SS or ISO timestamp — returns the
                index of the last message at or before this time
                (best-effort, based on session turn accounting).

        Returns:
            Message index (0-based, inclusive).
        """
        last = len(history) - 1

        if after_message is not None:
            return max(0, min(after_message, last))

        if after_tool_call is not None:
            # Scan backwards — the most recent match is usually the one wanted
            for i in range(last, -1, -1):
                for part in history[i].parts:
                    if part.function_call and part.function_call.id == after_tool_call:
                        return i
                    if part.function_response and part.function_response.call_id == after_tool_call:
                        return i
            raise ValueError(f"Tool call ID not found in history: {after_tool_call}")

        if after_timestamp is not None:
            # Best-effort: correlate with turn accounting timestamps
            # Turn accounting stores ISO start_time per turn; find the
            # last turn that started at or before the requested time.
            target = self._parse_fork_timestamp(after_timestamp)
            if target is not None and self._turn_accounting:
                # Each turn maps to a range of messages.  Walk turns in
                # reverse, find the last one whose start_time <= target,
                # then return the last message index in that turn's range.
                cumulative = 0
                turn_boundaries = []
                for ta in self._turn_accounting:
                    # Count messages contributed by this turn (1 user + N model + tool results)
                    fc_count = len(ta.get('function_calls', []))
                    # user message + model response + (tool results + model response) per fc
                    msgs_in_turn = 1 + 1 + fc_count * 2 if fc_count else 2
                    turn_boundaries.append((cumulative, cumulative + msgs_in_turn - 1, ta))
                    cumulative += msgs_in_turn

                for start_idx, end_idx, ta in reversed(turn_boundaries):
                    turn_start = ta.get('start_time', '')
                    if turn_start and turn_start <= target:
                        return min(end_idx, last)

            # Fallback: return full history
            return last

        # No specifier — full history
        return last

    @staticmethod
    def _parse_fork_timestamp(ts: str) -> Optional[str]:
        """Normalize a fork timestamp to ISO format for comparison.

        Accepts HH:MM:SS (interpreted as today) or full ISO format.
        Returns an ISO string suitable for lexicographic comparison
        with turn accounting ``start_time`` values, or ``None`` if
        the timestamp cannot be parsed.
        """
        # Full ISO format — use as-is
        if 'T' in ts or len(ts) > 8:
            return ts

        # HH:MM:SS — expand to today's date
        try:
            t = datetime.strptime(ts, "%H:%M:%S")
            today = datetime.now().replace(
                hour=t.hour, minute=t.minute, second=t.second, microsecond=0
            )
            return today.isoformat()
        except ValueError:
            return None

    def close_session(self) -> None:
        """Close the current session."""
        if self._session_plugin and self._session_config:
            state = self._get_session_state()
            self._session_plugin.on_session_end(state, self._session_config)


__all__ = ['JaatoSession']
