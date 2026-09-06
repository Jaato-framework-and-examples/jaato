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
from base64 import b64encode as _b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace as _dc_replace
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, TYPE_CHECKING

from .message_queue import (
    HIGH_PRIORITY_SOURCES,
    IDLE_ONLY_SOURCES,
    MessageQueue,
    QueuedMessage,
    SourceType,
)
from .session_history import SessionHistory
from .gc_support import (
    apply_gc_removal_list as _gc_apply_removal_list,
    build_gc_span_attributes as _gc_build_span_attributes,
    populate_gc_span_result as _gc_populate_span_result,
    run_gc as _gc_run,
)
from .tool_result_truncation import (
    cap_tool_results as _cap_tool_results_impl,
    truncate_results_to_fit as _truncate_results_to_fit_impl,
)
from .tool_result_builder import (
    extract_multimodal_attachments as _extract_multimodal_attachments_impl,
    normalize_result_dict as _normalize_result_dict_impl,
    split_executor_result as _split_executor_result_impl,
)
from .instruction_budget_builder import (
    TokenCountRequest as _TokenCountRequest,
    count_tokens as _builder_count_tokens,
    collect_instruction_texts as _builder_collect_instruction_texts,
    apply_instruction_counts as _builder_apply_instruction_counts,
)
from .instruction_suppression import (
    PIECE_CONSTANTS,
    PIECE_DISK,
    PIECE_SECURITY,
    normalize_suppression,
)
from .session_persistence import SessionPersistence
from .session_telemetry import (
    build_input_messages,
    classify_cache_outcome,
    response_to_openinference,
)

logger = logging.getLogger(__name__)

# Reserved ``call_id`` for media the MODEL generated (as opposed to media a
# tool produced).  Model output belongs to no tool call, but reuses the
# tool-output delivery channel so that clients need no second subscription;
# this id is how they tell the two apart.
#
# Imported rather than defined here: the daemon writes this value once and
# every CLIENT compares against it, so the client-side package is where it
# belongs.  Defining it on both sides makes it a shared constant with no
# owner, which is how the two drift.
from jaato_sdk.events import MODEL_MEDIA_CALL_ID       # noqa: F401  (re-export)

from .ai_tool_runner import ToolExecutor
from .session_context import set_current_session
from .tool_id_map import StreamScrubber
from .retry_utils import with_retry, RequestPacer, RetryCallback, RetryConfig, is_context_limit_error
from .token_accounting import TokenLedger
from jaato_sdk.plugins.base import HelpLines, UserCommand, OutputCallback
from .plugins.gc import GCConfig, GCPlugin, GCRemovalItem, GCResult, GCTriggerReason
from .plugins.gc.utils import (
    dedup_identical_tool_results,
    ensure_tool_call_integrity,
    estimate_history_tokens,
)
from .instruction_budget import (
    InstructionBudget,
    InstructionSource,
    estimate_tokens,
    SystemChildType,
    DEFAULT_SYSTEM_POLICIES,
    GCPolicy,
    PluginToolType,
    DEFAULT_TOOL_POLICIES,
    PayloadExceedsContextError,
)
from .instruction_token_cache import InstructionTokenCache
from .plugins.session import SessionPlugin, SessionConfig, SessionState, SessionInfo
from .plugins.streaming import StreamManager, StreamingCapable, StreamChunk, StreamUpdate
from .plugins.model_provider.base import UsageUpdateCallback, GCThresholdCallback
from jaato_sdk.plugins.model_provider.types import (
    Attachment,
    MediaDelta,
    CancelledException,
    CancelToken,
    DISCOVERABILITY_EAGER,
    DISCOVERABILITY_DEFERRED,
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
    replay_excerpt,
    tool_result_is_error,
    tool_result_status,
    unexecuted_call_error,
    unreadable_arguments_error,
)

if TYPE_CHECKING:
    from .jaato_runtime import JaatoRuntime
    from .plugins.model_provider.base import ModelProviderPlugin
    from .plugins.subagent.ui_hooks import AgentUIHooks
    from .plugins.telemetry import TelemetryPlugin
    from .plugins.thinking import ThinkingPlugin
    from .model_tiers import ModelTierConfig
    from .budget_control import BudgetControlConfig, BudgetTracker

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

# Truncation recovery (#749).  Finish reasons a turn may be CONTINUED
# from after it was cut short, rather than lost.
#
# Only the output cap.  A cap is an authoring mistake -- the model asked
# for more output than it was allowed -- and the corrective action is
# obvious and cheap: say less, in smaller steps.  ``SAFETY`` is not
# here on purpose: re-driving a filtered turn is a different question
# and should not happen automatically, and #751's message deliberately
# does not tell a filtered turn to shorten its output.  ``ERROR`` is a
# provider fault, which ``with_retry`` already owns.
TRUNCATION_RECOVERY_REASONS = frozenset({FinishReason.MAX_TOKENS})

# How many times ONE turn may be continued after hitting the output
# cap.  Small on purpose: the point is to give the model a chance to
# anchor differently, not to let a truncation that recurs identically
# loop.  Past the budget the turn ends exactly as it does today, with
# the reason preserved.
TRUNCATION_RECOVERY_BUDGET = 2


def _telemetry_json_default(obj: Any) -> str:
    """``json.dumps`` ``default=`` for telemetry span output.

    Tool results can carry raw binary (a multimodal ``readFile`` returns
    ``image_data`` bytes BEFORE ``_build_tool_result`` strips it).  A span
    attribute must never embed megabytes of binary nor crash the model loop on
    a non-serializable value — summarise bytes compactly and ``repr`` anything
    else exotic.
    """
    if isinstance(obj, (bytes, bytearray)):
        return f"<{type(obj).__name__}: {len(obj)} bytes>"
    return repr(obj)


def _telemetry_safe_json(value: Any) -> str:
    """Serialise a tool result for an OTel span without choking on binary."""
    return json.dumps(value, default=_telemetry_json_default)


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


# Introspection's tools (list_tools / get_tool_schemas) give the model SCHEMA
# VISIBILITY — the ability to discover deferred tools.
_INTROSPECTION_TOOL_NAMES = frozenset({"list_tools", "get_tool_schemas"})


def _has_deferred_to_discover(exposed_schemas, profile_plugins, preloaded,
                              tool_scopes, plugin_of) -> bool:
    """Is there genuinely something for the model to discover?

    True iff a discoverable (non-``core``) tool that belongs to a profile plugin
    which is NOT ``(preload)``-ed and NOT scoped-out is exposed.  When this is
    False, nothing is pending *discovery* — but introspection is NOT necessarily
    dead weight: a session with eager/preloaded real tools still needs it for
    *re-inspection* after GC offloads their schemas/instructions.  The full drop
    decision lives in :func:`_should_drop_introspection`.

    Pure (no session/registry deps) so it is unit-testable.  ``plugin_of(name)``
    maps a tool name to its owning plugin name (or ``None``).
    """
    profile = set(profile_plugins or [])
    pre = set(preloaded or [])
    scopes = tool_scopes or {}
    for sc in exposed_schemas:
        if sc.name in _INTROSPECTION_TOOL_NAMES:
            continue
        if getattr(sc, "discoverability", DISCOVERABILITY_DEFERRED) == DISCOVERABILITY_EAGER:
            continue  # eager — not something to "discover"
        pname = plugin_of(sc.name)
        if pname is None or pname not in profile or pname in pre:
            continue
        allow = scopes.get(pname)
        if allow is not None and sc.name not in allow:
            continue  # scoped out — not exposed
        return True
    return False


def _should_drop_introspection(has_deferred_to_discover, tool_names) -> bool:
    """Whether to drop introspection's discovery tools from a session's wire.

    Drop ONLY when the wire is genuinely empty of real tools: nothing is
    deferred to discover AND there are no real (non-introspection) tools present.

    Keeping introspection whenever real tools exist — even all-eager/preloaded
    ones with nothing currently deferred — is deliberate: preloading a tool only
    means its schema/instructions *start* in-context; GC can offload them under
    context pressure, after which the model needs ``list_tools`` /
    ``get_tool_schemas`` to RE-INSPECT the tool.  Dropping introspection there
    would both strip that capability AND (for a no-``suppress_base`` session)
    leave a "discover via list_tools" prose nudge with no list_tools on the wire
    — the model then invents the call, hits no-executor, and loops (the ex08
    lead hang).  A truly empty wire (e.g. ``plugins=[]``) has nothing to inspect,
    so introspection is correctly dropped — keeping the ``plugins=[]`` "no tools"
    semantic intact.

    Pure, so it is unit-testable.  ``tool_names`` is the current wire's tool
    names; ``has_deferred_to_discover`` is :func:`_has_deferred_to_discover`'s
    result.
    """
    if has_deferred_to_discover:
        return False
    return not any(n not in _INTROSPECTION_TOOL_NAMES for n in tool_names)


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
            plugins=["cli", "web_search"],
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
        # Lazily-loaded pricing table used to stamp cost on LLM telemetry
        # spans when the provider doesn't report one (mirrors the daemon's
        # core.py:_build_usage precedence, but computed while the span is
        # still open — the daemon boundary runs after the span closes).
        # ``_span_pricing`` is None until first use; ``_span_pricing_loaded``
        # guards the one-time load so cost-free sessions never read the JSON.
        self._span_pricing = None
        self._span_pricing_loaded = False
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

        # Tier to return to when the tier currently at the wheel finishes
        # its completion.  Armed by ``switch_tier`` when the entered tier
        # declares ``exit_on: completion``; consumed by
        # :meth:`_exit_completion_tier_if_settled`.  ``None`` = nothing
        # pending, which is every session using no such tier.
        #
        # Deferred rather than nested: the exit is state a lifecycle tool
        # stamps and the next settled completion consumes -- the same
        # shape as the Path-1 tool_choice retry above.  Running the tier's
        # completion INSIDE the ``enter_tier`` executor would mean a
        # provider call from a tool executor, which can run in a worker
        # thread under JAATO_PARALLEL_TOOLS with no lock over history, and
        # nothing in this session has ever done that.
        self._pending_tier_return: Optional[str] = None

        # Path 1 quirk state (server 0.6.195+).  When
        # ``LifecycleTools._execute_signal_completion`` returns a
        # ``validation_failed`` error, ``_execute_tools_and_continue``
        # stamps this with the failing tool's name so the NEXT
        # ``provider.complete()`` in the same turn can request
        # named-function ``tool_choice``.  Cleared after one consumed
        # call so a single retry-with-xgrammar is the contract.
        # ``None`` = no pending retry.  The provider plugin decides
        # whether to honor it (vllm honors when its
        # ``force_tool_choice_for_lifecycle`` quirk is True;
        # providers without the quirk ignore the kwarg).  See
        # ``feedback_llama31_vllm_auto_mode_stringifies_args`` and
        # ``project_backlog_vllm_provider_typed_tool_args`` for the
        # diagnosis + design.
        self._pending_tool_choice_name: Optional[str] = None

        # Profile-declared completion processors (kb-authored Python
        # under ``.jaato/scripts/processors/``).  Each entry is a
        # ``CompletionProcessor`` carrying a script path + optional
        # output template + on_error policy.  After
        # ``signal_completion`` validates against
        # ``_completion_payload_schema``, ``LifecycleTools`` runs each
        # processor in turn: probes for ``render`` (produces output
        # bytes, optionally writes to disk) and/or ``validate``
        # (returns error list, blocks completion).  Empty list = no
        # processors (agents handle output themselves; no semantic
        # post-checks).  ``CompletionProcessor`` typed as ``Any`` here
        # to avoid a top-level subagent-config import; concrete type
        # is ``shared.plugins.subagent.config.CompletionProcessor``.
        # See ``shared/completion_processors.py`` for the loader,
        # ledger builder, and invocation pipeline.
        self._completion_processors: List[Any] = []

        # Completion lifecycle tracking — flipped True by
        # ``LifecycleTools._execute_signal_completion`` on the first
        # successful invocation.  The completion-nudge guard reads this
        # at loop-exit (top-level: ``core.py`` model_thread finally;
        # subagent: end of ``_run_subagent_async``) to decide whether
        # to inject a nudge prompt back into the session asking the
        # agent to call ``signal_completion`` before terminating.
        # ``_completion_nudges_fired`` bounds the retry budget, and unlike
        # ``_signal_completion_called`` it is per SESSION: a nudge
        # re-prompts, so a per-turn budget is refunded by the very turn it
        # paid for and bounds nothing (#767).  See
        # ``_begin_turn_completion_state``.
        self._signal_completion_called: bool = False
        self._completion_nudges_fired: int = 0
        # Set in configure() when introspection's tools are dropped because there
        # is nothing deferred to discover — read by introspection's
        # get_system_instructions to suppress the now-mismatched discovery
        # guidance (keeps the instruction-gate aligned with the tool-gate).
        self._introspection_guidance_suppressed: bool = False

        # Per-turn model-tier config.  ``_tier_config`` is the resolved
        # view (built from profile.tiers or env vars).  ``_active_tier``
        # tracks which tier the session is currently operating in;
        # mutated by the ``enter_tier`` lifecycle tool, consulted by
        # provider model selection and by system-instruction assembly.
        # Both ``None`` means single-model mode — no ``enter_tier`` tool
        # is registered, no system-prompt augmentation, the provider
        # uses the legacy ``self._model_name``.
        self._tier_config: Optional['ModelTierConfig'] = None
        # Budget control (shared/budget_control.py).  ``None`` = unbudgeted,
        # which is the default and the pre-existing behaviour.  When set, the
        # tracker accumulates spend from the SAME accounting the session
        # already does (no new measurement path) and returns degrade rungs to
        # apply.  ``_budget_terminal_action`` latches the last terminal action
        # a rung asked for, so a caller can see WHY a session wound down.
        self._budget_tracker: Optional['BudgetTracker'] = None
        self._budget_terminal_action: Optional[str] = None
        # Set once an ``abort`` rung fires.  Gates EVERY subsequent turn —
        # see ``_refuse_if_budget_exhausted``.  Distinct from
        # ``_budget_terminal_action`` (which also latches finalize/escalate,
        # neither of which stops anything).
        self._budget_exhausted_reason: Optional[str] = None
        # True when the LAST send_message was refused by the budget gate
        # (no turn ran).  Read runner-side to suppress the post-turn
        # TurnCompletedEvent — see ``was_last_send_refused``.
        self._last_send_refused: bool = False
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
        # V2 cross-provider tiers: per-provider instance cache (provider_name ->
        # ModelProviderPlugin) so a tier that declares a DIFFERENT provider gets
        # its own cached instance, switched in O(1) by switch_tier without
        # re-paying create_provider's init cost on every text<->vision hop.
        # ``_active_provider_name`` tracks which provider self._provider IS
        # (the provider's own .name is unreliable for subclassed providers like
        # zhipuai-extends-anthropic, so we track the name we created it under).
        # Empty/None until the default provider is created in _ensure_provider.
        self._provider_cache: Dict[str, 'ModelProviderPlugin'] = {}
        self._active_provider_name: Optional[str] = None
        # Persistent provider base config (plugin_configs + skip_model_test) for
        # V2 cross-provider tier switches — set in configure(), read by
        # _provider_for_tier.  Distinct from _provider_lazy_pending (which is
        # cleared once the main provider is created).
        self._tier_provider_base: Optional[Dict[str, Any]] = None
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
        # The system instruction EXACTLY as it stood at the end of
        # ``configure()`` — after assembly (or override), after
        # ``{{!py:...}}`` prefetch expansion, and BEFORE any of the
        # runtime mutations that follow (deferred plugin instructions
        # injected when a tool first activates, pinned-reference blocks).
        #
        # This is the artifact a revive restores (issue #787): a session
        # whose persona carries a mandatory prefetch could not be woken at
        # all, because bootstrap re-ran the prefetch with an empty
        # ``agent_params`` and the script aborted session-prep.  Persisting
        # what was rendered removes the re-run instead of working around it.
        #
        # Deliberately the CONFIGURE-TIME value and not the live one: the
        # runtime additions are re-produced by the revived session itself
        # (a tool that activates again re-injects its deferred
        # instructions), so restoring the live value would DOUBLE them,
        # once per revive.
        self._rendered_system_instruction: Optional[str] = None
        # Granular partial-suppression: the canonical frozenset of framework
        # instruction pieces to drop (subset of {disk, constants, security};
        # see ``instruction_suppression``).  Empty = suppress nothing.  Ignored
        # when _system_instruction_override is set (override wins).
        self._suppress_base_instructions: FrozenSet[str] = frozenset()
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
        # Per-provider cache-plugin instances, keyed by the name the
        # provider was REGISTERED under (the same key ``_provider_cache``
        # uses).  A cross-provider tier switch re-wires from here, so a
        # switch back is O(1) and the plugin keeps the cache metrics and
        # prefix state it accumulated for that provider.  NOT keyed on
        # ``provider.name``: zhipuai subclasses anthropic and reports the
        # parent's name, so two tiers would collide on one plugin
        # instance built from the wrong ``plugin_configs`` section.
        self._cache_plugins_by_provider: Dict[str, Any] = {}
        # How many times this session has actually CHANGED tier binding.
        # Every switch re-reads the whole prefix cold at the new model, so
        # this is the multiplier on the cost of tier mode — the number that
        # says whether the tier feature paid for itself.  Counts real
        # changes only: an ``enter_tier`` to the tier already active is a
        # no-op and must not inflate it.  Reported on the LLM span as
        # ``jaato.tier.switches`` and monotonic for the session's life.
        self._tier_switch_count: int = 0
        # Post-connect bookkeeping that FAILED, per subsystem.  Both blocks
        # below are deliberately non-fatal (the provider is already
        # re-pointed by then; raising would leave the switch half-applied),
        # and the cost of that is a real regression recorded only in a log
        # line nobody reads at runtime:
        #
        #   cache re-wire fails      -> the session runs UNCACHED from here
        #   reliability retarget     -> patterns judged against the wrong model
        #
        # Three best-effort blocks is not the smell; three UNOBSERVABLE ones
        # is.  Both counters ride the LLM span alongside ``jaato.tier``, so a
        # consumer can see a degraded session instead of inferring it.
        self._tier_cache_rewire_failures: int = 0
        self._tier_reliability_retarget_failures: int = 0

        # Thinking mode
        self._thinking_plugin: Optional['ThinkingPlugin'] = None

        # Session persistence (plugin + config ownership and the
        # save/restore flow live on this collaborator; the
        # _session_plugin/_session_config properties below delegate to it)
        self._persistence = SessionPersistence(self)

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
        # End-user identity for telemetry user tracking (Langfuse Users view).
        # Set by the daemon from the authenticated client user when available
        # (``set_client_user_id``); otherwise resolved from the per-session
        # ``JAATO_TELEMETRY_USER_ID`` env at turn time. Emitted as the
        # OpenInference ``user.id`` span attribute.
        self._client_user_id: Optional[str] = None
        # Custom attributes stamped on every LLM (generation) telemetry span.
        # A vendor-neutral hook for external code — prefetch scripts, plugins —
        # to correlate generations with build-time context. Canonical use:
        # prompt-management linking (a prefetch that resolves a managed prompt
        # sets the backend's prompt-link keys here, e.g. Langfuse's
        # ``langfuse.observation.prompt.name`` / ``.version``). Merged in
        # :meth:`_build_llm_span_attributes`; see :meth:`set_llm_span_attributes`.
        self._llm_span_attributes: Dict[str, Any] = {}

        # Retry notification callback (client-configurable)
        self._on_retry: Optional[RetryCallback] = None

        # Request pacing (proactive rate limiting)
        # Reads AI_REQUEST_INTERVAL from env (default: 0 = disabled)
        self._pacer = RequestPacer()

        # Cancellation support
        self._cancel_token: Optional[CancelToken] = None
        self._parent_cancel_token: Optional[CancelToken] = None  # For parent→child propagation
        self._is_running: bool = False
        #: Guards the ONE question a delivering caller must not get wrong:
        #: "will a turn drain what I enqueue?"  Held across BOTH the
        #: check-and-enqueue in :meth:`offer_message` and the
        #: ``_is_running = False`` flip at the end of ``_run_chat_loop``, so
        #: the two cannot interleave.  See :meth:`offer_message` for why a
        #: lock rather than a timing argument.
        #:
        #: Held for microseconds, with NO callbacks invoked inside it -- the
        #: drain's ``_on_prompt_injected`` / ``_on_continuation_needed`` fire
        #: outside, so this lock has no re-entrancy surface.
        self._delivery_lock = threading.Lock()
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
        # GC LIFECYCLE observer: (phase, payload) for about_to_run /
        # started / completed.  Distinct from _gc_threshold_callback
        # above, which carries only the threshold crossing and whose
        # daemon handler renders it as PROSE for humans.
        self._gc_phase_callback: Optional[Any] = None

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
        # Per-plugin tool allow-lists (profile ``tools:[...]`` modifier).
        # Maps plugin name → list of allowed tool names.  A plugin absent
        # from this dict exposes all its tools.  Enforced per-session in
        # ``_apply_tool_scopes`` (mirroring the ``_tool_plugins``
        # plugin-level filter) so a tool outside its plugin's allow-list
        # never reaches the wire body or the provider's grammar surface.
        # Never mutates the shared registry — sibling subagents on the
        # same runtime keep their own scopes.
        self._tool_scopes: Dict[str, List[str]] = {}

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
        # When set, budget notices are ALSO collected here.  A rung pushed
        # from the cascade can land BETWEEN turns, and every client-facing
        # output channel this session has is turn-scoped — so the notice
        # must be handed back to the caller (the daemon) to emit instead.
        self._budget_notice_sink: Optional[List[str]] = None
        # Highest rung threshold already applied to THIS session, from any
        # source.  Latching lives on each BudgetTracker, so two trackers
        # watching the SAME ladder at different rates each latch
        # independently — and a lower rung applied later silently reverses a
        # higher one, because overlays are last-writer-wins per tier.  This
        # makes "at most once, in order" a property of the LADDER rather
        # than of whichever tracker noticed.
        self._budget_applied_rung_pct: float = 0.0

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

        # Truncation-recovery state (#749).  Counts the continuations
        # spent on the CURRENT turn after an output-cap truncation;
        # reset at the top of every turn, capped at
        # ``TRUNCATION_RECOVERY_BUDGET``.  Per-turn rather than
        # per-operation (the rewind counter's scope) because the thing
        # being bounded is a turn that keeps ending the same way.
        self._truncation_recovery_count: int = 0

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
        _user_id = self._resolve_telemetry_user_id()
        if _user_id:
            extra_attrs["user.id"] = _user_id

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
        """Set the daemon session manager ID for this session.

        This ID (e.g. ``"20260328_204308"``) identifies the daemon-side
        session-manager session and is the per-session source of truth
        for any consumer that needs the session id at execution time:
        emitted as the ``jaato.session_id`` telemetry span attribute,
        resolved by dynamic-instructions ``{{session_id}}``, and read by
        the ``memory`` plugin for the ``source_session`` provenance field.

        **Set on BOTH tiers.**  Daemon-side it is wired by ``JaatoClient``;
        runner-side it is stamped from ``envelope.session_id`` during
        ``bootstrap_session`` (``runner/session.py``).  The runner-side
        stamp is load-bearing: each sibling subagent has its own
        JaatoSession, so reading this per-session value (via
        ``get_current_session()``) is per-sibling-correct, whereas the
        previous fallback to the SHARED ``registry._session_id`` leaked
        the last-bootstrapped sibling's id across siblings.

        Args:
            session_id: The session manager's session ID.
        """
        self._daemon_session_id = session_id

    def set_client_user_id(self, user_id: Optional[str]) -> None:
        """Set the end-user identity for telemetry user tracking.

        The daemon wires this from the authenticated client user
        (``get_client_user(client_id)`` — WS/SSO deployments; IPC has no
        user). It is emitted as the OpenInference ``user.id`` span
        attribute so observability backends (Langfuse's Users view)
        attribute traces, token usage, and cost to the user.

        Takes precedence over the ``JAATO_TELEMETRY_USER_ID`` per-session
        env fallback used by keyless/local deployments.

        Args:
            user_id: End-user identifier (username, email, or subject
                claim), or ``None`` to leave it unset.
        """
        self._client_user_id = user_id

    def set_llm_span_attributes(
        self, attributes: Dict[str, Any], *, merge: bool = True,
    ) -> None:
        """Attach custom attributes to every LLM (generation) telemetry span.

        A vendor-neutral extension hook: external code that runs with a handle
        on the session — a ``{{!py:}}`` prefetch script (via
        ``context.session``), a plugin — can stamp attributes onto the
        generation spans of this session so observability backends correlate
        each LLM call with build-time context.

        The canonical use is **prompt-management → trace linking**: a prefetch
        that resolves a managed prompt records the backend's prompt-link keys,
        e.g. for Langfuse::

            context.session.set_llm_span_attributes({
                "langfuse.observation.prompt.name": name,
                "langfuse.observation.prompt.version": version,
            })

        so Langfuse links the generation to that prompt version and reports
        per-version performance. Core stays vendor-neutral — the keys are
        chosen by the caller.

        Values must be OTLP-compatible (str / bool / int / float, or sequences
        thereof). Applies to LLM spans opened after this call within the
        session.

        Args:
            attributes: Attribute key/value pairs to stamp on LLM spans.
            merge: When True (default) merge into any previously-set
                attributes; when False replace them (pass ``{}`` to clear).
        """
        if merge:
            self._llm_span_attributes.update(attributes or {})
        else:
            self._llm_span_attributes = dict(attributes or {})

    def _resolve_telemetry_user_id(self) -> Optional[str]:
        """Resolve the end-user id for telemetry ``user.id`` on this turn.

        Precedence: explicit ``set_client_user_id`` value (daemon-wired
        authenticated user) → ``JAATO_TELEMETRY_USER_ID`` per-session env
        (via :meth:`get_session_env`, so it honors the workspace ``.env`` /
        profile env) → ``None``. Never raises — telemetry must not break a
        turn.
        """
        try:
            if self._client_user_id:
                return self._client_user_id
            return self.get_session_env("JAATO_TELEMETRY_USER_ID")  # env: end-user id for telemetry user tracking (Langfuse Users)
        except Exception:
            return None

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

    def offer_message(
        self,
        text: str,
        source_id: Optional[str] = None,
        source_type: Optional[SourceType] = None,
        require_idle: bool = False,
    ) -> str:
        """Atomically enqueue this message, or report that a turn is needed.

        **This session is the authority on whether it is mid-turn**, and this
        is the only method that answers with that authority.  Callers on the
        far side of the RPC (the daemon) hold a REPLICA of that state which
        clears later than this one -- the daemon's ``_model_running`` stays
        True until ``session.send_message`` returns and the daemon's model
        thread unwinds, which is strictly after this session finished its
        turn.  A delivery decided on the replica is therefore decided on a
        state that can already be stale, and a message queued into a turn
        that has ended is never drained by anything.

        Returns:
            ``"queued"`` -- a turn is running and its end-of-turn drain WILL
            collect this message.  Not a guess: the enqueue happened while
            ``_is_running`` was True and could not have been overtaken by the
            flip to False, because both hold ``_delivery_lock``.  The final
            drain runs after that flip, so it necessarily sees this message.

            ``"busy"`` -- ONLY when *require_idle* is set: a turn is
            running and the caller asked not to add to the queue in that
            case.  Nothing was enqueued.

            ``"needs_turn"`` -- no turn is running, so nothing would ever
            drain this.  The message is deliberately NOT enqueued; the caller
            must start a turn with it.  A session cannot start its own turn
            (``inject_prompt``'s continuation callback exists only for the
            duration of a ``session.send_message`` RPC), so the decision is
            made here and the turn is started by whoever can.

        WHY A LOCK AND NOT A TIMING ARGUMENT.  The unlocked version reads
        ``_is_running``, and between that read and the enqueue the turn can
        end and run its final drain -- leaving the message queued behind a
        drain that has already happened.  That window is small and the
        failure is invisible: the caller is told "queued", which is what a
        healthy delivery also says.  Holding the lock across the check and
        the enqueue, and across the flip, removes the window rather than
        making it narrower.
        """
        actual_source_id = source_id or "unknown"
        actual_source_type = source_type or SourceType.USER

        with self._delivery_lock:
            if self._is_running:
                if require_idle:
                    # Backpressure probe: the caller has decided this peer is
                    # too far behind to take another queued message, and asks
                    # to deliver ONLY if a turn would start.  Answered here
                    # rather than from a daemon-side replica so a peer that
                    # went idle is not refused for a backlog it has drained.
                    return "busy"
                self._message_queue.put(
                    text, actual_source_id, actual_source_type,
                )
                queued = True
            else:
                queued = False

        # Callbacks OUTSIDE the lock -- see ``_delivery_lock``'s note on
        # keeping it free of re-entrancy.
        if queued:
            self._trace(
                f"OFFER_MESSAGE: queued for a running turn, agent_id="
                f"{self._agent_id}, source_type={actual_source_type.value}, "
                f"queue_size={len(self._message_queue)}"
            )
            if self._on_prompt_injected:
                self._on_prompt_injected(text)
            return "queued"

        self._trace(
            f"OFFER_MESSAGE: no turn running, agent_id={self._agent_id}, "
            f"source_type={actual_source_type.value} -- caller must drive"
        )
        return "needs_turn"

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

    def try_drain_pending_user(self) -> Optional[str]:
        """Atomically pop the first pending high-priority (USER/PARENT/SYSTEM)
        message for the daemon's post-turn drain.

        Multi-turn deadlock fix: a client send that races into the turn
        wind-down — ``turn.completed`` reaches the client (which sends the
        next turn) *before* the daemon clears ``_model_running`` — is
        forwarded by the daemon gate as an :meth:`inject_prompt`.  Finding
        the session idle with no active turn (and the per-RPC continuation
        callback already restored to ``None``), ``inject_prompt`` takes the
        else/queue branch, so the message sits with no drainer and the turn
        never runs.  The daemon's model-thread ``finally`` calls this after
        every runner-tier turn (via the ``session.try_drain_pending_user``
        RPC); when it returns text the daemon starts a fresh turn with it.

        Guarded on ``not _is_running`` so it never steals a message from an
        active turn (which drains the queue itself mid-turn) — in that case
        the running turn owns the message and this returns ``None``.

        Returns:
            The message text to run as the next turn, or ``None`` when no
            high-priority message is queued or a turn is already running.
        """
        if self._is_running:
            return None
        if not self._message_queue.has_parent_messages():
            return None
        msg = self._message_queue.pop_first_parent_message()
        if msg is None:
            return None
        if self._on_prompt_injected:
            self._on_prompt_injected(msg.text)
        return msg.text

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
                    # CLIENT-audience chunks are for viewers only and must
                    # not enter the conversation.
                    if chunk.audience.reaches_model() and chunk.content:
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

        # Drained BY TIER SET, not by named accessor.  ``message_queue``
        # already declares membership once (HIGH_PRIORITY_SOURCES /
        # IDLE_ONLY_SOURCES); enumerating tiers here as well meant adding a
        # source type required remembering to edit this function, and
        # forgetting was a SILENT DISCARD — the message sat in a tier nobody
        # popped and died with the session.  That is exactly what happened to
        # SIBLING: the tier and its accessors shipped with the addressing
        # work, and no drainer was ever wired.
        #
        # Order is an authority statement: high-priority first (a human or a
        # parent may steer), then idle-only (a child reports, a sibling
        # coordinates) — neither of which may interrupt work in progress.
        # DRAIN UNTIL THE QUEUE IS EMPTY, not once.
        #
        # A single pass leaves a hole for a message that arrives WHILE the
        # pass is running: the sender sees the target still busy (the daemon
        # clears ``_model_running`` only after this RPC unwinds), so it takes
        # the QUEUE branch -- onto a tier whose drainer has already gone past.
        # Nothing pops it afterwards, because ``_on_continuation_needed`` is
        # restored to None when the RPC returns.  The message is accepted and
        # then stranded, permanently.
        #
        # Reported by the perpetual-monologue cascade, where it is not a rare
        # race but the STEADY STATE: in a symmetric two-sibling loop the
        # sender's post-send narration keeps it "busy" across exactly the
        # interval when the fast half replies.  The tighter the loop, the more
        # reliably it strands.  A request/response cascade never sees it,
        # because the target's next turn drains it.
        #
        # The re-check is what ends the turn: every pass happens while the
        # continuation callback is still installed, so a late message either
        # lands before a check (drained here) or after the busy flag clears
        # (target reads IDLE -> the sender DRIVES it).
        #
        # Bounded: each pass POPS, so it can only spin against a producer
        # that never stops -- and an unbounded loop in turn teardown would
        # hang the turn rather than lose a message.  The cap is generous and
        # loud, never silent.
        # DIAGNOSTIC — the drain ITSELF, not its per-message lines.
        #
        # ``DRAIN_<TIER>_MESSAGE`` is emitted PER DRAINED MESSAGE, so silence
        # means BOTH "the drain never ran" and "it ran and saw an empty
        # queue".  Those are different bugs and no existing log separates
        # them: absent and empty, in the instrument.
        #
        # ``queue_at_entry`` is the discriminator.  A send that was told
        # "busy" put a message on this session's queue, so if the drain
        # enters with 0 the message is not where the sender thinks it is --
        # a different failure from a drain that never ran at all.
        #
        # Daemon log, deliberately: the per-message lines go to the provider
        # trace, which is a silent no-op unless JAATO_PROVIDER_TRACE is set,
        # and a diagnostic that needs an env var gets read after the run it
        # was needed for.  Greppable token: DRAIN_SUMMARY.
        _queue_at_entry = len(self._message_queue)
        _MAX_DRAIN_PASSES = 100
        for _pass in range(_MAX_DRAIN_PASSES):
            for tier_label, tier in (
                ("PRIORITY", HIGH_PRIORITY_SOURCES),
                ("IDLE_ONLY", IDLE_ONLY_SOURCES),
            ):
                while True:
                    msg = self._message_queue.pop_first_matching(
                        lambda _m: True, source_types=set(tier),
                    )
                    if msg is None:
                        break

                    drained_count += 1
                    collected_messages.append(msg.text)
                    self._trace(
                        f"DRAIN_{tier_label}_MESSAGE: agent_id={self._agent_id}, "
                        f"source_type={msg.source_type.value}, "
                        f"source_id={msg.source_id}, text={msg.text[:100]}..."
                    )

                    # Log the message for tracing (UI visibility)
                    if self._on_prompt_injected:
                        self._on_prompt_injected(msg.text)

            # Anything that arrived during the pass above goes round again.
            if len(self._message_queue) == 0:
                break
        else:
            logger.warning(
                "DRAIN_MESSAGES: agent_id=%s still had %d queued after %d "
                "passes — a producer is outpacing the drain; leaving the "
                "remainder for the next turn rather than spinning",
                self._agent_id, len(self._message_queue), _MAX_DRAIN_PASSES,
            )

        logger.info(
            "DRAIN_SUMMARY: agent_id=%s queue_at_entry=%d drained=%d "
            "passes=%d queue_at_exit=%d",
            self._agent_id, _queue_at_entry, drained_count, _pass + 1,
            len(self._message_queue),
        )

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

    def get_rendered_system_instruction(self) -> Optional[str]:
        """Return the system instruction as it stood at the end of
        ``configure()`` — the SNAPSHOT, not the live value.

        Differs from :meth:`get_system_instruction` in exactly one way:
        that one returns the live attribute, which keeps growing after
        configure (a plugin's deferred instructions are appended when one
        of its tools first activates; a pinned reference appends its
        content).  This one returns the frozen render.

        That distinction is the whole point of the accessor.  The daemon
        persists this value so a revive can restore the prompt instead of
        re-deriving it (issue #787), and re-deriving is what re-ran a
        mandatory ``{{!py:...}}`` prefetch with an empty ``agent_params``
        and made the session unwakeable.  Persisting the LIVE value
        instead would restore the runtime additions too — and the revived
        session re-produces them, so each revive would duplicate them.

        Returns:
            The rendered prompt, or ``None`` if ``configure()`` has not
            run yet (nothing has been rendered to snapshot).
        """
        return self._rendered_system_instruction

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
            # PROBE (cancel-leak prod-vs-isolation diagnostic):
            # Log cancel_token id() at the source of cancel() — pairs
            # with the VLLM_COMPLETE_ENTRY_CT logger.info trace in the
            # vLLM provider so we can correlate "session cancelled its
            # token T_session" with "provider's for-loop saw cancel on
            # token T_provider".  Same id() = H2 (instance mismatch)
            # ruled out.  Different id() = found the bug.
            #
            # Routed via logger.info (NOT self._trace) because
            # self._trace writes land in /tmp/provider_trace.log which
            # apparmor SILENTLY DENIES under the per-WS confined-runner
            # profile.  Empirically confirmed 2026-06-09 by peer's
            # first probe run: only logger.info traces landed.
            logger.info(
                "REQUEST_STOP_CT_CANCEL id=%s reason=%r",
                id(self._cancel_token),
                reason,
            )
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
        plugins: Optional[List[str]] = None,
        system_instructions: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        skip_provider: bool = False,
        preloaded_plugins: Optional[set] = None,
        skip_model_test: bool = False,
        system_instruction_override: Optional[str] = None,
        suppress_base_instructions: Any = False,
        workspace_path: Optional[str] = None,
        completion_payload_schema: Optional[Any] = None,
        tier_config: Optional['ModelTierConfig'] = None,
        budget_control: Optional['BudgetControlConfig'] = None,
        agent_params: Optional[Dict[str, Any]] = None,
        completion_processors: Optional[List[Any]] = None,
        tool_scopes: Optional[Dict[str, List[str]]] = None,
        tools: Optional[List[str]] = None,  # DEPRECATED alias for ``plugins``
    ) -> None:
        """Configure the session with plugins and instructions.

        Args:
            plugins: Optional list of plugin names to expose (e.g. ``"cli"``,
                   ``"web_search"``). If None, uses all exposed plugins from the
                   runtime's registry. (This is plugin names, NOT tool names —
                   per-tool allow-lists live in ``tool_scopes``.)
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
            tools: DEPRECATED alias for ``plugins`` (it always took plugin
                names, never tool names). Pass ``plugins=`` instead; ``tools=``
                still works with a one-time deprecation warning. ``plugins``
                wins if both are given.
        """
        # Back-compat: ``tools`` was a misleading name for the plugin-name list
        # (it never took tool names).  Honour it as a deprecated alias for
        # ``plugins`` and warn once.  The body below uses ``plugins``.
        if tools is not None:
            import warnings
            warnings.warn(
                "JaatoSession.configure(tools=...) is a deprecated alias for "
                "plugins=; it takes PLUGIN names (e.g. 'cli', 'web_search'), "
                "not tool names. Use plugins= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if plugins is None:
                plugins = tools

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

        # Profile-declared completion processors.  See
        # ``_completion_processors`` doc in __init__ for the contract
        # and ``shared/completion_processors.py`` for the loader +
        # invocation pipeline.
        if completion_processors is not None:
            self._completion_processors = list(completion_processors)

        # Tier mode: when a tier_config is supplied, the session's
        # initial model is overridden by the initial tier's model so the
        # provider connects to the right model from turn 0.  The active
        # tier is set to the config's initial_tier.  When None, the
        # session stays in single-model mode (legacy behaviour).
        # Budget control: build the tracker once per session.  Kept next to
        # the tier wiring because a degrade rung REBINDS the tier table this
        # block just installed.
        if budget_control is not None:
            from .budget_control import BudgetTracker
            self._budget_tracker = BudgetTracker(budget_control)
            logger.info(
                "Budget control active: limits=%s, %d degrade rung(s)",
                dict(budget_control.limits), len(budget_control.degrade),
            )

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
        # Store per-plugin tool allow-lists (profile ``tools:[...]``).
        self._tool_scopes = dict(tool_scopes) if tool_scopes else {}
        # Store tool plugin names
        self._tool_plugins = plugins

        # Re-initialize plugins with session-specific configs if provided
        if plugin_configs and self._runtime.registry:
            for plugin_name, config in plugin_configs.items():
                if plugins is None or plugin_name in plugins:
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
        # Persist the provider base config (plugin_configs + skip_model_test) for
        # V2 cross-provider tier switches.  _provider_lazy_pending is CLEARED once
        # the main provider is created (_ensure_provider), but _provider_for_tier
        # still needs these to build a tier's provider — without this it received
        # plugin_configs=None and built the tier provider with no api_key at all
        # (#354 cross-provider tier bug: "No <provider> API key found").
        self._tier_provider_base = {
            'skip_model_test': skip_model_test,
            'plugin_configs': plugin_configs,
        }

        # Create executor
        self._executor = ToolExecutor(ledger=self._runtime.ledger)

        # Get tool schemas and executors from runtime
        self._tools = self._runtime.get_tool_schemas(plugins, preloaded_plugins=self._preloaded_plugins)
        executors = self._runtime.get_executors(plugins)

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
                plugins, preloaded_plugins=self._preloaded_plugins
            )
            for schema in refreshed_schemas:
                if schema.name not in existing_names:
                    self._tools.append(schema)
                    existing_names.add(schema.name)
            for name, fn in self._runtime.get_executors(plugins).items():
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
                # Framework-reserved: lifecycle terminals (signal_completion —
                # session-level, NOT a registry core tool) PLUS the registry's
                # core tools (stream / event_bus / introspection / client host
                # tools) are exempt from a business catch-all "default"
                # evaluator, so a locked-down default-deny agent can still
                # complete.  Re-populated here every configure() so it survives
                # PermissionPlugin.shutdown() nulling _registry between sessions.
                #
                # Source is ``get_registered_core_tool_names()`` (the
                # ``register_core_tool`` set, == ``is_core_tool``) — NOT
                # ``get_core_tool_schemas()``, which also returns exposed
                # *plugin* tools flagged ``discoverability='core'`` (readFile,
                # the todo tools, ...).  Using the broad schema set would
                # silently exempt those powerful business tools from a
                # catch-all default evaluator — the exact "never
                # cli/file_edit/business tools" case #487/#488 prohibit.
                reserved = set(exposed_lifecycle_names)
                if self._runtime.registry:
                    reserved.update(
                        self._runtime.registry.get_registered_core_tool_names()
                    )
                self._runtime.permission_plugin.add_framework_reserved_tools(
                    reserved
                )

            # Apply per-plugin tool allow-lists (profile ``tools:[...]``)
            # to the assembled ``self._tools`` so scoped-out tools are
            # absent from discovery / token-budget accounting too — not
            # just the wire body (``_get_tools_for_provider`` is the
            # final guard).  Lifecycle / core-infra tools have no owning
            # plugin and always survive.  No-op without configured
            # scopes.
            if self._tool_scopes and self._tools:
                before = len(self._tools)
                self._tools = self._apply_tool_scopes(self._tools)
                dropped = before - len(self._tools)
                if dropped:
                    self._trace(
                        f"configure: tool_scopes dropped {dropped} "
                        f"tool(s) from the initial surface "
                        f"(scopes={self._tool_scopes})"
                    )

            # Gate introspection on the presence of something to discover.
            # Its [core] tools are always in the initial schema, but they are
            # dead weight (and invite a wasted discovery turn) when EVERY profile
            # tool is already eager — core, or its plugin is (preload)-ed.  The
            # deferred set is read from the LIVE registry, so a dynamic plugin
            # (mcp) that already surfaced discoverable tools keeps introspection
            # automatically — no hardcoded plugin list.  Conservative: only act
            # with an explicit profile plugin filter (an unfiltered session
            # exposes everything and may legitimately need discovery).
            if self._tool_plugins is not None and self._tools:
                reg = self._runtime.registry
                deferred = _has_deferred_to_discover(
                        reg.get_exposed_tool_schemas(), self._tool_plugins,
                        self._preloaded_plugins, self._tool_scopes,
                        lambda n: getattr(reg.get_plugin_for_tool(n), "name", None))
                if _should_drop_introspection(
                        deferred, [t.name for t in self._tools]):
                    # Empty wire -> drop introspection's tools AND flag the
                    # session so the introspection plugin suppresses its now-
                    # mismatched discovery GUIDANCE (read by
                    # introspection.get_system_instructions). See
                    # _should_drop_introspection for the full rationale (GC
                    # re-inspection + tool/instruction-gate alignment).
                    self._introspection_guidance_suppressed = True
                    n0 = len(self._tools)
                    self._tools = [t for t in self._tools
                                   if t.name not in _INTROSPECTION_TOOL_NAMES]
                    if len(self._tools) != n0:
                        self._trace(
                            "configure: dropped introspection — empty wire (no "
                            "deferred tools to discover, no eager tools to "
                            "re-inspect)")

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
        # ``suppress_base_instructions`` is normalized to the canonical
        # frozenset of pieces to drop (accepts bool / dict / list; see
        # ``instruction_suppression``).
        self._system_instruction_override = system_instruction_override
        self._suppress_base_instructions = normalize_suppression(
            suppress_base_instructions
        )
        _suppress = self._suppress_base_instructions

        # Build system instructions.
        #
        # Full-override path: skip assembly entirely — no disk I/O, no
        # enrichment churn.  Plugin state was already initialised earlier
        # in configure_tools(), so tool functionality is intact; only the
        # would-be-discarded enrichment text is skipped.
        #
        # Otherwise: assemble normally.  Each ``include_*`` gate drops one
        # framework-originated layer when its piece is in the suppression
        # set — ``disk`` (the .jaato/instructions baseline), ``constants``
        # (task-completion / parallel / turn-summary), ``security`` (the
        # untrusted-content boundary) — while the agent prompt and plugin
        # instructions always remain.  The partial-suppression path for
        # small-context models.  Base is lazy-loaded on first use, so
        # sessions that always suppress it never touch the disk.
        if system_instruction_override is not None:
            self._system_instruction = system_instruction_override
        else:
            self._system_instruction = self._runtime.get_system_instructions(
                plugin_names=plugins,
                additional=system_instructions,
                presentation_context=self._presentation_context,
                include_base=PIECE_DISK not in _suppress,
                include_constants=PIECE_CONSTANTS not in _suppress,
                include_security=PIECE_SECURITY not in _suppress,
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

        # Freeze what was rendered.  Everything that mutates
        # ``_system_instruction`` after this point is a RUNTIME addition
        # the revived session re-produces for itself, so the snapshot is
        # taken here rather than at save time.  See the field's docstring
        # in ``__init__`` and issue #787.
        self._rendered_system_instruction = self._system_instruction

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

        # NOTE: share_context (the subagent→parent push tool) was
        # previously registered here unconditionally as a session
        # built-in.  Extracted 2026-06-07 to a dedicated non-core
        # plugin ``shared/plugins/telepathy``.  Subagent profiles
        # that need to share context list ``telepathy`` in their
        # ``plugins:`` field; root sessions don't, so the tool
        # never appears on their surface.  The plugin also
        # implements ``is_tool_visible`` (per PR #241) for
        # belt-and-braces against the edge case where a profile
        # lists telepathy but the session has no parent.

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

        # Count the INITIAL wire tool-schema array (preloaded / core /
        # lifecycle tools that bypass the discovery-only schema counting
        # in ``_track_activated_tools_in_budget``).  Without this the
        # budget-GC denominator under-reports the true wire by the size
        # of the static tool array, so the GC threshold is structurally
        # unreachable and the session walks into a context-window 400
        # with GC idle.  See the method docstring for the full evidence.
        self._register_initial_wire_tool_schema_budget()

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
                # This session's OWN id, not the registry's — see the
                # note in create_provider.  Stamped by the runner during
                # bootstrap_session, so it is already set by the time the
                # provider is lazily created on the first turn.
                session_id=self._daemon_session_id,
            )
            # V2 cross-provider tiers: record which provider this instance IS and
            # seed the per-provider cache so switch_tier can compare against it
            # and reuse it on a switch back.
            self._active_provider_name = cfg['provider_name']
            if cfg['provider_name'] is not None:
                self._provider_cache[cfg['provider_name']] = self._provider
            # Propagate agent context to provider for trace identification.
            if hasattr(self._provider, 'set_agent_context'):
                self._provider.set_agent_context(
                    agent_type=self._agent_type,
                    agent_name=self._agent_name,
                    agent_id=self._agent_id,
                )
            # Resolve the real context window now that the provider exists.
            # The budget was created at configure() time with context_limit=0
            # because the provider is lazy-created and didn't exist yet.  This
            # is the single point where the model's actual limit (e.g. vLLM's
            # configured context_length) becomes the budget-GC denominator —
            # it runs on first model use, before any conversation grows or any
            # after-turn GC check, so the GC threshold is computed against the
            # true window from the very first turn.  See
            # _populate_instruction_budget for the failure mode this closes.
            if self._instruction_budget is not None:
                self._instruction_budget.context_limit = (
                    self._provider.get_context_limit()
                )
            # Wire cache plugin now that the provider exists.  Pre-defer
            # this fired at the end of configure() unconditionally.
            self._wire_cache_plugin()
            # Fail loud if a tier declaring a modality role (``vision``
            # implies image; any tier may declare one explicitly) maps to a
            # model the provider can't confirm accepts that input — runs
            # once here, alongside the context-window resolution above,
            # before any model work (the earliest point the provider
            # exists).
            self._validate_modality_tier_capabilities()
            # A session that STARTS in a speaking tier must ask for audio
            # too.  The initial tier never passes through
            # ``_connect_tier_entry`` (that is the SWITCH path), so without
            # this an outbound role only took effect after the first
            # ``enter_tier`` — the one case most likely to be tested first.
            self._request_active_tier_output_modalities()
            # Consume the stashed args — repeated calls become no-ops
            # (the fast-path above returns the cached provider).
            self._provider_lazy_pending = None
            return self._provider

    def _cache_plugin_config(self) -> Dict[str, Any]:
        """The config dict handed to this session's cache plugin.

        Reproduces the ``ProviderConfig.extra`` the ACTIVE provider was
        built with, so a cache knob authored in a profile reaches the
        cache plugin by the same route every other provider knob takes.

        The merge is NOT re-implemented here.  It is
        ``jaato_runtime.resolve_provider_extra`` — the same function
        ``create_provider`` uses to build the ``ProviderConfig`` the
        provider itself was initialized with — so the cache plugin and
        the provider it caches for cannot be configured differently.

        Why this is a second CALL and not a second read of a stored
        result: ``plugin_configs`` is a per-session argument, while
        ``runtime._provider_configs`` is runtime-level and shared by
        every session on that provider.  Storing the merged config back
        there would leak one session's profile knobs into all the others.
        The recomputation is forced by that scoping, and the defence
        against drift is that there is exactly one implementation of it.

        What was missing before: ``runtime._provider_config`` is assigned
        exactly once — ``ProviderConfig(project=..., location=...)`` in
        ``JaatoRuntime.connect`` — with an empty ``extra`` that nothing
        subsequently writes to, and ``create_provider``'s merge goes into
        a local copy that is never stored back.  Reading that base alone
        handed every cache plugin ``{}``, so ``enable_caching`` was
        silently ignored wherever it was written: Anthropic caching was
        reachable only through the ``JAATO_ANTHROPIC_ENABLE_CACHING`` env
        default inside ``initialize()``, and Google's explicit
        ``CachedContent`` path (a hard ``False`` default, no env
        fallback) was unreachable altogether.

        The profile lookup is keyed on ``_active_provider_name`` — the
        name the provider was CREATED under, which is the key the
        profile's ``plugin_configs`` uses.  ``provider.name`` is not
        interchangeable with it (zhipuai subclasses anthropic and reports
        the parent's name).

        Returns:
            A fresh dict; callers may mutate it freely.  ``api_key`` is
            absent, because ``resolve_provider_extra`` promotes it to the
            ``ProviderConfig.api_key`` field and it never reaches
            ``extra`` on the provider's side either.
        """
        from .jaato_runtime import resolve_provider_extra

        base_extra: Dict[str, Any] = {}
        if self._runtime and getattr(self._runtime, '_provider_config', None):
            base_extra = self._runtime._provider_config.extra

        pending = (getattr(self, '_tier_provider_base', None)
                   or getattr(self, '_provider_lazy_pending', None)
                   or {})
        profile_key = self._active_provider_name or getattr(
            self._provider, 'name', None)
        config, _promoted_api_key = resolve_provider_extra(
            base_extra, pending.get('plugin_configs'), profile_key)
        return config

    def _wire_cache_plugin(self) -> None:
        """Attach the cache plugin matching the CURRENTLY ACTIVE provider.

        The cache plugin is selected by matching the provider's ``name``
        property against available cache plugins' ``provider_name``.
        When found:
        - The plugin is initialized with the config from
          :meth:`_cache_plugin_config` (runtime extras + this session's
          ``plugin_configs[<provider>]`` profile knobs)
        - The plugin is told the active model, so any model-dependent
          policy (Anthropic's minimum-cacheable-size threshold, Google's
          ``CachedContent`` model binding) tracks a tier switch
        - The current InstructionBudget is set on the plugin
        - The plugin is attached to the provider via ``set_cache_plugin()``

        This is a Variant A integration (provider delegates to plugin).

        **Idempotent and re-runnable.**  Called from
        :meth:`_ensure_provider` when the provider first materializes,
        and again from :meth:`_connect_tier_entry` on every tier switch
        — model-driven (``enter_tier``) or framework-driven (a
        budget-control degrade rung rebinding the active tier).  Before
        it was re-runnable, a cross-provider tier ran with NO cache
        plugin attached (caching silently off for the rest of the
        session) and a same-provider switch left the plugin's model name
        pinned to whatever booted the session.  See
        ``docs/design/model-tier-prompt-cache.md`` §5.2.

        Plugin instances are cached per provider in
        :attr:`_cache_plugins_by_provider`, so switching back to a tier
        is O(1) and the plugin keeps the metrics and prefix-invalidation
        state it accumulated for that provider.  Discovery
        (``load_cache_plugin_for_provider``) scans entry points, which is
        not something to repeat on every hop.

        A provider with no matching cache plugin — openrouter, which
        caches internally, or any provider that cannot cache — clears
        :attr:`_cache_plugin` rather than leaving the previous
        provider's attached.  That slot is read for budget forwarding,
        usage extraction and telemetry, so a stale one would attribute
        the new provider's cache traffic to the old provider's counters.
        The old plugin stays in the per-provider cache, still attached to
        its own provider instance, ready for a switch back.
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

        model_name = getattr(self._provider, 'model_name', None)
        # Keyed on the REGISTRATION name, not ``provider.name`` — see the
        # attribute's comment for why those differ and why it matters.
        cache_key = self._active_provider_name or provider_name
        cache_plugin = self._cache_plugins_by_provider.get(cache_key)

        if cache_plugin is None:
            config = self._cache_plugin_config()
            # Include model name for threshold selection
            if model_name:
                config['model_name'] = model_name
            cache_plugin = load_cache_plugin_for_provider(provider_name, config)
            if cache_plugin is None:
                self._cache_plugin = None
                return
            self._cache_plugins_by_provider[cache_key] = cache_plugin

        # The active model is wiring, not configuration: a tier switch
        # changes it without changing anything else, and a plugin that
        # missed the change makes model-dependent decisions for the wrong
        # model.  Pushed on every call, including the first, so the
        # re-wire path and the initial path cannot diverge.
        if model_name and hasattr(cache_plugin, 'set_model_name'):
            cache_plugin.set_model_name(model_name)

        # Set the budget so the plugin can make policy-aware decisions
        if self._instruction_budget:
            cache_plugin.set_budget(self._instruction_budget)

        # Attach to provider (Variant A: provider delegates to plugin)
        if hasattr(self._provider, 'set_cache_plugin'):
            self._provider.set_cache_plugin(cache_plugin)

        self._cache_plugin = cache_plugin
        self._trace(
            f"CACHE_PLUGIN: Attached {cache_plugin.name} for provider "
            f"{provider_name} (model {model_name})"
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

        **Per-turn visibility filter.**  Plugins MAY implement an
        opt-in ``is_tool_visible(tool_name) -> bool`` method to gate
        their own tools on dynamic session state.  This method is
        consulted on EVERY ``provider.complete()`` call, so visibility
        decisions always reflect current state.  Plugins without the
        method are unaffected.  Tools whose owning plugin returns
        ``False`` are stripped from the array sent to the model.

        The canonical use case (server 0.6.x+) is the ``todo`` plugin
        hiding its 8 plan-required tools (``startPlan``,
        ``setStepStatus``, ``getPlanStatus``, ``completePlan``,
        ``addStep``, ``addDependentStep``, ``completeStepWithOutput``,
        ``getBlockedSteps``) when no active plan exists.  Without
        this filter, small models (e.g. Llama 3.1 8B AWQ on vLLM
        2026-06-07 smoke) misroute to ``completeStepWithOutput`` as
        a proxy for ``signal_completion`` because the description
        overlaps and the precondition isn't enforced at the schema
        layer.  Same shape as
        :meth:`LifecycleTools._should_hide_signal_completion` — gate
        at schema-export time rather than waiting for a clean
        runtime-rejection error from the plugin.

        A plugin's ``is_tool_visible`` predicate is asked about
        EVERY tool name in the session (not just the plugin's own
        tools); a sensible default is ``return True`` for unknown
        names so other plugins' tools flow through unchanged.

        Returns:
            Tools to pass, or empty list if provider manages its own.
        """
        uses_external = getattr(self._provider, 'uses_external_tools', lambda: True)()
        if not uses_external:
            return []
        if not self._tools:
            return self._tools

        # Per-session tool-scope allow-list filter (profile
        # ``tools:[...]`` modifier).  Applied FIRST, on every return
        # path, so a tool outside its plugin's allow-list never reaches
        # the provider's wire body or grammar surface regardless of how
        # ``self._tools`` was assembled.  This is the definitive
        # wire-absence guarantee — the property profile authors rely on
        # for context-shaving.  A no-op when no scopes are configured.
        scoped = self._apply_tool_scopes(self._tools)

        registry = getattr(self._runtime, 'registry', None)
        if registry is None:
            return scoped

        # Collect plugins that opt in to visibility gating.  Walk
        # ``list_exposed`` (rather than ``_plugins.keys()``) so
        # un-exposed plugins don't get consulted.
        try:
            exposed_names = registry.list_exposed()
        except Exception:
            return scoped
        filters = []
        for name in exposed_names:
            plugin = registry.get_plugin(name)
            if plugin is not None and hasattr(plugin, 'is_tool_visible'):
                filters.append(plugin)
        if not filters:
            return scoped

        visible: List['ToolSchema'] = []
        for tool in scoped:
            hidden = False
            for plugin in filters:
                try:
                    if not plugin.is_tool_visible(tool.name):
                        hidden = True
                        break
                except Exception:
                    # A buggy predicate must not break the turn — log
                    # and treat as visible (fail-open).
                    self._trace(
                        f"is_tool_visible raised for tool={tool.name!r} "
                        f"plugin={plugin.name!r}; treating as visible"
                    )
            if not hidden:
                visible.append(tool)
        return visible

    def _apply_tool_scopes(
        self, schemas: List['ToolSchema']
    ) -> List['ToolSchema']:
        """Filter ``schemas`` to honour per-plugin tool allow-lists.

        For each tool, looks up its owning plugin (via
        ``registry.get_plugin_for_tool``) and, if that plugin has an
        entry in ``self._tool_scopes``, drops the tool unless its name
        is in the allow-list.  Tools with no owning plugin (lifecycle
        tools like ``signal_completion``, core infra like stream /
        event-bus controls) have no scope and always survive.

        Mirrors the ``_tool_plugins`` plugin-level filter in
        :meth:`activate_discovered_tools`, one granularity finer.  Pure
        function over the input list — never mutates the registry or
        ``self._tools``.

        Returns the input unchanged when no scopes are configured (the
        overwhelmingly common case), so the per-turn cost is a single
        dict-emptiness check for unscoped sessions.
        """
        if not self._tool_scopes or not schemas:
            return schemas
        registry = getattr(self._runtime, 'registry', None)
        if registry is None:
            return schemas
        kept: List['ToolSchema'] = []
        for schema in schemas:
            plugin = registry.get_plugin_for_tool(schema.name)
            plugin_name = plugin.name if plugin is not None else None
            allow = (
                self._tool_scopes.get(plugin_name)
                if plugin_name is not None
                else None
            )
            if allow is not None and schema.name not in allow:
                self._trace(
                    f"tool_scope: dropping {schema.name!r} "
                    f"(plugin {plugin_name!r} allow-list={allow})"
                )
                continue
            kept.append(schema)
        return kept

    def _count_tokens(self, text: str) -> int:
        """Count tokens using cache, provider, or estimate (in that order).

        Thin wrapper over
        :func:`instruction_budget_builder.count_tokens` that supplies this
        session's runtime cache, provider, and provider-name resolution.
        """
        provider_name = self._provider_name_override or self._runtime.provider_name
        return _builder_count_tokens(
            text,
            cache=self._runtime.instruction_token_cache,
            provider=self._provider,
            provider_name=provider_name,
            on_trace=self._trace,
        )

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
        # Get context limit from the provider.  The provider is lazy-created
        # (``_ensure_provider``, first model use) to keep bootstrap off the
        # critical path, so at configure() time ``self._provider`` is usually
        # ``None`` — there is no limit to read yet.  In that case the budget
        # is created with ``context_limit = 0`` (an honest "unknown"; the
        # utilization/available helpers are already guarded against 0) and the
        # real limit is filled in by ``_ensure_provider`` the moment the
        # provider materializes, BEFORE any conversation grows or any GC check
        # runs.  We deliberately do NOT keep a hardcoded default (e.g. the old
        # 128_000): that masked a misconfigured/late provider and froze the
        # budget-GC denominator at a model-agnostic number, so the after-turn
        # GC threshold was computed against the wrong window and never tripped
        # — the session walked into a context-window 400 with GC idle.
        context_limit = 0
        if self._provider is not None:
            context_limit = self._provider.get_context_limit()

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
            suppress_base=PIECE_DISK in self._suppress_base_instructions,
            suppress_constants=PIECE_CONSTANTS in self._suppress_base_instructions,
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
        suppress_constants: bool = False,
    ) -> List['_TokenCountRequest']:
        """Collect all instruction texts that need token counting.

        Thin wrapper over
        :func:`instruction_budget_builder.collect_instruction_texts` that
        supplies this session's runtime, pinned references, and preloaded
        plugins, and records the returned deferred-plugin names onto
        ``self._deferred_plugin_instructions`` (the side effect this
        method has always carried). Returns just the request list, as
        before.

        ``suppress_base`` / ``suppress_constants`` drop the disk BASE layer
        and the framework constants respectively, keeping the budget in step
        with the wire prompt's ``include_base`` / ``include_constants`` gates.
        """
        requests, deferred_plugins = _builder_collect_instruction_texts(
            self._runtime,
            session_instructions,
            system_instruction_override=system_instruction_override,
            suppress_base=suppress_base,
            suppress_constants=suppress_constants,
            pinned_references=getattr(self, '_pinned_references', {}),
            preloaded_plugins=getattr(self, '_preloaded_plugins', set()),
        )
        # Record deferred-plugin names only when there are any — matches
        # the original, which touched this attr solely inside the
        # deferral branch (so minimal sessions that never defer a plugin
        # don't require the attribute to exist).
        if deferred_plugins:
            self._deferred_plugin_instructions.update(deferred_plugins)
        return requests

    def _apply_instruction_counts(
        self,
        requests: List['_TokenCountRequest'],
        context_limit: int,
    ) -> None:
        """Build budget children and parent totals from resolved token counts.

        Thin wrapper over
        :func:`instruction_budget_builder.apply_instruction_counts` (which
        mutates this session's budget in place) followed by the
        budget-update emission this method has always performed. Called
        once in Phase 1 (estimates/cached) and again after Phase 2
        completes (accurate counts for previously-estimated entries).
        """
        _builder_apply_instruction_counts(
            self._instruction_budget,
            requests,
            context_limit,
            on_trace=self._trace,
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

    def _assert_payload_fits_context(self) -> None:
        """Pre-flight refuse-send gate.

        Raises ``PayloadExceedsContextError`` when the framework's
        accounted prompt budget (post-GC) plus the provider's
        configured ``max_tokens`` cap would exceed the model's
        context window.

        Call this immediately before a ``provider.complete()`` dispatch,
        AFTER the outgoing prompt is in history and the budget has been
        refreshed (``_update_conversation_budget``) — otherwise the
        gate reads a budget that doesn't yet reflect the prompt about to
        be sent.  The call sites are the two dispatch chokepoints: the
        initial turn dispatch in ``_run_chat_loop`` (after the user
        message is appended) and the intra-turn tool-results dispatch.

        No-ops when:

        - ``self._instruction_budget`` is ``None`` (no budget tracking,
          can't gate)
        - ``self._provider`` is ``None`` (no wire-send pending)
        - ``budget.context_limit`` is ``0`` (honest-unknown limit — the
          framework does not invent a window it wasn't told)

        The gate uses ``budget.total_tokens()`` (post-GC) compared
        against ``budget.context_limit``.  When the provider exposes a
        ``get_max_output_tokens()`` method (vllm, openrouter,
        tensorrt_llm at time of writing), the comparison includes that
        cap; otherwise it fires only when the prompt ALONE exceeds the
        limit.

        Rationale: vLLM 0.22 rejects any request where ``prompt +
        max_tokens > max_model_len`` with the misleading template
        message ``"prompt contains at least N tokens"`` where ``N =
        max_model_len + 1 - max_tokens`` (a deterministic lower-bound
        template, NOT the real prompt size).  This gate refuses the
        doomed dispatch and surfaces a structured error pointing at
        the concrete knobs the operator can turn.
        """
        if not self._instruction_budget:
            return
        if not self._provider:
            return
        limit = self._instruction_budget.context_limit
        if limit <= 0:
            return

        total = self._instruction_budget.total_tokens()

        # Provider-side per-request output cap, if exposed.  Three
        # providers expose this today (vllm / openrouter /
        # tensorrt_llm); others inherit no method and the gate
        # degrades to prompt-only.
        max_tokens: Optional[int] = None
        get_max = getattr(self._provider, 'get_max_output_tokens', None)
        if callable(get_max):
            try:
                value = get_max()
            except Exception:
                value = None
            if isinstance(value, int) and value > 0:
                max_tokens = value

        if max_tokens is not None:
            projected = total + max_tokens
            if projected > limit:
                self._trace(
                    f"REFUSE_SEND: total={total} + max_tokens={max_tokens} "
                    f"= {projected} > context_limit={limit}"
                )
                raise PayloadExceedsContextError(
                    total_tokens=total,
                    max_output_tokens=max_tokens,
                    context_limit=limit,
                )
        else:
            if total > limit:
                self._trace(
                    f"REFUSE_SEND: total={total} > context_limit={limit} "
                    f"(provider exposes no max_tokens; gate on prompt alone)"
                )
                raise PayloadExceedsContextError(
                    total_tokens=total,
                    max_output_tokens=None,
                    context_limit=limit,
                )

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

            # Enforce per-plugin tool allow-list (profile ``tools:[...]``):
            # a scoped-out tool must not be activatable even if the model
            # discovers it via introspection.  One granularity finer than
            # the plugin filter above.
            if self._tool_scopes:
                plugin = self._runtime.registry.get_plugin_for_tool(tool_name)
                if plugin is not None:
                    allow = self._tool_scopes.get(plugin.name)
                    if allow is not None and tool_name not in allow:
                        self._trace(
                            f"activate_discovered_tools: skipping "
                            f"'{tool_name}' (outside plugin "
                            f"'{plugin.name}' allow-list={allow})"
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

    def _register_initial_wire_tool_schema_budget(self) -> None:
        """Count the INITIAL wire tool-schema array into the budget.

        The tool-definitions array (``tools[]``) ships on EVERY request
        and is part of real context utilization — but its tokens were
        only ever counted by :meth:`_track_activated_tools_in_budget`,
        which fires ONLY when a tool is DISCOVERED at runtime (deferred
        loading).  Tools that are PRELOADED or CORE — exposed at
        ``configure()`` and never "discovered": the lifecycle tools
        (``prepare_completion`` / ``signal_completion`` / ...), and any
        plugin carrying a ``(preload)`` modifier — therefore never had
        their schema tokens counted.  ``_collect_instruction_texts``
        counts plugin *instruction* text, not tool *schema* JSON, so it
        doesn't cover them either (and on the
        ``system_instruction_override`` short-circuit it returns before
        the plugin block entirely).

        Net effect of the gap: ``InstructionBudget.total_tokens()`` — the
        denominator the budget-GC trigger reads
        (``get_context_usage`` → ``utilization_percent``) — under-reported
        the true wire by the size of the initial tool array (a cascade
        stage's 16 tools is ~18-25K; ``signal_completion``'s schema alone
        is several KB).  GC measured conversation-only utilisation, sat
        below its threshold, and never reclaimed history while the real
        request walked into a context-window 400.  Empirically:
        session 20260610_082013 read 58% (37928 tok) while the wire that
        overflowed carried 63489 (97%).

        This counts the schema JSON for the tools present in
        ``self._tools`` at configure time (the initial / preloaded /
        core wire set) into a single ``PLUGIN`` child with the CORE tool
        policy (LOCKED).  LOCKED is deliberate: ``total_tokens()`` (the
        GC trigger + the UI budget bar + telemetry) now reflects the
        static wire, while ``gc_eligible_tokens()`` still EXCLUDES it —
        so GC measures true utilisation but only ever trims conversation.

        Disjoint from :meth:`_track_activated_tools_in_budget`: discovered
        tools are appended to ``self._tools`` AFTER this runs (via
        ``activate_discovered_tools``), so the two never double-count the
        same tool.
        """
        if not self._instruction_budget or not self._tools:
            return
        total = 0
        for schema in self._tools:
            try:
                schema_json = json.dumps({
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters,
                }, indent=2)
                total += self._count_tokens(schema_json)
            except Exception:
                continue
        if total <= 0:
            return
        try:
            self._instruction_budget.add_child(
                InstructionSource.PLUGIN,
                "wire_tool_schemas",
                total,
                DEFAULT_TOOL_POLICIES[PluginToolType.CORE],
                label="Wire tool schemas (initial)",
            )
            self._trace(
                f"BUDGET_WIRE_TOOLS: registered {total} tokens for "
                f"{len(self._tools)} initial wire tool schemas (LOCKED)"
            )
            self._emit_instruction_budget_update()
        except Exception as e:
            logger.warning(
                f"Failed to register initial wire tool schema budget: {e}"
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
                provider_name=self._provider_name_override,
                session_id=self._daemon_session_id,
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

    def _parts_from_user_message(
        self, message: str, attachments: List[Dict[str, Any]]
    ) -> List["Part"]:
        """Build a Part list from a user message text + wire attachments.

        Wire attachments are ``{mime_type, data: base64-str, display_name}``
        (the canonical user-message multimodal contract — client-expanded,
        JSON-safe).  ``data`` is base64-decoded to the ``bytes`` that
        ``Part.inline_data`` expects.  The text part precedes the inline-data
        parts; an empty ``message`` (image-only turn) yields parts with no text.
        """
        import base64
        parts: List[Part] = []
        if message:
            parts.append(Part.from_text(message))
        for att in attachments or []:
            data = att.get("data")
            if isinstance(data, str):      # base64 wire form → raw bytes
                data = base64.b64decode(data)
            parts.append(Part(inline_data={
                "mime_type": att.get("mime_type"),
                "data": data,
            }))
        return parts

    def send_message(
        self,
        message: str,
        on_output: Optional[OutputCallback] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_gc_threshold: Optional[GCThresholdCallback] = None,
        on_gc_phase: Optional[Any] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Send a message to the model.

        Args:
            message: The user's message text.
            on_output: Optional callback for real-time output.
                Signature: (source: str, text: str, mode: str) -> None
            on_usage_update: Optional callback for real-time token usage.
                Signature: (usage: TokenUsage) -> None
            on_gc_threshold: Optional callback when GC threshold is crossed.
            on_gc_phase: Optional ``(phase, payload)`` callback for the GC
                LIFECYCLE (about_to_run / started / completed).  Before it
                existed there was no bus signal for GC at all -- clients got
                a prose system message or nothing.
                Signature: (percent_used: float, threshold: float) -> None
            attachments: Optional user-message multimodal attachments, each a
                ``{mime_type, data: base64-str, display_name}`` dict (the
                client-expanded wire contract).  When present, the turn is
                routed through the multimodal parts loop (text + inline image/
                file Parts); the model receives them gated by its provider's
                vision/input modality (see resolve_modalities).

        Returns:
            The final model response text.

        Raises:
            RuntimeError: If session is not configured.
        """
        # Budget ceiling: refuse rather than silently serve an
        # over-budget turn.  See _refuse_if_budget_exhausted.
        _refusal = self._refuse_if_budget_exhausted()
        self._last_send_refused = _refusal is not None
        if _refusal is not None:
            logger.info("refusing turn: %s", _refusal)
            if on_output:
                on_output("system", f"[{_refusal} — session will not run "
                                    f"further turns]", "write")
            return f"[{_refusal}]"

        if not self._configured:
            raise RuntimeError("Session not configured. Call configure() first.")

        # User-message multimodal: attachments present → build a Part list
        # (text + inline image/file data) and route through the canonical parts
        # loop instead of the text-only path below.  Keeps one multimodal entry.
        if attachments:
            parts = self._parts_from_user_message(message, attachments)
            return self.send_message_with_parts(parts, on_output)

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
            user_id=self._resolve_telemetry_user_id(),
        ) as turn_span:
            self._current_turn_span = turn_span
            # Stamp the user prompt as the trace-level input on the AGENT root
            # span. Observability backends derive a trace's Input from its root
            # observation's ``input.value`` — Langfuse's ingestion reads that
            # key directly. Without it the trace-level Input column stays blank
            # even though child llm/tool spans carry their own messages.
            # Redaction (``JAATO_TELEMETRY_REDACT_CONTENT``, on by default) is
            # applied to this key by _SpanWrapper, exactly like tool input.
            try:
                if message:
                    turn_span.set_attribute("input.value", message)
                    turn_span.set_attribute("input.mime_type", "text/plain")
            except Exception:
                pass
            # Reset per-turn token accumulators for aggregating on the turn span
            self._turn_prompt_tokens = 0
            self._turn_completion_tokens = 0

            # Check and perform GC if needed (pre-send)
            if self._gc_plugin and self._gc_config and self._gc_config.check_before_send:
                self._maybe_collect_before_send()

            # Reset proactive GC tracking for this turn
            self._gc_threshold_crossed = False
            self._gc_threshold_callback = on_gc_threshold
            if on_gc_phase is not None:
                self._gc_phase_callback = on_gc_phase

            # Scrub provider tool-ids (t_xxxxxxxx / c_xxxxxxxx) out of user-facing
            # MODEL text before it reaches the client.  The id exists only at the
            # provider boundary, but the model's free-text narration can mention
            # one — and that must not surface to the user.  Runner-side, where the
            # _reverse map is populated at schema build.  StreamScrubber handles
            # ids split across streaming chunks; history keeps the raw ids (the
            # model stays on hashes), only the user-facing emission is scrubbed.
            _id_scrubber = StreamScrubber()
            _raw_output = on_output
            if _raw_output is not None:
                def on_output(source, text, mode, _raw=_raw_output, _s=_id_scrubber):
                    if source == "model" and text:
                        text = _s.feed(text)
                        if not text:
                            return
                    _raw(source, text, mode)

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
                # Stamp the final response as the trace-level output on the
                # AGENT root span (mirrors input.value above), so the trace's
                # Output column is populated from the root observation rather
                # than left blank. Redacted with input.value when enabled.
                try:
                    if response:
                        turn_span.set_attribute("output.value", response)
                        turn_span.set_attribute("output.mime_type", "text/plain")
                except Exception:
                    pass
                turn_span.set_status_ok()
            except Exception as e:
                turn_span.record_exception(e)
                turn_span.set_status_error(str(e))
                raise
            finally:
                # Emit any trailing partial-id fragment the scrubber held back at
                # the last chunk boundary (raw output — already scrubbed by flush).
                if _raw_output is not None:
                    _tail = _id_scrubber.flush()
                    if _tail:
                        _raw_output("model", _tail, "write")

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
            self._persistence.notify_turn_complete()

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

                    # Diagnostic: log the provider-reported wire size
                    # (the proactive denominator) alongside the
                    # InstructionBudget breakdown so the two can be
                    # compared and the wire_tool_schemas registration
                    # verified.  Only on a fresh per-turn crossing-check
                    # (gated by ``not _gc_threshold_crossed``), so it
                    # fires ~once per turn, not per stream chunk.
                    self._log_gc_denominator("proactive", usage.total_tokens)

                    if percent_used >= threshold:
                        self._gc_threshold_crossed = True
                        self._trace(f"PROACTIVE_GC: Threshold crossed ({percent_used:.1f}% >= {threshold}%)")

                        # Notify via callback if provided
                        if self._gc_threshold_callback:
                            self._gc_threshold_callback(percent_used, threshold)
                        # Typed counterpart: the prose callback above renders a
                        # human sentence, which a driver could only
                        # substring-match.  Same crossing, branchable values.
                        #
                        # Guarded like the callback above rather than relying on
                        # _emit_gc_phase's own None check: the payload coerces
                        # with float(), so building it eagerly does real work --
                        # and raises -- for a listener that isn't there.
                        if self._gc_phase_callback is not None:
                            self._emit_gc_phase("about_to_run", {
                                "percent_used": float(percent_used),
                                "threshold": float(threshold),
                                "trigger_reason": GCTriggerReason.THRESHOLD.value,
                                "strategy": (
                                    self._gc_plugin.name
                                    if self._gc_plugin else None
                                ),
                            })

            # Call original callback if provided
            if on_usage_update:
                on_usage_update(usage)

        return wrapped_callback

    def _dedup_history_for_gc(self) -> int:
        """Collapse byte-identical duplicate tool-results in history (GC Phase 0).

        A model that re-invokes the same tool and gets the identical
        payload back (the qwen3 ``listAvailableTemplates`` re-call
        pathology) bloats the wire with redundant copies that GC EVICTION
        cannot reclaim when they sit in the ``preserve_recent_turns``
        window.  This SHRINKS each earlier duplicate's result body to a
        marker (recency, message structure, and tool_call/tool_result
        pairing all preserved; only exact duplicates touched — zero data
        loss), then:

        1. invalidates the per-message token cache for the shrunk
           messages (their ``message_id`` is unchanged, so the cache would
           otherwise return the stale pre-dedup count); and
        2. re-syncs the CONVERSATION budget so the denominator reflects
           the smaller wire.

        Runs before the eviction phases so eviction only handles whatever
        is still over budget.  Returns the estimated tokens reclaimed
        (0 if nothing deduped).  No-op when ``dedup_identical_tool_results``
        is disabled on the GC config (default enabled).
        """
        if not self._instruction_budget:
            return 0
        if not getattr(
            self._gc_config, "dedup_identical_tool_results", True
        ):
            return 0

        history = self.get_history()
        new_history, chars_reclaimed, elided_ids = dedup_identical_tool_results(
            history
        )
        if not elided_ids:
            return 0

        for mid in elided_ids:
            self._msg_token_cache.pop(mid, None)
        self._history.replace(new_history)
        self._update_conversation_budget()
        self._emit_instruction_budget_update()

        freed_tokens = chars_reclaimed // 4
        self._trace(
            f"GC_DEDUP: collapsed {len(elided_ids)} duplicate tool-result "
            f"message(s), ~{freed_tokens} tokens reclaimed"
        )
        return freed_tokens

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

        # Phase 0: collapse byte-identical duplicate tool-results BEFORE
        # eviction.  This reclaims redundancy that sits inside the
        # preserve_recent_turns window (where eviction can't reach) by
        # shrinking — not removing — the duplicates.  Run first so the
        # context_usage / history below reflect the smaller wire and
        # eviction only handles whatever's still over budget.
        self._dedup_history_for_gc()

        context_usage = self.get_context_usage()
        history = self.get_history()

        # Diagnostic: the InstructionBudget denominator this GC decision
        # is actually evaluating (with wire_tool_schemas broken out).
        self._log_gc_denominator("after_turn")

        def _after(new_history, result, gc_span):
            if not result.success:
                return None
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
            return new_history

        # Use THRESHOLD as the reason since it was triggered by threshold crossing
        _new_history, result = self._run_gc(
            history, context_usage, GCTriggerReason.THRESHOLD,
            on_collected=_after,
        )

        return result

    def _build_llm_span_attributes(self) -> Dict[str, Any]:
        """Build the attribute dict to attach to an LLM telemetry span.

        Combines per-turn context (turn index, active model tier) with
        cache plugin state (anchor, BP3 strategy, totals) so external
        observers can correlate LLM calls with the GC ↔ cache
        coordination dance.

        ``jaato.tier*`` keys are emitted only in tier mode, so single-model
        sessions carry no dead keys.  The two ``*_failures`` counters are
        how the non-fatal post-connect bookkeeping in
        :meth:`_connect_tier_entry` becomes observable: a session whose
        cache plugin failed to re-attach is running uncached, and one whose
        reliability retarget failed is judging patterns against the wrong
        model.  Neither can be allowed to raise, so neither would be
        visible anywhere but a log without these.  The tier is
        what makes the cache figures on this span readable: reads and
        writes are per (model, prefix), so a span whose tier differs from
        its predecessor's is expected to show a full miss, and one whose
        tier is unchanged is not.  Deriving that from ``llm.model_name``
        instead does not work — two tiers may share a model, and a
        budget-control degrade rung rebinds a tier's model underneath it.
        """
        attrs: Dict[str, Any] = {
            "jaato.turn_index": int(self._turn_index),
        }
        active_tier = getattr(self, "_active_tier", None)
        if active_tier is not None:
            attrs["jaato.tier"] = active_tier
            attrs["jaato.tier.switches"] = int(
                getattr(self, "_tier_switch_count", 0))
            # Always emitted, not only when non-zero: a consumer must be
            # able to distinguish "zero failures" from "this build does not
            # report them", and querying for > 0 needs the field present on
            # the healthy spans too.
            attrs["jaato.tier.cache_rewire_failures"] = int(
                getattr(self, "_tier_cache_rewire_failures", 0))
            attrs["jaato.tier.reliability_retarget_failures"] = int(
                getattr(self, "_tier_reliability_retarget_failures", 0))
        cache = getattr(self, "_cache_plugin", None)
        if cache and hasattr(cache, "get_telemetry_attributes"):
            try:
                cache_attrs = cache.get_telemetry_attributes() or {}
                attrs.update(cache_attrs)
            except Exception as e:
                self._trace(f"LLM_TELEMETRY: cache attr fetch failed: {e}")
        # Custom attributes set via set_llm_span_attributes (e.g. a prefetch's
        # prompt-management link). Last so callers can override framework keys.
        if self._llm_span_attributes:
            attrs.update(self._llm_span_attributes)
        return attrs

    def _build_gc_span_attributes(
        self, context_usage: Dict[str, Any], pre_collect: bool = True,
    ) -> Dict[str, Any]:
        """Build the initial attribute dict for a GC telemetry span.

        Thin wrapper over :func:`gc_support.build_gc_span_attributes`
        supplying this session's budget and cache plugin. ``pre_collect``
        is reserved for future pre/post divergence (currently unused).
        """
        return _gc_build_span_attributes(
            context_usage,
            budget=self._instruction_budget,
            cache_plugin=getattr(self, "_cache_plugin", None),
        )

    def _populate_gc_span_result(self, gc_span: Any, result: 'GCResult') -> None:
        """Populate a GC span with attributes derived from the GC result.

        Thin wrapper over :func:`gc_support.populate_gc_span_result`.
        """
        _gc_populate_span_result(gc_span, result, on_trace=self._trace)

    def _run_gc(
        self,
        history: Any,
        context_usage: Dict[str, Any],
        trigger_reason: Any,
        *,
        on_collected: Any = None,
    ) -> "tuple[Any, GCResult]":
        """Run one GC pass through the shared, uniformly-instrumented path.

        Thin wrapper over :func:`gc_support.run_gc` supplying this session's
        plugin, config, budget, cache plugin, telemetry and trace.  Every GC
        path on this class goes through here: before it existed each wired its
        own span by hand and only ONE of four did so completely, so GC was
        observable for a subset of the GC that actually ran, with nothing
        marking which.
        """
        return _gc_run(
            gc_plugin=self._gc_plugin,
            history=history,
            context_usage=context_usage,
            gc_config=self._gc_config,
            trigger_reason=trigger_reason,
            budget=self._instruction_budget,
            cache_plugin=getattr(self, "_cache_plugin", None),
            telemetry=self._telemetry,
            on_trace=self._trace,
            on_collected=on_collected,
            on_phase=self._emit_gc_phase,
        )

    def _emit_gc_phase(self, phase: str, payload: Dict[str, Any]) -> None:
        """Forward one GC lifecycle phase to the registered observer.

        No-op when nothing is listening.  Exceptions are swallowed by
        :func:`gc_support.run_gc`'s wrapper -- an observer must never break
        the collection it observes.
        """
        if self._gc_phase_callback is None:
            return
        self._gc_phase_callback(phase, payload)

    def _apply_gc_removal_list(
        self, result: GCResult, gc_span: Any = None,
    ) -> None:
        """Apply GC removal list to instruction budget.

        Thin wrapper over :func:`gc_support.apply_gc_removal_list`
        supplying this session's budget, cache plugin, and trace. Must be
        called after a successful GC operation to keep the budget in sync
        with the history changes.
        """
        _gc_apply_removal_list(
            result,
            budget=self._instruction_budget,
            cache_plugin=getattr(self, '_cache_plugin', None),
            on_trace=self._trace,
            gc_span=gc_span,
        )

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

    @staticmethod
    def _abnormal_finish_message(finish_reason: FinishReason) -> str:
        """Human-readable banner for an abnormal terminal finish reason.

        Surfaced to clients via an ``on_output("system", ...)`` call (which
        the server adapter turns into an ``AgentOutputEvent(source="system")``)
        so an operator sees WHY a turn ended early instead of the truncation
        looking like a clean completion.  Mirrors the sibling notification
        the cancellation path emits in ``_handle_cancellation``.
        """
        return {
            FinishReason.MAX_TOKENS:
                "Model stopped early: hit the output-token limit "
                "(max_tokens); the response is truncated.",
            FinishReason.SAFETY:
                "Model stopped early: the provider's safety filter "
                "triggered (safety); the response may be incomplete.",
            FinishReason.ERROR:
                "Model stopped early: the provider reported an error "
                "(error).",
            FinishReason.INCOMPLETE:
                "Model stopped early: the response stream ended without "
                "the provider saying why (incomplete); what arrived is a "
                "fragment, not an answer.",
        }.get(
            finish_reason,
            f"Model stopped early: "
            f"{getattr(finish_reason, 'value', finish_reason)}.",
        )

    def _classify_finish_reason(
        self,
        response: ProviderResponse,
        turn_data: Optional[Dict[str, Any]] = None,
        on_output: Optional[OutputCallback] = None,
    ) -> Optional[TurnResult]:
        """Classify a provider response's finish reason.

        Returns a ``TurnResult`` for abnormal terminations (``SAFETY``,
        ``MAX_TOKENS``, ``ERROR``, ``INCOMPLETE``) and ``None`` for
        reasons that the chat loop should continue processing
        (``STOP``, ``UNKNOWN``, ``TOOL_USE``, ``CANCELLED``).

        ``UNKNOWN`` continues and ``INCOMPLETE`` does not, and the
        distinction is the whole of #687: ``UNKNOWN`` is "the turn
        ended, with a label we do not map", ``INCOMPLETE`` is "the turn
        never ended".  Providers raise ``StreamInterruptedError`` rather
        than return an ``INCOMPLETE`` response, so this branch is the
        backstop for any path that does not -- it must never be the
        list above.

        ``CANCELLED`` is handled separately by ``_handle_cancellation``
        because it requires additional logic (mid-turn interrupts,
        UI notification, model notification).

        Side effects (both best-effort, gated on the optional args):

        - ``turn_data``: when supplied, the response's finish reason is
          recorded as ``turn_data['finish_reason']`` (the lowercase enum
          value) for EVERY response — including normal ``STOP`` — so the
          terminal classification in a turn leaves the true reason on the
          turn-accounting dict.  That value rides
          ``on_agent_turn_completed`` → ``TurnCompletedEvent.finish_reason``
          out to clients, letting them branch deterministically instead of
          inferring truncation from empty output.
        - ``on_output``: when supplied AND the finish is abnormal, a
          human-readable ``source="system"`` banner is emitted so the
          abnormal stop is visible, not merely logged — the sibling of the
          cancellation notification.

        Deliberately does NOT touch history, so it can be driven on its
        own.  An abnormal finish also has a history consequence — the
        severed turn may carry a tool call that will never be dispatched
        (#751) — and that lives in :meth:`_finish_abnormally`, the
        wrapper every chat-loop exit calls instead of this method.
        """
        finish_reason = response.finish_reason
        if turn_data is not None and finish_reason is not None:
            turn_data['finish_reason'] = finish_reason.value
        if finish_reason in (
            FinishReason.STOP,
            FinishReason.UNKNOWN,
            FinishReason.TOOL_USE,
            FinishReason.CANCELLED,
        ):
            return None
        logger.warning(f"Model stopped with finish_reason={finish_reason}")
        if on_output is not None:
            on_output(
                "system", self._abnormal_finish_message(finish_reason), "write"
            )
        return TurnResult.from_finish_reason(
            finish_reason, response.get_text()
        )

    def _finish_abnormally(
        self,
        response: ProviderResponse,
        turn_data: Optional[Dict[str, Any]] = None,
        on_output: Optional[OutputCallback] = None,
    ) -> Optional[TurnResult]:
        """Classify a finish reason AND leave history usable afterwards.

        :meth:`_classify_finish_reason` decides whether the turn ended
        abnormally; it deliberately stays a pure classifier so it can be
        driven on its own.  But an abnormal finish is not only a verdict
        -- it is the moment history can be left structurally invalid,
        because the severed turn may carry a tool call that will now
        never be dispatched (#751).  Pairing the two here means every
        abnormal exit from ``_run_chat_loop`` reconciles, rather than
        each of its return sites remembering to.

        Returns whatever the classifier returned: a ``TurnResult`` for
        an abnormal stop (the caller should end the turn with it), or
        ``None`` to keep processing.
        """
        abnormal = self._classify_finish_reason(response, turn_data, on_output)
        if abnormal is not None:
            self._reconcile_unanswered_calls(response.finish_reason)
        return abnormal

    @staticmethod
    def _abnormal_parts_turn_text(response: ProviderResponse) -> str:
        """What ``send_message_with_parts`` returns on an abnormal stop.

        The parts loop has no ``TurnResult`` plumbing, so its terminal
        text is composed here: whatever the model got out before the
        stop, tagged with the reason, or a bare notice when it produced
        nothing at all.
        """
        response_text = response.get_text()
        if response_text:
            return f"{response_text}\n\n[Model stopped: {response.finish_reason}]"
        return f"[Model stopped unexpectedly: {response.finish_reason}]"

    def _finish_or_continue(
        self,
        response: ProviderResponse,
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any],
        context: str = "",
    ) -> Tuple[ProviderResponse, Optional[TurnResult]]:
        """Classify an abnormal finish, then try to continue from it.

        The two halves belong together at every chat-loop exit and are
        separate methods only because they answer different questions.
        :meth:`_finish_abnormally` decides whether the turn ended badly
        and leaves history usable (#751).
        :meth:`_recover_truncated_turn` decides whether it ended
        *recoverably* and, if so, gets the model to try again inside the
        same turn (#749).  Pairing them here means a new exit gets both
        by calling one method, rather than each site remembering the
        order -- and the order matters: history must be reconciled
        before another request is built from it.

        Args:
            response: The response that just arrived.
            use_streaming: Whether a continuation would stream.
            on_output: Client output callback.
            wrapped_usage_callback: Usage callback for a continuation.
            turn_data: Turn accounting.
            context: Trace label for the call site.

        Returns:
            ``(response, None)`` to keep processing -- either the finish
            was normal, or a truncation was continued and *response* is
            the continuation.  ``(response, turn_result)`` to end the
            turn with that result, exactly as before this issue.
        """
        abnormal = self._finish_abnormally(response, turn_data, on_output)
        if abnormal is None:
            return response, None
        continued = self._recover_truncated_turn(
            response, use_streaming, on_output, wrapped_usage_callback,
            turn_data, context=context,
        )
        if continued is None:
            return response, abnormal
        return continued, None

    def _truncation_nudge(
        self,
        response: ProviderResponse,
        attempt: int,
        answered_calls: int,
    ) -> str:
        """Compose what the model is told about its own truncated turn.

        The operator banner (:meth:`_abnormal_finish_message`) tells a
        HUMAN why the turn ended.  This tells the MODEL, which is a
        different message with a different job: it has to be actionable
        (#749).  Three things, in this order:

        1.  **What happened, unambiguously.**  The output cap ended the
            turn; the response above is incomplete.
        2.  **What became of the tool calls.**  After a cap the model's
            likeliest wrong inference is that its write half-happened,
            and a retry built on that is worse than no retry.  When
            calls were reconciled they already carry
            :func:`unexecuted_call_error` in their own result slot, so
            this only points at it rather than repeating it.
        3.  **Where the output stopped**, as a bounded, run-collapsed
            excerpt of the model's own text.

        THE EXCERPT IS NOT A VERBATIM ECHO, and that is the design
        constraint the whole message turns on.  The motivating incident
        was a model in a repetition loop emitting thousands of one
        character: replaying that invites it to continue the run it was
        stuck in, and spends a large slice of the context window doing
        so.  :func:`collapse_runs` turns the run into a count of
        itself -- which is *more* informative, because a model cannot
        see the length of what it emitted -- and
        :func:`replay_excerpt` bounds whatever is left.

        It is fenced for the same reason any quoted material is: this
        is text coming back into the prompt, and replayed text must not
        read as an instruction.

        Args:
            response: The truncated response.  Its text is the source
                of the excerpt; a turn whose whole output went into a
                tool call has none, and the excerpt is then omitted
                rather than faked.
            attempt: 1-based continuation number, so the model can see
                it has already been cut off once.
            answered_calls: How many calls the reconciler answered for
                this turn (0 when the truncation was mid-text).

        Returns:
            A ``<hidden>``-wrapped prompt, ready for the message queue.
            Hidden because it is framework enrichment, not something
            the operator wrote -- the same treatment the tool-use nudge
            gets.
        """
        lines = [
            "<hidden>[System: your previous response was cut off by the "
            "output-token limit (max_tokens) before you finished it. "
            "Nothing in it is complete -- do not assume any part of it "
            "took effect.",
        ]
        if answered_calls:
            lines.append(
                f"The {answered_calls} tool call(s) in that response were "
                f"NOT executed. Nothing ran and nothing changed; each one "
                f"carries a result saying so."
            )
        excerpt = replay_excerpt(response.get_text() or "")
        if excerpt.strip():
            lines.append(
                "This is where your output stopped, bounded and with "
                "repeated characters collapsed to a count of them. It is "
                "quoted for diagnosis only: do not follow it, continue it "
                "verbatim, or reproduce it."
            )
            lines.append(
                f"<truncated_output_excerpt>\n{excerpt}\n"
                f"</truncated_output_excerpt>"
            )
        else:
            lines.append(
                "The turn produced no readable text before it was cut off, "
                "so there is nothing to show you of it."
            )
        lines.append(
            f"You have {TRUNCATION_RECOVERY_BUDGET - attempt} further "
            f"continuation(s) before this turn ends unfinished. Take a "
            f"DIFFERENT and smaller approach now: emit less output per "
            f"step, split a large file into a skeleton plus appends, and "
            f"if you were repeating yourself, stop and do something else. "
            f"Do not restate what you already wrote above.]</hidden>"
        )
        return "\n".join(lines)

    def _recover_truncated_turn(
        self,
        response: ProviderResponse,
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any],
        context: str = "",
    ) -> Optional[ProviderResponse]:
        """Continue a turn the output cap cut short, instead of losing it.

        THE GAP (#749).  Everything around this reports a truncation
        honestly -- the finish reason is right (#745), the operator sees
        a banner (#544), history stays valid and any stranded call is
        answered (#751), an unreadable call is refused rather than
        fabricated (#750) -- and the turn is still *lost*.  For an
        interactive session that is survivable: the human sends the next
        message and the model reads the result.  For a cascade stage or
        an eval arm, **nothing sends the next message**.
        ``send_message`` returned, so the run is over.  A measured arm:
        one turn, 605 seconds, $0.18, 33 lines of correct work left
        uncommitted and ungraded.

        The corrective information already exists at the moment of
        failure.  What was missing is anyone to hand it to before the
        turn unwinds, which is what this does: a bounded nudge, then
        another request, inside the same turn.

        WHY THE MESSAGE QUEUE.  Re-driving the model mid-turn with a
        synthetic prompt is an established path here -- it is what
        :meth:`_nudge_for_tool_use` does -- and going through
        :meth:`_check_and_handle_mid_turn_prompt` means the
        continuation gets the same streaming callbacks, retry policy,
        history append, usage accounting and telemetry span as any
        other request.  A real user message queued in the meantime pops
        first, which is correct: a waiting human outranks the nudge.
        It inherits that helper's one quirk: on a NON-streaming provider
        the continuation's text is emitted once by the helper and again
        by parts processing.  Pre-existing on the tool-use nudge's path
        and invisible under streaming; not worth widening this fix to
        chase.

        PRECONDITIONS THE CALLER OWES.  History must already be valid
        -- the caller reconciles the severed turn's calls (via
        :meth:`_finish_abnormally`, or inline in the parts loop) before
        calling this, because the very next thing here is another
        request against that history.

        Args:
            response: The truncated response the caller was about to
                end the turn on.
            use_streaming: Whether the continuation streams.
            on_output: Client output callback.  Gets a ``system`` note
                per continuation -- a recovered truncation is still
                worth seeing, and the attempt count belongs in the
                operator's view as much as in the turn record.
            wrapped_usage_callback: Usage callback for the continuation.
            turn_data: Turn accounting.  Gains
                ``truncation_recoveries`` (how many continuations this
                turn spent) and has ``finish_reason`` refreshed to each
                continuation's, so a turn that recovers reports the
                reason it actually ended on.
            context: Trace label for the call site.

        Returns:
            A response that is no longer truncated and should be
            processed as if it had arrived first, or ``None`` when
            nothing was attempted (the finish reason does not qualify,
            or the turn was cancelled) or the budget ran out with the
            turn still truncated -- in which case the caller ends the
            turn exactly as it does today.
        """
        if response.finish_reason not in TRUNCATION_RECOVERY_REASONS:
            return None
        if self._is_cancelled():
            return None

        while response.finish_reason in TRUNCATION_RECOVERY_REASONS:
            if self._truncation_recovery_count >= TRUNCATION_RECOVERY_BUDGET:
                self._trace(
                    f"TRUNCATION_RECOVERY_EXHAUSTED count="
                    f"{self._truncation_recovery_count} ({context}), "
                    f"letting the turn end truncated"
                )
                return None

            self._truncation_recovery_count += 1
            attempt = self._truncation_recovery_count
            turn_data['truncation_recoveries'] = attempt
            answered = self._count_answered_calls(response)
            self._trace(
                f"TRUNCATION_RECOVERY attempt={attempt}/"
                f"{TRUNCATION_RECOVERY_BUDGET} ({context}) "
                f"answered_calls={answered} "
                f"text_chars={len(response.get_text() or '')}"
            )
            span = getattr(self, '_current_turn_span', None)
            if span is not None:
                try:
                    span.set_attribute("jaato.truncation.recovery", attempt)
                except Exception as exc:  # pragma: no cover - telemetry only
                    logger.debug(
                        f"Failed to set truncation telemetry: {exc}"
                    )
            if on_output is not None:
                # Complements the banner ``_finish_abnormally`` has
                # just emitted -- that one names the cap, this one says
                # the turn is not over because of it.
                on_output(
                    "system",
                    f"Continuing the truncated turn: telling the model it "
                    f"hit the output-token limit (max_tokens) and asking "
                    f"for a smaller step (attempt {attempt}/"
                    f"{TRUNCATION_RECOVERY_BUDGET}).",
                    "write",
                )

            self._message_queue.put(
                self._truncation_nudge(response, attempt, answered),
                "system",
                SourceType.SYSTEM,
            )
            continued = self._check_and_handle_mid_turn_prompt(
                use_streaming, on_output, wrapped_usage_callback, turn_data
            )
            if continued is None:
                self._trace(
                    f"TRUNCATION_RECOVERY_NO_RESPONSE ({context})"
                )
                return None

            response = continued
            if response.finish_reason is not None:
                turn_data['finish_reason'] = response.finish_reason.value
            if response.finish_reason in TRUNCATION_RECOVERY_REASONS:
                # Cut off again.  Answer whatever this attempt stranded
                # before the next request is built from that history.
                self._reconcile_unanswered_calls(response.finish_reason)

        return response

    @staticmethod
    def _count_answered_calls(response: ProviderResponse) -> int:
        """How many tool calls the severed *response* carried.

        The reconciler answers exactly these, so the count doubles as
        "how many calls the model must be told did not run".  Read from
        the response rather than from the reconciler's return value
        because a caller may have reconciled several steps earlier.
        """
        return sum(1 for p in response.parts if p.function_call)

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
            # Media the model produced is delivered out of band and never
            # becomes a Part, so a turn that spoke has no parts at all.
            # Counting it as content is what stops the nudge firing on a
            # good spoken answer; the provider normally also gives such a
            # turn a transcript Part, and this is the backstop for a
            # provider that sends none.
            spoke = response.media_chunks > 0
            is_empty = not spoke and (not response.parts or all(
                not p.text and not p.function_call for p in response.parts
            ))
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

        # 1.4 Path 1 quirk: signal_completion validation_failed →
        # request named-function tool_choice on the retry.
        #
        # When the lifecycle tool returns
        # ``{"error": "validation_failed", ...}`` (schema mismatch or
        # fatal processor error), stamp the failing tool's name on the
        # session so the NEXT ``provider.complete()`` in this turn can
        # request server-side xgrammar enforcement via named-function
        # ``tool_choice``.  Scoped to ``signal_completion`` because
        # that's the only tool today whose return shape carries the
        # ``"validation_failed"`` sentinel + a schema-validated payload
        # contract (see ``shared/lifecycle_tools.py:594``).  The
        # provider plugin decides whether to honor the request — vllm
        # honors it when ``force_tool_choice_for_lifecycle`` quirk is
        # True; providers without the quirk no-op.  Cleared after one
        # consumed call (see ``_do_send_tool_results``) so the contract
        # is "one xgrammar-enforced retry, then back to auto".
        self._maybe_stamp_lifecycle_retry_tool_choice(tool_results)

        # 1.5 signal_completion terminates the turn.
        #
        # Pre-2026-06-07 the loop continued past a successful
        # signal_completion call: the result was sent back to the
        # model with the task-completion spur appended (line 5708,
        # ``_send_tool_results_and_continue``), the model read
        # "After each action, continue working..." and either
        # (a) STRONG models recognized the request was already
        #     fulfilled, emitted plain text with no tool calls, and
        #     the outer ``while any(p.function_call ...)`` loop
        #     exited naturally — the spur worked harmlessly.
        # (b) WEAK models (Llama 3.1 8B AWQ in the vLLM smoke
        #     2026-06-07) took "continue working" literally and
        #     called signal_completion AGAIN with the same payload.
        #     Framework returned ``{status: completed, ...}`` plus
        #     the spur AGAIN.  Infinite loop until the harness
        #     timeout.
        #
        # The architectural fix: signal_completion is the terminal
        # tool by contract.  When it succeeds (= schema validation
        # passed if ``completion_payload_schema`` is declared, and
        # all configured completion processors passed — see
        # ``LifecycleTools._execute_signal_completion``), the
        # session is OVER.  ``hooks.on_agent_completed`` already
        # fired inside the executor; downstream consumers have
        # their event.  There is no model call left to make.
        # Terminating here is correct for ALL providers — strong
        # models save one wasted round-trip, weak models stop
        # looping.
        #
        # IMPORTANT: ``_signal_completion_called`` is set ONLY on
        # the validated-success path.  Schema-validation failures
        # and fatal-processor failures return the
        # ``validation_failed`` self-correction error without
        # setting the flag — so this guard does NOT terminate on
        # those paths; the normal continuation lets the model
        # retry signal_completion with a corrected payload.
        if getattr(self, "_signal_completion_called", False):
            final_text = ''.join(accumulated_text) if accumulated_text else ""
            self._trace(
                "SIGNAL_COMPLETION_TERMINATES_TURN: skipping continuation "
                "(spur + model round-trip) — session is over"
            )
            return None, TurnResult.success(final_text), False

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

        # 2.7. Intra-turn refuse-send gate (dispatch chokepoint).  Refresh the
        # budget so it reflects the post-GC tool-results payload about to be
        # sent, then refuse if it still exceeds the context window.  Runs
        # regardless of GC config (GC only tries to reduce; the gate is the
        # hard stop) — the refresh above is inside the GC branch and would be
        # skipped when GC is disabled.
        self._update_conversation_budget()
        self._assert_payload_fits_context()

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

        # 5. Classify finish reason for abnormal stops.  A turn cut off
        #    at the output cap is recoverable rather than terminal, so
        #    this may hand back a CONTINUATION to keep processing
        #    instead of a result to end on (#749).
        response, abnormal = self._finish_or_continue(
            response, use_streaming, on_output, wrapped_usage_callback,
            turn_data, context=f"after tool results ({context})",
        )
        if abnormal is not None:
            return None, abnormal, False

        # 6. Nudge if TOOL_USE without function calls
        response = self._nudge_for_tool_use(
            response, use_streaming, on_output, wrapped_usage_callback,
            turn_data, context=context,
        )

        # A delegated tier hands back HERE: the continuation has landed
        # and asks for nothing more, the earliest point the framework can
        # know its completion settled.
        self._exit_completion_tier_if_settled(response)

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

    def _begin_turn_completion_state(self) -> None:
        """Clear the completion flags that are per-TURN in meaning.

        Two flags looked session-lifetime but every reader asks a per-turn
        question:

        * ``_signal_completion_called`` -- read by the turn terminator
          (``_execute_tools_and_continue``: "session is over"), the nudge
          predicate, the quiescence hook (whose own comment says "called
          signal_completion DURING THIS TURN"), the signal_completion
          idempotency guard ("in the same tool batch"), the auto-finalize
          synthesizer, the subagent nudge loop, and the embedded nudge gate.
        * ``_session_quiescent_emitted`` -- the once-per-turn quiescence latch.

        ``_completion_nudges_fired`` IS NOT ONE OF THEM, and clearing it here
        cost the framework its only bound on the nudge loop (#767).  A nudge
        RE-PROMPTS THE SESSION, so the nudge's own turn ran this reset and
        handed the budget back the token it had just spent.  Both guards read
        the counter the same way and both were therefore unbounded: the
        top-level guard in ``core.py``'s model_thread re-armed itself every
        turn (observed: "nudge 1/2" logged three times in one session, and a
        conformance session that never signals turned 735 times in 40
        seconds), and the subagent guard's ``while ... < MAX_COMPLETION_NUDGES``
        in ``subagent/plugin.py`` -- written on the assumption that the
        counter only goes up -- could not terminate at all.  ``max_turns``,
        ``budget_control`` and the caller's own wall-clock were what actually
        stopped those sessions, at whatever they had spent by then.

        So the budget is per SESSION, which is what ``MAX_COMPLETION_NUDGES``
        already claimed to be.  A session that spends it terminates
        (``NudgeExhausted``), so "the next task on this session gets a fresh
        budget" describes a session that no longer exists -- and a
        completion-gated session is one-shot by construction.  The one real
        cost is an agent that answers each nudge with more work and needs a
        third: it is now cut off at two.  That is the declared ceiling doing
        its job, and raising it is a knob, not a bug.

        None of them is persisted, so this has no restore implications.

        Invisible for a one-shot session, where turn 0 is the only turn --
        which is why it survived.  On a SUSPEND/RESUME session the agent calls
        ``signal_completion`` every turn (``outcome=suspended`` ends the turn,
        the driver wakes the same session later), so the flag latched on turn
        0 and every later turn was TRUNCATED at its first tool batch: the
        terminator fired on a stale flag before the model could reach its
        exit.  Observed as a 2.9s turn with one tool call and no completion,
        six runs running, and misread as the model forgetting its exit.

        Called from both chat loops rather than from ``send_message``, since
        ``send_message`` may delegate to ``send_message_with_parts`` -- one
        reset per turn on either path, and neither path can miss it.  That
        is also why the truncation-recovery budget is reset here (#749):
        it is per-turn, and the parts loop is the path a per-loop reset
        would miss.
        """
        self._signal_completion_called = False
        self._session_quiescent_emitted = False
        self._truncation_recovery_count = 0

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
        self._begin_turn_completion_state()

        # Set output callback on executor
        if self._executor:
            self._executor.set_output_callback(on_output)

        # Set output callback on registry for enrichment notifications
        if self._runtime.registry and on_output:
            self._runtime.registry.set_output_callback(on_output, self._terminal_width)

        # Initialize cancellation support
        self._cancel_token = CancelToken()
        with self._delivery_lock:
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
            # ``total`` is the LAST response's total_tokens, which for a
            # prompt-inclusive provider is the end-of-turn CONTEXT SIZE — what
            # GC and the context displays want.  It is NOT what the turn cost:
            # a turn with a tool call has >=2 responses and each is billed, so
            # summing responses is the SPEND.  Both are legitimate and
            # different consumers want different ones; conflating them
            # undercounted a real 3-turn run by 41%.
            'spend_total': 0,
            'spend_prompt': 0,
            'spend_output': 0,
            # PROVIDER-REPORTED cost for this turn, accumulated across the
            # turn's responses exactly like ``spend_total``.  ``None`` means
            # the provider reported none -- distinct from ``0.0``, which
            # means it reported free.
            #
            # It used to be dropped: the turn-completed hook carried no cost
            # parameter, so ``_build_usage`` -- which HAS a
            # ``cost_usd_override`` -- was never given one, and the event
            # carried None for every provider that reports a real figure.
            # ``_resolve_span_cost`` meanwhile read ``usage.cost_usd`` for
            # telemetry, so ONE measurement survived on the span path and
            # died on the event path.
            'cost_usd': None,
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
            self._track_streaming_usage(turn_data, usage)
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

            # Pre-flight refuse-send gate (dispatch chokepoint).  The user
            # message is now in history; refresh the budget so it reflects
            # THIS outgoing prompt, then refuse to dispatch if the accounted
            # payload (plus the provider's max_tokens cap when known) exceeds
            # the context window.  Placed here — not at the top of
            # send_message — because that is before the prompt is appended, so
            # the budget there cannot see it (the single massive first-turn
            # prompt would slip through).  Runs once before the rewind retry
            # loop.
            self._update_conversation_budget()
            self._assert_payload_fits_context()

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
                    def streaming_callback(chunk) -> None:
                        nonlocal first_chunk_sent
                        # Model-generated media is not text: it goes to
                        # clients and never into the transcript, the
                        # reliability plugin's text scanner, or the
                        # interrupt-preservation buffer.
                        if isinstance(chunk, MediaDelta):
                            self._deliver_model_media(chunk)
                            return
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

            # Check finish_reason for abnormal termination.  An
            # output-cap truncation is continued rather than lost when
            # the per-turn budget allows, in which case ``response``
            # becomes the continuation and the turn goes on (#749).
            response, abnormal = self._finish_or_continue(
                response, use_streaming, on_output, wrapped_usage_callback,
                turn_data, context="after initial message",
            )
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
            self._budget_observe_turn(turn_data)

            if turn_data['total'] > 0:
                self._turn_accounting.append(turn_data)

            # Update instruction budget with conversation tokens
            self._update_conversation_budget()

            # Clean up cancellation state and activity phase.
            #
            # UNDER ``_delivery_lock``: an ``offer_message`` that has already
            # observed ``_is_running`` True must finish its enqueue BEFORE
            # this flip, or its message would land after the final drain
            # below and never be collected.  That is the whole strand, and a
            # lock is what makes its absence a fact rather than an argument
            # about instruction ordering.
            with self._delivery_lock:
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
                # RECORDED AS DUE, NOT EMITTED HERE.
                #
                # This runs INSIDE ``send_message``, and every driver fires
                # ``on_agent_turn_completed`` AFTER ``send_message`` returns.
                # Emitting here therefore put ``SessionTerminatedEvent``
                # BEFORE the final ``TurnCompletedEvent`` of the very turn it
                # is reporting the end of -- measured at 6.92s/6.92s, terminal
                # first.
                #
                # Nothing should follow a terminal event, and something did.
                # Worse, a consumer acted on it: the daemon's cascade policy
                # detaches a cid'd session's clients on SessionTerminated to
                # release its slot, so the TurnCompletedEvent that arrived
                # afterwards reached NOBODY.  A completion-gated cascade arm
                # came back turns=0, tokens=0 with its work done and its file
                # on disk -- a silent zero that reads as "the model did
                # nothing".
                #
                # The session knows WHETHER quiescence is due; only the
                # driver knows WHEN the turn's own events are finished.  So
                # the flag is set here and flushed by
                # :meth:`flush_session_quiescent` after the driver's
                # turn-completed hook.
                self._quiescent_due_reason = "natural"

            # (quiescence is flushed by the driver — see
            # ``flush_session_quiescent``)

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

    def _maybe_stamp_lifecycle_retry_tool_choice(
        self, tool_results: List[ToolResult],
    ) -> None:
        """Path 1 quirk hook (server 0.6.195+).

        Scan ``tool_results`` for ``signal_completion`` returning
        ``{"error": "validation_failed", ...}`` — the canonical
        self-correction shape from ``LifecycleTools._execute_signal_completion``
        when ``completion_payload_schema`` rejected the args.  When
        found, stamp ``self._pending_tool_choice_name`` with the tool
        name so the next ``provider.complete()`` call passes
        ``tool_choice={type: function, function: {name: ...}}`` to the
        provider.

        Only ``signal_completion`` is scoped today — it's the only
        tool whose return contract carries the ``"validation_failed"``
        sentinel + a typed-payload schema worth enforcing via
        server-side xgrammar.  Generalizing to other tools is
        mechanical (drop the name check) once another tool surfaces
        the same pattern.

        Non-validation_failed tool results, results without the
        sentinel, and results from other tools are no-ops.  The
        provider plugin decides whether to honor the request via its
        own quirk gate (vllm: ``force_tool_choice_for_lifecycle``);
        Stamping is GATED on the active provider HONORING the quirk —
        ``getattr(self._provider, "_force_tool_choice_for_lifecycle",
        False)`` (server 0.6.166+, mirrors the
        ``force_narration_between_tools`` provider-attr gate at the
        tool-result append site).  A provider whose ``complete()`` does
        not accept ``tool_choice`` never sets this attr, so the quirk is
        a silent NO-OP for it — no stamp, no ``tool_choice`` ever passed
        to its ``complete()``.  This replaces the prior
        "provider-agnostic stamp + providers ignore the kwarg" design,
        which ``TypeError``'d on every provider whose signature lacks the
        kwarg (openrouter codegen, 2026-06-11; only vllm/anthropic/
        tensorrt_llm accept it, and only vllm sets the attr).
        Profile-scoped via ``profile.quirks.force_tool_choice_for_lifecycle``.
        """
        # Quirk gate: only form the forced-tool_choice intent when the
        # active provider declares it honors the quirk.  Providers that
        # don't support the ``tool_choice`` kwarg never set this attr ->
        # no-op (no TypeError on their complete()).
        if not getattr(
            self._provider, "_force_tool_choice_for_lifecycle", False,
        ):
            return
        # PR-255 PROBE INSTRUMENTATION (TEMPORARY, 2026-06-08).
        #
        # Empirical disagreement between code-trace + grep on
        # /tmp/provider_trace.log (zero PENDING_TOOL_CHOICE all-time
        # despite a cascade flow that DEFINITELY hit signal_completion
        # validation_failed per peer 7:1's daemon log at 06:13:32).
        # Three competing hypotheses can't be resolved by reading
        # source alone: (1) function never called, (2) tr.result is
        # not a dict at scan time (peer's original hypothesis: spur
        # stringification reaches us), or (3) result.get("error") is
        # something other than "validation_failed".
        #
        # This single logger.info captures count / names / types /
        # error_keys-or-first-64-chars in one line per scan so the
        # next cascade re-run triangulates which holds.  Writes to
        # ``workspace/.jaato/logs/session_<sid>_*.log`` via the
        # module-level ``logger`` (per peer-verified path at
        # ``/home/apanoia/Sources/Jaato-framework-and-examples/jaato-based-kb-enablement-2.0/tests/runs/cascade_smoke/.jaato/logs/``).
        # Revert after the actual fix lands as PR-256.
        logger.info(
            "MAYBE_STAMP_SCAN count=%d names=%s types=%s error_keys=%s",
            len(tool_results),
            [t.name for t in tool_results],
            [type(t.result).__name__ for t in tool_results],
            [
                t.result.get("error") if isinstance(t.result, dict)
                else (t.result[:64] if isinstance(t.result, str) else None)
                for t in tool_results
            ],
        )

        for tr in tool_results:
            if tr.name != "signal_completion":
                continue
            result = tr.result
            # Result can be dict, string, or other; we only match the
            # dict shape with the canonical sentinel.
            if not isinstance(result, dict):
                continue
            if result.get("error") != "validation_failed":
                continue
            self._pending_tool_choice_name = tr.name
            self._trace(
                f"PENDING_TOOL_CHOICE: {tr.name} stamped after "
                f"validation_failed return — next provider.complete() "
                f"will request named-function tool_choice if the "
                f"provider honors the force_tool_choice_for_lifecycle "
                f"quirk"
            )
            return

    def _consume_pending_tool_choice(self) -> Optional[Dict[str, Any]]:
        """Return the pending tool_choice dict (OpenAI/vLLM wire shape)
        and clear the stamp.  Called once per ``provider.complete()``
        at every call site in the turn loop so the retry request fires
        for exactly one model call before reverting to auto.

        Returns ``None`` when no retry is pending.  Returns a dict
        ``{"type": "function", "function": {"name": <tool_name>}}``
        when ready to consume — the canonical OpenAI Chat Completions
        wire shape; other providers translate (or ignore) at their
        plugin layer.
        """
        # PR-256 PROBE: log consume outcome to distinguish B.1
        # ("consume returns None for context but not discovery") from
        # B.2/B.3 (consume returns name but provider-side or wire-side
        # breaks the chain).  Same destination as the entry probe in
        # PR-255 — per-session log via module-level ``logger``.
        # Revert with PR-257 alongside the actual fix.
        if not self._pending_tool_choice_name:
            logger.info("MAYBE_STAMP_CONSUME pending=None — no retry queued")
            return None
        name = self._pending_tool_choice_name
        self._pending_tool_choice_name = None
        logger.info(
            "MAYBE_STAMP_CONSUME pending=%r — returning OpenAI wire shape, "
            "cleared", name,
        )
        return {"type": "function", "function": {"name": name}}

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
                    fc_result_payload = _split_executor_result_impl(
                        result.executor_result
                    )[1]
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
                        is_error_result=tool_result_is_error(fc_result_payload),
                        result_status=tool_result_status(fc_result_payload),
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

        # ``stream_id`` is only known once ``start_stream`` returns, but
        # chunks can fire before then, so it is read from a mutable cell
        # rather than closed over.  A chunk that beats the assignment
        # carries "" and is still correlated by ``call_id`` -- which is
        # the key clients actually join on.
        _stream_cell = {"id": ""}
        _call_id = fc.id or ""

        def on_chunk(chunk: StreamChunk) -> None:
            """Route one streaming-tool chunk by its audience.

            Audience is data on the chunk, not a fixed policy of this
            callback.  Historically this wrapped every chunk in
            ``<hidden>`` -- "for the model, hidden from the user" -- which
            media inverts: a TTS tool's audio is for the user and may
            never reach the model at all.

            - MODEL (the default, so existing producers are unchanged):
              into the conversation via ``on_output``, wrapped in
              ``<hidden>`` exactly as before.
            - CLIENT: to subscribed clients only, via ``on_tool_output``;
              never passed to ``on_output``, so it never enters history.
            - BOTH: both of the above.

            Binary payloads go out base64-encoded and are never handed to
            ``on_output`` -- that path is text and would corrupt them.
            """
            if chunk.audience.reaches_model() and on_output:
                # Wrap in <hidden> so the hidden_content_filter strips it from user view
                # but the model still receives the streaming results.  Media
                # is skipped here: this is a text channel.
                if chunk.content:
                    on_output(
                        "streaming",
                        f"<hidden>[{base_name}] {chunk.content}</hidden>",
                        "append",
                    )

            if chunk.audience.reaches_client() and self._ui_hooks and _call_id:
                # sequence/chunk_type/metadata used to be discarded here;
                # pass the sequence through rather than re-counting.
                if chunk.is_media():
                    self._ui_hooks.on_tool_output(
                        agent_id=self._agent_id,
                        call_id=_call_id,
                        chunk=chunk.content,
                        stream_id=_stream_cell["id"],
                        sequence=chunk.sequence,
                        mime_type=chunk.mime_type,
                        data_b64=chunk.data_b64(),
                        # A tool's media stream has NO per-chunk terminal
                        # signal.  This read `chunk.chunk_type == "final"`,
                        # a category error: chunk_type is a CONTENT-kind
                        # hint (match / progress / result / stdout /
                        # stderr / display / file / input / summary /
                        # error), never a lifecycle marker, so the
                        # comparison could not be true and a client
                        # waiting on `final` for a tool's audio waited
                        # forever.  Saying False plainly is honest;
                        # inventing a terminal-chunk protocol is a design
                        # change.  Model speech is unaffected -- it
                        # carries `delta.final`.
                        final=False,
                    )
                elif chunk.content:
                    self._ui_hooks.on_tool_output(
                        agent_id=self._agent_id,
                        call_id=_call_id,
                        chunk=chunk.content,
                        stream_id=_stream_cell["id"],
                        sequence=chunk.sequence,
                    )

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
            _stream_cell["id"] = handle.stream_id

            # Format initial chunks for model.  CLIENT-audience chunks
            # are excluded -- this list is returned in the tool result and
            # so becomes conversation history.
            initial_content = []
            for chunk in handle.initial_chunks:
                if chunk.audience.reaches_model():
                    initial_content.append(chunk.content)

            return (True, {
                "stream_id": handle.stream_id,
                "tool_name": base_name,
                "status": handle.status.value,
                "initial_results": initial_content,
                # Counts what the model can SEE, not what arrived.
                # These were the same set until CLIENT-audience chunks
                # could be filtered out of the content, after which the
                # model was told "Received N initial results" over a
                # shorter list -- the count-vs-content divergence class.
                "initial_count": len(initial_content),
                "message": (
                    f"Streaming started. Received {len(initial_content)} initial results. "
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

        A call carrying ``fc.unreadable_args`` is **refused** rather than
        executed: the provider could not decode its arguments, so there
        is no call to run (#750).  It still travels the full path --
        hooks, span, ``ToolResult`` -- as a failed call, so the model is
        told its request was unreadable and the tool_use/tool_result
        pairing in history stays intact.
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

            if fc.unreadable_args is not None:
                # The provider could not decode this call's arguments, so
                # there is no call to run: executing it would be acting on
                # a request the model never made (#750).  Report it back
                # as a failed call instead -- an error result is still a
                # tool output, so history stays paired and the model can
                # re-emit the call.
                self._trace(
                    f"TOOL_REFUSED_UNREADABLE_ARGS name={name} "
                    f"call_id={fc.id} chars={len(fc.unreadable_args)}"
                )
                executor_result = (False, unreadable_arguments_error(fc))
            elif self._is_streaming_tool(name):
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
                tool_span.set_attribute("output.value", _telemetry_safe_json(result_dict_for_output))
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
            fc_result_payload = _split_executor_result_impl(executor_result)[1]
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
                is_error_result=tool_result_is_error(fc_result_payload),
                result_status=tool_result_status(fc_result_payload),
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

        Refuses a call carrying ``fc.unreadable_args`` on the same terms
        as ``_execute_single_tool`` -- see there for why.

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
                if fc.unreadable_args is not None:
                    # Unreadable arguments are refused, not executed (#750)
                    # -- see the sequential path for the reasoning.
                    executor_result = (False, unreadable_arguments_error(fc))
                elif self._is_streaming_tool(name):
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
                    tool_span.set_attribute("output.value", _telemetry_safe_json(result_dict_for_output))
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

        # Inject task-completion spur as a MODEL-FACING suffix on the last tool
        # result — NOT folded into ``result`` (that str()'d the structured dict
        # into a repr-string and broke the ledger / provenance / enrichment).
        # ``result`` stays structured; the converter appends ``model_suffix`` at
        # serialization time (render_result_for_model).
        if tool_results:
            last = tool_results[-1]
            hidden = f"<hidden>{_TASK_COMPLETION_INSTRUCTION}</hidden>"
            combined = f"{last.model_suffix}\n\n{hidden}" if last.model_suffix else hidden
            tool_results = tool_results[:-1] + [_dc_replace(last, model_suffix=combined)]

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
            # Model-facing suffix (keep ``result`` structured — see above).
            piggyback = (
                f"<user_message>{combined_prompt}</user_message>\n"
                f"The user has sent a new message during your tool execution. "
                f"Please address their input in your next response."
            )
            combined = (
                f"{last.model_suffix}\n\n{piggyback}"
                if last.model_suffix else piggyback
            )
            tool_results = tool_results[:-1] + [_dc_replace(last, model_suffix=combined)]
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

        # Phase 0: collapse byte-identical duplicate tool-results first
        # (shrink, not evict) — reclaims redundancy eviction can't reach in
        # the preserve_recent_turns window.  See _dedup_history_for_gc.
        self._dedup_history_for_gc()

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

        new_history, result = self._run_gc(
            history, context_usage, GCTriggerReason.CONTEXT_LIMIT,
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

    def _truncate_results_to_fit(
        self, tool_results: List[ToolResult], current_tokens: int, limit_tokens: int
    ) -> List[ToolResult]:
        """Truncate tool results to reduce token count (reactive recovery).

        Thin wrapper over
        :func:`tool_result_truncation.truncate_results_to_fit`.
        """
        return _truncate_results_to_fit_impl(
            tool_results, current_tokens, limit_tokens, on_trace=self._trace,
        )

    def _cap_tool_results(self, tool_results: List[ToolResult]) -> List[ToolResult]:
        """Proactively cap tool results before they enter history.

        Thin wrapper over :func:`tool_result_truncation.cap_tool_results`
        supplying the budget-derived context limit and current total. No-op
        when no budget (or an unknown context limit) is configured.
        """
        budget = self._instruction_budget
        if not budget or budget.context_limit == 0:
            return tool_results
        return _cap_tool_results_impl(
            tool_results,
            context_limit=budget.context_limit,
            current_total_tokens=budget.total_tokens(),
            on_trace=self._trace,
        )

    @staticmethod
    def _mime_to_modality(mime_type: Optional[str]) -> Optional[str]:
        """Map a MIME type to its canonical input-modality token, or None.

        Mirrors the modality vocabulary in
        ``shared/plugins/model_provider/base.py`` (image / audio / video /
        file).  Unknown types return ``None`` so the content gate leaves
        them untouched — it never over-strips content it can't classify.
        """
        if not mime_type:
            return None
        m = mime_type.split(";", 1)[0].strip().lower()
        if m.startswith("image/"):
            return "image"
        if m.startswith("audio/"):
            return "audio"
        if m.startswith("video/"):
            return "video"
        if m == "application/pdf":
            return "file"
        return None

    def _gate_tool_results_for_active_modalities(
        self, tool_results: List[ToolResult]
    ) -> List[ToolResult]:
        """Synthetic-self-correct content gate (multimodal-by-composition).

        The active model can only *see* the input modalities its provider
        declares (``provider.modalities()``).  When a tool returns
        attachment content of a modality the active model can't view
        (canonically a ``readFile`` image while in a text-only tier),
        sending the bytes would silently fail.  Instead this strips those
        attachments and appends a short, actionable note to the tool
        result telling the agent to ``enter_tier("vision")`` (or, with no
        vision tier, that the model can't view the content).

        Per ``docs/design/multimodal-model-support.md`` this is the
        load-bearing correctness piece: it turns the agent's mistake
        (reading an image in a non-vision tier) into a loud, self-correcting
        signal instead of a silent drop, and the turn continues so the
        agent can switch tiers and re-run the tool.

        No-op when the provider isn't set yet, or when every attachment's
        modality is supported (the common case — cheap set membership per
        attachment; vision-capable active models pass straight through).
        """
        provider = self._provider
        if provider is None:
            return tool_results
        return [
            self._gate_one_tool_result(r, provider) for r in tool_results
        ]

    def _gate_one_tool_result(
        self, result: ToolResult, provider: 'ModelProviderPlugin'
    ) -> ToolResult:
        """Apply the modality gate to a single tool result.

        Returns ``result`` unchanged when it has no attachments or all
        attachment modalities are supported; otherwise returns a copy with
        the unsupported attachments removed and a withheld-note appended to
        the result text.
        """
        if not result.attachments:
            return result
        kept: List[Any] = []
        withheld: Dict[str, int] = {}
        rerouted: List[Any] = []
        for att in result.attachments:
            modality = self._mime_to_modality(getattr(att, "mime_type", None))
            # None = unclassifiable; keep (don't over-strip).  Otherwise
            # keep iff the active model declares it.
            if modality is None or provider.supports_modality(modality):
                kept.append(att)
            else:
                withheld[modality] = withheld.get(modality, 0) + 1
                rerouted.append(att)
        if not withheld:
            return result
        # The gate is a ROUTER, not a filter.  Content the model cannot
        # consume is exactly the content a viewer might want, so the
        # withheld attachments are emitted to subscribed clients before
        # being stripped from the model's copy.  Delivery is best-effort:
        # a failure here must not fail the tool result.
        self._emit_withheld_attachments_to_clients(result, rerouted)
        note = self._build_withheld_attachment_note(withheld)
        self._trace(
            f"MODALITY_GATE: withheld {dict(withheld)} from tool "
            f"{result.name!r} (active model {self._model_name!r} lacks them)"
        )
        # Keep ``result`` structured; the withheld-attachment note is
        # model-facing only (append to model_suffix, appended at serialization).
        combined = (
            f"{result.model_suffix}\n\n{note}" if result.model_suffix else note
        )
        return _dc_replace(result, attachments=(kept or None), model_suffix=combined)

    def _model_media_stream_id(self, delta: 'MediaDelta') -> str:
        """A distinct stream id per utterance, not per agent.

        A constant id per agent collided every utterance in a session
        into one stream: a client keying chunks by ``stream_id`` then
        saw sequences restart at 0 mid-stream, so a retried turn's audio
        was spliced onto the first attempt's and no gap check could tell
        them apart.

        The provider restarts ``sequence`` at 0 for each turn, so that
        reset IS the utterance boundary — no extra plumbing needed.
        """
        if delta.sequence == 0:
            self._model_media_utterance = (
                getattr(self, "_model_media_utterance", 0) + 1)
        return f"model:{self._agent_id}:{getattr(self, '_model_media_utterance', 1)}"

    def _deliver_model_media(self, delta: 'MediaDelta') -> None:
        """Deliver one chunk of MODEL-generated media to subscribed clients.

        Model-emitted audio reuses the tool-output media channel rather
        than introducing a rival event, so all three existing subscription
        surfaces (SDK client, ``subscribeToEvents``, ``EventBus``) carry it
        with no new API.  Because that channel is keyed by ``call_id`` and
        this content belongs to no tool call, the reserved id
        :data:`MODEL_MEDIA_CALL_ID` is used; clients distinguish
        model-generated media from tool-produced media by that id.

        Model media is CLIENT-audience by construction: the model produced
        it, so replaying it back into the model's own history would be
        both redundant and, for audio, meaningless.  It therefore never
        touches ``on_output`` or the accumulated text buffer.

        Never raises -- a delivery failure must not abort generation.
        """
        hooks = getattr(self, "_ui_hooks", None)
        if hooks is None or not delta.data:
            return
        try:
            hooks.on_tool_output(
                agent_id=self._agent_id,
                call_id=MODEL_MEDIA_CALL_ID,
                chunk=delta.transcript or "",
                stream_id=self._model_media_stream_id(delta),
                sequence=delta.sequence,
                mime_type=delta.mime_type,
                data_b64=_b64encode(delta.data).decode("ascii"),
                final=delta.final,
            )
        except Exception:  # noqa: BLE001
            self._trace(
                f"MODEL_MEDIA: client delivery failed for a "
                f"{delta.mime_type!r} chunk"
            )

    def _emit_withheld_attachments_to_clients(
        self, result: ToolResult, attachments: List[Any]
    ) -> None:
        """Deliver model-unconsumable attachments to subscribed clients.

        The counterpart to :meth:`_gate_one_tool_result` stripping them:
        rather than destroying the bytes, publish them as CLIENT-audience
        media on the tool-output channel, correlated by the result's
        ``call_id``.  Each attachment is one single-chunk stream
        (``sequence=0``), and ``final`` is set on the last one so a client
        knows when to stop waiting.

        Never raises: a client-delivery problem must not turn a successful
        tool call into a failed one.  A failure is traced, not surfaced.

        Args:
            result: The tool result being gated; supplies ``call_id``.
            attachments: The attachments withheld from the model.
        """
        # ``getattr``: the gate also runs on sessions constructed without
        # the UI-hooks attribute at all (bare/unit-test construction), and
        # a missing sink is "nobody to deliver to", not an error.
        hooks = getattr(self, "_ui_hooks", None)
        if not attachments or hooks is None:
            return
        call_id = getattr(result, "call_id", "") or ""
        if not call_id:
            return
        stream_id = f"gated:{call_id}"
        last = len(attachments) - 1
        for index, att in enumerate(attachments):
            data = getattr(att, "data", None)
            mime_type = getattr(att, "mime_type", None)
            if not data or not mime_type:
                continue
            try:
                payload = (
                    data if isinstance(data, str)
                    else _b64encode(data).decode("ascii")
                )
                hooks.on_tool_output(
                    agent_id=self._agent_id,
                    call_id=call_id,
                    chunk=getattr(att, "display_name", "") or "",
                    stream_id=stream_id,
                    sequence=index,
                    mime_type=mime_type,
                    data_b64=payload,
                    final=(index == last),
                )
            except Exception:  # noqa: BLE001
                self._trace(
                    f"MODALITY_GATE: client delivery failed for a "
                    f"{mime_type!r} attachment on tool {result.name!r}"
                )
    def _resolve_withheld_target(
        self, withheld: Dict[str, int]
    ) -> Tuple[Optional[str], List[str], List[str]]:
        """Pick a tier the agent could actually switch to for withheld content.

        Args:
            withheld: modality kind -> count, as gathered by
                :meth:`_gate_one_tool_result`.

        Returns:
            ``(target, covered, stuck)``:

            * ``target`` — a tier declaring one of the withheld roles
              INBOUND that is **not** the tier the agent is already in, or
              ``None``.
            * ``covered`` — which withheld kinds that target actually
              accepts (naming the rest would send the agent back for
              content it still cannot see).
            * ``stuck`` — kinds whose ONLY declaring tier is the active
              one.  That is the self-referential case: the tier claims the
              role but its model can't fill it, so there is nothing to
              switch to and the note has to say something else entirely.

        The active tier is excluded because naming it produces a LOOP, not
        merely a poor message: ``enter_tier`` on the current tier is a
        documented no-op (``already_at_tier``), the agent re-runs the tool,
        and the gate emits the identical note — terminating only on the
        turn budget.  It is reachable exactly where
        :meth:`_validate_modality_tier_capabilities` declines to check (a
        tier on another provider), so the one gap in startup validation is
        the one the runtime backstop handled worst.
        """
        tier_config = self._tier_config
        if tier_config is None:
            return None, [], []
        from .model_tiers import DIRECTION_INBOUND

        active = getattr(self, "_active_tier", None)
        target: Optional[str] = None
        stuck: List[str] = []
        for kind in sorted(withheld):
            candidates = tier_config.tiers_for_modality(kind, DIRECTION_INBOUND)
            usable = [c for c in candidates if c != active]
            if candidates and not usable:
                stuck.append(kind)
            if usable and target is None:
                target = usable[0]
        covered = (
            [k for k in sorted(withheld)
             if k in tier_config.tiers[target].inbound_modalities]
            if target is not None else []
        )
        return target, covered, stuck

    def _build_withheld_attachment_note(self, withheld: Dict[str, int]) -> str:
        """Build the actionable note appended to a gated tool result.

        Three outcomes, in order:

        1. **A switchable tier exists** — name it, and name only the kinds
           it actually accepts.  What it does not cover is reported in two
           separate clauses, because "nothing accepts this" and "the only
           tier accepting this is the one you are in" are different facts
           and merging them makes the first one false.
        2. **The only tier declaring the role is the one the agent is in**
           — its model can't fill the role it claims.  Say that, rather
           than "no tier declares it" (false) or naming the active tier
           (a loop).  This is a profile bug and the note should read like
           one.
        3. **Nothing declares it** — say so, and point at the two fixes.

        The tier is found by ROLE, not by name
        (``ModelTierConfig.tiers_for_modality``), so a profile whose image
        tier is called something other than ``vision`` still produces an
        actionable note — and so the note generalises to audio / video /
        PDF the moment a tier declares those roles.  A tier literally named
        ``vision`` that declares no ``modalities`` still implies ``image``,
        so profiles written before the key behave unchanged.
        """
        kinds = ", ".join(sorted(withheld))
        model = self._model_name or "the current model"
        active = getattr(self, "_active_tier", None)
        target, covered, stuck = self._resolve_withheld_target(withheld)

        if target is not None:
            covers = ", ".join(sorted(covered))
            # What the suggested tier does NOT cover splits two ways, and
            # conflating them repeats the falsity the `stuck` branch below
            # exists to avoid: a kind whose only declaring tier is the one
            # the agent is in IS declared, so "no tier accepts it" is a lie
            # — it just doesn't work.
            rest = sorted(set(withheld) - set(covered))
            undeclared = [k for k in rest if k not in stuck]
            blocked = [k for k in rest if k in stuck]
            clauses = []
            if undeclared:
                clauses.append(
                    f"No tier accepts {', '.join(undeclared)} content.")
            if blocked:
                clauses.append(
                    f"The only tier declaring {', '.join(blocked)} is "
                    f"{active!r}, which you are already in — its model does "
                    f"not accept that input."
                )
            tail = ("  " + "  ".join(clauses)) if clauses else ""
            return (
                f"[Attachment withheld: the active model ({model}) can't "
                f"view {kinds} content.  Call enter_tier(\"{target}\") first "
                f"to view the {covers} content, then re-run this tool.{tail}]"
            )

        if stuck:
            return (
                f"[Attachment withheld: the active model ({model}) can't "
                f"view {kinds} content.  The only tier declaring "
                f"{', '.join(stuck)} is {active!r}, which you are already "
                f"in — its model does not accept that input, so switching "
                f"cannot help.  This is a profile error: map that tier to a "
                f"model accepting {', '.join(stuck)} input, or set "
                f"plugin_configs.<provider>.modalities to assert it.]"
            )

        return (
            f"[Attachment withheld: the active model ({model}) can't view "
            f"{kinds} content, and this session declares no tier that can.  "
            f"Use a model that accepts {kinds} input, or declare a tier with "
            f"`modalities: [{sorted(withheld)[0]}]` in the profile's "
            f"model_tiers.]"
        )

    def _validate_modality_tier_capabilities(self) -> None:
        """Fail loud when a tier declaring a modality role maps to a model
        the provider can't confirm accepts that input.

        Mirrors ``get_context_limit()``'s fail-fast at provider-resolution
        time: invoked once from :meth:`_ensure_provider` when the provider
        is first created, before any model work — the earliest point the
        provider exists (it's lazy-created).  No-op unless some tier
        declares ``modalities`` (which a tier named ``vision`` does
        implicitly).

        This is an earlier-warning over the content-boundary gate (which
        would otherwise surface the misconfiguration only at the first
        image): a tier claiming a role its model can't fill is a config
        error worth catching at startup, with an actionable message.

        Checks the role, not the name, so a differently-named image tier is
        validated too — the defect this closes is that renaming ``vision``
        silently disabled the check.

        Covers only tiers on the ACTIVE provider — see the comment in the
        loop for why, and for what does (and does not) back up the rest.

        Inbound roles are checked against ``provider.supports_modality``.
        Outbound roles are checked only when the provider implements
        ``supports_output_modality``; none do yet, so an outbound role is
        currently left unverified here rather than failing falsely — its
        inertness is reported by ``jaato-scaffold validate`` instead.

        Raises:
            ModelTierConfigError: A tier declares a modality its model
                doesn't accept (or, where checkable, can't emit) and no
                ``modalities`` knob asserts.
        """
        tier_config = self._tier_config
        provider = self._provider
        if tier_config is None or provider is None:
            return
        from .model_tiers import ModelTierConfigError
        for tier_name in tier_config.ordered_tier_names():
            entry = tier_config.tiers[tier_name]
            if not entry.declares_any_modality:
                continue
            # A tier on a DIFFERENT provider is NOT checked here — validating
            # it would eagerly create that provider (paying its init cost on
            # turn 1 even if the tier is never entered).  Only same-provider
            # tiers (the active provider owns the model) are fail-fast
            # checked at startup.
            #
            # There is NO lazy check on entry to make up for it: switch_tier
            # -> _connect_tier_entry -> provider.connect(model,
            # skip_model_test=True) touches no modality.  The runtime
            # content gate is the ONLY backstop for a cross-provider role,
            # so such a misconfiguration surfaces at the first piece of
            # content rather than at startup.  (An earlier comment here
            # claimed a lazy check existed; it never did.)
            if entry.provider and entry.provider != self._active_provider_name:
                continue
            provider_name = getattr(provider, "name", "the provider")
            for kind in sorted(entry.inbound_modalities):
                if provider.supports_modality(kind, model=entry.model):
                    continue
                raise ModelTierConfigError(
                    f"The {tier_name!r} tier declares the {kind!r} modality "
                    f"inbound but maps to {entry.model!r} ({provider_name}), "
                    f"which does not declare {kind} input.  Map the tier to a "
                    f"{kind}-capable model, or set "
                    f"plugin_configs.{provider_name}.modalities: "
                    f'["text", "{kind}"] to assert it.'
                )
            # Outbound is verified only against a provider that can answer.
            # No provider implements ``supports_output_modality`` yet (the
            # output half of the catalog's ``architecture.modality`` is still
            # discarded), so this is a forward hook: absent the method the
            # role is left unverified rather than failing falsely.  The
            # inertness itself is surfaced by ``jaato-scaffold validate``,
            # not by refusing to start.  See docs/design/binary-media-chunks.md.
            supports_out = getattr(provider, "supports_output_modality", None)
            if supports_out is None:
                continue
            for kind in sorted(entry.outbound_modalities):
                if supports_out(kind, model=entry.model):
                    continue
                raise ModelTierConfigError(
                    f"The {tier_name!r} tier declares the {kind!r} modality "
                    f"outbound but maps to {entry.model!r} ({provider_name}), "
                    f"which does not declare {kind} output.  Map the tier to "
                    f"a model that can emit {kind}, or drop the outbound role."
                )

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
        tool_results = self._gate_tool_results_for_active_modalities(tool_results)
        tool_result_parts = [Part(function_response=r) for r in tool_results]
        self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

        # Probe B (force_narration_between_tools, 2026-06-09).  Empirical
        # finding from kb cascade context-stage falsification on
        # qwen3-14b @ temp=0 (see
        # ``feedback_small_model_narration_skipping_is_structural``):
        # small models in tool-mode skip narration regardless of persona
        # prose AND in-context examples.  Structural failure that
        # framework-side forcing is the right level to fix.
        #
        # Approach: after each tool_result append, inject a synthetic
        # ``USER``-role message asking the model to extract observations
        # in 1-2 sentences before continuing.  Model's response is
        # naturally text-mode (user just asked for text) + may include
        # the next tool call.  Loop semantics unchanged.
        #
        # Gated by ``profile.quirks.force_narration_between_tools``
        # threaded through ``provider.extra["quirks"]`` (the canonical
        # per-profile quirk mechanism, symmetric with
        # ``force_tool_choice_for_lifecycle`` and
        # ``coerce_typed_tool_args``).  Profile-scoped so the qwen3-14b
        # narration-skipping quirk doesn't leak to haiku /
        # openrouter / other profile sets.
        force_narration = getattr(
            self._provider, "_force_narration_between_tools", False
        )
        if force_narration:
            narration_prompt = Message(
                role=Role.USER,
                parts=[Part(text=(
                    "Briefly extract what you observed from the previous "
                    "tool result in 1-2 sentences, then continue with your "
                    "next action."
                ))],
            )
            self._history.append(narration_prompt)
            self._trace(
                f"FORCE_NARRATION: injected synthetic user prompt after "
                f"{len(tool_results)} tool result(s)"
            )

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

                def streaming_callback(chunk) -> None:
                    if isinstance(chunk, MediaDelta):
                        self._deliver_model_media(chunk)
                        return
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

                # Path 1 quirk consumption: if signal_completion just
                # returned validation_failed, request named-function
                # tool_choice on this retry (provider decides whether
                # to honor via its own quirk gate).  Consumed inline
                # so the stamp clears for ALL subsequent calls in this
                # turn — single xgrammar-enforced retry is the
                # contract.  Only PASSED to provider.complete when
                # set: providers that don't yet accept the
                # ``tool_choice`` kwarg (most of them; only vllm
                # honors it today) never see an unknown kwarg in the
                # default no-quirk path.  The Protocol declares the
                # kwarg per
                # ``shared/plugins/model_provider/base.py:complete``;
                # explicit per-provider acceptance can land in a
                # follow-up sweep.
                _retry_tool_choice = self._consume_pending_tool_choice()
                _extra_complete_kwargs: Dict[str, Any] = {}
                if _retry_tool_choice is not None:
                    _extra_complete_kwargs["tool_choice"] = _retry_tool_choice
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
                            **_extra_complete_kwargs,
                        ),
                        context="complete_tool_results_streaming",
                        on_retry=self._on_retry,
                        cancel_token=self._cancel_token,
                        provider=self._provider
                    )
            else:
                _retry_tool_choice = self._consume_pending_tool_choice()
                _extra_complete_kwargs = {}
                if _retry_tool_choice is not None:
                    _extra_complete_kwargs["tool_choice"] = _retry_tool_choice
                with self._provider_access():
                    turn_result, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._get_effective_system_instruction(),
                            tools=self._get_tools_for_provider(),
                            **_extra_complete_kwargs,
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

                def streaming_callback(chunk) -> None:
                    if isinstance(chunk, MediaDelta):
                        self._deliver_model_media(chunk)
                        return
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
        # Executor returns (ok, result_dict) tuple, or a bare value.
        ok, result_data = _split_executor_result_impl(executor_result)

        # Mark results from untrusted-content tools (web_fetch / web_search /
        # MCP) so the provider converter wraps the model-facing text in the
        # untrusted-content boundary — indirect-prompt-injection mitigation.
        # See TRAIT_UNTRUSTED_CONTENT.
        from jaato_sdk.plugins.model_provider.types import TRAIT_UNTRUSTED_CONTENT
        _untrusted = bool(
            self._runtime.registry
            and TRAIT_UNTRUSTED_CONTENT in self._runtime.registry.get_tool_traits(fc.name)
        )
        _untrusted_source = fc.name if _untrusted else None

        # Check for multimodal result
        attachments: Optional[List[Attachment]] = None
        if isinstance(result_data, dict) and result_data.get('_multimodal'):
            attachments = _extract_multimodal_attachments_impl(result_data)
            result_data = {k: v for k, v in result_data.items()
                          if not k.startswith('_multimodal')
                          and k not in ('image_data', 'file_data')}

        # String results pass through directly so converters never
        # JSON-encode them (which would escape quotes, backslashes, etc.).
        if isinstance(result_data, str):
            # Run string-level enrichment (template extraction, etc.)
            enrichment_metadata: Optional[Dict[str, Any]] = None
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
                if enrichment.metadata:
                    enrichment_metadata = enrichment.metadata

            return ToolResult(
                call_id=fc.id,
                name=fc.name,
                result=result_data,
                is_error=not ok,
                attachments=attachments,
                enrichment_metadata=enrichment_metadata,
                untrusted=_untrusted,
                untrusted_source=_untrusted_source,
            )

        # Normalize the payload into the model-facing form: wrap non-dicts,
        # surface the permission advisory note, strip internal '_' keys, and
        # collapse single-key error dicts to a bare string.
        result_dict = _normalize_result_dict_impl(result_data, ok=ok)

        # Run tool result enrichment (e.g., template extraction)
        enrichment_metadata: Optional[Dict[str, Any]] = None
        if ok and self._runtime.registry:
            result_dict, enrichment_metadata = self._enrich_tool_result_dict(
                fc.name, result_dict, tool_args=fc.args
            )
            if not enrichment_metadata:
                # Distinguish "enrichment ran, produced nothing" from
                # "enrichment didn't run" — both leave None on the
                # ToolResult so processors don't get a misleading {}.
                enrichment_metadata = None

        return ToolResult(
            call_id=fc.id,
            name=fc.name,
            result=result_dict,
            is_error=not ok,
            attachments=attachments,
            enrichment_metadata=enrichment_metadata,
            untrusted=_untrusted,
            untrusted_source=_untrusted_source,
        )

    def _reconcile_unanswered_calls(
        self,
        finish_reason: Optional[FinishReason] = None,
    ) -> int:
        """Answer tool calls the turn abandoned, so history stays valid.

        THE INVARIANT.  Every ``tool_use`` block in history must have a
        matching ``tool_result``.  OpenAI/Azure-shaped upstreams enforce
        it on the *next* request and reject the whole conversation when
        it does not hold::

            No tool output found for function call call_mAyQ...

        THE HOLE THIS FILLS (#751).  A turn severed by the output cap
        can carry a **complete, well-formed** tool call: the arguments
        parsed cleanly, so #750's unreadable-arguments refusal never
        sees it, and the abnormal-finish path ends the turn before the
        call is ever dispatched.  The assistant message is in history,
        the call has no output, and the session is dead from the next
        request onward -- not degraded, stopped.

        Two neighbouring mechanisms deliberately do NOT cover it:

        * :meth:`_maybe_rewind` fires only when
          ``detect_truncated_tool_call`` recognises a *damaged* call
          (empty or incomplete arguments), and drops it from history
          when it does.  A complete call is not damaged, so no rewind.
        * ``unreadable_arguments_error`` keys on
          ``FunctionCall.unreadable_args``, which a complete call does
          not carry.

        WHAT IT WRITES.  A tool result per abandoned call, in the same
        shape a failed execution produces, saying the call was not run
        and why (see :func:`unexecuted_call_error`).  That is both the
        thing the contract needs and the place the model already looks
        for the outcome of the call it just made -- which is why the
        truncation nudge belongs here rather than in a free-floating
        user-role message.

        Reads HISTORY rather than the response, because history is what
        the next request is built from: a call the rewind path already
        dropped must not be answered, and a call that was executed
        normally already has its answer.  Only a trailing ``MODEL``
        message can hold unanswered calls, so the method is a no-op --
        and therefore idempotent -- once the results are appended.

        Args:
            finish_reason: Why the turn ended, passed through to the
                synthesised result so the model is told the cause.

        Returns:
            How many calls were answered (0 when there was nothing to
            reconcile).
        """
        messages = self._history.messages
        if not messages:
            return 0
        last = messages[-1]
        if last.role != Role.MODEL:
            return 0
        fcs = [p.function_call for p in last.parts if p.function_call]
        if not fcs:
            return 0

        tool_results = [
            ToolResult(
                call_id=fc.id,
                name=fc.name,
                result=unexecuted_call_error(fc, finish_reason),
                is_error=True,
            )
            for fc in fcs
        ]
        tool_results = self._gate_tool_results_for_active_modalities(
            tool_results
        )
        self._history.append(Message(
            role=Role.TOOL,
            parts=[Part(function_response=r) for r in tool_results],
        ))
        self._trace(
            f"RECONCILE_UNANSWERED: {len(fcs)} call(s) abandoned by "
            f"finish={getattr(finish_reason, 'value', finish_reason)}: "
            f"{[fc.name for fc in fcs]}"
        )
        return len(fcs)

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
        tool_results = self._gate_tool_results_for_active_modalities(tool_results)
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

        Returns:
            Tuple of ``(enriched_dict, combined_metadata)``.  ``combined_metadata``
            is the per-plugin enrichment metadata dict (``{plugin_name: meta}``)
            produced by ``registry.enrich_tool_result``.  Callers stash it on
            ``ToolResult.enrichment_metadata`` so completion processors see it
            via ``context.tool_calls[i].enrichment_metadata`` (see
            ``build_tool_call_ledger``).

        Args:
            tool_name: Name of the tool that produced the result.
            result_dict: The result dictionary to enrich.
            tool_args: Optional tool call arguments for context-aware enrichment.

        Returns:
            Tuple ``(enriched_dict, combined_metadata)`` — see method docstring.
        """
        enriched_dict = result_dict.copy()
        # Aggregate metadata across BOTH the file_writer single-call path
        # and the text-fields multi-call path.  Later calls overwrite
        # earlier per-plugin metadata for the same plugin name; in
        # practice each plugin contributes once per call so this is
        # last-write-wins on intentional duplicates only.
        combined_metadata: Dict[str, Any] = {}

        # Tools declaring the file_writer trait get full-JSON enrichment
        # (LSP diagnostics, artifact tracking, etc.).  Tools declaring the
        # greppable_content trait take the SAME full-dict path so that
        # result-rewriter enrichment plugins (e.g. result_grep) can inspect
        # and shrink structured payloads (call_service.body/headers) that the
        # text-field path below never sees.  Both route the whole result dict
        # through enrich_tool_result; the field-level path only fires for the
        # six well-known text keys, missing structured dicts entirely.
        from jaato_sdk.plugins.model_provider.types import (
            TRAIT_FILE_WRITER,
            TRAIT_GREPPABLE_CONTENT,
        )
        tool_traits = self._runtime.registry.get_tool_traits(tool_name)

        if TRAIT_FILE_WRITER in tool_traits or TRAIT_GREPPABLE_CONTENT in tool_traits:
            # Pass full result as JSON so enrichers see the entire payload
            # (LSP file-path extraction for file_writer; full-body grep for
            # greppable_content).
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
            if enrichment.metadata:
                combined_metadata.update(enrichment.metadata)
            return enriched_dict, combined_metadata

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
                    if enrichment.metadata:
                        combined_metadata.update(enrichment.metadata)

        return enriched_dict, combined_metadata

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

    def _track_streaming_usage(
        self,
        turn_data: Dict[str, Any],
        usage: TokenUsage,
    ) -> None:
        """Record a streaming usage CHUNK onto the turn.

        The streaming path fires this once per usage chunk, and a provider
        may emit several per response.  So everything here REPLACES: these
        are level readings (end-of-turn context size), and the last chunk
        wins.

        **Nothing here may write a ``spend_`` key.**  Spend accumulates,
        and accumulating per chunk would count one response many times.
        The spend keys are written in exactly one place —
        :meth:`_accumulate_turn_tokens`, which runs once per response on
        every path.  See ``docs/design/model-tier-prompt-cache.md`` §5.4.

        Extracted from the closure it used to live in so that rule is
        testable by EFFECT rather than only by shape: a shape check reads
        one function body, so moving a write into a helper the callback
        calls makes it silent, while driving this with two chunks and
        looking at ``turn_data`` catches the write wherever it hides.

        Args:
            turn_data: The turn-accounting dict, mutated in place.
            usage: The usage chunk just received.
        """
        if usage.total_tokens > 0:
            turn_data['prompt'] = usage.prompt_tokens
            turn_data['output'] = usage.output_tokens
            turn_data['total'] = usage.total_tokens
        # Cache tokens: capture when present (streaming path)
        if usage.cache_read_tokens is not None:
            turn_data['cache_read'] = usage.cache_read_tokens
        if usage.cache_creation_tokens is not None:
            turn_data['cache_creation'] = usage.cache_creation_tokens

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

        Cache token fields come in BOTH shapes, for the same reason the
        token counts do:

        * ``cache_read`` / ``cache_creation`` are REPLACED, matching
          prompt/output/total, and are also written by the streaming
          usage-callback (which fires per usage CHUNK, so it must not sum).
        * ``spend_cache_read`` / ``spend_cache_creation`` ACCUMULATE, and
          are written ONLY here — the once-per-response hook — exactly as
          ``spend_*`` and ``cost_usd`` are, and for the same reason: every
          response in a turn is separately billed for what it read from and
          wrote to the cache.

        The spend pair is the one that answers "what did this turn cost in
        cache traffic".  Replacing was actively misleading for a turn that
        switches model tier mid-flight: the switch re-reads the whole prefix
        cold at the new model, and reporting only the final leg hid exactly
        the miss the switch caused.  See
        ``docs/design/model-tier-prompt-cache.md`` §5.4.
        """
        if response.usage.total_tokens > 0:
            turn_tokens['prompt'] = response.usage.prompt_tokens
            turn_tokens['output'] = response.usage.output_tokens
            turn_tokens['total'] = response.usage.total_tokens
            # SPEND accumulates where the replaced fields do not.  This is
            # the correct hook precisely because it runs exactly ONCE per
            # response on every path; the streaming usage-callback does not
            # (it is streaming-only, and a provider emitting more than one
            # usage chunk per response would double-count there).
            turn_tokens['spend_total'] = (
                turn_tokens.get('spend_total', 0) + response.usage.total_tokens)
            turn_tokens['spend_prompt'] = (
                turn_tokens.get('spend_prompt', 0) + response.usage.prompt_tokens)
            turn_tokens['spend_output'] = (
                turn_tokens.get('spend_output', 0) + response.usage.output_tokens)
            # Cost accumulates for the SAME reason spend does: a turn with a
            # tool call has >= 2 responses and each is billed, so replacing
            # would report only the last one.  Stays None when the provider
            # reports nothing, so "no cost reported" and "the cost was zero"
            # remain different answers.
            if response.usage.cost_usd is not None:
                turn_tokens['cost_usd'] = (
                    (turn_tokens.get('cost_usd') or 0.0)
                    + response.usage.cost_usd)

        # Cache tokens: the level reading (last response) and the spend
        # reading (every response) — see the docstring for why both exist.
        if response.usage.cache_read_tokens is not None:
            turn_tokens['cache_read'] = response.usage.cache_read_tokens
            turn_tokens['spend_cache_read'] = (
                turn_tokens.get('spend_cache_read', 0)
                + response.usage.cache_read_tokens)
        if response.usage.cache_creation_tokens is not None:
            turn_tokens['cache_creation'] = response.usage.cache_creation_tokens
            turn_tokens['spend_cache_creation'] = (
                turn_tokens.get('spend_cache_creation', 0)
                + response.usage.cache_creation_tokens)

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

    def get_budget_usage(self, *, tracker_only: bool = False) -> Dict[str, float]:
        """This session's ABSOLUTE budget consumption, per dimension.

        The authoritative figure: it is what the per-session
        :class:`BudgetTracker` accumulated per RESPONSE, which is the same
        number the tracker's own percentage reports come from.  Callers
        reconcile a cascade pool against this rather than summing an event
        stream — events have proven both duplicable (turn.progress re-emits)
        and droppable (a cancelled turn's TurnCompletedEvent).

        Falls back to summing ``spend_total`` over ``turn_accounting`` when
        the session has no tracker (unbudgeted sessions still contribute to
        a cascade pool).  Returns ``{}`` when neither source has anything.

        ``tracker_only=True`` suppresses that fallback and is REQUIRED for
        persistence.  The two branches return incompatible shapes under one
        name: the tracker yields every declared dimension, the fallback
        yields a single synthetic ``tokens`` key.  A caller that persists
        the fallback silently OVERWRITES a real snapshot with one that can
        never satisfy a ``turns`` / ``usd`` / ``seconds`` ceiling -- so an
        unbudgeted reload does not merely stop enforcing, it destroys the
        snapshot the next reload would have enforced from.  Observed live
        2026-08-23: a session file holding ``{"tokens": 247004.0}`` where
        five dimensions had been in flight, which is the fallback's
        fingerprint and the proof the reloaded session had no tracker.

        Cascade-pool reconciliation still wants the fallback (an unbudgeted
        child spends real tokens against the shared pot), so it stays --
        behind an explicit opt-in rather than as the silent default.
        """
        if self._budget_tracker is not None:
            return self._budget_tracker.usage.as_dict()
        if tracker_only:
            return {}
        spend = sum(
            int(t.get("spend_total", 0) or 0) for t in self._turn_accounting
        )
        return {"tokens": float(spend)} if spend else {}

    def budget_exhausted_reason(self) -> Optional[str]:
        """Why this session stopped at its budget ceiling, or ``None``.

        Set only by an ``abort`` rung.  Read by the runner's
        ``session.send_message`` handler so the RPC result carries a TYPED
        exhaustion signal: without it a ceiling announced itself only in prose
        ("[Generation cancelled (...)]" and a system line), so a driver could
        not distinguish "stopped at the ceiling" from a normal finish without
        substring-matching -- the parse-the-log shape budgets exist to replace.
        """
        return self._budget_exhausted_reason

    def was_last_send_refused(self) -> bool:
        """True when the last ``send_message`` was refused by the budget gate.

        Read by the runner RPC handler to SUPPRESS the post-turn
        ``TurnCompletedEvent``.  A refused turn never ran, so reporting it as
        completed is doubly wrong: the handler sources its payload from
        ``turn_accounting[-1]``, and a refused turn appends nothing — so the
        event would carry the PREVIOUS turn's tokens and duration again.  A
        client counting turns over-counts; one summing tokens double-counts.

        The refusal is still visible to the client as an
        ``on_output("system", ...)`` line, so suppressing the event loses no
        information.

        NOT the mechanism that actually protects this today, despite what the
        wording above implies.  ``rpc._forward_post_turn_hooks`` gates on a NEW
        turn having landed in ``turn_accounting`` -- strictly stronger, since
        it covers every no-op path rather than just a budget refusal -- so a
        refused turn already emits nothing.  This accessor has no consumer and
        is kept only because a caller may want to ASK whether the last send
        was refused; do not add a second suppression path on top of it.
        """
        return self._last_send_refused

    def restore_budget_usage(self, usage: Dict[str, float]) -> None:
        """Re-seed this session's budget usage after a reload.

        Counterpart to :meth:`get_budget_usage`, which existed read-only for
        cascade-pool reconciliation.  Without a restore, an unloaded session
        came back with a zeroed tracker -- see
        :meth:`BudgetTracker.restore_usage` for why that silently disabled
        every cross-turn ceiling.

        No-op when the session runs unbudgeted.
        """
        if self._budget_tracker is None or not usage:
            return
        self._budget_tracker.restore_usage(usage)
        logger.info("budget: restored usage after reload: %s", usage)

    def restore_budget_exhausted(self, reason: Optional[str]) -> None:
        """Re-assert a ceiling that had already stopped this session.

        Usage alone was not enough.  A reloaded session held usage AT its
        limit with no memory of having been aborted, so it served one more
        turn: the rung does re-fire, but from ``_budget_observe_turn`` in the
        turn's ``finally`` -- after the turn it was supposed to prevent.  A
        goal that finished inside that turn passed a ceiling that had already
        fired.

        Restoring the latch makes the refusal land at turn START, which is
        where a ceiling has to act.

        No-op for an unbudgeted session, and for a session that was never
        stopped -- absence must not be turned into a refusal.
        """
        if not reason:
            return
        self._budget_exhausted_reason = reason
        logger.info("budget: restored exhaustion latch after reload: %s", reason)

    def _refuse_if_budget_exhausted(self) -> Optional[str]:
        """Return a refusal reason when an ``abort`` rung has already fired.

        ``abort`` wires to :meth:`request_stop`, which is a COOPERATIVE
        cancel of the in-flight turn — it neither terminates the session nor
        refuses later input.  Combined with rung latching (the 100% rung
        fires once and never again), that made every turn after the first
        abort effectively unbudgeted: a client that simply kept sending ran
        a ``turns: 4`` budget to 8 turns.

        So the ceiling is enforced HERE instead: once exhausted, the session
        refuses further turns rather than silently serving them.  Budget
        exhaustion means "this session is done", not "cancel this turn".
        """
        return self._budget_exhausted_reason

    def _budget_observe_response(self, response: ProviderResponse) -> None:
        """Feed one model response's spend to the tracker, then apply rungs.

        Tokens come from the response's own usage; cost is resolved through
        the SAME precedence the telemetry span uses
        (``_resolve_span_cost``: provider-reported -> pricing table -> None).
        A ``None`` cost leaves the ``usd`` dimension untouched — a budget
        must never hard-stop on a number it invented.
        """
        if self._budget_tracker is None:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        try:
            fired = self._budget_tracker.observe(
                tokens=int(getattr(usage, "total_tokens", 0) or 0),
                usd=self._resolve_span_cost(usage),
            )
            self._apply_budget_rungs(fired)
        except Exception as exc:  # noqa: BLE001
            # Budgeting is a guardrail, not part of the turn's contract —
            # never let it break a live turn.
            logger.warning("budget: response observation failed: %s", exc)

    def _budget_observe_turn(self, turn_data: Dict[str, Any]) -> None:
        """Feed one completed turn's wall-clock / tool-call / turn count."""
        if self._budget_tracker is None:
            return
        try:
            fired = self._budget_tracker.observe(
                turns=1,
                seconds=turn_data.get("duration_seconds") or 0.0,
                tool_calls=len(turn_data.get("function_calls") or ()),
            )
            self._apply_budget_rungs(fired)
        except Exception as exc:  # noqa: BLE001
            logger.warning("budget: turn observation failed: %s", exc)

    def _apply_budget_rungs(
        self, fired, origin: str = "self-enforced",
        pressure: Optional[str] = None,
    ) -> None:
        """Apply the degrade rungs that just crossed their threshold.

        ``origin`` names the MECHANISM, not whose ladder it was:

        * ``"self-enforced"`` — this session's own tracker crossed its own
          limit.  Note the ladder itself may have been INHERITED from the
          parent (a child that declared no budget takes the parent's), so
          "self-enforced" means "my tracker tripped it", not "my policy".
        * ``"cascade-pushed"`` — the shared pool crossed and the rung was
          pushed here from the daemon.

        Mechanism is the distinction a consumer actually needs, and it is
        why the marker exists at all: "I hit my own ceiling" invites a
        narrower retry, "the shared pot ran out" means the run is winding
        down and retrying is pointless.  Both paths land in this function
        and would otherwise emit identical lines.

        Earlier values were ``"session"`` / ``"cascade"``, which read as
        "whose ladder" — actively wrong for a profileless child
        self-enforcing the PARENT's ladder, which would have been labelled
        ``session``.  Same class as reporting a child's own pressure
        against a pool-triggered rung: defensible if you know the
        internals, wrong if you do not.

        Two effects, per ``docs/design/budget-control-degradation.md``:

        * **Overlay** — rebind entries in the live tier table (a brownout).
          The tier vocabulary and the model's cognitive role are untouched;
          only the model each tier points at changes.  When the rebind hits
          the tier the session is CURRENTLY in, the provider is re-connected
          immediately — ``switch_tier`` alone would not, because the tier
          name has not changed (see ``_is_connected_to``).
        * **Terminal action** — ``abort`` stops the run via the existing
          cooperative cancel; ``finalize`` / ``escalate`` are latched on
          ``_budget_terminal_action`` and surfaced, for the reactor layer to
          act on (that layer already owns agent-directed actions such as
          prompt injection).
        """
        if not fired:
            return
        from .budget_control import ACTION_ABORT, overlay_tier_table

        for rung in fired:
            # Rungs apply at most once and IN ORDER, per ladder — not per
            # tracker.  A pooled child runs the parent's ladder on its own
            # tracker AND receives pushes of the same ladder from the pool,
            # and the pool crosses first under concurrent spawn (each child
            # is handed the full remainder, so the pot depletes ~N times
            # faster than any one child).  Its own lower rung therefore
            # fires LATER in wall-clock and, unguarded, rebinds the tier back
            # to a model the cascade had already degraded away from — onto a
            # pricier one, at the moment the pot is most exhausted.
            #
            # Safe to compare thresholds across sources because under the
            # separate-budgets rule the two sources are always the SAME
            # ladder: a child that declared a budget keeps its own and is
            # never pushed; a child that declared none inherits the parent's,
            # which is exactly what gets pushed.
            if rung.at_percent <= self._budget_applied_rung_pct:
                logger.info(
                    "budget[%s]: skipping rung at %.0f%% — %.0f%% already "
                    "applied; a lower rung would rebind BACKWARDS onto a "
                    "model already degraded away from",
                    origin, rung.at_percent, self._budget_applied_rung_pct,
                )
                continue
            self._budget_applied_rung_pct = rung.at_percent
            # A cascade rung fired on the POOL's fraction; reporting this
            # child's own usage instead reads as a contradiction ("degrading
            # at 50% (tokens 32%)").  Caller supplies the pool's pressure.
            detail = pressure or (
                self._budget_tracker.describe_pressure()
                if self._budget_tracker is not None else "cascade pressure"
            )
            tag = f"budget[{origin}]"
            if rung.model_tiers:
                if self._tier_config is None:
                    # Rejected by the profile validator, but a session can be
                    # built without going through it (inline specs, tests).
                    logger.warning(
                        "budget: degrade rung at %.0f%% declares a model_tiers "
                        "overlay but the session has no tier config; ignoring",
                        rung.at_percent,
                    )
                else:
                    changes = overlay_tier_table(
                        self._tier_config.tiers, rung.model_tiers)
                    if changes:
                        logger.info(
                            "budget[%s]: degrading at %.0f%% (%s) — rebound %s",
                            origin, rung.at_percent, detail,
                            "; ".join(f"{k}: {v}" for k, v in changes.items()),
                        )
                        self._reconnect_active_tier_if_rebound()
                        self._surface_budget_event(
                            f"{tag} {detail}: degraded "
                            + "; ".join(f"{k} {v}" for k, v in changes.items())
                        )
            if rung.action:
                self._budget_terminal_action = rung.action
                logger.info(
                    "budget[%s]: terminal action '%s' at %.0f%% (%s)",
                    origin, rung.action, rung.at_percent, detail,
                )
                self._surface_budget_event(
                    f"{tag} {detail}: {rung.action}")
                if rung.action == ACTION_ABORT:
                    # Latch FIRST: request_stop only cancels the IN-FLIGHT
                    # turn (cooperative cancel).  Without the latch the next
                    # send_message would run unbudgeted — the rungs are
                    # latched so the 100% rung never re-fires — and the
                    # "ceiling" would be a one-shot interrupt the client can
                    # simply talk past.  A ceiling that only cancels one turn
                    # is not a ceiling.
                    self._budget_exhausted_reason = (
                        f"budget_exhausted ({origin}: {detail})")
                    self.request_stop(self._budget_exhausted_reason)

    def apply_cascade_degrade(
        self, rungs: List[Dict[str, Any]],
        pool_pressure: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply degrade rungs pushed down because the CASCADE pool crossed.

        The mid-flight half of cascade budgets: a child already running when
        the shared pool crosses must degrade too, rather than keeping the
        ceiling it was handed at spawn.  A pool that only constrained
        children at spawn would not be a shared budget — a sibling burning
        the envelope has to affect everyone still running, which is the
        whole point of the ceiling being aggregate.

        Deliberately routed through the SAME :meth:`_apply_budget_rungs` a
        session's own ladder uses, so a pushed rung produces identical
        machinery — tier rebind, active-tier re-connect, abort latch — and
        an identical client notice, differing only by its ``origin`` tag.

        Args:
            rungs: Wire form (``DegradeRung.to_dict()``) — re-parsed here so
                the runner validates what it was handed rather than trusting
                the daemon's serialisation.

        Returns:
            ``{"applied": <n>}``, or ``{"applied": 0, "error": ...}`` when
            the payload will not parse.  Never raises: a budget push must
            not break the session it is trying to constrain.
        """
        from .budget_control import DegradeRung
        try:
            parsed = [
                DegradeRung.from_dict(r, index=i)
                for i, r in enumerate(rungs or [])
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("apply_cascade_degrade: bad payload: %s", exc)
            return {"applied": 0, "error": str(exc)}
        if not parsed:
            return {"applied": 0, "notices": []}
        notices: List[str] = []
        previous_sink = self._budget_notice_sink
        self._budget_notice_sink = notices
        try:
            self._apply_budget_rungs(
                tuple(parsed), origin="cascade-pushed",
                pressure=pool_pressure)
        except Exception as exc:  # noqa: BLE001
            logger.warning("apply_cascade_degrade: apply failed: %s", exc)
            return {"applied": 0, "error": str(exc), "notices": notices}
        finally:
            self._budget_notice_sink = previous_sink
        # Handed back rather than emitted: the DAEMON emits these, because
        # it is not turn-scoped and this push may have landed between turns.
        return {"applied": len(parsed), "notices": notices}

    def _reconnect_active_tier_if_rebound(self) -> None:
        """Re-point the provider when the ACTIVE tier's binding just changed.

        The load-bearing half of a brownout: without this the overlay would
        sit in the tier table and only take effect the next time the agent
        happened to leave and re-enter the tier.
        """
        if self._tier_config is None or self._active_tier is None:
            return
        try:
            _, entry = self._tier_config.model_for(self._active_tier)
            if self._is_connected_to(entry):
                return
            self._connect_tier_entry(entry)
            self._model_name = entry.model
            logger.info(
                "budget: active tier '%s' re-pointed at %s",
                self._active_tier, entry.model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("budget: re-connect after degrade failed: %s", exc)

    def _surface_budget_event(self, message: str) -> None:
        """Make a budget decision visible to the client (best-effort).

        Uses ``_current_output_callback`` — the SAME per-turn channel the
        rest of the session emits ``on_output("system", ...)`` through.

        An earlier version routed this through ``self._ui_hooks``, which is
        never set on the runner path (the live production path), so every
        budget decision was silently dropped and the only evidence a budget
        had acted at all was a server-side log line.  ``on_agent_output``
        appears nowhere else in this file — that was the tell.
        """
        rendered = f"[{message}]"
        # Collect first: the sink is the only channel that works when this
        # runs outside a turn (a cascade push), because both
        # ``_current_output_callback`` and the runner's ``_ui_hooks`` shim
        # are installed per-send and restored afterwards.
        if self._budget_notice_sink is not None:
            self._budget_notice_sink.append(rendered)
        callback = self._current_output_callback
        if callback is None:
            return
        try:
            callback("system", rendered, "write")
        except Exception:  # noqa: BLE001
            pass

    def _record_token_usage(self, response: ProviderResponse) -> None:
        """Record token usage to the ledger (if any) and the budget tracker.

        The budget observation runs FIRST and unconditionally: budgeting must
        not silently depend on a ledger being configured.
        """
        self._budget_observe_response(response)
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

        When the provider reports a cost (``usage.cost_usd``), also sets
        ``gen_ai.usage.cost`` (Langfuse OTLP cost ingestion) and
        ``llm.cost.total`` (OpenInference / Arize Phoenix).

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
        output_msgs = response_to_openinference(response)
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

        # Cost (USD). Precedence mirrors the daemon's core.py:_build_usage:
        #   1. provider-reported ``usage.cost_usd`` (e.g. claude_cli's
        #      total_cost_usd, OpenRouter's cost) — fiscal truth, wins.
        #   2. operator pricing table (.jaato/pricing.json) computed from
        #      the model name + token counts, so cost lands on the span even
        #      for providers that don't report it on the wire.
        #   3. None — no source knew; the observability backend may still
        #      compute cost from model + token counts (e.g. Langfuse's
        #      model-pricing catalog).
        # We resolve here, while the span is open — the daemon boundary that
        # populates UsageBreakdown.cost_usd runs after the span has closed.
        # Two keys are emitted so pre-computed cost renders in either backend:
        #   - ``gen_ai.usage.cost``  → Langfuse's OTLP cost ingestion
        #   - ``llm.cost.total``     → OpenInference (Arize Phoenix)
        # Both are cost attributes (not token-count buckets), so emitting both
        # does not trip Langfuse's inclusive/exclusive token-bucket contract.
        cost_usd = self._resolve_span_cost(usage)
        if cost_usd is not None:
            span.set_attribute("gen_ai.usage.cost", cost_usd)
            span.set_attribute("llm.cost.total", cost_usd)

        # Cache outcome classification (hit/partial/warm/miss/unknown)
        # so external observers can correlate cache behavior with the
        # GC ↔ cache coordination dance.
        try:
            outcome = classify_cache_outcome(
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

    def _resolve_span_cost(self, usage) -> Optional[float]:
        """Resolve per-call cost (USD) for an LLM telemetry span.

        Precedence mirrors ``JaatoServer._build_usage`` so the span and the
        emitted ``UsageBreakdown`` agree:

        1. ``usage.cost_usd`` — provider-reported; fiscal truth, wins.
        2. Operator pricing table (``.jaato/pricing.json`` via
           ``shared.pricing``) computed from the model name + token counts.
        3. ``None`` — no source knew (the backend may still estimate).

        The pricing table is loaded lazily on first non-reported cost and
        cached on the session, so cost-free sessions never touch the JSON.
        Any failure to load/compute degrades to ``None`` (telemetry must
        never break a turn).

        Args:
            usage: The response ``TokenUsage``.

        Returns:
            Cost in USD, or ``None`` when no source can supply it.
        """
        if usage.cost_usd is not None:
            return usage.cost_usd
        if not self._model_name:
            return None
        try:
            if not self._span_pricing_loaded:
                from shared.pricing import load_pricing
                self._span_pricing = load_pricing(self.workspace_path)
                self._span_pricing_loaded = True
            if self._span_pricing is None or not self._span_pricing.has(self._model_name):
                return None
            return self._span_pricing.cost_for_usage(
                self._model_name,
                prompt_tokens=int(usage.prompt_tokens or 0),
                output_tokens=int(usage.output_tokens or 0),
                cache_read_tokens=usage.cache_read_tokens,
                cache_creation_tokens=usage.cache_creation_tokens,
            )
        except Exception as e:  # pragma: no cover - defensive
            self._trace(f"LLM_TELEMETRY: pricing-table cost lookup failed: {e}")
            return None

    def _record_input_messages_telemetry(self, span) -> None:
        """Record OpenInference input messages on a telemetry span.

        Converts the current session history (messages being sent to the
        provider) into OpenInference ``llm.input_messages.*`` indexed
        attributes on the LLM span, prepended with the system instruction.

        The system prompt is NOT part of ``_history.messages`` — it reaches the
        provider as the API's separate top-level ``system`` parameter — so
        without prepending it here the largest, most important input is silently
        dropped from every span. Emitting it as ``input_messages[0]`` with role
        ``system`` is the standard OpenInference representation and reflects
        per-turn changes (dynamic instructions, injected reminders, budget
        suppression). ``self._system_instruction`` is the effective prompt
        (already resolved to any ``system_instruction_override``). Content is
        routed through :meth:`set_input_messages`' redactor, so it is blanked
        automatically when content redaction is on.

        Args:
            span: The LLM span context to set attributes on.
        """
        input_msgs = build_input_messages(
            self._system_instruction, self._history.messages
        )
        if input_msgs:
            span.set_input_messages(input_msgs)

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

    def flush_session_quiescent(self) -> None:
        """Emit the pending ``on_session_quiescent`` notification, if any.

        MUST be called by a turn driver AFTER it has fired
        ``on_agent_turn_completed`` for the turn that just ended.

        ``SessionTerminatedEvent`` is terminal by contract, so it has to be
        the LAST thing a consumer sees for that session.  It could not be
        while the session emitted it inline: quiescence is detected inside
        ``send_message``, and every driver fires the turn-completed hook
        after ``send_message`` returns.  The terminal event therefore
        preceded the final turn event of the turn it was reporting.

        Idempotent, and safe to call after every turn: it does nothing unless
        the agent called ``signal_completion`` during this one.

        A driver that forgets to call it emits no terminal event at all --
        deliberately louder than the alternative, which was emitting one too
        early and having a consumer act on it.
        """
        reason = getattr(self, "_quiescent_due_reason", None)
        if reason is None:
            return
        self._quiescent_due_reason = None
        hooks = getattr(self, "_ui_hooks", None) or getattr(
            self, "_callbacks", None
        )
        if hooks is None or not hasattr(hooks, "on_session_quiescent"):
            return
        try:
            hooks.on_session_quiescent(agent_id=self._agent_id, reason=reason)
        except Exception as exc:  # noqa: BLE001 — a hook must not break wind-down
            logger.warning(
                "on_session_quiescent hook raised: %s — event emission "
                "skipped, session will still wind down correctly", exc,
            )

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
        """Get the context window limit for the current model.

        Returns ``0`` — the honest "unknown" sentinel — when the provider
        has not been materialized yet.  The provider is lazy-created
        (``_ensure_provider``, first model use), so before the first turn
        there is no model to read a window from.

        This mirrors the ``InstructionBudget`` design, which also starts at
        ``context_limit = 0`` (see ``_populate_instruction_budget``) rather
        than a hardcoded default: a fake non-zero limit (the old
        ``1_048_576``) masked a not-yet-materialized/misconfigured provider
        and, worse, poisoned the daemon-side ``_cached_context_limit`` — the
        cache only refreshes on a ``0`` reading (``core.py`` context-update
        handler), so a bogus 1M value was cached at ``initialize()`` time and
        never healed, making every ``ContextUpdatedEvent`` report 1M even for
        a tiny-context model (e.g. Gemini Nano ~9k).  Every consumer of this
        value already guards ``context_limit > 0`` / ``max(0, ...)``, and the
        runner RPC handler documents ``0`` as the provider-not-initialized
        signal daemon callers retry on.
        """
        if not self._provider:
            return 0
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

    def _log_gc_denominator(self, label: str, provider_total: int = 0) -> None:
        """Emit (``logger.info``) the GC-trigger denominator breakdown so
        the budget-GC decision is inspectable from the daemon log.

        Two denominators are in play and must be told apart:

        - **provider path** (proactive GC during streaming): the wire
          size vLLM itself reports (``usage.total_tokens``) — ground
          truth for what's actually on the wire.
        - **InstructionBudget path** (pre-send / after-turn GC):
          ``InstructionBudget.total_tokens()`` — what PR-274's
          ``wire_tool_schemas`` LOCKED PLUGIN child feeds into, and what
          ``gc_budget``'s threshold check reads via
          ``get_context_usage``.

        Logging both side-by-side (with the ``wire_tool_schemas`` child
        broken out) makes three things verifiable from one line: (a) did
        PR-274's tool-schema registration actually land in the
        InstructionBudget; (b) does the InstructionBudget total track the
        provider's true wire or diverge from it; (c) at a GC decision,
        which number the threshold is being compared against.

        Uses ``logger.info`` deliberately — NOT ``self._trace``, which
        apparmor can silently swallow on confined runner sessions (see
        ``feedback_apparmor_blocks_provider_trace_silently``).
        """
        ib = self._instruction_budget
        if ib is None:
            logger.info(
                "GC_DENOM[%s] no_instruction_budget provider_total=%d",
                label, provider_total,
            )
            return

        def _src_total(source) -> int:
            entry = ib.get_entry(source)
            return entry.total_tokens() if entry is not None else 0

        sys_t = _src_total(InstructionSource.SYSTEM)
        plugin_entry = ib.get_entry(InstructionSource.PLUGIN)
        plugin_t = plugin_entry.total_tokens() if plugin_entry is not None else 0
        wire_t = 0
        if plugin_entry is not None and "wire_tool_schemas" in plugin_entry.children:
            wire_t = plugin_entry.children["wire_tool_schemas"].tokens
        conv_t = _src_total(InstructionSource.CONVERSATION)
        logger.info(
            "GC_DENOM[%s] ib_total=%d (sys=%d plugin=%d[wire_tools=%d] "
            "conv=%d) limit=%d pct=%.1f%% gc_eligible=%d | provider_total=%d",
            label, ib.total_tokens(), sys_t, plugin_t, wire_t, conv_t,
            ib.context_limit, ib.utilization_percent(),
            ib.gc_eligible_tokens(), provider_total,
        )

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
        # Budget ceiling: refuse rather than silently serve an
        # over-budget turn.  See _refuse_if_budget_exhausted.
        _refusal = self._refuse_if_budget_exhausted()
        if _refusal is not None:
            logger.info("refusing turn: %s", _refusal)
            if on_output:
                on_output("system", f"[{_refusal} — session will not run "
                                    f"further turns]", "write")
            return f"[{_refusal}]"

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
        self._begin_turn_completion_state()

        if self._executor:
            self._executor.set_output_callback(on_output)

        turn_start = datetime.now()
        turn_data = {
            'prompt': 0,
            'output': 0,
            'total': 0,
            # ``total`` is the LAST response's total_tokens, which for a
            # prompt-inclusive provider is the end-of-turn CONTEXT SIZE — what
            # GC and the context displays want.  It is NOT what the turn cost:
            # a turn with a tool call has >=2 responses and each is billed, so
            # summing responses is the SPEND.  Both are legitimate and
            # different consumers want different ones; conflating them
            # undercounted a real 3-turn run by 41%.
            'spend_total': 0,
            'spend_prompt': 0,
            'spend_output': 0,
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
                # A severed turn can carry a complete-but-undispatched
                # tool call; leaving it unanswered invalidates history
                # for every later request (#751).  Same reconciliation
                # the non-parts loop gets via ``_finish_abnormally``.
                self._reconcile_unanswered_calls(response.finish_reason)
                # An output-cap truncation is recoverable here too
                # (#749).  The parts loop is a second, independent exit
                # -- a fix applied only to the main loop holds exactly
                # until an attachment is in the message.
                continued = self._recover_truncated_turn(
                    response, False, on_output, None, turn_data,
                    context="parts loop initial response",
                )
                if continued is None:
                    return self._abnormal_parts_turn_text(response)
                response = continued

            function_calls = list(response.get_function_calls())
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
                    if fc.unreadable_args is not None:
                        # Unreadable arguments are refused, not executed
                        # (#750) -- see _execute_single_tool.
                        executor_result = (
                            False, unreadable_arguments_error(fc),
                        )
                    elif self._executor:
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
                        fc_result_payload = _split_executor_result_impl(
                            executor_result
                        )[1]
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
                            is_error_result=tool_result_is_error(fc_result_payload),
                            result_status=tool_result_status(fc_result_payload),
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
                tool_results = self._gate_tool_results_for_active_modalities(tool_results)
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
                function_calls = list(response.get_function_calls())

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
            # Record the terminal response's finish reason on the turn
            # accounting (it rides ``TurnCompletedEvent.finish_reason``) and,
            # for an abnormal stop, surface a ``source="system"`` banner so a
            # truncated turn isn't mistaken for a clean completion.  Unlike
            # ``_run_chat_loop`` this parts loop has no per-continuation
            # classifier, so the terminal ``response`` here is the single
            # reliable capture point — it also covers a *continuation* that
            # ended abnormally, which the initial-response inline check above
            # never saw.
            if response is not None and response.finish_reason is not None:
                turn_data['finish_reason'] = response.finish_reason.value
                if (
                    response.finish_reason in (
                        FinishReason.MAX_TOKENS,
                        FinishReason.SAFETY,
                        FinishReason.ERROR,
                    )
                    and on_output is not None
                ):
                    on_output(
                        "system",
                        self._abnormal_finish_message(response.finish_reason),
                        "write",
                    )

            turn_end = datetime.now()
            turn_data['end_time'] = turn_end.isoformat()
            turn_data['duration_seconds'] = (turn_end - turn_start).total_seconds()
            self._budget_observe_turn(turn_data)

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

        new_history, result = self._run_gc(
            history, context_usage, GCTriggerReason.MANUAL,
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

        # Phase 0: collapse byte-identical duplicate tool-results BEFORE the
        # before-send GC check.  This is the GC path that fires on the
        # InstructionBudget utilisation (vs the streaming-flag-gated
        # after-turn path), so it's the one that catches a request about to
        # overflow the context window — dedup must run here too, or the
        # duplicate-catalog bloat is never reclaimed before the send.
        # Re-reads context_usage below so the GC_CHECK percent reflects the
        # smaller wire.
        self._dedup_history_for_gc()

        context_usage = self.get_context_usage()
        # Diagnostic: full denominator breakdown (sys/plugin[wire_tools]/conv)
        # so the GC_CHECK percent can be attributed to its sources and the
        # wire_tool_schemas registration verified.
        self._log_gc_denominator("before_send")
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

            def _after(new_history, result, gc_span):
                if result.success:
                    if result.items_collected == 0:
                        # GC ran but collected nothing — often surprising to
                        # operators debugging cascade overflow.  Surface the
                        # LOCKED-vs-eligible breakdown so the operator can
                        # tell WHICH class of "nothing trimmable" applies:
                        #
                        # - High locked_tokens, low eligible_tokens: most of
                        #   the budget is body-wired (persona, pinned refs,
                        #   tool schemas, framework instructions).  Fix:
                        #   externalize content, shrink persona, or extend
                        #   context window.  GC cannot help.
                        # - Low locked + low eligible: budget is mostly
                        #   empty; threshold was crossed by something else
                        #   (e.g. provider-reported usage from a streaming
                        #   chunk).  Investigate budget accuracy.
                        # - Moderate eligible but 0 collected: strategy
                        #   restrictions (gc_truncate's preserve_recent_turns,
                        #   gc_budget's per-source policies) prevented
                        #   removal.  Tune the strategy config.
                        #
                        # Routed via ``logger.info`` (NOT ``self._trace``)
                        # so the diagnostic lands in /tmp/jaato.log — the
                        # apparmor-confined runner can't write to the
                        # default ``self._trace`` path
                        # ([[project_backlog_apparmor_blocks_provider_trace_silently]]).
                        locked = (
                            self._instruction_budget.locked_tokens()
                            if self._instruction_budget else 0
                        )
                        eligible = (
                            self._instruction_budget.gc_eligible_tokens()
                            if self._instruction_budget else 0
                        )
                        preservable = (
                            self._instruction_budget.preservable_tokens()
                            if self._instruction_budget else 0
                        )
                        total = (
                            self._instruction_budget.total_tokens()
                            if self._instruction_budget else 0
                        )
                        context_limit = (
                            self._instruction_budget.context_limit
                            if self._instruction_budget else 0
                        )
                        logger.info(
                            "GC_NO_ITEMS_COLLECTED: GC triggered but freed "
                            "0 items.  budget breakdown: total=%d "
                            "locked=%d eligible=%d preservable=%d "
                            "context_limit=%d.  reason=%s.  "
                            "details=%s.  When locked >> eligible, the "
                            "budget is body-wired and GC cannot help — "
                            "reduce locked content (externalize references, "
                            "shrink persona/schemas) or extend the context "
                            "window.",
                            total, locked, eligible, preservable,
                            context_limit, reason.value if reason else None,
                            result.details,
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
                    return new_history
                return None

            _new_history, result = self._run_gc(
                history, context_usage, reason, on_collected=_after,
            )

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
    # The plugin/config and the save/restore flow live on the
    # ``SessionPersistence`` collaborator (``self._persistence``).  These
    # methods are thin delegations; the properties below keep
    # ``_session_plugin`` / ``_session_config`` readable for the handful
    # of external/internal readers that reach for them directly
    # (jaato_client, refresh_tools, revert).

    @property
    def _session_plugin(self) -> Optional[SessionPlugin]:
        """The attached session plugin, or None (owned by _persistence)."""
        return self._persistence.plugin

    @property
    def _session_config(self) -> Optional[SessionConfig]:
        """The session config paired with the plugin (owned by _persistence)."""
        return self._persistence.config

    def set_session_plugin(
        self,
        plugin: SessionPlugin,
        config: Optional[SessionConfig] = None
    ) -> None:
        """Set the session plugin for persistence."""
        self._persistence.set_plugin(plugin, config)

    def remove_session_plugin(self) -> None:
        """Remove the session plugin."""
        self._persistence.remove_plugin()

    def save_session(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> str:
        """Save the current session."""
        return self._persistence.save(session_id, user_inputs)

    def resume_session(self, session_id: str) -> SessionState:
        """Resume a previously saved session."""
        return self._persistence.resume(session_id)

    def list_sessions(self) -> List[SessionInfo]:
        """List all available sessions."""
        return self._persistence.list()

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session."""
        return self._persistence.delete(session_id)

    def _get_session_state(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> SessionState:
        """Build a SessionState snapshot (delegates to _persistence)."""
        return self._persistence.build_state(session_id, user_inputs)

    def _restore_session_state(self, state: SessionState) -> None:
        """Restore session state from a SessionState (delegates to _persistence)."""
        self._persistence.restore_state(state)

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
            RuntimeError: If the session has no provider — typically
                because the session was configured with
                ``skip_provider=True`` (auth-pending mode) and auth
                hasn't completed yet.
        """
        # Lazy-init the provider on first model use (deferred-provider-INIT
        # design 2026-05-13).  Mirrors send_message:3560 — provider
        # construction is deferred from configure() to first model
        # call to keep the bootstrap RPC critical path short.  Before
        # this fix, replay_messages did a bare ``if not self._provider``
        # check that surfaced as "Session not configured" on
        # forensic-fork sessions where send_message never fired
        # (canonical caller: ``session_ops.interrogate_session``
        # against a create_headless_session fork — the fork is fully
        # configured but its provider has never been materialised).
        # Idempotent + thread-safe per ``_ensure_provider``.
        self._ensure_provider()
        if not self._provider:
            raise RuntimeError(
                "Session has no provider — "
                "skip_provider (auth-pending) mode and auth has not "
                "completed yet, OR _ensure_provider() returned "
                "without setting one (check configure() succeeded)."
            )

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

        The assembled :attr:`_system_instruction` plus, in tier mode, one
        line of tier PROTOCOL.  **Byte-identical for the life of the
        session**, which is the whole point: this string is the head of
        every cached prefix, so anything mutable in it invalidates the
        cache on every change and takes tools and history down with it.

        It used to name the CURRENT tier ("You are currently operating in
        the ``executor`` tier"), rewritten on every switch.  The old
        docstring claimed the assembled instruction stayed "a stable cache
        anchor"; that was true of :attr:`_system_instruction` in memory
        and false of what went on the wire, because the provider folds the
        two into ONE content block and the cache breakpoint sits on it.
        The result was a full prefix invalidation per tier switch —
        including the cases that should have hit, such as returning to a
        tier the session had already used, two tiers sharing one model,
        and every implicit-prefix-caching upstream.  See
        ``docs/design/model-tier-prompt-cache.md`` §5.1.

        What the model gets instead is stable and sufficient:

        * where the session STARTED (``initial_tier``, a config value that
          never changes — budget rungs rebind a tier's *model*, not which
          tier is initial); and
        * the rule that the tier changes only via ``enter_tier``, whose
          result reports the tier landed in.

        So the current tier is derivable — start point plus the switches
        recorded in history — without restating mutable state in the one
        place that must not carry any.  The model does not need it to
        DECIDE anyway: ``enter_tier`` is chosen by the work about to be
        done, not by where it currently is, and entering the active tier
        is a documented no-op.
        """
        if self._active_tier is None:
            return self._system_instruction
        tier_line = (
            f"This session runs in multi-tier mode and started in the "
            f"`{self._tier_config.initial_tier}` tier.  Your active tier "
            f"changes only when you call `enter_tier`, which reports the "
            f"tier you land in."
        )
        if self._system_instruction:
            return self._system_instruction + "\n\n" + tier_line
        return tier_line

    def _provider_for_tier(self, provider_name: str, model: str) -> 'ModelProviderPlugin':
        """Cached provider instance for a cross-provider tier (V2).

        Creates + caches on first use, keyed by ``provider_name``.  Reuses the
        session's lazy-pending ``plugin_configs`` / ``skip_model_test`` so each
        provider reads its OWN ``plugin_configs`` section (e.g.
        ``plugin_configs.openrouter``).  A later switch back is O(1) (cache hit).
        """
        prov = self._provider_cache.get(provider_name)
        if prov is not None:
            return prov
        # Read the PERSISTENT base config, NOT _provider_lazy_pending — the
        # latter is cleared to None once the main provider is created, which
        # left cross-provider tier providers with plugin_configs=None (no
        # api_key).  _tier_provider_base survives that clear.
        cfg = self._tier_provider_base or {}
        prov = self._runtime.create_provider(
            model,
            provider_name=provider_name,
            skip_model_test=cfg.get('skip_model_test', True),
            plugin_configs=cfg.get('plugin_configs'),
        )
        if hasattr(prov, 'set_agent_context'):
            prov.set_agent_context(
                agent_type=self._agent_type,
                agent_name=self._agent_name,
                agent_id=self._agent_id,
            )
        self._provider_cache[provider_name] = prov
        return prov

    def _is_connected_to(self, entry) -> bool:
        """True if the live provider is already serving ``entry``'s binding.

        Compares the resolved (model, provider) pair rather than the tier
        name, so an in-place rebind of the active tier is correctly seen as
        a CHANGE.  ``entry.provider`` of ``None`` means "the session's own
        provider", which is by definition the active one.
        """
        if self._provider is None:
            return True  # nothing to reconnect
        if self._model_name != entry.model:
            return False
        return entry.provider is None or entry.provider == self._active_provider_name

    def _request_active_tier_output_modalities(self) -> None:
        """Stamp the ACTIVE tier's outbound roles onto a freshly built provider.

        The switch path (:meth:`_connect_tier_entry`) covers every later
        tier change; this covers the first one, which is not a change at
        all — the session simply starts there.
        """
        if self._tier_config is None or not self._active_tier:
            return
        entry = self._tier_config.tiers.get(self._active_tier)
        if entry is not None:
            self._request_tier_output_modalities(entry)

    def _request_tier_output_modalities(self, entry) -> None:
        """Ask the provider to emit what the entered tier declares.

        This is what turns ``modalities: {audio: outbound}`` from a
        declaration the startup check merely *validates* into a request
        that reaches the wire.  Without it an outbound role is inert: the
        tier says the model may speak, but nothing ever asks it to.

        Called on EVERY tier entry, including entries that declare no
        outbound role, because the empty set is the instruction that stops
        requesting audio — leaving a speaking tier must not leave the
        request stamped.

        Best-effort by design.  A provider that cannot emit media inherits
        a no-op :meth:`request_output_modalities`, so the common case costs
        one call; and a provider that raises must not fail the tier switch,
        which has already succeeded by this point.  A model that genuinely
        cannot do the job is refused far earlier, by the startup
        capability check.
        """
        provider = self._provider
        if provider is None:
            return
        request = getattr(provider, "request_output_modalities", None)
        if request is None:
            return
        kinds = getattr(entry, "outbound_modalities", frozenset()) or frozenset()
        try:
            request(kinds)
        except Exception:  # noqa: BLE001 - never fail a completed switch
            self._trace(
                f"TIER_OUTPUT_MODALITIES: provider refused {sorted(kinds)!r}"
            )

    def _exit_completion_tier_if_settled(self, response) -> None:
        """Leave an ``exit_on: completion`` tier once its work is done.

        "Settled" means the tier's response asks for nothing more: no
        function calls.  Deliberately not "one provider call" -- a
        delegated tier that legitimately calls a tool would be evicted
        mid-task -- and deliberately not "one turn", because a turn
        boundary is not a terminus (#767).

        Best-effort in the same way as the entry: a failed return must not
        fail a turn that has already produced its answer.  The pending
        target is cleared either way, so a tier cannot be armed to return
        twice.
        """
        target = self._pending_tier_return
        if target is None or response is None:
            return
        if response.has_function_calls():
            return                      # still working; it has not settled
        self._pending_tier_return = None
        if target == self._active_tier:
            return
        delegated_from = self._active_tier
        produced = "".join(p.text for p in response.parts if p.text).strip()
        spoke = getattr(response, "media_chunks", 0) or 0
        try:
            self.switch_tier(target)
            self._trace(f"TIER_EXIT_ON_COMPLETION: returned to {target}")
        except Exception as exc:  # noqa: BLE001 - never fail a finished turn
            self._trace(
                f"TIER_EXIT_ON_COMPLETION: return to {target} failed: {exc}")
            return
        self._report_delegated_tier(delegated_from, produced, spoke)

    def _report_delegated_tier(
        self, tier: str, produced: str, media_chunks: int,
    ) -> None:
        """Hand the delegated tier's outcome back as a mid-turn message.

        Returning the BINDING is not returning CONTROL.  The delegated
        tier's completion settling is what ENDS the turn, so switching
        back alone hands the wheel to a tier that no longer has a turn to
        steer -- and measurement showed exactly that: the model's manual
        `enter_tier` back disappeared, and the completion nudge was still
        the only thing that woke the caller to finish.

        Queuing the outcome as a mid-turn message resumes the caller
        through the path the framework already has for "something arrived
        while you were working", which the loop drains immediately after
        this returns.  It is not a nudge: a nudge tells an agent it forgot
        to finish, this tells it what its delegate produced.

        That also closes the other half.  Model media never enters history
        -- it is CLIENT-audience by construction -- so the caller could
        otherwise only learn what was said if the provider happened to
        send a transcript.  Here the report is written whether or not it
        did, and says so when it did not, which turns a silent hole into a
        stated one.
        """
        from .message_queue import SourceType
        if produced:
            body = f'The {tier} tier produced: "{produced}"'
        else:
            body = (
                f"The {tier} tier produced no text"
                + (f", though it emitted {media_chunks} media chunk(s)"
                   if media_chunks else "")
                + "."
            )
        if media_chunks:
            body += f"  ({media_chunks} media chunk(s) were delivered to the client.)"
        body += "  You are back in control; continue."
        self._message_queue.put(
            f"<hidden>{body}</hidden>", "tier-delegation", SourceType.SYSTEM)
        self._trace(f"TIER_DELEGATION_REPORT: queued outcome from {tier}")

    def _connect_tier_entry(self, entry) -> None:
        """Point the session's provider at ``entry``'s (provider, model).

        Shared by ``switch_tier`` (model-driven, via ``enter_tier``) and by
        budget-control degradation (framework-driven, after a rung rebinds
        the active tier).  Cross-provider entries swap ``self._provider`` to
        the cached instance for that provider — history is provider-neutral
        (Message/Part), so the conversation flows across the swap.

        The cache plugin is re-wired afterwards, because it is bound to a
        (provider, model) pair and this method changes both.  Without
        that, a cross-provider tier ran with no cache plugin at all and a
        same-provider tier ran with one still configured for the model
        that booted the session.

        Re-wiring is best-effort: a session that cannot attach a cache
        plugin should run uncached, not fail the tier switch.  The connect
        itself still raises, because a session pointed at the wrong model
        is not something to continue from.
        """
        if self._provider is None:
            return
        target_provider = entry.provider
        try:
            if target_provider and target_provider != self._active_provider_name:
                self._provider = self._provider_for_tier(
                    target_provider, entry.model)
                self._active_provider_name = target_provider
            self._provider.connect(entry.model, skip_model_test=True)
        except Exception as exc:
            logger.warning(
                "tier connect to %s/%s failed: %s",
                target_provider or self._active_provider_name,
                entry.model, exc,
            )
            raise
        # Everything from here down is bookkeeping ABOUT the switch, not
        # part of it: the connect has already happened and the caller is
        # about to update ``_model_name``.  None of it may raise, or a
        # switch lands half-applied — the provider re-pointed at the new
        # model while the session still believes it is on the old one.
        #
        # Asking the provider to emit the tier's outbound modalities is
        # exactly such bookkeeping, which is why it sits below the connect
        # rather than inside it.
        self._request_tier_output_modalities(entry)
        #
        # Counted here rather than in ``switch_tier`` so BOTH routes into a
        # binding change are seen: the model-driven one and the
        # budget-control rebind, which never changes the tier NAME.  A
        # ``switch_tier`` no-op returns before reaching this method, so an
        # ``enter_tier`` to the tier already active does not inflate it.
        self._tier_switch_count = getattr(self, '_tier_switch_count', 0) + 1
        try:
            self._wire_cache_plugin()
        except Exception as exc:  # noqa: BLE001
            self._tier_cache_rewire_failures = getattr(
                self, '_tier_cache_rewire_failures', 0) + 1
            logger.warning(
                "tier cache re-wire for %s/%s failed; continuing uncached: %s",
                self._active_provider_name, entry.model, exc,
            )
        try:
            self._retarget_reliability_model(entry.model)
        except Exception as exc:  # noqa: BLE001
            self._tier_reliability_retarget_failures = getattr(
                self, '_tier_reliability_retarget_failures', 0) + 1
            logger.warning(
                "tier reliability retarget for %s failed; records will name "
                "the previous model: %s", entry.model, exc,
            )

    def _retarget_reliability_model(self, model: str) -> None:
        """Tell the reliability plugin which model is now running.

        Reliability records — behavioural patterns, tool-failure history —
        are stamped with the active model.  That model is captured when the
        session is configured, so in a ``model_tiers`` session every record
        produced after a tier switch was filed under the model that STARTED
        the session, and the record could not say which tier misbehaved.

        Same shape as the cache-plugin re-wire alongside it, and the same
        two routes reach it: ``enter_tier`` and a budget-control rung
        rebinding the active tier in place.

        ``available_models`` is deliberately not re-supplied — it is the
        switchable-model catalogue, which a tier change does not alter, and
        ``set_model_context`` leaves it untouched when passed ``None``.

        Raises rather than swallowing.  Attribution must never fail a tier
        switch, but the ONE place that decides so is the caller's
        post-connect block, which also counts the failure onto
        ``jaato.tier.reliability_retarget_failures``.  A second try/except
        here looked like belt-and-braces and was the opposite: it ate the
        exception before the counter could see it, so the span reported a
        healthy session while every pattern was being judged against the
        wrong model.  Two layers of swallowing is one layer of hiding.
        """
        plugin = getattr(self._runtime, 'reliability_plugin', None) \
            if self._runtime else None
        if plugin is None:
            return
        plugin.set_model_context(model)

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

        # Short-circuit on the RESOLVED ENTRY, not merely the tier name.
        # A budget-control degrade rung can REBIND the active tier's model
        # in place (planner: opus -> flash); the tier name is unchanged, so a
        # name-only comparison would report "already_at_tier" and never
        # re-connect — the rebind would silently not take effect until the
        # agent happened to leave and re-enter the tier.  Comparing the
        # resolved (model, provider) closes that hole while keeping the
        # genuine no-op cheap.
        if actual_tier == self._active_tier and self._is_connected_to(entry):
            return {
                "status": "already_at_tier",
                "active_tier": actual_tier,
                "requested_tier": requested_tier,
                "model": entry.model,
            }

        self._connect_tier_entry(entry)

        previous_tier = self._active_tier
        self._active_tier = actual_tier
        self._model_name = entry.model

        # A tier that exits on completion is a DELEGATION: entered, one
        # completion, left again, with the model doing nothing to return.
        # That matters because the model in a specialist tier is routinely
        # the one LEAST able to hand back -- a speaking tier measured over
        # four runs never returned on its own; it said its sentence and
        # stopped, and only the completion nudge ever unblocked it.
        from .model_tiers import EXIT_ON_COMPLETION
        if entry.exit_on == EXIT_ON_COMPLETION and previous_tier != actual_tier:
            self._pending_tier_return = previous_tier
            self._trace(
                f"TIER_EXIT_ARMED: {actual_tier} exits on completion, "
                f"returning to {previous_tier}"
            )

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
        self._persistence.close()


__all__ = ['JaatoSession']
