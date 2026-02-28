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
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .message_queue import MessageQueue, QueuedMessage, SourceType
from .session_history import SessionHistory

logger = logging.getLogger(__name__)

from .ai_tool_runner import ToolExecutor
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
)

if TYPE_CHECKING:
    from .jaato_runtime import JaatoRuntime
    from .plugins.model_provider.base import ModelProviderPlugin
    from .plugins.subagent.ui_hooks import AgentUIHooks
    from .plugins.telemetry import TelemetryPlugin
    from .plugins.thinking import ThinkingPlugin

# Import framework instruction for tool result injection
from .jaato_runtime import _TASK_COMPLETION_INSTRUCTION

# Pattern to match @references in prompts
AT_REFERENCE_PATTERN = re.compile(r'@([\w./\-]+(?:\.\w+)?)')


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
        provider_name: Optional[str] = None
    ):
        """Initialize a session.

        Note: Use runtime.create_session() instead of calling this directly.

        Args:
            runtime: Parent JaatoRuntime providing shared resources.
            model: Model name to use for this session.
            provider_name: Optional provider override for cross-provider sessions.
                          If specified, uses a different AI provider than the runtime default.
        """
        self._runtime = runtime
        self._model_name = model
        self._provider_name_override = provider_name

        # Provider for this session (created during configure())
        self._provider: Optional['ModelProviderPlugin'] = None

        # Canonical conversation history owned by the session.
        # Phase 1: synced from provider after each provider operation.
        # Phase 2+: session is sole owner; provider receives messages
        # as parameters to stateless complete().
        self._history = SessionHistory()

        # Session always owns history and uses stateless provider.complete().
        # Legacy send_message()/send_tool_results() path removed in Phase 4.

        # Tool configuration
        self._executor: Optional[ToolExecutor] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._system_instruction: Optional[str] = None
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

        # UI hooks for agent lifecycle events
        self._ui_hooks: Optional['AgentUIHooks'] = None
        self._agent_id: str = "main"  # Unique ID for this agent

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

        # Mid-turn prompt interrupt tracking
        # When True, cancellation was triggered by a pending user prompt, not user cancellation
        self._mid_turn_interrupt: bool = False

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

        # Streaming tool support
        self._stream_manager: Optional[StreamManager] = None
        # Timeout for waiting on streaming updates when model is idle (seconds)
        self._streaming_wait_timeout: float = 5.0

    @property
    def _telemetry(self) -> 'TelemetryPlugin':
        """Get telemetry plugin from runtime."""
        return self._runtime.telemetry

    def set_terminal_width(self, width: int) -> None:
        """Set the terminal width for formatting.

        This affects enrichment notification formatting.

        Args:
            width: Terminal width in columns.
        """
        self._terminal_width = width

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
        """Check if session is configured and ready."""
        return self._provider is not None

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
            context = {"agent_type": agent_type}
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
            # High-priority (PARENT/USER/SYSTEM) → processed mid-turn
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

    def request_stop(self) -> bool:
        """Request cancellation of the current message processing.

        If a message is being processed, signals the cancel token to stop.
        The message loop will check this token and exit gracefully.

        Returns:
            True if a cancellation was requested (message was running),
            False if no message was running.

        Note:
            Cancellation is cooperative - it may not be immediate.
            The current streaming chunk will complete before stopping.
        """
        if self._cancel_token and self._is_running:
            self._cancel_token.cancel()
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
        preloaded_plugins: Optional[set] = None
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
        """
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

        # Create provider for this session (with optional provider override)
        # Skip if in auth-pending mode
        if not skip_provider:
            self._provider = self._runtime.create_provider(
                self._model_name,
                provider_name=self._provider_name_override
            )

            # Propagate agent context to provider for trace identification
            if hasattr(self._provider, 'set_agent_context'):
                self._provider.set_agent_context(
                    agent_type=self._agent_type,
                    agent_name=self._agent_name,
                    agent_id=self._agent_id
                )

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

            # Refresh runtime's tool cache to include the newly registered core tools
            self._runtime.refresh_tool_cache()

            # Re-fetch tools to include core tools, and update session tools
            self._tools = self._runtime.get_tool_schemas(tools)
            executors = self._runtime.get_executors(tools)
            for name, fn in executors.items():
                self._executor.register(name, fn)

        # Set permission plugin with agent context
        if self._runtime.permission_plugin:
            context = {"agent_type": self._agent_type}
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

        # Auto-wire plugins that need session access
        # Any plugin with set_session() will receive this session reference
        if self._runtime.registry:
            import threading
            self._trace(f"configure: wiring plugins with session, thread_id={threading.current_thread().ident}")
            for plugin_name in self._runtime.registry._exposed:
                plugin = self._runtime.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'set_session'):
                    plugin.set_session(self)

        # Build system instructions
        self._system_instruction = self._runtime.get_system_instructions(
            plugin_names=tools,
            additional=system_instructions,
            presentation_context=self._presentation_context,
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

        # Populate instruction budget after all configuration is complete
        self._populate_instruction_budget(session_instructions=system_instructions)

        # Wire cache plugin (after budget is populated so we can set it)
        self._wire_cache_plugin()

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
            self._history.append(Message(role=Role.MODEL, parts=history_parts))

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
        requests = self._collect_instruction_texts(session_instructions)

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
    ) -> List['_TokenCountRequest']:
        """Collect all instruction texts that need token counting.

        Gathers texts from SYSTEM children (base, client, framework) and
        PLUGIN children (per-plugin, per-formatter) into a flat list of
        ``_TokenCountRequest`` objects.

        Args:
            session_instructions: The user-provided system_instructions from configure().

        Returns:
            List of ``_TokenCountRequest`` — one per instruction text.
        """
        from .jaato_runtime import (
            _TASK_COMPLETION_INSTRUCTION,
            _PARALLEL_TOOL_GUIDANCE,
            _TURN_SUMMARY_INSTRUCTION,
            _is_parallel_tools_enabled,
        )

        requests: List[_TokenCountRequest] = []

        # --- SYSTEM children ---

        # 1. Base instructions from .jaato/instructions/ (or legacy single file)
        base_instructions = getattr(self._runtime, '_base_system_instructions', None)
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

        # Get schemas for requested tools from registry
        all_schemas = self._runtime.registry.get_exposed_tool_schemas()
        schema_map = {s.name: s for s in all_schemas}

        for tool_name in tool_names:
            if tool_name in current_tool_names:
                continue  # Already active
            if tool_name not in schema_map:
                continue  # Tool doesn't exist

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
        if not self._provider:
            raise RuntimeError("Session not configured. Call configure() first.")

        self._trace(f"SESSION_SEND_MESSAGE len={len(message)} streaming={self._use_streaming}")

        # Increment turn counter
        self._turn_index += 1

        # Notify reliability plugin of turn start
        if self._runtime.reliability_plugin:
            self._runtime.reliability_plugin.on_turn_start(self._turn_index)

        # Wrap entire turn with telemetry span
        with self._telemetry.turn_span(
            session_id=self._agent_id,
            agent_type=self._agent_type,
            agent_name=self._agent_name,
            turn_index=self._turn_index,
        ) as turn_span:
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
            processed_message = self._enrich_and_clean_prompt(message)

            try:
                response = self._run_chat_loop(processed_message, on_output, wrapped_usage_callback)
                turn_span.set_status_ok()
            except Exception as e:
                turn_span.record_exception(e)
                turn_span.set_status_error(str(e))
                raise

            # Record turn completion attributes
            turn_span.set_attribute("jaato.cancelled", self._is_cancelled())
            turn_span.set_attribute("jaato.streaming", self._use_streaming)

            # Proactive GC: if threshold was crossed during streaming, trigger GC now
            if self._gc_threshold_crossed and self._gc_plugin and self._gc_config:
                self._trace("PROACTIVE_GC: Threshold crossed during streaming, triggering post-turn GC")
                self._maybe_collect_after_turn()

            # Notify session plugin
            self._notify_session_turn_complete()

            # Notify reliability plugin of turn end
            if self._runtime.reliability_plugin:
                self._runtime.reliability_plugin.on_turn_end()

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

            # Sync budget with GC changes
            self._apply_gc_removal_list(result)
            self._emit_instruction_budget_update()

        return result

    def _apply_gc_removal_list(self, result: GCResult) -> None:
        """Apply GC removal list to instruction budget.

        This synchronizes the budget with the actual history changes made by GC.
        Must be called after a successful GC operation.

        Args:
            result: The GCResult containing the removal_list.
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

        # Notify cache plugin about GC so it can track prefix invalidation
        _cache = getattr(self, '_cache_plugin', None)
        if _cache and hasattr(_cache, 'on_gc_result'):
            try:
                _cache.on_gc_result(result)
            except Exception as e:
                self._trace(f"CACHE_PLUGIN: on_gc_result failed: {e}")

    def _enrich_and_clean_prompt(self, prompt: str) -> str:
        """Run prompt through enrichment pipeline and strip @references."""
        enriched_prompt = prompt

        # Run through plugin enrichment pipeline
        if self._runtime.registry:
            result = self._runtime.registry.enrich_prompt(prompt)
            enriched_prompt = result.prompt

        # Strip @references
        return AT_REFERENCE_PATTERN.sub(r'\1', enriched_prompt)

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
        self._mid_turn_interrupt = False  # Reset mid-turn interrupt flag for new message
        self._is_running = True
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

            # Send message (streaming or batched) with telemetry
            with self._telemetry.llm_span(
                model=self._model_name or "unknown",
                provider=self._provider.name if self._provider else "unknown",
                streaming=use_streaming,
            ) as llm_telemetry:
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
                            self._mid_turn_interrupt = True
                            if self._cancel_token:
                                self._cancel_token.cancel()
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

                    response, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._system_instruction,
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
                    response, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._system_instruction,
                            tools=self._get_tools_for_provider(),
                        ),
                        context="complete",
                        on_retry=self._on_retry,
                        cancel_token=self._cancel_token,
                        provider=self._provider
                    )

                # Record model response in session history
                self._add_model_response_to_history(response)
                self._record_token_usage(response)
                self._accumulate_turn_tokens(response, turn_data)
                # Track model response count for turn complexity
                self._turn_model_response_count += 1
                # Record token usage to telemetry span
                if response.usage:
                    llm_telemetry.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                    llm_telemetry.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
                    if response.usage.cache_read_tokens is not None:
                        llm_telemetry.set_attribute("gen_ai.usage.cache_read_tokens", response.usage.cache_read_tokens)
                    if response.usage.cache_creation_tokens is not None:
                        llm_telemetry.set_attribute("gen_ai.usage.cache_creation_tokens", response.usage.cache_creation_tokens)
                    if response.usage.reasoning_tokens is not None:
                        llm_telemetry.set_attribute("gen_ai.usage.reasoning_tokens", response.usage.reasoning_tokens)
            self._trace(f"SESSION_STREAMING_COMPLETE parts_count={len(response.parts)} finish={response.finish_reason}")

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
            if response.finish_reason not in (FinishReason.STOP, FinishReason.UNKNOWN, FinishReason.TOOL_USE, FinishReason.CANCELLED):
                logger.warning(f"Model stopped with finish_reason={response.finish_reason}")
                response_text = response.get_text()
                if response_text:
                    return f"{response_text}\n\n[Model stopped: {response.finish_reason}]"
                else:
                    return f"[Model stopped unexpectedly: {response.finish_reason}]"

            # Check for cancellation after initial message (including parent)
            if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                partial_text = response.get_text()

                # Check if this was a mid-turn interrupt (user prompt arrived during streaming)
                if self._mid_turn_interrupt:
                    self._trace(f"MID_TURN_INTERRUPT: Processing user prompt (partial response: {len(partial_text) if partial_text else 0} chars)")
                    # Reset the interrupt flag
                    self._mid_turn_interrupt = False
                    # Reset the cancel token for subsequent operations
                    self._cancel_token = CancelToken()

                    # Peek at the pending prompt for the callback (don't pop yet)
                    pending_prompts = self._message_queue.peek_all()
                    user_prompt_preview = ""
                    for msg in pending_prompts:
                        if msg.source_type in (SourceType.USER, SourceType.PARENT, SourceType.SYSTEM):
                            user_prompt_preview = msg.text[:100] if msg.text else ""
                            break

                    # Notify via callback for UI event emission
                    if self._on_mid_turn_interrupt:
                        self._on_mid_turn_interrupt(
                            len(partial_text) if partial_text else 0,
                            user_prompt_preview
                        )

                    # Process the mid-turn prompt - this sends it to the model
                    # The partial response is preserved in provider history
                    mid_turn_response = self._check_and_handle_mid_turn_prompt(
                        use_streaming, on_output, wrapped_usage_callback, turn_data
                    )

                    if mid_turn_response:
                        # Update response to continue from the mid-turn prompt response
                        response = mid_turn_response
                        # Fall through to normal processing - don't return
                    else:
                        # No mid-turn prompt found (race condition?) - return partial
                        self._trace("MID_TURN_INTERRUPT: No prompt in queue, returning partial")
                        if partial_text:
                            return partial_text
                        return ""
                else:
                    # Normal user cancellation
                    cancel_msg = "[Generation cancelled]"
                    if on_output and not cancellation_notified:
                        self._trace(f"CANCEL_NOTIFY: {cancel_msg} (after initial message)")
                        on_output("system", cancel_msg, "write")
                        cancellation_notified = True
                    elif cancellation_notified:
                        self._trace(f"CANCEL_DUPLICATE: {cancel_msg} (after initial message) - already notified!")
                    # Notify model of cancellation for context on next turn
                    self._notify_model_of_cancellation(cancel_msg, partial_text)
                    if partial_text:
                        return f"{partial_text}\n\n{cancel_msg}"
                    return cancel_msg

            # Handle function calling loop - process parts in order to support interleaved text/tools
            accumulated_text: List[str] = []
            self._trace(f"SESSION_PARTS_PROCESSING parts_count={len(response.parts)}")

            def get_pending_function_calls() -> List[FunctionCall]:
                """Extract function calls from response.parts."""
                return [p.function_call for p in response.parts if p.function_call]

            def get_all_text() -> str:
                """Concatenate all text from response.parts."""
                texts = [p.text for p in response.parts if p.text]
                return ''.join(texts) if texts else ''

            pending_calls = get_pending_function_calls()

            # Handle edge case: finish_reason indicates tool use but no function calls present
            # This can happen when the model "narrates" intent to call a tool but doesn't emit the call
            tool_use_nudge_attempts = 0
            max_tool_use_nudge_attempts = 3
            while (not pending_calls and response.finish_reason == FinishReason.TOOL_USE
                   and tool_use_nudge_attempts < max_tool_use_nudge_attempts):
                tool_use_nudge_attempts += 1
                self._trace(f"TOOL_USE_WITHOUT_CALL: Model indicated tool use but no function call emitted (attempt {tool_use_nudge_attempts}/{max_tool_use_nudge_attempts})")
                # Inject a prompt to push the model to execute the intended tool call
                nudge_prompt = (
                    "<hidden>Your response indicated TOOL_USE but contained no function call. "
                    "Do NOT describe or re-read files. Execute the tool call you just mentioned directly. "
                    "Your next response MUST start with the function call, not text.</hidden>"
                )
                self._message_queue.put(nudge_prompt, "system", SourceType.SYSTEM)
                # Process the injected prompt to get a new response
                nudge_response = self._check_and_handle_mid_turn_prompt(
                    use_streaming, on_output, wrapped_usage_callback, turn_data
                )
                if nudge_response:
                    response = nudge_response
                    pending_calls = get_pending_function_calls()
                    self._trace(f"TOOL_USE_NUDGE_RESULT: Got response with {len(pending_calls)} function calls")
                else:
                    # No response from nudge, break to avoid infinite loop
                    self._trace("TOOL_USE_NUDGE_NO_RESPONSE: Nudge did not produce a response")
                    break

            if tool_use_nudge_attempts >= max_tool_use_nudge_attempts and not pending_calls:
                self._trace("TOOL_USE_NUDGE_EXHAUSTED: Max attempts reached, model still not calling tools")

            while pending_calls:
                # Check for cancellation before processing tools (including parent)
                if self._is_cancelled():
                    cancel_msg = "[Cancelled during tool execution]"
                    if on_output and not cancellation_notified:
                        self._trace(f"CANCEL_NOTIFY: {cancel_msg} (before processing tools)")
                        on_output("system", cancel_msg, "write")
                        cancellation_notified = True
                    elif cancellation_notified:
                        self._trace(f"CANCEL_DUPLICATE: {cancel_msg} (before processing tools) - already notified!")
                    # Notify model of cancellation for context on next turn
                    all_text = get_all_text()
                    self._notify_model_of_cancellation(cancel_msg, all_text)
                    if all_text:
                        return f"{all_text}\n\n{cancel_msg}"
                    return cancel_msg

                # Process parts in order - emit text, collect function calls into groups
                current_fc_group: List[FunctionCall] = []
                for idx, part in enumerate(response.parts):
                    # Enhanced trace: show empty text parts (which indicate unknown SDK parts)
                    text_info = "empty" if part.text == "" else bool(part.text) if part.text else None
                    fc_info = part.function_call.name if part.function_call else None
                    self._trace(f"SESSION_PART[{idx}] text={text_info} fc={fc_info}")
                    if part.text:
                        # Before emitting text, execute any pending function calls
                        if current_fc_group:
                            tool_results = self._execute_function_call_group(
                                current_fc_group, turn_data, on_output, cancellation_notified
                            )
                            if self._is_cancelled():
                                cancel_msg = "[Cancelled after tool execution]"
                                if on_output and not cancellation_notified:
                                    on_output("system", cancel_msg, "write")
                                self._notify_model_of_cancellation(cancel_msg)
                                return cancel_msg
                            # Send tool results and get continuation
                            response = self._send_tool_results_and_continue(
                                tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                            )
                            if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                                # Check if this was a mid-turn interrupt
                                if self._mid_turn_interrupt:
                                    self._trace("MID_TURN_INTERRUPT: Processing user prompt after interleaved tool result streaming")
                                    self._mid_turn_interrupt = False
                                    self._cancel_token = CancelToken()
                                    mid_turn_response = self._check_and_handle_mid_turn_prompt(
                                        use_streaming, on_output, wrapped_usage_callback, turn_data
                                    )
                                    if mid_turn_response:
                                        response = mid_turn_response
                                    else:
                                        partial = get_all_text()
                                        if partial:
                                            return partial
                                        return ""
                                else:
                                    partial = get_all_text()
                                    cancel_msg = "[Generation cancelled]"
                                    if on_output and not cancellation_notified:
                                        on_output("system", cancel_msg, "write")
                                    self._notify_model_of_cancellation(cancel_msg, partial)
                                    return f"{partial}\n\n{cancel_msg}" if partial else cancel_msg

                            # Check for mid-turn prompts after interleaved tool execution
                            # Only inject if response doesn't have more function calls
                            # (otherwise we'd break the tool_use -> tool_result sequence)
                            response_has_fc = any(p.function_call for p in response.parts)
                            if not response_has_fc:
                                mid_turn_response = self._check_and_handle_mid_turn_prompt(
                                    use_streaming, on_output, wrapped_usage_callback, turn_data
                                )
                                if mid_turn_response:
                                    # Update response to the mid-turn prompt response
                                    response = mid_turn_response
                                    if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                                        partial = get_all_text()
                                        cancel_msg = "[Generation cancelled]"
                                        if on_output and not cancellation_notified:
                                            on_output("system", cancel_msg, "write")
                                        self._notify_model_of_cancellation(cancel_msg, partial)
                                        return f"{partial}\n\n{cancel_msg}" if partial else cancel_msg

                            current_fc_group = []

                        # Emit text (only in non-streaming mode)
                        if not use_streaming:
                            if on_output:
                                on_output("model", part.text, "write")
                            # Forward to parent for visibility
                            self._forward_to_parent("MODEL_OUTPUT", part.text)
                            # Notify reliability plugin of model text for pattern detection
                            if self._runtime.reliability_plugin:
                                self._runtime.reliability_plugin.on_model_text(part.text)
                        accumulated_text.append(part.text)

                    elif part.function_call:
                        current_fc_group.append(part.function_call)

                # Execute remaining function calls at end of parts
                if current_fc_group:
                    tool_results = self._execute_function_call_group(
                        current_fc_group, turn_data, on_output, cancellation_notified
                    )
                    if self._is_cancelled():
                        cancel_msg = "[Cancelled after tool execution]"
                        if on_output and not cancellation_notified:
                            on_output("system", cancel_msg, "write")
                        self._notify_model_of_cancellation(cancel_msg)
                        return cancel_msg

                    # Send tool results and get next response
                    response = self._send_tool_results_and_continue(
                        tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                    )
                    if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                        # Check if this was a mid-turn interrupt (user prompt during tool result streaming)
                        if self._mid_turn_interrupt:
                            self._trace("MID_TURN_INTERRUPT: Processing user prompt after tool result streaming")
                            self._mid_turn_interrupt = False
                            self._cancel_token = CancelToken()

                            if self._on_mid_turn_interrupt:
                                pending_prompts = self._message_queue.peek_all()
                                preview = ""
                                for pm in pending_prompts:
                                    if pm.source_type in (SourceType.USER, SourceType.PARENT, SourceType.SYSTEM):
                                        preview = pm.text[:100] if pm.text else ""
                                        break
                                partial = get_all_text()
                                self._on_mid_turn_interrupt(len(partial) if partial else 0, preview)

                            mid_turn_response = self._check_and_handle_mid_turn_prompt(
                                use_streaming, on_output, wrapped_usage_callback, turn_data
                            )
                            if mid_turn_response:
                                response = mid_turn_response
                                # Fall through to normal processing
                            else:
                                self._trace("MID_TURN_INTERRUPT: No prompt in queue after tool result streaming")
                                partial = get_all_text()
                                if partial:
                                    return partial
                                return ""
                        else:
                            partial = get_all_text()
                            cancel_msg = "[Generation cancelled]"
                            if on_output and not cancellation_notified:
                                on_output("system", cancel_msg, "write")
                            self._notify_model_of_cancellation(cancel_msg, partial)
                            return f"{partial}\n\n{cancel_msg}" if partial else cancel_msg

                    if response.finish_reason not in (FinishReason.STOP, FinishReason.UNKNOWN, FinishReason.TOOL_USE, FinishReason.CANCELLED):
                        import sys
                        print(f"[warning] Model stopped with finish_reason={response.finish_reason}", file=sys.stderr)
                        final_text = get_all_text()
                        if final_text:
                            return f"{final_text}\n\n[Model stopped: {response.finish_reason}]"
                        else:
                            return f"[Model stopped unexpectedly: {response.finish_reason}]"

                    # Check for mid-turn prompts at this natural pause point
                    # Only inject if response doesn't have more function calls
                    # (otherwise we'd break the tool_use -> tool_result sequence)
                    response_has_fc = any(p.function_call for p in response.parts)

                    # Handle edge case: TOOL_USE without function calls, or UNKNOWN with empty response
                    # Both indicate the model didn't properly continue after tool results
                    response_is_empty = not response.parts or all(
                        not p.text and not p.function_call for p in response.parts
                    )
                    needs_nudge = (
                        (not response_has_fc and response.finish_reason == FinishReason.TOOL_USE) or
                        (response.finish_reason == FinishReason.UNKNOWN and response_is_empty)
                    )
                    main_loop_nudge_attempts = 0
                    while needs_nudge and main_loop_nudge_attempts < max_tool_use_nudge_attempts:
                        main_loop_nudge_attempts += 1
                        nudge_reason = "TOOL_USE without function call" if response.finish_reason == FinishReason.TOOL_USE else "UNKNOWN with empty response"
                        self._trace(f"NUDGE_REQUIRED: {nudge_reason} (attempt {main_loop_nudge_attempts}/{max_tool_use_nudge_attempts})")
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
                            response_has_fc = any(p.function_call for p in response.parts)
                            # Recalculate whether we still need to nudge
                            response_is_empty = not response.parts or all(
                                not p.text and not p.function_call for p in response.parts
                            )
                            needs_nudge = (
                                (not response_has_fc and response.finish_reason == FinishReason.TOOL_USE) or
                                (response.finish_reason == FinishReason.UNKNOWN and response_is_empty)
                            )
                            self._trace(f"NUDGE_RESULT: Got response, has_fc={response_has_fc}, still_needs_nudge={needs_nudge}")
                        else:
                            self._trace("NUDGE_NO_RESPONSE: Nudge did not produce a response")
                            break

                    if not response_has_fc:
                        mid_turn_response = self._check_and_handle_mid_turn_prompt(
                            use_streaming, on_output, wrapped_usage_callback, turn_data
                        )
                        if mid_turn_response:
                            # The model responded to the injected prompt - use that response
                            response = mid_turn_response
                            if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                                partial = get_all_text()
                                cancel_msg = "[Generation cancelled]"
                                if on_output and not cancellation_notified:
                                    on_output("system", cancel_msg, "write")
                                self._notify_model_of_cancellation(cancel_msg, partial)
                                return f"{partial}\n\n{cancel_msg}" if partial else cancel_msg

                # Check for more function calls in the new response
                pending_calls = get_pending_function_calls()

            # Collect any remaining text from the final response
            for part in response.parts:
                if part.text:
                    if not use_streaming:
                        if on_output:
                            on_output("model", part.text, "write")
                        # Forward to parent for visibility
                        self._forward_to_parent("MODEL_OUTPUT", part.text)
                        # Notify reliability plugin of model text for pattern detection
                        if self._runtime.reliability_plugin:
                            self._runtime.reliability_plugin.on_model_text(part.text)
                    accumulated_text.append(part.text)

            # Final check for mid-turn prompts before completing the turn
            # This handles prompts that arrived while the model was generating its final response
            self._trace(f"FINAL_MID_TURN_CHECK: Starting drain loop, queue_size={len(self._message_queue)}")
            while True:
                mid_turn_response = self._check_and_handle_mid_turn_prompt(
                    use_streaming, on_output, wrapped_usage_callback, turn_data
                )
                if not mid_turn_response:
                    break

                # Process the response to the injected prompt
                if self._is_cancelled() or mid_turn_response.finish_reason == FinishReason.CANCELLED:
                    cancel_msg = "[Generation cancelled]"
                    if on_output and not cancellation_notified:
                        on_output("system", cancel_msg, "write")
                    final_text = ''.join(accumulated_text) if accumulated_text else ''
                    self._notify_model_of_cancellation(cancel_msg, final_text)
                    return f"{final_text}\n\n{cancel_msg}" if final_text else cancel_msg

                # Collect text from the mid-turn response
                for part in mid_turn_response.parts:
                    if part.text:
                        # Note: In streaming mode, text was already emitted by the callback
                        if not use_streaming:
                            if on_output:
                                on_output("model", part.text, "write")
                            # Forward to parent for visibility
                            self._forward_to_parent("MODEL_OUTPUT", part.text)
                            # Notify reliability plugin of model text for pattern detection
                            if self._runtime.reliability_plugin:
                                self._runtime.reliability_plugin.on_model_text(part.text)
                        accumulated_text.append(part.text)

                # Check if the mid-turn response triggered more function calls
                mid_turn_fc = [p.function_call for p in mid_turn_response.parts if p.function_call]
                if mid_turn_fc:
                    # Process these function calls and continue
                    tool_results = self._execute_function_call_group(
                        mid_turn_fc, turn_data, on_output, cancellation_notified
                    )
                    if self._is_cancelled():
                        cancel_msg = "[Cancelled after tool execution]"
                        if on_output and not cancellation_notified:
                            on_output("system", cancel_msg, "write")
                        self._notify_model_of_cancellation(cancel_msg)
                        return cancel_msg

                    response = self._send_tool_results_and_continue(
                        tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                    )

                    # Continue the main loop with this response if it has function calls
                    pending_calls = [p.function_call for p in response.parts if p.function_call]

                    # Handle edge case: TOOL_USE without function calls after tool results
                    inner_nudge_attempts = 0
                    while (not pending_calls and response.finish_reason == FinishReason.TOOL_USE
                           and inner_nudge_attempts < max_tool_use_nudge_attempts):
                        inner_nudge_attempts += 1
                        self._trace(f"TOOL_USE_WITHOUT_CALL: After tool results, no function call (attempt {inner_nudge_attempts}/{max_tool_use_nudge_attempts})")
                        nudge_prompt = (
                            "<hidden>Your response indicated TOOL_USE but contained no function call. "
                            "Do NOT describe or re-read files. Execute the tool call you just mentioned directly. "
                            "Your next response MUST start with the function call, not text.</hidden>"
                        )
                        self._message_queue.put(nudge_prompt, "system", SourceType.SYSTEM)
                        nudge_response = self._check_and_handle_mid_turn_prompt(
                            use_streaming, on_output, wrapped_usage_callback, turn_data
                        )
                        if nudge_response:
                            response = nudge_response
                            pending_calls = [p.function_call for p in response.parts if p.function_call]
                            self._trace(f"TOOL_USE_NUDGE_RESULT: Got response with {len(pending_calls)} function calls")
                        else:
                            self._trace("TOOL_USE_NUDGE_NO_RESPONSE: Nudge did not produce a response")
                            break

                    while pending_calls:
                        # Re-enter the main processing loop for tool calls
                        current_fc_group = []
                        for part in response.parts:
                            if part.text:
                                if not use_streaming:
                                    if on_output:
                                        on_output("model", part.text, "write")
                                    # Forward to parent for visibility
                                    self._forward_to_parent("MODEL_OUTPUT", part.text)
                                accumulated_text.append(part.text)
                            elif part.function_call:
                                current_fc_group.append(part.function_call)

                        if current_fc_group:
                            tool_results = self._execute_function_call_group(
                                current_fc_group, turn_data, on_output, cancellation_notified
                            )
                            if self._is_cancelled():
                                cancel_msg = "[Cancelled after tool execution]"
                                if on_output and not cancellation_notified:
                                    on_output("system", cancel_msg, "write")
                                self._notify_model_of_cancellation(cancel_msg)
                                return cancel_msg

                            response = self._send_tool_results_and_continue(
                                tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                            )

                        pending_calls = [p.function_call for p in response.parts if p.function_call]

                        # Handle edge case: TOOL_USE without function calls in nested loop
                        nested_nudge_attempts = 0
                        while (not pending_calls and response.finish_reason == FinishReason.TOOL_USE
                               and nested_nudge_attempts < max_tool_use_nudge_attempts):
                            nested_nudge_attempts += 1
                            self._trace(f"TOOL_USE_WITHOUT_CALL: In nested loop, no function call (attempt {nested_nudge_attempts}/{max_tool_use_nudge_attempts})")
                            nudge_prompt = (
                                "<hidden>Your response indicated TOOL_USE but contained no function call. "
                                "Do NOT describe or re-read files. Execute the tool call you just mentioned directly. "
                                "Your next response MUST start with the function call, not text.</hidden>"
                            )
                            self._message_queue.put(nudge_prompt, "system", SourceType.SYSTEM)
                            nudge_response = self._check_and_handle_mid_turn_prompt(
                                use_streaming, on_output, wrapped_usage_callback, turn_data
                            )
                            if nudge_response:
                                response = nudge_response
                                pending_calls = [p.function_call for p in response.parts if p.function_call]
                                self._trace(f"TOOL_USE_NUDGE_RESULT: Got response with {len(pending_calls)} function calls")
                            else:
                                self._trace("TOOL_USE_NUDGE_NO_RESPONSE: Nudge did not produce a response")
                                break

                    # Collect final text from nested response
                    for part in response.parts:
                        if part.text:
                            if not use_streaming:
                                if on_output:
                                    on_output("model", part.text, "write")
                                # Forward to parent for visibility
                                self._forward_to_parent("MODEL_OUTPUT", part.text)
                            accumulated_text.append(part.text)

                # Explicitly continue to check for more queued prompts
                # This ensures we drain the queue even after processing function calls
                continue

            # Safety check: process any prompts that might have been added during the final iteration
            # This handles the race condition where prompts arrive just as the drain loop exits
            final_queue_size = len(self._message_queue)
            if final_queue_size > 0:
                self._trace(f"FINAL_MID_TURN_CHECK: Queue not empty after drain loop! size={final_queue_size}, processing remaining")
                # Process remaining prompts (with a limit to prevent infinite loops)
                safety_iterations = 0
                max_safety_iterations = 10  # Prevent livelock
                while safety_iterations < max_safety_iterations:
                    safety_iterations += 1
                    remaining_response = self._check_and_handle_mid_turn_prompt(
                        use_streaming, on_output, wrapped_usage_callback, turn_data
                    )
                    if not remaining_response:
                        break
                    # Collect text from the response
                    for part in remaining_response.parts:
                        if part.text:
                            if not use_streaming:
                                if on_output:
                                    on_output("model", part.text, "write")
                                self._forward_to_parent("MODEL_OUTPUT", part.text)
                            accumulated_text.append(part.text)
                    # Note: We don't process function calls here to avoid complexity
                    # Any function calls in these responses will be logged but not executed
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
                    # Model responded to streaming updates
                    # Check if it called any tools (including dismiss_stream)
                    streaming_fc = [p.function_call for p in streaming_response.parts if p.function_call]

                    # Collect any text from the response
                    for part in streaming_response.parts:
                        if part.text:
                            if not use_streaming:
                                if on_output:
                                    on_output("model", part.text, "write")
                                self._forward_to_parent("MODEL_OUTPUT", part.text)
                            accumulated_text.append(part.text)

                    # Process any tool calls from the streaming response
                    if streaming_fc:
                        self._trace(f"STREAMING_CONTINUATION: Model called {len(streaming_fc)} tools")
                        # Execute the tools and continue
                        tool_results = self._execute_function_call_group(
                            streaming_fc, turn_data, on_output, cancellation_notified
                        )
                        if not self._is_cancelled():
                            response = self._send_tool_results_and_continue(
                                tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                            )
                            # Collect text from tool continuation response
                            for part in response.parts:
                                if part.text:
                                    if not use_streaming:
                                        if on_output:
                                            on_output("model", part.text, "write")
                                        self._forward_to_parent("MODEL_OUTPUT", part.text)
                                    accumulated_text.append(part.text)
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
            self._forward_to_parent("CANCELLED", "Generation cancelled")
            terminal_event_sent = True
            return "[Generation cancelled]"

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
            self._cancel_token = None
            self._set_activity_phase(ActivityPhase.IDLE)

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

        # Check if parallel execution is enabled
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

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_fc = {
                executor.submit(
                    self._execute_single_tool_for_parallel, fc, captured_channels
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
            fc: The function call (with :stream suffix).
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

        # Ensure session is set in thread-local for plugins that need it
        # This handles cases where tool execution might be in a different thread
        # context than where configure() was called
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
            # Check if this is a streaming tool (name ends with :stream)
            if self._is_streaming_tool(name):
                # Route to streaming execution
                executor_result = self._execute_streaming_tool(fc, on_output)
                tool_span.set_attribute("jaato.tool.streaming", True)
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

                executor_result = self._executor.execute(name, args, call_id=fc.id)

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
            tool_span.set_attribute("jaato.tool.success", fc_success)
            tool_span.set_attribute("jaato.tool.duration_seconds", fc_duration)
            if fc_error_message:
                tool_span.set_attribute("jaato.tool.error", fc_error_message)
                tool_span.set_status_error(fc_error_message)
            else:
                tool_span.set_status_ok()

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
    ) -> _ToolExecutionResult:
        """Execute a single tool for parallel execution.

        Similar to _execute_single_tool but:
        - Uses thread-local callback (not instance-level)
        - Does not emit start/end hooks (handled by caller)
        - Includes telemetry for this thread
        - Propagates session to worker thread's thread-local storage
        - Restores interactive plugin channels captured from spawning thread

        Args:
            fc: The function call to execute.
            captured_channels: Channel references captured from the spawning
                thread by ``_capture_interactive_channels()``.  Restored into
                this worker thread's thread-local storage so that permission
                and clarification requests route through the correct channel
                (e.g. ``ParentBridgedChannel`` for subagents).
        """
        name = fc.name
        args = fc.args

        fc_start = datetime.now()

        # Propagate session to this worker thread's thread-local storage
        # This is critical for plugins (like TODO) that use thread-local to
        # identify the current agent context. Without this, parallel tools
        # would see agent_name=None and fail to find the correct plan.
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

        # Wrap tool execution with telemetry span
        with self._telemetry.tool_span(
            tool_name=name,
            call_id=fc.id or "",
            plugin_type=plugin_type,
        ) as tool_span:
            # Check if this is a streaming tool (name ends with :stream)
            if self._is_streaming_tool(name):
                # Route to streaming execution
                executor_result = self._execute_streaming_tool(fc, None)
                tool_span.set_attribute("jaato.tool.streaming", True)
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

                # Pass callback directly - executor will set it in thread-local
                executor_result = self._executor.execute(
                    name, args, tool_output_callback=tool_output_callback, call_id=fc.id
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
            tool_span.set_attribute("jaato.tool.success", fc_success)
            tool_span.set_attribute("jaato.tool.duration_seconds", fc_duration)
            tool_span.set_attribute("jaato.tool.parallel", True)
            if fc_error_message:
                tool_span.set_attribute("jaato.tool.error", fc_error_message)
                tool_span.set_status_error(fc_error_message)
            else:
                tool_span.set_status_ok()

            # Convention-based telemetry enrichment (parallel path)
            if isinstance(executor_result, tuple) and len(executor_result) == 2:
                result_dict = executor_result[1]
                if isinstance(result_dict, dict):
                    telem = result_dict.get('_telemetry')
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
        # Append tool results to session history
        tool_result_parts = [Part(function_response=r) for r in tool_results]
        self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

        with self._telemetry.llm_span(
            model=self._model_name or "unknown",
            provider=self._provider.name if self._provider else "unknown",
            streaming=use_streaming,
        ) as llm_telemetry:
            if use_streaming:
                # Track first chunk to use "write" for new block, "append" for continuation
                first_chunk_after_tools = [False]  # Use list to allow mutation in closure

                def streaming_callback(chunk: str) -> None:
                    # Check for pending mid-turn prompts during tool result streaming
                    # This mirrors the interrupt detection in the initial streaming callback
                    if self._message_queue.has_parent_messages():
                        self._trace("MID_TURN_INTERRUPT: Detected pending user prompt during tool result streaming")
                        self._mid_turn_interrupt = True
                        if self._cancel_token:
                            self._cancel_token.cancel()

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

                response, _retry_stats = with_retry(
                    lambda: self._provider.complete(
                        self._history.messages,
                        system_instruction=self._system_instruction,
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
                response, _retry_stats = with_retry(
                    lambda: self._provider.complete(
                        self._history.messages,
                        system_instruction=self._system_instruction,
                        tools=self._get_tools_for_provider(),
                    ),
                    context="complete_tool_results",
                    on_retry=self._on_retry,
                    cancel_token=self._cancel_token,
                    provider=self._provider
                )

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
            if response.usage:
                llm_telemetry.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                llm_telemetry.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
                if response.usage.cache_read_tokens is not None:
                    llm_telemetry.set_attribute("gen_ai.usage.cache_read_tokens", response.usage.cache_read_tokens)
                if response.usage.cache_creation_tokens is not None:
                    llm_telemetry.set_attribute("gen_ai.usage.cache_creation_tokens", response.usage.cache_creation_tokens)
                if response.usage.reasoning_tokens is not None:
                    llm_telemetry.set_attribute("gen_ai.usage.reasoning_tokens", response.usage.reasoning_tokens)

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

        Child messages (SourceType.CHILD) - subagent status updates - are
        left in the queue and processed when the agent becomes idle via
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
        ) as llm_telemetry:
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
                response, _retry_stats = with_retry(
                    lambda: self._provider.complete(
                        self._history.messages,
                        system_instruction=self._system_instruction,
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
                self._trace(f"MID_TURN_PROMPT: Provider returned, finish_reason={response.finish_reason if response else 'None'}")
            else:
                response, _retry_stats = with_retry(
                    lambda: self._provider.complete(
                        self._history.messages,
                        system_instruction=self._system_instruction,
                        tools=self._get_tools_for_provider(),
                    ),
                    context="complete_mid_turn",
                    on_retry=self._on_retry,
                    cancel_token=self._cancel_token,
                    provider=self._provider
                )

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
            if response.usage:
                llm_telemetry.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                llm_telemetry.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
                if response.usage.cache_read_tokens is not None:
                    llm_telemetry.set_attribute("gen_ai.usage.cache_read_tokens", response.usage.cache_read_tokens)
                if response.usage.cache_creation_tokens is not None:
                    llm_telemetry.set_attribute("gen_ai.usage.cache_creation_tokens", response.usage.cache_creation_tokens)
                if response.usage.reasoning_tokens is not None:
                    llm_telemetry.set_attribute("gen_ai.usage.reasoning_tokens", response.usage.reasoning_tokens)

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

        # Strip internal metadata keys (prefixed with '_') before sending
        # to the model.  These carry scaffolding like _permission, _multimodal
        # flags, etc. that are not meaningful to the model.
        result_dict = {
            k: v for k, v in result_dict.items()
            if not k.startswith('_')
        }

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

        return enriched_dict

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

    def get_history(self) -> List[Message]:
        """Get current conversation history.

        Returns the session's canonical copy of the history. The session
        is the sole owner of conversation state; providers receive messages
        as parameters to ``complete()``.
        """
        return self._history.messages

    def get_turn_accounting(self) -> List[Dict[str, Any]]:
        """Get token usage and timing per turn."""
        return list(self._turn_accounting)

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
        if not self._provider:
            raise RuntimeError("Session not configured.")

        messages = [Message.from_text(Role.USER, prompt)]
        response = self._provider.complete(messages)
        return response.get_text() or ''

    def send_message_with_parts(
        self,
        parts: List[Part],
        on_output: OutputCallback
    ) -> str:
        """Send a message with custom Part objects."""
        if not self._provider:
            raise RuntimeError("Session not configured.")

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
            ) as llm_telemetry:
                response, _retry_stats = with_retry(
                    lambda: self._provider.complete(
                        self._history.messages,
                        system_instruction=self._system_instruction,
                        tools=self._get_tools_for_provider(),
                    ),
                    context="complete_with_parts",
                    on_retry=self._on_retry,
                    provider=self._provider
                )

                # Record model response in session history
                self._add_model_response_to_history(response)

                self._record_token_usage(response)
                self._accumulate_turn_tokens(response, turn_data)
                # Record token usage to telemetry span
                if response.usage:
                    llm_telemetry.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                    llm_telemetry.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
                    if response.usage.cache_read_tokens is not None:
                        llm_telemetry.set_attribute("gen_ai.usage.cache_read_tokens", response.usage.cache_read_tokens)
                    if response.usage.cache_creation_tokens is not None:
                        llm_telemetry.set_attribute("gen_ai.usage.cache_creation_tokens", response.usage.cache_creation_tokens)
                    if response.usage.reasoning_tokens is not None:
                        llm_telemetry.set_attribute("gen_ai.usage.reasoning_tokens", response.usage.reasoning_tokens)

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

                        executor_result = self._executor.execute(name, args, call_id=fc.id)

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

                # Append tool results to session history
                tool_result_parts = [Part(function_response=r) for r in tool_results]
                self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

                with self._telemetry.llm_span(
                    model=self._model_name or "unknown",
                    provider=self._provider.name if self._provider else "unknown",
                    streaming=False,
                ) as llm_telemetry:
                    response, _retry_stats = with_retry(
                        lambda: self._provider.complete(
                            self._history.messages,
                            system_instruction=self._system_instruction,
                            tools=self._get_tools_for_provider(),
                        ),
                        context="complete_tool_results_parts",
                        on_retry=self._on_retry,
                        provider=self._provider
                    )

                    # Record model response in session history
                    self._add_model_response_to_history(response)

                    self._record_token_usage(response)
                    self._accumulate_turn_tokens(response, turn_data)
                    # Record token usage to telemetry span
                    if response.usage:
                        llm_telemetry.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                        llm_telemetry.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
                        if response.usage.cache_read_tokens is not None:
                            llm_telemetry.set_attribute("gen_ai.usage.cache_read_tokens", response.usage.cache_read_tokens)
                        if response.usage.cache_creation_tokens is not None:
                            llm_telemetry.set_attribute("gen_ai.usage.cache_creation_tokens", response.usage.cache_creation_tokens)
                        if response.usage.reasoning_tokens is not None:
                            llm_telemetry.set_attribute("gen_ai.usage.reasoning_tokens", response.usage.reasoning_tokens)
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
            return None

        context_usage = self.get_context_usage()
        should_gc, reason = self._gc_plugin.should_collect(context_usage, self._gc_config)

        if should_gc and reason:
            self._trace(
                f"GC_BEFORE_SEND: triggering GC (reason={reason.value}, "
                f"usage={context_usage.get('percent_used', 0):.1f}%)"
            )
            history = self.get_history()
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

                # Sync budget with GC changes
                self._apply_gc_removal_list(result)
                self._emit_instruction_budget_update()

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
        )

    def _restore_session_state(self, state: SessionState) -> None:
        """Restore session state from a SessionState."""
        self.reset_session(state.history)
        self._turn_accounting = list(state.turn_accounting)

    def _notify_session_turn_complete(self) -> None:
        """Notify session plugin that a turn completed."""
        if not self._session_plugin or not self._session_config:
            return

        state = self._get_session_state()

        if hasattr(self._session_plugin, 'increment_turn_count'):
            self._session_plugin.increment_turn_count()

        self._session_plugin.on_turn_complete(state, self._session_config)

    def close_session(self) -> None:
        """Close the current session."""
        if self._session_plugin and self._session_config:
            state = self._get_session_state()
            self._session_plugin.on_session_end(state, self._session_config)


__all__ = ['JaatoSession']
