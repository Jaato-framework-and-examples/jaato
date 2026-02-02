"""Reliability plugin for tracking tool failures and adaptive trust."""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base import (
    CommandCompletion,
    CommandParameter,
    OutputCallback,
    ToolPlugin,
    UserCommand,
)
from ..model_provider.types import ToolSchema
from .types import (
    EscalationInfo,
    EscalationRule,
    FailureKey,
    FailureRecord,
    FailureSeverity,
    ModelSwitchConfig,
    ModelSwitchStrategy,
    ModelSwitchSuggestion,
    ModelToolProfile,
    ReliabilityConfig,
    ToolReliabilityState,
    TrustState,
    classify_failure,
)
from .persistence import (
    ReliabilityPersistence,
    SessionSettings,
    SessionReliabilityState,
)

logger = logging.getLogger(__name__)


class ReliabilityPlugin:
    """Plugin that tracks tool failures and adjusts trust dynamically.

    This plugin monitors tool execution results and maintains reliability
    profiles for each tool+parameters combination. When failures exceed
    configured thresholds, tools are escalated to require explicit approval.
    """

    def __init__(self, config: Optional[ReliabilityConfig] = None):
        self._config = config or ReliabilityConfig()

        # State tracking
        self._tool_states: Dict[str, ToolReliabilityState] = {}
        self._failure_history: List[FailureRecord] = []

        # Session context
        self._session_id: str = ""
        self._turn_index: int = 0
        self._session_settings: SessionSettings = SessionSettings()

        # Persistence
        self._persistence: Optional[ReliabilityPersistence] = None

        # Callbacks
        self._output_callback: Optional[OutputCallback] = None

        # Plugin registry for looking up plugin names
        self._registry = None

        # Workspace path for persistence
        self._workspace_path: Optional[str] = None

        # Event callbacks for UI integration
        # on_escalated: (failure_key, state, reason) -> None
        self._on_escalated: Optional[Callable[[str, TrustState, str], None]] = None
        # on_recovered: (failure_key, state) -> None
        self._on_recovered: Optional[Callable[[str, TrustState], None]] = None
        # on_blocked: (failure_key, reason) -> None
        self._on_blocked: Optional[Callable[[str, str], None]] = None
        # on_model_switch_suggested: (suggestion) -> None
        self._on_model_switch_suggested: Optional[Callable[[ModelSwitchSuggestion], None]] = None

        # Model reliability tracking
        self._model_profiles: Dict[Tuple[str, str], ModelToolProfile] = {}  # (model, failure_key) -> profile
        self._current_model: str = ""
        self._model_switch_config = ModelSwitchConfig()
        self._available_models: List[str] = []

    @property
    def name(self) -> str:
        return "reliability"

    # -------------------------------------------------------------------------
    # Plugin Protocol Implementation
    # -------------------------------------------------------------------------

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for model-callable tools."""
        return [
            ToolSchema(
                name="reliability_status",
                description="Check the reliability status of tools. Shows which tools are escalated, recovering, or blocked due to failures.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Specific tool to check (optional, shows all if omitted)"
                        },
                        "include_history": {
                            "type": "boolean",
                            "description": "Include recent failure history",
                            "default": False
                        }
                    }
                }
            ),
        ]

    def get_executors(self) -> Dict[str, Callable]:
        """Return executors for model-callable tools."""
        return {
            "reliability_status": self._execute_status,
        }

    def get_auto_approved_tools(self) -> List[str]:
        """These tools don't need permission checks."""
        return ["reliability_status"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands."""
        return [
            UserCommand(
                name="reliability",
                description="Manage reliability tracking and tool trust",
                parameters=[
                    CommandParameter(name="subcommand", description="Subcommand to run", required=False),
                    CommandParameter(name="args", description="Arguments", required=False, capture_rest=True),
                ],
            ),
        ]

    def get_command_completions(self, command_parts: List[str]) -> List[CommandCompletion]:
        """Provide completions for reliability commands."""
        if len(command_parts) == 1:
            return [
                CommandCompletion("status", "Show reliability status for all tools"),
                CommandCompletion("recovery", "Set recovery mode (auto|ask)"),
                CommandCompletion("reset", "Reset a tool to trusted state"),
                CommandCompletion("history", "Show recent failure history"),
                CommandCompletion("config", "Show current configuration"),
                CommandCompletion("settings", "Show or save settings"),
                CommandCompletion("model", "Model-specific reliability tracking"),
            ]

        if len(command_parts) == 2:
            subcommand = command_parts[1]
            if subcommand == "recovery":
                return [
                    CommandCompletion("auto", "Automatically recover tools after cooldown"),
                    CommandCompletion("ask", "Prompt before recovering tools"),
                    CommandCompletion("save", "Save recovery setting"),
                ]
            elif subcommand == "reset":
                # Return list of escalated/blocked tools
                return [
                    CommandCompletion(key, f"Reset {state.tool_name} ({state.state.value})")
                    for key, state in self._tool_states.items()
                    if state.state in (TrustState.ESCALATED, TrustState.BLOCKED)
                ]
            elif subcommand == "settings":
                return [
                    CommandCompletion("show", "Show effective settings"),
                    CommandCompletion("save", "Save settings to workspace or user"),
                    CommandCompletion("clear", "Clear settings at a level"),
                ]
            elif subcommand == "model":
                return [
                    CommandCompletion("status", "Show model reliability summary"),
                    CommandCompletion("compare", "Compare model reliability"),
                    CommandCompletion("suggest", "Enable model switch suggestions"),
                    CommandCompletion("auto", "Enable automatic model switching"),
                    CommandCompletion("disabled", "Disable model switching"),
                ]

        if len(command_parts) == 3:
            subcommand = command_parts[1]
            if subcommand == "recovery" and command_parts[2] == "save":
                return [
                    CommandCompletion("workspace", "Save to workspace (.jaato/reliability.json)"),
                    CommandCompletion("user", "Save as user default (~/.jaato/reliability.json)"),
                ]
            elif subcommand == "settings":
                arg = command_parts[2]
                if arg == "save":
                    return [
                        CommandCompletion("workspace", "Save to workspace"),
                        CommandCompletion("user", "Save as user default"),
                    ]
                elif arg == "clear":
                    return [
                        CommandCompletion("workspace", "Clear workspace settings"),
                        CommandCompletion("session", "Clear session overrides"),
                    ]

        return []

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions about reliability tracking."""
        return None  # No special instructions needed

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin."""
        if config:
            # Could load config overrides here
            pass
        logger.info("Reliability plugin initialized")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        # Could persist state here
        logger.info("Reliability plugin shutdown")

    # -------------------------------------------------------------------------
    # Auto-wiring Methods
    # -------------------------------------------------------------------------

    def set_plugin_registry(self, registry) -> None:
        """Called by registry during exposure."""
        self._registry = registry

    def set_workspace_path(self, path: str) -> None:
        """Called when workspace path is set. Loads persisted state."""
        self._workspace_path = path

        # Initialize persistence
        self._persistence = ReliabilityPersistence(workspace_path=path)

        # Load persisted tool states
        persisted_states = self._persistence.load_tool_states()
        self._tool_states.update(persisted_states)

        # Load persisted failure history
        persisted_history = self._persistence.load_failure_history()
        self._failure_history.extend(persisted_history)

        # Load effective settings
        recovery_mode = self._persistence.get_effective_setting("recovery_mode", self._session_settings)
        if recovery_mode is not None:
            self._config.enable_auto_recovery = recovery_mode == "auto"

        logger.info(f"Loaded {len(persisted_states)} tool states and {len(persisted_history)} history records")

    def set_output_callback(self, callback: OutputCallback) -> None:
        """Set callback for output messages."""
        self._output_callback = callback

    def set_session_context(self, session_id: str) -> None:
        """Set session context for failure tracking."""
        self._session_id = session_id

    def set_turn_index(self, turn_index: int) -> None:
        """Update current turn index."""
        self._turn_index = turn_index

    def set_reliability_hooks(
        self,
        on_escalated: Optional[Callable[[str, TrustState, str], None]] = None,
        on_recovered: Optional[Callable[[str, TrustState], None]] = None,
        on_blocked: Optional[Callable[[str, str], None]] = None,
        on_model_switch_suggested: Optional[Callable[[ModelSwitchSuggestion], None]] = None,
    ) -> None:
        """Set hooks for reliability lifecycle events.

        These hooks enable UI integration by notifying when tools are
        escalated, recovered, or blocked.

        Args:
            on_escalated: Called when a tool is escalated to require approval.
                Signature: (failure_key, state, reason) -> None
            on_recovered: Called when a tool recovers to trusted state.
                Signature: (failure_key, state) -> None
            on_blocked: Called when a tool is blocked due to security concern.
                Signature: (failure_key, reason) -> None
            on_model_switch_suggested: Called when a model switch might improve reliability.
                Signature: (suggestion: ModelSwitchSuggestion) -> None
        """
        self._on_escalated = on_escalated
        self._on_recovered = on_recovered
        self._on_blocked = on_blocked
        self._on_model_switch_suggested = on_model_switch_suggested

    def set_model_context(
        self,
        current_model: str,
        available_models: Optional[List[str]] = None,
    ) -> None:
        """Set model context for model-specific reliability tracking.

        Args:
            current_model: Name of the currently active model
            available_models: List of models that can be switched to
        """
        self._current_model = current_model
        if available_models is not None:
            self._available_models = available_models

    def set_model_switch_config(self, config: ModelSwitchConfig) -> None:
        """Set model switch configuration."""
        self._model_switch_config = config

    # -------------------------------------------------------------------------
    # Core Failure Tracking
    # -------------------------------------------------------------------------

    def on_tool_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        success: bool,
        result: Any,
        call_id: str = "",
        plugin_name: str = "",
    ) -> Optional[ToolReliabilityState]:
        """Hook called after tool execution. Updates reliability state.

        Args:
            tool_name: Name of the tool that was executed
            args: Arguments passed to the tool
            success: Whether the tool execution succeeded (from executor)
            result: The tool result (may contain error info even if success=True)
            call_id: Unique identifier for this call
            plugin_name: Name of the plugin that provided the tool

        Returns:
            Updated ToolReliabilityState if state changed, None otherwise
        """
        # Create failure key from invocation
        failure_key = FailureKey.from_invocation(tool_name, args)
        key_str = failure_key.to_string()

        # Determine if this is actually a failure
        is_failure = not success or self._is_error_result(result)

        # Track model-specific reliability if model context is set
        if self._current_model:
            self._track_model_attempt(key_str, tool_name, is_failure)

        # Get or create state
        state = self._get_or_create_state(key_str, tool_name)

        if is_failure:
            return self._handle_failure(state, failure_key, result, call_id, plugin_name)
        else:
            return self._handle_success(state)

    def _is_error_result(self, result: Any) -> bool:
        """Check if a result indicates an error even if execution 'succeeded'."""
        if not isinstance(result, dict):
            return False
        return "error" in result or result.get("status_code", 200) >= 400

    def _get_or_create_state(self, key_str: str, tool_name: str) -> ToolReliabilityState:
        """Get existing state or create new one."""
        if key_str not in self._tool_states:
            self._tool_states[key_str] = ToolReliabilityState(
                failure_key=key_str,
                tool_name=tool_name,
            )
        return self._tool_states[key_str]

    def _handle_failure(
        self,
        state: ToolReliabilityState,
        failure_key: FailureKey,
        result: Any,
        call_id: str,
        plugin_name: str,
    ) -> ToolReliabilityState:
        """Process a failure and potentially escalate."""
        now = datetime.now()

        # Classify the failure
        error_dict = result if isinstance(result, dict) else {"error": str(result)}
        severity, weight = classify_failure(failure_key.tool_name, error_dict)

        # Create failure record
        record = FailureRecord(
            failure_key=failure_key.to_string(),
            tool_name=failure_key.tool_name,
            plugin_name=plugin_name,
            timestamp=now,
            parameter_signature=failure_key.parameter_signature,
            error_type=type(result).__name__ if not isinstance(result, dict) else "error",
            error_message=str(error_dict.get("error", "")),
            severity=severity,
            weight=weight,
            call_id=call_id,
            session_id=self._session_id,
            turn_index=self._turn_index,
            http_status=error_dict.get("http_status") or error_dict.get("status_code"),
        )

        # Add to history
        self._failure_history.append(record)
        self._prune_history()

        # Update state counters
        state.consecutive_failures += 1
        state.total_failures += 1
        state.last_failure = now

        # Get applicable rule
        rule = self._config.get_rule_for_tool(
            failure_key.tool_name,
            plugin_name,
            failure_key.parameter_signature.split("|")[0].split("=")[1] if "domain=" in failure_key.parameter_signature else None
        )

        # Count failures in window (only matching severity)
        if severity in rule.severity_filter:
            state.failures_in_window = self._count_failures_in_window(
                failure_key.to_string(),
                rule.window_seconds,
                rule.severity_filter,
            )

        # Check escalation conditions
        should_escalate = False
        reason = ""

        # Use small tolerance for floating-point comparison due to time decay
        if rule.count_threshold and state.failures_in_window >= (rule.count_threshold - 0.01):
            should_escalate = True
            reason = f"{int(state.failures_in_window)} failures in {rule.window_seconds}s"

        elif rule.consecutive_threshold and state.consecutive_failures >= rule.consecutive_threshold:
            should_escalate = True
            reason = f"{state.consecutive_failures} consecutive failures"

        elif rule.rate_threshold and state.total_failures + state.total_successes >= 5:
            rate = state.total_failures / (state.total_failures + state.total_successes)
            if rate >= rule.rate_threshold:
                should_escalate = True
                reason = f"{rate:.0%} failure rate"

        # Critical severity always escalates to BLOCKED
        if severity == FailureSeverity.SECURITY:
            should_escalate = True
            was_blocked = state.state == TrustState.BLOCKED
            state.state = TrustState.BLOCKED
            reason = "Security concern detected"

            # Emit blocked hook
            if not was_blocked and self._on_blocked:
                self._on_blocked(failure_key.to_string(), reason)

        # Apply escalation if needed
        if should_escalate and state.state == TrustState.TRUSTED:
            state.state = TrustState.ESCALATED
            state.escalated_at = now
            state.escalation_reason = reason
            state.escalation_expires = now + timedelta(seconds=rule.escalation_duration_seconds)
            state.successes_needed = rule.success_count_to_recover

            if rule.notify_user:
                self._notify_escalation(state)

            # Emit escalated hook
            if self._on_escalated:
                self._on_escalated(failure_key.to_string(), state.state, reason)

        # If recovering, reset recovery progress
        elif state.state == TrustState.RECOVERING:
            state.state = TrustState.ESCALATED
            state.recovery_started = None
            state.successes_since_recovery = 0

            # Emit escalated hook (re-escalation from recovering)
            if self._on_escalated:
                self._on_escalated(
                    failure_key.to_string(),
                    state.state,
                    "Recovery interrupted by failure"
                )

        logger.debug(
            f"Failure recorded: {failure_key.to_string()} "
            f"severity={severity.value} state={state.state.value}"
        )

        return state

    def _handle_success(self, state: ToolReliabilityState) -> Optional[ToolReliabilityState]:
        """Process a success, potentially recovering trust."""
        now = datetime.now()

        # Update counters
        state.consecutive_failures = 0
        state.total_successes += 1
        state.last_success = now

        # Get applicable rule
        rule = self._config.get_rule_for_tool(state.tool_name)

        if state.state == TrustState.ESCALATED:
            # Check if cooldown period passed
            if state.escalation_expires and now >= state.escalation_expires:
                state.state = TrustState.RECOVERING
                state.recovery_started = now
                state.successes_since_recovery = 1
                state.successes_needed = rule.success_count_to_recover
                logger.debug(f"Tool {state.failure_key} entering recovery")
                return state

        elif state.state == TrustState.RECOVERING:
            state.successes_since_recovery += 1

            if state.successes_since_recovery >= state.successes_needed:
                # Full recovery
                state.state = TrustState.TRUSTED
                state.escalated_at = None
                state.escalation_reason = None
                state.escalation_expires = None
                state.recovery_started = None
                state.successes_since_recovery = 0
                self._notify_recovery(state)
                logger.info(f"Tool {state.failure_key} recovered to TRUSTED")

                # Emit recovered hook
                if self._on_recovered:
                    self._on_recovered(state.failure_key, state.state)

                return state

        return None  # No significant state change

    # -------------------------------------------------------------------------
    # Model-Specific Reliability Tracking
    # -------------------------------------------------------------------------

    def _track_model_attempt(self, failure_key: str, tool_name: str, is_failure: bool) -> None:
        """Track a model-specific tool attempt."""
        profile_key = (self._current_model, failure_key)

        # Get or create profile
        if profile_key not in self._model_profiles:
            self._model_profiles[profile_key] = ModelToolProfile(
                model_name=self._current_model,
                failure_key=failure_key,
            )

        profile = self._model_profiles[profile_key]
        profile.record_attempt(success=not is_failure)

        # Check for model switch suggestion on failure
        if is_failure and self._model_switch_config.strategy != ModelSwitchStrategy.DISABLED:
            self._check_model_switch_suggestion(failure_key, tool_name)

    def _check_model_switch_suggestion(self, failure_key: str, tool_name: str) -> None:
        """Check if switching models might improve reliability for this tool."""
        config = self._model_switch_config

        # Get current model's profile
        current_profile = self._model_profiles.get((self._current_model, failure_key))
        if not current_profile:
            return

        # Only suggest after reaching failure threshold
        if current_profile.failures < config.failure_threshold:
            return

        # Find better model from available models
        best_suggestion: Optional[ModelSwitchSuggestion] = None

        for model_name in self._available_models:
            if model_name == self._current_model:
                continue

            other_profile = self._model_profiles.get((model_name, failure_key))
            if not other_profile:
                # No data for this model - might be worth trying if preferred
                if model_name in config.preferred_models:
                    suggestion = ModelSwitchSuggestion(
                        current_model=self._current_model,
                        suggested_model=model_name,
                        failure_key=failure_key,
                        tool_name=tool_name,
                        current_success_rate=current_profile.success_rate,
                        suggested_success_rate=1.0,  # Unknown, assume good
                        improvement=1.0 - current_profile.success_rate,
                        reason=f"Preferred model with no failure history for this tool",
                        confidence="low",
                    )
                    if not best_suggestion or suggestion.improvement > best_suggestion.improvement:
                        best_suggestion = suggestion
                continue

            # Need minimum attempts for comparison
            if other_profile.total_attempts < config.min_attempts:
                continue

            # Calculate improvement
            improvement = other_profile.success_rate - current_profile.success_rate

            if improvement >= config.min_success_rate_diff:
                # Determine confidence based on sample size
                if other_profile.total_attempts >= 10:
                    confidence = "high"
                elif other_profile.total_attempts >= 5:
                    confidence = "medium"
                else:
                    confidence = "low"

                suggestion = ModelSwitchSuggestion(
                    current_model=self._current_model,
                    suggested_model=model_name,
                    failure_key=failure_key,
                    tool_name=tool_name,
                    current_success_rate=current_profile.success_rate,
                    suggested_success_rate=other_profile.success_rate,
                    improvement=improvement,
                    reason=f"{other_profile.success_rate:.0%} success vs {current_profile.success_rate:.0%}",
                    confidence=confidence,
                )

                if not best_suggestion or suggestion.improvement > best_suggestion.improvement:
                    best_suggestion = suggestion

        # Emit suggestion if found
        if best_suggestion and self._on_model_switch_suggested:
            self._on_model_switch_suggested(best_suggestion)

    def get_model_profile(
        self, model_name: str, failure_key: str
    ) -> Optional[ModelToolProfile]:
        """Get reliability profile for a model+tool combination."""
        return self._model_profiles.get((model_name, failure_key))

    def get_all_model_profiles(self) -> Dict[Tuple[str, str], ModelToolProfile]:
        """Get all model reliability profiles."""
        return dict(self._model_profiles)

    def get_model_reliability_summary(
        self, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of model reliability across all tools.

        Args:
            model_name: Specific model to summarize, or None for current model

        Returns:
            Summary dict with total_attempts, success_rate, tools_tracked, etc.
        """
        target_model = model_name or self._current_model
        if not target_model:
            return {"error": "No model context set"}

        profiles = [
            p for (m, _), p in self._model_profiles.items()
            if m == target_model
        ]

        if not profiles:
            return {
                "model": target_model,
                "total_attempts": 0,
                "total_failures": 0,
                "success_rate": 1.0,
                "tools_tracked": 0,
            }

        total_attempts = sum(p.total_attempts for p in profiles)
        total_failures = sum(p.failures for p in profiles)

        return {
            "model": target_model,
            "total_attempts": total_attempts,
            "total_failures": total_failures,
            "success_rate": 1.0 - (total_failures / total_attempts) if total_attempts > 0 else 1.0,
            "tools_tracked": len(profiles),
            "problematic_tools": [
                p.failure_key for p in profiles
                if p.success_rate < 0.7 and p.total_attempts >= 3
            ],
        }

    def get_model_switch_suggestion(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[ModelSwitchSuggestion]:
        """Manually check for model switch suggestion for a specific tool.

        Args:
            tool_name: Name of the tool
            args: Tool arguments (used to create failure key)

        Returns:
            ModelSwitchSuggestion if a better model is available, None otherwise
        """
        if not self._current_model or not self._available_models:
            return None

        failure_key = FailureKey.from_invocation(tool_name, args)
        key_str = failure_key.to_string()
        config = self._model_switch_config

        current_profile = self._model_profiles.get((self._current_model, key_str))
        if not current_profile or current_profile.failures < config.failure_threshold:
            return None

        best_suggestion: Optional[ModelSwitchSuggestion] = None

        for model_name in self._available_models:
            if model_name == self._current_model:
                continue

            other_profile = self._model_profiles.get((model_name, key_str))
            if not other_profile or other_profile.total_attempts < config.min_attempts:
                continue

            improvement = other_profile.success_rate - current_profile.success_rate

            if improvement >= config.min_success_rate_diff:
                confidence = "high" if other_profile.total_attempts >= 10 else "medium" if other_profile.total_attempts >= 5 else "low"

                suggestion = ModelSwitchSuggestion(
                    current_model=self._current_model,
                    suggested_model=model_name,
                    failure_key=key_str,
                    tool_name=tool_name,
                    current_success_rate=current_profile.success_rate,
                    suggested_success_rate=other_profile.success_rate,
                    improvement=improvement,
                    reason=f"{other_profile.success_rate:.0%} success vs {current_profile.success_rate:.0%}",
                    confidence=confidence,
                )

                if not best_suggestion or suggestion.improvement > best_suggestion.improvement:
                    best_suggestion = suggestion

        return best_suggestion

    def _count_failures_in_window(
        self,
        failure_key: str,
        window_seconds: int,
        severity_filter: set,
    ) -> float:
        """Count weighted failures in time window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)

        total = 0.0
        for record in self._failure_history:
            if record.failure_key != failure_key:
                continue
            if record.timestamp < cutoff:
                continue
            if record.severity not in severity_filter:
                continue

            # Apply time decay: 10% reduction per hour
            age_seconds = (now - record.timestamp).total_seconds()
            decay = max(0.1, 1.0 - (age_seconds / 3600) * 0.1)

            total += record.weight * decay

        return total

    def _prune_history(self) -> None:
        """Remove old history entries."""
        if len(self._failure_history) > self._config.max_history_entries:
            self._failure_history = self._failure_history[-self._config.max_history_entries:]

    # -------------------------------------------------------------------------
    # State Query Methods
    # -------------------------------------------------------------------------

    def get_state(self, failure_key: str) -> Optional[ToolReliabilityState]:
        """Get reliability state for a tool+params combination."""
        return self._tool_states.get(failure_key)

    def get_all_states(self) -> Dict[str, ToolReliabilityState]:
        """Get all tracked states."""
        return dict(self._tool_states)

    def get_escalated_tools(self) -> List[ToolReliabilityState]:
        """Get list of tools that are escalated or blocked."""
        return [
            state for state in self._tool_states.values()
            if state.state in (TrustState.ESCALATED, TrustState.BLOCKED, TrustState.RECOVERING)
        ]

    def is_escalated(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if a tool invocation is escalated.

        Returns:
            Tuple of (is_escalated, reason)
        """
        failure_key = FailureKey.from_invocation(tool_name, args)
        state = self._tool_states.get(failure_key.to_string())

        if not state:
            return (False, None)

        if state.state in (TrustState.ESCALATED, TrustState.BLOCKED):
            return (True, state.escalation_reason)

        return (False, None)

    def check_escalation(self, tool_name: str, args: Dict[str, Any]) -> EscalationInfo:
        """Check escalation status with rich details for permission prompts.

        This method provides detailed escalation information including:
        - Whether the tool is escalated
        - The reason for escalation
        - Recovery progress if applicable
        - Display-ready summary for permission prompts

        Args:
            tool_name: Name of the tool to check
            args: Arguments for the tool invocation

        Returns:
            EscalationInfo with full escalation context
        """
        failure_key = FailureKey.from_invocation(tool_name, args)
        key_str = failure_key.to_string()
        state = self._tool_states.get(key_str)

        if not state:
            return EscalationInfo(is_escalated=False)

        if state.state == TrustState.TRUSTED:
            return EscalationInfo(is_escalated=False, state=state.state)

        # Build escalation info
        is_escalated = state.state in (TrustState.ESCALATED, TrustState.BLOCKED, TrustState.RECOVERING)

        # Determine severity label
        if state.state == TrustState.BLOCKED:
            severity_label = "🚫 BLOCKED"
        elif state.state == TrustState.ESCALATED:
            severity_label = "⚠ ESCALATED"
        elif state.state == TrustState.RECOVERING:
            severity_label = "↻ RECOVERING"
        else:
            severity_label = None

        # Build window description
        window_desc = None
        rule = self._config.get_rule_for_tool(tool_name)
        if state.failures_in_window > 0:
            window_hours = rule.window_seconds / 3600
            if window_hours >= 1:
                window_desc = f"{int(state.failures_in_window)} failures in {window_hours:.0f}h"
            else:
                window_desc = f"{int(state.failures_in_window)} failures in {rule.window_seconds}s"

        # Recovery progress
        recovery_progress = None
        recovery_hint = None
        if state.state == TrustState.RECOVERING:
            recovery_progress = f"{state.successes_since_recovery}/{state.successes_needed} successes"
            remaining = state.successes_needed - state.successes_since_recovery
            if remaining > 0:
                recovery_hint = f"{remaining} more success{'es' if remaining > 1 else ''} needed to recover"
        elif state.state == TrustState.ESCALATED and state.escalation_expires:
            remaining_time = state.escalation_expires - datetime.now()
            if remaining_time.total_seconds() > 0:
                minutes = int(remaining_time.total_seconds() / 60)
                recovery_hint = f"Cooldown: {minutes}m before recovery eligible"

        # Build summary
        summary_parts = []
        if state.tool_name:
            summary_parts.append(f"Tool '{state.tool_name}'")
        if state.escalation_reason:
            summary_parts.append(state.escalation_reason)
        summary = " - ".join(summary_parts) if summary_parts else None

        return EscalationInfo(
            is_escalated=is_escalated,
            state=state.state,
            reason=state.escalation_reason,
            failure_count=state.total_failures,
            window_description=window_desc,
            recovery_progress=recovery_progress,
            recovery_hint=recovery_hint,
            severity_label=severity_label,
            summary=summary,
        )

    # -------------------------------------------------------------------------
    # Manual State Management
    # -------------------------------------------------------------------------

    def reset_tool(self, failure_key: str) -> bool:
        """Manually reset a tool to TRUSTED state."""
        if failure_key not in self._tool_states:
            return False

        state = self._tool_states[failure_key]
        state.state = TrustState.TRUSTED
        state.escalated_at = None
        state.escalation_reason = None
        state.escalation_expires = None
        state.recovery_started = None
        state.successes_since_recovery = 0
        state.consecutive_failures = 0
        state.failures_in_window = 0.0

        logger.info(f"Tool {failure_key} manually reset to TRUSTED")
        return True

    def reset_all(self) -> int:
        """Reset all tools to TRUSTED. Returns count of reset tools."""
        count = 0
        for state in self._tool_states.values():
            if state.state != TrustState.TRUSTED:
                state.state = TrustState.TRUSTED
                state.escalated_at = None
                state.escalation_reason = None
                state.escalation_expires = None
                state.recovery_started = None
                state.successes_since_recovery = 0
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Notifications
    # -------------------------------------------------------------------------

    def _notify_escalation(self, state: ToolReliabilityState) -> None:
        """Notify user when a tool is escalated."""
        if not self._output_callback:
            return

        expires_str = ""
        if state.escalation_expires:
            remaining = state.escalation_expires - datetime.now()
            minutes = int(remaining.total_seconds() / 60)
            expires_str = f"\n   Duration: {minutes}m until recovery eligible"

        self._output_callback(
            "reliability",
            f"Tool '{state.tool_name}' escalated to require approval\n"
            f"   Key: {state.failure_key}\n"
            f"   Reason: {state.escalation_reason}"
            f"{expires_str}",
            "write"
        )

    def _notify_recovery(self, state: ToolReliabilityState) -> None:
        """Notify user when a tool recovers."""
        if not self._output_callback:
            return

        self._output_callback(
            "reliability",
            f"Tool '{state.tool_name}' recovered to TRUSTED\n"
            f"   Key: {state.failure_key}",
            "write"
        )

    # -------------------------------------------------------------------------
    # Tool Executors
    # -------------------------------------------------------------------------

    def _execute_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reliability_status tool."""
        tool_name = args.get("tool_name")
        include_history = args.get("include_history", False)

        if tool_name:
            # Show specific tool
            matching = [
                state for key, state in self._tool_states.items()
                if tool_name in key or tool_name == state.tool_name
            ]
            if not matching:
                return {"message": f"No reliability data for tool '{tool_name}'"}

            result = {"tools": [self._format_state(s) for s in matching]}
        else:
            # Show all non-trusted
            escalated = self.get_escalated_tools()
            if not escalated:
                return {
                    "message": "All tools are TRUSTED",
                    "total_tracked": len(self._tool_states),
                }

            result = {
                "escalated_tools": [self._format_state(s) for s in escalated],
                "total_tracked": len(self._tool_states),
                "total_escalated": len(escalated),
            }

        if include_history:
            recent = self._failure_history[-10:]
            result["recent_failures"] = [
                {
                    "tool": r.tool_name,
                    "key": r.failure_key,
                    "error": r.error_message[:100],
                    "severity": r.severity.value,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in reversed(recent)
            ]

        return result

    def _format_state(self, state: ToolReliabilityState) -> Dict[str, Any]:
        """Format state for display."""
        result = {
            "key": state.failure_key,
            "tool": state.tool_name,
            "state": state.state.value,
            "failures": state.total_failures,
            "successes": state.total_successes,
        }

        if state.escalation_reason:
            result["reason"] = state.escalation_reason

        if state.state == TrustState.RECOVERING:
            result["recovery_progress"] = f"{state.successes_since_recovery}/{state.successes_needed}"

        if state.escalation_expires:
            remaining = state.escalation_expires - datetime.now()
            if remaining.total_seconds() > 0:
                result["cooldown_remaining"] = f"{int(remaining.total_seconds() / 60)}m"

        return result

    # -------------------------------------------------------------------------
    # User Command Handler
    # -------------------------------------------------------------------------

    def handle_command(self, subcommand: str, args: str) -> str:
        """Handle user commands."""
        if not subcommand or subcommand == "status":
            return self._cmd_status(args)
        elif subcommand == "recovery":
            return self._cmd_recovery(args)
        elif subcommand == "reset":
            return self._cmd_reset(args)
        elif subcommand == "history":
            return self._cmd_history(args)
        elif subcommand == "config":
            return self._cmd_config()
        elif subcommand == "settings":
            return self._cmd_settings(args)
        elif subcommand == "model":
            return self._cmd_model(args)
        else:
            return f"Unknown subcommand: {subcommand}\nUse: status, recovery, reset, history, config, settings, model"

    def _cmd_status(self, args: str) -> str:
        """Show reliability status."""
        escalated = self.get_escalated_tools()

        if not escalated:
            return f"All tools TRUSTED ({len(self._tool_states)} tracked)"

        lines = ["Reliability Status:", "-" * 60]
        for state in escalated:
            status_info = f"{state.state.value}"
            if state.state == TrustState.RECOVERING:
                status_info += f" ({state.successes_since_recovery}/{state.successes_needed})"

            lines.append(f"  {state.failure_key}")
            lines.append(f"    State: {status_info}")
            if state.escalation_reason:
                lines.append(f"    Reason: {state.escalation_reason}")

        lines.append("-" * 60)
        lines.append(f"{len(escalated)} escalated, {len(self._tool_states) - len(escalated)} trusted")

        return "\n".join(lines)

    def _cmd_recovery(self, args: str) -> str:
        """Set recovery mode."""
        parts = args.strip().lower().split() if args else []

        if not parts:
            current = "auto" if self._config.enable_auto_recovery else "ask"
            return f"Current recovery mode: {current}\nUsage: reliability recovery auto|ask [save workspace|user]"

        mode = parts[0]
        if mode == "auto":
            self._config.enable_auto_recovery = True
            self._session_settings.recovery_mode = "auto"
            result = "Recovery mode set to AUTO"
        elif mode == "ask":
            self._config.enable_auto_recovery = False
            self._session_settings.recovery_mode = "ask"
            result = "Recovery mode set to ASK"
        elif mode == "save":
            # Handle "recovery save workspace|user"
            if len(parts) < 2:
                return "Usage: reliability recovery save workspace|user"
            return self._save_recovery_setting(parts[1])
        else:
            current = "auto" if self._config.enable_auto_recovery else "ask"
            return f"Current recovery mode: {current}\nUsage: reliability recovery auto|ask [save workspace|user]"

        # Check for save subcommand
        if len(parts) >= 3 and parts[1] == "save":
            save_result = self._save_recovery_setting(parts[2])
            return f"{result}\n{save_result}"

        return result

    def _save_recovery_setting(self, level: str) -> str:
        """Save recovery setting to workspace or user level."""
        if not self._persistence:
            return "Error: Persistence not initialized (no workspace set)"

        mode = "auto" if self._config.enable_auto_recovery else "ask"

        if level == "workspace":
            if self._persistence.save_setting_to_workspace("recovery_mode", mode):
                return f"Recovery mode '{mode}' saved to workspace"
            return "Error: Failed to save to workspace"
        elif level == "user":
            if self._persistence.save_setting_to_user("recovery_mode", mode):
                return f"Recovery mode '{mode}' saved as user default"
            return "Error: Failed to save to user"
        else:
            return f"Unknown level: {level}. Use 'workspace' or 'user'"

    def _cmd_reset(self, args: str) -> str:
        """Reset a tool."""
        key = args.strip() if args else ""
        if not key:
            return "Usage: reliability reset <failure_key>\nUse tab completion to see escalated tools"

        if key == "all":
            count = self.reset_all()
            return f"Reset {count} tools to TRUSTED"

        if self.reset_tool(key):
            return f"Reset '{key}' to TRUSTED"
        else:
            return f"Tool '{key}' not found in reliability tracking"

    def _cmd_history(self, args: str) -> str:
        """Show failure history."""
        limit = 10
        if args:
            try:
                parts = args.split()
                if len(parts) >= 2 and parts[0] == "limit":
                    limit = int(parts[1])
            except ValueError:
                pass

        recent = self._failure_history[-limit:]
        if not recent:
            return "No failure history"

        lines = [f"Recent failures (last {len(recent)}):"]
        for record in reversed(recent):
            time_str = record.timestamp.strftime("%H:%M:%S")
            lines.append(
                f"  [{time_str}] {record.failure_key} "
                f"({record.severity.value}): {record.error_message[:50]}"
            )

        return "\n".join(lines)

    def _cmd_config(self) -> str:
        """Show current configuration."""
        rule = self._config.default_rule
        return (
            f"Reliability Configuration:\n"
            f"  Count threshold: {rule.count_threshold}\n"
            f"  Window: {rule.window_seconds}s\n"
            f"  Escalation duration: {rule.escalation_duration_seconds}s\n"
            f"  Recovery successes needed: {rule.success_count_to_recover}\n"
            f"  Auto recovery: {self._config.enable_auto_recovery}\n"
            f"  Max history: {self._config.max_history_entries}"
        )

    def _cmd_settings(self, args: str) -> str:
        """Show or manage settings."""
        parts = args.strip().lower().split() if args else []

        if not parts or parts[0] == "show":
            return self._show_settings()
        elif parts[0] == "save":
            if len(parts) < 2:
                return "Usage: reliability settings save workspace|user"
            return self._save_all_settings(parts[1])
        elif parts[0] == "clear":
            if len(parts) < 2:
                return "Usage: reliability settings clear workspace|session"
            return self._clear_settings(parts[1])
        else:
            return "Usage: reliability settings [show|save workspace|user|clear workspace|session]"

    def _show_settings(self) -> str:
        """Show effective settings with source info."""
        lines = [
            "Reliability Settings (effective):",
            "-" * 60,
            f"  {'Setting':<25} {'Value':<15} {'Source':<20}",
            "-" * 60,
        ]

        settings = [
            ("recovery_mode", "auto" if self._config.enable_auto_recovery else "ask"),
            ("nudge_level", self._session_settings.nudge_level or "default"),
            ("nudge_enabled", str(self._session_settings.nudge_enabled)),
        ]

        for key, value in settings:
            source = self._get_setting_source(key)
            lines.append(f"  {key:<25} {value:<15} {source:<20}")

        lines.append("-" * 60)
        return "\n".join(lines)

    def _get_setting_source(self, key: str) -> str:
        """Determine where a setting value comes from."""
        # Check session
        session_val = getattr(self._session_settings, key, None)
        if session_val is not None:
            return "session (override)"

        # Check workspace
        if self._persistence:
            workspace = self._persistence.load_workspace()
            if key in workspace.settings:
                return "workspace"

            # Check user
            user = self._persistence.load_user()
            if key in user.default_settings:
                return "user (default)"

        return "built-in default"

    def _save_all_settings(self, level: str) -> str:
        """Save all current settings to workspace or user level."""
        if not self._persistence:
            return "Error: Persistence not initialized (no workspace set)"

        settings_to_save = {
            "recovery_mode": "auto" if self._config.enable_auto_recovery else "ask",
        }

        if self._session_settings.nudge_level:
            settings_to_save["nudge_level"] = self._session_settings.nudge_level

        if level == "workspace":
            for key, value in settings_to_save.items():
                self._persistence.save_setting_to_workspace(key, value)
            return f"Settings saved to workspace:\n  " + "\n  ".join(
                f"{k}: {v}" for k, v in settings_to_save.items()
            )
        elif level == "user":
            for key, value in settings_to_save.items():
                self._persistence.save_setting_to_user(key, value)
            return f"Settings saved as user defaults:\n  " + "\n  ".join(
                f"{k}: {v}" for k, v in settings_to_save.items()
            )
        else:
            return f"Unknown level: {level}. Use 'workspace' or 'user'"

    def _clear_settings(self, level: str) -> str:
        """Clear settings at a specific level."""
        if level == "session":
            self._session_settings = SessionSettings()
            return "Session overrides cleared. Now inheriting from workspace/user."
        elif level == "workspace":
            if not self._persistence:
                return "Error: Persistence not initialized"
            workspace = self._persistence.load_workspace()
            workspace.settings.clear()
            self._persistence.save_workspace()
            return "Workspace settings cleared. Now inheriting from user defaults."
        else:
            return f"Unknown level: {level}. Use 'workspace' or 'session'"

    def _cmd_model(self, args: str) -> str:
        """Handle model reliability commands."""
        parts = args.strip().lower().split() if args else []

        if not parts or parts[0] == "status":
            return self._model_status(parts[1] if len(parts) > 1 else None)
        elif parts[0] == "compare":
            return self._model_compare()
        elif parts[0] == "suggest":
            self._model_switch_config.strategy = ModelSwitchStrategy.SUGGEST
            self._session_settings.model_switch_strategy = "suggest"
            return "Model switching set to SUGGEST: Will suggest better models on failure"
        elif parts[0] == "auto":
            self._model_switch_config.strategy = ModelSwitchStrategy.AUTO
            self._session_settings.model_switch_strategy = "auto"
            return "Model switching set to AUTO: Will automatically switch to better models"
        elif parts[0] == "disabled":
            self._model_switch_config.strategy = ModelSwitchStrategy.DISABLED
            self._session_settings.model_switch_strategy = "disabled"
            return "Model switching DISABLED"
        else:
            return (
                f"Unknown model subcommand: {parts[0]}\n"
                "Usage: reliability model [status|compare|suggest|auto|disabled]"
            )

    def _model_status(self, model_name: Optional[str] = None) -> str:
        """Show model reliability status."""
        target = model_name or self._current_model

        if not target:
            return "No model context set. Use set_model_context() to track model reliability."

        summary = self.get_model_reliability_summary(target)

        lines = [
            f"Model Reliability: {summary['model']}",
            "-" * 60,
            f"  Total attempts: {summary['total_attempts']}",
            f"  Total failures: {summary['total_failures']}",
            f"  Success rate: {summary['success_rate']:.1%}",
            f"  Tools tracked: {summary['tools_tracked']}",
        ]

        if summary.get('problematic_tools'):
            lines.append("")
            lines.append("  Problematic tools (< 70% success):")
            for tool in summary['problematic_tools'][:5]:
                profile = self._model_profiles.get((target, tool))
                if profile:
                    lines.append(f"    - {tool}: {profile.success_rate:.0%} ({profile.total_attempts} attempts)")

        lines.append("-" * 60)
        lines.append(f"Strategy: {self._model_switch_config.strategy.value}")

        return "\n".join(lines)

    def _model_compare(self) -> str:
        """Compare reliability across models."""
        if not self._model_profiles:
            return "No model reliability data collected yet."

        # Group by model
        model_stats: Dict[str, Dict[str, Any]] = {}
        for (model, _), profile in self._model_profiles.items():
            if model not in model_stats:
                model_stats[model] = {
                    "attempts": 0,
                    "failures": 0,
                    "tools": 0,
                }
            model_stats[model]["attempts"] += profile.total_attempts
            model_stats[model]["failures"] += profile.failures
            model_stats[model]["tools"] += 1

        lines = [
            "Model Reliability Comparison",
            "-" * 60,
            f"  {'Model':<30} {'Success':<10} {'Attempts':<10} {'Tools':<6}",
            "-" * 60,
        ]

        for model, stats in sorted(model_stats.items()):
            success_rate = 1.0 - (stats["failures"] / stats["attempts"]) if stats["attempts"] > 0 else 1.0
            current = " (current)" if model == self._current_model else ""
            lines.append(
                f"  {model:<30} {success_rate:>7.0%}   {stats['attempts']:<10} {stats['tools']:<6}{current}"
            )

        lines.append("-" * 60)
        return "\n".join(lines)


def create_plugin(config: Optional[Dict[str, Any]] = None) -> ReliabilityPlugin:
    """Factory function to create the plugin."""
    reliability_config = None
    if config:
        # Could parse config dict into ReliabilityConfig here
        pass
    return ReliabilityPlugin(reliability_config)


# -----------------------------------------------------------------------------
# Permission Integration
# -----------------------------------------------------------------------------


class ReliabilityPermissionWrapper:
    """Wraps permission plugin to inject reliability-based escalation.

    This wrapper intercepts permission checks and forces approval prompts
    for tools that have been escalated due to reliability concerns, even
    if they would normally be auto-approved via whitelist.

    Usage:
        permission_plugin = PermissionPlugin()
        reliability_plugin = ReliabilityPlugin()

        # Wrap the permission plugin
        wrapped = ReliabilityPermissionWrapper(permission_plugin, reliability_plugin)

        # Use wrapped plugin for permission checks
        executor.set_permission_plugin(wrapped)
    """

    def __init__(self, inner, reliability: ReliabilityPlugin):
        """Initialize wrapper.

        Args:
            inner: The permission plugin to wrap (PermissionPlugin instance)
            reliability: The reliability plugin for escalation checks
        """
        self._inner = inner
        self._reliability = reliability

    def __getattr__(self, name):
        """Delegate attribute access to inner plugin."""
        return getattr(self._inner, name)

    def check_permission(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check permission with reliability overlay.

        If the tool is escalated due to reliability concerns, this forces
        a permission prompt even if the tool would normally be whitelisted.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            context: Optional context (session_id, turn_number, etc.)
            call_id: Optional call identifier for parallel tool matching

        Returns:
            Tuple of (is_allowed, metadata_dict)
        """
        # Check if tool is escalated due to reliability
        escalation = self._reliability.check_escalation(tool_name, args)

        if escalation.is_escalated:
            # Force approval even if whitelisted
            return self._force_approval_check(
                tool_name, args, context, call_id, escalation
            )

        # Otherwise: delegate to inner permission plugin
        return self._inner.check_permission(tool_name, args, context, call_id)

    def _force_approval_check(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        call_id: Optional[str],
        escalation: EscalationInfo,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Override whitelist to require explicit approval.

        Modifies the context to include escalation information, then
        temporarily forces the permission policy to "ask" for this tool.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            context: Optional context dict
            call_id: Optional call identifier
            escalation: Escalation info from reliability check

        Returns:
            Tuple of (is_allowed, metadata_dict)
        """
        # Build enhanced context with escalation info
        enhanced_context = dict(context) if context else {}
        enhanced_context["_reliability_escalation"] = {
            "is_escalated": True,
            "state": escalation.state.value,
            "reason": escalation.reason,
            "failure_count": escalation.failure_count,
            "window_description": escalation.window_description,
            "recovery_progress": escalation.recovery_progress,
            "recovery_hint": escalation.recovery_hint,
            "severity_label": escalation.severity_label,
            "display_lines": escalation.to_display_lines(),
        }

        # Check if inner plugin has policy to manipulate
        inner_policy = getattr(self._inner, '_policy', None)

        if inner_policy is not None:
            # Temporarily remove tool from whitelist to force prompt
            was_in_whitelist = tool_name in inner_policy.whitelist_tools
            was_in_session_whitelist = tool_name in inner_policy.session_whitelist

            if was_in_whitelist:
                inner_policy.whitelist_tools.discard(tool_name)
            if was_in_session_whitelist:
                inner_policy.session_whitelist.discard(tool_name)

            try:
                # Call inner check with modified policy
                allowed, info = self._inner.check_permission(
                    tool_name, args, enhanced_context, call_id
                )

                # Add reliability metadata to result
                info["_reliability"] = {
                    "escalated": True,
                    "state": escalation.state.value,
                    "reason": escalation.reason,
                }

                return allowed, info
            finally:
                # Restore whitelist state
                if was_in_whitelist:
                    inner_policy.whitelist_tools.add(tool_name)
                if was_in_session_whitelist:
                    inner_policy.session_whitelist.add(tool_name)
        else:
            # No policy access, just delegate with enhanced context
            allowed, info = self._inner.check_permission(
                tool_name, args, enhanced_context, call_id
            )
            info["_reliability"] = {
                "escalated": True,
                "state": escalation.state.value,
                "reason": escalation.reason,
            }
            return allowed, info

    def get_formatted_prompt(
        self,
        tool_name: str,
        args: Dict[str, Any],
        channel_type: str = "ipc",
    ) -> Tuple[List[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Get formatted prompt with reliability info if escalated.

        Enhances the inner plugin's formatted prompt with reliability
        escalation information when applicable.

        Returns:
            Tuple of (prompt_lines, format_hint, language, raw_details, warnings, warning_level)
        """
        # Get escalation info
        escalation = self._reliability.check_escalation(tool_name, args)

        # Get base prompt from inner plugin
        result = self._inner.get_formatted_prompt(tool_name, args, channel_type)
        prompt_lines, format_hint, language, raw_details, warnings, warning_level = result

        if escalation.is_escalated:
            # Prepend reliability warning to prompt
            reliability_lines = escalation.to_display_lines()
            if reliability_lines:
                # Add separator
                reliability_lines.append("")
                # Combine with original prompt
                prompt_lines = reliability_lines + prompt_lines

            # Upgrade warning level if needed
            if escalation.state == TrustState.BLOCKED:
                warning_level = "error"
            elif escalation.state == TrustState.ESCALATED and warning_level != "error":
                warning_level = "warning"

        return prompt_lines, format_hint, language, raw_details, warnings, warning_level


def wrap_permission_plugin(permission_plugin, reliability_plugin: ReliabilityPlugin):
    """Convenience function to wrap a permission plugin with reliability checks.

    Args:
        permission_plugin: The PermissionPlugin instance to wrap
        reliability_plugin: The ReliabilityPlugin instance for escalation checks

    Returns:
        ReliabilityPermissionWrapper that can be used in place of the permission plugin
    """
    return ReliabilityPermissionWrapper(permission_plugin, reliability_plugin)
