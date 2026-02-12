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
    BehavioralPattern,
    BehavioralPatternType,
    EscalationInfo,
    EscalationRule,
    FailureKey,
    FailureRecord,
    FailureSeverity,
    ModelBehavioralProfile,
    ModelSwitchConfig,
    ModelSwitchStrategy,
    ModelSwitchSuggestion,
    ModelToolProfile,
    Nudge,
    NudgeConfig,
    NudgeLevel,
    NudgeType,
    PatternDetectionConfig,
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
from .patterns import PatternDetector
from .nudge import NudgeInjector, NudgeStrategy

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

        # Behavioral pattern detection
        self._pattern_config = PatternDetectionConfig()
        self._pattern_detector: Optional[PatternDetector] = None
        # on_pattern_detected: (pattern) -> None
        self._on_pattern_detected: Optional[Callable[[BehavioralPattern], None]] = None

        # Nudge injection
        self._nudge_config = NudgeConfig()
        self._nudge_injector: Optional[NudgeInjector] = None
        # on_nudge_injected: (nudge) -> None
        self._on_nudge_injected: Optional[Callable[[Nudge], None]] = None

        # Model behavioral profiles
        self._behavioral_profiles: Dict[str, ModelBehavioralProfile] = {}  # model_name -> profile

        # Telemetry integration
        self._telemetry = None  # Optional TelemetryPlugin

        # Per-session tool success tracking (for GC policy decisions)
        # Key: (session_id, tool_name) -> success_count
        self._session_tool_successes: Dict[Tuple[str, str], int] = {}

    @property
    def name(self) -> str:
        return "reliability"

    def set_telemetry(self, telemetry) -> None:
        """Set telemetry plugin for OpenTelemetry event emission.

        Args:
            telemetry: A TelemetryPlugin instance for emitting events.
        """
        self._telemetry = telemetry

    # -------------------------------------------------------------------------
    # Plugin Protocol Implementation
    # -------------------------------------------------------------------------

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for model-callable tools."""
        return [
            ToolSchema(
                name="reliability_status",
                description="Check reliability status including tool trust, behavioral patterns, and model performance. Use to understand failure history, detect stall patterns, and compare model behavior.",
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
                        },
                        "include_patterns": {
                            "type": "boolean",
                            "description": "Include detected behavioral patterns (stalls, loops)",
                            "default": False
                        },
                        "include_behavior": {
                            "type": "boolean",
                            "description": "Include model behavioral profile (stall rate, nudge effectiveness)",
                            "default": False
                        },
                        "model_name": {
                            "type": "string",
                            "description": "Specific model to show behavior for (uses current model if omitted)"
                        },
                        "compare_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of two model names to compare behavior between"
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
                CommandCompletion("patterns", "Behavioral pattern detection"),
                CommandCompletion("nudge", "Nudge injection settings"),
                CommandCompletion("behavior", "Model behavioral profiles"),
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
            elif subcommand == "patterns":
                return [
                    CommandCompletion("status", "Show pattern detection status"),
                    CommandCompletion("enable", "Enable pattern detection"),
                    CommandCompletion("disable", "Disable pattern detection"),
                    CommandCompletion("history", "Show detected patterns"),
                    CommandCompletion("clear", "Clear pattern history"),
                ]
            elif subcommand == "nudge":
                return [
                    CommandCompletion("status", "Show nudge injection status"),
                    CommandCompletion("off", "Disable nudge injection"),
                    CommandCompletion("gentle", "Only gentle reminders"),
                    CommandCompletion("direct", "Gentle + direct instructions"),
                    CommandCompletion("full", "All nudges including interrupts"),
                    CommandCompletion("history", "Show nudge history"),
                ]
            elif subcommand == "behavior":
                return [
                    CommandCompletion("status", "Show behavioral profile summary"),
                    CommandCompletion("compare", "Compare behavioral profiles"),
                    CommandCompletion("patterns", "Show pattern breakdown by model"),
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
            elif subcommand == "model":
                arg = command_parts[2]
                if arg == "status":
                    # Return list of tracked models
                    models = set(m for (m, _) in self._model_profiles.keys())
                    if self._current_model:
                        models.add(self._current_model)
                    return [
                        CommandCompletion(m, f"Show reliability for {m}")
                        for m in sorted(models)
                    ]
                elif arg in ("suggest", "auto", "disabled"):
                    return [
                        CommandCompletion("save", "Save strategy setting"),
                    ]
            elif subcommand == "behavior":
                arg = command_parts[2]
                if arg in ("status", "compare", "patterns"):
                    # Return list of tracked behavioral profiles
                    models = set(self._behavioral_profiles.keys())
                    if self._current_model:
                        models.add(self._current_model)
                    return [
                        CommandCompletion(m, f"Show behavioral profile for {m}")
                        for m in sorted(models)
                    ]

        if len(command_parts) == 4:
            subcommand = command_parts[1]
            if subcommand == "model" and command_parts[2] in ("suggest", "auto", "disabled") and command_parts[3] == "save":
                return [
                    CommandCompletion("workspace", "Save to workspace"),
                    CommandCompletion("user", "Save as user default"),
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
        logger.info("Reliability plugin shutdown")

    # -------------------------------------------------------------------------
    # Session Persistence
    # -------------------------------------------------------------------------

    def get_persistence_state(self) -> Dict[str, Any]:
        """Return session-level state for persistence.

        Captures tool reliability states, turn tracking, and escalation
        overrides so they survive session unload/reload.
        """
        if not self._tool_states and not self._turn_index:
            return {}

        state: Dict[str, Any] = {"version": 1}

        if self._tool_states:
            state["tool_states"] = {
                key: ts.to_dict()
                for key, ts in self._tool_states.items()
            }

        state["turn_index"] = self._turn_index
        state["session_id"] = self._session_id

        if self._session_settings and self._session_settings != SessionSettings():
            state["session_settings"] = self._session_settings.to_dict()

        return state

    def restore_persistence_state(self, state: Dict[str, Any]) -> None:
        """Restore session-level state from persistence.

        Rebuilds tool reliability states and turn tracking from the
        previously-saved session state.
        """
        if state.get("tool_states"):
            self._tool_states = {
                key: ToolReliabilityState.from_dict(ts_data)
                for key, ts_data in state["tool_states"].items()
            }

        self._turn_index = state.get("turn_index", 0)
        self._session_id = state.get("session_id", self._session_id)

        if state.get("session_settings"):
            self._session_settings = SessionSettings.from_dict(
                state["session_settings"]
            )

        logger.debug(
            f"Restored reliability state: {len(self._tool_states)} tool states, "
            f"turn_index={self._turn_index}"
        )

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
    # Behavioral Pattern Detection
    # -------------------------------------------------------------------------

    def set_pattern_detection_config(self, config: PatternDetectionConfig) -> None:
        """Set pattern detection configuration."""
        self._pattern_config = config
        if self._pattern_detector:
            self._pattern_detector._config = config

    def enable_pattern_detection(self, enabled: bool = True) -> None:
        """Enable or disable behavioral pattern detection.

        When enabled, the plugin will track tool calls and model outputs
        to detect stall patterns like repetitive calls and introspection loops.

        Args:
            enabled: Whether to enable pattern detection
        """
        if enabled and not self._pattern_detector:
            self._pattern_detector = PatternDetector(
                config=self._pattern_config,
                session_id=self._session_id,
                model_name=self._current_model,
            )
            # Set up combined pattern hook for user callback + nudge injection
            self._pattern_detector.set_pattern_hook(self._handle_pattern_detected)
        elif not enabled:
            self._pattern_detector = None

    def _handle_pattern_detected(self, pattern: BehavioralPattern) -> None:
        """Internal handler for detected patterns. Triggers nudges and user callback."""
        # Record to behavioral profile
        self._record_pattern_to_profile(pattern)

        # Emit telemetry event for pattern detection
        self._emit_pattern_event(pattern)

        # Inject nudge if injector is enabled
        if self._nudge_injector:
            nudge = self._nudge_injector.on_pattern_detected(pattern)
            # Record nudge to behavioral profile if it was actually sent
            if nudge:
                self._record_nudge_to_profile(nudge)
                # Emit telemetry event for nudge injection
                self._emit_nudge_event(nudge)
                # Call nudge hook
                if self._on_nudge_injected:
                    self._on_nudge_injected(nudge)

        # Call user's pattern hook if set
        if self._on_pattern_detected:
            self._on_pattern_detected(pattern)

    def _emit_pattern_event(self, pattern: BehavioralPattern) -> None:
        """Emit OpenTelemetry event for pattern detection."""
        if not self._telemetry or not getattr(self._telemetry, 'enabled', False):
            return

        try:
            # Get current span context if available
            span_id = self._telemetry.get_current_span_id()
            if not span_id:
                return  # No active span

            # We add the event to the current span
            # This requires getting the current span which varies by telemetry impl
            # For now, log the pattern data that can be scraped by metrics
            logger.info(
                "reliability.pattern_detected",
                extra={
                    "pattern_type": pattern.pattern_type.value,
                    "severity": pattern.severity.value,
                    "turn_index": pattern.turn_index,
                    "model_name": pattern.model_name,
                    "repetition_count": pattern.repetition_count,
                    "session_id": pattern.session_id,
                }
            )
        except Exception as e:
            logger.debug(f"Failed to emit pattern telemetry event: {e}")

    def _emit_nudge_event(self, nudge: Nudge) -> None:
        """Emit OpenTelemetry event for nudge injection."""
        if not self._telemetry or not getattr(self._telemetry, 'enabled', False):
            return

        try:
            # Get context from the triggering pattern
            pattern = nudge.pattern
            logger.info(
                "reliability.nudge_injected",
                extra={
                    "nudge_type": nudge.nudge_type.value,
                    "turn_index": pattern.turn_index if pattern else self._turn_index,
                    "model_name": pattern.model_name if pattern else self._current_model,
                    "pattern_type": pattern.pattern_type.value if pattern else None,
                    "session_id": pattern.session_id if pattern else self._session_id,
                }
            )
        except Exception as e:
            logger.debug(f"Failed to emit nudge telemetry event: {e}")

    def _emit_trust_state_event(
        self,
        failure_key: str,
        tool_name: str,
        old_state: str,
        new_state: str,
        reason: Optional[str] = None,
    ) -> None:
        """Emit OpenTelemetry event for trust state change."""
        if not self._telemetry or not getattr(self._telemetry, 'enabled', False):
            return

        try:
            logger.info(
                "reliability.trust_state_changed",
                extra={
                    "failure_key": failure_key,
                    "tool_name": tool_name,
                    "old_state": old_state,
                    "new_state": new_state,
                    "reason": reason,
                    "session_id": self._session_id,
                    "model_name": self._current_model,
                }
            )
        except Exception as e:
            logger.debug(f"Failed to emit trust state telemetry event: {e}")

    def _emit_failure_event(
        self,
        failure_key: str,
        tool_name: str,
        severity: str,
        error_type: str,
    ) -> None:
        """Emit OpenTelemetry event for tool failure."""
        if not self._telemetry or not getattr(self._telemetry, 'enabled', False):
            return

        try:
            logger.info(
                "reliability.tool_failure",
                extra={
                    "failure_key": failure_key,
                    "tool_name": tool_name,
                    "severity": severity,
                    "error_type": error_type,
                    "session_id": self._session_id,
                    "model_name": self._current_model,
                }
            )
        except Exception as e:
            logger.debug(f"Failed to emit failure telemetry event: {e}")

    def set_pattern_hook(
        self,
        on_pattern_detected: Optional[Callable[[BehavioralPattern], None]] = None,
    ) -> None:
        """Set hook for pattern detection events.

        Args:
            on_pattern_detected: Called when a behavioral pattern is detected.
                Signature: (pattern: BehavioralPattern) -> None
        """
        self._on_pattern_detected = on_pattern_detected
        if self._pattern_detector:
            self._pattern_detector.set_pattern_hook(on_pattern_detected)

    def on_turn_start(self, turn_index: int) -> None:
        """Called when a new turn begins. Resets pattern detection state.

        Args:
            turn_index: The index of the new turn
        """
        self._turn_index = turn_index
        if self._pattern_detector:
            self._pattern_detector.on_turn_start(turn_index)

        # Record turn start to behavioral profile
        if self._current_model:
            profile = self.get_behavioral_profile(self._current_model)
            profile.record_turn_start()

    def on_turn_end(self) -> Optional[BehavioralPattern]:
        """Called when a turn ends. Checks for announce-no-action patterns.

        Returns:
            BehavioralPattern if detected, None otherwise
        """
        if self._pattern_detector:
            return self._pattern_detector.on_turn_end()
        return None

    def on_tool_called(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Optional[BehavioralPattern]:
        """Hook called BEFORE tool execution for pattern detection.

        This should be called before on_tool_result to enable pattern
        detection for repetitive calls, introspection loops, etc.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool

        Returns:
            BehavioralPattern if a stall pattern is detected, None otherwise
        """
        if self._pattern_detector:
            return self._pattern_detector.on_tool_called(tool_name, args)
        return None

    def on_model_text(self, text: str) -> None:
        """Hook for model text output to track announcements.

        This enables detection of "announce-no-action" patterns where
        the model says "I'll do X" but doesn't follow through.

        Args:
            text: The text output by the model
        """
        if self._pattern_detector:
            self._pattern_detector.on_model_text(text)

    def get_pattern_detector(self) -> Optional[PatternDetector]:
        """Get the pattern detector instance (if enabled)."""
        return self._pattern_detector

    def get_detected_patterns(self) -> List[BehavioralPattern]:
        """Get all detected behavioral patterns."""
        if self._pattern_detector:
            return self._pattern_detector.get_detected_patterns()
        return []

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns."""
        if self._pattern_detector:
            return self._pattern_detector.get_pattern_summary()
        return {"total": 0, "by_type": {}, "by_severity": {}, "enabled": False}

    # -------------------------------------------------------------------------
    # Nudge Injection
    # -------------------------------------------------------------------------

    def set_nudge_config(self, config: NudgeConfig) -> None:
        """Set nudge injection configuration."""
        self._nudge_config = config
        if self._nudge_injector:
            self._nudge_injector.set_config(config)

    def set_nudge_level(self, level: NudgeLevel) -> None:
        """Set the nudge intensity level.

        Args:
            level: OFF, GENTLE, DIRECT, or FULL
        """
        self._nudge_config.level = level
        if self._nudge_injector:
            self._nudge_injector.set_level(level)

    def enable_nudge_injection(self, enabled: bool = True) -> None:
        """Enable or disable nudge injection.

        When enabled with pattern detection, nudges are automatically
        injected when stall patterns are detected.

        Args:
            enabled: Whether to enable nudge injection
        """
        if enabled and not self._nudge_injector:
            self._nudge_injector = NudgeInjector(
                config=self._nudge_config,
            )
            if self._on_nudge_injected:
                self._nudge_injector.set_nudge_hooks(on_nudge_injected=self._on_nudge_injected)
        elif not enabled:
            self._nudge_injector = None

    def set_nudge_hook(
        self,
        on_nudge_injected: Optional[Callable[[Nudge], None]] = None,
    ) -> None:
        """Set hook for nudge injection events.

        Args:
            on_nudge_injected: Called when a nudge is injected.
                Signature: (nudge: Nudge) -> None
        """
        self._on_nudge_injected = on_nudge_injected
        if self._nudge_injector:
            self._nudge_injector.set_nudge_hooks(on_nudge_injected=on_nudge_injected)

    def set_nudge_callbacks(
        self,
        inject_system_guidance: Optional[Callable[[str], None]] = None,
        inject_context_hint: Optional[Callable[[str], None]] = None,
        request_pause: Optional[Callable[[str], None]] = None,
        notify_user: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        """Set callbacks for injecting nudges into the session.

        These callbacks allow the plugin to inject messages into the
        model's context when patterns are detected, and optionally emit
        user-visible notifications.

        Args:
            inject_system_guidance: Inject as high-priority system message
            inject_context_hint: Inject as lower-priority context hint
            request_pause: Request user intervention (highest priority)
            notify_user: Emit a user-visible notification via the output
                callback (source, text, mode). Uses source="enrichment"
                to match the rendering pipeline used by template/memory/
                reference enrichment notifications.
        """
        if self._nudge_injector:
            self._nudge_injector.set_injection_callbacks(
                inject_system_guidance=inject_system_guidance,
                inject_context_hint=inject_context_hint,
                request_pause=request_pause,
                notify_user=notify_user,
            )

    def inject_nudge_for_pattern(self, pattern: BehavioralPattern) -> Optional[Nudge]:
        """Inject a nudge for a detected pattern.

        This is called automatically when pattern detection is enabled
        and a pattern is detected. Can also be called manually.

        Args:
            pattern: The detected behavioral pattern

        Returns:
            The injected Nudge if one was created, None otherwise
        """
        if self._nudge_injector:
            return self._nudge_injector.on_pattern_detected(pattern)
        return None

    def get_nudge_injector(self) -> Optional[NudgeInjector]:
        """Get the nudge injector instance (if enabled)."""
        return self._nudge_injector

    def get_nudge_summary(self) -> Dict[str, Any]:
        """Get summary of nudge injection activity."""
        if self._nudge_injector:
            return self._nudge_injector.get_nudge_summary()
        return {"total": 0, "effective": 0, "by_type": {}, "enabled": False}

    def mark_last_nudge_effective(self, effective: bool = True) -> None:
        """Mark the most recent nudge as effective (pattern stopped).

        Call this when a pattern stops after a nudge was sent.
        Updates both the nudge injector and behavioral profile.

        Args:
            effective: Whether the nudge was effective
        """
        if self._nudge_injector:
            # Get the nudge before marking (to access pattern info)
            recent_nudges = self._nudge_injector.get_recent_nudges(1)
            if recent_nudges and effective:
                nudge = recent_nudges[-1]
                self._record_nudge_effective_to_profile(nudge)

            self._nudge_injector.mark_last_nudge_effective(effective)

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

        # Notify pattern detector of tool result
        if self._pattern_detector:
            self._pattern_detector.on_tool_result(tool_name, success, result)

        # Get or create state
        state = self._get_or_create_state(key_str, tool_name)

        if is_failure:
            return self._handle_failure(state, failure_key, result, call_id, plugin_name)
        else:
            # Track per-session success for GC policy evaluation
            if self._session_id:
                session_tool_key = (self._session_id, tool_name)
                self._session_tool_successes[session_tool_key] = \
                    self._session_tool_successes.get(session_tool_key, 0) + 1

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

        # Emit telemetry event for failure
        self._emit_failure_event(
            failure_key.to_string(),
            failure_key.tool_name,
            severity.value,
            record.error_type,
        )

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
            old_state = state.state.value
            state.state = TrustState.BLOCKED
            reason = "Security concern detected"

            # Emit blocked hook and telemetry
            if not was_blocked:
                if self._on_blocked:
                    self._on_blocked(failure_key.to_string(), reason)
                self._emit_trust_state_event(
                    failure_key.to_string(),
                    state.tool_name,
                    old_state,
                    state.state.value,
                    reason,
                )

        # Apply escalation if needed
        if should_escalate and state.state == TrustState.TRUSTED:
            old_state = state.state.value
            state.state = TrustState.ESCALATED
            state.escalated_at = now
            state.escalation_reason = reason
            state.escalation_expires = now + timedelta(seconds=rule.escalation_duration_seconds)
            state.successes_needed = rule.success_count_to_recover

            if rule.notify_user:
                self._notify_escalation(state)

            # Emit escalated hook and telemetry
            if self._on_escalated:
                self._on_escalated(failure_key.to_string(), state.state, reason)
            self._emit_trust_state_event(
                failure_key.to_string(),
                state.tool_name,
                old_state,
                state.state.value,
                reason,
            )

        # If recovering, reset recovery progress
        elif state.state == TrustState.RECOVERING:
            old_state = state.state.value
            state.state = TrustState.ESCALATED
            state.recovery_started = None
            state.successes_since_recovery = 0
            re_escalation_reason = "Recovery interrupted by failure"

            # Emit escalated hook and telemetry (re-escalation from recovering)
            if self._on_escalated:
                self._on_escalated(
                    failure_key.to_string(),
                    state.state,
                    re_escalation_reason,
                )
            self._emit_trust_state_event(
                failure_key.to_string(),
                state.tool_name,
                old_state,
                state.state.value,
                re_escalation_reason,
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
                old_state = state.state.value
                state.state = TrustState.RECOVERING
                state.recovery_started = now
                state.successes_since_recovery = 1
                state.successes_needed = rule.success_count_to_recover
                logger.debug(f"Tool {state.failure_key} entering recovery")

                # Emit telemetry for entering recovery
                self._emit_trust_state_event(
                    state.failure_key,
                    state.tool_name,
                    old_state,
                    state.state.value,
                    "Cooldown expired, entering recovery",
                )
                return state

        elif state.state == TrustState.RECOVERING:
            state.successes_since_recovery += 1

            if state.successes_since_recovery >= state.successes_needed:
                # Full recovery
                old_state = state.state.value
                state.state = TrustState.TRUSTED
                state.escalated_at = None
                state.escalation_reason = None
                state.escalation_expires = None
                state.recovery_started = None
                state.successes_since_recovery = 0
                self._notify_recovery(state)
                logger.info(f"Tool {state.failure_key} recovered to TRUSTED")

                # Emit recovered hook and telemetry
                if self._on_recovered:
                    self._on_recovered(state.failure_key, state.state)
                self._emit_trust_state_event(
                    state.failure_key,
                    state.tool_name,
                    old_state,
                    state.state.value,
                    "Recovery complete",
                )

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

    def has_successful_execution(self, tool_name: str, session_id: str) -> bool:
        """Check if a tool has been successfully executed in a given session.

        Used by GC policy evaluation to determine if a discovered tool schema
        should be locked (if used) or remain ephemeral (if not yet used).

        Args:
            tool_name: Name of the tool to check
            session_id: Session ID to filter by

        Returns:
            True if the tool has at least one successful execution in this session
        """
        session_tool_key = (session_id, tool_name)
        return self._session_tool_successes.get(session_tool_key, 0) > 0

    def clear_session_data(self, session_id: str) -> None:
        """Clear per-session tracking data for a given session.

        Args:
            session_id: Session ID to clear data for
        """
        keys_to_remove = [k for k in self._session_tool_successes if k[0] == session_id]
        for key in keys_to_remove:
            del self._session_tool_successes[key]

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
        include_patterns = args.get("include_patterns", False)
        include_behavior = args.get("include_behavior", False)
        model_name = args.get("model_name")
        compare_models = args.get("compare_models")

        result: Dict[str, Any] = {}

        # Tool reliability status
        if tool_name:
            # Show specific tool
            matching = [
                state for key, state in self._tool_states.items()
                if tool_name in key or tool_name == state.tool_name
            ]
            if not matching:
                result["tools"] = {"message": f"No reliability data for tool '{tool_name}'"}
            else:
                result["tools"] = [self._format_state(s) for s in matching]
        else:
            # Show all non-trusted
            escalated = self.get_escalated_tools()
            if not escalated:
                result["tool_status"] = {
                    "message": "All tools are TRUSTED",
                    "total_tracked": len(self._tool_states),
                }
            else:
                result["tool_status"] = {
                    "escalated_tools": [self._format_state(s) for s in escalated],
                    "total_tracked": len(self._tool_states),
                    "total_escalated": len(escalated),
                }

        # Failure history
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

        # Behavioral patterns
        if include_patterns and self._pattern_detector:
            patterns = self._pattern_detector.get_recent_patterns(10)
            if patterns:
                result["recent_patterns"] = [
                    {
                        "type": p.pattern_type.value,
                        "severity": p.severity.value,
                        "tool_sequence": p.tool_sequence[-5:],  # Last 5 tools
                        "expected_action": p.expected_action,
                        "repetitions": p.repetition_count,
                        "timestamp": p.detected_at.isoformat(),
                    }
                    for p in patterns
                ]
            else:
                result["recent_patterns"] = {"message": "No patterns detected"}

            # Pattern statistics
            stats = self._pattern_detector.get_pattern_summary()
            if stats:
                result["pattern_stats"] = stats

        # Model behavioral profile
        if include_behavior:
            target_model = model_name or self._current_model
            if target_model:
                profile = self.get_behavioral_profile(target_model)
                result["behavioral_profile"] = {
                    "model": target_model,
                    "stall_rate": round(profile.stall_rate, 3),
                    "nudge_effectiveness": round(profile.nudge_effectiveness, 3),
                    "total_turns": profile.total_turns,
                    "stalled_turns": profile.stalled_turns,
                    "nudges_sent": profile.nudges_sent,
                    "nudges_effective": profile.nudges_effective,
                    "pattern_summary": {
                        pt.value: count
                        for pt, count in profile.pattern_counts.items()
                    } if profile.pattern_counts else {},
                }
            else:
                result["behavioral_profile"] = {"message": "No model set"}

        # Model comparison
        if compare_models and len(compare_models) >= 2:
            comparison = self.compare_behavioral_profiles(
                compare_models[0], compare_models[1]
            )
            if comparison:
                result["model_comparison"] = comparison
            else:
                result["model_comparison"] = {
                    "message": f"Cannot compare: need data for both {compare_models[0]} and {compare_models[1]}"
                }

        # Summary metrics for dashboard
        result["summary"] = self._get_observability_summary()

        return result

    def _get_observability_summary(self) -> Dict[str, Any]:
        """Get summary metrics for observability dashboard."""
        summary: Dict[str, Any] = {
            "tools": {
                "total_tracked": len(self._tool_states),
                "escalated": len(self.get_escalated_tools()),
                "total_failures": len(self._failure_history),
            }
        }

        # Pattern metrics
        if self._pattern_detector:
            patterns = self._pattern_detector.get_recent_patterns(100)
            if patterns:
                by_type: Dict[str, int] = {}
                by_severity: Dict[str, int] = {}
                for p in patterns:
                    pt = p.pattern_type.value
                    by_type[pt] = by_type.get(pt, 0) + 1
                    sev = p.severity.value
                    by_severity[sev] = by_severity.get(sev, 0) + 1
                summary["patterns"] = {
                    "total_detected": len(patterns),
                    "by_type": by_type,
                    "by_severity": by_severity,
                }

        # Nudge metrics
        if self._nudge_injector:
            nudge_history = self._nudge_injector.get_nudge_history()
            if nudge_history:
                summary["nudges"] = {
                    "total_sent": len(nudge_history),
                }

        # Model metrics
        if self._behavioral_profiles:
            model_summaries = []
            for model, profile in self._behavioral_profiles.items():
                model_summaries.append({
                    "model": model,
                    "stall_rate": round(profile.stall_rate, 3),
                    "nudge_effectiveness": round(profile.nudge_effectiveness, 3),
                    "turns": profile.total_turns,
                })
            summary["models"] = model_summaries

        return summary

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
        elif subcommand == "patterns":
            return self._cmd_patterns(args)
        elif subcommand == "nudge":
            return self._cmd_nudge(args)
        elif subcommand == "behavior":
            return self._cmd_behavior(args)
        else:
            return f"Unknown subcommand: {subcommand}\nUse: status, recovery, reset, history, config, settings, model, patterns, nudge, behavior"

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
            result = "Model switching set to SUGGEST: Will suggest better models on failure"
            if len(parts) >= 3 and parts[1] == "save":
                save_result = self._save_model_strategy(parts[2])
                return f"{result}\n{save_result}"
            return result
        elif parts[0] == "auto":
            self._model_switch_config.strategy = ModelSwitchStrategy.AUTO
            self._session_settings.model_switch_strategy = "auto"
            result = "Model switching set to AUTO: Will automatically switch to better models"
            if len(parts) >= 3 and parts[1] == "save":
                save_result = self._save_model_strategy(parts[2])
                return f"{result}\n{save_result}"
            return result
        elif parts[0] == "disabled":
            self._model_switch_config.strategy = ModelSwitchStrategy.DISABLED
            self._session_settings.model_switch_strategy = "disabled"
            result = "Model switching DISABLED"
            if len(parts) >= 3 and parts[1] == "save":
                save_result = self._save_model_strategy(parts[2])
                return f"{result}\n{save_result}"
            return result
        else:
            return (
                f"Unknown model subcommand: {parts[0]}\n"
                "Usage: reliability model [status|compare|suggest|auto|disabled]"
            )

    def _save_model_strategy(self, level: str) -> str:
        """Save model switch strategy to workspace or user level."""
        if not self._persistence:
            return "Error: Persistence not initialized (no workspace set)"

        strategy = self._model_switch_config.strategy.value

        if level == "workspace":
            if self._persistence.save_setting_to_workspace("model_switch_strategy", strategy):
                return f"Model strategy '{strategy}' saved to workspace"
            return "Error: Failed to save to workspace"
        elif level == "user":
            if self._persistence.save_setting_to_user("model_switch_strategy", strategy):
                return f"Model strategy '{strategy}' saved as user default"
            return "Error: Failed to save to user"
        else:
            return f"Unknown level: {level}. Use 'workspace' or 'user'"

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

    def _cmd_patterns(self, args: str) -> str:
        """Handle pattern detection commands."""
        parts = args.strip().lower().split() if args else []

        if not parts or parts[0] == "status":
            return self._patterns_status()
        elif parts[0] == "enable":
            self.enable_pattern_detection(True)
            return "Behavioral pattern detection ENABLED"
        elif parts[0] == "disable":
            self.enable_pattern_detection(False)
            return "Behavioral pattern detection DISABLED"
        elif parts[0] == "history":
            return self._patterns_history(int(parts[1]) if len(parts) > 1 else 10)
        elif parts[0] == "clear":
            if self._pattern_detector:
                self._pattern_detector.clear_history()
            return "Pattern history cleared"
        else:
            return (
                f"Unknown patterns subcommand: {parts[0]}\n"
                "Usage: reliability patterns [status|enable|disable|history|clear]"
            )

    def _patterns_status(self) -> str:
        """Show pattern detection status."""
        enabled = self._pattern_detector is not None
        summary = self.get_pattern_summary()

        lines = [
            "Behavioral Pattern Detection",
            "-" * 60,
            f"  Status: {'ENABLED' if enabled else 'DISABLED'}",
        ]

        if enabled and self._pattern_detector:
            config = self._pattern_detector._config
            lines.extend([
                "",
                "  Configuration:",
                f"    Repetitive call threshold: {config.repetitive_call_threshold}",
                f"    Introspection loop threshold: {config.introspection_loop_threshold}",
                f"    Max reads before action: {config.max_reads_before_action}",
            ])

        lines.extend([
            "",
            f"  Total patterns detected: {summary.get('total', 0)}",
        ])

        if summary.get('by_type'):
            lines.append("  By type:")
            for ptype, count in sorted(summary['by_type'].items()):
                lines.append(f"    - {ptype}: {count}")

        if summary.get('by_severity'):
            lines.append("  By severity:")
            for sev, count in sorted(summary['by_severity'].items()):
                lines.append(f"    - {sev}: {count}")

        lines.append("-" * 60)
        return "\n".join(lines)

    def _patterns_history(self, limit: int = 10) -> str:
        """Show detected pattern history."""
        if not self._pattern_detector:
            return "Pattern detection not enabled. Use 'reliability patterns enable' first."

        patterns = self._pattern_detector.get_recent_patterns(limit)

        if not patterns:
            return "No patterns detected yet."

        lines = [f"Recent Patterns (last {len(patterns)}):", "-" * 60]

        for pattern in reversed(patterns):
            time_str = pattern.detected_at.strftime("%H:%M:%S")
            lines.append(f"  [{time_str}] {pattern.pattern_type.value}")
            lines.append(f"    Severity: {pattern.severity.value}")
            lines.append(f"    Repetitions: {pattern.repetition_count}")
            if pattern.tool_sequence:
                recent = pattern.tool_sequence[-3:]
                lines.append(f"    Tools: {' → '.join(recent)}")
            if pattern.expected_action:
                lines.append(f"    Suggestion: {pattern.expected_action}")
            lines.append("")

        return "\n".join(lines)

    def _cmd_nudge(self, args: str) -> str:
        """Handle nudge injection commands."""
        parts = args.strip().lower().split() if args else []

        if not parts or parts[0] == "status":
            return self._nudge_status()
        elif parts[0] == "off":
            self._nudge_config.level = NudgeLevel.OFF
            self._session_settings.nudge_level = "off"
            if self._nudge_injector:
                self._nudge_injector.set_level(NudgeLevel.OFF)
            return "Nudge injection set to OFF"
        elif parts[0] == "gentle":
            self._nudge_config.level = NudgeLevel.GENTLE
            self._session_settings.nudge_level = "gentle"
            self.enable_nudge_injection(True)
            self._nudge_injector.set_level(NudgeLevel.GENTLE)
            return "Nudge injection set to GENTLE (reminders only)"
        elif parts[0] == "direct":
            self._nudge_config.level = NudgeLevel.DIRECT
            self._session_settings.nudge_level = "direct"
            self.enable_nudge_injection(True)
            self._nudge_injector.set_level(NudgeLevel.DIRECT)
            return "Nudge injection set to DIRECT (reminders + direct instructions)"
        elif parts[0] == "full":
            self._nudge_config.level = NudgeLevel.FULL
            self._session_settings.nudge_level = "full"
            self.enable_nudge_injection(True)
            self._nudge_injector.set_level(NudgeLevel.FULL)
            return "Nudge injection set to FULL (all nudges including interrupts)"
        elif parts[0] == "history":
            return self._nudge_history(int(parts[1]) if len(parts) > 1 else 10)
        else:
            return (
                f"Unknown nudge subcommand: {parts[0]}\n"
                "Usage: reliability nudge [status|off|gentle|direct|full|history]"
            )

    def _nudge_status(self) -> str:
        """Show nudge injection status."""
        enabled = self._nudge_injector is not None
        summary = self.get_nudge_summary()

        lines = [
            "Nudge Injection Status",
            "-" * 60,
            f"  Status: {'ENABLED' if enabled else 'DISABLED'}",
            f"  Level: {self._nudge_config.level.value}",
            f"  Cooldown: {self._nudge_config.cooldown_seconds}s between nudges",
        ]

        lines.extend([
            "",
            f"  Total nudges: {summary.get('total', 0)}",
            f"  Effective: {summary.get('effective', 0)}",
            f"  Acknowledged: {summary.get('acknowledged', 0)}",
            f"  Pending: {summary.get('pending', 0)}",
        ])

        if summary.get('by_type'):
            lines.append("")
            lines.append("  By type:")
            for ntype, count in sorted(summary['by_type'].items()):
                lines.append(f"    - {ntype}: {count}")

        effectiveness = summary.get('effectiveness_rate', 0)
        if summary.get('total', 0) > 0:
            lines.append("")
            lines.append(f"  Effectiveness rate: {effectiveness:.1%}")

        lines.append("-" * 60)
        return "\n".join(lines)

    def _nudge_history(self, limit: int = 10) -> str:
        """Show nudge injection history."""
        if not self._nudge_injector:
            return "Nudge injection not enabled. Use 'reliability nudge gentle|direct|full' first."

        nudges = self._nudge_injector.get_recent_nudges(limit)

        if not nudges:
            return "No nudges injected yet."

        lines = [f"Recent Nudges (last {len(nudges)}):", "-" * 60]

        for nudge in reversed(nudges):
            time_str = nudge.injected_at.strftime("%H:%M:%S")
            status = ""
            if nudge.effective:
                status = " ✓"
            elif nudge.acknowledged:
                status = " ○"
            lines.append(f"  [{time_str}] {nudge.nudge_type.value}{status}")
            lines.append(f"    Pattern: {nudge.pattern.pattern_type.value}")
            lines.append(f"    Message: {nudge.message[:60]}...")
            lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Model Behavioral Profile Methods
    # -------------------------------------------------------------------------

    def get_behavioral_profile(self, model_name: str) -> ModelBehavioralProfile:
        """Get or create behavioral profile for a model."""
        if model_name not in self._behavioral_profiles:
            self._behavioral_profiles[model_name] = ModelBehavioralProfile(model_name=model_name)
        return self._behavioral_profiles[model_name]

    def get_all_behavioral_profiles(self) -> Dict[str, ModelBehavioralProfile]:
        """Get all behavioral profiles."""
        return self._behavioral_profiles.copy()

    def get_behavioral_summary(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get behavioral summary for a model or all models."""
        if model_name:
            if model_name not in self._behavioral_profiles:
                return {"error": f"No behavioral profile for model: {model_name}"}
            return self._behavioral_profiles[model_name].get_summary()

        # Aggregate summary for all models
        summaries = []
        for name, profile in self._behavioral_profiles.items():
            summaries.append(profile.get_summary())

        total_turns = sum(p.total_turns for p in self._behavioral_profiles.values())
        total_stalled = sum(p.stalled_turns for p in self._behavioral_profiles.values())
        total_patterns = sum(p.total_patterns for p in self._behavioral_profiles.values())
        total_nudges = sum(p.nudges_sent for p in self._behavioral_profiles.values())
        total_effective = sum(p.nudges_effective for p in self._behavioral_profiles.values())

        return {
            "models_tracked": len(self._behavioral_profiles),
            "total_turns": total_turns,
            "total_stalled_turns": total_stalled,
            "overall_stall_rate": total_stalled / total_turns if total_turns > 0 else 0.0,
            "total_patterns": total_patterns,
            "total_nudges": total_nudges,
            "total_nudges_effective": total_effective,
            "overall_nudge_effectiveness": total_effective / total_nudges if total_nudges > 0 else 1.0,
            "models": summaries,
        }

    def compare_behavioral_profiles(self, model1: str, model2: str) -> Optional[Dict[str, Any]]:
        """Compare behavioral profiles of two models."""
        if model1 not in self._behavioral_profiles:
            return {"error": f"No behavioral profile for model: {model1}"}
        if model2 not in self._behavioral_profiles:
            return {"error": f"No behavioral profile for model: {model2}"}

        profile1 = self._behavioral_profiles[model1]
        profile2 = self._behavioral_profiles[model2]
        return profile1.compare_to(profile2)

    def _record_pattern_to_profile(self, pattern: BehavioralPattern) -> None:
        """Record a detected pattern to the model's behavioral profile."""
        model = pattern.model_name or self._current_model
        if not model:
            return

        profile = self.get_behavioral_profile(model)
        profile.record_pattern(pattern)
        profile.record_turn_stalled()

    def _record_nudge_to_profile(self, nudge: Nudge) -> None:
        """Record a nudge sent to the model's behavioral profile."""
        model = nudge.pattern.model_name or self._current_model
        if not model:
            return

        profile = self.get_behavioral_profile(model)
        profile.record_nudge_sent()

    def _record_nudge_effective_to_profile(self, nudge: Nudge) -> None:
        """Record that a nudge was effective to the model's behavioral profile."""
        model = nudge.pattern.model_name or self._current_model
        if not model:
            return

        profile = self.get_behavioral_profile(model)
        profile.record_nudge_effective()

    def _cmd_behavior(self, args: str) -> str:
        """Handle behavioral profile commands."""
        parts = args.strip().lower().split() if args else []

        if not parts or parts[0] == "status":
            model = parts[1] if len(parts) > 1 else None
            return self._behavior_status(model)
        elif parts[0] == "compare":
            if len(parts) < 2:
                return "Usage: reliability behavior compare <model1> [model2]"
            model1 = parts[1]
            model2 = parts[2] if len(parts) > 2 else self._current_model
            if not model2:
                return "Cannot compare: specify a second model or set current model context"
            return self._behavior_compare(model1, model2)
        elif parts[0] == "patterns":
            model = parts[1] if len(parts) > 1 else None
            return self._behavior_patterns(model)
        else:
            return (
                f"Unknown behavior subcommand: {parts[0]}\n"
                "Usage: reliability behavior [status|compare|patterns] [model]"
            )

    def _behavior_status(self, model: Optional[str] = None) -> str:
        """Show behavioral profile status."""
        if not self._behavioral_profiles:
            return "No behavioral profiles tracked yet.\nEnable pattern detection with 'reliability patterns enable'."

        if model:
            if model not in self._behavioral_profiles:
                return f"No behavioral profile for model: {model}"

            profile = self._behavioral_profiles[model]
            lines = [
                f"Behavioral Profile: {model}",
                "-" * 60,
                f"  Total turns: {profile.total_turns}",
                f"  Stalled turns: {profile.stalled_turns} ({profile.stall_rate:.1%})",
                "",
                f"  Total patterns: {profile.total_patterns}",
            ]

            if profile.most_common_pattern():
                lines.append(f"  Most common: {profile.most_common_pattern().value}")

            lines.extend([
                "",
                f"  Nudges sent: {profile.nudges_sent}",
                f"  Nudges effective: {profile.nudges_effective} ({profile.nudge_effectiveness:.1%})",
            ])

            if profile.first_seen:
                lines.append(f"  First seen: {profile.first_seen.strftime('%Y-%m-%d %H:%M')}")
            if profile.last_seen:
                lines.append(f"  Last seen: {profile.last_seen.strftime('%Y-%m-%d %H:%M')}")

            lines.append("-" * 60)
            return "\n".join(lines)

        # Summary of all models
        summary = self.get_behavioral_summary()
        lines = [
            "Behavioral Profile Summary",
            "-" * 60,
            f"  Models tracked: {summary['models_tracked']}",
            f"  Total turns: {summary['total_turns']}",
            f"  Overall stall rate: {summary['overall_stall_rate']:.1%}",
            f"  Total patterns: {summary['total_patterns']}",
            f"  Overall nudge effectiveness: {summary['overall_nudge_effectiveness']:.1%}",
            "",
            "  Per-model summary:",
        ]

        for model_summary in summary.get('models', []):
            name = model_summary['model']
            stall = model_summary['stall_rate']
            nudge_eff = model_summary['nudge_effectiveness']
            lines.append(f"    {name}: stall={stall:.1%}, nudge_eff={nudge_eff:.1%}")

        lines.append("-" * 60)
        return "\n".join(lines)

    def _behavior_compare(self, model1: str, model2: str) -> str:
        """Compare two models' behavioral profiles."""
        result = self.compare_behavioral_profiles(model1, model2)

        if "error" in result:
            return result["error"]

        lines = [
            f"Behavioral Comparison: {model1} vs {model2}",
            "-" * 60,
        ]

        # Get both profiles for detailed comparison
        p1 = self._behavioral_profiles[model1]
        p2 = self._behavioral_profiles[model2]

        lines.extend([
            f"  Stall rate:",
            f"    {model1}: {p1.stall_rate:.1%} ({p1.stalled_turns}/{p1.total_turns} turns)",
            f"    {model2}: {p2.stall_rate:.1%} ({p2.stalled_turns}/{p2.total_turns} turns)",
            "",
            f"  Nudge effectiveness:",
            f"    {model1}: {p1.nudge_effectiveness:.1%} ({p1.nudges_effective}/{p1.nudges_sent} nudges)",
            f"    {model2}: {p2.nudge_effectiveness:.1%} ({p2.nudges_effective}/{p2.nudges_sent} nudges)",
            "",
            f"  Pattern count:",
            f"    {model1}: {p1.total_patterns}",
            f"    {model2}: {p2.total_patterns}",
            "",
            f"  Recommendation: {result['recommendation']}",
            "-" * 60,
        ])

        return "\n".join(lines)

    def _behavior_patterns(self, model: Optional[str] = None) -> str:
        """Show pattern breakdown by model."""
        if not self._behavioral_profiles:
            return "No behavioral profiles tracked yet."

        models = [model] if model else list(self._behavioral_profiles.keys())
        lines = ["Pattern Breakdown by Model", "-" * 60]

        for m in models:
            if m not in self._behavioral_profiles:
                continue

            profile = self._behavioral_profiles[m]
            lines.append(f"  {m}:")

            if not profile.pattern_counts:
                lines.append("    No patterns detected")
            else:
                for ptype, count in sorted(
                    profile.pattern_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    avg_sev = profile.average_severity(ptype)
                    sev_str = f" (avg severity: {avg_sev:.1f})" if avg_sev else ""
                    lines.append(f"    {ptype.value}: {count}{sev_str}")

            lines.append("")

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
