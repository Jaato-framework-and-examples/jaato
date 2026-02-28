"""Nudge injection system for guiding models out of stall patterns.

This module provides the NudgeStrategy and NudgeInjector classes that
create and inject guidance messages when behavioral patterns are detected.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .types import (
    BehavioralPattern,
    BehavioralPatternType,
    Nudge,
    NudgeConfig,
    NudgeLevel,
    NudgeType,
    PatternSeverity,
)

logger = logging.getLogger(__name__)


class NudgeStrategy:
    """Determines what nudge to inject based on detected patterns.

    This class contains templates for different pattern types and severities,
    and creates appropriate nudges based on the detected pattern.
    """

    # Templates: pattern_type -> severity -> (nudge_type, template)
    NUDGE_TEMPLATES: Dict[BehavioralPatternType, Dict[PatternSeverity, tuple]] = {
        BehavioralPatternType.REPETITIVE_CALLS: {
            PatternSeverity.MINOR: (
                NudgeType.GENTLE_REMINDER,
                "You've called {tool_name} {count} times. Consider if you have the information needed to proceed."
            ),
            PatternSeverity.MODERATE: (
                NudgeType.DIRECT_INSTRUCTION,
                "NOTICE: You are in a loop calling {tool_name} repeatedly. Stop and take action on what you've learned."
            ),
            PatternSeverity.SEVERE: (
                NudgeType.INTERRUPT,
                "BLOCKED: Detected {count}x repeated calls to {tool_name}. The system is pausing for user intervention."
            ),
        },
        BehavioralPatternType.ANNOUNCE_NO_ACTION: {
            PatternSeverity.MINOR: (
                NudgeType.GENTLE_REMINDER,
                "You mentioned proceeding but haven't called an action tool yet. Please execute the planned action."
            ),
            PatternSeverity.MODERATE: (
                NudgeType.DIRECT_INSTRUCTION,
                "NOTICE: You've announced action {count} times without executing. Call the appropriate tool NOW."
            ),
            PatternSeverity.SEVERE: (
                NudgeType.INTERRUPT,
                "BLOCKED: Repeated announcements without action. System pausing for user input."
            ),
        },
        BehavioralPatternType.INTROSPECTION_LOOP: {
            PatternSeverity.MINOR: (
                NudgeType.GENTLE_REMINDER,
                "You have the tool list. Proceed with the actual task instead of re-querying tools."
            ),
            PatternSeverity.MODERATE: (
                NudgeType.DIRECT_INSTRUCTION,
                "NOTICE: Stop querying tool metadata. You have sufficient information. Execute the task."
            ),
            PatternSeverity.SEVERE: (
                NudgeType.INTERRUPT,
                "BLOCKED: Stuck in introspection loop. System pausing for user guidance."
            ),
        },
        BehavioralPatternType.READ_ONLY_LOOP: {
            PatternSeverity.MINOR: (
                NudgeType.GENTLE_REMINDER,
                "You've read {count} files without making changes. If you have what you need, proceed with edits."
            ),
            PatternSeverity.MODERATE: (
                NudgeType.DIRECT_INSTRUCTION,
                "NOTICE: You are in a read-only loop. You have sufficient context. Make the required changes now."
            ),
            PatternSeverity.SEVERE: (
                NudgeType.INTERRUPT,
                "BLOCKED: Stuck reading without action. System pausing - please specify what action to take."
            ),
        },
        BehavioralPatternType.ERROR_RETRY_LOOP: {
            PatternSeverity.MINOR: (
                NudgeType.GENTLE_REMINDER,
                "The same operation has failed {count} times. Consider trying a different approach."
            ),
            PatternSeverity.MODERATE: (
                NudgeType.DIRECT_INSTRUCTION,
                "NOTICE: Repeated failures with {tool_name}. Stop retrying the same way and try an alternative."
            ),
            PatternSeverity.SEVERE: (
                NudgeType.INTERRUPT,
                "BLOCKED: Repeated identical failures. System pausing for user guidance on how to proceed."
            ),
        },
        BehavioralPatternType.PLANNING_LOOP: {
            PatternSeverity.MINOR: (
                NudgeType.GENTLE_REMINDER,
                "You've been planning for a while. Consider starting execution with what you have."
            ),
            PatternSeverity.MODERATE: (
                NudgeType.DIRECT_INSTRUCTION,
                "NOTICE: Excessive planning without execution. Start implementing NOW."
            ),
            PatternSeverity.SEVERE: (
                NudgeType.INTERRUPT,
                "BLOCKED: Stuck in planning loop. System pausing - user needs to confirm direction."
            ),
        },
        BehavioralPatternType.VALIDATION_BYPASS: {
            PatternSeverity.MINOR: (
                NudgeType.DIRECT_INSTRUCTION,
                "Validation steps require subagent evidence. Use addDependentStep + completeStepWithOutput pattern."
            ),
            PatternSeverity.MODERATE: (
                NudgeType.DIRECT_INSTRUCTION,
                "NOTICE: You have attempted to complete a validation step without delegating to a validator subagent {count} times. "
                "Spawn a validator subagent and use addDependentStep NOW."
            ),
            PatternSeverity.SEVERE: (
                NudgeType.INTERRUPT,
                "BLOCKED: Repeated validation bypass attempts. System pausing for user intervention."
            ),
        },
    }

    # Default template for unknown patterns
    DEFAULT_TEMPLATES = {
        PatternSeverity.MINOR: (
            NudgeType.GENTLE_REMINDER,
            "A behavioral pattern has been detected. Please review your approach and proceed."
        ),
        PatternSeverity.MODERATE: (
            NudgeType.DIRECT_INSTRUCTION,
            "NOTICE: You appear to be stuck. Please take a different approach to make progress."
        ),
        PatternSeverity.SEVERE: (
            NudgeType.INTERRUPT,
            "BLOCKED: System has detected you are stuck. Pausing for user intervention."
        ),
    }

    def __init__(self):
        # Templates registered by plugins via PrerequisitePolicy.nudge_templates.
        # Maps policy_id → severity → (nudge_type, template_str).
        self._policy_templates: Dict[str, Dict[PatternSeverity, tuple]] = {}

    def register_policy_templates(
        self,
        policy_id: str,
        nudge_templates: Dict[PatternSeverity, tuple],
    ) -> None:
        """Register nudge templates declared by a plugin's PrerequisitePolicy.

        These templates take precedence over the built-in NUDGE_TEMPLATES
        and DEFAULT_TEMPLATES when the pattern carries a matching ``policy_id``.

        Args:
            policy_id: The policy identifier (e.g., "template_check").
            nudge_templates: Map of severity → (NudgeType, template_str).
        """
        self._policy_templates[policy_id] = nudge_templates

    def create_nudge(self, pattern: BehavioralPattern) -> Nudge:
        """Create appropriate nudge for detected pattern.

        Template lookup order:
        1. Policy-declared templates keyed by ``policy_id``
        2. Built-in ``NUDGE_TEMPLATES`` for well-known pattern types
        3. ``DEFAULT_TEMPLATES`` as fallback

        Args:
            pattern: The detected behavioral pattern

        Returns:
            A Nudge with appropriate type and message
        """
        # Policy-declared templates (keyed by policy_id) take priority
        templates = None
        if pattern.policy_id:
            templates = self._policy_templates.get(pattern.policy_id)
        if templates is None:
            templates = self.NUDGE_TEMPLATES.get(pattern.pattern_type, self.DEFAULT_TEMPLATES)
        nudge_type, template = templates.get(
            pattern.severity,
            self.DEFAULT_TEMPLATES[pattern.severity]
        )

        # Extract tool name from sequence
        tool_name = pattern.tool_sequence[-1] if pattern.tool_sequence else "unknown"

        # Format message with pattern context
        message = template.format(
            tool_name=tool_name,
            count=pattern.repetition_count,
            duration=f"{pattern.duration_seconds:.0f}s",
        )

        return Nudge(
            nudge_type=nudge_type,
            message=message,
            pattern=pattern,
            injected_at=datetime.now(),
        )

    def should_inject(self, nudge: Nudge, config: NudgeConfig) -> bool:
        """Determine if a nudge should be injected based on config.

        Args:
            nudge: The nudge to potentially inject
            config: The nudge configuration

        Returns:
            True if the nudge should be injected
        """
        if not config.enabled:
            return False

        if config.level == NudgeLevel.OFF:
            return False

        if config.level == NudgeLevel.GENTLE:
            return nudge.nudge_type == NudgeType.GENTLE_REMINDER

        if config.level == NudgeLevel.DIRECT:
            return nudge.nudge_type in (NudgeType.GENTLE_REMINDER, NudgeType.DIRECT_INSTRUCTION)

        # FULL level allows all nudge types
        return True


class NudgeInjector:
    """Manages nudge injection into the model context.

    This class coordinates between pattern detection and the session/client
    to inject appropriate guidance when stall patterns are detected.
    """

    def __init__(
        self,
        config: Optional[NudgeConfig] = None,
        strategy: Optional[NudgeStrategy] = None,
    ):
        self._config = config or NudgeConfig()
        self._strategy = strategy or NudgeStrategy()

        # History tracking
        self._nudge_history: List[Nudge] = []
        self._last_nudge_by_type: Dict[BehavioralPatternType, datetime] = {}

        # Callbacks for injection (model-facing)
        self._inject_system_guidance: Optional[Callable[[str], None]] = None
        self._inject_context_hint: Optional[Callable[[str], None]] = None
        self._request_pause: Optional[Callable[[str], None]] = None

        # Callback for user-visible notification (UI-facing).
        # Signature: (source: str, text: str, mode: str) -> None
        # Follows the same OutputCallback pattern used by enrichment
        # notifications in template/memory/reference plugins.
        self._notify_user: Optional[Callable[[str, str, str], None]] = None

        # Event callbacks
        self._on_nudge_injected: Optional[Callable[[Nudge], None]] = None
        self._on_nudge_acknowledged: Optional[Callable[[Nudge], None]] = None

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_config(self, config: NudgeConfig) -> None:
        """Update nudge configuration."""
        self._config = config

    def set_level(self, level: NudgeLevel) -> None:
        """Set the nudge level."""
        self._config.level = level

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable nudge injection."""
        self._config.enabled = enabled

    def set_injection_callbacks(
        self,
        inject_system_guidance: Optional[Callable[[str], None]] = None,
        inject_context_hint: Optional[Callable[[str], None]] = None,
        request_pause: Optional[Callable[[str], None]] = None,
        notify_user: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        """Set callbacks for injecting nudges into the session.

        Args:
            inject_system_guidance: Inject as system message (high priority)
            inject_context_hint: Inject as context hint (lower priority)
            request_pause: Request user intervention (highest priority)
            notify_user: Emit a user-visible notification via the output
                callback. Signature matches OutputCallback:
                (source: str, text: str, mode: str) -> None.
                Called with source="enrichment" to integrate with the
                enrichment notification rendering pipeline (same as
                template/memory/reference notifications).
        """
        self._inject_system_guidance = inject_system_guidance
        self._inject_context_hint = inject_context_hint
        self._request_pause = request_pause
        self._notify_user = notify_user

    def set_nudge_hooks(
        self,
        on_nudge_injected: Optional[Callable[[Nudge], None]] = None,
        on_nudge_acknowledged: Optional[Callable[[Nudge], None]] = None,
    ) -> None:
        """Set hooks for nudge lifecycle events.

        Args:
            on_nudge_injected: Called when a nudge is injected
            on_nudge_acknowledged: Called when a nudge is acknowledged
        """
        self._on_nudge_injected = on_nudge_injected
        self._on_nudge_acknowledged = on_nudge_acknowledged

    # -------------------------------------------------------------------------
    # Core Injection Logic
    # -------------------------------------------------------------------------

    def on_pattern_detected(self, pattern: BehavioralPattern) -> Optional[Nudge]:
        """Handle a detected pattern and potentially inject a nudge.

        Args:
            pattern: The detected behavioral pattern

        Returns:
            The injected Nudge if one was created, None otherwise
        """
        if not self._config.enabled:
            return None

        # Check cooldown
        if not self._check_cooldown(pattern.pattern_type):
            logger.debug(f"Nudge for {pattern.pattern_type.value} on cooldown")
            return None

        # Create nudge
        nudge = self._strategy.create_nudge(pattern)

        # Check if nudge should be injected based on level
        if not self._strategy.should_inject(nudge, self._config):
            logger.debug(f"Nudge type {nudge.nudge_type.value} filtered by level {self._config.level.value}")
            return None

        # Inject the nudge
        self._inject_nudge(nudge)

        # Record in history
        self._nudge_history.append(nudge)
        self._last_nudge_by_type[pattern.pattern_type] = datetime.now()

        # Emit event
        if self._on_nudge_injected:
            self._on_nudge_injected(nudge)

        logger.info(f"Nudge injected: {nudge.nudge_type.value} for {pattern.pattern_type.value}")

        return nudge

    def _check_cooldown(self, pattern_type: BehavioralPatternType) -> bool:
        """Check if enough time has passed since last nudge for this pattern type."""
        last_nudge = self._last_nudge_by_type.get(pattern_type)
        if not last_nudge:
            return True

        elapsed = (datetime.now() - last_nudge).total_seconds()
        return elapsed >= self._config.cooldown_seconds

    def _inject_nudge(self, nudge: Nudge) -> None:
        """Actually inject the nudge using configured callbacks.

        Performs two actions:
        1. Injects the nudge into the model's context (system guidance,
           context hint, or pause request depending on severity).
        2. Emits a user-visible notification via the output callback so
           the user sees the nudge in the UI alongside other enrichment
           notifications (template, memory, reference, etc.).
        """
        message = nudge.to_system_message()

        if nudge.nudge_type == NudgeType.INTERRUPT:
            # Interrupts request a pause
            if self._request_pause:
                self._request_pause(message)
            elif self._inject_system_guidance:
                self._inject_system_guidance(message)
        elif nudge.nudge_type == NudgeType.DIRECT_INSTRUCTION:
            # Direct instructions go to system guidance
            if self._inject_system_guidance:
                self._inject_system_guidance(message)
            elif self._inject_context_hint:
                self._inject_context_hint(message)
        else:
            # Gentle reminders go to context hints
            if self._inject_context_hint:
                self._inject_context_hint(message)
            elif self._inject_system_guidance:
                self._inject_system_guidance(message)

        # Emit user-visible notification
        if self._notify_user:
            user_message = self._format_user_notification(nudge)
            self._notify_user("enrichment", user_message, "write")

    @staticmethod
    def _format_user_notification(nudge: Nudge) -> str:
        """Format a nudge as a user-visible enrichment notification.

        Produces a compact notification string matching the style used by
        other enrichment plugins (template, memory, reference). Example::

            nudge <- reliability: Template check missing - writeNewFile called
                     without prior listAvailableTemplates

        Args:
            nudge: The nudge being injected.

        Returns:
            Formatted notification string for the output callback.
        """
        pattern = nudge.pattern
        pattern_label = pattern.pattern_type.value.replace("_", " ")
        tool_name = pattern.tool_sequence[-1] if pattern.tool_sequence else "unknown"

        severity_prefix = {
            PatternSeverity.MINOR: "",
            PatternSeverity.MODERATE: "[repeated] ",
            PatternSeverity.SEVERE: "[blocked] ",
        }.get(pattern.severity, "")

        return (
            f"  nudge \u2190 reliability: {severity_prefix}{pattern_label} "
            f"\u2014 {tool_name}"
        )

    # -------------------------------------------------------------------------
    # Acknowledgment & Effectiveness Tracking
    # -------------------------------------------------------------------------

    def acknowledge_nudge(self, nudge: Nudge, effective: bool = False) -> None:
        """Mark a nudge as acknowledged.

        Args:
            nudge: The nudge to acknowledge
            effective: Whether the nudge resolved the pattern
        """
        nudge.acknowledged = True
        nudge.effective = effective

        if self._on_nudge_acknowledged:
            self._on_nudge_acknowledged(nudge)

    def mark_last_nudge_effective(self, effective: bool = True) -> None:
        """Mark the most recent nudge as effective or not."""
        if self._nudge_history:
            last = self._nudge_history[-1]
            last.effective = effective
            last.acknowledged = True

    # -------------------------------------------------------------------------
    # State Query
    # -------------------------------------------------------------------------

    def get_nudge_history(self) -> List[Nudge]:
        """Get all nudge history."""
        return list(self._nudge_history)

    def get_recent_nudges(self, limit: int = 10) -> List[Nudge]:
        """Get recent nudges."""
        return self._nudge_history[-limit:]

    def get_pending_nudges(self) -> List[Nudge]:
        """Get nudges that haven't been acknowledged."""
        return [n for n in self._nudge_history if not n.acknowledged]

    def get_nudge_summary(self) -> Dict[str, Any]:
        """Get summary of nudge injection activity."""
        if not self._nudge_history:
            return {
                "total": 0,
                "effective": 0,
                "by_type": {},
                "effectiveness_rate": 0.0,
            }

        by_type: Dict[str, int] = {}
        effective_count = 0

        for nudge in self._nudge_history:
            ntype = nudge.nudge_type.value
            by_type[ntype] = by_type.get(ntype, 0) + 1
            if nudge.effective:
                effective_count += 1

        acknowledged = [n for n in self._nudge_history if n.acknowledged]
        effectiveness_rate = effective_count / len(acknowledged) if acknowledged else 0.0

        return {
            "total": len(self._nudge_history),
            "effective": effective_count,
            "acknowledged": len(acknowledged),
            "pending": len(self._nudge_history) - len(acknowledged),
            "by_type": by_type,
            "effectiveness_rate": effectiveness_rate,
        }

    def clear_history(self) -> None:
        """Clear nudge history."""
        self._nudge_history = []
        self._last_nudge_by_type = {}

    def reset(self) -> None:
        """Reset all state."""
        self.clear_history()

    # -------------------------------------------------------------------------
    # Manual Nudge Creation
    # -------------------------------------------------------------------------

    def create_manual_nudge(
        self,
        message: str,
        nudge_type: NudgeType = NudgeType.DIRECT_INSTRUCTION,
    ) -> Optional[Nudge]:
        """Create and inject a manual nudge (not from pattern detection).

        Args:
            message: The nudge message
            nudge_type: The type of nudge

        Returns:
            The injected Nudge if enabled, None otherwise
        """
        if not self._config.enabled:
            return None

        # Create a minimal pattern for tracking
        pattern = BehavioralPattern(
            pattern_type=BehavioralPatternType.TOOL_AVOIDANCE,  # Generic
            detected_at=datetime.now(),
            turn_index=0,
            session_id="",
            tool_sequence=[],
            repetition_count=0,
            duration_seconds=0.0,
            model_name="",
        )

        nudge = Nudge(
            nudge_type=nudge_type,
            message=message,
            pattern=pattern,
            injected_at=datetime.now(),
        )

        if not self._strategy.should_inject(nudge, self._config):
            return None

        self._inject_nudge(nudge)
        self._nudge_history.append(nudge)

        if self._on_nudge_injected:
            self._on_nudge_injected(nudge)

        return nudge
