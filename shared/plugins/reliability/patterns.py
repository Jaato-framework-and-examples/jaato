"""Behavioral pattern detection for reliability plugin.

This module detects patterns like:
- Repetitive tool calls (same tool called N times)
- Introspection loops (stuck calling list_tools, etc.)
- Announce-no-action (model says "proceeding" but only reads)
- Read-only stalls (avoiding write tools)
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from .types import (
    BehavioralPattern,
    BehavioralPatternType,
    PatternDetectionConfig,
    PatternSeverity,
    PrerequisitePolicy,
    ToolCall,
)

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detects behavioral patterns in tool usage.

    This class tracks tool calls and model outputs within a turn to detect
    stall patterns that indicate the model is stuck in an unproductive loop.

    Prerequisite policy enforcement:
        Plugins can declare ``PrerequisitePolicy`` objects (via their
        ``get_prerequisite_policies()`` method) which are registered here
        through ``register_prerequisite_policy()``. For each policy, the
        detector tracks when the prerequisite tool was last called and
        fires a ``BehavioralPattern`` when a gated tool is invoked without
        the prerequisite having been met within the lookback window.

        This keeps the detector generic — it doesn't know about templates
        or any specific domain. The owning plugins declare what should be
        enforced, and the detector enforces it.
    """

    def __init__(
        self,
        config: Optional[PatternDetectionConfig] = None,
        session_id: str = "",
        model_name: str = "",
    ):
        self._config = config or PatternDetectionConfig()
        self._session_id = session_id
        self._model_name = model_name

        # Turn state
        self._turn_history: List[ToolCall] = []
        self._turn_start: Optional[datetime] = None
        self._turn_index: int = 0

        # Tracking
        self._last_action_index: int = -1
        self._announced_actions: List[str] = []
        self._consecutive_reads: int = 0

        # Prerequisite policies registered by plugins.
        # Maps policy_id → PrerequisitePolicy for lookup.
        self._prerequisite_policies: Dict[str, PrerequisitePolicy] = {}
        # Cross-turn tracking: maps prerequisite_tool → last turn it was called.
        # Shared across policies that use the same prerequisite tool.
        self._last_prerequisite_tool_turn: Dict[str, Optional[int]] = {}
        # Reverse index: maps gated tool name → list of policies that gate it.
        self._gated_tool_to_policies: Dict[str, List[PrerequisitePolicy]] = {}

        # Pattern history (for reporting)
        self._detected_patterns: List[BehavioralPattern] = []

        # Callbacks
        self._on_pattern_detected: Optional[Callable[[BehavioralPattern], None]] = None

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def register_prerequisite_policy(self, policy: PrerequisitePolicy) -> None:
        """Register a prerequisite policy declared by a plugin.

        The detector will enforce this policy by checking that the
        prerequisite tool has been called within the lookback window
        before any of the gated tools are invoked.

        Args:
            policy: The prerequisite policy to enforce.
        """
        self._prerequisite_policies[policy.policy_id] = policy
        # Initialize cross-turn tracking for this prerequisite tool
        if policy.prerequisite_tool not in self._last_prerequisite_tool_turn:
            self._last_prerequisite_tool_turn[policy.prerequisite_tool] = None
        # Build reverse index
        for tool in policy.gated_tools:
            if tool not in self._gated_tool_to_policies:
                self._gated_tool_to_policies[tool] = []
            if policy not in self._gated_tool_to_policies[tool]:
                self._gated_tool_to_policies[tool].append(policy)

    def set_session_context(self, session_id: str, model_name: str = "") -> None:
        """Set session context for pattern records."""
        self._session_id = session_id
        if model_name:
            self._model_name = model_name

    def set_model_name(self, model_name: str) -> None:
        """Update the current model name."""
        self._model_name = model_name

    def set_pattern_hook(
        self,
        on_pattern_detected: Optional[Callable[[BehavioralPattern], None]] = None,
    ) -> None:
        """Set callback for pattern detection.

        Args:
            on_pattern_detected: Called when a pattern is detected.
                Signature: (pattern: BehavioralPattern) -> None
        """
        self._on_pattern_detected = on_pattern_detected

    # -------------------------------------------------------------------------
    # Turn Lifecycle
    # -------------------------------------------------------------------------

    def on_turn_start(self, turn_index: int) -> None:
        """Called when a new turn begins."""
        self._turn_history = []
        self._turn_start = datetime.now()
        self._turn_index = turn_index
        self._announced_actions = []
        self._consecutive_reads = 0
        self._last_action_index = -1

    def on_turn_end(self) -> Optional[BehavioralPattern]:
        """Called when a turn ends. Check for announce-no-action."""
        if not self._config.enabled:
            return None

        # Check if model announced action but didn't follow through
        pattern = self._check_announce_no_action()
        if pattern:
            self._emit_pattern(pattern)
            return pattern

        return None

    # -------------------------------------------------------------------------
    # Tool Call Tracking
    # -------------------------------------------------------------------------

    def on_tool_called(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Optional[BehavioralPattern]:
        """Called BEFORE tool execution. Returns pattern if detected.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool

        Returns:
            BehavioralPattern if a stall pattern is detected, None otherwise
        """
        if not self._config.enabled:
            return None

        # Record the call
        call = ToolCall(tool_name=tool_name, args=args, timestamp=datetime.now())
        self._turn_history.append(call)

        # Track cross-turn prerequisite tool calls
        if tool_name in self._last_prerequisite_tool_turn:
            self._last_prerequisite_tool_turn[tool_name] = self._turn_index

        # Track read vs action tools
        if tool_name in self._config.read_only_tools:
            self._consecutive_reads += 1
        elif tool_name in self._config.action_tools:
            self._consecutive_reads = 0

        # Check for prerequisite policy violations
        pattern = self._check_prerequisite_policies(tool_name)
        if pattern:
            self._emit_pattern(pattern)
            return pattern

        # Check for repetitive calls
        pattern = self._check_repetitive_calls(tool_name)
        if pattern:
            self._emit_pattern(pattern)
            return pattern

        # Check for introspection loop
        pattern = self._check_introspection_loop(tool_name)
        if pattern:
            self._emit_pattern(pattern)
            return pattern

        # Check for read-only stall
        pattern = self._check_read_only_stall(tool_name)
        if pattern:
            self._emit_pattern(pattern)
            return pattern

        return None

    def on_tool_result(
        self,
        tool_name: str,
        success: bool,
        result: Any = None,
    ) -> Optional[BehavioralPattern]:
        """Called after tool execution.

        Args:
            tool_name: Name of the tool that was executed
            success: Whether execution succeeded
            result: The tool result

        Returns:
            BehavioralPattern if error-retry loop detected, None otherwise
        """
        if not self._config.enabled:
            return None

        # Update the last tool call with result
        if self._turn_history:
            self._turn_history[-1].success = success

        # Track successful actions
        if tool_name in self._config.action_tools and success:
            self._last_action_index = len(self._turn_history) - 1
            self._announced_actions = []  # Action taken, clear announcements
            self._consecutive_reads = 0

        # Check for error-retry loop (same failing tool repeatedly)
        if not success:
            pattern = self._check_error_retry_loop(tool_name)
            if pattern:
                self._emit_pattern(pattern)
                return pattern

        return None

    # -------------------------------------------------------------------------
    # Model Output Tracking
    # -------------------------------------------------------------------------

    def on_model_text(self, text: str) -> None:
        """Called when model outputs text. Track announcements.

        Args:
            text: The text output by the model
        """
        if not self._config.enabled:
            return

        text_lower = text.lower()
        for phrase in self._config.announce_phrases:
            if phrase in text_lower:
                self._announced_actions.append(text)
                break

    # -------------------------------------------------------------------------
    # Pattern Detection Logic
    # -------------------------------------------------------------------------

    def _check_prerequisite_policies(self, tool_name: str) -> Optional[BehavioralPattern]:
        """Check if any registered prerequisite policy is violated.

        For each policy that gates this tool, verifies that the prerequisite
        tool was called within the lookback window (current turn + N previous
        turns). Returns the first violation found.

        Returns:
            BehavioralPattern if a prerequisite was violated, None if all
            policies are satisfied or no policies gate this tool.
        """
        policies = self._gated_tool_to_policies.get(tool_name)
        if not policies:
            return None

        for policy in policies:
            violation = self._check_single_prerequisite(tool_name, policy)
            if violation:
                return violation

        return None

    def _check_single_prerequisite(
        self,
        tool_name: str,
        policy: PrerequisitePolicy,
    ) -> Optional[BehavioralPattern]:
        """Check a single prerequisite policy for a gated tool call.

        Args:
            tool_name: The gated tool that was called.
            policy: The prerequisite policy to check.

        Returns:
            BehavioralPattern if the prerequisite was violated, None if satisfied.
        """
        lookback = policy.lookback_turns

        # Check cross-turn tracking for the prerequisite tool
        last_turn = self._last_prerequisite_tool_turn.get(policy.prerequisite_tool)
        if last_turn is not None:
            turns_since_check = self._turn_index - last_turn
            if turns_since_check <= lookback:
                return None  # Prerequisite satisfied

        # Also check the current turn's history (the prerequisite tool may
        # have been called earlier in this same turn).
        for call in self._turn_history:
            if call.tool_name == policy.prerequisite_tool:
                return None  # Prerequisite satisfied within this turn

        # Determine severity based on how many prior violations for this
        # specific policy.  Thresholds come from the policy itself so that
        # each policy can define its own escalation curve.
        violation_count = sum(
            1 for p in self._detected_patterns
            if getattr(p, 'policy_id', None) == policy.policy_id
        )
        severity = policy.get_severity(violation_count)

        duration = self._calculate_duration()
        expected_action = policy.expected_action_template.format(
            tool_name=tool_name,
            prerequisite_tool=policy.prerequisite_tool,
        )

        return BehavioralPattern(
            pattern_type=policy.pattern_type,
            detected_at=datetime.now(),
            turn_index=self._turn_index,
            session_id=self._session_id,
            tool_sequence=[tool_name],
            repetition_count=violation_count + 1,
            duration_seconds=duration,
            model_name=self._model_name,
            severity=severity,
            expected_action=expected_action,
            policy_id=policy.policy_id,
        )

    def _check_repetitive_calls(self, tool_name: str) -> Optional[BehavioralPattern]:
        """Check for same tool called repeatedly."""
        threshold = self._config.repetitive_call_threshold

        # Count recent consecutive calls to the same tool
        count = 0
        for call in reversed(self._turn_history):
            if call.tool_name == tool_name:
                count += 1
            else:
                break

        if count < threshold:
            return None

        # Determine severity based on count
        if count >= threshold + 3:
            severity = PatternSeverity.SEVERE
        elif count >= threshold + 1:
            severity = PatternSeverity.MODERATE
        else:
            severity = PatternSeverity.MINOR

        duration = self._calculate_duration()

        return BehavioralPattern(
            pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
            detected_at=datetime.now(),
            turn_index=self._turn_index,
            session_id=self._session_id,
            tool_sequence=[c.tool_name for c in self._turn_history[-count:]],
            repetition_count=count,
            duration_seconds=duration,
            model_name=self._model_name,
            severity=severity,
        )

    def _check_introspection_loop(self, tool_name: str) -> Optional[BehavioralPattern]:
        """Check for introspection tool loop (list_tools, get_schema, etc.)."""
        introspection_tools = self._config.introspection_tool_names
        threshold = self._config.introspection_loop_threshold

        if tool_name not in introspection_tools:
            return None

        # Count introspection calls since last action
        introspection_count = 0
        for call in reversed(self._turn_history):
            if call.tool_name in introspection_tools:
                introspection_count += 1
            elif call.tool_name in self._config.action_tools:
                break

        if introspection_count < threshold:
            return None

        # Determine severity
        if introspection_count >= threshold + 3:
            severity = PatternSeverity.SEVERE
        elif introspection_count >= threshold + 1:
            severity = PatternSeverity.MODERATE
        else:
            severity = PatternSeverity.MINOR

        duration = self._calculate_duration()

        return BehavioralPattern(
            pattern_type=BehavioralPatternType.INTROSPECTION_LOOP,
            detected_at=datetime.now(),
            turn_index=self._turn_index,
            session_id=self._session_id,
            tool_sequence=[c.tool_name for c in self._turn_history[-introspection_count:]],
            repetition_count=introspection_count,
            duration_seconds=duration,
            model_name=self._model_name,
            severity=severity,
            expected_action="Execute a task rather than introspecting",
        )

    def _check_read_only_stall(self, tool_name: str) -> Optional[BehavioralPattern]:
        """Check for read-only tool stall (no writes/actions)."""
        threshold = self._config.max_reads_before_action

        if self._consecutive_reads < threshold:
            return None

        # Only trigger if current tool is also a read
        if tool_name not in self._config.read_only_tools:
            return None

        # Determine severity
        if self._consecutive_reads >= threshold + 3:
            severity = PatternSeverity.SEVERE
        elif self._consecutive_reads >= threshold + 1:
            severity = PatternSeverity.MODERATE
        else:
            severity = PatternSeverity.MINOR

        duration = self._calculate_duration()

        # Get recent read tools
        recent_reads = []
        for call in reversed(self._turn_history):
            if call.tool_name in self._config.read_only_tools:
                recent_reads.insert(0, call.tool_name)
                if len(recent_reads) >= self._consecutive_reads:
                    break

        return BehavioralPattern(
            pattern_type=BehavioralPatternType.READ_ONLY_LOOP,
            detected_at=datetime.now(),
            turn_index=self._turn_index,
            session_id=self._session_id,
            tool_sequence=recent_reads,
            repetition_count=self._consecutive_reads,
            duration_seconds=duration,
            model_name=self._model_name,
            severity=severity,
            expected_action="Write or execute action to make progress",
        )

    def _check_announce_no_action(self) -> Optional[BehavioralPattern]:
        """Check if model announced action but didn't follow through."""
        if not self._announced_actions:
            return None

        # Check if there was an action after the announcement
        if self._last_action_index >= 0:
            return None

        # Check if there were only read tools (or no tools) after announcement
        action_taken = False
        for call in self._turn_history:
            if call.tool_name in self._config.action_tools and call.success:
                action_taken = True
                break

        if action_taken:
            return None

        # Determine severity based on number of announcements
        num_announcements = len(self._announced_actions)
        if num_announcements >= 3:
            severity = PatternSeverity.SEVERE
        elif num_announcements >= 2:
            severity = PatternSeverity.MODERATE
        else:
            severity = PatternSeverity.MINOR

        duration = self._calculate_duration()

        return BehavioralPattern(
            pattern_type=BehavioralPatternType.ANNOUNCE_NO_ACTION,
            detected_at=datetime.now(),
            turn_index=self._turn_index,
            session_id=self._session_id,
            tool_sequence=[c.tool_name for c in self._turn_history],
            repetition_count=num_announcements,
            duration_seconds=duration,
            model_name=self._model_name,
            severity=severity,
            last_model_text=self._announced_actions[-1] if self._announced_actions else None,
            expected_action="Execute the announced action",
        )

    def _check_error_retry_loop(self, tool_name: str) -> Optional[BehavioralPattern]:
        """Check for retrying same failing operation unchanged.

        Detects consecutive failures of the same tool with similar arguments.
        The threshold is configurable via ``error_retry_threshold`` in
        ``PatternDetectionConfig`` (default: 3).
        """
        threshold = self._config.error_retry_threshold

        recent_failures: List[ToolCall] = []
        for call in reversed(self._turn_history):
            if call.tool_name == tool_name and call.success is False:
                recent_failures.insert(0, call)
            elif call.tool_name == tool_name and call.success is True:
                break  # A success breaks the streak
            if len(recent_failures) >= threshold:
                break

        if len(recent_failures) < threshold:
            return None

        # Check if args are similar (simple check: same keys)
        first_args = set(recent_failures[0].args.keys())
        similar = all(set(f.args.keys()) == first_args for f in recent_failures[1:])

        if not similar:
            return None

        count = len(recent_failures)
        if count >= threshold + 2:
            severity = PatternSeverity.SEVERE
        elif count >= threshold:
            severity = PatternSeverity.MODERATE
        else:
            severity = PatternSeverity.MINOR

        duration = self._calculate_duration()

        return BehavioralPattern(
            pattern_type=BehavioralPatternType.ERROR_RETRY_LOOP,
            detected_at=datetime.now(),
            turn_index=self._turn_index,
            session_id=self._session_id,
            tool_sequence=[c.tool_name for c in recent_failures],
            repetition_count=count,
            duration_seconds=duration,
            model_name=self._model_name,
            severity=severity,
            expected_action="Try a different approach or parameters",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calculate_duration(self) -> float:
        """Calculate duration since turn start."""
        if not self._turn_start:
            return 0.0
        return (datetime.now() - self._turn_start).total_seconds()

    def _emit_pattern(self, pattern: BehavioralPattern) -> None:
        """Record and emit a detected pattern."""
        self._detected_patterns.append(pattern)
        logger.debug(
            f"Pattern detected: {pattern.pattern_type.value} "
            f"severity={pattern.severity.value} "
            f"repetitions={pattern.repetition_count}"
        )
        if self._on_pattern_detected:
            self._on_pattern_detected(pattern)

    # -------------------------------------------------------------------------
    # State Query
    # -------------------------------------------------------------------------

    def get_detected_patterns(self) -> List[BehavioralPattern]:
        """Get all detected patterns."""
        return list(self._detected_patterns)

    def get_recent_patterns(self, limit: int = 10) -> List[BehavioralPattern]:
        """Get recent detected patterns."""
        return self._detected_patterns[-limit:]

    def get_turn_history(self) -> List[ToolCall]:
        """Get tool call history for current turn."""
        return list(self._turn_history)

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns."""
        if not self._detected_patterns:
            return {"total": 0, "by_type": {}, "by_severity": {}}

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for pattern in self._detected_patterns:
            ptype = pattern.pattern_type.value
            psev = pattern.severity.value
            by_type[ptype] = by_type.get(ptype, 0) + 1
            by_severity[psev] = by_severity.get(psev, 0) + 1

        return {
            "total": len(self._detected_patterns),
            "by_type": by_type,
            "by_severity": by_severity,
        }

    def clear_history(self) -> None:
        """Clear pattern history."""
        self._detected_patterns = []

    def reset(self) -> None:
        """Reset all state, including cross-turn tracking.

        Registered policies are preserved — only runtime tracking state
        is cleared. Call ``register_prerequisite_policy()`` again only if
        you want to change the policy set.
        """
        self._turn_history = []
        self._turn_start = None
        self._turn_index = 0
        self._last_action_index = -1
        self._announced_actions = []
        self._consecutive_reads = 0
        # Reset cross-turn tracking for all prerequisite tools
        for tool in self._last_prerequisite_tool_turn:
            self._last_prerequisite_tool_turn[tool] = None
        self._detected_patterns = []
