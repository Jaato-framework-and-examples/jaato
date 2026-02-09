"""Core types for the reliability plugin."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
import hashlib
import json


class FailureSeverity(Enum):
    """Classification of failure severity."""

    # Low severity - expected/recoverable failures
    TRANSIENT = "transient"           # Network timeouts, rate limits (retried by framework)
    NOT_FOUND = "not_found"           # File/resource doesn't exist
    INVALID_INPUT = "invalid_input"   # Bad arguments (model error, not tool error)

    # Medium severity - concerning but not critical
    PERMISSION = "permission"         # Auth/access denied
    VALIDATION = "validation"         # Schema/format errors from external service
    TIMEOUT = "timeout"               # Operation exceeded time limit

    # High severity - indicates tool/service problems
    SERVER_ERROR = "server_error"     # 5xx errors, service unavailable
    CRASH = "crash"                   # Tool process crashed
    CORRUPTION = "corruption"         # Data integrity issues

    # Critical - requires immediate escalation
    SECURITY = "security"             # Potential security issue detected
    REPEATED_AUTH = "repeated_auth"   # Multiple auth failures (possible credential issue)


class TrustState(Enum):
    """Trust state for a tool+parameters combination."""

    TRUSTED = "trusted"         # Normal operation, uses standard permissions
    ESCALATED = "escalated"     # Requires explicit approval regardless of whitelist
    RECOVERING = "recovering"   # In cooldown, tracking successes for recovery
    BLOCKED = "blocked"         # Permanent block (critical failures)


@dataclass
class EscalationInfo:
    """Information about escalation status for a tool invocation.

    Returned by check_escalation() for use in permission prompts.
    """

    is_escalated: bool
    state: TrustState = TrustState.TRUSTED

    # Escalation details (only if is_escalated=True)
    reason: Optional[str] = None
    failure_count: int = 0
    window_description: Optional[str] = None  # e.g., "3 failures in 1 hour"

    # Recovery info
    recovery_progress: Optional[str] = None   # e.g., "1/3 successes"
    recovery_hint: Optional[str] = None       # e.g., "2 more successes needed"

    # Display formatting
    severity_label: Optional[str] = None      # e.g., "⚠ ESCALATED", "🚫 BLOCKED"
    summary: Optional[str] = None             # Brief summary for permission prompt

    def to_display_lines(self) -> List[str]:
        """Format escalation info for permission prompt display."""
        if not self.is_escalated:
            return []

        lines = []

        # Header with severity
        if self.severity_label:
            lines.append(f"{self.severity_label}")
        elif self.state == TrustState.BLOCKED:
            lines.append("🚫 BLOCKED - Security concern")
        elif self.state == TrustState.ESCALATED:
            lines.append("⚠ ESCALATED - Requires approval")
        elif self.state == TrustState.RECOVERING:
            lines.append("↻ RECOVERING - Proving reliability")

        # Reason
        if self.reason:
            lines.append(f"   Reason: {self.reason}")

        # Window description
        if self.window_description:
            lines.append(f"   History: {self.window_description}")

        # Recovery progress
        if self.recovery_progress:
            lines.append(f"   Progress: {self.recovery_progress}")

        # Hint
        if self.recovery_hint:
            lines.append(f"   Note: {self.recovery_hint}")

        return lines


@dataclass
class FailureKey:
    """Identifies a specific tool+parameters combination for tracking."""

    tool_name: str
    parameter_signature: str  # Normalized representation of key params

    def to_string(self) -> str:
        """Convert to string key for dict storage."""
        if self.parameter_signature:
            return f"{self.tool_name}|{self.parameter_signature}"
        return self.tool_name

    @classmethod
    def from_string(cls, key: str) -> "FailureKey":
        """Parse from string key."""
        if "|" in key:
            tool_name, signature = key.split("|", 1)
            return cls(tool_name=tool_name, parameter_signature=signature)
        return cls(tool_name=key, parameter_signature="")

    @classmethod
    def from_invocation(cls, tool_name: str, args: Dict[str, Any]) -> "FailureKey":
        """Create failure key from tool invocation."""
        key_params = cls._extract_key_params(tool_name, args)
        signature = cls._normalize_signature(key_params)
        return cls(tool_name=tool_name, parameter_signature=signature)

    @staticmethod
    def _extract_key_params(tool_name: str, args: Dict[str, Any]) -> Dict[str, str]:
        """Extract parameters relevant for failure tracking."""

        # File operations: track by path prefix
        if tool_name in ("readFile", "writeFile", "updateFile", "removeFile", "Read", "Write", "Edit"):
            path = args.get("path", "") or args.get("file_path", "")
            if path:
                try:
                    return {"path_prefix": str(Path(path).parent)}
                except Exception:
                    return {"path_prefix": "unknown"}
            return {}

        # HTTP requests: track by domain
        if tool_name in ("http_request", "fetch", "WebFetch"):
            url = args.get("url", "")
            if url:
                try:
                    parsed = urlparse(url)
                    result = {"domain": parsed.netloc}
                    # Add first path segment if present
                    path_parts = parsed.path.strip("/").split("/")
                    if path_parts and path_parts[0]:
                        result["path_prefix"] = path_parts[0]
                    return result
                except Exception:
                    return {"domain": "unknown"}
            return {}

        # CLI commands: track by command name
        if tool_name in ("bash", "Bash", "cli", "shell"):
            cmd = args.get("command", "")
            if cmd:
                # Extract first word (the actual command)
                cmd_parts = cmd.split()
                if cmd_parts:
                    return {"command": cmd_parts[0]}
            return {}

        # MCP tools: track by server name
        if tool_name.startswith("mcp_") or args.get("_mcp_server"):
            server = args.get("_mcp_server", "")
            if server:
                return {"mcp_server": server}
            return {}

        # Grep/Glob: track by path
        if tool_name in ("grep", "Grep", "glob", "Glob"):
            path = args.get("path", "")
            if path:
                try:
                    return {"path_prefix": str(Path(path).parent) if Path(path).is_file() else path}
                except Exception:
                    return {}
            return {}

        # Default: no specific tracking (just tool name)
        return {}

    @staticmethod
    def _normalize_signature(params: Dict[str, str]) -> str:
        """Create stable signature from key params."""
        if not params:
            return ""
        return "|".join(f"{k}={v}" for k, v in sorted(params.items()))


@dataclass
class FailureRecord:
    """Records a single failure event."""

    # Identity
    failure_key: str                  # e.g., "readFile|path_prefix=/etc"
    tool_name: str                    # e.g., "bash", "readFile", "http_request"
    plugin_name: str                  # e.g., "cli", "file_edit", "service_connector"
    timestamp: datetime

    # Parameter context
    parameter_signature: str          # Normalized key parameters
    domain: Optional[str] = None      # For HTTP: extracted domain
    service: Optional[str] = None     # For MCP: server name
    path_prefix: Optional[str] = None # For file ops: parent directory

    # Failure details
    error_type: str = "unknown"       # Exception class name or error category
    error_message: str = ""           # Actual error message
    severity: FailureSeverity = FailureSeverity.SERVER_ERROR
    weight: float = 1.0               # Failure weight (0.5 for model errors, 1.0 default)

    # Execution context
    call_id: str = ""                 # Links to permission/ledger records
    session_id: str = ""              # Session context
    turn_index: int = 0               # Where in conversation this occurred

    # Optional enrichment
    http_status: Optional[int] = None # For HTTP failures
    retry_count: Optional[int] = None # If framework retried before failing
    was_transient: Optional[bool] = None  # Framework's transient classification

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "failure_key": self.failure_key,
            "tool_name": self.tool_name,
            "plugin_name": self.plugin_name,
            "timestamp": self.timestamp.isoformat(),
            "parameter_signature": self.parameter_signature,
            "domain": self.domain,
            "service": self.service,
            "path_prefix": self.path_prefix,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "weight": self.weight,
            "call_id": self.call_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "http_status": self.http_status,
            "retry_count": self.retry_count,
            "was_transient": self.was_transient,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureRecord":
        """Deserialize from dictionary."""
        return cls(
            failure_key=data["failure_key"],
            tool_name=data["tool_name"],
            plugin_name=data.get("plugin_name", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            parameter_signature=data.get("parameter_signature", ""),
            domain=data.get("domain"),
            service=data.get("service"),
            path_prefix=data.get("path_prefix"),
            error_type=data.get("error_type", "unknown"),
            error_message=data.get("error_message", ""),
            severity=FailureSeverity(data.get("severity", "server_error")),
            weight=data.get("weight", 1.0),
            call_id=data.get("call_id", ""),
            session_id=data.get("session_id", ""),
            turn_index=data.get("turn_index", 0),
            http_status=data.get("http_status"),
            retry_count=data.get("retry_count"),
            was_transient=data.get("was_transient"),
        )


@dataclass
class ToolReliabilityState:
    """Current reliability state for a tool+parameters combination."""

    # Identity
    failure_key: str                  # e.g., "http_request|domain=api.example.com"
    tool_name: str                    # Base tool name for display
    state: TrustState = TrustState.TRUSTED

    # Failure tracking
    failures_in_window: float = 0.0   # Effective count (with decay/weights)
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None

    # Escalation info
    escalated_at: Optional[datetime] = None
    escalation_reason: Optional[str] = None
    escalation_expires: Optional[datetime] = None

    # Recovery tracking
    recovery_started: Optional[datetime] = None
    successes_since_recovery: int = 0
    successes_needed: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "failure_key": self.failure_key,
            "tool_name": self.tool_name,
            "state": self.state.value,
            "failures_in_window": self.failures_in_window,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "escalation_reason": self.escalation_reason,
            "escalation_expires": self.escalation_expires.isoformat() if self.escalation_expires else None,
            "recovery_started": self.recovery_started.isoformat() if self.recovery_started else None,
            "successes_since_recovery": self.successes_since_recovery,
            "successes_needed": self.successes_needed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolReliabilityState":
        """Deserialize from dictionary."""
        return cls(
            failure_key=data["failure_key"],
            tool_name=data["tool_name"],
            state=TrustState(data.get("state", "trusted")),
            failures_in_window=data.get("failures_in_window", 0.0),
            consecutive_failures=data.get("consecutive_failures", 0),
            total_failures=data.get("total_failures", 0),
            total_successes=data.get("total_successes", 0),
            last_failure=datetime.fromisoformat(data["last_failure"]) if data.get("last_failure") else None,
            last_success=datetime.fromisoformat(data["last_success"]) if data.get("last_success") else None,
            escalated_at=datetime.fromisoformat(data["escalated_at"]) if data.get("escalated_at") else None,
            escalation_reason=data.get("escalation_reason"),
            escalation_expires=datetime.fromisoformat(data["escalation_expires"]) if data.get("escalation_expires") else None,
            recovery_started=datetime.fromisoformat(data["recovery_started"]) if data.get("recovery_started") else None,
            successes_since_recovery=data.get("successes_since_recovery", 0),
            successes_needed=data.get("successes_needed", 3),
        )


@dataclass
class EscalationRule:
    """Defines when a tool should be escalated to requiring approval."""

    # Threshold type (pick one or more)
    count_threshold: Optional[int] = 3        # N failures triggers escalation
    consecutive_threshold: Optional[int] = None  # N consecutive failures
    rate_threshold: Optional[float] = None    # Failure rate (0.0-1.0)

    # Time window
    window_seconds: int = 3600                # Default: 1 hour window

    # What to count
    severity_filter: Set[FailureSeverity] = field(
        default_factory=lambda: {
            FailureSeverity.SERVER_ERROR,
            FailureSeverity.CRASH,
            FailureSeverity.SECURITY,
            FailureSeverity.PERMISSION,
            FailureSeverity.TIMEOUT,
        }
    )

    # Escalation behavior
    escalation_duration_seconds: int = 1800   # How long escalation lasts (30 min)
    notify_user: bool = True                  # Show notification on escalation

    # Recovery
    cooldown_seconds: int = 900               # Time without failures to auto-recover
    success_count_to_recover: int = 3         # Successful calls needed to recover


@dataclass
class ReliabilityConfig:
    """Global reliability plugin configuration."""

    # Default rule for all tools
    default_rule: EscalationRule = field(default_factory=EscalationRule)

    # Per-plugin overrides
    plugin_rules: Dict[str, EscalationRule] = field(default_factory=dict)

    # Per-tool overrides (highest priority)
    tool_rules: Dict[str, EscalationRule] = field(default_factory=dict)

    # Per-domain rules (for HTTP/service tools)
    domain_rules: Dict[str, EscalationRule] = field(default_factory=dict)

    # Global settings
    enable_auto_recovery: bool = True
    persist_across_sessions: bool = True
    max_history_entries: int = 1000

    def get_rule_for_tool(
        self,
        tool_name: str,
        plugin_name: str = "",
        domain: Optional[str] = None
    ) -> EscalationRule:
        """Returns the most specific applicable rule."""

        # 1. Tool-specific rule (highest priority)
        if tool_name in self.tool_rules:
            return self.tool_rules[tool_name]

        # 2. Domain-specific rule (for HTTP tools)
        if domain and domain in self.domain_rules:
            return self.domain_rules[domain]

        # 3. Plugin-specific rule
        if plugin_name and plugin_name in self.plugin_rules:
            return self.plugin_rules[plugin_name]

        # 4. Default rule
        return self.default_rule


def classify_failure(tool_name: str, error: Dict[str, Any]) -> Tuple[FailureSeverity, float]:
    """Classify failure severity and weight from error details.

    Returns:
        Tuple of (severity, weight) where weight is 0.5 for model errors, 1.0 otherwise.
    """
    error_msg = str(error.get("error", "")).lower()
    http_status = error.get("http_status") or error.get("status_code")

    # HTTP status-based classification
    if http_status:
        if http_status == 401:
            return (FailureSeverity.PERMISSION, 1.0)
        elif http_status == 403:
            return (FailureSeverity.PERMISSION, 1.0)
        elif http_status == 404:
            return (FailureSeverity.NOT_FOUND, 1.0)
        elif http_status == 429:
            return (FailureSeverity.TRANSIENT, 0.5)  # Rate limits are partially expected
        elif 500 <= http_status < 600:
            return (FailureSeverity.SERVER_ERROR, 1.0)

    # Message-based classification
    if any(x in error_msg for x in ["not found", "does not exist", "no such file"]):
        return (FailureSeverity.NOT_FOUND, 1.0)
    elif any(x in error_msg for x in ["permission denied", "access denied", "unauthorized", "forbidden"]):
        return (FailureSeverity.PERMISSION, 1.0)
    elif any(x in error_msg for x in ["timeout", "timed out", "deadline exceeded"]):
        return (FailureSeverity.TIMEOUT, 1.0)
    elif any(x in error_msg for x in ["rate limit", "too many requests", "quota"]):
        return (FailureSeverity.TRANSIENT, 0.5)
    elif any(x in error_msg for x in ["invalid", "required", "must be", "expected", "missing"]):
        # Model error - lower weight
        return (FailureSeverity.INVALID_INPUT, 0.5)
    elif any(x in error_msg for x in ["crash", "segfault", "killed", "terminated"]):
        return (FailureSeverity.CRASH, 1.0)
    elif any(x in error_msg for x in ["security", "injection", "malicious"]):
        return (FailureSeverity.SECURITY, 1.0)

    # Default to server error for unknown failures
    return (FailureSeverity.SERVER_ERROR, 1.0)


# -----------------------------------------------------------------------------
# Model Reliability Types
# -----------------------------------------------------------------------------


class ModelSwitchStrategy(Enum):
    """Strategy for model switching suggestions."""

    DISABLED = "disabled"   # Never suggest model switches
    SUGGEST = "suggest"     # Notify user when switch might help
    AUTO = "auto"           # Automatically switch on threshold


@dataclass
class ModelToolProfile:
    """Tracks a model's reliability with a specific tool."""

    model_name: str
    failure_key: str
    total_attempts: int = 0
    failures: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_attempts == 0:
            return 1.0
        return 1.0 - (self.failures / self.total_attempts)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 to 1.0)."""
        if self.total_attempts == 0:
            return 0.0
        return self.failures / self.total_attempts

    def record_attempt(self, success: bool) -> None:
        """Record an attempt with this tool."""
        self.total_attempts += 1
        if success:
            self.last_success = datetime.now()
        else:
            self.failures += 1
            self.last_failure = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "failure_key": self.failure_key,
            "total_attempts": self.total_attempts,
            "failures": self.failures,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelToolProfile":
        """Deserialize from dictionary."""
        return cls(
            model_name=data["model_name"],
            failure_key=data["failure_key"],
            total_attempts=data.get("total_attempts", 0),
            failures=data.get("failures", 0),
            last_failure=datetime.fromisoformat(data["last_failure"]) if data.get("last_failure") else None,
            last_success=datetime.fromisoformat(data["last_success"]) if data.get("last_success") else None,
        )


@dataclass
class ModelSwitchSuggestion:
    """Suggestion to switch models for better reliability."""

    current_model: str
    suggested_model: str
    failure_key: str
    tool_name: str

    # Comparison data
    current_success_rate: float
    suggested_success_rate: float
    improvement: float  # Absolute improvement (0.0 to 1.0)

    # Context
    reason: str
    confidence: str  # "low", "medium", "high" based on sample size

    def to_display_lines(self) -> List[str]:
        """Format suggestion for display."""
        lines = [
            f"Model switch suggested for '{self.tool_name}':",
            f"  Current: {self.current_model} ({self.current_success_rate:.0%} success)",
            f"  Better:  {self.suggested_model} ({self.suggested_success_rate:.0%} success)",
            f"  Improvement: +{self.improvement:.0%}",
            f"  Confidence: {self.confidence}",
        ]
        if self.reason:
            lines.append(f"  Reason: {self.reason}")
        return lines


@dataclass
class ModelSwitchConfig:
    """Configuration for model switching behavior."""

    strategy: ModelSwitchStrategy = ModelSwitchStrategy.SUGGEST
    failure_threshold: int = 3              # Failures before suggesting switch
    min_success_rate_diff: float = 0.3      # Min improvement to suggest (30%)
    min_attempts: int = 3                   # Min attempts before comparing
    preferred_models: List[str] = field(default_factory=list)  # Priority order


# -----------------------------------------------------------------------------
# Behavioral Pattern Detection Types
# -----------------------------------------------------------------------------


class BehavioralPatternType(Enum):
    """Types of behavioral patterns to detect."""

    # Repetitive patterns
    REPETITIVE_CALLS = "repetitive_calls"       # Same tool called N times with similar args
    INTROSPECTION_LOOP = "introspection_loop"   # Stuck calling list_tools, get_schema, etc.

    # Progress stalls
    ANNOUNCE_NO_ACTION = "announce_no_action"   # Model says "proceeding" but only reads
    READ_ONLY_LOOP = "read_only_loop"           # Only calling read tools, avoiding writes
    PLANNING_LOOP = "planning_loop"             # Infinite planning without execution

    # Avoidance patterns
    TOOL_AVOIDANCE = "tool_avoidance"           # Model avoids a specific tool repeatedly
    ERROR_RETRY_LOOP = "error_retry_loop"       # Retrying same failing operation unchanged


class PatternSeverity(Enum):
    """Severity levels for detected patterns."""

    MINOR = "minor"           # 2-3 repetitions, just starting
    MODERATE = "moderate"     # 4-5 repetitions, clear stall
    SEVERE = "severe"         # 6+ repetitions, intervention needed


@dataclass
class ToolCall:
    """Records a tool call for pattern detection."""

    tool_name: str
    args: Dict[str, Any]
    timestamp: datetime
    success: Optional[bool] = None  # Set after execution


@dataclass
class BehavioralPattern:
    """Records a detected behavioral pattern."""

    pattern_type: BehavioralPatternType
    detected_at: datetime
    turn_index: int
    session_id: str

    # Pattern specifics
    tool_sequence: List[str]              # Recent tool calls leading to detection
    repetition_count: int                 # How many times pattern repeated
    duration_seconds: float               # How long pattern has persisted

    # Context
    model_name: str
    last_model_text: Optional[str] = None        # What the model said (e.g., "Proceeding now")
    expected_action: Optional[str] = None        # What tool should have been called

    # Severity
    severity: PatternSeverity = PatternSeverity.MINOR

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "detected_at": self.detected_at.isoformat(),
            "turn_index": self.turn_index,
            "session_id": self.session_id,
            "tool_sequence": self.tool_sequence,
            "repetition_count": self.repetition_count,
            "duration_seconds": self.duration_seconds,
            "model_name": self.model_name,
            "last_model_text": self.last_model_text,
            "expected_action": self.expected_action,
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehavioralPattern":
        """Deserialize from dictionary."""
        return cls(
            pattern_type=BehavioralPatternType(data["pattern_type"]),
            detected_at=datetime.fromisoformat(data["detected_at"]),
            turn_index=data["turn_index"],
            session_id=data["session_id"],
            tool_sequence=data["tool_sequence"],
            repetition_count=data["repetition_count"],
            duration_seconds=data["duration_seconds"],
            model_name=data["model_name"],
            last_model_text=data.get("last_model_text"),
            expected_action=data.get("expected_action"),
            severity=PatternSeverity(data.get("severity", "minor")),
        )

    def to_display_lines(self) -> List[str]:
        """Format pattern for display."""
        severity_icons = {
            PatternSeverity.MINOR: "⚡",
            PatternSeverity.MODERATE: "⚠",
            PatternSeverity.SEVERE: "🚨",
        }
        icon = severity_icons.get(self.severity, "")

        lines = [
            f"{icon} {self.pattern_type.value.replace('_', ' ').title()}",
            f"  Severity: {self.severity.value}",
            f"  Repetitions: {self.repetition_count}",
        ]

        if self.tool_sequence:
            recent = self.tool_sequence[-5:]  # Last 5 tools
            lines.append(f"  Recent tools: {' → '.join(recent)}")

        if self.last_model_text:
            truncated = self.last_model_text[:100]
            if len(self.last_model_text) > 100:
                truncated += "..."
            lines.append(f"  Model said: \"{truncated}\"")

        if self.expected_action:
            lines.append(f"  Expected: {self.expected_action}")

        return lines


@dataclass
class PatternDetectionConfig:
    """Configuration for behavioral pattern detection."""

    enabled: bool = True

    # Repetition thresholds
    repetitive_call_threshold: int = 3        # Same tool N times triggers detection
    introspection_tool_names: Set[str] = field(
        default_factory=lambda: {"list_tools", "get_tool_schemas", "askPermission"}
    )
    introspection_loop_threshold: int = 2     # N introspection calls without action

    # Progress tracking
    read_only_tools: Set[str] = field(
        default_factory=lambda: {"readFile", "list_tools", "get_tool_schemas", "glob", "grep", "Read", "Glob", "Grep"}
    )
    action_tools: Set[str] = field(
        default_factory=lambda: {"writeFile", "updateFile", "bash", "removeFile", "Write", "Edit", "Bash"}
    )
    max_reads_before_action: int = 5          # N reads without action = stall

    # Time-based
    max_turn_duration_seconds: float = 120.0  # Turn taking too long = possible stall

    # Announce detection (requires text analysis)
    announce_phrases: List[str] = field(
        default_factory=lambda: [
            "proceeding now",
            "let me",
            "i'll now",
            "i will now",
            "executing",
            "running",
            "starting",
            "making the change",
        ]
    )


# -----------------------------------------------------------------------------
# Nudge Injection Types
# -----------------------------------------------------------------------------


class NudgeType(Enum):
    """Types of nudges to inject into the model's context."""

    GENTLE_REMINDER = "gentle"      # Soft suggestion, low urgency
    DIRECT_INSTRUCTION = "direct"   # Clear instruction, moderate urgency
    INTERRUPT = "interrupt"         # Stop and require user input


class NudgeLevel(Enum):
    """User-configurable nudge intensity levels."""

    OFF = "off"           # No nudges
    GENTLE = "gentle"     # Only gentle reminders
    DIRECT = "direct"     # Gentle + direct instructions
    FULL = "full"         # All nudges including interrupts


@dataclass
class Nudge:
    """A guidance injection for the model."""

    nudge_type: NudgeType
    message: str
    pattern: BehavioralPattern
    injected_at: datetime

    # Tracking
    acknowledged: bool = False
    effective: bool = False  # Did the nudge resolve the pattern?

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nudge_type": self.nudge_type.value,
            "message": self.message,
            "pattern": self.pattern.to_dict(),
            "injected_at": self.injected_at.isoformat(),
            "acknowledged": self.acknowledged,
            "effective": self.effective,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Nudge":
        """Deserialize from dictionary."""
        return cls(
            nudge_type=NudgeType(data["nudge_type"]),
            message=data["message"],
            pattern=BehavioralPattern.from_dict(data["pattern"]),
            injected_at=datetime.fromisoformat(data["injected_at"]),
            acknowledged=data.get("acknowledged", False),
            effective=data.get("effective", False),
        )

    def to_system_message(self) -> str:
        """Format nudge as a system message for injection."""
        if self.nudge_type == NudgeType.INTERRUPT:
            return f"[SYSTEM INTERRUPT] {self.message}"
        elif self.nudge_type == NudgeType.DIRECT_INSTRUCTION:
            return f"[NOTICE] {self.message}"
        else:
            return f"[Reminder] {self.message}"


@dataclass
class NudgeConfig:
    """Configuration for nudge injection behavior."""

    level: NudgeLevel = NudgeLevel.DIRECT  # Default: gentle + direct, no interrupts
    enabled: bool = True

    # Cooldown to avoid spamming nudges
    cooldown_seconds: float = 30.0  # Min time between nudges for same pattern type

    # Auto-escalation
    escalate_on_ignore: bool = True  # Escalate severity if nudge is ignored
    escalation_threshold: int = 2    # How many ignored nudges before escalation


# -----------------------------------------------------------------------------
# Model Behavioral Profile Types
# -----------------------------------------------------------------------------


@dataclass
class ModelBehavioralProfile:
    """Tracks a model's behavioral patterns and nudge responsiveness.

    This profile combines pattern detection data with nudge effectiveness
    to create a complete picture of how a model behaves and responds to
    guidance. Used for:
    - Comparing model behavior across different tools/tasks
    - Optimizing nudge strategies per model
    - Informing model switching decisions
    """

    model_name: str

    # Pattern tracking
    pattern_counts: Dict[BehavioralPatternType, int] = field(default_factory=dict)
    pattern_severities: Dict[BehavioralPatternType, List[PatternSeverity]] = field(default_factory=dict)

    # Nudge effectiveness
    nudges_sent: int = 0
    nudges_acknowledged: int = 0
    nudges_effective: int = 0  # Pattern stopped after nudge

    # Turn tracking for stall rate
    total_turns: int = 0
    stalled_turns: int = 0  # Turns with detected patterns
    turn_start_time: Optional[datetime] = None

    # First/last activity
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    @property
    def stall_rate(self) -> float:
        """Calculate what fraction of turns have stalled (0.0 to 1.0)."""
        if self.total_turns == 0:
            return 0.0
        return self.stalled_turns / self.total_turns

    @property
    def nudge_effectiveness(self) -> float:
        """Calculate how often nudges resolve patterns (0.0 to 1.0).

        Returns 1.0 if no nudges have been sent (assume effective until proven).
        """
        if self.nudges_sent == 0:
            return 1.0
        return self.nudges_effective / self.nudges_sent

    @property
    def nudge_acknowledgment_rate(self) -> float:
        """Calculate how often the model acknowledges nudges (0.0 to 1.0)."""
        if self.nudges_sent == 0:
            return 1.0
        return self.nudges_acknowledged / self.nudges_sent

    @property
    def total_patterns(self) -> int:
        """Total number of patterns detected for this model."""
        return sum(self.pattern_counts.values())

    def most_common_pattern(self) -> Optional[BehavioralPatternType]:
        """Return the most frequently occurring pattern type."""
        if not self.pattern_counts:
            return None
        return max(self.pattern_counts.items(), key=lambda x: x[1])[0]

    def average_severity(self, pattern_type: BehavioralPatternType) -> Optional[float]:
        """Calculate average severity for a pattern type.

        Returns None if no patterns of this type recorded.
        Severity values: MINOR=1, MODERATE=2, SEVERE=3
        """
        severities = self.pattern_severities.get(pattern_type, [])
        if not severities:
            return None

        severity_values = {
            PatternSeverity.MINOR: 1,
            PatternSeverity.MODERATE: 2,
            PatternSeverity.SEVERE: 3,
        }
        total = sum(severity_values.get(s, 1) for s in severities)
        return total / len(severities)

    def record_pattern(self, pattern: BehavioralPattern) -> None:
        """Record a detected pattern for this model."""
        ptype = pattern.pattern_type

        # Update count
        self.pattern_counts[ptype] = self.pattern_counts.get(ptype, 0) + 1

        # Track severity
        if ptype not in self.pattern_severities:
            self.pattern_severities[ptype] = []
        self.pattern_severities[ptype].append(pattern.severity)

        # Keep severity list bounded (last 100)
        if len(self.pattern_severities[ptype]) > 100:
            self.pattern_severities[ptype] = self.pattern_severities[ptype][-100:]

        # Update timestamps
        now = datetime.now()
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now

    def record_nudge_sent(self) -> None:
        """Record that a nudge was sent to this model."""
        self.nudges_sent += 1
        now = datetime.now()
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now

    def record_nudge_acknowledged(self) -> None:
        """Record that the model acknowledged a nudge."""
        self.nudges_acknowledged += 1

    def record_nudge_effective(self) -> None:
        """Record that a nudge was effective (pattern stopped)."""
        self.nudges_effective += 1
        self.nudges_acknowledged += 1  # Effective implies acknowledged

    def record_turn_start(self) -> None:
        """Record the start of a new turn."""
        self.total_turns += 1
        self.turn_start_time = datetime.now()
        now = datetime.now()
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now

    def record_turn_stalled(self) -> None:
        """Record that the current turn has stalled."""
        self.stalled_turns += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "pattern_counts": {k.value: v for k, v in self.pattern_counts.items()},
            "pattern_severities": {
                k.value: [s.value for s in v]
                for k, v in self.pattern_severities.items()
            },
            "nudges_sent": self.nudges_sent,
            "nudges_acknowledged": self.nudges_acknowledged,
            "nudges_effective": self.nudges_effective,
            "total_turns": self.total_turns,
            "stalled_turns": self.stalled_turns,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelBehavioralProfile":
        """Deserialize from dictionary."""
        profile = cls(model_name=data["model_name"])

        # Restore pattern counts
        for ptype_str, count in data.get("pattern_counts", {}).items():
            try:
                ptype = BehavioralPatternType(ptype_str)
                profile.pattern_counts[ptype] = count
            except ValueError:
                pass  # Skip unknown pattern types

        # Restore severity history
        for ptype_str, severities in data.get("pattern_severities", {}).items():
            try:
                ptype = BehavioralPatternType(ptype_str)
                profile.pattern_severities[ptype] = [
                    PatternSeverity(s) for s in severities
                ]
            except ValueError:
                pass  # Skip unknown types

        profile.nudges_sent = data.get("nudges_sent", 0)
        profile.nudges_acknowledged = data.get("nudges_acknowledged", 0)
        profile.nudges_effective = data.get("nudges_effective", 0)
        profile.total_turns = data.get("total_turns", 0)
        profile.stalled_turns = data.get("stalled_turns", 0)

        if data.get("first_seen"):
            profile.first_seen = datetime.fromisoformat(data["first_seen"])
        if data.get("last_seen"):
            profile.last_seen = datetime.fromisoformat(data["last_seen"])

        return profile

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for display."""
        return {
            "model": self.model_name,
            "total_turns": self.total_turns,
            "stalled_turns": self.stalled_turns,
            "stall_rate": self.stall_rate,
            "total_patterns": self.total_patterns,
            "most_common_pattern": self.most_common_pattern().value if self.most_common_pattern() else None,
            "nudges_sent": self.nudges_sent,
            "nudges_effective": self.nudges_effective,
            "nudge_effectiveness": self.nudge_effectiveness,
            "pattern_breakdown": {k.value: v for k, v in self.pattern_counts.items()},
        }

    def compare_to(self, other: "ModelBehavioralProfile") -> Dict[str, Any]:
        """Compare this profile to another model's profile.

        Returns dict with comparison metrics useful for model switching decisions.
        """
        return {
            "this_model": self.model_name,
            "other_model": other.model_name,
            "stall_rate_diff": self.stall_rate - other.stall_rate,  # Negative = this is better
            "nudge_effectiveness_diff": self.nudge_effectiveness - other.nudge_effectiveness,  # Positive = this is better
            "total_patterns_diff": self.total_patterns - other.total_patterns,  # Negative = this is better
            "this_better_stall": self.stall_rate < other.stall_rate,
            "this_better_nudge": self.nudge_effectiveness > other.nudge_effectiveness,
            "recommendation": self._get_recommendation(other),
        }

    def _get_recommendation(self, other: "ModelBehavioralProfile") -> str:
        """Generate recommendation based on comparison."""
        this_score = 0
        other_score = 0

        # Lower stall rate is better
        if self.stall_rate < other.stall_rate:
            this_score += 1
        elif other.stall_rate < self.stall_rate:
            other_score += 1

        # Higher nudge effectiveness is better
        if self.nudge_effectiveness > other.nudge_effectiveness:
            this_score += 1
        elif other.nudge_effectiveness > self.nudge_effectiveness:
            other_score += 1

        # Fewer total patterns is better
        if self.total_patterns < other.total_patterns:
            this_score += 1
        elif other.total_patterns < self.total_patterns:
            other_score += 1

        if this_score > other_score:
            return f"{self.model_name} appears better behaved"
        elif other_score > this_score:
            return f"{other.model_name} appears better behaved"
        else:
            return "Similar behavioral profiles"
