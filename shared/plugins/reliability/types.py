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
