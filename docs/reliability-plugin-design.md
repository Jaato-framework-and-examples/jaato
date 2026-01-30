# Reliability Plugin Design Document

## Overview

The **reliability plugin** is a cross-cutting concern that tracks failures from tools/plugins and dynamically adjusts permission requirements based on failure history. The core principle is **adaptive trust**: tools that work reliably maintain their approval status, while unreliable tools get escalated back to requiring explicit user approval.

## Design Goals

1. **Non-invasive integration** - Works alongside existing permission system without modifying it
2. **Configurable thresholds** - Different failure tolerances for different contexts
3. **Transparent operation** - Clear user notifications when escalation occurs
4. **Recovery paths** - Tools can "earn back" trust after a cooldown period
5. **Rich context** - Capture enough information to make intelligent escalation decisions
6. **Adaptive model selection** - Enable model switching when current model consistently fails at specific tools

---

## Architecture

### Plugin Type

The reliability plugin is a **hybrid plugin**:
- **Permission middleware** - Intercepts permission decisions to inject reliability-based overrides
- **Tool plugin** - Exposes tools for querying/managing reliability state
- **Event subscriber** - Listens to tool execution results across the system

### Integration Points

```
                    ┌─────────────────────────────────────────────┐
                    │              JaatoSession                   │
                    │                                             │
   User Request ───►│  ┌───────────────────┐                     │
                    │  │  ToolExecutor     │                     │
                    │  │                   │                     │
                    │  │  ┌─────────────┐  │    ┌──────────────┐ │
                    │  │  │ Permission  │◄─┼────│  Reliability │ │
                    │  │  │   Plugin    │  │    │    Plugin    │ │
                    │  │  └──────┬──────┘  │    └──────┬───────┘ │
                    │  │         │         │           │         │
                    │  │         ▼         │           │         │
                    │  │    execute()──────┼───────────┘         │
                    │  │         │         │     (result hook)   │
                    │  └─────────┼─────────┘                     │
                    │            ▼                               │
                    │       Tool Result                          │
                    └─────────────────────────────────────────────┘
```

---

## Failure Recording

### What to Capture

```python
@dataclass
class FailureRecord:
    # Identity - keyed by tool + parameters (see Design Decisions #2)
    failure_key: str                  # e.g., "readFile|path_prefix=/etc"
    tool_name: str                    # e.g., "bash", "readFile", "http_request"
    plugin_name: str                  # e.g., "cli", "file_edit", "service_connector"
    timestamp: datetime

    # Parameter context (extracted for tracking granularity)
    parameter_signature: str          # Normalized key parameters
    domain: Optional[str]             # For HTTP: extracted domain
    service: Optional[str]            # For MCP: server name
    path_prefix: Optional[str]        # For file ops: parent directory

    # Failure details
    error_type: str                   # Exception class name or error category
    error_message: str                # Actual error message
    severity: FailureSeverity         # See below
    weight: float = 1.0               # Failure weight (0.5 for model errors, 1.0 default)

    # Execution context
    call_id: str                      # Links to permission/ledger records
    session_id: str                   # Session context
    turn_index: int                   # Where in conversation this occurred

    # Optional enrichment
    http_status: Optional[int]        # For HTTP failures
    retry_count: Optional[int]        # If framework retried before failing
    was_transient: Optional[bool]     # Framework's transient classification
```

### Failure Severity Classification

```python
class FailureSeverity(Enum):
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
```

### Severity Classification Rules

```python
def classify_failure(tool_name: str, error: dict) -> FailureSeverity:
    error_msg = error.get("error", "").lower()
    http_status = error.get("http_status")

    # HTTP status-based classification
    if http_status:
        if http_status == 401:
            return FailureSeverity.PERMISSION
        elif http_status == 403:
            return FailureSeverity.PERMISSION
        elif http_status == 404:
            return FailureSeverity.NOT_FOUND
        elif http_status == 429:
            return FailureSeverity.TRANSIENT
        elif 500 <= http_status < 600:
            return FailureSeverity.SERVER_ERROR

    # Message-based classification
    if any(x in error_msg for x in ["not found", "does not exist", "no such file"]):
        return FailureSeverity.NOT_FOUND
    elif any(x in error_msg for x in ["permission denied", "access denied", "unauthorized"]):
        return FailureSeverity.PERMISSION
    elif any(x in error_msg for x in ["timeout", "timed out", "deadline exceeded"]):
        return FailureSeverity.TIMEOUT
    elif any(x in error_msg for x in ["rate limit", "too many requests", "quota"]):
        return FailureSeverity.TRANSIENT
    elif any(x in error_msg for x in ["invalid", "required", "must be", "expected"]):
        return FailureSeverity.INVALID_INPUT

    # Default to server error for unknown failures
    return FailureSeverity.SERVER_ERROR
```

---

## Failure Thresholds & Escalation Rules

### Threshold Configuration

```python
@dataclass
class EscalationRule:
    """Defines when a tool should be escalated to requiring approval."""

    # Threshold type (pick one)
    count_threshold: Optional[int] = None       # N failures triggers escalation
    consecutive_threshold: Optional[int] = None  # N consecutive failures
    rate_threshold: Optional[float] = None       # Failure rate (0.0-1.0)

    # Time window
    window_seconds: int = 3600                   # Default: 1 hour window

    # What to count
    severity_filter: Set[FailureSeverity] = field(
        default_factory=lambda: {
            FailureSeverity.SERVER_ERROR,
            FailureSeverity.CRASH,
            FailureSeverity.SECURITY,
        }
    )

    # Escalation behavior
    escalation_duration_seconds: int = 1800     # How long escalation lasts (30 min)
    notify_user: bool = True                    # Show notification on escalation

    # Recovery
    cooldown_seconds: int = 900                 # Time without failures to auto-recover
    success_count_to_recover: int = 3           # Successful calls needed to recover


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
```

### Example Configurations

```python
# Conservative: Escalate quickly, recover slowly
conservative_rule = EscalationRule(
    count_threshold=2,
    window_seconds=1800,       # 30 min
    severity_filter={FailureSeverity.SERVER_ERROR, FailureSeverity.CRASH},
    escalation_duration_seconds=3600,  # 1 hour
    cooldown_seconds=1800,
    success_count_to_recover=5,
)

# Lenient: More tolerance for transient issues
lenient_rule = EscalationRule(
    count_threshold=5,
    window_seconds=3600,       # 1 hour
    severity_filter={FailureSeverity.CRASH, FailureSeverity.SECURITY},
    escalation_duration_seconds=900,   # 15 min
    cooldown_seconds=600,
    success_count_to_recover=2,
)

# Rate-based: Focus on success rate
rate_based_rule = EscalationRule(
    rate_threshold=0.5,        # Escalate if >50% failure rate
    window_seconds=600,        # Over 10 min
    escalation_duration_seconds=1200,
    success_count_to_recover=3,
)
```

### Hierarchical Rule Resolution

```python
def get_rule_for_tool(self, tool_name: str, plugin_name: str, domain: Optional[str]) -> EscalationRule:
    """Returns the most specific applicable rule."""

    # 1. Tool-specific rule (highest priority)
    if tool_name in self.config.tool_rules:
        return self.config.tool_rules[tool_name]

    # 2. Domain-specific rule (for HTTP tools)
    if domain and domain in self.config.domain_rules:
        return self.config.domain_rules[domain]

    # 3. Plugin-specific rule
    if plugin_name in self.config.plugin_rules:
        return self.config.plugin_rules[plugin_name]

    # 4. Default rule
    return self.config.default_rule
```

---

## Permission System Integration

### Option A: Permission Plugin Wrapper (Recommended)

Wraps the existing permission plugin to inject reliability checks:

```python
class ReliabilityPermissionWrapper:
    """Wraps permission plugin to add reliability-based escalation."""

    def __init__(self, inner: PermissionPlugin, reliability: ReliabilityPlugin):
        self._inner = inner
        self._reliability = reliability

    def check_permission(
        self,
        tool_name: str,
        args: dict,
        context: Optional[dict] = None,
        call_id: Optional[str] = None,
    ) -> Tuple[bool, dict]:
        """Check permission with reliability overlay."""

        # First: check if tool is escalated due to reliability
        escalation = self._reliability.check_escalation(tool_name, args)

        if escalation.is_escalated:
            # Force approval even if whitelisted
            return self._force_approval_check(
                tool_name, args, context, call_id, escalation
            )

        # Otherwise: delegate to inner permission plugin
        return self._inner.check_permission(tool_name, args, context, call_id)

    def _force_approval_check(self, tool_name, args, context, call_id, escalation):
        """Override whitelist to require explicit approval."""

        # Modify context to include escalation reason
        enhanced_context = {
            **(context or {}),
            "_reliability_escalation": {
                "reason": escalation.reason,
                "failure_count": escalation.failure_count,
                "window": escalation.window_description,
                "recovery_hint": escalation.recovery_hint,
            }
        }

        # Call inner with modified policy that forces "ask"
        # (Implementation depends on permission plugin internals)
        ...
```

### Option B: Permission Hooks

Use existing permission hooks plus execution result tracking:

```python
class ReliabilityPlugin(ToolPlugin):
    def set_permission_hooks(self):
        """Called by permission plugin for event subscription."""
        return {
            "on_resolved": self._on_permission_resolved,
        }

    def _on_permission_resolved(
        self,
        tool_name: str,
        request_id: str,
        granted: bool,
        method: str,
    ):
        """Track permission decisions for correlation with failures."""
        self._pending_executions[request_id] = {
            "tool_name": tool_name,
            "granted": granted,
            "timestamp": datetime.now(),
        }
```

### Option C: Session Event Subscription

Subscribe to session events for execution results:

```python
def set_session(self, session: JaatoSession):
    """Auto-wired by session during configure()."""
    self._session = session

    # Subscribe to tool execution events
    session.subscribe_to_tool_results(self._on_tool_result)

def _on_tool_result(
    self,
    tool_name: str,
    args: dict,
    success: bool,
    result: Any,
    call_id: str,
):
    """Process tool execution result."""
    if not success or self._is_error_result(result):
        self._record_failure(tool_name, args, result, call_id)
    else:
        self._record_success(tool_name, call_id)
```

---

## Cross-Plugin Failure Reporting API

### Explicit Reporting Interface

Plugins can explicitly report failures with rich context:

```python
class ReliabilityReporter(Protocol):
    """Interface for plugins to report failures."""

    def report_failure(
        self,
        tool_name: str,
        error: Union[str, dict, Exception],
        *,
        severity: Optional[FailureSeverity] = None,
        domain: Optional[str] = None,
        service: Optional[str] = None,
        context: Optional[dict] = None,
        recoverable: bool = True,
    ) -> None:
        """Report a tool execution failure.

        Args:
            tool_name: Name of the failing tool
            error: Error details (string, dict with "error" key, or exception)
            severity: Override automatic severity classification
            domain: Domain for HTTP requests (extracted automatically if not provided)
            service: Service name for MCP/external services
            context: Additional context for analysis
            recoverable: Whether this failure type can recover automatically
        """
        ...

    def report_success(
        self,
        tool_name: str,
        *,
        domain: Optional[str] = None,
        service: Optional[str] = None,
    ) -> None:
        """Report successful execution (for recovery tracking)."""
        ...
```

### Auto-Detection from Tool Results

Analyze tool results automatically:

```python
def _analyze_tool_result(self, tool_name: str, result: Any) -> Optional[FailureRecord]:
    """Detect failures from tool result structure."""

    if not isinstance(result, dict):
        return None  # Not a failure

    # Standard error format
    if "error" in result:
        return self._create_failure_record(
            tool_name=tool_name,
            error_type=result.get("error_type", "unknown"),
            error_message=result["error"],
            http_status=result.get("http_status"),
            traceback=result.get("traceback"),
        )

    # HTTP response failures
    if "status_code" in result and result["status_code"] >= 400:
        return self._create_failure_record(
            tool_name=tool_name,
            error_type="http_error",
            error_message=result.get("body", ""),
            http_status=result["status_code"],
        )

    return None  # Success
```

### Event Bus Pattern

Alternative: plugins emit failure events to central bus:

```python
# In plugin registry or session
class FailureEventBus:
    def __init__(self):
        self._listeners: List[Callable] = []

    def subscribe(self, listener: Callable[[FailureEvent], None]):
        self._listeners.append(listener)

    def emit(self, event: FailureEvent):
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors break the bus

# Usage in plugins
def _execute_tool(self, args):
    try:
        result = self._do_work(args)
        return result
    except Exception as e:
        # Emit failure event
        self._failure_bus.emit(FailureEvent(
            tool_name=self.name,
            error=e,
            timestamp=datetime.now(),
        ))
        raise
```

---

## Escalation & Recovery Mechanics

### Escalation State Machine

```
                   ┌──────────────┐
                   │   TRUSTED    │◄─────────────────┐
                   │              │                  │
                   └──────┬───────┘                  │
                          │                          │
                  threshold exceeded                 │
                          │                          │
                          ▼                          │
                   ┌──────────────┐          ┌──────┴───────┐
                   │  ESCALATED   │─────────►│  RECOVERING  │
                   │              │ cooldown │              │
                   └──────┬───────┘ started  └──────┬───────┘
                          │                         │
                     more failures             N successes
                          │                         │
                          ▼                         │
                   ┌──────────────┐                 │
                   │   BLOCKED    │                 │
                   │ (permanent)  │                 │
                   └──────────────┘                 │
                                                   │
                   ◄───────────────────────────────┘
```

### State Definitions

```python
class TrustState(Enum):
    TRUSTED = "trusted"         # Normal operation, uses standard permissions
    ESCALATED = "escalated"     # Requires explicit approval regardless of whitelist
    RECOVERING = "recovering"   # In cooldown, tracking successes for recovery
    BLOCKED = "blocked"         # Permanent block (critical failures)


@dataclass
class ToolReliabilityState:
    """Current reliability state for a tool+parameters combination."""

    # Identity - keyed by tool + parameters (see Design Decisions #2)
    failure_key: str                  # e.g., "http_request|domain=api.example.com"
    tool_name: str                    # Base tool name for display
    state: TrustState

    # Failure tracking
    failures_in_window: float         # Effective count (with decay/weights)
    consecutive_failures: int
    last_failure: Optional[datetime]
    last_success: Optional[datetime]

    # Escalation info
    escalated_at: Optional[datetime]
    escalation_reason: Optional[str]
    escalation_expires: Optional[datetime]

    # Recovery tracking
    recovery_started: Optional[datetime]
    successes_since_recovery: int
    successes_needed: int
```

### Escalation Logic

```python
def check_and_update_state(
    self,
    tool_name: str,
    execution_success: bool,
    failure_record: Optional[FailureRecord] = None,
) -> ToolReliabilityState:
    """Update state after tool execution."""

    state = self._get_or_create_state(tool_name)
    rule = self._get_rule_for_tool(tool_name)

    if execution_success:
        return self._handle_success(state, rule)
    else:
        return self._handle_failure(state, rule, failure_record)

def _handle_failure(
    self,
    state: ToolReliabilityState,
    rule: EscalationRule,
    failure: FailureRecord,
) -> ToolReliabilityState:
    """Process a failure and potentially escalate."""

    # Update counters
    state.consecutive_failures += 1
    state.last_failure = datetime.now()

    # Count failures in window (only matching severity)
    if failure.severity in rule.severity_filter:
        state.failures_in_window = self._count_failures_in_window(
            state.tool_name, rule.window_seconds, rule.severity_filter
        )

    # Check escalation conditions
    should_escalate = False
    reason = ""

    if rule.count_threshold and state.failures_in_window >= rule.count_threshold:
        should_escalate = True
        reason = f"{state.failures_in_window} failures in {rule.window_seconds}s"

    elif rule.consecutive_threshold and state.consecutive_failures >= rule.consecutive_threshold:
        should_escalate = True
        reason = f"{state.consecutive_failures} consecutive failures"

    elif rule.rate_threshold:
        rate = self._calculate_failure_rate(state.tool_name, rule.window_seconds)
        if rate >= rule.rate_threshold:
            should_escalate = True
            reason = f"{rate:.0%} failure rate"

    # Critical severity always escalates
    if failure.severity == FailureSeverity.SECURITY:
        should_escalate = True
        state.state = TrustState.BLOCKED
        reason = "Security concern detected"

    if should_escalate and state.state == TrustState.TRUSTED:
        state.state = TrustState.ESCALATED
        state.escalated_at = datetime.now()
        state.escalation_reason = reason
        state.escalation_expires = datetime.now() + timedelta(
            seconds=rule.escalation_duration_seconds
        )

        if rule.notify_user:
            self._notify_escalation(state)

    return state

def _handle_success(
    self,
    state: ToolReliabilityState,
    rule: EscalationRule,
) -> ToolReliabilityState:
    """Process a success, potentially recovering trust."""

    state.consecutive_failures = 0
    state.last_success = datetime.now()

    if state.state == TrustState.ESCALATED:
        # Start recovery if cooldown period passed
        if state.escalation_expires and datetime.now() >= state.escalation_expires:
            state.state = TrustState.RECOVERING
            state.recovery_started = datetime.now()
            state.successes_since_recovery = 1
            state.successes_needed = rule.success_count_to_recover

    elif state.state == TrustState.RECOVERING:
        state.successes_since_recovery += 1

        if state.successes_since_recovery >= state.successes_needed:
            # Full recovery
            state.state = TrustState.TRUSTED
            state.escalated_at = None
            state.escalation_reason = None
            self._notify_recovery(state)

    return state
```

---

## User Notifications

### Escalation Notification

```python
def _notify_escalation(self, state: ToolReliabilityState):
    """Notify user when a tool is escalated."""

    message = (
        f"⚠️ Tool '{state.tool_name}' escalated to require approval\n"
        f"   Reason: {state.escalation_reason}\n"
        f"   Duration: until {state.escalation_expires.strftime('%H:%M:%S')}\n"
        f"   Recovery: {state.successes_needed} successful calls after cooldown"
    )

    self._output_callback("reliability", message, "write")
```

### Permission Request Enhancement

When an escalated tool requests permission, include context:

```python
def format_permission_request(
    self,
    tool_name: str,
    args: dict,
    context: dict,
) -> PermissionDisplayInfo:
    """Add reliability context to permission prompt."""

    escalation_info = context.get("_reliability_escalation")
    if not escalation_info:
        return None  # No escalation, use default formatting

    return PermissionDisplayInfo(
        summary=f"{tool_name} (escalated - requires approval)",
        details=self._format_escalation_details(escalation_info),
        format_hint="text",
        warnings=(
            f"This tool has been escalated due to: {escalation_info['reason']}\n"
            f"Recent failures: {escalation_info['failure_count']} in {escalation_info['window']}\n"
            f"{escalation_info['recovery_hint']}"
        ),
        warning_level="warning",
    )
```

---

## Persistence

### Storage Format

```python
@dataclass
class ReliabilityPersistence:
    """Persisted reliability data."""

    version: int = 1

    # Tool states (only non-trusted states need persistence)
    tool_states: Dict[str, ToolReliabilityState] = field(default_factory=dict)

    # Recent failure history (for threshold calculations)
    failure_history: List[FailureRecord] = field(default_factory=list)

    # Domain-level aggregations
    domain_stats: Dict[str, DomainStats] = field(default_factory=dict)

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    created: datetime = field(default_factory=datetime.now)
```

### Storage Location

```python
def _get_persistence_path(self) -> Path:
    """Get path for reliability data file."""

    # Project-level (preferred)
    if self._workspace_path:
        project_path = self._workspace_path / ".jaato" / "reliability.json"
        if project_path.parent.exists():
            return project_path

    # User-level fallback
    user_path = Path.home() / ".jaato" / "reliability.json"
    user_path.parent.mkdir(exist_ok=True)
    return user_path
```

### Persistence Strategy

```python
class ReliabilityPersistence:
    def save(self):
        """Save state to disk."""
        # Prune old history
        self._prune_history()

        # Serialize
        data = {
            "version": self.version,
            "tool_states": {
                name: asdict(state)
                for name, state in self.tool_states.items()
                if state.state != TrustState.TRUSTED
            },
            "failure_history": [
                asdict(f) for f in self.failure_history[-self.max_history:]
            ],
            "last_updated": datetime.now().isoformat(),
        }

        # Atomic write
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, default=str))
        tmp_path.rename(self._path)

    def load(self) -> bool:
        """Load state from disk. Returns True if loaded successfully."""
        if not self._path.exists():
            return False

        try:
            data = json.loads(self._path.read_text())
            # Version migration if needed
            if data.get("version", 1) < self.version:
                data = self._migrate(data)

            # Deserialize
            self.tool_states = {
                name: ToolReliabilityState(**state)
                for name, state in data.get("tool_states", {}).items()
            }
            self.failure_history = [
                FailureRecord(**f) for f in data.get("failure_history", [])
            ]
            return True
        except Exception as e:
            logger.warning(f"Failed to load reliability data: {e}")
            return False
```

---

## Exposed Tools

### Query Reliability Status

```python
def get_tool_schemas(self) -> List[ToolSchema]:
    return [
        ToolSchema(
            name="reliability_status",
            description="Check the reliability status of tools",
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
        ToolSchema(
            name="reliability_reset",
            description="Reset reliability state for a tool (requires user approval)",
            parameters={
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Tool to reset"
                    }
                },
                "required": ["tool_name"]
            }
        ),
    ]
```

---

## Use Case Examples

### 1. HTTP Service Degradation

```
Scenario: API endpoint starts returning 503 errors

Turn 1: http_request("https://api.example.com/data") → 503
  → Record: SERVER_ERROR severity
  → State: TRUSTED (1 failure, threshold=3)

Turn 2: http_request("https://api.example.com/data") → 503
  → Record: SERVER_ERROR severity
  → State: TRUSTED (2 failures)

Turn 3: http_request("https://api.example.com/data") → 503
  → Record: SERVER_ERROR severity
  → State: ESCALATED
  → Notification: "⚠️ Tool 'http_request' escalated (domain: api.example.com)"

Turn 4: http_request("https://api.example.com/data")
  → Permission prompt: "This tool has been escalated..."
  → User: "y" (allow once)
  → Result: 503
  → State: Still ESCALATED

Turn 5: (30 min later, escalation expired)
  → http_request("https://api.example.com/data") → 200 ✓
  → State: RECOVERING (1/3 successes)

Turn 6-7: Two more successful calls
  → State: TRUSTED (recovered)
  → Notification: "✓ Tool 'http_request' recovered for api.example.com"
```

### 2. CLI Command Failures

```
Scenario: Permission issues with file operations

Turn 1: bash("rm /protected/file") → "Permission denied"
  → Record: PERMISSION severity
  → State: TRUSTED (permission errors often expected)

Turn 2: bash("cat /protected/secrets") → "Permission denied"
  → Record: PERMISSION severity
  → State: TRUSTED (2 permission errors)

Turn 3: bash("sudo rm -rf /") → "Permission denied"
  → Record: SECURITY severity (sudo + dangerous command)
  → State: BLOCKED immediately
  → Notification: "🛑 Tool 'bash' blocked - security concern"

All future bash calls require explicit approval until manual reset.
```

### 3. MCP Server Issues

```
Scenario: Atlassian MCP server becomes unresponsive

Turn 1: mcp_atlassian_search("project:PROJ") → timeout
  → Record: TIMEOUT severity
  → Domain: "atlassian" (MCP server name)
  → State: TRUSTED

Turn 2: mcp_atlassian_create_issue(...) → timeout
  → Record: TIMEOUT severity
  → State: ESCALATED (domain-level: 2 timeouts in 5 min)
  → Notification includes MCP server name

Other MCP servers unaffected - only Atlassian tools escalated.
```

---

## Configuration Examples

### Project-Level Configuration

`.jaato/reliability.json`:
```json
{
  "default_rule": {
    "count_threshold": 3,
    "window_seconds": 3600,
    "escalation_duration_seconds": 1800
  },
  "plugin_rules": {
    "cli": {
      "count_threshold": 5,
      "severity_filter": ["crash", "security"]
    },
    "service_connector": {
      "rate_threshold": 0.4,
      "window_seconds": 300
    }
  },
  "domain_rules": {
    "api.production.com": {
      "count_threshold": 2,
      "consecutive_threshold": 2,
      "escalation_duration_seconds": 3600
    }
  }
}
```

### Environment Variables

```bash
# Disable reliability tracking entirely
JAATO_RELIABILITY_ENABLED=false

# Adjust default thresholds
JAATO_RELIABILITY_THRESHOLD=5
JAATO_RELIABILITY_WINDOW=1800

# Disable persistence (session-only tracking)
JAATO_RELIABILITY_PERSIST=false
```

---

## Adaptive Model Selection

When model errors (INVALID_INPUT severity) accumulate for a specific tool, the reliability plugin can suggest or trigger model switching.

### Model Reliability Profile

Track per-model success rates for each tool:

```python
@dataclass
class ModelToolProfile:
    """Tracks a model's reliability with a specific tool."""

    model_name: str
    failure_key: str
    total_attempts: int = 0
    failures: int = 0
    last_failure: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 1.0
        return 1.0 - (self.failures / self.total_attempts)


@dataclass
class ModelReliabilityData:
    """Aggregated model reliability across tools."""

    profiles: Dict[Tuple[str, str], ModelToolProfile]  # (model, failure_key) -> profile

    def get_best_model_for_tool(
        self,
        failure_key: str,
        available_models: List[str],
        min_attempts: int = 3,
    ) -> Optional[str]:
        """Find the model with best success rate for this tool."""

        candidates = []
        for model in available_models:
            key = (model, failure_key)
            if key in self.profiles:
                profile = self.profiles[key]
                if profile.total_attempts >= min_attempts:
                    candidates.append((model, profile.success_rate))

        if not candidates:
            return None

        # Return model with highest success rate
        return max(candidates, key=lambda x: x[1])[0]
```

### Model Switch Triggers

```python
class ModelSwitchStrategy(Enum):
    DISABLED = "disabled"       # Never suggest model switches
    SUGGEST = "suggest"         # Notify user when switch might help
    AUTO = "auto"               # Automatically switch on threshold


@dataclass
class ModelSwitchConfig:
    strategy: ModelSwitchStrategy = ModelSwitchStrategy.SUGGEST
    failure_threshold: int = 3              # Failures before suggesting switch
    min_success_rate_diff: float = 0.3      # Min improvement to suggest (30%)
    preferred_models: List[str] = field(default_factory=list)  # Priority order
```

### Integration with Session

```python
def _check_model_switch(
    self,
    failure_key: str,
    current_model: str,
) -> Optional[ModelSwitchSuggestion]:
    """Check if a different model would be more reliable for this tool."""

    if self._model_switch_config.strategy == ModelSwitchStrategy.DISABLED:
        return None

    current_profile = self._model_data.profiles.get((current_model, failure_key))
    if not current_profile or current_profile.failures < self._model_switch_config.failure_threshold:
        return None

    # Find better model
    available = self._get_available_models()
    best_model = self._model_data.get_best_model_for_tool(failure_key, available)

    if not best_model or best_model == current_model:
        return None

    best_profile = self._model_data.profiles[(best_model, failure_key)]
    improvement = best_profile.success_rate - current_profile.success_rate

    if improvement < self._model_switch_config.min_success_rate_diff:
        return None

    return ModelSwitchSuggestion(
        current_model=current_model,
        suggested_model=best_model,
        failure_key=failure_key,
        current_success_rate=current_profile.success_rate,
        suggested_success_rate=best_profile.success_rate,
    )


def _handle_model_switch_suggestion(self, suggestion: ModelSwitchSuggestion):
    """Handle a model switch suggestion based on strategy."""

    if self._model_switch_config.strategy == ModelSwitchStrategy.SUGGEST:
        self._output_callback(
            "reliability",
            f"💡 Model '{suggestion.suggested_model}' has {suggestion.suggested_success_rate:.0%} "
            f"success rate with '{suggestion.failure_key}' vs current {suggestion.current_success_rate:.0%}.\n"
            f"   Consider: model {suggestion.suggested_model}",
            "write"
        )

    elif self._model_switch_config.strategy == ModelSwitchStrategy.AUTO:
        self._session.switch_model(suggestion.suggested_model)
        self._output_callback(
            "reliability",
            f"⚡ Auto-switched to '{suggestion.suggested_model}' for better reliability "
            f"with '{suggestion.failure_key}'",
            "write"
        )
```

### User Command

```
reliability model                    # Show per-model tool profiles
reliability model suggest            # Enable model switch suggestions
reliability model auto               # Enable automatic model switching
reliability model disabled           # Disable model switching
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] FailureKey model with parameter extraction
- [ ] FailureRecord data model
- [ ] ToolReliabilityState tracking
- [ ] Basic escalation/recovery logic
- [ ] Integration with ToolExecutor result hook

### Phase 2: Permission Integration
- [ ] Permission plugin wrapper
- [ ] Enhanced permission request formatting
- [ ] User notification system

### Phase 3: User Commands
- [ ] `reliability status` command
- [ ] `reliability recovery auto|ask` with completion
- [ ] `reliability reset <failure_key>` with completion
- [ ] `reliability history` command

### Phase 4: Rich Analysis
- [ ] Severity classification rules
- [ ] Parameter-based tracking (tool + args)
- [ ] Rate-based thresholds with decay

### Phase 5: Persistence & Configuration
- [ ] JSON persistence layer (cross-session)
- [ ] Project/user config loading
- [ ] Migration support
- [ ] Session boundary tracking for decay

### Phase 6: Model Reliability
- [ ] ModelToolProfile tracking
- [ ] Model success rate comparison
- [ ] `reliability model suggest|auto|disabled` command
- [ ] Auto-switch integration with session

### Phase 7: Observability
- [ ] reliability_status tool (for model)
- [ ] OpenTelemetry integration
- [ ] Dashboard metrics

---

## Design Decisions

### 1. Model Errors Count Toward Reliability

**Decision**: Yes, failures from model errors (invalid arguments) **do count** toward reliability tracking.

**Rationale**: If a model consistently fails to use a tool correctly, this is valuable signal. The agent could potentially:
- Switch to a different model that handles the tool better
- Request clarification about tool usage
- Escalate to user for guidance

This enables **adaptive model selection** - if Gemini keeps failing at a particular tool but Claude handles it well, the system could suggest or auto-switch models.

```python
# Model error detection
if "invalid" in error_msg or "required" in error_msg or "must be" in error_msg:
    severity = FailureSeverity.INVALID_INPUT
    # Still counts toward thresholds, but with lower weight
    failure_weight = 0.5  # Half weight for model errors
```

### 2. Tracking Granularity: Tool Name + Parameters

**Decision**: Track failures at the **tool name + input parameters** level, not just tool name.

**Rationale**: Not all invocations of a tool are equivalent:
- `readFile("/etc/passwd")` failing is different from `readFile("/tmp/test.txt")` failing
- `http_request("api.example.com")` vs `http_request("api.other.com")`
- `grep("pattern", "/path/a")` vs `grep("pattern", "/path/b")`

**Implementation**: Use a **failure key** that combines tool name with relevant parameter values:

```python
@dataclass
class FailureKey:
    """Identifies a specific tool+parameters combination for tracking."""

    tool_name: str
    parameter_signature: str  # Hash or normalized representation of key params

    @classmethod
    def from_invocation(cls, tool_name: str, args: dict) -> "FailureKey":
        """Create failure key from tool invocation."""
        # Extract relevant parameters based on tool type
        key_params = cls._extract_key_params(tool_name, args)
        signature = cls._normalize_signature(key_params)
        return cls(tool_name=tool_name, parameter_signature=signature)

    @staticmethod
    def _extract_key_params(tool_name: str, args: dict) -> dict:
        """Extract parameters relevant for failure tracking."""

        # File operations: track by path
        if tool_name in ("readFile", "writeFile", "updateFile", "removeFile"):
            path = args.get("path", "")
            return {"path_prefix": str(Path(path).parent)}

        # HTTP requests: track by domain
        if tool_name in ("http_request", "fetch"):
            url = args.get("url", "")
            parsed = urlparse(url)
            return {"domain": parsed.netloc, "path_prefix": parsed.path.split("/")[1] if "/" in parsed.path else ""}

        # CLI commands: track by command prefix
        if tool_name == "bash":
            cmd = args.get("command", "")
            # Extract first word (the actual command)
            cmd_name = cmd.split()[0] if cmd else ""
            return {"command": cmd_name}

        # MCP tools: track by server + tool
        if tool_name.startswith("mcp_"):
            return {"mcp_server": args.get("_mcp_server", "")}

        # Default: hash all args
        return {"args_hash": hashlib.md5(json.dumps(args, sort_keys=True).encode()).hexdigest()[:8]}

    @staticmethod
    def _normalize_signature(params: dict) -> str:
        """Create stable signature from key params."""
        return "|".join(f"{k}={v}" for k, v in sorted(params.items()))
```

**State Tracking**: The `ToolReliabilityState` is now keyed by `FailureKey`:

```python
# Instead of:
tool_states: Dict[str, ToolReliabilityState]  # tool_name -> state

# Now:
tool_states: Dict[str, ToolReliabilityState]  # failure_key.to_string() -> state

# Example keys:
# "readFile|path_prefix=/etc"
# "http_request|domain=api.example.com|path_prefix=v1"
# "bash|command=rm"
```

### 3. User-Controlled Recovery Mode

**Decision**: Recovery mode is controlled via user command with completion support.

**Command**: `reliability recovery auto|ask`

```python
def get_user_commands(self) -> List[UserCommand]:
    return [
        UserCommand(
            name="reliability",
            description="Manage reliability tracking and recovery",
            subcommands=[
                UserCommand(
                    name="recovery",
                    description="Set recovery mode",
                    subcommands=[
                        UserCommand(
                            name="auto",
                            description="Automatically recover tools after cooldown + successes",
                            handler=lambda: self._set_recovery_mode("auto"),
                        ),
                        UserCommand(
                            name="ask",
                            description="Prompt user before recovering escalated tools",
                            handler=lambda: self._set_recovery_mode("ask"),
                        ),
                    ],
                ),
                UserCommand(
                    name="status",
                    description="Show reliability status for all tracked tools",
                    handler=self._show_status,
                ),
                UserCommand(
                    name="reset",
                    description="Reset reliability state for a tool",
                    handler=self._reset_tool,
                    # Completion provides list of escalated tools
                ),
            ],
        ),
    ]

def get_command_completions(self, command_parts: List[str]) -> List[str]:
    """Provide completions for reliability commands."""

    if command_parts == ["reliability"]:
        return ["recovery", "status", "reset"]

    if command_parts == ["reliability", "recovery"]:
        return ["auto", "ask"]

    if command_parts == ["reliability", "reset"]:
        # Return list of currently escalated tools
        return [
            key for key, state in self._tool_states.items()
            if state.state in (TrustState.ESCALATED, TrustState.BLOCKED)
        ]

    return []
```

**Recovery Mode Behavior**:

```python
class RecoveryMode(Enum):
    AUTO = "auto"  # Recover automatically after conditions met
    ASK = "ask"    # Prompt user before recovering

def _attempt_recovery(self, state: ToolReliabilityState) -> bool:
    """Attempt to recover a tool. Returns True if recovered."""

    if not self._can_recover(state):
        return False

    if self._recovery_mode == RecoveryMode.AUTO:
        self._do_recovery(state)
        self._notify_recovery(state, automatic=True)
        return True

    elif self._recovery_mode == RecoveryMode.ASK:
        # Queue recovery request for user
        self._pending_recoveries.append(state.failure_key)
        self._output_callback(
            "reliability",
            f"Tool '{state.failure_key}' ready for recovery. "
            f"Use 'reliability recover {state.failure_key}' to restore trust.",
            "write"
        )
        return False
```

### 4. Cross-Session Tracking

**Decision**: Reliability state **persists across sessions**.

**Rationale**:
- Tools that were unreliable yesterday are likely still unreliable today
- Prevents "reset by restart" exploitation
- Builds long-term reliability profile

**Implementation**: Already covered in Persistence section, but with refinements:

```python
@dataclass
class ReliabilityPersistence:
    """Persisted reliability data across sessions."""

    version: int = 2

    # Tool states persist across sessions
    tool_states: Dict[str, ToolReliabilityState] = field(default_factory=dict)

    # Failure history with session boundaries
    failure_history: List[FailureRecord] = field(default_factory=list)

    # Session metadata for decay calculations
    session_history: List[SessionInfo] = field(default_factory=list)

    # Configuration
    recovery_mode: RecoveryMode = RecoveryMode.AUTO


@dataclass
class SessionInfo:
    """Tracks session boundaries for failure aging."""
    session_id: str
    started: datetime
    ended: Optional[datetime]
    failure_count: int
    success_count: int
```

**Decay Mechanism**: Old failures have reduced weight:

```python
def _calculate_effective_failures(
    self,
    failure_key: str,
    window_seconds: int
) -> float:
    """Calculate failures with age-based decay."""

    now = datetime.now()
    effective_count = 0.0

    for failure in self._get_failures_for_key(failure_key):
        age_seconds = (now - failure.timestamp).total_seconds()

        if age_seconds > window_seconds:
            continue

        # Apply decay: failures lose 10% weight per hour
        decay_factor = max(0.1, 1.0 - (age_seconds / 3600) * 0.1)

        # Cross-session decay: additional 20% reduction per session boundary
        sessions_crossed = self._count_session_boundaries(failure.timestamp)
        session_decay = 0.8 ** sessions_crossed

        effective_count += failure.weight * decay_factor * session_decay

    return effective_count
```

### 5. Parameter-Level Tracking

**Decision**: Track at **tool name + input parameters** granularity.

This is covered in Decision #2 above. The key insight is that the same tool can have very different reliability profiles depending on what it's operating on:

```
Failure tracking examples:

readFile|path_prefix=/etc          → ESCALATED (permission errors)
readFile|path_prefix=/home/user    → TRUSTED (works fine)

http_request|domain=api.flaky.com  → ESCALATED (frequent 503s)
http_request|domain=api.stable.com → TRUSTED (always works)

bash|command=git                   → TRUSTED
bash|command=rm                    → ESCALATED (dangerous, user declined often)
```

---

## Additional User Commands

Based on Decision #3, the full command interface:

```
reliability                         # Show help
reliability status                  # Show all tracked tools with states
reliability status <failure_key>    # Show detailed status for specific key
reliability recovery auto           # Enable automatic recovery
reliability recovery ask            # Require confirmation for recovery
reliability reset <failure_key>     # Manually reset a tool to TRUSTED
reliability reset --all             # Reset all tools (requires confirmation)
reliability history [--limit N]     # Show recent failure history
reliability config                  # Show current configuration
```

### Command Output Examples

```
> reliability status

Reliability Status:
───────────────────────────────────────────────────────────────
  Tool/Parameters                  State       Failures  Recovery
───────────────────────────────────────────────────────────────
  http_request|domain=api.bad.com  ESCALATED   5/3       2/3 successes
  bash|command=rm                  BLOCKED     -         manual reset required
  readFile|path_prefix=/etc        ESCALATED   3/3       cooldown: 12m remaining
───────────────────────────────────────────────────────────────
  3 escalated, 47 trusted (not shown)
  Recovery mode: auto

> reliability reset "http_request|domain=api.bad.com"
✓ Reset 'http_request|domain=api.bad.com' to TRUSTED

> reliability recovery ask
✓ Recovery mode set to 'ask' - you'll be prompted before tools are restored
```
