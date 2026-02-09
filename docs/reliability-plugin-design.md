# Reliability Plugin Design Document

## Overview

The **reliability plugin** is a cross-cutting concern that tracks both **tool failures** and **behavioral patterns** to dynamically adjust the agentic loop. The core principles are:

- **Adaptive trust**: Tools that work reliably maintain approval status; unreliable tools get escalated back to requiring explicit user approval
- **Behavioral awareness**: Detect when models are stuck in unproductive loops (e.g., repeatedly calling introspection tools, announcing actions without executing)
- **Active intervention**: Inject nudges into the conversation to unstick stalled models

## Design Goals

1. **Non-invasive integration** - Works alongside existing permission system without modifying it
2. **Configurable thresholds** - Different failure tolerances for different contexts
3. **Transparent operation** - Clear user notifications when escalation occurs
4. **Recovery paths** - Tools can "earn back" trust after a cooldown period
5. **Rich context** - Capture enough information to make intelligent escalation decisions
6. **Adaptive model selection** - Enable model switching when current model consistently fails at specific tools
7. **Behavioral pattern detection** - Identify stall loops, repetitive calls, and announce-without-action patterns
8. **Nudge injection** - Actively guide models out of unproductive patterns

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

## Behavioral Pattern Detection

Beyond tool failures, the reliability plugin can detect **behavioral stalls** - patterns where the model is stuck in an unproductive loop even though individual tool calls succeed.

### Pattern Types

```python
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
```

### BehavioralPattern Record

```python
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
    last_model_text: Optional[str]        # What the model said (e.g., "Proceeding now")
    expected_action: Optional[str]        # What tool should have been called

    # Severity
    severity: PatternSeverity


class PatternSeverity(Enum):
    MINOR = "minor"           # 2-3 repetitions, just starting
    MODERATE = "moderate"     # 4-5 repetitions, clear stall
    SEVERE = "severe"         # 6+ repetitions, intervention needed
```

### Detection Engine

```python
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
        default_factory=lambda: {"readFile", "list_tools", "get_tool_schemas", "glob", "grep"}
    )
    action_tools: Set[str] = field(
        default_factory=lambda: {"writeFile", "updateFile", "bash", "removeFile"}
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
            "executing",
            "running",
            "starting",
        ]
    )


class PatternDetector:
    """Detects behavioral patterns in tool usage."""

    def __init__(self, config: PatternDetectionConfig):
        self._config = config
        self._turn_history: List[ToolCall] = []
        self._turn_start: Optional[datetime] = None
        self._last_action_index: int = -1
        self._announced_actions: List[str] = []

    def on_turn_start(self):
        """Called when a new turn begins."""
        self._turn_history = []
        self._turn_start = datetime.now()
        self._announced_actions = []

    def on_tool_called(self, tool_name: str, args: dict) -> Optional[BehavioralPattern]:
        """Called BEFORE tool execution. Returns pattern if detected."""

        self._turn_history.append(ToolCall(tool_name, args, datetime.now()))

        # Check for repetitive calls
        pattern = self._check_repetitive_calls(tool_name)
        if pattern:
            return pattern

        # Check for introspection loop
        pattern = self._check_introspection_loop(tool_name)
        if pattern:
            return pattern

        # Check for read-only stall
        pattern = self._check_read_only_stall(tool_name)
        if pattern:
            return pattern

        return None

    def on_model_text(self, text: str):
        """Called when model outputs text. Track announcements."""
        text_lower = text.lower()
        for phrase in self._config.announce_phrases:
            if phrase in text_lower:
                self._announced_actions.append(text)
                break

    def on_tool_result(self, tool_name: str, success: bool):
        """Called after tool execution."""
        if tool_name in self._config.action_tools and success:
            self._last_action_index = len(self._turn_history) - 1
            self._announced_actions = []  # Action taken, clear announcements

    def check_announce_no_action(self) -> Optional[BehavioralPattern]:
        """Check if model announced action but didn't follow through."""

        if not self._announced_actions:
            return None

        # Had announcements, check if any action tools were called after
        recent_tools = [t.tool_name for t in self._turn_history[-3:]]
        action_taken = any(t in self._config.action_tools for t in recent_tools)

        if not action_taken and len(self._announced_actions) >= 2:
            return BehavioralPattern(
                pattern_type=BehavioralPatternType.ANNOUNCE_NO_ACTION,
                detected_at=datetime.now(),
                turn_index=self._current_turn,
                session_id=self._session_id,
                tool_sequence=recent_tools,
                repetition_count=len(self._announced_actions),
                duration_seconds=self._turn_elapsed(),
                model_name=self._model_name,
                last_model_text=self._announced_actions[-1],
                expected_action="action tool (write, bash, etc.)",
                severity=self._calculate_severity(len(self._announced_actions)),
            )

        return None

    def _check_repetitive_calls(self, tool_name: str) -> Optional[BehavioralPattern]:
        """Detect same tool called repeatedly."""

        if len(self._turn_history) < self._config.repetitive_call_threshold:
            return None

        recent = self._turn_history[-self._config.repetitive_call_threshold:]
        if all(t.tool_name == tool_name for t in recent):
            return BehavioralPattern(
                pattern_type=BehavioralPatternType.REPETITIVE_CALLS,
                detected_at=datetime.now(),
                turn_index=self._current_turn,
                session_id=self._session_id,
                tool_sequence=[t.tool_name for t in recent],
                repetition_count=len(recent),
                duration_seconds=self._turn_elapsed(),
                model_name=self._model_name,
                last_model_text=None,
                expected_action=None,
                severity=self._calculate_severity(len(recent)),
            )

        return None

    def _check_introspection_loop(self, tool_name: str) -> Optional[BehavioralPattern]:
        """Detect stuck in introspection tools."""

        if tool_name not in self._config.introspection_tool_names:
            return None

        recent = self._turn_history[-self._config.introspection_loop_threshold:]
        introspection_count = sum(
            1 for t in recent if t.tool_name in self._config.introspection_tool_names
        )

        if introspection_count >= self._config.introspection_loop_threshold:
            return BehavioralPattern(
                pattern_type=BehavioralPatternType.INTROSPECTION_LOOP,
                detected_at=datetime.now(),
                turn_index=self._current_turn,
                session_id=self._session_id,
                tool_sequence=[t.tool_name for t in recent],
                repetition_count=introspection_count,
                duration_seconds=self._turn_elapsed(),
                model_name=self._model_name,
                last_model_text=None,
                expected_action="proceed with actual task",
                severity=PatternSeverity.MODERATE,
            )

        return None
```

### Hook Integration

The reliability plugin needs a new hook point - `on_tool_called` - in addition to `on_tool_result`:

```python
class ReliabilityPlugin:
    """Extended with behavioral pattern detection."""

    def set_tool_hooks(self):
        """Called by ToolExecutor to register hooks."""
        return {
            "on_tool_called": self._on_tool_called,    # NEW: before execution
            "on_tool_result": self._on_tool_result,    # existing: after execution
        }

    def set_model_output_hook(self):
        """Called by session to track model text output."""
        return self._on_model_text

    def _on_tool_called(self, tool_name: str, args: dict, call_id: str):
        """Hook called before tool execution."""

        pattern = self._pattern_detector.on_tool_called(tool_name, args)

        if pattern:
            self._handle_detected_pattern(pattern)

    def _on_model_text(self, text: str):
        """Hook called when model outputs text."""
        self._pattern_detector.on_model_text(text)

        # Also check for announce-no-action at text boundaries
        pattern = self._pattern_detector.check_announce_no_action()
        if pattern:
            self._handle_detected_pattern(pattern)
```

### Nudge Injection Mechanism

When a pattern is detected, inject guidance into the agentic loop:

```python
class NudgeType(Enum):
    """Types of nudges to inject."""

    GENTLE_REMINDER = "gentle"      # Soft suggestion
    DIRECT_INSTRUCTION = "direct"   # Clear instruction
    INTERRUPT = "interrupt"         # Stop and ask user


@dataclass
class Nudge:
    """A guidance injection for the model."""

    nudge_type: NudgeType
    message: str
    pattern: BehavioralPattern
    injected_at: datetime


class NudgeStrategy:
    """Determines what nudge to inject based on pattern."""

    NUDGE_TEMPLATES = {
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
    }

    def create_nudge(self, pattern: BehavioralPattern) -> Nudge:
        """Create appropriate nudge for detected pattern."""

        templates = self.NUDGE_TEMPLATES.get(pattern.pattern_type, {})
        nudge_type, template = templates.get(
            pattern.severity,
            (NudgeType.GENTLE_REMINDER, "Please proceed with the task.")
        )

        message = template.format(
            tool_name=pattern.tool_sequence[-1] if pattern.tool_sequence else "unknown",
            count=pattern.repetition_count,
        )

        return Nudge(
            nudge_type=nudge_type,
            message=message,
            pattern=pattern,
            injected_at=datetime.now(),
        )


class NudgeInjector:
    """Injects nudges into the agentic loop."""

    def __init__(self, session: JaatoSession):
        self._session = session
        self._injected_nudges: List[Nudge] = []

    def inject(self, nudge: Nudge) -> bool:
        """Inject nudge into the conversation. Returns True if injected."""

        self._injected_nudges.append(nudge)

        if nudge.nudge_type == NudgeType.INTERRUPT:
            # Pause execution and notify user
            self._session.request_pause(reason=nudge.message)
            self._notify_user(nudge)
            return True

        elif nudge.nudge_type == NudgeType.DIRECT_INSTRUCTION:
            # Inject as system message for next turn
            self._session.inject_system_guidance(nudge.message)
            self._notify_user(nudge, level="warning")
            return True

        elif nudge.nudge_type == NudgeType.GENTLE_REMINDER:
            # Inject as subtle context
            self._session.inject_context_hint(nudge.message)
            return True

        return False

    def _notify_user(self, nudge: Nudge, level: str = "info"):
        """Show nudge to user via output callback."""
        prefix = "⚠️" if level == "warning" else "ℹ️"
        self._output_callback(
            "reliability",
            f"{prefix} Pattern detected: {nudge.pattern.pattern_type.value}\n"
            f"   {nudge.message}",
            "write"
        )
```

### Session Integration

The session needs methods to accept nudge injections:

```python
class JaatoSession:
    """Extended with nudge injection points."""

    def inject_system_guidance(self, message: str):
        """Inject guidance that appears as system context in next turn."""
        self._pending_guidance.append({
            "role": "system",
            "content": f"[RELIABILITY GUIDANCE]: {message}",
        })

    def inject_context_hint(self, message: str):
        """Inject subtle hint into context (less prominent than guidance)."""
        self._context_hints.append(message)

    def request_pause(self, reason: str):
        """Pause the agentic loop for user intervention."""
        self._paused = True
        self._pause_reason = reason
        # Emit event for client to handle
        self._emit_event(PauseRequestedEvent(reason=reason))
```

### Model Behavioral Profiles

Extend model profiles to track behavioral patterns:

```python
@dataclass
class ModelBehavioralProfile:
    """Tracks a model's behavioral tendencies."""

    model_name: str

    # Pattern frequencies
    pattern_counts: Dict[BehavioralPatternType, int] = field(default_factory=dict)

    # Nudge effectiveness
    nudges_sent: int = 0
    nudges_effective: int = 0  # Pattern stopped after nudge

    # Stall tendency
    total_turns: int = 0
    stalled_turns: int = 0

    @property
    def stall_rate(self) -> float:
        if self.total_turns == 0:
            return 0.0
        return self.stalled_turns / self.total_turns

    @property
    def nudge_effectiveness(self) -> float:
        if self.nudges_sent == 0:
            return 1.0
        return self.nudges_effective / self.nudges_sent

    def most_common_pattern(self) -> Optional[BehavioralPatternType]:
        if not self.pattern_counts:
            return None
        return max(self.pattern_counts.items(), key=lambda x: x[1])[0]
```

### User Commands for Behavioral Tracking

```
reliability patterns                     # Show detected patterns summary
reliability patterns --model <name>      # Show patterns for specific model
reliability nudge gentle|direct|off      # Set nudge aggressiveness
reliability stalls                       # Show stall statistics
```

### Example Scenario

```
Turn 1:
  Model: "I'll restore the file from backup. Proceeding now."
  Tool: list_tools({category: 'filesystem'}) ✓
  → on_model_text: detected "proceeding now"
  → on_tool_called: list_tools (introspection tool)
  → No action tool followed announcement

Turn 2:
  Model: "I'm going to restore and apply the fix. Proceeding immediately."
  Tool: list_tools({category: 'filesystem'}) ✓
  → on_model_text: detected "proceeding" (2nd time)
  → on_tool_called: list_tools again
  → Pattern detected: ANNOUNCE_NO_ACTION (severity: MINOR)
  → Nudge injected: "You mentioned proceeding but haven't called an action tool yet."

Turn 3:
  Model: "You're right - I need to actually execute. Proceeding with the real steps now."
  Tool: list_tools({category: 'filesystem'}) ✓
  → on_model_text: detected "proceeding" (3rd time!)
  → Pattern detected: ANNOUNCE_NO_ACTION (severity: MODERATE)
  → Nudge injected: "NOTICE: You've announced action 3 times without executing."

Turn 4:
  Model: [still not taking action]
  Tool: list_tools(...) ✓
  → Pattern detected: ANNOUNCE_NO_ACTION (severity: SEVERE)
  → INTERRUPT: System pauses, notifies user
  → User sees: "⚠️ Model stuck in announce-no-action loop. Intervention needed."
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] FailureKey model with parameter extraction
- [ ] FailureRecord data model
- [ ] ToolReliabilityState tracking
- [ ] Basic escalation/recovery logic
- [ ] Integration with ToolExecutor result hook (`on_tool_result`)

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
- [ ] User-level persistence (`~/.jaato/reliability.json`)
- [ ] Workspace-level persistence (`.jaato/reliability.json`)
- [ ] Session state serialization for reconnect
- [ ] Three-level merge strategy (session > workspace > user)
- [ ] Pattern promotion (workspace → user for cross-workspace patterns)
- [ ] Migration support between versions
- [ ] Session boundary tracking for decay

### Phase 6: Model Reliability (Tool Failures)
- [ ] ModelToolProfile tracking
- [ ] Model success rate comparison
- [ ] `reliability model suggest|auto|disabled` command
- [ ] Auto-switch integration with session

### Phase 7: Behavioral Pattern Detection
- [ ] Add `on_tool_called` hook to ToolExecutor (before execution)
- [ ] Add model text output hook to session
- [ ] BehavioralPattern and PatternDetector implementation
- [ ] Pattern type detection (repetitive, introspection loop, announce-no-action)
- [ ] Severity escalation logic

### Phase 8: Nudge Injection System
- [ ] NudgeStrategy and nudge templates
- [ ] NudgeInjector implementation
- [ ] Session integration (`inject_system_guidance`, `inject_context_hint`, `request_pause`)
- [ ] `reliability nudge gentle|direct|off` command
- [ ] `reliability patterns` command

### Phase 9: Model Behavioral Profiles
- [ ] ModelBehavioralProfile tracking
- [ ] Nudge effectiveness tracking
- [ ] Stall rate per model
- [ ] Cross-reference with tool failure profiles for model switching decisions

### Phase 10: Observability
- [ ] reliability_status tool (for model)
- [ ] Pattern detection events for OpenTelemetry
- [ ] Dashboard metrics (failures + patterns)

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

### 6. Persistence Hierarchy: Session → Workspace → User

**Decision**: Three-level persistence hierarchy with different data at each level.

#### Persistence Levels

```
┌─────────────────────────────────────────────────────────────────┐
│                     SESSION (in-memory + IPC)                    │
│  • Current turn tool sequence                                   │
│  • Active behavioral patterns being tracked                     │
│  • Pending nudges                                               │
│  • Live escalation states                                       │
│  └──────────────────────────────────────────────────────────────┘
│                              ▲ restored on reconnect
│  ┌──────────────────────────────────────────────────────────────┐
│  │                  WORKSPACE (.jaato/reliability.json)          │
│  │  • Tool reliability states (per failure_key)                 │
│  │  • Failure history for this project                          │
│  │  • Model profiles for tools used in this workspace           │
│  │  • Behavioral pattern history (this project)                 │
│  │  • Workspace-specific thresholds/overrides                   │
│  └──────────────────────────────────────────────────────────────┘
│                              ▲ inherits defaults from
│  ┌──────────────────────────────────────────────────────────────┐
│  │                    USER (~/.jaato/reliability.json)           │
│  │  • Default configuration (thresholds, recovery mode)         │
│  │  • Cross-workspace model behavioral profiles                 │
│  │  • Global tool patterns (tools that fail everywhere)         │
│  │  • Nudge aggressiveness preference                           │
│  │  • Model switching preferences                               │
│  └──────────────────────────────────────────────────────────────┘
```

#### Data Structures Per Level

```python
@dataclass
class SessionReliabilityState:
    """In-memory state for current session. Restored on reconnect."""

    session_id: str
    started_at: datetime

    # Current turn tracking
    current_turn_index: int
    current_turn_tools: List[ToolCall]
    announced_actions: List[str]

    # Active patterns
    active_patterns: List[BehavioralPattern]
    pending_nudges: List[Nudge]

    # Live states (overlay on workspace)
    escalation_overrides: Dict[str, TrustState]  # Temporary escalations

    # Session-level settings (override workspace/user, restored on reconnect)
    session_settings: SessionSettings

    def serialize_for_reconnect(self) -> bytes:
        """Serialize for IPC persistence during disconnect."""
        ...

    @classmethod
    def restore_from_reconnect(cls, data: bytes) -> "SessionReliabilityState":
        """Restore after client reconnects to server."""
        ...


@dataclass
class SessionSettings:
    """Runtime settings for current session. Restored on reconnect."""

    # Nudge settings
    nudge_level: Optional[NudgeType] = None           # None = inherit from workspace/user
    nudge_enabled: bool = True

    # Recovery settings
    recovery_mode: Optional[RecoveryMode] = None      # None = inherit

    # Model switching
    model_switch_strategy: Optional[ModelSwitchStrategy] = None  # None = inherit

    # Pattern detection
    pattern_detection_enabled: bool = True

    def is_default(self) -> bool:
        """Check if all settings are inherited (no overrides)."""
        return (
            self.nudge_level is None
            and self.nudge_enabled
            and self.recovery_mode is None
            and self.model_switch_strategy is None
            and self.pattern_detection_enabled
        )


@dataclass
class WorkspaceReliabilityData:
    """Persisted per workspace in .jaato/reliability.json"""

    version: int = 1

    # Tool reliability (workspace-specific)
    tool_states: Dict[str, ToolReliabilityState]
    failure_history: List[FailureRecord]

    # Model profiles for this workspace
    model_tool_profiles: Dict[Tuple[str, str], ModelToolProfile]
    model_behavioral_profiles: Dict[str, ModelBehavioralProfile]

    # Behavioral patterns in this workspace
    pattern_history: List[BehavioralPattern]

    # Workspace-specific config overrides
    config_overrides: Optional[ReliabilityConfig]

    # Workspace-specific settings (override user defaults)
    settings: Optional[SessionSettings] = None

    # Session tracking for decay
    session_history: List[SessionInfo]


@dataclass
class UserReliabilityData:
    """Persisted per user in ~/.jaato/reliability.json"""

    version: int = 1

    # Default configuration
    default_config: ReliabilityConfig

    # Cross-workspace patterns
    global_model_behavioral_profiles: Dict[str, ModelBehavioralProfile]
    global_tool_patterns: Dict[str, ToolReliabilityState]  # Tools that fail everywhere

    # User default settings
    default_settings: SessionSettings

    # Legacy fields (for backwards compat, prefer default_settings)
    recovery_mode: RecoveryMode
    nudge_aggressiveness: NudgeType
    model_switch_strategy: ModelSwitchStrategy
    preferred_models: List[str]
```

#### Merge Strategy

```python
class ReliabilityPersistenceManager:
    """Manages the three-level persistence hierarchy."""

    def __init__(self, workspace_path: Optional[Path] = None):
        self._user_data = self._load_user_data()
        self._workspace_data = self._load_workspace_data(workspace_path)
        self._session_state: Optional[SessionReliabilityState] = None

    def get_effective_config(self) -> ReliabilityConfig:
        """Merge configs: user defaults < workspace overrides."""
        config = copy.deepcopy(self._user_data.default_config)

        if self._workspace_data.config_overrides:
            # Workspace overrides specific fields
            for field in fields(self._workspace_data.config_overrides):
                value = getattr(self._workspace_data.config_overrides, field.name)
                if value is not None:
                    setattr(config, field.name, value)

        return config

    def get_tool_state(self, failure_key: str) -> ToolReliabilityState:
        """Get tool state with session > workspace > user priority."""

        # Session override (temporary escalation)
        if self._session_state and failure_key in self._session_state.escalation_overrides:
            state = self._workspace_data.tool_states.get(failure_key)
            if state:
                state = copy.copy(state)
                state.state = self._session_state.escalation_overrides[failure_key]
                return state

        # Workspace state
        if failure_key in self._workspace_data.tool_states:
            return self._workspace_data.tool_states[failure_key]

        # User global pattern (tools that fail across workspaces)
        if failure_key in self._user_data.global_tool_patterns:
            return self._user_data.global_tool_patterns[failure_key]

        # Default: trusted
        return ToolReliabilityState(
            failure_key=failure_key,
            tool_name=failure_key.split("|")[0],
            state=TrustState.TRUSTED,
            ...
        )

    def get_model_profile(self, model: str, failure_key: str) -> ModelToolProfile:
        """Get model profile: workspace-specific > user global."""

        key = (model, failure_key)

        # Workspace-specific profile
        if key in self._workspace_data.model_tool_profiles:
            return self._workspace_data.model_tool_profiles[key]

        # User global (cross-workspace)
        # Note: We use tool_name only for global, not full failure_key
        tool_name = failure_key.split("|")[0]
        global_key = (model, tool_name)
        if global_key in self._user_data.global_model_behavioral_profiles:
            return self._user_data.global_model_behavioral_profiles[global_key]

        return ModelToolProfile(model_name=model, failure_key=failure_key)
```

#### Session Restore on Reconnect

When a client disconnects and reconnects to a running server:

```python
class JaatoServer:
    """Server-side session persistence for reconnect."""

    def __init__(self):
        self._session_snapshots: Dict[str, bytes] = {}

    def on_client_disconnect(self, session_id: str):
        """Snapshot session state for potential reconnect."""
        session = self._sessions.get(session_id)
        if session and session.reliability_plugin:
            snapshot = session.reliability_plugin.get_session_state().serialize_for_reconnect()
            self._session_snapshots[session_id] = snapshot
            # Keep snapshot for 5 minutes
            self._schedule_snapshot_cleanup(session_id, timeout=300)

    def on_client_reconnect(self, session_id: str, client: IpcClient):
        """Restore session state on reconnect."""
        if session_id in self._session_snapshots:
            snapshot = self._session_snapshots.pop(session_id)
            session = self._sessions.get(session_id)
            if session and session.reliability_plugin:
                state = SessionReliabilityState.restore_from_reconnect(snapshot)
                session.reliability_plugin.restore_session_state(state)

                # Notify client of restored state
                client.send(SessionRestoredEvent(
                    session_id=session_id,
                    turn_index=state.current_turn_index,
                    active_patterns=len(state.active_patterns),
                    pending_nudges=len(state.pending_nudges),
                ))
```

#### Promotion: Workspace → User

When patterns are consistent across workspaces, promote to user level:

```python
def _check_global_pattern_promotion(self, failure_key: str):
    """Promote workspace patterns to user-global if consistent."""

    tool_name = failure_key.split("|")[0]

    # Check if this tool fails in multiple workspaces
    # (requires tracking workspace identifiers in user data)
    workspaces_with_failures = self._user_data.get_workspaces_with_tool_failures(tool_name)

    if len(workspaces_with_failures) >= 3:
        # This tool fails across workspaces - promote to global
        avg_failure_rate = self._calculate_cross_workspace_failure_rate(tool_name)

        if avg_failure_rate > 0.5:  # Fails more than half the time everywhere
            self._user_data.global_tool_patterns[tool_name] = ToolReliabilityState(
                failure_key=tool_name,  # Generic, not workspace-specific
                tool_name=tool_name,
                state=TrustState.ESCALATED,
                escalation_reason=f"Fails across {len(workspaces_with_failures)} workspaces",
            )
            self._save_user_data()
```

#### File Locations

```python
def _get_workspace_path(self, workspace: Path) -> Path:
    """Workspace-level persistence."""
    return workspace / ".jaato" / "reliability.json"

def _get_user_path(self) -> Path:
    """User-level persistence."""
    return Path.home() / ".jaato" / "reliability.json"

def _get_session_snapshot_path(self, session_id: str) -> Path:
    """Temporary session snapshots (for server crash recovery)."""
    return Path(tempfile.gettempdir()) / "jaato_sessions" / f"{session_id}.snapshot"
```

#### Environment Variable Overrides

```bash
# Override persistence locations
JAATO_RELIABILITY_USER_PATH=~/.config/jaato/reliability.json
JAATO_RELIABILITY_WORKSPACE_PATH=.jaato/reliability.json

# Disable persistence levels
JAATO_RELIABILITY_NO_USER_PERSIST=true      # Don't save to user level
JAATO_RELIABILITY_NO_WORKSPACE_PERSIST=true # Don't save to workspace level
JAATO_RELIABILITY_SESSION_ONLY=true         # No persistence at all
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
reliability reset all               # Reset all tools (requires confirmation)
reliability history [limit N]       # Show recent failure history
reliability config                  # Show current configuration

# Settings management
reliability nudge gentle|direct|off # Set nudge level for this session
reliability nudge save workspace    # Save current nudge setting to workspace
reliability nudge save user         # Save current nudge setting as user default

reliability settings                # Show effective settings (merged view)
reliability settings save workspace # Save all current settings to workspace
reliability settings save user      # Save all current settings as user defaults
reliability settings clear workspace # Clear workspace overrides (inherit from user)
reliability settings clear session  # Clear session overrides (inherit from workspace)
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

> reliability settings

Reliability Settings (effective):
───────────────────────────────────────────────────────────────
  Setting                Value           Source
───────────────────────────────────────────────────────────────
  nudge_level            direct          session (override)
  nudge_enabled          true            user (default)
  recovery_mode          ask             workspace
  model_switch_strategy  suggest         user (default)
  pattern_detection      true            user (default)
───────────────────────────────────────────────────────────────

> reliability nudge gentle
✓ Nudge level set to 'gentle' for this session

> reliability nudge save workspace
✓ Nudge level 'gentle' saved to workspace (.jaato/reliability.json)
  Future sessions in this workspace will use this setting.

> reliability settings save user
✓ Current settings saved as user defaults (~/.jaato/reliability.json)
  Settings saved:
    - nudge_level: gentle
    - recovery_mode: ask
    - model_switch_strategy: suggest

> reliability settings clear session
✓ Session overrides cleared. Now inheriting from workspace/user settings.
```

### Settings Merge Implementation

```python
class ReliabilitySettingsManager:
    """Manages settings across session/workspace/user levels."""

    def __init__(
        self,
        user_data: UserReliabilityData,
        workspace_data: WorkspaceReliabilityData,
        session_state: SessionReliabilityState,
    ):
        self._user = user_data
        self._workspace = workspace_data
        self._session = session_state

    def get_effective_settings(self) -> EffectiveSettings:
        """Resolve settings with session > workspace > user priority."""
        return EffectiveSettings(
            nudge_level=self._resolve("nudge_level"),
            nudge_enabled=self._resolve("nudge_enabled"),
            recovery_mode=self._resolve("recovery_mode"),
            model_switch_strategy=self._resolve("model_switch_strategy"),
            pattern_detection_enabled=self._resolve("pattern_detection_enabled"),
        )

    def _resolve(self, setting: str) -> Tuple[Any, str]:
        """Resolve a single setting, returning (value, source)."""

        # Session override (highest priority)
        session_val = getattr(self._session.session_settings, setting, None)
        if session_val is not None:
            return (session_val, "session")

        # Workspace override
        if self._workspace.settings:
            workspace_val = getattr(self._workspace.settings, setting, None)
            if workspace_val is not None:
                return (workspace_val, "workspace")

        # User default (lowest priority)
        user_val = getattr(self._user.default_settings, setting, None)
        if user_val is not None:
            return (user_val, "user")

        # Hardcoded fallback
        return (self._get_hardcoded_default(setting), "default")

    def save_to_workspace(self, settings: Optional[List[str]] = None):
        """Save current session settings to workspace level."""

        if self._workspace.settings is None:
            self._workspace.settings = SessionSettings()

        if settings is None:
            # Save all non-default session settings
            for field in fields(SessionSettings):
                val = getattr(self._session.session_settings, field.name)
                if val is not None:
                    setattr(self._workspace.settings, field.name, val)
        else:
            # Save specific settings
            for setting in settings:
                val = getattr(self._session.session_settings, setting)
                setattr(self._workspace.settings, setting, val)

        self._persist_workspace()

    def save_to_user(self, settings: Optional[List[str]] = None):
        """Save current effective settings as user defaults."""

        effective = self.get_effective_settings()

        if settings is None:
            # Save all
            for field in fields(SessionSettings):
                val, _ = getattr(effective, field.name)
                setattr(self._user.default_settings, field.name, val)
        else:
            for setting in settings:
                val, _ = getattr(effective, setting)
                setattr(self._user.default_settings, setting, val)

        self._persist_user()

    def clear_session_overrides(self):
        """Clear all session-level setting overrides."""
        self._session.session_settings = SessionSettings()

    def clear_workspace_overrides(self):
        """Clear all workspace-level setting overrides."""
        self._workspace.settings = None
        self._persist_workspace()


@dataclass
class EffectiveSettings:
    """Resolved settings with source tracking."""

    nudge_level: Tuple[NudgeType, str]              # (value, source)
    nudge_enabled: Tuple[bool, str]
    recovery_mode: Tuple[RecoveryMode, str]
    model_switch_strategy: Tuple[ModelSwitchStrategy, str]
    pattern_detection_enabled: Tuple[bool, str]

    def format_table(self) -> str:
        """Format as table for display."""
        lines = [
            "Reliability Settings (effective):",
            "─" * 60,
            f"  {'Setting':<25} {'Value':<15} Source",
            "─" * 60,
        ]
        for field in fields(self):
            val, source = getattr(self, field.name)
            source_display = f"{source}" + (" (override)" if source == "session" else " (default)" if source == "user" else "")
            lines.append(f"  {field.name:<25} {str(val):<15} {source_display}")
        lines.append("─" * 60)
        return "\n".join(lines)
```

### Settings Persistence on Reconnect

When a client reconnects, session settings are restored:

```python
def on_client_reconnect(self, session_id: str, client: IpcClient):
    """Restore session state including settings on reconnect."""

    if session_id in self._session_snapshots:
        snapshot = self._session_snapshots.pop(session_id)
        state = SessionReliabilityState.restore_from_reconnect(snapshot)

        # Restore session settings
        session.reliability_plugin.restore_session_state(state)

        # Notify client what was restored
        client.send(SessionRestoredEvent(
            session_id=session_id,
            turn_index=state.current_turn_index,
            active_patterns=len(state.active_patterns),
            pending_nudges=len(state.pending_nudges),
            settings_restored=not state.session_settings.is_default(),
        ))

        if not state.session_settings.is_default():
            # Inform user their settings were preserved
            self._output_callback(
                "reliability",
                "✓ Session settings restored from before disconnect",
                "write"
            )
```
