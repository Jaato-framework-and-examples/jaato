"""Channel protocol and implementations for interactive permission approval.

Channels handle permission requests that cannot be decided by static policy rules.
They can prompt users, call external services, or use other mechanisms to get
approval for tool execution.
"""

import json
import logging
import os
import readline
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..base import PermissionDisplayInfo, OutputCallback

from ...message_queue import SourceType

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ChannelDecision(Enum):
    """Possible decisions from a permission channel.

    Scoped Approval Options (ALLOW_TURN vs ALLOW_UNTIL_IDLE):

    In typical interactive sessions, both behave identically because the session
    goes IDLE immediately after each turn ends. The distinction matters only in
    automated pipelines where multiple messages are sent programmatically without
    user interaction between them.

    ALLOW_TURN:
        - Approves all remaining tool executions for the current turn only
        - Clears when the session transitions to IDLE state
        - Use case: Interactive sessions where you trust the model for this response

    ALLOW_UNTIL_IDLE:
        - Approves all tool executions until the session goes idle (awaiting user input)
        - Persists across multiple consecutive turns if messages are sent programmatically
        - Clears when the session finally transitions to IDLE state
        - Use case: Automated pipelines, batch processing, multi-turn scripts

    Lifecycle:
        Interactive session:
            User message -> Turn starts -> Tools execute -> Turn ends -> IDLE
            (Both ALLOW_TURN and ALLOW_UNTIL_IDLE clear at IDLE)

        Automated pipeline:
            Script message 1 -> Turn 1 -> Turn ends (ALLOW_TURN clears, ALLOW_UNTIL_IDLE persists)
            Script message 2 -> Turn 2 -> Turn ends (ALLOW_TURN clears, ALLOW_UNTIL_IDLE persists)
            No more messages -> IDLE (ALLOW_UNTIL_IDLE clears)
    """
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_ONCE = "allow_once"      # Execute but don't remember
    ALLOW_SESSION = "allow_session"  # Add to session whitelist
    DENY_SESSION = "deny_session"    # Add to session blacklist
    ALLOW_ALL = "allow_all"          # Pre-approve all future requests in session
    ALLOW_TURN = "allow_turn"        # Allow all remaining tools this turn (clears on IDLE)
    ALLOW_UNTIL_IDLE = "allow_until_idle"  # Allow until session goes idle (clears on IDLE)
    TIMEOUT = "timeout"              # Channel didn't respond in time


@dataclass
class PermissionResponseOption:
    """A valid response option for permission prompts.

    This dataclass defines a single response option that users can give
    when answering a permission prompt. It serves as the single source
    of truth for valid responses, used by:
    - Channels to display options and parse user input
    - Clients to provide autocompletion for valid responses
    """
    short: str              # Short form (e.g., "y")
    full: str               # Full form (e.g., "yes")
    description: str        # User-facing description
    decision: ChannelDecision  # The decision this maps to

    def matches(self, input_text: str) -> bool:
        """Check if user input matches this option.

        Args:
            input_text: User's input (case-insensitive).

        Returns:
            True if input matches short or full form.
        """
        lower = input_text.lower().strip()
        return lower == self.short.lower() or lower == self.full.lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "short": self.short,
            "full": self.full,
            "description": self.description,
            "decision": self.decision.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PermissionResponseOption':
        """Create from dictionary."""
        return cls(
            short=data["short"],
            full=data["full"],
            description=data["description"],
            decision=ChannelDecision(data["decision"]),
        )


# Standard permission response options - the single source of truth
DEFAULT_PERMISSION_OPTIONS: List['PermissionResponseOption'] = [
    PermissionResponseOption("y", "yes", "Allow this tool execution", ChannelDecision.ALLOW),
    PermissionResponseOption("n", "no", "Deny this tool execution", ChannelDecision.DENY),
    PermissionResponseOption("a", "always", "Allow and whitelist for session", ChannelDecision.ALLOW_SESSION),
    PermissionResponseOption("t", "turn", "Allow remaining tools this turn", ChannelDecision.ALLOW_TURN),
    PermissionResponseOption("i", "idle", "Allow until session goes idle", ChannelDecision.ALLOW_UNTIL_IDLE),
    PermissionResponseOption("once", "once", "Allow once without remembering", ChannelDecision.ALLOW_ONCE),
    PermissionResponseOption("never", "never", "Deny and blacklist for session", ChannelDecision.DENY_SESSION),
    PermissionResponseOption("all", "all", "Allow all future requests in session", ChannelDecision.ALLOW_ALL),
]


def get_default_permission_options() -> List['PermissionResponseOption']:
    """Get the default list of permission response options.

    Returns a copy to prevent accidental modification of the defaults.
    """
    return list(DEFAULT_PERMISSION_OPTIONS)


@dataclass
class PermissionRequest:
    """Request sent to an channel for permission approval."""

    request_id: str
    timestamp: str
    tool_name: str
    arguments: Dict[str, Any]
    timeout_seconds: int = 30
    default_on_timeout: str = "deny"

    # Optional context for the channel
    context: Dict[str, Any] = field(default_factory=dict)

    # Valid response options - defaults to standard options
    # This allows clients to know what responses are valid for autocompletion
    response_options: List[PermissionResponseOption] = field(
        default_factory=get_default_permission_options
    )

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: int = 30,
        context: Optional[Dict[str, Any]] = None,
        response_options: Optional[List[PermissionResponseOption]] = None,
    ) -> 'PermissionRequest':
        """Create a new permission request with auto-generated ID and timestamp."""
        return cls(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            tool_name=tool_name,
            arguments=arguments,
            timeout_seconds=timeout,
            context=context or {},
            response_options=response_options or get_default_permission_options(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "timeout_seconds": self.timeout_seconds,
            "default_on_timeout": self.default_on_timeout,
            "context": self.context,
            "response_options": [opt.to_dict() for opt in self.response_options],
        }

    def get_option_for_input(self, input_text: str) -> Optional[PermissionResponseOption]:
        """Find the response option matching user input.

        Args:
            input_text: User's input string.

        Returns:
            Matching PermissionResponseOption, or None if no match.
        """
        for option in self.response_options:
            if option.matches(input_text):
                return option
        return None


@dataclass
class ChannelResponse:
    """Response from an channel regarding a permission request."""

    request_id: str
    decision: ChannelDecision
    reason: str = ""
    remember: bool = False  # Whether to remember this decision for the session
    remember_pattern: Optional[str] = None  # Pattern to remember (e.g., "git *")
    expires_at: Optional[str] = None  # ISO8601 expiration time

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelResponse':
        """Create from dictionary."""
        decision_str = data.get("decision", "deny")
        try:
            decision = ChannelDecision(decision_str)
        except ValueError:
            decision = ChannelDecision.DENY

        return cls(
            request_id=data.get("request_id", ""),
            decision=decision,
            reason=data.get("reason", ""),
            remember=data.get("remember", False),
            remember_pattern=data.get("remember_pattern"),
            expires_at=data.get("expires_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "remember": self.remember,
            "remember_pattern": self.remember_pattern,
            "expires_at": self.expires_at,
        }


class Channel(ABC):
    """Base class for permission channels.

    Channels are responsible for handling permission requests that cannot be
    decided by static policy rules. They implement various approval mechanisms.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this channel type."""
        ...

    @abstractmethod
    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        """Request permission from the channel.

        Args:
            request: The permission request to evaluate

        Returns:
            ChannelResponse with the decision
        """
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the channel with optional configuration."""
        pass

    def shutdown(self) -> None:
        """Clean up any resources used by the channel."""
        pass

    def set_output_callback(self, callback: Optional['OutputCallback']) -> None:
        """Set the output callback for real-time output.

        Channels that support interactive output (like ConsoleChannel) can override
        this to use the callback instead of direct print().

        Args:
            callback: OutputCallback function, or None to use default output.
        """
        pass  # Default implementation does nothing


class ConsoleChannel(Channel):
    """Channel that prompts the user in the console for approval.

    This channel is designed for interactive terminal sessions where a human
    can review and approve/deny tool execution requests.
    """

    # ANSI color codes for display
    ANSI_RESET = "\033[0m"
    ANSI_BOLD = "\033[1m"
    ANSI_DIM = "\033[2m"
    ANSI_RED = "\033[31m"
    ANSI_GREEN = "\033[32m"
    ANSI_YELLOW = "\033[33m"
    ANSI_CYAN = "\033[36m"

    def __init__(self):
        self._input_func: Callable[[], str] = input
        self._output_func: Callable[[str], None] = print
        self._default_output_func: Callable[[str], None] = print
        self._output_callback: Optional['OutputCallback'] = None
        self._skip_readline_history: bool = True
        self._use_colors: bool = True  # Can be disabled for non-terminal output

    def _read_input(self) -> str:
        """Read input, optionally avoiding readline history pollution.

        Permission responses (y/n/a/never/once) have no utility in history,
        so by default we remove them after reading.
        """
        if not self._skip_readline_history:
            return self._input_func()

        # Check if readline supports history manipulation (not available on all platforms)
        has_history_support = (
            hasattr(readline, 'get_current_history_length') and
            hasattr(readline, 'remove_history_item')
        )

        if not has_history_support:
            return self._input_func()

        try:
            history_len_before = readline.get_current_history_length()
            result = self._input_func()
            history_len_after = readline.get_current_history_length()

            # Remove the entry if history grew
            if history_len_after > history_len_before:
                readline.remove_history_item(history_len_after - 1)

            return result
        except (AttributeError, OSError):
            # Fallback if readline operations fail
            return self._input_func()

    @property
    def name(self) -> str:
        return "console"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize console channel.

        Config options:
            input_func: Custom input function (for testing)
            output_func: Custom output function (for testing)
            skip_readline_history: Whether to remove responses from readline history (default: True)
            use_colors: Whether to use ANSI colors for diff display (default: True)
        """
        if config:
            if "input_func" in config:
                self._input_func = config["input_func"]
            if "output_func" in config:
                self._output_func = config["output_func"]
                self._default_output_func = config["output_func"]
            if "skip_readline_history" in config:
                self._skip_readline_history = config["skip_readline_history"]
            if "use_colors" in config:
                self._use_colors = config["use_colors"]

    def set_output_callback(self, callback: Optional['OutputCallback']) -> None:
        """Set the output callback for permission prompts.

        When a callback is set, permission prompts are emitted via the callback
        with source="permission" instead of being printed directly.

        Args:
            callback: OutputCallback function, or None to use default print.
        """
        self._output_callback = callback
        if callback:
            # Wrap callback to match output_func signature
            def callback_wrapper(text: str) -> None:
                callback("permission", text, "append")
            self._output_func = callback_wrapper
        else:
            # Restore default output function
            self._output_func = self._default_output_func

    def _c(self, text: str, *codes: str) -> str:
        """Apply ANSI color codes to text if colors are enabled."""
        if not self._use_colors:
            return text
        return f"{''.join(codes)}{text}{self.ANSI_RESET}"

    def _colorize_diff_line(self, line: str) -> str:
        """Colorize a single diff line based on its prefix."""
        if not self._use_colors:
            return line

        if line.startswith('+++') or line.startswith('---'):
            return f"{self.ANSI_DIM}{line}{self.ANSI_RESET}"
        elif line.startswith('@@'):
            return f"{self.ANSI_CYAN}{line}{self.ANSI_RESET}"
        elif line.startswith('+'):
            return f"{self.ANSI_GREEN}{line}{self.ANSI_RESET}"
        elif line.startswith('-'):
            return f"{self.ANSI_RED}{line}{self.ANSI_RESET}"
        else:
            return line

    def _colorize_diff(self, diff_text: str) -> str:
        """Colorize a unified diff for terminal display."""
        lines = diff_text.split('\n')
        colorized = [self._colorize_diff_line(line) for line in lines]
        return '\n'.join(colorized)

    def _render_display_info(self, display_info: 'PermissionDisplayInfo') -> str:
        """Render PermissionDisplayInfo for console display.

        Args:
            display_info: Display info from the source plugin

        Returns:
            Formatted string for console output
        """
        from ..base import PermissionDisplayInfo  # Import here to avoid circular

        lines = []

        # Summary line
        lines.append(f"  {display_info.summary}")
        lines.append("")

        # Details with format-specific rendering
        if display_info.format_hint == "diff":
            lines.append(self._colorize_diff(display_info.details))
        else:
            # For text, json, code - display as-is
            lines.append(display_info.details)

        # Truncation warning
        if display_info.truncated:
            lines.append("")
            if display_info.original_lines:
                lines.append(f"  [Truncated: showing partial content, {display_info.original_lines} lines total]")
            else:
                lines.append("  [Truncated: content was too large to display in full]")

        return '\n'.join(lines)

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        """Prompt user in console for permission.

        Displays tool name, intent, and arguments, then asks for approval.
        If a PermissionDisplayInfo is provided in the context, uses that for
        custom rendering (e.g., colorized diffs for file operations).

        Supported responses:
            y/yes     -> ALLOW
            n/no      -> DENY
            a/always  -> ALLOW_SESSION (remember this tool for session)
            never     -> DENY_SESSION (block this tool for session)
            once      -> ALLOW_ONCE (don't remember)
            all       -> ALLOW_ALL (pre-approve all future requests in session)
        """
        from ..base import PermissionDisplayInfo  # Import here to avoid circular

        # Format the request for display
        self._output_func("")
        self._output_func(self._c("=" * 60, self.ANSI_BOLD))

        # Display agent type to clarify who is asking for permission
        agent_type = request.context.get("agent_type") if request.context else None
        agent_name = request.context.get("agent_name") if request.context else None
        if agent_type == "subagent":
            if agent_name:
                self._output_func(
                    f"{self._c('[askPermission]', self.ANSI_YELLOW)} "
                    f"Subagent '{agent_name}' requesting tool execution:"
                )
            else:
                self._output_func(
                    f"{self._c('[askPermission]', self.ANSI_YELLOW)} "
                    "Subagent requesting tool execution:"
                )
        else:
            self._output_func(
                f"{self._c('[askPermission]', self.ANSI_YELLOW)} "
                "Main agent requesting tool execution:"
            )

        # Display intent prominently if provided
        intent = request.context.get("intent") if request.context else None
        if intent:
            self._output_func(f"  Intent: {intent}")

        # Check for custom display info from source plugin
        display_info = request.context.get("display_info") if request.context else None
        if display_info and isinstance(display_info, PermissionDisplayInfo):
            # Use custom rendering from the source plugin
            self._output_func(self._render_display_info(display_info))
        else:
            # Default display: tool name and JSON arguments
            self._output_func(f"  {self._c('Tool:', self.ANSI_BOLD)} {request.tool_name}")
            self._output_func(f"  Arguments: {json.dumps(request.arguments, indent=4)}")

        self._output_func(self._c("=" * 60, self.ANSI_BOLD))
        self._output_func("")

        # Build options line from request.response_options
        options_line = self._build_options_line(request.response_options)
        self._output_func(options_line)

        try:
            response = self._read_input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason="User cancelled input",
            )

        # Parse response using request's response_options
        matched_option = request.get_option_for_input(response)
        if matched_option:
            return self._create_response_for_option(request, matched_option)
        else:
            # Unknown response, default to deny
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason=f"Unknown response: {response}",
            )

    def _build_options_line(self, options: List[PermissionResponseOption]) -> str:
        """Build a colorized options display line from response options.

        Args:
            options: List of valid response options.

        Returns:
            Formatted options string for display.
        """
        # Color mapping for different decision types
        color_map = {
            ChannelDecision.ALLOW: self.ANSI_GREEN,
            ChannelDecision.DENY: self.ANSI_RED,
            ChannelDecision.ALLOW_SESSION: self.ANSI_CYAN,
            ChannelDecision.DENY_SESSION: self.ANSI_YELLOW,
            ChannelDecision.ALLOW_ONCE: "",  # No special color
            ChannelDecision.ALLOW_ALL: "",   # No special color
            ChannelDecision.ALLOW_TURN: self.ANSI_CYAN,  # Same as session-level
            ChannelDecision.ALLOW_UNTIL_IDLE: self.ANSI_CYAN,  # Same as session-level
        }

        parts = ["Options: "]
        for i, opt in enumerate(options):
            color = color_map.get(opt.decision, "")
            if opt.short != opt.full:
                # Format: [y]es
                parts.append(f"[{self._c(opt.short, color)}]{opt.full[len(opt.short):]}")
            else:
                # Format: [once]
                parts.append(f"[{self._c(opt.full, color) if color else opt.full}]")
            if i < len(options) - 1:
                parts.append(", ")

        return "".join(parts)

    def _create_response_for_option(
        self, request: PermissionRequest, option: PermissionResponseOption
    ) -> ChannelResponse:
        """Create a ChannelResponse for a matched option.

        Args:
            request: The original permission request.
            option: The matched response option.

        Returns:
            ChannelResponse with appropriate decision and metadata.
        """
        # Determine if this decision needs a remember pattern
        remember_decisions = {
            ChannelDecision.ALLOW_SESSION,
            ChannelDecision.DENY_SESSION,
        }

        if option.decision in remember_decisions:
            pattern = self._create_remember_pattern(request)
            return ChannelResponse(
                request_id=request.request_id,
                decision=option.decision,
                reason=f"User chose: {option.full}",
                remember=True,
                remember_pattern=pattern,
            )
        else:
            return ChannelResponse(
                request_id=request.request_id,
                decision=option.decision,
                reason=f"User chose: {option.full}",
            )

    def _create_remember_pattern(self, request: PermissionRequest) -> str:
        """Create a pattern to remember for future requests.

        For CLI tools, uses the command prefix. For other tools, uses the tool name.
        """
        if request.tool_name == "cli_based_tool":
            command = request.arguments.get("command", "")
            # Extract the command name (first word)
            parts = command.split()
            if parts:
                return f"{parts[0]} *"
            return request.tool_name
        else:
            return request.tool_name


class WebhookChannel(Channel):
    """Channel that sends permission requests to an HTTP webhook.

    This channel is designed for integration with external approval systems,
    such as Slack bots, approval workflows, or custom dashboards.
    """

    def __init__(self):
        self._endpoint: Optional[str] = None
        self._timeout: int = 30
        self._headers: Dict[str, str] = {}
        self._auth_token: Optional[str] = None

    @property
    def name(self) -> str:
        return "webhook"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize webhook channel.

        Config options:
            endpoint: URL to send requests to (required)
            timeout: Request timeout in seconds
            headers: Additional headers to include
            auth_token: Bearer token for authorization
        """
        if not HAS_REQUESTS:
            raise RuntimeError("requests library required for WebhookChannel")

        if not config:
            raise ValueError("WebhookChannel requires configuration with 'endpoint'")

        self._endpoint = config.get("endpoint")
        if not self._endpoint:
            raise ValueError("WebhookChannel requires 'endpoint' in config")

        self._timeout = config.get("timeout", 30)
        self._headers = config.get("headers", {})
        self._auth_token = config.get("auth_token") or os.environ.get("PERMISSION_WEBHOOK_TOKEN")

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        """Send permission request to webhook and wait for response."""
        if not self._endpoint:
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason="Webhook endpoint not configured",
            )

        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }

        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        from shared.http import get_requests_kwargs

        proxy_kwargs = get_requests_kwargs(self._endpoint)
        if 'headers' in proxy_kwargs:
            headers.update(proxy_kwargs.pop('headers'))

        try:
            response = requests.post(
                self._endpoint,
                json=request.to_dict(),
                headers=headers,
                timeout=self._timeout,
                **proxy_kwargs,
            )

            if response.status_code == 200:
                data = response.json()
                return ChannelResponse.from_dict(data)
            else:
                return ChannelResponse(
                    request_id=request.request_id,
                    decision=ChannelDecision.DENY,
                    reason=f"Webhook returned status {response.status_code}",
                )

        except requests.Timeout:
            logger.warning(f"Permission webhook timeout after {self._timeout}s for request {request.request_id}")
            default_decision = ChannelDecision.DENY
            if request.default_on_timeout == "allow":
                default_decision = ChannelDecision.ALLOW

            return ChannelResponse(
                request_id=request.request_id,
                decision=default_decision,
                reason=f"Webhook timeout after {self._timeout}s",
            )

        except requests.RequestException as e:
            logger.error(f"Permission webhook request failed for {request.request_id}", exc_info=True)
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason=f"Webhook request failed: {e}\nTraceback: {traceback.format_exc()}",
            )


class FileChannel(Channel):
    """Channel that writes requests to a file and polls for responses.

    This channel is designed for scenarios where a separate process handles
    approval, such as a background service or manual file editing.

    Request files: {base_path}/requests/{request_id}.json
    Response files: {base_path}/responses/{request_id}.json
    """

    def __init__(self):
        self._base_path: Optional[Path] = None
        self._poll_interval: float = 0.5  # seconds between polls

    @property
    def name(self) -> str:
        return "file"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize file channel.

        Config options:
            base_path: Directory for request/response files (required)
            poll_interval: Seconds between polling attempts
        """
        if not config:
            raise ValueError("FileChannel requires configuration with 'base_path'")

        base_path = config.get("base_path")
        if not base_path:
            raise ValueError("FileChannel requires 'base_path' in config")

        self._base_path = Path(base_path)
        self._poll_interval = config.get("poll_interval", 0.5)

        # Create directories
        (self._base_path / "requests").mkdir(parents=True, exist_ok=True)
        (self._base_path / "responses").mkdir(parents=True, exist_ok=True)

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        """Write request file and poll for response file."""
        if not self._base_path:
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason="FileChannel base path not configured",
            )

        # Write request file
        request_file = self._base_path / "requests" / f"{request.request_id}.json"
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request.to_dict(), f, indent=2)

        # Poll for response
        response_file = self._base_path / "responses" / f"{request.request_id}.json"
        start_time = time.time()

        # Apply env var override at runtime (session env may not be loaded at init time)
        timeout = request.timeout_seconds
        env_timeout = os.environ.get("JAATO_PERMISSION_TIMEOUT")
        if env_timeout is not None:
            try:
                timeout = float(env_timeout)
            except ValueError:
                pass

        no_timeout = timeout <= 0  # 0 or negative means wait forever

        while no_timeout or time.time() - start_time < timeout:
            if response_file.exists():
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Clean up files
                    request_file.unlink(missing_ok=True)
                    response_file.unlink(missing_ok=True)
                    return ChannelResponse.from_dict(data)
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Failed to read permission response file for {request.request_id}", exc_info=True)
                    return ChannelResponse(
                        request_id=request.request_id,
                        decision=ChannelDecision.DENY,
                        reason=f"Failed to read response file: {e}\nTraceback: {traceback.format_exc()}",
                    )

            time.sleep(self._poll_interval)

        # Timeout - clean up request file
        request_file.unlink(missing_ok=True)

        default_decision = ChannelDecision.DENY
        if request.default_on_timeout == "allow":
            default_decision = ChannelDecision.ALLOW

        return ChannelResponse(
            request_id=request.request_id,
            decision=default_decision,
            reason=f"Timeout after {request.timeout_seconds}s waiting for response",
        )

    def shutdown(self) -> None:
        """Clean up any pending request files."""
        if self._base_path:
            requests_dir = self._base_path / "requests"
            if requests_dir.exists():
                for f in requests_dir.glob("*.json"):
                    f.unlink(missing_ok=True)


class QueueChannel(ConsoleChannel):
    """Channel that displays prompts via callback and receives input via queue.

    Designed for TUI integration where:
    - Permission prompts are shown in an output panel
    - User input comes through a shared queue from the main input handler
    - No direct stdin access needed (works with full-screen terminal UIs)
    """

    def __init__(self):
        super().__init__()
        self._output_callback: Optional[Callable[[str, str, str], None]] = None
        self._input_queue: Optional['queue.Queue[str]'] = None
        self._waiting_for_input: bool = False
        self._prompt_callback: Optional[Callable[[bool], None]] = None
        self._cancel_token: Optional[Any] = None  # CancelToken for interruption

    def set_callbacks(
        self,
        output_callback: Optional[Callable[[str, str, str], None]] = None,
        input_queue: Optional['queue.Queue[str]'] = None,
        prompt_callback: Optional[Callable[[bool], None]] = None,
        cancel_token: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Set the callbacks and queue for TUI integration.

        Args:
            output_callback: Called with (source, text, mode) to display output.
            input_queue: Queue to receive user input from the main input handler.
            prompt_callback: Called with True when waiting for input, False when done.
            cancel_token: Optional CancelToken to check for cancellation requests.
        """
        self._output_callback = output_callback
        self._input_queue = input_queue
        self._prompt_callback = prompt_callback
        self._cancel_token = cancel_token

    @property
    def waiting_for_input(self) -> bool:
        """Check if channel is waiting for user input."""
        return self._waiting_for_input

    def _output(self, text: str, mode: str = "append") -> None:
        """Output text via callback."""
        if self._output_callback:
            self._output_callback("permission", text, mode)

    def _read_input(self, timeout: float = 30.0) -> Optional[str]:
        """Read input from the queue with timeout, checking for cancellation.

        Args:
            timeout: Seconds to wait for input. 0 or negative means wait forever.

        Returns:
            User input string, None on timeout, or "__CANCELLED__" if cancelled.
        """
        import queue
        import time

        if not self._input_queue:
            return None

        # Apply env var override at runtime (session env may not be loaded at init time)
        env_timeout = os.environ.get("JAATO_PERMISSION_TIMEOUT")
        if env_timeout is not None:
            try:
                timeout = float(env_timeout)
            except ValueError:
                pass

        # Poll in short intervals to check for cancellation
        poll_interval = 0.1  # Check every 100ms
        elapsed = 0.0
        no_timeout = timeout <= 0  # 0 or negative means wait forever

        while no_timeout or elapsed < timeout:
            # Check for cancellation
            if self._cancel_token and hasattr(self._cancel_token, 'is_cancelled'):
                if self._cancel_token.is_cancelled:
                    return "__CANCELLED__"

            try:
                return self._input_queue.get(timeout=poll_interval)
            except queue.Empty:
                elapsed += poll_interval

        return None

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        """Wait for permission response from queue input.

        Note: Permission content is displayed via unified event flow (AgentOutputEvent).
        This method only handles input waiting.
        """
        # Signal that we're waiting for input
        self._waiting_for_input = True
        if self._prompt_callback:
            self._prompt_callback(True)

        try:
            # Wait for input from queue
            response_text = self._read_input(timeout=request.timeout_seconds)

            if response_text == "__CANCELLED__":
                # User cancelled via Ctrl+C
                return ChannelResponse(
                    request_id=request.request_id,
                    decision=ChannelDecision.DENY,
                    reason="User cancelled",
                )

            if response_text is None:
                # Timeout
                if request.default_on_timeout == "allow":
                    return ChannelResponse(
                        request_id=request.request_id,
                        decision=ChannelDecision.ALLOW,
                        reason="Timeout - default allow",
                    )
                return ChannelResponse(
                    request_id=request.request_id,
                    decision=ChannelDecision.TIMEOUT,
                    reason=f"No response within {request.timeout_seconds}s",
                )

            # Parse response using request's response_options (uses parent's method)
            matched_option = request.get_option_for_input(response_text)
            if matched_option:
                return self._create_response_for_option(request, matched_option)
            else:
                # Invalid input - treat as deny
                return ChannelResponse(
                    request_id=request.request_id,
                    decision=ChannelDecision.DENY,
                    reason=f"Invalid response: {response_text}",
                )

        finally:
            # Signal that we're done waiting
            self._waiting_for_input = False
            if self._prompt_callback:
                self._prompt_callback(False)


class ParentBridgedChannel(Channel):
    """Channel for subagents that routes permission requests through parent.

    Instead of blocking on a local queue waiting for user input, this channel:
    1. Forwards the permission request to the parent agent
    2. Waits for the parent's response on the session's injection queue
    3. Parses the response and returns it

    This unifies the communication model - all subagent input comes through
    the same injection queue mechanism.
    """

    def __init__(self):
        """Initialize the parent-bridged channel."""
        self._session: Optional[Any] = None  # JaatoSession reference
        self._pending_request_id: Optional[str] = None
        self._timeout: float = 300.0  # 5 minute timeout for parent response

    @property
    def name(self) -> str:
        return "parent_bridged"

    def set_session(self, session: Any) -> None:
        """Set the session reference for forwarding to parent.

        Args:
            session: JaatoSession instance that has parent reference.
        """
        self._session = session

    def _format_request_for_parent(self, request: PermissionRequest) -> str:
        """Format permission request as structured message for parent.

        Args:
            request: The permission request.

        Returns:
            Formatted string for parent to parse.
        """
        lines = [f'<permission_request request_id="{request.request_id}">']
        lines.append(f'  <tool_name>{request.tool_name}</tool_name>')

        # Format arguments
        lines.append('  <arguments>')
        for key, value in request.arguments.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:500] + '...'
            lines.append(f'    <arg name="{key}">{str_value}</arg>')
        lines.append('  </arguments>')

        # Include response options
        lines.append('  <options>')
        for opt in request.response_options:
            lines.append(f'    <option short="{opt.short}" full="{opt.full}">{opt.description}</option>')
        lines.append('  </options>')

        lines.append('</permission_request>')
        return '\n'.join(lines)

    def _parse_response_from_parent(self, response: str, request: PermissionRequest) -> ChannelResponse:
        """Parse parent's response into ChannelResponse.

        Args:
            response: Raw response string from parent.
            request: Original request (for context and options).

        Returns:
            Parsed ChannelResponse.
        """
        import re

        # Try to extract decision from structured response
        # Format: <permission_response request_id="..."><decision>yes</decision></permission_response>
        decision_pattern = r'<decision>(.*?)</decision>'
        match = re.search(decision_pattern, response, re.IGNORECASE | re.DOTALL)

        if match:
            decision_text = match.group(1).strip().lower()
        else:
            # Fallback: treat entire response as decision
            decision_text = response.strip().lower()

        # Match against response options
        for option in request.response_options:
            if option.matches(decision_text):
                return ChannelResponse(
                    request_id=request.request_id,
                    decision=option.decision,
                    reason=f"Parent responded: {decision_text}",
                    remember=(option.decision in [ChannelDecision.ALLOW_SESSION, ChannelDecision.DENY_SESSION]),
                )

        # Default to deny if unrecognized
        return ChannelResponse(
            request_id=request.request_id,
            decision=ChannelDecision.DENY,
            reason=f"Unrecognized response from parent: {decision_text}",
        )

    def _wait_for_response(self, request_id: str) -> Optional[str]:
        """Wait for parent's response on injection queue.

        Args:
            request_id: The request ID to match.

        Returns:
            Response string or None on timeout.
        """
        import queue as queue_module

        if not self._session:
            return None

        # Access the session's injection queue
        injection_queue = getattr(self._session, '_injection_queue', None)
        if not injection_queue:
            return None

        poll_interval = 0.1
        elapsed = 0.0
        held_messages = []  # Messages that aren't our response

        while elapsed < self._timeout:
            # Check for cancellation
            cancel_token = getattr(self._session, '_cancel_token', None)
            if cancel_token and hasattr(cancel_token, 'is_cancelled'):
                if cancel_token.is_cancelled:
                    # Put held messages back
                    for msg in held_messages:
                        injection_queue.put(msg)
                    return None

            try:
                message = injection_queue.get(timeout=poll_interval)

                # Check if this is our response
                if f'request_id="{request_id}"' in message or f"request_id='{request_id}'" in message:
                    # Put held messages back
                    for msg in held_messages:
                        injection_queue.put(msg)
                    return message

                # Check if it's a permission response without explicit request_id
                if '<permission_response' in message.lower() and self._pending_request_id == request_id:
                    for msg in held_messages:
                        injection_queue.put(msg)
                    return message

                # Check for simple yes/no/allow/deny responses (single pending request)
                simple_response = message.strip().lower()
                if self._pending_request_id == request_id and simple_response in ['y', 'yes', 'n', 'no', 'a', 'always', 'once', 'deny']:
                    for msg in held_messages:
                        injection_queue.put(msg)
                    return message

                # Not our response, hold it
                held_messages.append(message)

            except queue_module.Empty:
                elapsed += poll_interval

        # Timeout - put held messages back
        for msg in held_messages:
            injection_queue.put(msg)
        return None

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        """Request permission by forwarding to parent agent.

        Args:
            request: The permission request.

        Returns:
            ChannelResponse from parent.
        """
        if not self._session:
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason="No session configured for parent-bridged channel",
            )

        # Check if we have a parent
        parent_session = getattr(self._session, '_parent_session', None)
        if not parent_session:
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason="No parent session - this agent cannot request permission from parent",
            )

        self._pending_request_id = request.request_id

        # Format request for parent
        formatted_request = self._format_request_for_parent(request)

        # Forward to parent via session's mechanism
        forward_method = getattr(self._session, '_forward_to_parent', None)
        if forward_method:
            forward_method("PERMISSION_REQUESTED", formatted_request)
        else:
            # Fallback: direct inject to parent (CHILD source - from subagent)
            agent_id = getattr(self._session, '_agent_id', 'unknown')
            parent_session.inject_prompt(
                f"[SUBAGENT agent_id={agent_id} event=PERMISSION_REQUESTED]\n{formatted_request}",
                source_id=agent_id,
                source_type=SourceType.CHILD
            )

        # Wait for response
        response = self._wait_for_response(request.request_id)
        self._pending_request_id = None

        if response is None:
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.TIMEOUT,
                reason="Timeout waiting for parent response",
            )

        # Parse response
        return self._parse_response_from_parent(response, request)


def create_channel(channel_type: str, config: Optional[Dict[str, Any]] = None) -> Channel:
    """Factory function to create an channel by type.

    Args:
        channel_type: One of "console", "queue", "webhook", "file", "parent_bridged"
        config: Optional configuration for the channel

    Returns:
        Initialized Channel instance

    Raises:
        ValueError: If channel_type is unknown
    """
    channels = {
        "console": ConsoleChannel,
        "queue": QueueChannel,
        "webhook": WebhookChannel,
        "file": FileChannel,
        "parent_bridged": ParentBridgedChannel,
    }

    if channel_type not in channels:
        raise ValueError(f"Unknown channel type: {channel_type}. Available: {list(channels.keys())}")

    channel = channels[channel_type]()
    channel.initialize(config)
    return channel
