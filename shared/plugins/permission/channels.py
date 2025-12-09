"""Channel protocol and implementations for interactive permission approval.

Channels handle permission requests that cannot be decided by static policy rules.
They can prompt users, call external services, or use other mechanisms to get
approval for tool execution.
"""

import json
import os
import readline
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import PermissionDisplayInfo, OutputCallback

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ChannelDecision(Enum):
    """Possible decisions from an channel."""
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_ONCE = "allow_once"      # Execute but don't remember
    ALLOW_SESSION = "allow_session"  # Add to session whitelist
    DENY_SESSION = "deny_session"    # Add to session blacklist
    ALLOW_ALL = "allow_all"          # Pre-approve all future requests in session
    TIMEOUT = "timeout"              # Channel didn't respond in time


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

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: int = 30,
        context: Optional[Dict[str, Any]] = None
    ) -> 'PermissionRequest':
        """Create a new permission request with auto-generated ID and timestamp."""
        return cls(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            tool_name=tool_name,
            arguments=arguments,
            timeout_seconds=timeout,
            context=context or {},
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
        }


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

        # Colorized options line
        options = (
            f"Options: "
            f"[{self._c('y', self.ANSI_GREEN)}]es, "
            f"[{self._c('n', self.ANSI_RED)}]o, "
            f"[{self._c('a', self.ANSI_CYAN)}]lways, "
            f"[{self._c('never', self.ANSI_YELLOW)}], "
            f"[once], "
            f"[all]"
        )
        self._output_func(options)

        try:
            response = self._read_input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason="User cancelled input",
            )

        # Parse response
        if response in ("y", "yes"):
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.ALLOW,
                reason="User approved",
            )
        elif response in ("n", "no"):
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason="User denied",
            )
        elif response in ("a", "always"):
            # Create a pattern to remember
            pattern = self._create_remember_pattern(request)
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.ALLOW_SESSION,
                reason="User approved for session",
                remember=True,
                remember_pattern=pattern,
            )
        elif response == "never":
            pattern = self._create_remember_pattern(request)
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY_SESSION,
                reason="User denied for session",
                remember=True,
                remember_pattern=pattern,
            )
        elif response == "once":
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.ALLOW_ONCE,
                reason="User approved once",
            )
        elif response == "all":
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.ALLOW_ALL,
                reason="User pre-approved all future requests",
            )
        else:
            # Unknown response, default to deny
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason=f"Unknown response: {response}",
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

        try:
            response = requests.post(
                self._endpoint,
                json=request.to_dict(),
                headers=headers,
                timeout=self._timeout,
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
            default_decision = ChannelDecision.DENY
            if request.default_on_timeout == "allow":
                default_decision = ChannelDecision.ALLOW

            return ChannelResponse(
                request_id=request.request_id,
                decision=default_decision,
                reason=f"Webhook timeout after {self._timeout}s",
            )

        except requests.RequestException as e:
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason=f"Webhook request failed: {e}",
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

        while time.time() - start_time < request.timeout_seconds:
            if response_file.exists():
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Clean up files
                    request_file.unlink(missing_ok=True)
                    response_file.unlink(missing_ok=True)
                    return ChannelResponse.from_dict(data)
                except (json.JSONDecodeError, IOError) as e:
                    return ChannelResponse(
                        request_id=request.request_id,
                        decision=ChannelDecision.DENY,
                        reason=f"Failed to read response file: {e}",
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

    def set_callbacks(
        self,
        output_callback: Optional[Callable[[str, str, str], None]] = None,
        input_queue: Optional['queue.Queue[str]'] = None,
        prompt_callback: Optional[Callable[[bool], None]] = None,
        **kwargs,
    ) -> None:
        """Set the callbacks and queue for TUI integration.

        Args:
            output_callback: Called with (source, text, mode) to display output.
            input_queue: Queue to receive user input from the main input handler.
            prompt_callback: Called with True when waiting for input, False when done.
        """
        self._output_callback = output_callback
        self._input_queue = input_queue
        self._prompt_callback = prompt_callback

    @property
    def waiting_for_input(self) -> bool:
        """Check if channel is waiting for user input."""
        return self._waiting_for_input

    def _output(self, text: str, mode: str = "append") -> None:
        """Output text via callback."""
        if self._output_callback:
            self._output_callback("permission", text, mode)

    def _read_input(self, timeout: float = 30.0) -> Optional[str]:
        """Read input from the queue with timeout.

        Args:
            timeout: Seconds to wait for input.

        Returns:
            User input string, or None on timeout.
        """
        import queue

        if not self._input_queue:
            return None

        try:
            return self._input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        """Request permission by displaying in output panel and waiting for queue input."""
        from ..base import PermissionDisplayInfo

        # Display the permission request header
        self._output("=" * 60, "write")

        # Show agent type if subagent
        agent_type = request.context.get("agent_type") if request.context else None
        agent_name = request.context.get("agent_name") if request.context else None
        if agent_type == "subagent":
            if agent_name:
                self._output(f"  [Subagent: {agent_name}] Permission Request", "append")
            else:
                self._output("  [Subagent] Permission Request", "append")
        else:
            self._output("  Permission Request", "append")

        self._output("=" * 60, "append")
        self._output("", "append")

        # Tool name
        self._output(f"  Tool: {request.tool_name}", "append")

        # Intent if provided
        intent = request.context.get("intent") if request.context else None
        if intent:
            self._output(f"  Intent: {intent}", "append")

        # Check for custom display info
        display_info = request.context.get("display_info") if request.context else None
        if isinstance(display_info, PermissionDisplayInfo):
            self._output("", "append")
            self._output(f"  {display_info.summary}", "append")
            if display_info.details:
                self._output("", "append")
                for line in display_info.details.split('\n')[:20]:  # Limit lines
                    self._output(f"  {line}", "append")
                if display_info.truncated:
                    self._output("  [Content truncated]", "append")
        else:
            # Show arguments
            self._output("", "append")
            self._output("  Arguments:", "append")
            import json
            args_str = json.dumps(request.arguments, indent=2)
            for line in args_str.split('\n')[:15]:  # Limit lines
                self._output(f"    {line}", "append")

        # Show options
        self._output("", "append")
        self._output("  Options: [y]es, [n]o, [a]lways, [never], [once], [all]", "append")
        self._output("=" * 60, "append")

        # Signal that we're waiting for input
        self._waiting_for_input = True
        if self._prompt_callback:
            self._prompt_callback(True)

        try:
            # Wait for input from queue
            response_text = self._read_input(timeout=request.timeout_seconds)

            if response_text is None:
                # Timeout
                if request.default_on_timeout == "allow":
                    return ChannelResponse(
                        decision=ChannelDecision.ALLOW,
                        reason="Timeout - default allow",
                    )
                return ChannelResponse(
                    decision=ChannelDecision.TIMEOUT,
                    reason=f"No response within {request.timeout_seconds}s",
                )

            # Parse response
            response_lower = response_text.strip().lower()

            if response_lower in ('y', 'yes'):
                decision = ChannelDecision.ALLOW
            elif response_lower in ('n', 'no'):
                decision = ChannelDecision.DENY
            elif response_lower in ('a', 'always'):
                decision = ChannelDecision.ALLOW_SESSION
            elif response_lower == 'never':
                decision = ChannelDecision.DENY_SESSION
            elif response_lower == 'once':
                decision = ChannelDecision.ALLOW_ONCE
            elif response_lower == 'all':
                decision = ChannelDecision.ALLOW_ALL
            else:
                # Invalid input - treat as deny
                decision = ChannelDecision.DENY
                return ChannelResponse(
                    decision=decision,
                    reason=f"Invalid response: {response_text}",
                )

            return ChannelResponse(decision=decision)

        finally:
            # Signal that we're done waiting
            self._waiting_for_input = False
            if self._prompt_callback:
                self._prompt_callback(False)


def create_channel(channel_type: str, config: Optional[Dict[str, Any]] = None) -> Channel:
    """Factory function to create an channel by type.

    Args:
        channel_type: One of "console", "queue", "webhook", "file"
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
    }

    if channel_type not in channels:
        raise ValueError(f"Unknown channel type: {channel_type}. Available: {list(channels.keys())}")

    channel = channels[channel_type]()
    channel.initialize(config)
    return channel
