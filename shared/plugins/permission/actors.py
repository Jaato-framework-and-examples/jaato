"""Actor protocol and implementations for interactive permission approval.

Actors handle permission requests that cannot be decided by static policy rules.
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
from typing import Any, Callable, Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ActorDecision(Enum):
    """Possible decisions from an actor."""
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_ONCE = "allow_once"      # Execute but don't remember
    ALLOW_SESSION = "allow_session"  # Add to session whitelist
    DENY_SESSION = "deny_session"    # Add to session blacklist
    ALLOW_ALL = "allow_all"          # Pre-approve all future requests in session
    TIMEOUT = "timeout"              # Actor didn't respond in time


@dataclass
class PermissionRequest:
    """Request sent to an actor for permission approval."""

    request_id: str
    timestamp: str
    tool_name: str
    arguments: Dict[str, Any]
    timeout_seconds: int = 30
    default_on_timeout: str = "deny"

    # Optional context for the actor
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
class ActorResponse:
    """Response from an actor regarding a permission request."""

    request_id: str
    decision: ActorDecision
    reason: str = ""
    remember: bool = False  # Whether to remember this decision for the session
    remember_pattern: Optional[str] = None  # Pattern to remember (e.g., "git *")
    expires_at: Optional[str] = None  # ISO8601 expiration time

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActorResponse':
        """Create from dictionary."""
        decision_str = data.get("decision", "deny")
        try:
            decision = ActorDecision(decision_str)
        except ValueError:
            decision = ActorDecision.DENY

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


class Actor(ABC):
    """Base class for permission actors.

    Actors are responsible for handling permission requests that cannot be
    decided by static policy rules. They implement various approval mechanisms.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this actor type."""
        ...

    @abstractmethod
    def request_permission(self, request: PermissionRequest) -> ActorResponse:
        """Request permission from the actor.

        Args:
            request: The permission request to evaluate

        Returns:
            ActorResponse with the decision
        """
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the actor with optional configuration."""
        pass

    def shutdown(self) -> None:
        """Clean up any resources used by the actor."""
        pass


class ConsoleActor(Actor):
    """Actor that prompts the user in the console for approval.

    This actor is designed for interactive terminal sessions where a human
    can review and approve/deny tool execution requests.
    """

    def __init__(self):
        self._input_func: Callable[[], str] = input
        self._output_func: Callable[[str], None] = print
        self._skip_readline_history: bool = True

    def _read_input(self) -> str:
        """Read input, optionally avoiding readline history pollution.

        Permission responses (y/n/a/never/once) have no utility in history,
        so by default we remove them after reading.
        """
        if not self._skip_readline_history:
            return self._input_func()

        history_len_before = readline.get_current_history_length()
        result = self._input_func()
        history_len_after = readline.get_current_history_length()

        # Remove the entry if history grew
        if history_len_after > history_len_before:
            readline.remove_history_item(history_len_after - 1)

        return result

    @property
    def name(self) -> str:
        return "console"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize console actor.

        Config options:
            input_func: Custom input function (for testing)
            output_func: Custom output function (for testing)
            skip_readline_history: Whether to remove responses from readline history (default: True)
        """
        if config:
            if "input_func" in config:
                self._input_func = config["input_func"]
            if "output_func" in config:
                self._output_func = config["output_func"]
            if "skip_readline_history" in config:
                self._skip_readline_history = config["skip_readline_history"]

    def request_permission(self, request: PermissionRequest) -> ActorResponse:
        """Prompt user in console for permission.

        Displays tool name, intent, and arguments, then asks for approval.
        Supported responses:
            y/yes     -> ALLOW
            n/no      -> DENY
            a/always  -> ALLOW_SESSION (remember this tool for session)
            never     -> DENY_SESSION (block this tool for session)
            once      -> ALLOW_ONCE (don't remember)
            all       -> ALLOW_ALL (pre-approve all future requests in session)
        """
        # Format the request for display
        self._output_func("")
        self._output_func("=" * 60)
        self._output_func("[askPermission] Tool execution request:")
        # Display intent prominently if provided
        intent = request.context.get("intent") if request.context else None
        if intent:
            self._output_func(f"  Intent: {intent}")
        self._output_func(f"  Tool: {request.tool_name}")
        self._output_func(f"  Arguments: {json.dumps(request.arguments, indent=4)}")
        self._output_func("=" * 60)
        self._output_func("")
        self._output_func("Options: [y]es, [n]o, [a]lways, [never], [once], [all]")

        try:
            response = self._read_input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.DENY,
                reason="User cancelled input",
            )

        # Parse response
        if response in ("y", "yes"):
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.ALLOW,
                reason="User approved",
            )
        elif response in ("n", "no"):
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.DENY,
                reason="User denied",
            )
        elif response in ("a", "always"):
            # Create a pattern to remember
            pattern = self._create_remember_pattern(request)
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.ALLOW_SESSION,
                reason="User approved for session",
                remember=True,
                remember_pattern=pattern,
            )
        elif response == "never":
            pattern = self._create_remember_pattern(request)
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.DENY_SESSION,
                reason="User denied for session",
                remember=True,
                remember_pattern=pattern,
            )
        elif response == "once":
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.ALLOW_ONCE,
                reason="User approved once",
            )
        elif response == "all":
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.ALLOW_ALL,
                reason="User pre-approved all future requests",
            )
        else:
            # Unknown response, default to deny
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.DENY,
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


class WebhookActor(Actor):
    """Actor that sends permission requests to an HTTP webhook.

    This actor is designed for integration with external approval systems,
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
        """Initialize webhook actor.

        Config options:
            endpoint: URL to send requests to (required)
            timeout: Request timeout in seconds
            headers: Additional headers to include
            auth_token: Bearer token for authorization
        """
        if not HAS_REQUESTS:
            raise RuntimeError("requests library required for WebhookActor")

        if not config:
            raise ValueError("WebhookActor requires configuration with 'endpoint'")

        self._endpoint = config.get("endpoint")
        if not self._endpoint:
            raise ValueError("WebhookActor requires 'endpoint' in config")

        self._timeout = config.get("timeout", 30)
        self._headers = config.get("headers", {})
        self._auth_token = config.get("auth_token") or os.environ.get("PERMISSION_WEBHOOK_TOKEN")

    def request_permission(self, request: PermissionRequest) -> ActorResponse:
        """Send permission request to webhook and wait for response."""
        if not self._endpoint:
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.DENY,
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
                return ActorResponse.from_dict(data)
            else:
                return ActorResponse(
                    request_id=request.request_id,
                    decision=ActorDecision.DENY,
                    reason=f"Webhook returned status {response.status_code}",
                )

        except requests.Timeout:
            default_decision = ActorDecision.DENY
            if request.default_on_timeout == "allow":
                default_decision = ActorDecision.ALLOW

            return ActorResponse(
                request_id=request.request_id,
                decision=default_decision,
                reason=f"Webhook timeout after {self._timeout}s",
            )

        except requests.RequestException as e:
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.DENY,
                reason=f"Webhook request failed: {e}",
            )


class FileActor(Actor):
    """Actor that writes requests to a file and polls for responses.

    This actor is designed for scenarios where a separate process handles
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
        """Initialize file actor.

        Config options:
            base_path: Directory for request/response files (required)
            poll_interval: Seconds between polling attempts
        """
        if not config:
            raise ValueError("FileActor requires configuration with 'base_path'")

        base_path = config.get("base_path")
        if not base_path:
            raise ValueError("FileActor requires 'base_path' in config")

        self._base_path = Path(base_path)
        self._poll_interval = config.get("poll_interval", 0.5)

        # Create directories
        (self._base_path / "requests").mkdir(parents=True, exist_ok=True)
        (self._base_path / "responses").mkdir(parents=True, exist_ok=True)

    def request_permission(self, request: PermissionRequest) -> ActorResponse:
        """Write request file and poll for response file."""
        if not self._base_path:
            return ActorResponse(
                request_id=request.request_id,
                decision=ActorDecision.DENY,
                reason="FileActor base path not configured",
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
                    return ActorResponse.from_dict(data)
                except (json.JSONDecodeError, IOError) as e:
                    return ActorResponse(
                        request_id=request.request_id,
                        decision=ActorDecision.DENY,
                        reason=f"Failed to read response file: {e}",
                    )

            time.sleep(self._poll_interval)

        # Timeout - clean up request file
        request_file.unlink(missing_ok=True)

        default_decision = ActorDecision.DENY
        if request.default_on_timeout == "allow":
            default_decision = ActorDecision.ALLOW

        return ActorResponse(
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


def create_actor(actor_type: str, config: Optional[Dict[str, Any]] = None) -> Actor:
    """Factory function to create an actor by type.

    Args:
        actor_type: One of "console", "webhook", "file"
        config: Optional configuration for the actor

    Returns:
        Initialized Actor instance

    Raises:
        ValueError: If actor_type is unknown
    """
    actors = {
        "console": ConsoleActor,
        "webhook": WebhookActor,
        "file": FileActor,
    }

    if actor_type not in actors:
        raise ValueError(f"Unknown actor type: {actor_type}. Available: {list(actors.keys())}")

    actor = actors[actor_type]()
    actor.initialize(config)
    return actor
