"""Channel protocol and implementations for reference selection.

Channels handle the interactive selection of reference sources when the model
requests additional documentation. They support three communication protocols:
- Console: Interactive terminal prompts
- Webhook: HTTP-based external approval systems
- File: Filesystem-based for automation/scripting

The channel type is determined by configuration (matching the permission plugin pattern).
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .models import ReferenceSource, SelectionRequest, SelectionResponse


class SelectionChannel(ABC):
    """Base class for reference selection channels.

    Channels are responsible for presenting available references to the user
    and collecting their selection. The selection flow is triggered when
    the model calls the selectReferences tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this channel type."""
        ...

    @abstractmethod
    def present_selection(
        self,
        available_sources: List[ReferenceSource],
        context: Optional[str] = None
    ) -> List[str]:
        """Present available sources and get user selection.

        Args:
            available_sources: List of reference sources to choose from
            context: Optional context explaining why references are needed

        Returns:
            List of selected source IDs
        """
        ...

    @abstractmethod
    def notify_result(self, message: str) -> None:
        """Notify user of selection result."""
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the channel with optional configuration."""
        pass

    def shutdown(self) -> None:
        """Clean up any resources used by the channel."""
        pass


class ConsoleSelectionChannel(SelectionChannel):
    """Channel that prompts the user in the console for selection.

    Displays a numbered list of available references and accepts
    comma-separated indices, 'all', or 'none' as input.
    """

    def __init__(self):
        self._timeout: int = 60
        self._input_func: Callable[[], str] = input
        self._output_func: Callable[[str], None] = print
        self._skip_readline_history: bool = True

    @property
    def name(self) -> str:
        return "console"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize console channel.

        Config options:
            timeout: Input timeout in seconds (default: 60)
            input_func: Custom input function (for testing)
            output_func: Custom output function (for testing)
            skip_readline_history: Whether to skip adding to history (default: True)
        """
        if config:
            self._timeout = config.get("timeout", 60)
            if "input_func" in config:
                self._input_func = config["input_func"]
            if "output_func" in config:
                self._output_func = config["output_func"]
            if "skip_readline_history" in config:
                self._skip_readline_history = config["skip_readline_history"]

    def _read_input(self) -> str:
        """Read input, optionally avoiding readline history pollution."""
        if not self._skip_readline_history or not HAS_READLINE:
            return self._input_func()

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

            if history_len_after > history_len_before:
                readline.remove_history_item(history_len_after - 1)

            return result
        except (AttributeError, OSError):
            return self._input_func()

    def present_selection(
        self,
        available_sources: List[ReferenceSource],
        context: Optional[str] = None
    ) -> List[str]:
        """Display selection menu and collect user input."""
        self._output_func("")
        self._output_func("=" * 60)
        self._output_func("REFERENCE SELECTION")
        self._output_func("=" * 60)

        if context:
            self._output_func(f"\nContext: {context}\n")

        self._output_func("Available references:\n")

        for i, source in enumerate(available_sources, 1):
            tags_str = ", ".join(source.tags) if source.tags else "none"
            self._output_func(f"  [{i}] {source.name}")
            self._output_func(f"      {source.description}")
            self._output_func(f"      Type: {source.type.value} | Tags: {tags_str}")
            self._output_func("")

        self._output_func("Enter selection:")
        self._output_func("  - Numbers separated by commas (e.g., '1,3,4')")
        self._output_func("  - 'all' to select all")
        self._output_func("  - 'none' or empty to skip")
        self._output_func("")

        try:
            response = self._read_input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []

        if not response or response == 'none':
            return []

        if response == 'all':
            return [s.id for s in available_sources]

        selected_ids = []
        try:
            indices = [int(x.strip()) for x in response.split(',')]
            for idx in indices:
                if 1 <= idx <= len(available_sources):
                    selected_ids.append(available_sources[idx - 1].id)
        except ValueError:
            self._output_func("Invalid input, no references selected.")
            return []

        return selected_ids

    def notify_result(self, message: str) -> None:
        """Print result message to console."""
        self._output_func(f"\n{message}\n")


class WebhookSelectionChannel(SelectionChannel):
    """Channel that sends selection requests to an HTTP webhook.

    Designed for integration with external approval systems like
    Slack bots, web dashboards, or custom approval workflows.
    """

    def __init__(self):
        self._endpoint: Optional[str] = None
        self._timeout: int = 300
        self._headers: Dict[str, str] = {}
        self._auth_token: Optional[str] = None

    @property
    def name(self) -> str:
        return "webhook"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize webhook channel.

        Config options:
            endpoint: URL to send requests to (required)
            timeout: Request timeout in seconds (default: 300)
            headers: Additional headers to include
            auth_token: Bearer token for authorization
        """
        if not HAS_REQUESTS:
            raise RuntimeError("requests library required for WebhookSelectionChannel")

        if not config:
            raise ValueError("WebhookSelectionChannel requires configuration with 'endpoint'")

        self._endpoint = config.get("endpoint")
        if not self._endpoint:
            raise ValueError("WebhookSelectionChannel requires 'endpoint' in config")

        self._timeout = config.get("timeout", 300)
        self._headers = config.get("headers", {})
        self._auth_token = config.get("auth_token")

    def present_selection(
        self,
        available_sources: List[ReferenceSource],
        context: Optional[str] = None
    ) -> List[str]:
        """Send selection request to webhook and await response."""
        if not self._endpoint:
            return []

        request = SelectionRequest(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            available_sources=available_sources,
            context=context,
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
                return data.get("selected_ids", [])
            else:
                return []

        except (requests.Timeout, requests.RequestException):
            return []

    def notify_result(self, message: str) -> None:
        """Send result notification to webhook."""
        if not self._endpoint or not HAS_REQUESTS:
            return

        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }

        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        try:
            requests.post(
                self._endpoint,
                json={"type": "selection_result", "message": message},
                headers=headers,
                timeout=10,
            )
        except (requests.Timeout, requests.RequestException):
            pass


class FileSelectionChannel(SelectionChannel):
    """Channel that writes requests to a file and polls for responses.

    Designed for scenarios where a separate process handles selection,
    such as a background service, UI application, or manual file editing.
    """

    def __init__(self):
        self._base_path: Optional[Path] = None
        self._timeout: int = 300
        self._poll_interval: float = 0.5

    @property
    def name(self) -> str:
        return "file"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize file channel.

        Config options:
            base_path: Directory for request/response files (required)
            timeout: Polling timeout in seconds (default: 300)
            poll_interval: Seconds between polls (default: 0.5)
        """
        if not config:
            raise ValueError("FileSelectionChannel requires configuration with 'base_path'")

        base_path = config.get("base_path")
        if not base_path:
            raise ValueError("FileSelectionChannel requires 'base_path' in config")

        self._base_path = Path(base_path)
        self._timeout = config.get("timeout", 300)
        self._poll_interval = config.get("poll_interval", 0.5)

        # Create directories
        (self._base_path / "requests").mkdir(parents=True, exist_ok=True)
        (self._base_path / "responses").mkdir(parents=True, exist_ok=True)

    def present_selection(
        self,
        available_sources: List[ReferenceSource],
        context: Optional[str] = None
    ) -> List[str]:
        """Write request file and poll for response file."""
        if not self._base_path:
            return []

        request = SelectionRequest(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            available_sources=available_sources,
            context=context,
        )

        # Write request file
        request_file = self._base_path / "requests" / f"{request.request_id}.json"
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request.to_dict(), f, indent=2)

        # Poll for response
        response_file = self._base_path / "responses" / f"{request.request_id}.json"
        start_time = time.time()

        while time.time() - start_time < self._timeout:
            if response_file.exists():
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Clean up files
                    request_file.unlink(missing_ok=True)
                    response_file.unlink(missing_ok=True)
                    response = SelectionResponse.from_dict(data)
                    return response.selected_ids
                except (json.JSONDecodeError, IOError):
                    return []

            time.sleep(self._poll_interval)

        # Timeout - clean up request file
        request_file.unlink(missing_ok=True)
        return []

    def notify_result(self, message: str) -> None:
        """Write result to results file."""
        if not self._base_path:
            return

        result_file = self._base_path / "results" / "latest.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "message": message,
            }, f, indent=2)

    def shutdown(self) -> None:
        """Clean up any pending request files."""
        if self._base_path:
            requests_dir = self._base_path / "requests"
            if requests_dir.exists():
                for f in requests_dir.glob("*.json"):
                    f.unlink(missing_ok=True)


def create_channel(channel_type: str, config: Optional[Dict[str, Any]] = None) -> SelectionChannel:
    """Factory function to create an channel by type.

    Args:
        channel_type: One of "console", "webhook", "file"
        config: Optional configuration for the channel

    Returns:
        Initialized SelectionChannel instance

    Raises:
        ValueError: If channel_type is unknown
    """
    channels = {
        "console": ConsoleSelectionChannel,
        "webhook": WebhookSelectionChannel,
        "file": FileSelectionChannel,
    }

    if channel_type not in channels:
        raise ValueError(f"Unknown channel type: {channel_type}. Available: {list(channels.keys())}")

    channel = channels[channel_type]()
    channel.initialize(config)
    return channel
