"""Backend abstraction for RichClient.

This module provides an abstraction layer that allows RichClient to work
with either a direct JaatoClient connection or an IPC server connection.

Usage:
    # Direct mode (non-IPC)
    backend = DirectBackend(jaato_client)

    # IPC mode
    backend = IPCBackend(ipc_client)

    # RichClient uses backend uniformly
    client = RichClient(backend=backend)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Backend(ABC):
    """Abstract backend for RichClient communication.

    This abstraction allows RichClient to work identically with:
    - Direct JaatoClient (non-IPC mode)
    - IPC server connection (IPC mode)
    """

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model name."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Current provider name."""
        ...

    @property
    @abstractmethod
    def is_processing(self) -> bool:
        """Whether a message is being processed."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the backend is connected."""
        ...

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @abstractmethod
    def connect(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Connect to the backend."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the backend."""
        ...

    # =========================================================================
    # Messages
    # =========================================================================

    @abstractmethod
    def send_message(
        self,
        text: str,
        on_output: Optional[Callable[[str, str, str], None]] = None,
        attachments: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Send a message to the model.

        Args:
            text: The message text.
            on_output: Callback for streaming output (source, text, mode).
            attachments: Optional file attachments.

        Returns:
            The response text (for non-streaming), or None for async/streaming.
        """
        ...

    @abstractmethod
    def stop(self) -> bool:
        """Stop current processing.

        Returns:
            True if stop was initiated.
        """
        ...

    # =========================================================================
    # Commands
    # =========================================================================

    @abstractmethod
    def get_user_commands(self) -> Dict[str, Any]:
        """Get available user commands.

        Returns:
            Dict mapping command names to UserCommand objects.
        """
        ...

    @abstractmethod
    def execute_user_command(
        self,
        command_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, bool]:
        """Execute a user command.

        Args:
            command_name: The command to execute.
            args: Command arguments.

        Returns:
            Tuple of (result, shared_with_model).
        """
        ...

    @abstractmethod
    def get_model_completions(self, args: List[str]) -> List[Any]:
        """Get completions for the model command.

        Args:
            args: Arguments typed so far.

        Returns:
            List of CommandCompletion objects.
        """
        ...

    # =========================================================================
    # History & Context
    # =========================================================================

    @abstractmethod
    def get_history(self) -> List[Any]:
        """Get conversation history.

        Returns:
            List of Message objects.
        """
        ...

    @abstractmethod
    def get_context_usage(self) -> Dict[str, Any]:
        """Get current context usage.

        Returns:
            Dict with token counts.
        """
        ...

    @abstractmethod
    def get_context_limit(self) -> int:
        """Get context window limit.

        Returns:
            Maximum tokens allowed.
        """
        ...

    @abstractmethod
    def get_turn_boundaries(self) -> List[int]:
        """Get turn boundary indices.

        Returns:
            List of indices where turns end.
        """
        ...

    @abstractmethod
    def reset_session(self) -> None:
        """Clear conversation history."""
        ...

    # =========================================================================
    # Configuration
    # =========================================================================

    @abstractmethod
    def set_ui_hooks(self, hooks: Any) -> None:
        """Set UI hooks for callbacks."""
        ...

    @abstractmethod
    def configure_tools(
        self,
        registry: Any,
        permission_plugin: Any,
        ledger: Any,
    ) -> None:
        """Configure tools and plugins."""
        ...

    @abstractmethod
    def set_gc_plugin(self, gc_plugin: Any, gc_config: Any) -> None:
        """Set garbage collection plugin."""
        ...

    @abstractmethod
    def set_session_plugin(self, plugin: Any, config: Any) -> None:
        """Set session persistence plugin."""
        ...

    @abstractmethod
    def get_session_plugin(self) -> Optional[Any]:
        """Get the session plugin if configured."""
        ...

    @abstractmethod
    def refresh_tools(self) -> None:
        """Refresh tool definitions."""
        ...


class DirectBackend(Backend):
    """Backend that wraps a JaatoClient for direct (non-IPC) mode."""

    def __init__(self, jaato_client: Any):
        """Initialize with a JaatoClient instance.

        Args:
            jaato_client: The JaatoClient to wrap.
        """
        self._jaato = jaato_client

    @property
    def model_name(self) -> str:
        return self._jaato.model_name or ""

    @property
    def provider_name(self) -> str:
        return self._jaato.provider_name or ""

    @property
    def is_processing(self) -> bool:
        return self._jaato.is_processing if self._jaato else False

    @property
    def is_connected(self) -> bool:
        return self._jaato is not None

    def connect(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        if project_id and location:
            self._jaato.connect(project_id, location, model)
        else:
            self._jaato.connect(model=model)

    def disconnect(self) -> None:
        pass  # Direct mode doesn't need explicit disconnect

    def send_message(
        self,
        text: str,
        on_output: Optional[Callable[[str, str, str], None]] = None,
        attachments: Optional[List[str]] = None,
    ) -> Optional[str]:
        return self._jaato.send_message(text, on_output=on_output)

    def stop(self) -> bool:
        return self._jaato.stop() if self._jaato else False

    def get_user_commands(self) -> Dict[str, Any]:
        return self._jaato.get_user_commands() if self._jaato else {}

    def execute_user_command(
        self,
        command_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, bool]:
        return self._jaato.execute_user_command(command_name, args)

    def get_model_completions(self, args: List[str]) -> List[Any]:
        return self._jaato.get_model_completions(args) if self._jaato else []

    def get_history(self) -> List[Any]:
        return self._jaato.get_history() if self._jaato else []

    def get_context_usage(self) -> Dict[str, Any]:
        return self._jaato.get_context_usage() if self._jaato else {}

    def get_context_limit(self) -> int:
        return self._jaato.get_context_limit() if self._jaato else 0

    def get_turn_boundaries(self) -> List[int]:
        return self._jaato.get_turn_boundaries() if self._jaato else []

    def reset_session(self) -> None:
        if self._jaato:
            self._jaato.reset_session()

    def set_ui_hooks(self, hooks: Any) -> None:
        if self._jaato:
            self._jaato.set_ui_hooks(hooks)

    def configure_tools(
        self,
        registry: Any,
        permission_plugin: Any,
        ledger: Any,
    ) -> None:
        if self._jaato:
            self._jaato.configure_tools(registry, permission_plugin, ledger)

    def set_gc_plugin(self, gc_plugin: Any, gc_config: Any) -> None:
        if self._jaato:
            self._jaato.set_gc_plugin(gc_plugin, gc_config)

    def set_session_plugin(self, plugin: Any, config: Any) -> None:
        if self._jaato:
            self._jaato.set_session_plugin(plugin, config)

    def get_session_plugin(self) -> Optional[Any]:
        if self._jaato and hasattr(self._jaato, '_session_plugin'):
            return self._jaato._session_plugin
        return None

    def refresh_tools(self) -> None:
        if self._jaato and hasattr(self._jaato, '_session') and self._jaato._session:
            self._jaato._session.refresh_tools()


# IPCBackend will be implemented separately as it requires async handling
# and integration with the event-based IPC protocol
