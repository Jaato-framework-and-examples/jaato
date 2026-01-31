"""Backend abstraction for RichClient.

This module provides an async abstraction layer that allows RichClient to work
with either a direct JaatoClient connection or an IPC server connection.

All operations are async to support both:
- Direct mode: sync JaatoClient calls wrapped with asyncio.to_thread
- IPC mode: native async event-based communication

Usage:
    # Direct mode (non-IPC)
    backend = DirectBackend(jaato_client)

    # IPC mode (with automatic recovery)
    backend = IPCBackend(
        ipc_client,
        workspace_path=Path.cwd(),
        on_connection_status=lambda status: print(status.state),
    )

    # RichClient uses backend uniformly (async)
    await backend.send_message("Hello")
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ipc_recovery import ConnectionStatus, StatusCallback

logger = logging.getLogger(__name__)


@dataclass
class CommandInfo:
    """Command information for completion and routing."""
    name: str
    description: str


class Backend(ABC):
    """Abstract async backend for RichClient communication.

    This abstraction allows RichClient to work identically with:
    - Direct JaatoClient (non-IPC mode) - sync calls wrapped as async
    - IPC server connection (IPC mode) - native async
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
    async def connect(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Connect to the backend."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the backend."""
        ...

    # =========================================================================
    # Messages
    # =========================================================================

    @abstractmethod
    async def send_message(
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
            The response text, or None for streaming mode.
        """
        ...

    @abstractmethod
    async def stop(self) -> bool:
        """Stop current processing.

        Returns:
            True if stop was initiated.
        """
        ...

    # =========================================================================
    # Commands
    # =========================================================================

    @abstractmethod
    async def get_user_commands(self) -> Dict[str, Any]:
        """Get available user commands.

        Returns:
            Dict mapping command names to UserCommand objects.
        """
        ...

    @abstractmethod
    async def execute_user_command(
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
    async def get_model_completions(self, args: List[str]) -> List[Any]:
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
    async def get_history(self) -> List[Any]:
        """Get conversation history.

        Returns:
            List of Message objects.
        """
        ...

    @abstractmethod
    async def get_context_usage(self) -> Dict[str, Any]:
        """Get current context usage.

        Returns:
            Dict with token counts.
        """
        ...

    @abstractmethod
    async def get_context_limit(self) -> int:
        """Get context window limit.

        Returns:
            Maximum tokens allowed.
        """
        ...

    @abstractmethod
    async def get_turn_boundaries(self) -> List[int]:
        """Get turn boundary indices.

        Returns:
            List of indices where turns end.
        """
        ...

    @abstractmethod
    async def reset_session(self) -> None:
        """Clear conversation history."""
        ...

    # =========================================================================
    # Configuration
    # =========================================================================

    @abstractmethod
    def set_ui_hooks(self, hooks: Any) -> None:
        """Set UI hooks for callbacks (sync - called once at init)."""
        ...

    @abstractmethod
    def configure_tools(
        self,
        registry: Any,
        permission_plugin: Any,
        ledger: Any,
    ) -> None:
        """Configure tools and plugins (sync - called once at init)."""
        ...

    @abstractmethod
    def set_gc_plugin(self, gc_plugin: Any, gc_config: Any) -> None:
        """Set garbage collection plugin (sync - called once at init)."""
        ...

    @abstractmethod
    def set_session_plugin(self, plugin: Any, config: Any) -> None:
        """Set session persistence plugin (sync - called once at init)."""
        ...

    @abstractmethod
    def get_session_plugin(self) -> Optional[Any]:
        """Get the session plugin if configured."""
        ...

    @abstractmethod
    async def refresh_tools(self) -> None:
        """Refresh tool definitions."""
        ...

    # =========================================================================
    # Direct Access (for features not easily abstracted)
    # =========================================================================

    @abstractmethod
    def get_session(self) -> Optional[Any]:
        """Get the underlying session if available.

        Returns:
            JaatoSession for DirectBackend, None for IPCBackend.
        """
        ...

    @abstractmethod
    def get_client(self) -> Optional[Any]:
        """Get the underlying client if available.

        Returns:
            JaatoClient for DirectBackend, IPCClient for IPCBackend.
        """
        ...

    @abstractmethod
    def set_retry_callback(self, callback: Optional[Callable]) -> None:
        """Set callback for retry notifications."""
        ...


class DirectBackend(Backend):
    """Backend that wraps a JaatoClient for direct (non-IPC) mode.

    Sync JaatoClient calls are wrapped with asyncio.to_thread to make them async.
    """

    def __init__(self, jaato_client: Any):
        """Initialize with a JaatoClient instance.

        Args:
            jaato_client: The JaatoClient to wrap.
        """
        self._jaato = jaato_client
        self._model_name = ""
        self._provider_name = ""

    @property
    def model_name(self) -> str:
        return self._jaato.model_name or self._model_name if self._jaato else self._model_name

    @property
    def provider_name(self) -> str:
        return self._jaato.provider_name or self._provider_name if self._jaato else self._provider_name

    @property
    def is_processing(self) -> bool:
        return self._jaato.is_processing if self._jaato else False

    @property
    def is_connected(self) -> bool:
        return self._jaato is not None

    async def connect(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        def _connect():
            if project_id and location:
                self._jaato.connect(project_id, location, model)
            else:
                self._jaato.connect(model=model)

        await asyncio.to_thread(_connect)
        self._model_name = self._jaato.model_name or model or ""
        self._provider_name = self._jaato.provider_name or ""

    async def disconnect(self) -> None:
        pass  # Direct mode doesn't need explicit disconnect

    async def send_message(
        self,
        text: str,
        on_output: Optional[Callable[[str, str, str], None]] = None,
        attachments: Optional[List[str]] = None,
    ) -> Optional[str]:
        def _send():
            return self._jaato.send_message(text, on_output=on_output)

        return await asyncio.to_thread(_send)

    async def stop(self) -> bool:
        if self._jaato:
            return self._jaato.stop()
        return False

    async def get_user_commands(self) -> Dict[str, Any]:
        if not self._jaato:
            return {}
        return self._jaato.get_user_commands()

    async def execute_user_command(
        self,
        command_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, bool]:
        def _execute():
            return self._jaato.execute_user_command(command_name, args)

        return await asyncio.to_thread(_execute)

    async def get_model_completions(self, args: List[str]) -> List[Any]:
        if not self._jaato:
            return []
        return self._jaato.get_model_completions(args)

    async def get_history(self) -> List[Any]:
        if not self._jaato:
            return []
        return self._jaato.get_history()

    async def get_context_usage(self) -> Dict[str, Any]:
        if not self._jaato:
            return {}
        return self._jaato.get_context_usage()

    async def get_context_limit(self) -> int:
        if not self._jaato:
            return 0
        return self._jaato.get_context_limit()

    async def get_turn_boundaries(self) -> List[int]:
        if not self._jaato:
            return []
        return self._jaato.get_turn_boundaries()

    async def reset_session(self) -> None:
        if self._jaato:
            self._jaato.reset_session()

    def set_ui_hooks(self, hooks: Any) -> None:
        if self._jaato:
            self._jaato.set_ui_hooks(hooks)

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Verify authentication before loading tools.

        Args:
            allow_interactive: If True, allow interactive login if needed.
            on_message: Optional callback for status messages.

        Returns:
            True if authentication is configured.
        """
        if self._jaato:
            return self._jaato.verify_auth(allow_interactive, on_message)
        return False

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

    async def refresh_tools(self) -> None:
        if self._jaato and hasattr(self._jaato, '_session') and self._jaato._session:
            self._jaato._session.refresh_tools()

    def get_session(self) -> Optional[Any]:
        if self._jaato:
            return self._jaato.get_session()
        return None

    def get_client(self) -> Optional[Any]:
        return self._jaato

    def set_retry_callback(self, callback: Optional[Callable]) -> None:
        if self._jaato:
            session = self._jaato.get_session()
            if session:
                session.set_retry_callback(callback)


class IPCBackend(Backend):
    """Backend that wraps an IPCClient or IPCRecoveryClient for IPC mode.

    Uses native async event-based communication with the server.
    When use_recovery=True (default), wraps the client with IPCRecoveryClient
    for automatic reconnection on connection loss.
    """

    def __init__(
        self,
        ipc_client: Any,
        use_recovery: bool = True,
        workspace_path: Optional[Path] = None,
        on_connection_status: Optional["StatusCallback"] = None,
    ):
        """Initialize with an IPCClient instance.

        Args:
            ipc_client: The IPCClient to wrap.
            use_recovery: Whether to use automatic recovery (default: True).
                When True, wraps the client with IPCRecoveryClient.
            workspace_path: Workspace path for loading recovery config.
                Only used when use_recovery=True.
            on_connection_status: Callback for connection status changes.
                Only used when use_recovery=True.
        """
        self._raw_client = ipc_client
        self._client = ipc_client  # May be replaced with recovery client
        self._use_recovery = use_recovery
        self._workspace_path = workspace_path
        self._on_connection_status = on_connection_status
        self._recovery_client: Optional[Any] = None  # IPCRecoveryClient if used

        self._model_name = ""
        self._provider_name = ""
        self._is_processing = False
        self._user_commands: Dict[str, Any] = {}
        self._history: List[Any] = []
        self._context_usage: Dict[str, Any] = {}
        self._context_limit: int = 0
        self._available_models: List[str] = []  # Models from server for completion

        # Event handlers for async responses
        self._pending_history: Optional[asyncio.Future] = None
        self._pending_command_result: Optional[asyncio.Future] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def is_processing(self) -> bool:
        return self._is_processing

    @property
    def is_connected(self) -> bool:
        if self._recovery_client:
            return self._recovery_client.is_connected
        return self._client.is_connected if self._client else False

    @property
    def is_reconnecting(self) -> bool:
        """Check if the client is currently reconnecting."""
        if self._recovery_client:
            return self._recovery_client.is_reconnecting
        return False

    @property
    def connection_status(self) -> Optional["ConnectionStatus"]:
        """Get current connection status for UI display.

        Returns:
            ConnectionStatus if using recovery client, None otherwise.
        """
        if self._recovery_client:
            return self._recovery_client.get_status()
        return None

    def update_model_info(self, provider: str, model: str) -> None:
        """Update model info from server event."""
        self._provider_name = provider
        self._model_name = model

    def set_processing(self, processing: bool) -> None:
        """Update processing state from server event."""
        self._is_processing = processing

    def update_context(self, usage: Dict[str, Any], limit: int) -> None:
        """Update context info from server event."""
        self._context_usage = usage
        self._context_limit = limit

    def set_commands(self, commands: List[Dict[str, str]]) -> None:
        """Set available commands from server event."""
        self._user_commands = {
            cmd.get("name", ""): CommandInfo(
                name=cmd.get("name", ""),
                description=cmd.get("description", ""),
            )
            for cmd in commands
        }

    def set_models(self, models: List[str]) -> None:
        """Set available models from server event."""
        self._available_models = models

    async def connect(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        if self._use_recovery:
            # Import here to avoid circular imports
            from ipc_recovery import IPCRecoveryClient

            # Create recovery client wrapping the raw client's socket path
            self._recovery_client = IPCRecoveryClient(
                socket_path=self._raw_client.socket_path,
                auto_start=self._raw_client.auto_start,
                env_file=self._raw_client.env_file,
                workspace_path=self._workspace_path,
                on_status_change=self._on_connection_status,
            )
            self._client = self._recovery_client
            await self._recovery_client.connect()
        else:
            await self._client.connect()

    async def disconnect(self) -> None:
        if self._recovery_client:
            await self._recovery_client.close()
        else:
            await self._client.disconnect()

    async def send_message(
        self,
        text: str,
        on_output: Optional[Callable[[str, str, str], None]] = None,
        attachments: Optional[List[str]] = None,
    ) -> Optional[str]:
        # IPC mode uses event-based streaming, output comes via events
        await self._client.send_message(text, attachments)
        return None  # Response comes via events

    async def stop(self) -> bool:
        await self._client.stop()
        return True

    async def get_user_commands(self) -> Dict[str, Any]:
        return self._user_commands

    async def execute_user_command(
        self,
        command_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, bool]:
        # Convert args dict to list for IPC
        args_list = []
        if args:
            for key, value in args.items():
                if value is not None:
                    args_list.append(str(value))

        await self._client.execute_command(command_name, args_list)
        # Result comes via SystemMessageEvent
        return {}, False

    async def get_model_completions(self, args: List[str]) -> List[Any]:
        # Return models from server state snapshot
        # Format: (model_name, description) tuples for completion
        return [(model, "") for model in self._available_models]

    async def get_history(self) -> List[Any]:
        # Request history from server
        await self._client.request_history()
        # History comes via HistoryEvent - caller should handle the event
        return self._history

    async def get_context_usage(self) -> Dict[str, Any]:
        return self._context_usage

    async def get_context_limit(self) -> int:
        return self._context_limit

    async def get_turn_boundaries(self) -> List[int]:
        # Would need to be computed from history
        return []

    async def reset_session(self) -> None:
        await self._client.execute_command("reset", [])

    def set_ui_hooks(self, hooks: Any) -> None:
        pass  # IPC mode uses events, not hooks

    def configure_tools(
        self,
        registry: Any,
        permission_plugin: Any,
        ledger: Any,
    ) -> None:
        pass  # Server handles tool configuration

    def set_gc_plugin(self, gc_plugin: Any, gc_config: Any) -> None:
        pass  # Server handles GC

    def set_session_plugin(self, plugin: Any, config: Any) -> None:
        pass  # Server handles sessions

    def get_session_plugin(self) -> Optional[Any]:
        return None  # Server handles sessions

    async def refresh_tools(self) -> None:
        pass  # Server handles tool refresh

    def get_session(self) -> Optional[Any]:
        return None  # IPC mode doesn't have direct session access

    def get_client(self) -> Optional[Any]:
        """Get the underlying client.

        Returns:
            IPCRecoveryClient if recovery is enabled, IPCClient otherwise.
        """
        return self._client

    def get_raw_client(self) -> Optional[Any]:
        """Get the raw IPCClient (without recovery wrapper).

        Returns:
            The original IPCClient instance.
        """
        return self._raw_client

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for recovery reattachment.

        Should be called when a session is attached so the recovery client
        knows which session to reattach to after reconnection.

        Args:
            session_id: The session ID to track.
        """
        if self._recovery_client:
            self._recovery_client.set_session_id(session_id)

    def set_retry_callback(self, callback: Optional[Callable]) -> None:
        pass  # IPC mode handles retries via events
