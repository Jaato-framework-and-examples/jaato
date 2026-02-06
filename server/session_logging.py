"""Session-specific logging with ContextVars.

This module provides per-session/client log routing using Python's ContextVars.
Log messages are automatically routed to session-specific files based on the
current execution context.

Configuration:
    Set JAATO_SESSION_LOG_DIR in workspace .env file to customize log location.
    - Relative paths are resolved against the workspace directory
    - Absolute paths are used as-is
    - Default: .jaato/logs (relative to workspace)

Example .env:
    JAATO_SESSION_LOG_DIR=.jaato/logs
    # or absolute:
    JAATO_SESSION_LOG_DIR=/var/log/jaato/myproject
"""

import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Dict, Optional, Any
from contextlib import contextmanager


# Context variables for session routing
session_context: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
client_context: ContextVar[Optional[str]] = ContextVar('client_id', default=None)
workspace_context: ContextVar[Optional[str]] = ContextVar('workspace_path', default=None)
session_env_context: ContextVar[Optional[Dict[str, str]]] = ContextVar('session_env', default=None)

# Environment variable name for log directory configuration
SESSION_LOG_DIR_ENV = "JAATO_SESSION_LOG_DIR"
DEFAULT_SESSION_LOG_DIR = ".jaato/logs"


def set_logging_context(
    session_id: Optional[str] = None,
    client_id: Optional[str] = None,
    workspace_path: Optional[str] = None,
    session_env: Optional[Dict[str, str]] = None,
) -> None:
    """Set the logging context for the current execution context.

    Args:
        session_id: The session ID for log routing.
        client_id: The client ID for log routing.
        workspace_path: The workspace path for resolving relative log paths.
        session_env: Session-specific environment variables (from workspace .env).
    """
    if session_id is not None:
        session_context.set(session_id)
    if client_id is not None:
        client_context.set(client_id)
    if workspace_path is not None:
        workspace_context.set(workspace_path)
    if session_env is not None:
        session_env_context.set(session_env)


def clear_logging_context() -> None:
    """Clear the logging context."""
    session_context.set(None)
    client_context.set(None)
    workspace_context.set(None)
    session_env_context.set(None)


def get_logging_context() -> Dict[str, Any]:
    """Get the current logging context.

    Returns:
        Dict with session_id, client_id, workspace_path, session_env keys.
    """
    return {
        'session_id': session_context.get(),
        'client_id': client_context.get(),
        'workspace_path': workspace_context.get(),
        'session_env': session_env_context.get(),
    }


@contextmanager
def logging_context(
    session_id: Optional[str] = None,
    client_id: Optional[str] = None,
    workspace_path: Optional[str] = None,
    session_env: Optional[Dict[str, str]] = None,
):
    """Context manager for setting logging context.

    Usage:
        with logging_context(session_id="20251205_143022", client_id="ipc_1"):
            logger.info("This goes to session log")

    Args:
        session_id: The session ID for log routing.
        client_id: The client ID for log routing.
        workspace_path: The workspace path for resolving relative log paths.
        session_env: Session-specific environment variables.
    """
    # Save previous context
    prev_session = session_context.get()
    prev_client = client_context.get()
    prev_workspace = workspace_context.get()
    prev_env = session_env_context.get()

    try:
        set_logging_context(session_id, client_id, workspace_path, session_env)
        yield
    finally:
        # Restore previous context
        session_context.set(prev_session)
        client_context.set(prev_client)
        workspace_context.set(prev_workspace)
        session_env_context.set(prev_env)


class SessionContextFilter(logging.Filter):
    """Logging filter that adds session/client context to log records.

    This filter adds the following attributes to log records:
    - session_id: Current session ID or empty string
    - client_id: Current client ID or empty string
    - workspace_path: Current workspace path or empty string
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context attributes to the log record."""
        record.session_id = session_context.get() or ""
        record.client_id = client_context.get() or ""
        record.workspace_path = workspace_context.get() or ""
        return True


class SessionRoutingHandler(logging.Handler):
    """Logging handler that routes logs to per-session files.

    Logs are written to files named:
        {log_dir}/session_{session_id}_client_{client_id}.log

    The log directory is determined by:
    1. JAATO_SESSION_LOG_DIR from session's .env file
    2. JAATO_SESSION_LOG_DIR from global environment
    3. Default: .jaato/logs (relative to workspace)

    Relative paths are resolved against the workspace directory.
    """

    def __init__(
        self,
        level: int = logging.NOTSET,
        formatter: Optional[logging.Formatter] = None,
    ):
        """Initialize the session routing handler.

        Args:
            level: Minimum log level to handle.
            formatter: Log formatter to use. Defaults to standard format with context.
        """
        super().__init__(level)
        self._handlers: Dict[str, logging.FileHandler] = {}
        self._formatter = formatter or logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        self.setFormatter(self._formatter)

    def _get_log_dir(self, workspace: str, session_env: Optional[Dict[str, str]]) -> Path:
        """Get the log directory from configuration.

        Args:
            workspace: The workspace path.
            session_env: Session-specific environment variables.

        Returns:
            Absolute path to log directory.
        """
        # Priority: session_env -> global env -> default
        log_dir = None

        if session_env:
            log_dir = session_env.get(SESSION_LOG_DIR_ENV)

        if not log_dir:
            log_dir = os.environ.get(SESSION_LOG_DIR_ENV)

        if not log_dir:
            log_dir = DEFAULT_SESSION_LOG_DIR

        # Resolve relative paths against workspace
        path = Path(log_dir)
        if not path.is_absolute():
            path = Path(workspace) / path

        return path

    def _get_handler_key(self, session_id: str, client_id: str) -> str:
        """Generate a unique key for the handler cache.

        Args:
            session_id: The session ID.
            client_id: The client ID.

        Returns:
            Cache key string.
        """
        return f"{session_id}_{client_id}"

    def _get_or_create_handler(
        self,
        session_id: str,
        client_id: str,
        workspace: str,
        session_env: Optional[Dict[str, str]],
    ) -> Optional[logging.FileHandler]:
        """Get or create a file handler for the session/client.

        Args:
            session_id: The session ID.
            client_id: The client ID.
            workspace: The workspace path.
            session_env: Session-specific environment variables.

        Returns:
            FileHandler for the session/client, or None if creation fails.
        """
        key = self._get_handler_key(session_id, client_id)

        if key not in self._handlers:
            try:
                log_dir = self._get_log_dir(workspace, session_env)
                log_dir.mkdir(parents=True, exist_ok=True)

                log_file = log_dir / f"session_{session_id}_client_{client_id}.log"
                handler = logging.FileHandler(log_file, encoding='utf-8')
                handler.setFormatter(self._formatter)
                handler.setLevel(self.level)
                self._handlers[key] = handler

            except Exception as e:
                # Don't let logging errors break the application
                # Log to stderr as fallback
                import sys
                print(
                    f"[SessionRoutingHandler] Failed to create log file: {e}",
                    file=sys.stderr
                )
                return None

        return self._handlers[key]

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the appropriate session file.

        If no session context is set, the record is silently dropped
        (the central log handler will still receive it).

        Args:
            record: The log record to emit.
        """
        session_id = session_context.get()
        client_id = client_context.get()
        workspace = workspace_context.get()
        session_env = session_env_context.get()

        # Skip if no session context
        if not session_id or not workspace:
            return

        # Use "unknown" for client_id if not set
        if not client_id:
            client_id = "unknown"

        handler = self._get_or_create_handler(
            session_id, client_id, workspace, session_env
        )

        if handler:
            handler.emit(record)

    def close(self) -> None:
        """Close all file handlers."""
        for handler in self._handlers.values():
            try:
                handler.close()
            except Exception:
                pass
        self._handlers.clear()
        super().close()

    def close_session(self, session_id: str) -> None:
        """Close handlers for a specific session.

        Call this when a session ends to release file handles.

        Args:
            session_id: The session ID to close handlers for.
        """
        keys_to_remove = [
            key for key in self._handlers
            if key.startswith(f"{session_id}_")
        ]

        for key in keys_to_remove:
            try:
                self._handlers[key].close()
            except Exception:
                pass
            del self._handlers[key]


# Global session routing handler instance (set during server init)
_session_handler: Optional[SessionRoutingHandler] = None


def get_session_handler() -> Optional[SessionRoutingHandler]:
    """Get the global session routing handler."""
    return _session_handler


def set_session_handler(handler: Optional[SessionRoutingHandler]) -> None:
    """Set the global session routing handler."""
    global _session_handler
    _session_handler = handler


def configure_session_logging(
    level: int = logging.DEBUG,
    formatter: Optional[logging.Formatter] = None,
) -> SessionRoutingHandler:
    """Configure and install the session routing handler.

    This adds the SessionRoutingHandler to the root logger, enabling
    automatic per-session log routing based on ContextVars.

    Args:
        level: Minimum log level for session logs.
        formatter: Optional custom formatter.

    Returns:
        The configured SessionRoutingHandler.
    """
    handler = SessionRoutingHandler(level=level, formatter=formatter)

    # Add to root logger
    root = logging.getLogger()
    root.addHandler(handler)

    # Store globally for access
    set_session_handler(handler)

    return handler
