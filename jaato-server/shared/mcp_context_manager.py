"""
MCP Multi-Server Client Manager

A clean architecture for managing multiple simultaneous MCP server connections.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence, TextIO
from contextlib import asynccontextmanager
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, CallToolResult, Implementation

from shared.secret_scrub import scrub_env

logger = logging.getLogger(__name__)

# Reconnection policy when an MCP server is found dead at call time.
# Mirrors the philosophy of retry_utils.with_retry: a small handful of
# attempts with exponential backoff, then mark the server as failed so
# subsequent calls fail fast instead of repeatedly retrying a dead one.
_RECONNECT_MAX_ATTEMPTS = 3
_RECONNECT_BASE_DELAY = 1.0  # seconds
_RECONNECT_MAX_DELAY = 8.0  # seconds

# Type alias for progress callback - matches MCP SDK's ProgressFnT protocol
# Callable[[progress: float, total: float | None, message: str | None], Awaitable[None]]
ProgressCallback = Any  # Avoid import complexity, validated at runtime

# Client info sent to MCP servers during initialization
_CLIENT_INFO = Implementation(name="jaato", version="0.1.0")


class MCPServerUnavailableError(RuntimeError):
    """Raised when an MCP server cannot be reached after retries.

    Distinct from KeyError so callers can produce a meaningful error
    message ("server X is unreachable") instead of a confusing
    "Server 'X' not connected" KeyError that surfaces from registry
    lookups when the server died mid-session.
    """


@dataclass
class ServerConfig:
    """Configuration for an MCP server.

    ``scrub_secret_env`` holds operator-declared name globs (case-insensitive
    fnmatch) for secrets that must NOT be inherited by the server subprocess.
    The MCP server is model-invokable, possibly third-party code named in
    ``.mcp.json``; without scrubbing it inherits the runner's full ``os.environ``
    (provider key, tokens) and can exfiltrate them.  See :mod:`shared.secret_scrub`.
    """
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    scrub_secret_env: Sequence[str] = ()

    def to_stdio_params(self) -> StdioServerParameters:
        # Scrub declared secrets from the INHERITED environment only; the
        # per-server ``env`` (an explicit operator grant in .mcp.json) is
        # overlaid afterwards and never scrubbed — so a token the operator
        # deliberately hands this server still reaches it, while framework
        # secrets it was never granted do not leak in.
        inherited = scrub_env(os.environ, self.scrub_secret_env)
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env={**inherited, **(self.env or {})},
        )


@dataclass
class ServerConnection:
    """Holds an active connection to an MCP server.

    Tracks the per-connection async context managers so the connection
    can be cleanly torn down on disconnect or reconnect — without
    leaking the underlying subprocess or stdio handles.
    """
    config: ServerConfig
    session: ClientSession
    tools: list[Tool] = field(default_factory=list)
    # Async contexts owned by this connection, in entry order.
    # On disconnect we exit them in reverse order.
    contexts: list[Any] = field(default_factory=list)
    # Set to True when the server has been observed dead and reconnect
    # attempts have all failed.  Subsequent get_connection calls will
    # raise MCPServerUnavailableError instead of returning a stale ref.
    failed: bool = False
    # Timestamp of last failure (used for trace/log context).
    failed_at: float | None = None

    async def refresh_tools(self) -> list[Tool]:
        """Refresh the cached tool list."""
        result = await self.session.list_tools()
        self.tools = result.tools
        return self.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] = None,
        progress_callback: ProgressCallback = None,
    ) -> CallToolResult:
        """Call a tool on this server.

        Args:
            name: Name of the tool to call.
            arguments: Arguments to pass to the tool.
            progress_callback: Optional async callback for progress notifications.
                Signature: async def callback(progress: float, total: float | None, message: str | None) -> None

        Returns:
            CallToolResult from the MCP server.
        """
        return await self.session.call_tool(
            name,
            arguments or {},
            progress_callback=progress_callback,
        )

    async def aclose(self) -> None:
        """Close all contexts owned by this connection.

        Each context is closed in reverse order, with errors caught and
        logged so a single bad cleanup doesn't block the rest.
        """
        for ctx in reversed(self.contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except (Exception, asyncio.CancelledError) as exc:
                logger.debug(
                    "MCP connection '%s' context cleanup error: %s",
                    self.config.name, exc,
                )
        self.contexts.clear()


class MCPClientManager:
    """
    Manages multiple persistent MCP server connections.

    Usage:
        async with MCPClientManager() as manager:
            await manager.connect("atlassian", "mcp-atlassian")
            await manager.connect("github", "mcp-github")

            # Call tools
            result = await manager.call_tool("atlassian", "search_issues", {"query": "..."})

            # Or get session directly
            session = manager.get_session("github")
            await session.call_tool(...)

    Args:
        errlog: File-like object to receive server stderr output.
                Defaults to sys.stderr. Pass a custom TextIO to capture
                server initialization messages.
    """

    def __init__(
        self,
        errlog: TextIO | None = None,
        scrub_secret_env: Sequence[str] = (),
    ):
        self._connections: dict[str, ServerConnection] = {}
        self._task_group = None
        self._errlog = errlog if errlog is not None else sys.stderr
        # Operator-declared secret name globs stripped from every server
        # subprocess's inherited environment (see ServerConfig.to_stdio_params).
        # Defensively normalize the public knob: a lone string is a 1-item tuple
        # (NOT split into characters, which would disable scrubbing), and every
        # entry is coerced to str (a non-str would raise in fnmatch).
        if isinstance(scrub_secret_env, str):
            scrub_secret_env = (scrub_secret_env,)
        self._scrub_secret_env: Sequence[str] = tuple(
            str(p) for p in (scrub_secret_env or ())
        )

    @property
    def servers(self) -> list[str]:
        """List of connected server names."""
        return list(self._connections.keys())

    def get_connection(self, name: str) -> ServerConnection:
        """Get a server connection by name.

        Raises:
            KeyError: If the server has never been connected.
            MCPServerUnavailableError: If the server has been marked
                failed (reconnection attempts exhausted).
        """
        if name not in self._connections:
            raise KeyError(f"Server '{name}' not connected")
        conn = self._connections[name]
        if conn.failed:
            raise MCPServerUnavailableError(
                f"MCP server '{name}' is unavailable (reconnection failed)"
            )
        return conn

    def get_session(self, name: str) -> ClientSession:
        """Get a session by server name."""
        return self.get_connection(name).session

    async def connect(
        self,
        name: str,
        command: str,
        args: list[str] = None,
        env: dict[str, str] = None,
    ) -> ServerConnection:
        """Connect to an MCP server.

        On any failure during stdio_client → ClientSession → initialize
        → refresh_tools, **all partial contexts are cleaned up** so we
        don't leak subprocesses or stdio handles.  This is the fix for
        the previous behavior where a failed `session_ctx.__aenter__()`
        would leave `stdio_ctx` orphaned in the manager's context list.

        Args:
            name: Unique identifier for this connection
            command: Command to run the server
            args: Command arguments
            env: Additional environment variables

        Raises:
            ValueError: If a server with this name is already connected.
            Exception: Whatever the underlying connect/initialize raised
                (after partial-state cleanup).
        """
        if name in self._connections:
            raise ValueError(f"Server '{name}' already connected")

        config = ServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env,
            scrub_secret_env=self._scrub_secret_env,
        )

        return await self._do_connect(config)

    async def _do_connect(self, config: ServerConfig) -> ServerConnection:
        """Perform the actual connection sequence with cleanup on failure.

        Builds a fresh ServerConnection and registers it on success.
        On any exception, exits all entered contexts in reverse order
        before re-raising — so partial state never leaks.
        """
        contexts: list[Any] = []
        try:
            # Enter the stdio_client context with custom errlog
            stdio_ctx = stdio_client(config.to_stdio_params(), errlog=self._errlog)
            read, write = await stdio_ctx.__aenter__()
            contexts.append(stdio_ctx)

            # Enter the session context
            session_ctx = ClientSession(read, write, client_info=_CLIENT_INFO)
            session = await session_ctx.__aenter__()
            contexts.append(session_ctx)

            # Initialize the session
            await session.initialize()

            # Create connection object
            connection = ServerConnection(
                config=config, session=session, contexts=contexts,
            )
            await connection.refresh_tools()
        except Exception as exc:
            # Cleanup any contexts we managed to enter before failing.
            # Run in reverse order; ignore secondary cleanup errors so
            # the original exception is the one that propagates.
            logger.warning(
                "MCP server '%s' connect failed: %s. "
                "Cleaning up partial state.",
                config.name, exc,
            )
            for ctx in reversed(contexts):
                try:
                    await ctx.__aexit__(type(exc), exc, exc.__traceback__)
                except (Exception, asyncio.CancelledError):
                    pass
            raise

        self._connections[config.name] = connection
        return connection

    async def disconnect(self, name: str) -> None:
        """Disconnect from a specific server and clean up its contexts."""
        conn = self._connections.pop(name, None)
        if conn is None:
            return
        await conn.aclose()

    async def reconnect(self, name: str) -> ServerConnection:
        """Reconnect a server that was previously connected.

        Used when ``call_tool`` discovers the server is dead.  Tries
        a small number of attempts with exponential backoff; on
        success, replaces the old (failed) connection.  On exhaustion,
        marks the connection as failed and raises
        ``MCPServerUnavailableError``.
        """
        old_conn = self._connections.get(name)
        if old_conn is None:
            raise KeyError(f"Cannot reconnect unknown server '{name}'")

        config = old_conn.config

        # Best-effort cleanup of the dead connection's contexts before
        # we try to reconnect.
        try:
            await old_conn.aclose()
        except Exception:
            pass
        # Remove from registry so _do_connect can re-register on success.
        self._connections.pop(name, None)

        last_exc: Exception | None = None
        for attempt in range(1, _RECONNECT_MAX_ATTEMPTS + 1):
            delay = min(
                _RECONNECT_BASE_DELAY * (2 ** (attempt - 1)),
                _RECONNECT_MAX_DELAY,
            )
            try:
                logger.info(
                    "MCP server '%s' reconnection attempt %d/%d",
                    name, attempt, _RECONNECT_MAX_ATTEMPTS,
                )
                return await self._do_connect(config)
            except Exception as exc:
                last_exc = exc
                if attempt < _RECONNECT_MAX_ATTEMPTS:
                    logger.warning(
                        "MCP server '%s' reconnect attempt %d failed: %s. "
                        "Retrying in %.1fs",
                        name, attempt, exc, delay,
                    )
                    await asyncio.sleep(delay)

        # All attempts exhausted — install a failed sentinel so future
        # get_connection() calls fail fast with a clear error.
        sentinel = ServerConnection(
            config=config,
            session=None,  # type: ignore[arg-type]
            failed=True,
            failed_at=time.time(),
        )
        self._connections[name] = sentinel
        logger.error(
            "MCP server '%s' marked as failed after %d reconnection attempts",
            name, _RECONNECT_MAX_ATTEMPTS,
        )
        raise MCPServerUnavailableError(
            f"MCP server '{name}' is unavailable after "
            f"{_RECONNECT_MAX_ATTEMPTS} reconnection attempts: {last_exc}"
        )

    async def call_tool(
        self,
        server: str,
        tool_name: str,
        arguments: dict[str, Any] = None,
        progress_callback: ProgressCallback = None,
    ) -> CallToolResult:
        """Call a tool on a specific server.

        If the server is found dead (the underlying session call raises
        a connection-level error), attempts a reconnection with backoff
        and retries the call once on the fresh session.  If reconnection
        fails, raises ``MCPServerUnavailableError`` so callers can
        report the failure cleanly instead of getting a confusing
        ``KeyError`` or low-level transport exception.

        Args:
            server: Name of the server to call the tool on.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.
            progress_callback: Optional async callback for progress notifications.

        Returns:
            CallToolResult from the MCP server.
        """
        connection = self.get_connection(server)
        try:
            return await connection.call_tool(
                tool_name, arguments, progress_callback=progress_callback,
            )
        except (ConnectionError, BrokenPipeError, asyncio.IncompleteReadError) as exc:
            logger.warning(
                "MCP server '%s' connection lost during call_tool('%s'): %s. "
                "Attempting reconnection.",
                server, tool_name, exc,
            )
            new_conn = await self.reconnect(server)  # may raise MCPServerUnavailableError
            return await new_conn.call_tool(
                tool_name, arguments, progress_callback=progress_callback,
            )
    
    async def find_tool(self, tool_name: str) -> tuple[str, Tool] | None:
        """Find which server has a given tool."""
        for name, conn in self._connections.items():
            for tool in conn.tools:
                if tool.name == tool_name:
                    return (name, tool)
        return None
    
    async def call_tool_auto(
        self,
        tool_name: str,
        arguments: dict[str, Any] = None,
        progress_callback: ProgressCallback = None,
    ) -> CallToolResult:
        """Call a tool, automatically finding which server has it.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.
            progress_callback: Optional async callback for progress notifications.
                Signature: async def callback(progress: float, total: float | None, message: str | None) -> None

        Returns:
            CallToolResult from the MCP server.
        """
        result = await self.find_tool(tool_name)
        if not result:
            raise ValueError(f"Tool '{tool_name}' not found on any server")
        server_name, _ = result
        return await self.call_tool(
            server_name, tool_name, arguments, progress_callback=progress_callback
        )
    
    def all_tools(self) -> dict[str, list[Tool]]:
        """Get all tools from all servers."""
        return {name: conn.tools for name, conn in self._connections.items()}
    
    async def __aenter__(self) -> "MCPClientManager":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Close all contexts in reverse order with proper cleanup
        # On Windows, subprocess cleanup needs special handling to avoid pipe errors

        # Give pending operations a chance to complete
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass  # Event loop may be shutting down

        # Cancel any pending tasks in the current event loop
        if sys.version_info >= (3, 11):
            try:
                tasks = [t for t in asyncio.all_tasks() if not t.done()]
                if tasks:
                    for task in tasks:
                        task.cancel()
                    # Wait briefly for cancellation to complete
                    await asyncio.sleep(0.05)
            except (Exception, asyncio.CancelledError):
                pass  # Ignore errors during task cleanup

        # Close every live connection's contexts.  Each ServerConnection
        # owns its own context list (per-connection tracking is the
        # post-Tier-3 model — the previous global _contexts list could
        # leak partial state on failed connects).
        for conn in list(self._connections.values()):
            if conn.failed:
                continue  # Failed sentinels have no live contexts
            for ctx in reversed(conn.contexts):
                try:
                    await ctx.__aexit__(exc_type, exc_val, exc_tb)
                except (Exception, asyncio.CancelledError):
                    # Silently ignore cleanup errors — typically just
                    # resource warnings from subprocess cleanup on Windows.
                    pass

        # On Windows, give subprocess transports time to finish cleanup
        if sys.platform == 'win32':
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass  # Event loop may be shutting down

        self._connections.clear()

