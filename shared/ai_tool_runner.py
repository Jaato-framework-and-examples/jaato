"""Tool execution infrastructure for the jaato framework.

This module provides the ToolExecutor class for managing tool/function
execution with support for:
- Permission checking via PermissionPlugin
- Auto-backgrounding for long-running tasks
- Output callbacks for real-time feedback
"""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

from shared.token_accounting import TokenLedger
from shared.plugins.base import OutputCallback

if TYPE_CHECKING:
    from shared.plugins.registry import PluginRegistry
    from shared.plugins.permission import PermissionPlugin
    from shared.plugins.background.protocol import BackgroundCapable


class ToolExecutor:
    """Registry mapping tool names to callables.

    Executors should accept a single dict-like argument and return a JSON-serializable result.

    Supports optional permission checking via a PermissionPlugin. When a permission
    plugin is set, all tool executions are checked against the permission policy
    before execution.

    Supports auto-backgrounding for BackgroundCapable plugins. When a tool execution
    exceeds the plugin's configured threshold, it is automatically converted to a
    background task and a handle is returned.
    """
    def __init__(
        self,
        ledger: Optional[TokenLedger] = None,
        auto_background_enabled: bool = True,
        auto_background_pool_size: int = 4
    ):
        self._map: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._permission_plugin: Optional['PermissionPlugin'] = None
        self._permission_context: Dict[str, Any] = {}
        self._ledger: Optional[TokenLedger] = ledger

        # Registry reference for plugin lookups (set via set_registry)
        self._registry: Optional['PluginRegistry'] = None

        # Output callback for real-time output from plugins
        self._output_callback: Optional[OutputCallback] = None

        # Auto-background support
        self._auto_background_enabled = auto_background_enabled
        self._auto_background_pool: Optional[ThreadPoolExecutor] = None
        self._auto_background_pool_size = auto_background_pool_size

    def register(self, name: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
        self._map[name] = fn

    def set_ledger(self, ledger: Optional[TokenLedger]) -> None:
        """Set the ledger for recording events."""
        self._ledger = ledger

    def set_permission_plugin(
        self,
        plugin: Optional['PermissionPlugin'],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the permission plugin for access control.

        Args:
            plugin: PermissionPlugin instance, or None to disable permission checking.
            context: Optional context dict passed to permission checks (e.g., session_id).
        """
        self._permission_plugin = plugin
        self._permission_context = context or {}

    def set_registry(self, registry: Optional['PluginRegistry']) -> None:
        """Set the plugin registry for plugin lookups.

        Required for auto-background support to find BackgroundCapable plugins.

        Args:
            registry: PluginRegistry instance, or None to disable.
        """
        self._registry = registry

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set the output callback for real-time plugin output.

        When set, plugins that support output callbacks will receive this
        callback to emit real-time output during tool execution.

        The callback is passed to plugins via their set_output_callback()
        method if they implement it.

        Args:
            callback: OutputCallback function, or None to clear.
        """
        self._output_callback = callback

        # Forward callback to exposed plugins that support it
        if self._registry:
            for plugin_name in self._registry.list_exposed():
                plugin = self._registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'set_output_callback'):
                    plugin.set_output_callback(callback)

        # Also set on permission plugin if configured
        if self._permission_plugin and hasattr(self._permission_plugin, 'set_output_callback'):
            self._permission_plugin.set_output_callback(callback)

    def get_output_callback(self) -> Optional[OutputCallback]:
        """Get the current output callback.

        Returns:
            The current OutputCallback, or None if not set.
        """
        return self._output_callback

    def _get_auto_background_pool(self) -> ThreadPoolExecutor:
        """Get or create the thread pool for auto-background execution."""
        if self._auto_background_pool is None:
            self._auto_background_pool = ThreadPoolExecutor(
                max_workers=self._auto_background_pool_size
            )
        return self._auto_background_pool

    def _get_plugin_for_tool(self, tool_name: str) -> Optional['BackgroundCapable']:
        """Get the BackgroundCapable plugin that provides a tool.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The BackgroundCapable plugin, or None if not found or not capable.
        """
        if not self._registry:
            return None

        # Import here to avoid circular imports
        from shared.plugins.background.protocol import BackgroundCapable

        plugin = self._registry.get_plugin_for_tool(tool_name)
        if plugin and isinstance(plugin, BackgroundCapable):
            return plugin
        return None

    def _execute_sync(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute a tool synchronously (internal helper).

        This is the core execution logic, extracted to support auto-backgrounding.

        Args:
            name: Tool name.
            args: Arguments dict.

        Returns:
            Tuple of (success, result).
        """
        fn = self._map.get(name)
        if not fn:
            # Check if generic execution is allowed
            if os.environ.get('AI_EXECUTE_TOOLS', '').lower() in ('1', 'true', 'yes'):
                try:
                    return _generic_executor(name, args, debug=False)
                except Exception as exc:
                    return False, {'error': str(exc)}
            return False, {'error': f'No executor registered for {name}'}

        try:
            if fn.__name__ == 'mcp_based_tool':
                result = fn(name, args)
            else:
                result = fn(args)
            return True, result
        except Exception as exc:
            return False, {'error': str(exc)}

    def _execute_with_auto_background(
        self,
        name: str,
        args: Dict[str, Any],
        plugin: 'BackgroundCapable',
        threshold: float,
        permission_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any]:
        """Execute a tool with auto-background on timeout.

        Args:
            name: Tool name.
            args: Arguments dict.
            plugin: The BackgroundCapable plugin.
            threshold: Timeout threshold in seconds.
            permission_meta: Optional permission metadata to inject.

        Returns:
            Tuple of (success, result). If auto-backgrounded, result contains
            task handle info with auto_backgrounded=True.
        """
        pool = self._get_auto_background_pool()

        # Submit to thread pool
        future = pool.submit(self._execute_sync, name, args)

        try:
            # Wait up to threshold seconds
            ok, result = future.result(timeout=threshold)

            # Inject permission metadata if available
            if permission_meta and isinstance(result, dict):
                result['_permission'] = permission_meta

            return ok, result

        except FuturesTimeoutError:
            # Task exceeded threshold - convert to background
            try:
                handle = plugin.register_running_task(future, name, args)

                result = {
                    "auto_backgrounded": True,
                    "task_id": handle.task_id,
                    "plugin_name": handle.plugin_name,
                    "tool_name": handle.tool_name,
                    "threshold_seconds": threshold,
                    "message": f"Task exceeded {threshold}s threshold, continuing in background. "
                               f"Use task_id '{handle.task_id}' to check status."
                }

                # Inject permission metadata
                if permission_meta:
                    result['_permission'] = permission_meta

                # Record auto-background event
                if self._ledger:
                    self._ledger._record('auto-background', {
                        'tool': name,
                        'task_id': handle.task_id,
                        'threshold': threshold,
                    })

                return True, result

            except Exception as e:
                # Failed to register - wait for result and return normally
                try:
                    ok, result = future.result(timeout=300)  # Give it more time
                    if permission_meta and isinstance(result, dict):
                        result['_permission'] = permission_meta
                    return ok, result
                except Exception as inner_e:
                    return False, {'error': f'Auto-background failed: {e}, execution failed: {inner_e}'}

    def execute(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        debug = False
        try:
            debug = os.environ.get('AI_TOOL_RUNNER_DEBUG', '').lower() in ('1', 'true', 'yes')
        except Exception:
            debug = False

        # Track permission metadata for injection into result
        permission_meta = None

        # Check permissions if a permission plugin is set
        # Note: askPermission tool itself is always allowed
        if self._permission_plugin is not None and name != 'askPermission':
            try:
                allowed, perm_info = self._permission_plugin.check_permission(
                    name, args, self._permission_context
                )
                # Build permission metadata for result injection
                permission_meta = {
                    'decision': 'allowed' if allowed else 'denied',
                    'reason': perm_info.get('reason', ''),
                    'method': perm_info.get('method', 'unknown'),
                }
                # Record permission check to ledger
                if self._ledger is not None:
                    self._ledger._record('permission-check', {
                        'tool': name,
                        'args': args,
                        'allowed': allowed,
                        'reason': perm_info.get('reason', ''),
                        'method': perm_info.get('method', 'unknown'),
                    })
                if not allowed:
                    if debug:
                        print(f"[ai_tool_runner] permission denied for {name}: {perm_info.get('reason', '')}")
                    return False, {'error': f"Permission denied: {perm_info.get('reason', '')}", '_permission': permission_meta}
                if debug:
                    print(f"[ai_tool_runner] permission granted for {name}: {perm_info.get('reason', '')}")
            except Exception as perm_exc:
                if debug:
                    print(f"[ai_tool_runner] permission check failed for {name}: {perm_exc}")
                # Record permission error to ledger
                if self._ledger is not None:
                    self._ledger._record('permission-error', {
                        'tool': name,
                        'args': args,
                        'error': str(perm_exc),
                    })
                # On permission check failure, deny by default for safety
                return False, {'error': f'Permission check failed: {perm_exc}'}

        # Check for auto-background capability
        if self._auto_background_enabled and self._registry:
            bg_plugin = self._get_plugin_for_tool(name)
            if bg_plugin is not None:
                try:
                    threshold = bg_plugin.get_auto_background_threshold(name)
                    if threshold is not None and threshold > 0:
                        if debug:
                            print(f"[ai_tool_runner] using auto-background for {name} "
                                  f"(threshold={threshold}s)")
                        return self._execute_with_auto_background(
                            name, args, bg_plugin, threshold, permission_meta
                        )
                except Exception as e:
                    if debug:
                        print(f"[ai_tool_runner] auto-background check failed for {name}: {e}")
                    # Fall through to normal execution

        fn = self._map.get(name)
        if not fn:
            if debug:
                print(f"[ai_tool_runner] execute: no executor registered for {name}, attempting generic execution")
            # Check if generic execution is allowed via env var
            if os.environ.get('AI_EXECUTE_TOOLS', '').lower() in ('1', 'true', 'yes'):
                try:
                    ok, res = _generic_executor(name, args, debug=debug)
                    # Inject permission metadata if available
                    if permission_meta and isinstance(res, dict):
                        res['_permission'] = permission_meta
                    return ok, res
                except Exception as exc:
                    if debug:
                        print(f"[ai_tool_runner] generic executor failed for {name}: {exc}")
                    return False, {'error': str(exc)}
            else:
                return False, {'error': f'No executor registered for {name}'}
        try:
            if debug:
                print(f"[ai_tool_runner] execute: invoking {name} with args={args}")
            if fn.__name__ == 'mcp_based_tool':
                result = fn(name, args)
            else:
                result = fn(args)
            # Inject permission metadata if available and result is a dict
            if permission_meta and isinstance(result, dict):
                result['_permission'] = permission_meta
            return True, result
        except Exception as exc:
            if debug:
                print(f"[ai_tool_runner] execute: {name} raised {exc}")
            return False, {'error': str(exc)}


def _generic_executor(name: str, args: Dict[str, Any], debug: bool = False) -> Tuple[bool, Any]:
    """Generic fallback executor: attempt to run a CLI command or MCP client based on name/args.

    - If `name` looks like a CLI tool (contains '-cli' or 'confluence'), shell out accordingly.
    - If `name` looks like an MCP client command, attempt to call a MCP client function (placeholder).
    This is intentionally conservative and returns structured errors when not possible.
    """
    # Heuristics for CLI tools
    lname = name.lower() if name else ''
    if 'confluence' in lname or 'confluence-cli' in lname or lname.endswith('_get'):
        # Expect args to include page id; try to construct a reasonable command
        page_id = args.get('page_id') or args.get('page') or args.get('id')
        if not page_id:
            return False, {'error': 'generic_executor: missing page id'}
        cmd = ['confluence-cli', 'get', '--page', str(page_id)]
        if debug:
            print(f"[ai_tool_runner] generic_executor running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = proc.stdout or proc.stderr or ''
            return True, {'raw': out}
        except Exception as exc:
            return False, {'error': str(exc)}

    # MCP client placeholder: look for 'mcp' prefix
    if lname.startswith('mcp') or lname.startswith('mcp_'):
        # Placeholder: if you have an MCP client library, call it here.
        return False, {'error': 'MCP client execution not implemented in generic executor'}

    return False, {'error': f'generic_executor: cannot handle function {name}'}


__all__ = ['ToolExecutor']
