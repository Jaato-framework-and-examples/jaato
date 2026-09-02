"""Tool execution infrastructure for the jaato framework.

This module provides the ToolExecutor class for managing tool/function
execution with support for:
- Permission checking via PermissionPlugin
- Auto-backgrounding for long-running tasks
- Output callbacks for real-time feedback
"""

import contextlib
import json
import logging
import os
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from shared.safe_pool import SafeThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)

from shared.token_accounting import TokenLedger
from jaato_sdk.plugins.base import OutputCallback
from jaato_sdk.plugins.model_provider.types import (
    CancelledException,
    WithMetadata,
)

# Callback for streaming tool output during execution
# (chunk: str) -> None - simplified since call_id is known at call site
ToolOutputCallback = Callable[[str], None]

# Thread-local storage for tool output callbacks and cancel tokens.
# Used for parallel tool execution where each thread needs its own state.
_thread_local = threading.local()


def get_current_tool_output_callback() -> Optional[ToolOutputCallback]:
    """Get the tool output callback for the current thread.

    For use by plugins during parallel tool execution. Returns the callback
    set for this thread, or None if not in a parallel execution context.

    Returns:
        The current thread's ToolOutputCallback, or None.
    """
    return getattr(_thread_local, 'tool_output_callback', None)


def get_current_cancel_token():
    """Get the cancel token for the current thread.

    For use by plugins during tool execution to check if the operation has
    been cancelled. Returns the token set for this thread, or None if not
    in a tool execution context.

    Returns:
        The current thread's CancelToken, or None.
    """
    return getattr(_thread_local, 'cancel_token', None)


def in_trusted_bridge_context() -> bool:
    """Whether the current thread is executing inside a trusted tool bridge.

    A "trusted bridge" is a plugin-provided interpreter (today only the
    notebook plugin's Python tool bindings) whose **outer** invocation was
    already permission-approved by the user.  When the flag is set, tool
    calls made through the bridge inherit that approval — permission
    prompts for individual inner calls would be redundant because the user
    already saw and approved the full code (including all ``tools.X(...)``
    calls) when they approved the outer tool.

    Plugins enter the trusted context via
    :func:`push_trusted_bridge_context` before dispatching inner tool calls
    and exit via :func:`pop_trusted_bridge_context` after the outer call
    returns.  The context manager
    :func:`trusted_bridge_context` wraps both in a ``with`` block.

    Consumers (currently the permission plugin) call this from
    ``check_permission`` to short-circuit the approval check with an
    ALLOW decision when inside a trusted context.

    Returns:
        True if the current thread is inside a trusted bridge scope,
        False otherwise.
    """
    return bool(getattr(_thread_local, 'trusted_bridge_depth', 0))


def push_trusted_bridge_context() -> None:
    """Enter a trusted bridge scope on the current thread.

    Increments a per-thread depth counter so that nested entries (e.g. a
    bridge cell that itself uses another bridge) are correctly balanced.
    Callers MUST pair every ``push`` with a ``pop`` — prefer
    :func:`trusted_bridge_context` to guarantee cleanup on exceptions.
    """
    current = getattr(_thread_local, 'trusted_bridge_depth', 0)
    _thread_local.trusted_bridge_depth = current + 1


def pop_trusted_bridge_context() -> None:
    """Exit a trusted bridge scope on the current thread.

    Decrements the per-thread depth counter.  Underflow is silently
    clamped to zero — callers that ``pop`` without a matching ``push``
    indicate a bug but we prefer defensive clamping over exception noise
    in exit paths.
    """
    current = getattr(_thread_local, 'trusted_bridge_depth', 0)
    _thread_local.trusted_bridge_depth = max(0, current - 1)


@contextlib.contextmanager
def trusted_bridge_context():
    """Context manager form of push/pop for safe nested use.

    Example::

        with trusted_bridge_context():
            # Tool calls dispatched here skip permission prompts.
            result = backend.execute(cell_code)

    The scope is thread-local; other threads are unaffected.
    """
    push_trusted_bridge_context()
    try:
        yield
    finally:
        pop_trusted_bridge_context()

if TYPE_CHECKING:
    from shared.plugins.registry import PluginRegistry
    from shared.plugins.permission import PermissionPlugin
    from shared.plugins.background.protocol import BackgroundCapable
    from shared.plugins.reliability import ReliabilityPlugin
    from shared.runtime_limits import RuntimeLimits


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

        # Tool-specific output callback for streaming during execution
        # Set per-tool to route output to the correct tool tree entry
        self._tool_output_callback: Optional[ToolOutputCallback] = None

        # Auto-background support
        self._auto_background_enabled = auto_background_enabled
        self._auto_background_pool: Optional[ThreadPoolExecutor] = None
        self._auto_background_pool_size = auto_background_pool_size

        # Callback fired when an auto-backgrounded task completes.
        # Set by the session before execute(), captured per-task after threshold.
        self._task_done_callback: Optional[Callable] = None

        # Reliability plugin for tracking tool failures and adaptive trust
        self._reliability_plugin: Optional['ReliabilityPlugin'] = None

        # AppArmor thread-level confinement context factory.
        # When set, every tool execution is wrapped in this context
        # manager, which confines the current OS thread to the session's
        # AppArmor profile for the duration of the call.  This ensures
        # in-process file I/O (readFile, glob_files, file_edit) is
        # subject to the same AppArmor profile as subprocess commands.
        # Set via set_apparmor_context() from the server layer.
        self._apparmor_context: Optional[Callable] = None

        # Per-session runtime limits surfaced to subprocess-launching
        # plugins (cli, interactive_shell).  Set via the server layer
        # in the same hook that installs ``_apparmor_context``.
        #
        # ``_cgroup_attach`` is a zero-argument callable suitable for
        # ``subprocess.Popen(preexec_fn=...)``: it writes the forked
        # child's PID to the session's cgroup ``cgroup.procs`` between
        # fork() and exec(), so the new program comes up already inside
        # the cgroup with the kernel-enforced limits in effect.
        #
        # ``_runtime_limits`` carries the *application-layer* caps
        # (``tool_timeout_seconds``, ``max_output_bytes``) that have no
        # cgroup equivalent — plugins read them via ``get_runtime_limits()``
        # and apply them at the Python layer.  Both fields stay ``None``
        # when no profile-level runtime_limits is configured, leaving
        # the host's defaults in effect.
        self._cgroup_attach: Optional[Callable[[], None]] = None
        self._runtime_limits: Optional['RuntimeLimits'] = None

        # Phase 5 §5.10c: AppArmor child-profile transition callback
        # for subprocess-spawning plugins (cli, interactive_shell).
        # Zero-arg callable suitable for ``Popen(preexec_fn=...)`` that
        # writes ``changeprofile {profile}//child`` to
        # /proc/self/attr/current between fork() and exec(), so the
        # forked child enters the per-session ``//child`` sub-profile
        # before the new program starts.  ``//child`` drops the three
        # escape-vector rules the parent keeps for
        # ``apparmor_confine.__exit__`` — closes the verified escape at
        # apparmor.py:413-449.  ``None`` when the runner isn't confined
        # (e.g. JAATO_RUNNER_DISABLE_CONFINE=1, or daemon-side legacy
        # paths that never installed a session profile).
        #
        # Forwarded to plugins through the same channel as
        # ``_cgroup_attach`` (plugins that implement
        # ``set_apparmor_child_transition_callback`` get the callable;
        # the rest stay unchanged).  See
        # docs/design/phase5_5_10_apparmor_child_subprofile_audit.md.
        self._apparmor_child_transition: Optional[Callable[[], None]] = None

        # Zero-arg event-snapshot callable for cgroup.events (oom_kill,
        # populated, ...).  Used by ``execute()`` to take before/after
        # snapshots around each tool call and inject deltas into the
        # result's ``_telemetry`` dict, where the session's tool span
        # auto-forwards them as OTel attributes.  Returns ``None`` when
        # cgroups are unavailable, so the wrapper is safe to invoke
        # unconditionally.
        self._cgroup_event_reader: Optional[Callable[[], Optional[Dict[str, int]]]] = None

        # Plug-in transformer chains for the tool-dispatch boundary
        # (seat 2 of the four-seat pseudonymization design — see
        # docs/design/daemon-extensions.md and
        # project_backlog_pseudonymization_plugin_surface.md).
        # Each list entry is registered via ``register_*_transformer``;
        # ``execute()`` runs the args chain before ``_execute_impl`` and
        # the result chain on the returned value.  Empty lists = no-op
        # (full backwards-compat).  Per-transformer ``trusted_tools``
        # set lets a registration skip specific tool names so the
        # transformer applies only to *untrusted* tools.
        self._args_transformers: List[
            Tuple[Callable[[str, Dict[str, Any]], Dict[str, Any]],
                  Optional[Set[str]]]
        ] = []
        self._result_transformers: List[Callable[[str, Any], Any]] = []


    def register_args_transformer(
        self,
        fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        *,
        trusted_tools: Optional[Set[str]] = None,
    ) -> None:
        """Register a transformer for tool args before ``_execute_impl``.

        Plug-in surface for redaction / content-filter / audit consumers
        that need to inspect or mutate args before the tool runs.
        Multiple transformers stack — registered in order, applied as
        a chain.

        Args:
            fn: Callable receiving ``(tool_name, args)`` and returning
                the args dict to actually pass to the tool.  Must
                always return a dict (returning ``None`` would silently
                strip args).
            trusted_tools: Optional set of tool names this transformer
                should NOT touch — when provided, ``fn`` is invoked
                only for tools whose name is **not** in the set.  Use
                this to give a redaction transformer an allowlist of
                tools that legitimately need raw values (e.g. a tool
                that sends an email needs the real address, not a
                placeholder).  Default ``None`` = transformer applies
                to every tool.
        """
        self._args_transformers.append((fn, trusted_tools))

    def register_result_transformer(
        self, fn: Callable[[str, Any], Any]
    ) -> None:
        """Register a transformer for tool results before they return.

        Plug-in surface for re-redacting tool outputs (e.g. a tool
        returns a database query result that contains PII; the
        transformer pseudonymizes those values before the result is
        appended to history).  Multiple transformers stack — registered
        in order, applied as a chain.

        Args:
            fn: Callable receiving ``(tool_name, result)`` and returning
                the value to actually surface.  ``result`` is whatever
                the tool's executor returned (typically dict or str).
                Must return a value of the same shape; returning
                ``None`` would silently drop the result.
        """
        self._result_transformers.append(fn)

    def _apply_args_transformers(
        self, name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the args transformer chain in registration order.

        Each transformer's ``trusted_tools`` set determines whether it
        runs for this tool name.  Result of one transformer feeds the
        next.  Empty chain returns args unchanged (cheap fast path).
        """
        if not self._args_transformers:
            return args
        for fn, trusted in self._args_transformers:
            if trusted is not None and name in trusted:
                continue
            args = fn(name, args)
        return args

    def _apply_result_transformers(self, name: str, result: Any) -> Any:
        """Run the result transformer chain in registration order."""
        if not self._result_transformers:
            return result
        for fn in self._result_transformers:
            result = fn(name, result)
        return result

    def register(self, name: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
        self._map[name] = fn

    def clear_executors(self) -> None:
        """Clear all registered executors.

        Useful when refreshing tools after enabling/disabling.
        """
        self._map.clear()

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

    def update_permission_context(self, **kwargs) -> None:
        """Update the permission context dict with additional fields.

        Called by the session to inject per-turn state (turn_index,
        model_preamble) that evaluators can inspect.

        Args:
            **kwargs: Key-value pairs to merge into the context.
        """
        self._permission_context.update(kwargs)

    def set_reliability_plugin(self, plugin: Optional['ReliabilityPlugin']) -> None:
        """Set the reliability plugin for tracking tool failures.

        Args:
            plugin: ReliabilityPlugin instance, or None to disable reliability tracking.
        """
        self._reliability_plugin = plugin

    def set_registry(self, registry: Optional['PluginRegistry']) -> None:
        """Set the plugin registry for plugin lookups.

        Required for auto-background support to find BackgroundCapable plugins.

        Args:
            registry: PluginRegistry instance, or None to disable.
        """
        self._registry = registry

    def set_apparmor_context(self, context_factory: Optional[Callable]) -> None:
        """Set the AppArmor thread-level confinement context factory.

        When set, every tool execution is wrapped in the context manager
        returned by ``context_factory()``, which confines the current OS
        thread to the session's AppArmor profile.

        Args:
            context_factory: A zero-argument callable returning a context
                manager, or ``None`` to disable confinement.
        """
        self._apparmor_context = context_factory

    def set_runtime_limits(
        self,
        attach_callback: Optional[Callable[[], None]],
        limits: Optional['RuntimeLimits'],
        event_reader: Optional[Callable[[], Optional[Dict[str, int]]]] = None,
    ) -> None:
        """Install per-session cgroup attach + app-layer limits + event reader.

        Called by the server layer after the cgroup has been provisioned
        (or, for sessions without kernel limits, with ``attach_callback``
        set to a no-op).  Subprocess-launching plugins read attach +
        limits via :meth:`get_cgroup_attach` and :meth:`get_runtime_limits`,
        OR via the forwarded ``set_runtime_limits`` method on the plugin
        if it implements one — same pattern as ``set_tool_output_callback``.

        The ``event_reader`` is consumed *here* in :meth:`execute` rather
        than forwarded to plugins: snapshotting before/after each tool
        call and injecting deltas into the result's ``_telemetry`` dict
        means the existing OTel forwarder picks up
        ``jaato.cgroup.oom_kill_delta`` etc. without any plugin needing
        to know about cgroup telemetry.

        Args:
            attach_callback: Zero-argument callable suitable for use as
                ``Popen(preexec_fn=...)``.  Migrates the forked child
                into the session's cgroup before ``exec``.  ``None``
                means no attach (host defaults).
            limits: :class:`RuntimeLimits` carrying the app-layer caps
                (``tool_timeout_seconds``, ``max_output_bytes``).  May
                be ``None`` when no profile-level runtime_limits is set.
            event_reader: Zero-arg callable returning the current
                ``cgroup.events`` snapshot dict, or ``None`` when no
                cgroup is available.  Used by :meth:`execute` to compute
                per-tool deltas.
        """
        self._cgroup_attach = attach_callback
        self._runtime_limits = limits
        self._cgroup_event_reader = event_reader

        # Forward attach + limits to exposed plugins that support it.
        # event_reader is intentionally NOT forwarded — it's owned by
        # the executor's wrapper, not by individual plugins.
        if self._registry:
            for plugin_name in self._registry.list_exposed():
                plugin = self._registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'set_runtime_limits'):
                    plugin.set_runtime_limits(attach_callback, limits)

    def set_apparmor_child_transition_callback(
        self,
        callback: Optional[Callable[[], None]],
    ) -> None:
        """Install the AppArmor child-profile transition callback
        (Phase 5 §5.10c).

        Called once at runner-side bootstrap with a zero-arg callable
        built by
        :func:`server.apparmor.make_child_transition_callback`.  The
        callable writes ``changeprofile <session>//child`` to
        /proc/self/attr/current, suitable for use as
        ``Popen(preexec_fn=...)`` — runs between fork() and exec()
        so the forked child enters the per-session ``//child``
        sub-profile before the new program starts.

        Forwarded to plugins that implement
        ``set_apparmor_child_transition_callback`` (cli,
        interactive_shell) via the same mechanism as
        :meth:`set_runtime_limits`'s forwarding loop.  Plugins that
        don't implement the method (file_edit, todo, etc.) stay
        unaffected — only subprocess-spawning plugins care.

        Args:
            callback: Zero-arg ``preexec_fn``-style callable, or
                ``None`` when the runner isn't AppArmor-confined
                (e.g., JAATO_RUNNER_DISABLE_CONFINE=1 or a daemon-
                side legacy path).  ``None`` is forwarded too — a
                plugin that previously had a callback installed
                gets it cleared.
        """
        self._apparmor_child_transition = callback

        if self._registry:
            for plugin_name in self._registry.list_exposed():
                plugin = self._registry.get_plugin(plugin_name)
                if plugin and hasattr(
                    plugin, "set_apparmor_child_transition_callback",
                ):
                    plugin.set_apparmor_child_transition_callback(callback)

    def get_apparmor_child_transition_callback(
        self,
    ) -> Optional[Callable[[], None]]:
        """Return the AppArmor child-profile transition callback, or
        ``None`` if not set.

        Companion of :meth:`get_cgroup_attach`.  Subprocess-launching
        plugins compose this with the cgroup attach in their
        ``preexec_fn`` — AppArmor transition first, then cgroup
        attach, then exec (the new profile must apply during the
        cgroup write).
        """
        return self._apparmor_child_transition

    def get_cgroup_attach(self) -> Optional[Callable[[], None]]:
        """Return the cgroup-attach callable, or ``None`` if not set.

        Subprocess-launching plugins pass the result as
        ``Popen(preexec_fn=...)``; passing ``None`` is identical to not
        attaching, which is the correct behaviour when the session has
        no kernel-enforced limits.
        """
        return self._cgroup_attach

    def get_runtime_limits(self) -> Optional['RuntimeLimits']:
        """Return the per-session :class:`RuntimeLimits`, or ``None``.

        Plugins consult this to read app-layer caps such as
        ``tool_timeout_seconds`` and ``max_output_bytes``; the kernel
        portion has already been written to the cgroup at provision
        time and need not be re-read here.
        """
        return self._runtime_limits

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

    def set_tool_output_callback(self, callback: Optional[ToolOutputCallback]) -> None:
        """Set the callback for streaming tool output during execution.

        This callback is set per-tool-call to route output to the correct
        tool tree entry. The session sets this before each tool execution
        with a closure that includes the call_id.

        Args:
            callback: ToolOutputCallback function (chunk: str) -> None, or None to clear.
        """
        self._tool_output_callback = callback

        # Forward to exposed plugins that support it
        if self._registry:
            for plugin_name in self._registry.list_exposed():
                plugin = self._registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'set_tool_output_callback'):
                    plugin.set_tool_output_callback(callback)

    def set_task_done_callback(self, callback: Optional[Callable]) -> None:
        """Set the callback for when an auto-backgrounded task completes.

        The session sets this before each tool execution with a closure that
        captures the call_id. The executor stores it and registers it per-task
        on the mixin only when auto-backgrounding actually occurs.

        Args:
            callback: Callable(task_id, success, error, duration), or None to clear.
        """
        self._task_done_callback = callback

    def get_tool_output_callback(self) -> Optional[ToolOutputCallback]:
        """Get the current tool output callback.

        For parallel tool execution, checks thread-local storage first,
        then falls back to the instance-level callback.

        Returns:
            The current ToolOutputCallback, or None if not set.
        """
        # Check thread-local first (for parallel execution)
        thread_callback = getattr(_thread_local, 'tool_output_callback', None)
        if thread_callback is not None:
            return thread_callback
        # Fall back to instance-level callback (for sequential execution)
        return self._tool_output_callback

    def _get_auto_background_pool(self) -> ThreadPoolExecutor:
        """Get or create the thread pool for auto-background execution.

        Server 0.6.47+: uses :class:`SafeThreadPoolExecutor` so every
        submitted task starts with the registered AppArmor pre-task
        hook (defensive ``changeprofile unconfined``).  Closes the
        residual gap where workers stuck in a prior session's profile
        would EACCES on non-tool work scheduled here.
        """
        if self._auto_background_pool is None:
            self._auto_background_pool = SafeThreadPoolExecutor(
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

    def _can_resolve_executor(self, name: str) -> bool:
        """Check whether an executor can be resolved for the given tool name.

        Performs a lightweight lookup without executing anything. Used to
        skip permission prompts for tools that have no registered executor,
        avoiding unnecessary user interaction for calls that will
        unconditionally fail.

        Args:
            name: Tool name to check.

        Returns:
            True if an executor can be found via direct map, registry, or
            generic execution fallback.
        """
        # Direct map lookup
        if name in self._map:
            return True
        # Registry/plugin fallback
        if self._registry:
            plugin = self._registry.get_plugin_for_tool(name)
            if plugin and hasattr(plugin, 'get_executors'):
                if name in plugin.get_executors():
                    return True
            # Core executors (client-registered tools, dismiss_stream, etc.)
            if name in self._registry.get_core_executors():
                return True
        # Generic executor fallback
        if os.environ.get('AI_EXECUTE_TOOLS', '').lower() in ('1', 'true', 'yes'):  # env: treat unregistered tools as executable via the generic executor fallback
            return True
        return False

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
        if not fn and self._registry:
            # Fallback: try to get executor from registry
            # This handles tools discovered after session configuration (e.g., MCP tools)
            plugin = self._registry.get_plugin_for_tool(name)
            if plugin and hasattr(plugin, 'get_executors'):
                plugin_executors = plugin.get_executors()
                fn = plugin_executors.get(name)
                if fn:
                    # Cache it for future calls
                    self._map[name] = fn
            # Also check core executors (client-registered tools, dismiss_stream, etc.)
            if not fn:
                core_executors = self._registry.get_core_executors()
                fn = core_executors.get(name)
                if fn:
                    self._map[name] = fn
        if not fn:
            # Check if generic execution is allowed
            if os.environ.get('AI_EXECUTE_TOOLS', '').lower() in ('1', 'true', 'yes'):
                try:
                    return _generic_executor(name, args, debug=False)
                except Exception as exc:
                    logger.error(f"Generic executor failed for {name}", exc_info=True)
                    return False, {'error': str(exc), 'traceback': traceback.format_exc()}
            return False, {'error': f'No executor registered for {name}'}

        try:
            if fn.__name__ == 'mcp_based_tool':
                result = fn(name, args)
            else:
                result = fn(args)
            return self._normalize_executor_return(result)
        except Exception as exc:
            logger.error(f"Tool execution failed for {name}", exc_info=True)
            return False, {'error': str(exc), 'traceback': traceback.format_exc()}

    @staticmethod
    def _normalize_executor_return(result: Any) -> Tuple[bool, Any]:
        """Turn an executor's raw return into ``(ok, result)``.

        THREE SHAPES, and only one of them used to be distinguishable:

        - :class:`WithMetadata` -- a result plus side-channel keys for the
          session layer.  Merged, reported as success.
        - a 2-tuple -- the ``(ok, payload)`` contract that
          ``split_executor_result`` reads everywhere else.  Passed through
          UNCHANGED, because it is already the shape this method returns.
        - anything else -- a bare result, reported as success.

        THE BUG THIS REPLACES: the metadata convention was a bare
        ``(result_dict, metadata_dict)`` tuple, and this code unwrapped ANY
        2-tuple whose second element was a dict.  ``(ok, payload)`` has
        exactly that shape, so ``(False, receipt)`` was read as
        result=``False`` / metadata=``receipt``; the merge was skipped
        because ``False`` is not a dict; and the call returned
        ``(True, False)`` -- flag inverted, payload gone.  ``(True, {...})``
        became ``(True, True)``.  Nineteen executors return that contract.

        Naming the metadata convention (:class:`WithMetadata`) is what makes
        the bare tuple unambiguous.  Discriminating on
        ``isinstance(x[0], bool)`` would have ARBITRATED the ambiguity
        instead of removing it, and the next convention shaped
        ``(bool, dict)`` would rejoin the collision silently.
        """
        if isinstance(result, WithMetadata):
            merged = result.result
            if isinstance(merged, dict):
                merged.update(result.metadata)
            return True, merged
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        return True, result

    def _execute_with_auto_background(
        self,
        name: str,
        args: Dict[str, Any],
        plugin: 'BackgroundCapable',
        threshold: float,
        permission_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any]:
        """Execute a tool with auto-background on timeout.

        Uses the plugin's streaming executor from the start so that output
        is captured incrementally even if the task gets auto-backgrounded.

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
        # Get the executor function for this tool
        executor_fn = None
        if hasattr(plugin, 'get_executors'):
            executors = plugin.get_executors()
            executor_fn = executors.get(name)

        if executor_fn is None:
            # Fall back to sync execution if no executor found
            return self._execute_sync(name, args)

        try:
            # Start as background task immediately - this uses the streaming
            # executor which captures output incrementally.
            # Pass the current output callback explicitly for thread-safety
            # (in parallel execution, the callback is in thread-local, not instance).
            current_output_cb = self.get_tool_output_callback()
            handle = plugin.start_background(
                name, args, executor_fn=executor_fn,
                output_callback=current_output_cb,
            )
            task_id = handle.task_id

            # Wait up to threshold seconds for completion
            start_time = time.time()
            while time.time() - start_time < threshold:
                status = plugin.get_status(task_id)
                if status.value not in ('pending', 'running'):
                    # Task completed within threshold - get full result
                    task_result = plugin.get_result(task_id)
                    result = task_result.result
                    if task_result.status.value == 'failed':
                        if permission_meta and isinstance(result, dict):
                            result['_permission'] = permission_meta
                        return False, result or {'error': task_result.error}
                    if permission_meta and isinstance(result, dict):
                        result['_permission'] = permission_meta
                    return True, result
                time.sleep(0.1)  # Small poll interval

            # Task exceeded threshold - register done callback for UI completion
            if self._task_done_callback and hasattr(plugin, 'set_task_done_callback'):
                plugin.set_task_done_callback(task_id, self._task_done_callback)

            # Return as auto-backgrounded
            result = {
                "auto_backgrounded": True,
                "task_id": task_id,
                "plugin_name": handle.plugin_name,
                "tool_name": handle.tool_name,
                "threshold_seconds": threshold,
                "message": f"Task exceeded {threshold}s threshold, continuing in background. "
                           f"Use task_id '{task_id}' to check status and output."
            }

            # Inject permission metadata
            if permission_meta:
                result['_permission'] = permission_meta

            # Record auto-background event
            if self._ledger:
                self._ledger._record('auto-background', {
                    'tool': name,
                    'task_id': task_id,
                    'threshold': threshold,
                })

            return True, result

        except Exception as e:
            # If start_background fails, fall back to sync execution
            try:
                return self._execute_sync(name, args)
            except Exception as inner_e:
                return False, {'error': f'Background start failed: {e}, sync fallback failed: {inner_e}'}

    def execute(
        self,
        name: str,
        args: Dict[str, Any],
        tool_output_callback: Optional[ToolOutputCallback] = None,
        call_id: Optional[str] = None,
        cancel_token=None,
    ) -> Tuple[bool, Any]:
        """Execute a tool by name with the given arguments.

        Args:
            name: Tool name to execute.
            args: Arguments dict to pass to the tool.
            tool_output_callback: Optional callback for streaming output during execution.
                If provided, overrides the instance-level callback for this call only.
                This enables thread-safe parallel execution where each tool has its own callback.
            call_id: Optional unique identifier for this tool call (for parallel tool matching
                in permission UI).
            cancel_token: Optional CancelToken. When set, plugins can poll
                get_current_cancel_token() to abort long-running operations. Stored in
                thread-local so it is safe for parallel tool execution.

        Returns:
            Tuple of (success: bool, result: Any).
        """
        debug = False
        try:
            debug = os.environ.get('AI_TOOL_RUNNER_DEBUG', '').lower() in ('1', 'true', 'yes')  # env: verbose tool-executor debug logging
        except Exception as exc:
            logger.debug(f"Error checking debug env var: {exc}")
            debug = False

        # Set thread-local state for parallel execution support.
        # Plugins call get_current_tool_output_callback() / get_current_cancel_token()
        # from their executor to retrieve the per-thread values.
        if tool_output_callback is not None:
            _thread_local.tool_output_callback = tool_output_callback
        if cancel_token is not None:
            _thread_local.cancel_token = cancel_token

        # Snapshot cgroup.events so we can attribute kernel-killed
        # exits to *this* tool call.  No-op when cgroups are unavailable
        # — the no-op reader returns None and the post-call comparison
        # short-circuits.
        before_events: Optional[Dict[str, int]] = None
        if self._cgroup_event_reader is not None:
            before_events = self._cgroup_event_reader()

        # Apply args transformer chain (seat 2 of pseudonymization
        # design).  Untrusted tools see redacted args; trusted tools
        # (per each transformer's trusted_tools set) see raw args.  No
        # transformers registered = identity, no overhead.
        args = self._apply_args_transformers(name, args)

        try:
            success, result = self._execute_impl(name, args, debug, call_id)
        finally:
            if tool_output_callback is not None:
                _thread_local.tool_output_callback = None
            if cancel_token is not None:
                _thread_local.cancel_token = None

        # Apply result transformer chain — re-redact (or otherwise
        # transform) what the tool returned before it reaches the
        # session's history-append path or its caller.
        result = self._apply_result_transformers(name, result)

        # Compute event-counter deltas and inject into result's
        # ``_telemetry`` dict.  The session's tool span already
        # auto-forwards every key in ``_telemetry`` as an OTel
        # attribute (jaato_session.py:4914), so adding the deltas here
        # is the only step needed to surface them as
        # ``jaato.cgroup.oom_kill_delta`` etc. on the span.
        if before_events is not None and self._cgroup_event_reader is not None:
            after_events = self._cgroup_event_reader()
            if after_events is not None and isinstance(result, dict):
                self._inject_cgroup_deltas(result, before_events, after_events)

        return success, result

    @staticmethod
    def _inject_cgroup_deltas(
        result: Dict[str, Any],
        before: Dict[str, int],
        after: Dict[str, int],
    ) -> None:
        """Add cgroup.events deltas to a tool result's ``_telemetry`` dict.

        Only deltas > 0 are emitted — the common case (no kernel events
        during the tool call) produces no extra attributes, keeping
        spans clean.  ``populated`` is monotonic only in transitions
        and isn't useful as a delta, so it's skipped.

        Attribution caveat: when multiple tool calls run concurrently
        in the same per-session cgroup, an OOM in tool A also shows up
        as a non-zero delta on a parallel tool B that happened to
        straddle the event.  The heuristic is good enough for
        telemetry — operators correlating spans with dmesg can
        disambiguate when needed.
        """
        # Skip 'populated' — it's a level, not a counter; deltas are
        # noisy and uninteresting (the cgroup is "populated" while
        # any process exists in it).
        for key in ("oom", "oom_kill"):
            before_val = before.get(key, 0)
            after_val = after.get(key, 0)
            delta = after_val - before_val
            if delta > 0:
                telem = result.setdefault("_telemetry", {})
                if isinstance(telem, dict):
                    telem[f"jaato.cgroup.{key}_delta"] = delta

    def _execute_impl(
        self,
        name: str,
        args: Dict[str, Any],
        debug: bool,
        call_id: Optional[str] = None
    ) -> Tuple[bool, Any]:
        """Internal implementation of execute(), separated for try/finally wrapping.

        Checks executor existence before permission to avoid prompting the user
        for tools that have no registered executor and will unconditionally fail.
        """
        # Early exit: skip permission prompt if no executor can be resolved.
        # This avoids asking the user to approve a tool call that will
        # unconditionally fail with "No executor registered".
        if not self._can_resolve_executor(name):
            if debug:
                print(f"[ai_tool_runner] no executor resolvable for {name}, "
                      f"skipping permission check")
            return False, {'error': f'No executor registered for {name}'}

        # Track permission metadata for injection into result
        permission_meta = None

        # Check permissions if a permission plugin is set
        # Note: askPermission tool itself is always allowed
        if self._permission_plugin is not None and name != 'askPermission':
            try:
                allowed, perm_info = self._permission_plugin.check_permission(
                    name, args, self._permission_context, call_id
                )
                # Build permission metadata for result injection
                permission_meta = {
                    'decision': 'allowed' if allowed else 'denied',
                    'reason': perm_info.get('reason', ''),
                    'method': perm_info.get('method', 'unknown'),
                }
                if perm_info.get('was_edited'):
                    permission_meta['was_edited'] = True
                if perm_info.get('comment') and allowed:
                    permission_meta['comment'] = perm_info['comment']
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
                    # For comment decisions, use the reason directly (it already
                    # contains "Tool not executed. User comment: ...") instead of
                    # wrapping with "Permission denied:" prefix which is redundant.
                    reason = perm_info.get('reason', '')
                    if perm_info.get('method') == 'user_comment':
                        error_msg = reason
                    else:
                        error_msg = f"Permission denied: {reason}"
                    return False, {'error': error_msg, '_permission': permission_meta}
                # Use edited arguments if the user modified them during permission
                if perm_info.get('was_edited') and perm_info.get('modified_args'):
                    args = perm_info['modified_args']
                    if debug:
                        print(f"[ai_tool_runner] using edited args for {name}")
                if debug:
                    print(f"[ai_tool_runner] permission granted for {name}: {perm_info.get('reason', '')}")
            except Exception as perm_exc:
                logger.error(f"Permission check failed for {name}", exc_info=True)
                if debug:
                    print(f"[ai_tool_runner] permission check failed for {name}: {perm_exc}")
                # Record permission error to ledger
                if self._ledger is not None:
                    self._ledger._record('permission-error', {
                        'tool': name,
                        'args': args,
                        'error': str(perm_exc),
                        'traceback': traceback.format_exc(),
                    })
                # On permission check failure, deny by default for safety
                return False, {'error': f'Permission check failed: {perm_exc}', 'traceback': traceback.format_exc()}

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
                    logger.warning(f"Auto-background check failed for {name}", exc_info=True)
                    if debug:
                        print(f"[ai_tool_runner] auto-background check failed for {name}: {e}")
                    # Fall through to normal execution

        fn = self._map.get(name)
        if not fn and self._registry:
            # Fallback: try to get executor from registry
            # This handles tools discovered after session configuration (e.g., MCP tools)
            if debug:
                print(f"[ai_tool_runner] execute: executor not in _map for {name}, trying registry fallback")
            plugin = self._registry.get_plugin_for_tool(name)
            if debug:
                print(f"[ai_tool_runner] execute: get_plugin_for_tool({name}) returned {plugin.name if plugin else None}")
            if plugin and hasattr(plugin, 'get_executors'):
                plugin_executors = plugin.get_executors()
                if debug:
                    print(f"[ai_tool_runner] execute: plugin {plugin.name} has {len(plugin_executors)} executors: {list(plugin_executors.keys())[:5]}...")
                fn = plugin_executors.get(name)
                if fn:
                    # Cache it for future calls
                    self._map[name] = fn
                    if debug:
                        print(f"[ai_tool_runner] execute: found executor for {name} via registry fallback")
            # Also check core executors (client-registered tools, dismiss_stream, etc.)
            if not fn:
                core_executors = self._registry.get_core_executors()
                fn = core_executors.get(name)
                if fn:
                    self._map[name] = fn
                    if debug:
                        print(f"[ai_tool_runner] execute: found executor for {name} via core executors")
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
                    logger.error(f"Generic executor failed for {name}", exc_info=True)
                    if debug:
                        print(f"[ai_tool_runner] generic executor failed for {name}: {exc}")
                    return False, {'error': str(exc), 'traceback': traceback.format_exc()}
            else:
                return False, {'error': f'No executor registered for {name}'}
        # Get plugin name for reliability tracking
        plugin_name = ""
        if self._registry:
            plugin = self._registry.get_plugin_for_tool(name)
            if plugin:
                plugin_name = getattr(plugin, 'name', '')

        # Notify reliability plugin before execution
        if self._reliability_plugin:
            try:
                self._reliability_plugin.on_tool_called(name, args)
            except Exception as e:
                logger.debug(f"Reliability plugin on_tool_called failed: {e}")

        try:
            if debug:
                print(f"[ai_tool_runner] execute: invoking {name} with args={args}")
            # AppArmor thread-level confinement: confine by default,
            # opt out via TRAIT_FRAMEWORK_LEVEL.  Any tool that touches
            # the filesystem (directly or via side effects like save_to
            # downloads) is automatically sandboxed.  Only framework-
            # setup tools (spawn_subagent) declare the opt-out trait.
            from jaato_sdk.plugins.model_provider.types import TRAIT_FRAMEWORK_LEVEL
            is_framework_tool = (
                self._registry
                and TRAIT_FRAMEWORK_LEVEL in self._registry.get_tool_traits(name)
            )

            if is_framework_tool and self._apparmor_context:
                # Framework-level tools must run unconfined.  The thread
                # may be stuck in a session profile from a prior tool
                # call whose exit failed ("could not restore unconfined").
                # Actively try to escape confinement before executing.
                try:
                    import threading as _threading
                    attr_path = f"/proc/self/task/{_threading.get_native_id()}/attr/current"
                    with open(attr_path, "w") as _f:
                        _f.write("changeprofile unconfined")
                except (OSError, PermissionError):
                    pass  # Best effort — if we can't unconfine, the tool may still work

            ctx = (
                self._apparmor_context()
                if (self._apparmor_context and not is_framework_tool)
                else None
            )
            if ctx:
                ctx.__enter__()
            try:
                if fn.__name__ == 'mcp_based_tool':
                    result = fn(name, args)
                else:
                    result = fn(args)
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)
            # Normalize the executor's return; this path keeps the pieces
            # separately because it injects permission metadata and notifies
            # the reliability plugin before returning.
            ok, result = self._normalize_executor_return(result)
            # Inject permission metadata if available and result is a dict
            if permission_meta and isinstance(result, dict):
                result['_permission'] = permission_meta

            # Notify the reliability plugin of the REAL outcome.  This
            # passed a hardcoded ``True``: an executor that returned
            # ``(False, payload)`` -- a domain failure without an exception
            # -- was reported to reliability as a success, so its retry and
            # circuit-breaker policies never saw the failures they exist to
            # count.  The flag was available the whole time; nothing read it.
            if self._reliability_plugin:
                try:
                    self._reliability_plugin.on_tool_result(
                        name, args, ok, result, call_id or "", plugin_name
                    )
                except Exception as e:
                    logger.debug(f"Reliability plugin on_tool_result failed: {e}")

            return ok, result
        except CancelledException:
            # Tool was cancelled via CancelToken — not an error, not retried.
            # Return a structured result so the session can record it in history.
            logger.debug(f"Tool {name} was cancelled")
            return False, {'error': 'cancelled'}
        except Exception as exc:
            logger.error(f"Tool execution failed for {name}", exc_info=True)
            if debug:
                print(f"[ai_tool_runner] execute: {name} raised {exc}")
            error_result = {'error': str(exc), 'traceback': traceback.format_exc()}

            # Notify reliability plugin of failure
            if self._reliability_plugin:
                try:
                    self._reliability_plugin.on_tool_result(
                        name, args, False, error_result, call_id or "", plugin_name
                    )
                except Exception as e:
                    logger.debug(f"Reliability plugin on_tool_result failed: {e}")

            return False, error_result


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
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
            out = proc.stdout or proc.stderr or ''
            return True, {'raw': out}
        except Exception as exc:
            logger.error(f"Generic executor subprocess failed for {name}", exc_info=True)
            return False, {'error': str(exc), 'traceback': traceback.format_exc()}

    # MCP client placeholder: look for 'mcp' prefix
    if lname.startswith('mcp') or lname.startswith('mcp_'):
        # Placeholder: if you have an MCP client library, call it here.
        return False, {'error': 'MCP client execution not implemented in generic executor'}

    return False, {'error': f'generic_executor: cannot handle function {name}'}


__all__ = ['ToolExecutor']
