"""Permission plugin for controlling tool execution access.

This plugin intercepts tool execution requests and enforces access policies
through blacklist/whitelist rules and interactive channel approval.
"""

import fnmatch
import os
import tempfile
import json
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from jaato_sdk.plugins.base import TRAIT_SESSION_PERSISTENT
from jaato_sdk.plugins.model_provider.types import ToolSchema

from .policy import PermissionPolicy, PermissionDecision, PolicyMatch
from .evaluator import EvalContext, PolicyDecision as EvalDecision, load_evaluators
from .config_loader import load_config, PermissionConfig
from .channels import (
    Channel,
    ChannelDecision,
    ChannelResponse,
    PermissionRequest,
    PermissionResponseOption,
    ConsoleChannel,
    create_channel,
    get_default_permission_options,
    get_permission_options_with_edit,
    EDIT_PERMISSION_OPTION,
)
from jaato_sdk.plugins.base import UserCommand, CommandCompletion, PermissionDisplayInfo, OutputCallback, HelpLines
from ...ui_utils import format_permission_options, format_tool_args_summary
from shared.plugins.runner_forwarding import RunnerForwardingMixin
from shared.trace import trace as _trace_write

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..registry import PluginRegistry


class PermissionPlugin(RunnerForwardingMixin):
    """Plugin that provides permission control for tool execution.

    This plugin acts as a middleware layer that intercepts tool execution
    requests and enforces access policies. It can:
    - Block tools via blacklist rules
    - Allow tools via whitelist rules
    - Prompt a channel for approval when policy is ambiguous

    Key principle: The model calls tools directly, and the permission middleware
    intercepts the call to check policy and prompt for approval if needed.
    This avoids confusion where models call askPermission instead of actual tools.

    Usage:
    - Enable enforcement via: executor.set_permission_plugin(plugin)
    - Expose user commands via: registry.expose_tool("permission")
    - The middleware automatically prompts for permission when tools are called
    """

    # Thread-local storage for per-session channels
    # This allows subagents (which run in separate threads) to have their own
    # channels without modifying the shared plugin instance's default channel.
    _thread_local = threading.local()

    def __init__(self):
        self._config: Optional[PermissionConfig] = None
        self._policy: Optional[PermissionPolicy] = None
        self._channel: Optional[Channel] = None
        self._registry: Optional['PluginRegistry'] = None
        self._initialized = False
        self._wrapped_executors: Dict[str, Callable] = {}
        self._original_executors: Dict[str, Callable] = {}
        self._execution_log: List[Dict[str, Any]] = []
        # Framework-reserved tool names: framework machinery (core infra +
        # lifecycle terminals like ``signal_completion``) that a business
        # catch-all ``"default"`` evaluator must NOT be able to deny — else a
        # locked-down agent can do its work but never complete.  Populated at
        # session configure() from BOTH the registry's core tools AND the
        # session's lifecycle tools (which are NOT registry core tools — they
        # register session-level via ``executor.register``, so ``is_core_tool``
        # alone misses them).  Deliberately a self-contained set (no registry
        # lookup at check time) so it survives ``shutdown()`` nulling
        # ``_registry`` and is simply re-populated every configure.
        self._framework_reserved: Set[str] = set()
        self._allow_all: bool = False  # When True, auto-approve all requests
        # Suspension state flags for temporary permission bypasses
        self._turn_suspended: bool = False  # Allow all remaining tools this turn
        self._idle_suspended: bool = False  # Allow until session goes idle
        # Lock for serializing channel interactions (permission prompts)
        # This ensures only one permission prompt is shown at a time when
        # multiple tools request permission concurrently (parallel execution)
        self._channel_lock = threading.Lock()
        # One-shot approvals handed out by ``askPermission`` (see
        # :meth:`_grant_ask_once`).  Keyed by (tool_name, canonical args);
        # each key is consumed by the FIRST matching ``check_permission``.
        # Bounded so a model that spams askPermission without executing
        # cannot grow this without limit.
        self._ask_grants: "OrderedDict[str, None]" = OrderedDict()
        self._ask_grants_lock = threading.Lock()
        # Phase 3 §3.7 + peer-review M3: lock around per-session policy
        # mutations so a cross-session ``permission.add_rule`` RPC
        # arriving mid-ASK can't let the next call for the same tool
        # bypass the prompt nondeterministically.  Acquired on every
        # rule mutation (whitelist add, blacklist add, rule delete)
        # and around the ASK-resolution "rule miss check + channel
        # wait" critical section in :meth:`check_permission` —
        # see the ASK_CHANNEL branch where the lock is held across a
        # policy.check recheck plus the channel.request_permission
        # call so cross-session mutations queue behind the prompt.
        self._policy_lock = threading.Lock()
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Workspace path for evaluator resolution and EvalContext
        self._workspace_path: Optional[str] = None
        # Permission lifecycle hooks for UI integration
        # on_requested: (tool_name, request_id, tool_args, response_options, call_id) -> None
        self._on_permission_requested: Optional[Callable[[str, str, Dict[str, Any], List[PermissionResponseOption], Optional[str]], None]] = None
        self._on_permission_resolved: Optional[Callable[[str, str, bool, str], None]] = None
        # Phase 3 §3.7 deeper: cached RunnerRPCChannel instance.  When
        # the plugin runs runner-side, ASK decisions can't reach the
        # connected client through ConsoleChannel / WebhookChannel —
        # the runner is in an AppArmor-confined process with no
        # client connection.  Instead, ``_get_channel()`` checks
        # ``self._registry.runner_rpc_client`` and uses
        # :class:`RunnerRPCChannel` to relay through
        # ``client.prompt_operator``.  Resolved lazily on first ASK
        # so init-order (set_registry vs runner_rpc_client attach)
        # doesn't matter; cached on first hit.
        self._runner_rpc_channel: Optional[Channel] = None

    def _get_channel(self) -> Optional[Channel]:
        """Get the channel for the current thread.

        Returns, in priority order:

        1. The thread-local channel if set — used by subagents which
           run in separate threads with their own per-thread channel.
        2. The runner-RPC channel if a runner-side ``RunnerRPCClient``
           is attached to the plugin registry (Phase 3 §3.7 deeper) —
           the runner-side ASK relay through the daemon's
           ``client.prompt_operator`` RPC primitive.
        3. The plugin's default channel (ConsoleChannel etc.) — the
           in-process / pre-Phase-3 path.
        """
        thread_channel = getattr(self._thread_local, 'channel', None)
        if thread_channel is not None:
            return thread_channel
        runner_channel = self._get_runner_rpc_channel()
        if runner_channel is not None:
            return runner_channel
        return self._channel

    def _get_runner_rpc_channel(self) -> Optional[Channel]:
        """Resolve / cache the :class:`RunnerRPCChannel` for runner-side use.

        Phase 3 §3.7 deeper.  Returns ``None`` when the plugin runs
        daemon-side (no runner-RPC client attached to the registry)
        so the caller falls back to the in-process channel.

        Caches on first successful resolution: subsequent ASKs reuse
        the same channel instance.  The cached instance is reset on
        ``shutdown()`` so a subsequent ``initialize()`` (e.g.,
        re-expose with new config) re-resolves cleanly.
        """
        if self._runner_rpc_channel is not None:
            return self._runner_rpc_channel
        registry = self._registry
        if registry is None:
            return None
        rpc_client = getattr(registry, 'runner_rpc_client', None)
        if rpc_client is None:
            return None
        prompt_operator = getattr(rpc_client, 'prompt_operator', None)
        if prompt_operator is None:
            return None
        # Lazy-import to avoid a top-level dependency on the channel
        # module (which lazy-imports PromptPayload from .types in
        # turn — keeps the import graph DAG-shaped).
        from .runner_rpc_channel import RunnerRPCChannel
        self._runner_rpc_channel = RunnerRPCChannel(prompt_operator)
        self._trace(
            "_get_runner_rpc_channel: resolved RunnerRPCChannel "
            "(runner-side ASK relay active)"
        )
        return self._runner_rpc_channel

    def set_registry(self, registry: 'PluginRegistry') -> None:
        """Set the plugin registry for tool-to-plugin lookups.

        This enables the permission system to call format_permission_request()
        on the source plugin to get customized display info for approval UI.

        Args:
            registry: The PluginRegistry instance.
        """
        self._registry = registry

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set the output callback for real-time permission prompts.

        When set, permission prompts will be emitted via the callback
        instead of being printed directly to the console.

        Args:
            callback: OutputCallback function, or None to use default output.
        """
        # Forward to channel if it supports callbacks
        if self._channel and hasattr(self._channel, 'set_output_callback'):
            self._channel.set_output_callback(callback)

    def set_permission_hooks(
        self,
        on_requested: Optional[Callable[[str, str, Dict[str, Any], List[PermissionResponseOption], Optional[str]], None]] = None,
        on_resolved: Optional[Callable[[str, str, bool, str, str], None]] = None
    ) -> None:
        """Set hooks for permission lifecycle events.

        These hooks enable UI integration by notifying when permission
        requests start and complete.

        Args:
            on_requested: Called when permission prompt is shown.
                Signature: (tool_name, request_id, tool_args, response_options, call_id) -> None
                - tool_name: Name of the tool requesting permission
                - request_id: Unique identifier for this request
                - tool_args: Raw arguments dict passed to the tool (client formats display)
                - response_options: List of valid PermissionResponseOption objects
                  that can be used for autocompletion. Each option has:
                  - short: Short form (e.g., "y")
                  - full: Full form (e.g., "yes")
                  - description: User-facing description
                  - decision: The ChannelDecision this maps to
                - call_id: Unique identifier for the tool call (for parallel tool matching)
            on_resolved: Called when permission is resolved.
                Signature: (tool_name, request_id, granted, method) -> None
                method is one of: "yes", "always", "once", "never",
                "whitelist", "blacklist", "timeout", "default"
        """
        self._trace(f"set_permission_hooks: on_requested={on_requested is not None}, on_resolved={on_resolved is not None}")
        self._on_permission_requested = on_requested
        self._on_permission_resolved = on_resolved

    # The operator's runtime permission decisions are session-scoped and
    # must outlive an unload/reload of that same session.  See
    # TRAIT_SESSION_PERSISTENT and jaato #706/#707.
    plugin_traits = frozenset({TRAIT_SESSION_PERSISTENT})

    @property
    def name(self) -> str:
        return "permission"

    # ------------------------------------------------------------------
    # Session persistence (TRAIT_SESSION_PERSISTENT)
    # ------------------------------------------------------------------

    def get_persistence_state(self) -> Optional[Dict[str, Any]]:
        """Snapshot the operator's runtime permission decisions.

        Only the SESSION-scoped rules are persisted.  The base policy is
        re-read from ``permissions.json`` by ``initialize()``, so
        persisting it would freeze a copy of a file the operator can edit
        between runs.

        Sets are emitted as sorted lists: JSON has no set, and sorting
        makes the snapshot stable so an unchanged policy produces an
        unchanged journal entry.

        Returns ``None`` when nothing has been decided at runtime, so a
        session that never touched permissions writes no key at all.
        """
        if not self._policy:
            return None
        allow = sorted(self._policy.session_whitelist)
        deny = sorted(self._policy.session_blacklist)
        default = self._policy.session_default_policy
        if not allow and not deny and default is None:
            return None
        return {
            "session_whitelist": allow,
            "session_blacklist": deny,
            "session_default_policy": default,
        }

    def restore_persistence_state(self, state: Dict[str, Any]) -> None:
        """Re-apply persisted runtime permission decisions.

        Runs after ``initialize()`` has loaded ``permissions.json``, so
        these runtime decisions are layered ON TOP of the file rather than
        being overwritten by it — restoring earlier would reproduce #706.

        Reads defensively: a snapshot from an older plugin version, or a
        hand-edited session file, must not make the session unloadable.
        Unknown keys are ignored and malformed values are skipped, which
        degrades to "this rule was not restored" rather than to a failed
        load.  A dropped rule is visible in ``permissions show``; a failed
        load is not recoverable by the operator.
        """
        if not self._policy:
            return
        allow = state.get("session_whitelist")
        if isinstance(allow, list):
            for pattern in allow:
                if isinstance(pattern, str) and pattern:
                    self._policy.add_session_whitelist(pattern)
        deny = state.get("session_blacklist")
        if isinstance(deny, list):
            for pattern in deny:
                if isinstance(pattern, str) and pattern:
                    self._policy.add_session_blacklist(pattern)
        default = state.get("session_default_policy")
        if isinstance(default, str) and default:
            self._policy.session_default_policy = default

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("PERMISSION", msg)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the permission plugin.

        Args:
            config: Optional configuration dict. If not provided, loads from
                   file specified by PERMISSION_CONFIG_PATH or default locations.

                   Config options:
                   - config_path: Path to permissions.json file
                   - channel_type: Type of channel ("console", "webhook", "file")
                   - channel_config: Configuration for the channel
                   - policy: Inline policy dict (overrides file)
        """
        # Load configuration
        config = config or {}

        # Extract agent name for trace logging
        self._agent_name = config.get("agent_name")

        # Try to load from file first
        config_path = config.get("config_path")
        try:
            self._config = load_config(config_path)
        except FileNotFoundError:
            # Use inline config or defaults
            self._config = PermissionConfig()

        # Allow inline policy override
        if "policy" in config:
            policy_dict = config["policy"]
            self._policy = PermissionPolicy.from_config(policy_dict)
        else:
            self._policy = PermissionPolicy.from_config(self._config.to_policy_dict())

        # Store workspace path for evaluator context
        if config.get("workspace_path"):
            self._workspace_path = config["workspace_path"]

        # Load permission evaluators if configured
        evaluator_config = config.get("evaluators") if config else None
        if evaluator_config and isinstance(evaluator_config, dict):
            evaluators = load_evaluators(evaluator_config, workspace_path=self._workspace_path)
            if evaluators:
                self._policy.set_evaluators(evaluators)

        # Initialize channel
        channel_type = config.get("channel_type") or self._config.channel_type
        channel_config = config.get("channel_config", {})

        # Set default timeout from config
        if "timeout" not in channel_config:
            channel_config["timeout"] = self._config.channel_timeout

        # For webhook, ensure endpoint is set
        if channel_type == "webhook" and "endpoint" not in channel_config:
            channel_config["endpoint"] = self._config.channel_endpoint

        try:
            self._channel = create_channel(channel_type, channel_config)
        except (ValueError, RuntimeError) as e:
            # Fall back to console channel if configured channel fails
            print(f"Warning: Failed to initialize {channel_type} channel: {e}")
            print("Falling back to console channel")
            self._channel = ConsoleChannel()

        self._initialized = True
        self._trace(f"initialize: channel={channel_type}, allow_all={self._allow_all}")

    def shutdown(self) -> None:
        """Shutdown the permission plugin."""
        self._trace("shutdown: cleaning up")
        if self._channel:
            self._channel.shutdown()
        self._policy = None
        self._channel = None
        self._registry = None
        self._initialized = False
        self._wrapped_executors.clear()
        self._original_executors.clear()
        self._allow_all = False
        self._turn_suspended = False
        self._idle_suspended = False
        # Phase 3 §3.7 deeper: drop the cached runner-RPC channel so a
        # subsequent ``initialize()`` re-resolves against the (possibly
        # different) registry's ``runner_rpc_client`` attribute.
        self._runner_rpc_channel = None

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset (Phase 1, server 0.6.142+).

        Per Daniel's litmus test: per-session APPROVAL state should
        NOT survive into the next session.  If the user said "yes,
        always" / "yes, this turn" / "suspend until idle" in session
        A, those decisions are intentionally session-scoped — the
        next cascade stage should re-prompt based on its own tool
        usage, not silently inherit prior approvals.

        Per-session state CLEARED:
        - ``_allow_all``: per-session "approve all" flag.
        - ``_turn_suspended``: per-turn allow-all flag.
        - ``_idle_suspended``: per-session-until-idle flag.
        - ``_execution_log``: per-session tool-execution audit log.
        - ``_agent_name``: per-session identity.

        Survives the reset:
        - ``_config``: workspace-tier policy.
        - ``_policy``: same.
        - ``_channel``: re-wired by next session's lifecycle hooks.
        - ``_workspace_path``: constant within cascade.
        - ``_wrapped_executors`` / ``_original_executors``: re-wired
          by next session's expose-hook.
        - Persistent operator-set whitelist/blacklist on
          ``_policy``: by-design preserved (operator decision, not
          per-session).
        """
        self._trace(
            "reset_for_next_session: clearing per-session approval flags"
        )
        self._allow_all = False
        self._turn_suspended = False
        self._idle_suspended = False
        self._execution_log.clear()
        self._agent_name = None

    def get_config_schema(self) -> dict:
        """Return JSON Schema for this plugin's configuration."""
        return {
            "type": "object",
            "properties": {
                "evaluators": {
                    "type": "object",
                    "description": "Permission evaluator scripts. Maps tool names (or 'default') to script paths.",
                    "additionalProperties": {"type": "string"},
                },
                "policy": {
                    "type": "object",
                    "description": "Permission policy rules",
                    "properties": {
                        "defaultPolicy": {
                            "type": "string",
                            "enum": ["allow", "deny", "ask"],
                            "default": "deny",
                            "description": "Default action when no rule matches",
                        },
                        "sanitization": {
                            "type": "object",
                            "description": "Input sanitization for CLI commands",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable sanitization checks",
                                },
                                "block_shell_metacharacters": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Block shell metacharacters (;, |, &&, etc.)",
                                },
                                "block_dangerous_commands": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Block dangerous system commands",
                                },
                                "allowed_dangerous_commands": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                    "description": "Dangerous commands to allow (e.g. 'git')",
                                },
                                "path_scope": {
                                    "type": "object",
                                    "description": "Filesystem path restrictions",
                                    "properties": {
                                        "enabled": {
                                            "type": "boolean",
                                            "default": True,
                                            "description": "Enable path scope enforcement",
                                        },
                                        "allowed_roots": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "default": ["."],
                                            "description": "Allowed root directories",
                                        },
                                        "block_absolute": {
                                            "type": "boolean",
                                            "default": True,
                                            "description": "Block absolute paths",
                                        },
                                        "block_parent_traversal": {
                                            "type": "boolean",
                                            "default": True,
                                            "description": "Block parent directory traversal (../)",
                                        },
                                        "allow_home": {
                                            "type": "boolean",
                                            "default": False,
                                            "description": "Allow home directory access (~)",
                                        },
                                        "allow_tmp": {
                                            "type": "boolean",
                                            "default": True,
                                            "description": "Allow /tmp directory access",
                                        },
                                    },
                                },
                            },
                        },
                        "blacklist": {
                            "type": "object",
                            "description": "Tools and patterns to always deny",
                            "properties": {
                                "tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                    "description": "Tool names to blacklist",
                                },
                                "patterns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                    "description": "Glob patterns to blacklist",
                                },
                                "arguments": {
                                    "type": "object",
                                    "description": "Per-tool argument patterns to blacklist",
                                    "additionalProperties": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                        "whitelist": {
                            "type": "object",
                            "description": "Tools and patterns to always allow",
                            "properties": {
                                "tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                    "description": "Tool names to whitelist",
                                },
                                "patterns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                    "description": "Glob patterns to whitelist",
                                },
                                "arguments": {
                                    "type": "object",
                                    "description": "Per-tool argument patterns to whitelist",
                                    "additionalProperties": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

    def add_whitelist_tools(self, tools: List[str]) -> None:
        """Add tools to the permission whitelist.

        Use this to programmatically whitelist tools that should be auto-approved,
        such as those returned by plugins' get_auto_approved_tools().

        Phase 3 §3.7 + peer-review M3: acquires ``_policy_lock`` so
        the mutation can't race with an in-flight ASK on the same
        tool — without the lock, an operator's
        ``permission.add_rule`` arriving mid-prompt would let the
        next call for the tool bypass the prompt nondeterministically.

        Args:
            tools: List of tool names to whitelist.
        """
        if self._policy and tools:
            with self._policy_lock:
                for tool in tools:
                    self._policy.whitelist_tools.add(tool)

    def add_framework_reserved_tools(self, tools: List[str]) -> None:
        """Record framework-machinery tool names exempt from the catch-all
        ``"default"`` permission evaluator.

        Framework machinery = core infra (introspection, stream, event-bus,
        registered via ``register_core_tool``) + lifecycle terminals
        (``signal_completion``, registered session-level via
        ``executor.register`` — NOT a registry core tool, so ``is_core_tool``
        alone misses it).  A business default-deny evaluator (``DENY any tool
        not in my whitelist``) must not be able to veto these — else a
        locked-down agent does its work but can never complete.  A
        tool-SPECIFIC evaluator keyed to the name STILL governs (only the
        catch-all collateral is prevented).

        Called from :meth:`JaatoSession.configure` every session, so the set
        is re-populated even after :meth:`shutdown` nulls other state.

        Args:
            tools: Framework-reserved tool names to exempt.
        """
        self._framework_reserved.update(tools)

    # Suspension management methods

    def clear_turn_suspension(self) -> None:
        """Clear turn-scoped permission suspension.

        Called when a turn ends (model returns final response) to restore
        normal permission prompting for the next turn.
        """
        if self._turn_suspended:
            self._trace("clear_turn_suspension: clearing turn suspension")
            self._turn_suspended = False

    def clear_idle_suspension(self) -> None:
        """Clear idle-scoped permission suspension.

        Called when the session transitions to idle state (awaiting user input)
        to restore normal permission prompting.
        """
        if self._idle_suspended:
            self._trace("clear_idle_suspension: clearing idle suspension")
            self._idle_suspended = False

    def clear_all_suspensions(self) -> None:
        """Clear all temporary permission suspensions.

        Clears both turn and idle suspensions. Called by 'permissions resume'.
        Does NOT clear _allow_all (session-wide pre-approval).
        """
        self._trace("clear_all_suspensions: clearing all suspensions")
        self._turn_suspended = False
        self._idle_suspended = False

    def suspend_for_turn(self) -> None:
        """Suspend permission prompts for the remainder of this turn.

        All permission requests will be auto-approved until the turn ends.
        """
        self._trace("suspend_for_turn: activating turn suspension")
        self._turn_suspended = True

    def suspend_until_idle(self) -> None:
        """Suspend permission prompts until session goes idle.

        All permission requests will be auto-approved until the session
        returns to idle state (awaiting user input).
        """
        self._trace("suspend_until_idle: activating idle suspension")
        self._idle_suspended = True

    @property
    def is_suspended(self) -> bool:
        """Check if any suspension is currently active."""
        return self._turn_suspended or self._idle_suspended or self._allow_all

    @property
    def suspension_scope(self) -> Optional[str]:
        """Get the current suspension scope, if any.

        Returns:
            "turn" if turn-suspended, "idle" if idle-suspended,
            "session" if allow_all, None if not suspended.
        """
        if self._turn_suspended:
            return "turn"
        if self._idle_suspended:
            return "idle"
        if self._allow_all:
            return "session"
        return None

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return function declarations for permission tools.

        Note: The askPermission tool is NOT exposed by default. The model should
        call tools directly, and the permission middleware (set via
        executor.set_permission_plugin()) will prompt for approval when needed.

        Exposing askPermission to the model causes confusion - models tend to
        call askPermission instead of the actual tool, creating a redundant
        permission flow.
        """
        # Return empty list - askPermission is not exposed to the model
        # Permission enforcement happens via middleware, not via model tool calls
        return []

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executors for model tools and user commands.

        Exposure is controlled via the registry (expose_tool/unexpose_tool).

        Phase 3 §3.7: forwards via runner-RPC when a runner is
        attached.  ``askPermission`` ASK relays through
        ``client.prompt_operator`` (§3.2.1) once the channel
        migration lands; for now the daemon-side instance still
        owns the channel state and the runner-side ``askPermission``
        invocation forwards the ``_execute_ask_permission`` body
        unchanged.

        The cross-cutting ``check_permission`` method (line 1058)
        is NOT in this dict — it's called directly by
        ``ToolExecutor`` at every tool dispatch and stays daemon-
        side until the seat-flip routes the model loop to the
        runner.
        """
        return self.wrap_executors_for_runner_forwarding({
            "askPermission": self._execute_ask_permission,
            # User commands
            "permissions": self.execute_permissions,
        })

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the permission system."""
        # No system instructions needed - the model should just call tools directly.
        # The permission middleware handles tool access control transparently.
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved.

        The 'permissions' user command is auto-approved since it's
        invoked directly by the user for session management.
        """
        return ["permissions"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for on-the-fly permission management."""
        return [
            UserCommand(
                name="permissions",
                description="Manage session permissions: show, allow <pattern>, deny <pattern>, default <policy>, clear",
                share_with_model=False,
            )
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for permissions command arguments.

        Provides autocompletion for:
        - Subcommands: show, allow, deny, default, clear
        - Default policy options: allow, deny, ask
        - Tool names for allow/deny subcommands
        """
        if command != "permissions":
            return []

        # Subcommand completions
        subcommands = [
            CommandCompletion("show", "Display current effective policy"),
            CommandCompletion("status", "Quick view of suspension state"),
            CommandCompletion("check", "Test what decision a tool would get"),
            CommandCompletion("allow", "Add tool/pattern to session whitelist"),
            CommandCompletion("deny", "Add tool/pattern to session blacklist"),
            CommandCompletion("default", "Set session default policy"),
            CommandCompletion("suspend", "Suspend prompting (--turn for turn only)"),
            CommandCompletion("resume", "Resume normal prompting"),
            CommandCompletion("clear", "Reset all session modifications"),
            CommandCompletion("help", "Show detailed help for this command"),
        ]

        # Policy options for "default" subcommand
        default_options = [
            CommandCompletion("allow", "Auto-approve all tools"),
            CommandCompletion("deny", "Auto-deny all tools"),
            CommandCompletion("ask", "Prompt for each tool"),
        ]

        if not args:
            # No args yet - return all subcommands
            return subcommands

        if len(args) == 1:
            # Partial subcommand - filter matching ones
            partial = args[0].lower()
            return [c for c in subcommands if c.value.startswith(partial)]

        if len(args) == 2:
            subcommand = args[0].lower()
            partial = args[1].lower()

            if subcommand == "default":
                # "permissions default <partial>" - filter policy options
                return [c for c in default_options if c.value.startswith(partial)]

            if subcommand in ("allow", "deny"):
                # "permissions allow/deny <partial>" - provide tool names
                # Filter based on current status (don't show already allowed/denied)
                return self._get_tool_completions(partial, exclude_mode=subcommand)

            if subcommand == "check":
                # "permissions check <partial>" - provide all tool names
                return self._get_tool_completions(partial)

            if subcommand == "suspend":
                # "permissions suspend <partial>" - offer --turn flag
                suspend_options = [
                    CommandCompletion("--turn", "Suspend for this turn only"),
                ]
                return [c for c in suspend_options if c.value.startswith(partial)]

        return []

    def _get_tool_completions(
        self, partial: str, exclude_mode: Optional[str] = None
    ) -> List[CommandCompletion]:
        """Get tool name completions matching the partial input.

        Args:
            partial: Partial tool name to match.
            exclude_mode: If "allow", exclude tools already in session whitelist.
                         If "deny", exclude tools already in session blacklist.
                         Base config rules are NOT excluded since session rules
                         may need to override patterns (e.g., session blacklist
                         "create*" blocks a base-whitelisted "createPlan").
        """
        completions = []

        # Build exclusion set based on mode
        excluded: set = set()
        if self._policy and exclude_mode:
            if exclude_mode == "allow":
                # Only exclude tools already in SESSION whitelist
                # Tools in base whitelist may still need session whitelist entry
                # to override session blacklist patterns (e.g., "deny: create*")
                excluded = self._policy.session_whitelist
            elif exclude_mode == "deny":
                # Only exclude tools already in SESSION blacklist
                # Tools in base blacklist may still need session blacklist entry
                # to override session whitelist patterns
                excluded = self._policy.session_blacklist

        # Get tools from registry
        if self._registry:
            for decl in self._registry.get_exposed_tool_schemas():
                if decl.name in excluded:
                    continue
                if decl.name.lower().startswith(partial):
                    desc = decl.description or ""
                    # Truncate long descriptions
                    if len(desc) > 50:
                        desc = desc[:47] + "..."
                    completions.append(CommandCompletion(decl.name, desc))

        # Include our own tools (askPermission)
        for decl in self.get_tool_schemas():
            if decl.name in excluded:
                continue
            if decl.name.lower().startswith(partial):
                desc = decl.description or ""
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                completions.append(CommandCompletion(decl.name, desc))

        return completions

    def execute_permissions(self, args: Dict[str, Any]) -> str:
        """Execute the permissions user command.

        Subcommands:
            show              - Display current effective policy with diff from base
            status            - Quick view of current suspension state
            check <tool>      - Test what decision a tool would get (uses real evaluation)
            allow <pattern>   - Add tool/pattern to session whitelist
            deny <pattern>    - Add tool/pattern to session blacklist
            default <policy>  - Set session default policy (allow|deny|ask)
            suspend [--turn]  - Suspend prompting (until idle, or just this turn)
            resume            - Resume normal permission prompting
            clear             - Reset all session modifications

        Args:
            args: Dict with 'args' key containing list of command arguments

        Returns:
            Formatted string output for display to user
        """
        cmd_args = args.get("args", [])

        if not cmd_args:
            return self._permissions_show()

        subcommand = cmd_args[0].lower()

        if subcommand == "show":
            return self._permissions_show()
        elif subcommand == "status":
            return self._permissions_status()
        elif subcommand == "check":
            if len(cmd_args) < 2:
                return "Usage: permissions check <tool_name>"
            tool_name = cmd_args[1]
            return self._permissions_check(tool_name)
        elif subcommand == "allow":
            if len(cmd_args) < 2:
                return "Usage: permissions allow <tool_or_pattern>"
            pattern = " ".join(cmd_args[1:])
            return self._permissions_allow(pattern)
        elif subcommand == "deny":
            if len(cmd_args) < 2:
                return "Usage: permissions deny <tool_or_pattern>"
            pattern = " ".join(cmd_args[1:])
            return self._permissions_deny(pattern)
        elif subcommand == "default":
            if len(cmd_args) < 2:
                return "Usage: permissions default <allow|deny|ask>"
            policy = cmd_args[1].lower()
            return self._permissions_default(policy)
        elif subcommand == "suspend":
            # Check for --turn flag
            turn_only = "--turn" in cmd_args[1:] if len(cmd_args) > 1 else False
            return self._permissions_suspend(turn_only=turn_only)
        elif subcommand == "resume":
            return self._permissions_resume()
        elif subcommand == "clear":
            return self._permissions_clear()
        elif subcommand == "help":
            return self._permissions_help()
        else:
            return (
                f"Unknown subcommand: {subcommand}\n"
                "Usage: permissions <show|status|check|allow|deny|default|suspend|resume|clear|help>\n"
                "  show              - Display current effective policy\n"
                "  status            - Quick view of suspension state\n"
                "  check <tool>      - Test what decision a tool would get\n"
                "  allow <pattern>   - Add to session whitelist\n"
                "  deny <pattern>    - Add to session blacklist\n"
                "  default <policy>  - Set session default (allow|deny|ask)\n"
                "  suspend [--turn]  - Suspend prompting (until idle, or --turn for this turn only)\n"
                "  resume            - Resume normal prompting\n"
                "  clear             - Reset session modifications\n"
                "  help              - Show detailed help"
            )

    def _permissions_show(self) -> str:
        """Show current effective permission policy with diff from base."""
        lines = []
        lines.append("Effective Permission Policy")
        lines.append("═" * 27)
        lines.append("")

        if not self._policy:
            lines.append("Permission plugin not initialized.")
            return "\n".join(lines)

        # Suspension status
        if self._idle_suspended:
            lines.append("⚡ Status: SUSPENDED (until-idle)")
        elif self._turn_suspended:
            lines.append("⚡ Status: SUSPENDED (turn-scope)")
        elif self._allow_all:
            lines.append("⚡ Status: SUSPENDED (session-scope, allow-all)")
        else:
            lines.append("Status: Normal prompting")

        lines.append("")

        # Effective default policy
        session_default = self._policy.session_default_policy
        base_default = self._policy.default_policy
        if session_default:
            lines.append(f"Default Policy: {session_default} (session override, was: {base_default})")
        else:
            lines.append(f"Default Policy: {base_default}")

        lines.append("")

        # Session rules
        lines.append("Session Rules:")
        session_whitelist = sorted(self._policy.session_whitelist)
        session_blacklist = sorted(self._policy.session_blacklist)

        if not session_whitelist and not session_blacklist and not session_default:
            lines.append("  (none)")
        else:
            for pattern in session_whitelist:
                lines.append(f"  + allow: {pattern}")
            for pattern in session_blacklist:
                lines.append(f"  - deny:  {pattern}")

        lines.append("")

        # Base config
        lines.append("Base Config:")
        whitelist_tools = sorted(self._policy.whitelist_tools)
        whitelist_patterns = self._policy.whitelist_patterns
        blacklist_tools = sorted(self._policy.blacklist_tools)
        blacklist_patterns = self._policy.blacklist_patterns

        all_whitelist = whitelist_tools + whitelist_patterns
        all_blacklist = blacklist_tools + blacklist_patterns

        if all_whitelist:
            lines.append(f"  Whitelist: {', '.join(all_whitelist)}")
        else:
            lines.append("  Whitelist: (none)")

        if all_blacklist:
            lines.append(f"  Blacklist: {', '.join(all_blacklist)}")
        else:
            lines.append("  Blacklist: (none)")

        return "\n".join(lines)

    def _permissions_help(self) -> HelpLines:
        """Show detailed help for the permissions command."""
        return HelpLines(lines=[
            ("Permissions Command", "bold"),
            ("", ""),
            ("Manage tool execution permissions. Control which tools the model can use", ""),
            ("and how permission prompts are handled.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    permissions [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    show              Display the current effective permission policy", "dim"),
            ("                      Shows base config, session overrides, and status", "dim"),
            ("                      (this is the default when no subcommand is given)", "dim"),
            ("", ""),
            ("    status            Quick view of current suspension state", "dim"),
            ("                      Shows if permissions are suspended and why", "dim"),
            ("", ""),
            ("    check <tool>      Test what decision a tool would get", "dim"),
            ("                      Shows ALLOW/DENY/ASK and the matching rule", "dim"),
            ("", ""),
            ("    allow <pattern>   Add tool or pattern to session whitelist", "dim"),
            ("                      Patterns support wildcards (e.g., 'file_*')", "dim"),
            ("", ""),
            ("    deny <pattern>    Add tool or pattern to session blacklist", "dim"),
            ("                      Blocked tools will fail without prompting", "dim"),
            ("", ""),
            ("    default <policy>  Set the session default policy", "dim"),
            ("                      Options: allow, deny, ask", "dim"),
            ("", ""),
            ("    suspend           Suspend permission prompts until session goes idle", "dim"),
            ("    suspend --turn    Suspend prompts for the current turn only", "dim"),
            ("", ""),
            ("    resume            Resume normal permission prompting", "dim"),
            ("                      Clears any suspension state", "dim"),
            ("", ""),
            ("    clear             Reset all session permission modifications", "dim"),
            ("                      Removes whitelist/blacklist entries and default", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    permissions                      Show current policy", "dim"),
            ("    permissions allow Bash           Always allow Bash tool", "dim"),
            ("    permissions deny file_edit*      Block all file edit tools", "dim"),
            ("    permissions check web_search     Test what web_search would get", "dim"),
            ("    permissions default allow        Auto-approve all tools", "dim"),
            ("    permissions suspend              Stop prompting until idle", "dim"),
            ("    permissions suspend --turn       Stop prompting this turn only", "dim"),
            ("    permissions resume               Resume normal prompting", "dim"),
            ("    permissions clear                Reset to base config", "dim"),
            ("", ""),
            ("PERMISSION PROMPT RESPONSES", "bold"),
            ("    When prompted for permission, you can respond with:", ""),
            ("    [y]es     - Allow this execution", "dim"),
            ("    [n]o      - Deny this execution", "dim"),
            ("    [a]lways  - Allow and remember for this session", "dim"),
            ("    [never]   - Deny and block for this session", "dim"),
            ("    [once]    - Allow just this once", "dim"),
            ("    [t]urn    - Allow all remaining tools this turn", "dim"),
            ("    [i]dle    - Allow until session goes idle", "dim"),
            ("", ""),
            ("PATTERN MATCHING", "bold"),
            ("    Patterns support fnmatch-style wildcards:", ""),
            ("    *         - Match any characters", "dim"),
            ("    ?         - Match single character", "dim"),
            ("    [seq]     - Match any character in seq", "dim"),
            ("    [!seq]    - Match any character not in seq", "dim"),
            ("", ""),
            ("CONFIGURATION FILE", "bold"),
            ("    Base permissions can be configured in .jaato/permissions.json:", ""),
            ("", ""),
            ('    {', "dim"),
            ('      "default": "ask",', "dim"),
            ('      "whitelist": ["introspection*", "todo*"],', "dim"),
            ('      "blacklist": ["*dangerous*"]', "dim"),
            ('    }', "dim"),
        ])

    def _permissions_check(self, tool_name: str) -> str:
        """Check what decision a specific tool would get.

        This uses the actual policy.check() evaluation engine, ensuring
        the result exactly matches what would happen during tool execution.
        """
        if not self._policy:
            return "Error: Permission plugin not initialized."

        # Use the real evaluation engine
        match = self._policy.check(tool_name, {})

        # Format decision
        decision_symbol = {
            "ALLOW": "✓",
            "DENY": "✗",
            "ASK_CHANNEL": "?",
        }.get(match.decision.name, "•")

        lines = [f"{tool_name} → {decision_symbol} {match.decision.name}"]
        lines.append(f"  Reason: {match.reason}")

        if match.rule_type:
            lines.append(f"  Rule type: {match.rule_type}")

        if match.matched_rule:
            lines.append(f"  Matched rule: {match.matched_rule}")

        # Show helpful context for session rule interactions
        if match.rule_type == "session_whitelist" and tool_name in self._policy.session_whitelist:
            # Check if there's a pattern in session_blacklist that would have matched
            for pattern in self._policy.session_blacklist:
                if pattern != tool_name and fnmatch.fnmatch(tool_name, pattern):
                    lines.append(f"  Note: Explicit whitelist overrides blacklist pattern '{pattern}'")
                    break

        return "\n".join(lines)

    def _permissions_allow(self, pattern: str) -> str:
        """Add a pattern to the session whitelist."""
        if not self._policy:
            return "Error: Permission plugin not initialized."

        self._policy.add_session_whitelist(pattern)
        return f"+ Added to session whitelist: {pattern}"

    def _permissions_deny(self, pattern: str) -> str:
        """Add a pattern to the session blacklist."""
        if not self._policy:
            return "Error: Permission plugin not initialized."

        self._policy.add_session_blacklist(pattern)
        return f"- Added to session blacklist: {pattern}"

    def _permissions_default(self, policy: str) -> str:
        """Set the session default policy."""
        if not self._policy:
            return "Error: Permission plugin not initialized."

        if policy not in ("allow", "deny", "ask"):
            return "Invalid policy. Use: allow, deny, or ask"

        old_effective = self._policy.session_default_policy or self._policy.default_policy
        self._policy.set_session_default_policy(policy)
        return f"Session default policy: {policy} (was: {old_effective})"

    def _permissions_clear(self) -> str:
        """Clear all session permission modifications."""
        if not self._policy:
            return "Error: Permission plugin not initialized."

        self._policy.clear_session_rules()
        return "Session rules cleared.\nReverted to base config."

    def _permissions_status(self) -> str:
        """Show quick status of permission prompting state."""
        lines = []

        if self._idle_suspended:
            lines.append("Prompting: SUSPENDED (until-idle)")
            lines.append("  All tool requests auto-approved until session goes idle.")
            lines.append("  Use 'permissions resume' to restore prompting early.")
        elif self._turn_suspended:
            lines.append("Prompting: SUSPENDED (turn-scope)")
            lines.append("  All tool requests auto-approved for remainder of this turn.")
            lines.append("  Will auto-resume when turn completes.")
        elif self._allow_all:
            lines.append("Prompting: SUSPENDED (session-scope)")
            lines.append("  All tool requests auto-approved for this session.")
            lines.append("  Use 'permissions resume' to restore prompting.")
        else:
            lines.append("Prompting: NORMAL")
            lines.append("  Tools checked against whitelist/blacklist.")
            lines.append("  Unknown tools will prompt for approval.")

        return "\n".join(lines)

    def get_permission_status(self) -> Dict[str, Any]:
        """Get structured permission status for UI display.

        Returns:
            Dict with:
                - effective_default: "allow", "deny", or "ask"
                - suspension_scope: "turn", "idle", "session", or None
                - is_suspended: bool
        """
        # Determine effective default policy
        if self._policy:
            effective_default = (
                self._policy.session_default_policy or self._policy.default_policy
            )
        else:
            effective_default = "ask"

        return {
            "effective_default": effective_default,
            "suspension_scope": self.suspension_scope,
            "is_suspended": self.is_suspended,
        }

    def _permissions_suspend(self, turn_only: bool = False) -> str:
        """Suspend permission prompting.

        Args:
            turn_only: If True, suspend only for this turn. Otherwise until idle.
        """
        if turn_only:
            if self._turn_suspended:
                return "Turn suspension already active."
            self._turn_suspended = True
            return "Prompting suspended for this turn.\nWill auto-resume when turn completes."
        else:
            if self._idle_suspended:
                return "Idle suspension already active."
            self._idle_suspended = True
            return "Prompting suspended until session goes idle.\nUse 'permissions resume' to restore prompting early."

    def _permissions_resume(self) -> str:
        """Resume normal permission prompting."""
        was_suspended = self._turn_suspended or self._idle_suspended or self._allow_all

        self._turn_suspended = False
        self._idle_suspended = False
        self._allow_all = False

        if was_suspended:
            return "Prompting resumed. All suspensions cleared."
        else:
            return "Prompting was not suspended."

    def _execute_ask_permission(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the askPermission tool.

        This allows the model to proactively check if a tool is allowed
        before attempting to execute it. If approved, the tool is added to
        the session whitelist so the actual execution won't prompt again.
        """
        tool_name = args.get("tool_name", "")
        tool_args = args.get("arguments", {})
        intent = args.get("intent", "")
        self._trace(f"askPermission: tool={tool_name}, intent={intent!r}")

        if not tool_name:
            return {"error": "tool_name is required"}

        if not intent:
            return {"error": "intent is required - explain what you intend to achieve with this tool"}

        # Pass intent in context for channel to display
        context = {"intent": intent}
        allowed, perm_info = self.check_permission(tool_name, tool_args, context)

        # Carry THIS approval to the imminent real execution so the user is
        # not prompted twice for one decision -- and nothing further.
        #
        # This used to call ``add_session_whitelist(tool_name)`` on ANY
        # approval, which escalated a per-COMMAND grant into a whole-TOOL
        # session grant: with defaultPolicy=deny + whitelist.patterns=
        # ['git *'], `rm -rf /tmp` was denied on a fresh plugin but ALLOWED
        # once `git status` had been approved.  It also turned an explicit
        # ALLOW_ONCE into a session grant.  Every decision that legitimately
        # wants a session grant already takes it itself in
        # ``_handle_channel_response`` (ALLOW_SESSION whitelists, ALLOW_ALL /
        # ALLOW_TURN / ALLOW_UNTIL_IDLE set their own flags), so this line was
        # redundant where it was right and wrong everywhere else.
        #
        # Only an interactive approval can double-prompt; pattern, config and
        # default decisions re-evaluate deterministically to the same answer.
        if allowed and perm_info.get('method') == 'user_approved':
            self._grant_ask_once(tool_name, tool_args)

        return {
            "allowed": allowed,
            "reason": perm_info.get('reason', ''),
            "method": perm_info.get('method', 'unknown'),
            "tool_name": tool_name,
        }

    def _reliability_escalation_action(self, tool_name: str) -> Optional[str]:
        """Reliability Phase-2 enforcement action for ``tool_name`` in the
        current session, or ``None`` for no enforcement.

        Returns:
            ``"ask"``  — the reactor flagged the tool ESCALATED and the client is
                INTERACTIVE (terminal / web / chat): re-confirm with the user
                even when whitelisted / allow_all / suspended (Phase-2 increment 1).
            ``"deny"`` — escalated AND the client is HEADLESS (``ClientType.API``):
                no human to prompt, so block the tool (T1).  Non-blocking — a
                cascade never synchronously waits here (the §7c invariant); the
                reactor's nudge tells the model why, and T2/T3 layer out-of-band
                human approval on top.
            ``None``   — nothing is escalated (the common case), no session
                context, or the client type is unknown (no presentation context
                → no enforcement, the safe default).

        The escalated-tools set is written into session-attached state by the
        reliability reactor under the key ``reliability:escalated_tools``.
        """
        from shared.session_context import get_current_session
        try:
            sess = get_current_session()
        except LookupError:
            return None
        # T3 approved-override (§9 resume primitive): a human-approved tool —
        # written to ``reliability:approved_tools`` by the reactor's gate.released
        # resume handler — is ALLOWED even while still flagged escalated.
        # "approved" wins over "escalated", so the parked cascade's retried call
        # passes once the human approves.  Read BEFORE the escalated check so
        # approval short-circuits both the "ask" and "deny" branches.
        approved = sess.get_session_state("reliability:approved_tools")
        if approved and tool_name in approved:
            return None
        escalated = sess.get_session_state("reliability:escalated_tools")
        if not escalated or tool_name not in escalated:
            return None
        pres = getattr(sess, "_presentation_context", None)
        client_type = getattr(pres, "client_type", None)
        from jaato_sdk.events import ClientType
        if client_type in (ClientType.TERMINAL, ClientType.WEB, ClientType.CHAT):
            return "ask"
        if client_type == ClientType.API:
            return "deny"
        return None  # unknown presentation → no enforcement (safe default)

    _ASK_GRANT_LIMIT = 64

    @staticmethod
    def _ask_grant_key(tool_name: str, args: Optional[Dict[str, Any]]) -> str:
        """Canonical key identifying ONE specific tool call.

        Sorted-key JSON so argument ordering can't produce two keys for the
        same call; ``default=str`` because tool args may carry non-JSON
        values and this only has to be stable, not round-trippable.
        """
        return json.dumps(
            [tool_name, args or {}], sort_keys=True, default=str,
        )

    def _grant_ask_once(self, tool_name: str, args: Optional[Dict[str, Any]]) -> None:
        """Record that THIS exact call was approved via ``askPermission``.

        Consumed once, by :meth:`_consume_ask_grant`, to suppress the
        duplicate interactive prompt the imminent real execution would
        otherwise raise.  It grants nothing else: a different argument set,
        a second execution, or any other tool re-evaluates from scratch.
        """
        key = self._ask_grant_key(tool_name, args)
        with self._ask_grants_lock:
            self._ask_grants[key] = None
            self._ask_grants.move_to_end(key)
            while len(self._ask_grants) > self._ASK_GRANT_LIMIT:
                self._ask_grants.popitem(last=False)

    def _consume_ask_grant(self, tool_name: str, args: Optional[Dict[str, Any]]) -> bool:
        """Consume a one-shot ``askPermission`` grant for this exact call."""
        key = self._ask_grant_key(tool_name, args)
        with self._ask_grants_lock:
            return self._ask_grants.pop(key, "missing") is None

    def check_permission(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if a tool execution is permitted.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool
            context: Optional context for channel (session_id, turn_number, etc.)
            call_id: Optional unique identifier for this tool call (for parallel tool matching)

        Returns:
            Tuple of (is_allowed, metadata_dict) where metadata_dict contains:
            - 'reason': Human-readable reason string
            - 'method': Decision method ('whitelist', 'blacklist', 'default',
                       'sanitization', 'session_whitelist', 'session_blacklist',
                       'user_approved', 'user_denied', 'allow_all', 'timeout')
        """
        self._trace(f"check_permission: tool={tool_name} call_id={call_id}")

        # Trusted bridge: when a plugin-provided interpreter (today only the
        # notebook plugin's Python tool bindings) wraps dispatch in
        # trusted_bridge_context(), the outer tool call was already approved and
        # the user saw every inner ``tools.X(...)`` call in the approved code, so
        # re-prompting each inner call is redundant noise.  The bridge therefore
        # suppresses only the interactive PROMPT — it does NOT bypass the
        # operator's hard boundaries.  The short-circuit lives at the
        # ASK_CHANNEL branch below, AFTER evaluators, the blacklist, and
        # reliability escalation have run, so a blacklisted / evaluator-denied /
        # escalated tool is still refused even inside the bridge.  (A user
        # approving a cell cannot grant themselves override of the operator's
        # policy.)
        from shared.ai_tool_runner import in_trusted_bridge_context

        # Build evaluator context early — evaluators run even for
        # pre-approved tools so they can override approvals.
        eval_context = EvalContext(
            tool_name=tool_name,
            args=args,
            agent_type=context.get("agent_type", "main") if context else "main",
            agent_name=context.get("agent_name") if context else None,
            session_id=context.get("session_id") if context else None,
            workspace_path=getattr(self, '_workspace_path', None),
            turn_index=context.get("turn_index") if context else None,
            model_preamble=context.get("model_preamble") if context else None,
            # Snapshot (not the live list) of PRIOR decisions this session, so
            # an evaluator can reason over earlier behavior.  The current call's
            # decision isn't appended until after evaluation, so the log holds
            # exactly calls 1..N-1 here.
            execution_log=list(self._execution_log),
        )

        # Run evaluators before pre-approval short-circuits.
        # Evaluators can override pre-approvals (DENY overrides allow_all),
        # but FALLBACK preserves the pre-approval.
        run_evaluators = bool(self._policy and self._policy._evaluators)
        if run_evaluators:
            # Framework-reserved tools (core infra + lifecycle terminals such
            # as ``signal_completion``) are EXEMPT from the catch-all
            # ``"default"`` evaluator: a business default-deny (``DENY any tool
            # not in my whitelist``) must not be able to brick the framework
            # machinery the agent needs to complete its own lifecycle.  A
            # tool-SPECIFIC evaluator keyed to the tool name STILL runs —
            # explicitly governing a reserved tool is honored; only the
            # accidental catch-all collateral is prevented.  Keyed on the
            # self-contained ``_framework_reserved`` set (populated at
            # configure from BOTH registry core tools AND the session's
            # lifecycle tools) — NOT ``registry.is_core_tool``: signal_completion
            # is session-level (not a registry core tool), and the set survives
            # ``shutdown()`` nulling ``_registry`` between sessions.
            has_specific_evaluator = tool_name in self._policy._evaluators
            if not has_specific_evaluator and tool_name in self._framework_reserved:
                run_evaluators = False
                self._trace(
                    f"check_permission: framework-reserved tool '{tool_name}' "
                    f"exempt from the default evaluator (no tool-specific evaluator)"
                )
        if run_evaluators:
            from .evaluator import run_evaluator
            eval_result = run_evaluator(
                self._policy._evaluators, tool_name, args, eval_context
            )
            if eval_result.decision == EvalDecision.ALLOW_WITH_COMMENT:
                # Allow with advisory comment — proceed but inject feedback
                comment = eval_result.comment or ""
                self._log_decision(tool_name, args, "allow", f"Evaluator comment: {comment}")
                return True, {
                    'reason': 'Evaluator granted access with comment',
                    'method': 'evaluator_comment',
                    'comment': comment,
                }
            if eval_result.decision not in (
                EvalDecision.FALLBACK,
                EvalDecision.ALLOW,
                EvalDecision.ALLOW_ONCE,
                EvalDecision.ALLOW_TURN,
                EvalDecision.ALLOW_UNTIL_IDLE,
                EvalDecision.ALLOW_SESSION,
                EvalDecision.ALLOW_ALL,
            ):
                # Evaluator denied — override any pre-approval
                if eval_result.decision == EvalDecision.DENY_WITH_COMMENT:
                    comment = eval_result.comment or "Denied by evaluator"
                    self._log_decision(tool_name, args, "deny", f"Evaluator comment: {comment}")
                    return False, {
                        'reason': f"Tool not executed. Evaluator comment: {comment}",
                        'method': 'evaluator_comment',
                        'comment': comment,
                    }
                elif eval_result.decision == EvalDecision.DENY_SESSION:
                    if self._policy:
                        self._policy.add_session_blacklist(tool_name)
                    self._log_decision(tool_name, args, "deny", "Evaluator session blacklist")
                    return False, {
                        'reason': 'Evaluator denied access',
                        'method': 'evaluator_session_blacklist',
                    }
                else:  # DENY
                    self._log_decision(tool_name, args, "deny", "Evaluator denied access")
                    return False, {
                        'reason': 'Evaluator denied access',
                        'method': 'evaluator',
                    }

        # Reliability Phase-2 escalation enforcement (computed once): for an
        # escalated tool, interactive clients are re-confirmed by the user
        # ("ask") even when otherwise auto-approved; headless clients are denied
        # outright ("deny", T1) since there is no human to prompt.  None for
        # every normal call (a strict no-op until the reactor escalates).
        escalation_action = self._reliability_escalation_action(tool_name)
        if escalation_action == "deny":
            # T1 — headless escalation enforcement.  No human to prompt, so block
            # the escalated tool.  The reactor's nudge already told the model why;
            # T2/T3 layer out-of-band human approval on top.  Non-blocking — a
            # cascade never synchronously waits here (the §7c invariant).
            self._log_decision(
                tool_name, args, "deny", "reliability escalation (headless)"
            )
            return False, {
                'reason': "reliability escalation: tool flagged after repeated "
                          "failures and denied (headless session — no interactive "
                          "approval available). Reconsider the inputs/approach or "
                          "try a different tool.",
                'method': 'reliability_escalation_denied',
            }
        force_reescalation = (escalation_action == "ask")

        # Check suspension states in priority order:
        # 1. idle suspension (most conservative - clears on idle)
        # 2. turn suspension (clears on turn end)
        # 3. allow_all (session-wide, persists until session ends)
        if not force_reescalation and self._idle_suspended:
            self._log_decision(tool_name, args, "allow", "Permission suspended until idle")
            return True, {'reason': 'Permission suspended until idle', 'method': 'idle_suspension'}

        if not force_reescalation and self._turn_suspended:
            self._log_decision(tool_name, args, "allow", "Permission suspended for turn")
            return True, {'reason': 'Permission suspended for turn', 'method': 'turn_suspension'}

        # Check if user pre-approved all requests
        if not force_reescalation and self._allow_all:
            self._log_decision(tool_name, args, "allow", "Pre-approved all requests")
            return True, {'reason': 'Pre-approved all requests', 'method': 'allow_all'}

        if not self._policy:
            return True, {'reason': 'Permission plugin not initialized', 'method': 'not_initialized'}

        # Check if using ParentBridgedChannel (subagent mode)
        # In subagent mode, we don't invoke the parent's hooks as that would
        # incorrectly set the parent's UI to "waiting for permission input"
        from .channels import ParentBridgedChannel
        channel = self._get_channel()
        is_subagent_mode = isinstance(channel, ParentBridgedChannel)
        # Trusted-bridge inner call (see the note at the top of this method):
        # deny-layers still apply, but a resolved ALLOW is kept quiet — the
        # user already approved the outer cell, so per-inner-call UI events
        # would be redundant noise (mirrors the is_subagent_mode skip).
        is_trusted_bridge = in_trusted_bridge_context()

        # Evaluate against policy. Pass eval_context=None if evaluators
        # already ran above (for pre-approved tools) to avoid double execution.
        already_evaluated = bool(self._policy and self._policy._evaluators)
        match = self._policy.check(
            tool_name, args,
            eval_context=None if already_evaluated else eval_context,
        )

        # Reliability escalation re-gate: a whitelisted/auto-allowed tool the
        # reactor escalated (interactive client) must be re-confirmed — turn the
        # ALLOW into a channel prompt so the user decides whether to proceed.
        if force_reescalation and match.decision == PermissionDecision.ALLOW:
            import dataclasses
            match = dataclasses.replace(
                match, decision=PermissionDecision.ASK_CHANNEL,
                reason="reliability escalation: tool flagged after repeated "
                       "failures — re-confirm before running")

        if match.decision == PermissionDecision.ALLOW:
            # Apply scoped side effects from evaluator decisions
            if match.eval_result and match.rule_type == "evaluator":
                ed = match.eval_result.decision
                if ed == EvalDecision.ALLOW_TURN:
                    self._turn_suspended = True
                    method = 'evaluator_turn_suspension'
                elif ed == EvalDecision.ALLOW_UNTIL_IDLE:
                    self._idle_suspended = True
                    method = 'evaluator_idle_suspension'
                elif ed == EvalDecision.ALLOW_SESSION:
                    if self._policy:
                        self._policy.add_session_whitelist(tool_name)
                    method = 'evaluator_session_whitelist'
                elif ed == EvalDecision.ALLOW_ALL:
                    self._allow_all = True
                    method = 'evaluator_allow_all'
                else:
                    method = 'evaluator'
            else:
                method = match.rule_type or 'policy'
            self._log_decision(tool_name, args, "allow", match.reason)
            # Emit resolved hook for auto-approved (whitelist)
            # SKIP in subagent mode
            # Extract comment for ALLOW_WITH_COMMENT
            eval_comment = ""
            if (match.eval_result
                    and match.eval_result.decision == EvalDecision.ALLOW_WITH_COMMENT
                    and match.eval_result.comment):
                eval_comment = match.eval_result.comment
            if self._on_permission_resolved and not is_subagent_mode and not is_trusted_bridge:
                self._on_permission_resolved(tool_name, "", True, method, comment=eval_comment)
            result = {'reason': match.reason, 'method': method}
            # Inject advisory comment for ALLOW_WITH_COMMENT
            if (match.eval_result
                    and match.eval_result.decision == EvalDecision.ALLOW_WITH_COMMENT
                    and match.eval_result.comment):
                result['comment'] = match.eval_result.comment
            return True, result

        elif match.decision == PermissionDecision.DENY:
            # Apply scoped side effects from evaluator decisions
            if match.eval_result and match.eval_result.decision == EvalDecision.DENY_SESSION:
                if self._policy:
                    self._policy.add_session_blacklist(tool_name)
                method = 'evaluator_session_blacklist'
            elif match.rule_type == "evaluator_comment":
                method = 'evaluator_comment'
            else:
                method = match.rule_type or 'policy'
            self._log_decision(tool_name, args, "deny", match.reason)
            # Emit resolved hook for auto-denied (blacklist)
            # SKIP in subagent mode
            if self._on_permission_resolved and not is_subagent_mode:
                self._on_permission_resolved(tool_name, "", False, method)
            return False, {'reason': match.reason, 'method': method, 'comment': match.eval_result.comment if match.eval_result else None}

        elif match.decision == PermissionDecision.ASK_CHANNEL:
            # Trusted bridge suppresses the redundant interactive prompt — but
            # only a NORMAL ask (rule-miss → default). We reach here only after
            # the blacklist / evaluators / reliability-escalation-deny have all
            # passed (those return DENY earlier), so allowing here does not
            # bypass any hard boundary. A reliability-escalation re-confirm
            # (force_reescalation) is NOT suppressed: the reactor raised that
            # signal AFTER the cell was approved, so the user must still see it.
            if is_trusted_bridge and not force_reescalation:
                self._log_decision(
                    tool_name, args, "allow",
                    "trusted bridge (outer tool approved; prompt suppressed)",
                )
                return True, {
                    'reason': 'Allowed via trusted bridge context (outer tool '
                              'already approved); blacklist and evaluators still '
                              'enforced',
                    'method': 'trusted_bridge',
                }

            # A one-shot grant from ``askPermission`` suppresses the duplicate
            # prompt for the SAME call the model just pre-checked.  Placed
            # here deliberately, beside the trusted-bridge short-circuit and
            # under the same guarantee: we only reach this branch after the
            # blacklist, evaluators and reliability-escalation-deny have all
            # passed, so consuming a grant cannot bypass a hard boundary.  A
            # forced re-escalation is NOT suppressed, for the same reason it
            # isn't for the bridge.
            if not force_reescalation and self._consume_ask_grant(tool_name, args):
                self._log_decision(
                    tool_name, args, "allow",
                    "askPermission pre-approval consumed (prompt suppressed)",
                )
                return True, {
                    'reason': 'Approved via askPermission for this exact call; '
                              'blacklist and evaluators still enforced',
                    'method': 'ask_permission_once',
                }

            # Need to ask the channel (already retrieved above for subagent check)
            if not channel:
                self._log_decision(tool_name, args, "deny", "No channel configured")
                return False, {'reason': 'No channel configured for approval', 'method': 'no_channel'}

            # Phase 3 §3.7 + peer-review M3: hold ``_policy_lock``
            # across the rule-recheck + channel-wait critical section
            # so a cross-session ``permission.add_rule`` RPC (or any
            # whitelist/blacklist mutation) can't change the policy
            # state mid-prompt.  Without the lock, the rule-miss
            # decision and the user's response would evaluate against
            # different policy snapshots, leaving subsequent calls to
            # the same tool to behave nondeterministically depending
            # on which thread observed which snapshot.
            #
            # The response-side side effects (``add_session_whitelist``
            # etc. inside ``_handle_channel_response``) run while the
            # lock is still held, so they apply atomically with
            # respect to other mutations as well.
            self._trace(f"check_permission: acquiring policy lock for ASK on {tool_name}")
            with self._policy_lock:
                # Re-check the policy under the lock — a mutation may
                # have landed between the original (unlocked) check
                # above and our acquisition here.  If the policy now
                # resolves the tool unambiguously, drop the lock and
                # recurse to fall through ALLOW/DENY rather than
                # prompting the user about a tool that's already
                # decided.  The recursion sees the new policy state
                # in its own (unlocked) check, so its decision tree
                # lands directly in ALLOW/DENY without re-acquiring
                # the lock.
                recheck = self._policy.check(
                    tool_name, args,
                    eval_context=None if already_evaluated else eval_context,
                )
                policy_mutated = (
                    recheck.decision != PermissionDecision.ASK_CHANNEL
                )
                if policy_mutated:
                    self._trace(
                        f"check_permission: policy mutated mid-ASK "
                        f"for {tool_name}; recheck decision="
                        f"{recheck.decision}; will recurse after "
                        f"releasing lock"
                    )
                else:
                    # Serialize channel interactions to ensure only one permission prompt
                    # is shown at a time (important for parallel tool execution)
                    self._trace(f"check_permission: acquiring channel lock for {tool_name}")
                    with self._channel_lock:
                        # Re-check _allow_all after acquiring lock - another thread may have
                        # set it while we were waiting (e.g., user responded "all" to first prompt)
                        if self._allow_all:
                            self._trace(f"check_permission: allow_all set while waiting, auto-approving {tool_name}")
                            self._log_decision(tool_name, args, "allow", "Pre-approved all requests")
                            return True, {'reason': 'Pre-approved all requests', 'method': 'allow_all'}

                        # Get tool schema to check for editable content
                        tool_schema = self._get_tool_schema(tool_name)
                        editable = tool_schema.editable if tool_schema else None
                        self._trace(f"check_permission: tool_schema={tool_schema is not None}, editable={editable is not None}")

                        # Get permission options (with edit if tool is editable)
                        response_options = self._get_permission_options_for_tool(tool_name)

                        # Track current arguments (may be modified by edit)
                        current_args = args.copy()
                        original_args = args.copy()
                        was_edited = False

                        # Edit loop - user can edit multiple times before final decision
                        # Track the request_id of the last prompt sent to the client
                        # so that pre-validation can reference it when resolving.
                        last_prompted_request_id: str | None = None

                        while True:
                            # Get custom display info from source plugin if available
                            channel_type = channel.name if channel else "console"
                            display_info = self._get_display_info(tool_name, current_args, channel_type)

                            # Pre-validation: if the plugin already knows the operation
                            # will fail (e.g., targeted edit anchor not found), skip the
                            # permission prompt and let the executor return the error
                            # directly so the model can retry.
                            if display_info and display_info.pre_validation_error:
                                self._trace(f"check_permission: pre-validation failed for {tool_name}: {display_info.pre_validation_error}")
                                self._log_decision(tool_name, current_args, "allow", f"Pre-validation error (skipping prompt): {display_info.pre_validation_error}")
                                if self._on_permission_resolved and not is_subagent_mode:
                                    # Use last_prompted_request_id so the client can
                                    # match this resolution to its pending prompt and
                                    # clear the permission input mode.
                                    self._on_permission_resolved(tool_name, last_prompted_request_id or "", True, "pre_validation")
                                return True, {'reason': 'Pre-validation error, skipping prompt', 'method': 'pre_validation'}

                            # Build context with display info
                            request_context = dict(context) if context else {}
                            if display_info:
                                request_context["display_info"] = display_info
                            # Mark as edited in context for UI display
                            if was_edited:
                                request_context["was_edited"] = True
                            # Phase 4 §4.1 (J.A): propagate the active
                            # ``call_id`` so the runner-RPC channel can
                            # thread it through the PromptPayload, and
                            # the daemon's PromptOperatorHandler can
                            # include it in PermissionInputModeEvent
                            # for TUI per-tool-block correlation.
                            if call_id:
                                request_context["call_id"] = call_id

                            request = PermissionRequest.create(
                                tool_name=tool_name,
                                arguments=current_args,
                                timeout=self._config.channel_timeout if self._config else 30,
                                context=request_context,
                                response_options=response_options,
                                editable=editable,
                            )
                            # Set additional metadata for the channel/client
                            request.was_edited = was_edited
                            request.original_arguments = original_args if was_edited else None

                            # Emit permission requested hook with current args (client formats display)
                            # SKIP in subagent mode
                            if self._on_permission_requested and not is_subagent_mode:
                                self._on_permission_requested(
                                    tool_name, request.request_id, current_args, request.response_options, call_id
                                )
                            last_prompted_request_id = request.request_id

                            response = channel.request_permission(request)

                            # Handle EDIT decision - loop back after editing
                            if response.decision == ChannelDecision.EDIT:
                                if response.edited_arguments:
                                    current_args = response.edited_arguments
                                    was_edited = True
                                    self._trace(f"check_permission: content edited for {tool_name}")
                                # Continue loop to re-prompt with edited content
                                continue

                            # Final decision - exit loop
                            allowed, info = self._handle_channel_response(tool_name, current_args, response)

                            # Include edit metadata in info
                            if was_edited:
                                info['was_edited'] = True
                                info['modified_args'] = current_args
                                info['original_args'] = original_args

                            # Emit permission resolved hook
                            # SKIP in subagent mode
                            if self._on_permission_resolved and not is_subagent_mode:
                                self._on_permission_resolved(
                                    tool_name, request.request_id, allowed,
                                    info.get('method', 'unknown'),
                                    comment=info.get('comment', ''),
                                )

                            return allowed, info

            # Lock released.  ``policy_mutated`` is the only way out
            # of the ``with`` block without an inner return — recurse
            # so the new policy state is observed.
            if policy_mutated:
                return self.check_permission(tool_name, args, context, call_id)

        # Unknown decision type, deny by default
        return False, {'reason': 'Unknown policy decision', 'method': 'unknown'}

    def _handle_channel_response(
        self,
        tool_name: str,
        args: Dict[str, Any],
        response: ChannelResponse
    ) -> Tuple[bool, Dict[str, Any]]:
        """Handle response from an channel.

        Updates session rules if channel requests it.

        Returns:
            Tuple of (is_allowed, metadata_dict) with 'reason' and 'method'.
        """
        decision = response.decision

        if decision in (ChannelDecision.ALLOW, ChannelDecision.ALLOW_ONCE):
            self._log_decision(tool_name, args, "allow", response.reason)
            return True, {'reason': response.reason, 'method': 'user_approved'}

        elif decision == ChannelDecision.ALLOW_SESSION:
            # Add to session whitelist
            pattern = response.remember_pattern or tool_name
            if self._policy:
                self._policy.add_session_whitelist(pattern)
            self._log_decision(tool_name, args, "allow", f"Session whitelist: {pattern}")
            return True, {'reason': response.reason, 'method': 'session_whitelist'}

        elif decision == ChannelDecision.ALLOW_ALL:
            # Pre-approve all future requests in this session
            self._allow_all = True
            self._log_decision(tool_name, args, "allow", "Pre-approved all requests")
            return True, {'reason': response.reason, 'method': 'allow_all'}

        elif decision == ChannelDecision.ALLOW_TURN:
            # Suspend prompts for remainder of this turn
            self._turn_suspended = True
            self._log_decision(tool_name, args, "allow", "Permission suspended for turn")
            return True, {'reason': response.reason, 'method': 'turn_suspension'}

        elif decision == ChannelDecision.ALLOW_UNTIL_IDLE:
            # Suspend prompts until session goes idle
            self._idle_suspended = True
            self._log_decision(tool_name, args, "allow", "Permission suspended until idle")
            return True, {'reason': response.reason, 'method': 'idle_suspension'}

        elif decision == ChannelDecision.COMMENT:
            # Deny with user feedback — the comment text is in response.reason
            # and will be included in the tool error message so the model sees it
            self._log_decision(tool_name, args, "deny", f"User comment: {response.reason}")
            return False, {
                'reason': f"Tool not executed. User comment: {response.reason}",
                'method': 'user_comment',
                'comment': response.reason,
            }

        elif decision == ChannelDecision.ALLOW_COMMENT:
            # Allow with user feedback — the comment text is injected into
            # the tool result so the model sees it alongside the output
            self._log_decision(tool_name, args, "allow", f"User comment: {response.reason}")
            return True, {
                'reason': response.reason,
                'method': 'user_comment',
                'comment': response.reason,
            }

        elif decision == ChannelDecision.DENY:
            self._log_decision(tool_name, args, "deny", response.reason)
            return False, {'reason': response.reason, 'method': 'user_denied'}

        elif decision == ChannelDecision.DENY_SESSION:
            # Add to session blacklist
            pattern = response.remember_pattern or tool_name
            if self._policy:
                self._policy.add_session_blacklist(pattern)
            self._log_decision(tool_name, args, "deny", f"Session blacklist: {pattern}")
            return False, {'reason': response.reason, 'method': 'session_blacklist'}

        elif decision == ChannelDecision.TIMEOUT:
            self._log_decision(tool_name, args, "deny", "Channel timeout")
            return False, {'reason': response.reason, 'method': 'timeout'}

        elif decision == ChannelDecision.EDIT:
            # EDIT is handled in check_permission loop, but if we get here
            # it means the channel returned EDIT without edited content
            # Treat as a denial to force re-prompt
            self._log_decision(tool_name, args, "deny", "Edit requested but no content provided")
            return False, {'reason': 'Edit flow incomplete', 'method': 'edit_incomplete'}

        # Unknown decision, deny
        self._log_decision(tool_name, args, "deny", "Unknown channel decision")
        return False, {'reason': 'Unknown channel decision', 'method': 'unknown'}

    def _log_decision(
        self,
        tool_name: str,
        args: Dict[str, Any],
        decision: str,
        reason: str
    ) -> None:
        """Log a permission decision for auditing."""
        self._execution_log.append({
            "tool_name": tool_name,
            "arguments": args,
            "decision": decision,
            "reason": reason,
        })

    def _get_display_info(
        self,
        tool_name: str,
        args: Dict[str, Any],
        channel_type: str
    ) -> Optional[PermissionDisplayInfo]:
        """Get display info for a tool from its source plugin.

        Looks up the plugin that provides the tool and calls its
        format_permission_request() method if available.

        Args:
            tool_name: Name of the tool
            args: Arguments passed to the tool
            channel_type: Type of channel requesting display info

        Returns:
            PermissionDisplayInfo if plugin provides custom formatting, None otherwise.
        """
        if not self._registry:
            return None

        plugin = self._registry.get_plugin_for_tool(tool_name)
        if not plugin:
            return None

        if hasattr(plugin, 'format_permission_request'):
            try:
                return plugin.format_permission_request(tool_name, args, channel_type)
            except Exception:
                # If formatting fails, fall back to default
                return None

        return None

    def _get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """Get the ToolSchema for a given tool name.

        Looks up the plugin that provides the tool and finds the matching schema.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolSchema if found, None otherwise.
        """
        if not self._registry:
            return None

        plugin = self._registry.get_plugin_for_tool(tool_name)
        if not plugin:
            return None

        try:
            for schema in plugin.get_tool_schemas():
                if schema.name == tool_name:
                    return schema
        except Exception:
            pass

        return None

    def _get_permission_options_for_tool(self, tool_name: str) -> List[PermissionResponseOption]:
        """Get permission response options for a tool.

        If the tool has editable content, includes the 'edit' option.

        Args:
            tool_name: Name of the tool

        Returns:
            List of PermissionResponseOption objects.
        """
        schema = self._get_tool_schema(tool_name)
        if schema and schema.editable is not None:
            return get_permission_options_with_edit()
        return get_default_permission_options()

    def _build_prompt_lines(
        self,
        tool_name: str,
        args: Dict[str, Any],
        display_info: Optional[PermissionDisplayInfo],
        response_options: Optional[List[PermissionResponseOption]] = None,
        include_details: bool = True,
        include_options: bool = True
    ) -> List[str]:
        """Build prompt lines for UI display from request info.

        Args:
            tool_name: Name of the tool
            args: Arguments passed to the tool
            display_info: Optional custom display info from plugin
            response_options: List of valid response options (defaults to standard options)
            include_details: Whether to include details in the prompt. Set to False
                when details will be rendered separately (e.g., code blocks).
            include_options: Whether to include the options line. Set to False when
                options are displayed separately (e.g., in input area).

        Returns:
            List of strings representing the permission prompt.
        """
        lines = []

        if display_info:
            # Use custom display info
            lines.append(display_info.summary)
            if include_details and display_info.details:
                # Split details into lines
                for detail_line in display_info.details.split('\n'):
                    lines.append(detail_line)
        else:
            # Default: show tool name and args
            lines.append(f"Tool: {tool_name}")
            if args:
                lines.append(f"Args: {format_tool_args_summary(args, max_length=100)}")

        # Add options line if requested (may be shown separately in input area instead)
        if include_options:
            lines.append("")
            options = response_options or get_default_permission_options()
            lines.append(format_permission_options(options))

        return lines

    def get_formatted_prompt(
        self,
        tool_name: str,
        args: Dict[str, Any],
        channel_type: str = "ipc"
    ) -> Tuple[List[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Get formatted prompt lines for a permission request.

        This is used by the server to include pre-formatted prompts
        (including diffs for file edits) in permission events.

        Args:
            tool_name: Name of the tool
            args: Arguments passed to the tool
            channel_type: Type of channel ("console", "ipc", etc.)

        Returns:
            Tuple of (prompt_lines, format_hint, language, raw_details, warnings, warning_level).
            - prompt_lines: The formatted permission prompt
            - format_hint: "diff" for colored diff, "code" for code, None otherwise
            - language: Programming language when format_hint="code" (e.g., "python")
            - raw_details: Original details content when excluded from prompt_lines
                (e.g., code to be rendered separately)
            - warnings: Security/analysis warnings to display separately
            - warning_level: Severity level ("info", "warning", "error")
        """
        display_info = self._get_display_info(tool_name, args, channel_type)
        format_hint = display_info.format_hint if display_info else None
        language = display_info.language if display_info else None
        raw_details = None
        warnings = display_info.warnings if display_info else None
        warning_level = display_info.warning_level if display_info else None

        # When format_hint is "code", exclude details from prompt so they can be
        # rendered separately with syntax highlighting
        include_details = format_hint != "code"
        if not include_details and display_info and display_info.details:
            raw_details = display_info.details

        lines = self._build_prompt_lines(tool_name, args, display_info, include_details=include_details, include_options=False)
        return lines, format_hint, language, raw_details, warnings, warning_level

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the log of permission decisions."""
        return self._execution_log.copy()

    def clear_execution_log(self) -> None:
        """Clear the execution log."""
        self._execution_log.clear()

    def wrap_executor(
        self,
        name: str,
        executor: Callable[[Dict[str, Any]], Any]
    ) -> Callable[[Dict[str, Any]], Any]:
        """Wrap an executor with permission checking.

        Args:
            name: Tool name
            executor: Original executor function

        Returns:
            Wrapped executor that checks permissions before executing
        """
        self._original_executors[name] = executor

        def wrapped(args: Dict[str, Any]) -> Any:
            allowed, perm_info = self.check_permission(name, args)

            if not allowed:
                return {"error": f"Permission denied: {perm_info.get('reason', '')}", "_permission": perm_info}

            # Use modified args if content was edited by user
            final_args = perm_info.get('modified_args', args)

            result = executor(final_args)

            # Inject permission metadata if result is a dict
            if isinstance(result, dict):
                result['_permission'] = perm_info
                # Add feedback to model about edited content
                if perm_info.get('was_edited'):
                    result['_user_edited'] = True
                    result['_edit_notice'] = (
                        "Note: The user edited the content before execution. "
                        "The executed content differs from what you originally provided."
                    )
            return result

        self._wrapped_executors[name] = wrapped
        return wrapped

    def wrap_all_executors(
        self,
        executors: Dict[str, Callable[[Dict[str, Any]], Any]]
    ) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Wrap all executors in a dict with permission checking.

        Args:
            executors: Dict mapping tool names to executor functions

        Returns:
            Dict with wrapped executors
        """
        wrapped = {}
        for name, executor in executors.items():
            # Don't wrap our own askPermission tool
            if name == "askPermission":
                wrapped[name] = executor
            else:
                wrapped[name] = self.wrap_executor(name, executor)
        return wrapped

    # Interactivity protocol methods

    def supports_interactivity(self) -> bool:
        """Permission plugin requires user interaction for approval prompts.

        Returns:
            True - permission plugin has interactive approval features.
        """
        return True

    def get_supported_channels(self) -> List[str]:
        """Return list of channel types supported by permission plugin.

        Returns:
            List of supported channel types: console, queue, webhook, file, parent_bridged.
        """
        return ["console", "queue", "webhook", "file", "parent_bridged"]

    def configure_for_subagent(self, session: Any) -> None:
        """Configure this plugin for subagent mode in the current thread.

        Sets up the parent-bridged channel and stores it in thread-local storage
        so that permission requests from this subagent are forwarded to the
        parent agent. This doesn't affect the main agent's channel.

        IMPORTANT: This uses thread-local storage because plugins are singletons
        shared across all sessions. Each subagent runs in its own thread, so
        setting the channel in thread-local storage ensures isolation.

        Args:
            session: JaatoSession instance with parent reference.
        """
        from .channels import ParentBridgedChannel
        channel = ParentBridgedChannel()
        channel.set_session(session)
        # Store in thread-local storage, not the shared instance
        self._thread_local.channel = channel

    def set_channel(
        self,
        channel_type: str,
        channel_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the interaction channel for permission prompts.

        Args:
            channel_type: One of: console, queue, webhook, file
            channel_config: Optional channel-specific configuration

        Raises:
            ValueError: If channel_type is not supported
        """
        if channel_type not in self.get_supported_channels():
            raise ValueError(
                f"Channel type '{channel_type}' not supported. "
                f"Supported: {self.get_supported_channels()}"
            )

        # Create the channel with config
        self._channel = create_channel(channel_type, channel_config)


def create_plugin() -> PermissionPlugin:
    """Factory function to create the permission plugin instance."""
    return PermissionPlugin()
