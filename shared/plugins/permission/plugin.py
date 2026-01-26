"""Permission plugin for controlling tool execution access.

This plugin intercepts tool execution requests and enforces access policies
through blacklist/whitelist rules and interactive channel approval.
"""

import fnmatch
import os
import tempfile
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..model_provider.types import ToolSchema

from .policy import PermissionPolicy, PermissionDecision, PolicyMatch
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
)
from ..base import UserCommand, CommandCompletion, PermissionDisplayInfo, OutputCallback
from ...ui_utils import format_permission_options, format_tool_args_summary

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..registry import PluginRegistry


class PermissionPlugin:
    """Plugin that provides permission control for tool execution.

    This plugin acts as a middleware layer that intercepts tool execution
    requests and enforces access policies. It can:
    - Block tools via blacklist rules
    - Allow tools via whitelist rules
    - Prompt an channel for approval when policy is ambiguous

    The plugin has two distinct roles that are controlled independently:

    1. Permission enforcement (middleware):
       - Enabled via: executor.set_permission_plugin(plugin)
       - Wraps ToolExecutor.execute() to check permissions before any tool runs

    2. Proactive check tool (askPermission):
       - Enabled via: registry.expose_tool("permission")
       - Exposes askPermission tool for model to query permissions proactively

    Usage patterns:
    - Enforcement only: set_permission_plugin() without expose_tool()
    - Enforcement + proactive: set_permission_plugin() AND expose_tool()
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
        self._allow_all: bool = False  # When True, auto-approve all requests
        # Suspension state flags for temporary permission bypasses
        self._turn_suspended: bool = False  # Allow all remaining tools this turn
        self._idle_suspended: bool = False  # Allow until session goes idle
        # Lock for serializing channel interactions (permission prompts)
        # This ensures only one permission prompt is shown at a time when
        # multiple tools request permission concurrently (parallel execution)
        self._channel_lock = threading.Lock()
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Permission lifecycle hooks for UI integration
        # on_requested: (tool_name, request_id, tool_args, response_options, call_id) -> None
        self._on_permission_requested: Optional[Callable[[str, str, Dict[str, Any], List[PermissionResponseOption], Optional[str]], None]] = None
        self._on_permission_resolved: Optional[Callable[[str, str, bool, str], None]] = None

    def _get_channel(self) -> Optional[Channel]:
        """Get the channel for the current thread.

        Returns the thread-local channel if set (for subagents),
        otherwise returns the default channel.
        """
        thread_channel = getattr(self._thread_local, 'channel', None)
        return thread_channel if thread_channel is not None else self._channel

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
        on_resolved: Optional[Callable[[str, str, bool, str], None]] = None
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

    @property
    def name(self) -> str:
        return "permission"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [PERMISSION{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

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

    def add_whitelist_tools(self, tools: List[str]) -> None:
        """Add tools to the permission whitelist.

        Use this to programmatically whitelist tools that should be auto-approved,
        such as those returned by plugins' get_auto_approved_tools().

        Args:
            tools: List of tool names to whitelist.
        """
        if self._policy and tools:
            for tool in tools:
                self._policy.whitelist_tools.add(tool)

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
        """Return function declarations for the askPermission tool.

        The askPermission tool allows the model to proactively check if a tool
        is allowed before execution. Exposure is controlled via the registry:

        - registry.expose_tool("permission") -> askPermission available to model
        - Permission enforcement via executor.set_permission_plugin() is separate

        This separation allows:
        - Enforcement only: set_permission_plugin() without expose_tool()
        - Enforcement + proactive checks: both set_permission_plugin() and expose_tool()
        """
        return [
            ToolSchema(
                name="askPermission",
                description="Request permission to execute a tool or proceed with an action. "
                           "This is the ONLY valid way to ask for user approval - never ask in plain text. "
                           "You MUST explain your intent - what you are trying to achieve or discover.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool to check permission for"
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments that would be passed to the tool"
                        },
                        "intent": {
                            "type": "string",
                            "description": "Why you need to execute this tool - what you intend to achieve or discover"
                        }
                    },
                    "required": ["tool_name", "intent"]
                },
                category="system",
                discoverability="discoverable",
            )
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executors for model tools and user commands.

        Exposure is controlled via the registry (expose_tool/unexpose_tool).
        """
        return {
            "askPermission": self._execute_ask_permission,
            # User commands
            "permissions": self.execute_permissions,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the permission system."""
        return """Tool execution is controlled by a permission system.

CRITICAL: When you need user approval or permission to proceed, you MUST use the `askPermission` tool.
DO NOT ask for permission in plain text like "Do you approve?" or "Please confirm".
Plain text permission requests are NOT valid and will NOT be processed by the permission system.

WRONG (never do this):
- "Please review the plan. Do you approve?"
- "Should I proceed with this implementation?"
- "Do I have your permission to continue?"

CORRECT (always use the tool):
- Call `askPermission` with tool_name, intent, and optional arguments

The askPermission tool takes:
- tool_name: Name of the tool or action to check permission for
- intent: (REQUIRED) A clear explanation of what you intend to achieve or discover
- arguments: (optional) Arguments or details about the specific action

You MUST always provide an intent explaining WHY you need to execute the tool.
The intent should describe what you are trying to accomplish, not just repeat the command.

It returns whether the action is allowed and the reason for the decision.
If permission is denied, do not attempt to proceed with that action."""

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
        else:
            return (
                f"Unknown subcommand: {subcommand}\n"
                "Usage: permissions <show|status|check|allow|deny|default|suspend|resume|clear>\n"
                "  show              - Display current effective policy\n"
                "  status            - Quick view of suspension state\n"
                "  check <tool>      - Test what decision a tool would get\n"
                "  allow <pattern>   - Add to session whitelist\n"
                "  deny <pattern>    - Add to session blacklist\n"
                "  default <policy>  - Set session default (allow|deny|ask)\n"
                "  suspend [--turn]  - Suspend prompting (until idle, or --turn for this turn only)\n"
                "  resume            - Resume normal prompting\n"
                "  clear             - Reset session modifications"
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

        # If approved, add to session whitelist so actual execution won't prompt again
        if allowed and self._policy:
            self._policy.add_session_whitelist(tool_name)

        return {
            "allowed": allowed,
            "reason": perm_info.get('reason', ''),
            "method": perm_info.get('method', 'unknown'),
            "tool_name": tool_name,
        }

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

        # Check suspension states in priority order:
        # 1. idle suspension (most conservative - clears on idle)
        # 2. turn suspension (clears on turn end)
        # 3. allow_all (session-wide, persists until session ends)
        if self._idle_suspended:
            self._log_decision(tool_name, args, "allow", "Permission suspended until idle")
            return True, {'reason': 'Permission suspended until idle', 'method': 'idle_suspension'}

        if self._turn_suspended:
            self._log_decision(tool_name, args, "allow", "Permission suspended for turn")
            return True, {'reason': 'Permission suspended for turn', 'method': 'turn_suspension'}

        # Check if user pre-approved all requests
        if self._allow_all:
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

        # Evaluate against policy
        match = self._policy.check(tool_name, args)

        if match.decision == PermissionDecision.ALLOW:
            self._log_decision(tool_name, args, "allow", match.reason)
            method = match.rule_type or 'policy'
            # Emit resolved hook for auto-approved (whitelist)
            # SKIP in subagent mode
            if self._on_permission_resolved and not is_subagent_mode:
                self._on_permission_resolved(tool_name, "", True, method)
            return True, {'reason': match.reason, 'method': method}

        elif match.decision == PermissionDecision.DENY:
            self._log_decision(tool_name, args, "deny", match.reason)
            method = match.rule_type or 'policy'
            # Emit resolved hook for auto-denied (blacklist)
            # SKIP in subagent mode
            if self._on_permission_resolved and not is_subagent_mode:
                self._on_permission_resolved(tool_name, "", False, method)
            return False, {'reason': match.reason, 'method': method}

        elif match.decision == PermissionDecision.ASK_CHANNEL:
            # Need to ask the channel (already retrieved above for subagent check)
            if not channel:
                self._log_decision(tool_name, args, "deny", "No channel configured")
                return False, {'reason': 'No channel configured for approval', 'method': 'no_channel'}

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

                # Get custom display info from source plugin if available
                channel_type = channel.name if channel else "console"
                display_info = self._get_display_info(tool_name, args, channel_type)

                # Build context with display info
                request_context = dict(context) if context else {}
                if display_info:
                    request_context["display_info"] = display_info

                request = PermissionRequest.create(
                    tool_name=tool_name,
                    arguments=args,
                    timeout=self._config.channel_timeout if self._config else 30,
                    context=request_context,
                )

                # Emit permission requested hook with raw args (client formats display)
                # SKIP in subagent mode
                if self._on_permission_requested and not is_subagent_mode:
                    self._on_permission_requested(
                        tool_name, request.request_id, args, request.response_options, call_id
                    )

                response = channel.request_permission(request)
                allowed, info = self._handle_channel_response(tool_name, args, response)

                # Emit permission resolved hook
                # SKIP in subagent mode
                if self._on_permission_resolved and not is_subagent_mode:
                    self._on_permission_resolved(
                        tool_name, request.request_id, allowed, info.get('method', 'unknown')
                    )

                return allowed, info

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
            Tuple of (is_allowed, metadata_dict) with 'reason', 'method', and optional 'comment'.
        """
        decision = response.decision
        comment = response.comment  # User's additional instructions

        # Helper to build info dict with optional comment
        def make_info(reason: str, method: str) -> Dict[str, Any]:
            info = {'reason': reason, 'method': method}
            if comment:
                info['comment'] = comment
            return info

        if decision in (ChannelDecision.ALLOW, ChannelDecision.ALLOW_ONCE):
            self._log_decision(tool_name, args, "allow", response.reason)
            return True, make_info(response.reason, 'user_approved')

        elif decision == ChannelDecision.ALLOW_SESSION:
            # Add to session whitelist
            pattern = response.remember_pattern or tool_name
            if self._policy:
                self._policy.add_session_whitelist(pattern)
            self._log_decision(tool_name, args, "allow", f"Session whitelist: {pattern}")
            return True, make_info(response.reason, 'session_whitelist')

        elif decision == ChannelDecision.ALLOW_ALL:
            # Pre-approve all future requests in this session
            self._allow_all = True
            self._log_decision(tool_name, args, "allow", "Pre-approved all requests")
            return True, make_info(response.reason, 'allow_all')

        elif decision == ChannelDecision.ALLOW_TURN:
            # Suspend prompts for remainder of this turn
            self._turn_suspended = True
            self._log_decision(tool_name, args, "allow", "Permission suspended for turn")
            return True, make_info(response.reason, 'turn_suspension')

        elif decision == ChannelDecision.ALLOW_UNTIL_IDLE:
            # Suspend prompts until session goes idle
            self._idle_suspended = True
            self._log_decision(tool_name, args, "allow", "Permission suspended until idle")
            return True, make_info(response.reason, 'idle_suspension')

        elif decision == ChannelDecision.DENY:
            self._log_decision(tool_name, args, "deny", response.reason)
            return False, make_info(response.reason, 'user_denied')

        elif decision == ChannelDecision.DENY_SESSION:
            # Add to session blacklist
            pattern = response.remember_pattern or tool_name
            if self._policy:
                self._policy.add_session_blacklist(pattern)
            self._log_decision(tool_name, args, "deny", f"Session blacklist: {pattern}")
            return False, make_info(response.reason, 'session_blacklist')

        elif decision == ChannelDecision.TIMEOUT:
            self._log_decision(tool_name, args, "deny", "Channel timeout")
            return False, make_info(response.reason, 'timeout')

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

        lines = self._build_prompt_lines(tool_name, args, display_info, include_details=include_details, include_options=True)
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

            result = executor(args)
            # Inject permission metadata if result is a dict
            if isinstance(result, dict):
                result['_permission'] = perm_info
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
