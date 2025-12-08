"""Permission plugin for controlling tool execution access.

This plugin intercepts tool execution requests and enforces access policies
through blacklist/whitelist rules and interactive actor approval.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..model_provider.types import ToolSchema

from .policy import PermissionPolicy, PermissionDecision, PolicyMatch
from .config_loader import load_config, PermissionConfig
from .actors import (
    Actor,
    ActorDecision,
    ActorResponse,
    PermissionRequest,
    ConsoleActor,
    create_actor,
)
from ..base import UserCommand, CommandCompletion, PermissionDisplayInfo, OutputCallback

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
    - Prompt an actor for approval when policy is ambiguous

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

    def __init__(self):
        self._config: Optional[PermissionConfig] = None
        self._policy: Optional[PermissionPolicy] = None
        self._actor: Optional[Actor] = None
        self._registry: Optional['PluginRegistry'] = None
        self._initialized = False
        self._wrapped_executors: Dict[str, Callable] = {}
        self._original_executors: Dict[str, Callable] = {}
        self._execution_log: List[Dict[str, Any]] = []
        self._allow_all: bool = False  # When True, auto-approve all requests

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
        # Forward to actor if it supports callbacks
        if self._actor and hasattr(self._actor, 'set_output_callback'):
            self._actor.set_output_callback(callback)

    @property
    def name(self) -> str:
        return "permission"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the permission plugin.

        Args:
            config: Optional configuration dict. If not provided, loads from
                   file specified by PERMISSION_CONFIG_PATH or default locations.

                   Config options:
                   - config_path: Path to permissions.json file
                   - actor_type: Type of actor ("console", "webhook", "file")
                   - actor_config: Configuration for the actor
                   - policy: Inline policy dict (overrides file)
        """
        # Load configuration
        config = config or {}

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

        # Initialize actor
        actor_type = config.get("actor_type") or self._config.actor_type
        actor_config = config.get("actor_config", {})

        # Set default timeout from config
        if "timeout" not in actor_config:
            actor_config["timeout"] = self._config.actor_timeout

        # For webhook, ensure endpoint is set
        if actor_type == "webhook" and "endpoint" not in actor_config:
            actor_config["endpoint"] = self._config.actor_endpoint

        try:
            self._actor = create_actor(actor_type, actor_config)
        except (ValueError, RuntimeError) as e:
            # Fall back to console actor if configured actor fails
            print(f"Warning: Failed to initialize {actor_type} actor: {e}")
            print("Falling back to console actor")
            self._actor = ConsoleActor()

        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the permission plugin."""
        if self._actor:
            self._actor.shutdown()
        self._policy = None
        self._actor = None
        self._registry = None
        self._initialized = False
        self._wrapped_executors.clear()
        self._original_executors.clear()
        self._allow_all = False

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
                description="Request permission to execute a tool. You MUST explain your intent - "
                           "what you are trying to achieve or discover with this tool execution.",
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
                }
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
        return """Tool execution is controlled by a permission system. Before executing tools,
you may use `askPermission` to check if a tool is allowed.

The askPermission tool takes:
- tool_name: Name of the tool to check
- intent: (REQUIRED) A clear explanation of what you intend to achieve or discover with this tool
- arguments: (optional) Arguments that would be passed to the tool

You MUST always provide an intent explaining WHY you need to execute the tool.
The intent should describe what you are trying to accomplish, not just repeat the command.

It returns whether the tool is allowed and the reason for the decision.
If a tool is denied, do not attempt to execute it."""

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
            CommandCompletion("allow", "Add tool/pattern to session whitelist"),
            CommandCompletion("deny", "Add tool/pattern to session blacklist"),
            CommandCompletion("default", "Set session default policy"),
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

        return []

    def _get_tool_completions(
        self, partial: str, exclude_mode: Optional[str] = None
    ) -> List[CommandCompletion]:
        """Get tool name completions matching the partial input.

        Args:
            partial: Partial tool name to match.
            exclude_mode: If "allow", exclude tools already whitelisted.
                         If "deny", exclude tools already blacklisted.
        """
        completions = []

        # Build exclusion set based on mode
        excluded: set = set()
        if self._policy and exclude_mode:
            if exclude_mode == "allow":
                # Don't show tools already allowed
                excluded = self._policy.whitelist_tools | self._policy.session_whitelist
            elif exclude_mode == "deny":
                # Don't show tools already denied
                excluded = self._policy.blacklist_tools | self._policy.session_blacklist

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
            allow <pattern>   - Add tool/pattern to session whitelist
            deny <pattern>    - Add tool/pattern to session blacklist
            default <policy>  - Set session default policy (allow|deny|ask)
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
        elif subcommand == "clear":
            return self._permissions_clear()
        else:
            return (
                f"Unknown subcommand: {subcommand}\n"
                "Usage: permissions <show|allow|deny|default|clear>\n"
                "  show              - Display current effective policy\n"
                "  allow <pattern>   - Add to session whitelist\n"
                "  deny <pattern>    - Add to session blacklist\n"
                "  default <policy>  - Set session default (allow|deny|ask)\n"
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

    def _execute_ask_permission(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the askPermission tool.

        This allows the model to proactively check if a tool is allowed
        before attempting to execute it. If approved, the tool is added to
        the session whitelist so the actual execution won't prompt again.
        """
        tool_name = args.get("tool_name", "")
        tool_args = args.get("arguments", {})
        intent = args.get("intent", "")

        if not tool_name:
            return {"error": "tool_name is required"}

        if not intent:
            return {"error": "intent is required - explain what you intend to achieve with this tool"}

        # Pass intent in context for actor to display
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
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if a tool execution is permitted.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool
            context: Optional context for actor (session_id, turn_number, etc.)

        Returns:
            Tuple of (is_allowed, metadata_dict) where metadata_dict contains:
            - 'reason': Human-readable reason string
            - 'method': Decision method ('whitelist', 'blacklist', 'default',
                       'sanitization', 'session_whitelist', 'session_blacklist',
                       'user_approved', 'user_denied', 'allow_all', 'timeout')
        """
        # Check if user pre-approved all requests
        if self._allow_all:
            self._log_decision(tool_name, args, "allow", "Pre-approved all requests")
            return True, {'reason': 'Pre-approved all requests', 'method': 'allow_all'}

        if not self._policy:
            return True, {'reason': 'Permission plugin not initialized', 'method': 'not_initialized'}

        # Evaluate against policy
        match = self._policy.check(tool_name, args)

        if match.decision == PermissionDecision.ALLOW:
            self._log_decision(tool_name, args, "allow", match.reason)
            return True, {'reason': match.reason, 'method': match.rule_type or 'policy'}

        elif match.decision == PermissionDecision.DENY:
            self._log_decision(tool_name, args, "deny", match.reason)
            return False, {'reason': match.reason, 'method': match.rule_type or 'policy'}

        elif match.decision == PermissionDecision.ASK_ACTOR:
            # Need to ask the actor
            if not self._actor:
                self._log_decision(tool_name, args, "deny", "No actor configured")
                return False, {'reason': 'No actor configured for approval', 'method': 'no_actor'}

            # Get custom display info from source plugin if available
            actor_type = self._actor.name if self._actor else "console"
            display_info = self._get_display_info(tool_name, args, actor_type)

            # Build context with display info
            request_context = dict(context) if context else {}
            if display_info:
                request_context["display_info"] = display_info

            request = PermissionRequest.create(
                tool_name=tool_name,
                arguments=args,
                timeout=self._config.actor_timeout if self._config else 30,
                context=request_context,
            )

            response = self._actor.request_permission(request)
            return self._handle_actor_response(tool_name, args, response)

        # Unknown decision type, deny by default
        return False, {'reason': 'Unknown policy decision', 'method': 'unknown'}

    def _handle_actor_response(
        self,
        tool_name: str,
        args: Dict[str, Any],
        response: ActorResponse
    ) -> Tuple[bool, Dict[str, Any]]:
        """Handle response from an actor.

        Updates session rules if actor requests it.

        Returns:
            Tuple of (is_allowed, metadata_dict) with 'reason' and 'method'.
        """
        decision = response.decision

        if decision in (ActorDecision.ALLOW, ActorDecision.ALLOW_ONCE):
            self._log_decision(tool_name, args, "allow", response.reason)
            return True, {'reason': response.reason, 'method': 'user_approved'}

        elif decision == ActorDecision.ALLOW_SESSION:
            # Add to session whitelist
            pattern = response.remember_pattern or tool_name
            if self._policy:
                self._policy.add_session_whitelist(pattern)
            self._log_decision(tool_name, args, "allow", f"Session whitelist: {pattern}")
            return True, {'reason': response.reason, 'method': 'session_whitelist'}

        elif decision == ActorDecision.ALLOW_ALL:
            # Pre-approve all future requests in this session
            self._allow_all = True
            self._log_decision(tool_name, args, "allow", "Pre-approved all requests")
            return True, {'reason': response.reason, 'method': 'allow_all'}

        elif decision == ActorDecision.DENY:
            self._log_decision(tool_name, args, "deny", response.reason)
            return False, {'reason': response.reason, 'method': 'user_denied'}

        elif decision == ActorDecision.DENY_SESSION:
            # Add to session blacklist
            pattern = response.remember_pattern or tool_name
            if self._policy:
                self._policy.add_session_blacklist(pattern)
            self._log_decision(tool_name, args, "deny", f"Session blacklist: {pattern}")
            return False, {'reason': response.reason, 'method': 'session_blacklist'}

        elif decision == ActorDecision.TIMEOUT:
            self._log_decision(tool_name, args, "deny", "Actor timeout")
            return False, {'reason': response.reason, 'method': 'timeout'}

        # Unknown decision, deny
        self._log_decision(tool_name, args, "deny", "Unknown actor decision")
        return False, {'reason': 'Unknown actor decision', 'method': 'unknown'}

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
        actor_type: str
    ) -> Optional[PermissionDisplayInfo]:
        """Get display info for a tool from its source plugin.

        Looks up the plugin that provides the tool and calls its
        format_permission_request() method if available.

        Args:
            tool_name: Name of the tool
            args: Arguments passed to the tool
            actor_type: Type of actor requesting display info

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
                return plugin.format_permission_request(tool_name, args, actor_type)
            except Exception:
                # If formatting fails, fall back to default
                return None

        return None

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


def create_plugin() -> PermissionPlugin:
    """Factory function to create the permission plugin instance."""
    return PermissionPlugin()
