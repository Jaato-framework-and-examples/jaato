"""Permission plugin for controlling tool execution access.

This plugin intercepts tool execution requests and enforces access policies
through blacklist/whitelist rules and interactive actor approval.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from google.genai import types

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
        self._initialized = False
        self._wrapped_executors: Dict[str, Callable] = {}
        self._original_executors: Dict[str, Callable] = {}
        self._execution_log: List[Dict[str, Any]] = []

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
        self._initialized = False
        self._wrapped_executors.clear()
        self._original_executors.clear()

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
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
            types.FunctionDeclaration(
                name="askPermission",
                description="Check if a tool is allowed to be executed. Returns permission status.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool to check permission for"
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments that would be passed to the tool"
                        }
                    },
                    "required": ["tool_name"]
                }
            )
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor for askPermission tool.

        Exposure is controlled via the registry (expose_tool/unexpose_tool).
        """
        return {
            "askPermission": self._execute_ask_permission
        }

    def _execute_ask_permission(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the askPermission tool.

        This allows the model to proactively check if a tool is allowed
        before attempting to execute it.
        """
        tool_name = args.get("tool_name", "")
        tool_args = args.get("arguments", {})

        if not tool_name:
            return {"error": "tool_name is required"}

        allowed, reason = self.check_permission(tool_name, tool_args)

        return {
            "allowed": allowed,
            "reason": reason,
            "tool_name": tool_name,
        }

    def check_permission(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Check if a tool execution is permitted.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool
            context: Optional context for actor (session_id, turn_number, etc.)

        Returns:
            Tuple of (is_allowed, reason_string)
        """
        if not self._policy:
            return True, "Permission plugin not initialized"

        # Evaluate against policy
        match = self._policy.check(tool_name, args)

        if match.decision == PermissionDecision.ALLOW:
            self._log_decision(tool_name, args, "allow", match.reason)
            return True, match.reason

        elif match.decision == PermissionDecision.DENY:
            self._log_decision(tool_name, args, "deny", match.reason)
            return False, match.reason

        elif match.decision == PermissionDecision.ASK_ACTOR:
            # Need to ask the actor
            if not self._actor:
                self._log_decision(tool_name, args, "deny", "No actor configured")
                return False, "No actor configured for approval"

            request = PermissionRequest.create(
                tool_name=tool_name,
                arguments=args,
                timeout=self._config.actor_timeout if self._config else 30,
                context=context,
            )

            response = self._actor.request_permission(request)
            return self._handle_actor_response(tool_name, args, response)

        # Unknown decision type, deny by default
        return False, "Unknown policy decision"

    def _handle_actor_response(
        self,
        tool_name: str,
        args: Dict[str, Any],
        response: ActorResponse
    ) -> Tuple[bool, str]:
        """Handle response from an actor.

        Updates session rules if actor requests it.
        """
        decision = response.decision

        if decision in (ActorDecision.ALLOW, ActorDecision.ALLOW_ONCE):
            self._log_decision(tool_name, args, "allow", response.reason)
            return True, response.reason

        elif decision == ActorDecision.ALLOW_SESSION:
            # Add to session whitelist
            pattern = response.remember_pattern or tool_name
            if self._policy:
                self._policy.add_session_whitelist(pattern)
            self._log_decision(tool_name, args, "allow", f"Session whitelist: {pattern}")
            return True, response.reason

        elif decision == ActorDecision.DENY:
            self._log_decision(tool_name, args, "deny", response.reason)
            return False, response.reason

        elif decision == ActorDecision.DENY_SESSION:
            # Add to session blacklist
            pattern = response.remember_pattern or tool_name
            if self._policy:
                self._policy.add_session_blacklist(pattern)
            self._log_decision(tool_name, args, "deny", f"Session blacklist: {pattern}")
            return False, response.reason

        elif decision == ActorDecision.TIMEOUT:
            self._log_decision(tool_name, args, "deny", "Actor timeout")
            return False, response.reason

        # Unknown decision, deny
        self._log_decision(tool_name, args, "deny", "Unknown actor decision")
        return False, "Unknown actor decision"

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
            allowed, reason = self.check_permission(name, args)

            if not allowed:
                return {"error": f"Permission denied: {reason}"}

            return executor(args)

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
