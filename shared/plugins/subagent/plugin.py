"""Subagent plugin for delegating tasks to specialized subagents.

This plugin allows the parent model to spawn subagents with their own
tool configurations, enabling task delegation and specialization.

The plugin uses the shared JaatoRuntime to create lightweight sessions
for subagents, avoiding redundant provider connections.
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

from .config import (
    SubagentConfig, SubagentProfile, SubagentResult, GCProfileConfig,
    discover_profiles, expand_plugin_configs, _find_workspace_root,
    gc_profile_to_plugin_config
)
from ..base import UserCommand, CommandCompletion
from ..model_provider.types import ToolSchema
from ..gc import load_gc_plugin, GCConfig

if TYPE_CHECKING:
    from ...jaato_runtime import JaatoRuntime
    from .ui_hooks import AgentUIHooks
    from ...retry_utils import RetryCallback

logger = logging.getLogger(__name__)


def _get_env_connection() -> Dict[str, str]:
    """Get connection settings from environment variables.

    Returns:
        Dict with project, location, and model from environment.
    """
    return {
        'project': os.environ.get('PROJECT_ID', ''),
        'location': os.environ.get('LOCATION', ''),
        'model': os.environ.get('MODEL_NAME', 'gemini-2.5-flash'),
    }


class SubagentPlugin:
    """Plugin for spawning subagents with specialized tool configurations.

    The subagent plugin enables the parent model to delegate tasks to
    subagents that have their own:
    - Tool configurations (different plugins enabled)
    - System instructions
    - Model selection (optionally different from parent)

    This is useful for:
    - Specialized tasks requiring different tool sets
    - Isolating tool access for security
    - Running parallel subtasks with different capabilities

    Configuration example:
        {
            "project": "my-project",
            "location": "us-central1",
            "default_model": "gemini-2.5-flash",
            "profiles": {
                "code_assistant": {
                    "description": "Subagent for code analysis and generation",
                    "plugins": ["cli"],
                    "system_instructions": "You are a code analysis assistant.",
                    "max_turns": 5
                },
                "research_agent": {
                    "description": "Subagent for MCP-based research",
                    "plugins": ["mcp"],
                    "plugin_configs": {
                        "mcp": {"config_path": ".mcp-research.json"}
                    },
                    "max_turns": 10
                }
            },
            "allow_inline": true,
            "inline_allowed_plugins": ["cli", "todo"]
        }
    """

    def __init__(self):
        """Initialize the subagent plugin."""
        self._config: Optional[SubagentConfig] = None
        self._initialized: bool = False
        self._parent_plugins: List[str] = []
        # Lazy import to avoid circular dependencies
        self._registry_class = None
        self._client_class = None
        self._permission_plugin = None  # Optional permission plugin for subagents
        # Runtime reference for efficient session creation
        self._runtime: Optional['JaatoRuntime'] = None
        # UI hooks for agent lifecycle integration
        self._ui_hooks: Optional['AgentUIHooks'] = None
        self._subagent_counter: int = 0  # Counter for generating unique subagent IDs
        self._parent_agent_id: str = "main"  # Parent agent ID for nested subagents
        # Session registry for multi-turn conversations and bidirectional communication
        self._active_sessions: Dict[str, Dict[str, Any]] = {}  # agent_id -> session info
        self._sessions_lock = threading.Lock()  # Protect session registry access
        # Parent session reference for output forwarding and cancellation propagation
        self._parent_session: Optional[Any] = None  # JaatoSession reference
        # Thread pool for async subagent execution
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="subagent")
        # Shared state for inter-agent communication
        self._shared_state: Dict[str, Any] = {}  # key -> value (thread-safe via lock)
        self._state_lock = threading.Lock()  # Protect shared state access
        # Retry callback for subagent sessions (propagated from parent)
        self._retry_callback: Optional['RetryCallback'] = None
        # Plan reporter for subagent TodoPlugins (propagated from parent)
        self._plan_reporter: Optional[Any] = None  # TodoReporter instance

    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        return "subagent"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Configuration dict containing:
                - project: GCP project ID
                - location: Vertex AI region
                - default_model: Default model for subagents
                - profiles: Dict of named subagent profiles
                - allow_inline: Whether to allow inline subagent creation
                - inline_allowed_plugins: Plugins allowed for inline creation
                - auto_discover_profiles: Whether to auto-discover profiles from
                  profiles_dir (default: True)
                - profiles_dir: Directory to scan for profile files
                  (default: .jaato/profiles)

        If project/location are not provided in config, the plugin will
        attempt to read them from environment variables (PROJECT_ID, LOCATION,
        MODEL_NAME). The connection can also be set later via set_connection().

        Profile auto-discovery scans profiles_dir for .json and .yaml/.yml files,
        each containing a single profile definition. Discovered profiles are
        merged with explicitly configured profiles, with explicit profiles
        taking precedence on name conflicts.
        """
        if config:
            self._config = SubagentConfig.from_dict(config)
        else:
            # Minimal config - will try env vars as fallback
            self._config = SubagentConfig(project='', location='')

        # Try to fill in missing connection info from environment variables
        if not self._config.project or not self._config.location:
            env_conn = _get_env_connection()
            if not self._config.project and env_conn['project']:
                self._config.project = env_conn['project']
                logger.debug("Using PROJECT_ID from environment: %s", env_conn['project'])
            if not self._config.location and env_conn['location']:
                self._config.location = env_conn['location']
                logger.debug("Using LOCATION from environment: %s", env_conn['location'])
            if self._config.default_model == 'gemini-2.5-flash' and env_conn['model']:
                self._config.default_model = env_conn['model']
                logger.debug("Using MODEL_NAME from environment: %s", env_conn['model'])

        # Auto-discover profiles from profiles_dir if enabled
        if self._config.auto_discover_profiles:
            discovered = discover_profiles(self._config.profiles_dir)
            # Merge discovered profiles, with explicit profiles taking precedence
            for name, profile in discovered.items():
                if name not in self._config.profiles:
                    self._config.profiles[name] = profile
                else:
                    logger.debug(
                        "Skipping discovered profile '%s' - explicit profile exists",
                        name
                    )

        # Lazy import the classes we need
        from ..registry import PluginRegistry
        from ...jaato_client import JaatoClient
        self._registry_class = PluginRegistry
        self._client_class = JaatoClient

        self._initialized = True
        logger.info(
            "Subagent plugin initialized with %d profiles (connection: %s)",
            len(self._config.profiles) if self._config else 0,
            "configured" if (self._config.project and self._config.location) else "pending"
        )

    def shutdown(self) -> None:
        """Clean up plugin resources."""
        self._config = None
        self._initialized = False
        logger.info("Subagent plugin shutdown")

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return function declarations for subagent tools."""
        declarations = [
            ToolSchema(
                name='spawn_subagent',
                description=(
                    'Spawn a subagent to handle a specialized task. Returns immediately with '
                    'the agent_id while the subagent runs asynchronously in the background.\n\n'
                    'ASYNC BEHAVIOR: The subagent executes independently. Its output appears in '
                    'its own tab in the UI. Use list_active_subagents to check status (running/idle). '
                    'If you need results from a subagent before proceeding, check its status and '
                    'wait until it shows "idle" before spawning dependent tasks.\n\n'
                    'IMPORTANT: Always provide EITHER a profile name (for preconfigured agents) '
                    'OR a descriptive name (for inline agents). This helps identify agents in the UI.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Descriptive name for the subagent (e.g., 'bug_fixer', 'code_reviewer', "
                                "'file_analyzer'). Use this when creating inline agents without a profile. "
                                "If using a profile, this parameter is optional and the profile name will be used."
                            )
                        },
                        "profile": {
                            "type": "string",
                            "description": (
                                "Name of a preconfigured subagent profile. "
                                "Use list_subagent_profiles to see available profiles."
                            )
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "The task or prompt to send to the subagent. Be specific "
                                "about what you want the subagent to accomplish."
                            )
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Optional additional context to provide to the subagent. "
                                "Include relevant information from the current conversation."
                            )
                        },
                        "inline_config": {
                            "type": "object",
                            "description": (
                                "Optional overrides for subagent configuration. By default, "
                                "subagents inherit your current plugins. Only specify properties "
                                "you want to override."
                            ),
                            "properties": {
                                "plugins": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Override inherited plugins. If not specified, inherits "
                                        "parent's plugins. Use plugin names (e.g., 'cli'), NOT "
                                        "tool names (e.g., 'cli_based_tool')."
                                    )
                                },
                                "system_instructions": {
                                    "type": "string",
                                    "description": "Additional system instructions for the subagent"
                                },
                                "max_turns": {
                                    "type": "integer",
                                    "description": "Maximum conversation turns (default: 10)"
                                },
                                "gc": {
                                    "type": "object",
                                    "description": (
                                        "Garbage collection configuration for the subagent. "
                                        "Allows setting a different GC threshold than the parent."
                                    ),
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["truncate", "summarize", "hybrid"],
                                            "description": "GC strategy type (default: truncate)"
                                        },
                                        "threshold_percent": {
                                            "type": "number",
                                            "description": "Trigger GC when context usage exceeds this percentage"
                                        },
                                        "preserve_recent_turns": {
                                            "type": "integer",
                                            "description": "Number of recent turns to always preserve"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "required": ["task"]
                }
            ),
            ToolSchema(
                name='send_to_subagent',
                description=(
                    'Send a message to a running subagent. The message is injected into the '
                    'subagent\'s queue and will be processed at its next yield point. This '
                    'enables real-time guidance and course correction of running subagents. '
                    'Use this to provide additional instructions, ask questions, or redirect '
                    'the subagent while it is still executing.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "subagent_id": {
                            "type": "string",
                            "description": (
                                "ID of the active subagent session (returned by spawn_subagent). "
                                "Use list_active_subagents to see available sessions."
                            )
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "Message to inject into the subagent's queue. Will be processed "
                                "at the next yield point (after tool execution or model response)."
                            )
                        }
                    },
                    "required": ["subagent_id", "message"]
                }
            ),
            ToolSchema(
                name='close_subagent',
                description=(
                    'Close an active subagent session when the task is complete. '
                    'IMPORTANT: Use this IMMEDIATELY after a subagent reports task completion '
                    'to free resources and prevent wasting turns. While sessions auto-close '
                    'after max_turns, explicit closure is the preferred pattern to ensure '
                    'efficient resource usage.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "subagent_id": {
                            "type": "string",
                            "description": "ID of the subagent session to close"
                        }
                    },
                    "required": ["subagent_id"]
                }
            ),
            ToolSchema(
                name='cancel_subagent',
                description=(
                    'Cancel a running subagent, stopping its current operation immediately. '
                    'Use this when you need to interrupt a subagent that is taking too long '
                    'or when you no longer need its result. The subagent will stop at the '
                    'next cancellation checkpoint and return partial results if available. '
                    'After cancellation, the session remains active for follow-up messages.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "subagent_id": {
                            "type": "string",
                            "description": "ID of the subagent to cancel (use list_active_subagents to see IDs)"
                        }
                    },
                    "required": ["subagent_id"]
                }
            ),
            ToolSchema(
                name='list_active_subagents',
                description=(
                    'List currently active subagent sessions that can receive follow-up '
                    'messages. Shows subagent ID, profile, status (running/waiting), and turn count.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            ToolSchema(
                name='list_subagent_profiles',
                description=(
                    'List available subagent profiles. Use this to see what '
                    'specialized subagents are configured and their capabilities.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            ToolSchema(
                name='set_shared_state',
                description=(
                    'Store a value in shared state that can be accessed by all agents '
                    '(main and subagents). Use this for inter-agent communication '
                    'and sharing analysis results between parallel subagents.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Unique key for the state entry"
                        },
                        "value": {
                            "description": "Value to store (any JSON-serializable type)"
                        }
                    },
                    "required": ["key", "value"]
                }
            ),
            ToolSchema(
                name='get_shared_state',
                description=(
                    'Retrieve a value from shared state. Use this to access '
                    'data stored by other agents or previous operations.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Key of the state entry to retrieve"
                        }
                    },
                    "required": ["key"]
                }
            ),
            ToolSchema(
                name='list_shared_state',
                description=(
                    'List all keys in shared state. Use this to see what '
                    'data is available for inter-agent communication.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
        return declarations

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return mapping of tool names to executor functions."""
        return {
            'spawn_subagent': self._execute_spawn_subagent,
            'send_to_subagent': self._execute_send_to_subagent,
            'close_subagent': self._execute_close_subagent,
            'cancel_subagent': self._execute_cancel_subagent,
            'list_active_subagents': self._execute_list_active_subagents,
            'list_subagent_profiles': self._execute_list_profiles,
            'set_shared_state': self._execute_set_shared_state,
            'get_shared_state': self._execute_get_shared_state,
            'list_shared_state': self._execute_list_shared_state,
            # User command aliases
            'profiles': self._execute_list_profiles,
            'active': self._execute_list_active_subagents,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing subagent capabilities."""
        base_instructions = (
            "You have access to a subagent system that allows you to delegate "
            "tasks to specialized subagents.\n\n"
            "ASYNC EXECUTION: spawn_subagent returns immediately with an agent_id. "
            "The subagent runs asynchronously in the background.\n\n"
            "RECEIVING SUBAGENT OUTPUT: You will receive subagent output as "
            "[SUBAGENT agent_id=X event=Y] messages injected into your conversation. "
            "Events include:\n"
            "- MODEL_OUTPUT: Text the subagent is generating\n"
            "- TOOL_CALL: Tool the subagent is calling\n"
            "- TOOL_OUTPUT: Output from subagent's tool execution\n"
            "- COMPLETED: Subagent finished its task (includes final response)\n"
            "- ERROR: Subagent encountered an error\n\n"
            "REACTING TO OUTPUT: You can monitor subagent progress in real-time and use "
            "send_to_subagent to provide guidance or corrections. When you receive a "
            "COMPLETED event, use the result to decide next steps. If you need to spawn "
            "sequential dependent subagents, wait for COMPLETED before spawning the next.\n\n"
            "BIDIRECTIONAL COMMUNICATION:\n"
            "- Use send_to_subagent to inject messages into a running subagent\n"
            "- Multiple subagents can run concurrently\n\n"
            "LIFECYCLE MANAGEMENT:\n"
            "- Use close_subagent to free resources after receiving COMPLETED\n"
            "- Sessions auto-close after max_turns, but explicit closure is preferred\n"
            "- Use list_active_subagents to see running subagents\n\n"
            "GC CONFIGURATION:\n"
            "Subagents can have their own garbage collection (GC) settings independent of the parent. "
            "This is useful for testing GC behavior or when subagents need different context management. "
            "Use inline_config.gc to specify:\n"
            "- type: 'truncate', 'summarize', or 'hybrid'\n"
            "- threshold_percent: Trigger GC at this context usage (e.g., 5.0 for early testing)\n"
            "- preserve_recent_turns: Number of recent turns to keep after GC"
        )

        if not self._config or not self._config.profiles:
            return base_instructions

        profile_descriptions = []
        for name, profile in self._config.profiles.items():
            plugins_str = ", ".join(profile.plugins) if profile.plugins else "none"
            profile_descriptions.append(
                f"- {name}: {profile.description} (tools: {plugins_str})"
            )

        profiles_text = "\n".join(profile_descriptions)

        return (
            f"{base_instructions}\n\n"
            "Available subagent profiles:\n"
            f"{profiles_text}\n\n"
            "Use spawn_subagent with a profile name and task to delegate work. "
            "Without a profile, subagents inherit your current plugin configuration."
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved."""
        # Read-only tools are safe and can be auto-approved
        # spawn_subagent and send_to_subagent should require permission unless auto_approved
        return ['list_subagent_profiles', 'list_active_subagents']

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for direct invocation.

        Provides commands that users (human or agent) can type directly
        to interact with the subagent system without model mediation.
        """
        return [
            UserCommand(
                "profiles",
                "List available subagent profiles",
                share_with_model=True  # Model should know what profiles are available
            ),
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for subagent command arguments.

        The 'profiles' command takes no arguments, so no completions needed.
        """
        return []

    def add_profile(self, profile: SubagentProfile) -> None:
        """Add a subagent profile dynamically.

        Args:
            profile: SubagentProfile to add.
        """
        if self._config:
            self._config.add_profile(profile)

    def set_runtime(self, runtime: 'JaatoRuntime') -> None:
        """Set the runtime reference for efficient session creation.

        When a runtime is set, subagents will use runtime.create_session()
        instead of creating new JaatoClient instances, sharing the provider
        connection and plugin configuration.

        Args:
            runtime: JaatoRuntime instance from the parent agent.
        """
        self._runtime = runtime

    def set_parent_session(self, session: Any) -> None:
        """Set the parent session reference for cancellation propagation.

        When set, child subagent sessions will inherit the parent's cancel
        token, allowing automatic cancellation propagation from parent to
        children.

        Args:
            session: JaatoSession instance of the parent agent.
        """
        self._parent_session = session

    def set_connection(self, project: str, location: str, model: str) -> None:
        """Set the connection parameters for subagents.

        Call this to configure the GCP connection if not provided in config.
        Note: If set_runtime() is called, this is automatically populated.

        Args:
            project: GCP project ID.
            location: Vertex AI region.
            model: Default model name.
        """
        if self._config:
            self._config.project = project
            self._config.location = location
            self._config.default_model = model

    def set_parent_plugins(self, plugins: List[str]) -> None:
        """Set the parent's exposed plugins for inheritance.

        Subagents will use these plugins by default when no explicit
        inline_config is provided.

        Args:
            plugins: List of plugin names exposed in the parent agent.
        """
        self._parent_plugins = plugins

    def set_permission_plugin(self, plugin) -> None:
        """Set the permission plugin to use for subagent tool execution.

        When set, subagents will use this permission plugin with context
        indicating they are subagents, so permission prompts clearly
        identify who is requesting permission.

        Args:
            plugin: PermissionPlugin instance from parent agent.
        """
        self._permission_plugin = plugin

    def set_ui_hooks(self, hooks: 'AgentUIHooks') -> None:
        """Set UI hooks for subagent lifecycle events.

        This enables rich terminal UIs (like rich-client) to track subagent
        creation, execution, and completion.

        Args:
            hooks: Implementation of AgentUIHooks protocol.
        """
        self._ui_hooks = hooks

    def set_retry_callback(self, callback: Optional['RetryCallback']) -> None:
        """Set retry callback for subagent sessions.

        When set, subagent sessions will use this callback for retry
        notifications instead of printing to console. This ensures retry
        messages from subagents are routed through the same channel as
        the parent (e.g., to a rich client's output panel).

        Args:
            callback: Function called on each retry attempt.
                Signature: (message: str, attempt: int, max_attempts: int, delay: float) -> None
                Set to None to revert to console output.
        """
        self._retry_callback = callback

    def set_plan_reporter(self, reporter: Optional[Any]) -> None:
        """Set plan reporter for subagent TodoPlugins.

        When set, subagent TodoPlugins will use this reporter instead of
        creating a ConsoleReporter. This ensures subagent plans are
        displayed in the same location as the parent (e.g., in a rich
        client's status bar popup instead of console).

        Args:
            reporter: TodoReporter instance to use for subagent plans.
                Set to None to let subagents create their own reporters.
        """
        self._plan_reporter = reporter

    def _execute_list_profiles(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List available subagent profiles.

        Args:
            args: Tool arguments (unused).

        Returns:
            Dict containing list of available profiles.
        """
        if not self._config or not self._config.profiles:
            return {
                'profiles': [],
                'message': (
                    'No predefined profiles. Subagents inherit your current plugins by default - '
                    'just call spawn_subagent with a task.'
                ),
            }

        profiles = []
        for name, profile in self._config.profiles.items():
            profiles.append({
                'name': name,
                'description': profile.description,
                'plugins': profile.plugins,
                'max_turns': profile.max_turns,
                'auto_approved': profile.auto_approved,
            })

        return {
            'profiles': profiles,
            'inline_allowed': self._config.allow_inline,
            'inline_allowed_plugins': self._config.inline_allowed_plugins,
        }

    def _execute_send_to_subagent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to a subagent for processing.

        If the subagent is idle, the message is processed immediately.
        If the subagent is busy, the message is queued for mid-turn processing.

        Args:
            args: Tool arguments containing:
                - subagent_id: ID of the active subagent session
                - message: Message to send to the subagent

        Returns:
            Status dict with the subagent's response or error.
        """
        subagent_id = args.get('subagent_id', '')
        message = args.get('message', '')

        if not subagent_id:
            return {
                'success': False,
                'error': 'No subagent_id provided'
            }

        if not message:
            return {
                'success': False,
                'error': 'No message provided'
            }

        # Look up active session
        with self._sessions_lock:
            session_info = self._active_sessions.get(subagent_id)

        if not session_info:
            return {
                'success': False,
                'error': f'No active session found with ID: {subagent_id}. Use list_active_subagents to see available sessions.'
            }

        try:
            session = session_info['session']
            agent_id = session_info['agent_id']

            # Check if subagent is currently processing
            if session.is_running:
                # Subagent is busy - queue for mid-turn processing
                logger.info(f"SEND_TO_SUBAGENT: {subagent_id} is busy, queuing message")
                session.inject_prompt(message)
                return {
                    'success': True,
                    'status': 'queued',
                    'message': f'Subagent is busy. Message queued for processing.'
                }

            # Subagent is idle - process directly
            logger.info(f"SEND_TO_SUBAGENT: {subagent_id} is idle, processing directly")

            # Emit the parent's message to UI
            if self._ui_hooks:
                self._ui_hooks.on_agent_output(
                    agent_id=agent_id,
                    source="parent",
                    text=message,
                    mode="write"
                )

            # Create output callback for model response
            def output_callback(source: str, text: str, mode: str) -> None:
                if self._ui_hooks:
                    self._ui_hooks.on_agent_output(
                        agent_id=agent_id,
                        source=source,
                        text=text,
                        mode=mode
                    )

            # Process the message
            response = session.send_message(message, on_output=output_callback)

            # Update context after processing (match main agent behavior)
            if self._ui_hooks:
                usage = session.get_context_usage()
                # Debug: Log full usage info to trace token accounting issues
                logger.debug(
                    f"SUBAGENT_USAGE [{agent_id}]: "
                    f"total={usage.get('total_tokens', 0)}, "
                    f"prompt={usage.get('prompt_tokens', 0)}, "
                    f"output={usage.get('output_tokens', 0)}, "
                    f"context_limit={usage.get('context_limit', 'N/A')}, "
                    f"percent_used={usage.get('percent_used', 0):.2f}%, "
                    f"turns={usage.get('turns', 0)}, "
                    f"model={usage.get('model', 'unknown')}"
                )
                self._ui_hooks.on_agent_context_updated(
                    agent_id=agent_id,
                    total_tokens=usage.get('total_tokens', 0),
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0),
                    turns=usage.get('turns', 0),
                    percent_used=usage.get('percent_used', 0)
                )

            # Forward response to parent
            if self._parent_session:
                self._parent_session.inject_prompt(
                    f"[SUBAGENT agent_id={agent_id} event=MODEL_OUTPUT]\n{response}"
                )

            return {
                'success': True,
                'status': 'processed',
                'response': response
            }

        except Exception as e:
            logger.exception(f"Error sending to subagent {subagent_id}")
            return {
                'success': False,
                'error': f'Error processing message: {str(e)}'
            }

    def _execute_close_subagent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Close an active subagent session.

        Args:
            args: Tool arguments containing:
                - subagent_id: ID of the subagent session to close

        Returns:
            Dict with success status and message.
        """
        subagent_id = args.get('subagent_id', '')

        if not subagent_id:
            return {
                'success': False,
                'message': 'No subagent_id provided'
            }

        if subagent_id not in self._active_sessions:
            return {
                'success': False,
                'message': f'No active session found with ID: {subagent_id}'
            }

        self._close_session(subagent_id)
        return {
            'success': True,
            'message': f'Session {subagent_id} closed successfully'
        }

    def _execute_cancel_subagent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a running subagent operation.

        Args:
            args: Tool arguments containing:
                - subagent_id: ID of the subagent to cancel

        Returns:
            Dict with success status and message.
        """
        subagent_id = args.get('subagent_id', '')

        if not subagent_id:
            return {
                'success': False,
                'message': 'No subagent_id provided'
            }

        session_info = self._active_sessions.get(subagent_id)
        if not session_info:
            return {
                'success': False,
                'message': f'No active session found with ID: {subagent_id}'
            }

        session = session_info.get('session')
        if not session:
            return {
                'success': False,
                'message': f'Session {subagent_id} has no valid session object'
            }

        # Check if session is currently running
        if not session.is_running:
            return {
                'success': False,
                'message': f'Session {subagent_id} is not currently running (status: waiting)'
            }

        # Check if cancellation is supported
        if not session.supports_stop:
            return {
                'success': False,
                'message': f'Session {subagent_id} does not support cancellation (provider limitation)'
            }

        # Request cancellation
        cancelled = session.request_stop()
        if cancelled:
            # Notify UI hooks
            if self._ui_hooks:
                self._ui_hooks.on_agent_status_changed(
                    agent_id=subagent_id,
                    status="cancelled"
                )
            return {
                'success': True,
                'message': f'Cancellation requested for session {subagent_id}. The subagent will stop at the next checkpoint.'
            }
        else:
            return {
                'success': False,
                'message': f'Failed to cancel session {subagent_id} - may have already completed'
            }

    def _execute_list_active_subagents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List active subagent sessions.

        Args:
            args: Tool arguments (unused).

        Returns:
            Dict containing list of active sessions.
        """
        sessions = []

        # List active sessions
        with self._sessions_lock:
            for agent_id, info in self._active_sessions.items():
                session = info.get('session')
                is_running = session.is_running if session else False
                supports_stop = session.supports_stop if session else False
                sessions.append({
                    'agent_id': agent_id,
                    'profile': info['profile'].name,
                    'status': 'running' if is_running else 'idle',
                    'can_cancel': is_running and supports_stop,
                    'can_send': True,  # Can always inject prompts
                    'created_at': info['created_at'].isoformat(),
                    'last_activity': info['last_activity'].isoformat(),
                    'turn_count': info['turn_count'],
                    'max_turns': info['max_turns'],
                })

        if not sessions:
            return {
                'active_sessions': [],
                'message': 'No active subagent sessions'
            }

        return {
            'active_sessions': sessions,
            'count': len(sessions)
        }

    def cancel_all_running(self) -> int:
        """Cancel all currently running subagent operations.

        This is useful for propagating parent cancellation to all children,
        or for cleanup when the parent session is interrupted.

        Returns:
            Number of subagents that were cancelled.
        """
        cancelled_count = 0
        for agent_id, info in self._active_sessions.items():
            session = info.get('session')
            if session and session.is_running and session.supports_stop:
                if session.request_stop():
                    cancelled_count += 1
                    if self._ui_hooks:
                        self._ui_hooks.on_agent_status_changed(
                            agent_id=agent_id,
                            status="cancelled"
                        )
        return cancelled_count

    def _execute_set_shared_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store a value in shared state.

        Args:
            args: Tool arguments containing:
                - key: State key
                - value: Value to store

        Returns:
            Dict with success status.
        """
        key = args.get('key', '')
        if not key:
            return {'success': False, 'error': 'No key provided'}

        value = args.get('value')
        with self._state_lock:
            self._shared_state[key] = value

        return {
            'success': True,
            'message': f'State "{key}" set successfully'
        }

    def _execute_get_shared_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve a value from shared state.

        Args:
            args: Tool arguments containing:
                - key: State key to retrieve

        Returns:
            Dict with value or error.
        """
        key = args.get('key', '')
        if not key:
            return {'success': False, 'error': 'No key provided'}

        with self._state_lock:
            if key in self._shared_state:
                return {
                    'success': True,
                    'key': key,
                    'value': self._shared_state[key]
                }
            else:
                return {
                    'success': False,
                    'error': f'Key "{key}" not found in shared state'
                }

    def _execute_list_shared_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all keys in shared state.

        Args:
            args: Tool arguments (unused).

        Returns:
            Dict with list of keys.
        """
        with self._state_lock:
            keys = list(self._shared_state.keys())

        if not keys:
            return {
                'keys': [],
                'message': 'Shared state is empty'
            }

        return {
            'keys': keys,
            'count': len(keys)
        }

    def _close_session(self, agent_id: str) -> None:
        """Close and cleanup a subagent session.

        Args:
            agent_id: ID of the session to close.
        """
        if agent_id not in self._active_sessions:
            return

        session_info = self._active_sessions[agent_id]

        # Notify UI hooks of completion
        if self._ui_hooks:
            self._ui_hooks.on_agent_status_changed(
                agent_id=agent_id,
                status="done"
            )
            self._ui_hooks.on_agent_completed(
                agent_id=agent_id,
                completed_at=datetime.now(),
                success=True,
                token_usage=None,
                turns_used=session_info['turn_count']
            )

        # Remove from registry
        del self._active_sessions[agent_id]
        logger.info(f"Closed subagent session: {agent_id}")

    def _execute_spawn_subagent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn a subagent to handle a task.

        Args:
            args: Tool arguments containing:
                - task: The task to perform
                - profile: Optional profile name
                - context: Optional additional context
                - inline_config: Optional inline configuration

        Returns:
            SubagentResult as a dict.
        """
        if not self._initialized:
            return SubagentResult(
                success=False,
                response='',
                error='Subagent plugin not initialized'
            ).to_dict()

        task = args.get('task', '')
        if not task:
            return SubagentResult(
                success=False,
                response='',
                error='No task provided'
            ).to_dict()

        profile_name = args.get('profile')
        context = args.get('context', '')
        inline_config = args.get('inline_config')
        custom_name = args.get('name', '')

        # Resolve the profile or create inline
        if profile_name:
            profile = self._config.get_profile(profile_name) if self._config else None
            if not profile:
                available = list(self._config.profiles.keys()) if self._config else []
                return SubagentResult(
                    success=False,
                    response='',
                    error=f"Profile '{profile_name}' not found. Available: {available}"
                ).to_dict()
        else:
            # No profile specified - use inherited plugins with optional overrides
            if not self._parent_plugins:
                return SubagentResult(
                    success=False,
                    response='',
                    error='No plugins available to inherit. Configure parent plugins first.'
                ).to_dict()

            # inline_config can override specific properties, defaults come from parent
            plugins = self._parent_plugins
            system_instructions = None
            max_turns = 10
            gc_config = None

            if inline_config:
                # Override plugins only if explicitly specified
                if 'plugins' in inline_config:
                    plugins = inline_config['plugins']
                    # Validate plugins against allowed list if configured
                    if self._config and self._config.inline_allowed_plugins:
                        disallowed = set(plugins) - set(self._config.inline_allowed_plugins)
                        if disallowed:
                            return SubagentResult(
                                success=False,
                                response='',
                                error=f"Plugins not allowed for inline creation: {disallowed}"
                            ).to_dict()
                if 'system_instructions' in inline_config:
                    system_instructions = inline_config['system_instructions']
                if 'max_turns' in inline_config:
                    max_turns = inline_config['max_turns']
                # Parse gc config from inline_config
                if 'gc' in inline_config and inline_config['gc']:
                    gc_data = inline_config['gc']
                    gc_config = GCProfileConfig(
                        type=gc_data.get('type', 'truncate'),
                        threshold_percent=gc_data.get('threshold_percent', 80.0),
                        preserve_recent_turns=gc_data.get('preserve_recent_turns', 5),
                        notify_on_gc=gc_data.get('notify_on_gc', True),
                        summarize_middle_turns=gc_data.get('summarize_middle_turns'),
                        max_turns=gc_data.get('max_turns'),
                        plugin_config=gc_data.get('plugin_config', {}),
                    )

            # Use provided name, or fall back to legacy behavior
            if custom_name:
                name = custom_name
            else:
                # Backwards compatibility: use old naming scheme
                name = '_inline' if inline_config else '_inherited'

            profile = SubagentProfile(
                name=name,
                description='Subagent with inherited plugins',
                plugins=plugins,
                system_instructions=system_instructions,
                max_turns=max_turns,
                gc=gc_config,
            )

        # Build the full prompt
        full_prompt = task
        if context:
            full_prompt = f"Context:\n{context}\n\nTask:\n{task}"

        # Add profile's system instructions
        if profile.system_instructions:
            full_prompt = f"{profile.system_instructions}\n\n{full_prompt}"

        # Generate agent_id
        self._subagent_counter += 1
        if self._parent_agent_id == "main":
            agent_id = f"subagent_{self._subagent_counter}"
        else:
            agent_id = f"{self._parent_agent_id}.{profile.name}"

        # Capture parent's working directory so subagent runs in same context
        # This ensures relative paths (trace logs, workspaceRoot) resolve correctly
        parent_cwd = os.getcwd()

        # Submit to thread pool (always async)
        self._executor.submit(
            self._run_subagent_async,
            agent_id,
            profile,
            full_prompt,
            parent_cwd
        )

        # Return immediately with agent_id
        return {
            'success': True,
            'agent_id': agent_id,
            'status': 'spawned',
            'message': f'Subagent {agent_id} spawned. You will receive [SUBAGENT agent_id={agent_id} event=...] messages as it executes. Wait for COMPLETED event before using results.'
        }

    def _run_subagent_async(
        self,
        agent_id: str,
        profile: SubagentProfile,
        prompt: str,
        parent_cwd: str
    ) -> None:
        """Run a subagent asynchronously with output forwarding to parent.

        This method runs in a thread pool and forwards all output to the
        parent session's injection queue.

        Args:
            agent_id: Pre-generated agent ID.
            profile: SubagentProfile defining the subagent's configuration.
            prompt: The prompt to send to the subagent.
            parent_cwd: Parent's working directory for resolving relative paths.
        """
        # Change to parent's working directory so relative paths resolve correctly
        # This ensures trace logs, workspaceRoot, etc. work the same as parent
        try:
            os.chdir(parent_cwd)
        except OSError as e:
            if self._parent_session:
                self._parent_session.inject_prompt(
                    f"[SUBAGENT agent_id={agent_id} event=ERROR]\n"
                    f"Cannot change to workspace directory {parent_cwd}: {e}"
                )
            return

        # Resolve trace paths to absolute so they work even if CWD changes later
        # (e.g., when parent's _in_workspace() context exits and restores CWD)
        trace_log = os.environ.get("JAATO_TRACE_LOG")
        if trace_log and not os.path.isabs(trace_log):
            os.environ["JAATO_TRACE_LOG"] = os.path.abspath(trace_log)

        provider_trace = os.environ.get("JAATO_PROVIDER_TRACE")
        if provider_trace and not os.path.isabs(provider_trace):
            os.environ["JAATO_PROVIDER_TRACE"] = os.path.abspath(provider_trace)

        if not self._runtime:
            # No runtime - can't run async subagent
            if self._parent_session:
                self._parent_session.inject_prompt(
                    f"[SUBAGENT agent_id={agent_id} event=ERROR]\n"
                    f"Cannot spawn subagent: no runtime available"
                )
            return

        try:
            # Create session using the existing runtime-based method logic
            # but with the pre-generated agent_id and parent forwarding

            # Determine model: profile > config default > parent session
            model = profile.model or self._config.default_model
            if model is None and self._parent_session:
                model = getattr(self._parent_session, '_model_name', None)

            # Determine provider: profile > config default > parent session
            provider = profile.provider or self._config.default_provider
            if provider is None and self._parent_session:
                provider = getattr(self._parent_session, '_provider_name_override', None)

            # Notify UI hooks about agent creation
            if self._ui_hooks:
                self._ui_hooks.on_agent_created(
                    agent_id=agent_id,
                    agent_name=profile.name,
                    agent_type="subagent",
                    profile_name=profile.name,
                    parent_agent_id=self._parent_agent_id,
                    icon_lines=profile.icon,
                    created_at=datetime.now()
                )
                self._ui_hooks.on_agent_status_changed(
                    agent_id=agent_id,
                    status="active"
                )

            # Expand variables in plugin_configs
            expansion_context = {}
            raw_plugin_configs = profile.plugin_configs.copy() if profile.plugin_configs else {}
            expanded_configs = expand_plugin_configs(raw_plugin_configs, expansion_context)

            # Inject agent_name and workspace-aware configs into each plugin
            effective_plugin_configs = expanded_configs
            for plugin_name in (profile.plugins or []):
                if plugin_name not in effective_plugin_configs:
                    effective_plugin_configs[plugin_name] = {}
                effective_plugin_configs[plugin_name]["agent_name"] = profile.name
                if plugin_name == "todo" and self._plan_reporter:
                    effective_plugin_configs[plugin_name]["_injected_reporter"] = self._plan_reporter
                # Inject base_path for template plugin so it uses parent's workspace
                if plugin_name == "template":
                    effective_plugin_configs[plugin_name]["base_path"] = parent_cwd

            # Save parent session reference BEFORE create_session, because
            # create_session calls session.configure() which overwrites
            # self._parent_session to the new session (see line 514 in jaato_session.py)
            parent_session = self._parent_session
            logger.debug(f"SUBAGENT_DEBUG: Saved parent_session={parent_session} (is None={parent_session is None})")

            # Create session
            session = self._runtime.create_session(
                model=model,
                tools=profile.plugins,
                system_instructions=profile.system_instructions,
                plugin_configs=effective_plugin_configs if effective_plugin_configs else None,
                provider_name=provider
            )
            logger.debug(f"SUBAGENT_DEBUG: After create_session, self._parent_session={self._parent_session}")

            # Restore parent session reference (was overwritten by configure())
            self._parent_session = parent_session
            logger.debug(f"SUBAGENT_DEBUG: Restored self._parent_session={self._parent_session}")

            # Set agent context
            session.set_agent_context(
                agent_type="subagent",
                agent_name=profile.name
            )

            # Set parent session for output forwarding
            logger.debug(f"SUBAGENT_DEBUG: Setting session._parent_session to {parent_session}")
            session.set_parent_session(parent_session)
            logger.debug(f"SUBAGENT_DEBUG: session._parent_session is now {session._parent_session}")

            # Set parent cancel token for cancellation propagation
            if self._parent_session and hasattr(self._parent_session, '_cancel_token'):
                parent_token = self._parent_session._cancel_token
                if parent_token and hasattr(session, 'set_parent_cancel_token'):
                    session.set_parent_cancel_token(parent_token)

            # Pass UI hooks to session
            if self._ui_hooks:
                session.set_ui_hooks(self._ui_hooks, agent_id)

            # Set retry callback
            if self._retry_callback:
                session.set_retry_callback(self._retry_callback)

            # Configure GC for subagent if profile specifies it
            if profile.gc:
                try:
                    gc_plugin, gc_config = gc_profile_to_plugin_config(profile.gc)
                    session.set_gc_plugin(gc_plugin, gc_config)
                    logger.info(
                        "Configured GC for subagent %s: type=%s, threshold=%.1f%%",
                        agent_id, profile.gc.type, profile.gc.threshold_percent
                    )
                except ValueError as e:
                    logger.warning(
                        "Failed to configure GC for subagent %s: %s",
                        agent_id, e
                    )

            # Store session in registry BEFORE running
            with self._sessions_lock:
                self._active_sessions[agent_id] = {
                    'session': session,
                    'profile': profile,
                    'agent_id': agent_id,
                    'created_at': datetime.now(),
                    'last_activity': datetime.now(),
                    'turn_count': 0,
                    'max_turns': profile.max_turns,
                }

            # Wrap output callback for UI hooks (forwarding to parent is automatic now)
            def subagent_output_callback(source: str, text: str, mode: str) -> None:
                if self._ui_hooks:
                    self._ui_hooks.on_agent_output(
                        agent_id=agent_id,
                        source=source,
                        text=text,
                        mode=mode
                    )

            # Emit the initial prompt to UI
            if self._ui_hooks:
                self._ui_hooks.on_agent_output(
                    agent_id=agent_id,
                    source="user",
                    text=prompt,
                    mode="write"
                )

            # Run the initial conversation (output is automatically forwarded to parent)
            response = session.send_message(prompt, on_output=subagent_output_callback)

            # Note: Additional messages from parent via send_to_subagent are now
            # processed directly by _execute_send_to_subagent when the session is idle,
            # or queued for mid-turn processing if the session is busy.
            # No polling loop needed.

            # Update session info after completion
            usage = session.get_context_usage()
            # Debug: Log full usage info to trace token accounting issues
            logger.debug(
                f"SUBAGENT_ASYNC_USAGE [{agent_id}]: "
                f"total={usage.get('total_tokens', 0)}, "
                f"prompt={usage.get('prompt_tokens', 0)}, "
                f"output={usage.get('output_tokens', 0)}, "
                f"context_limit={usage.get('context_limit', 'N/A')}, "
                f"percent_used={usage.get('percent_used', 0):.2f}%, "
                f"turns={usage.get('turns', 0)}, "
                f"model={usage.get('model', 'unknown')}"
            )
            with self._sessions_lock:
                if agent_id in self._active_sessions:
                    self._active_sessions[agent_id]['last_activity'] = datetime.now()
                    self._active_sessions[agent_id]['turn_count'] = usage.get('turns', 1)

            # Notify UI hooks with accounting data
            if self._ui_hooks:
                turn_accounting = session.get_turn_accounting()
                for turn_idx, turn in enumerate(turn_accounting):
                    self._ui_hooks.on_agent_turn_completed(
                        agent_id=agent_id,
                        turn_number=turn_idx,
                        prompt_tokens=turn.get('prompt', 0),
                        output_tokens=turn.get('output', 0),
                        total_tokens=turn.get('total', 0),
                        duration_seconds=turn.get('duration_seconds', 0),
                        function_calls=turn.get('function_calls', [])
                    )

                self._ui_hooks.on_agent_context_updated(
                    agent_id=agent_id,
                    total_tokens=usage.get('total_tokens', 0),
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0),
                    turns=usage.get('turns', 0),
                    percent_used=usage.get('percent_used', 0)
                )

                history = session.get_history()
                self._ui_hooks.on_agent_history_updated(
                    agent_id=agent_id,
                    history=history
                )

                # Change status to "idle" - ready for more prompts via send_to_subagent
                self._ui_hooks.on_agent_status_changed(
                    agent_id=agent_id,
                    status="idle"
                )

        except Exception as e:
            logger.exception(f"Error in async subagent {agent_id}")
            # Forward error to parent
            if self._parent_session:
                self._parent_session.inject_prompt(
                    f"[SUBAGENT agent_id={agent_id} event=ERROR]\n"
                    f"Subagent execution failed: {str(e)}"
                )
            # Clean up session on error
            with self._sessions_lock:
                if agent_id in self._active_sessions:
                    del self._active_sessions[agent_id]

            if self._ui_hooks:
                self._ui_hooks.on_agent_status_changed(
                    agent_id=agent_id,
                    status="error"
                )


def create_plugin() -> SubagentPlugin:
    """Factory function to create the subagent plugin.

    Returns:
        SubagentPlugin instance.
    """
    return SubagentPlugin()
