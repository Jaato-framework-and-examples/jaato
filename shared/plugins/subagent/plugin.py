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
    gc_profile_to_plugin_config, validate_profile
)
from ..base import UserCommand, CommandCompletion, CommandParameter, HelpLines
from ..model_provider.types import ToolSchema
from ..gc import load_gc_plugin, GCConfig
from ...message_queue import SourceType

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
        # Retry callback for subagent sessions (propagated from parent)
        self._retry_callback: Optional['RetryCallback'] = None
        # Plan reporter for subagent TodoPlugins (propagated from parent)
        self._plan_reporter: Optional[Any] = None  # TodoReporter instance
        # Workspace path (set by registry broadcast in server mode)
        self._workspace_path: Optional[str] = None

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

    # =========================================================================
    # Persistence Methods
    # =========================================================================

    def get_persistence_state(self) -> Dict[str, Any]:
        """Export subagent registry for session persistence.

        Returns a lightweight registry suitable for storing in SessionState.metadata.
        The full state for each subagent should be saved separately to per-agent files
        using get_agent_full_state().

        Returns:
            Dict with 'version' and 'agents' list, suitable for JSON serialization.
        """
        from .serializer import serialize_subagent_registry

        with self._sessions_lock:
            return serialize_subagent_registry(self._active_sessions)

    def get_agent_full_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get full serializable state for a specific subagent.

        This is used to save per-agent state files to
        .jaato/sessions/{session_id}/subagents/{agent_id}.json

        Args:
            agent_id: The subagent ID.

        Returns:
            Full serializable state dict, or None if agent not found.
        """
        from .serializer import serialize_subagent_state

        with self._sessions_lock:
            session_info = self._active_sessions.get(agent_id)
            if not session_info:
                return None
            return serialize_subagent_state(session_info)

    def restore_persistence_state(
        self,
        registry_data: Dict[str, Any],
        agent_states: Dict[str, Dict[str, Any]],
        runtime: 'JaatoRuntime'
    ) -> int:
        """Restore subagents from persisted state.

        Recreates subagent sessions using the persisted registry and per-agent
        state files. Sessions are recreated using the runtime's create_session().

        Args:
            registry_data: Registry dict from SessionState.metadata["subagents"].
            agent_states: Dict mapping agent_id to full state dict (from per-agent files).
            runtime: JaatoRuntime to use for creating sessions.

        Returns:
            Number of subagents successfully restored.
        """
        from .serializer import deserialize_subagent_registry, deserialize_subagent_state
        from .config import expand_plugin_configs

        if not registry_data:
            return 0

        restored_count = 0
        agents = deserialize_subagent_registry(registry_data)

        for agent_info in agents:
            agent_id = agent_info['agent_id']

            # Get full state from per-agent file
            full_state = agent_states.get(agent_id)
            if not full_state:
                logger.warning(
                    "Skipping restore for subagent %s: no state file found",
                    agent_id
                )
                continue

            try:
                # Deserialize full state
                session_data = deserialize_subagent_state(full_state)
                profile = session_data.get('profile')
                history = session_data.get('history', [])
                turn_accounting = session_data.get('turn_accounting', [])

                if not profile:
                    logger.warning(
                        "Skipping restore for subagent %s: no profile data",
                        agent_id
                    )
                    continue

                # Determine model and provider
                model = profile.model or (self._config.default_model if self._config else None)
                provider = profile.provider or (self._config.default_provider if self._config else None)

                # Expand plugin configs
                effective_plugin_configs = expand_plugin_configs(
                    profile.plugin_configs.copy() if profile.plugin_configs else {},
                    {}
                )
                for plugin_name in (profile.plugins or []):
                    if plugin_name not in effective_plugin_configs:
                        effective_plugin_configs[plugin_name] = {}
                    effective_plugin_configs[plugin_name]["agent_name"] = profile.name

                # Save parent session before create_session because configure() on
                # the new session will overwrite self._parent_session
                parent_session = self._parent_session

                # Create session using runtime
                session = runtime.create_session(
                    model=model,
                    tools=profile.plugins,
                    system_instructions=profile.system_instructions,
                    plugin_configs=effective_plugin_configs if effective_plugin_configs else None,
                    provider_name=provider
                )

                # Restore parent session reference (was overwritten by configure())
                self._parent_session = parent_session

                # Restore history
                if history:
                    session.reset_session(history)

                # Restore turn accounting
                if turn_accounting:
                    session._turn_accounting = list(turn_accounting)

                # Set agent context
                session.set_agent_context(
                    agent_type="subagent",
                    agent_name=profile.name
                )

                # Set parent session for output forwarding
                if parent_session:
                    session.set_parent_session(parent_session)

                # Register in active sessions
                with self._sessions_lock:
                    self._active_sessions[agent_id] = {
                        'session': session,
                        'profile': profile,
                        'agent_id': agent_id,
                        'created_at': session_data.get('created_at', datetime.now()),
                        'last_activity': session_data.get('last_activity', datetime.now()),
                        'turn_count': session_data.get('turn_count', 0),
                        'max_turns': session_data.get('max_turns', profile.max_turns),
                    }

                # Update counter to avoid ID collisions
                # Extract numeric suffix from agent_id like "subagent_5"
                if agent_id.startswith("subagent_"):
                    try:
                        num = int(agent_id.split("_")[1])
                        if num >= self._subagent_counter:
                            self._subagent_counter = num + 1
                    except (IndexError, ValueError):
                        pass

                restored_count += 1
                logger.info("Restored subagent %s (profile: %s)", agent_id, profile.name)

            except Exception as e:
                logger.error(
                    "Failed to restore subagent %s: %s",
                    agent_id, e
                )
                continue

        logger.info("Restored %d/%d subagents", restored_count, len(agents))
        return restored_count

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return function declarations for subagent tools."""
        declarations = [
            ToolSchema(
                name='spawn_subagent',
                description=(
                    'DELEGATE work to a subagent that runs IN PARALLEL / IN THE BACKGROUND. '
                    'Use this to run CONCURRENT tasks, OFFLOAD work, or have a HELPER agent '
                    'handle specialized operations while you continue with other work.\n\n'
                    'KEY CAPABILITIES:\n'
                    '- Run tasks asynchronously without blocking\n'
                    '- Execute multiple operations in parallel\n'
                    '- Delegate specialized work to configured agent profiles\n'
                    '- Coordinate complex multi-step workflows\n\n'
                    'RETURNS: agent_id immediately. Subagent runs independently in background.\n\n'
                    'EVENT-DRIVEN PATTERN (RECOMMENDED):\n'
                    '1. Spawn the subagent with the task\n'
                    '2. Finish your turn - inform the user you delegated the task\n'
                    '3. When the subagent completes, you receive a COMPLETED event\n'
                    '4. THEN process results or spawn follow-up tasks\n\n'
                    'DO NOT poll list_active_subagents in a loop - wait for completion events.\n\n'
                    'IMPORTANT: Provide EITHER a profile name (preconfigured) OR a descriptive name (inline).'
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
                            "description": (
                                "Optional context to provide to the subagent. Can be either:\n"
                                "- A string: Simple text context\n"
                                "- An object with structured context:\n"
                                "  - files: {path: content} - relevant file contents\n"
                                "  - findings: [list of facts/conclusions]\n"
                                "  - notes: free-form guidance\n\n"
                                "TOKEN ECONOMY: Be selective about what you share:\n"
                                "- Share only content RELEVANT to the subagent's specific task\n"
                                "- For large files, share only the relevant sections/functions\n"
                                "- Prefer file PATHS over full content when the subagent can read them\n"
                                "- Use 'findings' to summarize insights instead of raw data\n"
                                "- Remember: every token shared reduces the subagent's working space"
                            ),
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "files": {
                                            "type": "object",
                                            "description": "Relevant file content: {path: content}. Share only sections relevant to the task, or just paths if the subagent can read them.",
                                            "additionalProperties": {"type": "string"}
                                        },
                                        "findings": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Key findings or facts"
                                        },
                                        "notes": {
                                            "type": "string",
                                            "description": "Free-form notes or guidance"
                                        }
                                    }
                                }
                            ]
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
                },
                category="coordination",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='send_to_subagent',
                description=(
                    'Send a message to a running subagent for guidance or course correction. '
                    'Use this for:\n'
                    '- Giving instructions or redirecting focus\n'
                    '- Asking questions about progress\n'
                    '- Providing feedback on subagent output\n\n'
                    'NOTE: To share FILES or FINDINGS from your memory, use share_context instead. '
                    'send_to_subagent is for conversational messages, not structured knowledge transfer.'
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
                },
                category="coordination",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='close_subagent',
                description=(
                    'Close an active subagent session when the task is complete.\n\n'
                    'WHEN TO USE:\n'
                    '- After a subagent reports task completion (COMPLETED event)\n'
                    '- When activity_phase is "idle" and you no longer need the subagent\n\n'
                    'WHEN NOT TO USE:\n'
                    '- If activity_phase is "waiting_for_llm", "streaming", or "executing_tool" - '
                    'the subagent is still working! Use cancel_subagent if you need to stop it.\n'
                    '- If you want to send more messages to the subagent later\n\n'
                    'While sessions auto-close after max_turns, explicit closure is preferred '
                    'to free resources immediately.'
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
                },
                category="coordination",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='cancel_subagent',
                description=(
                    'Cancel a running subagent, stopping its current operation immediately.\n\n'
                    'WHEN TO USE:\n'
                    '- When you no longer need the result and want to stop wasting resources\n'
                    '- When activity_phase is "executing_tool" and a local tool appears stuck\n'
                    '- After user explicitly requests cancellation\n\n'
                    'WHEN NOT TO USE:\n'
                    '- If activity_phase is "waiting_for_llm" - this is NORMAL! LLM calls can take '
                    '60-120+ seconds for reasoning models. The cloud will always respond eventually.\n'
                    '- If activity_phase is "streaming" - the subagent is actively receiving '
                    'its response and will finish soon.\n'
                    '- If activity_phase is "idle" - nothing to cancel, use close_subagent instead.\n\n'
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
                },
                category="coordination",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='list_active_subagents',
                description=(
                    'List currently active subagent sessions with detailed status information.\n\n'
                    'WHEN TO USE:\n'
                    '- When user asks about subagent status\n'
                    '- Before sending a message via send_to_subagent\n'
                    '- If you suspect a subagent might be stuck (after several minutes)\n\n'
                    'DO NOT use this in a polling loop to wait for completion. Instead, finish '
                    'your turn and wait for the COMPLETED event from the subagent.\n\n'
                    'RESPONSE FIELDS:\n'
                    '- agent_id: Unique identifier for the subagent\n'
                    '- profile: The subagent profile name\n'
                    '- activity_phase: Current activity (see below)\n'
                    '- phase_duration_sec: How long in current phase\n'
                    '- turn_count / max_turns: Progress tracking\n\n'
                    'ACTIVITY PHASES:\n'
                    '- "idle": Waiting for input, ready to receive messages\n'
                    '- "waiting_for_llm": Request sent, awaiting cloud response (can take 60-120+ sec)\n'
                    '- "streaming": Receiving tokens from LLM\n'
                    '- "executing_tool": Running a tool\n\n'
                    'IMPORTANT: "waiting_for_llm" is NOT stuck - reasoning models can take minutes. '
                    'Only "executing_tool" can potentially hang if a local tool is unresponsive.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                category="coordination",
                discoverability="discoverable",
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
                },
                category="coordination",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='validateProfile',
                description=(
                    'Validate a subagent profile JSON file against the expected schema. '
                    'Checks required fields, type constraints, plugin/config structure, '
                    'and GC sub-configuration. Returns structured validation results.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to a profile JSON file to validate."
                        }
                    },
                    "required": ["path"]
                },
                category="coordination",
                discoverability="discoverable",
            ),
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
            'validateProfile': self._execute_validate_profile,
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
            "CRITICAL - END YOUR TURN AFTER SPAWNING: After calling spawn_subagent, you MUST "
            "end your turn immediately. Do NOT continue generating text. Do NOT write what you "
            "think the subagent response might be. Just end your turn and WAIT for real events.\n\n"
            "⚠️ ABSOLUTE PROHIBITION - NEVER FABRICATE EVENTS ⚠️\n"
            "You must NEVER, under ANY circumstances, write text that looks like:\n"
            "  - '[SUBAGENT agent_id=X event=IDLE]'\n"
            "  - '[SUBAGENT agent_id=X event=COMPLETED]'\n"
            "  - 'Subagent X is now idle'\n"
            "  - Any variation of subagent status messages\n\n"
            "These event messages are EXCLUSIVELY generated by the SYSTEM, not by you. "
            "If you write these yourself, you are HALLUCINATING a fake event that has NOT happened. "
            "The subagent may still be actively working while you falsely claim it's idle!\n\n"
            "CONSEQUENCES OF FABRICATING EVENTS:\n"
            "- You will act on false information (subagent isn't actually idle)\n"
            "- You may close a subagent that's still working\n"
            "- You will corrupt the workflow and lose work in progress\n"
            "- The REAL event will arrive later, causing confusion\n\n"
            "CORRECT BEHAVIOR: After spawning or interacting with a subagent, END YOUR TURN "
            "and WAIT. The system will deliver real events to you when they occur. "
            "Do not predict, anticipate, or generate event text yourself.\n\n"
            "SUBAGENT EVENTS: You will receive status events as "
            "[SUBAGENT agent_id=X event=Y] messages when subagents complete or need input. "
            "Events you may receive:\n"
            "- COMPLETED: Subagent finished its task (includes final response)\n"
            "- IDLE: Subagent is ready for more work or cleanup\n"
            "- ERROR: Subagent encountered an error\n"
            "- CANCELLED: Subagent was cancelled\n"
            "- CLARIFICATION_REQUESTED: Subagent needs clarification (you must respond)\n"
            "- PERMISSION_REQUESTED: Subagent needs permission approval (you must respond)\n\n"
            "Note: You do NOT receive progress events (MODEL_OUTPUT, TOOL_CALL, TOOL_OUTPUT) - "
            "those are shown directly to the user in the subagent panel.\n\n"
            "REMINDER: These events come FROM THE SYSTEM TO YOU. You NEVER generate them yourself. "
            "If you find yourself typing '[SUBAGENT' - STOP. That's hallucination.\n\n"
            "SUBAGENT TURN LIFECYCLE:\n"
            "When a subagent completes a turn SUCCESSFULLY, you receive events in this order:\n"
            "1. COMPLETED event - contains the subagent's final response for that turn\n"
            "2. IDLE event - 'Subagent X is now idle and ready for input'\n\n"
            "The IDLE event confirms the subagent is ready for:\n"
            "- More instructions via send_to_subagent, OR\n"
            "- Cleanup via close_subagent\n\n"
            "HANDLING ABNORMAL TERMINATION (ERROR or CANCELLED):\n"
            "If a subagent fails or is cancelled, you receive ERROR or CANCELLED instead of COMPLETED+IDLE.\n"
            "When this happens, you have options:\n"
            "- Send a message via send_to_subagent to help it recover or retry the failed operation\n"
            "- Close the subagent via close_subagent if recovery is not feasible\n"
            "- Spawn a new subagent to retry the task from scratch\n"
            "- Report the failure to the user if intervention is needed\n"
            "Note: IDLE is NOT sent after ERROR or CANCELLED, but the subagent may still be responsive.\n\n"
            "IMPORTANT: You do NOT need to call list_active_subagents to check if a subagent "
            "finished. Just WAIT for the COMPLETED+IDLE or ERROR/CANCELLED events - they will arrive "
            "automatically. Only use list_active_subagents if you need to check activity_phase "
            "for a subagent that seems to be taking unusually long.\n\n"
            "UNDERSTANDING ACTIVITY PHASES (from list_active_subagents):\n"
            "- 'idle': Subagent finished its turn, waiting for input. You will have received "
            "COMPLETED + IDLE events already.\n"
            "- 'waiting_for_llm': Subagent sent request to cloud, awaiting response. "
            "This is NORMAL and can take 60-120+ seconds for thinking models. NOT stuck.\n"
            "- 'streaming': Subagent is receiving tokens from the model. Definitely alive.\n"
            "- 'executing_tool': Subagent is running a tool. Only this phase can potentially "
            "hang if a local tool is unresponsive.\n\n"
            "WHEN TO USE list_active_subagents:\n"
            "- To check activity_phase when a subagent has been silent for a very long time\n"
            "- To see all active subagents and their turn counts\n"
            "- Do NOT poll repeatedly - wait for events instead\n"
            "- Do NOT assume 'waiting_for_llm' means stuck - it means working\n\n"
            "CRITICAL - RESPONDING TO SUBAGENT REQUESTS:\n"
            "When you receive CLARIFICATION_REQUESTED or PERMISSION_REQUESTED, the subagent is "
            "BLOCKED waiting for your response. You have TWO options:\n\n"
            "OPTION 1 (Preferred): Answer autonomously based on context and common sense. "
            "Make reasonable decisions yourself without involving the user.\n\n"
            "OPTION 2: If you truly need user input, use request_clarification YOURSELF to ask "
            "the user, then forward their answer to the subagent. Do NOT just ask the user in "
            "plain text - they cannot directly answer the subagent.\n\n"
            "After deciding (or getting user input), respond via send_to_subagent:\n"
            "- For clarification: send_to_subagent(subagent_id, '<clarification_response request_id=\"...\"><answer index=\"1\">your answer</answer></clarification_response>')\n"
            "- For permission: send_to_subagent(subagent_id, '<permission_response request_id=\"...\"><decision>yes</decision></permission_response>')\n"
            "- Simple responses also work: send_to_subagent(subagent_id, 'yes') or send_to_subagent(subagent_id, 'blue')\n\n"
            "IMPORTANT: If you ask the user 'What should I answer?' in plain text, they CANNOT "
            "directly respond to the subagent. You must either decide yourself OR use "
            "request_clarification to formally ask, then call send_to_subagent with their answer.\n\n"
            "REACTING TO OUTPUT: You can monitor subagent progress in real-time and use "
            "send_to_subagent to provide guidance or corrections. When you receive a "
            "COMPLETED event, use the result to decide next steps. If you need to spawn "
            "sequential dependent subagents, wait for COMPLETED before spawning the next.\n\n"
            "BIDIRECTIONAL COMMUNICATION:\n"
            "- send_to_subagent: For guidance, instructions, questions, or sharing context with subagents\n"
            "- Subagents can share back to you using their native share_context tool\n"
            "- Multiple subagents can run concurrently\n\n"
            "LIFECYCLE MANAGEMENT:\n"
            "- When you receive COMPLETED + IDLE events, the subagent is ready for more work or cleanup\n"
            "- When you receive ERROR or CANCELLED, assess whether recovery is possible before closing\n"
            "- Use close_subagent to free resources when done with a subagent\n"
            "- Sessions auto-close after max_turns, but explicit closure is preferred\n\n"
            "GC CONFIGURATION:\n"
            "Subagents can have their own garbage collection (GC) settings independent of the parent. "
            "This is useful for testing GC behavior or when subagents need different context management. "
            "Use inline_config.gc to specify:\n"
            "- type: 'truncate', 'summarize', or 'hybrid'\n"
            "- threshold_percent: Trigger GC at this context usage (e.g., 5.0 for early testing)\n"
            "- preserve_recent_turns: Number of recent turns to keep after GC\n\n"
            "CONTEXT SHARING (TOKEN-AWARE):\n"
            "When spawning subagents, BE SELECTIVE about what you share:\n"
            "- Use context parameter: {files: {path: content}, findings: [...], notes: '...'}\n"
            "- Share only content RELEVANT to the subagent's specific task\n"
            "- For large files, share only the relevant sections (functions, classes)\n"
            "- Prefer file PATHS when the subagent has file_edit tools and can read them\n"
            "- Use 'findings' to share insights/conclusions instead of raw content\n"
            "- Every token you share reduces the subagent's working space for its task\n"
            "- DON'T share everything upfront - subagents can ASK for more context if needed,\n"
            "  and you'll see their request and can respond via send_to_subagent\n\n"
            "Example spawn with selective context:\n"
            "  spawn_subagent(task='fix auth bug in login()', context={files: {'auth.py': '<ONLY login() function>'}, findings: ['Uses JWT', 'Bug is in token validation']})\n\n"
            "ACTIVE COLLABORATION WITH TODO TOOLS:\n"
            "Use TODO planning tools for structured parent-child collaboration:\n\n"
            "PARENT WORKFLOW (before spawning):\n"
            "1. Create a plan with createPlan for your overall task\n"
            "2. Call subscribeToTasks() to receive events when subagents complete steps\n"
            "3. Use addDependentStep to create steps that wait on subagent deliverables\n"
            "4. Spawn the subagent with clear instructions to use TODO tools\n\n"
            "CHILD WORKFLOW (in subagent):\n"
            "1. Create its own plan with createPlan for its subtask\n"
            "2. Execute work and report progress with updateStep\n"
            "3. If you need additional context, ASK the parent - they see your output and can\n"
            "   respond via send_to_subagent with the information you need\n"
            "4. Complete with completePlan - this triggers events to parent\n\n"
            "SYNCHRONIZATION:\n"
            "- Parent's addDependentStep creates BLOCKED steps that auto-unblock when child completes\n"
            "- Use getBlockedSteps to see what's waiting on subagents\n"
            "- Use getTaskEvents to review cross-agent activity\n\n"
            "EXAMPLE COLLABORATION:\n"
            "  # Parent:\n"
            "  createPlan(title='Main task', steps=['Prepare', 'Delegate research', 'Synthesize'])\n"
            "  subscribeToTasks()  # Receive child events\n"
            "  addDependentStep(description='Wait for research results', wait_for='research findings')\n"
            "  spawn_subagent(profile='investigator-web-research', task='Research X. Use createPlan to track progress.')\n"
            "  # ... parent continues, blocked step unblocks when child's plan completes\n\n"
            "  # Child (investigator):\n"
            "  createPlan(title='Research X', steps=['Search', 'Fetch', 'Summarize'])\n"
            "  # ... does work, updates steps ...\n"
            "  completePlan(summary='Found 5 key findings...')  # Triggers event to parent\n\n"
            "This enables observable, traceable multi-agent workflows where both agents "
            "maintain plans and coordinate through the shared TODO event system.\n\n"
            "SPAWN ECONOMY - AVOID UNNECESSARY SPAWNS:\n"
            "Every spawn_subagent call creates a new worker with its own session, context window, "
            "and resource overhead. Before spawning, ask yourself:\n\n"
            "1. Is there already an idle subagent that can handle this? → Use send_to_subagent\n"
            "2. Is this a follow-up to work a subagent already did? → Use send_to_subagent\n"
            "3. Is this a clarification or short message? → Use send_to_subagent (NEVER spawn for this)\n"
            "4. Does this require a genuinely independent worker? → Only then use spawn_subagent\n\n"
            "ANTI-PATTERNS (do NOT do these):\n"
            "- Spawning a new subagent to send a clarification response to an existing one\n"
            "- Spawning a second implementer for fixes when the first one is idle and can continue\n"
            "- Spawning per-step: one subagent for scaffold, another for tests, another for fixes\n"
            "- Spawning to ask a subagent a question (use send_to_subagent instead)\n\n"
            "CORRECT PATTERNS:\n"
            "- Spawn ONE implementer for a plan's lifecycle, send follow-up tasks via send_to_subagent\n"
            "- Spawn multiple subagents ONLY for genuinely parallel, independent work streams\n"
            "- Spawn a new subagent ONLY when the task requires a different profile or isolated toolset\n"
            "- After a subagent completes and goes IDLE, reuse it with send_to_subagent for related work\n\n"
            "DECISION CHECKLIST before calling spawn_subagent:\n"
            "- Is this task independent and parallelizable with current work? → spawn\n"
            "- Is it a small clarification, fix, or follow-up? → send_to_subagent to existing agent\n"
            "- Does it require a unique profile or different external tooling? → spawn\n"
            "- Could the same idle subagent handle this with additional instructions? → send_to_subagent"
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

    def _execute_validate_profile(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a subagent profile JSON file against the expected schema.

        Reads the file, parses it as JSON, and runs validate_profile()
        to check required fields, type constraints, plugin/config structure,
        and GC sub-configuration.

        Args:
            args: Tool arguments with 'path' (string, required).

        Returns:
            Dict with 'valid', 'path', 'errors', and 'warnings' fields.
        """
        import json
        from pathlib import Path

        file_path = args.get("path", "")
        if not file_path:
            return {"valid": False, "path": "", "errors": ["'path' is required"], "warnings": []}

        path_obj = Path(file_path)
        if not path_obj.is_absolute() and self._workspace_path:
            path_obj = Path(self._workspace_path) / path_obj

        if not path_obj.exists():
            return {"valid": False, "path": str(path_obj), "errors": [f"File not found: {path_obj}"], "warnings": []}

        try:
            content = path_obj.read_text(encoding='utf-8')
        except (IOError, OSError) as e:
            return {"valid": False, "path": str(path_obj), "errors": [f"Cannot read file: {e}"], "warnings": []}

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return {"valid": False, "path": str(path_obj), "errors": [f"Invalid JSON: {e}"], "warnings": []}

        is_valid, errors, warnings = validate_profile(data)
        return {
            "valid": is_valid,
            "path": str(path_obj),
            "errors": errors,
            "warnings": warnings,
        }

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved."""
        # Read-only tools are safe and can be auto-approved
        # spawn_subagent and send_to_subagent should require permission unless auto_approved
        return ['list_subagent_profiles', 'list_active_subagents', 'validateProfile']

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for direct invocation.

        Provides commands that users (human or agent) can type directly
        to interact with the subagent system without model mediation.
        """
        return [
            UserCommand(
                "profiles",
                "List available subagent profiles",
                share_with_model=True,  # Model should know what profiles are available
                parameters=[
                    CommandParameter(
                        name="subcommand",
                        description="Subcommand: help",
                        required=False
                    )
                ]
            ),
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for subagent command arguments."""
        if command != "profiles":
            return []

        # No args yet - suggest help
        if not args or (len(args) == 1 and "help".startswith(args[0].lower())):
            return [CommandCompletion("help", "Show detailed help for this command")]

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

    def set_workspace_path(self, path: str) -> None:
        """Set the workspace path for subagent spawning.

        This is called by the PluginRegistry when broadcasting workspace path
        to all plugins. Subagents will use this path as their working directory.

        Args:
            path: Absolute path to the workspace root directory.
        """
        self._workspace_path = path
        logger.debug("SubagentPlugin: workspace path set to %s", path)

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

    def _execute_list_profiles(self, args: Dict[str, Any]):
        """List available subagent profiles.

        Args:
            args: Tool arguments with optional 'subcommand'.

        Returns:
            Dict containing list of available profiles, or HelpLines for help.
        """
        # Handle help subcommand
        subcommand = args.get("subcommand", "").strip().lower()
        if subcommand == "help":
            return self._cmd_help()

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

    def _cmd_help(self) -> HelpLines:
        """Return detailed help text for pager display."""
        return HelpLines(lines=[
            ("Profiles Command", "bold"),
            ("", ""),
            ("List available subagent profiles. Subagents are specialized agents", ""),
            ("that can be spawned to handle specific tasks with their own tools.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    profiles [help]", ""),
            ("", ""),
            ("ARGUMENTS", "bold"),
            ("    (none)            List all available subagent profiles", "dim"),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    profiles                  List all available profiles", "dim"),
            ("    profiles help             Show this help message", "dim"),
            ("", ""),
            ("PROFILE CONFIGURATION", "bold"),
            ("    Profiles are defined in .jaato/subagents.json:", ""),
            ("", ""),
            ('    {', "dim"),
            ('      "profiles": {', "dim"),
            ('        "researcher": {', "dim"),
            ('          "description": "Research and analysis tasks",', "dim"),
            ('          "plugins": ["web_search", "web_fetch"],', "dim"),
            ('          "max_turns": 10', "dim"),
            ('        }', "dim"),
            ('      }', "dim"),
            ('    }', "dim"),
            ("", ""),
            ("MODEL TOOLS", "bold"),
            ("    The model uses these tools to work with subagents:", ""),
            ("    spawn_subagent        Create a new subagent for a task", "dim"),
            ("    send_to_subagent      Send a message to an active subagent", "dim"),
            ("    list_active_subagents List currently running subagents", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - Subagents run asynchronously in the background", "dim"),
            ("    - Without profiles, subagents inherit parent's plugins", "dim"),
            ("    - Each profile can specify plugins, max turns, auto-approval", "dim"),
        ])

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
                # Use PARENT source type for priority processing
                logger.info(f"SEND_TO_SUBAGENT: {subagent_id} is busy, queuing message")
                session.inject_prompt(
                    message,
                    source_id=self._parent_session._agent_id if self._parent_session else "main",
                    source_type=SourceType.PARENT
                )
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

            # Create usage callback for real-time context updates during streaming
            # This ensures the status bar reflects actual token usage from the provider
            def usage_callback(usage) -> None:
                if self._ui_hooks and usage.total_tokens > 0:
                    context_limit = session.get_context_limit()
                    percent_used = (usage.total_tokens / context_limit * 100) if context_limit > 0 else 0
                    turn_accounting = session.get_turn_accounting()
                    self._ui_hooks.on_agent_context_updated(
                        agent_id=agent_id,
                        total_tokens=usage.total_tokens,
                        prompt_tokens=usage.prompt_tokens,
                        output_tokens=usage.output_tokens,
                        turns=len(turn_accounting),
                        percent_used=percent_used
                    )
                    # Also emit instruction budget for real-time budget panel updates
                    if session.instruction_budget:
                        self._ui_hooks.on_agent_instruction_budget_updated(
                            agent_id=agent_id,
                            budget_snapshot=session.instruction_budget.snapshot()
                        )

            # Process the message
            response = session.send_message(
                message,
                on_output=output_callback,
                on_usage_update=usage_callback
            )

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

            # Forward response to parent (CHILD source - status update)
            if self._parent_session:
                self._parent_session.inject_prompt(
                    f"[SUBAGENT agent_id={agent_id} event=MODEL_OUTPUT]\n{response}",
                    source_id=agent_id,
                    source_type=SourceType.CHILD
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

        If the session is still running, it will be cancelled first before
        being removed from the registry.

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

        with self._sessions_lock:
            if subagent_id not in self._active_sessions:
                return {
                    'success': False,
                    'message': f'No active session found with ID: {subagent_id}'
                }

            session_info = self._active_sessions[subagent_id]
            session = session_info.get('session')

            # If session is still running, cancel it first
            was_running = False
            if session and session.is_running:
                was_running = True
                if session.supports_stop:
                    session.request_stop()

            self._close_session_unlocked(subagent_id)

            if was_running:
                return {
                    'success': True,
                    'message': f'Session {subagent_id} cancelled and closed successfully'
                }
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
            Dict containing list of active sessions with activity phase info.
        """
        sessions = []

        # List active sessions
        with self._sessions_lock:
            for agent_id, info in self._active_sessions.items():
                session = info.get('session')
                is_running = session.is_running if session else False
                supports_stop = session.supports_stop if session else False

                # Get activity phase information
                activity_phase = session.activity_phase.value if session else "idle"
                phase_duration = session.phase_duration_seconds if session else None
                phase_started = session.phase_started_at.isoformat() if session and session.phase_started_at else None

                sessions.append({
                    'subagent_id': agent_id,  # Match parameter name for close/cancel/send tools
                    'profile': info['profile'].name,
                    'status': 'running' if is_running else 'idle',
                    'activity_phase': activity_phase,
                    'phase_duration_sec': round(phase_duration, 1) if phase_duration else None,
                    'phase_started_at': phase_started,
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

    def _format_shared_context(
        self,
        files: Optional[Dict[str, str]] = None,
        findings: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> str:
        """Format shared context into a structured message.

        Args:
            files: Dict of file paths to content/summaries from memory.
            findings: List of key findings or facts.
            notes: Free-form notes or guidance.

        Returns:
            Formatted context string in XML-like structure with instructions.
        """
        parts = []

        # Add instruction prefix so the receiving agent knows to use this content
        if files:
            parts.append(
                "IMPORTANT: The following files have been shared with you from the parent agent's memory. "
                "DO NOT re-read these files - use the content provided below directly. "
                "This saves time and avoids redundant tool calls."
            )
            parts.append("")

        parts.append('<shared_context>')

        if files:
            for path, content in files.items():
                parts.append(f'  <file path="{path}">')
                parts.append(f'    {content}')
                parts.append('  </file>')

        if findings:
            parts.append('  <findings>')
            for finding in findings:
                parts.append(f'    <finding>{finding}</finding>')
            parts.append('  </findings>')

        if notes:
            parts.append('  <notes>')
            parts.append(f'    {notes}')
            parts.append('  </notes>')

        parts.append('</shared_context>')
        return '\n'.join(parts)

    def _close_session(self, agent_id: str) -> None:
        """Close and cleanup a subagent session (thread-safe).

        Args:
            agent_id: ID of the session to close.
        """
        with self._sessions_lock:
            self._close_session_unlocked(agent_id)

    def _close_session_unlocked(self, agent_id: str) -> None:
        """Close and cleanup a subagent session (caller must hold lock).

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
            # Handle both string and structured context
            if isinstance(context, str):
                context_str = context
            elif isinstance(context, dict):
                # Validate context.files shape: must be dict {path: content}, not a list
                files_val = context.get('files')
                if files_val is not None and isinstance(files_val, list):
                    return SubagentResult(
                        success=False,
                        response='',
                        error=(
                            "context.files must be a dict mapping file paths to content "
                            "(e.g., {\"src/auth.py\": \"<content>\"}), not a list. "
                            "Fix the shape and retry."
                        )
                    ).to_dict()
                # Structured context with files/findings/notes
                context_str = self._format_shared_context(
                    files=files_val,
                    findings=context.get('findings'),
                    notes=context.get('notes')
                )
            else:
                context_str = str(context)
            full_prompt = f"Context:\n{context_str}\n\nTask:\n{task}"

        # Add profile's system instructions
        if profile.system_instructions:
            full_prompt = f"{profile.system_instructions}\n\n{full_prompt}"

        # Generate agent_id
        self._subagent_counter += 1
        if self._parent_agent_id == "main":
            agent_id = f"subagent_{self._subagent_counter}"
        else:
            agent_id = f"{self._parent_agent_id}.{profile.name}"

        # Get workspace path - priority order:
        # 1. Directly set path (from server registry broadcast)
        # 2. Runtime registry (for JaatoClient mode)
        # 3. JAATO_WORKSPACE_ROOT env var (if set by server context)
        # 4. os.getcwd() as last resort
        workspace_path = self._workspace_path
        if workspace_path is None and self._runtime and self._runtime.registry:
            workspace_path = self._runtime.registry.get_workspace_path()
        if workspace_path is None:
            workspace_path = os.environ.get("JAATO_WORKSPACE_ROOT")
        parent_cwd = workspace_path or os.getcwd()
        logger.debug(
            "SubagentPlugin.spawn_subagent: workspace resolution: "
            f"self._workspace_path={self._workspace_path}, "
            f"registry={self._runtime.registry.get_workspace_path() if self._runtime and self._runtime.registry else None}, "
            f"env={os.environ.get('JAATO_WORKSPACE_ROOT')}, "
            f"cwd={os.getcwd()}, "
            f"result={parent_cwd}"
        )

        # Submit to thread pool (always async)
        self._executor.submit(
            self._run_subagent_async,
            agent_id,
            profile,
            full_prompt,
            parent_cwd
        )

        # Return immediately with subagent_id (matches parameter name for close/cancel/send tools)
        return {
            'success': True,
            'subagent_id': agent_id,
            'status': 'spawned',
            'message': f'Subagent {agent_id} spawned and running in background. END YOUR TURN NOW. Do NOT continue generating text. Do NOT write fake completion events. Real events will be sent to you automatically.'
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
        # Get workspace path from runtime registry as authoritative source
        # The parent_cwd parameter might be wrong if spawn_subagent couldn't resolve it correctly
        workspace_path = parent_cwd
        if self._runtime and self._runtime.registry:
            registry_workspace = self._runtime.registry.get_workspace_path()
            if registry_workspace:
                workspace_path = registry_workspace
                logger.debug(
                    f"SubagentPlugin._run_subagent_async: using registry workspace {registry_workspace} "
                    f"instead of parent_cwd {parent_cwd}"
                )

        # Set workspace path for thread-safe operations
        # os.chdir() is process-wide and racy, so we also set an env var that
        # various components can use deterministically:
        # - OAuth token storage (github_models, anthropic, antigravity)
        # - Tool plugins (file_edit, cli) for path sandboxing
        os.environ["JAATO_WORKSPACE_ROOT"] = workspace_path

        # Change to parent's working directory so relative paths resolve correctly
        # This ensures trace logs, workspaceRoot, etc. work the same as parent
        try:
            os.chdir(workspace_path)
        except OSError as e:
            if self._parent_session:
                self._parent_session.inject_prompt(
                    f"[SUBAGENT agent_id={agent_id} event=ERROR]\n"
                    f"Cannot change to workspace directory {workspace_path}: {e}",
                    source_id=agent_id,
                    source_type=SourceType.CHILD
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
                    f"Cannot spawn subagent: no runtime available",
                    source_id=agent_id,
                    source_type=SourceType.CHILD
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
            # Pass workspace_path as override to ensure ${workspaceRoot} expands correctly
            # (fixes predefined profiles which have plugin_configs with workspace variables)
            expansion_context = {}
            raw_plugin_configs = profile.plugin_configs.copy() if profile.plugin_configs else {}
            expanded_configs = expand_plugin_configs(raw_plugin_configs, expansion_context, workspace_path)

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

            # Configure clarification and permission plugins for subagent mode
            # This routes their requests through the parent instead of blocking locally
            if self._runtime and self._runtime.registry:
                registry = self._runtime.registry

                # Configure clarification plugin
                clarification_plugin = registry.get_plugin('clarification')
                if clarification_plugin and hasattr(clarification_plugin, 'configure_for_subagent'):
                    clarification_plugin.configure_for_subagent(session)
                    logger.debug(f"SUBAGENT_DEBUG: Configured clarification plugin for subagent mode")

                # Configure permission plugin
                if self._runtime.permission_plugin and hasattr(self._runtime.permission_plugin, 'configure_for_subagent'):
                    self._runtime.permission_plugin.configure_for_subagent(session)
                    logger.debug(f"SUBAGENT_DEBUG: Configured permission plugin for subagent mode")

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
                    # Use profile name for traces (more meaningful than agent_id)
                    gc_plugin, gc_config = gc_profile_to_plugin_config(profile.gc, profile.name)
                    session.set_gc_plugin(gc_plugin, gc_config)
                    logger.info(
                        "Configured GC for subagent %s: type=%s, threshold=%.1f%%",
                        agent_id, profile.gc.type, profile.gc.threshold_percent
                    )
                    # Notify UI about GC config for status bar display
                    if self._ui_hooks and hasattr(self._ui_hooks, 'on_agent_gc_config'):
                        strategy = profile.gc.type
                        self._ui_hooks.on_agent_gc_config(
                            agent_id,
                            profile.gc.threshold_percent,
                            strategy,
                            target_percent=profile.gc.target_percent,
                            continuous_mode=profile.gc.continuous_mode,
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

            # Create usage callback for real-time context updates during streaming
            # This ensures the status bar reflects actual token usage from the provider
            def subagent_usage_callback(usage) -> None:
                if self._ui_hooks and usage.total_tokens > 0:
                    context_limit = session.get_context_limit()
                    percent_used = (usage.total_tokens / context_limit * 100) if context_limit > 0 else 0
                    turn_accounting = session.get_turn_accounting()
                    self._ui_hooks.on_agent_context_updated(
                        agent_id=agent_id,
                        total_tokens=usage.total_tokens,
                        prompt_tokens=usage.prompt_tokens,
                        output_tokens=usage.output_tokens,
                        turns=len(turn_accounting),
                        percent_used=percent_used
                    )
                    # Also emit instruction budget for real-time budget panel updates
                    if session.instruction_budget:
                        self._ui_hooks.on_agent_instruction_budget_updated(
                            agent_id=agent_id,
                            budget_snapshot=session.instruction_budget.snapshot()
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
            response = session.send_message(
                prompt,
                on_output=subagent_output_callback,
                on_usage_update=subagent_usage_callback
            )

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
            # Forward error to parent (CHILD source - status update)
            if self._parent_session:
                self._parent_session.inject_prompt(
                    f"[SUBAGENT agent_id={agent_id} event=ERROR]\n"
                    f"Subagent execution failed: {str(e)}",
                    source_id=agent_id,
                    source_type=SourceType.CHILD
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
