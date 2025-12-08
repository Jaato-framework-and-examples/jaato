"""Subagent plugin for delegating tasks to specialized subagents.

This plugin allows the parent model to spawn subagents with their own
tool configurations, enabling task delegation and specialization.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .config import SubagentConfig, SubagentProfile, SubagentResult
from ..base import UserCommand, CommandCompletion
from ..model_provider.types import ToolSchema

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

        If project/location are not provided in config, the plugin will
        attempt to read them from environment variables (PROJECT_ID, LOCATION,
        MODEL_NAME). The connection can also be set later via set_connection().
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
                    'Spawn a subagent to handle a specialized task. The subagent '
                    'has its own tool configuration and runs independently. Use this '
                    'to delegate tasks that require different capabilities or to '
                    'isolate tool access. The subagent will complete the task and '
                    'return the result.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
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
                                }
                            }
                        }
                    },
                    "required": ["task"]
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
            )
        ]
        return declarations

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return mapping of tool names to executor functions."""
        return {
            'spawn_subagent': self._execute_spawn_subagent,
            'list_subagent_profiles': self._execute_list_profiles,
            # User command alias
            'profiles': self._execute_list_profiles,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing subagent capabilities."""
        if not self._config or not self._config.profiles:
            return (
                "You have access to a subagent system that allows you to delegate "
                "tasks to specialized subagents. By default, subagents inherit your "
                "current plugin configuration. Use inline_config only to override "
                "specific properties like max_turns or system_instructions."
            )

        profile_descriptions = []
        for name, profile in self._config.profiles.items():
            plugins_str = ", ".join(profile.plugins) if profile.plugins else "none"
            profile_descriptions.append(
                f"- {name}: {profile.description} (tools: {plugins_str})"
            )

        profiles_text = "\n".join(profile_descriptions)

        return (
            "You have access to a subagent system for delegating specialized tasks.\n\n"
            "Available subagent profiles:\n"
            f"{profiles_text}\n\n"
            "Use spawn_subagent with a profile name and task to delegate work. "
            "Without a profile, subagents inherit your current plugin configuration."
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that should be auto-approved."""
        # list_subagent_profiles is safe and can be auto-approved
        # spawn_subagent should require permission unless the profile is auto_approved
        return ['list_subagent_profiles']

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

    def set_connection(self, project: str, location: str, model: str) -> None:
        """Set the connection parameters for subagents.

        Call this to configure the GCP connection if not provided in config.

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

            profile = SubagentProfile(
                name='_inline' if inline_config else '_inherited',
                description='Subagent with inherited plugins',
                plugins=plugins,
                system_instructions=system_instructions,
                max_turns=max_turns,
            )

        # Build the full prompt
        full_prompt = task
        if context:
            full_prompt = f"Context:\n{context}\n\nTask:\n{task}"

        # Add profile's system instructions
        if profile.system_instructions:
            full_prompt = f"{profile.system_instructions}\n\n{full_prompt}"

        # Run the subagent
        try:
            result = self._run_subagent(profile, full_prompt)
            return result.to_dict()
        except Exception as e:
            logger.exception("Error running subagent")
            return SubagentResult(
                success=False,
                response='',
                error=f"Subagent execution failed: {str(e)}"
            ).to_dict()

    def _run_subagent(self, profile: SubagentProfile, prompt: str) -> SubagentResult:
        """Run a subagent with the given profile and prompt.

        Args:
            profile: SubagentProfile defining the subagent's configuration.
            prompt: The prompt to send to the subagent.

        Returns:
            SubagentResult with the subagent's response.
        """
        if not self._config or not self._registry_class or not self._client_class:
            return SubagentResult(
                success=False,
                response='',
                error='Plugin not properly initialized'
            )

        # Validate connection config
        if not self._config.project or not self._config.location:
            return SubagentResult(
                success=False,
                response='',
                error='Connection not configured (project/location required)'
            )

        # Create a fresh plugin registry for the subagent
        registry = self._registry_class()
        registry.discover()

        # Expose only the plugins specified in the profile
        failed_plugins = []
        for plugin_name in profile.plugins:
            plugin_config = profile.plugin_configs.get(plugin_name, {})
            try:
                registry.expose_tool(plugin_name, plugin_config)
            except Exception as e:
                failed_plugins.append((plugin_name, str(e)))
                logger.warning("Failed to expose plugin %s: %s", plugin_name, e)

        # If any plugins failed, return error with available plugins
        if failed_plugins:
            available = registry.list_available()
            errors = "; ".join(f"'{p}': {e}" for p, e in failed_plugins)
            return SubagentResult(
                success=False,
                response='',
                error=f"Failed to expose plugins: {errors}. Available plugins: {available}"
            )

        # Create subagent client
        client = self._client_class()

        # Use profile's model or default
        model = profile.model or self._config.default_model

        try:
            client.connect(
                self._config.project,
                self._config.location,
                model
            )

            # Configure tools with permission plugin if available
            # Pass subagent context so permission prompts identify the requester
            if self._permission_plugin:
                # Create a subagent-specific context for permission checks
                # We need to set this on the executor after configure_tools
                client.configure_tools(registry, permission_plugin=self._permission_plugin)
                # Override the permission context to identify as subagent
                if client._executor:
                    client._executor.set_permission_plugin(
                        self._permission_plugin,
                        context={
                            "agent_type": "subagent",
                            "agent_name": profile.name
                        }
                    )
            else:
                client.configure_tools(registry)

            # Run the conversation (subagent output is not streamed)
            response = client.send_message(prompt, on_output=lambda src, txt, mode: None)

            # Get token usage
            usage = client.get_context_usage()
            token_usage = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
            }

            return SubagentResult(
                success=True,
                response=response,
                turns_used=usage.get('turns', 1),
                token_usage=token_usage,
            )

        except Exception as e:
            logger.exception("Subagent execution error")
            return SubagentResult(
                success=False,
                response='',
                error=str(e)
            )

        finally:
            # Clean up
            registry.unexpose_all()


def create_plugin() -> SubagentPlugin:
    """Factory function to create the subagent plugin.

    Returns:
        SubagentPlugin instance.
    """
    return SubagentPlugin()
