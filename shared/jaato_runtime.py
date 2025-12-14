"""JaatoRuntime - Shared environment for the jaato framework.

Provides shared resources that can be used across multiple sessions (main agent
and subagents). This separates the "environment" (connections, plugins, permissions)
from the "session" (conversation history, per-agent state).
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .token_accounting import TokenLedger
from .plugins.model_provider.types import ToolSchema
from .plugins.model_provider.base import ProviderConfig
from .plugins.model_provider import load_provider

if TYPE_CHECKING:
    from .plugins.registry import PluginRegistry
    from .plugins.permission import PermissionPlugin
    from .plugins.model_provider.base import ModelProviderPlugin


class JaatoRuntime:
    """Shared runtime environment for jaato agents.

    JaatoRuntime manages resources that are shared across the main agent
    and any subagents:
    - Provider configuration (project, location)
    - Plugin registry (discovered once, shared)
    - Permission plugin (shared across sessions)
    - Token ledger (aggregated accounting)

    Sessions created from this runtime share these resources while
    maintaining their own conversation history and tool configurations.

    Usage:
        # Create and configure runtime
        runtime = JaatoRuntime()
        runtime.connect(project_id, location)
        runtime.configure_plugins(registry, permission_plugin, ledger)

        # Create sessions from the runtime
        main_session = runtime.create_session(model="gemini-2.5-flash")
        sub_session = runtime.create_session(
            model="gemini-2.5-flash",
            tools=["cli", "web_search"],
            system_instructions="You are a research assistant."
        )
    """

    def __init__(self, provider_name: str = "google_genai"):
        """Initialize JaatoRuntime.

        Args:
            provider_name: Name of the model provider to use (default: 'google_genai').
        """
        self._provider_name: str = provider_name
        self._provider_config: Optional[ProviderConfig] = None

        # Connection info
        self._project: Optional[str] = None
        self._location: Optional[str] = None

        # Shared resources
        self._registry: Optional['PluginRegistry'] = None
        self._permission_plugin: Optional['PermissionPlugin'] = None
        self._ledger: Optional[TokenLedger] = None

        # Tool configuration cache (built from registry)
        self._all_tool_schemas: Optional[List[ToolSchema]] = None
        self._all_executors: Optional[Dict[str, Callable]] = None
        self._system_instructions: Optional[str] = None
        self._auto_approved_tools: List[str] = []

        # Connection state
        self._connected: bool = False

    @property
    def is_connected(self) -> bool:
        """Check if runtime is connected."""
        return self._connected

    @property
    def project(self) -> Optional[str]:
        """Get the configured project ID."""
        return self._project

    @property
    def location(self) -> Optional[str]:
        """Get the configured location."""
        return self._location

    @property
    def provider_name(self) -> str:
        """Get the model provider name."""
        return self._provider_name

    @property
    def registry(self) -> Optional['PluginRegistry']:
        """Get the plugin registry."""
        return self._registry

    @property
    def permission_plugin(self) -> Optional['PermissionPlugin']:
        """Get the permission plugin."""
        return self._permission_plugin

    @property
    def ledger(self) -> Optional[TokenLedger]:
        """Get the token ledger."""
        return self._ledger

    def connect(self, project: str, location: str) -> None:
        """Connect to the AI provider.

        Establishes the provider configuration that will be used for
        all sessions created from this runtime.

        Args:
            project: Cloud project ID (e.g., GCP project).
            location: Provider region (e.g., 'us-central1', 'global').
        """
        self._project = project
        self._location = location
        self._provider_config = ProviderConfig(project=project, location=location)
        self._connected = True

    def configure_plugins(
        self,
        registry: 'PluginRegistry',
        permission_plugin: Optional['PermissionPlugin'] = None,
        ledger: Optional[TokenLedger] = None
    ) -> None:
        """Configure plugins for the runtime.

        Sets up the shared plugin registry, permission plugin, and ledger
        that will be available to all sessions.

        Args:
            registry: PluginRegistry with exposed plugins.
            permission_plugin: Optional permission plugin for access control.
            ledger: Optional token ledger for accounting.
        """
        self._registry = registry
        self._permission_plugin = permission_plugin
        self._ledger = ledger

        # Give permission plugin access to registry for plugin lookups
        if permission_plugin:
            permission_plugin.set_registry(registry)

        # Cache tool configuration from registry
        self._cache_tool_configuration()

        # Configure subagent plugin with runtime reference
        self._configure_subagent_plugin()

        # Configure background plugin
        self._configure_background_plugin()

    def _cache_tool_configuration(self) -> None:
        """Cache tool schemas and executors from registry."""
        if not self._registry:
            return

        # Get all exposed tool schemas
        self._all_tool_schemas = self._registry.get_exposed_tool_schemas()

        # Add permission plugin schemas if available
        if self._permission_plugin:
            self._all_tool_schemas.extend(self._permission_plugin.get_tool_schemas())

        # Get all exposed executors
        self._all_executors = dict(self._registry.get_exposed_executors())

        # Add permission plugin executors
        if self._permission_plugin:
            for name, fn in self._permission_plugin.get_executors().items():
                self._all_executors[name] = fn

        # Build system instructions
        parts = []
        registry_instructions = self._registry.get_system_instructions()
        if registry_instructions:
            parts.append(registry_instructions)
        if self._permission_plugin:
            perm_instructions = self._permission_plugin.get_system_instructions()
            if perm_instructions:
                parts.append(perm_instructions)
        self._system_instructions = "\n\n".join(parts) if parts else None

        # Get auto-approved tools from plugins
        self._auto_approved_tools = self._registry.get_auto_approved_tools()

        # Add built-in user commands to auto-approved list
        # User commands are invoked directly by the user, not the model
        builtin_user_commands = ["model"]
        self._auto_approved_tools.extend(builtin_user_commands)

        if self._permission_plugin and self._auto_approved_tools:
            self._permission_plugin.add_whitelist_tools(self._auto_approved_tools)

    def _configure_subagent_plugin(self) -> None:
        """Configure subagent plugin with runtime reference."""
        if not self._registry:
            return

        try:
            subagent_plugin = self._registry.get_plugin('subagent')
            if not subagent_plugin:
                return

            # Pass runtime reference for session creation
            if hasattr(subagent_plugin, 'set_runtime'):
                subagent_plugin.set_runtime(self)

            # Pass parent's exposed plugins for inheritance
            if hasattr(subagent_plugin, 'set_parent_plugins'):
                exposed = self._registry.list_exposed()
                parent_plugins = [p for p in exposed if p != 'subagent']
                subagent_plugin.set_parent_plugins(parent_plugins)

            # Pass permission plugin for subagent tool execution
            if self._permission_plugin and hasattr(subagent_plugin, 'set_permission_plugin'):
                subagent_plugin.set_permission_plugin(self._permission_plugin)

        except (KeyError, AttributeError):
            pass

    def _configure_background_plugin(self) -> None:
        """Configure background plugin with registry reference."""
        if not self._registry:
            return

        try:
            background_plugin = self._registry.get_plugin('background')
            if background_plugin and hasattr(background_plugin, 'set_registry'):
                background_plugin.set_registry(self._registry)
        except (KeyError, AttributeError):
            pass

    def create_session(
        self,
        model: str,
        tools: Optional[List[str]] = None,
        system_instructions: Optional[str] = None
    ) -> 'JaatoSession':
        """Create a new session from this runtime.

        Sessions share the runtime's resources (registry, permissions, ledger)
        but have their own conversation history and can use different models
        or tool subsets.

        Args:
            model: Model name to use for this session.
            tools: Optional list of plugin names to expose. If None, uses all
                   exposed plugins from the registry.
            system_instructions: Optional additional system instructions to
                                prepend to the base instructions.

        Returns:
            JaatoSession configured with the specified settings.

        Raises:
            RuntimeError: If runtime is not connected or configured.
        """
        if not self._connected:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        if not self._registry:
            raise RuntimeError("Plugins not configured. Call configure_plugins() first.")

        # Import here to avoid circular dependency
        from .jaato_session import JaatoSession

        # Create session with runtime reference
        session = JaatoSession(self, model)

        # Configure session tools
        session.configure(
            tools=tools,
            system_instructions=system_instructions
        )

        return session

    def create_provider(self, model: str) -> 'ModelProviderPlugin':
        """Create a new provider instance for a session.

        Each session gets its own provider instance to maintain
        independent conversation state.

        Args:
            model: Model name to connect to.

        Returns:
            Initialized and connected ModelProviderPlugin.

        Raises:
            RuntimeError: If runtime is not connected.
        """
        if not self._connected or not self._provider_config:
            raise RuntimeError("Runtime not connected. Call connect() first.")

        provider = load_provider(self._provider_name, self._provider_config)
        provider.connect(model)
        return provider

    def get_tool_schemas(
        self,
        plugin_names: Optional[List[str]] = None
    ) -> List[ToolSchema]:
        """Get tool schemas, optionally filtered by plugin names.

        Args:
            plugin_names: Optional list of plugin names to include.
                         If None, returns all exposed tool schemas.

        Returns:
            List of ToolSchema objects.
        """
        if not self._registry:
            return []

        if plugin_names is None:
            # Return all cached schemas
            return list(self._all_tool_schemas) if self._all_tool_schemas else []

        # Filter to specific plugins
        schemas = []
        for name in plugin_names:
            plugin = self._registry.get_plugin(name)
            if plugin and hasattr(plugin, 'get_tool_schemas'):
                schemas.extend(plugin.get_tool_schemas())

        # Add permission plugin schemas if permission plugin is configured
        if self._permission_plugin:
            schemas.extend(self._permission_plugin.get_tool_schemas())

        return schemas

    def get_executors(
        self,
        plugin_names: Optional[List[str]] = None
    ) -> Dict[str, Callable]:
        """Get executors, optionally filtered by plugin names.

        Args:
            plugin_names: Optional list of plugin names to include.
                         If None, returns all exposed executors.

        Returns:
            Dict mapping tool names to executor functions.
        """
        if not self._registry:
            return {}

        if plugin_names is None:
            # Return all cached executors
            return dict(self._all_executors) if self._all_executors else {}

        # Filter to specific plugins
        executors = {}
        for name in plugin_names:
            plugin = self._registry.get_plugin(name)
            if plugin and hasattr(plugin, 'get_executors'):
                executors.update(plugin.get_executors())

        # Add permission plugin executors if configured
        if self._permission_plugin:
            executors.update(self._permission_plugin.get_executors())

        return executors

    def get_system_instructions(
        self,
        plugin_names: Optional[List[str]] = None,
        additional: Optional[str] = None
    ) -> Optional[str]:
        """Get system instructions, optionally filtered by plugin names.

        Args:
            plugin_names: Optional list of plugin names to include.
                         If None, returns full cached system instructions.
            additional: Optional additional instructions to prepend.

        Returns:
            Combined system instructions string, or None.
        """
        if plugin_names is None:
            base = self._system_instructions
        else:
            # Build from specific plugins
            parts = []
            if self._registry:
                for name in plugin_names:
                    plugin = self._registry.get_plugin(name)
                    if plugin and hasattr(plugin, 'get_system_instructions'):
                        instr = plugin.get_system_instructions()
                        if instr:
                            parts.append(instr)

            # Add permission plugin instructions
            if self._permission_plugin:
                perm_instr = self._permission_plugin.get_system_instructions()
                if perm_instr:
                    parts.append(perm_instr)

            base = "\n\n".join(parts) if parts else None

        # Combine with additional instructions
        if additional:
            if base:
                return f"{additional}\n\n{base}"
            return additional
        return base

    def list_available_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models from the provider.

        Args:
            prefix: Optional name prefix to filter by.

        Returns:
            List of model names.

        Raises:
            RuntimeError: If runtime is not connected.
        """
        if not self._connected or not self._provider_config:
            raise RuntimeError("Runtime not connected. Call connect() first.")

        # Create a temporary provider to list models
        # Note: initialize() sets up the client, connect() just selects a model
        # We don't need to connect to list available models
        provider = load_provider(self._provider_name, self._provider_config)
        return provider.list_models(prefix=prefix)


__all__ = ['JaatoRuntime']
