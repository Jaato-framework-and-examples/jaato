"""JaatoRuntime - Shared environment for the jaato framework.

Provides shared resources that can be used across multiple sessions (main agent
and subagents). This separates the "environment" (connections, plugins, permissions)
from the "session" (conversation history, per-agent state).
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .token_accounting import TokenLedger
from .instruction_token_cache import InstructionTokenCache
from jaato_sdk.plugins.model_provider.types import ToolSchema
from .plugins.model_provider.base import ProviderConfig
from .plugins.model_provider import load_provider
from .plugins.telemetry import TelemetryPlugin, create_plugin as create_telemetry_plugin

if TYPE_CHECKING:
    from .plugins.registry import PluginRegistry
    from .plugins.permission import PermissionPlugin
    from .plugins.reliability import ReliabilityPlugin
    from .plugins.model_provider.base import ModelProviderPlugin

# Framework-level instruction appended to all system prompts and tool results
_TASK_COMPLETION_INSTRUCTION = (
    "After each action, continue working until the request is truly fulfilled. "
    "Pause only for permissions or clarifications—never from uncertainty."
)

# Parallel tool execution guidance - encourages model to batch independent operations
_PARALLEL_TOOL_GUIDANCE = (
    "When you need to perform multiple independent operations (e.g., reading several files, "
    "searching multiple patterns, fetching multiple URLs), issue all tool calls in a single "
    "response rather than one at a time. Independent operations will execute in parallel, "
    "significantly reducing latency."
)

# Turn-end summary guidance - encourages model to summarize after complex tool-using turns
_TURN_SUMMARY_INSTRUCTION = (
    "After completing a complex turn involving multiple tool calls, provide a concise summary "
    "of what was done and why. This helps maintain context for future turns and enables "
    "efficient garbage collection of verbose intermediate outputs. Include: actions taken, "
    "goals accomplished, rationale for non-obvious decisions, and next steps if applicable. "
    "Skip summaries for simple single-tool lookups or direct conversational responses."
)


def _get_sandbox_guidance() -> Optional[str]:
    """Get sandbox guidance if workspace is configured.

    Returns sandbox awareness instructions if a workspace root is set,
    informing the model about path restrictions.
    """
    workspace = os.environ.get('JAATO_WORKSPACE_ROOT') or os.environ.get('workspaceRoot')
    if not workspace:
        return None

    return (
        f"SANDBOX ENVIRONMENT: You are operating in a sandboxed workspace. "
        f"File operations (read, write, glob, grep, cli) are restricted to: {workspace}\n"
        f"- Paths outside the workspace will be rejected\n"
        f"- Use relative paths or absolute paths within the workspace\n"
        f"- The .jaato/ directory may reference external configuration"
    )


def _is_parallel_tools_enabled() -> bool:
    """Check if parallel tool execution is enabled."""
    return os.environ.get(
        'JAATO_PARALLEL_TOOLS', 'true'
    ).lower() not in ('false', '0', 'no')


def _is_deferred_tools_enabled() -> bool:
    """Check if deferred tool loading is enabled.

    When enabled, only 'core' tools are loaded into the initial model context.
    Other tools can be discovered via the introspection plugin (list_tools,
    get_tool_schemas). This reduces initial context size significantly.

    Default is 'true' for token economy. Set JAATO_DEFERRED_TOOLS=false
    to disable and load all tools upfront.
    """
    return os.environ.get(
        'JAATO_DEFERRED_TOOLS', 'true'
    ).lower() not in ('false', '0', 'no')


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

    def __init__(self, provider_name: str = "google_genai",
                 workspace_path: Optional[Path] = None,
                 instruction_token_cache: Optional[InstructionTokenCache] = None):
        """Initialize JaatoRuntime.

        Args:
            provider_name: Name of the model provider to use (default: 'google_genai').
            workspace_path: Explicit workspace directory for loading instructions.
                When running as a daemon, the process cwd may differ from the
                client's workspace, so callers should pass the workspace path
                explicitly. Falls back to ``Path.cwd()`` when not provided.
            instruction_token_cache: Optional shared cache for instruction token
                counts.  When provided (e.g. from ``SessionManager``), cached
                counts survive across session creates/restores within the same
                daemon process.  When ``None``, a new per-runtime cache is
                created.
        """
        self._provider_name: str = provider_name
        self._workspace_path: Optional[Path] = workspace_path
        self._provider_config: Optional[ProviderConfig] = None

        # Multi-provider support: map provider_name -> ProviderConfig
        # Allows subagents to use different providers than the parent
        self._provider_configs: Dict[str, ProviderConfig] = {}

        # Connection info
        self._project: Optional[str] = None
        self._location: Optional[str] = None

        # Shared resources
        self._registry: Optional['PluginRegistry'] = None
        self._permission_plugin: Optional['PermissionPlugin'] = None
        self._reliability_plugin: Optional['ReliabilityPlugin'] = None
        self._ledger: Optional[TokenLedger] = None

        # Tool configuration cache (built from registry)
        self._all_tool_schemas: Optional[List[ToolSchema]] = None
        self._all_executors: Optional[Dict[str, Callable]] = None
        self._system_instructions: Optional[str] = None
        self._auto_approved_tools: List[str] = []

        # Formatter pipeline (optional, for collecting formatter instructions)
        self._formatter_pipeline: Optional[Any] = None

        # Base system instructions (loaded from .jaato/instructions/ or legacy single file)
        self._base_system_instructions: Optional[str] = None
        self._load_base_system_instructions()

        # Content-addressed token count cache (shared across sessions)
        self._instruction_token_cache: InstructionTokenCache = (
            instruction_token_cache or InstructionTokenCache()
        )

        # Connection state
        self._connected: bool = False

        # Telemetry plugin (created lazily, opt-in)
        self._telemetry: TelemetryPlugin = create_telemetry_plugin()

    def _load_base_system_instructions(self) -> None:
        """Load base system instructions from .jaato/instructions/ folder.

        Searches for instruction files in two locations (first match wins):
        1. Current working directory: .jaato/instructions/
        2. User config directory: ~/.jaato/instructions/

        All ``*.md`` files found in the instructions folder are sorted by
        filename (so numeric prefixes like ``00-``, ``10-``, ``15-`` control
        ordering) and concatenated with double-newline separators.

        Falls back to the legacy single-file path
        ``.jaato/system_instructions.md`` if no instructions folder exists
        in either location.

        The combined contents are prepended to all agent system instructions,
        ensuring consistent behavior across main agent and all subagents.
        """
        # Primary: look for an instructions/ folder
        # Use explicit workspace_path when provided (daemon mode), else cwd
        base = self._workspace_path or Path.cwd()
        search_dirs = [
            base / ".jaato" / "instructions",
            Path.home() / ".jaato" / "instructions",
        ]

        for instructions_dir in search_dirs:
            if instructions_dir.is_dir():
                parts = self._load_instruction_files(instructions_dir)
                if parts:
                    self._base_system_instructions = "\n\n".join(parts)
                    return

        # Fallback: legacy single-file path
        legacy_paths = [
            base / ".jaato" / "system_instructions.md",
            Path.home() / ".jaato" / "system_instructions.md",
        ]

        for path in legacy_paths:
            if path.exists() and path.is_file():
                try:
                    self._base_system_instructions = path.read_text(encoding='utf-8')
                    return
                except (IOError, OSError):
                    pass

    @staticmethod
    def _load_instruction_files(instructions_dir: Path) -> List[str]:
        """Load and concatenate all .md files from an instructions directory.

        Files are sorted lexicographically by filename, so numeric prefixes
        (e.g. ``00-system-instructions.md``, ``10-coding-standards.md``,
        ``15-review-policy.md``) control the order.

        ``README.md`` is excluded — it documents the folder layout and is
        not meant to be injected as system instructions.

        Args:
            instructions_dir: Path to the instructions directory.

        Returns:
            List of file contents (one entry per file), in sorted order.
            Empty list if no readable .md files are found.
        """
        parts: List[str] = []
        for md_file in sorted(instructions_dir.glob("*.md")):
            if md_file.name.upper() == "README.MD":
                continue  # Skip README files — they document the folder, not instructions
            if md_file.is_file():
                try:
                    content = md_file.read_text(encoding='utf-8')
                    if content.strip():
                        parts.append(content)
                except (IOError, OSError):
                    pass  # Silently skip unreadable files
        return parts

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
    def reliability_plugin(self) -> Optional['ReliabilityPlugin']:
        """Get the reliability plugin."""
        return self._reliability_plugin

    @property
    def ledger(self) -> Optional[TokenLedger]:
        """Get the token ledger."""
        return self._ledger

    @property
    def telemetry(self) -> TelemetryPlugin:
        """Get the telemetry plugin."""
        return self._telemetry

    @property
    def instruction_token_cache(self) -> InstructionTokenCache:
        """Get the instruction token cache.

        Shared across all sessions created from this runtime.  In daemon
        mode the same cache instance is passed from ``SessionManager`` so
        counts survive across session creates and restores.
        """
        return self._instruction_token_cache

    def set_formatter_pipeline(self, pipeline: Any) -> None:
        """Set the formatter pipeline for collecting formatter instructions.

        When set, get_system_instructions() will include instructions from
        output formatters that implement get_system_instructions(). This
        allows formatters to inform the model about rendering capabilities
        (e.g., mermaid diagram rendering) without being tool plugins.

        Args:
            pipeline: A FormatterPipeline instance (or any object with
                     a get_system_instructions() method).
        """
        self._formatter_pipeline = pipeline

    @property
    def deferred_tools_enabled(self) -> bool:
        """Check if deferred tool loading is enabled.

        When True, only 'core' tools are loaded into the initial model context.
        Other tools can be discovered via the introspection plugin.

        Controlled by JAATO_DEFERRED_TOOLS environment variable.
        """
        return _is_deferred_tools_enabled()

    def set_telemetry_plugin(self, plugin: TelemetryPlugin) -> None:
        """Set a custom telemetry plugin.

        Use this to configure OpenTelemetry tracing for observability.
        The plugin should be initialized before setting.

        Args:
            plugin: Configured TelemetryPlugin instance.

        Example:
            from shared.plugins.telemetry import create_otel_plugin

            telemetry = create_otel_plugin()
            telemetry.initialize({
                "enabled": True,
                "exporter": "otlp",
                "endpoint": "http://localhost:4317",
            })
            runtime.set_telemetry_plugin(telemetry)
        """
        self._telemetry = plugin

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
        # Register the primary provider config for multi-provider support
        self._provider_configs[self._provider_name] = self._provider_config
        self._connected = True

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Callable[[str], None]] = None,
        provider_name: Optional[str] = None
    ) -> bool:
        """Verify authentication before loading tools.

        This should be called BEFORE configure_plugins() or create_session()
        to ensure credentials are available. For providers that support
        interactive login (like Anthropic OAuth), this can trigger the login flow.

        Args:
            allow_interactive: If True and auth is not configured, attempt
                interactive login (e.g., browser-based OAuth).
            on_message: Optional callback for status messages during login.
            provider_name: Optional provider name to verify. If None, uses
                the runtime's default provider.

        Returns:
            True if authentication is configured and valid.
            False if authentication failed or was not completed.

        Raises:
            Various auth errors if allow_interactive=False and no credentials found.

        Example:
            runtime = JaatoRuntime(provider_name='anthropic')
            runtime.connect(project, location)

            # Verify auth with interactive login allowed
            if not runtime.verify_auth(allow_interactive=True, on_message=print):
                print("Authentication failed")
                return

            # Now safe to configure tools
            runtime.configure_plugins(registry, permission_plugin, ledger)
        """
        effective_provider = provider_name or self._provider_name

        # Create a temporary provider instance just for auth verification
        # We don't call initialize() yet - verify_auth is designed to work
        # before full initialization
        provider = load_provider(effective_provider, config=None)

        # Call verify_auth on the provider
        return provider.verify_auth(
            allow_interactive=allow_interactive,
            on_message=on_message
        )

    def register_provider(
        self,
        provider_name: str,
        config: Optional[ProviderConfig] = None
    ) -> None:
        """Register an additional provider for cross-provider subagent support.

        Allows subagents to use different AI providers than the parent agent.
        For example, the main agent can use Anthropic while a subagent uses
        Google GenAI for specific tasks.

        Args:
            provider_name: Name of the provider (e.g., 'anthropic', 'google_genai').
            config: Optional ProviderConfig. If None, creates a default config
                   using the runtime's project/location (may not work for all providers).

        Example:
            # Register Anthropic for subagents (uses ANTHROPIC_API_KEY env var)
            runtime.register_provider('anthropic')

            # Register Google GenAI with specific config
            runtime.register_provider('google_genai', ProviderConfig(
                project='my-project',
                location='us-central1'
            ))
        """
        if config is None:
            # Create default config - providers will use env vars for auth
            config = ProviderConfig(
                project=self._project or '',
                location=self._location or ''
            )
        self._provider_configs[provider_name] = config

    def configure_plugins(
        self,
        registry: 'PluginRegistry',
        permission_plugin: Optional['PermissionPlugin'] = None,
        ledger: Optional[TokenLedger] = None,
        reliability_plugin: Optional['ReliabilityPlugin'] = None,
    ) -> None:
        """Configure plugins for the runtime.

        Sets up the shared plugin registry, permission plugin, reliability plugin,
        and ledger that will be available to all sessions.

        Args:
            registry: PluginRegistry with exposed plugins.
            permission_plugin: Optional permission plugin for access control.
            ledger: Optional token ledger for accounting.
            reliability_plugin: Optional reliability plugin for failure tracking.
        """
        self._registry = registry
        self._permission_plugin = permission_plugin
        self._reliability_plugin = reliability_plugin
        self._ledger = ledger

        # Give permission plugin access to registry for plugin lookups
        if permission_plugin:
            permission_plugin.set_registry(registry)

        # Configure reliability plugin
        if reliability_plugin:
            reliability_plugin.set_registry(registry)
            # Connect telemetry if enabled
            if self._telemetry and self._telemetry.enabled:
                reliability_plugin.set_telemetry(self._telemetry)

        # Cache tool configuration from registry
        self._cache_tool_configuration()

        # Configure subagent plugin with runtime reference
        self._configure_subagent_plugin()

        # Configure background plugin
        self._configure_background_plugin()

    def _cache_tool_configuration(self) -> None:
        """Cache tool schemas and executors from registry.

        Uses get_enabled_* methods to respect disabled tools set in the registry.
        When JAATO_DEFERRED_TOOLS is enabled, only 'core' tools are included
        in the schema cache (other tools can be discovered via introspection).

        Call refresh_tool_cache() after enabling/disabling tools to update the cache.
        """
        if not self._registry:
            return

        # Get tool schemas based on deferred loading setting
        if _is_deferred_tools_enabled():
            # Deferred loading: only core tools in initial context
            self._all_tool_schemas = self._registry.get_core_tool_schemas()
        else:
            # Traditional: all enabled tools in initial context
            self._all_tool_schemas = self._registry.get_enabled_tool_schemas()

        # Add permission plugin schemas if available (but avoid duplicates)
        # Permission plugin may already be exposed via registry.expose_tool("permission")
        if self._permission_plugin:
            existing_names = {s.name for s in self._all_tool_schemas}
            for schema in self._permission_plugin.get_tool_schemas():
                if schema.name not in existing_names:
                    self._all_tool_schemas.append(schema)

        # Add reliability plugin schemas if available (but avoid duplicates)
        if self._reliability_plugin:
            existing_names = {s.name for s in self._all_tool_schemas}
            for schema in self._reliability_plugin.get_tool_schemas():
                if schema.name not in existing_names:
                    self._all_tool_schemas.append(schema)

        # Get enabled executors (respects disabled tools set)
        self._all_executors = dict(self._registry.get_enabled_executors())

        # Add permission plugin executors (dict update handles duplicates)
        if self._permission_plugin:
            for name, fn in self._permission_plugin.get_executors().items():
                self._all_executors[name] = fn

        # Add reliability plugin executors
        if self._reliability_plugin:
            for name, fn in self._reliability_plugin.get_executors().items():
                self._all_executors[name] = fn

        # Build system instructions
        parts = []

        # Core framework instructions (sandbox, parallel tools)
        sandbox_guidance = _get_sandbox_guidance()
        if sandbox_guidance:
            parts.append(sandbox_guidance)

        registry_instructions = self._registry.get_system_instructions()
        if registry_instructions:
            parts.append(registry_instructions)
        if self._permission_plugin:
            perm_instructions = self._permission_plugin.get_system_instructions()
            if perm_instructions:
                parts.append(perm_instructions)
        if self._reliability_plugin:
            reliability_instructions = self._reliability_plugin.get_system_instructions()
            if reliability_instructions:
                parts.append(reliability_instructions)
        self._system_instructions = "\n\n".join(parts) if parts else None

        # Get auto-approved tools from plugins
        self._auto_approved_tools = self._registry.get_auto_approved_tools()

        # Add built-in user commands to auto-approved list
        # User commands are invoked directly by the user, not the model
        builtin_user_commands = ["model"]
        self._auto_approved_tools.extend(builtin_user_commands)

        # Add reliability plugin's auto-approved tools
        if self._reliability_plugin:
            reliability_auto_approved = self._reliability_plugin.get_auto_approved_tools()
            self._auto_approved_tools.extend(reliability_auto_approved)

        if self._permission_plugin and self._auto_approved_tools:
            self._permission_plugin.add_whitelist_tools(self._auto_approved_tools)

    def refresh_tool_cache(self) -> None:
        """Refresh the cached tool configuration.

        Call this after enabling/disabling tools in the registry to update
        the cached schemas and executors.
        """
        self._cache_tool_configuration()

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
        system_instructions: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        provider_name: Optional[str] = None,
        preloaded_plugins: Optional[set] = None
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
            plugin_configs: Optional per-plugin configuration overrides.
                           Plugins will be re-initialized with these configs.
            provider_name: Optional provider override for cross-provider subagents.
                          If specified, the session uses a different AI provider
                          (e.g., 'anthropic', 'google_genai') than the runtime default.
            preloaded_plugins: Optional set of plugin names that should bypass
                              deferred tool loading. All their tools (including
                              discoverable) are loaded into the initial context.

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

        # Create session with runtime reference and optional provider override
        session = JaatoSession(self, model, provider_name=provider_name)

        # Configure session tools
        session.configure(
            tools=tools,
            system_instructions=system_instructions,
            plugin_configs=plugin_configs,
            preloaded_plugins=preloaded_plugins
        )

        return session

    def create_session_without_provider(
        self,
        model: str,
        tools: Optional[List[str]] = None,
        system_instructions: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> 'JaatoSession':
        """Create a session without provider (for auth-pending mode).

        This creates a session with user commands available but no model
        connection. Used when authentication is pending and the user needs
        to complete auth before the model can be used.

        Args:
            model: Model name (stored for later use after auth completes).
            tools: Optional list of plugin names to expose.
            system_instructions: Optional additional system instructions.
            plugin_configs: Optional per-plugin configuration overrides.

        Returns:
            JaatoSession configured without a provider.
        """
        if not self._connected:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        if not self._registry:
            raise RuntimeError("Plugins not configured. Call configure_plugins() first.")

        from .jaato_session import JaatoSession

        session = JaatoSession(self, model)
        session.configure(
            tools=tools,
            system_instructions=system_instructions,
            plugin_configs=plugin_configs,
            skip_provider=True  # Don't create provider
        )

        return session

    def create_provider(
        self,
        model: str,
        provider_name: Optional[str] = None
    ) -> 'ModelProviderPlugin':
        """Create a new provider instance for a session.

        Each session gets its own provider instance to maintain
        independent conversation state.

        Args:
            model: Model name to connect to.
            provider_name: Optional provider name override. If specified,
                          uses a different provider than the runtime's default.
                          The provider must be registered via register_provider()
                          or will be auto-registered with default config.

        Returns:
            Initialized and connected ModelProviderPlugin.

        Raises:
            RuntimeError: If runtime is not connected.
        """
        if not self._connected or not self._provider_config:
            raise RuntimeError("Runtime not connected. Call connect() first.")

        # Use specified provider or fall back to default
        effective_provider = provider_name or self._provider_name

        # Get or create provider config
        if effective_provider in self._provider_configs:
            config = self._provider_configs[effective_provider]
        else:
            # Auto-register provider with default config
            # This enables cross-provider subagents without explicit registration
            config = ProviderConfig(
                project=self._project or '',
                location=self._location or ''
            )
            self._provider_configs[effective_provider] = config

        # Inject workspace_path into config.extra for providers that need it
        # (e.g., GitHub Models provider for OAuth token resolution)
        if self._registry:
            workspace_path = self._registry.get_workspace_path()
            if workspace_path:
                # Create a copy of config with workspace_path in extra to avoid
                # modifying the stored config
                from dataclasses import replace
                extra_with_workspace = {**config.extra, 'workspace_path': workspace_path}
                config = replace(config, extra=extra_with_workspace)

        provider = load_provider(effective_provider, config)
        provider.connect(model)
        return provider

    def _get_core_plugins(self) -> List[str]:
        """Find all plugins that provide tools with discoverability='core'.

        These plugins are essential for the framework to function and should
        always be included regardless of profile configuration.

        Returns:
            List of plugin names that have at least one core tool.
        """
        if not self._registry:
            return []

        core_plugins = []
        for plugin_name, plugin in self._registry._plugins.items():
            if not hasattr(plugin, 'get_tool_schemas'):
                continue
            try:
                schemas = plugin.get_tool_schemas()
                for schema in schemas:
                    if getattr(schema, 'discoverability', None) == 'core':
                        core_plugins.append(plugin_name)
                        break  # Found one core tool, plugin qualifies
            except Exception:
                pass  # Skip plugins that fail to provide schemas

        return core_plugins

    def _get_essential_plugins(self, plugin_names: List[str]) -> List[str]:
        """Get plugin list with core plugins added automatically.

        Plugins that provide tools with discoverability='core' are essential
        for the framework to function (e.g., introspection for tool discovery).
        These plugins are automatically included even if not explicitly listed
        in profile definitions.

        Also ensures core plugins are properly exposed (initialized) in the
        registry so they function correctly.

        Args:
            plugin_names: Original list of plugin names from profile.

        Returns:
            Plugin list with core plugins added (if not already present).
        """
        # Find all plugins with core tools
        core_plugins = self._get_core_plugins()

        result = list(plugin_names)
        for name in core_plugins:
            if name not in result:
                result.append(name)
            # Ensure core plugin is exposed (initialized with registry access)
            if self._registry and name not in self._registry._exposed:
                try:
                    self._registry.expose_tool(name)
                except ValueError:
                    pass  # Plugin not discovered, skip

        return result

    def get_tool_schemas(
        self,
        plugin_names: Optional[List[str]] = None,
        preloaded_plugins: Optional[set] = None
    ) -> List[ToolSchema]:
        """Get tool schemas, optionally filtered by plugin names.

        When deferred tool loading is enabled, only 'core' tools are returned
        in the initial context. Other tools must be discovered via introspection
        (list_tools, get_tool_schemas). This applies to both main agents and
        subagents for consistent behavior.

        Plugins listed in ``preloaded_plugins`` bypass deferral — all their
        tools (including discoverable) are loaded into the initial context.

        Args:
            plugin_names: Optional list of plugin names to include.
                         If None, returns all exposed tool schemas.
            preloaded_plugins: Optional set of plugin names that should bypass
                              deferred tool loading.

        Returns:
            List of ToolSchema objects.
        """
        if not self._registry:
            return []

        if plugin_names is None:
            # Return all cached schemas
            return list(self._all_tool_schemas) if self._all_tool_schemas else []

        # Add essential plugins (introspection) when deferred tools is enabled
        effective_plugins = self._get_essential_plugins(plugin_names)

        # Filter to specific plugins
        schemas = []
        deferred_enabled = _is_deferred_tools_enabled()
        _preloaded = preloaded_plugins or set()
        for name in effective_plugins:
            plugin = self._registry.get_plugin(name)
            if plugin and hasattr(plugin, 'get_tool_schemas'):
                plugin_schemas = plugin.get_tool_schemas()
                if deferred_enabled and name not in _preloaded:
                    # Filter to core tools only - others discovered via introspection
                    plugin_schemas = [
                        s for s in plugin_schemas
                        if getattr(s, 'discoverability', 'discoverable') == 'core'
                    ]
                schemas.extend(plugin_schemas)

        # Add permission plugin schemas if permission plugin is configured
        if self._permission_plugin:
            permission_schemas = self._permission_plugin.get_tool_schemas()
            if deferred_enabled:
                # Permission tools should be core (always available)
                permission_schemas = [
                    s for s in permission_schemas
                    if getattr(s, 'discoverability', 'discoverable') == 'core'
                ]
            schemas.extend(permission_schemas)

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

        # Add essential plugins (introspection) when deferred tools is enabled
        effective_plugins = self._get_essential_plugins(plugin_names)

        # Filter to specific plugins
        executors = {}
        for name in effective_plugins:
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
        additional: Optional[str] = None,
        presentation_context: Optional['PresentationContext'] = None,
        preloaded_plugins: Optional[set] = None,
    ) -> Optional[str]:
        """Get system instructions, optionally filtered by plugin names.

        The final instructions are assembled in this order:
        1. Base system instructions from .jaato/instructions/ folder (if exists,
           falls back to legacy .jaato/system_instructions.md)
        2. Additional instructions passed as parameter
        3. Plugin-specific system instructions
        4. Formatter pipeline instructions (output rendering capabilities)
        5. Presentation context (client display constraints)
        6. Framework-level task completion instruction
        7. Parallel tool guidance
        8. Turn-end summary guidance

        This ensures base behavioral rules (like transparency, no silent pauses)
        apply consistently to all agents (main and subagents).

        Plugins listed in ``preloaded_plugins`` bypass deferral — their system
        instructions are included even if they have no core tools.

        Args:
            plugin_names: Optional list of plugin names to include.
                         If None, returns full cached system instructions.
            additional: Optional additional instructions to prepend.
            presentation_context: Optional client display context.  When
                provided, a compact display-constraint block is appended so
                the model can adapt its output format (tables, lists, etc.)
                to the client's capabilities.
            preloaded_plugins: Optional set of plugin names that should bypass
                              deferred tool loading for system instructions.

        Returns:
            Combined system instructions string, or None.
        """
        deferred_enabled = _is_deferred_tools_enabled()

        if plugin_names is None:
            # Use registry's method which runs enrichment pipeline
            if self._registry:
                plugin_instructions = self._registry.get_system_instructions(
                    run_enrichment=True,
                    skip_discoverable_only=deferred_enabled,
                )
            else:
                plugin_instructions = self._system_instructions
        else:
            # Add essential plugins (introspection) when deferred tools is enabled
            effective_plugins = self._get_essential_plugins(plugin_names)

            # Build from specific plugins, then run enrichment
            parts = []
            _preloaded = preloaded_plugins or set()
            if self._registry:
                for name in effective_plugins:
                    # When deferred tools are enabled, skip system instructions
                    # from plugins that have no core tools — their instructions
                    # will be injected when the model discovers their tools.
                    # Exception: preloaded plugins always include instructions.
                    if deferred_enabled and name not in _preloaded and not self._registry.plugin_has_core_tools(name):
                        continue
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

            plugin_instructions = "\n\n".join(parts) if parts else None

            # Run enrichment pipeline on combined instructions
            if plugin_instructions and self._registry:
                result = self._registry.enrich_system_instructions(plugin_instructions)
                plugin_instructions = result.instructions

        # Assemble final instructions: base -> additional -> plugin
        result_parts = []

        # 1. Base system instructions from .jaato/instructions/ (or legacy single file)
        if self._base_system_instructions:
            result_parts.append(self._base_system_instructions)

        # 2. Additional instructions passed as parameter
        if additional:
            result_parts.append(additional)

        # 3. Plugin-specific system instructions
        if plugin_instructions:
            result_parts.append(plugin_instructions)

        # 4. Formatter pipeline instructions (output rendering capabilities)
        if self._formatter_pipeline and hasattr(self._formatter_pipeline, 'get_system_instructions'):
            formatter_instructions = self._formatter_pipeline.get_system_instructions()
            if formatter_instructions:
                result_parts.append(formatter_instructions)

        # 5. Presentation context (client display constraints and capabilities)
        if presentation_context is not None:
            ctx_instruction = presentation_context.to_system_instruction()
            if ctx_instruction:
                result_parts.append(ctx_instruction)

        # 6. Framework-level task completion instruction (always included)
        result_parts.append(_TASK_COMPLETION_INSTRUCTION)

        # 7. Parallel tool guidance (when parallel execution is enabled)
        if _is_parallel_tools_enabled():
            result_parts.append(_PARALLEL_TOOL_GUIDANCE)

        # 8. Turn-end summary guidance (always included)
        result_parts.append(_TURN_SUMMARY_INSTRUCTION)

        return "\n\n".join(result_parts)

    def list_available_models(
        self,
        prefix: Optional[str] = None,
        provider_name: Optional[str] = None
    ) -> List[str]:
        """List available models from a provider.

        Args:
            prefix: Optional name prefix to filter by.
            provider_name: Optional provider to list models from. If not specified,
                          uses the runtime's default provider.

        Returns:
            List of model names.

        Raises:
            RuntimeError: If runtime is not connected.
        """
        if not self._connected or not self._provider_config:
            raise RuntimeError("Runtime not connected. Call connect() first.")

        # Use specified provider or fall back to default
        effective_provider = provider_name or self._provider_name

        # Get config for the provider
        if effective_provider in self._provider_configs:
            config = self._provider_configs[effective_provider]
        else:
            # Use default config for unregistered providers
            config = ProviderConfig(
                project=self._project or '',
                location=self._location or ''
            )

        # Create a temporary provider to list models
        # Note: initialize() sets up the client, connect() just selects a model
        # We don't need to connect to list available models
        provider = load_provider(effective_provider, config)
        return provider.list_models(prefix=prefix)


__all__ = ['JaatoRuntime']
