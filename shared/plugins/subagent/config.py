"""Configuration models for subagent plugin."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def expand_variables(
    value: Any,
    context: Optional[Dict[str, str]] = None,
    workspace_root_override: Optional[str] = None
) -> Any:
    """Expand ${variable} references in a value.

    Supports:
    - Environment variables: ${HOME}, ${USER}, ${PATH}
    - Context variables: ${projectPath}, ${workspaceRoot}, ${cwd}
    - Nested expansion in dicts and lists

    Args:
        value: Value to expand (string, dict, list, or other)
        context: Optional dict of context variables to expand
        workspace_root_override: Explicit workspace root to use instead of auto-detection.
            This is useful when the calling code knows the correct workspace root
            (e.g., from parent agent's config or environment).

    Returns:
        Value with variables expanded

    Examples:
        >>> expand_variables("${HOME}/projects", {})
        '/home/user/projects'

        >>> expand_variables({"path": "${projectPath}/.lsp.json"}, {"projectPath": "/app"})
        {'path': '/app/.lsp.json'}
    """
    if context is None:
        context = {}

    # Add default context variables
    # Use workspace_root_override if provided, otherwise auto-detect
    default_context = {
        'cwd': os.getcwd(),
        'workspaceRoot': _find_workspace_root(workspace_root_override),
        'HOME': os.environ.get('HOME', ''),
        'USER': os.environ.get('USER', ''),
    }
    # Merge with provided context (provided takes precedence)
    effective_context = {**default_context, **context}

    if isinstance(value, str):
        return _expand_string(value, effective_context)
    elif isinstance(value, dict):
        return {k: expand_variables(v, context, workspace_root_override) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_variables(item, context, workspace_root_override) for item in value]
    else:
        return value


def _expand_string(s: str, context: Dict[str, str]) -> str:
    """Expand ${variable} references in a string.

    Args:
        s: String containing ${variable} references
        context: Dict of variable names to values

    Returns:
        String with variables expanded
    """
    if '${' not in s:
        return s

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        # First check context, then environment
        if var_name in context:
            return context[var_name]
        return os.environ.get(var_name, match.group(0))  # Keep original if not found

    # Match ${VAR_NAME} pattern
    pattern = r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
    return re.sub(pattern, replace_var, s)


def _resolve_workspace_path(path: str) -> str:
    """Resolve a workspace path, handling relative paths.

    Relative paths (like ".") are resolved relative to the current
    working directory.

    Args:
        path: The workspace path (absolute or relative like ".").

    Returns:
        Absolute path to the workspace.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p.resolve())


def _find_workspace_root(override: Optional[str] = None) -> str:
    """Find the workspace root by looking for .git directory.

    Priority order:
    1. Explicit override parameter
    2. JAATO_WORKSPACE_ROOT environment variable
    3. workspaceRoot environment variable (from .env file)
    4. Search for .git or .jaato directory from cwd

    Note: Relative paths (like ".") are resolved relative to cwd.

    Args:
        override: Explicit workspace root path to use (takes precedence).

    Returns:
        Path to workspace root, or cwd if not found
    """
    # Priority 1: Explicit override
    if override:
        return _resolve_workspace_path(override)

    # Priority 2: JAATO_WORKSPACE_ROOT environment variable
    env_root = os.environ.get('JAATO_WORKSPACE_ROOT')
    if env_root:
        return _resolve_workspace_path(env_root)

    # Priority 3: workspaceRoot environment variable (common in .env files)
    env_workspace_root = os.environ.get('workspaceRoot')
    if env_workspace_root:
        return _resolve_workspace_path(env_workspace_root)

    # Priority 4: Search for .git or .jaato directory
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / '.git').exists():
            return str(parent)
        if (parent / '.jaato').exists():
            return str(parent)
    return str(current)


def expand_plugin_configs(
    plugin_configs: Dict[str, Dict[str, Any]],
    context: Optional[Dict[str, str]] = None,
    workspace_root_override: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Expand variables in all plugin configurations.

    Args:
        plugin_configs: Dict of plugin name -> config dict
        context: Optional context variables (e.g., projectPath)
        workspace_root_override: Explicit workspace root to use instead of auto-detection.
            If provided, ${workspaceRoot} will expand to this value.

    Returns:
        Plugin configs with all variables expanded

    Example:
        >>> configs = {
        ...     "lsp": {"config_path": "${projectPath}/.lsp.json"},
        ...     "mcp": {"config_path": "${projectPath}/.mcp.json"}
        ... }
        >>> expand_plugin_configs(configs, {"projectPath": "/app"})
        {'lsp': {'config_path': '/app/.lsp.json'}, 'mcp': {'config_path': '/app/.mcp.json'}}
    """
    return expand_variables(plugin_configs, context, workspace_root_override)


@dataclass
class GCProfileConfig:
    """Garbage collection configuration for a profile.

    Defines the GC strategy and its configuration for a subagent or main agent.

    Attributes:
        type: GC strategy type ('truncate', 'summarize', 'hybrid', 'budget').
        threshold_percent: Trigger GC when context usage exceeds this percentage.
        target_percent: Target usage after GC (default: 60.0).
        pressure_percent: When PRESERVABLE can be touched (0 = continuous mode).
        preserve_recent_turns: Number of recent turns to always preserve.
        notify_on_gc: Whether to inject a notification into history after GC.
        summarize_middle_turns: For hybrid strategy, number of middle turns to summarize.
        max_turns: Trigger GC when turn count exceeds this limit.
        plugin_config: Additional plugin-specific configuration.
    """
    type: str = "truncate"
    threshold_percent: float = 80.0
    target_percent: float = 60.0
    pressure_percent: Optional[float] = 90.0  # 0 or None = continuous mode
    preserve_recent_turns: int = 5
    notify_on_gc: bool = True
    summarize_middle_turns: Optional[int] = None  # For hybrid strategy
    max_turns: Optional[int] = None
    plugin_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def continuous_mode(self) -> bool:
        """True if continuous GC is enabled (pressure_percent is 0 or None)."""
        return not self.pressure_percent

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GCProfileConfig':
        """Create GCProfileConfig from a dictionary."""
        return cls(
            type=data.get('type', 'truncate'),
            threshold_percent=data.get('threshold_percent', 80.0),
            target_percent=data.get('target_percent', 60.0),
            pressure_percent=data.get('pressure_percent', 90.0),
            preserve_recent_turns=data.get('preserve_recent_turns', 5),
            notify_on_gc=data.get('notify_on_gc', True),
            summarize_middle_turns=data.get('summarize_middle_turns'),
            max_turns=data.get('max_turns'),
            plugin_config=data.get('plugin_config', {}),
        )


@dataclass
class SubagentProfile:
    """Configuration profile for a subagent.

    Defines what tools and capabilities a subagent has access to,
    allowing the parent model to delegate specialized tasks.

    Attributes:
        name: Unique identifier for this subagent profile.
        description: Human-readable description of what this subagent does.
        plugins: List of plugin names to enable for this subagent.
        plugin_configs: Per-plugin configuration overrides.
        system_instructions: Additional system instructions for the subagent.
        model: Optional model override (uses parent's model if not specified).
        provider: Optional provider override (e.g., 'anthropic', 'google_genai').
                  Allows subagents to use a different provider than the parent.
        max_turns: Maximum conversation turns before returning (default: 10).
        auto_approved: Whether this subagent can be spawned without permission.
        icon: Optional custom ASCII art icon (3 lines) for UI visualization.
        icon_name: Optional name of predefined icon (e.g., "code_assistant").
        gc: Optional garbage collection configuration for this subagent.
    """
    name: str
    description: str
    plugins: List[str] = field(default_factory=list)
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_instructions: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    max_turns: int = 10
    auto_approved: bool = False
    icon: Optional[List[str]] = None
    icon_name: Optional[str] = None
    gc: Optional[GCProfileConfig] = None


def _parse_profile_file(file_path: Path) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parse a single profile file (JSON or YAML).

    Args:
        file_path: Path to the profile file.

    Returns:
        Tuple of (profile_name, profile_data) or (None, None) on error.
    """
    try:
        content = file_path.read_text(encoding='utf-8')

        if file_path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                logger.warning(
                    "PyYAML not installed, skipping YAML profile: %s",
                    file_path
                )
                return None, None
        elif file_path.suffix == '.json':
            data = json.loads(content)
        else:
            logger.debug("Skipping non-profile file: %s", file_path)
            return None, None

        if not isinstance(data, dict):
            logger.warning("Profile file must contain a dict: %s", file_path)
            return None, None

        # Profile name is either explicit 'name' field or derived from filename
        name = data.get('name') or file_path.stem

        return name, data

    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in profile file %s: %s", file_path, e)
        return None, None
    except Exception as e:
        logger.warning("Error reading profile file %s: %s", file_path, e)
        return None, None


def discover_profiles(
    profiles_dir: str,
    base_path: Optional[str] = None
) -> Dict[str, 'SubagentProfile']:
    """Discover subagent profiles from a directory.

    Scans the specified directory for .json and .yaml/.yml files,
    parsing each as a subagent profile definition.

    Args:
        profiles_dir: Directory path to scan (relative or absolute).
        base_path: Base path for resolving relative profiles_dir.
                   Defaults to current working directory.

    Returns:
        Dict mapping profile names to SubagentProfile instances.
    """
    if base_path is None:
        base_path = os.getcwd()

    # Resolve the profiles directory path
    profiles_path = Path(profiles_dir)
    if not profiles_path.is_absolute():
        profiles_path = Path(base_path) / profiles_path

    if not profiles_path.exists():
        logger.debug("Profiles directory does not exist: %s", profiles_path)
        return {}

    if not profiles_path.is_dir():
        logger.warning("Profiles path is not a directory: %s", profiles_path)
        return {}

    profiles: Dict[str, SubagentProfile] = {}

    # Scan for profile files
    for file_path in profiles_path.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix not in ('.json', '.yaml', '.yml'):
            continue

        name, data = _parse_profile_file(file_path)
        if name is None or data is None:
            continue

        # Parse GC configuration if present
        gc_config = None
        if 'gc' in data and data['gc']:
            gc_config = GCProfileConfig.from_dict(data['gc'])

        # Create SubagentProfile from parsed data
        profile = SubagentProfile(
            name=name,
            description=data.get('description', ''),
            plugins=data.get('plugins', []),
            plugin_configs=data.get('plugin_configs', {}),
            system_instructions=data.get('system_instructions'),
            model=data.get('model'),
            provider=data.get('provider'),
            max_turns=data.get('max_turns', 10),
            auto_approved=data.get('auto_approved', False),
            icon=data.get('icon'),
            icon_name=data.get('icon_name'),
            gc=gc_config,
        )

        profiles[name] = profile
        logger.debug("Discovered profile '%s' from %s", name, file_path)

    if profiles:
        logger.info(
            "Discovered %d profile(s) from %s: %s",
            len(profiles),
            profiles_path,
            ", ".join(profiles.keys())
        )

    return profiles


def validate_profile(data: Any) -> Tuple[bool, List[str], List[str]]:
    """Validate a subagent profile JSON structure.

    Checks required fields, type constraints, and GC sub-configuration
    for a single profile definition (the format stored in .jaato/profiles/*.json).

    Args:
        data: Parsed JSON data from a profile file.

    Returns:
        Tuple of (is_valid, errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(data, dict):
        return False, ["File must contain a JSON object"], []

    # Required fields
    if not data.get("name"):
        errors.append("'name' is required")
    if not data.get("description"):
        errors.append("'description' is required")

    # plugins: list of strings
    plugins = data.get("plugins")
    if plugins is not None:
        if not isinstance(plugins, list):
            errors.append("'plugins' must be an array")
        elif not all(isinstance(p, str) for p in plugins):
            errors.append("'plugins' must contain only strings")

    # plugin_configs: dict of dicts
    plugin_configs = data.get("plugin_configs")
    if plugin_configs is not None:
        if not isinstance(plugin_configs, dict):
            errors.append("'plugin_configs' must be an object")
        else:
            for key, val in plugin_configs.items():
                if not isinstance(val, dict):
                    errors.append(f"plugin_configs['{key}'] must be an object")

    # max_turns: positive int
    max_turns = data.get("max_turns")
    if max_turns is not None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            errors.append("'max_turns' must be an integer")
        elif max_turns <= 0:
            errors.append("'max_turns' must be a positive integer")

    # auto_approved: bool
    auto_approved = data.get("auto_approved")
    if auto_approved is not None:
        if not isinstance(auto_approved, bool):
            errors.append("'auto_approved' must be a boolean")

    # model: string or null
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        errors.append("'model' must be a string or null")

    # provider: string or null
    provider = data.get("provider")
    if provider is not None and not isinstance(provider, str):
        errors.append("'provider' must be a string or null")

    # icon: list of exactly 3 strings or null
    icon = data.get("icon")
    if icon is not None:
        if not isinstance(icon, list):
            errors.append("'icon' must be an array of 3 strings or null")
        elif len(icon) != 3:
            errors.append(f"'icon' must contain exactly 3 strings, got {len(icon)}")
        elif not all(isinstance(line, str) for line in icon):
            errors.append("'icon' must contain only strings")

    # icon_name: string or null
    icon_name = data.get("icon_name")
    if icon_name is not None and not isinstance(icon_name, str):
        errors.append("'icon_name' must be a string or null")

    # GC sub-validation
    gc_data = data.get("gc")
    if gc_data is not None:
        if not isinstance(gc_data, dict):
            errors.append("'gc' must be an object or null")
        else:
            valid_gc_types = ("truncate", "summarize", "hybrid", "budget")
            gc_type = gc_data.get("type", "truncate")
            if gc_type not in valid_gc_types:
                errors.append(
                    f"gc.type '{gc_type}' is invalid. "
                    f"Must be one of: {', '.join(valid_gc_types)}"
                )

            # Numeric range checks
            for field_name in ("threshold_percent", "target_percent", "pressure_percent"):
                val = gc_data.get(field_name)
                if val is not None:
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        errors.append(f"gc.{field_name} must be a number")
                    elif val < 0 or val > 100:
                        errors.append(f"gc.{field_name} must be between 0 and 100")

            gc_preserve = gc_data.get("preserve_recent_turns")
            if gc_preserve is not None:
                if not isinstance(gc_preserve, int) or isinstance(gc_preserve, bool):
                    errors.append("gc.preserve_recent_turns must be an integer")
                elif gc_preserve < 0:
                    errors.append("gc.preserve_recent_turns must be non-negative")

            gc_max_turns = gc_data.get("max_turns")
            if gc_max_turns is not None:
                if not isinstance(gc_max_turns, int) or isinstance(gc_max_turns, bool):
                    errors.append("gc.max_turns must be an integer")
                elif gc_max_turns <= 0:
                    errors.append("gc.max_turns must be a positive integer")

    return len(errors) == 0, errors, warnings


@dataclass
class SubagentConfig:
    """Top-level configuration for the subagent plugin.

    Attributes:
        project: GCP project ID for Vertex AI.
        location: Vertex AI region (e.g., 'us-central1').
        default_model: Default model for subagents. None = inherit from parent.
        default_provider: Default provider for subagents. None = inherit from parent.
                         If set, MUST match default_model's provider.
        profiles: Dict of named subagent profiles.
        allow_inline: Whether to allow inline subagent creation.
        inline_allowed_plugins: Plugins allowed for inline subagent creation.
        auto_discover_profiles: Whether to auto-discover profiles from profiles_dir.
        profiles_dir: Directory to scan for profile files (default: .jaato/profiles).
    """
    project: str
    location: str
    default_model: Optional[str] = None  # None = inherit from parent
    default_provider: Optional[str] = None  # None = inherit from parent
    profiles: Dict[str, SubagentProfile] = field(default_factory=dict)
    allow_inline: bool = True
    inline_allowed_plugins: List[str] = field(default_factory=list)
    auto_discover_profiles: bool = True
    profiles_dir: str = ".jaato/profiles"

    def add_profile(self, profile: SubagentProfile) -> None:
        """Add a subagent profile."""
        self.profiles[profile.name] = profile

    def get_profile(self, name: str) -> Optional[SubagentProfile]:
        """Get a subagent profile by name."""
        return self.profiles.get(name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubagentConfig':
        """Create SubagentConfig from a dictionary.

        Args:
            data: Configuration dictionary with structure:
                {
                    "project": "...",
                    "location": "...",
                    "default_model": "...",
                    "profiles": {
                        "profile_name": {
                            "description": "...",
                            "plugins": [...],
                            ...
                        }
                    },
                    "allow_inline": true,
                    "inline_allowed_plugins": [...],
                    "auto_discover_profiles": true,
                    "profiles_dir": ".jaato/profiles"
                }

        Returns:
            SubagentConfig instance.
        """
        profiles = {}
        for name, profile_data in data.get('profiles', {}).items():
            # Parse GC configuration if present
            gc_config = None
            if 'gc' in profile_data and profile_data['gc']:
                gc_config = GCProfileConfig.from_dict(profile_data['gc'])

            profiles[name] = SubagentProfile(
                name=name,
                description=profile_data.get('description', ''),
                plugins=profile_data.get('plugins', []),
                plugin_configs=profile_data.get('plugin_configs', {}),
                system_instructions=profile_data.get('system_instructions'),
                model=profile_data.get('model'),
                provider=profile_data.get('provider'),
                max_turns=profile_data.get('max_turns', 10),
                auto_approved=profile_data.get('auto_approved', False),
                icon=profile_data.get('icon'),
                icon_name=profile_data.get('icon_name'),
                gc=gc_config,
            )

        return cls(
            project=data.get('project', ''),
            location=data.get('location', ''),
            default_model=data.get('default_model'),  # None = inherit from parent
            default_provider=data.get('default_provider'),  # None = inherit from parent
            profiles=profiles,
            allow_inline=data.get('allow_inline', True),
            inline_allowed_plugins=data.get('inline_allowed_plugins', []),
            auto_discover_profiles=data.get('auto_discover_profiles', True),
            profiles_dir=data.get('profiles_dir', '.jaato/profiles'),
        )


def gc_profile_to_plugin_config(
    gc_profile: GCProfileConfig,
    agent_name: Optional[str] = None
) -> tuple:
    """Convert a GCProfileConfig to a (GCPlugin, GCConfig) tuple.

    This helper function takes a profile's GC configuration and creates
    the actual GC plugin and config objects needed by JaatoSession.

    Args:
        gc_profile: GCProfileConfig from a subagent profile.
        agent_name: Optional agent name for trace logging identification.

    Returns:
        Tuple of (GCPlugin, GCConfig) ready to pass to session.set_gc_plugin().

    Raises:
        ValueError: If the GC plugin type is not found.

    Example:
        if profile.gc:
            gc_plugin, gc_config = gc_profile_to_plugin_config(profile.gc, agent_id)
            session.set_gc_plugin(gc_plugin, gc_config)
    """
    from ..gc import load_gc_plugin, GCConfig

    gc_type = gc_profile.type
    gc_plugin_name = gc_type if gc_type.startswith('gc_') else f'gc_{gc_type}'

    # Build plugin init config
    gc_init_config = {
        'preserve_recent_turns': gc_profile.preserve_recent_turns,
        'notify_on_gc': gc_profile.notify_on_gc,
    }
    if agent_name:
        gc_init_config['agent_name'] = agent_name
    if gc_profile.summarize_middle_turns is not None:
        gc_init_config['summarize_middle_turns'] = gc_profile.summarize_middle_turns
    # Merge plugin-specific config
    gc_init_config.update(gc_profile.plugin_config)

    gc_plugin = load_gc_plugin(gc_plugin_name, gc_init_config)

    # Create GCConfig for the session
    gc_config = GCConfig(
        threshold_percent=gc_profile.threshold_percent,
        max_turns=gc_profile.max_turns,
        preserve_recent_turns=gc_profile.preserve_recent_turns,
        plugin_config=gc_profile.plugin_config,
    )

    return gc_plugin, gc_config


@dataclass
class SubagentResult:
    """Result from a subagent execution.

    Attributes:
        success: Whether the subagent completed successfully.
        response: The subagent's final response text.
        turns_used: Number of conversation turns used.
        error: Error message if success is False.
        token_usage: Token usage statistics if available.
        agent_id: ID of the subagent session (for multi-turn conversations).
        output_streamed: Whether output was streamed via UI hooks (prevents double-display).
    """
    success: bool
    response: str
    turns_used: int = 0
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    agent_id: Optional[str] = None
    output_streamed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool response.

        When output_streamed is True, the response text is omitted from the
        result since it was already displayed to the user via UI hooks.
        This prevents the model from echoing the response in its output.
        """
        result: Dict[str, Any] = {
            'success': self.success,
            'turns_used': self.turns_used,
        }
        # Only include response text if it wasn't already streamed to UI
        if self.output_streamed:
            result['response_note'] = 'Response was streamed to the user interface. Do not repeat it.'
        else:
            result['response'] = self.response
        if self.error:
            result['error'] = self.error
        if self.token_usage:
            result['token_usage'] = self.token_usage
        if self.agent_id:
            result['agent_id'] = self.agent_id
        return result
