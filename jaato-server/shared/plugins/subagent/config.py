"""Configuration models for subagent plugin."""

import importlib.metadata
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol, Tuple, Union
from typing import runtime_checkable

from shared.runtime_limits import RuntimeLimits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secret resolver protocol and registry
#
# Allows premium (or third-party) packages to register secret backends
# (HashiCorp Vault, AWS Secrets Manager, SOPS, OS keyring, etc.) via the
# ``jaato.premium`` → ``secret_resolvers`` entry point.
#
# Config values like ``vault://secret/myapp#db_password`` are resolved
# transparently during variable expansion.
# ---------------------------------------------------------------------------

# Regex for ``scheme://path`` or ``scheme://path#key``.
_SECRET_URI_RE = re.compile(
    r'^(?P<scheme>[a-z][a-z0-9_+-]*)://'  # scheme (lowercase, RFC-ish)
    r'(?P<path>[^#]+)'                     # path (everything up to optional #)
    r'(?:#(?P<key>.+))?$'                  # optional #key fragment
)


@runtime_checkable
class SecretResolver(Protocol):
    """Protocol for secret backend resolvers.

    Each resolver handles one or more URI schemes (e.g. ``vault``, ``awssm``,
    ``sops``, ``keyring``).  The framework discovers resolvers via the
    ``jaato.premium`` → ``secret_resolvers`` entry point and dispatches
    ``scheme://path#key`` references to the matching resolver.

    Implementations live in the premium package (or any third-party package
    that registers the entry point).  The core framework only defines this
    protocol and the dispatch logic.
    """

    @property
    def schemes(self) -> FrozenSet[str]:
        """URI schemes this resolver handles (e.g. ``frozenset({"vault"})``).

        Must be lowercase.  A resolver may handle multiple schemes — for
        example a "cloud" resolver might handle both ``awssm`` and ``gcpsm``.
        """
        ...

    def resolve(self, scheme: str, path: str, key: Optional[str] = None) -> str:
        """Resolve a secret reference to its plaintext value.

        Args:
            scheme: The URI scheme (e.g. ``"vault"``).
            path: The path portion of the URI (e.g. ``"secret/myapp"``).
            key: Optional key/field within the secret (from the ``#fragment``).

        Returns:
            The resolved secret value as a string.

        Raises:
            SecretResolutionError: If the secret cannot be resolved (not found,
                auth failure, backend unreachable, etc.).
        """
        ...


class SecretResolutionError(Exception):
    """Raised when a secret URI cannot be resolved.

    Attributes:
        uri: The original ``scheme://path#key`` string that failed.
        reason: Human-readable explanation of the failure.
    """

    def __init__(self, uri: str, reason: str) -> None:
        self.uri = uri
        self.reason = reason
        super().__init__(f"Failed to resolve secret '{uri}': {reason}")


# ---------------------------------------------------------------------------
# Resolver registry — populated lazily from entry points.
# ---------------------------------------------------------------------------

_resolvers: Optional[Dict[str, 'SecretResolver']] = None


def _discover_secret_resolvers() -> Dict[str, 'SecretResolver']:
    """Discover secret resolvers from ``jaato.premium`` entry points.

    Looks for the ``secret_resolvers`` entry point which must return
    an iterable of :class:`SecretResolver` instances.

    Results are cached for the process lifetime.

    Returns:
        Dict mapping URI scheme → resolver instance.
    """
    global _resolvers
    if _resolvers is not None:
        return _resolvers

    _resolvers = {}

    eps = importlib.metadata.entry_points()
    if sys.version_info >= (3, 12):
        matches = eps.select(group="jaato.premium", name="secret_resolvers")
    elif sys.version_info >= (3, 10):
        matches = [ep for ep in eps.select(group="jaato.premium")
                   if ep.name == "secret_resolvers"]
    else:
        matches = [ep for ep in eps.get("jaato.premium", [])
                   if ep.name == "secret_resolvers"]

    for ep in matches:
        try:
            provider_fn = ep.load()
            resolvers = provider_fn()
            for resolver in resolvers:
                for scheme in resolver.schemes:
                    if scheme in _resolvers:
                        logger.warning(
                            "Duplicate secret resolver for scheme '%s' — "
                            "keeping first registered",
                            scheme,
                        )
                        continue
                    _resolvers[scheme] = resolver
                    logger.debug("Registered secret resolver: %s://", scheme)
        except Exception:
            logger.warning(
                "Failed to load secret_resolvers entry point",
                exc_info=True,
            )

    if _resolvers:
        logger.info(
            "Secret resolvers available for schemes: %s",
            ", ".join(sorted(_resolvers.keys())),
        )

    return _resolvers


def _resolve_secret_uri(value: str) -> str:
    """If *value* is a ``scheme://path[#key]`` URI with a registered resolver, resolve it.

    Returns the original string unchanged if:
    - It doesn't match the URI pattern.
    - No resolver is registered for the scheme.

    Raises:
        SecretResolutionError: Propagated from the resolver on failure.
    """
    m = _SECRET_URI_RE.match(value)
    if not m:
        return value

    scheme = m.group('scheme')
    resolvers = _discover_secret_resolvers()
    resolver = resolvers.get(scheme)
    if resolver is None:
        logger.warning(
            "Secret URI '%s' uses scheme '%s' but no resolver is registered "
            "for it (available: %s). The literal URI will be used as the value, "
            "which is almost certainly wrong.",
            value, scheme, list(resolvers.keys()) or "none",
        )
        return value

    path = m.group('path')
    key = m.group('key')  # May be None

    try:
        return resolver.resolve(scheme, path, key)
    except SecretResolutionError:
        raise
    except Exception as exc:
        raise SecretResolutionError(value, str(exc)) from exc


def reset_secret_resolvers() -> None:
    """Reset the cached secret resolvers (for testing)."""
    global _resolvers
    _resolvers = None


def parse_plugin_entry(entry: str) -> Tuple[str, bool]:
    """Parse a plugin entry that may have a ``(preload)`` suffix.

    Plugin names in profile ``plugins`` lists can include a ``(preload)``
    annotation to force all of the plugin's tools (including discoverable
    ones) to be loaded into the initial context rather than deferred.

    An optional space before the parenthesised annotation is accepted so
    that both ``"template(preload)"`` and ``"template (preload)"`` work.

    Args:
        entry: Plugin entry string, e.g. ``"template"``, ``"template(preload)"``,
            or ``"template (preload)"``.

    Returns:
        Tuple of (plugin_name, is_preloaded).

    Examples:
        >>> parse_plugin_entry("template(preload)")
        ('template', True)
        >>> parse_plugin_entry("template (preload)")
        ('template', True)
        >>> parse_plugin_entry("cli")
        ('cli', False)
    """
    match = re.match(r'^(\w+)\s*\(preload\)$', entry)
    if match:
        return match.group(1), True
    return entry, False


def parse_plugin_list(entries: List[str]) -> Tuple[List[str], set]:
    """Parse a list of plugin entries, separating names from preload annotations.

    Args:
        entries: List of plugin entry strings, possibly with ``(preload)`` suffixes.

    Returns:
        Tuple of (clean_plugin_names, preloaded_plugin_names_set).
    """
    clean_names: List[str] = []
    preloaded: set = set()
    for entry in entries:
        name, is_preloaded = parse_plugin_entry(entry)
        clean_names.append(name)
        if is_preloaded:
            preloaded.add(name)
    return clean_names, preloaded


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
    effective_cwd = workspace_root_override or os.environ.get('JAATO_WORKSPACE_ROOT') or os.getcwd()
    default_context = {
        'cwd': effective_cwd,
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
    """Expand ``${variable}`` references and secret URIs in a string.

    Two-phase expansion:

    1. **Variable substitution** — ``${VAR}`` patterns are replaced in
       this order: *context* (caller-provided), then the session-scoped
       env (``get_session_env``, populated from the session's
       ``env_file``), then the daemon's ``os.environ``.  Undefined
       variables are kept as-is (``${UNKNOWN}`` stays literal).

       The session-env tier exists so that plugins reading expanded
       config values (e.g. service_connector resolving
       ``base_url: http://127.0.0.1:${SERVICE_PORT}``) see the variables
       defined in the session's ``.env`` file even when the daemon was
       started without those vars in its process env.

    2. **Secret URI resolution** — if the *fully expanded* string matches
       ``scheme://path[#key]`` and a :class:`SecretResolver` is registered
       for that scheme, the value is resolved to its plaintext secret.
       This phase is a no-op when no premium resolvers are installed.

    Secret resolution only applies when the **entire** string is a URI
    (e.g. a config value ``vault://secret/myapp#db_password``).  URIs
    embedded in a larger string are not resolved — use ``${VAR}``
    indirection for those cases.

    Args:
        s: String containing ``${variable}`` references or a secret URI.
        context: Dict of variable names to values.

    Returns:
        String with variables expanded and secrets resolved.

    Raises:
        SecretResolutionError: If a secret URI is recognised but the
            resolver fails (auth error, not found, backend unreachable).
    """
    # Imported lazily to avoid a circular import — session_context lives
    # at the package root and importing it at module load time would
    # pull in the JaatoSession TYPE_CHECKING ladder.
    from shared.session_context import get_session_env

    # Phase 1: ${VAR} expansion
    if '${' in s:
        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            # First check context, then session env, then process env.
            if var_name in context:
                return context[var_name]
            session_val = get_session_env(var_name)
            if session_val is not None:
                return session_val
            return os.environ.get(var_name, match.group(0))  # Keep original if not found

        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        s = re.sub(pattern, replace_var, s)

    # Phase 2: secret URI resolution (entire-string match only)
    return _resolve_secret_uri(s)


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
        workspace = os.environ.get('JAATO_WORKSPACE_ROOT') or os.getcwd()
        p = Path(workspace) / p
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


def detect_workspace_tech_stack(workspace_path: str) -> str:
    """Detect the primary technology stack of a workspace by scanning for marker files.

    Checks for common build/config files at the workspace root to determine
    the project's language and build system.

    Args:
        workspace_path: Absolute path to the workspace root.

    Returns:
        Concise summary string (e.g., "Java project (Maven - pom.xml detected)")
        or empty string if nothing detected.
    """
    root = Path(workspace_path)
    detections = []

    # Check marker files in priority order
    markers = [
        ("pom.xml", "Java project (Maven - pom.xml detected)"),
        ("build.gradle", "Java/Kotlin project (Gradle - build.gradle detected)"),
        ("build.gradle.kts", "Kotlin project (Gradle KTS - build.gradle.kts detected)"),
        ("Cargo.toml", "Rust project (Cargo - Cargo.toml detected)"),
        ("go.mod", "Go project (go.mod detected)"),
        ("package.json", "JavaScript/TypeScript project (Node.js - package.json detected)"),
        ("pyproject.toml", "Python project (pyproject.toml detected)"),
        ("setup.py", "Python project (setup.py detected)"),
        ("requirements.txt", "Python project (requirements.txt detected)"),
        ("Gemfile", "Ruby project (Gemfile detected)"),
        ("composer.json", "PHP project (Composer - composer.json detected)"),
    ]

    for filename, description in markers:
        if (root / filename).exists():
            detections.append(description)

    # Check glob patterns for .NET
    if any(root.glob("*.sln")) or any(root.glob("*.csproj")):
        detections.append("C#/.NET project (.sln/.csproj detected)")

    if not detections:
        return ""

    return "; ".join(detections)


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
class CompletionArtifact:
    """A profile-declared file rendered from the agent's signal_completion payload.

    Output-side counterpart to dynamic-instructions prefetch scripts.
    The framework executes the renderer after ``signal_completion``'s
    payload validates against ``completion_payload_schema``; the agent
    never calls ``writeNewFile`` itself for these files — same body-
    wired pattern as input-side prefetch (the model produces the
    structured data, the body deterministically projects it onto disk).

    Attributes:
        renderer: Script path resolved through the standard
            ``script_loader`` tier (workspace ``.jaato/<path>`` →
            user ``~/.jaato/<path>``).  Script must define
            ``def render(payload: dict, context) -> str | bytes``.
        output: Output file path, with simple ``{field}`` templating.
            Substitutes from the payload first, then ``agent_params``,
            then a small set of session-derived values (``case_id``,
            ``agent_id``, ``workspace_path``).  Relative paths resolve
            under the session's ``workspace_path``.
        on_error: How a render failure (script raised, file write
            failed) is surfaced.  ``"fail_completion"`` returns a
            validation_failed-style error to the model so it retries;
            ``"warn"`` logs and continues, the completion still
            succeeds and the missing artifact is the operator's
            problem.  Default ``"fail_completion"``.
    """
    renderer: str
    output: str
    on_error: str = "fail_completion"


@dataclass
class SubagentProfile:
    """Configuration profile for a subagent.

    Defines what tools and capabilities a subagent has access to,
    allowing the parent model to delegate specialized tasks.

    Attributes:
        name: Unique identifier for this subagent profile.
        description: Human-readable description of what this subagent does.
        plugins: List of plugin names to enable for this subagent (clean names,
            ``(preload)`` suffixes stripped during parsing).
        preloaded_plugins: Set of plugin names that should bypass deferred tool
            loading — all their tools (including discoverable) are loaded into
            the initial context. Derived from ``(preload)`` annotations in the
            raw ``plugins`` list during profile parsing.
        plugin_configs: Per-plugin configuration overrides.
        system_instructions: **Deprecated.** Use agents (``.jaato/agents/``) instead.
            When an agent is specified via ``--agent``, its rendered markdown
            replaces this field.  Profiles should contain runtime config only.
        model: Optional model override (uses parent's model if not specified).
        provider: Optional provider override (e.g., 'anthropic', 'google_genai').
                  Allows subagents to use a different provider than the parent.
        max_turns: Maximum conversation turns before returning (default: 10).
        gc: Optional garbage collection configuration for this subagent.
        env: Session-scoped environment variables for this profile.
            Values support ``${VAR}`` expansion and secret URI resolution
            (e.g. ``vault://secret/myapp#db_password``).  For main sessions
            these are merged into ``JaatoServer._session_env``; for subagents
            they are applied to ``os.environ`` for the duration of the
            subagent thread and restored on exit.  Never leaks to other
            sessions or agents.
        inherits: Optional list of parent profile names. When set, this
            profile inherits fields from its parents. Resolved during
            ``discover_profiles()`` — after resolution, ``inherits`` is
            cleared and the profile is fully flattened.
        completion_payload_schema: Optional JSON Schema constraining the
            ``payload`` argument of ``signal_completion``. Either an inline
            dict or a string path resolved via the
            ``.jaato/completion_schemas/`` tier (absolute → workspace →
            home). When set, ``signal_completion``'s parameters carry the
            schema so providers enforce it at sampling time and
            ``LifecycleTools`` validates the payload server-side before
            emitting ``AgentCompletedEvent``. When ``None``, the legacy
            ``summary: str`` parameter is used. Inheritance follows the
            scalar-override rule (parents must agree or child overrides).
        runtime_limits: Optional per-session resource consumption caps —
            memory, PIDs, CPU weight, tool wall-clock timeout, and stdout
            cap.  Orthogonal to sandboxing (AppArmor): answers "how much
            can this session consume?" rather than "what can it touch?".
            The kernel-enforceable subset (memory, PIDs, CPU weight) is
            applied via cgroup v2 by ``server.cgroups.CgroupsManager``;
            the rest is read by the CLI/interactive_shell plugins at
            tool-call time.  ``None`` means "no limits" (host defaults).
    """
    name: str
    description: str
    plugins: List[str] = field(default_factory=list)
    preloaded_plugins: set = field(default_factory=set)
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_instructions: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    max_turns: int = 10
    gc: Optional[GCProfileConfig] = None
    env: Dict[str, str] = field(default_factory=dict)
    inherits: Optional[List[str]] = None
    completion_payload_schema: Optional[Union[str, Dict[str, Any]]] = None
    spawn_payload_schema: Optional[Union[str, Dict[str, Any]]] = None
    completion_artifacts: List[CompletionArtifact] = field(default_factory=list)
    runtime_limits: Optional[RuntimeLimits] = None
    # Per-turn model-tier config.  Empty dict means "single-model
    # mode" — the framework falls back to env vars (JAATO_TIER_*) at
    # session-init time, and from there to single-model behavior using
    # ``model``.  When non-empty, ``model`` is silently ignored (with a
    # warning at load time) because the active model is selected per
    # turn from ``model_tiers[<active_tier>]``.
    #
    # Single-level dict mixing tier→model entries (keys in
    # ``VALID_TIER_NAMES``) with reserved control keys (``initial`` /
    # ``fallback``).  Each tier entry is either a model-name string or
    # a dict with ``model`` (required) and ``provider`` (optional, V1
    # enforces same-provider across all tiers).  See
    # ``shared/model_tiers.py`` for the resolver and validation, and
    # ``project_backlog_per_turn_model`` for the full design.
    model_tiers: Dict[str, Any] = field(default_factory=dict)


def _normalize_inherits(value: Any) -> Optional[List[str]]:
    """Normalize the ``inherits`` field to a list of strings or None.

    Accepts a single string (``"readonly"``), a list of strings
    (``["readonly", "web_capable"]``), or None/absent.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return None


def _parse_completion_artifacts(value: Any) -> List[CompletionArtifact]:
    """Parse a profile's ``completion_artifacts`` list from raw JSON.

    Accepts a list of dicts each shaped like
    ``{"renderer": "scripts/foo.py", "output": "out/{case_id}/foo",
    "on_error": "fail_completion"}``.  Skips malformed entries with a
    warning rather than raising — partial profiles are loadable and
    the missing artifact surfaces at completion time as a normal
    "renderer not found" error.

    Returns an empty list when ``value`` is ``None``, missing, or not
    a list.
    """
    if not isinstance(value, list):
        return []
    out: List[CompletionArtifact] = []
    for entry in value:
        if not isinstance(entry, dict):
            logger.warning(
                "completion_artifacts: skipping non-dict entry: %r", entry,
            )
            continue
        renderer = entry.get("renderer")
        output = entry.get("output")
        if not isinstance(renderer, str) or not renderer.strip():
            logger.warning(
                "completion_artifacts: skipping entry without 'renderer': %r",
                entry,
            )
            continue
        if not isinstance(output, str) or not output.strip():
            logger.warning(
                "completion_artifacts: skipping entry without 'output': %r",
                entry,
            )
            continue
        on_error = entry.get("on_error", "fail_completion")
        if on_error not in ("fail_completion", "warn"):
            logger.warning(
                "completion_artifacts: invalid on_error=%r for renderer=%r, "
                "defaulting to 'fail_completion'",
                on_error, renderer,
            )
            on_error = "fail_completion"
        out.append(CompletionArtifact(
            renderer=renderer.strip(),
            output=output.strip(),
            on_error=on_error,
        ))
    return out


def build_inline_profile(
    data: Dict[str, Any],
    name: str = "<inline>",
    description: str = "Inline session spec",
) -> 'SubagentProfile':
    """Construct a ``SubagentProfile`` from a dict supplied by an SDK client.

    Mirrors the field set understood by ``_load_profiles_from_directory``
    so an inline spec on ``session.new`` accepts the same JSON shape as a
    profile file on disk. ``inherits`` is intentionally ignored — inline
    specs are atomic, not chained — and ``name`` / ``description`` default
    to safe placeholders since SDK clients aren't required to invent them.

    Args:
        data: The dict carried in ``CommandRequest.payload['spec']``.
            Recognized keys: ``model``, ``provider``, ``plugins``,
            ``plugin_configs``, ``system_instructions``, ``max_turns``,
            ``gc``, ``env``, ``completion_payload_schema``,
            ``runtime_limits``, ``model_tiers``.
        name: Display name for logs and traces. Default ``<inline>``.
        description: Human-readable description for the profile.

    Returns:
        A fully-formed ``SubagentProfile`` ready to hand to
        ``JaatoServer(profile=...)``.

    Raises:
        ValueError: If a structured sub-field (``gc``, ``runtime_limits``)
            fails to parse. Surfaced so the caller can emit a clear
            ``ErrorEvent`` rather than swallowing the failure.
    """
    gc_config = None
    if data.get('gc'):
        gc_config = GCProfileConfig.from_dict(data['gc'])

    runtime_limits = None
    if data.get('runtime_limits'):
        if not isinstance(data['runtime_limits'], dict):
            raise ValueError(
                "Invalid runtime_limits in inline spec: expected dict, "
                f"got {type(data['runtime_limits']).__name__}"
            )
        try:
            runtime_limits = RuntimeLimits.from_dict(data['runtime_limits'])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid runtime_limits in inline spec: {exc}")

    raw_plugins = data.get('plugins', [])
    clean_plugins, preloaded = parse_plugin_list(raw_plugins)

    raw_env = data.get('env', {})
    env = (
        {str(k): str(v) for k, v in raw_env.items()}
        if isinstance(raw_env, dict) else {}
    )

    raw_model_tiers = data.get('model_tiers') or {}
    model_tiers = (
        {str(k): v for k, v in raw_model_tiers.items()}
        if isinstance(raw_model_tiers, dict) else {}
    )

    return SubagentProfile(
        name=name,
        description=description,
        plugins=clean_plugins,
        preloaded_plugins=preloaded,
        plugin_configs=data.get('plugin_configs', {}),
        system_instructions=data.get('system_instructions'),
        model=data.get('model'),
        provider=data.get('provider'),
        max_turns=data.get('max_turns', 10),
        gc=gc_config,
        env=env,
        inherits=None,
        completion_payload_schema=data.get('completion_payload_schema'),
        spawn_payload_schema=data.get('spawn_payload_schema'),
        completion_artifacts=_parse_completion_artifacts(data.get('completion_artifacts')),
        runtime_limits=runtime_limits,
        model_tiers=model_tiers,
    )


def resolve_profiles(
    profiles: Dict[str, 'SubagentProfile'],
) -> Tuple[Dict[str, 'SubagentProfile'], Dict[str, str]]:
    """Resolve profile inheritance by merging parent fields into children.

    Performs a topological traversal of the inheritance graph. Each profile
    with an ``inherits`` field has its parents' fields merged in, following
    the merge semantics from ``docs/design/profile-inheritance.md``:

    - **Collection fields** (union): ``plugins``, ``preloaded_plugins``,
      ``env``, ``plugin_configs``
    - **Scalar fields** (agreement-or-override): ``model``, ``provider``,
      ``max_turns``, ``gc``
    - **Concatenation**: ``system_instructions`` (grandparent → parent → child)
    - **Never inherited**: ``name``, ``description``

    Conflicts between parents on scalar fields are hard errors unless the
    child explicitly overrides.  Cycles are detected and reported.

    After resolution, ``inherits`` is cleared on each profile (fully
    flattened).

    Args:
        profiles: Dict of profile name → SubagentProfile (unresolved).

    Returns:
        Tuple of (resolved_profiles, errors). ``errors`` maps profile
        names to error messages.  Profiles with errors are excluded from
        the resolved dict.
    """
    resolved: Dict[str, SubagentProfile] = {}
    errors: Dict[str, str] = {}
    resolving: set = set()  # cycle detection

    def _resolve(name: str) -> Optional[SubagentProfile]:
        """Recursively resolve a single profile."""
        if name in resolved:
            return resolved[name]
        if name in errors:
            return None
        if name in resolving:
            errors[name] = f"Inheritance cycle detected involving '{name}'"
            return None
        if name not in profiles:
            errors[name] = f"Profile '{name}' not found"
            return None

        profile = profiles[name]

        # No inheritance — resolve immediately
        if not profile.inherits:
            resolved[name] = profile
            return profile

        # Resolve parents first
        resolving.add(name)
        parent_profiles: List[SubagentProfile] = []
        for parent_name in profile.inherits:
            parent = _resolve(parent_name)
            if parent is None:
                errors[name] = (
                    f"Profile '{name}' inherits from '{parent_name}' "
                    f"which failed to resolve"
                )
                resolving.discard(name)
                return None
            parent_profiles.append(parent)
        resolving.discard(name)

        # Merge parents then apply child overrides
        merged = _merge_profiles(name, parent_profiles, profile, errors)
        if merged is None:
            return None

        resolved[name] = merged
        return merged

    # Resolve all profiles
    for name in list(profiles.keys()):
        _resolve(name)

    return resolved, errors


def _merge_profiles(
    child_name: str,
    parents: List['SubagentProfile'],
    child: 'SubagentProfile',
    errors: Dict[str, str],
) -> Optional['SubagentProfile']:
    """Merge parent profiles with a child profile.

    Args:
        child_name: Name of the child profile (for error messages).
        parents: Resolved parent profiles in declaration order.
        child: The child profile with its own overrides.
        errors: Error dict to populate on conflict.

    Returns:
        Merged SubagentProfile, or None if conflicts detected.
    """
    # --- Collection fields: union ---

    # plugins: parents first (in order), then child, deduplicated
    seen_plugins: set = set()
    merged_plugins: List[str] = []
    for parent in parents:
        for p in parent.plugins:
            if p not in seen_plugins:
                seen_plugins.add(p)
                merged_plugins.append(p)
    for p in child.plugins:
        if p not in seen_plugins:
            seen_plugins.add(p)
            merged_plugins.append(p)

    # preloaded_plugins: union
    merged_preloaded: set = set()
    for parent in parents:
        merged_preloaded |= parent.preloaded_plugins
    merged_preloaded |= child.preloaded_plugins

    # env: merge with conflict detection
    merged_env: Dict[str, str] = {}
    env_sources: Dict[str, str] = {}  # key → source profile name
    conflict_details: List[str] = []

    for parent in parents:
        for key, val in parent.env.items():
            if key in merged_env and merged_env[key] != val:
                # Conflict between parents
                if key not in child.env:
                    conflict_details.append(
                        f"  env['{key}']: '{env_sources[key]}' sets "
                        f"'{merged_env[key]}', '{parent.name}' sets '{val}'"
                    )
            else:
                merged_env[key] = val
                env_sources[key] = parent.name
    # Child overrides last
    merged_env.update(child.env)

    # plugin_configs: deep merge by plugin name
    merged_configs: Dict[str, Dict[str, Any]] = {}
    config_sources: Dict[str, Dict[str, str]] = {}  # plugin → key → source

    for parent in parents:
        for plugin_name, config in parent.plugin_configs.items():
            if plugin_name not in merged_configs:
                merged_configs[plugin_name] = {}
                config_sources[plugin_name] = {}
            for key, val in config.items():
                existing = merged_configs[plugin_name].get(key)
                if existing is not None and existing != val:
                    src = config_sources[plugin_name].get(key, "?")
                    # Only conflict if child doesn't override this specific key
                    child_plugin_config = child.plugin_configs.get(plugin_name, {})
                    if key not in child_plugin_config:
                        conflict_details.append(
                            f"  plugin_configs['{plugin_name}']['{key}']: "
                            f"'{src}' sets {existing!r}, '{parent.name}' sets {val!r}"
                        )
                else:
                    merged_configs[plugin_name][key] = val
                    config_sources[plugin_name][key] = parent.name
    # Child overrides last (per-plugin, per-key)
    for plugin_name, config in child.plugin_configs.items():
        if plugin_name not in merged_configs:
            merged_configs[plugin_name] = {}
        merged_configs[plugin_name].update(config)

    # --- Scalar fields: agreement-or-override ---

    # Collect parent values for scalar fields
    scalar_conflicts: List[str] = []

    def _resolve_scalar(field_name: str, child_val, default=None):
        """Resolve a scalar field across parents + child."""
        parent_vals = {}
        for parent in parents:
            val = getattr(parent, field_name)
            if val is not None and val != default:
                parent_vals[parent.name] = val

        # Child explicitly overrides
        if child_val is not None and child_val != default:
            return child_val

        # No parents set it
        if not parent_vals:
            return child_val  # keep child default

        # All parents agree
        unique_vals = set(str(v) for v in parent_vals.values())
        if len(unique_vals) == 1:
            return next(iter(parent_vals.values()))

        # Conflict
        details = ", ".join(
            f"'{name}': {val!r}" for name, val in parent_vals.items()
        )
        scalar_conflicts.append(
            f"  {field_name}: {details}"
        )
        return child_val

    merged_model = _resolve_scalar('model', child.model)
    merged_provider = _resolve_scalar('provider', child.provider)

    # max_turns: most restrictive (minimum) across parents, child can override
    parent_max_turns = [p.max_turns for p in parents if p.max_turns != 10]
    if child.max_turns != 10:
        merged_max_turns = child.max_turns
    elif parent_max_turns:
        merged_max_turns = min(parent_max_turns)
    else:
        merged_max_turns = 10

    # gc: agreement-or-override (compare as dicts for equality)
    merged_gc = _resolve_scalar('gc', child.gc)

    # runtime_limits: scalar-override (parents must agree or child
    # overrides).  Compared via str() inside _resolve_scalar — frozen
    # dataclasses with the same field values produce identical reprs,
    # so two parents declaring the same limits don't conflict.
    merged_runtime_limits = _resolve_scalar('runtime_limits', child.runtime_limits)

    # completion_payload_schema: scalar-override (parents must agree or
    # child overrides). Inline dicts and string paths both compared as-is
    # via str() in _resolve_scalar.
    merged_completion_schema = _resolve_scalar(
        'completion_payload_schema', child.completion_payload_schema
    )

    # spawn_payload_schema: same scalar-override semantics — symmetric
    # to completion_payload_schema but constrains the agent_params dict
    # passed to spawn_subagent at the spawn boundary.
    merged_spawn_schema = _resolve_scalar(
        'spawn_payload_schema', child.spawn_payload_schema
    )

    # completion_artifacts: concatenation across parent → child.  Each
    # entry is independent (different output paths, different
    # renderers); concatenating preserves both parent's and child's
    # declarations without conflict semantics.  Child entries appear
    # last so they take precedence if any future logic compares by
    # output-path uniqueness.
    merged_completion_artifacts: List[CompletionArtifact] = []
    for parent in parents:
        merged_completion_artifacts.extend(parent.completion_artifacts)
    merged_completion_artifacts.extend(child.completion_artifacts)

    # --- Concatenation: system_instructions ---
    instruction_parts = []
    for parent in parents:
        if parent.system_instructions:
            instruction_parts.append(parent.system_instructions)
    if child.system_instructions:
        instruction_parts.append(child.system_instructions)
    merged_instructions = "\n\n".join(instruction_parts) if instruction_parts else None

    # --- Check for conflicts ---
    all_conflicts = conflict_details + scalar_conflicts
    if all_conflicts:
        conflict_msg = (
            f"Profile '{child_name}' inherits from "
            f"{[p.name for p in parents]}.\n"
            f"Conflicts (override in '{child_name}' to resolve):\n"
            + "\n".join(all_conflicts)
        )
        errors[child_name] = conflict_msg
        return None

    return SubagentProfile(
        name=child.name,
        description=child.description,
        plugins=merged_plugins,
        preloaded_plugins=merged_preloaded,
        plugin_configs=merged_configs,
        system_instructions=merged_instructions,
        model=merged_model,
        provider=merged_provider,
        max_turns=merged_max_turns,
        gc=merged_gc,
        env=merged_env,
        inherits=None,  # Fully resolved
        completion_payload_schema=merged_completion_schema,
        spawn_payload_schema=merged_spawn_schema,
        completion_artifacts=merged_completion_artifacts,
        runtime_limits=merged_runtime_limits,
    )


def _parse_profile_file(
    file_path: Path,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Parse a single profile file (JSON or YAML).

    Args:
        file_path: Path to the profile file.

    Returns:
        Tuple of ``(profile_name, profile_data, None)`` on success, or
        ``(None, None, error_message)`` on parse/read failure.
    """
    try:
        content = file_path.read_text(encoding='utf-8')

        if file_path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                msg = f"PyYAML not installed, cannot parse YAML profile: {file_path.name}"
                logger.warning(msg)
                return None, None, msg
        elif file_path.suffix == '.json':
            data = json.loads(content)
        else:
            logger.debug("Skipping non-profile file: %s", file_path)
            return None, None, None

        if not isinstance(data, dict):
            msg = f"Profile file must contain a JSON object: {file_path.name}"
            logger.warning(msg)
            return None, None, msg

        # Profile name is either explicit 'name' field or derived from filename
        name = data.get('name') or file_path.stem

        return name, data, None

    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {file_path.name}: {e}"
        logger.warning(msg)
        return None, None, msg
    except Exception as e:
        msg = f"Error reading {file_path.name}: {e}"
        logger.warning(msg)
        return None, None, msg


@dataclass
class ProfileDiscoveryResult:
    """Result of profile discovery, carrying both valid profiles and parse errors.

    Callers that only need the profiles dict can use ``result.profiles``.
    Callers that want to surface actionable diagnostics (e.g. when a
    requested profile is missing) should also inspect ``result.errors``
    — a mapping from the profile file stem (e.g. ``"github-resolver"``)
    to the human-readable error message.
    """

    profiles: Dict[str, 'SubagentProfile'] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


def _scan_profiles_dir(
    directory: Path,
    profiles: Dict[str, 'SubagentProfile'],
    errors: Dict[str, str],
) -> None:
    """Scan a directory for profile files and populate profiles/errors dicts.

    Existing entries in ``profiles`` are never overwritten, so earlier
    directories (higher precedence) win over later ones.

    Args:
        directory: Directory to scan for .json/.yaml/.yml profile files.
        profiles: Accumulator dict — discovered profiles are added here.
        errors: Accumulator dict — parse errors are added here.
    """
    if not directory.is_dir():
        return

    found = 0
    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix not in ('.json', '.yaml', '.yml'):
            continue

        name, data, error = _parse_profile_file(file_path)
        if error:
            if file_path.stem not in errors:
                errors[file_path.stem] = error
            continue
        if name is None or data is None:
            continue
        if name in profiles:
            continue  # higher-precedence source already registered this name

        gc_config = None
        if 'gc' in data and data['gc']:
            gc_config = GCProfileConfig.from_dict(data['gc'])

        runtime_limits = None
        if 'runtime_limits' in data and data['runtime_limits']:
            try:
                runtime_limits = RuntimeLimits.from_dict(data['runtime_limits'])
            except (ValueError, TypeError) as exc:
                err = f"Invalid runtime_limits in profile '{name}': {exc}"
                logger.warning(err)
                if name not in errors:
                    errors[name] = err
                continue

        raw_plugins = data.get('plugins', [])
        clean_plugins, preloaded = parse_plugin_list(raw_plugins)

        # Parse env: must be a flat dict of string→string
        raw_env = data.get('env', {})
        env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}

        raw_model_tiers = data.get('model_tiers') or {}
        model_tiers = (
            {str(k): v for k, v in raw_model_tiers.items()}
            if isinstance(raw_model_tiers, dict) else {}
        )
        if model_tiers and data.get('model'):
            logger.warning(
                "Profile '%s' declares both 'model' and 'model_tiers'; "
                "'model' will be ignored — the active model is selected "
                "per turn from 'model_tiers[<active_tier>]'.", name,
            )

        profiles[name] = SubagentProfile(
            name=name,
            description=data.get('description', ''),
            plugins=clean_plugins,
            preloaded_plugins=preloaded,
            plugin_configs=data.get('plugin_configs', {}),
            system_instructions=data.get('system_instructions'),
            model=data.get('model'),
            provider=data.get('provider'),
            max_turns=data.get('max_turns', 10),
            gc=gc_config,
            env=env,
            inherits=_normalize_inherits(data.get('inherits')),
            completion_payload_schema=data.get('completion_payload_schema'),
            spawn_payload_schema=data.get('spawn_payload_schema'),
        completion_artifacts=_parse_completion_artifacts(data.get('completion_artifacts')),
            runtime_limits=runtime_limits,
            model_tiers=model_tiers,
        )
        if data.get('system_instructions'):
            import warnings
            warnings.warn(
                f"Profile '{name}' has 'system_instructions' which is deprecated. "
                f"Move the prompt to .jaato/agents/{name}.md and remove "
                f"system_instructions from the profile.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning(
                "Profile '%s' has deprecated 'system_instructions'. "
                "Move to .jaato/agents/%s.md instead.",
                name, name,
            )
        found += 1
        logger.debug("Discovered profile '%s' from %s", name, file_path)

    if found:
        logger.info(
            "Discovered %d profile(s) from %s: %s",
            found, directory,
            ", ".join(n for n, p in profiles.items()),
        )


def discover_profiles(
    profiles_dir: str,
    base_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> ProfileDiscoveryResult:
    """Discover subagent profiles from multiple sources.

    Scans up to three locations for ``.json`` / ``.yaml`` / ``.yml``
    profile files, in decreasing order of precedence:

    1. **Workspace** —
       * if ``config_root`` is set: ``<config_root>/profiles/``;
       * else: ``{base_path}/{profiles_dir}`` (today's behavior).
    2. **User**      — ``~/.jaato/profiles/``
    3. **Premium**   — profiles registered via ``jaato.premium`` entry points

    When the same profile name appears in multiple sources, the
    higher-precedence source wins.

    Args:
        profiles_dir: Directory path to scan (relative or absolute).
            Ignored when ``config_root`` is set.
        base_path: Base path for resolving relative profiles_dir.
                   Defaults to current working directory.
        config_root: Optional override for the workspace tier.  When set,
            scans ``<config_root>/profiles/`` instead of
            ``{base_path}/{profiles_dir}``.  See
            :func:`shared.config_resolver.resolve_config_search_path`.

    Returns:
        ProfileDiscoveryResult with discovered profiles and any parse errors.
    """
    if base_path is None:
        base_path = os.environ.get('JAATO_WORKSPACE_ROOT') or os.getcwd()

    # When no explicit ``config_root`` is provided, fall back to the
    # ``JAATO_CONFIG_ROOT`` env var.  ``JaatoServer._in_workspace``
    # exports it for the duration of session-bound work, so plugins
    # whose ``initialize()`` runs inside that context — including the
    # subagent plugin's first call here — pick up the override even
    # though the registry's ``set_config_root`` broadcast hasn't fired
    # yet (broadcasts run AFTER plugin init).
    effective_config_root = config_root or os.environ.get('JAATO_CONFIG_ROOT')

    profiles: Dict[str, SubagentProfile] = {}
    errors: Dict[str, str] = {}

    # 1.a Workspace profile-set overlay (NEW, optional).
    #
    # When ``JAATO_PROFILE_SET`` is set and ``<config_root>/profile-sets/<set>/``
    # exists, scan it FIRST so its entries land in the profiles dict before
    # the regular profiles/ scan — ``_scan_profiles_dir`` skips already-present
    # names, so first-scanned wins.  Used by the model-set switcher (e.g.
    # ``--model-set dumb``) to override per-agent ``model`` / ``provider`` /
    # ``plugin_configs`` while inheriting everything else from the regular
    # ``profiles/`` tier (typically via ``inherits: [_base_<agent>]``).
    #
    # When the env var isn't set, this is a no-op and behaviour matches
    # the pre-existing single-dir scan.
    profile_set = os.environ.get('JAATO_PROFILE_SET')
    if profile_set and effective_config_root:
        set_path = (
            Path(effective_config_root).expanduser().resolve()
            / "profile-sets" / profile_set
        )
        _scan_profiles_dir(set_path, profiles, errors)

    # 1.b Workspace tier — config_root override takes precedence; fall
    #    back to <base_path>/<profiles_dir> when no override is in effect.
    if effective_config_root:
        profiles_path = Path(effective_config_root).expanduser().resolve() / "profiles"
    else:
        profiles_path = Path(profiles_dir)
        if not profiles_path.is_absolute():
            profiles_path = Path(base_path) / profiles_path
    _scan_profiles_dir(profiles_path, profiles, errors)

    # 2. User-level profiles from ~/.jaato/profiles/
    #    Workspace profiles take precedence.
    user_profiles_path = Path.home() / ".jaato" / "profiles"
    _scan_profiles_dir(user_profiles_path, profiles, errors)

    # 3. Premium entry-point profiles (if installed).
    #    Workspace and user profiles take precedence over premium ones.
    premium_profiles = _discover_premium_profiles()
    for name, profile in premium_profiles.items():
        if name not in profiles:
            profiles[name] = profile

    # Resolve inheritance after all sources are scanned
    resolved, inheritance_errors = resolve_profiles(profiles)
    errors.update(inheritance_errors)

    return ProfileDiscoveryResult(profiles=resolved, errors=errors)


def _discover_premium_profiles() -> Dict[str, 'SubagentProfile']:
    """Discover profiles provided by the ``jaato.premium`` → ``profiles`` entry point.

    Returns an empty dict if no premium package is installed.
    """
    try:
        from shared.jaato_runtime import _get_premium_content_path
    except ImportError:
        return {}

    premium_dir = _get_premium_content_path("profiles")
    if not premium_dir or not Path(premium_dir).is_dir():
        return {}

    # Re-use discover_profiles logic on the premium path, but pass
    # an absolute path to avoid workspace-relative resolution.
    profiles: Dict[str, SubagentProfile] = {}
    for file_path in Path(premium_dir).iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix not in ('.json', '.yaml', '.yml'):
            continue

        name, data, _error = _parse_profile_file(file_path)
        if name is None or data is None:
            continue

        gc_config = None
        if 'gc' in data and data['gc']:
            gc_config = GCProfileConfig.from_dict(data['gc'])

        runtime_limits = None
        if 'runtime_limits' in data and data['runtime_limits']:
            try:
                runtime_limits = RuntimeLimits.from_dict(data['runtime_limits'])
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping premium profile '%s': invalid runtime_limits: %s",
                    name, exc,
                )
                continue

        raw_plugins = data.get('plugins', [])
        clean_plugins, preloaded = parse_plugin_list(raw_plugins)

        raw_env = data.get('env', {})
        env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}

        raw_model_tiers = data.get('model_tiers') or {}
        model_tiers = (
            {str(k): v for k, v in raw_model_tiers.items()}
            if isinstance(raw_model_tiers, dict) else {}
        )
        if model_tiers and data.get('model'):
            logger.warning(
                "Premium profile '%s' declares both 'model' and 'model_tiers'; "
                "'model' will be ignored.", name,
            )

        profile = SubagentProfile(
            name=name,
            description=data.get('description', ''),
            plugins=clean_plugins,
            preloaded_plugins=preloaded,
            plugin_configs=data.get('plugin_configs', {}),
            system_instructions=data.get('system_instructions'),
            model=data.get('model'),
            provider=data.get('provider'),
            max_turns=data.get('max_turns', 10),
            gc=gc_config,
            env=env,
            inherits=_normalize_inherits(data.get('inherits')),
            completion_payload_schema=data.get('completion_payload_schema'),
            spawn_payload_schema=data.get('spawn_payload_schema'),
        completion_artifacts=_parse_completion_artifacts(data.get('completion_artifacts')),
            runtime_limits=runtime_limits,
            model_tiers=model_tiers,
        )
        profiles[name] = profile
        logger.debug("Discovered premium profile '%s' from %s", name, file_path)

    if profiles:
        logger.info(
            "Discovered %d premium profile(s): %s",
            len(profiles), ", ".join(profiles.keys())
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

    # model: string or null
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        errors.append("'model' must be a string or null")

    # provider: string or null
    provider = data.get("provider")
    if provider is not None and not isinstance(provider, str):
        errors.append("'provider' must be a string or null")

    # env: dict of string keys to string values, or null
    env_data = data.get("env")
    if env_data is not None:
        if not isinstance(env_data, dict):
            errors.append("'env' must be an object or null")
        else:
            for key, val in env_data.items():
                if not isinstance(key, str):
                    errors.append(f"env key {key!r} must be a string")
                if not isinstance(val, str):
                    errors.append(f"env['{key}'] must be a string")

    # inherits: string, list of strings, or null
    inherits = data.get("inherits")
    if inherits is not None:
        if isinstance(inherits, str):
            pass  # Single string is valid (normalized to list during parsing)
        elif isinstance(inherits, list):
            for item in inherits:
                if not isinstance(item, str):
                    errors.append("'inherits' must be a string or list of strings")
                    break
        else:
            errors.append("'inherits' must be a string or list of strings")

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

    # runtime_limits sub-validation: delegate to RuntimeLimits.from_dict
    # which raises ValueError for any out-of-range value (kept in one
    # place rather than duplicating the rules here).
    runtime_data = data.get("runtime_limits")
    if runtime_data is not None:
        if not isinstance(runtime_data, dict):
            errors.append("'runtime_limits' must be an object or null")
        else:
            try:
                RuntimeLimits.from_dict(runtime_data)
            except (ValueError, TypeError) as exc:
                errors.append(f"runtime_limits: {exc}")

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

            # Parse runtime_limits (cgroup-enforced + app-enforced caps).
            # Validation runs in __post_init__ — bad values raise here so
            # the inline config rejects them at load time, same as gc.
            runtime_limits = None
            if 'runtime_limits' in profile_data and profile_data['runtime_limits']:
                runtime_limits = RuntimeLimits.from_dict(profile_data['runtime_limits'])

            # Parse plugin entries, separating (preload) annotations
            raw_plugins = profile_data.get('plugins', [])
            clean_plugins, preloaded = parse_plugin_list(raw_plugins)

            raw_env = profile_data.get('env', {})
            env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}

            raw_model_tiers = profile_data.get('model_tiers') or {}
            model_tiers = (
                {str(k): v for k, v in raw_model_tiers.items()}
                if isinstance(raw_model_tiers, dict) else {}
            )
            if model_tiers and profile_data.get('model'):
                logger.warning(
                    "Inline profile '%s' declares both 'model' and "
                    "'model_tiers'; 'model' will be ignored.", name,
                )

            profiles[name] = SubagentProfile(
                name=name,
                description=profile_data.get('description', ''),
                plugins=clean_plugins,
                preloaded_plugins=preloaded,
                plugin_configs=profile_data.get('plugin_configs', {}),
                system_instructions=profile_data.get('system_instructions'),
                model=profile_data.get('model'),
                provider=profile_data.get('provider'),
                max_turns=profile_data.get('max_turns', 10),
                gc=gc_config,
                env=env,
                inherits=_normalize_inherits(profile_data.get('inherits')),
                completion_payload_schema=profile_data.get('completion_payload_schema'),
                runtime_limits=runtime_limits,
                model_tiers=model_tiers,
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
        target_percent=gc_profile.target_percent,
        pressure_percent=gc_profile.pressure_percent,
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
