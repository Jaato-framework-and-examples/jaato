"""JaatoRuntime - Shared environment for the jaato framework.

Provides shared resources that can be used across multiple sessions (main agent
and subagents). This separates the "environment" (connections, plugins, permissions)
from the "session" (conversation history, per-agent state).
"""

import importlib.metadata
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .app_identity import FRAMEWORK_IDENTITY, AppIdentity, resolve_app_identity
from .token_accounting import TokenLedger
from .instruction_token_cache import InstructionTokenCache
from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    DISCOVERABILITY_EAGER,
    DISCOVERABILITY_DEFERRED,
)
from .plugins.model_provider.base import ProviderConfig
from .plugins.model_provider import load_provider
from .plugins.telemetry import TelemetryPlugin, create_plugin as create_telemetry_plugin

if TYPE_CHECKING:
    from .plugins.registry import PluginRegistry
    from .plugins.permission import PermissionPlugin
    from .plugins.reliability import ReliabilityPlugin
    from .plugins.model_provider.base import ModelProviderPlugin
    from .model_tiers import ModelTierConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Framework prompt constants
#
# These are functional defaults necessary for correct agent behavior.
# The jaato-premium package can provide enhanced versions via the
# ``jaato.premium`` → ``prompt_provider`` entry point.
# ---------------------------------------------------------------------------

# Anti-fabrication + relentless completion — core safety behavior
_TASK_COMPLETION_INSTRUCTION = (
    "After each action, continue working until the request is truly fulfilled. "
    "Pause only for permissions or clarifications—never from uncertainty. "
    "CRITICAL: Never fabricate results. Every completion claim must be backed by "
    "an actual tool call result or subagent output in this conversation. If a step "
    "cannot be verified, mark it failed honestly—do not invent success. "
    "CRITICAL: Never invent constraints that were not stated by the user. Do not "
    "claim token budget limits, time constraints, complexity ceilings, or any other "
    "restriction unless the user or the system explicitly told you about it. If you "
    "have work remaining, continue doing it—do not stop and summarize what is left."
)

# Parallel tool execution guidance — basic efficiency
_PARALLEL_TOOL_GUIDANCE = (
    "Batch independent tool calls. Before issuing a call, ask whether the NEXT "
    "call depends on this one's result. If it does not, issue them together in a "
    "single response — reading several files, searching several patterns, "
    "fetching several URLs, or inspecting several directories are all one "
    "response, not one response each. Only serialise when a call genuinely needs "
    "an earlier result as input. Independent calls execute in parallel, so a "
    "batch of eight costs about what one costs; the same eight issued one at a "
    "time costs eight round trips of latency and eight turns of context."
)

# Pre-call narration — needed for the OPERATOR, who is otherwise blind.
#
# _TURN_SUMMARY_INSTRUCTION below arrives only at the END of a turn, and
# exists for GC (see its comment).  A capable model can make dozens or
# hundreds of calls inside one turn, so the person watching the tool tree
# sees a wall of calls with no stated reason for any of them until the turn
# closes — and if the turn dies mid-flight (provider error, hung tool) they
# never learn why any of it happened.  Reported from live use 2026-08-29.
#
# Deliberately cheap: one short line before a BATCH, not per call.  A
# per-call rule would fight _PARALLEL_TOOL_GUIDANCE above, which wants
# several calls in one response.
_TOOL_NARRATION_GUIDANCE = (
    "Before a tool call or a batch of them, state in one short sentence what "
    "you are about to do and why — what you expect to learn or change. The "
    "person watching sees your tool calls as they happen and has no other "
    "window into your reasoning; a turn that dies mid-way should still leave "
    "them knowing what you were doing. One line per batch, not per call, and "
    "skip it only when the reason is already obvious from what you just said."
)

# Turn-end summary guidance — needed for GC to work effectively
_TURN_SUMMARY_INSTRUCTION = (
    "After completing a complex turn involving multiple tool calls, provide a concise summary "
    "of what was done and why. This helps maintain context for future turns and enables "
    "efficient garbage collection of verbose intermediate outputs. Include: actions taken, "
    "goals accomplished, rationale for non-obvious decisions, and next steps if applicable. "
    "Skip summaries for simple single-tool lookups or direct conversational responses."
)


def _apply_premium_prompt_overrides() -> None:
    """Load premium prompt overrides if a ``jaato.premium`` prompt provider is installed.

    Looks for a ``prompt_provider`` entry point in the ``jaato.premium``
    group.  If found, calls it to get a dict of ``{constant_name: value}``
    and patches the module-level constants.

    Called once at module load time.
    """
    global _TASK_COMPLETION_INSTRUCTION, _PARALLEL_TOOL_GUIDANCE, _TURN_SUMMARY_INSTRUCTION
    global _TOOL_NARRATION_GUIDANCE

    eps = importlib.metadata.entry_points()
    if sys.version_info >= (3, 12):
        premium_eps = eps.select(group="jaato.premium", name="prompt_provider")
    elif sys.version_info >= (3, 10):
        premium_eps = eps.select(group="jaato.premium")
        premium_eps = [ep for ep in premium_eps if ep.name == "prompt_provider"]
    else:
        premium_eps = [
            ep for ep in eps.get("jaato.premium", [])
            if ep.name == "prompt_provider"
        ]

    for ep in premium_eps:
        try:
            provider_fn = ep.load()
            overrides = provider_fn()
            if not isinstance(overrides, dict):
                logger.warning("Premium prompt_provider returned %s, expected dict", type(overrides).__name__)
                continue
            if "task_completion" in overrides:
                _TASK_COMPLETION_INSTRUCTION = overrides["task_completion"]
            if "parallel_tool_guidance" in overrides:
                _PARALLEL_TOOL_GUIDANCE = overrides["parallel_tool_guidance"]
            if "turn_summary" in overrides:
                _TURN_SUMMARY_INSTRUCTION = overrides["turn_summary"]
            if "tool_narration" in overrides:
                _TOOL_NARRATION_GUIDANCE = overrides["tool_narration"]
            logger.debug("Premium prompt overrides applied: %s", list(overrides.keys()))
            return  # Only one provider
        except Exception:
            logger.warning("Failed to load premium prompt_provider", exc_info=True)


_apply_premium_prompt_overrides()


# Cache for premium content paths — resolved once per entry-point name.
_premium_content_cache: Dict[str, Optional[str]] = {}


def _get_premium_content_path(name: str) -> Optional[str]:
    """Return a filesystem path provided by a ``jaato.premium`` entry point.

    Premium content entry points (``instructions``, ``profiles``, etc.)
    return a directory path where the premium package stores its content
    files.  Results are cached for the lifetime of the process.

    Args:
        name: The entry-point name within the ``jaato.premium`` group
            (e.g. ``"instructions"``, ``"profiles"``).

    Returns:
        Absolute path string, or ``None`` if no provider is registered.
    """
    if name in _premium_content_cache:
        return _premium_content_cache[name]

    result = None
    eps = importlib.metadata.entry_points()
    if sys.version_info >= (3, 12):
        matches = eps.select(group="jaato.premium", name=name)
    elif sys.version_info >= (3, 10):
        matches = [ep for ep in eps.select(group="jaato.premium") if ep.name == name]
    else:
        matches = [ep for ep in eps.get("jaato.premium", []) if ep.name == name]

    for ep in matches:
        try:
            provider_fn = ep.load()
            result = provider_fn()
            break
        except Exception:
            logger.warning("Failed to load premium content path '%s'", name, exc_info=True)

    _premium_content_cache[name] = result
    return result


def _get_sandbox_guidance() -> Optional[str]:
    """Get sandbox guidance if workspace is configured.

    Returns sandbox awareness instructions if a workspace root is set,
    informing the model about path restrictions.
    """
    from shared.session_context import get_workspace_root
    workspace = get_workspace_root() or os.environ.get('workspaceRoot')  # env: workspace root hint (VS Code-style) fallback when session context has none
    if not workspace:
        return None

    return (
        f"SANDBOX ENVIRONMENT: You are operating in a sandboxed workspace. "
        f"File operations (read, write, glob, grep, cli) are restricted to: {workspace}\n"
        f"- Paths outside the workspace will be rejected\n"
        f"- Use relative paths or absolute paths within the workspace\n"
        f"- The .jaato/ directory may reference external configuration"
    )


#: How the common ``cache:`` profile field reaches each provider's own
#: knobs.  Three mechanisms, three spellings, two layers -- which is the
#: whole reason the common field exists (see
#: ``docs/design/model-tier-prompt-cache.md`` §7).
#:
#: ``layer`` is the sub-dict the provider reads the knob from, or ``None``
#: for a flat extra.  ``enabled`` / ``ttl`` / ``history`` name the
#: provider's key, or are absent when that provider's mechanism has no
#: such control (Google places no history breakpoint, so ``history`` is
#: meaningless there rather than false).
#:
#: Coverage is asserted rather than trusted: a provider declaring
#: ``prompt_caching=True`` with no entry here fails
#: ``test_cache_profile_field.py``, so a new caching provider cannot land
#: with the common field silently inert for it.
CACHE_FIELD_DELIVERY: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "layer": None,
        "enabled": "enable_caching", "ttl": "cache_ttl",
        "history": "cache_history",
    },
    "google_genai": {
        # CachedContent holds system+tools; there is no history breakpoint
        # to switch on, and its TTL is a Google duration string.
        "layer": None,
        "enabled": "enable_caching", "ttl": "cache_ttl",
        "ttl_format": "seconds",
    },
    "openrouter": {
        # Caches internally rather than via a cache plugin, and reads its
        # knobs from the api_params sub-dict.
        "layer": "api_params",
        "enabled": "cache_prompt", "ttl": "cache_ttl",
    },
}

#: ``cache.ttl`` in the profile vocabulary -> seconds, for providers whose
#: API takes a duration.
_TTL_SECONDS = {"5m": 300, "1h": 3600}


def cache_field_to_provider_extra(
    cache: Any,
    provider_name: Optional[str],
    *,
    supports_caching: bool = True,
) -> Dict[str, Any]:
    """Translate the common ``cache:`` field into one provider's knobs.

    Returns the extras the profile field implies, ready to be laid down
    BENEATH ``plugin_configs.<provider>`` so the specific knob wins.

    Three things produce an empty dict, all deliberately silent rather
    than an error:

    * no ``cache:`` block -- the field is optional;
    * a provider that cannot cache (``prompt_caching=False``) -- §7's
      "degrades to a no-op rather than an error", which is what makes
      ``auto`` well-defined;
    * a provider with no delivery entry -- it cannot cache either.

    ``enabled: "auto"`` emits nothing for the enable key: it means "leave
    the provider's own default alone", so writing a value would be the
    opposite of what it says.  OpenRouter is the exception, because
    ``cache_prompt: "auto"`` is a real value in its API rather than an
    absence.
    """
    if cache is None or not supports_caching:
        return {}
    spec = CACHE_FIELD_DELIVERY.get(provider_name or "")
    if not spec:
        return {}

    flat: Dict[str, Any] = {}
    enabled = getattr(cache, "enabled", "auto")
    if enabled != "auto":
        flat[spec["enabled"]] = bool(enabled)
    elif provider_name == "openrouter":
        flat[spec["enabled"]] = "auto"

    ttl = getattr(cache, "ttl", None)
    if ttl:
        flat[spec["ttl"]] = (
            f"{_TTL_SECONDS[ttl]}s" if spec.get("ttl_format") == "seconds"
            else ttl)

    if "history" in spec:
        flat[spec["history"]] = bool(getattr(cache, "history", True))

    layer = spec["layer"]
    return flat if layer is None else {layer: flat}


def resolve_provider_extra(
    base_extra: Dict[str, Any],
    plugin_configs: Optional[Dict[str, Dict[str, Any]]],
    provider_name: Optional[str],
    cache_extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """The ONE definition of a provider's effective config extras.

    Providers are plugins, so their profile knobs live under
    ``plugin_configs[<provider>]``.  This folds that section onto the
    runtime-level base, child-wins, and promotes ``api_key`` out of the
    result — the universal auth-field contract reads
    ``ProviderConfig.api_key``, not ``config.extra["api_key"]``.

    Two callers, and they MUST agree:

    * :meth:`JaatoRuntime.create_provider`, which builds the
      ``ProviderConfig`` the provider itself is initialized with; and
    * ``JaatoSession._cache_plugin_config``, which builds the config for
      the cache plugin attached to that same provider.

    They cannot share the RESULT, only this function.  ``plugin_configs``
    is a per-CALL argument — each session creates its own provider
    instance from its own profile — while ``_provider_configs`` is
    runtime-level and shared by every session on that provider.  Writing
    the merged config back there would leak one session's profile knobs
    (credentials included) into every other session using the same
    provider.  So the merge is necessarily recomputed per caller, and the
    only defence against the two callers drifting apart is that there is
    exactly one of them, here.

    Args:
        base_extra: The runtime-level ``ProviderConfig.extra`` to fold onto.
        plugin_configs: The session profile's per-plugin config dict, or
            ``None``.
        provider_name: Which section of ``plugin_configs`` to read — the
            name the provider is REGISTERED under.  Not interchangeable
            with ``provider.name``: zhipuai subclasses the Anthropic
            provider and reports the parent's name, and only the
            registration name selects the right section.

    Returns:
        ``(extra, promoted_api_key)``.  ``promoted_api_key`` is ``None``
        when the profile supplies none, in which case the caller leaves
        ``ProviderConfig.api_key`` alone.
    """
    extra = _layer_onto(dict(base_extra), cache_extra or {})
    overrides = (plugin_configs or {}).get(provider_name or "")
    if not overrides:
        return extra, None
    overrides = dict(overrides)
    promoted_api_key = overrides.pop("api_key", None)
    extra = _layer_onto(extra, overrides)
    return extra, promoted_api_key


def _layer_onto(base: Dict[str, Any], top: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``top`` onto ``base``, descending ONE level into sub-dicts.

    A flat ``update`` is wrong here.  The common ``cache:`` field delivers
    OpenRouter's knobs inside ``api_params``, and a profile that also sets
    ``api_params.temperature`` would have that whole sub-dict replaced by
    whichever layer landed last -- losing the temperature or losing the
    cache setting depending on order, silently either way.

    One level is enough and is where it stops deliberately: every provider
    layer (``api_params``, ``routing``, ``load``, ``framework_overrides``)
    is a flat sub-dict of scalars, so a deeper merge would have no
    behaviour to justify it and would start guessing at intent.
    """
    out = dict(base)
    for key, value in top.items():
        if (isinstance(value, dict)
                and isinstance(out.get(key), dict)):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


def _is_parallel_tools_enabled() -> bool:
    """Check if parallel tool execution is enabled."""
    return os.environ.get(  # env: run multiple tool calls per turn in a thread pool (default true; max 8 concurrent)
        'JAATO_PARALLEL_TOOLS', 'true'
    ).lower() not in ('false', '0', 'no')


def _framework_prompt_constants() -> List[str]:
    """The framework-level prompt constants, in wire order.

    One place that decides which constants are live, so a new constant is
    added by editing a list rather than by growing
    ``get_system_instructions`` — that function is over the cyclomatic
    ceiling and frozen in the complexity baseline, so each new ``if`` there
    costs a baseline bump.

    Each entry is skipped when empty, which is how a premium prompt
    provider disables one: override it with ``""``.  ``_PARALLEL_TOOL_GUIDANCE``
    additionally requires parallel execution to be enabled — advertising
    batching to a model that will have its calls serialised anyway is a
    promise the runtime would not keep.
    """
    live = [
        _TASK_COMPLETION_INSTRUCTION,
        _PARALLEL_TOOL_GUIDANCE if _is_parallel_tools_enabled() else "",
        _TOOL_NARRATION_GUIDANCE,
        _TURN_SUMMARY_INSTRUCTION,
    ]
    return [c for c in live if c]


def _is_deferred_tools_enabled() -> bool:
    """Check if deferred tool loading is enabled.

    When enabled, only 'core' tools are loaded into the initial model context.
    Other tools can be discovered via the introspection plugin (list_tools,
    get_tool_schemas). This reduces initial context size significantly.

    Default is 'true' for token economy. Set JAATO_DEFERRED_TOOLS=false
    to disable and load all tools upfront.
    """
    return os.environ.get(  # env: load only core tools upfront, others discovered on demand (default true); false loads all
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
            plugins=["cli", "web_search"],
            system_instructions="You are a research assistant."
        )
    """

    def __init__(self, provider_name: str = "google_genai",
                 workspace_path: Optional[Path] = None,
                 config_root: Optional[str] = None,
                 instruction_token_cache: Optional[InstructionTokenCache] = None,
                 app_identity: Optional[AppIdentity] = None):
        """Initialize JaatoRuntime.

        Args:
            provider_name: Name of the model provider to use (default: 'google_genai').
            workspace_path: Explicit workspace directory for loading instructions.
                When running as a daemon, the process cwd may differ from the
                client's workspace, so callers should pass the workspace path
                explicitly. Falls back to ``Path.cwd()`` when not provided.
            config_root: Optional override for the read-only framework-config
                search root.  When unset, the daemon scans
                ``<workspace_path>/.jaato/`` for profiles, agents, prompts,
                references, completion_schemas, instructions, scripts,
                services etc.  When set, that workspace-anchored search is
                replaced with this path.  The ``~/.jaato/`` user tier is
                always honored regardless.  See ``shared/config_resolver.py``.
            instruction_token_cache: Optional shared cache for instruction token
                counts.  When provided (e.g. from ``SessionManager``), cached
                counts survive across session creates/restores within the same
                daemon process.  When ``None``, a new per-runtime cache is
                created.
            app_identity: The APPLICATION this runtime speaks for — the
                product built on the SDK, which upstream services should see
                instead of the framework's own name (see
                ``shared/app_identity.py``).  This is the programmatic
                surface for an embedder; when ``None`` the identity is
                resolved from ``JAATO_APP_*`` at provider-creation time and
                falls back to jaato's own identity, so an unconfigured
                deployment behaves exactly as before.
        """
        self._provider_name: str = provider_name
        self._workspace_path: Optional[Path] = workspace_path
        self._config_root: Optional[str] = config_root
        self._provider_config: Optional[ProviderConfig] = None

        # Explicit application identity from the embedder, or ``None`` to
        # resolve from the environment per provider creation.  NOT resolved
        # eagerly here: the daemon overlays a session's ``env`` onto
        # ``os.environ`` for the duration of a turn, so an identity frozen at
        # construction would ignore a workspace that names its own app.
        self._app_identity: Optional[AppIdentity] = app_identity

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

        # AppArmor confine-context factory (server 0.6.50+).  Set by
        # ``JaatoServer`` from the WS pre-initialize hook so sessions
        # created on this runtime can wrap their dynamic-instructions
        # expansion (and any other configure-time work) in
        # ``apparmor_confine(profile)``.  ``None`` means no confinement
        # applies (IPC sessions, AppArmor unavailable).  See
        # ``shared/safe_pool.py`` for the per-thread reset and
        # ``server/apparmor.py`` for the context manager itself.
        self._confine_context_factory: Optional[Callable] = None

        # Formatter pipeline (optional, for collecting formatter instructions)
        self._formatter_pipeline: Optional[Any] = None

        # Base system instructions (loaded from .jaato/instructions/ or
        # legacy single file).  Loaded **lazily** on first request via
        # ``get_base_system_instructions`` so sessions that supply
        # ``system_instruction_override`` (replacing the assembled prompt
        # entirely) never pay the disk-I/O cost.  ``_base_loaded`` flips
        # to True after the first load attempt — distinguishes "not yet
        # loaded" from "loaded but no instruction files found" (where
        # ``_base_system_instructions`` legitimately stays ``None``).
        self._base_system_instructions: Optional[str] = None
        self._base_loaded: bool = False

        # Content-addressed token count cache (shared across sessions)
        self._instruction_token_cache: InstructionTokenCache = (
            instruction_token_cache or InstructionTokenCache()
        )

        # Connection state
        self._connected: bool = False

        # Telemetry plugin (created lazily, opt-in)
        self._telemetry: TelemetryPlugin = create_telemetry_plugin()

        # Per-runtime event bus for session-isolated event coordination.
        # Subagents within this runtime share the bus; different runtimes
        # (and thus different sessions) get separate bus instances.
        from shared.event_bus import EventBus
        self._event_bus: EventBus = EventBus()

        # Subscribe telemetry to bus for plan/step context propagation
        self._telemetry.subscribe_to_bus(self._event_bus)

    def get_base_system_instructions(self) -> Optional[str]:
        """Return the base system instructions, loading on first call.

        Lazy resolution: the actual disk read happens only when a session
        first asks for the assembled prompt.  Sessions that supply a
        ``system_instruction_override`` never call this, so they pay
        nothing for the premium/workspace/user instruction files they
        won't use.  Subsequent calls return the cached value (None when
        no instruction files were found).

        The runtime-wide cache is intentional: every session sharing the
        same runtime sees the same base layer (it's framework config,
        not per-session state), so loading once and sharing keeps memory
        flat across N sessions.
        """
        if not self._base_loaded:
            self._load_base_system_instructions()
            self._base_loaded = True
        return self._base_system_instructions

    def _load_base_system_instructions(self) -> None:
        """Load base system instructions from .jaato/instructions/ folders.

        Three tiers, loaded in order and concatenated:
        1. Premium tier — the ``jaato.premium`` ``instructions`` content
           path, when a premium package is installed.  Loaded FIRST and
           always additive (the baseline behavioral layer).
        2. Workspace tier — ``<config_root>/instructions/`` when
           ``config_root`` is set, else
           ``<workspace_path>/.jaato/instructions/``.
        3. User tier — ``~/.jaato/instructions/``.

        The workspace and user tiers are searched first-match-wins (the
        workspace tier is preferred); the premium tier is independent of
        that choice and is always prepended on top.

        Within each loaded folder, all ``*.md`` files are sorted by
        filename (so numeric prefixes like ``00-``, ``10-``, ``15-`` control
        ordering) and concatenated with double-newline separators
        (``README.md`` is skipped — see ``_load_instruction_files``).

        Falls back to the legacy single-file path
        ``.jaato/system_instructions.md`` only when no tier
        (premium/workspace/user) yielded any content.

        The combined contents are prepended to all agent system instructions,
        ensuring consistent behavior across main agent and all subagents.
        """
        # Use explicit workspace_path when provided (daemon mode), else cwd
        base = self._workspace_path or Path.cwd()

        all_parts: List[str] = []

        # 1. Premium instructions (loaded first — baseline behavioral layer)
        premium_dir = _get_premium_content_path("instructions")
        if premium_dir and Path(premium_dir).is_dir():
            premium_parts = self._load_instruction_files(Path(premium_dir))
            all_parts.extend(premium_parts)

        # 2. Workspace or user instructions (layered on top of premium).
        #    The workspace tier honors ``config_root`` when set so the
        #    daemon can load instructions from a path the agent's
        #    sandboxed filesystem can't reach.
        if self._config_root:
            workspace_instructions = (
                Path(self._config_root).expanduser().resolve() / "instructions"
            )
        else:
            workspace_instructions = base / ".jaato" / "instructions"
        search_dirs = [
            workspace_instructions,
            Path.home() / ".jaato" / "instructions",
        ]

        for instructions_dir in search_dirs:
            if instructions_dir.is_dir():
                parts = self._load_instruction_files(instructions_dir)
                if parts:
                    all_parts.extend(parts)
                    break  # First match wins for workspace/user layer

        if all_parts:
            self._base_system_instructions = "\n\n".join(all_parts)
            return

        # Fallback: legacy single-file path.  Honors config_root for
        # the workspace tier so an out-of-tree project layout still
        # resolves to its single-file instructions when the user
        # hasn't migrated to the multi-file folder layout yet.
        if self._config_root:
            legacy_workspace = (
                Path(self._config_root).expanduser().resolve()
                / "system_instructions.md"
            )
        else:
            legacy_workspace = base / ".jaato" / "system_instructions.md"
        legacy_paths = [
            legacy_workspace,
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
    def event_bus(self) -> 'EventBus':
        """Get the per-runtime event bus.

        Each runtime has its own EventBus instance, ensuring session
        isolation. Subagents within the same runtime share this bus.
        """
        return self._event_bus

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

    def set_confine_context_factory(
        self, factory: Optional[Callable],
    ) -> None:
        """Set the AppArmor confine-context factory (server 0.6.50+).

        Called by ``JaatoServer`` from the WS pre-initialize hook so
        sessions created on this runtime can wrap their dynamic-
        instructions expansion (and any other configure-time work) in
        ``apparmor_confine(profile)``.  The factory is a zero-argument
        callable returning a context manager — same shape as
        :func:`server.apparmor.make_confine_context`.

        Sessions read this in ``create_session`` and propagate it onto
        the new ``JaatoSession`` via :meth:`JaatoSession.set_confine_context_factory`
        so ``configure()`` can use it.

        Setting to ``None`` clears the factory (no confinement applies).

        Args:
            factory: Zero-arg callable returning a context manager, or
                ``None``.
        """
        self._confine_context_factory = factory

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
        # Subscribe to the event bus so plan/step lifecycle events
        # automatically tag spans with plan_id/step_id.
        plugin.subscribe_to_bus(self._event_bus)

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
        provider_name: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
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
            plugin_configs: Optional per-plugin profile config dict, identical
                in shape to ``SubagentProfile.plugin_configs``.  When the
                profile carries provider-specific knobs under
                ``plugin_configs[provider_name]`` (e.g. an LM Studio bearer
                token, a custom NIM ``base_url``), they're merged into the
                ``ProviderConfig.extra`` handed to ``provider.verify_auth``
                so credential resolution at verify time matches what
                ``initialize()`` will see later.  Without this, providers
                fall back to environment-only credential discovery and
                profile-supplied secrets are invisible at verify time.

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

        # Create a temporary provider instance just for auth verification.
        # We don't call initialize() yet — verify_auth is designed to work
        # before full initialization (no clients, no network).
        provider = load_provider(effective_provider, config=None)

        # Build a lightweight ProviderConfig that surfaces any profile
        # knobs the provider may need to resolve credentials (host,
        # api_token, base_url, etc.).  Providers that don't read config
        # ignore the kwarg.
        verify_config: Optional[ProviderConfig] = None
        provider_overrides = (plugin_configs or {}).get(effective_provider)
        if provider_overrides:
            verify_config = ProviderConfig(extra=dict(provider_overrides))

        return provider.verify_auth(
            allow_interactive=allow_interactive,
            on_message=on_message,
            config=verify_config,
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

        t_start = time.perf_counter()

        # Get tool schemas based on deferred loading setting
        t0 = time.perf_counter()
        if _is_deferred_tools_enabled():
            # Deferred loading: only core tools in initial context
            self._all_tool_schemas = self._registry.get_core_tool_schemas()
        else:
            # Traditional: all enabled tools in initial context
            self._all_tool_schemas = self._registry.get_enabled_tool_schemas()
        schema_ms = (time.perf_counter() - t0) * 1000

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

        total_ms = (time.perf_counter() - t_start) * 1000
        if total_ms > 10.0:
            logger.debug(
                "_cache_tool_configuration: %.1fms (schemas=%.1fms)",
                total_ms, schema_ms,
            )

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
        plugins: Optional[List[str]] = None,
        system_instructions: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        provider_name: Optional[str] = None,
        preloaded_plugins: Optional[set] = None,
        skip_model_test: bool = False,
        system_instruction_override: Optional[str] = None,
        suppress_base_instructions: bool = False,
        workspace_path: Optional[str] = None,
        completion_payload_schema: Optional[Any] = None,
        tier_config: Optional['ModelTierConfig'] = None,
        budget_control: Optional[Any] = None,
        agent_params: Optional[Dict[str, Any]] = None,
        completion_processors: Optional[List[Any]] = None,
        agent_id: str = "main",
        tool_scopes: Optional[Dict[str, List[str]]] = None,
        tools: Optional[List[str]] = None,  # DEPRECATED alias for ``plugins``
    ) -> 'JaatoSession':
        """Create a new session from this runtime.

        Sessions share the runtime's resources (registry, permissions, ledger)
        but have their own conversation history and can use different models
        or tool subsets.

        Args:
            model: Model name to use for this session.
            plugins: Optional list of plugin names to expose (e.g. ``"cli"``,
                   ``"web_search"``). If None, uses all exposed plugins from the
                   registry. (Plugin names, NOT tool names — per-tool allow-lists
                   live in ``tool_scopes``. Mirrors ``SubagentProfile.plugins``.)
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
            skip_model_test: If True, skip the network call that verifies the
                model responds during provider creation.
            system_instruction_override: If provided, replaces the fully-assembled
                system instruction with this exact string.  The normal assembly
                pipeline still runs (for side effects like instruction budget
                accounting) but its output is discarded.  Used by
                session-manipulation tools that replay a session with an
                edited version of the materialised prompt.
            suppress_base_instructions: If True, drop the BASE layer (the
                .jaato/instructions/ files plus any premium-provided baseline)
                from the assembled system instruction while keeping the agent
                .md content, plugin instructions, and framework constants.
                Useful for fitting a session into a small model's context window
                — the framework-level baseline is usually the largest single
                contributor.  Ignored when ``system_instruction_override`` is
                set (full override supersedes partial suppression).
            workspace_path: If provided, overrides the runtime's workspace
                path for this session.  Used by fork-replay to point a temp
                session at a worktree snapshot without affecting other sessions
                sharing the same runtime.
            tool_scopes: Optional per-plugin tool allow-lists (profile
                ``tools:[...]`` modifier).  Maps plugin name → list of
                allowed tool names; the session drops every other tool the
                plugin ships from its own wire body + grammar surface.
                Applied per-session — the shared registry is never mutated,
                so sibling sessions on this runtime keep their own scopes.
            tools: DEPRECATED alias for ``plugins`` (it always took plugin
                names, never tool names). Pass ``plugins=`` instead; ``tools=``
                still works with a one-time deprecation warning. ``plugins``
                wins if both are given.

        Returns:
            JaatoSession configured with the specified settings.

        Raises:
            RuntimeError: If runtime is not connected or configured.
        """
        # Back-compat: ``tools`` was a misleading name for the plugin-name list.
        # Honour it as a deprecated alias for ``plugins`` and warn once.
        if tools is not None:
            import warnings
            warnings.warn(
                "JaatoRuntime.create_session(tools=...) is a deprecated alias "
                "for plugins=; it takes PLUGIN names (e.g. 'cli', 'web_search'), "
                "not tool names. Use plugins= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if plugins is None:
                plugins = tools

        if not self._connected:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        if not self._registry:
            raise RuntimeError("Plugins not configured. Call configure_plugins() first.")

        # Import here to avoid circular dependency
        from .jaato_session import JaatoSession

        # Create session with runtime reference and optional provider override
        t0 = time.perf_counter()
        session = JaatoSession(
            self, model, provider_name=provider_name, agent_id=agent_id,
        )
        # Propagate the AppArmor confine-context factory so the session's
        # configure() can wrap dynamic-instructions expansion in the
        # session's confinement (server 0.6.50+).  None means no
        # confinement applies.
        if self._confine_context_factory is not None:
            session.set_confine_context_factory(self._confine_context_factory)
        session_create_ms = (time.perf_counter() - t0) * 1000

        # Configure session tools
        t1 = time.perf_counter()
        session.configure(
            plugins=plugins,
            system_instructions=system_instructions,
            plugin_configs=plugin_configs,
            preloaded_plugins=preloaded_plugins,
            skip_model_test=skip_model_test,
            system_instruction_override=system_instruction_override,
            suppress_base_instructions=suppress_base_instructions,
            workspace_path=workspace_path,
            completion_payload_schema=completion_payload_schema,
            tier_config=tier_config,
            budget_control=budget_control,
            agent_params=agent_params,
            completion_processors=completion_processors,
            tool_scopes=tool_scopes,
        )
        session_configure_ms = (time.perf_counter() - t1) * 1000

        total_ms = session_create_ms + session_configure_ms
        if total_ms > 10.0:
            logger.debug(
                "create_session: %.1fms (construct=%.1fms, configure=%.1fms)",
                total_ms, session_create_ms, session_configure_ms,
            )

        return session

    def create_session_without_provider(
        self,
        model: str,
        plugins: Optional[List[str]] = None,
        system_instructions: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        tools: Optional[List[str]] = None,  # DEPRECATED alias for ``plugins``
    ) -> 'JaatoSession':
        """Create a session without provider (for auth-pending mode).

        This creates a session with user commands available but no model
        connection. Used when authentication is pending and the user needs
        to complete auth before the model can be used.

        Args:
            model: Model name (stored for later use after auth completes).
            plugins: Optional list of plugin names to expose (NOT tool names).
            system_instructions: Optional additional system instructions.
            plugin_configs: Optional per-plugin configuration overrides.
            tools: DEPRECATED alias for ``plugins``. ``plugins`` wins if both
                are given; ``tools=`` emits a one-time deprecation warning.

        Returns:
            JaatoSession configured without a provider.
        """
        if tools is not None:
            import warnings
            warnings.warn(
                "JaatoRuntime.create_session_without_provider(tools=...) is a "
                "deprecated alias for plugins=; it takes PLUGIN names, not tool "
                "names. Use plugins= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if plugins is None:
                plugins = tools

        if not self._connected:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        if not self._registry:
            raise RuntimeError("Plugins not configured. Call configure_plugins() first.")

        from .jaato_session import JaatoSession

        session = JaatoSession(self, model)
        session.configure(
            plugins=plugins,
            system_instructions=system_instructions,
            plugin_configs=plugin_configs,
            skip_provider=True  # Don't create provider
        )

        return session

    def _inject_session_extras(
        self,
        config: 'ProviderConfig',
        session_id: Optional[str] = None,
    ) -> 'ProviderConfig':
        """Stamp per-session context onto ``config.extra`` for the provider.

        Four keys, four different provenances:

        * ``workspace_path`` / ``config_root`` — from the registry (or the
          runtime's stored ``_config_root`` when there is no registry).
          Providers need them for auth-credential lookup and OAuth token
          resolution.  The env-var approach (``JAATO_CONFIG_ROOT`` exported
          by ``JaatoServer._in_workspace``) is not reliable for headless
          reactor-spawned sessions, whose ``send_message`` runs in a fresh
          thread with no tie to the parent's context-manager scope; carrying
          the values on the config makes them available at every call site
          without thread-local fragility.
        * ``session_id`` — from the CALLER, and never from the registry.
          The registry is shared across sibling subagents, so reading a
          session id from it hands every sibling whichever session
          bootstrapped last — the exact leak
          ``JaatoSession.set_daemon_session_id`` exists to close.  It is
          therefore stamped independently of the two branches above: a
          session with neither registry nor config_root still has an
          identity worth putting on the wire.
        * ``app_identity`` — WHICH APPLICATION is making these requests
          (``shared/app_identity.py``): from the embedder's
          ``app_identity=`` kwarg, else from ``JAATO_APP_*`` in the
          environment as it stands at this call.  Stamped only when
          something actually named an application; the framework's own
          identity is left implicit so an unconfigured deployment's config
          is byte-identical to what it was before app identity existed.
          Resolved here rather than at construction because the daemon
          overlays a session's ``env`` for the duration of a turn, and a
          workspace that names its own app should be attributed to it.

        Extracted from ``create_provider`` so that method stays under its
        complexity baseline; the behaviour is unchanged.

        Args:
            config: The provider config to stamp.
            session_id: The calling session's own id, if it has one.

        Returns:
            The config, replaced with an augmented ``extra`` when there is
            anything to add, else the original object unchanged.
        """
        from dataclasses import replace

        extra = dict(config.extra)
        if self._registry:
            workspace_path = self._registry.get_workspace_path()
            config_root = self._registry.get_config_root() or self._config_root
            if workspace_path:
                extra['workspace_path'] = workspace_path
            if config_root:
                extra['config_root'] = config_root
        elif self._config_root:
            extra['config_root'] = self._config_root

        if session_id:
            extra['session_id'] = session_id

        # ``app_identity`` — WHICH APPLICATION is making these requests.
        # Not a profile knob and not per-session in spirit (see
        # ``shared/app_identity.py``): it is the embedder's explicit
        # identity, else whatever ``JAATO_APP_*`` says under the session env
        # in force right now.  Stamped as a plain dict so every provider can
        # consume it without importing the dataclass.  A provider-specific
        # knob (``plugin_configs.openrouter.app_title``) still outranks it,
        # because the profile merge below runs after this stamp.
        #
        # The framework's OWN identity is deliberately not stamped: nothing
        # named an application, so there is nothing to say that a provider
        # falling back to its own defaults does not already say.  That keeps
        # an unconfigured deployment's config byte-identical to before.
        identity = resolve_app_identity(self._app_identity)
        if identity != FRAMEWORK_IDENTITY:
            extra['app_identity'] = identity.to_dict()

        return replace(config, extra=extra) if extra != config.extra else config

    def create_provider(
        self,
        model: str,
        provider_name: Optional[str] = None,
        skip_model_test: bool = False,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
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
            skip_model_test: If True, skip the network call that verifies the
                model responds during ``provider.connect()``.  The model will
                be validated on the first real message instead.  Used during
                bootstrap to reduce startup latency.
            plugin_configs: Optional per-plugin configuration dict from the
                session profile.  Providers are plugins (``PLUGIN_KIND =
                "model_provider"``), so their profile-level knobs live under
                ``plugin_configs[provider_name]`` and are merged into
                ``config.extra`` before provider initialization.

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

        config = self._inject_session_extras(config, session_id)

        # Merge profile-level provider config.  Providers are plugins, so
        # their profile knobs sit under ``plugin_configs[provider_name]``.
        # Child keys override the stored ProviderConfig.extra so a profile
        # can tune host, context length, load params, etc. per-session.
        #
        # Server 0.6.132+ (PR-149): ``api_key`` is promoted from the
        # provider_overrides dict to the top-level ``ProviderConfig.api_key``
        # field BEFORE the rest is merged into ``extra``.  Pre-PR-149 the
        # whole dict landed in ``extra``, so ``plugin_configs.<provider>.api_key:
        # pass://...`` was silently ignored by every provider's
        # ``initialize()`` (which reads ``config.api_key``, not
        # ``config.extra["api_key"]``).  Discovered v135 — openrouter
        # cascade failed APIKeyNotFoundError despite a correctly resolved
        # pass:// URI in plugin_configs.openrouter.api_key.  Same latent
        # bug for zhipuai (its working path was the stored-credential
        # fallback ``~/.jaato/zhipuai_auth.json``, not the documented
        # plugin_configs surface).
        # The merge itself lives in ``resolve_provider_extra`` because the
        # cache plugin attached to this provider has to reproduce it, and a
        # second inline copy of it here is what would let the two drift.
        # It cannot be shared by storing the result: ``plugin_configs`` is
        # per-session while ``_provider_configs`` is runtime-wide, so writing
        # back would leak this session's knobs into every other session on
        # this provider.  See the function's docstring.
        if plugin_configs:
            merged_extra, promoted_api_key = resolve_provider_extra(
                config.extra, plugin_configs, effective_provider)
            # Unconditional: ``replace`` with an equal ``extra`` yields an
            # equal config, so guarding it would only trade a branch for an
            # allocation on a path that goes on to do network I/O.
            from dataclasses import replace
            replace_kwargs: Dict[str, Any] = {"extra": merged_extra}
            if promoted_api_key:
                replace_kwargs["api_key"] = promoted_api_key
            config = replace(config, **replace_kwargs)

        # Fail loud at the provider credential boundary: if api_key is still
        # shaped like an unresolved secret URI (e.g. ``pass://...`` that passed
        # through because no resolver is registered — the providing plugin isn't
        # installed), refuse rather than send the literal URI as a credential
        # (which produced a confusing upstream 401 — the nebius regression).
        # ``_resolve_secret_uri`` stays lenient/pass-through so non-provider
        # consumers (service_connector) keep reporting "credential missing"
        # gracefully; the strict check lives here, where a literal secret URI
        # is unambiguously wrong.
        from .plugins.subagent.config import (
            looks_like_unresolved_secret_uri,
            looks_like_malformed_secret_uri,
            SecretResolutionError,
        )
        # Near-miss FIRST: a single-colon ``pass:...`` (the ``//``-dropped typo)
        # is invisible to the resolver (regex miss → passed through literally),
        # so it would otherwise leak to the provider as a bearer token and
        # produce a confusing upstream 401.  Fail loud with a did-you-mean.
        malformed_scheme = looks_like_malformed_secret_uri(config.api_key)
        if malformed_scheme:
            raise SecretResolutionError(
                config.api_key,
                f"provider '{effective_provider}' received a MALFORMED secret "
                f"URI as its api_key — a single-colon '{malformed_scheme}:...'. "
                f"Secret URIs require '//': did you mean "
                f"'{malformed_scheme}://<path>'?  (A resolver for "
                f"'{malformed_scheme}' IS registered; only the '//' is missing.)",
            )
        if looks_like_unresolved_secret_uri(config.api_key):
            raise SecretResolutionError(
                config.api_key,
                f"provider '{effective_provider}' received an unresolved secret "
                f"URI as its api_key — no resolver is registered for its scheme "
                f"(is the plugin that provides it, e.g. jaato-premium, "
                f"installed?). Refusing to send a literal secret URI as a "
                f"credential.",
            )

        t0 = time.perf_counter()
        provider = load_provider(effective_provider, config)
        load_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        provider.connect(model, skip_model_test=skip_model_test)
        connect_ms = (time.perf_counter() - t1) * 1000

        total_ms = load_ms + connect_ms
        if total_ms > 10.0:
            logger.debug(
                "create_provider(%s): %.1fms (load=%.1fms, connect=%.1fms)",
                effective_provider, total_ms, load_ms, connect_ms,
            )
        return provider

    def _get_core_plugins(self) -> List[str]:
        """Find all plugins that provide tools with discoverability='core'.

        .. note::

           This helper is no longer called by ``_get_essential_plugins`` — a
           profile's explicit plugin list is now authoritative and only
           ``introspection`` is auto-added.  The method is retained for
           potential diagnostic use.

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
                    if getattr(schema, 'discoverability', None) == DISCOVERABILITY_EAGER:
                        core_plugins.append(plugin_name)
                        break  # Found one core tool, plugin qualifies
            except Exception:
                pass  # Skip plugins that fail to provide schemas

        return core_plugins

    def _get_essential_plugins(self, plugin_names: List[str]) -> List[str]:
        """Get plugin list with only truly essential plugins added.

        Only ``introspection`` is unconditionally essential — it provides the
        ``list_tools`` / ``get_tool_schemas`` tools that the deferred-loading
        mechanism depends on.  All other core plugins are **not** auto-added;
        the profile's explicit plugin list is authoritative.  If a profile
        omits a plugin, that plugin's tools must not appear in the session,
        even if the plugin has ``discoverability='core'`` tools.

        Args:
            plugin_names: Plugin names from the profile (authoritative list).

        Returns:
            Plugin list with ``introspection`` added (if not already present).
        """
        result = list(plugin_names)

        # Only introspection is unconditionally essential (needed for deferred
        # tool discovery).  Other core plugins are NOT auto-added — the
        # profile's plugin list takes precedence.
        essential = ["introspection"]
        for name in essential:
            if name not in result:
                result.append(name)
            # Ensure the essential plugin is exposed in the registry
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
                        if getattr(s, 'discoverability', DISCOVERABILITY_DEFERRED) == DISCOVERABILITY_EAGER
                    ]
                schemas.extend(plugin_schemas)

        # Add permission plugin schemas if permission plugin is configured
        if self._permission_plugin:
            permission_schemas = self._permission_plugin.get_tool_schemas()
            if deferred_enabled:
                # Permission tools should be core (always available)
                permission_schemas = [
                    s for s in permission_schemas
                    if getattr(s, 'discoverability', DISCOVERABILITY_DEFERRED) == DISCOVERABILITY_EAGER
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

        # Add core tool executors (framework infrastructure, not plugin-specific)
        if self._registry:
            executors.update(self._registry.get_core_executors())

        return executors

    def get_system_instructions(
        self,
        plugin_names: Optional[List[str]] = None,
        additional: Optional[str] = None,
        presentation_context: Optional['PresentationContext'] = None,
        preloaded_plugins: Optional[set] = None,
        include_base: bool = True,
        include_constants: bool = True,
        include_security: bool = True,
    ) -> Optional[str]:
        """Get system instructions, optionally filtered by plugin names.

        The final instructions are assembled in this order:
        1. Base system instructions from .jaato/instructions/ folder (if exists,
           falls back to legacy .jaato/system_instructions.md) — skipped when
           ``include_base`` is False so a session can keep its own agent/plugin/
           framework content without carrying the framework-level baseline
           (useful for small-context models)
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
            include_constants: When False, skip the framework prompt constants
                              (task-completion/verification, parallel/batching,
                              turn-summary) — the granular counterpart of
                              ``include_base``.  Driven by
                              ``suppress_base_instructions: {constants: true}``.
            include_security: When False, skip the untrusted-content boundary.
                              Driven only by an explicit
                              ``suppress_base_instructions: {security: true}``
                              (the blanket ``true`` keeps it — it is the
                              indirect-prompt-injection defense).

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

        # 1. Base system instructions from .jaato/instructions/ (or legacy
        #    single file) — lazy-loaded on first request.  Skipped when
        #    include_base=False so sessions with suppress_base_instructions
        #    don't pay the disk I/O (and don't get the baseline content).
        if include_base:
            base = self.get_base_system_instructions()
            if base:
                result_parts.append(base)

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

        # 6. Framework-level prompt constants (provided by jaato-premium).
        #    Skipped when include_constants=False — the granular counterpart of
        #    include_base, for ``suppress_base_instructions: {constants: true}``.
        if include_constants:
            result_parts.extend(_framework_prompt_constants())

        # 7. Untrusted-content boundary (security baseline).  Included by
        # default — web_fetch/web_search/MCP tools are deferred-loaded, so
        # gating on tool presence would drop the instruction for a tool the
        # model can still discover + call.  Teaches the model to treat
        # boundary-wrapped tool results as data, not instructions
        # (indirect-prompt-injection defense).  Dropped ONLY when a session
        # explicitly opts in via ``suppress_base_instructions: {security:
        # true}`` (never by the blanket ``true``).
        if include_security:
            from jaato_sdk.plugins.model_provider.types import untrusted_boundary_instruction
            result_parts.append(untrusted_boundary_instruction())

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
