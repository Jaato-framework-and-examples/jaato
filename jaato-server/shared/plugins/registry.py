"""Plugin registry for discovering, loading, and managing plugins."""

import importlib
import importlib.metadata
import logging
import os
import pkgutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Callable, Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

from jaato_sdk.plugins.base import (
    ToolPlugin,
    EnrichmentPlugin,
    AnyPlugin,
    UserCommand,
    PromptEnrichmentResult,
    SystemInstructionEnrichmentResult,
    ToolResultEnrichmentResult,
    model_matches_requirements,
    OutputCallback,
)
from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    DISCOVERABILITY_EAGER,
    DISCOVERABILITY_DEFERRED,
)
from .streaming.protocol import StreamingCapable
from .entry_point_trust import (
    BUILTIN_PLUGIN_PACKAGE as _BUILTIN_PLUGIN_PACKAGE,
    PluginOrigin,
    entry_point_distribution,
    entry_point_module,
    evaluate_entry_point,
)
from .enrichment_formatter import (
    EnrichmentNotification,
    format_enrichment_notifications,
)
from shared.trace import trace as _trace_write

# Entry point group names by plugin kind
PLUGIN_ENTRY_POINT_GROUPS = {
    "tool": "jaato.plugins",
    "enrichment": "jaato.enrichment_plugins",
    "gc": "jaato.gc_plugins",
    "cache": "jaato.cache_plugins",
}


# Cross-tier plugin discovery.  ``PLUGIN_TIER = "daemon_callable"`` is
# the third tier value (alongside ``"daemon"`` / ``"runner"``): the
# plugin is discovered on BOTH sides, with the daemon-side instance
# owning the body and the runner-side instance being a
# ``DaemonForwardingMixin`` stub for schema-surfacing to the model.
# See ``shared/plugins/CLAUDE.md`` for the full pattern + the
# ``test_plugin_tier_partition`` static-analysis gate.
#
# Discovery semantics encoded here (consumed by
# :func:`_tier_filter_matches`):
#   - ``tier_filter="runner"`` accepts ``{"runner", "daemon_callable"}``
#   - ``tier_filter="daemon"`` accepts ``{"daemon", "daemon_callable"}``
#   - ``tier_filter=None`` accepts everything (Phase 2 behaviour)
_TIER_FILTER_ACCEPTS: Dict[str, Set[str]] = {
    "runner": {"runner", "daemon_callable"},
    "daemon": {"daemon", "daemon_callable"},
}


def _tier_filter_matches(
    plugin_tier: Optional[str], tier_filter: Optional[str],
) -> bool:
    """Whether *plugin_tier* should be loaded under *tier_filter*.

    Encapsulates the cross-tier acceptance rule so both
    ``_discover_via_entry_points`` and ``_discover_via_directory``
    share one implementation.  Plugins lacking a ``PLUGIN_TIER``
    annotation are excluded under ANY filter (the "annotate or be
    excluded" contract from §3.3.5) — silent exclusion is what the
    build-fail gate in ``test_plugin_tier_partition`` catches.

    Args:
        plugin_tier: Declared ``PLUGIN_TIER`` value, or ``None`` if
            absent.  Must be one of ``"daemon"`` / ``"runner"`` /
            ``"daemon_callable"`` for the filtered path to admit it.
        tier_filter: Filter to apply.  ``None`` means "no filter"
            and admits every annotated plugin.

    Returns:
        True iff the plugin should be loaded under this filter.
    """
    if tier_filter is None:
        # No filter: any annotated plugin passes.  Unannotated ones
        # (plugin_tier is None) still pass — the "annotate or be
        # excluded" rule only fires when a filter IS set, matching
        # the pre-§3.3.5 daemon-side behaviour where everything got
        # loaded.
        return True
    if plugin_tier is None:
        # Filter set but no annotation: skip per §3.3.5 contract.
        return False
    accepted = _TIER_FILTER_ACCEPTS.get(tier_filter, {tier_filter})
    return plugin_tier in accepted


def _trace(msg: str, include_traceback: bool = False, warning: bool = False) -> None:
    """Write trace message to log file for debugging.

    Args:
        msg: The message to log.
        include_traceback: If True, log at ERROR with the current exception
            traceback attached (for genuine plugin bugs / unexpected failures).
        warning: If True (and ``include_traceback`` is False), log at WARNING
            for a concise, user-visible one-liner — used for expected,
            actionable skips (e.g. a plugin's optional backend dependency is
            not installed) where a full traceback would read as a crash.
    """
    _trace_write("PluginRegistry", msg, include_traceback=include_traceback)
    # Also log to standard logger for visibility
    if include_traceback:
        logger.error(msg, exc_info=True)
    elif warning:
        logger.warning(msg)
    else:
        logger.debug(msg)


#: Which Protocol a discovered plugin must satisfy, per plugin kind.
#: Kinds absent from this map (``gc``, ``cache``) carry no structural
#: contract the registry can check, so they are admitted unchecked.
_PLUGIN_KIND_PROTOCOLS: Dict[str, type] = {
    "tool": ToolPlugin,
    "enrichment": EnrichmentPlugin,
}


def _protocol_gap(plugin: Any, plugin_kind: str) -> Optional[str]:
    """Render the protocol methods *plugin* is missing, or ``None``.

    Both discovery paths (entry point and directory) must refuse a
    plugin that fails its kind's Protocol check, and both must say
    which methods are missing rather than dropping it silently — the
    PR #171 / PR-210 lesson.  This helper holds the shared half so the
    two call sites differ only in the wording that identifies *where*
    the plugin came from.

    Args:
        plugin: The instantiated plugin.
        plugin_kind: The kind being discovered (``"tool"``,
            ``"enrichment"``, ...).

    Returns:
        A comma-separated list of missing method names, ``"(unknown)"``
        when the plugin fails the check but introspection yields
        nothing actionable, or ``None`` when the plugin satisfies the
        Protocol (or its kind declares none).
    """
    protocol = _PLUGIN_KIND_PROTOCOLS.get(plugin_kind)
    if protocol is None or isinstance(plugin, protocol):
        return None
    missing = _missing_protocol_methods(plugin, protocol)
    return ", ".join(missing) if missing else "(unknown)"


def _missing_protocol_methods(plugin: Any, protocol_cls: type) -> List[str]:
    """Return method/property names the protocol declares but ``plugin`` lacks.

    Used to produce a useful error when a plugin fails an
    ``isinstance(plugin, SomeProtocol)`` check — instead of just saying
    "doesn't implement the protocol", we tell the developer exactly which
    methods are missing.

    Only inspects attributes declared directly on ``protocol_cls`` (not
    inherited from ``Protocol`` itself), so the result is the *plugin
    author's* contract, not Python's protocol machinery.
    """
    required: List[str] = []
    for attr_name, attr_val in vars(protocol_cls).items():
        if attr_name.startswith("_"):
            continue
        if callable(attr_val) or isinstance(attr_val, property):
            required.append(attr_name)
    return sorted(name for name in required if not hasattr(plugin, name))


class PluginRegistry:
    """Manages plugin discovery, lifecycle, and tool exposure state.

    Usage:
        registry = PluginRegistry()
        registry.discover()

        print(registry.list_available())  # ['cli', 'mcp', ...]

        registry.expose_tool('cli', config={'extra_paths': ['/usr/local/bin']})
        registry.expose_tool('mcp')

        # Get tools for exposed plugins
        tool_schemas = registry.get_exposed_tool_schemas()
        executors = registry.get_exposed_executors()

        # Later, unexpose plugins
        registry.unexpose_tool('mcp')
        registry.unexpose_all()

    External Path Authorization:
        Plugins can authorize external paths (outside workspace) for model access.
        This is used by the references plugin to allow readFile to access
        referenced documents outside the workspace.

        # Register an authorized external path
        registry.authorize_external_path('/docs/external/guide.md', 'references')

        # Check if a path is authorized
        if registry.is_path_authorized('/docs/external/guide.md'):
            # Allow access

        # Clear authorizations from a specific plugin
        registry.clear_authorized_paths('references')

    Auto-Wiring:
        Plugins may implement optional methods for automatic dependency injection:

        - set_plugin_registry(registry): Called during expose_tool() to give plugins
          access to the registry for cross-plugin communication.

        - set_workspace_path(path): Called when workspace changes to notify plugins
          that need workspace-relative operations (sandboxing, relative paths, etc.).
          The registry broadcasts to all exposed plugins implementing this method.

        Example plugin implementation:
            class MyPlugin:
                def set_workspace_path(self, path: str) -> None:
                    self._workspace_path = path
                    self._sandbox.set_root(path)
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the plugin registry.

        Args:
            model_name: Optional model name for checking plugin requirements.
                       If provided, plugins with model_requirements that don't
                       match will be skipped during expose_tool().
        """
        self._plugins: Dict[str, AnyPlugin] = {}
        # Provenance for every registered plugin: name -> PluginOrigin.
        # Populated by both discovery paths and by register_plugin(), so
        # a shadowed built-in is visible without reading logs and so a
        # later claim on an occupied name can name the incumbent
        # (issue #684).
        self._plugin_sources: Dict[str, PluginOrigin] = {}
        self._exposed: Set[str] = set()
        self._enrichment_only: Set[str] = set()  # Plugins for prompt enrichment only
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._model_name: Optional[str] = model_name
        self._skipped_plugins: Dict[str, List[str]] = {}  # name -> required patterns
        # Plugins that raised during initialize() / shutdown(). Tracked
        # so callers can introspect lifecycle failures and so we don't
        # silently leave the registry in an inconsistent state.
        # name -> (lifecycle_phase, error_message)
        self._failed_plugins: Dict[str, tuple] = {}
        self._disabled_tools: Set[str] = set()  # Individual tools disabled by user
        self._output_callback: Optional[OutputCallback] = None
        self._terminal_width: int = 80
        # Authorized external paths: path -> (source plugin name, access mode)
        # access mode is "readonly" or "readwrite"
        self._authorized_external_paths: Dict[str, tuple] = {}
        # Denied external paths: path -> source plugin name (takes precedence over authorized)
        self._denied_external_paths: Dict[str, str] = {}
        # Core tools: framework-provided tools not from plugins
        self._core_tools: Dict[str, ToolSchema] = {}
        self._core_executors: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._core_auto_approved: Set[str] = set()  # Auto-approved core tools
        # Workspace path for plugins that need it
        self._workspace_path: Optional[str] = None
        # Optional override for the read-only framework-config root.
        # When None (the default), plugins that consult
        # ``shared.config_resolver.resolve_config_search_path`` fall back
        # to ``<workspace_path>/.jaato/`` (today's behavior).
        self._config_root: Optional[str] = None
        # Per-session framework values injected into every plugin's
        # config dict at ``initialize()`` time (server 0.6.129+).
        # See :meth:`_augment_plugin_config` for the contract.  Set
        # before :meth:`expose_all` fires; the post-init
        # ``set_workspace_path`` / ``set_config_root`` broadcasts
        # still run for idempotent refresh AND to propagate later
        # mid-session changes.
        self._session_id: Optional[str] = None
        self._agent_name: Optional[str] = None
        # Cache: tool_name -> plugin for get_plugin_for_tool() lookups
        self._tool_plugin_cache: Dict[str, ToolPlugin] = {}
        # Bootstrap timing: plugin name -> timing data
        self._discovery_timings: Dict[str, dict] = {}
        self._init_timings: Dict[str, dict] = {}
        # Category descriptions: category name -> human-readable description.
        # Starts empty — each plugin registers its own category via
        # ``register_category()`` in its ``set_plugin_registry()`` hook.
        # This ensures premium and third-party plugins can introduce new
        # categories without touching the registry's source code.
        self._category_descriptions: Dict[str, str] = {}

    def set_output_callback(
        self,
        callback: Optional[OutputCallback],
        terminal_width: int = 80
    ) -> None:
        """Set the output callback for enrichment notifications.

        When set, the registry will emit visual notifications about enrichments
        through this callback with source="enrichment".

        Args:
            callback: OutputCallback function (source, text, mode) -> None,
                     or None to disable notifications.
            terminal_width: Terminal width for word wrapping (default 80).
        """
        self._output_callback = callback
        self._terminal_width = terminal_width
        _trace(f"set_output_callback: callback={callback is not None}, terminal_width={terminal_width}")

    def _emit_enrichment_notifications(
        self,
        notifications: List[EnrichmentNotification],
        output_callback: Optional[OutputCallback] = None,
        terminal_width: Optional[int] = None
    ) -> None:
        """Emit enrichment notifications through the output callback.

        Args:
            notifications: List of enrichment notifications to display.
            output_callback: Optional callback to use instead of the registry's
                shared callback. Used for session-specific routing.
            terminal_width: Terminal width for formatting (uses registry default if not provided).
        """
        # Use provided callback or fall back to shared one
        callback = output_callback if output_callback is not None else self._output_callback
        width = terminal_width if terminal_width is not None else self._terminal_width

        _trace(f"_emit_enrichment_notifications: {len(notifications)} notifications, callback={callback is not None}")
        if not callback or not notifications:
            return

        formatted = format_enrichment_notifications(
            notifications,
            terminal_width=width
        )
        _trace(f"_emit_enrichment_notifications: formatted={repr(formatted[:100]) if formatted else 'empty'}")
        if formatted:
            callback("enrichment", formatted, "write")

    def _build_notification(
        self,
        target: str,
        plugin_name: str,
        metadata: Optional[Dict[str, Any]],
        size_bytes: int
    ) -> Optional[EnrichmentNotification]:
        """Build an enrichment notification from plugin metadata.

        Plugins can provide explicit notification info via metadata["notification"],
        or a fallback message is generated from known metadata patterns.

        Args:
            target: What was enriched ("prompt", "result", "system").
            plugin_name: Name of the plugin that performed enrichment.
            metadata: Plugin's returned metadata, may contain notification info.
            size_bytes: Size difference in bytes (can be negative for reductions).

        Returns:
            EnrichmentNotification if a message could be generated, None otherwise.
        """
        if metadata is None:
            metadata = {}

        # Check for explicit notification in metadata
        notif_data = metadata.get("notification")
        if isinstance(notif_data, dict):
            message = notif_data.get("message", "enriched content")
            notif_size = notif_data.get("size_bytes", abs(size_bytes) if size_bytes else None)
            return EnrichmentNotification(
                target=target,
                plugin=plugin_name,
                message=message,
                size_bytes=notif_size
            )

        # Generate fallback message from known metadata patterns
        message = self._generate_fallback_message(plugin_name, metadata)
        if message:
            return EnrichmentNotification(
                target=target,
                plugin=plugin_name,
                message=message,
                size_bytes=abs(size_bytes) if size_bytes else None
            )

        # Last resort: generic message
        return EnrichmentNotification(
            target=target,
            plugin=plugin_name,
            message="enriched content",
            size_bytes=abs(size_bytes) if size_bytes else None
        )

    def _generate_fallback_message(
        self,
        plugin_name: str,
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a notification message from known metadata patterns.

        Args:
            plugin_name: Name of the plugin for pattern matching.
            metadata: Plugin's returned metadata.

        Returns:
            A human-readable message, or None if no pattern matched.
        """
        # Memory plugin pattern
        if "memory_matches" in metadata:
            count = metadata["memory_matches"]
            if count > 0:
                # Include trigger keywords if available
                keywords = metadata.get("trigger_keywords", [])
                if keywords:
                    keyword_summary = ", ".join(f'"{k}"' for k in keywords[:3])
                    if len(keywords) > 3:
                        keyword_summary += f" +{len(keywords) - 3} more"
                    return f"added context about {count} memories (triggered by {keyword_summary})"
                return f"added context about {count} relevant memories"

        # Session plugin pattern
        if metadata.get("description_requested"):
            return "requested session description (turn threshold reached)"

        # LSP plugin pattern - show diagnostic counts
        if "total_errors" in metadata or "total_warnings" in metadata:
            errors = metadata.get("total_errors", 0)
            warnings = metadata.get("total_warnings", 0)
            files = metadata.get("files_checked", [])
            file_str = files[0] if len(files) == 1 else f"{len(files)} files"

            parts = []
            if errors > 0:
                parts.append(f"{errors} error{'s' if errors != 1 else ''}")
            if warnings > 0:
                parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")

            if parts:
                return f"found {', '.join(parts)} in {file_str}"
            else:
                return f"checked {file_str}, no issues found"

        # References plugin pattern — @mention expansion in prompts/tool results
        if "mentioned_references" in metadata:
            refs = metadata["mentioned_references"]
            if refs:
                if len(refs) == 1:
                    return f"expanded @{refs[0]}"
                elif len(refs) <= 3:
                    ref_list = ", ".join(f"@{r}" for r in refs)
                    return f"expanded {ref_list}"
                else:
                    return f"expanded {len(refs)} references"

        # References plugin pattern - tag-matched reference ID hints.
        # Always include the matched tags in the notification so the user
        # can see which tags triggered the match and diagnose false positives.
        if "tag_matched_references" in metadata:
            matched = metadata["tag_matched_references"]  # {source_id: [tags]}
            if matched:
                ids = list(matched.keys())
                # Collect unique matched tags across all hinted sources
                all_tags = sorted(set(t for tags in matched.values() for t in tags))
                tag_summary = ", ".join(all_tags[:5])
                if len(all_tags) > 5:
                    tag_summary += f" +{len(all_tags) - 5}"
                if len(ids) == 1:
                    return f"hinted @{ids[0]} (matched: {tag_summary})"
                elif len(ids) <= 3:
                    ref_list = ", ".join(f"@{r}" for r in ids)
                    return f"hinted {ref_list} (matched: {tag_summary})"
                else:
                    shown = ", ".join(f"@{r}" for r in ids[:3])
                    return f"hinted {shown} +{len(ids) - 3} more (matched: {tag_summary})"

        # References plugin pattern — transitively selected references.
        # Shows which references were auto-included and which parent source
        # triggered the inclusion.
        if "transitive_references" in metadata:
            transitive = metadata["transitive_references"]  # {id: [parent_ids]}
            if transitive:
                ids = list(transitive.keys())
                # Collect unique parent IDs across all transitive sources
                all_parents = sorted(set(p for parents in transitive.values() for p in parents))
                parent_str = ", ".join(f"@{p}" for p in all_parents[:3])
                if len(all_parents) > 3:
                    parent_str += f" +{len(all_parents) - 3}"
                if len(ids) == 1:
                    return f"transitively included @{ids[0]} (from {parent_str})"
                elif len(ids) <= 3:
                    ref_list = ", ".join(f"@{r}" for r in ids)
                    return f"transitively included {ref_list} (from {parent_str})"
                else:
                    shown = ", ".join(f"@{r}" for r in ids[:3])
                    return f"transitively included {shown} +{len(ids) - 3} more (from {parent_str})"

        # Template extraction pattern
        if "extracted_templates" in metadata:
            templates = metadata["extracted_templates"]
            if templates:
                if len(templates) == 1:
                    return f"extracted @{templates[0]} template"
                else:
                    return f"extracted {len(templates)} templates"

        return None

    def discover(
        self,
        plugin_kind: str = "tool",
        include_directory: bool = True,
        tier_filter: Optional[str] = None,
    ) -> List[str]:
        """Discover plugins via entry points and optionally directory scanning.

        Discovery order:
        1. Entry points (group based on plugin_kind) - for installed packages
        2. Directory scanning (optional) - for development/local plugins
        3. If plugin_kind is "tool", also discovers "enrichment" plugins
           (they are registered as enrichment-only automatically)

        Entry points allow external packages to register plugins:
            [project.entry-points."jaato.plugins"]
            my_plugin = "my_package.plugins:create_plugin"

        Args:
            plugin_kind: Kind of plugin to discover ('tool', 'enrichment', 'gc', etc.).
                        Only plugins with matching PLUGIN_KIND are loaded.
            include_directory: Also scan the plugins directory for local plugins.
                             Useful during development when package isn't installed.
            tier_filter: Optional tier restriction (Phase 3 §3.3.5).  When
                set to ``"daemon"``, only plugins declaring
                ``PLUGIN_TIER = "daemon"`` are loaded; ``"runner"`` loads
                only runner-tier plugins.  When ``None`` (default), no
                tier restriction — preserves Phase 2 behaviour where the
                daemon discovers everything.  Plugins without a
                ``PLUGIN_TIER`` annotation are SKIPPED when a filter is
                set (the contract is "annotate or be excluded"); this
                makes the partition deterministic and catches new
                plugins landing without an explicit tier.

        Returns:
            List of discovered plugin names.
        """
        discovered = []

        # First, discover via entry points (installed packages)
        discovered.extend(
            self._discover_via_entry_points(plugin_kind, tier_filter=tier_filter)
        )

        # Then, optionally scan the plugins directory (development mode)
        if include_directory:
            discovered.extend(
                self._discover_via_directory(
                    plugin_kind, tier_filter=tier_filter,
                )
            )

        # When discovering tool plugins, also discover enrichment plugins
        # so callers don't need to make a separate discover("enrichment") call
        if plugin_kind == "tool":
            discovered.extend(
                self._discover_via_entry_points(
                    "enrichment", tier_filter=tier_filter,
                )
            )
            if include_directory:
                discovered.extend(
                    self._discover_via_directory(
                        "enrichment", tier_filter=tier_filter,
                    )
                )

        return discovered

    def _discover_via_entry_points(
        self,
        plugin_kind: str,
        *,
        tier_filter: Optional[str] = None,
    ) -> List[str]:
        """Discover plugins registered via entry points.

        External packages can register plugins by adding to the appropriate
        entry point group in their pyproject.toml.

        Args:
            plugin_kind: Kind of plugin to discover ('tool', 'gc', etc.).
            tier_filter: Optional tier restriction (Phase 3 §3.3.5).
                When set, the loaded plugin's module-level
                ``PLUGIN_TIER`` must match; mismatched plugins are
                skipped (a debug trace records the skip reason).

        Returns:
            List of discovered plugin names.
        """
        discovered = []

        entry_point_group = PLUGIN_ENTRY_POINT_GROUPS.get(plugin_kind)
        if not entry_point_group:
            return discovered

        try:
            # Python 3.10+ API
            if sys.version_info >= (3, 10):
                eps = importlib.metadata.entry_points(group=entry_point_group)
            else:
                # Python 3.9 compatibility
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get(entry_point_group, [])

            for ep in eps:
                # Trust gate (issue #684): reserved-name, allowlist and
                # occupied-name checks, all decided from the entry
                # point's METADATA so a refused claim is never
                # ``ep.load()``-ed and its module never imported.
                origin = self._gate_entry_point(ep)
                if origin is None:
                    continue

                try:
                    # Phase 3 §3.3.5: tier filter — read PLUGIN_TIER from
                    # the entry-point's defining module.  ``ep.load()``
                    # imports the module already; we read the attr from
                    # the loaded factory's module.  Cross-tier
                    # ``"daemon_callable"`` plugins are admitted by
                    # BOTH ``"runner"`` and ``"daemon"`` filters — see
                    # :func:`_tier_filter_matches`.
                    create_plugin = ep.load()
                    if tier_filter is not None:
                        ep_module = getattr(create_plugin, "__module__", "")
                        ep_tier = self._lookup_module_tier(ep_module)
                        if not _tier_filter_matches(ep_tier, tier_filter):
                            _trace(
                                f" Entry point '{ep.name}': PLUGIN_TIER "
                                f"={ep_tier!r} not accepted by "
                                f"filter={tier_filter!r}; skipping"
                            )
                            continue

                    plugin = create_plugin()

                    # Verify protocol implementation.  Failing the
                    # protocol check used to be a debug-only _trace,
                    # which made plugins invisibly disappear from the
                    # registry — same footgun as missing PLUGIN_KIND
                    # and same footgun the directory-path side fixed
                    # at PR #171.  Promote to a real warning that
                    # names the missing methods so the author knows
                    # exactly what to add.  Mirrors the directory-
                    # path warning at ``_discover_via_directory``
                    # (both share ``_protocol_gap``).
                    gap = _protocol_gap(plugin, plugin_kind)
                    if gap is not None:
                        logger.warning(
                            "Entry point '%s' (%s): plugin does not "
                            "implement the %s protocol — it will be "
                            "silently skipped. Missing methods: %s. "
                            "Add them to the plugin class.",
                            ep.name,
                            getattr(ep, "value", "<unknown>"),
                            _PLUGIN_KIND_PROTOCOLS[plugin_kind].__name__,
                            gap,
                        )
                        continue

                    # A plugin's ``.name`` property need not match the
                    # name it was declared under, so re-run the trust
                    # gate on the name it actually claims — otherwise
                    # an entry point called ``harmless`` could return a
                    # plugin named ``permission`` (issue #684).
                    if not self._admit_claimed_name(plugin.name, origin):
                        continue

                    self._plugins[plugin.name] = plugin
                    self._plugin_sources[plugin.name] = PluginOrigin(
                        name=plugin.name,
                        via="entry_point",
                        module=origin.module,
                        distribution=origin.distribution,
                        entry_point=origin.entry_point,
                    )
                    discovered.append(plugin.name)

                    # Enrichment plugins are always enrichment-only
                    if plugin_kind == "enrichment":
                        self._enrichment_only.add(plugin.name)

                except Exception as exc:
                    _trace(f" Error loading entry point '{ep.name}': {exc}", include_traceback=True)

        except Exception as exc:
            # Entry points not available (package not installed)
            pass

        return discovered

    def _gate_entry_point(self, ep: Any) -> Optional[PluginOrigin]:
        """Apply the #684 trust policy to one entry point, pre-``load()``.

        Runs entirely off the entry point's metadata (``ep.name``,
        ``ep.value``, ``ep.dist``), so a refused claim never has its
        module imported — which matters because ``ep.load()`` executes
        the target module, and "merely being installed" was previously
        enough to run third-party code at discovery time.

        Three ways an entry point is turned away, each with a WARNING
        rather than a bare ``continue``:

        * the name is already registered by a *different* provider
          (duplicate claim — first writer wins, and the loser is named);
        * its distribution is outside a configured allowlist;
        * it claims a reserved built-in name without an operator opt-in
          (never available at all for the security-critical set).

        Args:
            ep: The ``importlib.metadata.EntryPoint`` under consideration.

        Returns:
            A :class:`PluginOrigin` describing the accepted claimant —
            its ``name`` is the *entry-point* name, refined to the
            plugin's own name once instantiated — or ``None`` when the
            entry point must be skipped.
        """
        module = entry_point_module(ep)
        distribution = entry_point_distribution(ep)
        origin = PluginOrigin(
            name=ep.name,
            via="entry_point",
            module=module,
            distribution=distribution,
            entry_point=ep.name,
        )

        if ep.name in self._plugins:
            # Pre-#684 this was a bare ``continue``.  Re-discovery of
            # the same provider is normal and stays quiet; a second
            # distribution claiming an occupied name is a shadow the
            # operator needs to see.
            self._warn_name_taken(ep.name, origin)
            return None

        decision = evaluate_entry_point(ep.name, module, distribution)
        if decision.warn:
            logger.warning("%s", decision.message)
        return origin if decision.allowed else None

    def _admit_claimed_name(self, claimed: str, origin: PluginOrigin) -> bool:
        """Re-apply the trust policy to the name a loaded plugin claims.

        ``_gate_entry_point`` can only judge the *declared* name.  A
        plugin whose ``name`` property returns something else would
        otherwise land in ``self._plugins`` under a name nobody vetted —
        including a reserved built-in one.  Called after instantiation
        and before registration; a plugin whose ``.name`` matches its
        entry-point name has already been cleared and passes straight
        through.

        Args:
            claimed: The value of the instantiated plugin's ``name``.
            origin: What :meth:`_gate_entry_point` accepted.

        Returns:
            True when the plugin may be registered under *claimed*.
        """
        if claimed == origin.entry_point:
            return True
        if claimed in self._plugins:
            self._warn_name_taken(claimed, origin)
            return False
        decision = evaluate_entry_point(
            claimed, origin.module, origin.distribution,
        )
        if decision.warn:
            logger.warning("%s", decision.message)
        return decision.allowed

    def _warn_name_taken(self, name: str, challenger: PluginOrigin) -> bool:
        """Log the loser when two providers claim the same plugin name.

        First writer wins — that ordering is unchanged, and is why
        entry points beat the directory scan.  What changes is that the
        skip is no longer silent: both the incumbent and the challenger
        are named, so an operator can see *which* distribution took over
        a plugin they thought was the built-in one.

        Two collisions are benign and stay quiet:

        * the same module re-offering a name it already holds
          (``discover()`` runs more than once per process, and the
          enrichment pass re-walks the same directory);
        * one built-in reaching the registry by two routes — the
          framework declares its plugins as entry points AND ships them
          in the scanned directory, and registers a few (the session
          plugin) directly.  A built-in losing to a built-in is not the
          shadow this warning is for.

        The second exemption is what keeps this warning worth reading.
        The framework's own entry points hit this branch on every single
        startup, so warning unconditionally would emit roughly one line
        per built-in — measured at 0 warnings across 42 discovered
        plugins on a normal installed tree, versus ~40 without the
        exemption.  A security warning that fires constantly trains
        operators to ignore it, which costs more than saying nothing.

        Args:
            name: The contested plugin name.
            challenger: The provider that arrived second and lost.

        Returns:
            True if a warning was emitted, False if the collision was a
            benign re-registration of a built-in.
        """
        incumbent = self._plugin_sources.get(name)
        if incumbent is not None and incumbent.module == challenger.module:
            return False
        if incumbent is not None and incumbent.builtin and challenger.builtin:
            return False
        held_by = incumbent.describe() if incumbent else "an earlier registration"
        logger.warning(
            "Plugin name '%s' is already provided by %s; ignoring the "
            "competing declaration from %s. Whichever loads first wins "
            "— rename one of them to make the outcome deterministic.",
            name,
            held_by,
            challenger.describe(),
        )
        return True

    @staticmethod
    def _lookup_module_tier(module_name: str) -> Optional[str]:
        """Read ``PLUGIN_TIER`` from a plugin module's package.

        Most plugin entry points point at
        ``<package>.<plugin_name>.plugin:create_plugin``; the
        ``PLUGIN_TIER`` annotation lives on the plugin package's
        ``__init__.py``.  We walk up to the package root by stripping
        the trailing ``.plugin`` (or whatever submodule the factory
        lives in).

        Returns the tier string (``"daemon"`` / ``"runner"``) or
        ``None`` if the module isn't loaded yet OR the package
        doesn't declare a tier.
        """
        if not module_name:
            return None
        # Try the module itself first, then its parent package.
        # ``shared.plugins.cli.plugin`` → ``shared.plugins.cli``.
        candidates = [module_name]
        if "." in module_name:
            candidates.append(module_name.rsplit(".", 1)[0])
        for candidate in candidates:
            mod = sys.modules.get(candidate)
            if mod is not None:
                tier = getattr(mod, "PLUGIN_TIER", None)
                if isinstance(tier, str):
                    return tier
        return None

    def _discover_via_directory(
        self,
        plugin_kind: str,
        plugin_dir: Optional[Path] = None,
        *,
        tier_filter: Optional[str] = None,
    ) -> List[str]:
        """Discover plugins by scanning the plugins directory.

        This is the fallback/development mode discovery that scans for Python
        modules with a create_plugin() factory function and matching PLUGIN_KIND.

        Args:
            plugin_kind: Kind of plugin to discover ('tool', 'gc', etc.).
            plugin_dir: Directory to scan. Defaults to this package's directory.
            tier_filter: Optional tier restriction (Phase 3 §3.3.5).
                When set, the module's ``PLUGIN_TIER`` must match;
                missing or mismatched annotations are skipped.  Read
                directly from the imported module so the pre-§3.3.5
                "annotate or be excluded" contract holds.

        Returns:
            List of discovered plugin names.
        """
        if plugin_dir is None:
            plugin_dir = Path(__file__).parent

        discovered = []

        for finder, name, ispkg in pkgutil.iter_modules([str(plugin_dir)]):
            # Skip internal modules
            if name.startswith('_') or name in ('base', 'registry'):
                continue

            # Skip if already loaded via entry points.  Pre-#684 this
            # was silent, which is exactly how a third-party entry
            # point could take over a built-in without anyone noticing:
            # the built-in module was simply never imported.  The
            # framework declaring its OWN plugins through entry points
            # lands here on every run, so only a foreign incumbent
            # warns.
            if name in self._plugins:
                self._warn_name_taken(
                    name,
                    PluginOrigin(
                        name=name,
                        via="directory",
                        module=f"{_BUILTIN_PLUGIN_PACKAGE}.{name}",
                    ),
                )
                continue

            try:
                t0 = time.perf_counter()
                module = importlib.import_module(f".{name}", package="shared.plugins")
                import_ms = (time.perf_counter() - t0) * 1000

                # Only act on modules that opt in via PLUGIN_KIND matching
                # this scan. Absent PLUGIN_KIND means the module belongs to
                # another registry (formatter_pipeline, JaatoRuntime direct
                # import, etc.) — skip silently, since create_plugin is a
                # convention shared across registries.
                if getattr(module, 'PLUGIN_KIND', None) != plugin_kind:
                    continue

                # Phase 3 §3.3.5: tier filter.  Cross-tier
                # ``"daemon_callable"`` plugins are admitted by BOTH
                # ``"runner"`` and ``"daemon"`` filters — see
                # :func:`_tier_filter_matches`.
                if tier_filter is not None:
                    module_tier = getattr(module, 'PLUGIN_TIER', None)
                    if not _tier_filter_matches(module_tier, tier_filter):
                        _trace(
                            f" Plugin '{name}': PLUGIN_TIER={module_tier!r} "
                            f"not accepted by filter={tier_filter!r}; "
                            f"skipping"
                        )
                        continue

                if hasattr(module, 'create_plugin'):
                    t1 = time.perf_counter()
                    plugin = module.create_plugin()
                    create_ms = (time.perf_counter() - t1) * 1000

                    # Verify protocol implementation.  Failing the
                    # protocol check used to be a debug-only trace, which
                    # made the plugin invisibly disappear from the
                    # registry — same footgun as missing PLUGIN_KIND.
                    # Promote to a real warning that names the missing
                    # methods so the author knows exactly what to add.
                    gap = _protocol_gap(plugin, plugin_kind)
                    if gap is not None:
                        logger.warning(
                            "Plugin '%s' does not implement the %s "
                            "protocol — it will be silently skipped. "
                            "Missing methods: %s. Add them to "
                            "%s/plugin.py.",
                            name,
                            _PLUGIN_KIND_PROTOCOLS[plugin_kind].__name__,
                            gap,
                            name,
                        )
                        continue

                    self._plugins[plugin.name] = plugin
                    self._plugin_sources[plugin.name] = PluginOrigin(
                        name=plugin.name,
                        via="directory",
                        module=module.__name__,
                        entry_point=None,
                    )
                    discovered.append(plugin.name)

                    # Enrichment plugins are always enrichment-only
                    if plugin_kind == "enrichment":
                        self._enrichment_only.add(plugin.name)

                    total_ms = import_ms + create_ms
                    if total_ms > 5.0:  # Log plugins taking >5ms
                        _trace(f" Plugin '{name}' discovery: import={import_ms:.1f}ms "
                               f"create={create_ms:.1f}ms total={total_ms:.1f}ms")
                    self._discovery_timings[name] = {
                        "import_ms": round(import_ms, 2),
                        "create_ms": round(create_ms, 2),
                    }

            except ModuleNotFoundError as exc:
                # A plugin's (declared) dependency is not installed in this
                # environment — e.g. interactive_shell needs pexpect on Unix.
                # The plugin correctly does NOT load; this is an environment-
                # provisioning issue, not a plugin bug, so emit a concise,
                # actionable one-line WARNING naming the missing module rather
                # than dumping a full traceback that reads as a crash mid-scan.
                missing = exc.name or str(exc)
                _trace(
                    f" Plugin '{name}' skipped: missing dependency "
                    f"'{missing}' — install it to enable this plugin.",
                    warning=True,
                )
            except Exception as exc:
                _trace(f" Error loading plugin '{name}': {exc}", include_traceback=True)

        return discovered

    def set_model_name(self, model_name: str) -> None:
        """Set the model name for checking plugin requirements.

        Args:
            model_name: The model name (e.g., 'gemini-3-pro-preview').
        """
        self._model_name = model_name
        # Clear skipped plugins as model changed
        self._skipped_plugins.clear()

    def get_model_name(self) -> Optional[str]:
        """Get the currently configured model name."""
        return self._model_name

    def list_available(self) -> List[str]:
        """List all discovered plugin names."""
        return list(self._plugins.keys())

    def get_plugin_source(self, name: str) -> Optional[PluginOrigin]:
        """Where the plugin registered under *name* came from.

        Provenance is recorded at registration time by both discovery
        paths and by :meth:`register_plugin`, so callers can tell a
        built-in apart from an out-of-tree plugin that claimed the same
        name (issue #684 item 4).

        Args:
            name: Registered plugin name.

        Returns:
            The :class:`PluginOrigin`, or ``None`` for a plugin that
            predates provenance tracking or was never registered.
        """
        return self._plugin_sources.get(name)

    def get_plugin_sources(self) -> Dict[str, PluginOrigin]:
        """Provenance for every registered plugin, by name.

        Returns a copy — callers reporting on the registry must not be
        able to reshape its record of what it loaded.
        """
        return dict(self._plugin_sources)

    def list_exposed(self) -> List[str]:
        """List currently exposed plugin names."""
        return list(self._exposed)

    def all_plugins(self) -> Dict[str, Any]:
        """Snapshot of every DISCOVERED plugin instance, by name.

        Distinct from :meth:`list_exposed` (the subset whose tools are
        surfaced to the model): this is the full set the registry
        discovered and instantiated.  Used by the AppArmor composer, which
        must grant for every plugin the confined runner actually
        initializes — ``expose_all`` (runner/session.py) inits ALL
        discovered runner-tier plugins regardless of ``profile.plugins``
        (the load-gate that would scope init to the declared set was
        disabled in #114), so grants scoped to ``profile.plugins`` leave
        auto-loaded plugins without their host-path rules.
        """
        return dict(self._plugins)

    def is_exposed(self, name: str) -> bool:
        """Check if a plugin's tools are currently exposed to the model."""
        return name in self._exposed

    def get_plugin_config_schema(self, name: str) -> dict:
        """Get the JSON Schema for a plugin's configuration.

        Returns the JSON Schema dict declared by the plugin's
        ``get_config_schema()`` method.  Returns an empty dict
        if the plugin doesn't exist or doesn't implement the method.

        Args:
            name: Plugin name.

        Returns:
            JSON Schema dict describing the plugin's config, or ``{}``.
        """
        plugin = self._plugins.get(name)
        if plugin and hasattr(plugin, 'get_config_schema'):
            return plugin.get_config_schema()
        return {}

    def register_category(self, name: str, description: str) -> None:
        """Register (or update) a tool category description.

        Plugins call this during ``initialize()`` when they introduce
        tools in a category not covered by the built-in list.  The
        introspection plugin reads ``get_category_descriptions()`` to
        populate the ``list_tools`` category summary, so registering
        here ensures new categories get a human-readable description
        instead of an empty string.

        Calling with a ``name`` that already exists overwrites the
        description — last-register-wins.  This lets premium plugins
        refine a built-in category's description if needed.

        Args:
            name: Category name (must match the ``category`` field on
                the plugin's ``ToolSchema`` declarations).
            description: Short human-readable description shown in the
                ``list_tools`` category summary.
        """
        self._category_descriptions[name] = description

    def get_category_descriptions(self) -> Dict[str, str]:
        """Return all registered category descriptions.

        Includes built-in categories seeded at registry creation time
        and any additional categories registered by plugins via
        ``register_category()``.
        """
        return dict(self._category_descriptions)

    def get_plugin(self, name: str) -> Optional[AnyPlugin]:
        """Get a plugin by name, or None if not found.

        Returns either a ToolPlugin or EnrichmentPlugin instance.
        """
        return self._plugins.get(name)

    def register_plugin(
        self,
        plugin: AnyPlugin,
        expose: bool = False,
        enrichment_only: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Manually register a plugin with the registry.

        Use this for plugins that aren't discovered via entry points or directory
        scanning, such as the session plugin which is configured separately.

        This allows the plugin to participate in prompt enrichment and other
        registry-managed features without being discovered.

        Accepts both ``ToolPlugin`` and ``EnrichmentPlugin`` instances.
        ``EnrichmentPlugin`` instances are automatically registered as
        enrichment-only regardless of the ``enrichment_only`` parameter.

        Args:
            plugin: The plugin instance to register (ToolPlugin or EnrichmentPlugin).
            expose: If True, also expose the plugin's tools (calls initialize).
                   Ignored for EnrichmentPlugin instances.
            enrichment_only: If True, only participate in prompt enrichment
                           (not included in get_exposed_tool_schemas/executors).
                           Automatically True for EnrichmentPlugin instances.
            config: Optional configuration dict if exposing.

        Example:
            # Register a ToolPlugin for prompt enrichment only
            registry.register_plugin(session_plugin, enrichment_only=True)

            # Register an EnrichmentPlugin (always enrichment-only)
            registry.register_plugin(my_enrichment_plugin)
        """
        self._plugins[plugin.name] = plugin
        self._plugin_sources[plugin.name] = PluginOrigin(
            name=plugin.name,
            via="registered",
            module=type(plugin).__module__ or "",
        )

        # EnrichmentPlugin instances are always enrichment-only
        if isinstance(plugin, EnrichmentPlugin) and not isinstance(plugin, ToolPlugin):
            self._enrichment_only.add(plugin.name)
        elif enrichment_only:
            self._enrichment_only.add(plugin.name)
        elif expose:
            self.expose_tool(plugin.name, config)

    def register_core_tool(
        self,
        schema: ToolSchema,
        executor: Callable[[Dict[str, Any]], Any],
        auto_approved: bool = False
    ) -> None:
        """Register a core framework tool.

        Core tools are provided directly by framework components (like StreamManager)
        rather than through plugins. They are included in get_exposed_tool_schemas()
        and get_exposed_executors() alongside plugin tools.

        Args:
            schema: The tool's schema.
            executor: The tool's executor function.
            auto_approved: If True, tool is auto-approved (no permission required).

        Example:
            # Register dismiss_stream from StreamManager
            for schema in stream_manager.get_tool_schemas():
                executor = stream_manager.get_executors()[schema.name]
                auto_approved = schema.name in stream_manager.get_auto_approved_tools()
                registry.register_core_tool(schema, executor, auto_approved)
        """
        self._core_tools[schema.name] = schema
        self._core_executors[schema.name] = executor
        if auto_approved:
            self._core_auto_approved.add(schema.name)
        _trace(f"Registered core tool: {schema.name} (auto_approved={auto_approved})")

    def is_core_tool(self, tool_name: str) -> bool:
        """Return True if ``tool_name`` is a framework core tool.

        Core tools are framework machinery registered via
        :meth:`register_core_tool` — the lifecycle terminal
        ``signal_completion``, introspection/event-bus tools, etc. — as
        opposed to a plugin-provided tool (``cli``, ``file_edit``, business
        tools).  Consumers use this to give framework machinery different
        treatment from the business tool surface (e.g. the permission
        pipeline exempts core tools from a catch-all ``"default"`` evaluator
        so a business default-deny can't brick the agent's ability to
        complete).
        """
        return tool_name in self._core_tools

    def get_registered_core_tool_names(self) -> Set[str]:
        """Return the names of tools registered via :meth:`register_core_tool`.

        This is the authoritative "framework machinery" set — stream
        controls, event-bus, introspection, client host tools — mirroring
        :meth:`is_core_tool`.  It is DELIBERATELY narrower than
        :meth:`get_core_tool_schemas`, which returns every exposed *plugin*
        tool whose ``discoverability == 'core'`` (an eager-loading
        performance flag applied to real business tools like ``readFile``
        and the ``todo`` tools).  Consumers that must distinguish framework
        machinery from the business tool surface — e.g. the permission
        pipeline's catch-all ``"default"`` evaluator exemption — must use
        THIS accessor, not ``get_core_tool_schemas()``, or they silently
        exempt powerful plugin tools from that evaluator (see #487/#488).
        """
        return set(self._core_tools.keys())

    def get_core_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Get executors for all registered core framework tools.

        Returns:
            Dict mapping tool names to executor callables.
        """
        return dict(self._core_executors)

    def expose_tool(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Expose a plugin's tools to the model.

        Calls the plugin's initialize() method if this is the first time
        exposing it, or if a new config is provided.

        If a model_name is set and the plugin has model_requirements that
        don't match, the plugin is skipped with a warning.

        Args:
            name: Plugin name to expose.
            config: Optional configuration dict for the plugin.

        Returns:
            True if the plugin was exposed, False if skipped due to model requirements.

        Raises:
            ValueError: If the plugin is not found.
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found. Available: {self.list_available()}")

        plugin = self._plugins[name]

        # Enrichment-only plugins don't provide tools — initialize them and
        # track in _enrichment_only instead of _exposed so that
        # get_exposed_tool_schemas / get_plugin_for_tool never call
        # get_tool_schemas() / get_executors() on them.
        if name in self._enrichment_only:
            if not hasattr(plugin, '_initialized'):
                # Server 0.6.129+: framework-known values
                # (workspace_path, config_root, session_id,
                # agent_name) injected into config via setdefault.
                # See :meth:`_augment_plugin_config` for the contract.
                effective_config = self._augment_plugin_config(config)
                plugin.initialize(effective_config)
                plugin._initialized = True
                if effective_config:
                    self._configs[name] = effective_config
                _trace(f" Enrichment-only plugin '{name}' initialized (not exposed)")
            return True

        # Check model requirements if model_name is set
        if self._model_name and hasattr(plugin, 'get_model_requirements'):
            requirements = plugin.get_model_requirements()
            if requirements and not model_matches_requirements(self._model_name, requirements):
                self._skipped_plugins[name] = requirements
                _trace(f" Plugin '{name}' skipped: "
                      f"model '{self._model_name}' not in {requirements}")
                return False

        # Initialize if not already exposed, or if new config provided
        if name not in self._exposed:
            # Server 0.6.129+: inject framework-known values into the
            # plugin's config before initialize.  See
            # :meth:`_augment_plugin_config`.
            effective_config = self._augment_plugin_config(config)
            t0 = time.perf_counter()
            try:
                plugin.initialize(effective_config)
            except Exception as exc:
                # Don't let one broken plugin take down the whole session.
                # Record the failure and skip exposing the plugin so the
                # rest of expose_all() can continue.
                logger.error(
                    "Plugin '%s' initialize() failed: %s. "
                    "Plugin will not be exposed.",
                    name, exc, exc_info=True,
                )
                self._failed_plugins[name] = ("initialize", str(exc))
                return False
            init_ms = (time.perf_counter() - t0) * 1000
            self._init_timings[name] = {"init_ms": round(init_ms, 2)}
            if init_ms > 5.0:  # Log plugins taking >5ms to initialize
                _trace(f" Plugin '{name}' initialize: {init_ms:.1f}ms")
            if effective_config:
                self._configs[name] = effective_config
            self._exposed.add(name)
            self._tool_plugin_cache.clear()
            # Wire up plugin with registry for authorized external paths
            if hasattr(plugin, 'set_plugin_registry'):
                plugin.set_plugin_registry(self)
                _trace(f" Plugin '{name}' wired with registry")
        elif config and config != self._configs.get(name):
            # Snapshot authorized/denied paths owned by this plugin so that
            # shutdown() + initialize() doesn't lose in-memory-only state
            # (e.g. sandbox paths added by a parent session that aren't in
            # the subagent's session-tier config file).
            prev_authorized = {
                path: (src, access)
                for path, (src, access) in self._authorized_external_paths.items()
                if src == name
            }
            prev_denied = {
                path: src
                for path, src in self._denied_external_paths.items()
                if src == name
            }

            # Re-initialize with new config.  Both shutdown and
            # initialize are wrapped so a failure in one doesn't leave
            # the registry in an inconsistent state.
            try:
                plugin.shutdown()
            except Exception as exc:
                logger.warning(
                    "Plugin '%s' shutdown() during re-init failed: %s. "
                    "Continuing with re-initialize.",
                    name, exc, exc_info=True,
                )
            # Server 0.6.129+: framework-key injection for re-init too.
            effective_config = self._augment_plugin_config(config)
            try:
                plugin.initialize(effective_config)
            except Exception as exc:
                logger.error(
                    "Plugin '%s' initialize() during re-init failed: %s. "
                    "Plugin will be unexposed.",
                    name, exc, exc_info=True,
                )
                self._failed_plugins[name] = ("re-initialize", str(exc))
                self._exposed.discard(name)
                self._configs.pop(name, None)
                self._tool_plugin_cache.clear()
                return False
            self._configs[name] = effective_config
            self._tool_plugin_cache.clear()
            # Re-wire after re-initialization
            if hasattr(plugin, 'set_plugin_registry'):
                plugin.set_plugin_registry(self)
                _trace(f" Plugin '{name}' re-wired with registry")

            # Re-broadcast workspace_path / config_root onto the
            # re-initialized plugin.  These are normally pushed by
            # ``set_workspace_path`` / ``set_config_root`` broadcast
            # methods that fire only on registry-level state changes —
            # not on per-plugin re-init.  Without this replay, a
            # subagent that re-exposes a plugin with a tweaked config
            # (e.g. agent_name injected) silently rebuilds the plugin
            # with no awareness of the broadcasted overrides, causing
            # path-anchored state (schema_store base, config-anchored
            # caches, etc.) to drift back to defaults.
            if hasattr(plugin, 'set_workspace_path') and self._workspace_path:
                try:
                    plugin.set_workspace_path(self._workspace_path)
                    _trace(f" Plugin '{name}' replayed set_workspace_path")
                except Exception as exc:
                    _trace(f" Plugin '{name}' set_workspace_path replay failed: {exc}")
            if (
                hasattr(plugin, 'set_config_root')
                and getattr(self, '_config_root', None) is not None
            ):
                try:
                    plugin.set_config_root(self._config_root)
                    _trace(f" Plugin '{name}' replayed set_config_root")
                except Exception as exc:
                    _trace(f" Plugin '{name}' set_config_root replay failed: {exc}")

            # Restore any authorized/denied paths that were lost during
            # re-initialization (the plugin's config reload may not have
            # all tiers — e.g. subagent has no parent's session config).
            for path, (src, access) in prev_authorized.items():
                if path not in self._authorized_external_paths:
                    self._authorized_external_paths[path] = (src, access)
                    _trace(f" Restored authorized path after re-init: {path}")
            for path, src in prev_denied.items():
                if path not in self._denied_external_paths:
                    self._denied_external_paths[path] = src
                    _trace(f" Restored denied path after re-init: {path}")

        return True

    def unexpose_tool(self, name: str) -> None:
        """Stop exposing a plugin's tools to the model.

        Calls the plugin's shutdown() method to clean up resources.
        Shutdown failures are caught and logged so they don't leave
        the registry in an inconsistent state — the plugin is always
        removed from the exposed set regardless of whether shutdown
        succeeded.

        Args:
            name: Plugin name to unexpose.
        """
        if name in self._exposed:
            try:
                self._plugins[name].shutdown()
            except Exception as exc:
                logger.warning(
                    "Plugin '%s' shutdown() failed: %s. "
                    "Continuing with unexpose.",
                    name, exc, exc_info=True,
                )
                self._failed_plugins[name] = ("shutdown", str(exc))
            # Always discard from exposed set so the registry stays
            # consistent even if shutdown raised.
            self._exposed.discard(name)
            self._configs.pop(name, None)
            self._tool_plugin_cache.clear()

    # Plugins that must always initialize regardless of the session's
    # requested plugin set.  ``introspection`` provides the
    # ``list_tools`` / ``get_tool_schemas`` tools that the
    # deferred-loading mechanism depends on; ``permission`` is the
    # framework's policy engine and is wired even when not in
    # profile.plugins (the runner-side bootstrap constructs it
    # separately, but the registry path should keep it included for
    # symmetry with other call sites).  Add other unconditionally-
    # essential plugins here only with care — anything in this set
    # pays its initialize cost for EVERY session regardless of
    # profile.
    _ALWAYS_INITIALIZE_PLUGINS = frozenset({
        "introspection",
        "permission",
    })

    def expose_all(
        self,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        requested_plugins: Optional[List[str]] = None,
    ) -> None:
        """Expose all discovered plugins' tools.

        Plugins that declare ``PARALLEL_INIT = True`` have their
        ``initialize()`` started in background threads so their I/O
        (e.g. MCP server connections) overlaps with the sequential
        initialization of other plugins.

        Args:
            config: Optional dict mapping plugin names to their configs.
            on_progress: Optional callback invoked with each plugin name
                before it is exposed.  Used by the server to emit
                per-plugin init progress events.
            requested_plugins: When provided, only initializes plugins
                whose name is in this list — PLUS the unconditionally-
                essential set (:attr:`_ALWAYS_INITIALIZE_PLUGINS`) AND
                all enrichment-only plugins (their enrichment fires
                for every session regardless of profile.plugins).
                Other discovered plugins remain registered in
                ``self._plugins`` but get NEITHER ``initialize()``
                NOR tool exposure — saving their per-session cost.

                ``None`` (default) preserves pre-2026-05-15 behaviour:
                every discovered plugin is initialized and exposed.

                Motivated by the runner-side bootstrap measurement
                showing the ``references`` plugin's
                ``SentenceTransformer`` model load took ~7.6s on
                EVERY session — even when ``references`` wasn't in
                ``profile.plugins`` — because ``expose_all`` ran
                ``initialize()`` unconditionally.  Filed as the
                architectural fix for the references-init bottleneck;
                see ``project_backlog_parallel_tool_main_thread_bottleneck``
                lineage.
        """
        import concurrent.futures

        config = config or {}

        # Resolve the effective initialize set.
        #
        # ``None`` ⇒ initialize every discovered plugin (back-compat
        # for daemon-side ``server/core.py`` callers + tests that
        # don't pass the kwarg).
        #
        # Non-None ⇒ union of:
        #   1. caller's requested plugins,
        #   2. ``_ALWAYS_INITIALIZE_PLUGINS`` (introspection +
        #      permission today),
        #   3. all enrichment-only plugins (their enrichment fires
        #      regardless of profile.plugins, so they need their
        #      ``initialize()`` to set up subscriptions).
        # Anything not in this union is left registered-but-
        # uninitialized: ``self._plugins[name]`` still holds the
        # instance, but ``expose_tool()`` doesn't fire for it, so
        # neither its tools nor its ``initialize()`` side-effects
        # reach the session.
        if requested_plugins is None:
            target_names: Optional[set] = None  # sentinel: "all"
        else:
            target_names = (
                set(requested_plugins)
                | set(self._ALWAYS_INITIALIZE_PLUGINS)
                | set(self._enrichment_only)
            )

        # Identify plugins that opt into parallel init (I/O-heavy, like MCP)
        parallel_names = [
            name for name, plugin in self._plugins.items()
            if getattr(plugin, 'PARALLEL_INIT', False)
            and name not in self._exposed
            and (target_names is None or name in target_names)
        ]

        # Pre-initialize I/O-heavy plugins in background threads so their
        # network I/O overlaps with the sequential init of other plugins.
        futures: Dict[str, concurrent.futures.Future] = {}
        if parallel_names:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(parallel_names),
                thread_name_prefix="plugin-init",
            )
            for name in parallel_names:
                plugin = self._plugins[name]
                # Server 0.6.129+: framework-key injection also for
                # parallel-init path (PARALLEL_INIT plugins like MCP).
                # See :meth:`_augment_plugin_config`.
                cfg = self._augment_plugin_config(config.get(name))
                _trace(f"Starting parallel init for plugin '{name}'")
                futures[name] = executor.submit(plugin.initialize, cfg)
            executor.shutdown(wait=False)

        # Run the normal sequential expose loop.  For plugins that were
        # pre-initialized above, expose_tool() will find them already
        # initialized (their initialize() is idempotent) and skip the
        # initialization step.
        skipped: List[str] = []
        for name in self._plugins:
            if target_names is not None and name not in target_names:
                # Plugin is discovered + registered but not requested
                # by this session — skip both initialize() and tool
                # exposure.  The plugin instance survives in
                # ``self._plugins`` so introspection still works, but
                # its tools won't appear in the session's tool list
                # and its per-session ``initialize()`` cost is avoided.
                skipped.append(name)
                continue
            # If this plugin is being pre-initialized, wait for it first
            if name in futures:
                try:
                    futures[name].result(timeout=30.0)
                except Exception as exc:
                    _trace(f"Parallel init for '{name}' failed: {exc}")
            if on_progress:
                on_progress(name)
            self.expose_tool(name, config.get(name))

        if skipped:
            _trace(
                f"expose_all: skipped initialize for {len(skipped)} "
                f"plugins not requested by session: {sorted(skipped)}"
            )

    def unexpose_all(self) -> None:
        """Stop exposing all plugins' tools."""
        for name in list(self._exposed):
            self.unexpose_tool(name)

    def collect_prerequisite_policies(self) -> list:
        """Collect prerequisite policies from all exposed plugins.

        Iterates over all exposed (and enrichment-only) plugins, calling
        ``get_prerequisite_policies()`` on those that implement it. Returns
        a flat list of PrerequisitePolicy objects that can be registered
        with the ReliabilityPlugin.

        This follows the same ``hasattr()``-based duck-typing pattern used
        for enrichment subscriptions.

        Returns:
            List of PrerequisitePolicy objects from all plugins.
        """
        policies = []
        all_names = self._exposed | getattr(self, '_enrichment_only', set())
        for name in all_names:
            try:
                plugin = self._plugins[name]
                if hasattr(plugin, 'get_prerequisite_policies'):
                    plugin_policies = plugin.get_prerequisite_policies()
                    if plugin_policies:
                        # Tag each policy with its owner plugin name
                        for p in plugin_policies:
                            if hasattr(p, 'owner_plugin') and not p.owner_plugin:
                                p.owner_plugin = name
                        policies.extend(plugin_policies)
                        _trace(f"Collected {len(plugin_policies)} prerequisite policies from '{name}'")
            except Exception as exc:
                _trace(f"Error collecting prerequisite policies from '{name}': {exc}")
        return policies

    def set_workspace_path(self, path: str) -> None:
        """Set workspace path and broadcast to all plugins that support it.

        Plugins implementing set_workspace_path(path) will be notified.
        This includes: lsp, mcp, file_edit, cli, and any other plugins
        that need to know the workspace root.

        Args:
            path: Absolute path to the workspace root directory.
        """
        self._workspace_path = path
        _trace(f"set_workspace_path: {path}")

        # Broadcast to ALL registered plugins that support it — not just the
        # ones exposed to the model.  A runner-tier plugin can need the per-
        # session workspace before/without model exposure (e.g. a pool-slot
        # runner whose plugin was initialized at template time, or the notebook
        # subprocess backend that spawns kernels rooted at the workspace).  The
        # exposed-only scope was a #344-class propagation gap: registered-but-
        # unexposed plugins silently kept their init-time (launch-dir) default.
        for name, plugin in self._plugins.items():
            if plugin and hasattr(plugin, 'set_workspace_path'):
                try:
                    plugin.set_workspace_path(path)
                    _trace(f"  -> {name}.set_workspace_path()")
                except Exception as exc:
                    _trace(f"  -> {name}.set_workspace_path() failed: {exc}")

    def get_workspace_path(self) -> Optional[str]:
        """Get the current workspace path.

        Returns:
            The workspace path, or None if not set.
        """
        return self._workspace_path

    def set_config_root(self, path: Optional[str]) -> None:
        """Set the read-only framework-config root override and broadcast.

        Plugins that implement ``set_config_root(path)`` will be
        notified.  Plugins without that method are skipped (parity
        with :meth:`set_workspace_path`).  When ``path`` is ``None``,
        plugins fall back to ``<workspace_path>/.jaato/`` for config
        reads (today's behavior).

        Args:
            path: Absolute path to the config root, or ``None`` to
                clear any prior override.
        """
        self._config_root: Optional[str] = path
        _trace(f"set_config_root: {path}")

        # Parity with set_workspace_path: broadcast to ALL registered plugins,
        # not just exposed ones (same #344-class propagation gap).
        for name, plugin in self._plugins.items():
            if plugin and hasattr(plugin, 'set_config_root'):
                try:
                    plugin.set_config_root(path)
                    _trace(f"  -> {name}.set_config_root()")
                except Exception as exc:
                    _trace(f"  -> {name}.set_config_root() failed: {exc}")

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Set the session_id injected into plugin configs at init.

        Server 0.6.129+: matches :meth:`set_workspace_path` /
        :meth:`set_config_root` shape but doesn't broadcast — session_id
        doesn't change mid-session, so no broadcast hook needed.  The
        value is read by :meth:`_augment_plugin_config` at
        :meth:`expose_tool` time.

        Args:
            session_id: Session identifier, or ``None`` to clear.
        """
        self._session_id = session_id

    @property
    def session_id(self) -> Optional[str]:
        """The daemon session this registry belongs to, or ``None``.

        A ``PluginRegistry`` is constructed per :class:`JaatoServer`, and
        a ``JaatoServer`` is per session — so this identifies the ONE
        session whose plugins this registry holds.  ``JaatoServer``
        stamps it during initialization (``core.py``, right after the
        registry is built).

        **Why a getter.**  ``set_session_id`` existed with no reader
        outside :meth:`_augment_plugin_config`, so a daemon-side plugin
        answering a FORWARDED call had no supported way to learn which
        session was asking.  ``daemon.plugin_execute`` ships
        ``plugin_name``, ``tool_name`` and ``args`` — no caller identity
        — so a plugin like ``subagent.list_siblings``, whose entire job
        is "tell me about the cascade around ME", could not name the
        caller and failed every forwarded call.

        The identity was never missing from the daemon, only from the
        CALL: it is one attribute away on the registry the plugin
        already holds (via ``set_plugin_registry``).

        Depends on the per-session invariant above; see
        ``test_registry_session_id_is_per_session``.
        """
        return self._session_id

    def set_agent_name(self, agent_name: Optional[str]) -> None:
        """Set the agent_name injected into plugin configs at init.

        Server 0.6.129+: mirrors :meth:`set_session_id`.  Read by
        plugins that emit trace logs scoped to the agent identity
        (e.g. ``file_edit``, ``memory``).
        """
        self._agent_name = agent_name

    def _augment_plugin_config(
        self, config: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Pre-populate plugin config with framework-known values.

        Server 0.6.129+ structural fix.  Replaces the prior per-call-
        site threading of framework values (``workspace_path``,
        ``config_root``, ``session_id``, ``agent_name``) into
        plugin_configs at ``core.py:1730`` (daemon-side) AND
        ``server/runner/session.py:_bootstrap_session`` (runner-side).
        Both sites had to manually populate these for every framework
        concept that crossed the boundary — the "12 sites per
        concept" pattern documented in
        ``project_backlog_profile_field_propagation_duplication``.

        Now: callers (daemon, runner, in-process subagent spawn) set
        framework values on the registry via :meth:`set_workspace_path`
        / :meth:`set_config_root` / :meth:`set_session_id` /
        :meth:`set_agent_name` BEFORE calling
        :meth:`expose_all` or :meth:`expose_tool`.  This method
        augments the per-plugin config dict at every
        ``plugin.initialize`` call site so each plugin sees the
        framework values in its config without each caller having to
        thread them per-plugin.

        Operator-supplied values win (``setdefault`` semantics) —
        explicit ``plugin_configs.<plugin>.workspace_path`` in a
        profile YAML overrides the framework-known value.

        Returns the augmented config dict, or the original
        ``config`` if no framework values are set yet (preserving
        the pre-fix behavior when callers bypass the setters).
        """
        framework_keys: Dict[str, Any] = {}
        if self._workspace_path is not None:
            framework_keys["workspace_path"] = self._workspace_path
        if self._config_root is not None:
            framework_keys["config_root"] = self._config_root
        if self._session_id is not None:
            framework_keys["session_id"] = self._session_id
        if self._agent_name is not None:
            framework_keys["agent_name"] = self._agent_name

        if not framework_keys:
            return config

        augmented = dict(config) if config else {}
        for k, v in framework_keys.items():
            augmented.setdefault(k, v)
        return augmented

    def get_config_root(self) -> Optional[str]:
        """Get the current read-only framework-config root override.

        Returns:
            The config_root override, or ``None`` if no override is in
            effect (the default — loaders fall back to
            ``<workspace_path>/.jaato/``).
        """
        return getattr(self, '_config_root', None)

    def broadcast_history_cleared(self) -> None:
        """Notify all plugins that the session history has been cleared.

        Plugins implementing ``on_history_cleared()`` use this hook to
        reset per-session enrichment tracking (e.g., which memory/
        template/reference hints were already injected).  Without it,
        enrichment plugins would never re-surface their hints after a
        ``reset`` command, because their dedup state would still reflect
        the wiped conversation.

        Includes enrichment-only plugins since the main dedup consumers
        (memory, references, template) live in either bucket depending on
        how they were discovered.
        """
        _trace("broadcast_history_cleared")
        for name in self._exposed | self._enrichment_only:
            plugin = self._plugins.get(name)
            if plugin and hasattr(plugin, 'on_history_cleared'):
                try:
                    plugin.on_history_cleared()
                    _trace(f"  -> {name}.on_history_cleared()")
                except Exception as exc:
                    _trace(f"  -> {name}.on_history_cleared() failed: {exc}")

    def get_bootstrap_timings(self) -> Dict[str, dict]:
        """Get per-plugin bootstrap timing data.

        Returns:
            Dict mapping plugin names to timing dicts with keys:
            ``import_ms``, ``create_ms`` (from discovery) and
            ``init_ms`` (from expose_tool/initialize).
        """
        result = {}
        for name in sorted(self._plugins.keys()):
            entry: dict = {}
            if name in self._discovery_timings:
                entry.update(self._discovery_timings[name])
            if name in self._init_timings:
                entry.update(self._init_timings[name])
            if entry:
                entry["total_ms"] = round(
                    entry.get("import_ms", 0) + entry.get("create_ms", 0) + entry.get("init_ms", 0),
                    2,
                )
                result[name] = entry
        return result

    def get_exposed_tool_schemas(
        self, exclude_runner_tier: bool = False,
    ) -> List[ToolSchema]:
        """Get ToolSchemas from all exposed plugins and core tools.

        ``exclude_runner_tier=True`` skips ``PLUGIN_TIER = "runner"``
        plugins.  Used by the daemon-side tool-id-registry emit, whose
        runner-tier tool names come from the runner via
        ``session_get_tool_schemas`` — NOT a daemon-side walk.  Invoking a
        runner-tier plugin's ``get_tool_schemas`` daemon-side runs its
        filesystem discovery ON THE DAEMON (e.g. ``prompt_library``'s
        prompt walk); on the event-loop thread during re-attach that
        blocked the loop ~15s and self-blocked the register-RPC send.
        """
        schemas = []
        # Add plugin tool schemas
        for name in self._exposed:
            if exclude_runner_tier:
                plugin = self._plugins.get(name)
                tier = self._lookup_module_tier(
                    getattr(type(plugin), "__module__", "")
                ) if plugin is not None else None
                if tier == "runner":
                    continue
            try:
                schemas.extend(self._plugins[name].get_tool_schemas())
            except Exception as exc:
                _trace(f" Error getting tool schemas from '{name}': {exc}", include_traceback=True)
        # Add core tool schemas
        schemas.extend(self._core_tools.values())
        return schemas

    def get_exposed_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Get executor callables from all exposed plugins and core tools."""
        executors = {}
        # Add plugin executors
        for name in self._exposed:
            try:
                executors.update(self._plugins[name].get_executors())
            except Exception as exc:
                _trace(f" Error getting executors from '{name}': {exc}", include_traceback=True)
        # Add core tool executors
        executors.update(self._core_executors)
        return executors

    # ==================== Individual Tool Management ====================

    def get_all_tool_names(self) -> List[str]:
        """Get names of all tools from exposed plugins.

        Returns:
            List of all tool names (regardless of enabled/disabled state).
        """
        names = []
        for name in self._exposed:
            try:
                schemas = self._plugins[name].get_tool_schemas()
                names.extend(schema.name for schema in schemas)
            except Exception as exc:
                _trace(f" Error getting tool names from '{name}': {exc}", include_traceback=True)
        return names

    def get_tool_traits(self, tool_name: str) -> "FrozenSet[str]":
        """Return the trait set declared on a tool's schema.

        Iterates exposed plugins to find the matching tool name and returns
        its ``traits`` frozenset.  Returns an empty frozenset when the tool
        is not found or the schema has no traits.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The tool's declared traits, or ``frozenset()`` if not found.
        """
        for name in self._exposed:
            try:
                schemas = self._plugins[name].get_tool_schemas()
                for schema in schemas:
                    if schema.name == tool_name:
                        return schema.traits
            except Exception:
                continue
        return frozenset()

    def disable_tool(self, tool_name: str) -> bool:
        """Disable a specific tool by name.

        Disabled tools are not exposed to the model but remain available
        for re-enabling later.

        Args:
            tool_name: Name of the tool to disable.

        Returns:
            True if tool was found and disabled, False if not found.
        """
        all_tools = self.get_all_tool_names()
        if tool_name not in all_tools:
            return False
        self._disabled_tools.add(tool_name)
        return True

    def enable_tool(self, tool_name: str) -> bool:
        """Enable a previously disabled tool.

        Args:
            tool_name: Name of the tool to enable.

        Returns:
            True if tool was found and enabled, False if not found.
        """
        all_tools = self.get_all_tool_names()
        if tool_name not in all_tools:
            return False
        self._disabled_tools.discard(tool_name)
        return True

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is currently enabled.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is enabled, False if disabled or not found.
        """
        return tool_name not in self._disabled_tools

    def disable_all_tools(self) -> int:
        """Disable all tools from exposed plugins.

        Returns:
            Number of tools disabled.
        """
        all_tools = self.get_all_tool_names()
        self._disabled_tools = set(all_tools)
        return len(all_tools)

    def enable_all_tools(self) -> int:
        """Enable all tools (clear all disabled).

        Returns:
            Number of tools that were re-enabled.
        """
        count = len(self._disabled_tools)
        self._disabled_tools.clear()
        return count

    def get_tool_status(self) -> List[Dict[str, Any]]:
        """Get status of all tools from exposed plugins.

        Returns:
            List of dicts with 'name', 'description', 'enabled', 'plugin' keys.
        """
        status = []
        for plugin_name in self._exposed:
            try:
                schemas = self._plugins[plugin_name].get_tool_schemas()
                for schema in schemas:
                    status.append({
                        'name': schema.name,
                        'description': schema.description,
                        'enabled': schema.name not in self._disabled_tools,
                        'plugin': plugin_name,
                    })
            except Exception as exc:
                _trace(f" Error getting tool status from '{plugin_name}': {exc}", include_traceback=True)
        return status

    def get_enabled_tool_schemas(self) -> List[ToolSchema]:
        """Get ToolSchemas from exposed plugins and core tools, excluding disabled.

        This is what should be sent to the model - only enabled tools.
        For plugins implementing StreamingCapable, auto-generates -stream
        variants for tools that support streaming.

        Returns:
            List of ToolSchema objects for enabled tools only.
        """
        schemas = []
        for name in self._exposed:
            try:
                plugin = self._plugins[name]
                plugin_schemas = plugin.get_tool_schemas()
                # Filter out disabled tools
                enabled_schemas = [s for s in plugin_schemas if s.name not in self._disabled_tools]
                schemas.extend(enabled_schemas)

                # Auto-generate -stream variants for streaming-capable plugins
                if isinstance(plugin, StreamingCapable):
                    for schema in enabled_schemas:
                        if plugin.supports_streaming(schema.name):
                            stream_schema = self._create_streaming_schema(schema)
                            if stream_schema.name not in self._disabled_tools:
                                schemas.append(stream_schema)
            except Exception as exc:
                _trace(f" Error getting tool schemas from '{name}': {exc}", include_traceback=True)

        # Add core tool schemas (excluding disabled)
        for name, schema in self._core_tools.items():
            if name not in self._disabled_tools:
                schemas.append(schema)

        return schemas

    def _create_streaming_schema(self, base_schema: ToolSchema) -> ToolSchema:
        """Create a -stream variant of a tool schema.

        The streaming variant has the same parameters but different behavior -
        it returns immediately with initial chunks and continues streaming
        in the background.

        Inherits category and discoverability from the base schema.  The
        suffix is ``-stream`` (server 0.6.65+) — was ``:stream`` previously,
        but the colon failed strict upstream tool-name regexes
        (Anthropic / Bedrock / OpenAI all enforce ``^[a-zA-Z0-9_-]{1,128}$``).
        Hyphen is universally regex-safe.

        Args:
            base_schema: The base tool schema to create a streaming variant for.

        Returns:
            A new ToolSchema for the -stream variant.
        """
        streaming_description = (
            f"{base_schema.description} "
            f"(STREAMING MODE: Returns immediately with initial results and stream_id. "
            f"More results will be automatically injected as they become available. "
            f"Call dismiss_stream(stream_id) when you have enough results.)"
        )
        return ToolSchema(
            name=f"{base_schema.name}-stream",
            description=streaming_description,
            parameters=base_schema.parameters,
            category=base_schema.category,
            discoverability=base_schema.discoverability,
        )

    def get_core_tool_schemas(self) -> List[ToolSchema]:
        """Get ToolSchemas for 'core' tools only (always loaded in context).

        This supports deferred tool loading by returning only tools marked
        with discoverability='core'. Other tools can be discovered via
        introspection (list_tools, get_tool_schemas).

        For plugins implementing StreamingCapable, auto-generates -stream
        variants for core tools that support streaming.

        Returns:
            List of ToolSchema objects for core tools only.
        """
        schemas = []
        for name in self._exposed:
            try:
                plugin = self._plugins[name]
                plugin_schemas = plugin.get_tool_schemas()
                # Filter to core, enabled tools only
                core_schemas = [
                    s for s in plugin_schemas
                    if s.name not in self._disabled_tools
                    and getattr(s, 'discoverability', DISCOVERABILITY_DEFERRED) == DISCOVERABILITY_EAGER
                ]
                schemas.extend(core_schemas)

                # Auto-generate -stream variants for streaming-capable plugins (core tools)
                if isinstance(plugin, StreamingCapable):
                    for schema in core_schemas:
                        if plugin.supports_streaming(schema.name):
                            stream_schema = self._create_streaming_schema(schema)
                            if stream_schema.name not in self._disabled_tools:
                                schemas.append(stream_schema)
            except Exception as exc:
                _trace(f" Error getting tool schemas from '{name}': {exc}", include_traceback=True)

        # Add core tool schemas that have discoverability='core' (excluding disabled)
        for name, schema in self._core_tools.items():
            if name not in self._disabled_tools:
                if getattr(schema, 'discoverability', DISCOVERABILITY_DEFERRED) == DISCOVERABILITY_EAGER:
                    schemas.append(schema)

        return schemas

    def get_enabled_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Get executor callables from exposed plugins and core tools, excluding disabled.

        Returns:
            Dict mapping tool names to executor callables for enabled tools only.
        """
        executors = {}
        for name in self._exposed:
            try:
                plugin_executors = self._plugins[name].get_executors()
                # Filter out disabled tools
                for tool_name, executor in plugin_executors.items():
                    if tool_name not in self._disabled_tools:
                        executors[tool_name] = executor
            except Exception as exc:
                _trace(f" Error getting executors from '{name}': {exc}", include_traceback=True)

        # Add core tool executors (excluding disabled)
        for tool_name, executor in self._core_executors.items():
            if tool_name not in self._disabled_tools:
                executors[tool_name] = executor

        return executors

    def list_disabled_tools(self) -> List[str]:
        """List all currently disabled tool names.

        Returns:
            List of disabled tool names.
        """
        return list(self._disabled_tools)

    # ==================== Streaming Tool Support ====================

    def is_streaming_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a streaming variant.

        Args:
            tool_name: Tool name to check (e.g., "grep_content-stream").

        Returns:
            True if this is a -stream variant.
        """
        return tool_name.endswith("-stream")

    def get_base_tool_name(self, tool_name: str) -> str:
        """Get the base tool name from a streaming variant.

        Args:
            tool_name: Tool name (e.g., "grep_content-stream").

        Returns:
            Base tool name (e.g., "grep_content").
        """
        if self.is_streaming_tool(tool_name):
            return tool_name[:-7]  # Remove "-stream" suffix
        return tool_name

    def get_streaming_plugin(self, tool_name: str) -> Optional[StreamingCapable]:
        """Get the streaming-capable plugin for a tool.

        Args:
            tool_name: Tool name (base or :stream variant).

        Returns:
            The StreamingCapable plugin, or None if not found or not streaming-capable.
        """
        base_name = self.get_base_tool_name(tool_name)

        for plugin_name in self._exposed:
            try:
                plugin = self._plugins[plugin_name]
                if base_name in plugin.get_executors():
                    if isinstance(plugin, StreamingCapable):
                        return plugin
                    return None
            except Exception:
                pass
        return None

    def get_streaming_tools(self) -> List[str]:
        """Get list of all tools that support streaming.

        Returns:
            List of base tool names (without :stream suffix) that support streaming.
        """
        streaming_tools = []
        for plugin_name in self._exposed:
            try:
                plugin = self._plugins[plugin_name]
                if isinstance(plugin, StreamingCapable):
                    for schema in plugin.get_tool_schemas():
                        if plugin.supports_streaming(schema.name):
                            streaming_tools.append(schema.name)
            except Exception as exc:
                _trace(f" Error getting streaming tools from '{plugin_name}': {exc}")
        return streaming_tools

    def plugin_has_core_tools(self, plugin_name: str) -> bool:
        """Check whether a plugin has at least one tool with discoverability='core'.

        Used by the runtime and session to decide whether a plugin's system
        instructions should be included in the initial model context when
        deferred tool loading is enabled.  Plugins whose tools are **all**
        discoverable should have their instructions deferred until the model
        actually discovers one of their tools.

        Args:
            plugin_name: Name of the plugin to check.

        Returns:
            True if the plugin provides at least one core tool, False otherwise.
        """
        plugin = self._plugins.get(plugin_name)
        if not plugin or not hasattr(plugin, 'get_tool_schemas'):
            return False
        try:
            for schema in plugin.get_tool_schemas():
                if getattr(schema, 'discoverability', DISCOVERABILITY_DEFERRED) == DISCOVERABILITY_EAGER:
                    return True
        except Exception:
            pass
        return False

    def get_system_instructions(
        self,
        run_enrichment: bool = True,
        skip_discoverable_only: bool = False,
    ) -> Optional[str]:
        """Combine system instructions from all exposed plugins.

        Collects raw system instructions from all exposed plugins, then
        optionally runs them through the system instruction enrichment
        pipeline (e.g., template extraction).

        Args:
            run_enrichment: If True (default), run system instruction
                enrichment plugins after combining raw instructions.
            skip_discoverable_only: If True, skip plugins whose tools are
                all discoverable (no core tools).  Used when deferred tool
                loading is enabled so that instructions for undiscovered
                tools do not pollute the initial model context.

        Returns:
            Combined (and optionally enriched) system instructions string,
            or None if no plugins have instructions.
        """
        instructions = []
        for name in self._exposed:
            try:
                if skip_discoverable_only and not self.plugin_has_core_tools(name):
                    continue
                plugin_instructions = self._plugins[name].get_system_instructions()
                if plugin_instructions:
                    instructions.append(plugin_instructions)
            except Exception as exc:
                _trace(f" Error getting system instructions from '{name}': {exc}", include_traceback=True)

        if not instructions:
            return None

        combined = "\n\n".join(instructions)

        # Run through enrichment pipeline if enabled
        if run_enrichment:
            result = self.enrich_system_instructions(combined)
            return result.instructions

        return combined

    def get_auto_approved_tools(self) -> List[str]:
        """Collect auto-approved tool names from all exposed plugins and core tools.

        Returns:
            List of tool names that should be whitelisted for permission checks.
        """
        tools = []
        # Include core auto-approved tools
        tools.extend(self._core_auto_approved)
        # Include plugin auto-approved tools
        for name in self._exposed:
            try:
                if hasattr(self._plugins[name], 'get_auto_approved_tools'):
                    auto_approved = self._plugins[name].get_auto_approved_tools()
                    if auto_approved:
                        tools.extend(auto_approved)
            except Exception as exc:
                _trace(f" Error getting auto-approved tools from '{name}': {exc}", include_traceback=True)
        return tools

    def get_exposed_user_commands(self) -> List[UserCommand]:
        """Collect user-facing commands from all exposed plugins.

        User commands are different from model tools - they are commands
        that users can type directly in the interactive client.

        Each UserCommand includes:
        - name: Command name for invocation and autocompletion
        - description: Brief description shown in autocompletion/help
        - share_with_model: If True, output is added to conversation history

        Returns:
            List of UserCommand objects from all exposed plugins.
        """
        commands: List[UserCommand] = []
        for name in self._exposed:
            try:
                if hasattr(self._plugins[name], 'get_user_commands'):
                    user_commands = self._plugins[name].get_user_commands()
                    if user_commands:
                        commands.extend(user_commands)
            except Exception as exc:
                _trace(f" Error getting user commands from '{name}': {exc}", include_traceback=True)
        return commands

    def get_plugin_for_tool(self, tool_name: str) -> Optional['ToolPlugin']:
        """Get the plugin that provides a specific tool.

        Uses a lazy cache (``_tool_plugin_cache``) that is populated on first
        lookup and invalidated when plugins are exposed/unexposed.  For dynamic
        plugins (e.g. MCP after ``mcp reload``), a cache hit is validated by
        checking the tool still exists in the plugin's executors; a stale entry
        is evicted and the full scan re-runs.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The ToolPlugin instance that provides this tool, or None if not found.
        """
        # Fast path: cached lookup
        cached = self._tool_plugin_cache.get(tool_name)
        if cached is not None:
            # Validate the cache entry — the plugin may have dropped this tool
            # (e.g. MCP reload removed a server).
            if tool_name in cached.get_executors():
                return cached
            # Stale entry — evict and fall through to full scan
            del self._tool_plugin_cache[tool_name]

        # Slow path: scan all exposed plugins
        _trace(f" get_plugin_for_tool: cache miss for '{tool_name}', scanning {len(self._exposed)} plugins")
        for name in self._exposed:
            try:
                plugin = self._plugins[name]
                executors = plugin.get_executors()
                if tool_name in executors:
                    _trace(f" get_plugin_for_tool: FOUND '{tool_name}' in plugin '{name}'")
                    self._tool_plugin_cache[tool_name] = plugin
                    return plugin
            except Exception as exc:
                _trace(f" get_plugin_for_tool: error getting executors from '{name}': {exc}", include_traceback=True)
        _trace(f" get_plugin_for_tool: '{tool_name}' NOT FOUND in any plugin")
        return None

    def invalidate_tool_cache(self) -> None:
        """Clear the tool-to-plugin lookup cache.

        Call this when a plugin's executor set changes outside the normal
        expose/unexpose lifecycle (e.g. after ``mcp reload``).
        """
        self._tool_plugin_cache.clear()

    def list_skipped_plugins(self) -> Dict[str, List[str]]:
        """List plugins that were skipped due to model requirements.

        Returns:
            Dict mapping plugin names to their required model patterns.
        """
        return dict(self._skipped_plugins)

    # ==================== External Path Authorization ====================

    def authorize_external_path(
        self,
        path: str,
        source_plugin: str,
        access: str = "readwrite"
    ) -> None:
        """Authorize an external path for model access.

        Plugins (like references) can use this to allow the model to access
        specific files outside the workspace that are needed for the task.

        Args:
            path: Absolute path to authorize (will be normalized).
            source_plugin: Name of the plugin granting authorization.
            access: Access mode - "readonly" for read-only, "readwrite" for
                   full access (default: "readwrite").
        """
        if access not in ("readonly", "readwrite"):
            raise ValueError(f"Invalid access mode: {access!r}. Must be 'readonly' or 'readwrite'.")
        # Normalize to absolute path
        normalized = os.path.realpath(os.path.abspath(path))
        self._authorized_external_paths[normalized] = (source_plugin, access)
        _trace(f"authorize_external_path: {normalized} (from {source_plugin}, access={access})")

    def authorize_external_paths(
        self,
        paths: List[str],
        source_plugin: str,
        access: str = "readwrite"
    ) -> None:
        """Authorize multiple external paths for model access.

        Args:
            paths: List of absolute paths to authorize.
            source_plugin: Name of the plugin granting authorization.
            access: Access mode - "readonly" for read-only, "readwrite" for
                   full access (default: "readwrite").
        """
        for path in paths:
            self.authorize_external_path(path, source_plugin, access=access)

    def deauthorize_external_path(self, path: str, source_plugin: str) -> bool:
        """Remove authorization for a single external path.

        Only removes the path if it was authorized by the specified plugin.
        This is the single-path counterpart to clear_authorized_paths().

        Args:
            path: Absolute path to deauthorize (will be normalized).
            source_plugin: Name of the plugin that originally authorized it.
                Only removes if the current authorization matches this plugin.

        Returns:
            True if the path was found and removed, False otherwise.
        """
        normalized = os.path.realpath(os.path.abspath(path))
        entry = self._authorized_external_paths.get(normalized)
        if entry and entry[0] == source_plugin:
            del self._authorized_external_paths[normalized]
            _trace(f"deauthorize_external_path: {normalized} (from {source_plugin})")
            return True
        return False

    def is_path_authorized(self, path: str, mode: str = "read") -> bool:
        """Check if a path is authorized for model access.

        This checks both exact path matches and parent directory matches
        (if a directory is authorized, all files within it are authorized).

        Args:
            path: Path to check (will be normalized).
            mode: Required access mode - "read" or "write" (default: "read").
                 A path authorized as "readwrite" satisfies both "read" and
                 "write" mode checks. A path authorized as "readonly" only
                 satisfies "read" mode checks.

        Returns:
            True if the path or a parent directory is authorized with
            sufficient access.
        """
        if not self._authorized_external_paths:
            return False

        # Normalize the path
        normalized = os.path.realpath(os.path.abspath(path))

        def _access_sufficient(entry_access: str) -> bool:
            """Check if the authorized access level is sufficient."""
            if mode == "read":
                return True  # Both "readonly" and "readwrite" allow reads
            # mode == "write": only "readwrite" allows writes
            return entry_access == "readwrite"

        # Check exact match
        if normalized in self._authorized_external_paths:
            _source, entry_access = self._authorized_external_paths[normalized]
            return _access_sufficient(entry_access)

        # Check if any authorized path is a parent directory
        for authorized_path, (_, entry_access) in self._authorized_external_paths.items():
            # Check if normalized path is under an authorized directory
            auth_with_sep = authorized_path.rstrip(os.sep) + os.sep
            if normalized.startswith(auth_with_sep):
                return _access_sufficient(entry_access)

        return False

    def get_path_authorization_source(self, path: str) -> Optional[str]:
        """Get the plugin that authorized a path.

        Args:
            path: Path to check (will be normalized).

        Returns:
            Name of the plugin that authorized this path, or None if not authorized.
        """
        if not self._authorized_external_paths:
            return None

        normalized = os.path.realpath(os.path.abspath(path))

        # Check exact match
        if normalized in self._authorized_external_paths:
            source, _access = self._authorized_external_paths[normalized]
            return source

        # Check parent directories
        for authorized_path, (source, _access) in self._authorized_external_paths.items():
            auth_with_sep = authorized_path.rstrip(os.sep) + os.sep
            if normalized.startswith(auth_with_sep):
                return source

        return None

    def get_path_access_mode(self, path: str) -> Optional[str]:
        """Get the access mode for an authorized path.

        Args:
            path: Path to check (will be normalized).

        Returns:
            Access mode ("readonly" or "readwrite"), or None if not authorized.
        """
        if not self._authorized_external_paths:
            return None

        normalized = os.path.realpath(os.path.abspath(path))

        # Check exact match
        if normalized in self._authorized_external_paths:
            _source, access = self._authorized_external_paths[normalized]
            return access

        # Check parent directories
        for authorized_path, (_source, access) in self._authorized_external_paths.items():
            auth_with_sep = authorized_path.rstrip(os.sep) + os.sep
            if normalized.startswith(auth_with_sep):
                return access

        return None

    def clear_authorized_paths(self, source_plugin: Optional[str] = None) -> int:
        """Clear authorized external paths.

        Args:
            source_plugin: If specified, only clear paths from this plugin.
                          If None, clear all authorized paths.

        Returns:
            Number of paths cleared.
        """
        if source_plugin is None:
            count = len(self._authorized_external_paths)
            self._authorized_external_paths.clear()
            _trace(f"clear_authorized_paths: cleared all {count} paths")
            return count

        # Clear only paths from the specified plugin
        to_remove = [
            path for path, (source, _access) in self._authorized_external_paths.items()
            if source == source_plugin
        ]
        for path in to_remove:
            del self._authorized_external_paths[path]

        _trace(f"clear_authorized_paths: cleared {len(to_remove)} paths from {source_plugin}")
        return len(to_remove)

    def list_authorized_paths(self) -> Dict[str, str]:
        """List all authorized external paths.

        Returns:
            Dict mapping normalized paths to the source plugin that authorized them.

        Note:
            For backward compatibility this returns source plugin names as values.
            Use list_authorized_paths_detailed() to get access mode info.
        """
        return {path: source for path, (source, _access) in self._authorized_external_paths.items()}

    def list_authorized_paths_detailed(self) -> Dict[str, Dict[str, str]]:
        """List all authorized external paths with access mode details.

        Returns:
            Dict mapping normalized paths to {"source": plugin_name, "access": mode}.
        """
        return {
            path: {"source": source, "access": access}
            for path, (source, access) in self._authorized_external_paths.items()
        }

    # ==================== External Path Denial ====================

    def deny_external_path(self, path: str, source_plugin: str) -> None:
        """Deny an external path, blocking model access even if otherwise allowed.

        Denied paths take precedence over authorized paths. This is used by
        the sandbox_manager plugin to implement session-level path blocking.

        Args:
            path: Absolute path to deny (will be normalized).
            source_plugin: Name of the plugin denying access.
        """
        # Normalize to absolute path
        normalized = os.path.realpath(os.path.abspath(path))
        self._denied_external_paths[normalized] = source_plugin
        _trace(f"deny_external_path: {normalized} (from {source_plugin})")

    def deny_external_paths(self, paths: List[str], source_plugin: str) -> None:
        """Deny multiple external paths.

        Args:
            paths: List of absolute paths to deny.
            source_plugin: Name of the plugin denying access.
        """
        for path in paths:
            self.deny_external_path(path, source_plugin)

    def is_path_denied(self, path: str) -> bool:
        """Check if a path is denied for model access.

        This checks both exact path matches and parent directory matches
        (if a directory is denied, all files within it are denied).

        Denial takes precedence over authorization - a path can be both
        authorized and denied, and denial wins.

        Args:
            path: Path to check (will be normalized).

        Returns:
            True if the path or a parent directory is denied.
        """
        if not self._denied_external_paths:
            return False

        # Normalize the path
        normalized = os.path.realpath(os.path.abspath(path))

        # Check exact match
        if normalized in self._denied_external_paths:
            return True

        # Check if any denied path is a parent directory
        for denied_path in self._denied_external_paths:
            # Check if normalized path is under a denied directory
            denied_with_sep = denied_path.rstrip(os.sep) + os.sep
            if normalized.startswith(denied_with_sep):
                return True

        return False

    def get_path_denial_source(self, path: str) -> Optional[str]:
        """Get the plugin that denied a path.

        Args:
            path: Path to check (will be normalized).

        Returns:
            Name of the plugin that denied this path, or None if not denied.
        """
        if not self._denied_external_paths:
            return None

        normalized = os.path.realpath(os.path.abspath(path))

        # Check exact match
        if normalized in self._denied_external_paths:
            return self._denied_external_paths[normalized]

        # Check parent directories
        for denied_path, source in self._denied_external_paths.items():
            denied_with_sep = denied_path.rstrip(os.sep) + os.sep
            if normalized.startswith(denied_with_sep):
                return source

        return None

    def clear_denied_paths(self, source_plugin: Optional[str] = None) -> int:
        """Clear denied external paths.

        Args:
            source_plugin: If specified, only clear paths from this plugin.
                          If None, clear all denied paths.

        Returns:
            Number of paths cleared.
        """
        if source_plugin is None:
            count = len(self._denied_external_paths)
            self._denied_external_paths.clear()
            _trace(f"clear_denied_paths: cleared all {count} paths")
            return count

        # Clear only paths from the specified plugin
        to_remove = [
            path for path, source in self._denied_external_paths.items()
            if source == source_plugin
        ]
        for path in to_remove:
            del self._denied_external_paths[path]

        _trace(f"clear_denied_paths: cleared {len(to_remove)} paths from {source_plugin}")
        return len(to_remove)

    def list_denied_paths(self) -> Dict[str, str]:
        """List all denied external paths.

        Returns:
            Dict mapping normalized paths to the source plugin that denied them.
        """
        return dict(self._denied_external_paths)

    # ==================== Prompt Enrichment ====================

    def _get_enrichment_priority(self, plugin: AnyPlugin) -> int:
        """Get the enrichment priority for a plugin.

        Lower values run first. Default is 50.

        Standard priorities:
        - 20: references (injects content first)
        - 40: template (scans injected content for templates)
        - 60: multimodal (handles @image references)
        - 80: memory (adds memory hints last)

        Args:
            plugin: The plugin to get priority for.

        Returns:
            The enrichment priority (lower = earlier).
        """
        if hasattr(plugin, 'get_enrichment_priority'):
            return plugin.get_enrichment_priority()
        return 50  # Default priority

    def get_prompt_enrichment_subscribers(self) -> List[AnyPlugin]:
        """Get plugins that subscribe to prompt enrichment, sorted by priority.

        Includes both exposed plugins and enrichment-only plugins.
        Plugins are sorted by enrichment priority (lower values run first).

        Returns:
            List of plugins that have subscribed to prompt enrichment,
            sorted by priority.
        """
        subscribers = []
        # Include both exposed and enrichment-only plugins
        all_enrichment_names = self._exposed | self._enrichment_only
        for name in all_enrichment_names:
            try:
                plugin = self._plugins[name]
                if (hasattr(plugin, 'subscribes_to_prompt_enrichment') and
                        plugin.subscribes_to_prompt_enrichment()):
                    subscribers.append(plugin)
            except Exception as exc:
                _trace(f" Error checking enrichment subscription for '{name}': {exc}", include_traceback=True)

        # Sort by priority (lower values first)
        subscribers.sort(key=self._get_enrichment_priority)
        return subscribers

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Run prompt through all subscribed enrichment plugins.

        Each subscribed plugin gets to inspect and optionally modify the prompt.
        Plugins are called in priority order (lower priority values first).

        Args:
            prompt: The user's original prompt text.

        Returns:
            PromptEnrichmentResult with the enriched prompt and combined metadata.
        """
        current_prompt = prompt
        combined_metadata: Dict[str, Any] = {}
        notifications: List[EnrichmentNotification] = []

        for plugin in self.get_prompt_enrichment_subscribers():
            try:
                if hasattr(plugin, 'enrich_prompt'):
                    before = current_prompt
                    result = plugin.enrich_prompt(current_prompt)
                    current_prompt = result.prompt
                    # Merge metadata, using plugin name as namespace
                    if result.metadata:
                        combined_metadata[plugin.name] = result.metadata

                    # Build notification if content changed
                    if current_prompt != before:
                        notif = self._build_notification(
                            target="prompt",
                            plugin_name=plugin.name,
                            metadata=result.metadata,
                            size_bytes=len(current_prompt) - len(before)
                        )
                        if notif:
                            notifications.append(notif)
            except Exception as exc:
                _trace(f" Error in prompt enrichment for '{plugin.name}': {exc}", include_traceback=True)

        # Emit notifications
        self._emit_enrichment_notifications(notifications)

        return PromptEnrichmentResult(prompt=current_prompt, metadata=combined_metadata)

    def _get_system_instruction_enrichment_priority(self, plugin: AnyPlugin) -> int:
        """Get the system instruction enrichment priority for a plugin.

        Lower values run first. Default is 50.

        Args:
            plugin: The plugin to get priority for.

        Returns:
            The enrichment priority (lower = earlier).
        """
        if hasattr(plugin, 'get_system_instruction_enrichment_priority'):
            return plugin.get_system_instruction_enrichment_priority()
        return 50  # Default priority

    def get_system_instruction_enrichment_subscribers(self) -> List[AnyPlugin]:
        """Get plugins that subscribe to system instruction enrichment.

        Includes both exposed plugins and enrichment-only plugins.
        Plugins are sorted by priority (lower values run first).

        Returns:
            List of plugins that subscribe to system instruction enrichment,
            sorted by priority.
        """
        subscribers = []
        all_enrichment_names = self._exposed | self._enrichment_only
        for name in all_enrichment_names:
            try:
                plugin = self._plugins[name]
                if (hasattr(plugin, 'subscribes_to_system_instruction_enrichment') and
                        plugin.subscribes_to_system_instruction_enrichment()):
                    subscribers.append(plugin)
            except Exception as exc:
                _trace(f" Error checking system instruction enrichment for '{name}': {exc}")

        subscribers.sort(key=self._get_system_instruction_enrichment_priority)
        return subscribers

    def enrich_system_instructions(
        self,
        instructions: str
    ) -> SystemInstructionEnrichmentResult:
        """Run system instructions through all subscribed enrichment plugins.

        Each subscribed plugin gets to inspect and optionally modify the
        combined system instructions. This is called after collecting raw
        instructions from all plugins.

        Args:
            instructions: Combined system instructions text.

        Returns:
            SystemInstructionEnrichmentResult with enriched instructions.
        """
        current_instructions = instructions
        combined_metadata: Dict[str, Any] = {}
        notifications: List[EnrichmentNotification] = []

        for plugin in self.get_system_instruction_enrichment_subscribers():
            try:
                if hasattr(plugin, 'enrich_system_instructions'):
                    before = current_instructions
                    result = plugin.enrich_system_instructions(current_instructions)
                    current_instructions = result.instructions
                    if result.metadata:
                        combined_metadata[plugin.name] = result.metadata

                    # Build notification if content changed
                    if current_instructions != before:
                        notif = self._build_notification(
                            target="system",
                            plugin_name=plugin.name,
                            metadata=result.metadata,
                            size_bytes=len(current_instructions) - len(before)
                        )
                        if notif:
                            notifications.append(notif)
            except Exception as exc:
                _trace(f" Error in system instruction enrichment for '{plugin.name}': {exc}")

        # Emit notifications
        self._emit_enrichment_notifications(notifications)

        return SystemInstructionEnrichmentResult(
            instructions=current_instructions,
            metadata=combined_metadata
        )

    def _get_tool_result_enrichment_priority(self, plugin: AnyPlugin) -> int:
        """Get the tool result enrichment priority for a plugin.

        Lower values run first. Default is 50.

        Args:
            plugin: The plugin to get priority for.

        Returns:
            The enrichment priority (lower = earlier).
        """
        if hasattr(plugin, 'get_tool_result_enrichment_priority'):
            return plugin.get_tool_result_enrichment_priority()
        return 50  # Default priority

    def get_tool_result_enrichment_subscribers(self) -> List[AnyPlugin]:
        """Get plugins that subscribe to tool result enrichment.

        Includes both exposed plugins and enrichment-only plugins.
        Plugins are sorted by priority (lower values run first).

        Returns:
            List of plugins that subscribe to tool result enrichment,
            sorted by priority.
        """
        subscribers = []
        all_enrichment_names = self._exposed | self._enrichment_only
        for name in all_enrichment_names:
            try:
                plugin = self._plugins[name]
                if (hasattr(plugin, 'subscribes_to_tool_result_enrichment') and
                        plugin.subscribes_to_tool_result_enrichment()):
                    subscribers.append(plugin)
            except Exception as exc:
                _trace(f" Error checking tool result enrichment for '{name}': {exc}")

        subscribers.sort(key=self._get_tool_result_enrichment_priority)
        return subscribers

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str,
        output_callback: Optional[OutputCallback] = None,
        terminal_width: Optional[int] = None,
        tool_args: Optional[Dict[str, Any]] = None
    ) -> ToolResultEnrichmentResult:
        """Run a tool result through all subscribed enrichment plugins.

        Each subscribed plugin gets to inspect and optionally modify the
        tool result before it is sent back to the model.

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's output as a string.
            output_callback: Optional callback for emitting notifications.
                If provided, this callback is used instead of the registry's
                shared callback. This is important for concurrent sessions
                (e.g., subagents) that need notifications routed to their
                specific output panel.
            terminal_width: Terminal width for formatting (uses registry default if not provided).
            tool_args: Optional tool call arguments, passed through to plugins
                that need them for context-aware enrichment (e.g., detecting
                which file was read by a CLI tool).

        Returns:
            ToolResultEnrichmentResult with enriched result.
        """
        current_result = result
        combined_metadata: Dict[str, Any] = {}
        notifications: List[EnrichmentNotification] = []

        for plugin in self.get_tool_result_enrichment_subscribers():
            try:
                if hasattr(plugin, 'enrich_tool_result'):
                    before = current_result
                    enrichment = plugin.enrich_tool_result(
                        tool_name, current_result, tool_args=tool_args
                    )
                    current_result = enrichment.result
                    if enrichment.metadata:
                        combined_metadata[plugin.name] = enrichment.metadata

                    # Build notification if content changed
                    if current_result != before:
                        notif = self._build_notification(
                            target="result",
                            plugin_name=plugin.name,
                            metadata=enrichment.metadata,
                            size_bytes=len(current_result) - len(before)
                        )
                        if notif:
                            notifications.append(notif)
            except Exception as exc:
                _trace(f" Error in tool result enrichment for '{plugin.name}': {exc}")

        # Emit notifications using provided callback or fallback to shared one
        self._emit_enrichment_notifications(
            notifications,
            output_callback=output_callback,
            terminal_width=terminal_width
        )

        return ToolResultEnrichmentResult(
            result=current_result,
            metadata=combined_metadata
        )
