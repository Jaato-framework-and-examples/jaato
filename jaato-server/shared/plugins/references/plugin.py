"""References plugin for managing documentation source injection.

This plugin maintains a catalog of reference sources (documentation, specs,
guides, etc.) and handles:
- AUTO sources: Included in system instructions, model fetches them at startup
- SELECTABLE sources: Model selects by ID or tags via selectReferences tool,
  or user selects via the 'references select <id>' command

The model uses selectReferences to directly select references and receive their
resolved paths. Selected sources have their paths authorized in the sandbox.
The model is responsible for fetching content using appropriate tools (CLI, MCP, etc.).

Enrichment Support:
- Prompt enrichment: Detects @reference-id mentions in user prompts
- Tool result enrichment: Detects @reference-id mentions in tool outputs
"""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from jaato_sdk.plugins.model_provider.types import ToolSchema
from ..subagent.config import expand_variables

from .models import ReferenceSource, ReferenceContents, InjectionMode, SourceType
from .channels import SelectionChannel, ConsoleSelectionChannel, QueueSelectionChannel, create_channel
from .config_loader import (
    load_config,
    ReferencesConfig,
    discover_references,
    resolve_source_paths,
    validate_reference_file,
)
from .bundle import (
    AmbiguousBundleRefError,
    BUNDLE_TIER_USER,
    BUNDLE_TIER_WORKSPACE,
    Bundle,
    BundleRef,
    ROOT_BUNDLE_NAME,
    VALID_BUNDLE_TIERS,
    detect_drift,
    discover_bundles,
    find_bundle,
    parse_bundle_ref,
    resolve_bundle_roots,
)
from .merge import (
    MergeOptions,
    MergeResult,
    MergeStatus,
    _read_bundle_manifest,
    _load_sources_from_dir,
    merge_bundle,
    parse_merge_args,
)
from .reconcile import ReconcileResult, ReconcileStatus, reconcile_bundle
from .embedding_types import (
    EmbeddingProviderProtocol,
    SemanticMatcherProtocol,
    SemanticMatch,
    create_semantic_matcher,
    discover_embedding_subsystem,
)
from jaato_sdk.plugins.base import (
    UserCommand,
    CommandParameter,
    CommandCompletion,
    HelpLines,
    PromptEnrichmentResult,
    ToolResultEnrichmentResult,
)

from shared.path_utils import normalize_for_comparison
from shared.session_context import get_current_session
from shared.trace import trace as _trace_write


# Maximum depth for transitive reference resolution to prevent runaway recursion
MAX_TRANSITIVE_DEPTH = 10


class ReferencesPlugin:
    """Plugin for managing reference source injection into model context.

    The plugin maintains a catalog of reference sources and:
    - AUTO sources: Included in system instructions for model to fetch
    - SELECTABLE sources: Model selects directly via selectReferences tool
      (by ID or tags), or user selects via 'references select <id>' command

    The model uses selectReferences to select and get resolved paths, then
    uses existing tools (CLI, MCP, URL fetch) to retrieve content.
    """

    def __init__(self):
        self._name = "references"
        self._config: Optional[ReferencesConfig] = None
        self._sources: List[ReferenceSource] = []
        self._channel: Optional[SelectionChannel] = None
        self._selected_source_ids: List[str] = []  # User-selected during session
        self._exclude_tools: List[str] = []  # Tools to exclude from schema
        self._initialized = False
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Selection lifecycle hooks for UI integration
        self._on_selection_requested: Optional[Callable[[str, List[str]], None]] = None
        self._on_selection_resolved: Optional[Callable[[str, List[str]], None]] = None
        # Project root for resolving relative paths (stored during initialize)
        self._project_root: Optional[str] = None
        # Workspace path set by PluginRegistry.set_workspace_path()
        self._workspace_path: Optional[str] = None
        # Plugin registry for cross-plugin communication (e.g., authorizing external paths)
        self._plugin_registry = None
        # Transitive reference metadata: maps each transitively discovered ID
        # to the set of parent source IDs that referenced it.
        # Populated during initialize() when transitive_injection is enabled,
        # and updated during runtime selection via _apply_transitive_selection().
        self._transitive_parent_map: Dict[str, Set[str]] = {}
        # One-time flag: when True, the next prompt enrichment call will emit
        # a lightweight transitive-selection hint so the model and user are
        # notified. Reset to False after the first emission.
        self._transitive_notification_pending: bool = False
        # Whether transitive reference injection is enabled.
        # Set during initialize() from the transitive_injection config option.
        # When True, runtime selections (selectReferences tool and
        # 'references select' command) also trigger transitive resolution.
        self._transitive_enabled: bool = True
        # Mapping from normalized resolved_path to (ref_id, ref_name) for
        # preselected LOCAL references. Built during initialize() and used
        # by enrich_tool_result() to detect when the model reads a
        # preselected reference file. Paths are normalized using
        # normalize_for_comparison() for cross-platform matching.
        self._preselected_paths: Dict[str, Tuple[str, str]] = {}
        # Semantic matching: one embedding provider (shared) and one
        # matcher per bundle. Each Bundle owns its own sidecar matrix and
        # its own matcher instance; bundles whose embedding_model does not
        # match the provider have ``bundle.matcher == None`` and are
        # skipped at query time.
        # Initialized during initialize() when embedding config is present
        # and sentence-transformers is installed.
        self._embedding_provider: Optional[EmbeddingProviderProtocol] = None
        self._bundles: List[Bundle] = []
        # Retained for back-compat with tests that directly inspect the
        # plugin state; points at the root bundle's matcher when present.
        # New code should iterate self._bundles instead.
        self._semantic_matcher: Optional[SemanticMatcherProtocol] = None
        # Semantic matching configuration.
        # lookup_strategy: "hybrid" (tags + semantic), "tags_only", "semantic_only"
        self._lookup_strategy: str = "hybrid"
        self._similarity_threshold: float = 0.75
        self._tag_similarity_threshold: float = 0.4
        self._max_matches_per_piece: int = 3
        # Per-session tracking of which reference hints have already been
        # injected into the model's context.  Split by pass so that a
        # reference first surfaced as a tag hint can later be upgraded to
        # an @mention expansion — but a repeated match of the same kind
        # is suppressed.  Cleared by on_history_cleared() on a true
        # session reset so references can be re-hinted in a fresh
        # conversation.
        self._surfaced_mention_ids: Set[str] = set()
        self._surfaced_tag_matched_ids: Set[str] = set()
        self._surfaced_semantic_ids: Set[str] = set()

    @property
    def name(self) -> str:
        return self._name

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("REFERENCES", msg)

    def set_plugin_registry(self, registry) -> None:
        """Set the plugin registry for cross-plugin communication.

        This enables the references plugin to authorize external paths for
        readFile access, allowing the model to read reference documents
        that are outside the workspace.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry
        registry.register_category("knowledge", "Reference sources, documentation, context retrieval")
        self._trace(f"set_plugin_registry: registry set")

    # NOTE: deliberately no ``set_session()`` here.  Plugin instances
    # are shared across subagents within a session, so storing the
    # session on ``self`` would leak the parent's session into a
    # subagent's set_session() call (and trip
    # tests/test_plugin_session_safety.py).  Auth paths instead read
    # ``get_current_session()`` from ``shared.session_context`` — the
    # session wiring sets the ContextVar around every tool execution.

    def set_workspace_path(self, path: str) -> None:
        """Update the workspace path for resolving reference source paths.

        Called by PluginRegistry.set_workspace_path() when a session binds
        to a specific workspace.  Re-derives _project_root and reloads the
        master catalog so sources resolve against the workspace.

        If the plugin initialized before the workspace was known (sources=0),
        this triggers a full catalog reload so that .jaato/references/ files
        are discovered.
        """
        self._workspace_path = path
        # Re-derive project root from workspace
        base_path_obj = Path(path).resolve()
        if '.jaato' in base_path_obj.parts:
            jaato_idx = base_path_obj.parts.index('.jaato')
            self._project_root = str(Path(*base_path_obj.parts[:jaato_idx]))
        else:
            self._project_root = str(base_path_obj)
        self._trace(f"set_workspace_path: workspace={path}, project_root={self._project_root}")

        # Reload catalog if sources weren't loaded during initialize()
        # (happens when workspace wasn't available at init time)
        if not self._sources:
            self._reload_catalog(path)

    def _reload_catalog(self, workspace_path: str) -> None:
        """Reload the master catalog from .jaato/references/ using the given workspace.

        Called by set_workspace_path() when sources weren't loaded during
        initialize() because the workspace wasn't available yet.

        Args:
            workspace_path: The workspace root path to scan for references.
        """
        try:
            self._config = load_config(None, workspace_path=workspace_path)
        except FileNotFoundError:
            self._config = ReferencesConfig()

        self._sources = self._config.sources
        for source in self._sources:
            self._resolve_source_for_context(source)

        self._trace(
            f"_reload_catalog: reloaded {len(self._sources)} sources "
            f"from workspace={workspace_path}"
        )

    def _authorize_source_path(self, source: ReferenceSource) -> bool:
        """Authorize a source's path for readonly access at every layer.

        Two cooperating layers are involved:

        1. **Application layer** (``sandbox_manager.add_path_programmatic``
           or registry fallback) — adds the path to the in-process
           allowlist so ``readFile`` and friends accept it.
        2. **Kernel layer** (per-session AppArmor reference fragment via
           ``ReferenceAuthorizer.authorize``) — only present in confined
           WS sessions.  Without this, the kernel ``open(2)`` would
           still EACCES even when the application says yes.

        Returns ``True`` when both layers succeeded (or when only the
        application layer applies because no AppArmor authorizer is
        installed), ``False`` when the kernel-layer fragment failed —
        in that case the application allowlist update is left in place
        but the caller should treat the reference as not selectable so
        the model gets a clear error instead of a silent EACCES later.

        Most call sites ignore the return value (catalog reload,
        transitive selection, enrichment-time mention auth) — for
        those, kernel-layer failure is logged via :meth:`_trace` and
        the user-facing tool path will surface it on a subsequent
        explicit selection.  ``selectReferences`` and the matching
        slash command propagate the failure into their result.
        """
        if not self._plugin_registry:
            return True

        if source.type != SourceType.LOCAL:
            return True

        # Get the resolved path
        resolved_path = self._resolve_path_for_access(source)
        if not resolved_path:
            return True

        path_str = str(resolved_path)

        # Layer 1: application-layer allowlist
        sandbox_plugin = self._plugin_registry.get_plugin("sandbox_manager")
        if sandbox_plugin and hasattr(sandbox_plugin, 'add_path_programmatic'):
            if sandbox_plugin.add_path_programmatic(path_str, access="readonly"):
                self._trace(f"authorized external path via sandbox: {path_str}")
            else:
                # Fall through to direct registry — sandbox plugin
                # exists but declined (e.g. validation error).  Ensure
                # something authorizes the path before we try the
                # kernel layer.
                self._plugin_registry.authorize_external_path(
                    path_str, self._name, access="readonly",
                )
                self._trace(
                    f"authorized external path via registry fallback "
                    f"(sandbox declined): {path_str}",
                )
        else:
            self._plugin_registry.authorize_external_path(
                path_str, self._name, access="readonly",
            )
            self._trace(
                f"authorized external path via registry fallback: {path_str}",
            )

        # Layer 2: kernel-layer AppArmor fragment (only when running
        # under a confined WS session).  Read the current session via
        # the ContextVar — never store on ``self``, plugin instances
        # are shared across subagents.  ``LookupError`` means we're
        # outside any session context (catalog discovery, tests) — no
        # kernel layer to mutate, treat as success at this layer.
        authorizer = self._current_reference_authorizer()
        if authorizer is not None:
            if not authorizer.authorize(source.id, path_str):
                self._trace(
                    f"AppArmor fragment FAILED for {source.id} "
                    f"({path_str}); kernel will deny reads",
                )
                return False
            self._trace(
                f"AppArmor fragment installed for {source.id} ({path_str})",
            )

        return True

    @staticmethod
    def _current_reference_authorizer():
        """Return the AppArmor reference authorizer for the current
        session context, or ``None`` if none is set.

        Centralises the ContextVar lookup so the auth/deauth paths
        share the same handling for the "no session in context" case
        (catalog discovery before configure(), unit tests that
        construct a plugin in isolation, etc.).
        """
        try:
            session = get_current_session()
        except LookupError:
            return None
        if session is None:
            return None
        getter = getattr(session, "get_reference_authorizer", None)
        if getter is None:
            return None
        return getter()

    def _deauthorize_source_path(self, source: ReferenceSource) -> bool:
        """Reverse :meth:`_authorize_source_path` at both layers.

        Returns ``True`` on success at every applicable layer; ``False``
        only when the kernel-layer fragment removal failed.  Most
        callers ignore the return value — deauthorization failure is
        not fatal (the rule will be cleared on session teardown), and
        the application-layer change always succeeds.
        """
        if not self._plugin_registry:
            return True

        if source.type != SourceType.LOCAL:
            return True

        resolved_path = self._resolve_path_for_access(source)
        if not resolved_path:
            return True

        path_str = str(resolved_path)

        # Layer 1: application-layer allowlist
        sandbox_plugin = self._plugin_registry.get_plugin("sandbox_manager")
        if sandbox_plugin and hasattr(sandbox_plugin, 'remove_path_programmatic'):
            if sandbox_plugin.remove_path_programmatic(path_str):
                self._trace(f"deauthorized external path via sandbox: {path_str}")
            else:
                self._plugin_registry.deauthorize_external_path(path_str, self._name)
                self._trace(
                    f"deauthorized external path via registry fallback "
                    f"(sandbox declined): {path_str}",
                )
        else:
            self._plugin_registry.deauthorize_external_path(path_str, self._name)
            self._trace(
                f"deauthorized external path via registry fallback: {path_str}",
            )

        # Layer 2: kernel-layer AppArmor fragment removal — same
        # ContextVar handling as _authorize_source_path.
        authorizer = self._current_reference_authorizer()
        if authorizer is not None:
            if not authorizer.deauthorize(source.id):
                self._trace(
                    f"AppArmor fragment removal FAILED for {source.id}",
                )
                return False
            self._trace(
                f"AppArmor fragment removed for {source.id}",
            )

        return True

    def _resolve_source_for_context(self, source: ReferenceSource) -> None:
        """Resolve a catalog source's path for the current project context.

        When a source from the master catalog is added to the active sources,
        its original relative path needs to be resolved against the current
        project root to create an absolute path that will work from any CWD.

        Args:
            source: The reference source to resolve (modified in-place).
        """
        if source.type != SourceType.LOCAL or not source.path:
            return

        # If already has a resolved_path that exists, nothing to do
        if source.resolved_path:
            resolved_obj = Path(source.resolved_path)
            if resolved_obj.is_absolute() and resolved_obj.exists():
                return
            # Also check resolved path relative to project root
            if self._project_root:
                from_root = Path(self._project_root) / source.resolved_path
                if from_root.exists():
                    source.resolved_path = str(from_root.resolve())
                    return

        # Try to resolve the original path against the project root
        if self._project_root:
            original_path = Path(source.path)
            if not original_path.is_absolute():
                from_root = Path(self._project_root) / source.path
                if from_root.exists():
                    source.resolved_path = str(from_root.resolve())
                    self._trace(f"resolved catalog source '{source.id}': {source.path} -> {source.resolved_path}")
                    return

        self._trace(f"could not resolve catalog source '{source.id}': {source.path}")

    def _resolve_path_for_access(self, source: ReferenceSource) -> Optional[Path]:
        """Resolve a source's path to an accessible filesystem location.

        Tries multiple strategies to find the file/directory:
        1. resolved_path as-is (may be relative or absolute)
        2. resolved_path relative to project root
        3. original path relative to project root
        4. original path as absolute

        Args:
            source: The reference source to resolve.

        Returns:
            Path object if found, None otherwise.
        """
        candidates = []

        # Try resolved_path first
        if source.resolved_path:
            resolved_obj = Path(source.resolved_path)
            candidates.append(resolved_obj)
            # Also try relative to project root
            if self._project_root and not resolved_obj.is_absolute():
                candidates.append(Path(self._project_root) / source.resolved_path)

        # Try original path
        if source.path:
            original_obj = Path(source.path)
            if original_obj.is_absolute():
                candidates.append(original_obj)
            elif self._project_root:
                candidates.append(Path(self._project_root) / source.path)

        # Return first existing path
        for path_obj in candidates:
            try:
                resolved = path_obj.resolve()
                if resolved.exists():
                    return resolved
            except (OSError, ValueError):
                continue

        return None

    def _get_reference_content(self, source: ReferenceSource) -> Optional[str]:
        """Get the content of a reference source for transitive detection.

        Only LOCAL and INLINE sources are supported for content extraction.
        URL and MCP sources would require external fetching which is deferred
        to the model.

        Args:
            source: The reference source to get content from.

        Returns:
            The content string if available, None otherwise.
        """
        if source.type == SourceType.INLINE:
            return source.content

        if source.type == SourceType.LOCAL:
            if not source.path and not source.resolved_path:
                self._trace(f"transitive:   '{source.id}' has no path")
                return None

            # Use the path resolution helper to find the file
            path_obj = self._resolve_path_for_access(source)

            if not path_obj:
                self._trace(
                    f"transitive:   '{source.id}' path not found "
                    f"(resolved={source.resolved_path}, original={source.path}, "
                    f"project_root={self._project_root}, workspace={self._workspace_path})"
                )
                return None

            # Handle directory sources - concatenate all file contents
            if path_obj.is_dir():
                contents: List[str] = []
                # Include common documentation file extensions
                doc_extensions = (
                    '.md', '.txt', '.json', '.yaml', '.yml',
                    '.html', '.htm', '.rst', '.adoc'
                )
                doc_files_found = 0
                try:
                    for item in sorted(path_obj.rglob("*")):
                        if item.is_file():
                            # Include files with doc extensions or README files (any extension)
                            is_doc_ext = item.suffix.lower() in doc_extensions
                            is_readme = item.stem.upper() == 'README'
                            if is_doc_ext or is_readme:
                                try:
                                    contents.append(item.read_text(encoding='utf-8'))
                                    doc_files_found += 1
                                except (IOError, OSError, UnicodeDecodeError):
                                    pass  # Skip unreadable files
                except (PermissionError, OSError) as e:
                    self._trace(f"transitive:   '{source.id}' dir scan error: {e}")

                if contents:
                    self._trace(f"transitive:   '{source.id}' dir -> {doc_files_found} doc files, {sum(len(c) for c in contents)} chars")
                    return "\n".join(contents)
                else:
                    self._trace(f"transitive:   '{source.id}' dir -> no doc files found in {path_obj}")
                    return None

            # Handle regular file
            if path_obj.is_file():
                try:
                    return path_obj.read_text(encoding='utf-8')
                except (IOError, OSError, UnicodeDecodeError):
                    return None

        return None

    def _find_referenced_ids(self, content: str, catalog_ids: Set[str]) -> Set[str]:
        """Find reference IDs mentioned in content.

        Searches for catalog IDs appearing as whole words in the content.
        This handles common patterns like:
        - Direct ID mentions: "skill-001-circuit-breaker"
        - Reference syntax: "@ref:skill-001" or "[[skill-001]]"
        - Prose mentions: "see skill-001-circuit-breaker for details"

        Args:
            content: The content to search for reference mentions.
            catalog_ids: Set of valid reference IDs from the catalog.

        Returns:
            Set of reference IDs found in the content.
        """
        found_ids: Set[str] = set()

        for ref_id in catalog_ids:
            # Escape special regex characters in the ID
            escaped_id = re.escape(ref_id)
            # Match as a whole word (with word boundaries or common delimiters)
            # Pattern allows for common reference syntaxes like @ref:id, [[id]], `id`
            pattern = rf'(?:^|[\s\[\]`@:,;()\'"{{}}])({escaped_id})(?:[\s\[\]`@:,;()\'"{{}}]|$)'
            if re.search(pattern, content, re.MULTILINE):
                found_ids.add(ref_id)

        return found_ids

    def _find_referenced_paths(
        self,
        content: str,
        source_resolved_path: str,
        path_to_ids: Dict[str, Set[str]]
    ) -> Set[str]:
        """Find reference IDs by resolving relative paths mentioned in content.

        Extracts file paths from markdown links and relative path patterns,
        resolves them against the source document's directory, and matches
        against catalog source resolved_paths.

        This complements _find_referenced_ids (which matches by catalog ID)
        so that transitive detection works when documents reference each
        other via relative paths rather than by catalog ID.

        Args:
            content: The document content to scan for path references.
            source_resolved_path: Resolved path of the document being scanned,
                relative to project root. Used as base for resolving relative
                paths found in content.
            path_to_ids: Mapping from normalized resolved_path to set of
                source IDs that share that path.

        Returns:
            Set of catalog source IDs referenced by path.
        """
        if not source_resolved_path or not path_to_ids:
            return set()

        found_ids: Set[str] = set()

        # Directory of the source document (for resolving relative paths)
        source_dir = os.path.dirname(source_resolved_path)

        # --- Extract paths from content ---
        extracted_paths: Set[str] = set()

        # Markdown links: [text](path) — skip URLs and anchors
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', content):
            link = match.group(1).strip()
            if link.startswith(('http://', 'https://', '#', 'mailto:')):
                continue
            # Strip anchor fragments: path.md#section → path.md
            if '#' in link:
                link = link.split('#')[0]
            if link:
                extracted_paths.add(link)

        # Explicit relative paths: ./foo or ../foo (not inside longer words)
        for match in re.finditer(r'(?:^|(?<=\s))(\.\./[\w./_-]+|\.\/[\w./_-]+)', content, re.MULTILINE):
            extracted_paths.add(match.group(1))

        if not extracted_paths:
            return found_ids

        # --- Resolve and match ---
        for raw_path in extracted_paths:
            # Resolve relative to source's directory
            resolved = os.path.normpath(os.path.join(source_dir, raw_path))
            resolved = resolved.replace('\\', '/')

            # Try exact match
            if resolved in path_to_ids:
                found_ids.update(path_to_ids[resolved])
                continue

            # Try with/without trailing slash (directory sources)
            alt = resolved.rstrip('/') if resolved.endswith('/') else resolved + '/'
            if alt in path_to_ids:
                found_ids.update(path_to_ids[alt])
                continue

        if found_ids:
            self._trace(
                f"transitive:   path matches: {sorted(found_ids)} "
                f"(from {len(extracted_paths)} extracted paths)"
            )

        return found_ids

    def _resolve_transitive_references(
        self,
        initial_ids: List[str],
        catalog_by_id: Dict[str, ReferenceSource],
        max_depth: int = MAX_TRANSITIVE_DEPTH
    ) -> Tuple[List[str], Dict[str, Set[str]]]:
        """Resolve transitive references from pre-selected sources.

        Starting from the initially selected reference IDs, reads their content
        and discovers mentions of other references via two strategies:
        1. **ID matching**: Finds catalog IDs mentioned as whole words in content.
        2. **Path matching**: Extracts relative paths (markdown links, ``./``
           and ``../`` patterns), resolves them against the source's directory,
           and matches against resolved_path of other LOCAL catalog sources.

        Recursively resolves discovered references until no new references are
        found or max depth is reached.

        Args:
            initial_ids: List of initially selected/pre-selected reference IDs.
            catalog_by_id: Mapping of reference ID to ReferenceSource.
            max_depth: Maximum recursion depth to prevent runaway resolution.

        Returns:
            Tuple of:
            - List of all resolved reference IDs (initial + transitively discovered),
              in order of discovery (initial IDs first, then discovered ones).
            - Parent map: dict mapping each transitively discovered ID to the
              set of parent source IDs that referenced it. Initial IDs are
              not included in this map.
        """
        if not initial_ids:
            return [], {}

        self._trace(f"transitive: starting from {initial_ids}")

        # Track all resolved IDs and order of discovery
        resolved_ids: List[str] = list(initial_ids)
        resolved_set: Set[str] = set(initial_ids)

        # Parent map: discovered_id → {parent_ids that referenced it}
        parent_map: Dict[str, Set[str]] = {}

        # IDs to process in this iteration
        pending: Set[str] = set(initial_ids)
        catalog_ids = set(catalog_by_id.keys())

        # Build resolved_path → source IDs mapping for path-based matching.
        # Only LOCAL sources with a resolved_path participate.
        path_to_ids: Dict[str, Set[str]] = {}
        for sid, source in catalog_by_id.items():
            if source.type == SourceType.LOCAL and source.resolved_path:
                norm = os.path.normpath(source.resolved_path).replace('\\', '/')
                path_to_ids.setdefault(norm, set()).add(sid)

        for depth in range(max_depth):
            if not pending:
                break

            newly_found: Set[str] = set()
            self._trace(f"transitive: [depth={depth}] scanning {sorted(pending)}")

            for ref_id in pending:
                source = catalog_by_id.get(ref_id)
                if not source:
                    continue

                # Get content from the source
                content = self._get_reference_content(source)
                if not content:
                    self._trace(f"transitive:   '{ref_id}' -> no content (type={source.type.value})")
                    continue

                self._trace(f"transitive:   '{ref_id}' -> {len(content)} chars")

                # Strategy 1: Find references by catalog ID mentioned in content
                mentioned_ids = self._find_referenced_ids(content, catalog_ids)

                # Strategy 2: Find references by resolving relative paths
                if source.resolved_path and path_to_ids:
                    mentioned_ids |= self._find_referenced_paths(
                        content, source.resolved_path, path_to_ids
                    )

                # Filter to only newly discovered ones for BFS progression
                new_mentions = mentioned_ids - resolved_set - {ref_id}
                if new_mentions:
                    self._trace(f"transitive:   '{ref_id}' => {sorted(new_mentions)}")
                    for mentioned_id in new_mentions:
                        newly_found.add(mentioned_id)
                        resolved_set.add(mentioned_id)
                        resolved_ids.append(mentioned_id)
                        parent_map.setdefault(mentioned_id, set()).add(ref_id)

                # Record parent relationships for IDs already resolved
                # (discovered earlier by a sibling at the same BFS depth).
                # This ensures multi-parent tracking is complete.
                initial_set = set(initial_ids)
                for mentioned_id in (mentioned_ids & resolved_set) - initial_set - {ref_id}:
                    parent_map.setdefault(mentioned_id, set()).add(ref_id)

            # Next iteration processes newly found IDs
            pending = newly_found

        if pending:
            self._trace(
                f"transitive: max depth {max_depth} reached, {len(pending)} unresolved"
            )

        # Final summary
        transitive_count = len(resolved_ids) - len(initial_ids)
        if transitive_count > 0:
            transitive_ids = resolved_ids[len(initial_ids):]
            self._trace(f"transitive: added {transitive_count}: {transitive_ids}")
        else:
            self._trace("transitive: no additional references found")

        return resolved_ids, parent_map

    def _apply_transitive_selection(
        self,
        newly_selected_ids: List[str],
    ) -> List[ReferenceSource]:
        """Run transitive resolution on newly selected references and apply results.

        Scans the content of the given newly selected sources for mentions of
        other catalog references, recursively discovers transitive dependencies,
        and adds any newly found sources to the selected set. Updates
        ``_transitive_parent_map`` and sets ``_transitive_notification_pending``
        so the next prompt enrichment notifies the model.

        Called from both ``_execute_select`` (model tool) and
        ``_cmd_references_select`` (user command) when
        ``self._transitive_enabled`` is True.

        Args:
            newly_selected_ids: IDs of sources that were just directly selected
                (already appended to ``_selected_source_ids`` and authorized).

        Returns:
            List of ReferenceSource objects that were transitively added.
            Empty list if transitive injection is disabled or nothing was found.
        """
        if not self._transitive_enabled or not newly_selected_ids:
            return []

        # Build catalog from all known sources
        catalog_by_id: Dict[str, ReferenceSource] = {
            s.id: s for s in self._sources
        }
        if self._config:
            for s in self._config.sources:
                if s.id not in catalog_by_id:
                    catalog_by_id[s.id] = s

        all_resolved, transitive_parent_map = self._resolve_transitive_references(
            newly_selected_ids, catalog_by_id
        )

        # Filter to only truly new IDs (not already selected)
        already_selected = set(self._selected_source_ids)
        transitive_sources: List[ReferenceSource] = []
        current_source_ids = {s.id for s in self._sources}

        for ref_id in all_resolved:
            if ref_id in already_selected:
                continue

            # Add to selected set and authorize
            self._selected_source_ids.append(ref_id)
            already_selected.add(ref_id)

            # Ensure source is in self._sources
            if ref_id not in current_source_ids and ref_id in catalog_by_id:
                source = catalog_by_id[ref_id]
                self._resolve_source_for_context(source)
                self._sources.append(source)
                current_source_ids.add(ref_id)

            source = next((s for s in self._sources if s.id == ref_id), None)
            if source:
                self._authorize_source_path(source)
                transitive_sources.append(source)

        # Merge new transitive parent mappings (only for truly new entries)
        for ref_id, parents in transitive_parent_map.items():
            if ref_id not in already_selected or ref_id in {s.id for s in transitive_sources}:
                self._transitive_parent_map.setdefault(ref_id, set()).update(parents)

        if transitive_sources:
            transitive_ids = [s.id for s in transitive_sources]
            self._trace(f"transitive (runtime): injected {len(transitive_ids)}: {transitive_ids}")
            self._transitive_notification_pending = True

        return transitive_sources

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict. If not provided, loads from
                   file specified by REFERENCES_CONFIG_PATH or default locations.

                   Config options:
                   - config_path: Path to references.json file
                   - channel_type: Type of channel ("console", "webhook", "file")
                   - channel_config: Configuration for the channel
                   - sources: Inline sources list (overrides file)
                   - preselected: List of source IDs to pre-select at startup.
                                  Sources are looked up from the master catalog
                                  and automatically added to available sources.
                   - transitive_injection: Enable transitive reference detection (default: True).
                                           When enabled, pre-selected references are scanned for
                                           mentions of other catalog references, which are then
                                           automatically injected. Also applies to runtime
                                           selections via selectReferences tool and
                                           'references select' command.
                   - workspace_path: Workspace root for resolving .jaato/references/.
                                     Falls back to base_path, then self._workspace_path.
                   - exclude_tools: List of tool names to exclude (e.g., ["selectReferences"])
        """
        config = config or {}

        # Expand variables in config values (e.g., ${projectPath}, ${workspaceRoot})
        config = expand_variables(config) if config else {}

        # Extract agent name for trace logging
        self._agent_name = config.get("agent_name")

        # Compute and store project root for path resolution
        # This is used when resolving paths for catalog sources and in _get_reference_content
        inline_base_path = config.get("base_path") or config.get("workspace_path") or self._workspace_path
        if not inline_base_path:
            self._trace("initialize: no base_path in config and no workspace set — project_root will be None")
        base_path_obj = Path(inline_base_path).resolve() if inline_base_path else None
        if base_path_obj is not None:
            if '.jaato' in base_path_obj.parts:
                jaato_idx = base_path_obj.parts.index('.jaato')
                self._project_root = str(Path(*base_path_obj.parts[:jaato_idx]))
            else:
                self._project_root = str(base_path_obj)

        # Try to load from file first (master catalog)
        config_path = config.get("config_path")
        try:
            self._config = load_config(config_path, workspace_path=inline_base_path)
        except FileNotFoundError:
            # Use defaults
            self._config = ReferencesConfig()

        # Build ID -> source lookup from master catalog
        catalog_by_id = {s.id: s for s in self._config.sources}

        # Allow inline sources override
        if "sources" in config:
            resolved_sources = []
            for s in config["sources"]:
                if isinstance(s, dict):
                    resolved_sources.append(ReferenceSource.from_dict(s))
                elif isinstance(s, str):
                    # Look up by ID from master catalog
                    if s in catalog_by_id:
                        resolved_sources.append(catalog_by_id[s])
                    else:
                        print(f"Warning: Source ID '{s}' not found in master catalog")
                else:
                    resolved_sources.append(s)

            # Resolve relative paths for inline sources against workspace
            if inline_base_path:
                resolve_source_paths(resolved_sources, inline_base_path, relative_to=self._project_root)
            self._sources = resolved_sources
        else:
            # Use sources from master catalog - resolve paths for current context
            self._sources = self._config.sources
            for source in self._sources:
                self._resolve_source_for_context(source)

        # Handle preselected - look up from catalog and add to sources if needed
        preselected = config.get("preselected", [])
        if preselected:
            current_ids = {s.id for s in self._sources}
            for sid in preselected:
                if sid not in current_ids and sid in catalog_by_id:
                    # Add from master catalog - need to resolve paths for current context
                    source = catalog_by_id[sid]
                    self._resolve_source_for_context(source)
                    self._sources.append(source)
                    current_ids.add(sid)

        # Initialize channel
        channel_type = config.get("channel_type") or self._config.channel_type
        channel_config = config.get("channel_config", {})

        # Set timeout from config
        if "timeout" not in channel_config:
            channel_config["timeout"] = self._config.channel_timeout

        # Set type-specific config
        if channel_type == "webhook" and "endpoint" not in channel_config:
            if self._config.channel_endpoint:
                channel_config["endpoint"] = self._config.channel_endpoint

        if channel_type == "file" and "base_path" not in channel_config:
            if self._config.channel_base_path:
                channel_config["base_path"] = self._config.channel_base_path

        try:
            self._channel = create_channel(channel_type, channel_config)
        except (ValueError, RuntimeError) as e:
            # Fall back to console channel if configured channel fails
            print(f"Warning: Failed to initialize {channel_type} channel: {e}")
            print("Falling back to console channel")
            self._channel = ConsoleSelectionChannel()
            self._channel.initialize({})

        # Initialize selected sources from preselected config
        # (sources were already added above, now just validate and track IDs)
        if preselected:
            available_ids = {s.id for s in self._sources}
            valid_preselected = [sid for sid in preselected if sid in available_ids]
            invalid = set(preselected) - available_ids - set(catalog_by_id.keys())
            if invalid:
                print(f"Warning: Preselected reference IDs not found: {invalid}")
            self._selected_source_ids = valid_preselected
        else:
            self._selected_source_ids = []

        # Resolve transitive references if enabled (default: True)
        # This scans pre-selected references for mentions of other catalog references
        # and automatically adds them to the selected set
        self._transitive_enabled = config.get("transitive_injection", True)
        if self._transitive_enabled and self._selected_source_ids:
            # Build complete catalog including inline sources
            full_catalog = dict(catalog_by_id)
            for source in self._sources:
                if source.id not in full_catalog:
                    full_catalog[source.id] = source

            # Resolve transitive references
            all_resolved, transitive_parent_map = self._resolve_transitive_references(
                self._selected_source_ids,
                full_catalog
            )
            self._transitive_parent_map = transitive_parent_map
            if transitive_parent_map:
                self._transitive_notification_pending = True

            # Add newly discovered sources to self._sources and self._selected_source_ids
            current_source_ids = {s.id for s in self._sources}
            for ref_id in all_resolved:
                if ref_id not in self._selected_source_ids:
                    self._selected_source_ids.append(ref_id)
                # Ensure source is in self._sources
                if ref_id not in current_source_ids and ref_id in full_catalog:
                    source = full_catalog[ref_id]
                    # Resolve paths for catalog sources added during transitive resolution
                    self._resolve_source_for_context(source)
                    self._sources.append(source)
                    current_source_ids.add(ref_id)

            # Log transitive injection results
            transitive_count = len(all_resolved) - len(valid_preselected) if preselected else 0
            if transitive_count > 0:
                transitive_ids = [rid for rid in all_resolved if rid not in valid_preselected]
                self._trace(f"transitive: injected {transitive_count} additional: {transitive_ids}")

        # Capture excluded tools
        self._exclude_tools = config.get("exclude_tools", [])
        self._initialized = True

        # Trace logging for debugging
        channel_type = config.get("channel_type") or self._config.channel_type
        self._trace(f"initialize: sources={len(self._sources)}, channel={channel_type}")

        # Log resolved paths for LOCAL sources (indicate if directory)
        for source in self._sources:
            if source.type == SourceType.LOCAL and source.resolved_path:
                path_obj = Path(source.resolved_path)
                is_dir = path_obj.is_dir() if path_obj.exists() else False
                path_type = "dir" if is_dir else "file"
                self._trace(f"initialize: resolved '{source.id}' ({path_type}): {source.path} -> {source.resolved_path}")

        if self._selected_source_ids:
            self._trace(f"initialize: preselected={self._selected_source_ids}")
        if self._exclude_tools:
            self._trace(f"initialize: exclude_tools={self._exclude_tools}")

        # Authorize paths for pre-selected sources so readFile can access them
        if self._selected_source_ids:
            for sid in self._selected_source_ids:
                source = next((s for s in self._sources if s.id == sid), None)
                if source:
                    self._authorize_source_path(source)

        # Build preselected paths index for reference-read detection.
        # Maps normalized resolved_path → (ref_id, ref_name) for all
        # preselected LOCAL sources (including transitively resolved ones).
        # For directory references, only the directory path is stored;
        # _detect_preselected_read uses startswith + "/" to match files
        # inside the directory without needing per-file entries.
        self._preselected_paths = {}
        for sid in self._selected_source_ids:
            source = next((s for s in self._sources if s.id == sid), None)
            if source and source.type == SourceType.LOCAL and source.resolved_path:
                norm = normalize_for_comparison(os.path.normpath(source.resolved_path))
                self._preselected_paths[norm] = (source.id, source.name)
                self._trace(
                    f"initialize: preselected_path '{source.id}': {norm}"
                )

        # --- Semantic matching initialization ---
        # Read semantic config from plugin config (passed via initialize(config))
        self._lookup_strategy = config.get("lookup_strategy", "hybrid")
        self._similarity_threshold = config.get("similarity_threshold", 0.75)
        self._tag_similarity_threshold = config.get("tag_similarity_threshold", 0.4)
        self._max_matches_per_piece = config.get("max_matches_per_piece", 3)

        # Always try to create the embedding provider so that the
        # compute_embedding tool works even when generating embeddings
        # for the first time (no existing sidecar/config yet).
        self._init_embedding_provider(config)

        # Discover bundles (root + subdirectories with their own manifests)
        # and load sub-bundle references into the flat catalog. Safe to
        # call unconditionally — a workspace with no manifest anywhere
        # yields an empty bundle list and becomes a no-op.
        self._discover_and_load_bundles()

        # Reconcile drift (new/edited/removed references) against each
        # bundle's sidecar. Bundles with reconcile_mode == "lazy" are
        # deferred to the first semantic query; "off" disables the pass.
        if self._lookup_strategy in ("hybrid", "semantic_only"):
            self._reconcile_eager_bundles()

        # Attach one matcher per compatible bundle. Bundles whose
        # embedding_model does not match the provider (or whose sidecar
        # fails to load) are left without a matcher and skipped at query
        # time; tag matching continues to work for their references.
        if self._lookup_strategy in ("hybrid", "semantic_only"):
            self._init_bundle_matchers(config)
        else:
            self._trace(
                f"initialize: semantic matching not configured "
                f"(strategy={self._lookup_strategy}, bundles={len(self._bundles)})"
            )

    def _init_embedding_provider(self, config: Dict[str, Any]) -> None:
        """Initialize the embedding provider via entry point discovery.

        Called unconditionally from ``initialize()`` so that the
        ``compute_embedding`` tool works even during first-time generation
        (when no sidecar or embedding config exists yet).

        The provider is discovered via the ``jaato.embedding`` entry point
        group. If no provider is registered, embedding features are
        unavailable and tag-based matching continues to work.
        """
        if self._embedding_provider is not None:
            return  # Already initialized (e.g. by _init_semantic_matching)

        provider, matcher = discover_embedding_subsystem(config)

        if provider is not None:
            self._embedding_provider = provider
            self._trace(
                f"_init_embedding_provider: discovered provider "
                f"(model='{provider.model_name}')"
            )
        else:
            self._trace(
                "_init_embedding_provider: no embedding provider registered "
                "via jaato.embedding entry point"
            )

        # Cache the discovered matcher for _init_semantic_matching()
        if matcher is not None:
            self._semantic_matcher = matcher

    def _discover_and_load_bundles(self) -> None:
        """Populate ``self._bundles`` and merge bundle refs into the catalog.

        Walks both tier roots (workspace then user) via
        :func:`resolve_bundle_roots`. Within each root the root bundle is
        discovered from ``<root>/embedding_config.json`` and sub-bundles
        from each immediate subdirectory containing its own manifest.
        Each bundle's reference JSONs are loaded via
        :func:`discover_references` and tagged with ``bundle_name`` so the
        flat catalog can still reason about membership.

        **Tier shadowing:** :func:`discover_bundles` walks workspace first
        and a workspace bundle hides any user bundle of the same name —
        the shadowed user bundle never appears in ``self._bundles`` and
        its references are not loaded into the catalog.

        Sources already loaded by :func:`load_config` (the workspace-tier
        root bundle) have their ``bundle_name`` left as the default empty
        string. Sources from the user-tier root bundle are loaded here
        and likewise carry ``bundle_name = ""`` (the empty name is the
        root-bundle sentinel; tier disambiguation lives on the
        :class:`Bundle` itself, not on individual references).
        """
        self._bundles = []

        if not self._config:
            return

        # Resolve the workspace tier root from the config (which may have
        # an absolute or relative ``references_dir``). The user tier always
        # lives under ``~/.jaato/references`` — :func:`resolve_bundle_roots`
        # handles that — but we splice in the user pair manually so that a
        # custom workspace ``references_dir`` (set in references.json) is
        # still honored.
        workspace_refs_dir: Optional[Path] = None
        cfg_refs = Path(self._config.references_dir)
        if cfg_refs.is_absolute():
            workspace_refs_dir = cfg_refs
        elif self._workspace_path:
            workspace_refs_dir = Path(self._workspace_path) / cfg_refs
        elif self._project_root:
            workspace_refs_dir = Path(self._project_root) / cfg_refs

        roots: List[Tuple[Path, str]] = []
        if workspace_refs_dir is not None:
            roots.append((workspace_refs_dir.resolve(), BUNDLE_TIER_WORKSPACE))
        else:
            self._trace(
                "_discover_and_load_bundles: workspace references_dir "
                "unresolved (no workspace_path); workspace tier skipped"
            )
        # User tier — always available.
        for path, tier in resolve_bundle_roots(workspace_path=None):
            if tier == BUNDLE_TIER_USER:
                roots.append((path, tier))
                break

        self._bundles = discover_bundles(roots)

        if not self._bundles:
            self._trace("_discover_and_load_bundles: no bundles found")
            return

        # Load bundle references into the flat catalog. The workspace-tier
        # root bundle's refs are already in self._sources from the initial
        # load_config() call; everything else (workspace sub-bundles, plus
        # all user-tier bundles that survived shadowing) is loaded here.
        existing_ids = {s.id for s in self._sources}
        project_root = self._project_root
        for bundle in self._bundles:
            # Skip the workspace-tier root — already loaded by load_config().
            if (
                bundle.name == ROOT_BUNDLE_NAME
                and bundle.tier == BUNDLE_TIER_WORKSPACE
            ):
                continue
            bundle_sources = discover_references(
                str(bundle.directory),
                base_path=str(bundle.directory.parent),
                project_root=project_root,
            )
            for source in bundle_sources:
                source.bundle_name = bundle.name
                if source.id in existing_ids:
                    self._trace(
                        f"_discover_and_load_bundles: skipping duplicate id "
                        f"'{source.id}' from bundle '{bundle.qualified_ref}'"
                    )
                    continue
                self._sources.append(source)
                existing_ids.add(source.id)

        self._trace(
            f"_discover_and_load_bundles: bundles="
            f"{[b.qualified_ref for b in self._bundles]}, "
            f"total_sources={len(self._sources)}"
        )

    def _reconcile_eager_bundles(self) -> None:
        """Run :func:`reconcile_bundle` for every eager-mode bundle with drift.

        Lazy and off bundles are skipped. Results are logged per bundle.
        When the provider is missing, reconcile returns
        :attr:`ReconcileStatus.UNAVAILABLE` and the bundle's sidecar is
        left untouched — tag matching still works for its references.
        """
        if not self._bundles:
            return
        for bundle in self._bundles:
            if bundle.reconcile_mode != "eager":
                continue
            result = reconcile_bundle(
                bundle, self._sources, self._embedding_provider
            )
            if result.status == ReconcileStatus.CLEAN:
                continue
            self._trace(
                f"reconcile[{bundle.display_name}]: {result.summary()}"
            )

    def _init_bundle_matchers(self, config: Dict[str, Any]) -> None:
        """Attach a semantic matcher to each compatible bundle.

        A bundle is "compatible" when the embedding provider is available
        *and* ``bundle.embedding_model`` equals the provider's model name.
        Mismatched bundles are left with ``bundle.matcher is None`` and
        logged once so the operator knows why semantic matching is
        skipping them.

        The legacy single-matcher field :attr:`_semantic_matcher` is kept
        in sync with the root bundle's matcher (if any) for the benefit of
        tests that still assert on it directly.
        """
        if not self._bundles:
            return

        if self._embedding_provider is None:
            self._trace(
                "_init_bundle_matchers: no embedding provider — "
                "all bundles left without a matcher"
            )
            return

        if not self._embedding_provider.available:
            self._embedding_provider.load_model()

        if not self._embedding_provider.available:
            self._trace(
                "_init_bundle_matchers: embedding provider failed to load"
            )
            self._embedding_provider = None
            return

        provider_model = config.get(
            "embedding_model", self._embedding_provider.model_name
        )

        for bundle in self._bundles:
            bundle.matcher = self._attach_matcher(bundle, provider_model)

        root = next(
            (b for b in self._bundles if b.name == ROOT_BUNDLE_NAME), None
        )
        self._semantic_matcher = root.matcher if root else None

        attached = sum(1 for b in self._bundles if b.matcher is not None)
        self._trace(
            f"_init_bundle_matchers: attached={attached}/{len(self._bundles)} "
            f"bundles (provider model='{provider_model}')"
        )

    def _attach_matcher(
        self, bundle: Bundle, provider_model: str,
    ) -> Optional[SemanticMatcherProtocol]:
        """Build and wire a matcher for a single bundle.

        Returns ``None`` when the bundle's embedding model does not match
        the active provider, when ``rows`` maps to no known catalog source
        (an empty or fully-orphan bundle), or when the matcher fails to
        load the sidecar. In all cases the method logs and the rest of
        the catalog is unaffected.
        """
        if bundle.embedding_model != provider_model:
            self._trace(
                f"_attach_matcher[{bundle.display_name}]: model mismatch — "
                f"bundle '{bundle.embedding_model}' vs provider "
                f"'{provider_model}'; skipping"
            )
            return None

        known_ids = {
            s.id for s in self._sources if s.bundle_name == bundle.name
        }
        index_to_source_id: Dict[int, str] = {
            i: sid
            for i, sid in enumerate(bundle.embedding_rows)
            if sid in known_ids
        }
        if not index_to_source_id:
            self._trace(
                f"_attach_matcher[{bundle.display_name}]: no rows map to "
                f"known catalog sources; skipping"
            )
            return None

        matcher = create_semantic_matcher()
        if matcher is None:
            self._trace(
                f"_attach_matcher[{bundle.display_name}]: no semantic_matcher "
                f"entry point registered"
            )
            return None

        if not matcher.validate_model(provider_model):
            self._trace(
                f"_attach_matcher[{bundle.display_name}]: matcher rejected "
                f"provider model '{provider_model}'"
            )
            return None

        loaded = matcher.load_index(
            sidecar_path=str(bundle.sidecar_path),
            embedding_model=bundle.embedding_model,
            embedding_dimensions=bundle.embedding_dimensions,
            index_to_source_id=index_to_source_id,
        )
        if not loaded:
            self._trace(
                f"_attach_matcher[{bundle.display_name}]: failed to load sidecar "
                f"{bundle.sidecar_path}"
            )
            return None

        matcher.set_provider(self._embedding_provider)
        return matcher

    # ----- Cross-bundle semantic matching helpers -----

    def _semantic_available(self) -> bool:
        """Whether at least one bundle has an attached matcher ready to query."""
        return any(
            b.matcher is not None and b.matcher.available
            for b in self._bundles
        )

    def _semantic_score_sources(
        self, query_vec: Any, source_ids: Set[str],
    ) -> Dict[str, float]:
        """Score ``source_ids`` across every bundle that owns them.

        Each bundle only knows about its own sources, so we partition
        ``source_ids`` by bundle before delegating. Missing entries in
        the merged dict mean the source isn't in any attached bundle.
        """
        scores: Dict[str, float] = {}
        for bundle in self._bundles:
            if bundle.matcher is None or not bundle.matcher.available:
                continue
            bundle_ids = source_ids & bundle.owned_source_ids
            if not bundle_ids:
                continue
            scores.update(bundle.matcher.score_sources(query_vec, bundle_ids))
        return scores

    def _semantic_embed_and_match(
        self,
        content: str,
        threshold: float,
        top_k: int,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[SemanticMatch]:
        """Embed ``content`` once and query every attached bundle.

        Runs the matrix-multiply per bundle, merges the per-bundle hits by
        score, and returns the global top-K. The single embedding avoids
        paying model cost per bundle.
        """
        if not self._embedding_provider or not self._embedding_provider.available:
            return []
        query_vec = self._embedding_provider.embed_text_as_array(content)
        if query_vec is None:
            return []

        all_matches: List[SemanticMatch] = []
        for bundle in self._bundles:
            if bundle.matcher is None or not bundle.matcher.available:
                continue
            matches = bundle.matcher.find_matches(
                query_vec,
                threshold=threshold,
                top_k=top_k,
                exclude_ids=exclude_ids,
            )
            all_matches.extend(matches)

        all_matches.sort(key=lambda m: m.score, reverse=True)
        return all_matches[:top_k]

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        self._trace("shutdown: cleaning up resources")
        if self._channel:
            self._channel.shutdown()
        self._channel = None
        self._sources = []
        self._selected_source_ids = []
        self._embedding_provider = None
        self._semantic_matcher = None
        for bundle in self._bundles:
            bundle.matcher = None
        self._bundles = []
        self._preselected_paths = {}
        self._transitive_parent_map = {}
        self._transitive_notification_pending = False
        self._initialized = False

        # Clear any authorized paths this plugin registered
        if self._plugin_registry:
            self._plugin_registry.clear_authorized_paths(self._name)

    def get_config_schema(self) -> dict:
        """Return JSON Schema for this plugin's configuration."""
        return {
            "type": "object",
            "properties": {
                "lookup_strategy": {
                    "type": "string",
                    "default": "hybrid",
                    "description": "Reference matching strategy",
                    "enum": ["hybrid", "tags_only", "semantic_only"],
                },
                "similarity_threshold": {
                    "type": "number",
                    "default": 0.75,
                    "description": "Minimum cosine similarity for semantic matching",
                },
                "tag_similarity_threshold": {
                    "type": "number",
                    "default": 0.4,
                    "description": "Minimum similarity for tag veto in hybrid mode",
                },
                "max_matches_per_piece": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum semantic matches per content piece",
                },
                "transitive_injection": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable transitive reference detection",
                },
                "preselected": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Source IDs to pre-select at startup",
                },
            },
        }

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool declarations for the references plugin.

        Tools can be excluded via the exclude_tools config option.
        """
        all_tools = [
            ToolSchema(
                name="selectReferences",
                description=(
                    "Select one or more reference sources by ID or by tags and return "
                    "their real resolved paths. A reference's path is only authorized "
                    "for readonly access after you select it with this tool — until "
                    "then the path is not accessible. Use listReferences first to "
                    "discover available IDs and tags. At least one of 'ids' or "
                    "'filter_tags' must be provided."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of reference source IDs to select. "
                                "Use listReferences to discover available IDs."
                            )
                        },
                        "filter_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Select all sources matching at least one of these tags."
                            )
                        }
                    },
                    "required": []
                },
                category="knowledge",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="listReferences",
                description=(
                    "List all available reference sources in the catalog, "
                    "including their access methods, tags, and current selection status. "
                    "Use this to discover what references are available before selecting."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "filter_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["all", "auto", "selectable"],
                            "description": "Filter by injection mode (default: all)"
                        }
                    },
                    "required": []
                },
                category="knowledge",
                discoverability="core",
            ),
            ToolSchema(
                name="validateReference",
                description=(
                    "Validate a single reference JSON file against the expected schema. "
                    "Checks for required fields, valid enum values, type-specific fields, "
                    "and tag format. Returns structured validation results with errors "
                    "and warnings."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to a reference JSON file to validate."
                        }
                    },
                    "required": ["path"]
                },
                category="knowledge",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="compute_embedding",
                description=(
                    "Compute a vector embedding for a text string or file contents. "
                    "Returns a float array representing the semantic meaning of the "
                    "input. Use this when building or updating reference indexes that "
                    "require semantic search capability."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": (
                                "The text to embed. Mutually exclusive with 'file'."
                            )
                        },
                        "file": {
                            "type": "string",
                            "description": (
                                "Path to a file whose contents should be embedded. "
                                "Mutually exclusive with 'input'. For large files, "
                                "content is truncated to the model's max input token limit."
                            )
                        }
                    },
                    "required": []
                },
                category="knowledge",
                discoverability="discoverable",
            ),
        ]

        # Filter out excluded tools
        if self._exclude_tools:
            return [t for t in all_tools if t.name not in self._exclude_tools]
        return all_tools

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return tool executors."""
        return {
            "selectReferences": self._execute_select,   # model tool
            "listReferences": self._execute_list,        # model tool
            "validateReference": self._execute_validate_reference,  # model tool
            "compute_embedding": self._execute_compute_embedding,  # model tool (gen-references agent)
            "references": self._execute_references_cmd,  # user command
        }

    def _execute_select(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model-driven reference selection by ID or tags.

        The model selects references directly — no user interaction is involved.
        Selected references have their paths authorized in the sandbox so the
        model can read them, and their resolved real paths are returned.

        When transitive injection is enabled, the content of each newly selected
        source is scanned for mentions of other catalog references. Any
        discovered references are automatically selected and included in the
        response with ``transitive: true`` and ``transitive_from`` fields.

        Args:
            args: Tool arguments with optional 'ids' (list of reference IDs)
                and/or 'filter_tags' (list of tags to match).

        Returns:
            Dict with status, selected sources (direct + transitive), and their
            resolved paths. Includes ``transitive_count`` when transitive
            sources were added.
        """
        ids = args.get("ids", [])
        filter_tags = args.get("filter_tags", [])
        self._trace(f"selectReferences: ids={ids}, filter_tags={filter_tags}")

        # Must provide at least one selection criterion
        if not ids and not filter_tags:
            return {
                "status": "error",
                "message": (
                    "At least one of 'ids' or 'filter_tags' must be provided. "
                    "Use listReferences to discover available reference IDs and tags."
                )
            }

        # Early check: no sources configured at all
        if not self._sources:
            return {
                "status": "no_sources",
                "message": "No reference sources available."
            }

        # Get sources not yet selected
        available = [
            s for s in self._sources
            if s.id not in self._selected_source_ids
        ]

        if not available:
            self._trace("selectReferences: all sources already selected")
            return {
                "status": "all_selected",
                "message": "All reference sources are already selected."
            }

        # Collect sources matching the criteria
        matched: List[ReferenceSource] = []
        matched_ids_set: set = set()

        # Match by explicit IDs
        if ids:
            available_by_id = {s.id: s for s in available}
            not_found = []
            already_selected = []
            for ref_id in ids:
                if ref_id in available_by_id:
                    if ref_id not in matched_ids_set:
                        matched.append(available_by_id[ref_id])
                        matched_ids_set.add(ref_id)
                elif ref_id in self._selected_source_ids:
                    already_selected.append(ref_id)
                else:
                    not_found.append(ref_id)

            if not_found:
                self._trace(f"selectReferences: IDs not found: {not_found}")
            if already_selected:
                self._trace(f"selectReferences: IDs already selected: {already_selected}")

        # Match by tags
        if filter_tags:
            for source in available:
                if source.id not in matched_ids_set:
                    if any(tag in source.tags for tag in filter_tags):
                        matched.append(source)
                        matched_ids_set.add(source.id)

        if not matched:
            self._trace("selectReferences: no sources matched criteria")
            # Build informative message
            parts = ["No unselected sources matched the criteria."]
            if ids:
                # Check which IDs were not found vs already selected
                all_source_ids = {s.id for s in self._sources}
                not_found = [i for i in ids if i not in all_source_ids]
                already = [i for i in ids if i in self._selected_source_ids]
                if not_found:
                    parts.append(f"IDs not found in catalog: {not_found}")
                if already:
                    parts.append(f"IDs already selected: {already}")
            return {
                "status": "none_matched",
                "message": " ".join(parts)
            }

        # Track selections and authorize paths.  When a kernel-layer
        # AppArmor fragment fails (confined WS only), roll back the
        # selection for that source — granting it would mislead the
        # model into believing it can read the file when the kernel
        # will refuse.  Other matched sources still proceed.
        selected_sources = []
        kernel_failed: List[Dict[str, str]] = []
        for source in matched:
            self._selected_source_ids.append(source.id)
            ok = self._authorize_source_path(source)
            if not ok:
                # Roll back the bookkeeping; sandbox_manager already
                # had the path added but the kernel will deny opens,
                # so leaving it selected would be a false promise.
                self._selected_source_ids.pop()
                resolved = self._resolve_path_for_access(source)
                kernel_failed.append({
                    "id": source.id,
                    "name": source.name,
                    "path": str(resolved) if resolved else source.path,
                })
                continue
            selected_sources.append(source)

        selected_ids = [s.id for s in selected_sources]
        self._trace(f"selectReferences: selected={selected_ids}")
        if kernel_failed:
            self._trace(
                f"selectReferences: kernel-layer authorization failed for "
                f"{[f['id'] for f in kernel_failed]}",
            )

        # Resolve transitive references from the newly selected sources
        transitive_sources = self._apply_transitive_selection(selected_ids)

        # Emit selection resolved hook for UI integration (include transitive)
        all_selected_ids = selected_ids + [s.id for s in transitive_sources]
        if self._on_selection_resolved:
            self._on_selection_resolved("selectReferences", all_selected_ids)

        # Build result with resolved paths for each source (direct + transitive)
        transitive_ids_set = {s.id for s in transitive_sources}
        source_results = []
        for source in selected_sources + transitive_sources:
            entry: Dict[str, Any] = {
                "id": source.id,
                "name": source.name,
                "description": source.description,
                "type": source.type.value,
                "tags": source.tags,
            }
            # Mark transitively included sources with their parent references
            if source.id in transitive_ids_set:
                entry["transitive"] = True
                parents = self._transitive_parent_map.get(source.id)
                if parents:
                    entry["transitive_from"] = sorted(parents)

            # Include resolved path for LOCAL sources
            if source.type == SourceType.LOCAL:
                resolved = self._resolve_path_for_access(source)
                entry["resolved_path"] = str(resolved) if resolved else source.path
                entry["is_directory"] = resolved.is_dir() if resolved else False
            elif source.type == SourceType.URL:
                entry["url"] = source.url
            elif source.type == SourceType.MCP:
                entry["server"] = source.server
                entry["tool"] = source.tool
                if source.args:
                    entry["args"] = source.args
            elif source.type == SourceType.INLINE:
                entry["content"] = source.content

            if source.fetch_hint:
                entry["fetch_hint"] = source.fetch_hint

            source_results.append(entry)

        # Status reflects whether everything actually worked.  A partial
        # failure (some kernel fragments rejected) needs a distinct
        # status so the model doesn't silently treat "selected but
        # unreadable" sources as available.
        if kernel_failed and not selected_sources:
            status = "kernel_authorization_failed"
        elif kernel_failed:
            status = "partial_success"
        else:
            status = "success"

        result: Dict[str, Any] = {
            "status": status,
            "selected_count": len(selected_sources),
            "sources": source_results,
        }
        if transitive_sources:
            result["transitive_count"] = len(transitive_sources)
        if kernel_failed:
            result["kernel_authorization_failed"] = kernel_failed
            result["kernel_authorization_failure_hint"] = (
                "AppArmor refused to grant readonly access to the path(s) "
                "above. Check the daemon log for 'AppArmor fragment "
                "rejected' or 'apparmor_parser reload failed' entries; "
                "common causes are unreadable parent directories or path "
                "characters that AppArmor treats as glob metacharacters."
            )

        result['_telemetry'] = {
            'jaato.references.operation': 'select',
            'jaato.references.selected_count': len(selected_sources),
            'jaato.references.kernel_failed_count': len(kernel_failed),
        }

        return result

    def _execute_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all available reference sources."""
        filter_tags = args.get("filter_tags", [])
        mode_filter = args.get("mode", "all")
        self._trace(f"listReferences: mode={mode_filter}, filter_tags={filter_tags}")

        # Early check: no sources configured at all
        if not self._sources:
            return {
                "sources": [],
                "total": 0,
                "selected_count": 0,
                "message": "No reference sources available."
            }

        sources = self._sources

        # Filter by mode
        if mode_filter == "auto":
            sources = [s for s in sources if s.mode == InjectionMode.AUTO]
        elif mode_filter == "selectable":
            sources = [s for s in sources if s.mode == InjectionMode.SELECTABLE]

        # Filter by tags
        if filter_tags:
            sources = [
                s for s in sources
                if any(tag in s.tags for tag in filter_tags)
            ]

        # Handle empty case with clear message
        if not sources:
            self._trace("listReferences: no sources match filters")
            return {
                "sources": [],
                "total": 0,
                "selected_count": 0,
                "message": "No reference sources available."
            }

        source_ids = [s.id for s in sources]
        self._trace(f"listReferences: returning {len(sources)} sources={source_ids}")

        source_entries = []
        for s in sources:
            entry = {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "type": s.type.value,
                "mode": s.mode.value,
                "tags": s.tags,
                "selected": s.id in self._selected_source_ids,
            }
            # Include resolved path for LOCAL sources so model knows real paths
            if s.type == SourceType.LOCAL:
                resolved = self._resolve_path_for_access(s)
                entry["resolved_path"] = str(resolved) if resolved else s.path
                entry["is_directory"] = resolved.is_dir() if resolved else False
            elif s.type == SourceType.URL:
                entry["url"] = s.url
            elif s.type == SourceType.MCP:
                entry["server"] = s.server
                entry["tool"] = s.tool
            elif s.type == SourceType.INLINE:
                entry["has_content"] = bool(s.content)
            source_entries.append(entry)

        return {
            "sources": source_entries,
            "total": len(sources),
            "selected_count": sum(
                1 for s in sources if s.id in self._selected_source_ids
            ),
            "_telemetry": {
                "jaato.references.operation": "list",
                "jaato.references.total": len(sources),
            },
        }

    def _execute_validate_reference(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single reference JSON file against the expected schema.

        Reads the file, parses it as JSON, and runs validate_reference_file()
        to check required fields, valid enum values, type-specific fields,
        and tag format.

        Args:
            args: Tool arguments with 'path' (string, required).

        Returns:
            Dict with 'valid', 'path', 'errors', and 'warnings' fields.
        """
        file_path = args.get("path", "")
        if not file_path:
            return {"valid": False, "path": "", "errors": ["'path' is required"], "warnings": []}

        # Resolve relative paths against project root
        path_obj = Path(file_path)
        if not path_obj.is_absolute() and self._project_root:
            path_obj = Path(self._project_root) / path_obj

        if not path_obj.exists():
            return {"valid": False, "path": str(path_obj), "errors": [f"File not found: {path_obj}"], "warnings": []}

        try:
            content = path_obj.read_text(encoding='utf-8')
        except (IOError, OSError) as e:
            return {"valid": False, "path": str(path_obj), "errors": [f"Cannot read file: {e}"], "warnings": []}

        try:
            import json
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return {"valid": False, "path": str(path_obj), "errors": [f"Invalid JSON: {e}"], "warnings": []}

        is_valid, errors, warnings = validate_reference_file(data)
        return {
            "valid": is_valid,
            "path": str(path_obj),
            "errors": errors,
            "warnings": warnings,
            "_telemetry": {
                "jaato.references.operation": "validate",
                "jaato.references.valid": is_valid,
                "jaato.references.error_count": len(errors),
                "jaato.references.warning_count": len(warnings),
            },
        }

    def _execute_compute_embedding(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compute a vector embedding for text or file contents.

        Called by the gen-references indexing agent to embed reference documents
        and store vectors in the sidecar matrix.  Also usable by any agent that
        needs to produce embeddings compatible with the reference index.

        Exactly one of ``input`` or ``file`` must be provided.

        Args:
            args: Tool arguments with ``input`` (str) or ``file`` (str path).

        Returns:
            Dict with ``embedding`` (list of floats), ``model``, ``dimensions``,
            and ``input_tokens`` keys.  On error, returns ``error`` key instead.
        """
        text_input = args.get("input")
        file_path = args.get("file")

        if text_input and file_path:
            return {"error": "'input' and 'file' are mutually exclusive — provide one, not both."}
        if not text_input and not file_path:
            return {"error": "Provide either 'input' (text) or 'file' (path)."}

        if not self._embedding_provider:
            return {
                "error": (
                    "No embedding provider available. Install a package that "
                    "registers a provider via the jaato.embedding entry point "
                    "(e.g. jaato-premium)."
                )
            }

        # Resolve file contents if file path given
        if file_path:
            path_obj = Path(file_path)
            if not path_obj.is_absolute() and self._project_root:
                path_obj = Path(self._project_root) / path_obj
            try:
                text_input = path_obj.read_text(encoding="utf-8")
            except (IOError, OSError) as e:
                return {"error": f"Cannot read file '{path_obj}': {e}"}

        result = self._embedding_provider.embed_text(text_input)
        if result is None:
            return {"error": "Embedding computation failed — provider returned None."}

        result_dict = result.to_dict()
        result_dict['_telemetry'] = {
            'jaato.references.operation': 'compute_embedding',
        }
        return result_dict

    def _execute_references_cmd(self, args: Dict[str, Any]) -> Any:
        """Execute the 'references' user command.

        Subcommands:
            list [all|selected|unselected]  - List reference sources
            select <ref-id>                 - Select a reference source
            unselect <ref-id>               - Unselect a reference source
            reload                          - Reload catalog from disk
            bundles                         - Show loaded knowledge bundles
            reconcile [<bundle>]            - Reconcile bundle sidecars
            help                            - Show usage help
        """
        subcommand = args.get("subcommand", "list")
        target = args.get("target", "")

        self._trace(f"references cmd: subcommand={subcommand}, target={target}")

        if subcommand == "list":
            return self._cmd_references_list(target)
        elif subcommand == "select":
            if not target:
                return {"error": "Usage: references select <ref-id>"}
            return self._cmd_references_select(target)
        elif subcommand == "unselect":
            if not target:
                return {"error": "Usage: references unselect <ref-id>"}
            return self._cmd_references_unselect(target)
        elif subcommand == "reload":
            return self._cmd_references_reload()
        elif subcommand == "bundles":
            return self._cmd_references_bundles()
        elif subcommand == "reconcile":
            return self._cmd_references_reconcile(target)
        elif subcommand == "merge":
            return self._cmd_references_merge(target)
        elif subcommand == "help":
            return self._cmd_references_help()
        else:
            return {
                "error": (
                    f"Unknown subcommand: {subcommand}. Use: list, select, "
                    f"unselect, reload, bundles, reconcile, merge, help"
                )
            }

    def _cmd_references_bundles(self) -> HelpLines:
        """Execute 'references bundles' — show loaded knowledge bundles.

        One row per bundle: tier, name, source count, model, dimensions,
        and current drift status. The tier column distinguishes
        workspace-tier bundles (``<workspace>/.jaato/references/``) from
        user-tier bundles (``~/.jaato/references/``); see :func:`discover_bundles`
        for shadowing semantics. Bundles without an attached matcher are
        flagged so the operator can see at a glance why semantic matching
        skips them.
        """
        lines: List[Tuple[str, str]] = [("BUNDLES", "bold"), ("", "")]
        if not self._bundles:
            lines.append(("    (no bundles — no embedding_config.json discovered)", ""))
            return HelpLines(lines=lines)

        # Stable ordering: workspace tier first (matching discovery order),
        # then user tier. Within each tier preserve the discovery order so
        # the root bundle still leads.
        ordered = sorted(
            self._bundles,
            key=lambda b: (
                0 if b.tier == BUNDLE_TIER_WORKSPACE else 1,
                self._bundles.index(b),
            ),
        )
        for bundle in ordered:
            own_count = sum(
                1 for s in self._sources if s.bundle_name == bundle.name
            )
            drift = detect_drift(bundle, self._sources)
            if bundle.matcher is None:
                status = "NO MATCHER"
            elif not drift.is_clean():
                status = drift.summary()
            else:
                status = "up-to-date"
            lines.append((
                f"  [{bundle.tier:<9}] "
                f"{bundle.display_name:<18} "
                f"{own_count:>3} refs  "
                f"model={bundle.embedding_model}  "
                f"dim={bundle.embedding_dimensions}  "
                f"{status}",
                "",
            ))
        return HelpLines(lines=lines)

    def _cmd_references_reconcile(self, target: str) -> Dict[str, Any]:
        """Execute 'references reconcile [<bundle-ref>] [--scope <tier>]'.

        Argument forms:

        * ``reconcile`` — reconcile every workspace-tier bundle (the
          default scope for write commands; user-tier reconciles must
          be explicit).
        * ``reconcile --scope user`` — reconcile every user-tier bundle.
        * ``reconcile --scope all`` — reconcile every loaded bundle.
        * ``reconcile <bundle-ref>`` — reconcile a single bundle. The
          ref is parsed by :func:`parse_bundle_ref`, so ``teammate``,
          ``workspace:teammate``, ``user:teammate``, ``root``, and
          ``user:(root)`` are all valid. Bare names that exist in both
          tiers raise :class:`AmbiguousBundleRefError` and the user
          must qualify.

        After a successful pass, re-attaches matchers so the next
        semantic query sees the new sidecar.
        """
        import shlex

        try:
            tokens = shlex.split(target or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}

        scope_filter: Optional[str] = None  # None = no filter, "all" handled below
        bundle_token: Optional[str] = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--scope":
                if i + 1 >= len(tokens):
                    return {"error": "--scope requires a value: workspace, user, or all"}
                value = tokens[i + 1]
                if value not in (*VALID_BUNDLE_TIERS, "all"):
                    return {
                        "error": (
                            f"Unknown scope {value!r}. Use 'workspace', 'user', or 'all'."
                        )
                    }
                scope_filter = value
                i += 2
                continue
            if tok.startswith("--scope="):
                value = tok.split("=", 1)[1]
                if value not in (*VALID_BUNDLE_TIERS, "all"):
                    return {
                        "error": (
                            f"Unknown scope {value!r}. Use 'workspace', 'user', or 'all'."
                        )
                    }
                scope_filter = value
                i += 1
                continue
            if bundle_token is not None:
                return {
                    "error": (
                        "Usage: references reconcile [<bundle-ref>] [--scope workspace|user|all]"
                    )
                }
            bundle_token = tok
            i += 1

        # A single bundle and a --scope filter are mutually exclusive — the
        # bundle ref already pins down a specific tier (or trips ambiguity).
        if bundle_token is not None and scope_filter is not None:
            return {
                "error": (
                    "Cannot combine a bundle reference with --scope; either "
                    "pick one bundle (e.g. 'workspace:teammate') or pick a "
                    "scope (e.g. '--scope user')."
                )
            }

        # Resolve target → list of bundles to reconcile.
        if bundle_token is not None:
            try:
                ref = parse_bundle_ref(bundle_token)
            except ValueError as e:
                return {"error": str(e)}
            try:
                hit = find_bundle(
                    self._bundles, ref, default_scope=BUNDLE_TIER_WORKSPACE,
                )
            except AmbiguousBundleRefError as e:
                return {"error": str(e)}
            if hit is None:
                return {
                    "error": (
                        f"Unknown bundle '{ref.display}'. Known bundles: "
                        f"{[b.qualified_ref for b in self._bundles] or '(none)'}"
                    )
                }
            candidates = [hit]
        else:
            # No bundle ref → scope filter applies. Default scope for the
            # write command is "workspace" (matches the contract documented
            # in the bundle module: write commands default to workspace).
            effective_scope = scope_filter or BUNDLE_TIER_WORKSPACE
            if effective_scope == "all":
                candidates = list(self._bundles)
            else:
                candidates = [b for b in self._bundles if b.tier == effective_scope]

        if not candidates:
            return {
                "status": "no_bundles",
                "message": "No bundles to reconcile (no embedding_config.json discovered).",
            }

        results: List[ReconcileResult] = []
        for bundle in candidates:
            results.append(
                reconcile_bundle(
                    bundle, self._sources, self._embedding_provider,
                )
            )

        # Re-attach matchers for any bundle whose sidecar changed.
        if self._lookup_strategy in ("hybrid", "semantic_only"):
            provider_model = (
                self._embedding_provider.model_name
                if self._embedding_provider
                else ""
            )
            for bundle, result in zip(candidates, results):
                if result.status in (ReconcileStatus.UPDATED, ReconcileStatus.CLEAN):
                    bundle.matcher = self._attach_matcher(bundle, provider_model)
            root = next(
                (b for b in self._bundles if b.name == ROOT_BUNDLE_NAME),
                None,
            )
            self._semantic_matcher = root.matcher if root else None

        lines: List[Tuple[str, str]] = [("RECONCILE", "bold"), ("", "")]
        for bundle, result in zip(candidates, results):
            # Prefix with tier so logs distinguish workspace and user
            # entries when the same bundle name exists in both — even if
            # discovery shadows them today, an --scope all run still
            # benefits from the disambiguation.
            lines.append((f"  [{bundle.tier}] {result.summary()}", ""))
            for sid, reason in result.skipped:
                lines.append((f"    skipped {sid}: {reason}", ""))

        return {
            "status": "ok",
            "results": [
                {
                    "bundle": r.bundle_name or "(root)",
                    "tier": b.tier,
                    "status": r.status.value,
                    "added": r.added,
                    "refreshed": r.refreshed,
                    "pruned": r.pruned,
                    "skipped": r.skipped,
                    "final_row_count": r.final_row_count,
                }
                for b, r in zip(candidates, results)
            ],
            "help_lines": HelpLines(lines=lines),
        }

    def _cmd_references_merge(self, raw_args: str) -> Dict[str, Any]:
        """Execute 'references merge <source-ref> [--into <target-ref>] [flags]'.

        Source and target both accept the ``[<scope>:]<name>`` syntax
        parsed by :func:`parse_bundle_ref` (e.g., ``teammate``,
        ``workspace:teammate``, ``user:teammate``). Bare names default
        to the workspace tier when both tiers contain a bundle of that
        name. Source can also be a directory path (used for unloaded
        bundles, e.g. a tarball that's been unpacked outside any tier
        root). Target defaults to ``workspace:(root)``.

        Cross-tier merges are supported — e.g. ``merge user:personal
        --into workspace:project`` lifts a user-tier bundle into the
        current workspace.

        On success the in-memory catalog is updated so the model sees
        the merged sources without waiting for a reload.

        Args:
            raw_args: The ``target`` parameter from :class:`UserCommand`,
                which captures the rest of the command line.
        """
        import shlex

        try:
            tokens = shlex.split(raw_args or "")
        except ValueError as e:
            return {"error": f"Failed to parse arguments: {e}"}

        source_arg, target_arg, options, parse_err = parse_merge_args(tokens)
        if parse_err:
            return {"error": parse_err}
        if not source_arg:
            return {
                "error": (
                    "Usage: references merge <source-ref> [--into <target-ref>] "
                    "[--on-conflict reject|prefix|newer] [--re-embed] [--dry-run]"
                )
            }

        # Resolve target bundle. Defaults to the workspace-tier root,
        # matching the documented "write commands default to workspace"
        # rule. When the user wrote ``--into user:notes`` (or any
        # ``scope:name`` form), the parsed BundleRef pins the tier.
        if target_arg:
            try:
                target_ref = parse_bundle_ref(target_arg)
            except ValueError as e:
                return {"error": f"--into: {e}"}
        else:
            target_ref = BundleRef(name=ROOT_BUNDLE_NAME, scope=BUNDLE_TIER_WORKSPACE)

        try:
            target_bundle = find_bundle(
                self._bundles, target_ref, default_scope=BUNDLE_TIER_WORKSPACE,
            )
        except AmbiguousBundleRefError as e:
            return {"error": f"--into: {e}"}
        if target_bundle is None:
            return {
                "error": (
                    f"Unknown target bundle '{target_ref.display}'. "
                    f"Loaded bundles: "
                    f"{[b.qualified_ref for b in self._bundles] or '(none)'}"
                )
            }

        # Resolve source: first try as a parsed BundleRef against the
        # loaded catalog, then fall back to a directory path. Path
        # fallback also covers the cross-tier-same-name case where a
        # user bundle was shadowed by a workspace bundle of the same
        # name and isn't visible to find_bundle.
        source_bundle: Optional[Bundle]
        source_sources: List[ReferenceSource]
        resolved_source_name: str

        # First, try bundle-ref resolution. parse_bundle_ref rejects
        # paths-with-colons that aren't valid scopes, so things like
        # ``./teammate:1.0`` won't be misinterpreted.
        source_lookup: Optional[Bundle] = None
        try:
            source_ref = parse_bundle_ref(source_arg)
        except ValueError:
            source_ref = None

        if source_ref is not None:
            try:
                source_lookup = find_bundle(
                    self._bundles, source_ref, default_scope=BUNDLE_TIER_WORKSPACE,
                )
            except AmbiguousBundleRefError as e:
                return {"error": str(e)}

        if source_lookup is not None:
            # Compare identity by (tier, name) — a workspace and a user
            # bundle that happen to share a name are still different
            # bundles for the purposes of merge.
            if (
                source_lookup.name == target_bundle.name
                and source_lookup.tier == target_bundle.tier
            ):
                return {
                    "error": (
                        f"Source and target are the same bundle "
                        f"('{target_bundle.qualified_ref}'); nothing to merge."
                    )
                }
            source_bundle = source_lookup
            source_sources = [
                s for s in self._sources if s.bundle_name == source_lookup.name
            ]
            resolved_source_name = source_lookup.qualified_ref
        else:
            source_path = Path(source_arg)
            if not source_path.is_absolute():
                if self._workspace_path:
                    source_path = Path(self._workspace_path) / source_path
                elif self._project_root:
                    source_path = Path(self._project_root) / source_path
            if not source_path.is_dir():
                return {
                    "error": (
                        f"Source '{source_arg}' is neither a loaded bundle nor a "
                        f"directory on disk. Loaded bundles: "
                        f"{[b.qualified_ref for b in self._bundles] or '(none)'}"
                    )
                }
            source_bundle = _read_bundle_manifest(source_path)
            if source_bundle is None:
                return {
                    "error": (
                        f"Source directory '{source_path}' has no "
                        f"embedding_config.json; cannot merge."
                    )
                }
            source_sources = _load_sources_from_dir(source_path)
            for s in source_sources:
                s.bundle_name = source_bundle.name
            resolved_source_name = str(source_path)

        target_sources = [
            s for s in self._sources if s.bundle_name == target_bundle.name
        ]

        result = merge_bundle(
            target=target_bundle,
            source_bundle=source_bundle,
            source_sources=source_sources,
            target_sources=target_sources,
            provider=self._embedding_provider,
            options=options,
        )

        self._trace(f"references merge: {result.summary()}")

        # On a real (non-dry-run) success, update the in-memory catalog
        # so the next prompt/tool call sees the merged sources without
        # waiting for a reload. When the source was a loaded bundle, we
        # remove its stale entries from the catalog first — those refs
        # have been physically copied into the target, so the target's
        # copy is authoritative from here on. (The source directory's
        # JSON files remain in place; the operator can rm them.)
        if result.status == MergeStatus.OK and result.added:
            rename_lookup = dict(result.renamed)  # old_id → new_id
            merged_source_ids = {s.id for s in source_sources}
            source_bundle_name = source_bundle.name if source_lookup else None

            if source_bundle_name is not None:
                # Drop the source-bundle copies of merged ids from the
                # live catalog. Ids left behind (e.g., skipped by --newer)
                # stay attached to the source bundle until reload.
                self._sources = [
                    s for s in self._sources
                    if not (
                        s.bundle_name == source_bundle_name
                        and s.id in merged_source_ids
                        and rename_lookup.get(s.id, s.id) in result.added
                    )
                ]

            existing_ids = {s.id for s in self._sources}
            for s in source_sources:
                new_id = rename_lookup.get(s.id, s.id)
                if new_id not in result.added or new_id in existing_ids:
                    continue
                # Reload the fresh JSON the merge just wrote — it is the
                # source of truth now (has the right id + source_hash).
                fresh_path = target_bundle.directory / f"{new_id}.json"
                if fresh_path.is_file():
                    try:
                        raw = json.loads(fresh_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    fresh = ReferenceSource.from_dict(raw)
                    fresh.bundle_name = target_bundle.name
                    self._resolve_source_for_context(fresh)
                    self._sources.append(fresh)
                    existing_ids.add(new_id)

            # Re-attach target's matcher so the new rows become queryable.
            if self._lookup_strategy in ("hybrid", "semantic_only") \
                    and self._embedding_provider is not None:
                target_bundle.matcher = self._attach_matcher(
                    target_bundle, self._embedding_provider.model_name,
                )
                # Keep the legacy ``self._semantic_matcher`` shim in sync
                # only when the target is the *workspace*-tier root —
                # that's the bundle the legacy field has always tracked.
                # A user-tier root merge never updates the shim.
                if (
                    target_bundle.name == ROOT_BUNDLE_NAME
                    and target_bundle.tier == BUNDLE_TIER_WORKSPACE
                ):
                    self._semantic_matcher = target_bundle.matcher

        # Build the HelpLines block for display.
        lines: List[Tuple[str, str]] = [("MERGE", "bold"), ("", "")]
        lines.append((f"  {result.summary()}", ""))
        if result.renamed:
            lines.append(("  renames:", "dim"))
            for old, new in result.renamed:
                lines.append((f"    {old} → {new}", "dim"))
        if result.skipped:
            lines.append(("  skipped:", "dim"))
            for sid, reason in result.skipped:
                lines.append((f"    {sid}: {reason}", "dim"))
        if result.conflicts:
            lines.append(("  conflicts:", "dim"))
            for sid in result.conflicts:
                lines.append((f"    {sid}", "dim"))

        payload: Dict[str, Any] = {
            "status": result.status.value,
            "source": resolved_source_name,
            "target": target_bundle.qualified_ref,
            "target_tier": target_bundle.tier,
            "added": result.added,
            "renamed": result.renamed,
            "skipped": result.skipped,
            "conflicts": result.conflicts,
            "final_row_count": result.final_row_count,
            "help_lines": HelpLines(lines=lines),
        }
        if result.error:
            payload["error"] = result.error
        return payload

    def _cmd_references_list(self, filter_arg: str) -> HelpLines:
        """Execute 'references list [all|selected|unselected]'."""
        filter_arg = filter_arg.strip().lower() if filter_arg else "all"

        if filter_arg == "selected":
            sources = [s for s in self._sources if s.id in self._selected_source_ids]
        elif filter_arg == "unselected":
            sources = [s for s in self._sources if s.id not in self._selected_source_ids]
        else:
            sources = self._sources

        return self._format_list_as_help_lines(sources, filter_arg)

    def _cmd_references_select(self, ref_id: str) -> Dict[str, Any]:
        """Execute 'references select <ref-id>'.

        Selects the reference and runs transitive resolution to automatically
        include any references mentioned within the selected source's content.
        """
        ref_id = ref_id.strip()

        # Look up the source
        source = next((s for s in self._sources if s.id == ref_id), None)
        if not source:
            return {"error": f"Reference '{ref_id}' not found."}

        if ref_id in self._selected_source_ids:
            return {"status": "already_selected", "message": f"Reference '{ref_id}' is already selected."}

        self._selected_source_ids.append(ref_id)
        if not self._authorize_source_path(source):
            # Kernel-layer (AppArmor) refused; roll back the selection
            # so the user doesn't see "selected" for a reference whose
            # files won't actually be readable.
            self._selected_source_ids.pop()
            resolved = self._resolve_path_for_access(source)
            return {
                "error": (
                    f"Reference '{ref_id}' could not be authorized at the "
                    f"kernel layer (AppArmor refused {resolved}). Check "
                    f"the daemon log for 'AppArmor fragment rejected' "
                    f"or 'apparmor_parser reload failed' entries."
                ),
            }
        self._trace(f"references select: selected '{ref_id}'")

        # Resolve transitive references from the newly selected source
        transitive_sources = self._apply_transitive_selection([ref_id])

        result: Dict[str, Any] = {
            "status": "selected",
            "message": f"Selected reference '{source.name}' ({ref_id}).",
            "source": source.to_instruction(),
        }
        if transitive_sources:
            transitive_ids = [s.id for s in transitive_sources]
            result["transitive_count"] = len(transitive_ids)
            result["transitive_ids"] = transitive_ids
            result["message"] += (
                f" Also transitively included {len(transitive_ids)} "
                f"referenced source(s): {', '.join(transitive_ids)}."
            )

        return result

    def _cmd_references_unselect(self, ref_id: str) -> Dict[str, Any]:
        """Execute 'references unselect <ref-id>'.

        Removes the reference from the selected set and deauthorizes its
        path from the sandbox so the model can no longer access it.
        """
        ref_id = ref_id.strip()

        if ref_id not in self._selected_source_ids:
            return {"error": f"Reference '{ref_id}' is not currently selected."}

        self._selected_source_ids.remove(ref_id)

        # Deauthorize the path so the model can no longer access it
        source = next((s for s in self._sources if s.id == ref_id), None)
        if source:
            self._deauthorize_source_path(source)

        self._trace(f"references unselect: unselected '{ref_id}'")

        name = source.name if source else ref_id
        return {
            "status": "unselected",
            "message": f"Unselected reference '{name}' ({ref_id}).",
        }

    def _cmd_references_reload(self) -> Dict[str, Any]:
        """Execute 'references reload'.

        Reloads the reference catalog from disk (config files and
        .jaato/references/ directory).  Previously selected sources are
        preserved when they still exist in the reloaded catalog; selections
        whose IDs are no longer present are dropped and their sandbox
        authorizations revoked.

        After reloading, transitive resolution is re-applied for any
        surviving selections.
        """
        workspace = self._workspace_path or self._project_root
        if not workspace:
            return {"error": "Cannot reload: no workspace path available."}

        # Snapshot previous state
        prev_ids = set(s.id for s in self._sources)
        prev_selected = list(self._selected_source_ids)

        # Deauthorize all currently-selected paths before reloading
        for sid in self._selected_source_ids:
            source = next((s for s in self._sources if s.id == sid), None)
            if source:
                self._deauthorize_source_path(source)

        # Clear authorized paths registered by this plugin
        if self._plugin_registry:
            self._plugin_registry.clear_authorized_paths(self._name)

        # Reload catalog from disk
        self._reload_catalog(workspace)

        new_ids = set(s.id for s in self._sources)
        added = new_ids - prev_ids
        removed = prev_ids - new_ids

        # Restore selections that still exist in the reloaded catalog
        surviving_selected = [sid for sid in prev_selected if sid in new_ids]
        dropped_selected = [sid for sid in prev_selected if sid not in new_ids]
        self._selected_source_ids = surviving_selected

        # Re-authorize paths for surviving selections
        for sid in surviving_selected:
            source = next((s for s in self._sources if s.id == sid), None)
            if source:
                self._authorize_source_path(source)

        # Re-apply transitive resolution for surviving selections
        self._transitive_parent_map = {}
        self._transitive_notification_pending = False
        if self._transitive_enabled and surviving_selected:
            full_catalog = {s.id: s for s in self._sources}
            all_resolved, transitive_parent_map = self._resolve_transitive_references(
                surviving_selected, full_catalog
            )
            self._transitive_parent_map = transitive_parent_map
            if transitive_parent_map:
                self._transitive_notification_pending = True

            current_source_ids = {s.id for s in self._sources}
            for ref_id in all_resolved:
                if ref_id not in self._selected_source_ids:
                    self._selected_source_ids.append(ref_id)
                if ref_id not in current_source_ids and ref_id in full_catalog:
                    source = full_catalog[ref_id]
                    self._resolve_source_for_context(source)
                    self._sources.append(source)
                    current_source_ids.add(ref_id)
                # Authorize transitively added sources
                src = next((s for s in self._sources if s.id == ref_id), None)
                if src and ref_id not in surviving_selected:
                    self._authorize_source_path(src)

        self._trace(
            f"references reload: sources={len(self._sources)}, "
            f"added={len(added)}, removed={len(removed)}, "
            f"selected={len(self._selected_source_ids)} "
            f"(dropped {len(dropped_selected)} stale selections)"
        )

        result: Dict[str, Any] = {
            "status": "reloaded",
            "total_sources": len(self._sources),
            "message": f"Reloaded {len(self._sources)} reference(s) from disk.",
        }
        if added:
            result["added"] = sorted(added)
            result["message"] += f" Added: {', '.join(sorted(added))}."
        if removed:
            result["removed"] = sorted(removed)
            result["message"] += f" Removed: {', '.join(sorted(removed))}."
        if dropped_selected:
            result["dropped_selected"] = sorted(dropped_selected)
            result["message"] += (
                f" Dropped {len(dropped_selected)} stale selection(s): "
                f"{', '.join(sorted(dropped_selected))}."
            )

        return result

    def _format_list_as_help_lines(self, sources: List[ReferenceSource], filter_label: str) -> HelpLines:
        """Format a list of reference sources as HelpLines for pager display."""
        lines: List[tuple] = []

        lines.append(("Reference Sources", "bold"))
        lines.append(("", ""))

        if not sources:
            lines.append((f"  No {filter_label} references found.", "dim"))
            return HelpLines(lines=lines)

        selected_set = set(self._selected_source_ids)

        lines.append((f"  Showing: {filter_label} ({len(sources)} source(s))", "dim"))
        lines.append(("", ""))

        for source in sources:
            is_selected = source.id in selected_set
            status = "[selected]" if is_selected else "[unselected]"

            lines.append((f"  {source.id}  {status}", "bold"))
            lines.append((f"    Name:        {source.name}", ""))
            lines.append((f"    Description: {source.description}", "dim"))
            lines.append((f"    Type: {source.type.value}  |  Mode: {source.mode.value}", "dim"))
            if source.tags:
                lines.append((f"    Tags: {', '.join(source.tags)}", "dim"))
            lines.append((f"    Access: {self._get_access_summary(source)}", "dim"))
            lines.append(("", ""))

        return HelpLines(lines=lines)

    def _cmd_references_help(self) -> HelpLines:
        """Return detailed help text for pager display."""
        return HelpLines(lines=[
            ("References Command", "bold"),
            ("", ""),
            ("Manage reference sources for the current session.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    references [subcommand] [target]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list [all|selected|unselected]", "dim"),
            ("        List reference sources, optionally filtered by selection status.", "dim"),
            ("        Default: all", "dim"),
            ("", ""),
            ("    select <ref-id>", "dim"),
            ("        Select a reference source by ID. The source's content instructions", "dim"),
            ("        are returned so the model can fetch and incorporate them.", "dim"),
            ("", ""),
            ("    unselect <ref-id>", "dim"),
            ("        Unselect a previously selected reference source.", "dim"),
            ("", ""),
            ("    reload", "dim"),
            ("        Reload the reference catalog from disk. Picks up new, changed,", "dim"),
            ("        or removed reference files without restarting the session.", "dim"),
            ("        Previously selected sources are preserved when they still", "dim"),
            ("        exist in the reloaded catalog.", "dim"),
            ("", ""),
            ("    bundles", "dim"),
            ("        Show loaded knowledge bundles. Each bundle is a directory", "dim"),
            ("        (root or subdirectory) with its own embedding_config.json", "dim"),
            ("        and sidecar matrix. The tier column distinguishes workspace", "dim"),
            ("        bundles (./jaato/references) from user bundles (~/.jaato/references).", "dim"),
            ("", ""),
            ("    reconcile [<bundle-ref>] [--scope workspace|user|all]", "dim"),
            ("        Bring a bundle's sidecar in sync with the catalog: embed", "dim"),
            ("        newly dropped refs, refresh stale ones, drop orphans.", "dim"),
            ("        With no argument, reconciles every workspace-tier bundle.", "dim"),
            ("        Use --scope user or --scope all to widen. <bundle-ref>", "dim"),
            ("        accepts 'name', 'workspace:name', or 'user:name'.", "dim"),
            ("", ""),
            ("    merge <source-ref> [--into <target-ref>] [flags]", "dim"),
            ("        Merge a knowledge bundle into another. <source-ref> is", "dim"),
            ("        either '[<scope>:]<name>' for a loaded bundle or a", "dim"),
            ("        directory path containing an embedding_config.json.", "dim"),
            ("        Default target is workspace:(root). Cross-tier merges", "dim"),
            ("        are supported: 'merge user:notes --into workspace:project'.", "dim"),
            ("        Flags:", "dim"),
            ("          --on-conflict reject|prefix|newer  (default: reject)", "dim"),
            ("          --re-embed                        (needed on model/dim mismatch)", "dim"),
            ("          --dry-run                         (preview without writing)", "dim"),
            ("", ""),
            ("    help", "dim"),
            ("        Show this help message.", "dim"),
            ("", ""),
            ("BUNDLE REFERENCES", "bold"),
            ("    Most subcommands accept a bundle reference of the form:", "dim"),
            ("        [<scope>:]<name>", "dim"),
            ("    where <scope> is 'workspace' or 'user' and <name> is a bundle", "dim"),
            ("    directory name (or 'root' / '(root)' for the root bundle).", "dim"),
            ("    Bare names are resolved against the workspace tier first.", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    references                              List all references", "dim"),
            ("    references list selected                Show only selected references", "dim"),
            ("    references select my-ref-001            Select a reference by ID", "dim"),
            ("    references reload                       Reload catalog from disk", "dim"),
            ("    references bundles                      Show loaded bundles", "dim"),
            ("    references reconcile                    Reconcile workspace bundles", "dim"),
            ("    references reconcile --scope all        Reconcile every loaded bundle", "dim"),
            ("    references reconcile user:teammate      Reconcile a single user bundle", "dim"),
            ("    references merge teammate               Merge a workspace bundle into root", "dim"),
            ("    references merge user:notes             Lift a user bundle into the workspace", "dim"),
            ("    references merge user:notes --into workspace:project", "dim"),
            ("    references merge ./incoming --on-conflict prefix", "dim"),
        ])

    def _get_access_summary(self, source: ReferenceSource) -> str:
        """Get brief access method description."""
        from .models import SourceType

        if source.type == SourceType.LOCAL:
            return f"File: {source.path}"
        elif source.type == SourceType.URL:
            return f"URL: {source.url}"
        elif source.type == SourceType.MCP:
            return f"MCP: {source.server}/{source.tool}"
        elif source.type == SourceType.INLINE:
            return "Inline content"
        return "Unknown"

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions with AUTO and pre-selected sources.

        AUTO sources and pre-selected sources are included in system instructions
        so the model knows to fetch them at the start of the session.
        """
        auto_sources = [
            s for s in self._sources
            if s.mode == InjectionMode.AUTO
        ]

        # Get pre-selected sources (selectable sources that were pre-selected via config)
        preselected_sources = [
            s for s in self._sources
            if s.mode == InjectionMode.SELECTABLE and s.id in self._selected_source_ids
        ]

        # Sources to fetch immediately = AUTO + pre-selected
        immediate_sources = auto_sources + preselected_sources

        auto_ids = [s.id for s in auto_sources]
        preselected_ids = [s.id for s in preselected_sources]
        self._trace(f"get_system_instructions: auto={auto_ids}, preselected={preselected_ids}")

        if not immediate_sources:
            # Still provide info about selectable sources (if selectReferences is available)
            selectable = [
                s for s in self._sources
                if s.mode == InjectionMode.SELECTABLE
                and s.id not in self._selected_source_ids
            ]
            # If selectReferences is excluded or no selectable sources, nothing to show
            if not selectable or "selectReferences" in self._exclude_tools:
                self._trace("get_system_instructions: no sources to inject")
                return None

            parts = [
                "# Reference Sources",
                "",
                "Additional reference sources are available for this session.",
            ]
            if "listReferences" not in self._exclude_tools:
                parts.append("Use `listReferences` to see available sources, their tags, and resolved paths.")
            parts.extend([
                "Use `selectReferences` with specific IDs or tags to select sources and",
                "get their resolved paths. IMPORTANT: A reference's real path is only",
                "authorized for readonly access AFTER you select it — until then its path",
                "is not accessible even if you know it from listReferences.",
                "",
                "When reporting sources from listReferences, always indicate selection status:",
                "- 'available but unselected' for sources not yet selected",
                "- 'selected' for sources already included",
                "",
                "Available tags: " + ", ".join(
                    sorted(set(tag for s in selectable for tag in s.tags))
                ),
            ])
            selectable_ids = [s.id for s in selectable]
            self._trace(f"get_system_instructions: injecting selectable hints={selectable_ids}")
            return "\n".join(parts)

        parts = [
            "# Reference Sources",
            "",
            "The following reference sources should be incorporated into your context.",
            "Fetch their content using the appropriate tools as described.",
            ""
        ]

        for source in immediate_sources:
            parts.append(source.to_instruction())
            # Annotate transitively selected sources so the model knows why
            # they were included and which parent source referenced them
            if source.id in self._transitive_parent_map:
                parents = self._transitive_parent_map[source.id]
                parent_refs = ", ".join(f"@{p}" for p in sorted(parents))
                parts.append(f"*(Transitively included — referenced by {parent_refs})*")
            parts.append("")

        # Mention remaining selectable sources (not pre-selected) if any
        # Only show if selectReferences tool is available
        if "selectReferences" not in self._exclude_tools:
            selectable = [
                s for s in self._sources
                if s.mode == InjectionMode.SELECTABLE
                and s.id not in self._selected_source_ids
            ]
            if selectable:
                parts.extend([
                    "---",
                    "",
                    "Additional reference sources are available.",
                    "Use `selectReferences` with IDs or tags to select them — their paths",
                    "become readonly-accessible only after selection.",
                    "Available tags: " + ", ".join(sorted(set(tag for s in selectable for tag in s.tags))),
                ])

        immediate_ids = [s.id for s in immediate_sources]
        self._trace(f"get_system_instructions: injecting immediate sources={immediate_ids}")
        return "\n".join(parts)

    def get_auto_approved_tools(self) -> List[str]:
        """All tools are auto-approved - this is a user-triggered plugin."""
        return ["selectReferences", "listReferences", "validateReference", "compute_embedding", "references"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for direct invocation.

        Single 'references' command with subcommands: list, select, unselect, reload.
        share_with_model=True so the model sees selection changes.
        """
        return [
            UserCommand(
                name="references",
                description="Manage reference sources (list|select|unselect|reload)",
                share_with_model=True,
                parameters=[
                    CommandParameter(
                        name="subcommand",
                        description="Action: list, select, unselect, or reload",
                        required=False,
                    ),
                    CommandParameter(
                        name="target",
                        description="Filter or reference ID",
                        required=False,
                        capture_rest=True,
                    ),
                ],
            ),
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for the references command.

        Provides multi-level completions:
        - Level 1: list, select, unselect
        - Level 2 for list: all, selected, unselected
        - Level 2 for select: IDs of unselected selectable sources
        - Level 2 for unselect: IDs of currently selected sources
        """
        if command != "references":
            return []

        subcommands = [
            CommandCompletion("list", "List reference sources"),
            CommandCompletion("select", "Select a reference source"),
            CommandCompletion("unselect", "Unselect a reference source"),
            CommandCompletion("reload", "Reload catalog from disk"),
            CommandCompletion("bundles", "Show loaded knowledge bundles"),
            CommandCompletion("reconcile", "Reconcile bundle sidecars"),
            CommandCompletion("merge", "Merge a bundle into another"),
            CommandCompletion("help", "Show detailed help"),
        ]

        if len(args) <= 1:
            if args:
                partial = args[0].lower()
                return [s for s in subcommands if s.value.startswith(partial)]
            return subcommands

        if len(args) == 2:
            subcommand = args[0].lower()
            partial = args[1].lower()

            if subcommand == "list":
                filters = [
                    CommandCompletion("all", "Show all references"),
                    CommandCompletion("selected", "Show only selected references"),
                    CommandCompletion("unselected", "Show only unselected references"),
                ]
                return [f for f in filters if f.value.startswith(partial)]

            if subcommand == "select":
                selected_set = set(self._selected_source_ids)
                options = [
                    CommandCompletion(s.id, s.name)
                    for s in self._sources
                    if s.id not in selected_set
                ]
                return [o for o in options if o.value.startswith(partial)]

            if subcommand == "unselect":
                selected_set = set(self._selected_source_ids)
                options = [
                    CommandCompletion(s.id, s.name)
                    for s in self._sources
                    if s.id in selected_set
                ]
                return [o for o in options if o.value.startswith(partial)]

            if subcommand == "reconcile":
                options = self._bundle_ref_completions(
                    description_prefix="Reconcile",
                    include_root=True,
                )
                return [o for o in options if o.value.startswith(partial)]

            if subcommand == "merge":
                # Suggest loaded bundles (other than the workspace root)
                # as first-argument completions for ``merge <source>``.
                # The workspace root is omitted because the default
                # target is also the workspace root and merging a
                # bundle into itself isn't meaningful; path-based
                # sources aren't enumerable so the user types them
                # manually.
                options = self._bundle_ref_completions(
                    description_prefix="Merge",
                    include_root=True,
                    exclude_workspace_root=True,
                )
                return [o for o in options if o.value.startswith(partial)]

        # Reconcile flag-value completions: ``reconcile --scope <tier>``.
        if len(args) >= 2 and args[0].lower() == "reconcile":
            partial = args[-1].lower()
            if len(args) >= 3 and args[-2] == "--scope":
                scopes = [
                    CommandCompletion("workspace", "Workspace tier (default)"),
                    CommandCompletion("user", "User tier (~/.jaato/references)"),
                    CommandCompletion("all", "Both tiers"),
                ]
                return [s for s in scopes if s.value.startswith(partial)]
            if partial.startswith("-") or not partial:
                flags = [CommandCompletion("--scope", "Filter by tier")]
                return [f for f in flags if f.value.startswith(partial or "-")]

        # Trailing-flag completions for 'references merge'.
        if len(args) >= 2 and args[0].lower() == "merge":
            partial = args[-1].lower()
            # Flag names come first.
            flags = [
                CommandCompletion("--into", "Target bundle (default: workspace:(root))"),
                CommandCompletion("--on-conflict", "reject | prefix | newer"),
                CommandCompletion("--re-embed", "Re-embed source against target model"),
                CommandCompletion("--dry-run", "Preview without writing"),
            ]
            if partial.startswith("-") or not partial:
                return [f for f in flags if f.value.startswith(partial or "-")]
            # Value completions for the flag immediately before us.
            if len(args) >= 3 and args[-2] == "--on-conflict":
                vals = [
                    CommandCompletion("reject", "Abort on id collision (default)"),
                    CommandCompletion("prefix", "Rename incoming ids"),
                    CommandCompletion("newer", "Keep the fresher side"),
                ]
                return [v for v in vals if v.value.startswith(partial)]
            if len(args) >= 3 and args[-2] == "--into":
                options = self._bundle_ref_completions(
                    description_prefix="Into",
                    include_root=True,
                )
                return [o for o in options if o.value.startswith(partial)]

        return []

    def _bundle_ref_completions(
        self,
        *,
        description_prefix: str,
        include_root: bool,
        exclude_workspace_root: bool = False,
    ) -> List[CommandCompletion]:
        """Build CommandCompletions for the loaded bundles.

        Each bundle is offered as a bare name (when unambiguous across
        tiers) and as a ``scope:name`` form (always). The ``root`` alias
        is offered for root bundles instead of the empty-string sentinel.

        Args:
            description_prefix: Verb used in the completion description
                (e.g., ``"Reconcile"`` → ``"Reconcile workspace:teammate"``).
            include_root: Whether to offer the root bundle alias.
            exclude_workspace_root: When True, omit the workspace-tier
                root bundle (used by ``merge`` because merging the root
                into itself isn't meaningful).
        """
        # Collect names to detect cross-tier ambiguity.
        from collections import Counter
        name_counts: Counter = Counter(b.name for b in self._bundles)

        completions: List[CommandCompletion] = []
        for b in self._bundles:
            if (
                exclude_workspace_root
                and b.name == ROOT_BUNDLE_NAME
                and b.tier == BUNDLE_TIER_WORKSPACE
            ):
                continue
            display_name = "root" if b.name == ROOT_BUNDLE_NAME else b.name
            if not include_root and b.name == ROOT_BUNDLE_NAME:
                continue
            qualified = f"{b.tier}:{display_name}"
            # Always offer the qualified form so users discover the syntax.
            completions.append(
                CommandCompletion(qualified, f"{description_prefix} {b.qualified_ref}")
            )
            # Offer the bare name only when it's unambiguous; with
            # multiple bundles sharing a name across tiers, the bare
            # form would be rejected at parse time anyway.
            if name_counts[b.name] == 1:
                completions.append(
                    CommandCompletion(display_name, f"{description_prefix} {b.qualified_ref}")
                )
        return completions

    # ==================== Prompt Enrichment ====================

    def get_enrichment_priority(self) -> int:
        """Return enrichment priority (lower = earlier).

        References plugin runs first (priority 20) to inject content that
        other plugins (like template) can then process.
        """
        return 20

    def subscribes_to_prompt_enrichment(self) -> bool:
        """Subscribe to prompt enrichment for @reference detection."""
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Detect references in user prompts via @id mentions and tag matching.

        Two detection passes (delegated to _enrich_content):
        1. @reference-id patterns are expanded with full instructions.
        2. Words matching unselected source tags trigger lightweight hints
           so the model knows to call selectReferences.

        Args:
            prompt: The user's prompt text.

        Returns:
            PromptEnrichmentResult with expanded/hinted references.
        """
        return self._enrich_content(prompt, "prompt")

    # ==================== Tool Result Enrichment ====================

    def get_tool_result_enrichment_priority(self) -> int:
        """Return tool result enrichment priority (lower = earlier)."""
        return 20

    def subscribes_to_tool_result_enrichment(self) -> bool:
        """Subscribe to tool result enrichment for reference detection."""
        return True

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str,
        tool_args: Optional[Dict[str, Any]] = None
    ) -> ToolResultEnrichmentResult:
        """Detect references in tool results via @id mentions, tag matching,
        preselected reference file reads, and reference-context annotations.

        Four detection passes:
        1. Preselected reference read detection: checks if tool_args contain
           a file path matching a preselected reference's resolved_path.
           When detected, sets ``pinned_reference`` metadata so the session
           can pin the content for GC protection.
        2. Reference-context annotation: when a markdown file in the **root**
           of a selected reference directory is read and the reference declares
           ``contents`` (templates, validation, policies, scripts), appends
           guidance so the model knows about available resources.
        3. @reference-id patterns are expanded with full instructions
           (delegated to _enrich_content).
        4. Words matching unselected source tags trigger lightweight hints
           so the model knows to call selectReferences (delegated to
           _enrich_content).

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's output as a string.
            tool_args: Optional tool call arguments for detecting which file
                was read (e.g., CLI ``command`` or readFile ``path``).

        Returns:
            ToolResultEnrichmentResult with expanded/hinted references and
            optional ``pinned_reference`` and ``reference_contents`` metadata.
        """
        enrichment = self._enrich_content(result, f"tool:{tool_name}")
        enriched_result = enrichment.prompt
        metadata = dict(enrichment.metadata) if enrichment.metadata else {}

        # Detect preselected reference reads from tool arguments
        if tool_args and self._preselected_paths:
            matched = self._detect_preselected_read(tool_args)
            if matched:
                ref_id, ref_name = matched
                metadata["pinned_reference"] = {
                    "ref_id": ref_id,
                    "ref_name": ref_name,
                }
                self._trace(
                    f"enrich_tool_result: detected preselected reference "
                    f"read: {ref_id} via {tool_name}"
                )

                # Annotate with reference-context if reading a root markdown
                annotation = self._build_contents_annotation(ref_id, tool_args)
                if annotation:
                    enriched_result = enriched_result + "\n\n" + annotation
                    metadata["reference_contents"] = ref_id

        if metadata:
            telemetry = metadata.get("_telemetry", {})
            if "pinned_reference" in metadata:
                telemetry["jaato.enrichment.references.pinned"] = True
            if telemetry:
                metadata["_telemetry"] = telemetry

        return ToolResultEnrichmentResult(
            result=enriched_result,
            metadata=metadata
        )

    def _is_root_markdown_read(
        self, ref_id: str, tool_args: Dict[str, Any]
    ) -> bool:
        """Check if tool_args indicate reading a markdown file in the reference root.

        A "root markdown" is a ``.md`` file directly inside the reference's
        resolved directory (not inside a subfolder like templates/ or validation/).

        Args:
            ref_id: The matched reference source ID.
            tool_args: The tool call arguments dict.

        Returns:
            True if a root-level markdown file is being read.
        """
        source = self.get_source_by_id(ref_id)
        if not source or not source.resolved_path:
            return False

        ref_dir = os.path.normpath(source.resolved_path)

        for value in tool_args.values():
            if not isinstance(value, str):
                continue
            norm_value = os.path.normpath(value)
            # Check the file is directly inside the reference root (not a subfolder)
            parent = os.path.dirname(norm_value)
            if os.path.normpath(parent) != ref_dir:
                continue
            # Check it's a markdown file
            if norm_value.lower().endswith(".md"):
                return True
        return False

    def _build_contents_annotation(
        self, ref_id: str, tool_args: Dict[str, Any]
    ) -> Optional[str]:
        """Build a reference-context annotation for a root markdown read.

        When the model reads a markdown file in the root of a reference
        directory that declares ``contents``, returns an annotation block
        informing the model about available templates, policies, scripts,
        and validation checks.

        Args:
            ref_id: The matched reference source ID.
            tool_args: The tool call arguments dict.

        Returns:
            Annotation string to append to the tool result, or None if not
            applicable (not a root markdown, or no contents declared).
        """
        if not tool_args:
            return None

        if not self._is_root_markdown_read(ref_id, tool_args):
            return None

        source = self.get_source_by_id(ref_id)
        if not source or not source.contents.has_any():
            return None

        ref_dir = source.resolved_path
        contents = source.contents
        sections: List[str] = []

        sections.append(f"---\n**Reference Context: {source.name}**")

        # Templates annotation
        if contents.templates:
            templates_dir = os.path.join(ref_dir, contents.templates)
            template_files = self._list_subfolder_files(
                templates_dir, extensions=(".tpl", ".tmpl")
            )
            if template_files:
                lines = [
                    "**Mandatory Templates** — Use `renderTemplateToFile` with these template IDs:"
                ]
                for tpl in template_files:
                    lines.append(f"  - `{tpl}`")
                sections.append("\n".join(lines))

        # Policies annotation
        if contents.policies:
            policies_dir = os.path.join(ref_dir, contents.policies)
            policy_files = self._list_subfolder_files(
                policies_dir, extensions=(".md",)
            )
            if policy_files:
                lines = [
                    "**Implementation Policies** — You must read and follow these constraints:"
                ]
                for pol in policy_files:
                    lines.append(f"  - `{os.path.join(policies_dir, pol)}`")
                sections.append("\n".join(lines))

        # Scripts annotation
        if contents.scripts:
            scripts_dir = os.path.join(ref_dir, contents.scripts)
            script_files = self._list_subfolder_files(scripts_dir)
            if script_files:
                lines = [
                    "**Helper Scripts** — Available for use during implementation:"
                ]
                for scr in script_files:
                    lines.append(f"  - `{os.path.join(scripts_dir, scr)}`")
                sections.append("\n".join(lines))

        # Validation annotation
        if contents.validation:
            validation_dir = os.path.join(ref_dir, contents.validation)
            validation_files = self._list_subfolder_files(validation_dir)
            if validation_files:
                lines = [
                    "**Post-Implementation Validation** — You MUST run these checks after implementation:"
                ]
                for val in validation_files:
                    lines.append(f"  - `{os.path.join(validation_dir, val)}`")
                sections.append("\n".join(lines))

        if len(sections) <= 1:
            # Only the header, no actual content found
            return None

        sections.append("---")

        self._trace(
            f"_build_contents_annotation: annotated {ref_id} with "
            f"{len(sections) - 2} content sections"
        )
        return "\n\n".join(sections)

    def _list_subfolder_files(
        self,
        directory: str,
        extensions: Optional[tuple] = None,
        max_files: int = 50
    ) -> List[str]:
        """List files in a subfolder, optionally filtering by extension.

        Args:
            directory: Absolute path to the subfolder.
            extensions: Tuple of file extensions to include (e.g., (".tpl", ".tmpl")).
                If None, includes all files.
            max_files: Maximum number of files to return.

        Returns:
            Sorted list of filenames relative to the directory.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []
        files: List[str] = []
        try:
            for item in sorted(dir_path.rglob("*")):
                if not item.is_file():
                    continue
                if extensions and not item.name.lower().endswith(extensions):
                    continue
                rel = str(item.relative_to(dir_path))
                files.append(rel)
                if len(files) >= max_files:
                    break
        except (PermissionError, OSError):
            pass
        return files

    def _detect_preselected_read(
        self, tool_args: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """Check if tool arguments reference a preselected reference file.

        Scans all string values in tool_args for paths matching any
        preselected reference's resolved_path.  Both sides of the comparison
        are normalized via ``normalize_for_comparison`` and ``os.path.normpath``
        so that Windows backslash paths, MSYS2 paths, and Unix paths all match
        correctly.

        Three matching strategies are tried in order:
        1. Exact match after normpath (handles file refs and expanded dir files).
        2. Directory containment via startswith with path separator (handles
           directory refs when the arg is a file inside the directory).
        3. Substring containment (handles CLI commands like
           ``cat /path/to/file.md`` where the path is embedded in a command).

        Args:
            tool_args: The tool call arguments dict (e.g., ``{"command": "cat foo.md"}``
                or ``{"path": "docs/spec.md"}``).

        Returns:
            ``(ref_id, ref_name)`` tuple if a preselected reference path was
            found in the arguments, ``None`` otherwise.
        """
        for value in tool_args.values():
            if not isinstance(value, str):
                continue
            # Normalize the argument value for comparison
            norm_value = normalize_for_comparison(value)
            norm_arg_path = normalize_for_comparison(os.path.normpath(value))
            for norm_path, (ref_id, ref_name) in self._preselected_paths.items():
                # 1. Exact path match (covers files and expanded dir entries)
                if norm_arg_path == norm_path:
                    return (ref_id, ref_name)
                # 2. Directory containment: arg is a file inside the ref dir
                #    Use startswith + "/" to avoid partial name matches
                #    (e.g., "/refs-old/file" should NOT match "/refs")
                if norm_arg_path.startswith(norm_path + "/"):
                    return (ref_id, ref_name)
                # 3. Substring fallback for CLI commands containing the path.
                #    Require a path boundary after the match (/, space, quote,
                #    or end-of-string) to avoid partial-name false positives
                #    like "/refs" matching "/refs-old/file".
                idx = norm_value.find(norm_path)
                if idx >= 0:
                    end_idx = idx + len(norm_path)
                    if end_idx >= len(norm_value) or norm_value[end_idx] in ('/', ' ', '"', "'"):
                        return (ref_id, ref_name)
        return None

    def get_preselected_paths(self) -> Dict[str, Tuple[str, str]]:
        """Return the preselected paths index.

        Returns:
            Mapping from normalized resolved_path to ``(ref_id, ref_name)``
            for all preselected LOCAL references.
        """
        return dict(self._preselected_paths)

    def get_source_by_id(self, ref_id: str) -> Optional[ReferenceSource]:
        """Look up a reference source by its ID.

        Args:
            ref_id: The reference source ID.

        Returns:
            The ``ReferenceSource`` if found, ``None`` otherwise.
        """
        return next((s for s in self._sources if s.id == ref_id), None)

    def file_belongs_to_reference_with_templates(
        self, file_path: str
    ) -> bool:
        """Check if a file path is inside a selected reference that declares templates.

        Used by the template plugin to suppress embedded template extraction
        when the reference already provides authoritative standalone templates.

        Args:
            file_path: Absolute path to the file being inspected.

        Returns:
            True if the file is inside a selected reference directory that
            has ``contents.templates`` set to a non-null value.
        """
        if not self._preselected_paths:
            return False

        norm_file = normalize_for_comparison(os.path.normpath(file_path))

        for norm_path, (ref_id, _ref_name) in self._preselected_paths.items():
            # Check if file is inside this reference directory
            if not (norm_file == norm_path or norm_file.startswith(norm_path + "/")):
                continue
            source = self.get_source_by_id(ref_id)
            if source and source.contents.templates:
                return True
        return False

    def _enrich_content(self, content: str, source_type: str) -> PromptEnrichmentResult:
        """Common enrichment logic for prompts and tool results.

        Detection passes:
        1.  @reference-id patterns — expands with full reference instructions.
        2.  Tag word matching — scans content for words matching tags on
            unselected selectable sources and appends lightweight reference ID
            hints so the model knows to call selectReferences.
        2b. Semantic matching — embeds the content and finds references whose
            embeddings are similar, excluding those already surfaced by
            passes 1 and 2. Only active when lookup_strategy is "hybrid"
            or "semantic_only" and the semantic matcher is available.
        3.  Transitive selection hint — one-time notification after init.

        Args:
            content: The content to enrich.
            source_type: Type of content for logging ("prompt" or "tool:name").

        Returns:
            PromptEnrichmentResult with expanded content.
        """
        if not self._sources:
            return PromptEnrichmentResult(prompt=content)

        enriched_content = content
        all_metadata: Dict[str, Any] = {}

        # --- Pass 1: @reference-id expansion ---
        source_ids = {s.id for s in self._sources}
        at_reference_pattern = re.compile(r'@([\w-]+)')
        matches = at_reference_pattern.findall(content)
        mentioned_ids = [m for m in matches if m in source_ids]

        if mentioned_ids:
            self._trace(f"enrich [{source_type}]: found references: {mentioned_ids}")
            mentioned_sources = [s for s in self._sources if s.id in mentioned_ids]

            # Paths are authorized on every mention so new references gain
            # access even when their instruction block is suppressed below.
            for source in mentioned_sources:
                self._authorize_source_path(source)

            # Dedup: skip injecting instruction blocks for references whose
            # full instructions were already expanded into this session's
            # history.  Without this, every tool result re-inlines the
            # same multi-KB Referenced Sources block.
            new_mentioned_sources = [
                s for s in mentioned_sources
                if s.id not in self._surfaced_mention_ids
            ]
            if new_mentioned_sources:
                instructions = [
                    source.to_instruction() for source in new_mentioned_sources
                ]
                reference_block = (
                    "\n\n---\n**Referenced Sources:**\n\n" +
                    "\n\n".join(instructions) +
                    "\n---"
                )
                enriched_content = enriched_content + reference_block
                all_metadata["mentioned_references"] = [
                    s.id for s in new_mentioned_sources
                ]
                all_metadata["source_type"] = source_type
                self._surfaced_mention_ids.update(
                    s.id for s in new_mentioned_sources
                )

        # --- Pass 2: tag-based reference ID hints ---
        # Only consider unselected selectable sources (not AUTO, not already selected)
        # and only if selectReferences is available
        if "selectReferences" not in self._exclude_tools:
            unselected = [
                s for s in self._sources
                if s.mode == InjectionMode.SELECTABLE
                and s.id not in self._selected_source_ids
                and s.tags
            ]

            if unselected:
                # Build tag → sources mapping
                tag_to_sources: Dict[str, List[ReferenceSource]] = {}
                for source in unselected:
                    for tag in source.tags:
                        tag_to_sources.setdefault(tag, []).append(source)

                # Case-insensitive word boundary match for each tag in content.
                # The boundary character class includes '.' and '/' so that
                # tags do not match inside dotted names (java.util, file.java)
                # or path segments (/usr/lib/java/).
                # Hyphens, spaces, and underscores are treated as
                # interchangeable separators so that "circuit-breaker"
                # matches "circuit breaker", "circuit_breaker", and
                # vice versa.
                # Sentence-coherence matching, shared with the memory
                # plugin (see ``shared.tag_coherence``).  A tag matches
                # if its full string appears verbatim in any sentence
                # OR if all its ≥3-char sub-tokens co-occur in the same
                # sentence — so ``circuit-breaker`` matches "circuit
                # breaker" or "circuit_breaker" in addition to the
                # literal hyphenated form.  In hybrid mode the
                # semantic veto below filters spurious component matches.
                from shared.tag_coherence import (
                    tag_coherent_in_segments,
                    text_segments,
                )
                segments = text_segments(content)
                matched_sources: Dict[str, List[str]] = {}  # source_id → [matched_tags]
                for tag, sources in tag_to_sources.items():
                    if tag_coherent_in_segments(tag, segments):
                        for source in sources:
                            matched_sources.setdefault(source.id, []).append(tag)

                # Exclude sources already handled by @reference-id expansion
                for mid in mentioned_ids:
                    matched_sources.pop(mid, None)

                # --- Semantic veto on tag matches ---
                # In hybrid mode, use embedding similarity to filter out
                # tag matches that are likely false positives (e.g., the
                # word "java" matching a Java reference when the content
                # is about the island of Java).
                vetoed_sources: Dict[str, float] = {}
                if (
                    matched_sources
                    and self._semantic_available()
                    and self._lookup_strategy == "hybrid"
                ):
                    query_vec = self._embedding_provider.embed_text_as_array(content)
                    if query_vec is not None:
                        scores = self._semantic_score_sources(
                            query_vec, set(matched_sources.keys())
                        )
                        for sid, score in scores.items():
                            if score < self._tag_similarity_threshold:
                                vetoed_sources[sid] = score
                        for sid in vetoed_sources:
                            matched_sources.pop(sid)
                        if vetoed_sources:
                            self._trace(
                                f"enrich [{source_type}]: tag matches vetoed by "
                                f"semantic similarity (threshold={self._tag_similarity_threshold}): "
                                f"{{{', '.join(f'{sid}: {score:.3f}' for sid, score in vetoed_sources.items())}}}"
                            )

                # Dedup: drop sources whose tag-match hint was already
                # injected in this session — otherwise every tool call
                # re-appends the same "Reference sources available" block.
                matched_sources = {
                    sid: tags
                    for sid, tags in matched_sources.items()
                    if sid not in self._surfaced_tag_matched_ids
                }

                if matched_sources:
                    self._trace(
                        f"enrich [{source_type}]: tag matches: "
                        f"{{{', '.join(f'{sid}: {tags}' for sid, tags in matched_sources.items())}}}"
                    )

                    # Build lightweight hint block showing which tags
                    # triggered the match for each source.
                    hint_lines = []
                    for source_id, tags in matched_sources.items():
                        source = next(s for s in self._sources if s.id == source_id)
                        hint_lines.append(
                            f"- @{source_id}: {source.name} (matched: {', '.join(tags)})"
                        )

                    hint_block = (
                        "\n\n---\n"
                        "**Reference sources available** — use `selectReferences` with IDs or tags to select:\n\n" +
                        "\n".join(hint_lines) +
                        "\n---"
                    )
                    enriched_content = enriched_content + hint_block
                    all_metadata["tag_matched_references"] = {
                        sid: tags for sid, tags in matched_sources.items()
                    }
                    self._surfaced_tag_matched_ids.update(matched_sources.keys())

        # --- Pass 2b: semantic matching ---
        # When lookup_strategy includes semantic matching ("hybrid" or
        # "semantic_only"), embed the content and find references whose
        # embeddings are similar.  Excludes sources already surfaced by
        # @reference-id expansion or tag matching to avoid duplicates.
        if (
            self._semantic_available()
            and self._lookup_strategy in ("hybrid", "semantic_only")
            and "selectReferences" not in self._exclude_tools
        ):
            # IDs already surfaced by earlier passes — no need to re-hint
            already_surfaced: set = set(mentioned_ids)
            already_surfaced.update(self._selected_source_ids)
            if "tag_matched_references" in all_metadata:
                already_surfaced.update(all_metadata["tag_matched_references"].keys())

            semantic_matches = self._semantic_embed_and_match(
                content=content,
                threshold=self._similarity_threshold,
                top_k=self._max_matches_per_piece,
                exclude_ids=already_surfaced,
            )

            # Only include matches for unselected selectable sources
            selectable_ids = {
                s.id for s in self._sources
                if s.mode == InjectionMode.SELECTABLE
                and s.id not in self._selected_source_ids
            }
            semantic_matches = [
                m for m in semantic_matches if m.source_id in selectable_ids
            ]

            # Dedup: drop semantic matches whose hint was already injected.
            semantic_matches = [
                m for m in semantic_matches
                if m.source_id not in self._surfaced_semantic_ids
            ]

            if semantic_matches:
                self._trace(
                    f"enrich [{source_type}]: semantic matches: "
                    f"{[(m.source_id, f'{m.score:.3f}') for m in semantic_matches]}"
                )

                hint_lines = []
                for match in semantic_matches:
                    source = next(
                        (s for s in self._sources if s.id == match.source_id), None
                    )
                    if source:
                        hint_lines.append(
                            f"- @{match.source_id}: {source.name} "
                            f"(similarity: {match.score:.2f})"
                        )

                if hint_lines:
                    hint_block = (
                        "\n\n---\n"
                        "**Semantically related references** — use "
                        "`selectReferences` with IDs to select:\n\n"
                        + "\n".join(hint_lines)
                        + "\n---"
                    )
                    enriched_content = enriched_content + hint_block
                    all_metadata["semantic_matched_references"] = {
                        m.source_id: round(m.score, 4)
                        for m in semantic_matches
                    }
                    self._surfaced_semantic_ids.update(
                        m.source_id for m in semantic_matches
                    )

        # --- Pass 3: one-time transitive selection hint ---
        # On the first prompt enrichment after initialization, notify the model
        # and user about references that were transitively selected from
        # pre-selected sources.  Only fires for prompts (not tool results)
        # because appending context to arbitrary tool output is confusing.
        if (self._transitive_notification_pending
                and source_type == "prompt"
                and self._transitive_parent_map):
            self._transitive_notification_pending = False

            hint_lines = []
            for tid, parents in self._transitive_parent_map.items():
                parent_refs = ", ".join(f"@{p}" for p in sorted(parents))
                source = next((s for s in self._sources if s.id == tid), None)
                name = source.name if source else tid
                hint_lines.append(f"- @{tid}: {name} (from {parent_refs})")

            hint_block = (
                "\n\n---\n"
                "**Transitively selected references** — auto-included from pre-selected sources:\n\n" +
                "\n".join(hint_lines) +
                "\n---"
            )
            enriched_content = enriched_content + hint_block
            all_metadata["transitive_references"] = {
                tid: sorted(parents)
                for tid, parents in self._transitive_parent_map.items()
            }

            self._trace(
                f"enrich [{source_type}]: transitive hint emitted for "
                f"{list(self._transitive_parent_map.keys())}"
            )

        if all_metadata:
            # Count total references expanded across all passes
            expanded_count = (
                len(all_metadata.get("mentioned_references", []))
                + len(all_metadata.get("tag_matched_references", {}))
                + len(all_metadata.get("semantic_matched_references", {}))
                + len(all_metadata.get("transitive_references", {}))
            )
            all_metadata["_telemetry"] = {
                "jaato.enrichment.references.expanded_count": expanded_count,
            }
            return PromptEnrichmentResult(
                prompt=enriched_content,
                metadata=all_metadata
            )

        return PromptEnrichmentResult(prompt=content)

    # Public API for programmatic access

    def get_sources(self) -> List[ReferenceSource]:
        """Get all configured reference sources."""
        return self._sources.copy()

    def get_selected_ids(self) -> List[str]:
        """Get IDs of selected sources (includes pre-selected and user-selected)."""
        return self._selected_source_ids.copy()

    def reset_selections(self) -> None:
        """Clear all session selections."""
        self._selected_source_ids.clear()

    def on_history_cleared(self) -> None:
        """Reset per-session enrichment tracking when history is wiped.

        Called by ``JaatoSession.reset_session()`` on a true history clear.
        Clears the surfaced-ID sets so previously hinted references can
        surface again in the fresh conversation — without this, the model
        would never see reference hints after a ``reset`` command.
        """
        self._surfaced_mention_ids.clear()
        self._surfaced_tag_matched_ids.clear()
        self._surfaced_semantic_ids.clear()
        self._trace("on_history_cleared: cleared surfaced reference tracking")

    # Interactivity protocol methods

    def supports_interactivity(self) -> bool:
        """References plugin requires user interaction for source selection.

        Returns:
            True - references plugin has interactive selection prompts.
        """
        return True

    def get_supported_channels(self) -> List[str]:
        """Return list of channel types supported by references plugin.

        Returns:
            List of supported channel types: console, webhook, file, queue.
        """
        return ["console", "webhook", "file", "queue"]

    def set_selection_hooks(
        self,
        on_requested: Optional[Callable[[str, List[str]], None]] = None,
        on_resolved: Optional[Callable[[str, List[str]], None]] = None
    ) -> None:
        """Set hooks for selection lifecycle events.

        These hooks enable UI integration by notifying when selection
        requests start and complete.

        Args:
            on_requested: Called when selection session starts.
                Signature: (tool_name, prompt_lines) -> None
            on_resolved: Called when selection is resolved.
                Signature: (tool_name, selected_ids) -> None
        """
        self._on_selection_requested = on_requested
        self._on_selection_resolved = on_resolved

    def set_channel(
        self,
        channel_type: str,
        channel_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the interaction channel for reference selection.

        Args:
            channel_type: One of: console, webhook, file
            channel_config: Optional channel-specific configuration

        Raises:
            ValueError: If channel_type is not supported
        """
        if channel_type not in self.get_supported_channels():
            raise ValueError(
                f"Channel type '{channel_type}' not supported. "
                f"Supported: {self.get_supported_channels()}"
            )

        # Create the channel with config
        from .channels import create_channel
        self._channel = create_channel(channel_type, channel_config)


def create_plugin() -> ReferencesPlugin:
    """Factory function to create the references plugin instance."""
    return ReferencesPlugin()
