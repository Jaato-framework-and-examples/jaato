"""Memory plugin for model self-curated persistent memory across sessions.

Supports the knowledge-curation lifecycle ("The School") where agents store
raw memories during sessions, and an advisor agent later curates them into
validated knowledge or promotes them to reference entries.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from jaato_sdk.plugins.base import (
    CommandCompletion,
    HelpLines,
    PromptEnrichmentResult,
    ToolResultEnrichmentResult,
    UserCommand,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema
from .indexer import MemoryIndexer
from .models import (
    ACTIVE_MATURITIES,
    MATURITY_DISMISSED,
    MATURITY_ESCALATED,
    MATURITY_RAW,
    MATURITY_VALIDATED,
    SCOPE_PROJECT,
    SCOPE_UNIVERSAL,
    VALID_MATURITIES,
    VALID_SCOPES,
    Memory,
)
from .storage import MemoryStorage
from shared.plugins.runner_forwarding import RunnerForwardingMixin
from shared.trace import trace as _trace_write


class MemoryPlugin(RunnerForwardingMixin):
    """Plugin for model self-curated persistent memory across sessions.

    This plugin allows the model to:
    1. Store valuable explanations/insights for future reference
    2. Retrieve stored memories when relevant
    3. Build a persistent knowledge base over time

    The plugin participates in the knowledge-curation lifecycle:
    - Working agents store memories with ``maturity="raw"``
    - Prompt enrichment surfaces **curated memories only** — the index is
      built from ``curated.jsonl`` (see ``initialize``); raw memories are the
      curator's queue and do NOT auto-surface as enrichment hints. (The
      ``retrieve_memories`` tool can still fetch raw via its ``maturity``
      filter — that is the tool surface, not enrichment.)  So a memory must be
      curated (raw→validated) before it appears in the 💡 hint — which makes
      the advisor REQUIRED for cross-session continuity, not optional.  See
      ``docs/design/agent-continuity.md``.
    - The advisor agent uses ``get_pending_curation`` (via storage) to
      review raw memories and transition them to validated/escalated/dismissed

    The plugin uses a two-phase retrieval system:
    - Phase 1: Prompt enrichment adds lightweight hints about CURATED memories
    - Phase 2: Model decides whether to retrieve full content via function calling
      (``retrieve_memories`` can also reach raw memories explicitly)
    """

    def __init__(self):
        """Initialize the memory plugin.

        Storage is created during initialize() with a relative path template.
        When set_workspace_path() is called (by PluginRegistry broadcast),
        storage is re-created under the correct workspace directory.
        """
        self._name = "memory"
        self._storage: Optional[MemoryStorage] = None
        self._indexer: Optional[MemoryIndexer] = None
        self._global_storage: Optional[MemoryStorage] = None
        self._global_indexer: Optional[MemoryIndexer] = None
        # Deployment write-side gate: which memory scopes the model may store.
        # Default permissive (all valid scopes); restrict per-deployment via the
        # ``allowed_scopes`` config (e.g. ``["project"]`` keeps a deployment off
        # the HOME/global tier entirely). Resolved from config in initialize().
        self._allowed_scopes: frozenset = VALID_SCOPES
        self._agent_name: Optional[str] = None
        self._session_id: Optional[str] = None
        # Server 0.6.168+: stashed by ``set_plugin_registry`` so
        # ``_get_session_id`` can read the always-fresh
        # ``registry._session_id`` for cascade-pool-reused slots.
        self._plugin_registry: Optional[Any] = None
        self._storage_path_template: str = ".jaato/memories.jsonl"
        # Memory IDs whose hint bullet has already been injected into the
        # model's context during this session.  Prevents the same "💡
        # Available Memories" block from being re-surfaced on every tool
        # call when the same memory keeps matching.  Cleared by
        # on_history_cleared() when the session history is wiped.
        self._surfaced_memory_ids: Set[str] = set()

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("MEMORY", msg)

    @staticmethod
    def _resolve_allowed_scopes(raw: Optional[List[str]]) -> frozenset:
        """Resolve the ``allowed_scopes`` write-side gate from config.

        Default (key absent / ``None``) is permissive — all ``VALID_SCOPES``.
        When set, it is the deployment policy for which scopes ``store_memory``
        may write (e.g. ``["project"]`` keeps a deployment off the HOME/global
        tier entirely). The parse is deterministic and LOUD, never a silent
        fallback: unknown scope strings are dropped with a WARNING, and an empty
        resolved set (all entries invalid, or an explicit ``[]``) is honored —
        it rejects every write — but logged at WARNING so a typo isn't mistaken
        for "allow all".
        """
        if raw is None:
            return VALID_SCOPES
        requested = [str(s).strip().lower() for s in raw]
        unknown = [s for s in requested if s not in VALID_SCOPES]
        if unknown:
            logging.getLogger(__name__).warning(
                "memory: allowed_scopes contains unknown scope(s) %s "
                "(valid: %s) — ignoring them", unknown, sorted(VALID_SCOPES))
        resolved = frozenset(s for s in requested if s in VALID_SCOPES)
        if not resolved:
            logging.getLogger(__name__).warning(
                "memory: allowed_scopes=%r resolved to EMPTY — every "
                "store_memory write will be rejected", raw)
        return resolved

    def _get_session_id(self) -> Optional[str]:
        """Return the current session's daemon ID, per-execution.

        Reads the CURRENTLY EXECUTING session via
        ``shared.session_context.get_current_session()`` and resolves its
        ``_daemon_session_id`` (with parent-walk for any session that
        lacks its own — the canonical resolver in ``dynamic_instructions``).
        Each sibling subagent has its own ``JaatoSession`` (stamped with
        its ``envelope.session_id`` at runner bootstrap), so this value is
        per-sibling-correct.

        Deliberately does NOT read ``registry._session_id``: the plugin
        registry is SHARED across sibling subagents, so its single
        ``_session_id`` is overwritten by whichever sibling bootstrapped
        last — reading it leaked one sibling's id into another's
        ``source_session``.  Fixed by stamping the id per-session
        (``runner/session.py`` ``bootstrap_session``) and reading it here.

        Falls back to ``self._session_id`` (the config-injection value
        from ``initialize``) only when there is no session in context —
        standalone unit tests that construct the plugin without going
        through a real turn.
        """
        try:
            from shared.session_context import get_current_session
            session = get_current_session()
        except LookupError:
            session = None
        if session is not None:
            from shared.dynamic_instructions import _resolve_session_id
            sid = _resolve_session_id(session)
            if sid:
                return sid
        return self._session_id

    def set_plugin_registry(self, registry: Any) -> None:
        """Auto-wiring hook called by the registry at
        ``expose_tool()`` time (registry.py:957-958).  Stashes a
        reference so ``_get_session_id`` can read the always-current
        ``registry._session_id`` at store_memory call time (handles
        cascade-pool-reuse HIT slots — see ``_get_session_id``
        docstring).
        """
        self._plugin_registry = registry

    def set_session(self, session: Any) -> None:
        """Auto-wiring hook called by the framework after plugin
        configure() — intentionally a NO-OP.

        **Must not store session state on ``self``.**  Plugin instances
        are SHARED across sibling subagents within a session (shared
        runtime registry), so a sibling's ``set_session`` would clobber
        another's stashed value → cross-subagent leakage.  Enforced by
        ``tests/test_plugin_session_safety.py``.

        Historically (PR-196, server 0.6.167+) this stashed
        ``session._daemon_session_id`` on ``self._session_id`` to
        populate the ``source_session`` field.  But ``_daemon_session_id``
        is set ONLY daemon-side (``JaatoClient.set_daemon_session_id``),
        never on the runner-side ``JaatoSession`` — and memory is
        ``PLUGIN_TIER = "runner"`` — so this path read ``None`` for every
        cascade session (peer 7:1 retry-49 empirical: 4/4 memories still
        ``source_session=null`` post-PR-196).  The session_id that
        actually reaches ``store_memory`` comes from the config-injection
        path (``initialize`` reads ``config['session_id']`` injected by
        ``registry._augment_plugin_config``) preferred via
        ``_get_session_id``'s registry read — see those docstrings.

        Args:
            session: The JaatoSession instance (unused).
        """
        # Intentionally empty — see docstring. session_id is resolved at
        # store time via _get_session_id (registry / config injection).

    def set_session_context(self, session_id: str) -> None:
        """Legacy compat shim — pre-0.6.167 callers may still use
        this method.  No framework code reaches it; safe to remove
        in a future release once any external callers (premium
        extensions, kb-side scripts) confirm they don't depend on it.
        """
        self._session_id = session_id

    @property
    def name(self) -> str:
        """Return plugin name."""
        return self._name

    @classmethod
    def get_apparmor_rules(
        cls,
        *,
        workspace_path: str,
        session_id: str,
        config_root: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> List[str]:
        """Contribute memory-plugin host paths to the AppArmor profile.

        Phase 2 of the plugin-apparmor-contribution refactor
        (template v23, 2026-05-16).  These paths used to be hardcoded
        in ``apparmor.py:PROFILE_TEMPLATE``; sessions without the
        memory plugin in ``profile.plugins`` no longer carry the
        grants (least-privilege).

        Memory storage uses three layouts at ``~/.jaato/memories``:
        - ``memories/raw/{id}.json`` — pending queue (one file per memory)
        - ``memories/curated.jsonl`` — curated knowledge base
        - ``memories.jsonl`` — legacy single-file store (retained for
          migration; readable but not writable on new sessions)

        Both the folder and its contents need ``rw`` so the plugin can
        create the parent directory on first write, enumerate raw/, and
        perform atomic tempfile+rename writes.
        """
        return [
            "@{HOME}/.jaato/memories/       rw,",
            "@{HOME}/.jaato/memories/**     rw,",
            "@{HOME}/.jaato/memories.jsonl  rw,",
        ]

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize storage backend and indexer.

        Args:
            config: Optional configuration dict with keys:
                - storage_path: Path to JSONL file (default: .jaato/memories.jsonl)
                - enrichment_limit: Max hints to show in prompt (default: 5)
        """
        config = config or {}
        self._agent_name = config.get("agent_name")
        self._allowed_scopes = self._resolve_allowed_scopes(config.get("allowed_scopes"))
        self._trace(f"initialize: allowed_scopes={sorted(self._allowed_scopes)}")
        # Server 0.6.168+ (real Bug B-class fix): read session_id
        # from config.  The registry's _augment_plugin_config
        # (registry.py:1337-1386) injects session_id via setdefault
        # at expose_tool() time after runner-side
        # ``registry.set_session_id(envelope.session_id)`` fires
        # (runner/session.py:271).  Mirrors how ``agent_name`` is
        # wired above — same framework-injection mechanism.
        #
        # PR-196 added ``set_session(session)`` reading
        # ``session._daemon_session_id`` as the wiring path.  That
        # attribute is set ONLY by JaatoClient (daemon-side wrapper)
        # via ``set_daemon_session_id`` — never on runner-side
        # JaatoSession.  Plugin auto-wiring runs on the RUNNER side
        # (memory is PLUGIN_TIER = "runner"), so PR-196 read None
        # for every cascade session.  Empirical: peer 7:1 retry-49
        # post-PR-196 still showed source_session=null on 4/4
        # memories.  This path (config injection) IS reached for
        # runner-side plugin init.
        self._session_id = config.get("session_id")
        self._storage_path_template = config.get("storage_path", ".jaato/memories.jsonl")

        self._storage = MemoryStorage(self._storage_path_template)
        self._indexer = MemoryIndexer()

        # Build index from CURATED memories only — raw memories are the
        # curator's queue and aren't surfaced as enrichment hints.
        #
        # The template is RELATIVE: at global registry-init time it resolves
        # against the daemon cwd, NOT the session workspace.  The real
        # per-session store is wired later by set_workspace_path().  So a
        # confined session is (correctly) denied this path here — tolerate it
        # and let set_workspace_path() resolve the workspace-tier store.  See
        # _safe_load_curated for why this must not disable the plugin.
        existing_memories = self._safe_load_curated(
            self._storage, tier="workspace (pre-set_workspace_path)")
        self._indexer.build_index(existing_memories)
        self._trace(f"initialize: storage_path={self._storage_path_template}, curated_memories={len(existing_memories)}")

        # Global storage at ~/.jaato/memories.jsonl — cross-session knowledge
        # shared by UNCONFINED agents.  This tier is OPTIONAL: a confined
        # session is correctly denied HOME, so the tier is simply absent for it
        # and the workspace tier is the only (priority) store.  Configurable via
        # "global_storage_path" for testing.
        global_path = config.get(
            "global_storage_path",
            str(Path.home() / ".jaato" / "memories.jsonl"),
        )
        self._global_storage = MemoryStorage(global_path)
        self._global_indexer = MemoryIndexer()
        global_memories = self._safe_load_curated(
            self._global_storage, tier="global (HOME)")
        self._global_indexer.build_index(global_memories)
        self._trace(f"initialize: global_path={global_path}, global_curated_memories={len(global_memories)}")

    def _safe_load_curated(self, storage: "MemoryStorage", tier: str) -> List["Memory"]:
        """Load a tier's curated memories, tolerating an inaccessible store.

        Memory has two tiers: a per-session WORKSPACE store (the priority,
        wired by set_workspace_path) and an OPTIONAL global HOME store.  At
        global registry init neither is guaranteed reachable — a confined
        session is *correctly* denied both the daemon-cwd-relative template and
        HOME.  An ``OSError`` loading a tier therefore means "this tier is
        absent here", not "the plugin is broken": degrade to an empty index so
        the plugin stays EXPOSED and set_workspace_path() can wire the real
        workspace store.  A non-OSError (a genuine bug) still propagates.
        """
        try:
            return storage.load_curated()
        except OSError as e:
            self._trace(
                f"initialize: {tier} memory tier not loadable here ({e}); "
                f"degrading to empty — the workspace tier is wired by "
                f"set_workspace_path()")
            return []

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        self._trace("shutdown")
        if self._indexer:
            self._indexer.clear()
        self._storage = None

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset (Phase 1, server 0.6.142+) — NO-OP.

        **Daniel-corrected (2026-05-20)**: this plugin was initially
        categorised as needing reset between cascade sessions; that
        was wrong.  Per Daniel's litmus test:

            "A plugin's state should SURVIVE this call if a subsequent
            session within the SAME cascade might benefit from it."

        Memories written by the model in session A of a cascade are
        EXACTLY the kind of context session B should be able to read.
        Wiping them between cascade stages would silently discard
        the persistence layer the model was authoring against — a
        textbook framework-side defeat of the model's intent.

        Survives the reset (the entire plugin state):
        - ``_storage``: per-workspace memory file pointers.
        - ``_indexer``: built memory index (search/recall structures).
        - ``_global_storage`` / ``_global_indexer``: cross-workspace
          memories (constant within any session).
        - Workspace + global path resolution + config — all
          constant within a cascade.

        ``shutdown()`` (final teardown at cascade end) still clears
        the index + drops the storage handle.
        """
        self._trace(
            "reset_for_next_session: NO-OP — memories are cross-session "
            "by-design (Daniel litmus test, 2026-05-20)"
        )

    def get_config_schema(self) -> Dict[str, Any]:
        """Return JSON Schema for this plugin's configuration."""
        return {
            "type": "object",
            "properties": {
                "storage_path": {
                    "type": "string",
                    "default": ".jaato/memories.jsonl",
                    "description": "Path to JSONL memory storage file",
                },
                "allowed_scopes": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(VALID_SCOPES)},
                    "default": sorted(VALID_SCOPES),
                    "description": (
                        "Write-side gate: which memory scopes the model may "
                        "store. Default permissive (all). Set e.g. [\"project\"] "
                        "to keep a deployment off the HOME/global tier entirely; "
                        "a disallowed scope is hard-rejected back to the model."
                    ),
                },
            },
        }

    def set_workspace_path(self, path: str) -> None:
        """Re-initialize storage under the correct workspace directory.

        Called by PluginRegistry.set_workspace_path() broadcast after
        plugin initialization. Resolves the relative storage path template
        against the workspace root so that each client's memories are
        isolated to its own workspace.
        """
        resolved = str(Path(path) / self._storage_path_template)
        self._trace(f"set_workspace_path: {path} -> {resolved}")
        self._storage = MemoryStorage(resolved)
        self._indexer = MemoryIndexer()
        existing = self._storage.load_curated()
        self._indexer.build_index(existing)

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool declarations for memory operations.

        Returns:
            List of ToolSchema objects for store_memory, retrieve_memories, list_memory_tags
        """
        return [
            ToolSchema(
                name='store_memory',
                description=(
                    'Store information from this conversation for retrieval in future sessions. '
                    'Use this when you provide a comprehensive explanation, architecture overview, '
                    'or useful insight that would help in future conversations about this topic. '
                    'Only store substantial, reusable information - not ephemeral responses. '
                    'Memories are created as "raw" and will later be reviewed by the advisor '
                    'agent for potential promotion to permanent knowledge.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": (
                                "The information to store (explanation, code pattern, "
                                "architecture notes, etc.). Be comprehensive but concise."
                            )
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Brief summary of what this memory contains "
                                "(1-2 sentences max)"
                            )
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 2},
                            "description": (
                                "Specific keywords for retrieval (minimum 2 characters each). "
                                "Tags must be distinctive enough to identify THIS memory "
                                "without matching unrelated ones. "
                                "Good: 'oauth_pkce_flow', 'postgresql_indexing', 'react_hooks'. "
                                "Bad: generic words like 'code', 'error', 'fix', 'config', "
                                "or single letters."
                            )
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "Your confidence in the accuracy of this memory (0.0-1.0). "
                                "Use 0.8-1.0 for well-tested facts, 0.5-0.7 for reasonable "
                                "beliefs, 0.1-0.4 for uncertain observations. Default: 0.5"
                            )
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["project", "universal"],
                            "description": (
                                "How broadly this memory applies. 'project' for codebase-specific "
                                "knowledge, 'universal' for generally applicable insights. "
                                "Default: 'project'"
                            )
                        },
                        "evidence": {
                            "type": "string",
                            "description": (
                                "What triggered this learning — error messages, tool results, "
                                "observations, or other evidence that substantiates this memory. "
                                "Helps the advisor agent assess validity during curation."
                            )
                        }
                    },
                    "required": ["content", "description", "tags"]
                },
                category="memory",
                discoverability="core",
            ),
            ToolSchema(
                name='retrieve_memories',
                description=(
                    'Retrieve previously stored memories. '
                    'When the prompt shows "💡 Available Memories" hints, prefer '
                    'a SINGLE call passing the listed memory IDs in `ids` — '
                    'one call covers all suggested memories, no need to '
                    'reconstruct tag queries per bullet. '
                    'Use `tags` only when exploring or when no IDs are known. '
                    'By default searches both workspace-local and global '
                    '(cross-session) memories.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Memory IDs to fetch directly (e.g. from "
                                "the IDs shown in 'Available Memories' hints). "
                                "Bypasses tag matching — fetches exactly these "
                                "memories regardless of maturity or scope."
                            )
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Tags to search for. Only used when `ids` is "
                                "not provided. Either `ids` or `tags` is required."
                            )
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max number of memories to retrieve (default: 3, ignored when `ids` is used)"
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["project", "universal"],
                            "description": (
                                "Filter by scope: 'project' (workspace-local only), "
                                "'universal' (global cross-session only). "
                                "If omitted, searches both. Ignored when `ids` is used."
                            )
                        },
                        "maturity": {
                            "type": "string",
                            "enum": ["raw", "validated", "escalated", "dismissed"],
                            "description": (
                                "Filter by maturity state. If omitted, returns "
                                "active memories only (raw + validated). "
                                "Ignored when `ids` is used."
                            )
                        }
                    },
                    "required": []
                },
                category="memory",
                discoverability="core",
            ),
            ToolSchema(
                name='list_memory_tags',
                description=(
                    'List all available memory tags to discover what has been stored. '
                    'Useful for exploring the knowledge base or finding related topics.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                category="memory",
                discoverability="core",
            ),
            ToolSchema(
                name='update_memory',
                description=(
                    'Update fields on an existing memory. '
                    'Used by the advisor agent to curate memories: '
                    'promote (maturity="validated"), dismiss (maturity="dismissed"), '
                    'or adjust confidence/tags/content.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "ID of the memory to update"
                        },
                        "maturity": {
                            "type": "string",
                            "enum": ["raw", "validated", "escalated", "dismissed"],
                            "description": "New maturity state"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Updated confidence score"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Replacement tags (overwrites existing)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Replacement content (for merge operations)"
                        }
                    },
                    "required": ["id"]
                },
                category="memory",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='delete_memory',
                description=(
                    'Permanently delete a memory by ID. '
                    'Used for cleanup after merging duplicate memories.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "ID of the memory to delete"
                        }
                    },
                    "required": ["id"]
                },
                category="memory",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return tool executors.

        Phase 3 §3.9: forwards via runner-RPC when a runner is
        attached.  ``~/.jaato/memories/`` is rw under every
        session's profile (template line 334), so the runner
        writes ``memories/raw/<id>.json`` and ``curated.jsonl``
        directly via tempfile-rename — same concurrency story as
        today.  Embedding-cache sharing is a criterion-2 daemon
        placement deferred per parent §4.2; revisit if cross-runner
        RAM cost bites.

        Returns:
            Dict mapping tool names to executor functions
        """
        return self.wrap_executors_for_runner_forwarding({
            "store_memory": self._execute_store,
            "retrieve_memories": self._execute_retrieve,
            "list_memory_tags": self._execute_list_tags,
            "update_memory": self._execute_update,
            "delete_memory": self._execute_delete,
            # User command
            "memory": self.execute_memory,
        })

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing memory capabilities.

        Includes guidance on the knowledge-curation lifecycle so that
        agents understand their memories will be reviewed and potentially
        promoted to permanent knowledge.

        Returns:
            Instructions for the model about memory usage
        """
        return (
            "# Persistent Memory\n\n"
            "You have access to a persistent memory system with two tiers:\n"
            "- **Project memories** (`scope=\"project\"`, default) — stored in the "
            "workspace, available within this session and future sessions in the "
            "same workspace.\n"
            "- **Universal memories** (`scope=\"universal\"`) — stored globally at "
            "`~/.jaato/memories.jsonl`, shared across all sessions and workspaces. "
            "Use this for knowledge that benefits any future session or agent.\n\n"
            "## Two use cases\n\n"
            "**Context snapshots** (keeping your context clean):\n"
            "When your context is getting large and you need to preserve data for "
            "later retrieval within this session, store it as a project-scoped "
            "memory. This offloads data from your active context while keeping it "
            "accessible via `retrieve_memories`. Good for: large tool outputs, "
            "intermediate analysis results, file inventories.\n\n"
            "**Cross-session knowledge** (persistent learning):\n"
            "When you discover something genuinely useful for future sessions or "
            "other agents — a non-obvious pattern, a gotcha, a successful approach "
            "— store it with `scope=\"universal\"`. Tag it with your agent name "
            "(e.g. `\"agent:gen-references\"`) and set `confidence` honestly.\n\n"
            "## How to use\n\n"
            "- `store_memory` — save a new memory (project or universal scope)\n"
            "- `retrieve_memories` — search by tags, optionally filtering by "
            "`scope` and `maturity`\n"
            "- `list_memory_tags` — discover what topics have been stored\n"
            "- `update_memory` — update maturity, confidence, tags, or content "
            "on an existing memory (used by the advisor agent for curation)\n"
            "- `delete_memory` — permanently delete a memory (for merge cleanup)\n\n"
            "When you see 💡 **Available Memories** hints — whether in "
            "the user's prompt or appended to a tool result — make a "
            "**single** `retrieve_memories(ids=[...])` call passing the "
            "listed memory IDs.  The hint already shows the exact "
            "command to run.  Do not make one call per bullet with "
            "reconstructed tag queries — that's wasteful and surfaces "
            "the same memories multiple times.\n\n"
            "## Knowledge curation lifecycle\n\n"
            "Your memories are part of a learning pipeline:\n"
            "1. Created as **raw** — awaiting review by the advisor agent\n"
            "2. Advisor may **validate** it (confirmed valuable, kept as memory)\n"
            "3. Advisor may **escalate** it to a permanent reference\n"
            "4. Advisor may **dismiss** it (incorrect, trivial, or superseded)\n\n"
            "To help the advisor assess effectively:\n"
            "- Set `confidence` honestly (0.0–1.0)\n"
            "- Set `scope` — project-specific or universally applicable?\n"
            "- Provide `evidence` — what triggered this learning?\n\n"
            "## Best practices\n\n"
            "- Only store substantial, reusable information (not ephemeral responses)\n"
            "- Use **specific, distinctive** tags: 'oauth_pkce_flow', "
            "'postgresql_indexing', 'celery_retry_policy'. "
            "Avoid generic tags like 'code', 'error', 'fix' that match too broadly\n"
            "- Write clear descriptions for future retrieval\n"
            "- Include evidence: error messages, command outputs, or observations\n"
            "- For universal memories: tag with `\"agent:<your-name>\"` for provenance\n"
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return list of auto-approved tools.

        All memory tools are safe - read-only or self-directed writes.
        The 'memory' user command is also auto-approved since it's
        invoked directly by the user.

        Returns:
            List of tool names that don't require permission
        """
        return ["store_memory", "retrieve_memories", "list_memory_tags", "memory"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for memory management.

        Returns:
            List of UserCommand objects for the memory command
        """
        return [
            UserCommand(
                name="memory",
                description="Manage persistent memories: list, remove <id>, edit <id>",
                share_with_model=False,
            )
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for memory command arguments.

        Provides autocompletion for:
        - Subcommands: list, remove, edit, help
        - Memory IDs for remove/edit subcommands
        """
        if command != "memory":
            return []

        # Subcommand completions
        subcommands = [
            CommandCompletion("list", "List all stored memories"),
            CommandCompletion("remove", "Remove a memory by ID"),
            CommandCompletion("edit", "Edit a memory in external editor"),
            CommandCompletion("help", "Show detailed help"),
        ]

        if not args:
            return subcommands

        if len(args) == 1:
            # Partial subcommand - filter matching ones
            partial = args[0].lower()
            return [c for c in subcommands if c.value.startswith(partial)]

        if len(args) == 2:
            subcommand = args[0].lower()
            partial = args[1].lower()

            if subcommand in ("remove", "edit"):
                # Provide memory ID completions
                return self._get_memory_id_completions(partial)

        return []

    def get_memory_metadata(self) -> List[Dict[str, Any]]:
        """Return lightweight memory metadata for completion caches.

        Returns:
            List of dicts with id, description, tags, and lifecycle fields
            for each memory.
        """
        if not self._storage:
            return []
        return [
            {
                "id": m.id,
                "description": m.description,
                "tags": m.tags,
                "maturity": m.maturity,
                "confidence": m.confidence,
                "scope": m.scope,
            }
            for m in self._storage.load_all()
        ]

    def _get_memory_id_completions(self, partial: str) -> List[CommandCompletion]:
        """Get memory ID completions matching partial input."""
        if not self._storage:
            return []

        completions = []
        for mem in self._storage.load_all():
            if mem.id.lower().startswith(partial):
                # Truncate description for display
                desc = mem.description[:40] + "..." if len(mem.description) > 40 else mem.description
                completions.append(CommandCompletion(mem.id, desc))

        return completions

    # ===== Prompt Enrichment Protocol =====

    def get_enrichment_priority(self) -> int:
        """Return enrichment priority (lower = earlier).

        Memory runs at priority 80 - late in the pipeline so it can
        analyze the fully enriched prompt for memory matching.
        """
        return 80

    def subscribes_to_prompt_enrichment(self) -> bool:
        """Subscribe to enrich prompts with memory hints.

        Returns:
            True to receive prompts before they're sent to model
        """
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Analyze prompt and inject hints about available memories.

        This is the key method that:
        1. Extracts keywords/concepts from the user prompt
        2. Queries the index for matching memories
        3. Injects lightweight hints (NOT full content)

        Args:
            prompt: User's original prompt text

        Returns:
            PromptEnrichmentResult with enriched prompt and metadata
        """
        enriched_text, metadata = self._enrich_text(prompt)
        return PromptEnrichmentResult(prompt=enriched_text, metadata=metadata)

    # ==================== Tool Result Enrichment ====================

    def get_tool_result_enrichment_priority(self) -> int:
        """Return tool result enrichment priority (lower = earlier)."""
        return 80

    def subscribes_to_tool_result_enrichment(self) -> bool:
        """Subscribe so memory hints are injected into tool results too.

        Without this, memories only surface at the start of a turn based
        on the user's message.  Tool outputs that mention topics with
        associated memories would miss them.  Mirrors the behaviour of
        the references plugin which enriches both prompts and tool
        results.
        """
        return True

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> ToolResultEnrichmentResult:
        """Inject memory hints into a tool result before the model sees it.

        Uses the same keyword-extraction and tag-index path as
        ``enrich_prompt``.  Called in the function-calling loop right
        after a tool returns, so matching memories influence the
        model's next reasoning step within the same turn.

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's output as a string.
            tool_args: Tool call arguments (unused here; kept for
                protocol compatibility with other enrichers).

        Returns:
            ToolResultEnrichmentResult with hints appended.
        """
        enriched, metadata = self._enrich_text(result)
        return ToolResultEnrichmentResult(result=enriched, metadata=metadata)

    # ==================== Shared enrichment core ====================

    def _enrich_text(self, text: str) -> tuple:
        """Shared core for prompt and tool-result enrichment.

        Runs the keyword → index → hint pipeline on arbitrary text and
        returns ``(enriched_text, metadata)``.  Both ``enrich_prompt``
        and ``enrich_tool_result`` wrap this with their respective
        result types so the model sees the same "💡 Available Memories"
        hint block regardless of which surface triggered the match.

        Args:
            text: The text to analyse (user prompt or tool output).

        Returns:
            Tuple of ``(enriched_text, metadata_dict)``.  When no
            memories match, ``enriched_text`` equals ``text`` and
            metadata carries ``memory_matches: 0``.
        """
        if not self._indexer or not self._storage:
            return text, {"error": "Plugin not initialized"}

        # Find matching memories from BOTH workspace and global stores
        # using paragraph-coherence matching (compound tags must have
        # their components co-occur in some paragraph of `text`).
        matches = self._indexer.find_matches_in_text(text, limit=5)
        if self._global_indexer:
            global_matches = self._global_indexer.find_matches_in_text(text, limit=3)
            # Deduplicate by ID and merge (workspace takes priority)
            seen_ids = {m.id for m in matches}
            for gm in global_matches:
                if gm.id not in seen_ids:
                    matches.append(gm)

        if not matches:
            return text, {"memory_matches": 0}

        # Dedup: drop matches whose hint bullet was already injected into
        # this session's history.  Keeps per-turn enrichment informative
        # (new matches still surface) without re-spamming the same block
        # on every tool call.  When every match has already been surfaced,
        # return the text unchanged so no "added context" notification
        # fires either.
        new_matches = [m for m in matches if m.id not in self._surfaced_memory_ids]
        if not new_matches:
            return text, {
                "memory_matches": 0,
                "suppressed_duplicates": [m.id for m in matches],
            }
        matches = new_matches

        # Build hint section.  Each bullet shows the memory ID and a
        # short description; the closing line tells the agent how to
        # fetch ALL listed memories in a single call (using the `ids`
        # parameter).  This avoids the historical pattern of one
        # retrieve_memories call per bullet with overlapping tag sets.
        ids_list = [m.id for m in matches]
        hint_lines = [
            "",
            "💡 **Available Memories** — fetch them in ONE call:",
            f"  retrieve_memories(ids={ids_list!r})",
            "",
            "  Listed below for reference:",
        ]
        for memory_meta in matches:
            hint_lines.append(
                f"  - {memory_meta.id}: {memory_meta.description}"
            )

        enriched_text = text + "\n" + "\n".join(hint_lines)

        # Collect ONLY the tags that actually triggered the match
        # (i.e. were topically present in the text per the indexer's
        # coherence rules).  Showing all tags from matched memories is
        # misleading — administrative tags like "lesson" or
        # "agent:foo" appear first but didn't drive the match.  The
        # user/operator wants to see why each memory surfaced.
        from .indexer import MemoryIndexer
        segments = self._indexer._segments(text) if self._indexer else []
        triggering_tags = []
        seen_tags = set()
        for m in matches:
            for tag in m.tags:
                tag_lower = tag.lower()
                if tag_lower in seen_tags:
                    continue
                if MemoryIndexer._tag_coherent_in_paragraphs(tag, segments):
                    seen_tags.add(tag_lower)
                    triggering_tags.append(tag)

        # Build notification message with the triggering tags
        tag_summary = ", ".join(f'"{t}"' for t in triggering_tags[:3])
        if len(triggering_tags) > 3:
            tag_summary += f" +{len(triggering_tags) - 3} more"

        # `trigger_keywords` in metadata kept for downstream telemetry
        # consumers — narrowed to the same triggering set.
        matched_tags = triggering_tags

        metadata = {
            "memory_matches": len(matches),
            "matched_ids": [m.id for m in matches],
            "trigger_keywords": matched_tags,
            "notification": {
                "message": f"added context about {len(matches)} memories (tags: {tag_summary})"
            },
            "_telemetry": {
                "jaato.enrichment.memory.matches": len(matches),
                "jaato.enrichment.memory.trigger_keywords": len(matched_tags),
            },
        }
        # Remember what we injected so the same bullet doesn't reappear
        # on the next tool call within this session.
        self._surfaced_memory_ids.update(m.id for m in matches)
        return enriched_text, metadata

    def on_history_cleared(self) -> None:
        """Reset per-session enrichment tracking when history is wiped.

        Called by ``JaatoSession.reset_session()`` on a true history clear
        (not a GC-driven reset that restores history).  Clears the
        ``_surfaced_memory_ids`` set so memories can surface again in the
        fresh conversation — otherwise the model would never see the
        hint bullet after a reset.
        """
        self._surfaced_memory_ids.clear()
        self._trace("on_history_cleared: cleared surfaced memory tracking")

    # ===== Tool Executors =====

    def _execute_store(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute store_memory tool.

        Creates a new memory with ``maturity="raw"``.  The optional
        ``confidence``, ``scope``, and ``evidence`` fields help the
        advisor agent during later curation.

        Args:
            args: Tool arguments (content, description, tags, and optional
                confidence, scope, evidence)

        Returns:
            Result dict with status and memory_id
        """
        description = args.get("description", "")
        tags = args.get("tags", [])
        self._trace(f"store_memory: description={description!r}, tags={tags}")

        # Validate + normalize scope, then apply the deployment write-side gate
        # (allowed_scopes) FIRST — before per-request content checks. A disallowed
        # scope is HARD-REJECTED back to the model (so it re-stores with an
        # allowed scope) rather than silently down-scoped (which would hide the
        # policy and make the model believe it stored a wider scope than it did).
        # A deployment with allowed_scopes=["project"] thus never writes to the
        # HOME/global tier at all.
        scope = args.get("scope", SCOPE_PROJECT)
        if scope not in VALID_SCOPES:
            scope = SCOPE_PROJECT
        if scope not in self._allowed_scopes:
            return {
                "status": "rejected",
                "error": (
                    f"scope '{scope}' is not allowed for this deployment "
                    f"(allowed scopes: {sorted(self._allowed_scopes)}). "
                    f"Re-store this memory with an allowed scope."
                ),
                "allowed_scopes": sorted(self._allowed_scopes),
            }

        if not self._storage or not self._indexer:
            return {
                "status": "error",
                "message": "Memory plugin not initialized"
            }

        # Validate and normalize tags: strip whitespace, reject single-char tags
        raw_tags = args.get("tags", [])
        valid_tags = [
            tag.strip() for tag in raw_tags
            if isinstance(tag, str) and len(tag.strip()) >= 2
        ]
        if not valid_tags:
            return {
                "status": "error",
                "message": (
                    "All tags were rejected — each tag must be a meaningful "
                    "word or phrase (at least 2 characters). "
                    f"Received: {raw_tags!r}"
                )
            }

        # Validate confidence (clamp to 0.0-1.0)
        confidence = args.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5

        # Create memory object — always starts as raw
        memory = Memory(
            id=f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]}",
            content=args["content"],
            description=args["description"],
            tags=valid_tags,
            timestamp=datetime.now().isoformat(),
            usage_count=0,
            maturity=MATURITY_RAW,
            confidence=confidence,
            scope=scope,
            evidence=args.get("evidence"),
            source_agent=self._agent_name,
            source_session=self._get_session_id(),
        )

        # Route to the appropriate store based on scope.  New memories
        # always land in the raw queue — they are NOT added to the
        # indexer because the indexer mirrors the curated store only.
        # The curator promotes them to the indexer via update_memory.
        if scope == SCOPE_UNIVERSAL and self._global_storage:
            self._global_storage.save(memory)
        else:
            self._storage.save(memory)

        return {
            "status": "success",
            "memory_id": memory.id,
            "message": f"Stored memory: {memory.description}",
            "tags": memory.tags,
            "maturity": memory.maturity,
            "confidence": memory.confidence,
            "scope": memory.scope,
            # Convention-based telemetry: jaato_session forwards these
            # as span attributes on the enclosing tool_span.
            "_telemetry": {
                "jaato.memory.operation": "store",
                "jaato.memory.maturity": memory.maturity,
                "jaato.memory.confidence": memory.confidence,
                "jaato.memory.scope": memory.scope,
                "jaato.memory.has_evidence": memory.evidence is not None,
                "jaato.memory.source_agent": memory.source_agent or "",
                "jaato.memory.tag_count": len(memory.tags),
            },
        }

    def _execute_retrieve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retrieve_memories tool.

        Two modes:

        - ``ids`` provided → fetch those memory IDs directly from
          either store, bypassing tag/scope/maturity filtering.  This
          is the preferred path when the agent has IDs from an
          enrichment hint — one call covers all surfaced memories.
        - otherwise → search by ``tags`` (legacy keyword path).
          Returns active memories (raw, validated) by default;
          ``scope`` and ``maturity`` narrow the search.

        Args:
            args: Tool arguments. Either ``ids`` or ``tags`` should be
                present.  Other supported keys: ``limit``, ``scope``,
                ``maturity``.

        Returns:
            Result dict with memories list including lifecycle metadata.
        """
        if not self._storage:
            return {
                "status": "error",
                "message": "Memory plugin not initialized"
            }

        ids = args.get("ids") or []
        tags = args.get("tags", [])
        limit = args.get("limit", 3)
        scope = args.get("scope")  # None = both, "project", "universal"
        maturity = args.get("maturity")  # None = active only, or specific maturity

        # ── ID fetch path ────────────────────────────────────────────
        if ids:
            self._trace(f"retrieve_memories: ids={ids}")
            id_set = set(ids)
            memories: List[Memory] = []
            seen: set = set()
            for store in (self._storage, self._global_storage):
                if not store:
                    continue
                for mem in store.load_all():
                    if mem.id in id_set and mem.id not in seen:
                        memories.append(mem)
                        seen.add(mem.id)
            if not memories:
                return {
                    "status": "no_results",
                    "message": f"No memories found for ids: {ids}"
                }
            # Preserve requested order so the agent receives results in
            # the order it asked for.
            order = {mid: i for i, mid in enumerate(ids)}
            memories.sort(key=lambda m: order.get(m.id, len(order)))
            # Skip the tags/maturity/scope filtering and limit truncation
            # — the agent asked for these specific memories explicitly.

        # ── Tag search path (legacy) ────────────────────────────────
        else:
            self._trace(f"retrieve_memories: tags={tags}, limit={limit}, scope={scope}, maturity={maturity}")
            # Determine active_only based on maturity filter
            active_only = maturity is None  # default: only active (raw, validated)

            # Search the appropriate store(s)
            memories = []
            if scope != SCOPE_UNIVERSAL and self._storage:
                memories.extend(self._storage.search_by_tags(tags, limit=limit, active_only=active_only))
            if scope != SCOPE_PROJECT and self._global_storage:
                memories.extend(self._global_storage.search_by_tags(tags, limit=limit, active_only=active_only))

            # Filter by specific maturity if requested
            if maturity is not None:
                memories = [m for m in memories if m.maturity == maturity]

            # Filter by specific scope if requested
            if scope is not None:
                memories = [m for m in memories if m.scope == scope]

            # Sort by recency (newest first) and truncate to limit
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            memories = memories[:limit]

            if not memories:
                return {
                    "status": "no_results",
                    "message": f"No memories found for tags: {tags}"
                }

        # Update usage statistics
        for mem in memories:
            mem.usage_count += 1
            mem.last_accessed = datetime.now().isoformat()
            self._storage.update(mem)

        # Compute summary stats for telemetry
        maturities_retrieved = list({m.maturity for m in memories})
        scopes_retrieved = list({m.scope for m in memories})
        avg_confidence = sum(m.confidence for m in memories) / len(memories)

        return {
            "status": "success",
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "description": m.description,
                    "content": m.content,
                    "tags": m.tags,
                    "stored": m.timestamp,
                    "usage_count": m.usage_count,
                    "maturity": m.maturity,
                    "confidence": m.confidence,
                    "scope": m.scope,
                }
                for m in memories
            ],
            "_telemetry": {
                "jaato.memory.operation": "retrieve",
                "jaato.memory.count_retrieved": len(memories),
                "jaato.memory.maturities_retrieved": maturities_retrieved,
                "jaato.memory.scopes_retrieved": scopes_retrieved,
                "jaato.memory.avg_confidence": round(avg_confidence, 3),
            },
        }

    def _execute_list_tags(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_memory_tags tool.

        Args:
            args: Tool arguments (none)

        Returns:
            Result dict with all tags
        """
        self._trace("list_memory_tags")
        if not self._indexer:
            return {
                "status": "error",
                "message": "Memory plugin not initialized"
            }

        tags = self._indexer.get_all_tags()
        memory_count = self._indexer.get_memory_count()

        # Maturity breakdown for telemetry
        maturity_counts = {}
        if self._storage:
            maturity_counts = self._storage.count_by_maturity()

        return {
            "status": "success",
            "tags": sorted(tags),
            "count": len(tags),
            "memory_count": memory_count,
            "message": f"Found {memory_count} memories with {len(tags)} unique tags",
            "_telemetry": {
                "jaato.memory.operation": "list_tags",
                "jaato.memory.total_count": memory_count,
                "jaato.memory.tag_count": len(tags),
                "jaato.memory.count_raw": maturity_counts.get(MATURITY_RAW, 0),
                "jaato.memory.count_validated": maturity_counts.get(MATURITY_VALIDATED, 0),
                "jaato.memory.count_escalated": maturity_counts.get(MATURITY_ESCALATED, 0),
                "jaato.memory.count_dismissed": maturity_counts.get(MATURITY_DISMISSED, 0),
            },
        }

    def _execute_update(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute update_memory tool.

        Finds the memory by ID in either workspace or global storage,
        applies the requested field updates, and persists.

        Args:
            args: Tool arguments (id required, plus optional maturity,
                confidence, tags, content).

        Returns:
            Result dict with updated memory status.
        """
        memory_id = args.get("id", "")
        if not memory_id:
            return {"status": "error", "message": "'id' is required"}

        # Find the memory in either store
        memory = None
        target_storage = None
        target_indexer = None

        if self._storage:
            memory = self._storage.get_by_id(memory_id)
            if memory:
                target_storage = self._storage
                target_indexer = self._indexer

        if memory is None and self._global_storage:
            memory = self._global_storage.get_by_id(memory_id)
            if memory:
                target_storage = self._global_storage
                target_indexer = self._global_indexer

        if memory is None:
            return {"status": "error", "message": f"Memory '{memory_id}' not found"}

        # Apply updates
        if "maturity" in args:
            new_maturity = args["maturity"]
            if new_maturity in VALID_MATURITIES:
                memory.maturity = new_maturity
            else:
                return {"status": "error", "message": f"Invalid maturity: {new_maturity}"}

        if "confidence" in args:
            try:
                memory.confidence = max(0.0, min(1.0, float(args["confidence"])))
            except (TypeError, ValueError):
                pass

        if "tags" in args and isinstance(args["tags"], list):
            memory.tags = [t.strip() for t in args["tags"] if isinstance(t, str) and len(t.strip()) >= 2]

        if "content" in args and isinstance(args["content"], str):
            memory.content = args["content"]

        target_storage.update(memory)
        # Rebuild the indexer from the curated store rather than
        # patching incrementally — updates can promote a raw memory
        # to curated, demote a curated memory to dismissed (which
        # removes it), or just modify tags.  All cases stay correct
        # if we rebuild from disk.  Updates are rare (curator only),
        # so the cost is acceptable.
        if target_indexer:
            target_indexer.clear()
            target_indexer.build_index(target_storage.load_curated())

        self._trace(f"update_memory: id={memory_id}, maturity={memory.maturity}")
        return {
            "status": "success",
            "memory_id": memory_id,
            "maturity": memory.maturity,
            "confidence": memory.confidence,
            "message": f"Memory updated: {memory.description}",
        }

    def _execute_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute delete_memory tool.

        Finds and deletes the memory by ID from whichever store
        (workspace or global) contains it.

        Args:
            args: Tool arguments (id required).

        Returns:
            Result dict with deletion status.
        """
        memory_id = args.get("id", "")
        if not memory_id:
            return {"status": "error", "message": "'id' is required"}

        # Try workspace store first, then global
        deleted = False
        if self._storage and self._storage.delete(memory_id):
            deleted = True
        elif self._global_storage and self._global_storage.delete(memory_id):
            deleted = True

        if deleted:
            self._trace(f"delete_memory: id={memory_id}")
            return {"status": "success", "message": f"Memory '{memory_id}' deleted"}
        else:
            return {"status": "error", "message": f"Memory '{memory_id}' not found"}

    # ===== User Command Executor =====

    def execute_memory(self, args: Dict[str, Any]) -> str:
        """Execute the memory user command.

        Subcommands:
            list              - List all stored memories
            remove <id>       - Remove a memory by ID
            edit <id>         - Edit a memory in external editor
            help              - Show detailed help

        Args:
            args: Dict with 'args' key containing list of command arguments

        Returns:
            Formatted string output for display to user
        """
        cmd_args = args.get("args", [])

        if not cmd_args:
            return self._memory_list()

        subcommand = cmd_args[0].lower()

        if subcommand == "list":
            return self._memory_list()
        elif subcommand == "remove":
            if len(cmd_args) < 2:
                return "Usage: memory remove <memory_id>"
            memory_id = cmd_args[1]
            return self._memory_remove(memory_id)
        elif subcommand == "edit":
            if len(cmd_args) < 2:
                return "Usage: memory edit <memory_id>"
            memory_id = cmd_args[1]
            return self._memory_edit(memory_id)
        elif subcommand == "help":
            return self._memory_help()
        else:
            return (
                f"Unknown subcommand: {subcommand}\n"
                "Usage: memory <list|remove|edit|help>\n"
                "  list              - List all stored memories\n"
                "  remove <id>       - Remove a memory by ID\n"
                "  edit <id>         - Edit a memory in external editor\n"
                "  help              - Show detailed help"
            )

    def _memory_list(self) -> HelpLines:
        """List all stored memories with lifecycle metadata.

        Returns HelpLines for pager display (same pattern as session list).
        Shows maturity, confidence, and scope alongside existing metadata.
        """
        if not self._storage:
            return HelpLines(lines=[("Error: Memory plugin not initialized.", "error")])

        memories = self._storage.load_all()

        if not memories:
            return HelpLines(lines=[("No memories stored yet.", "dim")])

        # Group by maturity for summary
        maturity_counts = self._storage.count_by_maturity()

        lines = []
        lines.append(("Stored Memories", "bold"))
        lines.append(("═" * 15, "bold"))

        # Show maturity summary
        summary_parts = []
        for mat in (MATURITY_RAW, MATURITY_VALIDATED, MATURITY_ESCALATED, MATURITY_DISMISSED):
            count = maturity_counts.get(mat, 0)
            if count > 0:
                summary_parts.append(f"{mat}: {count}")
        if summary_parts:
            lines.append((f"  ({', '.join(summary_parts)})", "dim"))
        lines.append(("", ""))

        for mem in memories:
            tags_str = ", ".join(mem.tags[:3])
            if len(mem.tags) > 3:
                tags_str += f" +{len(mem.tags) - 3} more"

            # Maturity indicator
            maturity_icon = {
                MATURITY_RAW: "○",
                MATURITY_VALIDATED: "◑",
                MATURITY_ESCALATED: "●",
                MATURITY_DISMISSED: "✗",
            }.get(mem.maturity, "?")

            lines.append((f"{maturity_icon} ID: {mem.id}", ""))
            lines.append((f"  Description: {mem.description}", "dim"))
            lines.append((f"  Tags: {tags_str}", "dim"))
            lines.append((f"  Created: {mem.timestamp[:10]}  |  Maturity: {mem.maturity}  |  Confidence: {mem.confidence:.0%}  |  Scope: {mem.scope}", "dim"))
            lines.append((f"  Used: {mem.usage_count} times", "dim"))
            if mem.source_agent:
                lines.append((f"  Source: {mem.source_agent}", "dim"))
            lines.append(("", ""))

        lines.append((f"Total: {len(memories)} memories", "bold"))
        return HelpLines(lines=lines)

    def _memory_remove(self, memory_id: str) -> str:
        """Remove a memory by ID."""
        if not self._storage or not self._indexer:
            return "Error: Memory plugin not initialized."

        # Check if memory exists first
        memory = self._storage.get_by_id(memory_id)
        if not memory:
            return f"Error: Memory not found: {memory_id}"

        # Delete from storage
        deleted = self._storage.delete(memory_id)

        if deleted:
            # Rebuild index from curated only — raw isn't indexed.
            existing_memories = self._storage.load_curated()
            self._indexer.clear()
            self._indexer.build_index(existing_memories)
            return f"Removed memory: {memory_id}\n  Was: {memory.description}"
        else:
            return f"Error: Failed to remove memory: {memory_id}"

    def _memory_edit(self, memory_id: str) -> str:
        """Edit a memory in external editor."""
        if not self._storage or not self._indexer:
            return "Error: Memory plugin not initialized."

        # Get the memory
        memory = self._storage.get_by_id(memory_id)
        if not memory:
            return f"Error: Memory not found: {memory_id}"

        # Get editor
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"

        # Prepare memory as YAML for editing (including lifecycle fields)
        memory_dict = {
            "description": memory.description,
            "content": memory.content,
            "tags": memory.tags,
            "maturity": memory.maturity,
            "confidence": memory.confidence,
            "scope": memory.scope,
            "evidence": memory.evidence,
        }

        # Create temp file with memory content
        try:
            import yaml
            HAS_YAML = True
        except ImportError:
            HAS_YAML = False

        try:
            # Format as YAML or JSON
            if HAS_YAML:
                content = (
                    f"# Edit memory: {memory_id}\n"
                    f"# Modify the fields below and save to update the memory.\n"
                    f"# Close without saving to cancel.\n"
                    f"#\n"
                    f"# Fields:\n"
                    f"#   description: Brief summary (1-2 sentences)\n"
                    f"#   content: Full content/explanation\n"
                    f"#   tags: List of keywords for retrieval\n"
                    f"#   maturity: raw | validated | escalated | dismissed\n"
                    f"#   confidence: 0.0 to 1.0\n"
                    f"#   scope: project | universal\n"
                    f"#   evidence: What triggered this learning (optional)\n"
                    f"\n"
                )
                import yaml
                content += yaml.safe_dump(
                    memory_dict,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
                suffix = ".yaml"
            else:
                content = (
                    f"// Edit memory: {memory_id}\n"
                    f"// Modify the fields below and save to update the memory.\n"
                    f"// Close without saving to cancel.\n"
                    f"\n"
                )
                content += json.dumps(memory_dict, indent=2, ensure_ascii=False)
                suffix = ".json"

            # Write to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=suffix,
                delete=False,
                encoding='utf-8',
            ) as f:
                f.write(content)
                temp_path = f.name

            original_content = content

            # Open in editor
            result = subprocess.run([editor, temp_path], check=False)

            if result.returncode != 0:
                os.unlink(temp_path)
                return f"Editor exited with code {result.returncode}. Edit cancelled."

            # Read back edited content
            with open(temp_path, 'r', encoding='utf-8') as f:
                edited_content = f.read()

            os.unlink(temp_path)

            # Check if content was modified
            if edited_content.strip() == original_content.strip():
                return "No changes made."

            # Parse edited content
            # Strip comment lines
            lines = []
            for line in edited_content.split('\n'):
                stripped = line.strip()
                if not stripped.startswith('#') and not stripped.startswith('//'):
                    lines.append(line)
            clean_content = '\n'.join(lines)

            try:
                if HAS_YAML:
                    parsed = yaml.safe_load(clean_content)
                else:
                    parsed = json.loads(clean_content)
            except Exception as e:
                return f"Error parsing edited content: {e}\nEdit cancelled."

            # Validate schema
            validation_error = self._validate_memory_schema(parsed)
            if validation_error:
                return f"Validation error: {validation_error}\nEdit cancelled."

            # Update memory (core + lifecycle fields)
            memory.description = parsed["description"]
            memory.content = parsed["content"]
            memory.tags = parsed["tags"]
            if "maturity" in parsed:
                memory.maturity = parsed["maturity"]
            if "confidence" in parsed:
                memory.confidence = float(parsed["confidence"])
            if "scope" in parsed:
                memory.scope = parsed["scope"]
            if "evidence" in parsed:
                memory.evidence = parsed["evidence"]

            # Save updated memory (routes to raw or curated based on
            # current location and new maturity).
            self._storage.update(memory)

            # Rebuild index from curated only — raw isn't indexed.
            existing_memories = self._storage.load_curated()
            self._indexer.clear()
            self._indexer.build_index(existing_memories)

            return f"Updated memory: {memory_id}\n  Description: {memory.description}"

        except Exception as e:
            # Clean up temp file if it exists
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            return f"Error editing memory: {e}"

    def _validate_memory_schema(self, data: Dict[str, Any]) -> Optional[str]:
        """Validate that edited memory data conforms to schema.

        Validates both the original core fields and the lifecycle fields
        added for the knowledge-curation system.

        Args:
            data: Parsed memory data dict

        Returns:
            Error message if invalid, None if valid
        """
        # Required fields
        required_fields = ["description", "content", "tags"]
        for fld in required_fields:
            if fld not in data:
                return f"Missing required field: {fld}"

        # Type validation — core fields
        if not isinstance(data["description"], str):
            return "description must be a string"
        if not isinstance(data["content"], str):
            return "content must be a string"
        if not isinstance(data["tags"], list):
            return "tags must be a list"
        if not all(isinstance(tag, str) for tag in data["tags"]):
            return "all tags must be strings"

        # Non-empty validation
        if not data["description"].strip():
            return "description cannot be empty"
        if not data["content"].strip():
            return "content cannot be empty"
        if not data["tags"]:
            return "tags cannot be empty"

        # Tag quality: each tag must be at least 2 characters
        short_tags = [tag for tag in data["tags"] if len(tag.strip()) < 2]
        if short_tags:
            return (
                f"tags must be meaningful words (at least 2 characters each), "
                f"got: {short_tags!r}"
            )

        # Lifecycle field validation (optional in schema, validated when present)
        if "maturity" in data:
            if data["maturity"] not in VALID_MATURITIES:
                return (
                    f"maturity must be one of {sorted(VALID_MATURITIES)}, "
                    f"got: {data['maturity']!r}"
                )

        if "confidence" in data:
            try:
                conf = float(data["confidence"])
                if not (0.0 <= conf <= 1.0):
                    return "confidence must be between 0.0 and 1.0"
            except (TypeError, ValueError):
                return f"confidence must be a number, got: {data['confidence']!r}"

        if "scope" in data:
            if data["scope"] not in VALID_SCOPES:
                return (
                    f"scope must be one of {sorted(VALID_SCOPES)}, "
                    f"got: {data['scope']!r}"
                )

        if "evidence" in data:
            if data["evidence"] is not None and not isinstance(data["evidence"], str):
                return "evidence must be a string or null"

        return None

    def _memory_help(self) -> HelpLines:
        """Show detailed help for the memory command."""
        return HelpLines(lines=[
            ("Memory Command", "bold"),
            ("", ""),
            ("Manage persistent memories stored by the AI. Memories persist across", ""),
            ("sessions and help the AI recall context, patterns, and lessons learned.", ""),
            ("", ""),
            ("Memories go through a knowledge-curation lifecycle:", ""),
            ("  raw -> validated -> escalated (promoted to reference)", "dim"),
            ("               \\-> dismissed (rejected by advisor)", "dim"),
            ("", ""),
            ("USAGE", "bold"),
            ("    memory [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list              List all stored memories with metadata", "dim"),
            ("                      Shows ID, description, tags, maturity, confidence", "dim"),
            ("", ""),
            ("    remove <id>       Remove a memory by its ID", "dim"),
            ("                      The memory will be permanently deleted", "dim"),
            ("", ""),
            ("    edit <id>         Edit a memory in your external editor ($EDITOR)", "dim"),
            ("                      Opens the memory in YAML format for editing", "dim"),
            ("                      Validates the schema on save", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    memory                         List all memories (default)", "dim"),
            ("    memory list                    List all memories", "dim"),
            ("    memory remove mem_20240101_... Remove a specific memory", "dim"),
            ("    memory edit mem_20240101_...   Edit a specific memory", "dim"),
            ("", ""),
            ("EDIT FORMAT", "bold"),
            ("    When editing, the memory is presented in YAML format with:", ""),
            ("      description: Brief summary of the memory", "dim"),
            ("      content: Full content/explanation", "dim"),
            ("      tags: List of keywords for retrieval", "dim"),
            ("      maturity: raw | validated | escalated | dismissed", "dim"),
            ("      confidence: 0.0 to 1.0 (accuracy self-assessment)", "dim"),
            ("      scope: project | universal", "dim"),
            ("      evidence: What triggered this learning (optional)", "dim"),
            ("", ""),
            ("    Lines starting with # are comments and will be ignored.", ""),
            ("", ""),
            ("MATURITY LIFECYCLE", "bold"),
            ("    ○ raw          Fresh from agent, awaiting advisor review", "dim"),
            ("    ◑ validated    Advisor confirmed valuable, kept as memory", "dim"),
            ("    ● escalated    Promoted to permanent reference (knowledge)", "dim"),
            ("    ✗ dismissed    Rejected by advisor (incorrect/trivial)", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - Memories are stored in .jaato/memories.jsonl", "dim"),
            ("    - Each memory has a unique ID starting with 'mem_'", "dim"),
            ("    - Use Tab completion for memory IDs in remove/edit", "dim"),
            ("    - Only active memories (raw, validated) appear in prompt hints", "dim"),
        ])


def create_plugin() -> MemoryPlugin:
    """Factory function to create the memory plugin instance.

    Returns:
        MemoryPlugin instance
    """
    return MemoryPlugin()
