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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from jaato_sdk.plugins.base import (
    CommandCompletion,
    HelpLines,
    PromptEnrichmentResult,
    ToolResultEnrichmentResult,
    UserCommand,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema, DISCOVERABILITY_EAGER, DISCOVERABILITY_DEFERRED
from .indexer import MemoryIndexer
from .models import (
    ACTIVE_MATURITIES,
    CURATED_MATURITIES,
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
from .similarity import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DuplicateMatch,
    content_tokens,
    duplicate_fields,
    duplicate_note,
    duplicate_telemetry,
    find_near_duplicate,
    recent_store_window,
    resolve_threshold,
)
from .storage import MemoryStorage
from jaato_server.shared.plugins.runner_forwarding import RunnerForwardingMixin
from jaato_server.shared.trace import trace as _trace_write


#: How many of this session's own writes are kept for near-duplicate
#: comparison (#973).  Large enough to cover a runaway store loop — the
#: reported incident wrote 83 memories in one turn — and small enough that
#: the per-store scan stays trivial.
RECENT_STORE_CACHE_SIZE = 200


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
        self._storage_path_template: str = ".jaato/memories"
        # Memory IDs whose hint bullet has already been injected into the
        # model's context during this session.  Prevents the same "💡
        # Available Memories" block from being re-surfaced on every tool
        # call when the same memory keeps matching.  Cleared by
        # on_history_cleared() when the session history is wiped.
        self._surfaced_memory_ids: Set[str] = set()
        # ── Near-duplicate reporting (#973) ──────────────────────────
        # Score at or above which a new memory is reported as a
        # near-duplicate of one already stored.  Resolved from config in
        # initialize(); see ``similarity`` for the measurement behind the
        # default.
        self._duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD
        # OFF by default, deliberately.  Reporting a duplicate costs an
        # advisory field; REJECTING one on a mistuned threshold silently
        # discards a real memory, which for this plugin is the worst
        # available failure.  Opting in is the explicit act.
        self._reject_duplicates: bool = False
        # Art. 15(4) (#1123).  OFF by default: on, it withholds every
        # memory a curator has not marked, and a deployment that never ran
        # a curator would silently lose its whole learning loop on upgrade.
        # ``validate`` warns when it is set with no curator configured,
        # because there the knob means "never re-inject anything".
        self._require_curation: bool = False
        # What THIS plugin instance has written, oldest first.  The pool
        # that catches a runaway store loop on a store with nothing
        # curated yet — the incident in #973 was 83 writes inside a single
        # turn — at zero I/O.  Capped because the plugin deliberately
        # survives ``reset_for_next_session``, so nothing else bounds it.
        self._recent_stores: List[Memory] = []
        # Which memory ids each session RETRIEVED (#1232), keyed by the
        # executing session's own id -- what the rail highlights beside the
        # ones a session wrote (``source_session``).  Bounded by
        # ``RETRIEVED_SESSION_MEMORY``; see ``_note_retrieved``.
        self._retrieved_by_session: Dict[str, Set[str]] = {}

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
            from jaato_server.shared.session_context import get_current_session
            session = get_current_session()
        except LookupError:
            session = None
        if session is not None:
            from jaato_server.shared.dynamic_instructions import _resolve_session_id
            sid = _resolve_session_id(session)
            if sid:
                return sid
        return self._session_id

    def _model_provenance(self) -> Optional[Dict[str, Any]]:
        """Which model is writing this memory (Art. 15(4), #1123).

        Read PER EXECUTION off the currently executing session, the same
        way :meth:`_get_session_id` reads the session id and for the same
        reason: this plugin instance is SHARED across sibling subagents,
        so anything stashed on ``self`` would be whichever sibling wrote
        last.  ``JaatoSession._model_provenance`` is the one definition of
        the stamp (#1109), so a memory and a piece of model media name
        their binding identically.

        Returns ``None`` when there is no session in context -- a
        standalone unit test, a script -- which the record carries as
        *provenance unknown*.  Never invented: a memory whose author
        cannot be established must not claim one.
        """
        try:
            from jaato_server.shared.session_context import get_current_session
            session = get_current_session()
        except LookupError:
            return None
        resolver = getattr(session, "_model_provenance", None)
        if not callable(resolver):
            return None
        try:
            return resolver()
        except Exception:  # noqa: BLE001 -- a stamp must not fail a store
            self._trace("store_memory: provenance unavailable")
            return None

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

        Memory storage lives in the ``~/.jaato/memories`` DIRECTORY:
        - ``memories/raw/{id}.json`` — pending queue (one file per memory)
        - ``memories/curated.jsonl`` — curated knowledge base

        The sibling ``memories.jsonl`` grant is the legacy single-file
        store.  It is read — once, by ``MemoryStore._recover_legacy_file``,
        which migrates a populated one into ``curated.jsonl`` (#912) — so
        this grant is load-bearing rather than reserved for a migration
        nobody had written.  ``rw`` because the read resolves through the
        same path rules; the migration never writes to it.

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
                - storage_path: Directory holding the memory store —
                  ``raw/`` + ``curated.jsonl`` (default: .jaato/memories).
                  A legacy ``*.jsonl`` path still resolves to the
                  sibling directory named after its stem.
                - enrichment_limit: Max hints to show in prompt (default: 5)
                - duplicate_threshold: Similarity at or above which a new
                  memory is REPORTED as a near-duplicate (default: 0.75).
                - reject_duplicates: Refuse the write instead of reporting
                  it (default: False).
        """
        config = config or {}
        self._agent_name = config.get("agent_name")
        self._allowed_scopes = self._resolve_allowed_scopes(config.get("allowed_scopes"))
        self._duplicate_threshold = resolve_threshold(
            config.get("duplicate_threshold"))
        self._reject_duplicates = bool(config.get("reject_duplicates", False))
        self._require_curation = bool(config.get("require_curation", False))
        self._trace(
            f"initialize: duplicate_threshold={self._duplicate_threshold}, "
            f"reject_duplicates={self._reject_duplicates}, "
            f"require_curation={self._require_curation}")
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
        self._storage_path_template = config.get("storage_path", ".jaato/memories")

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

        # Global storage at ~/.jaato/memories — cross-session knowledge
        # shared by UNCONFINED agents.  This tier is OPTIONAL: a confined
        # session is correctly denied HOME, so the tier is simply absent for it
        # and the workspace tier is the only (priority) store.  Configurable via
        # "global_storage_path" for testing.
        global_path = config.get(
            "global_storage_path",
            str(Path.home() / ".jaato" / "memories"),
        )
        self._global_storage = MemoryStorage(global_path)
        self._global_indexer = MemoryIndexer()
        global_memories = self._safe_load_curated(
            self._global_storage, tier="global (HOME)")
        self._global_indexer.build_index(global_memories)
        self._trace(f"initialize: global_path={global_path}, global_curated_memories={len(global_memories)}")

    def _safe_load_curated(self, storage: "MemoryStorage", tier: str) -> List["Memory"]:
        """Load a tier's curated memories, tolerating an inaccessible store.

        Called from ``initialize`` (to build the index) and from
        ``_duplicate_pools`` (to scan for near-duplicates), so ``tier``
        names both the tier and the caller's purpose.

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
                f"{tier} memory tier not loadable here ({e}); "
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
                    "default": ".jaato/memories",
                    "description": (
                        "Directory holding the memory store "
                        "(raw/ + curated.jsonl). A legacy *.jsonl path "
                        "resolves to the sibling directory named after "
                        "its stem."
                    ),
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
                "duplicate_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": DEFAULT_DUPLICATE_THRESHOLD,
                    "description": (
                        "Token-overlap score at or above which store_memory "
                        "reports the new memory as a near-duplicate of one "
                        "already stored (fields duplicate_of / "
                        "duplicate_similarity / duplicate_source, plus a note "
                        "on `message`). Reporting only — the store still "
                        "succeeds unless reject_duplicates is set. Raise it to "
                        "report less, lower it to report more."
                    ),
                },
                "reject_duplicates": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Refuse a near-duplicate write instead of storing it "
                        "and reporting. OFF by default: a threshold that is "
                        "wrong in the rejecting direction silently discards "
                        "real memories, and an agent legitimately re-storing "
                        "a fact with better wording is a real pattern. Turn "
                        "this on only with duplicate_threshold tuned against "
                        "your own store."
                    ),
                },
                "require_curation": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Re-inject only memories a curator has marked "
                        "(EU AI Act Art. 15(4): a system that continues to "
                        "learn must not feed possibly biased outputs back as "
                        "inputs unmitigated). Memories are STORED as before; "
                        "retrieval withholds any without a `curated_by` "
                        "stamp and says how many. OFF by default: on, a "
                        "deployment that never ran a curator loses its whole "
                        "learning loop -- which is why `jaato-scaffold "
                        "validate` warns when this is set with no curator "
                        "configured."
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
                    'agent for potential promotion to permanent knowledge. '
                    'If the result carries `duplicate_of`, the store already held '
                    'essentially this fact under that memory ID: STOP re-storing it '
                    'and retrieve that ID instead if you need it. Rephrasing a fact '
                    'you have already stored does not make it a new memory.'
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
                discoverability=DISCOVERABILITY_EAGER,
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
                    '(cross-session) memories. '
                    'The result reports `matched` (how many memories matched '
                    'in total) beside `count` (how many are in this result). '
                    'When `truncated` is true you are seeing a SUBSET: call '
                    'again with a larger `limit` before answering as though '
                    'this were everything you know.'
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
                            "description": (
                                "Max number of memories to retrieve "
                                "(default: 3, ignored when `ids` is used). "
                                "The default suits a targeted lookup; for "
                                "'what do you know about X' pass a larger "
                                "value. The result's `matched` and "
                                "`truncated` fields tell you whether this "
                                "limit cut anything off."
                            )
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
                discoverability=DISCOVERABILITY_EAGER,
            ),
            ToolSchema(
                name='list_memory_tags',
                description=(
                    'List all available memory tags to discover what has been stored. '
                    'Useful for exploring the knowledge base or finding related topics. '
                    'Tags and memory_count describe the CURATED store; '
                    'pending_curation reports how many raw memories are still '
                    'awaiting review, which no tag search can reach — use '
                    "retrieve_memories with maturity='raw' for those."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                category="memory",
                discoverability=DISCOVERABILITY_EAGER,
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
                discoverability=DISCOVERABILITY_DEFERRED,
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
                discoverability=DISCOVERABILITY_DEFERRED,
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
            "`~/.jaato/memories`, shared across all sessions and workspaces. "
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

        Workspace tier only, from THIS copy of the plugin.  No longer the
        source of the ``MemoryListEvent`` the daemon pushes after a
        ``memory`` command (#1232): that push is answered by the copy that
        holds the store, through ``JaatoServer.memory_list_event``, because
        on a runner-served session the daemon's copy is not it.  Kept as the
        capability marker ``command_router`` reads (a plugin whose
        completions are dynamic) and for in-process callers.

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

    # ===== The memory rail (#1232) =====
    #
    # Read and curate the store for a HUMAN, through the daemon's memory
    # verbs (``shared/plugins/memory/verbs.py``).  Every method below runs
    # on the plugin copy that holds the store -- the runner's, on a
    # runner-served session -- because the verbs are answered by a runner
    # RPC and never by the daemon's own copy of this plugin.

    #: How many sessions' retrieval sets are remembered.  The instance
    #: survives ``reset_for_next_session`` on a pool slot, so nothing else
    #: bounds the map; oldest-first eviction keeps the one a live rail asks
    #: about.
    RETRIEVED_SESSION_MEMORY = 64

    def _tiers(self) -> List[Tuple[str, Any, Any]]:
        """``(tier, storage, indexer)`` for each tier this plugin holds.

        Workspace first: an id present in both (possible only by hand-edit)
        resolves to the workspace copy, the precedence ``_execute_update``
        and ``_execute_delete`` already use.  A tier whose storage is
        ``None`` is absent here, not "empty".
        """
        tiers = []
        if self._storage is not None:
            tiers.append(("workspace", self._storage, self._indexer))
        if self._global_storage is not None:
            tiers.append(("global", self._global_storage, self._global_indexer))
        return tiers

    def _locate_memory(self, memory_id: str) -> Optional[Tuple[Memory, str, Any, Any]]:
        """``(memory, tier, storage, indexer)`` for ``memory_id``, or ``None``."""
        for tier, storage, indexer in self._tiers():
            memory = storage.get_by_id(memory_id)
            if memory is not None:
                return memory, tier, storage, indexer
        return None

    def _note_retrieved(self, memories: List[Memory]) -> None:
        """Remember which ids THIS session retrieved, for the rail's highlight.

        Keyed by the executing session's own id (``_get_session_id``, the
        per-sibling resolution ``source_session`` already uses), so a
        subagent's retrievals are its own.  Bounded by
        :attr:`RETRIEVED_SESSION_MEMORY` sessions.

        Best-effort by construction: it runs inside the retrieval path, and a
        highlight for a rail must never be able to fail a retrieval.
        """
        retrieved = getattr(self, "_retrieved_by_session", None)
        if retrieved is None or not memories:
            return
        try:
            sid = self._get_session_id()
        except Exception:  # noqa: BLE001 -- a highlight never fails a retrieval
            return
        if not sid:
            return
        seen = retrieved.pop(sid, set())
        seen.update(m.id for m in memories)
        retrieved[sid] = seen
        while len(retrieved) > self.RETRIEVED_SESSION_MEMORY:
            retrieved.pop(next(iter(retrieved)))

    def memory_row(
        self, memory: Memory, tier: str, session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One memory as the rail lists it: every field but the content.

        ``tier`` says which store it came from (``workspace`` / ``global``)
        -- deliberately not ``scope``, which already means *how broadly the
        memory applies* (``project`` / ``universal``).  Timestamps, usage
        and both provenance stamps are the stored record's own, passed
        through unchanged: ``curated_by`` is ``None`` on every raw memory,
        and ``generated_by`` is ``None`` on a record predating #1123.
        """
        row: Dict[str, Any] = {
            "id": memory.id,
            "description": memory.description,
            "tags": list(memory.tags),
            "maturity": memory.maturity,
            "confidence": memory.confidence,
            "scope": memory.scope,
            "tier": tier,
            "timestamp": memory.timestamp,
            "last_accessed": memory.last_accessed,
            "usage_count": memory.usage_count,
            "generated_by": memory.generated_by,
            "curated_by": memory.curated_by,
            "source_agent": memory.source_agent,
            "source_session": memory.source_session,
        }
        if session_id:
            row["written_this_session"] = memory.source_session == session_id
            row["retrieved_this_session"] = (
                memory.id in self._retrieved_by_session.get(session_id, ()))
        return row

    def memory_rows(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every memory in both tiers, raw and curated, as rail rows.

        The whole store, deliberately including RAW -- the curator's queue,
        which prompt enrichment never surfaces.  Showing it to a person is
        the point: an unvetted memory is the one somebody needs to look at.
        """
        rows: List[Dict[str, Any]] = []
        for tier, storage, _indexer in self._tiers():
            rows.extend(self.memory_row(m, tier, session_id)
                        for m in storage.load_all())
        return rows

    def memory_record(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """One memory WITH its content and evidence, or ``None``."""
        found = self._locate_memory(memory_id)
        if found is None:
            return None
        memory, tier, _storage, _indexer = found
        record = self.memory_row(memory, tier)
        record["content"] = memory.content
        record["evidence"] = memory.evidence
        return record

    def edit_memory_structured(
        self,
        memory_id: str,
        *,
        description: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        maturity: Optional[str] = None,
        curator: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The editor-free ``memory edit``: a structured edit, or a curation.

        ``memory edit`` spawns ``$EDITOR`` in the process that holds this
        plugin -- the runner, on the daemon's host -- which a browser cannot
        drive.  This is the same edit as data.  ``None`` leaves a field as it
        is.  The merged record is validated by :meth:`_validate_memory_schema`
        -- the validator the editor path uses -- BEFORE anything is written,
        so a refused edit changes nothing.

        A ``maturity`` that differs from the current one moves through
        :meth:`_stamp_curation`, the one helper that records ``curated_by``
        (the AST guard on every ``maturity`` writer requires it): approve is
        ``validated``, dismiss is ``dismissed``.  ``curator`` is the stamp's
        identity -- the daemon passes the HUMAN the transport authenticated,
        because a rail action has no model in context.

        Returns ``{"ok": True, "memory": <row>}`` or ``{"ok": False,
        "error", "category"}`` with ``category`` ``not_found`` / ``invalid``.
        A memory DISMISSED out of the raw queue is unlinked by the storage
        layer (it keeps no dismissed trace), so its answer carries the row
        as it was written and the next list no longer shows it.
        """
        found = self._locate_memory(memory_id)
        if found is None:
            return {"ok": False, "category": "not_found",
                    "error": f"Memory not found: {memory_id}"}
        memory, tier, storage, indexer = found
        draft = {
            "description": memory.description if description is None else description,
            "content": memory.content if content is None else content,
            "tags": list(memory.tags) if tags is None else tags,
            "maturity": memory.maturity if maturity is None else maturity,
        }
        err = self._validate_memory_schema(draft)
        if err:
            return {"ok": False, "category": "invalid", "error": err}
        memory.description = draft["description"]
        memory.content = draft["content"]
        memory.tags = [t.strip() for t in draft["tags"]]
        if draft["maturity"] != memory.maturity:
            memory.maturity = draft["maturity"]
            self._stamp_curation(memory, draft["maturity"], curator=curator)
        storage.update(memory)
        if indexer is not None:
            indexer.clear()
            indexer.build_index(storage.load_curated())
        self._trace(
            f"edit_memory_structured: id={memory_id} maturity={memory.maturity}")
        return {"ok": True, "memory": self.memory_row(memory, tier)}

    def remove_memory(self, memory_id: str) -> Dict[str, Any]:
        """Remove a memory through the existing delete path (``delete_memory``).

        Not a second deletion route: this is :meth:`_execute_delete`, the
        tool's own executor, with its answer reshaped for the rail.
        """
        result = self._execute_delete({"id": memory_id})
        if result.get("status") == "success":
            return {"ok": True, "memory_id": memory_id}
        return {"ok": False, "category": "not_found",
                "error": str(result.get("error") or f"Memory not found: {memory_id}")}

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

    # ===== Near-duplicate reporting (#973) =====

    def _duplicate_pools(self) -> List[tuple]:
        """Assemble the pools a new memory is compared against.

        Two pools, ordered so the nearest-in-time evidence is named first
        when scores tie:

        1. ``session`` — what THIS plugin instance has written.  Catches a
           runaway store loop on a store with nothing curated yet, which
           every deployment's first session is, at zero I/O.
        2. ``curated`` — both curated stores, workspace and global.  This
           is the pool that would have caught the reported incident: all
           three repeated facts were already in ``curated.jsonl``.  Both
           tiers are scanned regardless of the new memory's scope, because
           "you already know this" does not become false when the fact is
           filed under the other scope, and because ``retrieve_memories``
           searches both by default.

        THE RAW QUEUE IS DELIBERATELY NOT A POOL.  Three reasons, and the
        third is the one that makes the omission safe rather than merely
        cheap:

        - Raw is unindexed **by design** (see ``_execute_store``: new
          memories are not added to the indexer because the indexer mirrors
          the curated store).  Scanning it per write would add a second,
          hidden read path into the curator's private queue and make every
          store O(queue) in file opens — growing precisely when curation
          has fallen behind, i.e. when the system is already unhealthy.
        - Holding un-consolidated near-duplicates until someone merges them
          is what the queue is FOR.  "Is this a duplicate of something
          awaiting curation?" is the curator's question, asked once per
          drain, not the producer's, asked once per write.
        - The raw-versus-raw case that actually bites — a loop — happens
          inside one session, and pool 1 covers it without touching disk.

        What is therefore NOT caught: a memory duplicating a raw one left
        by an *earlier* session that the curator has not yet drained.  That
        is one duplicate per session rather than 83 per turn, and it is
        material the curator is about to consolidate anyway.

        Returns:
            ``(source_label, memories)`` pairs for
            :func:`~.similarity.find_near_duplicate`.
        """
        pools: List[tuple] = [("session", list(self._recent_stores))]
        curated: List[Memory] = []
        for storage, tier in (
            (self._storage, "workspace (duplicate scan)"),
            (self._global_storage, "global (duplicate scan)"),
        ):
            if storage is not None:
                curated.extend(self._safe_load_curated(storage, tier=tier))
        pools.append(("curated", curated))
        return pools

    def _duplicate_verdict(
        self, memory: Memory,
    ) -> tuple:
        """Decide what a new memory's resemblance to the store means.

        Args:
            memory: The memory about to be written.  Not yet saved, so it
                cannot match itself.

        Returns:
            ``(match, rejection)``.  ``match`` is the
            :class:`~.similarity.DuplicateMatch` found, or ``None``.
            ``rejection`` is a complete tool-result dict to return INSTEAD
            of storing — non-``None`` only when a match was found and the
            deployment opted into ``reject_duplicates``.  The pair shape
            keeps the decision (here) separate from the two things the
            caller does with it, and keeps ``_execute_store`` to one
            branch.
        """
        match = find_near_duplicate(
            content_tokens(memory.description, memory.content),
            self._duplicate_pools(),
            threshold=self._duplicate_threshold,
        )
        if match is not None:
            self._trace(
                f"store_memory: near-duplicate of {match.memory_id} "
                f"(similarity={match.similarity}, source={match.source}, "
                f"reject={self._reject_duplicates})")
        if match is None or not self._reject_duplicates:
            return match, None
        return match, {
            "status": "rejected",
            "error": (
                f"Not stored: this is a near-duplicate (similarity "
                f"{match.similarity:.2f}) of {match.memory_id}, which is "
                f"already stored: {match.description!r}. This deployment "
                f"rejects duplicate memories. Retrieve that memory instead, "
                f"or store genuinely new information."
            ),
            **duplicate_fields(match),
        }

    def _remember_store(self, memory: Memory) -> None:
        """Record a successful write in this session's own-writes pool.

        Args:
            memory: The memory just saved.
        """
        self._recent_stores.append(memory)
        self._recent_stores = recent_store_window(
            self._recent_stores, RECENT_STORE_CACHE_SIZE)

    def _execute_store(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute store_memory tool.

        Creates a new memory with ``maturity="raw"``.  The optional
        ``confidence``, ``scope``, and ``evidence`` fields help the
        advisor agent during later curation.

        Args:
            args: Tool arguments (content, description, tags, and optional
                confidence, scope, evidence)

        Returns:
            On success, a dict carrying ``status="success"``, ``memory_id``,
            ``message``, ``tags``, ``maturity``, ``confidence`` and
            ``scope``.

            **Near-duplicate reporting (#973).**  When the new memory
            closely resembles one already stored, three further keys are
            present — and they are *absent*, not ``None``, otherwise, so
            ``"duplicate_of" in result`` is the whole test and a clean
            store keeps exactly the shape it has always had:

            - ``duplicate_of`` — the id of the memory it resembles, usable
              directly as ``retrieve_memories(ids=[…])``;
            - ``duplicate_similarity`` — the score, ``0.0``–``1.0``;
            - ``duplicate_source`` — ``"session"`` (this same session wrote
              it) or ``"curated"`` (it is in the curated store).

            ``message`` carries the same finding as prose, because it is the
            anchor field tool-result enrichment writes back to (#922) and
            the part a model reads most reliably.

            **The store still succeeds.**  The result describes what is
            true, and what is true is that the memory was written *and*
            that the store already held something very like it.  A
            deployment that would rather fail the call sets
            ``reject_duplicates``, and then gets ``status="rejected"`` with
            the same three keys and nothing written.

            Failure shapes are unchanged: ``status="rejected"`` for a
            disallowed scope and ``status="error"`` for an uninitialized
            plugin or unusable tags.
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
                "error": "Memory plugin not initialized"
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
                "error": (
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
            # Art. 15(4) (#1123): WHICH MODEL wrote it.  Stamped here, by
            # the plugin, and never taken from ``args`` -- provenance a
            # subject asserts about itself is not provenance, and
            # ``store_memory``'s schema deliberately has no such
            # parameter for a model to fill in.
            generated_by=self._model_provenance(),
        )

        # Is this something we already know?  Asked BEFORE the write, so
        # the candidate cannot match itself.  Reporting never blocks the
        # write; ``rejection`` is non-None only under the opt-in knob.
        duplicate, rejection = self._duplicate_verdict(memory)
        if rejection is not None:
            return rejection

        # Route to the appropriate store based on scope.  New memories
        # always land in the raw queue — they are NOT added to the
        # indexer because the indexer mirrors the curated store only.
        # The curator promotes them to the indexer via update_memory.
        if scope == SCOPE_UNIVERSAL and self._global_storage:
            self._global_storage.save(memory)
        else:
            self._storage.save(memory)
        self._remember_store(memory)

        return {
            "status": "success",
            "memory_id": memory.id,
            "message": (
                f"Stored memory: {memory.description}"
                f"{duplicate_note(duplicate)}"
            ),
            **duplicate_fields(duplicate),
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
                **duplicate_telemetry(duplicate),
            },
        }

    @staticmethod
    def _retrieval_message(returned: int, matched: int, limit: int) -> str:
        """Say in prose how much of the match set this result carries.

        A field is what a caller reads; prose is what a MODEL reads, and
        #982's whole failure was a model reasoning correctly from a result
        that did not mention the 23 memories it was not shown.  The
        sentence is also the anchor field this result is enriched through
        (#922) — the wordiest string in the dict — which is why it is
        always present rather than only on a truncated result.

        Args:
            returned: How many memories the payload carries.
            matched: How many matched in total.
            limit: The ``limit`` that produced the truncation.

        Returns:
            A complete sentence.  On a truncated result it names both
            numbers and the remedy; otherwise it states plainly that this
            is everything, because "3 of 3" and "3 of 26" must not be the
            same sentence.
        """
        if returned >= matched:
            return (
                f"Returned all {matched} matching "
                f"{'memory' if matched == 1 else 'memories'}."
            )
        return (
            f"Returned {returned} of {matched} matching memories — "
            f"truncated by limit={limit}. Call retrieve_memories again "
            f"with a larger `limit` to see the rest."
        )

    def _search_stores_for(
        self,
        tags: List[str],
        scope: Optional[str],
        maturity: Optional[str],
    ) -> List[Memory]:
        """Query both stores for every match, without truncating either.

        Args:
            tags: Tags to search for (ignored on the maturity path).
            scope: ``"project"`` / ``"universal"`` to query one store, or
                ``None`` for both.
            maturity: A maturity to query, or ``None`` for the active
                tag search.

        Returns:
            The concatenated matches, unordered and possibly holding the
            same id twice when both stores carry it.  Ranking, de-duping
            and truncation belong to :meth:`_search_both_stores`.
        """
        found: List[Memory] = []
        stores = (
            self._storage if scope != SCOPE_UNIVERSAL else None,
            self._global_storage if scope != SCOPE_PROJECT else None,
        )
        for store in stores:
            if store is None:
                continue
            if maturity is not None:
                # MATURITY QUERIES GO TO THE MATURITY STORE.
                #
                # ``search_by_tags`` reads ``self._curated.load_all()`` only.
                # Since 3f019999 split the raw queue (a folder) from the
                # curated store (a file), NO tag-search query can return a raw
                # memory however well tagged -- the store that path reads no
                # longer contains any.  ``search_by_maturity`` was added in
                # that same commit to source ``raw`` from the raw queue, and
                # the tool handler was never repointed at it: it had zero
                # production callers, only its own test.
                #
                # So ``retrieve_memories(maturity="raw")`` -- which the
                # shipped memory-advisor persona opens Pass 2 with, and which
                # this plugin's own docstring and the tool schema both promise
                # -- returned nothing, always.  Two curator sessions concluded
                # their store was empty with twelve files on disk.
                found.extend(store.search_by_maturity({maturity}, limit=None))
            else:
                # Tagless/active search keeps the legacy path: no maturity was
                # asked for, so the curated store is the right source.
                found.extend(
                    store.search_by_tags(tags, limit=None, active_only=True))
        return found

    def _search_both_stores(
        self,
        tags: List[str],
        scope: Optional[str],
        maturity: Optional[str],
    ) -> List[Memory]:
        """Rank every match across both stores, ready to be truncated once.

        Args:
            tags: Tags to search for (ignored on the maturity path).
            scope: ``"project"`` / ``"universal"`` / ``None`` for both.
            maturity: A maturity to query, or ``None`` for active search.

        Returns:
            Every matching memory, de-duplicated by id, ordered by tag
            overlap (descending) then recency.  **Not truncated** — the
            length IS the ``matched`` figure the caller reports, and the
            caller slices it to ``limit``.

        WHY THIS IS ONE FUNCTION AND NOT TWO CALLS (#982).  This used to
        be ``search_by_tags(tags, limit=limit)`` against each store
        separately, merged, re-sorted by timestamp and truncated to
        ``limit`` again.  Two things were wrong with that, and only the
        first was reported:

        - **The total was unrecoverable.**  Each store had already thrown
          away its own count, so nothing downstream could say whether 3
          results meant 3 matches or 26.  A complete answer and a
          12%-complete answer were byte-identical, both stamped
          ``success``.
        - **The page was the wrong page.**  With 26 matches split across
          two stores and ``limit=3``, each store returned ITS top 3 by tag
          overlap, and the merged 6 were then ranked by TIMESTAMP alone —
          so the surviving 3 were the newest of an arbitrary sample, not
          the best 3 overall, and a highly-relevant older memory could not
          be returned however well it matched.

        Ranking here by ``(overlap, timestamp)`` is not a new ordering: it
        is exactly what ``search_by_tags`` computes per store, applied
        across both for the first time.  The overlap is recomputed rather
        than propagated because it is a set intersection on data already
        in hand, and threading a score through the return type would buy
        nothing.
        """
        found = self._search_stores_for(tags, scope, maturity)

        # Scope filter BEFORE counting: `matched` must mean "what an
        # unlimited call would have returned", and an unlimited call
        # filters too.
        if scope is not None:
            found = [m for m in found if m.scope == scope]

        unique: Dict[str, Memory] = {}
        for mem in found:
            unique.setdefault(mem.id, mem)

        wanted = set(tags)
        return sorted(
            unique.values(),
            key=lambda m: (len(set(m.tags) & wanted), m.timestamp),
            reverse=True,
        )

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
            On success, a dict carrying ``status="success"``, the
            ``memories`` list with lifecycle metadata, and **four fields
            about completeness** (#982):

            - ``count`` — how many memories this payload carries
              (unchanged: it has always been the RETURNED count);
            - ``matched`` — how many matched in total, before ``limit``.
              On the ``ids`` path nothing is truncated, so it equals
              ``count``;
            - ``truncated`` — whether ``matched > count``;
            - ``message`` — the same fact in prose, always present.

            ``matched`` exists because ``count`` alone cannot distinguish
            "3 is all there was" from "3 is the maximum you asked for",
            and an agent reading the second as the first states a subset
            of what it knows with full confidence.  A ``no_results``
            result carries ``count``/``matched`` of 0 and
            ``truncated=False``, so a caller can read the same four fields
            on every non-error outcome instead of branching on status
            first.
        """
        if not self._storage:
            return {
                "status": "error",
                "error": "Memory plugin not initialized"
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
                    "count": 0,
                    "matched": 0,
                    "truncated": False,
                    "message": f"No memories found for ids: {ids}"
                }
            # Preserve requested order so the agent receives results in
            # the order it asked for.
            order = {mid: i for i, mid in enumerate(ids)}
            memories.sort(key=lambda m: order.get(m.id, len(order)))
            # Skip the tags/maturity/scope filtering and limit truncation
            # — the agent asked for these specific memories explicitly.
            # So nothing was dropped: everything found IS everything
            # returned, and `matched` says so rather than being omitted.
            matched = len(memories)
            truncated = False

        # ── Tag search path (legacy) ────────────────────────────────
        else:
            self._trace(f"retrieve_memories: tags={tags}, limit={limit}, scope={scope}, maturity={maturity}")
            # EVERY match, from both stores, ranked once.  `matched` is
            # meaningless if either store truncated on the way here (#982),
            # and so is the page: see _search_both_stores.
            memories = self._search_both_stores(tags, scope, maturity)
            matched = len(memories)
            truncated = matched > limit
            memories = memories[:limit]

            if not memories:
                return {
                    "status": "no_results",
                    "count": 0,
                    "matched": 0,
                    "truncated": False,
                    "message": f"No memories found for tags: {tags}"
                }

        # Art. 15(4) (#1123): the CURATION GATE, applied to everything
        # about to be handed back -- both the explicit-ids path and the
        # tag search, because a gate on one of two paths is a gate the
        # model routes around by asking for ids.
        memories, withheld = self._apply_curation_gate(memories)
        all_withheld = self._all_withheld_result(memories, withheld, matched)
        if all_withheld is not None:
            return all_withheld

        # Record usage BY ID, not by writing the object back -- see
        # ``_record_retrieval_usage`` for why that distinction cost 10 of
        # 32 live curator decisions when it was the other way round.
        self._record_retrieval_usage(memories)

        # Compute summary stats for telemetry
        maturities_retrieved = list({m.maturity for m in memories})
        scopes_retrieved = list({m.scope for m in memories})
        avg_confidence = sum(m.confidence for m in memories) / len(memories)

        return {
            "status": "success",
            "count": len(memories),
            "matched": matched,
            "truncated": truncated,
            "message": self._retrieval_message(len(memories), matched, limit)
            + self._withheld_note(withheld),
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
                "jaato.memory.count_matched": matched,
                "jaato.memory.truncated": truncated,
                "jaato.memory.maturities_retrieved": maturities_retrieved,
                "jaato.memory.scopes_retrieved": scopes_retrieved,
                "jaato.memory.avg_confidence": round(avg_confidence, 3),
                "jaato.memory.withheld_uncurated": withheld,
            },
        }

    def _record_retrieval_usage(self, memories: List[Memory]) -> None:
        """Bump usage for everything a retrieval returned, BY ID.

        Extracted from ``_execute_retrieve`` so that function stays at
        its complexity baseline when the #1123 curation gate lands beside
        it.  The behaviour and every reason for it are unchanged:

        This used to be ``self._storage.update(mem)`` with the full
        retrieved object -- which routed on the maturity the memory had AT
        RETRIEVAL TIME.  Under parallel tool execution a curator decision
        landing between the read and this write-back was silently undone:
        a validation reverted (stale-raw upserted over it), a dismissal
        resurrected (stale object re-added via the "not anywhere yet"
        branch).  10 of 32 live decisions lost, and a re-decide livelock.

        It also wrote every memory through the PROJECT store regardless of
        which store it came from, so a global-store memory would have been
        copied into the project raw queue.  ``record_usage`` no-ops on a
        store that does not hold the id, so offering it to both is exact.

        The in-hand copies are bumped too, because the response reflects
        THIS read -- they are display only and never written back, so they
        cannot carry staleness anywhere.
        """
        for mem in memories:
            mem.usage_count += 1
            mem.last_accessed = datetime.now().isoformat()
            if self._storage:
                self._storage.record_usage(mem.id)
            if self._global_storage:
                self._global_storage.record_usage(mem.id)
        self._note_retrieved(memories)

    @staticmethod
    def _withheld_note(withheld: int) -> str:
        """The clause naming memories the curation gate held back.

        Empty when none were, so a deployment that does not set
        ``require_curation`` reads exactly the message it always did.
        """
        if not withheld:
            return ""
        return (f"  {withheld} further match(es) were withheld as uncurated "
                f"(require_curation).")

    def _all_withheld_result(
        self, kept: List[Memory], withheld: int, matched: int,
    ) -> Optional[Dict[str, Any]]:
        """The result for a retrieval the gate emptied, or ``None``.

        Its own method so ``_execute_retrieve`` stays at its complexity
        baseline -- the ratchet is a ratchet, and new logic goes in a
        helper rather than into a raised number.

        ``no_results`` rather than an error: nothing went wrong, the
        deployment's policy applied.  What the message must not do is
        leave the model believing the store is empty, so it says how many
        matched, that they are STORED, and what makes them retrievable.
        """
        if kept or not withheld:
            return None
        return {
            "status": "no_results",
            "count": 0,
            "matched": matched,
            "truncated": False,
            "withheld_uncurated": withheld,
            "message": (
                f"{withheld} memory/memories matched and were WITHHELD: "
                f"this deployment sets require_curation, and none of them "
                f"carries a curator's mark. They are stored, not lost -- "
                f"a curator promoting them makes them retrievable."),
        }

    def _apply_curation_gate(
        self, memories: List[Memory],
    ) -> Tuple[List[Memory], int]:
        """Withhold uncurated memories when the deployment requires curation.

        Article 15(4) is about systems that "continue to learn after being
        placed on the market": feedback loops must be addressed so that
        possibly biased outputs do not feed back as inputs without
        mitigation.  jaato's learning loop is this plugin -- the model
        writes memories during a session and they are re-injected into
        later sessions.

        The plugin's own docstring has always described the raw -> curated
        lifecycle ("The School": agents store raw memories, an advisor
        curates them), and nothing in a profile could REQUIRE it.  An
        uncurated memory was re-injected exactly as a curated one, so
        "this deployment's learning loop is reviewed" was not a statement
        ``validate`` could check or a dossier could print.

        Three properties:

        * **Off by default.**  ``require_curation: false`` is
          byte-identical to the behaviour before #1123, and the empty
          ``withheld`` count costs a caller nothing.
        * **Withheld, never deleted.**  The memory is stored and a
          curator promoting it makes it retrievable; the result SAYS how
          many were held back, because a silently shorter list is a model
          reasoning from a subset it believes is the whole.
        * **It gates RETRIEVAL, not storage.**  Writing continues, which
          is what leaves the curator something to curate.

        Returns:
            ``(kept, withheld_count)``.
        """
        # ``getattr``: this method is reached by ``MemoryPlugin.__new__``
        # doubles that never ran ``__init__``, the pattern this plugin's
        # tests already use.  The default is the pre-#1123 behaviour --
        # gate off -- which is the safe direction for an object that
        # predates the attribute, the #881 rule.
        if not getattr(self, "_require_curation", False):
            return memories, 0
        kept = [m for m in memories if m.is_curated]
        return kept, len(memories) - len(kept)

    def _execute_list_tags(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_memory_tags tool.

        Args:
            args: Tool arguments (none)

        Returns:
            Result dict describing BOTH stores, because they answer
            different questions and only one of them is tag-searchable:

            * ``tags`` / ``count`` / ``memory_count`` — the CURATED store,
              which is what the indexer builds from and what tag search
              reads.
            * ``pending_curation`` — how many memories sit in the RAW
              queue awaiting curator review.  Unreachable by tag search
              (``retrieve_memories`` with ``maturity='raw'`` is the way
              in), so it is reported here or a curator cannot learn its
              queue is non-empty.
        """
        self._trace("list_memory_tags")
        if not self._indexer:
            return {
                "status": "error",
                "error": "Memory plugin not initialized"
            }

        tags = self._indexer.get_all_tags()
        memory_count = self._indexer.get_memory_count()

        # Maturity breakdown.  Counts the RAW QUEUE as well as the curated
        # store — unlike ``memory_count``, which comes from the indexer and
        # so describes the curated store only (the same store ``tags``
        # describes; the two must keep agreeing).
        maturity_counts = {}
        if self._storage:
            maturity_counts = self._storage.count_by_maturity()
        pending_curation = maturity_counts.get(MATURITY_RAW, 0)

        # THE RAW QUEUE MUST BE VISIBLE TO THE MODEL, NOT ONLY TO TELEMETRY.
        #
        # This handler used to answer "Found 0 memories" while holding
        # "raw: 12" in the same dict, because the raw count went to
        # ``_telemetry`` (which the model never sees) and nothing else said
        # the queue existed.  A curator agent asked what was in the store,
        # was told nothing was, and correctly concluded there was nothing to
        # curate — with twelve raw memories on disk.  Two curator sessions
        # reasoned soundly from that false premise; one hedged that "the
        # memory write hasn't landed yet".
        #
        # ``memory_count`` keeps its meaning (curated only) rather than
        # being widened to the true total: it is the count of the store
        # ``tags`` indexes, and a number that silently changed denominator
        # would break the callers that pair them.  The queue gets its own
        # name instead, and the message names the retrieval that reaches it,
        # since tag search cannot (``search_by_tags`` reads the curated
        # store only — see ``_execute_retrieve``).
        message = (
            f"Found {memory_count} curated memories "
            f"with {len(tags)} unique tags"
        )
        if pending_curation:
            message += (
                f"; {pending_curation} raw awaiting curation "
                f"(retrieve_memories with maturity='raw')"
            )

        return {
            "status": "success",
            "tags": sorted(tags),
            "count": len(tags),
            "memory_count": memory_count,
            "pending_curation": pending_curation,
            "message": message,
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
            return {"status": "error", "error": "'id' is required"}

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
            return {"status": "error", "error": f"Memory '{memory_id}' not found"}

        # Apply updates
        if "maturity" in args:
            new_maturity = args["maturity"]
            if new_maturity in VALID_MATURITIES:
                memory.maturity = new_maturity
                self._stamp_curation(memory, new_maturity)
            else:
                return {"status": "error", "error": f"Invalid maturity: {new_maturity}"}

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

    def _stamp_curation(
        self,
        memory: Any,
        maturity: str,
        curator: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record WHO approved a memory, at the moment of approval (#1123).

        ``curated_by`` is the second of the two provenance fields, and it
        answers a different question from ``generated_by``: who WROTE
        this, and who APPROVED it.  Approval happens exactly here --
        ``update_memory`` promoting a memory into a curated maturity is
        the promotion path the plugin's own instructions and
        ``validate``'s ``require_curation_without_curator`` remedy both
        name -- so this is where the stamp belongs.

        Without it nothing in the tree ever wrote the field, so
        ``Memory.is_curated`` was ``False`` for every memory that would
        ever exist and ``require_curation: true`` withheld the entire
        corpus, permanently, with the model told that a curator promoting
        them would make them retrievable.

        **A withdrawn approval is withdrawn.**  Demoting out of a curated
        maturity CLEARS the stamp rather than leaving it: a dismissed
        memory still carrying ``curated_by`` reads as approved to
        ``is_curated``, which is the gate deciding what the model sees.

        The stamp is the CURATOR's provenance -- the session running the
        promotion -- never the author's, which ``generated_by`` already
        holds.  ``None`` when no session is in context: a promotion whose
        approver cannot be established still promotes, and records the
        approval without claiming an approver it did not observe.

        ``curator`` (#1232) is an approver the CALLER observed -- the memory
        rail's verb passes the person the daemon's transport authenticated
        (``{"kind": "human", "via": "memory.update", "user": ...}``).  It
        replaces the model provenance rather than joining it: a rail action
        runs on an RPC thread with no session in context, and a person
        clicking Approve is not the model.  The same helper either way, so
        there is still exactly one writer of the stamp.
        """
        if maturity not in CURATED_MATURITIES:
            memory.curated_by = None
            return
        stamp: Dict[str, Any] = {"at": datetime.now(timezone.utc).isoformat()}
        if curator:
            stamp.update(curator)
            memory.curated_by = stamp
            return
        provenance = self._model_provenance()
        if provenance:
            stamp.update(provenance)
        agent = self._agent_name
        if agent:
            stamp.setdefault("agent", agent)
        memory.curated_by = stamp

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
            return {"status": "error", "error": "'id' is required"}

        # Try workspace store first, then global.  The deleted tier's index
        # is rebuilt (#1232): a curated memory lives in it, and one left
        # there would keep surfacing as an enrichment hint for an id the
        # store no longer holds.
        deleted = False
        for _tier, storage, indexer in self._tiers():
            if storage.delete(memory_id):
                deleted = True
                if indexer is not None:
                    indexer.clear()
                    indexer.build_index(storage.load_curated())
                break

        if deleted:
            self._trace(f"delete_memory: id={memory_id}")
            return {"status": "success", "message": f"Memory '{memory_id}' deleted"}
        else:
            return {"status": "error", "error": f"Memory '{memory_id}' not found"}

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
                # Through the same helper update_memory uses (#1123): a
                # human curator promoting a memory in the EDITOR is the
                # approval `curated_by` exists to record, and a second
                # writer of `maturity` that skipped the stamp would
                # reproduce the defect one command over -- a `validated`
                # record with no curator, which `require_curation` then
                # withholds.  The schema validator above has already
                # refused a maturity outside VALID_MATURITIES.
                memory.maturity = parsed["maturity"]
                self._stamp_curation(memory, parsed["maturity"])
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
            ("    - Memories are stored in .jaato/memories/ (raw/ + curated.jsonl)", "dim"),
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
