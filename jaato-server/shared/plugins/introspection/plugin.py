"""Tool Discovery & Introspection Plugin implementation.

This plugin provides tools for agents to discover and query available tools
at runtime, enabling dynamic tool selection and self-documentation.

The plugin supports deferred tool loading for token economy:
- Only "core" tools are loaded into initial context
- "discoverable" tools can be queried via list_tools and get_tool_schemas
- Models request schemas on-demand, reducing initial context overhead
"""

import threading
from typing import Any, Callable, Dict, List, Optional, Set

from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    TRAIT_REPLAY_SAFE,
    DISCOVERABILITY_EAGER,
    DISCOVERABILITY_DEFERRED,
)
from shared.plugins.runner_forwarding import RunnerForwardingMixin
from shared.tool_id_map import name_to_id
from ..streaming import StreamingCapable

# Thread-local storage for session reference per agent context
# This prevents subagents from overwriting parent's session reference
_thread_local = threading.local()


class IntrospectionPlugin(RunnerForwardingMixin):
    """Plugin that provides tool discovery and introspection capabilities.

    This plugin exposes tools for the LLM to:
    - list_tools: Discover available tools with optional category filtering
    - get_tool_schemas: Get full schemas for specific tools (enables on-demand loading)

    The plugin receives access to the PluginRegistry via set_plugin_registry(),
    which is called automatically by the registry during expose_tool().

    Deferred Loading:
        Tools have a "discoverability" attribute:
        - "core": Always in initial context
        - "discoverable": Schema provided on-demand via get_tool_schemas

        This plugin's tools are marked as "core" since they're needed for discovery.
        The _accessed_tools set tracks which tools the model has requested schemas for,
        useful for telemetry and GC decisions.
    """

    def __init__(self):
        self._initialized = False
        self._registry = None  # Set via set_plugin_registry()
        self._accessed_tools: Set[str] = set()  # Track tools model has requested

    @property
    def name(self) -> str:
        return "introspection"

    @property
    def _session(self):
        """Get the session for the current thread context.

        Uses thread-local storage so each agent (main or subagent) gets
        its own session reference, preventing subagents from overwriting
        the parent's session.
        """
        return getattr(_thread_local, 'session', None)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the introspection plugin.

        Args:
            config: Optional configuration dict (currently unused).
        """
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the introspection plugin."""
        self._initialized = False

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset — NO-OP for this plugin.

        Phase 1 hotfix (server 0.6.148+): added to satisfy the
        ``ToolPlugin`` / ``EnrichmentPlugin`` protocol's runtime
        ``isinstance`` check.  Per Daniel's litmus test (see
        ``docs/design/runner-cascade-sharing.md`` §4.3), this
        plugin holds no per-session state that the next cascade
        session would benefit from having cleared.  Override in
        future PRs if the litmus test changes.
        """
        pass


    def set_plugin_registry(self, registry) -> None:
        """Receive the plugin registry for tool discovery.

        This is called automatically by the PluginRegistry during expose_tool()
        when it detects this method exists on the plugin.

        Args:
            registry: The PluginRegistry instance.
        """
        self._registry = registry
        registry.register_category("system", "Shell commands, environment, system operations")

    def set_session(self, session) -> None:
        """Receive the session for tool activation.

        This is called automatically by the plugin wiring system. When tools
        are discovered via get_tool_schemas, they need to be activated in
        the session so the provider can use them.

        Stores in thread-local storage so each agent context gets its own session.

        Args:
            session: The JaatoSession instance.
        """
        _thread_local.session = session

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for introspection tools.

        Both tools are marked as 'core' discoverability since they're required
        for the deferred tool loading mechanism to work.
        """
        return [
            ToolSchema(
                name="list_tools",
                description="Discover available tools. "
                           "Without arguments: returns available categories with tool counts and IDs. "
                           "With category_id: returns tools in that category.",
                parameters={
                    "type": "object",
                    "properties": {
                        "category_id": {
                            "type": "string",
                            "description": (
                                "The 'id' field (NOT the 'name') from the category summary. "
                                "Must be obtained from a prior list_tools() call. "
                                "If omitted, returns the category summary."
                            ),
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "If true, include full descriptions. Default is false.",
                            "default": False,
                        },
                    },
                    "required": []
                },
                category="system",
                discoverability=DISCOVERABILITY_EAGER,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
            ToolSchema(
                name="get_tool_schemas",
                description="ENABLE tools so you can call them, and get their "
                           "schemas.  This is the activation step: a discoverable "
                           "tool is NOT callable until you pass its id here -- "
                           "list_tools only shows you what exists.  Returns full "
                           "parameter specifications, types, required/optional "
                           "flags and descriptions, plus an 'activated' list "
                           "naming the tools now available to call.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The 'id' field (NOT the 'name') of tools. Must be obtained from a prior list_tools() call."
                        }
                    },
                    "required": ["tool_ids"]
                },
                category="system",
                discoverability=DISCOVERABILITY_EAGER,
                traits=frozenset({TRAIT_REPLAY_SAFE}),
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executors for introspection tools.

        Phase 3 §3.10 wave 4: forwards via runner-RPC when a runner
        is attached; falls through to in-process otherwise.
        """
        return self.wrap_executors_for_runner_forwarding({
            "list_tools": self._execute_list_tools,
            "get_tool_schemas": self._execute_get_tool_schemas,
        })

    def get_accessed_tools(self) -> Set[str]:
        """Get the set of tools the model has requested schemas for.

        This is useful for:
        - Telemetry: Understanding which tools the model uses
        - GC decisions: Preserving schemas the model has accessed

        Returns:
            Set of tool names that have been accessed via get_tool_schemas.
        """
        return self._accessed_tools.copy()

    def clear_accessed_tools(self) -> None:
        """Clear the accessed tools tracking.

        Call this when resetting the session or for fresh tracking.
        """
        self._accessed_tools.clear()

    def evaluate_gc_policy(self, entry: 'SourceEntry', budget: 'InstructionBudget') -> 'GCPolicy':
        """Evaluate GC policy for a discovered tool schema entry.

        This method is called by the GC system to determine whether a tool schema
        with CONDITIONAL policy should be kept (LOCKED) or removed (EPHEMERAL).

        Logic:
        - If the tool has been successfully executed at least once → LOCKED
        - Otherwise → EPHEMERAL (can be GC'd and re-discovered if needed later)

        Args:
            entry: The instruction budget entry for the tool schema
            budget: The full instruction budget (provides session_id)

        Returns:
            GCPolicy.LOCKED if tool has been used successfully, EPHEMERAL otherwise
        """
        from ..instruction_budget import GCPolicy

        # Extract tool name from entry metadata
        tool_name = entry.metadata.get("tool_name") if entry.metadata else None
        if not tool_name:
            return GCPolicy.EPHEMERAL

        # Get session ID from budget
        session_id = budget.session_id
        if not session_id:
            return GCPolicy.EPHEMERAL

        # Query reliability plugin for usage statistics
        if self._registry:
            reliability = self._registry.get_plugin('reliability')
            if reliability and hasattr(reliability, 'has_successful_execution'):
                try:
                    if reliability.has_successful_execution(tool_name, session_id):
                        return GCPolicy.LOCKED
                except Exception:
                    # If query fails, default to ephemeral
                    pass

        return GCPolicy.EPHEMERAL

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the introspection plugin.

        2026-05-15 fix: reframe the explore-tools guidance from
        unconditional (``"ALWAYS explore"``) to conditional
        (``"WHEN REQUIRED, explore"``).

        **Why the wording change matters.**  The prior text was an
        absolute imperative.  Personas that need to scope the
        agent to a narrow tool set (e.g. a typed-completion stage
        that says "only call signal_completion; do not call
        list_tools / get_tool_schemas") had to FIGHT the framework
        instruction.  Same model, same persona, two contradictory
        directives — the model has to weigh them, and the
        framework's strong "ALWAYS" word often won.

        Reframed as conditional ("WHEN REQUIRED"), the framework
        instruction becomes a fallback for unclear cases: explore
        WHEN your current information is insufficient.  The
        persona's explicit "your only tool is X" statement
        satisfies the qualifier — exploration isn't required, the
        agent doesn't explore.  Both instructions coexist without
        contradiction.

        See ``feedback_kernel_scoping_beats_persona_prose``:
        framework-level guidance should not unconditionally
        contradict session-level contracts.  Personas already
        narrow the tool surface via the persona text; the
        framework just needs to not override that.

        Profiles that want NO framework prompt at all retain the
        ``suppress_base_instructions: true`` flag — the
        documented escape hatch.  This change is for the typical
        case where the persona is the authority on tool scoping
        and the framework defers.

        2026-07-26 addition: an anti-fabrication guardrail
        ("only call tools in your list by their exact ids; NEVER
        invent a ``t_<name>`` id; discover unseen capabilities via
        ``list_tools`` instead of guessing").  It rides INSIDE the
        same deferred-tools gate on purpose — the guidance to
        discover-don't-guess is only coherent when discovery tools
        are actually on the wire.  When the gate suppresses this
        block (nothing deferred), every real tool is already loaded
        with its real id, so there is nothing to discover and the
        prompt must not nudge toward absent ``list_tools`` (the
        ex08 loop).  Motivated by a live leak where a small exec
        model read a human tool name (``delete_memory``) from
        another plugin's instructions and fabricated
        ``t_delete_memory`` for the deferred (unloaded) tool.
        """
        # Align with the TOOL gate: when the session dropped introspection's
        # tools because nothing is deferred to discover (jaato_session.configure
        # -> _introspection_guidance_suppressed), suppress this discovery
        # guidance too. Otherwise the prompt nudges the model to call list_tools
        # / get_tool_schemas that aren't on its wire -> it invents them, hits
        # no-executor, and loops (the ex08 0-tool subagent hang). Tool-gate and
        # instruction-gate must stay aligned.
        session = self._session
        if session is not None and getattr(
            session, "_introspection_guidance_suppressed", False
        ):
            return None
        return (
            "You have a DYNAMIC tool system with discoverable capabilities.\n\n"
            "CAPABILITY DISCOVERY:\n"
            "When your current information is insufficient to act, explore "
            "available tools before concluding 'I cannot do X'.  Skip discovery "
            "when the persona has already named the tool(s) to call.\n"
            "Only call tools that appear in your available tool list, using "
            "their exact ids.  NEVER invent, guess, or construct a tool id or "
            "name (e.g. do not turn a human name you saw in prose into "
            "`t_<name>`) — the opaque ids come only from your tool list and "
            "from `get_tool_schemas`.  If you need a capability you don't "
            "currently see, find it with `list_tools(category_id=...)`, then "
            "ENABLE it with `get_tool_schemas(tool_ids=[...])`, rather than "
            "guessing.\n\n"
            "TOOL DISCOVERY WORKFLOW (when required):\n"
            "1. `list_tools()` - See all categories with IDs and tool counts\n"
            "2. `list_tools(category_id='...')` - See tools in a specific category.\n"
            "   Listing a tool does NOT make it callable; entries you cannot yet\n"
            "   call are marked `available: false`.\n"
            "3. `get_tool_schemas(tool_ids=[...])` - ENABLE those tools.  This is\n"
            "   the step that makes a discovered tool callable: it returns an\n"
            "   `activated` list alongside the schemas.  Skipping it and trying to\n"
            "   reach the capability another way (a shell, a notebook `tools`\n"
            "   bridge) is never necessary and never correct.\n"
            "4. Call the tools directly, by the real names in `activated`\n\n"
            "CATEGORY QUICK REFERENCE:\n"
            "- coordination: Task tracking, TODO, DELEGATE work, SUBAGENTS, parallel execution\n"
            "- filesystem: Read, write, search files\n"
            "- code: Analysis, editing, LSP diagnostics\n"
            "- web: Fetch URLs, web search\n"
            "- system: Shell commands, environment\n"
            "- communication: Ask user questions\n\n"
            "STREAMING TOOLS:\n"
            "Tools with `streaming: true` support incremental results. To use streaming:\n"
            "- Call `<tool_name>:stream` instead of `<tool_name>` (e.g., `grep_content:stream`)\n"
            "- You'll receive a stream_id and initial results immediately\n"
            "- More results arrive automatically as the tool finds them\n"
            "- Call `dismiss_stream(stream_id)` when you have enough results"
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return introspection tools as auto-approved (read-only, no security implications)."""
        return ["list_tools", "get_tool_schemas"]

    def get_user_commands(self) -> List:
        """Return user commands (none for this plugin)."""
        return []

    def _get_session_allowed_schemas(self) -> List[ToolSchema]:
        """Get tool schemas filtered by the current session's allowed plugins.

        When a session is created with a profile that specifies an explicit
        plugin list (``_tool_plugins``), only tools from those plugins (plus
        ``introspection``) should be visible.  If no session exists or the
        session has no plugin restriction (``_tool_plugins is None``), all
        globally-exposed schemas are returned.

        Returns:
            Filtered list of ToolSchema objects visible to the current session.
        """
        all_schemas = self._registry.get_exposed_tool_schemas()

        session = self._session
        if session is None:
            return all_schemas

        allowed_plugins = getattr(session, '_tool_plugins', None)
        if allowed_plugins is None:
            return all_schemas

        # Build the effective set: profile plugins + introspection (essential)
        allowed_set = set(allowed_plugins)
        allowed_set.add("introspection")

        # Filter schemas to those whose owning plugin is in the allowed set
        filtered = []
        for schema in all_schemas:
            plugin = self._registry.get_plugin_for_tool(schema.name)
            if plugin is None:
                # Core tools registered directly on the registry have no
                # owning plugin — keep them (e.g. dismiss_stream).
                filtered.append(schema)
                continue
            if plugin.name in allowed_set:
                filtered.append(schema)

        return filtered

    def _unknown_category_response(
        self,
        category_id: Any,
        all_schemas: List[Any],
        category_hints: Dict[str, str],
        preloaded_tools_hint: List[str],
    ) -> Dict[str, Any]:
        """Answer an unrecognised ``category_id`` with the ids that ARE valid.

        A wrong ``category_id`` is not a typo -- it is the expected first
        move.  Ids are hashed (``c_cbf61858`` for ``filesystem``), so no
        amount of reasoning gets a model to one; the only way to hold a
        valid id is to have called ``list_tools()`` with no arguments
        already.  A model that has not is certain to guess, and observed
        sessions guess a plausible category *name* on turn one.

        So this returns the full category summary -- the same payload the
        no-argument call returns -- alongside the error, turning the dead
        end into a self-correcting call.  When the guess matches a visible
        category's name, ``did_you_mean`` names that category's id outright,
        because "you passed the name, here is the id" is the single most
        likely correction.

        Args:
            category_id: The unrecognised value, echoed back in the error.
            all_schemas: Tool schemas visible to THIS session.
            category_hints: Registry-registered category descriptions.
            preloaded_tools_hint: Names of already-loaded tools.

        Returns:
            The category summary payload plus an ``error`` string and,
            when applicable, a ``did_you_mean`` correction.
        """
        response = self._build_category_summary(
            all_schemas, category_hints, preloaded_tools_hint
        )
        response["error"] = (
            f"Unknown category_id '{category_id}'. Category ids are hashed "
            f"and cannot be guessed -- every valid id is listed under "
            f"'categories' below. Retry with one of them."
        )
        # Guess matched a category NAME rather than its id: name the id.
        # Drawn from the FILTERED summary, so a category this session
        # cannot see is not disclosed by the correction.
        guess = str(category_id).strip().lower()
        for entry in response.get("categories", []):
            if str(entry.get("name", "")).lower() == guess:
                response["did_you_mean"] = {
                    "name": entry["name"],
                    "category_id": entry["id"],
                    "retry": f"list_tools(category_id='{entry['id']}')",
                }
                break
        response["_telemetry"]["jaato.introspection.operation"] = (
            "list_tools.unknown_category"
        )
        return response

    def _category_summary_entry(
        self,
        cat: str,
        session_counts: Dict[str, int],
        global_counts: Dict[str, int],
        category_hints: Dict[str, str],
        category_plugins: Dict[str, Set[str]],
        allowed_plugins: Optional[Set[str]],
    ) -> Optional[Dict[str, Any]]:
        """Build one category's summary entry, or ``None`` to hide it.

        Hiding is a disclosure decision, not a formatting one, which is
        why it lives with the entry rather than at the call site: a
        category whose tools all belong to plugins outside the session's
        profile names a capability the model cannot invoke, and listing it
        reliably primes hallucinated calls ("MCP server failed", etc.).
        :meth:`_unknown_category_response` replays this same payload, so
        whatever is withheld here stays withheld on the recovery path too.

        Args:
            cat: Category name.
            session_counts: Tools per category visible to THIS session.
            global_counts: Tools per category across all exposed plugins.
            category_hints: Registry-registered category descriptions.
            category_plugins: Plugins contributing to each category.
            allowed_plugins: The session's plugin set, or ``None`` when the
                caller has an unrestricted view.

        Returns:
            An ``{id, name, tool_count, description}`` dict -- plus an
            ``availability`` note when the session sees only some of the
            category's tools -- or ``None`` when the category must not be
            disclosed to this session.
        """
        available = session_counts.get(cat, 0)
        total = global_counts.get(cat, 0)

        # Skip categories the session has zero access to.  When a profile's
        # plugin list excludes a plugin, that plugin's registered category
        # (e.g. ``"MCP"`` from ``mcp.plugin.set_plugin_registry``) and any
        # category description still live in the registry — but listing them
        # in introspection output leaks plugin/protocol names the model
        # cannot actually invoke and primes hallucinations ("MCP server
        # failed", etc.).  Two exceptions:
        #   1. No allowed-plugin filter is in effect — caller has
        #      unrestricted view (e.g. an admin-tier session enumerating
        #      the daemon).
        #   2. The category is intrinsically empty for everyone
        #      (``total == 0``) AND has no description hint — nothing to leak.
        if available == 0 and allowed_plugins is not None:
            if total > 0:
                # Session-invisible plugin contributes tools to this
                # category — hide entirely.  Don't surface the plugin name
                # in an "enable X" hint either.
                return None
            # total == 0: only a category description hint references this
            # category.  If a hint is registered AND the category isn't part
            # of the session's plugin set, hide it too — that's exactly the
            # MCP leak path (registered description with no tools).
            if cat in category_hints:
                return None

        entry: Dict[str, Any] = {
            "id": name_to_id(cat, prefix="c"),
            "name": cat,
            "tool_count": available,
            "description": category_hints.get(cat, ""),
        }

        # Annotate partial availability only when the session already sees
        # SOME tools in this category — the all-zero case is filtered above.
        if available == 0 and total == 0:
            entry["availability"] = "no tools loaded"
        elif available < total and allowed_plugins is not None:
            missing = sorted(
                category_plugins.get(cat, set()) - allowed_plugins
            )
            if missing:
                entry["availability"] = (
                    f"partial ({available}/{total} tools — "
                    f"{', '.join(missing)} not enabled for this profile)"
                )
            else:
                entry["availability"] = (
                    f"partial ({available}/{total} tools)"
                )
        # else: fully available — no extra annotation needed

        return entry

    def _build_category_summary(
        self,
        all_schemas: List[Any],
        category_hints: Dict[str, str],
        preloaded_tools_hint: List[str],
    ) -> Dict[str, Any]:
        """Build the no-argument ``list_tools`` payload: the category summary.

        This is the only place a caller can learn a valid ``category_id``:
        ids are hashed (``name_to_id(cat, prefix="c")``) precisely so they
        cannot be derived from a category's name.  It is therefore reached
        from two directions -- the deliberate no-argument call, and
        :meth:`_unknown_category_response`, which replays this same payload
        so a wrong guess is recoverable in one round-trip instead of two.

        Categories the session has no access to are filtered out here, so
        the recovery path inherits that filtering and cannot leak the id of
        a category the caller could not use anyway.

        Args:
            all_schemas: Tool schemas visible to THIS session (profile-filtered).
            category_hints: Registry-registered category descriptions.
            preloaded_tools_hint: Names of tools already in the session's
                initial schema, surfaced so their absence from the
                categories is not read as unavailability.

        Returns:
            The category summary payload, with ``categories`` carrying an
            ``{id, name, tool_count, description}`` entry per visible category.
        """
        # Count tools visible to THIS session (profile-filtered)
        session_counts: Dict[str, int] = {}
        for schema in all_schemas:
            cat = schema.category or "uncategorized"
            session_counts[cat] = session_counts.get(cat, 0) + 1

        # Count ALL tools across ALL exposed plugins (unfiltered)
        # and track which plugins contribute to each category.
        global_counts: Dict[str, int] = {}
        category_plugins: Dict[str, Set[str]] = {}
        for schema in self._registry.get_exposed_tool_schemas():
            cat = schema.category or "uncategorized"
            global_counts[cat] = global_counts.get(cat, 0) + 1
            plugin = self._registry.get_plugin_for_tool(schema.name)
            if plugin:
                category_plugins.setdefault(cat, set()).add(plugin.name)

        # Determine which plugins the session has enabled
        session = self._session
        allowed_plugins: Optional[Set[str]] = None
        if session:
            raw = getattr(session, '_tool_plugins', None)
            if raw is not None:
                allowed_plugins = set(raw)
                allowed_plugins.add("introspection")

        # Merge all known categories: from schemas + registered descriptions.
        all_categories = dict.fromkeys(
            sorted(
                set(session_counts.keys())
                | set(global_counts.keys())
                | set(category_hints.keys())
            )
        )

        categories_list = []
        for cat in all_categories:
            entry = self._category_summary_entry(
                cat,
                session_counts,
                global_counts,
                category_hints,
                category_plugins,
                allowed_plugins,
            )
            if entry is None:
                continue
            categories_list.append(entry)

        response: Dict[str, Any] = {
            "categories": categories_list,
            "total_tools": len(all_schemas),
            "hint": "Call list_tools(category_id='<id>') to see tools in a specific category.",
            "_telemetry": {
                "jaato.introspection.operation": "list_tools",
                "jaato.introspection.total_tools": len(all_schemas),
            },
        }
        if preloaded_tools_hint:
            response["preloaded_tools_already_in_schema"] = sorted(
                set(preloaded_tools_hint)
            )
            response["preloaded_tools_note"] = (
                "These tools are PRELOADED in your initial tool schema "
                "and intentionally do NOT appear in the categories above. "
                "list_tools returns only deferred (discoverable) tools. "
                "Call preloaded tools directly without further discovery."
            )
        return response

    def _execute_list_tools(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the list_tools tool.

        Args:
            args: Dictionary with optional 'category_id' and 'verbose' keys.

        Returns:
            - If no category_id: the category summary — every visible
              category with its hashed id and tool count.
            - If category_id resolves: the tools in that category.
            - If category_id does NOT resolve: the same category summary
              plus an ``error``, so the caller holds a valid id without a
              second round-trip.  See :meth:`_unknown_category_response`.
        """
        if not self._registry:
            return {"error": "Registry not available. Plugin not properly initialized."}

        category_id = args.get("category_id")
        verbose = args.get("verbose", False)

        # Get tool schemas filtered by session's allowed plugins
        all_schemas = self._get_session_allowed_schemas()

        # ── Soft-nudge for preloaded tools ──────────────────────────────
        # When a profile preloads a plugin (e.g. ``service_connector(preload)``
        # in policy_admin.json), that plugin's tools land in the session's
        # initial tool schema — they don't need ``list_tools`` discovery.
        # But agents (especially smaller models) habitually call
        # ``list_tools`` to "verify" a tool exists, then conclude
        # "absent from list_tools result → tool unavailable" because
        # list_tools by design returns only deferred tools.
        #
        # Surface the preloaded tools in the response as a hint so the
        # agent doesn't draw the wrong inference from absence-from-list.
        # Per memory ``project_backlog_introspection_soft_nudge_for_preloaded``.
        preloaded_tools_hint: List[str] = []
        if self._session is not None:
            preloaded_plugins = getattr(self._session, "_preloaded_plugins", None) or set()
            if preloaded_plugins and self._registry is not None:
                for tool_schema in self._registry.get_exposed_tool_schemas():
                    plugin = self._registry.get_plugin_for_tool(tool_schema.name)
                    if plugin and plugin.name in preloaded_plugins:
                        preloaded_tools_hint.append(tool_schema.name)

        # Read category descriptions from the registry — needed by both
        # the summary path and the category-detail path.
        category_hints = (
            self._registry.get_category_descriptions()
            if self._registry
            else {}
        )

        # Build the set of all known categories.
        global_cats: Set[str] = set(category_hints.keys())
        for schema in self._registry.get_exposed_tool_schemas():
            global_cats.add(schema.category or "uncategorized")

        # Build reverse lookup: category hash ID → category name.
        cat_id_to_name = {name_to_id(c, prefix="c"): c for c in global_cats}

        # Resolve category_id → category name.
        category: Optional[str] = None
        if category_id is not None:
            category = cat_id_to_name.get(str(category_id))
            if category is None:
                return self._unknown_category_response(
                    category_id,
                    all_schemas,
                    category_hints,
                    preloaded_tools_hint,
                )

        # If no category specified, return category summary only
        if not category:
            return self._build_category_summary(
                all_schemas, category_hints, preloaded_tools_hint
            )

        # Category specified - return tools in that category.
        # No string-based validation needed — category was resolved from
        # category_index which is already validated against sorted_cats.

        # Build the set of tool names visible to THIS session so we can
        # mark tools from non-profile plugins as "not available".
        session_tool_names = {s.name for s in all_schemas}

        # Iterate the GLOBAL schema set so tools from plugins not in
        # this session's profile are still listed (with an availability
        # note).  This matches the summary which also uses global counts.
        tools = []
        for schema in self._registry.get_exposed_tool_schemas():
            # Apply category filter (treat None as "uncategorized")
            schema_category = schema.category or "uncategorized"
            if schema_category != category:
                continue

            # Find which plugin provides this tool
            plugin = self._registry.get_plugin_for_tool(schema.name)
            plugin_source = plugin.name if plugin else "unknown"

            # Check if tool is enabled
            is_enabled = self._registry.is_tool_enabled(schema.name)

            # Check if tool supports streaming
            supports_streaming = False
            if plugin and isinstance(plugin, StreamingCapable):
                try:
                    supports_streaming = plugin.supports_streaming(schema.name)
                except Exception:
                    pass  # Plugin may not implement the method correctly

            # Build tool entry
            tool_entry = {
                "id": name_to_id(schema.name),
                "name": schema.name,
                "enabled": is_enabled,
                "streaming": supports_streaming,
            }

            # Mark tools the session can't call -- and say how to fix that.
            # ``get_tool_schemas`` is the ENABLING call, not a reference
            # lookup: it is the only place ``activate_discovered_tools`` runs.
            # Carrying that here, on the entry that reports the tool as
            # unavailable, puts the affordance in the data the model is
            # reading at the moment it needs it, instead of relying on the
            # system prompt having been attended to.
            if schema.name not in session_tool_names:
                tool_entry["available"] = False
                tool_entry["activate_with"] = (
                    f"get_tool_schemas(tool_ids=['{tool_entry['id']}'])"
                )

            if verbose:
                tool_entry["description"] = schema.description
            else:
                # Use first sentence or truncate for concise output
                desc = schema.description
                first_sentence_end = desc.find(". ")
                if first_sentence_end > 0:
                    tool_entry["description"] = desc[:first_sentence_end + 1]
                elif len(desc) > 100:
                    tool_entry["description"] = desc[:97] + "..."
                else:
                    tool_entry["description"] = desc

            tools.append(tool_entry)

        # Sort by ID for consistent output
        tools.sort(key=lambda t: t["id"])

        result = {
            "category": category,
            "tool_count": len(tools),
            "tools": tools,
        }

        if tools:
            result["hint"] = (
                "Call get_tool_schemas(tool_ids=['<id>']) to ENABLE tools "
                "and get their full parameter details.  Tools marked "
                "available=false are not callable until you do -- listing "
                "them here does NOT make them callable."
            )

            # Add streaming hint if any tools support streaming
            streaming_tools = [t["id"] for t in tools if t.get("streaming")]
            if streaming_tools:
                result["streaming_hint"] = (
                    f"Tools with streaming=true support incremental results. "
                    f"Call '<tool_id>:stream' (e.g., '{streaming_tools[0]}:stream') "
                    f"to receive results as they're found. Use dismiss_stream(stream_id) when done."
                )

        result['_telemetry'] = {
            'jaato.introspection.operation': 'list_tools',
            'jaato.introspection.tool_count': len(tools),
            'jaato.introspection.category': category,
        }

        return result

    def _execute_get_tool_schemas(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the get_tool_schemas tool.

        Args:
            args: Dictionary with required 'tool_ids' key (array of tool IDs).

        Returns:
            Dictionary with schemas for requested tools and tracking info.
        """
        from shared.tool_id_map import id_to_name as _id_to_name

        if not self._registry:
            return {"error": "Registry not available. Plugin not properly initialized."}

        tool_ids = args.get("tool_ids", [])
        if not tool_ids:
            return {"error": "tool_ids is required (array of tool IDs)"}

        if not isinstance(tool_ids, list):
            tool_ids = [tool_ids]

        # Get schemas filtered by session's allowed plugins
        all_schemas = self._get_session_allowed_schemas()
        schema_map = {s.name: s for s in all_schemas}

        # Build results
        schemas = []
        not_found = []

        # Collect tools that need activation (discoverable tools not yet in provider)
        tools_to_activate = []

        for tool_id in tool_ids:
            tool_name = _id_to_name(tool_id)
            if tool_name in schema_map:
                target_schema = schema_map[tool_name]

                # Track this access
                self._accessed_tools.add(tool_name)

                # Check if this is a discoverable tool that needs activation
                if getattr(target_schema, 'discoverability', DISCOVERABILITY_DEFERRED) == DISCOVERABILITY_DEFERRED:
                    tools_to_activate.append(tool_name)

                # Find plugin source
                plugin = self._registry.get_plugin_for_tool(tool_name)

                # Build detailed schema response
                schema_entry = {
                    "id": name_to_id(target_schema.name),
                    "name": target_schema.name,
                    "description": target_schema.description,
                    "enabled": self._registry.is_tool_enabled(tool_name),
                }

                if target_schema.category:
                    schema_entry["category_id"] = name_to_id(target_schema.category, prefix="c")
                    schema_entry["category"] = target_schema.category

                # Format parameters in a more readable way
                params = target_schema.parameters
                if params:
                    schema_entry["parameters"] = self._format_parameters(params)

                schemas.append(schema_entry)
            else:
                not_found.append(tool_id)

        # Build response
        result = {
            "schemas": schemas,
            "count": len(schemas),
        }

        if not_found:
            result["not_found"] = not_found
            result["hint"] = "Use list_tools() to see available tool IDs."

        # Activate discovered tools so the model can actually call them
        # This adds the tool schemas to the provider's declared tools
        if tools_to_activate and self._session:
            activated = self._session.activate_discovered_tools(tools_to_activate)
            if activated:
                result["activated"] = activated
                result["activation_note"] = "These tools are now available to call."

        result['_telemetry'] = {
            'jaato.introspection.operation': 'get_tool_schemas',
            'jaato.introspection.count': len(schemas),
        }

        return result

    def _format_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Format JSON Schema parameters into a more readable structure.

        Args:
            params: JSON Schema object for parameters.

        Returns:
            Formatted parameter documentation.
        """
        if not params or params.get("type") != "object":
            return params

        properties = params.get("properties", {})
        required = set(params.get("required", []))

        formatted_params = {}
        for name, prop in properties.items():
            param_info = {
                "type": prop.get("type", "any"),
                "required": name in required,
            }

            if "description" in prop:
                param_info["description"] = prop["description"]

            if "enum" in prop:
                param_info["allowed_values"] = prop["enum"]

            if "default" in prop:
                param_info["default"] = prop["default"]

            if "items" in prop:
                param_info["items_type"] = prop["items"].get("type", "any")

            formatted_params[name] = param_info

        return formatted_params


def create_plugin() -> IntrospectionPlugin:
    """Factory function to create the introspection plugin instance."""
    return IntrospectionPlugin()
