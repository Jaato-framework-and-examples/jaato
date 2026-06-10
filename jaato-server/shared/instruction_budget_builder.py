"""Instruction-budget builder helpers extracted from ``JaatoSession``.

These functions own the *construction* mechanics of the two-phase
instruction budget — counting tokens for a text, collecting the
instruction texts that need counting, and applying resolved counts onto
an :class:`InstructionBudget`. They take their dependencies explicitly
(runtime, provider, cache, budget) rather than reaching through a session
object, so the build logic is unit-testable in isolation.

The *orchestration* around these — lazy provider materialization, the
background ``count_tokens`` refinement thread, budget-update event
emission, and the live conversation/thinking/tool budget mutations —
stays on :class:`JaatoSession`, which calls these helpers and owns the
resulting state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .instruction_budget import (
    DEFAULT_SYSTEM_POLICIES,
    DEFAULT_TOOL_POLICIES,
    GCPolicy,
    InstructionBudget,
    InstructionSource,
    PluginToolType,
    SystemChildType,
    estimate_tokens,
)

# Trace callback: receives a single human-readable diagnostic string.
TraceFn = Callable[[str], None]


@dataclass
class TokenCountRequest:
    """A pending token-count request for a single instruction text.

    Used during two-phase instruction budget population: Phase 1 resolves
    counts from cache or estimates, Phase 2 refines cache misses via
    background ``provider.count_tokens()`` calls.
    """
    text: str
    source: InstructionSource
    child_key: str
    gc_policy: GCPolicy
    label: str
    token_count: int = 0
    is_estimate: bool = False


def count_tokens(
    text: str,
    *,
    cache: Any,
    provider: Any,
    provider_name: str,
    on_trace: TraceFn,
) -> int:
    """Count tokens for ``text`` using cache, provider, or estimate.

    Lookup order:
    1. ``InstructionTokenCache`` — instant, shared across sessions.
    2. ``provider.count_tokens()`` — accurate HTTP call; result is stored
       in the cache for future hits.
    3. ``estimate_tokens()`` — chars/4 approximation fallback.

    Args:
        text: The text to count tokens for.
        cache: The shared ``InstructionTokenCache``.
        provider: The active model provider (may be ``None`` or lack
            ``count_tokens``), in which case the estimate is used.
        provider_name: Provider key for cache lookups.
        on_trace: Diagnostic trace callback.

    Returns:
        Token count (actual or estimated).
    """
    if not text:
        return 0

    # 1. Check instruction token cache
    cached = cache.get(provider_name, text)
    if isinstance(cached, int):
        return cached

    # 2. Try provider API
    if provider and hasattr(provider, 'count_tokens'):
        try:
            result = provider.count_tokens(text)
            # Ensure we got an int (handles mocked providers returning MagicMock)
            if isinstance(result, int):
                cache.put(provider_name, text, result)
                return result
            else:
                on_trace(
                    f"count_tokens returned non-int ({type(result).__name__}), "
                    f"falling back to estimate"
                )
        except Exception as e:
            on_trace(
                f"count_tokens FAILED ({type(e).__name__}: {e}), "
                f"falling back to estimate (text length: {len(text)} chars)"
            )

    # 3. Estimate fallback
    est = estimate_tokens(text)
    on_trace(f"count_tokens: using estimate={est} (from {len(text)} chars)")
    return est


def collect_instruction_texts(
    runtime: Any,
    session_instructions: Optional[str],
    *,
    system_instruction_override: Optional[str] = None,
    suppress_base: bool = False,
    pinned_references: Optional[Dict[str, Any]] = None,
    preloaded_plugins: Optional[Set[str]] = None,
) -> Tuple[List[TokenCountRequest], Set[str]]:
    """Collect all instruction texts that need token counting.

    When ``system_instruction_override`` is set the wire-level system
    message is exactly that string (or empty), regardless of what the
    assembly pipeline produced. In that case the budget reflects what
    *actually* reaches the model — a single ``OVERRIDE`` entry (or nothing
    for an empty override) instead of walking BASE/CLIENT/FRAMEWORK/plugin
    texts that never make it to the wire. This avoids the silent lie where
    the budget showed thousands of tokens of premium instructions while
    the model received an empty system message.

    When ``system_instruction_override`` is ``None`` the assembly pipeline
    is authoritative and we walk SYSTEM children (base, client, framework,
    pinned references) plus PLUGIN children (per-plugin, per-formatter)
    into a flat list of ``TokenCountRequest`` objects.

    Args:
        runtime: The ``JaatoRuntime`` (source of base instructions,
            plugin registry, formatter pipeline).
        session_instructions: The user-provided system_instructions.
        system_instruction_override: When not ``None``, supplants the
            whole assembly — see above.
        suppress_base: Skip BASE instructions while keeping other children.
        pinned_references: Mapping of ref_id -> pinned reference (with
            ``content`` / ``ref_name`` attributes), promoted to SYSTEM.
        preloaded_plugins: Plugin names whose instructions are always
            included even under deferred tool loading.

    Returns:
        A tuple of ``(requests, deferred_plugin_names)`` — the texts to
        count, and the set of plugin names whose instructions were
        deferred (because they have no core tools yet); the caller records
        these so it can inject them when the model discovers their tools.
    """
    from .jaato_runtime import (
        _TASK_COMPLETION_INSTRUCTION,
        _PARALLEL_TOOL_GUIDANCE,
        _TURN_SUMMARY_INSTRUCTION,
        _is_parallel_tools_enabled,
    )

    pinned_references = pinned_references or {}
    preloaded_plugins = preloaded_plugins or set()
    deferred_plugins: Set[str] = set()

    # Override-aware short-circuit: the wire system message is the
    # override, period.  No need to walk the assembly pipeline (and
    # — paired with the lazy base loader — no disk I/O fires for
    # ``runtime.get_base_system_instructions()`` on this session).
    if system_instruction_override is not None:
        if system_instruction_override == "":
            return [], deferred_plugins
        return [TokenCountRequest(
            text=system_instruction_override,
            source=InstructionSource.SYSTEM,
            child_key=SystemChildType.OVERRIDE.value,
            gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.OVERRIDE],
            label="Override (system_instruction_override)",
        )], deferred_plugins

    requests: List[TokenCountRequest] = []

    # --- SYSTEM children ---

    # 1. Base instructions from .jaato/instructions/ (or legacy single
    #    file) — lazy-loaded the first time any session asks.  Skipped
    #    entirely when suppress_base is set; other SYSTEM children
    #    (CLIENT / FRAMEWORK / pinned refs) and PLUGIN children below
    #    still contribute, so the agent keeps its agent-.md content
    #    and tool instructions.
    if not suppress_base and runtime:
        base_instructions = runtime.get_base_system_instructions()
    else:
        base_instructions = None
    if base_instructions:
        requests.append(TokenCountRequest(
            text=base_instructions,
            source=InstructionSource.SYSTEM,
            child_key=SystemChildType.BASE.value,
            gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.BASE],
            label="Base Instructions",
        ))

    # 2. Client-provided session instructions (programmatic)
    if session_instructions:
        requests.append(TokenCountRequest(
            text=session_instructions,
            source=InstructionSource.SYSTEM,
            child_key=SystemChildType.CLIENT.value,
            gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.CLIENT],
            label="Client Instructions",
        ))

    # 3. Framework constants (concatenated into one request)
    framework_parts = [_TASK_COMPLETION_INSTRUCTION]
    if _is_parallel_tools_enabled():
        framework_parts.append(_PARALLEL_TOOL_GUIDANCE)
    framework_parts.append(_TURN_SUMMARY_INSTRUCTION)
    framework_text = "\n\n".join(framework_parts)
    requests.append(TokenCountRequest(
        text=framework_text,
        source=InstructionSource.SYSTEM,
        child_key=SystemChildType.FRAMEWORK.value,
        gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.FRAMEWORK],
        label="Framework",
    ))

    # 4. Pinned preselected references (content read by the model and
    #    promoted to system instruction for GC protection)
    for ref_id, pinned in pinned_references.items():
        child_key = f"{SystemChildType.SELECTED_REFERENCES.value}:{ref_id}"
        requests.append(TokenCountRequest(
            text=pinned.content,
            source=InstructionSource.SYSTEM,
            child_key=child_key,
            gc_policy=DEFAULT_SYSTEM_POLICIES[SystemChildType.SELECTED_REFERENCES],
            label=f"ref: {pinned.ref_name}",
        ))

    # --- PLUGIN children ---
    # When deferred tool loading is enabled, only include system
    # instructions from plugins that have at least one core tool.
    # Instructions for discoverable-only plugins are deferred until
    # the model activates one of their tools via get_tool_schemas.

    from .jaato_runtime import _is_deferred_tools_enabled
    deferred_enabled = _is_deferred_tools_enabled()

    if runtime.registry:
        for plugin_name in runtime.registry._exposed:
            if deferred_enabled and plugin_name not in preloaded_plugins and not runtime.registry.plugin_has_core_tools(plugin_name):
                # Remember this plugin's instructions are deferred so we
                # can inject them when the model discovers its tools.
                # Exception: preloaded plugins always include instructions.
                plugin = runtime.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'get_system_instructions'):
                    instr = plugin.get_system_instructions()
                    if instr:
                        deferred_plugins.add(plugin_name)
                continue
            plugin = runtime.registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, 'get_system_instructions'):
                instr = plugin.get_system_instructions()
                if instr:
                    requests.append(TokenCountRequest(
                        text=instr,
                        source=InstructionSource.PLUGIN,
                        child_key=plugin_name,
                        gc_policy=DEFAULT_TOOL_POLICIES[PluginToolType.CORE],
                        label=plugin_name,
                    ))

    # Formatter pipeline instructions (output rendering capabilities)
    formatter_pipeline = getattr(runtime, '_formatter_pipeline', None)
    if formatter_pipeline and hasattr(formatter_pipeline, '_formatters'):
        for formatter in formatter_pipeline._formatters:
            if hasattr(formatter, 'get_system_instructions'):
                instr = formatter.get_system_instructions()
                if instr:
                    requests.append(TokenCountRequest(
                        text=instr,
                        source=InstructionSource.PLUGIN,
                        child_key=formatter.name,
                        gc_policy=GCPolicy.PRESERVABLE,
                        label=formatter.name,
                    ))

    return requests, deferred_plugins


def apply_instruction_counts(
    budget: InstructionBudget,
    requests: List[TokenCountRequest],
    context_limit: int,
    *,
    on_trace: TraceFn,
) -> None:
    """Build budget children and parent totals from resolved token counts.

    Mutates ``budget`` in place. Called once in Phase 1 (with
    estimates/cached values) and again after Phase 2 completes (with
    accurate counts for previously-estimated entries). The caller is
    responsible for emitting the budget-update event afterward.

    Args:
        budget: The ``InstructionBudget`` to populate.
        requests: List of resolved ``TokenCountRequest`` objects.
        context_limit: Context window size for percentage logging.
        on_trace: Diagnostic trace callback.
    """
    # Group by source to compute parent totals
    source_totals: Dict[InstructionSource, int] = {}

    for req in requests:
        source_totals.setdefault(req.source, 0)
        source_totals[req.source] += req.token_count

        # Check if child already exists (Phase 2 update path)
        parent_entry = budget.get_entry(req.source)
        existing = parent_entry.children.get(req.child_key) if parent_entry else None
        if existing is not None:
            existing.tokens = req.token_count
        else:
            if req.token_count > 0:
                budget.add_child(
                    req.source,
                    req.child_key,
                    req.token_count,
                    req.gc_policy,
                    label=req.label,
                )

    # Update parent totals
    for source, total in source_totals.items():
        budget.update_tokens(source, total)

    # ENRICHMENT and CONVERSATION start at 0
    budget.update_tokens(InstructionSource.ENRICHMENT, 0)
    budget.update_tokens(InstructionSource.CONVERSATION, 0)

    # Log summary
    total_initial = sum(source_totals.values())
    estimate_count = sum(1 for r in requests if r.is_estimate)
    try:
        pct = (total_initial / context_limit * 100) if context_limit else 0
        on_trace(
            f"BUDGET_CALC: Budget {'updated' if any(not r.is_estimate for r in requests) else 'initial'} — "
            f"total={total_initial} tokens ({pct:.1f}% of {context_limit}), "
            f"estimates={estimate_count}/{len(requests)}"
        )
    except (TypeError, ValueError):
        on_trace(
            f"BUDGET_CALC: Budget applied — total={total_initial} tokens, "
            f"estimates={estimate_count}/{len(requests)}"
        )
