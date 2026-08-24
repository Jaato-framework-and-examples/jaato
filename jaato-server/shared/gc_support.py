"""GC support helpers extracted from ``JaatoSession``.

These functions cover the cohesive, low-coupling parts of the garbage-
collection flow: building the pre-GC telemetry span attributes, populating
the post-GC span from a ``GCResult``, and synchronizing the instruction
budget with the history changes a GC pass made (including summary-entry
creation and cache-plugin notification).

They take their dependencies explicitly (budget, cache plugin, trace
callback) instead of reaching through a session, so the logic is
unit-testable in isolation. The GC *orchestration* — the four collect
paths (after-turn, before-send, context-limit recovery, manual) and the
Phase-0 dedup — stays on :class:`JaatoSession`, which calls these helpers
through thin wrapper methods (preserved so existing tests can still
monkeypatch them on a session instance).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .instruction_budget import GCPolicy, InstructionSource
from .plugins.gc import GCResult

# Trace callback: receives a single human-readable diagnostic string.
TraceFn = Callable[[str], None]


def build_gc_span_attributes(
    context_usage: Dict[str, Any],
    *,
    budget: Any,
    cache_plugin: Any,
) -> Dict[str, Any]:
    """Build the initial attribute dict for a GC telemetry span.

    Captures budget state, cache anchor (if a cache plugin is active), and
    context usage at the moment GC is about to run. These are the "before"
    values; :func:`populate_gc_span_result` adds the "after" values once
    GC completes.

    Args:
        context_usage: Output of ``session.get_context_usage()``.
        budget: The active ``InstructionBudget`` (or ``None``).
        cache_plugin: The active cache plugin (or ``None``).

    Returns:
        Dict of OTel-friendly attributes.
    """
    attrs: Dict[str, Any] = {
        "gc.percent_used": float(context_usage.get("percent_used", 0)),
        "gc.tokens_total": int(context_usage.get("total_tokens", 0)),
        "gc.context_limit": int(context_usage.get("context_limit", 0)),
    }
    if budget:
        try:
            attrs["gc.tokens_before"] = int(budget.total_tokens())
        except Exception:
            pass
    # Cache anchor (if any cache plugin exposes it)
    if cache_plugin and hasattr(cache_plugin, "get_cache_anchor_message_id"):
        try:
            anchor = cache_plugin.get_cache_anchor_message_id()
            if anchor:
                attrs["gc.cache_anchor_message_id"] = anchor
        except Exception:
            pass
    return attrs


def populate_gc_span_result(
    gc_span: Any,
    result: GCResult,
    *,
    on_trace: TraceFn,
) -> None:
    """Populate a GC span with attributes derived from the GC result.

    Called after ``gc_plugin.collect()`` returns. The span receives
    per-phase counts and aggregate metrics so external observers can
    correlate GC operations with subsequent cache hit/miss outcomes.

    Args:
        gc_span: The active OTel span (or no-op span when telemetry is
            disabled).
        result: The ``GCResult`` from ``gc_plugin.collect()``.
        on_trace: Diagnostic trace callback.
    """
    if not gc_span:
        return
    try:
        gc_span.set_attribute("gc.success", bool(result.success))
        gc_span.set_attribute("gc.items_collected", int(result.items_collected))
        gc_span.set_attribute("gc.tokens_freed", int(result.tokens_freed))
        gc_span.set_attribute("gc.tokens_after", int(result.tokens_after))
        # Per-phase counts come from result.details
        details = result.details or {}
        for key in (
            "ephemeral_removed",
            "partial_removed",
            "preservable_removed",
            "enrichment_cleared",
            "tokens_to_free",
            "target_tokens",
        ):
            if key in details:
                val = details[key]
                # bool first to avoid being treated as int
                if isinstance(val, bool):
                    gc_span.set_attribute(f"gc.{key}", val)
                elif isinstance(val, (int, float)):
                    gc_span.set_attribute(f"gc.{key}", val)
    except Exception as e:
        on_trace(f"GC_TELEMETRY: failed to populate span attrs: {e}")


def run_gc(
    *,
    gc_plugin: Any,
    history: Any,
    context_usage: Dict[str, Any],
    gc_config: Any,
    trigger_reason: Any,
    budget: Any,
    cache_plugin: Any,
    telemetry: Any,
    on_trace: TraceFn,
    on_collected: Optional[Callable[[Any, GCResult, Any], Any]] = None,
) -> Tuple[Any, GCResult]:
    """Run one GC pass with its telemetry span, uniformly.

    THE single place a GC pass happens.  Before this existed the four collect
    paths on :class:`JaatoSession` each wired their own instrumentation by
    hand, and only half of them did:

    ===============================  =====================
    site                             telemetry span
    ===============================  =====================
    ``_maybe_collect_after_turn``    yes
    ``_maybe_collect_before_send``   yes
    ``_try_gc_for_context_recovery`` NO
    ``manual_gc``                    NO
    ===============================  =====================

    So an operator watching spans saw a subset of the GC that actually ran,
    with nothing marking the difference -- context-limit recovery and manual
    compaction were invisible, and those are exactly the passes someone
    debugging an overflow goes looking for.  A partially-firing observable is
    worse than none: it reads as complete.  Routing every path through here
    makes instrumentation a property of "a GC pass" rather than of whichever
    call site remembered to add it.

    ``on_collected(new_history, result, gc_span)`` runs INSIDE the span and
    returns the history to hand back (or ``None`` to keep what it was given).
    The four paths genuinely differ after collect -- one re-appends a trailing
    MODEL message held back for a retry, one skips the budget sync -- so that
    stays with the caller; only the parts that should never have differed move
    here.

    Args:
        gc_plugin: The active GC plugin.
        history: Conversation history to collect over.
        context_usage: Output of ``session.get_context_usage()``.
        gc_config: The active ``GCConfig``.
        trigger_reason: ``GCTriggerReason`` for this pass.
        budget: The active ``InstructionBudget`` (or ``None``).
        cache_plugin: The active cache plugin (or ``None``).
        telemetry: Telemetry facade exposing ``gc_span``.
        on_trace: Diagnostic trace callback.
        on_collected: Optional post-collect hook, run inside the span.

    Returns:
        ``(new_history, result)`` exactly as ``gc_plugin.collect`` produced,
        with ``new_history`` replaced by ``on_collected``'s return when it
        returns one.
    """
    attrs = build_gc_span_attributes(
        context_usage, budget=budget, cache_plugin=cache_plugin,
    )
    reason_value = getattr(trigger_reason, "value", trigger_reason)
    with telemetry.gc_span(
        trigger_reason=reason_value,
        strategy=gc_plugin.name,
        attributes=attrs,
    ) as gc_span:
        new_history, result = gc_plugin.collect(
            history, context_usage, gc_config, trigger_reason, budget=budget,
        )
        if on_collected is not None:
            replacement = on_collected(new_history, result, gc_span)
            if replacement is not None:
                new_history = replacement
        populate_gc_span_result(gc_span, result, on_trace=on_trace)
    return new_history, result


def apply_gc_removal_list(
    result: GCResult,
    *,
    budget: Any,
    cache_plugin: Any,
    on_trace: TraceFn,
    gc_span: Any = None,
) -> None:
    """Apply a GC removal list to the instruction budget.

    Synchronizes the budget with the history changes a GC pass made: each
    removal drops a child entry (or bulk-clears a whole source), any
    summary the strategy produced is added as a CONVERSATION child, and
    the cache plugin is notified so it can track prefix invalidation. Must
    be called after a successful GC operation.

    Args:
        result: The ``GCResult`` containing the removal_list.
        budget: The ``InstructionBudget`` to sync (no-op if ``None``).
        cache_plugin: The active cache plugin (or ``None``).
        on_trace: Diagnostic trace callback.
        gc_span: Optional active GC telemetry span; passed to the cache
            plugin's ``on_gc_result`` so it can emit cache-invalidation
            events on the same span.
    """
    if not budget or not result.removal_list:
        return

    for item in result.removal_list:
        if item.child_key:
            # Remove specific child entry
            budget.remove_child(item.source, item.child_key)
        else:
            # Bulk clear entire source (e.g., ENRICHMENT)
            entry = budget.get_entry(item.source)
            if entry:
                entry.tokens = 0
                entry.children.clear()

    # If summary was created (summarize/hybrid plugins), add summary entry
    summary_tokens = result.details.get("summary_tokens")
    if summary_tokens and summary_tokens > 0:
        # Find or create a unique summary key
        conv_entry = budget.get_entry(InstructionSource.CONVERSATION)
        if conv_entry:
            # Count existing summaries to generate unique key
            summary_count = sum(
                1 for key in conv_entry.children.keys()
                if key.startswith("gc_summary_")
            )
            summary_key = f"gc_summary_{summary_count + 1}"
            budget.add_child(
                source=InstructionSource.CONVERSATION,
                child_key=summary_key,
                tokens=summary_tokens,
                gc_policy=GCPolicy.PRESERVABLE,
                label=f"Context Summary #{summary_count + 1}",
                metadata={"created_by": result.plugin_name},
            )

    on_trace(
        f"GC_BUDGET_SYNC: Applied {len(result.removal_list)} removals to budget"
    )

    # Notify cache plugin about GC so it can track prefix invalidation.
    # The cache plugin may emit a 'cache.prefix_invalidated' event on the
    # active gc_span (when provided) so the GC<->cache coordination is
    # visible in the trace.
    if cache_plugin and hasattr(cache_plugin, 'on_gc_result'):
        try:
            # Try the span-aware signature first; fall back to legacy
            # call if the cache plugin only accepts the result.
            try:
                cache_plugin.on_gc_result(result, gc_span=gc_span)
            except TypeError:
                cache_plugin.on_gc_result(result)
        except Exception as e:
            on_trace(f"CACHE_PLUGIN: on_gc_result failed: {e}")
