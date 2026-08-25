"""Tool-result truncation/capping helpers extracted from ``JaatoSession``.

Two pure transforms that shrink oversized tool results so a turn fits the
model's context window:

- :func:`cap_tool_results` — *proactive* cap applied before results enter
  history (a single result can dwarf the whole window, so it uses a hard
  per-result character cap rather than removal math).
- :func:`truncate_results_to_fit` — *reactive* truncation during
  context-limit recovery, using the model-reported current/limit token
  counts to remove just enough from the largest results.

Both are side-effect free apart from the injected ``on_trace`` callback —
they read no session state and return new ``ToolResult`` lists — so they
are unit-testable in isolation. ``JaatoSession`` keeps thin wrappers that
supply the budget-derived inputs and its trace callback, plus the
stateful budget/ledger sync (``_sync_budget_after_truncation``) and the
history mutation (``_remove_tool_results_from_history``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List

from jaato_sdk.plugins.model_provider.types import ToolResult

# Trace callback: receives a single human-readable diagnostic string.
TraceFn = Callable[[str], None]

# Lines to keep from the start of a truncated result.
TRUNCATION_PRESERVE_LINES = 20
# Minimum characters to keep when using char-based truncation.
TRUNCATION_PRESERVE_CHARS = 2000
# Target 80% of context limit to leave headroom after truncation.
TRUNCATION_TARGET_PERCENT = 0.80

TRUNCATION_NOTICE = (
    "\n\n[NOTICE: This tool result was automatically truncated because it caused "
    "the prompt to exceed the model's context window. Only the first {kept} "
    "of {total} are shown above ({removed_tokens} estimated tokens removed). "
    "If you need more content, re-invoke the tool with offset/limit parameters "
    "to read in smaller chunks.]"
)


def truncate_results_to_fit(
    tool_results: List[ToolResult],
    current_tokens: int,
    limit_tokens: int,
    *,
    on_trace: TraceFn,
) -> List[ToolResult]:
    """Truncate tool results to reduce token count, preserving first lines.

    Strategy:
    - Targets 80% of the model's context limit to leave headroom.
    - Targets the largest results first (most likely culprits).
    - Preserves the first N lines of content so the model retains context.
    - Appends a notice informing the model about the truncation.
    - Never removes the tool result itself (models expect one response per call).
    - Continues truncating multiple tool results until target is reached.

    Args:
        tool_results: The original tool results.
        current_tokens: Current total tokens as reported by the model error.
        limit_tokens: Maximum allowed tokens as reported by the model error.
        on_trace: Diagnostic trace callback.

    Returns:
        A new list of tool results with large ones truncated.
    """
    # Estimate size of each result
    result_sizes = []
    for i, tr in enumerate(tool_results):
        result_str = str(tr.result) if tr.result is not None else ""
        estimated_tokens = len(result_str) / 4  # ~4 chars per token
        result_sizes.append((i, estimated_tokens, result_str))

    total_result_tokens = sum(size for _, size, _ in result_sizes)

    # Calculate target: reduce to 80% of limit to leave headroom
    # target_removal = how many tokens we need to remove from current
    target_context = int(limit_tokens * TRUNCATION_TARGET_PERCENT)
    target_removal = current_tokens - target_context

    on_trace(
        f"CONTEXT_LIMIT_RECOVERY: truncate called with current={current_tokens}, "
        f"limit={limit_tokens}, target_context={target_context} (80%), "
        f"target_removal={target_removal}, total_result_tokens={total_result_tokens}, "
        f"num_results={len(tool_results)}"
    )

    # If we couldn't extract valid token counts, be aggressive: cut 50% of results
    if target_removal <= 0:
        target_removal = int(total_result_tokens * 0.5)
        on_trace(f"CONTEXT_LIMIT_RECOVERY: using aggressive default target_removal={target_removal}")

    # Sort indices by size descending to truncate largest first
    sized_indices = sorted(
        range(len(result_sizes)),
        key=lambda j: result_sizes[j][1],
        reverse=True,
    )

    truncated = list(tool_results)  # shallow copy
    tokens_removed = 0.0
    preserve_lines = TRUNCATION_PRESERVE_LINES

    for j in sized_indices:
        if tokens_removed >= target_removal:
            break

        idx, size, result_str = result_sizes[j]
        tr = tool_results[idx]

        # Skip small results (< 200 tokens estimated) — not worth truncating
        if size < 200:
            on_trace(f"CONTEXT_LIMIT_RECOVERY: skipping result {idx} (size={size} < 200)")
            continue

        # Split into lines and try line-based truncation first
        lines = result_str.split('\n')

        # Calculate how much content to keep (in characters)
        # Keep enough to preserve context but remove overflow + safety margin
        chars_to_remove = int(target_removal * 4)  # tokens -> chars
        chars_to_keep = max(2000, len(result_str) - chars_to_remove)  # Keep at least 2000 chars

        if len(lines) > preserve_lines:
            # Line-based truncation: keep first N lines
            kept_lines = lines[:preserve_lines]
            kept_text = '\n'.join(kept_lines)
            truncation_unit = "lines"
            truncation_kept = preserve_lines
            truncation_total = len(lines)
        elif len(result_str) > chars_to_keep:
            # Character-based truncation: content has few lines but is large
            # Keep first chars_to_keep characters
            kept_text = result_str[:chars_to_keep]
            # Try to break at a word boundary
            last_space = kept_text.rfind(' ', max(0, chars_to_keep - 200))
            if last_space > chars_to_keep // 2:
                kept_text = kept_text[:last_space]
            truncation_unit = "characters"
            truncation_kept = len(kept_text)
            truncation_total = len(result_str)
            on_trace(
                f"CONTEXT_LIMIT_RECOVERY: using char-based truncation for result {idx} "
                f"(lines={len(lines)}, chars={len(result_str)} -> {len(kept_text)})"
            )
        else:
            on_trace(
                f"CONTEXT_LIMIT_RECOVERY: skipping result {idx} "
                f"(lines={len(lines)}, chars={len(result_str)} — already small enough)"
            )
            continue

        kept_tokens = len(kept_text) / 4
        removed_tokens = size - kept_tokens

        if removed_tokens <= 0:
            continue

        # Build the truncated content with notice
        notice = TRUNCATION_NOTICE.format(
            kept=f"{truncation_kept} {truncation_unit}",
            total=f"{truncation_total} {truncation_unit}",
            removed_tokens=f"{int(removed_tokens):,}",
        )
        truncated_content = kept_text + notice

        # ``replace`` rather than a fresh ToolResult: rebuilding by hand
        # silently dropped every field not listed, and the one that
        # mattered was ``untrusted``.  The inversion was the dangerous
        # part -- the BIGGER an untrusted result, the likelier it was
        # truncated, so the payloads most worth wrapping were exactly the
        # ones that lost their boundary.  ``replace`` also means the next
        # field added to ToolResult cannot be dropped here by omission.
        truncated[idx] = replace(
            tr,
            result=truncated_content,
            attachments=None,  # Drop attachments to reduce size
        )
        tokens_removed += removed_tokens

    return truncated


def cap_tool_results(
    tool_results: List[ToolResult],
    *,
    context_limit: int,
    current_total_tokens: int,
    on_trace: TraceFn,
) -> List[ToolResult]:
    """Proactively cap tool results before they enter history.

    Estimates the aggregate token size of all results and, if they would
    push the context beyond 80% of the model's limit, truncates the
    largest results with a hard character cap.

    Uses a direct cap approach (not the removal-based math in
    :func:`truncate_results_to_fit`) because a single oversized result can
    be many times larger than the entire context window — the removal
    formula underflows in that case.

    Args:
        tool_results: The tool results about to be appended to history.
        context_limit: The model's context window size in tokens.
        current_total_tokens: The budget's current total token count.
        on_trace: Diagnostic trace callback.

    Returns:
        The original list (unchanged) if results fit, or a new list with
        large results truncated.
    """
    # Estimate per-result sizes
    result_sizes = []
    total_result_tokens = 0
    for tr in tool_results:
        result_str = str(tr.result) if tr.result is not None else ""
        tokens = len(result_str) / 4  # ~4 chars per token
        result_sizes.append((tr, result_str, tokens))
        total_result_tokens += tokens

    # Cap: available space to reach 80% of context limit
    target = int(context_limit * TRUNCATION_TARGET_PERCENT)
    cap_tokens = max(0, target - current_total_tokens)

    if total_result_tokens <= cap_tokens:
        on_trace(
            f"PROACTIVE_CAP: result_tokens={int(total_result_tokens)}, "
            f"cap_tokens={int(cap_tokens)}, action=passed"
        )
        return tool_results

    on_trace(
        f"PROACTIVE_CAP: result_tokens={int(total_result_tokens)}, "
        f"cap_tokens={int(cap_tokens)}, action=truncating"
    )

    # Hard cap: each result gets at most cap_tokens (divided equally
    # if multiple, but in practice one result dominates).
    n_results = len(tool_results)
    per_result_cap_tokens = max(
        TRUNCATION_PRESERVE_CHARS // 4,
        cap_tokens // max(1, n_results),
    )
    per_result_cap_chars = int(per_result_cap_tokens * 4)

    truncated = []
    for tr, result_str, tokens in result_sizes:
        if tokens <= per_result_cap_tokens:
            truncated.append(tr)
            continue

        # Truncate to hard character cap
        kept_text = result_str[:per_result_cap_chars]

        # Try to break at a word or line boundary
        last_newline = kept_text.rfind('\n', max(0, per_result_cap_chars - 500))
        if last_newline > per_result_cap_chars // 2:
            kept_text = kept_text[:last_newline]
        else:
            last_space = kept_text.rfind(' ', max(0, per_result_cap_chars - 200))
            if last_space > per_result_cap_chars // 2:
                kept_text = kept_text[:last_space]

        # Determine units for the notice
        original_lines = result_str.count('\n') + 1
        kept_lines = kept_text.count('\n') + 1
        if original_lines > 1:
            unit_kept = f"{kept_lines} lines"
            unit_total = f"{original_lines} lines"
        else:
            unit_kept = f"{len(kept_text):,} characters"
            unit_total = f"{len(result_str):,} characters"

        removed_tokens = int(tokens - len(kept_text) / 4)
        notice = TRUNCATION_NOTICE.format(
            kept=unit_kept,
            total=unit_total,
            removed_tokens=f"{removed_tokens:,}",
        )

        on_trace(
            f"PROACTIVE_CAP: truncated result '{tr.name}' from "
            f"{int(tokens)} to {int(len(kept_text)/4)} tokens "
            f"(cap={per_result_cap_tokens})"
        )

        # See the note in ``truncate_results_to_fit`` -- same rebuild,
        # same dropped untrusted mark, same fix.
        truncated.append(replace(
            tr,
            result=kept_text + notice,
            attachments=None,  # Drop attachments to reduce size
        ))

    return truncated
