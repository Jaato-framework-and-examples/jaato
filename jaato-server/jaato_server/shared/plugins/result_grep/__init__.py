"""result_grep plugin — model-directed filtering of tool-result content.

A *result-rewriter* plugin: while the model has grep-mode active, the results
of OTHER tools (those declaring the ``greppable_content`` trait, plus any tool
whose result carries a large text field) are filtered down to the lines matching
a regex (± context), instead of being sent to the model — and stored in history —
in full.  This cuts context cost on bulk-content tools (e.g. ``call_service``
registry/HTTP lookups) when the model already knows what it is scanning for.

Filtering is **format-agnostic line-grep** (``re.search`` per line) so it works
across any serialized format the model scans — JSON, YAML, XML, TOML, CSV, logs,
plain text.  The sole format-specific step is expanding compact (single-line)
JSON before grepping, because the framework serializes tool result dicts as
compact JSON; per-format structural filtering is deliberately avoided as it does
not generalize.

Two hats, one plugin:

1. **Tools** (``grep_mode_start`` / ``grep_mode_stop`` / ``grep_mode_status`` /
   ``grep_once``) let the model toggle/scope the filter and discover which tools
   are greppable.
2. **Tool-result enrichment** (``enrich_tool_result``): while a filter is
   active, intervening tool results are filtered.  Every filtered result carries
   a visible banner naming the pattern + how much was elided, so the omission is
   always model-visible — never a silent truncation.

The filter is **model-directed** by construction (the model turns it on and
knows it is receiving a slice), which keeps it on the right side of the
silent-truncation bug class that the parallel-tool-result fix closed.

Per-session state is keyed by the session object (``WeakKeyDictionary``), so a
filter set on any thread is visible when results are enriched on the main turn
thread — correct under parallel tool execution.
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

# Runner-tier: per-session in-memory state only (no daemon-only state, no FS
# writes).  Same tier as permission / todo.
PLUGIN_TIER = "runner"

from .plugin import ResultGrepPlugin, create_plugin

__all__ = [
    "ResultGrepPlugin",
    "create_plugin",
    "PLUGIN_KIND",
    "PLUGIN_TIER",
]
