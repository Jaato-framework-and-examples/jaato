"""ResultGrepPlugin — model-directed, regex-based filtering of tool results.

See the package docstring for the high-level design.  This module implements:

- the three model-facing tools (``grep_mode_start`` / ``grep_mode_stop`` /
  ``grep_mode_status``),
- the per-session filter state (keyed by the session object so it is correct
  under parallel tool execution), and
- the ``enrich_tool_result`` hook that performs the filtering.

Filtering algorithm (``_apply``) — **format-agnostic line-grep**: the result is
matched line-by-line with ``re.search`` and matching lines are kept with
``before``/``after`` context (non-contiguous windows separated by an elision
marker).  Because it operates on *lines*, not format-specific structure, it works
uniformly across every serialized format the model might scan — YAML, XML, TOML,
CSV, logs, plain text — all of which arrive line-oriented.

The **one** format-specific concession is *compact-JSON expansion*: the framework
serializes tool result dicts as compact (single-line) JSON, which would be one
unsearchable line; so when the input parses as JSON it is pretty-printed before
grepping.  This is a serialization concession, NOT structure-aware filtering — we
deliberately do not build per-format structural extractors (JSON records, YAML
blocks, ...), which do not generalize.  Known bound: a large multi-line string
*embedded as a single JSON field value* stays one escaped line; the model's
escape hatch is ``grep_mode_stop`` for the full result.

When the input was a JSON object the output is re-packaged as a JSON object
(``{"_grep_filtered": {...}, "content": ...}``) so the session's full-dict
enrichment path can ``json.loads`` it back; otherwise a raw ``banner + slice``
string is returned for the text-field path.  Every filtered result carries a
visible banner — the omission is never silent.
"""

import json
import re
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    TRAIT_GREPPABLE_CONTENT,
    DISCOVERABILITY_DEFERRED,
)
from jaato_sdk.plugins.base import ToolResultEnrichmentResult


# Enrichment priority: run LAST (high number = later; cf. artifact_tracker=20,
# lsp=30) so we filter the FULLY enriched payload rather than stripping lines a
# lower-priority enricher would have annotated.
_ENRICHMENT_PRIORITY = 90

# Elision marker inserted between non-contiguous kept windows.
_ELISION = "  ⋮"  # "  ⋮"

# The session's text-field enrichment path keys these on the result dict; we
# never filter LSP diagnostics out from under the model (they are already a
# distilled annotation, and grepping them is net-harmful).
_LSP_KEY = "_lsp_diagnostics"

# Tool names this plugin owns — never filter our own results.
_OWN_TOOLS = frozenset({"grep_mode_start", "grep_mode_stop", "grep_mode_status"})


@dataclass
class GrepFilter:
    """An active grep filter for one session.

    Attributes:
        pattern: The raw regex string (shown to the model in banners).
        regex: The compiled, validated pattern (validated at grep_mode_start).
        before: Context lines to keep before each match (``grep -B``).
        after: Context lines to keep after each match (``grep -A``).
        tools: Optional set of tool names to scope filtering to; ``None`` means
            "all greppable/text results".
        max_matches: Optional cap on the number of match anchors expanded.
    """

    pattern: str
    regex: "re.Pattern[str]"
    before: int = 0
    after: int = 0
    tools: Optional[frozenset] = None
    max_matches: Optional[int] = None


class ResultGrepPlugin:
    """Tool + result-rewriter plugin implementing model-directed grep-mode."""

    def __init__(self) -> None:
        # session object -> GrepFilter.  WeakKeyDictionary so the entry is
        # reclaimed automatically when the session is gc'd (no leak), and so a
        # filter set on ANY thread is visible when results are enriched on the
        # main turn thread (state is keyed by session identity, not by thread).
        self._filters: "weakref.WeakKeyDictionary[Any, GrepFilter]" = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.Lock()
        # Current session per thread (set by set_session, the plugin wiring
        # hook).  Used to resolve "which session am I serving" inside tool
        # executors and enrich_tool_result, neither of which receives a session.
        self._tl = threading.local()
        self._registry = None  # set via set_plugin_registry()

    # ------------------------------------------------------------------ wiring

    @property
    def name(self) -> str:
        return "result_grep"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_system_instructions(self) -> Optional[str]:
        """No always-on instructions.

        The grep-mode tools are discoverable and self-describing; injecting
        standing prompt text here would spend context on every turn, which is
        contrary to this plugin's purpose (saving context).  Personas that want
        the model to lean on grep-mode can mention it in their own prose.
        """
        return None

    def reset_for_next_session(self) -> None:
        """Drop the current session's grep-mode (transient scan state).

        Per the cascade-sharing litmus ("survive only if a subsequent session
        might benefit"): grep-mode is an ephemeral, per-session scanning toggle —
        it must NOT carry into the next session.  Filters for other sessions are
        keyed by their own session object and are untouched.
        """
        session = self._current_session()
        if session is None:
            return
        with self._lock:
            self._filters.pop(session, None)

    def set_session(self, session: Any) -> None:
        """Record the session serving the current thread (plugin wiring hook).

        Mirrors todo's thread-local session tracking so parallel subagents on
        different threads don't clobber each other's current-session reference.
        """
        self._tl.session = session

    def set_plugin_registry(self, registry: Any) -> None:
        """Receive the shared plugin registry (auto-wired during expose).

        Used by ``_greppable_tools`` to enumerate which exposed tools declare
        the ``greppable_content`` trait, so the model can discover them.
        """
        self._registry = registry

    def _current_session(self) -> Any:
        return getattr(self._tl, "session", None)

    # --------------------------------------------------------------- discovery

    def _greppable_tools(self) -> List[str]:
        """Exposed tools that declare the ``greppable_content`` trait.

        These are the tools whose FULL result is routed through enrichment, so
        grep-mode can filter their structured payloads.  Tools that merely carry
        a large text field (stdout/content/...) are also filtered while active,
        but the trait-bearing set is what the model is told it can rely on.
        """
        registry = self._registry
        if registry is None:
            return []
        out: List[str] = []
        for name in registry.list_exposed():
            if TRAIT_GREPPABLE_CONTENT in registry.get_tool_traits(name):
                out.append(name)
        return sorted(out)

    # ------------------------------------------------------------------- tools

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [
            ToolSchema(
                name="grep_mode_start",
                description=(
                    "Start grep-mode: until grep_mode_stop, the results of "
                    "OTHER tool calls are filtered to the lines matching a "
                    "regular expression (with optional context lines), instead "
                    "of being returned in full. Use this when you are scanning "
                    "a tool's bulk output for something specific and do not need "
                    "the whole result — it cuts the context the result consumes. "
                    "The full result is always one grep_mode_stop away, and "
                    "every filtered result is clearly banner-marked as filtered. "
                    "Call grep_mode_status to see which tools support filtering."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Pattern matched against each line of the result using "
                                "Python `re` regular-expression syntax with `re.search` "
                                "semantics (matches anywhere in a line; use ^ / $ to anchor "
                                "to line start/end). Case-sensitive by default — prefix the "
                                "inline flag (?i) for case-insensitive matching, e.g. "
                                "'(?i)spring-data'. An invalid pattern is rejected with the "
                                "specific error so you can correct it."
                            ),
                        },
                        "before": {
                            "type": "integer",
                            "description": "Context lines to keep before each match (like grep -B). Default 0.",
                        },
                        "after": {
                            "type": "integer",
                            "description": "Context lines to keep after each match (like grep -A). Default 0.",
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of tool names to scope filtering to. "
                                "Omit to filter all greppable/text results."
                            ),
                        },
                        "max_matches": {
                            "type": "integer",
                            "description": "Optional cap on the number of matches expanded per result.",
                        },
                    },
                    "required": ["pattern"],
                },
                category="search",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
            ToolSchema(
                name="grep_mode_stop",
                description=(
                    "Stop grep-mode. Subsequent tool results are returned in "
                    "full again. No-op if grep-mode is not active."
                ),
                parameters={"type": "object", "properties": {}},
                category="search",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
            ToolSchema(
                name="grep_mode_status",
                description=(
                    "Report whether grep-mode is active (and with what pattern/"
                    "scope), and list which currently-available tools declare "
                    "greppable content."
                ),
                parameters={"type": "object", "properties": {}},
                category="search",
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
        ]

    def get_executors(self) -> Dict[str, Callable]:
        return {
            "grep_mode_start": self._exec_start,
            "grep_mode_stop": self._exec_stop,
            "grep_mode_status": self._exec_status,
        }

    def get_user_commands(self) -> List[Any]:
        return []

    def get_auto_approved_tools(self) -> List[str]:
        # Pure per-session state toggles — no FS/network side effects.
        return ["grep_mode_start", "grep_mode_stop", "grep_mode_status"]

    def get_command_completions(self, command: str, args: List[str]) -> List[Any]:
        return []

    # --------------------------------------------------------------- executors

    def _exec_start(self, args: Dict[str, Any]) -> Dict[str, Any]:
        session = self._current_session()
        if session is None:
            return {"error": "grep_mode_start: no active session context."}

        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return {"error": "grep_mode_start requires a non-empty 'pattern'."}
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"error": f"grep_mode_start: invalid regex {pattern!r}: {exc}"}

        before = _coerce_nonneg_int(args.get("before"), 0)
        after = _coerce_nonneg_int(args.get("after"), 0)
        max_matches = args.get("max_matches")
        max_matches = (
            _coerce_nonneg_int(max_matches, None) if max_matches is not None else None
        )
        tools_arg = args.get("tools")
        tools = (
            frozenset(t for t in tools_arg if isinstance(t, str))
            if isinstance(tools_arg, list) and tools_arg
            else None
        )

        with self._lock:
            self._filters[session] = GrepFilter(
                pattern=pattern,
                regex=regex,
                before=before,
                after=after,
                tools=tools,
                max_matches=max_matches,
            )

        greppable = self._greppable_tools()
        result: Dict[str, Any] = {
            "status": "grep-mode active",
            "pattern": pattern,
            "before": before,
            "after": after,
            "greppable_tools": greppable,
            "note": (
                "Results of greppable tools (and any tool with a large text "
                "field) will be filtered to matching lines until grep_mode_stop. "
                "Each filtered result is banner-marked; the full result is "
                "available via grep_mode_stop."
            ),
        }
        if tools is not None:
            scoped = sorted(tools)
            result["scoped_to"] = scoped
            not_greppable = [
                t for t in scoped if greppable and t not in greppable
            ]
            if not_greppable:
                result["warning"] = (
                    "These scoped tools do not declare greppable content; their "
                    "structured results will not be filtered unless they carry a "
                    f"large text field: {not_greppable}"
                )
        return result

    def _exec_stop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        session = self._current_session()
        if session is None:
            return {"error": "grep_mode_stop: no active session context."}
        with self._lock:
            had = self._filters.pop(session, None)
        return {
            "status": "grep-mode stopped" if had else "grep-mode was not active",
        }

    def _exec_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        session = self._current_session()
        greppable = self._greppable_tools()
        with self._lock:
            f = self._filters.get(session) if session is not None else None
        if f is None:
            return {"active": False, "greppable_tools": greppable}
        return {
            "active": True,
            "pattern": f.pattern,
            "before": f.before,
            "after": f.after,
            "scoped_to": sorted(f.tools) if f.tools else None,
            "max_matches": f.max_matches,
            "greppable_tools": greppable,
        }

    # --------------------------------------------------------- enrichment hook

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return True

    def get_tool_result_enrichment_priority(self) -> int:
        return _ENRICHMENT_PRIORITY

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> ToolResultEnrichmentResult:
        """Filter ``result`` when grep-mode is active for the current session.

        Passes the result through unchanged when grep-mode is off, the tool is
        one of ours, or the tool is out of an explicit scope — so the cost when
        inactive is a single dict lookup.
        """
        if tool_name in _OWN_TOOLS:
            return ToolResultEnrichmentResult(result=result)

        session = self._current_session()
        if session is None:
            return ToolResultEnrichmentResult(result=result)
        with self._lock:
            f = self._filters.get(session)
        if f is None:
            return ToolResultEnrichmentResult(result=result)
        if f.tools is not None and tool_name not in f.tools:
            return ToolResultEnrichmentResult(result=result)

        return self._apply(result, f)

    # ------------------------------------------------------------------ filter

    def _apply(self, result: str, f: GrepFilter) -> ToolResultEnrichmentResult:
        """Apply ``f`` to ``result`` and ALWAYS return a valid JSON envelope.

        Output shape (every path): a JSON object
        ``{"_grep_filtered": {pattern, lines_shown, lines_total, note},
        "content": <filtered slice>}`` (carrying any ``_lsp_diagnostics`` through
        if the input was a dict that had it).

        **Why always JSON** (regression #289): the session's full-dict
        enrichment path (``TRAIT_FILE_WRITER`` / ``TRAIT_GREPPABLE_CONTENT``)
        does ``json.loads()`` on our returned string and, on ``JSONDecodeError``,
        DISCARDS our output (stuffs it under ``_lsp_diagnostics`` and keeps the
        original UNFILTERED result — ``jaato_session.py`` ~6690).  Because
        ``result_grep`` runs last in the priority-ordered enrichment chain, an
        upstream enricher (e.g. LSP appending Java diagnostics to a
        ``call_service`` result whose body mentions a Java FQCN) can have already
        rewritten the result's JSON into non-JSON before we see it.  The old code
        keyed its OUTPUT format on the INPUT's parseability and returned a raw
        ``banner + text`` string in that case → the full-dict path discarded it →
        that one result of a parallel batch silently passed through unfiltered
        (the "arbitrary subset" symptom).  Emitting valid JSON unconditionally
        makes ``json.loads`` always succeed, so every result is filtered
        consistently regardless of upstream mutation.  A JSON-object string is
        also accepted verbatim by the text-field / bare-string paths (they take
        our output as the field/result value as-is), so this is safe everywhere.
        """
        obj, is_json = _try_json(result)

        lsp = None
        if is_json and isinstance(obj, dict) and _LSP_KEY in obj:
            obj = dict(obj)
            lsp = obj.pop(_LSP_KEY)

        if is_json:
            text = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False)
        else:
            # Non-JSON input — most often a result whose JSON an upstream
            # enricher already mangled (see #289), or a genuine plain-text
            # field.  Line-grep it as-is; we still wrap the slice in valid JSON.
            text = result

        slice_text, kept, total = _line_grep(
            text, f.regex, f.before, f.after, f.max_matches
        )
        meta = {"pattern": f.pattern, "lines_shown": kept, "lines_total": total}

        payload: Dict[str, Any] = {
            "_grep_filtered": {
                "pattern": f.pattern,
                "lines_shown": kept,
                "lines_total": total,
                "note": "grep-mode active; call grep_mode_stop for the full result",
            },
            "content": slice_text,
        }
        if lsp is not None:
            payload[_LSP_KEY] = lsp
        return ToolResultEnrichmentResult(
            result=json.dumps(payload, ensure_ascii=False), metadata=meta
        )


# --------------------------------------------------------------------- helpers


def _coerce_nonneg_int(value: Any, default: Optional[int]) -> Optional[int]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n >= 0 else default


def _try_json(s: str):
    """Return ``(obj, True)`` if ``s`` parses as JSON, else ``(None, False)``."""
    try:
        return json.loads(s), True
    except (ValueError, TypeError):
        return None, False


def _line_grep(
    text: str,
    regex: "re.Pattern[str]",
    before: int,
    after: int,
    max_matches: Optional[int],
):
    """Keep lines matching ``regex`` with ``before``/``after`` context.

    Returns ``(slice_text, kept_line_count, total_line_count)``.  Non-contiguous
    kept windows are separated by an elision marker.  When nothing matches, the
    slice is a short explicit "no match" note (kept=0) rather than empty output.
    """
    lines = text.split("\n")
    total = len(lines)
    match_idx = [i for i, line in enumerate(lines) if regex.search(line)]
    if max_matches is not None:
        match_idx = match_idx[:max_matches]
    if not match_idx:
        return (f"(no lines matched /{regex.pattern}/)", 0, total)

    keep = set()
    for i in match_idx:
        lo = max(0, i - before)
        hi = min(total, i + after + 1)
        keep.update(range(lo, hi))

    out: List[str] = []
    prev: Optional[int] = None
    for j in sorted(keep):
        if prev is not None and j > prev + 1:
            out.append(_ELISION)
        out.append(lines[j])
        prev = j
    return ("\n".join(out), len(keep), total)


def create_plugin() -> ResultGrepPlugin:
    return ResultGrepPlugin()
