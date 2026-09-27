"""Conversation history as renderable units, paged from the bottom up.

The attach replay (``JaatoServer._emit_conversation_replay``) used to walk
the stored history top to bottom and re-emit it as ONE unbroken stream of
output events, with the model's text sent raw -- never through the output
formatter pipeline the live stream goes through.  Two consequences:

* a reconnecting client got the whole conversation or nothing.  A chat
  surface wants the opposite: the most recent screenful now, older ones
  only when the person scrolls up.
* what it got was not what it had been shown.  Fenced code arrived as raw
  markdown rather than ``<j-code>``, tables as pipes rather than
  ``<j-table>``, so a replayed transcript lost the rendering the live one
  had.

This module is the one answer to both.  :func:`build_units` turns a history
into a list of :class:`ReplayUnit` -- the smallest pieces that may be shown
on their own -- and runs every model-text unit through a formatter.
:func:`paginate` cuts that list into pages from the END, by a line budget,
never inside a unit.

What makes a unit unbreakable
-----------------------------

A page boundary may fall only BETWEEN units, so a unit must never split a
thing the formatter renders as one: a fenced code block, a table, a
notebook cell.  Model text is therefore segmented at blank lines that sit
OUTSIDE a fence (fence detection is the code-block formatter's own
``open_fence`` / ``Fence.closes``, imported rather than restated, so the two
cannot disagree about which lines are code).  A markdown table is a run of
consecutive ``|`` lines -- a blank line ends it -- so blank-line
segmentation keeps it whole by construction; ``<notebook-cell>`` and
``<j-*>`` markup a model quotes verbatim is kept whole the same way the
fence is.  Segments of one text part share a ``group`` id so a client can
join them back into one block; each keeps its trailing blank lines so
concatenating a group's formatted segments reproduces the whole part.

A single unit bigger than the page budget becomes a page on its own: the
budget is a target, the unit boundary is the rule.

Cursors
-------

A page names the unit it starts at with a cursor ``"<index>:<digest>"``,
and the next older page is requested ``before`` that cursor.  The index
alone would be wrong the moment GC drops messages from the front of the
history (every later index shifts), so the digest -- over the unit's RAW
content, never its formatted text, which depends on formatter config -- is
checked first, and on a mismatch the unit is looked up by digest nearest
the recorded index.  A cursor that matches nothing is reported ``stale``
rather than silently resolved to some other page: a client told "stale"
re-requests the latest page, a client silently handed a different page
shows the person a hole in the conversation.

Nothing here imports server state; the caller supplies the history and the
formatter.  That keeps the module testable without a daemon and makes
"which formatter" a decision of the one place that owns formatter config
(``JaatoServer``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from jaato_server.shared.plugins.code_block_formatter.plugin import open_fence

#: Default page budget, in rendered lines.  About two tall screens: enough
#: that a first page usually holds the last exchange whole, small enough
#: that a chat client paints it at once.
DEFAULT_PAGE_LINES = 120

#: Hard ceiling on a requested budget, so a client cannot ask for "one page"
#: and receive the whole history in a single frame.
MAX_PAGE_LINES = 2000

_HIDDEN_RE = re.compile(r"<hidden>.*?</hidden>", re.DOTALL)

# Server markup a model may quote verbatim.  An open tag with no close yet
# keeps the segment going across blank lines, as an open fence does.
_MARKUP_OPEN_RE = re.compile(r"<(notebook-cell|nb-row|j-[a-z]+)\b[^>]*(?<!/)>")
_MARKUP_CLOSE_TMPL = "</{tag}>"

UNIT_KINDS = ("user", "model", "thinking", "tools")


@dataclass
class ReplayUnit:
    """One piece of a transcript that may be shown on its own.

    Attributes:
        kind: ``user`` (a prompt), ``model`` (a segment of model text,
            formatted), ``thinking`` (a reasoning part, verbatim) or
            ``tools`` (every function call of one model message, already
            completed).
        text: What to render -- FORMATTED for ``model``, raw otherwise,
            empty for ``tools``.
        raw: The unformatted source the digest is taken over.
        group: Units carved from one text part share this id; a client
            joins consecutive units of one group into one block.
        turn: 0-based index of the user turn this unit belongs to, so a
            client can draw turn separators; ``-1`` before the first
            user message.
        tools: For ``tools`` units, one dict per call:
            ``{call_id, tool_name, tool_args, tool_class, success}``.
        lines: Rendered line estimate; what the page budget counts.
    """

    kind: str
    text: str = ""
    raw: str = ""
    group: str = ""
    turn: int = -1
    tools: List[Dict[str, Any]] = field(default_factory=list)
    lines: int = 1

    @property
    def digest(self) -> str:
        """Stable identity over the RAW content (never the formatted text)."""
        h = hashlib.sha256()
        h.update(self.kind.encode())
        h.update(b"\0")
        h.update(self.raw.encode("utf-8", "replace"))
        for t in self.tools:
            h.update(b"\0")
            h.update(str(t.get("call_id") or "").encode())
            h.update(str(t.get("tool_name") or "").encode())
        return h.hexdigest()[:12]

    def cursor(self, index: int) -> str:
        """The cursor naming this unit at ``index``."""
        return f"{index}:{self.digest}"

    def to_dict(self, index: int) -> Dict[str, Any]:
        """Wire shape of one unit (``HistoryPageEvent.units`` entry)."""
        d: Dict[str, Any] = {
            "id": self.cursor(index),
            "kind": self.kind,
            "group": self.group,
            "turn": self.turn,
            "lines": self.lines,
        }
        if self.kind == "tools":
            d["tools"] = [dict(t) for t in self.tools]
        else:
            d["text"] = self.text
        return d


def _count_lines(text: str) -> int:
    stripped = text.rstrip("\n")
    return stripped.count("\n") + 1 if stripped else 1


def segment_text(text: str) -> List[str]:
    """Split model text into unbreakable segments.

    Splits at blank lines OUTSIDE a fenced block and outside quoted server
    markup.  Each segment keeps the blank lines that follow it, so
    ``"".join(segment_text(t)) == t``.

    Args:
        text: Raw model text (markdown).

    Returns:
        The segments, in order; ``[]`` for empty text.
    """
    if not text:
        return []
    segments: List[str] = []
    current: List[str] = []
    fence = None
    markup_close: Optional[str] = None
    pending_break = False

    lines = text.splitlines(keepends=True)
    for line in lines:
        body = line.rstrip("\n").rstrip("\r")
        blank = not body.strip()
        inside = fence is not None or markup_close is not None

        if not inside and not blank and pending_break and current:
            segments.append("".join(current))
            current = []
        pending_break = False

        current.append(line)

        if fence is not None:
            if fence.closes(body):
                fence = None
        elif markup_close is not None:
            if markup_close in body:
                markup_close = None
        else:
            opened = open_fence(body)
            if opened is not None:
                fence = opened[1]
            else:
                m = _MARKUP_OPEN_RE.search(body)
                if m:
                    close = _MARKUP_CLOSE_TMPL.format(tag=m.group(1))
                    if close not in body[m.end():]:
                        markup_close = close
            if blank:
                pending_break = True

    if current:
        segments.append("".join(current))
    return segments


def _role(msg: Any) -> str:
    role = getattr(msg, "role", "")
    return role.value if hasattr(role, "value") else str(role)


def _tool_outcomes(history: Sequence[Any]) -> Dict[str, bool]:
    """``call_id -> success`` from every function response in the history."""
    out: Dict[str, bool] = {}
    for msg in history:
        for part in (getattr(msg, "parts", None) or []):
            fr = getattr(part, "function_response", None)
            if fr is not None and getattr(fr, "call_id", None):
                out[fr.call_id] = not bool(getattr(fr, "is_error", False))
    return out


def build_units(
    history: Sequence[Any],
    format_text: Optional[Callable[[str], str]] = None,
    classify: Optional[Callable[[str], Any]] = None,
) -> List[ReplayUnit]:
    """Turn a stored history into renderable units, in chronological order.

    Args:
        history: ``Message`` objects (anything with ``role`` / ``parts``).
        format_text: Renders one model-text segment the way the live stream
            is rendered (the output formatter pipeline).  ``None`` leaves
            the text raw -- the pre-pipeline behaviour.  A formatter that
            raises leaves that one segment raw rather than losing it.
        classify: ``tool_name -> tool_class`` (the daemon's
            ``classify_tool``); ``None`` omits the class.

    Returns:
        The units.  ``tool`` role messages produce none: their outcome is
        folded into the ``tools`` unit of the call they answer, as the live
        tool tree shows it.
    """
    outcomes = _tool_outcomes(history)
    units: List[ReplayUnit] = []
    turn = -1

    for mi, msg in enumerate(history):
        role = _role(msg)
        parts = getattr(msg, "parts", None) or []

        if role == "user":
            text = getattr(msg, "text", None) or ""
            text = _HIDDEN_RE.sub("", text)
            if not text.strip():
                continue
            turn += 1
            units.append(ReplayUnit(
                kind="user", text=text, raw=text, group=f"m{mi}",
                turn=turn, lines=_count_lines(text),
            ))
            continue

        if role != "model":
            continue

        for pi, part in enumerate(parts):
            thought = getattr(part, "thought", None)
            if thought:
                units.append(ReplayUnit(
                    kind="thinking", text=thought, raw=thought,
                    group=f"m{mi}p{pi}t", turn=turn,
                    lines=_count_lines(thought),
                ))
            # Not ``elif``: a part may carry reasoning AND text (#1290).
            ptext = getattr(part, "text", None)
            if ptext:
                for seg in segment_text(ptext):
                    rendered = seg
                    if format_text is not None:
                        try:
                            rendered = format_text(seg)
                        except Exception:  # noqa: BLE001 -- keep the text
                            rendered = seg
                    units.append(ReplayUnit(
                        kind="model", text=rendered, raw=seg,
                        group=f"m{mi}p{pi}", turn=turn,
                        lines=_count_lines(rendered),
                    ))

        calls = [p.function_call for p in parts
                 if getattr(p, "function_call", None)]
        if calls:
            tools = []
            for fc in calls:
                entry: Dict[str, Any] = {
                    "call_id": fc.id,
                    "tool_name": fc.name,
                    "tool_args": dict(fc.args or {}),
                    # An unanswered call (cancelled mid-batch) reads as
                    # success=None: the replay does not claim an outcome
                    # it never recorded.
                    "success": outcomes.get(fc.id),
                }
                if classify is not None:
                    try:
                        entry["tool_class"] = classify(fc.name)
                    except Exception:  # noqa: BLE001
                        pass
                tools.append(entry)
            units.append(ReplayUnit(
                kind="tools", group=f"m{mi}c", turn=turn, tools=tools,
                raw="", lines=len(tools),
            ))

    return units


@dataclass
class Page:
    """One page of units, chronological, plus what is needed to go further.

    Attributes:
        units: ``(index, unit)`` pairs, oldest first.
        before: Cursor to pass for the next OLDER page; ``""`` when this
            page starts at the first unit.
        total: Units in the whole history.
        stale: The requested cursor names no unit any more; ``units`` is
            empty and the client should re-request the latest page.
    """

    units: List[tuple] = field(default_factory=list)
    before: str = ""
    total: int = 0
    stale: bool = False

    @property
    def has_more(self) -> bool:
        return bool(self.before)


def resolve_cursor(units: Sequence[ReplayUnit], cursor: str) -> Optional[int]:
    """Index of the unit ``cursor`` names, or ``None`` when none does.

    The index is tried first; on a digest mismatch (GC shifted the list)
    the unit with that digest NEAREST the recorded index wins.
    """
    try:
        idx_s, digest = cursor.split(":", 1)
        idx = int(idx_s)
    except (ValueError, AttributeError):
        return None
    if 0 <= idx < len(units) and units[idx].digest == digest:
        return idx
    matches = [i for i, u in enumerate(units) if u.digest == digest]
    if not matches:
        return None
    return min(matches, key=lambda i: abs(i - idx))


def clamp_page_lines(max_lines: Optional[int]) -> int:
    """A requested budget, bounded; ``0`` / ``None`` / junk -> the default."""
    try:
        n = int(max_lines or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return DEFAULT_PAGE_LINES
    return min(n, MAX_PAGE_LINES)


def paginate(
    units: Sequence[ReplayUnit],
    before: str = "",
    max_lines: Optional[int] = None,
) -> Page:
    """Cut the page that ends just before ``before`` (or at the end).

    Walks BACKWARDS from the end, taking whole units while the line budget
    allows.  The first unit is always taken, so a unit larger than the
    budget is a page by itself rather than an infinite loop of empty pages.

    Args:
        units: Every unit of the history, chronological.
        before: A cursor from a previous page; ``""`` for the latest page.
        max_lines: Page budget in rendered lines (see
            :func:`clamp_page_lines`).
    """
    budget = clamp_page_lines(max_lines)
    total = len(units)
    if before:
        end = resolve_cursor(units, before)
        if end is None:
            return Page(total=total, stale=True)
    else:
        end = total

    start = end
    used = 0
    while start > 0:
        cost = max(1, units[start - 1].lines)
        if start < end and used + cost > budget:
            break
        used += cost
        start -= 1

    page_units = [(i, units[i]) for i in range(start, end)]
    before_cursor = units[start].cursor(start) if start > 0 else ""
    return Page(units=page_units, before=before_cursor, total=total)
