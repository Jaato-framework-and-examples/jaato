"""The per-arm report: one row per arm, rendered to HTML (and to PDF).

WHAT THIS ANSWERS THAT THE PIVOT DOES NOT
=========================================

:mod:`jaato_eval.report` pivots a sweep into (task, profile set) cells and
answers *which configuration won*.  That is the right aggregate for a
decision and the wrong artefact for the other question a sweep raises
constantly: **what happened to arm 3, and can I go look at it upstream?**

Answering that used to mean reading ``results.jsonl`` by hand,
cross-referencing ``.jaato/logs/session_*.log`` for anything the result
did not carry, and querying the provider's API out of band for the rest.
This document is that lookup, done once, at write time.

The two are complementary and BOTH appear in the document — the pivot
first, because it is what a reader decides from, then the per-arm tables.

THE SESSION ID IS THE POINT
===========================

The column that changes what this report is worth is ``session id``.
OpenRouter's console groups by exactly that id, so a sweep's ids appear
verbatim in its Sessions view — and the row becomes a join onto the
provider's own record of the arm (request count, routed upstream,
per-request cost, generation ids).  Without it the two views cannot be
joined at all, which is why persisting it (jaato #777) was the blocking
half of this feature and the rendering the easy half.

RENDERING: HTML ALWAYS, PDF ON REQUEST
======================================

jaato-eval had no rendering dependency and that is worth protecting, so
the split is:

* **HTML is unconditional and dependency-free.**  It carries print CSS,
  so "open it and print to PDF" is a complete answer, and it is the
  source of truth for the layout.
* **``--pdf`` requires the ``report`` extra** (``pip install
  'jaato-eval[report]'``) and fails loudly with that line when it is
  missing.  It renders THE SAME HTML, so the two artefacts cannot drift.

Rejected: reportlab (hand-built table layout, non-optional dependency)
and pandoc / wkhtmltopdf (an external binary the sweep host must have —
a new failure mode on exactly the machines that run sweeps unattended).

EVERY UNKNOWN PRINTS AS ``—``
=============================

The engine's three-valued discipline extends to the columns: a field it
could not establish renders as an em dash, never as ``0``, ``$0.00`` or
an empty cell.  ``cost —`` does not mean free, ``nudges —`` does not mean
none fired, and the footnotes under each table say so, because a reader
who mistakes one for the other draws the opposite conclusion from the
sweep.
"""
from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .report import Cell, build_cells
from .sign_off import MAX_COMPLETION_NUDGES

#: What an unestablished value renders as.  One glyph, used everywhere, so
#: "we did not find out" never wears the costume of a measurement.
UNKNOWN = "—"

#: Fixed columns, before the per-grader verdict columns.  Order is the
#: reading order of the question: which arm, how did it end, what served
#: it, where do I go to look it up, what did it cost, what did it say.
_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("arm", "set / repeat"),
    ("state", "roll-up"),
    ("model", "bound, not the set name"),
    ("provider", "jaato provider plugin"),
    ("upstream", "who the gateway routed to"),
    ("session id", "the provider-console join key"),
    ("turns", "turns the session consumed"),
    ("cost", "USD, summed across turns"),
    ("budget", "spend against what was allowed"),
    ("nudges", "completion nudges drawn"),
    ("duration", "seconds, agent run only"),
    ("finish", "provider finish reason"),
    ("payload", "canonical hash — determinism across repeats"),
    ("notes", "blocked reason, or an ungraded sign-off"),
)


class ReportDependencyError(RuntimeError):
    """``--pdf`` was asked for and the optional renderer is not installed.

    Carries the install line rather than the import error: the caller
    wants to know what to run, not which module was missing.
    """


# ---------------------------------------------------------------------------
# Cell formatting.  Each returns display text; none of them invent a value.
# ---------------------------------------------------------------------------

def _text(value: Any) -> str:
    """A scalar as display text, with unknown collapsing to :data:`UNKNOWN`."""
    if value is None or value == "":
        return UNKNOWN
    return str(value)


def _money(value: Any) -> str:
    """USD to four places, or :data:`UNKNOWN`.

    Four places because a cheap arm costs $0.0017 and rounding it to
    ``$0.00`` prints the one number this report exists to make visible as
    the number that means "free".
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return UNKNOWN
    return f"${float(value):.4f}"


def _det_cell(cell: Any) -> str:
    """Render ``det``, disclosing arms that never answered.

    Mirrors ``report._det_cell`` — the markdown and HTML views must not
    disagree about a column's meaning.  The share is already over every
    arm that ran, so no denominator is spelled out; the bracketed count
    appears only when an arm produced no payload, which is the case a
    reader would otherwise mistake for a disagreement.

    Args:
        cell: The pivot cell being rendered.

    Returns:
        HTML for the cell.
    """
    qual = ""
    if cell.answered < cell.exercised and cell.answered:
        qual = (f' <span class="qual">({cell.answered} of '
                f'{cell.exercised} answered)</span>')
    if cell.determinism is None:
        return f"&mdash;{qual}"
    return f"{cell.determinism * 100:.0f}%{qual}"


def _percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return UNKNOWN
    return f"{float(value) * 100:.0f}%"


def budget_cell(record: Dict[str, Any]) -> str:
    """Spend read against what this arm was ALLOWED.

    The framework has two independent gates and an arm is subject to
    exactly one of them: a session declaring its own ``budget_control`` is
    on its own books and does not draw on the task pool.  So this renders
    whichever gate applied, and names it — a ceiling shown without saying
    which pot it came from is how a reader concludes a pool failed to bind
    when it was merely bypassed.

    The pool case also carries **what was already gone on arrival**, which
    is the fact the results file could never show: three arms sharing a
    $6.00 pool spent $3.81 + $0.17 + $2.03, and the third was killed
    mid-work with ``budget_exhausted`` and recorded BLOCKED.  From its own
    row that is a model failure.  ``$2.03 / pool $6.00 (63% already
    consumed on arrival)`` is the same arm, correctly described.
    """
    spent = (record.get("usage") or {}).get("cost_usd")
    own = (record.get("budget_ceiling") or {}).get("usd")
    if isinstance(own, (int, float)):
        return f"{_money(spent)} / own {_money(own)}"
    pool = (record.get("pool_limits") or {}).get("usd")
    if not isinstance(pool, (int, float)):
        return UNKNOWN
    arrival = _arrival_fraction(record.get("pool_on_arrival"))
    tail = "" if arrival is None else f" ({_percent(arrival)} consumed on arrival)"
    return f"{_money(spent)} / pool {_money(pool)}{tail}"


def _arrival_fraction(snapshot: Any) -> Optional[float]:
    """The pool's usage fraction when the arm started, if it was read.

    ``declared: false`` is a real answer — the cid carried no pool — and
    is reported as no fraction rather than as ``0%``, which would claim a
    full pool this engine never saw.
    """
    if not isinstance(snapshot, dict) or not snapshot.get("declared"):
        return None
    fraction = snapshot.get("usage_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        return None
    return float(fraction)


def finish_cell(record: Dict[str, Any]) -> str:
    """The finish reason, with the upstream's own word beside it.

    OpenRouter normalises what the upstream said, and the normalisation is
    lossy in exactly the case a reader needs: Gemini's
    ``MALFORMED_FUNCTION_CALL`` arrives as a generic ``error``, and two
    arms of one sweep were explicable only by querying the provider's API
    for the word behind it.  Rendered as ``error (MALFORMED_FUNCTION_CALL)``
    once jaato #766 carries it; until then the parenthetical is simply
    absent — not printed empty, which would suggest the upstream said
    nothing.
    """
    reason = _text(record.get("finish_reason"))
    native = record.get("native_finish_reason")
    return f"{reason} ({native})" if native else reason


def nudges_cell(record: Dict[str, Any]) -> str:
    """Completion nudges drawn, and whether that is the ceiling.

    ``2/2`` is a distinct fact from ``2``: an arm at the ceiling is one
    nudge from being terminated ``NudgeExhausted``, and three of one
    sweep's BLOCKED arms were explained by nothing else.
    """
    count = record.get("completion_nudges")
    if not isinstance(count, int) or isinstance(count, bool):
        return UNKNOWN
    return f"{count}/{MAX_COMPLETION_NUDGES}"


def notes_cell(record: Dict[str, Any]) -> str:
    """Why this arm is not an ordinary PASS, when it is not.

    ``blocked_reason`` and ``error`` are DIFFERENT facts and both are
    shown: the first means there was nothing to grade, the second means
    the arm produced evidence and ended badly anyway (a missing
    ``signal_completion``, jaato #773).  Collapsing them into one "error"
    column would erase the distinction the whole engine is built on.
    """
    parts: List[str] = []
    if record.get("blocked_reason"):
        parts.append(str(record["blocked_reason"]))
    if record.get("error"):
        parts.append(f"graded without a sign-off: {record['error']}")
    return " · ".join(parts) or ""


def _short_hash(value: Any) -> str:
    """First 12 hex characters — enough to compare repeats by eye."""
    return str(value)[:12] if value else UNKNOWN


# ---------------------------------------------------------------------------
# Row + table assembly
# ---------------------------------------------------------------------------

def grader_ids(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Grader ids across a task's arms, in first-seen (manifest) order.

    One column per grader rather than one blob column: a task with a cheap
    script gate and an expensive judge is read by scanning down the judge
    column, and a blob makes that a per-cell parse.
    """
    seen: List[str] = []
    for record in records:
        for verdict in record.get("verdicts") or []:
            identifier = verdict.get("grader_id")
            if identifier and identifier not in seen:
                seen.append(str(identifier))
    return seen


def _verdict_states(record: Dict[str, Any]) -> Dict[str, str]:
    return {str(v.get("grader_id")): str(v.get("state"))
            for v in (record.get("verdicts") or []) if v.get("grader_id")}


def row_cells(record: Dict[str, Any], graders: Sequence[str]) -> List[str]:
    """One arm as display strings, fixed columns then one per grader."""
    states = _verdict_states(record)
    usage = record.get("usage") or {}
    return [
        f"{record.get('profile_set') or 'default'} #{record.get('repeat', 0)}",
        _text(record.get("state")),
        _text(record.get("model")),
        _text(record.get("provider")),
        _text(record.get("upstream_provider")),
        _text(record.get("session_id")),
        _text(record.get("turns")),
        _money(usage.get("cost_usd")),
        budget_cell(record),
        nudges_cell(record),
        f"{float(record.get('duration_seconds') or 0.0):.1f}s",
        finish_cell(record),
        _short_hash(record.get("payload_hash")),
        notes_cell(record),
    ] + [states.get(g, UNKNOWN) for g in graders]


def _by_task(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group arms by task, preserving each task's arms in results order."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_id") or "?")].append(record)
    return dict(grouped)


def _sort_key(record: Dict[str, Any]) -> Tuple[str, int]:
    return (str(record.get("profile_set") or ""), int(record.get("repeat") or 0))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

#: Print CSS included: option 1 of the issue's rendering options is that
#: the browser is a working PDF renderer, so the stylesheet has to be
#: printable rather than merely pretty on screen.  Landscape because the
#: per-arm table is wide by construction — it is a provider console's
#: session list, and those are wide too.
_STYLE = """
/* EVERY COLOUR IS A TOKEN AND THE BODY PAINTS ITS OWN GROUND.
   A page that sets ink and leaves the background to the viewer inherits
   whatever ground the host supplies — which, in a dark-themed browser or
   panel, is near-black behind #1a1a1a text.  The first shipped version
   did exactly that and was illegible on open.  So: a complete light
   palette on bare :root, the same tokens redefined for a dark viewer, and
   `background` stated explicitly rather than assumed. */
:root {
  color-scheme: light dark;
  --bg: #ffffff; --surface: #f6f7f8; --rule: #d8dade; --ink: #1a1a1a;
  --muted: #5f6368; --pass: #1b7f3b; --fail: #b3261e; --blocked: #8a6d00;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181c; --surface: #1e2126; --rule: #333941; --ink: #e6e8eb;
    --muted: #9aa0a6; --pass: #5fd08a; --fail: #ff8d84; --blocked: #e3c25f;
  }
}
* { box-sizing: border-box; }
body { font: 13px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--ink); margin: 2rem; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; }
.sub { color: var(--muted); margin: 0 0 1.5rem; }
/* Inline sibling of .sub: same muted tone, no block margin, so it can sit
   beside a number inside a table cell. */
.qual { color: var(--muted); font-size: .85em; white-space: nowrap; }
table { border-collapse: collapse; width: 100%; margin-bottom: .5rem;
        background: var(--bg); }
th, td { border-bottom: 1px solid var(--rule); padding: .35rem .5rem;
         text-align: left; vertical-align: top; }
th { font-weight: 600; font-size: .78rem; text-transform: uppercase;
     letter-spacing: .03em; color: var(--muted); white-space: nowrap;
     background: var(--surface); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code, .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace;
              font-size: .82em; }
.state-PASS { color: var(--pass); font-weight: 600; }
.state-FAIL { color: var(--fail); font-weight: 600; }
.state-BLOCKED { color: var(--blocked); font-weight: 600; }
.note { color: var(--muted); font-size: .82rem; margin: .25rem 0 1.5rem;
        max-width: 60em; }
.wrap { overflow-x: auto; }
@media print {
  /* Paper is white whatever the viewer's theme is, and this block comes
     last so it wins over the dark palette above.  Without it, printing
     from a dark-themed browser either wastes a cartridge on the ground or
     drops it and prints pale ink on white. */
  :root {
    --bg: #ffffff; --surface: #ffffff; --rule: #bfc3c8; --ink: #000000;
    --muted: #3c4043; --pass: #14602c; --fail: #8c1d17; --blocked: #6a5400;
  }
  @page { size: A4 landscape; margin: 12mm; }
  body { margin: 0; font-size: 9pt; }
  h2 { break-after: avoid; }
  table { break-inside: auto; }
  tr { break-inside: avoid; }
  thead { display: table-header-group; }
  .wrap { overflow-x: visible; }
}
"""

#: Columns rendered right-aligned + tabular, so figures line up down the
#: page.  Named rather than indexed: an inserted column must not silently
#: re-assign the alignment of every column after it.
_NUMERIC = {"turns", "cost", "duration"}

#: Columns whose content is an identifier rather than prose.
_MONO = {"session id", "model", "payload", "upstream"}


def _esc(value: str) -> str:
    """Escape for element TEXT."""
    return html.escape(str(value), quote=False)


def _attr(value: str) -> str:
    """Escape for an ATTRIBUTE value — quotes included.

    Separate from :func:`_esc` because the values that reach attributes
    are not all ours: a ``script`` grader's identifier is the command the
    manifest declared, and a command containing a quote would otherwise
    close the attribute it sits in.
    """
    return html.escape(str(value), quote=True)


def _cell_html(header: str, value: str) -> str:
    classes: List[str] = []
    if header in _NUMERIC:
        classes.append("num")
    if header in _MONO:
        classes.append("mono")
    if header == "state":
        classes.append(f"state-{value}")
    attribute = f' class="{_attr(" ".join(classes))}"' if classes else ""
    return f"<td{attribute}>{_esc(value)}</td>"


def _arm_table(records: Sequence[Dict[str, Any]]) -> str:
    graders = grader_ids(records)
    headers = [name for name, _ in _COLUMNS] + list(graders)
    titles = {name: title for name, title in _COLUMNS}
    # A grader column's tooltip is its own identifier — which, for a
    # `script` grader, IS the command the manifest declared, and is
    # regularly wider than the column.  That makes it untrusted text
    # reaching an attribute, hence `_attr` rather than `_esc`.
    head = "".join(
        f'<th title="{_attr(titles.get(h, h))}">{_esc(h)}</th>'
        for h in headers)
    body = []
    for record in sorted(records, key=_sort_key):
        cells = row_cells(record, graders)
        body.append("<tr>" + "".join(
            _cell_html(header, value) for header, value in zip(headers, cells)
        ) + "</tr>")
    return ('<div class="wrap"><table><thead><tr>' + head
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def _pivot_table(cells: Dict[Tuple[str, str], Cell]) -> str:
    """The existing aggregate, rendered into the same document.

    Kept rather than replaced: this answers which configuration won, and
    the per-arm tables answer what happened to one arm.  A document
    carrying only the second makes the reader re-derive the first.
    """
    head = "".join(f"<th>{_esc(h)}</th>" for h in (
        "task", "profile set", "pass rate", "pass", "fail", "blocked",
        "cost USD", "tokens", "det"))
    rows = []
    for key in sorted(cells):
        c = cells[key]
        rows.append("<tr>" + "".join((
            f"<td>{_esc(c.task_id)}</td>",
            f"<td>{_esc(c.profile_set)}</td>",
            f'<td class="num">{_percent(c.pass_rate)}</td>',
            f'<td class="num">{c.passed}</td>',
            f'<td class="num">{c.failed}</td>',
            f'<td class="num">{c.blocked}</td>',
            f'<td class="num">{_money(c.cost_usd)}</td>',
            f'<td class="num">{c.tokens}</td>',
            f'<td class="num">{_det_cell(c)}</td>',
        )) + "</tr>")
    return ('<div class="wrap"><table><thead><tr>' + head
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


#: Said once per document rather than once per column, because every one
#: of these is a place a reader can silently draw the opposite conclusion.
_FOOTNOTES = (
    "<strong>—</strong> means <em>not established</em>, never zero. "
    "<strong>det</strong> is the largest group of arms that agreed with "
    "each other, over every arm that ran: 100% = byte-identical across all "
    "repeats, 0% = no two arms matched. A bracketed count means some arm "
    "produced no payload, lowering the share without having disagreed; — "
    "means fewer than two arms answered, so agreement could not be "
    "established either way. "
    "A cost of — means neither the provider nor <code>.jaato/pricing.json</code> "
    "reported one; it does not mean free. Nudges of — means the count could not "
    "be read from the session log, not that none fired. "
    "<strong>upstream</strong> and the parenthetical in <strong>finish</strong> "
    "stay — until jaato #766 carries the provider's own words off the wire. "
    "<strong>session id</strong> is the provider console's own key: OpenRouter "
    "groups its Sessions view by exactly this string. "
    "<strong>budget</strong> names which gate applied — <em>own</em> for a "
    "profile-declared <code>budget_control</code> (such a session is on its own "
    "books and does not draw on the task pool), <em>pool</em> for the task's "
    "cascade pool, with what was already consumed when the arm arrived."
)


def render_html(records: Iterable[Dict[str, Any]], *,
                title: str = "jaato-eval sweep report") -> str:
    """A standalone HTML document: the pivot, then one table per task.

    Self-contained by design — no external stylesheet, no script, no
    webfont — so it survives being emailed, committed, or opened from a
    file:// URL on a machine with no network, which is where sweep
    artefacts actually get read.
    """
    records = list(records)
    body: List[str] = [
        f"<h1>{_esc(title)}</h1>",
        f'<p class="sub">{len(records)} arm(s) · '
        f"{len(_by_task(records))} task(s)</p>",
    ]
    if not records:
        body.append("<p>No results.</p>")
    else:
        body.append("<h2>Which configuration won</h2>")
        body.append(_pivot_table(build_cells(records)))
        for task_id, arms in sorted(_by_task(records).items()):
            body.append(f"<h2>{_esc(task_id)}</h2>")
            body.append(_arm_table(arms))
        body.append(f'<p class="note">{_FOOTNOTES}</p>')
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head>"
            "<meta charset=\"utf-8\">"
            f"<title>{_esc(title)}</title>"
            f"<style>{_STYLE}</style></head><body>"
            + "".join(body) + "</body></html>\n")


def write_html(records: Iterable[Dict[str, Any]], path: Path, *,
               title: str = "jaato-eval sweep report") -> Path:
    """Render and write the HTML document.  Always available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(records, title=title), encoding="utf-8")
    return path


def write_pdf(records: Iterable[Dict[str, Any]], path: Path, *,
              title: str = "jaato-eval sweep report") -> Path:
    """Render the SAME HTML to PDF, via the optional ``report`` extra.

    Raises:
        ReportDependencyError: when weasyprint is not installed, carrying
            the install line.  Loud rather than a silent HTML-only
            fallback: a sweep run unattended with ``--pdf`` in its command
            must not quietly produce a different artefact than it was
            asked for.
    """
    try:
        from weasyprint import HTML  # type: ignore import-not-found
    except ImportError as exc:
        raise ReportDependencyError(
            "--pdf needs the optional renderer: "
            "pip install 'jaato-eval[report]'  "
            "(the HTML report is always written and prints to PDF from any "
            "browser, so this is a convenience rather than a requirement)"
        ) from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=render_html(records, title=title)).write_pdf(str(path))
    return path
