"""Pivot the results file into something a human decides from.

The table answers the question the sweep was run to answer: for each
(task, profile set) cell, how often did it pass, what did it cost, and
how deterministic was it.

Two rules the arithmetic follows, both consequences of three-valued
verdicts:

**BLOCKED is never in a denominator.**  Pass rate is ``PASS / (PASS +
FAIL)``.  An arm that did not run is not evidence about the
configuration, and folding it in either way biases the comparison.

**BLOCKED is always visible.**  It gets its own column, because a cell
with a 100% pass rate over two arms and eight blocked ones is not a
result — it is a broken runner, and a report that hid the eight would
read as success.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .verdict import BLOCKED, FAIL, PASS


@dataclass
class Cell:
    """One (task, profile set) cell of the pivot."""

    task_id: str
    profile_set: str
    passed: int = 0
    failed: int = 0
    blocked: int = 0
    cost_usd: Optional[float] = None
    tokens: int = 0
    seconds: float = 0.0
    turns: int = 0
    payload_hashes: set = field(default_factory=set)
    blocked_reasons: List[str] = field(default_factory=list)

    @property
    def exercised(self) -> int:
        return self.passed + self.failed

    @property
    def pass_rate(self) -> Optional[float]:
        """``None`` when nothing was exercised — not ``0.0``.

        Zero would say "it always failed"; the truth is "we never found
        out", and the two must not print the same.
        """
        return (self.passed / self.exercised) if self.exercised else None

    @property
    def determinism(self) -> Optional[float]:
        """Share of arms that produced the modal payload hash.

        ``None`` when no arm produced a payload.  ``1.0`` means every arm
        emitted byte-identical output.
        """
        if not self.payload_hashes:
            return None
        return 1.0 / len(self.payload_hashes) if len(self.payload_hashes) else None


def build_cells(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Cell]:
    """Aggregate JSONL records into pivot cells."""
    cells: Dict[Tuple[str, str], Cell] = {}
    hashes: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for r in records:
        key = (r.get("task_id", "?"), r.get("profile_set") or "default")
        cell = cells.get(key)
        if cell is None:
            cell = cells[key] = Cell(task_id=key[0], profile_set=key[1])

        state = r.get("state")
        if state == PASS:
            cell.passed += 1
        elif state == FAIL:
            cell.failed += 1
        else:
            cell.blocked += 1
            reason = r.get("blocked_reason") or _first_blocked_verdict(r)
            if reason:
                cell.blocked_reasons.append(reason)

        usage = r.get("usage") or {}
        cost = usage.get("cost_usd")
        if isinstance(cost, (int, float)):
            cell.cost_usd = (cell.cost_usd or 0.0) + float(cost)
        spend = usage.get("spend_total_tokens") or 0
        if isinstance(spend, (int, float)):
            cell.tokens += int(spend)
        cell.seconds += float(r.get("duration_seconds") or 0.0)
        cell.turns += int(r.get("turns") or 0)

        h = r.get("payload_hash")
        if h:
            hashes[key].append(h)

    for key, hs in hashes.items():
        cells[key].payload_hashes = set(hs)
    return cells


def _first_blocked_verdict(record: Dict[str, Any]) -> str:
    for v in record.get("verdicts", []) or []:
        if v.get("state") == BLOCKED and v.get("blocked_reason"):
            return f"{v.get('grader_id')}: {v['blocked_reason']}"
    return ""


def render_markdown(records: Iterable[Dict[str, Any]]) -> str:
    """Render the pivot as a markdown table plus a blocked-reason digest."""
    cells = build_cells(records)
    if not cells:
        return "No results.\n"

    lines = [
        "| task | profile set | pass rate | pass | fail | blocked | cost USD | tokens | det |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(cells):
        c = cells[key]
        rate = "—" if c.pass_rate is None else f"{c.pass_rate * 100:.0f}%"
        cost = "—" if c.cost_usd is None else f"{c.cost_usd:.4f}"
        det = "—" if c.determinism is None else f"{c.determinism * 100:.0f}%"
        lines.append(
            f"| {c.task_id} | {c.profile_set} | {rate} | {c.passed} | {c.failed} "
            f"| {c.blocked} | {cost} | {c.tokens} | {det} |")

    digest = _blocked_digest(cells)
    if digest:
        lines.append("")
        lines.append("## Blocked — nothing was exercised")
        lines.append("")
        lines.extend(digest)

    lines.append("")
    lines.append("_Pass rate excludes blocked arms from the denominator; "
                 "`det` is the share of arms sharing the modal payload hash "
                 "(100% = byte-identical across repeats). A cost of `—` means "
                 "neither the provider nor `.jaato/pricing.json` reported one — "
                 "it does not mean free._")
    return "\n".join(lines) + "\n"


def _blocked_digest(cells: Dict[Tuple[str, str], Cell]) -> List[str]:
    """Distinct blocked reasons, most common first, with counts."""
    tally: Dict[str, int] = defaultdict(int)
    for cell in cells.values():
        for reason in cell.blocked_reasons:
            tally[reason] += 1
    return [f"- ({n}×) {reason}"
            for reason, n in sorted(tally.items(), key=lambda kv: -kv[1])]
