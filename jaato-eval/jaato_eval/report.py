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

from collections import Counter, defaultdict
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
    #: How many arms produced each payload hash.  A Counter rather than a
    #: set because the modal SHARE is the statistic being reported, and a
    #: set discards exactly the frequencies that share is computed from.
    payload_hash_counts: "Counter[str]" = field(default_factory=Counter)
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
    def payload_hashes(self) -> set:
        """The distinct payload hashes, derived from the counts."""
        return set(self.payload_hash_counts)

    @property
    def answered(self) -> int:
        """Arms that produced a payload hash at all.

        The denominator of :attr:`determinism`, and deliberately not
        :attr:`exercised`: an arm that died before emitting a payload is
        not evidence that the arms disagreed, only that we never found out
        what it would have said.
        """
        return sum(self.payload_hash_counts.values())

    @property
    def determinism(self) -> Optional[float]:
        """Share of arms that produced the single most repeated payload.

        The largest group of arms that agreed with each other, over every
        arm that ran (:attr:`exercised`) — and ``0.0`` when no payload was
        produced more than once, because a group of one is not agreement.

        ``None`` — rendered as an em dash — until at least two arms have
        answered.  One observation cannot agree with anything, and it
        cannot disagree either: ``0%`` would claim the arms differed when
        the truth is that only one of them spoke.

        Three earlier readings of this column were wrong (jaato #798):

        * ``1 / distinct_hashes`` discarded how many arms stood behind each
          hash, so two arms agreeing out of three printed 50% where the
          answer is 67%.
        * An arm that produced no payload was dropped from BOTH sides of
          the fraction, so one arm answering out of two printed ``100%``
          under a footer calling it "byte-identical across repeats".
        * Any modal SHARE has a floor of ``1/n`` rather than 0, so total
          disagreement read as 50% across two arms and 25% across four —
          the same printed number meaning "nothing matched" in one cell
          and "half matched" in another.

        Silent arms stay in the denominator on purpose: they did not
        disagree, but they did not reproduce the answer either, and a
        determinism figure that ignored them would claim 100% from a
        minority of the runs.  :attr:`answered` is rendered beside the
        share so a reader can see which it is.
        """
        if self.answered < 2:
            return None
        modal = self.payload_hash_counts.most_common(1)[0][1]
        return 0.0 if modal == 1 else modal / self.exercised


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
        cells[key].payload_hash_counts = Counter(hs)
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
        det = _det_cell(c)
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
                 "`det` is the largest group of arms that agreed with each "
                 "other, over every arm that ran — 100% = byte-identical "
                 "across all repeats, 0% = no two arms matched. A count in "
                 "brackets means some arm produced no payload at all, so it "
                 "lowered the share without having disagreed. `—` means "
                 "fewer than two arms answered, so agreement could not be "
                 "established either way. A cost of `—` means "
                 "neither the provider nor `.jaato/pricing.json` reported one — "
                 "it does not mean free._")
    return "\n".join(lines) + "\n"


def _det_cell(cell: Cell) -> str:
    """Render the ``det`` column, disclosing arms that never answered.

    The share is already over every arm that ran, so it needs no
    denominator spelled out — ``0%`` means no two arms matched and
    ``100%`` means they all did.  What the percentage cannot show is WHY
    it is below 100: arms that disagreed and arms that produced nothing
    both pull it down.  The count is therefore appended only when some
    arm stayed silent, which is exactly when the reader would otherwise
    mistake a crash for a disagreement.

    Args:
        cell: The pivot cell being rendered.

    Returns:
        ``"67%"``; ``"67% (2 of 3 answered)"`` when an arm produced no
        payload; ``"—"`` when fewer than two arms answered at all.
    """
    qual = ""
    if cell.answered < cell.exercised:
        qual = f" ({cell.answered} of {cell.exercised} answered)"
    if cell.determinism is None:
        return f"—{qual}"
    return f"{cell.determinism * 100:.0f}%{qual}"


def _blocked_digest(cells: Dict[Tuple[str, str], Cell]) -> List[str]:
    """Distinct blocked reasons, most common first, with counts."""
    tally: Dict[str, int] = defaultdict(int)
    for cell in cells.values():
        for reason in cell.blocked_reasons:
            tally[reason] += 1
    return [f"- ({n}×) {reason}"
            for reason, n in sorted(tally.items(), key=lambda kv: -kv[1])]
