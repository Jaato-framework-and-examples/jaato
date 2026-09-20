"""Transitive reference expansion: bounded, deterministic, and honest.

``_resolve_transitive_references`` computes the **reachable set** from the
selected references -- the transitive closure's row -- not a neighbourhood.
Its only bound was ``MAX_TRANSITIVE_DEPTH = 10``, and depth is a
*logarithmic* control over an *exponential* quantity: a frontier of
out-degree ``d`` reaches ``d**k`` nodes at depth ``k``, so depth 10 binds
only on catalogs larger than ``d**10``.

Measured on a 200-entry catalog before this change:

    mentions per document | references resolved from ONE selection
    ----------------------+---------------------------------------
                        1 |   4      (depth actually binds)
                        2 | 147
                        3 | 200      <- the entire catalog
                        4 | 200

Three mentions per document -- a "see also" naming three siblings -- is
enough to resolve everything from any starting point.  Every resolved
reference is manifested to the model AND path-authorized, so this is a
context and an authorization-surface question, not only a latency one.

The guards here cover the three separable properties:

1. **Determinism.**  ``pending`` and ``new_mentions`` are sets.  Iterating
   them directly makes discovery order vary across processes (string hash
   randomisation).  Unbounded that only shuffles the manifest; with a
   limit it decides WHICH references survive -- measured at six of
   twenty-five differing between two ``PYTHONHASHSEED`` values.  So the
   sorting must land BEFORE the bound, not with it.
2. **The bound.**  ``max_transitive_references`` binds regardless of link
   structure, which is what ``max_depth`` cannot do.
3. **Honesty.**  A truncated neighbourhood that does not say so is read by
   the model as a complete one, and "this reference does not exist" is
   unfalsifiable from its side.

The matcher rewrite is here too because it is the same call path: the
per-id regex scan was O(catalog x content) and is now O(content).
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

import pytest

from shared.plugins.references.plugin import ReferencesPlugin
from shared.plugins.references.models import (
    InjectionMode,
    ReferenceSource,
    SourceType,
)
from shared.tests.reversion import Reversion

_TARGET = "jaato-server/shared/plugins/references/plugin.py"

REVERSIONS = [
    Reversion(
        target=_TARGET,
        find="            for ref_id in sorted(pending):",
        replace="            for ref_id in pending:",
        because=(
            "iterating the frontier set directly makes expansion order "
            "vary across processes; with a cap it changes WHICH references "
            "survive"
        ),
        test=("TestDeterministicOrder::"
              "test_the_frontier_is_expanded_in_sorted_order"),
    ),
    Reversion(
        target=_TARGET,
        find="                    for mentioned_id in sorted(new_mentions):",
        replace="                    for mentioned_id in new_mentions:",
        because=(
            "discoveries appended in set order make the resolved list "
            "non-reproducible, and it reaches the prompt-cache prefix"
        ),
        test=("TestDeterministicOrder::"
              "test_discoveries_are_appended_in_sorted_order"),
    ),
    Reversion(
        target=_TARGET,
        find=(
            "        found_ids: Set[str] = plain & "
            "set(_ID_BOUNDARY_RE.split(content))"
        ),
        replace="        found_ids: Set[str] = set()",
        because="the one-pass tokenise is what finds ordinary ids at all",
        test="TestOnePassMatching::test_a_plain_id_is_found",
    ),
    Reversion(
        target=_TARGET,
        find="        for ref_id in needs_regex:",
        replace="        for ref_id in ():",
        because=(
            "an id containing a boundary char is unreachable by tokenising "
            "and must keep the original per-id matcher"
        ),
        test=("TestOnePassMatching::"
              "test_an_id_containing_a_boundary_char_is_found"),
    ),
    Reversion(
        target=_TARGET,
        find=(
            "                    for mentioned_id in sorted(new_mentions):\n"
            "                        if self._expansion_at_limit("
            "len(resolved_ids), limit):"
        ),
        replace=(
            "                    for mentioned_id in sorted(new_mentions):\n"
            "                        if False:"
        ),
        because="without the inner check one frontier can overshoot the cap",
        test="TestTheBound::test_the_cap_stops_expansion",
    ),
    Reversion(
        target=_TARGET,
        find="        self._last_transitive_truncation = truncated",
        replace="        self._last_transitive_truncation = None",
        because=(
            "a truncated neighbourhood that does not say so is read as a "
            "complete one"
        ),
        test="TestTheBound::test_truncation_is_reported",
    ),
]


# ---------------------------------------------------------------- helpers

def _inline(ref_id: str, mentions: List[str]) -> ReferenceSource:
    """An INLINE source whose body mentions *mentions*.

    INLINE keeps the whole suite off disk: ``_get_reference_content``
    returns ``source.content`` directly for this type, so a catalog is
    built and traversed in memory.
    """
    body = "\n".join(f"see {m}" for m in mentions)
    return ReferenceSource(
        id=ref_id, name=ref_id, description="", type=SourceType.INLINE,
        mode=InjectionMode.SELECTABLE, content=f"# {ref_id}\n{body}\n",
    )


def _catalog(*sources: ReferenceSource) -> Dict[str, ReferenceSource]:
    return {s.id: s for s in sources}


# ------------------------------------------------------------------ tests

class TestDeterministicOrder:
    """Sorting has to land before the bound, not with it."""

    def test_discoveries_are_appended_in_sorted_order(self) -> None:
        """One node's discoveries are appended ascending.

        Thirty candidates: with the fix this holds by construction, and
        without it the odds of the set iterating in sorted order are
        1/30!.
        """
        kids = [f"kid-{i:02d}" for i in range(30)]
        cat = _catalog(_inline("root", kids), *[_inline(k, []) for k in kids])
        resolved, _ = ReferencesPlugin()._resolve_transitive_references(
            ["root"], cat)

        found = resolved[1:]
        assert found == sorted(found), (
            "depth-1 discoveries were not appended in sorted order")

    def test_the_frontier_is_expanded_in_sorted_order(self) -> None:
        """The frontier itself is walked ascending, not in set order.

        Eight parents each owning one child: the children's order in the
        resolved list mirrors the order their parents were expanded in,
        so this reads the frontier's iteration order without depending on
        any one process's hash seed (1/8! to pass by luck).
        """
        parents = [f"p-{i}" for i in range(8)]
        cat = _catalog(
            _inline("root", parents),
            *[_inline(p, [f"c-{p}"]) for p in parents],
            *[_inline(f"c-{p}", []) for p in parents],
        )
        resolved, _ = ReferencesPlugin()._resolve_transitive_references(
            ["root"], cat)

        children = [r for r in resolved if r.startswith("c-")]
        assert children == sorted(children), (
            "children appeared in their parents' set-iteration order")


class TestOnePassMatching:
    """O(content) instead of O(catalog x content), same answers."""

    def test_a_plain_id_is_found(self) -> None:
        p = ReferencesPlugin()
        assert p._find_referenced_ids(
            "see alpha-01 for details", {"alpha-01", "beta-02"}) == {"alpha-01"}

    def test_an_id_containing_a_boundary_char_is_found(self) -> None:
        """Tokenising cannot reach these, so the regex fallback must.

        Dropping the fallback would silently narrow the graph for any
        catalog whose ids contain a space, bracket or quote.
        """
        p = ReferencesPlugin()
        assert p._find_referenced_ids(
            "mentions (foo bar) here", {"foo bar"}) == {"foo bar"}

    def test_matches_the_original_per_id_regex(self) -> None:
        """Equivalence with the implementation this replaced."""
        def original(content: str, ids: Set[str]) -> Set[str]:
            out = set()
            for rid in ids:
                pat = (rf'(?:^|[\s\[\]`@:,;()\'"{{}}])({re.escape(rid)})'
                       rf'(?:[\s\[\]`@:,;()\'"{{}}]|$)')
                if re.search(pat, content, re.MULTILINE):
                    out.add(rid)
            return out

        ids = {"a-1", "b.2", "c_3", "foo bar", "q'x", "brack[et", "plain"}
        samples = [
            "see a-1 and [[b.2]] and `c_3`",
            "@ref:plain, (foo bar); q'x",
            "nothing here at all",
            "brack[et appears\nplain on another line",
            "a-1a should NOT match a-1 as a prefix",
        ]
        p = ReferencesPlugin()
        for s in samples:
            assert p._find_referenced_ids(s, ids) == original(s, ids), s


class TestTheBound:
    """``max_transitive_references`` binds whatever the graph looks like."""

    @staticmethod
    def _wide_catalog(n: int = 60) -> Dict[str, ReferenceSource]:
        kids = [f"kid-{i:02d}" for i in range(n)]
        return _catalog(_inline("root", kids),
                        *[_inline(k, []) for k in kids])

    def test_unbounded_is_the_default(self) -> None:
        """Absent configuration, behaviour is exactly as before."""
        p = ReferencesPlugin()
        resolved, _ = p._resolve_transitive_references(
            ["root"], self._wide_catalog())
        assert len(resolved) == 61
        assert p._last_transitive_truncation is None

    def test_the_cap_stops_expansion(self) -> None:
        p = ReferencesPlugin()
        resolved, _ = p._resolve_transitive_references(
            ["root"], self._wide_catalog(), max_references=10)
        assert len(resolved) == 10

    def test_truncation_is_reported(self) -> None:
        """And it claims no figure it cannot know."""
        p = ReferencesPlugin()
        p._resolve_transitive_references(
            ["root"], self._wide_catalog(), max_references=10)

        rec = p._last_transitive_truncation
        assert rec is not None, "expansion was cut and said nothing"
        assert rec["reason"] == "max_transitive_references"
        assert rec["limit"] == 10
        assert rec["resolved"] == 10
        assert "does not mean a reference does not exist" in rec["note"]
        # The walk stops early, so how many MORE it would have found is
        # unknown.  A fabricated count is worse than an absent one.
        assert "dropped" not in rec

    def test_the_cap_keeps_the_same_references_whatever_the_hash_seed(
        self,
    ) -> None:
        """The property the sorting exists for, stated directly."""
        cat = self._wide_catalog()
        runs = [
            tuple(ReferencesPlugin()._resolve_transitive_references(
                ["root"], cat, max_references=12)[0])
            for _ in range(3)
        ]
        assert len(set(runs)) == 1

    @pytest.mark.parametrize(
        "raw", [None, 0, -5, "twelve", 12.5, True, False])
    def test_a_malformed_limit_falls_back_to_unbounded(self, raw) -> None:
        """Never to an invented ceiling.

        Silently applying a limit nobody configured would cut a
        neighbourhood for a reason no operator could find.  ``0`` is the
        0-disables convention this tree already uses.
        """
        assert ReferencesPlugin()._coerce_max_transitive(raw) is None

    def test_a_positive_limit_is_taken(self) -> None:
        assert ReferencesPlugin()._coerce_max_transitive(25) == 25
