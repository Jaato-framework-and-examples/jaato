"""When ``max_transitive_references`` cuts a depth, the links decide what survives.

The cap existed, and what it kept was an accident of spelling: each depth
was walked parent by parent in id order, and each parent's discoveries were
admitted in id order, until the limit.  So under a cap of 25 the reference
the selection most needed lost to one whose id sorted earlier, and a
parent's declared ``depends-on`` lost to a sibling's passing mention.

Now each depth is read WHOLE, then ordered by ``links.rank_frontier``:

1. how many parents at the previous depth DECLARE ``depends-on`` to it
   (counted by its current version, the routing ``supersedes`` applies);
2. how many parents reach it at all;
3. the id, so the order stays total and reproducible.

Unbounded, the SET of references is unchanged; only their order within a
depth is.  The truncation record names the ranking, so a reader of a cut
neighbourhood knows what the cut preferred, and every parent of an admitted
reference is still recorded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from jaato_server.shared.plugins.references.links import ReferenceLink
from jaato_server.shared.plugins.references.models import (
    InjectionMode,
    ReferenceSource,
    SourceType,
)
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.tests.reversion import Reversion

_LINKS = "jaato-server/jaato_server/shared/plugins/references/links.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"

REVERSIONS = [
    Reversion(
        target=_LINKS,
        find="    return sorted(candidates, key=lambda cid: (-declared_votes(cid), nearness(cid), -len(candidates[cid]), cid))\n",
        replace="    return sorted(candidates, key=lambda cid: (nearness(cid), -len(candidates[cid]), cid))\n",
        because=(
            "a parent's declared depends-on would lose the cut to a "
            "sibling's passing mention whose id sorts earlier"
        ),
        test="TestDeclaredDependenciesFirst::test_a_declared_dependency_survives_the_cut",
    ),
    Reversion(
        target=_LINKS,
        find="    return sorted(candidates, key=lambda cid: (-declared_votes(cid), nearness(cid), -len(candidates[cid]), cid))\n",
        replace="    return sorted(candidates, key=lambda cid: (-declared_votes(cid), nearness(cid), cid))\n",
        because=(
            "a reference several selected documents point to would lose the "
            "cut to one a single document names"
        ),
        test="TestMoreParentsFirst::test_a_shared_reference_survives_the_cut",
    ),
    Reversion(
        target=_LINKS,
        find="            if cid in {index.current_version(t) for t in index.expanding_targets(parent)}\n",
        replace="            if cid in index.expanding_targets(parent)\n",
        because=(
            "a dependency declared on a reference that was since superseded "
            "is expanded to the newer one, and the ranking must count the "
            "same id the expansion admits"
        ),
        test="TestDeclaredDependenciesFirst::test_a_superseded_dependency_counts_for_its_successor",
    ),
    Reversion(
        target=_PLUGIN,
        find="                parent_map.setdefault(mentioned_id, set()).update(candidates[mentioned_id])\n",
        replace="                parent_map.setdefault(mentioned_id, set()).update(sorted(candidates[mentioned_id])[:1])\n",
        because="a reference two parents reach would record only one of them",
        test="TestNothingElseChanged::test_every_parent_is_recorded",
    ),
    Reversion(
        target=_PLUGIN,
        find='            "ranked_by": ranked_by,\n',
        replace="",
        because="a cut neighbourhood would not say what the cut preferred",
        test="TestNothingElseChanged::test_the_truncation_record_names_the_ranking",
    ),
]


def _ref(ref_id: str, mentions: List[str] = (), links: Optional[List[ReferenceLink]] = None) -> ReferenceSource:
    body = "\n".join(f"see {m}" for m in mentions)
    return ReferenceSource(
        id=ref_id, name=ref_id, description="", type=SourceType.INLINE,
        mode=InjectionMode.SELECTABLE, content=f"# {ref_id}\n{body}\n",
        links=list(links or []),
    )


def _catalog(*sources: ReferenceSource) -> Dict[str, ReferenceSource]:
    return {s.id: s for s in sources}


def _resolve(cat, start, limit=None):
    p = ReferencesPlugin()
    resolved, parents = p._resolve_transitive_references([start], cat, max_references=limit)
    return p, resolved, parents


class TestDeclaredDependenciesFirst:
    def test_a_declared_dependency_survives_the_cut(self):
        # root mentions nine "a-" references and declares depends-on z-dep,
        # which it also mentions.  One slot: the declaration wins it.
        noise = [f"a-{i}" for i in range(9)]
        cat = _catalog(
            _ref("root", noise + ["z-dep"], [ReferenceLink("z-dep", "depends-on")]),
            *[_ref(n) for n in noise], _ref("z-dep"),
        )
        _, resolved, _ = _resolve(cat, "root", limit=2)
        assert resolved == ["root", "z-dep"]

    def test_a_superseded_dependency_counts_for_its_successor(self):
        cat = _catalog(
            _ref("root", ["aa"], [ReferenceLink("old", "depends-on")]),
            _ref("aa"), _ref("old"),
            _ref("new", links=[ReferenceLink("old", "supersedes")]),
        )
        _, resolved, _ = _resolve(cat, "root", limit=2)
        assert resolved == ["root", "new"]


class TestMoreParentsFirst:
    def test_a_shared_reference_survives_the_cut(self):
        # Depth 1: p1..p3.  Depth 2: all three mention zz-shared; only p1
        # mentions aa-once.  One slot left at depth 2.
        cat = _catalog(
            _ref("root", ["p1", "p2", "p3"]),
            _ref("p1", ["aa-once", "zz-shared"]),
            _ref("p2", ["zz-shared"]),
            _ref("p3", ["zz-shared"]),
            _ref("aa-once"), _ref("zz-shared"),
        )
        _, resolved, _ = _resolve(cat, "root", limit=5)
        assert resolved == ["root", "p1", "p2", "p3", "zz-shared"]


class TestNothingElseChanged:
    def test_unbounded_the_set_is_the_same(self):
        cat = _catalog(
            _ref("root", ["p1", "p2"]),
            _ref("p1", ["aa-once", "zz-shared"]),
            _ref("p2", ["zz-shared"]),
            _ref("aa-once"), _ref("zz-shared"),
        )
        p, resolved, _ = _resolve(cat, "root")
        assert sorted(resolved) == ["aa-once", "p1", "p2", "root", "zz-shared"]
        assert p._last_transitive_truncation is None

    def test_every_parent_is_recorded(self):
        cat = _catalog(
            _ref("root", ["p1", "p2"]),
            _ref("p1", ["shared"]), _ref("p2", ["shared"]), _ref("shared"),
        )
        _, _, parents = _resolve(cat, "root")
        assert parents["shared"] == {"p1", "p2"}

    def test_the_truncation_record_names_the_ranking(self):
        cat = _catalog(_ref("root", ["a", "b", "c"]), _ref("a"), _ref("b"), _ref("c"))
        p, _, _ = _resolve(cat, "root", limit=2)
        rec = p._last_transitive_truncation
        assert rec is not None
        assert "depends-on" in rec["ranked_by"]
