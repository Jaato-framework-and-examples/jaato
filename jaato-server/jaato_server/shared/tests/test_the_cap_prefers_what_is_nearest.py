"""Where a workspace has embeddings, the transitive cap keeps the nearest references.

``test_the_cap_keeps_what_the_links_rank_first.py`` covers the link
ranking: declared ``depends-on`` first, then how many parents reach a
reference, then its id.  With a vector index, a depth that is CUT is ranked
by similarity to the selection between those first two tiers -- the
nearest references rather than the best-linked ones.

Properties, each a way it could go wrong:

1. **The nearest survive.**  A reference close to what was selected beats
   one more parents happen to mention.
2. **A declared dependency still comes first.**  An author's ``depends-on``
   outranks a vector's opinion.
3. **Similarity is asked only when the depth is cut.**  Unbounded, or with
   room for every candidate, nothing is embedded.
4. **A depth with an unindexed candidate falls back to links.**  A
   reference with no vector cannot be compared, and ranking it below the
   indexed ones would prefer a reference for being in an indexed bundle.
5. **The selection is embedded from the fields the index was made from**
   (``embedding_text``: name, description, tags, fetch hint).
6. **The truncation record says which ranking cut the depth.**
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from jaato_server.shared.plugins.references.bundle import ReferenceBundle
from jaato_server.shared.plugins.references.links import (
    FRONTIER_RANKING,
    FRONTIER_RANKING_SIMILARITY,
    ReferenceLink,
)
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
        find="        return -similarity[cid] if similarity is not None else 0.0\n",
        replace="        return 0.0\n",
        because="the cut would keep the best-linked references, not the nearest",
        test="TestTheNearestSurvive::test_a_near_reference_beats_a_more_mentioned_one",
    ),
    Reversion(
        target=_LINKS,
        find="    return sorted(candidates, key=lambda cid: (-declared_votes(cid), nearness(cid), -len(candidates[cid]), cid))\n",
        replace="    return sorted(candidates, key=lambda cid: (nearness(cid), -declared_votes(cid), -len(candidates[cid]), cid))\n",
        because="a vector's opinion would outrank what the author declared",
        test="TestTheNearestSurvive::test_a_declared_dependency_still_comes_first",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if room is not None and len(candidates) > room:\n",
        replace="        if True:\n",
        because="every selection would pay for an embedding whose ranking changes nothing",
        test="TestOnlyWhenCut::test_unbounded_embeds_nothing",
    ),
    Reversion(
        target=_PLUGIN,
        find="        unscored = candidate_ids - scores.keys()\n"
             "        if unscored:\n",
        replace="        unscored = candidate_ids - scores.keys()\n"
                "        if False:\n",
        because=(
            "a candidate with no vector cannot be compared; a partial ranking "
            "would fail or prefer references for being indexed"
        ),
        test="TestPartialIndex::test_an_unindexed_candidate_falls_back_to_links",
    ),
    Reversion(
        target=_PLUGIN,
        find="        texts = [embedding_text(catalog_by_id[i]) for i in initial_ids if i in catalog_by_id]\n",
        replace="        texts = [i for i in initial_ids if i in catalog_by_id]\n",
        because=(
            "the query would be made from different text than the vectors it "
            "is compared against"
        ),
        test="TestTheQuery::test_the_selection_is_embedded_from_its_metadata",
    ),
    Reversion(
        target=_PLUGIN,
        find="        ranked_by = FRONTIER_RANKING_SIMILARITY if scores is not None else FRONTIER_RANKING\n",
        replace="        ranked_by = FRONTIER_RANKING\n",
        because="a cut neighbourhood would misstate what the cut preferred",
        test="TestTheNearestSurvive::test_the_truncation_record_names_similarity",
    ),
]


class _Provider:
    model_name = "fake"
    available = True

    def __init__(self) -> None:
        self.texts: List[str] = []

    def embed_text_as_array(self, text: str):
        self.texts.append(text)
        return [1.0]


class _Matcher:
    available = True

    def __init__(self, scores: Dict[str, float]) -> None:
        self._scores = scores

    def score_sources(self, query_vec, source_ids: Set[str]) -> Dict[str, float]:
        return {sid: self._scores[sid] for sid in source_ids if sid in self._scores}


def _ref(ref_id: str, mentions: List[str] = (), links: Optional[List[ReferenceLink]] = None,
         description: str = "") -> ReferenceSource:
    body = "\n".join(f"see {m}" for m in mentions)
    return ReferenceSource(
        id=ref_id, name=ref_id, description=description, type=SourceType.INLINE,
        mode=InjectionMode.SELECTABLE, content=f"# {ref_id}\n{body}\n",
        links=list(links or []),
    )


def _plugin(scores: Dict[str, float]):
    p = ReferencesPlugin()
    provider = _Provider()
    p._embedding_provider = provider
    bundle = ReferenceBundle(name="", directory=Path("/nonexistent"))
    bundle.owned_source_ids = set(scores)
    bundle.matcher = _Matcher(scores)
    p._bundles = [bundle]
    return p, provider


def _resolve(p, cat, start, limit=None):
    resolved, _ = p._resolve_transitive_references([start], cat, max_references=limit)
    return resolved


# root mentions three references, each reached once, so under a cut at
# depth 1 the link ranking falls through to the id and similarity alone
# separates them.
def _three(**overrides):
    cat = {
        "root": _ref("root", ["aa-far", "mm-mid", "zz-near"], description="payments retries"),
        "aa-far": _ref("aa-far"), "mm-mid": _ref("mm-mid"), "zz-near": _ref("zz-near"),
    }
    cat.update(overrides)
    return cat


class TestTheNearestSurvive:
    def test_a_near_reference_beats_a_more_mentioned_one(self):
        # Depth 1: p1, p2.  Depth 2: both mention "aa-shared", only p1
        # mentions "zz-near".  One slot left at depth 2; the near one wins.
        cat = {
            "root": _ref("root", ["p1", "p2"]),
            "p1": _ref("p1", ["aa-shared", "zz-near"]),
            "p2": _ref("p2", ["aa-shared"]),
            "aa-shared": _ref("aa-shared"), "zz-near": _ref("zz-near"),
        }
        p, _ = _plugin({"aa-shared": 0.1, "zz-near": 0.9, "p1": 0.5, "p2": 0.5})
        assert _resolve(p, cat, "root", limit=4) == ["root", "p1", "p2", "zz-near"]

    def test_a_declared_dependency_still_comes_first(self):
        cat = _three(root=_ref("root", ["aa-far", "mm-mid", "zz-near"],
                               [ReferenceLink("aa-far", "depends-on")]))
        p, _ = _plugin({"aa-far": 0.1, "mm-mid": 0.5, "zz-near": 0.9})
        assert _resolve(p, cat, "root", limit=3) == ["root", "aa-far", "zz-near"]

    def test_the_truncation_record_names_similarity(self):
        p, _ = _plugin({"aa-far": 0.1, "mm-mid": 0.5, "zz-near": 0.9})
        _resolve(p, _three(), "root", limit=2)
        assert p._last_transitive_truncation["ranked_by"] == FRONTIER_RANKING_SIMILARITY


class TestOnlyWhenCut:
    def test_unbounded_embeds_nothing(self):
        p, provider = _plugin({"aa-far": 0.1, "mm-mid": 0.5, "zz-near": 0.9})
        assert sorted(_resolve(p, _three(), "root")) == ["aa-far", "mm-mid", "root", "zz-near"]
        assert provider.texts == []

    def test_room_for_every_candidate_embeds_nothing(self):
        p, provider = _plugin({"aa-far": 0.1, "mm-mid": 0.5, "zz-near": 0.9})
        _resolve(p, _three(), "root", limit=4)
        assert provider.texts == []


class TestPartialIndex:
    def test_an_unindexed_candidate_falls_back_to_links(self):
        # "aa-far" has no vector: similarity cannot compare it, so the
        # depth is ranked by links (here: by id) and the record says so.
        p, _ = _plugin({"mm-mid": 0.5, "zz-near": 0.9})
        assert _resolve(p, _three(), "root", limit=2) == ["root", "aa-far"]
        assert p._last_transitive_truncation["ranked_by"] == FRONTIER_RANKING

    def test_no_index_ranks_by_links(self):
        p = ReferencesPlugin()
        assert _resolve(p, _three(), "root", limit=2) == ["root", "aa-far"]
        assert p._last_transitive_truncation["ranked_by"] == FRONTIER_RANKING


class TestTheQuery:
    def test_the_selection_is_embedded_from_its_metadata(self):
        p, provider = _plugin({"aa-far": 0.1, "mm-mid": 0.5, "zz-near": 0.9})
        _resolve(p, _three(), "root", limit=2)
        assert len(provider.texts) == 1
        assert "payments retries" in provider.texts[0]
