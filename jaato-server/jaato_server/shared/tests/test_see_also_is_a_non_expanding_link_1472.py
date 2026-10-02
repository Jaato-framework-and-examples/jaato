"""``see-also`` is a declared link that a selection OFFERS and never pulls in (#1472).

A person-approved catalog design needed a fifth relation for two real edges
the four did not express: coexisting alternatives (generated REST clients vs
``invokeApi``) and two sides of one topic (what a service throws vs what a
client catches).  Mapping them to ``elaborates`` says something false;
dropping them loses the navigation; and every ``proposeReference`` carrying
``see-also`` was refused.

1. ``proposeReference`` accepts ``see-also``.
2. A selection does NOT expand a ``see-also`` target -- even when the body
   mentions it, since the declared edge wins for its pair -- and offers it
   under ``related``, the path ``elaborates`` already uses.
3. It is not a ``depends-on`` parent in the cap ranking.
4. ``jaato-scaffold validate`` accepts it.
5. The tool schema's ``enum`` is ``LINK_RELS``, and ``explain`` renders the
   same vocabulary from the module's own table.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from jaato_server.shared.plugins.references import create_plugin
from jaato_server.shared.plugins.references.links import (
    LINK_RELS, REL_DOCS, LinkIndex, parse_links, rank_frontier)
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.scaffold.validate import _check_reference_links
from jaato_server.shared.tests.reversion import Reversion

_LINKS = "jaato-server/jaato_server/shared/plugins/references/links.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"

REVERSIONS = [
    Reversion(
        target=_LINKS,
        find='    REL_SEE_ALSO: {\n',
        replace='    "x-see-also-removed": {\n',
        because=(
            "without see-also in the vocabulary every proposal carrying it is "
            "refused, which is the reported defect"
        ),
        test="TestAProposalAcceptsSeeAlso::test_propose_reference_accepts_see_also",
    ),
    Reversion(
        target=_LINKS,
        find='OFFERED_RELS = frozenset({REL_ELABORATES, REL_SEE_ALSO})\n',
        replace='OFFERED_RELS = frozenset({REL_ELABORATES})\n',
        because=(
            "a see-also target the selection never offers is a link the "
            "agent cannot follow; the relation is pure navigation"
        ),
        test="TestASelectionOffersAndNeverExpands::test_a_mentioned_see_also_target_is_offered_not_expanded",
    ),
    Reversion(
        target=_LINKS,
        find='        "selection": "not expanded; offered as an optional neighbour (related)",\n'
             '        "expands": False},\n    REL_SUPERSEDES',
        replace='        "selection": "not expanded; offered as an optional neighbour (related)",\n'
                '        "expands": True},\n    REL_SUPERSEDES',
        because=(
            "a see-also that expands pulls a whole neighbourhood into every "
            "selection and counts as a depends-on parent in the cap ranking, "
            "the opposite of 'costs nothing unless followed'"
        ),
        test="TestASelectionOffersAndNeverExpands::test_a_mentioned_see_also_target_is_offered_not_expanded",
    ),
    Reversion(
        target=_PLUGIN,
        find='                                    "rel": {"type": "string", "enum": list(LINK_RELS)},\n',
        replace='                                    "rel": {"type": "string", "enum": ["depends-on", "elaborates", "supersedes", "contradicts"]},\n',
        because=(
            "a schema enum restated beside the vocabulary drifts from it, and "
            "the model is then told a relation the plugin accepts does not exist"
        ),
        test="TestOneVocabulary::test_the_schema_enum_is_link_rels",
    ),
]


def _inline(ref_id: str, content: str = "body", **extra: Any) -> Dict[str, Any]:
    src = {"id": ref_id, "name": ref_id.upper(), "description": f"about {ref_id}",
           "type": "inline", "mode": "selectable", "content": content}
    src.update(extra)
    return src


@pytest.fixture
def make():
    made = []

    def _make(sources):
        p = create_plugin()
        p.initialize({"sources": sources, "transitive_injection": True, "exclude_tools": []})
        made.append(p)
        return p

    yield _make
    for p in made:
        p.shutdown()


class TestAProposalAcceptsSeeAlso:
    def test_propose_reference_accepts_see_also(self, tmp_path):
        plugin = ReferencesPlugin()
        plugin._workspace_path = str(tmp_path)
        plugin._sources = []
        result = plugin._execute_propose({
            "id": "rest-clients", "name": "Generated REST clients", "content": "# clients",
            "links": [{"to": "invoke-api", "rel": "see-also",
                       "note": "a coexisting alternative"}],
        })
        assert result.get("success") is True, result
        claim_files = list((tmp_path / ".jaato" / "references-claims").glob("*.json"))
        assert len(claim_files) == 1
        claim = json.loads(claim_files[0].read_text())
        assert claim["reference"]["links"] == [{"to": "invoke-api", "rel": "see-also",
                                   "note": "a coexisting alternative"}]


class TestASelectionOffersAndNeverExpands:
    def test_a_mentioned_see_also_target_is_offered_not_expanded(self, make):
        p = make([
            _inline("throws", "What a service may throw; see catches.",
                    links=[{"to": "catches", "rel": "see-also"}]),
            _inline("catches"),
        ])
        result = p._execute_select({"ids": ["throws"]})
        assert sorted(s["id"] for s in result["sources"]) == ["throws"]
        assert result["related"] == [{"id": "catches", "rel": "see-also", "from": "throws"}]

    def test_it_is_not_a_depends_on_parent_in_the_ranking(self):
        index = LinkIndex([
            SimpleNamespace(id="p", links=parse_links([{"to": "b", "rel": "see-also"}])),
            SimpleNamespace(id="a", links=[]), SimpleNamespace(id="b", links=[]),
        ])
        # Equal reach; only a declared depends-on would lift b over a.
        assert rank_frontier({"a": {"p"}, "b": {"p"}}, index) == ["a", "b"]

    def test_the_model_reads_it(self):
        src = ReferenceSourceFromDict({"id": "x", "name": "X", "description": "d", "type": "inline",
                                       "mode": "selectable", "content": "c",
                                       "links": [{"to": "y", "rel": "see-also"}]})
        assert "see also `y`" in src.to_instruction()


def ReferenceSourceFromDict(d):  # noqa: N802 -- local helper
    from jaato_server.shared.plugins.references.models import ReferenceSource
    return ReferenceSource.from_dict(d)


class TestValidateAccepts:
    def test_validate_accepts_see_also(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        root = tmp_path / "cfg"
        (root / "references").mkdir(parents=True)
        for ref in ({"id": "a", "links": [{"to": "b", "rel": "see-also"}]}, {"id": "b"}):
            base = {"name": ref["id"], "description": "d", "type": "inline", "content": "x"}
            (root / "references" / f"{ref['id']}.json").write_text(json.dumps({**base, **ref}))
        out: List[Any] = []
        _check_reference_links(str(root), out)
        assert out == []


class TestOneVocabulary:
    def test_the_schema_enum_is_link_rels(self):
        p = ReferencesPlugin()
        schema = next(s for s in p.get_tool_schemas() if s.name == "proposeReference")
        rel = schema.parameters["properties"]["links"]["items"]["properties"]["rel"]
        assert rel["enum"] == list(LINK_RELS)
        assert "see-also" in LINK_RELS

    def test_explain_renders_the_same_table(self):
        from jaato_server.shared.scaffold.explain import _reference_link_rels
        assert [d["rel"] for d in _reference_link_rels()] == list(LINK_RELS)
        assert {d["rel"]: d["expands"] for d in _reference_link_rels()} == {
            r: d["expands"] for r, d in REL_DOCS.items()}
