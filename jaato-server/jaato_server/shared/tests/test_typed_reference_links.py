"""A reference may DECLARE typed edges to others, and each ``rel`` decides the traversal.

The wikiLLM brainstorm (``docs/design/wikillm-brainstorm.md`` §5, Seam 3).
Until now the only edge between references was a MENTION: an id or path
that happens to appear in a body.  Nobody authors it, renaming an id
silently deletes every inbound one, and "mentioned" is the only relation.
A reference can now carry ``links: [{to, rel, note?}]`` with a closed
vocabulary, and the relation decides the expansion policy:

1. ``depends-on`` is **expanded**: selecting the source pulls the target
   in, even when the body never mentions it and even when the source has
   no readable body at all (a URL reference).
2. ``elaborates`` is **not expanded**, and wins over a mention of the same
   target: it is offered on the selection as ``related`` instead.
3. ``supersedes`` is declared on the NEWER reference and **routes** the
   older one to it: a selection or an expansion that reaches the older
   reference gets the newer one instead, never both, and says so.
4. ``contradicts`` is **listed only**: never expanded, never hinted to a
   working agent.
5. A **dangling** edge (target not in the catalog) is kept and marked,
   never dropped.  ``listReferences`` shows each reference's edges and the
   reverse index (``linked_from``).
6. A **malformed** edge is a validation error, not a quiet no-op.
7. The model reads the declared edges where it reads the reference: the
   instruction carries a ``**Links**`` line.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from jaato_server.shared.plugins.references import create_plugin
from jaato_server.shared.plugins.references.config_loader import validate_reference_file
from jaato_server.shared.plugins.references.links import LinkIndex, link_errors
from jaato_server.shared.plugins.references.models import ReferenceSource
from jaato_server.shared.tests.reversion import Reversion

_LINKS = "jaato-server/jaato_server/shared/plugins/references/links.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_MODELS = "jaato-server/jaato_server/shared/plugins/references/models.py"

REVERSIONS = [
    Reversion(
        target=_LINKS,
        find='    base = (set(mentions or ()) - index.declared_targets(source_id)) | expanding\n',
        replace='    base = set(mentions or ()) | expanding\n',
        because=(
            "a declared edge must win over the mention of the same target; "
            "without it an 'elaborates' target the body happens to name is "
            "pulled in anyway, and the declaration changes nothing"
        ),
        test="TestElaboratesIsAHint::test_a_mentioned_elaborates_target_is_not_expanded",
    ),
    Reversion(
        target=_LINKS,
        find='    if mentions is None and not expanding:\n        return None\n',
        replace='    if mentions is None:\n        return None\n',
        because=(
            "a reference with no readable body (a URL) can only reach its "
            "prerequisites through a declared depends-on; dropping it there "
            "makes the declaration inert exactly where inference cannot help"
        ),
        test="TestDependsOnExpands::test_a_url_reference_expands_its_declared_dependency",
    ),
    Reversion(
        target=_LINKS,
        find='    return {index.current_version(n) for n in base}\n',
        replace='    return base\n',
        because=(
            "an expansion that reaches a superseded reference would hand the "
            "agent the outdated document the catalog says was replaced"
        ),
        test="TestSupersedesRoutes::test_an_expansion_reaching_the_old_reference_gets_the_new_one",
    ),
    Reversion(
        target=_PLUGIN,
        find='        matched, superseded = self._route_superseded(matched, links)\n',
        replace='        superseded = []\n',
        because=(
            "a request for a superseded reference must be routed to the "
            "newer one and say so, not quietly serve the old one"
        ),
        test="TestSupersedesRoutes::test_selecting_the_old_reference_gets_the_new_one_and_says_so",
    ),
    Reversion(
        target=_LINKS,
        find='            if link.to not in self.ids:\n                entry["dangling"] = True\n',
        replace='            if link.to not in self.ids:\n                continue\n',
        because=(
            "an edge to a reference that is gone is exactly what a curator "
            "needs to see; dropping it silently is the defect declared edges "
            "exist to end"
        ),
        test="TestDanglingIsKept::test_listing_marks_a_dangling_edge",
    ),
    Reversion(
        target=_LINKS,
        find='        if link.get("rel") not in LINK_RELS:\n'
             '            errors.append(f"{where}.rel must be one of: {\', \'.join(LINK_RELS)}")\n',
        replace='',
        because=(
            "an unknown relation is one nothing acts on, and a reader would "
            "trust it; it must be refused at validation"
        ),
        test="TestMalformedIsRefused::test_an_unknown_rel_is_an_error",
    ),
    Reversion(
        target=_MODELS,
        find='            return "\\n\\n".join([f"### {self.name}", *self._links_lines(), f"{self.content}"])\n',
        replace='            return f"### {self.name}\\n\\n{self.content}"\n',
        because=(
            "the model navigates by what it reads; declared edges it never "
            "sees can only steer the framework, not the agent"
        ),
        test="TestTheModelReadsTheEdges::test_the_instruction_names_the_links",
    ),
]


def _inline(ref_id: str, content: str = "body", **extra: Any) -> Dict[str, Any]:
    src = {
        "id": ref_id,
        "name": ref_id.upper(),
        "description": f"about {ref_id}",
        "type": "inline",
        "mode": "selectable",
        "content": content,
    }
    src.update(extra)
    return src


def _plugin(sources: List[Dict[str, Any]]):
    plugin = create_plugin()
    plugin.initialize({
        "sources": sources,
        "transitive_injection": True,
        "exclude_tools": [],
    })
    return plugin


@pytest.fixture
def make():
    made = []

    def _make(sources):
        p = _plugin(sources)
        made.append(p)
        return p

    yield _make
    for p in made:
        p.shutdown()


def _ids(result: Dict[str, Any]) -> List[str]:
    return sorted(s["id"] for s in result["sources"])


class TestDependsOnExpands:
    def test_a_declared_dependency_is_pulled_in_without_a_mention(self, make):
        p = make([
            _inline("guide", "No mention here.", links=[{"to": "glossary", "rel": "depends-on"}]),
            _inline("glossary"),
        ])
        result = p._execute_select({"ids": ["guide"]})
        assert _ids(result) == ["glossary", "guide"]

    def test_a_url_reference_expands_its_declared_dependency(self, make):
        p = make([
            {"id": "api", "name": "API", "description": "remote", "type": "url",
             "mode": "selectable", "url": "https://example.invalid/api",
             "links": [{"to": "glossary", "rel": "depends-on"}]},
            _inline("glossary"),
        ])
        result = p._execute_select({"ids": ["api"]})
        assert "glossary" in _ids(result)


class TestElaboratesIsAHint:
    def test_a_mentioned_elaborates_target_is_not_expanded(self, make):
        p = make([
            _inline("guide", "See deep-dive for more.",
                    links=[{"to": "deep-dive", "rel": "elaborates"}]),
            _inline("deep-dive"),
        ])
        result = p._execute_select({"ids": ["guide"]})
        assert _ids(result) == ["guide"]
        assert result["related"] == [{"id": "deep-dive", "rel": "elaborates", "from": "guide"}]

    def test_an_undeclared_mention_still_expands(self, make):
        # Inference is kept beside declaration.
        p = make([_inline("guide", "See deep-dive for more."), _inline("deep-dive")])
        assert _ids(p._execute_select({"ids": ["guide"]})) == ["deep-dive", "guide"]


class TestSupersedesRoutes:
    def test_selecting_the_old_reference_gets_the_new_one_and_says_so(self, make):
        p = make([
            _inline("adr-1"),
            _inline("adr-2", links=[{"to": "adr-1", "rel": "supersedes"}]),
        ])
        result = p._execute_select({"ids": ["adr-1"]})
        assert _ids(result) == ["adr-2"]
        assert result["superseded"] == [{"requested": "adr-1", "replaced_by": "adr-2"}]
        assert "adr-1" not in p.get_selected_ids()

    def test_an_expansion_reaching_the_old_reference_gets_the_new_one(self, make):
        p = make([
            _inline("guide", "Per adr-1, we do X."),
            _inline("adr-1"),
            _inline("adr-2", links=[{"to": "adr-1", "rel": "supersedes"}]),
        ])
        result = p._execute_select({"ids": ["guide"]})
        assert _ids(result) == ["adr-2", "guide"]

    def test_an_ambiguous_succession_is_routed_nowhere(self):
        a = ReferenceSource.from_dict(_inline("old"))
        b = ReferenceSource.from_dict(_inline("new-a", links=[{"to": "old", "rel": "supersedes"}]))
        c = ReferenceSource.from_dict(_inline("new-b", links=[{"to": "old", "rel": "supersedes"}]))
        assert LinkIndex([a, b, c]).current_version("old") == "old"

    def test_a_cycle_terminates(self):
        a = ReferenceSource.from_dict(_inline("a", links=[{"to": "b", "rel": "supersedes"}]))
        b = ReferenceSource.from_dict(_inline("b", links=[{"to": "a", "rel": "supersedes"}]))
        index = LinkIndex([a, b])
        assert (index.current_version("a"), index.current_version("b")) == ("a", "b")


class TestContradictsIsListedOnly:
    def test_never_expanded_never_hinted(self, make):
        p = make([
            _inline("claim", "Unlike rebuttal, we think Y.",
                    links=[{"to": "rebuttal", "rel": "contradicts"}]),
            _inline("rebuttal"),
        ])
        result = p._execute_select({"ids": ["claim"]})
        assert _ids(result) == ["claim"]
        assert "related" not in result

    def test_listed_with_its_reverse(self, make):
        p = make([
            _inline("claim", links=[{"to": "rebuttal", "rel": "contradicts"}]),
            _inline("rebuttal"),
        ])
        listing = {e["id"]: e for e in p._execute_list({})["sources"]}
        assert listing["claim"]["links"] == [{"to": "rebuttal", "rel": "contradicts"}]
        assert listing["rebuttal"]["linked_from"] == [{"from": "claim", "rel": "contradicts"}]


class TestDanglingIsKept:
    def test_listing_marks_a_dangling_edge(self, make):
        p = make([_inline("guide", links=[{"to": "gone", "rel": "depends-on"}])])
        entry = p._execute_list({})["sources"][0]
        assert entry["links"] == [{"to": "gone", "rel": "depends-on", "dangling": True}]

    def test_a_dangling_dependency_is_not_selected(self, make):
        p = make([_inline("guide", links=[{"to": "gone", "rel": "depends-on"}])])
        assert _ids(p._execute_select({"ids": ["guide"]})) == ["guide"]


class TestMalformedIsRefused:
    def test_an_unknown_rel_is_an_error(self):
        ok, errors, _ = validate_reference_file({
            "id": "a", "name": "A", "description": "d", "type": "inline", "content": "x",
            "links": [{"to": "b", "rel": "see-also"}],
        })
        assert not ok
        assert any("rel must be one of" in e for e in errors)

    def test_other_shapes(self):
        assert link_errors(None) == []
        assert link_errors("b") == ["'links' must be an array"]
        assert any("names the reference itself" in e
                   for e in link_errors([{"to": "a", "rel": "depends-on"}], source_id="a"))
        assert any("unknown keys" in e
                   for e in link_errors([{"to": "b", "rel": "depends-on", "weight": 2}]))

    def test_loading_is_lenient(self):
        src = ReferenceSource.from_dict(_inline("a", links=[
            {"to": "b", "rel": "see-also"}, {"to": "c", "rel": "depends-on"},
        ]))
        assert [(l.to, l.rel) for l in src.links] == [("c", "depends-on")]


class TestTheModelReadsTheEdges:
    def test_the_instruction_names_the_links(self):
        src = ReferenceSource.from_dict(_inline("adr-2", links=[
            {"to": "adr-1", "rel": "supersedes"}, {"to": "glossary", "rel": "depends-on"},
        ]))
        assert "**Links**: supersedes `adr-1`; depends on `glossary`" in src.to_instruction()

    def test_round_trip(self):
        data = _inline("a", links=[{"to": "b", "rel": "elaborates", "note": "deeper"}])
        assert ReferenceSource.from_dict(data).to_dict()["links"] == data["links"]
