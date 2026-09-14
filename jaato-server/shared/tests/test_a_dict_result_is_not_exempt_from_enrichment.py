"""A dict tool result reaches the enrichment chain whatever its fields are named.

Regression tests for #922.  ``store_memory`` returns

    {"status": "success", "memory_id": "mem_...", "message": "Stored memory: ...",
     "tags": [...], "maturity": "raw", "confidence": 0.9, "scope": "project"}

and the session enriched dict results only through a six-name field
allowlist (``result``/``content``/``stdout``/``output``/``text``/``data``)
with a 100-character floor.  ``message`` is in neither set and the observed
message was 83 characters, so ``memory`` and ``references`` — the only two
plugins implementing ``enrich_tool_result`` — never ran on the pairing they
exist for, with nothing in any log to say why.

The session now renders the dict's scalar fields as a text view, runs the
chain once over it, and writes back what enrichment returned.
"""

from typing import Any, Dict, Tuple

import pytest

from shared.tool_result_builder import (
    ENRICHMENT_FALLBACK_KEY,
    apply_text_view_enrichment,
    pick_anchor_field,
    tool_result_text_view,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/tool_result_builder.py",
        find=(
            "    anchor = pick_anchor_field(result_dict)\n"
            "\n"
            "    lines: List[str] = []"
        ),
        replace=(
            "    anchor = pick_anchor_field(result_dict)\n"
            "    if anchor not in TEXT_FIELD_PREFERENCE:\n"
            '        return "", "", None\n'
            "\n"
            "    lines: List[str] = []"
        ),
        test="test_store_memory_is_enriched",
        because="a dict result whose text lives outside the six well-known "
                "field names never reaching the enrichment chain at all",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find=(
            "        header, body, anchor = _tool_result_text_view_impl(enriched_dict)\n"
            "        text_view = header + body\n"
        ),
        replace=(
            "        header, body, anchor = _tool_result_text_view_impl(enriched_dict)\n"
            '        text_view = header + body if len(body) >= 100 else ""\n'
        ),
        test="test_a_short_result_is_still_enriched",
        because="the 100-character floor that excluded the 83-character "
                "store_memory message the defect was found on",
    ),
]


STORE_MEMORY_RESULT: Dict[str, Any] = {
    "status": "success",
    "memory_id": "mem_20260909_carburador",
    "message": "Stored memory: rebuilding the Vespa carburettor jet by jet",
    "tags": ["vespa", "carburador", "restauracion"],
    "maturity": "raw",
    "confidence": 0.9,
    "scope": "project",
}


def _view(result: Dict[str, Any]) -> str:
    header, body, _ = tool_result_text_view(result)
    return header + body


# ---------------------------------------------------------------- text view


class TestTextView:

    def test_the_message_field_reaches_the_chain(self):
        """The defect: `message` is not one of the six names."""
        assert "Stored memory: rebuilding the Vespa carburettor" in _view(
            STORE_MEMORY_RESULT
        )

    def test_the_tags_reach_the_chain_too(self):
        """The 3/3 tag match in #922 lives in a list, not a text field."""
        view = _view(STORE_MEMORY_RESULT)
        assert "tags: vespa, carburador, restauracion" in view

    def test_short_text_is_not_excluded(self):
        """No length floor — the message that failed was 83 characters."""
        assert "ok" in _view({"message": "ok"})

    def test_each_field_is_its_own_line(self):
        """Enrichers match on line/sentence segments: joining unrelated
        fields onto one line would manufacture co-occurrences (see
        memory/tests/test_enrichment_surfaces.py::TestSegmentationAvoids...)."""
        header, _, _ = tool_result_text_view(STORE_MEMORY_RESULT)
        assert "status: success\n" in header
        assert "maturity: raw\n" in header
        assert "success" not in header.split("\n")[1]

    def test_anchor_is_the_prose_field(self):
        header, body, anchor = tool_result_text_view(STORE_MEMORY_RESULT)
        assert anchor == "message"
        assert body == STORE_MEMORY_RESULT["message"]
        assert "message:" not in header

    def test_well_known_names_still_win(self):
        """A tool already using a conventional name keeps its hint there,
        even when another field holds a longer string."""
        result = {"content": "short", "note": "x" * 500}
        assert pick_anchor_field(result) == "content"

    def test_the_wordiest_field_anchors_when_no_convention_matches(self):
        assert pick_anchor_field({"summary": "x" * 40, "scope": "project"}) == "summary"

    def test_a_one_word_status_never_outweighs_a_sentence(self):
        """Counting characters would anchor on `success` (7) over `a\nb\nc` (5)."""
        assert pick_anchor_field({"status": "success", "message": "a\nb\nc"}) == "message"

    def test_nested_payloads_are_left_to_the_trait_path(self):
        """Dicts and lists of dicts are what TRAIT_GREPPABLE_CONTENT hands
        an enricher whole; rendering them here would duplicate that."""
        view = _view({"rows": [{"secret": "nested"}], "message": "hello"})
        assert "nested" not in view
        assert "hello" in view

    def test_internal_keys_are_not_rendered(self):
        view = _view({"_lsp_diagnostics": "ERROR at line 5", "message": "hello"})
        assert "ERROR at line 5" not in view

    def test_a_dict_with_no_text_yields_an_empty_view(self):
        header, body, anchor = tool_result_text_view({"rows": [{"a": 1}]})
        assert (header, body, anchor) == ("", "", None)

    def test_a_context_field_is_capped(self):
        header, _, _ = tool_result_text_view(
            {"message": "hi", "log": "y" * 5000}
        )
        assert len(header) < 1000


# --------------------------------------------------------------- write-back


class TestWriteBack:

    def test_an_appended_hint_lands_on_the_anchor_field(self):
        result = dict(STORE_MEMORY_RESULT)
        header, body, anchor = tool_result_text_view(result)
        hint = "\n\n💡 **Available Memories** — fetch them in ONE call:"

        assert apply_text_view_enrichment(
            result, header, body, anchor, header + body + hint
        )

        assert result["message"] == STORE_MEMORY_RESULT["message"] + hint
        # The header fields are context only — never written back.
        assert result["status"] == "success"
        assert result["tags"] == ["vespa", "carburador", "restauracion"]

    def test_an_in_place_rewrite_replaces_only_the_body(self):
        """`references` expands an @ref-id mention inside the text."""
        result = {"status": "success", "message": "see @auto-testvespa1"}
        header, body, anchor = tool_result_text_view(result)

        apply_text_view_enrichment(
            result, header, body, anchor,
            header + "see [Vespa restoration notes]",
        )

        assert result["message"] == "see [Vespa restoration notes]"
        assert result["status"] == "success"

    def test_an_unchanged_result_reports_no_change(self):
        result = dict(STORE_MEMORY_RESULT)
        header, body, anchor = tool_result_text_view(result)
        assert not apply_text_view_enrichment(
            result, header, body, anchor, header + body
        )
        assert result == STORE_MEMORY_RESULT

    def test_a_whole_view_rewrite_replaces_the_anchor(self):
        """`result_grep` returns a JSON envelope, header and all."""
        result = {"status": "success", "message": "a\nb\nc"}
        header, body, anchor = tool_result_text_view(result)
        envelope = '{"_grep_filtered": {"lines_shown": 1}, "content": "b"}'

        apply_text_view_enrichment(result, header, body, anchor, envelope)

        assert result["message"] == envelope

    def test_without_a_string_field_the_addition_lands_under_its_own_key(self):
        result = {"count": 3, "tags": ["vespa"]}
        header, body, anchor = tool_result_text_view(result)
        assert anchor is None

        apply_text_view_enrichment(
            result, header, body, anchor, header + body + "\n💡 hint",
        )

        assert result[ENRICHMENT_FALLBACK_KEY] == "💡 hint"
        assert result["count"] == 3


# ------------------------------------------------------- session dispatch


class _Enricher:
    """Stand-in for a subscribed enrichment plugin: records what it saw."""

    def __init__(self, hint: str = "\n💡 hint"):
        self.seen: list = []
        self._hint = hint

    def enrich_tool_result(self, tool_name, result, output_callback=None,
                           terminal_width=None, tool_args=None):
        self.seen.append((tool_name, result))
        return _Enrichment(result + self._hint, {"memory": {"memory_matches": 1}})


class _Enrichment:
    def __init__(self, result: str, metadata: Dict[str, Any]):
        self.result = result
        self.metadata = metadata


class _Registry:
    def __init__(self, enricher: _Enricher, traits=frozenset()):
        self._enricher = enricher
        self._traits = traits

    def get_tool_traits(self, tool_name: str) -> frozenset:
        return self._traits

    def enrich_tool_result(self, tool_name, result, **kwargs):
        return self._enricher.enrich_tool_result(tool_name, result, **kwargs)


@pytest.fixture
def session_dispatch():
    """Call ``JaatoSession._enrich_tool_result_dict`` unbound, over a stub
    session carrying just the attributes that method touches."""
    from shared.jaato_session import JaatoSession

    def _run(result_dict: Dict[str, Any], enricher: _Enricher,
             traits=frozenset()) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        class _Stub:
            _current_output_callback = None
            _terminal_width = 80
            _current_turn_span = None

            def __init__(self):
                self._runtime = type("_RT", (), {"registry": _Registry(enricher, traits)})()
                self.traces: list = []

            def _trace(self, msg: str) -> None:
                self.traces.append(msg)

            def _check_and_pin_reference(self, metadata, text) -> None:
                pass

            _emit_enrichment_telemetry = JaatoSession._emit_enrichment_telemetry
            _enrich_tool_result_dict = JaatoSession._enrich_tool_result_dict

        stub = _Stub()
        enriched, metadata = stub._enrich_tool_result_dict(
            "store_memory", result_dict, tool_args={}
        )
        return enriched, metadata, stub.traces

    return _run


class TestSessionDispatch:

    def test_store_memory_is_enriched(self, session_dispatch):
        """#922 end to end: the enricher runs, and its hint reaches the
        model on the field the tool actually used."""
        enricher = _Enricher()
        enriched, metadata, _ = session_dispatch(dict(STORE_MEMORY_RESULT), enricher)

        assert enricher.seen, "the enrichment chain never ran for store_memory"
        _, seen_text = enricher.seen[0]
        assert "carburettor" in seen_text and "carburador" in seen_text
        assert enriched["message"].endswith("💡 hint")
        assert metadata == {"memory": {"memory_matches": 1}}

    def test_a_short_result_is_still_enriched(self, session_dispatch):
        """The message the defect was found on measured 83 characters, and a
        tag match does not need 100 characters of context to be valid."""
        enricher = _Enricher()
        enriched, _, _ = session_dispatch(
            {"message": "Stored memory: a note"}, enricher
        )

        assert enricher.seen, "a short dict result never reached the chain"
        assert enriched["message"].endswith("\U0001F4A1 hint")

    def test_the_chain_runs_once_per_result(self, session_dispatch):
        """The old loop invoked the chain per matching field, so a dict with
        both `content` and `output` got two hint blocks."""
        enricher = _Enricher()
        session_dispatch(
            {"content": "a" * 200, "output": "b" * 200}, enricher
        )
        assert len(enricher.seen) == 1

    def test_a_textless_result_is_skipped_out_loud(self, session_dispatch):
        """The silence in #922 was half the defect: a skip now says so."""
        enricher = _Enricher()
        enriched, metadata, traces = session_dispatch({"rows": [{"a": 1}]}, enricher)

        assert not enricher.seen
        assert enriched == {"rows": [{"a": 1}]}
        assert metadata == {}
        assert any("ENRICH_SKIP" in t for t in traces)

    def test_a_run_is_traced(self, session_dispatch):
        _, _, traces = session_dispatch(dict(STORE_MEMORY_RESULT), _Enricher())
        assert any(t.startswith("ENRICH [store_memory]") for t in traces)

    def test_the_trait_path_still_gets_the_whole_json(self, session_dispatch):
        from jaato_sdk.plugins.model_provider.types import TRAIT_FILE_WRITER

        enricher = _Enricher()
        session_dispatch(
            {"path": "X.java", "files_modified": ["X.java"]},
            enricher,
            traits=frozenset({TRAIT_FILE_WRITER}),
        )
        _, seen_text = enricher.seen[0]
        assert seen_text.startswith("{") and '"path": "X.java"' in seen_text
