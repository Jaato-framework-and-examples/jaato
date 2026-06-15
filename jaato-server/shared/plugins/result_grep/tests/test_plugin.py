"""Tests for the result_grep plugin (model-directed tool-result filtering)."""

import json

import pytest

from shared.plugins.result_grep import create_plugin, PLUGIN_KIND, PLUGIN_TIER
from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    TRAIT_GREPPABLE_CONTENT,
)


class _FakeSession:
    """Minimal stand-in; the plugin only uses identity for filter keying."""


class _FakeRegistry:
    """Exposes a fixed tool->traits map for greppability discovery tests."""

    def __init__(self, traits_by_tool):
        self._traits = traits_by_tool

    def list_exposed(self):
        return list(self._traits.keys())

    def get_tool_traits(self, name):
        return self._traits.get(name, frozenset())


@pytest.fixture
def plugin():
    p = create_plugin()
    p.set_session(_FakeSession())
    return p


# --------------------------------------------------------------- declaration


def test_plugin_kind_and_tier():
    assert PLUGIN_KIND == "tool"
    assert PLUGIN_TIER == "runner"


def test_exposes_three_tools_and_subscribes_last(plugin):
    names = {s.name for s in plugin.get_tool_schemas()}
    assert names == {"grep_mode_start", "grep_mode_stop", "grep_mode_status"}
    assert plugin.subscribes_to_tool_result_enrichment() is True
    # Runs after LSP(30)/artifact_tracker(20) so it filters the enriched payload.
    assert plugin.get_tool_result_enrichment_priority() > 30


def test_all_tools_auto_approved(plugin):
    assert set(plugin.get_auto_approved_tools()) == {
        "grep_mode_start",
        "grep_mode_stop",
        "grep_mode_status",
    }


# ----------------------------------------------------------- passthrough/off


def test_passthrough_when_mode_off(plugin):
    raw = json.dumps({"a": 1, "b": [1, 2, 3]})
    out = plugin.enrich_tool_result("call_service", raw)
    assert out.result == raw  # byte-identical: zero behavior change when off


def test_own_tools_never_filtered(plugin):
    plugin._exec_start({"pattern": "zzz"})
    raw = json.dumps({"active": True})
    out = plugin.enrich_tool_result("grep_mode_status", raw)
    assert out.result == raw


# --------------------------------------------------------- json-dict contract


def test_json_dict_result_filtered_and_parses_back(plugin):
    """The full-dict enrichment path requires a JSON object back out."""
    plugin._exec_start({"pattern": "spring-data-commons"})
    big = {
        "status": 200,
        "body": {
            "docs": [
                {"id": "io.x:spring-data-commons:1.0"},
                {"id": "io.x:other-thing:2.0"},
                {"id": "io.x:spring-data-web:3.0"},
            ]
        },
    }
    out = plugin.enrich_tool_result("call_service", json.dumps(big))
    parsed = json.loads(out.result)  # must be valid JSON object
    assert isinstance(parsed, dict)
    assert parsed["_grep_filtered"]["pattern"] == "spring-data-commons"
    assert "spring-data-commons" in parsed["content"]
    assert "other-thing" not in parsed["content"]
    assert parsed["_grep_filtered"]["lines_shown"] < parsed["_grep_filtered"]["lines_total"]


def test_context_lines_before_after(plugin):
    plugin._exec_start({"pattern": "NEEDLE", "before": 1, "after": 1})
    doc = {"lines": ["a", "b", "NEEDLE", "d", "e"]}
    out = plugin.enrich_tool_result("call_service", json.dumps(doc, indent=2))
    parsed = json.loads(out.result)
    # pretty-printed; the NEEDLE line plus one neighbour each side is retained.
    assert "NEEDLE" in parsed["content"]
    assert '"b"' in parsed["content"] or "b" in parsed["content"]


def test_lsp_diagnostics_preserved(plugin):
    plugin._exec_start({"pattern": "zzz-no-match"})
    payload = {"path": "X.java", "_lsp_diagnostics": "ERROR at line 5"}
    out = plugin.enrich_tool_result("updateFile", json.dumps(payload))
    parsed = json.loads(out.result)
    assert parsed["_lsp_diagnostics"] == "ERROR at line 5"


# ------------------------------------------------------------ text-field path


def test_text_result_filtered_into_json_envelope(plugin):
    """Even a plain-text (non-JSON) result is wrapped in a valid JSON envelope."""
    plugin._exec_start({"pattern": "MATCH"})
    text = "line one\nMATCH here\nline three"
    out = plugin.enrich_tool_result("cli", text)
    parsed = json.loads(out.result)  # ALWAYS valid JSON (regression #289)
    assert parsed["_grep_filtered"]["pattern"] == "MATCH"
    assert "MATCH here" in parsed["content"]
    assert "line one" not in parsed["content"]  # non-matching elided


def test_no_match_is_explicit_not_empty(plugin):
    plugin._exec_start({"pattern": "WILLNOTMATCH"})
    out = plugin.enrich_tool_result("cli", "alpha\nbeta\ngamma")
    parsed = json.loads(out.result)
    assert "no lines matched" in parsed["content"]


def test_nonjson_input_still_returns_valid_json_regression_289(plugin):
    """#289: an upstream enricher (LSP) can mangle a result's JSON into non-JSON
    BEFORE result_grep (which runs last). Output must STILL be valid JSON, else
    the session's full-dict json.loads fails and the result passes unfiltered.
    """
    plugin._exec_start({"pattern": "(?i)slf4j"})
    # Simulate LSP@30 having appended markdown diagnostics to a call_service
    # JSON result -> the string result_grep receives is no longer valid JSON.
    mangled = (
        '{"status": 200, "body": {"docs": [{"a": "slf4j-api"}]}}\n'
        "\n## LSP Diagnostics\n- org.slf4j.MDC: cannot resolve symbol\n"
    )
    out = plugin.enrich_tool_result("call_service", mangled)
    parsed = json.loads(out.result)  # MUST NOT raise (the whole point of #289)
    assert "_grep_filtered" in parsed
    assert "slf4j" in parsed["content"].lower()  # the matching line survived


# ------------------------------------------------------------------- scoping


def test_scope_limits_which_tools_filter(plugin):
    plugin._exec_start({"pattern": "x", "tools": ["call_service"]})
    raw = json.dumps({"x": "x marks"})
    # in scope -> filtered (changes)
    assert plugin.enrich_tool_result("call_service", raw).result != raw
    # out of scope -> passthrough
    assert plugin.enrich_tool_result("readFile", raw).result == raw


# --------------------------------------------------------- lifecycle / errors


def test_stop_restores_passthrough(plugin):
    plugin._exec_start({"pattern": "a"})
    plugin._exec_stop({})
    raw = json.dumps({"a": "alpha"})
    assert plugin.enrich_tool_result("call_service", raw).result == raw


def test_pattern_syntax_documented_as_python_re(plugin):
    """The pattern param must name the dialect so the model knows what to send."""
    schema = next(s for s in plugin.get_tool_schemas() if s.name == "grep_mode_start")
    desc = schema.parameters["properties"]["pattern"]["description"]
    assert "Python `re`" in desc
    assert "(?i)" in desc  # case-insensitivity escape hatch is documented


def test_inline_ignorecase_flag_works(plugin):
    """Documented contract: '(?i)' inline flag gives case-insensitive matching."""
    plugin._exec_start({"pattern": "(?i)spring-data"})
    out = plugin.enrich_tool_result("cli", "X SPRING-DATA y\nother")
    assert "SPRING-DATA" in json.loads(out.result)["content"]


def test_line_anchors_are_per_line(plugin):
    """Documented contract: ^/$ anchor to line start/end (re.search per line)."""
    plugin._exec_start({"pattern": "^needle$"})
    out = plugin.enrich_tool_result("cli", "a needle b\nneedle\nc")
    content = json.loads(out.result)["content"]
    # only the exact-line 'needle' matches, not the embedded one
    assert "needle" in content
    assert "a needle b" not in content


def test_invalid_regex_rejected(plugin):
    res = plugin._exec_start({"pattern": "("})
    assert "error" in res
    # filter must NOT have been installed
    raw = json.dumps({"a": 1})
    assert plugin.enrich_tool_result("call_service", raw).result == raw


def test_empty_pattern_rejected(plugin):
    assert "error" in plugin._exec_start({"pattern": ""})


# ------------------------------------------------------- greppability discovery


def test_status_lists_greppable_tools(plugin):
    plugin.set_plugin_registry(
        _FakeRegistry(
            {
                "call_service": frozenset({TRAIT_GREPPABLE_CONTENT}),
                "readFile": frozenset(),
            }
        )
    )
    status = plugin._exec_status({})
    assert status["active"] is False
    assert status["greppable_tools"] == ["call_service"]


def test_start_ack_reports_greppable_and_warns_on_non_greppable(plugin):
    plugin.set_plugin_registry(
        _FakeRegistry({"call_service": frozenset({TRAIT_GREPPABLE_CONTENT})})
    )
    ack = plugin._exec_start({"pattern": "x", "tools": ["readFile"]})
    assert ack["greppable_tools"] == ["call_service"]
    assert "warning" in ack  # readFile is not greppable


# --------------------------------------------------- per-session isolation


def test_filter_is_per_session():
    p = create_plugin()
    s1, s2 = _FakeSession(), _FakeSession()
    p.set_session(s1)
    p._exec_start({"pattern": "x"})
    # switch to a second session that never started grep-mode
    p.set_session(s2)
    raw = json.dumps({"x": "x marks"})
    assert p.enrich_tool_result("call_service", raw).result == raw  # unaffected
    # back to s1 -> still active
    p.set_session(s1)
    assert p.enrich_tool_result("call_service", raw).result != raw
