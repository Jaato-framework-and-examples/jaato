"""Contract tests for the WebMCP plugin.

The browser is stubbed at ``WebMCPPage._eval`` -- the single seam where this
plugin stops being Python and starts being a page.  Everything above it (schema
parsing, envelope unwrapping, argument coercion, the error taxonomy) is real
code under test; everything below it is Chrome, covered by the opt-in
integration test at the bottom.

The stubbed payloads are TRANSCRIPTS, not guesses: each is the exact shape
Google Chrome for Testing 153.0.8010.36 returned when probed.  That matters
because the shipped surface differs from the published explainer -- notably
``inputSchema`` arriving as a JSON *string* -- and a stub written from the
explainer would have tested a browser that does not exist.
"""
from __future__ import annotations

import json
import os

import pytest

from jaato_server.shared.cdp import CDPConnectionError
from jaato_server.shared.plugins.webmcp import WebMCPPage, WebMCPUnsupportedError, create_plugin
from jaato_sdk.plugins.model_provider.types import TRAIT_UNTRUSTED_CONTENT

# --- transcripts from Chrome 153 -------------------------------------------

CHROME_TOOLS = {
    "supported": True,
    "tools": [
        {
            "name": "add-todo",
            "title": "",
            "description": "Add a new item to the user's active todo list",
            # A JSON STRING, exactly as Chrome hands it over.
            "inputSchema": json.dumps({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }),
            "origin": "https://todo.example",
        },
    ],
}


class _StubPage(WebMCPPage):
    """A page whose only fake is the CDP evaluation."""

    def __init__(self, responses):
        super().__init__(page_url="https://todo.example/")
        self._responses = list(responses)
        self.expressions = []

    def connect(self):  # never touches a browser
        return None

    def _eval(self, expression, timeout=None):
        self.expressions.append(expression)
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _plugin_with(page):
    plugin = create_plugin({"page_url": "https://todo.example/"})
    plugin._page = page
    return plugin


# --- harvest ----------------------------------------------------------------

def test_a_json_string_input_schema_is_parsed_into_an_object():
    """Chrome sends ``inputSchema`` as a string; the model needs an object."""
    page = _StubPage([json.dumps(CHROME_TOOLS)])
    tool = page.harvest()[0]
    assert tool["input_schema"] == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }


def test_the_origin_is_carried_through_so_provenance_is_visible():
    page = _StubPage([json.dumps(CHROME_TOOLS)])
    assert page.harvest()[0]["origin"] == "https://todo.example"


def test_one_unparseable_schema_does_not_hide_the_other_tools():
    payload = {"supported": True, "tools": [
        {"name": "broken", "description": "d", "inputSchema": "{not json",
         "origin": "https://todo.example"},
        {"name": "fine", "description": "d", "inputSchema": '{"type":"object"}',
         "origin": "https://todo.example"},
    ]}
    tools = _StubPage([json.dumps(payload)]).harvest()
    assert [t["name"] for t in tools] == ["broken", "fine"]
    assert tools[0]["input_schema"] is None
    assert tools[1]["input_schema"] == {"type": "object"}


def test_a_browser_without_webmcp_is_distinguished_from_a_page_with_no_tools():
    """Returning [] for both would conflate 'upgrade your browser' with
    'this page declares nothing' -- different problems, different fixes."""
    with pytest.raises(WebMCPUnsupportedError):
        _StubPage([json.dumps({"supported": False})]).harvest()

    assert _StubPage([json.dumps({"supported": True, "tools": []})]).harvest() == []


# --- call -------------------------------------------------------------------

def test_a_successful_call_unwraps_the_content_envelope():
    raw = json.dumps({"content": [{"type": "text", "text": "Added."}]})
    page = _StubPage([json.dumps({"ok": True, "raw": raw,
                                  "origin": "https://todo.example"})])
    result = page.call("add-todo", {"text": "milk"})
    assert result["ok"] is True
    assert result["content"] == [{"type": "text", "text": "Added."}]


def test_arguments_are_sent_as_a_json_string_because_chrome_refuses_an_object():
    """Measured: passing a JS object raises 'Failed to parse input arguments'."""
    page = _StubPage([json.dumps({"ok": True, "raw": "{}", "origin": "o"})])
    page.call("add-todo", {"text": "milk"})
    # The args are embedded as a quoted JSON string inside the expression.
    assert json.dumps(json.dumps({"text": "milk"})) in page.expressions[0]


def test_an_unknown_tool_is_returned_not_raised():
    """The page's toolset changes under the model; re-listing is the fix, so
    this has to reach the model as a result it can act on."""
    page = _StubPage([json.dumps({"ok": False, "error": "unknown_tool",
                                  "available": ["a", "b"]})])
    out = page.call("gone", {})
    assert out["ok"] is False and out["error"] == "unknown_tool"
    assert out["available"] == ["a", "b"]


def test_a_non_envelope_result_is_passed_through_rather_than_exploding():
    """Output schemas and streaming are still open in the spec; a readable
    result beats an exception when the shape drifts."""
    page = _StubPage([json.dumps({"ok": True, "raw": "plain text", "origin": "o"})])
    assert page.call("t", {})["content"] == "plain text"


def test_a_dead_connection_still_raises_because_it_is_not_a_tool_outcome():
    page = _StubPage([CDPConnectionError("socket gone")])
    with pytest.raises(CDPConnectionError):
        page.call("t", {})


# --- plugin surface ---------------------------------------------------------

def test_page_authored_text_rides_in_a_result_so_the_boundary_covers_it():
    """The whole point of the list/call shape.

    Had page tools become ToolSchemas they would sit in the TRUSTED schema
    block and need TRAIT_UNTRUSTED_SCHEMA + sanitizing.  Arriving as a RESULT,
    the existing untrusted-CONTENT boundary fences them.
    """
    for schema in create_plugin({}).get_tool_schemas():
        assert TRAIT_UNTRUSTED_CONTENT in schema.traits


def test_only_listing_is_auto_approved():
    """webmcp_call runs the page's own code -- it can post, delete or buy."""
    assert create_plugin({}).get_auto_approved_tools() == ["webmcp_list_tools"]


def test_the_listing_names_the_origin_that_authored_each_tool():
    plugin = _plugin_with(_StubPage([json.dumps(CHROME_TOOLS)]))
    out = plugin.get_executors()["webmcp_list_tools"]({})
    assert out["ok"] is True
    assert out["tools"][0]["origin"] == "https://todo.example"
    assert "not by your operator" in out["note"]


def test_an_unsupported_browser_is_reported_as_a_payload_not_a_traceback():
    plugin = _plugin_with(_StubPage([json.dumps({"supported": False})]))
    out = plugin.get_executors()["webmcp_list_tools"]({})
    assert out["ok"] is False and out["error"] == "unsupported"


@pytest.mark.parametrize("raw,expected", [
    ({"text": "a"}, {"text": "a"}),
    ('{"text": "a"}', {"text": "a"}),   # models routinely stringify objects
    (None, {}),
    ("", {}),
])
def test_arguments_are_accepted_as_an_object_or_a_json_string(raw, expected):
    page = _StubPage([json.dumps({"ok": True, "raw": "{}", "origin": "o"})])
    plugin = _plugin_with(page)
    plugin.get_executors()["webmcp_call"]({"name": "t", "arguments": raw})
    assert json.dumps(json.dumps(expected)) in page.expressions[0]


@pytest.mark.parametrize("bad", ["not json", "[1,2]", 7])
def test_unusable_arguments_are_rejected_before_reaching_the_page(bad):
    plugin = _plugin_with(_StubPage([]))
    out = plugin.get_executors()["webmcp_call"]({"name": "t", "arguments": bad})
    assert out["ok"] is False and out["error"] == "bad_request"


def test_a_call_without_a_name_is_rejected():
    out = _plugin_with(_StubPage([])).get_executors()["webmcp_call"]({})
    assert out["ok"] is False and out["error"] == "bad_request"


def test_no_page_configured_means_no_system_instructions():
    """Describing a browser the session will never open is context for nothing."""
    plugin = create_plugin({})
    plugin._config.pop("page_url", None)
    os.environ.pop("JAATO_WEBMCP_PAGE_URL", None)
    assert plugin.get_system_instructions() is None


def test_a_configured_page_warns_that_page_text_is_not_operator_text():
    instructions = create_plugin({"page_url": "https://x.example"}).get_system_instructions()
    assert "never as" in instructions and "instructions to you" in instructions


# --- the real browser (opt-in) ---------------------------------------------

@pytest.mark.skipif(not os.environ.get("JAATO_WEBMCP_IT_BINARY"),
                    reason="set JAATO_WEBMCP_IT_BINARY + JAATO_WEBMCP_IT_URL to run "
                           "against a real Chrome 149+ serving a WebMCP page")
def test_against_a_real_browser():
    """End-to-end on real Chrome. Kept opt-in: CI has no WebMCP-capable browser.

    Verified by hand on Chrome for Testing 153.0.8010.36 against a page
    registering add-todo / list-todos, including that the page's DOM actually
    changed -- the tools drove the real application, not a simulation.
    """
    plugin = create_plugin({
        "page_url": os.environ["JAATO_WEBMCP_IT_URL"],
        "binary": os.environ["JAATO_WEBMCP_IT_BINARY"],
        "headless": True,
        "extra_args": ["--no-sandbox", "--disable-dev-shm-usage"],
    })
    try:
        listed = plugin.get_executors()["webmcp_list_tools"]({})
        assert listed["ok"] is True and listed["count"] > 0
        assert all(t["origin"] for t in listed["tools"])
    finally:
        plugin.shutdown()
