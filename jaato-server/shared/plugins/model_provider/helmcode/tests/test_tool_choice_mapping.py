"""Helmcode forwards a name-bearing tool_choice through name_to_id (the wire id).

The tools array is hashed (name_to_id); a tool_choice that NAMES a function —
e.g. forcing ``signal_completion`` so the backend grammar-constrains its args
to the completion schema — must use the same wire id, else the API rejects
it ("Tool X not found in tools list").
"""

from shared.tool_id_map import name_to_id
from shared.plugins.model_provider.helmcode.provider import HelmcodeProvider


def _force(name):
    return {"type": "function", "function": {"name": name}}


def test_profile_api_params_tool_choice_name_mapped():
    p = HelmcodeProvider()
    p._api_params = {"tool_choice": _force("signal_completion")}
    kw = {"tools": [object()]}
    p._apply_api_params(kw, None)
    assert kw["tool_choice"]["function"]["name"] == name_to_id("signal_completion")


def test_per_call_tool_choice_name_mapped():
    p = HelmcodeProvider()
    kw = {"tools": [object()]}
    p._apply_api_params(kw, _force("signal_completion"))
    assert kw["tool_choice"]["function"]["name"] == name_to_id("signal_completion")


def test_string_tool_choice_passes_through():
    p = HelmcodeProvider()
    kw = {"tools": [object()]}
    p._apply_api_params(kw, "required")
    assert kw["tool_choice"] == "required"


def test_tool_choice_dropped_without_tools():
    p = HelmcodeProvider()
    kw = {}
    p._apply_api_params(kw, "required")
    assert "tool_choice" not in kw
