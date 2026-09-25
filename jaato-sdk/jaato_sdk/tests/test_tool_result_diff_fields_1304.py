"""``tool_result_diff_fields`` and ``ToolCallEndEvent.diff`` /
``diff_truncated`` / ``path`` (jaato/#1304 phase 3).
"""

from __future__ import annotations

from jaato_sdk.events import ToolCallEndEvent, deserialize_event, serialize_event
from jaato_sdk.plugins.model_provider.types import tool_result_diff_fields


def test_extracts_all_three_fields_when_present() -> None:
    payload = {
        "success": True,
        "path": "src/main.py",
        "diff": "--- a/src/main.py\n+++ b/src/main.py\n",
        "diff_truncated": False,
    }
    assert tool_result_diff_fields(payload) == {
        "path": "src/main.py",
        "diff": "--- a/src/main.py\n+++ b/src/main.py\n",
        "diff_truncated": False,
    }


def test_returns_empty_dict_for_a_result_with_none_of_the_keys() -> None:
    assert tool_result_diff_fields({"success": True, "lines": 3}) == {}


def test_returns_empty_dict_for_a_non_dict_result() -> None:
    assert tool_result_diff_fields("some string result") == {}
    assert tool_result_diff_fields(None) == {}


def test_ignores_wrongly_typed_values_rather_than_propagating_them() -> None:
    """A tool that happens to have a ``diff`` key holding something other
    than a string (an int, e.g.) must not have that value reach the
    wire under a field the event declares as ``Optional[str]``."""
    assert tool_result_diff_fields({"diff": 123, "path": None}) == {}


def test_tool_call_end_event_diff_and_path_round_trip() -> None:
    original = ToolCallEndEvent(
        agent_id="main",
        tool_name="writeNewFile",
        call_id="call_1",
        success=True,
        diff="--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+print(1)",
        diff_truncated=False,
        path="new.py",
    )
    restored = deserialize_event(serialize_event(original))
    assert isinstance(restored, ToolCallEndEvent)
    assert restored.diff == original.diff
    assert restored.diff_truncated is False
    assert restored.path == "new.py"


def test_tool_call_end_event_defaults_have_no_diff() -> None:
    event = ToolCallEndEvent(agent_id="main", tool_name="readFile")
    assert event.diff is None
    assert event.diff_truncated is None
    assert event.path is None
