"""``jaato_server.shared.tool_classification`` -- the server-side mirror of
``jaato-web-coder-ui/src/protocol/toolClass.ts`` (jaato/#1304 phase 3).
"""

from __future__ import annotations

from jaato_server.shared.tool_classification import (
    HOUSEKEEPING_TOOLS,
    TOOL_CLASSES,
    classify_tool,
)


def test_classes_the_issues_own_housekeeping_list() -> None:
    for name in (
        "createPlan",
        "startPlan",
        "setStepStatus",
        "completePlan",
        "list_tools",
        "get_tool_schemas",
        "listReferences",
        "list_subagent_profiles",
        "subscribeToEvents",
    ):
        assert classify_tool(name) == "housekeeping"
    # Every entry classify_tool answers "housekeeping" for is also in the
    # set auto_allow_housekeeping whitelists -- one list, not two.
    assert HOUSEKEEPING_TOOLS == {
        "createPlan",
        "startPlan",
        "setStepStatus",
        "completePlan",
        "list_tools",
        "get_tool_schemas",
        "listReferences",
        "list_subagent_profiles",
        "subscribeToEvents",
    }


def test_classes_file_edit_as_write() -> None:
    assert classify_tool("writeNewFile") == "write"
    assert classify_tool("updateFile") == "write"
    assert classify_tool("removeFile") == "write"


def test_classes_a_subprocess_shell_notebook_call_as_exec() -> None:
    assert classify_tool("cli_based_tool") == "exec"
    assert classify_tool("shell_input") == "exec"
    assert classify_tool("notebook_execute") == "exec"


def test_classes_a_look_only_call_as_read() -> None:
    assert classify_tool("readFile") == "read"
    assert classify_tool("glob_files") == "read"


def test_classes_subagent_session_verbs_as_agent() -> None:
    assert classify_tool("spawn_subagent") == "agent"
    assert classify_tool("send_to_session") == "agent"


def test_falls_back_to_other_for_anything_it_has_no_opinion_about() -> None:
    """Never a guess -- an MCP tool or an unclassified in-tree tool is
    ``"other"``, not silently assigned a class nobody asserted."""
    assert classify_tool("mcp__server__tool") == "other"
    assert classify_tool("store_memory") == "other"
    assert classify_tool("call_service") == "other"


def test_every_class_the_function_can_return_is_declared() -> None:
    for name in (
        "createPlan",
        "writeNewFile",
        "cli_based_tool",
        "readFile",
        "spawn_subagent",
        "an-mcp-tool",
    ):
        assert classify_tool(name) in TOOL_CLASSES
