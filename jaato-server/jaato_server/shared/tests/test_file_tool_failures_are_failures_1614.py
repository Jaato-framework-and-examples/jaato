"""A file tool that failed is reported as a failure (#1614).

WHAT WAS WRONG.  On a confined host the kernel refused a ``readFile`` and the
session record said::

    {"result": {"error": "Failed to read file: [Errno 13] Permission denied: ..."},
     "is_error": false}

``file_edit`` (53 sites) and ``filesystem_query`` (14 sites) answered every
failure with a BARE ``{"error": ...}`` dict, and
``ToolExecutor._normalize_executor_return`` reads a bare value as success.  So
``ok=True`` reached the persisted ``is_error``, the reliability plugin, the
telemetry span and the ``TOOL_RUNNER`` trace.  The #1053 shape, two plugins
over.  Every failure return is now the explicit ``(False, payload)`` contract
with the payload unchanged; the AST guard that keeps it so is
``test_subagent_failures_are_visible_1053`` (now over all three plugins).

A SECOND PATH HAD TO LEARN THE CONTRACT.  ``glob_files`` / ``grep_content``
are BackgroundCapable, so the ToolExecutor runs them through
``_execute_with_auto_background``: the mixin stores whatever the executor
returned and the wait reported every completed task as ``(True, result)``.
An explicit ``(False, payload)`` would have reached the model as a successful
call whose payload is a TUPLE.  ``background.mixin._split_explicit_contract``
unpacks it and marks the task FAILED, so both paths agree.

WHAT THE MODEL READS.  The error TEXT is unchanged.  For an ``error``-only
payload ``normalize_result_dict`` collapses the dict to the bare error string
on ``ok=False`` -- the framework's rendering for every explicit failure in
the tree.  Dict-response converters (Google GenAI, GitHub Models) wrap a
failure as ``{"error": <text>}`` and so produce exactly the bytes they did
before; text-content converters (Anthropic, the OpenAI-compatible family)
now send the error text instead of ``{"error": "<text>"}`` JSON, beside the
wire's own error flag where it has one.  Multi-key payloads (glob/grep's
``files``/``matches``, moveFile's ``source``, multiFileEdit's batch report)
are not collapsed and reach the model byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from jaato_sdk.plugins.model_provider.types import CancelToken
from jaato_server.shared.ai_tool_runner import ToolExecutor
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_result_builder import normalize_result_dict

_FILE_EDIT = "jaato-server/jaato_server/shared/plugins/file_edit/plugin.py"
_MIXIN = "jaato-server/jaato_server/shared/plugins/background/mixin.py"

REVERSIONS = [
    Reversion(
        target=_FILE_EDIT,
        find=(
            '            return False, {"error": f"Failed to read file: {e}"}\n'
            "\n    def _write_line_ending("
        ),
        replace=(
            '            return {"error": f"Failed to read file: {e}"}\n'
            "\n    def _write_line_ending("
        ),
        test="test_a_refused_read_reaches_the_executor_as_a_failure",
        because=(
            "readFile's OSError path is a bare dict again, so the "
            "kernel-refused read is persisted with is_error=false"
        ),
    ),
    Reversion(
        target=_MIXIN,
        find="            ok, result = _split_explicit_contract(result)\n",
        replace="            ok = True\n",
        test="test_a_backgrounded_glob_failure_stays_a_failure",
        because=(
            "the auto-background path stores the (False, payload) tuple as a "
            "completed result, so the call is ok=True with a tuple payload"
        ),
    ),
]


class _Registry:
    """The registry surface ToolExecutor reads, and no more."""

    def __init__(self, plugin: Any) -> None:
        self._by_tool = {t: plugin for t in plugin.get_executors()}

    def get_plugin_for_tool(self, tool_name: str) -> Optional[Any]:
        return self._by_tool.get(tool_name)

    def get_plugin(self, name: str) -> Optional[Any]:
        return None

    def get_tool_traits(self, tool_name: str) -> frozenset:
        return frozenset()


def _executor(plugin: Any) -> ToolExecutor:
    ex = ToolExecutor(auto_background_enabled=True)
    ex.set_registry(_Registry(plugin))
    for tool, fn in plugin.get_executors().items():
        ex.register(tool, fn)
    return ex


@pytest.fixture
def file_edit(tmp_path: Path):
    from jaato_server.shared.plugins.file_edit.plugin import FileEditPlugin

    ws = tmp_path / "ws"
    ws.mkdir()
    p = FileEditPlugin()
    p.initialize({"workspace_root": str(ws),
                  "backup_dir": str(tmp_path / "backups")})
    p.set_workspace_path(str(ws))
    yield p, ws
    p.shutdown()


@pytest.fixture
def fs_query(tmp_path: Path):
    from jaato_server.shared.plugins.filesystem_query.plugin import (
        FilesystemQueryPlugin,
    )

    ws = tmp_path / "ws"
    ws.mkdir()
    p = FilesystemQueryPlugin()
    p.initialize({"workspace_root": str(ws), "allow_tmp": False})
    yield p, ws
    p.shutdown()


def _payload(result: Any) -> dict:
    """The executor payload, minus scaffolding ToolExecutor may add."""
    assert isinstance(result, dict), result
    return {k: v for k, v in result.items() if not k.startswith("_")}


def test_a_refused_read_reaches_the_executor_as_a_failure(file_edit, monkeypatch):
    """The incident: the kernel answers EACCES to the open.

    Simulated at the read call, because the suite may run as root and a
    ``chmod 000`` file is readable by root.
    """
    from jaato_server.shared.plugins.file_edit import plugin as fe_module

    plugin, ws = file_edit
    (ws / "page.md").write_text("secret\n")

    def refused(*_a, **_k):
        raise PermissionError(13, "Permission denied", str(ws / "page.md"))

    monkeypatch.setattr(fe_module, "read_text_verified", refused)
    ok, result = _executor(plugin).execute(
        "readFile", {"path": "page.md"}, cancel_token=CancelToken(),
    )
    assert ok is False, result
    assert _payload(result) == {
        "error": f"Failed to read file: [Errno 13] Permission denied: "
                 f"'{ws / 'page.md'}'",
    }


@pytest.mark.parametrize("args, fragment", [
    ({"path": "missing.md"}, "File not found"),
    ({"path": "."}, "Not a file"),
    ({}, "path is required"),
])
def test_read_failures_that_need_no_kernel_are_failures(file_edit, args, fragment):
    """Failures every user (root included) can hit."""
    plugin, _ = file_edit
    ok, result = _executor(plugin).execute(
        "readFile", args, cancel_token=CancelToken(),
    )
    assert ok is False, result
    assert fragment in result["error"]


def test_a_successful_read_is_still_a_success(file_edit):
    """Successes stay bare, which split_executor_result reads as ok=True."""
    plugin, ws = file_edit
    (ws / "page.md").write_text("hello\n")
    ok, result = _executor(plugin).execute(
        "readFile", {"path": "page.md"}, cancel_token=CancelToken(),
    )
    assert ok is True, result
    assert "hello" in result


def test_a_failed_batch_is_a_failure_and_its_report_is_unchanged(file_edit):
    plugin, _ = file_edit
    ok, result = _executor(plugin).execute(
        "multiFileEdit",
        {"operations": [{"action": "delete", "path": "nope.txt"}]},
        cancel_token=CancelToken(),
    )
    assert ok is False, result
    payload = _payload(result)
    assert payload["success"] is False
    assert payload["rollback_completed"] is not None
    # Multi-key: not collapsed, so the model reads the same report.
    assert normalize_result_dict(dict(payload), ok=False) == payload


def test_a_backgrounded_glob_failure_stays_a_failure(fs_query, tmp_path):
    """glob_files runs through the auto-background path."""
    plugin, ws = fs_query
    ok, result = _executor(plugin).execute(
        "glob_files", {"pattern": "*", "root": str(ws / "missing")},
        cancel_token=CancelToken(),
    )
    assert ok is False, result
    assert _payload(result) == {
        "error": f"Root path does not exist: {ws / 'missing'}",
        "files": [],
        "total": 0,
    }


def test_a_backgrounded_grep_success_is_still_a_success(fs_query):
    plugin, ws = fs_query
    (ws / "a.txt").write_text("needle\n")
    ok, result = _executor(plugin).execute(
        "grep_content", {"pattern": "needle", "path": str(ws)},
        cancel_token=CancelToken(),
    )
    assert ok is True, result
    assert result["total_matches"] == 1


def test_the_error_text_the_model_reads_is_unchanged():
    """What reaches a converter, before and after the flag flipped."""
    from jaato_sdk.plugins.model_provider.types import render_result_for_model

    payload = {"error": "Failed to read file: [Errno 13] Permission denied"}
    before = normalize_result_dict(dict(payload), ok=True)
    after = normalize_result_dict(dict(payload), ok=False)
    assert before == payload
    assert after == payload["error"]
    # Text-content wires: the error text, no longer inside a JSON object.
    assert render_result_for_model(before) == (
        '{"error": "Failed to read file: [Errno 13] Permission denied"}')
    assert render_result_for_model(after) == payload["error"]


def test_a_dict_response_wire_sends_the_same_bytes():
    """Google GenAI wraps a failure as ``{"error": text}``: unchanged."""
    pytest.importorskip("google.genai")
    from jaato_sdk.plugins.model_provider.types import ToolResult
    from jaato_server.shared.plugins.model_provider.google_genai.converters import (  # noqa: E501
        tool_result_to_sdk_part,
    )

    payload = {"error": "Failed to read file: [Errno 13] Permission denied"}
    part_before = tool_result_to_sdk_part(ToolResult(
        call_id="c", name="readFile",
        result=normalize_result_dict(dict(payload), ok=True), is_error=False))
    part_after = tool_result_to_sdk_part(ToolResult(
        call_id="c", name="readFile",
        result=normalize_result_dict(dict(payload), ok=False), is_error=True))
    assert part_after.function_response.response == \
        part_before.function_response.response
