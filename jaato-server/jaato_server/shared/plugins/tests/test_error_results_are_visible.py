"""An error result must be visible to the canonical checker.

``tool_result_is_error`` is what populates ``ToolCallEndEvent.is_error_result``
— the field documented as "computed deeper error check — success=True but error
body".  It recognises exactly one convention::

    return "error" in result or result.get("status_code", 200) >= 400

A plugin returning ``{"status": "error", "message": ...}`` has no ``error`` key,
so the checker says False, ``is_error_result`` stays False, and
``tool.call_end`` reports a clean call.  A consumer watching the event stream
cannot see the failure at all.

Found when the cascade-coordination example's error probe honestly reported
"0 errors" on a run where the tool failed EVERY time.  31 sites across four
plugins were invisible this way; 444 sites already used the ``error`` key.

This guard is AST-based rather than textual: it looks at real dict literals, so
a docstring or comment mentioning the shape cannot trip it, and a dict split
across lines cannot hide from it.
"""
import ast
import io
import pathlib

import pytest

# Plugins whose error dicts are checked.  Kept explicit rather than globbing the
# whole tree: a new plugin should be added here deliberately, and the failure
# message says why.
PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _error_dicts_missing_the_key(path: pathlib.Path):
    """Dict literals that say status='error' but carry no 'error' key."""
    try:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {k.value: v for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        status = pairs.get("status")
        if (isinstance(status, ast.Constant) and status.value == "error"
                and "error" not in pairs):
            out.append(node.lineno)
    return out


def _plugin_sources():
    return [p for p in PLUGIN_ROOT.rglob("*.py")
            if "/tests/" not in str(p) and "/build/" not in str(p)]


def test_the_scan_actually_finds_files():
    """Guard the guard — an empty file list would make this vacuously green."""
    assert len(_plugin_sources()) > 20


def test_no_plugin_returns_an_error_the_checker_cannot_see():
    offenders = []
    for path in _plugin_sources():
        for line in _error_dicts_missing_the_key(path):
            offenders.append(f"{path.relative_to(PLUGIN_ROOT)}:{line}")
    assert not offenders, (
        "these dict literals say status='error' but carry no 'error' key, so "
        "tool_result_is_error() returns False, ToolCallEndEvent.is_error_result "
        "stays False, and tool.call_end reports the call as clean — a consumer "
        "watching events cannot see the failure:\n  " + "\n  ".join(offenders)
    )


def test_the_checker_still_recognises_the_canonical_shape():
    """Pin the contract this guard is enforcing, so a change to one is visible
    against the other."""
    from jaato_sdk.plugins.model_provider.types import tool_result_is_error
    assert tool_result_is_error({"error": "boom"})
    assert tool_result_is_error({"status_code": 500})
    assert not tool_result_is_error({"status": "ok"})
    assert not tool_result_is_error("not a dict")
    # The shape this migration produced:
    assert tool_result_is_error({"status": "error", "error": "boom"})
