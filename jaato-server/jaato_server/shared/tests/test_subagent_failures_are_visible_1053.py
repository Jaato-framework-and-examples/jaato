"""A subagent domain failure reaches the reliability plugin as a failure (#1053).

WHAT WAS WRONG.  ``_execute_spawn_subagent`` and its four siblings returned a
BARE DICT on their failure paths::

    return SubagentResult(success=False, error="Profile 'x' not found…").to_dict()

``ToolExecutor._normalize_executor_return`` documents what becomes of that
shape — *"anything else — a bare result, reported as success"* — so ``ok`` was
``True``, and ``ok`` is exactly what is handed to the reliability plugin::

    ok, result = self._normalize_executor_return(result)
    ...
    self._reliability_plugin.on_tool_result(name, args, ok, result, ...)

``PatternDetector.on_tool_result`` then recorded ``success=True``, and
``_check_error_retry_loop`` only walks entries with ``success is False``.  So
26 of the plugin's 33 failure returns were outside the set its own
circuit-breaker and retry policies scan.

THE PRECEDENT IS IN THE SAME FILE.  ``_execute_send_to_sibling`` and
``_execute_list_siblings`` were already converted, with a comment giving this
exact argument (*"making a failing tool invisible to anything watching the
event stream"*).  Five executors were left behind; this closes them.

WHY IT IS SAFE, and the reason the fix is a flag rather than a payload change:
``normalize_result_dict`` reshapes on ``not ok`` in exactly one case — an
``error``-only dict collapses to a bare string — and ``SubagentResult.to_dict``
always emits at least ``success`` and ``turns_used`` beside it.  The
model-facing payload is therefore identical either way, which
``test_the_model_facing_payload_is_unchanged`` pins.

SCOPE.  Failure returns become explicit; SUCCESS returns stay bare, which is
the documented contract (``split_executor_result``: a bare value is ``ok=True``)
and what 144 sites elsewhere in the tree rely on.

#1614 WIDENED IT to ``file_edit`` and ``filesystem_query``.  Their executors
answered every failure with a bare ``{"error": ...}`` dict, so a readFile the
kernel refused was persisted with ``is_error: false``.  Those plugins never
declare ``success=False``; their failure shape is a returned dict literal
carrying an ``error`` key, so the guard gained that rule, scoped to the
``_execute_*`` executors (a non-executor helper may return an ``error`` dict
that its own caller reads, which is not a tool result).  One guard, one rule
-- a failure return must carry the flag -- over every module listed in
``_PLUGINS``.  Unlike the subagent payloads, many of these ARE ``error``-only,
so ``normalize_result_dict`` now collapses them to the bare error string:
the text the model reads is unchanged, the JSON wrapper around it is not
(see ``test_file_tool_failures_are_failures_1614``).
"""

from __future__ import annotations

import ast
import pathlib

from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_result_builder import normalize_result_dict, split_executor_result

_PLUGIN = "jaato-server/jaato_server/shared/plugins/subagent/plugin.py"
_FILE_EDIT = "jaato-server/jaato_server/shared/plugins/file_edit/plugin.py"
_FS_QUERY = "jaato-server/jaato_server/shared/plugins/filesystem_query/plugin.py"
#: Every plugin whose executors this guard holds to the explicit contract.
_PLUGINS = (_PLUGIN, _FILE_EDIT, _FS_QUERY)
_ROOT = pathlib.Path(__file__).resolve().parents[4]


REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find=(
            "                return False, SubagentResult(\n"
            "                    success=False,\n"
            "                    response='',\n"
            "                    error=f\"Profile '{profile_name}' not found."
            " Available: {available}\"\n"
            "                ).to_dict()"
        ),
        replace=(
            "                return SubagentResult(\n"
            "                    success=False,\n"
            "                    response='',\n"
            "                    error=f\"Profile '{profile_name}' not found."
            " Available: {available}\"\n"
            "                ).to_dict()"
        ),
        test="test_no_failure_return_is_a_bare_value",
        because=(
            "the profile-not-found path returns a bare dict again, so "
            "ToolExecutor reports ok=True and the reliability plugin's "
            "error-retry loop detector never sees the failure that #1052's "
            "spawn loop was made of"
        ),
    ),
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
        test="test_no_failure_return_is_a_bare_value",
        because=(
            "readFile's OSError path returns a bare dict again -- the "
            "kernel-refused read #1614 found persisted as is_error=false"
        ),
    ),
    Reversion(
        target=_FS_QUERY,
        find='            return False, {"error": "Pattern is required", "files": [], "total": 0}',
        replace='            return {"error": "Pattern is required", "files": [], "total": 0}',
        test="test_no_failure_return_is_a_bare_value",
        because=(
            "glob_files' missing-pattern path returns a bare dict again, so "
            "ToolExecutor reports the refusal as a success"
        ),
    ),
]


# --------------------------------------------------------------- the AST guard

def _is_contract_failure(value: ast.expr) -> bool:
    """``return False, payload`` — the shape ToolExecutor reads as a failure."""
    if not isinstance(value, ast.Tuple) or len(value.elts) != 2:
        return False
    flag = value.elts[0]
    return isinstance(flag, ast.Constant) and flag.value is False


def _declares_success_false(keywords) -> bool:
    """Does this keyword list carry ``success=False``?"""
    for kw in keywords:
        if kw.arg != "success":
            continue
        if isinstance(kw.value, ast.Constant) and kw.value.value is False:
            return True
    return False


def _is_result_type_failure(value: ast.expr) -> bool:
    """``SubagentResult(success=False, ...).to_dict()`` — a bare dict."""
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if not isinstance(func, ast.Attribute) or func.attr != "to_dict":
        return False
    if not isinstance(func.value, ast.Call):
        return False
    return _declares_success_false(func.value.keywords)


def _is_literal_dict_failure(value: ast.expr) -> bool:
    """``{"success": False, ...}`` — also a bare dict."""
    if not isinstance(value, ast.Dict):
        return False
    for key, val in zip(value.keys, value.values):
        if not (isinstance(key, ast.Constant) and key.value == "success"):
            continue
        if isinstance(val, ast.Constant) and val.value is False:
            return True
    return False


def _is_error_dict(value: ast.expr) -> bool:
    """``{"error": <not None>, ...}`` -- the file tools' failure shape (#1614).

    An ``error`` key whose value is the literal ``None`` is the ABSENCE of
    one (``tool_result_is_error`` reads it that way), so it is not counted.
    """
    if not isinstance(value, ast.Dict):
        return False
    for key, val in zip(value.keys, value.values):
        if isinstance(key, ast.Constant) and key.value == "error":
            return not (isinstance(val, ast.Constant) and val.value is None)
    return False


def _returned_values(tree: ast.AST):
    """Every ``return`` in *tree* that returns something, as (lineno, value)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            yield node.lineno, node.value


def _executor_lines(tree: ast.AST) -> set:
    """Line numbers of the returns inside ``_execute_*`` executors."""
    lines = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and fn.name.startswith("_execute_"):
            lines.update(lineno for lineno, _ in _returned_values(fn))
    return lines


def _failure_returns(tree: ast.AST):
    """Split self-declared failure returns into (bare, contract) line lists."""
    bare, contract = [], []
    in_executor = _executor_lines(tree)
    for lineno, value in _returned_values(tree):
        if _is_contract_failure(value):
            contract.append(lineno)
        elif _is_result_type_failure(value) or _is_literal_dict_failure(value):
            bare.append(lineno)
        elif lineno in in_executor and _is_error_dict(value):
            bare.append(lineno)
    return bare, contract


def test_no_failure_return_is_a_bare_value():
    """THE GUARD.  A declared failure must carry the flag that says so.

    Structural rather than per-function: a failure path added later is
    covered whether or not its author remembers the contract, which is the
    one property a hand-written list of executors does not have.
    """
    problems = []
    for plugin in _PLUGINS:
        tree = ast.parse((_ROOT / plugin).read_text())
        bare, contract = _failure_returns(tree)
        if bare:
            problems.append(f"{plugin}: lines {bare}")
        assert contract, (
            f"the scan found no failure returns at all in {plugin} — it is "
            "broken")
    assert not problems, (
        "failure return(s) are bare values: " + "; ".join(problems) + ". "
        "ToolExecutor reports those as ok=True, so the reliability plugin's "
        "retry and circuit-breaker policies, telemetry and the persisted "
        "is_error cannot see them (#1053, #1614)."
    )


def test_the_guard_can_see_the_shape_it_forbids():
    """A guard that cannot fail proves nothing."""
    bare, _ = _failure_returns(ast.parse(
        "def f():\n"
        "    return {'success': False, 'error': 'x'}\n"
    ))
    assert bare == [2]


def test_the_guard_sees_an_executor_error_dict_and_nothing_else():
    """The #1614 rule: an ``error`` dict, in an executor, not ``error: None``."""
    bare, _ = _failure_returns(ast.parse(
        "def _execute_x():\n"
        "    return {'error': 'boom'}\n"
        "def _execute_y():\n"
        "    return {'error': None, 'result': 1}\n"
        "def _helper():\n"
        "    return {'ok': False, 'error': 'not a tool result'}\n"
    ))
    assert bare == [2]


# ------------------------------------------------------- the contract itself

def test_a_failure_splits_to_ok_false():
    assert split_executor_result((False, {"error": "boom"})) == (
        False, {"error": "boom"})


def test_a_bare_value_still_means_success():
    """Unchanged, and load-bearing: the success returns are left bare."""
    assert split_executor_result({"success": True}) == (True, {"success": True})


def test_the_model_facing_payload_is_unchanged():
    """Why the flag could be flipped without touching what the model reads.

    ``normalize_result_dict`` collapses to a bare string only when ``error``
    is the ONLY remaining key.  ``SubagentResult.to_dict`` always emits
    ``success`` and ``turns_used`` beside it, so the collapse cannot fire and
    both flags render the same dict.
    """
    payload = {
        "success": False,
        "turns_used": 0,
        "response": "",
        "error": "Profile 'summarizer' not found. Available: ['writer']",
    }
    assert (normalize_result_dict(dict(payload), ok=False)
            == normalize_result_dict(dict(payload), ok=True))


def test_an_error_only_dict_would_have_collapsed():
    """The branch above is real — it just cannot reach a SubagentResult."""
    assert normalize_result_dict({"error": "boom"}, ok=False) == "boom"
    assert normalize_result_dict({"error": "boom"}, ok=True) == {"error": "boom"}


# --------------------------------------------------- the consequence it buys

def test_the_retry_loop_detector_now_counts_these_failures():
    """The point of the change, driven through the real detector.

    Three consecutive failures of one tool with the same argument KEYS is
    exactly #1052's loop: the task prose was reworded each iteration and the
    keys never changed.  Fed ``success=True`` — what the bare dict produced —
    the detector sees nothing.
    """
    from jaato_server.shared.plugins.reliability.patterns import PatternDetector
    from jaato_server.shared.plugins.reliability.types import PatternDetectionConfig

    def drive(reported_success: bool):
        det = PatternDetector(PatternDetectionConfig(enabled=True))
        fired = None
        for prose in ("Escribir un resumen", "Generar un resumen",
                      "Realizar un resumen"):
            det.on_tool_called("spawn_subagent",
                             {"profile": "summarizer", "inputs": prose})
            fired = det.on_tool_result("spawn_subagent", reported_success) or fired
        return fired

    assert drive(reported_success=False) is not None, (
        "the detector did not fire on three identical-key failures")
    assert drive(reported_success=True) is None, (
        "reported as successes — which is what the bare dict did — the "
        "detector is blind, which is #1053")
