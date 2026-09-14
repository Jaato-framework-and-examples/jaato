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
"""

from __future__ import annotations

import ast
import pathlib

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
from shared.tool_result_builder import normalize_result_dict, split_executor_result

_PLUGIN = "jaato-server/shared/plugins/subagent/plugin.py"
_ROOT = pathlib.Path(__file__).resolve().parents[3]


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


def _returned_values(tree: ast.AST):
    """Every ``return`` in *tree* that returns something, as (lineno, value)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            yield node.lineno, node.value


def _failure_returns(tree: ast.AST):
    """Split self-declared failure returns into (bare, contract) line lists."""
    bare, contract = [], []
    for lineno, value in _returned_values(tree):
        if _is_contract_failure(value):
            contract.append(lineno)
        elif _is_result_type_failure(value) or _is_literal_dict_failure(value):
            bare.append(lineno)
    return bare, contract


def test_no_failure_return_is_a_bare_value():
    """THE GUARD.  A declared failure must carry the flag that says so.

    Structural rather than per-function: a failure path added later is
    covered whether or not its author remembers the contract, which is the
    one property a hand-written list of executors does not have.
    """
    tree = ast.parse((_ROOT / _PLUGIN).read_text())
    bare, contract = _failure_returns(tree)
    assert not bare, (
        f"{len(bare)} failure return(s) at lines {bare} are bare values. "
        "ToolExecutor reports those as ok=True, so the reliability plugin's "
        "retry and circuit-breaker policies cannot see them (#1053)."
    )
    assert contract, "the scan found no failure returns at all — it is broken"


def test_the_guard_can_see_the_shape_it_forbids():
    """A guard that cannot fail proves nothing."""
    bare, _ = _failure_returns(ast.parse(
        "def f():\n"
        "    return {'success': False, 'error': 'x'}\n"
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
    from shared.plugins.reliability.patterns import PatternDetector
    from shared.plugins.reliability.types import PatternDetectionConfig

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
