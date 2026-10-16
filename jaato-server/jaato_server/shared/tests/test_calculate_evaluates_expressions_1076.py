"""``calculate`` evaluates an expression on the interpreter the packages declare.

The evaluator walked the AST with an ``elif isinstance(node, ast.Num)``
branch kept for Python 3.7.  ``ast.Num`` was removed in Python 3.14, so on
3.14 every expression whose root was not a bare literal (``2+3``, ``abs(-1)``)
reached that branch and the tool answered
``Error: module 'ast' has no attribute 'Num'``.  Nothing tested the plugin,
which is how the move to 3.14 (#1076) found it.
"""

from jaato_server.shared.plugins.calculator.plugin import CalculatorPlugin


def _calc(expression: str) -> str:
    return CalculatorPlugin()._calculate({"expression": expression})


def test_a_binary_expression_is_evaluated():
    result = _calc("2+3*4")
    assert "14" in result and "Error" not in result, result


def test_a_function_call_and_unary_minus_are_evaluated():
    result = _calc("abs(-7) + round(2.6)")
    assert "10" in result and "Error" not in result, result


def test_a_non_numeric_constant_is_refused_by_name():
    assert "Unsupported constant type" in _calc("'a'")


# Reversions -- discovered by name by the meta-guard
# (test_every_guard_detects_its_own_reversion), which asserts each makes
# the NAMED test fail.
from jaato_server.shared.tests.reversion import Reversion  # noqa: E402

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/calculator/plugin.py",
        find=(
            '                raise ValueError(f"Unsupported constant type: '
            '{type(node.value).__name__}")\n\n'
            "            elif isinstance(node, ast.BinOp):\n"
        ),
        replace=(
            '                raise ValueError(f"Unsupported constant type: '
            '{type(node.value).__name__}")\n\n'
            "            elif isinstance(node, ast.Num):\n"
            "                return node.n\n\n"
            "            elif isinstance(node, ast.BinOp):\n"
        ),
        because=(
            "ast.Num was removed in Python 3.14: a branch naming it raises "
            "AttributeError for every expression that is not a bare literal, "
            "so the tool answers an error for 2+3"
        ),
        test="test_a_binary_expression_is_evaluated",
    ),
]
