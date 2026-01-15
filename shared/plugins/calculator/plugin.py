# shared/plugins/calculator/plugin.py

import ast
import json
import operator
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from jaato import ToolSchema


class CalculatorPlugin:
    """Plugin that provides mathematical calculation tools."""

    @property
    def name(self) -> str:
        """Unique identifier for this plugin."""
        return "calculator"

    def __init__(self):
        self.precision = 2
        self._agent_name: Optional[str] = None

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [CALCULATOR{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Called by registry with configuration.

        Args:
            config: Dict with plugin settings
        """
        config = config or {}
        self._agent_name = config.get("agent_name")
        self.precision = config.get("precision", 2)
        self._trace(f"initialize: precision={self.precision}")

    def shutdown(self) -> None:
        """Cleanup when plugin is disabled."""
        self._trace("shutdown")

    def get_tool_schemas(self):
        """Declare the tools this plugin provides."""
        return [
            ToolSchema(
                name="add",
                description="Add two numbers together and return the result",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["a", "b"]
                },
                category="code",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="subtract",
                description="Subtract second number from first number (a - b)",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number to subtract"
                        }
                    },
                    "required": ["a", "b"]
                },
                category="code",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="multiply",
                description="Multiply two numbers together and return the result",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["a", "b"]
                },
                category="code",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="divide",
                description="Divide first number by second number (a / b)",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "Numerator"
                        },
                        "b": {
                            "type": "number",
                            "description": "Denominator (cannot be zero)"
                        }
                    },
                    "required": ["a", "b"]
                },
                category="code",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="calculate",
                description="Evaluate a mathematical expression safely. Supports basic operations (+, -, *, /, **, %, parentheses) and common math functions",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', '(10 + 5) / 3')"
                        }
                    },
                    "required": ["expression"]
                },
                category="code",
                discoverability="discoverable",
            )
        ]

    def get_executors(self):
        """Map tool names to executor functions."""
        return {
            "add": self._add,
            "subtract": self._subtract,
            "multiply": self._multiply,
            "divide": self._divide,
            "calculate": self._calculate
        }

    def _add(self, args: Dict[str, Any]) -> str:
        """
        Add two numbers.

        Args:
            args: Dict with 'a' and 'b' numbers.

        Returns formatted result or error message.
        """
        a = args.get('a')
        b = args.get('b')
        self._trace(f"add: a={a}, b={b}")
        try:
            if a is None or b is None:
                return "Error: Both 'a' and 'b' are required"
            result = float(a) + float(b)
            return json.dumps({
                "operation": "addition",
                "operands": [a, b],
                "result": round(result, self.precision)
            }, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def _subtract(self, args: Dict[str, Any]) -> str:
        """
        Subtract b from a.

        Args:
            args: Dict with 'a' and 'b' numbers.

        Returns formatted result or error message.
        """
        a = args.get('a')
        b = args.get('b')
        self._trace(f"subtract: a={a}, b={b}")
        try:
            if a is None or b is None:
                return "Error: Both 'a' and 'b' are required"
            result = float(a) - float(b)
            return json.dumps({
                "operation": "subtraction",
                "operands": [a, b],
                "result": round(result, self.precision)
            }, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def _multiply(self, args: Dict[str, Any]) -> str:
        """
        Multiply two numbers.

        Args:
            args: Dict with 'a' and 'b' numbers.

        Returns formatted result or error message.
        """
        a = args.get('a')
        b = args.get('b')
        self._trace(f"multiply: a={a}, b={b}")
        try:
            if a is None or b is None:
                return "Error: Both 'a' and 'b' are required"
            result = float(a) * float(b)
            return json.dumps({
                "operation": "multiplication",
                "operands": [a, b],
                "result": round(result, self.precision)
            }, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def _divide(self, args: Dict[str, Any]) -> str:
        """
        Divide a by b.

        Args:
            args: Dict with 'a' and 'b' numbers.

        Returns formatted result or error message.
        """
        a = args.get('a')
        b = args.get('b')
        self._trace(f"divide: a={a}, b={b}")
        try:
            if a is None or b is None:
                return "Error: Both 'a' and 'b' are required"

            a = float(a)
            b = float(b)

            # Validate input
            if b == 0:
                return "Error: Division by zero is not allowed"

            result = a / b
            return json.dumps({
                "operation": "division",
                "operands": [a, b],
                "result": round(result, self.precision)
            }, indent=2)
        except ZeroDivisionError:
            return "Error: Division by zero is not allowed"
        except Exception as e:
            return f"Error: {str(e)}"

    def _calculate(self, args: Dict[str, Any]) -> str:
        """
        Evaluate a mathematical expression safely using AST parsing.

        Args:
            args: Dict with 'expression' string.

        Returns formatted result or error message.
        """
        expression = args.get('expression')
        self._trace(f"calculate: expression={expression!r}")
        try:
            # Validate input
            if not expression:
                return "Error: expression required"

            # Ensure expression is a string
            expression = str(expression)

            # Use safe AST-based evaluation instead of eval()
            result = self._safe_eval(expression)

            # Format output clearly
            return json.dumps({
                "expression": expression,
                "result": round(result, self.precision)
            }, indent=2)

        except ZeroDivisionError:
            return "Error: Division by zero in expression"
        except SyntaxError as e:
            return f"Error: Invalid expression syntax - {e}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression using AST parsing.

        Only allows numbers, basic arithmetic operators, and safe functions.
        No arbitrary code execution possible.

        Args:
            expression: Mathematical expression string.

        Returns:
            Numeric result of the expression.

        Raises:
            ValueError: If expression contains disallowed operations.
        """
        # Supported binary operators
        _operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
            ast.USub: operator.neg,  # Unary minus
            ast.UAdd: operator.pos,  # Unary plus
        }

        # Supported functions
        _functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': pow,
        }

        def _eval_node(node):
            """Recursively evaluate an AST node."""
            if isinstance(node, ast.Constant):  # Python 3.8+
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

            elif isinstance(node, ast.Num):  # Python 3.7 compatibility
                return node.n

            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in _operators:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                return _operators[op_type](left, right)

            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in _operators:
                    raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
                operand = _eval_node(node.operand)
                return _operators[op_type](operand)

            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function calls are allowed")
                func_name = node.func.id
                if func_name not in _functions:
                    raise ValueError(f"Unsupported function: {func_name}")
                args = [_eval_node(arg) for arg in node.args]
                return _functions[func_name](*args)

            elif isinstance(node, ast.Expression):
                return _eval_node(node.body)

            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")

        # Parse and evaluate
        tree = ast.parse(expression, mode='eval')
        return _eval_node(tree)

    # ==================== Required Protocol Methods ====================

    def get_system_instructions(self) -> Optional[str]:
        """Instructions for the model about calculator tools."""
        return None  # Tool descriptions are self-explanatory

    def get_auto_approved_tools(self) -> List[str]:
        """All calculator tools are safe, read-only operations."""
        return ["add", "subtract", "multiply", "divide", "calculate"]

    def get_user_commands(self) -> List:
        """Calculator provides model tools only, no user commands."""
        return []
