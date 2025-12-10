# shared/plugins/calculator/plugin.py

from shared.plugins.model_provider.types import ToolSchema
import json


class CalculatorPlugin:
    """Plugin that provides mathematical calculation tools."""

    def __init__(self):
        self.precision = 2

    def initialize(self, config: dict):
        """
        Called by registry with configuration.

        Args:
            config: Dict with plugin settings
        """
        self.precision = config.get("precision", 2)

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
                }
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
                }
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
                }
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
                }
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
                }
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

    def _add(self, a: float, b: float) -> str:
        """
        Add two numbers.

        Returns formatted result or error message.
        """
        try:
            result = a + b
            return json.dumps({
                "operation": "addition",
                "operands": [a, b],
                "result": round(result, self.precision)
            }, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def _subtract(self, a: float, b: float) -> str:
        """
        Subtract b from a.

        Returns formatted result or error message.
        """
        try:
            result = a - b
            return json.dumps({
                "operation": "subtraction",
                "operands": [a, b],
                "result": round(result, self.precision)
            }, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def _multiply(self, a: float, b: float) -> str:
        """
        Multiply two numbers.

        Returns formatted result or error message.
        """
        try:
            result = a * b
            return json.dumps({
                "operation": "multiplication",
                "operands": [a, b],
                "result": round(result, self.precision)
            }, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def _divide(self, a: float, b: float) -> str:
        """
        Divide a by b.

        Returns formatted result or error message.
        """
        try:
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

    def _calculate(self, expression: str) -> str:
        """
        Evaluate a mathematical expression safely.

        Returns formatted result or error message.
        """
        try:
            # Validate input
            if not expression:
                return "Error: expression required"

            # Safe eval for math only - restrict builtins to prevent code execution
            # Only allow basic math operations
            allowed_names = {
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'pow': pow,
            }

            result = eval(expression, {"__builtins__": allowed_names}, {})

            # Format output clearly
            return json.dumps({
                "expression": expression,
                "result": round(result, self.precision)
            }, indent=2)

        except ZeroDivisionError:
            return "Error: Division by zero in expression"
        except SyntaxError as e:
            return f"Error: Invalid expression syntax - {e}"
        except NameError as e:
            return f"Error: Invalid operation or function in expression - {e}"
        except Exception as e:
            return f"Error: {str(e)}"
