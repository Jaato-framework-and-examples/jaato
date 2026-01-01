"""Library module with exportable symbols."""


def hello() -> str:
    """Return a greeting."""
    return "Hello, World!"


def goodbye() -> str:
    """Return a farewell."""
    return "Goodbye!"


class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b
