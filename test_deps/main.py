"""Main module that imports from lib."""

from lib import hello, Calculator


def main():
    """Main entry point."""
    print(hello())

    calc = Calculator()
    result = calc.add(2, 3)
    print(f"2 + 3 = {result}")


if __name__ == "__main__":
    main()
