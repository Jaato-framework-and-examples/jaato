"""Enforce that plugins never store session state on self in set_session().

Plugin instances are shared across subagents within a session. If a plugin
stores session-derived state on ``self`` (e.g. ``self._session = session``),
a subagent's ``set_session()`` call will overwrite the parent's reference.

The correct patterns are:
1. Use ``threading.local()`` (thread-local storage) for per-thread state
2. Use ``get_current_session()`` from ``shared.session_context`` (ContextVar)
3. Don't store session at all — access it when needed via the ContextVar

This test discovers all plugin modules, finds ``set_session`` methods,
and verifies none of them assign to ``self._*``.
"""

import ast
import os
from pathlib import Path

import pytest


PLUGINS_DIR = Path(__file__).parent.parent / "plugins"

# Modules where self._ assignments in set_session are legitimate because
# the class is instantiated per-subagent (not shared via registry).
ALLOWED_CLASSES = frozenset({
    # ParentBridgedChannel instances are created per-subagent, not shared.
    "ParentBridgedChannel",
})


def _find_plugin_py_files():
    """Yield all .py files under the plugins directory."""
    for root, _dirs, files in os.walk(PLUGINS_DIR):
        # Skip test directories
        if "/tests/" in root or root.endswith("/tests"):
            continue
        for fname in files:
            if fname.endswith(".py"):
                yield Path(root) / fname


def _find_set_session_self_assignments(filepath: Path):
    """Parse a Python file and find set_session methods that assign to self.

    Returns list of (class_name, line_number, assignment_target) tuples
    for any ``self._* = ...`` found inside a ``set_session`` method,
    unless the class is in ALLOWED_CLASSES.
    """
    source = filepath.read_text()
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_name = node.name
        if class_name in ALLOWED_CLASSES:
            continue

        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != "set_session":
                continue

            # Walk the method body looking for self._ = assignments
            for child in ast.walk(item):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and target.attr.startswith("_")
                        ):
                            violations.append((
                                class_name,
                                child.lineno,
                                f"self.{target.attr}",
                            ))

    return violations


def test_no_self_assignments_in_set_session():
    """All plugin set_session() methods must use thread-local or ContextVar.

    Detects ``self._foo = ...`` inside ``set_session()`` methods, which
    would cause cross-subagent data leakage since plugin instances are
    shared across subagents within a session.
    """
    all_violations = []

    for filepath in _find_plugin_py_files():
        violations = _find_set_session_self_assignments(filepath)
        for class_name, lineno, target in violations:
            rel = filepath.relative_to(PLUGINS_DIR)
            all_violations.append(f"  {rel}:{lineno} {class_name}.set_session() assigns {target}")

    if all_violations:
        msg = (
            "Plugin set_session() methods must not assign to self._*\n"
            "Use threading.local() or shared.session_context.get_current_session() instead.\n"
            "Violations:\n" + "\n".join(all_violations)
        )
        pytest.fail(msg)
