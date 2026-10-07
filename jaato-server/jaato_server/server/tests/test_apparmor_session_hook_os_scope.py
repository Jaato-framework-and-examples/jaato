"""Regression: ``_apparmor_session_hook`` must not shadow module ``os``.

A local ``import os`` inside ``websocket.py``'s ``_apparmor_session_hook``
made Python treat ``os`` as function-local throughout the function, so the
earlier ``os.path.realpath(...)`` calls raised
``UnboundLocalError: cannot access local variable 'os'`` — caught as a
"Session hook failed" WARNING, silently skipping AppArmor/cgroup
provisioning for WS-provisioned sessions.  Surfaced live during the gossip
two-daemon co-validation.

AST guard (no import needed): the nested ``_apparmor_session_hook`` must
reference ``os`` but never re-import it locally — so ``os`` always resolves
to the module-level import.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_apparmor_session_hook_has_no_local_import_os() -> None:
    websocket_py = (
        Path(__file__).resolve().parents[1] / "websocket.py"
    )
    tree = ast.parse(websocket_py.read_text())
    hook = _find_function(tree, "_apparmor_session_hook")
    assert hook is not None, (
        "_apparmor_session_hook not found in websocket.py — was it "
        "renamed?  Update this regression guard."
    )

    local_os_imports = [
        n for n in ast.walk(hook)
        if (isinstance(n, ast.Import)
            and any(a.name == "os" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and n.module == "os")
    ]
    assert not local_os_imports, (
        "_apparmor_session_hook re-imports ``os`` locally — this shadows "
        "the module-level import and raises UnboundLocalError at the "
        "earlier os.path.realpath() call.  Use the module-level os."
    )

    # Sanity: the function does use ``os`` (guard stays meaningful).
    uses_os = any(
        isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "os"
        for n in ast.walk(hook)
    )
    assert uses_os, (
        "_apparmor_session_hook no longer references os.* — if os use was "
        "removed entirely, this guard can be deleted."
    )
