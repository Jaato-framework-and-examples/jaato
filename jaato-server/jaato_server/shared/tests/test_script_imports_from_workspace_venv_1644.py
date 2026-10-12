"""A processor or prefetch script may import from the session's venv (#1644).

Completion processors and ``{{!py:...}}`` prefetch scripts are imported by
``script_loader.load_script_symbol`` into the runner's own interpreter, so
a third-party import in either resolved against the daemon's environment
only.  A cascade bringing its own Python had two wrong options: install
into the daemon's venv (shared by every tenant), or ``sys.path.insert``
from the script (which changes the path of a slot that serves several
sessions in turn, #1171).

``plugin_configs.cli.script_imports: workspace_venv`` is the opt-in.  The
rules these tests hold, each against the real loader:

- a package only the venv has imports, at load AND at call time;
- the venv is on ``sys.path`` for the load and each call only, and what was
  loaded from it leaves ``sys.modules`` between calls (parked with the
  session's scope, so module-level state survives across calls);
- the daemon's copy of a module wins (the venv is appended, not prepended);
- jaato's own packages are never taken from the venv;
- the default is the daemon's environment;
- both processor load sites in ``LifecycleTools`` and the prefetch site
  pass the session's scope.

Every package name is unique per test, because what a test imports stays
in ``sys.modules`` for the rest of the process.
"""

from __future__ import annotations

import ast
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from jaato_server.shared.completion_processors import load_processors
from jaato_server.shared.dynamic_instructions import (
    RenderContext,
    expand_py_placeholders,
)
from jaato_server.shared.plugins.subagent.config import CompletionProcessor
from jaato_server.shared.script_imports import (
    resolve_script_import_scope,
    script_import_scope_for,
)
from jaato_server.shared.tests.reversion import Reversion

_LOADER = "jaato-server/jaato_server/shared/script_loader.py"
_IMPORTS = "jaato-server/jaato_server/shared/script_imports.py"
_DYNPROMPT = "jaato-server/jaato_server/shared/dynamic_instructions.py"
_LIFECYCLE = "jaato-server/jaato_server/shared/lifecycle_tools.py"

REVERSIONS = [
    Reversion(
        target=_LOADER,
        find="    return import_scope.wrap(fn) if import_scope is not None else fn\n",
        replace="    return fn\n",
        test="test_a_processor_imports_a_package_only_the_venv_has",
        because="the scope covers the load but not the call, so a script "
                "that imports lazily inside validate() cannot find the package",
    ),
    Reversion(
        target=_IMPORTS,
        find="            sys.path.extend(added)\n",
        replace="            sys.path[:0] = added\n",
        test="test_the_daemons_copy_of_a_module_wins",
        because="prepending the venv lets it replace a module the daemon "
                "already provides",
    ),
    Reversion(
        target=_IMPORTS,
        find="            sys.meta_path.insert(0, guard)\n",
        replace="            pass\n",
        test="test_jaato_names_are_never_taken_from_the_venv",
        because="without the guard a jaato-named package only the venv has "
                "is importable from it",
    ),
    Reversion(
        target=_IMPORTS,
        find="                    self._parked[name] = sys.modules.pop(name)\n",
        replace="                    pass\n",
        test="test_the_venv_leaves_the_process_between_calls",
        because="modules loaded from the venv stay in sys.modules, so the "
                "slot's next session can import them",
    ),
    Reversion(
        target=_DYNPROMPT,
        find="            import_scope=script_import_scope_for(context.session),\n",
        replace="            import_scope=None,\n",
        test="test_a_prefetch_script_imports_from_the_venv",
        because="prefetch scripts load without the session's scope",
    ),
    Reversion(
        target=_LIFECYCLE,
        find="\n                import_scope=script_import_scope_for(self._session),\n",
        replace="\n",
        test="test_every_processor_load_site_passes_the_sessions_scope",
        because="one of the two processor load sites loads without the scope",
    ),
]


# ── fixtures ──────────────────────────────────────────────────────────


def _venv(ws: Path) -> Path:
    """A venv-shaped tree under ``ws``; returns its site-packages."""
    site = ws / ".jaato" / "tool-venv" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    return site


def _package(site: Path, name: str, body: str) -> None:
    pkg = site / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(body)


def _session(ws: Path, mode: Optional[str] = "workspace_venv") -> Any:
    cli: Dict[str, Any] = {"workspace_venv": ".jaato/tool-venv"}
    if mode is not None:
        cli["script_imports"] = mode
    return SimpleNamespace(
        workspace_path=str(ws),
        declared_plugin_config=lambda name: cli if name == "cli" else None,
    )


def _processor(ws: Path, body: str):
    scripts = ws / ".jaato" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "gate.py").write_text(body)
    return CompletionProcessor(script="scripts/gate.py")


def _uniq(stem: str) -> str:
    return f"{stem}_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


# ── tests ─────────────────────────────────────────────────────────────


def test_a_processor_imports_a_package_only_the_venv_has(ws):
    pkg = _uniq("onlyvenv")
    lazy = _uniq("lazyvenv")
    site = _venv(ws)
    _package(site, pkg, "VALUE = 'from-venv'\n")
    _package(site, lazy, "VALUE = 'lazy-from-venv'\n")
    proc = _processor(ws, (
        f"import {pkg}\n"
        "def validate(payload, context):\n"
        f"    import {lazy}\n"
        f"    return [{pkg}.VALUE, {lazy}.VALUE]\n"
    ))
    session = _session(ws)
    loaded = load_processors([proc], str(ws), None,
                             import_scope=script_import_scope_for(session))
    assert loaded[0].load_error is None, loaded[0].load_error
    assert loaded[0].validate_fn({}, None) == ["from-venv", "lazy-from-venv"]


def test_without_the_opt_in_the_venv_is_not_seen(ws):
    pkg = _uniq("notseen")
    _package(_venv(ws), pkg, "VALUE = 1\n")
    proc = _processor(ws, f"import {pkg}\ndef validate(p, c):\n    return []\n")
    session = _session(ws, mode=None)
    assert script_import_scope_for(session) is None
    loaded = load_processors([proc], str(ws), None,
                             import_scope=script_import_scope_for(session))
    assert loaded[0].load_error is not None


def test_the_venv_leaves_the_process_between_calls(ws):
    pkg = _uniq("parked")
    site = _venv(ws)
    _package(site, pkg, "CALLS = []\n")
    proc = _processor(ws, (
        f"import {pkg}\n"
        "def validate(payload, context):\n"
        f"    {pkg}.CALLS.append(1)\n"
        f"    return [str(len({pkg}.CALLS))]\n"
    ))
    session = _session(ws)
    loaded = load_processors([proc], str(ws), None,
                             import_scope=script_import_scope_for(session))
    validate = loaded[0].validate_fn
    assert validate({}, None) == ["1"]
    assert pkg not in sys.modules
    assert str(site) not in sys.path
    # Parked with the session's scope, so the same module object comes back.
    assert validate({}, None) == ["2"]
    assert pkg not in sys.modules


def test_the_daemons_copy_of_a_module_wins(ws, monkeypatch):
    # A stdlib module the suite is unlikely to have imported; the venv
    # carries an impostor of it.
    name = "tabnanny"
    site = _venv(ws)
    (site / f"{name}.py").write_text("FROM_VENV = True\n")
    monkeypatch.delitem(sys.modules, name, raising=False)
    scope, _ = resolve_script_import_scope(
        {"workspace_venv": ".jaato/tool-venv", "script_imports": "workspace_venv"},
        str(ws),
    )
    assert scope is not None
    with scope:
        mod = __import__(name)
    assert not hasattr(mod, "FROM_VENV")


def test_jaato_names_are_never_taken_from_the_venv(ws):
    fake = _uniq("jaato_fake")
    _package(_venv(ws), fake, "VALUE = 'impostor'\n")
    scope, _ = resolve_script_import_scope(
        {"workspace_venv": ".jaato/tool-venv", "script_imports": "workspace_venv"},
        str(ws),
    )
    with scope:
        with pytest.raises(ModuleNotFoundError):
            __import__(fake)


def test_a_prefetch_script_imports_from_the_venv(ws):
    pkg = _uniq("prefetchvenv")
    _package(_venv(ws), pkg, "VALUE = 'prefetched'\n")
    scripts = ws / ".jaato" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "pf.py").write_text(
        f"import {pkg}\n"
        "def render(context, args):\n"
        f"    return {pkg}.VALUE\n"
    )
    ctx = RenderContext(session=_session(ws), runtime=None, registry=None,
                        workspace_path=str(ws), config_root=None)
    assert expand_py_placeholders("x {{!py:scripts/pf.py}} y", ctx) == "x prefetched y"


def test_every_processor_load_site_passes_the_sessions_scope():
    path = Path(__file__).resolve().parents[1] / "lifecycle_tools.py"
    calls = [
        node for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "load_processors"
    ]
    assert len(calls) == 2, "expected the two processor load sites"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "import_scope" in kw, f"line {call.lineno} loads without the scope"
        assert ast.unparse(kw["import_scope"]) == "script_import_scope_for(self._session)"
