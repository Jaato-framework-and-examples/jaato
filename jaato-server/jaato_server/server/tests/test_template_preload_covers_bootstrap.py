"""The pre-warm template imports what ``session.bootstrap`` imports.

The pool template forks slots so a session inherits a warm interpreter.
It used to warm only plugin discovery, while what a bootstrap actually
spends its time importing (the session stack, ``anthropic``, ``mcp``,
lazily imported helpers) was imported after the fork: about 2 s per
session, and about 86 MB of private memory per serving slot, because a
module imported after ``fork()`` is not shared with the template.

Covered here, against a fresh interpreter doing what the template does:

* a real ``bootstrap_session`` (echo provider) after the template's
  warm-up imports no jaato module and no new third-party package;
* the warm-up leaves one thread alive, so the template can fork;
* the heap is frozen (``gc.freeze``), so a slot's collector does not
  dirty the pages it inherited;
* the template actually plans and preloads (an AST check of
  ``_run_template_mode``: the behavioural cases call ``plan()`` and
  ``preload()`` directly, so they cannot see the call being dropped);
* ``PLUGIN_PRELOAD`` declarations: an out-of-tree one is honoured only
  for a distribution listed in ``JAATO_PLUGIN_ALLOW_PRELOAD``, a
  malformed one is ignored, and a declaration is read from source
  without importing its package;
* a missing entry is skipped quietly, a broken one with a warning, and
  neither raises.
"""
from __future__ import annotations

import ast
import importlib
import json
import logging
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from jaato_server.server.runner import template_preload
from jaato_server.shared.tests.reversion import Reversion

_PRELOAD = "jaato-server/jaato_server/server/runner/template_preload.py"
_MAIN = "jaato-server/jaato_server/server/runner/__main__.py"
_MCP = "jaato-server/jaato_server/shared/plugins/mcp/__init__.py"
_MAIN_PATH = Path(__file__).resolve().parents[1] / "runner" / "__main__.py"

REVERSIONS = [
    Reversion(
        target=_PRELOAD,
        find='''    "jaato_server.server.runner.session",
    "jaato_server.shared.jaato_client",
    "jaato_server.shared.jaato_runtime",
    "jaato_server.shared.jaato_session",
    "anthropic",
''',
        replace="",
        because="the session stack is imported after the fork again, so "
                "every slot imports and holds its own copy",
        test="test_a_bootstrap_after_the_warm_up_imports_nothing_new",
    ),
    Reversion(
        target=_PRELOAD,
        find="    gc.freeze()\n",
        replace="",
        because="the first collection in each slot copies every inherited "
                "page it touches",
        test="test_the_warm_up_freezes_the_heap_and_stays_single_threaded",
    ),
    Reversion(
        target=_MAIN,
        find=(
            "    template_preload.log_report(\n"
            "        template_preload.preload(preload_plan.modules), "
            "preload_plan)\n"
        ),
        replace="",
        because="the template stops preloading, whatever the list says",
        test="test_the_template_calls_the_preload",
    ),
    Reversion(
        target=_MCP,
        find=(
            "PLUGIN_PRELOAD = (\n"
            '    "mcp",\n'
            '    "mcp.client.stdio",\n'
            '    "jaato_server.shared.mcp_context_manager",\n'
            ")\n"
        ),
        replace="",
        because="the mcp plugin no longer declares what it imports lazily, "
                "so every slot imports mcp after the fork",
        test="test_a_bootstrap_after_the_warm_up_imports_nothing_new",
    ),
    Reversion(
        target=_PRELOAD,
        find=(
            '        if not builtin and normalize_distribution(dist or "") '
            "not in allowed:\n"
        ),
        replace="        if False:\n",
        because="an out-of-tree declaration runs its import-time code in "
                "the template with nobody having opted in",
        test="test_an_out_of_tree_declaration_needs_its_distribution_listed",
    ),
]


# The subprocess does, in order, what the template does at startup
# (runner-tier discovery, the plan, the preload), then what a pool slot
# does on its first session.  A fresh interpreter, because the test
# process has already imported most of the tree.
_PROBE = textwrap.dedent(
    """
    import gc, json, os, sys, tempfile, threading
    from jaato_server.shared.plugins.registry import PluginRegistry
    registry = PluginRegistry()
    registry.discover(tier_filter="runner")
    from jaato_server.server.runner import template_preload
    report = template_preload.preload(template_preload.plan(registry).modules)
    out = {"threads": report.threads, "frozen": gc.get_freeze_count(),
           "unfrozen": len(gc.get_objects()), "failed": report.failed}
    before = set(sys.modules)
    from jaato_server.server.runner.session import bootstrap_session
    from jaato_server.shared.session_envelope import SessionInitEnvelope
    ws = tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, ".jaato"))
    host = bootstrap_session(SessionInitEnvelope(
        session_id="preload_probe", workspace_path=ws, profile_name="",
        provider_name="echo", model_name="echo",
        config_root=os.path.join(ws, ".jaato")))
    out["ready"] = host.is_ready
    stdlib = sys.stdlib_module_names
    prior_tops = {m.split(".")[0] for m in before}
    out["new"] = sorted(
        m for m in set(sys.modules) - before
        if m.startswith("jaato")
        or (m.split(".")[0] not in stdlib
            and m.split(".")[0] not in prior_tops)
    )
    print("PROBE " + json.dumps(out), flush=True)
    os._exit(0)
    """
)


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> dict:
    home = tmp_path_factory.mktemp("home")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("JAATO_", "OTEL_", "LANGFUSE_"))}
    env["HOME"] = str(home)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, cwd=str(home),
        capture_output=True, text=True, timeout=300,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE ")]
    assert lines, (
        f"probe produced no result (exit {proc.returncode}):\n"
        f"{proc.stderr[-4000:]}"
    )
    return json.loads(lines[-1][len("PROBE "):])


def test_a_bootstrap_after_the_warm_up_imports_nothing_new(probe):
    assert probe["ready"], "the probe's bootstrap did not produce a session"
    assert probe["new"] == [], (
        "session.bootstrap imported modules the pool template did not, so "
        "every slot imports them privately after the fork.  Declare each in "
        "the PLUGIN_PRELOAD of the plugin or provider package that imports "
        "it, or in CORE_MODULES in server/runner/template_preload.py when "
        "the framework does (after checking it reads no environment at "
        "import): " + ", ".join(probe["new"])
    )


def test_the_warm_up_freezes_the_heap_and_stays_single_threaded(probe):
    assert probe["threads"] == 1, "a preloaded module started a thread"
    # Something earlier in discovery freezes a few hundred objects of its
    # own, so the frozen count alone proves nothing; what matters is that
    # the preloaded heap is in the permanent generation, leaving almost
    # nothing for a slot's collector to touch.
    assert probe["unfrozen"] < probe["frozen"] // 20, (
        f"the heap was not frozen after the preload: {probe['unfrozen']} "
        f"tracked objects outside the permanent generation, "
        f"{probe['frozen']} inside"
    )
    assert probe["failed"] == {}, f"preload entries failed: {probe['failed']}"


def test_the_template_calls_the_preload():
    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"))
    func = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_run_template_mode"
    )
    calls = {
        ast.unparse(n.func) for n in ast.walk(func) if isinstance(n, ast.Call)
    }
    for name in ("template_preload.plan", "template_preload.preload"):
        assert name in calls, f"_run_template_mode no longer calls {name}()"


@pytest.fixture(autouse=True)
def _no_freeze_in_this_process(monkeypatch):
    """The in-process cases call preload(); freezing pytest's own heap would
    keep every object alive so far out of later tests' collections."""
    monkeypatch.setattr(template_preload.gc, "freeze", lambda: None)


def test_a_missing_entry_is_skipped_quietly_and_a_broken_one_loudly(
    caplog, monkeypatch,
):
    real = template_preload.importlib.import_module

    def _import(name):
        if name == "broken_entry":
            raise RuntimeError("import-time failure")
        return real(name)

    monkeypatch.setattr(template_preload.importlib, "import_module", _import)
    caplog.set_level(logging.DEBUG, logger="jaato_server.server.runner.template")
    report = template_preload.preload(
        ["json", "jaato_no_such_module_xyz", "broken_entry"]
    )
    assert report.loaded == ["json"]
    assert report.missing == ["jaato_no_such_module_xyz"]
    assert list(report.failed) == ["broken_entry"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "broken_entry" in warnings[0].getMessage()


def test_a_missing_dependency_of_an_installed_entry_is_a_failure(monkeypatch):
    def _import(name):
        raise ModuleNotFoundError("No module named 'dep'", name="dep")

    monkeypatch.setattr(template_preload.importlib, "import_module", _import)
    report = template_preload.preload(["installed_entry"])
    assert report.missing == []
    assert "installed_entry" in report.failed


# ---------------------------------------------------------------- declarations


class _Origin:
    def __init__(self, module, distribution=None):
        self.module = module
        self.distribution = distribution


class _Registry:
    def __init__(self, *origins):
        self._origins = {o.module: o for o in origins}

    def get_plugin_sources(self):
        return dict(self._origins)


@pytest.fixture
def fake_package(monkeypatch):
    """An imported out-of-tree package ``acme_preload_pkg`` declaring preloads."""

    def _make(declaration):
        mod = types.ModuleType("acme_preload_pkg")
        mod.PLUGIN_PRELOAD = declaration
        monkeypatch.setitem(sys.modules, "acme_preload_pkg", mod)
        monkeypatch.setattr(template_preload, "_owners_from_providers",
                            lambda: [])
        return _Registry(_Origin("acme_preload_pkg.plugin", "acme-tools"))

    monkeypatch.delenv("JAATO_PLUGIN_ALLOW_PRELOAD", raising=False)
    return _make


def test_an_out_of_tree_declaration_needs_its_distribution_listed(
    fake_package, monkeypatch,
):
    registry = fake_package(("acme_preload_dep",))
    refused = template_preload.plan(registry)
    assert "acme_preload_dep" not in refused.modules
    assert "JAATO_PLUGIN_ALLOW_PRELOAD" in refused.ignored["acme_preload_pkg"]

    monkeypatch.setenv("JAATO_PLUGIN_ALLOW_PRELOAD", "Acme_Tools")
    honoured = template_preload.plan(registry)
    assert "acme_preload_dep" in honoured.modules
    assert honoured.declared == {"acme_preload_pkg": ("acme_preload_dep",)}
    core = template_preload.CORE_MODULES
    assert honoured.modules[:len(core)] == core


def test_a_malformed_declaration_is_ignored(fake_package, monkeypatch):
    monkeypatch.setenv("JAATO_PLUGIN_ALLOW_PRELOAD", "acme-tools")
    result = template_preload.plan(fake_package("not-a-tuple"))
    assert result.ignored["acme_preload_pkg"].startswith("malformed")
    assert result.modules == template_preload.CORE_MODULES


def test_a_declaration_is_read_without_importing_its_package(
    tmp_path, monkeypatch,
):
    pkg = tmp_path / "acme_unimported_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        'PLUGIN_PRELOAD = ("json", "acme_unimported_pkg.sub")\n'
        'raise RuntimeError("importing this package runs its code")\n'
    )
    (tmp_path / "acme_computed_pkg").mkdir()
    (tmp_path / "acme_computed_pkg" / "__init__.py").write_text(
        'PLUGIN_PRELOAD = tuple(["json"])\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert template_preload.read_declaration("acme_unimported_pkg") == (
        "json", "acme_unimported_pkg.sub")
    assert "acme_unimported_pkg" not in sys.modules
    with pytest.raises(ValueError):
        template_preload.read_declaration("acme_computed_pkg")


def test_every_in_tree_declaration_is_a_literal_matching_the_package():
    """The template reads an unimported package's declaration from source,
    so a computed one would read differently there than in a live import."""
    owners = [p for p, _, builtin in template_preload._owners_from_providers()
              if builtin]
    plugins = Path(template_preload.__file__).resolve().parents[2] \
        / "shared" / "plugins"
    owners += [f"jaato_server.shared.plugins.{d.name}"
               for d in sorted(plugins.iterdir())
               if (d / "__init__.py").is_file()]
    checked = 0
    for package in owners:
        literal = template_preload._literal_declaration(package)
        if literal is None:
            continue
        live = getattr(importlib.import_module(package), "PLUGIN_PRELOAD")
        assert tuple(literal) == tuple(live), package
        checked += 1
    assert checked >= 7
