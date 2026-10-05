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
* the template actually calls the preload (an AST check of
  ``_run_template_mode``: the behavioural cases call ``preload()``
  directly, so they cannot see the call being dropped);
* a missing entry is skipped quietly, a broken one with a warning, and
  neither raises.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jaato_server.server.runner import template_preload
from jaato_server.shared.tests.reversion import Reversion

_PRELOAD = "jaato-server/jaato_server/server/runner/template_preload.py"
_MAIN = "jaato-server/jaato_server/server/runner/__main__.py"
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
        find="    template_preload.log_report(template_preload.preload())\n",
        replace="",
        because="the template stops preloading, whatever the list says",
        test="test_the_template_calls_the_preload",
    ),
]


# The subprocess does, in order, what the template does at startup
# (runner-tier discovery, then the preload), then what a pool slot does
# on its first session.  A fresh interpreter, because the test process
# has already imported most of the tree.
_PROBE = textwrap.dedent(
    """
    import gc, json, os, sys, tempfile, threading
    from jaato_server.shared.plugins.registry import PluginRegistry
    PluginRegistry().discover(tier_filter="runner")
    from jaato_server.server.runner import template_preload
    report = template_preload.preload()
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
        "every slot imports them privately after the fork.  Add them to "
        "BOOTSTRAP_MODULES in server/runner/template_preload.py (after "
        "checking they read no environment at import): "
        + ", ".join(probe["new"])
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
    assert "template_preload.preload" in calls, (
        "_run_template_mode no longer calls template_preload.preload()"
    )


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
