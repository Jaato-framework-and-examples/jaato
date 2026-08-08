"""Tests for shared/script_loader.py — resolve + load user script symbols."""

from pathlib import Path
from unittest.mock import patch

from shared.script_loader import load_script_symbol, resolve_script_path


# ---------------------------------------------------------------------------
# resolve_script_path
# ---------------------------------------------------------------------------

class TestResolveScriptPath:

    def test_absolute_path_exists(self, tmp_path):
        script = tmp_path / "script.py"
        script.write_text("def execute(): pass\n")
        assert resolve_script_path(str(script)) == script

    def test_absolute_path_missing(self):
        assert resolve_script_path("/nonexistent/script.py") is None

    def test_workspace_tier(self, tmp_path):
        ws = tmp_path / "workspace"
        subdir = ws / ".jaato" / "policies"
        subdir.mkdir(parents=True)
        script = subdir / "eval.py"
        script.write_text("def evaluate(t, a, c): pass\n")
        assert (
            resolve_script_path("policies/eval.py", workspace_path=str(ws))
            == script
        )

    def test_workspace_tier_flat_file(self, tmp_path):
        ws = tmp_path / "workspace"
        jaato = ws / ".jaato"
        jaato.mkdir(parents=True)
        script = jaato / "loose.py"
        script.write_text("def execute(p, e, c): pass\n")
        assert (
            resolve_script_path("loose.py", workspace_path=str(ws)) == script
        )

    def test_home_tier_fallback(self, tmp_path):
        # Point Path.home() at a fake home with a script in .jaato/
        fake_home = tmp_path / "home"
        jaato = fake_home / ".jaato"
        jaato.mkdir(parents=True)
        script = jaato / "global.py"
        script.write_text("def execute(p, e, c): pass\n")

        with patch("shared.script_loader.Path.home", return_value=fake_home):
            assert resolve_script_path("global.py") == script

    def test_workspace_tier_wins_over_home(self, tmp_path):
        fake_home = tmp_path / "home"
        home_jaato = fake_home / ".jaato"
        home_jaato.mkdir(parents=True)
        home_script = home_jaato / "shared.py"
        home_script.write_text("def execute(p, e, c): pass  # home\n")

        ws = tmp_path / "workspace"
        ws_jaato = ws / ".jaato"
        ws_jaato.mkdir(parents=True)
        ws_script = ws_jaato / "shared.py"
        ws_script.write_text("def execute(p, e, c): pass  # workspace\n")

        with patch("shared.script_loader.Path.home", return_value=fake_home):
            assert (
                resolve_script_path("shared.py", workspace_path=str(ws))
                == ws_script
            )

    def test_relative_no_match_returns_none(self, tmp_path):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        with patch("shared.script_loader.Path.home", return_value=fake_home):
            assert resolve_script_path("nowhere.py", workspace_path=str(tmp_path)) is None

    def test_relative_without_workspace_still_checks_home(self, tmp_path):
        fake_home = tmp_path / "home"
        jaato = fake_home / ".jaato"
        jaato.mkdir(parents=True)
        script = jaato / "only_home.py"
        script.write_text("def execute(p, e, c): pass\n")

        with patch("shared.script_loader.Path.home", return_value=fake_home):
            assert resolve_script_path("only_home.py") == script


# ---------------------------------------------------------------------------
# load_script_symbol
# ---------------------------------------------------------------------------

class TestLoadScriptSymbol:

    def test_loads_named_symbol(self, tmp_path):
        script = tmp_path / "good.py"
        script.write_text(
            "def execute(params, event, ctx):\n"
            "    return 'ok'\n"
        )
        fn = load_script_symbol(script, symbol="execute")
        assert fn is not None
        assert fn({}, {}, None) == "ok"

    def test_custom_symbol_name(self, tmp_path):
        script = tmp_path / "custom.py"
        script.write_text("def evaluate(t, a, c): return 'allow'\n")
        fn = load_script_symbol(script, symbol="evaluate")
        assert fn is not None
        assert fn(None, None, None) == "allow"

    def test_missing_symbol_returns_none(self, tmp_path):
        script = tmp_path / "empty.py"
        script.write_text("x = 42\n")
        assert load_script_symbol(script, symbol="execute") is None

    def test_non_callable_symbol_returns_none(self, tmp_path):
        script = tmp_path / "notcallable.py"
        script.write_text("execute = 'not a function'\n")
        assert load_script_symbol(script, symbol="execute") is None

    def test_syntax_error_returns_none(self, tmp_path):
        script = tmp_path / "broken.py"
        script.write_text("def execute(:\n    pass\n")  # syntax error
        assert load_script_symbol(script, symbol="execute") is None

    def test_import_time_exception_returns_none(self, tmp_path):
        script = tmp_path / "boom.py"
        script.write_text("raise RuntimeError('import failed')\n")
        assert load_script_symbol(script, symbol="execute") is None

    def test_module_prefix_does_not_affect_outcome(self, tmp_path):
        script = tmp_path / "prefixed.py"
        script.write_text("def execute(p, e, c): return 1\n")
        fn = load_script_symbol(
            script, symbol="execute", module_prefix="_jaato_reactor"
        )
        assert fn is not None
        assert fn(None, None, None) == 1

    def test_same_filename_different_dirs_do_not_collide(self, tmp_path):
        # Two scripts with the same stem living in different directories
        # must load independently without sys.modules clobbering.
        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        a_dir.mkdir()
        b_dir.mkdir()
        (a_dir / "same.py").write_text("def execute(p, e, c): return 'a'\n")
        (b_dir / "same.py").write_text("def execute(p, e, c): return 'b'\n")

        fn_a = load_script_symbol(a_dir / "same.py", symbol="execute")
        fn_b = load_script_symbol(b_dir / "same.py", symbol="execute")
        assert fn_a(None, None, None) == "a"
        assert fn_b(None, None, None) == "b"


# ---------------------------------------------------------------------------
# Helper-module invalidation across load_script_symbol calls
#
# Server 0.6.60+: when a handler under <config_root>/.jaato/scripts/
# imports a sibling helper, the helper lands in sys.modules. Without
# invalidation, daemon-cached helper objects shadow on-disk edits until
# daemon restart — empirically demonstrated 2026-05-07 by the
# kb-enablement-2.0 cascade slice ε.11 (added is_handled symbol to
# _session_chain.py; cascade_after_discovery's from-import died with
# `cannot import name 'is_handled' from '_session_chain'` because the
# daemon had stale _session_chain cached).
#
# These tests pin the mtime-tracking + reload-on-stale contract.
# ---------------------------------------------------------------------------

import os
import sys
import time

import pytest


def _make_jaato_scripts_layout(root: Path) -> Path:
    """Create a .jaato/scripts/reactors/ layout under ``root``.

    Returns the reactors directory. The parent path is structured so
    ``_scripts_root_for`` finds a ``.jaato/scripts/`` ancestor.
    """
    reactors = root / ".jaato" / "scripts" / "reactors"
    reactors.mkdir(parents=True)
    return reactors


def _bump_mtime(path: Path) -> None:
    """Force the file's mtime to advance.

    ``os.path.getmtime`` resolution is filesystem-dependent (often 1s
    on ext4); pad with a small sleep + explicit ``os.utime`` to
    guarantee the new mtime sorts AFTER the recorded one.
    """
    time.sleep(0.05)
    now = time.time() + 1.0  # +1s to clear any fs-resolution roundoff
    os.utime(path, (now, now))


@pytest.fixture(autouse=True)
def _clear_helper_tracker():
    """Reset the module-level tracker between tests for isolation.

    Drops sys.modules entries whose source path is under any
    pytest-tmp_path subtree — that's the only place this test file
    creates helper modules. Bare module names like ``_helper`` would
    otherwise leak across tests via Python's import cache.
    """
    from shared import script_loader as sl

    def _drop_test_helpers() -> None:
        to_drop = []
        for name, mod in list(sys.modules.items()):
            mod_file = getattr(mod, "__file__", "") or ""
            mod_file = str(mod_file)
            # Test fixtures only ever live under pytest's tmp_path
            # (typically /tmp/pytest-of-* on Linux, /private/var on
            # macOS). Be generous with the prefix match — anything
            # touching pytest's tmp area is a fixture-created module.
            if "/pytest-of-" in mod_file or "/pytest-" in mod_file:
                to_drop.append(name)
        for name in to_drop:
            sys.modules.pop(name, None)

    with sl._tracked_helpers_lock:
        sl._tracked_helpers.clear()
    _drop_test_helpers()
    yield
    with sl._tracked_helpers_lock:
        sl._tracked_helpers.clear()
    _drop_test_helpers()


class TestHelperInvalidationOnSecondLoad:
    """Edits to a helper module become visible on the NEXT load_script_symbol
    call (not deferred to daemon restart)."""

    def test_helper_edit_visible_on_next_load(self, tmp_path):
        reactors = _make_jaato_scripts_layout(tmp_path)
        helper = reactors / "_helper.py"
        handler = reactors / "handler.py"

        # v1 of the helper: only ``existing`` symbol.
        helper.write_text("def existing(): return 'v1'\n")
        handler.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "from _helper import existing\n"
            "def execute(p, e, c): return existing()\n"
        )

        fn1 = load_script_symbol(handler, symbol="execute")
        assert fn1 is not None
        assert fn1(None, None, None) == "v1"

        # Edit helper: change return value + bump mtime.
        helper.write_text("def existing(): return 'v2'\n")
        _bump_mtime(helper)

        # Second load — without invalidation, this would still see v1
        # because _helper is cached in sys.modules. With 0.6.60+
        # invalidation, the mtime check triggers an importlib.reload.
        fn2 = load_script_symbol(handler, symbol="execute")
        assert fn2 is not None
        assert fn2(None, None, None) == "v2"

    def test_helper_new_symbol_visible_on_next_load(self, tmp_path):
        """The empirical 7:3 case: helper gets a NEW symbol that the
        next handler-load tries to import. Without invalidation, the
        from-import raises ImportError on the new symbol."""
        reactors = _make_jaato_scripts_layout(tmp_path)
        helper = reactors / "_helper.py"
        handler = reactors / "handler.py"

        # v1 of the helper: only ``existing`` symbol.
        helper.write_text("def existing(): return 'a'\n")
        handler.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "from _helper import existing\n"
            "def execute(p, e, c): return existing()\n"
        )
        fn1 = load_script_symbol(handler, symbol="execute")
        assert fn1 is not None

        # Edit BOTH: helper gains a new symbol; handler now imports it.
        helper.write_text(
            "def existing(): return 'a'\n"
            "def added(): return 'b'\n"
        )
        _bump_mtime(helper)
        handler.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "from _helper import existing, added  # <- new symbol\n"
            "def execute(p, e, c): return added()\n"
        )

        # This is the 7:3 reproducer shape: helper edit + handler edit
        # that depends on the new helper symbol. Without 0.6.60's mtime
        # invalidation, the second load_script_symbol would die with
        # ``cannot import name 'added' from '_helper'``.
        fn2 = load_script_symbol(handler, symbol="execute")
        assert fn2 is not None
        assert fn2(None, None, None) == "b"

    def test_helper_unchanged_no_reload(self, tmp_path):
        """When the helper isn't modified, ``importlib.reload`` is NOT
        called. Avoids spurious reload churn on hot-path script loads."""
        reactors = _make_jaato_scripts_layout(tmp_path)
        helper = reactors / "_helper.py"
        handler = reactors / "handler.py"
        helper.write_text("def existing(): return 'v1'\n")
        handler.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "from _helper import existing\n"
            "def execute(p, e, c): return existing()\n"
        )

        # First load establishes the tracker.
        load_script_symbol(handler, symbol="execute")

        with patch("shared.script_loader.importlib.reload") as mock_reload:
            # Second load with helper unchanged.
            load_script_symbol(handler, symbol="execute")
            mock_reload.assert_not_called()


class TestHelperInvalidationFiltering:
    """Only modules under .jaato/scripts/ are tracked. Stdlib + 3rd-party
    + framework modules are NOT re-stat'd or reloaded across calls."""

    def test_stdlib_imports_not_tracked(self, tmp_path):
        from shared import script_loader as sl

        reactors = _make_jaato_scripts_layout(tmp_path)
        handler = reactors / "handler.py"
        handler.write_text(
            "import json  # stdlib — must NOT land in tracker\n"
            "def execute(p, e, c): return json.dumps({})\n"
        )

        load_script_symbol(handler, symbol="execute")

        # No stdlib path should appear in tracker.
        for tracked in sl._tracked_helpers:
            assert "/.jaato/scripts/" in tracked, (
                f"Tracker contains a non-jaato-scripts path: {tracked}. "
                "Stdlib / 3rd-party / framework modules must be filtered "
                "out — they are out of scope for the helper-invalidation "
                "contract."
            )

    def test_handler_outside_jaato_scripts_no_tracking(self, tmp_path):
        """Permission evaluators (loaded from ``shared/plugins/permission/``)
        and other call sites that don't use ``.jaato/scripts/`` paths
        are not affected — _scripts_root_for returns None, no helpers
        are tracked."""
        from shared import script_loader as sl

        # Handler outside any .jaato/scripts/ tree.
        helper = tmp_path / "_helper.py"
        handler = tmp_path / "handler.py"
        helper.write_text("def existing(): return 'v1'\n")
        handler.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "from _helper import existing\n"
            "def execute(p, e, c): return existing()\n"
        )

        load_script_symbol(handler, symbol="execute")

        # Tracker stays empty — handler isn't under .jaato/scripts/.
        assert len(sl._tracked_helpers) == 0


class TestHelperInvalidationEdgeCases:
    """Robustness around deletes, syntax errors, and concurrent loads."""

    def test_helper_deleted_removes_from_tracker(self, tmp_path):
        from shared import script_loader as sl

        reactors = _make_jaato_scripts_layout(tmp_path)
        helper = reactors / "_helper.py"
        handler = reactors / "handler.py"
        helper.write_text("def existing(): return 'v1'\n")
        handler.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "from _helper import existing\n"
            "def execute(p, e, c): return existing()\n"
        )
        load_script_symbol(handler, symbol="execute")
        # Tracker stores whatever the resolver produced; pick up the
        # key by inspection rather than reconstructing it (path
        # normalization details aren't part of the public contract).
        assert len(sl._tracked_helpers) == 1
        helper_path = next(iter(sl._tracked_helpers.keys()))
        assert helper_path.endswith("_helper.py")

        # Delete the helper. Next load's stale-check should remove it
        # from the tracker (file no longer exists). No exception
        # propagates.
        helper.unlink()

        # Now write a new handler that doesn't reference _helper, so
        # the load can proceed without re-trying the broken from-import.
        handler.write_text("def execute(p, e, c): return 'standalone'\n")
        _bump_mtime(handler)

        load_script_symbol(handler, symbol="execute")

        # Tracker no longer carries the deleted helper.
        assert helper_path not in sl._tracked_helpers

    def test_syntax_error_in_helper_logs_warning_keeps_tracker(self, tmp_path, caplog):
        """When a helper edit introduces a syntax error, the reload fails
        but the tracker preserves the OLD mtime so the next call retries
        after the operator fixes the syntax."""
        import logging
        from shared import script_loader as sl

        reactors = _make_jaato_scripts_layout(tmp_path)
        helper = reactors / "_helper.py"
        handler = reactors / "handler.py"
        helper.write_text("def existing(): return 'v1'\n")
        handler.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "from _helper import existing\n"
            "def execute(p, e, c): return existing()\n"
        )
        load_script_symbol(handler, symbol="execute")
        # Pick up the tracker's key by inspection (see deletion test
        # for rationale).
        assert len(sl._tracked_helpers) == 1
        helper_path = next(iter(sl._tracked_helpers.keys()))
        original_mtime = sl._tracked_helpers[helper_path]

        # Edit helper with a syntax error.
        helper.write_text("def existing( syntaxerror\n")
        _bump_mtime(helper)

        with caplog.at_level(logging.WARNING):
            # Refresh fires; reload fails; warning logged; tracker keeps
            # OLD mtime so the next refresh retries after a fix.
            load_script_symbol(handler, symbol="execute")

        # Tracker should NOT have advanced its recorded mtime past the
        # broken edit — the operator's next fix-and-save needs to be
        # detected as stale.
        assert sl._tracked_helpers[helper_path] == original_mtime
        # Warning was logged.
        assert any(
            "Failed to reload helper" in record.message
            for record in caplog.records
        )
