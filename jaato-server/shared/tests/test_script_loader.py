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
