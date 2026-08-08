"""Tests for ``shared.dynamic_instructions`` placeholder expansion."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from shared.dynamic_instructions import (
    DynamicInstructionsError,
    RenderContext,
    expand_py_placeholders,
)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Workspace dir with a ``.jaato/scripts/`` folder ready to populate."""
    scripts_dir = tmp_path / ".jaato" / "scripts"
    scripts_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def base_context(tmp_workspace: Path) -> RenderContext:
    """A RenderContext anchored at the tmp workspace, no other handles."""
    return RenderContext(
        session=None,
        runtime=None,
        registry=None,
        workspace_path=str(tmp_workspace),
        config_root=None,
        agent_params={"case_id": "CASE-001", "tomador_dni": "12345678Z"},
        env=dict(os.environ),
    )


class TestExpandPyPlaceholders:
    def test_no_placeholder_returns_unchanged(self, base_context: RenderContext):
        content = "Hello, world. No placeholders here."
        assert expand_py_placeholders(content, base_context) == content

    def test_basic_substitution(self, tmp_workspace: Path, base_context: RenderContext):
        script = tmp_workspace / ".jaato" / "scripts" / "hello.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return "HELLO"
        """))
        out = expand_py_placeholders(
            "Greeting: {{!py:scripts/hello.py}}", base_context,
        )
        assert out == "Greeting: HELLO"

    def test_args_passed_through(self, tmp_workspace: Path, base_context: RenderContext):
        script = tmp_workspace / ".jaato" / "scripts" / "echo.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return f"args={args}"
        """))
        out = expand_py_placeholders(
            "Echo: {{!py:scripts/echo.py one two three}}", base_context,
        )
        assert out == "Echo: args=['one', 'two', 'three']"

    def test_agent_params_visible_to_script(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "use_params.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return f"DNI={context.agent_params['tomador_dni']}"
        """))
        out = expand_py_placeholders(
            "{{!py:scripts/use_params.py}}", base_context,
        )
        assert out == "DNI=12345678Z"

    # --- Optional-placeholder behaviour (server 0.6.48+ ?-modifier) ---
    #
    # Pre-0.6.48 the framework swallowed all error paths and substituted
    # an inline error sentinel.  Server 0.6.48+ moved that to opt-in via
    # the ``?`` modifier; default behaviour (without ``?``) raises
    # DynamicInstructionsError.  These tests pin the optional-mode
    # contract so consumers using ``{{!py?:script.py}}`` keep getting
    # the legacy swallow behaviour.

    def test_optional_missing_script_renders_inline_marker(
        self, base_context: RenderContext,
    ):
        out = expand_py_placeholders(
            "Lookup: {{!py?:scripts/does_not_exist.py}}", base_context,
        )
        assert out == "Lookup: [script not found: scripts/does_not_exist.py]"

    def test_optional_missing_render_function_renders_load_error(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "no_render.py"
        script.write_text("def something_else(): pass\n")
        out = expand_py_placeholders(
            "{{!py?:scripts/no_render.py}}", base_context,
        )
        assert "[script load error: scripts/no_render.py]" in out

    def test_optional_render_exception_renders_error_marker(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "boom.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                raise RuntimeError("kaboom")
        """))
        out = expand_py_placeholders(
            "{{!py?:scripts/boom.py}}", base_context,
        )
        assert "[script error: scripts/boom.py: kaboom]" in out

    def test_optional_non_string_return_coerced_to_str(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "returns_dict.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return {"k": "v"}
        """))
        out = expand_py_placeholders(
            "{{!py?:scripts/returns_dict.py}}", base_context,
        )
        assert "{'k': 'v'}" in out

    def test_optional_failure_sentinel_passed_through(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        """Optional placeholder + script returning [prefetch error: ...]
        sentinel: legacy swallow behaviour preserved.
        """
        script = tmp_workspace / ".jaato" / "scripts" / "self_fail.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return "[prefetch error: cannot resolve inputs]"
        """))
        out = expand_py_placeholders(
            "{{!py?:scripts/self_fail.py}}", base_context,
        )
        assert out == "[prefetch error: cannot resolve inputs]"

    # --- Strict-placeholder behaviour (server 0.6.48+ default) ---
    #
    # Without the ``?`` modifier, every error path raises
    # DynamicInstructionsError so the session-creation path can convert
    # it to a structured ErrorEvent and abort cleanly.  Closes the
    # silent-fabrication class diagnosed by 7:3 in the kb-enablement-2.0
    # cascade probe v6.

    def test_strict_missing_script_raises(self, base_context: RenderContext):
        with pytest.raises(DynamicInstructionsError) as excinfo:
            expand_py_placeholders(
                "{{!py:scripts/does_not_exist.py}}", base_context,
            )
        assert excinfo.value.script_ref == "scripts/does_not_exist.py"
        assert "script not found" in excinfo.value.reason

    def test_strict_missing_render_function_raises(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "no_render.py"
        script.write_text("def something_else(): pass\n")
        with pytest.raises(DynamicInstructionsError) as excinfo:
            expand_py_placeholders(
                "{{!py:scripts/no_render.py}}", base_context,
            )
        assert excinfo.value.script_ref == "scripts/no_render.py"
        assert "load failed" in excinfo.value.reason

    def test_strict_render_exception_raises(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "boom.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                raise RuntimeError("kaboom")
        """))
        with pytest.raises(DynamicInstructionsError) as excinfo:
            expand_py_placeholders(
                "{{!py:scripts/boom.py}}", base_context,
            )
        assert excinfo.value.script_ref == "scripts/boom.py"
        assert "kaboom" in excinfo.value.reason
        assert "RuntimeError" in excinfo.value.reason

    def test_strict_non_string_return_raises(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "returns_dict.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return {"k": "v"}
        """))
        with pytest.raises(DynamicInstructionsError) as excinfo:
            expand_py_placeholders(
                "{{!py:scripts/returns_dict.py}}", base_context,
            )
        assert excinfo.value.script_ref == "scripts/returns_dict.py"
        assert "non-string" in excinfo.value.reason
        assert "dict" in excinfo.value.reason

    def test_strict_prefetch_error_sentinel_raises(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        """The convention 7:3 documented: scripts return
        [prefetch error: ...] to deliberately signal failure.  Strict
        mode treats this as an abort signal — no silent fabrication.
        """
        script = tmp_workspace / ".jaato" / "scripts" / "self_fail.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return "[prefetch error: cannot resolve inputs]"
        """))
        with pytest.raises(DynamicInstructionsError) as excinfo:
            expand_py_placeholders(
                "{{!py:scripts/self_fail.py}}", base_context,
            )
        assert excinfo.value.script_ref == "scripts/self_fail.py"
        assert "[prefetch error:" in excinfo.value.reason

    def test_strict_script_error_sentinel_raises(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        """A script that catches its own exception and rebuilds the
        framework's [script error: ...] shape: still treated as
        deliberate failure signal.
        """
        script = tmp_workspace / ".jaato" / "scripts" / "rebuild_sentinel.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return "[script error: rebuild_sentinel.py: something failed]"
        """))
        with pytest.raises(DynamicInstructionsError):
            expand_py_placeholders(
                "{{!py:scripts/rebuild_sentinel.py}}", base_context,
            )

    def test_strict_success_passes_through(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        """Strict mode with a successful render: no changes vs legacy."""
        script = tmp_workspace / ".jaato" / "scripts" / "ok.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return "OK"
        """))
        out = expand_py_placeholders(
            "Result: {{!py:scripts/ok.py}}", base_context,
        )
        assert out == "Result: OK"

    def test_strict_and_optional_interleaved(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        """A template with one strict and one optional placeholder:
        the strict failure raises BEFORE the optional swallow could
        run.  Pins the order of operations.
        """
        bad = tmp_workspace / ".jaato" / "scripts" / "fails.py"
        bad.write_text(textwrap.dedent("""\
            def render(context, args):
                raise RuntimeError("nope")
        """))
        good = tmp_workspace / ".jaato" / "scripts" / "ok.py"
        good.write_text("def render(c, a): return 'OK'\n")
        # Strict placeholder appears first; raises immediately.
        with pytest.raises(DynamicInstructionsError):
            expand_py_placeholders(
                "{{!py:scripts/fails.py}}\n{{!py?:scripts/ok.py}}",
                base_context,
            )

    def test_non_failure_string_with_bracket_passes_through(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        """A string starting with `[` but NOT one of the framework's
        failure sentinels passes through verbatim.  Important so prose
        like "[INFO] config loaded" doesn't accidentally trigger abort.
        """
        script = tmp_workspace / ".jaato" / "scripts" / "info.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return "[INFO] all good"
        """))
        out = expand_py_placeholders(
            "{{!py:scripts/info.py}}", base_context,
        )
        assert out == "[INFO] all good"

    def test_multiple_placeholders_in_one_template(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        (tmp_workspace / ".jaato" / "scripts" / "a.py").write_text(
            'def render(c, a): return "ALPHA"\n'
        )
        (tmp_workspace / ".jaato" / "scripts" / "b.py").write_text(
            'def render(c, a): return "BETA"\n'
        )
        out = expand_py_placeholders(
            "Top: {{!py:scripts/a.py}}\nBottom: {{!py:scripts/b.py}}",
            base_context,
        )
        assert out == "Top: ALPHA\nBottom: BETA"

    def test_workspace_overrides_user_tier(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        # Confirm the workspace tier is consulted (resolve_script_path
        # already handles the precedence; this test just exercises
        # the integration).
        (tmp_workspace / ".jaato" / "scripts" / "from_ws.py").write_text(
            'def render(c, a): return "FROM_WORKSPACE"\n'
        )
        out = expand_py_placeholders(
            "{{!py:scripts/from_ws.py}}", base_context,
        )
        assert out == "FROM_WORKSPACE"

    def test_early_out_when_no_py_marker(self, base_context: RenderContext):
        # Content with {{!command}} (shell) but no {{!py:}} should pass
        # through untouched (this expander is not responsible for shell).
        content = "Status: {{!git status}}"
        assert expand_py_placeholders(content, base_context) == content
