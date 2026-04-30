"""Tests for ``shared.dynamic_instructions`` placeholder expansion."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from shared.dynamic_instructions import (
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

    def test_missing_script_renders_inline_marker(self, base_context: RenderContext):
        out = expand_py_placeholders(
            "Lookup: {{!py:scripts/does_not_exist.py}}", base_context,
        )
        assert out == "Lookup: [script not found: scripts/does_not_exist.py]"

    def test_missing_render_function_renders_load_error(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "no_render.py"
        script.write_text("def something_else(): pass\n")
        out = expand_py_placeholders(
            "{{!py:scripts/no_render.py}}", base_context,
        )
        assert "[script load error: scripts/no_render.py]" in out

    def test_render_exception_renders_error_marker(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "boom.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                raise RuntimeError("kaboom")
        """))
        out = expand_py_placeholders(
            "{{!py:scripts/boom.py}}", base_context,
        )
        assert "[script error: scripts/boom.py: kaboom]" in out

    def test_non_string_return_coerced_to_str(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        script = tmp_workspace / ".jaato" / "scripts" / "returns_dict.py"
        script.write_text(textwrap.dedent("""\
            def render(context, args):
                return {"k": "v"}
        """))
        out = expand_py_placeholders(
            "{{!py:scripts/returns_dict.py}}", base_context,
        )
        assert "{'k': 'v'}" in out

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


# --------------------------------------------------------------------
# Output-side: completion-artifact rendering
# --------------------------------------------------------------------

from dataclasses import dataclass

from shared.dynamic_instructions import (
    ArtifactRenderResult,
    render_completion_artifacts,
)


@dataclass
class _FakeArtifact:
    """Minimal stand-in for ``CompletionArtifact`` in unit tests.

    Avoids importing the subagent-config module (which has a heavy
    transitive dependency chain).  The render function only reads
    ``renderer``, ``output``, ``on_error`` via getattr, so any object
    matching that shape works.
    """
    renderer: str
    output: str
    on_error: str = "fail_completion"


class TestRenderCompletionArtifacts:
    def test_empty_list_returns_empty_result(self, base_context: RenderContext):
        result = render_completion_artifacts({}, base_context, [])
        assert isinstance(result, ArtifactRenderResult)
        assert result.written == [] and result.failed == [] and result.warned == []

    def test_basic_artifact_writes_file(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        renderer = tmp_workspace / ".jaato" / "scripts" / "policy_md.py"
        renderer.write_text(textwrap.dedent("""\
            def render(payload, context):
                return f"# Policy {payload['poliza_id']}\\nprima: {payload['prima_anual_eur']} EUR"
        """))
        artifact = _FakeArtifact(
            renderer="scripts/policy_md.py",
            output="output/{case_id}/policy.md",
        )
        payload = {"poliza_id": "POL-001", "prima_anual_eur": 759.38}
        # Note: case_id comes from agent_params (CASE-001) per the fixture
        result = render_completion_artifacts(payload, base_context, [artifact])
        assert len(result.written) == 1
        assert not result.failed and not result.warned
        out_path = Path(result.written[0])
        assert out_path.is_absolute()
        assert "CASE-001" in str(out_path)
        assert out_path.read_text(encoding="utf-8").startswith("# Policy POL-001")

    def test_renderer_not_found_fails(self, base_context: RenderContext):
        artifact = _FakeArtifact(
            renderer="scripts/missing.py",
            output="out.md",
        )
        result = render_completion_artifacts({}, base_context, [artifact])
        assert result.failed and "renderer not found" in result.failed[0][1]

    def test_renderer_not_found_warned_when_on_error_warn(
        self, base_context: RenderContext,
    ):
        artifact = _FakeArtifact(
            renderer="scripts/missing.py",
            output="out.md",
            on_error="warn",
        )
        result = render_completion_artifacts({}, base_context, [artifact])
        assert not result.failed and result.warned

    def test_renderer_exception_fails(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        (tmp_workspace / ".jaato" / "scripts" / "boom.py").write_text(
            "def render(payload, context): raise RuntimeError('kaboom')\n"
        )
        artifact = _FakeArtifact(
            renderer="scripts/boom.py",
            output="out.md",
        )
        result = render_completion_artifacts({}, base_context, [artifact])
        assert result.failed and "kaboom" in result.failed[0][1]

    def test_output_template_unknown_field_fails(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        (tmp_workspace / ".jaato" / "scripts" / "ok.py").write_text(
            'def render(payload, context): return "x"\n'
        )
        artifact = _FakeArtifact(
            renderer="scripts/ok.py",
            output="out/{nonexistent_field}/x.md",
        )
        result = render_completion_artifacts({}, base_context, [artifact])
        assert result.failed and "unknown field" in result.failed[0][1]

    def test_payload_fields_substitute_in_output_path(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        (tmp_workspace / ".jaato" / "scripts" / "ok.py").write_text(
            'def render(payload, context): return "content"\n'
        )
        artifact = _FakeArtifact(
            renderer="scripts/ok.py",
            output="output/{poliza_id}.md",
        )
        result = render_completion_artifacts(
            {"poliza_id": "POL-XYZ"}, base_context, [artifact],
        )
        assert len(result.written) == 1
        assert "POL-XYZ.md" in result.written[0]

    def test_bytes_return_writes_binary(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        (tmp_workspace / ".jaato" / "scripts" / "binary.py").write_text(
            'def render(payload, context): return b"\\x00\\x01\\x02"\n'
        )
        artifact = _FakeArtifact(
            renderer="scripts/binary.py",
            output="out.bin",
        )
        result = render_completion_artifacts({}, base_context, [artifact])
        assert len(result.written) == 1
        assert Path(result.written[0]).read_bytes() == b"\x00\x01\x02"

    def test_multiple_artifacts_independent(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        (tmp_workspace / ".jaato" / "scripts" / "a.py").write_text(
            'def render(p, c): return "A"\n'
        )
        (tmp_workspace / ".jaato" / "scripts" / "b.py").write_text(
            'def render(p, c): return "B"\n'
        )
        result = render_completion_artifacts(
            {},
            base_context,
            [
                _FakeArtifact(renderer="scripts/a.py", output="a.md"),
                _FakeArtifact(renderer="scripts/b.py", output="b.md"),
            ],
        )
        assert len(result.written) == 2

    def test_one_failure_does_not_skip_others(
        self, tmp_workspace: Path, base_context: RenderContext,
    ):
        (tmp_workspace / ".jaato" / "scripts" / "ok.py").write_text(
            'def render(p, c): return "ok"\n'
        )
        result = render_completion_artifacts(
            {},
            base_context,
            [
                _FakeArtifact(renderer="scripts/missing.py", output="x.md", on_error="warn"),
                _FakeArtifact(renderer="scripts/ok.py", output="y.md"),
            ],
        )
        assert len(result.written) == 1 and len(result.warned) == 1
