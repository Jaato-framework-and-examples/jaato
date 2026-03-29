"""Tests for runtime-injectable permission evaluators."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shared.plugins.permission.evaluator import (
    EvalContext,
    PolicyDecision,
    _resolve_evaluator_path,
    load_evaluators,
    run_evaluator,
)


# ---------------------------------------------------------------------------
# EvalContext
# ---------------------------------------------------------------------------

class TestEvalContext:

    def test_defaults(self):
        ctx = EvalContext(tool_name="cli", args={"cmd": "ls"})
        assert ctx.tool_name == "cli"
        assert ctx.agent_type == "main"
        assert ctx.agent_name is None
        assert ctx.session_id is None
        assert ctx.workspace_path is None
        assert ctx.extra == {}

    def test_all_fields(self):
        ctx = EvalContext(
            tool_name="cli",
            args={"cmd": "ls"},
            agent_type="subagent",
            agent_name="researcher",
            session_id="20260329_140505",
            workspace_path="/home/user/project",
            extra={"custom": True},
        )
        assert ctx.agent_type == "subagent"
        assert ctx.extra["custom"] is True


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestResolveEvaluatorPath:

    def test_absolute_path_exists(self, tmp_path):
        script = tmp_path / "eval.py"
        script.write_text("def evaluate(t, a, c): pass")
        assert _resolve_evaluator_path(str(script)) == script

    def test_absolute_path_missing(self):
        assert _resolve_evaluator_path("/nonexistent/eval.py") is None

    def test_relative_workspace_path(self, tmp_path):
        ws = tmp_path / "workspace"
        jaato_dir = ws / ".jaato" / "policies"
        jaato_dir.mkdir(parents=True)
        script = jaato_dir / "eval.py"
        script.write_text("def evaluate(t, a, c): pass")
        result = _resolve_evaluator_path("policies/eval.py", workspace_path=str(ws))
        assert result == script

    def test_relative_no_workspace_no_home(self, tmp_path):
        # No workspace, no ~/.jaato match
        result = _resolve_evaluator_path("nonexistent/eval.py")
        # May or may not find in ~/.jaato — just verify no crash
        assert result is None or result.is_file()


# ---------------------------------------------------------------------------
# Load evaluators
# ---------------------------------------------------------------------------

class TestLoadEvaluators:

    def test_load_valid_script(self, tmp_path):
        script = tmp_path / ".jaato" / "eval.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "from shared.plugins.permission.evaluator import PolicyDecision\n"
            "def evaluate(tool_name, args, context):\n"
            "    return PolicyDecision.DENY\n"
        )
        result = load_evaluators(
            {"default": "eval.py"},
            workspace_path=str(tmp_path),
        )
        assert "default" in result
        assert callable(result["default"])

    def test_load_missing_script_skipped(self, tmp_path):
        result = load_evaluators(
            {"default": "nonexistent.py"},
            workspace_path=str(tmp_path),
        )
        assert result == {}

    def test_load_script_without_evaluate_function(self, tmp_path):
        script = tmp_path / ".jaato" / "broken.py"
        script.parent.mkdir(parents=True)
        script.write_text("x = 42\n")
        result = load_evaluators(
            {"default": "broken.py"},
            workspace_path=str(tmp_path),
        )
        assert result == {}

    def test_load_multiple_evaluators(self, tmp_path):
        jaato = tmp_path / ".jaato"
        jaato.mkdir()
        (jaato / "global.py").write_text(
            "def evaluate(t, a, c): return 'fallback'\n"
        )
        (jaato / "cli.py").write_text(
            "def evaluate(t, a, c): return 'deny'\n"
        )
        result = load_evaluators(
            {"default": "global.py", "cli_based_tool": "cli.py"},
            workspace_path=str(tmp_path),
        )
        assert len(result) == 2
        assert "default" in result
        assert "cli_based_tool" in result


# ---------------------------------------------------------------------------
# Run evaluator
# ---------------------------------------------------------------------------

class TestRunEvaluator:

    def _ctx(self, tool="cli", args=None):
        return EvalContext(tool_name=tool, args=args or {})

    def test_allow_decision(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.ALLOW,
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result == PolicyDecision.ALLOW

    def test_deny_decision(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result == PolicyDecision.DENY

    def test_fallback_decision(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.FALLBACK,
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result == PolicyDecision.FALLBACK

    def test_tool_specific_takes_precedence(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.ALLOW,
            "cli_based_tool": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli_based_tool", {}, self._ctx("cli_based_tool"))
        assert result == PolicyDecision.DENY

    def test_falls_back_to_default(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.ALLOW,
            "other_tool": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli_based_tool", {}, self._ctx("cli_based_tool"))
        assert result == PolicyDecision.ALLOW

    def test_no_matching_evaluator_returns_fallback(self):
        evaluators = {
            "other_tool": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli_based_tool", {}, self._ctx("cli_based_tool"))
        assert result == PolicyDecision.FALLBACK

    def test_empty_evaluators_returns_fallback(self):
        result = run_evaluator({}, "cli", {}, self._ctx())
        assert result == PolicyDecision.FALLBACK

    def test_exception_in_evaluator_returns_fallback(self):
        def bad_eval(t, a, c):
            raise RuntimeError("boom")

        evaluators = {"default": bad_eval}
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result == PolicyDecision.FALLBACK

    def test_string_return_value_accepted(self):
        evaluators = {
            "default": lambda t, a, c: "deny",
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result == PolicyDecision.DENY

    def test_invalid_return_value_returns_fallback(self):
        evaluators = {
            "default": lambda t, a, c: 42,
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result == PolicyDecision.FALLBACK

    def test_evaluator_receives_correct_args(self):
        received = {}

        def capture_eval(tool_name, args, context):
            received["tool"] = tool_name
            received["args"] = args
            received["context"] = context
            return PolicyDecision.FALLBACK

        evaluators = {"default": capture_eval}
        ctx = EvalContext(
            tool_name="writeFile",
            args={"path": "/tmp/x"},
            agent_type="subagent",
            agent_name="writer",
        )
        run_evaluator(evaluators, "writeFile", {"path": "/tmp/x"}, ctx)
        assert received["tool"] == "writeFile"
        assert received["args"]["path"] == "/tmp/x"
        assert received["context"].agent_type == "subagent"
        assert received["context"].agent_name == "writer"
