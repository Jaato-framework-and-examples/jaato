"""Tests for runtime-injectable permission evaluators.

Path resolution is now owned by ``shared.script_loader`` and covered by
``shared/tests/test_script_loader.py``. Tests here focus on the evaluator-
specific contract: loading the ``evaluate`` symbol, decision coercion,
and the run_evaluator dispatch pipeline.
"""

import pytest

from shared.plugins.permission.evaluator import (
    EvalContext,
    EvalResult,
    PolicyDecision,
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
        assert ctx.execution_log == []
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

    def test_execution_log_field(self):
        log = [{"tool_name": "cli", "arguments": {}, "decision": "allow",
                "reason": "x"}]
        ctx = EvalContext(tool_name="cli", args={}, execution_log=log)
        assert ctx.execution_log == log

    def test_evaluator_can_decide_on_execution_log(self, tmp_path):
        # An evaluator that denies a tool once it has already run >= 2 times
        # this session — reasoning purely from EvalContext.execution_log.
        script = tmp_path / ".jaato" / "rate_limit.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "from shared.plugins.permission.evaluator import PolicyDecision\n"
            "def evaluate(tool_name, args, context):\n"
            "    prior = sum(1 for e in context.execution_log\n"
            "                if e.get('tool_name') == tool_name)\n"
            "    return PolicyDecision.DENY if prior >= 2 else PolicyDecision.FALLBACK\n"
        )
        evaluators = load_evaluators({"default": "rate_limit.py"},
                                     workspace_path=str(tmp_path))
        entry = {"tool_name": "cli", "arguments": {}, "decision": "allow", "reason": ""}

        def _decide(log):
            ctx = EvalContext(tool_name="cli", args={}, execution_log=log)
            return run_evaluator(evaluators, "cli", {}, ctx).decision

        assert _decide([]) == PolicyDecision.FALLBACK
        assert _decide([entry]) == PolicyDecision.FALLBACK
        assert _decide([entry, entry]) == PolicyDecision.DENY


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
        assert result.decision == PolicyDecision.ALLOW

    def test_deny_decision(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.DENY

    def test_fallback_decision(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.FALLBACK,
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.FALLBACK

    def test_tool_specific_takes_precedence(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.ALLOW,
            "cli_based_tool": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli_based_tool", {}, self._ctx("cli_based_tool"))
        assert result.decision == PolicyDecision.DENY

    def test_falls_back_to_default(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.ALLOW,
            "other_tool": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli_based_tool", {}, self._ctx("cli_based_tool"))
        assert result.decision == PolicyDecision.ALLOW

    def test_no_matching_evaluator_returns_fallback(self):
        evaluators = {
            "other_tool": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli_based_tool", {}, self._ctx("cli_based_tool"))
        assert result.decision == PolicyDecision.FALLBACK

    def test_empty_evaluators_returns_fallback(self):
        result = run_evaluator({}, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.FALLBACK

    def test_exception_in_evaluator_returns_fallback(self):
        def bad_eval(t, a, c):
            raise RuntimeError("boom")

        evaluators = {"default": bad_eval}
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.FALLBACK

    def test_string_return_value_accepted(self):
        evaluators = {
            "default": lambda t, a, c: "deny",
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.DENY

    def test_invalid_return_value_returns_fallback(self):
        evaluators = {
            "default": lambda t, a, c: 42,
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.FALLBACK

    def test_deny_with_comment_via_eval_result(self):
        evaluators = {
            "default": lambda t, a, c: EvalResult(
                PolicyDecision.DENY_WITH_COMMENT,
                comment="This command is blocked during maintenance window",
            ),
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.DENY_WITH_COMMENT
        assert result.comment == "This command is blocked during maintenance window"

    def test_deny_with_comment_via_tuple(self):
        evaluators = {
            "default": lambda t, a, c: (PolicyDecision.DENY_WITH_COMMENT, "Not allowed here"),
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.DENY_WITH_COMMENT
        assert result.comment == "Not allowed here"

    def test_deny_with_comment_via_string_tuple(self):
        evaluators = {
            "default": lambda t, a, c: ("deny_with_comment", "Use a safer command"),
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.decision == PolicyDecision.DENY_WITH_COMMENT
        assert result.comment == "Use a safer command"

    def test_bare_decision_has_no_comment(self):
        evaluators = {
            "default": lambda t, a, c: PolicyDecision.DENY,
        }
        result = run_evaluator(evaluators, "cli", {}, self._ctx())
        assert result.comment is None

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
