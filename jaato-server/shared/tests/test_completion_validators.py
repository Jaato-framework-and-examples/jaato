"""Tests for shared/completion_validators.py — semantic validator hook.

Covers:

- ``build_tool_call_ledger`` pairs function_calls with function_responses
  by call_id, surfaces unpaired calls as ``success=False`` with
  ``{"error": "no_response"}``.
- ``load_validators`` resolves paths via ``resolve_script_path``,
  surfaces missing files as ``load_error``.
- ``invoke_validators`` runs each, aggregates errors, prefixes each
  error with the validator path, handles validator raises and
  non-list returns.
- End-to-end via ``LifecycleTools._execute_signal_completion``: a
  configured validator that returns errors causes
  ``signal_completion`` to return the ``validation_failed`` shape
  and skip the on_agent_completed hook.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
    ToolResult,
)

from shared.completion_validators import (
    build_tool_call_ledger,
    invoke_validators,
    load_validators,
)
from shared.lifecycle_tools import LifecycleTools


# ---------------------------------------------------------------------------
# build_tool_call_ledger
# ---------------------------------------------------------------------------

class TestBuildToolCallLedger:

    def test_empty_history_returns_empty(self):
        assert build_tool_call_ledger([]) == []

    def test_pairs_call_with_response_by_call_id(self):
        history = [
            Message(
                role=Role.MODEL,
                parts=[Part.from_function_call(FunctionCall(
                    id="c1", name="renderTemplateToFile",
                    args={"output_path": "src/A.java"},
                ))],
            ),
            Message(
                role=Role.TOOL,
                parts=[Part.from_function_response(ToolResult(
                    call_id="c1", name="renderTemplateToFile",
                    result={"path": "src/A.java"},
                ))],
            ),
        ]
        ledger = build_tool_call_ledger(history)
        assert len(ledger) == 1
        entry = ledger[0]
        assert entry["name"] == "renderTemplateToFile"
        assert entry["args"] == {"output_path": "src/A.java"}
        assert entry["result"] == {"path": "src/A.java"}
        assert entry["success"] is True
        assert entry["call_id"] == "c1"

    def test_failed_call_marked_success_false(self):
        history = [
            Message(
                role=Role.MODEL,
                parts=[Part.from_function_call(FunctionCall(
                    id="c2", name="renderTemplateToFile",
                    args={"output_path": "src/B.java"},
                ))],
            ),
            Message(
                role=Role.TOOL,
                parts=[Part.from_function_response(ToolResult(
                    call_id="c2", name="renderTemplateToFile",
                    result={"error": "Variable shape validation failed",
                            "message": "missing fieldNamePascal"},
                ))],
            ),
        ]
        ledger = build_tool_call_ledger(history)
        assert len(ledger) == 1
        assert ledger[0]["success"] is False
        assert ledger[0]["result"]["error"] == "Variable shape validation failed"

    def test_unpaired_call_marked_no_response(self):
        history = [
            Message(
                role=Role.MODEL,
                parts=[Part.from_function_call(FunctionCall(
                    id="dangling", name="renderTemplateToFile", args={},
                ))],
            ),
        ]
        ledger = build_tool_call_ledger(history)
        assert len(ledger) == 1
        assert ledger[0]["success"] is False
        assert ledger[0]["result"] == {"error": "no_response"}

    def test_multiple_calls_in_chronological_order(self):
        history = [
            Message(
                role=Role.MODEL,
                parts=[
                    Part.from_function_call(FunctionCall(
                        id="c1", name="tool_a", args={},
                    )),
                    Part.from_function_call(FunctionCall(
                        id="c2", name="tool_b", args={},
                    )),
                ],
            ),
            Message(
                role=Role.TOOL,
                parts=[
                    Part.from_function_response(ToolResult(
                        call_id="c1", name="tool_a", result={"ok": True},
                    )),
                    Part.from_function_response(ToolResult(
                        call_id="c2", name="tool_b", result={"ok": True},
                    )),
                ],
            ),
            Message(
                role=Role.MODEL,
                parts=[Part.from_function_call(FunctionCall(
                    id="c3", name="tool_c", args={},
                ))],
            ),
            Message(
                role=Role.TOOL,
                parts=[Part.from_function_response(ToolResult(
                    call_id="c3", name="tool_c", result={"ok": True},
                ))],
            ),
        ]
        ledger = build_tool_call_ledger(history)
        assert [e["name"] for e in ledger] == ["tool_a", "tool_b", "tool_c"]
        assert [e["turn_index"] for e in ledger] == [0, 0, 2]

    def test_non_dict_result_wrapped(self):
        history = [
            Message(
                role=Role.MODEL,
                parts=[Part.from_function_call(FunctionCall(
                    id="c1", name="tool", args={},
                ))],
            ),
            Message(
                role=Role.TOOL,
                parts=[Part.from_function_response(ToolResult(
                    call_id="c1", name="tool", result="plain string",
                ))],
            ),
        ]
        ledger = build_tool_call_ledger(history)
        # Wrapped so validator sees consistent dict shape
        assert ledger[0]["result"] == {"result": "plain string"}
        # "error" not in wrapped dict → success True
        assert ledger[0]["success"] is True


# ---------------------------------------------------------------------------
# load_validators
# ---------------------------------------------------------------------------

class TestLoadValidators:

    def _write_validator(self, path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    def test_loads_validate_callable(self, tmp_path):
        ws = tmp_path / "ws"
        validator_path = ws / ".jaato" / "scripts" / "validators" / "v.py"
        self._write_validator(validator_path, "def validate(p, t, w, c): return []\n")
        loaded = load_validators(
            ["scripts/validators/v.py"],
            workspace_path=str(ws),
            config_root=None,
        )
        assert len(loaded) == 1
        path_str, fn, err = loaded[0]
        assert path_str == "scripts/validators/v.py"
        assert err is None
        assert callable(fn)

    def test_missing_file_surfaces_load_error(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        loaded = load_validators(
            ["scripts/validators/nope.py"],
            workspace_path=str(ws),
            config_root=None,
        )
        assert len(loaded) == 1
        _, fn, err = loaded[0]
        assert fn is None
        assert err is not None
        assert "could not be located" in err

    def test_missing_validate_symbol_surfaces_load_error(self, tmp_path):
        ws = tmp_path / "ws"
        validator_path = ws / ".jaato" / "scripts" / "validators" / "v.py"
        # No "validate" symbol
        self._write_validator(validator_path, "def not_validate(p): return []\n")
        loaded = load_validators(
            ["scripts/validators/v.py"],
            workspace_path=str(ws),
            config_root=None,
        )
        assert len(loaded) == 1
        _, fn, err = loaded[0]
        assert fn is None
        assert err is not None
        assert "validate" in err

    def test_config_root_overrides_workspace(self, tmp_path):
        ws = tmp_path / "ws"
        cr = tmp_path / "cr"
        validator_path = cr / "scripts" / "validators" / "v.py"
        self._write_validator(
            validator_path,
            "def validate(p, t, w, c): return ['fired from config_root']\n",
        )
        loaded = load_validators(
            ["scripts/validators/v.py"],
            workspace_path=str(ws),
            config_root=str(cr),
        )
        _, fn, err = loaded[0]
        assert err is None
        result = fn({}, [], Path(ws), {})
        assert result == ["fired from config_root"]


# ---------------------------------------------------------------------------
# invoke_validators
# ---------------------------------------------------------------------------

class TestInvokeValidators:

    def test_empty_loaded_returns_empty(self):
        assert invoke_validators([], {}, [], Path("."), {}) == []

    def test_validator_pass_emits_no_errors(self):
        def ok(p, t, w, c): return []
        loaded = [("ok.py", ok, None)]
        assert invoke_validators(loaded, {}, [], Path("."), {}) == []

    def test_validator_errors_prefixed_with_path(self):
        def bad(p, t, w, c): return ["file X missing"]
        loaded = [("bad.py", bad, None)]
        errors = invoke_validators(loaded, {}, [], Path("."), {})
        assert errors == ["[bad.py] file X missing"]

    def test_validator_raise_surfaced(self):
        def boom(p, t, w, c): raise RuntimeError("kaboom")
        loaded = [("boom.py", boom, None)]
        errors = invoke_validators(loaded, {}, [], Path("."), {})
        assert len(errors) == 1
        assert "boom.py" in errors[0]
        assert "RuntimeError" in errors[0]
        assert "kaboom" in errors[0]

    def test_validator_non_list_return_surfaced(self):
        def wrong(p, t, w, c): return "not a list"
        loaded = [("wrong.py", wrong, None)]
        errors = invoke_validators(loaded, {}, [], Path("."), {})
        assert len(errors) == 1
        assert "expected list[str]" in errors[0]

    def test_load_error_surfaced_at_invocation(self):
        loaded = [("missing.py", None, "missing.py could not be located")]
        errors = invoke_validators(loaded, {}, [], Path("."), {})
        assert errors == ["missing.py could not be located"]

    def test_multiple_validators_aggregated_not_short_circuited(self):
        def fails1(p, t, w, c): return ["err A"]
        def fails2(p, t, w, c): return ["err B", "err C"]
        loaded = [("a.py", fails1, None), ("b.py", fails2, None)]
        errors = invoke_validators(loaded, {}, [], Path("."), {})
        # Aggregated — agent sees full picture on retry
        assert errors == ["[a.py] err A", "[b.py] err B", "[b.py] err C"]


# ---------------------------------------------------------------------------
# End-to-end via LifecycleTools._execute_signal_completion
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "files": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["summary"],
}


class _StubHooks:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def on_agent_completed(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _StubRuntime:
    def __init__(self, config_root: Optional[str] = None) -> None:
        self._config_root = config_root


class _StubSession:
    def __init__(
        self,
        validators: List[str],
        workspace_path: str,
        config_root: Optional[str] = None,
        history: Optional[List[Message]] = None,
    ) -> None:
        self._completion_payload_schema = SAMPLE_SCHEMA
        self._completion_validators = validators
        self._agent_id = "main"
        self.workspace_path = workspace_path
        self._ui_hooks = _StubHooks()
        self._agent_params: Dict[str, Any] = {}
        self._signal_completion_called = False
        self._history = history or []
        self.runtime = _StubRuntime(config_root=config_root)

    def get_context_usage(self) -> Dict[str, Any]:
        return {}

    def get_history(self) -> List[Message]:
        return self._history


class TestSignalCompletionWithValidators:

    def _write(self, path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    def test_validator_passes_emits_event(self, tmp_path):
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "validators" / "ok.py",
            "def validate(payload, tool_calls, workspace_path, ctx):\n"
            "    return []\n",
        )
        session = _StubSession(
            validators=["scripts/validators/ok.py"],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "Done", "files": []}
        )
        assert result["status"] == "completed"
        assert len(session._ui_hooks.calls) == 1

    def test_validator_errors_block_event_and_surface_in_result(self, tmp_path):
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "validators" / "files_exist.py",
            "def validate(payload, tool_calls, workspace_path, ctx):\n"
            "    return ['Customer.java: claimed but not in workspace']\n",
        )
        session = _StubSession(
            validators=["scripts/validators/files_exist.py"],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "Done", "files": [{"output_path": "Customer.java"}]}
        )
        assert result["error"] == "validation_failed"
        assert "validator_errors" in result
        assert any(
            "Customer.java" in e for e in result["validator_errors"]
        )
        # Event NOT emitted — the agent must retry.
        assert len(session._ui_hooks.calls) == 0

    def test_validator_can_inspect_tool_call_ledger(self, tmp_path):
        """Validator receives the ledger built from session history.

        Mirrors the kb-enablement-2.0 codegen_files_exist.py shape: walk
        tool_calls for renderTemplateToFile failures and cross-check
        against payload.files[].
        """
        ws = tmp_path / "ws"
        # Validator that fails iff any tool_call has success=False
        # and its output_path is in payload.files[].
        self._write(
            ws / ".jaato" / "scripts" / "validators" / "render_succeeded.py",
            "def validate(payload, tool_calls, workspace_path, ctx):\n"
            "    errors = []\n"
            "    failed = {tc['args'].get('output_path'): tc['result'].get('error')\n"
            "              for tc in tool_calls\n"
            "              if tc['name'] == 'renderTemplateToFile' and not tc['success']}\n"
            "    for f in payload.get('files', []):\n"
            "        op = f.get('output_path')\n"
            "        if op in failed:\n"
            "            errors.append(f\"{op}: claimed rendered but renderTemplateToFile FAILED with: {failed[op]}\")\n"
            "    return errors\n",
        )
        history = [
            Message(
                role=Role.MODEL,
                parts=[Part.from_function_call(FunctionCall(
                    id="c1", name="renderTemplateToFile",
                    args={"output_path": "src/Customer.java"},
                ))],
            ),
            Message(
                role=Role.TOOL,
                parts=[Part.from_function_response(ToolResult(
                    call_id="c1", name="renderTemplateToFile",
                    result={"error": "Variable shape validation failed"},
                ))],
            ),
        ]
        session = _StubSession(
            validators=["scripts/validators/render_succeeded.py"],
            workspace_path=str(ws),
            history=history,
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {
                "summary": "Done",
                "files": [{"output_path": "src/Customer.java"}],
            }
        )
        assert result["error"] == "validation_failed"
        errs = result["validator_errors"]
        assert len(errs) == 1
        assert "src/Customer.java" in errs[0]
        assert "Variable shape validation failed" in errs[0]

    def test_missing_validator_path_surfaced_as_error(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        session = _StubSession(
            validators=["scripts/validators/typo.py"],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "Done", "files": []}
        )
        assert result["error"] == "validation_failed"
        assert any(
            "could not be located" in e
            for e in result["validator_errors"]
        )

    def test_no_validators_configured_skips_validator_step(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        session = _StubSession(validators=[], workspace_path=str(ws))
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "Done", "files": []}
        )
        assert result["status"] == "completed"
        assert len(session._ui_hooks.calls) == 1
