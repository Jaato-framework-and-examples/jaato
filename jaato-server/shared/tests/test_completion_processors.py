"""Tests for shared/completion_processors.py — unified processor hook.

Server 0.6.125+: replaces the prior split between
``completion_artifacts`` (renderers) and ``completion_validators``
(checkers) with one ``completion_processors`` field.  Each kb
processor module exposes ``render`` and/or ``validate`` — framework
probes which symbols are present and dispatches accordingly.

Covers:

- ``build_tool_call_ledger`` pairs function_calls with responses by
  call_id; surfaces unpaired calls as ``success=False``.
- ``load_processors`` resolves paths, probes for render/validate
  symbols, surfaces load errors (missing file, no callable symbols,
  malformed entries).
- ``invoke_processors`` runs each processor: validate first, then
  render; aggregates errors per ``on_error`` policy; writes files
  for render entries with declared ``output``.
- End-to-end via ``LifecycleTools._execute_signal_completion``:
  fatal failures block the completion + return validation_failed
  shape; warnings log but proceed.
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

from shared.completion_processors import (
    build_tool_call_ledger,
    collect_failure_messages,
    invoke_processors,
    load_processors,
    LoadedProcessor,
    ProcessorInvocationResult,
)
from shared.dynamic_instructions import RenderContext
from shared.lifecycle_tools import LifecycleTools
from shared.plugins.subagent.config import CompletionProcessor


# ---------------------------------------------------------------------------
# build_tool_call_ledger (unchanged from old completion_validators tests —
# pin the contract this module owns now)
# ---------------------------------------------------------------------------

class TestBuildToolCallLedger:

    def test_empty_history_returns_empty(self):
        assert build_tool_call_ledger([]) == []

    def test_pairs_call_with_response_by_call_id(self):
        history = [
            Message(role=Role.MODEL, parts=[Part.from_function_call(
                FunctionCall(id="c1", name="renderTemplateToFile",
                             args={"output_path": "src/A.java"}),
            )]),
            Message(role=Role.TOOL, parts=[Part.from_function_response(
                ToolResult(call_id="c1", name="renderTemplateToFile",
                           result={"path": "src/A.java"}),
            )]),
        ]
        ledger = build_tool_call_ledger(history)
        assert len(ledger) == 1
        assert ledger[0]["name"] == "renderTemplateToFile"
        assert ledger[0]["success"] is True
        assert ledger[0]["result"] == {"path": "src/A.java"}

    def test_failed_call_marked_success_false(self):
        history = [
            Message(role=Role.MODEL, parts=[Part.from_function_call(
                FunctionCall(id="c2", name="render", args={}),
            )]),
            Message(role=Role.TOOL, parts=[Part.from_function_response(
                ToolResult(call_id="c2", name="render",
                           result={"error": "boom", "message": "details"}),
            )]),
        ]
        ledger = build_tool_call_ledger(history)
        assert ledger[0]["success"] is False
        assert ledger[0]["result"]["error"] == "boom"

    def test_unpaired_call_marked_no_response(self):
        history = [
            Message(role=Role.MODEL, parts=[Part.from_function_call(
                FunctionCall(id="dangling", name="t", args={}),
            )]),
        ]
        ledger = build_tool_call_ledger(history)
        assert ledger[0]["success"] is False
        assert ledger[0]["result"] == {"error": "no_response"}


# ---------------------------------------------------------------------------
# load_processors
# ---------------------------------------------------------------------------

class TestLoadProcessors:

    def _write(self, path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    def test_loads_render_only_processor(self, tmp_path):
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "p.py",
            "def render(payload, context): return 'x'\n",
        )
        proc = CompletionProcessor(
            script="scripts/processors/p.py",
            output="out.md",
        )
        loaded = load_processors([proc], workspace_path=str(ws), config_root=None)
        assert len(loaded) == 1
        assert loaded[0].render_fn is not None
        assert loaded[0].validate_fn is None
        assert loaded[0].load_error is None

    def test_loads_validate_only_processor(self, tmp_path):
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "p.py",
            "def validate(payload, context): return []\n",
        )
        proc = CompletionProcessor(script="scripts/processors/p.py")
        loaded = load_processors([proc], workspace_path=str(ws), config_root=None)
        assert loaded[0].render_fn is None
        assert loaded[0].validate_fn is not None
        assert loaded[0].load_error is None

    def test_loads_both_render_and_validate(self, tmp_path):
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "p.py",
            "def render(payload, context): return 'x'\n"
            "def validate(payload, context): return []\n",
        )
        proc = CompletionProcessor(
            script="scripts/processors/p.py",
            output="out.md",
        )
        loaded = load_processors([proc], workspace_path=str(ws), config_root=None)
        assert loaded[0].render_fn is not None
        assert loaded[0].validate_fn is not None

    def test_missing_file_surfaces_load_error(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        proc = CompletionProcessor(script="scripts/processors/nope.py")
        loaded = load_processors([proc], workspace_path=str(ws), config_root=None)
        assert loaded[0].render_fn is None
        assert loaded[0].validate_fn is None
        assert "could not be located" in loaded[0].load_error

    def test_no_symbols_surfaces_load_error(self, tmp_path):
        """Module that exposes neither render nor validate is a kb
        authoring error — surfaced at signal_completion time."""
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "p.py",
            "def not_relevant(): return None\n",
        )
        proc = CompletionProcessor(script="scripts/processors/p.py")
        loaded = load_processors([proc], workspace_path=str(ws), config_root=None)
        assert loaded[0].render_fn is None
        assert loaded[0].validate_fn is None
        assert "neither 'render' nor 'validate'" in loaded[0].load_error


# ---------------------------------------------------------------------------
# invoke_processors
# ---------------------------------------------------------------------------

def _ctx(tmp_path: Path, tool_calls=None) -> RenderContext:
    """Minimal RenderContext fixture for invocation tests."""
    return RenderContext(
        session=None, runtime=None, registry=None,
        workspace_path=str(tmp_path),
        config_root=None,
        agent_params={},
        env={},
        tool_calls=list(tool_calls or []),
    )


class TestInvokeProcessors:

    def test_empty_loaded_returns_clean_result(self, tmp_path):
        result = invoke_processors([], payload={}, context=_ctx(tmp_path))
        assert result.failed == []
        assert result.warned == []
        assert result.written == []

    def test_load_error_routes_through_on_error(self, tmp_path):
        proc = CompletionProcessor(
            script="missing.py", on_error="fail_completion",
        )
        lp = LoadedProcessor(processor=proc, load_error="not found")
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.failed) == 1
        assert result.failed[0][1] == "not found"

        # Same load_error with on_error=warn → bucketed as warning
        proc2 = CompletionProcessor(script="missing.py", on_error="warn")
        lp2 = LoadedProcessor(processor=proc2, load_error="not found")
        result2 = invoke_processors([lp2], payload={}, context=_ctx(tmp_path))
        assert result2.failed == []
        assert len(result2.warned) == 1

    def test_validate_passes_no_errors(self, tmp_path):
        proc = CompletionProcessor(script="p.py")
        lp = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: [],
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert result.failed == []
        assert result.warned == []

    def test_validate_returns_errors_prefixed_with_script(self, tmp_path):
        proc = CompletionProcessor(script="p.py")
        lp = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: ["claim X failed", "claim Y missing"],
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.failed) == 2
        assert result.failed[0][1] == "[p.py] claim X failed"
        assert result.failed[1][1] == "[p.py] claim Y missing"

    def test_validate_raise_caught_and_surfaced(self, tmp_path):
        def boom(payload, ctx): raise RuntimeError("kaboom")
        proc = CompletionProcessor(script="p.py")
        lp = LoadedProcessor(processor=proc, validate_fn=boom)
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.failed) == 1
        msg = result.failed[0][1]
        assert "p.py" in msg
        assert "RuntimeError" in msg
        assert "kaboom" in msg

    def test_validate_non_list_return_surfaced(self, tmp_path):
        proc = CompletionProcessor(script="p.py")
        lp = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: "not a list",
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.failed) == 1
        # Server 0.6.160+ accepts list[str] OR ProcessorResult TypedDict;
        # the error message names both.
        assert "list[str] or ProcessorResult" in result.failed[0][1]

    def test_validate_processorresult_typeddict_shape_errors_route_to_failed(
        self, tmp_path,
    ):
        """Server 0.6.160+: validate can return the
        ``ProcessorResult`` TypedDict shape from
        ``jaato_sdk.cascade_authoring``.  ``errors[]`` entries land
        in ``result.failed`` (subject to ``on_error`` policy, same
        as the legacy ``list[str]`` path)."""
        proc = CompletionProcessor(script="p.py")  # default on_error=fail_completion
        lp = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: {
                "errors": ["fatal-1", "fatal-2"],
                "warnings": [],
            },
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.failed) == 2
        assert result.warned == []
        assert "[p.py] fatal-1" == result.failed[0][1]
        assert "[p.py] fatal-2" == result.failed[1][1]

    def test_validate_processorresult_warnings_never_escalate(
        self, tmp_path,
    ):
        """Server 0.6.160+: ``warnings[]`` from the
        ``ProcessorResult`` TypedDict always land in ``result.warned``
        regardless of the processor's ``on_error`` policy — they're
        advisory by definition.

        Compare with the ``on_error="fail_completion"`` (default)
        proc here: a legacy ``list[str]`` return would route entries
        to ``result.failed``, but warnings explicitly bypass the
        bucket and stay advisory."""
        proc = CompletionProcessor(script="p.py")  # default on_error=fail_completion
        lp = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: {
                "errors": [],
                "warnings": ["advisory-1", "advisory-2"],
            },
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert result.failed == [], (
            "warnings[] must NEVER produce failed entries — they're "
            "advisory regardless of on_error policy"
        )
        assert len(result.warned) == 2
        assert "[p.py] advisory-1" == result.warned[0][1]
        assert "[p.py] advisory-2" == result.warned[1][1]

    def test_validate_processorresult_mixed_errors_and_warnings(
        self, tmp_path,
    ):
        """The full expressiveness gain — one processor returns
        both fatal errors AND advisory warnings in a single call."""
        proc = CompletionProcessor(script="p.py")
        lp = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: {
                "errors": ["agent must retry: X"],
                "warnings": ["audit-only: Y"],
            },
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.failed) == 1
        assert len(result.warned) == 1
        assert "X" in result.failed[0][1]
        assert "Y" in result.warned[0][1]

    def test_validate_processorresult_partial_keys_accepted(
        self, tmp_path,
    ):
        """``ProcessorResult`` is ``TypedDict(total=False)`` — authors
        can return only ``errors`` or only ``warnings``.  Framework
        treats missing keys as empty lists."""
        proc = CompletionProcessor(script="p.py")
        # errors-only
        lp = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: {"errors": ["e1"]},
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.failed) == 1
        assert result.warned == []
        # warnings-only
        lp2 = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: {"warnings": ["w1"]},
        )
        result2 = invoke_processors([lp2], payload={}, context=_ctx(tmp_path))
        assert result2.failed == []
        assert len(result2.warned) == 1

    def test_validate_processorresult_warnings_with_on_error_warn_proc(
        self, tmp_path,
    ):
        """When the processor declares ``on_error="warn"`` AND
        returns the new TypedDict shape: errors[] downgrade via the
        proc-level policy to ``result.warned`` (existing _bucket
        behavior), and warnings[] also land in ``result.warned``.
        Net: no entry should ever reach ``result.failed``."""
        proc = CompletionProcessor(script="p.py", on_error="warn")
        lp = LoadedProcessor(
            processor=proc,
            validate_fn=lambda payload, ctx: {
                "errors": ["downgraded-1"],
                "warnings": ["advisory-1"],
            },
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert result.failed == []
        # Both the downgraded error AND the warning land in result.warned
        assert len(result.warned) == 2
        # Order: errors processed first, then warnings (matches the
        # implementation in invoke_processors).
        warned_msgs = [msg for (_proc, msg) in result.warned]
        assert any("downgraded-1" in m for m in warned_msgs)
        assert any("advisory-1" in m for m in warned_msgs)

    def test_render_with_output_writes_file(self, tmp_path):
        proc = CompletionProcessor(
            script="p.py", output="out.md",
        )
        lp = LoadedProcessor(
            processor=proc,
            render_fn=lambda payload, ctx: "hello world",
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.written) == 1
        written_path = Path(result.written[0])
        assert written_path.read_text() == "hello world"
        assert result.failed == []

    def test_render_bytes_output_writes_binary(self, tmp_path):
        proc = CompletionProcessor(script="p.py", output="out.bin")
        lp = LoadedProcessor(
            processor=proc,
            render_fn=lambda payload, ctx: b"\x00\x01\x02",
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        written_path = Path(result.written[0])
        assert written_path.read_bytes() == b"\x00\x01\x02"

    def test_render_no_output_runs_for_side_effect(self, tmp_path):
        """``output: None`` means validator-as-renderer — runs the
        render fn for side-effect, no file written."""
        proc = CompletionProcessor(script="p.py")  # no output
        lp = LoadedProcessor(
            processor=proc,
            render_fn=lambda payload, ctx: "log-only",
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert result.written == [""]
        assert result.failed == []

    def test_render_raise_caught_and_surfaced(self, tmp_path):
        def boom(payload, ctx): raise ValueError("bad render")
        proc = CompletionProcessor(script="p.py", output="out.md")
        lp = LoadedProcessor(processor=proc, render_fn=boom)
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        assert len(result.failed) == 1
        assert "render raised" in result.failed[0][1]
        assert "bad render" in result.failed[0][1]

    def test_render_output_template_substitution(self, tmp_path):
        """``output: out/{case_id}.md`` substitutes from payload."""
        proc = CompletionProcessor(script="p.py", output="out/{case_id}.md")
        lp = LoadedProcessor(
            processor=proc,
            render_fn=lambda payload, ctx: "content",
        )
        result = invoke_processors(
            [lp], payload={"case_id": "case42"}, context=_ctx(tmp_path),
        )
        written_path = Path(result.written[0])
        assert written_path.name == "case42.md"

    def test_render_and_validate_both_run(self, tmp_path):
        """A module with both symbols: both fire; validate failures
        block but render still runs (processors are independent)."""
        proc = CompletionProcessor(script="p.py", output="out.md")
        lp = LoadedProcessor(
            processor=proc,
            render_fn=lambda payload, ctx: "produced",
            validate_fn=lambda payload, ctx: ["validation failed"],
        )
        result = invoke_processors([lp], payload={}, context=_ctx(tmp_path))
        # Validate flagged a failure
        assert any("validation failed" in m for _, m in result.failed)
        # Render still ran and wrote the file
        assert any(p.endswith("out.md") for p in result.written)

    def test_multiple_processors_aggregated_not_short_circuited(self, tmp_path):
        proc_a = CompletionProcessor(script="a.py")
        proc_b = CompletionProcessor(script="b.py")
        lp_a = LoadedProcessor(
            processor=proc_a,
            validate_fn=lambda payload, ctx: ["a-err"],
        )
        lp_b = LoadedProcessor(
            processor=proc_b,
            validate_fn=lambda payload, ctx: ["b-err1", "b-err2"],
        )
        result = invoke_processors([lp_a, lp_b], payload={}, context=_ctx(tmp_path))
        # All three errors land in result.failed — no whack-a-mole
        assert [m for _, m in result.failed] == [
            "[a.py] a-err",
            "[b.py] b-err1",
            "[b.py] b-err2",
        ]


# ---------------------------------------------------------------------------
# collect_failure_messages
# ---------------------------------------------------------------------------

class TestCollectFailureMessages:

    def test_flattens_failed_bucket(self):
        proc1 = CompletionProcessor(script="a.py")
        proc2 = CompletionProcessor(script="b.py")
        result = ProcessorInvocationResult(
            failed=[(proc1, "[a.py] err1"), (proc2, "[b.py] err2")],
        )
        assert collect_failure_messages(result) == [
            "[a.py] err1", "[b.py] err2",
        ]


# ---------------------------------------------------------------------------
# End-to-end via LifecycleTools
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
        processors: List[CompletionProcessor],
        workspace_path: str,
        config_root: Optional[str] = None,
        history: Optional[List[Message]] = None,
    ) -> None:
        self._completion_payload_schema = SAMPLE_SCHEMA
        self._completion_processors = processors
        self._agent_id = "main"
        self.workspace_path = workspace_path
        self._workspace_path = workspace_path
        self._ui_hooks = _StubHooks()
        self._agent_params: Dict[str, Any] = {}
        self._signal_completion_called = False
        self._history = history or []
        self.runtime = _StubRuntime(config_root=config_root)
        self._runtime = self.runtime
        self._config_root = config_root

    def get_context_usage(self) -> Dict[str, Any]:
        return {}

    def get_history(self) -> List[Message]:
        return self._history


class TestSignalCompletionWithProcessors:

    def _write(self, path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    def test_validate_pass_emits_event(self, tmp_path):
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "ok.py",
            "def validate(payload, context):\n    return []\n",
        )
        session = _StubSession(
            processors=[CompletionProcessor(script="scripts/processors/ok.py")],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion({"summary": "Done", "files": []})
        assert result["status"] == "completed"
        assert len(session._ui_hooks.calls) == 1

    def test_validate_errors_block_event(self, tmp_path):
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "files.py",
            "def validate(payload, context):\n"
            "    return ['Customer.java: claimed but missing']\n",
        )
        session = _StubSession(
            processors=[CompletionProcessor(
                script="scripts/processors/files.py",
            )],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "Done", "files": [{"output_path": "Customer.java"}]},
        )
        assert result["error"] == "validation_failed"
        assert "processor_errors" in result
        assert any("Customer.java" in e for e in result["processor_errors"])
        assert len(session._ui_hooks.calls) == 0

    def test_render_writes_file_and_emits_event(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "renderer.py",
            "def render(payload, context):\n"
            "    return f'summary: {payload[\"summary\"]}'\n",
        )
        session = _StubSession(
            processors=[CompletionProcessor(
                script="scripts/processors/renderer.py",
                output="report.md",
            )],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "All good", "files": []},
        )
        assert result["status"] == "completed"
        assert "artifacts_written" in result
        written = ws / "report.md"
        assert written.read_text() == "summary: All good"
        assert len(session._ui_hooks.calls) == 1

    def test_render_and_validate_same_module(self, tmp_path):
        """Processor module that exposes BOTH render and validate —
        render produces output AND validate checks claims; both run."""
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "audit.py",
            "def render(payload, context):\n"
            "    return 'audit-record'\n"
            "def validate(payload, context):\n"
            "    if not payload.get('summary'):\n"
            "        return ['summary required']\n"
            "    return []\n",
        )
        session = _StubSession(
            processors=[CompletionProcessor(
                script="scripts/processors/audit.py",
                output="audit/{summary}.log",
            )],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "OK", "files": []},
        )
        assert result["status"] == "completed"
        assert (ws / "audit" / "OK.log").read_text() == "audit-record"

    def test_validator_can_read_tool_calls_from_context(self, tmp_path):
        """Pin: ``context.tool_calls`` carries the pre-computed ledger.

        Mirrors the kb-enablement-2.0 codegen_files_exist.py pattern:
        cross-check payload claims against actual tool outcomes.
        """
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "files_exist.py",
            "def validate(payload, context):\n"
            "    failed = {tc['args'].get('output_path'): tc['result'].get('error')\n"
            "              for tc in context.tool_calls\n"
            "              if tc['name'] == 'renderTemplateToFile' and not tc['success']}\n"
            "    return [f\"{op}: failed with {err}\"\n"
            "            for f in payload.get('files', [])\n"
            "            for op in [f.get('output_path')]\n"
            "            if op in failed\n"
            "            for err in [failed[op]]]\n",
        )
        history = [
            Message(role=Role.MODEL, parts=[Part.from_function_call(
                FunctionCall(id="c1", name="renderTemplateToFile",
                             args={"output_path": "src/Customer.java"}),
            )]),
            Message(role=Role.TOOL, parts=[Part.from_function_response(
                ToolResult(call_id="c1", name="renderTemplateToFile",
                           result={"error": "shape validation failed"}),
            )]),
        ]
        session = _StubSession(
            processors=[CompletionProcessor(
                script="scripts/processors/files_exist.py",
            )],
            workspace_path=str(ws),
            history=history,
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "x", "files": [{"output_path": "src/Customer.java"}]},
        )
        assert result["error"] == "validation_failed"
        errs = result["processor_errors"]
        assert any("src/Customer.java" in e for e in errs)
        assert any("shape validation failed" in e for e in errs)

    def test_processor_with_no_symbols_surfaces_load_error(self, tmp_path):
        """A processor module exposing neither render nor validate is
        a kb authoring error — surfaced as a fail_completion."""
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "broken.py",
            "def not_relevant(): return None\n",
        )
        session = _StubSession(
            processors=[CompletionProcessor(
                script="scripts/processors/broken.py",
            )],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "x", "files": []},
        )
        assert result["error"] == "validation_failed"
        errs = result["processor_errors"]
        assert any("neither 'render' nor 'validate'" in e for e in errs)

    def test_no_processors_skips_pipeline(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        session = _StubSession(processors=[], workspace_path=str(ws))
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "done", "files": []},
        )
        assert result["status"] == "completed"
        assert len(session._ui_hooks.calls) == 1

    def test_warn_policy_does_not_block_completion(self, tmp_path):
        ws = tmp_path / "ws"
        self._write(
            ws / ".jaato" / "scripts" / "processors" / "soft.py",
            "def validate(payload, context):\n"
            "    return ['advisory issue']\n",
        )
        session = _StubSession(
            processors=[CompletionProcessor(
                script="scripts/processors/soft.py",
                on_error="warn",
            )],
            workspace_path=str(ws),
        )
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"summary": "done", "files": []},
        )
        # Validator complained but on_error=warn → completion succeeds
        assert result["status"] == "completed"
        assert len(session._ui_hooks.calls) == 1
