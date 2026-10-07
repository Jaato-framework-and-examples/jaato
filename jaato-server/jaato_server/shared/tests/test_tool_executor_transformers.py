"""Tests for ToolExecutor args + result transformer plug-in surface.

Seat 2 of the four-seat pseudonymization design — see
``project_backlog_pseudonymization_plugin_surface.md`` and
``jaato-premium/docs/design/pseudonymization-four-seat.md``.

The transformers are deliberately generic — any per-call rewrite of
args or result is allowed.  These tests cover the behavioural contract
(registration, chaining, trusted-tools allowlist, fast-path when no
transformers, no interference with existing telemetry/cgroup wiring),
not specific pseudonymization semantics.
"""

from typing import Any, Dict
import pytest
from unittest.mock import MagicMock

from ..ai_tool_runner import ToolExecutor


# Helpers ─────────────────────────────────────────────────────────────


def _executor() -> ToolExecutor:
    """Bare ToolExecutor with one trivially-registered tool."""
    ex = ToolExecutor()
    ex.register("echo", lambda args: {"got": args})
    return ex


# Args transformer ─────────────────────────────────────────────────────


class TestArgsTransformer:

    def test_unset_chain_passes_args_unchanged(self):
        ex = _executor()
        success, result = ex.execute("echo", {"k": "v"})
        assert success is True
        assert result["got"] == {"k": "v"}

    def test_single_args_transformer_runs(self):
        ex = _executor()

        def tag(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            return {**args, "_tagged_by": name}

        ex.register_args_transformer(tag)
        success, result = ex.execute("echo", {"k": "v"})
        assert success is True
        assert result["got"] == {"k": "v", "_tagged_by": "echo"}

    def test_multiple_args_transformers_chain_in_registration_order(self):
        ex = _executor()
        order: list = []

        def first(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            order.append("first")
            return {**args, "step1": True}

        def second(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            order.append("second")
            # second sees first's output, including its added key
            assert args.get("step1") is True
            return {**args, "step2": True}

        ex.register_args_transformer(first)
        ex.register_args_transformer(second)
        success, result = ex.execute("echo", {"k": "v"})
        assert success is True
        assert result["got"]["step1"] is True
        assert result["got"]["step2"] is True
        assert order == ["first", "second"]

    def test_trusted_tools_allowlist_skips_transformer(self):
        ex = _executor()
        ex.register("send_email", lambda args: {"sent_to": args.get("to")})

        def redact(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            return {**args, "to": "<REDACTED>"}

        # send_email is trusted; echo is NOT — transformer applies
        # only to echo
        ex.register_args_transformer(redact, trusted_tools={"send_email"})

        # send_email skips the transformer → sees raw args
        _, result_email = ex.execute(
            "send_email", {"to": "alice@example.com"}
        )
        assert result_email["sent_to"] == "alice@example.com"

        # echo gets the redacted args
        _, result_echo = ex.execute("echo", {"to": "alice@example.com"})
        assert result_echo["got"]["to"] == "<REDACTED>"

    def test_trusted_tools_per_transformer(self):
        """Different transformers can have different trusted lists —
        they compose independently."""
        ex = _executor()
        ex.register("a", lambda args: dict(args))
        ex.register("b", lambda args: dict(args))

        def t1(name, args):
            return {**args, "t1": True}

        def t2(name, args):
            return {**args, "t2": True}

        ex.register_args_transformer(t1, trusted_tools={"a"})
        ex.register_args_transformer(t2, trusted_tools={"b"})

        _, ra = ex.execute("a", {})
        _, rb = ex.execute("b", {})
        # a is trusted by t1 (skipped) but seen by t2
        assert "t1" not in ra and "t2" in ra
        # b is trusted by t2 (skipped) but seen by t1
        assert "t1" in rb and "t2" not in rb


# Result transformer ───────────────────────────────────────────────────


class TestResultTransformer:

    def test_unset_chain_passes_result_unchanged(self):
        ex = _executor()
        _, result = ex.execute("echo", {"k": "v"})
        assert result == {"got": {"k": "v"}}

    def test_single_result_transformer_runs(self):
        ex = _executor()

        def annotate(name: str, result: Any) -> Any:
            if isinstance(result, dict):
                return {**result, "_annotated": True}
            return result

        ex.register_result_transformer(annotate)
        _, result = ex.execute("echo", {"k": "v"})
        assert result["_annotated"] is True
        assert result["got"] == {"k": "v"}

    def test_multiple_result_transformers_chain_in_registration_order(self):
        ex = _executor()
        order: list = []

        def first(name, result):
            order.append("first")
            result["step1"] = True
            return result

        def second(name, result):
            order.append("second")
            # second sees first's mutation
            assert result.get("step1") is True
            result["step2"] = True
            return result

        ex.register_result_transformer(first)
        ex.register_result_transformer(second)
        _, result = ex.execute("echo", {})
        assert result["step1"] is True
        assert result["step2"] is True
        assert order == ["first", "second"]

    def test_result_transformer_can_replace_result_object(self):
        """Transformer is free to return a new object (not just mutate)."""
        ex = _executor()

        def replace(name, result):
            return {"replaced": True, "original_keys": list(result.keys())}

        ex.register_result_transformer(replace)
        _, result = ex.execute("echo", {"k": "v"})
        assert result == {"replaced": True, "original_keys": ["got"]}


# Composition with existing telemetry pipeline ─────────────────────────


class TestComposesWithExistingPipeline:
    """Confirm the transformer chains don't interfere with
    cgroup-event-delta injection or thread-local plumbing."""

    def test_telemetry_dict_added_after_result_transformer(self):
        """Cgroup deltas use ``_telemetry`` dict on result.  When a
        result transformer returns a dict, the cgroup pipeline still
        gets a chance to inject deltas."""
        ex = _executor()

        # Stub cgroup event reader so the delta path runs.
        before, after = {"oom_kill": 0}, {"oom_kill": 1}
        snapshots = iter([before, after])
        ex._cgroup_event_reader = lambda: next(snapshots)

        def add_marker(name, result):
            return {**result, "_marker": "from_transformer"}

        ex.register_result_transformer(add_marker)
        _, result = ex.execute("echo", {})
        # Transformer ran
        assert result["_marker"] == "from_transformer"
        # Cgroup delta also injected on the post-transform result
        assert result.get("_telemetry", {}).get("jaato.cgroup.oom_kill_delta") == 1

    def test_args_transformer_runs_before_executor_sees_args(self):
        """The tool's executor receives the transformed args, not the
        original — verifies the chain runs *before* _execute_impl."""
        ex = ToolExecutor()
        seen: list = []

        def capture(args):
            seen.append(args)
            return {}

        ex.register("capture", capture)

        def mutate(name, args):
            return {**args, "added": True}

        ex.register_args_transformer(mutate)
        ex.execute("capture", {"orig": True})
        assert seen == [{"orig": True, "added": True}]
