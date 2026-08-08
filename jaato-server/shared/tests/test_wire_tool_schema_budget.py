"""Tests for initial wire tool-schema budget accounting (server 0.6.x+).

Regression cover for the budget-GC blind-spot: the tool-definitions
array (``tools[]``) ships on every request but its tokens were only
counted on the DISCOVERY path (``_track_activated_tools_in_budget``), so
PRELOADED / CORE / lifecycle tools — exposed at configure, never
"discovered" — were never counted.  That made ``total_tokens()`` (the
GC trigger denominator) under-report the true wire, so the threshold was
structurally unreachable and sessions walked into context-window 400s
with GC idle (kb cascade session 20260610_082013: 58% measured vs 97%
real wire).

``_register_initial_wire_tool_schema_budget`` closes the gap by counting
the initial wire tool schemas into a LOCKED PLUGIN child — counted by
``total_tokens()`` (trigger sees true utilisation) but excluded from
``gc_eligible_tokens()`` (GC still trims only conversation).
"""

from types import SimpleNamespace

from shared.instruction_budget import InstructionSource
from shared.tests.test_two_phase_budget import _make_session


def _schema(name, description, parameters=None):
    return SimpleNamespace(
        name=name,
        description=description,
        parameters=parameters or {"type": "object", "properties": {}},
    )


def _prep(session):
    """Populate the budget + silence the trace/emit side-effects so the
    method under test can run in isolation."""
    session._populate_instruction_budget(session_instructions="persona text")
    session._trace = lambda *a, **k: None
    session._emit_instruction_budget_update = lambda: None


class TestInitialWireToolSchemaBudget:

    def test_wire_tools_counted_into_total(self):
        s = _make_session()
        _prep(s)
        total_before = s._instruction_budget.total_tokens()

        s._tools = [
            _schema("signal_completion", "Finalize the agent run " * 40),
            _schema("prepare_completion", "Set one field " * 40),
            _schema("readFile", "Read a file from the workspace " * 40),
        ]
        s._register_initial_wire_tool_schema_budget()

        total_after = s._instruction_budget.total_tokens()
        # The denominator now reflects the wire tool array.
        assert total_after > total_before

    def test_wire_tools_are_locked_excluded_from_gc_eligible(self):
        s = _make_session()
        _prep(s)
        gc_eligible_before = s._instruction_budget.gc_eligible_tokens()

        s._tools = [
            _schema("signal_completion", "Finalize the agent run " * 40),
            _schema("readFile", "Read a file " * 40),
        ]
        s._register_initial_wire_tool_schema_budget()

        # Counted in total (measured) but NOT in gc_eligible (trimmable):
        # GC measures the true wire yet only ever trims conversation.
        assert s._instruction_budget.gc_eligible_tokens() == gc_eligible_before
        assert s._instruction_budget.total_tokens() > gc_eligible_before

    def test_wire_tool_schemas_child_present(self):
        s = _make_session()
        _prep(s)
        s._tools = [_schema("signal_completion", "x " * 100)]
        s._register_initial_wire_tool_schema_budget()

        plugin_entry = s._instruction_budget.get_entry(InstructionSource.PLUGIN)
        assert plugin_entry is not None
        assert "wire_tool_schemas" in plugin_entry.children
        assert plugin_entry.children["wire_tool_schemas"].tokens > 0

    def test_noop_when_no_tools(self):
        s = _make_session()
        _prep(s)
        s._tools = None
        before = s._instruction_budget.total_tokens()
        s._register_initial_wire_tool_schema_budget()
        assert s._instruction_budget.total_tokens() == before

    def test_noop_when_no_budget(self):
        s = _make_session()
        s._instruction_budget = None
        s._tools = [_schema("x", "y")]
        s._trace = lambda *a, **k: None
        s._emit_instruction_budget_update = lambda: None
        # Must not raise.
        s._register_initial_wire_tool_schema_budget()

    def test_crosses_gc_threshold_repro(self):
        """End-to-end of the bug: conversation-only is below 80%, but
        adding the wire tool array pushes utilisation over the trigger —
        exactly what was missing in session 20260610_082013."""
        s = _make_session()
        _prep(s)
        s._instruction_budget.context_limit = 1000
        # Simulate ~58% conversation-only (the silent-GC regime).
        s._instruction_budget.add_child(
            InstructionSource.CONVERSATION, "turns", 580,
            s._instruction_budget.get_entry(InstructionSource.CONVERSATION).gc_policy
            if s._instruction_budget.get_entry(InstructionSource.CONVERSATION) else None,
        )
        below = s._instruction_budget.utilization_percent()

        # A big static tool array (~30% of the window) that was invisible.
        s._tools = [_schema(f"tool_{i}", "schema body " * 60) for i in range(8)]
        s._register_initial_wire_tool_schema_budget()
        after = s._instruction_budget.utilization_percent()

        assert after > below
        # The static tools alone move utilisation materially upward —
        # the GC trigger can now see the true wall.
        assert after - below > 5.0


class TestGcDenominatorDiagnostic:
    """The GC_DENOM diagnostic must expose the InstructionBudget
    breakdown — including the wire_tool_schemas child — so the budget-GC
    denominator is inspectable from the daemon log (closes the PR-274
    verification gap)."""

    def test_logs_wire_tools_in_breakdown(self, caplog):
        import logging
        s = _make_session()
        _prep(s)
        s._tools = [_schema("signal_completion", "x " * 200)]
        s._register_initial_wire_tool_schema_budget()

        with caplog.at_level(logging.INFO, logger="shared.jaato_session"):
            s._log_gc_denominator("test", provider_total=12345)

        line = "\n".join(r.getMessage() for r in caplog.records)
        assert "GC_DENOM[test]" in line
        assert "wire_tools=" in line
        # The wire_tools figure must be non-zero (registration landed).
        assert "wire_tools=0" not in line
        assert "provider_total=12345" in line

    def test_handles_missing_budget(self, caplog):
        import logging
        s = _make_session()
        s._instruction_budget = None
        with caplog.at_level(logging.INFO, logger="shared.jaato_session"):
            s._log_gc_denominator("test", provider_total=7)
        assert "no_instruction_budget" in "\n".join(
            r.getMessage() for r in caplog.records
        )
