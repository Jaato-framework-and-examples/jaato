"""Tests for shared/lifecycle_tools.py — typed completion payloads."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from shared.lifecycle_tools import LifecycleTools


SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["billing", "tech"]},
        "severity": {"type": "integer", "minimum": 1, "maximum": 5},
        "summary": {"type": "string"},
    },
    "required": ["category", "severity"],
}


class StubHooks:
    """Captures on_agent_completed kwargs for assertion."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def on_agent_completed(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class StubSession:
    """Minimal session double exposing what LifecycleTools needs."""

    def __init__(
        self,
        schema: Optional[Any] = None,
        agent_id: str = "main",
        workspace_path: Optional[str] = None,
    ) -> None:
        self._completion_payload_schema = schema
        self._agent_id = agent_id
        self.workspace_path = workspace_path
        self._ui_hooks = StubHooks()

    def get_context_usage(self) -> Dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# get_tool_schemas — dynamic shape based on profile schema
# ---------------------------------------------------------------------------

class TestGetToolSchemas:

    def test_legacy_summary_when_no_schema(self):
        lt = LifecycleTools(StubSession(schema=None))
        schemas = lt.get_tool_schemas()
        assert len(schemas) == 1
        params = schemas[0].parameters
        assert "summary" in params["properties"]
        assert params["properties"]["summary"]["type"] == "string"
        assert params["required"] == ["summary"]
        assert "payload" not in params["properties"]

    def test_typed_payload_when_schema_present(self):
        lt = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
        schemas = lt.get_tool_schemas()
        params = schemas[0].parameters
        assert "payload" in params["properties"]
        assert params["properties"]["payload"] == SAMPLE_SCHEMA
        assert params["required"] == ["payload"]
        assert "summary" not in params["properties"]

    def test_tool_name_unchanged(self):
        lt_legacy = LifecycleTools(StubSession(schema=None))
        lt_typed = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
        assert lt_legacy.get_tool_schemas()[0].name == "signal_completion"
        assert lt_typed.get_tool_schemas()[0].name == "signal_completion"


# ---------------------------------------------------------------------------
# _execute_signal_completion — legacy path
# ---------------------------------------------------------------------------

class TestExecuteLegacy:

    def test_emits_with_summary(self):
        session = StubSession(schema=None)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion({"summary": "Done"})
        assert result["status"] == "completed"
        assert result["summary"] == "Done"
        assert "payload" not in result
        assert len(session._ui_hooks.calls) == 1
        call = session._ui_hooks.calls[0]
        assert call["payload"] is None

    def test_missing_summary_emits_empty(self):
        session = StubSession(schema=None)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion({})
        assert result["summary"] == ""
        assert len(session._ui_hooks.calls) == 1


# ---------------------------------------------------------------------------
# _execute_signal_completion — typed path
# ---------------------------------------------------------------------------

class TestExecuteTyped:

    def test_valid_payload_emits_event(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        payload = {"category": "billing", "severity": 3, "summary": "Refund"}
        result = lt._execute_signal_completion({"payload": payload})

        assert result["status"] == "completed"
        assert result["payload"] == payload
        assert result["summary"] == "Refund"  # derived from payload.summary
        assert len(session._ui_hooks.calls) == 1
        assert session._ui_hooks.calls[0]["payload"] == payload

    def test_invalid_payload_returns_error_no_event(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        # severity is wrong type
        result = lt._execute_signal_completion(
            {"payload": {"category": "billing", "severity": "high"}}
        )

        assert result["error"] == "validation_failed"
        assert "validation_error" in result
        assert "schema_path" in result
        assert len(session._ui_hooks.calls) == 0  # NO event emitted

    def test_missing_required_field_returns_error(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"payload": {"category": "billing"}}  # severity missing
        )
        assert result["error"] == "validation_failed"
        assert len(session._ui_hooks.calls) == 0

    def test_payload_without_summary_field_derives_empty_summary(self):
        # Schema doesn't require summary; payload omits it
        schema = {
            "type": "object",
            "properties": {"foo": {"type": "string"}},
            "required": ["foo"],
        }
        session = StubSession(schema=schema)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion({"payload": {"foo": "bar"}})
        assert result["summary"] == ""
        assert result["payload"] == {"foo": "bar"}

    def test_enum_constraint_enforced(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"payload": {"category": "invalid_cat", "severity": 1}}
        )
        assert result["error"] == "validation_failed"

    def test_minimum_constraint_enforced(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"payload": {"category": "tech", "severity": 0}}  # below min
        )
        assert result["error"] == "validation_failed"


# ---------------------------------------------------------------------------
# Schema resolution from session field (path string)
# ---------------------------------------------------------------------------

class TestSchemaResolutionFromPath:

    def test_resolves_path_from_workspace_at_construction(self, tmp_path):
        import json
        ws = tmp_path / "workspace"
        schemas_dir = ws / ".jaato" / "completion_schemas"
        schemas_dir.mkdir(parents=True)
        schema_file = schemas_dir / "test.json"
        schema_file.write_text(json.dumps(SAMPLE_SCHEMA))

        session = StubSession(schema="test.json", workspace_path=str(ws))
        lt = LifecycleTools(session)

        # Schema was resolved at construction
        params = lt.get_tool_schemas()[0].parameters
        assert "payload" in params["properties"]
        assert params["properties"]["payload"] == SAMPLE_SCHEMA

    def test_unresolvable_path_falls_back_to_legacy_summary(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        session = StubSession(
            schema="missing.json", workspace_path=str(ws)
        )
        lt = LifecycleTools(session)

        # Unresolvable schema → resolver returns None → legacy shape
        params = lt.get_tool_schemas()[0].parameters
        assert "summary" in params["properties"]
        assert "payload" not in params["properties"]


# ---------------------------------------------------------------------------
# Hook absence handling
# ---------------------------------------------------------------------------

class TestHookAbsence:

    def test_missing_hooks_returns_error(self):
        class NoHooks:
            _completion_payload_schema = None
            _agent_id = "x"
            workspace_path = None
            _ui_hooks = None

            def get_context_usage(self):
                return {}

        lt = LifecycleTools(NoHooks())
        result = lt._execute_signal_completion({"summary": "hi"})
        assert result["error"] == "No UI hooks available"


# ---------------------------------------------------------------------------
# Interactive-root filter (server 0.6.61+)
#
# A root session connected via an interactive client (TUI, web, chat)
# does NOT see signal_completion in its tool surface.  Sessions need to
# stay alive across user turns; signal_completion would terminate them.
# Subagents (any client) and headless API root sessions still see it.
# ---------------------------------------------------------------------------

from jaato_sdk.events import ClientType, PresentationContext


def _make_pctx(client_type: ClientType) -> PresentationContext:
    """Minimum-viable PresentationContext fixture for filter tests."""
    return PresentationContext(client_type=client_type)


class StubInteractiveSession(StubSession):
    """StubSession + parent + presentation_context for the filter tests."""

    def __init__(
        self,
        schema=None,
        parent: object = None,
        client_type: ClientType = ClientType.API,
    ) -> None:
        super().__init__(schema=schema)
        self._parent_session = parent
        self._presentation_context = _make_pctx(client_type)


class TestInteractiveRootFilter:
    """``signal_completion`` exposure depends on (root-vs-subagent, client_type)."""

    def test_terminal_root_hides_signal_completion(self):
        """TUI session at the top level — tool is HIDDEN."""
        lt = LifecycleTools(StubInteractiveSession(client_type=ClientType.TERMINAL))
        schemas = lt.get_tool_schemas()
        assert all(s.name != "signal_completion" for s in schemas), (
            f"Interactive root should not see signal_completion; got "
            f"{[s.name for s in schemas]}"
        )

    def test_web_root_hides_signal_completion(self):
        """WS browser session at the top level — tool is HIDDEN."""
        lt = LifecycleTools(StubInteractiveSession(client_type=ClientType.WEB))
        assert all(
            s.name != "signal_completion" for s in lt.get_tool_schemas()
        )

    def test_chat_root_hides_signal_completion(self):
        """Chat platform session at the top level — tool is HIDDEN."""
        lt = LifecycleTools(StubInteractiveSession(client_type=ClientType.CHAT))
        assert all(
            s.name != "signal_completion" for s in lt.get_tool_schemas()
        )

    def test_api_root_keeps_signal_completion(self):
        """Headless API client at the top level — tool is EXPOSED.

        Cascade entry points (handoff_test, kb-enablement-2.0
        orchestrators) connect as API clients and rely on
        signal_completion to drive the typed-payload completion contract.
        """
        lt = LifecycleTools(StubInteractiveSession(client_type=ClientType.API))
        names = [s.name for s in lt.get_tool_schemas()]
        assert "signal_completion" in names

    def test_subagent_keeps_signal_completion_even_in_terminal(self):
        """Subagent of any client_type — tool is EXPOSED.

        Subagents need to terminate cleanly to bubble their typed
        payloads up to the parent.  The parent's interactive client_type
        doesn't transitively hide signal_completion from the children.
        """
        parent = object()  # sentinel non-None parent session
        lt = LifecycleTools(StubInteractiveSession(
            parent=parent,
            client_type=ClientType.TERMINAL,
        ))
        names = [s.name for s in lt.get_tool_schemas()]
        assert "signal_completion" in names

    def test_no_presentation_context_keeps_signal_completion(self):
        """Defensive default: missing presentation_context → expose tool.

        Unknown client_type is treated as cascade-friendly (the
        load-bearing case where signal_completion's contract is
        established).  Better to leave the tool than to silently break
        cascades when the presentation context wasn't wired through.
        """
        # Plain StubSession has no _presentation_context attribute.
        lt = LifecycleTools(StubSession())
        names = [s.name for s in lt.get_tool_schemas()]
        assert "signal_completion" in names

    def test_typed_payload_schema_still_filtered_when_interactive_root(self):
        """The filter applies BEFORE the schema-shape decision.

        A profile that declares completion_payload_schema AND is loaded
        in an interactive root still hides signal_completion — the
        filter doesn't care about the payload shape, only about the
        client/session combo.
        """
        lt = LifecycleTools(StubInteractiveSession(
            schema=SAMPLE_SCHEMA,
            client_type=ClientType.TERMINAL,
        ))
        assert all(
            s.name != "signal_completion" for s in lt.get_tool_schemas()
        )

    def test_enter_tier_unaffected_by_filter(self):
        """The filter only hides signal_completion; enter_tier stays.

        Future tier-mode sessions must still see enter_tier even when
        running in interactive root (the per-turn model switching is a
        cost optimization, not a completion contract).
        """
        # Tier mode requires _tier_config — set on the session.
        session = StubInteractiveSession(client_type=ClientType.TERMINAL)
        session._tier_config = object()  # sentinel non-None
        lt = LifecycleTools(session)
        names = [s.name for s in lt.get_tool_schemas()]
        assert "signal_completion" not in names
        assert "enter_tier" in names
