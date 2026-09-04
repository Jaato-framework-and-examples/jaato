"""Tests for shared/lifecycle_tools.py — typed completion payloads."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

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

    def test_no_schema_hides_signal_completion(self):
        """2026-06-07 schema gate: no ``completion_payload_schema``
        → ``signal_completion`` is HIDDEN entirely (returns no
        schema), not the legacy ``{summary: string}`` shape.
        Profiles that want signal_completion must declare a schema.
        See ``test_signal_completion_schema_gate.py`` for the gate
        contract; this test just pins the get_tool_schemas() output."""
        lt = LifecycleTools(StubSession(schema=None))
        schemas = lt.get_tool_schemas()
        assert all(s.name != "signal_completion" for s in schemas), (
            f"signal_completion must be hidden when no schema is "
            f"declared; got {[s.name for s in schemas]}"
        )

    def test_typed_payload_when_schema_present(self):
        """Option G (server 0.6.115+): tool's parameters ARE the schema.

        Top-level properties of completion_payload_schema become the
        tool's flat args directly — no ``payload`` wrapper.
        """
        lt = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
        schemas = lt.get_tool_schemas()
        params = schemas[0].parameters
        # The schema IS the parameter spec.  No "payload" wrapper.
        assert params == SAMPLE_SCHEMA
        # Top-level properties exposed as flat tool args.
        assert "category" in params["properties"]
        assert "severity" in params["properties"]
        # The legacy "payload" key must NOT appear at the parameter level.
        assert "payload" not in params["properties"]

    def test_tool_name_unchanged_with_schema(self):
        """When the tool IS exposed (schema declared), its name is
        always ``signal_completion`` — preserved verbatim regardless
        of schema shape."""
        lt_typed = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
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
        """Option G: args dict IS the payload — no wrapper."""
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        payload = {"category": "billing", "severity": 3, "summary": "Refund"}
        # Args are passed flat (no "payload" key).
        result = lt._execute_signal_completion(payload)

        assert result["status"] == "completed"
        assert result["payload"] == payload
        assert result["summary"] == "Refund"  # derived from payload.summary
        assert len(session._ui_hooks.calls) == 1
        # Downstream consumers still receive payload= as a flat dict.
        assert session._ui_hooks.calls[0]["payload"] == payload

    def test_invalid_payload_returns_error_no_event(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        # severity is wrong type — passed as flat args
        result = lt._execute_signal_completion(
            {"category": "billing", "severity": "high"}
        )

        assert result["error"] == "validation_failed"
        assert "validation_error" in result
        assert "schema_path" in result
        assert len(session._ui_hooks.calls) == 0  # NO event emitted

    def test_missing_required_field_returns_error(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        # severity missing — passed as flat args
        result = lt._execute_signal_completion({"category": "billing"})
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
        result = lt._execute_signal_completion({"foo": "bar"})
        assert result["summary"] == ""
        assert result["payload"] == {"foo": "bar"}

    def test_enum_constraint_enforced(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"category": "invalid_cat", "severity": 1}
        )
        assert result["error"] == "validation_failed"

    def test_minimum_constraint_enforced(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"category": "tech", "severity": 0}  # below min
        )
        assert result["error"] == "validation_failed"

    def test_legacy_wrapped_payload_now_fails(self):
        """Option G removes the legacy ``{"payload": {...}}`` wrapper.

        A caller that still passes the old shape will now have an arg
        named ``payload`` which is not in the schema — surfaces as a
        validation failure (likely on additionalProperties:false, or
        missing required fields).  This test documents the breaking
        change explicitly.
        """
        # Schema with additionalProperties:false to surface the failure cleanly.
        schema = {
            **SAMPLE_SCHEMA,
            "additionalProperties": False,
        }
        session = StubSession(schema=schema)
        lt = LifecycleTools(session)
        result = lt._execute_signal_completion(
            {"payload": {"category": "billing", "severity": 3}}
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

        # Option G: resolved schema IS the tool's parameters.
        params = lt.get_tool_schemas()[0].parameters
        assert params == SAMPLE_SCHEMA

    def test_unresolvable_path_hides_signal_completion(self, tmp_path):
        """2026-06-07 schema gate: an unresolvable
        ``completion_payload_schema: path`` reference yields a
        resolved schema of ``None`` → tool is HIDDEN (same outcome
        as a profile that doesn't declare a schema at all)."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        session = StubSession(
            schema="missing.json", workspace_path=str(ws)
        )
        lt = LifecycleTools(session)

        # Unresolvable path → schema is None → schema gate hides tool.
        schemas = lt.get_tool_schemas()
        assert all(s.name != "signal_completion" for s in schemas), (
            f"Unresolvable schema path must hide signal_completion; "
            f"got {[s.name for s in schemas]}"
        )


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
        """Headless API client at the top level + declared schema —
        tool is EXPOSED.  Cascade entry points (handoff_test,
        kb-enablement-2.0 orchestrators) connect as API clients and
        rely on signal_completion to drive the typed-payload
        completion contract.

        Schema declaration is REQUIRED (2026-06-07 gate); a profile
        without ``completion_payload_schema`` would hide the tool
        even on an API root.
        """
        lt = LifecycleTools(StubInteractiveSession(
            schema=SAMPLE_SCHEMA, client_type=ClientType.API,
        ))
        names = [s.name for s in lt.get_tool_schemas()]
        assert "signal_completion" in names

    def test_subagent_keeps_signal_completion_even_in_terminal(self):
        """Subagent of any client_type + declared schema — tool is
        EXPOSED.  Subagents need to terminate cleanly to bubble
        their typed payloads up to the parent.  The parent's
        interactive client_type doesn't transitively hide
        signal_completion from the children.

        Schema declaration is REQUIRED (2026-06-07 gate); a subagent
        whose profile doesn't declare ``completion_payload_schema``
        no longer sees the tool — schema gate applies uniformly to
        root + subagent.
        """
        parent = object()  # sentinel non-None parent session
        lt = LifecycleTools(StubInteractiveSession(
            schema=SAMPLE_SCHEMA,
            parent=parent,
            client_type=ClientType.TERMINAL,
        ))
        names = [s.name for s in lt.get_tool_schemas()]
        assert "signal_completion" in names

    def test_no_presentation_context_keeps_signal_completion(self):
        """Defensive default with declared schema: missing
        ``presentation_context`` → expose tool.  Unknown client_type
        is treated as cascade-friendly (the load-bearing case where
        signal_completion's contract is established).  Better to
        leave the tool than to silently break cascades when the
        presentation context wasn't wired through.

        Schema declaration is REQUIRED (2026-06-07 gate); without a
        schema the tool would still be hidden regardless of
        presentation_context.
        """
        # Plain StubSession has no _presentation_context attribute.
        lt = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
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
        # Tier mode requires _tier_config — set on the session.  A real
        # ModelTierConfig, not a sentinel: the enter_tier schema is now
        # built FROM the declared tiers (names + per-tier descriptions),
        # so the builder reads the config rather than merely checking it
        # for None.
        from shared.model_tiers import ModelTierConfig
        session = StubInteractiveSession(client_type=ClientType.TERMINAL)
        session._tier_config = ModelTierConfig.from_unified_dict(
            {"executor": "e", "initial": "executor", "fallback": "executor"})
        lt = LifecycleTools(session)
        names = [s.name for s in lt.get_tool_schemas()]
        assert "signal_completion" not in names


# ---------------------------------------------------------------------------
# prepare_completion / query_completion / arg-less signal_completion
# (server 0.6.198+, 2026-06-09).  Composition-burden mitigation for
# small models — see
# [[feedback_small_model_narration_skipping_is_structural]] and
# [[feedback_validation_passed_but_new_blocker_is_real_outcome]].
# ---------------------------------------------------------------------------


# Nested schema with arrays-of-objects mirroring kb cascade context
# stage shape — exercises the pending-required-fields walker on a
# realistic-shape schema.
NESTED_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "service": {
            "type": "string",
            "description": "Service name (e.g. 'billing', 'customer').",
        },
        "stack_config": {
            "type": "object",
            "description": "Tech stack configuration.",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Primary language (e.g. 'java', 'python').",
                },
                "framework": {
                    "type": "string",
                    "description": "Framework (e.g. 'spring-boot', 'fastapi').",
                },
            },
            "required": ["language", "framework"],
        },
        "endpoints": {
            "type": "array",
            "description": "REST endpoints exposed by this service.",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "read", "update", "delete"],
                    },
                    "path": {"type": "string"},
                },
                "required": ["operation", "path"],
            },
        },
    },
    "required": ["service", "stack_config", "endpoints"],
}


# A2 helper to keep call sites compact.
def _set(lt, path, value):
    return lt._execute_prepare_completion({"field_path": path, "value": value})


class TestPrepareCompletionA2Accept:
    """A2 shape: prepare_completion takes (field_path, value).  Both
    required.  Minimum-valid emission is no longer ``{}``."""

    def test_top_level_string_accepted(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = _set(lt, "service", "billing")
        assert result["accepted"] == {"service": "billing"}
        assert result["rejected"] == {}
        assert lt._accumulated_payload == {"service": "billing"}
        assert result["is_complete"] is False

    def test_nested_object_path_accepted(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "stack_config.language", "java")
        _set(lt, "stack_config.framework", "spring-boot")
        assert lt._accumulated_payload["stack_config"] == {
            "language": "java",
            "framework": "spring-boot",
        }

    def test_array_index_path_accepted(self):
        """endpoints[0].operation sets the nested field at index 0
        (creating intermediate empty objects as needed)."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "endpoints[0].operation", "create")
        _set(lt, "endpoints[0].path", "/customers")
        assert lt._accumulated_payload["endpoints"] == [
            {"operation": "create", "path": "/customers"},
        ]

    def test_whole_object_at_array_index_accepted(self):
        """endpoints[2] receives a whole-object value — supports the
        ``set the whole element at that index`` shape."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "endpoints[0]", {"operation": "create", "path": "/a"})
        _set(lt, "endpoints[1]", {"operation": "delete", "path": "/b"})
        assert lt._accumulated_payload["endpoints"] == [
            {"operation": "create", "path": "/a"},
            {"operation": "delete", "path": "/b"},
        ]

    def test_last_write_wins(self):
        """Q7: re-setting the same path overwrites the prior value."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "service", "billing")
        result = _set(lt, "service", "payments")
        assert result["accepted"] == {"service": "payments"}
        assert lt._accumulated_payload["service"] == "payments"


class TestPrepareCompletionA2Reject:
    """A2 rejects malformed inputs surgically — bad field_path,
    bad value, unknown path."""

    def test_missing_field_path_argument(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_prepare_completion({"value": "x"})
        assert result["error"] == "invalid_argument"

    def test_missing_value_argument(self):
        """value is required even when None-ish; missing → reject."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_prepare_completion({"field_path": "service"})
        assert result["error"] == "invalid_argument"

    def test_falsy_value_accepted_not_rejected(self):
        """value=0 / False / "" / [] / {} are valid — distinguish
        from 'missing'.  The schema's type check decides; for
        ``service: string``, value="" is type-valid (empty string)."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = _set(lt, "service", "")
        assert result["accepted"] == {"service": ""}
        assert result["rejected"] == {}

    def test_value_type_mismatch_rejected(self):
        """endpoints expects array; setting endpoints to a string
        is a type violation → rejected per-path."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = _set(lt, "endpoints", "not-an-array")
        assert "endpoints" in result["rejected"]
        # Accumulated NOT mutated for rejected paths.
        assert "endpoints" not in lt._accumulated_payload

    def test_enum_violation_rejected(self):
        """endpoints[0].operation has enum constraint; "INVALID_OP"
        is out-of-enum → rejected."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = _set(lt, "endpoints[0].operation", "INVALID_OP")
        assert "endpoints[0].operation" in result["rejected"]

    def test_unknown_field_path_rejected(self):
        """field_path that doesn't match the schema's structure
        gets a helpful 'does not match' message."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = _set(lt, "nonexistent_field", "x")
        assert "nonexistent_field" in result["rejected"]
        assert "does not match" in result["rejected"]["nonexistent_field"]

    def test_malformed_path_syntax_rejected(self):
        """Unmatched brackets / non-integer indices → reject as
        invalid syntax."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = _set(lt, "endpoints[abc].operation", "create")
        assert "endpoints[abc].operation" in result["rejected"]


class TestPrepareCompletionA2Pending:
    """Pending walker behavior unchanged from PR-268-as-shipped.
    A2 just changes the entry surface; walker still uses the
    original schema with required[] intact."""

    def test_pending_descriptions_jit_delivered(self):
        """Pending fields surface schema descriptions for JIT
        delivery — small models don't need to retain persona prose
        when each response carries field-level guidance."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        # No state set yet — all three top-level required are pending.
        result = lt._execute_query_completion({})
        pending = {p["path"]: p for p in result["pending_required_fields_with_descriptions"]}
        assert pending["service"]["description"] == (
            "Service name (e.g. 'billing', 'customer')."
        )

    def test_nested_required_surface_when_parent_present(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "stack_config.language", "java")
        result = lt._execute_query_completion({})
        pending_paths = {
            p["path"]
            for p in result["pending_required_fields_with_descriptions"]
        }
        assert "stack_config.framework" in pending_paths
        assert "stack_config" not in pending_paths

    def test_array_per_item_required_surface(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "endpoints[0].operation", "create")
        result = lt._execute_query_completion({})
        pending_paths = {
            p["path"]
            for p in result["pending_required_fields_with_descriptions"]
        }
        assert "endpoints[0].path" in pending_paths


class TestEnrichedPendingDescriptor:
    """Pending-field descriptors carry the full schema fragment for
    each path: scalar constraints, nested object properties, array
    items shape, polymorphic branches.  Closes the "B1 first-contact
    opacity" gap surfaced by peer 2026-06-09 — small models get the
    target structure per pending path without N rejection-retry
    rounds against an opaque {type: object} or {type: array}."""

    CONSTRAINED_SCHEMA = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Unique customer identifier.",
                "pattern": "^[a-z][a-z0-9_]{2,15}$",
                "minLength": 3,
                "maxLength": 16,
            },
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "exclusiveMinimum": False,
            },
            "increment": {
                "type": "number",
                "multipleOf": 0.5,
            },
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "uniqueItems": True,
                "items": {"type": "string"},
            },
        },
        "required": ["user_id", "score"],
    }

    def test_scalar_constraints_surfaced(self):
        lt = LifecycleTools(StubSession(schema=self.CONSTRAINED_SCHEMA))
        result = lt._execute_query_completion({})
        pending = {p["path"]: p for p in result["pending_required_fields_with_descriptions"]}
        user_id = pending["user_id"]
        assert user_id["pattern"] == "^[a-z][a-z0-9_]{2,15}$"
        assert user_id["minLength"] == 3
        assert user_id["maxLength"] == 16
        score = pending["score"]
        assert score["minimum"] == 0
        assert score["maximum"] == 100
        assert score["exclusiveMinimum"] is False

    def test_nested_object_properties_expanded(self):
        schema = {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "description": "Domain model spec.",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {
                            "type": "string",
                            "pattern": r"^v[0-9]+\.[0-9]+$",
                        },
                    },
                    "required": ["name", "version"],
                },
            },
            "required": ["model"],
        }
        lt = LifecycleTools(StubSession(schema=schema))
        result = lt._execute_query_completion({})
        model_desc = next(
            p for p in result["pending_required_fields_with_descriptions"]
            if p["path"] == "model"
        )
        assert "properties" in model_desc
        assert "name" in model_desc["properties"]
        assert "version" in model_desc["properties"]
        assert model_desc["properties"]["version"]["pattern"] == r"^v[0-9]+\.[0-9]+$"
        assert model_desc["required"] == ["name", "version"]

    def test_array_items_expanded_as_nested_descriptor(self):
        schema = {
            "type": "object",
            "properties": {
                "endpoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["create", "read"],
                            },
                            "path": {"type": "string"},
                        },
                        "required": ["operation", "path"],
                    },
                },
            },
            "required": ["endpoints"],
        }
        lt = LifecycleTools(StubSession(schema=schema))
        result = lt._execute_query_completion({})
        endpoints_desc = next(
            p for p in result["pending_required_fields_with_descriptions"]
            if p["path"] == "endpoints"
        )
        # Back-compat shallow fields AND enriched nested items both present.
        assert endpoints_desc["items_type"] == "object"
        assert endpoints_desc["items_required"] == ["operation", "path"]
        assert "items" in endpoints_desc
        items_desc = endpoints_desc["items"]
        assert "operation" in items_desc["properties"]
        assert items_desc["properties"]["operation"]["enum"] == ["create", "read"]
        assert items_desc["properties"]["path"]["type"] == "string"

    def test_polymorphic_branches_expanded(self):
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"success": {"type": "boolean"}},
                            "required": ["success"],
                        },
                        {
                            "type": "object",
                            "properties": {"error": {"type": "string"}},
                            "required": ["error"],
                        },
                    ],
                },
            },
            "required": ["result"],
        }
        lt = LifecycleTools(StubSession(schema=schema))
        result = lt._execute_query_completion({})
        result_desc = next(
            p for p in result["pending_required_fields_with_descriptions"]
            if p["path"] == "result"
        )
        assert "oneOf" in result_desc
        assert len(result_desc["oneOf"]) == 2
        branch_0 = result_desc["oneOf"][0]
        assert "success" in branch_0["properties"]

    def test_ref_resolved_in_nested_properties(self):
        schema = {
            "type": "object",
            "$defs": {
                "Field": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["str", "int"],
                        },
                    },
                    "required": ["name", "type"],
                },
            },
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Field"},
                },
            },
            "required": ["fields"],
        }
        lt = LifecycleTools(StubSession(schema=schema))
        result = lt._execute_query_completion({})
        fields_desc = next(
            p for p in result["pending_required_fields_with_descriptions"]
            if p["path"] == "fields"
        )
        items = fields_desc["items"]
        assert items["properties"]["name"]["type"] == "string"
        assert items["properties"]["type"]["enum"] == ["str", "int"]

    def test_recursion_depth_bounded(self):
        """Cyclic $defs don't blow up — recursion stops at the bound."""
        schema = {
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "child": {"$ref": "#/$defs/Node"},
                    },
                },
            },
            "properties": {
                "root": {"$ref": "#/$defs/Node"},
            },
            "required": ["root"],
        }
        lt = LifecycleTools(StubSession(schema=schema))
        # If recursion were unbounded, this would never return.
        result = lt._execute_query_completion({})
        root_desc = next(
            p for p in result["pending_required_fields_with_descriptions"]
            if p["path"] == "root"
        )
        assert "value" in root_desc["properties"]

    def test_existing_basic_tests_still_pass(self):
        """Sanity: existing path+type+description expectations hold;
        enriched fields are ADDITIONS, not replacements."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_query_completion({})
        pending = {p["path"]: p for p in result["pending_required_fields_with_descriptions"]}
        assert pending["service"]["description"] == (
            "Service name (e.g. 'billing', 'customer')."
        )
        assert pending["service"]["type"] == "string"


class TestPrepareCompletionRefResolution:
    """Regression for the kb cascade $ref bug 2026-06-09: the kb's
    generation_context.schema.json uses ``$ref: #/$defs/Field``
    patterns where the ``$defs`` lives at root.  Without RefResolver
    binding to root, leaf validation fails with
    ``PointerToNowhere: '/$defs/Field' does not exist`` for any
    field whose value-schema embeds a $ref.  Wide-impact class.
    """

    SCHEMA_WITH_REFS = {
        "type": "object",
        "$defs": {
            "Field": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["string", "integer", "boolean"],
                    },
                },
                "required": ["name", "type"],
            },
        },
        "properties": {
            "model": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Field"},
                    },
                },
                "required": ["name", "fields"],
            },
        },
        "required": ["model"],
    }

    def test_setting_ref_typed_value_resolves_against_root(self):
        """Setting model.fields[0] to a well-formed Field instance
        succeeds (no PointerToNowhere)."""
        lt = LifecycleTools(StubSession(schema=self.SCHEMA_WITH_REFS))
        result = _set(lt, "model.fields[0]", {
            "name": "customer_id", "type": "string",
        })
        assert result["accepted"] == {
            "model.fields[0]": {"name": "customer_id", "type": "string"},
        }
        assert result["rejected"] == {}

    def test_partial_ref_typed_value_accepted_per_relaxed_root(self):
        """A partial Field (missing 'type') succeeds at the
        per-call validator because $defs's required[] is also
        stripped recursively.  Final signal_completion would catch
        it via the strict full-schema gate."""
        lt = LifecycleTools(StubSession(schema=self.SCHEMA_WITH_REFS))
        result = _set(lt, "model.fields[0]", {"name": "customer_id"})
        # Per-call: relaxed.  Allowed.
        assert result["accepted"] == {"model.fields[0]": {"name": "customer_id"}}
        # Pending walker still surfaces the missing 'type' for the user.
        # (Walker uses ORIGINAL schema with required intact.)
        result = lt._execute_query_completion({})
        pending_paths = {
            p["path"]
            for p in result["pending_required_fields_with_descriptions"]
        }
        assert "model.fields[0].type" in pending_paths

    def test_ref_typed_value_with_wrong_enum_rejected(self):
        """The relaxed root still enforces enum constraints — only
        required[] is stripped.  ``type: "INVALID"`` violates the
        Field.type enum → rejected."""
        lt = LifecycleTools(StubSession(schema=self.SCHEMA_WITH_REFS))
        result = _set(lt, "model.fields[0]", {
            "name": "customer_id", "type": "INVALID",
        })
        assert "model.fields[0]" in result["rejected"]


class TestPrepareCompletionA2IsComplete:
    def test_is_complete_flips_when_all_required_set(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "service", "billing")
        _set(lt, "stack_config.language", "java")
        _set(lt, "stack_config.framework", "spring-boot")
        r = _set(lt, "endpoints[0]", {"operation": "create", "path": "/x"})
        assert r["is_complete"] is True


class TestPathParser:
    """Unit tests for ``_parse_field_path`` — covers the path syntax
    surface the persona prose teaches the model."""

    def _parse(self, path):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        return lt._parse_field_path(path)

    def test_simple_key(self):
        assert self._parse("service") == ["service"]

    def test_dot_nested(self):
        assert self._parse("stack_config.language") == [
            "stack_config", "language",
        ]

    def test_array_index_followed_by_dot(self):
        assert self._parse("endpoints[0].operation") == [
            "endpoints", 0, "operation",
        ]

    def test_bare_array_index_at_end(self):
        assert self._parse("endpoints[2]") == ["endpoints", 2]

    def test_multiple_array_indices(self):
        assert self._parse("matrix[0][1]") == ["matrix", 0, 1]

    def test_unmatched_bracket_raises(self):
        with pytest.raises(ValueError):
            self._parse("endpoints[0")

    def test_non_integer_index_raises(self):
        with pytest.raises(ValueError):
            self._parse("endpoints[abc]")


class TestQueryCompletion:
    """query_completion is read-only and returns the accumulated
    snapshot + pending + is_complete."""

    def test_query_does_not_mutate(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "service", "billing")
        before = dict(lt._accumulated_payload)
        result = lt._execute_query_completion({})
        after = dict(lt._accumulated_payload)
        assert before == after
        assert result["accumulated"] == {"service": "billing"}
        assert result["is_complete"] is False

    def test_query_includes_pending_with_descriptions(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_query_completion({})
        assert "pending_required_fields_with_descriptions" in result
        # Three top-level required fields → three pending entries.
        # (Nested required only surface when parent is present.)
        assert len(result["pending_required_fields_with_descriptions"]) == 3

    def test_query_with_no_schema_errors(self):
        lt = LifecycleTools(StubSession(schema=None))
        result = lt._execute_query_completion({})
        assert result["error"] == "no_completion_schema"


class TestArglessSignalCompletion:
    """signal_completion() arg-less synthesizes from accumulated
    state when is_complete=True; rejects with pending list when
    is_complete=False.  Same as PR-268 — A2 just changes the
    accumulator's entry surface, not the synthesis behavior."""

    def test_argless_with_incomplete_state_rejects(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        _set(lt, "service", "billing")
        result = lt._execute_signal_completion({})
        assert result["error"] == "validation_failed"
        assert "pending_required_fields_with_descriptions" in result

    def test_argless_with_empty_accumulated_falls_through_to_legacy(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_signal_completion({})
        assert result["error"] == "validation_failed"

    def test_argless_with_complete_accumulated_proceeds(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        session._signal_completion_called = False
        lt = LifecycleTools(session)
        _set(lt, "category", "billing")
        _set(lt, "severity", 3)
        result = lt._execute_signal_completion({})
        assert result.get("status") == "completed"
        assert result.get("payload") == {"category": "billing", "severity": 3}


class TestLegacySinglShotPreserved:
    """signal_completion(args=full_payload) — legacy single-shot
    path UNCHANGED.  Capable models keep working."""

    def test_full_args_bypasses_accumulator(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        session._signal_completion_called = False
        lt = LifecycleTools(session)
        _set(lt, "category", "billing")
        _set(lt, "severity", 2)
        result = lt._execute_signal_completion({
            "category": "tech",
            "severity": 5,
        })
        assert result["status"] == "completed"
        assert result["payload"] == {"category": "tech", "severity": 5}

    def test_partial_args_rejected_via_existing_jsonschema_path(self):
        lt = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
        _set(lt, "category", "billing")
        _set(lt, "severity", 3)
        result = lt._execute_signal_completion({"category": "billing"})
        assert result["error"] == "validation_failed"


class TestToolRegistration:
    """prepare_completion / query_completion are only registered when
    completion_payload_schema is declared."""

    def test_tools_present_with_schema(self):
        lt = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
        names = [s.name for s in lt.get_tool_schemas()]
        assert "prepare_completion" in names
        assert "query_completion" in names
        assert "signal_completion" in names

    def test_tools_absent_without_schema(self):
        """Without a completion_payload_schema, signal_completion
        itself is hidden (2026-06-07 gate) AND
        prepare/query_completion don't register either."""
        lt = LifecycleTools(StubSession(schema=None))
        names = [s.name for s in lt.get_tool_schemas()]
        assert "prepare_completion" not in names
        assert "query_completion" not in names

    def test_executors_match_registered_tools(self):
        """get_executors() returns callables for prepare/query
        symmetric with get_tool_schemas()."""
        lt = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
        executors = lt.get_executors()
        assert "prepare_completion" in executors
        assert "query_completion" in executors

    def test_tools_auto_approved(self):
        """prepare_completion and query_completion are auto-approved
        — operator shouldn't see permission prompts mid-cascade for
        these orchestration tools."""
        lt = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
        approved = lt.get_auto_approved_tools()
        assert "prepare_completion" in approved
        assert "query_completion" in approved
        assert "signal_completion" in approved


# ---------------------------------------------------------------------------
# Processor-gated composite is_complete + server-side auto-finalize
# (server 0.6.199+)
# ---------------------------------------------------------------------------

_GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "service": {"type": "string"},
        "extra": {"type": "string"},
    },
    "required": ["service"],
}


class _AutoFinalizeStubSession:
    """Session double exercising the prepare→composite-is_complete→
    auto-finalize path without the processor-loading / render-context
    machinery (that's covered in test_completion_processors)."""

    def __init__(self, schema, auto_finalize=False):
        self._completion_payload_schema = schema
        self._agent_id = "main"
        self.workspace_path = None
        self._ui_hooks = StubHooks()
        self._completion_processors = []  # finalization block skipped
        self._agent_params = {}
        self._signal_completion_called = False
        self.runtime = None

        class _Prov:
            _auto_finalize_on_complete = auto_finalize
        self._provider = _Prov()

    def get_context_usage(self):
        return {}

    def get_history(self):
        return []


class TestCompositeIsCompleteAndAutoFinalize:

    def test_incomplete_gate_keeps_is_complete_false(self):
        lt = LifecycleTools(_AutoFinalizeStubSession(_GATE_SCHEMA))
        lt._run_completeness_gate = lambda payload: ["api absent for REST run"]
        result = lt._execute_prepare_completion(
            {"field_path": "service", "value": "billing"}
        )
        # Schema floor met, but the completeness gate says not done.
        assert result["is_complete"] is False
        assert result["still_needed"] == ["api absent for REST run"]

    def test_gate_clear_makes_is_complete_true(self):
        lt = LifecycleTools(_AutoFinalizeStubSession(_GATE_SCHEMA))
        lt._run_completeness_gate = lambda payload: []
        result = lt._execute_prepare_completion(
            {"field_path": "service", "value": "billing"}
        )
        assert result["is_complete"] is True
        assert "still_needed" not in result

    def test_gate_not_run_until_floor_met(self):
        # required=[service, extra]: with only `service` set, the floor
        # is NOT met, so the completeness gate must not even be consulted.
        schema = {
            "type": "object",
            "properties": {"service": {"type": "string"},
                           "extra": {"type": "string"}},
            "required": ["service", "extra"],
        }
        lt = LifecycleTools(_AutoFinalizeStubSession(schema))

        def _boom(payload):
            raise AssertionError("gate must not run before floor is met")
        lt._run_completeness_gate = _boom
        result = lt._execute_prepare_completion(
            {"field_path": "service", "value": "billing"}
        )
        assert result["is_complete"] is False
        assert "still_needed" not in result

    def test_auto_finalize_synthesizes_when_quirk_on(self):
        session = _AutoFinalizeStubSession(_GATE_SCHEMA, auto_finalize=True)
        lt = LifecycleTools(session)
        lt._run_completeness_gate = lambda payload: []
        result = lt._execute_prepare_completion(
            {"field_path": "service", "value": "billing"}
        )
        assert result["is_complete"] is True
        assert result.get("auto_finalized") is True
        # Server-side signal_completion fired: event emitted + flag set.
        assert session._signal_completion_called is True
        assert len(session._ui_hooks.calls) == 1
        assert session._ui_hooks.calls[0]["success"] is True

    def test_no_auto_finalize_when_quirk_off(self):
        session = _AutoFinalizeStubSession(_GATE_SCHEMA, auto_finalize=False)
        lt = LifecycleTools(session)
        lt._run_completeness_gate = lambda payload: []
        result = lt._execute_prepare_completion(
            {"field_path": "service", "value": "billing"}
        )
        assert result["is_complete"] is True
        assert "auto_finalized" not in result
        assert session._signal_completion_called is False
        assert session._ui_hooks.calls == []

    def test_auto_finalize_idempotent_no_double_event(self):
        session = _AutoFinalizeStubSession(_GATE_SCHEMA, auto_finalize=True)
        lt = LifecycleTools(session)
        lt._run_completeness_gate = lambda payload: []
        lt._execute_prepare_completion(
            {"field_path": "service", "value": "billing"}
        )
        # A second prepare after completion must NOT re-emit the event
        # (idempotency guard in _execute_signal_completion).
        lt._execute_prepare_completion(
            {"field_path": "extra", "value": "x"}
        )
        assert len(session._ui_hooks.calls) == 1


class TestFinalizationInvariantBlocksAutoFinalize:
    """A phase:'finalization' processor's errors[] (on_error=fail_completion)
    blocks EVERY signal_completion — including the server-side auto-
    synthesized one — so a hard invariant (e.g. discovery_result absent)
    cannot be bypassed by auto-finalize.  Mirrors 7:1's
    context_discovery_present.py."""

    def test_finalization_error_blocks_auto_finalize(self):
        from shared.completion_processors import LoadedProcessor
        from shared.plugins.subagent.config import CompletionProcessor

        session = _AutoFinalizeStubSession(_GATE_SCHEMA, auto_finalize=True)
        # Non-empty so the finalization processor block runs in
        # _execute_signal_completion.
        session._completion_processors = [
            CompletionProcessor(
                script="discovery_present.py",
                phase="finalization",
                on_error="fail_completion",
            )
        ]
        lt = LifecycleTools(session)
        # Completeness gate clears → composite is_complete True →
        # auto-finalize fires → signal_completion runs finalization procs.
        lt._run_completeness_gate = lambda payload: []
        # Pre-load the finalization processor (skip script-file loading)
        # — it reports a hard invariant violation.
        lt._processors_loaded = [
            LoadedProcessor(
                processor=session._completion_processors[0],
                validate_fn=lambda payload, ctx: {
                    "errors": ["discovery_result.json absent"],
                    "warnings": [], "incomplete": [],
                },
            )
        ]

        result = lt._execute_prepare_completion(
            {"field_path": "service", "value": "billing"}
        )

        # is_complete still True (the completeness gate cleared), but the
        # auto-finalize was REJECTED by the finalization invariant.
        assert result["is_complete"] is True
        assert result.get("auto_finalized") is True
        assert "auto_finalize_rejected" in result
        assert result["auto_finalize_rejected"]["error"] == "validation_failed"
        # Crucially: NOT finalized — no event, flag stays False, so the
        # turn loop does NOT terminate (model retains its turn).
        assert session._signal_completion_called is False
        assert session._ui_hooks.calls == []
