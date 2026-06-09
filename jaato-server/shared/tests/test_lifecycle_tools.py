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
        # Tier mode requires _tier_config — set on the session.
        session = StubInteractiveSession(client_type=ClientType.TERMINAL)
        session._tier_config = object()  # sentinel non-None
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


class TestPrepareCompletionAccept:
    """prepare_completion merges accepted partials into accumulated
    state and surfaces what's pending."""

    def test_simple_field_accepted_and_merged(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_prepare_completion({"service": "billing"})

        assert result["accepted"] == {"service": "billing"}
        assert result["rejected"] == {}
        assert lt._accumulated_payload == {"service": "billing"}
        assert result["is_complete"] is False
        # Two top-level required still pending (stack_config, endpoints).
        pending_paths = {p["path"] for p in result["pending_required_fields_with_descriptions"]}
        assert "stack_config" in pending_paths or any(
            p.startswith("stack_config.") for p in pending_paths
        )
        assert "endpoints" in pending_paths

    def test_nested_object_merges_per_subkey(self):
        """Successive prepare_completion calls into the same nested
        key merge per-subkey (last-write-wins per leaf)."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        lt._execute_prepare_completion({"stack_config": {"language": "java"}})
        lt._execute_prepare_completion({"stack_config": {"framework": "spring-boot"}})

        assert lt._accumulated_payload["stack_config"] == {
            "language": "java",
            "framework": "spring-boot",
        }

    def test_idempotent_last_write_wins(self):
        """Calling prepare_completion twice with conflicting values
        for the same field — Q7 contract: last-write-wins."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        lt._execute_prepare_completion({"service": "billing"})
        result = lt._execute_prepare_completion({"service": "payments"})

        assert result["accepted"] == {"service": "payments"}
        assert lt._accumulated_payload["service"] == "payments"


class TestPrepareCompletionReject:
    """prepare_completion rejects malformed fields per-field, keeping
    well-formed siblings in accepted (Q6)."""

    def test_per_field_rejection_keeps_siblings(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_prepare_completion({
            "service": "billing",
            # endpoints requires array; this is an object → reject.
            "endpoints": {"not_an_array": True},
        })

        assert "service" in result["accepted"]
        assert "endpoints" in result["rejected"]
        # Accepted siblings ARE merged into accumulated.
        assert lt._accumulated_payload["service"] == "billing"
        # Rejected field is NOT in accumulated.
        assert "endpoints" not in lt._accumulated_payload

    def test_unknown_field_rejected_with_valid_keys_hint(self):
        """Unknown top-level keys get a helpful rejection message
        listing the valid schema keys."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_prepare_completion({"nonexistent": "value"})
        assert "nonexistent" in result["rejected"]
        assert "service" in result["rejected"]["nonexistent"]  # lists valid keys

    def test_enum_violation_rejected(self):
        """Enum-typed field gets rejected with the value-vs-allowed
        info from jsonschema."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_prepare_completion({
            "endpoints": [{"operation": "INVALID_ENUM", "path": "/x"}],
        })
        assert "endpoints" in result["rejected"]


class TestPrepareCompletionPending:
    """Pending fields surface schema descriptions for JIT delivery."""

    def test_pending_includes_descriptions(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_prepare_completion({})

        pending = {p["path"]: p for p in result["pending_required_fields_with_descriptions"]}
        # Top-level required field with description JIT-delivered.
        assert pending["service"]["description"] == (
            "Service name (e.g. 'billing', 'customer')."
        )
        assert pending["service"]["type"] == "string"

    def test_nested_required_paths_surface(self):
        """When the parent object IS present in accumulated but
        nested required fields are missing, walker surfaces the
        nested paths (not just the parent)."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        lt._execute_prepare_completion({"stack_config": {"language": "java"}})
        result = lt._execute_prepare_completion({})

        pending_paths = {p["path"] for p in result["pending_required_fields_with_descriptions"]}
        # stack_config.framework missing — should be surfaced
        # specifically, not generic "stack_config" since the parent
        # IS in accumulated.
        assert "stack_config.framework" in pending_paths
        assert "stack_config" not in pending_paths

    def test_array_of_objects_per_item_pending(self):
        """If an endpoint is partially populated, the walker
        surfaces missing fields per array index."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        lt._execute_prepare_completion({
            "endpoints": [{"operation": "create"}],  # missing path
        })
        result = lt._execute_prepare_completion({})

        pending_paths = {p["path"] for p in result["pending_required_fields_with_descriptions"]}
        assert "endpoints[0].path" in pending_paths


class TestPrepareCompletionIsComplete:
    """is_complete flips True iff full-schema jsonschema.validate
    passes on accumulated state."""

    def test_is_complete_false_until_all_required_set(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        lt._execute_prepare_completion({"service": "billing"})
        r = lt._execute_prepare_completion({
            "stack_config": {"language": "java", "framework": "spring-boot"},
        })
        assert r["is_complete"] is False  # endpoints still missing
        r = lt._execute_prepare_completion({
            "endpoints": [{"operation": "create", "path": "/customers"}],
        })
        assert r["is_complete"] is True


class TestQueryCompletion:
    """query_completion is read-only and returns the accumulated
    snapshot + pending + is_complete."""

    def test_query_does_not_mutate(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        lt._execute_prepare_completion({"service": "billing"})

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
        # (nested required only surface when parent is present)
        assert len(result["pending_required_fields_with_descriptions"]) == 3

    def test_query_with_no_schema_errors(self):
        """Sessions without completion_payload_schema can't use
        query_completion — surfaces an explicit error."""
        lt = LifecycleTools(StubSession(schema=None))
        result = lt._execute_query_completion({})
        assert result["error"] == "no_completion_schema"


class TestArglessSignalCompletion:
    """signal_completion() arg-less synthesizes from accumulated
    state when is_complete=True; rejects with pending list when
    is_complete=False."""

    def test_argless_with_incomplete_state_rejects(self):
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        lt._execute_prepare_completion({"service": "billing"})

        result = lt._execute_signal_completion({})
        assert result["error"] == "validation_failed"
        # Same shape as query_completion's pending list so agent gets
        # the JIT-delivered descriptions without an extra tool call.
        assert "pending_required_fields_with_descriptions" in result

    def test_argless_with_empty_accumulated_falls_through_to_legacy(self):
        """When NO accumulator state AND no args, signal_completion
        falls through to existing schema validation against ``args``
        (which is empty) → validation_failed via the existing
        jsonschema path.  Confirms no behavioral regression on
        first-time calls."""
        lt = LifecycleTools(StubSession(schema=NESTED_SCHEMA))
        result = lt._execute_signal_completion({})
        assert result["error"] == "validation_failed"

    def test_argless_with_complete_accumulated_proceeds(self):
        """When accumulated state is complete, arg-less signal_completion
        synthesizes the payload + proceeds to validation/processors/event
        emission.  Verified by checking that the result includes the
        synthesized payload (not the error path)."""
        # Use SAMPLE_SCHEMA (simpler) for this test since processor wiring
        # is the existing-tests' responsibility.
        session = StubSession(schema=SAMPLE_SCHEMA)
        # Existing tests need these on session; mirror their setup.
        session._signal_completion_called = False

        lt = LifecycleTools(session)
        # Populate accumulator via prepare_completion.
        lt._execute_prepare_completion({"category": "billing", "severity": 3})
        assert lt._accumulated_payload == {"category": "billing", "severity": 3}

        result = lt._execute_signal_completion({})
        # Synthesis path: result has the synthesized payload.
        assert result.get("status") == "completed"
        assert result.get("payload") == {"category": "billing", "severity": 3}


class TestLegacySinglShotPreserved:
    """signal_completion(args=full_payload) still works as the legacy
    single-shot path — accumulator is bypassed entirely."""

    def test_full_args_bypasses_accumulator(self):
        session = StubSession(schema=SAMPLE_SCHEMA)
        session._signal_completion_called = False
        lt = LifecycleTools(session)

        # Even if accumulator has data, full args wins.
        lt._execute_prepare_completion({"category": "billing", "severity": 2})
        result = lt._execute_signal_completion({
            "category": "tech",
            "severity": 5,
        })

        assert result["status"] == "completed"
        assert result["payload"] == {"category": "tech", "severity": 5}

    def test_partial_args_rejected_via_existing_jsonschema_path(self):
        """signal_completion(args=partial) — Q9 contract: doesn't
        merge with accumulator, instead fails the existing
        jsonschema.validate gate.  Confirms no surprise mixing."""
        lt = LifecycleTools(StubSession(schema=SAMPLE_SCHEMA))
        # Populate accumulator to test that args=partial doesn't pull
        # from it.
        lt._execute_prepare_completion({"category": "billing", "severity": 3})

        # args=partial (missing severity, required) → existing
        # validation_failed path.
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
