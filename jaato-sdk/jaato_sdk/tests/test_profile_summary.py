"""Tests for ``ProfileSummary`` and ``SessionProfilesEvent`` v1.0 contract.

Pins the stable shape exposed to external clients (daruma-operate and
similar orchestrators).  Any change to these tests is a wire-protocol
contract change and requires bumping ``schema_version``.
"""

from jaato_sdk.events import (
    ProfileParseError,
    ProfileSummary,
    SessionProfilesEvent,
    deserialize_event,
    serialize_event,
)


def test_profile_summary_minimal():
    """A name is the only required field; everything else has a safe default."""
    p = ProfileSummary(name="researcher")
    assert p.description == ""
    assert p.plugins == []
    assert p.preloaded_plugins == []
    assert p.plugin_configs == {}
    assert p.model is None
    assert p.provider is None
    assert p.max_turns == 10
    assert p.model_tiers == {}
    assert p.gc is None
    assert p.runtime_limits is None
    assert p.completion_payload_schema is None
    assert p.env_var_names == []


def test_profile_summary_env_var_names_only_keys_no_values():
    """``env_var_names`` exposes only variable names, never values."""
    # ProfileSummary itself has no notion of env values — it stores
    # only the names list. This pins the contract: callers building a
    # ProfileSummary cannot accidentally include values.
    p = ProfileSummary(
        name="researcher",
        env_var_names=["ANTHROPIC_API_KEY", "DATABASE_URL"],
    )
    serialized = p.model_dump()
    assert serialized["env_var_names"] == ["ANTHROPIC_API_KEY", "DATABASE_URL"]
    # Make sure no value-bearing field exists in the serialized shape.
    assert "env" not in serialized
    assert "env_values" not in serialized
    assert "env_vars" not in serialized


def test_session_profiles_event_default_schema_version():
    """``schema_version`` defaults to ``"1.0"``."""
    ev = SessionProfilesEvent()
    assert ev.schema_version == "1.0"
    assert ev.profiles == []
    assert ev.parse_errors == []


def test_session_profiles_event_round_trip_full_shape():
    """Every field round-trips correctly through JSON serialisation."""
    summary = ProfileSummary(
        name="researcher",
        description="Deep research profile",
        plugins=["cli", "web_search"],
        preloaded_plugins=["cli"],
        plugin_configs={"cli": {"timeout": 30}},
        model="claude-sonnet-4-5",
        provider="anthropic",
        max_turns=25,
        model_tiers={"initial": "fast", "fast": "claude-haiku"},
        gc={"type": "budget", "threshold_percent": 80.0},
        runtime_limits={"memory_limit_bytes": 1_000_000_000},
        completion_payload_schema={"type": "object"},
        env_var_names=["ANTHROPIC_API_KEY"],
    )
    parse_err = ProfileParseError(
        name="broken",
        error="invalid JSON: unexpected token",
    )

    ev = SessionProfilesEvent(
        profiles=[summary],
        parse_errors=[parse_err],
    )

    restored = deserialize_event(serialize_event(ev))

    assert isinstance(restored, SessionProfilesEvent)
    assert restored.schema_version == "1.0"
    assert len(restored.profiles) == 1
    p = restored.profiles[0]
    assert p.name == "researcher"
    assert p.plugins == ["cli", "web_search"]
    assert p.plugin_configs == {"cli": {"timeout": 30}}
    assert p.gc == {"type": "budget", "threshold_percent": 80.0}
    assert p.runtime_limits == {"memory_limit_bytes": 1_000_000_000}
    assert p.model_tiers == {"initial": "fast", "fast": "claude-haiku"}
    assert p.completion_payload_schema == {"type": "object"}
    assert p.env_var_names == ["ANTHROPIC_API_KEY"]

    assert len(restored.parse_errors) == 1
    err = restored.parse_errors[0]
    assert err.name == "broken"
    assert err.error == "invalid JSON: unexpected token"


def test_parse_errors_separate_from_profiles():
    """Parse errors live in their own field, not mixed into profiles.

    Pre-1.0 behavior put broken files as fake profile entries with
    ``description="[parse error] ..."``. The new contract separates
    them so a picker can render them distinctly (banner, log, hide).
    """
    ev = SessionProfilesEvent(
        profiles=[ProfileSummary(name="good")],
        parse_errors=[ProfileParseError(name="broken", error="bad")],
    )
    # Profiles list never contains parse-error sentinels.
    for p in ev.profiles:
        assert "[parse error]" not in p.description
    # Errors are addressable by name on the dedicated field.
    assert ev.parse_errors[0].name == "broken"


def test_deprecated_fields_are_absent():
    """``icon_name`` and ``system_instructions`` are not exposed.

    Both were deprecated. They must not creep back in via ProfileSummary
    additions — clients should not rely on them. ``inherits`` is also
    absent because it's already resolved during discovery.
    """
    p = ProfileSummary(name="x")
    serialized = p.model_dump()
    assert "icon_name" not in serialized
    assert "system_instructions" not in serialized
    assert "inherits" not in serialized
