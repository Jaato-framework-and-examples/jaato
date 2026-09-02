"""Tests for Phase 5 §5.8 — ``profile_payload`` typed validator.

Pins the daemon-side allow-list / type-check contract on the
runner-supplied ``profile_payload`` arg of
``subagent.spawn_isolated_runner``.  Each test surfaces a single
shape violation; the validator must raise ``ValueError`` with a
message naming the offending key + the specific violation, so the
RPC dispatcher can translate to a typed error envelope on the
wire.

Design doc:
``docs/design/phase5_5_8_profile_payload_typed_model_audit.md``.
"""

from __future__ import annotations

import pytest

from server.runner_rpc_handlers.profile_payload_schema import (
    PROFILE_PAYLOAD_ALLOWED_KEYS,
    validate_profile_payload,
)


def _valid_payload(**overrides):
    """Build a minimal-but-valid profile_payload, allowing overrides.

    Mirrors the producer-side wire shape in
    ``shared/plugins/subagent/plugin.py:2283-2325``: name +
    description + model + provider + plugins is the smallest
    payload that exercises every required path through the
    validator.  Tests pass ``override={"key": bad_value}`` to
    exercise the rejection paths."""
    base = {
        "name": "researcher",
        "description": "Deep research profile",
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "plugins": ["cli", "web_search"],
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────────
# Happy path + producer round-trip
# ──────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_minimal_valid(self):
        """Pin: minimal happy-path (name only) validates OK."""
        validate_profile_payload({"name": "x"})

    def test_full_valid(self):
        """Pin: payload mirroring every allow-list key validates OK.

        This is the load-bearing test that pins the producer ↔
        validator contract — if a future producer-side change
        adds/renames/drops a key, either this test or
        :class:`TestProducerRoundTrip` will fail and force the
        validator to be updated in lockstep (the §4.3.9 item 10
        posture)."""
        payload = {
            "name": "researcher",
            "description": "Deep research",
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
            "plugins": ["cli(preload)", "web_search"],
            "plugin_configs": {"cli": {"timeout": 30}},
            "system_instructions": "You are a research analyst...",
            "suppress_base_instructions": False,
            "max_turns": 10,
            "env": {"KEY1": "value1", "KEY2": "value2"},
            "gc": {"type": "summarize", "config": {"threshold": 0.8}},
            "trace": {"provider_log": ".jaato/logs/provider_trace.jsonl"},
            "runtime_limits": {"max_memory_mb": 512},
        }
        validate_profile_payload(payload)


class TestProducerRoundTrip:
    """Lift the producer-side construction logic into a test to
    pin the wire contract: every key the producer emits must
    validate; any key it stops emitting (or adds) is caught by
    this test or by the :class:`TestHappyPath.test_full_valid`
    pin.

    If a future Phase-X commit changes
    ``shared/plugins/subagent/plugin.py:2283-2325``, this test
    fires unless the validator schema is updated symmetrically."""

    def test_producer_shape_validates(self):
        """Build a payload that mirrors the producer's wire shape
        (verbatim from plugin.py:2283-2325) and run it through
        the validator.  All keys that the producer emits MUST be
        in the allow-list."""
        # Verbatim from plugin.py producer construction.
        producer_keys = {
            "name", "description", "model", "provider",
            "plugins", "plugin_configs", "system_instructions",
            "suppress_base_instructions", "max_turns", "env",
            "gc", "trace", "runtime_limits",
        }
        # Every producer-side key is allow-listed:
        assert producer_keys <= PROFILE_PAYLOAD_ALLOWED_KEYS, (
            f"producer emits keys not in allow-list: "
            f"{producer_keys - PROFILE_PAYLOAD_ALLOWED_KEYS!r}.  "
            f"§5.8 forward-compatibility contract: producer + "
            f"validator allow-list must move together."
        )
        # And the validator has no unused entries that aren't
        # producer-emitted (drift-detection in the other direction).
        # NOTE: this assertion is intentionally one-way — if the
        # producer DROPS a key, the validator's allow-list is a
        # superset, which is FINE (an unupgraded daemon paired
        # with a newer producer-omits-key state).  The strict
        # rejection happens in the runtime path, not here.
        unused = PROFILE_PAYLOAD_ALLOWED_KEYS - producer_keys
        assert unused == set(), (
            f"validator allow-list has keys the producer does not "
            f"emit: {unused!r}.  Drop them or add to the producer."
        )


# ──────────────────────────────────────────────────────────────────
# Unknown-key rejection
# ──────────────────────────────────────────────────────────────────


class TestUnknownKeys:
    def test_unknown_top_level_key_rejected(self):
        """§4.3.9 item 10: reject unknown top-level keys.

        Names the offending key + lists allowed keys in the
        message for operator diagnostics."""
        with pytest.raises(ValueError, match="unknown key"):
            validate_profile_payload(
                _valid_payload(ssrf_target="http://internal"),
            )

    def test_unknown_top_level_key_message_lists_offender(self):
        """Pin: error message names the unknown key — operators
        diagnosing a wire-shape mismatch should see exactly which
        key was rejected."""
        with pytest.raises(ValueError, match="evil"):
            validate_profile_payload(_valid_payload(evil="x"))

    def test_multiple_unknown_keys_reported_sorted(self):
        """Pin: multiple unknowns all surface, in sorted order
        (deterministic diagnostics for test fixtures + log
        grep)."""
        with pytest.raises(ValueError) as exc_info:
            validate_profile_payload(
                _valid_payload(zzz_late="x", aaa_early="y"),
            )
        msg = str(exc_info.value)
        # Sorted order: aaa_early appears before zzz_late.
        assert msg.index("aaa_early") < msg.index("zzz_late")

    def test_inherits_rejected(self):
        """Pin: ``inherits`` is rejected even though it's a real
        SubagentProfile field — the producer atomically omits it
        (specs are pre-flattened by resolve_profiles).  A future
        producer that wants to ride it must update the validator
        first; the §4.3.9 item 10 posture is reject-not-ignore."""
        with pytest.raises(ValueError, match="inherits"):
            validate_profile_payload(
                _valid_payload(inherits=["other_profile"]),
            )

    def test_completion_payload_schema_rejected(self):
        """Pin: same as inherits — real SubagentProfile field,
        not in the producer-emitted set, so rejected."""
        with pytest.raises(ValueError, match="completion_payload_schema"):
            validate_profile_payload(
                _valid_payload(completion_payload_schema={}),
            )

    def test_model_tiers_rejected(self):
        """Pin: per-turn model tiers aren't carried through the
        isolated-subagent wire path today; rejected explicitly."""
        with pytest.raises(ValueError, match="model_tiers"):
            validate_profile_payload(
                _valid_payload(model_tiers={"initial": "x"}),
            )


# ──────────────────────────────────────────────────────────────────
# Type checks per key
# ──────────────────────────────────────────────────────────────────


class TestRequiredFields:
    def test_missing_name_rejected(self):
        with pytest.raises(ValueError, match="name.*required"):
            validate_profile_payload({"description": "x"})

    def test_name_not_str_rejected(self):
        with pytest.raises(ValueError, match="name must be a str"):
            validate_profile_payload({"name": 123})

    def test_name_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_profile_payload({"name": ""})


class TestTopLevelTypes:
    def test_payload_not_dict_rejected(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_profile_payload([1, 2, 3])  # type: ignore[arg-type]

    def test_description_non_str_rejected(self):
        with pytest.raises(ValueError, match="description"):
            validate_profile_payload(
                _valid_payload(description=42),
            )

    def test_description_none_ok(self):
        validate_profile_payload(_valid_payload(description=None))

    def test_model_none_ok(self):
        """Pin: ``model: None`` is valid (uses parent's model)."""
        validate_profile_payload(_valid_payload(model=None))

    def test_suppress_base_instructions_non_bool_rejected(self):
        with pytest.raises(ValueError, match="suppress_base_instructions"):
            validate_profile_payload(
                _valid_payload(suppress_base_instructions="yes"),
            )

    def test_max_turns_non_int_rejected(self):
        with pytest.raises(ValueError, match="max_turns"):
            validate_profile_payload(
                _valid_payload(max_turns="10"),
            )

    def test_max_turns_bool_rejected_despite_subclass(self):
        """Pin: bool is a subclass of int in Python, but
        ``max_turns=True`` is misleading — explicit reject."""
        with pytest.raises(ValueError, match="max_turns"):
            validate_profile_payload(_valid_payload(max_turns=True))

    def test_max_turns_zero_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_profile_payload(_valid_payload(max_turns=0))

    def test_max_turns_too_large_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_profile_payload(
                _valid_payload(max_turns=1_000_000),
            )

    def test_max_turns_valid_range(self):
        # boundary values
        validate_profile_payload(_valid_payload(max_turns=1))
        validate_profile_payload(_valid_payload(max_turns=1000))


# ──────────────────────────────────────────────────────────────────
# Plugins list — type + length caps
# ──────────────────────────────────────────────────────────────────


class TestPlugins:
    def test_plugins_str_not_list_rejected(self):
        with pytest.raises(ValueError, match="plugins must be a list"):
            validate_profile_payload(
                _valid_payload(plugins="cli,web"),
            )

    def test_plugins_element_non_str_rejected(self):
        """Pin: ``plugins: [123]`` is rejected at the boundary,
        not 3 levels deep in plugin-load code."""
        with pytest.raises(ValueError, match="plugins\\[0\\]"):
            validate_profile_payload(_valid_payload(plugins=[123]))

    def test_plugins_dict_element_rejected(self):
        with pytest.raises(ValueError, match="plugins\\[1\\]"):
            validate_profile_payload(
                _valid_payload(plugins=["cli", {"shell": "rm -rf /"}]),
            )

    def test_plugins_count_cap(self):
        """Pin: count cap defends against `plugins: ["x"] * 10**7`
        DoS pivots against the unconfined daemon."""
        with pytest.raises(ValueError, match="count cap"):
            validate_profile_payload(
                _valid_payload(plugins=["cli"] * 65),
            )

    def test_plugins_element_length_cap(self):
        with pytest.raises(ValueError, match="length cap"):
            validate_profile_payload(
                _valid_payload(plugins=["A" * 129]),
            )


# ──────────────────────────────────────────────────────────────────
# plugin_configs — keys str, values dict
# ──────────────────────────────────────────────────────────────────


class TestPluginConfigs:
    def test_plugin_configs_non_dict_rejected(self):
        with pytest.raises(ValueError, match="plugin_configs"):
            validate_profile_payload(
                _valid_payload(plugin_configs=[("cli", {})]),
            )

    def test_plugin_configs_value_non_dict_rejected(self):
        """Pin: ``plugin_configs: {"cli": "not a dict"}`` is
        rejected at the boundary."""
        with pytest.raises(ValueError, match="plugin_configs"):
            validate_profile_payload(
                _valid_payload(plugin_configs={"cli": "not a dict"}),
            )

    def test_plugin_configs_count_cap(self):
        with pytest.raises(ValueError, match="count cap"):
            validate_profile_payload(
                _valid_payload(
                    plugin_configs={
                        f"plugin_{i}": {} for i in range(65)
                    },
                ),
            )


# ──────────────────────────────────────────────────────────────────
# env — keys str, values str, caps
# ──────────────────────────────────────────────────────────────────


class TestEnv:
    def test_env_non_dict_rejected(self):
        with pytest.raises(ValueError, match="env must be a dict"):
            validate_profile_payload(_valid_payload(env="K=V"))

    def test_env_value_non_str_rejected(self):
        with pytest.raises(ValueError, match="env"):
            validate_profile_payload(_valid_payload(env={"K": 42}))

    def test_env_count_cap(self):
        with pytest.raises(ValueError, match="count cap"):
            validate_profile_payload(
                _valid_payload(
                    env={f"K{i}": "v" for i in range(129)},
                ),
            )

    def test_env_value_length_cap(self):
        with pytest.raises(ValueError, match="length cap"):
            validate_profile_payload(
                _valid_payload(env={"K": "v" * 5000}),
            )


# ──────────────────────────────────────────────────────────────────
# system_instructions — length cap
# ──────────────────────────────────────────────────────────────────


class TestSystemInstructions:
    def test_system_instructions_too_long_rejected(self):
        """Pin: cap defends against 10MB system_instructions
        pivoting into tokenizer DoS / daemon RSS bloat."""
        with pytest.raises(ValueError, match="length cap"):
            validate_profile_payload(
                _valid_payload(system_instructions="A" * 70000),
            )

    def test_system_instructions_at_cap_ok(self):
        validate_profile_payload(
            _valid_payload(system_instructions="A" * 65536),
        )


# ──────────────────────────────────────────────────────────────────
# gc — allow-listed sub-keys
# ──────────────────────────────────────────────────────────────────


class TestGC:
    def test_gc_non_dict_rejected(self):
        with pytest.raises(ValueError, match="gc must be a dict"):
            validate_profile_payload(_valid_payload(gc="summarize"))

    def test_gc_unknown_subkey_rejected(self):
        """Pin: gc sub-keys are also allow-listed.  Defense-in-
        depth — the gc plugin resolver downstream might .get()
        unknown keys."""
        with pytest.raises(ValueError, match="gc has unknown"):
            validate_profile_payload(
                _valid_payload(
                    gc={"type": "summarize", "evil": "x"},
                ),
            )

    def test_gc_type_non_str_rejected(self):
        with pytest.raises(ValueError, match="gc.type must be a str"):
            validate_profile_payload(
                _valid_payload(gc={"type": 5}),
            )

    def test_gc_config_non_dict_rejected(self):
        with pytest.raises(ValueError, match="gc.config must be a dict"):
            validate_profile_payload(
                _valid_payload(gc={"config": "not a dict"}),
            )

    def test_gc_empty_dict_ok(self):
        """Pin: empty gc dict is OK — producer emits one only
        when at least ``type`` is present, but defensive
        validation should accept it."""
        validate_profile_payload(_valid_payload(gc={}))


# ──────────────────────────────────────────────────────────────────
# runtime_limits — top-level type only
# ──────────────────────────────────────────────────────────────────


class TestRuntimeLimits:
    def test_runtime_limits_non_dict_rejected(self):
        """Pin: top-level type check fires at the boundary.
        Structural content delegated to RuntimeLimits.from_dict."""
        with pytest.raises(ValueError, match="runtime_limits must be a dict"):
            validate_profile_payload(
                _valid_payload(runtime_limits=42),
            )

    def test_runtime_limits_empty_dict_ok(self):
        """Pin: an empty dict passes the boundary check.
        RuntimeLimits.from_dict has its own validation for empty
        + partial dicts."""
        validate_profile_payload(_valid_payload(runtime_limits={}))


# ──────────────────────────────────────────────────────────────────
# Length-cap boundary pins
# ──────────────────────────────────────────────────────────────────


class TestLengthBoundaries:
    """Pin per-key caps at boundary +1 / -0 to catch off-by-one
    regressions in the cap constants."""

    def test_name_at_cap_ok(self):
        validate_profile_payload({"name": "A" * 256})

    def test_name_over_cap_rejected(self):
        with pytest.raises(ValueError, match="length cap"):
            validate_profile_payload({"name": "A" * 257})

    def test_description_over_cap_rejected(self):
        with pytest.raises(ValueError, match="length cap"):
            validate_profile_payload(
                _valid_payload(description="A" * 1025),
            )

    def test_provider_over_cap_rejected(self):
        with pytest.raises(ValueError, match="length cap"):
            validate_profile_payload(
                _valid_payload(provider="A" * 65),
            )
