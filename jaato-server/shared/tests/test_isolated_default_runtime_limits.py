"""Unit tests for Phase 5 §5.1 — isolated-subagent default RuntimeLimits.

Covers :data:`ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS` + the
:func:`apply_isolated_defaults` merge helper that lives in
``shared/runtime_limits.py``.

The merge helper is consumed by
``SessionManager._spawn_isolated_runner`` to fill in the
isolation-implies-bounds invariant whenever a SubagentProfile omits
the corresponding field.  See
``docs/design/phase5_5_1_isolated_default_runtime_limits_audit.md``.
"""

from __future__ import annotations

from shared.runtime_limits import (
    ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS,
    RuntimeLimits,
    apply_isolated_defaults,
)


class TestDefaultConstant:
    """The module-level default carries the planned conservative values."""

    def test_default_memory_max_mb(self):
        assert ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS.memory_max_mb == 2048

    def test_default_pids_max(self):
        assert ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS.pids_max == 128

    def test_default_cpu_weight(self):
        assert ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS.cpu_weight == 100

    def test_default_tool_timeout_seconds(self):
        assert (
            ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS.tool_timeout_seconds
            == 120.0
        )

    def test_default_max_output_bytes(self):
        assert (
            ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS.max_output_bytes
            == 1_048_576
        )

    def test_default_has_kernel_limits(self):
        """Default must trigger cgroup provision — the entire point of
        §5.1 is closing the "no kernel limits → skip cgroup" gap."""
        assert (
            ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS.has_kernel_limits()
            is True
        )


class TestApplyIsolatedDefaultsNoneSupplied:
    """``supplied=None`` returns the full default per the
    isolation-implies-bounds invariant."""

    def test_returns_default_values(self):
        result = apply_isolated_defaults(None)
        assert result.memory_max_mb == 2048
        assert result.pids_max == 128
        assert result.cpu_weight == 100
        assert result.tool_timeout_seconds == 120.0
        assert result.max_output_bytes == 1_048_576

    def test_returns_fresh_instance_not_module_default(self):
        """Caller mustn't be able to mutate the shared default object
        via the returned instance.  RuntimeLimits is frozen, so the
        practical concern is identity-equality; we still defensive-
        copy to keep the property robust if the dataclass ever
        un-freezes."""
        result = apply_isolated_defaults(None)
        assert result is not ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS
        assert result == ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS


class TestApplyIsolatedDefaultsEmptySupplied:
    """``supplied=RuntimeLimits()`` (all fields None) behaves the same
    as ``supplied=None`` — explicit-empty doesn't bypass the default
    per the user's plan-Q2 decision."""

    def test_empty_supplied_fills_all_fields_from_default(self):
        result = apply_isolated_defaults(RuntimeLimits())
        assert result == ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS


class TestApplyIsolatedDefaultsPerFieldMerge:
    """Per-field merge: supplied wins when non-None; default fills
    when supplied is None.  Mirrors the security mental model
    "isolation implies bounds; explicit values are honoured"."""

    def test_supplied_memory_wins_default_pids_fills(self):
        supplied = RuntimeLimits(memory_max_mb=4096)
        result = apply_isolated_defaults(supplied)
        assert result.memory_max_mb == 4096
        assert result.pids_max == 128
        assert result.cpu_weight == 100

    def test_supplied_pids_wins_default_memory_fills(self):
        supplied = RuntimeLimits(pids_max=64)
        result = apply_isolated_defaults(supplied)
        assert result.pids_max == 64
        assert result.memory_max_mb == 2048

    def test_supplied_cpu_weight_low_value_preserved(self):
        """A profile that intentionally starves a subagent
        (cpu_weight=1) keeps the explicit minimum — defaults
        never override a deliberate tightening."""
        supplied = RuntimeLimits(cpu_weight=1)
        result = apply_isolated_defaults(supplied)
        assert result.cpu_weight == 1
        assert result.memory_max_mb == 2048

    def test_supplied_tool_timeout_wins(self):
        supplied = RuntimeLimits(tool_timeout_seconds=30.0)
        result = apply_isolated_defaults(supplied)
        assert result.tool_timeout_seconds == 30.0
        assert result.max_output_bytes == 1_048_576

    def test_supplied_max_output_bytes_wins(self):
        supplied = RuntimeLimits(max_output_bytes=512)
        result = apply_isolated_defaults(supplied)
        assert result.max_output_bytes == 512
        assert result.tool_timeout_seconds == 120.0

    def test_all_fields_supplied_default_does_not_override(self):
        supplied = RuntimeLimits(
            memory_max_mb=8192,
            pids_max=256,
            cpu_weight=200,
            tool_timeout_seconds=300.0,
            max_output_bytes=8_388_608,
        )
        result = apply_isolated_defaults(supplied)
        assert result == supplied


class TestApplyIsolatedDefaultsExtra:
    """``RuntimeLimits.extra`` (forward-compat dict for unknown keys)
    travels with the supplied value, not the default."""

    def test_extra_preserved_from_supplied(self):
        supplied = RuntimeLimits(extra={"future_field": "v1"})
        result = apply_isolated_defaults(supplied)
        assert result.extra == {"future_field": "v1"}

    def test_extra_is_independent_copy(self):
        """Mutating the result's extra dict must not mutate the
        supplied input — defensive deep-copy on the merge boundary."""
        supplied_extra = {"future_field": "v1"}
        supplied = RuntimeLimits(extra=supplied_extra)
        result = apply_isolated_defaults(supplied)
        # ``RuntimeLimits.extra`` is typed as Mapping, but pytest
        # routinely passes plain dicts.  Either way the helper must
        # not alias the underlying dict.
        assert result.extra is not supplied_extra

    def test_extra_empty_default_when_supplied_is_none(self):
        result = apply_isolated_defaults(None)
        assert dict(result.extra) == {}


class TestApplyIsolatedDefaultsIdempotence:
    """Applying the helper twice yields equal results — the helper
    is a pure function of its input."""

    def test_double_apply_equal_to_single_apply(self):
        supplied = RuntimeLimits(memory_max_mb=4096)
        once = apply_isolated_defaults(supplied)
        twice = apply_isolated_defaults(once)
        assert once == twice

    def test_double_apply_returns_distinct_instances(self):
        once = apply_isolated_defaults(None)
        twice = apply_isolated_defaults(once)
        assert once is not twice
        assert once == twice


