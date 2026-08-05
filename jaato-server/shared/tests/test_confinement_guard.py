"""Fail-closed guard: an in-process session from a confinement-bearing profile.

An in-process session (shared runtime, no runner subprocess) cannot apply a
profile's kernel runtime_limits — silently ignoring them would run unconfined.
The guard rejects instead.
"""

from types import SimpleNamespace

import pytest

from shared.runtime_limits import (
    ConfinementUnavailableError,
    RuntimeLimits,
    assert_inprocess_can_honor,
    profile_requires_kernel_confinement,
)


def _profile(limits):
    return SimpleNamespace(name="p", runtime_limits=limits)


def test_kernel_limits_require_confinement():
    p = _profile(RuntimeLimits(memory_max_mb=512))
    assert profile_requires_kernel_confinement(p) is True


def test_no_limits_do_not_require_confinement():
    assert profile_requires_kernel_confinement(_profile(None)) is False
    assert profile_requires_kernel_confinement(SimpleNamespace(name="p")) is False


def test_empty_limits_do_not_require_confinement():
    # runtime_limits present but no cgroup-enforced field set
    assert profile_requires_kernel_confinement(_profile(RuntimeLimits())) is False


def test_assert_raises_for_kernel_limited_profile():
    p = _profile(RuntimeLimits(pids_max=64))
    with pytest.raises(ConfinementUnavailableError) as exc:
        assert_inprocess_can_honor(p)
    assert "in-process" in str(exc.value).lower()
    assert "'p'" in str(exc.value)


def test_assert_passes_for_unconfined_profile():
    assert_inprocess_can_honor(_profile(None))              # no raise
    assert_inprocess_can_honor(_profile(RuntimeLimits()))   # no raise


def test_cpu_weight_also_triggers():
    assert profile_requires_kernel_confinement(_profile(RuntimeLimits(cpu_weight=100))) is True
