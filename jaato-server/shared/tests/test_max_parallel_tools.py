"""Tool concurrency is a ceiling like the others, not a bare literal.

THE GAP (issue #862).  ``JaatoSession`` capped both of its thread pools at
``min(len(work), 8)`` — a literal, in two places.  ``JAATO_PARALLEL_TOOLS``
turns parallelism off entirely and nothing set the WIDTH, so the only lever
between "eight at once" and "one at a time" was an off switch.

Eight is a fine desktop default and a poor fit for two shapes this tree
already supports: a confined runner whose ``pids_max`` a profile deliberately
made small (eight simultaneous ``cli`` subprocesses hit the cgroup ceiling and
fail non-deterministically), and a rate-limited service behind
``service_connector`` (a burst of eight is the wrong shape whatever the memory
ceiling says).  Both are ``runtime_limits`` questions, so the knob belongs in
``runtime_limits``.

What these tests pin, half by half:

  * the value type — validation, ``from_dict``, the round trip through
    ``_runtime_limits_to_dict``, and that ``apply_isolated_defaults`` does not
    DROP the field (a hand-written constructor call would have);
  * the session — one helper answers for both pools, so the ceiling cannot
    apply to tool execution and not to the token-count fan-out;
  * the wire — the envelope carries it, and an envelope built before the field
    existed still bootstraps;
  * inheritance — MIN across every layer, so a child may only ever narrow the
    pool it was spawned under.

The fourth is the one with teeth.  ``runtime_limits`` as a whole is
child-REPLACES, and a ceiling that a child can widen is not a ceiling.
"""

from __future__ import annotations

import pytest

from shared.plugins.subagent.config import (
    SubagentProfile,
    _merge_runtime_limits,
    _runtime_limits_to_dict,
)
from shared.runtime_limits import (
    DEFAULT_MAX_PARALLEL_TOOLS,
    ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS,
    RuntimeLimits,
    apply_isolated_defaults,
)
from shared.session_envelope import (
    SESSION_ENVELOPE_VERSION,
    SessionInitEnvelope,
)


def _profile(name: str, limits) -> SubagentProfile:
    """A minimal profile carrying nothing but ``runtime_limits``."""
    return SubagentProfile(name=name, description="d", runtime_limits=limits)


def _envelope(**kwargs) -> SessionInitEnvelope:
    """A minimal well-formed session envelope."""
    base = dict(
        session_id="s1",
        workspace_path=None,
        profile_name=None,
        provider_name="anthropic",
        model_name="m",
    )
    base.update(kwargs)
    return SessionInitEnvelope(**base)


class TestValueType:
    """``RuntimeLimits.max_parallel_tools`` — parsing and validation."""

    def test_absent_means_unset(self):
        # Unset is NOT "the default written down": the session needs to
        # distinguish the two to report where the value came from.
        assert RuntimeLimits().max_parallel_tools is None
        assert RuntimeLimits.from_dict({}).max_parallel_tools is None

    def test_from_dict_reads_the_key(self):
        assert RuntimeLimits.from_dict(
            {"max_parallel_tools": 3},
        ).max_parallel_tools == 3

    def test_from_dict_does_not_park_it_in_extra(self):
        # A known field parked in ``extra`` would validate nothing and
        # reach no consumer — the silent-no-op class this tree names.
        limits = RuntimeLimits.from_dict({"max_parallel_tools": 3})
        assert "max_parallel_tools" not in limits.extra

    def test_one_is_legitimate(self):
        # Distinct from JAATO_PARALLEL_TOOLS=false: the pool still runs,
        # so the parallel path's hooks and ordering are unchanged.
        assert RuntimeLimits(max_parallel_tools=1).max_parallel_tools == 1

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "4", True])
    def test_rejects_non_positive_ints(self, bad):
        # ``True`` is an int in Python and would silently mean "1 worker";
        # a profile author writing a bool meant something else entirely.
        with pytest.raises(ValueError, match="max_parallel_tools"):
            RuntimeLimits(max_parallel_tools=bad)

    def test_rejects_absurd_width(self):
        with pytest.raises(ValueError, match="sanity ceiling"):
            RuntimeLimits(max_parallel_tools=100_000)

    def test_round_trips_through_the_snapshot_dict(self):
        # ``_runtime_limits_to_dict`` is the inverse ``profile_to_snapshot``
        # uses; a field it forgets is a field a revived session loses.
        limits = RuntimeLimits(pids_max=16, max_parallel_tools=2)
        assert RuntimeLimits.from_dict(
            _runtime_limits_to_dict(limits),
        ) == limits


class TestIsolatedDefaults:
    """``apply_isolated_defaults`` must not drop a field it never heard of."""

    def test_supplied_width_survives(self):
        merged = apply_isolated_defaults(RuntimeLimits(max_parallel_tools=2))
        assert merged.max_parallel_tools == 2
        # ...and the isolated defaults still fill the rest.
        assert merged.pids_max == ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS.pids_max

    def test_unset_width_stays_unset(self):
        # The isolated defaults deliberately do NOT pin a width: the
        # framework default already fits under pids_max=128, and pinning
        # one here would decouple concurrency from the pids ceiling an
        # operator tightens.
        assert apply_isolated_defaults(None).max_parallel_tools is None

    def test_every_field_is_carried(self):
        # The generic walk is the point: this fails loudly if someone
        # re-hand-writes the constructor and forgets the next field.
        supplied = RuntimeLimits(
            memory_max_mb=64, pids_max=8, cpu_weight=50,
            tool_timeout_seconds=1.0, max_output_bytes=99,
            max_parallel_tools=2,
        )
        assert apply_isolated_defaults(supplied) == supplied


class TestSessionPoolWidth:
    """``JaatoSession._parallel_worker_cap`` — one answer for both pools."""

    @staticmethod
    def _session(width):
        # Constructed without __init__: the helper reads exactly one
        # attribute, and a real session drags a runtime + provider in.
        from shared.jaato_session import JaatoSession
        session = JaatoSession.__new__(JaatoSession)
        session._max_parallel_tools = width
        return session

    def test_unset_uses_the_framework_default(self):
        session = self._session(None)
        assert session._parallel_worker_cap(100) == DEFAULT_MAX_PARALLEL_TOOLS

    def test_default_is_the_literal_it_replaced(self):
        # The pre-#862 behaviour of an unconfigured tree, pinned.
        assert DEFAULT_MAX_PARALLEL_TOOLS == 8

    def test_pending_below_the_ceiling_wins(self):
        assert self._session(None)._parallel_worker_cap(3) == 3

    def test_declared_ceiling_caps_the_pool(self):
        assert self._session(2)._parallel_worker_cap(100) == 2

    def test_a_ceiling_wider_than_the_work_does_not_inflate_it(self):
        assert self._session(64)._parallel_worker_cap(3) == 3

    def test_never_returns_zero(self):
        # ThreadPoolExecutor(max_workers=0) raises; a caller with nothing
        # pending wants a pool it can close, not an exception.
        assert self._session(4)._parallel_worker_cap(0) == 1

    def test_both_pools_read_the_same_helper(self):
        # The literal appeared TWICE.  Pin that neither site grew its own
        # copy back: no bare ``, 8)`` cap remains in the module.
        import inspect
        from shared import jaato_session

        source = inspect.getsource(jaato_session)
        assert "_parallel_worker_cap" in source
        assert source.count("_parallel_worker_cap(") >= 3  # def + 2 sites
        assert "min(len(function_calls), 8)" not in source
        assert "min(len(cache_misses), 8)" not in source


class TestEnvelope:
    """The wire — pool-served sessions are the default, so env vars won't do."""

    def test_round_trips(self):
        env = _envelope(max_parallel_tools=2)
        assert SessionInitEnvelope.from_dict(
            env.to_dict(),
        ).max_parallel_tools == 2

    def test_version_was_bumped(self):
        assert SESSION_ENVELOPE_VERSION >= 6
        assert _envelope().to_dict()["schema_version"] == SESSION_ENVELOPE_VERSION

    def test_older_envelope_without_the_key_bootstraps(self):
        wire = _envelope().to_dict()
        wire.pop("max_parallel_tools")
        wire["schema_version"] = 5
        assert SessionInitEnvelope.from_dict(wire).max_parallel_tools is None

    @pytest.mark.parametrize("junk", ["oops", 0, -3, None, {}])
    def test_malformed_width_reads_as_unset(self, junk):
        # Lenient at the boundary on purpose: the value is validated where
        # it is authored, and refusing the envelope would turn one bad
        # profile key into a session that cannot bootstrap at all.
        wire = _envelope().to_dict()
        wire["max_parallel_tools"] = junk
        assert SessionInitEnvelope.from_dict(wire).max_parallel_tools is None


class TestInheritance:
    """MIN across every layer — a child may only ever TIGHTEN the width."""

    def test_nothing_declared_stays_none(self):
        merged, conflicts = _merge_runtime_limits(
            [_profile("p", None)], _profile("c", None),
        )
        assert merged is None and conflicts == []

    def test_parent_width_is_inherited(self):
        merged, _ = _merge_runtime_limits(
            [_profile("p", RuntimeLimits(max_parallel_tools=2))],
            _profile("c", None),
        )
        assert merged.max_parallel_tools == 2

    def test_child_may_narrow(self):
        merged, _ = _merge_runtime_limits(
            [_profile("p", RuntimeLimits(max_parallel_tools=6))],
            _profile("c", RuntimeLimits(max_parallel_tools=2)),
        )
        assert merged.max_parallel_tools == 2

    def test_child_may_not_widen(self):
        # The whole point.  ``runtime_limits`` is otherwise child-REPLACES,
        # so without the min() a child could hand itself eight workers
        # under a parent that narrowed the pool to two.
        merged, _ = _merge_runtime_limits(
            [_profile("p", RuntimeLimits(max_parallel_tools=2))],
            _profile("c", RuntimeLimits(max_parallel_tools=8)),
        )
        assert merged.max_parallel_tools == 2

    def test_narrowest_parent_wins(self):
        merged, conflicts = _merge_runtime_limits(
            [
                _profile("a", RuntimeLimits(max_parallel_tools=4)),
                _profile("b", RuntimeLimits(max_parallel_tools=2)),
            ],
            _profile("c", None),
        )
        assert merged.max_parallel_tools == 2
        # Divergent widths are not a conflict — the minimum is well-defined.
        assert conflicts == []

    def test_parents_differing_only_in_width_do_not_conflict(self):
        merged, conflicts = _merge_runtime_limits(
            [
                _profile("a", RuntimeLimits(pids_max=64, max_parallel_tools=4)),
                _profile("b", RuntimeLimits(pids_max=64, max_parallel_tools=2)),
            ],
            _profile("c", None),
        )
        assert conflicts == []
        assert merged.pids_max == 64
        assert merged.max_parallel_tools == 2

    def test_ceilings_still_conflict(self):
        # The other half of the block keeps its scalar-override rule: a
        # cgroup file takes ONE value, so two parents disagreeing about a
        # ceiling with no child override is still an error to report.
        merged, conflicts = _merge_runtime_limits(
            [
                _profile("a", RuntimeLimits(pids_max=64)),
                _profile("b", RuntimeLimits(pids_max=32)),
            ],
            _profile("c", None),
        )
        assert merged is None
        assert len(conflicts) == 1 and "runtime_limits" in conflicts[0]

    def test_child_block_replaces_ceilings_but_inherits_the_width(self):
        merged, _ = _merge_runtime_limits(
            [_profile("p", RuntimeLimits(memory_max_mb=512, max_parallel_tools=2))],
            _profile("c", RuntimeLimits(memory_max_mb=1024)),
        )
        assert merged.memory_max_mb == 1024
        assert merged.max_parallel_tools == 2


class TestExplainRuntimeReportsIt:
    """``jaato-scaffold explain runtime`` renders the effective value."""

    def test_data_names_the_field_and_its_default(self):
        from shared.scaffold import explain

        data, text = explain.runtime()
        rows = {r["name"]: r for r in data["runtime_limits"]["fields"]}
        assert rows["max_parallel_tools"]["layer"] == "session"
        assert str(DEFAULT_MAX_PARALLEL_TOOLS) in rows["max_parallel_tools"]["effective"]
        assert "max_parallel_tools" in text

    def test_every_dataclass_field_is_reported(self):
        # The report is a curated table over a dataclass; this is what
        # stops it drifting when the next field lands.
        import dataclasses
        from shared.scaffold import explain

        data, _ = explain.runtime()
        reported = {r["name"] for r in data["runtime_limits"]["fields"]}
        declared = {
            f.name for f in dataclasses.fields(RuntimeLimits)
            if f.name != "extra"
        }
        assert reported == declared

    def test_inheritance_rule_is_stated(self):
        from shared.scaffold import explain

        _, text = explain.runtime()
        assert "MIN across every layer" in text
