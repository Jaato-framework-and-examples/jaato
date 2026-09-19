"""The envelope's ``gc`` had a producer, a schema, a reader — and no
consumer (#1133).

Third in the family ``test_envelope_carries_budget_control.py`` and
``test_envelope_carries_runtime_limits.py`` belong to, and the worst of
the three.  For ``budget_control`` the field existed and nobody
populated it.  For ``runtime_limits`` the value rode a different
mechanism to an executor no session dispatches through.  Here the field
was populated, serialized, deserialized, documented in its own
docstring, and **no file under ``server/runner/`` read it at all** — so
on the runner-served path, the default, ``JaatoSession._gc_plugin``
stayed ``None`` whatever the profile declared, while ``core.py`` kept
resolving the same config to fill the TUI and web-rail readouts.  A
strategy displayed and never run.

Measured against ``main`` @ ``e81225ca`` before the fix::

    A. envelope.gc                  PASS  {'type': 'budget'}
    B. forwarded to create_session  FAIL  gc appears in NO kwarg
    C. runner session._gc_plugin    FAIL  None
       control: after set_gc_plugin PASS  BudgetGCPlugin

The control is the load-bearing line: the same session object, handed a
plugin the way the embedded path hands one, reports it — so ``None`` in
C was the absence of an install and not a blind probe.

**And the producer was broken too**, which is why this file asserts the
knobs and not merely the presence of a block.  It read
``getattr(gc_obj, "config", None)`` — an attribute ``GCProfileConfig``
does not have — so ``gc_config`` was always ``{}`` and the envelope
carried nothing but ``{"type": ...}``.  Wiring only the consumer would
have installed the strategy at *framework defaults* and silently
discarded every number the profile declared: the same silent-ignore
shape one layer over, wearing the fix as a disguise.

Three write-sides for one dataclass existed, each drifted differently —
the producer (1 of 12 fields), the session-snapshot serializer (10 of
12, missing ``target_percent`` and ``pressure_percent``), and
``from_dict`` (12, the only complete one).  ``GCProfileConfig.to_dict``
is the missing symmetric half, derived from the fields so it cannot be
edited out of date.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from shared.plugins.subagent.config import (
    GCProfileConfig,
    gc_profile_to_plugin_config,
)
from shared.session_envelope import SessionInitEnvelope
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/server/runner/session.py",
        find="    _install_gc(session, envelope)\n",
        replace="    pass  # _install_gc(session, envelope)\n",
        test="TestTheConsumer::test_the_bootstrap_actually_calls_the_installer",
        because="the runner never installing the GC strategy the envelope "
                "carries, so every daemon-served session grows its history "
                "unbounded while the UI reports a strategy that never runs",
    ),
    Reversion(
        target="jaato-server/server/runner_spawn.py",
        find=(
            "            to_dict = getattr(gc_obj, \"to_dict\", None)\n"
            "            if callable(to_dict):\n"
            "                gc_dict = to_dict()\n"
        ),
        replace=(
            "            to_dict = None\n"
            "            if getattr(gc_obj, \"config\", None) is not None:\n"
            "                gc_dict = {\"type\": gc_obj.type}\n"
        ),
        test="TestTheProducer::test_the_envelope_carries_every_declared_knob",
        because="the producer reading an attribute GCProfileConfig does not "
                "have, so the envelope carried only the strategy name and "
                "every declared threshold was dropped before the consumer "
                "could see it",
    ),
    Reversion(
        target="jaato-server/shared/plugins/subagent/serializer.py",
        find="            profile_data['gc'] = profile.gc.to_dict()\n",
        replace=(
            "            profile_data['gc'] = {\n"
            "                'type': profile.gc.type,\n"
            "                'threshold_percent': profile.gc.threshold_percent,\n"
            "            }\n"
        ),
        test="TestTheSnapshot::test_a_revived_profile_keeps_every_gc_field",
        because="a hand-written key list on the snapshot write side, which "
                "is how target_percent and pressure_percent were lost across "
                "a revive while every other field survived",
    ),
]


# ----------------------------------------------------------------------
# Fixtures — a profile whose every GC knob is set to a value nothing
# else in the tree uses, so a default cannot be mistaken for a carry.
# ----------------------------------------------------------------------

def _gc() -> GCProfileConfig:
    return GCProfileConfig(
        type="budget",
        threshold_percent=55.0,
        target_percent=30.0,
        pressure_percent=77.0,
        preserve_recent_turns=9,
        notify_on_gc=False,
        max_turns=123,
        media_bytes_threshold=4096,
        evict_consumed_media=False,
        media_evict_mime_prefixes=["audio/", "video/"],
        plugin_config={"marker": "carried"},
    )


def _profile(**kw):
    base = dict(
        name="p", description="d", provider="echo", model="echo-1",
        plugins=[], preloaded_plugins=set(), plugin_configs={},
        tool_scopes={}, model_tiers={}, gc=_gc(), runtime_limits=None,
        env={}, completion_payload_schema=None, spawn_payload_schema=None,
        completion_processors=[], budget_control=None, quirks={},
        apparmor=False, apparmor_fragments=None,
        system_instructions=None, agent_params={},
        suppress_base_instructions=False, config_root=None, inherits=None,
        icon=None, description_for_model=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _envelope(gc=None, workspace_path="/tmp/jaato-no-such-workspace"):
    return SessionInitEnvelope(
        session_id="s1", workspace_path=workspace_path, profile_name="p",
        model_name="echo-1", provider_name="echo", plugins=[], gc=gc,
    )


class _FakeSession:
    """Just enough of JaatoSession for the install to land on."""

    def __init__(self):
        self._gc_plugin = None
        self._gc_config = None

    def set_gc_plugin(self, plugin, config=None):
        self._gc_plugin = plugin
        self._gc_config = config


# ----------------------------------------------------------------------
# 1. The one definition
# ----------------------------------------------------------------------

class TestOneSerializer:
    """``to_dict`` / ``from_dict`` are symmetric and complete."""

    def test_every_field_is_carried(self):
        """Coverage is asserted against the dataclass, not a list.

        A list would have to be edited whenever a field is added, which
        is the defect this replaces rather than a test of it.
        """
        assert set(_gc().to_dict()) == set(
            GCProfileConfig.__dataclass_fields__
        )

    def test_it_round_trips_through_json(self):
        original = _gc()
        revived = GCProfileConfig.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )
        assert revived == original

    def test_a_tuple_normalises_to_the_declared_list(self):
        """``media_evict_mime_prefixes`` is declared ``Optional[List[str]]``.

        A tuple reaches it from callers that build the dataclass by
        hand; the dict has to survive ``json.dumps`` and compare equal
        after a round trip either way.
        """
        out = GCProfileConfig(media_evict_mime_prefixes=("audio/",)).to_dict()
        assert out["media_evict_mime_prefixes"] == ["audio/"]
        assert json.loads(json.dumps(out)) == out


# ----------------------------------------------------------------------
# 2. The producer
# ----------------------------------------------------------------------

class TestTheProducer:
    """``build_session_envelope`` runs for pool-served and cold-spawned
    sessions alike — the path every session takes."""

    def test_the_envelope_carries_every_declared_knob(self):
        """THE producer regression: one field of twelve used to survive."""
        from server.runner_spawn import build_session_envelope

        env = build_session_envelope(
            server=SimpleNamespace(
                _profile=_profile(), config_root=None,
                _main_agent_id="main", _cascade_driver_id=None,
            ),
            session_id="s1", workspace_path="/tmp/ws", profile_name="p",
        )

        assert env.gc == _gc().to_dict()
        # Named individually as well: an equality against ``to_dict``
        # would still pass if BOTH sides lost the same field.
        assert env.gc["threshold_percent"] == 55.0
        assert env.gc["target_percent"] == 30.0
        assert env.gc["pressure_percent"] == 77.0
        assert env.gc["preserve_recent_turns"] == 9
        assert env.gc["media_bytes_threshold"] == 4096
        assert env.gc["plugin_config"] == {"marker": "carried"}

    def test_a_profile_with_no_gc_carries_none(self):
        from server.runner_spawn import build_session_envelope

        env = build_session_envelope(
            server=SimpleNamespace(
                _profile=_profile(gc=None), config_root=None,
                _main_agent_id="main", _cascade_driver_id=None,
            ),
            session_id="s1", workspace_path="/tmp/ws", profile_name="p",
        )
        assert env.gc is None


# ----------------------------------------------------------------------
# 3. The consumer — the step that did not exist
# ----------------------------------------------------------------------

class TestTheConsumer:
    """``server/runner/`` read ``envelope.gc`` nowhere at all."""

    def test_the_bootstrap_actually_calls_the_installer(self):
        """THE regression — and it has to be asserted on the CALL SITE.

        Every other test in this class exercises ``_install_gc``
        directly, which says the installer works and says nothing about
        whether anything invokes it.  That is precisely the pre-#1133
        state: the resolution existed (``core.py`` ran it for the UI),
        and no one called it for the session.  A test that imports the
        function and calls it would have passed on the broken tree.

        So this walks the bootstrap's AST and requires the call.  Source
        inspection is the tool that matches the property — the sibling
        guards ``test_budget_mid_turn_955.py`` and
        ``test_orphan_bound_observes_attachment_812.py`` assert their
        call sites the same way, for the same reason.
        """
        import ast
        import inspect

        from server.runner import session as mod

        tree = ast.parse(inspect.getsource(mod))
        callers = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "_install_gc"
                for c in ast.walk(node)
            )
        }

        assert callers, (
            "no function in server/runner/session.py calls _install_gc(). "
            "The installer exists and nothing invokes it — which is the "
            "shape #1133 was: a resolution with no consumer."
        )

    def test_a_runner_built_session_holds_the_gc_plugin(self):
        """The installer itself, given a declared block."""
        from server.runner.session import _install_gc

        sess = _FakeSession()
        _install_gc(sess, _envelope(gc=_gc().to_dict()))

        assert sess._gc_plugin is not None
        assert sess._gc_config is not None

    def test_the_declared_numbers_reach_the_config(self):
        """Installing at framework defaults would pass the test above.

        This is the half that makes a consumer-only fix insufficient:
        the plugin arrives, and every threshold the profile declared is
        gone.
        """
        from server.runner.session import _install_gc

        sess = _FakeSession()
        _install_gc(sess, _envelope(gc=_gc().to_dict()))

        cfg = sess._gc_config
        assert cfg.threshold_percent == 55.0
        assert cfg.target_percent == 30.0
        assert cfg.pressure_percent == 77.0
        assert cfg.preserve_recent_turns == 9
        assert cfg.media_bytes_threshold == 4096

    def test_no_declaration_installs_nothing(self):
        """The control.

        Without it, the two tests above would pass on a tree that
        installs a default strategy unconditionally — which is a
        different defect, not a fix.
        """
        from server.runner.session import _install_gc

        sess = _FakeSession()
        _install_gc(sess, _envelope(gc=None))

        assert sess._gc_plugin is None
        assert sess._gc_config is None

    def test_an_unresolvable_block_does_not_fail_the_bootstrap(self):
        """A session that cannot install GC is every pre-#1133 session.

        Raising here would turn a silent degradation into a refused
        bootstrap — strictly worse than the defect being fixed.
        """
        from server.runner.session import _install_gc

        sess = _FakeSession()
        _install_gc(sess, _envelope(gc={"type": "no-such-strategy"}))

        assert sess._gc_plugin is None

    def test_it_warns_when_a_declared_strategy_does_not_install(self, caplog):
        """Best-effort must not mean silent — that is the whole issue."""
        from server.runner.session import _install_gc

        with caplog.at_level("WARNING"):
            _install_gc(_FakeSession(), _envelope(gc={"type": "nope"}))

        assert any("GC install failed" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# 4. The third write-side
# ----------------------------------------------------------------------

class TestTheSnapshot:
    """``serialize_subagent_state`` listed ten of twelve fields."""

    def test_a_revived_profile_keeps_every_gc_field(self):
        from shared.plugins.subagent.serializer import (
            deserialize_subagent_state, serialize_subagent_state,
        )
        from shared.plugins.subagent.config import SubagentProfile

        profile = SubagentProfile(name="p", description="d", gc=_gc())
        # ``serialize_subagent_state`` takes the plugin's own
        # ``_active_sessions`` entry — a plain dict, with ``session``
        # absent here because none of the history it would read bears
        # on the GC block.
        blob = serialize_subagent_state({
            "session": None, "profile": profile, "agent_id": "a",
            "created_at": None, "last_activity": None, "turn_count": 0,
        })

        revived = deserialize_subagent_state(
            json.loads(json.dumps(blob))
        )["profile"].gc

        assert revived == _gc()
        # The two the old list dropped, named so a regression says which.
        assert revived.target_percent == 30.0
        assert revived.pressure_percent == 77.0


# ----------------------------------------------------------------------
# 5. One resolution, not two
# ----------------------------------------------------------------------

def test_the_runner_uses_the_frameworks_own_resolver():
    """``_install_gc`` adds a caller, not a second reading of ``gc:``.

    The daemon (for its UI readout) and an in-process subagent both
    resolve a profile block through ``gc_profile_to_plugin_config``.  A
    runner that resolved it its own way would be free to disagree with
    the number the operator is shown.
    """
    from server.runner.session import _install_gc

    sess = _FakeSession()
    _install_gc(sess, _envelope(gc=_gc().to_dict()))
    direct_plugin, direct_cfg = gc_profile_to_plugin_config(_gc())

    assert type(sess._gc_plugin) is type(direct_plugin)
    assert sess._gc_config.threshold_percent == direct_cfg.threshold_percent
    assert sess._gc_config.target_percent == direct_cfg.target_percent
    assert sess._gc_config.preserve_recent_turns == \
        direct_cfg.preserve_recent_turns
