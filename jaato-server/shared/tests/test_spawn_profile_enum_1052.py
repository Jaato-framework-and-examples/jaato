"""``spawn_subagent``'s ``profile`` names the profiles that exist (#1052).

``profile`` was an unconstrained string.  A production Telegram bot's model
invented ``profile="summarizer"`` — a name in no profiles tier — got a
"not found" result, read it as retryable, re-worded the task and spawned
again, once per permission prompt, until the operator denied the tool.

#944 established the move for the *presence* of ``profile``: a contract in
the schema beats an error corrected after the fact, because a runtime error
alone is a retry loop that spends turns and can exhaust the completion-nudge
budget.  This applies the same move to the *value*.

WHAT THE ENUM IS NOT.  The issue claimed an invented name "cannot be
emitted".  It can.  An ``enum`` is enforced only under grammar-constrained
decoding (``strict: true``, opt-in via ``api_params.strict_tools``, which
this framework deliberately does not turn on); elsewhere it is read as part
of the description, and under the ``prose_tool_calls`` quirk the whole
parameter schema is prompt-injected text.  Nothing in this framework
validates tool arguments against the schema before dispatch either — so the
runtime not-found refusal is still the only layer that actually refuses.
``TheEnumIsNotAWall`` below pins that, so nobody later mistakes this for a
guarantee and deletes the backstop.

REMOTE SPAWN IS NOT EXEMPT, and ``TheRemoteConstraint`` records the cost
rather than leaving it to be discovered: ``server=`` forwards the name to a
peer that resolves it against its own config root, and no predicate here
decides whether a peer is reachable (see ``_spawn_profile_enum``'s
docstring for why both candidate predicates are unsound).
"""

from __future__ import annotations

from shared.plugins.subagent.config import SubagentConfig, SubagentProfile
from shared.plugins.subagent.plugin import SubagentPlugin
from shared.tool_result_builder import split_executor_result
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/plugins/subagent/plugin.py",
        # Removes the ENUM SPREAD, which is what #1052 added.  The
        # previous version of this reversion added two keys beside it
        # and left `**self._spawn_profile_enum()` in place, so the enum
        # survived, the named test passed, and the case certified
        # nothing (#1065).
        find=(
            "                            **self._spawn_profile_enum(),\n"
            '                            "description": self._spawn_profile_param_text(),\n'
        ),
        replace=(
            '                            "description": self._spawn_profile_param_text(),\n'
        ),
        test="TestTheEnumNamesWhatExists::test_the_enum_lists_the_discovered_profiles",
        because="the pre-#1052 unconstrained string, which let a model name "
                "a profile that exists nowhere and retry the not-found "
                "result forever",
    ),
    Reversion(
        target="jaato-server/shared/plugins/subagent/plugin.py",
        find=(
            "        names = self._available_profile_names()\n"
            "        if not names:\n"
            "            return {}\n"
            '        return {"enum": sorted(names)}'
        ),
        replace=(
            "        names = self._available_profile_names()\n"
            '        return {"enum": sorted(names)}'
        ),
        test="TestWhenTheEnumIsWithheld::test_no_enum_when_nothing_is_available",
        because="emitting ``enum: []``, which makes every value invalid and "
                "is rejected outright by some providers",
    ),
    Reversion(
        target="jaato-server/shared/plugins/subagent/plugin.py",
        find=(
            "        if self._inline_allowed():\n"
            "            return {}\n"
            "        names = self._available_profile_names()"
        ),
        replace=(
            "        names = self._available_profile_names()"
        ),
        test="TestWhenTheEnumIsWithheld::test_no_enum_when_inline_spawning_is_allowed",
        because="the enum drifting off the predicate ``required`` and "
                "``inline_config`` already read, so the schema's surfaces "
                "could disagree about ``allow_inline``",
    ),
    Reversion(
        target="jaato-server/shared/plugins/subagent/plugin.py",
        find='        return {"enum": sorted(names)}',
        replace='        return {"enum": names}',
        test="TestTheEnumNamesWhatExists::test_the_enum_is_sorted_not_in_discovery_order",
        because="an enum ordered by an unsorted ``iterdir()``, which varies "
                "per host and churns the prompt-cache prefix the tool "
                "schema sits in",
    ),
]


def _plugin(*, allow_inline=False, profile_names=(), self_profile=None):
    """A plugin initialized just far enough to build its tool schemas."""
    plugin = SubagentPlugin()
    plugin._initialized = True
    cfg = SubagentConfig(project="", location="")
    cfg.allow_inline = allow_inline
    for name in profile_names:
        cfg.add_profile(SubagentProfile(name=name, description=f"{name} desc"))
    plugin._config = cfg
    plugin._self_profile_name = self_profile
    plugin._parent_plugins = ["cli"]
    return plugin


def _profile_param(plugin):
    for schema in plugin.get_tool_schemas():
        if schema.name == "spawn_subagent":
            return schema.parameters["properties"]["profile"]
    raise AssertionError("spawn_subagent schema missing")


class TestTheEnumNamesWhatExists:

    def test_the_enum_lists_the_discovered_profiles(self):
        param = _profile_param(_plugin(profile_names=["researcher", "scribe"]))
        assert param["enum"] == ["researcher", "scribe"]

    def test_the_enum_is_sorted_not_in_discovery_order(self):
        """``_scan_profiles_dir`` builds the dict from an unsorted
        ``iterdir()``; the tool schema sits in the prompt-cache prefix, so
        a per-host order would re-read the whole prefix for nothing."""
        param = _profile_param(_plugin(profile_names=["zebra", "alpha", "mid"]))
        assert param["enum"] == ["alpha", "mid", "zebra"]

    def test_the_invented_name_from_the_incident_is_not_offered(self):
        param = _profile_param(_plugin(profile_names=["researcher"]))
        assert "summarizer" not in param["enum"]

    def test_the_own_profile_is_excluded_like_the_self_spawn_guard(self):
        """``_available_profile_names`` already drops it, and
        ``_execute_spawn_subagent`` refuses it — offering it in the schema
        would invite exactly the spawn the runtime blocks."""
        param = _profile_param(
            _plugin(profile_names=["me", "other"], self_profile="me"))
        assert param["enum"] == ["other"]

    def test_the_parameter_keeps_its_type_and_description(self):
        param = _profile_param(_plugin(profile_names=["researcher"]))
        assert param["type"] == "string"
        assert "REQUIRED" in param["description"]


class TestWhenTheEnumIsWithheld:

    def test_no_enum_when_nothing_is_available(self):
        """``enum: []`` makes every value invalid and some providers reject
        it outright.  The state is reported in prose by the runtime gate
        and by ``list_subagent_profiles``, not by an empty schema set."""
        assert "enum" not in _profile_param(_plugin(profile_names=[]))

    def test_the_property_survives_an_empty_set(self):
        """It is in ``required`` when inline is disallowed, and a required
        property absent from ``properties`` is an unsatisfiable schema."""
        plugin = _plugin(profile_names=[])
        for schema in plugin.get_tool_schemas():
            if schema.name == "spawn_subagent":
                assert "profile" in schema.parameters["properties"]
                assert "profile" in schema.parameters["required"]
                return
        raise AssertionError("spawn_subagent schema missing")

    def test_no_enum_when_only_the_own_profile_exists(self):
        param = _profile_param(_plugin(profile_names=["me"], self_profile="me"))
        assert "enum" not in param

    def test_no_enum_when_inline_spawning_is_allowed(self):
        """Hung off ``_inline_allowed`` — the predicate ``required`` and
        ``inline_config`` already read — so the surfaces cannot disagree."""
        param = _profile_param(
            _plugin(allow_inline=True, profile_names=["researcher"]))
        assert "enum" not in param

    def test_no_config_at_all_withholds_the_enum(self):
        plugin = SubagentPlugin()
        plugin._initialized = True
        plugin._config = None
        plugin._parent_plugins = ["cli"]
        assert "enum" not in _profile_param(plugin)


class TestTheEnumFollowsTheSamePredicateAsRequired:
    """One predicate, read by both, so they cannot drift apart."""

    def test_enum_and_required_move_together(self):
        denied = _plugin(allow_inline=False, profile_names=["researcher"])
        allowed = _plugin(allow_inline=True, profile_names=["researcher"])

        for plugin, wants_enum in ((denied, True), (allowed, False)):
            for schema in plugin.get_tool_schemas():
                if schema.name != "spawn_subagent":
                    continue
                params = schema.parameters
                required_has_profile = "profile" in params["required"]
                has_enum = "enum" in params["properties"]["profile"]
                assert has_enum is wants_enum
                assert required_has_profile is wants_enum


class TestTheSchemaIsRebuiltPerExposure:
    """Subagents share the parent's ``PluginRegistry`` (#944), so a cached
    schema would leak one agent's profile set into another's tool list."""

    def test_a_newly_discovered_profile_appears_without_a_restart(self):
        plugin = _plugin(profile_names=["researcher"])
        assert _profile_param(plugin)["enum"] == ["researcher"]
        plugin._config.add_profile(SubagentProfile(name="scribe", description="d"))
        assert _profile_param(plugin)["enum"] == ["researcher", "scribe"]

    def test_the_enum_follows_a_config_swap(self):
        plugin = _plugin(profile_names=["researcher"])
        assert "enum" in _profile_param(plugin)
        plugin._config.allow_inline = True
        assert "enum" not in _profile_param(plugin)


class TestTheEnumIsNotAWall:
    """The claim "cannot be emitted" holds only under grammar-constrained
    decoding.  These pin the layers that must therefore stay."""

    def test_the_runtime_not_found_refusal_still_exists(self):
        """The real backstop.  Nothing in this framework validates tool
        arguments against the schema before dispatch, so an invented name
        still reaches the executor on every non-strict provider.

        Read through ``split_executor_result``, the contract #1053 put on
        this path: the same refusal now also carries ``ok=False``, so the
        two layers can be asserted together rather than one of them
        pinning a return shape the other owns.
        """
        plugin = _plugin(profile_names=["researcher"])
        ok, result = split_executor_result(
            plugin._execute_spawn_subagent(
                {"task": "t", "profile": "summarizer"}))
        assert ok is False, "the refusal must reach the reliability plugin (#1053)"
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_the_schema_is_plain_json_no_framework_validator_runs(self):
        """The enum is a value in a dict handed to the provider; it is not
        an assertion this process enforces."""
        param = _profile_param(_plugin(profile_names=["researcher"]))
        import json
        json.dumps(param)  # what ``_prose_tools`` does: it becomes prompt text


class TestTheRemoteConstraint:
    """``server=`` forwards the name to a peer that resolves it against its
    own ``config_root``.  The enum is built from the LOCAL set and is not
    exempted for remote spawns — no predicate available at schema-build time
    decides whether a peer is reachable.  Recorded, not hidden."""

    def test_the_enum_is_local_only(self):
        param = _profile_param(_plugin(profile_names=["local-only"]))
        assert param["enum"] == ["local-only"]

    def test_the_server_parameter_says_so(self):
        for schema in _plugin(profile_names=["p"]).get_tool_schemas():
            if schema.name == "spawn_subagent":
                desc = schema.parameters["properties"]["server"]["description"]
                assert "#1052" in desc
                assert "declared locally" in desc
                return
        raise AssertionError("spawn_subagent schema missing")

    def test_a_remote_spawn_never_resolves_the_name_locally(self):
        """Why a local stub profile of the same name is a sufficient
        workaround: the remote branch returns before profile resolution, so
        the stub satisfies the schema and changes nothing the peer runs."""
        plugin = _plugin(profile_names=["local-only"])
        seen = {}

        def handler(**kwargs):
            seen.update(kwargs)
            return {"success": True, "subagent_id": "remote_1"}

        plugin._remote_spawn_handler = handler
        ok, result = split_executor_result(
            plugin._execute_spawn_subagent(
                {"task": "t", "profile": "peer-only", "server": "peer-a"}))
        assert ok is True
        assert result["success"] is True
        assert seen["profile_name"] == "peer-only"
