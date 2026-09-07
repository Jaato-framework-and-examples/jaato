"""Tier NAMES a deployment chooses — `coder`, `reviewer`, `researcher` (#831).

#818 decoupled a tier's ROLE from its NAME (`modalities` moved the image role
off the literal string ``vision``; ``description`` let a profile replace the
framework's prose).  It deliberately left the name set closed, and this is the
follow-on: a profile may now bind a model to a tier it names itself.

The ceiling that mattered was **arity, not aesthetics**.  Four names, one of
which (``vision``) carries a built-in image role, meant a deployment wanting
four cognitive bindings had to overload ``vision`` and inherit an
``image: inbound`` role it never asked for — which the startup capability check
then failed the session on.

What a canonical name gets for free, a free name must supply, and the tests
below pin both halves:

* the framework keeps prose, order, an env spelling and (for ``vision``) an
  implicit modality role for the four names it knows;
* a free name gets none of those, so it MUST carry a ``description`` — which
  is what stops the model reading "routes this session to <model>." as the
  only thing it is told about a tier.

The gate was never one site.  Coverage here follows the seven the issue
enumerated: the config gate, ``model_for``, the ``enter_tier`` executor,
budget-control's degrade overlay, ``jaato-scaffold validate``, the two
documentation surfaces, and the env path that deliberately has no free-name
spelling.
"""

import pytest

from shared.model_tiers import (
    CANONICAL_TIER_NAMES,
    MAX_DECLARED_TIERS,
    ModelTierConfig,
    ModelTierConfigError,
    TierEntry,
    is_canonical_tier_name,
    tier_name_error,
)

CODER = {"model": "anthropic/claude-sonnet-4-6",
         "description": "THE CODER.  Write and edit code to an agreed plan."}


def _cfg(d):
    return ModelTierConfig.from_unified_dict(d)


def _four_role_ladder():
    """The ladder the issue opens with: four cognitive bindings, no vision."""
    return _cfg({
        "planner": {"model": "p", "description": "plan."},
        "coder": dict(CODER),
        "reviewer": {"model": "r", "description": "review a diff."},
        "researcher": {"model": "s", "description": "read docs."},
        "initial": "planner",
        "fallback": "planner",
    })


class TestTheNamePredicate:
    """``tier_name_error`` is the one predicate the seven sites share."""

    @pytest.mark.parametrize("name", sorted(CANONICAL_TIER_NAMES))
    def test_canonical_names_pass(self, name):
        assert tier_name_error(name) is None
        assert is_canonical_tier_name(name)

    @pytest.mark.parametrize(
        "name", ["coder", "reviewer", "researcher", "code_reviewer", "a1", "x_9"])
    def test_free_names_pass_the_pattern(self, name):
        assert tier_name_error(name) is None
        assert not is_canonical_tier_name(name)

    @pytest.mark.parametrize("name", [
        "Coder",        # case-only collision with `coder` — one tier to the
                        # model, two to the config
        "my coder",     # whitespace in a JSON-schema enum value
        "coder-2",      # punctuation
        "1coder",       # must start with a letter
        "c",            # single char
        "c" * 33,       # over the length cap
        "",
    ])
    def test_unusable_names_are_refused_with_a_reason(self, name):
        reason = tier_name_error(name)
        assert reason and isinstance(reason, str)

    @pytest.mark.parametrize("name", ["initial", "fallback"])
    def test_reserved_control_keys_are_not_tier_names(self, name):
        # The unified-dict parser splits on these BEFORE the name check, so
        # a tier called `initial` would be unaddressable rather than merely
        # confusing.  Refusing it by name says so.
        reason = tier_name_error(name)
        assert reason and "reserved control key" in reason

    @pytest.mark.parametrize("name", [None, 7, ["coder"]])
    def test_a_non_string_is_a_reason_not_a_crash(self, name):
        # Raw profile dicts reach the predicate unvalidated.  ``True`` is
        # the realistic one: a YAML 1.1 parser reads an unquoted `on:` key
        # as a boolean, so the name never arrives as a string at all.
        assert tier_name_error(name) is not None

    def test_a_yaml_coerced_boolean_key_is_refused(self):
        assert tier_name_error(True) is not None


class TestTheGateOpens:
    """Site 1 — ``ModelTierConfig.__post_init__``."""

    def test_the_issues_own_repro_now_parses(self):
        cfg = _cfg({"coder": dict(CODER), "initial": "coder",
                    "fallback": "coder"})
        assert cfg.tiers["coder"].model == CODER["model"]

    def test_four_cognitive_bindings_without_overloading_vision(self):
        cfg = _four_role_ladder()
        assert set(cfg.tiers) == {"planner", "coder", "reviewer", "researcher"}
        # The whole point: no tier inherited an image role it never asked for.
        assert cfg.tiers_for_modality("image") == ()

    def test_a_free_name_requires_a_description(self):
        with pytest.raises(ModelTierConfigError, match="description.*required"):
            _cfg({"coder": "some-model", "initial": "coder",
                  "fallback": "coder"})

    def test_a_free_name_requires_a_description_in_rich_form_too(self):
        with pytest.raises(ModelTierConfigError, match="description.*required"):
            _cfg({"coder": {"model": "m"}, "initial": "coder",
                  "fallback": "coder"})

    def test_a_misspelled_control_key_is_told_so(self):
        # `initail: executor` is a legal free name, so it lands in the
        # description branch.  "Needs a description" alone is true about the
        # wrong problem, so the message names the control keys.  Pinned on
        # BOTH surfaces (see TestScaffoldValidate) because validate is the
        # one an author runs first and this is the one they hit if they skip
        # it.
        with pytest.raises(ModelTierConfigError) as exc:
            _cfg({"executor": "e", "initail": "executor"})
        assert "initial" in str(exc.value) and "fallback" in str(exc.value)

    def test_a_canonical_name_still_needs_no_description(self):
        cfg = _cfg({"executor": "e", "initial": "executor",
                    "fallback": "executor"})
        assert cfg.tiers["executor"].description is None

    def test_direct_construction_is_gated_too(self):
        # Premium / test code builds these without from_unified_dict.
        with pytest.raises(ModelTierConfigError, match="description.*required"):
            ModelTierConfig(tiers={"coder": TierEntry("m")},
                            initial_tier="coder", tier_fallback="coder")

    def test_an_unusable_name_is_still_refused(self):
        with pytest.raises(ModelTierConfigError, match="not a usable tier name"):
            _cfg({"Coder": dict(CODER), "initial": "Coder",
                  "fallback": "Coder"})

    def test_the_arity_ceiling_is_enforced(self):
        raw = {f"tier_{i}": {"model": "m", "description": "d"}
               for i in range(MAX_DECLARED_TIERS + 1)}
        raw["initial"] = raw["fallback"] = "tier_0"
        with pytest.raises(ModelTierConfigError, match="at most"):
            _cfg(raw)

    def test_exactly_the_ceiling_is_allowed(self):
        raw = {f"tier_{i}": {"model": "m", "description": "d"}
               for i in range(MAX_DECLARED_TIERS)}
        raw["initial"] = raw["fallback"] = "tier_0"
        assert len(_cfg(raw).tiers) == MAX_DECLARED_TIERS


class TestOrderIsCacheStable:
    """Suggested direction 3 — canonical first, free names alphabetically.

    The order feeds the ``enter_tier`` tool schema, which sits in the
    prompt-cache prefix, so declaring a free tier must not reshuffle the
    canonical ones ahead of it.
    """

    def test_canonical_first_then_free_alphabetically(self):
        cfg = _cfg({
            "researcher": {"model": "s", "description": "d"},
            "coder": dict(CODER),
            "executor": "e",
            "planner": "p",
            "initial": "planner", "fallback": "planner",
        })
        assert cfg.ordered_tier_names() == (
            "planner", "executor", "coder", "researcher")

    def test_order_is_independent_of_declaration_order(self):
        a = _cfg({"coder": dict(CODER), "planner": "p",
                  "initial": "planner", "fallback": "planner"})
        b = _cfg({"planner": "p", "coder": dict(CODER),
                  "initial": "planner", "fallback": "planner"})
        assert a.ordered_tier_names() == b.ordered_tier_names()


class TestOnlyVisionKeepsBuiltInMeaning:
    """Not asking for: changing what ``vision`` implies."""

    def test_a_free_name_gets_no_implicit_role(self):
        from shared.model_tiers import IMPLICIT_TIER_MODALITIES
        assert set(IMPLICIT_TIER_MODALITIES) == {"vision"}
        cfg = _cfg({"coder": dict(CODER), "initial": "coder",
                    "fallback": "coder"})
        assert cfg.tiers["coder"].inbound_modalities == frozenset()

    def test_a_free_name_may_declare_the_image_role_itself(self):
        cfg = _cfg({
            "executor": "e",
            "eyes": {"model": "v", "description": "look at pictures.",
                     "modalities": ["image"]},
            "initial": "executor", "fallback": "executor",
        })
        assert cfg.tiers_for_modality("image") == ("eyes",)


class TestDescribeTier:
    """Gap 2 — the placeholder bullet is now unreachable for a declared tier."""

    def test_a_free_tiers_bullet_is_its_own_description(self):
        cfg = _cfg({"coder": dict(CODER), "initial": "coder",
                    "fallback": "coder"})
        assert cfg.describe_tier("coder") == CODER["description"]

    def test_no_declared_tier_can_reach_the_placeholder(self):
        # Every declared tier either is canonical (framework prose) or
        # carries a description (enforced at construction).
        cfg = _four_role_ladder()
        for name in cfg.ordered_tier_names():
            assert "routes this session to" not in cfg.describe_tier(name)

    def test_the_placeholder_survives_for_an_undeclared_name(self):
        # describe_tier accepts names from outside; the backstop stays.
        cfg = _cfg({"coder": dict(CODER), "initial": "coder",
                    "fallback": "coder"})
        assert "unspecified model" in cfg.describe_tier("never_declared")


class TestModelFor:
    """Site 2 — ``model_for`` resolves a free name like any other."""

    def test_a_declared_free_tier_resolves_to_itself(self):
        cfg = _four_role_ladder()
        assert cfg.model_for("coder") == ("coder", cfg.tiers["coder"])

    def test_an_undeclared_but_usable_name_routes_to_fallback(self):
        cfg = _four_role_ladder()
        name, _ = cfg.model_for("vision")
        assert name == "planner"
        name, _ = cfg.model_for("undeclared_tier")
        assert name == "planner"

    def test_an_unusable_name_still_raises(self):
        cfg = _four_role_ladder()
        with pytest.raises(ModelTierConfigError):
            cfg.model_for("Coder")


class TestEnterTierSchemaAndExecutor:
    """Site 3 — the schema advertises it, so the executor must accept it."""

    def _tools(self, cfg):
        from types import SimpleNamespace
        from shared.lifecycle_tools import LifecycleTools
        return LifecycleTools(SimpleNamespace(
            _tier_config=cfg, _completion_payload_schema=None,
            workspace_path=None, runtime=None))

    def test_the_schema_offers_the_free_name_with_its_own_prose(self):
        schema = self._tools(_four_role_ladder())._enter_tier_schema()
        assert schema.parameters["properties"]["name"]["enum"] == [
            "planner", "coder", "researcher", "reviewer"]
        assert "`coder` — THE CODER." in schema.description

    def test_the_executor_accepts_what_the_schema_advertises(self):
        # Opening only the config gate produced a schema offering `coder`
        # and an executor rejecting it.  Pinned so it cannot return.
        cfg = _four_role_ladder()
        tools = self._tools(cfg)
        switched = []
        tools._session.switch_tier = lambda n: (
            switched.append(n) or {"status": "ok"})
        assert tools._execute_enter_tier({"name": "coder"}) == {"status": "ok"}
        assert switched == ["coder"]

    def test_a_canonical_but_undeclared_name_is_still_addressable(self):
        # Documented behaviour: it routes to fallback and reports
        # `fallback_used`, rather than being refused outright.
        cfg = _four_role_ladder()
        tools = self._tools(cfg)
        tools._session.switch_tier = lambda n: {"status": "fallback_used"}
        assert tools._execute_enter_tier(
            {"name": "vision"}) == {"status": "fallback_used"}

    def test_a_hallucinated_name_is_still_refused(self):
        tools = self._tools(_four_role_ladder())
        out = tools._execute_enter_tier({"name": "codr"})
        assert out["error"] == "invalid_tier"
        assert "coder" in out["message"]


class TestDegradeOverlay:
    """Site 4 — a brownout rung may rebind a deployment-named tier."""

    def test_a_rung_may_rebind_a_free_tier(self):
        from shared.budget_control import BudgetControlConfig
        bc = BudgetControlConfig.from_dict({
            "limits": {"usd": 10},
            "degrade": [{"at": 80, "model_tiers": {"coder": "cheap-model"}}],
        })
        assert bc.degrade[0].model_tiers["coder"].model == "cheap-model"

    def test_a_rung_still_may_not_name_a_control_key(self):
        from shared.budget_control import (
            BudgetControlConfig, BudgetControlConfigError,
        )
        with pytest.raises(BudgetControlConfigError, match="control key"):
            BudgetControlConfig.from_dict({
                "limits": {"usd": 10},
                "degrade": [{"at": 80, "model_tiers": {"initial": "m"}}],
            })

    def test_a_rung_still_may_not_name_an_unusable_tier(self):
        from shared.budget_control import (
            BudgetControlConfig, BudgetControlConfigError,
        )
        with pytest.raises(BudgetControlConfigError, match="usable tier name"):
            BudgetControlConfig.from_dict({
                "limits": {"usd": 10},
                "degrade": [{"at": 80, "model_tiers": {"Coder": "m"}}],
            })

    def test_a_rung_still_may_not_set_a_free_tiers_description(self):
        # A rung rebinds a tier's MODEL, never its role — and for a free
        # tier the description is exactly what the base table had to carry.
        from shared.budget_control import (
            BudgetControlConfig, BudgetControlConfigError,
        )
        with pytest.raises(BudgetControlConfigError, match="not valid in an overlay"):
            BudgetControlConfig.from_dict({
                "limits": {"usd": 10},
                "degrade": [{"at": 80, "model_tiers": {
                    "coder": {"model": "m", "description": "d"}}}],
            })

    def test_the_overlay_carries_a_free_tiers_description_forward(self):
        from shared.budget_control import overlay_tier_table
        cfg = _four_role_ladder()
        overlay_tier_table(cfg.tiers, {"coder": TierEntry("cheap")})
        assert cfg.tiers["coder"].model == "cheap"
        assert cfg.tiers["coder"].description == CODER["description"]


class TestScaffoldValidate:
    """Site 5 — the surface an author runs BEFORE paying for a session."""

    def _diags(self, model_tiers):
        return [(sev, code) for sev, code, _ in self._raw(model_tiers)]

    def _raw(self, model_tiers):
        from shared.scaffold import validate
        out = []
        validate._check_model_tiers(
            model_tiers,
            lambda sev, code, msg, where=None: out.append((sev, code, msg)))
        return out

    def test_a_described_free_tier_is_clean(self):
        assert self._diags({"coder": dict(CODER)}) == []

    def test_an_undescribed_free_tier_is_an_error(self):
        assert ("error", "tier_description_required") in self._diags(
            {"coder": {"model": "m"}})

    def test_a_misspelled_control_key_is_told_so(self):
        # The static half of the pair in TestTheGateOpens.
        msgs = [m for _sev, _code, m in self._raw({"executor": "e",
                                                   "initail": "executor"})]
        assert any("initial" in m and "fallback" in m for m in msgs)

    def test_the_shorthand_cannot_satisfy_a_free_tier(self):
        assert ("error", "tier_description_required") in self._diags(
            {"coder": "some-model"})

    def test_an_unusable_name_is_still_an_error(self):
        assert ("error", "unknown_tier") in self._diags({"Coder": dict(CODER)})

    def test_control_keys_are_not_flagged(self):
        assert self._diags({"executor": "e", "initial": "executor",
                            "fallback": "executor"}) == []


class TestDocumentationSurfaces:
    """Site 6 — derived, so they track the installed framework for free."""

    def test_explain_tiers_names_the_free_name_rule(self):
        from shared.scaffold import explain
        data, text = explain.tiers()
        assert data["canonical_tier_names"] == sorted(CANONICAL_TIER_NAMES)
        assert data["max_tiers"] == MAX_DECLARED_TIERS
        assert "description" in data["free_tier_names"]
        assert "YOUR OWN" in text

    def test_profile_field_constraints_mention_the_pattern(self):
        from shared.model_tiers import TIER_NAME_PATTERN
        from shared.scaffold.introspect import _profile_field_constraints
        assert TIER_NAME_PATTERN in _profile_field_constraints()["model_tiers"]


class TestEnvPathHasNoFreeNameSpelling:
    """Site 7 — stated explicitly rather than half-built."""

    def test_env_keys_cover_only_the_cognitive_canonical_tiers(self):
        from shared.model_tiers import ENV_TIER_MODEL_KEYS
        assert set(ENV_TIER_MODEL_KEYS) == {
            "planner", "dispatcher", "executor"}

    def test_env_built_configs_are_canonical_only(self):
        cfg = ModelTierConfig.from_env({
            "JAATO_TIER_EXECUTOR": "e", "JAATO_TIER_INITIAL": "executor",
            "JAATO_TIER_FALLBACK": "executor"})
        assert set(cfg.tiers) <= CANONICAL_TIER_NAMES
