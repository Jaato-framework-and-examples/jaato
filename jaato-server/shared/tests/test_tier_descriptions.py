"""Per-tier ``description`` — profile prose reaching the model's tool schema.

Two behaviours land together here, because they are the same change seen
from either side:

* the ``enter_tier`` tool advertises ONLY the tiers a profile declares
  (previously all four framework names, so a model could ask for a tier
  that silently routed to ``tier_fallback``); and
* each advertised tier's bullet is that tier's ``description`` when the
  profile set one, else the framework's own wording for the name.

The tool block sits in the prompt-cache prefix, so the ordering and
build-once properties are asserted too — a schema that reordered between
processes would invalidate the cache for no semantic reason.
"""

from types import SimpleNamespace

import pytest

from shared.lifecycle_tools import LifecycleTools
from shared.model_tiers import (
    DEFAULT_TIER_DESCRIPTIONS,
    ModelTierConfig,
    ModelTierConfigError,
    TierEntry,
)


def _cfg(tier_dict):
    return ModelTierConfig.from_unified_dict(tier_dict)


def _schema(tier_dict):
    """Build the enter_tier schema for a session with these tiers."""
    session = SimpleNamespace(
        _tier_config=_cfg(tier_dict) if tier_dict is not None else None,
        _completion_payload_schema=None,
        workspace_path=None,
        runtime=None,
    )
    return LifecycleTools(session)._enter_tier_schema()


# --------------------------------------------------------------- parsing


class TestTierEntryDescription:
    def test_rich_form_carries_description(self):
        cfg = _cfg({
            "executor": {"model": "m", "description": "grind through edits"},
            "initial": "executor", "fallback": "executor",
        })
        assert cfg.tiers["executor"].description == "grind through edits"

    def test_description_is_stripped(self):
        cfg = _cfg({
            "executor": {"model": "m", "description": "  padded  "},
            "initial": "executor", "fallback": "executor",
        })
        assert cfg.tiers["executor"].description == "padded"

    def test_shorthand_has_no_description(self):
        cfg = _cfg({"executor": "m", "initial": "executor",
                    "fallback": "executor"})
        assert cfg.tiers["executor"].description is None

    @pytest.mark.parametrize("bad", ["", "   ", 7, [], {}])
    def test_non_string_or_empty_description_rejected(self, bad):
        with pytest.raises(ModelTierConfigError, match="description"):
            _cfg({
                "executor": {"model": "m", "description": bad},
                "initial": "executor", "fallback": "executor",
            })


class TestDescribeTier:
    def test_profile_description_wins(self):
        cfg = _cfg({
            "planner": {"model": "m", "description": "our own words"},
            "initial": "planner", "fallback": "planner",
        })
        assert cfg.describe_tier("planner") == "our own words"

    def test_falls_back_to_framework_default(self):
        cfg = _cfg({"planner": "m", "initial": "planner",
                    "fallback": "planner"})
        assert cfg.describe_tier("planner") == DEFAULT_TIER_DESCRIPTIONS[
            "planner"]

    def test_unknown_undescribed_tier_names_the_model(self):
        # Not reachable today (names are a closed set) but the free-form
        # tier work depends on this not raising.
        cfg = ModelTierConfig(
            tiers={"executor": TierEntry(model="m")},
            initial_tier="executor", tier_fallback="executor",
        )
        object.__setattr__(cfg, "tiers", {"weird": TierEntry(model="mystery")})
        assert "mystery" in cfg.describe_tier("weird")


class TestOrderedTierNames:
    def test_canonical_not_alphabetical(self):
        # sorted() would put dispatcher first; TIER_ORDER must win.
        cfg = _cfg({
            "executor": "e", "planner": "p", "dispatcher": "d",
            "initial": "dispatcher", "fallback": "dispatcher",
        })
        assert cfg.ordered_tier_names() == (
            "planner", "dispatcher", "executor")

    def test_only_declared_tiers(self):
        cfg = _cfg({"planner": "p", "executor": "e",
                    "initial": "planner", "fallback": "planner"})
        assert cfg.ordered_tier_names() == ("planner", "executor")


# ------------------------------------------------------------ tool schema


class TestEnterTierSchema:
    def test_enum_lists_only_declared_tiers(self):
        s = _schema({"planner": "p", "executor": "e",
                     "initial": "planner", "fallback": "planner"})
        assert s.parameters["properties"]["name"]["enum"] == [
            "planner", "executor"]
        # The regression this closes: dispatcher/vision were advertised
        # unconditionally, and asking for one silently hit the fallback.
        assert "dispatcher" not in s.description
        assert "`vision`" not in s.description

    def test_bullet_uses_profile_description(self):
        s = _schema({
            "executor": {"model": "m",
                         "description": "apply the migration plan verbatim"},
            "initial": "executor", "fallback": "executor",
        })
        assert "* `executor` — apply the migration plan verbatim" in s.description
        # ...and the framework's own executor prose is gone.
        assert "Cheapest" not in s.description

    def test_bullet_falls_back_to_framework_prose(self):
        s = _schema({"executor": "m", "initial": "executor",
                     "fallback": "executor"})
        assert DEFAULT_TIER_DESCRIPTIONS["executor"] in s.description

    def test_names_the_starting_tier(self):
        s = _schema({"planner": "p", "executor": "e",
                     "initial": "executor", "fallback": "executor"})
        assert "starts in `executor`" in s.description

    def test_bullet_order_is_canonical(self):
        s = _schema({
            "executor": "e", "planner": "p",
            "initial": "planner", "fallback": "planner",
        })
        assert s.description.index("`planner`") < s.description.index(
            "`executor`")

    def test_schema_is_deterministic(self):
        # Cache-prefix stability: two builds of the same config must be
        # byte-identical.
        tiers = {"executor": "e", "vision": {"model": "v",
                                             "description": "look"},
                 "initial": "executor", "fallback": "executor"}
        a, b = _schema(tiers), _schema(tiers)
        assert a.description == b.description
        assert a.parameters == b.parameters

    def test_no_tier_config_advertises_all_known_tiers(self):
        # Unreachable via get_tool_schemas (gated on _tier_config) but
        # tests call the builder directly.
        s = _schema(None)
        assert s.parameters["properties"]["name"]["enum"] == [
            "planner", "dispatcher", "executor", "vision"]


# --------------------------------------------------------- budget overlay


class TestBudgetOverlayAndDescriptions:
    """A brownout rebinds a tier's MODEL, never its role.

    So a degrade rung may not declare a description (the tool schema was
    already built and cached — accepting one would be a lie), and applying
    a rung must not drop the description the profile did declare.
    """

    def test_rung_declaring_a_description_is_rejected(self):
        from shared.budget_control import (
            BudgetControlConfig, BudgetControlConfigError,
        )
        with pytest.raises(BudgetControlConfigError, match="description"):
            BudgetControlConfig.from_dict({
                "limits": {"usd": 1.0},
                "degrade": [{
                    "at": 50,
                    "model_tiers": {
                        "executor": {"model": "cheap", "description": "nope"},
                    },
                }],
            })

    def test_overlay_preserves_the_base_description(self):
        from shared.budget_control import overlay_tier_table

        tiers = {"executor": TierEntry(model="pricey",
                                       description="apply the plan")}
        changes = overlay_tier_table(
            tiers, {"executor": TierEntry(model="cheap")})

        assert changes == {"executor": "pricey -> cheap"}
        assert tiers["executor"].model == "cheap"
        assert tiers["executor"].description == "apply the plan"

    def test_overlay_of_an_undeclared_tier_has_no_description(self):
        from shared.budget_control import overlay_tier_table

        tiers = {}
        overlay_tier_table(tiers, {"executor": TierEntry(model="cheap")})
        assert tiers["executor"].description is None
