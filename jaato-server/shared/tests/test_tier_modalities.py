"""The `modalities` key on a tier entry — declaring a ROLE, not a name.

``vision`` used to be a magic string: three places branched on it to decide
where an image could be viewed. A tier now says which non-text input
modalities it fills, and those places resolve the tier by role. The name
``vision`` keeps implying ``image`` so every profile written before the key
behaves unchanged.

Covers parsing/validation, the name-implied role on every construction path,
and the contract that the locally-defined modality tokens match the
provider-layer constants they are duplicated from.
"""

import pytest

from shared.model_tiers import (
    IMPLICIT_TIER_MODALITIES,
    ModelTierConfig,
    ModelTierConfigError,
    TierEntry,
    VALID_TIER_MODALITIES,
)


def _cfg(d):
    return ModelTierConfig.from_unified_dict(d)


class TestModalityTokenContract:
    """The tokens are duplicated from ``model_provider.base`` to keep this
    module's import cheap (~14ms vs ~196ms; it is on the profile-load path).
    Duplication is only safe while a test pins the two equal."""

    def test_tokens_match_the_provider_layer(self):
        from shared.plugins.model_provider import base as pb
        assert VALID_TIER_MODALITIES == {
            pb.MODALITY_IMAGE, pb.MODALITY_AUDIO,
            pb.MODALITY_VIDEO, pb.MODALITY_FILE,
        }

    def test_text_is_not_a_tier_modality(self):
        # Every model accepts text, so a tier declaring it asserts nothing.
        from shared.plugins.model_provider import base as pb
        assert pb.MODALITY_TEXT not in VALID_TIER_MODALITIES

    def test_gate_classifications_are_all_declarable(self):
        # Whatever _mime_to_modality can produce, a tier must be able to
        # claim — otherwise the gate could withhold content no tier can
        # ever be declared for.
        from shared.jaato_session import JaatoSession
        produced = {
            JaatoSession._mime_to_modality(m)
            for m in ("image/png", "audio/wav", "video/mp4", "application/pdf")
        }
        assert produced <= VALID_TIER_MODALITIES


class TestParsing:
    def test_declared_modalities_parsed(self):
        cfg = _cfg({"planner": {"model": "m", "modalities": ["image"]},
                    "initial": "planner", "fallback": "planner"})
        assert cfg.tiers["planner"].modalities == frozenset({"image"})

    def test_multiple_modalities(self):
        cfg = _cfg({"planner": {"model": "m", "modalities": ["image", "file"]},
                    "initial": "planner", "fallback": "planner"})
        assert cfg.tiers["planner"].modalities == frozenset({"image", "file"})

    def test_tokens_are_normalised(self):
        cfg = _cfg({"planner": {"model": "m", "modalities": ["  IMAGE "]},
                    "initial": "planner", "fallback": "planner"})
        assert cfg.tiers["planner"].modalities == frozenset({"image"})

    def test_undeclared_is_empty(self):
        cfg = _cfg({"planner": "m", "initial": "planner", "fallback": "planner"})
        assert cfg.tiers["planner"].modalities == frozenset()

    def test_unknown_modality_rejected(self):
        with pytest.raises(ModelTierConfigError, match="not a modality"):
            _cfg({"planner": {"model": "m", "modalities": ["smell"]},
                  "initial": "planner", "fallback": "planner"})

    def test_text_rejected_with_an_explanation(self):
        with pytest.raises(ModelTierConfigError, match="asserts nothing"):
            _cfg({"planner": {"model": "m", "modalities": ["text"]},
                  "initial": "planner", "fallback": "planner"})

    @pytest.mark.parametrize("bad", ["image", 7, {"image": True}])
    def test_non_list_rejected(self, bad):
        # A bare string is the likely typo and must not be parsed as a
        # sequence of characters.
        with pytest.raises(ModelTierConfigError, match="must be a list"):
            _cfg({"planner": {"model": "m", "modalities": bad},
                  "initial": "planner", "fallback": "planner"})

    @pytest.mark.parametrize("bad", [[""], ["  "], [None], [5]])
    def test_bad_entries_rejected(self, bad):
        with pytest.raises(ModelTierConfigError, match="non-empty strings"):
            _cfg({"planner": {"model": "m", "modalities": bad},
                  "initial": "planner", "fallback": "planner"})


class TestImplicitVisionRole:
    """``vision`` keeps meaning "the image tier" without declaring it."""

    def test_shorthand_vision_implies_image(self):
        cfg = _cfg({"executor": "e", "vision": "v",
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers["vision"].modalities == frozenset({"image"})

    def test_rich_vision_implies_image(self):
        cfg = _cfg({"executor": "e", "vision": {"model": "v"},
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers["vision"].modalities == frozenset({"image"})

    def test_direct_construction_also_implies_image(self):
        # __post_init__ applies it, so a config built in code (tests,
        # out-of-tree callers) can't silently lose the role and go quiet.
        cfg = ModelTierConfig(
            tiers={"executor": TierEntry("e"), "vision": TierEntry("v")},
            initial_tier="executor", tier_fallback="executor")
        assert cfg.tiers["vision"].modalities == frozenset({"image"})

    def test_vision_may_add_roles_but_keeps_image(self):
        cfg = _cfg({"executor": "e",
                    "vision": {"model": "v", "modalities": ["file"]},
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers["vision"].modalities == frozenset({"image", "file"})

    def test_only_vision_has_an_implicit_role(self):
        assert set(IMPLICIT_TIER_MODALITIES) == {"vision"}


class TestTiersForModality:
    def test_finds_by_role_not_name(self):
        cfg = _cfg({"executor": "e",
                    "planner": {"model": "p", "modalities": ["image"]},
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers_for_modality("image") == ("planner",)

    def test_empty_when_no_tier_declares_it(self):
        cfg = _cfg({"executor": "e", "vision": "v",
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers_for_modality("audio") == ()

    def test_canonical_order_when_several_match(self):
        cfg = _cfg({
            "executor": {"model": "e", "modalities": ["image"]},
            "planner": {"model": "p", "modalities": ["image"]},
            "initial": "executor", "fallback": "executor"})
        # planner precedes executor in TIER_ORDER, so the gate's "first
        # match" is deterministic rather than dict-insertion dependent.
        assert cfg.tiers_for_modality("image") == ("planner", "executor")


class TestDescribeTierWithRole:
    def test_role_supplies_prose_when_none_given(self):
        cfg = _cfg({"executor": {"model": "e", "modalities": ["image"]},
                    "initial": "executor", "fallback": "executor"})
        # 'executor' has framework default prose, which still wins.
        assert "mechanical tool calls" in cfg.describe_tier("executor")

    def test_explicit_description_still_wins(self):
        cfg = _cfg({"vision": {"model": "v", "description": "peek at things"},
                    "initial": "vision", "fallback": "vision"})
        assert cfg.describe_tier("vision") == "peek at things"


class TestBudgetOverlayPreservesRole:
    """A brownout rebinds a tier's model, never its role.

    Load-bearing: the gate and the startup check resolve the image tier BY
    ROLE, so an overlay that dropped it would leave a mid-run session with
    nowhere to send images — exactly when budget is tight.
    """

    def test_rung_declaring_modalities_is_rejected(self):
        from shared.budget_control import (
            BudgetControlConfig, BudgetControlConfigError,
        )
        with pytest.raises(BudgetControlConfigError, match="modalities"):
            BudgetControlConfig.from_dict({
                "limits": {"usd": 1.0},
                "degrade": [{"at": 50, "model_tiers": {
                    "vision": {"model": "cheap", "modalities": ["image"]}}}],
            })

    def test_overlay_carries_the_role_forward(self):
        from shared.budget_control import overlay_tier_table
        tiers = {"vision": TierEntry("pricey", modalities=frozenset({"image"}))}
        overlay_tier_table(tiers, {"vision": TierEntry("cheap")})
        assert tiers["vision"].model == "cheap"
        assert tiers["vision"].modalities == frozenset({"image"})


class TestDescribeTierAnnouncesTheRole:
    """A declared role the tier's NAME doesn't imply is announced in the
    tool description.

    Without it a `planner` tier declaring `modalities: [image]` read as pure
    cognitive prose, so the model had no reason to switch there for an image
    — it would only find out from the content gate after trying and failing.
    """

    def test_non_implied_role_is_appended(self):
        cfg = _cfg({"executor": "e",
                    "planner": {"model": "p", "modalities": ["image"]},
                    "initial": "executor", "fallback": "executor"})
        prose = cfg.describe_tier("planner")
        assert "deep thought" in prose            # framework base kept
        assert "view image content" in prose      # role announced

    def test_vision_does_not_double_announce(self):
        # image is vision's implicit role and its default prose covers it.
        cfg = _cfg({"executor": "e", "vision": "v",
                    "initial": "executor", "fallback": "executor"})
        assert cfg.describe_tier("vision").count("view image content") == 1

    def test_explicit_description_owns_the_whole_bullet(self):
        cfg = _cfg({"executor": "e",
                    "planner": {"model": "p", "modalities": ["image"],
                                "description": "my own words"},
                    "initial": "executor", "fallback": "executor"})
        assert cfg.describe_tier("planner") == "my own words"

    def test_role_reaches_the_tool_schema(self):
        from types import SimpleNamespace
        from shared.lifecycle_tools import LifecycleTools
        cfg = _cfg({"executor": "e",
                    "planner": {"model": "p", "modalities": ["file"]},
                    "initial": "executor", "fallback": "executor"})
        schema = LifecycleTools(SimpleNamespace(
            _tier_config=cfg, _completion_payload_schema=None,
            workspace_path=None, runtime=None))._enter_tier_schema()
        assert "view file content" in schema.description
