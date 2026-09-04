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
        assert cfg.tiers["planner"].inbound_modalities == frozenset({"image"})

    def test_multiple_modalities(self):
        cfg = _cfg({"planner": {"model": "m", "modalities": ["image", "file"]},
                    "initial": "planner", "fallback": "planner"})
        assert cfg.tiers["planner"].inbound_modalities == frozenset({"image", "file"})

    def test_tokens_are_normalised(self):
        cfg = _cfg({"planner": {"model": "m", "modalities": ["  IMAGE "]},
                    "initial": "planner", "fallback": "planner"})
        assert cfg.tiers["planner"].inbound_modalities == frozenset({"image"})

    def test_undeclared_is_empty(self):
        cfg = _cfg({"planner": "m", "initial": "planner", "fallback": "planner"})
        assert cfg.tiers["planner"].inbound_modalities == frozenset()

    def test_unknown_modality_rejected(self):
        with pytest.raises(ModelTierConfigError, match="not a modality"):
            _cfg({"planner": {"model": "m", "modalities": ["smell"]},
                  "initial": "planner", "fallback": "planner"})

    def test_text_rejected_with_an_explanation(self):
        with pytest.raises(ModelTierConfigError, match="asserts nothing"):
            _cfg({"planner": {"model": "m", "modalities": ["text"]},
                  "initial": "planner", "fallback": "planner"})

    @pytest.mark.parametrize("bad", ["image", 7, 3.5])
    def test_non_list_non_map_rejected(self, bad):
        # A bare string is the likely typo and must not be walked as a
        # sequence of characters.  A dict is NOT in this list — that is the
        # direction-map form (see TestDirections).
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
        assert cfg.tiers["vision"].inbound_modalities == frozenset({"image"})

    def test_rich_vision_implies_image(self):
        cfg = _cfg({"executor": "e", "vision": {"model": "v"},
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers["vision"].inbound_modalities == frozenset({"image"})

    def test_direct_construction_also_implies_image(self):
        # __post_init__ applies it, so a config built in code (tests,
        # out-of-tree callers) can't silently lose the role and go quiet.
        cfg = ModelTierConfig(
            tiers={"executor": TierEntry("e"), "vision": TierEntry("v")},
            initial_tier="executor", tier_fallback="executor")
        assert cfg.tiers["vision"].inbound_modalities == frozenset({"image"})

    def test_vision_may_add_roles_but_keeps_image(self):
        cfg = _cfg({"executor": "e",
                    "vision": {"model": "v", "modalities": ["file"]},
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers["vision"].inbound_modalities == frozenset({"image", "file"})

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
        tiers = {"vision": TierEntry("pricey", inbound_modalities=frozenset({"image"}))}
        overlay_tier_table(tiers, {"vision": TierEntry("cheap")})
        assert tiers["vision"].model == "cheap"
        assert tiers["vision"].inbound_modalities == frozenset({"image"})


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
        assert "accept image input" in prose      # role announced

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
        assert "accept file input" in schema.description


class TestDirections:
    """The direction-qualified map form: `{image: inbound|outbound|bidirectional}`."""

    def _entry(self, mods):
        cfg = _cfg({"planner": {"model": "m", "modalities": mods},
                    "initial": "planner", "fallback": "planner"})
        return cfg.tiers["planner"]

    def test_inbound_map(self):
        e = self._entry({"image": "inbound"})
        assert e.inbound_modalities == frozenset({"image"})
        assert e.outbound_modalities == frozenset()

    def test_outbound_map(self):
        e = self._entry({"audio": "outbound"})
        assert e.inbound_modalities == frozenset()
        assert e.outbound_modalities == frozenset({"audio"})

    def test_bidirectional_lands_in_both_sets(self):
        # Which is exactly why the stored form is two sets and not the map.
        e = self._entry({"audio": "bidirectional"})
        assert e.inbound_modalities == frozenset({"audio"})
        assert e.outbound_modalities == frozenset({"audio"})

    def test_mixed_directions(self):
        e = self._entry({"image": "inbound", "audio": "outbound"})
        assert e.inbound_modalities == frozenset({"image"})
        assert e.outbound_modalities == frozenset({"audio"})

    def test_direction_is_normalised(self):
        assert self._entry({"audio": "  OUTBOUND "}).outbound_modalities == \
            frozenset({"audio"})

    def test_list_sugar_is_inbound(self):
        e = self._entry(["image"])
        assert e.inbound_modalities == frozenset({"image"})
        assert e.outbound_modalities == frozenset()

    @pytest.mark.parametrize("bad", ["both", "duplex", "inout", "io"])
    def test_near_miss_spellings_suggest_bidirectional(self, bad):
        # These are the spellings an author actually reaches for; the error
        # has to name the right one rather than just listing the enum.
        with pytest.raises(ModelTierConfigError, match="bidirectional"):
            self._entry({"audio": bad})

    @pytest.mark.parametrize("bad", ["", "   ", 7, None, True])
    def test_malformed_direction_rejected(self, bad):
        with pytest.raises(ModelTierConfigError, match="direction"):
            self._entry({"audio": bad})

    def test_modalities_for_rejects_bidirectional_as_a_query(self):
        # bidirectional is a DECLARATION spelling; querying for it would be
        # ambiguous since the role lands in both sets.
        e = self._entry({"audio": "bidirectional"})
        assert e.modalities_for("inbound") == frozenset({"audio"})
        assert e.modalities_for("outbound") == frozenset({"audio"})
        with pytest.raises(ValueError, match="direction must be"):
            e.modalities_for("bidirectional")

    def test_vision_keeps_inbound_image_alongside_a_declared_outbound(self):
        cfg = _cfg({"executor": "e",
                    "vision": {"model": "v", "modalities": {"audio": "outbound"}},
                    "initial": "executor", "fallback": "executor"})
        v = cfg.tiers["vision"]
        assert v.inbound_modalities == frozenset({"image"})
        assert v.outbound_modalities == frozenset({"audio"})


class TestDirectionalLookup:
    def test_tiers_for_modality_defaults_to_inbound(self):
        cfg = _cfg({"executor": "e",
                    "planner": {"model": "p", "modalities": {"audio": "outbound"}},
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers_for_modality("audio") == ()               # inbound
        assert cfg.tiers_for_modality("audio", "outbound") == ("planner",)

    def test_bidirectional_tier_appears_in_both(self):
        cfg = _cfg({"executor": "e",
                    "planner": {"model": "p",
                                "modalities": {"audio": "bidirectional"}},
                    "initial": "executor", "fallback": "executor"})
        assert cfg.tiers_for_modality("audio", "inbound") == ("planner",)
        assert cfg.tiers_for_modality("audio", "outbound") == ("planner",)

    def test_bad_direction_raises(self):
        cfg = _cfg({"executor": "e", "initial": "executor", "fallback": "executor"})
        with pytest.raises(ValueError, match="direction must be"):
            cfg.tiers_for_modality("image", "sideways")


class TestOutboundDescribedButInert:
    def test_outbound_role_announced_separately_from_inbound(self):
        cfg = _cfg({"executor": {"model": "e",
                                 "modalities": {"audio": "bidirectional"}},
                    "initial": "executor", "fallback": "executor"})
        prose = cfg.describe_tier("executor")
        assert "accept audio input" in prose        # inbound clause
        assert "produce audio output" in prose      # outbound clause

    def test_startup_check_skips_outbound_when_provider_cannot_answer(self):
        # No provider implements supports_output_modality yet; the role must
        # be left unverified rather than failing falsely.
        from types import SimpleNamespace
        from shared.jaato_session import JaatoSession
        cfg = _cfg({"executor": {"model": "m", "modalities": {"audio": "outbound"}},
                    "initial": "executor", "fallback": "executor"})
        s = JaatoSession.__new__(JaatoSession)
        s._tier_config = cfg
        s._active_provider_name = "p"
        s._provider = SimpleNamespace(
            name="p", supports_modality=lambda kind, model=None: False)
        s._validate_modality_tier_capabilities()  # no raise

    def test_startup_check_enforces_outbound_when_provider_can_answer(self):
        from types import SimpleNamespace
        from shared.jaato_session import JaatoSession
        cfg = _cfg({"executor": {"model": "m", "modalities": {"audio": "outbound"}},
                    "initial": "executor", "fallback": "executor"})
        s = JaatoSession.__new__(JaatoSession)
        s._tier_config = cfg
        s._active_provider_name = "p"
        s._provider = SimpleNamespace(
            name="p",
            supports_modality=lambda kind, model=None: True,
            supports_output_modality=lambda kind, model=None: False,
        )
        with pytest.raises(ModelTierConfigError, match="outbound"):
            s._validate_modality_tier_capabilities()
